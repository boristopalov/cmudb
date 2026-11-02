use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Schema, Value};
use crate::processor::executor::{Batch, Executor, ExecutorError, Row, collect_rows};
use crate::processor::executors::{SpillHeap, fill_batch, not_opened, take_batch};
use crate::processor::plan::{
    ExternalMergeSortPlanNode, NullOrdering, SortDirection, SortExpr, SortPlanNode,
};
use crate::table_heap::{RecordId, TableHeapIterator, Tuple};
use std::sync::Arc;
use std::vec::IntoIter as VecIter;

pub struct SortExecutor<'a> {
    plan: &'a SortPlanNode,
    child: Box<dyn Executor + 'a>,
    batch_size: usize,
    sorted: VecIter<Row>,
}

impl<'a> SortExecutor<'a> {
    pub fn new(plan: &'a SortPlanNode, child: Box<dyn Executor + 'a>, batch_size: usize) -> Self {
        Self {
            plan,
            child,
            batch_size,
            sorted: Vec::new().into_iter(),
        }
    }
}

impl<'a> Executor for SortExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let buf = collect_rows(self.child.as_mut())?;
        let sorted = sorted_by_exprs(
            buf,
            |row| &row.0,
            &self.plan.order_by_exprs,
            &self.plan.schema,
        )?;
        self.sorted = sorted.into_iter();
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        Ok(take_batch(&mut self.sorted, self.batch_size))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
    }
}

/// Sorts its input in memory while the input's tuple bytes stay within its memory budget,
/// and otherwise spills sorted runs to scratch heaps and merges them.
pub struct ExternalMergeSortExecutor<'a> {
    plan: &'a ExternalMergeSortPlanNode,
    child: Box<dyn Executor + 'a>,
    bpm: Arc<BufferPoolManager>,
    batch_size: usize,
    memory_budget_bytes: usize,
    output: Option<SortedOutput>,
}

/// Where an [`ExternalMergeSortExecutor`] reads its sorted rows from.
enum SortedOutput {
    InMemory(VecIter<Tuple>),
    Spilled {
        iter: TableHeapIterator,
        run: SpillHeap,
    },
}

impl<'a> ExternalMergeSortExecutor<'a> {
    pub fn new(
        plan: &'a ExternalMergeSortPlanNode,
        child: Box<dyn Executor + 'a>,
        bpm: Arc<BufferPoolManager>,
        batch_size: usize,
        memory_budget_bytes: usize,
    ) -> Self {
        Self {
            plan,
            child,
            bpm,
            batch_size,
            memory_budget_bytes,
            output: None,
        }
    }
}

/// One evaluated `ORDER BY` key of a row, ordered in its key's direction with NULL
/// placed where its key says.
struct SortValue {
    value: Option<Value>,
    direction: SortDirection,
    nulls: NullOrdering,
}

impl Ord for SortValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let null_vs_value = match self.nulls {
            NullOrdering::First => Ordering::Less,
            NullOrdering::Last => Ordering::Greater,
        };
        match (&self.value, &other.value) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => null_vs_value,
            (Some(_), None) => null_vs_value.reverse(),
            (Some(a), Some(b)) => match self.direction {
                SortDirection::Ascending => a.cmp(b),
                SortDirection::Descending => a.cmp(b).reverse(),
            },
        }
    }
}

impl PartialEq for SortValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for SortValue {}

impl PartialOrd for SortValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A row's `ORDER BY` keys, compared key by key.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct SortKey(Vec<SortValue>);

fn eval_sort_key(
    order_by: &[SortExpr],
    tuple: &Tuple,
    schema: &Schema,
) -> Result<SortKey, ExecutorError> {
    order_by
        .iter()
        .map(|key| {
            Ok(SortValue {
                value: key.expr.evaluate(tuple, schema)?,
                direction: key.direction,
                nulls: key.nulls,
            })
        })
        .collect::<Result<_, ExecutorError>>()
        .map(SortKey)
}

/// Rows ordered by `order_by` evaluated on each row's tuple, with each key's NULLs placed
/// where that key says; equal rows keep their input order.
pub(super) fn sorted_by_exprs<T>(
    rows: Vec<T>,
    tuple_of: impl Fn(&T) -> &Tuple,
    order_by: &[SortExpr],
    schema: &Schema,
) -> Result<Vec<T>, ExecutorError> {
    let mut keyed = rows
        .into_iter()
        .map(|row| Ok((eval_sort_key(order_by, tuple_of(&row), schema)?, row)))
        .collect::<Result<Vec<_>, ExecutorError>>()?;
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, row)| row).collect())
}

impl<'a> ExternalMergeSortExecutor<'a> {
    fn sort_in_memory(&self, rows: Vec<Tuple>) -> Result<Vec<Tuple>, ExecutorError> {
        sorted_by_exprs(rows, |t| t, &self.plan.order_by_exprs, &self.plan.schema)
    }

    fn spill_sorted_run(&self, rows: &mut Vec<Tuple>) -> Result<SpillHeap, ExecutorError> {
        let sorted = self.sort_in_memory(std::mem::take(rows))?;
        let run = SpillHeap::new(self.bpm.clone())?;
        for t in sorted {
            run.insert(&t)?;
        }
        Ok(run)
    }

    fn merge_runs(&self, a: &SpillHeap, b: &SpillHeap) -> Result<SpillHeap, ExecutorError> {
        let (order_by, schema) = (&self.plan.order_by_exprs, &self.plan.schema);
        let out = SpillHeap::new(self.bpm.clone())?;
        let mut a_rows = a.iter();
        let mut b_rows = b.iter();
        let next_keyed =
            |rows: &mut TableHeapIterator| -> Result<Option<(SortKey, Tuple)>, ExecutorError> {
                match rows.next().transpose()? {
                    Some((t, _)) => Ok(Some((eval_sort_key(order_by, &t, schema)?, t))),
                    None => Ok(None),
                }
            };
        let mut a_head = next_keyed(&mut a_rows)?;
        let mut b_head = next_keyed(&mut b_rows)?;
        loop {
            let take_a = match (&a_head, &b_head) {
                (Some((ka, _)), Some((kb, _))) => ka <= kb,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => return Ok(out),
            };
            let (head, rows) = if take_a {
                (&mut a_head, &mut a_rows)
            } else {
                (&mut b_head, &mut b_rows)
            };
            if let Some((_, t)) = head.take() {
                out.insert(&t)?;
            }
            *head = next_keyed(rows)?;
        }
    }

    /// Merges `runs` pairwise into one run, freeing each run once it has been merged.
    fn merge_all(&self, mut runs: Vec<SpillHeap>) -> Result<SpillHeap, ExecutorError> {
        while runs.len() > 1 {
            let mut next_runs = Vec::with_capacity(runs.len().div_ceil(2));
            let mut pending = runs.into_iter();
            while let Some(a) = pending.next() {
                match pending.next() {
                    Some(b) => {
                        next_runs.push(self.merge_runs(&a, &b)?);
                        a.free()?;
                        b.free()?;
                    }
                    None => next_runs.push(a),
                }
            }
            runs = next_runs;
        }
        runs.pop().ok_or(ExecutorError::NoSortRuns)
    }

    /// Reads the child to the end, spilling a sorted run each time the buffered tuple bytes
    /// pass the memory budget. Returns the runs and the rows still buffered.
    fn read_input(&mut self) -> Result<(Vec<SpillHeap>, Vec<Tuple>), ExecutorError> {
        let mut runs = Vec::new();
        let mut buffered = Vec::new();
        let mut buffered_bytes: usize = 0;
        while let Some(batch) = self.child.next_batch()? {
            for (tup, _) in batch {
                buffered_bytes += tup.data.len();
                buffered.push(tup);
                if buffered_bytes > self.memory_budget_bytes {
                    runs.push(self.spill_sorted_run(&mut buffered)?);
                    buffered_bytes = 0;
                }
            }
        }
        Ok((runs, buffered))
    }
}

impl<'a> Executor for ExternalMergeSortExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.output = None;
        self.child.open()?;
        let (mut runs, buffered) = self.read_input()?;
        let output = if runs.is_empty() {
            SortedOutput::InMemory(self.sort_in_memory(buffered)?.into_iter())
        } else {
            let mut rest = buffered;
            if !rest.is_empty() {
                runs.push(self.spill_sorted_run(&mut rest)?);
            }
            let run = self.merge_all(runs)?;
            SortedOutput::Spilled {
                iter: run.iter(),
                run,
            }
        };
        self.output = Some(output);
        Ok(())
    }

    /// Yields the sorted rows with [`RecordId::RESERVED`], since a sort's rows map back to no
    /// single table slot.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let output = self
            .output
            .as_mut()
            .ok_or_else(|| not_opened("ExternalMergeSortExecutor"))?;
        match output {
            SortedOutput::InMemory(rows) => Ok(take_batch(
                &mut rows.by_ref().map(|t| (t, RecordId::RESERVED)),
                self.batch_size,
            )),
            SortedOutput::Spilled { iter, .. } => {
                let mut rows = iter
                    .by_ref()
                    .map(|item| item.map(|(tup, _)| (tup, RecordId::RESERVED)));
                fill_batch(&mut rows, self.batch_size)
            }
        }
    }

    /// Closes the child and deletes the pages of the spilled run, if the sort spilled.
    fn close(&mut self) -> Result<(), ExecutorError> {
        let child = self.child.close();
        let freed = match self.output.take() {
            Some(SortedOutput::Spilled { run, .. }) => run.free().map_err(ExecutorError::from),
            Some(SortedOutput::InMemory(_)) | None => Ok(()),
        };
        child.and(freed)
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::executor::{DEFAULT_BATCH_SIZE, DEFAULT_SORT_MEMORY_BUDGET_BYTES};
    use crate::processor::executors::test_support::*;
    use crate::processor::plan::{Expression, Plan};

    #[test]
    fn external_sort_merges_spilled_runs_in_order() {
        let (bpm, _dir) = crate::testing::bpm(64);
        let schema = int_schema(&["k", "v"]);
        let plan = ExternalMergeSortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: vec![SortExpr::ascending(col(0))],
        };
        let exec = ExternalMergeSortExecutor::new(
            &plan,
            values_child(&plan.child),
            bpm,
            DEFAULT_BATCH_SIZE,
            DEFAULT_SORT_MEMORY_BUDGET_BYTES,
        );

        let mut a = int_rows(&schema, &[&[5, 0], &[1, 0], &[3, 0]]);
        let mut b = int_rows(&schema, &[&[4, 1], &[2, 1], &[1, 1], &[9, 1]]);
        let run_a = exec.spill_sorted_run(&mut a).unwrap();
        let run_b = exec.spill_sorted_run(&mut b).unwrap();
        assert!(a.is_empty() && b.is_empty(), "spilling drains the buffer");

        let merged = exec.merge_runs(&run_a, &run_b).unwrap();
        let rows = decode_ints(&schema, merged.iter().map(|r| r.unwrap().0));
        assert_eq!(
            rows,
            vec![
                vec![1, 0],
                vec![1, 1],
                vec![2, 1],
                vec![3, 0],
                vec![4, 1],
                vec![5, 0],
                vec![9, 1],
            ],
            "ties keep the left run's row first"
        );

        let empty = exec.spill_sorted_run(&mut Vec::new()).unwrap();
        let merged_with_empty = exec.merge_runs(&empty, &run_a).unwrap();
        assert_eq!(
            decode_ints(&schema, merged_with_empty.iter().map(|r| r.unwrap().0)),
            vec![vec![1, 0], vec![3, 0], vec![5, 0]]
        );
    }

    #[test]
    fn descending_sort_keeps_tied_rows_in_input_order() {
        let schema = int_schema(&["k", "v"]);
        let rows = values_plan(&schema, &[&[1, 0], &[2, 0], &[1, 1], &[2, 1], &[1, 2]]);
        let plan = SortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: vec![SortExpr::descending(col(0))],
        };
        let mut sort = SortExecutor::new(&plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        assert_eq!(
            decode_ints(&schema, drain(&mut sort)),
            vec![vec![2, 0], vec![2, 1], vec![1, 0], vec![1, 1], vec![1, 2]],
            "rows with equal keys keep their input order"
        );
    }

    #[test]
    fn external_sort_merges_descending_runs_in_descending_order() {
        let (bpm, _dir) = crate::testing::bpm(64);
        let schema = int_schema(&["k", "v"]);
        let plan = ExternalMergeSortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: vec![SortExpr::descending(col(0))],
        };
        let exec = ExternalMergeSortExecutor::new(
            &plan,
            values_child(&plan.child),
            bpm,
            DEFAULT_BATCH_SIZE,
            DEFAULT_SORT_MEMORY_BUDGET_BYTES,
        );

        let mut a = int_rows(&schema, &[&[5, 0], &[1, 0], &[3, 0]]);
        let mut b = int_rows(&schema, &[&[4, 1], &[1, 1], &[9, 1]]);
        let run_a = exec.spill_sorted_run(&mut a).unwrap();
        let run_b = exec.spill_sorted_run(&mut b).unwrap();
        let merged = exec.merge_runs(&run_a, &run_b).unwrap();
        assert_eq!(
            decode_ints(&schema, merged.iter().map(|r| r.unwrap().0)),
            vec![
                vec![9, 1],
                vec![5, 0],
                vec![4, 1],
                vec![3, 0],
                vec![1, 0],
                vec![1, 1],
            ],
            "keys descend and ties keep the left run's row first"
        );
    }

    #[test]
    fn external_sort_merges_runs_with_nulls_where_each_key_places_them() {
        let (bpm, _dir) = crate::testing::bpm(64);
        let schema = int_schema(&["k"]);
        for (direction, nulls, expected) in [
            (
                SortDirection::Ascending,
                NullOrdering::First,
                [None, None, Some(1), Some(2), Some(3), Some(5)],
            ),
            (
                SortDirection::Ascending,
                NullOrdering::Last,
                [Some(1), Some(2), Some(3), Some(5), None, None],
            ),
            (
                SortDirection::Descending,
                NullOrdering::First,
                [None, None, Some(5), Some(3), Some(2), Some(1)],
            ),
            (
                SortDirection::Descending,
                NullOrdering::Last,
                [Some(5), Some(3), Some(2), Some(1), None, None],
            ),
        ] {
            let plan = ExternalMergeSortPlanNode {
                schema: schema.clone(),
                child: Box::new(values_plan(&schema, &[])),
                order_by_exprs: vec![SortExpr::new(col(0), direction).with_nulls(nulls)],
            };
            let exec = ExternalMergeSortExecutor::new(
                &plan,
                values_child(&plan.child),
                bpm.clone(),
                DEFAULT_BATCH_SIZE,
                DEFAULT_SORT_MEMORY_BUDGET_BYTES,
            );
            let mut a = nullable_int_rows(&schema, &[&[Some(2)], &[None], &[Some(5)]]);
            let mut b = nullable_int_rows(&schema, &[&[None], &[Some(1)], &[Some(3)]]);
            let run_a = exec.spill_sorted_run(&mut a).unwrap();
            let run_b = exec.spill_sorted_run(&mut b).unwrap();
            let merged = exec.merge_runs(&run_a, &run_b).unwrap();
            assert_eq!(
                decode_nullable_ints(&schema, merged.iter().map(|r| r.unwrap().0)),
                expected,
                "{direction:?} NULLS {nulls:?}"
            );
        }
    }

    fn external_sort_of(plan: &Plan) -> &ExternalMergeSortPlanNode {
        match plan {
            Plan::ExternalMergeSort(p) => p,
            _ => panic!("expected an external merge sort plan"),
        }
    }

    fn descending_external_sort(rows: &[&[i32]]) -> Plan {
        let schema = int_schema(&["k"]);
        Plan::ExternalMergeSort(ExternalMergeSortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, rows)),
            order_by_exprs: vec![SortExpr::descending(col(0))],
        })
    }

    const SPILL_EVERY_ROW: usize = 0;

    #[test]
    fn external_sort_within_its_budget_sorts_without_scratch_pages() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let plan = descending_external_sort(&[&[2], &[7], &[1]]);
        let node = external_sort_of(&plan);
        let mut exec = ExternalMergeSortExecutor::new(
            node,
            values_child(&node.child),
            bpm.clone(),
            DEFAULT_BATCH_SIZE,
            DEFAULT_SORT_MEMORY_BUDGET_BYTES,
        );
        assert_eq!(
            decode_ints(&node.schema, drain(&mut exec)),
            vec![vec![7], vec![2], vec![1]]
        );
        exec.close().unwrap();
        assert_eq!(bpm.allocated_page_count(), 0, "the sort spilled");
    }

    #[test]
    fn external_sort_past_its_budget_spills_and_close_frees_every_scratch_page() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let input: Vec<[i32; 1]> = (0..40).map(|i| [(i * 17) % 40]).collect();
        let rows: Vec<&[i32]> = input.iter().map(|r| &r[..]).collect();
        let plan = descending_external_sort(&rows);
        let node = external_sort_of(&plan);
        let mut exec = ExternalMergeSortExecutor::new(
            node,
            values_child(&node.child),
            bpm.clone(),
            3,
            SPILL_EVERY_ROW,
        );

        let sorted = decode_ints(&node.schema, drain(&mut exec));

        assert_eq!(sorted, (0..40).rev().map(|i| vec![i]).collect::<Vec<_>>());
        assert!(
            bpm.allocated_page_count() > 40,
            "expected a run per row plus merge output, got {} pages",
            bpm.allocated_page_count()
        );
        assert!(
            bpm.live_page_count() > 0,
            "the merged run is read until close"
        );
        exec.close().unwrap();
        assert_eq!(bpm.live_page_count(), 0, "scratch pages outlived close");
    }

    #[test]
    fn dropping_an_external_sort_without_closing_it_frees_its_scratch_pages() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let plan = descending_external_sort(&[&[3], &[1], &[2]]);
        let node = external_sort_of(&plan);
        let mut exec = ExternalMergeSortExecutor::new(
            node,
            values_child(&node.child),
            bpm.clone(),
            DEFAULT_BATCH_SIZE,
            SPILL_EVERY_ROW,
        );
        exec.open().unwrap();
        assert!(bpm.live_page_count() > 0, "the sort did not spill");
        drop(exec);
        assert_eq!(bpm.live_page_count(), 0);
    }

    #[test]
    fn a_failed_external_sort_frees_the_runs_it_spilled() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let schema = int_schema(&["a"]);
        let plan = ExternalMergeSortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[&[2], &[1], &[0]])),
            order_by_exprs: vec![SortExpr::ascending(Expression::Binary {
                left: Box::new(Expression::Constant(Some(Value::INT(6)))),
                right: Box::new(col(0)),
                op: crate::processor::plan::Op::Div,
            })],
        };
        let mut exec = ExternalMergeSortExecutor::new(
            &plan,
            values_child(&plan.child),
            bpm.clone(),
            DEFAULT_BATCH_SIZE,
            SPILL_EVERY_ROW,
        );
        let err = exec.open().unwrap_err();
        assert!(
            matches!(err, ExecutorError::DivisionByZero { dividend: 6 }),
            "{err}"
        );
        assert!(
            bpm.allocated_page_count() >= 2,
            "the rows before the zero were not spilled"
        );
        assert_eq!(
            bpm.live_page_count(),
            0,
            "runs spilled before the error leaked"
        );
    }

    #[test]
    fn sort_key_evaluation_errors_are_reported_not_sorted_as_null() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[2], &[1]]);
        let plan = SortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: vec![SortExpr::ascending(divide_by_zero(0))],
        };
        let mut sort = SortExecutor::new(&plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        let err = sort.open().unwrap_err();
        assert!(
            matches!(err, ExecutorError::DivisionByZero { .. }),
            "{err:?}"
        );
    }
}
