mod aggregate;
mod ddl;
mod dml;
mod join;
mod pipeline;
mod scan;
mod sort;
#[cfg(test)]
pub(crate) mod test_support;

pub use aggregate::{AggregationExecutor, WindowFunctionExecutor};
pub use ddl::{DdlExecutor, DdlOp};
pub use dml::{DeleteExecutor, InsertExecutor, UpdateExecutor};
pub use join::{HashJoinExecutor, NestedIndexJoinExecutor, NestedLoopJoinExecutor};
pub use pipeline::{FilterExecutor, LimitExecutor, ProjectionExecutor};
pub use scan::{IndexScanExecutor, SeqScanExecutor, ValuesExecutor};
pub use sort::{ExternalMergeSortExecutor, SortExecutor};

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Schema, Value};
use crate::processor::executor::{Batch, ExecutorError, Row};
use crate::processor::plan::Expression;
use crate::table_heap::{FreeError, RecordId, TableHeap, TableHeapError, TableHeapIterator, Tuple};
use log::warn;
use std::sync::Arc;
/// Pulls up to `batch_size` rows from a fallible row iterator.
fn fill_batch<I, E>(iter: &mut I, batch_size: usize) -> Result<Option<Batch>, ExecutorError>
where
    I: Iterator<Item = Result<Row, E>>,
    ExecutorError: From<E>,
{
    let mut rows = Vec::with_capacity(batch_size);
    while rows.len() < batch_size {
        match iter.next() {
            Some(row) => rows.push(row?),
            None => break,
        }
    }
    Ok(Batch::from_rows(rows))
}

/// Pulls up to `batch_size` rows from an iterator over already materialized rows.
fn take_batch<I>(iter: &mut I, batch_size: usize) -> Option<Batch>
where
    I: Iterator<Item = Row>,
{
    Batch::from_rows(iter.by_ref().take(batch_size).collect())
}

/// A batch holding only `tuple`, which maps back to no table slot.
fn single_row(tuple: Tuple) -> Option<Batch> {
    Batch::from_rows(vec![(tuple, RecordId::RESERVED)])
}

fn not_opened(executor: &'static str) -> ExecutorError {
    ExecutorError::NotOpened { executor }
}

fn is_true(predicate_result: Option<Value>) -> bool {
    matches!(predicate_result, Some(Value::BOOLEAN(true)))
}

fn row_values(schema: &Schema, tuple: &Tuple) -> Result<Vec<Option<Value>>, ExecutorError> {
    (0..schema.cols.len())
        .map(|i| schema.get_value(tuple, i).map_err(ExecutorError::from))
        .collect()
}

/// The left row's values followed by the right row's, or by NULLs when there is no right row.
fn joined_row(
    left_schema: &Schema,
    left: &Tuple,
    right_schema: &Schema,
    right: Option<&Tuple>,
) -> Result<Vec<Option<Value>>, ExecutorError> {
    let mut row = row_values(left_schema, left)?;
    match right {
        Some(right) => row.extend(row_values(right_schema, right)?),
        None => row.resize(left_schema.cols.len() + right_schema.cols.len(), None),
    }
    Ok(row)
}

fn eval_key(
    exprs: &[Expression],
    tuple: &Tuple,
    schema: &Schema,
) -> Result<ExprKey, ExecutorError> {
    exprs
        .iter()
        .map(|e| e.evaluate(tuple, schema))
        .collect::<Result<_, _>>()
        .map(ExprKey)
}

#[derive(Eq, PartialEq, Hash, PartialOrd, Ord)]
struct ExprKey(Vec<Option<Value>>);

impl ExprKey {
    fn contains_null(&self) -> bool {
        self.0.iter().any(Option::is_none)
    }
}

/// A scratch heap of spilled rows, such as a sort run or a hash join partition.
/// [`SpillHeap::free`] deletes its pages; dropping a heap that was not freed deletes them
/// too and logs a failure, since `Drop` cannot return it.
struct SpillHeap {
    heap: Option<TableHeap>,
}

impl SpillHeap {
    const FREED: &'static str = "a spill heap holds its pages until free consumes it";

    fn new(bpm: Arc<BufferPoolManager>) -> Result<Self, TableHeapError> {
        Ok(Self {
            heap: Some(TableHeap::new(bpm)?),
        })
    }

    fn insert(&self, t: &Tuple) -> Result<(), TableHeapError> {
        self.heap.as_ref().expect(Self::FREED).insert(t).map(drop)
    }

    fn iter(&self) -> TableHeapIterator {
        self.heap.as_ref().expect(Self::FREED).iter()
    }

    /// Deletes the heap's pages. On failure it keeps the pages it could not delete, and
    /// dropping it tries again.
    fn free(mut self) -> Result<(), TableHeapError> {
        let heap = self.heap.take().expect(Self::FREED);
        heap.free().map_err(|FreeError { error, rest }| {
            self.heap = Some(rest);
            error
        })
    }
}

impl Drop for SpillHeap {
    fn drop(&mut self) {
        if let Some(heap) = self.heap.take()
            && let Err(e) = heap.free()
        {
            warn!(
                "could not free the pages of a dropped spill heap from page {}: {}",
                e.rest.first_page_id, e.error
            );
        }
    }
}

/// Frees every heap, carrying on past failures, and returns the first one.
fn free_all(heaps: impl IntoIterator<Item = SpillHeap>) -> Result<(), ExecutorError> {
    let mut first_failure = Ok(());
    for heap in heaps {
        let freed = heap.free();
        if first_failure.is_ok() {
            first_failure = freed;
        }
    }
    first_failure.map_err(ExecutorError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::IndexType;
    use crate::concurrency::{IsolationLevel, TransactionManager};
    use crate::processor::executor::{
        DEFAULT_BATCH_SIZE, DEFAULT_SORT_MEMORY_BUDGET_BYTES, Executor, ExecutorContext,
        collect_rows,
    };
    use crate::processor::executors::test_support::*;
    use crate::processor::plan::{
        AggType, AggregationPlanNode, ColIdx, ColumnAssignment, DeletePlanNode,
        ExternalMergeSortPlanNode, FilterPlanNode, HashJoinPlanNode, InsertPlanNode, JoinPredicate,
        JoinType, LimitPlanNode, NestedIndexJoinPlanNode, NestedLoopJoinPlanNode, Plan,
        ProjectionPlanNode, SeqScanPlanNode, SortExpr, SortPlanNode, UpdatePlanNode,
        WindowFunctionPlanNode,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn joined_row_pads_a_missing_right_row_with_nulls() {
        let left = int_schema(&["a", "b"]);
        let right = int_schema(&["c"]);
        let lt = &int_rows(&left, &[&[1, 2]])[0];
        let rt = &int_rows(&right, &[&[3]])[0];
        assert_eq!(
            joined_row(&left, lt, &right, Some(rt)).unwrap(),
            vec![
                Some(Value::INT(1)),
                Some(Value::INT(2)),
                Some(Value::INT(3))
            ]
        );
        assert_eq!(
            joined_row(&left, lt, &right, None).unwrap(),
            vec![Some(Value::INT(1)), Some(Value::INT(2)), None]
        );
    }

    /// Opens `exec`, drains it, closes it and returns how many child closes `closes` counted.
    fn closes_after_running(exec: &mut dyn Executor, closes: &Rc<Cell<usize>>) -> usize {
        exec.open().expect("open");
        collect_rows(exec).expect("next_batch");
        exec.close().expect("close");
        closes.get()
    }

    #[test]
    fn every_executor_with_inputs_closes_each_of_them() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[1], &[2]]);
        let pair_schema = int_schema(&["l", "r"]);
        let col_a = || col(0);
        let always: JoinPredicate = Expression::Constant(Some(Value::BOOLEAN(true)));
        let (bpm, _dir) = crate::testing::bpm(32);
        let closes = Rc::new(Cell::new(0));
        let child = || CountsCloses::wrap(values_child(&rows), &closes);

        let filter = FilterPlanNode {
            schema: schema.clone(),
            predicate: Expression::Constant(Some(Value::BOOLEAN(true))),
            child: Box::new(values_plan(&schema, &[])),
        };
        let projection = ProjectionPlanNode {
            schema: schema.clone(),
            expressions: vec![col_a()],
            child: Box::new(values_plan(&schema, &[])),
        };
        let limit = LimitPlanNode {
            schema: schema.clone(),
            limit: 1,
            child: Box::new(values_plan(&schema, &[])),
        };
        let aggregation = AggregationPlanNode {
            schema: int_schema(&["count"]),
            child: Box::new(values_plan(&schema, &[])),
            group_bys: vec![],
            aggregates: vec![(col_a(), AggType::COUNT)],
        };
        let window = WindowFunctionPlanNode {
            schema: int_schema(&["a", "count"]),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: None,
            partition_by_exprs: None,
            aggregates: vec![(col_a(), AggType::COUNT)],
            columns: vec![col_a()],
        };
        let sort = SortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: vec![SortExpr::ascending(col_a())],
        };
        let external_sort = ExternalMergeSortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: vec![SortExpr::ascending(col_a())],
        };
        let nested_loop = NestedLoopJoinPlanNode {
            schema: pair_schema.clone(),
            left: Box::new(values_plan(&schema, &[])),
            right: Box::new(values_plan(&schema, &[])),
            predicate: always,
            join_type: JoinType::Inner,
        };
        let hash = HashJoinPlanNode {
            schema: pair_schema.clone(),
            left: Box::new(values_plan(&schema, &[])),
            right: Box::new(values_plan(&schema, &[])),
            left_exprs: vec![col_a()],
            right_exprs: vec![col_a()],
            residual: None,
            join_type: JoinType::Inner,
        };

        let mut cases: Vec<(&str, Box<dyn Executor + '_>, usize)> = vec![
            ("filter", Box::new(FilterExecutor::new(&filter, child())), 1),
            (
                "projection",
                Box::new(ProjectionExecutor::new(&projection, child())),
                1,
            ),
            ("limit", Box::new(LimitExecutor::new(&limit, child())), 1),
            (
                "aggregation",
                Box::new(AggregationExecutor::new(
                    &aggregation,
                    child(),
                    DEFAULT_BATCH_SIZE,
                )),
                1,
            ),
            (
                "window",
                Box::new(WindowFunctionExecutor::new(
                    &window,
                    child(),
                    DEFAULT_BATCH_SIZE,
                )),
                1,
            ),
            (
                "sort",
                Box::new(SortExecutor::new(&sort, child(), DEFAULT_BATCH_SIZE)),
                1,
            ),
            (
                "external sort",
                Box::new(ExternalMergeSortExecutor::new(
                    &external_sort,
                    child(),
                    bpm.clone(),
                    DEFAULT_BATCH_SIZE,
                    DEFAULT_SORT_MEMORY_BUDGET_BYTES,
                )),
                1,
            ),
            (
                "nested loop join",
                Box::new(NestedLoopJoinExecutor::new(
                    &nested_loop,
                    child(),
                    child(),
                    DEFAULT_BATCH_SIZE,
                )),
                2,
            ),
            (
                "hash join",
                Box::new(HashJoinExecutor::new(
                    &hash,
                    child(),
                    child(),
                    bpm.clone(),
                    DEFAULT_BATCH_SIZE,
                )),
                2,
            ),
        ];
        for (name, exec, inputs) in &mut cases {
            closes.set(0);
            assert_eq!(
                closes_after_running(exec.as_mut(), &closes),
                *inputs,
                "{name}"
            );
        }
    }

    #[test]
    fn every_executor_that_writes_or_probes_a_table_closes_its_input() {
        let (catalog, bpm, _dir) = crate::testing::catalog(32);
        let schema = int_schema(&["a"]);
        let table_oid = catalog
            .create_table("t".to_string(), schema.clone())
            .expect("create table");
        let index_oid = catalog
            .create_index(
                bpm.clone(),
                table_oid,
                "t_a".to_string(),
                &["a"],
                IndexType::BPlusTree,
            )
            .expect("create index");
        let txn_mgr = Arc::new(TransactionManager::new(bpm.clone()));
        let ctx = ExecutorContext::new(
            &catalog,
            bpm.clone(),
            txn_mgr.begin(IsolationLevel::SnapshotIsolation),
            txn_mgr.clone(),
        );
        let rows = values_plan(&schema, &[&[1], &[2]]);
        let scan = Plan::SeqScan(SeqScanPlanNode {
            table_oid,
            schema: schema.clone(),
        });
        let closes = Rc::new(Cell::new(0));

        let insert = InsertPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            table_oid,
        };
        let mut exec = InsertExecutor::new(
            &ctx,
            &insert,
            CountsCloses::wrap(values_child(&rows), &closes),
        );
        assert_eq!(closes_after_running(&mut exec, &closes), 1, "insert");

        let probe = NestedIndexJoinPlanNode {
            schema: int_schema(&["l", "r"]),
            child: Box::new(values_plan(&schema, &[])),
            key_predicate: col(0),
            table_oid,
            table_schema: schema.clone(),
            index_oid,
            join_type: JoinType::Inner,
        };
        closes.set(0);
        let mut exec = NestedIndexJoinExecutor::new(
            &ctx,
            &probe,
            CountsCloses::wrap(values_child(&rows), &closes),
            DEFAULT_BATCH_SIZE,
        );
        assert_eq!(
            closes_after_running(&mut exec, &closes),
            1,
            "nested index join"
        );

        let update = UpdatePlanNode {
            schema: schema.clone(),
            assignments: vec![ColumnAssignment {
                col_idx: ColIdx(0),
                value: col(0),
            }],
            child: Box::new(values_plan(&schema, &[])),
            table_oid,
        };
        closes.set(0);
        let scan_child = || {
            CountsCloses::wrap(
                Box::new(SeqScanExecutor::new(
                    &ctx,
                    scan_node(&scan),
                    DEFAULT_BATCH_SIZE,
                )),
                &closes,
            )
        };
        let mut exec = UpdateExecutor::new(&ctx, &update, scan_child());
        assert_eq!(closes_after_running(&mut exec, &closes), 1, "update");

        let delete = DeletePlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            table_oid,
        };
        closes.set(0);
        let mut exec = DeleteExecutor::new(&ctx, &delete, scan_child());
        assert_eq!(closes_after_running(&mut exec, &closes), 1, "delete");
    }

    fn scan_node(plan: &Plan) -> &SeqScanPlanNode {
        let Plan::SeqScan(scan) = plan else {
            panic!("expected a SeqScan plan");
        };
        scan
    }
}
