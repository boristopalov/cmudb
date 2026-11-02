use crate::buffer_pool::BufferPoolManager;
use crate::catalog::index_schema::{TableIndex, TableIndexIter};
use crate::catalog::{IndexInfo, Schema, TableInfo, Value};
use crate::concurrency::mvcc;
use crate::index::IndexError;
use crate::processor::executor::{
    Batch, Executor, ExecutorContext, ExecutorError, Row, collect_rows,
};
use crate::processor::executors::{
    ExprKey, SpillHeap, eval_key, free_all, is_true, joined_row, not_opened,
};
use crate::processor::plan::{
    Expression, HashJoinPlanNode, InputRow, JoinPredicate, JoinRow, JoinType,
    NestedIndexJoinPlanNode, NestedLoopJoinPlanNode,
};
use crate::table_heap::{RecordId, TableHeapIterator, Tuple};
use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Bound;
use std::sync::Arc;
use std::vec::IntoIter as VecIter;

/// The outer (left) side of a join: rows are pulled from the child a batch at a time and
/// handed to the join one at a time.
struct OuterRows {
    buf: VecIter<Row>,
    exhausted: bool,
}

impl OuterRows {
    fn new() -> Self {
        Self {
            buf: Vec::new().into_iter(),
            exhausted: false,
        }
    }

    fn next(&mut self, child: &mut dyn Executor) -> Result<Option<Tuple>, ExecutorError> {
        loop {
            if let Some((tuple, _)) = self.buf.next() {
                return Ok(Some(tuple));
            }
            if self.exhausted {
                return Ok(None);
            }
            match child.next_batch()? {
                Some(batch) => self.buf = batch.into_iter(),
                None => self.exhausted = true,
            }
        }
    }
}

/// Joins every left row with every buffered right row that satisfies the predicate. One left
/// row can match more rows than fit in a batch, so `right_cursor` keeps the position in the
/// right rows across `next_batch` calls.
pub struct NestedLoopJoinExecutor<'a> {
    plan: &'a NestedLoopJoinPlanNode,
    left: Box<dyn Executor + 'a>,
    right: Box<dyn Executor + 'a>,
    batch_size: usize,

    right_tuples: Vec<Tuple>,
    outer: OuterRows,

    right_cursor: usize,
    curr_left: Option<Tuple>,
    left_matched: bool,
}

impl<'a> NestedLoopJoinExecutor<'a> {
    pub fn new(
        plan: &'a NestedLoopJoinPlanNode,
        left: Box<dyn Executor + 'a>,
        right: Box<dyn Executor + 'a>,
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            left,
            right,
            batch_size,
            right_tuples: Vec::new(),
            outer: OuterRows::new(),
            right_cursor: 0,
            curr_left: None,
            left_matched: false,
        }
    }
}

impl<'a> Executor for NestedLoopJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.left.open()?;
        self.right.open()?;
        self.right_tuples = collect_rows(self.right.as_mut())?
            .into_iter()
            .map(|(tup, _)| tup)
            .collect();
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let mut out: Vec<Row> = Vec::with_capacity(self.batch_size);
        while out.len() < self.batch_size {
            if self.curr_left.is_none() {
                let Some(lt) = self.outer.next(self.left.as_mut())? else {
                    break;
                };
                self.curr_left = Some(lt);
                self.right_cursor = 0;
                self.left_matched = false;
            }
            let Some(lt) = self.curr_left.as_ref() else {
                continue;
            };
            let left_schema = self.left.schema();
            let right_schema = self.right.schema();

            while self.right_cursor < self.right_tuples.len() && out.len() < self.batch_size {
                let rt = &self.right_tuples[self.right_cursor];
                self.right_cursor += 1;
                let predicate_result = self.plan.predicate.evaluate_on(&JoinRow {
                    left: InputRow {
                        tuple: lt,
                        schema: left_schema,
                    },
                    right: InputRow {
                        tuple: rt,
                        schema: right_schema,
                    },
                })?;
                if is_true(predicate_result) {
                    self.left_matched = true;
                    let row = joined_row(left_schema, lt, right_schema, Some(rt))?;
                    out.push((self.plan.schema.encode_tuple(&row)?, RecordId::RESERVED));
                }
            }

            let left_row_finished = self.right_cursor >= self.right_tuples.len();
            if !left_row_finished {
                break;
            }
            if matches!(self.plan.join_type, JoinType::Left) && !self.left_matched {
                let row = joined_row(left_schema, lt, right_schema, None)?;
                out.push((self.plan.schema.encode_tuple(&row)?, RecordId::RESERVED));
            }
            self.curr_left = None;
        }
        Ok(Batch::from_rows(out))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        let left = self.left.close();
        let right = self.right.close();
        left.and(right)
    }
}

pub struct NestedIndexJoinExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a NestedIndexJoinPlanNode,
    child: Box<dyn Executor + 'a>,
    batch_size: usize,
    inner_table: Option<Arc<TableInfo>>,

    index: Option<Arc<IndexInfo>>,
    outer: OuterRows,
    inner_matches: Option<TableIndexIter>,
    curr_left: Option<Tuple>,
    left_matched: bool,
}

impl<'a> NestedIndexJoinExecutor<'a> {
    pub fn new(
        ctx: &'a ExecutorContext<'a>,
        plan: &'a NestedIndexJoinPlanNode,
        child: Box<dyn Executor + 'a>,
        batch_size: usize,
    ) -> Self {
        Self {
            ctx,
            plan,
            child,
            batch_size,
            inner_table: None,
            index: None,
            outer: OuterRows::new(),
            inner_matches: None,
            curr_left: None,
            left_matched: false,
        }
    }
}

impl<'a> Executor for NestedIndexJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let info = self.ctx.catalog.get_index(self.plan.index_oid)?;
        self.index = Some(info);

        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.inner_table = Some(table);
        Ok(())
    }
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let index = self
            .index
            .clone()
            .ok_or_else(|| not_opened("NestedIndexJoinExecutor"))?;
        let table = self
            .inner_table
            .clone()
            .ok_or_else(|| not_opened("NestedIndexJoinExecutor"))?;

        let mut out: Vec<Row> = Vec::with_capacity(self.batch_size);
        while out.len() < self.batch_size {
            if self.curr_left.is_none() {
                let Some(outer) = self.outer.next(self.child.as_mut())? else {
                    break;
                };
                let key = self
                    .plan
                    .key_predicate
                    .evaluate(&outer, self.child.schema())?;
                self.inner_matches = key.map(|v| entries_with_key(&index.index, v)).transpose()?;
                self.curr_left = Some(outer);
                self.left_matched = false;
            }
            let Some(outer) = self.curr_left.as_ref() else {
                continue;
            };
            let child_schema = self.child.schema();

            let mut inner_finished = true;
            while let Some(matches) = self.inner_matches.as_mut() {
                if out.len() >= self.batch_size {
                    inner_finished = false;
                    break;
                }
                let Some(entry) = matches.next() else {
                    break;
                };
                let (key, rid) = entry?;
                let Some(inner_tuple) = mvcc::read_visible_through_index(
                    &self.ctx.txn,
                    &self.ctx.txn_mgr,
                    &table.heap,
                    &index.index,
                    &key,
                    rid,
                )?
                else {
                    continue;
                };
                let row = joined_row(child_schema, outer, &table.schema, Some(&inner_tuple))?;
                out.push((self.plan.schema.encode_tuple(&row)?, RecordId::RESERVED));
                self.left_matched = true;
            }
            if !inner_finished {
                break;
            }

            if matches!(self.plan.join_type, JoinType::Left) && !self.left_matched {
                let row = joined_row(child_schema, outer, &table.schema, None)?;
                out.push((self.plan.schema.encode_tuple(&row)?, RecordId::RESERVED));
            }
            self.curr_left = None;
            self.inner_matches = None;
        }
        Ok(Batch::from_rows(out))
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
    }
}

/// Every index entry whose key equals `key`. Entries can be stale, so callers re-check each
/// one against the version they see.
fn entries_with_key(index: &TableIndex, key: Value) -> Result<TableIndexIter, IndexError> {
    let probe = [key];
    index.scan(Bound::Included(&probe[..]), Bound::Included(&probe[..]))
}

/// Number of on-disk partitions each hash join input is split into.
const HASH_JOIN_PARTITIONS: usize = 64;

/// A left-input row held in a hash join's build table, with whether any right row matched it.
struct BuildRow {
    tuple: Tuple,
    matched: bool,
}

/// Grace hash join: `open` spills both inputs into [`HASH_JOIN_PARTITIONS`] partitions by
/// key hash, then `next` joins one partition pair at a time with an in-memory build table
/// over the left partition, freeing each pair's pages once it is joined. `close` frees the
/// pairs not joined yet. A NULL key matches nothing.
pub struct HashJoinExecutor<'a> {
    plan: &'a HashJoinPlanNode,
    left: Box<dyn Executor + 'a>,
    right: Box<dyn Executor + 'a>,
    bpm: Arc<BufferPoolManager>,
    batch_size: usize,
    left_partitions: VecDeque<SpillHeap>,
    right_partitions: VecDeque<SpillHeap>,

    right_partition_iter: Option<TableHeapIterator>,
    build_table: Option<HashMap<ExprKey, Vec<BuildRow>>>,
    matches_buffer: VecDeque<Tuple>,
}

impl<'a> HashJoinExecutor<'a> {
    pub fn new(
        plan: &'a HashJoinPlanNode,
        left: Box<dyn Executor + 'a>,
        right: Box<dyn Executor + 'a>,
        bpm: Arc<BufferPoolManager>,
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            left,
            right,
            bpm,
            batch_size,
            left_partitions: VecDeque::new(),
            right_partitions: VecDeque::new(),
            right_partition_iter: None,
            build_table: None,
            matches_buffer: VecDeque::new(),
        }
    }

    fn build_partition(
        &self,
        partition: &SpillHeap,
    ) -> Result<HashMap<ExprKey, Vec<BuildRow>>, ExecutorError> {
        let mut table: HashMap<ExprKey, Vec<BuildRow>> = HashMap::new();
        for item in partition.iter() {
            let (tuple, _) = item?;
            let key = eval_key(&self.plan.left_exprs, &tuple, self.left.schema())?;
            table.entry(key).or_default().push(BuildRow {
                tuple,
                matched: false,
            });
        }
        Ok(table)
    }

    /// Frees the partition pair at the front of the queues, which has just been joined.
    fn free_joined_pair(&mut self) -> Result<(), ExecutorError> {
        self.right_partition_iter = None;
        self.build_table = None;
        let left = self.left_partitions.pop_front();
        let right = self.right_partitions.pop_front();
        free_all(left.into_iter().chain(right))
    }
}

/// Whether a hash join's residual condition holds for a pair of rows with equal keys; true
/// when there is none.
fn residual_holds(residual: Option<&JoinPredicate>, pair: &JoinRow) -> Result<bool, ExecutorError> {
    match residual {
        Some(predicate) => Ok(is_true(predicate.evaluate_on(pair)?)),
        None => Ok(true),
    }
}

fn partition_of(key: &ExprKey) -> usize {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    (h.finish() % HASH_JOIN_PARTITIONS as u64) as usize
}

fn spill_into_partitions(
    input: &mut dyn Executor,
    key_exprs: &[Expression],
    partitions: &VecDeque<SpillHeap>,
) -> Result<(), ExecutorError> {
    while let Some(batch) = input.next_batch()? {
        for (tuple, _) in batch {
            let key = eval_key(key_exprs, &tuple, input.schema())?;
            partitions[partition_of(&key)].insert(&tuple)?;
        }
    }
    Ok(())
}

impl<'a> Executor for HashJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.left.open()?;
        self.right.open()?;
        for _ in 0..HASH_JOIN_PARTITIONS {
            self.left_partitions
                .push_back(SpillHeap::new(self.bpm.clone())?);
            self.right_partitions
                .push_back(SpillHeap::new(self.bpm.clone())?);
        }
        spill_into_partitions(
            self.left.as_mut(),
            &self.plan.left_exprs,
            &self.left_partitions,
        )?;
        spill_into_partitions(
            self.right.as_mut(),
            &self.plan.right_exprs,
            &self.right_partitions,
        )?;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let mut out: Vec<Row> = Vec::with_capacity(self.batch_size);
        loop {
            while out.len() < self.batch_size {
                let Some(tup) = self.matches_buffer.pop_front() else {
                    break;
                };
                out.push((tup, RecordId::RESERVED));
            }
            if out.len() >= self.batch_size {
                break;
            }

            let (Some(left_partition), Some(right_partition)) =
                (self.left_partitions.front(), self.right_partitions.front())
            else {
                break;
            };

            let build_table = match self.build_table.take() {
                Some(table) => table,
                None => self.build_partition(left_partition)?,
            };
            let build_table = self.build_table.insert(build_table);
            let right_rows = self
                .right_partition_iter
                .get_or_insert_with(|| right_partition.iter());

            let Some(item) = right_rows.next() else {
                if matches!(self.plan.join_type, JoinType::Left) {
                    for row in build_table.values().flatten().filter(|row| !row.matched) {
                        let values =
                            joined_row(self.left.schema(), &row.tuple, self.right.schema(), None)?;
                        self.matches_buffer
                            .push_back(self.plan.schema.encode_tuple(&values)?);
                    }
                }
                self.free_joined_pair()?;
                continue;
            };

            let (rt, _) = item?;
            let key = eval_key(&self.plan.right_exprs, &rt, self.right.schema())?;
            if key.contains_null() {
                continue;
            }
            for row in build_table.get_mut(&key).into_iter().flatten() {
                let pair = JoinRow {
                    left: InputRow {
                        tuple: &row.tuple,
                        schema: self.left.schema(),
                    },
                    right: InputRow {
                        tuple: &rt,
                        schema: self.right.schema(),
                    },
                };
                if !residual_holds(self.plan.residual.as_ref(), &pair)? {
                    continue;
                }
                row.matched = true;
                let values = joined_row(
                    self.left.schema(),
                    &row.tuple,
                    self.right.schema(),
                    Some(&rt),
                )?;
                self.matches_buffer
                    .push_back(self.plan.schema.encode_tuple(&values)?);
            }
        }
        Ok(Batch::from_rows(out))
    }

    /// Closes both inputs and deletes the pages of the partitions not joined yet.
    fn close(&mut self) -> Result<(), ExecutorError> {
        self.right_partition_iter = None;
        self.build_table = None;
        let left = std::mem::take(&mut self.left_partitions);
        let right = std::mem::take(&mut self.right_partitions);
        let freed = free_all(left.into_iter().chain(right));
        let left = self.left.close();
        let right = self.right.close();
        freed.and(left).and(right)
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::executor::DEFAULT_BATCH_SIZE;
    use crate::processor::executors::test_support::*;
    use crate::table_heap::TableHeapError;

    fn values_hash_join(left: &[&[i32]], right: &[&[i32]]) -> HashJoinPlanNode {
        let side = int_schema(&["k"]);
        HashJoinPlanNode {
            schema: int_schema(&["lk", "rk"]),
            left: Box::new(values_plan(&side, left)),
            right: Box::new(values_plan(&side, right)),
            left_exprs: vec![col(0)],
            right_exprs: vec![col(0)],
            residual: None,
            join_type: JoinType::Inner,
        }
    }

    fn hash_join_exec<'a>(
        plan: &'a HashJoinPlanNode,
        bpm: &Arc<BufferPoolManager>,
    ) -> HashJoinExecutor<'a> {
        HashJoinExecutor::new(
            plan,
            values_child(&plan.left),
            values_child(&plan.right),
            bpm.clone(),
            DEFAULT_BATCH_SIZE,
        )
    }

    #[test]
    fn hash_join_frees_each_partition_pair_once_it_is_joined() {
        let (bpm, _dir) = crate::testing::bpm(16);
        let plan = values_hash_join(&[&[1], &[2], &[3]], &[&[2], &[3], &[4]]);
        let mut exec = hash_join_exec(&plan, &bpm);
        exec.open().unwrap();
        assert_eq!(bpm.live_page_count(), 2 * HASH_JOIN_PARTITIONS);

        let mut joined = 0;
        while let Some(batch) = exec.next_batch().unwrap() {
            joined += batch.len();
        }
        assert_eq!(joined, 2);
        assert_eq!(
            bpm.live_page_count(),
            0,
            "joined partitions kept their pages"
        );
        exec.close().unwrap();
        assert_eq!(bpm.live_page_count(), 0);
    }

    #[test]
    fn closing_a_hash_join_before_it_finishes_frees_the_partitions_left() {
        let (bpm, _dir) = crate::testing::bpm(16);
        let rows: Vec<[i32; 1]> = (0..200).map(|k| [k]).collect();
        let rows: Vec<&[i32]> = rows.iter().map(|r| &r[..]).collect();
        let plan = values_hash_join(&rows, &rows);
        let mut exec = HashJoinExecutor::new(
            &plan,
            values_child(&plan.left),
            values_child(&plan.right),
            bpm.clone(),
            1,
        );
        exec.open().unwrap();
        exec.next_batch().unwrap().expect("a first match");
        assert!(bpm.live_page_count() > 0);

        exec.close().unwrap();
        assert_eq!(bpm.live_page_count(), 0);
    }

    #[test]
    fn dropping_an_opened_hash_join_without_closing_it_frees_its_partitions() {
        let (bpm, _dir) = crate::testing::bpm(16);
        let plan = values_hash_join(&[&[1]], &[&[1]]);
        let mut exec = hash_join_exec(&plan, &bpm);
        exec.open().unwrap();
        assert_eq!(bpm.live_page_count(), 2 * HASH_JOIN_PARTITIONS);
        drop(exec);
        assert_eq!(bpm.live_page_count(), 0);
    }

    #[test]
    fn closing_a_hash_join_reports_a_partition_it_could_not_free() {
        let (bpm, _dir) = crate::testing::bpm(16);
        let plan = values_hash_join(&[&[1]], &[&[1]]);
        let mut exec = hash_join_exec(&plan, &bpm);
        exec.open().unwrap();
        let first_page = exec
            .left_partitions
            .back()
            .and_then(|p| p.heap.as_ref())
            .map(|h| h.first_page_id)
            .unwrap();
        let pin = bpm.read_page(first_page).unwrap();

        let err = exec.close().unwrap_err();
        assert!(
            matches!(
                err,
                ExecutorError::TableHeap(TableHeapError::Bpm(
                    crate::buffer_pool::BpmError::PagePinned { page_id }
                )) if page_id == first_page
            ),
            "{err:?}"
        );
        assert_eq!(
            bpm.live_page_count(),
            1,
            "close stopped at the pinned page instead of freeing the other partitions"
        );
        drop(pin);
    }
}
