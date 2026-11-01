use crate::buffer_pool::BufferPoolManager;
use crate::catalog::index_schema::{IndexValue, TableIndex};
use crate::catalog::{
    Catalog, CatalogError, DataType, IndexInfo, IndexType, Schema, TableInfo, Value,
};
use crate::disk::PAGE_SIZE;
use crate::index::{IndexError, IndexIter};
use crate::processor::plan::{
    AggType, AggregationPlanNode, Arithmetic, CreateIndexPlanNode, CreateTablePlanNode,
    DeletePlanNode, DropIndexPlanNode, DropTablePlanNode, Expression, ExternalMergeSortPlanNode,
    FilterPlanNode, HashJoinPlanNode, IndexScanPlanNode, InsertPlanNode, JoinType, LimitPlanNode,
    NestedIndexJoinPlanNode, NestedLoopJoinPlanNode, ProjectionPlanNode, SeqScanPlanNode,
    SortPlanNode, UpdatePlanNode, ValuesPlanNode, WindowFunctionPlanNode,
};
use crate::table_heap::{RecordId, TableHeap, TableHeapError, TableHeapIterator, Tuple};
use std::cmp::Ordering;
use std::collections::hash_map::IntoIter;
use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Bound;
use std::sync::Arc;
use std::vec::IntoIter as VecIter;

#[derive(Debug)]
pub struct ExecutorError(pub String);

impl From<CatalogError> for ExecutorError {
    fn from(e: CatalogError) -> Self {
        ExecutorError(format!("catalog error: {:?}", e))
    }
}

impl From<TableHeapError> for ExecutorError {
    fn from(e: TableHeapError) -> Self {
        ExecutorError(format!("table heap error: {:?}", e))
    }
}

impl From<IndexError> for ExecutorError {
    fn from(e: IndexError) -> Self {
        ExecutorError(format!("index error: {:?}", e))
    }
}

/// A pull-based plan operator. `open` prepares state and opens the children, then each
/// `next` call yields one row until it returns `Ok(None)`.
pub trait Executor {
    fn open(&mut self) -> Result<(), ExecutorError>;
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError>;
    /// Layout of the tuples `next` yields.
    fn schema(&self) -> &Schema;
    /// Releases resources early; dropping the executor releases the rest.
    fn close(&mut self) -> Result<(), ExecutorError>;
}

fn not_opened(executor: &str) -> ExecutorError {
    ExecutorError(format!("{executor}::next called before open"))
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

pub struct SeqScanExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a SeqScanPlanNode,
    iter: Option<TableHeapIterator>,
}

impl<'a> SeqScanExecutor<'a> {
    pub fn new(catalog: &'a Catalog, plan: &'a SeqScanPlanNode) -> Self {
        Self {
            catalog,
            plan,
            iter: None,
        }
    }
}

impl<'a> Executor for SeqScanExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.catalog.get_table(self.plan.table_oid)?;
        self.iter = Some(table.heap.iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("SeqScanExecutor"))?;
        iter.next().transpose().map_err(ExecutorError::from)
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct InsertExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a InsertPlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
    done: bool,
}

impl<'a> InsertExecutor<'a> {
    pub fn new(
        catalog: &'a Catalog,
        plan: &'a InsertPlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            catalog,
            plan,
            child,
            table: None,
            done: false,
        }
    }
}

impl<'a> Executor for InsertExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let table = self.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.done {
            return Err(ExecutorError("already executed".to_string()));
        }

        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("InsertExecutor"))?;

        let mut count: i32 = 0;
        while let Some(res) = self.child.next()? {
            let rid = table.heap.insert(&res.0)?;
            for idx in self.catalog.get_table_indexes(self.plan.table_oid)? {
                idx.as_ref().index.insert(&res.0, rid)?;
            }
            count += 1;
        }

        self.done = true;
        Ok(Some((inserted_count_tuple(count), RecordId::RESERVED)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// The single row `InsertExecutor` yields: the inserted-row count as raw native-endian
/// `i32` bytes, not encoded with the plan's schema.
fn inserted_count_tuple(count: i32) -> Tuple {
    Tuple {
        data: count.to_ne_bytes().to_vec(),
    }
}

/// Yields one tuple per row of a `VALUES` list.
pub struct ValuesExecutor<'a> {
    plan: &'a ValuesPlanNode,
    cursor: u32,
}

impl<'a> ValuesExecutor<'a> {
    pub fn new(plan: &'a ValuesPlanNode) -> Self {
        Self { plan, cursor: 0 }
    }
}

impl<'a> Executor for ValuesExecutor<'a> {
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.cursor as usize >= self.plan.rows.len() {
            return Ok(None);
        }
        let exprs = &self.plan.rows[self.cursor as usize];
        let no_input_row = Tuple { data: vec![] };
        let vals: Vec<Option<Value>> = exprs
            .iter()
            .map(|e| e.evaluate(&no_input_row, self.schema()))
            .collect::<Result<Vec<Option<Value>>, ExecutorError>>()?;

        let tuple = self.schema().encode_tuple(&vals)?;
        self.cursor += 1;
        Ok(Some((tuple, RecordId::RESERVED)))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct UpdateExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a UpdatePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> UpdateExecutor<'a> {
    pub fn new(
        catalog: &'a Catalog,
        plan: &'a UpdatePlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            catalog,
            plan,
            child,
            table: None,
        }
    }
}

impl<'a> Executor for UpdateExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let table = self.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("UpdateExecutor"))?;

        let Some((tuple_to_update, rid)) = self.child.next()? else {
            return Ok(None);
        };

        let mut new_values = row_values(self.schema(), &tuple_to_update)?;
        for assignment in &self.plan.assignments {
            new_values[assignment.col_idx as usize] =
                assignment.value.evaluate(&tuple_to_update, self.schema())?;
        }

        let new_tuple = self.schema().encode_tuple(new_values.as_ref())?;
        let (new_rid, _) = table.heap.update(rid, &new_tuple)?;

        for idx in self.catalog.get_table_indexes(self.plan.table_oid)? {
            idx.as_ref().index.remove(&tuple_to_update)?;
            idx.as_ref().index.insert(&new_tuple, new_rid)?;
        }
        Ok(Some((new_tuple, new_rid)))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct DeleteExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a DeletePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> DeleteExecutor<'a> {
    pub fn new(
        catalog: &'a Catalog,
        plan: &'a DeletePlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            catalog,
            plan,
            child,
            table: None,
        }
    }
}

impl<'a> Executor for DeleteExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let table = self.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("DeleteExecutor"))?;

        let Some((tup, rid)) = self.child.next()? else {
            return Ok(None);
        };

        table.heap.delete(&rid)?;
        for idx in self.catalog.get_table_indexes(self.plan.table_oid)? {
            idx.as_ref().index.remove(&tup)?;
        }
        Ok(Some((tup, rid)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct IndexScanExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a IndexScanPlanNode,
    table: Option<Arc<TableInfo>>,
    iter: Option<IndexIter>,
}

impl<'a> IndexScanExecutor<'a> {
    pub fn new(catalog: &'a Catalog, plan: &'a IndexScanPlanNode) -> Self {
        Self {
            catalog,
            plan,
            table: None,
            iter: None,
        }
    }
}

impl<'a> Executor for IndexScanExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let info = self.catalog.get_index(self.plan.index_oid)?;
        let iter = info.as_ref().index.scan(
            self.plan.start.as_ref().map(Vec::as_slice),
            self.plan.end.as_ref().map(Vec::as_slice),
        )?;
        self.iter = Some(iter);

        let table = self.catalog.get_table(info.as_ref().table_oid)?;
        self.table = Some(table);

        Ok(())
    }
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("IndexScanExecutor"))?;
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("IndexScanExecutor"))?;

        let Some(item) = iter.next() else {
            return Ok(None);
        };
        let (_key, IndexValue::IndexValue(rid)) = item?;

        let (rid, tuple) = table.heap.get(rid)?;
        Ok(Some((tuple, rid)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct AggregationExecutor<'a> {
    plan: &'a AggregationPlanNode,
    child: Box<dyn Executor + 'a>,
    iter: Option<IntoIter<ExprKey, AggVal>>,
}

impl<'a> AggregationExecutor<'a> {
    pub fn new(plan: &'a AggregationPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self {
            plan,
            child,
            iter: None,
        }
    }
}

impl<'a> Executor for AggregationExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let mut agg_hash_table: HashMap<ExprKey, AggVal> = HashMap::new();
        while let Some((tuple, _)) = self.child.next()? {
            let schema = self.child.schema();
            let groups = ExprKey(
                self.plan
                    .group_bys
                    .iter()
                    .map(|expr| expr.evaluate(&tuple, schema).ok()?)
                    .collect(),
            );
            let entry = agg_hash_table.entry(groups).or_insert_with(|| {
                AggVal(
                    self.plan
                        .aggregates
                        .iter()
                        .map(|(_, agg_type)| AggState::new(agg_type))
                        .collect(),
                )
            });
            for (state, (expr, _)) in entry.0.iter_mut().zip(self.plan.aggregates.iter()) {
                state.update(expr.evaluate(&tuple, schema)?)?;
            }
        }
        self.iter = Some(agg_hash_table.into_iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("AggregationExecutor"))?;
        let Some((key, val)) = iter.next() else {
            return Ok(None);
        };
        let mut out: Vec<Option<Value>> = Vec::with_capacity(key.0.len() + val.0.len());
        out.extend(key.0);
        for v in val.0 {
            out.push(v.value()?);
        }

        let tuple = self.schema().encode_tuple(&out)?;
        Ok(Some((tuple, RecordId::RESERVED)))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

#[derive(Eq, PartialEq, Hash)]
struct ExprKey(Vec<Option<Value>>);

impl ExprKey {
    fn contains_null(&self) -> bool {
        self.0.iter().any(Option::is_none)
    }
}

struct AggVal(Vec<AggState>);

enum AggState {
    Max(Option<Value>),
    Min(Option<Value>),
    Sum(Option<Value>),
    Avg { sum: Option<Value>, count: i32 },
    Count(i32),
    Rank(i32),
}

impl AggState {
    fn new(agg_type: &AggType) -> Self {
        match agg_type {
            AggType::AVG => Self::Avg {
                sum: None,
                count: 0,
            },
            AggType::MAX => Self::Max(None),
            AggType::MIN => Self::Min(None),
            AggType::SUM => Self::Sum(None),
            AggType::COUNT => Self::Count(0),
            AggType::RANK => Self::Rank(1),
        }
    }

    /// Moves a RANK to `pos` when a new peer group starts; other aggregates ignore it.
    fn update_rank(&mut self, new_peer_group: bool, pos: i32) {
        if let Self::Rank(rank) = self
            && new_peer_group
        {
            *rank = pos;
        }
    }

    /// Folds one input value in. NULLs are skipped, so `COUNT(col)` counts non-NULL values;
    /// RANK ignores values and moves through `update_rank`. Errors when a SUM or AVG
    /// overflows INT.
    fn update(&mut self, v: Option<Value>) -> Result<(), ExecutorError> {
        let Some(v) = v else { return Ok(()) };
        match self {
            Self::Count(count) => *count += 1,
            Self::Sum(current) => *current = Some(add_to(*current, v)?),
            Self::Avg { sum, count } => {
                *sum = Some(add_to(*sum, v)?);
                *count += 1;
            }
            Self::Max(current) => *current = Some(current.map_or(v, |c| c.max(v))),
            Self::Min(current) => *current = Some(current.map_or(v, |c| c.min(v))),
            Self::Rank(_) => {}
        }
        Ok(())
    }

    /// The aggregate over the values folded in so far. AVG of INTs truncates toward zero.
    fn value(&self) -> Result<Option<Value>, ExecutorError> {
        Ok(match self {
            Self::Sum(v) | Self::Max(v) | Self::Min(v) => *v,
            Self::Count(c) | Self::Rank(c) => Some(Value::INT(*c)),
            Self::Avg { sum: None, .. } => None,
            Self::Avg {
                sum: Some(sum),
                count,
            } => Some(Arithmetic::Div.apply(*sum, Value::INT(*count))?),
        })
    }
}

fn add_to(total: Option<Value>, v: Value) -> Result<Value, ExecutorError> {
    match total {
        Some(total) => Arithmetic::Add.apply(total, v),
        None => Ok(v),
    }
}

pub struct NestedLoopJoinExecutor<'a> {
    plan: &'a NestedLoopJoinPlanNode,
    left: Box<dyn Executor + 'a>,
    right: Box<dyn Executor + 'a>,

    right_tuples: Vec<Tuple>,

    right_cursor: usize,
    curr_left: Option<Tuple>,
    left_matched: bool,
}

impl<'a> NestedLoopJoinExecutor<'a> {
    pub fn new(
        plan: &'a NestedLoopJoinPlanNode,
        left: Box<dyn Executor + 'a>,
        right: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            plan,
            left,
            right,
            right_tuples: Vec::new(),
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
        while let Some((tup, _)) = self.right.next()? {
            self.right_tuples.push(tup);
        }
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        loop {
            if self.curr_left.is_none() {
                let Some((lt, _)) = self.left.next()? else {
                    return Ok(None);
                };
                self.curr_left = Some(lt);
                self.right_cursor = 0;
                self.left_matched = false;
            };
            let Some(lt) = self.curr_left.as_ref() else {
                continue;
            };

            while self.right_cursor < self.right_tuples.len() {
                let rt = &self.right_tuples[self.right_cursor];
                self.right_cursor += 1;
                let predicate_result = self.plan.predicate.evaluate_join(
                    lt,
                    self.left.schema(),
                    rt,
                    self.right.schema(),
                )?;
                if is_true(predicate_result) {
                    self.left_matched = true;
                    let row = joined_row(self.left.schema(), lt, self.right.schema(), Some(rt))?;
                    let out = self.schema().encode_tuple(&row)?;
                    return Ok(Some((out, RecordId::RESERVED)));
                }
            }

            let unmatched_left_row = match self.plan.join_type {
                JoinType::Left if !self.left_matched => Some(joined_row(
                    self.left.schema(),
                    lt,
                    self.right.schema(),
                    None,
                )?),
                _ => None,
            };
            self.curr_left = None;
            if let Some(row) = unmatched_left_row {
                let out = self.schema().encode_tuple(&row)?;
                return Ok(Some((out, RecordId::RESERVED)));
            }
        }
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct NestedIndexJoinExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a NestedIndexJoinPlanNode,
    child: Box<dyn Executor + 'a>,
    inner_table: Option<Arc<TableInfo>>,

    index: Option<Arc<IndexInfo>>,
    inner_matches: Option<IndexIter>,
    curr_left: Option<Tuple>,
    left_matched: bool,
}

impl<'a> NestedIndexJoinExecutor<'a> {
    pub fn new(
        catalog: &'a Catalog,
        plan: &'a NestedIndexJoinPlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            catalog,
            plan,
            child,
            inner_table: None,
            index: None,
            inner_matches: None,
            curr_left: None,
            left_matched: false,
        }
    }
}

impl<'a> Executor for NestedIndexJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let info = self.catalog.get_index(self.plan.index_oid)?;
        self.index = Some(info);

        let table = self.catalog.get_table(self.plan.table_oid)?;
        self.inner_table = Some(table);
        Ok(())
    }
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| not_opened("NestedIndexJoinExecutor"))?;
        let table = self
            .inner_table
            .as_ref()
            .ok_or_else(|| not_opened("NestedIndexJoinExecutor"))?;

        loop {
            if self.curr_left.is_none() {
                let Some((outer, _)) = self.child.next()? else {
                    return Ok(None);
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

            if let Some(entry) = self.inner_matches.as_mut().and_then(Iterator::next) {
                let (_, IndexValue::IndexValue(rid)) = entry?;
                let (_, inner_tuple) = table.heap.get(rid)?;
                let row = joined_row(
                    self.child.schema(),
                    outer,
                    &table.schema,
                    Some(&inner_tuple),
                )?;
                let out = self.schema().encode_tuple(&row)?;
                self.left_matched = true;
                return Ok(Some((out, RecordId::RESERVED)));
            }

            let unmatched_left_row = match self.plan.join_type {
                JoinType::Left if !self.left_matched => {
                    Some(joined_row(self.child.schema(), outer, &table.schema, None)?)
                }
                _ => None,
            };
            self.curr_left = None;
            self.inner_matches = None;
            if let Some(row) = unmatched_left_row {
                let out = self.schema().encode_tuple(&row)?;
                return Ok(Some((out, RecordId::RESERVED)));
            }
        }
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Every index entry whose key equals `key`.
fn entries_with_key(index: &TableIndex, key: Value) -> Result<IndexIter, IndexError> {
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
/// over the left partition. A NULL key matches nothing.
pub struct HashJoinExecutor<'a> {
    plan: &'a HashJoinPlanNode,
    left: Box<dyn Executor + 'a>,
    right: Box<dyn Executor + 'a>,
    bpm: Arc<BufferPoolManager>,
    left_partitions: Vec<TableHeap>,
    right_partitions: Vec<TableHeap>,

    right_partition_iter: Option<TableHeapIterator>,
    build_table: Option<HashMap<ExprKey, Vec<BuildRow>>>,
    partition_cursor: usize,
    matches_buffer: VecDeque<Tuple>,
}

impl<'a> HashJoinExecutor<'a> {
    pub fn new(
        plan: &'a HashJoinPlanNode,
        left: Box<dyn Executor + 'a>,
        right: Box<dyn Executor + 'a>,
        bpm: Arc<BufferPoolManager>,
    ) -> Self {
        Self {
            plan,
            left,
            right,
            bpm,
            left_partitions: Vec::new(),
            right_partitions: Vec::new(),
            right_partition_iter: None,
            build_table: None,
            partition_cursor: 0,
            matches_buffer: VecDeque::new(),
        }
    }

    fn build_partition(
        &self,
        partition: usize,
    ) -> Result<HashMap<ExprKey, Vec<BuildRow>>, ExecutorError> {
        let mut table: HashMap<ExprKey, Vec<BuildRow>> = HashMap::new();
        for item in self.left_partitions[partition].iter() {
            let (tuple, _) = item?;
            let key = eval_key(&self.plan.left_exprs, &tuple, self.left.schema())?;
            table.entry(key).or_default().push(BuildRow {
                tuple,
                matched: false,
            });
        }
        Ok(table)
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
    partitions: &[TableHeap],
) -> Result<(), ExecutorError> {
    while let Some((tuple, _)) = input.next()? {
        let key = eval_key(key_exprs, &tuple, input.schema())?;
        partitions[partition_of(&key)].insert(&tuple)?;
    }
    Ok(())
}

impl<'a> Executor for HashJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.left.open()?;
        self.right.open()?;
        for _ in 0..HASH_JOIN_PARTITIONS {
            self.left_partitions.push(TableHeap::new(self.bpm.clone())?);
            self.right_partitions
                .push(TableHeap::new(self.bpm.clone())?);
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

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        loop {
            if let Some(out) = self.matches_buffer.pop_back() {
                return Ok(Some((out, RecordId::RESERVED)));
            };

            if self.partition_cursor >= self.left_partitions.len() {
                return Ok(None);
            };

            let build_table = match self.build_table.take() {
                Some(table) => table,
                None => self.build_partition(self.partition_cursor)?,
            };
            let build_table = self.build_table.insert(build_table);
            let right_rows = self
                .right_partition_iter
                .get_or_insert_with(|| self.right_partitions[self.partition_cursor].iter());

            let Some(item) = right_rows.next() else {
                if matches!(self.plan.join_type, JoinType::Left) {
                    for row in build_table.values().flatten().filter(|row| !row.matched) {
                        let values =
                            joined_row(self.left.schema(), &row.tuple, self.right.schema(), None)?;
                        self.matches_buffer
                            .push_back(self.plan.schema.encode_tuple(&values)?);
                    }
                }
                self.right_partition_iter = None;
                self.partition_cursor += 1;
                self.build_table = None;
                continue;
            };

            let (rt, _) = item?;
            let key = eval_key(&self.plan.right_exprs, &rt, self.right.schema())?;
            if key.contains_null() {
                continue;
            }
            for row in build_table.get_mut(&key).into_iter().flatten() {
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
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct SortExecutor<'a> {
    plan: &'a SortPlanNode,
    child: Box<dyn Executor + 'a>,
    sorted: VecIter<(Tuple, RecordId)>,
}

impl<'a> SortExecutor<'a> {
    pub fn new(plan: &'a SortPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self {
            plan,
            child,
            sorted: Vec::new().into_iter(),
        }
    }
}

impl<'a> Executor for SortExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let mut buf: Vec<(Tuple, RecordId)> = Vec::new();
        while let Some(row) = self.child.next()? {
            buf.push(row);
        }
        let schema = &self.plan.schema;
        let order_by = &self.plan.order_by_exprs;
        buf.sort_by(|a, b| cmp_tuples(&a.0, &b.0, order_by, schema));
        self.sorted = buf.into_iter();
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        Ok(self.sorted.next())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
    }
}

pub struct ExternalMergeSortExecutor<'a> {
    plan: &'a ExternalMergeSortPlanNode,
    child: Box<dyn Executor + 'a>,
    bpm: Arc<BufferPoolManager>,
    iter: Option<TableHeapIterator>,
}

impl<'a> ExternalMergeSortExecutor<'a> {
    pub fn new(
        plan: &'a ExternalMergeSortPlanNode,
        child: Box<dyn Executor + 'a>,
        bpm: Arc<BufferPoolManager>,
    ) -> Self {
        Self {
            plan,
            child,
            bpm,
            iter: None,
        }
    }
}

fn cmp_tuples(a: &Tuple, b: &Tuple, exprs: &[Expression], schema: &Schema) -> Ordering {
    for expr in exprs {
        let va = expr.evaluate(a, schema).ok().flatten();
        let vb = expr.evaluate(b, schema).ok().flatten();
        match va.cmp(&vb) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

/// Input bytes the external merge sort sorts in memory before spilling them as one run.
const SORT_RUN_BUDGET_BYTES: usize = 5_000 * PAGE_SIZE;

impl<'a> ExternalMergeSortExecutor<'a> {
    fn spill_sorted_run(&self, rows: &mut Vec<Tuple>) -> Result<TableHeap, ExecutorError> {
        let (order_by, schema) = (&self.plan.order_by_exprs, &self.plan.schema);
        rows.sort_by(|a, b| cmp_tuples(a, b, order_by, schema));
        let run = TableHeap::new(self.bpm.clone())?;
        for t in rows.drain(..) {
            run.insert(&t)?;
        }
        Ok(run)
    }

    fn merge_runs(&self, a: &TableHeap, b: &TableHeap) -> Result<TableHeap, ExecutorError> {
        let (order_by, schema) = (&self.plan.order_by_exprs, &self.plan.schema);
        let out = TableHeap::new(self.bpm.clone())?;
        let mut a_rows = a.iter();
        let mut b_rows = b.iter();
        let mut a_head = a_rows.next().transpose()?.map(|(t, _)| t);
        let mut b_head = b_rows.next().transpose()?.map(|(t, _)| t);
        loop {
            let take_a = match (&a_head, &b_head) {
                (Some(ta), Some(tb)) => cmp_tuples(ta, tb, order_by, schema) != Ordering::Greater,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => return Ok(out),
            };
            if take_a {
                if let Some(t) = a_head.take() {
                    out.insert(&t)?;
                }
                a_head = a_rows.next().transpose()?.map(|(t, _)| t);
            } else {
                if let Some(t) = b_head.take() {
                    out.insert(&t)?;
                }
                b_head = b_rows.next().transpose()?.map(|(t, _)| t);
            }
        }
    }
}

impl<'a> Executor for ExternalMergeSortExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let mut run_bytes: usize = 0;
        let mut buf: Vec<Tuple> = Vec::new();
        let mut runs: Vec<TableHeap> = Vec::new();

        while let Some((tup, _)) = self.child.next()? {
            run_bytes += tup.data.len();
            buf.push(tup);
            if run_bytes >= SORT_RUN_BUDGET_BYTES {
                runs.push(self.spill_sorted_run(&mut buf)?);
                run_bytes = 0;
            }
        }
        runs.push(self.spill_sorted_run(&mut buf)?);

        while runs.len() > 1 {
            let mut next_runs: Vec<TableHeap> = Vec::with_capacity(runs.len().div_ceil(2));
            let mut pending = runs.into_iter();
            while let Some(a) = pending.next() {
                match pending.next() {
                    Some(b) => next_runs.push(self.merge_runs(&a, &b)?),
                    None => next_runs.push(a),
                }
            }
            runs = next_runs;
        }

        let sorted = runs
            .pop()
            .ok_or_else(|| ExecutorError("external sort produced no runs".to_string()))?;
        self.iter = Some(sorted.iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("ExternalMergeSortExecutor"))?;
        let Some(item) = iter.next() else {
            return Ok(None);
        };
        let (tup, _) = item?;
        Ok(Some((tup, RecordId::RESERVED)))
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct LimitExecutor<'a> {
    plan: &'a LimitPlanNode,
    child: Box<dyn Executor + 'a>,
    count: u32,
}

impl<'a> LimitExecutor<'a> {
    pub fn new(plan: &'a LimitPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self {
            plan,
            child,
            count: 0,
        }
    }
}

impl<'a> Executor for LimitExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()
    }
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.count >= self.plan.limit {
            return Ok(None);
        }
        let row = self.child.next()?;
        if row.is_some() {
            self.count += 1;
        }
        Ok(row)
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct WindowFunctionExecutor<'a> {
    plan: &'a WindowFunctionPlanNode,
    child: Box<dyn Executor + 'a>,
    iter: Option<VecIter<Tuple>>,
}

impl<'a> WindowFunctionExecutor<'a> {
    pub fn new(plan: &'a WindowFunctionPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self {
            plan,
            child,
            iter: None,
        }
    }
}

impl<'a> Executor for WindowFunctionExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let mut tups: Vec<Tuple> = vec![];
        while let Some((t, _)) = self.child.next()? {
            tups.push(t);
        }

        let child_schema = self.child.schema();
        let partition_by: &[Expression] = self.plan.partition_by_exprs.as_deref().unwrap_or(&[]);
        let order_by: &[Expression] = self.plan.order_by_exprs.as_deref().unwrap_or(&[]);
        let sort_exprs: Vec<Expression> = partition_by.iter().chain(order_by).cloned().collect();
        tups.sort_by(|a, b| cmp_tuples(a, b, &sort_exprs, child_schema));

        let running = self.plan.order_by_exprs.is_some();
        let fresh_aggregates = || -> Vec<AggState> {
            self.plan
                .aggregates
                .iter()
                .map(|(_, agg_type)| AggState::new(agg_type))
                .collect()
        };
        let mut partition_aggregates = fresh_aggregates();
        let mut peer_buffer: Vec<Tuple> = Vec::new();
        let mut output_tuples: Vec<Tuple> = Vec::with_capacity(tups.len());

        let mut curr_rank = 1;
        let mut row_num = 1;
        let mut prev_partition_key: Option<ExprKey> = None;
        let mut prev_order_key: Option<ExprKey> = None;

        for t in tups.iter() {
            let curr_partition_key = eval_key(partition_by, t, child_schema)?;
            let new_partition = prev_partition_key.as_ref() != Some(&curr_partition_key);
            let curr_order_key = eval_key(order_by, t, child_schema)?;
            let new_peer_group = new_partition || prev_order_key.as_ref() != Some(&curr_order_key);

            let flush_now = if new_partition {
                prev_partition_key.is_some()
            } else {
                running && new_peer_group && !peer_buffer.is_empty()
            };
            if flush_now {
                flush_peer_group(
                    &mut peer_buffer,
                    &mut partition_aggregates,
                    &self.plan.aggregates,
                    child_schema,
                    &self.plan.schema,
                    &mut output_tuples,
                )?;
            }
            if new_partition {
                partition_aggregates = fresh_aggregates();
                row_num = 1;
            }
            if new_peer_group {
                curr_rank = row_num;
            }
            for state in partition_aggregates.iter_mut() {
                state.update_rank(new_peer_group, curr_rank);
            }

            peer_buffer.push(t.clone());
            row_num += 1;
            prev_partition_key = Some(curr_partition_key);
            prev_order_key = Some(curr_order_key);
        }

        if !peer_buffer.is_empty() {
            flush_peer_group(
                &mut peer_buffer,
                &mut partition_aggregates,
                &self.plan.aggregates,
                child_schema,
                &self.plan.schema,
                &mut output_tuples,
            )?;
        }

        self.iter = Some(output_tuples.into_iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("WindowFunctionExecutor"))?;
        Ok(iter.next().map(|tup| (tup, RecordId::RESERVED)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct FilterExecutor<'a> {
    plan: &'a FilterPlanNode,
    child: Box<dyn Executor + 'a>,
}

impl<'a> FilterExecutor<'a> {
    pub fn new(plan: &'a FilterPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self { plan, child }
    }
}

impl<'a> Executor for FilterExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        while let Some((tup, rid)) = self.child.next()? {
            if is_true(self.plan.predicate.evaluate(&tup, self.child.schema())?) {
                return Ok(Some((tup, rid)));
            }
        }
        Ok(None)
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct ProjectionExecutor<'a> {
    plan: &'a ProjectionPlanNode,
    child: Box<dyn Executor + 'a>,
}

impl<'a> ProjectionExecutor<'a> {
    pub fn new(plan: &'a ProjectionPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self { plan, child }
    }
}

impl<'a> Executor for ProjectionExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let Some((tup, rid)) = self.child.next()? else {
            return Ok(None);
        };
        let child_schema = self.child.schema();
        let vals: Vec<Option<Value>> = self
            .plan
            .expressions
            .iter()
            .map(|e| e.evaluate(&tup, child_schema))
            .collect::<Result<_, _>>()?;
        let out = self.plan.schema.encode_tuple(&vals)?;
        Ok(Some((out, rid)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// One-shot status schema used by DDL executors: a single INT column that reports rows affected.
fn ddl_status_schema() -> Schema {
    Schema::new(vec![(DataType::INT, "status".to_string())])
}

pub struct CreateTableExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a CreateTablePlanNode,
    status_schema: Schema,
    done: bool,
}

impl<'a> CreateTableExecutor<'a> {
    pub fn new(catalog: &'a Catalog, plan: &'a CreateTablePlanNode) -> Self {
        Self {
            catalog,
            plan,
            status_schema: ddl_status_schema(),
            done: false,
        }
    }
}

impl<'a> Executor for CreateTableExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.done {
            return Err(ExecutorError("already executed".to_string()));
        }
        self.done = true;
        match self
            .catalog
            .create_table(self.plan.table_name.clone(), self.plan.schema.clone())
        {
            Ok(_oid) => {
                let tup = self.status_schema.encode_tuple(&[Some(Value::INT(1))])?;
                Ok(Some((tup, RecordId::RESERVED)))
            }
            Err(CatalogError::TableExists) if self.plan.if_not_exists => {
                let tup = self.status_schema.encode_tuple(&[Some(Value::INT(0))])?;
                Ok(Some((tup, RecordId::RESERVED)))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn schema(&self) -> &Schema {
        &self.status_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct DropTableExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a DropTablePlanNode,
    status_schema: Schema,
    done: bool,
}

impl<'a> DropTableExecutor<'a> {
    pub fn new(catalog: &'a Catalog, plan: &'a DropTablePlanNode) -> Self {
        Self {
            catalog,
            plan,
            status_schema: ddl_status_schema(),
            done: false,
        }
    }
}

impl<'a> Executor for DropTableExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.done {
            return Err(ExecutorError("already executed".to_string()));
        }
        self.done = true;
        let dropped = self.catalog.drop_table(self.plan.table_oid)?;
        if !dropped && !self.plan.if_exists {
            return Err(ExecutorError(format!(
                "table with oid {} does not exist",
                self.plan.table_oid
            )));
        }
        let tup = self
            .status_schema
            .encode_tuple(&[Some(Value::INT(if dropped { 1 } else { 0 }))])?;
        Ok(Some((tup, RecordId::RESERVED)))
    }

    fn schema(&self) -> &Schema {
        &self.status_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct CreateIndexExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a CreateIndexPlanNode,
    bpm: Arc<BufferPoolManager>,
    status_schema: Schema,
    done: bool,
}

impl<'a> CreateIndexExecutor<'a> {
    pub fn new(
        catalog: &'a Catalog,
        plan: &'a CreateIndexPlanNode,
        bpm: Arc<BufferPoolManager>,
    ) -> Self {
        Self {
            catalog,
            plan,
            bpm,
            status_schema: ddl_status_schema(),
            done: false,
        }
    }
}

impl<'a> Executor for CreateIndexExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.done {
            return Err(ExecutorError("already executed".to_string()));
        }
        self.done = true;

        let table = self.catalog.get_table(self.plan.table_oid)?;
        let col_names: Vec<String> = self
            .plan
            .key_columns
            .iter()
            .map(|idx| {
                table
                    .schema
                    .cols
                    .get(*idx as usize)
                    .map(|c| c.name.clone())
                    .ok_or(CatalogError::ColumnOutOfBounds)
            })
            .collect::<Result<_, _>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(String::as_str).collect();

        self.catalog.create_index(
            self.bpm.clone(),
            self.plan.table_oid,
            self.plan.index_name.clone(),
            &col_refs,
            IndexType::BPlusTree,
        )?;
        let tup = self.status_schema.encode_tuple(&[Some(Value::INT(1))])?;
        Ok(Some((tup, RecordId::RESERVED)))
    }

    fn schema(&self) -> &Schema {
        &self.status_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct DropIndexExecutor<'a> {
    catalog: &'a Catalog,
    plan: &'a DropIndexPlanNode,
    status_schema: Schema,
    done: bool,
}

impl<'a> DropIndexExecutor<'a> {
    pub fn new(catalog: &'a Catalog, plan: &'a DropIndexPlanNode) -> Self {
        Self {
            catalog,
            plan,
            status_schema: ddl_status_schema(),
            done: false,
        }
    }
}

impl<'a> Executor for DropIndexExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.done {
            return Err(ExecutorError("already executed".to_string()));
        }
        self.done = true;
        let dropped = self.catalog.drop_index(self.plan.index_oid)?;
        if !dropped && !self.plan.if_exists {
            return Err(ExecutorError(format!(
                "index with oid {} does not exist",
                self.plan.index_oid
            )));
        }
        let tup = self
            .status_schema
            .encode_tuple(&[Some(Value::INT(if dropped { 1 } else { 0 }))])?;
        Ok(Some((tup, RecordId::RESERVED)))
    }

    fn schema(&self) -> &Schema {
        &self.status_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Folds every buffered peer row into `aggregates`, then emits each of those rows followed
/// by the aggregates' current values, draining `buffer`.
fn flush_peer_group(
    buffer: &mut Vec<Tuple>,
    aggregates: &mut [AggState],
    plan_aggs: &[(Expression, AggType)],
    child_schema: &Schema,
    out_schema: &Schema,
    output: &mut Vec<Tuple>,
) -> Result<(), ExecutorError> {
    for buf_tup in buffer.iter() {
        for (state, (expr, _)) in aggregates.iter_mut().zip(plan_aggs.iter()) {
            state.update(expr.evaluate(buf_tup, child_schema)?)?;
        }
    }
    for buf_tup in buffer.drain(..) {
        let mut combined = row_values(child_schema, &buf_tup)?;
        for agg in aggregates.iter() {
            combined.push(agg.value()?);
        }
        let out = out_schema.encode_tuple(&combined)?;
        output.push(out);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::processor::plan::{LEFT_TUPLE, Plan};
    use crate::replacer::ArcReplacer;
    use tempfile::TempDir;

    fn buffer_pool() -> (Arc<BufferPoolManager>, TempDir) {
        let num_frames = 64;
        let dir = tempfile::tempdir().expect("tempdir");
        let dm = DiskManager::new(&dir.path().join("db"), &dir.path().join("log"))
            .expect("disk manager");
        let bpm = Arc::new(create_buffer_pool_manager(
            num_frames,
            ArcReplacer::new(num_frames),
            DiskScheduler::new(dm),
        ));
        (bpm, dir)
    }

    fn int_schema(names: &[&str]) -> Schema {
        Schema::new(
            names
                .iter()
                .map(|n| (DataType::INT, n.to_string()))
                .collect(),
        )
    }

    fn col(col_idx: u32) -> Expression {
        Expression::Column {
            tuple_idx: LEFT_TUPLE,
            col_idx,
            dtype: DataType::INT,
        }
    }

    fn int_rows(schema: &Schema, rows: &[&[i32]]) -> Vec<Tuple> {
        rows.iter()
            .map(|r| {
                let vals: Vec<Option<Value>> = r.iter().map(|v| Some(Value::INT(*v))).collect();
                schema.encode_tuple(&vals).expect("encode")
            })
            .collect()
    }

    fn decode_ints(schema: &Schema, tuples: impl IntoIterator<Item = Tuple>) -> Vec<Vec<i32>> {
        tuples
            .into_iter()
            .map(|t| {
                row_values(schema, &t)
                    .expect("decode")
                    .into_iter()
                    .map(|v| match v {
                        Some(Value::INT(i)) => i,
                        other => panic!("expected INT, got {other:?}"),
                    })
                    .collect()
            })
            .collect()
    }

    fn values_plan(schema: &Schema, rows: &[&[i32]]) -> Plan {
        Plan::Values(ValuesPlanNode {
            schema: schema.clone(),
            rows: rows
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|v| Expression::Constant(Some(Value::INT(*v))))
                        .collect()
                })
                .collect(),
        })
    }

    fn values_child<'a>(plan: &'a Plan) -> Box<dyn Executor + 'a> {
        let Plan::Values(values) = plan else {
            panic!("expected a Values plan");
        };
        Box::new(ValuesExecutor::new(values))
    }

    fn drain(exec: &mut dyn Executor) -> Vec<Tuple> {
        exec.open().expect("open");
        let mut out = Vec::new();
        while let Some((t, _)) = exec.next().expect("next") {
            out.push(t);
        }
        out
    }

    #[test]
    fn external_sort_merges_spilled_runs_in_order() {
        let (bpm, _dir) = buffer_pool();
        let schema = int_schema(&["k", "v"]);
        let plan = ExternalMergeSortPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: vec![col(0)],
        };
        let exec = ExternalMergeSortExecutor::new(&plan, values_child(&plan.child), bpm);

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
    fn limit_stops_pulling_from_its_child_once_reached() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[1], &[2], &[3]]);
        let plan = LimitPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            limit: 2,
        };
        let mut exec = LimitExecutor::new(&plan, values_child(&rows));
        assert_eq!(
            decode_ints(&schema, drain(&mut exec)),
            vec![vec![1], vec![2]]
        );
        assert!(exec.next().unwrap().is_none());
    }

    #[test]
    fn count_skips_nulls_and_rank_ignores_values() {
        let mut count = AggState::new(&AggType::COUNT);
        count.update(Some(Value::INT(7))).unwrap();
        count.update(None).unwrap();
        count.update(Some(Value::INT(8))).unwrap();
        assert_eq!(count.value().unwrap(), Some(Value::INT(2)));

        let mut rank = AggState::new(&AggType::RANK);
        rank.update(Some(Value::INT(100))).unwrap();
        assert_eq!(rank.value().unwrap(), Some(Value::INT(1)));
        rank.update_rank(false, 5);
        assert_eq!(
            rank.value().unwrap(),
            Some(Value::INT(1)),
            "same peer group keeps its rank"
        );
        rank.update_rank(true, 5);
        assert_eq!(rank.value().unwrap(), Some(Value::INT(5)));
    }

    #[test]
    fn avg_of_float_values_is_a_float() {
        let mut avg = AggState::new(&AggType::AVG);
        avg.update(Some(Value::FLOAT(1.5.into()))).unwrap();
        avg.update(Some(Value::FLOAT(2.0.into()))).unwrap();
        assert_eq!(avg.value().unwrap(), Some(Value::FLOAT(1.75.into())));
    }

    #[test]
    fn sum_that_overflows_int_is_an_error() {
        let mut sum = AggState::new(&AggType::SUM);
        sum.update(Some(Value::INT(i32::MAX))).unwrap();
        assert!(sum.update(Some(Value::INT(1))).is_err());
    }

    #[test]
    fn avg_of_only_nulls_is_null() {
        let mut avg = AggState::new(&AggType::AVG);
        avg.update(None).unwrap();
        assert_eq!(avg.value().unwrap(), None);
    }

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

    #[test]
    fn next_before_open_is_an_error() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[1]]);
        let agg_plan = AggregationPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            group_bys: vec![],
            aggregates: vec![],
        };
        let mut agg = AggregationExecutor::new(&agg_plan, values_child(&rows));
        let err = agg.next().unwrap_err();
        assert!(err.0.contains("before open"), "{}", err.0);
    }
}
