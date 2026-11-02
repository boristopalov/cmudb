use crate::buffer_pool::BufferPoolManager;
use crate::catalog::index_schema::{TableIndex, TableIndexIter};
use crate::catalog::{
    Catalog, CatalogError, DataType, IndexInfo, IndexType, Schema, TableInfo, Value,
};
use crate::concurrency::mvcc::{self, MvccError};
use crate::concurrency::{Transaction, TransactionManager};
use crate::disk::PAGE_SIZE;
use crate::index::IndexError;
use crate::processor::plan::{
    AggType, AggregationPlanNode, Arithmetic, CreateIndexPlanNode, CreateTablePlanNode,
    DeletePlanNode, DropIndexPlanNode, DropTablePlanNode, Expression, ExternalMergeSortPlanNode,
    FilterPlanNode, HashJoinPlanNode, IndexScanPlanNode, InputRow, InsertPlanNode, JoinRow,
    JoinType, LimitPlanNode, NestedIndexJoinPlanNode, NestedLoopJoinPlanNode, NullOrdering, Plan,
    ProjectionPlanNode, SeqScanPlanNode, SortDirection, SortExpr, SortPlanNode, UpdatePlanNode,
    ValuesPlanNode, WindowFunctionPlanNode,
};
use crate::table_heap::{RecordId, TableHeap, TableHeapError, TableHeapIterator, Tuple};
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
        ExecutorError(format!("catalog error: {e}"))
    }
}

impl From<TableHeapError> for ExecutorError {
    fn from(e: TableHeapError) -> Self {
        ExecutorError(format!("table heap error: {e}"))
    }
}

impl From<IndexError> for ExecutorError {
    fn from(e: IndexError) -> Self {
        ExecutorError(format!("index error: {:?}", e))
    }
}

impl From<MvccError> for ExecutorError {
    fn from(e: MvccError) -> Self {
        ExecutorError(e.to_string())
    }
}

/// Everything an executor tree needs beyond its plan: the catalog, the buffer pool (for
/// spill files) and the transaction it runs in.
///
/// Executors only read and write; finishing the transaction is up to whoever created the
/// context. If any write fails the transaction is tainted and can only be aborted.
pub struct ExecutorContext<'a> {
    pub catalog: &'a Catalog,
    pub bpm: Arc<BufferPoolManager>,
    pub txn: Arc<Transaction>,
    pub txn_mgr: Arc<TransactionManager>,
}

impl<'a> ExecutorContext<'a> {
    pub fn new(
        catalog: &'a Catalog,
        bpm: Arc<BufferPoolManager>,
        txn: Arc<Transaction>,
        txn_mgr: Arc<TransactionManager>,
    ) -> Self {
        Self {
            catalog,
            bpm,
            txn,
            txn_mgr,
        }
    }
}

/// Default number of rows an executor aims to put in each [`Batch`].
pub const DEFAULT_BATCH_SIZE: usize = 1024;

/// A single row flowing through the executor tree, with the record id it came from, or
/// [`RecordId::RESERVED`] for rows that do not map back to a table slot.
pub type Row = (Tuple, RecordId);

/// A batch of rows passed between executors. Batching spreads the per-call cost of the
/// pull model (dynamic dispatch, `Option`/`Result` wrapping, state checks) over many rows.
#[derive(Debug, Default)]
pub struct Batch {
    pub rows: Vec<Row>,
}

impl Batch {
    /// Wraps `rows` in a batch, or returns `None` when `rows` is empty, so that an empty
    /// batch is not handed to a parent.
    pub fn from_rows(rows: Vec<Row>) -> Option<Self> {
        if rows.is_empty() {
            None
        } else {
            Some(Self { rows })
        }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Row> {
        self.rows.iter()
    }
}

impl IntoIterator for Batch {
    type Item = Row;
    type IntoIter = VecIter<Row>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.into_iter()
    }
}

/// A pull-based plan operator using a vectorized volcano model. `open` prepares state and
/// opens the children, then each `next_batch` call yields a batch of at least one row until
/// it returns `Ok(None)`, which it then returns on every later call. Executors that produce
/// rows themselves (scans, joins, sorts, aggregations) emit at most their configured batch
/// size; pass-through executors (filter, projection, limit, update, delete) emit at most what
/// their child handed them.
pub trait Executor {
    fn open(&mut self) -> Result<(), ExecutorError>;
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError>;
    /// Layout of the tuples `next_batch` yields.
    fn schema(&self) -> &Schema;
    /// Releases resources early; dropping the executor releases the rest.
    fn close(&mut self) -> Result<(), ExecutorError>;
}

/// Row-at-a-time view over an executor, for callers (the shell, tests) that consume rows one
/// by one. It pulls a batch at a time from the executor.
pub struct RowCursor<'e> {
    exec: &'e mut dyn Executor,
    buf: VecIter<Row>,
}

impl<'e> RowCursor<'e> {
    pub fn new(exec: &'e mut dyn Executor) -> Self {
        Self {
            exec,
            buf: Vec::new().into_iter(),
        }
    }

    pub fn next_row(&mut self) -> Result<Option<Row>, ExecutorError> {
        loop {
            if let Some(row) = self.buf.next() {
                return Ok(Some(row));
            }
            match self.exec.next_batch()? {
                Some(batch) => self.buf = batch.into_iter(),
                None => return Ok(None),
            }
        }
    }
}

impl Iterator for RowCursor<'_> {
    type Item = Result<Row, ExecutorError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_row().transpose()
    }
}

/// Drains an opened executor and returns all of its rows.
pub fn collect_rows(exec: &mut dyn Executor) -> Result<Vec<Row>, ExecutorError> {
    let mut rows = Vec::new();
    while let Some(batch) = exec.next_batch()? {
        rows.extend(batch);
    }
    Ok(rows)
}

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

fn not_opened(executor: &str) -> ExecutorError {
    ExecutorError(format!("{executor}::next_batch called before open"))
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
    ctx: &'a ExecutorContext<'a>,
    plan: &'a SeqScanPlanNode,
    batch_size: usize,
    iter: Option<TableHeapIterator>,
}

impl<'a> SeqScanExecutor<'a> {
    pub fn new(ctx: &'a ExecutorContext<'a>, plan: &'a SeqScanPlanNode, batch_size: usize) -> Self {
        Self {
            ctx,
            plan,
            batch_size,
            iter: None,
        }
    }
}

impl<'a> Executor for SeqScanExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.iter = Some(table.heap.iter());
        Ok(())
    }

    /// Emits the version of each tuple that this transaction's snapshot sees. The scan's end
    /// is fixed when it opens, so rows the transaction inserts while scanning (e.g.
    /// `INSERT INTO t SELECT * FROM t`) are not fed back into it.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("SeqScanExecutor"))?;
        let (txn, txn_mgr) = (&*self.ctx.txn, &*self.ctx.txn_mgr);
        let mut rows = Vec::with_capacity(self.batch_size);
        while rows.len() < self.batch_size {
            let next = iter.next_with(|rid, meta, bytes| {
                mvcc::visible_version(txn, txn_mgr, rid, meta, bytes)
                    .map(|data| (Tuple { data }, rid))
            });
            match next {
                Some(row) => rows.push(row?),
                None => break,
            }
        }
        Ok(Batch::from_rows(rows))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Inserts every row its child yields, then yields one row holding the inserted-row count,
/// laid out by [`rows_affected_schema`].
pub struct InsertExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a InsertPlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
    count_schema: Schema,
    done: bool,
}

impl<'a> InsertExecutor<'a> {
    pub fn new(
        ctx: &'a ExecutorContext<'a>,
        plan: &'a InsertPlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            ctx,
            plan,
            child,
            table: None,
            count_schema: rows_affected_schema(),
            done: false,
        }
    }
}

impl<'a> Executor for InsertExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.done {
            return Ok(None);
        }

        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("InsertExecutor"))?;

        let indexes = self.ctx.catalog.get_table_indexes(self.plan.table_oid)?;
        let mut count: u64 = 0;
        while let Some(batch) = self.child.next_batch()? {
            for (tuple, _) in batch {
                mvcc::insert_tuple(&self.ctx.txn, &table.heap, &indexes, &tuple)?;
                count += 1;
            }
        }

        self.done = true;
        let count = i32::try_from(count).map_err(|_| {
            ExecutorError(format!(
                "inserted {count} rows, more than the count column can hold"
            ))
        })?;
        let count_row = self.count_schema.encode_tuple(&[Some(Value::INT(count))])?;
        Ok(single_row(count_row))
    }

    /// The layout of the count row, [`rows_affected_schema`], not the table's.
    fn schema(&self) -> &Schema {
        &self.count_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// The schema of the row `InsertExecutor` yields: one `INT` column named `count`.
pub fn rows_affected_schema() -> Schema {
    Schema::new(vec![(DataType::INT, "count".to_string())])
}

/// Runs a DML executor to the end and returns how many rows it inserted, updated or deleted.
/// `plan` is the plan `exec` was built from.
///
/// Fails with an error for a plan that is not an INSERT, UPDATE or DELETE.
pub fn rows_affected(plan: &Plan, exec: &mut dyn Executor) -> Result<u64, ExecutorError> {
    match plan {
        Plan::Insert(_) => {
            let rows = collect_rows(exec)?;
            let [(count_row, _)] = rows.as_slice() else {
                return Err(ExecutorError(format!(
                    "INSERT yielded {} rows instead of one count row",
                    rows.len()
                )));
            };
            match exec.schema().get_value(count_row, 0)? {
                Some(Value::INT(n)) => u64::try_from(n)
                    .map_err(|_| ExecutorError(format!("INSERT reported a negative count {n}"))),
                other => Err(ExecutorError(format!(
                    "INSERT count row holds {other:?} instead of an INT"
                ))),
            }
        }
        Plan::Update(_) | Plan::Delete(_) => {
            let mut n = 0u64;
            while let Some(batch) = exec.next_batch()? {
                n += batch.len() as u64;
            }
            Ok(n)
        }
        _ => Err(ExecutorError(
            "rows_affected called on a plan that is not INSERT, UPDATE or DELETE".to_string(),
        )),
    }
}

/// Yields one tuple per row of a `VALUES` list, up to `batch_size` per batch.
pub struct ValuesExecutor<'a> {
    plan: &'a ValuesPlanNode,
    batch_size: usize,
    cursor: usize,
}

impl<'a> ValuesExecutor<'a> {
    pub fn new(plan: &'a ValuesPlanNode, batch_size: usize) -> Self {
        Self {
            plan,
            batch_size,
            cursor: 0,
        }
    }
}

impl<'a> Executor for ValuesExecutor<'a> {
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let start = self.cursor.min(self.plan.rows.len());
        let end = start
            .saturating_add(self.batch_size)
            .min(self.plan.rows.len());
        let no_input_row = Tuple { data: vec![] };
        let no_input_columns = Schema::new(vec![]);
        let mut rows = Vec::with_capacity(end - start);
        for exprs in &self.plan.rows[start..end] {
            let vals: Vec<Option<Value>> = exprs
                .iter()
                .map(|e| e.evaluate(&no_input_row, &no_input_columns))
                .collect::<Result<Vec<Option<Value>>, ExecutorError>>()?;
            rows.push((self.plan.schema.encode_tuple(&vals)?, RecordId::RESERVED));
        }
        self.cursor = end;
        Ok(Batch::from_rows(rows))
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
    ctx: &'a ExecutorContext<'a>,
    plan: &'a UpdatePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> UpdateExecutor<'a> {
    pub fn new(
        ctx: &'a ExecutorContext<'a>,
        plan: &'a UpdatePlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            ctx,
            plan,
            child,
            table: None,
        }
    }
}

impl<'a> Executor for UpdateExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    /// Updates every row of the next child batch in place and returns the updated rows.
    /// Each row keeps its record id, and the version it replaces moves to the undo chain.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("UpdateExecutor"))?;

        let Some(batch) = self.child.next_batch()? else {
            return Ok(None);
        };

        let schema = &self.plan.schema;
        let indexes = self.ctx.catalog.get_table_indexes(self.plan.table_oid)?;
        let mut out = Vec::with_capacity(batch.len());
        for (tuple_to_update, rid) in batch {
            let mut new_values = row_values(schema, &tuple_to_update)?;
            for assignment in &self.plan.assignments {
                new_values[assignment.col_idx.index()] =
                    assignment.value.evaluate(&tuple_to_update, schema)?;
            }

            let new_tuple = schema.encode_tuple(new_values.as_ref())?;
            mvcc::update_tuple(
                &self.ctx.txn,
                &self.ctx.txn_mgr,
                &table.heap,
                &indexes,
                rid,
                &tuple_to_update,
                &new_tuple,
            )?;
            out.push((new_tuple, rid));
        }
        Ok(Batch::from_rows(out))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct DeleteExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a DeletePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> DeleteExecutor<'a> {
    pub fn new(
        ctx: &'a ExecutorContext<'a>,
        plan: &'a DeletePlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            ctx,
            plan,
            child,
            table: None,
        }
    }
}

impl<'a> Executor for DeleteExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    /// Deletes every row of the next child batch and returns the deleted rows.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("DeleteExecutor"))?;

        let Some(batch) = self.child.next_batch()? else {
            return Ok(None);
        };

        for (_, rid) in batch.iter() {
            mvcc::delete_tuple(&self.ctx.txn, &self.ctx.txn_mgr, &table.heap, *rid)?;
        }
        Ok(Some(batch))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct IndexScanExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a IndexScanPlanNode,
    batch_size: usize,
    table: Option<Arc<TableInfo>>,
    index: Option<Arc<IndexInfo>>,
    iter: Option<TableIndexIter>,
}

impl<'a> IndexScanExecutor<'a> {
    pub fn new(
        ctx: &'a ExecutorContext<'a>,
        plan: &'a IndexScanPlanNode,
        batch_size: usize,
    ) -> Self {
        Self {
            ctx,
            plan,
            batch_size,
            table: None,
            index: None,
            iter: None,
        }
    }
}

impl<'a> Executor for IndexScanExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let info = self.ctx.catalog.get_index(self.plan.index_oid)?;
        let iter = info.as_ref().index.scan(
            self.plan.start.as_ref().map(Vec::as_slice),
            self.plan.end.as_ref().map(Vec::as_slice),
        )?;
        self.iter = Some(iter);

        let table = self.ctx.catalog.get_table(info.as_ref().table_oid)?;
        self.table = Some(table);
        self.index = Some(info);

        Ok(())
    }
    /// Index entries can be stale, so a row is emitted only if the version this transaction
    /// sees still has the key it was found under.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("IndexScanExecutor"))?;
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| not_opened("IndexScanExecutor"))?;

        let index = self.index.as_ref().ok_or_else(|| {
            ExecutorError("IndexScanExecutor::next_batch called before open".to_string())
        })?;

        let mut rows = Vec::with_capacity(self.batch_size);
        while rows.len() < self.batch_size {
            let Some(item) = iter.next() else {
                break;
            };
            let (key, rid) = item?;
            if let Some(tuple) = mvcc::read_visible_through_index(
                &self.ctx.txn,
                &self.ctx.txn_mgr,
                &table.heap,
                &index.index,
                &key,
                rid,
            )? {
                rows.push((tuple, rid));
            }
        }
        Ok(Batch::from_rows(rows))
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
    batch_size: usize,
    iter: Option<IntoIter<ExprKey, AggVal>>,
}

impl<'a> AggregationExecutor<'a> {
    pub fn new(
        plan: &'a AggregationPlanNode,
        child: Box<dyn Executor + 'a>,
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            child,
            batch_size,
            iter: None,
        }
    }
}

impl<'a> Executor for AggregationExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let mut agg_hash_table: HashMap<ExprKey, AggVal> = HashMap::new();
        while let Some(batch) = self.child.next_batch()? {
            let schema = self.child.schema();
            for (tuple, _) in batch {
                let groups = eval_key(&self.plan.group_bys, &tuple, schema)?;
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
        }
        self.iter = Some(agg_hash_table.into_iter());
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("AggregationExecutor"))?;
        let mut rows = Vec::with_capacity(self.batch_size);
        for (key, val) in iter.by_ref().take(self.batch_size) {
            let mut out: Vec<Option<Value>> = Vec::with_capacity(key.0.len() + val.0.len());
            out.extend(key.0);
            for v in val.0 {
                out.push(v.value()?);
            }
            rows.push((self.plan.schema.encode_tuple(&out)?, RecordId::RESERVED));
        }
        Ok(Batch::from_rows(rows))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

#[derive(Eq, PartialEq, Hash, PartialOrd, Ord)]
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
        Ok(())
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
        Ok(())
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
/// over the left partition. A NULL key matches nothing.
pub struct HashJoinExecutor<'a> {
    plan: &'a HashJoinPlanNode,
    left: Box<dyn Executor + 'a>,
    right: Box<dyn Executor + 'a>,
    bpm: Arc<BufferPoolManager>,
    batch_size: usize,
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
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            left,
            right,
            bpm,
            batch_size,
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

            if self.partition_cursor >= self.left_partitions.len() {
                break;
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
        Ok(Batch::from_rows(out))
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

pub struct ExternalMergeSortExecutor<'a> {
    plan: &'a ExternalMergeSortPlanNode,
    child: Box<dyn Executor + 'a>,
    bpm: Arc<BufferPoolManager>,
    batch_size: usize,
    iter: Option<TableHeapIterator>,
}

impl<'a> ExternalMergeSortExecutor<'a> {
    pub fn new(
        plan: &'a ExternalMergeSortPlanNode,
        child: Box<dyn Executor + 'a>,
        bpm: Arc<BufferPoolManager>,
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            child,
            bpm,
            batch_size,
            iter: None,
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
fn sorted_by_exprs<T>(
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

/// Input bytes the external merge sort sorts in memory before spilling them as one run.
const SORT_RUN_BUDGET_BYTES: usize = 5_000 * PAGE_SIZE;

impl<'a> ExternalMergeSortExecutor<'a> {
    fn spill_sorted_run(&self, rows: &mut Vec<Tuple>) -> Result<TableHeap, ExecutorError> {
        let (order_by, schema) = (&self.plan.order_by_exprs, &self.plan.schema);
        let sorted = sorted_by_exprs(std::mem::take(rows), |t| t, order_by, schema)?;
        let run = TableHeap::new(self.bpm.clone())?;
        for t in sorted {
            run.insert(&t)?;
        }
        Ok(run)
    }

    fn merge_runs(&self, a: &TableHeap, b: &TableHeap) -> Result<TableHeap, ExecutorError> {
        let (order_by, schema) = (&self.plan.order_by_exprs, &self.plan.schema);
        let out = TableHeap::new(self.bpm.clone())?;
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
}

impl<'a> Executor for ExternalMergeSortExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let mut run_bytes: usize = 0;
        let mut buf: Vec<Tuple> = Vec::new();
        let mut runs: Vec<TableHeap> = Vec::new();

        while let Some(batch) = self.child.next_batch()? {
            for (tup, _) in batch {
                run_bytes += tup.data.len();
                buf.push(tup);
                if run_bytes >= SORT_RUN_BUDGET_BYTES {
                    runs.push(self.spill_sorted_run(&mut buf)?);
                    run_bytes = 0;
                }
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

    /// Yields the sorted rows with [`RecordId::RESERVED`], since their record ids point into
    /// a scratch heap.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("ExternalMergeSortExecutor"))?;
        let mut rows = iter
            .by_ref()
            .map(|item| item.map(|(tup, _)| (tup, RecordId::RESERVED)));
        fill_batch(&mut rows, self.batch_size)
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
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.count >= self.plan.limit {
            return Ok(None);
        }
        let Some(mut batch) = self.child.next_batch()? else {
            return Ok(None);
        };
        let remaining = (self.plan.limit - self.count) as usize;
        batch.rows.truncate(remaining);
        self.count += batch.len() as u32;
        Ok(Some(batch))
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
    batch_size: usize,
    iter: Option<VecIter<Tuple>>,
}

impl<'a> WindowFunctionExecutor<'a> {
    pub fn new(
        plan: &'a WindowFunctionPlanNode,
        child: Box<dyn Executor + 'a>,
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            child,
            batch_size,
            iter: None,
        }
    }
}

impl<'a> Executor for WindowFunctionExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let tups: Vec<Tuple> = collect_rows(self.child.as_mut())?
            .into_iter()
            .map(|(t, _)| t)
            .collect();

        let child_schema = self.child.schema();
        let partition_by: &[Expression] = self.plan.partition_by_exprs.as_deref().unwrap_or(&[]);
        let order_by: &[Expression] = self.plan.order_by_exprs.as_deref().unwrap_or(&[]);
        let sort_exprs: Vec<SortExpr> = partition_by
            .iter()
            .chain(order_by)
            .cloned()
            .map(SortExpr::ascending)
            .collect();
        let tups = sorted_by_exprs(tups, |t| t, &sort_exprs, child_schema)?;

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
                    self.plan,
                    &mut peer_buffer,
                    &mut partition_aggregates,
                    child_schema,
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
                self.plan,
                &mut peer_buffer,
                &mut partition_aggregates,
                child_schema,
                &mut output_tuples,
            )?;
        }

        self.iter = Some(output_tuples.into_iter());
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("WindowFunctionExecutor"))?;
        let mut rows = iter.by_ref().map(|tup| (tup, RecordId::RESERVED));
        Ok(take_batch(&mut rows, self.batch_size))
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

    /// Filters child batches, pulling until at least one row passes, so that an empty batch
    /// is not handed up. A NULL predicate rejects the row.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        while let Some(batch) = self.child.next_batch()? {
            let child_schema = self.child.schema();
            let mut rows = Vec::with_capacity(batch.len());
            for (tup, rid) in batch {
                if is_true(self.plan.predicate.evaluate(&tup, child_schema)?) {
                    rows.push((tup, rid));
                }
            }
            if let Some(out) = Batch::from_rows(rows) {
                return Ok(Some(out));
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

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let Some(batch) = self.child.next_batch()? else {
            return Ok(None);
        };
        let child_schema = self.child.schema();
        let mut rows = Vec::with_capacity(batch.len());
        for (tup, rid) in batch {
            let vals: Vec<Option<Value>> = self
                .plan
                .expressions
                .iter()
                .map(|e| e.evaluate(&tup, child_schema))
                .collect::<Result<_, _>>()?;
            rows.push((self.plan.schema.encode_tuple(&vals)?, rid));
        }
        Ok(Batch::from_rows(rows))
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

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        match self
            .catalog
            .create_table(self.plan.table_name.clone(), self.plan.schema.clone())
        {
            Ok(_oid) => {
                let tup = self.status_schema.encode_tuple(&[Some(Value::INT(1))])?;
                Ok(single_row(tup))
            }
            Err(CatalogError::TableExists) if self.plan.if_not_exists => {
                let tup = self.status_schema.encode_tuple(&[Some(Value::INT(0))])?;
                Ok(single_row(tup))
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

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.done {
            return Ok(None);
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
        Ok(single_row(tup))
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

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.done {
            return Ok(None);
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
                    .get(idx.index())
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
        Ok(single_row(tup))
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

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.done {
            return Ok(None);
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
        Ok(single_row(tup))
    }

    fn schema(&self) -> &Schema {
        &self.status_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Folds every buffered peer row into `aggregates`, then emits each of those rows as the
/// plan's `columns` followed by the aggregates' current values, draining `buffer`.
fn flush_peer_group(
    plan: &WindowFunctionPlanNode,
    buffer: &mut Vec<Tuple>,
    aggregates: &mut [AggState],
    child_schema: &Schema,
    output: &mut Vec<Tuple>,
) -> Result<(), ExecutorError> {
    for buf_tup in buffer.iter() {
        for (state, (expr, _)) in aggregates.iter_mut().zip(plan.aggregates.iter()) {
            state.update(expr.evaluate(buf_tup, child_schema)?)?;
        }
    }
    for buf_tup in buffer.drain(..) {
        let mut combined = plan
            .columns
            .iter()
            .map(|e| e.evaluate(&buf_tup, child_schema))
            .collect::<Result<Vec<_>, _>>()?;
        for agg in aggregates.iter() {
            combined.push(agg.value()?);
        }
        output.push(plan.schema.encode_tuple(&combined)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::plan::{ColIdx, Plan};

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
            col: ColIdx(col_idx),
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
        Box::new(ValuesExecutor::new(values, DEFAULT_BATCH_SIZE))
    }

    fn drain(exec: &mut dyn Executor) -> Vec<Tuple> {
        exec.open().expect("open");
        collect_rows(exec)
            .expect("next_batch")
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }

    #[test]
    fn rows_affected_rejects_a_plan_that_is_not_dml() {
        let plan = values_plan(&int_schema(&["a"]), &[&[1]]);
        let mut exec = values_child(&plan);
        exec.open().expect("open");

        let err = rows_affected(&plan, &mut *exec).unwrap_err();

        assert!(err.0.contains("not INSERT, UPDATE or DELETE"), "{}", err.0);
    }

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

    fn nullable_int_rows(schema: &Schema, rows: &[&[Option<i32>]]) -> Vec<Tuple> {
        rows.iter()
            .map(|r| {
                let vals: Vec<Option<Value>> = r.iter().map(|v| v.map(Value::INT)).collect();
                schema.encode_tuple(&vals).expect("encode")
            })
            .collect()
    }

    fn decode_nullable_ints(
        schema: &Schema,
        tuples: impl IntoIterator<Item = Tuple>,
    ) -> Vec<Option<i32>> {
        tuples
            .into_iter()
            .map(|t| match row_values(schema, &t).expect("decode")[..] {
                [Some(Value::INT(i))] => Some(i),
                [None] => None,
                ref other => panic!("expected one INT or NULL, got {other:?}"),
            })
            .collect()
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

    fn divide_by_zero(col_idx: u32) -> Expression {
        Expression::Binary {
            left: Box::new(col(col_idx)),
            right: Box::new(Expression::Constant(Some(Value::INT(0)))),
            op: crate::processor::plan::Op::Div,
        }
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
        assert!(err.0.contains("division by zero"), "{}", err.0);
    }

    #[test]
    fn group_by_evaluation_errors_are_reported_not_grouped_as_null() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[2], &[1]]);
        let plan = AggregationPlanNode {
            schema: int_schema(&["g"]),
            child: Box::new(values_plan(&schema, &[])),
            group_bys: vec![divide_by_zero(0)],
            aggregates: vec![],
        };
        let mut agg = AggregationExecutor::new(&plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        let err = agg.open().unwrap_err();
        assert!(err.0.contains("division by zero"), "{}", err.0);
    }

    #[test]
    fn window_output_is_its_columns_followed_by_its_aggregates() {
        let schema = int_schema(&["dept", "amount"]);
        let rows = values_plan(&schema, &[&[1, 10], &[1, 30], &[2, 5]]);
        let doubled_amount = Expression::Binary {
            left: Box::new(col(1)),
            right: Box::new(Expression::Constant(Some(Value::INT(2)))),
            op: crate::processor::plan::Op::Mul,
        };
        let out_schema = int_schema(&["doubled", "dept_total"]);
        let plan = WindowFunctionPlanNode {
            schema: out_schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: None,
            partition_by_exprs: Some(vec![col(0)]),
            aggregates: vec![(col(1), AggType::SUM)],
            columns: vec![doubled_amount],
        };
        let mut window =
            WindowFunctionExecutor::new(&plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        assert_eq!(
            decode_ints(&out_schema, drain(&mut window)),
            vec![vec![20, 40], vec![60, 40], vec![10, 5]]
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
        assert!(exec.next_batch().unwrap().is_none());
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
        let mut agg = AggregationExecutor::new(&agg_plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        let err = agg.next_batch().unwrap_err();
        assert!(err.0.contains("before open"), "{}", err.0);
    }
}
