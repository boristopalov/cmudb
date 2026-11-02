use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Catalog, CatalogError, DataType, Schema, Value};
use crate::concurrency::mvcc::MvccError;
use crate::concurrency::{Transaction, TransactionManager};
use crate::disk::PAGE_SIZE;
use crate::index::IndexError;
use crate::processor::plan::{Arithmetic, Plan};
use crate::table_heap::{RecordId, TableHeapError, Tuple};
use std::fmt;
use std::sync::Arc;
use std::vec::IntoIter as VecIter;

/// Why an executor failed.
#[derive(Debug)]
pub enum ExecutorError {
    /// A catalog lookup, or encoding or decoding a tuple, failed.
    Catalog(CatalogError),
    /// Reading or writing a table heap failed.
    TableHeap(TableHeapError),
    /// Reading or writing an index failed.
    Index(IndexError),
    /// A versioned read or write was rejected, such as by a write-write conflict.
    Mvcc(MvccError),
    /// A column of an input row could not be decoded.
    ColumnRead { col: u32, source: CatalogError },
    /// `next_batch` ran before `open`.
    NotOpened { executor: &'static str },
    /// AND, OR or a filter got an operand other than a BOOLEAN or NULL.
    NotBoolean { value: Value },
    /// A comparison between values of types that do not compare.
    Incomparable { left: Value, right: Value },
    /// Arithmetic on a value that is not an INT, BIGINT or FLOAT.
    ArithmeticOperands {
        op: Arithmetic,
        left: Value,
        right: Value,
    },
    /// INT or BIGINT division by zero.
    DivisionByZero { dividend: i64 },
    /// INT or BIGINT arithmetic whose result does not fit the operands' type.
    IntegerOverflow {
        op: Arithmetic,
        left: i64,
        right: i64,
    },
    /// An INSERT wrote more rows than its INT count column holds.
    CountOverflow { count: u64 },
    /// An INSERT executor yielded other than one count row.
    CountRows { rows: usize },
    /// An INSERT's count row holds something other than a non-negative INT.
    BadCount { value: Option<Value> },
    /// `rows_affected` was given a plan that is not INSERT, UPDATE or DELETE.
    NotDml,
    /// An external sort ended with no sorted run.
    NoSortRuns,
    /// DROP TABLE without IF EXISTS named a table that does not exist, by oid when the plan
    /// had one.
    TableNotFound { oid: Option<u32> },
    /// DROP INDEX without IF EXISTS named an index that does not exist, by oid when the plan
    /// had one.
    IndexNotFound { oid: Option<u32> },
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutorError::Catalog(e) => write!(f, "catalog error: {e}"),
            ExecutorError::TableHeap(e) => write!(f, "table heap error: {e}"),
            ExecutorError::Index(e) => write!(f, "index error: {e}"),
            ExecutorError::Mvcc(e) => write!(f, "{e}"),
            ExecutorError::ColumnRead { col, source } => {
                write!(f, "cannot read column {col}: {source}")
            }
            ExecutorError::NotOpened { executor } => {
                write!(f, "{executor}::next_batch called before open")
            }
            ExecutorError::NotBoolean { value } => {
                write!(f, "expected a BOOLEAN operand, got {value:?}")
            }
            ExecutorError::Incomparable { left, right } => {
                write!(f, "cannot compare {left:?} with {right:?}")
            }
            ExecutorError::ArithmeticOperands { op, left, right } => {
                write!(f, "cannot apply {op:?} to {left:?} and {right:?}")
            }
            ExecutorError::DivisionByZero { dividend } => {
                write!(f, "division by zero: {dividend} / 0")
            }
            ExecutorError::IntegerOverflow { op, left, right } => {
                write!(f, "integer overflow: {op:?} of {left} and {right}")
            }
            ExecutorError::CountOverflow { count } => {
                write!(
                    f,
                    "inserted {count} rows, more than the count column can hold"
                )
            }
            ExecutorError::CountRows { rows } => {
                write!(f, "INSERT yielded {rows} rows instead of one count row")
            }
            ExecutorError::BadCount { value } => write!(
                f,
                "INSERT count row holds {value:?} instead of a non-negative INT"
            ),
            ExecutorError::NotDml => {
                f.write_str("rows_affected called on a plan that is not INSERT, UPDATE or DELETE")
            }
            ExecutorError::NoSortRuns => f.write_str("external sort produced no runs"),
            ExecutorError::TableNotFound { oid } => missing_drop_target(f, "table", *oid),
            ExecutorError::IndexNotFound { oid } => missing_drop_target(f, "index", *oid),
        }
    }
}

impl std::error::Error for ExecutorError {}

impl From<CatalogError> for ExecutorError {
    fn from(e: CatalogError) -> Self {
        ExecutorError::Catalog(e)
    }
}

impl From<TableHeapError> for ExecutorError {
    fn from(e: TableHeapError) -> Self {
        ExecutorError::TableHeap(e)
    }
}

impl From<IndexError> for ExecutorError {
    fn from(e: IndexError) -> Self {
        ExecutorError::Index(e)
    }
}

impl From<MvccError> for ExecutorError {
    fn from(e: MvccError) -> Self {
        ExecutorError::Mvcc(e)
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
    /// Tuple bytes an `ORDER BY` sorts in memory; past this it spills sorted runs to scratch
    /// pages in the buffer pool and merges them.
    pub sort_memory_budget_bytes: usize,
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
            sort_memory_budget_bytes: DEFAULT_SORT_MEMORY_BUDGET_BYTES,
        }
    }

    /// The same context with `ORDER BY` spilling once its input passes `bytes` of tuple data.
    pub fn with_sort_memory_budget(self, bytes: usize) -> Self {
        Self {
            sort_memory_budget_bytes: bytes,
            ..self
        }
    }
}

/// Default number of rows an executor aims to put in each [`Batch`].
pub const DEFAULT_BATCH_SIZE: usize = 1024;

/// Default for [`ExecutorContext::sort_memory_budget_bytes`].
pub const DEFAULT_SORT_MEMORY_BUDGET_BYTES: usize = 5_000 * PAGE_SIZE;

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
    /// Releases resources early and closes every child, carrying on past a failure and
    /// returning the first one; dropping the executor releases the rest.
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
                return Err(ExecutorError::CountRows { rows: rows.len() });
            };
            let value = exec.schema().get_value(count_row, 0)?;
            match value {
                Some(Value::INT(n)) => {
                    u64::try_from(n).map_err(|_| ExecutorError::BadCount { value })
                }
                _ => Err(ExecutorError::BadCount { value }),
            }
        }
        Plan::Update(_) | Plan::Delete(_) => {
            let mut n = 0u64;
            while let Some(batch) = exec.next_batch()? {
                n += batch.len() as u64;
            }
            Ok(n)
        }
        _ => Err(ExecutorError::NotDml),
    }
}

/// Writes the message for a DROP whose `kind` of object (table or index) does not exist.
fn missing_drop_target(f: &mut fmt::Formatter<'_>, kind: &str, oid: Option<u32>) -> fmt::Result {
    match oid {
        Some(oid) => write!(f, "{kind} with oid {oid} does not exist"),
        None => write!(f, "{kind} does not exist"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::InsertError;
    use crate::processor::executors::test_support::*;
    #[test]
    fn executor_errors_display_readable_messages() {
        let cases = [
            (
                ExecutorError::from(IndexError::Insert(InsertError::DuplicateKey)),
                "index error: insert failed: key already exists",
            ),
            (
                ExecutorError::from(TableHeapError::NewPageWithoutId),
                "table heap error: new page has no page_id",
            ),
            (
                ExecutorError::from(CatalogError::ColumnOutOfBounds),
                "catalog error: the column index is past the schema's last column",
            ),
            (
                ExecutorError::ColumnRead {
                    col: 3,
                    source: CatalogError::TupleSchemaMismatch,
                },
                "cannot read column 3: the tuple does not match the table's schema",
            ),
            (
                ExecutorError::NotOpened {
                    executor: "SeqScanExecutor",
                },
                "SeqScanExecutor::next_batch called before open",
            ),
            (
                ExecutorError::IntegerOverflow {
                    op: Arithmetic::Mul,
                    left: i32::MAX.into(),
                    right: 2,
                },
                "integer overflow: Mul of 2147483647 and 2",
            ),
            (
                ExecutorError::CountOverflow { count: 1 << 40 },
                "inserted 1099511627776 rows, more than the count column can hold",
            ),
            (
                ExecutorError::CountRows { rows: 2 },
                "INSERT yielded 2 rows instead of one count row",
            ),
            (
                ExecutorError::BadCount {
                    value: Some(Value::INT(-1)),
                },
                "INSERT count row holds Some(INT(-1)) instead of a non-negative INT",
            ),
            (ExecutorError::NoSortRuns, "external sort produced no runs"),
            (
                ExecutorError::TableNotFound { oid: Some(7) },
                "table with oid 7 does not exist",
            ),
            (
                ExecutorError::IndexNotFound { oid: Some(8) },
                "index with oid 8 does not exist",
            ),
            (
                ExecutorError::IndexNotFound { oid: None },
                "index does not exist",
            ),
        ];
        for (err, message) in cases {
            assert_eq!(err.to_string(), message, "{err:?}");
        }
    }

    #[test]
    fn rows_affected_rejects_a_plan_that_is_not_dml() {
        let plan = values_plan(&int_schema(&["a"]), &[&[1]]);
        let mut exec = values_child(&plan);
        exec.open().expect("open");

        let err = rows_affected(&plan, &mut *exec).unwrap_err();

        assert!(matches!(err, ExecutorError::NotDml), "{err:?}");
    }
}
