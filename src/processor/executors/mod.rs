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
pub use ddl::{CreateIndexExecutor, CreateTableExecutor, DropIndexExecutor, DropTableExecutor};
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
    use crate::processor::executors::test_support::*;

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
}
