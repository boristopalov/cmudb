use crate::catalog::index_schema::TableIndexIter;
use crate::catalog::{IndexInfo, Schema, TableInfo, Value};
use crate::concurrency::mvcc;
use crate::processor::executor::{Batch, Executor, ExecutorContext, ExecutorError};
use crate::processor::executors::not_opened;
use crate::processor::plan::{IndexScanPlanNode, SeqScanPlanNode, ValuesPlanNode};
use crate::table_heap::{TableHeapIterator, Tuple};
use std::sync::Arc;

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
                    .map(|data| (Tuple { data }, Some(rid)))
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
            rows.push((self.plan.schema.encode_tuple(&vals)?, None));
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

        let index = self
            .index
            .as_ref()
            .ok_or_else(|| not_opened("IndexScanExecutor"))?;

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
                rows.push((tuple, Some(rid)));
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
