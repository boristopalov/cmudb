use crate::catalog::index_schema::{IndexKey, IndexValue, TableIndex};
use crate::catalog::{Catalog, CatalogError, IndexInfo, Schema, TableInfo, Value};
use crate::index::IndexError;
use crate::processor::plan::{
    DeletePlanNode, IndexScanPlanNode, InsertPlanNode, SeqScanPlanNode, UpdatePlanNode,
    ValuesPlanNode,
};
use crate::table_heap::{RecordId, TableHeapError, TableHeapIterator, Tuple};
use std::sync::Arc;

#[derive(Debug)]
pub struct ExecutorError(pub String);

pub struct ExecutorContext<'a> {
    pub catalog: &'a Catalog,
}

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

/// Executors in the executor tree can be:
/// - leaves, i.e. no children executors
/// - unary, i.e. one child exectur
/// - binary, i.e. two children
/// in many cases, close() doesn't do anything; the executor simply going out of scope is enough for rust to clean things up for us
/// the next() method returns only 1 tuple at a time, with no batching, for simplicity.
pub trait Executor {
    fn open(&mut self) -> Result<(), ExecutorError>;
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError>;
    fn schema(&self) -> &Schema; // output schema
    fn close(&mut self) -> Result<(), ExecutorError>;
}

pub struct SeqScanExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a SeqScanPlanNode,
    iter: Option<TableHeapIterator>,
}

impl<'a> SeqScanExecutor<'a> {
    pub fn new(ctx: &'a ExecutorContext<'a>, plan: &'a SeqScanPlanNode) -> Self {
        Self {
            ctx,
            plan,
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

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| ExecutorError("SeqScanExecutor::next called before open".to_string()))?;
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
    ctx: &'a ExecutorContext<'a>,
    plan: &'a InsertPlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
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
            done: false,
        }
    }
}

impl<'a> Executor for InsertExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.done {
            return Err(ExecutorError("already executed".to_string()));
        }

        // Here we call child.next() in a loop, in case there are multiple values to insert
        // Recall that each child.next() call returns a single (tuple, record_id)
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| ExecutorError("InsertExecutor::next called before open".to_string()))?;

        let mut count: i32 = 0;
        while let Some(res) = self.child.next()? {
            let rid = table.heap.insert(&res.0)?;
            for idx in self.ctx.catalog.get_table_indexes(self.plan.table_oid)? {
                idx.as_ref().index.insert(&res.0, rid)?;
            }
            count += 1;
        }

        self.done = true;

        // This is a synthetic tuple, we need to tell our parent how many items we inserted
        Ok(Some((
            Tuple {
                data: count.to_ne_bytes().to_vec(),
            },
            RecordId::RESERVED,
        )))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Retrieves tuples, one by one, from a list of Vec<Expressions>.
/// corresponds to the VALUES sql keyword.
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

        // exprs represents a single tuple
        // each expr repsents the value in that particular column in the tuple at exprs.index_of(expr)
        let dummy = Tuple { data: vec![] };
        let vals: Vec<Value> = exprs
            .iter()
            .map(|e| e.evaluate(&dummy, self.schema()))
            .collect::<Result<Vec<Value>, ExecutorError>>()?;

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
    ctx: &'a ExecutorContext<'a>,
    plan: &'a UpdatePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> Executor for UpdateExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| ExecutorError("UpdateExecutor::next called before open".to_string()))?;

        let Some((tuple_to_update, rid)) = self.child.next()? else {
            return Ok(None);
        };

        // we need to evaluate each expression
        // target_exprs can have some or all columns
        // so we need to update the values in those columns, and keep the values of the old columns,
        // and then return the updated tuple
        // First, populate new_values with the old values
        let mut new_values: Vec<Value> = Vec::with_capacity(self.schema().cols.len());
        for i in 0..self.schema().cols.len() {
            new_values.push(self.schema().get_value(&tuple_to_update, i)?);
        }

        // update new_values
        for expr in self.plan.target_exprs.iter() {
            let col_to_update_idx = expr.0;
            let new_val = expr.1.evaluate(&tuple_to_update, self.schema())?;
            new_values[col_to_update_idx as usize] = new_val;
        }

        let new_tuple = self.schema().encode_tuple(new_values.as_ref())?;
        let (new_rid, _) = table.heap.update(rid, &new_tuple)?;

        // to update the index we remove first, and then re-insert
        // not ideal :/
        for idx in self.ctx.catalog.get_table_indexes(self.plan.table_oid)? {
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
    ctx: &'a ExecutorContext<'a>,
    plan: &'a DeletePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> Executor for DeleteExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| ExecutorError("DeleteExecutor::next called before open".to_string()))?;

        let Some((tup, rid)) = self.child.next()? else {
            return Ok(None);
        };

        table.heap.delete(&rid)?;
        for idx in self.ctx.catalog.get_table_indexes(self.plan.table_oid)? {
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
    ctx: &'a ExecutorContext<'a>,
    plan: &'a IndexScanPlanNode,
    table: Option<Arc<TableInfo>>,
    iter: Option<Box<dyn Iterator<Item = Result<(IndexKey, IndexValue), IndexError>>>>,
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

        Ok(())
    }
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self.iter.as_mut().ok_or_else(|| {
            ExecutorError("ndexScanExecutor::next called before open".to_string())
        })?;
        let table = self.table.as_ref().ok_or_else(|| {
            ExecutorError("IndexScanExecutor::next called before open".to_string())
        })?;

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
