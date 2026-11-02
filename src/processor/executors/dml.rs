use crate::catalog::{Schema, TableInfo, Value};
use crate::concurrency::mvcc;
use crate::processor::executor::{
    Batch, Executor, ExecutorContext, ExecutorError, rows_affected_schema,
};
use crate::processor::executors::{not_opened, row_values, single_row};
use crate::processor::plan::{DeletePlanNode, InsertPlanNode, UpdatePlanNode};
use std::sync::Arc;

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
        let count = i32::try_from(count).map_err(|_| ExecutorError::CountOverflow { count })?;
        let count_row = self.count_schema.encode_tuple(&[Some(Value::INT(count))])?;
        Ok(single_row(count_row))
    }

    /// The layout of the count row, [`rows_affected_schema`], not the table's.
    fn schema(&self) -> &Schema {
        &self.count_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
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
            let rid = rid.ok_or(ExecutorError::NoRecordId {
                executor: "UpdateExecutor",
            })?;
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
            out.push((new_tuple, Some(rid)));
        }
        Ok(Batch::from_rows(out))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
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
            let rid = rid.ok_or(ExecutorError::NoRecordId {
                executor: "DeleteExecutor",
            })?;
            mvcc::delete_tuple(&self.ctx.txn, &self.ctx.txn_mgr, &table.heap, rid)?;
        }
        Ok(Some(batch))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concurrency::{IsolationLevel, TransactionManager};
    use crate::processor::executors::test_support::*;
    use crate::processor::plan::{ColIdx, ColumnAssignment};

    #[test]
    fn update_and_delete_refuse_input_rows_with_no_record_id() {
        let (catalog, bpm, _dir) = crate::testing::catalog(16);
        let schema = int_schema(&["a"]);
        let table_oid = catalog
            .create_table("t".to_string(), schema.clone())
            .expect("create table");
        let txn_mgr = Arc::new(TransactionManager::new(bpm.clone()));
        let ctx = ExecutorContext::new(
            &catalog,
            bpm,
            txn_mgr.begin(IsolationLevel::SnapshotIsolation),
            txn_mgr.clone(),
        );
        let rows = values_plan(&schema, &[&[1]]);

        let update = UpdatePlanNode {
            schema: schema.clone(),
            assignments: vec![ColumnAssignment {
                col_idx: ColIdx(0),
                value: col(0),
            }],
            child: Box::new(values_plan(&schema, &[])),
            table_oid,
        };
        let mut exec = UpdateExecutor::new(&ctx, &update, values_child(&rows));
        exec.open().expect("open");
        let err = exec.next_batch().unwrap_err();
        assert!(
            matches!(
                err,
                ExecutorError::NoRecordId {
                    executor: "UpdateExecutor"
                }
            ),
            "{err:?}"
        );

        let delete = DeletePlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            table_oid,
        };
        let mut exec = DeleteExecutor::new(&ctx, &delete, values_child(&rows));
        exec.open().expect("open");
        let err = exec.next_batch().unwrap_err();
        assert!(
            matches!(
                err,
                ExecutorError::NoRecordId {
                    executor: "DeleteExecutor"
                }
            ),
            "{err:?}"
        );
    }
}
