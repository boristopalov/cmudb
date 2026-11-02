use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Catalog, CatalogError, DataType, IndexType, Schema, Value};
use crate::processor::executor::{Batch, Executor, ExecutorError};
use crate::processor::executors::single_row;
use crate::processor::plan::{
    CreateIndexPlanNode, CreateTablePlanNode, DropIndexPlanNode, DropTablePlanNode,
};
use std::sync::Arc;

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
        let dropped = match self.plan.table_oid {
            Some(oid) => self.catalog.drop_table(oid)?,
            None => false,
        };
        if !dropped && !self.plan.if_exists {
            return Err(ExecutorError::TableNotFound {
                oid: self.plan.table_oid,
            });
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
        let dropped = match self.plan.index_oid {
            Some(oid) => self.catalog.drop_index(oid)?,
            None => false,
        };
        if !dropped && !self.plan.if_exists {
            return Err(ExecutorError::IndexNotFound {
                oid: self.plan.index_oid,
            });
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
