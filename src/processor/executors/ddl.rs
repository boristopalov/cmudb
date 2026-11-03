use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Catalog, CatalogError, DataType, IndexType, Schema, Value};
use crate::processor::executor::{Batch, Executor, ExecutorError};
use crate::processor::executors::single_row;
use crate::processor::plan::{
    CreateIndexPlanNode, CreateTablePlanNode, DropIndexPlanNode, DropTablePlanNode,
};
use std::sync::Arc;

/// The catalog change a [`DdlExecutor`] makes.
pub enum DdlOp<'a> {
    CreateTable(&'a CreateTablePlanNode),
    DropTable(&'a DropTablePlanNode),
    CreateIndex(&'a CreateIndexPlanNode),
    DropIndex(&'a DropIndexPlanNode),
}

/// Makes one catalog change on its first `next_batch`, then yields one row laid out by
/// [`ddl_status_schema`]: 1 when the catalog changed, 0 when IF EXISTS or IF NOT EXISTS
/// left it as it was.
pub struct DdlExecutor<'a> {
    catalog: &'a Catalog,
    bpm: Arc<BufferPoolManager>,
    op: DdlOp<'a>,
    status_schema: Schema,
    done: bool,
}

/// The layout of a DDL statement's status row: one INT column named `status`.
fn ddl_status_schema() -> Schema {
    Schema::new(vec![(DataType::INT, "status".to_string())])
}

impl<'a> DdlExecutor<'a> {
    pub fn new(catalog: &'a Catalog, bpm: Arc<BufferPoolManager>, op: DdlOp<'a>) -> Self {
        Self {
            catalog,
            bpm,
            op,
            status_schema: ddl_status_schema(),
            done: false,
        }
    }

    /// Applies the change and returns whether the catalog changed.
    fn apply(&self) -> Result<bool, ExecutorError> {
        match self.op {
            DdlOp::CreateTable(plan) => {
                match self
                    .catalog
                    .create_table(plan.table_name.clone(), plan.schema.clone())
                {
                    Ok(_oid) => Ok(true),
                    Err(CatalogError::TableExists) if plan.if_not_exists => Ok(false),
                    Err(e) => Err(e.into()),
                }
            }
            DdlOp::DropTable(plan) => {
                let dropped = match plan.table_oid {
                    Some(oid) => self.catalog.drop_table(oid)?,
                    None => false,
                };
                if !dropped && !plan.if_exists {
                    return Err(ExecutorError::TableNotFound {
                        oid: plan.table_oid,
                    });
                }
                Ok(dropped)
            }
            DdlOp::CreateIndex(plan) => {
                self.create_index(plan)?;
                Ok(true)
            }
            DdlOp::DropIndex(plan) => {
                let dropped = match plan.index_oid {
                    Some(oid) => self.catalog.drop_index(oid)?,
                    None => false,
                };
                if !dropped && !plan.if_exists {
                    return Err(ExecutorError::IndexNotFound {
                        oid: plan.index_oid,
                    });
                }
                Ok(dropped)
            }
        }
    }

    fn create_index(&self, plan: &CreateIndexPlanNode) -> Result<(), ExecutorError> {
        let table = self.catalog.get_table(plan.table_oid)?;
        let col_names: Vec<String> = plan
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
            plan.table_oid,
            plan.index_name.clone(),
            &col_refs,
            IndexType::BPlusTree,
        )?;
        Ok(())
    }
}

impl<'a> Executor for DdlExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let changed = self.apply()?;
        let status = self
            .status_schema
            .encode_tuple(&[Some(Value::INT(i32::from(changed)))])?;
        Ok(single_row(status))
    }

    fn schema(&self) -> &Schema {
        &self.status_schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::IndexOid;
    use crate::processor::executor::collect_rows;
    use crate::processor::executors::test_support::*;
    use crate::processor::plan::ColIdx;

    fn status(
        catalog: &Catalog,
        bpm: &Arc<BufferPoolManager>,
        op: DdlOp,
    ) -> Result<i32, ExecutorError> {
        let mut exec = DdlExecutor::new(catalog, bpm.clone(), op);
        exec.open()?;
        let rows = collect_rows(&mut exec)?;
        let [(row, _)] = rows.as_slice() else {
            panic!("expected one status row, got {}", rows.len());
        };
        match exec.schema().get_value(row, 0)? {
            Some(Value::INT(n)) => Ok(n),
            other => panic!("expected an INT status, got {other:?}"),
        }
    }

    #[test]
    fn status_is_one_when_the_catalog_changes_and_zero_when_if_exists_skips_it() {
        let (catalog, bpm, _dir) = crate::testing::catalog(16);
        let create = CreateTablePlanNode {
            table_name: "t".to_string(),
            schema: int_schema(&["a"]),
            if_not_exists: true,
        };
        assert_eq!(
            status(&catalog, &bpm, DdlOp::CreateTable(&create)).unwrap(),
            1
        );
        assert_eq!(
            status(&catalog, &bpm, DdlOp::CreateTable(&create)).unwrap(),
            0
        );

        let table_oid = catalog.table_oid("t").unwrap();
        let index = CreateIndexPlanNode {
            index_name: "t_a".to_string(),
            table_oid,
            key_columns: vec![ColIdx(0)],
        };
        assert_eq!(
            status(&catalog, &bpm, DdlOp::CreateIndex(&index)).unwrap(),
            1
        );

        let index_oid = catalog.index_oid("t", "t_a").unwrap();
        let drop_index = DropIndexPlanNode {
            index_oid: Some(index_oid),
            if_exists: true,
        };
        assert_eq!(
            status(&catalog, &bpm, DdlOp::DropIndex(&drop_index)).unwrap(),
            1
        );
        assert_eq!(
            status(&catalog, &bpm, DdlOp::DropIndex(&drop_index)).unwrap(),
            0
        );

        let drop_table = DropTablePlanNode {
            table_oid: Some(table_oid),
            if_exists: true,
        };
        assert_eq!(
            status(&catalog, &bpm, DdlOp::DropTable(&drop_table)).unwrap(),
            1
        );
        assert_eq!(
            status(&catalog, &bpm, DdlOp::DropTable(&drop_table)).unwrap(),
            0
        );
    }

    #[test]
    fn dropping_a_missing_object_without_if_exists_is_an_error() {
        let (catalog, bpm, _dir) = crate::testing::catalog(16);
        let drop_table = DropTablePlanNode {
            table_oid: None,
            if_exists: false,
        };
        let err = status(&catalog, &bpm, DdlOp::DropTable(&drop_table)).unwrap_err();
        assert!(
            matches!(err, ExecutorError::TableNotFound { oid: None }),
            "{err:?}"
        );

        let drop_index = DropIndexPlanNode {
            index_oid: Some(IndexOid(42)),
            if_exists: false,
        };
        let err = status(&catalog, &bpm, DdlOp::DropIndex(&drop_index)).unwrap_err();
        assert!(
            matches!(
                err,
                ExecutorError::IndexNotFound {
                    oid: Some(IndexOid(42))
                }
            ),
            "{err:?}"
        );
    }
}
