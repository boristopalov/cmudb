use crate::processor::executor::{
    Executor, ExecutorContext, InsertExecutor, SeqScanExecutor, ValuesExecutor,
};
use crate::processor::plan::Plan;

pub fn create_executor<'a>(ctx: &'a ExecutorContext, plan: &'a Plan) -> Box<dyn Executor + 'a> {
    match plan {
        Plan::SeqScan(p) => Box::new(SeqScanExecutor::new(ctx, p)),
        Plan::Insert(p) => {
            let child = create_executor(ctx, &p.child);
            Box::new(InsertExecutor::new(ctx, p, child))
        }
        Plan::Values(p) => Box::new(ValuesExecutor::new(p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Catalog, DataType, Schema, Value};
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::processor::plan::{Expression, InsertPlanNode, SeqScanPlanNode, ValuesPlanNode};
    use crate::replacer::ArcReplacer;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_catalog() -> (Catalog, tempfile::TempDir) {
        let num_frames = 16;
        let dir = tempdir().expect("tempdir");
        let dm = DiskManager::new(&dir.path().join("db"), &dir.path().join("log"))
            .expect("disk manager");
        let scheduler = DiskScheduler::new(dm);
        let replacer = ArcReplacer::new(num_frames);
        let bpm = Arc::new(create_buffer_pool_manager(num_frames, replacer, scheduler));
        (Catalog::new(bpm), dir)
    }

    fn make_schema() -> Schema {
        Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::BOOLEAN, "active".to_string()),
        ])
    }

    #[test]
    fn insert_then_seq_scan_returns_inserted_rows() {
        let (mut catalog, _dir) = make_catalog();
        let table_oid = catalog
            .create_table("t".to_string(), make_schema())
            .expect("create table");

        let expected: Vec<Vec<Option<Value>>> = vec![
            vec![Some(Value::INT(1)), Some(Value::BOOLEAN(true))],
            vec![Some(Value::INT(2)), Some(Value::BOOLEAN(false))],
            vec![Some(Value::INT(42)), Some(Value::BOOLEAN(true))],
        ];

        let rows: Vec<Vec<Expression>> = expected
            .iter()
            .map(|row| row.iter().cloned().map(Expression::Constant).collect())
            .collect();

        let insert_plan = Plan::Insert(InsertPlanNode {
            schema: make_schema(),
            child: Box::new(Plan::Values(ValuesPlanNode {
                schema: make_schema(),
                rows,
            })),
            table_oid,
        });

        let ctx = ExecutorContext { catalog: &catalog };

        let mut insert_exec = create_executor(&ctx, &insert_plan);
        insert_exec.open().expect("open insert");
        let (count_tuple, _) = insert_exec
            .next()
            .expect("insert next")
            .expect("insert produces a count tuple");
        let count = i32::from_ne_bytes(count_tuple.data.as_slice().try_into().unwrap());
        assert_eq!(count, 3, "insert should report 3 rows");
        drop(insert_exec);

        let scan_plan = Plan::SeqScan(SeqScanPlanNode {
            table_oid,
            schema: make_schema(),
        });
        let mut scan_exec = create_executor(&ctx, &scan_plan);
        scan_exec.open().expect("open scan");

        let mut got = Vec::new();
        while let Some((tuple, _rid)) = scan_exec.next().expect("scan next") {
            let row: Vec<Option<Value>> = (0..make_schema().cols.len())
                .map(|i| make_schema().get_value(&tuple, i).expect("decode"))
                .collect();
            got.push(row);
        }

        assert_eq!(got.len(), 3, "scan should yield all inserted rows");
        assert_eq!(got, expected);
    }
}
