use crate::buffer_pool::BufferPoolManager;
use crate::catalog::Catalog;
use crate::processor::executor::{
    AggregationExecutor, DeleteExecutor, Executor, ExternalMergeSortExecutor, HashJoinExecutor,
    IndexScanExecutor, InsertExecutor, LimitExecutor, NestedIndexJoinExecutor,
    NestedLoopJoinExecutor, SeqScanExecutor, SortExecutor, UpdateExecutor, ValuesExecutor,
    WindowFunctionExecutor,
};
use crate::processor::plan::Plan;
use std::sync::Arc;

pub fn create_executor<'a>(
    catalog: &'a Catalog,
    bpm: &Arc<BufferPoolManager>,
    plan: &'a Plan,
) -> Box<dyn Executor + 'a> {
    match plan {
        Plan::SeqScan(p) => Box::new(SeqScanExecutor::new(catalog, p)),
        Plan::Insert(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(InsertExecutor::new(catalog, p, child))
        }
        Plan::Values(p) => Box::new(ValuesExecutor::new(p)),
        Plan::Update(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(UpdateExecutor::new(catalog, p, child))
        }
        Plan::Delete(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(DeleteExecutor::new(catalog, p, child))
        }
        Plan::IndexScan(p) => Box::new(IndexScanExecutor::new(catalog, p)),
        Plan::Aggregation(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(AggregationExecutor::new(p, child))
        }
        Plan::NestedLoopJoin(p) => {
            let left = create_executor(catalog, bpm, &p.left);
            let right = create_executor(catalog, bpm, &p.right);
            Box::new(NestedLoopJoinExecutor::new(p, left, right))
        }
        Plan::NestedIndexJoin(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(NestedIndexJoinExecutor::new(catalog, p, child))
        }
        Plan::HashJoin(p) => {
            let left = create_executor(catalog, bpm, &p.left);
            let right = create_executor(catalog, bpm, &p.right);
            Box::new(HashJoinExecutor::new(p, left, right, bpm.clone()))
        }
        Plan::Sort(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(SortExecutor::new(p, child))
        }
        Plan::ExternalMergeSort(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(ExternalMergeSortExecutor::new(p, child, bpm.clone()))
        }
        Plan::Limit(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(LimitExecutor::new(p, child))
        }
        Plan::WindowFunction(p) => {
            let child = create_executor(catalog, bpm, &p.child);
            Box::new(WindowFunctionExecutor::new(p, child))
        }
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

    fn make_catalog() -> (Catalog, Arc<BufferPoolManager>, tempfile::TempDir) {
        let num_frames = 16;
        let dir = tempdir().expect("tempdir");
        let dm = DiskManager::new(&dir.path().join("db"), &dir.path().join("log"))
            .expect("disk manager");
        let scheduler = DiskScheduler::new(dm);
        let replacer = ArcReplacer::new(num_frames);
        let bpm = Arc::new(create_buffer_pool_manager(num_frames, replacer, scheduler));
        (Catalog::new(bpm.clone()), bpm, dir)
    }

    fn make_schema() -> Schema {
        Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::BOOLEAN, "active".to_string()),
        ])
    }

    #[test]
    fn insert_then_seq_scan_returns_inserted_rows() {
        let (mut catalog, bpm, _dir) = make_catalog();
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

        let mut insert_exec = create_executor(&catalog, &bpm, &insert_plan);
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
        let mut scan_exec = create_executor(&catalog, &bpm, &scan_plan);
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
