use crate::buffer_pool::BufferPoolManager;
use crate::catalog::Catalog;
use crate::processor::executor::{
    AggregationExecutor, CreateIndexExecutor, CreateTableExecutor, DEFAULT_BATCH_SIZE,
    DeleteExecutor, DropIndexExecutor, DropTableExecutor, Executor, ExternalMergeSortExecutor,
    FilterExecutor, HashJoinExecutor, IndexScanExecutor, InsertExecutor, LimitExecutor,
    NestedIndexJoinExecutor, NestedLoopJoinExecutor, ProjectionExecutor, SeqScanExecutor,
    SortExecutor, UpdateExecutor, ValuesExecutor, WindowFunctionExecutor,
};
use crate::processor::plan::Plan;
use std::sync::Arc;

/// Builds the executor tree for `plan`, exchanging batches of up to [`DEFAULT_BATCH_SIZE`] rows.
pub fn create_executor<'a>(
    catalog: &'a Catalog,
    bpm: &Arc<BufferPoolManager>,
    plan: &'a Plan,
) -> Box<dyn Executor + 'a> {
    create_executor_with_batch_size(catalog, bpm, plan, DEFAULT_BATCH_SIZE)
}

/// Builds the executor tree for `plan`, with every row-producing executor in the tree
/// targeting `batch_size` rows per batch.
pub fn create_executor_with_batch_size<'a>(
    catalog: &'a Catalog,
    bpm: &Arc<BufferPoolManager>,
    plan: &'a Plan,
    batch_size: usize,
) -> Box<dyn Executor + 'a> {
    assert!(batch_size > 0, "batch size must be at least 1");
    let build = |child: &'a Plan| create_executor_with_batch_size(catalog, bpm, child, batch_size);
    match plan {
        Plan::SeqScan(p) => Box::new(SeqScanExecutor::new(catalog, p, batch_size)),
        Plan::Insert(p) => {
            let child = build(&p.child);
            Box::new(InsertExecutor::new(catalog, p, child))
        }
        Plan::Values(p) => Box::new(ValuesExecutor::new(p, batch_size)),
        Plan::Update(p) => {
            let child = build(&p.child);
            Box::new(UpdateExecutor::new(catalog, p, child))
        }
        Plan::Delete(p) => {
            let child = build(&p.child);
            Box::new(DeleteExecutor::new(catalog, p, child))
        }
        Plan::IndexScan(p) => Box::new(IndexScanExecutor::new(catalog, p, batch_size)),
        Plan::Aggregation(p) => {
            let child = build(&p.child);
            Box::new(AggregationExecutor::new(p, child, batch_size))
        }
        Plan::NestedLoopJoin(p) => {
            let left = build(&p.left);
            let right = build(&p.right);
            Box::new(NestedLoopJoinExecutor::new(p, left, right, batch_size))
        }
        Plan::NestedIndexJoin(p) => {
            let child = build(&p.child);
            Box::new(NestedIndexJoinExecutor::new(catalog, p, child, batch_size))
        }
        Plan::HashJoin(p) => {
            let left = build(&p.left);
            let right = build(&p.right);
            Box::new(HashJoinExecutor::new(
                p,
                left,
                right,
                bpm.clone(),
                batch_size,
            ))
        }
        Plan::Sort(p) => {
            let child = build(&p.child);
            Box::new(SortExecutor::new(p, child, batch_size))
        }
        Plan::ExternalMergeSort(p) => {
            let child = build(&p.child);
            Box::new(ExternalMergeSortExecutor::new(
                p,
                child,
                bpm.clone(),
                batch_size,
            ))
        }
        Plan::Limit(p) => {
            let child = build(&p.child);
            Box::new(LimitExecutor::new(p, child))
        }
        Plan::WindowFunction(p) => {
            let child = build(&p.child);
            Box::new(WindowFunctionExecutor::new(p, child, batch_size))
        }
        Plan::Filter(p) => {
            let child = build(&p.child);
            Box::new(FilterExecutor::new(p, child))
        }
        Plan::Projection(p) => {
            let child = build(&p.child);
            Box::new(ProjectionExecutor::new(p, child))
        }
        Plan::CreateTable(p) => Box::new(CreateTableExecutor::new(catalog, p)),
        Plan::DropTable(p) => Box::new(DropTableExecutor::new(catalog, p)),
        Plan::CreateIndex(p) => Box::new(CreateIndexExecutor::new(catalog, p, bpm.clone())),
        Plan::DropIndex(p) => Box::new(DropIndexExecutor::new(catalog, p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Catalog, DataType, Schema, Value};
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::processor::executor::collect_rows;
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
        let (catalog, bpm, _dir) = make_catalog();
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
        let batch = insert_exec
            .next_batch()
            .expect("insert next_batch")
            .expect("insert produces a count tuple");
        assert_eq!(batch.len(), 1);
        let (count_tuple, _) = batch.into_iter().next().unwrap();
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
        for (tuple, _rid) in collect_rows(&mut *scan_exec).expect("scan") {
            let row: Vec<Option<Value>> = (0..make_schema().cols.len())
                .map(|i| make_schema().get_value(&tuple, i).expect("decode"))
                .collect();
            got.push(row);
        }

        assert_eq!(got.len(), 3, "scan should yield all inserted rows");
        assert_eq!(got, expected);
    }
}
