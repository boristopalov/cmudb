use crate::processor::executor::{DEFAULT_BATCH_SIZE, Executor, ExecutorContext};
use crate::processor::executors::{
    AggregationExecutor, DdlExecutor, DdlOp, DeleteExecutor, ExternalMergeSortExecutor,
    FilterExecutor, HashJoinExecutor, IndexScanExecutor, InsertExecutor, LimitExecutor,
    NestedIndexJoinExecutor, NestedLoopJoinExecutor, ProjectionExecutor, SeqScanExecutor,
    SortExecutor, UpdateExecutor, ValuesExecutor, WindowFunctionExecutor,
};
use crate::processor::plan::Plan;

/// Builds the executor tree for `plan`, exchanging batches of up to [`DEFAULT_BATCH_SIZE`] rows.
pub fn create_executor<'a>(ctx: &'a ExecutorContext<'a>, plan: &'a Plan) -> Box<dyn Executor + 'a> {
    create_executor_with_batch_size(ctx, plan, DEFAULT_BATCH_SIZE)
}

/// Builds the executor tree for `plan`, with every row-producing executor in the tree
/// targeting `batch_size` rows per batch.
pub fn create_executor_with_batch_size<'a>(
    ctx: &'a ExecutorContext<'a>,
    plan: &'a Plan,
    batch_size: usize,
) -> Box<dyn Executor + 'a> {
    assert!(batch_size > 0, "batch size must be at least 1");
    let catalog = ctx.catalog;
    let bpm = &ctx.bpm;
    let build = |child: &'a Plan| create_executor_with_batch_size(ctx, child, batch_size);
    let ddl =
        |op| -> Box<dyn Executor + 'a> { Box::new(DdlExecutor::new(catalog, bpm.clone(), op)) };
    match plan {
        Plan::SeqScan(p) => Box::new(SeqScanExecutor::new(ctx, p, batch_size)),
        Plan::Insert(p) => {
            let child = build(&p.child);
            Box::new(InsertExecutor::new(ctx, p, child))
        }
        Plan::Values(p) => Box::new(ValuesExecutor::new(p, batch_size)),
        Plan::Update(p) => {
            let child = build(&p.child);
            Box::new(UpdateExecutor::new(ctx, p, child))
        }
        Plan::Delete(p) => {
            let child = build(&p.child);
            Box::new(DeleteExecutor::new(ctx, p, child))
        }
        Plan::IndexScan(p) => Box::new(IndexScanExecutor::new(ctx, p, batch_size)),
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
            Box::new(NestedIndexJoinExecutor::new(ctx, p, child, batch_size))
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
                ctx.sort_memory_budget_bytes,
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
        Plan::CreateTable(p) => ddl(DdlOp::CreateTable(p)),
        Plan::DropTable(p) => ddl(DdlOp::DropTable(p)),
        Plan::CreateIndex(p) => ddl(DdlOp::CreateIndex(p)),
        Plan::DropIndex(p) => ddl(DdlOp::DropIndex(p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::catalog::{DataType, Schema, Value};
    use crate::concurrency::{IsolationLevel, TransactionManager};

    use crate::processor::executor::ExecutorContext;
    use crate::processor::executor::{collect_rows, rows_affected};
    use crate::processor::plan::{Expression, InsertPlanNode, SeqScanPlanNode, ValuesPlanNode};

    use std::sync::Arc;

    fn make_schema() -> Schema {
        Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::BOOLEAN, "active".to_string()),
        ])
    }

    #[test]
    fn insert_then_seq_scan_returns_inserted_rows() {
        let (catalog, bpm, _dir) = crate::testing::catalog(16);
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

        let txn_mgr = Arc::new(TransactionManager::new(bpm.clone()));
        let ctx = ExecutorContext::new(
            &catalog,
            bpm.clone(),
            txn_mgr.begin(IsolationLevel::SnapshotIsolation),
            txn_mgr.clone(),
        );
        let mut insert_exec = create_executor(&ctx, &insert_plan);
        insert_exec.open().expect("open insert");
        let count = rows_affected(&insert_plan, &mut *insert_exec).expect("insert");
        assert_eq!(count, 3, "insert should report 3 rows");
        drop(insert_exec);
        txn_mgr.commit(&ctx.txn).expect("commit insert");

        let scan_plan = Plan::SeqScan(SeqScanPlanNode {
            table_oid,
            schema: make_schema(),
        });
        let ctx = ExecutorContext::new(
            &catalog,
            bpm.clone(),
            txn_mgr.begin(IsolationLevel::SnapshotIsolation),
            txn_mgr.clone(),
        );
        let mut scan_exec = create_executor(&ctx, &scan_plan);
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
