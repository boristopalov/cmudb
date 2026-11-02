//! Batch-boundary tests for the batched executor model.
//!
//! Every query here is run with a range of batch sizes, including sizes smaller than a
//! join's fan-out, smaller than the number of groups an aggregation emits, and larger than
//! the whole input. For each run we check the `next_batch()` contract (no empty batches,
//! no batch larger than the configured size, `None` stays `None`) and that the rows produced
//! are the same no matter how they were batched.

use cmudb::buffer_pool::BufferPoolManager;
use cmudb::catalog::{Catalog, DataType, IndexType, Schema, Value};
use cmudb::concurrency::{IsolationLevel, TransactionManager};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::processor::executor::ExecutorContext;
use cmudb::processor::executor::RowCursor;
use cmudb::processor::executor_factory::create_executor_with_batch_size;
use cmudb::processor::plan::{
    AggType, AggregationPlanNode, ColumnAssignment, DeletePlanNode, Expression,
    ExternalMergeSortPlanNode, FilterPlanNode, HashJoinPlanNode, IndexScanPlanNode, InsertPlanNode,
    JoinType, LimitPlanNode, NestedIndexJoinPlanNode, NestedLoopJoinPlanNode, Op, Plan,
    ProjectionPlanNode, SeqScanPlanNode, SortExpr, SortPlanNode, UpdatePlanNode, ValuesPlanNode,
    WindowFunctionPlanNode,
};
use cmudb::replacer::ArcReplacer;
use std::collections::HashMap;
use std::ops::Bound;
use std::sync::Arc;
use tempfile::{TempDir, tempdir};

/// Batch sizes to exercise: 1 (degenerate), sizes below the join fan-out (7),
/// sizes right at / above it, and one larger than every input.
const BATCH_SIZES: [usize; 7] = [1, 2, 3, 7, 8, 64, 1024];

/// Rows in `l` with a key that matches something in `r`
const MATCHING_LEFT_ROWS: i32 = 25;
/// Distinct join keys; every key appears in `r` this many times
const NUM_KEYS: i32 = 5;
const RIGHT_ROWS_PER_KEY: i32 = 7;
/// Key of the single `l` row that has no partner in `r` or `s`
const UNMATCHED_KEY: i32 = 99;
/// Rows in `s`, which has a unique, indexed key (the B+ tree index rejects duplicate keys)
const S_ROWS: i32 = 40;

type Rows = Vec<Vec<Option<Value>>>;

impl Env {
    /// A context running in a fresh transaction; commit it with [`Env::commit`].
    fn ctx(&self) -> ExecutorContext<'_> {
        ExecutorContext::new(
            &self.catalog,
            self.bpm.clone(),
            self.txn_mgr.begin(IsolationLevel::SnapshotIsolation),
            self.txn_mgr.clone(),
        )
    }

    fn commit(&self, ctx: &ExecutorContext<'_>) {
        self.txn_mgr.commit(&ctx.txn).expect("commit");
    }
}

struct Env {
    catalog: Catalog,
    bpm: Arc<BufferPoolManager>,
    txn_mgr: Arc<TransactionManager>,
    l_oid: u32,
    r_oid: u32,
    s_oid: u32,
    s_index_oid: u32,
    _dir: TempDir,
}

fn int(v: i32) -> Option<Value> {
    Some(Value::INT(v))
}

fn col(tuple_idx: u8, col_idx: u32) -> Expression {
    Expression::Column {
        tuple_idx,
        col_idx,
        dtype: DataType::INT,
    }
}

fn binary(left: Expression, op: Op, right: Expression) -> Expression {
    Expression::Binary {
        left: Box::new(left),
        right: Box::new(right),
        op,
    }
}

fn l_schema() -> Schema {
    Schema::new(vec![
        (DataType::INT, "id".to_string()),
        (DataType::INT, "k".to_string()),
    ])
}

fn r_schema() -> Schema {
    Schema::new(vec![
        (DataType::INT, "k".to_string()),
        (DataType::INT, "tag".to_string()),
    ])
}

fn joined_schema() -> Schema {
    let mut s = Schema::new(vec![
        (DataType::INT, "l.id".to_string()),
        (DataType::INT, "l.k".to_string()),
        (DataType::INT, "r.k".to_string()),
        (DataType::INT, "r.tag".to_string()),
    ]);
    s.join_offset = Some(2);
    s
}

fn s_schema() -> Schema {
    Schema::new(vec![
        (DataType::INT, "k".to_string()),
        (DataType::INT, "v".to_string()),
    ])
}

/// `s(k, v)`: unique keys 0..40 with v = 2k, indexed on k
fn s_rows() -> Rows {
    (0..S_ROWS).map(|k| vec![int(k), int(2 * k)]).collect()
}

/// `l(id, k)`: ids 0..25 with k = id % 5, plus one row (id 25) whose key matches nothing.
fn l_rows() -> Rows {
    let mut rows: Rows = (0..MATCHING_LEFT_ROWS)
        .map(|id| vec![int(id), int(id % NUM_KEYS)])
        .collect();
    rows.push(vec![int(MATCHING_LEFT_ROWS), int(UNMATCHED_KEY)]);
    rows
}

/// `r(k, tag)`: every key 0..5 appears 7 times, so each matching left row fans out to 7 rows.
fn r_rows() -> Rows {
    (0..NUM_KEYS)
        .flat_map(|k| (0..RIGHT_ROWS_PER_KEY).map(move |tag| vec![int(k), int(tag)]))
        .collect()
}

fn scan(table_oid: u32, schema: Schema) -> Box<Plan> {
    Box::new(Plan::SeqScan(SeqScanPlanNode { table_oid, schema }))
}

fn values_plan(schema: Schema, rows: &Rows) -> Plan {
    Plan::Values(ValuesPlanNode {
        schema,
        rows: rows
            .iter()
            .map(|r| r.iter().cloned().map(Expression::Constant).collect())
            .collect(),
    })
}

/// Builds a fresh database with `l`, `r` and `s` (plus an index on `s.k`), loading every table
/// through the batched insert path with an awkward batch size.
fn setup() -> Env {
    let num_frames = 64;
    let dir = tempdir().expect("tempdir");
    let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk");
    let bpm = Arc::new(create_buffer_pool_manager(
        num_frames,
        ArcReplacer::new(num_frames),
        DiskScheduler::new(dm),
    ));
    let catalog = Catalog::new(bpm.clone());
    let l_oid = catalog
        .create_table("l".to_string(), l_schema())
        .expect("create l");
    let r_oid = catalog
        .create_table("r".to_string(), r_schema())
        .expect("create r");
    let s_oid = catalog
        .create_table("s".to_string(), s_schema())
        .expect("create s");
    let s_index_oid = catalog
        .create_index(
            bpm.clone(),
            s_oid,
            "s_k".to_string(),
            &["k"],
            IndexType::BPlusTree,
        )
        .expect("create index");

    let env = Env {
        catalog,
        txn_mgr: Arc::new(TransactionManager::new(bpm.clone())),
        bpm,
        l_oid,
        r_oid,
        s_oid,
        s_index_oid,
        _dir: dir,
    };
    for (oid, schema, rows) in [
        (l_oid, l_schema(), l_rows()),
        (r_oid, r_schema(), r_rows()),
        (s_oid, s_schema(), s_rows()),
    ] {
        let plan = Plan::Insert(InsertPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(schema, &rows)),
            table_oid: oid,
        });
        let ctx = env.ctx();
        let mut exec = create_executor_with_batch_size(&ctx, &plan, 3);
        exec.open().expect("open insert");
        let batch = exec.next_batch().expect("insert").expect("count row");
        assert_eq!(batch.len(), 1);
        let count = i32::from_ne_bytes(batch.rows[0].0.data.as_slice().try_into().unwrap());
        assert_eq!(count, rows.len() as i32);
        assert!(exec.next_batch().expect("insert after end").is_none());
        drop(exec);
        env.commit(&ctx);
    }
    env
}

/// Runs `plan` with the given batch size, checking the batch contract along the way, and
/// returns the decoded rows in the order they were produced.
fn run(env: &Env, plan: &Plan, batch_size: usize) -> Rows {
    let ctx = env.ctx();
    let mut exec = create_executor_with_batch_size(&ctx, plan, batch_size);
    exec.open().expect("open");
    let schema = exec.schema().clone();
    let mut rows = Vec::new();
    while let Some(batch) = exec.next_batch().expect("next_batch") {
        assert!(!batch.is_empty(), "executors must not emit empty batches");
        assert!(
            batch.len() <= batch_size,
            "batch of {} rows exceeds batch size {batch_size}",
            batch.len()
        );
        for (tuple, _) in batch {
            rows.push(decode(&schema, &tuple));
        }
    }
    for _ in 0..3 {
        assert!(
            exec.next_batch().expect("next_batch after end").is_none(),
            "an exhausted executor must keep returning None"
        );
    }
    exec.close().expect("close");
    drop(exec);
    env.commit(&ctx);
    rows
}

fn decode(schema: &Schema, tuple: &cmudb::table_heap::Tuple) -> Vec<Option<Value>> {
    (0..schema.cols.len())
        .map(|i| schema.get_value(tuple, i).expect("decode"))
        .collect()
}

fn sorted(mut rows: Rows) -> Rows {
    rows.sort();
    rows
}

/// Runs `plan` at every batch size and asserts that all runs produce `expected`.
/// When `ordered` is false the comparison ignores row order.
fn assert_all_batch_sizes(env: &Env, plan: &Plan, expected: &Rows, ordered: bool) {
    for bs in BATCH_SIZES {
        let got = run(env, plan, bs);
        if ordered {
            assert_eq!(&got, expected, "batch size {bs}");
        } else {
            assert_eq!(sorted(got), sorted(expected.clone()), "batch size {bs}");
        }
    }
}

/// Expected result of `l JOIN inner ON l.k = inner.k`, where inner's key is its first column
fn expected_join(inner: &Rows, left_outer: bool) -> Rows {
    let mut out = Vec::new();
    for l in l_rows() {
        let matches: Vec<_> = inner.iter().filter(|r| r[0] == l[1]).collect();
        if matches.is_empty() && left_outer {
            out.push(vec![l[0], l[1], None, None]);
        }
        for r in matches {
            out.push(vec![l[0], l[1], r[0], r[1]]);
        }
    }
    out
}

#[test]
fn seq_scan_and_values_preserve_order_across_batches() {
    let env = setup();
    assert_all_batch_sizes(&env, &scan(env.l_oid, l_schema()), &l_rows(), true);
    assert_all_batch_sizes(&env, &values_plan(r_schema(), &r_rows()), &r_rows(), true);
}

#[test]
fn selective_filter_and_projection() {
    let env = setup();
    let plan = Plan::Projection(ProjectionPlanNode {
        schema: Schema::new(vec![(DataType::INT, "id_x10".to_string())]),
        expressions: vec![binary(col(0, 0), Op::Mul, Expression::Constant(int(10)))],
        child: Box::new(Plan::Filter(FilterPlanNode {
            schema: l_schema(),
            predicate: binary(col(0, 1), Op::Eq, Expression::Constant(int(2))),
            child: scan(env.l_oid, l_schema()),
        })),
    });
    let expected: Rows = l_rows()
        .into_iter()
        .filter(|r| r[1] == int(2))
        .map(|r| vec![r[0].map(|v| v * Value::INT(10))])
        .collect();
    assert_eq!(
        expected.len(),
        5,
        "k = 2 keeps 5 of 26 rows, so most child batches filter down to nothing"
    );
    assert_all_batch_sizes(&env, &plan, &expected, true);
}

#[test]
fn limit_truncates_at_batch_edges() {
    let env = setup();
    for limit in [0u32, 1, 2, 5, 7, 8, 26, 100] {
        let plan = Plan::Limit(LimitPlanNode {
            schema: l_schema(),
            child: scan(env.l_oid, l_schema()),
            limit,
        });
        let expected: Rows = l_rows().into_iter().take(limit as usize).collect();
        assert_all_batch_sizes(&env, &plan, &expected, true);
    }
}

#[test]
fn nested_loop_join_fan_out_spans_batches() {
    let env = setup();
    for (join_type, left_outer) in [(JoinType::Inner, false), (JoinType::Left, true)] {
        let plan = Plan::NestedLoopJoin(NestedLoopJoinPlanNode {
            schema: joined_schema(),
            left: scan(env.l_oid, l_schema()),
            right: scan(env.r_oid, r_schema()),
            predicate: binary(col(0, 1), Op::Eq, col(1, 0)),
            join_type,
        });
        let expected = expected_join(&r_rows(), left_outer);
        assert_eq!(expected.len(), 175 + left_outer as usize);
        let keeps_left_order_then_right_order = true;
        assert_all_batch_sizes(&env, &plan, &expected, keeps_left_order_then_right_order);
    }
}

#[test]
fn nested_index_join_spans_batches() {
    let env = setup();
    for (join_type, left_outer) in [(JoinType::Inner, false), (JoinType::Left, true)] {
        let plan = Plan::NestedIndexJoin(NestedIndexJoinPlanNode {
            schema: joined_schema(),
            child: scan(env.l_oid, l_schema()),
            key_predicate: col(0, 1),
            table_oid: env.s_oid,
            table_schema: s_schema(),
            index_oid: env.s_index_oid,
            join_type,
        });
        let expected = expected_join(&s_rows(), left_outer);
        assert_eq!(expected.len(), 25 + left_outer as usize);
        assert_all_batch_sizes(&env, &plan, &expected, true);
    }
}

#[test]
fn hash_join_fan_out_spans_batches() {
    let env = setup();
    for (join_type, left_outer) in [(JoinType::Inner, false), (JoinType::Left, true)] {
        let plan = Plan::HashJoin(HashJoinPlanNode {
            schema: joined_schema(),
            left: scan(env.l_oid, l_schema()),
            right: scan(env.r_oid, r_schema()),
            left_exprs: vec![col(0, 1)],
            right_exprs: vec![col(0, 0)],
            join_type,
        });
        assert_all_batch_sizes(&env, &plan, &expected_join(&r_rows(), left_outer), false);
    }
}

#[test]
fn aggregation_groups_span_batches() {
    let env = setup();
    let plan = Plan::Aggregation(AggregationPlanNode {
        schema: Schema::new(vec![
            (DataType::INT, "k".to_string()),
            (DataType::INT, "count".to_string()),
            (DataType::INT, "sum".to_string()),
        ]),
        child: scan(env.l_oid, l_schema()),
        group_bys: vec![col(0, 1)],
        aggregates: vec![(col(0, 0), AggType::COUNT), (col(0, 0), AggType::SUM)],
    });
    let mut groups: HashMap<i32, (i32, i32)> = HashMap::new();
    for r in l_rows() {
        let (Some(Value::INT(id)), Some(Value::INT(k))) = (r[0], r[1]) else {
            unreachable!()
        };
        let e = groups.entry(k).or_default();
        e.0 += 1;
        e.1 += id;
    }
    let expected: Rows = groups
        .into_iter()
        .map(|(k, (count, sum))| vec![int(k), int(count), int(sum)])
        .collect();
    assert_eq!(expected.len(), 6);
    assert_all_batch_sizes(&env, &plan, &expected, false);
}

#[test]
fn sorts_emit_in_order_across_batches() {
    let env = setup();
    let order_by = vec![
        SortExpr::ascending(col(0, 1)),
        SortExpr::ascending(col(0, 0)),
    ];
    let mut expected = l_rows();
    expected.sort_by_key(|r| (r[1], r[0]));

    let sort = Plan::Sort(SortPlanNode {
        schema: l_schema(),
        child: scan(env.l_oid, l_schema()),
        order_by_exprs: order_by.clone(),
    });
    assert_all_batch_sizes(&env, &sort, &expected, true);

    let external = Plan::ExternalMergeSort(ExternalMergeSortPlanNode {
        schema: l_schema(),
        child: scan(env.l_oid, l_schema()),
        order_by_exprs: order_by,
    });
    assert_all_batch_sizes(&env, &external, &expected, true);
}

#[test]
fn window_function_output_spans_batches() {
    let env = setup();
    let plan = Plan::WindowFunction(WindowFunctionPlanNode {
        schema: Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::INT, "k".to_string()),
            (DataType::INT, "sum".to_string()),
        ]),
        child: scan(env.l_oid, l_schema()),
        order_by_exprs: None,
        partition_by_exprs: Some(vec![col(0, 1)]),
        aggregates: vec![(col(0, 0), AggType::SUM)],
        columns: vec![col(0, 0), col(0, 1)],
    });
    let rows = l_rows();
    let expected: Rows = rows
        .iter()
        .map(|r| {
            let sum: i32 = rows
                .iter()
                .filter(|o| o[1] == r[1])
                .map(|o| match o[0] {
                    Some(Value::INT(id)) => id,
                    _ => unreachable!(),
                })
                .sum();
            vec![r[0], r[1], int(sum)]
        })
        .collect();
    assert_all_batch_sizes(&env, &plan, &expected, false);
}

#[test]
fn index_scan_range_spans_batches() {
    let env = setup();
    let plan = Plan::IndexScan(IndexScanPlanNode {
        schema: s_schema(),
        start: Bound::Included(vec![Value::INT(5)]),
        end: Bound::Included(vec![Value::INT(30)]),
        index_oid: env.s_index_oid,
    });
    let expected: Rows = s_rows()
        .into_iter()
        .filter(|r| (int(5)..=int(30)).contains(&r[0]))
        .collect();
    assert_eq!(expected.len(), 26);
    assert_all_batch_sizes(&env, &plan, &expected, true);
}

#[test]
fn update_and_delete_touch_every_row_once() {
    for bs in BATCH_SIZES {
        let env = setup();

        let update = Plan::Update(UpdatePlanNode {
            schema: s_schema(),
            assignments: vec![ColumnAssignment {
                col_idx: 1,
                value: binary(col(0, 1), Op::Add, Expression::Constant(int(100))),
            }],
            child: scan(env.s_oid, s_schema()),
            table_oid: env.s_oid,
        });
        let updated = run(&env, &update, bs);
        assert_eq!(updated.len(), S_ROWS as usize, "batch size {bs}");

        let expected: Rows = s_rows()
            .into_iter()
            .map(|r| vec![r[0], r[1].map(|v| v + Value::INT(100))])
            .collect();
        assert_eq!(
            sorted(run(&env, &scan(env.s_oid, s_schema()), bs)),
            expected,
            "batch size {bs}"
        );

        let delete = Plan::Delete(DeletePlanNode {
            schema: s_schema(),
            child: Box::new(Plan::Filter(FilterPlanNode {
                schema: s_schema(),
                predicate: binary(col(0, 0), Op::Lt, Expression::Constant(int(10))),
                child: scan(env.s_oid, s_schema()),
            })),
            table_oid: env.s_oid,
        });
        let deleted = run(&env, &delete, bs);
        assert_eq!(deleted.len(), 10, "batch size {bs}");

        let remaining: Rows = expected.into_iter().filter(|r| r[0] >= int(10)).collect();
        assert_eq!(
            sorted(run(&env, &scan(env.s_oid, s_schema()), bs)),
            remaining,
            "batch size {bs}"
        );

        let via_index = Plan::IndexScan(IndexScanPlanNode {
            schema: s_schema(),
            start: Bound::Unbounded,
            end: Bound::Unbounded,
            index_oid: env.s_index_oid,
        });
        assert_eq!(
            run(&env, &via_index, bs),
            remaining,
            "index disagrees with the heap after batched update and delete, batch size {bs}"
        );
    }
}

#[test]
fn row_cursor_reads_rows_one_at_a_time() {
    let env = setup();
    let plan = scan(env.l_oid, l_schema());
    for bs in BATCH_SIZES {
        let ctx = env.ctx();
        let mut exec = create_executor_with_batch_size(&ctx, &plan, bs);
        exec.open().unwrap();
        let schema = exec.schema().clone();
        let mut cursor = RowCursor::new(&mut *exec);
        let mut got = Vec::new();
        while let Some((tuple, _)) = cursor.next_row().unwrap() {
            got.push(decode(&schema, &tuple));
        }
        assert!(cursor.next_row().unwrap().is_none());
        assert_eq!(got, l_rows(), "batch size {bs}");
    }
}
