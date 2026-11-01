//! SQL query benchmarks through the same parser, planner and executors the tests use.
//! Run with `cargo bench --features bench --bench query`.

#[path = "../tests/common/mod.rs"]
mod common;

use cmudb::parser::Binder;
use cmudb::planner::Planner;
use cmudb::processor::plan::Plan;
use common::{Db, Row, int, ints};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::hint::black_box;
use std::time::Duration;

const POOL_FRAMES: usize = 1024;
const ROWS: usize = 10_000;
const JOIN_ROWS: usize = 100;
const INSERT_CHUNK: usize = 1_000;

fn values(rows: std::ops::Range<usize>, f: impl Fn(usize) -> String) -> String {
    rows.map(f).collect::<Vec<_>>().join(", ")
}

fn insert_rows(db: &Db, table: &str, rows: usize, f: impl Fn(usize) -> String + Copy) {
    for start in (0..rows).step_by(INSERT_CHUNK) {
        let end = (start + INSERT_CHUNK).min(rows);
        db.exec(&format!(
            "INSERT INTO {table} VALUES {}",
            values(start..end, f)
        ));
    }
}

fn row_t(i: usize) -> String {
    format!("({i}, {})", (i * 7919) % ROWS)
}

fn row_u(i: usize) -> String {
    format!("({}, {i})", i * (ROWS / JOIN_ROWS))
}

fn sum_of_v() -> i32 {
    (0..ROWS).map(|i| ((i * 7919) % ROWS) as i32).sum()
}

fn loaded_db() -> Db {
    let db = Db::new(POOL_FRAMES);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE TABLE u (tid INTEGER, w INTEGER)");
    insert_rows(&db, "t", ROWS, row_t);
    insert_rows(&db, "u", JOIN_ROWS, row_u);
    db.exec("CREATE INDEX t_id ON t (id)");
    db
}

/// What a benchmark query has to return, checked once before it is timed.
enum Expect {
    Rows(usize),
    Count(i32),
    Exact(Vec<Vec<i32>>),
    SortedBy { rows: usize, col: usize },
}

/// One timed read query, with the result and plan it has to produce.
struct Case {
    name: &'static str,
    sql: &'static str,
    elements: usize,
    expect: Expect,
    plan: Option<fn(&Plan) -> bool>,
}

fn plan_of(db: &Db, sql: &str) -> Plan {
    let stmt = Parser::parse_sql(&GenericDialect, sql)
        .expect("parse")
        .remove(0);
    let bound = Binder::new(&db.catalog)
        .bind(vec![stmt])
        .unwrap_or_else(|e| panic!("bind {sql}: {}", e.0))
        .remove(0);
    Planner::new(&db.catalog)
        .plan(bound)
        .unwrap_or_else(|e| panic!("plan {sql}: {}", e.0))
}

fn plan_contains(plan: &Plan, wanted: fn(&Plan) -> bool) -> bool {
    wanted(plan)
        || match plan {
            Plan::Filter(p) => plan_contains(&p.child, wanted),
            Plan::Projection(p) => plan_contains(&p.child, wanted),
            Plan::Aggregation(p) => plan_contains(&p.child, wanted),
            Plan::Sort(p) => plan_contains(&p.child, wanted),
            Plan::Limit(p) => plan_contains(&p.child, wanted),
            _ => false,
        }
}

fn check_case(db: &Db, case: &Case) {
    if let Some(wanted) = case.plan {
        assert!(
            plan_contains(&plan_of(db, case.sql), wanted),
            "{}: the planner no longer picks the plan this benchmark is named after",
            case.name
        );
    }
    let rows: Vec<Row> = db.exec(case.sql);
    match &case.expect {
        Expect::Rows(n) => assert_eq!(rows.len(), *n, "{}: row count", case.name),
        Expect::Count(n) => assert_eq!(rows, vec![vec![int(*n)]], "{}: COUNT(*)", case.name),
        Expect::Exact(want) => assert_eq!(&ints(&rows), want, "{}: result", case.name),
        Expect::SortedBy { rows: n, col } => {
            let got = ints(&rows);
            assert_eq!(got.len(), *n, "{}: row count", case.name);
            assert!(
                got.windows(2).all(|w| w[0][*col] <= w[1][*col]),
                "{}: rows are sorted by column {col}",
                case.name
            );
        }
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "seq_scan_10k",
            sql: "SELECT * FROM t",
            elements: ROWS,
            expect: Expect::Rows(ROWS),
            plan: None,
        },
        Case {
            name: "filter_10pct_of_10k",
            sql: "SELECT * FROM t WHERE v < 1000",
            elements: ROWS,
            expect: Expect::Rows(ROWS / 10),
            plan: None,
        },
        Case {
            name: "aggregate_10k",
            sql: "SELECT COUNT(*), SUM(v) FROM t",
            elements: ROWS,
            expect: Expect::Exact(vec![vec![ROWS as i32, sum_of_v()]]),
            plan: None,
        },
        Case {
            name: "sort_10k",
            sql: "SELECT * FROM t ORDER BY v",
            elements: ROWS,
            expect: Expect::SortedBy { rows: ROWS, col: 1 },
            plan: None,
        },
        Case {
            name: "nested_loop_join_10k_x_100",
            sql: "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid",
            elements: ROWS,
            expect: Expect::Count(JOIN_ROWS as i32),
            plan: Some(|p| matches!(p, Plan::NestedLoopJoin(_))),
        },
        Case {
            name: "index_join_100_into_10k",
            sql: "SELECT COUNT(*) FROM u JOIN t ON u.tid = t.id",
            elements: JOIN_ROWS,
            expect: Expect::Count(JOIN_ROWS as i32),
            plan: Some(|p| matches!(p, Plan::NestedIndexJoin(_))),
        },
        Case {
            name: "index_point_lookup",
            sql: "SELECT * FROM t WHERE id = 4242",
            elements: 1,
            expect: Expect::Rows(1),
            plan: Some(|p| matches!(p, Plan::IndexScan(_))),
        },
    ]
}

fn reads(c: &mut Criterion) {
    let db = loaded_db();
    let mut group = c.benchmark_group("query");
    for case in cases() {
        check_case(&db, &case);
        group.throughput(Throughput::Elements(case.elements as u64));
        group.bench_function(case.name, |b| b.iter(|| black_box(db.exec(case.sql))));
    }
    group.finish();
}

fn writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    group.throughput(Throughput::Elements(INSERT_CHUNK as u64));
    let sql = format!("INSERT INTO t VALUES {}", values(0..INSERT_CHUNK, row_t));
    group.bench_function("insert_1k_rows", |b| {
        b.iter_batched(
            || {
                let db = Db::new(POOL_FRAMES);
                db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
                db
            },
            |db| {
                db.exec(&sql);
                db
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = config();
    targets = reads, writes
}
criterion_main!(benches);
