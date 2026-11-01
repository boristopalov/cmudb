//! SQL query benchmarks through the same parser, planner and executors the tests use.
//! Run with `cargo bench --features bench --bench query`.

#[path = "../tests/common/mod.rs"]
mod common;

use common::Db;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
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

fn loaded_db() -> Db {
    let db = Db::new(POOL_FRAMES);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE TABLE u (tid INTEGER, w INTEGER)");
    insert_rows(&db, "t", ROWS, row_t);
    insert_rows(&db, "u", JOIN_ROWS, row_u);
    db.exec("CREATE INDEX t_id ON t (id)");
    db
}

fn reads(c: &mut Criterion) {
    let db = loaded_db();
    let mut group = c.benchmark_group("query");
    group.throughput(Throughput::Elements(ROWS as u64));
    let queries = [
        ("seq_scan_10k", "SELECT * FROM t"),
        ("filter_10pct_of_10k", "SELECT * FROM t WHERE v < 1000"),
        ("aggregate_10k", "SELECT COUNT(*), SUM(v) FROM t"),
        ("sort_10k", "SELECT * FROM t ORDER BY v"),
        (
            "nested_loop_join_10k_x_100",
            "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid",
        ),
    ];
    for (name, sql) in queries {
        group.bench_function(name, |b| b.iter(|| black_box(db.exec(sql))));
    }

    group.throughput(Throughput::Elements(JOIN_ROWS as u64));
    group.bench_function("index_join_100_into_10k", |b| {
        b.iter(|| black_box(db.exec("SELECT COUNT(*) FROM u JOIN t ON u.tid = t.id")))
    });

    group.throughput(Throughput::Elements(1));
    group.bench_function("index_point_lookup", |b| {
        b.iter(|| black_box(db.exec("SELECT * FROM t WHERE id = 4242")))
    });
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
