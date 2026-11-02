//! SQL query benchmarks through the same parser, planner and executors the tests use.
//! Run with `cargo bench --features bench --bench query`.

#[path = "../tests/common/mod.rs"]
mod common;

#[path = "../tests/bench_cases/mod.rs"]
mod bench_cases;

use bench_cases::{INSERT_CHUNK, cases, check_case, check_insert, empty_db, insert_sql, loaded_db};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

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
    let sql = insert_sql();
    check_insert(&sql);
    group.bench_function("insert_1k_rows", |b| {
        b.iter_batched(
            empty_db,
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
