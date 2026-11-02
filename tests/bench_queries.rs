//! Checks that every SQL benchmark in `benches/query.rs` still returns its expected rows and
//! still gets the plan it is named after, without timing anything.

mod bench_cases;
mod common;

#[test]
fn every_query_benchmark_returns_its_expected_rows_and_plan() {
    let db = bench_cases::loaded_db();
    for case in bench_cases::cases() {
        bench_cases::check_case(&db, &case);
    }
}

#[test]
fn the_insert_benchmark_stores_every_row() {
    bench_cases::check_insert(&bench_cases::insert_sql());
}
