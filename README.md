# cmudb

This is a Rust implementation of a relational DBMS. It takes heavy inspiration from CMU's BusTub (https://github.com/cmu-db/bustub).

## What's done:
- disk-backed pages for tables
- Adaptive replacement cache policy for pages
- B+ tree indexes with optimistic latch coupling
- An in-memory catalog
- Most executors described in the CMU course
- parser (using sqlparser crate)
- binder
- planner

## What's missing:
- optimizer
- mvcc
- support for more types, only a few basic types are supported (int, booleans, floats, timestamps)


## Things I would change if I have the time:
- Pages as typed views: Currently pages are fully decoded and encoded as we read from and write to them. A better approach is for pages act as typed views over raw bytes, which would save quite a bit of i/o work.
- Improved execution model: A volcano-style iteration model is used here, and notably there is no batching going on, so rows that the db system materializes are pulled 1 by 1.

## Benchmarks

Benchmarks use [criterion](https://docs.rs/criterion) and sit behind the `bench` feature, so
CI (`cargo clippy --all-targets`, `cargo test`, `cargo llvm-cov`) neither builds nor runs them.
Run them by hand on a quiet machine:

```sh
cargo bench --features bench                  # everything, about 4 minutes
cargo bench --features bench --bench storage  # buffer pool and table heap
cargo bench --features bench --bench index    # B+ tree insert, search, scan
cargo bench --features bench --bench query    # SQL: scan, filter, aggregate, sort, joins, insert
cargo bench --features bench --bench index -- search   # filter by name
```

Criterion keeps the previous run in `target/criterion` and prints the change against it, so
to compare a branch, run the same bench on the base branch first and then on the branch.
Only treat a change as real when it is larger than the run-to-run spread (the bracketed range).
The query benches run SQL through the same harness as `tests/common`, so they measure whichever
executor model the branch has.

Baseline, one run on a 4 vCPU Xeon @ 2.80GHz cloud VM, rustc 1.97, release build, on the
code as of the CI pull request, before batched execution, page views and MVCC (median, with
criterion's 95% interval):

| Benchmark | Time | Per element |
|---|---|---|
| buffer_pool/read_hit | 181 ns (176 to 186) | |
| buffer_pool/write_hit | 186 ns (176 to 196) | |
| buffer_pool/read_miss_evict | 47.5 µs (45.5 to 49.4) | |
| table_heap/insert_10k (64 frames) | 14.4 ms (13.7 to 14.9) | 1.4 µs |
| table_heap/scan_10k (64 frames) | 11.9 ms (11.5 to 12.4) | 1.2 µs |
| table_heap/get_random | 26.8 µs (25.4 to 28.7) | |
| b_plus_tree/insert_seq_10k | 249 ms (245 to 254) | 25 µs |
| b_plus_tree/insert_random_10k | 244 ms (238 to 253) | 24 µs |
| b_plus_tree/search_random_of_100k | 52.4 µs (50.5 to 54.8) | |
| b_plus_tree/scan_range_1k_of_100k | 166 µs (162 to 173) | 166 ns |
| b_plus_tree/scan_full_100k | 10.6 ms (10.2 to 11.1) | 106 ns |
| query/seq_scan_10k | 20.2 ms (19.7 to 20.9) | 2.0 µs |
| query/filter_10pct_of_10k | 17.8 ms (17.3 to 18.5) | 1.8 µs |
| query/aggregate_10k | 16.5 ms (16.4 to 16.6) | 1.6 µs |
| query/sort_10k | 22.4 ms (22.2 to 22.8) | 2.2 µs |
| query/nested_loop_join_10k_x_100 | 57.9 ms (57.1 to 59.0) | |
| query/index_join_100_into_10k | 2.6 ms (2.5 to 2.7) | 26 µs per outer row |
| query/index_point_lookup | 35.0 µs (33.6 to 36.9) | |
| query/insert_1k_rows | 4.9 ms (4.8 to 5.0) | 4.9 µs |

The table heap benches use a 64-frame pool on a heap of about 270 pages, so scans and random
gets include disk reads and evictions. The B+ tree benches use a 1024-frame pool, so their cost is
mostly decoding whole pages: with typed page views on the same pool, a search takes about 0.9 µs.
