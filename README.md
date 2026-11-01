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
Before timing, every bench checks what it reads: the heap and B+ tree return every inserted
row and key, and each query returns the expected rows. The join and index query benches also
check that the planner still picks the plan they are named after, so a change fails loudly
instead of showing up as a speedup.
The query benches run SQL through the same harness as `tests/common`, so they measure whichever
executor model the branch has.

Baseline, one run on a 4 vCPU Xeon @ 2.80GHz cloud VM, rustc 1.97, release build, on the
code as of the CI pull request, before batched execution, page views and MVCC (median, with
criterion's 95% interval):

| Benchmark | Time | Per element |
|---|---|---|
| buffer_pool/read_hit | 171 ns (166 to 176) | |
| buffer_pool/write_hit | 164 ns (161 to 167) | |
| buffer_pool/read_miss_evict | 48.2 µs (45.6 to 50.8) | |
| table_heap/insert_10k (64 frames) | 12.6 ms (12.2 to 13.0) | 1.3 µs |
| table_heap/scan_10k (64 frames) | 10.8 ms (10.3 to 11.5) | 1.1 µs |
| table_heap/get_random | 27.2 µs (25.5 to 28.6) | |
| b_plus_tree/insert_seq_10k | 246 ms (243 to 250) | 25 µs |
| b_plus_tree/insert_random_10k | 240 ms (235 to 246) | 24 µs |
| b_plus_tree/search_random_of_100k | 51.3 µs (50.1 to 53.0) | |
| b_plus_tree/scan_range_1k_of_100k | 155 µs (153 to 156) | 155 ns |
| b_plus_tree/scan_full_100k | 10.8 ms (10.2 to 11.3) | 108 ns |
| query/seq_scan_10k | 21.9 ms (21.0 to 23.0) | 2.2 µs |
| query/filter_10pct_of_10k | 18.6 ms (18.2 to 19.4) | 1.9 µs |
| query/aggregate_10k | 20.2 ms (18.5 to 21.6) | 2.0 µs |
| query/sort_10k | 24.3 ms (23.2 to 25.2) | 2.4 µs |
| query/nested_loop_join_10k_x_100 | 60.9 ms (59.2 to 62.7) | |
| query/index_join_100_into_10k | 2.4 ms (2.4 to 2.5) | 24 µs per outer row |
| query/index_point_lookup | 33.8 µs (32.9 to 34.9) | |
| query/insert_1k_rows | 4.9 ms (4.7 to 5.2) | 4.9 µs |

The table heap benches use a 64-frame pool on a heap of about 270 pages, so scans and random
gets include disk reads and evictions. The B+ tree benches use a 1024-frame pool, so their cost is
mostly decoding whole pages rather than I/O.
