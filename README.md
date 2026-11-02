# cmudb

This is a Rust implementation of a relational DBMS. It takes heavy inspiration from CMU's BusTub (https://github.com/cmu-db/bustub).

## Build and run

```sh
cargo build
cargo run      # SQL shell on a fresh temporary database; end statements with `;`, quit with \q
cargo run -- mydb   # the same shell on the database in directory mydb/, kept across runs
cargo test
```

`cargo run -- mydb` keeps tables, rows and indexes in `mydb/cmudb.data`, written after every
commit and on exit. There is no WAL yet, so a crash or kill between those writes can damage the
file. Data folders made before data file version 2 do not open.

CONTRIBUTING.md lists the checks CI runs, and ARCHITECTURE.md maps the code.

## What's done:
- disk-backed pages for tables, read and written in place through typed views over the page bytes
- a single data file per database holding the pages, the catalog and the transaction clocks, so
  tables, rows and indexes survive a restart; the file is locked while a process has it open
- Adaptive replacement cache policy for pages
- B+ tree indexes with optimistic latch coupling
- Most executors described in the CMU course, exchanging batches of rows (pull-based, 1024 rows per batch by default)
- MVCC with snapshot isolation (`BEGIN`/`COMMIT`/`ROLLBACK` in the shell; see docs/mvcc.md)
- parser (using sqlparser crate), binder and a rule-based planner, each failing with its own
  typed error, and a clear "not supported" error for SQL they do not handle
- planner choices: an `IndexScan` for `col = constant` on a column with a single-column index, a
  nested index join for an inner join whose right table has a single-column index on its join
  column, a hash join for other equi-joins, and an
  external merge sort for `ORDER BY` that spills sorted runs to disk once its input passes a
  memory budget
- scratch pages from sorts and hash joins are freed and their page ids reused

SQL the shell runs:
- `CREATE TABLE`, `CREATE INDEX`, `DROP TABLE`, `DROP INDEX`, `INSERT ... VALUES`, `UPDATE`, `DELETE`
- `SELECT` with `WHERE`, inner and left joins of two tables (`JOIN ... ON`), `GROUP BY` with
  `COUNT`, `SUM`, `AVG`, `MIN` and `MAX`, `SELECT DISTINCT`, `ORDER BY` and `LIMIT`
- window functions: those aggregates and `RANK()` with `OVER (PARTITION BY ... ORDER BY ...)`
- column types `INT` (32-bit), `BIGINT`, `FLOAT` (64-bit), `BOOLEAN`, `TIMESTAMP`
  (microseconds, no time zone, written as `'YYYY-MM-DD HH:MM:SS'`), `TEXT` and `VARCHAR(n)`
  (at most `n` characters); `INT` and `BIGINT` values widen to `FLOAT` where needed
- operators `+ - * /`, comparisons, `AND` and `OR`

## What's missing:
- a cost-based optimizer (the planner uses fixed rules and no statistics)
- serializable isolation
- a write-ahead log and crash recovery
- `HAVING`, subqueries, `WITH`, `OFFSET`, `IS NULL`, `NOT`, `LIKE`, `IN`, `CAST`, `%`, string
  functions and `||`, joins of more than two tables, window frames (`ROWS`/`RANGE`) and window
  functions in a `SELECT` that also aggregates
- `CHAR(n)`; indexes on plain `TEXT` columns (index a `VARCHAR(n)` column instead)
- a row must fit in one 8 KiB page, and an `UPDATE` that grows a row past its page's free space
  fails instead of moving the row
- `DROP TABLE` and `DROP INDEX` do not free their pages, and GC does not reclaim deleted rows'
  slots or dead index entries

## Benchmarks

Benchmarks use [criterion](https://docs.rs/criterion) and sit behind the `bench` feature. CI
type-checks them on every pull request but never runs them. Run them by hand on a quiet machine:

```sh
cargo bench --features bench                  # everything, about 4 minutes
cargo bench --features bench --bench storage  # buffer pool and table heap
cargo bench --features bench --bench index    # B+ tree insert, search, scan
cargo bench --features bench --bench query    # SQL: scan, filter, aggregate, sort, joins, insert
cargo bench --features bench --bench index -- search   # filter by name
```

Criterion keeps the previous run in `target/criterion` and prints the change against it, so
to compare a branch, run the same bench on the base branch first and then on the branch.
The bracketed range is one run's confidence interval, and separate runs on the same machine
differ by more than that: `sort_10k` ranged from 7.77 to 8.37 ms across three runs on one machine.
Run each side two or three times and only treat a change as real when it is clearly larger
than the gap between those runs, roughly 10% or more.
When you run them, every bench first checks what it reads: the heap and B+ tree return every
inserted row and key, and each query returns the expected rows. The join and index query
benches also check that the planner still picks the plan they are named after, so a change
fails the run instead of showing up as a speedup. CI never runs a benchmark, but
`tests/bench_queries.rs` runs the query benches' result and plan checks as an ordinary test.
The query benches run SQL through `cmudb::Database`, via the `tests/common` harness, so they measure whichever
executor model the branch has. Their timings include parsing, binding and planning each
statement, not just execution, which matters most for small queries like `index_point_lookup`.

Baseline, on commit d97bbc8: the median of three runs on a 4 vCPU Intel Xeon @ 2.10GHz cloud
VM, rustc 1.99.1, release build. Time is the middle run's median, with the lowest and highest
median of the three in brackets:

| Benchmark | Time | Per element |
|---|---|---|
| buffer_pool/read_hit | 129 ns (122 to 131) | |
| buffer_pool/write_hit | 130 ns (119 to 132) | |
| buffer_pool/read_miss_evict | 27.3 µs (26.6 to 27.8) | |
| table_heap/insert_10k (64 frames) | 4.51 ms (4.32 to 4.66) | 451 ns |
| table_heap/scan_10k (64 frames) | 5.84 ms (5.63 to 5.96) | 584 ns |
| table_heap/get_random | 16 µs (15.7 to 16.7) | |
| b_plus_tree/insert_seq_10k | 7.87 ms (7.62 to 8.12) | 787 ns |
| b_plus_tree/insert_random_10k | 8.7 ms (8.7 to 8.71) | 870 ns |
| b_plus_tree/search_random_of_100k | 617 ns (601 to 652) | |
| b_plus_tree/scan_range_1k_of_100k | 41.8 µs (41 to 43) | 41.8 ns |
| b_plus_tree/scan_full_100k | 3.65 ms (3.61 to 3.69) | 36.5 ns |
| query/seq_scan_10k | 5.44 ms (5.15 to 5.53) | 544 ns |
| query/filter_10pct_of_10k | 2.82 ms (2.77 to 2.84) | 282 ns |
| query/aggregate_10k | 2.88 ms (2.82 to 2.92) | 288 ns |
| query/sort_10k | 7.81 ms (7.77 to 8.37) | 781 ns |
| query/hash_join_10k_x_100 | 9.66 ms (9.22 to 9.69) | |
| query/index_join_100_into_10k | 237 µs (230 to 242) | 2.37 µs per outer row |
| query/index_point_lookup | 9.64 µs (9.22 to 9.79) | |
| query/insert_1k_rows | 2.52 ms (2.34 to 2.55) | 2.52 µs |

The table heap benches use a 64-frame pool on a heap of about 270 pages, so scans and random
gets include disk reads and evictions. The bench asserts that the heap is larger than the pool,
and, before timing, that every read in two laps of `read_miss_evict`'s fixed cycle misses. The B+ tree lookup and scan benches use a
1024-frame pool and assert that the whole 100k-key tree stays in it, so their cost is buffer
pool hits and page reads rather than I/O.
