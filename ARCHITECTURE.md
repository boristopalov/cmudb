# Architecture

A one-minute map of cmudb. Rules the compiler or tests can check live in code; this file
holds the ones that cut across modules. MVCC details are in `docs/mvcc.md`.

## Layer map

A SQL statement flows top to bottom:

| Layer | Code | Output |
|---|---|---|
| Parse | `sqlparser` crate | `Statement` AST |
| Bind | `src/parser.rs` (`Binder`) | `BoundStatement`, names resolved against the catalog |
| Plan | `src/planner.rs` (`Planner`) | `Plan` tree (`src/processor/plan.rs`) |
| Build | `src/processor/executor_factory.rs` | `Box<dyn Executor>` tree |
| Execute | `src/processor/executor.rs` (trait, `Batch`, errors), `src/processor/executors/` (one file per family) | `Batch`es of rows, pulled with `next_batch` |
| Access | `src/catalog/`, `src/table_heap.rs`, `src/index/`, `src/concurrency/` | tuples, index entries, visible versions |
| Pages | `src/table_page.rs`, `src/index/page_view.rs` | typed views over a pinned frame's bytes |
| Buffer pool | `src/buffer_pool.rs`, `src/replacer.rs` (ARC) | `PageRef` / `PageMut` guards |
| Disk | `src/disk.rs` | `DiskScheduler` thread over `DiskManager` |

`cmudb::Database::open(dir)` keeps a database in `dir/cmudb.data`; `Database::scratch` uses a
temporary directory instead. Page 0 of the data file is a header that points at a chain of
metadata pages (`src/database_file.rs`) holding the catalog, the page allocator's next and free
ids, and the transaction clocks. The data file is locked while a `Database` has it open, so a
second process gets `PersistError::InUse`. Page `id` lives at byte `id * PAGE_SIZE` of the data file.
Page, frame, table, index and record ids are the newtypes in `src/ids.rs`; a missing page
link is `Option<PageId>`, and only the encode/decode helpers there know its on-disk sentinel.

Pages of an opened database go through a write-ahead log, `dir/cmudb.log` (`src/wal.rs`): the
`DiskManager` appends every page write to it and reads a page's newest image from it first, so
the data file changes only at a log checkpoint. `Database::checkpoint`, run after every commit
through `Database` and on close, writes the metadata chain and flushes every dirty page, then
records a durable point and fsyncs the log; it holds the transaction manager's commit latch
meanwhile, so a durable point never splits a commit or an abort. Opening replays the log up to
its last durable point and drops the rest. Since evicted pages can carry uncommitted versions,
transactions also log a `Writes` record before their first write, an `Undo` record with each
version they overwrite (under the page latch, before the page can be logged) and an `End`
record when they finish; a transaction without a durable `End` is rolled back on open from its
undo images. Once the log passes `CHECKPOINT_LOG_BYTES`, a durable point also copies the
logged pages into the data file and replaces the log with one holding only the records of
transactions still running. After a failed fsync of the log, the data file or the log's
directory, the log refuses every later page and durable point, since a later fsync can succeed
without the bytes the failed one covered; reopening recovers from the log. Opening a table whose page links point past the recorded
allocation fails with `TableHeapError::UnallocatedPageLink`.

`cmudb::Database` (`src/database.rs`) runs the whole pipeline and
owns the session's transaction block (`BEGIN`, `COMMIT`, `ROLLBACK`, autocommit). It returns a
typed `StatementResult` per statement. The shell (`src/main.rs`) only prints those results, and
`tests/common` is a thin wrapper over `Database`. One `Database` is one session: callers of
`execute` share its `BEGIN` block, and `execute_in` runs SQL in a transaction the caller owns.

## Pins and latches

- `read_page` returns a `PageRef`: a pin plus a shared latch. `write_page` and `new_page`
  return a `PageMut`: a pin plus an exclusive latch, and the page is marked dirty. Dropping
  the guard releases both. There is no separate unpin call.
- Latches are `parking_lot` locks and are not reentrant. Asking for a page your thread
  already holds deadlocks, even read after read once a writer is queued.
- Every live guard pins a frame. With every frame pinned, fetching a page that isn't
  resident fails with `BpmError::NoFreeFrame`. Keep guards in a block scope and don't keep one across a
  `next_batch` return; scans copy what they need (`BPlusTreeIter` copies the leaf) or run a
  closure under the latch (`TableHeapIterator::next_with`, `TableHeap::write_with`).
- Outside the B+ tree, hold one page guard at a time. `TableHeap::insert` drops the new
  tail page before it latches the old tail to link it.
- B+ tree latches go top down: header, root, then each child before its parent is
  released (`step_down`). Pessimistic inserts and removes write-latch the header and the
  path, releasing ancestors once a child is safe (`TreeContext::descend_into`): for an
  insert, an entry of the longest key still fits it; for a remove, losing one leaves it at
  least minimally full by bytes. Searches,
  scans and the optimistic write path read-crab from the header, holding its read latch
  until the root is latched (`get_root_page_from_header`), so a root split or collapse
  cannot swap the root in between; an optimistic write trades the leaf's read latch for a
  write latch while it still holds the parent. A sibling is latched only while its parent is held. Range scans
  hold one leaf at a time.
- Across modules: an index's `unique_latch` (several indexes in ascending oid order), then
  page latches, then the transaction manager's version map. Commit takes `commit_latch`
  before any page latch.
- `delete_page` frees a page's frame, its slot on disk and its page id; `new_page` hands
  freed ids out again, lowest first, before it grows the pool. It fails while a guard pins
  the page, so only scratch pages no other reader can reach are deleted. Executors spill to
  `SpillHeap`s, which free their pages (`TableHeap::free`) when freed or dropped: `ORDER BY`
  frees each sorted run once it is merged or the sort closes, and hash join frees each
  partition pair once it is joined and the rest on close.

## Adding an executor

1. Add a plan node struct and a `Plan` variant in `src/processor/plan.rs`.
2. Implement `Executor` in the family's file under `src/processor/executors/` (scan, dml, join,
   aggregate, sort, pipeline, ddl) and re-export it from `executors/mod.rs`. `close` closes every
   child (`executors/mod.rs` tests check this). `next_batch` never returns an empty batch or
   more rows than its batch size, and keeps returning `None` once it has;
   `tests/batch_execution.rs` checks this across batch sizes.
3. Add its arm to `create_executor_with_batch_size` in `executor_factory.rs`.
4. Have `src/planner.rs` emit it, or, until it does, build its plan by hand in
   `tests/executors.rs`.
5. Reach it through SQL in tests, through `Database` (or `tests/common`, which wraps it).

## Tests

| Kind | Where |
|---|---|
| Unit | `#[cfg(test)] mod tests` beside the code |
| Storage and index integration, concurrency soaks | `tests/buffer_pool_integration.rs`, `tests/table_heap_integration.rs`, `tests/b_plus_tree_integration.rs` |
| SQL end to end | `tests/sql_e2e.rs` on the `tests/common` harness; `tests/window_functions.rs`, `tests/e2e.rs`, `tests/mvcc.rs` and `tests/batch_execution.rs` |
| Hand-built plans | `tests/executors.rs` |
| Persistence across reopen | `tests/persistence.rs` |
| Shell binary | `tests/cli.rs` |
| Fixtures | `cmudb::testing` (`bpm(frames)`, `catalog(frames)`), built for unit tests and, through the `testing` feature, for `tests/` and `benches/` |
| Benchmark result and plan checks | `tests/bench_queries.rs`, cases in `tests/bench_cases` |

A known gap is a typed error plus a test marked `#[ignore = "known gap: …"]` that names it,
never a comment (see `CONTRIBUTING.md`).
