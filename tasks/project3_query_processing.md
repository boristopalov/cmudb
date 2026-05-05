SUMMARY:
- Goal: Implement CMU DB project 3 (query processing / execution engine) on top of the existing buffer pool (project 1) and concurrent B+ tree (project 2).
- Context: Codebase has buffer pool + B+ tree working, plus skeleton `catalog/catalog.rs` (really just Schema/DataType/Tuple primitives) and a stub `table_heap.rs` (`insert_tuple` doesn't actually write bytes; `encode` is empty; no get/update/delete/iter). Nothing above the storage layer exists — no system-level Catalog (table registry), no executors, no expressions, no planner, no SQL parser. Project description: https://15445.courses.cs.cmu.edu/fall2025/project3/
- Outcome: A Volcano-style execution engine that can run hand-built plan trees end-to-end against real tables, with SQL parsing layered on top last.

## Architectural overview

The full pipeline (build bottom-up):

```
SQL → Parser → AST → Binder → bound stmt → Planner → plan tree → Optimizer → plan tree → Executor tree → tuples
                              (uses Catalog)                                                  ↓
                                                                                       TableHeap + Index
                                                                                            ↓
                                                                                        Buffer pool
```

What each component is for:
- **Catalog** (system-level, missing): registry of `table_oid → TableInfo {schema, name, table_heap}` and `index_oid → IndexInfo`. Executors deal in oids/column indices, not names — the catalog does that translation once.
- **TableHeap** (stub): row storage. Pages of tuples managed via buffer pool. Indexes hold `(key → RecordId)` pointers into it.
- **Tuple/Value/Schema**: row representation passed between executors. Currently `Tuple` borrows `&[u8]` — needs an owned variant with `from_values` / `get_value(col_idx)`.
- **Executor (Volcano)**: `trait Executor { fn init(&mut self); fn next(&mut self) -> Option<Tuple>; fn schema(&self) -> &Schema; }`. Each operator pulls one tuple at a time from its child so memory stays bounded.
- **Expression tree**: `ColumnValue`, `Constant`, `Comparison`, `Logic`, `Arithmetic`. Evaluates against a Tuple+Schema. Used by Filter, Join, Projection.
- **Parser/Binder/Planner/Optimizer**: SQL frontend. Deferrable — hand-build plan trees in tests until executors are working.

Decision on SQL parsing: use the `sqlparser` crate directly (not full DataFusion). DataFusion would replace the very layers we want to build ourselves.

PLAN:

## Phase 0 — make storage real (do FIRST)

Without a working TableHeap, nothing above it can be tested. Tests at this phase use `TableHeap` directly.

1. Define page layout for `TablePage` and document it as a comment at the top of `table_heap.rs`. Slot directory grows from header forward; tuple data grows from end of page backward; meet in the middle.
2. Implement `TablePage::encode` (currently empty) and finish `decode`. Round-trip test.
3. Implement real `TableHeap::insert_tuple(&Tuple) -> RecordId`. Must actually write tuple bytes into the page (current code only updates metadata in-memory and never persists). Allocate a new page when current page has insufficient free space; link via `next_page_id`.
4. Implement `TableHeap::get_tuple(rid: RecordId) -> Option<Tuple>`.
5. Implement `TableHeap::update_tuple(rid, new_tuple)` — for now, in-place if size matches; otherwise mark deleted + insert new. Note the new RID for index updates.
6. Implement `TableHeap::delete_tuple(rid)` as tombstone (add `is_deleted` flag in `TupleMeta`). Don't actually reclaim space yet.
7. Implement `TableHeap::iter() -> TableIterator` that walks pages via `next_page_id` and yields `(RecordId, Tuple)` for all non-deleted tuples.
8. (Maybe) Convert `Tuple` to owned form: `Tuple { data: Vec<u8>, rid: RecordId }`. Add `Tuple::from_values(&[Value], &Schema)` and `Tuple::get_value(&Schema, col_idx) -> Value`. Keep schema validation.
9. (Maybe) Add `Value` comparison (`PartialOrd`) and basic arithmetic (Add/Sub/Mul/Div as needed for expressions later — can defer until Phase 2).

Test: create heap, insert 100 tuples, iterate them back, look one up by RecordId, delete one, update one, re-iterate.

## Phase 1 — Catalog + Volcano machinery

10. Build a real system-level `Catalog` (separate from current schema-primitives module — consider renaming current `catalog.rs` to `schema.rs` or `types.rs`). Mirrors the bustub layout the user pasted: `tables_: HashMap<table_oid_t, TableInfo>`, `table_names_`, `indexes_`, `index_names_`, `next_table_oid_`. Methods: `create_table`, `get_table(name|oid)`, `create_index`, `get_index`, `get_table_indexes(table_name)`. `TableInfo` owns the `TableHeap`. Drop the `Transaction*`/`LockManager*`/`LogManager*` parameters — those are project 4.
11. Define `Executor` trait with `init`, `next`, `schema`.
12. Define plan node types as data: `Plan::SeqScan { table_oid }`, `Plan::Insert { table_oid, child }`, etc. Plan is a pure data structure; the executor is the runtime.
13. `SeqScanExecutor` — wraps `TableHeap::iter`.
14. `InsertExecutor` — pulls tuples from child, writes to heap, updates every index returned by `catalog.get_table_indexes`.
15. `ValuesExecutor` — emits a fixed list of literal tuples (used as the child of Insert).

Test: hand-build `Insert(Values([...]))`, run, then `SeqScan(t)` to read back. **No SQL yet.**

## Phase 2 — expressions + simple executors

16. Expression tree: `Expr::ColumnValue(idx)`, `Expr::Constant(Value)`, `Expr::Comparison(op, l, r)`, `Expr::Logic(And/Or/Not, ...)`, `Expr::Arithmetic(...)`. Method: `evaluate(&Tuple, &Schema) -> Value`.
17. `FilterExecutor` (predicate `Expr` + child).
18. `ProjectionExecutor` (list of `Expr` + child; output schema derived from exprs).
19. `LimitExecutor`.

Test: `Filter(SeqScan(t), col0 > 5)` against a populated heap returns expected rows.

## Phase 3 — the hard executors

20. `IndexScanExecutor` — uses B+ tree to get RIDs, then resolves each via `TableHeap::get_tuple`.
21. `DeleteExecutor` — child yields RIDs; tombstone in heap; remove from indexes.
22. `UpdateExecutor` — similar; if RID changes, update indexes.
23. `NestedLoopJoinExecutor`.
24. `HashJoinExecutor`.
25. `AggregationExecutor` (GROUP BY + COUNT/SUM/MIN/MAX/AVG; build hash table in `init`).
26. `SortExecutor` (in-memory for now).
27. `TopNExecutor` (heap of size N).

## Phase 4 — SQL frontend (only after executors work)

28. Add `sqlparser` crate. Parse `SELECT/INSERT/DELETE/UPDATE/CREATE TABLE/CREATE INDEX`.
29. Binder: resolve table/column names against Catalog; produce a bound statement with oids + column indices.
30. Planner: bound statement → plan tree. Start minimal: `SELECT * FROM t WHERE x = 5`. Expand.
31. Optimizer (last, optional for correctness): rule-based passes. Predicate pushdown. NLJ → HashJoin when join condition is equality. Pick IndexScan over SeqScan when predicate matches an index prefix.

## What to ignore

- **Transactions / MVCC / concurrency in executors** — that's project 4. Single-threaded, single-version world.
- **`Transaction*` / `LockManager*` / `LogManager*`** parameters in the bustub Catalog reference — drop them.
- **DataFusion as a whole** — only use `sqlparser` (the crate DataFusion uses internally).
- **Persisting the Catalog** — bustub catalog is non-persistent; ours can be too.
- **Variable-length types (VARCHAR)** — the existing `DataType` enum only has BOOLEAN/INT/TIMESTAMP and they're all inlined. Keep it that way for project 3.

## Files likely to be created/modified

- `src/table_heap.rs` — major rewrite (Phase 0).
- `src/catalog/catalog.rs` — likely split: keep schema/types here (or rename), add new `src/catalog/system_catalog.rs` for the table/index registry.
- `src/execution/` (new) — `executor.rs` (trait), `plan.rs` (plan node enum), `expression.rs`, plus one file per executor.
- `src/lib.rs` — wire new modules.
- `Cargo.toml` — add `sqlparser` only at Phase 4.

LOGS:
- Date: 2026-05-04
  - What changed: Created this task file. No code changes yet. Discussed architecture with user; decided on phased bottom-up approach starting with TableHeap, deferring SQL parsing until executors work.
  - Why: User asked for a clear plan before starting, to avoid trying to build several layers at once.
  - How validated: N/A (planning only).

ISSUES/ERRORS:
- (none yet)

## Concrete next step for the next agent

Start at Phase 0 step 1: design and document the `TablePage` byte layout, then implement `encode`/`decode` round-trip with a test. The current `insert_tuple` is misleading — it updates `tuples_meta` in memory but never writes the tuple's bytes into the page buffer, so nothing is actually persisted. Confirm the layout decision with the user before writing code if anything is ambiguous.
