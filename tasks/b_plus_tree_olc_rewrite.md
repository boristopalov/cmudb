SUMMARY:
- Goal: Replace the current half-baked concurrency layer in `src/index/b_plus_tree.rs` with a Context-based Optimistic Latch Coupling (OLC) implementation that matches the CMU 15-445 Fall 2025 Project 2 spec (https://15445.courses.cs.cmu.edu/fall2025/project2/).
- Context: The project was left after concurrency tests started hanging. Diagnosis (see ANALYSIS below) shows the current code mixes three protocols (OLFIT-style version word, ad-hoc path-of-tuples, BPM frame RwLock). The CMU project explicitly mandates OLC, which is structurally different from what the code is trying to do.
- Outcome: A single, coherent OLC implementation, with the in-page latch word and `Vec<(PageId, usize)>` path machinery deleted. `b_plus_tree_latches_soak_concurrent` integration test no longer hangs.

---

## ANALYSIS (state at start of this task)

### Concurrency issues in `b_plus_tree.rs` (in priority order)

1. **Path-of-tuples is fundamentally unsound.** `find_leaf_and_path_optimistic` does a lock-free descent and returns `Vec<(parent_id, child_idx)>`. The writer climbs back up taking write latches, but by then the parent may have split/merged and `child_idx` no longer points at the page descended through. Used in `parent.insert_separator(child_idx, …)` and `parent.remove_separator(child_idx)` → silent corruption.
2. **Empty-tree insert race** (lines 588-612, already TODO'd in code). Multiple threads each see `root_page_id == INVALID`, each allocate a leaf, last writer wins.
3. **Header has no read-side version check.** `write_header` takes `WriteLatchGuard`; `decode_header` ignores it. Reader can pick up a stale `root_page_id` mid-update.
4. **Root contraction at end of `remove`** (lines 1248-1270) runs without the header latch and then calls `rebuild_internal_separator_keys`, which write-latches the entire internal index. Probable source of the hang under `num_frames = 2`.
5. **`rebuild_internal_separator_keys` is a global O(internal nodes) write-latch walk per remove.** Exists only because merges happen without the parent latched, so separators have to be patched up later.
6. **No backoff in optimistic restart loops** (`is_locked` → `continue 'restart`). Spin-without-yield = livelock under contention.
7. **`new_page` exposes a not-yet-initialized page** (BPM inserts into `page_table` before caller writes contents). Other readers can decode garbage.
8. **`LatchMap` in `lock.rs:14-44`** is dead code that pretends to lock.

### Why these aren't inevitable B+ tree bugs

They're symptoms of mixing three protocols:
- **OLFIT-style** in-page version word (`lock.rs::WriteLatchGuard`) — meant for lock-free reads + version validation.
- **Ad-hoc path-of-tuples** descent — a single-threaded artifact carried into concurrent code.
- **BPM frame `RwLock`** — provides physical exclusion, but the rest of the code acts as if it doesn't.

The CMU project specifies a fourth protocol (OLC) that the code is loosely shaped like but doesn't actually implement. Aligning to OLC eliminates whole categories of bug rather than fixing them one at a time.

---

## TARGET PROTOCOL: OLC (per CMU 15-445 Fall 2025 Project 2)

OLC = "optimistic latch coupling / crabbing." Two-phase:

### Phase 1: Optimistic attempt (insert / remove / search)
- Read-latch the header, snapshot `root_page_id`, release header.
- Read-couple top-down: latch child, immediately drop parent. (≤2 latches held at any moment during descent.)
- At leaf: for search, read-latch and read; for insert/remove, *write*-latch.
- Check if the leaf is **safe**:
  - insert safe: `size + 1 <= max_size` (won't split)
  - remove safe: `size > min_size` (won't underflow)
- If safe: do the op, release the leaf, done.
- If not safe: drop everything, fall through to phase 2.

### Phase 2: Pessimistic fallback (insert / remove only)
- Take the **header write latch first** (because root might change).
- Write-latch top-down. After latching each new child, check if child is safe:
  - safe → release ALL ancestors (and header) above this child
  - not safe → keep ancestors latched
- At leaf: do the op. If it splits/merges, propagate up using the still-latched ancestors in `Context::write_set` (the parent is `write_set[len-2]` after popping the leaf). Sibling lookups go through the parent's `children[idx ± 1]` and `bpm.write_page` to add siblings to the set.
- If propagation reaches the top and `write_set` is empty, the held header guard lets you create a new root atomically.

### "Only 2 latches at once" — partial truth

Holds for read descent and for pessimistic descent through safe nodes. Does NOT hold when intermediate nodes are unsafe — then the entire unsafe ancestor chain must be held simultaneously, because the propagating split/merge modifies all of them as one logical op. The "optimistic" in OLC refers to the *statistical* property: most ops don't propagate, so most ops hold ≤2 latches.

---

## TARGET DESIGN

### `Context` type (replaces the path-of-tuples)

```rust
pub struct Context<'a> {
    pub header: Option<PageMut<'a>>,        // Some only when root may change (pessimistic mode)
    pub root_page_id: PageId,               // snapshot at op start
    pub read_set: Vec<PageRef<'a>>,         // ancestors during optimistic descent
    pub write_set: Vec<PageMut<'a>>,        // ancestors + leaf during pessimistic descent
}
```

Key idea: **the path is a stack of held latches, not a snapshot of indices**. Holding the guard prevents anyone else from modifying that page, so `child_idx` into the parent is valid by construction.

If `Context<'a>` lifetimes get gnarly (likely, since `PageMut`/`PageRef` borrow from the BPM), the easiest workaround is to make the guards own an `Arc<BufferPoolManager>` and become effectively `'static`. BusTub takes this shape.

### Safe predicates

```rust
fn safe_for_insert(size: u32, max_size: u32) -> bool { size + 1 <= max_size }
fn safe_for_remove(size: u32, min_size: u32) -> bool { size > min_size }
```

Apply to the **child you just latched**, not to the parent.

### Descent skeletons

See the "Sketch" section in the conversation immediately preceding this file's creation. Two functions:
- `descend_optimistic(key, op) -> Result<PageMut /* leaf */, Restart>` — read-couples down, returns write-latched leaf if safe.
- `descend_pessimistic(key, op, &mut Context)` — write-latches top-down, populating the Context.

### Top-level flow

```rust
fn insert(&self, key, val) -> IndexResult {
    if let Ok(leaf) = self.descend_optimistic(&key, Op::Insert) {
        // happy path: just update the leaf
        return self.commit_leaf_insert(leaf, key, val);
    }
    let mut ctx = Context::new();
    self.descend_pessimistic(&key, Op::Insert, &mut ctx)?;
    self.do_insert_with_propagation(key, val, &mut ctx)
}
```

Same shape for remove.

---

## WHAT TO DELETE

- `src/index/lock.rs::WriteLatchGuard` and the entire in-page latch word concept. Under OLC the BPM frame `RwLock` is the only lock; in-page version validation is for OLFIT, which we are not implementing. Bonus: removes transient lock state from the on-disk page format.
- `find_leaf_and_path_optimistic` in `b_plus_tree.rs`.
- The `Vec<(PageId, usize)>` path threaded through `insert` / `remove`.
- `rebuild_internal_separator_keys` — separator updates fold into merge/redistribute primitives where (parent, child, sibling) are all in `write_set`.
- The `_latch` field threaded through `encode`/`decode` for every page kind. (Once removed, update `b_plus_page.rs` codecs and `page_codec.rs` `COMMON_HEADER_BYTES` / `PAGE_LATCH_BYTES`.)

---

PLAN:

1. **Define `Context` and safe predicates** in a new module (e.g. `src/index/context.rs` or inline at top of `b_plus_tree.rs`). Include `Op` enum and `Restart` error.
3. **Port `search` to OLC** (read-couple descent, no restart needed). Verify search-only tests still pass.
4. **Port `insert` to OLC**: write `descend_optimistic` and the happy-path commit; then write `descend_pessimistic` and propagation. Handle empty-tree case entirely inside the pessimistic path (header write latch is held, recheck `root_page_id`, allocate, write header, done — no race).
5. **Port `remove` to OLC**: same shape. Fold separator updates into merge/redistribute (where parent + sibling are in `write_set`).
6. **Delete dead code** listed in "WHAT TO DELETE" above.
7. **Strip the in-page latch from the page format.** Update `COMMON_HEADER_BYTES` / `PAGE_LATCH_BYTES`, page codecs, and any tests that read/assert on the latch word (notably `tests/b_plus_tree_integration.rs::assert_latch_well_formed` — replace with something else, e.g. a kind-byte check).
8. **Run the soak test.** `cargo test --test b_plus_tree_integration b_plus_tree_latches_soak_concurrent -- --nocapture`. Confirm it no longer hangs.

---

LOGS:

- Date: 2026-05-02
  - What changed: Wrote this task file. No code changes yet.
  - Why: Captured prior conversation's analysis + agreed direction so the next agent can pick up cold.
  - How validated (tests run + results): N/A.

- Date: 2026-05-02
  - What changed: New search() implemented and tested
  - Why: search() is in-line with cmu OLC
  - How validated (tests run + results): tests passed.

- Date: 2026-05-03
  - What changed: New insert() implemented and tested
  - Why: insert() is in-line with cmu OLC
  - How validated (tests run + results): tests passed.

- Date: 2026-05-03
  - What changed: New remove() implemented and tested
  - Why: remove() is in-line with cmu OLC
  - How validated (tests run + results): unit tests and integration tests passed.

- Date: 2026-05-03
  - What changed: code to delete has been deleted.
  - Why: not needed
  - How validated (tests run + results): unit tests and integration tests passed.




---

ISSUES/ERRORS:
(none yet — fill in as you go)
**RESOLVED**: Concurrency integration test continues to deadlock. However, this time we consisently deadlock at the same spot, rather than sporadically, so the root cause is likely not a race but a logic bug/forgetting to drop something.

---

## REFERENCES

- CMU 15-445 Fall 2025 Project 2: https://15445.courses.cs.cmu.edu/fall2025/project2/
- Key constraints from the project page:
  - "Use the *optimistic* latch coupling/crabbing technique"
  - "Never acquire the same read latch twice in a single thread"
  - "Release latches in the same order (from the header page to the bottom)"
  - "Store the root page id in the context and acquire write guard of header page when modifying the B+Tree"
  - "To find a parent of the current node, look at the back of `write_set_`"
  - "We do not test your iterator for thread-safe leaf scans" (so `LeafPagesIter` doesn't need to be concurrency-safe)
- Conceptual distinction worth keeping in mind: "optimistic" in CMU OLC means "optimistically try the read-coupling descent, fall back to pessimistic write-latch descent if the leaf is unsafe." It does NOT mean OLFIT-style version validation. Mixing the two was the original mistake.
