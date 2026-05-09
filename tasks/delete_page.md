SUMMARY:
- Goal: Add a `delete_page(page_id)` primitive to `BufferPoolManager` that frees the in-memory frame AND reclaims the on-disk slot, so that throwaway pages (hash-join partitions, external merge-sort intermediates) don't leak buffer frames or grow the db file forever.
- Context: BusTub Project 3 explicitly requires that pages used for `HashJoinExecutor` partitions and `ExternalMergeSortExecutor` intermediate result pages are *actually deleted* after use. Today we have no delete primitive: `BpmError::FramePinned` exists but no caller, and `DiskManager` only exposes read/write. Without delete, dirty throwaway pages get flushed on eviction (wasted IO) and `DiskManager::page_lookup_table` / db-file slots leak monotonically.
- Outcome: A new `BufferPoolManager::delete_page(page_id) -> Result<(), BpmError>` that errors with `FramePinned` when the page is in use, otherwise frees the frame and reclaims the disk slot. `HashJoinExecutor::close` (and later `ExternalMergeSortExecutor::close`) call it for every page they own. Existing call sites of `read_page`/`write_page`/`flush_page` are untouched.

PLAN:
1. Replace `DiskRequest::is_write: bool` with `op: DiskOp` enum (`Read | Write | Delete`). Update the worker loop in `disk.rs` to dispatch on the variant. Update the four `DiskRequest { ... }` literals in `buffer_pool.rs` (`read_page` miss, `write_page` miss, `get_free_frame` flush, `flush_page`) — mechanical rename `is_write: false` → `op: DiskOp::Read`, `is_write: true` → `op: DiskOp::Write`. No public method signatures on `DiskScheduler`/`DiskManager` change.
2. Add `DiskManager::delete_page(id, data) -> io::Result<PageData>`:
   - Lock `page_lookup_table` and `free_slots`.
   - If `id` is present, remove its offset from the table and push it onto `free_slots`. Bump `num_deletes`.
   - If absent, log a warning and return Ok (lenient — page may have been created in memory and never flushed). Do not return NotFound; deletion of a never-written page is a normal path for executor cleanup.
   - Return the same `data` buffer back through the promise (matches the existing `Result<PageData, io::Error>` promise type — no change to channel signature).
3. Add `BufferPoolManager::delete_page(self: &Arc<Self>, page_id: usize) -> Result<(), BpmError>`:
   - Lock `meta`. Look up `frame_id` in `page_table`.
   - If found: take `frames[frame_id].write()`. If `pin_count > 0` → return `BpmError::FramePinned` without touching disk. Else: `frame.clear()`, `meta.page_table.remove(&page_id)`, `meta.free_list.push(frame_id)`, `meta.replacer.set_evictable(frame_id, false)` (replacer has no `remove`; non-evictable + lingering history entry is acceptable — frame_id will rebind on next `record_access`).
   - Drop frame guard, drop `meta`.
   - Issue a `DiskOp::Delete` request to the scheduler, await the promise. Map errors to `BpmError::ChannelClosed` / `BpmError::Io` exactly as `flush_page` does today.
   - If the page wasn't in `page_table` (cold delete), skip the in-memory work and go straight to the disk delete.
4. Wire up `HashJoinExecutor::close` to delete its partition pages:
   - For each heap in `left_partitions` and `right_partitions`, walk the page chain (`first_page_id` → follow `TablePage::next_page_id` until 0) collecting page IDs, then call `bpm.delete_page(pid)` for each.
   - Walking the chain requires reading each page once via `bpm.read_page`. Drop the read guard before deleting (otherwise pin_count > 0 → `FramePinned`).
   - Swallow individual `delete_page` errors with a `warn!` — `close` shouldn't fail the whole query if a single page can't be reclaimed. Errors here are best-effort cleanup.
5. Tests:
   - Unit test on BPM: create a page, drop the guard, `delete_page` it, assert subsequent `read_page` returns an error (NotFound from disk) AND that the frame is back on the free list (indirectly: allocate `num_frames` more pages successfully).
   - Unit test on BPM: create a page, hold the guard, attempt `delete_page` → assert `BpmError::FramePinned`. Drop guard, retry, assert Ok.
   - Unit test on `DiskManager`: write a page, delete it, write a different page → assert the second write reuses the freed slot (compare `db_file` length stays constant).
   - Integration: existing hash join tests should still pass. Add an assertion in one test that the db file size after the join is no larger than before (modulo the inputs themselves).

NOTES ON SCOPE / NON-GOALS:
- Public signatures of `DiskScheduler::schedule`, `DiskManager::read_page`/`write_page`, `BufferPoolManager::{read_page, write_page, flush_page, new_page}` are unchanged. The only breaking type change is the field rename inside `DiskRequest` (`is_write` → `op`), and `DiskRequest` is only constructed by the BPM, so blast radius is the four literals listed in step 1.
- We deliberately do NOT add a `remove(frame_id)` method to `ArcReplacer` in this task. The lingering ghost entry is correct — when the frame is reused, `record_access` rebinds it. If profiling later shows the ghost-list bloat matters, that's a separate task.
- We do NOT support deleting a page that another thread might still be about to read. Caller is expected to own the page IDs (hash join / merge sort own their partition / spill pages). Concurrent access to a deleted page is undefined behavior at the executor level.

LOGS:
- Date:
  - What changed:
  - Why:
  - How validated (tests run + results):

ISSUES/ERRORS:
- Problem:
  - Symptoms:
  - Root cause (if known):
  - Fix / workaround:
  - Follow-ups:
