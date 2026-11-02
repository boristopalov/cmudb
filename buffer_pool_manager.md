# Buffer Pool Manager

`BufferPoolManager` (`src/buffer_pool.rs`) caches disk pages in a fixed set of frames.
Callers get pages as RAII guards and never touch pin counts, dirty flags or the replacer.
The rules for holding guards are in `ARCHITECTURE.md`.

## State
- **Frames**: `Vec<RwLock<FrameHeader>>`. A frame holds a page buffer, its page id and a
  dirty flag. The frame's `RwLock` is the page latch.
- **Meta** (one `Mutex`): the page table (`page_id -> frame_id`), the free list, the ARC
  replacer, every frame's pin count, the next new page id and the deleted page ids waiting
  for reuse. Pins sit beside the replacer so a frame is marked evictable only while nothing
  pins it.
- **Disk scheduler**: a worker thread that runs page reads and writes; the pool waits on
  each request's reply.

## Guards
- `read_page(id)` returns a `PageRef`: one pin and the frame's read latch.
- `write_page(id)` returns a `PageMut`: one pin and the write latch, with the page marked
  dirty.
- `new_page()` allocates a page id, reusing the lowest deleted one if there is one, and
  returns a dirty `PageMut` for the caller to initialise.
- `delete_page(id)` drops an unpinned page from its frame and the replacer without writing
  it, frees its slot on disk and returns its id for reuse.
- Dropping either guard releases the latch, then the pin. A frame whose pin count reaches
  zero becomes evictable. Unpinning never flushes.

## Fetching a page
- **Hit**: pin the frame under the meta lock, release the meta lock, then take the frame
  latch. If the frame was given to another page in between, drop the pin and retry.
- **Miss**: take a frame from the free list, or ask the replacer for a victim and write it
  back first if it is dirty. Read the page from disk into the frame, map it and pin it. If
  the read fails, the victim is put back. The meta lock is held for the whole miss, both
  disk requests included, so misses run one at a time and hits and guard drops on other
  threads wait for them.
- With no free frame and nothing evictable, the fetch fails with `BpmError::NoFreeFrame`.
- `flush_page(id)` writes a resident page to disk and clears its dirty flag without
  changing its pins.
