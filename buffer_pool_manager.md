# Buffer Pool Manager

The buffer pool manager (BPM) mediates access to database pages between memory and disk. It owns the in-memory frames, coordinates pin/unpin, eviction, and integrates with the disk scheduler and ARC-based replacer. Callers never manipulate latches or dirty/evictable flags directly; they interact through sessions and page handles.

## Architecture
- **Frames**: Fixed-size slots (`FrameHeader`) holding a page buffer, page_id, pin count, and dirty flag. Stored as a `Vec<RwLock<FrameHeader>>`.
- **Page table**: Maps `page_id -> frame_id` so the BPM can find cached pages.
- **Replacer**: ARC policy tracks recency/frequency and chooses victims among evictable (unpinned) frames.
- **Disk scheduler**: Asynchronous request queue to read/write pages; BPM schedules I/O when pages are missing or need flushing.
- **Free list**: Pool of unused frames; used before eviction.

## Page Handles
- **PageRef/PageMut**: Lightweight RAII guards holding a mapped lock guard to the frame. 

## Lifecycle and Operations
- **Pin (read)**: Look up in page table; on hit, pin_count++, mark nonevictable, return `PageRef`. On miss, get a free/evictable frame, load from disk via scheduler, install mapping, pin_count=1, nonevictable, return `PageRef`.
- **Pin (write)**: Same as read, but mark dirty and return `PageMut`. Write path assumes the page exists; new pages come from `new_page`.
- **New page**: Allocates a page_id, reserves a frame (free/evictable), clears it, marks dirty and pinned, installs mapping, nonevictable, returns `PageMut` to initialize.
- **Unpin**: Decrements pin_count; when it reaches zero, marks the frame evictable in the replacer. Flush is not automatic on unpin.
- **Flush page**: Schedules a write of the frame buffer to disk and clears the dirty flag. Does not change pinning/evictable state.
- **Eviction**: When no free frame is available, the replacer selects an evictable frame. If dirty, the BPM flushes it before reuse, removes its mapping, clears metadata, and rebinds it to the new page.
