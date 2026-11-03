use crate::disk;
use crate::disk::DiskRequest;
use crate::disk::PAGE_SIZE;
use crate::ids::{FrameId, PageId, PageIdOutOfRange};
use crate::replacer;
use log::{info, warn};
#[cfg(feature = "deadlock_detection")]
use parking_lot::deadlock;
#[cfg(feature = "deadlock_detection")]
use std::sync::Once;
#[cfg(feature = "deadlock_detection")]
use std::time::Duration;

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::fmt;
use std::io;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::thread;

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

#[cfg(feature = "deadlock_detection")]
const DEADLOCK_POLL_INTERVAL: Duration = Duration::from_secs(1);

type Page = [u8; disk::PAGE_SIZE];

#[derive(Debug)]
struct FrameHeader {
    frame_id: FrameId,
    page_id: Option<PageId>,
    buf: Page,
    dirty: AtomicBool,
}

#[derive(Debug)]
struct EvictedPage {
    page_id: PageId,
    data: Page,
}

#[derive(Debug)]
struct FrameReservation {
    frame_id: FrameId,
    evicted: Option<EvictedPage>,
}

#[derive(Debug)]
struct Meta {
    replacer: replacer::ArcReplacer,
    free_list: Vec<FrameId>,
    page_table: HashMap<PageId, FrameId>,
    /// Pins per frame. Kept beside the replacer under one lock, so a frame is marked
    /// evictable only while nothing pins it.
    pins: Vec<u32>,
    /// One past the highest page id handed out.
    next_page_id: usize,
    /// Deleted page ids below `next_page_id`, handed out again lowest first.
    free_page_ids: BTreeSet<PageId>,
}

impl Meta {
    fn allocate_page_id(&mut self) -> Result<PageId, BpmError> {
        if let Some(page_id) = self.free_page_ids.pop_first() {
            return Ok(page_id);
        }
        let page_id = PageId::try_from(self.next_page_id).map_err(BpmError::PageIdsExhausted)?;
        self.next_page_id += 1;
        Ok(page_id)
    }

    fn pins(&mut self, frame_id: FrameId) -> &mut u32 {
        &mut self.pins[frame_id.0]
    }
}

/// Which page ids a pool has handed out: every id below `next_page_id` except
/// `free_page_ids`, which were deleted and are handed out again first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageAllocation {
    pub next_page_id: usize,
    pub free_page_ids: Vec<PageId>,
}

#[derive(Debug)]
pub struct BufferPoolManager {
    frames: Vec<RwLock<FrameHeader>>,
    meta: Mutex<Meta>,
    scheduler: Mutex<disk::DiskScheduler>,
}

#[cfg(feature = "deadlock_detection")]
fn start_deadlock_detector() {
    static START: Once = Once::new();
    START.call_once(|| {
        std::thread::spawn(|| {
            loop {
                std::thread::sleep(DEADLOCK_POLL_INTERVAL);
                let deadlocks = deadlock::check_deadlock();
                if deadlocks.is_empty() {
                    eprintln!("no deadlocks!");
                    continue;
                }
                eprintln!("{} deadlocks detected", deadlocks.len());
                for (i, threads) in deadlocks.iter().enumerate() {
                    eprintln!("Deadlock #{}", i);
                    for t in threads {
                        eprintln!("Thread Id {:#?}", t.thread_id());
                        eprintln!("{:#?}", t.backtrace());
                    }
                }
            }
        });
    });
}

#[derive(Debug)]
pub struct PageRef<'a> {
    frame_guard: ManuallyDrop<RwLockReadGuard<'a, FrameHeader>>,
    frame_id: FrameId,
    page_id: PageId,
    bpm: Arc<BufferPoolManager>,
}

#[derive(Debug)]
pub struct PageMut<'a> {
    frame_guard: ManuallyDrop<RwLockWriteGuard<'a, FrameHeader>>,
    page_id: PageId,
    bpm: Arc<BufferPoolManager>,
}

impl<'a> PageRef<'a> {
    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn data(&self) -> &[u8] {
        &self.frame_guard.buf
    }

    pub fn page_id(&self) -> PageId {
        self.page_id
    }
}

impl<'a> Drop for PageRef<'a> {
    fn drop(&mut self) {
        let frame_id = self.frame_guard.frame_id;
        // SAFETY: `frame_guard` is dropped exactly once, here in `Drop::drop`, and is never read afterwards.
        unsafe { ManuallyDrop::drop(&mut self.frame_guard) };
        self.bpm.release_pin(frame_id);
    }
}

impl<'a> PageMut<'a> {
    pub fn frame_id(&self) -> FrameId {
        self.frame_guard.frame_id
    }

    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    pub fn data(&self) -> &[u8] {
        &self.frame_guard.buf
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.frame_guard.buf
    }
}

impl<'a> Drop for PageMut<'a> {
    fn drop(&mut self) {
        let frame_id = self.frame_guard.frame_id;
        // SAFETY: `frame_guard` is dropped exactly once, here in `Drop::drop`, and is never read afterwards.
        unsafe { ManuallyDrop::drop(&mut self.frame_guard) };
        self.bpm.release_pin(frame_id);
    }
}

#[derive(Debug)]
pub enum BpmError {
    NoFreeFrame,
    /// The frame id is past the end of the pool's frames.
    FrameOutOfRange {
        frame_id: FrameId,
    },
    /// A flush asked for a page that is not in any frame.
    PageNotCached {
        page_id: PageId,
    },
    ChannelClosed,
    Io(io::Error),
    FrameInUse,
    /// A delete asked for a page that a live guard still pins.
    PagePinned {
        page_id: PageId,
    },
    /// A delete asked for a page that is in no frame and not on disk.
    NoSuchPage {
        page_id: PageId,
    },
    /// [`BufferPoolManager::resume_allocation`] on a pool that already handed out pages.
    PoolInUse,
    /// Every page id up to [`PageId::MAX`] is in use.
    PageIdsExhausted(PageIdOutOfRange),
}

impl fmt::Display for BpmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpmError::NoFreeFrame => {
                write!(f, "No free frame available")
            }
            BpmError::Io(e) => {
                write!(f, "io error: {}", e)
            }
            BpmError::FrameOutOfRange { frame_id } => {
                write!(f, "Frame {frame_id} is out of range")
            }
            BpmError::PageNotCached { page_id } => {
                write!(f, "Page {page_id} is not in the buffer pool")
            }
            BpmError::ChannelClosed => write!(f, "Channel closed"),
            BpmError::FrameInUse => write!(f, "Frame in use"),
            BpmError::PagePinned { page_id } => {
                write!(f, "Page {page_id} is pinned and cannot be deleted")
            }
            BpmError::NoSuchPage { page_id } => {
                write!(f, "Page {page_id} is in no frame and not on disk")
            }
            BpmError::PoolInUse => {
                write!(f, "the buffer pool already handed out pages")
            }
            BpmError::PageIdsExhausted(e) => write!(f, "no page id left to hand out: {e}"),
        }
    }
}

type PageRefResult<'a> = Result<PageRef<'a>, BpmError>;
type PageMutResult<'a> = Result<PageMut<'a>, BpmError>;

impl FrameHeader {
    pub fn clear(&mut self) {
        self.dirty.store(false, Ordering::Release);
        self.page_id = None;
    }
}

impl BufferPoolManager {
    fn tid() -> String {
        format!("{:?}", thread::current().id())
    }

    pub fn new(
        num_frames: usize,
        replacer: replacer::ArcReplacer,
        scheduler: disk::DiskScheduler,
    ) -> Self {
        assert_eq!(
            replacer.capacity(),
            num_frames,
            "replacer capacity differs from the number of buffer pool frames"
        );
        #[cfg(feature = "deadlock_detection")]
        start_deadlock_detector();
        let mut frames = Vec::with_capacity(num_frames);
        let mut free_list = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            frames.push(RwLock::new(FrameHeader {
                frame_id: FrameId(i),
                page_id: None,
                buf: [0u8; disk::PAGE_SIZE],
                dirty: AtomicBool::new(false),
            }));
            free_list.push(FrameId(i));
        }

        Self {
            frames,
            meta: Mutex::new(Meta {
                replacer,
                free_list,
                page_table: HashMap::with_capacity(num_frames),
                pins: vec![0; num_frames],
                next_page_id: 0,
                free_page_ids: BTreeSet::new(),
            }),
            scheduler: Mutex::new(scheduler),
        }
    }

    pub fn new_page(self: &Arc<Self>) -> PageMutResult<'_> {
        info!("tid={} new_page requested", Self::tid());
        let mut meta = self.meta.lock();
        let reservation = self.get_free_frame(&mut meta)?;
        let frame_id = reservation.frame_id;
        let page_id = match meta.allocate_page_id() {
            Ok(page_id) => page_id,
            Err(e) => {
                self.restore_frame(&mut meta, reservation);
                return Err(e);
            }
        };
        let mut frame = self.frame(frame_id)?.write();
        frame.clear();
        frame.page_id = Some(page_id);
        frame.dirty.store(true, Ordering::Release);
        *meta.pins(frame_id) = 1;
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, false);

        Ok(PageMut {
            frame_guard: ManuallyDrop::new(frame),
            page_id,
            bpm: Arc::clone(self),
        })
    }

    /// Deletes `page_id`: drops it from its frame, if it is in one, without writing it, and
    /// frees its slot on disk for a later page. Fails with [`BpmError::PagePinned`] while a
    /// guard holds the page, and with [`BpmError::NoSuchPage`] for a page that is unallocated
    /// or already deleted. [`BufferPoolManager::new_page`] hands the id out again.
    pub(crate) fn delete_page(&self, page_id: PageId) -> Result<(), BpmError> {
        let mut meta = self.meta.lock();
        let resident = match meta.page_table.get(&page_id).copied() {
            Some(frame_id) if meta.pins[frame_id.0] > 0 => {
                return Err(BpmError::PagePinned { page_id });
            }
            Some(frame_id) => {
                self.frame(frame_id)?.write().clear();
                meta.page_table.remove(&page_id);
                meta.free_list.push(frame_id);
                true
            }
            None => false,
        };
        let on_disk = self
            .scheduler
            .lock()
            .disk_manager()
            .deallocate_page(page_id);
        if !resident && !on_disk {
            return Err(BpmError::NoSuchPage { page_id });
        }
        meta.replacer.remove(page_id);
        meta.free_page_ids.insert(page_id);
        info!("tid={} delete_page page_id={page_id}", Self::tid());
        Ok(())
    }

    /// The disk manager behind this pool's disk scheduler.
    pub fn disk_manager(&self) -> Arc<disk::DiskManager> {
        self.scheduler.lock().disk_manager()
    }

    /// Returns how many pages this pool has handed out and not deleted.
    pub fn live_page_count(&self) -> usize {
        let meta = self.meta.lock();
        meta.next_page_id - meta.free_page_ids.len()
    }

    pub fn write_page(self: &Arc<Self>, page_id: PageId) -> PageMutResult<'_> {
        let mut meta = loop {
            let mut meta = self.meta.lock();
            let Some(frame_id) = meta.page_table.get(&page_id).copied() else {
                break meta;
            };
            info!(
                "tid={} write_page cache hit page_id={page_id} frame_id={frame_id}",
                Self::tid()
            );
            meta.replacer.record_access(frame_id, page_id);
            meta.pins[frame_id.0] += 1;
            meta.replacer.set_evictable(frame_id, false);
            drop(meta);

            let frame = self.frame(frame_id)?.write();
            let frame_was_reassigned_after_lookup = frame.page_id != Some(page_id);
            if frame_was_reassigned_after_lookup {
                drop(frame);
                self.release_pin(frame_id);
                continue;
            }
            frame.dirty.store(true, Ordering::Release);
            return Ok(PageMut {
                frame_guard: ManuallyDrop::new(frame),
                page_id,
                bpm: Arc::clone(self),
            });
        };

        info!(
            "tid={} write_page cache miss page_id={page_id}",
            Self::tid()
        );
        let frame = self.load_into_free_frame(&mut meta, page_id, true)?;
        Ok(PageMut {
            frame_guard: ManuallyDrop::new(frame),
            page_id,
            bpm: Arc::clone(self),
        })
    }

    pub fn read_page(self: &Arc<Self>, page_id: PageId) -> PageRefResult<'_> {
        info!("tid={} read_page page_id={page_id}", Self::tid());
        let mut meta = loop {
            let mut meta = self.meta.lock();
            let Some(frame_id) = meta.page_table.get(&page_id).copied() else {
                break meta;
            };
            info!(
                "tid={} read_page cache hit page_id={page_id} frame_id={frame_id}",
                Self::tid()
            );
            meta.replacer.record_access(frame_id, page_id);
            meta.pins[frame_id.0] += 1;
            meta.replacer.set_evictable(frame_id, false);
            drop(meta);

            let frame = self.frame(frame_id)?.read();
            let frame_was_reassigned_after_lookup = frame.page_id != Some(page_id);
            if frame_was_reassigned_after_lookup {
                drop(frame);
                self.release_pin(frame_id);
                continue;
            }
            return Ok(PageRef {
                frame_guard: ManuallyDrop::new(frame),
                frame_id,
                page_id,
                bpm: Arc::clone(self),
            });
        };

        info!("tid={} read_page cache miss page_id={page_id}", Self::tid());
        let frame = self.load_into_free_frame(&mut meta, page_id, false)?;
        let frame_id = frame.frame_id;
        Ok(PageRef {
            frame_guard: ManuallyDrop::new(RwLockWriteGuard::downgrade(frame)),
            frame_id,
            page_id,
            bpm: Arc::clone(self),
        })
    }

    fn frame(&self, frame_id: FrameId) -> Result<&RwLock<FrameHeader>, BpmError> {
        self.frames
            .get(frame_id.0)
            .ok_or(BpmError::FrameOutOfRange { frame_id })
    }

    fn load_into_free_frame(
        &self,
        meta: &mut Meta,
        page_id: PageId,
        dirty: bool,
    ) -> Result<RwLockWriteGuard<'_, FrameHeader>, BpmError> {
        let reservation = self.get_free_frame(meta)?;
        let frame_id = reservation.frame_id;
        let mut frame = self.frame(frame_id)?.write();
        info!(
            "tid={} reserved frame_id={frame_id} for page_id={page_id}",
            Self::tid()
        );

        let page_data = match self.read_from_disk(page_id) {
            Ok(data) => data,
            Err(e) => {
                drop(frame);
                self.restore_frame(meta, reservation);
                return Err(e);
            }
        };

        frame.buf = *page_data;
        frame.page_id = Some(page_id);
        *meta.pins(frame_id) = 1;
        frame.dirty.store(dirty, Ordering::Release);
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, false);
        Ok(frame)
    }

    fn read_from_disk(&self, page_id: PageId) -> Result<disk::PageData, BpmError> {
        self.run_disk_request(page_id, false, Box::new([0u8; PAGE_SIZE]))
    }

    fn write_to_disk(&self, page_id: PageId, page: &Page) -> Result<(), BpmError> {
        self.run_disk_request(page_id, true, Box::new(*page))
            .map(drop)
    }

    fn run_disk_request(
        &self,
        page_id: PageId,
        is_write: bool,
        data: disk::PageData,
    ) -> Result<disk::PageData, BpmError> {
        let (tx, rx) = mpsc::channel::<Result<disk::PageData, io::Error>>();
        let request = DiskRequest {
            page_id,
            is_write,
            data,
            promise: tx,
        };
        self.scheduler
            .lock()
            .schedule(vec![request])
            .map_err(|_| BpmError::ChannelClosed)?;
        match rx.recv() {
            Ok(Ok(data)) => Ok(data),
            Ok(Err(e)) => Err(BpmError::Io(e)),
            Err(_) => Err(BpmError::ChannelClosed),
        }
    }

    fn get_free_frame(&self, meta: &mut Meta) -> Result<FrameReservation, BpmError> {
        if let Some(frame_id) = meta.free_list.pop() {
            info!(
                "tid={} get_free_frame: using free_list frame_id={frame_id}",
                Self::tid()
            );
            return Ok(FrameReservation {
                frame_id,
                evicted: None,
            });
        }

        info!(
            "tid={} get_free_frame: no free frames, attempting eviction",
            Self::tid()
        );
        let Some(eviction) = meta.replacer.begin_eviction() else {
            return Err(BpmError::NoFreeFrame);
        };
        let frame_id = eviction.frame_id();
        info!(
            "tid={} get_free_frame: replacer returned victim frame_id={frame_id}",
            Self::tid()
        );

        if meta.pins[frame_id.0] > 0 {
            warn!(
                "tid={} get_free_frame: attempted to evict pinned frame_id={frame_id}",
                Self::tid()
            );
            drop(eviction);
            meta.replacer.set_evictable(frame_id, false);
            return Err(BpmError::FrameInUse);
        }
        let mut frame = self.frame(frame_id)?.write();

        let Some(page_id) = frame.page_id else {
            eviction.commit();
            return Ok(FrameReservation {
                frame_id,
                evicted: None,
            });
        };

        if frame.dirty.load(Ordering::Acquire) {
            info!(
                "tid={} get_free_frame: flushing dirty page_id={page_id} before eviction",
                Self::tid()
            );
            if let Err(e) = self.write_to_disk(page_id, &frame.buf) {
                warn!(
                    "tid={} get_free_frame: flush of page_id={page_id} failed, keeping it in frame_id={frame_id}: {e}",
                    Self::tid()
                );
                return Err(e);
            }
        }

        eviction.commit();
        meta.page_table.remove(&page_id);
        let data = frame.buf;
        frame.clear();
        Ok(FrameReservation {
            frame_id,
            evicted: Some(EvictedPage { page_id, data }),
        })
    }

    fn restore_frame(&self, meta: &mut Meta, reservation: FrameReservation) {
        let frame_id = reservation.frame_id;
        let Some(EvictedPage { page_id, data }) = reservation.evicted else {
            meta.free_list.push(frame_id);
            return;
        };

        info!("restore_frame: restoring evicted page_id={page_id} into frame_id={frame_id}");
        let evicted_page_was_reloaded_elsewhere = meta.page_table.contains_key(&page_id);
        if evicted_page_was_reloaded_elsewhere {
            info!("restore_frame: page_id={page_id} already mapped, freeing frame_id={frame_id}");
            meta.free_list.push(frame_id);
            return;
        }
        let Ok(frame_lock) = self.frame(frame_id) else {
            meta.free_list.push(frame_id);
            return;
        };

        let mut frame = frame_lock.write();
        frame.page_id = Some(page_id);
        frame.buf = data;
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, true);
    }

    fn release_pin(&self, frame_id: FrameId) {
        let mut meta = self.meta.lock();
        match meta.pins(frame_id).checked_sub(1) {
            Some(remaining) => {
                *meta.pins(frame_id) = remaining;
                if remaining == 0 {
                    meta.replacer.set_evictable(frame_id, true);
                }
            }
            None => warn!("tried to unpin a frame with 0 pins frame_id={frame_id}"),
        }
    }

    /// Returns one past the highest page id this pool has handed out; ids run densely
    /// from 0, and [`BufferPoolManager::new_page`] hands out deleted ids again before it
    /// grows this count.
    pub fn allocated_page_count(&self) -> usize {
        self.meta.lock().next_page_id
    }

    /// Reports whether `page_id` currently sits in a frame, so a read of it would not touch disk.
    pub fn is_resident(&self, page_id: PageId) -> bool {
        self.meta.lock().page_table.contains_key(&page_id)
    }

    /// Makes the next `count` page writes to disk fail with an injected I/O error.
    #[cfg(any(test, feature = "testing"))]
    pub fn fail_next_disk_writes(&self, count: u32) {
        self.scheduler.lock().disk_manager().fail_next_writes(count);
    }

    /// The ids [`BufferPoolManager::new_page`] has handed out so far, as one past the highest
    /// id and the deleted ids below it.
    pub fn allocation(&self) -> PageAllocation {
        let meta = self.meta.lock();
        PageAllocation {
            next_page_id: meta.next_page_id,
            free_page_ids: meta.free_page_ids.iter().copied().collect(),
        }
    }

    /// Whether `page_id` was handed out by [`BufferPoolManager::new_page`] and not deleted
    /// since.
    pub fn page_is_allocated(&self, page_id: PageId) -> bool {
        let meta = self.meta.lock();
        page_id.as_usize() < meta.next_page_id && !meta.free_page_ids.contains(&page_id)
    }

    /// Resumes handing out page ids where `allocation` left off, for a pool over a data file
    /// written by an earlier pool. Forgets the free pages on disk, so reading one fails as it
    /// did before. Fails with [`BpmError::PoolInUse`] once this pool has handed out a page.
    pub fn resume_allocation(&self, allocation: &PageAllocation) -> Result<(), BpmError> {
        let mut meta = self.meta.lock();
        if meta.next_page_id != 0 || !meta.free_page_ids.is_empty() {
            return Err(BpmError::PoolInUse);
        }
        let dm = self.scheduler.lock().disk_manager();
        for &page_id in &allocation.free_page_ids {
            dm.deallocate_page(page_id);
        }
        meta.next_page_id = allocation.next_page_id;
        meta.free_page_ids = allocation.free_page_ids.iter().copied().collect();
        Ok(())
    }

    /// Writes every dirty page in the pool to disk and waits until the data file is on the
    /// storage device. Returns how many pages it wrote. A page latched for writing is
    /// written once its guard drops.
    pub fn flush_all(&self) -> Result<usize, BpmError> {
        let mut written = 0;
        for frame_lock in &self.frames {
            let frame = frame_lock.read();
            let Some(page_id) = frame.page_id else {
                continue;
            };
            if frame.dirty.load(Ordering::Acquire) {
                self.write_to_disk(page_id, &frame.buf)?;
                frame.dirty.store(false, Ordering::Release);
                written += 1;
            }
        }
        if written > 0 {
            self.scheduler
                .lock()
                .disk_manager()
                .sync()
                .map_err(BpmError::Io)?;
        }
        Ok(written)
    }

    pub fn flush_page(&self, page_id: PageId) -> Result<(), BpmError> {
        info!("flush_page: flushing page page_id={page_id}");
        let Some(frame_id) = self.meta.lock().page_table.get(&page_id).copied() else {
            warn!("tried to flush page with page_id={page_id} but could not find its frame");
            return Err(BpmError::PageNotCached { page_id });
        };

        let frame = self.frame(frame_id)?.read();
        let frame_was_reassigned_after_lookup = frame.page_id != Some(page_id);
        if frame_was_reassigned_after_lookup {
            return Err(BpmError::PageNotCached { page_id });
        }
        self.write_to_disk(page_id, &frame.buf)?;
        frame.dirty.store(false, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::replacer::ArcReplacer;
    use crate::testing::pid;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[test]
    fn flush_all_writes_each_dirty_page_once_and_a_new_pool_reads_them_back() {
        let dir = tempfile::tempdir().unwrap();
        let pool = |frames| {
            let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).unwrap();
            Arc::new(BufferPoolManager::new(
                frames,
                ArcReplacer::new(frames),
                DiskScheduler::new(dm),
            ))
        };
        let bpm = pool(4);
        for fill in 0..3u8 {
            bpm.new_page().unwrap().data_mut().fill(fill + 1);
        }
        bpm.delete_page(pid(1)).unwrap();
        assert_eq!(
            bpm.flush_all().unwrap(),
            2,
            "the deleted page is not written"
        );
        assert_eq!(
            bpm.flush_all().unwrap(),
            0,
            "nothing is dirty after a flush"
        );
        let allocation = bpm.allocation();
        assert_eq!(
            allocation,
            PageAllocation {
                next_page_id: 3,
                free_page_ids: vec![pid(1)],
            }
        );

        let reopened = pool(4);
        reopened.resume_allocation(&allocation).unwrap();
        assert_eq!(reopened.read_page(pid(0)).unwrap().data()[0], 1);
        assert_eq!(reopened.read_page(pid(2)).unwrap().data()[0], 3);
        assert!(
            matches!(reopened.read_page(pid(1)), Err(BpmError::Io(e)) if e.kind() == io::ErrorKind::NotFound),
            "a page freed before the restart stays unreadable"
        );
        assert_eq!(reopened.new_page().unwrap().page_id(), pid(1));
        assert!(matches!(
            reopened.resume_allocation(&allocation),
            Err(BpmError::PoolInUse)
        ));
        assert_eq!(
            BpmError::PoolInUse.to_string(),
            "the buffer pool already handed out pages"
        );
    }

    #[test]
    fn a_deleted_page_id_is_handed_out_again_before_a_new_one() {
        let (bpm, _dir) = crate::testing::bpm(2);
        let ids: Vec<PageId> = (0..4).map(|_| bpm.new_page().unwrap().page_id()).collect();
        bpm.delete_page(pid(ids[2])).unwrap();
        bpm.delete_page(pid(ids[1])).unwrap();
        assert_eq!(bpm.live_page_count(), 2);
        assert_eq!(bpm.allocated_page_count(), 4);

        let reused: Vec<PageId> = (0..3).map(|_| bpm.new_page().unwrap().page_id()).collect();
        assert_eq!(reused, vec![ids[1], ids[2], pid(4)], "lowest free id first");
        assert_eq!(bpm.allocated_page_count(), 5);
        assert_eq!(bpm.live_page_count(), 5);
    }

    #[test]
    fn a_reused_page_id_reads_its_new_bytes_not_the_deleted_pages() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let deleted = {
            let mut page = bpm.new_page().unwrap();
            page.data_mut().fill(1);
            page.page_id()
        };
        drop(bpm.new_page().unwrap());
        assert!(!bpm.is_resident(pid(deleted)), "setup expected an eviction");
        bpm.delete_page(pid(deleted)).unwrap();

        let reused = {
            let mut page = bpm.new_page().unwrap();
            page.data_mut().fill(2);
            page.page_id()
        };
        assert_eq!(reused, deleted);
        drop(bpm.new_page().unwrap());
        assert!(
            !bpm.is_resident(pid(reused)),
            "setup expected a second eviction"
        );
        assert!(
            bpm.read_page(pid(reused))
                .unwrap()
                .data()
                .iter()
                .all(|&b| b == 2),
            "the reused id read stale bytes"
        );
    }

    #[test]
    fn a_flush_that_finds_its_frame_given_to_another_page_reports_the_page_as_not_cached() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let page_id = bpm.new_page().unwrap().page_id();
        let frame_id = bpm.meta.lock().page_table[&page_id];
        bpm.frames[frame_id.0].write().page_id = Some(pid(page_id.get() + 1));

        assert!(matches!(
            bpm.flush_page(pid(page_id)),
            Err(BpmError::PageNotCached { page_id: p }) if p == page_id
        ));
    }

    #[test]
    fn concurrent_deletes_and_allocations_never_hand_out_a_live_page_id_twice() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let bpm = Arc::clone(&bpm);
                thread::spawn(move || {
                    let mut owned = Vec::new();
                    for round in 0..2_000 {
                        let page_id = {
                            let mut page = bpm.new_page().unwrap();
                            let page_id = page.page_id();
                            page.data_mut()[..8]
                                .copy_from_slice(&u64::from(page_id.get()).to_le_bytes());
                            page_id
                        };
                        owned.push(page_id);
                        if round % 2 == 1 {
                            let victim = owned.swap_remove(round % owned.len());
                            let stored = u64::from_le_bytes(
                                bpm.read_page(pid(victim)).unwrap().data()[..8]
                                    .try_into()
                                    .unwrap(),
                            );
                            assert_eq!(
                                stored,
                                u64::from(victim.get()),
                                "page {victim} was handed out twice"
                            );
                            bpm.delete_page(pid(victim)).unwrap();
                        }
                    }
                    owned
                })
            })
            .collect();
        let mut live: Vec<PageId> = workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect();
        let count = live.len();
        live.sort_unstable();
        live.dedup();
        assert_eq!(live.len(), count, "two workers own the same page id");
        assert_eq!(bpm.live_page_count(), count);
        assert!(
            bpm.allocated_page_count() <= count + 4,
            "deleted ids were not reused: {} ids for {count} live pages",
            bpm.allocated_page_count()
        );
    }

    #[test]
    fn deleting_a_resident_page_frees_its_frame_without_writing_it() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let page_id = bpm.new_page().unwrap().page_id();
        assert_eq!(bpm.live_page_count(), 1);

        bpm.delete_page(pid(page_id)).unwrap();

        assert_eq!(bpm.live_page_count(), 0);
        assert!(!bpm.is_resident(pid(page_id)));
        {
            let meta = bpm.meta.lock();
            assert_eq!(
                meta.free_list,
                vec![FrameId(0)],
                "the frame went back to the free list"
            );
            assert_eq!(
                meta.replacer.size(),
                0,
                "the replacer still tracks the frame"
            );
        }
        assert!(
            matches!(bpm.read_page(pid(page_id)), Err(BpmError::Io(_))),
            "a deleted page that was never written has nothing on disk to read"
        );
        let next = bpm.new_page().unwrap().page_id();
        assert_eq!(next, page_id, "the deleted id is handed out again");
    }

    #[test]
    fn deleting_an_evicted_page_frees_it_on_disk() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let evicted = bpm.new_page().unwrap().page_id();
        let resident = bpm.new_page().unwrap().page_id();
        assert!(!bpm.is_resident(pid(evicted)), "setup expected an eviction");

        bpm.delete_page(pid(evicted)).unwrap();

        assert_eq!(bpm.live_page_count(), 1);
        assert!(bpm.is_resident(pid(resident)));
        assert!(matches!(bpm.read_page(pid(evicted)), Err(BpmError::Io(_))));
    }

    #[test]
    fn deleting_a_pinned_or_unknown_page_fails_and_changes_nothing() {
        let (bpm, _dir) = crate::testing::bpm(2);
        let guard = bpm.new_page().unwrap();
        let pinned = guard.page_id();
        assert!(matches!(
            bpm.delete_page(pid(pinned)),
            Err(BpmError::PagePinned { page_id }) if page_id == pinned
        ));
        drop(guard);
        assert!(bpm.is_resident(pid(pinned)));

        bpm.delete_page(pid(pinned)).unwrap();
        for missing in [pinned, pid(99)] {
            assert!(matches!(
                bpm.delete_page(pid(missing)),
                Err(BpmError::NoSuchPage { page_id }) if page_id == missing
            ));
        }
        assert_eq!(bpm.live_page_count(), 0);
    }

    #[test]
    fn allocated_page_count_counts_new_pages_without_allocating() {
        let (bpm, _dir) = crate::testing::bpm(2);
        assert_eq!(bpm.allocated_page_count(), 0);
        for expected in 1..=3 {
            drop(bpm.new_page().unwrap());
            assert_eq!(bpm.allocated_page_count(), expected);
            assert_eq!(bpm.allocated_page_count(), expected);
        }
    }

    #[test]
    fn a_page_stops_being_resident_once_it_is_evicted() {
        let (bpm, _dir) = crate::testing::bpm(2);
        let first = bpm.new_page().unwrap().page_id();
        assert!(bpm.is_resident(pid(first)));
        assert!(!bpm.is_resident(pid(first.get() + 100)));
        let second = bpm.new_page().unwrap().page_id();
        let third = bpm.new_page().unwrap().page_id();
        assert!(bpm.is_resident(pid(second)));
        assert!(bpm.is_resident(pid(third)));
        assert!(!bpm.is_resident(pid(first)));
        drop(bpm.read_page(pid(first)).unwrap());
        assert!(bpm.is_resident(pid(first)));
    }

    fn finishes_within(limit: Duration, work: impl FnOnce() + Send + 'static) -> bool {
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            work();
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(limit).is_ok()
    }

    #[test]
    fn dropping_a_guard_with_zero_pins_still_releases_the_frame_lock() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let page = bpm.new_page().unwrap();
        let page_id = page.page_id();
        bpm.meta.lock().pins[page.frame_id().0] = 0;
        drop(page);

        let finished = finishes_within(Duration::from_secs(5), move || {
            drop(bpm.write_page(pid(page_id)).unwrap());
        });
        assert!(finished, "frame write lock was leaked by the guard drop");
    }

    #[test]
    #[should_panic(expected = "replacer capacity differs")]
    fn new_rejects_a_replacer_sized_for_a_different_pool() {
        let dir = tempfile::tempdir().unwrap();
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).unwrap();
        BufferPoolManager::new(2, ArcReplacer::new(3), DiskScheduler::new(dm));
    }

    #[test]
    fn concurrent_access_retries_when_a_cached_frame_is_reassigned() {
        let (bpm, _dir) = crate::testing::bpm(2);
        let page_count = 6;
        for _ in 0..page_count {
            drop(bpm.new_page().unwrap());
        }
        let missing_frames = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..6)
            .map(|worker| {
                let bpm = Arc::clone(&bpm);
                let missing_frames = Arc::clone(&missing_frames);
                thread::spawn(move || {
                    for round in 0..3_000 {
                        let page_id = (worker * 7 + round) % page_count;
                        let reported_missing = if worker % 3 == 0 {
                            matches!(
                                bpm.write_page(pid(page_id)),
                                Err(BpmError::FrameOutOfRange { .. })
                            )
                        } else {
                            matches!(
                                bpm.read_page(pid(page_id)),
                                Err(BpmError::FrameOutOfRange { .. })
                            )
                        };
                        if reported_missing {
                            missing_frames.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        assert_eq!(
            missing_frames.load(Ordering::Relaxed),
            0,
            "a page that exists was reported as FrameOutOfRange"
        );
    }

    #[test]
    fn flushing_a_page_that_is_not_cached_reports_it() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let first = bpm.new_page().unwrap().page_id();
        let second = bpm.new_page().unwrap().page_id();
        assert!(!bpm.is_resident(pid(first)));
        assert!(matches!(
            bpm.flush_page(pid(first)),
            Err(BpmError::PageNotCached { page_id }) if page_id == first
        ));
        assert!(bpm.flush_page(pid(second)).is_ok());
    }

    #[test]
    fn a_frame_id_past_the_pool_is_out_of_range() {
        let (bpm, _dir) = crate::testing::bpm(2);
        assert!(bpm.frame(FrameId(1)).is_ok());
        assert!(matches!(
            bpm.frame(FrameId(2)),
            Err(BpmError::FrameOutOfRange {
                frame_id: FrameId(2)
            })
        ));
    }

    #[test]
    fn last_guard_drop_makes_the_frame_evictable_again() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let page_id = bpm.new_page().unwrap().page_id();
        let first = bpm.read_page(pid(page_id)).unwrap();
        let second = bpm.read_page(pid(page_id)).unwrap();
        drop(first);
        assert!(matches!(
            bpm.new_page(),
            Err(BpmError::FrameInUse | BpmError::NoFreeFrame)
        ));
        drop(second);
        assert!(
            bpm.new_page().is_ok(),
            "frame stayed pinned after its last guard dropped"
        );
    }

    const VICTIM_FILL: u8 = 0xA1;

    #[derive(Clone, Copy)]
    enum FrameRequest {
        NewPage,
        ReadMiss,
        WriteMiss,
    }

    fn assert_a_failed_victim_flush_keeps_the_victim_cached(request: FrameRequest) {
        let (bpm, _dir) = crate::testing::bpm(2);
        let on_disk = bpm.new_page().unwrap().page_id();
        bpm.flush_page(pid(on_disk)).unwrap();
        let victim = {
            let mut page = bpm.new_page().unwrap();
            page.data_mut().fill(VICTIM_FILL);
            page.page_id()
        };
        let other = bpm.new_page().unwrap().page_id();
        assert!(
            !bpm.is_resident(pid(on_disk)),
            "setup expected the first page to be evicted"
        );
        let victim_frame = bpm.meta.lock().page_table[&victim];
        let allocated = bpm.allocated_page_count();

        bpm.fail_next_disk_writes(1);
        let result = match request {
            FrameRequest::NewPage => bpm.new_page().map(drop),
            FrameRequest::ReadMiss => bpm.read_page(pid(on_disk)).map(drop),
            FrameRequest::WriteMiss => bpm.write_page(pid(on_disk)).map(drop),
        };
        match result {
            Err(BpmError::Io(e)) => assert_eq!(e.to_string(), disk::INJECTED_WRITE_FAILURE),
            other => panic!("expected the injected write failure, got {other:?}"),
        }

        {
            let meta = bpm.meta.lock();
            assert_eq!(
                meta.page_table.get(&victim),
                Some(&victim_frame),
                "the victim lost its page table entry"
            );
            assert_eq!(meta.page_table.len(), 2, "page table entries");
            assert!(meta.free_list.is_empty(), "the victim frame was freed");
            assert_eq!(meta.pins[victim_frame.0], 0, "victim pins");
            assert_eq!(
                meta.replacer.size(),
                2,
                "the victim stopped being evictable"
            );
        }
        {
            let frame = bpm.frames[victim_frame.0].read();
            assert_eq!(frame.page_id, Some(victim));
            assert!(
                frame.dirty.load(Ordering::Acquire),
                "the victim lost its dirty flag"
            );
            assert!(frame.buf.iter().all(|&b| b == VICTIM_FILL));
        }
        assert!(bpm.is_resident(pid(other)));
        assert!(!bpm.is_resident(pid(on_disk)));
        assert_eq!(
            bpm.allocated_page_count(),
            allocated,
            "a failed request handed out a page id"
        );

        bpm.flush_page(pid(victim)).unwrap();
        assert!(
            bpm.read_from_disk(victim)
                .unwrap()
                .iter()
                .all(|&b| b == VICTIM_FILL),
            "the later flush wrote different bytes"
        );
        drop(bpm.read_page(pid(on_disk)).unwrap());
        assert!(
            !bpm.is_resident(pid(victim)),
            "the retried miss picked a different victim"
        );
        assert!(
            bpm.read_page(pid(victim))
                .unwrap()
                .data()
                .iter()
                .all(|&b| b == VICTIM_FILL)
        );
    }

    #[test]
    fn a_failed_victim_flush_during_new_page_keeps_the_victim_cached_and_dirty() {
        assert_a_failed_victim_flush_keeps_the_victim_cached(FrameRequest::NewPage);
    }

    #[test]
    fn a_failed_victim_flush_during_a_read_miss_keeps_the_victim_cached_and_dirty() {
        assert_a_failed_victim_flush_keeps_the_victim_cached(FrameRequest::ReadMiss);
    }

    #[test]
    fn a_failed_victim_flush_during_a_write_miss_keeps_the_victim_cached_and_dirty() {
        assert_a_failed_victim_flush_keeps_the_victim_cached(FrameRequest::WriteMiss);
    }

    fn mark_pinned_frame_evictable(bpm: &BufferPoolManager, frame_id: FrameId) {
        let mut meta = bpm.meta.lock();
        assert!(meta.pins[frame_id.0] > 0, "setup expected a pinned frame");
        meta.replacer.set_evictable(frame_id, true);
    }

    #[test]
    fn a_pinned_victim_stays_in_the_pool_after_frame_in_use() {
        let (bpm, _dir) = crate::testing::bpm(1);
        let pinned = bpm.new_page().unwrap();
        let pinned_id = pinned.page_id();
        let frame_id = pinned.frame_id();
        mark_pinned_frame_evictable(&bpm, frame_id);

        assert!(matches!(bpm.new_page(), Err(BpmError::FrameInUse)));
        {
            let meta = bpm.meta.lock();
            assert_eq!(meta.page_table.get(&pinned_id), Some(&frame_id));
            assert!(meta.free_list.is_empty(), "the pinned frame was freed");
            assert_eq!(meta.pins[frame_id.0], 1, "pins changed");
        }

        drop(pinned);
        assert_eq!(
            bpm.meta.lock().replacer.size(),
            1,
            "the frame did not become evictable once unpinned"
        );
        let replacement = bpm.new_page().unwrap().page_id();
        assert!(bpm.is_resident(pid(replacement)));
        assert!(!bpm.is_resident(pid(pinned_id)));
    }

    #[test]
    fn eviction_after_frame_in_use_picks_another_victim() {
        let (bpm, _dir) = crate::testing::bpm(2);
        let pinned = bpm.new_page().unwrap();
        let pinned_id = pinned.page_id();
        let unpinned = bpm.new_page().unwrap().page_id();
        mark_pinned_frame_evictable(&bpm, pinned.frame_id());

        assert!(matches!(bpm.new_page(), Err(BpmError::FrameInUse)));
        assert_eq!(
            bpm.meta.lock().replacer.size(),
            1,
            "the pinned victim was left evictable"
        );

        let replacement = bpm.new_page().unwrap().page_id();
        assert!(bpm.is_resident(pid(pinned_id)));
        assert!(bpm.is_resident(pid(replacement)));
        assert!(!bpm.is_resident(pid(unpinned)));
        drop(pinned);
    }

    const MISS_STALL_WINDOW: Duration = Duration::from_millis(200);
    const UNBLOCKED_LIMIT: Duration = Duration::from_secs(5);

    fn wait_until(limit: Duration, condition: impl Fn() -> bool) -> bool {
        let deadline = std::time::Instant::now() + limit;
        while std::time::Instant::now() < deadline {
            if condition() {
                return true;
            }
            thread::sleep(Duration::from_millis(1));
        }
        condition()
    }

    #[derive(Clone, Copy, PartialEq)]
    enum StalledDiskRequest {
        DirtyVictimWriteBack,
        PageRead,
    }

    fn assert_a_hit_waits_while_a_miss_is_stalled_on(stalled: StalledDiskRequest) {
        let (bpm, _dir) = crate::testing::bpm(2);
        let evicted = bpm.new_page().unwrap().page_id();
        let victim = bpm.new_page().unwrap().page_id();
        let resident = bpm.new_page().unwrap().page_id();
        if stalled == StalledDiskRequest::PageRead {
            bpm.flush_page(pid(victim)).unwrap();
        }
        let victim_frame = bpm.meta.lock().page_table[&victim];
        assert_eq!(
            bpm.frames[victim_frame.0]
                .read()
                .dirty
                .load(Ordering::Acquire),
            stalled == StalledDiskRequest::DirtyVictimWriteBack,
            "setup expected the victim to be dirty only when the write-back is the stalled request"
        );
        assert!(
            !bpm.is_resident(pid(evicted)),
            "setup expected the first page to be evicted"
        );
        let resident_pin = bpm.read_page(pid(resident)).unwrap();

        let disk_stalled = bpm.scheduler.lock();
        let miss = {
            let bpm = Arc::clone(&bpm);
            thread::spawn(move || bpm.read_page(pid(evicted)).is_ok())
        };
        assert!(
            wait_until(UNBLOCKED_LIMIT, || bpm.meta.is_locked()),
            "the miss never took the meta lock"
        );

        let (hit_done_tx, hit_done_rx) = mpsc::channel();
        {
            let bpm = Arc::clone(&bpm);
            thread::spawn(move || {
                let hit = bpm.read_page(pid(resident)).is_ok();
                let _ = hit_done_tx.send(hit);
            });
        }
        assert!(
            hit_done_rx.recv_timeout(MISS_STALL_WINDOW).is_err(),
            "a hit finished while a miss was waiting on disk; if misses no longer hold the meta \
             lock across disk I/O, update the miss bullet in buffer_pool_manager.md and flip these tests"
        );

        drop(disk_stalled);
        assert_eq!(
            hit_done_rx.recv_timeout(UNBLOCKED_LIMIT).ok(),
            Some(true),
            "the hit did not finish once the miss's disk requests could run"
        );
        assert!(
            miss.join().unwrap(),
            "the miss failed after its disk requests ran"
        );
        drop(resident_pin);
    }

    #[test]
    fn a_hit_waits_while_a_miss_holds_the_meta_lock_through_the_dirty_victim_write_back() {
        assert_a_hit_waits_while_a_miss_is_stalled_on(StalledDiskRequest::DirtyVictimWriteBack);
    }

    #[test]
    fn a_hit_waits_while_a_miss_holds_the_meta_lock_through_the_page_read() {
        assert_a_hit_waits_while_a_miss_is_stalled_on(StalledDiskRequest::PageRead);
    }

    #[test]
    fn new_page_fails_once_page_ids_run_out_and_keeps_the_frame() {
        let (bpm, _dir) = crate::testing::bpm(1);
        bpm.resume_allocation(&PageAllocation {
            next_page_id: PageId::MAX.as_usize() + 1,
            free_page_ids: vec![],
        })
        .unwrap();
        let err = bpm.new_page().err().unwrap();
        assert!(matches!(err, BpmError::PageIdsExhausted(_)), "got {err}");
        assert_eq!(
            err.to_string(),
            format!(
                "no page id left to hand out: page id {} is past the largest page id, {}",
                u64::from(PageId::MAX.get()) + 1,
                PageId::MAX
            )
        );
        assert!(
            matches!(bpm.new_page(), Err(BpmError::PageIdsExhausted(_))),
            "the failed allocation gave its frame back instead of leaking it"
        );
    }
}
