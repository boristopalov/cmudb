use crate::disk;
use crate::disk::DiskRequest;
use crate::disk::PAGE_SIZE;
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

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc;

#[cfg(feature = "deadlock_detection")]
const DEADLOCK_POLL_INTERVAL: Duration = Duration::from_secs(1);

type Page = [u8; disk::PAGE_SIZE];

#[derive(Debug)]
struct FrameHeader {
    frame_id: usize,
    page_id: Option<usize>,
    buf: Page,
    pin_count: AtomicU32,
    dirty: AtomicBool,
}

#[derive(Debug)]
struct FrameReservation {
    frame_id: usize,
    page_id: Option<usize>,
    reclaimed_page: Option<Page>,
}

#[derive(Debug)]
struct Meta {
    replacer: replacer::ArcReplacer,
    free_list: Vec<usize>,
    page_table: HashMap<usize, usize>,
}

// Must be able to handle DBs greater than hardware memory limit
// Buffer pool can get from memory or from disk
#[derive(Debug)]
pub struct BufferPoolManager {
    frames: Vec<RwLock<FrameHeader>>,
    /// Protects all metadata that participates in eviction/mapping: replacer, page_table, free_list,
    /// global pin count. Lock order: meta -> frame.
    meta: Mutex<Meta>,
    scheduler: Mutex<disk::DiskScheduler>,
    next_page_id: AtomicUsize,
    // num_frames: u32,
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
    frame_id: usize,
    bpm: Arc<BufferPoolManager>,
}

#[derive(Debug)]
pub struct PageMut<'a> {
    frame_guard: ManuallyDrop<RwLockWriteGuard<'a, FrameHeader>>,
    bpm: Arc<BufferPoolManager>,
}

impl<'a> PageRef<'a> {
    pub fn frame_id(&self) -> usize {
        self.frame_id
    }

    pub fn data(&self) -> &[u8] {
        &self.frame_guard.buf
    }

    pub fn page_id(&self) -> Option<usize> {
        self.frame_guard.page_id
    }
}

impl<'a> Drop for PageRef<'a> {
    fn drop(&mut self) {
        let frame_id = self.frame_guard.frame_id;
        let released_last_pin = self.frame_guard.unpin();
        // SAFETY: `frame_guard` is dropped exactly once, here in `Drop::drop`, and is never read afterwards.
        unsafe { ManuallyDrop::drop(&mut self.frame_guard) };
        if released_last_pin {
            self.bpm.mark_evictable_after_frame_released(frame_id);
        }
    }
}

impl<'a> PageMut<'a> {
    pub fn frame_id(&self) -> usize {
        self.frame_guard.frame_id
    }

    pub fn page_id(&self) -> Option<usize> {
        self.frame_guard.page_id
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.frame_guard.buf
    }
}

impl<'a> Drop for PageMut<'a> {
    fn drop(&mut self) {
        let frame_id = self.frame_guard.frame_id;
        let released_last_pin = self.frame_guard.unpin();
        // SAFETY: `frame_guard` is dropped exactly once, here in `Drop::drop`, and is never read afterwards.
        unsafe { ManuallyDrop::drop(&mut self.frame_guard) };
        if released_last_pin {
            self.bpm.mark_evictable_after_frame_released(frame_id);
        }
    }
}

#[derive(Debug)]
pub enum BpmError {
    Poisoned,
    NoFreeFrame,
    FrameMissing,
    MutexPanic,
    FramePinned,
    ChannelClosed,
    Io(io::Error),
    FrameInUse,
}

impl fmt::Display for BpmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BpmError::Poisoned => write!(f, "Lock poisoned"),
            BpmError::NoFreeFrame => {
                write!(f, "No free frame available")
            }
            BpmError::Io(e) => {
                write!(f, "io error: {}", e)
            }
            BpmError::FrameMissing => write!(f, "Frame missing"),
            BpmError::MutexPanic => write!(f, "Mutex panicked"),
            BpmError::FramePinned => write!(f, "Cannot evict a frame which is pinned"),
            BpmError::ChannelClosed => write!(f, "Channel closed"),
            BpmError::FrameInUse => write!(f, "Frame in use"),
        }
    }
}

type PageRefResult<'a> = Result<PageRef<'a>, BpmError>;
type PageMutResult<'a> = Result<PageMut<'a>, BpmError>;

impl FrameHeader {
    fn unpin(&self) -> bool {
        match self
            .pin_count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pins| {
                pins.checked_sub(1)
            }) {
            Ok(previous_pins) => previous_pins == 1,
            Err(_) => {
                warn!(
                    "tried to unpin a frame with 0 pins frame_id={}",
                    self.frame_id
                );
                false
            }
        }
    }

    pub fn clear(&mut self) {
        self.dirty.store(false, Ordering::Release);
        self.pin_count.store(0, Ordering::Release);
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
        #[cfg(feature = "deadlock_detection")]
        start_deadlock_detector();
        let mut frames = Vec::with_capacity(num_frames);
        let mut free_list = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            frames.push(RwLock::new(FrameHeader {
                frame_id: i,
                page_id: None,
                buf: [0u8; disk::PAGE_SIZE],
                pin_count: AtomicU32::new(0),
                dirty: AtomicBool::new(false),
            }));
            free_list.push(i);
        }

        Self {
            frames,
            meta: Mutex::new(Meta {
                replacer,
                free_list,
                page_table: HashMap::with_capacity(num_frames),
            }),
            scheduler: Mutex::new(scheduler),
            next_page_id: AtomicUsize::new(0),
        }
    }

    pub fn new_page(self: &Arc<Self>) -> Result<PageMut, BpmError> {
        info!("tid={} new_page requested", Self::tid());
        let mut meta = self.meta.lock();
        let reservation = self.get_free_frame(&mut meta)?;
        let frame_id = reservation.frame_id;
        let page_id = self.get_next_page_id();
        let mut frame = self
            .frames
            .get(frame_id)
            .ok_or(BpmError::FrameMissing)?
            .write();
        frame.clear();
        frame.page_id = Some(page_id);
        frame.dirty.store(true, Ordering::Release);
        frame.pin_count.store(1, Ordering::Release);
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, false);

        return Ok(PageMut {
            frame_guard: ManuallyDrop::new(frame),
            bpm: Arc::clone(&self),
        });
    }

    pub fn write_page(self: &Arc<Self>, page_id: usize) -> PageMutResult {
        // case 1: frame exists.
        // Retries if the frame is evicted and reassigned between our reservation and acquiring the write lock:
        // a concurrent guard drop can take pin_count to 0 and re-mark the frame evictable,
        // letting another thread claim it before we get the lock.
        let mut meta = loop {
            let mut meta = self.meta.lock();

            if let Some(frame_id) = meta.page_table.get(&page_id).copied() {
                info!(
                    "tid={} write_page cache hit page_id={page_id} frame_id={frame_id}",
                    Self::tid()
                );
                meta.replacer.record_access(frame_id, page_id);
                meta.replacer.set_evictable(frame_id, false);
                drop(meta);

                let frame = self
                    .frames
                    .get(frame_id)
                    .ok_or(BpmError::FrameMissing)?
                    .write();
                if frame.page_id != Some(page_id) {
                    drop(frame);
                    continue;
                }
                frame.pin_count.fetch_add(1, Ordering::AcqRel);
                frame.dirty.store(true, Ordering::Release);

                return Ok(PageMut {
                    frame_guard: ManuallyDrop::new(frame),
                    bpm: Arc::clone(&self),
                });
            }

            break meta;
        };

        // case 2: frame not in memory, check for free frame
        let reservation = self.get_free_frame(&mut meta)?;
        let frame_id = reservation.frame_id;
        let mut frame = self
            .frames
            .get(frame_id)
            .ok_or(BpmError::FrameMissing)?
            .write();
        info!(
            "tid={} write_page cache miss page_id={page_id} reserved_frame={frame_id}",
            Self::tid()
        );

        // read the existing page from disk
        let (tx, rx) = mpsc::channel::<Result<disk::PageData, std::io::Error>>();
        let disk_request = DiskRequest {
            page_id,
            is_write: false,
            data: Box::new([0u8; disk::PAGE_SIZE]),
            promise: tx,
        };

        if self.scheduler.lock().schedule(vec![disk_request]).is_err() {
            drop(frame);
            self.restore_frame(&mut meta, reservation);
            return Err(BpmError::ChannelClosed);
        }

        // wait for the io request to complete
        let page_data = match rx.recv() {
            Ok(Ok(data)) => data,
            Ok(Err(e)) => {
                drop(frame);
                self.restore_frame(&mut meta, reservation);
                return Err(BpmError::Io(e));
            }
            Err(_) => {
                drop(frame);
                self.restore_frame(&mut meta, reservation);
                return Err(BpmError::ChannelClosed);
            }
        };

        // bring this page into the free frame
        frame.buf = *page_data;
        frame.page_id = Some(page_id);
        frame.pin_count.store(1, Ordering::Release);
        frame.dirty.store(true, Ordering::Release);
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, false);

        return Ok(PageMut {
            frame_guard: ManuallyDrop::new(frame),
            bpm: Arc::clone(&self),
        });
    }

    pub fn read_page(self: &Arc<Self>, page_id: usize) -> PageRefResult {
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
            meta.replacer.set_evictable(frame_id, false);
            drop(meta);

            let frame = self
                .frames
                .get(frame_id)
                .ok_or(BpmError::FrameMissing)?
                .read();
            if frame.page_id != Some(page_id) {
                continue;
            }
            frame.pin_count.fetch_add(1, Ordering::AcqRel);
            return Ok(PageRef {
                frame_guard: ManuallyDrop::new(frame),
                frame_id,
                bpm: Arc::clone(self),
            });
        };

        // case 2: page not cached -- issue an io request and load into memory
        info!("tid={} read_page cache miss page_id={page_id}", Self::tid());
        let (tx, rx) = mpsc::channel::<Result<disk::PageData, std::io::Error>>();
        let disk_request = DiskRequest {
            page_id,
            is_write: false,
            data: Box::new([0u8; PAGE_SIZE]),
            promise: tx,
        };

        // reserve a frame before issuing IO
        let reservation = self.get_free_frame(&mut meta)?;
        let frame_id = reservation.frame_id;
        let mut frame = self
            .frames
            .get(frame_id)
            .ok_or(BpmError::FrameMissing)?
            .write();
        info!("read_page reserved frame_id={frame_id} for page_id={page_id}",);

        if self.scheduler.lock().schedule(vec![disk_request]).is_err() {
            drop(frame);
            self.restore_frame(&mut meta, reservation);
            return Err(BpmError::ChannelClosed);
        }

        // wait for the io request to complete
        let page_data = match rx.recv() {
            Ok(Ok(data)) => data,
            Ok(Err(e)) => {
                drop(frame);
                self.restore_frame(&mut meta, reservation);
                return Err(BpmError::Io(e));
            }
            Err(_) => {
                drop(frame);
                self.restore_frame(&mut meta, reservation);
                return Err(BpmError::ChannelClosed);
            }
        };

        frame.buf = *page_data;
        frame.page_id = Some(page_id);
        frame.pin_count.store(1, Ordering::Release);
        frame.dirty.store(false, Ordering::Release);
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, false);

        let read_guard = parking_lot::RwLockWriteGuard::downgrade(frame);

        return Ok(PageRef {
            frame_guard: ManuallyDrop::new(read_guard),
            frame_id,
            bpm: Arc::clone(&self),
        });
    }

    fn get_free_frame(&self, meta: &mut Meta) -> Result<FrameReservation, BpmError> {
        info!("tid={} get_free_frame: looking for free frame", Self::tid());
        {
            if let Some(frame_id) = meta.free_list.pop() {
                info!(
                    "tid={} get_free_frame: using free_list frame_id={frame_id}",
                    Self::tid()
                );
                return Ok(FrameReservation {
                    frame_id,
                    page_id: None,
                    reclaimed_page: None,
                });
            }
        }

        // no free frame, we need to evict an existing frame
        info!(
            "tid={} get_free_frame: no free frames, attempting eviction",
            Self::tid()
        );

        let Some(frame_id) = meta.replacer.evict() else {
            return Err(BpmError::NoFreeFrame);
        };
        info!(
            "tid={} get_free_frame: replacer returned victim frame_id={})",
            Self::tid(),
            frame_id,
        );

        // Check the victim frame's state
        let mut frame = self
            .frames
            .get(frame_id)
            .ok_or(BpmError::FrameMissing)?
            .write();

        if frame.pin_count.load(Ordering::Acquire) > 0 {
            warn!(
                "tid={} get_free_frame: attempted to evict pinned frame_id={frame_id}; requeuing",
                Self::tid()
            );
            return Err(BpmError::FrameInUse);
        }

        // if there is no page_id on the frame, we can just return the frame
        // no page data to worry about in this case
        let Some(page_id) = frame.page_id else {
            return Ok(FrameReservation {
                frame_id,
                page_id: None,
                reclaimed_page: None,
            });
        };

        // no need to flush if the frame is not dirty
        if !frame.dirty.load(Ordering::Acquire) {
            meta.page_table.remove(&page_id);
            let buf = frame.buf.clone();
            frame.clear();
            return Ok(FrameReservation {
                frame_id,
                page_id: Some(page_id),
                reclaimed_page: Some(buf),
            });
        };

        info!(
            "tid={} get_free_frame: flushing dirty page_id={} before eviction",
            Self::tid(),
            page_id
        );
        let (tx, rx) = mpsc::channel::<Result<disk::PageData, std::io::Error>>();
        let disk_request = DiskRequest {
            page_id,
            is_write: true,
            data: Box::new(frame.buf),
            promise: tx,
        };
        if let Err(_) = self.scheduler.lock().schedule(vec![disk_request]) {
            return Err(BpmError::ChannelClosed);
        }
        match rx.recv() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                return Err(BpmError::Io(e));
            }
            Err(_) => {
                return Err(BpmError::ChannelClosed);
            }
        }

        meta.page_table.remove(&page_id);
        let buf = frame.buf.clone();
        frame.clear();
        Ok(FrameReservation {
            frame_id,
            page_id: Some(page_id),
            reclaimed_page: Some(buf),
        })
    }

    fn restore_frame(&self, meta: &mut Meta, reservation: FrameReservation) {
        let frame_id = reservation.frame_id;

        let Some(page_id) = reservation.page_id else {
            meta.free_list.push(frame_id);
            return;
        };
        let Some(reclaimed_page) = reservation.reclaimed_page else {
            meta.free_list.push(frame_id);
            return;
        };

        // if we evicted and took a page, check if we have that old page already loaded into a frame
        // if it is then the frame we evicted doesn't need to hold the old page and we can just push it back to the free list
        // if the old page is not loaded into a frame, get the frame that we marked for eviction, and want to restore, reset it, make sure
        // the old page ID is in the frame, and add it back to the page table
        info!("restore_frame: restoring evicted page for frame_id={frame_id}");
        if meta.page_table.contains_key(&page_id) {
            info!(
                "restore_frame: page_id={page_id} already mapped elsewhere, returning frame {frame_id} to free_list"
            );
            meta.free_list.push(frame_id);
            return;
        }
        // if we have this frame loaded into memory already,
        // just update it directly
        if let Some(frame_lock) = self.frames.get(frame_id) {
            let mut frame = frame_lock.write();
            frame.page_id = Some(page_id);
            frame.buf = reclaimed_page;
            meta.page_table.insert(page_id, frame_id);
        // if we do not have this frame loaded into memory,
        // mark it as a free frame
        } else {
            info!("restore_frame: returning free frame_id={frame_id} to free_list");
            meta.free_list.push(frame_id);
        }
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, true);
    }

    fn mark_evictable_after_frame_released(&self, frame_id: usize) {
        self.meta.lock().replacer.set_evictable(frame_id, true);
    }

    fn get_next_page_id(&self) -> usize {
        self.next_page_id
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
    }

    pub fn flush_page(&self, page_id: usize) -> Result<(), BpmError> {
        info!("flush_page: flushing page page_id={page_id}");

        let frame_id = {
            let meta = self.meta.lock();
            if let Some(fid) = meta.page_table.get(&page_id).copied() {
                fid
            } else {
                warn!("tried to flush page with page_id={page_id} but could not find its frame");
                return Err(BpmError::FrameMissing);
            }
        };

        let frame_lock = self.frames.get(frame_id).ok_or(BpmError::FrameMissing)?;
        let frame = frame_lock.read();

        let (tx, rx) = mpsc::channel::<Result<disk::PageData, std::io::Error>>();
        let disk_request = DiskRequest {
            page_id,
            is_write: true,
            data: Box::new(frame.buf),
            promise: tx,
        };

        if self
            .scheduler
            .lock()
            .schedule(vec![disk_request])
            .map_err(|_| BpmError::ChannelClosed)
            .is_err()
        {
            return Err(BpmError::ChannelClosed);
        }

        match rx.recv() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                return Err(BpmError::Io(e));
            }
            Err(_) => {
                return Err(BpmError::ChannelClosed);
            }
        };

        // if we get here we should have successfully flushed
        frame.dirty.store(false, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::replacer::ArcReplacer;
    use std::time::Duration;

    fn make_bpm(num_frames: usize) -> (Arc<BufferPoolManager>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).unwrap();
        let bpm = BufferPoolManager::new(
            num_frames,
            ArcReplacer::new(num_frames),
            DiskScheduler::new(dm),
        );
        (Arc::new(bpm), dir)
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
        let (bpm, _dir) = make_bpm(1);
        let page = bpm.new_page().unwrap();
        let page_id = page.page_id().unwrap();
        page.frame_guard.pin_count.store(0, Ordering::Release);
        drop(page);

        let finished = finishes_within(Duration::from_secs(5), move || {
            drop(bpm.write_page(page_id).unwrap());
        });
        assert!(finished, "frame write lock was leaked by the guard drop");
    }

    #[test]
    fn concurrent_reads_never_report_a_missing_frame_for_a_cached_page() {
        let (bpm, _dir) = make_bpm(2);
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
                        if let Err(BpmError::FrameMissing) = bpm.read_page(page_id) {
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
            "read_page reported FrameMissing for a page that exists"
        );
    }
}
