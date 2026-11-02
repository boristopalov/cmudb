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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

#[cfg(feature = "deadlock_detection")]
const DEADLOCK_POLL_INTERVAL: Duration = Duration::from_secs(1);

type Page = [u8; disk::PAGE_SIZE];

#[derive(Debug)]
struct FrameHeader {
    frame_id: usize,
    page_id: Option<usize>,
    buf: Page,
    dirty: AtomicBool,
}

#[derive(Debug)]
struct EvictedPage {
    page_id: usize,
    data: Page,
}

#[derive(Debug)]
struct FrameReservation {
    frame_id: usize,
    evicted: Option<EvictedPage>,
}

#[derive(Debug)]
struct Meta {
    replacer: replacer::ArcReplacer,
    free_list: Vec<usize>,
    page_table: HashMap<usize, usize>,
    /// Pins per frame. Kept beside the replacer under one lock, so a frame is marked
    /// evictable only while nothing pins it.
    pins: Vec<u32>,
}

#[derive(Debug)]
pub struct BufferPoolManager {
    frames: Vec<RwLock<FrameHeader>>,
    meta: Mutex<Meta>,
    scheduler: Mutex<disk::DiskScheduler>,
    next_page_id: AtomicUsize,
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
        // SAFETY: `frame_guard` is dropped exactly once, here in `Drop::drop`, and is never read afterwards.
        unsafe { ManuallyDrop::drop(&mut self.frame_guard) };
        self.bpm.release_pin(frame_id);
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
        // SAFETY: `frame_guard` is dropped exactly once, here in `Drop::drop`, and is never read afterwards.
        unsafe { ManuallyDrop::drop(&mut self.frame_guard) };
        self.bpm.release_pin(frame_id);
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
                frame_id: i,
                page_id: None,
                buf: [0u8; disk::PAGE_SIZE],
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
                pins: vec![0; num_frames],
            }),
            scheduler: Mutex::new(scheduler),
            next_page_id: AtomicUsize::new(0),
        }
    }

    pub fn new_page(self: &Arc<Self>) -> PageMutResult<'_> {
        info!("tid={} new_page requested", Self::tid());
        let mut meta = self.meta.lock();
        let reservation = self.get_free_frame(&mut meta)?;
        let frame_id = reservation.frame_id;
        let page_id = self.get_next_page_id();
        let mut frame = self.frame(frame_id)?.write();
        frame.clear();
        frame.page_id = Some(page_id);
        frame.dirty.store(true, Ordering::Release);
        meta.pins[frame_id] = 1;
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, false);

        Ok(PageMut {
            frame_guard: ManuallyDrop::new(frame),
            bpm: Arc::clone(self),
        })
    }

    pub fn write_page(self: &Arc<Self>, page_id: usize) -> PageMutResult<'_> {
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
            meta.pins[frame_id] += 1;
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
            bpm: Arc::clone(self),
        })
    }

    pub fn read_page(self: &Arc<Self>, page_id: usize) -> PageRefResult<'_> {
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
            meta.pins[frame_id] += 1;
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
                bpm: Arc::clone(self),
            });
        };

        info!("tid={} read_page cache miss page_id={page_id}", Self::tid());
        let frame = self.load_into_free_frame(&mut meta, page_id, false)?;
        let frame_id = frame.frame_id;
        Ok(PageRef {
            frame_guard: ManuallyDrop::new(RwLockWriteGuard::downgrade(frame)),
            frame_id,
            bpm: Arc::clone(self),
        })
    }

    fn frame(&self, frame_id: usize) -> Result<&RwLock<FrameHeader>, BpmError> {
        self.frames.get(frame_id).ok_or(BpmError::FrameMissing)
    }

    fn load_into_free_frame(
        &self,
        meta: &mut Meta,
        page_id: usize,
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
        meta.pins[frame_id] = 1;
        frame.dirty.store(dirty, Ordering::Release);
        meta.page_table.insert(page_id, frame_id);
        meta.replacer.record_access(frame_id, page_id);
        meta.replacer.set_evictable(frame_id, false);
        Ok(frame)
    }

    fn read_from_disk(&self, page_id: usize) -> Result<disk::PageData, BpmError> {
        self.run_disk_request(page_id, false, Box::new([0u8; PAGE_SIZE]))
    }

    fn write_to_disk(&self, page_id: usize, page: &Page) -> Result<(), BpmError> {
        self.run_disk_request(page_id, true, Box::new(*page))
            .map(drop)
    }

    fn run_disk_request(
        &self,
        page_id: usize,
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
        let Some(frame_id) = meta.replacer.evict() else {
            return Err(BpmError::NoFreeFrame);
        };
        info!(
            "tid={} get_free_frame: replacer returned victim frame_id={frame_id}",
            Self::tid()
        );

        if meta.pins[frame_id] > 0 {
            warn!(
                "tid={} get_free_frame: attempted to evict pinned frame_id={frame_id}",
                Self::tid()
            );
            return Err(BpmError::FrameInUse);
        }
        let mut frame = self.frame(frame_id)?.write();

        let Some(page_id) = frame.page_id else {
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
            self.write_to_disk(page_id, &frame.buf)?;
        }

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

    fn release_pin(&self, frame_id: usize) {
        let mut meta = self.meta.lock();
        match meta.pins[frame_id].checked_sub(1) {
            Some(remaining) => {
                meta.pins[frame_id] = remaining;
                if remaining == 0 {
                    meta.replacer.set_evictable(frame_id, true);
                }
            }
            None => warn!("tried to unpin a frame with 0 pins frame_id={frame_id}"),
        }
    }

    fn get_next_page_id(&self) -> usize {
        self.next_page_id.fetch_add(1, Ordering::AcqRel)
    }

    /// Returns how many page ids this pool has handed out; ids run densely from 0.
    pub fn allocated_page_count(&self) -> usize {
        self.next_page_id.load(Ordering::Acquire)
    }

    /// Reports whether `page_id` currently sits in a frame, so a read of it would not touch disk.
    pub fn is_resident(&self, page_id: usize) -> bool {
        self.meta.lock().page_table.contains_key(&page_id)
    }

    pub fn flush_page(&self, page_id: usize) -> Result<(), BpmError> {
        info!("flush_page: flushing page page_id={page_id}");
        let Some(frame_id) = self.meta.lock().page_table.get(&page_id).copied() else {
            warn!("tried to flush page with page_id={page_id} but could not find its frame");
            return Err(BpmError::FrameMissing);
        };

        let frame = self.frame(frame_id)?.read();
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

    #[test]
    fn allocated_page_count_counts_new_pages_without_allocating() {
        let (bpm, _dir) = make_bpm(2);
        assert_eq!(bpm.allocated_page_count(), 0);
        for expected in 1..=3 {
            drop(bpm.new_page().unwrap());
            assert_eq!(bpm.allocated_page_count(), expected);
            assert_eq!(bpm.allocated_page_count(), expected);
        }
    }

    #[test]
    fn a_page_stops_being_resident_once_it_is_evicted() {
        let (bpm, _dir) = make_bpm(2);
        let first = bpm.new_page().unwrap().page_id().unwrap();
        assert!(bpm.is_resident(first));
        assert!(!bpm.is_resident(first + 100));
        let second = bpm.new_page().unwrap().page_id().unwrap();
        let third = bpm.new_page().unwrap().page_id().unwrap();
        assert!(bpm.is_resident(second));
        assert!(bpm.is_resident(third));
        assert!(!bpm.is_resident(first));
        drop(bpm.read_page(first).unwrap());
        assert!(bpm.is_resident(first));
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
        bpm.meta.lock().pins[page.frame_id()] = 0;
        drop(page);

        let finished = finishes_within(Duration::from_secs(5), move || {
            drop(bpm.write_page(page_id).unwrap());
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
                        let reported_missing = if worker % 3 == 0 {
                            matches!(bpm.write_page(page_id), Err(BpmError::FrameMissing))
                        } else {
                            matches!(bpm.read_page(page_id), Err(BpmError::FrameMissing))
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
            "a page that exists was reported as FrameMissing"
        );
    }

    #[test]
    fn last_guard_drop_makes_the_frame_evictable_again() {
        let (bpm, _dir) = make_bpm(1);
        let page_id = bpm.new_page().unwrap().page_id().unwrap();
        let first = bpm.read_page(page_id).unwrap();
        let second = bpm.read_page(page_id).unwrap();
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
}
