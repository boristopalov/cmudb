use log::{info, warn};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    fmt,
    fs::{File, OpenOptions},
    io,
    os::unix::fs::FileExt,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
        mpsc::{self, SendError},
    },
    thread::{self, JoinHandle},
};

pub const PAGE_SIZE: usize = 8192;

pub type PageData = Box<[u8; PAGE_SIZE]>;

#[derive(Debug)]
pub struct DiskRequest {
    pub page_id: usize,
    pub is_write: bool,
    pub data: PageData,
    pub promise: mpsc::Sender<Result<PageData, std::io::Error>>,
}

#[derive(Debug)]
pub struct DiskScheduler {
    tx: Option<mpsc::Sender<DiskRequest>>,
    handle: Option<JoinHandle<()>>,
    dm: Arc<DiskManager>,
}

#[derive(Debug)]
pub enum DiskSchedulerError {
    Shutdown,
    ChannelClosed(DiskRequest),
    WorkerPanicked,
}

impl fmt::Display for DiskSchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiskSchedulerError::Shutdown => write!(f, "disk scheduler is shut down"),
            DiskSchedulerError::ChannelClosed(req) => {
                write!(f, "failed to send disk request for page {}", req.page_id)
            }
            DiskSchedulerError::WorkerPanicked => write!(f, "disk worker thread panicked"),
        }
    }
}

impl From<SendError<DiskRequest>> for DiskSchedulerError {
    fn from(err: SendError<DiskRequest>) -> Self {
        DiskSchedulerError::ChannelClosed(err.0)
    }
}

impl DiskScheduler {
    pub fn new(disk_manager: DiskManager) -> Self {
        let (tx, rx) = mpsc::channel::<DiskRequest>();
        let dm = Arc::new(disk_manager);
        let worker_dm = dm.clone();
        info!("DiskScheduler: starting worker thread");
        let handle = thread::spawn(move || {
            let dm = worker_dm;
            for req in rx {
                info!(
                    "disk worker handling page_id={} is_write={}",
                    req.page_id, req.is_write
                );
                let r = if req.is_write {
                    dm.write_page(req.page_id, req.data)
                } else {
                    dm.read_page(req.page_id, req.data)
                };
                if let Err(e) = req.promise.send(r) {
                    warn!(
                        "disk worker failed to deliver result for page_id={} is_write={} err={}",
                        req.page_id, req.is_write, e
                    );
                }
            }
        });

        Self {
            tx: Some(tx),
            handle: Some(handle),
            dm,
        }
    }

    pub fn disk_manager(&self) -> Arc<DiskManager> {
        self.dm.clone()
    }

    pub fn schedule(&self, requests: Vec<DiskRequest>) -> Result<(), DiskSchedulerError> {
        let tx = self.tx.as_ref().ok_or(DiskSchedulerError::Shutdown)?;
        info!("DiskScheduler: scheduling {} request(s)", requests.len());
        for r in requests {
            tx.send(r)?;
        }
        Ok(())
    }

    /// shutdown closes the DiskRequest channel, waits for any existing requests to drain, and joins on the io thread
    pub fn shutdown(mut self) -> Result<(), DiskSchedulerError> {
        // dropping the sender closes the channel and lets the worker drain then exit
        self.tx.take();
        info!("DiskScheduler: shutting down worker");
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| DiskSchedulerError::WorkerPanicked)?;
        }
        Ok(())
    }
}

impl Drop for DiskScheduler {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug)]
struct PageStore {
    db_file: File,
    page_offsets: HashMap<usize, u64>,
    free_offsets: Vec<u64>,
}

impl PageStore {
    fn offset_of(&self, id: usize) -> io::Result<u64> {
        self.page_offsets.get(&id).copied().ok_or_else(|| {
            warn!("read_page: page_id={id} missing from lookup table");
            io::Error::new(io::ErrorKind::NotFound, format!("page {id} missing"))
        })
    }

    fn offset_for_write(&mut self, id: usize) -> io::Result<u64> {
        if let Some(&offset) = self.page_offsets.get(&id) {
            return Ok(offset);
        }
        if let Some(offset) = self.free_offsets.pop() {
            info!("write_page: reusing free offset={offset} for page_id={id}");
            return Ok(offset);
        }
        self.db_file.metadata().map(|m| m.len())
    }

    fn log_page_mapping(&self, context: &str) {
        warn!(
            "{context}; page_id->offset mapping: {:?}",
            self.page_offsets
        );
    }
}

#[derive(Debug)]
pub struct DiskManager {
    num_writes: AtomicU32,
    store: Mutex<PageStore>,
}

fn create_log_file_if_missing(log_path: &Path) -> io::Result<()> {
    OpenOptions::new()
        .append(true)
        .create(true)
        .open(log_path)
        .map(drop)
}

impl DiskManager {
    pub fn new<P: AsRef<Path>>(db_path: P, log_path: P) -> io::Result<Self> {
        let db_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(db_path)?;
        create_log_file_if_missing(log_path.as_ref())?;

        Ok(Self {
            num_writes: AtomicU32::new(0),
            store: Mutex::new(PageStore {
                db_file,
                page_offsets: HashMap::new(),
                free_offsets: Vec::new(),
            }),
        })
    }

    pub fn read_page(&self, id: usize, mut buf: PageData) -> io::Result<PageData> {
        let store = self.store.lock();
        let offset = store.offset_of(id)?;
        info!("read_page: page_id={id} offset={offset}");
        if let Err(e) = store.db_file.read_exact_at(&mut *buf, offset) {
            store.log_page_mapping(&format!(
                "read_page error page_id={id} offset={offset} err={e}"
            ));
            return Err(e);
        }
        Ok(buf)
    }

    pub fn write_page(&self, id: usize, data: PageData) -> io::Result<PageData> {
        let mut store = self.store.lock();
        let offset = store.offset_for_write(id)?;
        if let Err(e) = store.db_file.write_all_at(&*data, offset) {
            store.log_page_mapping(&format!(
                "write_page error page_id={id} offset={offset} err={e}"
            ));
            return Err(e);
        }
        store.page_offsets.insert(id, offset);
        info!("write_page: wrote page_id={id} at offset={offset}");
        self.num_writes.fetch_add(1, Ordering::AcqRel);
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(fill: u8) -> PageData {
        Box::new([fill; PAGE_SIZE])
    }

    fn manager(dir: &tempfile::TempDir) -> DiskManager {
        DiskManager::new(dir.path().join("db"), dir.path().join("log")).unwrap()
    }

    #[test]
    fn write_then_read_round_trips_and_overwrites_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let dm = manager(&dir);
        dm.write_page(7, page(0xAB)).unwrap();
        dm.write_page(3, page(0x01)).unwrap();
        assert_eq!(*dm.read_page(7, page(0)).unwrap(), [0xAB; PAGE_SIZE]);
        assert_eq!(*dm.read_page(3, page(0)).unwrap(), [0x01; PAGE_SIZE]);

        // Rewriting a page reuses its offset instead of growing the file.
        let len = std::fs::metadata(dir.path().join("db")).unwrap().len();
        dm.write_page(7, page(0xCD)).unwrap();
        assert_eq!(std::fs::metadata(dir.path().join("db")).unwrap().len(), len);
        assert_eq!(*dm.read_page(7, page(0)).unwrap(), [0xCD; PAGE_SIZE]);
        assert_eq!(dm.num_writes.load(Ordering::Acquire), 3);
    }

    #[test]
    fn reading_an_unwritten_page_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let err = manager(&dir).read_page(42, page(0)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn freed_slots_are_reused_before_growing_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let dm = manager(&dir);
        dm.write_page(1, page(1)).unwrap();
        dm.store.lock().free_offsets.push(0);
        dm.write_page(2, page(2)).unwrap();
        assert_eq!(dm.store.lock().page_offsets.get(&2), Some(&0));
        assert_eq!(*dm.read_page(2, page(0)).unwrap(), [2; PAGE_SIZE]);
    }

    fn finishes_within(limit: std::time::Duration, work: impl FnOnce() + Send + 'static) -> bool {
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            work();
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(limit).is_ok()
    }

    #[test]
    fn read_error_is_returned_instead_of_deadlocking() {
        let dir = tempfile::tempdir().unwrap();
        let dm = manager(&dir);
        dm.write_page(1, page(1)).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join("db"))
            .unwrap()
            .set_len(0)
            .unwrap();
        let (result_tx, result_rx) = mpsc::channel();
        let finished = finishes_within(std::time::Duration::from_secs(5), move || {
            let _ = result_tx.send(dm.read_page(1, page(0)).map(|_| ()));
        });
        assert!(finished, "read_page hung while reporting an I/O error");
        let err = result_rx.recv().unwrap().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn concurrent_reads_and_writes_do_not_deadlock() {
        let dir = tempfile::tempdir().unwrap();
        let dm = Arc::new(manager(&dir));
        for id in 0..8 {
            dm.write_page(id, page(id as u8)).unwrap();
        }
        let finished = finishes_within(std::time::Duration::from_secs(20), move || {
            let workers: Vec<_> = (0..8)
                .map(|worker| {
                    let dm = dm.clone();
                    thread::spawn(move || {
                        for round in 0..2_000 {
                            let id = (worker + round) % 8;
                            if worker % 2 == 0 {
                                dm.write_page(id, page(id as u8)).unwrap();
                            } else {
                                let data = dm.read_page(id, page(0)).unwrap();
                                assert_eq!(*data, [id as u8; PAGE_SIZE], "page {id} was torn");
                            }
                        }
                    })
                })
                .collect();
            for w in workers {
                w.join().unwrap();
            }
        });
        assert!(finished, "concurrent read_page/write_page deadlocked");
    }

    #[test]
    fn scheduler_runs_requests_in_order_and_shuts_down() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = DiskScheduler::new(manager(&dir));
        let (wtx, wrx) = mpsc::channel();
        let (rtx, rrx) = mpsc::channel();
        scheduler
            .schedule(vec![
                DiskRequest {
                    page_id: 5,
                    is_write: true,
                    data: page(9),
                    promise: wtx,
                },
                DiskRequest {
                    page_id: 5,
                    is_write: false,
                    data: page(0),
                    promise: rtx,
                },
            ])
            .unwrap();
        wrx.recv().unwrap().unwrap();
        assert_eq!(*rrx.recv().unwrap().unwrap(), [9; PAGE_SIZE]);
        assert!(Arc::strong_count(&scheduler.disk_manager()) >= 2);
        scheduler.shutdown().unwrap();
    }

    #[test]
    fn scheduler_survives_a_dropped_promise() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = DiskScheduler::new(manager(&dir));
        let (tx, rx) = mpsc::channel();
        drop(rx);
        scheduler
            .schedule(vec![DiskRequest {
                page_id: 1,
                is_write: true,
                data: page(1),
                promise: tx,
            }])
            .unwrap();
        // The worker logs the failed delivery and keeps serving requests.
        let (tx, rx) = mpsc::channel();
        scheduler
            .schedule(vec![DiskRequest {
                page_id: 1,
                is_write: false,
                data: page(0),
                promise: tx,
            }])
            .unwrap();
        assert_eq!(*rx.recv().unwrap().unwrap(), [1; PAGE_SIZE]);
    }

    #[test]
    fn scheduler_errors_display() {
        let (tx, _rx) = mpsc::channel();
        let req = DiskRequest {
            page_id: 3,
            is_write: false,
            data: page(0),
            promise: tx,
        };
        let closed: DiskSchedulerError = SendError(req).into();
        assert_eq!(closed.to_string(), "failed to send disk request for page 3");
        assert_eq!(
            DiskSchedulerError::Shutdown.to_string(),
            "disk scheduler is shut down"
        );
        assert_eq!(
            DiskSchedulerError::WorkerPanicked.to_string(),
            "disk worker thread panicked"
        );
    }
}
