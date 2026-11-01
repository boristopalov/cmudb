use log::{info, warn};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    fmt,
    fs::{File, OpenOptions},
    io::{self, Read, Seek},
    os::unix::fs::FileExt,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
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
pub struct DiskManager {
    num_flushes: AtomicU32, // this is for log flushes, i think
    num_writes: AtomicU32,  // this is for page writes
    num_deletes: AtomicU32,
    log_file: Mutex<File>,
    page_lookup_table: Mutex<HashMap<usize, u64>>, // maps page IDs to file offsets in the db file
    free_slots: Mutex<Vec<u64>>,                   // keeps tracks of free offsets ("slots")
    should_flush_log: AtomicBool,
    db_file: Mutex<File>,
}

impl DiskManager {
    pub fn new<P: AsRef<Path>>(db_path: P, log_path: P) -> io::Result<Self> {
        let db_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(db_path)?;
        let log_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(log_path)?;

        Ok(Self {
            num_flushes: AtomicU32::new(0),
            num_writes: AtomicU32::new(0),
            num_deletes: AtomicU32::new(0),
            log_file: Mutex::new(log_file),
            page_lookup_table: Mutex::new(HashMap::new()),
            free_slots: Mutex::new(Vec::new()),
            should_flush_log: AtomicBool::new(false),
            db_file: Mutex::new(db_file),
        })
    }

    pub fn read_page(&self, id: usize, mut buf: PageData) -> std::io::Result<PageData> {
        debug_assert_eq!(buf.len(), PAGE_SIZE);
        let table = self.page_lookup_table.lock();

        let offset = match table.get(&id) {
            Some(offset) => *offset,
            None => {
                warn!("read_page: page_id={id} missing from lookup table");
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("page {id} missing"),
                ));
            }
        };

        info!("read_page: page_id={id} offset={offset}");

        let mut file = self.db_file.lock();
        let result: io::Result<()> = (|| {
            file.seek(io::SeekFrom::Start(offset))?;
            file.read_exact(&mut *buf)?;
            Ok(())
        })();

        if let Err(ref e) = result {
            self.log_page_mapping(&format!(
                "read_page error page_id={id} offset={offset} err={e}"
            ));
        }

        result.map(|_| buf)
    }

    pub fn write_page(&self, id: usize, mut data: PageData) -> std::io::Result<PageData> {
        debug_assert_eq!(data.len(), PAGE_SIZE);
        let file = self.db_file.lock();
        let mut table = self.page_lookup_table.lock();
        let mut free_slots = self.free_slots.lock();

        if let Some(&offset) = table.get(&id) {
            let res = file.write_all_at(&mut *data, offset);
            if let Err(ref e) = res {
                self.log_page_mapping(&format!(
                    "write_page error page_id={id} offset={offset} err={e}"
                ));
            }
            res?;
            table.insert(id, offset);
            info!("write_page: appended page_id={id} at offset={offset}");
            self.num_writes.fetch_add(1, Ordering::AcqRel);
            return Ok(data);
        };

        match free_slots.pop() {
            // we have a free slot
            Some(slot) => {
                let res = file.write_all_at(&mut *data, slot);
                if let Err(ref e) = res {
                    self.log_page_mapping(&format!(
                        "write_page error page_id={id} offset={slot} err={e}"
                    ));
                }
                res?;
                table.insert(id, slot);
                info!("write_page: reused slot offset={slot} for page_id={id}");
                slot
            }
            // free_slots is empty, directly append to the end of the file
            None => {
                let offset = file.metadata()?.len();
                let res = file.write_all_at(&mut *data, offset);
                if let Err(ref e) = res {
                    self.log_page_mapping(&format!(
                        "write_page error page_id={id} offset={offset} err={e}"
                    ));
                }
                res?;
                table.insert(id, offset);
                info!("write_page: appended page_id={id} at offset={offset}");
                offset
            }
        };
        self.num_writes.fetch_add(1, Ordering::AcqRel);

        Ok(data)
    }

    // pub fn delete_page(&self, id: usize) {}

    fn log_page_mapping(&self, context: &str) {
        let snapshot = self.page_lookup_table.lock().clone();
        warn!("{context}; page_id->offset mapping: {:?}", snapshot);
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
        dm.free_slots.lock().push(0);
        dm.write_page(2, page(2)).unwrap();
        assert_eq!(dm.page_lookup_table.lock().get(&2), Some(&0));
        assert_eq!(*dm.read_page(2, page(0)).unwrap(), [2; PAGE_SIZE]);
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
