//! The write-ahead log of a database opened with [`crate::Database::open`].
//!
//! Pages are written to the log, not to the data file. A read of a page finds its newest
//! image in the log first and falls back to the data file. A durable point (a
//! [`Wal::durable_point`] record followed by an fsync) marks a state recovery can return to:
//! on open, the log is replayed up to its last durable point and everything after it is
//! dropped. Copying the logged pages into the data file and starting a fresh log is a
//! checkpoint; it happens at a durable point once the log outgrows
//! [`CHECKPOINT_LOG_BYTES`].
//!
//! Pages can reach the log while a transaction that wrote them is still running, so the log
//! also records what rolling such a transaction back needs: a [`TxnRecord::Writes`] before
//! its first write, a [`TxnRecord::Undo`] with the version each update replaced, and a
//! [`TxnRecord::End`] once it committed or aborted. A transaction with no `End` at the last
//! durable point is handed back by [`Wal::open`] as a [`RecoveredTxn`] to roll back.
//!
//! Each record is a kind byte, a little-endian `u32` payload length, the CRC-32 of the kind,
//! length and payload, and the payload. A record whose checksum does not match ends the log,
//! so a torn write at the tail is dropped with the rest of the tail.

use crate::concurrency::{Timestamp, TxnId};
use crate::disk::{PAGE_SIZE, PageData};
use crate::ids::PageId;
use crate::table_heap::RecordId;
use crate::table_page::TupleMeta;
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// The first bytes of every log file, followed by [`LOG_VERSION`].
const LOG_MAGIC: [u8; 8] = *b"CMUDBWAL";
/// The version of the record layout. A log with another version is refused.
const LOG_VERSION: u32 = 1;
const LOG_HEADER_LEN: u64 = LOG_MAGIC.len() as u64 + 4;

/// Kind byte, payload length, checksum.
const RECORD_HEADER_LEN: usize = 1 + 4 + 4;

/// The log size past which a durable point also checkpoints.
pub const CHECKPOINT_LOG_BYTES: u64 = 4 << 20;

const KIND_PAGE: u8 = 1;
const KIND_DURABLE: u8 = 2;
const KIND_WRITES: u8 = 3;
const KIND_UNDO: u8 = 4;
const KIND_END: u8 = 5;

/// Payload bytes of a page record before the page image: its page id.
const PAGE_ID_LEN: usize = 8;

/// A transaction's entry in the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnRecord {
    /// The transaction is about to write its first tuple.
    Writes(TxnId),
    /// The transaction overwrote `rid` for the first time, replacing the committed version
    /// `meta` / `data`.
    Undo {
        txn: TxnId,
        rid: RecordId,
        meta: TupleMeta,
        data: Vec<u8>,
    },
    /// The transaction committed or aborted, and every slot it wrote carries the outcome.
    End(TxnId),
}

/// The version an unfinished transaction replaced in one slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoImage {
    pub meta: TupleMeta,
    pub data: Vec<u8>,
}

/// A transaction that had written tuples and not ended at the log's last durable point. A
/// slot it stamped holds its uncommitted version: the one in `undo` replaced it, and a slot
/// with no entry there is one it inserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredTxn {
    pub id: TxnId,
    pub undo: HashMap<RecordId, UndoImage>,
}

/// What [`Wal::open`] found in the log.
#[derive(Debug, Default)]
pub struct Recovery {
    /// Pages copied from the log into the data file.
    pub pages_replayed: usize,
    /// Transactions to roll back, by id.
    pub unfinished: Vec<RecoveredTxn>,
}

/// The open log. Not synchronized; [`crate::disk::DiskManager`] keeps it behind its lock.
#[derive(Debug)]
pub struct Wal {
    path: PathBuf,
    file: File,
    /// Length of the log file: where the next record goes.
    end: u64,
    /// Transaction records not yet written to the file. They go out before the next page
    /// or durable point, so the file holds records in the order they were logged.
    buffered: Vec<u8>,
    /// Offset in the log of the newest image of each page logged since the last checkpoint.
    pages: HashMap<PageId, u64>,
    /// Something was logged since the last durable point.
    dirty: bool,
    /// The encoded records of every transaction that has written and not ended, kept so a
    /// checkpoint can carry them into the fresh log.
    open_txns: BTreeMap<TxnId, Vec<u8>>,
    checkpoint_bytes: u64,
    /// An fsync of the log, the data file or the log's directory failed. A later fsync of
    /// the same file can succeed without the bytes the failed one covered, so the log then
    /// refuses every page and durable point until the database is reopened and recovered.
    broken: bool,
    /// For tests: how many fsyncs succeed before one fails.
    #[cfg(test)]
    syncs_before_failure: Option<u32>,
}

/// A record read back from the log, with the page image left in the file.
enum Decoded {
    Page { page_id: PageId, image_offset: u64 },
    Durable,
    Txn(TxnRecord),
}

impl Wal {
    /// Opens the log at `path`, creating it if it is missing or empty. Every page logged up
    /// to the last durable point is copied into `data` (at `page_id * PAGE_SIZE`) and
    /// `data` is synced; a fresh log then replaces the old one, holding only the records of
    /// the transactions handed back in [`Recovery::unfinished`].
    pub fn open(path: &Path, data: &File) -> io::Result<(Self, Recovery)> {
        let (durable, pages) = Self::replay(path)?;
        let mut unfinished: BTreeMap<TxnId, RecoveredTxn> = BTreeMap::new();
        let mut open_txns: BTreeMap<TxnId, Vec<u8>> = BTreeMap::new();
        for record in durable {
            track_open_txn(&mut open_txns, &record);
            match record {
                TxnRecord::Writes(id) => {
                    unfinished.entry(id).or_insert_with(|| RecoveredTxn {
                        id,
                        undo: HashMap::new(),
                    });
                }
                TxnRecord::Undo {
                    txn,
                    rid,
                    meta,
                    data,
                } => {
                    unfinished
                        .entry(txn)
                        .or_insert_with(|| RecoveredTxn {
                            id: txn,
                            undo: HashMap::new(),
                        })
                        .undo
                        .entry(rid)
                        .or_insert(UndoImage { meta, data });
                }
                TxnRecord::End(id) => {
                    unfinished.remove(&id);
                }
            }
        }
        let mut wal = Self {
            path: path.to_path_buf(),
            file: OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)?,
            end: 0,
            buffered: Vec::new(),
            pages,
            dirty: false,
            open_txns,
            checkpoint_bytes: CHECKPOINT_LOG_BYTES,
            broken: false,
            #[cfg(test)]
            syncs_before_failure: None,
        };
        let pages_replayed = wal.pages.len();
        wal.checkpoint(data)?;
        Ok((
            wal,
            Recovery {
                pages_replayed,
                unfinished: unfinished.into_values().collect(),
            },
        ))
    }

    /// Reads the log at `path` and returns the transaction records and the newest image of
    /// each page up to its last durable point. A missing or empty file is an empty log.
    fn replay(path: &Path) -> io::Result<(Vec<TxnRecord>, HashMap<PageId, u64>)> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok((Vec::new(), HashMap::new()));
            }
            Err(e) => return Err(e),
        };
        let len = file.metadata()?.len();
        if len == 0 {
            return Ok((Vec::new(), HashMap::new()));
        }
        let mut reader = BufReader::new(file);
        let mut header = [0u8; LOG_HEADER_LEN as usize];
        reader
            .read_exact(&mut header)
            .map_err(|_| not_a_log("the log header is cut short"))?;
        if header[..LOG_MAGIC.len()] != LOG_MAGIC {
            return Err(not_a_log("the log header is not a cmudb log header"));
        }
        let version = u32::from_le_bytes(header[LOG_MAGIC.len()..].try_into().expect("4 bytes"));
        if version != LOG_VERSION {
            return Err(not_a_log("the log was written with another log version"));
        }

        let mut durable = Vec::new();
        let mut durable_pages = HashMap::new();
        let mut since_durable = Vec::new();
        let mut pages_since_durable = Vec::new();
        let mut offset = LOG_HEADER_LEN;
        while let Some((decoded, record_len)) = read_record(&mut reader, offset)? {
            offset += record_len;
            match decoded {
                Decoded::Page {
                    page_id,
                    image_offset,
                } => pages_since_durable.push((page_id, image_offset)),
                Decoded::Txn(record) => since_durable.push(record),
                Decoded::Durable => {
                    durable.append(&mut since_durable);
                    durable_pages.extend(pages_since_durable.drain(..));
                }
            }
        }
        Ok((durable, durable_pages))
    }

    /// Appends a page image, after the transaction records logged before it. Reads of
    /// `page_id` return it from now on.
    pub fn log_page(&mut self, page_id: PageId, page: &[u8; PAGE_SIZE]) -> io::Result<()> {
        self.refuse_if_broken()?;
        let mut payload = Vec::with_capacity(PAGE_ID_LEN + PAGE_SIZE);
        payload.extend_from_slice(&u64::from(page_id.get()).to_le_bytes());
        payload.extend_from_slice(page);
        let record_start = self.write_out_with(KIND_PAGE, &payload)?;
        self.pages.insert(
            page_id,
            record_start + (RECORD_HEADER_LEN + PAGE_ID_LEN) as u64,
        );
        self.dirty = true;
        Ok(())
    }

    /// Reads the newest logged image of `page_id` into `buf`, or returns `None` when the log
    /// holds no image of it.
    pub fn read_page(&self, page_id: PageId, buf: &mut PageData) -> Option<io::Result<()>> {
        let offset = *self.pages.get(&page_id)?;
        Some(self.file.read_exact_at(&mut buf[..], offset))
    }

    /// Logs a transaction record. It is kept in memory and reaches the file ahead of the
    /// next page image or durable point.
    pub fn log_txn(&mut self, record: &TxnRecord) {
        let mut encoded = Vec::new();
        encode_txn_record(&mut encoded, record);
        track_open_txn_encoded(&mut self.open_txns, record, &encoded);
        self.buffered.extend_from_slice(&encoded);
        self.dirty = true;
    }

    /// Makes everything logged so far the state recovery returns to, then checkpoints into
    /// `data` if the log has outgrown its limit. Does nothing when nothing was logged since
    /// the last durable point.
    pub fn durable_point(&mut self, data: &File) -> io::Result<()> {
        self.refuse_if_broken()?;
        if !self.dirty {
            return Ok(());
        }
        self.write_out_with(KIND_DURABLE, &[])?;
        self.sync(|wal| wal.file.sync_data())?;
        self.dirty = false;
        if self.end > self.checkpoint_bytes {
            self.checkpoint(data)?;
        }
        Ok(())
    }

    /// Copies every logged page into `data`, syncs it, and replaces the log with one that
    /// holds only the records of transactions still open, followed by a durable point. The
    /// replacement is written aside and renamed over the log, so a crash in between leaves the
    /// old log to replay again.
    fn checkpoint(&mut self, data: &File) -> io::Result<()> {
        let mut image: PageData = Box::new([0u8; PAGE_SIZE]);
        for (&page_id, &offset) in &self.pages {
            self.file.read_exact_at(&mut image[..], offset)?;
            data.write_all_at(&image[..], u64::from(page_id.get()) * PAGE_SIZE as u64)?;
        }
        self.sync(|_| data.sync_data())?;

        let mut fresh = Vec::new();
        fresh.extend_from_slice(&LOG_MAGIC);
        fresh.extend_from_slice(&LOG_VERSION.to_le_bytes());
        for records in self.open_txns.values() {
            fresh.extend_from_slice(records);
        }
        encode_record(&mut fresh, KIND_DURABLE, &[]);
        let aside = self.path.with_extension("log-new");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&aside)?;
        file.write_all_at(&fresh, 0)?;
        self.sync(|_| file.sync_data())?;
        std::fs::rename(&aside, &self.path)?;
        self.file = file;
        self.end = fresh.len() as u64;
        self.buffered.clear();
        self.pages.clear();
        self.dirty = false;
        self.sync(|wal| match wal.path.parent() {
            Some(dir) => File::open(dir)?.sync_all(),
            None => Ok(()),
        })
    }

    /// Runs one fsync, and marks the log broken if it fails.
    fn sync(&mut self, sync: impl FnOnce(&Self) -> io::Result<()>) -> io::Result<()> {
        #[cfg(test)]
        if let Some(left) = self.syncs_before_failure.as_mut() {
            if *left == 0 {
                self.syncs_before_failure = None;
                self.broken = true;
                return Err(io::Error::other("injected fsync failure"));
            }
            *left -= 1;
        }
        let result = sync(self);
        if result.is_err() {
            self.broken = true;
        }
        result
    }

    fn refuse_if_broken(&self) -> io::Result<()> {
        if self.broken {
            return Err(io::Error::other(
                "an earlier fsync of the write-ahead log failed, so it takes no more writes; \
                 reopen the database to recover from the log",
            ));
        }
        Ok(())
    }

    /// Writes the buffered transaction records and then one record of `kind`, returning the
    /// offset of that record. On failure the buffered records stay buffered.
    fn write_out_with(&mut self, kind: u8, payload: &[u8]) -> io::Result<u64> {
        let buffered = self.buffered.len();
        let record_start = self.end + buffered as u64;
        encode_record(&mut self.buffered, kind, payload);
        if let Err(e) = self.file.write_all_at(&self.buffered, self.end) {
            self.buffered.truncate(buffered);
            return Err(e);
        }
        self.end += self.buffered.len() as u64;
        self.buffered.clear();
        Ok(record_start)
    }

    /// How many bytes the log file holds, records still in memory excluded.
    #[cfg(test)]
    fn len(&self) -> u64 {
        self.end
    }

    /// Checkpoints at durable points once the log is longer than `bytes`.
    #[cfg(any(test, feature = "testing"))]
    pub fn set_checkpoint_bytes(&mut self, bytes: u64) {
        self.checkpoint_bytes = bytes;
    }
}

fn not_a_log(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.to_string())
}

fn track_open_txn(open: &mut BTreeMap<TxnId, Vec<u8>>, record: &TxnRecord) {
    let mut encoded = Vec::new();
    encode_txn_record(&mut encoded, record);
    track_open_txn_encoded(open, record, &encoded);
}

fn track_open_txn_encoded(open: &mut BTreeMap<TxnId, Vec<u8>>, record: &TxnRecord, encoded: &[u8]) {
    match record {
        TxnRecord::Writes(txn) | TxnRecord::Undo { txn, .. } => {
            open.entry(*txn).or_default().extend_from_slice(encoded);
        }
        TxnRecord::End(txn) => {
            open.remove(txn);
        }
    }
}

fn encode_record(out: &mut Vec<u8>, kind: u8, payload: &[u8]) {
    let len = u32::try_from(payload.len()).expect("a log record payload fits in a u32");
    let len_bytes = len.to_le_bytes();
    let crc = crc32(&[&[kind], &len_bytes, payload]);
    out.push(kind);
    out.extend_from_slice(&len_bytes);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(payload);
}

fn encode_txn_record(out: &mut Vec<u8>, record: &TxnRecord) {
    match record {
        TxnRecord::Writes(txn) => encode_record(out, KIND_WRITES, &txn.to_le_bytes()),
        TxnRecord::End(txn) => encode_record(out, KIND_END, &txn.to_le_bytes()),
        TxnRecord::Undo {
            txn,
            rid,
            meta,
            data,
        } => {
            let mut payload = Vec::with_capacity(8 + 4 + 4 + 8 + 1 + data.len());
            payload.extend_from_slice(&txn.to_le_bytes());
            payload.extend_from_slice(&rid.page_id.get().to_le_bytes());
            payload.extend_from_slice(&rid.slot_id.to_le_bytes());
            payload.extend_from_slice(&meta.ts.to_le_bytes());
            payload.push(u8::from(meta.is_deleted));
            payload.extend_from_slice(data);
            encode_record(out, KIND_UNDO, &payload);
        }
    }
}

/// Reads the record at `offset`, returning it and its length in the file, or `None` at the
/// end of the log: the end of the file, a record cut short, a checksum that does not match
/// or a kind this version does not know.
fn read_record(reader: &mut impl Read, offset: u64) -> io::Result<Option<(Decoded, u64)>> {
    let mut header = [0u8; RECORD_HEADER_LEN];
    if !read_fully(reader, &mut header)? {
        return Ok(None);
    }
    let kind = header[0];
    let len_bytes: [u8; 4] = header[1..5].try_into().expect("4 bytes");
    let crc = u32::from_le_bytes(header[5..9].try_into().expect("4 bytes"));
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_PAYLOAD_LEN {
        return Ok(None);
    }
    let mut payload = vec![0u8; len];
    if !read_fully(reader, &mut payload)? || crc32(&[&[kind], &len_bytes, &payload]) != crc {
        return Ok(None);
    }
    let record_len = (RECORD_HEADER_LEN + len) as u64;
    let decoded = match (kind, payload.len()) {
        (KIND_PAGE, n) if n == PAGE_ID_LEN + PAGE_SIZE => {
            let Ok(page_id) = PageId::try_from(u64_at(&payload, 0)) else {
                return Ok(None);
            };
            Decoded::Page {
                page_id,
                image_offset: offset + (RECORD_HEADER_LEN + PAGE_ID_LEN) as u64,
            }
        }
        (KIND_DURABLE, 0) => Decoded::Durable,
        (KIND_WRITES, 8) => Decoded::Txn(TxnRecord::Writes(u64_at(&payload, 0))),
        (KIND_END, 8) => Decoded::Txn(TxnRecord::End(u64_at(&payload, 0))),
        (KIND_UNDO, n) if n >= UNDO_FIXED_LEN => {
            let Ok(page_id) = PageId::try_from(u64::from(u32_at(&payload, 8))) else {
                return Ok(None);
            };
            Decoded::Txn(TxnRecord::Undo {
                txn: u64_at(&payload, 0),
                rid: RecordId {
                    page_id,
                    slot_id: u32_at(&payload, 12),
                },
                meta: TupleMeta {
                    ts: u64_at(&payload, 16) as Timestamp,
                    is_deleted: payload[24] != 0,
                },
                data: payload[UNDO_FIXED_LEN..].to_vec(),
            })
        }
        _ => return Ok(None),
    };
    Ok(Some((decoded, record_len)))
}

/// Transaction id, page id, slot id, timestamp and deleted flag of an undo record.
const UNDO_FIXED_LEN: usize = 8 + 4 + 4 + 8 + 1;

/// The longest payload a record can have: an undo record holding a whole page of tuple bytes.
const MAX_PAYLOAD_LEN: usize = UNDO_FIXED_LEN + PAGE_SIZE;

/// Fills `buf`, or returns `false` if the reader ends first.
fn read_fully(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<bool> {
    match reader.read_exact(buf) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

const CRC_TABLE: [u32; 256] = crc_table();

const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

/// CRC-32 (IEEE) of the concatenation of `parts`.
fn crc32(parts: &[&[u8]]) -> u32 {
    let mut c = !0u32;
    for part in parts {
        for &b in *part {
            c = CRC_TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
        }
    }
    !c
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Files {
        dir: tempfile::TempDir,
        data: File,
    }

    impl Files {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let data = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(dir.path().join("data"))
                .unwrap();
            Self { dir, data }
        }

        fn log_path(&self) -> PathBuf {
            self.dir.path().join("cmudb.log")
        }

        fn open(&self) -> (Wal, Recovery) {
            Wal::open(&self.log_path(), &self.data).unwrap()
        }

        fn data_page(&self, page_id: u32) -> Vec<u8> {
            let mut buf = vec![0u8; PAGE_SIZE];
            self.data
                .read_exact_at(&mut buf, u64::from(page_id) * PAGE_SIZE as u64)
                .unwrap();
            buf
        }
    }

    fn page(fill: u8) -> [u8; PAGE_SIZE] {
        [fill; PAGE_SIZE]
    }

    fn read(wal: &Wal, page_id: u32) -> Option<u8> {
        let mut buf: PageData = Box::new([0u8; PAGE_SIZE]);
        wal.read_page(PageId::new(page_id), &mut buf).map(|r| {
            r.unwrap();
            buf[0]
        })
    }

    fn rid(page_id: u32, slot_id: u32) -> RecordId {
        RecordId {
            page_id: PageId::new(page_id),
            slot_id,
        }
    }

    fn undo(txn: TxnId, rid: RecordId, ts: Timestamp, data: &[u8]) -> TxnRecord {
        TxnRecord::Undo {
            txn,
            rid,
            meta: TupleMeta {
                ts,
                is_deleted: false,
            },
            data: data.to_vec(),
        }
    }

    #[test]
    fn a_logged_page_reads_back_from_the_log_and_not_the_data_file() {
        let files = Files::new();
        let (mut wal, recovery) = files.open();
        assert_eq!(recovery.pages_replayed, 0);
        assert!(recovery.unfinished.is_empty());
        assert_eq!(read(&wal, 3), None);
        wal.log_page(PageId::new(3), &page(1)).unwrap();
        wal.log_page(PageId::new(3), &page(2)).unwrap();
        assert_eq!(read(&wal, 3), Some(2), "the newest image wins");
        assert_eq!(files.data.metadata().unwrap().len(), 0);
    }

    #[test]
    fn reopening_replays_pages_up_to_the_last_durable_point_only() {
        let files = Files::new();
        {
            let (mut wal, _) = files.open();
            wal.log_page(PageId::new(0), &page(1)).unwrap();
            wal.log_page(PageId::new(1), &page(1)).unwrap();
            wal.durable_point(&files.data).unwrap();
            wal.log_page(PageId::new(1), &page(2)).unwrap();
            wal.log_page(PageId::new(2), &page(2)).unwrap();
        }
        let (wal, recovery) = files.open();
        assert_eq!(recovery.pages_replayed, 2);
        assert_eq!(files.data_page(0), page(1));
        assert_eq!(
            files.data_page(1),
            page(1),
            "page 1's later image was not durable"
        );
        assert_eq!(
            files.data.metadata().unwrap().len(),
            2 * PAGE_SIZE as u64,
            "page 2 was never durable"
        );
        assert_eq!(read(&wal, 1), None, "replayed pages live in the data file");
        assert!(wal.len() > 0);
    }

    #[test]
    fn a_torn_tail_ends_the_log_at_the_last_whole_durable_point() {
        let files = Files::new();
        let durable_end = {
            let (mut wal, _) = files.open();
            wal.log_page(PageId::new(0), &page(1)).unwrap();
            wal.durable_point(&files.data).unwrap();
            let end = wal.len();
            wal.log_page(PageId::new(0), &page(2)).unwrap();
            wal.durable_point(&files.data).unwrap();
            end
        };
        let log = OpenOptions::new()
            .write(true)
            .open(files.log_path())
            .unwrap();
        log.set_len(durable_end + (RECORD_HEADER_LEN + PAGE_ID_LEN + 100) as u64)
            .unwrap();
        files.open();
        assert_eq!(files.data_page(0), page(1));
    }

    #[test]
    fn a_flipped_byte_is_caught_by_the_checksum() {
        let files = Files::new();
        let page_start = {
            let (mut wal, _) = files.open();
            wal.log_page(PageId::new(0), &page(1)).unwrap();
            wal.durable_point(&files.data).unwrap();
            let start = wal.len();
            wal.log_page(PageId::new(0), &page(2)).unwrap();
            wal.durable_point(&files.data).unwrap();
            start
        };
        let log = OpenOptions::new()
            .write(true)
            .open(files.log_path())
            .unwrap();
        log.write_all_at(&[9], page_start + 100).unwrap();
        files.open();
        assert_eq!(files.data_page(0), page(1));
    }

    #[test]
    fn a_transaction_without_an_end_is_handed_back_with_its_first_undo_image() {
        let files = Files::new();
        {
            let (mut wal, _) = files.open();
            wal.log_txn(&TxnRecord::Writes(10));
            wal.log_txn(&undo(10, rid(1, 0), 3, b"old"));
            wal.log_txn(&undo(10, rid(1, 0), 4, b"newer"));
            wal.log_txn(&TxnRecord::Writes(11));
            wal.log_txn(&TxnRecord::Writes(12));
            wal.log_txn(&undo(12, rid(2, 5), 1, b"x"));
            wal.log_txn(&TxnRecord::End(12));
            wal.durable_point(&files.data).unwrap();
            wal.log_txn(&TxnRecord::End(10));
        }
        let (_, recovery) = files.open();
        let mut unfinished = recovery.unfinished;
        unfinished.sort_by_key(|t| t.id);
        assert_eq!(unfinished.len(), 2, "{unfinished:?}");
        assert_eq!(unfinished[0].id, 10, "its end was not durable");
        assert_eq!(
            unfinished[0].undo[&rid(1, 0)],
            UndoImage {
                meta: TupleMeta {
                    ts: 3,
                    is_deleted: false
                },
                data: b"old".to_vec()
            }
        );
        assert_eq!(unfinished[1].id, 11);
        assert!(unfinished[1].undo.is_empty(), "11 only inserted");
    }

    #[test]
    fn a_checkpoint_carries_open_transactions_into_the_fresh_log() {
        let files = Files::new();
        {
            let (mut wal, _) = files.open();
            wal.set_checkpoint_bytes(0);
            wal.log_txn(&TxnRecord::Writes(7));
            wal.log_txn(&undo(7, rid(3, 1), 2, b"before"));
            wal.log_txn(&TxnRecord::Writes(8));
            wal.log_page(PageId::new(3), &page(7)).unwrap();
            wal.log_txn(&TxnRecord::End(8));
            wal.durable_point(&files.data).unwrap();
            assert_eq!(files.data_page(3), page(7), "the checkpoint wrote the page");
            assert_eq!(read(&wal, 3), None);
            assert!(!files.dir.path().join("cmudb.log-new").exists());
        }
        let (_, recovery) = files.open();
        assert_eq!(recovery.pages_replayed, 0);
        assert_eq!(recovery.unfinished.len(), 1);
        assert_eq!(recovery.unfinished[0].id, 7);
        assert_eq!(recovery.unfinished[0].undo.len(), 1);
    }

    #[test]
    fn a_failed_log_sync_refuses_every_later_write_until_a_reopen() {
        let files = Files::new();
        {
            let (mut wal, _) = files.open();
            wal.log_page(PageId::new(0), &page(1)).unwrap();
            wal.durable_point(&files.data).unwrap();
            wal.syncs_before_failure = Some(0);
            wal.log_page(PageId::new(0), &page(2)).unwrap();
            wal.durable_point(&files.data).unwrap_err();
            wal.log_page(PageId::new(1), &page(3))
                .expect_err("no page is logged after a failed sync");
            wal.log_txn(&TxnRecord::Writes(7));
            wal.durable_point(&files.data)
                .expect_err("a later sync could succeed without the bytes the failed one held");
        }
        let (mut wal, _) = files.open();
        assert_eq!(
            files.data_page(0),
            page(2),
            "reopening recovers from the log"
        );
        wal.log_page(PageId::new(1), &page(3)).unwrap();
        wal.durable_point(&files.data).unwrap();
    }

    #[test]
    fn a_failed_directory_sync_after_a_checkpoint_breaks_the_log() {
        let files = Files::new();
        {
            let (mut wal, _) = files.open();
            wal.set_checkpoint_bytes(0);
            wal.syncs_before_failure = Some(3);
            wal.log_page(PageId::new(0), &page(1)).unwrap();
            wal.durable_point(&files.data)
                .expect_err("the log, data file and new log synced; the directory failed");
            assert_eq!(
                files.data_page(0),
                page(1),
                "the checkpoint copied the page"
            );
            wal.log_page(PageId::new(0), &page(2)).unwrap_err();
            wal.durable_point(&files.data).unwrap_err();
        }
        let (_, recovery) = files.open();
        assert_eq!(recovery.pages_replayed, 0);
        assert_eq!(files.data_page(0), page(1));
    }

    #[test]
    fn a_failed_data_file_sync_in_a_checkpoint_keeps_the_old_log() {
        let files = Files::new();
        {
            let (mut wal, _) = files.open();
            wal.set_checkpoint_bytes(0);
            wal.syncs_before_failure = Some(1);
            wal.log_page(PageId::new(0), &page(1)).unwrap();
            wal.durable_point(&files.data).unwrap_err();
            wal.durable_point(&files.data).unwrap_err();
        }
        let (_, recovery) = files.open();
        assert_eq!(
            recovery.pages_replayed, 1,
            "the old log still held the page"
        );
        assert_eq!(files.data_page(0), page(1));
    }

    #[test]
    fn a_durable_point_with_nothing_logged_writes_nothing() {
        let files = Files::new();
        let (mut wal, _) = files.open();
        let len = wal.len();
        wal.durable_point(&files.data).unwrap();
        assert_eq!(wal.len(), len);
    }

    #[test]
    fn transaction_records_wait_in_memory_for_the_next_page() {
        let files = Files::new();
        let (mut wal, _) = files.open();
        let len = wal.len();
        wal.log_txn(&TxnRecord::Writes(1));
        assert_eq!(wal.len(), len);
        wal.log_page(PageId::new(0), &page(0)).unwrap();
        assert!(wal.len() > len + PAGE_SIZE as u64);
    }

    #[test]
    fn a_failed_page_write_keeps_the_records_before_it_buffered() {
        let files = Files::new();
        let (mut wal, _) = files.open();
        wal.log_txn(&TxnRecord::Writes(1));
        let writable = std::mem::replace(&mut wal.file, File::open(files.log_path()).unwrap());
        wal.log_page(PageId::new(0), &page(0)).unwrap_err();
        assert_eq!(read(&wal, 0), None);
        wal.file = writable;
        wal.durable_point(&files.data).unwrap();
        drop(wal);
        let (_, recovery) = files.open();
        assert_eq!(recovery.unfinished.len(), 1, "the record was still logged");
    }

    #[test]
    fn a_file_that_is_not_a_log_is_refused() {
        let files = Files::new();
        std::fs::write(files.log_path(), b"definitely not a log").unwrap();
        let err = Wal::open(&files.log_path(), &files.data).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        std::fs::write(files.log_path(), b"CMU").unwrap();
        let err = Wal::open(&files.log_path(), &files.data).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let mut other_version = LOG_MAGIC.to_vec();
        other_version.extend_from_slice(&(LOG_VERSION + 1).to_le_bytes());
        std::fs::write(files.log_path(), other_version).unwrap();
        let err = Wal::open(&files.log_path(), &files.data).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(&[b"123456789"]), 0xCBF4_3926);
        assert_eq!(crc32(&[b"1234", b"56789"]), 0xCBF4_3926);
    }
}
