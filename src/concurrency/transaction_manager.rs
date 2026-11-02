use super::transaction::{
    IsolationLevel, TXN_START_ID, Timestamp, Transaction, TransactionState, is_txn_id,
};
use crate::buffer_pool::BufferPoolManager;
use crate::table_heap::{RecordId, SlotWrite, TableHeap, TableHeapError};
use crate::table_page::TupleMeta;
use parking_lot::{Mutex, RwLock};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub enum TxnError {
    /// The tuple was changed by a transaction this one cannot see (one that is still running,
    /// or one that committed after this transaction's snapshot).
    WriteConflict(RecordId),
    /// The transaction is not in a state that allows the requested operation.
    InvalidState(TransactionState),
    Storage(String),
}

impl fmt::Display for TxnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TxnError::WriteConflict(rid) => write!(f, "write-write conflict on record ({rid})"),
            TxnError::InvalidState(s) => write!(f, "transaction is {s:?}"),
            TxnError::Storage(e) => write!(f, "storage error: {e}"),
        }
    }
}

impl From<TableHeapError> for TxnError {
    fn from(e: TableHeapError) -> Self {
        TxnError::Storage(e.0)
    }
}

/// One older version of a tuple: a full image of it and the timestamp that wrote it.
///
/// Undo entries form a singly linked list from newest to oldest. Entries are shared by
/// reference count, so a reader that grabbed a chain keeps it alive even while garbage
/// collection unlinks its tail.
#[derive(Debug)]
pub struct UndoLog {
    /// Commit timestamp of the version this entry restores. It is a commit timestamp and not
    /// a transaction id, since a transaction only overwrites versions it can see.
    pub ts: Timestamp,
    pub is_deleted: bool,
    pub data: Vec<u8>,
    prev: Mutex<Option<Arc<UndoLog>>>,
}

impl UndoLog {
    /// The next older version, if any.
    pub fn prev(&self) -> Option<Arc<UndoLog>> {
        self.prev.lock().clone()
    }
}

/// What a [`TransactionManager::garbage_collect`] pass reclaimed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GcStats {
    pub watermark: Timestamp,
    /// Undo entries unlinked (and freed once no reader holds them).
    pub undo_logs_removed: usize,
    /// Record ids whose whole chain was dropped.
    pub chains_removed: usize,
}

/// Hands out transactions and timestamps, owns the version chains, and commits, aborts
/// and garbage-collects.
///
/// Every transaction from [`TransactionManager::begin`] ends in `commit` or `abort`. Until it
/// does, its snapshot pins the GC watermark.
///
/// Lock order: page latch, then `versions`. The commit latch is taken before page latches.
pub struct TransactionManager {
    bpm: Arc<BufferPoolManager>,
    next_txn_id: AtomicU64,
    last_commit_ts: AtomicU64,
    /// Serializes commits so commit timestamps are published in order.
    commit_latch: Mutex<()>,
    /// read_ts -> number of running transactions reading at it. Its smallest key is the
    /// watermark.
    running: Mutex<BTreeMap<Timestamp, usize>>,
    /// Head of the undo chain of every record id that has older versions.
    versions: RwLock<HashMap<RecordId, Arc<UndoLog>>>,
}

impl TransactionManager {
    pub fn new(bpm: Arc<BufferPoolManager>) -> Self {
        Self {
            bpm,
            next_txn_id: AtomicU64::new(TXN_START_ID),
            last_commit_ts: AtomicU64::new(0),
            commit_latch: Mutex::new(()),
            running: Mutex::new(BTreeMap::new()),
            versions: RwLock::new(HashMap::new()),
        }
    }

    /// Starts a transaction reading as of the newest commit. It registers under the same lock
    /// the watermark is computed under, so no transaction begins with a snapshot older than a
    /// watermark GC acted on.
    pub fn begin(&self, isolation_level: IsolationLevel) -> Arc<Transaction> {
        let mut running = self.running.lock();
        let read_ts = self.last_commit_ts.load(Ordering::Acquire);
        let id = self.next_txn_id.fetch_add(1, Ordering::Relaxed);
        *running.entry(read_ts).or_insert(0) += 1;
        Arc::new(Transaction::new(id, read_ts, isolation_level))
    }

    /// The newest commit timestamp.
    pub fn last_commit_ts(&self) -> Timestamp {
        self.last_commit_ts.load(Ordering::Acquire)
    }

    /// The oldest snapshot any running transaction reads at, or the newest commit timestamp
    /// when nothing is running. No reader can ever need a version older than the newest one
    /// at or below the watermark.
    pub fn watermark(&self) -> Timestamp {
        let running = self.running.lock();
        running
            .keys()
            .next()
            .copied()
            .unwrap_or_else(|| self.last_commit_ts())
    }

    fn unregister(&self, txn: &Transaction) {
        let mut running = self.running.lock();
        if let Some(n) = running.get_mut(&txn.read_ts()) {
            *n -= 1;
            if *n == 0 {
                running.remove(&txn.read_ts());
            }
        }
    }

    /// Commits `txn`, stamping its writes with a fresh commit timestamp. A tainted
    /// transaction cannot commit, only abort. A read-only transaction takes no new timestamp.
    ///
    /// If stamping fails partway (a buffer pool error), the slots already stamped are put
    /// back to the transaction id, the transaction is tainted and its write set is kept, so
    /// [`TransactionManager::abort`] can still roll everything back. Nothing is published
    /// before every write is stamped, so un-stamping makes the failed commit invisible; if
    /// un-stamping fails too, the error says so rather than claim a clean rollback.
    ///
    /// The new timestamp is published only once every write carries it, so a transaction that
    /// begins at the commit timestamp sees all of them.
    pub fn commit(&self, txn: &Transaction) -> Result<Timestamp, TxnError> {
        match txn.state() {
            TransactionState::Running => {}
            s => return Err(TxnError::InvalidState(s)),
        }
        let rids: Vec<RecordId> = txn
            .write_entries()
            .into_iter()
            .map(|(rid, _)| rid)
            .collect();
        let read_only = rids.is_empty();
        if read_only {
            let ts = self.last_commit_ts();
            txn.finish(TransactionState::Committed, Some(ts));
            self.unregister(txn);
            return Ok(ts);
        }
        let _commit = self.commit_latch.lock();
        let commit_ts = self.last_commit_ts() + 1;
        for (i, rid) in rids.iter().enumerate() {
            if let Err(e) = self.stamp(*rid, txn.id(), commit_ts) {
                for done in &rids[..i] {
                    if let Err(undo) = self.stamp(*done, commit_ts, txn.id()) {
                        txn.set_tainted();
                        return Err(TxnError::Storage(format!(
                            "commit failed ({e}) and could not be undone on ({done}): {undo}"
                        )));
                    }
                }
                txn.set_tainted();
                return Err(e);
            }
        }
        self.last_commit_ts.store(commit_ts, Ordering::Release);
        txn.finish(TransactionState::Committed, Some(commit_ts));
        self.unregister(txn);
        Ok(commit_ts)
    }

    /// Re-stamps `rid` from timestamp `from` to `to`, keeping its deleted flag.
    fn stamp(&self, rid: RecordId, from: Timestamp, to: Timestamp) -> Result<(), TxnError> {
        TableHeap::write_slot(&self.bpm, rid, |meta, _| {
            if meta.ts != from {
                return Err(TxnError::Storage(format!(
                    "slot ({rid}) is stamped {} rather than {from}",
                    meta.ts
                )));
            }
            Ok(SlotWrite {
                meta: TupleMeta { ts: to, ..meta },
                data: None,
            })
        })
    }

    /// Rolls back every write of `txn`. Tuples it updated or deleted get their previous
    /// version back from the head of their undo chain; tuples it inserted become deletions
    /// at timestamp 0, invisible to everyone.
    ///
    /// Each write is forgotten as soon as it is rolled back. If one fails (a buffer pool
    /// error), the rest stay in the write set and the transaction stays registered, since
    /// its writes are still in the heap; calling `abort` again picks up where it stopped.
    ///
    /// While this transaction owns a slot nobody else can push onto its chain, and GC keeps
    /// the chain of every slot holding an uncommitted write, so the chain's head is the entry
    /// this transaction pushed. The head is popped only after its image is written back,
    /// under the same page latch, so a failed write leaves the chain whole for the retry.
    pub fn abort(&self, txn: &Transaction) -> Result<(), TxnError> {
        match txn.state() {
            TransactionState::Running | TransactionState::Tainted => {}
            s => return Err(TxnError::InvalidState(s)),
        }
        txn.set_tainted();
        for (rid, pushed_undo) in txn.write_entries() {
            TableHeap::write_slot_then(
                &self.bpm,
                rid,
                |meta, _| {
                    if meta.ts != txn.id() {
                        return Err(TxnError::Storage(format!(
                            "aborting slot ({rid}) stamped {} rather than this txn",
                            meta.ts
                        )));
                    }
                    if !pushed_undo {
                        let write = SlotWrite {
                            meta: TupleMeta {
                                ts: 0,
                                is_deleted: true,
                            },
                            data: None,
                        };
                        return Ok::<_, TxnError>((write, None));
                    }
                    let head = self.version_head(rid).ok_or_else(|| {
                        TxnError::Storage(format!("undo chain of ({rid}) vanished before abort"))
                    })?;
                    let write = SlotWrite {
                        meta: TupleMeta {
                            ts: head.ts,
                            is_deleted: head.is_deleted,
                        },
                        data: Some(head.data.clone()),
                    };
                    Ok((write, Some(head)))
                },
                |popped| {
                    if let Some(head) = popped {
                        let mut versions = self.versions.write();
                        match head.prev() {
                            Some(prev) => versions.insert(rid, prev),
                            None => versions.remove(&rid),
                        };
                    }
                },
            )?;
            txn.forget_write(rid);
        }
        txn.finish(TransactionState::Aborted, None);
        self.unregister(txn);
        Ok(())
    }

    /// The newest undo entry of `rid`. Read it under the slot's page latch, together with the
    /// slot itself, so the pair is consistent.
    pub fn version_head(&self, rid: RecordId) -> Option<Arc<UndoLog>> {
        self.versions.read().get(&rid).cloned()
    }

    /// Pushes an older version of `rid` onto its chain, from under the slot's page write
    /// latch once the slot is overwritten.
    pub(crate) fn push_version(&self, rid: RecordId, meta: TupleMeta, data: Vec<u8>) {
        debug_assert!(
            !is_txn_id(meta.ts),
            "undo entries hold committed versions only"
        );
        let mut versions = self.versions.write();
        let prev = versions.get(&rid).cloned();
        versions.insert(
            rid,
            Arc::new(UndoLog {
                ts: meta.ts,
                is_deleted: meta.is_deleted,
                data,
                prev: Mutex::new(prev),
            }),
        );
    }

    /// Number of record ids that currently have an undo chain.
    pub fn num_version_chains(&self) -> usize {
        self.versions.read().len()
    }

    /// Length of the undo chain of `rid`.
    pub fn version_chain_len(&self, rid: RecordId) -> usize {
        let mut len = 0;
        let mut cur = self.version_head(rid);
        while let Some(log) = cur {
            len += 1;
            cur = log.prev();
        }
        len
    }

    /// Unlinks every undo entry no running transaction can read.
    ///
    /// A reader at `read_ts >= watermark` stops at the first version with
    /// `ts <= read_ts`. So for each record id: if the heap version is committed at or below
    /// the watermark, the whole chain is dead; otherwise everything older than the first
    /// entry at or below the watermark is. Each slot is decided under its page latch, which
    /// keeps writers and aborts of that slot out meanwhile.
    pub fn garbage_collect(&self) -> Result<GcStats, TxnError> {
        let watermark = self.watermark();
        let mut stats = GcStats {
            watermark,
            ..GcStats::default()
        };
        let rids: Vec<RecordId> = self.versions.read().keys().copied().collect();
        for rid in rids {
            TableHeap::read_slot(&self.bpm, rid, |meta, _| {
                let mut versions = self.versions.write();
                let Some(head) = versions.get(&rid).cloned() else {
                    return;
                };
                if !is_txn_id(meta.ts) && meta.ts <= watermark {
                    versions.remove(&rid);
                    stats.chains_removed += 1;
                    stats.undo_logs_removed += chain_len(Some(head));
                    return;
                }
                let mut cur = Some(head);
                while let Some(log) = cur {
                    if log.ts <= watermark {
                        let tail = log.prev.lock().take();
                        stats.undo_logs_removed += chain_len(tail);
                        break;
                    }
                    cur = log.prev();
                }
            })?;
        }
        Ok(stats)
    }
}

fn chain_len(mut cur: Option<Arc<UndoLog>>) -> usize {
    let mut len = 0;
    while let Some(log) = cur {
        len += 1;
        cur = log.prev();
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::replacer::ArcReplacer;
    use crate::table_heap::Tuple;
    use tempfile::{TempDir, tempdir};

    fn setup() -> (Arc<TransactionManager>, TableHeap, TempDir) {
        let dir = tempdir().unwrap();
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).unwrap();
        let bpm = Arc::new(create_buffer_pool_manager(
            8,
            ArcReplacer::new(8),
            DiskScheduler::new(dm),
        ));
        let mgr = Arc::new(TransactionManager::new(bpm.clone()));
        (mgr, TableHeap::new(bpm).unwrap(), dir)
    }

    fn insert_committed(heap: &TableHeap, data: &[u8]) -> RecordId {
        heap.insert(&Tuple {
            data: data.to_vec(),
        })
        .unwrap()
    }

    /// The first write of `rid` in `txn`, as the MVCC layer makes it.
    fn overwrite(mgr: &TransactionManager, heap: &TableHeap, txn: &Transaction, rid: RecordId) {
        heap.write_with_then(
            rid,
            |meta, bytes| {
                let write = SlotWrite {
                    meta: TupleMeta {
                        ts: txn.id(),
                        is_deleted: false,
                    },
                    data: Some(vec![b'x'; bytes.len()]),
                };
                Ok::<_, TableHeapError>((write, (meta, bytes.to_vec())))
            },
            |(meta, bytes)| mgr.push_version(rid, meta, bytes),
        )
        .unwrap();
        txn.record_write(rid, true);
    }

    fn slot(heap: &TableHeap, rid: RecordId) -> (Timestamp, Vec<u8>) {
        let (meta, t) = heap.get_with_meta(rid).unwrap();
        (meta.ts, t.data)
    }

    #[test]
    fn a_commit_that_fails_midway_stays_invisible_and_abort_resumes() {
        let (mgr, heap, _dir) = setup();
        let rids: Vec<_> = [b"aaaa", b"bbbb", b"cccc"]
            .iter()
            .map(|d| insert_committed(&heap, *d))
            .collect();
        let txn = mgr.begin(IsolationLevel::SnapshotIsolation);
        for rid in &rids {
            overwrite(&mgr, &heap, &txn, *rid);
        }
        let stray = TupleMeta {
            ts: 7,
            is_deleted: false,
        };
        heap.update_meta(rids[1], stray).unwrap();

        assert!(mgr.commit(&txn).is_err());
        assert_eq!(txn.state(), TransactionState::Tainted);
        assert_eq!(mgr.last_commit_ts(), 0);
        assert_eq!(slot(&heap, rids[0]).0, txn.id());
        assert_eq!(slot(&heap, rids[1]).0, 7);
        assert_eq!(slot(&heap, rids[2]).0, txn.id());

        assert!(mgr.abort(&txn).is_err());
        assert_eq!(slot(&heap, rids[0]), (0, b"aaaa".to_vec()));
        assert_eq!(mgr.num_version_chains(), 2);

        heap.update_meta(
            rids[1],
            TupleMeta {
                ts: txn.id(),
                ..stray
            },
        )
        .unwrap();
        mgr.abort(&txn).unwrap();
        assert_eq!(txn.state(), TransactionState::Aborted);
        assert_eq!(mgr.num_version_chains(), 0);
        assert_eq!(slot(&heap, rids[1]), (0, b"bbbb".to_vec()));
        assert_eq!(slot(&heap, rids[2]), (0, b"cccc".to_vec()));
    }

    #[test]
    fn a_failed_restore_keeps_the_undo_chain_for_the_retry() {
        let (mgr, heap, _dir) = setup();
        let rid = insert_committed(&heap, b"abcd");
        let txn = mgr.begin(IsolationLevel::SnapshotIsolation);
        overwrite(&mgr, &heap, &txn, rid);
        mgr.push_version(rid, TupleMeta::default(), vec![0; 3]);
        assert_eq!(mgr.version_chain_len(rid), 2);

        assert!(mgr.abort(&txn).is_err());
        assert_eq!(mgr.version_chain_len(rid), 2);
        assert_eq!(slot(&heap, rid), (txn.id(), b"xxxx".to_vec()));

        let bad = mgr.version_head(rid).unwrap();
        mgr.versions.write().insert(rid, bad.prev().unwrap());
        mgr.abort(&txn).unwrap();
        assert_eq!(slot(&heap, rid), (0, b"abcd".to_vec()));
        assert_eq!(mgr.num_version_chains(), 0);
    }

    #[test]
    fn a_failed_first_write_pushes_no_undo_entry() {
        let (mgr, heap, _dir) = setup();
        let rid = insert_committed(&heap, b"abcd");
        let txn = mgr.begin(IsolationLevel::SnapshotIsolation);
        let res = heap.write_with_then(
            rid,
            |meta, bytes| {
                let write = SlotWrite {
                    meta: TupleMeta {
                        ts: txn.id(),
                        is_deleted: false,
                    },
                    data: Some(vec![0; bytes.len() + 1]),
                };
                Ok::<_, TableHeapError>((write, (meta, bytes.to_vec())))
            },
            |(meta, bytes)| mgr.push_version(rid, meta, bytes),
        );
        assert!(res.is_err());
        assert_eq!(mgr.num_version_chains(), 0);
        assert_eq!(slot(&heap, rid), (0, b"abcd".to_vec()));
    }
}
