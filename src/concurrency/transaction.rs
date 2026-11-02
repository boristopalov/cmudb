use crate::table_heap::RecordId;
use parking_lot::Mutex;
use std::collections::HashMap;

pub type TxnId = u64;
pub type Timestamp = u64;

/// Transaction ids are allocated from here upwards. A tuple stamped with a value at or above
/// this is an uncommitted write of the transaction with that id; anything below is a commit
/// timestamp. Snapshots are commit timestamps, so uncommitted writes of other transactions
/// sort after every snapshot and stay invisible to it.
pub const TXN_START_ID: TxnId = 1 << 62;

/// Whether `ts` is a transaction id (an uncommitted write) rather than a commit timestamp.
pub fn is_txn_id(ts: u64) -> bool {
    ts >= TXN_START_ID
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Running,
    /// A write failed (for example on a write-write conflict). The transaction can no longer
    /// commit, only abort.
    Tainted,
    Committed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    #[default]
    SnapshotIsolation,
}

#[derive(Debug)]
struct TxnInner {
    state: TransactionState,
    commit_ts: Option<Timestamp>,
    /// Every record id this transaction wrote, and whether the first write pushed an undo
    /// entry (false for tuples this transaction inserted).
    write_set: HashMap<RecordId, bool>,
}

/// A transaction handle. Created by [`super::TransactionManager::begin`] and shared with the
/// executors that run on its behalf.
#[derive(Debug)]
pub struct Transaction {
    id: TxnId,
    read_ts: Timestamp,
    isolation_level: IsolationLevel,
    inner: Mutex<TxnInner>,
}

impl Transaction {
    pub(crate) fn new(id: TxnId, read_ts: Timestamp, isolation_level: IsolationLevel) -> Self {
        Self {
            id,
            read_ts,
            isolation_level,
            inner: Mutex::new(TxnInner {
                state: TransactionState::Running,
                commit_ts: None,
                write_set: HashMap::new(),
            }),
        }
    }

    /// The transaction id, which also stamps its uncommitted writes until commit.
    pub fn id(&self) -> TxnId {
        self.id
    }

    /// The transaction id without the [`TXN_START_ID`] bit, for display only.
    pub fn display_id(&self) -> u64 {
        self.id ^ TXN_START_ID
    }

    /// The snapshot this transaction reads: every version committed at or before it.
    pub fn read_ts(&self) -> Timestamp {
        self.read_ts
    }

    pub fn isolation_level(&self) -> IsolationLevel {
        self.isolation_level
    }

    pub fn state(&self) -> TransactionState {
        self.inner.lock().state
    }

    pub fn commit_ts(&self) -> Option<Timestamp> {
        self.inner.lock().commit_ts
    }

    /// Marks a running transaction as unable to commit.
    pub fn set_tainted(&self) {
        let mut inner = self.inner.lock();
        if inner.state == TransactionState::Running {
            inner.state = TransactionState::Tainted;
        }
    }

    /// Record ids this transaction has written.
    pub fn write_set(&self) -> Vec<RecordId> {
        self.inner.lock().write_set.keys().copied().collect()
    }

    /// Records a write to `rid`. Only the first write decides whether an undo entry exists,
    /// since later writes by the same transaction overwrite its own version in place.
    pub(crate) fn record_write(&self, rid: RecordId, pushed_undo: bool) {
        self.inner
            .lock()
            .write_set
            .entry(rid)
            .or_insert(pushed_undo);
    }

    /// The write set as `(record id, pushed an undo entry)` pairs.
    pub(crate) fn write_entries(&self) -> Vec<(RecordId, bool)> {
        self.inner
            .lock()
            .write_set
            .iter()
            .map(|(rid, pushed)| (*rid, *pushed))
            .collect()
    }

    /// Forgets a write once it has been rolled back.
    pub(crate) fn forget_write(&self, rid: RecordId) {
        self.inner.lock().write_set.remove(&rid);
    }

    pub(crate) fn finish(&self, state: TransactionState, commit_ts: Option<Timestamp>) {
        let mut inner = self.inner.lock();
        inner.state = state;
        inner.commit_ts = commit_ts;
    }
}
