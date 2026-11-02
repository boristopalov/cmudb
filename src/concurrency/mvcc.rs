//! Visibility and write rules shared by the executors.

use super::transaction::{Transaction, is_txn_id};
use super::transaction_manager::{TransactionManager, TxnError};
use crate::catalog::IndexInfo;
use crate::catalog::index_schema::{IndexKey, TableIndex};
use crate::index::IndexError;
use crate::table_heap::{RecordId, SlotWrite, TableHeap, TableHeapError, Tuple};
use crate::table_page::TupleMeta;
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum MvccError {
    Txn(TxnError),
    /// The key is already taken by a version this transaction can see.
    DuplicateKey(String),
    Index(IndexError),
}

impl fmt::Display for MvccError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MvccError::Txn(e) => write!(f, "{e}"),
            MvccError::DuplicateKey(index) => {
                write!(f, "DuplicateKey: duplicate key in index {index}")
            }
            MvccError::Index(e) => write!(f, "index error: {e}"),
        }
    }
}

impl From<TxnError> for MvccError {
    fn from(e: TxnError) -> Self {
        MvccError::Txn(e)
    }
}

impl From<TableHeapError> for MvccError {
    fn from(e: TableHeapError) -> Self {
        MvccError::Txn(e.into())
    }
}

impl From<IndexError> for MvccError {
    fn from(e: IndexError) -> Self {
        MvccError::Index(e)
    }
}

/// Whether `txn` sees a version stamped `ts` without consulting older versions: its own
/// writes, and anything committed at or before its snapshot.
fn sees(txn: &Transaction, ts: u64) -> bool {
    ts == txn.id() || ts <= txn.read_ts()
}

/// Returns the bytes of the version of `rid` that `txn` sees, or `None` if it sees no live
/// version. `meta` and `bytes` are the newest version from the heap.
///
/// Call with the slot's page latch held, so the slot and its undo chain are read as one
/// consistent pair. Undo entries are ordered newest first and each holds a full image, so
/// the version `txn` sees is the first one at or before its snapshot.
pub fn visible_version(
    txn: &Transaction,
    txn_mgr: &TransactionManager,
    rid: RecordId,
    meta: TupleMeta,
    bytes: &[u8],
) -> Option<Vec<u8>> {
    if sees(txn, meta.ts) {
        return (!meta.is_deleted).then(|| bytes.to_vec());
    }
    let mut cur = txn_mgr.version_head(rid);
    while let Some(log) = cur {
        if log.ts <= txn.read_ts() {
            return (!log.is_deleted).then(|| log.data.clone());
        }
        cur = log.prev();
    }
    None
}

/// Reads the version of `rid` that `txn` sees.
pub fn read_visible(
    txn: &Transaction,
    txn_mgr: &TransactionManager,
    heap: &TableHeap,
    rid: RecordId,
) -> Result<Option<Tuple>, TableHeapError> {
    heap.read_with(rid, |meta, bytes| {
        visible_version(txn, txn_mgr, rid, meta, bytes).map(|data| Tuple { data })
    })
}

/// Reads the version of `rid` that `txn` sees, if it is still filed under `key` in `index`.
/// Index entries outlive the versions that created them, so a lookup re-checks the key.
pub fn read_visible_through_index(
    txn: &Transaction,
    txn_mgr: &TransactionManager,
    heap: &TableHeap,
    index: &TableIndex,
    key: &IndexKey,
    rid: RecordId,
) -> Result<Option<Tuple>, MvccError> {
    let Some(tuple) = read_visible(txn, txn_mgr, heap, rid)? else {
        return Ok(None);
    };
    let still_has_key = index.key_of_tuple(&tuple)?.as_ref() == Some(key);
    Ok(still_has_key.then_some(tuple))
}

/// Taints `txn` if `res` failed: a statement that fails halfway leaves partial writes behind,
/// and there are no statement-level savepoints, so the whole transaction can only abort.
fn taint_on_err<T>(txn: &Transaction, res: Result<T, MvccError>) -> Result<T, MvccError> {
    if res.is_err() {
        txn.set_tainted();
    }
    res
}

/// Fails with a write conflict when the slot holds another transaction's uncommitted write,
/// or a commit after `txn`'s snapshot. The first writer wins and the second aborts without
/// waiting.
fn check_write(txn: &Transaction, rid: RecordId, meta: TupleMeta) -> Result<(), TxnError> {
    let written_by_other_txn = meta.ts != txn.id();
    let uncommitted_or_newer = is_txn_id(meta.ts) || meta.ts > txn.read_ts();
    if written_by_other_txn && uncommitted_or_newer {
        return Err(TxnError::WriteConflict(rid));
    }
    Ok(())
}

/// Writes a new version of `rid`: `Some(tuple)` to update it, `None` to delete it.
///
/// A deleted slot is only reachable if a caller writes a tuple it cannot see, and is refused.
/// The first write of a tuple in a transaction keeps the version it replaces on the undo
/// chain. That push happens only after the slot is overwritten, under the same page latch,
/// so a failed write leaves no orphan undo entry.
fn write_version(
    txn: &Transaction,
    txn_mgr: &TransactionManager,
    heap: &TableHeap,
    rid: RecordId,
    new: Option<&Tuple>,
) -> Result<(), MvccError> {
    let pushed_undo = heap.write_with_then(
        rid,
        |meta, bytes| {
            check_write(txn, rid, meta)?;
            if meta.is_deleted {
                return Err(MvccError::from(TableHeapError::Deleted { rid }));
            }
            let first_write_in_txn = meta.ts != txn.id();
            let undo = first_write_in_txn.then(|| (meta, bytes.to_vec()));
            let write = SlotWrite {
                meta: TupleMeta {
                    ts: txn.id(),
                    is_deleted: new.is_none(),
                },
                data: new.map(|t| t.data.clone()),
            };
            Ok((write, undo))
        },
        |undo| match undo {
            Some((meta, bytes)) => {
                txn_mgr.push_version(rid, meta, bytes);
                true
            }
            None => false,
        },
    )?;
    txn.record_write(rid, pushed_undo);
    Ok(())
}

/// Indexes sorted by oid, the one order their unique latches are taken in.
fn sorted(indexes: &[Arc<IndexInfo>]) -> Vec<&Arc<IndexInfo>> {
    let mut out: Vec<_> = indexes.iter().collect();
    out.sort_by_key(|i| i.oid);
    out
}

/// Fails if the key `tuple` has in `info` belongs to a live version other than `own`.
/// Call with the index's unique latch held.
///
/// It judges the newest version behind each entry, not the one `txn` can see, so two live
/// rows with the same key cannot both commit. A row another transaction is still writing
/// conflicts even when its new version left the key, since an abort could bring the key
/// back. A committed row that no longer holds the key is a stale entry and is skipped.
fn check_unique(
    txn: &Transaction,
    heap: &TableHeap,
    info: &IndexInfo,
    tuple: &Tuple,
    own: Option<RecordId>,
) -> Result<(), MvccError> {
    let index = &info.index;
    let Some(key) = index.key_of_tuple(tuple)? else {
        return Ok(());
    };
    for rid in index.entries_for_key(&key)? {
        if Some(rid) == own {
            continue;
        }
        let (meta, stored_key) = heap.read_with(rid, |meta, bytes| {
            let t = Tuple {
                data: bytes.to_vec(),
            };
            (meta, index.key_of_tuple(&t))
        })?;
        let holds_key = stored_key?.as_ref() == Some(&key);
        let pending_elsewhere = is_txn_id(meta.ts) && meta.ts != txn.id();
        if !holds_key && !pending_elsewhere {
            continue;
        }
        check_write(txn, rid, meta)?;
        if !meta.is_deleted && holds_key {
            return Err(MvccError::DuplicateKey(info.name.clone()));
        }
    }
    Ok(())
}

/// Inserts `tuple` as a new uncommitted version and files it in every index.
pub fn insert_tuple(
    txn: &Transaction,
    heap: &TableHeap,
    indexes: &[Arc<IndexInfo>],
    tuple: &Tuple,
) -> Result<RecordId, MvccError> {
    taint_on_err(txn, {
        let indexes = sorted(indexes);
        let _latches: Vec<_> = indexes
            .iter()
            .map(|i| i.index.unique_latch.lock())
            .collect();
        (|| {
            for info in &indexes {
                check_unique(txn, heap, info, tuple, None)?;
            }
            let rid = heap.insert_with_meta(
                TupleMeta {
                    ts: txn.id(),
                    is_deleted: false,
                },
                tuple,
            )?;
            txn.record_write(rid, false);
            for info in &indexes {
                info.index.insert(tuple, rid)?;
            }
            Ok(rid)
        })()
    })
}

/// Replaces the version of `rid` that `txn` sees (`old`) with `new`, in place. Indexes whose
/// key changed get an entry for the new key; the old entry stays for older snapshots.
pub fn update_tuple(
    txn: &Transaction,
    txn_mgr: &TransactionManager,
    heap: &TableHeap,
    indexes: &[Arc<IndexInfo>],
    rid: RecordId,
    old: &Tuple,
    new: &Tuple,
) -> Result<(), MvccError> {
    taint_on_err(txn, {
        (|| {
            write_version(txn, txn_mgr, heap, rid, Some(new))?;
            for info in sorted(indexes) {
                let index = &info.index;
                let new_key = index.key_of_tuple(new)?;
                if new_key.is_none() || index.key_of_tuple(old)? == new_key {
                    continue;
                }
                let _latch = index.unique_latch.lock();
                check_unique(txn, heap, info, new, Some(rid))?;
                index.insert(new, rid)?;
            }
            Ok(())
        })()
    })
}

/// Deletes the version of `rid` that `txn` sees. Index entries stay for older snapshots.
pub fn delete_tuple(
    txn: &Transaction,
    txn_mgr: &TransactionManager,
    heap: &TableHeap,
    rid: RecordId,
) -> Result<(), MvccError> {
    taint_on_err(txn, write_version(txn, txn_mgr, heap, rid, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::catalog::{Catalog, CatalogError, DataType, IndexType, Schema, TableInfo, Value};
    use crate::concurrency::{IsolationLevel, TransactionState};
    use tempfile::TempDir;

    struct ShortRowTable {
        mgr: TransactionManager,
        table: Arc<TableInfo>,
        indexes: Vec<Arc<IndexInfo>>,
        short_rid: RecordId,
        _catalog: Catalog,
        _dir: TempDir,
    }

    fn short_tuple(v: i32) -> Tuple {
        Schema::new(vec![(DataType::INT, "id".to_string())])
            .encode_tuple(&[Some(Value::INT(v))])
            .unwrap()
    }

    /// A table `(id, v)` indexed on `v`, holding one committed row too short to have a `v`.
    fn table_with_a_short_row() -> ShortRowTable {
        let (catalog, bpm, dir) = crate::testing::catalog(32);
        let schema = Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::INT, "v".to_string()),
        ]);
        let oid = catalog.create_table("t".into(), schema).unwrap();
        catalog
            .create_index(bpm.clone(), oid, "t_v".into(), &["v"], IndexType::BPlusTree)
            .unwrap();
        let table = catalog.get_table(oid).unwrap();
        let short_rid = table.heap.insert(&short_tuple(1)).unwrap();
        ShortRowTable {
            mgr: TransactionManager::new(bpm),
            indexes: catalog.get_table_indexes(oid).unwrap(),
            table,
            short_rid,
            _catalog: catalog,
            _dir: dir,
        }
    }

    fn is_short_tuple_error(err: &MvccError) -> bool {
        matches!(
            err,
            MvccError::Index(IndexError::Catalog(CatalogError::TupleSchemaMismatch))
        )
    }

    #[test]
    fn update_to_a_short_tuple_fails_after_the_write_and_taints_the_transaction() {
        let t = table_with_a_short_row();
        let txn = t.mgr.begin(IsolationLevel::SnapshotIsolation);

        let err = update_tuple(
            &txn,
            &t.mgr,
            &t.table.heap,
            &t.indexes,
            t.short_rid,
            &short_tuple(1),
            &short_tuple(2),
        )
        .expect_err("a new version too short for the indexed column is an error");

        assert!(is_short_tuple_error(&err), "unexpected error: {err}");
        let (written_by, data) = t
            .table
            .heap
            .read_with(t.short_rid, |meta, bytes| (meta.ts, bytes.to_vec()))
            .unwrap();
        assert_eq!(
            (written_by, data),
            (txn.id(), short_tuple(2).data),
            "the slot holds the new version before the key error"
        );
        assert_eq!(
            txn.state(),
            TransactionState::Tainted,
            "the slot was already overwritten, so the transaction can only abort"
        );
        let entries: Vec<_> = t.indexes[0]
            .index
            .scan(std::ops::Bound::Unbounded, std::ops::Bound::Unbounded)
            .unwrap()
            .collect();
        assert!(entries.is_empty(), "no index entry was added");
    }

    #[test]
    fn reading_a_short_tuple_through_an_index_is_an_error() {
        let t = table_with_a_short_row();
        let txn = t.mgr.begin(IsolationLevel::SnapshotIsolation);

        let err = read_visible_through_index(
            &txn,
            &t.mgr,
            &t.table.heap,
            &t.indexes[0].index,
            &IndexKey(Vec::new()),
            t.short_rid,
        )
        .expect_err("a visible tuple too short for the index key is an error");

        assert!(is_short_tuple_error(&err), "unexpected error: {err}");
    }
}
