use crate::buffer_pool::{BpmError, BufferPoolManager};
use crate::table_page::{
    NO_NEXT_PAGE, TablePage, TablePageError, TablePageMut, TablePageRef, TupleMeta,
};
use parking_lot::{Mutex, MutexGuard};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub struct TableHeapError(pub String);

impl From<TablePageError> for TableHeapError {
    fn from(err: TablePageError) -> Self {
        TableHeapError(err.0)
    }
}

impl From<BpmError> for TableHeapError {
    fn from(err: BpmError) -> Self {
        TableHeapError(format!("buffer pool error: {err}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecordId {
    pub page_id: u32,
    pub slot_id: u32,
}

impl RecordId {
    pub const RESERVED: RecordId = RecordId {
        page_id: u32::MAX,
        slot_id: u32::MAX,
    };
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "page_id: {}, slot_id: {}", self.page_id, self.slot_id)
    }
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub data: Vec<u8>,
}

pub struct TableHeap {
    bpm: Arc<BufferPoolManager>,
    /// The page where the table starts.
    pub first_page_id: usize,
    tail: Mutex<Tail>,
}

struct Tail {
    page_id: usize,
    num_tuples: u32,
}

/// Streams the live `(Tuple, RecordId)` pairs that were in the heap when the iterator was created.
/// Rows inserted afterwards (for example the new versions written by an UPDATE that is scanning the
/// same heap) are not yielded. It holds its own buffer pool handle, so it does not borrow the heap.
pub struct TableHeapIterator {
    bpm: Arc<BufferPoolManager>,
    current_page_id: u32,
    current_slot_id: u32,
    last_page_id: u32,
    last_page_slots: u32,
    finished: bool,
}

impl TableHeapIterator {
    /// Advances to the next slot and calls `f` with its record id, metadata and bytes while
    /// the page's read latch is held, so `f` sees the slot exactly as one writer left it.
    /// Slots for which `f` returns `None` are skipped. Every slot is visited, deleted ones
    /// included; callers decide what is visible. The rest of a page is visited under one
    /// latch acquisition.
    pub fn next_with<R>(
        &mut self,
        mut f: impl FnMut(RecordId, TupleMeta, &[u8]) -> Option<R>,
    ) -> Option<Result<R, TableHeapError>> {
        loop {
            if self.finished
                || (self.current_page_id == self.last_page_id
                    && self.current_slot_id >= self.last_page_slots)
            {
                self.finished = true;
                return None;
            }
            let page_guard = match self.bpm.read_page(self.current_page_id as usize) {
                Ok(g) => g,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(TableHeapError(format!(
                        "failed to read page {} during scan: {e}",
                        self.current_page_id
                    ))));
                }
            };

            let page = match TablePageRef::new(page_guard.data()) {
                Ok(p) => p,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(TableHeapError(format!(
                        "failed to read page {} during scan: {}",
                        self.current_page_id, e.0
                    ))));
                }
            };

            loop {
                if self.current_page_id == self.last_page_id
                    && self.current_slot_id >= self.last_page_slots
                {
                    self.finished = true;
                    return None;
                }
                let slot_id = self.current_slot_id;
                let Some(slot) = page.slot(slot_id) else {
                    match page.next_page_id() {
                        Some(next) if self.current_page_id != self.last_page_id => {
                            self.current_page_id = next;
                            self.current_slot_id = 0;
                            break;
                        }
                        _ => {
                            self.finished = true;
                            return None;
                        }
                    }
                };
                self.current_slot_id += 1;
                let bytes = match page.tuple_bytes(&slot) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        self.finished = true;
                        return Some(Err(TableHeapError(format!(
                            "corrupt slot {slot_id} on page {} during scan: {}",
                            self.current_page_id, e.0
                        ))));
                    }
                };
                let rid = RecordId {
                    page_id: self.current_page_id,
                    slot_id,
                };
                if let Some(out) = f(rid, slot.meta, bytes) {
                    return Some(Ok(out));
                }
            }
        }
    }
}

/// Plain iteration yields every tuple whose newest version is not a deletion, ignoring
/// timestamps. Transactional reads go through [`TableHeapIterator::next_with`] instead.
impl Iterator for TableHeapIterator {
    type Item = Result<(Tuple, RecordId), TableHeapError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_with(|rid, meta, bytes| {
            (!meta.is_deleted).then(|| {
                (
                    Tuple {
                        data: bytes.to_vec(),
                    },
                    rid,
                )
            })
        })
    }
}

/// The new state of a slot written by [`TableHeap::write_with`].
pub struct SlotWrite {
    pub meta: TupleMeta,
    /// Replacement bytes, or `None` to keep the current ones.
    pub data: Option<Vec<u8>>,
}

impl TableHeap {
    pub fn new(bpm: Arc<BufferPoolManager>) -> Result<Self, TableHeapError> {
        let page_id = {
            let mut page_guard = bpm
                .new_page()
                .map_err(|e| TableHeapError(format!("failed to allocate first page: {e}")))?;
            let page_id = record_page_id(page_guard.page_id())?;
            TablePageMut::init(page_guard.data_mut());
            page_id as usize
        };
        Ok(Self {
            bpm,
            first_page_id: page_id,
            tail: Mutex::new(Tail {
                page_id,
                num_tuples: 0,
            }),
        })
    }

    pub fn iter(&self) -> TableHeapIterator {
        let tail = self.tail.lock();
        TableHeapIterator {
            bpm: self.bpm.clone(),
            current_page_id: self.first_page_id as u32,
            current_slot_id: 0,
            last_page_id: tail.page_id as u32,
            last_page_slots: tail.num_tuples,
            finished: false,
        }
    }

    /// Appends a tuple stamped with timestamp 0, i.e. visible to every transaction. Used for
    /// non-transactional heaps such as spill files.
    pub fn insert(&self, t: &Tuple) -> Result<RecordId, TableHeapError> {
        self.insert_with_meta(TupleMeta::default(), t)
    }

    /// Appends a tuple with the given MVCC metadata.
    pub fn insert_with_meta(&self, meta: TupleMeta, t: &Tuple) -> Result<RecordId, TableHeapError> {
        let mut tail = self.tail.lock();
        self.insert_locked(meta, t, &mut tail)
    }

    pub fn last_page_id(&self) -> usize {
        self.tail.lock().page_id
    }

    fn insert_locked(
        &self,
        meta: TupleMeta,
        t: &Tuple,
        tail: &mut MutexGuard<'_, Tail>,
    ) -> Result<RecordId, TableHeapError> {
        if t.data.len() > TablePageRef::MAX_TUPLE_SIZE {
            return Err(TableHeapError(format!(
                "tuple of size {} exceeds maximum {}",
                t.data.len(),
                TablePageRef::MAX_TUPLE_SIZE
            )));
        }
        let slot_id = {
            let mut guard = self.bpm.write_page(tail.page_id)?;
            let mut page = TablePageMut::new(guard.data_mut())?;
            if page.can_insert(t.data.len()) {
                Some(page.insert_with_meta(meta, &t.data)?)
            } else {
                None
            }
        };
        let (page_id, slot_id) = match slot_id {
            Some(slot_id) => (tail.page_id as u32, slot_id),
            None => {
                let (new_id, slot_id) = {
                    let mut guard = self.bpm.new_page()?;
                    let page_id = record_page_id(guard.page_id())?;
                    let slot_id =
                        TablePage::init(guard.data_mut()).insert_with_meta(meta, &t.data)?;
                    (page_id, slot_id)
                };
                {
                    let mut guard = self.bpm.write_page(tail.page_id)?;
                    TablePageMut::new(guard.data_mut())?.set_next_page_id(new_id);
                }
                (new_id, slot_id)
            }
        };
        **tail = Tail {
            page_id: page_id as usize,
            num_tuples: slot_id + 1,
        };
        Ok(RecordId { page_id, slot_id })
    }

    /// Marks the tuple as deleted, keeping its timestamp. Its bytes stay on the page and the
    /// space is not reused. Transactional deletes go through [`TableHeap::write_with`].
    pub fn delete(&self, rid: &RecordId) -> Result<(), TableHeapError> {
        self.write_with(*rid, |meta, _| {
            if meta.is_deleted {
                return Err(TableHeapError(format!("record {rid} already deleted")));
            }
            Ok(SlotWrite {
                meta: TupleMeta {
                    is_deleted: true,
                    ..meta
                },
                data: None,
            })
        })
    }

    /// Returns a tuple whose newest version is not a deletion, ignoring timestamps.
    pub fn get(&self, rid: RecordId) -> Result<(RecordId, Tuple), TableHeapError> {
        let (meta, tuple) = self.get_with_meta(rid)?;
        if meta.is_deleted {
            return Err(TableHeapError(format!("record {rid} has been deleted")));
        }
        Ok((rid, tuple))
    }

    /// Returns the newest version of a tuple and its metadata, deleted or not.
    pub fn get_with_meta(&self, rid: RecordId) -> Result<(TupleMeta, Tuple), TableHeapError> {
        self.read_with(rid, |meta, bytes| {
            (
                meta,
                Tuple {
                    data: bytes.to_vec(),
                },
            )
        })
    }

    /// Calls `f` with the slot's metadata and bytes under the page's read latch.
    pub fn read_with<R>(
        &self,
        rid: RecordId,
        f: impl FnOnce(TupleMeta, &[u8]) -> R,
    ) -> Result<R, TableHeapError> {
        Self::read_slot(&self.bpm, rid, f)
    }

    /// Calls `f` with the slot's metadata and bytes under the page's write latch and applies
    /// the [`SlotWrite`] it returns before releasing the latch, so a check and the write it
    /// guards are atomic with respect to every other reader and writer of the slot.
    pub fn write_with<E>(
        &self,
        rid: RecordId,
        f: impl FnOnce(TupleMeta, &[u8]) -> Result<SlotWrite, E>,
    ) -> Result<(), E>
    where
        E: From<TableHeapError>,
    {
        Self::write_slot(&self.bpm, rid, f)
    }

    /// [`TableHeap::read_with`] for callers that only hold the buffer pool, such as the
    /// transaction manager. Record ids name pages directly, so no heap handle is needed.
    pub fn read_slot<R>(
        bpm: &Arc<BufferPoolManager>,
        rid: RecordId,
        f: impl FnOnce(TupleMeta, &[u8]) -> R,
    ) -> Result<R, TableHeapError> {
        let page_guard = bpm.read_page(rid.page_id as usize).map_err(|e| {
            TableHeapError(format!(
                "failed to acquire read lock on page {} for read of {rid}: {e}",
                rid.page_id
            ))
        })?;
        let table_page = TablePageRef::new(page_guard.data())?;
        let slot = table_page.slot(rid.slot_id).ok_or_else(|| {
            TableHeapError(format!(
                "slot {} does not exist on page {} (page has {} slots)",
                rid.slot_id,
                rid.page_id,
                table_page.num_tuples()
            ))
        })?;
        let bytes = table_page.tuple_bytes(&slot)?;
        Ok(f(slot.meta, bytes))
    }

    /// [`TableHeap::write_with`] for callers that only hold the buffer pool.
    pub fn write_slot<E>(
        bpm: &Arc<BufferPoolManager>,
        rid: RecordId,
        f: impl FnOnce(TupleMeta, &[u8]) -> Result<SlotWrite, E>,
    ) -> Result<(), E>
    where
        E: From<TableHeapError>,
    {
        let mut page_guard = bpm.write_page(rid.page_id as usize).map_err(|e| {
            TableHeapError(format!(
                "failed to acquire write lock on page {} for write of {rid}: {e}",
                rid.page_id
            ))
        })?;
        let mut table_page =
            TablePageMut::new(page_guard.data_mut()).map_err(TableHeapError::from)?;
        let slot = table_page.slot(rid.slot_id).ok_or_else(|| {
            TableHeapError(format!(
                "slot {} does not exist on page {} (page has {} slots)",
                rid.slot_id,
                rid.page_id,
                table_page.num_tuples()
            ))
        })?;
        let write = {
            let bytes = table_page
                .tuple_bytes(&slot)
                .map_err(TableHeapError::from)?;
            f(slot.meta, bytes)?
        };
        match write.data {
            Some(data) => table_page
                .update_in_place(rid.slot_id, write.meta, &data)
                .map_err(TableHeapError::from)?,
            None => table_page
                .update_meta(rid.slot_id, write.meta)
                .map_err(TableHeapError::from)?,
        }
        Ok(())
    }

    /// Overwrites a slot's metadata.
    pub fn update_meta(&self, rid: RecordId, meta: TupleMeta) -> Result<(), TableHeapError> {
        self.write_with(rid, |_, _| Ok(SlotWrite { meta, data: None }))
    }

    /// Overwrites a tuple and its metadata in place, keeping its record id. The new tuple
    /// is the same size as the old one, or the write fails.
    pub fn update_in_place(
        &self,
        rid: RecordId,
        meta: TupleMeta,
        new: &Tuple,
    ) -> Result<(), TableHeapError> {
        self.write_with(rid, |_, _| {
            Ok(SlotWrite {
                meta,
                data: Some(new.data.clone()),
            })
        })
    }
}

/// Converts a buffer pool page id into the `u32` stored in record ids and page links.
fn record_page_id(page_id: Option<usize>) -> Result<u32, TableHeapError> {
    let page_id = page_id.ok_or_else(|| TableHeapError("new page has no page_id".to_string()))?;
    u32::try_from(page_id)
        .ok()
        .filter(|&id| id != NO_NEXT_PAGE)
        .ok_or_else(|| TableHeapError(format!("page id {page_id} exceeds RID capacity")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::replacer::ArcReplacer;
    use tempfile::tempdir;

    fn make_heap() -> (TableHeap, tempfile::TempDir) {
        let dir = tempdir().expect("tempdir");
        let dm = DiskManager::new(&dir.path().join("db"), &dir.path().join("log"))
            .expect("disk manager");
        let scheduler = DiskScheduler::new(dm);
        let replacer = ArcReplacer::new(8);
        let bpm = Arc::new(create_buffer_pool_manager(8, replacer, scheduler));
        let heap = TableHeap::new(bpm).expect("create heap");
        (heap, dir)
    }

    #[test]
    fn insert_returns_rid_at_first_slot() {
        let (heap, _dir) = make_heap();
        let rid = heap
            .insert(&Tuple {
                data: b"hello world".to_vec(),
            })
            .expect("insert");
        assert_eq!(rid.slot_id, 0);
    }

    #[test]
    fn get_returns_inserted_tuple_bytes() {
        let (heap, _dir) = make_heap();
        let payload = b"the quick brown fox".to_vec();
        let rid = heap
            .insert(&Tuple {
                data: payload.clone(),
            })
            .expect("insert");
        let (returned_rid, got) = heap.get(rid).expect("get");
        assert_eq!(got.data, payload);
        assert_eq!(returned_rid, rid);
    }

    #[test]
    fn delete_tombstones_and_blocks_get() {
        let (heap, _dir) = make_heap();
        let rid = heap
            .insert(&Tuple {
                data: b"to be deleted".to_vec(),
            })
            .expect("insert");
        heap.delete(&rid).expect("delete");
        assert!(
            heap.get(rid).is_err(),
            "get should fail on a tombstoned record"
        );
        assert!(
            heap.delete(&rid).is_err(),
            "double-delete should be rejected"
        );
    }

    #[test]
    fn update_in_place_keeps_rid_and_rejects_size_change() {
        let (heap, _dir) = make_heap();
        let rid = heap
            .insert(&Tuple {
                data: b"before".to_vec(),
            })
            .expect("insert");
        let meta = TupleMeta {
            ts: 9,
            is_deleted: false,
        };
        heap.update_in_place(
            rid,
            meta,
            &Tuple {
                data: b"after!".to_vec(),
            },
        )
        .expect("update");
        let (got_meta, got) = heap.get_with_meta(rid).expect("get");
        assert_eq!(got.data, b"after!");
        assert_eq!(got_meta, meta);
        assert!(
            heap.update_in_place(rid, meta, &Tuple { data: vec![0; 3] })
                .is_err()
        );
    }

    #[test]
    fn write_with_error_leaves_slot_untouched() {
        let (heap, _dir) = make_heap();
        let rid = heap
            .insert(&Tuple {
                data: b"keep".to_vec(),
            })
            .unwrap();
        let res: Result<(), TableHeapError> =
            heap.write_with(rid, |_, _| Err(TableHeapError("refused".to_string())));
        assert!(res.is_err());
        let (meta, t) = heap.get_with_meta(rid).unwrap();
        assert_eq!(t.data, b"keep");
        assert_eq!(meta, TupleMeta::default());
    }

    #[test]
    fn mixed_ops() {
        let (heap, _dir) = make_heap();
        let mut live: Vec<(RecordId, Vec<u8>)> = Vec::new();

        for i in 0..10 {
            let payload = format!("tuple-{i:02}").into_bytes();
            let rid = heap
                .insert(&Tuple {
                    data: payload.clone(),
                })
                .expect("insert");
            live.push((rid, payload));

            for (rid, expected) in &live {
                let (_, got) = heap.get(*rid).expect("get live");
                assert_eq!(&got.data, expected, "iteration {i}: live tuple mismatch");
            }

            if i % 3 == 0 {
                let (old_rid, _) = live.pop().expect("at least one live tuple");
                let new_payload = format!("TUPLE-{i:02}").into_bytes();
                let t = Tuple {
                    data: new_payload.clone(),
                };
                heap.update_in_place(old_rid, TupleMeta::default(), &t)
                    .expect("update");
                live.push((old_rid, new_payload));
            }

            if i % 4 == 0 && !live.is_empty() {
                let (rid, _) = live.remove(0);
                heap.delete(&rid).expect("delete");
                assert!(heap.get(rid).is_err());
            }
        }

        for (rid, expected) in &live {
            let (_, got) = heap.get(*rid).expect("final get");
            assert_eq!(&got.data, expected);
        }
    }

    #[test]
    fn last_page_has_no_next_page_even_when_the_table_starts_at_page_zero() {
        let (heap, _dir) = make_heap();
        assert_eq!(heap.first_page_id, 0);
        let guard = heap.bpm.read_page(heap.first_page_id).unwrap();
        assert!(
            TablePageRef::new(guard.data())
                .unwrap()
                .next_page_id()
                .is_none()
        );
    }

    #[test]
    fn full_page_links_to_a_new_tail_page() {
        let (heap, _dir) = make_heap();
        let big = Tuple {
            data: vec![1; TablePageRef::MAX_TUPLE_SIZE],
        };
        let first = heap.insert(&big).unwrap();
        let second = heap.insert(&big).unwrap();
        assert_ne!(first.page_id, second.page_id);
        let guard = heap.bpm.read_page(first.page_id as usize).unwrap();
        assert_eq!(
            TablePageRef::new(guard.data()).unwrap().next_page_id(),
            Some(second.page_id)
        );
        drop(guard);
        let scanned: Vec<RecordId> = heap.iter().map(|r| r.unwrap().1).collect();
        assert_eq!(scanned, vec![first, second]);
    }
}
