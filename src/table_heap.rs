use crate::buffer_pool::{BpmError, BufferPoolManager};
use crate::ids::PageId;
pub use crate::ids::RecordId;
use crate::table_page::{TablePage, TablePageError, TablePageMut, TablePageRef, TupleMeta};
use parking_lot::{Mutex, MutexGuard};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum TableHeapError {
    /// The record's slot is past the end of its page's slot directory.
    SlotMissing { rid: RecordId, num_slots: u16 },
    /// The newest version of the record is a deletion.
    Deleted { rid: RecordId },
    /// The tuple is larger than a table page can hold.
    TooLarge { size: usize, max: usize },
    /// The buffer pool could not hand out a page.
    Bpm(BpmError),
    /// A table page rejected a read or write.
    Page {
        page_id: PageId,
        source: TablePageError,
    },
    /// Following the heap's page links from its first page came back to `page_id`.
    PageChainCycle { page_id: PageId },
    /// Page `from` links to page `to`, which the buffer pool has not handed out.
    UnallocatedPageLink { from: PageId, to: PageId },
}

impl fmt::Display for TableHeapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableHeapError::SlotMissing { rid, num_slots } => write!(
                f,
                "slot {} does not exist on page {} (page has {num_slots} slots)",
                rid.slot_id, rid.page_id
            ),
            TableHeapError::Deleted { rid } => write!(f, "record {rid} has been deleted"),
            TableHeapError::TooLarge { size, max } => {
                write!(
                    f,
                    "a row of {size} bytes does not fit in a page, which holds at most {max}"
                )
            }
            TableHeapError::Bpm(e) => write!(f, "buffer pool error: {e}"),
            TableHeapError::Page { page_id, source } => write!(f, "page {page_id}: {source}"),
            TableHeapError::PageChainCycle { page_id } => {
                write!(f, "the table's page links loop back to page {page_id}")
            }
            TableHeapError::UnallocatedPageLink { from, to } => write!(
                f,
                "page {from} of the table links to page {to}, which is not allocated"
            ),
        }
    }
}

impl From<BpmError> for TableHeapError {
    fn from(err: BpmError) -> Self {
        TableHeapError::Bpm(err)
    }
}

impl TablePageError {
    fn on_page(self, page_id: PageId) -> TableHeapError {
        TableHeapError::Page {
            page_id,
            source: self,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub data: Vec<u8>,
}

/// Why [`TableHeap::free`] stopped, with the part of the heap it had not deleted yet.
pub(crate) struct FreeError {
    pub(crate) error: TableHeapError,
    pub(crate) rest: TableHeap,
}

impl fmt::Debug for FreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreeError")
            .field("error", &self.error)
            .field("rest_first_page_id", &self.rest.first_page_id)
            .finish()
    }
}

pub struct TableHeap {
    bpm: Arc<BufferPoolManager>,
    /// The page where the table starts.
    pub first_page_id: PageId,
    tail: Mutex<Tail>,
    /// Free bytes an insert leaves on a page that already holds rows, so that rows there can
    /// grow (see [`TablePage::update_in_place`]).
    update_reserve: usize,
}

struct Tail {
    page_id: PageId,
    num_tuples: u32,
}

/// Streams the live `(Tuple, RecordId)` pairs that were in the heap when the iterator was created.
/// Rows inserted afterwards (for example the new versions written by an UPDATE that is scanning the
/// same heap) are not yielded. It holds its own buffer pool handle, so it does not borrow the heap.
pub struct TableHeapIterator {
    bpm: Arc<BufferPoolManager>,
    current_page_id: PageId,
    current_slot_id: u32,
    last_page_id: PageId,
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
            let page_guard = match self.bpm.read_page(self.current_page_id) {
                Ok(g) => g,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e.into()));
                }
            };

            let page = match TablePageRef::new(page_guard.data()) {
                Ok(p) => p,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e.on_page(self.current_page_id)));
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
                        return Some(Err(e.on_page(self.current_page_id)));
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
            let mut page_guard = bpm.new_page()?;
            let page_id = page_guard.page_id();
            TablePageMut::init(page_guard.data_mut());
            page_id
        };
        Ok(Self {
            bpm,
            first_page_id: page_id,
            tail: Mutex::new(Tail {
                page_id,
                num_tuples: 0,
            }),
            update_reserve: 0,
        })
    }

    /// The heap with inserts leaving `bytes` free on every page that already holds a row, for
    /// updates that make rows longer. A row inserted on a fresh page is stored whatever its size.
    pub fn with_update_reserve(self, bytes: usize) -> Self {
        Self {
            update_reserve: bytes,
            ..self
        }
    }

    /// The heap whose pages start at `first_page_id`, as an earlier [`TableHeap::new`] over
    /// the same data file left it. Follows the page links to find where inserts go next.
    pub fn open(
        bpm: Arc<BufferPoolManager>,
        first_page_id: PageId,
    ) -> Result<Self, TableHeapError> {
        let mut page_id = first_page_id;
        let mut seen = std::collections::HashSet::new();
        let num_tuples = loop {
            if !seen.insert(page_id) {
                return Err(TableHeapError::PageChainCycle { page_id });
            }
            let guard = bpm.read_page(page_id)?;
            let page = TablePageRef::new(guard.data()).map_err(|e| e.on_page(page_id))?;
            match page.next_page_id() {
                Some(next) if !bpm.page_is_allocated(next) => {
                    return Err(TableHeapError::UnallocatedPageLink {
                        from: page_id,
                        to: next,
                    });
                }
                Some(next) => page_id = next,
                None => break u32::from(page.num_tuples()),
            }
        };
        Ok(Self {
            bpm,
            first_page_id,
            tail: Mutex::new(Tail {
                page_id,
                num_tuples,
            }),
            update_reserve: 0,
        })
    }

    pub fn iter(&self) -> TableHeapIterator {
        let tail = self.tail.lock();
        TableHeapIterator {
            bpm: self.bpm.clone(),
            current_page_id: self.first_page_id,
            current_slot_id: 0,
            last_page_id: tail.page_id,
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

    /// Deletes every page of the heap from the buffer pool and from disk, consuming it. Meant
    /// for scratch heaps such as spilled sort runs that no other reader holds. On failure the
    /// pages not yet deleted come back in [`FreeError::rest`], so a later `free` can finish.
    pub(crate) fn free(mut self) -> Result<(), FreeError> {
        loop {
            match self.free_first_page() {
                Ok(Some(next)) => self.first_page_id = next,
                Ok(None) => return Ok(()),
                Err(error) => return Err(FreeError { error, rest: self }),
            }
        }
    }

    /// Deletes the heap's first page and returns the id of the page after it.
    fn free_first_page(&self) -> Result<Option<PageId>, TableHeapError> {
        let page_id = self.first_page_id;
        let next = {
            let guard = self.bpm.read_page(page_id)?;
            TablePageRef::new(guard.data())
                .map_err(|e| e.on_page(page_id))?
                .next_page_id()
        };
        self.bpm.delete_page(page_id)?;
        Ok(next)
    }

    pub fn last_page_id(&self) -> PageId {
        self.tail.lock().page_id
    }

    fn insert_locked(
        &self,
        meta: TupleMeta,
        t: &Tuple,
        tail: &mut MutexGuard<'_, Tail>,
    ) -> Result<RecordId, TableHeapError> {
        if t.data.len() > TablePageRef::MAX_TUPLE_SIZE {
            return Err(TableHeapError::TooLarge {
                size: t.data.len(),
                max: TablePageRef::MAX_TUPLE_SIZE,
            });
        }
        let tail_page_id = tail.page_id;
        let slot_id = {
            let mut guard = self.bpm.write_page(tail.page_id)?;
            let mut page =
                TablePageMut::new(guard.data_mut()).map_err(|e| e.on_page(tail_page_id))?;
            let reserve = if page.num_tuples() == 0 {
                0
            } else {
                self.update_reserve
            };
            if page.can_insert(t.data.len().saturating_add(reserve)) {
                Some(
                    page.insert_with_meta(meta, &t.data)
                        .map_err(|e| e.on_page(tail_page_id))?,
                )
            } else {
                None
            }
        };
        let (page_id, slot_id) = match slot_id {
            Some(slot_id) => (tail_page_id, slot_id),
            None => {
                let (new_id, slot_id) = {
                    let mut guard = self.bpm.new_page()?;
                    let page_id = guard.page_id();
                    let slot_id = TablePage::init(guard.data_mut())
                        .insert_with_meta(meta, &t.data)
                        .map_err(|e| e.on_page(page_id))?;
                    (page_id, slot_id)
                };
                {
                    let mut guard = self.bpm.write_page(tail.page_id)?;
                    TablePageMut::new(guard.data_mut())
                        .map_err(|e| e.on_page(tail_page_id))?
                        .set_next_page_id(new_id);
                }
                (new_id, slot_id)
            }
        };
        **tail = Tail {
            page_id,
            num_tuples: slot_id + 1,
        };
        Ok(RecordId { page_id, slot_id })
    }

    /// Marks the tuple as deleted, keeping its timestamp. Its bytes stay on the page and the
    /// space is not reused. Transactional deletes go through [`TableHeap::write_with`].
    pub fn delete(&self, rid: &RecordId) -> Result<(), TableHeapError> {
        self.write_with(*rid, |meta, _| {
            if meta.is_deleted {
                return Err(TableHeapError::Deleted { rid: *rid });
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
            return Err(TableHeapError::Deleted { rid });
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
        let page_guard = bpm.read_page(rid.page_id)?;
        let table_page =
            TablePageRef::new(page_guard.data()).map_err(|e| e.on_page(rid.page_id))?;
        let slot = table_page
            .slot(rid.slot_id)
            .ok_or_else(|| TableHeapError::SlotMissing {
                rid,
                num_slots: table_page.num_tuples(),
            })?;
        let bytes = table_page
            .tuple_bytes(&slot)
            .map_err(|e| e.on_page(rid.page_id))?;
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
        Self::write_slot_then(bpm, rid, |meta, bytes| Ok((f(meta, bytes)?, ())), |()| ())
    }

    /// [`TableHeap::write_slot`] with a second step. `f` plans the write and hands a value
    /// to `then`, which runs only once the slot is overwritten, still under the page write
    /// latch. Bookkeeping that has to match the slot, such as an undo chain, goes in `then`,
    /// so a write that fails leaves both untouched.
    pub fn write_slot_then<T, R, E>(
        bpm: &Arc<BufferPoolManager>,
        rid: RecordId,
        f: impl FnOnce(TupleMeta, &[u8]) -> Result<(SlotWrite, T), E>,
        then: impl FnOnce(T) -> R,
    ) -> Result<R, E>
    where
        E: From<TableHeapError>,
    {
        let mut page_guard = bpm.write_page(rid.page_id).map_err(TableHeapError::from)?;
        let mut table_page =
            TablePageMut::new(page_guard.data_mut()).map_err(|e| e.on_page(rid.page_id))?;
        let slot = table_page
            .slot(rid.slot_id)
            .ok_or_else(|| TableHeapError::SlotMissing {
                rid,
                num_slots: table_page.num_tuples(),
            })?;
        let (write, planned) = {
            let bytes = table_page
                .tuple_bytes(&slot)
                .map_err(|e| e.on_page(rid.page_id))?;
            f(slot.meta, bytes)?
        };
        match write.data {
            Some(data) => table_page.update_in_place(rid.slot_id, write.meta, &data),
            None => table_page.update_meta(rid.slot_id, write.meta),
        }
        .map_err(|e| e.on_page(rid.page_id))?;
        Ok(then(planned))
    }

    /// [`TableHeap::write_slot_then`] on this heap's buffer pool.
    pub fn write_with_then<T, R, E>(
        &self,
        rid: RecordId,
        f: impl FnOnce(TupleMeta, &[u8]) -> Result<(SlotWrite, T), E>,
        then: impl FnOnce(T) -> R,
    ) -> Result<R, E>
    where
        E: From<TableHeapError>,
    {
        Self::write_slot_then(&self.bpm, rid, f, then)
    }

    /// Overwrites a slot's metadata.
    pub fn update_meta(&self, rid: RecordId, meta: TupleMeta) -> Result<(), TableHeapError> {
        self.write_with(rid, |_, _| Ok(SlotWrite { meta, data: None }))
    }

    /// Overwrites a tuple and its metadata in place, keeping its record id. The new tuple may be
    /// longer or shorter than the old one; when its page has no room for it, the write fails with
    /// [`TablePageError::UpdateDoesNotFit`] and changes nothing.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::pid;

    fn make_heap() -> (TableHeap, tempfile::TempDir) {
        let (bpm, dir) = crate::testing::bpm(8);
        (TableHeap::new(bpm).expect("create heap"), dir)
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
            matches!(heap.get(rid), Err(TableHeapError::Deleted { rid: r }) if r == rid),
            "get should fail on a tombstoned record"
        );
        assert!(
            matches!(heap.delete(&rid), Err(TableHeapError::Deleted { rid: r }) if r == rid),
            "double-delete should be rejected"
        );
    }

    #[test]
    fn update_in_place_keeps_rid_when_the_size_changes() {
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
        for data in [vec![3; 3], vec![4; 40]] {
            heap.update_in_place(rid, meta, &Tuple { data: data.clone() })
                .expect("update to another size");
            assert_eq!(heap.get(rid).unwrap().1.data, data);
        }
        heap.insert(&Tuple { data: vec![5] }).unwrap();
        let too_long = Tuple {
            data: vec![0; TablePageRef::MAX_TUPLE_SIZE],
        };
        assert!(matches!(
            heap.update_in_place(rid, meta, &too_long),
            Err(TableHeapError::Page {
                page_id,
                source: TablePageError::UpdateDoesNotFit { len, .. },
            }) if page_id == rid.page_id && len == too_long.data.len()
        ));
    }

    #[test]
    fn an_update_reserve_keeps_room_on_each_page_for_rows_to_grow() {
        let (heap, _dir) = make_heap();
        let heap = heap.with_update_reserve(1000);
        let row = Tuple { data: vec![1; 100] };
        let rids: Vec<_> = (0..200).map(|_| heap.insert(&row).unwrap()).collect();
        let first_page: Vec<_> = rids
            .iter()
            .filter(|r| r.page_id == rids[0].page_id)
            .collect();
        assert!(rids.iter().any(|r| r.page_id != rids[0].page_id));
        let grown = Tuple {
            data: vec![2; 100 + 1000 / first_page.len()],
        };
        for rid in &first_page {
            heap.update_in_place(**rid, TupleMeta::default(), &grown)
                .expect("the reserve holds every row of the page growing a little");
        }
        let big = Tuple {
            data: vec![3; TablePageRef::MAX_TUPLE_SIZE],
        };
        let rid = heap.insert(&big).unwrap();
        assert_eq!(rid.slot_id, 0, "a fresh page takes a row of any size");
    }

    #[test]
    fn write_with_error_leaves_slot_untouched() {
        let (heap, _dir) = make_heap();
        let rid = heap
            .insert(&Tuple {
                data: b"keep".to_vec(),
            })
            .unwrap();
        #[derive(Debug)]
        enum WriteRefused {
            Refused,
            Heap,
        }
        impl From<TableHeapError> for WriteRefused {
            fn from(_: TableHeapError) -> Self {
                WriteRefused::Heap
            }
        }
        let res = heap.write_with(rid, |_, _| Err(WriteRefused::Refused));
        assert!(matches!(res, Err(WriteRefused::Refused)));
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
                assert!(matches!(heap.get(rid), Err(TableHeapError::Deleted { .. })));
            }
        }

        for (rid, expected) in &live {
            let (_, got) = heap.get(*rid).expect("final get");
            assert_eq!(&got.data, expected);
        }
    }

    #[test]
    fn insert_rejects_a_tuple_larger_than_a_page() {
        let (heap, _dir) = make_heap();
        let too_big = Tuple {
            data: vec![0; TablePageRef::MAX_TUPLE_SIZE + 1],
        };
        assert!(matches!(
            heap.insert(&too_big),
            Err(TableHeapError::TooLarge { size, max })
                if size == TablePageRef::MAX_TUPLE_SIZE + 1 && max == TablePageRef::MAX_TUPLE_SIZE
        ));
    }

    #[test]
    fn reads_and_writes_of_a_missing_slot_report_slot_missing() {
        let (heap, _dir) = make_heap();
        let rid = heap
            .insert(&Tuple {
                data: b"only".to_vec(),
            })
            .unwrap();
        let missing = RecordId {
            slot_id: rid.slot_id + 1,
            ..rid
        };
        assert!(matches!(
            heap.get(missing),
            Err(TableHeapError::SlotMissing { rid: r, num_slots: 1 }) if r == missing
        ));
        assert!(matches!(
            heap.update_meta(missing, TupleMeta::default()),
            Err(TableHeapError::SlotMissing { rid: r, num_slots: 1 }) if r == missing
        ));
    }

    #[test]
    fn a_corrupt_page_is_reported_with_its_page_id() {
        let (heap, _dir) = make_heap();
        let rid = heap
            .insert(&Tuple {
                data: b"row".to_vec(),
            })
            .unwrap();
        {
            let mut guard = heap.bpm.write_page(rid.page_id).unwrap();
            guard.data_mut().fill(u8::MAX);
        }
        let is_corrupt_header = |r: Result<_, TableHeapError>| {
            matches!(
                r,
                Err(TableHeapError::Page {
                    page_id,
                    source: TablePageError::CorruptHeader { .. },
                }) if page_id == rid.page_id
            )
        };
        assert!(is_corrupt_header(heap.get(rid).map(|_| ())));
        assert!(is_corrupt_header(heap.delete(&rid)));
        assert!(is_corrupt_header(heap.iter().next().unwrap().map(|_| ())));
    }

    #[test]
    fn errors_display_the_record_and_page_they_concern() {
        let rid = RecordId {
            page_id: pid(3),
            slot_id: 4,
        };
        assert_eq!(
            TableHeapError::Deleted { rid }.to_string(),
            "record page_id: 3, slot_id: 4 has been deleted"
        );
        assert_eq!(
            TableHeapError::Page {
                page_id: pid(3),
                source: TablePageError::AlreadyDeleted { slot: 4 },
            }
            .to_string(),
            "page 3: slot 4 already deleted"
        );
        assert_eq!(
            TableHeapError::Bpm(BpmError::NoFreeFrame).to_string(),
            "buffer pool error: No free frame available"
        );
    }

    #[test]
    fn open_finds_the_tail_so_inserts_continue_after_the_last_row() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let heap = TableHeap::new(bpm.clone()).unwrap();
        let big = Tuple {
            data: vec![7; 3000],
        };
        let rids: Vec<RecordId> = (0..5).map(|_| heap.insert(&big).unwrap()).collect();
        assert_ne!(
            heap.first_page_id,
            heap.last_page_id(),
            "the heap spans pages"
        );

        let reopened = TableHeap::open(bpm.clone(), heap.first_page_id).unwrap();
        assert_eq!(reopened.last_page_id(), heap.last_page_id());
        let next = reopened.insert(&big).unwrap();
        assert!(
            !rids.contains(&next),
            "the new row did not overwrite an old one"
        );
        assert_eq!(reopened.iter().count(), 6);
    }

    #[test]
    fn open_rejects_a_page_chain_that_loops() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let heap = TableHeap::new(bpm.clone()).unwrap();
        let first = heap.first_page_id;
        {
            let mut guard = bpm.write_page(heap.first_page_id).unwrap();
            TablePageMut::new(guard.data_mut())
                .unwrap()
                .set_next_page_id(first);
        }
        let err = TableHeap::open(bpm, heap.first_page_id).err().unwrap();
        assert!(matches!(err, TableHeapError::PageChainCycle { page_id } if page_id == first));
        assert_eq!(
            err.to_string(),
            format!("the table's page links loop back to page {first}")
        );
    }

    #[test]
    fn open_rejects_a_link_to_a_page_that_is_not_allocated() {
        let (bpm, _dir) = crate::testing::bpm(8);
        let heap = TableHeap::new(bpm.clone()).unwrap();
        let first = heap.first_page_id;
        let past_the_end = pid(bpm.allocation().next_page_id);
        {
            let mut guard = bpm.write_page(heap.first_page_id).unwrap();
            TablePageMut::new(guard.data_mut())
                .unwrap()
                .set_next_page_id(past_the_end);
        }
        let err = TableHeap::open(bpm, heap.first_page_id).err().unwrap();
        assert!(matches!(
            err,
            TableHeapError::UnallocatedPageLink { from, to } if from == first && to == past_the_end
        ));
        assert_eq!(
            err.to_string(),
            format!(
                "page {first} of the table links to page {past_the_end}, which is not allocated"
            )
        );
    }

    #[test]
    fn last_page_has_no_next_page_even_when_the_table_starts_at_page_zero() {
        let (heap, _dir) = make_heap();
        assert_eq!(heap.first_page_id, pid(0));
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
        let guard = heap.bpm.read_page(first.page_id).unwrap();
        assert_eq!(
            TablePageRef::new(guard.data()).unwrap().next_page_id(),
            Some(second.page_id)
        );
        drop(guard);
        let scanned: Vec<RecordId> = heap.iter().map(|r| r.unwrap().1).collect();
        assert_eq!(scanned, vec![first, second]);
    }

    #[test]
    fn free_deletes_every_page_of_the_heap() {
        let (heap, _dir) = make_heap();
        let bpm = heap.bpm.clone();
        let big = Tuple {
            data: vec![1; TablePageRef::MAX_TUPLE_SIZE],
        };
        let rids: Vec<RecordId> = (0..3).map(|_| heap.insert(&big).unwrap()).collect();
        assert_eq!(bpm.live_page_count(), 3, "one page per tuple");

        heap.free().unwrap();

        assert_eq!(bpm.live_page_count(), 0);
        for rid in rids {
            assert!(
                bpm.read_page(pid(rid.page_id)).is_err(),
                "page {} survived",
                rid.page_id
            );
        }
    }

    #[test]
    fn a_failed_free_returns_the_pages_it_did_not_delete_so_a_retry_finishes() {
        let (bpm, _dir) = crate::testing::bpm(2);
        let heap = TableHeap::new(bpm.clone()).unwrap();
        let big = Tuple {
            data: vec![1; TablePageRef::MAX_TUPLE_SIZE],
        };
        for _ in 0..3 {
            heap.insert(&big).unwrap();
        }
        let first_page_id = heap.first_page_id;
        let pins = [bpm.new_page().unwrap(), bpm.new_page().unwrap()];

        let err = heap.free().unwrap_err();

        assert!(
            matches!(err.error, TableHeapError::Bpm(BpmError::NoFreeFrame)),
            "{err:?}"
        );
        assert_eq!(err.rest.first_page_id, first_page_id);
        assert_eq!(bpm.live_page_count(), 5, "no page was deleted");
        drop(pins);
        err.rest.free().unwrap();
        assert_eq!(
            bpm.live_page_count(),
            2,
            "only the two pinned pages are left"
        );
    }
}
