use crate::buf::{BufReader, BufWriter, BufferError};
use crate::buffer_pool::{BpmError, BufferPoolManager};
use crate::disk::PAGE_SIZE;
use parking_lot::{Mutex, MutexGuard};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub struct TableHeapError(pub String);

#[derive(Debug)]
pub struct TablePageError(String);

impl From<TablePageError> for TableHeapError {
    fn from(err: TablePageError) -> Self {
        TableHeapError(err.0)
    }
}

impl From<BufferError> for TablePageError {
    fn from(err: BufferError) -> Self {
        TablePageError(format!("buffer codec error: {err:?}"))
    }
}

impl From<BpmError> for TableHeapError {
    fn from(err: BpmError) -> Self {
        TableHeapError(format!("buffer pool error: {err}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

pub struct TupleMeta {
    offset: u32,
    size: u32,
    is_deleted: u8,
    // timestamp - add later
}

// Table pages are:
// - backward-growing
// - always fully decoded/encoded for simplicity when read or modified. Our b+ pages are done like this.
//   If if were to redo it i would probably make them typed views over the bytes instead of having to decode/encode all the time
pub struct TablePage {
    next_page_id: u32, // when scanning a table we use this as a pointer to jump to the next page
    num_tuples: u16,
    num_deleted_tuples: u16,
    free_space_offset: u32, // tuples are inserted back-to-front in the page, this tracks where to insert
    tuples_meta: Vec<TupleMeta>,
    // ... tuple data ...
}

impl TablePage {
    const HEADER_SIZE: usize = 12;
    const SLOT_SIZE: usize = 9;
    const MAX_TUPLE_SIZE: usize = PAGE_SIZE - Self::HEADER_SIZE - Self::SLOT_SIZE;

    fn empty() -> Self {
        Self {
            next_page_id: 0,
            num_tuples: 0,
            num_deleted_tuples: 0,
            free_space_offset: PAGE_SIZE as u32,
            tuples_meta: Vec::new(),
        }
    }

    fn can_insert(&self, tuple_len: usize) -> bool {
        let directory_end = Self::HEADER_SIZE + (self.tuples_meta.len() + 1) * Self::SLOT_SIZE;
        directory_end + tuple_len <= self.free_space_offset as usize
    }

    fn insert(&mut self, tuple: &Tuple, data: &mut [u8]) -> Result<u32, TablePageError> {
        if !self.can_insert(tuple.data.len()) {
            return Err(TablePageError("tuple does not fit on page".to_string()));
        }
        let offset = self.next_free_offset();
        let size = tuple.data.len() as u32;
        let slot = self.num_tuples as u32;
        self.insert_tuple_metadata(TupleMeta {
            offset,
            size,
            is_deleted: 0,
        });
        self.num_tuples += 1;
        let mut staged = [0; PAGE_SIZE];
        staged.copy_from_slice(data);
        self.encode(&mut staged)?;
        staged[(offset - size) as usize..offset as usize].copy_from_slice(&tuple.data);
        data.copy_from_slice(&staged);
        Ok(slot)
    }

    pub fn decode(data: &[u8]) -> Result<Self, TablePageError> {
        let mut reader = BufReader::new(data)?;
        let next_page_id = reader.read_u32_le()?;
        let num_tuples = reader.read_u16_le()?;
        let num_deleted_tuples = reader.read_u16_le()?;
        let free_space_offset = reader.read_u32_le()?;
        let mut tuples_meta: Vec<TupleMeta> = Vec::with_capacity(num_tuples as usize);
        for i in 0..num_tuples {
            tuples_meta.push(TupleMeta {
                offset: reader.read_u32_le().map_err(|e| {
                    TablePageError(format!("failed to read offset for slot {i}: {e:?}"))
                })?,
                size: reader.read_u32_le().map_err(|e| {
                    TablePageError(format!("failed to read size for slot {i}: {e:?}"))
                })?,
                is_deleted: reader.read_u8().map_err(|e| {
                    TablePageError(format!("failed to read tombstone for slot {i}: {e:?}"))
                })?,
            });
        }
        Ok(Self {
            next_page_id,
            num_tuples,
            num_deleted_tuples,
            free_space_offset,
            tuples_meta,
        })
    }

    /// Writes the header and slot directory back into `data`. Tuple bytes (which
    /// live in `data[free_space_offset..PAGE_SIZE]`) are written directly by
    /// callers and are not touched here.
    pub fn encode(&self, data: &mut [u8]) -> Result<(), TablePageError> {
        let mut writer = BufWriter::new(data)?;
        writer.write_u32_le(self.next_page_id)?;
        writer.write_u16_le(self.num_tuples)?;
        writer.write_u16_le(self.num_deleted_tuples)?;
        writer.write_u32_le(self.free_space_offset)?;
        for meta in &self.tuples_meta {
            writer.write_u32_le(meta.offset)?;
            writer.write_u32_le(meta.size)?;
            writer.write_u8(meta.is_deleted)?;
        }
        if writer.pos() > self.free_space_offset as usize {
            return Err(TablePageError(format!(
                "slot directory ends at {} but tuple area starts at {}",
                writer.pos(),
                self.free_space_offset
            )));
        }
        Ok(())
    }

    fn insert_tuple_metadata(&mut self, meta: TupleMeta) {
        self.free_space_offset -= meta.size;
        self.tuples_meta.push(meta);
    }

    fn next_free_offset(&self) -> u32 {
        self.free_space_offset
    }
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub data: Vec<u8>,
}

pub struct TableHeap {
    bpm: Arc<BufferPoolManager>,
    pub first_page_id: usize, // the page where the table starts
    tail: Mutex<Tail>,
}

struct Tail {
    page_id: usize,
    num_tuples: u32,
}

/// TableHeapIterator streams back tuples of (Tuple, RecordId) from the table heap.
/// separating this from the table heap avoid lifetime issues. Not sure if it's the right call.
pub struct TableHeapIterator {
    bpm: Arc<BufferPoolManager>,
    current_page_id: u32,
    current_slot_id: u32,
    last_page_id: u32,
    last_page_slots: u32,
    finished: bool,
}

impl Iterator for TableHeapIterator {
    type Item = Result<(Tuple, RecordId), TableHeapError>;

    fn next(&mut self) -> Option<Self::Item> {
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

            let page = match TablePage::decode(page_guard.data()) {
                Ok(p) => p,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(TableHeapError(format!(
                        "failed to decode page {} during scan: {}",
                        self.current_page_id, e.0
                    ))));
                }
            };
            if self.current_slot_id as usize >= page.tuples_meta.len() {
                if self.current_page_id == self.last_page_id || page.next_page_id == 0 {
                    self.finished = true;
                    return None;
                }
                self.current_page_id = page.next_page_id;
                self.current_slot_id = 0;
                continue;
            }

            let slot = self.current_slot_id;
            let meta = &page.tuples_meta[slot as usize];
            self.current_slot_id += 1;
            if meta.is_deleted == 1 {
                continue;
            }
            let start = (meta.offset - meta.size) as usize;
            let end = meta.offset as usize;
            let data = page_guard.data()[start..end].to_vec();

            return Some(Ok((
                Tuple { data },
                RecordId {
                    page_id: self.current_page_id,
                    slot_id: slot,
                },
            )));
        }
    }
}

impl TableHeap {
    pub fn new(bpm: Arc<BufferPoolManager>) -> Result<Self, TableHeapError> {
        let page_id = {
            let mut page_guard = bpm
                .new_page()
                .map_err(|e| TableHeapError(format!("failed to allocate first page: {e}")))?;
            let page_id = page_guard
                .page_id()
                .ok_or_else(|| TableHeapError("new page has no page_id".to_string()))?;
            u32::try_from(page_id)
                .map_err(|_| TableHeapError("page ID exceeds RID capacity".to_string()))?;
            TablePage::empty().encode(page_guard.data_mut())?;
            page_id
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

    /// Scan only slots present at creation. Later deletions remain visible;
    /// this bounds appends without providing transaction snapshot isolation.
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

    pub fn insert(&self, t: &Tuple) -> Result<RecordId, TableHeapError> {
        let mut tail = self.tail.lock();
        self.insert_locked(t, &mut tail)
    }

    pub fn last_page_id(&self) -> usize {
        self.tail.lock().page_id
    }

    fn insert_locked(
        &self,
        t: &Tuple,
        tail: &mut MutexGuard<'_, Tail>,
    ) -> Result<RecordId, TableHeapError> {
        if t.data.len() > TablePage::MAX_TUPLE_SIZE {
            return Err(TableHeapError(format!(
                "tuple of size {} exceeds maximum {}",
                t.data.len(),
                TablePage::MAX_TUPLE_SIZE
            )));
        }
        let slot_id = {
            let mut guard = self.bpm.write_page(tail.page_id)?;
            let mut page = TablePage::decode(guard.data_mut())?;
            if page.can_insert(t.data.len()) {
                Some(page.insert(t, guard.data_mut())?)
            } else {
                None
            }
        };
        let (page_id, slot_id) = match slot_id {
            Some(slot_id) => (tail.page_id as u32, slot_id),
            None => {
                let (new_id, slot_id) = {
                    let mut guard = self.bpm.new_page()?;
                    let page_id = guard
                        .page_id()
                        .ok_or_else(|| TableHeapError("new page has no page_id".to_string()))?;
                    let page_id = u32::try_from(page_id)
                        .map_err(|_| TableHeapError("page ID exceeds RID capacity".to_string()))?;
                    let slot_id = TablePage::empty().insert(t, guard.data_mut())?;
                    (page_id, slot_id)
                };
                {
                    let mut guard = self.bpm.write_page(tail.page_id)?;
                    guard.data_mut()[..4].copy_from_slice(&new_id.to_le_bytes());
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

    /// Logically deletes a tuple from the table heap by marking the tuple as tombstoned.
    /// Note that this does not physically reclaim space on disk.
    pub fn delete(&self, rid: &RecordId) -> Result<(), TableHeapError> {
        let tail = self.tail.lock();
        self.delete_locked(rid, &tail)
    }

    fn delete_locked(
        &self,
        rid: &RecordId,
        _tail: &MutexGuard<'_, Tail>,
    ) -> Result<(), TableHeapError> {
        let mut page_guard = self.bpm.write_page(rid.page_id as usize).map_err(|e| {
            TableHeapError(format!(
                "failed to acquire write lock on page {} for delete of {rid}: {e}",
                rid.page_id
            ))
        })?;
        let data = page_guard.data_mut();
        let mut table_page = TablePage::decode(&data)?;
        let num_slots = table_page.tuples_meta.len();

        let meta = table_page
            .tuples_meta
            .get_mut(rid.slot_id as usize)
            .ok_or_else(|| {
                TableHeapError(format!(
                    "slot {} does not exist on page {} (page has {num_slots} slots)",
                    rid.slot_id, rid.page_id
                ))
            })?;

        if meta.is_deleted == 1 {
            return Err(TableHeapError(format!("record {rid} already deleted")));
        }
        meta.is_deleted = 1;
        table_page.num_deleted_tuples += 1;

        // we don't actually delete the tuple data
        table_page.encode(data)?;
        Ok(())
    }

    pub fn get(&self, rid: RecordId) -> Result<(RecordId, Tuple), TableHeapError> {
        let page_guard = self.bpm.read_page(rid.page_id as usize).map_err(|e| {
            TableHeapError(format!(
                "failed to acquire read lock on page {} for get of {rid}: {e}",
                rid.page_id
            ))
        })?;
        let data = page_guard.data();
        let table_page = TablePage::decode(data)?;
        let num_slots = table_page.tuples_meta.len();
        let meta = table_page
            .tuples_meta
            .get(rid.slot_id as usize)
            .ok_or_else(|| {
                TableHeapError(format!(
                    "slot {} does not exist on page {} (page has {num_slots} slots)",
                    rid.slot_id, rid.page_id
                ))
            })?;
        if meta.is_deleted == 1 {
            return Err(TableHeapError(format!("record {rid} has been deleted")));
        }

        let start = (meta.offset - meta.size) as usize;
        let end = meta.offset as usize;
        let bytes = page_guard.data()[start..end].to_vec();

        let t = Tuple { data: bytes };
        Ok((rid, t))
    }

    /// Append the replacement before tombstoning the original, so insertion
    /// failures (including oversized tuples and allocation failures) preserve it.
    pub fn update<'a>(
        &self,
        rid: RecordId,
        new: &'a Tuple,
    ) -> Result<(RecordId, &'a Tuple), TableHeapError> {
        let mut tail = self.tail.lock();
        self.get(rid)?;
        let new_rid = self.insert_locked(new, &mut tail)?;
        if let Err(err) = self.delete_locked(&rid, &tail) {
            if let Err(cleanup) = self.delete_locked(&new_rid, &tail) {
                return Err(TableHeapError(format!(
                    "update could not delete original: {}; replacement {new_rid} cleanup failed: {}",
                    err.0, cleanup.0
                )));
            }
            return Err(err);
        }
        Ok((new_rid, new))
    }
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
    fn update_replaces_tuple_and_invalidates_old_rid() {
        let (heap, _dir) = make_heap();
        let old_rid = heap
            .insert(&Tuple {
                data: b"before".to_vec(),
            })
            .expect("insert");
        let t = Tuple {
            data: b"after!".to_vec(),
        };
        let (new_rid, returned) = heap.update(old_rid, &t).expect("update");
        assert_eq!(returned.data, b"after!");
        assert_ne!(new_rid, old_rid, "update should produce a new RID");
        assert!(
            heap.get(old_rid).is_err(),
            "old RID should be tombstoned after update"
        );
        let (_, got) = heap.get(new_rid).expect("get new");
        assert_eq!(got.data, b"after!");
    }

    #[test]
    fn mixed_ops() {
        let (heap, _dir) = make_heap();
        // keep track of records we don't delete
        let mut live: Vec<(RecordId, Vec<u8>)> = Vec::new();

        for i in 0..10 {
            let payload = format!("tuple-{i:02}").into_bytes();
            let rid = heap
                .insert(&Tuple {
                    data: payload.clone(),
                })
                .expect("insert");
            live.push((rid, payload));

            // Every live tuple should still read back correctly after each insert.
            for (rid, expected) in &live {
                let (_, got) = heap.get(*rid).expect("get live");
                assert_eq!(&got.data, expected, "iteration {i}: live tuple mismatch");
            }

            // update some tuples
            if i % 3 == 0 {
                let (old_rid, _) = live.pop().expect("at least one live tuple");
                let new_payload = format!("updated-{i:02}").into_bytes();
                let t = Tuple {
                    data: new_payload.clone(),
                };
                let (new_rid, _) = heap.update(old_rid, &t).expect("update");
                assert!(heap.get(old_rid).is_err());
                live.push((new_rid, new_payload));
            }

            // delete some tuples
            if i % 4 == 0 && !live.is_empty() {
                let (rid, _) = live.remove(0);
                heap.delete(&rid).expect("delete");
                assert!(heap.get(rid).is_err());
            }
        }

        // re-read non-deleted tuples
        for (rid, expected) in &live {
            let (_, got) = heap.get(*rid).expect("final get");
            assert_eq!(&got.data, expected);
        }
    }
}
