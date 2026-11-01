use crate::buffer_pool::{BpmError, BufferPoolManager};
use crate::table_page::{NO_NEXT_PAGE, TablePage, TablePageError, TablePageMut, TablePageRef};
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

            let slot = self.current_slot_id;
            let Some(meta) = page.slot(slot) else {
                match page.next_page_id() {
                    Some(next) if self.current_page_id != self.last_page_id => {
                        self.current_page_id = next;
                        self.current_slot_id = 0;
                        continue;
                    }
                    _ => {
                        self.finished = true;
                        return None;
                    }
                }
            };
            self.current_slot_id += 1;
            if meta.is_deleted {
                continue;
            }
            let data = match page.tuple_bytes(&meta) {
                Ok(bytes) => bytes.to_vec(),
                Err(e) => {
                    self.finished = true;
                    return Some(Err(TableHeapError(format!(
                        "corrupt slot {slot} on page {} during scan: {}",
                        self.current_page_id, e.0
                    ))));
                }
            };

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
                Some(page.insert(&t.data)?)
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
                    let slot_id = TablePage::init(guard.data_mut()).insert(&t.data)?;
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

    /// Marks the tuple as deleted. Its bytes stay on the page and the space is not reused.
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
        let mut table_page = TablePageMut::new(page_guard.data_mut())?;
        let meta = table_page.slot(rid.slot_id).ok_or_else(|| {
            TableHeapError(format!(
                "slot {} does not exist on page {} (page has {} slots)",
                rid.slot_id,
                rid.page_id,
                table_page.num_tuples()
            ))
        })?;
        if meta.is_deleted {
            return Err(TableHeapError(format!("record {rid} already deleted")));
        }
        table_page.mark_deleted(rid.slot_id)?;
        Ok(())
    }

    pub fn get(&self, rid: RecordId) -> Result<(RecordId, Tuple), TableHeapError> {
        let page_guard = self.bpm.read_page(rid.page_id as usize).map_err(|e| {
            TableHeapError(format!(
                "failed to acquire read lock on page {} for get of {rid}: {e}",
                rid.page_id
            ))
        })?;
        let table_page = TablePageRef::new(page_guard.data())?;
        let meta = table_page.slot(rid.slot_id).ok_or_else(|| {
            TableHeapError(format!(
                "slot {} does not exist on page {} (page has {} slots)",
                rid.slot_id,
                rid.page_id,
                table_page.num_tuples()
            ))
        })?;
        if meta.is_deleted {
            return Err(TableHeapError(format!("record {rid} has been deleted")));
        }

        let bytes = table_page.tuple_bytes(&meta)?.to_vec();

        let t = Tuple { data: bytes };
        Ok((rid, t))
    }

    /// Writes `new` as a new record, then marks `rid` deleted, and returns the new record id. If the
    /// new record cannot be written, the original is left untouched.
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
                let new_payload = format!("updated-{i:02}").into_bytes();
                let t = Tuple {
                    data: new_payload.clone(),
                };
                let (new_rid, _) = heap.update(old_rid, &t).expect("update");
                assert!(heap.get(old_rid).is_err());
                live.push((new_rid, new_payload));
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
