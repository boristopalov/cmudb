use crate::buf::{BufReader, BufWriter};
use crate::buffer_pool::BufferPoolManager;
use crate::disk::PAGE_SIZE;
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub struct TableHeapError(String);

#[derive(Debug)]
pub struct TablePageError(String);

impl From<TablePageError> for TableHeapError {
    fn from(err: TablePageError) -> Self {
        TableHeapError(err.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordId {
    pub page_id: u32,
    pub slot_id: u32,
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
    pub fn decode(data: &[u8]) -> Result<Self, TablePageError> {
        let mut reader = BufReader::new(data).unwrap();
        let next_page_id = reader.read_u32_le().unwrap();
        let num_tuples = reader.read_u16_le().unwrap();
        let num_deleted_tuples = reader.read_u16_le().unwrap();
        let free_space_offset = reader.read_u32_le().unwrap();
        let mut tuples_meta: Vec<TupleMeta> = Vec::with_capacity(num_tuples as usize);
        for _ in 0..num_tuples {
            let t = TupleMeta {
                offset: reader.read_u32_le().unwrap(),
                size: reader.read_u32_le().unwrap(),
                is_deleted: reader.read_u8().unwrap(),
            };
            tuples_meta.push(t);
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
        let mut writer = BufWriter::new(data).unwrap();
        writer.write_u32_le(self.next_page_id).unwrap();
        writer.write_u16_le(self.num_tuples).unwrap();
        writer.write_u16_le(self.num_deleted_tuples).unwrap();
        writer.write_u32_le(self.free_space_offset).unwrap();
        for meta in &self.tuples_meta {
            writer.write_u32_le(meta.offset).unwrap();
            writer.write_u32_le(meta.size).unwrap();
            writer.write_u8(meta.is_deleted).unwrap();
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

#[derive(Debug)]
pub struct Tuple {
    pub data: Vec<u8>,
}

pub struct TableHeap {
    bpm: Arc<BufferPoolManager>,
    first_page_id: usize, // the page where the table starts
    last_page_id: usize,  // the page where we insert new tuples into
}

pub struct TableHeapIterator<'a> {
    table: &'a TableHeap,
    current_page_id: u32,
    current_slot_id: u32,
}

impl<'a> Iterator for TableHeapIterator<'a> {
    type Item = (RecordId, Tuple);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.table.first_page_id >= self.table.last_page_id {
                return None;
            }
            let page_guard = self
                .table
                .bpm
                .read_page(self.current_page_id as usize)
                .unwrap();

            let page = TablePage::decode(page_guard.data()).ok()?;
            if self.current_slot_id as usize >= page.tuples_meta.len() {
                self.current_page_id = page.next_page_id;
                self.current_slot_id = 0;
                continue;
            }

            let slot = self.current_slot_id;
            let meta = page.tuples_meta.get(slot as usize).unwrap();
            if meta.is_deleted == 1 {
                continue;
            }
            let start = (meta.offset - meta.size) as usize;
            let end = meta.offset as usize;
            let data = page_guard.data()[start..end].to_vec();

            self.current_slot_id += 1;
            return Some((
                RecordId {
                    page_id: self.current_page_id,
                    slot_id: slot,
                },
                Tuple { data },
            ));
        }
    }
}

impl TableHeap {
    pub fn new(bpm: Arc<BufferPoolManager>) -> Result<Self, TableHeapError> {
        let mut page_guard = bpm
            .new_page()
            .map_err(|e| TableHeapError(format!("failed to allocate first page: {e}")))?;
        let page_id = page_guard
            .page_id()
            .ok_or_else(|| TableHeapError("new page has no page_id".to_string()))?;
        let initial = TablePage {
            next_page_id: 0,
            num_tuples: 0,
            num_deleted_tuples: 0,
            free_space_offset: PAGE_SIZE as u32,
            tuples_meta: Vec::new(),
        };
        initial.encode(page_guard.data_mut())?;
        drop(page_guard);
        Ok(Self {
            bpm,
            first_page_id: page_id,
            last_page_id: page_id,
        })
    }

    pub fn insert(&self, t: &Tuple) -> Result<RecordId, TableHeapError> {
        let mut page_guard = self.bpm.write_page(self.last_page_id).unwrap();
        let data = page_guard.data_mut();
        let mut table_page = TablePage::decode(&data)?;
        let tuple_len = t.data.len();
        let offset = table_page.next_free_offset();
        let meta = TupleMeta {
            offset,
            size: tuple_len as u32,
            is_deleted: 0,
        };

        let start = (offset - meta.size) as usize;
        let end = offset as usize;

        table_page.insert_tuple_metadata(meta);
        table_page.num_tuples += 1;

        // write the tuple into the page
        data[start..end].copy_from_slice(&t.data);

        let slot_id = (table_page.tuples_meta.len() - 1) as u32;
        table_page.encode(data)?;

        Ok(RecordId {
            page_id: self.last_page_id as u32,
            slot_id,
        })
    }

    /// Logically deletes a tuple from the table heap by marking the tuple as tombstoned.
    /// Note that this does not physically reclaim space on disk.
    pub fn delete(&self, rid: &RecordId) -> Result<(), TableHeapError> {
        let mut page_guard = self.bpm.write_page(rid.page_id as usize).unwrap();
        let data = page_guard.data_mut();
        let mut table_page = TablePage::decode(&data)?;

        // mark the tuple as tombstoned in the metadata
        let meta = table_page
            .tuples_meta
            .get_mut(rid.slot_id as usize)
            .unwrap();

        if meta.is_deleted == 1 {
            return Err(TableHeapError(("record already deleted").to_string()));
        }
        meta.is_deleted = 1;
        table_page.num_deleted_tuples += 1;

        // we don't actually delete the tuple data
        table_page.encode(data)?;
        Ok(())
    }

    pub fn get(&self, rid: RecordId) -> Result<(RecordId, Tuple), TableHeapError> {
        let page_guard = self.bpm.read_page(rid.page_id as usize).unwrap();
        let data = page_guard.data();
        let table_page = TablePage::decode(data)?;
        let meta = table_page.tuples_meta.get(rid.slot_id as usize).unwrap();
        if meta.is_deleted == 1 {
            return Err(TableHeapError(("record has been deleted").to_string()));
        }

        let start = (meta.offset - meta.size) as usize;
        let end = meta.offset as usize;
        let bytes = page_guard.data()[start..end].to_vec();

        let t = Tuple { data: bytes };
        Ok((rid, t))
    }

    // pub fn update(&self, rid: RecordId, new: Tuple) -> Result<(RecordId, Tuple), TableHeapError> {
    //     let mut page_guard = self.bpm.write_page(rid.page_id as usize).unwrap();
    //     let data = page_guard.data_mut();
    //     let mut table_page = TablePage::decode(&data)?;
    //     let meta = table_page
    //         .tuples_meta
    //         .get_mut(rid.slot_id as usize)
    //         .unwrap();
    //     if meta.is_deleted == 1 {
    //         return Err(TableHeapError(("record has been deleted").to_string()));
    //     }

    //     // update the tuple metadata
    //     // the offset remains the same, just make sure the size is updated
    //     meta.size = new.data.len() as u32;
    //     let start = (meta.offset - meta.size) as usize;
    //     let end = meta.offset as usize;

    //     // update the tuple bytes
    //     data[start..end].copy_from_slice(&new.data);

    //     let t = Tuple {
    //         data: new.data,
    //         schema: None,
    //     };

    //     // TODO: call table_page.encode()
    //     Ok((rid, t))
    // }

    /// Poor man's update: deletes the old record, then inserts the new tuple
    pub fn update(&self, rid: RecordId, new: Tuple) -> Result<(RecordId, Tuple), TableHeapError> {
        self.delete(&rid)?;
        let new_rid = self.insert(&new)?;
        let t = Tuple { data: new.data };
        Ok((new_rid, t))
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
        let (new_rid, returned) = heap
            .update(
                old_rid,
                Tuple {
                    data: b"after!".to_vec(),
                },
            )
            .expect("update");
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
                let (new_rid, _) = heap
                    .update(
                        old_rid,
                        Tuple {
                            data: new_payload.clone(),
                        },
                    )
                    .expect("update");
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
