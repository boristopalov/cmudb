use crate::buf::{self, BufReader, BufWriter};
use crate::disk::PAGE_SIZE;
use crate::{buffer_pool::BufferPoolManager, catalog::Tuple};
use std::sync::Arc;

pub type PageId = usize;

pub enum TableHeapError {}

pub struct TupleMeta {
    offset: u32,
    size: u32,
}

pub struct TablePage {
    header: TablePageHeader,
    tuples_meta: Vec<TupleMeta>,
}

impl TablePage {
    pub fn decode(data: &[u8]) -> Self {
        let mut reader = BufReader::new(data).unwrap();
        let next_page_id = reader.read_u32_le().unwrap();
        let num_tuples = reader.read_u16_le().unwrap();
        let num_deleted_tuples = reader.read_u16_le().unwrap();
        let free_space_offset = reader.read_u32_le().unwrap();
        let table_page_header = TablePageHeader {
            next_page_id,
            num_tuples,
            num_deleted_tuples,
            free_space_offset,
        };

        let mut tuples_meta: Vec<TupleMeta> = Vec::with_capacity(num_tuples as usize);
        for _ in 0..num_tuples {
            let t = TupleMeta {
                offset: reader.read_u32_le().unwrap(),
                size: reader.read_u32_le().unwrap(),
            };
            tuples_meta.push(t);
        }
        Self {
            header: table_page_header,
            tuples_meta,
        }
    }

    pub fn encode(&self) -> Result<(), TableHeapError> {
        let buf = &mut [0u8; PAGE_SIZE];
        let mut writer = BufWriter::new(buf);
        // etc...
        //

        Ok(())
    }

    pub fn insert_tuple_metadata(&mut self, meta: TupleMeta) {
        self.header.free_space_offset += meta.size;
        self.tuples_meta.push(meta);
    }

    pub fn next_free_offset(&self) -> u32 {
        self.header.free_space_offset
    }
}

pub struct TablePageHeader {
    next_page_id: u32,
    num_tuples: u16,
    num_deleted_tuples: u16,
    free_space_offset: u32,
}

pub struct TableHeap {
    bpm: Arc<BufferPoolManager>,
}

impl TableHeap {
    pub fn new(bpm: Arc<BufferPoolManager>) -> Self {
        Self { bpm }
    }

    pub fn insert_tuple(&self, t: Tuple) -> Result<(), TableHeapError> {
        let mut page_guard = self.bpm.write_page(t.record_id.page_id as usize).unwrap();
        let data = page_guard.data_mut();
        let mut table_page = TablePage::decode(&data);
        let tuple_len = t.data.len() + 4 + 4; // record id is 8 bytes
        let offset = table_page.next_free_offset();
        let meta = TupleMeta {
            offset,
            size: tuple_len as u32,
        };
        table_page.insert_tuple_metadata(meta);

        Ok(())
    }
}
