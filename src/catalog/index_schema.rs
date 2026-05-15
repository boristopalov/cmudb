use std::ops::Bound;
use std::sync::Arc;

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Column, DataType, IndexType, Schema, Value, is_null};
use crate::index::{
    BPlusTree, Index, IndexError, IndexIter, InsertError, InsertResult, RemoveError, RemoveResult,
};
use crate::table_heap::{RecordId, Tuple};

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
pub struct IndexKey(pub Vec<u8>);

#[derive(Debug)]
pub enum IndexValue {
    IndexValue(RecordId),
}

/// We do not allow nulls in indexes!!!
pub struct TableIndex {
    indexed_cols: Vec<Column>, // offsets are into the table tuple's payload region
    pub indexed_col_idxs: Vec<usize>, // each indexed column's position in the table schema (for null-bit lookup)
    table_bitmap_len: usize,          // bytes of bitmap prefix on tuples from the indexed table
    pub schema: Schema,               // schema of the index key, not the tuple
    key_size: usize,
    idx: Box<dyn Index>,
}

/// represents an index of a table
/// does NOT support nulls, i.e. our indexes contain only non-nulled
impl TableIndex {
    pub fn new(
        bpm: Arc<BufferPoolManager>,
        index_name: String,
        indexed_cols: Vec<Column>,
        indexed_col_idxs: Vec<usize>,
        table_bitmap_len: usize,
        index_type: IndexType,
    ) -> Result<Self, IndexError> {
        let cols_for_schema: Vec<(DataType, String)> = indexed_cols
            .iter()
            .map(|c| (c.dtype, c.name.clone()))
            .collect();
        let schema = Schema::new(cols_for_schema);
        let key_size = schema.data_len;

        let idx = match index_type {
            IndexType::BPlusTree => BPlusTree::new(index_name, key_size as u32, bpm)?,
        };

        Ok(Self {
            indexed_cols,
            indexed_col_idxs,
            table_bitmap_len,
            schema,
            key_size,
            idx: Box::new(idx),
        })
    }

    pub fn encode_key(&self, tuple: &Tuple) -> IndexKey {
        let mut data = Vec::with_capacity(self.key_size);
        for c in self.indexed_cols.iter() {
            c.dtype
                .encode_from_tuple(tuple, self.table_bitmap_len, c.offset, &mut data);
        }
        IndexKey(data)
    }

    fn encode_key_from_values(&self, vals: &[Value]) -> IndexKey {
        assert_eq!(vals.len(), self.indexed_cols.len());
        let mut data = Vec::with_capacity(self.key_size);
        for v in vals.iter() {
            // Sort-friendly encoding: must match encode_from_tuple's per-type format
            // exactly so search/scan keys line up with insert keys.
            match v {
                Value::BOOLEAN(b) => data.push(*b as u8),
                Value::INT(i) => {
                    data.extend_from_slice(&((*i as u32) ^ 0x8000_0000).to_be_bytes());
                }
                Value::FLOAT(f) => {
                    data.extend_from_slice(&((f.0 as u32) ^ 0x8000_0000).to_be_bytes());
                }
                Value::TIMESTAMP(t) => {
                    data.extend_from_slice(&((*t as u64) ^ 0x8000_0000_0000_0000).to_be_bytes());
                }
            }
        }
        IndexKey(data)
    }

    fn key_has_null(&self, tuple: &Tuple) -> bool {
        self.indexed_col_idxs
            .iter()
            .any(|&i| is_null(&tuple.data, i))
    }

    pub fn insert(&self, tuple: &Tuple, rid: RecordId) -> InsertResult {
        if self.key_has_null(tuple) {
            return Ok(()); // skip inserting a null value. we could return an error instead, idk
        }
        let key = self.encode_key(tuple);
        self.idx.insert(key, IndexValue::IndexValue(rid))
    }
    pub fn remove(&self, tuple: &Tuple) -> RemoveResult {
        if self.key_has_null(tuple) {
            return Ok(()); // wasn't indexed, nothing to remove
        }
        let key = self.encode_key(tuple);
        self.idx.remove(key)
    }

    /// search looks up a key in the index
    pub fn search(&self, vals: &[Value]) -> Result<Option<RecordId>, IndexError> {
        let key = self.encode_key_from_values(vals);
        match self.idx.search(&key)? {
            Some(IndexValue::IndexValue(rid)) => Ok(Some(rid)),
            _ => Ok(None),
        }
    }

    /// scans the index with start/end key bounds
    pub fn scan(
        &self,
        start: Bound<&[Value]>,
        end: Bound<&[Value]>,
    ) -> Result<IndexIter, IndexError> {
        let to_key = |b: Bound<&[Value]>| -> Bound<IndexKey> {
            match b {
                Bound::Included(v) => Bound::Included(self.encode_key_from_values(v)),
                Bound::Excluded(v) => Bound::Excluded(self.encode_key_from_values(v)),
                Bound::Unbounded => Bound::Unbounded,
            }
        };
        self.idx.scan((to_key(start), to_key(end)))
    }
}

#[cfg(test)]
mod tests {
    use crate::catalog::index_schema::IndexType;
    use crate::catalog::index_schema::TableIndex;
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::replacer::ArcReplacer;
    use std::sync::Arc;
    use tempfile::tempdir;

    use crate::buffer_pool::BufferPoolManager;
    use crate::catalog::{DataType, Schema, Value};

    fn make_bpm(num_frames: usize) -> (Arc<BufferPoolManager>, tempfile::TempDir) {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("db");
        let log_path = dir.path().join("log");
        let dm = DiskManager::new(&db_path, &log_path).expect("disk manager");
        let scheduler = DiskScheduler::new(dm);
        let replacer = ArcReplacer::new(num_frames);
        (
            Arc::new(create_buffer_pool_manager(num_frames, replacer, scheduler)),
            dir,
        )
    }

    #[test]
    fn test_index_encoder() {
        // 1. create a schema
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::BOOLEAN, "b".to_string()),
            (DataType::INT, "c".to_string()),
            (DataType::INT, "d".to_string()),
        ]);
        // each tuple in this schema should be 4 + 1 + 4 + 4 bytes
        assert_eq!(13, schema.data_len);

        // 2. create a tuple that abides to this schema
        let values = vec![
            Some(Value::INT(12)),
            Some(Value::BOOLEAN(true)),
            Some(Value::INT(13)),
            Some(Value::INT(14)),
        ];
        let tup = schema.encode_tuple(&values).unwrap();

        // 3. create an index
        let index_cols = vec![
            schema.cols.get(0).unwrap().clone(),
            schema.cols.get(2).unwrap().clone(),
        ];
        let index_col_idxs = vec![0, 2];

        let (bpm, _tmp) = make_bpm(10);

        let index = TableIndex::new(
            bpm,
            "mock_index".to_string(),
            index_cols,
            index_col_idxs,
            schema.bitmap_len,
            IndexType::BPlusTree,
        )
        .unwrap();
        assert_eq!(8, index.schema.data_len);

        let tuples = vec![tup];
        for t in tuples.iter() {
            let index_key = index.encode_key(t).0;
            assert_eq!(2 * 4, index_key.len());
        }
    }
}
