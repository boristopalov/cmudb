use std::sync::Arc;

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Column, DataType, IndexType, Schema, Value};
use crate::index::{
    BPlusTree, Index, IndexError, InsertError, InsertResult, RemoveError, RemoveResult,
};
use crate::table_heap::{RecordId, Tuple};

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
pub struct IndexKey(pub Vec<u8>);

#[derive(Debug)]
pub enum IndexValue {
    IndexValue(RecordId),
}

pub struct TableIndex {
    indexed_cols: Vec<Column>, // offsets are into the table tuple
    pub schema: Schema,        // schema of the index key, not the tuple
    key_size: usize,
    idx: Box<dyn Index>,
}

impl TableIndex {
    pub fn new(
        bpm: Arc<BufferPoolManager>,
        index_name: String,
        indexed_cols: Vec<Column>,
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
            schema,
            key_size,
            idx: Box::new(idx),
        })
    }

    pub fn encode_key(&self, tuple: &Tuple) -> IndexKey {
        let mut data = Vec::with_capacity(self.schema.data_len);
        for c in self.indexed_cols.iter() {
            c.dtype.encode_from_tuple(tuple, c.offset, &mut data);
        }
        IndexKey(data)
    }

    pub fn insert(&self, tuple: &Tuple, rid: RecordId) -> InsertResult {
        let key = self.encode_key(tuple);
        self.idx.insert(key, IndexValue::IndexValue(rid))
    }
    pub fn remove(&self, tuple: &Tuple) -> RemoveResult {
        let key = self.encode_key(tuple);
        self.idx.remove(key)
    }
    pub fn search(&self, tuple: &Tuple) -> Result<Option<RecordId>, IndexError> {
        let key = self.encode_key(tuple);
        match self.idx.search(&key)? {
            Some(IndexValue::IndexValue(rid)) => Ok(Some(rid)),
            _ => Ok(None),
        }
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
            Value::INT(12),
            Value::BOOLEAN(true),
            Value::INT(13),
            Value::INT(14),
        ];
        let tup = schema.encode_tuple(&values).unwrap();

        // 3. create an index
        let index_cols = vec![
            schema.cols.get(0).unwrap().clone(),
            schema.cols.get(2).unwrap().clone(),
        ];

        let (bpm, _tmp) = make_bpm(10);

        let index = TableIndex::new(
            bpm,
            "mock_index".to_string(),
            index_cols,
            IndexType::BPlusTree,
        )
        .unwrap();

        // we expect each index key to be: 4 bytes + 4 bytes long
        assert_eq!(8, index.schema.data_len);

        // // 3. create index keys
        let tuples = vec![tup];
        for t in tuples.iter() {
            let index_key = index.encode_key(t).0;
            assert_eq!(8, index_key.len());
        }
    }
}
