use std::ops::Bound;
use std::sync::Arc;

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{DataType, IndexType, Schema, Value};
use crate::index::{BPlusTree, Index, IndexError, IndexIter, InsertResult, RemoveResult};
use crate::table_heap::{RecordId, Tuple};

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
pub struct IndexKey(pub Vec<u8>);

#[derive(Debug)]
pub enum IndexValue {
    IndexValue(RecordId),
}

/// An index over some columns of a table. Rows with a NULL in any key column are not stored.
pub struct TableIndex {
    table_schema: Schema,
    /// Position of each key column in the table schema.
    pub indexed_col_idxs: Vec<usize>,
    /// Schema of the index key (the key columns only).
    pub schema: Schema,
    idx: Box<dyn Index + Send + Sync>,
}

impl TableIndex {
    /// Creates an empty index over `indexed_col_idxs` of `table_schema`.
    ///
    /// Panics if a column index is out of range for `table_schema`.
    pub fn new(
        bpm: Arc<BufferPoolManager>,
        index_name: String,
        table_schema: Schema,
        indexed_col_idxs: Vec<usize>,
        index_type: IndexType,
    ) -> Result<Self, IndexError> {
        let key_cols: Vec<(DataType, String)> = indexed_col_idxs
            .iter()
            .map(|&i| {
                let col = table_schema
                    .cols
                    .get(i)
                    .unwrap_or_else(|| panic!("indexed column {i} is not in the table schema"));
                (col.dtype, col.name.clone())
            })
            .collect();
        let schema = Schema::new(key_cols);

        let idx = match index_type {
            IndexType::BPlusTree => BPlusTree::new(index_name, schema.data_len as u32, bpm)?,
        };

        Ok(Self {
            table_schema,
            indexed_col_idxs,
            schema,
            idx: Box::new(idx),
        })
    }

    /// Encodes the key columns of a table tuple, or returns `None` when any of them is NULL.
    pub fn key_of_tuple(&self, tuple: &Tuple) -> Option<IndexKey> {
        let values = self
            .indexed_col_idxs
            .iter()
            .map(|&i| {
                self.table_schema
                    .get_value(tuple, i)
                    .expect("indexed column index was validated against the table schema")
            })
            .collect::<Option<Vec<Value>>>()?;
        Some(self.key_of_values(&values))
    }

    fn key_of_values(&self, vals: &[Value]) -> IndexKey {
        assert_eq!(
            vals.len(),
            self.indexed_col_idxs.len(),
            "an index key needs one value per key column"
        );
        let mut data = Vec::with_capacity(self.schema.data_len);
        for v in vals {
            v.encode_order_preserving(&mut data);
        }
        IndexKey(data)
    }

    /// Adds `rid` under the tuple's key. A tuple with a NULL key column is skipped.
    pub fn insert(&self, tuple: &Tuple, rid: RecordId) -> InsertResult {
        match self.key_of_tuple(tuple) {
            Some(key) => self.idx.insert(key, IndexValue::IndexValue(rid)),
            None => Ok(()),
        }
    }

    /// Removes the tuple's key. A tuple with a NULL key column is skipped.
    pub fn remove(&self, tuple: &Tuple) -> RemoveResult {
        match self.key_of_tuple(tuple) {
            Some(key) => self.idx.remove(key),
            None => Ok(()),
        }
    }

    /// Whether storing `new` in place of `old` would hit a key another row already holds.
    /// `old` is `None` for a new row. A NULL key column, or a key equal to `old`'s, is no
    /// conflict.
    pub fn conflicts(&self, old: Option<&Tuple>, new: &Tuple) -> Result<bool, IndexError> {
        let Some(key) = self.key_of_tuple(new) else {
            return Ok(false);
        };
        if old.and_then(|old| self.key_of_tuple(old)).as_ref() == Some(&key) {
            return Ok(false);
        }
        Ok(self.idx.search(&key)?.is_some())
    }

    /// Looks up the record stored under the key made of `vals`.
    pub fn search(&self, vals: &[Value]) -> Result<Option<RecordId>, IndexError> {
        let key = self.key_of_values(vals);
        match self.idx.search(&key)? {
            Some(IndexValue::IndexValue(rid)) => Ok(Some(rid)),
            None => Ok(None),
        }
    }

    /// Iterates the entries whose keys fall between `start` and `end`, in key order.
    pub fn scan(
        &self,
        start: Bound<&[Value]>,
        end: Bound<&[Value]>,
    ) -> Result<IndexIter, IndexError> {
        let to_key = |b: Bound<&[Value]>| -> Bound<IndexKey> {
            match b {
                Bound::Included(v) => Bound::Included(self.key_of_values(v)),
                Bound::Excluded(v) => Bound::Excluded(self.key_of_values(v)),
                Bound::Unbounded => Bound::Unbounded,
            }
        };
        self.idx.scan((to_key(start), to_key(end)))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;
    use std::sync::Arc;

    use ordered_float::OrderedFloat;
    use tempfile::tempdir;

    use super::{IndexValue, TableIndex};
    use crate::buffer_pool::BufferPoolManager;
    use crate::catalog::{DataType, IndexType, Schema, Value};
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::replacer::ArcReplacer;
    use crate::table_heap::RecordId;

    fn make_bpm(num_frames: usize) -> (Arc<BufferPoolManager>, tempfile::TempDir) {
        let dir = tempdir().expect("tempdir");
        let dm =
            DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk manager");
        let scheduler = DiskScheduler::new(dm);
        let replacer = ArcReplacer::new(num_frames);
        (
            Arc::new(create_buffer_pool_manager(num_frames, replacer, scheduler)),
            dir,
        )
    }

    fn rid(slot_id: u32) -> RecordId {
        RecordId {
            page_id: 0,
            slot_id,
        }
    }

    #[test]
    fn key_of_tuple_encodes_only_the_indexed_columns() {
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::BOOLEAN, "b".to_string()),
            (DataType::INT, "c".to_string()),
            (DataType::INT, "d".to_string()),
        ]);
        let tuple = schema
            .encode_tuple(&[
                Some(Value::INT(12)),
                Some(Value::BOOLEAN(true)),
                Some(Value::INT(13)),
                Some(Value::INT(14)),
            ])
            .unwrap();
        let (bpm, _dir) = make_bpm(10);
        let index =
            TableIndex::new(bpm, "i".into(), schema, vec![0, 2], IndexType::BPlusTree).unwrap();

        assert_eq!(8, index.schema.data_len);
        let key = index.key_of_tuple(&tuple).expect("no NULL key columns");
        assert_eq!(key, index.key_of_values(&[Value::INT(12), Value::INT(13)]));
    }

    #[test]
    fn rows_with_a_null_key_column_are_not_indexed() {
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::INT, "b".to_string()),
        ]);
        let (bpm, _dir) = make_bpm(10);
        let index = TableIndex::new(
            bpm,
            "i".into(),
            schema.clone(),
            vec![0],
            IndexType::BPlusTree,
        )
        .unwrap();
        let null_key = schema.encode_tuple(&[None, Some(Value::INT(1))]).unwrap();

        assert!(index.key_of_tuple(&null_key).is_none());
        index
            .insert(&null_key, rid(0))
            .expect("NULL keys are skipped");
        index.remove(&null_key).expect("NULL keys are skipped");
        let entries: Vec<_> = index
            .scan(Bound::Unbounded, Bound::Unbounded)
            .unwrap()
            .collect();
        assert!(entries.is_empty(), "a NULL-keyed row must not be stored");
    }

    #[test]
    fn float_keys_with_the_same_integer_part_are_distinct_and_ordered() {
        let schema = Schema::new(vec![(DataType::FLOAT, "f".to_string())]);
        let (bpm, _dir) = make_bpm(16);
        let index = TableIndex::new(
            bpm,
            "i".into(),
            schema.clone(),
            vec![0],
            IndexType::BPlusTree,
        )
        .unwrap();
        let floats = [3.7_f32, -1.5, 3.2, -0.25, 0.5];
        for (slot, f) in floats.iter().enumerate() {
            let tuple = schema
                .encode_tuple(&[Some(Value::FLOAT(OrderedFloat(*f)))])
                .unwrap();
            index
                .insert(&tuple, rid(slot as u32))
                .expect("distinct floats must not collide");
        }

        assert_eq!(
            index.search(&[Value::FLOAT(OrderedFloat(3.2))]).unwrap(),
            Some(rid(2))
        );
        let scanned: Vec<u32> = index
            .scan(Bound::Unbounded, Bound::Unbounded)
            .unwrap()
            .map(|e| {
                let (_, IndexValue::IndexValue(r)) = e.unwrap();
                r.slot_id
            })
            .collect();
        assert_eq!(scanned, vec![1, 3, 4, 2, 0], "scan follows float order");
    }
}
