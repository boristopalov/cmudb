use std::ops::Bound;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{DataType, IndexType, Schema, Value};
use crate::index::b_plus_page::{MAX_KEY_LEN, PageId};
use crate::index::{BPlusTree, Index, IndexError, InsertError, InsertResult, RemoveResult};
use crate::table_heap::{RecordId, Tuple};

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
pub struct IndexKey(pub Vec<u8>);

#[derive(Debug)]
pub enum IndexValue {
    IndexValue(RecordId),
}

/// Bytes appended to every index key: the record id, big-endian, so that entries for the
/// same user key sort by record id.
const RID_SUFFIX_LEN: usize = 8;

fn rid_suffix(rid: RecordId) -> [u8; RID_SUFFIX_LEN] {
    let mut out = [0u8; RID_SUFFIX_LEN];
    out[..4].copy_from_slice(&rid.page_id.get().to_be_bytes());
    out[4..].copy_from_slice(&rid.slot_id.to_be_bytes());
    out
}

/// Smallest and largest record ids, used to bracket all entries of one user key.
const RID_MIN: RecordId = RecordId {
    page_id: PageId::new(0),
    slot_id: 0,
};
const RID_MAX: RecordId = RecordId::RESERVED;

/// The schema of the key made of `indexed_col_idxs` of `table_schema`, and the longest B+ tree
/// key it makes: the user key followed by the record id, where a user key is at most the sum
/// of its columns' [`DataType::key_width`]s. A key with a TEXT column, or longer than a tree
/// holds, gets the longest key a tree holds, so that inserting a value too long for it fails
/// with [`IndexError::KeyTooLong`].
///
/// Panics if a column index is out of range for `table_schema`.
fn key_schema(table_schema: &Schema, indexed_col_idxs: &[usize]) -> (Schema, u32) {
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
    let user_key_len = key_cols.iter().try_fold(0usize, |len, (dtype, _)| {
        len.checked_add(dtype.key_width()?)
    });
    let tree_key_len = user_key_len
        .and_then(|len| len.checked_add(RID_SUFFIX_LEN))
        .map_or(MAX_KEY_LEN, |len| len.min(MAX_KEY_LEN));
    let tree_key_len = u32::try_from(tree_key_len).expect("MAX_KEY_LEN fits in u32");
    (Schema::new(key_cols), tree_key_len)
}

/// Iterator over `(user key, record id)` index entries.
pub type TableIndexIter = Box<dyn Iterator<Item = Result<(IndexKey, RecordId), IndexError>>>;

/// An index over some columns of a table. Rows with a NULL in any key column are not stored.
///
/// Under MVCC an index entry stays when its tuple changes, so older snapshots can still reach
/// the old version through it. Several record ids can therefore be filed under the same user
/// key (a deleted row and its re-inserted replacement, say), so each B+ tree key is the user
/// key followed by the record id, and every entry is unique. Readers re-check that the
/// version they can see still has the key they looked up.
///
/// Uniqueness of user keys is enforced by the MVCC write path, which checks the newest
/// version behind each entry while holding [`TableIndex::unique_latch`].
pub struct TableIndex {
    table_schema: Schema,
    /// Position of each key column in the table schema.
    pub indexed_col_idxs: Vec<usize>,
    /// Schema of the index key (the key columns only).
    pub schema: Schema,
    pub index_type: IndexType,
    /// The page [`TableIndex::open`] finds the index at.
    pub header_page_id: PageId,
    idx: Box<dyn Index + Send + Sync>,
    /// Serializes "check the key is free, then claim it" across writers of this index.
    pub unique_latch: Mutex<()>,
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
        let (schema, tree_key_len) = key_schema(&table_schema, &indexed_col_idxs);
        let tree = match index_type {
            IndexType::BPlusTree => BPlusTree::new(index_name, tree_key_len, bpm)?,
        };
        Ok(Self::over(
            tree,
            index_type,
            table_schema,
            indexed_col_idxs,
            schema,
        ))
    }

    /// The index whose tree starts at `header_page_id`, as an earlier [`TableIndex::new`]
    /// over the same data file left it.
    ///
    /// Panics if a column index is out of range for `table_schema`.
    pub fn open(
        bpm: Arc<BufferPoolManager>,
        index_name: String,
        table_schema: Schema,
        indexed_col_idxs: Vec<usize>,
        index_type: IndexType,
        header_page_id: PageId,
    ) -> Result<Self, IndexError> {
        let (schema, tree_key_len) = key_schema(&table_schema, &indexed_col_idxs);
        let tree = match index_type {
            IndexType::BPlusTree => BPlusTree::open(index_name, tree_key_len, header_page_id, bpm)?,
        };
        Ok(Self::over(
            tree,
            index_type,
            table_schema,
            indexed_col_idxs,
            schema,
        ))
    }

    fn over(
        tree: BPlusTree,
        index_type: IndexType,
        table_schema: Schema,
        indexed_col_idxs: Vec<usize>,
        schema: Schema,
    ) -> Self {
        Self {
            table_schema,
            indexed_col_idxs,
            schema,
            index_type,
            header_page_id: tree.header_page_id,
            idx: Box::new(tree),
            unique_latch: Mutex::new(()),
        }
    }

    /// Encodes the key columns of a table tuple, or returns `Ok(None)` when any of them is
    /// NULL.
    ///
    /// Fails with [`IndexError::Catalog`] when a key column cannot be read from the tuple, such
    /// as a tuple too short for the table schema.
    pub fn key_of_tuple(&self, tuple: &Tuple) -> Result<Option<IndexKey>, IndexError> {
        let mut values = Vec::with_capacity(self.indexed_col_idxs.len());
        for &i in &self.indexed_col_idxs {
            match self.table_schema.get_value(tuple, i)? {
                Some(v) => values.push(v),
                None => return Ok(None),
            }
        }
        Ok(Some(self.key_of_values(&values)))
    }

    fn key_of_values(&self, vals: &[Value]) -> IndexKey {
        assert_eq!(
            vals.len(),
            self.indexed_col_idxs.len(),
            "an index key needs one value per key column"
        );
        let mut data = Vec::new();
        for v in vals {
            v.encode_order_preserving(&mut data);
        }
        IndexKey(data)
    }

    fn entry_key(&self, key: &IndexKey, rid: RecordId) -> IndexKey {
        let mut data = Vec::with_capacity(key.0.len() + RID_SUFFIX_LEN);
        data.extend_from_slice(&key.0);
        data.extend_from_slice(&rid_suffix(rid));
        IndexKey(data)
    }

    /// Files `rid` under the tuple's key. A tuple with a NULL key column is skipped, and adding
    /// an entry that already exists is a no-op. Fails with [`IndexError::KeyTooLong`], giving
    /// the lengths of the user key, when the key is longer than the index holds.
    pub fn insert(&self, tuple: &Tuple, rid: RecordId) -> InsertResult {
        let Some(key) = self.key_of_tuple(tuple)? else {
            return Ok(());
        };
        match self
            .idx
            .insert(self.entry_key(&key, rid), IndexValue::IndexValue(rid))
        {
            Err(IndexError::Insert(InsertError::DuplicateKey)) => Ok(()),
            Err(IndexError::KeyTooLong { max, actual }) => Err(IndexError::KeyTooLong {
                max: max.saturating_sub(RID_SUFFIX_LEN),
                actual: actual.saturating_sub(RID_SUFFIX_LEN),
            }),
            other => other,
        }
    }

    /// Removes the entry filing `rid` under the tuple's key. A tuple with a NULL key column is
    /// skipped.
    pub fn remove(&self, tuple: &Tuple, rid: RecordId) -> RemoveResult {
        match self.key_of_tuple(tuple)? {
            Some(key) => self.idx.remove(self.entry_key(&key, rid)),
            None => Ok(()),
        }
    }

    /// Returns every record id filed under the key made of `vals`. Stale entries are
    /// included; see the type-level docs.
    pub fn search(&self, vals: &[Value]) -> Result<Vec<RecordId>, IndexError> {
        self.scan(Bound::Included(vals), Bound::Included(vals))?
            .map(|r| r.map(|(_, rid)| rid))
            .collect()
    }

    /// Record ids filed under exactly `key`, a key made by [`TableIndex::key_of_tuple`].
    pub fn entries_for_key(&self, key: &IndexKey) -> Result<Vec<RecordId>, IndexError> {
        let iter = self.scan_keys(
            Bound::Included(self.entry_key(key, RID_MIN)),
            Bound::Included(self.entry_key(key, RID_MAX)),
        )?;
        iter.map(|r| r.map(|(_, rid)| rid)).collect()
    }

    /// Iterates the `(user key, record id)` entries whose keys fall between `start` and
    /// `end`, in key order.
    ///
    /// Every entry for a user key sorts between that key followed by `RID_MIN` and the key
    /// followed by `RID_MAX`, so including a key includes that whole range, and excluding it
    /// excludes it.
    pub fn scan(
        &self,
        start: Bound<&[Value]>,
        end: Bound<&[Value]>,
    ) -> Result<TableIndexIter, IndexError> {
        let bracket = |vals: &[Value], rid| self.entry_key(&self.key_of_values(vals), rid);
        let start = match start {
            Bound::Included(v) => Bound::Included(bracket(v, RID_MIN)),
            Bound::Excluded(v) => Bound::Excluded(bracket(v, RID_MAX)),
            Bound::Unbounded => Bound::Unbounded,
        };
        let end = match end {
            Bound::Included(v) => Bound::Included(bracket(v, RID_MAX)),
            Bound::Excluded(v) => Bound::Excluded(bracket(v, RID_MIN)),
            Bound::Unbounded => Bound::Unbounded,
        };
        self.scan_keys(start, end)
    }

    fn scan_keys(
        &self,
        start: Bound<IndexKey>,
        end: Bound<IndexKey>,
    ) -> Result<TableIndexIter, IndexError> {
        let iter = self.idx.scan((start, end))?;
        Ok(Box::new(iter.map(|r| {
            r.map(|(mut key, IndexValue::IndexValue(rid))| {
                key.0.truncate(key.0.len().saturating_sub(RID_SUFFIX_LEN));
                (key, rid)
            })
        })))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;

    use ordered_float::OrderedFloat;

    use super::TableIndex;

    use crate::catalog::{CatalogError, DataType, IndexType, Schema, Value};
    use crate::index::IndexError;

    use crate::table_heap::RecordId;
    use crate::testing::pid;

    fn rid(slot_id: u32) -> RecordId {
        RecordId {
            page_id: pid(0),
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
        let (bpm, _dir) = crate::testing::bpm(10);
        let index =
            TableIndex::new(bpm, "i".into(), schema, vec![0, 2], IndexType::BPlusTree).unwrap();

        assert_eq!(8, index.schema.data_len);
        let key = index
            .key_of_tuple(&tuple)
            .unwrap()
            .expect("no NULL key columns");
        assert_eq!(key, index.key_of_values(&[Value::INT(12), Value::INT(13)]));
    }

    #[test]
    fn a_tuple_too_short_for_the_table_schema_is_an_error_not_a_panic() {
        let table_schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::INT, "b".to_string()),
            (DataType::INT, "c".to_string()),
        ]);
        let one_col = Schema::new(vec![(DataType::INT, "a".to_string())]);
        let short = one_col.encode_tuple(&[Some(Value::INT(1))]).unwrap();
        let (bpm, _dir) = crate::testing::bpm(10);
        let index =
            TableIndex::new(bpm, "i".into(), table_schema, vec![2], IndexType::BPlusTree).unwrap();

        assert!(matches!(
            index.key_of_tuple(&short),
            Err(IndexError::Catalog(CatalogError::TupleSchemaMismatch))
        ));
        assert!(matches!(
            index.insert(&short, rid(0)),
            Err(IndexError::Catalog(CatalogError::TupleSchemaMismatch))
        ));
        assert!(matches!(
            index.remove(&short, rid(0)),
            Err(IndexError::Catalog(CatalogError::TupleSchemaMismatch))
        ));
        let entries: Vec<_> = index
            .scan(Bound::Unbounded, Bound::Unbounded)
            .unwrap()
            .collect();
        assert!(entries.is_empty(), "a failed insert stores nothing");
    }

    #[test]
    fn rows_with_a_null_key_column_are_not_indexed() {
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::INT, "b".to_string()),
        ]);
        let (bpm, _dir) = crate::testing::bpm(10);
        let index = TableIndex::new(
            bpm,
            "i".into(),
            schema.clone(),
            vec![0],
            IndexType::BPlusTree,
        )
        .unwrap();
        let null_key = schema.encode_tuple(&[None, Some(Value::INT(1))]).unwrap();

        assert!(index.key_of_tuple(&null_key).unwrap().is_none());
        index
            .insert(&null_key, rid(0))
            .expect("NULL keys are skipped");
        index
            .remove(&null_key, rid(0))
            .expect("NULL keys are skipped");
        let entries: Vec<_> = index
            .scan(Bound::Unbounded, Bound::Unbounded)
            .unwrap()
            .collect();
        assert!(entries.is_empty(), "a NULL-keyed row is not stored");
    }

    #[test]
    fn float_keys_with_the_same_integer_part_are_distinct_and_ordered() {
        let schema = Schema::new(vec![(DataType::FLOAT, "f".to_string())]);
        let (bpm, _dir) = crate::testing::bpm(16);
        let index = TableIndex::new(
            bpm,
            "i".into(),
            schema.clone(),
            vec![0],
            IndexType::BPlusTree,
        )
        .unwrap();
        let floats = [3.7_f64, -1.5, 3.2, -0.25, 0.5];
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
            vec![rid(2)]
        );
        let scanned: Vec<u32> = index
            .scan(Bound::Unbounded, Bound::Unbounded)
            .unwrap()
            .map(|e| e.unwrap().1.slot_id)
            .collect();
        assert_eq!(scanned, vec![1, 3, 4, 2, 0], "scan follows float order");
    }

    #[test]
    fn same_key_can_be_filed_under_several_rids() {
        let schema = Schema::new(vec![(DataType::INT, "a".to_string())]);
        let (bpm, _tmp) = crate::testing::bpm(16);
        let index = TableIndex::new(
            bpm,
            "idx".to_string(),
            schema.clone(),
            vec![0],
            IndexType::BPlusTree,
        )
        .unwrap();
        let t = |v| schema.encode_tuple(&[Some(Value::INT(v))]).unwrap();
        let rid = |p: u32, s| RecordId {
            page_id: pid(p),
            slot_id: s,
        };
        index.insert(&t(5), rid(3, 1)).unwrap();
        index.insert(&t(5), rid(0, 0)).unwrap();
        index
            .insert(&t(5), rid(0, 0))
            .expect("adding an existing entry again is a no-op");
        index.insert(&t(4), rid(9, 9)).unwrap();
        index.insert(&t(6), rid(1, 1)).unwrap();

        assert_eq!(
            index.search(&[Value::INT(5)]).unwrap(),
            vec![rid(0, 0), rid(3, 1)]
        );
        let below_six: Vec<_> = index
            .scan(
                std::ops::Bound::Excluded(&[Value::INT(4)][..]),
                std::ops::Bound::Excluded(&[Value::INT(6)][..]),
            )
            .unwrap()
            .map(|r| r.unwrap().1)
            .collect();
        assert_eq!(below_six, vec![rid(0, 0), rid(3, 1)]);
        index.remove(&t(5), rid(0, 0)).unwrap();
        assert_eq!(index.search(&[Value::INT(5)]).unwrap(), vec![rid(3, 1)]);
    }
}
