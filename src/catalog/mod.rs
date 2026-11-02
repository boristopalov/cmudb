pub mod index_schema;
use crate::{buffer_pool::BufferPoolManager, table_heap::TableHeap};
use crate::{catalog::index_schema::TableIndex, table_heap::Tuple};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::AcqRel;

#[derive(Debug)]
pub enum CatalogError {
    TupleSchemaMismatch,
    ColumnOutOfBounds,
    ColumnNotFound,
    TableExists,
    TableDoesNotExist,
    TableCreationError(String),
    IndexExists,
    IndexDoesNotExist,
    IndexCreationError(String),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatalogError::TupleSchemaMismatch => {
                write!(f, "the tuple does not match the table's schema")
            }
            CatalogError::ColumnOutOfBounds => {
                write!(f, "the column index is past the schema's last column")
            }
            CatalogError::ColumnNotFound => write!(f, "column not found"),
            CatalogError::TableExists => write!(f, "table already exists"),
            CatalogError::TableDoesNotExist => write!(f, "table does not exist"),
            CatalogError::TableCreationError(msg) => write!(f, "table creation failed: {msg}"),
            CatalogError::IndexExists => write!(f, "index already exists"),
            CatalogError::IndexDoesNotExist => write!(f, "index does not exist"),
            CatalogError::IndexCreationError(msg) => write!(f, "index creation failed: {msg}"),
        }
    }
}

pub struct TableInfo {
    pub name: String,
    pub schema: Schema,
    pub heap: TableHeap,
}

pub struct IndexInfo {
    pub oid: u32,
    pub name: String,
    pub table_oid: u32,
    pub index: TableIndex,
}

pub enum IndexType {
    BPlusTree,
}

/// Registry of tables and indexes, shared by reference-counted handles. Executors look tables and
/// indexes up here.
pub struct Catalog {
    next_table_oid: AtomicU32,
    next_index_oid: AtomicU32,
    bpm: Arc<BufferPoolManager>,
    entries: RwLock<CatalogEntries>,
}

#[derive(Default)]
struct CatalogEntries {
    tables: HashMap<u32, Arc<TableInfo>>,
    table_oids_by_name: HashMap<String, u32>,
    indexes: HashMap<u32, Arc<IndexInfo>>,
    index_oids_by_table_name: HashMap<String, HashMap<String, u32>>,
}

impl Catalog {
    pub fn new(bpm: Arc<BufferPoolManager>) -> Self {
        Self {
            next_table_oid: AtomicU32::new(0),
            next_index_oid: AtomicU32::new(0),
            bpm,
            entries: RwLock::new(CatalogEntries::default()),
        }
    }

    fn next_table_oid(&self) -> u32 {
        self.next_table_oid.fetch_add(1, AcqRel)
    }

    fn next_index_oid(&self) -> u32 {
        self.next_index_oid.fetch_add(1, AcqRel)
    }

    pub fn create_table(&self, name: String, schema: Schema) -> Result<u32, CatalogError> {
        let mut entries = self.entries.write().unwrap();
        if entries.table_oids_by_name.contains_key(&name) {
            return Err(CatalogError::TableExists);
        }

        let heap = TableHeap::new(self.bpm.clone())
            .map_err(|e| CatalogError::TableCreationError(e.to_string()))?;
        let tinfo = TableInfo {
            name: name.clone(),
            schema,
            heap,
        };
        let oid = self.next_table_oid();

        entries.tables.insert(oid, Arc::new(tinfo));
        entries.table_oids_by_name.insert(name, oid);
        Ok(oid)
    }

    /// Drops a table by oid, along with any indexes associated with it.
    /// Returns `Ok(true)` if the table existed and was dropped, `Ok(false)` otherwise.
    pub fn drop_table(&self, oid: u32) -> Result<bool, CatalogError> {
        let mut entries = self.entries.write().unwrap();
        let Some(tinfo) = entries.tables.remove(&oid) else {
            return Ok(false);
        };
        entries.table_oids_by_name.remove(&tinfo.name);
        if let Some(table_indexes) = entries.index_oids_by_table_name.remove(&tinfo.name) {
            for ioid in table_indexes.values() {
                entries.indexes.remove(ioid);
            }
        }
        Ok(true)
    }

    pub fn table_oid(&self, name: &str) -> Result<u32, CatalogError> {
        let entries = self.entries.read().unwrap();
        entries
            .table_oids_by_name
            .get(name)
            .copied()
            .ok_or(CatalogError::TableDoesNotExist)
    }

    pub fn get_table(&self, oid: u32) -> Result<Arc<TableInfo>, CatalogError> {
        let entries = self.entries.read().unwrap();
        entries
            .tables
            .get(&oid)
            .cloned()
            .ok_or(CatalogError::TableDoesNotExist)
    }

    /// Creates an index over `col_names` of the table and fills it with the table's existing rows.
    pub fn create_index(
        &self,
        bpm: Arc<BufferPoolManager>,
        table_oid: u32,
        index_name: String,
        col_names: &[&str],
        index_type: IndexType,
    ) -> Result<u32, CatalogError> {
        let mut entries = self.entries.write().unwrap();
        let table = entries
            .tables
            .get(&table_oid)
            .cloned()
            .ok_or(CatalogError::TableDoesNotExist)?;
        let indexed_col_idxs = col_names
            .iter()
            .map(|cname| {
                table
                    .schema
                    .col_idx(cname)
                    .ok_or(CatalogError::ColumnNotFound)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let name_taken = entries
            .index_oids_by_table_name
            .get(&table.name)
            .is_some_and(|table_indexes| table_indexes.contains_key(&index_name));
        if name_taken {
            return Err(CatalogError::IndexExists);
        }

        let index = TableIndex::new(
            bpm,
            index_name.clone(),
            table.schema.clone(),
            indexed_col_idxs,
            index_type,
        )
        .map_err(|e| CatalogError::IndexCreationError(format!("failed to create index: {e}")))?;
        for row in table.heap.iter() {
            let (tuple, rid) = row.map_err(|e| {
                CatalogError::IndexCreationError(format!("failed to scan table: {e}"))
            })?;
            let key = index.key_of_tuple(&tuple).map_err(|e| {
                CatalogError::IndexCreationError(format!("failed to index row {rid}: {e}"))
            })?;
            if let Some(key) = key {
                let taken = index.entries_for_key(&key).map_err(|e| {
                    CatalogError::IndexCreationError(format!("failed to index row {rid}: {e}"))
                })?;
                if !taken.is_empty() {
                    return Err(CatalogError::IndexCreationError(format!(
                        "row {rid} duplicates the key of another row"
                    )));
                }
            }
            index.insert(&tuple, rid).map_err(|e| {
                CatalogError::IndexCreationError(format!("failed to index row {rid}: {e}"))
            })?;
        }

        let oid = self.next_index_oid();
        let info = IndexInfo {
            oid,
            name: index_name.clone(),
            table_oid,
            index,
        };
        entries.indexes.insert(oid, Arc::new(info));
        entries
            .index_oids_by_table_name
            .entry(table.name.clone())
            .or_default()
            .insert(index_name, oid);
        Ok(oid)
    }

    /// Drops an index by oid. Returns `Ok(true)` if the index existed and was dropped.
    pub fn drop_index(&self, oid: u32) -> Result<bool, CatalogError> {
        let mut entries = self.entries.write().unwrap();
        let Some(info) = entries.indexes.remove(&oid) else {
            return Ok(false);
        };
        let table_name = entries
            .tables
            .get(&info.table_oid)
            .map(|tinfo| tinfo.name.clone());
        if let Some(table_indexes) =
            table_name.and_then(|name| entries.index_oids_by_table_name.get_mut(&name))
        {
            table_indexes.remove(&info.name);
        }
        Ok(true)
    }

    pub fn index_oid(&self, table_name: &str, index_name: &str) -> Result<u32, CatalogError> {
        let entries = self.entries.read().unwrap();
        let indexes = entries
            .index_oids_by_table_name
            .get(table_name)
            .ok_or(CatalogError::TableDoesNotExist)?;

        indexes
            .get(index_name)
            .copied()
            .ok_or(CatalogError::IndexDoesNotExist)
    }

    pub fn get_index(&self, oid: u32) -> Result<Arc<IndexInfo>, CatalogError> {
        let entries = self.entries.read().unwrap();
        entries
            .indexes
            .get(&oid)
            .cloned()
            .ok_or(CatalogError::IndexDoesNotExist)
    }

    pub fn get_table_indexes(&self, table_oid: u32) -> Result<Vec<Arc<IndexInfo>>, CatalogError> {
        let entries = self.entries.read().unwrap();
        if !entries.tables.contains_key(&table_oid) {
            return Err(CatalogError::TableDoesNotExist);
        }
        Ok(entries
            .indexes
            .values()
            .filter(|info| info.table_oid == table_oid)
            .cloned()
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub cols: Vec<Column>,
    /// Size of the column payload in bytes, excluding the null bitmap.
    pub data_len: usize,
    /// Size in bytes of the null bitmap that starts every encoded tuple.
    pub bitmap_len: usize,
}

impl Schema {
    pub fn new(columns: Vec<(DataType, String)>) -> Self {
        let bitmap_len = columns.len().div_ceil(8);
        let mut offset: usize = 0;
        let cols: Vec<Column> = columns
            .into_iter()
            .map(|(dtype, name)| {
                let col = Column {
                    name,
                    dtype,
                    offset,
                };
                offset += col.dtype.size();
                col
            })
            .collect();
        Schema {
            cols,
            data_len: offset,
            bitmap_len,
        }
    }

    /// Returns a value given a tuple and a column index. `Ok(None)` means the
    /// slot is null per the tuple's bitmap.
    pub fn get_value(&self, tuple: &Tuple, col_idx: usize) -> Result<Option<Value>, CatalogError> {
        let col = self
            .cols
            .get(col_idx)
            .ok_or(CatalogError::ColumnOutOfBounds)?;
        if is_null(&tuple.data, col_idx)? {
            return Ok(None);
        }
        let bytes = tuple
            .data
            .get(self.bitmap_len..)
            .and_then(|payload| payload.get(col.payload_range()))
            .ok_or(CatalogError::TupleSchemaMismatch)?;
        Ok(Some(Value::decode(bytes, &col.dtype)))
    }

    pub fn encode_tuple(&self, values: &[Option<Value>]) -> Result<Tuple, CatalogError> {
        if values.len() != self.cols.len() {
            return Err(CatalogError::TupleSchemaMismatch);
        }
        let mut data = vec![0u8; self.bitmap_len];
        data.reserve(self.data_len);
        for (i, (col, val)) in self.cols.iter().zip(values).enumerate() {
            match val {
                Some(v) => {
                    if v.data_type() != col.dtype {
                        return Err(CatalogError::TupleSchemaMismatch);
                    }
                    v.encode(&mut data);
                }
                None => {
                    data[i / 8] |= 1 << (i % 8);
                    data.resize(data.len() + col.dtype.size(), 0);
                }
            }
        }
        Ok(Tuple { data })
    }

    pub fn col_idx(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|c| c.name == name)
    }
}

/// Reads the null bit for `col_idx` from a tuple's bitmap prefix. Fails with
/// [`CatalogError::TupleSchemaMismatch`] when the tuple is too short to hold that bit.
pub fn is_null(data: &[u8], col_idx: usize) -> Result<bool, CatalogError> {
    let byte = data
        .get(col_idx / 8)
        .ok_or(CatalogError::TupleSchemaMismatch)?;
    Ok((byte >> (col_idx % 8)) & 1 == 1)
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub dtype: DataType,
    /// Byte offset of the column within the payload, which follows the null bitmap.
    pub offset: usize,
}

impl Column {
    fn payload_range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.dtype.size()
    }
}

/// A non-NULL SQL value. FLOAT is totally ordered through `OrderedFloat`: NaN equals itself and
/// sorts above every other float.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum Value {
    BOOLEAN(bool),
    INT(i32),
    FLOAT(OrderedFloat<f32>),
    TIMESTAMP(i64),
}

impl Value {
    pub fn data_type(&self) -> DataType {
        match self {
            Value::BOOLEAN(_) => DataType::BOOLEAN,
            Value::INT(_) => DataType::INT,
            Value::FLOAT(_) => DataType::FLOAT,
            Value::TIMESTAMP(_) => DataType::TIMESTAMP,
        }
    }
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Value::BOOLEAN(b) => out.push(*b as u8),
            Value::INT(i) => out.extend_from_slice(&i.to_le_bytes()),
            Value::FLOAT(f) => out.extend_from_slice(&f.to_le_bytes()),
            Value::TIMESTAMP(t) => out.extend_from_slice(&t.to_le_bytes()),
        }
    }

    fn decode(bytes: &[u8], dtype: &DataType) -> Self {
        match dtype {
            DataType::BOOLEAN => Value::BOOLEAN(bytes[0] != 0),
            DataType::INT => Value::INT(i32::from_le_bytes(bytes.try_into().unwrap())),
            DataType::FLOAT => {
                Value::FLOAT(OrderedFloat(f32::from_le_bytes(bytes.try_into().unwrap())))
            }
            DataType::TIMESTAMP => Value::TIMESTAMP(i64::from_le_bytes(bytes.try_into().unwrap())),
        }
    }
}

const SIGN_BIT_32: u32 = 1 << 31;
const SIGN_BIT_64: u64 = 1 << 63;

impl Value {
    /// Appends a big-endian encoding of the value whose bytewise order equals the `Ord` order of
    /// `Value`, so that index keys sort like the values they encode.
    pub fn encode_order_preserving(&self, out: &mut Vec<u8>) {
        match self {
            Value::BOOLEAN(b) => out.push(*b as u8),
            Value::INT(i) => out.extend_from_slice(&((*i as u32) ^ SIGN_BIT_32).to_be_bytes()),
            Value::FLOAT(f) => out.extend_from_slice(&order_preserving_f32_bits(f.0).to_be_bytes()),
            Value::TIMESTAMP(t) => {
                out.extend_from_slice(&((*t as u64) ^ SIGN_BIT_64).to_be_bytes())
            }
        }
    }
}

/// Maps an `f32` to a `u32` that orders like `OrderedFloat<f32>`: every NaN maps to one value above
/// positive infinity, and `-0.0` maps to the same value as `0.0`.
fn order_preserving_f32_bits(f: f32) -> u32 {
    let canonical = if f.is_nan() {
        f32::NAN
    } else if f == 0.0 {
        0.0
    } else {
        f
    };
    let bits = canonical.to_bits();
    if bits & SIGN_BIT_32 == 0 {
        bits | SIGN_BIT_32
    } else {
        !bits
    }
}

impl std::ops::Add for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        use Value::*;
        match (self, rhs) {
            (INT(a), INT(b)) => INT(a + b),
            (FLOAT(a), FLOAT(b)) => FLOAT(a + b),
            _ => unreachable!("type-checked at plan time"),
        }
    }
}

impl std::ops::Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Value {
        use Value::*;
        match (self, rhs) {
            (INT(a), INT(b)) => INT(a - b),
            (FLOAT(a), FLOAT(b)) => FLOAT(a - b),
            _ => unreachable!("type-checked at plan time"),
        }
    }
}

impl std::ops::Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        use Value::*;
        match (self, rhs) {
            (INT(a), INT(b)) => INT(a * b),
            (FLOAT(a), FLOAT(b)) => FLOAT(a * b),
            _ => unreachable!("type-checked at plan time"),
        }
    }
}

impl std::ops::Div for Value {
    type Output = Value;
    fn div(self, rhs: Value) -> Value {
        use Value::*;
        match (self, rhs) {
            (INT(a), INT(b)) => INT(a / b),
            (FLOAT(a), FLOAT(b)) => FLOAT(a / b),
            _ => unreachable!("type-checked at plan time"),
        }
    }
}

/// The type of a column. Every type has a fixed width (see `size`), so every tuple of a
/// table has the same length. MVCC relies on that: updates overwrite the tuple in place (see
/// `TablePage::update_in_place`). A variable-length type would need out-of-place updates (a
/// new slot plus a forwarding pointer, or delete + insert) before it can be added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    BOOLEAN,
    /// Signed 32-bit integer.
    INT,
    /// 32-bit IEEE 754 float.
    FLOAT,
    /// Signed 64-bit timestamp, interpreted as UTC.
    TIMESTAMP,
}

impl DataType {
    fn size(&self) -> usize {
        match self {
            DataType::BOOLEAN => 1,
            DataType::INT => 4,
            DataType::FLOAT => 4,
            DataType::TIMESTAMP => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_errors_display_readable_messages() {
        let cases = [
            (
                CatalogError::TupleSchemaMismatch,
                "the tuple does not match the table's schema",
            ),
            (
                CatalogError::ColumnOutOfBounds,
                "the column index is past the schema's last column",
            ),
            (CatalogError::ColumnNotFound, "column not found"),
            (CatalogError::TableExists, "table already exists"),
            (CatalogError::TableDoesNotExist, "table does not exist"),
            (
                CatalogError::TableCreationError("no free frame".to_string()),
                "table creation failed: no free frame",
            ),
            (CatalogError::IndexExists, "index already exists"),
            (CatalogError::IndexDoesNotExist, "index does not exist"),
            (
                CatalogError::IndexCreationError("failed to index row 3".to_string()),
                "index creation failed: failed to index row 3",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected, "display of {err:?}");
        }
    }

    #[test]
    fn test_schema_encode_decode_roundtrip() {
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::BOOLEAN, "b".to_string()),
            (DataType::FLOAT, "c".to_string()),
            (DataType::TIMESTAMP, "d".to_string()),
        ]);
        assert_eq!(4 + 1 + 4 + 8, schema.data_len);
        assert_eq!(1, schema.bitmap_len);

        let values = vec![
            Some(Value::INT(-42)),
            Some(Value::BOOLEAN(true)),
            Some(Value::FLOAT(OrderedFloat(3.5))),
            Some(Value::TIMESTAMP(1_700_000_000_000)),
        ];
        let tuple = schema.encode_tuple(&values).unwrap();
        assert_eq!(schema.bitmap_len + schema.data_len, tuple.data.len());

        for (i, expected) in values.iter().enumerate() {
            let got = schema.get_value(&tuple, i).unwrap();
            assert_eq!(*expected, got);
        }

        assert!(matches!(
            schema.get_value(&tuple, 99),
            Err(CatalogError::ColumnOutOfBounds)
        ));
    }

    #[test]
    fn test_schema_encode_decode_with_nulls() {
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::BOOLEAN, "b".to_string()),
            (DataType::INT, "c".to_string()),
        ]);

        let values = vec![Some(Value::INT(7)), None, Some(Value::INT(9))];
        let tuple = schema.encode_tuple(&values).unwrap();
        assert_eq!(schema.bitmap_len + schema.data_len, tuple.data.len());

        assert_eq!(Some(Value::INT(7)), schema.get_value(&tuple, 0).unwrap());
        assert_eq!(None, schema.get_value(&tuple, 1).unwrap());
        assert_eq!(Some(Value::INT(9)), schema.get_value(&tuple, 2).unwrap());
    }

    #[test]
    fn get_value_of_a_tuple_shorter_than_its_schema_is_an_error() {
        let schema = Schema::new(vec![(DataType::INT, "a".to_string())]);
        assert!(matches!(
            schema.get_value(&Tuple { data: vec![] }, 0),
            Err(CatalogError::TupleSchemaMismatch)
        ));
        assert!(matches!(
            schema.get_value(&Tuple { data: vec![0] }, 0),
            Err(CatalogError::TupleSchemaMismatch)
        ));
        assert!(matches!(
            is_null(&[], 0),
            Err(CatalogError::TupleSchemaMismatch)
        ));
    }

    #[test]
    fn test_schema_col_idx() {
        let schema = Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::BOOLEAN, "active".to_string()),
            (DataType::FLOAT, "score".to_string()),
        ]);

        assert_eq!(Some(0), schema.col_idx("id"));
        assert_eq!(Some(1), schema.col_idx("active"));
        assert_eq!(Some(2), schema.col_idx("score"));
        assert_eq!(None, schema.col_idx("missing"));
    }

    #[test]
    fn test_schema_encode_tuple_type_mismatch() {
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::BOOLEAN, "b".to_string()),
        ]);

        let bad = vec![Some(Value::INT(1)), Some(Value::INT(2))];
        assert!(matches!(
            schema.encode_tuple(&bad),
            Err(CatalogError::TupleSchemaMismatch)
        ));
    }

    #[test]
    fn order_preserving_encoding_sorts_like_value_ord() {
        let mut values = vec![
            Value::FLOAT(OrderedFloat(f32::NAN)),
            Value::FLOAT(OrderedFloat(f32::INFINITY)),
            Value::FLOAT(OrderedFloat(3.7)),
            Value::FLOAT(OrderedFloat(3.2)),
            Value::FLOAT(OrderedFloat(0.0)),
            Value::FLOAT(OrderedFloat(-0.0)),
            Value::FLOAT(OrderedFloat(-1.5)),
            Value::FLOAT(OrderedFloat(-3.7)),
            Value::FLOAT(OrderedFloat(f32::NEG_INFINITY)),
            Value::FLOAT(OrderedFloat(-f32::NAN)),
        ];
        values.extend([i32::MIN, -7, -1, 0, 1, 7, i32::MAX].map(Value::INT));
        values.extend([i64::MIN, -1, 0, 1, i64::MAX].map(Value::TIMESTAMP));
        values.extend([false, true].map(Value::BOOLEAN));
        let encode = |v: &Value| {
            let mut out = Vec::new();
            v.encode_order_preserving(&mut out);
            out
        };
        for a in &values {
            for b in &values {
                if a.data_type() == b.data_type() {
                    assert_eq!(
                        a.cmp(b),
                        encode(a).cmp(&encode(b)),
                        "encoded order of {a:?} vs {b:?} must match Value order"
                    );
                }
            }
        }
    }

    fn test_catalog() -> (Catalog, tempfile::TempDir) {
        let (catalog, _bpm, dir) = crate::testing::catalog(32);
        (catalog, dir)
    }

    #[test]
    fn create_index_backfills_rows_already_in_the_table() {
        let (catalog, _dir) = test_catalog();
        let schema = Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::INT, "v".to_string()),
        ]);
        let oid = catalog.create_table("t".into(), schema.clone()).unwrap();
        let table = catalog.get_table(oid).unwrap();
        let mut rids = Vec::new();
        for id in 0..5 {
            let tuple = schema
                .encode_tuple(&[Some(Value::INT(id)), Some(Value::INT(id * 10))])
                .unwrap();
            rids.push(table.heap.insert(&tuple).unwrap());
        }

        let ioid = catalog
            .create_index(
                catalog.bpm.clone(),
                oid,
                "t_id".into(),
                &["id"],
                IndexType::BPlusTree,
            )
            .unwrap();

        let index = catalog.get_index(ioid).unwrap();
        for (id, rid) in rids.iter().enumerate() {
            assert_eq!(
                index.index.search(&[Value::INT(id as i32)]).unwrap(),
                vec![*rid],
                "row with id {id} is findable through the new index"
            );
        }
    }

    #[test]
    fn create_index_fails_and_registers_nothing_when_an_existing_row_is_too_short() {
        let (catalog, _dir) = test_catalog();
        let schema = Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::INT, "v".to_string()),
        ]);
        let oid = catalog.create_table("t".into(), schema).unwrap();
        let table = catalog.get_table(oid).unwrap();
        let one_col = Schema::new(vec![(DataType::INT, "id".to_string())]);
        let short = one_col.encode_tuple(&[Some(Value::INT(1))]).unwrap();
        table.heap.insert(&short).unwrap();

        let result = catalog.create_index(
            catalog.bpm.clone(),
            oid,
            "t_v".into(),
            &["v"],
            IndexType::BPlusTree,
        );

        assert!(matches!(result, Err(CatalogError::IndexCreationError(_))));
        assert!(catalog.get_table_indexes(oid).unwrap().is_empty());
    }

    #[test]
    fn create_index_fails_and_registers_nothing_when_existing_rows_have_duplicate_keys() {
        let (catalog, _dir) = test_catalog();
        let schema = Schema::new(vec![(DataType::INT, "id".to_string())]);
        let oid = catalog.create_table("t".into(), schema.clone()).unwrap();
        let table = catalog.get_table(oid).unwrap();
        let dup = schema.encode_tuple(&[Some(Value::INT(1))]).unwrap();
        table.heap.insert(&dup).unwrap();
        table.heap.insert(&dup).unwrap();

        let result = catalog.create_index(
            catalog.bpm.clone(),
            oid,
            "t_id".into(),
            &["id"],
            IndexType::BPlusTree,
        );

        assert!(matches!(result, Err(CatalogError::IndexCreationError(_))));
        assert!(catalog.get_table_indexes(oid).unwrap().is_empty());
    }
}
