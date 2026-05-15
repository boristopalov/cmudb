pub mod index_schema;
use crate::table_heap::TableHeapIterator;
use crate::{buffer_pool::BufferPoolManager, table_heap::TableHeap};
use crate::{catalog::index_schema::TableIndex, table_heap::Tuple};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
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
    IndexExists,
    IndexDoesNotExist,
    IndexCreationError(String),
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

/// Stores atomic reference counted pointers to tables and indexes.
/// Executors should retrieve tables and indexes from here so that they can do work.
///
/// Lock ordering, to avoid deadlocks: always acquire in this order if multiple are needed:
///   tables -> table_names -> indexes -> index_names
pub struct Catalog {
    next_table_oid: AtomicU32,
    next_index_oid: AtomicU32,
    bpm: Arc<BufferPoolManager>,
    tables: RwLock<HashMap<u32, Arc<TableInfo>>>, // table oid -> table info
    table_names: RwLock<HashMap<String, u32>>,    // table name -> table oid
    indexes: RwLock<HashMap<u32, Arc<IndexInfo>>>, // index oid -> index info
    index_names: RwLock<HashMap<String, HashMap<String, u32>>>, // table name -> index name -> index oid
}

impl Catalog {
    pub fn new(bpm: Arc<BufferPoolManager>) -> Self {
        Self {
            next_table_oid: AtomicU32::new(0),
            next_index_oid: AtomicU32::new(0),
            bpm,
            tables: RwLock::new(HashMap::new()),
            table_names: RwLock::new(HashMap::new()),
            indexes: RwLock::new(HashMap::new()),
            index_names: RwLock::new(HashMap::new()),
        }
    }

    fn next_table_oid(&self) -> u32 {
        self.next_table_oid.fetch_add(1, AcqRel)
    }

    fn next_index_oid(&self) -> u32 {
        self.next_index_oid.fetch_add(1, AcqRel)
    }

    pub fn create_table(&self, name: String, schema: Schema) -> Result<u32, CatalogError> {
        let mut tables = self.tables.write().unwrap();
        let mut table_names = self.table_names.write().unwrap();

        if table_names.contains_key(&name) {
            return Err(CatalogError::TableExists);
        }

        let heap = TableHeap::new(self.bpm.clone()).unwrap();
        let tinfo = TableInfo {
            name: name.clone(),
            schema,
            heap,
        };
        let oid = self.next_table_oid();

        tables.insert(oid, Arc::new(tinfo));
        table_names.insert(name, oid);
        Ok(oid)
    }

    /// Drops a table by oid, along with any indexes associated with it.
    /// Returns `Ok(true)` if the table existed and was dropped, `Ok(false)` otherwise.
    pub fn drop_table(&self, oid: u32) -> Result<bool, CatalogError> {
        let mut tables = self.tables.write().unwrap();
        let mut table_names = self.table_names.write().unwrap();
        let mut indexes = self.indexes.write().unwrap();
        let mut index_names = self.index_names.write().unwrap();

        let Some(tinfo) = tables.remove(&oid) else {
            return Ok(false);
        };
        table_names.remove(&tinfo.name);

        // drop any indexes on this table
        if let Some(table_indexes) = index_names.remove(&tinfo.name) {
            for (_iname, ioid) in table_indexes {
                indexes.remove(&ioid);
            }
        }
        Ok(true)
    }

    pub fn table_oid(&self, name: &str) -> Result<u32, CatalogError> {
        let table_names = self.table_names.read().unwrap();
        table_names
            .get(name)
            .copied()
            .ok_or(CatalogError::TableDoesNotExist)
    }

    pub fn get_table(&self, oid: u32) -> Result<Arc<TableInfo>, CatalogError> {
        let tables = self.tables.read().unwrap();
        tables
            .get(&oid)
            .cloned()
            .ok_or(CatalogError::TableDoesNotExist)
    }

    /// TODO: populate the index if the table it is being created for has data in it
    pub fn create_index(
        &self,
        bpm: Arc<BufferPoolManager>,
        table_oid: u32,
        name: String, // index name
        col_names: &[&str],
        index_type: IndexType,
    ) -> Result<u32, CatalogError> {
        let (table_name, indexed_cols, indexed_col_idxs, table_bitmap_len) = {
            let tables = self.tables.read().unwrap();
            let tinfo = tables
                .get(&table_oid)
                .ok_or(CatalogError::TableDoesNotExist)?;

            let mut indexed_cols = Vec::with_capacity(col_names.len());
            let mut indexed_col_idxs = Vec::with_capacity(col_names.len());
            for cname in col_names {
                let idx = tinfo
                    .schema
                    .col_idx(cname)
                    .ok_or(CatalogError::ColumnNotFound)?;
                indexed_cols.push(tinfo.schema.cols[idx].clone());
                indexed_col_idxs.push(idx);
            }
            (
                tinfo.name.clone(),
                indexed_cols,
                indexed_col_idxs,
                tinfo.schema.bitmap_len,
            )
        };

        let mut indexes = self.indexes.write().unwrap();
        let mut index_names = self.index_names.write().unwrap();

        if let Some(table_indexes) = index_names.get(&table_name) {
            if table_indexes.contains_key(&name) {
                return Err(CatalogError::IndexExists);
            }
        }

        let index = TableIndex::new(
            bpm,
            name.clone(),
            indexed_cols,
            indexed_col_idxs,
            table_bitmap_len,
            index_type,
        )
        .map_err(|_| CatalogError::IndexCreationError(("failed to create index").to_string()))?;
        let oid = self.next_index_oid();
        let info = IndexInfo {
            oid,
            name: name.clone(),
            table_oid,
            index,
        };
        indexes.insert(oid, Arc::new(info));
        index_names.entry(table_name).or_default().insert(name, oid);
        Ok(oid)
    }

    /// Drops an index by oid. Returns `Ok(true)` if the index existed and was dropped.
    pub fn drop_index(&self, oid: u32) -> Result<bool, CatalogError> {
        let tables = self.tables.read().unwrap();
        let mut indexes = self.indexes.write().unwrap();
        let mut index_names = self.index_names.write().unwrap();

        let Some(info) = indexes.remove(&oid) else {
            return Ok(false);
        };

        // remove from index_names: look up the table name via table_oid
        if let Some(tinfo) = tables.get(&info.table_oid) {
            if let Some(table_indexes) = index_names.get_mut(&tinfo.name) {
                table_indexes.remove(&info.name);
            }
        }
        Ok(true)
    }

    pub fn index_oid(&self, table_name: &str, index_name: &str) -> Result<u32, CatalogError> {
        let index_names = self.index_names.read().unwrap();
        let indexes = index_names
            .get(table_name)
            .ok_or(CatalogError::TableDoesNotExist)?;

        indexes
            .get(index_name)
            .copied()
            .ok_or(CatalogError::IndexDoesNotExist)
    }

    pub fn get_index(&self, oid: u32) -> Result<Arc<IndexInfo>, CatalogError> {
        let indexes = self.indexes.read().unwrap();
        indexes
            .get(&oid)
            .cloned()
            .ok_or(CatalogError::IndexDoesNotExist)
    }

    pub fn get_table_indexes(&self, table_oid: u32) -> Result<Vec<Arc<IndexInfo>>, CatalogError> {
        let tables = self.tables.read().unwrap();
        if !tables.contains_key(&table_oid) {
            return Err(CatalogError::TableDoesNotExist);
        }
        let indexes = self.indexes.read().unwrap();
        Ok(indexes
            .values()
            .filter(|info| info.table_oid == table_oid)
            .cloned()
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub cols: Vec<Column>,
    pub data_len: usize, // payload size only (sum of column sizes); does not include the bitmap prefix
    pub bitmap_len: usize, // bytes of null bitmap that prefix every encoded tuple
    /// hacky way to track how many columns are from the left join table
    /// note this assumes 2-way joins only
    /// this is needed to correctly resolve column indices after a join, since
    /// after a join the combined schema does not make any distinction between
    /// the left table's columns and the right table.
    pub join_offset: Option<usize>,
}

impl Schema {
    pub fn new(columns: Vec<(DataType, String)>) -> Self {
        let bitmap_len = (columns.len() + 7) / 8;
        let mut offset: usize = 0;
        let cols: Vec<Column> = columns
            .into_iter()
            .map(|(dtype, name)| {
                let inlined = dtype.inline();
                let col = Column {
                    name,
                    dtype,
                    offset,
                    inlined,
                };
                offset += col.dtype.size();
                col
            })
            .collect();
        Schema {
            cols,
            data_len: offset,
            bitmap_len,
            join_offset: None,
        }
    }

    /// Returns a value given a tuple and a column index. `Ok(None)` means the
    /// slot is null per the tuple's bitmap.
    pub fn get_value(&self, tuple: &Tuple, col_idx: usize) -> Result<Option<Value>, CatalogError> {
        let col = self
            .cols
            .get(col_idx)
            .ok_or(CatalogError::ColumnOutOfBounds)?;
        if is_null(&tuple.data, col_idx) {
            return Ok(None);
        }
        let start = self.bitmap_len + col.offset;
        let end = start + col.dtype.size();
        let bytes = &tuple.data[start..end];
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

/// Reads the null bit for `col_idx` from a tuple's bitmap prefix.
pub fn is_null(data: &[u8], col_idx: usize) -> bool {
    (data[col_idx / 8] >> (col_idx % 8)) & 1 == 1
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub dtype: DataType,
    pub offset: usize, // offset into the payload region (after the bitmap prefix)
    pub inlined: bool, // whether the data is inlined (true for fixed-width data, false for variable-length)
}

/// Defines the value types we support.
/// Note: FLOAT uses the OrderedFloat struct from the ordered-float crate.
/// NaN is treated as greater than all other values, and equal to itself.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum Value {
    BOOLEAN(bool),
    INT(i32),
    FLOAT(OrderedFloat<f32>),
    TIMESTAMP(i64),
    // ENUM // assume static, i.e. can't add a member after creating
    // VARCHAR
    // BIGINT
    // VECTOR
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

/// Metadata for a table schema
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    BOOLEAN,
    INT,   // signed, 32 bit
    FLOAT, // 32 bit floating point
    TIMESTAMP, // assume UTC
           // ENUM,
           // VARCHAR(usize),
           // VECTOR(usize),
}

// TODO: this feels weird
impl DataType {
    pub fn encode_from_tuple(
        &self,
        tuple: &Tuple,
        bitmap_len: usize,
        offset: usize,
        out: &mut Vec<u8>,
    ) {
        let off = bitmap_len + offset;
        match self {
            DataType::BOOLEAN => {
                let b = tuple.data.get(off).unwrap();
                out.push(*b);
            }
            DataType::INT => {
                // assumes data is stored little-endian
                let val = i32::from_le_bytes(tuple.data[off..off + 4].try_into().unwrap());
                let big_e_converted = ((val as u32) ^ 0x8000_0000).to_be_bytes();
                out.extend_from_slice(&big_e_converted);
            }
            DataType::FLOAT => {
                let val = f32::from_le_bytes(tuple.data[off..off + 4].try_into().unwrap());
                let big_e_converted = ((val as u32) ^ 0x8000_0000).to_be_bytes();
                out.extend_from_slice(&big_e_converted);
            }
            DataType::TIMESTAMP => {
                let val = i64::from_le_bytes(tuple.data[off..off + 8].try_into().unwrap());
                let big_e_converted = ((val as u64) ^ 0x8000_0000_0000_0000).to_be_bytes();
                out.extend_from_slice(&big_e_converted);
            }
        }
    }

    fn size(&self) -> usize {
        match self {
            DataType::BOOLEAN => 1,
            DataType::INT => 4,
            DataType::FLOAT => 4,
            DataType::TIMESTAMP => 8,
        }
    }

    fn inline(&self) -> bool {
        match self {
            DataType::BOOLEAN => true,
            DataType::INT => true,
            DataType::FLOAT => true,
            DataType::TIMESTAMP => true,
        }
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

    // second value is wrong type
    let bad = vec![Some(Value::INT(1)), Some(Value::INT(2))];
    assert!(matches!(
        schema.encode_tuple(&bad),
        Err(CatalogError::TupleSchemaMismatch)
    ));
}
