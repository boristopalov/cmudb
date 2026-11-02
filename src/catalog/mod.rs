pub mod index_schema;
pub mod timestamp;
use crate::{buffer_pool::BufferPoolManager, table_heap::TableHeap};
use crate::{catalog::index_schema::TableIndex, table_heap::Tuple};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::{AcqRel, Acquire};

#[derive(Debug)]
pub enum CatalogError {
    TupleSchemaMismatch,
    /// A value of type `found` was stored in column `column` of type `expected`, and no
    /// implicit conversion exists between the two.
    ColumnType {
        column: String,
        expected: DataType,
        found: DataType,
    },
    /// A value was stored in column `column` of type `expected` that the type cannot hold,
    /// such as a BIGINT past the INT range.
    ColumnOutOfRange {
        column: String,
        expected: DataType,
        value: Value,
    },
    /// A string of `len` characters was stored in column `column`, which holds at most
    /// `max_chars`.
    ColumnTooLong {
        column: String,
        max_chars: u32,
        len: usize,
    },
    ColumnOutOfBounds,
    /// An encoded row of `size` bytes is past what a tuple can address.
    RowTooLarge {
        size: usize,
    },
    ColumnNotFound,
    TableExists,
    TableDoesNotExist,
    TableCreationError(String),
    IndexExists,
    IndexDoesNotExist,
    IndexCreationError(String),
    /// A table from a [`CatalogSnapshot`] could not be opened.
    TableOpenError(String),
    /// An index from a [`CatalogSnapshot`] could not be opened.
    IndexOpenError(String),
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatalogError::TupleSchemaMismatch => {
                write!(f, "the tuple does not match the table's schema")
            }
            CatalogError::ColumnType {
                column,
                expected,
                found,
            } => write!(f, "column `{column}` is {expected}, got {found}"),
            CatalogError::ColumnOutOfRange {
                column,
                expected,
                value,
            } => write!(
                f,
                "{value} is out of range for {expected} column `{column}`"
            ),
            CatalogError::ColumnTooLong {
                column,
                max_chars,
                len,
            } => write!(
                f,
                "value of {len} characters is too long for VARCHAR({max_chars}) column `{column}`"
            ),
            CatalogError::ColumnOutOfBounds => {
                write!(f, "the column index is past the schema's last column")
            }
            CatalogError::RowTooLarge { size } => {
                write!(f, "a row of {size} bytes is too large to store")
            }
            CatalogError::ColumnNotFound => write!(f, "column not found"),
            CatalogError::TableExists => write!(f, "table already exists"),
            CatalogError::TableDoesNotExist => write!(f, "table does not exist"),
            CatalogError::TableCreationError(msg) => write!(f, "table creation failed: {msg}"),
            CatalogError::IndexExists => write!(f, "index already exists"),
            CatalogError::IndexDoesNotExist => write!(f, "index does not exist"),
            CatalogError::IndexCreationError(msg) => write!(f, "index creation failed: {msg}"),
            CatalogError::TableOpenError(msg) => write!(f, "opening a table failed: {msg}"),
            CatalogError::IndexOpenError(msg) => write!(f, "opening an index failed: {msg}"),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    BPlusTree,
}

/// The tables and indexes a [`Catalog`] holds, with the first page of each, which is all
/// [`Catalog::restore`] needs to open them again. Entries are in oid order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogSnapshot {
    pub next_table_oid: u32,
    pub next_index_oid: u32,
    pub tables: Vec<TableEntry>,
    pub indexes: Vec<IndexEntry>,
}

/// One table of a [`CatalogSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableEntry {
    pub oid: u32,
    pub name: String,
    pub columns: Vec<(DataType, String)>,
    /// The first page of the table's heap.
    pub first_page_id: usize,
}

/// One index of a [`CatalogSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub oid: u32,
    pub name: String,
    pub table_oid: u32,
    /// Position of each key column in the table's columns.
    pub key_columns: Vec<usize>,
    pub index_type: IndexType,
    /// The header page of the index's tree.
    pub header_page_id: usize,
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

    /// The catalog `snapshot` describes, with its tables and indexes opened from the pages
    /// of `bpm` it names.
    pub fn restore(
        bpm: Arc<BufferPoolManager>,
        snapshot: &CatalogSnapshot,
    ) -> Result<Self, CatalogError> {
        let mut entries = CatalogEntries::default();
        for t in &snapshot.tables {
            let schema = Schema::new(t.columns.clone());
            let heap = TableHeap::open(bpm.clone(), t.first_page_id)
                .map_err(|e| CatalogError::TableOpenError(format!("table {}: {e}", t.name)))?
                .with_update_reserve(schema.update_reserve());
            let info = TableInfo {
                name: t.name.clone(),
                schema,
                heap,
            };
            entries.tables.insert(t.oid, Arc::new(info));
            entries.table_oids_by_name.insert(t.name.clone(), t.oid);
        }
        for i in &snapshot.indexes {
            let table = entries.tables.get(&i.table_oid).cloned().ok_or_else(|| {
                CatalogError::IndexOpenError(format!(
                    "index {} is on table oid {}, which does not exist",
                    i.name, i.table_oid
                ))
            })?;
            if let Some(&col) = i
                .key_columns
                .iter()
                .find(|&&c| c >= table.schema.cols.len())
            {
                return Err(CatalogError::IndexOpenError(format!(
                    "index {} keys on column {col}, past the end of table {}",
                    i.name, table.name
                )));
            }
            let index = TableIndex::open(
                bpm.clone(),
                i.name.clone(),
                table.schema.clone(),
                i.key_columns.clone(),
                i.index_type,
                i.header_page_id,
            )
            .map_err(|e| CatalogError::IndexOpenError(format!("index {}: {e}", i.name)))?;
            let info = IndexInfo {
                oid: i.oid,
                name: i.name.clone(),
                table_oid: i.table_oid,
                index,
            };
            entries.indexes.insert(i.oid, Arc::new(info));
            entries
                .index_oids_by_table_name
                .entry(table.name.clone())
                .or_default()
                .insert(i.name.clone(), i.oid);
        }
        Ok(Self {
            next_table_oid: AtomicU32::new(snapshot.next_table_oid),
            next_index_oid: AtomicU32::new(snapshot.next_index_oid),
            bpm,
            entries: RwLock::new(entries),
        })
    }

    /// What [`Catalog::restore`] needs to open this catalog's tables and indexes again.
    pub fn snapshot(&self) -> CatalogSnapshot {
        let entries = self.entries.read().unwrap();
        let mut tables: Vec<TableEntry> = entries
            .tables
            .iter()
            .map(|(&oid, t)| TableEntry {
                oid,
                name: t.name.clone(),
                columns: t
                    .schema
                    .cols
                    .iter()
                    .map(|c| (c.dtype, c.name.clone()))
                    .collect(),
                first_page_id: t.heap.first_page_id,
            })
            .collect();
        tables.sort_by_key(|t| t.oid);
        let mut indexes: Vec<IndexEntry> = entries
            .indexes
            .iter()
            .map(|(&oid, i)| IndexEntry {
                oid,
                name: i.name.clone(),
                table_oid: i.table_oid,
                key_columns: i.index.indexed_col_idxs.clone(),
                index_type: i.index.index_type,
                header_page_id: i.index.header_page_id,
            })
            .collect();
        indexes.sort_by_key(|i| i.oid);
        CatalogSnapshot {
            next_table_oid: self.next_table_oid.load(Acquire),
            next_index_oid: self.next_index_oid.load(Acquire),
            tables,
            indexes,
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
            .map_err(|e| CatalogError::TableCreationError(e.to_string()))?
            .with_update_reserve(schema.update_reserve());
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

    /// The oids of every index called `index_name`, across all tables, in ascending order.
    pub fn index_oids_named(&self, index_name: &str) -> Vec<u32> {
        let entries = self.entries.read().unwrap();
        let mut oids: Vec<u32> = entries
            .index_oids_by_table_name
            .values()
            .filter_map(|table_indexes| table_indexes.get(index_name).copied())
            .collect();
        oids.sort_unstable();
        oids
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
    /// Size in bytes of the fixed-width part of the payload, excluding the null bitmap. A text
    /// column takes a [`TEXT_DESCRIPTOR_LEN`]-byte descriptor there; its characters follow
    /// the fixed-width part.
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
                offset += col.dtype.fixed_width();
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
        if col.dtype.is_text() {
            return text_at(&tuple.data, bytes).map(Some);
        }
        Ok(Some(Value::decode(bytes, &col.dtype)))
    }

    /// Encodes one row, converting each value to its column's type as [`Value::coerce_to`]
    /// does. Fails with [`CatalogError::ColumnType`] or [`CatalogError::ColumnOutOfRange`] when
    /// a value does not convert, and [`CatalogError::TupleSchemaMismatch`] when the row has
    /// other than one value per column.
    pub fn encode_tuple(&self, values: &[Option<Value>]) -> Result<Tuple, CatalogError> {
        if values.len() != self.cols.len() {
            return Err(CatalogError::TupleSchemaMismatch);
        }
        let mut data = vec![0u8; self.bitmap_len];
        data.reserve(self.data_len);
        let mut texts = Vec::new();
        for (i, (col, val)) in self.cols.iter().zip(values).enumerate() {
            match val.as_ref().map(|v| col.coerce(v.clone())).transpose()? {
                Some(Value::TEXT(s)) => {
                    texts.push((data.len(), s));
                    data.resize(data.len() + TEXT_DESCRIPTOR_LEN, 0);
                }
                Some(v) => v.encode(&mut data),
                None => {
                    data[i / 8] |= 1 << (i % 8);
                    data.resize(data.len() + col.dtype.fixed_width(), 0);
                }
            }
        }
        for (descriptor_at, s) in texts {
            let descriptor = text_descriptor(data.len(), s.len())?;
            data[descriptor_at..descriptor_at + TEXT_DESCRIPTOR_LEN].copy_from_slice(&descriptor);
            data.extend_from_slice(s.as_bytes());
        }
        Ok(Tuple { data })
    }

    pub fn col_idx(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|c| c.name == name)
    }

    /// Free bytes a table of this schema keeps on each page for updates that make rows longer:
    /// [`TEXT_UPDATE_RESERVE`] when it has a text column, none when every row has one length.
    pub fn update_reserve(&self) -> usize {
        if self.cols.iter().any(|c| c.dtype.is_text()) {
            TEXT_UPDATE_RESERVE
        } else {
            0
        }
    }
}

/// Free bytes each page of a table with a text column keeps for rows that grow, an eighth of a
/// page.
pub const TEXT_UPDATE_RESERVE: usize = crate::disk::PAGE_SIZE / 8;

/// Bytes a text column takes in the fixed-width part of a tuple: the `u32` offset of its
/// characters from the start of the tuple, then their `u32` length in bytes.
pub const TEXT_DESCRIPTOR_LEN: usize = 8;

/// The descriptor of a text value whose UTF-8 bytes start `offset` bytes into the tuple. Fails
/// with [`CatalogError::RowTooLarge`] when either does not fit a `u32`.
fn text_descriptor(offset: usize, len: usize) -> Result<[u8; TEXT_DESCRIPTOR_LEN], CatalogError> {
    let too_large = || CatalogError::RowTooLarge { size: offset + len };
    let offset = u32::try_from(offset).map_err(|_| too_large())?;
    let len = u32::try_from(len).map_err(|_| too_large())?;
    let mut out = [0u8; TEXT_DESCRIPTOR_LEN];
    out[..4].copy_from_slice(&offset.to_le_bytes());
    out[4..].copy_from_slice(&len.to_le_bytes());
    Ok(out)
}

/// The text value `descriptor` points at in `tuple`. Fails with
/// [`CatalogError::TupleSchemaMismatch`] when it points past the tuple or at bytes that are not
/// UTF-8.
fn text_at(tuple: &[u8], descriptor: &[u8]) -> Result<Value, CatalogError> {
    let word = |at: usize| u32::from_le_bytes(descriptor[at..at + 4].try_into().unwrap()) as usize;
    let (offset, len) = (word(0), word(4));
    let bytes = offset
        .checked_add(len)
        .and_then(|end| tuple.get(offset..end))
        .ok_or(CatalogError::TupleSchemaMismatch)?;
    let text = std::str::from_utf8(bytes).map_err(|_| CatalogError::TupleSchemaMismatch)?;
    Ok(Value::text(text))
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
    /// Converts `value` to this column's type, naming the column in the error.
    pub fn coerce(&self, value: Value) -> Result<Value, CatalogError> {
        value.clone().coerce_to(self.dtype).map_err(|e| match e {
            CoercionError::WrongType => CatalogError::ColumnType {
                column: self.name.clone(),
                expected: self.dtype,
                found: value.data_type(),
            },
            CoercionError::OutOfRange => CatalogError::ColumnOutOfRange {
                column: self.name.clone(),
                expected: self.dtype,
                value,
            },
            CoercionError::TooLong { max_chars, len } => CatalogError::ColumnTooLong {
                column: self.name.clone(),
                max_chars,
                len,
            },
        })
    }

    fn payload_range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.dtype.fixed_width()
    }
}

/// A non-NULL SQL value. FLOAT is totally ordered through `OrderedFloat`: NaN equals itself and
/// sorts above every other float. The derived order only ranks values of the same type; values of
/// different types compare through [`Value::compare`]. TEXT orders by its UTF-8 bytes, which is
/// Unicode code point order.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Value {
    BOOLEAN(bool),
    INT(i32),
    BIGINT(i64),
    FLOAT(OrderedFloat<f64>),
    /// Microseconds since 1970-01-01 00:00:00 UTC.
    TIMESTAMP(i64),
    /// A string, in a TEXT or VARCHAR column.
    TEXT(Arc<str>),
}

/// Why a value could not be converted to another type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoercionError {
    /// No implicit conversion exists between the two types, such as BOOLEAN to INT.
    WrongType,
    /// The conversion exists but the value does not fit the target type.
    OutOfRange,
    /// A string of `len` characters stored as a `VARCHAR(max_chars)`.
    TooLong { max_chars: u32, len: usize },
}

impl Value {
    pub fn data_type(&self) -> DataType {
        match self {
            Value::BOOLEAN(_) => DataType::BOOLEAN,
            Value::INT(_) => DataType::INT,
            Value::BIGINT(_) => DataType::BIGINT,
            Value::FLOAT(_) => DataType::FLOAT,
            Value::TIMESTAMP(_) => DataType::TIMESTAMP,
            Value::TEXT(_) => DataType::TEXT,
        }
    }

    /// A TEXT value.
    pub fn text(s: &str) -> Value {
        Value::TEXT(Arc::from(s))
    }

    /// The narrowest integer value holding `i`: an INT when it fits 32 bits, otherwise a BIGINT.
    pub fn integer(i: i64) -> Value {
        i32::try_from(i).map_or(Value::BIGINT(i), Value::INT)
    }

    /// A FLOAT value.
    pub fn float(f: f64) -> Value {
        Value::FLOAT(OrderedFloat(f))
    }

    /// The value as an `i64`, for INT and BIGINT values.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::INT(i) => Some(i64::from(*i)),
            Value::BIGINT(i) => Some(*i),
            _ => None,
        }
    }

    /// The value as an `f64`, for INT, BIGINT and FLOAT values. A BIGINT beyond 2^53 rounds to
    /// the nearest `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::INT(i) => Some(f64::from(*i)),
            Value::BIGINT(i) => Some(*i as f64),
            Value::FLOAT(f) => Some(f.0),
            _ => None,
        }
    }

    /// Converts the value to `to` the way an INSERT or UPDATE stores it in a column of that type:
    /// INT widens to BIGINT and FLOAT, BIGINT widens to FLOAT and narrows to INT when it fits.
    /// FLOAT does not convert to an integer type, BOOLEAN and TIMESTAMP convert to nothing but
    /// themselves, and TEXT converts to a VARCHAR that holds as many characters.
    pub fn coerce_to(self, to: DataType) -> Result<Value, CoercionError> {
        if self.data_type() == to {
            return Ok(self);
        }
        match (self, to) {
            (Value::TEXT(s), DataType::VARCHAR(max_chars)) => {
                let len = s.chars().count();
                if len <= max_chars as usize {
                    Ok(Value::TEXT(s))
                } else {
                    Err(CoercionError::TooLong { max_chars, len })
                }
            }
            (Value::INT(i), DataType::BIGINT) => Ok(Value::BIGINT(i64::from(i))),
            (Value::BIGINT(i), DataType::INT) => i32::try_from(i)
                .map(Value::INT)
                .map_err(|_| CoercionError::OutOfRange),
            (Value::INT(i), DataType::FLOAT) => Ok(Value::float(f64::from(i))),
            (Value::BIGINT(i), DataType::FLOAT) => Ok(Value::float(i as f64)),
            _ => Err(CoercionError::WrongType),
        }
    }

    /// The value converted to `to` without losing anything: an integer to a wider integer type or
    /// to a FLOAT that holds it exactly, or a whole FLOAT to an integer type that holds it.
    /// `None` when no such conversion exists.
    pub fn exactly_as(self, to: DataType) -> Option<Value> {
        if self.data_type() == to {
            return Some(self);
        }
        match (&self, to) {
            (Value::INT(_) | Value::BIGINT(_), DataType::INT | DataType::BIGINT) => {
                self.clone().coerce_to(to).ok()
            }
            (Value::INT(_) | Value::BIGINT(_), DataType::FLOAT) => {
                let i = self.as_i64()?;
                ((i.unsigned_abs() as f64) < I64_EXACT_FLOAT_BOUND).then(|| Value::float(i as f64))
            }
            (Value::FLOAT(f), DataType::INT | DataType::BIGINT) => {
                let f = *f;
                if f.0.fract() != 0.0 || f.0.abs() >= I64_EXACT_FLOAT_BOUND {
                    return None;
                }
                Value::BIGINT(f.0 as i64).coerce_to(to).ok()
            }
            _ => None,
        }
    }

    /// Orders two values, comparing INT, BIGINT and FLOAT numerically. `None` when the types do
    /// not compare, such as BOOLEAN with INT.
    pub fn compare(&self, other: &Value) -> Option<std::cmp::Ordering> {
        if self.data_type() == other.data_type() {
            return Some(self.cmp(other));
        }
        if let (Some(a), Some(b)) = (self.as_i64(), other.as_i64()) {
            return Some(a.cmp(&b));
        }
        let (a, b) = (self.as_f64()?, other.as_f64()?);
        Some(OrderedFloat(a).cmp(&OrderedFloat(b)))
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Value::BOOLEAN(b) => out.push(*b as u8),
            Value::INT(i) => out.extend_from_slice(&i.to_le_bytes()),
            Value::BIGINT(i) => out.extend_from_slice(&i.to_le_bytes()),
            Value::FLOAT(f) => out.extend_from_slice(&f.to_le_bytes()),
            Value::TIMESTAMP(t) => out.extend_from_slice(&t.to_le_bytes()),
            Value::TEXT(s) => out.extend_from_slice(s.as_bytes()),
        }
    }

    fn decode(bytes: &[u8], dtype: &DataType) -> Self {
        match dtype {
            DataType::BOOLEAN => Value::BOOLEAN(bytes[0] != 0),
            DataType::INT => Value::INT(i32::from_le_bytes(bytes.try_into().unwrap())),
            DataType::BIGINT => Value::BIGINT(i64::from_le_bytes(bytes.try_into().unwrap())),
            DataType::FLOAT => Value::float(f64::from_le_bytes(bytes.try_into().unwrap())),
            DataType::TIMESTAMP => Value::TIMESTAMP(i64::from_le_bytes(bytes.try_into().unwrap())),
            DataType::TEXT | DataType::VARCHAR(_) => Value::text(&String::from_utf8_lossy(bytes)),
        }
    }
}

/// Integers below this magnitude (2^53) convert to `f64` and back without rounding.
const I64_EXACT_FLOAT_BOUND: f64 = 9_007_199_254_740_992.0;

impl fmt::Display for Value {
    /// Writes the value the way the shell prints it; a TIMESTAMP prints as
    /// `YYYY-MM-DD HH:MM:SS`, with a fraction when it has microseconds.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::BOOLEAN(b) => write!(f, "{b}"),
            Value::INT(i) => write!(f, "{i}"),
            Value::BIGINT(i) => write!(f, "{i}"),
            Value::FLOAT(x) => write!(f, "{}", x.0),
            Value::TIMESTAMP(t) => f.write_str(&timestamp::format(*t)),
            Value::TEXT(s) => f.write_str(s),
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
            Value::BIGINT(i) | Value::TIMESTAMP(i) => {
                out.extend_from_slice(&((*i as u64) ^ SIGN_BIT_64).to_be_bytes())
            }
            Value::FLOAT(f) => out.extend_from_slice(&order_preserving_f64_bits(f.0).to_be_bytes()),
            Value::TEXT(s) => encode_order_preserving_bytes(s.as_bytes(), out),
        }
    }
}

/// The most bytes one character takes in UTF-8.
const MAX_UTF8_CHAR_LEN: usize = 4;
/// Bytes that end an order-preserving string encoding: a zero byte, then [`TEXT_TERMINATOR`].
const TEXT_KEY_TERMINATOR_LEN: usize = 2;

/// Byte that ends an order-preserving string encoding, after a zero byte.
const TEXT_TERMINATOR: u8 = 0x00;
/// Byte written after each zero byte of a string, so that the terminator sorts first.
const TEXT_ESCAPE: u8 = 0xFF;

/// Appends `bytes` with every zero byte followed by [`TEXT_ESCAPE`], then a zero byte and
/// [`TEXT_TERMINATOR`], so that a string sorts before every longer string it is a prefix of.
fn encode_order_preserving_bytes(bytes: &[u8], out: &mut Vec<u8>) {
    for &b in bytes {
        out.push(b);
        if b == 0 {
            out.push(TEXT_ESCAPE);
        }
    }
    out.extend_from_slice(&[0, TEXT_TERMINATOR]);
}

/// Maps an `f64` to a `u64` that orders like `OrderedFloat<f64>`: every NaN maps to one value above
/// positive infinity, and `-0.0` maps to the same value as `0.0`.
fn order_preserving_f64_bits(f: f64) -> u64 {
    let canonical = if f.is_nan() {
        f64::NAN
    } else if f == 0.0 {
        0.0
    } else {
        f
    };
    let bits = canonical.to_bits();
    if bits & SIGN_BIT_64 == 0 {
        bits | SIGN_BIT_64
    } else {
        !bits
    }
}

/// The type of a column. Every type but TEXT and VARCHAR has a fixed width (see
/// [`DataType::fixed_width`]); a text value's characters follow the fixed-width part of its
/// tuple, so tuples of a table with a text column differ in length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    BOOLEAN,
    /// Signed 32-bit integer (`INT`, `INTEGER`, `INT4`).
    INT,
    /// Signed 64-bit integer (`BIGINT`, `INT8`).
    BIGINT,
    /// 64-bit IEEE 754 float (`FLOAT`, `REAL`, `DOUBLE`, `FLOAT4`, `FLOAT8`).
    FLOAT,
    /// Microseconds since 1970-01-01 00:00:00 UTC, with no time zone.
    TIMESTAMP,
    /// A string of any length (`TEXT`, `VARCHAR`).
    TEXT,
    /// A string of at most this many characters (`VARCHAR(n)`).
    VARCHAR(u32),
}

impl DataType {
    /// Bytes the type takes in the fixed-width part of a tuple.
    fn fixed_width(&self) -> usize {
        match self {
            DataType::BOOLEAN => 1,
            DataType::INT => 4,
            DataType::BIGINT => 8,
            DataType::FLOAT => 8,
            DataType::TIMESTAMP => 8,
            DataType::TEXT | DataType::VARCHAR(_) => TEXT_DESCRIPTOR_LEN,
        }
    }

    /// Bytes an index key spends on a value of this type, padded with zeros after the value's
    /// [`Value::encode_order_preserving`] encoding, so that every key of an index has one length.
    /// A `VARCHAR(n)` takes the encoding of `n` four-byte characters; TEXT has no bound and so
    /// no width.
    pub fn key_width(&self) -> Option<usize> {
        match self {
            DataType::TEXT => None,
            DataType::VARCHAR(n) => (*n as usize)
                .checked_mul(MAX_UTF8_CHAR_LEN)
                .and_then(|bytes| bytes.checked_add(TEXT_KEY_TERMINATOR_LEN)),
            fixed => Some(fixed.fixed_width()),
        }
    }

    /// True for TEXT and VARCHAR.
    pub fn is_text(&self) -> bool {
        matches!(self, DataType::TEXT | DataType::VARCHAR(_))
    }

    /// The type of the values a column of this type holds: TEXT for a VARCHAR, the type itself
    /// otherwise.
    pub fn value_type(self) -> DataType {
        match self {
            DataType::VARCHAR(_) => DataType::TEXT,
            other => other,
        }
    }

    /// True for INT, BIGINT and FLOAT.
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataType::INT | DataType::BIGINT | DataType::FLOAT)
    }

    /// The type arithmetic on these two types yields: the wider of the two, where INT is
    /// narrower than BIGINT and BIGINT narrower than FLOAT. `None` unless both are numeric.
    pub fn common_numeric(self, other: DataType) -> Option<DataType> {
        let rank = |t: DataType| match t {
            DataType::INT => Some(0),
            DataType::BIGINT => Some(1),
            DataType::FLOAT => Some(2),
            _ => None,
        };
        let (a, b) = (rank(self)?, rank(other)?);
        Some(if a >= b { self } else { other })
    }

    /// True when a value of this type can be stored in a column of type `to` (see
    /// [`Value::coerce_to`]); a BIGINT stored as INT can still be out of range.
    pub fn coerces_to(self, to: DataType) -> bool {
        self == to
            || (self.is_text() && to.is_text())
            || matches!(
                (self, to),
                (DataType::INT, DataType::BIGINT | DataType::FLOAT)
                    | (DataType::BIGINT, DataType::INT | DataType::FLOAT)
            )
    }

    /// True when values of the two types can be compared with `=`, `<` and the like.
    pub fn compares_with(self, other: DataType) -> bool {
        self.value_type() == other.value_type() || self.common_numeric(other).is_some()
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::BOOLEAN => f.write_str("BOOLEAN"),
            DataType::INT => f.write_str("INT"),
            DataType::BIGINT => f.write_str("BIGINT"),
            DataType::FLOAT => f.write_str("FLOAT"),
            DataType::TIMESTAMP => f.write_str("TIMESTAMP"),
            DataType::TEXT => f.write_str("TEXT"),
            DataType::VARCHAR(n) => write!(f, "VARCHAR({n})"),
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
        assert_eq!(4 + 1 + 8 + 8, schema.data_len);
        assert_eq!(1, schema.bitmap_len);

        let values = vec![
            Some(Value::INT(-42)),
            Some(Value::BOOLEAN(true)),
            Some(Value::FLOAT(OrderedFloat(3.5))),
            Some(Value::TIMESTAMP(1_700_000_000_000_000)),
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
        let err = schema.encode_tuple(&bad).unwrap_err();
        assert!(
            matches!(
                &err,
                CatalogError::ColumnType { column, expected: DataType::BOOLEAN, found: DataType::INT }
                    if column == "b"
            ),
            "{err:?}"
        );
        assert_eq!(err.to_string(), "column `b` is BOOLEAN, got INT");
    }

    #[test]
    fn encode_tuple_widens_numbers_and_range_checks_narrowing() {
        let schema = Schema::new(vec![
            (DataType::INT, "i".to_string()),
            (DataType::BIGINT, "big".to_string()),
            (DataType::FLOAT, "f".to_string()),
        ]);
        let tuple = schema
            .encode_tuple(&[
                Some(Value::BIGINT(7)),
                Some(Value::INT(-3)),
                Some(Value::INT(2)),
            ])
            .unwrap();
        assert_eq!(schema.get_value(&tuple, 0).unwrap(), Some(Value::INT(7)));
        assert_eq!(
            schema.get_value(&tuple, 1).unwrap(),
            Some(Value::BIGINT(-3))
        );
        assert_eq!(
            schema.get_value(&tuple, 2).unwrap(),
            Some(Value::float(2.0))
        );

        let err = schema
            .encode_tuple(&[Some(Value::BIGINT(1 << 40)), None, None])
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "1099511627776 is out of range for INT column `i`"
        );
        assert!(matches!(
            schema.encode_tuple(&[None, Some(Value::float(1.0)), None]),
            Err(CatalogError::ColumnType { .. })
        ));
    }

    #[test]
    fn exactly_as_converts_only_without_loss() {
        assert_eq!(
            Value::INT(3).exactly_as(DataType::FLOAT),
            Some(Value::float(3.0))
        );
        assert_eq!(
            Value::float(3.0).exactly_as(DataType::INT),
            Some(Value::INT(3))
        );
        assert_eq!(Value::float(3.5).exactly_as(DataType::INT), None);
        assert_eq!(Value::float(f64::NAN).exactly_as(DataType::BIGINT), None);
        assert_eq!(Value::BIGINT(1 << 40).exactly_as(DataType::INT), None);
        assert_eq!(
            Value::BIGINT((1 << 53) + 1).exactly_as(DataType::FLOAT),
            None
        );
        assert_eq!(Value::BOOLEAN(true).exactly_as(DataType::INT), None);
    }

    #[test]
    fn values_print_like_sql_literals() {
        assert_eq!(Value::INT(-4).to_string(), "-4");
        assert_eq!(Value::BIGINT(1 << 40).to_string(), "1099511627776");
        assert_eq!(Value::float(2.5).to_string(), "2.5");
        assert_eq!(Value::BOOLEAN(false).to_string(), "false");
        assert_eq!(
            Value::TIMESTAMP(1_700_000_000_000_000).to_string(),
            "2023-11-14 22:13:20"
        );
    }

    #[test]
    fn order_preserving_encoding_sorts_like_value_ord() {
        let mut values = vec![
            Value::FLOAT(OrderedFloat(f64::NAN)),
            Value::FLOAT(OrderedFloat(f64::INFINITY)),
            Value::FLOAT(OrderedFloat(3.7)),
            Value::FLOAT(OrderedFloat(3.2)),
            Value::FLOAT(OrderedFloat(0.0)),
            Value::FLOAT(OrderedFloat(-0.0)),
            Value::FLOAT(OrderedFloat(-1.5)),
            Value::FLOAT(OrderedFloat(-3.7)),
            Value::FLOAT(OrderedFloat(f64::NEG_INFINITY)),
            Value::FLOAT(OrderedFloat(-f64::NAN)),
        ];
        values.extend([i32::MIN, -7, -1, 0, 1, 7, i32::MAX].map(Value::INT));
        values.extend([i64::MIN, -1, 0, 1, i64::MAX].map(Value::TIMESTAMP));
        values.extend([i64::MIN, -1, 0, 1, i64::MAX].map(Value::BIGINT));
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

    #[test]
    fn restore_reopens_what_snapshot_describes() {
        let (catalog, bpm, _dir) = crate::testing::catalog(32);
        let schema = Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::FLOAT, "v".to_string()),
        ]);
        let oid = catalog.create_table("t".into(), schema.clone()).unwrap();
        let tuple = schema
            .encode_tuple(&[Some(Value::INT(4)), Some(Value::FLOAT(OrderedFloat(1.0)))])
            .unwrap();
        let rid = catalog.get_table(oid).unwrap().heap.insert(&tuple).unwrap();
        catalog
            .create_index(bpm.clone(), oid, "t_v".into(), &["v"], IndexType::BPlusTree)
            .unwrap();

        let snapshot = catalog.snapshot();
        let restored = Catalog::restore(bpm, &snapshot).unwrap();
        assert_eq!(restored.snapshot(), snapshot);
        let ioid = restored.index_oid("t", "t_v").unwrap();
        assert_eq!(
            restored
                .get_index(ioid)
                .unwrap()
                .index
                .search(&[Value::FLOAT(OrderedFloat(1.0))])
                .unwrap(),
            vec![rid]
        );
        assert_eq!(restored.create_table("u".into(), schema).unwrap(), oid + 1);
    }

    #[test]
    fn restore_rejects_an_index_that_does_not_fit_its_table() {
        let (catalog, bpm, _dir) = crate::testing::catalog(32);
        let schema = Schema::new(vec![(DataType::INT, "id".to_string())]);
        let oid = catalog.create_table("t".into(), schema).unwrap();
        catalog
            .create_index(
                bpm.clone(),
                oid,
                "t_id".into(),
                &["id"],
                IndexType::BPlusTree,
            )
            .unwrap();
        let good = catalog.snapshot();

        let mut no_table = good.clone();
        no_table.indexes[0].table_oid = 9;
        let mut bad_column = good.clone();
        bad_column.indexes[0].key_columns = vec![3];
        let mut not_a_tree = good.clone();
        not_a_tree.indexes[0].header_page_id = good.tables[0].first_page_id;
        let mut not_a_heap = good;
        not_a_heap.tables[0].first_page_id = not_a_heap.indexes[0].header_page_id;

        for (snapshot, expected) in [
            (
                no_table,
                "index t_id is on table oid 9, which does not exist",
            ),
            (
                bad_column,
                "index t_id keys on column 3, past the end of table t",
            ),
            (not_a_tree, "index t_id: "),
        ] {
            match Catalog::restore(bpm.clone(), &snapshot) {
                Err(CatalogError::IndexOpenError(msg)) => {
                    assert!(msg.starts_with(expected), "{msg}")
                }
                other => panic!("expected an index error, got {:?}", other.err()),
            }
        }
        let err = Catalog::restore(bpm, &not_a_heap).err().unwrap();
        assert!(
            err.to_string()
                .starts_with("opening a table failed: table t: "),
            "{err}"
        );
    }
}
