use std::fmt;

#[derive(Debug)]
pub struct Schema {
    pub cols: Vec<Column>,
    pub data_len: usize,
}

impl Schema {
    pub fn new(dtypes: Vec<DataType>) -> Self {
        let mut offset: usize = 0;
        let cols: Vec<Column> = dtypes
            .into_iter()
            .map(|dtype| {
                let inlined = dtype.clone().inline();
                let col = Column {
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    dtype: DataType,
    offset: usize, // offset for this column from the beginning of a tuple
    inlined: bool, // whether the data is inlined (true for fixed-width data, false for variable-length)
}

/// Metadata for a table schema
#[derive(Debug, Clone)]
pub enum DataType {
    BOOLEAN,
    INT, // signed, 32 bit
    // FLOAT, // 32 bit
    TIMESTAMP, // assume UTC

               // not inline
               // ENUM,
               // VARCHAR(usize),
               // VECTOR(usize),
}

pub enum Value {
    BOOLEAN(bool),
    INT(i32),
    // FLOAT(f32),
    TIMESTAMP(i64),
    // ENUM // assume static, i.e. can't add a member after creating
    // VARCHAR
    // VECTOR
}

impl Value {
    fn data_type(&self) -> DataType {
        match self {
            Value::BOOLEAN(_) => DataType::BOOLEAN,
            Value::INT(_) => DataType::INT,
            // Value::FLOAT(_) => DataType::FLOAT,
            Value::TIMESTAMP(_) => DataType::TIMESTAMP,
        }
    }
}

impl DataType {
    fn encode_from_tuple(&self, tuple_data: &[u8], offset: usize, out: &mut Vec<u8>) {
        match self {
            DataType::BOOLEAN => {
                let b = tuple_data[offset];
                out.push(b);
            }
            DataType::INT => {
                // assumes data is stored little-endian
                let val = i32::from_le_bytes(tuple_data[offset..offset + 4].try_into().unwrap());
                let big_e_converted = ((val as u32) ^ 0x8000_0000).to_be_bytes();
                out.extend_from_slice(&big_e_converted);
            }
            // fuck it don't support FLOAT
            // i have a feeling it will cause headaches
            // DataType::FLOAT => {
            //     let val = i32::from_le_bytes(tuple_data[offset..offset+4].try_into().unwrap());

            // },
            DataType::TIMESTAMP => {
                let val = i64::from_le_bytes(tuple_data[offset..offset + 8].try_into().unwrap());
                let big_e_converted = ((val as u64) ^ 0x8000_0000_0000_0000).to_be_bytes();
                out.extend_from_slice(&big_e_converted);
            }
        }
    }

    fn size(&self) -> usize {
        match self {
            DataType::BOOLEAN => 1,
            DataType::INT => 4,
            DataType::TIMESTAMP => 8,
        }
    }

    fn inline(&self) -> bool {
        match self {
            DataType::BOOLEAN => true,
            DataType::INT => true,
            DataType::TIMESTAMP => true,
        }
    }
}

#[derive(Debug)]
pub enum CatalogError {
    TupleSchemaMismatch,
}

#[derive(Debug)]
pub struct Tuple<'a> {
    pub record_id: RecordId,
    pub data: &'a [u8],
}

impl<'a> Tuple<'a> {
    pub fn new(schema: &Schema, record_id: RecordId, data: &'a [u8]) -> Result<Self, CatalogError> {
        if data.len() != schema.data_len {
            return Err(CatalogError::TupleSchemaMismatch);
        }
        Ok(Self { record_id, data })
    }

    pub fn validate(&self, schema: &Schema) -> Result<(), CatalogError> {
        if self.data.len() != schema.data_len {
            return Err(CatalogError::TupleSchemaMismatch);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct RecordId {
    pub page_id: u32,
    pub slot_id: u32,
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "page_id: {}, slot_id: {}", self.page_id, self.slot_id)
    }
}

#[derive(Debug, PartialEq, PartialOrd, Ord, Eq, Clone)]
pub struct IndexKey(pub Vec<u8>);

#[derive(Debug)]
pub enum IndexValue {
    IndexValue(RecordId),
    PageId,
}

#[derive(Debug)]
pub struct IndexDefinition {
    indexed_cols: Vec<Column>, // offsets store tuple offsets
    pub schema: Schema,        // schema of the index itself, not the tuple
}

impl IndexDefinition {
    pub fn new(cols: Vec<Column>) -> Self {
        let dtypes = cols.iter().map(|c| c.dtype.clone()).collect();
        Self {
            indexed_cols: cols,
            schema: Schema::new(dtypes),
        }
    }

    pub fn encode_key(&self, tuple_bytes: &[u8]) -> IndexKey {
        let mut data = Vec::with_capacity(self.schema.data_len);
        for c in self.indexed_cols.iter() {
            c.dtype.encode_from_tuple(tuple_bytes, c.offset, &mut data);
        }
        IndexKey(data)
    }
}

#[test]
fn test_index_encoder() {
    // 1. create a schema
    let schema = Schema::new(vec![
        DataType::INT,
        DataType::BOOLEAN,
        DataType::INT,
        DataType::INT,
    ]);
    // each tuple in this schema should be 4 + 1 + 4 + 4 bytes
    assert_eq!(13, schema.data_len);

    // 2. create a tuple that abides to this schema
    let i1 = 12i32.to_le_bytes();
    let i2 = 13i32.to_le_bytes();
    let i3 = 14i32.to_le_bytes();
    let b1: u8 = 1;
    let mut buf = Vec::new();
    buf.extend_from_slice(&i1);
    buf.extend_from_slice(&i2);
    buf.extend_from_slice(&i3);
    buf.push(b1);

    let t1 = Tuple::new(
        &schema,
        RecordId {
            page_id: 1,
            slot_id: 2,
        },
        buf.as_slice(),
    )
    .unwrap();

    // 3. create an index
    let index_cols = vec![
        schema.cols.get(0).unwrap().clone(),
        schema.cols.get(2).unwrap().clone(),
    ];
    // we expect each index key to be: 4 bytes + 4 bytes long
    let index = IndexDefinition::new(index_cols);
    assert_eq!(8, index.schema.data_len);

    // // 3. create index keys
    let tuples = vec![t1];
    for t in tuples.iter() {
        let index_key = index.encode_key(t.data).0;
        assert_eq!(8, index_key.len());
    }
}
