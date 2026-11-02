pub mod b_plus_page;
pub mod b_plus_tree;
pub mod context;
mod page_codec;
pub mod page_view;
use std::fmt;
use std::ops::Bound;

use crate::buffer_pool::BpmError;
use crate::catalog::index_schema::{IndexKey, IndexValue};
use crate::catalog::{CatalogError, DataType};
use crate::index::page_codec::PageCodecError;
pub use b_plus_tree::*;
pub type InsertResult = Result<(), IndexError>;
pub type RemoveResult = Result<(), IndexError>;

pub type IndexIter = Box<dyn Iterator<Item = Result<(IndexKey, IndexValue), IndexError>>>;
pub trait Index {
    fn insert(&self, key: IndexKey, val: IndexValue) -> InsertResult;
    fn remove(&self, key: IndexKey) -> RemoveResult;
    fn search(&self, key: &IndexKey) -> Result<Option<IndexValue>, IndexError>;
    fn scan(&self, range: (Bound<IndexKey>, Bound<IndexKey>)) -> Result<IndexIter, IndexError>;
}

#[derive(Debug)]
pub enum IndexError {
    Insert(InsertError),
    Remove(RemoveError),
    KeyLength {
        expected: usize,
        actual: usize,
    },
    UnsupportedKeyLength(u32),
    BpmError(BpmError),
    PageError(PageCodecError),
    Structure(&'static str),
    /// A key column could not be read from a table tuple.
    Catalog(CatalogError),
    /// A key column has a type with no fixed key width, such as TEXT.
    UnindexableColumn {
        column: String,
        dtype: DataType,
    },
}

#[derive(Debug)]
pub enum InsertError {
    DuplicateKey,
}

#[derive(Debug)]
pub enum RemoveError {
    KeyNotFound,
}

impl From<BpmError> for IndexError {
    fn from(err: BpmError) -> Self {
        IndexError::BpmError(err)
    }
}

impl From<PageCodecError> for IndexError {
    fn from(err: PageCodecError) -> Self {
        IndexError::PageError(err)
    }
}

impl From<CatalogError> for IndexError {
    fn from(err: CatalogError) -> Self {
        IndexError::Catalog(err)
    }
}

impl From<RemoveError> for IndexError {
    fn from(err: RemoveError) -> Self {
        IndexError::Remove(err)
    }
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::Insert(e) => write!(f, "insert failed: {e}"),
            IndexError::Remove(e) => write!(f, "remove failed: {e}"),
            IndexError::KeyLength { expected, actual } => {
                write!(f, "key is {actual} bytes but the index expects {expected}")
            }
            IndexError::UnsupportedKeyLength(len) => {
                write!(f, "an index key of {len} bytes cannot fit in a page")
            }
            IndexError::BpmError(e) => write!(f, "BPM error: {e}"),
            IndexError::PageError(e) => write!(f, "page decode/encode error: {e}"),
            IndexError::Structure(msg) => write!(f, "b+ tree structure violated: {msg}"),
            IndexError::Catalog(e) => write!(f, "cannot read the key from the tuple: {e}"),
            IndexError::UnindexableColumn { column, dtype } => write!(
                f,
                "column `{column}` is {dtype}, which cannot be an index key; declare it VARCHAR(n) to index it"
            ),
        }
    }
}

impl fmt::Display for InsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InsertError::DuplicateKey => write!(f, "key already exists"),
        }
    }
}

impl fmt::Display for RemoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RemoveError::KeyNotFound => write!(f, "key not found"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_convert_and_display() {
        let e: IndexError = RemoveError::KeyNotFound.into();
        assert_eq!(e.to_string(), "remove failed: key not found");
        let e: IndexError = PageCodecError::BadMagic.into();
        assert_eq!(e.to_string(), "page decode/encode error: bad page magic");
        let e: IndexError = BpmError::NoFreeFrame.into();
        assert!(e.to_string().starts_with("BPM error"));
        let e: IndexError = CatalogError::TupleSchemaMismatch.into();
        assert_eq!(
            e.to_string(),
            "cannot read the key from the tuple: the tuple does not match the table's schema"
        );
        let cases = [
            (
                IndexError::Insert(InsertError::DuplicateKey),
                "insert failed: key already exists",
            ),
            (
                IndexError::KeyLength {
                    expected: 8,
                    actual: 3,
                },
                "key is 3 bytes but the index expects 8",
            ),
            (
                IndexError::UnsupportedKeyLength(0),
                "an index key of 0 bytes cannot fit in a page",
            ),
            (IndexError::Structure("x"), "b+ tree structure violated: x"),
        ];
        for (err, msg) in cases {
            assert_eq!(err.to_string(), msg);
        }
    }
}
