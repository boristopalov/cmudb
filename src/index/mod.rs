pub mod b_plus_page;
pub mod b_plus_tree;
pub mod context;
mod page_codec;
use std::fmt;
use std::ops::Bound;

use crate::buffer_pool::BpmError;
use crate::catalog::index_schema::{IndexKey, IndexValue};
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

// TODO: clean these up
#[derive(Debug)]
pub enum IndexError {
    Insert(InsertError),
    Remove(RemoveError),
    Lookup,
    BpmError(BpmError),
    PageError(PageCodecError),
    LatchError,
}

#[derive(Debug)]
pub enum InsertError {
    DuplicateKey,
    InvalidKey,
    InvalidValue,
    BpmError,
    BadRoot,
    GenericError(String),
}

#[derive(Debug)]
pub enum RemoveError {
    KeyNotFound,
    GenericError(String),
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
            IndexError::BpmError(e) => write!(f, "BPM error: {e}"),
            IndexError::PageError(e) => write!(f, "page decode/encode error: {e}"),
            IndexError::LatchError => write!(f, "error reading or acquiring latch"),
            IndexError::Lookup => write!(f, "lookup error"),
        }
    }
}

impl fmt::Display for InsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InsertError::DuplicateKey => write!(f, "key already exists"),
            InsertError::InvalidKey => write!(f, "invalid key"),
            InsertError::InvalidValue => write!(f, "invalid value"),
            InsertError::BpmError => write!(f, "bpm error"),
            InsertError::BadRoot => write!(f, "error getting root while inserting"),
            InsertError::GenericError(msg) => write!(f, "error while inserting: {msg}"),
        }
    }
}

impl fmt::Display for RemoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RemoveError::KeyNotFound => write!(f, "key not found"),
            RemoveError::GenericError(msg) => write!(f, "error while removing: {msg}"),
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
        assert_eq!(IndexError::Lookup.to_string(), "lookup error");
        assert_eq!(
            IndexError::LatchError.to_string(),
            "error reading or acquiring latch"
        );
        let inserts = [
            (InsertError::DuplicateKey, "key already exists"),
            (InsertError::InvalidKey, "invalid key"),
            (InsertError::InvalidValue, "invalid value"),
            (InsertError::BpmError, "bpm error"),
            (InsertError::BadRoot, "error getting root while inserting"),
            (
                InsertError::GenericError("boom".into()),
                "error while inserting: boom",
            ),
        ];
        for (err, msg) in inserts {
            assert_eq!(IndexError::Insert(err).to_string(), format!("insert failed: {msg}"));
        }
        assert_eq!(
            RemoveError::GenericError("x".into()).to_string(),
            "error while removing: x"
        );
    }
}
