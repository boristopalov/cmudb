pub mod b_plus_page;
pub mod b_plus_tree;
pub mod context;
mod page_codec;
use std::fmt;

use crate::buffer_pool::BpmError;
use crate::catalog::index_schema::{IndexKey, IndexValue};
use crate::index::page_codec::PageCodecError;
pub use b_plus_tree::*;
pub type InsertResult = Result<(), IndexError>;
pub type RemoveResult = Result<(), IndexError>;

pub trait Index {
    fn insert(&self, key: IndexKey, val: IndexValue) -> InsertResult;
    fn remove(&self, key: IndexKey) -> RemoveResult;
    fn search(&self, key: &IndexKey) -> Result<Option<IndexValue>, IndexError>;
    // fn scan(...) // TODO: not sure that this belongs here
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
