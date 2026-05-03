use crate::buf::{BufReader, BufWriter, BufferError};
use crate::catalog::RecordId;
use crate::index::b_plus_page::{INVALID_PAGE_ID, PageId};
use std::fmt;

const MAGIC: [u8; 4] = *b"CMDB";
const VERSION: u16 = 2;

pub const KIND_HEADER: u8 = 0;
pub const KIND_INTERNAL: u8 = 1;
pub const KIND_LEAF: u8 = 2;

pub const KIND_OFFSET: usize = 6;

pub const U32_BYTES: usize = 4;
pub const U64_BYTES: usize = 8;
pub const COMMON_HEADER_BYTES: usize = 8; // MAGIC(4) + VERSION(2) + KIND(1) + RESERVED(1)

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageCodecError {
    BadMagic,
    UnsupportedVersion(u16),
    WrongPageKind { expected: u8, actual: u8 },
    InvalidPageKind(u8),
    InvalidKeyLen { expected: usize, actual: usize },
    InvalidSize,
    Malformed(&'static str),
    BufferError(BufferError),
}

impl From<BufferError> for PageCodecError {
    fn from(err: BufferError) -> Self {
        PageCodecError::BufferError(err)
    }
}

impl fmt::Display for PageCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PageCodecError::BufferError(BufferError::BufferTooSmall { needed, actual }) => write!(
                f,
                "buffer too small (needed {needed} bytes, got {actual} bytes)"
            ),
            PageCodecError::BadMagic => write!(f, "bad page magic"),
            PageCodecError::InvalidPageKind(found_kind) => {
                write!(f, "invalid page kind: {found_kind}")
            }
            PageCodecError::UnsupportedVersion(v) => write!(f, "unsupported page version {v}"),
            PageCodecError::WrongPageKind { expected, actual } => {
                write!(f, "wrong page kind (expected {expected}, got {actual})")
            }
            PageCodecError::InvalidKeyLen { expected, actual } => {
                write!(f, "invalid key length (expected {expected}, got {actual})")
            }
            PageCodecError::InvalidSize => write!(f, "invalid page size fields"),
            PageCodecError::BufferError(BufferError::Overflow) => {
                write!(f, "numeric overflow decoding page")
            }
            PageCodecError::Malformed(msg) => write!(f, "malformed page: {msg}"),
        }
    }
}

impl std::error::Error for PageCodecError {}

pub fn encode_page_id(page_id: PageId) -> Result<u64, PageCodecError> {
    if page_id == INVALID_PAGE_ID {
        return Ok(u64::MAX);
    }
    u64::try_from(page_id).map_err(|_| PageCodecError::BufferError(BufferError::Overflow))
}

pub fn decode_page_id(raw: u64) -> Result<PageId, PageCodecError> {
    if raw == u64::MAX {
        return Ok(INVALID_PAGE_ID);
    }
    usize::try_from(raw).map_err(|_| PageCodecError::BufferError(BufferError::Overflow))
}

/// returns a Result tuple consisting of the size of the page, max size, and whether the page is a leaf page
// pub fn peek_size_and_kind(r: &mut BufReader<'_>) -> Result<u32, u32, bool> {

// }

pub fn encode_record_id(rid: &RecordId) -> Result<(u64, u64), PageCodecError> {
    Ok((
        u64::try_from(rid.page_id)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
        u64::try_from(rid.slot_id)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
    ))
}

pub fn decode_record_id(page_id: u64, slot_id: u64) -> Result<RecordId, PageCodecError> {
    Ok(RecordId {
        page_id: u32::try_from(page_id)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
        slot_id: u32::try_from(slot_id)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
    })
}

pub fn write_common_header(w: &mut BufWriter<'_>, kind: u8) -> Result<(), PageCodecError> {
    w.write_bytes(&MAGIC)
        .map_err(|b| PageCodecError::BufferError(b))?;
    w.write_u16_le(VERSION)
        .map_err(|b| PageCodecError::BufferError(b))?;
    w.write_u8(kind)
        .map_err(|b| PageCodecError::BufferError(b))?;
    w.write_u8(0).map_err(|b| PageCodecError::BufferError(b))?;
    Ok(())
}

pub fn read_common_header(r: &mut BufReader<'_>) -> Result<u8, PageCodecError> {
    let magic = r
        .read_bytes(4)
        .map_err(|b| PageCodecError::BufferError(b))?;
    if magic != MAGIC {
        return Err(PageCodecError::BadMagic);
    }
    let version = r
        .read_u16_le()
        .map_err(|b| PageCodecError::BufferError(b))?;
    // dont need this...
    if version != VERSION {
        return Err(PageCodecError::UnsupportedVersion(version));
    }
    let kind = r.read_u8().map_err(|b| PageCodecError::BufferError(b))?;
    let _reserved = r.read_u8().map_err(|b| PageCodecError::BufferError(b))?;
    Ok(kind)
}
