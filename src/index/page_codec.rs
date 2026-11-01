use crate::buf::{BufReader, BufWriter, BufferError};
use crate::index::b_plus_page::{INVALID_PAGE_ID, PageId};
use crate::table_heap::RecordId;
use std::fmt;

const MAGIC: [u8; 4] = *b"CMDB";
const VERSION: u16 = 2;

pub const KIND_HEADER: u8 = 0;
pub const KIND_INTERNAL: u8 = 1;
pub const KIND_LEAF: u8 = 2;

pub const U32_BYTES: usize = 4;
pub const U64_BYTES: usize = 8;

const VERSION_BYTES: usize = size_of::<u16>();
const KIND_BYTES: usize = size_of::<u8>();
const RESERVED_BYTES: usize = size_of::<u8>();

pub const KIND_OFFSET: usize = MAGIC.len() + VERSION_BYTES;
pub const COMMON_HEADER_BYTES: usize = KIND_OFFSET + KIND_BYTES + RESERVED_BYTES;

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

pub fn encode_record_id(rid: &RecordId) -> Result<(u64, u64), PageCodecError> {
    Ok((u64::from(rid.page_id), u64::from(rid.slot_id)))
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
    w.write_bytes(&MAGIC)?;
    w.write_u16_le(VERSION)?;
    w.write_u8(kind)?;
    w.write_u8(0)?;
    Ok(())
}

pub fn read_common_header(r: &mut BufReader<'_>) -> Result<u8, PageCodecError> {
    if r.read_bytes(MAGIC.len())? != MAGIC {
        return Err(PageCodecError::BadMagic);
    }
    let version = r.read_u16_le()?;
    if version != VERSION {
        return Err(PageCodecError::UnsupportedVersion(version));
    }
    let kind = r.read_u8()?;
    let _reserved = r.read_u8()?;
    Ok(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::PAGE_SIZE;

    #[test]
    fn common_header_occupies_common_header_bytes() {
        let mut buf = vec![0u8; PAGE_SIZE];
        let mut w = BufWriter::new(&mut buf).unwrap();
        write_common_header(&mut w, KIND_INTERNAL).unwrap();
        assert_eq!(w.pos(), COMMON_HEADER_BYTES);
        assert_eq!(COMMON_HEADER_BYTES, 8);
        assert_eq!(buf[KIND_OFFSET], KIND_INTERNAL);
        let mut r = BufReader::new(&buf).unwrap();
        read_common_header(&mut r).unwrap();
        assert_eq!(r.remaining(), PAGE_SIZE - COMMON_HEADER_BYTES);
    }

    #[test]
    fn common_header_round_trips() {
        let mut buf = vec![0u8; PAGE_SIZE];
        write_common_header(&mut BufWriter::new(&mut buf).unwrap(), KIND_LEAF).unwrap();
        assert_eq!(&buf[..4], b"CMDB");
        assert_eq!(buf[KIND_OFFSET], KIND_LEAF);
        let kind = read_common_header(&mut BufReader::new(&buf).unwrap()).unwrap();
        assert_eq!(kind, KIND_LEAF);
    }

    #[test]
    fn common_header_rejects_bad_magic_version_and_short_buffers() {
        let mut buf = vec![0u8; PAGE_SIZE];
        write_common_header(&mut BufWriter::new(&mut buf).unwrap(), KIND_HEADER).unwrap();

        let mut bad_magic = buf.clone();
        bad_magic[0] = b'X';
        assert_eq!(
            read_common_header(&mut BufReader::new(&bad_magic).unwrap()),
            Err(PageCodecError::BadMagic)
        );

        let mut bad_version = buf.clone();
        bad_version[4] = 9;
        assert_eq!(
            read_common_header(&mut BufReader::new(&bad_version).unwrap()),
            Err(PageCodecError::UnsupportedVersion(9))
        );

        let mut tiny = [0u8; 3];
        assert_eq!(
            BufWriter::new(&mut tiny).err(),
            Some(BufferError::BufferTooSmall {
                needed: PAGE_SIZE,
                actual: 3
            })
        );
        assert!(BufReader::new(&tiny).is_err());
    }

    #[test]
    fn page_and_record_ids_round_trip() {
        assert_eq!(encode_page_id(INVALID_PAGE_ID).unwrap(), u64::MAX);
        assert_eq!(decode_page_id(u64::MAX).unwrap(), INVALID_PAGE_ID);
        assert_eq!(decode_page_id(encode_page_id(17).unwrap()).unwrap(), 17);

        let rid = RecordId {
            page_id: 4,
            slot_id: 9,
        };
        let (p, s) = encode_record_id(&rid).unwrap();
        assert_eq!(decode_record_id(p, s).unwrap(), rid);
        assert!(decode_record_id(u64::from(u32::MAX) + 1, 0).is_err());
        assert!(decode_record_id(0, u64::from(u32::MAX) + 1).is_err());
    }

    #[test]
    fn errors_display_readably() {
        let cases = [
            (PageCodecError::BadMagic, "bad page magic"),
            (
                PageCodecError::UnsupportedVersion(3),
                "unsupported page version 3",
            ),
            (
                PageCodecError::WrongPageKind {
                    expected: 1,
                    actual: 2,
                },
                "wrong page kind (expected 1, got 2)",
            ),
            (PageCodecError::InvalidPageKind(7), "invalid page kind: 7"),
            (
                PageCodecError::InvalidKeyLen {
                    expected: 8,
                    actual: 4,
                },
                "invalid key length (expected 8, got 4)",
            ),
            (PageCodecError::InvalidSize, "invalid page size fields"),
            (PageCodecError::Malformed("x"), "malformed page: x"),
            (
                PageCodecError::BufferError(BufferError::Overflow),
                "numeric overflow decoding page",
            ),
            (
                BufferError::BufferTooSmall {
                    needed: 8,
                    actual: 2,
                }
                .into(),
                "buffer too small (needed 8 bytes, got 2 bytes)",
            ),
        ];
        for (err, msg) in cases {
            assert_eq!(err.to_string(), msg);
        }
    }
}
