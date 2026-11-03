use crate::buf::{BufReader, BufWriter, BufferError};
use crate::ids::{PageId, RecordId};
use std::fmt;
use std::num::TryFromIntError;

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

/// A size or id that does not fit the type it is encoded or decoded as.
pub const OVERFLOW: PageCodecError = PageCodecError::BufferError(BufferError::Overflow);

impl From<TryFromIntError> for PageCodecError {
    fn from(_: TryFromIntError) -> Self {
        OVERFLOW
    }
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

pub fn encode_page_id(page_id: Option<PageId>) -> u64 {
    PageId::encode_u64(page_id)
}

pub fn decode_page_id(raw: u64) -> Result<Option<PageId>, PageCodecError> {
    PageId::decode_u64(raw).map_err(|_| PageCodecError::BufferError(BufferError::Overflow))
}

/// Decodes the id of an internal page's child, which names a page. Fails with
/// [`PageCodecError::Malformed`] for the "no page" value.
pub fn decode_child_page_id(raw: u64) -> Result<PageId, PageCodecError> {
    decode_page_id(raw)?.ok_or(PageCodecError::Malformed(
        "an internal page child names no page",
    ))
}

pub fn encode_record_id(rid: &RecordId) -> Result<(u64, u64), PageCodecError> {
    Ok((u64::from(rid.page_id.get()), u64::from(rid.slot_id)))
}

pub fn decode_record_id(page_id: u64, slot_id: u64) -> Result<RecordId, PageCodecError> {
    Ok(RecordId {
        page_id: PageId::try_from(page_id)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
        slot_id: u32::try_from(slot_id)?,
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

/// Reads the common header and fails with `WrongPageKind` unless the page is of `expected` kind.
pub fn read_common_header_of_kind(
    r: &mut BufReader<'_>,
    expected: u8,
) -> Result<(), PageCodecError> {
    let actual = read_common_header(r)?;
    if actual != expected {
        return Err(PageCodecError::WrongPageKind { expected, actual });
    }
    Ok(())
}

/// Validates the common header in place and returns the page kind, without a cursor.
pub fn peek_common_header(buf: &[u8]) -> Result<u8, PageCodecError> {
    if buf.len() < COMMON_HEADER_BYTES {
        return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
            needed: COMMON_HEADER_BYTES,
            actual: buf.len(),
        }));
    }
    if buf[0..4] != MAGIC {
        return Err(PageCodecError::BadMagic);
    }
    let version = u16::from_le_bytes([buf[4], buf[5]]);
    if version != VERSION {
        return Err(PageCodecError::UnsupportedVersion(version));
    }
    Ok(buf[KIND_OFFSET])
}

/// Like `peek_common_header`, but also checks the kind.
pub fn expect_common_header(buf: &[u8], expected: u8) -> Result<(), PageCodecError> {
    let actual = peek_common_header(buf)?;
    if actual != expected {
        return Err(PageCodecError::WrongPageKind { expected, actual });
    }
    Ok(())
}

/// Writes the common header at the start of `buf`.
pub fn put_common_header(buf: &mut [u8], kind: u8) {
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4..6].copy_from_slice(&VERSION.to_le_bytes());
    buf[KIND_OFFSET] = kind;
    buf[7] = 0;
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
        assert_eq!(encode_page_id(None), u64::MAX);
        assert_eq!(decode_page_id(u64::MAX).unwrap(), None);
        let page = Some(PageId::new(17));
        assert_eq!(decode_page_id(encode_page_id(page)).unwrap(), page);
        assert_eq!(decode_child_page_id(17).unwrap(), PageId::new(17));
        assert!(matches!(
            decode_child_page_id(u64::MAX),
            Err(PageCodecError::Malformed(_))
        ));

        let rid = RecordId {
            page_id: PageId::new(4),
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

    #[test]
    fn page_ids_decode_to_a_page_or_none_and_reject_the_rest() {
        let page = PageId::new(9);
        assert_eq!(decode_page_id(encode_page_id(Some(page))), Ok(Some(page)));
        assert_eq!(decode_page_id(encode_page_id(None)), Ok(None));
        assert_eq!(
            decode_page_id(u64::from(u32::MAX)),
            Err(PageCodecError::BufferError(BufferError::Overflow))
        );
        assert_eq!(decode_child_page_id(u64::from(page.get())), Ok(page));
        assert_eq!(
            decode_child_page_id(encode_page_id(None)),
            Err(PageCodecError::Malformed(
                "an internal page child names no page"
            ))
        );
    }
}
