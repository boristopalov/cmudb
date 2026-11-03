//! The on-disk layout of a database's metadata: which page ids are in use, the transaction
//! clocks and the catalog, so that [`crate::Database::open`] can pick a database up where the
//! last run left it.
//!
//! Page [`HEADER_PAGE_ID`] of the data file is the header page. It names the first page of
//! the metadata chain, a linked list of pages whose payloads, concatenated, are one encoded
//! [`DatabaseMeta`]. Every other page belongs to a table heap or an index, as before.
//!
//! A column type is stored as a one-byte tag followed by that type's own parameters, such as
//! the character limit of a `VARCHAR(n)` or `CHAR(n)`, so a new type adds a tag (and the
//! parameter bytes it needs) without changing how the existing ones are read.

use crate::buffer_pool::{BpmError, BufferPoolManager, PageAllocation};
use crate::catalog::{CatalogError, CatalogSnapshot, DataType, IndexEntry, IndexType, TableEntry};
use crate::concurrency::{Timestamp, TxnId};
use crate::disk::PAGE_SIZE;
use std::collections::HashSet;
use std::fmt;
use std::io;
use std::sync::Arc;

/// The page that holds the [`DatabaseHeader`]. It is the first page a new database allocates.
pub const HEADER_PAGE_ID: usize = 0;

/// The first bytes of the header page of every cmudb data file.
const HEADER_MAGIC: [u8; 8] = *b"CMUDB\0db";

/// The version of this layout and of the table and index pages the data file holds. A data file
/// written with another version is refused. Version 2 gave each table page slot a capacity, so
/// rows can change length.
pub const FORMAT_VERSION: u32 = 2;

const HEADER_VERSION_OFF: usize = 8;
const HEADER_META_PAGE_OFF: usize = 12;

/// The first bytes of every metadata chain page.
const CHAIN_MAGIC: [u8; 4] = *b"CMMC";
const CHAIN_NEXT_OFF: usize = 4;
const CHAIN_LEN_OFF: usize = 12;
const CHAIN_PAYLOAD_OFF: usize = 16;

/// Payload bytes one metadata chain page holds.
pub const CHAIN_PAGE_CAPACITY: usize = PAGE_SIZE - CHAIN_PAYLOAD_OFF;

/// The `next` link of the last page of the metadata chain.
const NO_NEXT_CHAIN_PAGE: u64 = u64::MAX;

const TYPE_TAG_BOOLEAN: u8 = 0;
const TYPE_TAG_INT: u8 = 1;
const TYPE_TAG_FLOAT: u8 = 2;
const TYPE_TAG_TIMESTAMP: u8 = 3;
const TYPE_TAG_BIGINT: u8 = 4;
const TYPE_TAG_TEXT: u8 = 5;
/// Followed by the type's character limit, a `u32`.
const TYPE_TAG_VARCHAR: u8 = 6;
/// Followed by the type's length in characters, a `u32`.
const TYPE_TAG_CHAR: u8 = 7;

const INDEX_TAG_BPLUS_TREE: u8 = 0;

/// Why opening or writing a database's metadata failed.
#[derive(Debug)]
pub enum PersistError {
    Io(io::Error),
    Bpm(BpmError),
    /// The data file's header page is not a cmudb header.
    NotADatabase,
    /// Another open database holds the lock on the data file.
    InUse,
    /// The data file was written with another [`FORMAT_VERSION`].
    UnsupportedFormat(u32),
    /// The metadata chain or its payload is damaged.
    CorruptMeta(&'static str),
    /// A column type tag this version does not know.
    UnknownColumnType(u8),
    /// An index type tag this version does not know.
    UnknownIndexType(u8),
    /// The catalog the metadata describes could not be opened.
    Catalog(CatalogError),
    /// Rolling back the transaction `BEGIN` left open failed, so nothing was written.
    AbortOpenTransaction(String),
}

impl fmt::Display for PersistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistError::Io(e) => write!(f, "io error: {e}"),
            PersistError::Bpm(e) => write!(f, "buffer pool error: {e}"),
            PersistError::NotADatabase => f.write_str("the data file is not a cmudb database"),
            PersistError::InUse => f.write_str("the database is open in another process"),
            PersistError::UnsupportedFormat(v) => write!(
                f,
                "the data file has format version {v}, but this cmudb reads version {FORMAT_VERSION}"
            ),
            PersistError::CorruptMeta(what) => {
                write!(f, "the database metadata is damaged: {what}")
            }
            PersistError::UnknownColumnType(tag) => write!(f, "unknown column type tag {tag}"),
            PersistError::UnknownIndexType(tag) => write!(f, "unknown index type tag {tag}"),
            PersistError::Catalog(e) => write!(f, "catalog: {e}"),
            PersistError::AbortOpenTransaction(e) => {
                write!(f, "rolling back the open transaction failed: {e}")
            }
        }
    }
}

impl std::error::Error for PersistError {}

impl From<io::Error> for PersistError {
    fn from(e: io::Error) -> Self {
        PersistError::Io(e)
    }
}

impl From<BpmError> for PersistError {
    fn from(e: BpmError) -> Self {
        PersistError::Bpm(e)
    }
}

impl From<CatalogError> for PersistError {
    fn from(e: CatalogError) -> Self {
        PersistError::Catalog(e)
    }
}

/// Everything about a database that lives outside its table and index pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatabaseMeta {
    pub allocation: PageAllocation,
    pub last_commit_ts: Timestamp,
    pub next_txn_id: TxnId,
    pub catalog: CatalogSnapshot,
}

impl DatabaseMeta {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::default();
        w.u64(self.allocation.next_page_id as u64);
        w.u64(self.allocation.free_page_ids.len() as u64);
        for &id in &self.allocation.free_page_ids {
            w.u64(id as u64);
        }
        w.u64(self.last_commit_ts);
        w.u64(self.next_txn_id);
        let c = &self.catalog;
        w.u32(c.next_table_oid);
        w.u32(c.next_index_oid);
        w.u32(c.tables.len() as u32);
        for t in &c.tables {
            w.u32(t.oid);
            w.str(&t.name);
            w.u64(t.first_page_id as u64);
            w.u32(t.columns.len() as u32);
            for (dtype, name) in &t.columns {
                w.str(name);
                encode_type(&mut w, *dtype);
            }
        }
        w.u32(c.indexes.len() as u32);
        for i in &c.indexes {
            w.u32(i.oid);
            w.str(&i.name);
            w.u32(i.table_oid);
            w.u32(i.key_columns.len() as u32);
            for &col in &i.key_columns {
                w.u32(col as u32);
            }
            w.u8(match i.index_type {
                IndexType::BPlusTree => INDEX_TAG_BPLUS_TREE,
            });
            w.u64(i.header_page_id as u64);
        }
        w.0
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, PersistError> {
        let mut r = Reader(bytes);
        let next_page_id = r.usize()?;
        let free_count = r.u64()?;
        let free_page_ids = (0..free_count)
            .map(|_| r.usize())
            .collect::<Result<_, _>>()?;
        let last_commit_ts = r.u64()?;
        let next_txn_id = r.u64()?;
        let next_table_oid = r.u32()?;
        let next_index_oid = r.u32()?;
        let tables = (0..r.u32()?)
            .map(|_| {
                Ok(TableEntry {
                    oid: r.u32()?,
                    name: r.str()?,
                    first_page_id: r.usize()?,
                    columns: (0..r.u32()?)
                        .map(|_| {
                            let name = r.str()?;
                            Ok((decode_type(&mut r)?, name))
                        })
                        .collect::<Result<_, PersistError>>()?,
                })
            })
            .collect::<Result<_, PersistError>>()?;
        let indexes = (0..r.u32()?)
            .map(|_| {
                Ok(IndexEntry {
                    oid: r.u32()?,
                    name: r.str()?,
                    table_oid: r.u32()?,
                    key_columns: (0..r.u32()?)
                        .map(|_| r.u32().map(|c| c as usize))
                        .collect::<Result<_, _>>()?,
                    index_type: match r.u8()? {
                        INDEX_TAG_BPLUS_TREE => IndexType::BPlusTree,
                        tag => return Err(PersistError::UnknownIndexType(tag)),
                    },
                    header_page_id: r.usize()?,
                })
            })
            .collect::<Result<_, PersistError>>()?;
        if !r.0.is_empty() {
            return Err(PersistError::CorruptMeta(
                "trailing bytes after the catalog",
            ));
        }
        Ok(Self {
            allocation: PageAllocation {
                next_page_id,
                free_page_ids,
            },
            last_commit_ts,
            next_txn_id,
            catalog: CatalogSnapshot {
                next_table_oid,
                next_index_oid,
                tables,
                indexes,
            },
        })
    }
}

fn encode_type(w: &mut Writer, dtype: DataType) {
    w.u8(match dtype {
        DataType::BOOLEAN => TYPE_TAG_BOOLEAN,
        DataType::INT => TYPE_TAG_INT,
        DataType::FLOAT => TYPE_TAG_FLOAT,
        DataType::TIMESTAMP => TYPE_TAG_TIMESTAMP,
        DataType::BIGINT => TYPE_TAG_BIGINT,
        DataType::TEXT => TYPE_TAG_TEXT,
        DataType::VARCHAR(_) => TYPE_TAG_VARCHAR,
        DataType::CHAR(_) => TYPE_TAG_CHAR,
    });
    if let DataType::VARCHAR(chars) | DataType::CHAR(chars) = dtype {
        w.u32(chars);
    }
}

fn decode_type(r: &mut Reader<'_>) -> Result<DataType, PersistError> {
    match r.u8()? {
        TYPE_TAG_BOOLEAN => Ok(DataType::BOOLEAN),
        TYPE_TAG_INT => Ok(DataType::INT),
        TYPE_TAG_FLOAT => Ok(DataType::FLOAT),
        TYPE_TAG_TIMESTAMP => Ok(DataType::TIMESTAMP),
        TYPE_TAG_BIGINT => Ok(DataType::BIGINT),
        TYPE_TAG_TEXT => Ok(DataType::TEXT),
        TYPE_TAG_VARCHAR => Ok(DataType::VARCHAR(r.u32()?)),
        TYPE_TAG_CHAR => Ok(DataType::CHAR(r.u32()?)),
        tag => Err(PersistError::UnknownColumnType(tag)),
    }
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], PersistError> {
        if self.0.len() < n {
            return Err(PersistError::CorruptMeta("the metadata ends early"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, PersistError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, PersistError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, PersistError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn usize(&mut self) -> Result<usize, PersistError> {
        usize::try_from(self.u64()?)
            .map_err(|_| PersistError::CorruptMeta("a page id does not fit in usize"))
    }

    fn str(&mut self) -> Result<String, PersistError> {
        let len = self.u32()? as usize;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| PersistError::CorruptMeta("a name is not UTF-8"))
    }
}

/// How many metadata chain pages a payload of `len` bytes takes, at least one.
pub fn chain_pages_for(len: usize) -> usize {
    len.div_ceil(CHAIN_PAGE_CAPACITY).max(1)
}

/// Formats page [`HEADER_PAGE_ID`] of a new database, whose metadata chain starts at
/// `first_chain_page`.
pub fn write_header(
    bpm: &Arc<BufferPoolManager>,
    first_chain_page: usize,
) -> Result<(), PersistError> {
    let mut page = bpm.write_page(HEADER_PAGE_ID)?;
    let buf = page.data_mut();
    buf.fill(0);
    buf[..HEADER_MAGIC.len()].copy_from_slice(&HEADER_MAGIC);
    buf[HEADER_VERSION_OFF..HEADER_VERSION_OFF + 4].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf[HEADER_META_PAGE_OFF..HEADER_META_PAGE_OFF + 8]
        .copy_from_slice(&(first_chain_page as u64).to_le_bytes());
    Ok(())
}

/// The first page of the metadata chain, read from the header page. Fails with
/// [`PersistError::NotADatabase`] or [`PersistError::UnsupportedFormat`] for a data file this
/// version cannot read.
pub fn read_header(bpm: &Arc<BufferPoolManager>) -> Result<usize, PersistError> {
    let page = bpm.read_page(HEADER_PAGE_ID)?;
    let buf = page.data();
    if buf[..HEADER_MAGIC.len()] != HEADER_MAGIC {
        return Err(PersistError::NotADatabase);
    }
    let version = u32::from_le_bytes(
        buf[HEADER_VERSION_OFF..HEADER_VERSION_OFF + 4]
            .try_into()
            .unwrap(),
    );
    if version != FORMAT_VERSION {
        return Err(PersistError::UnsupportedFormat(version));
    }
    let first = u64::from_le_bytes(
        buf[HEADER_META_PAGE_OFF..HEADER_META_PAGE_OFF + 8]
            .try_into()
            .unwrap(),
    );
    usize::try_from(first).map_err(|_| PersistError::CorruptMeta("bad first chain page"))
}

/// Writes `payload` across the metadata chain `chain`, in order. Pages past the end of the
/// payload are written empty.
///
/// Panics if `chain` holds fewer than [`chain_pages_for`]`(payload.len())` pages.
pub fn write_chain(
    bpm: &Arc<BufferPoolManager>,
    chain: &[usize],
    payload: &[u8],
) -> Result<(), PersistError> {
    assert!(
        chain.len() >= chain_pages_for(payload.len()),
        "a metadata payload of {} bytes needs more than {} chain pages",
        payload.len(),
        chain.len()
    );
    let mut parts = payload.chunks(CHAIN_PAGE_CAPACITY);
    for (i, &page_id) in chain.iter().enumerate() {
        let part = parts.next().unwrap_or(&[]);
        let next = chain.get(i + 1).map_or(NO_NEXT_CHAIN_PAGE, |&id| id as u64);
        let mut page = bpm.write_page(page_id)?;
        let buf = page.data_mut();
        buf[..CHAIN_MAGIC.len()].copy_from_slice(&CHAIN_MAGIC);
        buf[CHAIN_NEXT_OFF..CHAIN_NEXT_OFF + 8].copy_from_slice(&next.to_le_bytes());
        buf[CHAIN_LEN_OFF..CHAIN_LEN_OFF + 4].copy_from_slice(&(part.len() as u32).to_le_bytes());
        buf[CHAIN_PAYLOAD_OFF..CHAIN_PAYLOAD_OFF + part.len()].copy_from_slice(part);
    }
    Ok(())
}

/// Reads the metadata chain that starts at `first`: its page ids, in order, and its payload.
pub fn read_chain(
    bpm: &Arc<BufferPoolManager>,
    first: usize,
) -> Result<(Vec<usize>, Vec<u8>), PersistError> {
    let mut chain = Vec::new();
    let mut seen = HashSet::new();
    let mut payload = Vec::new();
    let mut next = first as u64;
    while next != NO_NEXT_CHAIN_PAGE {
        let page_id = usize::try_from(next)
            .map_err(|_| PersistError::CorruptMeta("a chain link does not fit in usize"))?;
        if !seen.insert(page_id) {
            return Err(PersistError::CorruptMeta("the metadata chain loops"));
        }
        let page = bpm.read_page(page_id)?;
        let buf = page.data();
        if buf[..CHAIN_MAGIC.len()] != CHAIN_MAGIC {
            return Err(PersistError::CorruptMeta(
                "a metadata chain page has the wrong magic",
            ));
        }
        let len =
            u32::from_le_bytes(buf[CHAIN_LEN_OFF..CHAIN_LEN_OFF + 4].try_into().unwrap()) as usize;
        if len > CHAIN_PAGE_CAPACITY {
            return Err(PersistError::CorruptMeta(
                "a metadata chain page claims more bytes than it holds",
            ));
        }
        payload.extend_from_slice(&buf[CHAIN_PAYLOAD_OFF..CHAIN_PAYLOAD_OFF + len]);
        next = u64::from_le_bytes(buf[CHAIN_NEXT_OFF..CHAIN_NEXT_OFF + 8].try_into().unwrap());
        chain.push(page_id);
    }
    Ok((chain, payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    fn sample() -> DatabaseMeta {
        DatabaseMeta {
            allocation: PageAllocation {
                next_page_id: 40,
                free_page_ids: vec![3, 17, 39],
            },
            last_commit_ts: 12,
            next_txn_id: (1 << 62) + 7,
            catalog: CatalogSnapshot {
                next_table_oid: 3,
                next_index_oid: 2,
                tables: vec![
                    TableEntry {
                        oid: 0,
                        name: "t".into(),
                        columns: vec![
                            (DataType::INT, "id".into()),
                            (DataType::BOOLEAN, "ok".into()),
                            (DataType::FLOAT, "f".into()),
                            (DataType::TIMESTAMP, "at".into()),
                            (DataType::TEXT, "note".into()),
                            (DataType::VARCHAR(20), "name".into()),
                            (DataType::CHAR(2), "code".into()),
                        ],
                        first_page_id: 2,
                    },
                    TableEntry {
                        oid: 2,
                        name: "naïve".into(),
                        columns: vec![],
                        first_page_id: 9,
                    },
                ],
                indexes: vec![IndexEntry {
                    oid: 1,
                    name: "t_id".into(),
                    table_oid: 0,
                    key_columns: vec![0, 3],
                    index_type: IndexType::BPlusTree,
                    header_page_id: 5,
                }],
            },
        }
    }

    #[test]
    fn meta_round_trips_through_its_encoding() {
        let meta = sample();
        assert_eq!(DatabaseMeta::decode(&meta.encode()).unwrap(), meta);
    }

    #[test]
    fn every_truncation_of_the_encoding_is_rejected() {
        let bytes = sample().encode();
        for len in 0..bytes.len() {
            assert!(
                matches!(
                    DatabaseMeta::decode(&bytes[..len]),
                    Err(PersistError::CorruptMeta(_))
                ),
                "decoding the first {len} bytes"
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(matches!(
            DatabaseMeta::decode(&longer),
            Err(PersistError::CorruptMeta(_))
        ));
    }

    #[test]
    fn column_types_encode_as_their_stable_tags() {
        for (dtype, tag) in [
            (DataType::BOOLEAN, TYPE_TAG_BOOLEAN),
            (DataType::INT, TYPE_TAG_INT),
            (DataType::FLOAT, TYPE_TAG_FLOAT),
            (DataType::TIMESTAMP, TYPE_TAG_TIMESTAMP),
            (DataType::BIGINT, TYPE_TAG_BIGINT),
            (DataType::TEXT, TYPE_TAG_TEXT),
        ] {
            let mut w = Writer::default();
            encode_type(&mut w, dtype);
            assert_eq!(w.0, vec![tag], "tag of {dtype:?}");
            assert_eq!(decode_type(&mut Reader(&w.0)).unwrap(), dtype);
        }
        let mut w = Writer::default();
        encode_type(&mut w, DataType::VARCHAR(300));
        assert_eq!(
            w.0,
            [&[TYPE_TAG_VARCHAR][..], &300u32.to_le_bytes()].concat()
        );
        assert_eq!(
            decode_type(&mut Reader(&w.0)).unwrap(),
            DataType::VARCHAR(300)
        );
        assert!(matches!(
            decode_type(&mut Reader(&[TYPE_TAG_VARCHAR, 1])),
            Err(PersistError::CorruptMeta(_))
        ));
        let mut w = Writer::default();
        encode_type(&mut w, DataType::CHAR(12));
        assert_eq!(w.0, [&[TYPE_TAG_CHAR][..], &12u32.to_le_bytes()].concat());
        assert_eq!(decode_type(&mut Reader(&w.0)).unwrap(), DataType::CHAR(12));
        assert!(matches!(
            decode_type(&mut Reader(&[TYPE_TAG_CHAR, 1, 0])),
            Err(PersistError::CorruptMeta(_))
        ));
        assert!(matches!(
            decode_type(&mut Reader(&[200])),
            Err(PersistError::UnknownColumnType(200))
        ));
    }

    #[test]
    fn an_unknown_index_type_is_rejected() {
        let mut bytes = sample().encode();
        let tag_at = bytes.len() - 9;
        assert_eq!(bytes[tag_at], INDEX_TAG_BPLUS_TREE);
        bytes[tag_at] = 9;
        assert!(matches!(
            DatabaseMeta::decode(&bytes),
            Err(PersistError::UnknownIndexType(9))
        ));
    }

    #[test]
    fn a_chain_round_trips_payloads_of_any_length() {
        let (bpm, _dir) = testing::bpm(8);
        let pages: Vec<usize> = (0..4)
            .map(|_| bpm.new_page().unwrap().page_id().unwrap())
            .collect();
        for len in [
            0,
            1,
            CHAIN_PAGE_CAPACITY,
            CHAIN_PAGE_CAPACITY + 1,
            3 * CHAIN_PAGE_CAPACITY,
        ] {
            let payload: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let chain = &pages[..chain_pages_for(len).max(2)];
            write_chain(&bpm, chain, &payload).unwrap();
            let (read, back) = read_chain(&bpm, chain[0]).unwrap();
            assert_eq!(read, chain, "chain of a {len}-byte payload");
            assert_eq!(back, payload, "payload of {len} bytes");
        }
    }

    #[test]
    fn a_damaged_chain_is_reported() {
        let (bpm, _dir) = testing::bpm(8);
        let a = bpm.new_page().unwrap().page_id().unwrap();
        let b = bpm.new_page().unwrap().page_id().unwrap();
        write_chain(&bpm, &[a, b], b"x").unwrap();
        bpm.write_page(b).unwrap().data_mut()[CHAIN_NEXT_OFF..CHAIN_NEXT_OFF + 8]
            .copy_from_slice(&(a as u64).to_le_bytes());
        assert!(matches!(
            read_chain(&bpm, a),
            Err(PersistError::CorruptMeta("the metadata chain loops"))
        ));
        bpm.write_page(b).unwrap().data_mut()[0] = 0;
        assert!(matches!(
            read_chain(&bpm, a),
            Err(PersistError::CorruptMeta(_))
        ));
    }

    #[test]
    fn a_header_round_trips_and_a_foreign_one_is_refused() {
        let (bpm, _dir) = testing::bpm(8);
        drop(bpm.new_page().unwrap());
        write_header(&bpm, 7).unwrap();
        assert_eq!(read_header(&bpm).unwrap(), 7);

        bpm.write_page(HEADER_PAGE_ID).unwrap().data_mut()[HEADER_VERSION_OFF] = 99;
        assert!(matches!(
            read_header(&bpm),
            Err(PersistError::UnsupportedFormat(99))
        ));
        bpm.write_page(HEADER_PAGE_ID).unwrap().data_mut()[0] = b'X';
        assert!(matches!(read_header(&bpm), Err(PersistError::NotADatabase)));
    }

    #[test]
    #[should_panic(expected = "needs more than 1 chain pages")]
    fn writing_a_payload_into_too_short_a_chain_panics() {
        let (bpm, _dir) = testing::bpm(8);
        let a = bpm.new_page().unwrap().page_id().unwrap();
        write_chain(&bpm, &[a], &vec![0; CHAIN_PAGE_CAPACITY + 1]).unwrap();
    }

    #[test]
    fn errors_display_readable_messages() {
        let cases = [
            (
                PersistError::NotADatabase,
                "the data file is not a cmudb database",
            ),
            (
                PersistError::InUse,
                "the database is open in another process",
            ),
            (
                PersistError::UnsupportedFormat(7),
                "the data file has format version 7, but this cmudb reads version 2",
            ),
            (
                PersistError::CorruptMeta("x"),
                "the database metadata is damaged: x",
            ),
            (
                PersistError::UnknownColumnType(4),
                "unknown column type tag 4",
            ),
            (
                PersistError::UnknownIndexType(4),
                "unknown index type tag 4",
            ),
            (
                PersistError::Catalog(CatalogError::TableExists),
                "catalog: table already exists",
            ),
            (
                PersistError::AbortOpenTransaction("boom".into()),
                "rolling back the open transaction failed: boom",
            ),
            (
                PersistError::Bpm(BpmError::NoFreeFrame),
                "buffer pool error: No free frame available",
            ),
            (PersistError::Io(io::Error::other("disk")), "io error: disk"),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }
}
