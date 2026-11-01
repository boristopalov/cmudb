//! Typed, zero-copy view over a slotted table page.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! offset 0   next_page_id       u32   (NO_NEXT_PAGE on the last page)
//! offset 4   num_tuples         u16
//! offset 6   num_deleted_tuples u16
//! offset 8   free_space_offset  u32   (tuple data grows backwards from PAGE_SIZE)
//! offset 12  slot directory     num_tuples * { offset u32, size u32, is_deleted u8 }
//! ...        free space
//! free_space_offset..PAGE_SIZE  tuple bytes
//! ```
//!
//! `TablePage<&[u8]>` reads straight out of a pinned frame and `TablePage<&mut [u8]>`
//! writes straight into one: there is no intermediate decoded representation, so reading
//! one tuple touches one slot instead of the whole directory.

use crate::buf::{get_u16_le, get_u32_le, put_u16_le, put_u32_le};
use crate::disk::PAGE_SIZE;
use std::fmt;

#[derive(Debug)]
pub struct TablePageError(pub String);

impl fmt::Display for TablePageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A single slot directory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TupleMeta {
    /// End offset (exclusive) of the tuple's bytes within the page.
    pub offset: u32,
    pub size: u32,
    pub is_deleted: bool,
}

impl TupleMeta {
    fn range(&self) -> std::ops::Range<usize> {
        (self.offset - self.size) as usize..self.offset as usize
    }
}

/// `next_page_id` of the last page of a table. Page 0 is a valid page, so it cannot mark the end.
pub const NO_NEXT_PAGE: u32 = u32::MAX;

const NEXT_PAGE_ID_OFF: usize = 0;
const NUM_TUPLES_OFF: usize = 4;
const NUM_DELETED_OFF: usize = 6;
const FREE_SPACE_OFF: usize = 8;

const SLOT_OFFSET_OFF: usize = 0;
const SLOT_SIZE_OFF: usize = 4;
const SLOT_DELETED_OFF: usize = 8;

pub struct TablePage<B> {
    buf: B,
}

pub type TablePageRef<'a> = TablePage<&'a [u8]>;
pub type TablePageMut<'a> = TablePage<&'a mut [u8]>;

impl<B: AsRef<[u8]>> TablePage<B> {
    pub const HEADER_SIZE: usize = 12;
    pub const SLOT_SIZE: usize = 9;
    pub const MAX_TUPLE_SIZE: usize = PAGE_SIZE - Self::HEADER_SIZE - Self::SLOT_SIZE;

    /// Wraps an initialized page, validating the header once so that later accessors can
    /// index the buffer directly.
    pub fn new(buf: B) -> Result<Self, TablePageError> {
        let page = Self { buf };
        let len = page.bytes().len();
        if len != PAGE_SIZE {
            return Err(TablePageError(format!(
                "table page must be {PAGE_SIZE} bytes, got {len}"
            )));
        }
        let directory_end = page.directory_end();
        let free = page.free_space_offset() as usize;
        if directory_end > free || free > PAGE_SIZE {
            return Err(TablePageError(format!(
                "slot directory ends at {directory_end} but tuple area starts at {free}"
            )));
        }
        Ok(page)
    }

    #[inline]
    fn bytes(&self) -> &[u8] {
        self.buf.as_ref()
    }

    /// The page that continues this table, or `None` on the last page.
    pub fn next_page_id(&self) -> Option<u32> {
        Some(get_u32_le(self.bytes(), NEXT_PAGE_ID_OFF)).filter(|&id| id != NO_NEXT_PAGE)
    }

    pub fn num_tuples(&self) -> u16 {
        get_u16_le(self.bytes(), NUM_TUPLES_OFF)
    }

    pub fn num_deleted_tuples(&self) -> u16 {
        get_u16_le(self.bytes(), NUM_DELETED_OFF)
    }

    pub fn free_space_offset(&self) -> u32 {
        get_u32_le(self.bytes(), FREE_SPACE_OFF)
    }

    fn directory_end(&self) -> usize {
        Self::HEADER_SIZE + self.num_tuples() as usize * Self::SLOT_SIZE
    }

    fn slot_base(slot: u32) -> usize {
        Self::HEADER_SIZE + slot as usize * Self::SLOT_SIZE
    }

    /// Returns the slot directory entry for `slot`, or `None` if the slot does not exist.
    pub fn slot(&self, slot: u32) -> Option<TupleMeta> {
        if slot >= u32::from(self.num_tuples()) {
            return None;
        }
        let base = Self::slot_base(slot);
        let b = self.bytes();
        Some(TupleMeta {
            offset: get_u32_le(b, base + SLOT_OFFSET_OFF),
            size: get_u32_le(b, base + SLOT_SIZE_OFF),
            is_deleted: b[base + SLOT_DELETED_OFF] != 0,
        })
    }

    /// Borrows the tuple bytes described by `meta` directly out of the page.
    pub fn tuple_bytes(&self, meta: &TupleMeta) -> Result<&[u8], TablePageError> {
        let free = self.free_space_offset();
        if meta.size > meta.offset
            || meta.offset - meta.size < free
            || meta.offset as usize > PAGE_SIZE
        {
            return Err(TablePageError(format!(
                "slot points outside tuple area (offset {}, size {}, tuple area starts at {free})",
                meta.offset, meta.size
            )));
        }
        Ok(&self.bytes()[meta.range()])
    }

    pub fn can_insert(&self, tuple_len: usize) -> bool {
        self.directory_end() + Self::SLOT_SIZE + tuple_len <= self.free_space_offset() as usize
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> TablePage<B> {
    /// Formats `buf` as an empty table page and returns a view over it. The rest of the
    /// buffer is left untouched; it is unreachable until slots point into it.
    pub fn init(mut buf: B) -> Self {
        let b = buf.as_mut();
        put_u32_le(b, NEXT_PAGE_ID_OFF, NO_NEXT_PAGE);
        put_u16_le(b, NUM_TUPLES_OFF, 0);
        put_u16_le(b, NUM_DELETED_OFF, 0);
        put_u32_le(b, FREE_SPACE_OFF, PAGE_SIZE as u32);
        Self { buf }
    }

    #[inline]
    fn bytes_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut()
    }

    pub fn set_next_page_id(&mut self, page_id: u32) {
        put_u32_le(self.bytes_mut(), NEXT_PAGE_ID_OFF, page_id);
    }

    /// Appends `data` as a new tuple, writing only its slot, its bytes and the header.
    /// Returns the new slot id.
    pub fn insert(&mut self, data: &[u8]) -> Result<u32, TablePageError> {
        if !self.can_insert(data.len()) {
            return Err(TablePageError("tuple does not fit on page".to_string()));
        }
        let num_tuples = self.num_tuples();
        let slot = u32::from(num_tuples);
        let new_num_tuples = num_tuples
            .checked_add(1)
            .ok_or_else(|| TablePageError("table page slot count overflow".to_string()))?;
        let offset = self.free_space_offset();
        let size = data.len() as u32;
        let start = offset - size;
        let base = Self::slot_base(slot);

        let b = self.bytes_mut();
        b[start as usize..offset as usize].copy_from_slice(data);
        put_u32_le(b, base + SLOT_OFFSET_OFF, offset);
        put_u32_le(b, base + SLOT_SIZE_OFF, size);
        b[base + SLOT_DELETED_OFF] = 0;
        put_u16_le(b, NUM_TUPLES_OFF, new_num_tuples);
        put_u32_le(b, FREE_SPACE_OFF, start);
        Ok(slot)
    }

    /// Tombstones `slot`. The tuple bytes are left in place.
    pub fn mark_deleted(&mut self, slot: u32) -> Result<(), TablePageError> {
        let meta = self.slot(slot).ok_or_else(|| {
            TablePageError(format!(
                "slot {slot} does not exist (page has {} slots)",
                self.num_tuples()
            ))
        })?;
        if meta.is_deleted {
            return Err(TablePageError(format!("slot {slot} already deleted")));
        }
        let deleted = self.num_deleted_tuples().saturating_add(1);
        let b = self.bytes_mut();
        b[Self::slot_base(slot) + SLOT_DELETED_OFF] = 1;
        put_u16_le(b, NUM_DELETED_OFF, deleted);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_produces_empty_valid_page() {
        let mut buf = [0xabu8; PAGE_SIZE];
        let page = TablePage::init(&mut buf[..]);
        assert_eq!(page.num_tuples(), 0);
        assert_eq!(page.num_deleted_tuples(), 0);
        assert_eq!(page.next_page_id(), None);
        assert_eq!(page.free_space_offset() as usize, PAGE_SIZE);
        assert!(page.slot(0).is_none());
        assert!(TablePageRef::new(&buf[..]).is_ok());
    }

    #[test]
    fn insert_and_read_back_in_place() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        assert_eq!(page.insert(b"alpha").unwrap(), 0);
        assert_eq!(page.insert(b"beta!").unwrap(), 1);
        page.set_next_page_id(7);

        let page = TablePageRef::new(&buf[..]).unwrap();
        assert_eq!(page.num_tuples(), 2);
        assert_eq!(page.next_page_id(), Some(7));
        let m0 = page.slot(0).unwrap();
        let m1 = page.slot(1).unwrap();
        assert_eq!(page.tuple_bytes(&m0).unwrap(), b"alpha");
        assert_eq!(page.tuple_bytes(&m1).unwrap(), b"beta!");
        assert_eq!(
            m0.offset as usize, PAGE_SIZE,
            "tuples grow backwards from the end of the page"
        );
        assert_eq!(m1.offset, m0.offset - m0.size);
        assert_eq!(page.free_space_offset(), m1.offset - m1.size);
    }

    #[test]
    fn mark_deleted_sets_tombstone_and_counter() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"x").unwrap();
        page.insert(b"y").unwrap();
        page.mark_deleted(1).unwrap();
        assert!(page.mark_deleted(1).is_err(), "double delete rejected");
        assert!(page.mark_deleted(2).is_err(), "missing slot rejected");
        assert_eq!(page.num_deleted_tuples(), 1);
        assert!(!page.slot(0).unwrap().is_deleted);
        assert!(page.slot(1).unwrap().is_deleted);
    }

    #[test]
    fn insert_rejects_when_full() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        let big = vec![1u8; TablePageMut::MAX_TUPLE_SIZE];
        page.insert(&big).unwrap();
        assert!(!page.can_insert(0));
        assert!(page.insert(b"").is_err());
        let meta = page.slot(0).unwrap();
        assert_eq!(page.tuple_bytes(&meta).unwrap(), &big[..]);
    }

    #[test]
    fn new_rejects_corrupt_header() {
        let mut buf = [0u8; PAGE_SIZE];
        TablePage::init(&mut buf[..]);
        put_u16_le(&mut buf, NUM_TUPLES_OFF, u16::MAX);
        assert!(
            TablePageRef::new(&buf[..]).is_err(),
            "more slots than fit before the tuple area"
        );
    }

    #[test]
    fn new_rejects_free_space_offset_past_page_end() {
        let mut buf = [0u8; PAGE_SIZE];
        TablePage::init(&mut buf[..]);
        put_u32_le(&mut buf, FREE_SPACE_OFF, PAGE_SIZE as u32 + 1);
        assert!(TablePageRef::new(&buf[..]).is_err());
    }

    #[test]
    fn tuple_bytes_rejects_out_of_range_slot() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"abc").unwrap();
        let bogus = TupleMeta {
            offset: 10,
            size: 4,
            is_deleted: false,
        };
        assert!(page.tuple_bytes(&bogus).is_err());
    }
}
