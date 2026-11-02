//! Typed, zero-copy view over a slotted table page.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! offset 0   next_page_id       u32   (NO_NEXT_PAGE on the last page)
//! offset 4   num_tuples         u16
//! offset 6   num_deleted_tuples u16
//! offset 8   free_space_offset  u32   (tuple data grows backwards from PAGE_SIZE)
//! offset 12  slot directory     num_tuples * { offset u32, size u16, capacity u16, ts u64, is_deleted u8 }
//! ...        free space
//! free_space_offset..PAGE_SIZE  tuple bytes
//! ```
//!
//! `TablePage<&[u8]>` reads straight out of a pinned frame and `TablePage<&mut [u8]>`
//! writes straight into one: there is no intermediate decoded representation, so reading
//! one tuple touches one slot instead of the whole directory.
//!
//! A slot owns the `capacity` bytes that end at its `offset`; its tuple is the first `size` of
//! them. An update keeps the slot (and so the record id): a tuple no larger than the capacity is
//! written in place, and a larger one gets a new region on the same page, after compacting the
//! page when its free space is too small. Shrinking a tuple keeps its slot's capacity, so writing
//! back an earlier image of the tuple, as an abort does, fits where it was.
//!
//! Each slot carries the MVCC metadata of the newest version of its tuple ([`TupleMeta`]):
//! the timestamp that wrote it (a commit timestamp, or a transaction id while uncommitted)
//! and whether that version is a deletion. Older versions live in undo chains owned by the
//! transaction manager.

use crate::buf::{get_u16_le, get_u32_le, get_u64_le, put_u16_le, put_u32_le, put_u64_le};
use crate::disk::PAGE_SIZE;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TablePageError {
    /// The buffer is not exactly one page long.
    WrongSize { len: usize },
    /// The header's slot directory runs into the tuple area, or the tuple area starts past
    /// the end of the page.
    CorruptHeader {
        directory_end: usize,
        free_space_offset: usize,
    },
    /// A slot's byte range lies outside the page's tuple area, or its tuple is larger than its
    /// capacity.
    SlotOutOfBounds {
        offset: u32,
        size: u32,
        capacity: u32,
        tuple_area_start: u32,
    },
    /// The tuple and its slot do not fit in the page's free space.
    DoesNotFit { len: usize },
    /// The page already holds `u16::MAX` slots.
    SlotCountOverflow,
    /// The slot id is past the end of the slot directory.
    SlotMissing { slot: u32, num_slots: u16 },
    /// The slot is already a deletion marker.
    AlreadyDeleted { slot: u32 },
    /// The deleted-tuple counter would overflow or drop below zero for a change to `slot`.
    CorruptDeletedCounter { count: u16, slot: u32 },
    /// An update of `slot` to `len` bytes does not fit on the page, which has `available` bytes
    /// left for it even after compaction. A row cannot move to another page.
    UpdateDoesNotFit {
        slot: u32,
        len: usize,
        available: usize,
    },
}

impl fmt::Display for TablePageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TablePageError::WrongSize { len } => {
                write!(f, "table page must be {PAGE_SIZE} bytes, got {len}")
            }
            TablePageError::CorruptHeader {
                directory_end,
                free_space_offset,
            } => write!(
                f,
                "slot directory ends at {directory_end} but tuple area starts at {free_space_offset}"
            ),
            TablePageError::SlotOutOfBounds {
                offset,
                size,
                capacity,
                tuple_area_start,
            } => write!(
                f,
                "slot points outside tuple area (offset {offset}, size {size}, capacity {capacity}, tuple area starts at {tuple_area_start})"
            ),
            TablePageError::DoesNotFit { len } => {
                write!(f, "tuple of {len} bytes does not fit on page")
            }
            TablePageError::SlotCountOverflow => write!(f, "table page slot count overflow"),
            TablePageError::SlotMissing { slot, num_slots } => {
                write!(f, "slot {slot} does not exist (page has {num_slots} slots)")
            }
            TablePageError::AlreadyDeleted { slot } => write!(f, "slot {slot} already deleted"),
            TablePageError::CorruptDeletedCounter { count, slot } => write!(
                f,
                "deleted-tuple counter {count} is corrupt for a change to slot {slot}"
            ),
            TablePageError::UpdateDoesNotFit {
                slot,
                len,
                available,
            } => write!(
                f,
                "the updated row in slot {slot} needs {len} bytes but its page has {available} bytes left for it, and rows cannot move to another page"
            ),
        }
    }
}

/// MVCC metadata stored with the newest version of every tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TupleMeta {
    /// Commit timestamp of the transaction that wrote this version, or that transaction's id
    /// (see [`crate::concurrency::TXN_START_ID`]) while it is still uncommitted.
    pub ts: u64,
    /// Whether this version is a deletion marker.
    pub is_deleted: bool,
}

/// A single slot directory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    /// End offset (exclusive) of the slot's region within the page.
    pub offset: u32,
    /// Length of the tuple, which starts where the region does.
    pub size: u32,
    /// Length of the region, at least `size`.
    pub capacity: u32,
    pub meta: TupleMeta,
}

impl Slot {
    fn region_start(&self) -> u32 {
        self.offset - self.capacity
    }

    fn region(&self) -> std::ops::Range<usize> {
        self.region_start() as usize..self.offset as usize
    }

    fn range(&self) -> std::ops::Range<usize> {
        let start = self.region_start() as usize;
        start..start + self.size as usize
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
const SLOT_CAPACITY_OFF: usize = 6;
const SLOT_TS_OFF: usize = 8;
const SLOT_DELETED_OFF: usize = 16;

pub struct TablePage<B> {
    buf: B,
}

pub type TablePageRef<'a> = TablePage<&'a [u8]>;
pub type TablePageMut<'a> = TablePage<&'a mut [u8]>;

impl<B: AsRef<[u8]>> TablePage<B> {
    pub const HEADER_SIZE: usize = 12;
    pub const SLOT_SIZE: usize = 17;
    pub const MAX_TUPLE_SIZE: usize = PAGE_SIZE - Self::HEADER_SIZE - Self::SLOT_SIZE;

    /// Wraps an initialized page, validating the header once so that later accessors can
    /// index the buffer directly.
    pub fn new(buf: B) -> Result<Self, TablePageError> {
        let page = Self { buf };
        let len = page.bytes().len();
        if len != PAGE_SIZE {
            return Err(TablePageError::WrongSize { len });
        }
        let directory_end = page.directory_end();
        let free = page.free_space_offset() as usize;
        if directory_end > free || free > PAGE_SIZE {
            return Err(TablePageError::CorruptHeader {
                directory_end,
                free_space_offset: free,
            });
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
    pub fn slot(&self, slot: u32) -> Option<Slot> {
        if slot >= u32::from(self.num_tuples()) {
            return None;
        }
        let base = Self::slot_base(slot);
        let b = self.bytes();
        Some(Slot {
            offset: get_u32_le(b, base + SLOT_OFFSET_OFF),
            size: u32::from(get_u16_le(b, base + SLOT_SIZE_OFF)),
            capacity: u32::from(get_u16_le(b, base + SLOT_CAPACITY_OFF)),
            meta: TupleMeta {
                ts: get_u64_le(b, base + SLOT_TS_OFF),
                is_deleted: b[base + SLOT_DELETED_OFF] != 0,
            },
        })
    }

    /// Borrows the tuple bytes described by `meta` directly out of the page.
    pub fn tuple_bytes(&self, meta: &Slot) -> Result<&[u8], TablePageError> {
        self.check_region(meta)?;
        Ok(&self.bytes()[meta.range()])
    }

    fn check_region(&self, meta: &Slot) -> Result<(), TablePageError> {
        let free = self.free_space_offset();
        if meta.size > meta.capacity
            || meta.capacity > meta.offset
            || meta.region_start() < free
            || meta.offset as usize > PAGE_SIZE
        {
            return Err(TablePageError::SlotOutOfBounds {
                offset: meta.offset,
                size: meta.size,
                capacity: meta.capacity,
                tuple_area_start: free,
            });
        }
        Ok(())
    }

    fn free_bytes(&self) -> usize {
        (self.free_space_offset() as usize).saturating_sub(self.directory_end())
    }

    /// Every slot of the page, in slot order.
    fn slots(&self) -> impl Iterator<Item = Slot> + '_ {
        (0..u32::from(self.num_tuples())).filter_map(|slot| self.slot(slot))
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

    /// Appends `data` as a new live tuple stamped with timestamp 0 (visible to everyone).
    /// Returns the new slot id.
    pub fn insert(&mut self, data: &[u8]) -> Result<u32, TablePageError> {
        self.insert_with_meta(TupleMeta::default(), data)
    }

    /// Appends `data` as a new tuple, writing only its slot, its bytes and the header.
    /// Returns the new slot id.
    pub fn insert_with_meta(
        &mut self,
        meta: TupleMeta,
        data: &[u8],
    ) -> Result<u32, TablePageError> {
        if !self.can_insert(data.len()) {
            return Err(TablePageError::DoesNotFit { len: data.len() });
        }
        let num_tuples = self.num_tuples();
        let slot = u32::from(num_tuples);
        let new_num_tuples = num_tuples
            .checked_add(1)
            .ok_or(TablePageError::SlotCountOverflow)?;
        let offset = self.free_space_offset();
        let size = tuple_len(data)?;
        let start = offset - u32::from(size);
        let base = Self::slot_base(slot);
        let deleted = self.num_deleted_tuples() + u16::from(meta.is_deleted);

        let b = self.bytes_mut();
        b[start as usize..offset as usize].copy_from_slice(data);
        put_u32_le(b, base + SLOT_OFFSET_OFF, offset);
        put_u16_le(b, base + SLOT_SIZE_OFF, size);
        put_u16_le(b, base + SLOT_CAPACITY_OFF, size);
        put_u64_le(b, base + SLOT_TS_OFF, meta.ts);
        b[base + SLOT_DELETED_OFF] = u8::from(meta.is_deleted);
        put_u16_le(b, NUM_TUPLES_OFF, new_num_tuples);
        put_u16_le(b, NUM_DELETED_OFF, deleted);
        put_u32_le(b, FREE_SPACE_OFF, start);
        Ok(slot)
    }

    fn existing_slot(&self, slot: u32) -> Result<Slot, TablePageError> {
        self.slot(slot).ok_or_else(|| TablePageError::SlotMissing {
            slot,
            num_slots: self.num_tuples(),
        })
    }

    /// Tombstones `slot`, keeping its timestamp. The tuple bytes are left in place.
    pub fn mark_deleted(&mut self, slot: u32) -> Result<(), TablePageError> {
        let current = self.existing_slot(slot)?;
        if current.meta.is_deleted {
            return Err(TablePageError::AlreadyDeleted { slot });
        }
        if self.num_deleted_tuples() == u16::MAX {
            return Err(TablePageError::CorruptDeletedCounter {
                count: u16::MAX,
                slot,
            });
        }
        self.update_meta(
            slot,
            TupleMeta {
                is_deleted: true,
                ..current.meta
            },
        )
    }

    /// Overwrites the MVCC metadata of `slot`, keeping the deleted-tuple counter in sync. A
    /// counter that would overflow or drop below zero means the page is corrupt, and the
    /// write is refused with nothing changed.
    pub fn update_meta(&mut self, slot: u32, meta: TupleMeta) -> Result<(), TablePageError> {
        let current = self.existing_slot(slot)?;
        let count = self.num_deleted_tuples();
        let deleted = match (current.meta.is_deleted, meta.is_deleted) {
            (false, true) => count.checked_add(1),
            (true, false) => count.checked_sub(1),
            _ => Some(count),
        }
        .ok_or(TablePageError::CorruptDeletedCounter { count, slot })?;
        let base = Self::slot_base(slot);
        let b = self.bytes_mut();
        put_u64_le(b, base + SLOT_TS_OFF, meta.ts);
        b[base + SLOT_DELETED_OFF] = u8::from(meta.is_deleted);
        put_u16_le(b, NUM_DELETED_OFF, deleted);
        Ok(())
    }

    /// Overwrites the bytes and metadata of `slot`, keeping the slot. A tuple that fits the
    /// slot's capacity is written where the old one was; a larger one moves to a new region of
    /// the page, compacting the page first when its free space is too small, and fails with
    /// [`TablePageError::UpdateDoesNotFit`] when even that leaves too little room. The slot's
    /// range, the room and the deleted-tuple counter are checked before anything is written.
    pub fn update_in_place(
        &mut self,
        slot: u32,
        meta: TupleMeta,
        data: &[u8],
    ) -> Result<(), TablePageError> {
        let current = self.existing_slot(slot)?;
        self.check_region(&current)?;
        let size = tuple_len(data)?;
        let placement = self.place(&current, slot, data.len())?;
        self.update_meta(slot, meta)?;
        let region = match placement {
            Placement::InPlace => current,
            Placement::Free => self.claim_free_region(slot, size),
            Placement::AfterCompaction => {
                self.compact_without(slot);
                self.claim_free_region(slot, size)
            }
        };
        let start = region.region_start() as usize;
        let base = Self::slot_base(slot);
        let b = self.bytes_mut();
        b[start..start + data.len()].copy_from_slice(data);
        put_u16_le(b, base + SLOT_SIZE_OFF, size);
        Ok(())
    }

    /// Where an update of `current` to `len` bytes goes.
    fn place(&self, current: &Slot, slot: u32, len: usize) -> Result<Placement, TablePageError> {
        if len <= current.capacity as usize {
            return Ok(Placement::InPlace);
        }
        if len <= self.free_bytes() {
            return Ok(Placement::Free);
        }
        let other_regions: usize = self
            .slots()
            .enumerate()
            .filter(|&(i, _)| i as u32 != slot)
            .map(|(_, s)| s.capacity as usize)
            .sum();
        let available = PAGE_SIZE - self.directory_end() - other_regions;
        if len <= available {
            Ok(Placement::AfterCompaction)
        } else {
            Err(TablePageError::UpdateDoesNotFit {
                slot,
                len,
                available,
            })
        }
    }

    /// Gives `slot` a new region of `size` bytes at the start of the free space and returns the
    /// slot as it now stands. The caller has checked that the free space holds `size` bytes.
    fn claim_free_region(&mut self, slot: u32, size: u16) -> Slot {
        let end = self.free_space_offset();
        let start = end - u32::from(size);
        let base = Self::slot_base(slot);
        let b = self.bytes_mut();
        put_u32_le(b, base + SLOT_OFFSET_OFF, end);
        put_u16_le(b, base + SLOT_CAPACITY_OFF, size);
        put_u32_le(b, FREE_SPACE_OFF, start);
        self.slot(slot)
            .expect("the slot was read before it was moved")
    }

    /// Packs every slot's region but `skip`'s against the end of the page, in slot order, and
    /// leaves `skip` with an empty region, so that the free space is all the page has left.
    fn compact_without(&mut self, skip: u32) {
        let slots: Vec<Slot> = self.slots().collect();
        let regions: Vec<Vec<u8>> = slots
            .iter()
            .map(|s| self.bytes()[s.region()].to_vec())
            .collect();
        let mut end = PAGE_SIZE as u32;
        let b = self.bytes_mut();
        for (i, (s, region)) in slots.iter().zip(regions).enumerate() {
            let base = Self::slot_base(i as u32);
            if i as u32 == skip {
                put_u32_le(b, base + SLOT_OFFSET_OFF, end);
                put_u16_le(b, base + SLOT_CAPACITY_OFF, 0);
                continue;
            }
            let start = end - s.capacity;
            b[start as usize..end as usize].copy_from_slice(&region);
            put_u32_le(b, base + SLOT_OFFSET_OFF, end);
            end = start;
        }
        put_u32_le(b, FREE_SPACE_OFF, end);
    }
}

/// Where [`TablePage::update_in_place`] writes a tuple.
enum Placement {
    /// Over the old tuple, inside the slot's region.
    InPlace,
    /// In a new region taken from the free space.
    Free,
    /// In a new region taken from the free space once the page is compacted.
    AfterCompaction,
}

/// The length of `data` as a slot stores it. Fails with [`TablePageError::DoesNotFit`] for a
/// tuple longer than a page.
fn tuple_len(data: &[u8]) -> Result<u16, TablePageError> {
    u16::try_from(data.len())
        .ok()
        .filter(|&len| usize::from(len) <= TablePage::<&[u8]>::MAX_TUPLE_SIZE)
        .ok_or(TablePageError::DoesNotFit { len: data.len() })
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
        assert_eq!((m0.size, m0.capacity), (5, 5));
        assert_eq!(m1.offset, m0.offset - m0.capacity);
        assert_eq!(page.free_space_offset(), m1.offset - m1.capacity);
    }

    #[test]
    fn mark_deleted_sets_tombstone_and_counter() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"x").unwrap();
        page.insert(b"y").unwrap();
        page.mark_deleted(1).unwrap();
        assert_eq!(
            page.mark_deleted(1),
            Err(TablePageError::AlreadyDeleted { slot: 1 }),
            "double delete rejected"
        );
        assert_eq!(
            page.mark_deleted(2),
            Err(TablePageError::SlotMissing {
                slot: 2,
                num_slots: 2
            }),
            "missing slot rejected"
        );
        assert_eq!(page.num_deleted_tuples(), 1);
        assert!(!page.slot(0).unwrap().meta.is_deleted);
        assert!(page.slot(1).unwrap().meta.is_deleted);
    }

    #[test]
    fn meta_round_trips_and_tracks_deleted_count() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        let live = TupleMeta {
            ts: 42,
            is_deleted: false,
        };
        let dead = TupleMeta {
            ts: 1 << 62,
            is_deleted: true,
        };
        page.insert_with_meta(live, b"aa").unwrap();
        page.insert_with_meta(dead, b"bb").unwrap();
        assert_eq!(page.num_deleted_tuples(), 1);
        assert_eq!(page.slot(0).unwrap().meta, live);
        assert_eq!(page.slot(1).unwrap().meta, dead);

        page.update_meta(1, live).unwrap();
        assert_eq!(page.num_deleted_tuples(), 0);
        page.update_meta(0, dead).unwrap();
        page.update_meta(0, dead).unwrap();
        assert_eq!(page.num_deleted_tuples(), 1);
    }

    #[test]
    fn update_in_place_rewrites_bytes_and_keeps_the_slot_when_the_size_changes() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"old").unwrap();
        page.insert(b"keep").unwrap();
        let meta = TupleMeta {
            ts: 7,
            is_deleted: false,
        };
        page.update_in_place(0, meta, b"new").unwrap();
        page.update_in_place(0, meta, b"a").unwrap();
        let s0 = page.slot(0).unwrap();
        assert_eq!(page.tuple_bytes(&s0).unwrap(), b"a");
        assert_eq!(
            (s0.offset, s0.capacity),
            (PAGE_SIZE as u32, 3),
            "a shorter tuple stays where it was and the slot keeps its capacity"
        );

        let free_before = page.free_space_offset();
        page.update_in_place(0, meta, b"longer").unwrap();
        let s0 = page.slot(0).unwrap();
        assert_eq!(page.tuple_bytes(&s0).unwrap(), b"longer");
        assert_eq!(s0.meta, meta);
        assert_eq!(
            (s0.offset, s0.capacity),
            (free_before, 6),
            "a longer tuple moves to a new region taken from the free space"
        );
        assert_eq!(page.free_space_offset(), free_before - 6);
        let s1 = page.slot(1).unwrap();
        assert_eq!(page.tuple_bytes(&s1).unwrap(), b"keep");
    }

    #[test]
    fn update_in_place_compacts_the_page_to_make_room_for_a_longer_tuple() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        let row = vec![1u8; 1000];
        let mut slots = 0;
        while page.can_insert(row.len()) {
            page.insert(&row).unwrap();
            slots += 1;
        }
        let meta = TupleMeta::default();
        page.update_in_place(0, meta, b"short").unwrap();
        page.mark_deleted(2).unwrap();
        let grown = vec![2u8; 1000 + page.free_bytes() + 1];
        assert_eq!(
            page.update_in_place(1, meta, &grown),
            Err(TablePageError::UpdateDoesNotFit {
                slot: 1,
                len: grown.len(),
                available: grown.len() - 1,
            }),
            "slot 0 keeps its capacity, so compaction frees nothing"
        );

        let fits = vec![2u8; 1000 + page.free_bytes()];
        page.update_in_place(1, meta, &fits).unwrap();
        assert_eq!(page.free_space_offset() as usize, page.directory_end());
        for slot in 0..slots {
            let s = page.slot(slot).unwrap();
            let expected: &[u8] = match slot {
                0 => b"short",
                1 => &fits,
                _ => &row,
            };
            assert_eq!(page.tuple_bytes(&s).unwrap(), expected, "slot {slot}");
        }
        assert!(page.slot(2).unwrap().meta.is_deleted);
        assert_eq!(page.num_deleted_tuples(), 1);

        page.update_in_place(0, meta, &row)
            .expect("an earlier image of a slot always fits its capacity");
        let s0 = page.slot(0).unwrap();
        assert_eq!(page.tuple_bytes(&s0).unwrap(), &row[..]);
    }

    #[test]
    fn update_in_place_that_does_not_fit_writes_nothing() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"old").unwrap();
        let before = buf;
        let mut page = TablePageMut::new(&mut buf[..]).unwrap();
        let live = TupleMeta {
            ts: 3,
            is_deleted: false,
        };
        let too_long = vec![0u8; TablePageMut::MAX_TUPLE_SIZE + 1];
        assert_eq!(
            page.update_in_place(0, live, &too_long),
            Err(TablePageError::DoesNotFit {
                len: too_long.len()
            })
        );
        let max = vec![0u8; TablePageMut::MAX_TUPLE_SIZE];
        page.insert(b"x").unwrap();
        assert!(matches!(
            page.update_in_place(0, live, &max),
            Err(TablePageError::UpdateDoesNotFit { slot: 0, .. })
        ));
        let after = TablePageRef::new(&buf[..]).unwrap();
        let s0 = after.slot(0).unwrap();
        assert_eq!(after.tuple_bytes(&s0).unwrap(), b"old");
        assert_eq!(s0.meta, TupleMeta::default());
        assert_eq!(buf[PAGE_SIZE - 3..], before[PAGE_SIZE - 3..]);
    }

    #[test]
    fn mark_deleted_rejects_an_overflowing_deleted_counter() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"x").unwrap();
        put_u16_le(&mut buf, NUM_DELETED_OFF, u16::MAX);
        let mut page = TablePageMut::new(&mut buf[..]).unwrap();
        assert_eq!(
            page.mark_deleted(0),
            Err(TablePageError::CorruptDeletedCounter {
                count: u16::MAX,
                slot: 0
            }),
            "counter overflow is reported"
        );
        assert!(
            !page.slot(0).unwrap().meta.is_deleted,
            "a rejected delete leaves the slot live"
        );
    }

    #[test]
    fn update_meta_rejects_a_corrupt_deleted_counter() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"x").unwrap();
        page.mark_deleted(0).unwrap();
        put_u16_le(&mut buf, NUM_DELETED_OFF, 0);
        let mut page = TablePageMut::new(&mut buf[..]).unwrap();
        let live = TupleMeta {
            is_deleted: false,
            ..page.slot(0).unwrap().meta
        };
        assert_eq!(
            page.update_meta(0, live),
            Err(TablePageError::CorruptDeletedCounter { count: 0, slot: 0 })
        );
        assert!(page.slot(0).unwrap().meta.is_deleted);
        assert_eq!(page.num_deleted_tuples(), 0);
    }

    #[test]
    fn update_in_place_with_a_corrupt_counter_writes_nothing() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"old").unwrap();
        page.mark_deleted(0).unwrap();
        put_u16_le(&mut buf, NUM_DELETED_OFF, 0);
        let mut page = TablePageMut::new(&mut buf[..]).unwrap();
        let before = page.slot(0).unwrap();
        let live = TupleMeta {
            is_deleted: false,
            ..before.meta
        };
        assert_eq!(
            page.update_in_place(0, live, b"new"),
            Err(TablePageError::CorruptDeletedCounter { count: 0, slot: 0 })
        );
        let after = page.slot(0).unwrap();
        assert_eq!(after.meta, before.meta);
        assert_eq!(page.tuple_bytes(&after).unwrap(), b"old");
    }

    #[test]
    fn insert_rejects_when_full() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        let big = vec![1u8; TablePageMut::MAX_TUPLE_SIZE];
        page.insert(&big).unwrap();
        assert!(!page.can_insert(0));
        assert_eq!(page.insert(b""), Err(TablePageError::DoesNotFit { len: 0 }));
        let meta = page.slot(0).unwrap();
        assert_eq!(page.tuple_bytes(&meta).unwrap(), &big[..]);
    }

    #[test]
    fn new_rejects_corrupt_header() {
        let mut buf = [0u8; PAGE_SIZE];
        TablePage::init(&mut buf[..]);
        put_u16_le(&mut buf, NUM_TUPLES_OFF, u16::MAX);
        assert!(
            matches!(
                TablePageRef::new(&buf[..]),
                Err(TablePageError::CorruptHeader {
                    free_space_offset: PAGE_SIZE,
                    ..
                })
            ),
            "more slots than fit before the tuple area"
        );
    }

    #[test]
    fn new_rejects_free_space_offset_past_page_end() {
        let mut buf = [0u8; PAGE_SIZE];
        TablePage::init(&mut buf[..]);
        put_u32_le(&mut buf, FREE_SPACE_OFF, PAGE_SIZE as u32 + 1);
        assert!(matches!(
            TablePageRef::new(&buf[..]),
            Err(TablePageError::CorruptHeader {
                free_space_offset,
                ..
            }) if free_space_offset == PAGE_SIZE + 1
        ));
    }

    #[test]
    fn new_rejects_a_buffer_that_is_not_one_page() {
        let buf = [0u8; PAGE_SIZE - 1];
        assert!(matches!(
            TablePageRef::new(&buf[..]),
            Err(TablePageError::WrongSize { len }) if len == PAGE_SIZE - 1
        ));
    }

    #[test]
    fn insert_rejects_a_page_with_u16_max_slots() {
        let mut buf = [0u8; PAGE_SIZE];
        TablePage::init(&mut buf[..]);
        put_u16_le(&mut buf, NUM_TUPLES_OFF, u16::MAX);
        put_u32_le(&mut buf, FREE_SPACE_OFF, u32::MAX);
        let mut page = TablePage { buf: &mut buf[..] };
        assert_eq!(page.insert(b""), Err(TablePageError::SlotCountOverflow));
    }

    #[test]
    fn tuple_bytes_rejects_out_of_range_slot() {
        let mut buf = [0u8; PAGE_SIZE];
        let mut page = TablePage::init(&mut buf[..]);
        page.insert(b"abc").unwrap();
        let bogus = Slot {
            offset: 10,
            size: 4,
            capacity: 4,
            meta: TupleMeta::default(),
        };
        assert_eq!(
            page.tuple_bytes(&bogus),
            Err(TablePageError::SlotOutOfBounds {
                offset: 10,
                size: 4,
                capacity: 4,
                tuple_area_start: page.free_space_offset()
            })
        );
        let overfull = Slot {
            size: 5,
            ..page.slot(0).unwrap()
        };
        assert!(matches!(
            page.tuple_bytes(&overfull),
            Err(TablePageError::SlotOutOfBounds { size: 5, .. })
        ));
    }
}
