use crate::buf::{BufReader, BufWriter, BufferError};
use crate::catalog::index_schema::IndexKey;
use crate::disk::PAGE_SIZE;
use crate::index::page_codec::{
    COMMON_HEADER_BYTES, KIND_HEADER, KIND_INTERNAL, KIND_LEAF, KIND_OFFSET, PageCodecError,
    U32_BYTES, U64_BYTES, decode_page_id, decode_record_id, encode_page_id, encode_record_id,
    read_common_header, write_common_header,
};
use crate::index::{InsertError, RemoveError};
use crate::table_heap::RecordId;
use log::{error, info};
use std::collections::VecDeque;
use std::convert::TryFrom;

pub type PageId = usize;

pub const INVALID_PAGE_ID: PageId = usize::MAX;
pub const NUM_TOMBSTONES: u32 = 256;

pub(crate) const PAGE_ID_ENCODED_BYTES: usize = U64_BYTES;
pub(crate) const RECORD_ID_ENCODED_BYTES: usize = 2 * U64_BYTES;
const TOMBSTONE_COUNT_ENCODED_BYTES: usize = U32_BYTES;
pub(crate) const TOMBSTONE_SLOT_ENCODED_BYTES: usize = U32_BYTES;

const SIZE_FIELD_BYTES: usize = U32_BYTES;
const MAX_SIZE_FIELD_BYTES: usize = U32_BYTES;
const CHILD_COUNT_FIELD_BYTES: usize = U32_BYTES;
const RESERVED_FIELD_BYTES: usize = U32_BYTES;
const KEY_LEN_FIELD_BYTES: usize = U32_BYTES;

pub(crate) const INTERNAL_FIXED_HEADER_BYTES: usize = COMMON_HEADER_BYTES
    + SIZE_FIELD_BYTES
    + MAX_SIZE_FIELD_BYTES
    + CHILD_COUNT_FIELD_BYTES
    + RESERVED_FIELD_BYTES;
pub(crate) const LEAF_FIXED_HEADER_BYTES: usize =
    COMMON_HEADER_BYTES + SIZE_FIELD_BYTES + MAX_SIZE_FIELD_BYTES + PAGE_ID_ENCODED_BYTES;
pub(crate) const HEADER_FIXED_HEADER_BYTES: usize = COMMON_HEADER_BYTES
    + KEY_LEN_FIELD_BYTES
    + RESERVED_FIELD_BYTES
    + PAGE_ID_ENCODED_BYTES
    + PAGE_ID_ENCODED_BYTES;

#[derive(Debug, PartialEq)]
pub enum PageType {
    Internal,
    Leaf,
}

#[derive(Debug)]
pub struct Base {
    page_type: PageType,
    size: u32,
    max_size: u32,
}

impl Base {
    pub fn new(page_type: PageType, max_size: u32) -> Self {
        Self {
            page_type,
            size: 0,
            max_size,
        }
    }

    pub fn is_leaf_page(&self) -> bool {
        self.page_type == PageType::Leaf
    }

    pub fn min_size(&self) -> u32 {
        self.max_size / 2
    }
}

pub enum Page {
    Internal(InternalPage),
    Leaf(LeafPage),
}

pub enum PageKind {
    Internal,
    Leaf,
}

impl Page {
    pub fn decode(buf: &[u8], key_len: u32) -> Result<Self, PageCodecError> {
        match buf[KIND_OFFSET] {
            KIND_INTERNAL => Ok(Self::Internal(InternalPage::decode(buf, key_len)?)),
            KIND_LEAF => Ok(Self::Leaf(LeafPage::decode(buf, key_len)?)),
            k => Err(PageCodecError::InvalidPageKind(k)),
        }
    }

    pub fn kind(buf: &[u8]) -> Result<PageKind, PageCodecError> {
        match buf[KIND_OFFSET] {
            KIND_INTERNAL => Ok(PageKind::Internal),
            KIND_LEAF => Ok(PageKind::Leaf),
            k => Err(PageCodecError::InvalidPageKind(k)),
        }
    }
}

#[derive(Debug)]
pub struct InternalPage {
    base: Base,
    pub children: Vec<PageId>,
    pub keys: Vec<IndexKey>,
}

impl InternalPage {
    pub fn new(max_size: u32) -> Self {
        Self {
            base: Base::new(PageType::Internal, max_size),
            children: Vec::new(),
            keys: Vec::new(),
        }
    }

    /// Returns the PageId of the child at the given index
    pub fn get_child(&self, idx: usize) -> Option<&PageId> {
        self.children.get(idx)
    }

    /// Returns the most keys an internal page with `key_len`-byte keys can encode into one page.
    pub fn max_size_for_layout(key_len: usize) -> Option<u32> {
        if key_len == 0 {
            return None;
        }
        let leftmost_child_bytes = PAGE_ID_ENCODED_BYTES;
        let bytes_per_key_and_right_child = key_len.checked_add(PAGE_ID_ENCODED_BYTES)?;
        let space_for_keys =
            PAGE_SIZE.checked_sub(INTERNAL_FIXED_HEADER_BYTES + leftmost_child_bytes)?;
        u32::try_from(space_for_keys / bytes_per_key_and_right_child)
            .ok()
            .filter(|&n| n > 0)
    }

    pub fn validate(&self) {
        if self.children.is_empty() && self.keys.is_empty() {
            return;
        }
        assert_eq!(
            self.children.len(),
            self.keys.len() + 1,
            "internal page invariant violated: children.len() must equal keys.len() + 1"
        );
        assert_eq!(
            self.base.size as usize,
            self.keys.len(),
            "internal page invariant violated: base.size must match keys.len()"
        );
        assert!(
            (self.keys.len() as u32) <= self.base.max_size.saturating_add(1),
            "internal page invariant violated: keys.len() must be <= max_size (+1 overflow)"
        );
        assert!(
            self.keys.windows(2).all(|w| w[0] <= w[1]),
            "internal page invariant violated: keys must be sorted"
        );
    }

    /// Returns the index `i` of the child whose subtree holds `key`: the number of keys `<= key`,
    /// so `keys[i - 1] <= key < keys[i]` wherever those keys exist.
    pub fn find_child_index(&self, key: &IndexKey) -> usize {
        self.keys.partition_point(|k| k <= key)
    }

    /// Inserts `sep_key` and `right_child`, with the child directly to the right of the key.
    pub fn insert_separator(&mut self, sep_key: IndexKey, right_child: PageId) {
        let left_child_index = self.find_child_index(&sep_key);
        assert!(
            left_child_index < self.children.len(),
            "left_child_index out of bounds"
        );
        self.keys.insert(left_child_index, sep_key);
        self.children.insert(left_child_index + 1, right_child);
        self.base.size = self.keys.len() as u32;
        self.validate();
    }

    pub fn insert_key(&mut self, index: usize, key: IndexKey) {
        self.keys.insert(index, key);
        self.base.size = self.keys.len() as u32;
    }

    pub fn remove_key(&mut self, index: usize) -> IndexKey {
        let key = self.keys.remove(index);
        self.base.size = self.keys.len() as u32;
        key
    }

    pub fn min_size(&self) -> u32 {
        self.base.min_size()
    }

    /// Moves the keys and children above the middle key into the empty `right` page, removes the
    /// middle key from this page and returns it for the parent.
    pub fn split_into(&mut self, right: &mut InternalPage) -> IndexKey {
        assert!(
            right.children.is_empty() && right.keys.is_empty(),
            "right page must be empty before split"
        );
        self.validate();

        let mid = self.keys.len() / 2;

        right.keys = self.keys.split_off(mid + 1);
        let sep_key = self
            .keys
            .pop()
            .expect("internal page must have at least one key");
        right.children = self.children.split_off(mid + 1);

        self.base.size = self.keys.len() as u32;
        right.base.size = right.keys.len() as u32;

        self.validate();
        right.validate();

        sep_key
    }

    pub fn absorb_right(&mut self, sep_key: IndexKey, right: &mut InternalPage) {
        self.validate();
        right.validate();

        self.keys.push(sep_key);
        self.keys.append(&mut right.keys);
        self.children.append(&mut right.children);
        self.base.size = self.keys.len() as u32;

        self.validate();
    }

    pub fn encode(&self, buf: &mut [u8], key_len: u32) -> Result<(), PageCodecError> {
        let key_len = usize::try_from(key_len)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?;
        if key_len == 0 {
            return Err(PageCodecError::Malformed("key_len must be > 0"));
        }
        self.validate();
        for k in self.keys.iter() {
            if k.0.len() != key_len {
                return Err(PageCodecError::InvalidKeyLen {
                    expected: key_len,
                    actual: k.0.len(),
                });
            }
        }

        let child_count = self.children.len();
        if child_count != 0 && child_count != self.keys.len() + 1 {
            return Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != keys.len + 1",
            ));
        }
        info!(
            "bptree/page_encode kind=internal keys={} children={}",
            self.keys.len(),
            child_count
        );

        let header_bytes = INTERNAL_FIXED_HEADER_BYTES;
        let child_bytes = child_count
            .checked_mul(PAGE_ID_ENCODED_BYTES)
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        let key_bytes = self
            .keys
            .len()
            .checked_mul(key_len)
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        let needed = header_bytes
            .checked_add(child_bytes)
            .and_then(|n| n.checked_add(key_bytes))
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        if needed > PAGE_SIZE {
            return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
                needed,
                actual: PAGE_SIZE,
            }));
        };

        buf.fill(0);
        let mut w = BufWriter::new(buf)?;
        write_common_header(&mut w, KIND_INTERNAL)?;
        w.write_u32_le(self.base.size)?;
        w.write_u32_le(self.base.max_size)?;
        w.write_u32_le(
            u32::try_from(child_count)
                .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
        )?;
        w.write_u32_le(0)?;

        for &child in self.children.iter() {
            w.write_u64_le(encode_page_id(child)?)?;
        }
        for k in self.keys.iter() {
            w.write_bytes(&k.0)?;
        }
        debug_assert_eq!(w.pos(), needed);
        Ok(())
    }

    pub fn decode(buf: &[u8], key_len: u32) -> Result<Self, PageCodecError> {
        let key_len = usize::try_from(key_len)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?;
        if key_len == 0 {
            return Err(PageCodecError::Malformed("key_len must be > 0"));
        }
        let mut r = BufReader::new(buf)?;
        let kind = read_common_header(&mut r)?;
        if kind != KIND_INTERNAL {
            return Err(PageCodecError::WrongPageKind {
                expected: KIND_INTERNAL,
                actual: kind,
            });
        }
        let size = r.read_u32_le()?;
        let max_size = r.read_u32_le()?;
        let child_count = r.read_u32_le()? as usize;
        let _reserved = r.read_u32_le()?;
        info!(
            "bptree/page_decode kind=internal size={} max_size={} children={}",
            size, max_size, child_count
        );

        if child_count == 0 {
            if size != 0 {
                return Err(PageCodecError::Malformed(
                    "internal page has keys but zero children",
                ));
            }
            let mut page = InternalPage::new(max_size);
            page.base.size = 0;
            return Ok(page);
        }
        if child_count != (size as usize) + 1 {
            return Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != size + 1",
            ));
        }

        let child_bytes = child_count
            .checked_mul(PAGE_ID_ENCODED_BYTES)
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        let key_bytes = (size as usize)
            .checked_mul(key_len)
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        let needed = INTERNAL_FIXED_HEADER_BYTES
            .checked_add(child_bytes)
            .and_then(|n| n.checked_add(key_bytes))
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        if needed > PAGE_SIZE {
            return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
                needed,
                actual: PAGE_SIZE,
            }));
        }
        if r.remaining() < needed.saturating_sub(r.pos) {
            return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
                needed,
                actual: PAGE_SIZE,
            }));
        }

        let mut children = Vec::with_capacity(child_count);
        for _ in 0..child_count {
            children.push(decode_page_id(r.read_u64_le()?)?);
        }

        let mut keys = Vec::with_capacity(child_count);
        for _ in 0..size {
            let bytes = r.read_bytes(key_len)?;
            keys.push(IndexKey(bytes.to_vec()));
        }

        let mut page = InternalPage::new(max_size);
        page.base.size = size;
        page.children = children;
        page.keys = keys;
        page.validate();
        Ok(page)
    }

    pub fn is_full(&self) -> bool {
        self.base.size == self.base.max_size
    }

    pub fn is_underfull(&self) -> bool {
        self.base.size < self.base.min_size()
    }

    /// Removes the separator key between `children[left_child_index]` and
    /// `children[left_child_index + 1]`, and returns it with that right child.
    pub fn remove_separator(&mut self, left_child_index: usize) -> (IndexKey, PageId) {
        assert!(
            left_child_index < self.keys.len(),
            "left_child_index out of bounds for keys"
        );
        assert!(
            left_child_index + 1 < self.children.len(),
            "left_child_index out of bounds for children"
        );
        let sep_key = self.keys.remove(left_child_index);
        let right_child = self.children.remove(left_child_index + 1);
        self.base.size = self.keys.len() as u32;
        self.validate();
        (sep_key, right_child)
    }

    pub fn pop_last_child_and_key(&mut self) -> (IndexKey, PageId) {
        let child = self.children.pop().expect("internal page has no children");
        let key = self.keys.pop().expect("internal page has no keys");
        self.base.size = self.keys.len() as u32;
        self.validate();
        (key, child)
    }

    pub fn pop_first_child_and_key(&mut self) -> (IndexKey, PageId) {
        let child = self.children.remove(0);
        let key = self.keys.remove(0);
        self.base.size = self.keys.len() as u32;
        self.validate();
        (key, child)
    }

    pub fn replace_key(&mut self, idx: usize, new_key: IndexKey) -> IndexKey {
        assert!(idx < self.keys.len(), "replace_key index out of bounds");
        let old = std::mem::replace(&mut self.keys[idx], new_key);
        self.validate();
        old
    }

    /// Inserts a new leftmost child and separator key.
    /// After insertion, `key` separates `child` (new `children[0]`) and the old `children[0]`.
    pub fn prepend_child_and_key(&mut self, key: IndexKey, child: PageId) {
        self.children.insert(0, child);
        self.keys.insert(0, key);
        self.base.size = self.keys.len() as u32;
        self.validate();
    }

    /// Appends a separator key and a new rightmost child.
    /// After insertion, `key` separates the old rightmost child and `child` (new rightmost).
    pub fn append_key_and_child(&mut self, key: IndexKey, child: PageId) {
        self.keys.push(key);
        self.children.push(child);
        self.base.size = self.keys.len() as u32;
        self.validate();
    }

    pub fn is_overfull(&self) -> bool {
        self.base.size > self.base.max_size
    }

    pub fn would_be_underfull(&self) -> bool {
        self.base.size == self.base.min_size()
    }
}

#[derive(Debug)]
pub struct LeafPage {
    base: Base,
    pub keys: Vec<IndexKey>,
    pub record_ids: Vec<RecordId>,
    /// Page id of the next leaf to the right, or `INVALID_PAGE_ID` for the last leaf.
    pub next_leaf_id: PageId,
    /// Slots of deleted keys still stored in the page, oldest deletion first.
    pub tombstones: VecDeque<u32>,
    pub tombstone_capacity: usize,
}

impl LeafPage {
    /// Returns the most entries a leaf with `key_len`-byte keys can encode into one page,
    /// leaving room for an empty tombstone section.
    pub fn max_size_for_layout(key_len: usize) -> Option<u32> {
        if key_len == 0 {
            return None;
        }
        let entry_bytes = key_len.checked_add(RECORD_ID_ENCODED_BYTES)?;
        let space_for_entries =
            PAGE_SIZE.checked_sub(LEAF_FIXED_HEADER_BYTES + TOMBSTONE_COUNT_ENCODED_BYTES)?;
        u32::try_from(space_for_entries / entry_bytes)
            .ok()
            .filter(|&n| n > 0)
    }

    /// Returns how many tombstones (at most `requested`) fit beside a full leaf of `max_size` entries.
    pub fn tombstone_capacity_for_layout(max_size: u32, key_len: u32, requested: u32) -> u32 {
        Self::tombstone_slots_beside_full_leaf(max_size, key_len)
            .map_or(0, |fit| fit.min(requested))
    }

    fn tombstone_slots_beside_full_leaf(max_size: u32, key_len: u32) -> Option<u32> {
        let key_len = usize::try_from(key_len).ok().filter(|&k| k > 0)?;
        let entry_bytes = key_len.checked_add(RECORD_ID_ENCODED_BYTES)?;
        let entries_bytes = usize::try_from(max_size).ok()?.checked_mul(entry_bytes)?;
        let live_bytes = LEAF_FIXED_HEADER_BYTES.checked_add(entries_bytes)?;
        let space_for_slots = PAGE_SIZE
            .checked_sub(live_bytes)?
            .checked_sub(TOMBSTONE_COUNT_ENCODED_BYTES)?;
        u32::try_from(space_for_slots / TOMBSTONE_SLOT_ENCODED_BYTES).ok()
    }

    pub fn new(max_size: u32, num_tombstones: u32) -> Self {
        let tombstone_capacity = usize::try_from(num_tombstones).unwrap_or(0);
        Self {
            base: Base::new(PageType::Leaf, max_size),
            keys: Vec::new(),
            record_ids: Vec::new(),
            next_leaf_id: INVALID_PAGE_ID,
            tombstones: VecDeque::with_capacity(tombstone_capacity),
            tombstone_capacity,
        }
    }

    pub fn validate(&self) {
        assert_eq!(
            self.base.size as usize,
            self.keys.len(),
            "leaf page invariant violated: base.size must match keys.len()"
        );
        assert_eq!(
            self.keys.len(),
            self.record_ids.len(),
            "leaf page invariant violated: keys.len() must equal record_ids.len()"
        );
        assert!(
            (self.keys.len() as u32) <= self.base.max_size.saturating_add(1),
            "leaf page invariant violated: keys.len() must be <= max_size (+1 overflow)"
        );
        assert!(
            self.keys.windows(2).all(|w| w[0] <= w[1]),
            "leaf page invariant violated: keys must be sorted"
        );
        assert!(
            self.tombstones.len() <= self.tombstone_capacity,
            "leaf page invariant violated: tombstones exceeds capacity"
        );
        assert!(
            self.tombstones
                .iter()
                .all(|&i| (i as usize) < self.keys.len()),
            "leaf page invariant violated: tombstone index out of bounds"
        );
    }

    pub fn lower_bound(&self, key: &IndexKey) -> usize {
        self.keys.partition_point(|k| k < key)
    }

    pub fn upper_bound(&self, key: &IndexKey) -> usize {
        self.keys.partition_point(|k| k <= key)
    }

    /// Returns the record id stored for `key`, or `None` if it is absent or tombstoned.
    pub fn get(&self, key: &IndexKey) -> Option<RecordId> {
        let pos = match self.keys.binary_search(key) {
            Ok(pos) => pos,
            Err(_) => {
                error!("get(): could not find key in page; key={:?}", key);
                return None;
            }
        };
        if self.is_tombstoned(pos as u32) {
            error!("get(): key is tombstoned; key={:?}", key);
            return None;
        }
        let rid = &self.record_ids[pos];
        Some(RecordId {
            page_id: rid.page_id,
            slot_id: rid.slot_id,
        })
    }

    pub fn is_tombstoned(&self, slot: u32) -> bool {
        self.tombstones.iter().any(|&s| s == slot)
    }

    fn remove_tombstone(&mut self, slot: u32) -> bool {
        if let Some(pos) = self.tombstones.iter().position(|&s| s == slot) {
            self.tombstones.remove(pos);
            return true;
        }
        false
    }

    fn adjust_tombstones_after_insert(&mut self, insert_at: u32) {
        for s in self.tombstones.iter_mut() {
            if *s >= insert_at {
                *s += 1;
            }
        }
    }

    fn adjust_tombstones_after_remove(&mut self, removed_at: u32) {
        self.tombstones.retain(|&slot| slot != removed_at);
        for slot in self.tombstones.iter_mut() {
            if *slot > removed_at {
                *slot -= 1;
            }
        }
    }

    fn physical_remove_at(&mut self, slot: u32) {
        let idx = slot as usize;
        self.keys.remove(idx);
        self.record_ids.remove(idx);
        self.base.size = self.keys.len() as u32;
        self.adjust_tombstones_after_remove(slot);
    }

    pub fn encode(&self, buf: &mut [u8], key_len: u32) -> Result<(), PageCodecError> {
        let key_len = usize::try_from(key_len)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?;
        if key_len == 0 {
            return Err(PageCodecError::Malformed("key_len must be > 0"));
        }
        self.validate();
        for k in self.keys.iter() {
            if k.0.len() != key_len {
                return Err(PageCodecError::InvalidKeyLen {
                    expected: key_len,
                    actual: k.0.len(),
                });
            }
        }

        let n = self.keys.len();
        info!(
            "bptree/page_encode kind=leaf keys={} tombstones={}",
            n,
            self.tombstones.len()
        );
        let entry_bytes = key_len
            .checked_add(RECORD_ID_ENCODED_BYTES)
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        let live_needed = LEAF_FIXED_HEADER_BYTES
            .checked_add(
                n.checked_mul(entry_bytes)
                    .ok_or(PageCodecError::BufferError(BufferError::Overflow))?,
            )
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        if live_needed > PAGE_SIZE {
            return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
                needed: live_needed,
                actual: PAGE_SIZE,
            }));
        }

        let available = PAGE_SIZE.saturating_sub(live_needed);
        let tombstone_bytes = TOMBSTONE_COUNT_ENCODED_BYTES
            .checked_add(
                self.tombstones
                    .len()
                    .checked_mul(TOMBSTONE_SLOT_ENCODED_BYTES)
                    .ok_or(PageCodecError::BufferError(BufferError::Overflow))?,
            )
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        if tombstone_bytes > available {
            return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
                needed: live_needed
                    .checked_add(tombstone_bytes)
                    .ok_or(PageCodecError::BufferError(BufferError::Overflow))?,
                actual: PAGE_SIZE,
            }));
        }

        buf.fill(0);
        let mut w = BufWriter::new(buf)?;
        write_common_header(&mut w, KIND_LEAF)?;
        w.write_u32_le(self.base.size)?;
        w.write_u32_le(self.base.max_size)?;
        w.write_u64_le(encode_page_id(self.next_leaf_id)?)?;

        for (k, rid) in self.keys.iter().zip(self.record_ids.iter()) {
            w.write_bytes(&k.0)?;
            let (pid, sid) = encode_record_id(rid)?;
            w.write_u64_le(pid)?;
            w.write_u64_le(sid)?;
        }

        w.write_u32_le(
            u32::try_from(self.tombstones.len())
                .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
        )?;
        for &slot in self.tombstones.iter() {
            w.write_u32_le(slot)?;
        }

        debug_assert!(w.pos() <= PAGE_SIZE);
        Ok(())
    }

    pub fn decode(buf: &[u8], key_len: u32) -> Result<Self, PageCodecError> {
        let key_len = usize::try_from(key_len)
            .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?;
        if key_len == 0 {
            return Err(PageCodecError::Malformed("key_len must be > 0"));
        }
        let mut r = BufReader::new(buf)?;
        let kind = read_common_header(&mut r)?;
        if kind != KIND_LEAF {
            return Err(PageCodecError::WrongPageKind {
                expected: KIND_LEAF,
                actual: kind,
            });
        }
        let size = r.read_u32_le()?;
        let max_size = r.read_u32_le()?;
        let next_leaf_id = decode_page_id(r.read_u64_le()?)?;
        info!(
            "bptree/page_decode kind=leaf size={} max_size={} next_leaf_id={}",
            size, max_size, next_leaf_id
        );

        let n = size as usize;
        let entry_bytes = key_len
            .checked_add(RECORD_ID_ENCODED_BYTES)
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        let live_needed = LEAF_FIXED_HEADER_BYTES
            .checked_add(
                n.checked_mul(entry_bytes)
                    .ok_or(PageCodecError::BufferError(BufferError::Overflow))?,
            )
            .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
        if live_needed > PAGE_SIZE {
            return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
                needed: live_needed,
                actual: PAGE_SIZE,
            }));
        }

        let mut keys = Vec::with_capacity(n);
        let mut record_ids = Vec::with_capacity(n);
        for _ in 0..n {
            let bytes = r.read_bytes(key_len)?;
            keys.push(IndexKey(bytes.to_vec()));
            let pid = r.read_u64_le()?;
            let sid = r.read_u64_le()?;
            record_ids.push(decode_record_id(pid, sid)?);
        }

        let cap = LeafPage::tombstone_capacity_for_layout(
            max_size,
            u32::try_from(key_len)
                .map_err(|_| PageCodecError::BufferError(BufferError::Overflow))?,
            NUM_TOMBSTONES,
        );
        let mut page = LeafPage::new(max_size, cap);
        page.base.size = size;
        page.keys = keys;
        page.record_ids = record_ids;
        page.next_leaf_id = next_leaf_id;

        let has_tombstone_section = r.remaining() >= TOMBSTONE_COUNT_ENCODED_BYTES;
        if has_tombstone_section {
            let tombstone_count = r.read_u32_le()? as usize;
            if tombstone_count > 0 {
                let needed = tombstone_count
                    .checked_mul(TOMBSTONE_SLOT_ENCODED_BYTES)
                    .ok_or(PageCodecError::BufferError(BufferError::Overflow))?;
                if r.remaining() < needed {
                    return Err(PageCodecError::Malformed(
                        "leaf tombstones section exceeds remaining bytes",
                    ));
                }
                let mut tombstones = VecDeque::with_capacity(tombstone_count);
                for _ in 0..tombstone_count {
                    tombstones.push_back(r.read_u32_le()?);
                }
                if tombstones.len() > page.tombstone_capacity {
                    return Err(PageCodecError::Malformed(
                        "leaf tombstones count exceeds tombstone capacity",
                    ));
                }
                if tombstones.iter().any(|&i| (i as usize) >= page.keys.len()) {
                    return Err(PageCodecError::Malformed(
                        "leaf tombstones index out of bounds",
                    ));
                }
                page.tombstones = tombstones;
            }
        }

        page.validate();
        Ok(page)
    }
}

impl LeafPage {
    /// Inserts `key`, reviving it if it is tombstoned.
    ///
    /// Returns `InsertError::DuplicateKey` if `key` is already live in this page.
    pub fn insert(&mut self, key: IndexKey, record_id: RecordId) -> Result<(), InsertError> {
        match self.keys.binary_search(&key) {
            Ok(pos) => {
                let slot = Self::slot(pos);
                if !self.is_tombstoned(slot) {
                    error!("attempted to insert duplicate key into leaf page");
                    return Err(InsertError::DuplicateKey);
                }
                self.record_ids[pos] = record_id;
                self.remove_tombstone(slot);
                self.validate();
                Ok(())
            }
            Err(pos) => {
                let insert_at = Self::slot(pos);
                self.keys.insert(pos, key);
                self.record_ids.insert(pos, record_id);
                self.adjust_tombstones_after_insert(insert_at);
                self.base.size = self.keys.len() as u32;
                self.validate();
                Ok(())
            }
        }
    }

    /// Splits this leaf into `right` and returns the separator key for the parent.
    /// For leaf pages, the separator key is the first key in the new right page and remains in the leaf.
    pub fn split_into(&mut self, right: &mut LeafPage, right_page_id: PageId) -> IndexKey {
        assert!(
            right.keys.is_empty() && right.record_ids.is_empty(),
            "right page must be empty before split"
        );
        self.validate();
        assert!(!self.keys.is_empty(), "cannot split an empty leaf page");

        let mid = self.keys.len() / 2;
        let mid_u32 = Self::slot(mid);

        right.keys = self.keys.split_off(mid);
        right.record_ids = self.record_ids.split_off(mid);

        let mut left_tombstones = VecDeque::with_capacity(self.tombstones.len());
        let mut right_tombstones = VecDeque::with_capacity(self.tombstones.len());
        for slot in self.tombstones.drain(..) {
            if slot < mid_u32 {
                left_tombstones.push_back(slot);
            } else {
                right_tombstones.push_back(slot - mid_u32);
            }
        }
        self.tombstones = left_tombstones;
        right.tombstones = right_tombstones;

        right.next_leaf_id = self.next_leaf_id;
        self.next_leaf_id = right_page_id;

        self.base.size = self.keys.len() as u32;
        right.base.size = right.keys.len() as u32;

        self.validate();
        right.validate();

        right.keys[0].clone()
    }

    /// Appends `right`'s entries to this leaf. `right`'s tombstones count as newer than this
    /// page's, so when the merged tombstones overflow, this page's oldest are purged first.
    pub fn absorb_right(&mut self, right: &mut LeafPage) {
        self.validate();
        right.validate();

        let left_len = Self::slot(self.keys.len());
        self.keys.append(&mut right.keys);
        self.record_ids.append(&mut right.record_ids);
        self.next_leaf_id = right.next_leaf_id;
        self.base.size = self.keys.len() as u32;

        for slot in right.tombstones.drain(..) {
            self.tombstones.push_back(slot + left_len);
        }
        while self.tombstones.len() > self.tombstone_capacity {
            self.purge_oldest_tombstone();
        }

        self.validate();
    }

    pub fn is_full(&self) -> bool {
        self.base.size == self.base.max_size
    }

    pub fn is_overfull(&self) -> bool {
        self.base.size > self.base.max_size
    }

    pub fn min_size(&self) -> u32 {
        self.base.min_size()
    }

    pub fn is_underfull(&self) -> bool {
        self.base.size < self.base.min_size()
    }

    pub fn would_be_underfull(&self) -> bool {
        self.base.size == self.base.min_size()
    }

    pub fn key_exists(&self, key: &IndexKey) -> bool {
        let found = self.keys.binary_search(key).is_ok();
        if !found {
            error!("could not find key in page: {:?}", key);
        }
        found
    }

    pub fn remove_key(&mut self, key: &IndexKey) -> Result<(), RemoveError> {
        info!("attempting to removing key from page: {:?}", key);
        let pos = match self.keys.binary_search(key) {
            Ok(pos) => pos,
            Err(_) => {
                error!("could not find key in page: {:?}", key);
                return Err(RemoveError::KeyNotFound);
            }
        };
        let mut slot = Self::slot(pos);
        if self.is_tombstoned(slot) {
            return Err(RemoveError::KeyNotFound);
        }

        let page_keeps_no_tombstones = self.tombstone_capacity == 0;
        if page_keeps_no_tombstones {
            self.physical_remove_at(slot);
            self.validate();
            return Ok(());
        }

        if self.tombstones.len() == self.tombstone_capacity {
            let purged = self.purge_oldest_tombstone();
            if purged < slot {
                slot -= 1;
            }
        }

        self.tombstones.push_back(slot);

        self.validate();
        Ok(())
    }

    fn purge_oldest_tombstone(&mut self) -> u32 {
        let oldest = self
            .tombstones
            .pop_front()
            .expect("purge_oldest_tombstone called with no tombstones");
        self.physical_remove_at(oldest);
        oldest
    }

    fn slot(index: usize) -> u32 {
        u32::try_from(index).expect("leaf slot index exceeds u32")
    }

    pub fn pop_first(&mut self) -> Option<(IndexKey, RecordId, bool)> {
        if self.keys.is_empty() {
            return None;
        }
        let was_tombstoned = self.is_tombstoned(0);
        let key = self.keys.remove(0);
        let rid = self.record_ids.remove(0);
        self.base.size = self.keys.len() as u32;
        self.adjust_tombstones_after_remove(0);
        self.validate();
        Some((key, rid, was_tombstoned))
    }

    pub fn pop_last(&mut self) -> Option<(IndexKey, RecordId, bool)> {
        let slot = u32::try_from(self.keys.len().saturating_sub(1)).ok()?;
        let was_tombstoned = self.is_tombstoned(slot);
        let key = self.keys.pop()?;
        let rid = self.record_ids.pop().expect("record_ids must match keys");
        self.remove_tombstone(slot);
        self.base.size = self.keys.len() as u32;
        self.validate();
        Some((key, rid, was_tombstoned))
    }

    pub fn push_front(&mut self, key: IndexKey, record_id: RecordId, tombstoned: bool) {
        self.adjust_tombstones_after_insert(0);
        self.keys.insert(0, key);
        self.record_ids.insert(0, record_id);
        if tombstoned {
            self.tombstones.push_back(0);
        }
        self.base.size = self.keys.len() as u32;
        self.validate();
    }

    pub fn push_back(&mut self, key: IndexKey, record_id: RecordId, tombstoned: bool) {
        let slot = Self::slot(self.keys.len());
        self.keys.push(key);
        self.record_ids.push(record_id);
        if tombstoned {
            self.tombstones.push_back(slot);
        }
        self.base.size = self.keys.len() as u32;
        self.validate();
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IndexKey, &RecordId)> {
        self.keys
            .iter()
            .zip(self.record_ids.iter())
            .enumerate()
            .filter_map(|(i, (k, rid))| {
                if self.is_tombstoned(i as u32) {
                    None
                } else {
                    Some((k, rid))
                }
            })
    }
}

/// HeaderPage is a special page which is kept separate and not in the tree itself.
#[derive(Debug)]
pub struct HeaderPage {
    pub root_page_id: PageId,
    pub first_leaf_id: PageId,
    pub key_len: u32,
}

impl HeaderPage {
    pub fn encode(&self, buf: &mut [u8]) -> Result<(), PageCodecError> {
        if self.key_len == 0 {
            return Err(PageCodecError::Malformed("key_len must be > 0"));
        }
        info!(
            "bptree/page_encode kind=header root={} first_leaf={} key_len={}",
            self.root_page_id, self.first_leaf_id, self.key_len
        );
        buf.fill(0);
        let mut w = BufWriter::new(buf)?;
        write_common_header(&mut w, KIND_HEADER)?;
        w.write_u32_le(self.key_len)?;
        w.write_u32_le(0)?;
        w.write_u64_le(encode_page_id(self.root_page_id)?)?;
        w.write_u64_le(encode_page_id(self.first_leaf_id)?)?;
        debug_assert_eq!(w.pos(), HEADER_FIXED_HEADER_BYTES);
        Ok(())
    }

    pub fn decode(buf: &[u8]) -> Result<Self, PageCodecError> {
        let mut r = BufReader::new(buf)?;
        let kind = read_common_header(&mut r)?;
        if kind != KIND_HEADER {
            return Err(PageCodecError::WrongPageKind {
                expected: KIND_HEADER,
                actual: kind,
            });
        }
        let key_len = r.read_u32_le()?;
        let _reserved = r.read_u32_le()?;
        let root_page_id = decode_page_id(r.read_u64_le()?)?;
        let first_leaf_id = decode_page_id(r.read_u64_le()?)?;
        info!(
            "bptree/page_decode kind=header root={} first_leaf={} key_len={}",
            root_page_id, first_leaf_id, key_len
        );
        if key_len == 0 {
            return Err(PageCodecError::Malformed("header key_len must be > 0"));
        }
        Ok(Self {
            root_page_id,
            first_leaf_id,
            key_len,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::buf::BufferError;
    use crate::catalog::index_schema::IndexKey;
    use crate::disk::PAGE_SIZE;
    use crate::index::InsertError;
    use crate::index::b_plus_page::{
        HEADER_FIXED_HEADER_BYTES, HeaderPage, INTERNAL_FIXED_HEADER_BYTES, INVALID_PAGE_ID,
        InternalPage, LEAF_FIXED_HEADER_BYTES, LeafPage, NUM_TOMBSTONES, PAGE_ID_ENCODED_BYTES,
        RECORD_ID_ENCODED_BYTES, TOMBSTONE_COUNT_ENCODED_BYTES, TOMBSTONE_SLOT_ENCODED_BYTES,
    };
    use crate::index::page_codec::PageCodecError;
    use crate::table_heap::RecordId;
    use std::collections::VecDeque;

    #[test]
    fn test_internal_page_split() {
        let mut left = InternalPage::new(10);
        left.children = vec![1, 2, 3, 4, 5, 6];
        left.keys = vec![
            IndexKey(vec![10]),
            IndexKey(vec![20]),
            IndexKey(vec![30]),
            IndexKey(vec![40]),
            IndexKey(vec![50]),
        ];
        left.base.size = left.keys.len() as u32;
        left.validate();

        let mut right = InternalPage::new(10);
        let sep = left.split_into(&mut right);

        assert_eq!(sep, IndexKey(vec![30]));
        assert_eq!(left.keys, vec![IndexKey(vec![10]), IndexKey(vec![20])]);
        assert_eq!(left.children, vec![1, 2, 3]);
        assert_eq!(right.keys, vec![IndexKey(vec![40]), IndexKey(vec![50])]);
        assert_eq!(right.children, vec![4, 5, 6]);
    }

    #[test]
    fn test_leaf_page_split_and_merge() {
        let mut left = LeafPage::new(10, NUM_TOMBSTONES);
        left.keys = vec![
            IndexKey(vec![10]),
            IndexKey(vec![20]),
            IndexKey(vec![30]),
            IndexKey(vec![40]),
        ];
        left.record_ids = vec![
            RecordId {
                page_id: 1,
                slot_id: 0,
            },
            RecordId {
                page_id: 1,
                slot_id: 1,
            },
            RecordId {
                page_id: 1,
                slot_id: 2,
            },
            RecordId {
                page_id: 1,
                slot_id: 3,
            },
        ];
        left.base.size = left.keys.len() as u32;
        left.next_leaf_id = 999;
        left.tombstones = VecDeque::from([1u32, 3u32]);
        left.validate();

        let mut right = LeafPage::new(10, NUM_TOMBSTONES);
        let sep = left.split_into(&mut right, 123);

        assert_eq!(sep, IndexKey(vec![30]));
        assert_eq!(left.keys, vec![IndexKey(vec![10]), IndexKey(vec![20])]);
        assert_eq!(right.keys, vec![IndexKey(vec![30]), IndexKey(vec![40])]);
        assert_eq!(left.tombstones, VecDeque::from([1u32]));
        assert_eq!(right.tombstones, VecDeque::from([1u32]));
        assert_eq!(left.next_leaf_id, 123);
        assert_eq!(right.next_leaf_id, 999);

        left.absorb_right(&mut right);
        assert_eq!(
            left.keys,
            vec![
                IndexKey(vec![10]),
                IndexKey(vec![20]),
                IndexKey(vec![30]),
                IndexKey(vec![40])
            ]
        );
        assert_eq!(left.tombstones, VecDeque::from([1u32, 3u32]));
        assert_eq!(left.next_leaf_id, 999);
    }

    #[test]
    fn test_leaf_page_merge_treats_absorbed_tombstones_as_more_recent() {
        let mut left = LeafPage::new(10, 1);
        left.keys = vec![IndexKey(vec![10]), IndexKey(vec![20])];
        left.record_ids = vec![
            RecordId {
                page_id: 1,
                slot_id: 0,
            },
            RecordId {
                page_id: 1,
                slot_id: 1,
            },
        ];
        left.base.size = left.keys.len() as u32;
        left.tombstones = VecDeque::from([0u32]);
        left.validate();

        let mut right = LeafPage::new(10, 1);
        right.keys = vec![IndexKey(vec![30]), IndexKey(vec![40])];
        right.record_ids = vec![
            RecordId {
                page_id: 2,
                slot_id: 0,
            },
            RecordId {
                page_id: 2,
                slot_id: 1,
            },
        ];
        right.base.size = right.keys.len() as u32;
        right.tombstones = VecDeque::from([0u32]);
        right.validate();

        left.absorb_right(&mut right);
        assert_eq!(
            left.keys,
            vec![IndexKey(vec![20]), IndexKey(vec![30]), IndexKey(vec![40])]
        );
        assert_eq!(
            left.tombstones,
            VecDeque::from([1u32]),
            "only the absorbed tombstone for key 30 survives, at its shifted slot"
        );
        assert!(
            !left.key_exists(&IndexKey(vec![10])),
            "left's older tombstone is purged from the page"
        );
        assert!(left.get(&IndexKey(vec![30])).is_none());
    }

    #[test]
    fn test_internal_page_merge() {
        let mut left = InternalPage::new(10);
        left.children = vec![1, 2, 3];
        left.keys = vec![IndexKey(vec![10]), IndexKey(vec![20])];
        left.base.size = left.keys.len() as u32;

        let mut right = InternalPage::new(10);
        right.children = vec![5, 6, 7];
        right.keys = vec![IndexKey(vec![90]), IndexKey(vec![100])];
        right.base.size = right.keys.len() as u32;

        left.absorb_right(IndexKey(vec![50]), &mut right);

        assert_eq!(
            vec![
                IndexKey(vec![10]),
                IndexKey(vec![20]),
                IndexKey(vec![50]),
                IndexKey(vec![90]),
                IndexKey(vec![100]),
            ],
            left.keys
        );
        assert_eq!(vec![1, 2, 3, 5, 6, 7], left.children);
        assert_eq!(0, right.keys.len());
        assert_eq!(0, right.children.len());
    }

    #[test]
    fn test_header_page_encode_decode_roundtrip() {
        let mut buf = [0u8; PAGE_SIZE];
        let hdr = HeaderPage {
            root_page_id: 42,
            first_leaf_id: INVALID_PAGE_ID,
            key_len: 8,
        };
        hdr.encode(&mut buf).unwrap();
        let decoded = HeaderPage::decode(&buf).unwrap();
        assert_eq!(decoded.root_page_id, 42);
        assert_eq!(decoded.first_leaf_id, INVALID_PAGE_ID);
        assert_eq!(decoded.key_len, 8);
    }

    #[test]
    fn test_leaf_page_encode_decode_roundtrip() {
        let mut buf = [0u8; PAGE_SIZE];
        let key_len = 4u32;
        let mut leaf = LeafPage::new(10, NUM_TOMBSTONES);
        leaf.keys = vec![
            IndexKey(vec![0, 0, 0, 10]),
            IndexKey(vec![0, 0, 0, 20]),
            IndexKey(vec![0, 0, 0, 30]),
        ];
        leaf.record_ids = vec![
            RecordId {
                page_id: 1,
                slot_id: 7,
            },
            RecordId {
                page_id: 2,
                slot_id: 8,
            },
            RecordId {
                page_id: 3,
                slot_id: 9,
            },
        ];
        leaf.base.size = leaf.keys.len() as u32;
        leaf.next_leaf_id = 123;
        leaf.encode(&mut buf, key_len).unwrap();

        let decoded = LeafPage::decode(&buf, key_len).unwrap();
        assert_eq!(decoded.base.size, 3);
        assert_eq!(decoded.base.max_size, 10);
        assert_eq!(decoded.next_leaf_id, 123);
        assert_eq!(decoded.keys, leaf.keys);
        assert_eq!(decoded.record_ids.len(), leaf.record_ids.len());
        for (a, b) in decoded.record_ids.iter().zip(leaf.record_ids.iter()) {
            assert_eq!(a.page_id, b.page_id);
            assert_eq!(a.slot_id, b.slot_id);
        }
    }

    #[test]
    fn test_leaf_page_delete_stores_tombstones_and_eviction_is_fifo() {
        let mut leaf = LeafPage::new(10, 2);
        leaf.insert(
            IndexKey(vec![10]),
            RecordId {
                page_id: 1,
                slot_id: 1,
            },
        )
        .unwrap();
        leaf.insert(
            IndexKey(vec![20]),
            RecordId {
                page_id: 1,
                slot_id: 2,
            },
        )
        .unwrap();
        leaf.insert(
            IndexKey(vec![30]),
            RecordId {
                page_id: 1,
                slot_id: 3,
            },
        )
        .unwrap();

        assert!(leaf.remove_key(&IndexKey(vec![10])).is_ok());
        assert_eq!(leaf.keys.len(), 3);
        assert_eq!(leaf.tombstones.len(), 1);
        assert_eq!(leaf.tombstones[0], 0);
        assert!(leaf.get(&IndexKey(vec![10])).is_none());

        assert!(leaf.remove_key(&IndexKey(vec![20])).is_ok());
        assert_eq!(leaf.keys.len(), 3);
        assert_eq!(leaf.tombstones.len(), 2);
        assert_eq!(leaf.tombstones[0], 0);
        assert_eq!(leaf.tombstones[1], 1);
        assert!(leaf.get(&IndexKey(vec![20])).is_none());

        assert!(leaf.remove_key(&IndexKey(vec![30])).is_ok());
        assert_eq!(
            leaf.keys.len(),
            2,
            "the oldest tombstone (key 10) is purged"
        );
        assert_eq!(
            Vec::from(leaf.tombstones.clone()),
            vec![0, 1],
            "keys 20 and 30 shift down one slot after key 10 is purged"
        );
        assert!(leaf.get(&IndexKey(vec![30])).is_none());

        let mut buf = [0u8; PAGE_SIZE];
        leaf.encode(&mut buf, 1).unwrap();
        let decoded = LeafPage::decode(&buf, 1).unwrap();
        assert_eq!(decoded.keys.len(), 2);
        assert_eq!(decoded.tombstones.len(), 2);
        assert_eq!(decoded.tombstones[0], 0);
        assert_eq!(decoded.tombstones[1], 1);
        assert!(decoded.get(&IndexKey(vec![20])).is_none());
        assert!(decoded.get(&IndexKey(vec![30])).is_none());
    }
    #[test]
    fn test_leaf_page_encode_errors_if_tombstones_do_not_fit() {
        let key_len = 1u32;
        let slack_after_entries = 8;
        let entry_bytes = key_len as usize + RECORD_ID_ENCODED_BYTES;
        let max_size = (PAGE_SIZE - LEAF_FIXED_HEADER_BYTES - slack_after_entries) / entry_bytes;
        assert_eq!(
            LEAF_FIXED_HEADER_BYTES + max_size * entry_bytes + slack_after_entries,
            PAGE_SIZE
        );
        let two_tombstones_bytes = TOMBSTONE_COUNT_ENCODED_BYTES + 2 * TOMBSTONE_SLOT_ENCODED_BYTES;
        assert!(two_tombstones_bytes > slack_after_entries);

        let mut leaf = LeafPage::new(max_size as u32, 2);
        leaf.keys = vec![IndexKey(vec![0u8]); max_size];
        leaf.record_ids = (0..max_size as u32)
            .map(|i| RecordId {
                page_id: i,
                slot_id: 0,
            })
            .collect();
        leaf.base.size = leaf.keys.len() as u32;
        leaf.tombstones = VecDeque::from([0u32, 1u32]);
        leaf.validate();

        let mut buf = [0u8; PAGE_SIZE];
        let err = leaf.encode(&mut buf, key_len).unwrap_err();
        assert!(matches!(
            err,
            PageCodecError::BufferError(BufferError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_internal_page_encode_decode_roundtrip() {
        let mut buf = [0u8; PAGE_SIZE];
        let key_len = 2u32;
        let mut internal = InternalPage::new(10);
        internal.children = vec![11, 22, 33];
        internal.keys = vec![IndexKey(vec![1, 2]), IndexKey(vec![3, 4])];
        internal.base.size = internal.keys.len() as u32;
        internal.encode(&mut buf, key_len).unwrap();

        let decoded = InternalPage::decode(&buf, key_len).unwrap();
        assert_eq!(decoded.base.size, 2);
        assert_eq!(decoded.base.max_size, 10);
        assert_eq!(decoded.children, internal.children);
        assert_eq!(decoded.keys, internal.keys);
    }

    #[test]
    fn test_decode_rejects_bad_magic() {
        let mut buf = [0u8; PAGE_SIZE];
        let hdr = HeaderPage {
            root_page_id: 1,
            first_leaf_id: 2,
            key_len: 4,
        };
        hdr.encode(&mut buf).unwrap();
        buf[0] ^= 0xff;
        assert!(matches!(
            HeaderPage::decode(&buf),
            Err(PageCodecError::BadMagic)
        ));
    }

    fn rid(n: u32) -> RecordId {
        RecordId {
            page_id: n,
            slot_id: 0,
        }
    }

    #[test]
    fn fixed_header_sizes_match_the_encoded_field_layout() {
        assert_eq!(INTERNAL_FIXED_HEADER_BYTES, 8 + 4 * 4);
        assert_eq!(LEAF_FIXED_HEADER_BYTES, 8 + 2 * 4 + 8);
        assert_eq!(HEADER_FIXED_HEADER_BYTES, 8 + 2 * 4 + 2 * 8);
        assert_eq!(PAGE_ID_ENCODED_BYTES, 8);
    }

    #[test]
    fn internal_max_size_for_layout_is_the_largest_page_that_encodes() {
        for key_len in [1u32, 4, 8, 37, 255] {
            let max = InternalPage::max_size_for_layout(key_len as usize).unwrap() as usize;
            let page_with = |n: usize| {
                let mut page = InternalPage::new(max as u32);
                page.keys = vec![IndexKey(vec![0; key_len as usize]); n];
                page.children = (0..=n).collect();
                page.base.size = n as u32;
                page
            };
            let mut buf = [0u8; PAGE_SIZE];
            assert!(
                page_with(max).encode(&mut buf, key_len).is_ok(),
                "key_len {key_len}: max keys fit"
            );
            assert!(
                page_with(max + 1).encode(&mut buf, key_len).is_err(),
                "key_len {key_len}: one more key overflows"
            );
        }
    }

    #[test]
    fn leaf_max_size_for_layout_leaves_room_for_the_tombstone_count() {
        for key_len in [1u32, 4, 8, 37, 255] {
            let max = LeafPage::max_size_for_layout(key_len as usize).unwrap() as usize;
            let entry_bytes = key_len as usize + RECORD_ID_ENCODED_BYTES;
            let used = LEAF_FIXED_HEADER_BYTES + max * entry_bytes + TOMBSTONE_COUNT_ENCODED_BYTES;
            assert!(used <= PAGE_SIZE, "key_len {key_len}: max entries fit");
            assert!(
                used + entry_bytes > PAGE_SIZE,
                "key_len {key_len}: one more entry overflows"
            );
        }
        assert_eq!(LeafPage::max_size_for_layout(0), None);
        assert_eq!(InternalPage::max_size_for_layout(0), None);
    }

    #[test]
    fn tombstone_capacity_fills_the_space_beside_a_full_leaf() {
        let key_len = 8u32;
        let max_size = LeafPage::max_size_for_layout(key_len as usize).unwrap();
        let free = PAGE_SIZE
            - LEAF_FIXED_HEADER_BYTES
            - max_size as usize * (key_len as usize + RECORD_ID_ENCODED_BYTES)
            - TOMBSTONE_COUNT_ENCODED_BYTES;
        let fit = (free / TOMBSTONE_SLOT_ENCODED_BYTES) as u32;
        assert_eq!(
            LeafPage::tombstone_capacity_for_layout(max_size, key_len, u32::MAX),
            fit
        );
        assert_eq!(
            LeafPage::tombstone_capacity_for_layout(max_size, key_len, 1),
            1.min(fit)
        );
        assert_eq!(LeafPage::tombstone_capacity_for_layout(max_size, 0, 10), 0);
        assert_eq!(
            LeafPage::tombstone_capacity_for_layout(u32::MAX, key_len, 10),
            0
        );
    }

    #[test]
    fn find_child_index_counts_keys_less_than_or_equal_to_the_search_key() {
        let mut page = InternalPage::new(10);
        page.children = vec![1, 2, 3];
        page.keys = vec![IndexKey(vec![10]), IndexKey(vec![20])];
        page.base.size = 2;
        assert_eq!(page.find_child_index(&IndexKey(vec![5])), 0);
        assert_eq!(page.find_child_index(&IndexKey(vec![10])), 1);
        assert_eq!(page.find_child_index(&IndexKey(vec![15])), 1);
        assert_eq!(page.find_child_index(&IndexKey(vec![20])), 2);
        assert_eq!(page.find_child_index(&IndexKey(vec![99])), 2);
    }

    #[test]
    fn leaf_insert_rejects_live_duplicates_and_revives_tombstoned_keys() {
        let mut leaf = LeafPage::new(10, 2);
        leaf.insert(IndexKey(vec![1]), rid(1)).unwrap();
        assert!(matches!(
            leaf.insert(IndexKey(vec![1]), rid(2)),
            Err(InsertError::DuplicateKey)
        ));
        leaf.remove_key(&IndexKey(vec![1])).unwrap();
        leaf.insert(IndexKey(vec![1]), rid(3)).unwrap();
        assert_eq!(leaf.get(&IndexKey(vec![1])).unwrap().page_id, 3);
        assert!(leaf.tombstones.is_empty());
    }

    #[test]
    fn leaf_without_tombstone_capacity_deletes_keys_immediately() {
        let mut leaf = LeafPage::new(10, 0);
        leaf.insert(IndexKey(vec![1]), rid(1)).unwrap();
        leaf.insert(IndexKey(vec![2]), rid(2)).unwrap();
        leaf.remove_key(&IndexKey(vec![1])).unwrap();
        assert_eq!(leaf.keys, vec![IndexKey(vec![2])]);
        assert!(leaf.tombstones.is_empty());
    }

    #[test]
    fn leaf_encoding_always_writes_a_tombstone_count() {
        let mut leaf = LeafPage::new(10, NUM_TOMBSTONES);
        leaf.insert(IndexKey(vec![7]), rid(1)).unwrap();
        let mut buf = [0xFFu8; PAGE_SIZE];
        leaf.encode(&mut buf, 1).unwrap();
        let count_at = LEAF_FIXED_HEADER_BYTES + 1 + RECORD_ID_ENCODED_BYTES;
        assert_eq!(
            &buf[count_at..count_at + TOMBSTONE_COUNT_ENCODED_BYTES],
            &[0, 0, 0, 0]
        );
        assert!(LeafPage::decode(&buf, 1).unwrap().tombstones.is_empty());
    }
}
