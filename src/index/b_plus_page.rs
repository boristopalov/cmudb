use crate::buf::{BufReader, BufWriter, BufferError};
use crate::catalog::index_schema::IndexKey;
use crate::disk::PAGE_SIZE;
use crate::index::page_codec::{
    COMMON_HEADER_BYTES, KIND_HEADER, KIND_INTERNAL, KIND_LEAF, OVERFLOW, PageCodecError,
    U32_BYTES, U64_BYTES, decode_page_id, decode_record_id, encode_page_id, encode_record_id,
    peek_common_header, read_common_header_of_kind, write_common_header,
};
use crate::index::{InsertError, RemoveError};
use crate::table_heap::RecordId;
use log::{error, info};
use std::collections::VecDeque;
use std::convert::TryFrom;
use std::num::NonZeroUsize;

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

/// Sums byte counts, failing with an overflow error if any part or the total overflows.
fn total_bytes<const N: usize>(parts: [Option<usize>; N]) -> Result<usize, PageCodecError> {
    parts
        .into_iter()
        .try_fold(0usize, |sum, part| part.and_then(|p| sum.checked_add(p)))
        .ok_or(OVERFLOW)
}

/// Fails with `BufferTooSmall` when `needed` bytes do not fit in one page.
fn ensure_fits_page(needed: usize) -> Result<(), PageCodecError> {
    if needed > PAGE_SIZE {
        return Err(PageCodecError::BufferError(BufferError::BufferTooSmall {
            needed,
            actual: PAGE_SIZE,
        }));
    }
    Ok(())
}

/// The byte sizes and offsets of internal and leaf pages whose keys are `key_len` bytes wide.
/// The owned codecs here and the in-place views in `page_view` both read the layout from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    key_len: NonZeroUsize,
}

impl Layout {
    /// Fails with `Malformed` for a zero `key_len`.
    pub fn new(key_len: u32) -> Result<Self, PageCodecError> {
        NonZeroUsize::new(usize::try_from(key_len)?)
            .map(|key_len| Self { key_len })
            .ok_or(PageCodecError::Malformed("key_len must be > 0"))
    }

    pub fn key_len(self) -> usize {
        self.key_len.get()
    }

    /// Bytes of one leaf entry: the key and its record id.
    pub fn leaf_entry_bytes(self) -> usize {
        self.key_len().saturating_add(RECORD_ID_ENCODED_BYTES)
    }

    /// Bytes an internal page with `children` child ids and `keys` keys encodes to.
    pub fn internal_bytes(self, children: usize, keys: usize) -> Result<usize, PageCodecError> {
        total_bytes([
            Some(INTERNAL_FIXED_HEADER_BYTES),
            children.checked_mul(PAGE_ID_ENCODED_BYTES),
            keys.checked_mul(self.key_len()),
        ])
    }

    /// Bytes a leaf's fixed header and `entries` entries take, up to its tombstone section.
    pub fn leaf_live_bytes(self, entries: usize) -> Result<usize, PageCodecError> {
        total_bytes([
            Some(LEAF_FIXED_HEADER_BYTES),
            entries.checked_mul(self.leaf_entry_bytes()),
        ])
    }

    /// Bytes a leaf with `entries` entries and `tombstones` tombstones encodes to.
    pub fn leaf_bytes(self, entries: usize, tombstones: usize) -> Result<usize, PageCodecError> {
        total_bytes([
            Some(self.leaf_live_bytes(entries)?),
            Some(TOMBSTONE_COUNT_ENCODED_BYTES),
            tombstones.checked_mul(TOMBSTONE_SLOT_ENCODED_BYTES),
        ])
    }

    /// Whether a leaf with `entries` entries has room left in its page for a tombstone count.
    pub fn leaf_has_tombstone_section(self, entries: usize) -> bool {
        self.leaf_bytes(entries, 0).is_ok_and(|n| n <= PAGE_SIZE)
    }

    /// Offset of child id `idx` in an internal page.
    pub fn internal_child_offset(self, idx: usize) -> usize {
        INTERNAL_FIXED_HEADER_BYTES + idx * PAGE_ID_ENCODED_BYTES
    }

    /// Offset of key `idx` in an internal page with `children` child ids.
    pub fn internal_key_offset(self, children: usize, idx: usize) -> usize {
        self.internal_child_offset(children) + idx * self.key_len()
    }

    /// Offset of entry `idx` in a leaf page.
    pub fn leaf_entry_offset(self, idx: usize) -> usize {
        LEAF_FIXED_HEADER_BYTES + idx * self.leaf_entry_bytes()
    }

    /// Offset of the tombstone count in a leaf with `entries` entries.
    pub fn leaf_tombstone_count_offset(self, entries: usize) -> usize {
        self.leaf_entry_offset(entries)
    }

    /// Offset of tombstone `i` in a leaf with `entries` entries.
    pub fn leaf_tombstone_offset(self, entries: usize, i: usize) -> usize {
        self.leaf_tombstone_count_offset(entries)
            + TOMBSTONE_COUNT_ENCODED_BYTES
            + i * TOMBSTONE_SLOT_ENCODED_BYTES
    }

    /// The most keys an internal page can encode into one page, or `None` when not even one fits.
    pub fn internal_max_keys(self) -> Option<u32> {
        let key_and_right_child = self.key_len().checked_add(PAGE_ID_ENCODED_BYTES)?;
        let space = PAGE_SIZE.checked_sub(self.internal_bytes(1, 0).ok()?)?;
        u32::try_from(space / key_and_right_child)
            .ok()
            .filter(|&n| n > 0)
    }

    /// The most entries a leaf can encode into one page beside an empty tombstone section, or
    /// `None` when not even one fits.
    pub fn leaf_max_entries(self) -> Option<u32> {
        let space = PAGE_SIZE.checked_sub(self.leaf_bytes(0, 0).ok()?)?;
        u32::try_from(space / self.leaf_entry_bytes())
            .ok()
            .filter(|&n| n > 0)
    }

    /// How many tombstones (at most `requested`) fit beside a full leaf of `max_size` entries.
    pub fn tombstone_capacity(self, max_size: u32, requested: u32) -> u32 {
        usize::try_from(max_size)
            .ok()
            .and_then(|entries| self.leaf_bytes(entries, 0).ok())
            .and_then(|full| PAGE_SIZE.checked_sub(full))
            .and_then(|space| u32::try_from(space / TOMBSTONE_SLOT_ENCODED_BYTES).ok())
            .map_or(0, |fit| fit.min(requested))
    }

    fn check_key(self, key: &IndexKey) -> Result<(), PageCodecError> {
        if key.0.len() != self.key_len() {
            return Err(PageCodecError::InvalidKeyLen {
                expected: self.key_len(),
                actual: key.0.len(),
            });
        }
        Ok(())
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
    pub fn decode(buf: &[u8], layout: Layout) -> Result<Self, PageCodecError> {
        match Self::kind(buf)? {
            PageKind::Internal => Ok(Self::Internal(InternalPage::decode(buf, layout)?)),
            PageKind::Leaf => Ok(Self::Leaf(LeafPage::decode(buf, layout)?)),
        }
    }

    pub fn kind(buf: &[u8]) -> Result<PageKind, PageCodecError> {
        match peek_common_header(buf)? {
            KIND_INTERNAL => Ok(PageKind::Internal),
            KIND_LEAF => Ok(PageKind::Leaf),
            k => Err(PageCodecError::InvalidPageKind(k)),
        }
    }
}

fn min_size_of(max_size: u32) -> u32 {
    max_size / 2
}

#[derive(Debug)]
pub struct InternalPage {
    max_size: u32,
    children: Vec<PageId>,
    keys: Vec<IndexKey>,
}

impl InternalPage {
    pub fn new(max_size: u32) -> Self {
        Self {
            max_size,
            children: Vec::new(),
            keys: Vec::new(),
        }
    }

    /// A root over `left` and `right`, separated by `separator`.
    pub fn new_root(max_size: u32, left: PageId, separator: IndexKey, right: PageId) -> Self {
        let root = Self {
            max_size,
            children: vec![left, right],
            keys: vec![separator],
        };
        root.debug_check();
        root
    }

    pub fn children(&self) -> &[PageId] {
        &self.children
    }

    pub fn keys(&self) -> &[IndexKey] {
        &self.keys
    }

    /// The number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Returns the PageId of the child at the given index
    pub fn get_child(&self, idx: usize) -> Option<&PageId> {
        self.children.get(idx)
    }

    /// Fails with `Malformed` when the children do not number one more than the keys (unless
    /// both are empty), the keys exceed `max_size + 1`, or the keys are out of order.
    pub fn check(&self) -> Result<(), PageCodecError> {
        if self.children.is_empty() && self.keys.is_empty() {
            return Ok(());
        }
        if self.children.len() != self.keys.len() + 1 {
            return Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != size + 1",
            ));
        }
        if self.keys.len() > self.max_size as usize + 1 {
            return Err(PageCodecError::Malformed("internal size exceeds max size"));
        }
        if !self.keys.is_sorted() {
            return Err(PageCodecError::Malformed("internal keys out of order"));
        }
        Ok(())
    }

    fn debug_check(&self) {
        debug_assert_eq!(self.check(), Ok(()), "internal page after a mutation");
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
        self.debug_check();
    }

    pub fn insert_key(&mut self, index: usize, key: IndexKey) {
        self.keys.insert(index, key);
    }

    pub fn remove_key(&mut self, index: usize) -> IndexKey {
        self.keys.remove(index)
    }

    pub fn min_size(&self) -> u32 {
        min_size_of(self.max_size)
    }

    /// Moves the keys and children above the middle key into the empty `right` page, removes the
    /// middle key from this page and returns it for the parent.
    pub fn split_into(&mut self, right: &mut InternalPage) -> IndexKey {
        assert!(
            right.children.is_empty() && right.keys.is_empty(),
            "right page must be empty before split"
        );
        self.debug_check();

        let mid = self.keys.len() / 2;

        right.keys = self.keys.split_off(mid + 1);
        let sep_key = self
            .keys
            .pop()
            .expect("internal page must have at least one key");
        right.children = self.children.split_off(mid + 1);

        self.debug_check();
        right.debug_check();

        sep_key
    }

    pub fn absorb_right(&mut self, sep_key: IndexKey, right: &mut InternalPage) {
        self.debug_check();
        right.debug_check();

        self.keys.push(sep_key);
        self.keys.append(&mut right.keys);
        self.children.append(&mut right.children);

        self.debug_check();
    }

    pub fn encode(&self, buf: &mut [u8], layout: Layout) -> Result<(), PageCodecError> {
        self.check()?;
        for key in &self.keys {
            layout.check_key(key)?;
        }
        let needed = layout.internal_bytes(self.children.len(), self.keys.len())?;
        ensure_fits_page(needed)?;
        info!(
            "bptree/page_encode kind=internal keys={} children={}",
            self.keys.len(),
            self.children.len()
        );

        buf.fill(0);
        let mut w = BufWriter::new(buf)?;
        write_common_header(&mut w, KIND_INTERNAL)?;
        w.write_u32_le(u32::try_from(self.keys.len())?)?;
        w.write_u32_le(self.max_size)?;
        w.write_u32_le(u32::try_from(self.children.len())?)?;
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

    pub fn decode(buf: &[u8], layout: Layout) -> Result<Self, PageCodecError> {
        let mut r = BufReader::new(buf)?;
        read_common_header_of_kind(&mut r, KIND_INTERNAL)?;
        let size = usize::try_from(r.read_u32_le()?)?;
        let max_size = r.read_u32_le()?;
        let child_count = usize::try_from(r.read_u32_le()?)?;
        let _reserved = r.read_u32_le()?;
        info!(
            "bptree/page_decode kind=internal size={} max_size={} children={}",
            size, max_size, child_count
        );

        if child_count == 0 && size != 0 {
            return Err(PageCodecError::Malformed(
                "internal page has keys but zero children",
            ));
        }
        if child_count != 0 && Some(child_count) != size.checked_add(1) {
            return Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != size + 1",
            ));
        }
        ensure_fits_page(layout.internal_bytes(child_count, size)?)?;

        let mut children = Vec::with_capacity(child_count);
        for _ in 0..child_count {
            children.push(decode_page_id(r.read_u64_le()?)?);
        }
        let mut keys = Vec::with_capacity(size);
        for _ in 0..size {
            keys.push(IndexKey(r.read_bytes(layout.key_len())?.to_vec()));
        }

        let page = Self {
            max_size,
            children,
            keys,
        };
        page.check()?;
        Ok(page)
    }

    pub fn is_full(&self) -> bool {
        self.keys.len() == self.max_size as usize
    }

    pub fn is_underfull(&self) -> bool {
        self.keys.len() < self.min_size() as usize
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
        self.debug_check();
        (sep_key, right_child)
    }

    pub fn pop_last_child_and_key(&mut self) -> (IndexKey, PageId) {
        let child = self.children.pop().expect("internal page has no children");
        let key = self.keys.pop().expect("internal page has no keys");
        self.debug_check();
        (key, child)
    }

    pub fn pop_first_child_and_key(&mut self) -> (IndexKey, PageId) {
        let child = self.children.remove(0);
        let key = self.keys.remove(0);
        self.debug_check();
        (key, child)
    }

    pub fn replace_key(&mut self, idx: usize, new_key: IndexKey) -> IndexKey {
        assert!(idx < self.keys.len(), "replace_key index out of bounds");
        let old = std::mem::replace(&mut self.keys[idx], new_key);
        self.debug_check();
        old
    }

    /// Inserts a new leftmost child and separator key.
    /// After insertion, `key` separates `child` (new `children[0]`) and the old `children[0]`.
    pub fn prepend_child_and_key(&mut self, key: IndexKey, child: PageId) {
        self.children.insert(0, child);
        self.keys.insert(0, key);
        self.debug_check();
    }

    /// Appends a separator key and a new rightmost child.
    /// After insertion, `key` separates the old rightmost child and `child` (new rightmost).
    pub fn append_key_and_child(&mut self, key: IndexKey, child: PageId) {
        self.keys.push(key);
        self.children.push(child);
        self.debug_check();
    }

    pub fn is_overfull(&self) -> bool {
        self.keys.len() > self.max_size as usize
    }

    pub fn would_be_underfull(&self) -> bool {
        self.keys.len() == self.min_size() as usize
    }
}

#[derive(Debug)]
pub struct LeafPage {
    max_size: u32,
    entries: Vec<(IndexKey, RecordId)>,
    /// Page id of the next leaf to the right, or `INVALID_PAGE_ID` for the last leaf.
    next_leaf_id: PageId,
    /// Slots of deleted keys still stored in the page, oldest deletion first.
    tombstones: VecDeque<u32>,
    tombstone_capacity: usize,
}

impl LeafPage {
    pub fn new(max_size: u32, num_tombstones: u32) -> Self {
        let tombstone_capacity = usize::try_from(num_tombstones).unwrap_or(0);
        Self {
            max_size,
            entries: Vec::new(),
            next_leaf_id: INVALID_PAGE_ID,
            tombstones: VecDeque::with_capacity(tombstone_capacity),
            tombstone_capacity,
        }
    }

    /// The number of entries, tombstoned ones included.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn next_leaf_id(&self) -> PageId {
        self.next_leaf_id
    }

    #[cfg(test)]
    pub(crate) fn set_next_leaf_id(&mut self, next_leaf_id: PageId) {
        self.next_leaf_id = next_leaf_id;
    }

    /// The smallest key, tombstoned or not.
    pub fn first_key(&self) -> Option<&IndexKey> {
        self.entries.first().map(|(key, _)| key)
    }

    /// Every key in order, tombstoned ones included.
    pub fn keys(&self) -> impl Iterator<Item = &IndexKey> {
        self.entries.iter().map(|(key, _)| key)
    }

    pub fn tombstones_full(&self) -> bool {
        self.tombstones.len() >= self.tombstone_capacity
    }

    /// Fails with `Malformed` when the entries exceed `max_size + 1`, the keys are out of order,
    /// the tombstones exceed their capacity, or a tombstone names a slot past the last entry.
    pub fn check(&self) -> Result<(), PageCodecError> {
        if self.entries.len() > self.max_size as usize + 1 {
            return Err(PageCodecError::Malformed("leaf size exceeds max size"));
        }
        if !self.keys().is_sorted() {
            return Err(PageCodecError::Malformed("leaf keys out of order"));
        }
        if self.tombstones.len() > self.tombstone_capacity {
            return Err(PageCodecError::Malformed(
                "leaf tombstones count exceeds tombstone capacity",
            ));
        }
        if self
            .tombstones
            .iter()
            .any(|&slot| slot as usize >= self.entries.len())
        {
            return Err(PageCodecError::Malformed(
                "leaf tombstones index out of bounds",
            ));
        }
        Ok(())
    }

    fn debug_check(&self) {
        debug_assert_eq!(self.check(), Ok(()), "leaf page after a mutation");
    }

    /// Like `slice::binary_search` over the keys.
    pub fn search(&self, key: &IndexKey) -> Result<usize, usize> {
        self.entries.binary_search_by(|(k, _)| k.cmp(key))
    }

    pub fn lower_bound(&self, key: &IndexKey) -> usize {
        self.entries.partition_point(|(k, _)| k < key)
    }

    pub fn upper_bound(&self, key: &IndexKey) -> usize {
        self.entries.partition_point(|(k, _)| k <= key)
    }

    /// Returns the record id stored for `key`, or `None` if it is absent or tombstoned.
    pub fn get(&self, key: &IndexKey) -> Option<RecordId> {
        let pos = match self.search(key) {
            Ok(pos) => pos,
            Err(_) => {
                error!("get(): could not find key in page; key={:?}", key);
                return None;
            }
        };
        if self.is_tombstoned(Self::slot(pos)) {
            error!("get(): key is tombstoned; key={:?}", key);
            return None;
        }
        Some(self.entries[pos].1)
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
        self.entries.remove(slot as usize);
        self.adjust_tombstones_after_remove(slot);
    }

    pub fn encode(&self, buf: &mut [u8], layout: Layout) -> Result<(), PageCodecError> {
        self.check()?;
        for key in self.keys() {
            layout.check_key(key)?;
        }
        let needed = layout.leaf_bytes(self.entries.len(), self.tombstones.len())?;
        ensure_fits_page(needed)?;
        info!(
            "bptree/page_encode kind=leaf keys={} tombstones={}",
            self.entries.len(),
            self.tombstones.len()
        );

        buf.fill(0);
        let mut w = BufWriter::new(buf)?;
        write_common_header(&mut w, KIND_LEAF)?;
        w.write_u32_le(u32::try_from(self.entries.len())?)?;
        w.write_u32_le(self.max_size)?;
        w.write_u64_le(encode_page_id(self.next_leaf_id)?)?;

        for (k, rid) in self.entries.iter() {
            w.write_bytes(&k.0)?;
            let (pid, sid) = encode_record_id(rid)?;
            w.write_u64_le(pid)?;
            w.write_u64_le(sid)?;
        }

        w.write_u32_le(u32::try_from(self.tombstones.len())?)?;
        for &slot in self.tombstones.iter() {
            w.write_u32_le(slot)?;
        }

        debug_assert_eq!(w.pos(), needed);
        Ok(())
    }

    pub fn decode(buf: &[u8], layout: Layout) -> Result<Self, PageCodecError> {
        let mut r = BufReader::new(buf)?;
        read_common_header_of_kind(&mut r, KIND_LEAF)?;
        let size = usize::try_from(r.read_u32_le()?)?;
        let max_size = r.read_u32_le()?;
        let next_leaf_id = decode_page_id(r.read_u64_le()?)?;
        info!(
            "bptree/page_decode kind=leaf size={} max_size={} next_leaf_id={}",
            size, max_size, next_leaf_id
        );
        ensure_fits_page(layout.leaf_live_bytes(size)?)?;

        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            let key = IndexKey(r.read_bytes(layout.key_len())?.to_vec());
            let pid = r.read_u64_le()?;
            let sid = r.read_u64_le()?;
            entries.push((key, decode_record_id(pid, sid)?));
        }
        let tombstones = if layout.leaf_has_tombstone_section(size) {
            Self::decode_tombstones(&mut r)?
        } else {
            VecDeque::new()
        };

        let page = Self {
            max_size,
            entries,
            next_leaf_id,
            tombstones,
            tombstone_capacity: layout.tombstone_capacity(max_size, NUM_TOMBSTONES) as usize,
        };
        page.check()?;
        Ok(page)
    }

    /// Reads the tombstone count and slots that follow the entries.
    fn decode_tombstones(r: &mut BufReader<'_>) -> Result<VecDeque<u32>, PageCodecError> {
        let count = usize::try_from(r.read_u32_le()?)?;
        let needed = count
            .checked_mul(TOMBSTONE_SLOT_ENCODED_BYTES)
            .ok_or(OVERFLOW)?;
        if r.remaining() < needed {
            return Err(PageCodecError::Malformed(
                "leaf tombstones section exceeds remaining bytes",
            ));
        }
        (0..count)
            .map(|_| Ok(r.read_u32_le()?))
            .collect::<Result<VecDeque<u32>, PageCodecError>>()
    }
}

impl LeafPage {
    /// Inserts `key`, reviving it if it is tombstoned.
    ///
    /// Returns `InsertError::DuplicateKey` if `key` is already live in this page.
    pub fn insert(&mut self, key: IndexKey, record_id: RecordId) -> Result<(), InsertError> {
        match self.search(&key) {
            Ok(pos) => {
                let slot = Self::slot(pos);
                if !self.is_tombstoned(slot) {
                    error!("attempted to insert duplicate key into leaf page");
                    return Err(InsertError::DuplicateKey);
                }
                self.entries[pos].1 = record_id;
                self.remove_tombstone(slot);
                self.debug_check();
                Ok(())
            }
            Err(pos) => {
                let insert_at = Self::slot(pos);
                self.entries.insert(pos, (key, record_id));
                self.adjust_tombstones_after_insert(insert_at);
                self.debug_check();
                Ok(())
            }
        }
    }

    /// Splits this leaf into `right` and returns the separator key for the parent.
    /// For leaf pages, the separator key is the first key in the new right page and remains in the leaf.
    pub fn split_into(&mut self, right: &mut LeafPage, right_page_id: PageId) -> IndexKey {
        assert!(
            right.entries.is_empty(),
            "right page must be empty before split"
        );
        self.debug_check();
        assert!(!self.entries.is_empty(), "cannot split an empty leaf page");

        let mid = self.entries.len() / 2;
        let mid_u32 = Self::slot(mid);

        right.entries = self.entries.split_off(mid);

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

        self.debug_check();
        right.debug_check();

        right.entries[0].0.clone()
    }

    /// Appends `right`'s entries to this leaf. `right`'s tombstones count as newer than this
    /// page's, so when the merged tombstones overflow, this page's oldest are purged first.
    pub fn absorb_right(&mut self, right: &mut LeafPage) {
        self.debug_check();
        right.debug_check();

        let left_len = Self::slot(self.entries.len());
        self.entries.append(&mut right.entries);
        self.next_leaf_id = right.next_leaf_id;

        for slot in right.tombstones.drain(..) {
            self.tombstones.push_back(slot + left_len);
        }
        while self.tombstones.len() > self.tombstone_capacity {
            self.purge_oldest_tombstone();
        }

        self.debug_check();
    }

    pub fn is_full(&self) -> bool {
        self.entries.len() == self.max_size as usize
    }

    pub fn is_overfull(&self) -> bool {
        self.entries.len() > self.max_size as usize
    }

    pub fn min_size(&self) -> u32 {
        min_size_of(self.max_size)
    }

    pub fn is_underfull(&self) -> bool {
        self.entries.len() < self.min_size() as usize
    }

    pub fn would_be_underfull(&self) -> bool {
        self.entries.len() == self.min_size() as usize
    }

    pub fn key_exists(&self, key: &IndexKey) -> bool {
        let found = self.search(key).is_ok();
        if !found {
            error!("could not find key in page: {:?}", key);
        }
        found
    }

    pub fn remove_key(&mut self, key: &IndexKey) -> Result<(), RemoveError> {
        info!("attempting to removing key from page: {:?}", key);
        let pos = match self.search(key) {
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
            self.debug_check();
            return Ok(());
        }

        if self.tombstones.len() == self.tombstone_capacity {
            let purged = self.purge_oldest_tombstone();
            if purged < slot {
                slot -= 1;
            }
        }

        self.tombstones.push_back(slot);

        self.debug_check();
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
        if self.entries.is_empty() {
            return None;
        }
        let was_tombstoned = self.is_tombstoned(0);
        let (key, rid) = self.entries.remove(0);
        self.adjust_tombstones_after_remove(0);
        self.debug_check();
        Some((key, rid, was_tombstoned))
    }

    pub fn pop_last(&mut self) -> Option<(IndexKey, RecordId, bool)> {
        let slot = u32::try_from(self.entries.len().saturating_sub(1)).ok()?;
        let was_tombstoned = self.is_tombstoned(slot);
        let (key, rid) = self.entries.pop()?;
        self.remove_tombstone(slot);
        self.debug_check();
        Some((key, rid, was_tombstoned))
    }

    pub fn push_front(&mut self, key: IndexKey, record_id: RecordId, tombstoned: bool) {
        self.adjust_tombstones_after_insert(0);
        self.entries.insert(0, (key, record_id));
        if tombstoned {
            self.tombstones.push_back(0);
        }
        self.debug_check();
    }

    pub fn push_back(&mut self, key: IndexKey, record_id: RecordId, tombstoned: bool) {
        let slot = Self::slot(self.entries.len());
        self.entries.push((key, record_id));
        if tombstoned {
            self.tombstones.push_back(slot);
        }
        self.debug_check();
    }

    /// The live entries in key order, skipping tombstoned ones.
    pub fn iter(&self) -> impl Iterator<Item = (&IndexKey, &RecordId)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.is_tombstoned(Self::slot(i)))
            .map(|(_, (k, rid))| (k, rid))
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
        Layout::new(self.key_len)?;
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
        read_common_header_of_kind(&mut r, KIND_HEADER)?;
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
        InternalPage, LEAF_FIXED_HEADER_BYTES, Layout, LeafPage, MAX_SIZE_FIELD_BYTES,
        NUM_TOMBSTONES, PAGE_ID_ENCODED_BYTES, Page, RECORD_ID_ENCODED_BYTES, SIZE_FIELD_BYTES,
        TOMBSTONE_COUNT_ENCODED_BYTES, TOMBSTONE_SLOT_ENCODED_BYTES,
    };
    use crate::index::page_codec::KIND_HEADER;
    use crate::index::page_codec::PageCodecError;
    use crate::table_heap::RecordId;
    use std::collections::VecDeque;

    fn layout(key_len: u32) -> Layout {
        Layout::new(key_len).unwrap()
    }

    fn keys_of(leaf: &LeafPage) -> Vec<IndexKey> {
        leaf.keys().cloned().collect()
    }

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
        left.check().unwrap();

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
        left.entries = vec![
            IndexKey(vec![10]),
            IndexKey(vec![20]),
            IndexKey(vec![30]),
            IndexKey(vec![40]),
        ]
        .into_iter()
        .zip(vec![
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
        ])
        .collect();
        left.next_leaf_id = 999;
        left.tombstones = VecDeque::from([1u32, 3u32]);
        left.check().unwrap();

        let mut right = LeafPage::new(10, NUM_TOMBSTONES);
        let sep = left.split_into(&mut right, 123);

        assert_eq!(sep, IndexKey(vec![30]));
        assert_eq!(keys_of(&left), vec![IndexKey(vec![10]), IndexKey(vec![20])]);
        assert_eq!(
            keys_of(&right),
            vec![IndexKey(vec![30]), IndexKey(vec![40])]
        );
        assert_eq!(left.tombstones, VecDeque::from([1u32]));
        assert_eq!(right.tombstones, VecDeque::from([1u32]));
        assert_eq!(left.next_leaf_id, 123);
        assert_eq!(right.next_leaf_id, 999);

        left.absorb_right(&mut right);
        assert_eq!(
            keys_of(&left),
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
        left.entries = vec![IndexKey(vec![10]), IndexKey(vec![20])]
            .into_iter()
            .zip(vec![
                RecordId {
                    page_id: 1,
                    slot_id: 0,
                },
                RecordId {
                    page_id: 1,
                    slot_id: 1,
                },
            ])
            .collect();
        left.tombstones = VecDeque::from([0u32]);
        left.check().unwrap();

        let mut right = LeafPage::new(10, 1);
        right.entries = vec![IndexKey(vec![30]), IndexKey(vec![40])]
            .into_iter()
            .zip(vec![
                RecordId {
                    page_id: 2,
                    slot_id: 0,
                },
                RecordId {
                    page_id: 2,
                    slot_id: 1,
                },
            ])
            .collect();
        right.tombstones = VecDeque::from([0u32]);
        right.check().unwrap();

        left.absorb_right(&mut right);
        assert_eq!(
            keys_of(&left),
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

        let mut right = InternalPage::new(10);
        right.children = vec![5, 6, 7];
        right.keys = vec![IndexKey(vec![90]), IndexKey(vec![100])];

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
        leaf.entries = vec![
            IndexKey(vec![0, 0, 0, 10]),
            IndexKey(vec![0, 0, 0, 20]),
            IndexKey(vec![0, 0, 0, 30]),
        ]
        .into_iter()
        .zip(vec![
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
        ])
        .collect();
        leaf.next_leaf_id = 123;
        leaf.encode(&mut buf, layout(key_len)).unwrap();

        let decoded = LeafPage::decode(&buf, layout(key_len)).unwrap();
        assert_eq!(decoded.len() as u32, 3);
        assert_eq!(decoded.max_size, 10);
        assert_eq!(decoded.next_leaf_id, 123);
        assert_eq!(keys_of(&decoded), keys_of(&leaf));
        assert_eq!(decoded.entries, leaf.entries);
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
        assert_eq!(leaf.len(), 3);
        assert_eq!(leaf.tombstones.len(), 1);
        assert_eq!(leaf.tombstones[0], 0);
        assert!(leaf.get(&IndexKey(vec![10])).is_none());

        assert!(leaf.remove_key(&IndexKey(vec![20])).is_ok());
        assert_eq!(leaf.len(), 3);
        assert_eq!(leaf.tombstones.len(), 2);
        assert_eq!(leaf.tombstones[0], 0);
        assert_eq!(leaf.tombstones[1], 1);
        assert!(leaf.get(&IndexKey(vec![20])).is_none());

        assert!(leaf.remove_key(&IndexKey(vec![30])).is_ok());
        assert_eq!(leaf.len(), 2, "the oldest tombstone (key 10) is purged");
        assert_eq!(
            Vec::from(leaf.tombstones.clone()),
            vec![0, 1],
            "keys 20 and 30 shift down one slot after key 10 is purged"
        );
        assert!(leaf.get(&IndexKey(vec![30])).is_none());

        let mut buf = [0u8; PAGE_SIZE];
        leaf.encode(&mut buf, layout(1)).unwrap();
        let decoded = LeafPage::decode(&buf, layout(1)).unwrap();
        assert_eq!(decoded.len(), 2);
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
        leaf.entries = (0..max_size as u32)
            .map(|i| (IndexKey(vec![0u8]), rid(i)))
            .collect();
        leaf.tombstones = VecDeque::from([0u32, 1u32]);
        leaf.check().unwrap();

        let mut buf = [0u8; PAGE_SIZE];
        let err = leaf.encode(&mut buf, layout(key_len)).unwrap_err();
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
        internal.encode(&mut buf, layout(key_len)).unwrap();

        let decoded = InternalPage::decode(&buf, layout(key_len)).unwrap();
        assert_eq!(decoded.len() as u32, 2);
        assert_eq!(decoded.max_size, 10);
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
            let max = layout(key_len).internal_max_keys().unwrap() as usize;
            let page_with = |n: usize| {
                let mut page = InternalPage::new(max as u32);
                page.keys = vec![IndexKey(vec![0; key_len as usize]); n];
                page.children = (0..=n).collect();
                page
            };
            let mut buf = [0u8; PAGE_SIZE];
            assert!(
                page_with(max).encode(&mut buf, layout(key_len)).is_ok(),
                "key_len {key_len}: max keys fit"
            );
            assert!(
                page_with(max + 1)
                    .encode(&mut buf, layout(key_len))
                    .is_err(),
                "key_len {key_len}: one more key overflows"
            );
        }
    }

    #[test]
    fn leaf_max_size_for_layout_leaves_room_for_the_tombstone_count() {
        for key_len in [1u32, 4, 8, 37, 255] {
            let max = layout(key_len).leaf_max_entries().unwrap() as usize;
            let entry_bytes = key_len as usize + RECORD_ID_ENCODED_BYTES;
            let used = LEAF_FIXED_HEADER_BYTES + max * entry_bytes + TOMBSTONE_COUNT_ENCODED_BYTES;
            assert!(used <= PAGE_SIZE, "key_len {key_len}: max entries fit");
            assert!(
                used + entry_bytes > PAGE_SIZE,
                "key_len {key_len}: one more entry overflows"
            );
        }
        assert_eq!(
            Layout::new(0),
            Err(PageCodecError::Malformed("key_len must be > 0"))
        );
        assert_eq!(layout(PAGE_SIZE as u32).leaf_max_entries(), None);
        assert_eq!(layout(PAGE_SIZE as u32).internal_max_keys(), None);
    }

    #[test]
    fn tombstone_capacity_fills_the_space_beside_a_full_leaf() {
        let key_len = 8u32;
        let max_size = layout(key_len).leaf_max_entries().unwrap();
        let free = PAGE_SIZE
            - LEAF_FIXED_HEADER_BYTES
            - max_size as usize * (key_len as usize + RECORD_ID_ENCODED_BYTES)
            - TOMBSTONE_COUNT_ENCODED_BYTES;
        let fit = (free / TOMBSTONE_SLOT_ENCODED_BYTES) as u32;
        assert_eq!(layout(key_len).tombstone_capacity(max_size, u32::MAX), fit);
        assert_eq!(layout(key_len).tombstone_capacity(max_size, 1), 1.min(fit));
        assert_eq!(layout(key_len).tombstone_capacity(u32::MAX, 10), 0);
    }

    #[test]
    fn find_child_index_counts_keys_less_than_or_equal_to_the_search_key() {
        let mut page = InternalPage::new(10);
        page.children = vec![1, 2, 3];
        page.keys = vec![IndexKey(vec![10]), IndexKey(vec![20])];
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
        assert_eq!(keys_of(&leaf), vec![IndexKey(vec![2])]);
        assert!(leaf.tombstones.is_empty());
    }

    #[test]
    fn leaf_encoding_always_writes_a_tombstone_count() {
        let mut leaf = LeafPage::new(10, NUM_TOMBSTONES);
        leaf.insert(IndexKey(vec![7]), rid(1)).unwrap();
        let mut buf = [0xFFu8; PAGE_SIZE];
        leaf.encode(&mut buf, layout(1)).unwrap();
        let count_at = LEAF_FIXED_HEADER_BYTES + 1 + RECORD_ID_ENCODED_BYTES;
        assert_eq!(
            &buf[count_at..count_at + TOMBSTONE_COUNT_ENCODED_BYTES],
            &[0, 0, 0, 0]
        );
        assert!(
            LeafPage::decode(&buf, layout(1))
                .unwrap()
                .tombstones
                .is_empty()
        );
    }

    fn encoded_internal(key_len: u32, keys: &[u8]) -> [u8; PAGE_SIZE] {
        let mut page = InternalPage::new_root(10, 0, IndexKey(vec![keys[0]; key_len as usize]), 1);
        for (child, &k) in (2..).zip(&keys[1..]) {
            page.append_key_and_child(IndexKey(vec![k; key_len as usize]), child);
        }
        let mut buf = [0u8; PAGE_SIZE];
        page.encode(&mut buf, layout(key_len)).unwrap();
        buf
    }

    #[test]
    fn internal_decode_reports_corrupt_pages_as_errors() {
        let key_len = 2u32;
        let good = encoded_internal(key_len, &[10, 20, 30]);
        let lay = layout(key_len);
        let size_off = crate::index::page_codec::COMMON_HEADER_BYTES;
        let max_size_off = size_off + SIZE_FIELD_BYTES;
        let child_count_off = max_size_off + MAX_SIZE_FIELD_BYTES;

        let mut unsorted = good;
        unsorted[lay.internal_key_offset(4, 0)..lay.internal_key_offset(4, 1)].fill(99);
        assert_eq!(
            InternalPage::decode(&unsorted, lay).unwrap_err(),
            PageCodecError::Malformed("internal keys out of order")
        );

        let mut over_max = good;
        over_max[max_size_off..max_size_off + 4].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            InternalPage::decode(&over_max, lay).unwrap_err(),
            PageCodecError::Malformed("internal size exceeds max size")
        );

        let mut huge = good;
        huge[size_off..size_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        huge[child_count_off..child_count_off + 4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            InternalPage::decode(&huge, lay).unwrap_err(),
            PageCodecError::Malformed("internal page has keys but zero children")
        );

        let mut too_many_children = good;
        let children = u32::MAX - 1;
        too_many_children[size_off..size_off + 4].copy_from_slice(&(children - 1).to_le_bytes());
        too_many_children[child_count_off..child_count_off + 4]
            .copy_from_slice(&children.to_le_bytes());
        assert!(matches!(
            InternalPage::decode(&too_many_children, lay),
            Err(PageCodecError::BufferError(
                BufferError::BufferTooSmall { .. }
            ))
        ));
    }

    #[test]
    fn leaf_decode_reports_bad_tombstones_as_errors() {
        let key_len = 1u32;
        let lay = layout(key_len);
        let mut leaf = LeafPage::new(10, 2);
        for k in [1u8, 2, 3] {
            leaf.insert(IndexKey(vec![k]), rid(u32::from(k))).unwrap();
        }
        leaf.remove_key(&IndexKey(vec![2])).unwrap();
        let mut good = [0u8; PAGE_SIZE];
        leaf.encode(&mut good, lay).unwrap();
        let count_off = lay.leaf_tombstone_count_offset(3);

        let mut past_end = good;
        let first = lay.leaf_tombstone_offset(3, 0);
        past_end[first..first + 4].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            LeafPage::decode(&past_end, lay).unwrap_err(),
            PageCodecError::Malformed("leaf tombstones index out of bounds")
        );

        let mut over_capacity = good;
        let cap = lay.tombstone_capacity(10, NUM_TOMBSTONES);
        over_capacity[count_off..count_off + 4].copy_from_slice(&(cap + 1).to_le_bytes());
        assert_eq!(
            LeafPage::decode(&over_capacity, lay).unwrap_err(),
            PageCodecError::Malformed("leaf tombstones count exceeds tombstone capacity")
        );

        let mut past_page = good;
        past_page[count_off..count_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            LeafPage::decode(&past_page, lay).unwrap_err(),
            PageCodecError::Malformed("leaf tombstones section exceeds remaining bytes")
        );
    }

    #[test]
    fn decoding_a_short_or_unknown_buffer_is_an_error() {
        let lay = layout(1);
        assert!(matches!(
            Page::decode(&[0u8; 3], lay),
            Err(PageCodecError::BufferError(
                BufferError::BufferTooSmall { .. }
            ))
        ));
        assert!(Page::kind(&[]).is_err());
        let mut header = [0u8; PAGE_SIZE];
        HeaderPage {
            root_page_id: 1,
            first_leaf_id: 1,
            key_len: 1,
        }
        .encode(&mut header)
        .unwrap();
        assert_eq!(
            Page::kind(&header).err(),
            Some(PageCodecError::InvalidPageKind(KIND_HEADER))
        );
        assert!(LeafPage::decode(&[0u8; 3], lay).is_err());
        assert!(InternalPage::decode(&[0u8; 3], lay).is_err());
    }

    #[test]
    fn encoding_a_page_that_fails_its_check_is_an_error() {
        let lay = layout(1);
        let mut buf = [0u8; PAGE_SIZE];

        let mut leaf = LeafPage::new(10, 0);
        leaf.entries = vec![(IndexKey(vec![2]), rid(2)), (IndexKey(vec![1]), rid(1))];
        assert_eq!(
            leaf.encode(&mut buf, lay),
            Err(PageCodecError::Malformed("leaf keys out of order"))
        );

        let mut internal = InternalPage::new(10);
        internal.keys = vec![IndexKey(vec![1])];
        assert_eq!(
            internal.encode(&mut buf, lay),
            Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != size + 1"
            ))
        );

        let wrong_width = InternalPage::new_root(10, 0, IndexKey(vec![1, 2]), 1);
        assert_eq!(
            wrong_width.encode(&mut buf, lay),
            Err(PageCodecError::InvalidKeyLen {
                expected: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn layout_sizes_and_offsets_match_the_encoded_pages() {
        for key_len in [1u32, 3, 8, 255] {
            let lay = layout(key_len);
            let keys = [10u8, 20, 30];
            let buf = encoded_internal(key_len, &keys);
            for (i, &k) in keys.iter().enumerate() {
                let off = lay.internal_key_offset(keys.len() + 1, i);
                assert_eq!(buf[off..off + key_len as usize], vec![k; key_len as usize]);
            }
            for child in 0..=keys.len() {
                let off = lay.internal_child_offset(child);
                assert_eq!(
                    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()),
                    child as u64
                );
            }
            assert_eq!(
                lay.internal_bytes(4, 3).unwrap(),
                lay.internal_key_offset(4, 3)
            );

            let mut leaf = LeafPage::new(10, NUM_TOMBSTONES);
            for k in keys {
                leaf.insert(IndexKey(vec![k; key_len as usize]), rid(u32::from(k)))
                    .unwrap();
            }
            leaf.remove_key(&IndexKey(vec![20; key_len as usize]))
                .unwrap();
            let mut buf = [0xAAu8; PAGE_SIZE];
            leaf.encode(&mut buf, lay).unwrap();
            let end = lay.leaf_bytes(3, 1).unwrap();
            assert_eq!(end, lay.leaf_tombstone_offset(3, 1));
            assert!(
                buf[end..].iter().all(|&b| b == 0),
                "key_len {key_len}: nothing is written past leaf_bytes"
            );
            let first_tombstone = lay.leaf_tombstone_offset(3, 0);
            assert_eq!(
                buf[first_tombstone..first_tombstone + 4],
                1u32.to_le_bytes()
            );
            let second_entry = lay.leaf_entry_offset(1);
            assert_eq!(
                buf[second_entry..second_entry + key_len as usize],
                vec![20; key_len as usize]
            );
        }
        assert_eq!(
            layout(1).leaf_live_bytes(usize::MAX),
            Err(crate::index::page_codec::OVERFLOW)
        );
        for entries in 0..PAGE_SIZE {
            let live = layout(1).leaf_live_bytes(entries).unwrap();
            assert_eq!(
                layout(1).leaf_has_tombstone_section(entries),
                live + TOMBSTONE_COUNT_ENCODED_BYTES <= PAGE_SIZE,
                "{entries} entries"
            );
        }
    }

    #[test]
    fn page_codec_error_converts_failed_int_conversions_to_overflow() {
        let err: PageCodecError = u32::try_from(u64::MAX).unwrap_err().into();
        assert_eq!(err, PageCodecError::BufferError(BufferError::Overflow));
    }
}
