use crate::buf::{BufReader, BufWriter, BufferError, put_u32_le, put_u64_le};
use crate::catalog::index_schema::IndexKey;
use crate::disk::PAGE_SIZE;
use crate::ids::RecordId;
use crate::index::page_codec::{
    COMMON_HEADER_BYTES, KIND_HEADER, KIND_INTERNAL, KIND_LEAF, PageCodecError, U32_BYTES,
    U64_BYTES, decode_child_page_id, decode_page_id, decode_record_id, encode_page_id,
    encode_record_id, peek_common_header, put_common_header, read_common_header_of_kind,
    write_common_header,
};
use crate::index::slotted::{EntryArea, SLOT_ENCODED_BYTES};
use crate::index::{InsertError, RemoveError};
use log::{error, info};
use std::collections::VecDeque;
use std::convert::TryFrom;
use std::num::NonZeroUsize;

pub use crate::ids::PageId;

/// How many deleted keys a leaf keeps as tombstones before it removes the oldest for good.
pub const LEAF_TOMBSTONE_CAPACITY: u32 = 16;

pub(crate) const PAGE_ID_ENCODED_BYTES: usize = U64_BYTES;
pub(crate) const RECORD_ID_ENCODED_BYTES: usize = 2 * U64_BYTES;
pub(crate) const TOMBSTONE_COUNT_ENCODED_BYTES: usize = U32_BYTES;
pub(crate) const TOMBSTONE_SLOT_ENCODED_BYTES: usize = U32_BYTES;

pub(crate) const SIZE_FIELD_BYTES: usize = U32_BYTES;
const RESERVED_FIELD_BYTES: usize = U32_BYTES;
const KEY_LEN_FIELD_BYTES: usize = U32_BYTES;

pub(crate) const NODE_SIZE_OFFSET: usize = COMMON_HEADER_BYTES;
/// Offset of a leaf's next-leaf id and of an internal page's leftmost child id.
pub(crate) const NODE_PAGE_ID_OFFSET: usize =
    COMMON_HEADER_BYTES + SIZE_FIELD_BYTES + RESERVED_FIELD_BYTES;

pub(crate) const INTERNAL_FIXED_HEADER_BYTES: usize = NODE_PAGE_ID_OFFSET + PAGE_ID_ENCODED_BYTES;
pub(crate) const LEAF_FIXED_HEADER_BYTES: usize = NODE_PAGE_ID_OFFSET + PAGE_ID_ENCODED_BYTES;
pub(crate) const HEADER_FIXED_HEADER_BYTES: usize = COMMON_HEADER_BYTES
    + KEY_LEN_FIELD_BYTES
    + RESERVED_FIELD_BYTES
    + PAGE_ID_ENCODED_BYTES
    + PAGE_ID_ENCODED_BYTES;

/// The fewest entries of the longest key a leaf and an internal page each hold. Splits and
/// merges by bytes keep every page at least half as full as this allows.
const MIN_ENTRIES_PER_PAGE: usize = 4;

const fn leaf_slots_start(tombstone_capacity: usize) -> usize {
    LEAF_FIXED_HEADER_BYTES
        + TOMBSTONE_COUNT_ENCODED_BYTES
        + tombstone_capacity * TOMBSTONE_SLOT_ENCODED_BYTES
}

const INTERNAL_AREA: EntryArea = EntryArea::new(INTERNAL_FIXED_HEADER_BYTES, PAGE_ID_ENCODED_BYTES);

/// The longest key whose entries fit `MIN_ENTRIES_PER_PAGE` times in an entry area.
const fn longest_key_for(area: EntryArea) -> usize {
    area.capacity() / MIN_ENTRIES_PER_PAGE - area.entry_cost(0)
}

/// The longest key a B+ tree with the default tombstone capacity can hold.
pub const MAX_KEY_LEN: usize = {
    let leaf = longest_key_for(EntryArea::new(
        leaf_slots_start(LEAF_TOMBSTONE_CAPACITY as usize),
        RECORD_ID_ENCODED_BYTES,
    ));
    let internal = longest_key_for(INTERNAL_AREA);
    if leaf < internal { leaf } else { internal }
};

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

/// Which end of a page an entry is taken from when a sibling lends one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    First,
    Last,
}

/// The byte layout of the internal and leaf pages of a tree whose keys are at most
/// `max_key_len` bytes long, and the byte thresholds its splits and merges keep pages within.
/// The owned codecs here and the in-place views in `page_view` both read the layout from it.
///
/// A leaf is a fixed header (size, next leaf id), a tombstone section of `tombstone_capacity`
/// slots, then an [`EntryArea`] of `{key, page_id u64, slot_id u64}` entries. An internal page is
/// a fixed header (size, leftmost child id) then an [`EntryArea`] of `{key, child u64}` entries,
/// each child to the right of its key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    max_key_len: NonZeroUsize,
    tombstone_capacity: usize,
}

impl Layout {
    /// Fails with `Malformed` for a zero `max_key_len` or one past [`MAX_KEY_LEN`].
    pub fn new(max_key_len: u32) -> Result<Self, PageCodecError> {
        Self::build(max_key_len, LEAF_TOMBSTONE_CAPACITY)
    }

    /// Like [`Layout::new`], with leaves keeping `tombstones` tombstones.
    #[cfg(test)]
    pub(crate) fn with_tombstone_capacity(
        max_key_len: u32,
        tombstones: u32,
    ) -> Result<Self, PageCodecError> {
        Self::build(max_key_len, tombstones)
    }

    fn build(max_key_len: u32, tombstones: u32) -> Result<Self, PageCodecError> {
        let max_key_len = NonZeroUsize::new(usize::try_from(max_key_len)?)
            .ok_or(PageCodecError::Malformed("key_len must be > 0"))?;
        let tombstone_capacity = usize::try_from(tombstones)?;
        let leaf_slots = TOMBSTONE_SLOT_ENCODED_BYTES
            .checked_mul(tombstone_capacity)
            .and_then(|bytes| bytes.checked_add(leaf_slots_start(0)))
            .filter(|&start| start < PAGE_SIZE)
            .ok_or(PageCodecError::Malformed("tombstones do not fit in a leaf"))?;
        let leaf_area = EntryArea::new(leaf_slots, RECORD_ID_ENCODED_BYTES);
        if max_key_len.get() > longest_key_for(leaf_area).min(longest_key_for(INTERNAL_AREA)) {
            return Err(PageCodecError::Malformed(
                "keys that long do not fit four to a page",
            ));
        }
        Ok(Self {
            max_key_len,
            tombstone_capacity,
        })
    }

    pub fn max_key_len(self) -> usize {
        self.max_key_len.get()
    }

    pub fn tombstone_capacity(self) -> usize {
        self.tombstone_capacity
    }

    pub fn leaf_area(self) -> EntryArea {
        EntryArea::new(
            leaf_slots_start(self.tombstone_capacity),
            RECORD_ID_ENCODED_BYTES,
        )
    }

    pub fn internal_area(self) -> EntryArea {
        INTERNAL_AREA
    }

    /// Offset of the tombstone count in a leaf page.
    pub fn leaf_tombstone_count_offset(self) -> usize {
        LEAF_FIXED_HEADER_BYTES
    }

    /// Offset of tombstone `i` in a leaf page.
    pub fn leaf_tombstone_offset(self, i: usize) -> usize {
        LEAF_FIXED_HEADER_BYTES + TOMBSTONE_COUNT_ENCODED_BYTES + i * TOMBSTONE_SLOT_ENCODED_BYTES
    }

    /// Bytes the largest leaf entry takes, its slot included.
    pub fn leaf_max_entry_bytes(self) -> usize {
        self.leaf_area().entry_cost(self.max_key_len())
    }

    /// Bytes the largest internal entry takes, its slot included.
    pub fn internal_max_entry_bytes(self) -> usize {
        INTERNAL_AREA.entry_cost(self.max_key_len())
    }

    /// The fewest entry bytes a leaf other than the root holds. An underfull leaf that cannot
    /// borrow from its sibling merges with it, and the two fit in one page.
    pub fn leaf_min_used(self) -> usize {
        (self.leaf_area().capacity() - self.leaf_max_entry_bytes()) / 2
    }

    /// The fewest entry bytes an internal page other than the root holds. An underfull page
    /// that cannot borrow merges with its sibling and their separator, which fit in one
    /// page.
    pub fn internal_min_used(self) -> usize {
        (INTERNAL_AREA.capacity() - 2 * self.internal_max_entry_bytes()) / 2
    }

    fn check_key(self, key: &IndexKey) -> Result<(), PageCodecError> {
        if key.0.len() > self.max_key_len() {
            return Err(PageCodecError::KeyTooLong {
                max: self.max_key_len(),
                actual: key.0.len(),
            });
        }
        Ok(())
    }
}

/// The index that splits entries costing `costs` into two runs of nearly equal bytes: the
/// first `split` entries go left. With `drop_middle`, the entry at `split` goes to neither
/// side, as the separator an internal split moves up.
fn balanced_split(costs: &[usize], drop_middle: bool) -> usize {
    let total: usize = costs.iter().sum();
    let mut best = (usize::MAX, 1);
    let mut left = 0usize;
    for (split, &cost) in costs.iter().enumerate() {
        let right = total - left - if drop_middle { cost } else { 0 };
        if split > 0 {
            let gap = left.abs_diff(right);
            if gap < best.0 {
                best = (gap, split);
            }
        }
        left += cost;
    }
    best.1
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

#[derive(Debug, Clone)]
pub struct InternalPage {
    layout: Layout,
    children: Vec<PageId>,
    keys: Vec<IndexKey>,
}

impl InternalPage {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            children: Vec::new(),
            keys: Vec::new(),
        }
    }

    /// A root over `left` and `right`, separated by `separator`.
    pub fn new_root(layout: Layout, left: PageId, separator: IndexKey, right: PageId) -> Self {
        let root = Self {
            layout,
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

    fn cost(&self, key: &IndexKey) -> usize {
        INTERNAL_AREA.entry_cost(key.0.len())
    }

    /// Bytes the keys and their right children take in the page.
    pub fn used_bytes(&self) -> usize {
        self.keys.iter().map(|k| self.cost(k)).sum()
    }

    /// Fails with `Malformed` when the children do not number one more than the keys (unless
    /// both are empty) or the keys are out of order, and with `KeyTooLong` for a key longer
    /// than the layout holds.
    pub fn check(&self) -> Result<(), PageCodecError> {
        if self.children.is_empty() && self.keys.is_empty() {
            return Ok(());
        }
        if self.children.len() != self.keys.len() + 1 {
            return Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != size + 1",
            ));
        }
        if !self.keys.is_sorted() {
            return Err(PageCodecError::Malformed("internal keys out of order"));
        }
        self.keys
            .iter()
            .try_for_each(|key| self.layout.check_key(key))
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

    /// Moves the keys and children to the right of a middle key into the empty `right` page,
    /// removes the middle key from this page and returns it for the parent. The middle key is
    /// the one that leaves the two pages' key bytes closest to equal.
    pub fn split_into(&mut self, right: &mut InternalPage) -> IndexKey {
        assert!(
            right.children.is_empty() && right.keys.is_empty(),
            "right page must be empty before split"
        );
        assert!(self.keys.len() >= 3, "an internal split needs three keys");
        self.debug_check();

        let costs: Vec<usize> = self.keys.iter().map(|k| self.cost(k)).collect();
        let mid = balanced_split(&costs, true).min(self.keys.len() - 2);

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

    pub fn encode(&self, buf: &mut [u8]) -> Result<(), PageCodecError> {
        self.check()?;
        let Some(&leftmost) = self.children.first() else {
            return Err(PageCodecError::Malformed("internal page has no children"));
        };
        let used = self.used_bytes();
        ensure_fits_page(INTERNAL_FIXED_HEADER_BYTES + used)?;
        info!(
            "bptree/page_encode kind=internal keys={} children={}",
            self.keys.len(),
            self.children.len()
        );

        if buf.len() != PAGE_SIZE {
            return Err(PageCodecError::InvalidSize);
        }
        buf.fill(0);
        put_common_header(buf, KIND_INTERNAL);
        put_u32_le(buf, NODE_SIZE_OFFSET, u32::try_from(self.keys.len())?);
        put_u64_le(buf, NODE_PAGE_ID_OFFSET, encode_page_id(Some(leftmost)));

        let key_bytes: usize = self.keys.iter().map(|k| k.0.len()).sum();
        let mut at = PAGE_SIZE - key_bytes - self.keys.len() * PAGE_ID_ENCODED_BYTES;
        for (i, (key, &child)) in self.keys.iter().zip(&self.children[1..]).enumerate() {
            let child = encode_page_id(Some(child)).to_le_bytes();
            at = INTERNAL_AREA.put_entry(buf, i, at, &key.0, &child);
        }
        debug_assert_eq!(at, PAGE_SIZE);
        Ok(())
    }

    pub fn decode(buf: &[u8], layout: Layout) -> Result<Self, PageCodecError> {
        let mut r = BufReader::new(buf)?;
        read_common_header_of_kind(&mut r, KIND_INTERNAL)?;
        let size = usize::try_from(r.read_u32_le()?)?;
        let _reserved = r.read_u32_le()?;
        let leftmost = decode_child_page_id(r.read_u64_le()?)?;
        info!("bptree/page_decode kind=internal size={size}");
        INTERNAL_AREA.check(buf, size)?;

        let mut children = Vec::with_capacity(size + 1);
        children.push(leftmost);
        let mut keys = Vec::with_capacity(size);
        for i in 0..size {
            keys.push(IndexKey(INTERNAL_AREA.key(buf, size, i).to_vec()));
            let child = INTERNAL_AREA.payload(buf, size, i);
            children.push(decode_child_page_id(u64::from_le_bytes(
                child.try_into().map_err(|_| PageCodecError::InvalidSize)?,
            ))?);
        }

        let page = Self {
            layout,
            children,
            keys,
        };
        page.check()?;
        Ok(page)
    }

    /// Whether the keys take more bytes than one page holds, so the page splits.
    pub fn is_overfull(&self) -> bool {
        self.used_bytes() > INTERNAL_AREA.capacity()
    }

    pub fn is_underfull(&self) -> bool {
        self.used_bytes() < self.layout.internal_min_used()
    }

    /// Whether the page stays at least minimally full after giving up the key at `edge` (and
    /// the child beside it) to a sibling.
    pub fn can_lend(&self, edge: Edge) -> bool {
        let key = match edge {
            Edge::First => self.keys.first(),
            Edge::Last => self.keys.last(),
        };
        key.is_some_and(|k| self.used_bytes() - self.cost(k) >= self.layout.internal_min_used())
    }

    /// Whether replacing key `idx` with `new_key` leaves the keys within one page.
    pub fn fits_replacing(&self, idx: usize, new_key: &IndexKey) -> bool {
        let old = self.cost(&self.keys[idx]);
        self.used_bytes() - old + self.cost(new_key) <= INTERNAL_AREA.capacity()
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
}

#[derive(Debug, Clone)]
pub struct LeafPage {
    layout: Layout,
    entries: Vec<(IndexKey, RecordId)>,
    /// Page id of the next leaf to the right, or `None` for the last leaf.
    next_leaf_id: Option<PageId>,
    /// Slots of deleted keys still stored in the page, oldest deletion first.
    tombstones: VecDeque<u32>,
}

impl LeafPage {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            entries: Vec::new(),
            next_leaf_id: None,
            tombstones: VecDeque::with_capacity(layout.tombstone_capacity()),
        }
    }

    /// The number of entries, tombstoned ones included.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn next_leaf_id(&self) -> Option<PageId> {
        self.next_leaf_id
    }

    #[cfg(test)]
    pub(crate) fn set_next_leaf_id(&mut self, next_leaf_id: Option<PageId>) {
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
        self.tombstones.len() >= self.layout.tombstone_capacity()
    }

    fn cost(&self, key: &IndexKey) -> usize {
        self.layout.leaf_area().entry_cost(key.0.len())
    }

    /// Bytes the entries take in the page, tombstoned ones included.
    pub fn used_bytes(&self) -> usize {
        self.keys().map(|k| self.cost(k)).sum()
    }

    /// Fails with `Malformed` when the keys are out of order, the tombstones exceed their
    /// capacity, or a tombstone names a slot past the last entry, and with `KeyTooLong` for a
    /// key longer than the layout holds.
    pub fn check(&self) -> Result<(), PageCodecError> {
        if !self.keys().is_sorted() {
            return Err(PageCodecError::Malformed("leaf keys out of order"));
        }
        if self.tombstones.len() > self.layout.tombstone_capacity() {
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
        self.keys().try_for_each(|key| self.layout.check_key(key))
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

    pub fn encode(&self, buf: &mut [u8]) -> Result<(), PageCodecError> {
        self.check()?;
        let area = self.layout.leaf_area();
        let used = self.used_bytes();
        ensure_fits_page(PAGE_SIZE - area.capacity() + used)?;
        info!(
            "bptree/page_encode kind=leaf keys={} tombstones={}",
            self.entries.len(),
            self.tombstones.len()
        );

        if buf.len() != PAGE_SIZE {
            return Err(PageCodecError::InvalidSize);
        }
        buf.fill(0);
        put_common_header(buf, KIND_LEAF);
        put_u32_le(buf, NODE_SIZE_OFFSET, u32::try_from(self.entries.len())?);
        put_u64_le(buf, NODE_PAGE_ID_OFFSET, encode_page_id(self.next_leaf_id));
        put_u32_le(
            buf,
            self.layout.leaf_tombstone_count_offset(),
            u32::try_from(self.tombstones.len())?,
        );
        for (i, &slot) in self.tombstones.iter().enumerate() {
            put_u32_le(buf, self.layout.leaf_tombstone_offset(i), slot);
        }

        let mut at = PAGE_SIZE - (used - self.entries.len() * SLOT_ENCODED_BYTES);
        for (i, (key, rid)) in self.entries.iter().enumerate() {
            at = area.put_entry(buf, i, at, &key.0, &encode_record_id_bytes(rid)?);
        }
        debug_assert_eq!(at, PAGE_SIZE);
        Ok(())
    }

    pub fn decode(buf: &[u8], layout: Layout) -> Result<Self, PageCodecError> {
        let mut r = BufReader::new(buf)?;
        read_common_header_of_kind(&mut r, KIND_LEAF)?;
        let size = usize::try_from(r.read_u32_le()?)?;
        let _reserved = r.read_u32_le()?;
        let next_leaf_id = decode_page_id(r.read_u64_le()?)?;
        info!("bptree/page_decode kind=leaf size={size} next_leaf_id={next_leaf_id:?}");
        let area = layout.leaf_area();
        area.check(buf, size)?;

        let tombstone_count = usize::try_from(r.read_u32_le()?)?;
        if tombstone_count > layout.tombstone_capacity() {
            return Err(PageCodecError::Malformed(
                "leaf tombstones count exceeds tombstone capacity",
            ));
        }
        let tombstones = (0..tombstone_count)
            .map(|_| Ok(r.read_u32_le()?))
            .collect::<Result<VecDeque<u32>, PageCodecError>>()?;

        let mut entries = Vec::with_capacity(size);
        for i in 0..size {
            let key = IndexKey(area.key(buf, size, i).to_vec());
            entries.push((key, decode_record_id_bytes(area.payload(buf, size, i))?));
        }

        let page = Self {
            layout,
            entries,
            next_leaf_id,
            tombstones,
        };
        page.check()?;
        Ok(page)
    }
}

/// A record id as the 16 payload bytes of a leaf entry.
pub(crate) fn encode_record_id_bytes(
    rid: &RecordId,
) -> Result<[u8; RECORD_ID_ENCODED_BYTES], PageCodecError> {
    let (page_id, slot_id) = encode_record_id(rid)?;
    let mut out = [0u8; RECORD_ID_ENCODED_BYTES];
    out[..U64_BYTES].copy_from_slice(&page_id.to_le_bytes());
    out[U64_BYTES..].copy_from_slice(&slot_id.to_le_bytes());
    Ok(out)
}

/// The record id in a leaf entry's payload, or an error if it does not fit in `u32`.
pub(crate) fn decode_record_id_bytes(payload: &[u8]) -> Result<RecordId, PageCodecError> {
    let half = |range: std::ops::Range<usize>| -> Result<u64, PageCodecError> {
        let bytes = payload.get(range).ok_or(PageCodecError::InvalidSize)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().map_err(|_| PageCodecError::InvalidSize)?,
        ))
    };
    decode_record_id(
        half(0..U64_BYTES)?,
        half(U64_BYTES..RECORD_ID_ENCODED_BYTES)?,
    )
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

    /// Splits this leaf into `right` and returns the separator key for the parent: the first
    /// key of `right`, which also stays in the leaf level. The split leaves the two leaves'
    /// entry bytes closest to equal.
    pub fn split_into(&mut self, right: &mut LeafPage, right_page_id: PageId) -> IndexKey {
        assert!(
            right.entries.is_empty(),
            "right page must be empty before split"
        );
        self.debug_check();
        assert!(self.entries.len() >= 2, "a leaf split needs two entries");

        let costs: Vec<usize> = self.keys().map(|k| self.cost(k)).collect();
        let mid = balanced_split(&costs, false);
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
        self.next_leaf_id = Some(right_page_id);

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
        while self.tombstones.len() > self.layout.tombstone_capacity() {
            self.purge_oldest_tombstone();
        }

        self.debug_check();
    }

    /// Whether the entries take more bytes than one page holds, so the leaf splits.
    pub fn is_overfull(&self) -> bool {
        self.used_bytes() > self.layout.leaf_area().capacity()
    }

    pub fn is_underfull(&self) -> bool {
        self.used_bytes() < self.layout.leaf_min_used()
    }

    /// Whether the leaf stays at least minimally full after giving up the entry at `edge`.
    pub fn can_lend(&self, edge: Edge) -> bool {
        let entry = match edge {
            Edge::First => self.entries.first(),
            Edge::Last => self.entries.last(),
        };
        entry.is_some_and(|(k, _)| self.used_bytes() - self.cost(k) >= self.layout.leaf_min_used())
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

        let page_keeps_no_tombstones = self.layout.tombstone_capacity() == 0;
        if page_keeps_no_tombstones {
            self.physical_remove_at(slot);
            self.debug_check();
            return Ok(());
        }

        if self.tombstones_full() {
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
    pub root_page_id: Option<PageId>,
    pub first_leaf_id: Option<PageId>,
    /// The longest key the tree holds.
    pub max_key_len: u32,
}

impl HeaderPage {
    pub fn encode(&self, buf: &mut [u8]) -> Result<(), PageCodecError> {
        Layout::new(self.max_key_len)?;
        info!(
            "bptree/page_encode kind=header root={:?} first_leaf={:?} max_key_len={}",
            self.root_page_id, self.first_leaf_id, self.max_key_len
        );
        buf.fill(0);
        let mut w = BufWriter::new(buf)?;
        write_common_header(&mut w, KIND_HEADER)?;
        w.write_u32_le(self.max_key_len)?;
        w.write_u32_le(0)?;
        w.write_u64_le(encode_page_id(self.root_page_id))?;
        w.write_u64_le(encode_page_id(self.first_leaf_id))?;
        debug_assert_eq!(w.pos(), HEADER_FIXED_HEADER_BYTES);
        Ok(())
    }

    pub fn decode(buf: &[u8]) -> Result<Self, PageCodecError> {
        let mut r = BufReader::new(buf)?;
        read_common_header_of_kind(&mut r, KIND_HEADER)?;
        let max_key_len = r.read_u32_le()?;
        let _reserved = r.read_u32_le()?;
        let root_page_id = decode_page_id(r.read_u64_le()?)?;
        let first_leaf_id = decode_page_id(r.read_u64_le()?)?;
        info!(
            "bptree/page_decode kind=header root={:?} first_leaf={:?} max_key_len={}",
            root_page_id, first_leaf_id, max_key_len
        );
        if max_key_len == 0 {
            return Err(PageCodecError::Malformed("header key_len must be > 0"));
        }
        Ok(Self {
            root_page_id,
            first_leaf_id,
            max_key_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::page_codec::COMMON_HEADER_BYTES;
    use crate::testing::pid;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn layout(max_key_len: u32) -> Layout {
        Layout::new(max_key_len).unwrap()
    }

    fn key(bytes: &[u8]) -> IndexKey {
        IndexKey(bytes.to_vec())
    }

    fn rid(n: u32) -> RecordId {
        RecordId {
            page_id: pid(n),
            slot_id: n.wrapping_mul(3),
        }
    }

    fn keys_of(leaf: &LeafPage) -> Vec<IndexKey> {
        leaf.keys().cloned().collect()
    }

    fn leaf_with(l: Layout, keys: &[IndexKey]) -> LeafPage {
        let mut leaf = LeafPage::new(l);
        for (i, k) in keys.iter().enumerate() {
            leaf.insert(k.clone(), rid(i as u32)).unwrap();
        }
        leaf
    }

    fn internal_with(l: Layout, keys: &[IndexKey]) -> InternalPage {
        let mut page = InternalPage::new(l);
        page.keys = keys.to_vec();
        page.children = (0..=keys.len()).map(pid).collect();
        page.check().unwrap();
        page
    }

    #[test]
    fn internal_split_of_equal_keys_moves_the_middle_key_up() {
        let l = layout(1);
        let mut left = internal_with(
            l,
            &[key(&[10]), key(&[20]), key(&[30]), key(&[40]), key(&[50])],
        );
        let mut right = InternalPage::new(l);
        let sep = left.split_into(&mut right);

        assert_eq!(sep, key(&[30]));
        assert_eq!(left.keys, vec![key(&[10]), key(&[20])]);
        assert_eq!(left.children, [0, 1, 2].map(pid));
        assert_eq!(right.keys, vec![key(&[40]), key(&[50])]);
        assert_eq!(right.children, [3, 4, 5].map(pid));
    }

    #[test]
    fn splits_balance_bytes_rather_than_entry_counts() {
        let l = layout(500);
        let long = |b: u8| IndexKey(vec![b; 500]);

        let mut left = internal_with(l, &[long(1), key(&[2]), key(&[3]), key(&[4]), key(&[5])]);
        let mut right = InternalPage::new(l);
        let sep = left.split_into(&mut right);
        assert_eq!(sep, key(&[2]), "the long key is as many bytes as the rest");
        assert_eq!(left.keys, vec![long(1)]);
        assert_eq!(right.keys, vec![key(&[3]), key(&[4]), key(&[5])]);

        let mut leaf = leaf_with(l, &[key(&[1]), key(&[2]), key(&[3]), key(&[4]), long(5)]);
        let mut right = LeafPage::new(l);
        let sep = leaf.split_into(&mut right, pid(9));
        assert_eq!(sep, long(5));
        assert_eq!(leaf.len(), 4);
        assert_eq!(right.len(), 1);
    }

    #[test]
    fn leaf_split_and_merge_carry_tombstones_and_links() {
        let l = layout(1);
        let mut left = leaf_with(l, &[key(&[10]), key(&[20]), key(&[30]), key(&[40])]);
        left.next_leaf_id = Some(pid(999));
        left.tombstones = VecDeque::from([1u32, 3u32]);
        left.check().unwrap();

        let mut right = LeafPage::new(l);
        let sep = left.split_into(&mut right, pid(123));

        assert_eq!(sep, key(&[30]));
        assert_eq!(keys_of(&left), vec![key(&[10]), key(&[20])]);
        assert_eq!(keys_of(&right), vec![key(&[30]), key(&[40])]);
        assert_eq!(left.tombstones, VecDeque::from([1u32]));
        assert_eq!(right.tombstones, VecDeque::from([1u32]));
        assert_eq!(left.next_leaf_id, Some(pid(123)));
        assert_eq!(right.next_leaf_id, Some(pid(999)));

        left.absorb_right(&mut right);
        assert_eq!(
            keys_of(&left),
            vec![key(&[10]), key(&[20]), key(&[30]), key(&[40])]
        );
        assert_eq!(left.tombstones, VecDeque::from([1u32, 3u32]));
        assert_eq!(left.next_leaf_id, Some(pid(999)));

        let mut five = leaf_with(l, &[key(&[1]), key(&[2]), key(&[3]), key(&[4]), key(&[5])]);
        let mut right = LeafPage::new(l);
        five.split_into(&mut right, pid(1));
        assert_eq!((five.len(), right.len()), (2, 3));
    }

    #[test]
    fn leaf_merge_treats_absorbed_tombstones_as_more_recent() {
        let l = Layout::with_tombstone_capacity(1, 1).unwrap();
        let mut left = leaf_with(l, &[key(&[10]), key(&[20])]);
        left.tombstones = VecDeque::from([0u32]);
        let mut right = leaf_with(l, &[key(&[30]), key(&[40])]);
        right.tombstones = VecDeque::from([0u32]);

        left.absorb_right(&mut right);
        assert_eq!(keys_of(&left), vec![key(&[20]), key(&[30]), key(&[40])]);
        assert_eq!(
            left.tombstones,
            VecDeque::from([1u32]),
            "only the absorbed tombstone for key 30 survives, at its shifted slot"
        );
        assert!(
            !left.key_exists(&key(&[10])),
            "left's older tombstone is purged from the page"
        );
        assert!(left.get(&key(&[30])).is_none());
    }

    #[test]
    fn internal_merge_pulls_the_separator_down() {
        let l = layout(1);
        let mut left = internal_with(l, &[key(&[10]), key(&[20])]);
        let mut right = InternalPage::new(l);
        right.children = [5, 6, 7].map(pid).to_vec();
        right.keys = vec![key(&[90]), key(&[100])];

        left.absorb_right(key(&[50]), &mut right);

        assert_eq!(
            left.keys,
            vec![key(&[10]), key(&[20]), key(&[50]), key(&[90]), key(&[100])]
        );
        assert_eq!(left.children, [0, 1, 2, 5, 6, 7].map(pid));
        assert!(right.keys.is_empty() && right.children.is_empty());
    }

    #[test]
    fn header_page_round_trips_and_rejects_bad_magic() {
        let mut buf = [0u8; PAGE_SIZE];
        let hdr = HeaderPage {
            root_page_id: Some(pid(42)),
            first_leaf_id: None,
            max_key_len: 8,
        };
        hdr.encode(&mut buf).unwrap();
        let decoded = HeaderPage::decode(&buf).unwrap();
        assert_eq!(decoded.root_page_id, Some(pid(42)));
        assert_eq!(decoded.first_leaf_id, None);
        assert_eq!(decoded.max_key_len, 8);

        buf[0] ^= 0xff;
        assert!(matches!(
            HeaderPage::decode(&buf),
            Err(PageCodecError::BadMagic)
        ));
        let too_long = HeaderPage {
            max_key_len: u32::try_from(MAX_KEY_LEN + 1).unwrap(),
            ..hdr
        };
        assert!(too_long.encode(&mut buf).is_err());
    }

    #[test]
    fn leaf_and_internal_pages_round_trip_keys_of_any_length() {
        let l = layout(300);
        let keys = [
            key(&[]),
            key(&[0, 0, 0, 10]),
            key(&[0, 0, 0, 10, 0]),
            key(&[7; 300]),
            key(&[9]),
        ];
        let mut leaf = leaf_with(l, &keys);
        leaf.next_leaf_id = Some(pid(123));
        leaf.remove_key(&keys[2]).unwrap();
        let mut buf = [0xAAu8; PAGE_SIZE];
        leaf.encode(&mut buf).unwrap();
        let decoded = LeafPage::decode(&buf, l).unwrap();
        assert_eq!(decoded.entries, leaf.entries);
        assert_eq!(decoded.tombstones, leaf.tombstones);
        assert_eq!(decoded.next_leaf_id, Some(pid(123)));
        assert_eq!(decoded.used_bytes(), leaf.used_bytes());

        let internal = internal_with(l, &keys[1..]);
        internal.encode(&mut buf).unwrap();
        let decoded = InternalPage::decode(&buf, l).unwrap();
        assert_eq!(decoded.children, internal.children);
        assert_eq!(decoded.keys, internal.keys);
        let slots_end = INTERNAL_FIXED_HEADER_BYTES + 4 * SLOT_ENCODED_BYTES;
        let first_entry = PAGE_SIZE - (decoded.used_bytes() - 4 * SLOT_ENCODED_BYTES);
        assert!(
            buf[slots_end..first_entry].iter().all(|&b| b == 0),
            "free space between the slots and the entries is zeroed"
        );
    }

    #[test]
    fn leaf_delete_stores_tombstones_and_eviction_is_fifo() {
        let l = Layout::with_tombstone_capacity(1, 2).unwrap();
        let mut leaf = leaf_with(l, &[key(&[10]), key(&[20]), key(&[30])]);

        leaf.remove_key(&key(&[10])).unwrap();
        assert_eq!(leaf.len(), 3);
        assert_eq!(leaf.tombstones, VecDeque::from([0]));
        assert!(leaf.get(&key(&[10])).is_none());

        leaf.remove_key(&key(&[20])).unwrap();
        assert_eq!(leaf.len(), 3);
        assert_eq!(leaf.tombstones, VecDeque::from([0, 1]));

        leaf.remove_key(&key(&[30])).unwrap();
        assert_eq!(leaf.len(), 2, "the oldest tombstone (key 10) is purged");
        assert_eq!(
            leaf.tombstones,
            VecDeque::from([0, 1]),
            "keys 20 and 30 shift down one slot after key 10 is purged"
        );

        let mut buf = [0u8; PAGE_SIZE];
        leaf.encode(&mut buf).unwrap();
        let decoded = LeafPage::decode(&buf, l).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.tombstones, VecDeque::from([0, 1]));
        assert!(decoded.get(&key(&[20])).is_none());
        assert!(decoded.get(&key(&[30])).is_none());
    }

    #[test]
    fn fixed_header_sizes_match_the_encoded_field_layout() {
        assert_eq!(INTERNAL_FIXED_HEADER_BYTES, 8 + 2 * 4 + 8);
        assert_eq!(LEAF_FIXED_HEADER_BYTES, 8 + 2 * 4 + 8);
        assert_eq!(HEADER_FIXED_HEADER_BYTES, 8 + 2 * 4 + 2 * 8);
        assert_eq!(NODE_SIZE_OFFSET, COMMON_HEADER_BYTES);
        let l = layout(1);
        assert_eq!(l.leaf_tombstone_count_offset(), LEAF_FIXED_HEADER_BYTES);
        assert_eq!(
            PAGE_SIZE - l.leaf_area().capacity(),
            l.leaf_tombstone_offset(LEAF_TOMBSTONE_CAPACITY as usize)
        );
    }

    #[test]
    fn four_entries_of_the_longest_key_fit_a_leaf_and_an_internal_page() {
        assert_eq!(MAX_KEY_LEN, 2007);
        let max = u32::try_from(MAX_KEY_LEN).unwrap();
        assert!(Layout::new(max).is_ok());
        assert!(matches!(
            Layout::new(max + 1),
            Err(PageCodecError::Malformed(_))
        ));
        assert_eq!(
            Layout::new(0),
            Err(PageCodecError::Malformed("key_len must be > 0"))
        );
        assert!(Layout::with_tombstone_capacity(1, u32::MAX).is_err());

        let l = layout(max);
        let longest = |b: u8| IndexKey(vec![b; MAX_KEY_LEN]);
        let mut buf = [0u8; PAGE_SIZE];
        let mut leaf = leaf_with(l, &[longest(1), longest(2), longest(3), longest(4)]);
        assert!(!leaf.is_overfull());
        leaf.encode(&mut buf).unwrap();
        leaf.insert(longest(5), rid(5)).unwrap();
        assert!(leaf.is_overfull());
        assert!(matches!(
            leaf.encode(&mut buf),
            Err(PageCodecError::BufferError(
                BufferError::BufferTooSmall { .. }
            ))
        ));

        let mut internal = internal_with(l, &[longest(1), longest(2), longest(3), longest(4)]);
        assert!(!internal.is_overfull());
        internal.encode(&mut buf).unwrap();
        internal.append_key_and_child(longest(5), pid(5));
        assert!(internal.is_overfull());
        assert!(internal.encode(&mut buf).is_err());
    }

    #[test]
    fn fullness_thresholds_follow_entry_bytes() {
        let l = layout(2000);
        let long = |b: u8| IndexKey(vec![b; 2000]);
        let mut leaf = leaf_with(l, &[long(1)]);
        assert!(leaf.is_underfull());
        assert!(!leaf.can_lend(Edge::First));
        leaf.insert(long(2), rid(2)).unwrap();
        assert!(
            !leaf.is_underfull(),
            "two long entries are more than a third of a page"
        );
        assert!(!leaf.can_lend(Edge::Last));
        leaf.insert(key(&[0]), rid(0)).unwrap();
        assert!(leaf.can_lend(Edge::First), "a short entry is cheap to lend");
        assert!(!leaf.can_lend(Edge::Last));
        assert!(!LeafPage::new(l).can_lend(Edge::First));

        let internal = internal_with(l, &[key(&[1]), long(2), long(3), long(4), long(5)]);
        assert!(!internal.fits_replacing(0, &long(1)));
        assert!(internal.fits_replacing(1, &long(9)));
        assert!(internal.fits_replacing(0, &key(&[0; 30])));
        assert!(internal.can_lend(Edge::First));
        assert!(!internal_with(l, &[long(2), long(3)]).can_lend(Edge::Last));
    }

    /// An underfull page that cannot borrow from its sibling merges with it, so the two (and,
    /// for internal pages, their separator) fit in one page whatever their key lengths.
    #[test]
    fn a_page_that_cannot_borrow_always_merges_into_one_page() {
        let mut rng = StdRng::seed_from_u64(7);
        let l = layout(1500);
        let mut leaf_merges = 0;
        let mut internal_merges = 0;
        for _ in 0..4000 {
            let mut random_keys = |base: u8| -> Vec<IndexKey> {
                let n = rng.random_range(1..12);
                let mut keys: Vec<IndexKey> = (0..n)
                    .map(|i| {
                        let len = rng.random_range(1..=1500);
                        let mut k = vec![base, i];
                        k.resize(len.max(2), 0);
                        IndexKey(k)
                    })
                    .collect();
                keys.sort();
                keys.dedup();
                keys
            };
            let (a, b) = (random_keys(1), random_keys(3));

            let (mut left, mut right) = (leaf_with(l, &a), leaf_with(l, &b));
            if !left.is_overfull()
                && !right.is_overfull()
                && left.is_underfull()
                && !right.can_lend(Edge::First)
            {
                left.absorb_right(&mut right);
                assert!(!left.is_overfull());
                leaf_merges += 1;
            }

            let (mut left, mut right) = (internal_with(l, &a), internal_with(l, &b));
            let separator = IndexKey(vec![2; 1500]);
            if !left.is_overfull()
                && !right.is_overfull()
                && left.is_underfull()
                && !right.can_lend(Edge::First)
            {
                left.absorb_right(separator, &mut right);
                assert!(!left.is_overfull());
                internal_merges += 1;
            }
        }
        assert!(leaf_merges > 100 && internal_merges > 100);
    }

    #[test]
    fn find_child_index_counts_keys_less_than_or_equal_to_the_search_key() {
        let page = internal_with(layout(2), &[key(&[10]), key(&[20, 0])]);
        assert_eq!(page.find_child_index(&key(&[5])), 0);
        assert_eq!(page.find_child_index(&key(&[10])), 1);
        assert_eq!(page.find_child_index(&key(&[10, 0])), 1);
        assert_eq!(page.find_child_index(&key(&[20])), 1);
        assert_eq!(page.find_child_index(&key(&[20, 0])), 2);
        assert_eq!(page.find_child_index(&key(&[99])), 2);
    }

    #[test]
    fn leaf_insert_rejects_live_duplicates_and_revives_tombstoned_keys() {
        let mut leaf = LeafPage::new(layout(1));
        leaf.insert(key(&[1]), rid(1)).unwrap();
        assert!(matches!(
            leaf.insert(key(&[1]), rid(2)),
            Err(InsertError::DuplicateKey)
        ));
        leaf.remove_key(&key(&[1])).unwrap();
        leaf.insert(key(&[1]), rid(3)).unwrap();
        assert_eq!(leaf.get(&key(&[1])), Some(rid(3)));
        assert!(leaf.tombstones.is_empty());
    }

    #[test]
    fn leaf_without_tombstone_capacity_deletes_keys_immediately() {
        let l = Layout::with_tombstone_capacity(1, 0).unwrap();
        let mut leaf = leaf_with(l, &[key(&[1]), key(&[2])]);
        leaf.remove_key(&key(&[1])).unwrap();
        assert_eq!(keys_of(&leaf), vec![key(&[2])]);
        assert!(leaf.tombstones.is_empty());
    }

    #[test]
    fn leaf_encoding_zeroes_the_tombstone_section_past_the_count() {
        let l = layout(1);
        let leaf = leaf_with(l, &[key(&[7])]);
        let mut buf = [0xFFu8; PAGE_SIZE];
        leaf.encode(&mut buf).unwrap();
        let count_at = l.leaf_tombstone_count_offset();
        assert!(
            buf[count_at..PAGE_SIZE - l.leaf_area().capacity()]
                .iter()
                .all(|&b| b == 0)
        );
        assert!(LeafPage::decode(&buf, l).unwrap().tombstones.is_empty());
    }

    #[test]
    fn internal_decode_reports_corrupt_pages_as_errors() {
        let l = layout(2);
        let mut good = [0u8; PAGE_SIZE];
        internal_with(l, &[key(&[10, 10]), key(&[20, 20]), key(&[30, 30])])
            .encode(&mut good)
            .unwrap();

        let mut unsorted = good;
        unsorted[PAGE_SIZE - 3 * (2 + PAGE_ID_ENCODED_BYTES)] = 99;
        assert_eq!(
            InternalPage::decode(&unsorted, l).unwrap_err(),
            PageCodecError::Malformed("internal keys out of order")
        );

        let mut huge = good;
        put_u32_le(&mut huge, NODE_SIZE_OFFSET, u32::MAX);
        assert!(matches!(
            InternalPage::decode(&huge, l),
            Err(PageCodecError::Malformed(_))
        ));

        let mut into_header = good;
        into_header[INTERNAL_FIXED_HEADER_BYTES..INTERNAL_FIXED_HEADER_BYTES + 2]
            .copy_from_slice(&4u16.to_le_bytes());
        assert!(matches!(
            InternalPage::decode(&into_header, l),
            Err(PageCodecError::Malformed(_))
        ));

        let mut too_long = [0u8; PAGE_SIZE];
        internal_with(layout(5), &[key(&[1; 5])])
            .encode(&mut too_long)
            .unwrap();
        assert_eq!(
            InternalPage::decode(&too_long, l).unwrap_err(),
            PageCodecError::KeyTooLong { max: 2, actual: 5 }
        );
    }

    #[test]
    fn leaf_decode_reports_bad_tombstones_as_errors() {
        let l = layout(1);
        let mut leaf = leaf_with(l, &[key(&[1]), key(&[2]), key(&[3])]);
        leaf.remove_key(&key(&[2])).unwrap();
        let mut good = [0u8; PAGE_SIZE];
        leaf.encode(&mut good).unwrap();

        let mut past_end = good;
        put_u32_le(&mut past_end, l.leaf_tombstone_offset(0), 3);
        assert_eq!(
            LeafPage::decode(&past_end, l).unwrap_err(),
            PageCodecError::Malformed("leaf tombstones index out of bounds")
        );

        let mut over_capacity = good;
        put_u32_le(
            &mut over_capacity,
            l.leaf_tombstone_count_offset(),
            LEAF_TOMBSTONE_CAPACITY + 1,
        );
        assert_eq!(
            LeafPage::decode(&over_capacity, l).unwrap_err(),
            PageCodecError::Malformed("leaf tombstones count exceeds tombstone capacity")
        );
    }

    #[test]
    fn decoding_a_short_or_unknown_buffer_is_an_error() {
        let l = layout(1);
        assert!(matches!(
            Page::decode(&[0u8; 3], l),
            Err(PageCodecError::BufferError(
                BufferError::BufferTooSmall { .. }
            ))
        ));
        assert!(Page::kind(&[]).is_err());
        let mut header = [0u8; PAGE_SIZE];
        HeaderPage {
            root_page_id: Some(pid(1)),
            first_leaf_id: Some(pid(1)),
            max_key_len: 1,
        }
        .encode(&mut header)
        .unwrap();
        assert_eq!(
            Page::kind(&header).err(),
            Some(PageCodecError::InvalidPageKind(KIND_HEADER))
        );
        assert!(LeafPage::decode(&[0u8; 3], l).is_err());
        assert!(InternalPage::decode(&[0u8; 3], l).is_err());
    }

    #[test]
    fn encoding_a_page_that_fails_its_check_is_an_error() {
        let l = layout(1);
        let mut buf = [0u8; PAGE_SIZE];

        let mut leaf = LeafPage::new(l);
        leaf.entries = vec![(key(&[2]), rid(2)), (key(&[1]), rid(1))];
        assert_eq!(
            leaf.encode(&mut buf),
            Err(PageCodecError::Malformed("leaf keys out of order"))
        );

        let mut internal = InternalPage::new(l);
        internal.keys = vec![key(&[1])];
        assert_eq!(
            internal.encode(&mut buf),
            Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != size + 1"
            ))
        );
        assert_eq!(
            InternalPage::new(l).encode(&mut buf),
            Err(PageCodecError::Malformed("internal page has no children"))
        );

        let mut too_long = InternalPage::new(l);
        too_long.keys = vec![key(&[1, 2])];
        too_long.children = vec![pid(0), pid(1)];
        assert_eq!(
            too_long.encode(&mut buf),
            Err(PageCodecError::KeyTooLong { max: 1, actual: 2 })
        );
        let mut too_long = LeafPage::new(l);
        too_long.entries = vec![(key(&[1, 2]), rid(1))];
        assert_eq!(
            too_long.encode(&mut buf),
            Err(PageCodecError::KeyTooLong { max: 1, actual: 2 })
        );
        assert_eq!(
            leaf_with(l, &[key(&[1])]).encode(&mut [0u8; 3]),
            Err(PageCodecError::InvalidSize)
        );
    }

    #[test]
    fn page_codec_error_converts_failed_int_conversions_to_overflow() {
        let err: PageCodecError = u32::try_from(u64::MAX).unwrap_err().into();
        assert_eq!(err, PageCodecError::BufferError(BufferError::Overflow));
    }
}
