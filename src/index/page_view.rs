//! Typed, zero-copy views over B+ tree pages.
//!
//! These read and write the same on-disk layout as the owned `HeaderPage`, `InternalPage`
//! and `LeafPage` codecs in `b_plus_page`, but operate directly on the pinned frame's bytes.
//! A view over `&[u8]` (from `PageRef::data()`) is read-only; a view over `&mut [u8]` (from
//! `PageMut::data_mut()`) can also mutate in place. Borrowing the buffer ties a view's
//! lifetime to the page guard, so a view cannot outlive its latch.
//!
//! The hot paths (traversal, point lookups, scans, and inserts/removes that don't change the
//! tree's shape) go through these views. Structural modifications (splits, merges,
//! redistribution) still decode into the owned representation, mutate, and encode back.

use crate::buf::{get_u32_le, get_u64_le, put_u32_le, put_u64_le};
use crate::disk::PAGE_SIZE;
use crate::index::b_plus_page::{
    HEADER_FIXED_HEADER_BYTES, INTERNAL_FIXED_HEADER_BYTES, INVALID_PAGE_ID,
    LEAF_FIXED_HEADER_BYTES, LeafPage, NUM_TOMBSTONES, PAGE_ID_ENCODED_BYTES, PageId,
    RECORD_ID_ENCODED_BYTES, TOMBSTONE_SLOT_ENCODED_BYTES,
};
use crate::index::page_codec::{
    COMMON_HEADER_BYTES, KIND_HEADER, KIND_INTERNAL, KIND_LEAF, PageCodecError, decode_record_id,
    expect_common_header, peek_common_header, put_common_header,
};
use crate::table_heap::RecordId;
use std::cmp::Ordering;

#[inline]
fn page_id_from_raw(raw: u64) -> PageId {
    if raw == u64::MAX {
        INVALID_PAGE_ID
    } else {
        raw as PageId
    }
}

#[inline]
fn page_id_to_raw(page_id: PageId) -> u64 {
    if page_id == INVALID_PAGE_ID {
        u64::MAX
    } else {
        page_id as u64
    }
}

fn check_len(buf: &[u8]) -> Result<(), PageCodecError> {
    if buf.len() != PAGE_SIZE {
        return Err(PageCodecError::InvalidSize);
    }
    Ok(())
}

fn check_key_len(key_len: u32) -> Result<usize, PageCodecError> {
    match usize::try_from(key_len) {
        Ok(0) | Err(_) => Err(PageCodecError::Malformed("key_len must be > 0")),
        Ok(k) => Ok(k),
    }
}

/// Binary search for the first index in `0..n` where `pred` is false, for a `pred` that is
/// true on a prefix of the range (same contract as `slice::partition_point`).
#[inline]
fn partition_point(n: usize, mut pred: impl FnMut(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

const HDR_KEY_LEN_OFF: usize = COMMON_HEADER_BYTES;
const HDR_ROOT_OFF: usize = COMMON_HEADER_BYTES + 8;
const HDR_FIRST_LEAF_OFF: usize = COMMON_HEADER_BYTES + 16;

pub struct HeaderPageView<B> {
    buf: B,
}

impl<B: AsRef<[u8]>> HeaderPageView<B> {
    pub fn new(buf: B) -> Result<Self, PageCodecError> {
        let b = buf.as_ref();
        check_len(b)?;
        expect_common_header(b, KIND_HEADER)?;
        if get_u32_le(b, HDR_KEY_LEN_OFF) == 0 {
            return Err(PageCodecError::Malformed("header key_len must be > 0"));
        }
        Ok(Self { buf })
    }

    pub fn key_len(&self) -> u32 {
        get_u32_le(self.buf.as_ref(), HDR_KEY_LEN_OFF)
    }

    pub fn root_page_id(&self) -> PageId {
        page_id_from_raw(get_u64_le(self.buf.as_ref(), HDR_ROOT_OFF))
    }

    pub fn first_leaf_id(&self) -> PageId {
        page_id_from_raw(get_u64_le(self.buf.as_ref(), HDR_FIRST_LEAF_OFF))
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> HeaderPageView<B> {
    /// Formats `buf` as a header page for an empty tree.
    pub fn init(mut buf: B, key_len: u32) -> Result<Self, PageCodecError> {
        check_key_len(key_len)?;
        let b = buf.as_mut();
        check_len(b)?;
        b[..HEADER_FIXED_HEADER_BYTES].fill(0);
        put_common_header(b, KIND_HEADER);
        put_u32_le(b, HDR_KEY_LEN_OFF, key_len);
        put_u64_le(b, HDR_ROOT_OFF, page_id_to_raw(INVALID_PAGE_ID));
        put_u64_le(b, HDR_FIRST_LEAF_OFF, page_id_to_raw(INVALID_PAGE_ID));
        Ok(Self { buf })
    }

    pub fn set_root_page_id(&mut self, page_id: PageId) {
        put_u64_le(self.buf.as_mut(), HDR_ROOT_OFF, page_id_to_raw(page_id));
    }

    pub fn set_first_leaf_id(&mut self, page_id: PageId) {
        put_u64_le(
            self.buf.as_mut(),
            HDR_FIRST_LEAF_OFF,
            page_id_to_raw(page_id),
        );
    }
}

const NODE_SIZE_OFF: usize = COMMON_HEADER_BYTES;
const NODE_MAX_SIZE_OFF: usize = COMMON_HEADER_BYTES + 4;
const INTERNAL_CHILD_COUNT_OFF: usize = COMMON_HEADER_BYTES + 8;
const LEAF_NEXT_LEAF_OFF: usize = COMMON_HEADER_BYTES + 8;

/// Either kind of tree node, dispatched on the page's kind byte.
pub enum NodeView<B> {
    Internal(InternalPageView<B>),
    Leaf(LeafPageView<B>),
}

impl<B: AsRef<[u8]>> NodeView<B> {
    pub fn new(buf: B, key_len: u32) -> Result<Self, PageCodecError> {
        match peek_common_header(buf.as_ref())? {
            KIND_INTERNAL => Ok(Self::Internal(InternalPageView::new(buf, key_len)?)),
            KIND_LEAF => Ok(Self::Leaf(LeafPageView::new(buf, key_len)?)),
            k => Err(PageCodecError::InvalidPageKind(k)),
        }
    }

    pub fn is_full(&self) -> bool {
        match self {
            Self::Internal(n) => n.is_full(),
            Self::Leaf(n) => n.is_full(),
        }
    }

    pub fn would_be_underfull(&self) -> bool {
        match self {
            Self::Internal(n) => n.would_be_underfull(),
            Self::Leaf(n) => n.would_be_underfull(),
        }
    }
}

/// Read-only view over an internal page: `child_count` u64 child ids followed by
/// `size` keys of `key_len` bytes each.
pub struct InternalPageView<B> {
    buf: B,
    key_len: usize,
}

impl<B: AsRef<[u8]>> InternalPageView<B> {
    pub fn new(buf: B, key_len: u32) -> Result<Self, PageCodecError> {
        let key_len = check_key_len(key_len)?;
        let b = buf.as_ref();
        check_len(b)?;
        expect_common_header(b, KIND_INTERNAL)?;
        let size = get_u32_le(b, NODE_SIZE_OFF) as usize;
        let child_count = get_u32_le(b, INTERNAL_CHILD_COUNT_OFF) as usize;
        if child_count == 0 && size != 0 {
            return Err(PageCodecError::Malformed(
                "internal page has keys but zero children",
            ));
        }
        if child_count != 0 && child_count != size + 1 {
            return Err(PageCodecError::Malformed(
                "internal invariant violated: child_count != size + 1",
            ));
        }
        let needed = child_count
            .checked_mul(PAGE_ID_ENCODED_BYTES)
            .and_then(|c| size.checked_mul(key_len).and_then(|k| c.checked_add(k)))
            .and_then(|n| n.checked_add(INTERNAL_FIXED_HEADER_BYTES))
            .ok_or(PageCodecError::InvalidSize)?;
        if needed > PAGE_SIZE {
            return Err(PageCodecError::InvalidSize);
        }
        Ok(Self { buf, key_len })
    }

    #[inline]
    fn bytes(&self) -> &[u8] {
        self.buf.as_ref()
    }

    pub fn size(&self) -> u32 {
        get_u32_le(self.bytes(), NODE_SIZE_OFF)
    }

    pub fn max_size(&self) -> u32 {
        get_u32_le(self.bytes(), NODE_MAX_SIZE_OFF)
    }

    pub fn min_size(&self) -> u32 {
        self.max_size() / 2
    }

    pub fn child_count(&self) -> usize {
        get_u32_le(self.bytes(), INTERNAL_CHILD_COUNT_OFF) as usize
    }

    pub fn child(&self, idx: usize) -> Option<PageId> {
        if idx >= self.child_count() {
            return None;
        }
        let off = INTERNAL_FIXED_HEADER_BYTES + idx * PAGE_ID_ENCODED_BYTES;
        Some(page_id_from_raw(get_u64_le(self.bytes(), off)))
    }

    /// Raw bytes of key `idx`. Panics if `idx >= size()`.
    pub fn key(&self, idx: usize) -> &[u8] {
        assert!(
            idx < self.size() as usize,
            "internal key index out of bounds"
        );
        let keys_off = INTERNAL_FIXED_HEADER_BYTES + self.child_count() * PAGE_ID_ENCODED_BYTES;
        let off = keys_off + idx * self.key_len;
        &self.bytes()[off..off + self.key_len]
    }

    /// Same contract as `InternalPage::find_child_index`: the number of keys `<= key`.
    pub fn find_child_index(&self, key: &[u8]) -> usize {
        partition_point(self.size() as usize, |i| self.key(i) <= key)
    }

    /// The child page to descend into when looking for `key`.
    pub fn child_for(&self, key: &[u8]) -> Option<PageId> {
        self.child(self.find_child_index(key))
    }

    pub fn is_full(&self) -> bool {
        self.size() == self.max_size()
    }

    pub fn would_be_underfull(&self) -> bool {
        self.size() == self.min_size()
    }
}

/// Why an in-place leaf insert was refused. Neither case modifies the page.
#[derive(Debug, PartialEq, Eq)]
pub enum LeafInsertError {
    /// The key is already present and live.
    DuplicateKey,
    /// The leaf has no room for another live entry, so it needs a split.
    Full,
}

/// View over a leaf page: `size` entries of `{key, page_id u64, slot_id u64}`, followed by a
/// `u32` tombstone count and that many `u32` slot indexes (oldest first).
pub struct LeafPageView<B> {
    buf: B,
    key_len: usize,
}

impl<B: AsRef<[u8]>> LeafPageView<B> {
    pub fn new(buf: B, key_len: u32) -> Result<Self, PageCodecError> {
        let key_len_usize = check_key_len(key_len)?;
        let view = Self {
            buf,
            key_len: key_len_usize,
        };
        let b = view.bytes();
        check_len(b)?;
        expect_common_header(b, KIND_LEAF)?;
        let size = view.size() as usize;
        let live_end = size
            .checked_mul(view.entry_bytes())
            .and_then(|n| n.checked_add(LEAF_FIXED_HEADER_BYTES))
            .ok_or(PageCodecError::InvalidSize)?;
        if live_end > PAGE_SIZE {
            return Err(PageCodecError::InvalidSize);
        }
        if live_end + TOMBSTONE_SLOT_ENCODED_BYTES <= PAGE_SIZE {
            let count = get_u32_le(b, live_end) as usize;
            if live_end + TOMBSTONE_SLOT_ENCODED_BYTES * (count + 1) > PAGE_SIZE {
                return Err(PageCodecError::Malformed(
                    "leaf tombstones section exceeds remaining bytes",
                ));
            }
            if count > view.tombstone_capacity() {
                return Err(PageCodecError::Malformed(
                    "leaf tombstones count exceeds tombstone capacity",
                ));
            }
        }
        view.validate_entries()?;
        Ok(view)
    }

    /// A view over bytes that already passed `new` and have not changed since, such as a
    /// scan's private copy of a leaf. It skips the O(entries) checks `new` repeats.
    pub(crate) fn trusted(buf: B, key_len: u32) -> Self {
        Self {
            buf,
            key_len: key_len as usize,
        }
    }

    /// Rejects the corruptions that `LeafPage::decode` rejects, so a bad page read through a
    /// view fails with an error instead of a panic: too many entries, keys out of order,
    /// record ids that do not fit in `u32`, and tombstones naming a slot past the end.
    fn validate_entries(&self) -> Result<(), PageCodecError> {
        let size = self.size() as usize;
        if self.size() > self.max_size().saturating_add(1) {
            return Err(PageCodecError::Malformed("leaf size exceeds max size"));
        }
        for idx in 0..size {
            let off = self.entry_off(idx) + self.key_len;
            decode_record_id(
                get_u64_le(self.bytes(), off),
                get_u64_le(self.bytes(), off + 8),
            )?;
            if idx > 0 && self.key(idx - 1) > self.key(idx) {
                return Err(PageCodecError::Malformed("leaf keys out of order"));
            }
        }
        if (0..self.tombstone_count()).any(|i| self.tombstone(i) as usize >= size) {
            return Err(PageCodecError::Malformed(
                "leaf tombstones index out of bounds",
            ));
        }
        Ok(())
    }

    #[inline]
    fn bytes(&self) -> &[u8] {
        self.buf.as_ref()
    }

    #[inline]
    fn entry_bytes(&self) -> usize {
        self.key_len + RECORD_ID_ENCODED_BYTES
    }

    #[inline]
    fn entry_off(&self, idx: usize) -> usize {
        LEAF_FIXED_HEADER_BYTES + idx * self.entry_bytes()
    }

    /// Offset of the tombstone count (immediately after the live entries).
    #[inline]
    fn tomb_count_off(&self) -> usize {
        self.entry_off(self.size() as usize)
    }

    fn has_tomb_section(&self) -> bool {
        self.tomb_count_off() + TOMBSTONE_SLOT_ENCODED_BYTES <= PAGE_SIZE
    }

    /// One past the last byte the page's logical contents occupy.
    fn content_end(&self) -> usize {
        if !self.has_tomb_section() {
            return self.tomb_count_off();
        }
        self.tomb_count_off() + TOMBSTONE_SLOT_ENCODED_BYTES * (1 + self.tombstone_count())
    }

    pub fn size(&self) -> u32 {
        get_u32_le(self.bytes(), NODE_SIZE_OFF)
    }

    pub fn max_size(&self) -> u32 {
        get_u32_le(self.bytes(), NODE_MAX_SIZE_OFF)
    }

    pub fn min_size(&self) -> u32 {
        self.max_size() / 2
    }

    pub fn next_leaf_id(&self) -> PageId {
        page_id_from_raw(get_u64_le(self.bytes(), LEAF_NEXT_LEAF_OFF))
    }

    pub fn is_full(&self) -> bool {
        self.size() == self.max_size()
    }

    pub fn is_underfull(&self) -> bool {
        self.size() < self.min_size()
    }

    pub fn would_be_underfull(&self) -> bool {
        self.size() == self.min_size()
    }

    /// Raw bytes of key `idx`. Panics if `idx >= size()`.
    pub fn key(&self, idx: usize) -> &[u8] {
        assert!(idx < self.size() as usize, "leaf key index out of bounds");
        let off = self.entry_off(idx);
        &self.bytes()[off..off + self.key_len]
    }

    /// Record id of entry `idx`. Panics if `idx >= size()`. `new` has already checked that
    /// every stored id fits in `u32`, so the conversion below cannot truncate.
    pub fn record_id(&self, idx: usize) -> RecordId {
        assert!(idx < self.size() as usize, "leaf entry index out of bounds");
        let off = self.entry_off(idx) + self.key_len;
        let page_id = get_u64_le(self.bytes(), off);
        let slot_id = get_u64_le(self.bytes(), off + 8);
        decode_record_id(page_id, slot_id).expect("record ids were validated in LeafPageView::new")
    }

    pub fn tombstone_capacity(&self) -> usize {
        LeafPage::tombstone_capacity_for_layout(
            self.max_size(),
            self.key_len as u32,
            NUM_TOMBSTONES,
        ) as usize
    }

    pub fn tombstone_count(&self) -> usize {
        if !self.has_tomb_section() {
            return 0;
        }
        get_u32_le(self.bytes(), self.tomb_count_off()) as usize
    }

    /// The `i`-th tombstone (oldest first).
    pub fn tombstone(&self, i: usize) -> u32 {
        assert!(i < self.tombstone_count(), "tombstone index out of bounds");
        let off = self.tomb_count_off() + TOMBSTONE_SLOT_ENCODED_BYTES * (1 + i);
        get_u32_le(self.bytes(), off)
    }

    pub fn is_tombstoned(&self, slot: u32) -> bool {
        (0..self.tombstone_count()).any(|i| self.tombstone(i) == slot)
    }

    /// Like `slice::binary_search` over the keys.
    pub fn search(&self, key: &[u8]) -> Result<usize, usize> {
        let n = self.size() as usize;
        let pos = partition_point(n, |i| self.key(i) < key);
        if pos < n && self.key(pos).cmp(key) == Ordering::Equal {
            Ok(pos)
        } else {
            Err(pos)
        }
    }

    pub fn lower_bound(&self, key: &[u8]) -> usize {
        partition_point(self.size() as usize, |i| self.key(i) < key)
    }

    pub fn upper_bound(&self, key: &[u8]) -> usize {
        partition_point(self.size() as usize, |i| self.key(i) <= key)
    }

    pub fn key_exists(&self, key: &[u8]) -> bool {
        self.search(key).is_ok()
    }

    /// Point lookup; tombstoned keys are reported as absent.
    pub fn get(&self, key: &[u8]) -> Option<RecordId> {
        let pos = self.search(key).ok()?;
        if self.is_tombstoned(pos as u32) {
            return None;
        }
        Some(self.record_id(pos))
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> LeafPageView<B> {
    #[inline]
    fn bytes_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut()
    }

    fn set_size(&mut self, size: u32) {
        put_u32_le(self.bytes_mut(), NODE_SIZE_OFF, size);
    }

    fn set_tombstone_count(&mut self, count: usize) {
        let off = self.tomb_count_off();
        put_u32_le(self.bytes_mut(), off, count as u32);
    }

    fn set_tombstone(&mut self, i: usize, slot: u32) {
        let off = self.tomb_count_off() + TOMBSTONE_SLOT_ENCODED_BYTES * (1 + i);
        put_u32_le(self.bytes_mut(), off, slot);
    }

    fn write_entry(&mut self, idx: usize, key: &[u8], rid: RecordId) {
        let key_len = self.key_len;
        let off = self.entry_off(idx);
        let b = self.bytes_mut();
        b[off..off + key_len].copy_from_slice(key);
        put_u64_le(b, off + key_len, u64::from(rid.page_id));
        put_u64_le(b, off + key_len + 8, u64::from(rid.slot_id));
    }

    /// Rewrites the tombstone list in place, keeping only slots for which `f` returns
    /// `Some`, and zeroes any bytes freed at the end of the section.
    fn rewrite_tombstones(&mut self, mut f: impl FnMut(u32) -> Option<u32>) {
        let count = self.tombstone_count();
        let mut kept = 0;
        for i in 0..count {
            if let Some(slot) = f(self.tombstone(i)) {
                self.set_tombstone(kept, slot);
                kept += 1;
            }
        }
        let end = self.content_end();
        self.set_tombstone_count(kept);
        let new_end = self.content_end();
        self.bytes_mut()[new_end..end].fill(0);
    }

    /// Inserts `key -> rid` in sorted position, shifting later entries (and the trailing
    /// tombstone section) right by one entry with a single memmove. If `key` exists but is
    /// tombstoned, it is revived with the new record id instead, which takes no extra space.
    pub fn insert(&mut self, key: &[u8], rid: RecordId) -> Result<(), LeafInsertError> {
        assert_eq!(key.len(), self.key_len, "leaf insert key length mismatch");
        match self.search(key) {
            Ok(pos) => {
                let slot = pos as u32;
                if !self.is_tombstoned(slot) {
                    return Err(LeafInsertError::DuplicateKey);
                }
                self.write_entry(pos, key, rid);
                self.rewrite_tombstones(|s| (s != slot).then_some(s));
                Ok(())
            }
            Err(pos) => {
                let entry = self.entry_bytes();
                let end = self.content_end();
                if self.size() >= self.max_size()
                    || !self.has_tomb_section()
                    || end + entry > PAGE_SIZE
                {
                    return Err(LeafInsertError::Full);
                }
                let off = self.entry_off(pos);
                self.bytes_mut().copy_within(off..end, off + entry);
                let size = self.size();
                self.set_size(size + 1);
                self.write_entry(pos, key, rid);
                let insert_at = pos as u32;
                self.rewrite_tombstones(|s| Some(if s >= insert_at { s + 1 } else { s }));
                Ok(())
            }
        }
    }

    /// Physically removes entry `slot`, shifting later entries and the tombstone section left,
    /// dropping any tombstone for `slot` and renumbering later ones.
    fn physical_remove_at(&mut self, slot: u32) {
        let entry = self.entry_bytes();
        let idx = slot as usize;
        let end = self.content_end();
        let off = self.entry_off(idx);
        let b = self.bytes_mut();
        b.copy_within(off + entry..end, off);
        b[end - entry..end].fill(0);
        let size = self.size();
        self.set_size(size - 1);
        self.rewrite_tombstones(|s| match s.cmp(&slot) {
            Ordering::Equal => None,
            Ordering::Greater => Some(s - 1),
            Ordering::Less => Some(s),
        });
    }

    /// Same semantics as `LeafPage::remove_key`: tombstones `key`, physically removing the
    /// oldest tombstoned entry first if the tombstone FIFO is full (or removing `key`
    /// immediately if the page has no tombstone capacity). The page is untouched on error.
    pub fn remove_key(&mut self, key: &[u8]) -> Result<(), crate::index::RemoveError> {
        use crate::index::RemoveError;
        let pos = self.search(key).map_err(|_| RemoveError::KeyNotFound)?;
        let mut slot = pos as u32;
        if self.is_tombstoned(slot) {
            return Err(RemoveError::KeyNotFound);
        }

        let capacity = self.tombstone_capacity();
        if capacity == 0 || !self.has_tomb_section() {
            self.physical_remove_at(slot);
            return Ok(());
        }

        if self.tombstone_count() >= capacity {
            let evicted = self.tombstone(0);
            let mut first = true;
            self.rewrite_tombstones(|s| {
                if first {
                    first = false;
                    None
                } else {
                    Some(s)
                }
            });
            self.physical_remove_at(evicted);
            if evicted < slot {
                slot -= 1;
            }
        }

        let count = self.tombstone_count();
        let tombstone_fits_on_page = self.content_end() + TOMBSTONE_SLOT_ENCODED_BYTES <= PAGE_SIZE;
        if !tombstone_fits_on_page {
            self.physical_remove_at(slot);
            return Ok(());
        }
        self.set_tombstone_count(count + 1);
        self.set_tombstone(count, slot);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::index_schema::IndexKey;
    use crate::index::b_plus_page::{HeaderPage, InternalPage};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rid(n: u32) -> RecordId {
        RecordId {
            page_id: n,
            slot_id: n.wrapping_mul(7),
        }
    }

    fn encode_leaf(leaf: &LeafPage, key_len: u32) -> Box<[u8; PAGE_SIZE]> {
        let mut buf = Box::new([0u8; PAGE_SIZE]);
        leaf.encode(&mut buf[..], key_len).unwrap();
        buf
    }

    #[test]
    fn header_view_matches_codec() {
        let mut buf = [0xffu8; PAGE_SIZE];
        let mut view = HeaderPageView::init(&mut buf[..], 8).unwrap();
        view.set_root_page_id(42);
        let decoded = HeaderPage::decode(&buf).unwrap();
        assert_eq!(decoded.root_page_id, 42);
        assert_eq!(decoded.first_leaf_id, INVALID_PAGE_ID);
        assert_eq!(decoded.key_len, 8);

        let mut expected = [0u8; PAGE_SIZE];
        HeaderPage {
            root_page_id: 42,
            first_leaf_id: INVALID_PAGE_ID,
            key_len: 8,
        }
        .encode(&mut expected)
        .unwrap();
        assert_eq!(
            buf[..HEADER_FIXED_HEADER_BYTES],
            expected[..HEADER_FIXED_HEADER_BYTES]
        );

        let view = HeaderPageView::new(&expected[..]).unwrap();
        assert_eq!(view.root_page_id(), 42);
        assert_eq!(view.first_leaf_id(), INVALID_PAGE_ID);
        assert_eq!(view.key_len(), 8);
    }

    #[test]
    fn header_view_rejects_wrong_kind_and_bad_magic() {
        let mut buf = [0u8; PAGE_SIZE];
        LeafPage::new(4, 0).encode(&mut buf, 1).unwrap();
        assert!(matches!(
            HeaderPageView::new(&buf[..]),
            Err(PageCodecError::WrongPageKind { .. })
        ));
        buf[0] ^= 0xff;
        assert!(matches!(
            HeaderPageView::new(&buf[..]),
            Err(PageCodecError::BadMagic)
        ));
    }

    #[test]
    fn internal_view_matches_owned_page() {
        let key_len = 2u32;
        let mut internal = InternalPage::new(10);
        internal.children = vec![11, 22, 33, INVALID_PAGE_ID];
        internal.keys = vec![
            IndexKey(vec![1, 2]),
            IndexKey(vec![3, 4]),
            IndexKey(vec![3, 9]),
        ];
        internal.insert_key(0, IndexKey(vec![0, 0]));
        internal.remove_key(0);
        let mut buf = [0u8; PAGE_SIZE];
        internal.encode(&mut buf, key_len).unwrap();

        let view = InternalPageView::new(&buf[..], key_len).unwrap();
        assert_eq!(view.size(), 3);
        assert_eq!(view.max_size(), 10);
        assert_eq!(view.child_count(), 4);
        for (i, c) in internal.children.iter().enumerate() {
            assert_eq!(view.child(i), Some(*c));
        }
        assert_eq!(view.child(4), None);
        for (i, k) in internal.keys.iter().enumerate() {
            assert_eq!(view.key(i), &k.0[..]);
        }
        for probe in [[0u8, 0], [1, 2], [2, 0], [3, 4], [3, 5], [3, 9], [9, 9]] {
            let key = IndexKey(probe.to_vec());
            assert_eq!(
                view.find_child_index(&probe),
                internal.find_child_index(&key),
                "probe {probe:?}"
            );
        }
        assert!(matches!(
            NodeView::new(&buf[..], key_len),
            Ok(NodeView::Internal(_))
        ));
        assert!(matches!(
            LeafPageView::new(&buf[..], key_len),
            Err(PageCodecError::WrongPageKind { .. })
        ));
    }

    #[test]
    fn leaf_view_reads_match_owned_page() {
        let key_len = 1u32;
        let mut leaf = LeafPage::new(16, NUM_TOMBSTONES);
        for k in [10u8, 20, 30, 40, 50] {
            leaf.insert(IndexKey(vec![k]), rid(k as u32)).unwrap();
        }
        leaf.next_leaf_id = 77;
        leaf.remove_key(&IndexKey(vec![30])).unwrap();
        let buf = encode_leaf(&leaf, key_len);

        let view = LeafPageView::new(&buf[..], key_len).unwrap();
        assert_eq!(view.size(), 5);
        assert_eq!(view.next_leaf_id(), 77);
        assert_eq!(view.tombstone_count(), 1);
        assert!(view.is_tombstoned(2));
        for probe in 0u8..60 {
            let key = IndexKey(vec![probe]);
            assert_eq!(view.get(&[probe]), leaf.get(&key), "get {probe}");
            assert_eq!(view.lower_bound(&[probe]), leaf.lower_bound(&key));
            assert_eq!(view.upper_bound(&[probe]), leaf.upper_bound(&key));
            assert_eq!(view.search(&[probe]), leaf.keys.binary_search(&key));
        }
    }

    /// Each corruption is rejected by both the view and the owned decoder, with an error
    /// rather than a panic.
    #[test]
    fn corrupt_leaf_pages_are_errors_not_panics() {
        let key_len = 1u32;
        let mut leaf = LeafPage::new(16, NUM_TOMBSTONES);
        for k in [10u8, 20, 30] {
            leaf.insert(IndexKey(vec![k]), rid(u32::from(k))).unwrap();
        }
        leaf.remove_key(&IndexKey(vec![20])).unwrap();
        let good = encode_leaf(&leaf, key_len);
        let entry_bytes = key_len as usize + RECORD_ID_ENCODED_BYTES;
        let tomb_count_off = LEAF_FIXED_HEADER_BYTES + 3 * entry_bytes;
        let first_rid_off = LEAF_FIXED_HEADER_BYTES + key_len as usize;

        type Corruption = Box<dyn Fn(&mut [u8])>;
        let corruptions: [(&str, Corruption); 5] = [
            (
                "tombstone past the last slot",
                Box::new(move |b| put_u32_le(b, tomb_count_off + TOMBSTONE_SLOT_ENCODED_BYTES, 3)),
            ),
            (
                "page id wider than u32",
                Box::new(move |b| put_u64_le(b, first_rid_off, u64::from(u32::MAX) + 1)),
            ),
            (
                "slot id wider than u32",
                Box::new(move |b| put_u64_le(b, first_rid_off + 8, u64::from(u32::MAX) + 1)),
            ),
            (
                "keys out of order",
                Box::new(|b| b[LEAF_FIXED_HEADER_BYTES] = 99),
            ),
            (
                "size above max size",
                Box::new(|b| put_u32_le(b, NODE_MAX_SIZE_OFF, 1)),
            ),
        ];
        for (what, corrupt) in corruptions {
            let mut buf = good.clone();
            corrupt(&mut buf[..]);
            assert!(
                matches!(
                    LeafPageView::new(&buf[..], key_len),
                    Err(PageCodecError::Malformed(_) | PageCodecError::BufferError(_))
                ),
                "view rejects {what}"
            );
            assert!(
                LeafPage::decode(&buf[..], key_len).is_err(),
                "decode rejects {what}"
            );
        }
    }

    /// Runs a random mix of inserts and removes against both the owned `LeafPage` and the
    /// in-place `LeafPageView`, checking after every step that the view's bytes are identical
    /// to encoding the owned page.
    fn differential(key_len: u32, max_size: u32, tombstones: u32, seed: u64, ops: usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut owned = LeafPage::new(max_size, tombstones);
        owned.next_leaf_id = 5;
        let mut buf = encode_leaf(&owned, key_len);
        let owned_with_persisted_tombstone_capacity = LeafPage::decode(&buf[..], key_len).unwrap();
        owned = owned_with_persisted_tombstone_capacity;

        for step in 0..ops {
            let k: Vec<u8> = (0..key_len).map(|_| rng.random_range(0..12u8)).collect();
            let key = IndexKey(k.clone());
            let r = rid(rng.random_range(0..1000));
            if rng.random_bool(0.6) {
                let mut view = LeafPageView::new(&mut buf[..], key_len).unwrap();
                match view.insert(&k, r) {
                    Ok(()) => owned.insert(key, r).unwrap(),
                    Err(LeafInsertError::DuplicateKey) => {
                        assert!(owned.get(&key).is_some(), "step {step}: false duplicate")
                    }
                    Err(LeafInsertError::Full) => {
                        assert!(owned.is_full(), "step {step}: false full")
                    }
                }
            } else {
                let mut view = LeafPageView::new(&mut buf[..], key_len).unwrap();
                let got = view.remove_key(&k).is_ok();
                let want = owned.remove_key(&key).is_ok();
                assert_eq!(got, want, "step {step}: remove result");
            }
            let expected = encode_leaf(&owned, key_len);
            assert!(
                buf[..] == expected[..],
                "step {step}: view bytes diverged from owned encoding"
            );
            LeafPageView::new(&buf[..], key_len).unwrap();
        }
    }

    #[test]
    fn leaf_view_mutations_match_owned_codec_with_tombstones() {
        for seed in 0..20 {
            differential(1, 12, NUM_TOMBSTONES, seed, 300);
        }
    }

    #[test]
    fn leaf_view_mutations_match_owned_codec_with_small_tombstone_fifo() {
        let key_len = 1u32;
        let max_size = 480;
        assert_eq!(
            LeafPage::tombstone_capacity_for_layout(max_size, key_len, NUM_TOMBSTONES),
            1,
            "480 live entries leave room for the tombstone count and a single tombstone, so every second delete is physical"
        );
        for seed in 0..10 {
            differential(key_len, max_size, NUM_TOMBSTONES, seed, 300);
        }
    }

    #[test]
    fn leaf_view_mutations_match_owned_codec_with_wider_keys() {
        for seed in 0..10 {
            differential(3, 40, NUM_TOMBSTONES, seed, 400);
        }
    }

    #[test]
    fn leaf_view_insert_reports_full_without_modifying() {
        let key_len = 1u32;
        let mut leaf = LeafPage::new(3, NUM_TOMBSTONES);
        for k in [1u8, 2, 3] {
            leaf.insert(IndexKey(vec![k]), rid(k as u32)).unwrap();
        }
        let mut buf = encode_leaf(&leaf, key_len);
        let before = buf.clone();
        let mut view = LeafPageView::new(&mut buf[..], key_len).unwrap();
        assert_eq!(view.insert(&[9], rid(9)), Err(LeafInsertError::Full));
        assert_eq!(
            view.insert(&[2], rid(9)),
            Err(LeafInsertError::DuplicateKey)
        );
        assert!(buf[..] == before[..]);

        let mut view = LeafPageView::new(&mut buf[..], key_len).unwrap();
        view.remove_key(&[2]).unwrap();
        view.insert(&[2], rid(42)).unwrap();
        assert_eq!(view.get(&[2]), Some(rid(42)));
        assert_eq!(view.tombstone_count(), 0);
    }

    #[test]
    fn leaf_view_rejects_tombstone_count_over_capacity() {
        let key_len = 1u32;
        let leaf = LeafPage::new(4, NUM_TOMBSTONES);
        let mut buf = encode_leaf(&leaf, key_len);
        let off = LEAF_FIXED_HEADER_BYTES;
        put_u32_le(&mut buf[..], off, u32::MAX);
        assert!(LeafPageView::new(&buf[..], key_len).is_err());
    }
}
