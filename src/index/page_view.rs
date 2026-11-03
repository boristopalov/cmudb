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
use crate::ids::RecordId;
use crate::index::b_plus_page::{
    HEADER_FIXED_HEADER_BYTES, Layout, NODE_PAGE_ID_OFFSET, NODE_SIZE_OFFSET, PageId,
    decode_record_id_bytes, encode_record_id_bytes,
};
use crate::index::page_codec::{
    COMMON_HEADER_BYTES, KIND_HEADER, KIND_INTERNAL, KIND_LEAF, PageCodecError,
    decode_child_page_id, decode_page_id, encode_page_id, expect_common_header, peek_common_header,
    put_common_header,
};
use crate::index::slotted::EntryArea;
use std::cmp::Ordering;

fn check_len(buf: &[u8]) -> Result<(), PageCodecError> {
    if buf.len() != PAGE_SIZE {
        return Err(PageCodecError::InvalidSize);
    }
    Ok(())
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

    /// The longest key the tree holds.
    pub fn max_key_len(&self) -> u32 {
        get_u32_le(self.buf.as_ref(), HDR_KEY_LEN_OFF)
    }

    /// The tree's root, or `None` for an empty tree.
    pub fn root_page_id(&self) -> Result<Option<PageId>, PageCodecError> {
        decode_page_id(get_u64_le(self.buf.as_ref(), HDR_ROOT_OFF))
    }

    /// The tree's leftmost leaf, or `None` for an empty tree.
    pub fn first_leaf_id(&self) -> Result<Option<PageId>, PageCodecError> {
        decode_page_id(get_u64_le(self.buf.as_ref(), HDR_FIRST_LEAF_OFF))
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> HeaderPageView<B> {
    /// Formats `buf` as a header page for an empty tree of keys up to `max_key_len` bytes.
    pub fn init(mut buf: B, max_key_len: u32) -> Result<Self, PageCodecError> {
        Layout::new(max_key_len)?;
        let b = buf.as_mut();
        check_len(b)?;
        b[..HEADER_FIXED_HEADER_BYTES].fill(0);
        put_common_header(b, KIND_HEADER);
        put_u32_le(b, HDR_KEY_LEN_OFF, max_key_len);
        put_u64_le(b, HDR_ROOT_OFF, encode_page_id(None));
        put_u64_le(b, HDR_FIRST_LEAF_OFF, encode_page_id(None));
        Ok(Self { buf })
    }

    pub fn set_root_page_id(&mut self, page_id: Option<PageId>) {
        put_u64_le(self.buf.as_mut(), HDR_ROOT_OFF, encode_page_id(page_id));
    }

    pub fn set_first_leaf_id(&mut self, page_id: Option<PageId>) {
        put_u64_le(
            self.buf.as_mut(),
            HDR_FIRST_LEAF_OFF,
            encode_page_id(page_id),
        );
    }
}

/// The entry count of a node page, read before the page's entry area is checked.
fn node_size(buf: &[u8]) -> usize {
    get_u32_le(buf, NODE_SIZE_OFFSET) as usize
}

/// Either kind of tree node, dispatched on the page's kind byte.
pub enum NodeView<B> {
    Internal(InternalPageView<B>),
    Leaf(LeafPageView<B>),
}

impl<B: AsRef<[u8]>> NodeView<B> {
    pub fn new(buf: B, layout: Layout) -> Result<Self, PageCodecError> {
        match peek_common_header(buf.as_ref())? {
            KIND_INTERNAL => Ok(Self::Internal(InternalPageView::new(buf, layout)?)),
            KIND_LEAF => Ok(Self::Leaf(LeafPageView::new(buf, layout)?)),
            k => Err(PageCodecError::InvalidPageKind(k)),
        }
    }

    /// Whether one more entry of the longest key might not fit, so an insert below may
    /// split this page.
    pub fn may_split_on_insert(&self) -> bool {
        match self {
            Self::Internal(n) => n.may_split_on_insert(),
            Self::Leaf(n) => n.may_split_on_insert(),
        }
    }

    /// Whether losing one entry of the longest key might leave this page underfull, so a
    /// remove below may rebalance it.
    pub fn may_underflow_on_remove(&self) -> bool {
        match self {
            Self::Internal(n) => n.may_underflow_on_remove(),
            Self::Leaf(n) => n.may_underflow_on_remove(),
        }
    }
}

/// Read-only view over an internal page: the leftmost child id in the header, then `size`
/// entries of `{key, child u64}`, each child to the right of its key.
pub struct InternalPageView<B> {
    buf: B,
    layout: Layout,
}

impl<B: AsRef<[u8]>> InternalPageView<B> {
    pub fn new(buf: B, layout: Layout) -> Result<Self, PageCodecError> {
        let b = buf.as_ref();
        check_len(b)?;
        expect_common_header(b, KIND_INTERNAL)?;
        layout.internal_area().check(b, node_size(b))?;
        Ok(Self { buf, layout })
    }

    #[inline]
    fn bytes(&self) -> &[u8] {
        self.buf.as_ref()
    }

    #[inline]
    fn area(&self) -> EntryArea {
        self.layout.internal_area()
    }

    pub fn size(&self) -> u32 {
        get_u32_le(self.bytes(), NODE_SIZE_OFFSET)
    }

    fn len(&self) -> usize {
        self.size() as usize
    }

    pub fn child_count(&self) -> usize {
        self.len() + 1
    }

    /// Child `idx`, or `None` past the last child. Fails like `InternalPage::decode` for a
    /// child slot that names no page or a page id out of range.
    pub fn child(&self, idx: usize) -> Result<Option<PageId>, PageCodecError> {
        if idx >= self.child_count() {
            return Ok(None);
        }
        let raw = if idx == 0 {
            get_u64_le(self.bytes(), NODE_PAGE_ID_OFFSET)
        } else {
            let payload = self.area().payload(self.bytes(), self.len(), idx - 1);
            u64::from_le_bytes(payload.try_into().expect("a child id is eight bytes"))
        };
        decode_child_page_id(raw).map(Some)
    }

    /// Raw bytes of key `idx`. Panics if `idx >= size()`.
    pub fn key(&self, idx: usize) -> &[u8] {
        self.area().key(self.bytes(), self.len(), idx)
    }

    /// Bytes the keys and their children take.
    pub fn used_bytes(&self) -> usize {
        self.area().used(self.bytes(), self.len())
    }

    /// Same contract as `InternalPage::find_child_index`: the number of keys `<= key`.
    pub fn find_child_index(&self, key: &[u8]) -> usize {
        partition_point(self.len(), |i| self.key(i) <= key)
    }

    /// The child page to descend into when looking for `key`.
    pub fn child_for(&self, key: &[u8]) -> Result<Option<PageId>, PageCodecError> {
        self.child(self.find_child_index(key))
    }

    pub fn may_split_on_insert(&self) -> bool {
        self.area().free(self.bytes(), self.len()) < self.layout.internal_max_entry_bytes()
    }

    pub fn may_underflow_on_remove(&self) -> bool {
        self.used_bytes()
            .saturating_sub(self.layout.internal_max_entry_bytes())
            < self.layout.internal_min_used()
    }
}

/// Why an in-place leaf insert was refused. Neither case modifies the page.
#[derive(Debug, PartialEq, Eq)]
pub enum LeafInsertError {
    /// The key is already present and live.
    DuplicateKey,
    /// The leaf has no room for the entry, so it needs a split.
    Full,
}

/// View over a leaf page: the next leaf id in the header, a `u32` tombstone count and the
/// layout's fixed number of `u32` tombstone slots (oldest first), then `size` entries of
/// `{key, page_id u64, slot_id u64}`.
pub struct LeafPageView<B> {
    buf: B,
    layout: Layout,
}

impl<B: AsRef<[u8]>> LeafPageView<B> {
    /// Fails on every corruption that would make a read through the view panic: an entry area
    /// whose slots or entries leave the page, too many tombstones, or a tombstone naming a slot
    /// past the end. Keys out of order give wrong answers rather than a panic and are caught by
    /// `LeafPage::decode`; a record id wider than `u32` is reported by `record_id`.
    pub fn new(buf: B, layout: Layout) -> Result<Self, PageCodecError> {
        let view = Self { buf, layout };
        let b = view.bytes();
        check_len(b)?;
        expect_common_header(b, KIND_LEAF)?;
        let size = node_size(b);
        layout.leaf_area().check(b, size)?;
        let count = get_u32_le(b, layout.leaf_tombstone_count_offset()) as usize;
        if count > layout.tombstone_capacity() {
            return Err(PageCodecError::Malformed(
                "leaf tombstones count exceeds tombstone capacity",
            ));
        }
        if (0..count).any(|i| view.tombstone(i) as usize >= size) {
            return Err(PageCodecError::Malformed(
                "leaf tombstones index out of bounds",
            ));
        }
        Ok(view)
    }

    #[inline]
    fn bytes(&self) -> &[u8] {
        self.buf.as_ref()
    }

    #[inline]
    fn area(&self) -> EntryArea {
        self.layout.leaf_area()
    }

    fn len(&self) -> usize {
        self.size() as usize
    }

    pub fn size(&self) -> u32 {
        get_u32_le(self.bytes(), NODE_SIZE_OFFSET)
    }

    pub fn next_leaf_id(&self) -> Result<Option<PageId>, PageCodecError> {
        decode_page_id(get_u64_le(self.bytes(), NODE_PAGE_ID_OFFSET))
    }

    /// Bytes the entries take, tombstoned ones included.
    pub fn used_bytes(&self) -> usize {
        self.area().used(self.bytes(), self.len())
    }

    pub fn may_split_on_insert(&self) -> bool {
        self.area().free(self.bytes(), self.len()) < self.layout.leaf_max_entry_bytes()
    }

    pub fn may_underflow_on_remove(&self) -> bool {
        self.used_bytes()
            .saturating_sub(self.layout.leaf_max_entry_bytes())
            < self.layout.leaf_min_used()
    }

    /// Raw bytes of key `idx`. Panics if `idx >= size()`.
    pub fn key(&self, idx: usize) -> &[u8] {
        self.area().key(self.bytes(), self.len(), idx)
    }

    /// Record id of entry `idx`, or an error if the stored id does not fit in `u32`. Panics if
    /// `idx >= size()`.
    pub fn record_id(&self, idx: usize) -> Result<RecordId, PageCodecError> {
        decode_record_id_bytes(self.area().payload(self.bytes(), self.len(), idx))
    }

    pub fn tombstone_capacity(&self) -> usize {
        self.layout.tombstone_capacity()
    }

    pub fn tombstone_count(&self) -> usize {
        get_u32_le(self.bytes(), self.layout.leaf_tombstone_count_offset()) as usize
    }

    /// The `i`-th tombstone (oldest first).
    pub fn tombstone(&self, i: usize) -> u32 {
        assert!(
            i < self.tombstone_capacity(),
            "tombstone index out of bounds"
        );
        get_u32_le(self.bytes(), self.layout.leaf_tombstone_offset(i))
    }

    pub fn is_tombstoned(&self, slot: u32) -> bool {
        (0..self.tombstone_count()).any(|i| self.tombstone(i) == slot)
    }

    /// Like `slice::binary_search` over the keys.
    pub fn search(&self, key: &[u8]) -> Result<usize, usize> {
        let n = self.len();
        let pos = partition_point(n, |i| self.key(i) < key);
        if pos < n && self.key(pos).cmp(key) == Ordering::Equal {
            Ok(pos)
        } else {
            Err(pos)
        }
    }

    pub fn lower_bound(&self, key: &[u8]) -> usize {
        partition_point(self.len(), |i| self.key(i) < key)
    }

    pub fn upper_bound(&self, key: &[u8]) -> usize {
        partition_point(self.len(), |i| self.key(i) <= key)
    }

    pub fn key_exists(&self, key: &[u8]) -> bool {
        self.search(key).is_ok()
    }

    /// Point lookup; tombstoned keys are reported as absent.
    pub fn get(&self, key: &[u8]) -> Result<Option<RecordId>, PageCodecError> {
        let Ok(pos) = self.search(key) else {
            return Ok(None);
        };
        if self.is_tombstoned(pos as u32) {
            return Ok(None);
        }
        self.record_id(pos).map(Some)
    }
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> LeafPageView<B> {
    #[inline]
    fn bytes_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut()
    }

    fn set_size(&mut self, size: usize) {
        put_u32_le(self.bytes_mut(), NODE_SIZE_OFFSET, size as u32);
    }

    fn set_tombstone_count(&mut self, count: usize) {
        let off = self.layout.leaf_tombstone_count_offset();
        put_u32_le(self.bytes_mut(), off, count as u32);
    }

    fn set_tombstone(&mut self, i: usize, slot: u32) {
        let off = self.layout.leaf_tombstone_offset(i);
        put_u32_le(self.bytes_mut(), off, slot);
    }

    /// Rewrites the tombstone list in place, keeping only slots for which `f` returns
    /// `Some`, and zeroes the slots it frees.
    fn rewrite_tombstones(&mut self, mut f: impl FnMut(u32) -> Option<u32>) {
        let count = self.tombstone_count();
        let mut kept = 0;
        for i in 0..count {
            if let Some(slot) = f(self.tombstone(i)) {
                self.set_tombstone(kept, slot);
                kept += 1;
            }
        }
        for i in kept..count {
            self.set_tombstone(i, 0);
        }
        self.set_tombstone_count(kept);
    }

    /// Inserts `key -> rid` in sorted position. If `key` exists but is tombstoned, it is
    /// revived with the new record id instead, which takes no extra space.
    pub fn insert(&mut self, key: &[u8], rid: RecordId) -> Result<(), LeafInsertError> {
        let payload = encode_record_id_bytes(&rid).expect("a u32 record id encodes");
        match self.search(key) {
            Ok(pos) => {
                let slot = pos as u32;
                if !self.is_tombstoned(slot) {
                    return Err(LeafInsertError::DuplicateKey);
                }
                let (area, n) = (self.area(), self.len());
                area.set_payload(self.bytes_mut(), n, pos, &payload);
                self.rewrite_tombstones(|s| (s != slot).then_some(s));
                Ok(())
            }
            Err(pos) => {
                let (area, n) = (self.area(), self.len());
                if area.free(self.bytes(), n) < area.entry_cost(key.len()) {
                    return Err(LeafInsertError::Full);
                }
                area.insert(self.bytes_mut(), n, pos, key, &payload);
                self.set_size(n + 1);
                let insert_at = pos as u32;
                self.rewrite_tombstones(|s| Some(if s >= insert_at { s + 1 } else { s }));
                Ok(())
            }
        }
    }

    /// Physically removes entry `slot`, dropping any tombstone for it and renumbering later
    /// ones.
    fn physical_remove_at(&mut self, slot: u32) {
        let (area, n) = (self.area(), self.len());
        area.remove(self.bytes_mut(), n, slot as usize);
        self.set_size(n - 1);
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
        if capacity == 0 {
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
        self.set_tombstone_count(count + 1);
        self.set_tombstone(count, slot);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::index_schema::IndexKey;
    use crate::index::b_plus_page::{HeaderPage, InternalPage, LeafPage, MAX_KEY_LEN};
    use crate::testing::pid;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rid(n: u32) -> RecordId {
        RecordId {
            page_id: pid(n),
            slot_id: n.wrapping_mul(7),
        }
    }

    fn layout(max_key_len: u32) -> Layout {
        Layout::new(max_key_len).unwrap()
    }

    fn encode_leaf(leaf: &LeafPage) -> Box<[u8; PAGE_SIZE]> {
        let mut buf = Box::new([0u8; PAGE_SIZE]);
        leaf.encode(&mut buf[..]).unwrap();
        buf
    }

    fn key(bytes: &[u8]) -> IndexKey {
        IndexKey(bytes.to_vec())
    }

    #[test]
    fn header_view_matches_codec() {
        let mut buf = [0xffu8; PAGE_SIZE];
        let mut view = HeaderPageView::init(&mut buf[..], 8).unwrap();
        view.set_root_page_id(Some(pid(42)));
        let decoded = HeaderPage::decode(&buf).unwrap();
        assert_eq!(decoded.root_page_id, Some(pid(42)));
        assert_eq!(decoded.first_leaf_id, None);
        assert_eq!(decoded.max_key_len, 8);

        let mut expected = [0u8; PAGE_SIZE];
        HeaderPage {
            root_page_id: Some(pid(42)),
            first_leaf_id: None,
            max_key_len: 8,
        }
        .encode(&mut expected)
        .unwrap();
        assert_eq!(
            buf[..HEADER_FIXED_HEADER_BYTES],
            expected[..HEADER_FIXED_HEADER_BYTES]
        );

        let view = HeaderPageView::new(&expected[..]).unwrap();
        assert_eq!(view.root_page_id(), Ok(Some(pid(42))));
        assert_eq!(view.first_leaf_id(), Ok(None));
        assert_eq!(view.max_key_len(), 8);
    }

    #[test]
    fn header_view_rejects_wrong_kind_bad_magic_and_unsupported_key_lengths() {
        let mut buf = [0u8; PAGE_SIZE];
        LeafPage::new(layout(1)).encode(&mut buf).unwrap();
        assert!(matches!(
            HeaderPageView::new(&buf[..]),
            Err(PageCodecError::WrongPageKind { .. })
        ));
        buf[0] ^= 0xff;
        assert!(matches!(
            HeaderPageView::new(&buf[..]),
            Err(PageCodecError::BadMagic)
        ));
        let too_long = u32::try_from(MAX_KEY_LEN + 1).unwrap();
        assert!(HeaderPageView::init(&mut buf[..], too_long).is_err());
        assert!(HeaderPageView::init(&mut buf[..], 0).is_err());
    }

    #[test]
    fn internal_view_matches_owned_page() {
        let l = layout(8);
        let mut internal = InternalPage::new_root(l, pid(11), key(&[1, 2]), pid(22));
        internal.append_key_and_child(key(&[3, 4, 5, 6, 7]), pid(33));
        internal.append_key_and_child(key(&[3, 9]), pid(0));
        internal.append_key_and_child(key(&[4]), pid(44));
        let mut buf = [0u8; PAGE_SIZE];
        internal.encode(&mut buf).unwrap();

        let view = InternalPageView::new(&buf[..], l).unwrap();
        assert_eq!(view.size(), 4);
        assert_eq!(view.child_count(), 5);
        assert_eq!(view.used_bytes(), internal.used_bytes());
        for (i, c) in internal.children().iter().enumerate() {
            assert_eq!(view.child(i), Ok(Some(*c)));
        }
        assert_eq!(view.child(5), Ok(None));
        for (i, k) in internal.keys().iter().enumerate() {
            assert_eq!(view.key(i), &k.0[..]);
        }
        let probes: [&[u8]; 10] = [
            &[],
            &[0, 0],
            &[1, 2],
            &[1, 2, 0],
            &[3],
            &[3, 4, 5, 6, 7],
            &[3, 5],
            &[3, 9],
            &[4],
            &[9, 9],
        ];
        for probe in probes {
            assert_eq!(
                view.find_child_index(probe),
                internal.find_child_index(&key(probe)),
                "probe {probe:?}"
            );
        }
        assert!(matches!(
            NodeView::new(&buf[..], l),
            Ok(NodeView::Internal(_))
        ));
        assert!(matches!(
            LeafPageView::new(&buf[..], l),
            Err(PageCodecError::WrongPageKind { .. })
        ));
    }

    #[test]
    fn leaf_view_reads_match_owned_page() {
        let mut leaf = LeafPage::new(layout(4));
        let keys: [&[u8]; 6] = [&[10], &[20, 1, 1], &[20], &[30, 0], &[40], &[50, 5, 5, 5]];
        for (i, k) in keys.iter().enumerate() {
            leaf.insert(key(k), rid(i as u32)).unwrap();
        }
        leaf.set_next_leaf_id(Some(pid(77)));
        leaf.remove_key(&key(&[30, 0])).unwrap();
        let buf = encode_leaf(&leaf);

        let view = LeafPageView::new(&buf[..], layout(4)).unwrap();
        assert_eq!(view.size(), 6);
        assert_eq!(view.next_leaf_id(), Ok(Some(pid(77))));
        assert_eq!(view.tombstone_count(), 1);
        assert!(view.is_tombstoned(3));
        assert_eq!(view.used_bytes(), leaf.used_bytes());
        for a in 0u8..60 {
            for probe in [vec![a], vec![a, 0], vec![a, 1, 1], vec![a, 5, 5, 5]] {
                let k = key(&probe);
                assert_eq!(view.get(&probe).unwrap(), leaf.get(&k), "get {probe:?}");
                assert_eq!(view.lower_bound(&probe), leaf.lower_bound(&k));
                assert_eq!(view.upper_bound(&probe), leaf.upper_bound(&k));
                assert_eq!(view.search(&probe), leaf.search(&k));
            }
        }
    }

    /// Every corruption is an error rather than a panic: the view rejects the ones that
    /// would make it panic when it is built, reports a bad record id when it is read, and the
    /// owned decoder rejects all of them.
    #[test]
    fn corrupt_leaf_pages_are_errors_not_panics() {
        let l = layout(4);
        let mut leaf = LeafPage::new(l);
        for k in [10u8, 20, 30] {
            leaf.insert(key(&[k, k]), rid(u32::from(k))).unwrap();
        }
        leaf.remove_key(&key(&[20, 20])).unwrap();
        let good = encode_leaf(&leaf);
        let area = l.leaf_area();
        let slots_start = PAGE_SIZE - area.capacity();
        let first_entry = PAGE_SIZE - 3 * (2 + area.payload_len());
        let corrupted = |corrupt: &dyn Fn(&mut [u8])| {
            let mut buf = good.clone();
            corrupt(&mut buf[..]);
            buf
        };

        let rejected_by_view = [
            (
                "tombstone past the last slot",
                corrupted(&|b| put_u32_le(b, l.leaf_tombstone_offset(0), 3)),
            ),
            (
                "tombstone count past capacity",
                corrupted(&|b| put_u32_le(b, l.leaf_tombstone_count_offset(), u32::MAX)),
            ),
            (
                "more entries than slots fit in the page",
                corrupted(&|b| put_u32_le(b, NODE_SIZE_OFFSET, u32::MAX)),
            ),
            (
                "an entry that starts inside the slot array",
                corrupted(&|b| {
                    b[slots_start..slots_start + 2].copy_from_slice(&0u16.to_le_bytes())
                }),
            ),
            (
                "entries out of order",
                corrupted(&|b| {
                    let (first, second) = (b[slots_start], b[slots_start + 1]);
                    b[slots_start + 2] = first;
                    b[slots_start + 3] = second;
                    b[slots_start] = 0xff;
                    b[slots_start + 1] = 0x1f;
                }),
            ),
        ];
        for (what, buf) in &rejected_by_view {
            assert!(
                matches!(
                    LeafPageView::new(&buf[..], l),
                    Err(PageCodecError::Malformed(_))
                ),
                "view rejects {what}"
            );
            assert!(
                LeafPage::decode(&buf[..], l).is_err(),
                "decode rejects {what}"
            );
        }

        let first_rid_off = first_entry + 2;
        let wide_ids = [
            corrupted(&|b| put_u64_le(b, first_rid_off, u64::from(u32::MAX) + 1)),
            corrupted(&|b| put_u64_le(b, first_rid_off + 8, u64::from(u32::MAX) + 1)),
        ];
        for buf in &wide_ids {
            let view = LeafPageView::new(&buf[..], l).unwrap();
            assert!(
                view.record_id(0).is_err(),
                "a wide record id is reported, not truncated"
            );
            assert!(view.get(&[10, 10]).is_err());
            assert!(LeafPage::decode(&buf[..], l).is_err());
        }

        let out_of_order = corrupted(&|b| b[first_entry] = 99);
        assert!(
            LeafPage::decode(&out_of_order[..], l).is_err(),
            "decode rejects keys out of order instead of asserting"
        );
    }

    /// Runs a random mix of inserts and removes of keys of varying lengths against both the
    /// owned `LeafPage` and the in-place `LeafPageView`, checking after every step that the
    /// view's bytes are identical to encoding the owned page.
    fn differential(l: Layout, key_lens: &[usize], seed: u64, ops: usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut owned = LeafPage::new(l);
        owned.set_next_leaf_id(Some(pid(5)));
        let mut buf = encode_leaf(&owned);
        let mut fulls = 0;

        for step in 0..ops {
            let len = key_lens[rng.random_range(0..key_lens.len())];
            let k = vec![rng.random_range(0..12u8); len];
            let index_key = IndexKey(k.clone());
            let r = rid(rng.random_range(0..1000));
            let mut view = LeafPageView::new(&mut buf[..], l).unwrap();
            if rng.random_bool(0.6) {
                match view.insert(&k, r) {
                    Ok(()) => owned.insert(index_key, r).unwrap(),
                    Err(LeafInsertError::DuplicateKey) => {
                        assert!(
                            owned.get(&index_key).is_some(),
                            "step {step}: false duplicate"
                        )
                    }
                    Err(LeafInsertError::Full) => {
                        fulls += 1;
                        let mut grown = owned.clone();
                        grown.insert(index_key, r).unwrap();
                        assert!(grown.is_overfull(), "step {step}: false full");
                    }
                }
            } else {
                let got = view.remove_key(&k).is_ok();
                let want = owned.remove_key(&index_key).is_ok();
                assert_eq!(got, want, "step {step}: remove result");
            }
            let expected = encode_leaf(&owned);
            assert!(
                buf[..] == expected[..],
                "step {step}: view bytes diverged from owned encoding"
            );
            let view = LeafPageView::new(&buf[..], l).unwrap();
            assert_eq!(view.used_bytes(), owned.used_bytes());
            assert_eq!(
                view.may_split_on_insert(),
                l.leaf_area().capacity() - owned.used_bytes() < l.leaf_max_entry_bytes()
            );
        }
        if key_lens.iter().any(|&len| len > 1000) {
            assert!(fulls > 0, "long keys fill the leaf");
        }
    }

    #[test]
    fn leaf_view_mutations_match_owned_codec_with_short_keys() {
        for seed in 0..20 {
            differential(layout(3), &[0, 1, 2, 3], seed, 300);
        }
    }

    #[test]
    fn leaf_view_mutations_match_owned_codec_with_long_keys() {
        let l = layout(u32::try_from(MAX_KEY_LEN).unwrap());
        for seed in 0..10 {
            differential(l, &[1, 300, 1500, MAX_KEY_LEN], seed, 400);
        }
    }

    #[test]
    fn leaf_view_mutations_match_owned_codec_with_one_or_no_tombstones() {
        for tombstones in [0, 1] {
            let l = Layout::with_tombstone_capacity(2000, tombstones).unwrap();
            for seed in 0..10 {
                differential(l, &[1, 700, 2000], seed, 300);
            }
        }
    }

    #[test]
    fn leaf_view_insert_reports_full_without_modifying() {
        let l = layout(2000);
        let mut leaf = LeafPage::new(l);
        for k in [1u8, 2, 3, 4] {
            leaf.insert(key(&[k; 2000]), rid(u32::from(k))).unwrap();
        }
        let mut buf = encode_leaf(&leaf);
        let before = buf.clone();
        let mut view = LeafPageView::new(&mut buf[..], l).unwrap();
        assert!(view.may_split_on_insert());
        assert_eq!(view.insert(&[9; 2000], rid(9)), Err(LeafInsertError::Full));
        assert_eq!(
            view.insert(&[2; 2000], rid(9)),
            Err(LeafInsertError::DuplicateKey)
        );
        assert!(buf[..] == before[..]);

        let mut view = LeafPageView::new(&mut buf[..], l).unwrap();
        view.insert(&[9; 10], rid(9)).unwrap();
        view.remove_key(&[2; 2000]).unwrap();
        view.insert(&[2; 2000], rid(42)).unwrap();
        assert_eq!(view.get(&[2; 2000]).unwrap(), Some(rid(42)));
        assert_eq!(view.get(&[9; 10]).unwrap(), Some(rid(9)));
        assert_eq!(view.tombstone_count(), 0);
    }

    #[test]
    fn node_views_report_when_a_change_below_may_restructure_them() {
        let l = layout(100);
        let mut leaf = LeafPage::new(l);
        let mut i = 0u32;
        while l.leaf_area().capacity() - leaf.used_bytes() >= l.leaf_max_entry_bytes() {
            let buf = encode_leaf(&leaf);
            assert!(!NodeView::new(&buf[..], l).unwrap().may_split_on_insert());
            leaf.insert(IndexKey(i.to_be_bytes().repeat(25)), rid(i))
                .unwrap();
            i += 1;
        }
        let buf = encode_leaf(&leaf);
        let node = NodeView::new(&buf[..], l).unwrap();
        assert!(node.may_split_on_insert());
        assert!(!node.may_underflow_on_remove());

        let empty = encode_leaf(&LeafPage::new(l));
        assert!(
            NodeView::new(&empty[..], l)
                .unwrap()
                .may_underflow_on_remove()
        );

        let mut internal = InternalPage::new_root(l, pid(1), key(&[1]), pid(2));
        let mut buf = [0u8; PAGE_SIZE];
        internal.encode(&mut buf).unwrap();
        let node = NodeView::new(&buf[..], l).unwrap();
        assert!(!node.may_split_on_insert());
        assert!(node.may_underflow_on_remove());
        let mut next = 3u32;
        while !internal.is_overfull() {
            let separator = [&[2u8][..], &next.to_be_bytes().repeat(12)].concat();
            internal.append_key_and_child(IndexKey(separator), pid(next));
            next += 1;
        }
        internal.pop_last_child_and_key();
        internal.encode(&mut buf).unwrap();
        let node = NodeView::new(&buf[..], l).unwrap();
        assert!(node.may_split_on_insert());
        assert!(!node.may_underflow_on_remove());
    }

    #[test]
    fn a_corrupt_child_slot_is_a_codec_error_not_a_missing_child() {
        let l = layout(1);
        let internal = InternalPage::new_root(l, pid(11), key(&[5]), pid(22));
        let mut buf = [0u8; PAGE_SIZE];
        internal.encode(&mut buf).unwrap();
        let second_child = PAGE_SIZE - size_of::<u64>();

        put_u64_le(&mut buf, second_child, u64::MAX);
        let view = InternalPageView::new(&buf[..], l).unwrap();
        assert_eq!(view.child(0), Ok(Some(pid(11))));
        assert_eq!(
            view.child_for(&[9]),
            Err(PageCodecError::Malformed(
                "an internal page child names no page"
            ))
        );
        assert_eq!(
            InternalPage::decode(&buf, l).err(),
            view.child_for(&[9]).err()
        );

        put_u64_le(&mut buf, NODE_PAGE_ID_OFFSET, u64::from(u32::MAX));
        let view = InternalPageView::new(&buf[..], l).unwrap();
        assert!(matches!(view.child(0), Err(PageCodecError::BufferError(_))));
    }
}
