use crate::buf::{BufReader, BufWriter, BufferError};
use crate::catalog::index_schema::IndexKey;
use crate::disk::PAGE_SIZE;
use crate::index::RemoveError;
use crate::index::page_codec::{
    COMMON_HEADER_BYTES, KIND_HEADER, KIND_INTERNAL, KIND_LEAF, KIND_OFFSET, PageCodecError,
    U32_BYTES, U64_BYTES, decode_page_id, decode_record_id, encode_page_id, encode_record_id,
    read_common_header, write_common_header,
};
use crate::table_heap::RecordId;
use log::{error, info};
use std::collections::VecDeque;
use std::convert::TryFrom;

pub type PageId = usize;

pub const INVALID_PAGE_ID: PageId = usize::MAX;
pub const NUM_TOMBSTONES: u32 = 256;

// each page id is written to disk as a u64
const PAGE_ID_ENCODED_BYTES: usize = U64_BYTES;
const RECORD_ID_ENCODED_BYTES: usize = 2 * U64_BYTES;
const TOMBSTONE_SLOT_ENCODED_BYTES: usize = U32_BYTES;

// Per-page fixed headers (common header + per-kind fields).
const INTERNAL_FIXED_HEADER_BYTES: usize = COMMON_HEADER_BYTES + (4 * U32_BYTES); // size, max_size, child_count, reserved
const LEAF_FIXED_HEADER_BYTES: usize = COMMON_HEADER_BYTES + (2 * U32_BYTES) + U64_BYTES; // size, max_size, next_leaf_id
const HEADER_FIXED_HEADER_BYTES: usize = COMMON_HEADER_BYTES + (2 * U32_BYTES) + (2 * U64_BYTES); // key_len, reserved, root_page_id, first_leaf_id

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

// probably a simpler way of doing this
pub enum Page {
    Internal(InternalPage),
    Leaf(LeafPage),
}

// also probably a better way of doing this (see kind())
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
    pub children: Vec<PageId>, // page_ids that child nodes are in
    pub keys: Vec<IndexKey>,   // invariant: children.len() == keys.len() + 1
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

    // TODO: simplify this horrid calculation
    pub fn max_size_for_layout(key_len: usize) -> Option<u32> {
        if key_len == 0 {
            return None;
        }
        // Internal layout:
        // - fixed header
        // - (n+1) child pointers (u64)
        // - n keys (key_len bytes)
        //
        // bytes = INTERNAL_FIXED_HEADER_BYTES + (n+1)*PAGE_ID + n*key_len
        // => n <= floor((PAGE_SIZE - INTERNAL_FIXED_HEADER_BYTES - PAGE_ID) / (key_len + PAGE_ID))
        let denom = key_len.checked_add(PAGE_ID_ENCODED_BYTES)?;
        let fixed = INTERNAL_FIXED_HEADER_BYTES.checked_add(PAGE_ID_ENCODED_BYTES)?;
        let slack = PAGE_SIZE.checked_sub(fixed)?;
        let n = slack / denom;
        u32::try_from(n).ok().filter(|&v| v > 0)
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

    /// Given a key, returns the index of the child page that should be searched next
    /// in order to continue searching for the given key.
    /// The returned index should be used to look up the child page ID in the page's list of children when searching,
    /// or to determine where to insert a child.
    pub fn find_child_index(&self, key: &IndexKey) -> usize {
        // Returns i such that:
        // - i == 0: key < keys[0]
        // - 0 < i < keys.len(): keys[i-1] <= key < keys[i]
        // - i == keys.len(): keys[last] <= key
        self.keys.partition_point(|k| k <= key)
    }

    /// Inserts a child, such that the inserted child is to the right of the separator key.
    ///
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

    pub fn split_into(&mut self, right: &mut InternalPage) -> IndexKey {
        assert!(
            right.children.is_empty() && right.keys.is_empty(),
            "right page must be empty before split"
        );
        self.validate();

        let mid = self.keys.len() / 2;

        // Right gets keys[mid+1..] and children[mid+1..]. The middle key is promoted to the parent.
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

    /// Removes the separator key between `children[left_child_index]` and `children[left_child_index+1]`,
    /// and removes (and returns) the right child pointer (`children[left_child_index+1]`).
    /// TODO: can we make these internal
    /// The interface for this class is massive and impossible to reason with
    /// need to reduce the surface area.
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

    // pub fn init_as_root(&mut self, left_child: PageId, sep_key: IndexKey, right_child: PageId) {
    //     self.children = vec![left_child, right_child];
    //     self.keys = vec![sep_key];
    //     self.base.size = 1;
    //     self.validate();
    // }
}

#[derive(Debug)]
pub struct LeafPage {
    base: Base,
    pub keys: Vec<IndexKey>,
    pub record_ids: Vec<RecordId>,
    pub next_leaf_id: PageId, // for range scans
    pub tombstones: VecDeque<u32>,
    pub tombstone_capacity: usize,
}

impl LeafPage {
    pub fn max_size_for_layout(key_len: usize) -> Option<u32> {
        if key_len == 0 {
            return None;
        }
        // Leaf layout:
        // - fixed header
        // - n entries: key_len + RecordId
        // - trailing tombstone count (u32)
        //
        // bytes = LEAF_FIXED_HEADER_BYTES + n*(key_len + RecordId) + u32
        let entry_bytes = key_len.checked_add(RECORD_ID_ENCODED_BYTES)?;
        let fixed = LEAF_FIXED_HEADER_BYTES.checked_add(TOMBSTONE_SLOT_ENCODED_BYTES)?;
        let slack = PAGE_SIZE.checked_sub(fixed)?;
        let n = slack / entry_bytes;
        u32::try_from(n).ok().filter(|&v| v > 0)
    }

    pub fn tombstone_capacity_for_layout(max_size: u32, key_len: u32, requested: u32) -> u32 {
        let key_len = match usize::try_from(key_len) {
            Ok(v) => v,
            Err(_) => return 0,
        };
        if key_len == 0 {
            return 0;
        }
        let entry_bytes = match key_len.checked_add(RECORD_ID_ENCODED_BYTES) {
            Some(v) => v,
            None => return 0,
        };
        let max_size = match usize::try_from(max_size) {
            Ok(v) => v,
            Err(_) => return 0,
        };
        let live_needed =
            match LEAF_FIXED_HEADER_BYTES.checked_add(match max_size.checked_mul(entry_bytes) {
                Some(v) => v,
                None => return 0,
            }) {
                Some(v) => v,
                None => return 0,
            };
        if live_needed >= PAGE_SIZE {
            return 0;
        }
        let slack = PAGE_SIZE - live_needed;
        if slack < TOMBSTONE_SLOT_ENCODED_BYTES {
            return 0;
        }
        let fit = (slack - TOMBSTONE_SLOT_ENCODED_BYTES) / TOMBSTONE_SLOT_ENCODED_BYTES;
        let requested = requested as usize;
        u32::try_from(requested.min(fit)).unwrap_or(0)
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

    /// Performs a binary search on the leaf page's list of keys for the given key
    /// Returns an Option<RecordId> which will contain the record ID
    /// for the given key is found, if the given key is found.
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
        // Remove tombstones pointing at the removed slot; shift all later slots left by 1.
        let mut out = VecDeque::with_capacity(self.tombstones.len());
        for s in self.tombstones.drain(..) {
            if s == removed_at {
                continue;
            }
            if s > removed_at {
                out.push_back(s - 1);
            } else {
                out.push_back(s);
            }
        }
        self.tombstones = out;
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

        // Tombstones are stored as a trailing section in the free space after the live entries:
        // u32 count, followed by `count` u32 slot indexes (oldest-first).
        // Tombstones are part of the logical page contents; if they don't fit, we error.
        let available = PAGE_SIZE.saturating_sub(live_needed);
        let tombstone_bytes = TOMBSTONE_SLOT_ENCODED_BYTES
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

        // Make decoding deterministic even when the remainder is all zeros by always writing a
        // tombstone count (which may be 0).
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

        // Optional trailing tombstones section.
        if r.remaining() >= TOMBSTONE_SLOT_ENCODED_BYTES {
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
    pub fn insert(&mut self, key: IndexKey, record_id: RecordId) -> Result<(), ()> {
        match self.keys.binary_search(&key) {
            Ok(pos) => {
                let slot = u32::try_from(pos).map_err(|_| ())?;
                if !self.is_tombstoned(slot) {
                    error!("attempted to insert duplicate key into leaf page");
                    return Err(());
                }
                self.record_ids[pos] = record_id;
                self.remove_tombstone(slot);
                self.validate();
                Ok(())
            }
            Err(pos) => {
                let insert_at = u32::try_from(pos).map_err(|_| ())?;
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
        let mid_u32 = u32::try_from(mid).expect("mid fits u32");

        right.keys = self.keys.split_off(mid);
        right.record_ids = self.record_ids.split_off(mid);

        // Tombstone slot indexes move with the corresponding key entries.
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

    pub fn absorb_right(&mut self, right: &mut LeafPage) {
        self.validate();
        right.validate();

        let left_len = u32::try_from(self.keys.len()).expect("keys.len fits u32");
        self.keys.append(&mut right.keys);
        self.record_ids.append(&mut right.record_ids);
        self.next_leaf_id = right.next_leaf_id;
        self.base.size = self.keys.len() as u32;

        // Keep tombstones across merges; deletes from the merged-in leaf (`right`) are considered
        // more recent than deletes already recorded in `self`, so we append them after `self`'s
        // existing tombstones. Right slots are offset by the old left length.
        for slot in right.tombstones.drain(..) {
            self.tombstones.push_back(slot + left_len);
        }
        while self.tombstones.len() > self.tombstone_capacity {
            let evicted = self
                .tombstones
                .pop_front()
                .expect("len > capacity implies non-empty");
            self.physical_remove_at(evicted);
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
        match self.keys.binary_search(key) {
            Ok(_) => return true,
            Err(_) => {
                error!("could not find key in page: {:?}", key);
                return false;
            }
        };
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
        let mut slot = u32::try_from(pos).expect("pos fits u32");
        if self.is_tombstoned(slot) {
            return Err(RemoveError::KeyNotFound);
        }

        // If there is no tombstone space, fall back to immediate physical deletion.
        if self.tombstone_capacity == 0 {
            self.physical_remove_at(slot);
            self.validate();
            return Ok(());
        }

        // If the tombstone FIFO is full, physically delete the oldest tombstoned key to make room.
        if self.tombstones.len() == self.tombstone_capacity {
            let evicted = self
                .tombstones
                .pop_front()
                .expect("len == capacity implies non-empty");
            self.physical_remove_at(evicted);
            if evicted < slot {
                slot -= 1;
            }
        }

        self.tombstones.push_back(slot);

        self.validate();
        Ok(())
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
        let slot = u32::try_from(self.keys.len()).expect("len fits u32");
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
    use crate::index::b_plus_page::{
        HeaderPage, INVALID_PAGE_ID, InternalPage, LeafPage, NUM_TOMBSTONES,
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
        left.tombstones = VecDeque::from([1u32, 3u32]); // keys 20 and 40 tombstoned
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
        left.tombstones = VecDeque::from([0u32]); // delete key 10
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
        right.tombstones = VecDeque::from([0u32]); // delete key 30
        right.validate();

        // Capacity is 1, but merging would create 2 tombstones. Since the absorbed leaf's deletes are
        // considered more recent, the older tombstone from `left` must be evicted and physically
        // deleted first.
        left.absorb_right(&mut right);
        assert_eq!(
            left.keys,
            vec![IndexKey(vec![20]), IndexKey(vec![30]), IndexKey(vec![40])]
        );
        assert_eq!(left.tombstones, VecDeque::from([1u32])); // key 30 shifted to slot 1
        assert!(left.get(&IndexKey(vec![10])).is_none()); // physically removed
        assert!(left.get(&IndexKey(vec![30])).is_none()); // still tombstoned
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

        // Tombstones capacity is 2: the oldest (10) is evicted (physically deleted) when adding 30.
        assert!(leaf.remove_key(&IndexKey(vec![30])).is_ok());
        assert_eq!(leaf.keys.len(), 2);
        assert_eq!(leaf.tombstones.len(), 2);
        assert_eq!(leaf.tombstones[0], 0); // key 20 shifted to slot 0 after deleting key 10
        assert_eq!(leaf.tombstones[1], 1); // key 30 is now at slot 1
        assert!(leaf.get(&IndexKey(vec![30])).is_none());

        // Tombstones are persisted as a trailing section in the leaf encoding.
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
        // Fill the page so that only 8 bytes remain after the live entries, then try to encode 2
        // tombstones (4 bytes count + 2 * 4 bytes indexes = 12 bytes) => must error.
        let key_len = 1u32;
        let max_size = 480u32; // 24 + 480*(1+16)=8184 => 8 bytes slack
        let mut leaf = LeafPage::new(max_size, 2);
        leaf.keys = vec![IndexKey(vec![0u8]); 480];
        leaf.record_ids = (0u32..480u32)
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
}
