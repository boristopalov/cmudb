//! Variable-length entries of a B+ tree page: each entry is a key followed by a fixed-size
//! payload, the entries are packed in key order against the end of the page, and an array of
//! `u16` start offsets after the page's header finds them. Entry `i` runs from its start to the
//! next entry's start, or to the end of the page for the last entry. Free space sits between
//! the slot array and the first entry.
//!
//! The owned codecs in `b_plus_page` and the in-place views in `page_view` both read and
//! write entries through [`EntryArea`].

use crate::buf::{get_u16_le, put_u16_le};
use crate::disk::PAGE_SIZE;
use crate::index::page_codec::PageCodecError;

pub(crate) const SLOT_ENCODED_BYTES: usize = size_of::<u16>();

const _: () = assert!(
    PAGE_SIZE <= u16::MAX as usize,
    "a slot's u16 offset reaches every byte of a page"
);

fn page_offset(at: usize) -> u16 {
    u16::try_from(at).expect("a page offset fits in u16")
}

const SLOTS_PAST_PAGE: PageCodecError = PageCodecError::Malformed("slot array runs past the page");
const ENTRY_OUT_OF_BOUNDS: PageCodecError =
    PageCodecError::Malformed("entry offsets are out of order or outside the page");

/// Where a page's slot array starts and how many payload bytes follow each entry's key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryArea {
    slots_start: usize,
    payload_len: usize,
}

impl EntryArea {
    pub(crate) const fn new(slots_start: usize, payload_len: usize) -> Self {
        assert!(
            slots_start < PAGE_SIZE,
            "the slot array starts inside the page"
        );
        Self {
            slots_start,
            payload_len,
        }
    }

    /// Bytes the slots and entries of a page can take together.
    pub const fn capacity(self) -> usize {
        PAGE_SIZE - self.slots_start
    }

    /// Bytes an entry with a `key_len`-byte key takes, its slot included.
    pub const fn entry_cost(self, key_len: usize) -> usize {
        SLOT_ENCODED_BYTES + key_len + self.payload_len
    }

    pub const fn payload_len(self) -> usize {
        self.payload_len
    }

    #[inline]
    fn slot_offset(self, i: usize) -> usize {
        self.slots_start + i * SLOT_ENCODED_BYTES
    }

    #[inline]
    fn start(self, buf: &[u8], i: usize) -> usize {
        usize::from(get_u16_le(buf, self.slot_offset(i)))
    }

    #[inline]
    fn end(self, buf: &[u8], n: usize, i: usize) -> usize {
        if i + 1 < n {
            self.start(buf, i + 1)
        } else {
            PAGE_SIZE
        }
    }

    /// Offset of the first entry byte, or `PAGE_SIZE` when the area holds no entries.
    #[inline]
    fn data_start(self, buf: &[u8], n: usize) -> usize {
        if n == 0 {
            PAGE_SIZE
        } else {
            self.start(buf, 0)
        }
    }

    /// Bytes `n` entries take, their slots included.
    pub fn used(self, buf: &[u8], n: usize) -> usize {
        n * SLOT_ENCODED_BYTES + (PAGE_SIZE - self.data_start(buf, n))
    }

    /// Bytes left for more entries and their slots.
    pub fn free(self, buf: &[u8], n: usize) -> usize {
        self.capacity() - self.used(buf, n)
    }

    /// Fails with `Malformed` unless `buf` holds `n` entries whose slots end before the first
    /// entry and whose entries lie in order within the page, each at least a payload long.
    /// Every other method trusts an area that passed this check.
    pub fn check(self, buf: &[u8], n: usize) -> Result<(), PageCodecError> {
        if buf.len() != PAGE_SIZE {
            return Err(PageCodecError::InvalidSize);
        }
        let slots_end = n
            .checked_mul(SLOT_ENCODED_BYTES)
            .and_then(|bytes| bytes.checked_add(self.slots_start))
            .filter(|&end| end <= PAGE_SIZE)
            .ok_or(SLOTS_PAST_PAGE)?;
        if n == 0 {
            return Ok(());
        }
        if self.start(buf, 0) < slots_end {
            return Err(ENTRY_OUT_OF_BOUNDS);
        }
        for i in 0..n {
            let start = self.start(buf, i);
            let end = self.end(buf, n, i);
            if start.saturating_add(self.payload_len) > end {
                return Err(ENTRY_OUT_OF_BOUNDS);
            }
        }
        Ok(())
    }

    /// The key of entry `i` of `n`. Panics if `i >= n`.
    pub fn key(self, buf: &[u8], n: usize, i: usize) -> &[u8] {
        assert!(i < n, "entry index out of bounds");
        let end = self.end(buf, n, i);
        &buf[self.start(buf, i)..end - self.payload_len]
    }

    /// The payload of entry `i` of `n`. Panics if `i >= n`.
    pub fn payload(self, buf: &[u8], n: usize, i: usize) -> &[u8] {
        assert!(i < n, "entry index out of bounds");
        let end = self.end(buf, n, i);
        &buf[end - self.payload_len..end]
    }

    /// Writes entry `i` at offset `at` and points slot `i` at it, returning the offset after
    /// it. For encoders that lay out every entry in order from `PAGE_SIZE - total bytes`.
    pub fn put_entry(
        self,
        buf: &mut [u8],
        i: usize,
        at: usize,
        key: &[u8],
        payload: &[u8],
    ) -> usize {
        assert_eq!(payload.len(), self.payload_len, "entry payload length");
        let slot = self.slot_offset(i);
        put_u16_le(buf, slot, page_offset(at));
        let key_end = at + key.len();
        buf[at..key_end].copy_from_slice(key);
        buf[key_end..key_end + payload.len()].copy_from_slice(payload);
        key_end + payload.len()
    }

    /// Overwrites the payload of entry `i` of `n`.
    pub fn set_payload(self, buf: &mut [u8], n: usize, i: usize, payload: &[u8]) {
        assert_eq!(payload.len(), self.payload_len, "entry payload length");
        let end = self.end(buf, n, i);
        buf[end - self.payload_len..end].copy_from_slice(payload);
    }

    /// Inserts `key` and `payload` as entry `pos` of `n`, moving the entries before it down
    /// the page. Panics if the entry does not fit in [`EntryArea::free`] bytes.
    pub fn insert(self, buf: &mut [u8], n: usize, pos: usize, key: &[u8], payload: &[u8]) {
        assert!(pos <= n, "entry insert position out of bounds");
        let len = key.len() + payload.len();
        assert!(
            self.free(buf, n) >= self.entry_cost(key.len()),
            "entry does not fit in the page"
        );
        let data_start = self.data_start(buf, n);
        let at = if pos < n {
            self.start(buf, pos)
        } else {
            PAGE_SIZE
        };
        buf.copy_within(data_start..at, data_start - len);
        for i in 0..pos {
            let moved = self.start(buf, i) - len;
            put_u16_le(buf, self.slot_offset(i), page_offset(moved));
        }
        let slot = self.slot_offset(pos);
        buf.copy_within(slot..self.slot_offset(n), slot + SLOT_ENCODED_BYTES);
        self.put_entry(buf, pos, at - len, key, payload);
    }

    /// Removes entry `pos` of `n`, moving the entries before it up the page and zeroing the
    /// bytes it frees.
    pub fn remove(self, buf: &mut [u8], n: usize, pos: usize) {
        assert!(pos < n, "entry remove position out of bounds");
        let start = self.start(buf, pos);
        let len = self.end(buf, n, pos) - start;
        let data_start = self.data_start(buf, n);
        buf.copy_within(data_start..start, data_start + len);
        buf[data_start..data_start + len].fill(0);
        for i in 0..pos {
            let moved = self.start(buf, i) + len;
            put_u16_le(buf, self.slot_offset(i), page_offset(moved));
        }
        let slot = self.slot_offset(pos);
        let slots_end = self.slot_offset(n);
        buf.copy_within(slot + SLOT_ENCODED_BYTES..slots_end, slot);
        buf[slots_end - SLOT_ENCODED_BYTES..slots_end].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AREA: EntryArea = EntryArea::new(16, 2);

    fn keys(buf: &[u8], n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| AREA.key(buf, n, i).to_vec()).collect()
    }

    fn encoded(entries: &[(&[u8], [u8; 2])]) -> Vec<u8> {
        let mut buf = vec![0u8; PAGE_SIZE];
        let total: usize = entries.iter().map(|(k, p)| k.len() + p.len()).sum();
        let mut at = PAGE_SIZE - total;
        for (i, (k, p)) in entries.iter().enumerate() {
            at = AREA.put_entry(&mut buf, i, at, k, p);
        }
        assert_eq!(at, PAGE_SIZE);
        buf
    }

    #[test]
    fn inserts_and_removes_match_encoding_the_entries_in_order() {
        let mut buf = vec![0u8; PAGE_SIZE];
        let mut n = 0;
        for (pos, key) in [
            (0, &b"m"[..]),
            (0, b"aaa"),
            (2, b"zz"),
            (1, b""),
            (2, b"cc"),
        ] {
            AREA.insert(&mut buf, n, pos, key, &[key.len() as u8, 9]);
            n += 1;
        }
        let expected = encoded(&[
            (b"aaa", [3, 9]),
            (b"", [0, 9]),
            (b"cc", [2, 9]),
            (b"m", [1, 9]),
            (b"zz", [2, 9]),
        ]);
        assert!(
            buf == expected,
            "entries inserted at positions encode like the same list written in order"
        );
        AREA.check(&buf, n).unwrap();
        assert_eq!(AREA.used(&buf, n), 5 * 2 + (3 + 2 + 1 + 2) + 5 * 2);
        assert_eq!(AREA.payload(&buf, n, 2), &[2, 9]);

        AREA.remove(&mut buf, n, 1);
        AREA.remove(&mut buf, n - 1, 3);
        n -= 2;
        assert_eq!(
            keys(&buf, n),
            vec![b"aaa".to_vec(), b"cc".to_vec(), b"m".to_vec()]
        );
        let expected = encoded(&[(b"aaa", [3, 9]), (b"cc", [2, 9]), (b"m", [1, 9])]);
        assert!(buf == expected, "removed entries leave zeroed bytes");

        AREA.set_payload(&mut buf, n, 0, &[7, 7]);
        assert_eq!(AREA.payload(&buf, n, 0), &[7, 7]);
        for i in (0..n).rev() {
            AREA.remove(&mut buf, n, i);
            n -= 1;
        }
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(AREA.free(&buf, 0), AREA.capacity());
    }

    #[test]
    fn check_rejects_slots_that_overlap_or_leave_the_page() {
        let good = encoded(&[(b"ab", [0, 0]), (b"cd", [0, 0])]);
        AREA.check(&good, 2).unwrap();
        assert_eq!(AREA.check(&good, PAGE_SIZE), Err(SLOTS_PAST_PAGE));
        assert_eq!(AREA.check(&good, usize::MAX), Err(SLOTS_PAST_PAGE));

        let mut reversed = good.clone();
        let (first, second) = (AREA.start(&good, 0), AREA.start(&good, 1));
        put_u16_le(&mut reversed, AREA.slot_offset(0), second as u16);
        put_u16_le(&mut reversed, AREA.slot_offset(1), first as u16);
        assert_eq!(AREA.check(&reversed, 2), Err(ENTRY_OUT_OF_BOUNDS));

        let mut into_slots = good.clone();
        put_u16_le(&mut into_slots, AREA.slot_offset(0), 0);
        assert_eq!(AREA.check(&into_slots, 2), Err(ENTRY_OUT_OF_BOUNDS));

        let mut too_short = good;
        put_u16_le(&mut too_short, AREA.slot_offset(1), (PAGE_SIZE - 1) as u16);
        assert_eq!(AREA.check(&too_short, 2), Err(ENTRY_OUT_OF_BOUNDS));
    }
}

#[cfg(test)]
mod properties {
    use super::*;
    use proptest::prelude::*;

    const MAX_PAYLOAD: usize = 16;

    #[derive(Debug, Clone)]
    enum Op {
        Insert {
            at: prop::sample::Index,
            key: Vec<u8>,
            payload: [u8; MAX_PAYLOAD],
        },
        Remove {
            at: prop::sample::Index,
        },
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            3 => (
                any::<prop::sample::Index>(),
                prop_oneof![
                    prop::collection::vec(any::<u8>(), 0..=8),
                    prop::collection::vec(any::<u8>(), 0..=600),
                ],
                any::<[u8; MAX_PAYLOAD]>(),
            )
                .prop_map(|(at, key, payload)| Op::Insert { at, key, payload }),
            2 => any::<prop::sample::Index>().prop_map(|at| Op::Remove { at }),
        ]
    }

    /// The bytes `put_entry` writes for `entries` laid out in order on a zeroed page, the
    /// encoding the B+ tree page codecs produce.
    fn encoded(area: EntryArea, entries: &[(Vec<u8>, Vec<u8>)]) -> Vec<u8> {
        let mut buf = vec![0u8; PAGE_SIZE];
        let total: usize = entries.iter().map(|(k, p)| k.len() + p.len()).sum();
        let mut at = PAGE_SIZE - total;
        for (i, (k, p)) in entries.iter().enumerate() {
            at = area.put_entry(&mut buf, i, at, k, p);
        }
        buf
    }

    fn cost(area: EntryArea, entries: &[(Vec<u8>, Vec<u8>)]) -> usize {
        entries.iter().map(|(k, _)| area.entry_cost(k.len())).sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 512,
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        /// Any sequence of inserts and removes on a page with entries keeps the area passing
        /// `check`, keeps `free` equal to the capacity left by the entries, and leaves the page
        /// byte for byte equal to encoding the same entries in order, so removed bytes are
        /// zeroed. Inserts that do not fit are skipped, which also fills pages to capacity.
        #[test]
        fn inserts_and_removes_match_encoding_the_model(
            slots_start in 0usize..64,
            payload_len in 0usize..=MAX_PAYLOAD,
            ops in prop::collection::vec(op(), 1..200),
        ) {
            let area = EntryArea::new(slots_start, payload_len);
            let mut buf = vec![0u8; PAGE_SIZE];
            let mut model: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for op in ops {
                let n = model.len();
                match op {
                    Op::Insert { at, key, payload } => {
                        if area.free(&buf, n) < area.entry_cost(key.len()) {
                            continue;
                        }
                        let pos = at.index(n + 1);
                        let payload = payload[..payload_len].to_vec();
                        area.insert(&mut buf, n, pos, &key, &payload);
                        model.insert(pos, (key, payload));
                    }
                    Op::Remove { at } => {
                        if n == 0 {
                            continue;
                        }
                        let pos = at.index(n);
                        area.remove(&mut buf, n, pos);
                        model.remove(pos);
                    }
                }
                let n = model.len();
                prop_assert_eq!(area.check(&buf, n), Ok(()));
                prop_assert_eq!(area.free(&buf, n), area.capacity() - cost(area, &model));
                prop_assert!(buf == encoded(area, &model), "page differs from encoding the model");
                for (i, (key, payload)) in model.iter().enumerate() {
                    prop_assert_eq!(area.key(&buf, n, i), key.as_slice());
                    prop_assert_eq!(area.payload(&buf, n, i), payload.as_slice());
                }
            }
        }
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    const AREA: EntryArea = EntryArea::new(16, 2);
    const MAX_ENTRIES: usize = 3;
    const KEY_LENS: usize = 3;

    /// Key lengths of `n` entries, one per base-`KEY_LENS` digit of `shape`. Harnesses loop over
    /// every shape with concrete lengths and leave the key and payload bytes symbolic, since
    /// symbolic lengths make CBMC run out of memory on an 8 KiB page. Cases with two or more
    /// entries also run out of memory in a 15 GB sandbox, so only smaller cases are proofs here.
    fn key_lens(n: usize, shape: usize) -> [usize; MAX_ENTRIES] {
        core::array::from_fn(|i| {
            if i < n {
                shape / KEY_LENS.pow(i as u32) % KEY_LENS
            } else {
                0
            }
        })
    }

    fn shapes(n: usize) -> usize {
        KEY_LENS.pow(n as u32)
    }

    #[derive(Clone, Copy)]
    struct Entry {
        key: [u8; KEY_LENS - 1],
        key_len: usize,
        payload: [u8; 2],
    }

    impl Entry {
        fn any_with_key_len(key_len: usize) -> Self {
            Entry {
                key: kani::any(),
                key_len,
                payload: kani::any(),
            }
        }

        fn key(&self) -> &[u8] {
            &self.key[..self.key_len]
        }
    }

    /// A zeroed page with `entries` laid out by [`EntryArea::put_entry`], the way the page
    /// encoders in `b_plus_page` write them.
    fn encode(entries: &[Entry]) -> [u8; PAGE_SIZE] {
        let mut buf = [0u8; PAGE_SIZE];
        let total: usize = entries
            .iter()
            .map(|e| AREA.entry_cost(e.key_len) - SLOT_ENCODED_BYTES)
            .sum();
        let mut at = PAGE_SIZE - total;
        for (i, e) in entries.iter().enumerate() {
            at = AREA.put_entry(&mut buf, i, at, e.key(), &e.payload);
        }
        assert_eq!(at, PAGE_SIZE);
        buf
    }

    fn any_entries(n: usize, shape: usize) -> [Entry; MAX_ENTRIES] {
        let lens = key_lens(n, shape);
        core::array::from_fn(|i| Entry::any_with_key_len(lens[i]))
    }

    fn assert_entry(buf: &[u8], n: usize, i: usize, expected: &Entry) {
        assert_eq!(AREA.key(buf, n, i), expected.key());
        assert_eq!(AREA.payload(buf, n, i), &expected.payload);
    }

    /// Reads of an area that passed `check` stay inside the page without panicking, and its
    /// used bytes fit within its capacity, so `free` cannot underflow. The page bytes are
    /// unconstrained apart from passing `check`.
    #[kani::proof]
    #[kani::unwind(5)]
    fn reads_of_a_checked_area_stay_in_bounds() {
        let buf: [u8; PAGE_SIZE] = kani::any();
        let n: usize = kani::any();
        kani::assume(n <= MAX_ENTRIES);
        kani::assume(AREA.check(&buf, n).is_ok());
        assert!(AREA.used(&buf, n) <= AREA.capacity());
        let i: usize = kani::any();
        kani::assume(i < n);
        let key = AREA.key(&buf, n, i);
        let payload = AREA.payload(&buf, n, i);
        assert_eq!(payload.len(), AREA.payload_len());
        assert!(key.len() + payload.len() <= AREA.capacity());
    }

    /// `n` entries written in order with `put_entry`, for every key-length shape, form an
    /// area that passes `check` and reads back as the same entries.
    fn encoded_entries_pass_check_and_read_back(n: usize) {
        for shape in 0..shapes(n) {
            let entries = any_entries(n, shape);
            let buf = encode(&entries[..n]);
            assert_eq!(AREA.check(&buf, n), Ok(()));
            for (i, e) in entries[..n].iter().enumerate() {
                assert_entry(&buf, n, i, e);
            }
        }
    }

    /// Inserting at `pos` of an area holding `n` encoded entries, for every key-length shape,
    /// leaves an area that passes `check`, has exactly `entry_cost` fewer free bytes, and is
    /// byte for byte the encoding of the new list of entries.
    fn insert_matches_encoding_the_new_list(n: usize, pos: usize) {
        for shape in 0..shapes(n) {
            for new_len in 0..KEY_LENS {
                let before = any_entries(n, shape);
                let new = Entry::any_with_key_len(new_len);
                let mut buf = encode(&before[..n]);
                let free_before = AREA.free(&buf, n);

                AREA.insert(&mut buf, n, pos, new.key(), &new.payload);

                assert_eq!(AREA.check(&buf, n + 1), Ok(()));
                assert_eq!(
                    AREA.free(&buf, n + 1),
                    free_before - AREA.entry_cost(new_len)
                );
                let mut after = before;
                after.copy_within(pos..n, pos + 1);
                after[pos] = new;
                let byte: usize = kani::any();
                kani::assume(byte < PAGE_SIZE);
                assert_eq!(buf[byte], encode(&after[..=n])[byte]);
            }
        }
    }

    /// One proof per entry count and position, so each CBMC run stays small.
    macro_rules! proofs {
        ($($name:ident => $check:ident($($arg:expr),*);)*) => {
            $(
                #[kani::proof]
                #[kani::unwind(30)]
                fn $name() {
                    $check($($arg),*);
                }
            )*
        };
    }

    proofs! {
        encode_0 => encoded_entries_pass_check_and_read_back(0);
        encode_1 => encoded_entries_pass_check_and_read_back(1);
        insert_into_0_at_0 => insert_matches_encoding_the_new_list(0, 0);
    }
}
