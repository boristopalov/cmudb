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
        put_u16_le(
            buf,
            slot,
            u16::try_from(at).expect("a page offset fits in u16"),
        );
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
            put_u16_le(buf, self.slot_offset(i), moved as u16);
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
            put_u16_le(buf, self.slot_offset(i), moved as u16);
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
