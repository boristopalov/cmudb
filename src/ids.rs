//! Typed ids for pages, buffer pool frames, tables, indexes and records, so one kind of id
//! cannot be passed where another is expected.

use std::fmt;

/// The id of a page in the data file. Page `id` lives at byte `id * PAGE_SIZE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(u32);

/// A page id that does not fit in the `u32` that page links and record ids store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageIdOutOfRange(pub u64);

impl fmt::Display for PageIdOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "page id {} is past the largest page id, {}",
            self.0,
            PageId::MAX.0
        )
    }
}

impl std::error::Error for PageIdOutOfRange {}

impl PageId {
    /// The largest page id. The `u32` above it encodes "no page" in a page link.
    pub const MAX: PageId = PageId(u32::MAX - 1);

    const NO_PAGE_U32: u32 = u32::MAX;
    const NO_PAGE_U64: u64 = u64::MAX;

    /// The page id `raw`.
    ///
    /// # Panics
    ///
    /// Panics if `raw` is past [`PageId::MAX`].
    pub const fn new(raw: u32) -> PageId {
        assert!(raw <= PageId::MAX.0, "page id past PageId::MAX");
        PageId(raw)
    }

    /// The id as the `u32` that page links and record ids store.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The id as an index, for page-sized slots in the data file.
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Encodes a page link as the `u32` a table page stores; `None` means no page.
    pub fn encode_u32(link: Option<PageId>) -> u32 {
        link.map_or(Self::NO_PAGE_U32, PageId::get)
    }

    /// Decodes a page link stored by [`PageId::encode_u32`].
    pub fn decode_u32(raw: u32) -> Option<PageId> {
        (raw != Self::NO_PAGE_U32).then_some(PageId(raw))
    }

    /// Encodes a page link as the `u64` an index page stores; `None` means no page.
    pub fn encode_u64(link: Option<PageId>) -> u64 {
        link.map_or(Self::NO_PAGE_U64, |id| u64::from(id.0))
    }

    /// Decodes a page link stored by [`PageId::encode_u64`]. Fails for a value that is
    /// neither "no page" nor a page id.
    pub fn decode_u64(raw: u64) -> Result<Option<PageId>, PageIdOutOfRange> {
        if raw == Self::NO_PAGE_U64 {
            return Ok(None);
        }
        PageId::try_from(raw).map(Some)
    }
}

impl TryFrom<u64> for PageId {
    type Error = PageIdOutOfRange;

    fn try_from(raw: u64) -> Result<Self, Self::Error> {
        u32::try_from(raw)
            .ok()
            .filter(|&id| id <= PageId::MAX.0)
            .map(PageId)
            .ok_or(PageIdOutOfRange(raw))
    }
}

impl TryFrom<usize> for PageId {
    type Error = PageIdOutOfRange;

    fn try_from(raw: usize) -> Result<Self, Self::Error> {
        let raw = u64::try_from(raw).unwrap_or(u64::MAX);
        PageId::try_from(raw)
    }
}

impl fmt::Display for PageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The index of a frame in the buffer pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(pub usize);

impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The catalog's id for a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableOid(pub u32);

impl fmt::Display for TableOid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The catalog's id for an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexOid(pub u32);

impl fmt::Display for IndexOid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Where a tuple lives: a table page and a slot on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecordId {
    pub page_id: PageId,
    pub slot_id: u32,
}

impl RecordId {
    /// The id of rows that map back to no table slot.
    pub const RESERVED: RecordId = RecordId {
        page_id: PageId(u32::MAX),
        slot_id: u32::MAX,
    };
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "page_id: {}, slot_id: {}", self.page_id, self.slot_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_links_round_trip_through_both_encodings() {
        for link in [
            None,
            Some(PageId::new(0)),
            Some(PageId::new(17)),
            Some(PageId::MAX),
        ] {
            assert_eq!(PageId::decode_u32(PageId::encode_u32(link)), link);
            assert_eq!(PageId::decode_u64(PageId::encode_u64(link)), Ok(link));
        }
    }

    #[test]
    fn decoding_a_u64_past_the_largest_page_id_fails() {
        let raw = u64::from(u32::MAX);
        assert_eq!(PageId::decode_u64(raw), Err(PageIdOutOfRange(raw)));
        assert_eq!(
            PageId::try_from(usize::MAX),
            Err(PageIdOutOfRange(u64::MAX))
        );
    }

    #[test]
    #[should_panic(expected = "page id past PageId::MAX")]
    fn new_rejects_the_no_page_value() {
        PageId::new(u32::MAX);
    }

    #[test]
    fn ids_display_as_their_number() {
        assert_eq!(PageId::new(3).to_string(), "3");
        assert_eq!(FrameId(4).to_string(), "4");
        assert_eq!(TableOid(5).to_string(), "5");
        assert_eq!(IndexOid(6).to_string(), "6");
        assert_eq!(
            RecordId {
                page_id: PageId::new(1),
                slot_id: 2
            }
            .to_string(),
            "page_id: 1, slot_id: 2"
        );
    }
}
