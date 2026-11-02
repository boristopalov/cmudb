//! Fixtures shared by unit and integration tests. Built under `cfg(test)` or with the
//! `testing` feature, which the crate's own dev-dependency turns on for `tests/` and
//! `benches/`.
//!
//! Each fixture returns the `TempDir` holding its files; keep it alive for as long as the
//! pool is used, since dropping it deletes the files.

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::Catalog;
use crate::database::scratch_pool;
use crate::ids::PageId;
use std::sync::Arc;
use tempfile::TempDir;

/// A buffer pool of `frames` frames over files in a new scratch directory.
///
/// Panics if the directory or its files cannot be created.
pub fn bpm(frames: usize) -> (Arc<BufferPoolManager>, TempDir) {
    scratch_pool(frames).expect("create a temporary buffer pool")
}

/// An empty catalog over a new [`bpm`] of `frames` frames, and that pool.
///
/// Panics if the directory or its files cannot be created.
pub fn catalog(frames: usize) -> (Catalog, Arc<BufferPoolManager>, TempDir) {
    let (bpm, dir) = bpm(frames);
    (Catalog::new(bpm.clone()), bpm, dir)
}

/// A number or a [`PageId`] that a test names a page by.
pub trait IntoPageId {
    /// The page id. Panics if the number is negative or past [`PageId::MAX`].
    fn into_page_id(self) -> PageId;
}

impl IntoPageId for PageId {
    fn into_page_id(self) -> PageId {
        self
    }
}

macro_rules! into_page_id_for_integers {
    ($($int:ty),*) => {$(
        impl IntoPageId for $int {
            fn into_page_id(self) -> PageId {
                u64::try_from(i128::from(self))
                    .ok()
                    .and_then(|n| PageId::try_from(n).ok())
                    .unwrap_or_else(|| panic!("page {self} is negative or past PageId::MAX"))
            }
        }
    )*};
}

into_page_id_for_integers!(i32, u32, u64);

impl IntoPageId for usize {
    fn into_page_id(self) -> PageId {
        PageId::try_from(self).expect("a page id up to PageId::MAX")
    }
}

/// Page `n`, for tests that name pages by number.
///
/// Panics if `n` is negative or past [`PageId::MAX`].
pub fn pid(n: impl IntoPageId) -> PageId {
    n.into_page_id()
}
