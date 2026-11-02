//! Fixtures shared by unit and integration tests. Built under `cfg(test)` or with the
//! `testing` feature, which the crate's own dev-dependency turns on for `tests/` and
//! `benches/`.
//!
//! Each fixture returns the `TempDir` holding its files; keep it alive for as long as the
//! pool is used, since dropping it deletes the files.

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::Catalog;
use crate::database::scratch_pool;
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
