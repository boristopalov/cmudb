pub mod buf;
pub mod buffer_pool;
pub mod catalog;
pub mod concurrency;
pub mod database;
pub mod disk;
pub mod index;
pub mod parser;
pub mod planner;
pub mod processor;
pub mod replacer;
pub mod table_heap;
pub mod table_page;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use database::{Database, DdlOp, DmlOp, QueryError, StatementResult, TxnOp};

/// Convenience helper to construct a buffer pool manager with caller-provided components.
pub fn create_buffer_pool_manager(
    num_frames: usize,
    replacer: replacer::ArcReplacer,
    scheduler: disk::DiskScheduler,
) -> buffer_pool::BufferPoolManager {
    buffer_pool::BufferPoolManager::new(num_frames, replacer, scheduler)
}
