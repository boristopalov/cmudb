pub mod buf;
pub mod buffer_pool;
pub mod catalog;
pub mod disk;
pub mod index;
pub mod replacer;
pub mod table_heap;

/// Convenience helper to construct a buffer pool manager with caller-provided components.
pub fn create_buffer_pool_manager(
    num_frames: usize,
    replacer: replacer::ArcReplacer,
    scheduler: disk::DiskScheduler,
) -> buffer_pool::BufferPoolManager {
    buffer_pool::BufferPoolManager::new(num_frames, replacer, scheduler)
}
