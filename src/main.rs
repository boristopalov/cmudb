use cmudb::create_buffer_pool_manager;
use cmudb::{disk, replacer};

fn main() {
    // TODO: wire up real disk manager, scheduler, and replacer parameters.
    let num_frames: usize = 0;
    let arc_replacer: replacer::ArcReplacer = replacer::ArcReplacer::new(num_frames);
    let disk_manager =
        disk::DiskManager::new("cmudb.data", "cmudb.log").expect("failed to create disk manager");
    let scheduler: disk::DiskScheduler = disk::DiskScheduler::new(disk_manager);

    let _bpm = create_buffer_pool_manager(num_frames, arc_replacer, scheduler);
}
