use log::info;
use std::sync::{Arc, Barrier, mpsc};
use std::thread;

use cmudb::buffer_pool::{BpmError, BufferPoolManager};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::replacer::ArcReplacer;
use env_logger::Env;
use parking_lot::RwLock;
use tempfile::tempdir;

fn init_logger() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug")).try_init();
    });
}

fn make_bpm(num_frames: usize) -> (Arc<BufferPoolManager>, tempfile::TempDir) {
    init_logger();
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("db");
    let log_path = dir.path().join("log");
    let dm = DiskManager::new(&db_path, &log_path).expect("disk manager");
    let scheduler = DiskScheduler::new(dm);
    let replacer = ArcReplacer::new(num_frames);
    (
        Arc::new(create_buffer_pool_manager(num_frames, replacer, scheduler)),
        dir,
    )
}

#[test]
fn write_flush_read_roundtrip() {
    let (bpm, _tmp) = make_bpm(2);

    {
        let mut page = bpm.new_page().expect("new page");
        page.data_mut()[0..4].copy_from_slice(&[1, 2, 3, 4]);
    }
    let read = bpm.read_page(0).expect("read flushed page");
    assert_eq!(&read.data()[0..4], &[1, 2, 3, 4]);
}

#[test]
fn cached_then_io_read_paths() {
    let (bpm, _tmp) = make_bpm(1);

    // Create and cache page 0
    {
        let mut page0 = bpm.new_page().unwrap();
        page0.data_mut()[0..4].copy_from_slice(&[9, 9, 9, 9]);
    }

    // Cached path
    {
        {
            let cached = bpm.read_page(0).unwrap();
            assert_eq!(&cached.data()[0..4], &[9, 9, 9, 9]);
        }
    }

    // Force eviction of page 0 so the next read goes through IO
    {
        let mut page1 = bpm.new_page().unwrap();
        page1.data_mut()[0] = 7;
    }

    // Read page 0 again; this should trigger IO (page 0 was evicted)
    {
        let reread = bpm.read_page(0).unwrap();
        assert_eq!(&reread.data()[0..4], &[9, 9, 9, 9]);
    }
}

#[test]
fn no_free_frame_when_all_pinned() {
    let (bpm, _tmp) = make_bpm(1);
    let err = bpm.read_page(1).unwrap_err();
    assert!(
        matches!(err, BpmError::NoFreeFrame | BpmError::FramePinned),
        "unexpected error: {err}"
    );
}

#[test]
fn page_not_found_when_not_inserted() {
    let (bpm, _tmp) = make_bpm(1);
    bpm.new_page().unwrap(); // this immediately unpins since we don't assign this to anything
    let err = bpm.read_page(1).unwrap_err();
    assert!(matches!(err, BpmError::Io(_)), "unexpected error: {err}");
}

#[test]
fn eviction_flushes_and_clears_mapping() {
    let (bpm, _tmp) = make_bpm(1);

    // Page 0 with data
    {
        let mut page0 = bpm.new_page().unwrap();
        page0.data_mut()[0..3].copy_from_slice(&[5, 6, 7]);
    }

    // Allocate page 1 to force eviction of page 0
    {
        let mut page1 = bpm.new_page().unwrap();
        page1.data_mut()[0] = 42;
    }

    // Page 0 should have been flushed and unmapped; re-read should still see the data
    {
        let reread = bpm.read_page(0).expect("page 0 re-read after eviction");
        assert_eq!(&reread.data()[0..3], &[5, 6, 7]);
    }
}

#[test]
fn io_error_restores_free_frame() {
    let (bpm, _tmp) = make_bpm(1);

    // Page 99 is missing; first read should error
    assert!(matches!(
        bpm.read_page(99).unwrap_err(),
        BpmError::Io(_) | BpmError::ChannelClosed
    ));

    // A subsequent new_page should succeed, proving the reserved frame was returned
    {
        bpm.new_page()
            .expect("frame should be available after IO error");
    }
}

#[test]
fn concurrent_writer_and_blocked_reader_respect_pins() {
    let (bpm, _tmp) = make_bpm(1);

    {
        let mut page0 = bpm.new_page().unwrap();
        page0.data_mut()[0] = 11;
    } // frame_id=0 should be now set to evictable

    let barrier = Arc::new(Barrier::new(2));
    let bpm_writer = bpm.clone();
    let bpm_reader = bpm.clone();
    let writer_barrier = barrier.clone();
    let reader_barrier = barrier.clone();

    let writer = thread::spawn(move || {
        {
            let mut page = bpm_writer.write_page(0).unwrap();
            page.data_mut()[0] = 22;
        }
        writer_barrier.wait(); // allow reader's first attempt while pinned
        writer_barrier.wait(); // reader finished first attempt
        writer_barrier.wait(); // let reader proceed to second attempt
    });

    let reader = thread::spawn(move || {
        reader_barrier.wait(); // wait for writer to pin
        // this sometimes panics on NoFreeFrame for some reason?
        let err = bpm_reader.read_page(1).unwrap_err();
        assert!(
            matches!(err, BpmError::Io(_)),
            "unexpected error while all frames pinned: {err}"
        );
        reader_barrier.wait(); // signal writer first attempt done
        reader_barrier.wait(); // wait for writer to unpin
    });
    writer.join().unwrap();
    reader.join().unwrap();

    info!("expecting to evict frame 0");
    bpm.new_page().unwrap(); // should now evict page 0

    // Verify page 0 changes persisted through eviction
    {
        let reread = bpm.read_page(0).unwrap();
        assert_eq!(reread.data()[0], 22);
    }
}

#[test]
fn duplicate_miss_does_not_leak_frames() {
    let (bpm, _tmp) = make_bpm(1);

    {
        let mut p = bpm.new_page().unwrap();
        p.data_mut()[0] = 99;
    }

    // Evict page 0 so it's only on disk
    {
        let mut p1 = bpm.new_page().unwrap();
        p1.data_mut()[0] = 1;
    }

    // Two concurrent reads for page 0 (miss path)
    let (tx, rx) = mpsc::channel();
    for _ in 0..2 {
        let bpm_clone = bpm.clone();
        let tx = tx.clone();
        let handle = thread::spawn(move || {
            let (_, result) = match bpm_clone.read_page(0) {
                Ok(p) => {
                    let b = p.data()[0];
                    (Some(()), Ok(b))
                }
                Err(e) => (None, Err(e)),
            };
            tx.send(result).unwrap();
        });
        handle.join().unwrap();
    }

    let vals: Vec<_> = rx.iter().take(2).collect();
    for v in vals {
        assert_eq!(v.unwrap(), 99);
    }

    // After concurrent misses, we should still be able to allocate a new page (no leaked frames)
    {
        bpm.new_page().unwrap();
    }
}

#[test]
fn test_eviction() {
    const NUM_FRAMES: usize = 3;
    const NUM_PAGES: usize = 2;
    const NUM_THREADS: usize = 20;

    let (bpm, _tmp) = make_bpm(NUM_FRAMES);
    for _ in 0..NUM_PAGES {
        let val = 123;
        {
            let mut page = bpm.new_page().unwrap();
            page.data_mut()[0] = val;
        }
    }
    let mut handles = Vec::new();
    for _ in 0..NUM_THREADS {
        let bpm_clone = bpm.clone();
        let handle = thread::spawn(move || {
            for i in 0..NUM_PAGES {
                let val = 42;
                {
                    match bpm_clone.write_page(i) {
                        Ok(mut p) => {
                            p.data_mut()[0] = val;
                        }
                        Err(e) => panic!("unexpected error: {e}"),
                    };
                }
            }
        });
        handles.push(handle);
    }
    // Make the test fail if any thread panicked
    for h in handles {
        h.join().unwrap(); // unwrap() will panic if the thread panicked
    }
}

#[test]
fn soak_randomized_workload() {
    // Runs a longer, mixed read/write workload across more pages than frames to force evictions.
    // warning: this puts a lot of stress on the machine
    const NUM_FRAMES: usize = 66;
    const NUM_PAGES: usize = 100;
    const THREADS: usize = 20;
    const OPS_PER_THREAD: usize = 5000;

    let (bpm, _tmp) = make_bpm(NUM_FRAMES);

    // Seed pages on disk so reads/writes have targets to churn on.
    let mut expected = Vec::with_capacity(NUM_PAGES);
    for _ in 0..NUM_PAGES {
        let val = rand::random::<u8>();
        {
            let mut page = bpm.new_page().unwrap();
            page.data_mut()[0] = val;
        }
        expected.push(RwLock::new(val));
    }

    let expected = Arc::new(expected);
    let start_barrier = Arc::new(Barrier::new(THREADS));

    let mut handles = Vec::new();
    for t in 0..THREADS {
        let bpm = bpm.clone();
        let expected = expected.clone();
        let barrier = start_barrier.clone();
        let handle = thread::spawn(move || {
            barrier.wait();
            for _ in 0..OPS_PER_THREAD {
                let page_id = rand::random_range(0.0..(NUM_PAGES as f64 - 1 as f64)) as usize;
                let should_read = rand::random_bool(0.5);
                if should_read {
                    // Read path
                    match bpm.read_page(page_id) {
                        Ok(page) => {
                            let expected_val = *expected[page_id].read();
                            assert_eq!(page.data()[0], expected_val);
                        }
                        Err(BpmError::NoFreeFrame) => continue, // sometimes all frames will be full
                        Err(e) => panic!("unexpected read error: {e}"),
                    }
                } else {
                    // Write path
                    match bpm.write_page(page_id) {
                        Ok(mut page) => {
                            let mut expected_val = expected[page_id].write();
                            let new_val = expected_val.wrapping_add(1).wrapping_add(t as u8);
                            page.data_mut()[0] = new_val;
                            *expected_val = new_val;
                        }
                        Err(BpmError::NoFreeFrame) => continue,
                        Err(e) => panic!("unexpected write error: {e}"),
                    }
                }
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    // After the churn, each page should still match the expected value.
    for pid in 0..NUM_PAGES {
        let expected_val = *expected[pid].read();
        let page_val = {
            let page = bpm.read_page(pid).unwrap();
            page.data()[0]
        };
        assert_eq!(page_val, expected_val, "page {} mismatch", pid);
    }

    // And we should still be able to allocate a new page, proving no leaked pins/frames.
    bpm.new_page().unwrap();
}
