mod common;

use std::sync::{Arc, Barrier};
use std::thread;

use cmudb::buffer_pool::{BpmError, BufferPoolManager};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::replacer::ArcReplacer;
use common::seed::{seed_from_env, stream_rng};
use env_logger::Env;
use parking_lot::RwLock;
use rand::Rng;
use tempfile::tempdir;

fn init_logger() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("warn")).try_init();
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

    {
        let mut page0 = bpm.new_page().unwrap();
        page0.data_mut()[0..4].copy_from_slice(&[9, 9, 9, 9]);
    }

    {
        let cached = bpm.read_page(0).unwrap();
        assert_eq!(&cached.data()[0..4], &[9, 9, 9, 9]);
    }

    let evicts_page_0 = bpm.new_page().unwrap();
    drop(evicts_page_0);

    {
        let reread = bpm.read_page(0).unwrap();
        assert_eq!(&reread.data()[0..4], &[9, 9, 9, 9]);
    }
}

#[test]
fn no_free_frame_when_all_pinned() {
    let (bpm, _tmp) = make_bpm(1);
    let mut pinned = bpm.new_page().expect("pin the only frame");
    pinned.data_mut()[0] = 42;

    let err = bpm.read_page(1).unwrap_err();
    assert!(
        matches!(err, BpmError::NoFreeFrame),
        "unexpected error: {err}"
    );
    assert!(matches!(
        bpm.write_page(1).unwrap_err(),
        BpmError::NoFreeFrame
    ));
    assert!(matches!(bpm.new_page().unwrap_err(), BpmError::NoFreeFrame));

    drop(pinned);
    drop(bpm.new_page().expect("released frame should be evictable"));
    let reread = bpm.read_page(0).expect("evicted page should persist");
    assert_eq!(reread.data()[0], 42);
}

#[test]
fn frame_stays_pinned_until_last_read_guard_drops() {
    let (bpm, _tmp) = make_bpm(1);
    {
        let mut page = bpm.new_page().unwrap();
        page.data_mut()[0] = 7;
    }

    let first = bpm.read_page(0).unwrap();
    let second = bpm.read_page(0).unwrap();
    assert!(matches!(bpm.new_page().unwrap_err(), BpmError::NoFreeFrame));

    drop(first);
    assert!(matches!(bpm.new_page().unwrap_err(), BpmError::NoFreeFrame));
    assert_eq!(second.data()[0], 7);

    drop(second);
    drop(
        bpm.new_page()
            .expect("last read guard should release frame"),
    );
    let reread = bpm.read_page(0).expect("evicted page should persist");
    assert_eq!(reread.data()[0], 7);
}

#[test]
fn page_not_found_when_not_inserted() {
    let (bpm, _tmp) = make_bpm(1);

    let err = bpm.read_page(1).unwrap_err();
    assert!(
        matches!(err, BpmError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound),
        "unexpected error: {err}"
    );

    drop(bpm.new_page().unwrap());
    let err = bpm.read_page(1).unwrap_err();
    assert!(
        matches!(err, BpmError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound),
        "unexpected error: {err}"
    );
}

#[test]
fn eviction_flushes_and_clears_mapping() {
    let (bpm, _tmp) = make_bpm(1);

    {
        let mut page0 = bpm.new_page().unwrap();
        page0.data_mut()[0..3].copy_from_slice(&[5, 6, 7]);
    }

    {
        let mut page1 = bpm.new_page().unwrap();
        page1.data_mut()[0] = 42;
    }

    {
        let reread = bpm.read_page(0).expect("page 0 re-read after eviction");
        assert_eq!(&reread.data()[0..3], &[5, 6, 7]);
    }
}

#[test]
fn io_error_restores_free_frame() {
    let (bpm, _tmp) = make_bpm(1);

    assert!(matches!(
        bpm.read_page(99).unwrap_err(),
        BpmError::Io(_) | BpmError::ChannelClosed
    ));

    drop(
        bpm.new_page()
            .expect("frame should be available after IO error"),
    );
}

#[test]
fn concurrent_writer_and_blocked_reader_respect_pins() {
    let (bpm, _tmp) = make_bpm(1);
    {
        let mut page0 = bpm.new_page().unwrap();
        page0.data_mut()[0] = 11;
    }

    let writer_pinned = Arc::new(Barrier::new(2));
    let reader_tried_while_pinned = Arc::new(Barrier::new(2));
    let writer_unpinned = Arc::new(Barrier::new(2));

    let writer = {
        let bpm = bpm.clone();
        let (pinned, tried, unpinned) = (
            writer_pinned.clone(),
            reader_tried_while_pinned.clone(),
            writer_unpinned.clone(),
        );
        thread::spawn(move || {
            let mut page = bpm.write_page(0).unwrap();
            page.data_mut()[0] = 22;
            pinned.wait();
            tried.wait();
            drop(page);
            unpinned.wait();
        })
    };

    let reader = thread::spawn(move || {
        writer_pinned.wait();
        let err = bpm_read_missing_page(&bpm);
        assert!(
            matches!(err, BpmError::NoFreeFrame),
            "only frame is pinned by the writer, got {err}"
        );
        reader_tried_while_pinned.wait();
        writer_unpinned.wait();
        let err = bpm_read_missing_page(&bpm);
        assert!(
            matches!(err, BpmError::Io(_)),
            "frame is free again, so the miss reaches disk, got {err}"
        );
        bpm
    });
    writer.join().unwrap();
    let bpm = reader.join().unwrap();

    drop(
        bpm.new_page()
            .expect("page 0 is unpinned and can be evicted"),
    );
    let reread = bpm.read_page(0).unwrap();
    assert_eq!(reread.data()[0], 22, "write survives eviction");
}

fn bpm_read_missing_page(bpm: &Arc<BufferPoolManager>) -> BpmError {
    const NEVER_ALLOCATED: usize = 1;
    bpm.read_page(NEVER_ALLOCATED).unwrap_err()
}

#[test]
fn duplicate_miss_does_not_leak_frames() {
    let (bpm, _tmp) = make_bpm(1);

    {
        let mut p = bpm.new_page().unwrap();
        p.data_mut()[0] = 99;
    }

    let evicts_page_0 = bpm.new_page().unwrap();
    drop(evicts_page_0);

    let both_started = Arc::new(Barrier::new(2));
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let bpm = bpm.clone();
            let both_started = both_started.clone();
            thread::spawn(move || {
                both_started.wait();
                loop {
                    match bpm.read_page(0) {
                        Ok(p) => return p.data()[0],
                        Err(BpmError::NoFreeFrame) => thread::yield_now(),
                        Err(e) => panic!("unexpected error on concurrent miss: {e}"),
                    }
                }
            })
        })
        .collect();
    for r in readers {
        assert_eq!(r.join().unwrap(), 99);
    }

    drop(bpm.new_page().expect("concurrent misses leak no frames"));
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
    for h in handles {
        h.join().expect("writer thread panicked");
    }
}

#[test]
fn soak_randomized_workload_with_more_pages_than_frames() {
    const NUM_FRAMES: usize = 66;
    const NUM_PAGES: usize = 100;
    const THREADS: usize = 20;
    const OPS_PER_THREAD: usize = 5000;

    let report = seed_from_env();
    let seed = report.seed();
    let (bpm, _tmp) = make_bpm(NUM_FRAMES);

    let mut setup_rng = stream_rng(seed, THREADS as u64);
    let mut expected = Vec::with_capacity(NUM_PAGES);
    for _ in 0..NUM_PAGES {
        let val = setup_rng.random::<u8>();
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
            let mut rng = stream_rng(seed, t as u64);
            barrier.wait();
            for _ in 0..OPS_PER_THREAD {
                let page_id = rng.random_range(0..NUM_PAGES);
                let should_read = rng.random_bool(0.5);
                if should_read {
                    match bpm.read_page(page_id) {
                        Ok(page) => {
                            let expected_val = *expected[page_id].read();
                            assert_eq!(page.data()[0], expected_val);
                        }
                        Err(BpmError::NoFreeFrame) => continue,
                        Err(e) => panic!("unexpected read error: {e}"),
                    }
                } else {
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

    for pid in 0..NUM_PAGES {
        let expected_val = *expected[pid].read();
        let page_val = {
            let page = bpm.read_page(pid).unwrap();
            page.data()[0]
        };
        assert_eq!(page_val, expected_val, "page {} mismatch", pid);
    }

    drop(
        bpm.new_page()
            .expect("no leaked pins or frames after the churn"),
    );
}

const HOT_PAGES: usize = 6;
const HOT_FRAMES: usize = 4;
const HOT_READERS: usize = 8;
const HOT_READS_PER_THREAD: usize = 20_000;
const HOT_DEADLOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

fn read_hot_page(
    bpm: &Arc<BufferPoolManager>,
    page_id: usize,
) -> Option<cmudb::buffer_pool::PageRef<'_>> {
    match bpm.read_page(page_id) {
        Ok(page) => {
            assert_eq!(
                page.data()[..8],
                (page_id as u64).to_le_bytes(),
                "a pinned frame holds the page it was pinned for"
            );
            Some(page)
        }
        Err(BpmError::NoFreeFrame) => {
            thread::yield_now();
            None
        }
        Err(e) => panic!("unexpected error reading page {page_id}: {e:?}"),
    }
}

#[test]
fn readers_racing_on_hot_pages_never_deadlock_or_evict_a_pinned_frame() {
    let (bpm, _tmp) = make_bpm(HOT_FRAMES);
    for page_id in 0..HOT_PAGES {
        let mut page = bpm.new_page().expect("new page");
        page.data_mut()[..8].copy_from_slice(&(page_id as u64).to_le_bytes());
    }

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    for reader in 0..HOT_READERS {
        let bpm = Arc::clone(&bpm);
        let done_tx = done_tx.clone();
        thread::spawn(move || {
            for i in 0..HOT_READS_PER_THREAD {
                let outer_id = (reader + i) % HOT_PAGES;
                let Some(outer) = read_hot_page(&bpm, outer_id) else {
                    continue;
                };
                let inner = read_hot_page(&bpm, (outer_id + 1) % HOT_PAGES);
                drop(inner);
                drop(outer);
            }
            done_tx.send(()).expect("test still waiting");
        });
    }
    drop(done_tx);
    for _ in 0..HOT_READERS {
        done_rx
            .recv_timeout(HOT_DEADLOCK_TIMEOUT)
            .expect("every reader finishes; a timeout means the buffer pool deadlocked");
    }

    let all_frames_released: Vec<_> = (0..HOT_FRAMES).map(|_| bpm.new_page().map(drop)).collect();
    assert!(
        all_frames_released.iter().all(Result::is_ok),
        "no pin outlives its guard"
    );
}
