use std::sync::{Arc, Barrier};
use std::thread;

use cmudb::buffer_pool::{BpmError, BufferPoolManager};
use cmudb::catalog::index_schema::{IndexKey, IndexValue};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::index::BPlusTree;
use cmudb::index::Index;
use cmudb::replacer::ArcReplacer;
use cmudb::table_heap::RecordId;
use env_logger::Env;
use rand::Rng;
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

fn make_key(k: u8, key_len: usize) -> IndexKey {
    let mut v = vec![0u8; key_len];
    v[0] = k;
    IndexKey(v)
}

fn allocated_page_ids(bpm: &Arc<BufferPoolManager>, max_pages: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for pid in 0..max_pages {
        match bpm.read_page(pid) {
            Ok(_guard) => out.push(pid),
            Err(BpmError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(e) => panic!("read_page({pid}) failed: {e}"),
        }
    }
    out
}

#[test]
fn b_plus_tree_single_thread() {
    let num_frames = 500;
    let key_len = 20;
    let inserts = 500;
    let removes = 500;

    let (bpm, _tmp) = make_bpm(num_frames);
    let tree = BPlusTree::new("idx".to_string(), key_len as u32, bpm.clone()).unwrap();

    for i in 0..inserts {
        let k = (i % 200) as u8;
        let key = make_key(k, key_len);
        let _ = tree.insert(
            key,
            IndexValue::IndexValue(RecordId {
                page_id: i,
                slot_id: 0,
            }),
        );
    }

    for i in 0..removes {
        let k = (i % 200) as u8;
        let key = make_key(k, key_len);
        let _ = tree.remove(key);
    }
}

#[test]
fn b_plus_tree_soak_concurrent() {
    let num_frames = 50;
    let key_len = 128;
    let threads = 20;
    let ops_per_thread = 500;
    let key_space = 256;
    let max_pages = 100;

    let (bpm, _tmp) = make_bpm(num_frames);
    let tree = Arc::new(BPlusTree::new("idx".to_string(), key_len as u32, bpm.clone()).unwrap());
    assert_eq!(tree.header_page_id, 0);

    let barrier = Arc::new(Barrier::new(threads));
    let mut handles = Vec::with_capacity(threads);

    for t in 0..threads {
        let tree = tree.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            let mut rng = rand::rng();
            barrier.wait();

            for op_i in 0..ops_per_thread {
                let k = rng.random_range(0..key_space) as u8;
                let key = make_key(k, key_len);

                // 30% chance to insert a key
                // 10% chance to remove a key
                // 60% chance to search for a key
                match rng.random_range(0..100u32) {
                    0..=89 => {
                        let _ = tree.insert(
                            key,
                            IndexValue::IndexValue(RecordId {
                                page_id: (t | op_i) as u32,
                                slot_id: op_i as u32,
                            }),
                        );
                    }
                    90..=99 => {
                        let _ = tree.remove(key);
                    }
                    _ => {
                        let _ = tree.search(&key);
                    }
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread");
    }

    let pids = allocated_page_ids(&bpm, max_pages);
    assert!(!pids.is_empty(), "expected at least the header page");
}
