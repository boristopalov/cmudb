mod common;

use std::sync::{Arc, Barrier};
use std::thread;

use cmudb::buffer_pool::{BpmError, BufferPoolManager};
use cmudb::catalog::index_schema::{IndexKey, IndexValue};
use cmudb::index::BPlusTree;
use cmudb::index::Index;
use cmudb::table_heap::RecordId;
use cmudb::testing::pid;
use common::seed::{Stream, stream_rng, with_seed};
use rand::Rng;

fn make_bpm(num_frames: usize) -> (Arc<BufferPoolManager>, tempfile::TempDir) {
    cmudb::testing::bpm(num_frames)
}

fn make_key(k: u8, key_len: usize) -> IndexKey {
    let mut v = vec![0u8; key_len];
    v[0] = k;
    IndexKey(v)
}

fn allocated_page_ids(bpm: &Arc<BufferPoolManager>, max_pages: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for page in 0..max_pages {
        match bpm.read_page(pid(page)) {
            Ok(_guard) => out.push(page),
            Err(BpmError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(e) => panic!("read_page({page}) failed: {e}"),
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
                page_id: pid(i),
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
    with_seed(run_b_plus_tree_soak);
}

fn run_b_plus_tree_soak(seed: u64) {
    let num_frames = 50;
    let key_len = 128;
    let threads = 20;
    let ops_per_thread = 500;
    let key_space = 256;
    let max_pages = 100;

    let (bpm, _tmp) = make_bpm(num_frames);
    let tree = Arc::new(BPlusTree::new("idx".to_string(), key_len as u32, bpm.clone()).unwrap());
    assert_eq!(tree.header_page_id, pid(0));

    let barrier = Arc::new(Barrier::new(threads));
    let mut handles = Vec::with_capacity(threads);

    for t in 0..threads {
        let tree = tree.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            let mut rng = stream_rng(seed, Stream::Worker(t));
            barrier.wait();

            for op_i in 0..ops_per_thread {
                let k = rng.random_range(0..key_space) as u8;
                let key = make_key(k, key_len);

                match rng.random_range(0..100u32) {
                    0..=89 => {
                        let _ = tree.insert(
                            key,
                            IndexValue::IndexValue(RecordId {
                                page_id: pid(t | op_i),
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
