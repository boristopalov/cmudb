//! B+ tree insert, lookup and scan benchmarks. Run with `cargo bench --features bench --bench index`.

use cmudb::buffer_pool::BufferPoolManager;
use cmudb::catalog::index_schema::{IndexKey, IndexValue};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::index::{BPlusTree, Index};
use cmudb::replacer::ArcReplacer;
use cmudb::table_heap::RecordId;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::ops::Bound;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

const POOL_FRAMES: usize = 1024;
const KEY_LEN: u32 = 8;
const INSERT_KEYS: u64 = 10_000;
const LOOKUP_KEYS: u64 = 100_000;
const RANGE_LEN: u64 = 1_000;

fn make_tree() -> (BPlusTree, Arc<BufferPoolManager>, TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk");
    let bpm = Arc::new(create_buffer_pool_manager(
        POOL_FRAMES,
        ArcReplacer::new(POOL_FRAMES),
        DiskScheduler::new(dm),
    ));
    let tree = BPlusTree::new("bench".to_string(), KEY_LEN, bpm.clone()).expect("tree");
    (tree, bpm, dir)
}

fn key(k: u64) -> IndexKey {
    IndexKey(k.to_be_bytes().to_vec())
}

fn rid(k: u64) -> RecordId {
    RecordId {
        page_id: (k / 32) as u32,
        slot_id: (k % 32) as u32,
    }
}

fn value(k: u64) -> IndexValue {
    IndexValue::IndexValue(rid(k))
}

fn rid_of(found: IndexValue) -> RecordId {
    match found {
        IndexValue::IndexValue(rid) => rid,
    }
}

fn insert_all(tree: &BPlusTree, keys: &[u64]) {
    for &k in keys {
        tree.insert(key(k), value(k)).expect("insert");
    }
}

fn count_entries(tree: &BPlusTree, range: (Bound<IndexKey>, Bound<IndexKey>)) -> usize {
    tree.scan(range)
        .expect("scan")
        .try_fold(0usize, |n, r| r.map(|_| n + 1))
        .expect("entry")
}

fn assert_tree_holds(tree: &BPlusTree, n: u64) {
    let entries: Vec<(IndexKey, RecordId)> = tree
        .scan((Bound::Unbounded, Bound::Unbounded))
        .expect("scan")
        .map(|r| r.map(|(k, v)| (k, rid_of(v))))
        .collect::<Result<_, _>>()
        .expect("entry");
    let want: Vec<(IndexKey, RecordId)> = (0..n).map(|k| (key(k), rid(k))).collect();
    assert!(
        entries == want,
        "a full scan returns every inserted key, in order, with its record id"
    );
    for k in (0..n).step_by(97) {
        assert_eq!(
            tree.search(&key(k)).expect("search").map(rid_of),
            Some(rid(k)),
            "key {k} maps to its record id"
        );
    }
}

fn shuffled(n: u64) -> Vec<u64> {
    let mut keys: Vec<u64> = (0..n).collect();
    keys.shuffle(&mut rand::rng());
    keys
}

fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("b_plus_tree");
    group.throughput(Throughput::Elements(INSERT_KEYS));
    let sequential: Vec<u64> = (0..INSERT_KEYS).collect();
    let random = shuffled(INSERT_KEYS);
    for keys in [&sequential, &random] {
        let (tree, _bpm, _dir) = make_tree();
        insert_all(&tree, keys);
        assert_tree_holds(&tree, INSERT_KEYS);
    }
    for (name, keys) in [
        ("insert_seq_10k", &sequential),
        ("insert_random_10k", &random),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                make_tree,
                |(tree, bpm, dir)| {
                    insert_all(&tree, keys);
                    (tree, bpm, dir)
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn lookup_and_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("b_plus_tree");
    let (tree, bpm, _dir) = make_tree();
    insert_all(&tree, &shuffled(LOOKUP_KEYS));
    let probes = shuffled(LOOKUP_KEYS);
    assert_tree_holds(&tree, LOOKUP_KEYS);
    let tree_pages = bpm.allocated_page_count();
    assert!(
        tree_pages < POOL_FRAMES && (0..tree_pages).all(|p| bpm.is_resident(p)),
        "the 100k-key tree ({tree_pages} pages) fits in the {POOL_FRAMES}-frame pool, so lookups and scans never read from disk"
    );
    assert_eq!(
        count_entries(
            &tree,
            (Bound::Included(key(0)), Bound::Excluded(key(RANGE_LEN)))
        ),
        RANGE_LEN as usize,
        "a range scan returns exactly the keys in range"
    );

    group.throughput(Throughput::Elements(1));
    let mut i = 0;
    group.bench_function("search_random_of_100k", |b| {
        b.iter(|| {
            i = (i + 1) % probes.len();
            black_box(tree.search(&key(probes[i])).expect("search"));
        })
    });

    group.throughput(Throughput::Elements(RANGE_LEN));
    let mut start = 0;
    group.bench_function("scan_range_1k_of_100k", |b| {
        b.iter(|| {
            start = (start + RANGE_LEN) % (LOOKUP_KEYS - RANGE_LEN);
            let range = (
                Bound::Included(key(start)),
                Bound::Excluded(key(start + RANGE_LEN)),
            );
            let hits = count_entries(&tree, range);
            black_box(hits);
        })
    });

    group.throughput(Throughput::Elements(LOOKUP_KEYS));
    group.bench_function("scan_full_100k", |b| {
        b.iter(|| {
            let hits = count_entries(&tree, (Bound::Unbounded, Bound::Unbounded));
            black_box(hits);
        })
    });
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = config();
    targets = insert, lookup_and_scan
}
criterion_main!(benches);
