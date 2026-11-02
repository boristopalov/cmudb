//! Buffer pool and table heap benchmarks. Run with `cargo bench --features bench --bench storage`.

use cmudb::buffer_pool::BufferPoolManager;
use cmudb::ids::PageId;
use cmudb::table_heap::{RecordId, TableHeap, Tuple};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

const POOL_FRAMES: usize = 64;
const HOT_PAGES: usize = 8;
const COLD_PAGES: usize = POOL_FRAMES * 4;
const TUPLE_BYTES: usize = 100;
const HEAP_ROWS: usize = 10_000;

fn allocate_pages(bpm: &Arc<BufferPoolManager>, count: usize) -> Vec<PageId> {
    (0..count)
        .map(|_| bpm.new_page().expect("new page").page_id())
        .collect()
}

fn tuple(i: usize) -> Tuple {
    Tuple {
        data: vec![(i % 251) as u8; TUPLE_BYTES],
    }
}

fn filled_heap(rows: usize) -> (TableHeap, Vec<RecordId>, Arc<BufferPoolManager>, TempDir) {
    let (bpm, dir) = cmudb::testing::bpm(POOL_FRAMES);
    let heap = TableHeap::new(bpm.clone()).expect("heap");
    let rids = (0..rows)
        .map(|i| heap.insert(&tuple(i)).expect("insert"))
        .collect();
    (heap, rids, bpm, dir)
}

fn count_rows(heap: &TableHeap) -> usize {
    heap.iter()
        .try_fold(0usize, |n, r| r.map(|_| n + 1))
        .expect("scan")
}

fn assert_heap_holds(heap: &TableHeap, rids: &[RecordId]) {
    assert_eq!(
        count_rows(heap),
        rids.len(),
        "a scan sees every inserted row"
    );
    for (i, rid) in rids.iter().enumerate() {
        let (_, got) = heap.get(*rid).expect("get");
        assert_eq!(
            got.data,
            tuple(i).data,
            "row {i} reads back what was inserted"
        );
    }
}

fn buffer_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool");
    group.throughput(Throughput::Elements(1));

    let (bpm, _dir) = cmudb::testing::bpm(POOL_FRAMES);
    let hot = allocate_pages(&bpm, HOT_PAGES);
    assert!(
        hot.iter().all(|&p| bpm.is_resident(p)),
        "read_hit and write_hit only measure hits if every hot page is resident"
    );
    let mut i = 0;
    group.bench_function("read_hit", |b| {
        b.iter(|| {
            i = (i + 1) % hot.len();
            let page = bpm.read_page(hot[i]).expect("read");
            black_box(page.data()[0]);
        })
    });
    group.bench_function("write_hit", |b| {
        b.iter(|| {
            i = (i + 1) % hot.len();
            let mut page = bpm.write_page(hot[i]).expect("write");
            page.data_mut()[0] = i as u8;
        })
    });

    let cold = allocate_pages(&bpm, COLD_PAGES);
    let mut j = 0;
    for _ in 0..2 * cold.len() {
        j = (j + 1) % cold.len();
        assert!(
            !bpm.is_resident(cold[j]),
            "read_miss_evict only measures misses if every read in the cycle misses"
        );
        drop(bpm.read_page(cold[j]).expect("read"));
    }
    group.bench_function("read_miss_evict", |b| {
        b.iter(|| {
            j = (j + 1) % cold.len();
            let page = bpm.read_page(cold[j]).expect("read");
            black_box(page.data()[0]);
        })
    });
    group.finish();
}

fn table_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("table_heap");
    group.throughput(Throughput::Elements(HEAP_ROWS as u64));

    group.bench_function("insert_10k", |b| {
        b.iter_batched(
            || {
                let (bpm, dir) = cmudb::testing::bpm(POOL_FRAMES);
                let heap = TableHeap::new(bpm.clone()).expect("heap");
                (heap, bpm, dir)
            },
            |(heap, bpm, dir)| {
                for i in 0..HEAP_ROWS {
                    heap.insert(&tuple(i)).expect("insert");
                }
                (heap, bpm, dir)
            },
            BatchSize::PerIteration,
        )
    });

    let (heap, rids, _bpm, _dir) = filled_heap(HEAP_ROWS);
    assert_heap_holds(&heap, &rids);
    let pages: HashSet<PageId> = rids.iter().map(|r| r.page_id).collect();
    assert!(
        pages.len() > POOL_FRAMES,
        "the heap spans {} pages, more than the {POOL_FRAMES}-frame pool, so scans and gets evict",
        pages.len()
    );
    group.bench_function("scan_10k", |b| b.iter(|| black_box(count_rows(&heap))));

    let mut shuffled = rids.clone();
    shuffled.shuffle(&mut rand::rng());
    group.throughput(Throughput::Elements(1));
    let mut k = 0;
    group.bench_function("get_random", |b| {
        b.iter(|| {
            k = (k + 1) % shuffled.len();
            black_box(heap.get(shuffled[k]).expect("get"));
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
    targets = buffer_pool, table_heap
}
criterion_main!(benches);
