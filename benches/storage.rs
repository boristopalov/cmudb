//! Buffer pool and table heap benchmarks. Run with `cargo bench --features bench --bench storage`.

use cmudb::buffer_pool::BufferPoolManager;
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::replacer::ArcReplacer;
use cmudb::table_heap::{RecordId, TableHeap, Tuple};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

const POOL_FRAMES: usize = 64;
const HOT_PAGES: usize = 8;
const COLD_PAGES: usize = POOL_FRAMES * 4;
const TUPLE_BYTES: usize = 100;
const HEAP_ROWS: usize = 10_000;

fn make_bpm(frames: usize) -> (Arc<BufferPoolManager>, TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk");
    let bpm = Arc::new(create_buffer_pool_manager(
        frames,
        ArcReplacer::new(frames),
        DiskScheduler::new(dm),
    ));
    (bpm, dir)
}

fn allocate_pages(bpm: &Arc<BufferPoolManager>, count: usize) -> Vec<usize> {
    (0..count)
        .map(|_| {
            bpm.new_page()
                .expect("new page")
                .page_id()
                .expect("page id")
        })
        .collect()
}

fn tuple(i: usize) -> Tuple {
    Tuple {
        data: vec![(i % 251) as u8; TUPLE_BYTES],
    }
}

fn filled_heap(rows: usize) -> (TableHeap, Vec<RecordId>, Arc<BufferPoolManager>, TempDir) {
    let (bpm, dir) = make_bpm(POOL_FRAMES);
    let heap = TableHeap::new(bpm.clone()).expect("heap");
    let rids = (0..rows)
        .map(|i| heap.insert(&tuple(i)).expect("insert"))
        .collect();
    (heap, rids, bpm, dir)
}

fn buffer_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool");
    group.throughput(Throughput::Elements(1));

    let (bpm, _dir) = make_bpm(POOL_FRAMES);
    let hot = allocate_pages(&bpm, HOT_PAGES);
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
                let (bpm, dir) = make_bpm(POOL_FRAMES);
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
    group.bench_function("scan_10k", |b| {
        b.iter(|| {
            black_box(
                heap.iter()
                    .try_fold(0usize, |n, r| r.map(|_| n + 1))
                    .expect("scan"),
            )
        })
    });

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
