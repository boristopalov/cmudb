use cmudb::buffer_pool::BufferPoolManager;
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler, PAGE_SIZE};
use cmudb::replacer::ArcReplacer;
use cmudb::table_heap::{RecordId, TableHeap, TableHeapIterator, Tuple};
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use tempfile::{TempDir, tempdir};

fn make_heap(frames: usize) -> (TableHeap, Arc<BufferPoolManager>, TempDir) {
    let dir = tempdir().unwrap();
    let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).unwrap();
    let bpm = Arc::new(create_buffer_pool_manager(
        frames,
        ArcReplacer::new(frames),
        DiskScheduler::new(dm),
    ));
    // The table need not start at page zero.
    drop(bpm.new_page().unwrap());
    let heap = TableHeap::new(bpm.clone()).unwrap();
    (heap, bpm, dir)
}

fn rows(iter: TableHeapIterator) -> Vec<(RecordId, Vec<u8>)> {
    iter.map(|r| {
        let (tuple, rid) = r.unwrap();
        (rid, tuple.data)
    })
    .collect()
}

#[test]
fn growth_preserves_rids_and_order_with_interleaved_pages_and_eviction() {
    let (heap, bpm, _dir) = make_heap(1);
    let mut expected = Vec::new();
    let mut unrelated = Vec::new();
    for i in 0..40u8 {
        let tuple = Tuple {
            data: vec![i; 1000],
        };
        let rid = heap.insert(&tuple).unwrap();
        expected.push((rid, tuple.data));
        let mut other = bpm.new_page().unwrap();
        unrelated.push(other.page_id().unwrap());
        other.data_mut().fill(0xa5);
    }
    assert!(
        expected
            .windows(2)
            .any(|w| w[0].0.page_id != w[1].0.page_id)
    );
    for (rid, data) in &expected {
        assert_eq!(heap.get(*rid).unwrap().1.data, *data);
    }
    assert_eq!(rows(heap.iter()), expected);
    for id in unrelated {
        assert!(bpm.read_page(id).unwrap().data().iter().all(|b| *b == 0xa5));
    }
}

#[test]
fn exact_fit_accounts_for_header_and_each_slot_including_empty_tuples() {
    const MAX_PAYLOAD: usize = PAGE_SIZE - 12 - 9;
    for spare in [0, 5, 9] {
        let (heap, _bpm, _dir) = make_heap(1);
        let tuple = Tuple {
            data: vec![0x7b; MAX_PAYLOAD - spare],
        };
        let first = heap.insert(&tuple).unwrap();
        let empty = heap.insert(&Tuple { data: vec![] }).unwrap();
        assert_eq!(first.page_id == empty.page_id, spare == 9);
        assert_eq!(heap.get(first).unwrap().1.data, tuple.data);
        assert!(heap.get(empty).unwrap().1.data.is_empty());
        assert_eq!(rows(heap.iter()).len(), 2);
    }

    let (heap, _bpm, _dir) = make_heap(1);
    let slots = (PAGE_SIZE - 12) / 9;
    for slot in 0..slots {
        let rid = heap.insert(&Tuple { data: vec![] }).unwrap();
        assert_eq!(rid.page_id as usize, heap.first_page_id);
        assert_eq!(rid.slot_id as usize, slot);
    }
    let next = heap.insert(&Tuple { data: vec![] }).unwrap();
    assert_ne!(next.page_id as usize, heap.first_page_id);
    assert_eq!(next.slot_id, 0);
    assert_eq!(rows(heap.iter()).len(), slots + 1);
}

#[test]
fn oversized_insert_and_update_leave_page_bytes_and_original_row_unchanged() {
    let (heap, bpm, _dir) = make_heap(1);
    let original = Tuple {
        data: b"keep me".to_vec(),
    };
    let rid = heap.insert(&original).unwrap();
    let before = bpm.read_page(heap.first_page_id).unwrap().data().to_vec();
    let oversized = Tuple {
        data: vec![0xff; PAGE_SIZE - 12 - 9 + 1],
    };
    assert!(heap.insert(&oversized).is_err());
    assert!(heap.update(rid, &oversized).is_err());
    assert_eq!(heap.last_page_id(), heap.first_page_id);
    assert_eq!(bpm.read_page(heap.first_page_id).unwrap().data(), before);
    assert_eq!(heap.get(rid).unwrap().1.data, original.data);
    assert_eq!(rows(heap.iter()), vec![(rid, original.data)]);
    assert_eq!(heap.insert(&Tuple { data: vec![1] }).unwrap().slot_id, 1);
}

#[test]
fn unavailable_frame_during_insert_or_update_preserves_existing_rows() {
    let (heap, bpm, _dir) = make_heap(1);
    let original = Tuple {
        data: vec![7; PAGE_SIZE - 12 - 9],
    };
    let rid = heap.insert(&original).unwrap();
    // Evict the table and pin the only frame, making insertion fail with a
    // buffer-pool error rather than an oversized-tuple rejection.
    let pinned = bpm.new_page().unwrap();
    let replacement = Tuple { data: vec![8; 100] };
    assert!(heap.insert(&replacement).is_err());
    assert!(heap.update(rid, &replacement).is_err());
    drop(pinned);
    assert_eq!(heap.last_page_id(), heap.first_page_id);
    assert_eq!(heap.get(rid).unwrap().1.data, original.data);
    let (new_rid, _) = heap.update(rid, &replacement).unwrap();
    assert!(heap.get(rid).is_err());
    assert_eq!(heap.get(new_rid).unwrap().1.data, replacement.data);
    assert_eq!(rows(heap.iter()), vec![(new_rid, replacement.data)]);
}

#[test]
fn scans_exclude_later_tail_slots_and_pages_including_initially_empty_heaps() {
    for original_count in [0, 3, 12] {
        let (heap, _bpm, _dir) = make_heap(1);
        let expected: Vec<_> = (0..original_count)
            .map(|i| {
                let tuple = Tuple {
                    data: vec![i; 1000],
                };
                (heap.insert(&tuple).unwrap(), tuple.data)
            })
            .collect();
        let untouched = heap.iter();
        let mut started = heap.iter();
        if original_count > 0 {
            let (tuple, rid) = started.next().unwrap().unwrap();
            assert_eq!((rid, tuple.data), expected[0]);
        } else {
            assert!(started.next().is_none());
        }
        for _ in 0..20 {
            heap.insert(&Tuple {
                data: vec![255; 1000],
            })
            .unwrap();
        }
        assert_eq!(rows(untouched), expected);
        let remaining = rows(started);
        assert_eq!(remaining, expected[usize::from(original_count > 0)..]);
        assert_eq!(rows(heap.iter()).len(), original_count as usize + 20);
    }
}

#[test]
fn concurrent_inserts_and_scan_capture_are_serialized_across_page_boundaries() {
    const THREADS: usize = 6;
    const PER_THREAD: usize = 32;
    let (heap, _bpm, _dir) = make_heap(1);
    let heap = Arc::new(heap);
    let barrier = Arc::new(Barrier::new(THREADS));
    let (tx, rx) = mpsc::channel();
    let handles: Vec<_> = (0..THREADS)
        .map(|worker| {
            let heap = heap.clone();
            let barrier = barrier.clone();
            let tx = tx.clone();
            thread::spawn(move || {
                barrier.wait();
                let mut inserted = Vec::new();
                for seq in 0..PER_THREAD {
                    let mut data = vec![worker as u8; 1024];
                    data[..8].copy_from_slice(&(seq as u64).to_le_bytes());
                    let rid = heap.insert(&Tuple { data: data.clone() }).unwrap();
                    inserted.push((rid, data));
                    tx.send(()).unwrap();
                }
                inserted
            })
        })
        .collect();
    drop(tx);
    let mut scans = Vec::new();
    for (i, ()) in rx.into_iter().enumerate() {
        if i % 12 == 0 {
            scans.push(heap.iter());
        }
    }
    let mut expected = Vec::new();
    for handle in handles {
        let inserted = handle.join().unwrap();
        assert!(
            inserted
                .windows(2)
                .all(|w| { (w[0].0.page_id, w[0].0.slot_id) < (w[1].0.page_id, w[1].0.slot_id) })
        );
        expected.extend(inserted);
    }
    expected.sort_by_key(|(rid, _)| (rid.page_id, rid.slot_id));
    assert_eq!(expected.len(), THREADS * PER_THREAD);
    assert_eq!(rows(heap.iter()), expected);
    for scan in scans {
        let captured = rows(scan);
        assert!(!captured.is_empty());
        assert_eq!(captured, expected[..captured.len()]);
    }
    for (rid, data) in expected {
        assert_eq!(heap.get(rid).unwrap().1.data, data);
    }
}

#[test]
fn updating_invalid_or_deleted_rid_does_not_append_a_replacement() {
    let (heap, _bpm, _dir) = make_heap(1);
    let tuple = Tuple { data: vec![1] };
    let rid = heap.insert(&tuple).unwrap();
    assert!(
        heap.update(RecordId { slot_id: 10, ..rid }, &tuple)
            .is_err()
    );
    assert_eq!(rows(heap.iter()).len(), 1);
    heap.delete(&rid).unwrap();
    assert!(heap.update(rid, &tuple).is_err());
    assert!(rows(heap.iter()).is_empty());
}
