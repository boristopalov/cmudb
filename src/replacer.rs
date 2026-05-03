use std::process::exit;

use log::{error, info, warn};
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArcList {
    Mru,
    Mfu,
    GhostMru,
    GhostMfu,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Node {
    page_id: usize,
    frame_id: Option<usize>, // None if ghost frame (TODO: i don't think this is needed)
    evictable: bool,
    list: ArcList,
    next: Option<usize>,
    prev: Option<usize>,
}

#[derive(Default, Debug)]
struct ListHead {
    head: Option<usize>, // MRU
    tail: Option<usize>, // LRU
    length: usize,
}

/// Adaptive Replacement Cache (ARC)
///
/// ARC has:
///      - two lists to track cached pages
///      - two lists to track recently evicted pages
///      - target size that adapts to the workload
#[derive(Default, Debug)]
pub struct ArcReplacer {
    /// node ids used as indices
    /// TODO: should be ref, not Node?>
    nodes: Vec<Node>,

    page_indices: HashMap<usize, usize>,

    frame_indices: HashMap<usize, usize>,

    /// mru_list tracks frames and their corresponding pages that were recently accessed EXACTLY once
    /// A hit on mru_list moves the frame from mru_list to mfu_list
    /// size of mru_list and mfu_list is dynamic
    mru_list: ListHead,

    /// mfu_list tracks frames + pages that were recently accessed more than one time
    /// ArcReplace keeps a most-frequently-accessed ordering, so the head of mfu_list is always the most recently accessed page,
    /// and the tail is always the least recently accessed.
    /// Conesequently, when we evict, we evict the least recently accessed page.
    mfu_list: ListHead,

    /// ghost lists track pages that are no longer in the buffer pool but were recently evicted
    /// They are used as a weighting mechanism for the mru_target_size
    /// Lets us know: "what would have happened with a different target size"

    /// mru_ghost_list maintains a list of pages that were evicted from the mru_list,
    /// i.e. recent one-off pages.
    /// A hit in the MRU ghost lists implies that recency was under-accounted for.
    /// Increase mru_target_size on a hit on this list.
    /// sizes of ghost list are fixed (capacity == cache size)
    mru_ghost_list: ListHead,

    /// mfu_ghost_list maintains a list of pages that were evicted from the mfu_list,
    /// i.e. old but frequently accessed pages.
    /// A hit in the MFU ghost list implies that frequency was under-accounted for
    /// Decrease mru_target_size on a hit on this list
    mfu_ghost_list: ListHead,

    /// target size for the MRU list
    /// note: actual MRU list size does not have to be equal to mru_target_size
    mru_target_size: usize,

    /// total number of frames (should be equal to the capacity of the buffer pool)
    capacity: usize,

    /// current number of evictable frames
    curr_size: usize,
}

impl ArcReplacer {
    pub fn new(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            page_indices: HashMap::new(),
            frame_indices: HashMap::new(),
            mru_list: ListHead::default(),
            mfu_list: ListHead::default(),
            mru_ghost_list: ListHead::default(),
            mfu_ghost_list: ListHead::default(),
            mru_target_size: 0,
            capacity: cap,
            curr_size: 0,
        }
    }
    /// Returns the number of evictable frames
    pub fn size(&self) -> usize {
        self.curr_size
    }

    /// Sets whether a frame is evictable or not.
    /// When the pin count of a page hits 0, its corresponding frame should be marked as evictable.
    /// The Buffer pool manager manages pin counts.
    pub fn set_evictable(&mut self, frame_id: usize, evictable: bool) {
        let frame_idx = self.frame_indices.get(&frame_id);
        match frame_idx {
            Some(idx) => {
                let node = &mut self.nodes[*idx];
                let was_evictable = node.evictable;
                node.evictable = evictable;
                if evictable && !was_evictable {
                    self.curr_size += 1;
                } else if !evictable && was_evictable {
                    self.curr_size -= 1;
                }
                info!(
                    "set_evictable frame_id={frame_id} evictable={evictable} list={:?}",
                    node.list
                );
            }
            None => {
                warn!("set_evictable called for unknown frame_id={frame_id}");
                return;
            }
        }
    }

    fn is_full(&self) -> bool {
        self.mru_list.length + self.mfu_list.length >= self.capacity
    }

    /// Records that the given page has been access at the current timestamp, in the given frame.
    /// Should be called after a page has been pinned to a frame by a BufferPoolManager.
    pub fn record_access(&mut self, frame_id: usize, page_id: usize) {
        let page_idx = self.page_indices.get(&page_id).copied();
        if let Some(idx) = page_idx {
            match self.nodes[idx].list {
                ArcList::Mfu | ArcList::Mru => {
                    info!("record_access page_id={page_id} frame_id={frame_id} -> front of MFU");
                    let node = &mut self.nodes[idx];
                    node.frame_id = Some(frame_id);
                    self.frame_indices.insert(frame_id, idx);
                    self.move_to_front(idx, ArcList::Mfu);
                }
                ArcList::GhostMru => {
                    let prev_target = self.mru_target_size;
                    let delta = if self.mru_ghost_list.length >= self.mfu_ghost_list.length {
                        1
                    } else {
                        self.mfu_ghost_list.length / self.mru_ghost_list.length
                    };
                    self.mru_target_size = (self.mru_target_size + delta).min(self.capacity);
                    info!(
                        "record_access ghost MRU hit page_id={page_id} frame_id={frame_id} mru_target_size={prev_target}->{}, mru_len={}, mfu_len={} -> MFU",
                        self.mru_target_size, self.mru_list.length, self.mfu_list.length
                    );
                    if self.is_full() && self.evict().is_none() {
                        warn!("record_access ghost MRU hit but no evictable frames");
                        return;
                    }
                    let node = &mut self.nodes[idx];
                    node.frame_id = Some(frame_id);
                    node.evictable = false;
                    self.frame_indices.insert(frame_id, idx);
                    self.move_to_front(idx, ArcList::Mfu);
                }
                ArcList::GhostMfu => {
                    let prev_target = self.mru_target_size;
                    let delta = if self.mfu_ghost_list.length >= self.mru_ghost_list.length {
                        1
                    } else {
                        self.mru_ghost_list.length / self.mfu_ghost_list.length
                    };
                    self.mru_target_size = self.mru_target_size.saturating_sub(delta);
                    info!(
                        "record_access ghost MFU hit page_id={page_id} frame_id={frame_id} mru_target_size={prev_target}->{}, mru_len={}, mfu_len={}",
                        self.mru_target_size, self.mru_list.length, self.mfu_list.length
                    );
                    if self.is_full() && self.evict().is_none() {
                        warn!("record_access ghost MFU hit but no evictable frames");
                        return;
                    }
                    let node = &mut self.nodes[idx];
                    node.frame_id = Some(frame_id);
                    node.evictable = false;
                    self.frame_indices.insert(frame_id, idx);
                    self.move_to_front(idx, ArcList::Mfu);
                }
            }
            return;
        }

        // page is not in the replacer and also missed in both ghost lists
        info!(
            "record_access new page_id={page_id} frame_id={frame_id} -> insert into MRU (ghost miss)"
        );

        if self.mru_list.length + self.mru_ghost_list.length == self.capacity {
            if !self.trim_ghost_tail(ArcList::GhostMru) {
                error!("cannot insert new node: MRU ghost list unexpectedly empty");
                return;
            }
        } else {
            if self.mru_list.length + self.mru_ghost_list.length > self.capacity {
                error!(
                    "mru_list.length + mru_ghost_list.length should not be > capacity. exiting..."
                );
                exit(1);
            }
            if self.mru_list.length
                + self.mru_ghost_list.length
                + self.mfu_list.length
                + self.mfu_ghost_list.length
                == 2 * self.capacity
            {
                if !self.trim_ghost_tail(ArcList::GhostMfu) {
                    error!("cannot insert new node: MFU ghost list unexpectedly empty");
                    return;
                }
            }
        }

        let new_node = Node {
            page_id,
            frame_id: Some(frame_id),
            evictable: false,
            list: ArcList::Mru,
            next: None,
            prev: None,
        };
        self.nodes.push(new_node);
        let idx = self.nodes.len() - 1;
        self.page_indices.insert(page_id, idx);
        self.frame_indices.insert(frame_id, idx);
        self.push(idx, ArcList::Mru);
    }

    /// Looks for a victim frame and evicts it.
    ///
    /// Returns ID of the evicted frame, or None if no evictable frames are available
    pub fn evict(&mut self) -> Option<usize> {
        info!(
            "evict: attempting eviction mru_target_size={} mru_len={} mfu_len={}",
            self.mru_target_size, self.mru_list.length, self.mfu_list.length
        );
        if self.mru_list.length >= self.mru_target_size {
            match self.evict_from_list(ArcList::Mru) {
                Some(id) => return Some(id),
                None => return self.evict_from_list(ArcList::Mfu),
            }
        } else if self.mru_list.length < self.mru_target_size {
            match self.evict_from_list(ArcList::Mfu) {
                Some(id) => return Some(id),
                None => return self.evict_from_list(ArcList::Mru),
            }
        }
        warn!("evict: no evictable frames available");
        None
    }

    fn evict_from_list(&mut self, list: ArcList) -> Option<usize> {
        let ghost_list = if list == ArcList::Mru {
            ArcList::GhostMru
        } else {
            ArcList::GhostMfu
        };
        if matches!(list, ArcList::GhostMru | ArcList::GhostMfu) {
            warn!("evict_from_list called on ghost list {:?}", list);
            return None;
        }

        let mut current = match list {
            ArcList::Mru => self.mru_list.tail,
            ArcList::Mfu => self.mfu_list.tail,
            ArcList::GhostMfu => panic!("called evict_from_list on GhostMfu"),
            ArcList::GhostMru => panic!("called evict_from_list on GhostMru"),
        }?;

        loop {
            if !self.nodes[current].evictable {
                let prev = self.nodes[current].prev;
                let node = &self.nodes[current];
                warn!(
                    "evict_from_list: frame not evictable list={:?} page_id={} frame_id={:?}",
                    list, node.page_id, node.frame_id
                );
                match prev {
                    Some(p) => {
                        current = p;
                        continue;
                    }
                    None => {
                        warn!(
                            "evict_from_list: no evictable frames found in list {:?}",
                            list
                        );
                        return None;
                    }
                }
            }

            let page_id = self.nodes[current].page_id;
            let frame_id = self.nodes[current].frame_id.take();

            info!(
                "evict_from_list: evicted frame_id={:?} page_id={page_id} from list={:?} -> {:?}",
                frame_id, list, ghost_list
            );

            if let Some(fid) = frame_id {
                self.frame_indices.remove(&fid);
                self.move_to_front(current, ghost_list);
                self.curr_size -= 1;
                return Some(fid);
            } else {
                warn!(
                    "evict_from_list: node missing frame_id list={:?} page_id={}",
                    list, page_id
                );
                return None;
            }
        }
    }

    fn trim_ghost_tail(&mut self, ghost_list: ArcList) -> bool {
        let tail = match ghost_list {
            ArcList::GhostMru => self.mru_ghost_list.tail,
            ArcList::GhostMfu => self.mfu_ghost_list.tail,
            _ => {
                warn!("trim_ghost_tail called on non-ghost list {:?}", ghost_list);
                return false;
            }
        };

        let Some(idx) = tail else {
            warn!(
                "trim_ghost_tail: ghost list empty list={:?} mru_ghost_len={} mfu_ghost_len={}",
                ghost_list, self.mru_ghost_list.length, self.mfu_ghost_list.length
            );
            return false;
        };

        let page_id = self.nodes[idx].page_id;
        info!(
            "trim_ghost_tail: drop ghost page_id={page_id} from list={:?}",
            ghost_list
        );
        self.detach(idx);
        self.page_indices.remove(&page_id);
        true
    }

    fn list_mut(&mut self, list: ArcList) -> &mut ListHead {
        match list {
            ArcList::Mfu => &mut self.mfu_list,
            ArcList::Mru => &mut self.mru_list,
            ArcList::GhostMru => &mut self.mru_ghost_list,
            ArcList::GhostMfu => &mut self.mfu_ghost_list,
        }
    }

    /// detach removes a node
    fn detach(&mut self, node_idx: usize) {
        // Pull the fields we need out of the node before mutably borrowing self again.
        let (list, next, prev) = {
            let node = &self.nodes[node_idx];
            (node.list, node.next, node.prev)
        };

        {
            let list = self.list_mut(list);

            if list.head == Some(node_idx) {
                list.head = next;
            }
            if list.tail == Some(node_idx) {
                list.tail = prev;
            }

            list.length -= 1;
        }

        if let Some(p) = prev {
            self.nodes[p].next = next;
        }
        if let Some(n) = next {
            self.nodes[n].prev = prev;
        }

        self.nodes[node_idx].prev = None;
        self.nodes[node_idx].next = None;
    }

    /// push to the front of one of the tracking lists based on the list provided
    fn push(&mut self, node_idx: usize, list: ArcList) {
        let old_head = {
            let list = self.list_mut(list);
            let old_head = list.head;
            if old_head.is_none() {
                // list was empty, so tail will also become the new node
                list.tail = Some(node_idx);
            }
            list.head = Some(node_idx);
            list.length += 1;
            old_head
        };

        self.nodes[node_idx].list = list;
        self.nodes[node_idx].prev = None;
        self.nodes[node_idx].next = old_head;

        if let Some(h) = old_head {
            self.nodes[h].prev = Some(node_idx);
        }
    }

    /// Move an existing node to the front of the target list, keeping indices intact.
    fn move_to_front(&mut self, node_idx: usize, list: ArcList) {
        self.detach(node_idx);
        self.push(node_idx, list);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger::Env;
    use std::sync::Once;

    fn init_logger() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = env_logger::Builder::from_env(Env::default().default_filter_or("warn"))
                .is_test(true)
                .try_init();
        });
    }

    #[test]
    fn test_sample() {
        init_logger();
        let mut arc_replacer = ArcReplacer::new(7);
        arc_replacer.record_access(1, 1);
        arc_replacer.record_access(2, 2);
        arc_replacer.record_access(3, 3);
        arc_replacer.record_access(4, 4);
        arc_replacer.record_access(5, 5);
        arc_replacer.record_access(6, 6);
        arc_replacer.set_evictable(1, true);
        arc_replacer.set_evictable(2, true);
        arc_replacer.set_evictable(3, true);
        arc_replacer.set_evictable(4, true);
        arc_replacer.set_evictable(5, true);
        arc_replacer.set_evictable(6, false);
        assert_eq!(5, arc_replacer.size());

        arc_replacer.record_access(1, 1);
        assert_eq!(Some(2), arc_replacer.evict());
        assert_eq!(Some(3), arc_replacer.evict());
        assert_eq!(Some(4), arc_replacer.evict());
        assert_eq!(2, arc_replacer.size());

        arc_replacer.record_access(2, 7);
        arc_replacer.set_evictable(2, true);
        arc_replacer.record_access(3, 2);
        arc_replacer.set_evictable(3, true);
        assert_eq!(4, arc_replacer.size());

        arc_replacer.record_access(4, 3);
        arc_replacer.set_evictable(4, true);
        arc_replacer.record_access(7, 4);
        arc_replacer.set_evictable(7, true);
        assert_eq!(6, arc_replacer.size());

        assert_eq!(Some(5), arc_replacer.evict());
        assert_eq!(Some(1), arc_replacer.evict());

        arc_replacer.record_access(5, 1);
        arc_replacer.set_evictable(5, true);
        assert_eq!(5, arc_replacer.size());

        assert_eq!(Some(2), arc_replacer.evict());
    }

    #[test]
    fn test_sample_two() {
        init_logger();
        let mut arc_replacer = ArcReplacer::new(3);
        arc_replacer.record_access(1, 1);
        arc_replacer.set_evictable(1, true);
        arc_replacer.record_access(2, 2);
        arc_replacer.set_evictable(2, true);
        arc_replacer.record_access(3, 3);
        arc_replacer.set_evictable(3, true);
        assert_eq!(3, arc_replacer.size());

        assert_eq!(Some(1), arc_replacer.evict());
        assert_eq!(Some(2), arc_replacer.evict());
        assert_eq!(Some(3), arc_replacer.evict());
        assert_eq!(0, arc_replacer.size());

        arc_replacer.record_access(3, 4);
        arc_replacer.set_evictable(3, true);

        arc_replacer.record_access(2, 1);
        arc_replacer.set_evictable(2, true);
        assert_eq!(2, arc_replacer.size());

        arc_replacer.record_access(1, 3);
        arc_replacer.set_evictable(1, true);

        assert_eq!(Some(3), arc_replacer.evict());
        assert_eq!(Some(2), arc_replacer.evict());
        assert_eq!(Some(1), arc_replacer.evict());

        arc_replacer.record_access(1, 1);
        arc_replacer.set_evictable(1, true);

        arc_replacer.record_access(2, 4);
        arc_replacer.set_evictable(2, true);

        arc_replacer.record_access(3, 5);
        arc_replacer.set_evictable(3, true);
        assert_eq!(Some(1), arc_replacer.evict());

        arc_replacer.record_access(1, 6);
        arc_replacer.set_evictable(1, true);
        assert_eq!(Some(2), arc_replacer.evict());

        arc_replacer.record_access(2, 7);
        arc_replacer.set_evictable(2, true);
        assert_eq!(Some(3), arc_replacer.evict());

        arc_replacer.record_access(3, 5);
        arc_replacer.set_evictable(3, true);
        assert_eq!(Some(3), arc_replacer.evict());

        arc_replacer.record_access(3, 2);
        arc_replacer.set_evictable(3, true);
        assert_eq!(Some(1), arc_replacer.evict());

        arc_replacer.record_access(1, 3);
        arc_replacer.set_evictable(1, true);

        assert_eq!(Some(2), arc_replacer.evict());
        assert_eq!(Some(3), arc_replacer.evict());
        assert_eq!(Some(1), arc_replacer.evict());
    }

    #[test]
    fn test_arc_replacer_performance() {
        init_logger();
        println!("BEGIN");
        println!("This test will see how your RecordAccess performs when the list is large.");
        println!(
            "If this takes above 3s on average, you might get into trouble trying to get full score in some following projects..."
        );
        println!(
            "if you care, you may want to think of what could be very slow when the list is very large, and how to make that faster"
        );

        let bpm_size: usize = 256 << 10; // ~1GB if frames are 4KB
        let mut arc_replacer = ArcReplacer::new(bpm_size);

        for i in 0..bpm_size {
            arc_replacer.record_access(i, i);
            arc_replacer.set_evictable(i, true);
        }

        let rounds: usize = 10;
        let mut access_frame_id: usize = 256 << 9;
        let mut access_times_ms: Vec<u128> = Vec::with_capacity(rounds);

        for _round in 0..rounds {
            let start_time = std::time::Instant::now();
            for _i in 0..bpm_size {
                arc_replacer.record_access(access_frame_id, access_frame_id);
                access_frame_id = (access_frame_id + 1) % bpm_size;
            }
            access_times_ms.push(start_time.elapsed().as_millis());
        }

        let total_ms: u128 = access_times_ms.iter().copied().sum();
        let avg_s: f64 = (total_ms as f64) / 1000.0 / (access_times_ms.len() as f64);

        println!("END");
        println!("Average time used: {avg_s}s.");
        println!(
            "If this takes above 3s on average, you might get into trouble trying to get full score in some following projects..."
        );
        println!("If you care, try optimizing RecordAccess for a bit");

        assert!(avg_s < 3.0, "Average time used: {avg_s}s");
    }
}
