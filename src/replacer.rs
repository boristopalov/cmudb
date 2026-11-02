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
enum ResidentList {
    Mru,
    Mfu,
}

impl ResidentList {
    fn list(self) -> ArcList {
        match self {
            ResidentList::Mru => ArcList::Mru,
            ResidentList::Mfu => ArcList::Mfu,
        }
    }

    fn ghost(self) -> GhostList {
        match self {
            ResidentList::Mru => GhostList::Mru,
            ResidentList::Mfu => GhostList::Mfu,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GhostList {
    Mru,
    Mfu,
}

impl GhostList {
    fn list(self) -> ArcList {
        match self {
            GhostList::Mru => ArcList::GhostMru,
            GhostList::Mfu => ArcList::GhostMfu,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Node {
    page_id: usize,
    frame_id: Option<usize>,
    evictable: bool,
    list: ArcList,
    next: Option<usize>,
    prev: Option<usize>,
}

#[derive(Default, Debug)]
struct ListHead {
    most_recent: Option<usize>,
    least_recent: Option<usize>,
    length: usize,
}

/// Adaptive Replacement Cache (ARC).
///
/// Resident pages live in an MRU list (seen once) and an MFU list (seen more than once).
/// Two ghost lists remember pages recently evicted from each, and hits on them steer
/// `mru_target_size`, the share of frames the MRU list aims to hold.
#[derive(Default, Debug)]
pub struct ArcReplacer {
    nodes: Vec<Node>,
    unused_node_indices: Vec<usize>,
    page_indices: HashMap<usize, usize>,
    frame_indices: HashMap<usize, usize>,
    mru_list: ListHead,
    mfu_list: ListHead,
    mru_ghost_list: ListHead,
    mfu_ghost_list: ListHead,
    mru_target_size: usize,
    capacity: usize,
    evictable_count: usize,
}

impl ArcReplacer {
    pub fn new(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            unused_node_indices: Vec::new(),
            page_indices: HashMap::new(),
            frame_indices: HashMap::new(),
            mru_list: ListHead::default(),
            mfu_list: ListHead::default(),
            mru_ghost_list: ListHead::default(),
            mfu_ghost_list: ListHead::default(),
            mru_target_size: 0,
            capacity: cap,
            evictable_count: 0,
        }
    }

    /// Returns the number of frames this replacer was built for.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of evictable frames.
    pub fn size(&self) -> usize {
        self.evictable_count
    }

    /// Marks whether `frame_id` may be chosen by `evict`. Unknown frames are ignored.
    pub fn set_evictable(&mut self, frame_id: usize, evictable: bool) {
        let Some(&idx) = self.frame_indices.get(&frame_id) else {
            warn!("set_evictable called for unknown frame_id={frame_id}");
            return;
        };
        let node = &mut self.nodes[idx];
        let was_evictable = node.evictable;
        node.evictable = evictable;
        if evictable && !was_evictable {
            self.evictable_count += 1;
        } else if !evictable && was_evictable {
            self.evictable_count -= 1;
        }
        info!(
            "set_evictable frame_id={frame_id} evictable={evictable} list={:?}",
            node.list
        );
    }

    fn is_full(&self) -> bool {
        self.mru_list.length + self.mfu_list.length >= self.capacity
    }

    /// Records an access to `page_id`, now held in `frame_id`. A new frame starts non-evictable.
    pub fn record_access(&mut self, frame_id: usize, page_id: usize) {
        match self.page_indices.get(&page_id).copied() {
            Some(idx) => self.record_repeat_access(idx, frame_id, page_id),
            None => self.record_first_access(frame_id, page_id),
        }
    }

    fn record_repeat_access(&mut self, idx: usize, frame_id: usize, page_id: usize) {
        match self.nodes[idx].list {
            ArcList::Mfu | ArcList::Mru => {
                info!("record_access page_id={page_id} frame_id={frame_id} -> front of MFU");
                if let Some(previous_frame) = self.nodes[idx].frame_id.replace(frame_id)
                    && previous_frame != frame_id
                {
                    self.frame_indices.remove(&previous_frame);
                }
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
                self.revive_ghost(idx, frame_id);
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
                self.revive_ghost(idx, frame_id);
            }
        }
    }

    fn revive_ghost(&mut self, idx: usize, frame_id: usize) {
        match self.frame_indices.get(&frame_id).copied() {
            Some(previous_holder) => self.give_up_frame(previous_holder),
            None => assert!(
                !self.is_full(),
                "ghost hit admits frame {frame_id} into a full replacer; evict a frame first"
            ),
        }
        let node = &mut self.nodes[idx];
        node.frame_id = Some(frame_id);
        node.evictable = false;
        self.frame_indices.insert(frame_id, idx);
        self.move_to_front(idx, ArcList::Mfu);
    }

    fn record_first_access(&mut self, frame_id: usize, page_id: usize) {
        info!(
            "record_access new page_id={page_id} frame_id={frame_id} -> insert into MRU (ghost miss)"
        );

        let mru_side = self.mru_list.length + self.mru_ghost_list.length;
        if mru_side == self.capacity {
            if !self.trim_ghost_tail(GhostList::Mru) {
                error!("cannot insert new node: MRU ghost list unexpectedly empty");
                return;
            }
        } else {
            assert!(
                mru_side < self.capacity,
                "MRU list plus MRU ghost list exceeds capacity: {} + {} > {}",
                self.mru_list.length,
                self.mru_ghost_list.length,
                self.capacity
            );
            let tracked = mru_side + self.mfu_list.length + self.mfu_ghost_list.length;
            if tracked == 2 * self.capacity && !self.trim_ghost_tail(GhostList::Mfu) {
                error!("cannot insert new node: MFU ghost list unexpectedly empty");
                return;
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
        let idx = match self.unused_node_indices.pop() {
            Some(idx) => {
                self.nodes[idx] = new_node;
                idx
            }
            None => {
                self.nodes.push(new_node);
                self.nodes.len() - 1
            }
        };
        self.page_indices.insert(page_id, idx);
        self.frame_indices.insert(frame_id, idx);
        self.push(idx, ArcList::Mru);
    }

    /// Picks a victim frame, moves its page to a ghost list and returns the frame id.
    ///
    /// Returns `None` if no frame is evictable.
    pub fn evict(&mut self) -> Option<usize> {
        info!(
            "evict: attempting eviction mru_target_size={} mru_len={} mfu_len={}",
            self.mru_target_size, self.mru_list.length, self.mfu_list.length
        );
        let (preferred, fallback) = if self.mru_list.length >= self.mru_target_size {
            (ResidentList::Mru, ResidentList::Mfu)
        } else {
            (ResidentList::Mfu, ResidentList::Mru)
        };
        let victim = self
            .evict_from_list(preferred)
            .or_else(|| self.evict_from_list(fallback));
        if victim.is_none() {
            warn!("evict: no evictable frames available");
        }
        victim
    }

    fn evict_from_list(&mut self, resident: ResidentList) -> Option<usize> {
        let list = resident.list();
        let mut current = self.list(list).least_recent?;

        while !self.nodes[current].evictable {
            let node = &self.nodes[current];
            warn!(
                "evict_from_list: frame not evictable list={:?} page_id={} frame_id={:?}",
                list, node.page_id, node.frame_id
            );
            let Some(more_recent) = node.prev else {
                warn!("evict_from_list: no evictable frames found in list {list:?}");
                return None;
            };
            current = more_recent;
        }

        let page_id = self.nodes[current].page_id;
        let Some(frame_id) = self.nodes[current].frame_id else {
            warn!("evict_from_list: node missing frame_id list={list:?} page_id={page_id}");
            return None;
        };
        info!(
            "evict_from_list: evicted frame_id={frame_id} page_id={page_id} from list={list:?} -> {:?}",
            resident.ghost().list()
        );
        self.give_up_frame(current);
        Some(frame_id)
    }

    fn give_up_frame(&mut self, idx: usize) {
        let resident = match self.nodes[idx].list {
            ArcList::Mru => ResidentList::Mru,
            ArcList::Mfu => ResidentList::Mfu,
            ghost => panic!("frame index points at a node on ghost list {ghost:?}"),
        };
        let node = &mut self.nodes[idx];
        if let Some(frame_id) = node.frame_id.take() {
            self.frame_indices.remove(&frame_id);
        }
        if std::mem::replace(&mut node.evictable, false) {
            self.evictable_count -= 1;
        }
        self.move_to_front(idx, resident.ghost().list());
    }

    fn trim_ghost_tail(&mut self, ghost: GhostList) -> bool {
        let ghost_list = ghost.list();
        let Some(idx) = self.list(ghost_list).least_recent else {
            warn!(
                "trim_ghost_tail: ghost list empty list={:?} mru_ghost_len={} mfu_ghost_len={}",
                ghost_list, self.mru_ghost_list.length, self.mfu_ghost_list.length
            );
            return false;
        };

        let page_id = self.nodes[idx].page_id;
        debug_assert!(
            self.nodes[idx].frame_id.is_none(),
            "ghost node for page {page_id} still holds a frame"
        );
        info!("trim_ghost_tail: drop ghost page_id={page_id} from list={ghost_list:?}");
        self.detach(idx);
        self.page_indices.remove(&page_id);
        self.unused_node_indices.push(idx);
        true
    }

    fn list(&self, list: ArcList) -> &ListHead {
        match list {
            ArcList::Mfu => &self.mfu_list,
            ArcList::Mru => &self.mru_list,
            ArcList::GhostMru => &self.mru_ghost_list,
            ArcList::GhostMfu => &self.mfu_ghost_list,
        }
    }

    fn list_mut(&mut self, list: ArcList) -> &mut ListHead {
        match list {
            ArcList::Mfu => &mut self.mfu_list,
            ArcList::Mru => &mut self.mru_list,
            ArcList::GhostMru => &mut self.mru_ghost_list,
            ArcList::GhostMfu => &mut self.mfu_ghost_list,
        }
    }

    fn detach(&mut self, node_idx: usize) {
        let Node {
            list, next, prev, ..
        } = self.nodes[node_idx];

        let head = self.list_mut(list);
        if head.most_recent == Some(node_idx) {
            head.most_recent = next;
        }
        if head.least_recent == Some(node_idx) {
            head.least_recent = prev;
        }
        head.length -= 1;

        if let Some(p) = prev {
            self.nodes[p].next = next;
        }
        if let Some(n) = next {
            self.nodes[n].prev = prev;
        }
        self.nodes[node_idx].prev = None;
        self.nodes[node_idx].next = None;
    }

    fn push(&mut self, node_idx: usize, list: ArcList) {
        let head = self.list_mut(list);
        let old_most_recent = head.most_recent.replace(node_idx);
        if old_most_recent.is_none() {
            head.least_recent = Some(node_idx);
        }
        head.length += 1;

        self.nodes[node_idx].list = list;
        self.nodes[node_idx].prev = None;
        self.nodes[node_idx].next = old_most_recent;
        if let Some(h) = old_most_recent {
            self.nodes[h].prev = Some(node_idx);
        }
    }

    fn move_to_front(&mut self, node_idx: usize, list: ArcList) {
        self.detach(node_idx);
        self.push(node_idx, list);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "MRU list plus MRU ghost list exceeds capacity")]
    fn oversized_mru_history_panics_instead_of_exiting_the_process() {
        let mut arc_replacer = ArcReplacer::new(1);
        arc_replacer.mru_ghost_list.length = 2;
        arc_replacer.record_access(1, 1);
    }

    #[test]
    fn node_storage_stays_bounded_by_twice_the_capacity() {
        let capacity = 2;
        let mut arc_replacer = ArcReplacer::new(capacity);
        for frame_id in 0..capacity {
            arc_replacer.record_access(frame_id, frame_id);
            arc_replacer.set_evictable(frame_id, true);
        }
        for page_id in capacity..1_000 {
            let frame_id = arc_replacer.evict().expect("an evictable frame");
            arc_replacer.record_access(frame_id, page_id);
            arc_replacer.set_evictable(frame_id, true);
        }
        assert!(
            arc_replacer.nodes.len() <= 2 * capacity,
            "replacer kept {} nodes for capacity {capacity}",
            arc_replacer.nodes.len()
        );
    }

    const GHOST_PAGE: usize = 12;
    const PAGE_IN_REUSED_FRAME: usize = 13;
    const REUSED_FRAME: usize = 0;
    const OTHER_FRAME: usize = 1;

    fn full_replacer_with_a_ghost_mru_page() -> ArcReplacer {
        let mut arc_replacer = ArcReplacer::new(2);
        arc_replacer.record_access(REUSED_FRAME, 10);
        arc_replacer.record_access(REUSED_FRAME, 10);
        arc_replacer.set_evictable(REUSED_FRAME, true);
        arc_replacer.record_access(OTHER_FRAME, 11);
        arc_replacer.set_evictable(OTHER_FRAME, true);
        assert_eq!(arc_replacer.evict(), Some(OTHER_FRAME));
        arc_replacer.record_access(OTHER_FRAME, 11);
        arc_replacer.set_evictable(OTHER_FRAME, true);
        assert_eq!(arc_replacer.evict(), Some(REUSED_FRAME));
        arc_replacer.record_access(REUSED_FRAME, GHOST_PAGE);
        arc_replacer.set_evictable(REUSED_FRAME, true);
        assert_eq!(arc_replacer.evict(), Some(REUSED_FRAME));
        arc_replacer.record_access(REUSED_FRAME, PAGE_IN_REUSED_FRAME);
        arc_replacer.set_evictable(REUSED_FRAME, true);
        assert_eq!(
            arc_replacer.mru_list.length + arc_replacer.mfu_list.length,
            2
        );
        assert_eq!(
            arc_replacer.nodes[arc_replacer.page_indices[&GHOST_PAGE]].list,
            ArcList::GhostMru
        );
        arc_replacer
    }

    #[test]
    fn ghost_hit_on_a_full_replacer_never_hands_out_the_admitted_frame() {
        let mut arc_replacer = full_replacer_with_a_ghost_mru_page();

        arc_replacer.record_access(REUSED_FRAME, GHOST_PAGE);

        assert_eq!(
            arc_replacer.size(),
            1,
            "only the other frame should stay evictable"
        );
        assert_eq!(
            arc_replacer.evict(),
            Some(OTHER_FRAME),
            "the ghost hit must not take the other frame away from the caller"
        );
        assert_eq!(
            arc_replacer.evict(),
            None,
            "the reused frame now holds the non-evictable revived page"
        );
    }

    #[test]
    fn ghost_hit_moves_the_reused_frames_previous_page_to_its_ghost_list() {
        let mut arc_replacer = full_replacer_with_a_ghost_mru_page();

        arc_replacer.record_access(REUSED_FRAME, GHOST_PAGE);

        let displaced = arc_replacer.page_indices[&PAGE_IN_REUSED_FRAME];
        assert_eq!(arc_replacer.nodes[displaced].list, ArcList::GhostMru);
        assert_eq!(arc_replacer.nodes[displaced].frame_id, None);
        let revived = arc_replacer.frame_indices[&REUSED_FRAME];
        assert_eq!(arc_replacer.nodes[revived].page_id, GHOST_PAGE);
        assert_eq!(arc_replacer.nodes[revived].list, ArcList::Mfu);
        assert_eq!(
            arc_replacer.frame_indices[&OTHER_FRAME],
            arc_replacer.page_indices[&11]
        );
    }

    #[test]
    #[should_panic(expected = "ghost hit admits frame 2 into a full replacer")]
    fn ghost_hit_on_a_full_replacer_with_an_untracked_frame_panics() {
        let mut arc_replacer = full_replacer_with_a_ghost_mru_page();
        arc_replacer.record_access(2, GHOST_PAGE);
    }

    #[test]
    fn page_moved_to_another_frame_forgets_its_old_frame() {
        let mut arc_replacer = ArcReplacer::new(4);
        arc_replacer.record_access(1, 10);
        arc_replacer.record_access(2, 10);
        assert!(!arc_replacer.frame_indices.contains_key(&1));
        arc_replacer.set_evictable(2, true);
        assert_eq!(arc_replacer.size(), 1);
        arc_replacer.set_evictable(1, true);
        assert_eq!(
            arc_replacer.size(),
            1,
            "stale frame 1 changed the evictable count"
        );
    }

    #[test]
    fn test_sample() {
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
        let bpm_size: usize = 256 << 10;
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

        assert!(avg_s < 3.0, "Average time used: {avg_s}s");
    }
}
