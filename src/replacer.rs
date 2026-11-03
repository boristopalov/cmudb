use crate::ids::{FrameId, PageId};
use log::{info, warn};
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
    page_id: PageId,
    frame_id: Option<FrameId>,
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
    page_indices: HashMap<PageId, usize>,
    frame_indices: HashMap<FrameId, usize>,
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
    pub fn set_evictable(&mut self, frame_id: FrameId, evictable: bool) {
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

    /// Forgets `page_id` entirely, whether it holds a frame or sits on a ghost list, for a
    /// page that is deleted rather than evicted, so a later page that reuses the id starts
    /// fresh instead of counting as a ghost hit. Returns the frame the page held, if any.
    pub fn remove(&mut self, page_id: PageId) -> Option<FrameId> {
        let idx = self.page_indices.remove(&page_id)?;
        let node = &mut self.nodes[idx];
        let frame_id = node.frame_id.take();
        if std::mem::replace(&mut node.evictable, false) {
            self.evictable_count -= 1;
        }
        if let Some(frame_id) = frame_id {
            self.frame_indices.remove(&frame_id);
        }
        info!(
            "remove page_id={page_id} frame_id={frame_id:?} list={:?}",
            self.nodes[idx].list
        );
        self.detach(idx);
        self.unused_node_indices.push(idx);
        frame_id
    }

    fn is_full(&self) -> bool {
        self.mru_list.length + self.mfu_list.length >= self.capacity
    }

    /// Records an access to `page_id`, now held in `frame_id`. A new frame starts non-evictable.
    pub fn record_access(&mut self, frame_id: FrameId, page_id: PageId) {
        match self.page_indices.get(&page_id).copied() {
            Some(idx) => self.record_repeat_access(idx, frame_id, page_id),
            None => self.record_first_access(frame_id, page_id),
        }
    }

    fn record_repeat_access(&mut self, idx: usize, frame_id: FrameId, page_id: PageId) {
        match self.nodes[idx].list {
            ArcList::Mfu | ArcList::Mru => {
                info!("record_access page_id={page_id} frame_id={frame_id} -> front of MFU");
                if self.nodes[idx].frame_id != Some(frame_id) {
                    self.move_to_new_frame(idx, frame_id);
                }
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

    fn revive_ghost(&mut self, idx: usize, frame_id: FrameId) {
        self.release_frame_for_admission(frame_id, "ghost hit");
        let node = &mut self.nodes[idx];
        node.frame_id = Some(frame_id);
        node.evictable = false;
        self.frame_indices.insert(frame_id, idx);
        self.move_to_front(idx, ArcList::Mfu);
    }

    fn move_to_new_frame(&mut self, idx: usize, frame_id: FrameId) {
        self.take_frame_from_its_holder(frame_id);
        let node = &mut self.nodes[idx];
        if let Some(previous_frame) = node.frame_id.replace(frame_id) {
            self.frame_indices.remove(&previous_frame);
        }
        if std::mem::replace(&mut node.evictable, false) {
            self.evictable_count -= 1;
        }
        self.frame_indices.insert(frame_id, idx);
    }

    fn take_frame_from_its_holder(&mut self, frame_id: FrameId) -> bool {
        match self.frame_indices.get(&frame_id).copied() {
            Some(previous_holder) => {
                self.give_up_frame(previous_holder);
                true
            }
            None => false,
        }
    }

    fn release_frame_for_admission(&mut self, frame_id: FrameId, admission: &str) {
        let freed_a_tracked_frame = self.take_frame_from_its_holder(frame_id);
        assert!(
            freed_a_tracked_frame || !self.is_full(),
            "{admission} admits frame {frame_id} into a full replacer; evict a frame first"
        );
    }

    fn record_first_access(&mut self, frame_id: FrameId, page_id: PageId) {
        info!(
            "record_access new page_id={page_id} frame_id={frame_id} -> insert into MRU (ghost miss)"
        );

        self.release_frame_for_admission(frame_id, "first access");

        let mru_side = self.mru_list.length + self.mru_ghost_list.length;
        if mru_side == self.capacity {
            self.trim_ghost_tail(GhostList::Mru);
        } else {
            assert!(
                mru_side < self.capacity,
                "MRU list plus MRU ghost list exceeds capacity: {} + {} > {}",
                self.mru_list.length,
                self.mru_ghost_list.length,
                self.capacity
            );
            let tracked = mru_side + self.mfu_list.length + self.mfu_ghost_list.length;
            if tracked == 2 * self.capacity {
                self.trim_ghost_tail(GhostList::Mfu);
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
    pub fn evict(&mut self) -> Option<FrameId> {
        self.begin_eviction().map(Eviction::commit)
    }

    /// Picks a victim frame like [`ArcReplacer::evict`], but keeps it restorable until
    /// [`Eviction::commit`] is called. Dropping the returned [`Eviction`] uncommitted puts
    /// the frame back in its list, at its old position, still evictable, and leaves the
    /// MRU target size as it was.
    ///
    /// Returns `None` if no frame is evictable.
    pub fn begin_eviction(&mut self) -> Option<Eviction<'_>> {
        info!(
            "evict: attempting eviction mru_target_size={} mru_len={} mfu_len={}",
            self.mru_target_size, self.mru_list.length, self.mfu_list.length
        );
        let (preferred, fallback) = if self.mru_list.length >= self.mru_target_size {
            (ResidentList::Mru, ResidentList::Mfu)
        } else {
            (ResidentList::Mfu, ResidentList::Mru)
        };
        let Some(undo) = self
            .evict_from_list(preferred)
            .or_else(|| self.evict_from_list(fallback))
        else {
            warn!("evict: no evictable frames available");
            return None;
        };
        Some(Eviction {
            replacer: self,
            undo: Some(undo),
        })
    }

    fn evict_from_list(&mut self, resident: ResidentList) -> Option<EvictionUndo> {
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
        let more_recent = self.nodes[current].prev;
        let Some(frame_id) = self.nodes[current].frame_id else {
            warn!("evict_from_list: node missing frame_id list={list:?} page_id={page_id}");
            return None;
        };
        info!(
            "evict_from_list: evicted frame_id={frame_id} page_id={page_id} from list={list:?} -> {:?}",
            resident.ghost().list()
        );
        self.give_up_frame(current);
        Some(EvictionUndo {
            node_idx: current,
            frame_id,
            list: resident,
            more_recent,
        })
    }

    fn undo_eviction(&mut self, undo: EvictionUndo) {
        let EvictionUndo {
            node_idx,
            frame_id,
            list,
            more_recent,
        } = undo;
        info!(
            "undo_eviction: restoring frame_id={frame_id} page_id={} to list={:?}",
            self.nodes[node_idx].page_id,
            list.list()
        );
        self.detach(node_idx);
        self.insert_after(node_idx, list.list(), more_recent);
        self.nodes[node_idx].frame_id = Some(frame_id);
        self.nodes[node_idx].evictable = true;
        self.frame_indices.insert(frame_id, node_idx);
        self.evictable_count += 1;
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

    fn trim_ghost_tail(&mut self, ghost: GhostList) {
        let ghost_list = ghost.list();
        let idx = self.list(ghost_list).least_recent.unwrap_or_else(|| {
            panic!(
                "{ghost_list:?} is empty with {} resident and {} ghost pages tracked for capacity {}",
                self.mru_list.length + self.mfu_list.length,
                self.mru_ghost_list.length + self.mfu_ghost_list.length,
                self.capacity
            )
        });

        let page_id = self.nodes[idx].page_id;
        debug_assert!(
            self.nodes[idx].frame_id.is_none(),
            "ghost node for page {page_id} still holds a frame"
        );
        info!("trim_ghost_tail: drop ghost page_id={page_id} from list={ghost_list:?}");
        self.detach(idx);
        self.page_indices.remove(&page_id);
        self.unused_node_indices.push(idx);
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

    fn insert_after(&mut self, node_idx: usize, list: ArcList, more_recent: Option<usize>) {
        let Some(more_recent) = more_recent else {
            self.push(node_idx, list);
            return;
        };
        let less_recent = self.nodes[more_recent].next;
        self.nodes[more_recent].next = Some(node_idx);
        self.nodes[node_idx].list = list;
        self.nodes[node_idx].prev = Some(more_recent);
        self.nodes[node_idx].next = less_recent;
        match less_recent {
            Some(less_recent) => self.nodes[less_recent].prev = Some(node_idx),
            None => self.list_mut(list).least_recent = Some(node_idx),
        }
        self.list_mut(list).length += 1;
    }

    fn move_to_front(&mut self, node_idx: usize, list: ArcList) {
        self.detach(node_idx);
        self.push(node_idx, list);
    }
}

#[derive(Debug)]
struct EvictionUndo {
    node_idx: usize,
    frame_id: FrameId,
    list: ResidentList,
    more_recent: Option<usize>,
}

/// A victim chosen by [`ArcReplacer::begin_eviction`]. It holds the replacer mutably, so
/// nothing else changes the replacer until it is committed or dropped.
///
/// Dropping it without calling [`Eviction::commit`] restores the victim as it was.
#[derive(Debug)]
#[must_use = "dropping an uncommitted eviction restores the victim"]
pub struct Eviction<'a> {
    replacer: &'a mut ArcReplacer,
    undo: Option<EvictionUndo>,
}

impl Eviction<'_> {
    /// Returns the victim frame's id.
    pub fn frame_id(&self) -> FrameId {
        self.undo
            .as_ref()
            .map(|undo| undo.frame_id)
            .expect("an uncommitted eviction keeps its undo record")
    }

    /// Makes the eviction final and returns the victim frame's id.
    pub fn commit(mut self) -> FrameId {
        self.undo
            .take()
            .map(|undo| undo.frame_id)
            .expect("an uncommitted eviction keeps its undo record")
    }
}

impl Drop for Eviction<'_> {
    fn drop(&mut self) {
        if let Some(undo) = self.undo.take() {
            self.replacer.undo_eviction(undo);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    trait AsFrame {
        fn frame(self) -> FrameId;
    }
    impl AsFrame for usize {
        fn frame(self) -> FrameId {
            FrameId(self)
        }
    }
    impl AsFrame for i32 {
        fn frame(self) -> FrameId {
            FrameId(usize::try_from(self).unwrap())
        }
    }
    impl AsFrame for FrameId {
        fn frame(self) -> FrameId {
            self
        }
    }

    trait AsPage {
        fn page(self) -> PageId;
    }
    impl AsPage for usize {
        fn page(self) -> PageId {
            PageId::try_from(self).unwrap()
        }
    }
    impl AsPage for i32 {
        fn page(self) -> PageId {
            PageId::new(u32::try_from(self).unwrap())
        }
    }
    impl AsPage for PageId {
        fn page(self) -> PageId {
            self
        }
    }

    fn f(id: impl AsFrame) -> FrameId {
        id.frame()
    }

    fn p(id: impl AsPage) -> PageId {
        id.page()
    }

    #[test]
    #[should_panic(expected = "MRU list plus MRU ghost list exceeds capacity")]
    fn oversized_mru_history_panics_instead_of_exiting_the_process() {
        let mut arc_replacer = ArcReplacer::new(1);
        arc_replacer.mru_ghost_list.length = 2;
        arc_replacer.record_access(f(1), p(1));
    }

    #[test]
    fn node_storage_stays_bounded_by_twice_the_capacity() {
        let capacity = 2;
        let mut arc_replacer = ArcReplacer::new(capacity);
        for frame_id in 0..capacity {
            arc_replacer.record_access(f(frame_id), p(frame_id));
            arc_replacer.set_evictable(f(frame_id), true);
        }
        for page_id in capacity..1_000 {
            let frame_id = arc_replacer.evict().expect("an evictable frame");
            arc_replacer.record_access(f(frame_id), p(page_id));
            arc_replacer.set_evictable(f(frame_id), true);
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
        arc_replacer.record_access(f(REUSED_FRAME), p(10));
        arc_replacer.record_access(f(REUSED_FRAME), p(10));
        arc_replacer.set_evictable(f(REUSED_FRAME), true);
        arc_replacer.record_access(f(OTHER_FRAME), p(11));
        arc_replacer.set_evictable(f(OTHER_FRAME), true);
        assert_eq!(arc_replacer.evict(), Some(f(OTHER_FRAME)));
        arc_replacer.record_access(f(OTHER_FRAME), p(11));
        arc_replacer.set_evictable(f(OTHER_FRAME), true);
        assert_eq!(arc_replacer.evict(), Some(f(REUSED_FRAME)));
        arc_replacer.record_access(f(REUSED_FRAME), p(GHOST_PAGE));
        arc_replacer.set_evictable(f(REUSED_FRAME), true);
        assert_eq!(arc_replacer.evict(), Some(f(REUSED_FRAME)));
        arc_replacer.record_access(f(REUSED_FRAME), p(PAGE_IN_REUSED_FRAME));
        arc_replacer.set_evictable(f(REUSED_FRAME), true);
        assert_eq!(
            arc_replacer.mru_list.length + arc_replacer.mfu_list.length,
            2
        );
        assert_eq!(
            arc_replacer.nodes[arc_replacer.page_indices[&p(GHOST_PAGE)]].list,
            ArcList::GhostMru
        );
        arc_replacer
    }

    #[test]
    fn ghost_hit_on_a_full_replacer_never_hands_out_the_admitted_frame() {
        let mut arc_replacer = full_replacer_with_a_ghost_mru_page();

        arc_replacer.record_access(f(REUSED_FRAME), p(GHOST_PAGE));

        assert_eq!(
            arc_replacer.size(),
            1,
            "only the other frame should stay evictable"
        );
        assert_eq!(
            arc_replacer.evict(),
            Some(f(OTHER_FRAME)),
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

        arc_replacer.record_access(f(REUSED_FRAME), p(GHOST_PAGE));

        let displaced = arc_replacer.page_indices[&p(PAGE_IN_REUSED_FRAME)];
        assert_eq!(arc_replacer.nodes[displaced].list, ArcList::GhostMru);
        assert_eq!(arc_replacer.nodes[displaced].frame_id, None);
        let revived = arc_replacer.frame_indices[&f(REUSED_FRAME)];
        assert_eq!(arc_replacer.nodes[revived].page_id, p(GHOST_PAGE));
        assert_eq!(arc_replacer.nodes[revived].list, ArcList::Mfu);
        assert_eq!(
            arc_replacer.frame_indices[&f(OTHER_FRAME)],
            arc_replacer.page_indices[&p(11)]
        );
    }

    #[test]
    #[should_panic(expected = "ghost hit admits frame 2 into a full replacer")]
    fn ghost_hit_on_a_full_replacer_with_an_untracked_frame_panics() {
        let mut arc_replacer = full_replacer_with_a_ghost_mru_page();
        arc_replacer.record_access(f(2), p(GHOST_PAGE));
    }

    const NEW_PAGE: usize = 12;

    fn replacer_with_evictable_pages_in_frames_0_and_1(capacity: usize) -> ArcReplacer {
        let mut arc_replacer = ArcReplacer::new(capacity);
        arc_replacer.record_access(f(REUSED_FRAME), p(10));
        arc_replacer.set_evictable(f(REUSED_FRAME), true);
        arc_replacer.record_access(f(OTHER_FRAME), p(11));
        arc_replacer.set_evictable(f(OTHER_FRAME), true);
        arc_replacer
    }

    fn assert_first_access_took_the_reused_frame_from_page_10(arc_replacer: &mut ArcReplacer) {
        let admitted = arc_replacer.frame_indices[&f(REUSED_FRAME)];
        assert_eq!(arc_replacer.nodes[admitted].page_id, p(NEW_PAGE));
        assert_eq!(arc_replacer.nodes[admitted].list, ArcList::Mru);
        assert_eq!(
            arc_replacer.size(),
            1,
            "only the other frame should stay evictable"
        );
        assert_eq!(arc_replacer.evict(), Some(f(OTHER_FRAME)));
        assert_eq!(
            arc_replacer.evict(),
            None,
            "the reused frame now holds the non-evictable new page"
        );
    }

    #[test]
    fn first_access_into_a_tracked_frame_moves_its_previous_page_to_its_ghost_list() {
        let mut arc_replacer = replacer_with_evictable_pages_in_frames_0_and_1(3);

        arc_replacer.record_access(f(REUSED_FRAME), p(NEW_PAGE));

        let displaced = arc_replacer.page_indices[&p(10)];
        assert_eq!(arc_replacer.nodes[displaced].list, ArcList::GhostMru);
        assert_eq!(arc_replacer.nodes[displaced].frame_id, None);
        assert_first_access_took_the_reused_frame_from_page_10(&mut arc_replacer);
    }

    #[test]
    fn first_access_into_a_tracked_frame_of_a_full_mru_list_admits_the_page() {
        let mut arc_replacer = replacer_with_evictable_pages_in_frames_0_and_1(2);

        arc_replacer.record_access(f(REUSED_FRAME), p(NEW_PAGE));

        assert!(
            !arc_replacer.page_indices.contains_key(&p(10)),
            "a full MRU side drops the displaced page's ghost to make room"
        );
        assert_eq!(arc_replacer.mru_ghost_list.length, 0);
        assert_first_access_took_the_reused_frame_from_page_10(&mut arc_replacer);
    }

    #[test]
    #[should_panic(expected = "first access admits frame 2 into a full replacer")]
    fn first_access_into_a_full_replacer_with_an_untracked_frame_panics() {
        let mut arc_replacer = replacer_with_evictable_pages_in_frames_0_and_1(2);
        arc_replacer.record_access(f(2), p(NEW_PAGE));
    }

    #[test]
    fn repeat_access_into_another_tracked_frame_moves_its_previous_page_to_its_ghost_list() {
        let mut arc_replacer = replacer_with_evictable_pages_in_frames_0_and_1(3);

        arc_replacer.record_access(f(OTHER_FRAME), p(10));

        let displaced = arc_replacer.page_indices[&p(11)];
        assert_eq!(arc_replacer.nodes[displaced].list, ArcList::GhostMru);
        assert_eq!(arc_replacer.nodes[displaced].frame_id, None);
        let moved = arc_replacer.frame_indices[&f(OTHER_FRAME)];
        assert_eq!(arc_replacer.nodes[moved].page_id, p(10));
        assert_eq!(arc_replacer.nodes[moved].list, ArcList::Mfu);
        assert!(!arc_replacer.frame_indices.contains_key(&f(REUSED_FRAME)));
        assert_eq!(
            arc_replacer.size(),
            0,
            "the moved page starts non-evictable in its new frame"
        );
        assert_eq!(arc_replacer.evict(), None);
    }

    #[test]
    fn repeat_access_into_an_untracked_frame_of_a_full_replacer_moves_the_page() {
        let mut arc_replacer = replacer_with_evictable_pages_in_frames_0_and_1(2);

        arc_replacer.record_access(f(2), p(10));

        assert!(!arc_replacer.frame_indices.contains_key(&f(REUSED_FRAME)));
        assert_eq!(
            arc_replacer.nodes[arc_replacer.frame_indices[&f(2)]].page_id,
            p(10)
        );
        assert_eq!(arc_replacer.size(), 1);
        assert_eq!(arc_replacer.evict(), Some(f(OTHER_FRAME)));
        assert_eq!(arc_replacer.evict(), None);
    }

    #[test]
    fn page_moved_to_another_frame_forgets_its_old_frame() {
        let mut arc_replacer = ArcReplacer::new(4);
        arc_replacer.record_access(f(1), p(10));
        arc_replacer.record_access(f(2), p(10));
        assert!(!arc_replacer.frame_indices.contains_key(&f(1)));
        arc_replacer.set_evictable(f(2), true);
        assert_eq!(arc_replacer.size(), 1);
        arc_replacer.set_evictable(f(1), true);
        assert_eq!(
            arc_replacer.size(),
            1,
            "stale frame 1 changed the evictable count"
        );
    }

    fn replacer_with_evictable_frames(frames: &[usize]) -> ArcReplacer {
        let mut arc_replacer = ArcReplacer::new(frames.len() + 1);
        for &frame_id in frames {
            arc_replacer.record_access(f(frame_id), p(frame_id * 10));
            arc_replacer.set_evictable(f(frame_id), true);
        }
        arc_replacer
    }

    fn drain_evictions(arc_replacer: &mut ArcReplacer) -> Vec<FrameId> {
        std::iter::from_fn(|| arc_replacer.evict()).collect()
    }

    fn assert_uncommitted_eviction_changes_nothing(setup: impl Fn(&mut ArcReplacer)) {
        let mut untouched = replacer_with_evictable_frames(&[1, 2, 3, 4]);
        let mut restored = replacer_with_evictable_frames(&[1, 2, 3, 4]);
        setup(&mut untouched);
        setup(&mut restored);

        let victim = restored
            .begin_eviction()
            .expect("an evictable frame")
            .frame_id();
        assert_eq!(restored.size(), untouched.size(), "evictable count");
        assert_eq!(restored.mru_target_size, untouched.mru_target_size);
        assert_eq!(
            restored.mru_ghost_list.length,
            untouched.mru_ghost_list.length
        );
        assert_eq!(
            restored.mfu_ghost_list.length,
            untouched.mfu_ghost_list.length
        );
        assert_eq!(restored.frame_indices, untouched.frame_indices);

        assert_eq!(
            restored.nodes[restored.frame_indices[&f(victim)]].list,
            untouched.nodes[untouched.frame_indices[&f(victim)]].list,
            "the victim went back to a different list"
        );
        for frame_id in 1..=4 {
            restored.set_evictable(f(frame_id), true);
            untouched.set_evictable(f(frame_id), true);
        }
        assert_eq!(
            drain_evictions(&mut restored),
            drain_evictions(&mut untouched),
            "eviction order changed after an uncommitted eviction"
        );
    }

    #[test]
    fn an_uncommitted_eviction_of_the_least_recent_frame_restores_it() {
        assert_uncommitted_eviction_changes_nothing(|_| {});
    }

    #[test]
    fn an_uncommitted_eviction_from_the_middle_of_a_list_restores_its_position() {
        assert_uncommitted_eviction_changes_nothing(|arc_replacer| {
            arc_replacer.set_evictable(f(1), false);
        });
    }

    #[test]
    fn an_uncommitted_eviction_from_the_mfu_list_restores_it_there() {
        assert_uncommitted_eviction_changes_nothing(|arc_replacer| {
            arc_replacer.record_access(f(2), p(20));
            arc_replacer.record_access(f(3), p(30));
            arc_replacer.mru_target_size = 3;
        });
    }

    #[test]
    fn a_restored_victim_is_not_counted_as_a_ghost_hit_on_its_next_access() {
        let mut arc_replacer = replacer_with_evictable_frames(&[1, 2]);
        let victim = arc_replacer
            .begin_eviction()
            .expect("an evictable frame")
            .frame_id();
        arc_replacer.record_access(f(victim), p(victim.0 * 10));
        assert_eq!(arc_replacer.mru_target_size, 0, "MRU target size moved");
        assert_eq!(arc_replacer.mru_ghost_list.length, 0);
        assert_eq!(arc_replacer.mfu_list.length, 1);
    }

    #[test]
    fn a_committed_eviction_moves_the_victim_to_a_ghost_list() {
        let mut arc_replacer = replacer_with_evictable_frames(&[1, 2]);
        let eviction = arc_replacer.begin_eviction().expect("an evictable frame");
        assert_eq!(eviction.frame_id(), f(1));
        assert_eq!(eviction.commit(), f(1));
        assert_eq!(arc_replacer.size(), 1);
        assert_eq!(arc_replacer.mru_ghost_list.length, 1);
        assert!(!arc_replacer.frame_indices.contains_key(&f(1)));
    }

    #[test]
    fn a_removed_page_frees_its_frame_and_leaves_no_ghost() {
        let mut arc_replacer = ArcReplacer::new(2);
        arc_replacer.record_access(f(1), p(10));
        arc_replacer.record_access(f(2), p(20));
        arc_replacer.set_evictable(f(1), true);
        arc_replacer.set_evictable(f(2), true);

        assert_eq!(arc_replacer.remove(p(10)), Some(f(1)));
        assert_eq!(
            arc_replacer.remove(p(10)),
            None,
            "page 10 is no longer tracked"
        );
        assert_eq!(arc_replacer.size(), 1);
        assert!(
            !arc_replacer.page_indices.contains_key(&p(10)),
            "page 10 left a ghost"
        );
        assert_eq!(arc_replacer.evict(), Some(f(2)));
        assert_eq!(arc_replacer.evict(), None);

        arc_replacer.record_access(f(1), p(30));
        arc_replacer.record_access(f(2), p(40));
        arc_replacer.set_evictable(f(1), true);
        assert_eq!(
            arc_replacer.evict(),
            Some(f(1)),
            "a removed frame is tracked again once a new page uses it"
        );
    }

    #[test]
    fn removing_a_pinned_page_leaves_the_evictable_count_alone() {
        let mut arc_replacer = ArcReplacer::new(2);
        arc_replacer.record_access(f(0), p(10));
        arc_replacer.record_access(f(1), p(11));
        arc_replacer.set_evictable(f(1), true);
        assert_eq!(arc_replacer.remove(p(10)), Some(f(0)));
        assert_eq!(arc_replacer.size(), 1);
        assert_eq!(arc_replacer.evict(), Some(f(1)));
    }

    #[test]
    fn a_reused_page_id_is_not_a_ghost_hit_once_its_ghost_is_removed() {
        let mut arc_replacer = ArcReplacer::new(1);
        arc_replacer.record_access(f(0), p(10));
        arc_replacer.set_evictable(f(0), true);
        assert_eq!(arc_replacer.evict(), Some(f(0)));
        assert_eq!(arc_replacer.mru_ghost_list.length, 1);

        assert_eq!(arc_replacer.remove(p(10)), None, "a ghost holds no frame");
        assert_eq!(arc_replacer.mru_ghost_list.length, 0);

        arc_replacer.record_access(f(0), p(10));
        assert_eq!(
            arc_replacer.nodes[arc_replacer.page_indices[&p(10)]].list,
            ArcList::Mru,
            "a reused id was admitted to MFU as if it were a ghost hit"
        );
        assert_eq!(arc_replacer.mru_target_size, 0);
    }

    #[test]
    fn test_sample() {
        let mut arc_replacer = ArcReplacer::new(7);
        arc_replacer.record_access(f(1), p(1));
        arc_replacer.record_access(f(2), p(2));
        arc_replacer.record_access(f(3), p(3));
        arc_replacer.record_access(f(4), p(4));
        arc_replacer.record_access(f(5), p(5));
        arc_replacer.record_access(f(6), p(6));
        arc_replacer.set_evictable(f(1), true);
        arc_replacer.set_evictable(f(2), true);
        arc_replacer.set_evictable(f(3), true);
        arc_replacer.set_evictable(f(4), true);
        arc_replacer.set_evictable(f(5), true);
        arc_replacer.set_evictable(f(6), false);
        assert_eq!(5, arc_replacer.size());

        arc_replacer.record_access(f(1), p(1));
        assert_eq!(Some(f(2)), arc_replacer.evict());
        assert_eq!(Some(f(3)), arc_replacer.evict());
        assert_eq!(Some(f(4)), arc_replacer.evict());
        assert_eq!(2, arc_replacer.size());

        arc_replacer.record_access(f(2), p(7));
        arc_replacer.set_evictable(f(2), true);
        arc_replacer.record_access(f(3), p(2));
        arc_replacer.set_evictable(f(3), true);
        assert_eq!(4, arc_replacer.size());

        arc_replacer.record_access(f(4), p(3));
        arc_replacer.set_evictable(f(4), true);
        arc_replacer.record_access(f(7), p(4));
        arc_replacer.set_evictable(f(7), true);
        assert_eq!(6, arc_replacer.size());

        assert_eq!(Some(f(5)), arc_replacer.evict());
        assert_eq!(Some(f(1)), arc_replacer.evict());

        arc_replacer.record_access(f(5), p(1));
        arc_replacer.set_evictable(f(5), true);
        assert_eq!(5, arc_replacer.size());

        assert_eq!(Some(f(2)), arc_replacer.evict());
    }

    #[test]
    fn test_sample_two() {
        let mut arc_replacer = ArcReplacer::new(3);
        arc_replacer.record_access(f(1), p(1));
        arc_replacer.set_evictable(f(1), true);
        arc_replacer.record_access(f(2), p(2));
        arc_replacer.set_evictable(f(2), true);
        arc_replacer.record_access(f(3), p(3));
        arc_replacer.set_evictable(f(3), true);
        assert_eq!(3, arc_replacer.size());

        assert_eq!(Some(f(1)), arc_replacer.evict());
        assert_eq!(Some(f(2)), arc_replacer.evict());
        assert_eq!(Some(f(3)), arc_replacer.evict());
        assert_eq!(0, arc_replacer.size());

        arc_replacer.record_access(f(3), p(4));
        arc_replacer.set_evictable(f(3), true);

        arc_replacer.record_access(f(2), p(1));
        arc_replacer.set_evictable(f(2), true);
        assert_eq!(2, arc_replacer.size());

        arc_replacer.record_access(f(1), p(3));
        arc_replacer.set_evictable(f(1), true);

        assert_eq!(Some(f(3)), arc_replacer.evict());
        assert_eq!(Some(f(2)), arc_replacer.evict());
        assert_eq!(Some(f(1)), arc_replacer.evict());

        arc_replacer.record_access(f(1), p(1));
        arc_replacer.set_evictable(f(1), true);

        arc_replacer.record_access(f(2), p(4));
        arc_replacer.set_evictable(f(2), true);

        arc_replacer.record_access(f(3), p(5));
        arc_replacer.set_evictable(f(3), true);
        assert_eq!(Some(f(1)), arc_replacer.evict());

        arc_replacer.record_access(f(1), p(6));
        arc_replacer.set_evictable(f(1), true);
        assert_eq!(Some(f(2)), arc_replacer.evict());

        arc_replacer.record_access(f(2), p(7));
        arc_replacer.set_evictable(f(2), true);
        assert_eq!(Some(f(3)), arc_replacer.evict());

        arc_replacer.record_access(f(3), p(5));
        arc_replacer.set_evictable(f(3), true);
        assert_eq!(Some(f(3)), arc_replacer.evict());

        arc_replacer.record_access(f(3), p(2));
        arc_replacer.set_evictable(f(3), true);
        assert_eq!(Some(f(1)), arc_replacer.evict());

        arc_replacer.record_access(f(1), p(3));
        arc_replacer.set_evictable(f(1), true);

        assert_eq!(Some(f(2)), arc_replacer.evict());
        assert_eq!(Some(f(3)), arc_replacer.evict());
        assert_eq!(Some(f(1)), arc_replacer.evict());
    }

    #[test]
    fn test_arc_replacer_performance() {
        let bpm_size: usize = 256 << 10;
        let mut arc_replacer = ArcReplacer::new(bpm_size);

        for i in 0..bpm_size {
            arc_replacer.record_access(f(i), p(i));
            arc_replacer.set_evictable(f(i), true);
        }

        let rounds: usize = 10;
        let mut access_frame_id: usize = 256 << 9;
        let mut access_times_ms: Vec<u128> = Vec::with_capacity(rounds);

        for _round in 0..rounds {
            let start_time = std::time::Instant::now();
            for _i in 0..bpm_size {
                arc_replacer.record_access(f(access_frame_id), p(access_frame_id));
                access_frame_id = (access_frame_id + 1) % bpm_size;
            }
            access_times_ms.push(start_time.elapsed().as_millis());
        }

        let total_ms: u128 = access_times_ms.iter().copied().sum();
        let avg_s: f64 = (total_ms as f64) / 1000.0 / (access_times_ms.len() as f64);

        assert!(avg_s < 3.0, "Average time used: {avg_s}s");
    }
}
