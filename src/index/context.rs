use std::collections::VecDeque;

use crate::buffer_pool::PageMut;
use crate::index::b_plus_page::PageId;

/// Write latches held by a pessimistic insert or remove: the header, and the pages from the
/// highest latched ancestor down to the current page.
pub struct TreeContext<'a> {
    pub header: Option<PageMut<'a>>,
    pub root_page_id: PageId,
    pub write_set: VecDeque<PageMut<'a>>,
}

impl<'a> TreeContext<'a> {
    pub fn new(header: PageMut<'a>, root_page_id: PageId) -> Self {
        TreeContext {
            header: Some(header),
            root_page_id,
            write_set: VecDeque::new(),
        }
    }

    /// Makes `child` the current page. When `child_is_safe`, the operation cannot propagate
    /// above it, so every ancestor latch and the header latch are released.
    pub fn descend_into(&mut self, child: PageMut<'a>, child_is_safe: bool) {
        self.write_set.push_back(child);
        if child_is_safe {
            self.release_ancestors();
        }
    }

    fn release_ancestors(&mut self) {
        let current = self.write_set.pop_back();
        self.write_set.clear();
        self.write_set.extend(current);
        self.header = None;
    }
}
