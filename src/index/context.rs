use std::collections::VecDeque;

use crate::{
    buffer_pool::{PageMut, PageRef},
    index::b_plus_page::INVALID_PAGE_ID,
};

pub struct TreeContext<'a> {
    pub header: Option<PageMut<'a>>,

    pub root_page_id: usize,

    pub read_set: VecDeque<PageRef<'a>>,

    pub write_set: VecDeque<PageMut<'a>>,
}

impl<'a> TreeContext<'a> {
    pub fn new() -> Self {
        TreeContext {
            header: None,
            root_page_id: INVALID_PAGE_ID,
            read_set: VecDeque::new(),
            write_set: VecDeque::new(),
        }
    }
    /// drops all ancestors, and current node
    pub fn drop_read_ancestors(&mut self) {
        while self.read_set.pop_front().is_some() {}
    }

    /// drops all ancestors, but keeps current node
    pub fn drop_write_ancestors(&mut self) {
        if self.write_set.len() > 1 {
            let leaf = self.write_set.pop_back().unwrap();
            while self.write_set.pop_front().is_some() {}
            self.write_set.push_back(leaf);
        }
        self.header = None;
    }
}
