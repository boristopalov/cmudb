use crate::buffer_pool::{BufferPoolManager, PageMut, PageRef};
use crate::catalog::index_schema::{IndexKey, IndexValue};
use crate::disk::PAGE_SIZE;
use crate::ids::RecordId;
use crate::index::b_plus_page::{
    Edge, HeaderPage, InternalPage, Layout, LeafPage, Page, PageId, PageKind,
};
use crate::index::context::TreeContext;
use crate::index::page_codec::{KIND_LEAF, PageCodecError, peek_common_header};
use crate::index::page_view::{
    HeaderPageView, InternalPageView, LeafInsertError, LeafPageView, NodeView,
};
use crate::index::{
    Index, IndexError, IndexIter, InsertError, InsertResult, RemoveError, RemoveResult,
};
use log::{debug, info, warn};

use std::collections::HashSet;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

const PRINT_TREE_LIMIT: usize = 10_000;

const LATCH_NOT_HELD: IndexError =
    IndexError::Structure("a page latch needed for restructuring was released during descent");
const HEADER_LATCH_RELEASED: IndexError =
    IndexError::Structure("the root changed after the header latch was released");
const EMPTY_SIBLING: IndexError =
    IndexError::Structure("sibling chosen to lend an entry has none to give");
const SPLIT_ABOVE_LATCHED_PATH: IndexError =
    IndexError::Structure("a split reached the highest latched page, which is not the root");
const NO_CHILD_FOR_KEY: IndexError =
    IndexError::Structure("internal page has no child to descend into for the key");
const ROOT_WITHOUT_FIRST_LEAF: IndexError =
    IndexError::Structure("the header names a root but no first leaf");

/// What rebalancing an underfull page did to it and its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rebalance {
    /// The page borrowed from its sibling, and the parent's separator between them changed.
    Redistributed,
    /// The page merged with its sibling, and the parent lost a separator.
    Merged,
    /// The page stayed underfull and the parent unchanged: the page has no sibling, or
    /// borrowing would have replaced the separator with a longer one that the parent has no
    /// room for.
    Kept,
}

struct Split {
    left_id: PageId,
    separator: IndexKey,
    right_id: PageId,
}

fn safe_for_insert(page: &NodeView<&[u8]>) -> bool {
    !page.may_split_on_insert()
}

fn safe_for_remove(page: &NodeView<&[u8]>) -> bool {
    !page.may_underflow_on_remove()
}

/// The end of a page's sibling that lends an entry to it.
fn lending_edge(sibling_is_right: bool) -> Edge {
    if sibling_is_right {
        Edge::First
    } else {
        Edge::Last
    }
}

/// The child of the internal page in `buf` to descend into for `key`, or `None` when `buf`
/// holds a leaf. A leaf is only told apart by its kind here; whoever reads it validates it.
fn child_for_key(buf: &[u8], layout: Layout, key: &IndexKey) -> Result<Option<PageId>, IndexError> {
    if peek_common_header(buf)? == KIND_LEAF {
        return Ok(None);
    }
    InternalPageView::new(buf, layout)?
        .child_for(&key.0)?
        .ok_or(NO_CHILD_FOR_KEY)
        .map(Some)
}

/// Moves `current` down to `child`; `child` is latched before `current` is released.
fn step_down<'a>(current: &mut PageRef<'a>, child: PageRef<'a>) {
    drop(std::mem::replace(current, child));
}

/// Disk-backed B+ tree mapping byte-string keys of up to a fixed maximum length to record ids.
///
/// The shape it maintains is checked by `assert_tree_is_well_formed` in this module's tests.
#[derive(Debug)]
pub struct BPlusTree {
    pub header_page_id: PageId,
    layout: Layout,
    index_name: String,
    bpm: Arc<BufferPoolManager>,
}

pub struct LeafPagesIter {
    bpm: Arc<BufferPoolManager>,
    layout: Layout,
    next_leaf_id: Option<PageId>,
    seen: HashSet<PageId>,
    finished: bool,
}

impl Iterator for LeafPagesIter {
    type Item = Result<(PageId, LeafPage), IndexError>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(leaf_id) = self.next_leaf_id.filter(|_| !self.finished) else {
            self.finished = true;
            return None;
        };
        if !self.seen.insert(leaf_id) {
            self.finished = true;
            return Some(Err(IndexError::PageError(PageCodecError::Malformed(
                "leaf chain contains a cycle",
            ))));
        }

        let page = match self.bpm.read_page(leaf_id) {
            Ok(page) => page,
            Err(e) => {
                self.finished = true;
                return Some(Err(IndexError::BpmError(e)));
            }
        };

        match LeafPage::decode(page.data(), self.layout) {
            Ok(leaf) => {
                self.next_leaf_id = leaf.next_leaf_id();
                Some(Ok((leaf_id, leaf)))
            }
            Err(e) => {
                self.finished = true;
                Some(Err(IndexError::PageError(e)))
            }
        }
    }
}

impl BPlusTree {
    /// An empty tree of keys up to `max_key_len` bytes long. Fails with
    /// [`IndexError::UnsupportedKeyLength`] for zero or a length past
    /// [`MAX_KEY_LEN`](crate::index::b_plus_page::MAX_KEY_LEN).
    pub fn new(
        index_name: String,
        max_key_len: u32,
        bpm: Arc<BufferPoolManager>,
    ) -> Result<Self, IndexError> {
        Self::with_layout(index_name, Self::layout_for(max_key_len)?, bpm)
    }

    /// Like [`BPlusTree::new`], over pages of `layout`.
    pub(crate) fn with_layout(
        index_name: String,
        layout: Layout,
        bpm: Arc<BufferPoolManager>,
    ) -> Result<Self, IndexError> {
        let max_key_len = u32::try_from(layout.max_key_len())
            .map_err(|_| IndexError::UnsupportedKeyLength(u32::MAX))?;
        let header_page_id = {
            let mut page = bpm.new_page()?;
            let pid = page.page_id();
            HeaderPageView::init(page.data_mut(), max_key_len)?;
            pid
        };
        info!("created new b+ tree; header_page_id={header_page_id}");

        Ok(Self {
            index_name,
            layout,
            header_page_id,
            bpm,
        })
    }

    /// The tree whose header page is `header_page_id`, as an earlier [`BPlusTree::new`] over
    /// the same data file left it. Fails with [`IndexError::MaxKeyLength`] when the header was
    /// written for another maximum key length.
    pub fn open(
        index_name: String,
        max_key_len: u32,
        header_page_id: PageId,
        bpm: Arc<BufferPoolManager>,
    ) -> Result<Self, IndexError> {
        let tree = Self {
            index_name,
            layout: Self::layout_for(max_key_len)?,
            header_page_id,
            bpm,
        };
        tree.read_header()?;
        Ok(tree)
    }

    /// The page layout for keys up to `max_key_len` bytes. Fails with
    /// [`IndexError::UnsupportedKeyLength`] when pages cannot hold enough of them.
    fn layout_for(max_key_len: u32) -> Result<Layout, IndexError> {
        Layout::new(max_key_len).map_err(|_| IndexError::UnsupportedKeyLength(max_key_len))
    }

    /// The longest key the tree holds.
    pub fn max_key_len(&self) -> usize {
        self.layout.max_key_len()
    }

    /// Acquires a read latch on the header page and copies out its fields.
    fn read_header(&self) -> Result<HeaderPage, IndexError> {
        let page = self.bpm.read_page(self.header_page_id)?;
        self.header_from_bytes(page.data())
    }

    fn header_from_bytes(&self, buf: &[u8]) -> Result<HeaderPage, IndexError> {
        let view = HeaderPageView::new(buf)?;
        let actual = view.max_key_len() as usize;
        if actual != self.layout.max_key_len() {
            return Err(IndexError::MaxKeyLength {
                expected: self.layout.max_key_len(),
                actual,
            });
        }
        Ok(HeaderPage {
            root_page_id: view.root_page_id()?,
            first_leaf_id: view.first_leaf_id()?,
            max_key_len: view.max_key_len(),
        })
    }

    /// Writes `header`'s root and first-leaf pointers into the latched header page in place.
    fn store_header(page: &mut PageMut<'_>, header: &HeaderPage) -> Result<(), IndexError> {
        let mut view = HeaderPageView::new(page.data_mut())?;
        view.set_root_page_id(header.root_page_id);
        view.set_first_leaf_id(header.first_leaf_id);
        Ok(())
    }

    /// Fails with [`IndexError::KeyTooLong`] for a key longer than the tree holds.
    fn check_key_len(&self, key: &IndexKey) -> Result<(), IndexError> {
        if key.0.len() <= self.layout.max_key_len() {
            Ok(())
        } else {
            Err(IndexError::KeyTooLong {
                max: self.layout.max_key_len(),
                actual: key.0.len(),
            })
        }
    }

    fn decode_leaf(&self, page_id: PageId) -> Result<LeafPage, IndexError> {
        let page = self.bpm.read_page(page_id)?;
        Ok(LeafPage::decode(page.data(), self.layout)?)
    }

    fn allocate_page(&self) -> Result<(PageId, PageMut<'_>), IndexError> {
        let page = self.bpm.new_page()?;
        Ok((page.page_id(), page))
    }

    fn new_leaf(&self) -> LeafPage {
        LeafPage::new(self.layout)
    }

    /// Write-latches the path from the root to the leaf for `key`, releasing ancestors whenever
    /// a child is `is_safe`. The leaf ends up as the last page in `ctx.write_set`.
    fn latch_path_to_leaf<'a>(
        &'a self,
        ctx: &mut TreeContext<'a>,
        key: &IndexKey,
        is_safe: fn(&NodeView<&[u8]>) -> bool,
    ) -> Result<(), IndexError> {
        ctx.write_set
            .push_back(self.bpm.write_page(ctx.root_page_id)?);
        loop {
            let current = ctx.write_set.back().ok_or(LATCH_NOT_HELD)?;
            let Some(child_id) = child_for_key(current.data(), self.layout, key)? else {
                return Ok(());
            };
            let child = self.bpm.write_page(child_id)?;
            let child_is_safe = is_safe(&NodeView::new(child.data(), self.layout)?);
            ctx.descend_into(child, child_is_safe);
        }
    }

    fn insert_pessimistic(&self, key: IndexKey, record_id: RecordId) -> InsertResult {
        let mut header_page = self.bpm.write_page(self.header_page_id)?;
        let mut header = self.header_from_bytes(header_page.data())?;

        let Some(root_page_id) = header.root_page_id else {
            info!("insert into empty tree: creating new tree");
            let (leaf_id, mut leaf_page) = self.allocate_page()?;
            let mut leaf = self.new_leaf();
            leaf.insert(key, record_id)
                .map_err(|_| IndexError::Insert(InsertError::DuplicateKey))?;
            leaf.encode(leaf_page.data_mut())?;
            header.root_page_id = Some(leaf_id);
            header.first_leaf_id = Some(leaf_id);
            Self::store_header(&mut header_page, &header)?;
            return Ok(());
        };

        let mut ctx = TreeContext::new(header_page, root_page_id);
        self.latch_path_to_leaf(&mut ctx, &key, safe_for_insert)?;
        let mut leaf_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
        if self.insert_in_place(&mut leaf_page, &key, record_id)? {
            return Ok(());
        }
        let mut leaf = LeafPage::decode(leaf_page.data(), self.layout)?;
        leaf.insert(key, record_id)
            .map_err(|_| IndexError::Insert(InsertError::DuplicateKey))?;
        if !leaf.is_overfull() {
            leaf.encode(leaf_page.data_mut())?;
            return Ok(());
        }

        let (right_id, mut right_page) = self.allocate_page()?;
        let mut right_leaf = self.new_leaf();
        let separator = leaf.split_into(&mut right_leaf, right_id);
        leaf.encode(leaf_page.data_mut())?;
        right_leaf.encode(right_page.data_mut())?;
        let mut split = Split {
            left_id: leaf_page.page_id(),
            separator,
            right_id,
        };

        while let Some(mut parent_page) = ctx.write_set.pop_back() {
            let mut parent = InternalPage::decode(parent_page.data_mut(), self.layout)?;
            parent.insert_separator(split.separator, split.right_id);
            if !parent.is_overfull() {
                parent.encode(parent_page.data_mut())?;
                return Ok(());
            }

            let (right_id, mut right_page) = self.allocate_page()?;
            let mut right = InternalPage::new(self.layout);
            let separator = parent.split_into(&mut right);
            parent.encode(parent_page.data_mut())?;
            right.encode(right_page.data_mut())?;
            split = Split {
                left_id: parent_page.page_id(),
                separator,
                right_id,
            };
        }

        self.grow_new_root(&mut ctx, &mut header, split)
    }

    fn grow_new_root(
        &self,
        ctx: &mut TreeContext<'_>,
        header: &mut HeaderPage,
        split: Split,
    ) -> InsertResult {
        if split.left_id != ctx.root_page_id {
            return Err(SPLIT_ABOVE_LATCHED_PATH);
        }
        let header_page = ctx.header.as_mut().ok_or(HEADER_LATCH_RELEASED)?;
        let (root_id, mut root_page) = self.allocate_page()?;
        let root =
            InternalPage::new_root(self.layout, split.left_id, split.separator, split.right_id);
        root.encode(root_page.data_mut())?;
        header.root_page_id = Some(root_id);
        Self::store_header(header_page, header)?;
        Ok(())
    }

    /// Inserts into the latched leaf's bytes without decoding it. Returns `Ok(false)` when the
    /// leaf has no room, leaving it untouched for the caller to split.
    fn insert_in_place(
        &self,
        leaf_page: &mut PageMut<'_>,
        key: &IndexKey,
        record_id: RecordId,
    ) -> Result<bool, IndexError> {
        match LeafPageView::new(leaf_page.data_mut(), self.layout)?.insert(&key.0, record_id) {
            Ok(()) => Ok(true),
            Err(LeafInsertError::DuplicateKey) => {
                Err(IndexError::Insert(InsertError::DuplicateKey))
            }
            Err(LeafInsertError::Full) => Ok(false),
        }
    }

    fn remove_pessimistic(&self, key: IndexKey) -> RemoveResult {
        let header_page = self.bpm.write_page(self.header_page_id)?;
        let mut header = self.header_from_bytes(header_page.data())?;
        let Some(root_page_id) = header.root_page_id else {
            return Err(IndexError::Remove(RemoveError::KeyNotFound));
        };

        let mut ctx = TreeContext::new(header_page, root_page_id);
        self.latch_path_to_leaf(&mut ctx, &key, safe_for_remove)?;
        let mut leaf_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
        if self.remove_in_place_if_final(&mut ctx, &mut header, &mut leaf_page, &key)? {
            return Ok(());
        }
        let mut leaf = LeafPage::decode(leaf_page.data(), self.layout)?;
        leaf.remove_key(&key)?;
        if !leaf.is_underfull() {
            leaf.encode(leaf_page.data_mut())?;
            return Ok(());
        }

        let mut parent_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
        let mut parent = InternalPage::decode(parent_page.data_mut(), self.layout)?;
        let mut outcome = self.rebalance_leaf(&key, leaf, &mut leaf_page, &mut parent)?;
        parent.encode(parent_page.data_mut())?;

        while outcome != Rebalance::Kept {
            if parent_page.page_id() == ctx.root_page_id {
                if let [only_child] = parent.children() {
                    header.root_page_id = Some(*only_child);
                    let header_page = ctx.header.as_mut().ok_or(HEADER_LATCH_RELEASED)?;
                    Self::store_header(header_page, &header)?;
                }
                return Ok(());
            }
            if !parent.is_underfull() {
                return Ok(());
            }
            let mut grandparent_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
            let mut grandparent = InternalPage::decode(grandparent_page.data_mut(), self.layout)?;
            outcome = self.rebalance_internal(&key, parent, &mut parent_page, &mut grandparent)?;
            grandparent.encode(grandparent_page.data_mut())?;
            parent_page = grandparent_page;
            parent = grandparent;
        }
        Ok(())
    }

    /// Removes `key` from the latched leaf's bytes when the leaf is the root or cannot become
    /// underfull. For a root leaf, the header page that an emptied root updates is latched and
    /// validated before the leaf changes. Returns `Ok(false)` without touching the leaf otherwise,
    /// so the caller removes from a decoded copy and writes it back once rebalancing has latched
    /// every page it needs.
    fn remove_in_place_if_final(
        &self,
        ctx: &mut TreeContext<'_>,
        header: &mut HeaderPage,
        leaf_page: &mut PageMut<'_>,
        key: &IndexKey,
    ) -> Result<bool, IndexError> {
        let leaf_is_root = leaf_page.page_id() == ctx.root_page_id;
        let mut leaf = LeafPageView::new(leaf_page.data_mut(), self.layout)?;
        if !leaf_is_root {
            if leaf.may_underflow_on_remove() {
                return Ok(false);
            }
            leaf.remove_key(&key.0)?;
            return Ok(true);
        }
        let header_page = ctx.header.as_mut().ok_or(HEADER_LATCH_RELEASED)?;
        let mut header_view = HeaderPageView::new(header_page.data_mut())?;
        leaf.remove_key(&key.0)?;
        if leaf.size() == 0 {
            header.root_page_id = None;
            header.first_leaf_id = None;
            header_view.set_root_page_id(None);
            header_view.set_first_leaf_id(None);
        }
        Ok(true)
    }

    /// The index and id of the sibling an underfull child borrows from or merges with, or
    /// `None` when the parent has no other child, which only a parent that rebalancing left
    /// underfull can have.
    fn sibling_of(parent: &InternalPage, child_idx: usize) -> Option<(usize, PageId)> {
        let sibling_idx = if child_idx == 0 { 1 } else { child_idx - 1 };
        let sibling_id = parent.get_child(sibling_idx).copied()?;
        Some((sibling_idx, sibling_id))
    }

    fn rebalance_leaf(
        &self,
        key: &IndexKey,
        mut leaf: LeafPage,
        leaf_page: &mut PageMut<'_>,
        parent: &mut InternalPage,
    ) -> Result<Rebalance, IndexError> {
        let leaf_idx = parent.find_child_index(key);
        let Some((sibling_idx, sibling_id)) = Self::sibling_of(parent, leaf_idx) else {
            leaf.encode(leaf_page.data_mut())?;
            return Ok(Rebalance::Kept);
        };
        let mut sibling_page = self.bpm.write_page(sibling_id)?;
        let mut sibling = LeafPage::decode(sibling_page.data_mut(), self.layout)?;
        let sibling_is_right = sibling_idx > leaf_idx;
        let edge = lending_edge(sibling_is_right);
        let leaf_before_borrowing = leaf.clone();

        while leaf.is_underfull() && sibling.can_lend(edge) {
            let borrowed = if sibling_is_right {
                sibling.pop_first()
            } else {
                sibling.pop_last()
            };
            let (k, rid, tombstoned) = borrowed.ok_or(EMPTY_SIBLING)?;
            if tombstoned && leaf.tombstones_full() {
                continue;
            }
            if sibling_is_right {
                leaf.push_back(k, rid, tombstoned);
            } else {
                leaf.push_front(k, rid, tombstoned);
            }
        }

        if leaf.is_underfull() {
            if sibling_is_right {
                leaf.absorb_right(&mut sibling);
                parent.remove_separator(leaf_idx);
                leaf.encode(leaf_page.data_mut())?;
            } else {
                sibling.absorb_right(&mut leaf);
                parent.remove_separator(sibling_idx);
                sibling.encode(sibling_page.data_mut())?;
            }
            return Ok(Rebalance::Merged);
        }

        let (separator_idx, new_separator) = if sibling_is_right {
            (leaf_idx, sibling.first_key().ok_or(EMPTY_SIBLING)?.clone())
        } else {
            (sibling_idx, leaf.first_key().ok_or(EMPTY_SIBLING)?.clone())
        };
        if !parent.fits_replacing(separator_idx, &new_separator) {
            leaf_before_borrowing.encode(leaf_page.data_mut())?;
            return Ok(Rebalance::Kept);
        }
        parent.replace_key(separator_idx, new_separator);
        sibling.encode(sibling_page.data_mut())?;
        leaf.encode(leaf_page.data_mut())?;
        Ok(Rebalance::Redistributed)
    }

    fn rebalance_internal(
        &self,
        key: &IndexKey,
        mut node: InternalPage,
        node_page: &mut PageMut<'_>,
        parent: &mut InternalPage,
    ) -> Result<Rebalance, IndexError> {
        let node_idx = parent.find_child_index(key);
        let Some((sibling_idx, sibling_id)) = Self::sibling_of(parent, node_idx) else {
            return Ok(Rebalance::Kept);
        };
        let mut sibling_page = self.bpm.write_page(sibling_id)?;
        let mut sibling = InternalPage::decode(sibling_page.data_mut(), self.layout)?;
        let sibling_is_right = sibling_idx > node_idx;

        if !sibling.can_lend(lending_edge(sibling_is_right)) {
            if sibling_is_right {
                let separator = parent.keys()[node_idx].clone();
                node.absorb_right(separator, &mut sibling);
                parent.remove_separator(node_idx);
                node.encode(node_page.data_mut())?;
            } else {
                let separator = parent.keys()[sibling_idx].clone();
                sibling.absorb_right(separator, &mut node);
                parent.remove_separator(sibling_idx);
                sibling.encode(sibling_page.data_mut())?;
            }
            return Ok(Rebalance::Merged);
        }

        let (separator_idx, rotated_key) = if sibling_is_right {
            (node_idx, sibling.keys().first())
        } else {
            (sibling_idx, sibling.keys().last())
        };
        if !parent.fits_replacing(separator_idx, rotated_key.ok_or(EMPTY_SIBLING)?) {
            return Ok(Rebalance::Kept);
        }
        if sibling_is_right {
            let (rotated_key, rotated_child) = sibling.pop_first_child_and_key();
            let old_separator = parent.replace_key(node_idx, rotated_key);
            node.append_key_and_child(old_separator, rotated_child);
        } else {
            let (rotated_key, rotated_child) = sibling.pop_last_child_and_key();
            let old_separator = parent.replace_key(sibling_idx, rotated_key);
            node.prepend_child_and_key(old_separator, rotated_child);
        }
        sibling.encode(sibling_page.data_mut())?;
        node.encode(node_page.data_mut())?;
        Ok(Rebalance::Redistributed)
    }

    pub fn leaf_pages(&self) -> Result<LeafPagesIter, IndexError> {
        let header = self.read_header()?;
        Ok(LeafPagesIter {
            bpm: self.bpm.clone(),
            layout: self.layout,
            next_leaf_id: header.first_leaf_id,
            seen: std::collections::HashSet::new(),
            finished: false,
        })
    }

    /// Prints a best-effort view of the tree structure (page ids, levels, counts, and leaf links).
    pub fn print_tree(&self) -> Result<(), IndexError> {
        let header = self.read_header()?;
        fn fmt_pid(pid: impl Into<Option<PageId>>) -> String {
            match pid.into() {
                Some(pid) => pid.to_string(),
                None => "INVALID".to_string(),
            }
        }
        fn fmt_children(children: &[PageId]) -> String {
            let mut s = String::from("[");
            for (i, &pid) in children.iter().enumerate() {
                if i != 0 {
                    s.push_str(", ");
                }
                s.push_str(&fmt_pid(pid));
            }
            s.push(']');
            s
        }

        println!(
            "BPlusTree {{ index_name: {}, header_page_id: {}, max_key_len: {} }}",
            self.index_name,
            fmt_pid(self.header_page_id),
            self.layout.max_key_len(),
        );
        println!(
            "HeaderPage {{ root_page_id: {}, first_leaf_id: {} }}",
            fmt_pid(header.root_page_id),
            fmt_pid(header.first_leaf_id)
        );

        let Some(root_page_id) = header.root_page_id else {
            println!("(empty)");
            return Ok(());
        };

        println!("tree:");
        let mut visited: HashSet<PageId> = HashSet::new();
        let mut nodes_printed = 0usize;

        fn print_node(
            tree: &BPlusTree,
            page_id: PageId,
            prefix: &str,
            is_last: bool,
            is_root: bool,
            visited: &mut HashSet<PageId>,
            nodes_printed: &mut usize,
        ) -> Result<(), IndexError> {
            if *nodes_printed >= PRINT_TREE_LIMIT {
                println!("{prefix}... truncated after {nodes_printed} nodes");
                return Ok(());
            }
            *nodes_printed += 1;

            let branch = if is_root {
                ""
            } else if is_last {
                "└── "
            } else {
                "├── "
            };

            if !visited.insert(page_id) {
                println!("{prefix}{branch}* cycle/revisit: page {}", fmt_pid(page_id));
                return Ok(());
            }

            let page = tree.bpm.read_page(page_id)?;
            match LeafPage::decode(page.data(), tree.layout) {
                Ok(leaf) => {
                    println!(
                        "{prefix}{branch}leaf page_id={} keys={} next_leaf_id={}",
                        fmt_pid(page_id),
                        leaf.len(),
                        fmt_pid(leaf.next_leaf_id())
                    );
                    Ok(())
                }
                Err(PageCodecError::WrongPageKind { .. }) => {
                    match InternalPage::decode(page.data(), tree.layout) {
                        Ok(internal) => {
                            let children = internal.children().to_vec();
                            println!(
                                "{prefix}{branch}internal page_id={} keys={} children={}",
                                fmt_pid(page_id),
                                internal.len(),
                                fmt_children(&children)
                            );

                            let child_prefix = if is_root {
                                String::new()
                            } else if is_last {
                                format!("{prefix}    ")
                            } else {
                                format!("{prefix}│   ")
                            };

                            for (i, child) in children.iter().copied().enumerate() {
                                let child_is_last = i + 1 == children.len();
                                print_node(
                                    tree,
                                    child,
                                    &child_prefix,
                                    child_is_last,
                                    false,
                                    visited,
                                    nodes_printed,
                                )?;
                            }
                            Ok(())
                        }
                        Err(PageCodecError::WrongPageKind { .. }) => {
                            println!(
                                "{prefix}{branch}unknown page_id={} (neither leaf nor internal)",
                                fmt_pid(page_id)
                            );
                            Ok(())
                        }
                        Err(e) => Err(IndexError::PageError(e)),
                    }
                }
                Err(e) => Err(IndexError::PageError(e)),
            }
        }

        print_node(
            self,
            root_page_id,
            "",
            true,
            false,
            &mut visited,
            &mut nodes_printed,
        )?;

        println!("leaf chain:");
        let mut leaf_seen: HashSet<PageId> = HashSet::new();
        let mut leaf_id = header.first_leaf_id;
        let mut idx = 0usize;
        while let Some(current_leaf_id) = leaf_id {
            if idx >= PRINT_TREE_LIMIT {
                println!("  ... truncated after {idx} leaves");
                break;
            }
            if !leaf_seen.insert(current_leaf_id) {
                println!("  * cycle/revisit at leaf {current_leaf_id}");
                break;
            }

            let leaf = self.decode_leaf(current_leaf_id)?;
            println!(
                "  [{idx}] page_id={} keys={} next_leaf_id={}",
                current_leaf_id,
                leaf.len(),
                fmt_pid(leaf.next_leaf_id())
            );
            leaf_id = leaf.next_leaf_id();
            idx += 1;
        }

        Ok(())
    }

    /// Read-latches the root, crabbing from the header: the header's read latch is held until
    /// the root is latched, so a split or collapse cannot replace the root in between. Returns
    /// `None` for an empty tree.
    fn get_root_page_from_header(&self) -> Result<Option<PageRef<'_>>, IndexError> {
        let mut current = self.bpm.read_page(self.header_page_id)?;
        let header = self.header_from_bytes(current.data())?;
        let Some(root_page_id) = header.root_page_id else {
            warn!(
                "tid={:?} header root_page_id is invalid (empty tree); header={:?}",
                std::thread::current().id(),
                header
            );
            return Ok(None);
        };
        let root = self.bpm.read_page(root_page_id)?;
        step_down(&mut current, root);
        Ok(Some(current))
    }

    /// Read-crabs to the leaf for `key` and write-latches it. Returns `None` when the tree is
    /// empty or its root is a leaf, cases that only the pessimistic paths handle.
    fn latch_leaf_for_write(&self, key: &IndexKey) -> Result<Option<PageMut<'_>>, IndexError> {
        let Some(mut current) = self.get_root_page_from_header()? else {
            return Ok(None);
        };
        loop {
            let Some(child_id) = child_for_key(current.data(), self.layout, key)? else {
                return Ok(None);
            };
            let child = self.bpm.read_page(child_id)?;
            if matches!(Page::kind(child.data())?, PageKind::Leaf) {
                return self.relatch_child_for_write(&current, child).map(Some);
            }
            step_down(&mut current, child);
        }
    }

    /// Trades `child`'s read latch for a write latch while `_parent_latch` stays held, so no
    /// split or merge can reach the child in between.
    fn relatch_child_for_write<'a>(
        &'a self,
        _parent_latch: &PageRef<'a>,
        child: PageRef<'a>,
    ) -> Result<PageMut<'a>, IndexError> {
        let child_id = child.page_id();
        drop(child);
        Ok(self.bpm.write_page(child_id)?)
    }

    /// find_leaf looks for the leaf page that would contain `key`, if `key` exists in our tree,
    /// and returns it still read-latched so the caller can read it through a `LeafPageView`.
    /// This is useful for index scans, where we use a key as a lower bound. The key does not need
    /// to exist to be used as a bound.
    fn find_leaf(&self, key: &IndexKey) -> Result<Option<PageRef<'_>>, IndexError> {
        let Some(mut current) = self.get_root_page_from_header()? else {
            return Ok(None);
        };
        while let Some(child_id) = child_for_key(current.data(), self.layout, key)? {
            let child = self.bpm.read_page(child_id)?;
            step_down(&mut current, child);
        }
        Ok(Some(current))
    }
}

impl Index for BPlusTree {
    fn insert(&self, key: IndexKey, val: IndexValue) -> InsertResult {
        info!(
            "b_plus_tree inserting key with len {} and value {:?}: ",
            key.0.len(),
            val
        );
        self.check_key_len(&key)?;
        let IndexValue::IndexValue(record_id) = val;

        if let Some(mut leaf_page) = self.latch_leaf_for_write(&key)?
            && self.insert_in_place(&mut leaf_page, &key, record_id)?
        {
            return Ok(());
        }
        self.insert_pessimistic(key, record_id)
    }

    fn remove(&self, key: IndexKey) -> RemoveResult {
        info!("b_plus_tree removing key {:?}", key);
        if key.0.len() > self.layout.max_key_len() {
            return Err(IndexError::Remove(RemoveError::KeyNotFound));
        }

        if let Some(mut leaf_page) = self.latch_leaf_for_write(&key)? {
            let mut leaf = LeafPageView::new(leaf_page.data_mut(), self.layout)?;
            if !leaf.may_underflow_on_remove() {
                leaf.remove_key(&key.0)?;
                return Ok(());
            }
            if !leaf.key_exists(&key.0) {
                return Err(IndexError::Remove(RemoveError::KeyNotFound));
            }
        }
        self.remove_pessimistic(key)
    }

    fn search(&self, key: &IndexKey) -> Result<Option<IndexValue>, IndexError> {
        debug!(
            "tid={:?} search(): searching for {:?}",
            std::thread::current().id(),
            key
        );
        if key.0.len() > self.layout.max_key_len() {
            return Ok(None);
        }
        let Some(page) = self.find_leaf(key)? else {
            return Ok(None);
        };
        let leaf = LeafPageView::new(page.data(), self.layout)?;
        Ok(leaf.get(&key.0)?.map(IndexValue::IndexValue))
    }

    fn scan(&self, range: (Bound<IndexKey>, Bound<IndexKey>)) -> Result<IndexIter, IndexError> {
        let header = self.read_header()?;
        if header.root_page_id.is_none() {
            return Ok(Box::new(std::iter::empty()));
        }
        let start_page = match range.start_bound() {
            Bound::Unbounded => {
                let first_leaf_id = header.first_leaf_id.ok_or(ROOT_WITHOUT_FIRST_LEAF)?;
                Some(self.bpm.read_page(first_leaf_id)?)
            }
            Bound::Excluded(key) | Bound::Included(key) => self.find_leaf(key)?,
        };
        let Some(start_page) = start_page else {
            return Ok(Box::new(std::iter::empty()));
        };
        let start_leaf_id = start_page.page_id();
        let leaf = LeafPageView::new(start_page.data(), self.layout)?;
        let start_slot = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Excluded(key) => leaf.upper_bound(&key.0),
            Bound::Included(key) => leaf.lower_bound(&key.0),
        };
        let mut buf: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        buf.copy_from_slice(start_page.data());

        Ok(Box::new(BPlusTreeIter {
            bpm: self.bpm.clone(),
            layout: self.layout,
            leaf: Some(buf),
            seen: HashSet::from([start_leaf_id]),
            slot_idx: start_slot,
            upper_bound: range.end_bound().cloned(),
        }))
    }
}

/// Range scan over the leaf chain. The iterator outlives any latch the scan could hold, so it
/// keeps a private copy of the current leaf's bytes, reusing one buffer for every leaf, and
/// reads entries through a `LeafPageView` over that copy.
pub struct BPlusTreeIter {
    bpm: Arc<BufferPoolManager>,
    layout: Layout,
    /// Copy of the current leaf's bytes; `None` once the scan is finished.
    leaf: Option<Box<[u8; PAGE_SIZE]>>,
    seen: HashSet<PageId>,
    slot_idx: usize,
    upper_bound: Bound<IndexKey>,
}

impl BPlusTreeIter {
    /// Copies leaf `leaf_id` into the iterator's buffer. Returns `Ok(false)` if `leaf_id` is
    /// the end of the chain.
    fn load_leaf(&mut self, leaf_id: Option<PageId>) -> Result<bool, IndexError> {
        let Some(leaf_id) = leaf_id else {
            return Ok(false);
        };
        if !self.seen.insert(leaf_id) {
            return Err(IndexError::PageError(PageCodecError::Malformed(
                "leaf chain contains a cycle",
            )));
        }
        let page = self.bpm.read_page(leaf_id)?;
        LeafPageView::new(page.data(), self.layout)?;
        if let Some(buf) = self.leaf.as_mut() {
            buf.copy_from_slice(page.data());
        }
        Ok(true)
    }
}

impl Iterator for BPlusTreeIter {
    type Item = Result<(IndexKey, IndexValue), IndexError>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let buf = self.leaf.as_ref()?;
            let leaf = match LeafPageView::new(&buf[..], self.layout) {
                Ok(leaf) => leaf,
                Err(e) => {
                    self.leaf = None;
                    return Some(Err(IndexError::PageError(e)));
                }
            };

            if self.slot_idx >= leaf.size() as usize {
                let next_leaf_id = leaf.next_leaf_id().map_err(IndexError::PageError);
                match next_leaf_id.and_then(|id| self.load_leaf(id)) {
                    Ok(true) => {
                        self.slot_idx = 0;
                        continue;
                    }
                    Ok(false) => {
                        self.leaf = None;
                        return None;
                    }
                    Err(e) => {
                        self.leaf = None;
                        return Some(Err(e));
                    }
                }
            }

            let i = self.slot_idx;
            self.slot_idx += 1;

            if leaf.is_tombstoned(i as u32) {
                continue;
            }

            let key = leaf.key(i);
            let past_upper = match &self.upper_bound {
                Bound::Included(u) => key > &u.0[..],
                Bound::Excluded(u) => key >= &u.0[..],
                Bound::Unbounded => false,
            };
            if past_upper {
                self.leaf = None;
                return None;
            }

            let rid = match leaf.record_id(i) {
                Ok(rid) => rid,
                Err(e) => {
                    self.leaf = None;
                    return Some(Err(IndexError::PageError(e)));
                }
            };
            return Some(Ok((IndexKey(key.to_vec()), IndexValue::IndexValue(rid))));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buf::{get_u16_le, put_u32_le};
    use crate::index::b_plus_page::MAX_KEY_LEN;

    use crate::ids::RecordId;
    use crate::testing::pid;

    const FOUR_ENTRY_LEAF_KEY_LEN: u32 = 2000;

    const NO_TOMBSTONE_KEY_LEN: u32 = 567;

    /// How many entries of the longest key fit a leaf and an internal page of `tree`.
    fn longest_entries_per_page(tree: &BPlusTree) -> (usize, usize) {
        let l = tree.layout;
        (
            l.leaf_area().capacity() / l.leaf_max_entry_bytes(),
            l.internal_area().capacity() / l.internal_max_entry_bytes(),
        )
    }

    /// A tree of `NO_TOMBSTONE_KEY_LEN`-byte keys whose leaves keep no tombstones.
    fn no_tombstone_tree(bpm: &Arc<BufferPoolManager>) -> BPlusTree {
        let layout = Layout::with_tombstone_capacity(NO_TOMBSTONE_KEY_LEN, 0).unwrap();
        BPlusTree::with_layout("idx".to_string(), layout, bpm.clone()).unwrap()
    }

    #[test]
    fn no_tombstone_layout_gives_thirteen_entry_leaves_without_tombstones() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let tree = no_tombstone_tree(&bpm);
        assert_eq!(longest_entries_per_page(&tree).0, 13);
        assert_eq!(tree.layout.tombstone_capacity(), 0);
    }

    #[test]
    fn failed_rebalance_leaves_the_underfull_leaf_unchanged() {
        let key_len = NO_TOMBSTONE_KEY_LEN as usize;
        let (bpm, _tmp) = crate::testing::bpm(32);
        let tree = no_tombstone_tree(&bpm);
        for k in 0..15u8 {
            let rid = RecordId {
                page_id: pid(u32::from(k)),
                slot_id: 0,
            };
            tree.insert(make_key(k, key_len), IndexValue::IndexValue(rid))
                .unwrap();
        }
        let header = tree.read_header().unwrap();
        let root = InternalPage::decode(
            bpm.read_page(header.root_page_id.unwrap()).unwrap().data(),
            tree.layout,
        )
        .unwrap();
        assert_eq!(
            root.children().len(),
            2,
            "15 inserts split the root leaf once"
        );
        let (left_id, right_id) = (root.children()[0], root.children()[1]);
        let left = tree.decode_leaf(left_id).unwrap();
        assert!(
            !left.can_lend(Edge::Last),
            "the left leaf becomes underfull when it loses an entry"
        );
        let left_before = bpm.read_page(left_id).unwrap().data().to_vec();

        bpm.write_page(right_id).unwrap().data_mut()[0] ^= 0xff;
        let removed = tree.remove(make_key(0, key_len));
        assert!(
            matches!(
                removed,
                Err(IndexError::PageError(PageCodecError::BadMagic))
            ),
            "remove fails while decoding the corrupt sibling, got {removed:?}"
        );
        assert_eq!(
            bpm.read_page(left_id).unwrap().data(),
            &left_before[..],
            "the failed remove leaves the leaf's bytes untouched"
        );
        assert!(
            tree.search(&make_key(0, key_len)).unwrap().is_some(),
            "the key survives the failed remove"
        );

        bpm.write_page(right_id).unwrap().data_mut()[0] ^= 0xff;
        tree.remove(make_key(0, key_len)).unwrap();
        assert!(tree.search(&make_key(0, key_len)).unwrap().is_none());
    }

    #[test]
    fn a_corrupt_tombstone_makes_every_operation_fail_without_panicking() {
        let key_len = 8usize;
        let (bpm, _tmp) = crate::testing::bpm(8);
        let tree = BPlusTree::new("idx".to_string(), key_len as u32, bpm.clone()).unwrap();
        for k in 1..=3u8 {
            let rid = RecordId {
                page_id: pid(u32::from(k)),
                slot_id: 0,
            };
            tree.insert(make_key(k, key_len), IndexValue::IndexValue(rid))
                .unwrap();
        }
        tree.remove(make_key(2, key_len)).unwrap();
        let root_leaf = tree.read_header().unwrap().root_page_id.unwrap();
        let first_tombstone = tree.layout.leaf_tombstone_offset(0);
        put_u32_le(
            bpm.write_page(root_leaf).unwrap().data_mut(),
            first_tombstone,
            40,
        );

        let corrupt = |r: &Result<_, IndexError>| {
            matches!(r, Err(IndexError::PageError(PageCodecError::Malformed(_))))
        };
        assert!(corrupt(&tree.search(&make_key(1, key_len)).map(|_| ())));
        let rid = RecordId {
            page_id: pid(9),
            slot_id: 0,
        };
        assert!(corrupt(
            &tree.insert(make_key(9, key_len), IndexValue::IndexValue(rid))
        ));
        assert!(corrupt(&tree.remove(make_key(1, key_len)).map(|_| ())));
        assert!(corrupt(
            &tree
                .scan((Bound::Unbounded, Bound::Unbounded))
                .and_then(|it| it.collect::<Result<Vec<_>, _>>())
                .map(|_| ())
        ));
    }

    /// A tree of two leaves with no tombstones, returning the tree and the two leaf ids.
    fn two_leaf_tree(bpm: &Arc<BufferPoolManager>) -> (BPlusTree, PageId, PageId) {
        let tree = no_tombstone_tree(bpm);
        for k in 0..15u8 {
            let rid = RecordId {
                page_id: pid(u32::from(k)),
                slot_id: 0,
            };
            tree.insert(
                make_key(k, NO_TOMBSTONE_KEY_LEN as usize),
                IndexValue::IndexValue(rid),
            )
            .unwrap();
        }
        let root = InternalPage::decode(
            bpm.read_page(tree.read_header().unwrap().root_page_id.unwrap())
                .unwrap()
                .data(),
            tree.layout,
        )
        .unwrap();
        (tree, root.children()[0], root.children()[1])
    }

    fn scan_all(tree: &BPlusTree) -> Result<Vec<(IndexKey, IndexValue)>, IndexError> {
        tree.scan((Bound::Unbounded, Bound::Unbounded))?.collect()
    }

    #[test]
    fn a_corrupt_leaf_in_the_middle_of_a_scan_is_an_error() {
        let (bpm, _tmp) = crate::testing::bpm(16);
        let (tree, _, right) = two_leaf_tree(&bpm);
        assert_eq!(scan_all(&tree).unwrap().len(), 15);
        let mut page = bpm.write_page(right).unwrap();
        let slots_start = PAGE_SIZE - tree.layout.leaf_area().capacity();
        let first_entry = usize::from(get_u16_le(page.data(), slots_start));
        let high_byte_of_first_page_id = first_entry + NO_TOMBSTONE_KEY_LEN as usize + 7;
        page.data_mut()[high_byte_of_first_page_id] = 0xff;
        drop(page);
        assert!(
            matches!(scan_all(&tree), Err(IndexError::PageError(_))),
            "the second leaf's first record id does not fit in u32"
        );
    }

    #[test]
    fn a_cycle_in_the_leaf_chain_is_an_error() {
        let (bpm, _tmp) = crate::testing::bpm(16);
        let (tree, left, right) = two_leaf_tree(&bpm);
        let mut leaf = tree.decode_leaf(right).unwrap();
        leaf.set_next_leaf_id(Some(left));
        leaf.encode(bpm.write_page(right).unwrap().data_mut())
            .unwrap();
        assert!(matches!(
            scan_all(&tree),
            Err(IndexError::PageError(PageCodecError::Malformed(
                "leaf chain contains a cycle"
            )))
        ));
    }

    #[test]
    fn removing_the_last_key_of_a_root_leaf_empties_the_tree() {
        let key_len = NO_TOMBSTONE_KEY_LEN as usize;
        let (bpm, _tmp) = crate::testing::bpm(8);
        let tree = no_tombstone_tree(&bpm);
        let rid = RecordId {
            page_id: pid(1),
            slot_id: 1,
        };
        tree.insert(make_key(3, key_len), IndexValue::IndexValue(rid))
            .unwrap();
        tree.remove(make_key(3, key_len)).unwrap();
        let header = tree.read_header().unwrap();
        assert_eq!(header.root_page_id, None);
        assert_eq!(header.first_leaf_id, None);
    }

    #[test]
    fn four_entry_leaf_key_len_gives_four_entry_leaves_and_internal_pages() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let tree = BPlusTree::new("idx".to_string(), FOUR_ENTRY_LEAF_KEY_LEN, bpm).unwrap();
        assert_eq!(longest_entries_per_page(&tree), (4, 4));
    }

    /// Builds trees page by page, in shapes that inserts and removes reach only rarely.
    struct HandBuilt {
        tree: BPlusTree,
        bpm: Arc<BufferPoolManager>,
        leaves: Vec<PageId>,
        _tmp: tempfile::TempDir,
    }

    impl HandBuilt {
        /// An empty tree of keys up to 2000 bytes whose leaves keep no tombstones, so that every
        /// remove takes its entry off the page.
        fn new() -> Self {
            let (bpm, tmp) = crate::testing::bpm(64);
            let layout = Layout::with_tombstone_capacity(2000, 0).unwrap();
            let tree = BPlusTree::with_layout("idx".to_string(), layout, bpm.clone()).unwrap();
            Self {
                tree,
                bpm,
                leaves: Vec::new(),
                _tmp: tmp,
            }
        }

        /// Writes a leaf of `keys` and adds it to the end of the leaf chain.
        fn leaf(&mut self, keys: &[IndexKey]) -> PageId {
            let mut leaf = LeafPage::new(self.tree.layout);
            for k in keys {
                leaf.insert(k.clone(), rid_of(k)).unwrap();
            }
            let (id, mut page) = self.tree.allocate_page().unwrap();
            leaf.encode(page.data_mut()).unwrap();
            drop(page);
            if let Some(&last) = self.leaves.last() {
                let mut prev = self.tree.decode_leaf(last).unwrap();
                prev.set_next_leaf_id(Some(id));
                prev.encode(self.bpm.write_page(last).unwrap().data_mut())
                    .unwrap();
            }
            self.leaves.push(id);
            id
        }

        /// Writes an internal page whose children are `children`, separated by `keys`.
        fn internal(&self, children: &[PageId], keys: &[IndexKey]) -> PageId {
            let mut node =
                InternalPage::new_root(self.tree.layout, children[0], keys[0].clone(), children[1]);
            for (k, &child) in keys[1..].iter().zip(&children[2..]) {
                node.append_key_and_child(k.clone(), child);
            }
            let (id, mut page) = self.tree.allocate_page().unwrap();
            node.encode(page.data_mut()).unwrap();
            id
        }

        fn set_root(&self, root: PageId) {
            let mut header = self.bpm.write_page(self.tree.header_page_id).unwrap();
            let mut view = HeaderPageView::new(header.data_mut()).unwrap();
            view.set_root_page_id(Some(root));
            view.set_first_leaf_id(Some(self.leaves[0]));
        }

        fn page_bytes(&self, id: PageId) -> Vec<u8> {
            self.bpm.read_page(id).unwrap().data().to_vec()
        }

        fn keys_of_leaf(&self, id: PageId) -> Vec<IndexKey> {
            self.tree.decode_leaf(id).unwrap().keys().cloned().collect()
        }

        fn finds(&self, k: &IndexKey) -> bool {
            matches!(
                self.tree.search(k),
                Ok(Some(IndexValue::IndexValue(rid))) if rid == rid_of(k)
            )
        }
    }

    fn rid_of(k: &IndexKey) -> RecordId {
        RecordId {
            page_id: pid(u32::from(k.0[0])),
            slot_id: k.0.len() as u32,
        }
    }

    fn long(first_byte: u8) -> IndexKey {
        IndexKey(vec![first_byte; 2000])
    }

    fn short(first_byte: u8) -> IndexKey {
        IndexKey(vec![first_byte])
    }

    #[test]
    fn an_underfull_leaf_stays_as_it_is_when_its_parent_has_no_room_for_a_longer_separator() {
        let mut t = HandBuilt::new();
        let a = t.leaf(&[IndexKey(vec![1; 1000]), IndexKey(vec![2; 1000]), short(3)]);
        let b = t.leaf(&[long(5), long(6), long(7), long(8)]);
        let fillers: Vec<PageId> = [0x90, 0xA0, 0xB0, 0xC0]
            .into_iter()
            .map(|k| t.leaf(&[long(k)]))
            .collect();
        let root = t.internal(
            &[&[a, b][..], &fillers].concat(),
            &[short(4), long(0x90), long(0xA0), long(0xB0), long(0xC0)],
        );
        t.set_root(root);
        let (root_before, b_before) = (t.page_bytes(root), t.page_bytes(b));

        t.tree.remove(short(3)).unwrap();

        assert_eq!(
            t.keys_of_leaf(a),
            vec![IndexKey(vec![1; 1000]), IndexKey(vec![2; 1000])],
            "the leaf lost the key and borrowed nothing"
        );
        assert!(t.tree.decode_leaf(a).unwrap().is_underfull());
        assert_eq!(t.page_bytes(b), b_before, "the sibling lent nothing");
        assert_eq!(
            t.page_bytes(root),
            root_before,
            "a 2000-byte separator does not fit where the 1-byte one was"
        );
        assert!(!t.finds(&short(3)));
        for k in [IndexKey(vec![1; 1000]), long(5), long(8), long(0xC0)] {
            assert!(t.finds(&k), "{:?} is still found", k.0[0]);
        }
        assert_eq!(scan_all(&t.tree).unwrap().len(), 10);
    }

    #[test]
    fn an_internal_page_left_with_one_child_keeps_serving_removes_and_inserts() {
        let mut t = HandBuilt::new();
        let l1 = t.leaf(&[short(0x10)]);
        let l2 = t.leaf(&[short(0x20), short(0x21)]);
        let s_leaves: Vec<PageId> = [short(0x30), long(0x40), long(0x50), long(0x60)]
            .iter()
            .map(|k| t.leaf(std::slice::from_ref(k)))
            .collect();
        let mut fillers = Vec::new();
        for k in [0x70, 0x80, 0x90, 0xA0] {
            let left = t.leaf(&[long(k)]);
            let right = t.leaf(&[long(k + 1)]);
            fillers.push(t.internal(&[left, right], &[long(k + 1)]));
        }
        let n = t.internal(&[l1, l2], &[short(0x20)]);
        let s = t.internal(&s_leaves, &[long(0x40), long(0x50), long(0x60)]);
        let root = t.internal(
            &[&[n, s][..], &fillers].concat(),
            &[short(0x30), long(0x70), long(0x80), long(0x90), long(0xA0)],
        );
        t.set_root(root);
        let (root_before, s_before) = (t.page_bytes(root), t.page_bytes(s));

        t.tree.remove(short(0x21)).unwrap();

        let node = InternalPage::decode(&t.page_bytes(n), t.tree.layout).unwrap();
        assert_eq!(node.children(), &[l1], "the two leaves merged");
        assert!(node.is_empty() && node.is_underfull());
        assert_eq!(
            t.page_bytes(root),
            root_before,
            "a 2000-byte key does not rotate into the root"
        );
        assert_eq!(t.page_bytes(s), s_before, "the sibling lent nothing");
        assert!(t.finds(&short(0x20)) && !t.finds(&short(0x21)));

        t.tree.remove(short(0x10)).unwrap();
        t.tree.remove(short(0x20)).unwrap();
        assert!(
            t.keys_of_leaf(l1).is_empty(),
            "the only child has no sibling to merge with"
        );
        assert!(!t.finds(&short(0x10)) && !t.finds(&short(0x20)));

        t.tree
            .insert(short(0x15), IndexValue::IndexValue(rid_of(&short(0x15))))
            .unwrap();
        assert!(t.finds(&short(0x15)));
        let scanned: Vec<u8> = scan_all(&t.tree)
            .unwrap()
            .into_iter()
            .map(|(k, _)| k.0[0])
            .collect();
        assert_eq!(
            scanned,
            vec![
                0x15, 0x30, 0x40, 0x50, 0x60, 0x70, 0x71, 0x80, 0x81, 0x90, 0x91, 0xA0, 0xA1
            ]
        );
    }

    fn make_key(first_byte: u8, key_len: usize) -> IndexKey {
        let mut v = vec![0u8; key_len];
        v[0] = first_byte;
        IndexKey(v)
    }

    fn sum_keys_in_leaf_chain(tree: &BPlusTree) -> usize {
        let header = tree.read_header().expect("header");
        let mut total = 0usize;
        let mut leaf_id = header.first_leaf_id;
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = leaf_id {
            assert!(seen.insert(id), "cycle in leaf chain at {id}");
            let leaf = tree.decode_leaf(id).expect("leaf decode");
            total += leaf.len();
            leaf_id = leaf.next_leaf_id();
        }
        total
    }

    fn count_leaves_in_chain(tree: &BPlusTree) -> usize {
        let header = tree.read_header().expect("header");
        let mut count = 0usize;
        let mut leaf_id = header.first_leaf_id;
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = leaf_id {
            assert!(seen.insert(id), "cycle in leaf chain at {id}");
            let leaf = tree.decode_leaf(id).expect("leaf decode");
            count += 1;
            leaf_id = leaf.next_leaf_id();
        }
        count
    }

    #[test]
    fn leaf_pages_iterator_walks_leaf_chain() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        assert_eq!(
            tree.leaf_pages().unwrap().count(),
            0,
            "an empty tree has no leaves"
        );

        for i in 0u8..20u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: pid(i as u32),
                    slot_id: i as u32 + 10,
                }),
            )
            .unwrap();
        }

        let mut leaf_count = 0usize;
        let mut keys_first_bytes: Vec<u8> = Vec::new();
        for res in tree.leaf_pages().unwrap() {
            let (_leaf_id, leaf) = res.unwrap();
            leaf_count += 1;
            for (k, rid) in leaf.iter() {
                keys_first_bytes.push(k.0[0]);
                assert_eq!(rid.page_id, pid(k.0[0] as u32));
                assert_eq!(rid.slot_id, rid.page_id.get() + 10);
            }
        }

        assert_eq!(leaf_count, count_leaves_in_chain(&tree));
        assert_eq!(keys_first_bytes.len(), sum_keys_in_leaf_chain(&tree));
        assert!(tree.print_tree().is_ok());
        assert!(keys_first_bytes.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(keys_first_bytes, (0u8..20u8).collect::<Vec<_>>());
    }

    #[test]
    fn insert_splits_root_leaf_into_internal_root() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: pid(i as u32),
                    slot_id: 0,
                }),
            )
            .unwrap();
        }

        let header = tree.read_header().unwrap();
        assert_ne!(header.root_page_id, None);
        assert_ne!(header.first_leaf_id, None);

        let root_page = bpm.read_page(header.root_page_id.unwrap()).unwrap();
        let root = InternalPage::decode(root_page.data(), tree.layout).unwrap();
        assert_eq!(root.len(), 1);
        assert_eq!(root.children().len(), 2);

        let left_leaf_id = root.get_child(0).unwrap();
        let right_leaf_id = root.get_child(1).unwrap();
        assert_eq!(header.first_leaf_id, Some(*left_leaf_id));

        let left_leaf_page = bpm.read_page(*left_leaf_id).unwrap();
        let left_leaf = LeafPage::decode(left_leaf_page.data(), tree.layout).unwrap();
        let right_leaf_page = bpm.read_page(*right_leaf_id).unwrap();
        let right_leaf = LeafPage::decode(right_leaf_page.data(), tree.layout).unwrap();

        assert_eq!(left_leaf.next_leaf_id(), Some(*right_leaf_id));
        assert_eq!(right_leaf.next_leaf_id(), None);
        assert_eq!(left_leaf.len(), 2);
        assert_eq!(right_leaf.len(), 3);
    }

    #[test]
    fn insert_duplicate_key_errors() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = 32u32;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        let key = make_key(7, key_len as usize);
        tree.insert(
            key.clone(),
            IndexValue::IndexValue(RecordId {
                page_id: pid(1),
                slot_id: 1,
            }),
        )
        .unwrap();

        let err = tree
            .insert(
                key,
                IndexValue::IndexValue(RecordId {
                    page_id: pid(2),
                    slot_id: 2,
                }),
            )
            .unwrap_err();
        assert!(matches!(err, IndexError::Insert(InsertError::DuplicateKey)));
    }

    #[test]
    fn print_tree_smoke() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: pid(i as u32),
                    slot_id: 0,
                }),
            )
            .unwrap();
        }

        tree.print_tree().unwrap();
    }

    #[test]
    fn remove_missing_key_errors_and_does_not_mutate_tree() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = 32u32;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        let missing = make_key(9, key_len as usize);
        let err = tree.remove(missing).unwrap_err();
        assert!(matches!(
            err,
            IndexError::Remove(crate::index::RemoveError::KeyNotFound)
        ));

        for i in 0u8..3u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: pid(i as u32),
                    slot_id: 0,
                }),
            )
            .unwrap();
        }

        let before_header = tree.read_header().unwrap();
        let before_total = sum_keys_in_leaf_chain(&tree);
        let before_leaves = count_leaves_in_chain(&tree);

        let missing = make_key(200, key_len as usize);
        let err = tree.remove(missing).unwrap_err();
        assert!(matches!(
            err,
            IndexError::Remove(crate::index::RemoveError::KeyNotFound)
        ));

        let after_header = tree.read_header().unwrap();
        assert_eq!(before_header.root_page_id, after_header.root_page_id);
        assert_eq!(before_header.first_leaf_id, after_header.first_leaf_id);
        assert_eq!(before_total, sum_keys_in_leaf_chain(&tree));
        assert_eq!(before_leaves, count_leaves_in_chain(&tree));
    }

    #[test]
    fn remove_last_key_marks_tombstone_and_search_returns_none() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = 32u32;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        let key = make_key(7, key_len as usize);
        tree.insert(
            key.clone(),
            IndexValue::IndexValue(RecordId {
                page_id: pid(1),
                slot_id: 1,
            }),
        )
        .unwrap();

        tree.remove(key.clone()).unwrap();

        assert!(tree.search(&key).unwrap().is_none());

        let err = tree.remove(key).unwrap_err();
        assert!(matches!(
            err,
            IndexError::Remove(crate::index::RemoveError::KeyNotFound)
        ));
    }

    #[test]
    fn remove_marks_tombstones_without_changing_leaf_key_counts() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: pid(i as u32),
                    slot_id: 0,
                }),
            )
            .unwrap();
        }

        let (_left_leaf_id, _right_leaf_id) = {
            let header = tree.read_header().unwrap();
            let root_page = bpm.read_page(header.root_page_id.unwrap()).unwrap();
            let root = InternalPage::decode(root_page.data(), tree.layout).unwrap();
            assert_eq!(root.len(), 1);
            assert_eq!(root.children().len(), 2);

            let left_leaf_id = *root.get_child(0).unwrap();
            let right_leaf_id = *root.get_child(1).unwrap();
            let left_leaf = tree.decode_leaf(left_leaf_id).unwrap();
            let right_leaf = tree.decode_leaf(right_leaf_id).unwrap();
            assert_eq!(left_leaf.len(), 2);
            assert_eq!(right_leaf.len(), 3);
            assert_eq!(left_leaf.next_leaf_id(), Some(right_leaf_id));
            assert_eq!(right_leaf.next_leaf_id(), None);

            (left_leaf_id, right_leaf_id)
        };

        let before_total = sum_keys_in_leaf_chain(&tree);
        tree.remove(make_key(4, key_len as usize)).unwrap();
        tree.remove(make_key(0, key_len as usize)).unwrap();
        assert_eq!(before_total, sum_keys_in_leaf_chain(&tree));
        assert!(
            tree.search(&make_key(4, key_len as usize))
                .unwrap()
                .is_none()
        );
        assert!(
            tree.search(&make_key(0, key_len as usize))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_key_too_long_for_the_tree_is_rejected_on_insert_and_absent_otherwise() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let tree = BPlusTree::new("idx".to_string(), 8, bpm).unwrap();
        let rid = |n| {
            IndexValue::IndexValue(RecordId {
                page_id: pid(n),
                slot_id: 1,
            })
        };
        let short = IndexKey(vec![1, 2, 3]);
        let long = IndexKey(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        tree.insert(short.clone(), rid(1)).unwrap();
        tree.insert(IndexKey(vec![]), rid(2)).unwrap();
        assert!(matches!(
            tree.insert(long.clone(), rid(3)).unwrap_err(),
            IndexError::KeyTooLong { max: 8, actual: 9 }
        ));
        assert!(tree.search(&long).unwrap().is_none());
        assert!(matches!(
            tree.remove(long).unwrap_err(),
            IndexError::Remove(RemoveError::KeyNotFound)
        ));
        assert!(tree.search(&short).unwrap().is_some());
        tree.remove(short.clone()).unwrap();
        assert!(tree.search(&short).unwrap().is_none());
        assert_eq!(scan_all(&tree).unwrap().len(), 1);
    }

    #[test]
    fn key_lengths_that_cannot_fit_a_page_are_rejected() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let past_max = u32::try_from(MAX_KEY_LEN + 1).unwrap();
        for key_len in [0, past_max, 1 << 20] {
            let err = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap_err();
            assert!(matches!(err, IndexError::UnsupportedKeyLength(l) if l == key_len));
        }
    }

    #[test]
    fn search_on_empty_tree_returns_none() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = 32u32;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm).unwrap();
        let key = make_key(1, key_len as usize);
        assert!(tree.search(&key).unwrap().is_none());
        let from_key = (Bound::Included(key.clone()), Bound::Unbounded);
        assert_eq!(tree.scan(from_key).unwrap().count(), 0);
    }

    #[test]
    fn search_finds_inserted_keys_and_returns_none_for_missing() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: pid(i as u32),
                    slot_id: i as u32 + 10,
                }),
            )
            .unwrap();
        }

        for i in 0u8..5u8 {
            let key = make_key(i, key_len as usize);
            let IndexValue::IndexValue(rid) = tree.search(&key).unwrap().expect("key present");
            assert_eq!(rid.page_id, pid(i as u32));
            assert_eq!(rid.slot_id, i as u32 + 10);
        }

        let missing = make_key(9, key_len as usize);
        assert!(tree.search(&missing).unwrap().is_none());
    }

    mod remove_many_keys {
        use super::*;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use std::collections::BTreeMap;

        const SCAN_CHECK_EVERY: usize = 211;

        fn key_for(n: u32, key_len: usize) -> IndexKey {
            let mut v = vec![0u8; key_len];
            v[..4].copy_from_slice(&n.to_be_bytes());
            IndexKey(v)
        }

        /// The longest key [`varied_key_for`] makes.
        const VARIED_MAX_KEY_LEN: usize = 2000;

        /// A key for `n` whose length, between 4 and `VARIED_MAX_KEY_LEN` bytes, varies with `n`.
        fn varied_key_for(n: u32) -> IndexKey {
            let len = 4 + (n as usize).wrapping_mul(2_654_435_761) % (VARIED_MAX_KEY_LEN - 3);
            key_for(n, len)
        }

        /// The keys a test tree holds.
        #[derive(Clone, Copy)]
        enum Keys {
            /// Every key `n` is [`key_for`] `n` at this length.
            Fixed(usize),
            /// Key `n` is [`varied_key_for`] `n`.
            Varied,
        }

        impl Keys {
            fn key(self, n: u32) -> IndexKey {
                match self {
                    Keys::Fixed(len) => key_for(n, len),
                    Keys::Varied => varied_key_for(n),
                }
            }

            fn max_key_len(self) -> usize {
                match self {
                    Keys::Fixed(len) => len,
                    Keys::Varied => VARIED_MAX_KEY_LEN,
                }
            }

            /// Whether every page but the root stays at least minimally full. With keys of
            /// varied lengths a page may stay underfull when its parent has no room for a
            /// longer separator.
            fn keeps_pages_full(self) -> bool {
                matches!(self, Keys::Fixed(_))
            }
        }

        fn rid_for(n: u32) -> RecordId {
            RecordId {
                page_id: pid(n),
                slot_id: n.wrapping_mul(7),
            }
        }

        struct SubtreeBounds {
            lower: Option<IndexKey>,
            upper: Option<IndexKey>,
        }

        impl SubtreeBounds {
            fn contains(&self, key: &IndexKey) -> bool {
                self.lower.as_ref().is_none_or(|l| key >= l)
                    && self.upper.as_ref().is_none_or(|u| key < u)
            }
        }

        fn walk_subtree(
            tree: &BPlusTree,
            page_id: PageId,
            bounds: SubtreeBounds,
            depth: usize,
            is_root: bool,
            keeps_pages_full: bool,
            leaves_in_order: &mut Vec<(PageId, usize)>,
        ) {
            let page = tree.bpm.read_page(page_id).unwrap();
            match Page::decode(page.data(), tree.layout).unwrap() {
                Page::Leaf(leaf) => {
                    assert!(
                        leaf.keys().all(|k| bounds.contains(k)),
                        "leaf {page_id} holds a key outside its parent's separator range"
                    );
                    leaves_in_order.push((page_id, depth));
                }
                Page::Internal(node) => {
                    if is_root {
                        assert!(
                            node.children().len() >= 2,
                            "internal root {page_id} has fewer than two children"
                        );
                    } else if keeps_pages_full {
                        assert!(
                            !node.is_underfull(),
                            "non-root internal page {page_id} is below minimum occupancy"
                        );
                    }
                    assert!(
                        node.keys().iter().all(|k| bounds.contains(k)),
                        "internal page {page_id} holds a separator outside its parent's range"
                    );
                    drop(page);
                    for (i, &child) in node.children().iter().enumerate() {
                        let child_bounds = SubtreeBounds {
                            lower: if i == 0 {
                                bounds.lower.clone()
                            } else {
                                Some(node.keys()[i - 1].clone())
                            },
                            upper: node.keys().get(i).cloned().or(bounds.upper.clone()),
                        };
                        walk_subtree(
                            tree,
                            child,
                            child_bounds,
                            depth + 1,
                            false,
                            keeps_pages_full,
                            leaves_in_order,
                        );
                    }
                }
            }
        }

        fn assert_tree_is_well_formed(tree: &BPlusTree, keys: Keys) {
            let header = tree.read_header().unwrap();
            let Some(root_page_id) = header.root_page_id else {
                assert_eq!(
                    header.first_leaf_id, None,
                    "empty tree still points at a first leaf"
                );
                return;
            };
            let mut leaves_in_order = Vec::new();
            let unbounded = SubtreeBounds {
                lower: None,
                upper: None,
            };
            walk_subtree(
                tree,
                root_page_id,
                unbounded,
                0,
                true,
                keys.keeps_pages_full(),
                &mut leaves_in_order,
            );
            let depths: HashSet<usize> = leaves_in_order.iter().map(|&(_, d)| d).collect();
            assert_eq!(
                depths.len(),
                1,
                "leaves sit at different depths: {depths:?}"
            );
            let tree_order: Vec<PageId> = leaves_in_order.iter().map(|&(id, _)| id).collect();
            let chain_order: Vec<PageId> =
                tree.leaf_pages().unwrap().map(|r| r.unwrap().0).collect();
            assert_eq!(
                tree_order, chain_order,
                "leaf chain disagrees with in-order leaf traversal"
            );
        }

        struct Harness {
            tree: BPlusTree,
            model: BTreeMap<u32, RecordId>,
            keys: Keys,
            ops: usize,
            _bpm: Arc<BufferPoolManager>,
            _tmp: tempfile::TempDir,
        }

        impl Harness {
            fn new(keys: Keys) -> Self {
                let (bpm, tmp) = crate::testing::bpm(64);
                let max_key_len = keys.max_key_len() as u32;
                let tree = BPlusTree::new("idx".to_string(), max_key_len, bpm.clone()).unwrap();
                Self {
                    tree,
                    model: BTreeMap::new(),
                    keys,
                    ops: 0,
                    _bpm: bpm,
                    _tmp: tmp,
                }
            }

            fn insert(&mut self, n: u32) {
                self.tree
                    .insert(self.keys.key(n), IndexValue::IndexValue(rid_for(n)))
                    .unwrap_or_else(|e| panic!("insert {n} failed: {e}"));
                self.model.insert(n, rid_for(n));
                self.after_op(n);
            }

            fn remove(&mut self, n: u32) {
                self.tree
                    .remove(self.keys.key(n))
                    .unwrap_or_else(|e| panic!("remove {n} failed: {e}"));
                self.model.remove(&n);
                self.after_op(n);
            }

            fn after_op(&mut self, n: u32) {
                let expected = self.model.get(&n).copied();
                let found = self
                    .tree
                    .search(&self.keys.key(n))
                    .unwrap()
                    .map(|IndexValue::IndexValue(rid)| rid);
                assert_eq!(found, expected, "search({n}) disagrees with the model");
                self.ops += 1;
                if self.ops.is_multiple_of(SCAN_CHECK_EVERY) {
                    self.check_full_scan();
                }
            }

            fn check_full_scan(&self) {
                assert_tree_is_well_formed(&self.tree, self.keys);
                let scanned: Vec<(IndexKey, RecordId)> = self
                    .tree
                    .scan((Bound::Unbounded, Bound::Unbounded))
                    .unwrap()
                    .map(|r| {
                        let (k, IndexValue::IndexValue(rid)) = r.unwrap();
                        (k, rid)
                    })
                    .collect();
                let expected: Vec<(IndexKey, RecordId)> = self
                    .model
                    .iter()
                    .map(|(&n, &rid)| (self.keys.key(n), rid))
                    .collect();
                assert_eq!(
                    scanned.len(),
                    expected.len(),
                    "scan length disagrees with the model after {} ops",
                    self.ops
                );
                assert!(
                    scanned == expected,
                    "scan contents disagree with the model after {} ops",
                    self.ops
                );
                for &n in self.model.keys() {
                    assert!(
                        self.tree.search(&self.keys.key(n)).unwrap().is_some(),
                        "search({n}) lost a live key after {} ops",
                        self.ops
                    );
                }
            }
        }

        fn insert_then_remove_all(keys: Keys, inserts: &[u32], removes: &[u32]) {
            let mut h = Harness::new(keys);
            for &n in inserts {
                h.insert(n);
            }
            h.check_full_scan();
            for &n in removes {
                h.remove(n);
            }
            h.check_full_scan();
            assert!(h.model.is_empty());
        }

        const KEY_SHAPES: [Keys; 4] = [
            Keys::Fixed(2000),
            Keys::Fixed(600),
            Keys::Fixed(64),
            Keys::Varied,
        ];
        const N: u32 = 2500;

        #[test]
        fn sequential_insert_then_sequential_remove_keeps_tree_consistent() {
            let keys: Vec<u32> = (0..N).collect();
            for shape in KEY_SHAPES {
                insert_then_remove_all(shape, &keys, &keys);
            }
        }

        #[test]
        fn reversed_insert_then_reversed_remove_keeps_tree_consistent() {
            let keys: Vec<u32> = (0..N).rev().collect();
            for shape in KEY_SHAPES {
                insert_then_remove_all(shape, &keys, &keys);
            }
        }

        #[test]
        fn sequential_insert_then_reversed_remove_keeps_tree_consistent() {
            let keys: Vec<u32> = (0..N).collect();
            let rev: Vec<u32> = (0..N).rev().collect();
            for shape in KEY_SHAPES {
                insert_then_remove_all(shape, &keys, &rev);
            }
        }

        #[test]
        fn random_insert_then_random_remove_keeps_tree_consistent() {
            for (seed, keys) in KEY_SHAPES.into_iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(seed as u64);
                let mut inserts: Vec<u32> = (0..N).collect();
                inserts.shuffle(&mut rng);
                let mut removes = inserts.clone();
                removes.shuffle(&mut rng);
                insert_then_remove_all(keys, &inserts, &removes);
            }
        }

        #[test]
        fn removed_keys_can_be_reinserted_and_removed_again() {
            for (seed, keys) in KEY_SHAPES.into_iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(200 + seed as u64);
                let mut h = Harness::new(keys);
                let mut keys: Vec<u32> = (0..N).collect();
                keys.shuffle(&mut rng);
                for &n in &keys {
                    h.insert(n);
                }
                let evens: Vec<u32> = keys.iter().copied().filter(|n| n % 2 == 0).collect();
                for &n in &evens {
                    h.remove(n);
                }
                for &n in &evens {
                    h.insert(n);
                }
                keys.shuffle(&mut rng);
                for &n in &keys {
                    h.remove(n);
                }
                h.check_full_scan();
                assert!(h.model.is_empty());
            }
        }

        #[test]
        fn interleaved_inserts_and_removes_keep_tree_consistent() {
            for (seed, keys) in KEY_SHAPES.into_iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(100 + seed as u64);
                let mut h = Harness::new(keys);
                let mut pool: Vec<u32> = (0..N).collect();
                pool.shuffle(&mut rng);
                let mut live: Vec<u32> = Vec::new();
                for n in pool {
                    h.insert(n);
                    live.push(n);
                    if live.len() > 50 && rand::Rng::random_bool(&mut rng, 0.45) {
                        let i = rand::Rng::random_range(&mut rng, 0..live.len());
                        let victim = live.swap_remove(i);
                        h.remove(victim);
                    }
                }
                live.shuffle(&mut rng);
                for n in live {
                    h.remove(n);
                }
                h.check_full_scan();
                assert!(h.model.is_empty());
            }
        }

        mod concurrent_root_changes {
            use super::*;
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::thread;

            const KEY_LEN: usize = 2000;
            const KEYS: u32 = 300;
            const ROUNDS: u64 = 60;
            const WRITERS: usize = 2;
            const READERS: u64 = 2;
            const FRAMES: usize = 256;
            const KEEP_EVERY: u32 = 10;

            #[derive(Clone, Copy)]
            enum Op {
                Insert(u32),
                Remove(u32),
            }

            fn apply(tree: &BPlusTree, present: &[AtomicBool], op: Op) {
                match op {
                    Op::Insert(n) => {
                        tree.insert(key_for(n, KEY_LEN), IndexValue::IndexValue(rid_for(n)))
                            .unwrap_or_else(|e| panic!("insert {n} failed: {e}"));
                        present[n as usize].store(true, Ordering::Release);
                    }
                    Op::Remove(n) => {
                        present[n as usize].store(false, Ordering::Release);
                        tree.remove(key_for(n, KEY_LEN))
                            .unwrap_or_else(|e| panic!("remove {n} failed: {e}"));
                    }
                }
            }

            fn finds(tree: &BPlusTree, n: u32) -> bool {
                matches!(
                    tree.search(&key_for(n, KEY_LEN)),
                    Ok(Some(IndexValue::IndexValue(rid))) if rid == rid_for(n)
                )
            }

            fn keys_a_reader_missed(
                tree: &BPlusTree,
                present: &[AtomicBool],
                writers_done: &AtomicBool,
                seed: u64,
            ) -> Vec<u32> {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut missed = Vec::new();
                while !writers_done.load(Ordering::Acquire) {
                    let n = rand::Rng::random_range(&mut rng, 0..KEYS);
                    let is_present = || present[n as usize].load(Ordering::Acquire);
                    if is_present() && !finds(tree, n) && is_present() {
                        missed.push(n);
                    }
                }
                missed
            }

            /// Runs each writer's ops on its own thread while readers search for random keys.
            /// Returns the keys a reader failed to find although a writer had finished inserting
            /// them before the search started and had not started removing them when it ended.
            fn keys_missed_during(
                tree: &BPlusTree,
                present: &[AtomicBool],
                writer_ops: Vec<Vec<Op>>,
                seed: u64,
            ) -> Vec<u32> {
                let writers_done = AtomicBool::new(false);
                let writers_done = &writers_done;
                thread::scope(|s| {
                    let readers: Vec<_> = (0..READERS)
                        .map(|r| {
                            s.spawn(move || {
                                keys_a_reader_missed(tree, present, writers_done, seed * 31 + r)
                            })
                        })
                        .collect();
                    let writers: Vec<_> = writer_ops
                        .into_iter()
                        .map(|ops| {
                            s.spawn(move || {
                                for op in ops {
                                    apply(tree, present, op);
                                }
                            })
                        })
                        .collect();
                    let writer_results: Vec<_> = writers.into_iter().map(|w| w.join()).collect();
                    writers_done.store(true, Ordering::Release);
                    let missed = readers
                        .into_iter()
                        .flat_map(|r| r.join().expect("reader thread panicked"))
                        .collect();
                    for result in writer_results {
                        if let Err(panic) = result {
                            std::panic::resume_unwind(panic);
                        }
                    }
                    missed
                })
            }

            fn deal_round_robin(keys: &[u32], op: fn(u32) -> Op) -> Vec<Vec<Op>> {
                let mut per_writer = vec![Vec::new(); WRITERS];
                for (i, &n) in keys.iter().enumerate() {
                    per_writer[i % WRITERS].push(op(n));
                }
                per_writer
            }

            fn assert_tree_holds_exactly(tree: &BPlusTree, present: &[AtomicBool], round: u64) {
                assert_tree_is_well_formed(tree, Keys::Fixed(KEY_LEN));
                let expected: Vec<IndexKey> = (0..KEYS)
                    .filter(|&n| present[n as usize].load(Ordering::Acquire))
                    .map(|n| key_for(n, KEY_LEN))
                    .collect();
                let scanned: Vec<IndexKey> = tree
                    .scan((Bound::Unbounded, Bound::Unbounded))
                    .unwrap()
                    .map(|r| r.unwrap().0)
                    .collect();
                assert!(
                    scanned == expected,
                    "round {round}: scan holds {} keys, expected {}",
                    scanned.len(),
                    expected.len()
                );
            }

            fn fresh_tree() -> (BPlusTree, Vec<AtomicBool>, tempfile::TempDir) {
                let (bpm, tmp) = crate::testing::bpm(FRAMES);
                let tree = BPlusTree::new("idx".to_string(), KEY_LEN as u32, bpm).unwrap();
                assert_eq!(
                    longest_entries_per_page(&tree),
                    (4, 4),
                    "test expects four-entry pages"
                );
                let present = (0..KEYS).map(|_| AtomicBool::new(false)).collect();
                (tree, present, tmp)
            }

            #[test]
            fn readers_and_writers_see_every_key_while_the_root_splits() {
                for round in 0..ROUNDS {
                    let (tree, present, _tmp) = fresh_tree();
                    let mut keys: Vec<u32> = (0..KEYS).collect();
                    keys.shuffle(&mut StdRng::seed_from_u64(round));
                    let missed = keys_missed_during(
                        &tree,
                        &present,
                        deal_round_robin(&keys, Op::Insert),
                        round,
                    );
                    assert!(
                        missed.is_empty(),
                        "round {round}: readers missed inserted keys {missed:?}"
                    );
                    assert_tree_holds_exactly(&tree, &present, round);
                }
            }

            #[test]
            fn readers_see_every_kept_key_while_the_root_collapses() {
                for round in 0..ROUNDS {
                    let (tree, present, _tmp) = fresh_tree();
                    for n in 0..KEYS {
                        apply(&tree, &present, Op::Insert(n));
                    }
                    let mut doomed: Vec<u32> = (0..KEYS).filter(|n| n % KEEP_EVERY != 0).collect();
                    doomed.shuffle(&mut StdRng::seed_from_u64(1000 + round));
                    let missed = keys_missed_during(
                        &tree,
                        &present,
                        deal_round_robin(&doomed, Op::Remove),
                        round,
                    );
                    assert!(
                        missed.is_empty(),
                        "round {round}: readers missed kept keys {missed:?}"
                    );
                    assert_tree_holds_exactly(&tree, &present, round);
                }
            }
        }
    }

    fn wide_key(k: u16, key_len: usize) -> IndexKey {
        let mut v = vec![0u8; key_len];
        v[..2].copy_from_slice(&k.to_be_bytes());
        IndexKey(v)
    }

    fn collect_scan(
        tree: &BPlusTree,
        range: (Bound<IndexKey>, Bound<IndexKey>),
    ) -> Vec<(IndexKey, RecordId)> {
        tree.scan(range)
            .expect("scan")
            .map(|r| match r.expect("scan item") {
                (k, IndexValue::IndexValue(rid)) => (k, rid),
            })
            .collect()
    }

    /// Random inserts and removes against a BTreeMap model, with keys wide enough that leaves
    /// and internal pages hold ~20 entries, so splits, merges, redistribution and tombstone
    /// revival all run through the in-place and owned paths. Point lookups and range scans
    /// agree with the model throughout. Inserts dominate the first half so the tree grows,
    /// and removes dominate the second half so it shrinks.
    #[test]
    fn random_ops_match_btreemap_model() {
        random_ops_match_model(400, |k| wide_key(k, 400));
    }

    /// Like `random_ops_match_btreemap_model`, with keys from 2 to 1500 bytes long, so pages
    /// hold from a handful to hundreds of entries and split and merge by bytes.
    #[test]
    fn random_ops_with_keys_of_varied_lengths_match_btreemap_model() {
        random_ops_match_model(1500, |k| wide_key(k, 2 + usize::from(k) * 37 % 1499));
    }

    fn random_ops_match_model(max_key_len: usize, wide_key: impl Fn(u16) -> IndexKey) {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::collections::BTreeMap;
        use std::collections::btree_map::Entry;

        let (bpm, _dir) = crate::testing::bpm(512);
        let tree = BPlusTree::new("idx".to_string(), max_key_len as u32, bpm).unwrap();
        let (leaf_entries, internal_entries) = longest_entries_per_page(&tree);
        assert!(leaf_entries < 25 && internal_entries < 25);

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut model: BTreeMap<IndexKey, RecordId> = BTreeMap::new();

        for step in 0..6000u32 {
            let k = rng.random_range(0..800u16);
            let key = wide_key(k);
            let insert_prob = if step < 3000 { 0.75 } else { 0.3 };
            if rng.random_bool(insert_prob) {
                let rid = RecordId {
                    page_id: pid(step),
                    slot_id: u32::from(k),
                };
                let res = tree.insert(key.clone(), IndexValue::IndexValue(rid));
                match model.entry(key) {
                    Entry::Occupied(_) => assert!(
                        res.is_err(),
                        "step {step}: duplicate insert of {k} succeeded"
                    ),
                    Entry::Vacant(slot) => {
                        res.unwrap_or_else(|e| panic!("step {step}: insert {k} failed: {e}"));
                        slot.insert(rid);
                    }
                }
            } else {
                let res = tree.remove(key.clone());
                if model.remove(&key).is_some() {
                    res.unwrap_or_else(|e| panic!("step {step}: remove {k} failed: {e}"));
                } else {
                    assert!(res.is_err(), "step {step}: remove of absent {k} succeeded");
                }
            }

            if step % 500 == 499 {
                for probe in 0..800u16 {
                    let key = wide_key(probe);
                    let got = match tree.search(&key) {
                        Ok(Some(IndexValue::IndexValue(rid))) => Some(rid),
                        Ok(None) => None,
                        other => panic!("step {step}: search {probe}: {other:?}"),
                    };
                    assert_eq!(got, model.get(&key).copied(), "step {step}: search {probe}");
                }
                let all = collect_scan(&tree, (Bound::Unbounded, Bound::Unbounded));
                let want: Vec<_> = model.iter().map(|(k, v)| (k.clone(), *v)).collect();
                assert_eq!(all, want, "step {step}: full scan");

                let (lo, hi) = (wide_key(200), wide_key(600));
                let got = collect_scan(
                    &tree,
                    (Bound::Excluded(lo.clone()), Bound::Included(hi.clone())),
                );
                let want: Vec<_> = model
                    .range((Bound::Excluded(lo), Bound::Included(hi)))
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                assert_eq!(got, want, "step {step}: range scan");
            }
        }
    }

    #[test]
    fn a_header_with_a_root_but_no_first_leaf_fails_a_full_scan() {
        let key_len = 8usize;
        let (bpm, _tmp) = crate::testing::bpm(8);
        let tree = BPlusTree::new("idx".to_string(), key_len as u32, bpm.clone()).unwrap();
        let rid = RecordId {
            page_id: pid(1),
            slot_id: 0,
        };
        tree.insert(make_key(1, key_len), IndexValue::IndexValue(rid))
            .unwrap();
        let mut header = tree.read_header().unwrap();
        header.first_leaf_id = None;
        header
            .encode(bpm.write_page(tree.header_page_id).unwrap().data_mut())
            .unwrap();

        let err = tree
            .scan((Bound::Unbounded, Bound::Unbounded))
            .err()
            .unwrap();
        assert!(
            matches!(
                err,
                IndexError::Structure("the header names a root but no first leaf")
            ),
            "got {err:?}"
        );
        assert!(
            tree.search(&make_key(1, key_len)).unwrap().is_some(),
            "a keyed lookup goes through the root and still works"
        );
    }
}
