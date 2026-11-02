use crate::buffer_pool::{BufferPoolManager, PageMut, PageRef};
use crate::catalog::index_schema::{IndexKey, IndexValue};
use crate::disk::PAGE_SIZE;
use crate::ids::RecordId;
use crate::index::b_plus_page::{
    HeaderPage, InternalPage, LeafPage, NUM_TOMBSTONES, Page, PageId, PageKind,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rebalance {
    Redistributed,
    Merged,
}

struct Split {
    left_id: PageId,
    separator: IndexKey,
    right_id: PageId,
}

fn leaf_has_spare_entry(leaf: &LeafPage) -> bool {
    leaf.keys.len() > leaf.min_size() as usize
}

fn internal_has_spare_key(node: &InternalPage) -> bool {
    node.keys.len() > node.min_size() as usize
}

fn safe_for_insert(page: &NodeView<&[u8]>) -> bool {
    !page.is_full()
}

fn safe_for_remove(page: &NodeView<&[u8]>) -> bool {
    match page {
        NodeView::Internal(node) => node.size() > node.min_size(),
        NodeView::Leaf(leaf) => leaf.size() > leaf.min_size(),
    }
}

/// The child of the internal page in `buf` to descend into for `key`, or `None` when `buf`
/// holds a leaf. A leaf is only told apart by its kind here; whoever reads it validates it.
fn child_for_key(buf: &[u8], key_len: u32, key: &IndexKey) -> Result<Option<PageId>, IndexError> {
    if peek_common_header(buf)? == KIND_LEAF {
        return Ok(None);
    }
    InternalPageView::new(buf, key_len)?
        .child_for(&key.0)
        .ok_or(NO_CHILD_FOR_KEY)
        .map(Some)
}

/// Moves `current` down to `child`; `child` is latched before `current` is released.
fn step_down<'a>(current: &mut PageRef<'a>, child: PageRef<'a>) {
    drop(std::mem::replace(current, child));
}

/// Disk-backed B+ tree mapping fixed-length keys to record ids.
///
/// The shape it maintains is checked by `assert_tree_is_well_formed` in this module's tests.
#[derive(Debug)]
pub struct BPlusTree {
    pub header_page_id: PageId,
    key_len: u32,
    index_name: String,
    leaf_max_size: u32,
    internal_max_size: u32,
    bpm: Arc<BufferPoolManager>,
}

pub struct LeafPagesIter {
    bpm: Arc<BufferPoolManager>,
    key_len: u32,
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

        match LeafPage::decode(page.data(), self.key_len) {
            Ok(leaf) => {
                self.next_leaf_id = leaf.next_leaf_id;
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
    pub fn new(
        index_name: String,
        key_len: u32,
        bpm: Arc<BufferPoolManager>,
    ) -> Result<Self, IndexError> {
        let key_len_usize =
            usize::try_from(key_len).map_err(|_| IndexError::UnsupportedKeyLength(key_len))?;
        let (Some(leaf_capacity), Some(internal_capacity)) = (
            LeafPage::max_size_for_layout(key_len_usize),
            InternalPage::max_size_for_layout(key_len_usize),
        ) else {
            return Err(IndexError::UnsupportedKeyLength(key_len));
        };

        let header_page_id = {
            let mut page = bpm.new_page()?;
            let pid = page.page_id();
            HeaderPageView::init(page.data_mut(), key_len)?;
            pid
        };
        info!("created new b+ tree; header_page_id={header_page_id}");

        Ok(Self {
            index_name,
            key_len,
            header_page_id,
            leaf_max_size: leaf_capacity,
            internal_max_size: internal_capacity,
            bpm,
        })
    }

    /// The tree whose header page is `header_page_id`, as an earlier [`BPlusTree::new`] over
    /// the same data file left it. Fails with [`IndexError::KeyLength`] when the header was
    /// written for keys of another length.
    pub fn open(
        index_name: String,
        key_len: u32,
        header_page_id: PageId,
        bpm: Arc<BufferPoolManager>,
    ) -> Result<Self, IndexError> {
        let key_len_usize =
            usize::try_from(key_len).map_err(|_| IndexError::UnsupportedKeyLength(key_len))?;
        let (Some(leaf_capacity), Some(internal_capacity)) = (
            LeafPage::max_size_for_layout(key_len_usize),
            InternalPage::max_size_for_layout(key_len_usize),
        ) else {
            return Err(IndexError::UnsupportedKeyLength(key_len));
        };
        let tree = Self {
            index_name,
            key_len,
            header_page_id,
            leaf_max_size: leaf_capacity,
            internal_max_size: internal_capacity,
            bpm,
        };
        tree.read_header()?;
        Ok(tree)
    }

    /// Acquires a read latch on the header page and copies out its fields.
    fn read_header(&self) -> Result<HeaderPage, IndexError> {
        let page = self.bpm.read_page(self.header_page_id)?;
        self.header_from_bytes(page.data())
    }

    fn header_from_bytes(&self, buf: &[u8]) -> Result<HeaderPage, IndexError> {
        let view = HeaderPageView::new(buf)?;
        if view.key_len() != self.key_len {
            return Err(IndexError::KeyLength {
                expected: self.key_len as usize,
                actual: view.key_len() as usize,
            });
        }
        Ok(HeaderPage {
            root_page_id: view.root_page_id()?,
            first_leaf_id: view.first_leaf_id()?,
            key_len: view.key_len(),
        })
    }

    /// Writes `header`'s root and first-leaf pointers into the latched header page in place.
    fn store_header(page: &mut PageMut<'_>, header: &HeaderPage) -> Result<(), IndexError> {
        let mut view = HeaderPageView::new(page.data_mut())?;
        view.set_root_page_id(header.root_page_id);
        view.set_first_leaf_id(header.first_leaf_id);
        Ok(())
    }

    fn check_key_len(&self, key: &IndexKey) -> Result<(), IndexError> {
        if key.0.len() == self.key_len as usize {
            Ok(())
        } else {
            Err(IndexError::KeyLength {
                expected: self.key_len as usize,
                actual: key.0.len(),
            })
        }
    }

    fn decode_leaf(&self, page_id: PageId) -> Result<LeafPage, IndexError> {
        let page = self.bpm.read_page(page_id)?;
        Ok(LeafPage::decode(page.data(), self.key_len)?)
    }

    fn allocate_page(&self) -> Result<(PageId, PageMut<'_>), IndexError> {
        let page = self.bpm.new_page()?;
        Ok((page.page_id(), page))
    }

    fn new_leaf(&self) -> LeafPage {
        let tombstone_capacity = LeafPage::tombstone_capacity_for_layout(
            self.leaf_max_size,
            self.key_len,
            NUM_TOMBSTONES,
        );
        LeafPage::new(self.leaf_max_size, tombstone_capacity)
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
            let Some(child_id) = child_for_key(current.data(), self.key_len, key)? else {
                return Ok(());
            };
            let child = self.bpm.write_page(child_id)?;
            let child_is_safe = is_safe(&NodeView::new(child.data(), self.key_len)?);
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
            leaf.encode(leaf_page.data_mut(), self.key_len)?;
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
        let mut leaf = LeafPage::decode(leaf_page.data(), self.key_len)?;
        leaf.insert(key, record_id)
            .map_err(|_| IndexError::Insert(InsertError::DuplicateKey))?;
        if !leaf.is_overfull() {
            leaf.encode(leaf_page.data_mut(), self.key_len)?;
            return Ok(());
        }

        let (right_id, mut right_page) = self.allocate_page()?;
        let mut right_leaf = self.new_leaf();
        let separator = leaf.split_into(&mut right_leaf, right_id);
        leaf.encode(leaf_page.data_mut(), self.key_len)?;
        right_leaf.encode(right_page.data_mut(), self.key_len)?;
        let mut split = Split {
            left_id: leaf_page.page_id(),
            separator,
            right_id,
        };

        while let Some(mut parent_page) = ctx.write_set.pop_back() {
            let mut parent = InternalPage::decode(parent_page.data_mut(), self.key_len)?;
            parent.insert_separator(split.separator, split.right_id);
            if !parent.is_overfull() {
                parent.encode(parent_page.data_mut(), self.key_len)?;
                return Ok(());
            }

            let (right_id, mut right_page) = self.allocate_page()?;
            let mut right = InternalPage::new(self.internal_max_size);
            let separator = parent.split_into(&mut right);
            parent.encode(parent_page.data_mut(), self.key_len)?;
            right.encode(right_page.data_mut(), self.key_len)?;
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
        let mut root = InternalPage::new(self.internal_max_size);
        root.children.push(split.left_id);
        root.insert_separator(split.separator, split.right_id);
        root.encode(root_page.data_mut(), self.key_len)?;
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
        match LeafPageView::new(leaf_page.data_mut(), self.key_len)?.insert(&key.0, record_id) {
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
        let mut leaf = LeafPage::decode(leaf_page.data(), self.key_len)?;
        leaf.remove_key(&key)?;
        if !leaf.is_underfull() {
            leaf.encode(leaf_page.data_mut(), self.key_len)?;
            return Ok(());
        }

        let mut parent_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
        let mut parent = InternalPage::decode(parent_page.data_mut(), self.key_len)?;
        let mut outcome = self.rebalance_leaf(&key, leaf, &mut leaf_page, &mut parent)?;
        parent.encode(parent_page.data_mut(), self.key_len)?;

        while outcome == Rebalance::Merged {
            if parent_page.page_id() == ctx.root_page_id {
                if parent.children.len() == 1 {
                    header.root_page_id = Some(parent.children[0]);
                    let header_page = ctx.header.as_mut().ok_or(HEADER_LATCH_RELEASED)?;
                    Self::store_header(header_page, &header)?;
                }
                return Ok(());
            }
            if !parent.is_underfull() {
                return Ok(());
            }
            let mut grandparent_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
            let mut grandparent = InternalPage::decode(grandparent_page.data_mut(), self.key_len)?;
            outcome = self.rebalance_internal(&key, parent, &mut parent_page, &mut grandparent)?;
            grandparent.encode(grandparent_page.data_mut(), self.key_len)?;
            parent_page = grandparent_page;
            parent = grandparent;
        }
        Ok(())
    }

    /// Removes `key` from the latched leaf's bytes when the leaf is the root or stays above min
    /// size. For a root leaf, the header page that an emptied root updates is latched and
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
        let mut leaf = LeafPageView::new(leaf_page.data_mut(), self.key_len)?;
        if !leaf_is_root {
            if leaf.size() <= leaf.min_size() {
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

    fn sibling_of(parent: &InternalPage, child_idx: usize) -> Result<(usize, PageId), IndexError> {
        let sibling_idx = if child_idx == 0 { 1 } else { child_idx - 1 };
        let sibling_id = parent
            .get_child(sibling_idx)
            .copied()
            .ok_or(IndexError::Structure(
                "underfull node has no sibling to borrow from or merge with",
            ))?;
        Ok((sibling_idx, sibling_id))
    }

    fn rebalance_leaf(
        &self,
        key: &IndexKey,
        mut leaf: LeafPage,
        leaf_page: &mut PageMut<'_>,
        parent: &mut InternalPage,
    ) -> Result<Rebalance, IndexError> {
        let leaf_idx = parent.find_child_index(key);
        let (sibling_idx, sibling_id) = Self::sibling_of(parent, leaf_idx)?;
        let mut sibling_page = self.bpm.write_page(sibling_id)?;
        let mut sibling = LeafPage::decode(sibling_page.data_mut(), self.key_len)?;
        let sibling_is_right = sibling_idx > leaf_idx;

        while leaf.is_underfull() && leaf_has_spare_entry(&sibling) {
            let borrowed = if sibling_is_right {
                sibling.pop_first()
            } else {
                sibling.pop_last()
            };
            let (k, rid, tombstoned) = borrowed.ok_or(EMPTY_SIBLING)?;
            let leaf_tombstones_full = leaf.tombstones.len() >= leaf.tombstone_capacity;
            if tombstoned && leaf_tombstones_full {
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
                leaf.encode(leaf_page.data_mut(), self.key_len)?;
            } else {
                sibling.absorb_right(&mut leaf);
                parent.remove_separator(sibling_idx);
                sibling.encode(sibling_page.data_mut(), self.key_len)?;
            }
            return Ok(Rebalance::Merged);
        }

        if sibling_is_right {
            let new_separator = sibling.keys.first().ok_or(EMPTY_SIBLING)?.clone();
            parent.replace_key(leaf_idx, new_separator);
        } else {
            let new_separator = leaf.keys.first().ok_or(EMPTY_SIBLING)?.clone();
            parent.replace_key(sibling_idx, new_separator);
        }
        sibling.encode(sibling_page.data_mut(), self.key_len)?;
        leaf.encode(leaf_page.data_mut(), self.key_len)?;
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
        let (sibling_idx, sibling_id) = Self::sibling_of(parent, node_idx)?;
        let mut sibling_page = self.bpm.write_page(sibling_id)?;
        let mut sibling = InternalPage::decode(sibling_page.data_mut(), self.key_len)?;
        let sibling_is_right = sibling_idx > node_idx;

        if !internal_has_spare_key(&sibling) {
            if sibling_is_right {
                let separator = parent.keys[node_idx].clone();
                node.absorb_right(separator, &mut sibling);
                parent.remove_separator(node_idx);
                node.encode(node_page.data_mut(), self.key_len)?;
            } else {
                let separator = parent.keys[sibling_idx].clone();
                sibling.absorb_right(separator, &mut node);
                parent.remove_separator(sibling_idx);
                sibling.encode(sibling_page.data_mut(), self.key_len)?;
            }
            return Ok(Rebalance::Merged);
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
        sibling.encode(sibling_page.data_mut(), self.key_len)?;
        node.encode(node_page.data_mut(), self.key_len)?;
        Ok(Rebalance::Redistributed)
    }

    pub fn leaf_pages(&self) -> Result<LeafPagesIter, IndexError> {
        let header = self.read_header()?;
        Ok(LeafPagesIter {
            bpm: self.bpm.clone(),
            key_len: self.key_len,
            next_leaf_id: header.first_leaf_id,
            seen: std::collections::HashSet::new(),
            finished: false,
        })
    }

    /// Prints a best-effort view of the tree structure (page ids, levels, counts, and leaf links).
    pub fn print_tree(&self) -> Result<(), IndexError> {
        let header = self.read_header()?;
        fn fmt_pid(pid: Option<PageId>) -> String {
            pid.map_or_else(|| "INVALID".to_string(), |pid| pid.to_string())
        }
        fn fmt_children(children: &[PageId]) -> String {
            let mut s = String::from("[");
            for (i, &pid) in children.iter().enumerate() {
                if i != 0 {
                    s.push_str(", ");
                }
                s.push_str(&pid.to_string());
            }
            s.push(']');
            s
        }

        println!(
            "BPlusTree {{ index_name: {}, header_page_id: {}, key_len: {}, leaf_max_size: {}, internal_max_size: {} }}",
            self.index_name,
            self.header_page_id,
            self.key_len,
            self.leaf_max_size,
            self.internal_max_size
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
                println!("{prefix}{branch}* cycle/revisit: page {page_id}");
                return Ok(());
            }

            let page = tree.bpm.read_page(page_id)?;
            match LeafPage::decode(page.data(), tree.key_len) {
                Ok(leaf) => {
                    println!(
                        "{prefix}{branch}leaf page_id={} keys={} next_leaf_id={}",
                        page_id,
                        leaf.keys.len(),
                        fmt_pid(leaf.next_leaf_id)
                    );
                    Ok(())
                }
                Err(PageCodecError::WrongPageKind { .. }) => {
                    match InternalPage::decode(page.data(), tree.key_len) {
                        Ok(internal) => {
                            let children = internal.children.clone();
                            println!(
                                "{prefix}{branch}internal page_id={} keys={} children={}",
                                page_id,
                                internal.keys.len(),
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
                                page_id
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
                leaf.keys.len(),
                fmt_pid(leaf.next_leaf_id)
            );
            leaf_id = leaf.next_leaf_id;
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
                "tid={:?} header has no root page (empty tree); header={:?}",
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
            let Some(child_id) = child_for_key(current.data(), self.key_len, key)? else {
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
        while let Some(child_id) = child_for_key(current.data(), self.key_len, key)? {
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
        self.check_key_len(&key)?;

        if let Some(mut leaf_page) = self.latch_leaf_for_write(&key)? {
            let mut leaf = LeafPageView::new(leaf_page.data_mut(), self.key_len)?;
            if leaf.size() > leaf.min_size() {
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
        self.check_key_len(key)?;
        let Some(page) = self.find_leaf(key)? else {
            return Ok(None);
        };
        let leaf = LeafPageView::new(page.data(), self.key_len)?;
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
        let leaf = LeafPageView::new(start_page.data(), self.key_len)?;
        let start_slot = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Excluded(key) => leaf.upper_bound(&key.0),
            Bound::Included(key) => leaf.lower_bound(&key.0),
        };
        let mut buf: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        buf.copy_from_slice(start_page.data());

        Ok(Box::new(BPlusTreeIter {
            bpm: self.bpm.clone(),
            key_len: self.key_len,
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
    key_len: u32,
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
        LeafPageView::new(page.data(), self.key_len)?;
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
            let leaf = match LeafPageView::new(&buf[..], self.key_len) {
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
    use crate::buf::put_u32_le;

    use crate::index::b_plus_page::{
        LEAF_FIXED_HEADER_BYTES, RECORD_ID_ENCODED_BYTES, TOMBSTONE_SLOT_ENCODED_BYTES,
    };

    use crate::table_heap::RecordId;
    use crate::testing::pid;

    const FOUR_ENTRY_LEAF_KEY_LEN: u32 = 2000;

    const NO_TOMBSTONE_KEY_LEN: u32 = 567;

    #[test]
    fn no_tombstone_key_len_gives_fourteen_entry_leaves_without_tombstones() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let tree = BPlusTree::new("idx".to_string(), NO_TOMBSTONE_KEY_LEN, bpm).unwrap();
        assert_eq!(tree.leaf_max_size, 14);
        assert_eq!(
            LeafPage::tombstone_capacity_for_layout(
                tree.leaf_max_size,
                NO_TOMBSTONE_KEY_LEN,
                NUM_TOMBSTONES
            ),
            0
        );
    }

    #[test]
    fn failed_rebalance_leaves_the_underfull_leaf_unchanged() {
        let key_len = NO_TOMBSTONE_KEY_LEN as usize;
        let (bpm, _tmp) = crate::testing::bpm(32);
        let tree = BPlusTree::new("idx".to_string(), NO_TOMBSTONE_KEY_LEN, bpm.clone()).unwrap();
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
            NO_TOMBSTONE_KEY_LEN,
        )
        .unwrap();
        assert_eq!(
            root.children.len(),
            2,
            "15 inserts split the root leaf once"
        );
        let (left_id, right_id) = (root.children[0], root.children[1]);
        let left = tree.decode_leaf(left_id).unwrap();
        assert_eq!(
            left.keys.len() as u32,
            left.min_size(),
            "left leaf sits at min size"
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
        let first_tombstone = LEAF_FIXED_HEADER_BYTES
            + 3 * (key_len + RECORD_ID_ENCODED_BYTES)
            + TOMBSTONE_SLOT_ENCODED_BYTES;
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
        let tree = BPlusTree::new("idx".to_string(), NO_TOMBSTONE_KEY_LEN, bpm.clone()).unwrap();
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
            NO_TOMBSTONE_KEY_LEN,
        )
        .unwrap();
        (tree, root.children[0], root.children[1])
    }

    fn scan_all(tree: &BPlusTree) -> Result<Vec<(IndexKey, IndexValue)>, IndexError> {
        tree.scan((Bound::Unbounded, Bound::Unbounded))?.collect()
    }

    #[test]
    fn a_corrupt_leaf_in_the_middle_of_a_scan_is_an_error() {
        let (bpm, _tmp) = crate::testing::bpm(16);
        let (tree, _, right) = two_leaf_tree(&bpm);
        assert_eq!(scan_all(&tree).unwrap().len(), 15);
        let high_byte_of_first_page_id =
            LEAF_FIXED_HEADER_BYTES + NO_TOMBSTONE_KEY_LEN as usize + 7;
        bpm.write_page(right).unwrap().data_mut()[high_byte_of_first_page_id] = 0xff;
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
        leaf.next_leaf_id = Some(left);
        leaf.encode(
            bpm.write_page(right).unwrap().data_mut(),
            NO_TOMBSTONE_KEY_LEN,
        )
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
        let tree = BPlusTree::new("idx".to_string(), NO_TOMBSTONE_KEY_LEN, bpm).unwrap();
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
    fn four_entry_leaf_key_len_gives_four_entry_leaves() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let tree = BPlusTree::new("idx".to_string(), FOUR_ENTRY_LEAF_KEY_LEN, bpm).unwrap();
        assert_eq!(tree.leaf_max_size, 4);
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
            total += leaf.keys.len();
            leaf_id = leaf.next_leaf_id;
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
            leaf_id = leaf.next_leaf_id;
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
        let root = InternalPage::decode(root_page.data(), header.key_len).unwrap();
        assert_eq!(root.keys.len(), 1);
        assert_eq!(root.children.len(), 2);

        let left_leaf_id = root.get_child(0).unwrap();
        let right_leaf_id = root.get_child(1).unwrap();
        assert_eq!(header.first_leaf_id, Some(*left_leaf_id));

        let left_leaf_page = bpm.read_page(*left_leaf_id).unwrap();
        let left_leaf = LeafPage::decode(left_leaf_page.data(), header.key_len).unwrap();
        let right_leaf_page = bpm.read_page(*right_leaf_id).unwrap();
        let right_leaf = LeafPage::decode(right_leaf_page.data(), header.key_len).unwrap();

        assert_eq!(left_leaf.next_leaf_id, Some(*right_leaf_id));
        assert_eq!(right_leaf.next_leaf_id, None);
        assert_eq!(left_leaf.keys.len(), 2);
        assert_eq!(right_leaf.keys.len(), 3);
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
            let root = InternalPage::decode(root_page.data(), header.key_len).unwrap();
            assert_eq!(root.keys.len(), 1);
            assert_eq!(root.children.len(), 2);

            let left_leaf_id = *root.get_child(0).unwrap();
            let right_leaf_id = *root.get_child(1).unwrap();
            let left_leaf = tree.decode_leaf(left_leaf_id).unwrap();
            let right_leaf = tree.decode_leaf(right_leaf_id).unwrap();
            assert_eq!(left_leaf.keys.len(), 2);
            assert_eq!(right_leaf.keys.len(), 3);
            assert_eq!(left_leaf.next_leaf_id, Some(right_leaf_id));
            assert_eq!(right_leaf.next_leaf_id, None);

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
    fn wrong_key_length_is_a_key_length_error_for_every_operation() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        let tree = BPlusTree::new("idx".to_string(), 8, bpm).unwrap();
        let short = IndexKey(vec![1, 2, 3]);
        let rid = IndexValue::IndexValue(RecordId {
            page_id: pid(1),
            slot_id: 1,
        });
        let is_key_length = |e: IndexError| {
            matches!(
                e,
                IndexError::KeyLength {
                    expected: 8,
                    actual: 3
                }
            )
        };
        assert!(is_key_length(tree.insert(short.clone(), rid).unwrap_err()));
        assert!(is_key_length(tree.remove(short.clone()).unwrap_err()));
        assert!(is_key_length(tree.search(&short).unwrap_err()));
    }

    #[test]
    fn key_lengths_that_cannot_fit_a_page_are_rejected() {
        let (bpm, _tmp) = crate::testing::bpm(10);
        for key_len in [0, 1 << 20] {
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
            leaves_in_order: &mut Vec<(PageId, usize)>,
        ) {
            let page = tree.bpm.read_page(page_id).unwrap();
            match Page::decode(page.data(), tree.key_len).unwrap() {
                Page::Leaf(leaf) => {
                    assert!(
                        leaf.keys.iter().all(|k| bounds.contains(k)),
                        "leaf {page_id} holds a key outside its parent's separator range"
                    );
                    leaves_in_order.push((page_id, depth));
                }
                Page::Internal(node) => {
                    if is_root {
                        assert!(
                            node.children.len() >= 2,
                            "internal root {page_id} has fewer than two children"
                        );
                    } else {
                        assert!(
                            node.keys.len() >= node.min_size() as usize,
                            "non-root internal page {page_id} is below minimum occupancy"
                        );
                    }
                    assert!(
                        node.keys.iter().all(|k| bounds.contains(k)),
                        "internal page {page_id} holds a separator outside its parent's range"
                    );
                    drop(page);
                    for (i, &child) in node.children.iter().enumerate() {
                        let child_bounds = SubtreeBounds {
                            lower: if i == 0 {
                                bounds.lower.clone()
                            } else {
                                Some(node.keys[i - 1].clone())
                            },
                            upper: node.keys.get(i).cloned().or(bounds.upper.clone()),
                        };
                        walk_subtree(tree, child, child_bounds, depth + 1, false, leaves_in_order);
                    }
                }
            }
        }

        fn assert_tree_is_well_formed(tree: &BPlusTree) {
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
            walk_subtree(tree, root_page_id, unbounded, 0, true, &mut leaves_in_order);
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
            key_len: usize,
            ops: usize,
            _bpm: Arc<BufferPoolManager>,
            _tmp: tempfile::TempDir,
        }

        impl Harness {
            fn new(key_len: usize) -> Self {
                let (bpm, tmp) = crate::testing::bpm(64);
                let tree = BPlusTree::new("idx".to_string(), key_len as u32, bpm.clone()).unwrap();
                Self {
                    tree,
                    model: BTreeMap::new(),
                    key_len,
                    ops: 0,
                    _bpm: bpm,
                    _tmp: tmp,
                }
            }

            fn insert(&mut self, n: u32) {
                self.tree
                    .insert(key_for(n, self.key_len), IndexValue::IndexValue(rid_for(n)))
                    .unwrap_or_else(|e| panic!("insert {n} failed: {e}"));
                self.model.insert(n, rid_for(n));
                self.after_op(n);
            }

            fn remove(&mut self, n: u32) {
                self.tree
                    .remove(key_for(n, self.key_len))
                    .unwrap_or_else(|e| panic!("remove {n} failed: {e}"));
                self.model.remove(&n);
                self.after_op(n);
            }

            fn after_op(&mut self, n: u32) {
                let expected = self.model.get(&n).copied();
                let found = self
                    .tree
                    .search(&key_for(n, self.key_len))
                    .unwrap()
                    .map(|IndexValue::IndexValue(rid)| rid);
                assert_eq!(found, expected, "search({n}) disagrees with the model");
                self.ops += 1;
                if self.ops.is_multiple_of(SCAN_CHECK_EVERY) {
                    self.check_full_scan();
                }
            }

            fn check_full_scan(&self) {
                assert_tree_is_well_formed(&self.tree);
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
                    .map(|(&n, &rid)| (key_for(n, self.key_len), rid))
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
                        self.tree
                            .search(&key_for(n, self.key_len))
                            .unwrap()
                            .is_some(),
                        "search({n}) lost a live key after {} ops",
                        self.ops
                    );
                }
            }
        }

        fn insert_then_remove_all(key_len: usize, inserts: &[u32], removes: &[u32]) {
            let mut h = Harness::new(key_len);
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

        const KEY_LENS: [usize; 3] = [2000, 600, 64];
        const N: u32 = 2500;

        #[test]
        fn sequential_insert_then_sequential_remove_keeps_tree_consistent() {
            let keys: Vec<u32> = (0..N).collect();
            for key_len in KEY_LENS {
                insert_then_remove_all(key_len, &keys, &keys);
            }
        }

        #[test]
        fn reversed_insert_then_reversed_remove_keeps_tree_consistent() {
            let keys: Vec<u32> = (0..N).rev().collect();
            for key_len in KEY_LENS {
                insert_then_remove_all(key_len, &keys, &keys);
            }
        }

        #[test]
        fn sequential_insert_then_reversed_remove_keeps_tree_consistent() {
            let keys: Vec<u32> = (0..N).collect();
            let rev: Vec<u32> = (0..N).rev().collect();
            for key_len in KEY_LENS {
                insert_then_remove_all(key_len, &keys, &rev);
            }
        }

        #[test]
        fn random_insert_then_random_remove_keeps_tree_consistent() {
            for (seed, key_len) in KEY_LENS.into_iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(seed as u64);
                let mut inserts: Vec<u32> = (0..N).collect();
                inserts.shuffle(&mut rng);
                let mut removes = inserts.clone();
                removes.shuffle(&mut rng);
                insert_then_remove_all(key_len, &inserts, &removes);
            }
        }

        #[test]
        fn removed_keys_can_be_reinserted_and_removed_again() {
            for (seed, key_len) in KEY_LENS.into_iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(200 + seed as u64);
                let mut h = Harness::new(key_len);
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
            for (seed, key_len) in KEY_LENS.into_iter().enumerate() {
                let mut rng = StdRng::seed_from_u64(100 + seed as u64);
                let mut h = Harness::new(key_len);
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
                assert_tree_is_well_formed(tree);
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
                    tree.internal_max_size, 4,
                    "test expects four-key internal pages"
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
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::collections::BTreeMap;
        use std::collections::btree_map::Entry;

        let key_len = 400usize;
        let (bpm, _dir) = crate::testing::bpm(512);
        let tree = BPlusTree::new("idx".to_string(), key_len as u32, bpm).unwrap();
        assert!(tree.leaf_max_size < 25 && tree.internal_max_size < 25);

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut model: BTreeMap<IndexKey, RecordId> = BTreeMap::new();

        for step in 0..6000u32 {
            let k = rng.random_range(0..800u16);
            let key = wide_key(k, key_len);
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
                    let key = wide_key(probe, key_len);
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

                let (lo, hi) = (wide_key(200, key_len), wide_key(600, key_len));
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
