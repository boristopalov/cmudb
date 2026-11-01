use crate::buffer_pool::{BufferPoolManager, PageMut, PageRef};
use crate::catalog::index_schema::{IndexKey, IndexValue};
use crate::disk::PAGE_SIZE;
use crate::index::b_plus_page::{
    HeaderPage, INVALID_PAGE_ID, InternalPage, LeafPage, NUM_TOMBSTONES, Page, PageId, PageKind,
};
use crate::index::context::TreeContext;
use crate::index::page_codec::PageCodecError;
use crate::index::page_view::{HeaderPageView, LeafInsertError, LeafPageView, NodeView};
use crate::index::{
    Index, IndexError, IndexIter, InsertError, InsertResult, RemoveError, RemoveResult,
};
use crate::table_heap::RecordId;
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
const UNNUMBERED_FRAME: IndexError = IndexError::Structure("latched frame has no page id");
const SPLIT_ABOVE_LATCHED_PATH: IndexError =
    IndexError::Structure("a split reached the highest latched page, which is not the root");
const NO_CHILD_FOR_KEY: IndexError =
    IndexError::Structure("internal page has no child to descend into for the key");

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
/// holds a leaf.
fn child_for_key(buf: &[u8], key_len: u32, key: &IndexKey) -> Result<Option<PageId>, IndexError> {
    match NodeView::new(buf, key_len)? {
        NodeView::Leaf(_) => Ok(None),
        NodeView::Internal(node) => node.child_for(&key.0).ok_or(NO_CHILD_FOR_KEY).map(Some),
    }
}

fn page_id_of(page: &PageMut<'_>) -> Result<PageId, IndexError> {
    page.page_id().ok_or(UNNUMBERED_FRAME)
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
    next_leaf_id: PageId,
    seen: HashSet<PageId>,
    finished: bool,
}

impl Iterator for LeafPagesIter {
    type Item = Result<(PageId, LeafPage), IndexError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.next_leaf_id == INVALID_PAGE_ID {
            self.finished = true;
            return None;
        }

        let leaf_id = self.next_leaf_id;
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
            let pid = page_id_of(&page)?;
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
            root_page_id: view.root_page_id(),
            first_leaf_id: view.first_leaf_id(),
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
        Ok((page_id_of(&page)?, page))
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

        if header.root_page_id == INVALID_PAGE_ID {
            info!("insert into empty tree: creating new tree");
            let (leaf_id, mut leaf_page) = self.allocate_page()?;
            let mut leaf = self.new_leaf();
            leaf.insert(key, record_id)
                .map_err(|_| IndexError::Insert(InsertError::DuplicateKey))?;
            leaf.encode(leaf_page.data_mut(), self.key_len)?;
            header.root_page_id = leaf_id;
            header.first_leaf_id = leaf_id;
            Self::store_header(&mut header_page, &header)?;
            return Ok(());
        }

        let mut ctx = TreeContext::new(header_page, header.root_page_id);
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
            left_id: page_id_of(&leaf_page)?,
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
                left_id: page_id_of(&parent_page)?,
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
        header.root_page_id = root_id;
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
        if header.root_page_id == INVALID_PAGE_ID {
            return Err(IndexError::Remove(RemoveError::KeyNotFound));
        }

        let mut ctx = TreeContext::new(header_page, header.root_page_id);
        self.latch_path_to_leaf(&mut ctx, &key, safe_for_remove)?;
        let mut leaf_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
        let leaf_is_root = page_id_of(&leaf_page)? == ctx.root_page_id;
        let (leaf_is_empty, leaf_is_underfull) = {
            let mut view = LeafPageView::new(leaf_page.data_mut(), self.key_len)?;
            view.remove_key(&key.0)?;
            (view.size() == 0, view.is_underfull())
        };

        if leaf_is_root {
            if leaf_is_empty {
                header.root_page_id = INVALID_PAGE_ID;
                header.first_leaf_id = INVALID_PAGE_ID;
                let header_page = ctx.header.as_mut().ok_or(HEADER_LATCH_RELEASED)?;
                Self::store_header(header_page, &header)?;
            }
            return Ok(());
        }
        if !leaf_is_underfull {
            return Ok(());
        }
        let leaf = LeafPage::decode(leaf_page.data(), self.key_len)?;

        let mut parent_page = ctx.write_set.pop_back().ok_or(LATCH_NOT_HELD)?;
        let mut parent = InternalPage::decode(parent_page.data_mut(), self.key_len)?;
        let mut outcome = self.rebalance_leaf(&key, leaf, &mut leaf_page, &mut parent)?;
        parent.encode(parent_page.data_mut(), self.key_len)?;

        while outcome == Rebalance::Merged {
            if parent_page.page_id() == Some(ctx.root_page_id) {
                if parent.children.len() == 1 {
                    header.root_page_id = parent.children[0];
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
        fn fmt_pid(pid: PageId) -> String {
            if pid == INVALID_PAGE_ID {
                "INVALID".to_string()
            } else {
                pid.to_string()
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
            "BPlusTree {{ index_name: {}, header_page_id: {}, key_len: {}, leaf_max_size: {}, internal_max_size: {} }}",
            self.index_name,
            fmt_pid(self.header_page_id),
            self.key_len,
            self.leaf_max_size,
            self.internal_max_size
        );
        println!(
            "HeaderPage {{ root_page_id: {}, first_leaf_id: {} }}",
            fmt_pid(header.root_page_id),
            fmt_pid(header.first_leaf_id)
        );

        if header.root_page_id == INVALID_PAGE_ID {
            println!("(empty)");
            return Ok(());
        }

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
            match LeafPage::decode(page.data(), tree.key_len) {
                Ok(leaf) => {
                    println!(
                        "{prefix}{branch}leaf page_id={} keys={} next_leaf_id={}",
                        fmt_pid(page_id),
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
                                fmt_pid(page_id),
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

                            let printable_children: Vec<PageId> = children
                                .into_iter()
                                .filter(|&c| c != INVALID_PAGE_ID)
                                .collect();
                            for (i, child) in printable_children.iter().copied().enumerate() {
                                let child_is_last = i + 1 == printable_children.len();
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
            header.root_page_id,
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
        while leaf_id != INVALID_PAGE_ID {
            if idx >= PRINT_TREE_LIMIT {
                println!("  ... truncated after {idx} leaves");
                break;
            }
            if !leaf_seen.insert(leaf_id) {
                println!("  * cycle/revisit at leaf {}", fmt_pid(leaf_id));
                break;
            }

            let leaf = self.decode_leaf(leaf_id)?;
            println!(
                "  [{idx}] page_id={} keys={} next_leaf_id={}",
                fmt_pid(leaf_id),
                leaf.keys.len(),
                fmt_pid(leaf.next_leaf_id)
            );
            leaf_id = leaf.next_leaf_id;
            idx += 1;
        }

        Ok(())
    }

    fn get_root_page_from_header(&self) -> Result<Option<PageRef<'_>>, IndexError> {
        let header = self.read_header()?;
        if header.root_page_id == INVALID_PAGE_ID {
            warn!(
                "tid={:?} header root_page_id is invalid (empty tree); header={:?}",
                std::thread::current().id(),
                header
            );
            return Ok(None);
        }
        Ok(Some(self.bpm.read_page(header.root_page_id)?))
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
        let child_id = child.page_id().ok_or(UNNUMBERED_FRAME)?;
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
        Ok(leaf.get(&key.0).map(IndexValue::IndexValue))
    }

    fn scan(&self, range: (Bound<IndexKey>, Bound<IndexKey>)) -> Result<IndexIter, IndexError> {
        let header = self.read_header()?;
        if header.root_page_id == INVALID_PAGE_ID {
            return Ok(Box::new(std::iter::empty()));
        }
        let start_page = match range.start_bound() {
            Bound::Unbounded => Some(self.bpm.read_page(header.first_leaf_id)?),
            Bound::Excluded(key) | Bound::Included(key) => self.find_leaf(key)?,
        };
        let Some(start_page) = start_page else {
            return Ok(Box::new(std::iter::empty()));
        };
        let start_leaf_id = start_page.page_id().ok_or(UNNUMBERED_FRAME)?;
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
    fn load_leaf(&mut self, leaf_id: PageId) -> Result<bool, IndexError> {
        if leaf_id == INVALID_PAGE_ID {
            return Ok(false);
        }
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
                let next_leaf_id = leaf.next_leaf_id();
                match self.load_leaf(next_leaf_id) {
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

            let val = IndexValue::IndexValue(leaf.record_id(i));
            return Some(Ok((IndexKey(key.to_vec()), val)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::replacer::ArcReplacer;
    use crate::table_heap::RecordId;
    use tempfile::tempdir;

    const FOUR_ENTRY_LEAF_KEY_LEN: u32 = 2000;

    #[test]
    fn four_entry_leaf_key_len_gives_four_entry_leaves() {
        let (bpm, _tmp) = make_bpm(10);
        let tree = BPlusTree::new("idx".to_string(), FOUR_ENTRY_LEAF_KEY_LEN, bpm).unwrap();
        assert_eq!(tree.leaf_max_size, 4);
    }

    fn make_bpm(num_frames: usize) -> (Arc<BufferPoolManager>, tempfile::TempDir) {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("db");
        let log_path = dir.path().join("log");
        let dm = DiskManager::new(&db_path, &log_path).expect("disk manager");
        let scheduler = DiskScheduler::new(dm);
        let replacer = ArcReplacer::new(num_frames);
        (
            Arc::new(create_buffer_pool_manager(num_frames, replacer, scheduler)),
            dir,
        )
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
        while leaf_id != INVALID_PAGE_ID {
            assert!(seen.insert(leaf_id), "cycle in leaf chain at {leaf_id}");
            let leaf = tree.decode_leaf(leaf_id).expect("leaf decode");
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
        while leaf_id != INVALID_PAGE_ID {
            assert!(seen.insert(leaf_id), "cycle in leaf chain at {leaf_id}");
            let leaf = tree.decode_leaf(leaf_id).expect("leaf decode");
            count += 1;
            leaf_id = leaf.next_leaf_id;
        }
        count
    }

    #[test]
    fn leaf_pages_iterator_walks_leaf_chain() {
        let (bpm, _tmp) = make_bpm(10);
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
                    page_id: i as u32,
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
                assert_eq!(rid.page_id as u8, k.0[0]);
                assert_eq!(rid.slot_id, rid.page_id + 10);
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
        let (bpm, _tmp) = make_bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: i as u32,
                    slot_id: 0,
                }),
            )
            .unwrap();
        }

        let header = tree.read_header().unwrap();
        assert_ne!(header.root_page_id, INVALID_PAGE_ID);
        assert_ne!(header.first_leaf_id, INVALID_PAGE_ID);

        let root_page = bpm.read_page(header.root_page_id).unwrap();
        let root = InternalPage::decode(root_page.data(), header.key_len).unwrap();
        assert_eq!(root.keys.len(), 1);
        assert_eq!(root.children.len(), 2);

        let left_leaf_id = root.get_child(0).unwrap();
        let right_leaf_id = root.get_child(1).unwrap();
        assert_eq!(header.first_leaf_id, *left_leaf_id);

        let left_leaf_page = bpm.read_page(*left_leaf_id).unwrap();
        let left_leaf = LeafPage::decode(left_leaf_page.data(), header.key_len).unwrap();
        let right_leaf_page = bpm.read_page(*right_leaf_id).unwrap();
        let right_leaf = LeafPage::decode(right_leaf_page.data(), header.key_len).unwrap();

        assert_eq!(left_leaf.next_leaf_id, *right_leaf_id);
        assert_eq!(right_leaf.next_leaf_id, INVALID_PAGE_ID);
        assert_eq!(left_leaf.keys.len(), 2);
        assert_eq!(right_leaf.keys.len(), 3);
    }

    #[test]
    fn insert_duplicate_key_errors() {
        let (bpm, _tmp) = make_bpm(10);
        let key_len = 32u32;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        let key = make_key(7, key_len as usize);
        tree.insert(
            key.clone(),
            IndexValue::IndexValue(RecordId {
                page_id: 1,
                slot_id: 1,
            }),
        )
        .unwrap();

        let err = tree
            .insert(
                key,
                IndexValue::IndexValue(RecordId {
                    page_id: 2,
                    slot_id: 2,
                }),
            )
            .unwrap_err();
        assert!(matches!(err, IndexError::Insert(InsertError::DuplicateKey)));
    }

    #[test]
    fn print_tree_smoke() {
        let (bpm, _tmp) = make_bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: i as u32,
                    slot_id: 0,
                }),
            )
            .unwrap();
        }

        tree.print_tree().unwrap();
    }

    #[test]
    fn remove_missing_key_errors_and_does_not_mutate_tree() {
        let (bpm, _tmp) = make_bpm(10);
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
                    page_id: i as u32,
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
        let (bpm, _tmp) = make_bpm(10);
        let key_len = 32u32;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        let key = make_key(7, key_len as usize);
        tree.insert(
            key.clone(),
            IndexValue::IndexValue(RecordId {
                page_id: 1,
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
        let (bpm, _tmp) = make_bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: i as u32,
                    slot_id: 0,
                }),
            )
            .unwrap();
        }

        let (_left_leaf_id, _right_leaf_id) = {
            let header = tree.read_header().unwrap();
            let root_page = bpm.read_page(header.root_page_id).unwrap();
            let root = InternalPage::decode(root_page.data(), header.key_len).unwrap();
            assert_eq!(root.keys.len(), 1);
            assert_eq!(root.children.len(), 2);

            let left_leaf_id = *root.get_child(0).unwrap();
            let right_leaf_id = *root.get_child(1).unwrap();
            let left_leaf = tree.decode_leaf(left_leaf_id).unwrap();
            let right_leaf = tree.decode_leaf(right_leaf_id).unwrap();
            assert_eq!(left_leaf.keys.len(), 2);
            assert_eq!(right_leaf.keys.len(), 3);
            assert_eq!(left_leaf.next_leaf_id, right_leaf_id);
            assert_eq!(right_leaf.next_leaf_id, INVALID_PAGE_ID);

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
        let (bpm, _tmp) = make_bpm(10);
        let tree = BPlusTree::new("idx".to_string(), 8, bpm).unwrap();
        let short = IndexKey(vec![1, 2, 3]);
        let rid = IndexValue::IndexValue(RecordId {
            page_id: 1,
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
        let (bpm, _tmp) = make_bpm(10);
        for key_len in [0, 1 << 20] {
            let err = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap_err();
            assert!(matches!(err, IndexError::UnsupportedKeyLength(l) if l == key_len));
        }
    }

    #[test]
    fn search_on_empty_tree_returns_none() {
        let (bpm, _tmp) = make_bpm(10);
        let key_len = 32u32;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm).unwrap();
        let key = make_key(1, key_len as usize);
        assert!(tree.search(&key).unwrap().is_none());
        let from_key = (Bound::Included(key.clone()), Bound::Unbounded);
        assert_eq!(tree.scan(from_key).unwrap().count(), 0);
    }

    #[test]
    fn search_finds_inserted_keys_and_returns_none_for_missing() {
        let (bpm, _tmp) = make_bpm(10);
        let key_len = FOUR_ENTRY_LEAF_KEY_LEN;
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        for i in 0u8..5u8 {
            tree.insert(
                make_key(i, key_len as usize),
                IndexValue::IndexValue(RecordId {
                    page_id: i as u32,
                    slot_id: i as u32 + 10,
                }),
            )
            .unwrap();
        }

        for i in 0u8..5u8 {
            let key = make_key(i, key_len as usize);
            let IndexValue::IndexValue(rid) = tree.search(&key).unwrap().expect("key present");
            assert_eq!(rid.page_id, i as u32);
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
                page_id: n,
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
            if header.root_page_id == INVALID_PAGE_ID {
                assert_eq!(
                    header.first_leaf_id, INVALID_PAGE_ID,
                    "empty tree still points at a first leaf"
                );
                return;
            }
            let mut leaves_in_order = Vec::new();
            let unbounded = SubtreeBounds {
                lower: None,
                upper: None,
            };
            walk_subtree(
                tree,
                header.root_page_id,
                unbounded,
                0,
                true,
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
            key_len: usize,
            ops: usize,
            _bpm: Arc<BufferPoolManager>,
            _tmp: tempfile::TempDir,
        }

        impl Harness {
            fn new(key_len: usize) -> Self {
                let (bpm, tmp) = make_bpm(64);
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
        let (bpm, _dir) = make_bpm(512);
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
                    page_id: step,
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
}
