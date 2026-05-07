use crate::buffer_pool::{BufferPoolManager, PageMut, PageRef};
use crate::catalog::index_schema::{IndexKey, IndexValue};
use crate::index::b_plus_page::{
    HeaderPage, INVALID_PAGE_ID, InternalPage, LeafPage, NUM_TOMBSTONES, Page, PageId, PageKind,
};
use crate::index::context::TreeContext;
use crate::index::page_codec::PageCodecError;
use crate::index::{
    Index, IndexError, IndexIter, InsertError, InsertResult, RemoveError, RemoveResult,
};
use log::{debug, error, info, warn};

use std::collections::HashSet;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

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
    seen: std::collections::HashSet<PageId>,
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

// Invariants:
//      - each internal node has n-1 keys and n childrem
//      - root has at least two children
//      - for each key in internal node:
//              - left subtree's keys are all l.t. key
//              - right subtree's keys are all g.t. key
//      - each internal node has at least (n / 2) children filled
//      - every leaf node has at least (n / 2) keys
//      - every key in the table is in a leaf node
impl BPlusTree {
    pub fn new(
        index_name: String,
        key_len: u32,
        bpm: Arc<BufferPoolManager>,
    ) -> Result<Self, IndexError> {
        let key_len_usize =
            usize::try_from(key_len).map_err(|_| IndexError::Insert(InsertError::InvalidKey))?;
        if key_len_usize == 0 {
            return Err(IndexError::Insert(InsertError::InvalidKey));
        }

        let leaf_capacity = LeafPage::max_size_for_layout(key_len_usize)
            .ok_or(IndexError::Insert(InsertError::InvalidKey))?;
        let internal_capacity = InternalPage::max_size_for_layout(key_len_usize)
            .ok_or(IndexError::Insert(InsertError::InvalidKey))?;
        if leaf_capacity == 0 || internal_capacity == 0 {
            return Err(IndexError::Insert(InsertError::InvalidKey));
        }

        let header_page_id = {
            let mut page = bpm.new_page().map_err(IndexError::BpmError)?;
            let pid = page.page_id().unwrap();
            let header = HeaderPage {
                root_page_id: INVALID_PAGE_ID,
                first_leaf_id: INVALID_PAGE_ID,
                key_len,
            };
            header
                .encode(page.data_mut())
                .map_err(IndexError::PageError)?;
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

    /// Acquires a read latch on the header page, decodes it,
    /// and returns the result.
    fn read_header(&self) -> Result<HeaderPage, IndexError> {
        info!("decoding b_plus_tree header");
        let page = self
            .bpm
            .read_page(self.header_page_id)
            .map_err(IndexError::BpmError)?;
        let header = HeaderPage::decode(page.data()).map_err(IndexError::PageError)?;
        if header.key_len != self.key_len {
            return Err(IndexError::Insert(InsertError::InvalidKey));
        }
        Ok(header)
    }

    fn decode_leaf(&self, page_id: PageId) -> Result<LeafPage, IndexError> {
        let page = self.bpm.read_page(page_id).map_err(IndexError::BpmError)?;
        LeafPage::decode(page.data(), self.key_len).map_err(IndexError::PageError)
    }

    fn insert_pessimistic(&self, key: IndexKey, val: IndexValue) -> Result<(), IndexError> {
        let record_id = match val {
            IndexValue::IndexValue(rid) => rid,
            _ => return Err(IndexError::Insert(InsertError::InvalidValue)),
        };

        let mut ctx = TreeContext::new();

        // acquire a write latch on the header
        let mut header_page = self
            .bpm
            .write_page(self.header_page_id)
            .map_err(IndexError::BpmError)?;
        let mut header =
            HeaderPage::decode(header_page.data_mut()).map_err(IndexError::PageError)?;

        ctx.header = Some(header_page);
        ctx.root_page_id = header.root_page_id;

        if header.root_page_id == INVALID_PAGE_ID {
            info!("insert into empty tree: creating new tree");
            let (new_leaf_id, mut new_leaf_page) = {
                let page = self.bpm.new_page().map_err(IndexError::BpmError)?;
                (page.page_id().unwrap(), page)
            };
            let tombstone_cap = LeafPage::tombstone_capacity_for_layout(
                self.leaf_max_size,
                self.key_len,
                NUM_TOMBSTONES,
            );
            let mut leaf = LeafPage::new(self.leaf_max_size, tombstone_cap);
            leaf.insert(key, record_id)
                .map_err(|_| IndexError::Insert(InsertError::DuplicateKey))?;
            leaf.encode(new_leaf_page.data_mut(), self.key_len)
                .map_err(IndexError::PageError)?;

            header.root_page_id = new_leaf_id;
            header.first_leaf_id = new_leaf_id;
            header.encode(ctx.header.as_mut().unwrap().data_mut())?;
            return Ok(());
        }

        // read the root page and add it to the traversal context
        ctx.write_set
            .push_back(self.bpm.write_page(header.root_page_id)?);

        // look for the leaf to insert in
        let mut leaf = loop {
            let current = ctx.write_set.back_mut().unwrap();
            match Page::decode(current.data_mut(), self.key_len)? {
                Page::Internal(node) => {
                    let next_page_id = node.children[node.find_child_index(&key)];

                    // acquire a write latch on the next page
                    let mut next_page = self
                        .bpm
                        .write_page(next_page_id)
                        .map_err(IndexError::BpmError)?;

                    // now we see if we can release the ancestors or not
                    // we can release ancestors if the child is "safe",
                    // i.e. we know it can't be split on this insert
                    // if a leaf is not full, it means it can accept another key without splitting.
                    let child_is_safe = match Page::decode(next_page.data_mut(), self.key_len)? {
                        Page::Internal(internal) => !internal.is_full(),
                        Page::Leaf(leaf) => !leaf.is_full(),
                    };

                    // make sure to add the child page to the write_set before dropping latches
                    ctx.write_set.push_back(next_page);
                    if child_is_safe {
                        ctx.drop_write_ancestors();
                    }
                }
                Page::Leaf(leaf) => break leaf,
            }
        };
        // add the key to the leaf's list of keys.
        leaf.insert(key, record_id)
            .map_err(|_| IndexError::Insert(InsertError::DuplicateKey))?;

        let current = ctx.write_set.back_mut().unwrap();
        if !leaf.is_overfull() {
            // leaf has space, straightforward case
            leaf.encode(current.data_mut(), self.key_len)
                .map_err(IndexError::PageError)?;
            return Ok(());
        }

        // leaf has overfilled, must split it and reorg the tree
        // 1. create a new page for the new leaf
        // the new leaf will be on the RHS
        let (mut right_child_id, mut right_child_page) = {
            let page = self.bpm.new_page().map_err(IndexError::BpmError)?;
            (page.page_id().unwrap(), page)
        };
        let tombstone_cap = LeafPage::tombstone_capacity_for_layout(
            self.leaf_max_size,
            self.key_len,
            NUM_TOMBSTONES,
        );
        let mut right_leaf = LeafPage::new(self.leaf_max_size, tombstone_cap);

        // 2. split the leaf and get the separator key
        // right_child sits immediately after left_child, with sep_key between them
        let mut sep_key = leaf.split_into(&mut right_leaf, right_child_id);
        leaf.encode(current.data_mut(), self.key_len)
            .map_err(IndexError::PageError)?;
        right_leaf
            .encode(right_child_page.data_mut(), self.key_len)
            .map_err(IndexError::PageError)?;
        // TODO: no unwrap
        let mut left_child_id = current.page_id().unwrap();

        // 3. propagate up the tree using the context we have built up,
        // making sure the split leafs are correctly ordered
        // in the parent's list of children (and keys)
        let mut root_split = false;
        let mut root_sep_key: Option<IndexKey> = None;

        // explicitly pop, since the write_set has the leaf at the end
        // we don't need to actually do anything with the leaf here
        ctx.write_set.pop_back();

        // if, after popping the leaf, the traversal context write set is empty,
        // it means that the root was the leaf
        if ctx.write_set.is_empty() {
            root_split = true;
            root_sep_key = Some(sep_key);
        } else {
            while let Some(mut parent) = ctx.write_set.pop_back() {
                let mut parent_node = InternalPage::decode(parent.data_mut(), self.key_len)?;
                parent_node.insert_separator(sep_key, right_child_id);

                if !parent_node.is_overfull() {
                    parent_node.encode(parent.data_mut(), self.key_len)?;
                    break;
                }

                // if the parent node is full, split again
                (right_child_id, right_child_page) = {
                    let page = self.bpm.new_page().map_err(IndexError::BpmError)?;
                    (page.page_id().unwrap(), page)
                };
                let mut right_internal = InternalPage::new(self.internal_max_size);
                sep_key = parent_node.split_into(&mut right_internal);

                parent_node
                    .encode(parent.data_mut(), self.key_len)
                    .map_err(IndexError::PageError)?;
                right_internal
                    .encode(right_child_page.data_mut(), self.key_len)
                    .map_err(IndexError::PageError)?;

                left_child_id = parent.page_id().unwrap();
                if parent.page_id().unwrap() == ctx.root_page_id {
                    root_split = true;
                    root_sep_key = Some(sep_key);
                    break;
                }
            }
        }

        // if the root node was split, we need to allocate a new root node.
        // even though 'parent' latch was dropped, we should still be free of potential races here since
        // we hold a latch on the header throughout the pessimstic insert process,
        // so we know that header.root_page_id has not changed
        if root_split {
            let mut new_root_page = self.bpm.new_page()?;
            header.root_page_id = new_root_page.page_id().unwrap();
            let mut new_root = InternalPage::new(self.internal_max_size);
            new_root.children.push(left_child_id);
            new_root.insert_separator(root_sep_key.unwrap(), right_child_id);
            new_root
                .encode(new_root_page.data_mut(), self.key_len)
                .map_err(IndexError::PageError)?;
            header.encode(ctx.header.as_mut().unwrap().data_mut())?;
        }
        Ok(())
    }

    fn remove_pessimistic(&self, key: IndexKey) -> RemoveResult {
        let mut ctx = TreeContext::new();

        // acquire a write latch on the header
        let mut header_page = self
            .bpm
            .write_page(self.header_page_id)
            .map_err(IndexError::BpmError)?;
        let mut header =
            HeaderPage::decode(header_page.data_mut()).map_err(IndexError::PageError)?;

        if header.root_page_id == INVALID_PAGE_ID {
            return Err(IndexError::Remove(RemoveError::KeyNotFound));
        }

        ctx.header = Some(header_page);
        ctx.root_page_id = header.root_page_id;

        // read the root page and add it to the traversal context
        ctx.write_set
            .push_back(self.bpm.write_page(header.root_page_id)?);

        let mut leaf = loop {
            let current = ctx.write_set.back_mut().unwrap();
            match Page::decode(current.data_mut(), self.key_len)? {
                Page::Internal(node) => {
                    let next_page_id = node.children[node.find_child_index(&key)];

                    // acquire a write latch on the next page
                    let mut next_page = self
                        .bpm
                        .write_page(next_page_id)
                        .map_err(IndexError::BpmError)?;

                    // now we see if we can release the ancestors or not
                    // we can release ancestors if the child is "safe",
                    // i.e. we know it won't be merged with another node if removing a key from it.
                    // if a leaf is > the min size, it means it can remove another key without merging.
                    let child_is_safe = match Page::decode(next_page.data_mut(), self.key_len)? {
                        Page::Internal(internal) => !internal.would_be_underfull(),
                        Page::Leaf(leaf) => !leaf.would_be_underfull(),
                    };

                    // make sure to add the child page to the write_set before dropping latches
                    ctx.write_set.push_back(next_page);
                    if child_is_safe {
                        ctx.drop_write_ancestors();
                    }
                }
                Page::Leaf(mut leaf) => {
                    leaf.remove_key(&key)?;
                    if current.page_id().unwrap() == header.root_page_id {
                        leaf.encode(current.data_mut(), self.key_len)
                            .map_err(IndexError::PageError)?;
                        if leaf.keys.len() == 0 {
                            // empty root
                            header.root_page_id = INVALID_PAGE_ID;
                            header.first_leaf_id = INVALID_PAGE_ID;
                            header.encode(ctx.header.as_mut().unwrap().data_mut())?;
                        }
                        return Ok(());
                    }
                    if !leaf.is_underfull() {
                        leaf.encode(current.data_mut(), self.key_len)
                            .map_err(IndexError::PageError)?;
                        return Ok(());
                    }
                    break leaf;
                }
            }
        };

        // we know leaf is underfilled if we arrive here
        let mut current = ctx.write_set.pop_back().unwrap();
        let parent_page = ctx.write_set.back_mut().unwrap();
        let mut parent = InternalPage::decode(parent_page.data_mut(), self.key_len)?;
        // TODO: we use find_child_index here, it's doing duplicate work since we already found indices while descending
        // probably not a huge deal but can be a performance improvement if we store the indices.
        let parent_idx_of_leaf = parent.find_child_index(&key);

        // get a sibling of this node
        // if the sibling can afford losing a key, move it to this node
        // if not (i.e. sibling would be underfull if key moved), we need to merge the sibling with this node.
        // in this case we must update the parent as it would have 1 less child after the merge, so we must remove
        // a separator key from the parent to maintain the invariant of sep_keys.len() == children.len() - 1
        // if this node is already the left-most child of the parent, then the sibling is the child to the right of this node
        // otherwise it's the sibling to the left
        let sibling_idx = if parent_idx_of_leaf == 0 {
            1
        } else {
            parent_idx_of_leaf - 1
        };
        let sibling_is_right = sibling_idx > parent_idx_of_leaf;

        // latch the sibling
        if let Some(sibling_page_id) = parent.get_child(sibling_idx) {
            let mut sibling_page = self.bpm.write_page(*sibling_page_id)?;
            match LeafPage::decode(sibling_page.data_mut(), self.key_len) {
                Ok(mut sibling) => {
                    if sibling.would_be_underfull() {
                        // we need to merge
                        if sibling_is_right {
                            leaf.absorb_right(&mut sibling);
                            parent.remove_separator(parent_idx_of_leaf);
                        } else {
                            sibling.absorb_right(&mut leaf);
                            parent.remove_separator(parent_idx_of_leaf - 1);
                        }
                        leaf.encode(current.data_mut(), self.key_len)?;
                        sibling.encode(sibling_page.data_mut(), self.key_len)?;
                        parent.encode(parent_page.data_mut(), self.key_len)?;
                    } else {
                        // we can redistribute from the sibling
                        if sibling_is_right {
                            // sibling is to the right, take its smallest entry,
                            // and move it to the back of this node
                            if let Some(res) = sibling.pop_first() {
                                leaf.push_back(res.0, res.1, res.2);
                                // update the separator key
                                if let Some(new_sep_key) = sibling.keys.get(0) {
                                    parent.keys[parent_idx_of_leaf] = new_sep_key.clone();
                                } else {
                                    error!("failed to get sibling.keys[0]")
                                }
                            } else {
                                error!("tried to pop from sibling to redistribute but failed");
                                return Err(IndexError::Remove(RemoveError::GenericError(
                                    ("tried to pop from sibling to redistribute but failed")
                                        .to_string(),
                                )));
                            }
                        } else {
                            // sibling is to the left, take its largest entry,
                            // and move it to the front of this node
                            if let Some(res) = sibling.pop_last() {
                                leaf.push_front(res.0, res.1, res.2);
                                // update the separator key
                                if let Some(new_sep_key) = leaf.keys.get(0) {
                                    parent.keys[parent_idx_of_leaf] = new_sep_key.clone();
                                } else {
                                    error!("failed to get leaf.keys[0]")
                                }
                            } else {
                                error!("tried to pop from sibling to redistribute but failed");
                                return Err(IndexError::Remove(RemoveError::GenericError(
                                    ("tried to pop from sibling to redistribute but failed")
                                        .to_string(),
                                )));
                            }
                        }

                        // make sure the siblings are encoded/updated
                        sibling.encode(sibling_page.data_mut(), self.key_len)?;
                        leaf.encode(current.data_mut(), self.key_len)?;
                        parent.encode(parent_page.data_mut(), self.key_len)?;
                        return Ok(());
                    }
                }
                Err(e) => {
                    return Err(IndexError::from(e));
                }
            }
        } else {
            unreachable!("parent node must always have at least 2 children");
        }

        // if we reach here it means we performed a merge, so we need to traverse up the stack
        // at the beginning of each iteration, 'child' is the node that just lost the key
        // in the first iteration, it lost a key since we merged in the logic above
        // we traverse up the stack only while merging, so at n > 1 iterations, we have also dropped a key from parent.
        let mut merged;
        loop {
            let mut child_page = ctx.write_set.pop_back().unwrap();
            let mut child = InternalPage::decode(child_page.data_mut(), self.key_len)?;
            if !child.is_underfull() {
                return Ok(());
            }

            // parent is underfull here
            // check if we are at the root
            if ctx.write_set.is_empty() {
                if child.children.len() == 1 {
                    // root.child becomes the root
                    if let Some(c) = child.get_child(0) {
                        header.root_page_id = *c;
                        header.encode(ctx.header.as_mut().unwrap().data_mut())?;
                    }
                }
                return Ok(());
            }
            let mut parent_page = ctx.write_set.pop_back().unwrap();
            let mut parent = InternalPage::decode(parent_page.data_mut(), self.key_len)?;

            // again, go through the same process of either merging or redistributing.
            // this probably could have been combined with the above logic since there's
            // a lot of duplicated logic but it's probably easier to read this way
            let child_idx = parent.find_child_index(&key);
            let sibling_idx = if parent_idx_of_leaf == 0 {
                1
            } else {
                parent_idx_of_leaf - 1
            };
            let sibling_is_right = sibling_idx > child_idx;
            let sibling_page_id = parent.get_child(sibling_idx).unwrap();
            let mut sibling_page = self.bpm.write_page(*sibling_page_id)?;
            let mut sibling = InternalPage::decode(sibling_page.data_mut(), self.key_len)?;

            if sibling.would_be_underfull() {
                // merge
                if sibling_is_right {
                    let sep = parent.keys[child_idx].clone();
                    child.absorb_right(sep, &mut sibling);
                    parent.remove_separator(child_idx);
                } else {
                    let sep = parent.keys[child_idx - 1].clone();
                    sibling.absorb_right(sep, &mut child);
                    parent.remove_separator(child_idx - 1);
                }
                merged = true;
            } else {
                // redistribute
                if sibling_is_right {
                    let (rotated_key, rotated_child) = sibling.pop_first_child_and_key();
                    let old_sep = parent.replace_key(child_idx, rotated_key);
                    child.append_key_and_child(old_sep, rotated_child);
                } else {
                    let (rotated_key, rotated_child) = sibling.pop_last_child_and_key();
                    let old_sep = parent.replace_key(child_idx - 1, rotated_key);
                    child.prepend_child_and_key(old_sep, rotated_child);
                }
                merged = false;
            }
            child.encode(child_page.data_mut(), self.key_len)?;
            sibling.encode(sibling_page.data_mut(), self.key_len)?;
            parent.encode(parent_page.data_mut(), self.key_len)?;
            if !merged {
                return Ok(());
            }
        }
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
        use std::collections::HashSet;

        let header = self.read_header()?;
        fn fmt_pid(pid: PageId) -> String {
            if pid == INVALID_PAGE_ID {
                "INVALID".to_string()
            } else {
                pid.to_string()
            }
        }
        fn fmt_children(children: &Vec<PageId>) -> String {
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
            if *nodes_printed >= 10_000 {
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

            let page = tree.bpm.read_page(page_id).map_err(IndexError::BpmError)?;
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
                            let mut children = Vec::with_capacity(internal.children.len());
                            for i in 0..internal.children.len() {
                                children
                                    .push(internal.get_child(i).ok_or(IndexError::Lookup)?.clone());
                            }
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

        // Print the tree starting from the root.
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
            if idx >= 10_000 {
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
                "tid={:?} search(): header root_page_id is invalid (empty tree); header={:?}",
                std::thread::current().id(),
                header
            );
            return Ok(None);
        }
        Ok(Some(
            self.bpm
                .read_page(header.root_page_id)
                .map_err(IndexError::BpmError)?,
        ))
    }

    /// find_leaf looks for the leaf page that would contain `key`, if `key` exists in our tree.
    /// This is useful for index scans, where we use a key as a lower bound. The key does not need
    /// to exist to be used as a bound.
    fn find_leaf(&self, key: &IndexKey) -> Result<LeafPage, IndexError> {
        let Some(root_page) = self.get_root_page_from_header()? else {
            return Err(IndexError::Lookup);
        };

        let mut current = root_page;
        loop {
            match Page::decode(current.data(), self.key_len)? {
                Page::Internal(node) => {
                    let next_page_id = node.children[node.find_child_index(key)];

                    // acquire a read latch on the next page,
                    // then drop the latch on the current page by reassigning
                    let next_page = self
                        .bpm
                        .read_page(next_page_id)
                        .map_err(IndexError::BpmError)?;
                    current = next_page;
                    // current.page_id()?;
                }
                Page::Leaf(leaf) => {
                    // we've arrived at a leaf!
                    return Ok(leaf);
                }
            }
        }
    }
}

impl Index for BPlusTree {
    fn insert(&self, key: IndexKey, val: IndexValue) -> InsertResult {
        info!(
            "b_plus_tree inserting key with len {} and value {:?}: ",
            key.0.len(),
            val
        );
        if key.0.len() != self.key_len as usize {
            return Err(IndexError::Insert(InsertError::InvalidKey));
        }

        let header = self.read_header()?;
        if header.root_page_id == INVALID_PAGE_ID {
            // Empty tree: we must pessimistically insert
            // drop the header, otherwise insert_pessimistic would deadlock
            // when trying to acquire the write latch on the header
            drop(header);
            return self.insert_pessimistic(key, val);
        }

        let Some(root_page) = self.get_root_page_from_header()? else {
            return Err(IndexError::Insert(InsertError::BadRoot));
        };
        let mut current = root_page;
        let leaf_page: Option<PageMut<'_>>;

        // look for the right leaf to insert at
        // if we have to split the leaf, we fall back to pessimistic insert
        loop {
            match Page::decode(current.data(), self.key_len)? {
                Page::Internal(node) => {
                    let next_page_id = node.children[node.find_child_index(&key)];

                    // acquire a read latch on the next page,
                    // then drop the latch on the current page by reassigning
                    let next_page = self
                        .bpm
                        .read_page(next_page_id)
                        .map_err(IndexError::BpmError)?;

                    let next_page_kind = Page::kind(next_page.data())?;
                    if matches!(next_page_kind, PageKind::Leaf) {
                        // if the next page is a leaf page, we need to acquire a write latch on it
                        // should be fine to drop the read latch on the next page, since we still hold
                        // the latch on the current page, which measn that the leaf cannot split even
                        // during the window that we drop the read latch on it.
                        drop(next_page);
                        leaf_page = Some(self.bpm.write_page(next_page_id)?);
                        break;
                    }

                    current = next_page;
                }
                Page::Leaf(_) => {
                    if current.page_id() == Some(header.root_page_id) {
                        drop(header);
                        drop(current);
                        return self.insert_pessimistic(key, val);
                    }
                    // the only time we should enter this arm is if the root itself is a leaf.
                    // we shouldn't get here, since we should have already exited the loop
                    // when we found the candidate leaf while looking ahead in the arm above.
                    error!("unexpectedly decoded leaf page during insert");
                    return Err(IndexError::Insert(InsertError::GenericError(
                        "unexpectedly decoded leaf page instead of internal page during insert"
                            .to_string(),
                    )));
                }
            }
        }
        let mut lp = leaf_page.unwrap();
        let mut leaf = LeafPage::decode(lp.data_mut(), self.key_len)?;
        if leaf.is_full() {
            // if the leaf is full we will have to split after inserting,
            // so fall back to pessimistic insert.
            // But if we know the parent is not full, we maybe don't need to fall back?
            // Probably not worth the trouble right now but could be done for performance.
            // TODO: can we avoid manually dropping?
            drop(lp);
            drop(header);
            drop(current);
            return self.insert_pessimistic(key, val);
        }

        // leaf is not full- simple case
        let record_id = match val {
            IndexValue::IndexValue(rid) => rid,
            _ => return Err(IndexError::Insert(InsertError::InvalidValue)),
        };

        leaf.insert(key, record_id)
            .map_err(|_| IndexError::Insert(InsertError::DuplicateKey))?;
        leaf.encode(lp.data_mut(), self.key_len)?;
        Ok(())
    }

    fn remove(&self, key: IndexKey) -> RemoveResult {
        info!("b_plus_tree removing key {:?}", key);
        if key.0.len() != self.key_len as usize {
            return Err(IndexError::Insert(InsertError::InvalidKey));
        }

        let header = self.read_header()?;
        if header.root_page_id == INVALID_PAGE_ID {
            return Err(IndexError::Remove(RemoveError::KeyNotFound));
        }

        let Some(root_page) = self.get_root_page_from_header()? else {
            return Err(IndexError::Insert(InsertError::BadRoot));
        };
        let mut current = root_page;
        let leaf_page: Option<PageMut<'_>>;

        loop {
            match Page::decode(current.data(), self.key_len)? {
                Page::Internal(node) => {
                    let next_page_index = node.find_child_index(&key);
                    let next_page_id = node.children[next_page_index];

                    // acquire a read latch on the next page,
                    // then drop the latch on the current page by reassigning
                    let next_page = self
                        .bpm
                        .read_page(next_page_id)
                        .map_err(IndexError::BpmError)?;

                    let next_page_kind = Page::kind(next_page.data())?;
                    if matches!(next_page_kind, PageKind::Leaf) {
                        // if the next page is a leaf page, we need to acquire a write latch on it
                        // should be fine to drop the read latch on the next page, since we still hold
                        // the latch on the current page
                        drop(next_page);
                        leaf_page = Some(self.bpm.write_page(next_page_id)?);
                        break;
                    }
                    current = next_page;
                }

                Page::Leaf(_) => {
                    if current.page_id() == Some(header.root_page_id) {
                        drop(header);
                        drop(current);
                        return self.remove_pessimistic(key);
                    }
                    // the only time we should enter this arm is if the root itself is a leaf.
                    // we shouldn't get here, since we should have already exited the loop
                    // when we found the candidate leaf while looking ahead in the arm above.
                    error!("unexpectedly decoded leaf page during insert");
                    return Err(IndexError::Remove(RemoveError::GenericError(
                        "unexpectedly decoded leaf page instead of internal page during remove"
                            .to_string(),
                    )));
                }
            }
        }

        let mut lp = leaf_page.unwrap();
        let mut leaf = LeafPage::decode(lp.data_mut(), self.key_len)?;

        // check if the leaf would be below the minimum size if we delete a key from it
        if leaf.would_be_underfull() {
            if !leaf.key_exists(&key) {
                return Err(IndexError::Remove(RemoveError::KeyNotFound));
            }
            // fall back to pessimistic removal.
            drop(lp);
            drop(header);
            drop(current);
            return self.remove_pessimistic(key);
        }

        leaf.remove_key(&key).map_err(IndexError::Remove)?;
        leaf.encode(lp.data_mut(), self.key_len)?;
        Ok(())
    }

    fn search(&self, key: &IndexKey) -> Result<Option<IndexValue>, IndexError> {
        debug!(
            "tid={:?} search(): searching for {:?}",
            std::thread::current().id(),
            key
        );
        if key.0.len() != self.key_len as usize {
            return Err(IndexError::Insert(InsertError::InvalidKey));
        }

        let leaf = self.find_leaf(key)?;

        // we've arrived at a leaf, look for the key
        let record_id = leaf.get(key).map(IndexValue::IndexValue);
        return Ok(record_id);
    }

    fn scan(&self, range: (Bound<IndexKey>, Bound<IndexKey>)) -> Result<IndexIter, IndexError> {
        // figure out which leaf page to start at, and which slot in the leaf page to start at
        let (start_leaf, start_slot) = match range.start_bound() {
            // start at the beginning
            Bound::Unbounded => {
                let header = self.read_header()?;
                let page_guard = self.bpm.read_page(header.first_leaf_id)?;

                let start_leaf = LeafPage::decode(page_guard.data(), self.key_len)?;
                (start_leaf, 0)
            }
            Bound::Excluded(key) => {
                let leaf = self.find_leaf(key)?;
                let start_slot = leaf.upper_bound(key);
                (leaf, start_slot)
            }
            Bound::Included(key) => {
                let leaf = self.find_leaf(key)?;
                let start_slot = leaf.lower_bound(key);
                (leaf, start_slot)
            }
        };

        let leaf_iter = LeafPagesIter {
            bpm: self.bpm.clone(),
            next_leaf_id: start_leaf.next_leaf_id,
            seen: HashSet::new(),
            key_len: self.key_len,
            finished: false,
        };

        let da_iter = BPlusTreeIter {
            leaves_iter: leaf_iter,
            current_leaf: Some(start_leaf),
            slot_idx: start_slot,
            upper_bound: range.end_bound().cloned(),
        };
        Ok(Box::new(da_iter))
    }
}

pub struct BPlusTreeIter {
    leaves_iter: LeafPagesIter,
    current_leaf: Option<LeafPage>,
    slot_idx: usize,
    upper_bound: Bound<IndexKey>,
}

impl Iterator for BPlusTreeIter {
    type Item = Result<(IndexKey, IndexValue), IndexError>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let leaf = self.current_leaf.as_ref()?;

            if self.slot_idx >= leaf.keys.len() {
                match self.leaves_iter.next() {
                    None => {
                        self.current_leaf = None;
                        return None;
                    }
                    Some(Err(e)) => {
                        self.current_leaf = None;
                        return Some(Err(e));
                    }
                    Some(Ok((_, next_leaf))) => {
                        self.current_leaf = Some(next_leaf);
                        self.slot_idx = 0;
                        continue; // re-check num_slots in case the leaf is empty
                    }
                }
            }

            let i = self.slot_idx;
            self.slot_idx += 1;

            if leaf.is_tombstoned(i as u32) {
                continue;
            }

            let rid = leaf.record_ids[i];
            let val = IndexValue::IndexValue(rid);

            let past_upper = match &self.upper_bound {
                Bound::Included(u) => &leaf.keys[i] > u,
                Bound::Excluded(u) => &leaf.keys[i] >= u,
                Bound::Unbounded => false,
            };
            if past_upper {
                self.current_leaf = None;
                return None;
            }

            // fuck this clone
            return Some(Ok((leaf.keys[i].clone(), val)));
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
        let key_len = 2000u32; // leaf_max_size=4 => multiple leaves
        let tree = BPlusTree::new("idx".to_string(), key_len, bpm.clone()).unwrap();

        // Empty tree: iterator yields nothing.
        assert_eq!(tree.leaf_pages().unwrap().count(), 0);

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
        let key_len = 2000u32;
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
        let key_len = 2000u32;
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
        let key_len = 2000u32; // leaf_max_size=4 => triggers a split at 5 inserts
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

            let left_leaf_id = root.get_child(0).unwrap().clone();
            let right_leaf_id = root.get_child(1).unwrap().clone();
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
    fn search_finds_inserted_keys_and_returns_none_for_missing() {
        let (bpm, _tmp) = make_bpm(10);
        let key_len = 2000u32; // leaf_max_size=4 => triggers a split at 5 inserts
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
            let found = tree.search(&key).unwrap().expect("key present");
            match found {
                IndexValue::IndexValue(rid) => {
                    assert_eq!(rid.page_id, i as u32);
                    assert_eq!(rid.slot_id, i as u32 + 10);
                }
                _ => panic!("unexpected IndexValue variant"),
            }
        }

        let missing = make_key(9, key_len as usize);
        assert!(tree.search(&missing).unwrap().is_none());
    }
}

// TODO: implement table heap pages
// TablePage stores:
//      -
//
// Full testing flow looks something like:
//      1. Create new Schema
//      2. Create mock Tuple's
//      3. Create IndexDefinition from provided schema and provided index cols
//      3 (cont). This should also create the IndexEncoder
//      4. Iterate through tuples and create idx keys using the encoder
//      4 (cont). encoder.encode(&tuple.data) -> [u8]
//      5. Create new b+ tree and insert each index key into it
//

// TODO:
// implement traversal + insertion of a page
//
// 1. get page structure and invariants down for internal pages
// 2a. get
// 2b. for leaf pages, we can store mock data for now
// 3. get traversal/search down... i think we need to get insertion down first
// insert(key, value)
//
// for inserts/removals:
// we start from the root of the tree
// we keep track of any pages we visit in a stack
// when inserting or removing from the tree,
// lock the header page (which is stored separate, not part of the tree itself)
// we do this bc the root page id is stored in the header
