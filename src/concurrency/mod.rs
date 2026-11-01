//! Multi-version concurrency control.
//!
//! cmudb runs transactions under snapshot isolation, following the design of CMU BusTub's
//! MVCC project with a few deliberate changes (see `docs/mvcc.md`):
//!
//! - **Timestamps.** A global counter of commit timestamps. A transaction reads as of the
//!   newest commit timestamp at the moment it begins (its `read_ts`). Transaction ids start
//!   at [`TXN_START_ID`] (2^62), so a version stamped with an uncommitted transaction's id
//!   sorts after every snapshot.
//! - **Versions.** The newest version of a tuple lives in the table heap, together with its
//!   [`TupleMeta`](crate::table_page::TupleMeta) (timestamp + deleted flag). Older versions
//!   form an undo chain per record id, newest first, owned by the [`TransactionManager`].
//!   Each undo entry is a full image of an older version.
//! - **Writes** happen in place under the page write latch: check for a write-write
//!   conflict, push the old version onto the chain, overwrite the heap slot.
//! - **Commit** stamps every written slot with the commit timestamp and then publishes it.
//!   **Abort** restores every written slot from the head of its chain.
//! - **Garbage collection** drops undo entries that no running snapshot can reach.

pub mod mvcc;
pub mod transaction;
pub mod transaction_manager;

pub use transaction::{
    IsolationLevel, TXN_START_ID, Timestamp, Transaction, TransactionState, TxnId, is_txn_id,
};
pub use transaction_manager::{GcStats, TransactionManager, TxnError, UndoLog};
