# MVCC in cmudb

cmudb runs every statement inside a transaction under **snapshot isolation**. The design
follows CMU BusTub's MVCC project (Fall 2023/2024), with a few changes where cmudb's shape
allows something simpler. Code: `src/concurrency/`.

## Timestamps

- `TransactionManager` keeps `last_commit_ts`. `begin()` reads it as the transaction's
  `read_ts` and registers the transaction in the running set (under the same lock the
  watermark is computed under).
- Transaction ids start at `TXN_START_ID = 2^62`. A tuple stamped with a value at or above
  that is an uncommitted write; it sorts after every snapshot, so other transactions never
  see it.
- `commit()` takes the commit latch, picks `last_commit_ts + 1`, stamps every slot in the
  write set, and only then publishes the new `last_commit_ts`. A read-only commit doesn't
  consume a timestamp.

## Where versions live

- **Newest version: in the heap.** Each slot now carries `TupleMeta { ts, is_deleted }`
  (slot entry grew from 9 to 17 bytes, so the on-disk table page layout changed).
- **Older versions: undo chains.** `TransactionManager` maps each record id to the head of
  a newest-first chain of `UndoLog { ts, is_deleted, data, prev }`. Every entry is a full
  image of an older version.
- **Updates are in place while the row fits its page.** A row keeps its record id as long as
  its new image fits on its page. Each slot records the capacity of its byte region, and the
  capacity never shrinks, so writing an older image back on abort always fits. A new image
  longer than the capacity moves within its page. Tables with a `TEXT` or `VARCHAR` column
  leave `TEXT_UPDATE_RESERVE` bytes free on each page for rows to grow into.
- **A row that outgrows its page moves.** When even a compacted page has no room for the
  new image (`UpdateDoesNotFit`), `mvcc::update_tuple` turns the update into a delete of
  the old record id plus an insert of the new image at a new record id, filed in every
  index under the new record id, all under the indexes' unique latches. Older snapshots
  keep reading the old record id and its undo chain and see nothing at the new one; abort
  restores the old slot and kills the new one, as for any delete and insert. Uniqueness is
  checked only for keys the update changes. `UpdateExecutor` skips record ids it moved
  rows to, so a scan that reaches a moved row again does not update it twice.

## Visibility

A transaction sees a version stamped `ts` if `ts == txn.id` (its own write) or
`ts <= read_ts`. If the heap version isn't visible it walks the chain and takes the first
entry with `ts <= read_ts`; deleted means invisible. The heap slot and the chain head are
read under the same page read latch, so they are a consistent pair.

## Writes and conflicts

Update/delete run inside `TableHeap::write_with`, under the page write latch:

1. If the newest version belongs to another transaction (a txn id) or was committed after
   our snapshot (`ts > read_ts`), it's a write-write conflict: the transaction is
   **tainted** and the statement fails. There is no waiting; the caller must abort.
2. Overwrite the slot, stamped with our txn id.
3. If that was the first write of the tuple by this transaction, push the version it
   replaced onto the chain, still under the same latch. A write that fails pushes nothing.

Inserts append a slot stamped with the txn id. Any failed write (conflict, duplicate key,
storage error) taints the transaction, since there are no statement-level savepoints.

## Abort

For each written slot: if we pushed an undo entry, restore the slot from the chain head
and pop it; if we inserted the tuple, mark it deleted at timestamp 0 (invisible to all).
Both happen under the page write latch.

## Failure handling

- Undo chains change only after the slot write succeeds, under the same latch: a write
  pushes its undo entry afterwards, and abort pops the head after restoring it. A failed
  write leaves the slot and its chain in step, so a retried abort restores the right image.
- If commit can't stamp a slot (a buffer pool error), it re-stamps the slots it already did
  back to the transaction id, taints the transaction and keeps its write set. Nothing was
  published, so an `abort` then rolls everything back.
- `abort` forgets each write as it rolls it back. If one fails, the rest stay in the write
  set and the transaction stays registered (its writes are still in the heap), so `abort`
  can be called again.
- Every transaction must be committed or aborted. One that is dropped while running keeps
  its snapshot in the running set and pins the watermark, so GC stops reclaiming.
- In the shell, a failed statement aborts its transaction immediately. Inside `BEGIN`, later
  statements are refused until `ROLLBACK` (or `COMMIT`, which rolls back), as in Postgres. A
  GC failure after a commit is only logged; it never turns a commit into an error.

## Garbage collection

`watermark = min(read_ts of running txns)`, or `last_commit_ts` when idle. A reader stops
at the first version `<= read_ts`, so for each chain: if the heap version is committed at
or below the watermark the whole chain is dropped; otherwise everything after the first
entry `<= watermark` is unlinked. Entries are `Arc`s, so a reader already walking a chain
keeps what it holds. The REPL runs GC after every commit/abort; tests call it directly.

## Indexes

- Index entries are never removed when a tuple changes: older snapshots may still need
  them. So one user key can map to several record ids (a deleted row and its replacement,
  a row that moved to a new key). B+ tree keys are therefore `user key ++ record id`, which
  keeps every entry unique; lookups scan the `[key ++ 0, key ++ max]` range.
- Readers re-check that the version they see still has the key they looked up.
- Uniqueness of user keys is enforced by the MVCC write path: under the index's
  `unique_latch`, look at the *newest* version behind each entry for the key. A live one is
  a duplicate; an uncommitted or post-snapshot one is a write-write conflict.
- An update that changes an indexed column adds an entry for the new key (same record id).

## Isolation level

Snapshot isolation. Dirty reads, non-repeatable reads, phantoms and lost updates are
prevented; write skew is allowed (there's a test documenting it).

## What was kept from BusTub, and what changed

Kept:
- The timestamp scheme: commit-ts counter, txn ids from 2^62 as temporary timestamps.
- Newest version in the heap, older versions in per-RID undo chains.
- No waiting on conflict: taint and abort (first writer wins).
- Atomic check-and-update under the page latch (BusTub's `UpdateTupleInPlace` with a check
  function / `UpdateTupleAndUndoLink`), and reading tuple + undo link together.
- Watermark-based GC.

Changed:
- **Full-image undo logs instead of partial (modified-fields) deltas.** cmudb rows are
  small, so the memory saving is small, and full images make
  reconstruction a lookup rather than a merge of partial columns.
- **Chains are `Arc` links owned by the version map**, not `(txn_id, log_idx)` pointers into
  each transaction's undo buffer. BusTub has to keep finished transactions in `txn_map_`
  until GC proves none of their undo logs are reachable; here a transaction is dropped as
  soon as it finishes, and GC just unlinks entries.
- **Multiple unique indexes, and key-changing updates, work under MVCC.** BusTub supports a
  single primary-key index and reuses the deleted tuple's RID on re-insert. Appending the
  RID to index keys removes that restriction.
- **Updates are in place unless the row outgrows its page.** BusTub's update-the-primary-key
  path turns into delete + insert; with RID-suffixed keys that isn't needed. A row moves
  (delete + insert) only when its new image no longer fits on its page.

## Not done yet

- Serializable isolation (BusTub's commit-time predicate verification).
- Removing dead index entries and reclaiming deleted heap slots during GC. Until then,
  every key-changing update, delete and aborted insert leaves a B+ tree entry behind, and
  the uniqueness check scans all entries for a key, so heavy churn on one key gets slower.
- Recovery of versions left uncommitted on disk. `Database::open` resumes `last_commit_ts`
  and the transaction id counter from the data file, and closing a database rolls back the
  block `BEGIN` left open, so committed rows are visible after a reopen and rolled-back ones
  are not. Undo chains are not stored: after a reopen nothing is running, so only the newest
  version of each row matters. A transaction still running on the transaction manager when the
  database is written (or a process killed mid-commit) leaves its versions stamped with its
  id; no new transaction gets that id, so they stay invisible, but their rows stay locked
  against writers. Undoing them needs a WAL.
- `CREATE`/`DROP` are not transactional. `CREATE INDEX` fills the new index from the
  newest version of each row, outside any transaction. If another transaction has an
  uncommitted update or delete open at that moment, the index can miss the row once that
  transaction rolls back, and older snapshots can't find old keys through it. Run
  `CREATE INDEX` while no other transaction is open.
