//! Snapshot isolation behaviour of the MVCC layer, driven through SQL.

use cmudb::catalog::{Catalog, Value};
use cmudb::concurrency::{IsolationLevel, Transaction, TransactionManager, TransactionState};
use cmudb::{Database, DmlOp, StatementResult};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

type Rows = Vec<Vec<Option<i32>>>;

struct Db {
    db: Database,
    bpm: Arc<cmudb::buffer_pool::BufferPoolManager>,
    mgr: Arc<TransactionManager>,
}

impl Db {
    fn new() -> Self {
        let db = Database::scratch(64).unwrap();
        Self {
            bpm: db.bpm().clone(),
            mgr: db.txn_mgr().clone(),
            db,
        }
    }

    fn catalog(&self) -> &Catalog {
        self.db.catalog()
    }

    fn begin(&self) -> Arc<Transaction> {
        self.mgr.begin(IsolationLevel::SnapshotIsolation)
    }

    /// Runs `sql` in `txn`. Returns the rows it produced (for INSERT, the count).
    fn exec(&self, txn: &Arc<Transaction>, sql: &str) -> Result<Rows, String> {
        let results = self.db.execute_in(txn, sql).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for result in results {
            match result {
                StatementResult::Count {
                    op: DmlOp::Insert,
                    rows,
                } => out.push(vec![Some(i32::try_from(rows).unwrap())]),
                StatementResult::Rows { rows, .. } => out.extend(rows.into_iter().map(|row| {
                    row.into_iter()
                        .map(|v| match v {
                            Some(Value::INT(v)) => Some(v),
                            Some(Value::BOOLEAN(b)) => Some(b as i32),
                            None => None,
                            Some(other) => panic!("unexpected value {other:?}"),
                        })
                        .collect()
                })),
                StatementResult::Count { .. }
                | StatementResult::Ddl(_)
                | StatementResult::Txn(_) => {}
            }
        }
        Ok(out)
    }

    /// Runs `sql` in a transaction of its own and commits it.
    fn auto(&self, sql: &str) -> Rows {
        let txn = self.begin();
        let rows = self
            .exec(&txn, sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
        self.mgr.commit(&txn).unwrap();
        rows
    }

    fn query(&self, txn: &Arc<Transaction>, sql: &str) -> Rows {
        let mut rows = self.exec(txn, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        rows.sort();
        rows
    }

    fn commit(&self, txn: &Transaction) {
        self.mgr.commit(txn).unwrap();
    }

    fn abort(&self, txn: &Transaction) {
        self.mgr.abort(txn).unwrap();
    }
}

fn rows(v: &[&[i32]]) -> Rows {
    v.iter()
        .map(|r| r.iter().map(|x| Some(*x)).collect())
        .collect()
}

/// `accounts(id, balance)` with a unique index on `id`.
fn accounts(db: &Db, initial: &[(i32, i32)]) {
    db.auto("CREATE TABLE accounts (id INTEGER, balance INTEGER)");
    db.auto("CREATE INDEX accounts_id ON accounts (id)");
    let values: Vec<String> = initial.iter().map(|(i, b)| format!("({i}, {b})")).collect();
    db.auto(&format!(
        "INSERT INTO accounts VALUES {}",
        values.join(", ")
    ));
}

const ALL: &str = "SELECT * FROM accounts";

#[test]
fn snapshot_ignores_commits_after_it_began() {
    let db = Db::new();
    accounts(&db, &[(1, 100), (2, 200)]);

    let old = db.begin();
    db.auto("UPDATE accounts SET balance = 150 WHERE id = 1");
    db.auto("DELETE FROM accounts WHERE id = 2");
    db.auto("INSERT INTO accounts VALUES (3, 300)");

    assert_eq!(
        db.query(&old, ALL),
        rows(&[&[1, 100], &[2, 200]]),
        "no non-repeatable reads and no phantoms: `old` keeps reading its snapshot"
    );
    assert_eq!(
        db.query(&old, "SELECT * FROM accounts WHERE id = 2"),
        rows(&[&[2, 200]])
    );
    assert!(
        db.query(&old, "SELECT * FROM accounts WHERE id = 3")
            .is_empty()
    );
    db.commit(&old);

    let new = db.begin();
    assert_eq!(db.query(&new, ALL), rows(&[&[1, 150], &[3, 300]]));
    db.commit(&new);
}

#[test]
fn uncommitted_writes_are_visible_only_to_their_transaction() {
    let db = Db::new();
    accounts(&db, &[(1, 100), (2, 200)]);

    let writer = db.begin();
    db.exec(&writer, "UPDATE accounts SET balance = 0 WHERE id = 1")
        .unwrap();
    db.exec(&writer, "DELETE FROM accounts WHERE id = 2")
        .unwrap();
    db.exec(&writer, "INSERT INTO accounts VALUES (3, 300)")
        .unwrap();
    assert_eq!(
        db.query(&writer, ALL),
        rows(&[&[1, 0], &[3, 300]]),
        "a transaction reads its own writes through a scan"
    );
    assert_eq!(
        db.query(&writer, "SELECT * FROM accounts WHERE id = 3"),
        rows(&[&[3, 300]]),
        "a transaction reads its own writes through the index"
    );

    let reader = db.begin();
    assert_eq!(
        db.query(&reader, ALL),
        rows(&[&[1, 100], &[2, 200]]),
        "no dirty reads, even for a transaction that begins after the writes"
    );
    assert!(
        db.query(&reader, "SELECT * FROM accounts WHERE id = 3")
            .is_empty()
    );

    db.commit(&writer);
    assert_eq!(
        db.query(&reader, ALL),
        rows(&[&[1, 100], &[2, 200]]),
        "the reader keeps its snapshot after the writer commits"
    );
    db.commit(&reader);
}

#[test]
fn second_writer_of_a_row_is_aborted() {
    let db = Db::new();
    accounts(&db, &[(1, 100)]);

    let t1 = db.begin();
    let t2 = db.begin();
    db.exec(&t1, "UPDATE accounts SET balance = 1 WHERE id = 1")
        .unwrap();
    let err = db
        .exec(&t2, "UPDATE accounts SET balance = 2 WHERE id = 1")
        .unwrap_err();
    assert!(
        err.contains("write-write conflict"),
        "the row holds t1's uncommitted write, so t2 fails right away instead of waiting: {err}"
    );
    assert_eq!(t2.state(), TransactionState::Tainted);
    assert!(db.mgr.commit(&t2).is_err(), "a tainted txn cannot commit");
    db.abort(&t2);

    db.commit(&t1);
    let t3 = db.begin();
    assert_eq!(db.query(&t3, ALL), rows(&[&[1, 1]]));
    db.commit(&t3);
}

#[test]
fn lost_update_is_prevented() {
    let db = Db::new();
    accounts(&db, &[(1, 100)]);

    let t1 = db.begin();
    let t2 = db.begin();
    assert_eq!(db.query(&t1, ALL), rows(&[&[1, 100]]));
    assert_eq!(db.query(&t2, ALL), rows(&[&[1, 100]]));
    db.exec(&t1, "UPDATE accounts SET balance = balance + 10")
        .unwrap();
    db.commit(&t1);
    let err = db
        .exec(&t2, "UPDATE accounts SET balance = balance + 20")
        .unwrap_err();
    assert!(
        err.contains("write-write conflict"),
        "t1 committed after t2's snapshot, so t2's update would overwrite it blindly: {err}"
    );
    db.abort(&t2);

    let t3 = db.begin();
    assert_eq!(db.query(&t3, ALL), rows(&[&[1, 110]]));
    db.commit(&t3);
}

/// Snapshot isolation is not serializable. Two doctors are on call and each may go off call
/// if the other is still on. Both check, both see two on call, both leave: the writes touch
/// different rows, so neither conflicts. This documents the anomaly.
#[test]
fn write_skew_is_allowed_under_snapshot_isolation() {
    let db = Db::new();
    db.auto("CREATE TABLE doctors (id INTEGER, on_call INTEGER)");
    db.auto("INSERT INTO doctors VALUES (1, 1), (2, 1)");

    let alice = db.begin();
    let bob = db.begin();
    let on_call = "SELECT * FROM doctors WHERE on_call = 1";
    assert_eq!(db.query(&alice, on_call).len(), 2);
    assert_eq!(db.query(&bob, on_call).len(), 2);
    db.exec(&alice, "UPDATE doctors SET on_call = 0 WHERE id = 1")
        .unwrap();
    db.exec(&bob, "UPDATE doctors SET on_call = 0 WHERE id = 2")
        .unwrap();
    db.commit(&alice);
    db.commit(&bob);

    let check = db.begin();
    assert!(db.query(&check, on_call).is_empty());
    db.commit(&check);
}

#[test]
fn abort_restores_every_kind_of_write() {
    let db = Db::new();
    accounts(&db, &[(1, 100), (2, 200), (3, 300)]);

    let t = db.begin();
    db.exec(&t, "UPDATE accounts SET balance = balance + 1 WHERE id = 1")
        .unwrap();
    db.exec(&t, "UPDATE accounts SET balance = balance + 1 WHERE id = 1")
        .unwrap();
    db.exec(&t, "DELETE FROM accounts WHERE id = 1").unwrap();
    db.exec(&t, "UPDATE accounts SET id = 20 WHERE id = 2")
        .unwrap();
    db.exec(&t, "DELETE FROM accounts WHERE id = 3").unwrap();
    db.exec(&t, "INSERT INTO accounts VALUES (3, 333), (4, 400)")
        .unwrap();
    assert_eq!(db.query(&t, ALL), rows(&[&[3, 333], &[4, 400], &[20, 200]]));
    db.abort(&t);
    assert_eq!(t.state(), TransactionState::Aborted);
    assert_eq!(
        db.mgr.num_version_chains(),
        0,
        "abort pops the undo entries it pushed"
    );

    let after = db.begin();
    assert_eq!(
        db.query(&after, ALL),
        rows(&[&[1, 100], &[2, 200], &[3, 300]])
    );
    for (id, balance) in [(1, 100), (2, 200), (3, 300)] {
        assert_eq!(
            db.query(&after, &format!("SELECT * FROM accounts WHERE id = {id}")),
            rows(&[&[id, balance]]),
            "the index agrees with the heap"
        );
    }
    for gone in [4, 20] {
        assert!(
            db.query(&after, &format!("SELECT * FROM accounts WHERE id = {gone}"))
                .is_empty()
        );
    }
    db.commit(&after);
    let keys_claimed_by_the_aborted_txn = "INSERT INTO accounts VALUES (4, 4), (20, 20)";
    db.auto(keys_claimed_by_the_aborted_txn);
    let t = db.begin();
    assert_eq!(
        db.query(&t, "SELECT * FROM accounts WHERE id = 20"),
        rows(&[&[20, 20]])
    );
    db.commit(&t);
}

#[test]
fn failed_statement_taints_and_rolls_back_cleanly() {
    let db = Db::new();
    accounts(&db, &[(1, 100), (2, 200)]);

    let t = db.begin();
    db.exec(&t, "UPDATE accounts SET balance = 0 WHERE id = 1")
        .unwrap();
    let second_row_collides_halfway = "INSERT INTO accounts VALUES (5, 5), (2, 2)";
    let err = db.exec(&t, second_row_collides_halfway).unwrap_err();
    assert!(err.contains("duplicate key"), "{err}");
    assert_eq!(t.state(), TransactionState::Tainted);
    db.abort(&t);

    let after = db.begin();
    assert_eq!(db.query(&after, ALL), rows(&[&[1, 100], &[2, 200]]));
    db.commit(&after);
}

#[test]
fn unique_index_across_versions_and_transactions() {
    let db = Db::new();
    accounts(&db, &[(1, 100)]);

    let t = db.begin();
    assert!(
        db.exec(&t, "INSERT INTO accounts VALUES (1, 5)").is_err(),
        "a live committed key is taken"
    );
    db.abort(&t);

    let a = db.begin();
    let b = db.begin();
    db.exec(&a, "INSERT INTO accounts VALUES (7, 1)").unwrap();
    let err = db
        .exec(&b, "INSERT INTO accounts VALUES (7, 2)")
        .unwrap_err();
    assert!(
        err.contains("write-write conflict"),
        "of two transactions racing for the same new key, the second conflicts: {err}"
    );
    db.abort(&b);
    db.commit(&a);

    let before = db.begin();
    db.auto("DELETE FROM accounts WHERE id = 1");
    db.auto("INSERT INTO accounts VALUES (1, 111)");
    let after = db.begin();
    let by_key = "SELECT * FROM accounts WHERE id = 1";
    let two_rids_one_key = "after a delete and re-insert the index holds two record ids for \
                            the key, and each snapshot finds the right one";
    assert_eq!(
        db.query(&before, by_key),
        rows(&[&[1, 100]]),
        "{two_rids_one_key}"
    );
    assert_eq!(
        db.query(&after, by_key),
        rows(&[&[1, 111]]),
        "{two_rids_one_key}"
    );

    db.auto("UPDATE accounts SET id = 2 WHERE id = 1");
    db.auto("INSERT INTO accounts VALUES (1, 1)");
    let latest = db.begin();
    let moved = "moving a row to a new key frees the old one, while old snapshots still find \
                 it by the old key";
    assert_eq!(db.query(&latest, by_key), rows(&[&[1, 1]]), "{moved}");
    assert_eq!(
        db.query(&latest, "SELECT * FROM accounts WHERE id = 2"),
        rows(&[&[2, 111]]),
        "{moved}"
    );
    assert_eq!(db.query(&before, by_key), rows(&[&[1, 100]]), "{moved}");
    assert_eq!(db.query(&after, by_key), rows(&[&[1, 111]]), "{moved}");
    for t in [before, after, latest] {
        db.commit(&t);
    }
}

/// An index entry left behind when a row moves to a new key only blocks that old key while
/// the move is uncommitted, because an abort would bring the key back.
#[test]
fn old_key_of_a_moved_row_is_free_once_the_move_commits() {
    let db = Db::new();
    accounts(&db, &[(5, 50)]);

    let mover = db.begin();
    db.exec(&mover, "UPDATE accounts SET id = 6 WHERE id = 5")
        .unwrap();
    let early = db.begin();
    let err = db
        .exec(&early, "INSERT INTO accounts VALUES (5, 1)")
        .unwrap_err();
    assert!(
        err.contains("write-write conflict"),
        "the move could still roll back and reclaim the key: {err}"
    );
    db.abort(&early);

    let late = db.begin();
    db.commit(&mover);
    db.exec(&late, "INSERT INTO accounts VALUES (5, 2)")
        .expect("a committed move leaves only a stale entry for the old key");
    db.commit(&late);

    let check = db.begin();
    assert_eq!(db.query(&check, ALL), rows(&[&[5, 2], &[6, 50]]));
    db.commit(&check);
}

#[test]
fn garbage_collection_keeps_what_running_snapshots_need() {
    let db = Db::new();
    accounts(&db, &[(1, 0), (2, 0)]);

    let old = db.begin();
    for i in 1..=5 {
        db.auto(&format!("UPDATE accounts SET balance = {i} WHERE id = 1"));
    }
    let mid = db.begin();
    for i in 6..=9 {
        db.auto(&format!("UPDATE accounts SET balance = {i} WHERE id = 1"));
    }
    db.auto("DELETE FROM accounts WHERE id = 2");
    let rid1 = {
        let table = db
            .catalog()
            .get_table(db.catalog().table_oid("accounts").unwrap())
            .unwrap();
        let first_slot_holds_id_1_and_updates_stay_in_place = table.heap.iter().next();
        first_slot_holds_id_1_and_updates_stay_in_place
            .unwrap()
            .unwrap()
            .1
    };
    assert_eq!(db.mgr.version_chain_len(rid1), 9);

    let stats = db.mgr.garbage_collect().unwrap();
    assert_eq!(stats.watermark, old.read_ts());
    assert_eq!(
        stats.undo_logs_removed, 0,
        "`old` still needs the version at its snapshot (balance 0), and nothing is older"
    );
    assert_eq!(db.query(&old, ALL), rows(&[&[1, 0], &[2, 0]]));
    db.commit(&old);

    let stats = db.mgr.garbage_collect().unwrap();
    assert_eq!(stats.watermark, mid.read_ts());
    assert_eq!(
        stats.undo_logs_removed, 5,
        "with `mid` the oldest reader, everything below balance 5 is unreachable"
    );
    assert_eq!(db.mgr.version_chain_len(rid1), 4);
    assert_eq!(db.query(&mid, ALL), rows(&[&[1, 5], &[2, 0]]));
    db.commit(&mid);

    let stats = db.mgr.garbage_collect().unwrap();
    assert_eq!(
        stats.chains_removed, 2,
        "nobody is reading old versions any more"
    );
    assert_eq!(db.mgr.num_version_chains(), 0);
    let t = db.begin();
    assert_eq!(db.query(&t, ALL), rows(&[&[1, 9]]));
    db.commit(&t);
}

#[test]
fn concurrent_increments_never_lose_an_update() {
    let db = Db::new();
    accounts(&db, &[(1, 0)]);
    let committed = AtomicUsize::new(0);
    let threads = 8;
    let attempts = 40;

    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                for _ in 0..attempts {
                    let t = db.begin();
                    match db.exec(&t, "UPDATE accounts SET balance = balance + 1 WHERE id = 1") {
                        Ok(_) => {
                            db.commit(&t);
                            committed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            assert!(e.contains("write-write conflict"), "{e}");
                            db.abort(&t);
                        }
                    }
                }
            });
        }
        s.spawn(|| {
            for _ in 0..20 {
                db.mgr.garbage_collect().unwrap();
                std::thread::yield_now();
            }
        });
    });

    let n = committed.load(Ordering::Relaxed) as i32;
    assert!(n > 0);
    let t = db.begin();
    assert_eq!(db.query(&t, ALL), rows(&[&[1, n]]));
    db.commit(&t);
}

#[test]
fn concurrent_transfers_preserve_the_total_in_every_snapshot() {
    const ACCOUNTS: i32 = 6;
    const TOTAL: i32 = ACCOUNTS * 100;
    let db = Db::new();
    let initial: Vec<(i32, i32)> = (0..ACCOUNTS).map(|i| (i, 100)).collect();
    accounts(&db, &initial);
    let transfers = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for w in 0..4 {
            let db = &db;
            let transfers = &transfers;
            s.spawn(move || {
                for i in 0..60 {
                    let from = (w + i) % ACCOUNTS;
                    let to = (w * 3 + i * 7 + 1) % ACCOUNTS;
                    if from == to {
                        continue;
                    }
                    let t = db.begin();
                    let ok = db
                        .exec(
                            &t,
                            &format!("UPDATE accounts SET balance = balance - 5 WHERE id = {from}"),
                        )
                        .and_then(|_| {
                            db.exec(
                                &t,
                                &format!(
                                    "UPDATE accounts SET balance = balance + 5 WHERE id = {to}"
                                ),
                            )
                        });
                    match ok {
                        Ok(_) => {
                            db.commit(&t);
                            transfers.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => db.abort(&t),
                    }
                }
            });
        }
        for _ in 0..2 {
            let db = &db;
            s.spawn(move || {
                for _ in 0..40 {
                    let t = db.begin();
                    let balances = db.query(&t, ALL);
                    assert_eq!(balances.len(), ACCOUNTS as usize);
                    let total: i32 = balances.iter().map(|r| r[1].unwrap()).sum();
                    assert_eq!(total, TOTAL, "a snapshot saw a half-done transfer");
                    db.commit(&t);
                    db.mgr.garbage_collect().unwrap();
                }
            });
        }
    });

    assert!(transfers.load(Ordering::Relaxed) > 0);
    let t = db.begin();
    let total: i32 = db.query(&t, ALL).iter().map(|r| r[1].unwrap()).sum();
    assert_eq!(total, TOTAL);
    db.commit(&t);
    db.mgr.garbage_collect().unwrap();
    assert_eq!(db.mgr.num_version_chains(), 0);
}

#[test]
fn null_to_value_update_stays_in_place() {
    let db = Db::new();
    db.auto("CREATE TABLE n (id INTEGER, v INTEGER)");
    db.auto("INSERT INTO n VALUES (1, NULL)");
    let old = db.begin();
    db.auto("UPDATE n SET v = 7 WHERE id = 1");
    let new = db.begin();
    assert_eq!(
        db.query(&new, "SELECT * FROM n"),
        vec![vec![Some(1), Some(7)]]
    );
    assert_eq!(db.query(&old, "SELECT * FROM n"), vec![vec![Some(1), None]]);
    db.commit(&old);
    db.commit(&new);
}

mod failure_paths {
    use super::*;
    use cmudb::concurrency::mvcc;
    use cmudb::table_heap::{TableHeap, Tuple};

    fn tuple(b: u8) -> Tuple {
        Tuple { data: vec![b; 16] }
    }

    fn read(
        mgr: &TransactionManager,
        heap: &TableHeap,
        rid: cmudb::table_heap::RecordId,
    ) -> Option<Vec<u8>> {
        let t = mgr.begin(IsolationLevel::SnapshotIsolation);
        let got = mvcc::read_visible(&t, mgr, heap, rid)
            .unwrap()
            .map(|t| t.data);
        mgr.commit(&t).unwrap();
        got
    }

    /// The pool has a single frame, so pinning another page leaves no room to latch the table
    /// page, and both commit and abort fail until the pin is dropped.
    #[test]
    fn failed_commit_keeps_the_write_set_so_abort_can_undo_it() {
        let (bpm, _dir) = cmudb::testing::bpm(1);
        let heap = TableHeap::new(bpm.clone()).unwrap();
        let mgr = TransactionManager::new(bpm.clone());

        let t0 = mgr.begin(IsolationLevel::SnapshotIsolation);
        let a = mvcc::insert_tuple(&t0, &heap, &[], &tuple(1)).unwrap();
        mgr.commit(&t0).unwrap();
        let committed = mgr.last_commit_ts();

        let t1 = mgr.begin(IsolationLevel::SnapshotIsolation);
        mvcc::update_tuple(&t1, &mgr, &heap, &[], a, &tuple(1), &tuple(2)).unwrap();
        let c = mvcc::insert_tuple(&t1, &heap, &[], &tuple(3)).unwrap();

        let pin = bpm.new_page().unwrap();
        assert!(mgr.commit(&t1).is_err());
        assert_eq!(t1.state(), TransactionState::Tainted);
        assert_eq!(mgr.last_commit_ts(), committed, "nothing was published");
        assert!(mgr.abort(&t1).is_err());
        assert_eq!(
            t1.write_set().len(),
            2,
            "abort can't reach the page either, so it keeps what it still has to undo"
        );
        assert_eq!(
            mgr.watermark(),
            t1.read_ts(),
            "its writes are still in the heap"
        );
        drop(pin);

        mgr.abort(&t1).unwrap();
        assert_eq!(t1.state(), TransactionState::Aborted);
        assert_eq!(read(&mgr, &heap, a), Some(tuple(1).data));
        assert_eq!(read(&mgr, &heap, c), None);
        assert_eq!(mgr.num_version_chains(), 0);

        let t2 = mgr.begin(IsolationLevel::SnapshotIsolation);
        mvcc::update_tuple(&t2, &mgr, &heap, &[], a, &tuple(1), &tuple(4))
            .expect("the row is writable again");
        mgr.commit(&t2).unwrap();
        assert_eq!(read(&mgr, &heap, a), Some(tuple(4).data));
    }

    #[test]
    fn rejected_write_leaves_no_undo_entry() {
        let db = Db::new();
        let heap = TableHeap::new(db.bpm.clone()).unwrap();
        let t0 = db.begin();
        let a = mvcc::insert_tuple(&t0, &heap, &[], &tuple(1)).unwrap();
        db.commit(&t0);

        let t1 = db.begin();
        let wider = Tuple { data: vec![9; 17] };
        assert!(mvcc::update_tuple(&t1, &db.mgr, &heap, &[], a, &tuple(1), &wider).is_err());
        assert_eq!(t1.state(), TransactionState::Tainted);
        assert_eq!(db.mgr.num_version_chains(), 0);
        assert!(t1.write_set().is_empty());
        db.abort(&t1);
        assert_eq!(read(&db.mgr, &heap, a), Some(tuple(1).data));
    }
}
