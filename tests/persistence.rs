//! A database opened with `Database::open` keeps its tables, rows and indexes across
//! closing and reopening it.
use cmudb::catalog::Value;
use cmudb::database::Row;
use cmudb::database_file::{CHAIN_PAGE_CAPACITY, DatabaseMeta, PersistError};
use cmudb::processor::plan::Plan;
use cmudb::{Database, QueryError, StatementResult};
use ordered_float::OrderedFloat;
use std::path::Path;

const FRAMES: usize = 32;

fn open(dir: &Path) -> Database {
    Database::open(dir, FRAMES).expect("open the database")
}

fn rows(db: &Database, sql: &str) -> Vec<Row> {
    let results = db.execute(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    match results.into_iter().last() {
        Some(StatementResult::Rows { rows, .. }) => rows,
        other => panic!("{sql} returned no rows: {other:?}"),
    }
}

fn ints(db: &Database, sql: &str) -> Vec<Vec<i32>> {
    rows(db, sql)
        .into_iter()
        .map(|r| {
            r.into_iter()
                .map(|v| match v {
                    Some(Value::INT(i)) => i,
                    other => panic!("{sql}: expected INT, got {other:?}"),
                })
                .collect()
        })
        .collect()
}

fn run(db: &Database, sql: &str) {
    db.execute(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn uses_index(db: &Database, sql: &str) -> bool {
    let mut plan = &db.plan(sql).unwrap()[0];
    loop {
        match plan {
            Plan::IndexScan(_) => return true,
            Plan::Projection(p) => plan = &p.child,
            Plan::Filter(f) => plan = &f.child,
            _ => return false,
        }
    }
}

#[test]
fn tables_rows_and_indexes_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE t (id INTEGER, ok BOOLEAN, f FLOAT);\
             CREATE INDEX t_id ON t (id);\
             INSERT INTO t VALUES (1, true, 1.5), (2, NULL, 2.5), (3, false, 3.5);\
             UPDATE t SET f = 9.5 WHERE id = 2;\
             DELETE FROM t WHERE id = 3;\
             CREATE TABLE u (a INTEGER);\
             INSERT INTO u VALUES (10), (20);\
             CREATE TABLE gone (a INTEGER);\
             DROP TABLE gone;",
        );
        db.close().unwrap();
    }

    let db = open(dir.path());
    assert_eq!(
        rows(&db, "SELECT * FROM t;"),
        vec![
            vec![
                Some(Value::INT(1)),
                Some(Value::BOOLEAN(true)),
                Some(Value::FLOAT(OrderedFloat(1.5)))
            ],
            vec![
                Some(Value::INT(2)),
                None,
                Some(Value::FLOAT(OrderedFloat(9.5)))
            ],
        ],
        "committed rows, updates and deletes are all as they were"
    );
    assert_eq!(ints(&db, "SELECT a FROM u;"), vec![vec![10], vec![20]]);
    assert!(
        db.execute("SELECT * FROM gone;").is_err(),
        "a dropped table stays dropped"
    );

    let by_id = "SELECT id FROM t WHERE id = 2;";
    assert!(uses_index(&db, by_id), "the reopened index is planned");
    assert_eq!(ints(&db, by_id), vec![vec![2]]);
    assert!(
        db.execute("INSERT INTO t VALUES (1, true, 0.0);").is_err(),
        "the reopened index still enforces unique keys"
    );

    run(
        &db,
        "INSERT INTO t VALUES (4, true, 4.5); INSERT INTO u VALUES (30);",
    );
    assert_eq!(
        ints(&db, "SELECT id FROM t WHERE id >= 2;"),
        vec![vec![2], vec![4]],
        "new rows go into the reopened heap and index"
    );
    assert_eq!(
        ints(&db, "SELECT a FROM u;"),
        vec![vec![10], vec![20], vec![30]]
    );
}

#[test]
fn a_database_can_be_reopened_again_and_again() {
    let dir = tempfile::tempdir().unwrap();
    for round in 0..4 {
        let db = open(dir.path());
        if round == 0 {
            run(
                &db,
                "CREATE TABLE t (a INTEGER); CREATE INDEX t_a ON t (a);",
            );
        }
        run(&db, &format!("INSERT INTO t VALUES ({round});"));
        let expected: Vec<Vec<i32>> = (0..=round).map(|r| vec![r]).collect();
        assert_eq!(ints(&db, "SELECT a FROM t;"), expected, "round {round}");
        db.close().unwrap();
    }
}

#[test]
fn a_transaction_left_open_is_rolled_back_on_close() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);\
             BEGIN; INSERT INTO t VALUES (2); UPDATE t SET a = 100 WHERE a = 1;",
        );
        db.close().unwrap();
    }

    let db = open(dir.path());
    assert_eq!(ints(&db, "SELECT a FROM t;"), vec![vec![1]]);
    run(
        &db,
        "BEGIN; UPDATE t SET a = 5 WHERE a = 1; INSERT INTO t VALUES (6); COMMIT;",
    );
    assert_eq!(ints(&db, "SELECT a FROM t;"), vec![vec![5], vec![6]]);
}

#[test]
fn committed_data_is_on_disk_without_closing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    run(
        &db,
        "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);\
         BEGIN; INSERT INTO t VALUES (2); COMMIT;",
    );
    std::mem::forget(db);

    let db = open(dir.path());
    assert_eq!(ints(&db, "SELECT a FROM t;"), vec![vec![1], vec![2]]);
}

#[test]
fn dropping_a_database_writes_it_like_close() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        let txn = db
            .txn_mgr()
            .begin(cmudb::concurrency::IsolationLevel::SnapshotIsolation);
        db.execute("CREATE TABLE t (a INTEGER);").unwrap();
        db.execute_in(&txn, "INSERT INTO t VALUES (7);").unwrap();
        db.txn_mgr().commit(&txn).unwrap();
    }

    let db = open(dir.path());
    assert_eq!(
        ints(&db, "SELECT a FROM t;"),
        vec![vec![7]],
        "a commit made on the transaction manager is written when the database drops"
    );
}

#[test]
fn freed_pages_stay_free_across_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let before = {
        let db = Database::open(dir.path(), FRAMES)
            .unwrap()
            .with_sort_memory_budget(64);
        run(&db, "CREATE TABLE t (a INTEGER);");
        let values: Vec<String> = (0..2000).map(|i| format!("({})", 2000 - i)).collect();
        run(&db, &format!("INSERT INTO t VALUES {};", values.join(", ")));
        assert_eq!(rows(&db, "SELECT a FROM t ORDER BY a;").len(), 2000);
        let allocation = db.bpm().allocation();
        assert!(
            !allocation.free_page_ids.is_empty(),
            "the sort spilled runs and freed their pages"
        );
        db.close().unwrap();
        allocation
    };

    let db = open(dir.path());
    assert_eq!(db.bpm().allocation(), before);
    run(&db, "CREATE TABLE u (a INTEGER);");
    let u = db
        .catalog()
        .get_table(db.catalog().table_oid("u").unwrap())
        .unwrap();
    assert_eq!(
        u.heap.first_page_id, before.free_page_ids[0],
        "the next new page takes the lowest freed id"
    );
    assert_eq!(db.bpm().allocation().next_page_id, before.next_page_id);
}

#[test]
fn a_catalog_larger_than_a_page_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let names: Vec<String> = (0..150)
        .map(|i| format!("table_with_a_rather_long_name_number_{i:04}"))
        .collect();
    {
        let db = Database::open(dir.path(), 256).unwrap();
        for (i, name) in names.iter().enumerate() {
            run(
                &db,
                &format!(
                    "CREATE TABLE {name} (first_column INTEGER, second_column INTEGER);\
                     INSERT INTO {name} VALUES ({i}, {i});"
                ),
            );
        }
        let meta = DatabaseMeta {
            allocation: db.bpm().allocation(),
            last_commit_ts: db.txn_mgr().last_commit_ts(),
            next_txn_id: db.txn_mgr().next_txn_id(),
            catalog: db.catalog().snapshot(),
        };
        assert!(
            meta.encode().len() > CHAIN_PAGE_CAPACITY,
            "the metadata spills onto a second chain page"
        );
        db.close().unwrap();
    }

    let db = Database::open(dir.path(), 256).unwrap();
    for (i, name) in names.iter().enumerate() {
        assert_eq!(
            ints(&db, &format!("SELECT first_column FROM {name};")),
            vec![vec![i as i32]],
            "{name}"
        );
    }
}

#[test]
fn a_file_that_is_not_a_database_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(cmudb::database::DATA_FILE), vec![7u8; 8192]).unwrap();
    assert!(matches!(
        Database::open(dir.path(), FRAMES),
        Err(PersistError::NotADatabase)
    ));
}

#[test]
fn a_new_database_is_on_disk_before_its_first_commit() {
    let dir = tempfile::tempdir().unwrap();
    std::mem::forget(open(dir.path()));
    let db = open(dir.path());
    assert!(db.catalog().snapshot().tables.is_empty());
    assert!(matches!(
        db.execute("SELECT * FROM t;"),
        Err(QueryError::RolledBack { .. })
    ));
}

#[test]
#[ignore = "known gap: BOR-71, pages evicted between commits can link past the recorded allocation"]
fn a_kill_while_a_block_grows_a_table_leaves_a_database_that_opens() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path(), 8).unwrap();
        run(&db, "CREATE TABLE t (a INTEGER);");
        let values: Vec<String> = (0..20_000).map(|i| format!("({i})")).collect();
        run(
            &db,
            &format!("BEGIN; INSERT INTO t VALUES {};", values.join(", ")),
        );
        std::mem::forget(db);
    }

    let db = Database::open(dir.path(), 8).expect("reopen after a kill");
    run(&db, "CREATE TABLE u (a INTEGER);");
    let values: Vec<String> = (0..20_000).map(|i| format!("({i})")).collect();
    run(&db, &format!("INSERT INTO u VALUES {};", values.join(", ")));
    assert_eq!(rows(&db, "SELECT a FROM t;").len(), 0);
    assert_eq!(rows(&db, "SELECT a FROM u;").len(), 20_000);
}
