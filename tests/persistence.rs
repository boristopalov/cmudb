//! A database opened with `Database::open` keeps its tables, rows and indexes across
//! closing and reopening it.
use cmudb::catalog::Value;
use cmudb::concurrency::IsolationLevel;
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
fn bigint_and_float_values_keep_their_full_width_across_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE w (n BIGINT, x FLOAT); CREATE INDEX w_n ON w (n);\
             INSERT INTO w VALUES (9000000000, 0.1), (-9000000000, 16777217);",
        );
        db.close().unwrap();
    }

    let db = open(dir.path());
    assert_eq!(
        rows(&db, "SELECT n, x FROM w WHERE n = 9000000000;"),
        vec![vec![
            Some(Value::BIGINT(9_000_000_000)),
            Some(Value::FLOAT(OrderedFloat(0.1)))
        ]]
    );
    assert_eq!(
        rows(&db, "SELECT x FROM w WHERE n = -9000000000;"),
        vec![vec![Some(Value::FLOAT(OrderedFloat(16_777_217.0)))]]
    );
}

#[test]
fn char_columns_keep_their_length_and_index_across_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE c (code CHAR(3), n INT); CREATE INDEX c_code ON c (code);\
             INSERT INTO c VALUES ('us', 1), ('fra', 2);",
        );
        db.close().unwrap();
    }

    let db = open(dir.path());
    let query = "SELECT n FROM c WHERE code = 'us ';";
    assert!(uses_index(&db, query));
    assert_eq!(rows(&db, query), vec![vec![Some(Value::INT(1))]]);
    assert_eq!(
        rows(&db, "SELECT code FROM c WHERE n = 1;"),
        vec![vec![Some(Value::text("us "))]]
    );
    let err = db
        .execute("INSERT INTO c VALUES ('deu1', 3);")
        .expect_err("the CHAR(3) length is kept on disk");
    assert!(err.to_string().contains("CHAR(3)"), "{err}");
}

#[test]
fn text_rows_and_varchar_indexes_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE p (name VARCHAR(10), bio TEXT); CREATE INDEX p_name ON p (name);\
             INSERT INTO p VALUES ('ada', 'wrote the first program'), ('grace', NULL);\
             UPDATE p SET bio = 'found a moth in the relay' WHERE name = 'grace';",
        );
        db.close().unwrap();
    }

    let db = open(dir.path());
    let query = "SELECT bio FROM p WHERE name = 'grace';";
    assert!(uses_index(&db, query));
    assert_eq!(
        rows(&db, query),
        vec![vec![Some(Value::text("found a moth in the relay"))]]
    );
    let err = db
        .execute("INSERT INTO p VALUES ('much too long a name', NULL);")
        .expect_err("the VARCHAR(10) limit is kept on disk");
    assert!(err.to_string().contains("VARCHAR(10)"), "{err}");
    assert_eq!(
        rows(
            &db,
            "SELECT name FROM p WHERE bio = 'wrote the first program';"
        ),
        vec![vec![Some(Value::text("ada"))]]
    );
}

#[test]
fn text_indexes_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let body = |n: i32| format!("note {n:03} {}", "z".repeat((n as usize * 41) % 900));
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE notes (id INTEGER, body TEXT); CREATE INDEX by_body ON notes (body);",
        );
        for n in 0..200 {
            run(
                &db,
                &format!("INSERT INTO notes VALUES ({n}, '{}');", body(n)),
            );
        }
        db.close().unwrap();
    }

    let db = open(dir.path());
    let query = format!("SELECT id FROM notes WHERE body = '{}';", body(150));
    assert!(uses_index(&db, &query));
    assert_eq!(ints(&db, &query), vec![vec![150]]);
    let range = "SELECT id FROM notes WHERE body >= 'note 010' AND body < 'note 013';";
    assert_eq!(ints(&db, range), vec![vec![10], vec![11], vec![12]]);
    run(
        &db,
        &format!("INSERT INTO notes VALUES (500, '{}');", body(500)),
    );
    assert_eq!(
        ints(
            &db,
            &format!("SELECT id FROM notes WHERE body = '{}';", body(500))
        ),
        vec![vec![500]]
    );
}

#[test]
fn a_data_file_of_the_previous_format_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    open(dir.path()).close().unwrap();
    let path = dir.path().join(cmudb::database::DATA_FILE);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        0,
        "a new database's pages wait in the log until a checkpoint"
    );
    open(dir.path()).close().unwrap();
    let mut file = std::fs::read(&path).unwrap();
    assert!(
        !file.is_empty(),
        "opening replays the log into the data file"
    );
    let version_at = 8;
    file[version_at..version_at + 4].copy_from_slice(&2u32.to_le_bytes());
    std::fs::write(&path, file).unwrap();

    let Err(err) = Database::open(dir.path(), FRAMES) else {
        panic!("a version 2 file is refused");
    };
    assert!(matches!(err, PersistError::UnsupportedFormat(2)), "{err}");
    assert_eq!(
        err.to_string(),
        "the data file has format version 2, but this cmudb reads version 3"
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
    db.abandon();

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
fn a_directory_open_in_one_database_cannot_be_opened_again_until_it_closes() {
    let dir = tempfile::tempdir().unwrap();
    let first = open(dir.path());
    let second = Database::open(dir.path(), FRAMES).err().unwrap();
    assert!(matches!(second, PersistError::InUse), "{second}");
    first.close().unwrap();
    open(dir.path()).close().unwrap();
}

#[test]
fn a_new_database_is_on_disk_before_its_first_commit() {
    let dir = tempfile::tempdir().unwrap();
    open(dir.path()).abandon();
    let db = open(dir.path());
    assert!(db.catalog().snapshot().tables.is_empty());
    assert!(matches!(
        db.execute("SELECT * FROM t;"),
        Err(QueryError::RolledBack { .. })
    ));
}

#[test]
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
        db.abandon();
    }

    let db = Database::open(dir.path(), 8).expect("reopen after a kill");
    run(&db, "CREATE TABLE u (a INTEGER);");
    let values: Vec<String> = (0..20_000).map(|i| format!("({i})")).collect();
    run(&db, &format!("INSERT INTO u VALUES {};", values.join(", ")));
    assert_eq!(rows(&db, "SELECT a FROM t;").len(), 0);
    assert_eq!(rows(&db, "SELECT a FROM u;").len(), 20_000);
}

fn begin(db: &Database) -> std::sync::Arc<cmudb::concurrency::Transaction> {
    db.txn_mgr().begin(IsolationLevel::SnapshotIsolation)
}

#[test]
fn a_kill_rolls_back_a_transaction_whose_writes_another_commit_made_durable() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE t (a INTEGER, b TEXT); CREATE INDEX t_a ON t (a);\
             INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three');",
        );
        let open_txn = begin(&db);
        db.execute_in(
            &open_txn,
            "UPDATE t SET b = 'a much longer text than before' WHERE a = 1;\
             UPDATE t SET b = 'twice' WHERE a = 1;\
             DELETE FROM t WHERE a = 2; INSERT INTO t VALUES (4, 'four');\
             UPDATE t SET a = 40 WHERE a = 4;",
        )
        .unwrap();
        run(&db, "INSERT INTO t VALUES (5, 'five');");
        db.abandon();
    }

    let db = open(dir.path());
    assert_eq!(
        rows(&db, "SELECT a, b FROM t ORDER BY a;"),
        vec![
            vec![Some(Value::INT(1)), Some(Value::text("one"))],
            vec![Some(Value::INT(2)), Some(Value::text("two"))],
            vec![Some(Value::INT(3)), Some(Value::text("three"))],
            vec![Some(Value::INT(5)), Some(Value::text("five"))],
        ],
        "the open transaction's writes are gone and the commit after them is kept"
    );
    run(
        &db,
        "UPDATE t SET b = 'mine' WHERE a = 1; DELETE FROM t WHERE a = 2;\
         INSERT INTO t VALUES (4, 'again');",
    );
    assert_eq!(
        ints(
            &db,
            "SELECT a FROM t WHERE b = 'mine' OR b = 'again' ORDER BY a;"
        ),
        vec![vec![1], vec![4]],
        "the rows the open transaction held can be written again"
    );
    assert_eq!(
        ints(&db, "SELECT a FROM t WHERE a = 40;"),
        Vec::<Vec<i32>>::new()
    );
}

#[test]
fn a_kill_rolls_back_an_update_that_moved_its_row_to_another_page() {
    let dir = tempfile::tempdir().unwrap();
    let long = |c: char, n: usize| c.to_string().repeat(n);
    {
        let db = open(dir.path());
        run(
            &db,
            "CREATE TABLE t (a INTEGER, b TEXT); CREATE INDEX t_a ON t (a);",
        );
        for a in 1..=6 {
            run(
                &db,
                &format!("INSERT INTO t VALUES ({a}, '{}');", long('x', 1200)),
            );
        }
        let open_txn = begin(&db);
        db.execute_in(
            &open_txn,
            &format!("UPDATE t SET b = '{}' WHERE a = 1;", long('y', 4000)),
        )
        .unwrap();
        run(&db, "INSERT INTO t VALUES (7, 'seven');");
        db.abandon();
    }

    let db = open(dir.path());
    assert_eq!(
        rows(&db, "SELECT a, b FROM t WHERE a = 1 OR a = 7 ORDER BY a;"),
        vec![
            vec![Some(Value::INT(1)), Some(Value::text(&long('x', 1200)))],
            vec![Some(Value::INT(7)), Some(Value::text("seven"))],
        ],
        "the moved row is back at its old version and the later commit is kept"
    );
    assert_eq!(ints(&db, "SELECT COUNT(*) FROM t;"), vec![vec![7]]);
    run(
        &db,
        &format!("UPDATE t SET b = '{}' WHERE a = 1;", long('z', 4000)),
    );
    assert_eq!(
        ints(
            &db,
            "SELECT a FROM t WHERE b <> 'seven' AND a < 3 ORDER BY a;"
        ),
        vec![vec![1], vec![2]],
        "the row the open transaction moved can be moved again"
    );
}

#[test]
fn a_kill_after_writes_evicted_from_a_small_pool_keeps_only_the_commits() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<String> = (0..5_000).map(|i| format!("({i})")).collect();
    {
        let db = Database::open(dir.path(), 8).unwrap();
        run(
            &db,
            "CREATE TABLE t (a INTEGER); CREATE INDEX t_a ON t (a);",
        );
        run(&db, &format!("INSERT INTO t VALUES {};", values.join(", ")));
        let open_txn = begin(&db);
        db.execute_in(&open_txn, "UPDATE t SET a = a + 100000;")
            .unwrap();
        db.execute_in(
            &open_txn,
            &format!("INSERT INTO t VALUES {};", values.join(", ")),
        )
        .unwrap();
        run(&db, "CREATE TABLE u (a INTEGER); INSERT INTO u VALUES (1);");
        db.abandon();
    }

    let db = Database::open(dir.path(), 8).expect("reopen after a kill");
    assert_eq!(rows(&db, "SELECT a FROM t;").len(), 5_000);
    assert_eq!(
        ints(&db, "SELECT a FROM t WHERE a >= 100000;"),
        Vec::<Vec<i32>>::new()
    );
    assert_eq!(
        ints(&db, "SELECT a FROM t WHERE a = 4321;"),
        vec![vec![4321]]
    );
    assert_eq!(ints(&db, "SELECT a FROM u;"), vec![vec![1]]);
    run(&db, "UPDATE t SET a = a + 1000000;");
    assert_eq!(
        ints(&db, "SELECT a FROM t WHERE a = 1004321;"),
        vec![vec![1004321]],
        "every row the open transaction held can be written again"
    );
}

#[test]
fn an_open_transaction_survives_a_checkpoint_and_is_rolled_back_after_a_kill() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        db.bpm().disk_manager().set_checkpoint_log_bytes(0);
        run(
            &db,
            "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2);",
        );
        let open_txn = begin(&db);
        db.execute_in(
            &open_txn,
            "UPDATE t SET a = 10 WHERE a = 1; INSERT INTO t VALUES (3);",
        )
        .unwrap();
        for i in 0..5 {
            run(&db, &format!("INSERT INTO t VALUES ({});", 100 + i));
        }
        db.abandon();
    }

    let db = open(dir.path());
    assert_eq!(
        ints(&db, "SELECT a FROM t ORDER BY a;"),
        vec![
            vec![1],
            vec![2],
            vec![100],
            vec![101],
            vec![102],
            vec![103],
            vec![104]
        ]
    );
    db.close().unwrap();
    let db = open(dir.path());
    assert_eq!(
        rows(&db, "SELECT a FROM t;").len(),
        7,
        "recovery is not repeated"
    );
}

#[test]
fn a_commit_that_is_not_durable_yet_is_lost_whole_by_a_kill() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(&db, "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);");
        let txn = begin(&db);
        db.execute_in(
            &txn,
            "INSERT INTO t VALUES (2); UPDATE t SET a = 10 WHERE a = 1;",
        )
        .unwrap();
        db.txn_mgr().commit(&txn).unwrap();
        db.bpm().flush_all().unwrap();
        db.abandon();
    }

    let db = open(dir.path());
    assert_eq!(
        ints(&db, "SELECT a FROM t ORDER BY a;"),
        vec![vec![1]],
        "a commit on the transaction manager is durable at the next commit through the \
         database, and flushing pages alone does not make it so"
    );
}

#[test]
fn a_torn_write_at_the_end_of_the_log_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        run(&db, "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);");
        db.abandon();
    }
    let log = dir.path().join(cmudb::database::LOG_FILE);
    let mut bytes = std::fs::read(&log).unwrap();
    bytes.extend_from_slice(&[1, 0x10, 0x20, 0, 0, 7, 7, 7, 7, 42, 42]);
    std::fs::write(&log, bytes).unwrap();

    let db = open(dir.path());
    assert_eq!(ints(&db, "SELECT a FROM t;"), vec![vec![1]]);
}

#[test]
fn a_log_that_is_not_a_cmudb_log_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    open(dir.path()).close().unwrap();
    std::fs::write(
        dir.path().join(cmudb::database::LOG_FILE),
        b"not a log at all",
    )
    .unwrap();
    assert!(matches!(
        Database::open(dir.path(), FRAMES),
        Err(PersistError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidData
    ));
}
