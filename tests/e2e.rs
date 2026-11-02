mod common;

use cmudb::table_heap::RecordId;
use common::{Db, ints};

#[test]
fn create_insert_select_round_trip() {
    let db = Db::new(64);
    db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER);");
    assert_eq!(
        ints(&db.exec("INSERT INTO users (id, age) VALUES (1, 25), (2, 50);")),
        vec![[2]]
    );
    assert_eq!(
        ints(&db.exec("SELECT * FROM users;")),
        vec![[1, 25], [2, 50]]
    );
}

/// The record id of every slot in `table`'s heap, in heap order.
fn heap_rids(db: &Db, table: &str) -> Vec<RecordId> {
    let heap = &db.catalog().get_table(db.table_oid(table)).unwrap().heap;
    heap.iter().map(|slot| slot.unwrap().1).collect()
}

#[test]
fn scan_driven_sql_update_visits_each_original_row_once() {
    for row_count in [8, 700] {
        let db = Db::new(1);
        db.exec("CREATE TABLE items (id INTEGER, value INTEGER)");
        let values = (0..row_count)
            .map(|id| format!("({id}, 0)"))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(
            ints(&db.exec(&format!("INSERT INTO items VALUES {values}"))),
            vec![[row_count]]
        );
        let original_rids = heap_rids(&db, "items");
        assert_eq!(original_rids.len(), row_count as usize);

        assert_eq!(
            ints(&db.exec("UPDATE items SET value = value + 1")),
            vec![[row_count]],
            "the update changes each original row once"
        );

        let expected: Vec<[i32; 2]> = (0..row_count).map(|id| [id, 1]).collect();
        assert_eq!(ints(&db.exec("SELECT * FROM items")), expected);
        assert_eq!(
            heap_rids(&db, "items"),
            original_rids,
            "updates happen in place, so every row keeps its record id"
        );
    }
}

#[test]
fn where_and_join_queries_over_table_scans_open_their_children() {
    let db = Db::new(16);
    db.exec("CREATE TABLE t (a INTEGER, b INTEGER)");
    db.exec("INSERT INTO t VALUES (1, 2), (3, 4), (5, 6)");
    assert_eq!(db.exec("SELECT a FROM t WHERE b > 3").len(), 2);
    assert_eq!(
        db.exec("SELECT * FROM t AS x JOIN t AS y ON x.a = y.a")
            .len(),
        3
    );
}
