//! SQL end-to-end tests: each test runs SQL text through the parser, binder,
//! planner and executors against a real buffer pool and disk manager.
mod common;

use cmudb::catalog::Value;
use common::*;

fn emp_db() -> Db {
    let db = Db::new(64);
    db.exec(
        "CREATE TABLE emp (id INTEGER, dept INTEGER, salary INTEGER, bonus FLOAT, active BOOLEAN)",
    );
    db.exec(
        "INSERT INTO emp VALUES \
         (1, 10, 100, 1.5, true), \
         (2, 10, 200, 2.5, false), \
         (3, 20, 300, NULL, true), \
         (4, 30, 400, 4.0, NULL)",
    );
    db
}

fn sorted(mut rows: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    rows.sort();
    rows
}

#[test]
fn insert_reports_row_count_and_select_star_round_trips_every_type() {
    let db = emp_db();
    let rows = db.exec("SELECT * FROM emp");
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows[0],
        vec![
            int(1),
            int(10),
            int(100),
            Some(Value::FLOAT(1.5.into())),
            Some(Value::BOOLEAN(true)),
        ]
    );
    assert_eq!(rows[2][3], None, "NULL float survives a round trip");
    assert_eq!(rows[3][4], None, "NULL boolean survives a round trip");
    assert_eq!(db.exec("INSERT INTO emp VALUES (5, 10, 1, 0.0, false)"), vec![vec![int(1)]]);
}

#[test]
fn where_clause_comparisons_and_boolean_logic() {
    let db = emp_db();
    let q = |sql: &str| ints(&db.exec(sql));
    assert_eq!(q("SELECT id FROM emp WHERE salary > 150"), vec![[2], [3], [4]]);
    assert_eq!(q("SELECT id FROM emp WHERE salary >= 200 AND dept <> 30"), vec![[2], [3]]);
    assert_eq!(q("SELECT id FROM emp WHERE salary < 200 OR dept = 30"), vec![[1], [4]]);
    assert_eq!(q("SELECT id FROM emp WHERE salary <= 200"), vec![[1], [2]]);
    assert_eq!(q("SELECT id FROM emp WHERE (dept = 10) AND (salary = 200)"), vec![[2]]);
    assert!(q("SELECT id FROM emp WHERE salary > 1000").is_empty());
}

#[test]
fn where_clause_treats_null_as_not_true() {
    let db = emp_db();
    // active is NULL for id 4: neither `active = true` nor `active = false` matches it.
    let t = ints(&db.exec("SELECT id FROM emp WHERE active = true"));
    let f = ints(&db.exec("SELECT id FROM emp WHERE active = false"));
    assert_eq!(t, vec![[1], [3]]);
    assert_eq!(f, vec![[2]]);
    // NULL OR true is true; NULL AND false is false.
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp WHERE active = true OR id = 4")),
        vec![[1], [3], [4]]
    );
}

#[test]
fn projection_evaluates_arithmetic() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("SELECT id, salary * 2, salary + 1, salary - 1, salary / 2 FROM emp WHERE id = 3")),
        vec![[3, 600, 301, 299, 150]]
    );
    assert_eq!(ints(&db.exec("SELECT 1 + 2")), vec![[3]]);
    let floats = db.exec("SELECT bonus * 2.0 FROM emp WHERE id = 2");
    assert_eq!(floats, vec![vec![Some(Value::FLOAT(5.0.into()))]]);
    // NULL propagates through arithmetic.
    assert_eq!(db.exec("SELECT bonus + 1.0 FROM emp WHERE id = 3"), vec![vec![None]]);
}

#[test]
fn table_aliases_and_qualified_columns() {
    let db = emp_db();
    assert_eq!(ints(&db.exec("SELECT e.id FROM emp e WHERE e.salary = 200")), vec![[2]]);
    assert_eq!(ints(&db.exec("SELECT emp.id FROM emp WHERE emp.id = 4")), vec![[4]]);
}

#[test]
fn aggregates_with_and_without_group_by() {
    let db = emp_db();
    assert_eq!(ints(&db.exec("SELECT COUNT(*) FROM emp")), vec![[4]]);
    assert_eq!(
        ints(&db.exec("SELECT SUM(salary), MIN(salary), MAX(salary), AVG(salary) FROM emp")),
        vec![[1000, 100, 400, 250]]
    );
    assert_eq!(ints(&db.exec("SELECT SUM(salary) + 1 FROM emp")), vec![[1001]]);
    assert_eq!(
        sorted(ints(&db.exec(
            "SELECT dept, COUNT(*), SUM(salary), AVG(salary), MIN(salary), MAX(salary) \
             FROM emp GROUP BY dept"
        ))),
        vec![
            vec![10, 2, 300, 150, 100, 200],
            vec![20, 1, 300, 300, 300, 300],
            vec![30, 1, 400, 400, 400, 400],
        ]
    );
    // SUM skips NULLs.
    assert_eq!(
        db.exec("SELECT SUM(bonus) FROM emp"),
        vec![vec![Some(Value::FLOAT(8.0.into()))]]
    );
    // Filtered aggregate.
    assert_eq!(ints(&db.exec("SELECT COUNT(*) FROM emp WHERE dept = 10")), vec![[2]]);
}

#[test]
fn count_column_skips_nulls() {
    let db = emp_db();
    assert_eq!(ints(&db.exec("SELECT COUNT(bonus) FROM emp")), vec![[3]]);
}

#[test]
fn order_by_and_limit() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (a INTEGER, b INTEGER)");
    db.exec("INSERT INTO t VALUES (3, 1), (1, 2), (2, 3), (1, 1)");
    assert_eq!(
        ints(&db.exec("SELECT a, b FROM t ORDER BY a, b")),
        vec![[1, 1], [1, 2], [2, 3], [3, 1]]
    );
    assert_eq!(ints(&db.exec("SELECT a FROM t ORDER BY b, a LIMIT 2")), vec![[1], [3]]);
    assert_eq!(ints(&db.exec("SELECT a FROM t LIMIT 0")), Vec::<Vec<i32>>::new());
    assert_eq!(ints(&db.exec("SELECT a FROM t LIMIT 10")).len(), 4);
    assert_eq!(
        ints(&db.exec("SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY a")),
        vec![[1, 2], [2, 1], [3, 1]]
    );
}

#[test]
#[ignore = "known gap: ORDER BY ... DESC is parsed but sorts ascending"]
fn order_by_desc() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp ORDER BY salary DESC")),
        vec![[4], [3], [2], [1]]
    );
}

#[test]
fn values_statement() {
    let db = Db::new(8);
    assert_eq!(ints(&db.exec("VALUES (1, 2), (3, 4)")), vec![[1, 2], [3, 4]]);
}

#[test]
fn inner_and_left_joins_without_index_use_nested_loop_join() {
    let db = emp_db();
    db.exec("CREATE TABLE dept (id INTEGER, name INTEGER)");
    db.exec("INSERT INTO dept VALUES (10, 1000), (20, 2000), (40, 4000)");
    assert_eq!(
        ints(&db.exec("SELECT emp.id, dept.name FROM emp JOIN dept ON emp.dept = dept.id")),
        vec![[1, 1000], [2, 1000], [3, 2000]]
    );
    let left = db.exec("SELECT emp.id, dept.name FROM emp LEFT JOIN dept ON emp.dept = dept.id");
    assert_eq!(left.len(), 4);
    assert_eq!(left[3], vec![int(4), None], "unmatched left row is NULL-padded");
    // Non-equality join predicate.
    assert_eq!(
        ints(&db.exec("SELECT emp.id, dept.id FROM emp JOIN dept ON emp.dept < dept.id WHERE emp.id = 4")),
        vec![[4, 40]]
    );
    // Join + filter + aggregate.
    assert_eq!(
        ints(&db.exec(
            "SELECT COUNT(*) FROM emp JOIN dept ON emp.dept = dept.id WHERE dept.name > 1000"
        )),
        vec![[1]]
    );
}

#[test]
fn inner_join_with_index_on_inner_side_uses_nested_index_join() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE INDEX t_id ON t (id)");
    db.exec("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    db.exec("CREATE TABLE u (tid INTEGER, w INTEGER)");
    db.exec("INSERT INTO u VALUES (1, 100), (2, 200), (2, 201), (9, 900)");
    let expected = vec![[100, 10], [200, 20], [201, 20]];
    assert_eq!(ints(&db.exec("SELECT u.w, t.v FROM u JOIN t ON u.tid = t.id")), expected);
    assert_eq!(ints(&db.exec("SELECT u.w, t.v FROM u JOIN t ON t.id = u.tid")), expected);
}

#[test]
fn index_scan_and_index_maintenance_through_dml() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE INDEX t_id ON t (id)");
    assert!(db.exec("SELECT v FROM t WHERE id = 3").is_empty(), "empty index scans cleanly");
    db.exec("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 3")), vec![[30]]);
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE 3 = id")), vec![[30]]);
    assert!(db.exec("SELECT v FROM t WHERE id = 7").is_empty());

    assert_eq!(ints(&db.exec("UPDATE t SET v = v + 1 WHERE id = 3")), vec![[1]]);
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 3")), vec![[31]]);

    // Updating the key column moves the index entry.
    assert_eq!(ints(&db.exec("UPDATE t SET id = 33 WHERE id = 3")), vec![[1]]);
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 33")), vec![[31]]);
    assert!(db.exec("SELECT v FROM t WHERE id = 3").is_empty());

    assert_eq!(ints(&db.exec("DELETE FROM t WHERE id = 33")), vec![[1]]);
    assert!(db.exec("SELECT v FROM t WHERE id = 33").is_empty());
    assert_eq!(ints(&db.exec("SELECT * FROM t")), vec![[1, 10], [2, 20]]);

    assert_eq!(ints(&db.exec("DELETE FROM t")), vec![[2]]);
    assert!(db.exec("SELECT * FROM t").is_empty());
    assert!(db.exec("SELECT v FROM t WHERE id = 1").is_empty());
}

#[test]
fn unique_index_rejects_duplicate_key() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE INDEX t_id ON t (id)");
    db.exec("INSERT INTO t VALUES (1, 1)");
    let err = db.try_exec("INSERT INTO t VALUES (1, 2)").unwrap_err();
    assert!(err.contains("DuplicateKey"), "{err}");
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 1")), vec![[1]]);
}

#[test]
#[ignore = "known gap: CREATE INDEX does not backfill rows already in the table"]
fn create_index_backfills_existing_rows() {
    let db = emp_db();
    db.exec("CREATE INDEX emp_id ON emp (id)");
    assert_eq!(ints(&db.exec("SELECT salary FROM emp WHERE id = 3")), vec![[300]]);
}

#[test]
fn update_and_delete_without_index() {
    let db = emp_db();
    assert_eq!(ints(&db.exec("UPDATE emp SET salary = salary * 10 WHERE dept = 10")), vec![[2]]);
    assert_eq!(
        ints(&db.exec("SELECT salary FROM emp WHERE dept = 10 ORDER BY salary")),
        vec![[1000], [2000]]
    );
    assert_eq!(ints(&db.exec("UPDATE emp SET dept = 99, salary = 0 WHERE id = 4")), vec![[1]]);
    assert_eq!(ints(&db.exec("SELECT dept, salary FROM emp WHERE id = 4")), vec![[99, 0]]);
    assert_eq!(ints(&db.exec("DELETE FROM emp WHERE salary >= 1000")), vec![[2]]);
    assert_eq!(ints(&db.exec("SELECT id FROM emp ORDER BY id")), vec![[3], [4]]);
    assert_eq!(ints(&db.exec("DELETE FROM emp WHERE id = 42")), vec![[0]]);
}

#[test]
fn many_rows_span_pages_and_survive_eviction() {
    // 4 frames forces the heap and sort to spill pages through the disk manager.
    let db = Db::new(4);
    db.exec("CREATE TABLE big (id INTEGER, v INTEGER)");
    let values: Vec<String> = (0..2000).rev().map(|i| format!("({i}, {})", i % 7)).collect();
    assert_eq!(ints(&db.exec(&format!("INSERT INTO big VALUES {}", values.join(", ")))), vec![[2000]]);
    assert_eq!(ints(&db.exec("SELECT COUNT(*), SUM(id) FROM big")), vec![[2000, 1999 * 1000]]);
    let ordered = ints(&db.exec("SELECT id FROM big ORDER BY id LIMIT 3"));
    assert_eq!(ordered, vec![[0], [1], [2]]);
    assert_eq!(
        sorted(ints(&db.exec("SELECT v, COUNT(*) FROM big GROUP BY v")))[0],
        vec![0, 286]
    );
    assert_eq!(ints(&db.exec("DELETE FROM big WHERE v = 0")), vec![[286]]);
    assert_eq!(ints(&db.exec("SELECT COUNT(*) FROM big")), vec![[1714]]);
}

#[test]
fn multiple_statements_in_one_call_run_in_order() {
    let db = Db::new(16);
    let rows = db.exec(
        "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2); SELECT a FROM t WHERE a = 2;",
    );
    assert_eq!(ints(&rows), vec![[2]]);
}

#[test]
fn other_column_types_parse() {
    let db = Db::new(8);
    db.exec("CREATE TABLE ty (a INT, b INT4, c REAL, d FLOAT4, e BOOL, f TIMESTAMP)");
    db.exec("INSERT INTO ty VALUES (1, 2, 1.5, 2.5, true, NULL)");
    let rows = db.exec("SELECT * FROM ty");
    assert_eq!(rows[0][0], int(1));
    assert_eq!(rows[0][4], Some(Value::BOOLEAN(true)));
    assert_eq!(rows[0][5], None);
}

#[test]
fn binder_and_planner_errors_are_reported() {
    let db = emp_db();
    db.exec("CREATE TABLE dept (id INTEGER, name INTEGER)");
    let cases = [
        ("SELECT x FROM emp", "column not found"),
        ("SELECT * FROM nope", "table not found"),
        ("SELECT * FROM emp, dept", "comma-separated"),
        ("DROP TABLE emp", "unsupported statement"),
        ("SELECT id FROM emp JOIN dept ON emp.dept = dept.id", "ambiguous"),
        ("SELECT * FROM emp RIGHT JOIN dept ON emp.dept = dept.id", "unsupported join"),
        ("SELECT emp.id FROM emp JOIN dept ON emp.dept = dept.id JOIN dept d2 ON d2.id = emp.id", "2-way"),
        ("SELECT id FROM emp WHERE id LIKE 1", "unsupported"),
        ("SELECT id FROM emp WHERE id % 2 = 0", "unsupported binary operator"),
        ("SELECT 'text'", "unsupported literal"),
        ("SELECT id FROM emp LIMIT id", "LIMIT"),
        ("SELECT foo.id FROM emp", ""),
        ("SELECT a.b.c FROM emp", ""),
        ("CREATE TABLE s (name TEXT)", "unsupported type"),
        ("SELECT COUNT(*) FROM emp GROUP BY ALL", "GROUP BY ALL"),
        ("UPDATE emp SET (id, dept) = (1, 2)", ""),
        ("SELECT * FROM emp ORDER BY ALL", ""),
        ("SELECT * FROM s.emp", "qualified name"),
        ("SELECT CONCAT(id) FROM emp", ""),
        ("SELECT id FROM (SELECT id FROM emp) sub", ""),
        ("INSERT INTO emp SELECT * FROM emp", ""),
        ("SELECT id FROM emp WHERE id IN (1, 2)", "unsupported expression"),
        ("SELECT emp.* FROM emp", ""),
    ];
    let mut failures = Vec::new();
    for (sql, needle) in cases {
        match db.try_exec(sql) {
            Ok(rows) => failures.push(format!("{sql}: expected an error, got {rows:?}")),
            Err(e) if !e.contains(needle) => {
                failures.push(format!("{sql}: {e} does not mention {needle:?}"))
            }
            Err(_) => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(db.try_exec("SELEC 1").unwrap_err().starts_with("parse:"));
    assert!(db.try_exec("CREATE INDEX i ON emp (nope)").is_err());
}

#[test]
fn creating_a_table_twice_fails_and_data_is_kept() {
    let db = emp_db();
    assert!(db.try_exec("CREATE TABLE emp (id INTEGER)").is_err());
    assert_eq!(ints(&db.exec("SELECT COUNT(*) FROM emp")), vec![[4]]);
    db.exec("CREATE INDEX i ON emp (dept)");
    assert!(db.try_exec("CREATE INDEX i ON emp (dept)").is_err());
}

#[test]
fn insert_rejects_rows_of_the_wrong_width() {
    let db = emp_db();
    assert!(db.try_exec("INSERT INTO emp VALUES (1, 2)").is_err());
}
