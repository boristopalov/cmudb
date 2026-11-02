//! SQL end-to-end tests: each test runs SQL text through the parser, binder,
//! planner and executors against a real buffer pool and disk manager.
mod common;

use cmudb::QueryError;
use cmudb::catalog::Value;
use cmudb::parser::{BinderError, Unsupported};
use cmudb::processor::plan::Plan;
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
    assert_eq!(
        db.exec("INSERT INTO emp VALUES (5, 10, 1, 0.0, false)"),
        vec![vec![int(1)]]
    );
}

#[test]
fn where_clause_comparisons_and_boolean_logic() {
    let db = emp_db();
    let q = |sql: &str| ints(&db.exec(sql));
    assert_eq!(
        q("SELECT id FROM emp WHERE salary > 150"),
        vec![[2], [3], [4]]
    );
    assert_eq!(
        q("SELECT id FROM emp WHERE salary >= 200 AND dept <> 30"),
        vec![[2], [3]]
    );
    assert_eq!(
        q("SELECT id FROM emp WHERE salary < 200 OR dept = 30"),
        vec![[1], [4]]
    );
    assert_eq!(q("SELECT id FROM emp WHERE salary <= 200"), vec![[1], [2]]);
    assert_eq!(
        q("SELECT id FROM emp WHERE (dept = 10) AND (salary = 200)"),
        vec![[2]]
    );
    assert!(q("SELECT id FROM emp WHERE salary > 1000").is_empty());
}

#[test]
fn where_clause_treats_null_as_not_true() {
    let db = emp_db();
    let t = ints(&db.exec("SELECT id FROM emp WHERE active = true"));
    let f = ints(&db.exec("SELECT id FROM emp WHERE active = false"));
    assert_eq!(t, vec![[1], [3]]);
    assert_eq!(f, vec![[2]]);
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp WHERE active = true OR id = 4")),
        vec![[1], [3], [4]]
    );
}

#[test]
fn projection_evaluates_arithmetic() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec(
            "SELECT id, salary * 2, salary + 1, salary - 1, salary / 2 FROM emp WHERE id = 3"
        )),
        vec![[3, 600, 301, 299, 150]]
    );
    assert_eq!(ints(&db.exec("SELECT 1 + 2")), vec![[3]]);
    let floats = db.exec("SELECT bonus * 2.0 FROM emp WHERE id = 2");
    assert_eq!(floats, vec![vec![Some(Value::FLOAT(5.0.into()))]]);
    assert_eq!(
        db.exec("SELECT bonus + 1.0 FROM emp WHERE id = 3"),
        vec![vec![None]]
    );
}

#[test]
fn table_aliases_and_qualified_columns() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("SELECT e.id FROM emp e WHERE e.salary = 200")),
        vec![[2]]
    );
    assert_eq!(
        ints(&db.exec("SELECT emp.id FROM emp WHERE emp.id = 4")),
        vec![[4]]
    );
}

#[test]
fn aggregates_with_and_without_group_by() {
    let db = emp_db();
    assert_eq!(ints(&db.exec("SELECT COUNT(*) FROM emp")), vec![[4]]);
    assert_eq!(
        ints(&db.exec("SELECT SUM(salary), MIN(salary), MAX(salary), AVG(salary) FROM emp")),
        vec![[1000, 100, 400, 250]]
    );
    assert_eq!(
        ints(&db.exec("SELECT SUM(salary) + 1 FROM emp")),
        vec![[1001]]
    );
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
    assert_eq!(
        db.exec("SELECT SUM(bonus) FROM emp"),
        vec![vec![Some(Value::FLOAT(8.0.into()))]]
    );
    assert_eq!(
        ints(&db.exec("SELECT COUNT(*) FROM emp WHERE dept = 10")),
        vec![[2]]
    );
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
    assert_eq!(
        ints(&db.exec("SELECT a FROM t ORDER BY b, a LIMIT 2")),
        vec![[1], [3]]
    );
    assert_eq!(
        ints(&db.exec("SELECT a FROM t LIMIT 0")),
        Vec::<Vec<i32>>::new()
    );
    assert_eq!(ints(&db.exec("SELECT a FROM t LIMIT 10")).len(), 4);
    assert_eq!(
        ints(&db.exec("SELECT a, COUNT(*) FROM t GROUP BY a ORDER BY a")),
        vec![[1, 2], [2, 1], [3, 1]]
    );
}

#[test]
fn order_by_desc() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp ORDER BY salary DESC")),
        vec![[4], [3], [2], [1]]
    );
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp ORDER BY salary ASC")),
        vec![[1], [2], [3], [4]]
    );
}

#[test]
fn order_by_mixes_directions_per_key() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (a INTEGER, b INTEGER)");
    db.exec("INSERT INTO t VALUES (3, 1), (1, 2), (2, 3), (1, 1)");
    assert_eq!(
        ints(&db.exec("SELECT a, b FROM t ORDER BY a DESC, b")),
        vec![[3, 1], [2, 3], [1, 1], [1, 2]]
    );
    assert_eq!(
        ints(&db.exec("SELECT a, b FROM t ORDER BY a, b DESC")),
        vec![[1, 2], [1, 1], [2, 3], [3, 1]]
    );
    assert_eq!(
        ints(&db.exec("SELECT a FROM t ORDER BY b DESC, a DESC LIMIT 2")),
        vec![[2], [1]]
    );
}

#[test]
fn order_by_puts_nulls_last_ascending_and_first_descending_by_default() {
    let db = emp_db();
    assert_eq!(
        db.exec("SELECT id FROM emp ORDER BY bonus"),
        vec![vec![int(1)], vec![int(2)], vec![int(4)], vec![int(3)]],
        "NULL sorts above every value, so it comes last ascending"
    );
    assert_eq!(
        db.exec("SELECT id FROM emp ORDER BY bonus DESC"),
        vec![vec![int(3)], vec![int(4)], vec![int(2)], vec![int(1)]],
        "NULL sorts above every value, so it comes first descending"
    );
}

#[test]
fn order_by_nulls_first_or_last_places_nulls_in_either_direction() {
    let db = emp_db();
    for (sql, expected) in [
        (
            "SELECT id FROM emp ORDER BY bonus ASC NULLS FIRST",
            [3, 1, 2, 4],
        ),
        (
            "SELECT id FROM emp ORDER BY bonus NULLS FIRST",
            [3, 1, 2, 4],
        ),
        (
            "SELECT id FROM emp ORDER BY bonus ASC NULLS LAST",
            [1, 2, 4, 3],
        ),
        (
            "SELECT id FROM emp ORDER BY bonus DESC NULLS FIRST",
            [3, 4, 2, 1],
        ),
        (
            "SELECT id FROM emp ORDER BY bonus DESC NULLS LAST",
            [4, 2, 1, 3],
        ),
    ] {
        let ids: Vec<i32> = ints(&db.exec(sql)).into_iter().map(|r| r[0]).collect();
        assert_eq!(ids, expected, "{sql}");
    }
}

#[test]
fn order_by_nulls_option_applies_per_key() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp ORDER BY active DESC NULLS LAST, id DESC")),
        vec![[3], [1], [2], [4]]
    );
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp ORDER BY active NULLS FIRST, bonus DESC NULLS LAST")),
        vec![[4], [2], [1], [3]]
    );
}

const SPILL_TEST_ROWS: i32 = 600;
const SPILL_TEST_BUDGET_BYTES: usize = 256;
const SPILL_TEST_FRAMES: usize = 32;

fn spill_test_rows(db: &Db) {
    db.exec("CREATE TABLE s (id INTEGER, k INTEGER, v INTEGER)");
    let values: Vec<String> = (0..SPILL_TEST_ROWS)
        .map(|id| {
            let k = if id % 11 == 0 {
                "NULL".to_string()
            } else {
                ((id * 37) % 50).to_string()
            };
            format!("({id}, {k}, {})", (id * 13) % 7)
        })
        .collect();
    db.exec(&format!("INSERT INTO s VALUES {}", values.join(", ")));
}

#[test]
fn order_by_past_the_sort_memory_budget_spills_and_matches_the_in_memory_sort() {
    let in_memory = Db::new(SPILL_TEST_FRAMES);
    spill_test_rows(&in_memory);

    for sql in [
        "SELECT id FROM s ORDER BY k",
        "SELECT id, k, v FROM s ORDER BY k DESC, v",
        "SELECT id FROM s ORDER BY k NULLS FIRST, v DESC",
        "SELECT id FROM s ORDER BY k ASC NULLS LAST, id DESC",
        "SELECT id FROM s ORDER BY k DESC NULLS FIRST, v",
        "SELECT id FROM s ORDER BY k DESC NULLS LAST, v DESC",
        "SELECT id FROM s ORDER BY v DESC, k LIMIT 25",
    ] {
        let spilling = Db::with_sort_memory_budget(SPILL_TEST_FRAMES, SPILL_TEST_BUDGET_BYTES);
        spill_test_rows(&spilling);
        let pages_before = spilling.bpm().allocated_page_count();
        let live_before = spilling.bpm().live_page_count();

        let spilled = spilling.exec(sql);

        assert!(
            spilling.bpm().allocated_page_count() > pages_before,
            "{sql}: the sort did not spill"
        );
        assert_eq!(
            spilling.bpm().live_page_count(),
            live_before,
            "{sql}: spilled runs were not freed"
        );
        assert_eq!(
            nullable_ints(&spilled),
            nullable_ints(&in_memory.exec(sql)),
            "{sql}"
        );
    }
}

#[test]
fn order_by_within_the_sort_memory_budget_allocates_no_pages() {
    let db = Db::new(SPILL_TEST_FRAMES);
    spill_test_rows(&db);
    let pages_before = db.bpm().allocated_page_count();
    let ids: Vec<i32> = ints(&db.exec("SELECT id FROM s ORDER BY id DESC"))
        .into_iter()
        .map(|r| r[0])
        .collect();
    assert_eq!(ids.first(), Some(&(SPILL_TEST_ROWS - 1)));
    assert_eq!(db.bpm().allocated_page_count(), pages_before);
}

#[test]
fn order_by_plans_an_external_merge_sort() {
    let db = emp_db();
    let plan = db.plan_of("SELECT id FROM emp ORDER BY salary DESC LIMIT 2");
    let Plan::Limit(limit) = plan else {
        panic!("expected LIMIT at the top");
    };
    let Plan::Projection(projection) = *limit.child else {
        panic!("expected a projection under LIMIT");
    };
    assert!(
        matches!(*projection.child, Plan::ExternalMergeSort(_)),
        "ORDER BY is not planned as an external merge sort"
    );
}

#[test]
fn order_by_desc_applies_after_group_by() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("SELECT dept, COUNT(*) FROM emp GROUP BY dept ORDER BY dept DESC")),
        vec![[30, 1], [20, 1], [10, 2]]
    );
}

#[test]
fn order_by_options_that_are_not_supported_are_rejected() {
    let db = emp_db();
    for (sql, expected) in [
        ("SELECT id FROM emp ORDER BY id WITH FILL", "WITH FILL"),
        ("SELECT id FROM emp ORDER BY id INTERPOLATE", "INTERPOLATE"),
    ] {
        let err = db.try_exec(sql).unwrap_err();
        assert!(err.contains(expected), "{sql}: {err}");
    }
}

#[test]
fn values_statement() {
    let db = Db::new(8);
    assert_eq!(
        ints(&db.exec("VALUES (1, 2), (3, 4)")),
        vec![[1, 2], [3, 4]]
    );
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
    assert_eq!(
        left[3],
        vec![int(4), None],
        "unmatched left row is NULL-padded"
    );
    assert_eq!(
        ints(&db.exec(
            "SELECT emp.id, dept.id FROM emp JOIN dept ON emp.dept < dept.id WHERE emp.id = 4"
        )),
        vec![[4, 40]]
    );
    assert_eq!(
        ints(&db.exec(
            "SELECT COUNT(*) FROM emp JOIN dept ON emp.dept = dept.id WHERE dept.name > 1000"
        )),
        vec![[1]]
    );
}

#[test]
fn group_by_and_order_by_read_right_side_join_columns() {
    let db = emp_db();
    db.exec("CREATE TABLE dept (id INTEGER, name INTEGER)");
    db.exec("INSERT INTO dept VALUES (10, 1000), (20, 2000), (40, 4000)");
    assert_eq!(
        ints(&db.exec(
            "SELECT emp.id FROM emp JOIN dept ON emp.dept = dept.id \
             ORDER BY dept.name DESC, emp.id DESC"
        )),
        vec![[3], [2], [1]]
    );
    assert_eq!(
        ints(&db.exec(
            "SELECT dept.name, COUNT(*), SUM(emp.salary) FROM emp JOIN dept ON emp.dept = dept.id \
             GROUP BY dept.name ORDER BY dept.name DESC"
        )),
        vec![[2000, 1, 300], [1000, 2, 300]]
    );
    assert_eq!(
        ints(&db.exec(
            "SELECT dept.id, COUNT(*) FROM emp LEFT JOIN dept ON emp.dept = dept.id \
             WHERE dept.id > 0 GROUP BY dept.id ORDER BY dept.id"
        )),
        vec![[10, 2], [20, 1]]
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
    assert_eq!(
        ints(&db.exec("SELECT u.w, t.v FROM u JOIN t ON u.tid = t.id")),
        expected
    );
    assert_eq!(
        ints(&db.exec("SELECT u.w, t.v FROM u JOIN t ON t.id = u.tid")),
        expected
    );
}

#[test]
fn index_scan_and_index_maintenance_through_dml() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE INDEX t_id ON t (id)");
    assert!(
        db.exec("SELECT v FROM t WHERE id = 3").is_empty(),
        "empty index scans cleanly"
    );
    db.exec("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 3")), vec![[30]]);
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE 3 = id")), vec![[30]]);
    assert!(db.exec("SELECT v FROM t WHERE id = 7").is_empty());

    assert_eq!(
        ints(&db.exec("UPDATE t SET v = v + 1 WHERE id = 3")),
        vec![[1]]
    );
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 3")), vec![[31]]);

    assert_eq!(
        ints(&db.exec("UPDATE t SET id = 33 WHERE id = 3")),
        vec![[1]]
    );
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
    assert_eq!(
        ints(&db.exec("SELECT id, v FROM t WHERE v > 0")),
        vec![[1, 1]],
        "the rejected row is not left in the table"
    );
}

#[test]
fn update_rejected_by_a_unique_index_leaves_the_table_and_every_index_unchanged() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE INDEX t_id ON t (id)");
    db.exec("CREATE INDEX t_v ON t (v)");
    db.exec("INSERT INTO t VALUES (1, 10), (2, 20)");

    let err = db
        .try_exec("UPDATE t SET v = 99, id = 2 WHERE id = 1")
        .unwrap_err();
    assert!(err.contains("DuplicateKey"), "{err}");

    assert_eq!(
        sorted(ints(&db.exec("SELECT id, v FROM t WHERE id > id - 1"))),
        vec![vec![1, 10], vec![2, 20]],
        "the table keeps both rows unchanged"
    );
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 1")), vec![[10]]);
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 2")), vec![[20]]);
    assert_eq!(ints(&db.exec("SELECT id FROM t WHERE v = 10")), vec![[1]]);
    assert!(db.exec("SELECT id FROM t WHERE v = 99").is_empty());

    assert_eq!(
        ints(&db.exec("UPDATE t SET v = v + 1 WHERE id = 1")),
        vec![[1]],
        "an update that keeps its key is not a conflict with itself"
    );
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 1")), vec![[11]]);
    assert_eq!(ints(&db.exec("SELECT id FROM t WHERE v = 11")), vec![[1]]);
}

#[test]
fn index_lookups_with_a_constant_of_another_numeric_type_match_a_scan() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)");
    let scanned = ints(&db.exec("SELECT v FROM t WHERE id = 2.0"));
    assert_eq!(scanned, vec![[20]]);
    db.exec("CREATE INDEX t_id ON t (id)");
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE id = 2.0")), scanned);
    assert_eq!(ints(&db.exec("SELECT v FROM t WHERE 2.0 = id")), scanned);
    assert!(db.exec("SELECT v FROM t WHERE id = 2.5").is_empty());

    db.exec("CREATE TABLE f (x FLOAT, w INTEGER)");
    db.exec("INSERT INTO f VALUES (2.0, 200), (2.5, 250)");
    db.exec("CREATE INDEX f_x ON f (x)");
    assert_eq!(ints(&db.exec("SELECT w FROM f WHERE x = 2")), vec![[200]]);
    assert_eq!(
        ints(&db.exec("SELECT f.w, t.v FROM f JOIN t ON f.x = t.id")),
        vec![[200, 20]]
    );
    assert_eq!(
        ints(&db.exec("SELECT t.v, f.w FROM t JOIN f ON t.id = f.x")),
        vec![[20, 200]]
    );
}

#[test]
fn create_index_backfills_existing_rows() {
    let db = emp_db();
    db.exec("CREATE INDEX emp_id ON emp (id)");
    assert_eq!(
        ints(&db.exec("SELECT salary FROM emp WHERE id = 3")),
        vec![[300]]
    );
}

#[test]
fn update_and_delete_without_index() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("UPDATE emp SET salary = salary * 10 WHERE dept = 10")),
        vec![[2]]
    );
    assert_eq!(
        ints(&db.exec("SELECT salary FROM emp WHERE dept = 10 ORDER BY salary")),
        vec![[1000], [2000]]
    );
    assert_eq!(
        ints(&db.exec("UPDATE emp SET dept = 99, salary = 0 WHERE id = 4")),
        vec![[1]]
    );
    assert_eq!(
        ints(&db.exec("SELECT dept, salary FROM emp WHERE id = 4")),
        vec![[99, 0]]
    );
    assert_eq!(
        ints(&db.exec("DELETE FROM emp WHERE salary >= 1000")),
        vec![[2]]
    );
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp ORDER BY id")),
        vec![[3], [4]]
    );
    assert_eq!(ints(&db.exec("DELETE FROM emp WHERE id = 42")), vec![[0]]);
}

#[test]
fn many_rows_span_pages_and_survive_eviction() {
    const FRAMES_FEWER_THAN_TABLE_PAGES: usize = 4;
    let db = Db::new(FRAMES_FEWER_THAN_TABLE_PAGES);
    db.exec("CREATE TABLE big (id INTEGER, v INTEGER)");
    let values: Vec<String> = (0..2000)
        .rev()
        .map(|i| format!("({i}, {})", i % 7))
        .collect();
    assert_eq!(
        ints(&db.exec(&format!("INSERT INTO big VALUES {}", values.join(", ")))),
        vec![[2000]]
    );
    assert_eq!(
        ints(&db.exec("SELECT COUNT(*), SUM(id) FROM big")),
        vec![[2000, 1999 * 1000]]
    );
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
fn insert_values_naming_a_column_is_a_bind_error_and_inserts_nothing() {
    let db = Db::new(8);
    db.exec("CREATE TABLE t (a INTEGER)");
    assert_eq!(
        db.bind_error("INSERT INTO t VALUES (a)"),
        BinderError::ColumnNotFound { column: "a".into() },
        "a VALUES row has no input columns to read"
    );
    assert!(db.exec("SELECT * FROM t").is_empty());
    db.exec("INSERT INTO t VALUES (1 + 2)");
    assert_eq!(ints(&db.exec("SELECT a FROM t")), vec![vec![3]]);
}

#[test]
fn binder_and_planner_errors_are_reported() {
    let db = emp_db();
    db.exec("CREATE TABLE dept (id INTEGER, name INTEGER)");
    let unsupported = BinderError::Unsupported;
    let cases = [
        (
            "SELECT x FROM emp",
            BinderError::ColumnNotFound { column: "x".into() },
        ),
        (
            "SELECT * FROM nope",
            BinderError::TableNotFound {
                table: "nope".into(),
            },
        ),
        (
            "SELECT * FROM emp, dept",
            unsupported(Unsupported::CommaSeparatedFrom),
        ),
        ("DROP TABLE emp", unsupported(Unsupported::Statement)),
        (
            "SELECT id FROM emp JOIN dept ON emp.dept = dept.id",
            BinderError::AmbiguousColumn {
                column: "id".into(),
            },
        ),
        (
            "SELECT * FROM emp RIGHT JOIN dept ON emp.dept = dept.id",
            unsupported(Unsupported::JoinType),
        ),
        (
            "SELECT emp.id FROM emp JOIN dept ON emp.dept = dept.id JOIN dept d2 ON d2.id = emp.id",
            unsupported(Unsupported::MultiWayJoin),
        ),
        (
            "SELECT id FROM emp WHERE id LIKE 1",
            BinderError::UnsupportedExpression {
                expr: "id LIKE 1".into(),
            },
        ),
        (
            "SELECT id FROM emp WHERE id % 2 = 0",
            BinderError::UnsupportedOperator {
                operator: sqlparser::ast::BinaryOperator::Modulo,
            },
        ),
        (
            "SELECT 'text'",
            BinderError::UnsupportedLiteral {
                literal: "'text'".into(),
            },
        ),
        (
            "SELECT id FROM emp LIMIT id",
            unsupported(Unsupported::LimitClause),
        ),
        (
            "SELECT foo.id FROM emp",
            BinderError::TableNotInScope {
                table: "foo".into(),
            },
        ),
        (
            "SELECT a.b.c FROM emp",
            unsupported(Unsupported::CompoundIdentifier),
        ),
        (
            "CREATE TABLE s (name TEXT)",
            BinderError::UnsupportedType {
                type_name: "TEXT".into(),
            },
        ),
        (
            "SELECT COUNT(*) FROM emp GROUP BY ALL",
            unsupported(Unsupported::GroupByAll),
        ),
        (
            "UPDATE emp SET (id, dept) = (1, 2)",
            unsupported(Unsupported::TupleAssignment),
        ),
        (
            "SELECT * FROM emp ORDER BY ALL",
            BinderError::ColumnNotFound {
                column: "ALL".into(),
            },
        ),
        (
            "SELECT * FROM s.emp",
            BinderError::QualifiedName {
                name: "s.emp".into(),
            },
        ),
        (
            "SELECT CONCAT(id) FROM emp",
            BinderError::UnsupportedExpression {
                expr: "CONCAT(id)".into(),
            },
        ),
        (
            "SELECT id FROM (SELECT id FROM emp) sub",
            unsupported(Unsupported::FromTarget),
        ),
        (
            "INSERT INTO emp SELECT * FROM emp",
            unsupported(Unsupported::InsertSource),
        ),
        (
            "SELECT id FROM emp WHERE id IN (1, 2)",
            BinderError::UnsupportedExpression {
                expr: "id IN (1, 2)".into(),
            },
        ),
        (
            "SELECT emp.* FROM emp",
            unsupported(Unsupported::SelectItem),
        ),
        (
            "CREATE INDEX i ON emp (nope)",
            BinderError::ColumnNotFound {
                column: "nope".into(),
            },
        ),
    ];
    let failures: Vec<String> = cases
        .into_iter()
        .filter_map(|(sql, expected)| match db.query_error(sql) {
            QueryError::Bind(e) if e == expected => None,
            other => Some(format!("{sql}: expected {expected:?}, got {other:?}")),
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(matches!(db.query_error("SELEC 1"), QueryError::Parse(_)));
}

#[test]
fn creating_a_table_twice_fails_and_data_is_kept() {
    let db = emp_db();
    assert!(db.try_exec("CREATE TABLE emp (id INTEGER)").is_err());
    assert_eq!(ints(&db.exec("SELECT COUNT(*) FROM emp")), vec![[4]]);
    db.exec("CREATE INDEX i ON emp (id)");
    assert!(db.try_exec("CREATE INDEX i ON emp (id)").is_err());
}

#[test]
fn insert_rejects_rows_of_the_wrong_width() {
    let db = emp_db();
    assert!(db.try_exec("INSERT INTO emp VALUES (1, 2)").is_err());
}

#[test]
fn select_over_a_stored_tuple_shorter_than_its_schema_is_an_error_not_a_panic() {
    use cmudb::concurrency::{IsolationLevel, mvcc};
    use cmudb::table_heap::Tuple;

    let without_bitmap = vec![];
    let bitmap_without_payload = vec![0];
    for short in [without_bitmap, bitmap_without_payload] {
        let db = Db::new(16);
        db.exec("CREATE TABLE t (a INTEGER)");
        let table = db.catalog().get_table(db.table_oid("t")).expect("table");
        let txn = db.txn_mgr().begin(IsolationLevel::SnapshotIsolation);
        mvcc::insert_tuple(
            &txn,
            &table.heap,
            &[],
            &Tuple {
                data: short.clone(),
            },
        )
        .expect("the heap stores any bytes");
        db.txn_mgr().commit(&txn).expect("commit");
        let err = db
            .try_exec("SELECT * FROM t")
            .expect_err("decoding a short tuple fails the query");
        assert!(
            err.contains("the tuple does not match the table's schema"),
            "tuple {short:?} gave {err}"
        );
    }
}
