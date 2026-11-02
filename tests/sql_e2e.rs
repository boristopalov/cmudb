//! SQL end-to-end tests: each test runs SQL text through the parser, binder,
//! planner and executors against a real buffer pool and disk manager.
mod common;

use cmudb::QueryError;
use cmudb::catalog::{CatalogError, Value};
use cmudb::parser::{BinderError, Unsupported};
use cmudb::processor::executor::ExecutorError;
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
        ints(&db.exec("SELECT SUM(salary), MIN(salary), MAX(salary) FROM emp")),
        vec![[1000, 100, 400]]
    );
    assert_eq!(
        db.exec("SELECT AVG(salary) FROM emp"),
        vec![vec![Some(Value::float(250.0))]]
    );
    assert_eq!(
        ints(&db.exec("SELECT SUM(salary) + 1 FROM emp")),
        vec![[1001]]
    );
    assert_eq!(
        sorted(ints(&db.exec(
            "SELECT dept, COUNT(*), SUM(salary), MIN(salary), MAX(salary) \
             FROM emp GROUP BY dept"
        ))),
        vec![
            vec![10, 2, 300, 100, 200],
            vec![20, 1, 300, 300, 300],
            vec![30, 1, 400, 400, 400],
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
        (
            "SELECT id FROM emp ORDER BY id WITH FILL",
            Unsupported::OrderByWithFill,
        ),
        (
            "SELECT id FROM emp ORDER BY id INTERPOLATE",
            Unsupported::OrderByInterpolate,
        ),
    ] {
        assert_eq!(
            db.bind_error(sql),
            BinderError::Unsupported(expected),
            "{sql}"
        );
    }
}

#[test]
fn having_is_rejected_instead_of_returning_every_group() {
    let db = emp_db();
    for sql in [
        "SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 1",
        "SELECT COUNT(*) FROM emp HAVING COUNT(*) > 1",
        "SELECT id FROM emp HAVING id > 1",
    ] {
        let err = db.bind_error(sql);
        assert_eq!(err, BinderError::Unsupported(Unsupported::Having), "{sql}");
        assert_eq!(err.to_string(), "HAVING is not supported", "{sql}");
    }
}

#[test]
fn select_clauses_the_binder_does_not_bind_are_rejected() {
    let db = emp_db();
    for (sql, expected) in [
        ("SELECT TOP 1 id FROM emp", Unsupported::Top),
        ("SELECT id INTO emp2 FROM emp", Unsupported::SelectInto),
        ("SELECT id FROM emp PREWHERE id = 1", Unsupported::Prewhere),
        (
            "SELECT id FROM emp START WITH id = 1 CONNECT BY PRIOR id = dept",
            Unsupported::ConnectBy,
        ),
        ("SELECT id FROM emp CLUSTER BY id", Unsupported::ClusterBy),
        (
            "SELECT id FROM emp DISTRIBUTE BY id",
            Unsupported::DistributeBy,
        ),
        ("SELECT id FROM emp SORT BY id", Unsupported::SortBy),
        (
            "SELECT id FROM emp LATERAL VIEW explode(dept) x AS y",
            Unsupported::LateralView,
        ),
    ] {
        assert_eq!(
            db.bind_error(sql),
            BinderError::Unsupported(expected),
            "{sql}"
        );
    }
}

#[test]
fn wildcard_options_are_rejected_instead_of_returning_every_column() {
    let db = emp_db();
    for (sql, expected, message) in [
        (
            "SELECT * EXCLUDE (dept) FROM emp",
            Unsupported::WildcardExclude,
            "SELECT * EXCLUDE is not supported",
        ),
        (
            "SELECT * EXCLUDE dept FROM emp",
            Unsupported::WildcardExclude,
            "SELECT * EXCLUDE is not supported",
        ),
        (
            "SELECT * EXCEPT (dept) FROM emp",
            Unsupported::WildcardExcept,
            "SELECT * EXCEPT is not supported",
        ),
        (
            "SELECT * REPLACE (salary * 2 AS salary) FROM emp",
            Unsupported::WildcardReplace,
            "SELECT * REPLACE is not supported",
        ),
        (
            "SELECT * RENAME (dept AS department) FROM emp",
            Unsupported::WildcardRename,
            "SELECT * RENAME is not supported",
        ),
        (
            "SELECT * ILIKE 'd%' FROM emp",
            Unsupported::WildcardIlike,
            "SELECT * ILIKE is not supported",
        ),
    ] {
        let err = db.bind_error(sql);
        assert_eq!(err, BinderError::Unsupported(expected), "{sql}");
        assert_eq!(err.to_string(), message, "{sql}");
    }
    assert_eq!(
        db.exec("SELECT * FROM emp").len(),
        4,
        "a bare * still selects every row"
    );
}

#[test]
fn query_clauses_the_binder_does_not_bind_are_rejected() {
    let db = emp_db();
    for (sql, expected) in [
        ("WITH x AS (SELECT 1) SELECT id FROM emp", Unsupported::With),
        (
            "SELECT id FROM emp FETCH FIRST 1 ROWS ONLY",
            Unsupported::Fetch,
        ),
        ("SELECT id FROM emp FOR UPDATE", Unsupported::LockingClause),
        ("SELECT id FROM emp FOR SHARE", Unsupported::LockingClause),
        ("SELECT id FROM emp FOR JSON AUTO", Unsupported::ForClause),
        (
            "SELECT id FROM emp SETTINGS max_threads = 1",
            Unsupported::QuerySettings,
        ),
        ("SELECT id FROM emp FORMAT JSON", Unsupported::FormatClause),
        (
            "SELECT id FROM emp |> WHERE id = 1",
            Unsupported::PipeOperator,
        ),
        (
            "(SELECT id FROM emp) FETCH FIRST 1 ROWS ONLY",
            Unsupported::Fetch,
        ),
        (
            "(SELECT id FROM emp FOR UPDATE)",
            Unsupported::LockingClause,
        ),
        ("VALUES (1) FETCH FIRST 1 ROWS ONLY", Unsupported::Fetch),
    ] {
        let err = db.bind_error(sql);
        assert_eq!(err, BinderError::Unsupported(expected), "{sql}");
        assert_eq!(err.to_string(), expected.to_string(), "{sql}");
    }
}

#[test]
fn order_by_and_limit_outside_a_select_are_rejected_instead_of_ignored() {
    let db = emp_db();
    for (sql, expected) in [
        (
            "(SELECT id FROM emp) ORDER BY id DESC",
            Unsupported::OrderByWithoutSelect,
        ),
        (
            "(SELECT id FROM emp) LIMIT 1",
            Unsupported::LimitWithoutSelect,
        ),
        (
            "VALUES (2), (1) ORDER BY 1",
            Unsupported::OrderByWithoutSelect,
        ),
        ("VALUES (2), (1) LIMIT 1", Unsupported::LimitWithoutSelect),
    ] {
        assert_eq!(
            db.bind_error(sql),
            BinderError::Unsupported(expected),
            "{sql}"
        );
    }
}

#[test]
fn insert_source_clauses_the_binder_does_not_bind_are_rejected() {
    let db = emp_db();
    for (sql, expected) in [
        (
            "INSERT INTO emp VALUES (5, 10, 500, 5.0, true) LIMIT 1",
            Unsupported::LimitWithoutSelect,
        ),
        (
            "INSERT INTO emp VALUES (5, 10, 500, 5.0, true) ORDER BY 1",
            Unsupported::OrderByWithoutSelect,
        ),
        (
            "INSERT INTO emp VALUES (5, 10, 500, 5.0, true) FETCH FIRST 1 ROWS ONLY",
            Unsupported::Fetch,
        ),
        (
            "INSERT INTO emp SELECT * FROM emp LIMIT 1",
            Unsupported::InsertSource,
        ),
        (
            "INSERT INTO emp SELECT * FROM emp ORDER BY 1",
            Unsupported::InsertSource,
        ),
    ] {
        assert_eq!(
            db.bind_error(sql),
            BinderError::Unsupported(expected),
            "{sql}"
        );
    }
    assert_eq!(
        ints(&db.exec("SELECT COUNT(*) FROM emp")),
        vec![vec![4]],
        "a rejected INSERT adds no rows"
    );
}

#[test]
fn parenthesized_select_keeps_its_own_order_by_and_limit() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec("(SELECT id FROM emp ORDER BY id DESC LIMIT 2)")),
        vec![vec![4], vec![3]]
    );
}

#[test]
fn values_statement() {
    let db = Db::new(8);
    assert_eq!(
        ints(&db.exec("VALUES (1, 2), (3, 4)")),
        vec![[1, 2], [3, 4]]
    );
}

fn is_hash_join(plan: &Plan) -> bool {
    matches!(plan, Plan::HashJoin(_))
}

fn is_nested_loop_join(plan: &Plan) -> bool {
    matches!(plan, Plan::NestedLoopJoin(_))
}

fn sorted_nullable(mut rows: Vec<Vec<Option<i32>>>) -> Vec<Vec<Option<i32>>> {
    rows.sort();
    rows
}

#[test]
fn inner_and_left_equi_joins_without_index_use_hash_join() {
    let db = emp_db();
    db.exec("CREATE TABLE dept (id INTEGER, name INTEGER)");
    db.exec("INSERT INTO dept VALUES (10, 1000), (20, 2000), (40, 4000)");
    let inner = "SELECT emp.id, dept.name FROM emp JOIN dept ON emp.dept = dept.id";
    assert!(plan_contains(&db.plan_of(inner), is_hash_join));
    assert_eq!(
        sorted(ints(&db.exec(inner))),
        vec![[1, 1000], [2, 1000], [3, 2000]]
    );
    let left = "SELECT emp.id, dept.name FROM emp LEFT JOIN dept ON emp.dept = dept.id";
    assert!(plan_contains(&db.plan_of(left), is_hash_join));
    assert_eq!(
        sorted_nullable(nullable_ints(&db.exec(left))),
        vec![
            vec![Some(1), Some(1000)],
            vec![Some(2), Some(1000)],
            vec![Some(3), Some(2000)],
            vec![Some(4), None],
        ],
        "unmatched left row is NULL-padded"
    );
    assert_eq!(
        ints(&db.exec(
            "SELECT COUNT(*) FROM emp JOIN dept ON emp.dept = dept.id WHERE dept.name > 1000"
        )),
        vec![[1]]
    );
}

#[test]
fn joins_without_a_column_equality_across_sides_use_nested_loop_join() {
    let db = emp_db();
    db.exec("CREATE TABLE dept (id INTEGER, name INTEGER)");
    db.exec("INSERT INTO dept VALUES (10, 1000), (20, 2000), (40, 4000)");
    let less_than =
        "SELECT emp.id, dept.id FROM emp JOIN dept ON emp.dept < dept.id WHERE emp.id = 4";
    assert!(plan_contains(&db.plan_of(less_than), is_nested_loop_join));
    assert_eq!(ints(&db.exec(less_than)), vec![[4, 40]]);
    for sql in [
        "SELECT emp.id FROM emp JOIN dept ON emp.dept = dept.id OR emp.id = 1",
        "SELECT emp.id FROM emp JOIN dept ON emp.dept = emp.id",
        "SELECT emp.id FROM emp JOIN dept ON emp.bonus = dept.id",
    ] {
        assert!(
            plan_contains(&db.plan_of(sql), is_nested_loop_join),
            "{sql}: no same-type left.col = right.col conjunct to hash on"
        );
    }
}

/// Two tables whose join columns hold NULLs, duplicate keys on both sides and keys with no
/// partner on either side, spread over enough rows to fill several hash partitions.
fn hash_join_tables() -> Db {
    let db = Db::new(256);
    db.exec("CREATE TABLE l (k INTEGER, k2 INTEGER, a INTEGER)");
    db.exec("CREATE TABLE r (k INTEGER, k2 INTEGER, b INTEGER)");
    let key = |i: i32| match i % 11 {
        0 => "NULL".to_string(),
        m => (i % 37 + m % 2).to_string(),
    };
    let l_rows: Vec<String> = (0..300)
        .map(|i| format!("({}, {}, {i})", key(i), i % 3))
        .collect();
    let r_rows: Vec<String> = (0..200)
        .map(|i| format!("({}, {}, {})", key(i + 5), i % 4, 1000 + i))
        .collect();
    db.exec(&format!("INSERT INTO l VALUES {}", l_rows.join(", ")));
    db.exec(&format!("INSERT INTO r VALUES {}", r_rows.join(", ")));
    db
}

#[test]
fn hash_joins_return_the_nested_loop_join_rows() {
    let db = hash_join_tables();
    for join in ["JOIN", "LEFT JOIN"] {
        for on in [
            "l.k = r.k",
            "r.k = l.k",
            "l.k = r.k AND l.k2 = r.k2",
            "l.k = r.k AND l.a < r.b - 1050",
            "l.k = r.k AND l.k2 = 1",
            "l.k = r.k AND r.k2 = r.k2 AND l.a > 100",
        ] {
            for where_clause in ["", " WHERE l.a > 50"] {
                let sql = format!("SELECT l.a, r.b FROM l {join} r ON {on}{where_clause}");
                let hashed = db.plan_of(&sql);
                assert!(plan_contains(&hashed, is_hash_join), "{sql}");
                let nested = with_nested_loop_joins(db.plan_of(&sql));
                assert!(plan_contains(&nested, is_nested_loop_join), "{sql}");
                let expected = sorted_nullable(nullable_ints(&db.run_plan(&nested).unwrap()));
                assert!(
                    !expected.is_empty(),
                    "{sql}: the reference join returns rows"
                );
                assert_eq!(
                    sorted_nullable(nullable_ints(&db.run_plan(&hashed).unwrap())),
                    expected,
                    "{sql}"
                );
            }
        }
    }
}

#[test]
fn a_hash_join_query_frees_its_spill_pages_and_a_rerun_reuses_them() {
    let db = hash_join_tables();
    let sql = "SELECT l.a, r.b FROM l JOIN r ON l.k = r.k";
    assert!(plan_contains(&db.plan_of(sql), is_hash_join));
    let live_before = db.bpm().live_page_count();
    db.exec(sql);
    assert_eq!(
        db.bpm().live_page_count(),
        live_before,
        "the join leaves no spill page allocated"
    );
    let after_first_run = db.bpm().allocated_page_count();
    db.exec(sql);
    assert_eq!(
        db.bpm().allocated_page_count(),
        after_first_run,
        "a second run of the same join allocates no new page ids"
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
    let err = db.query_error("INSERT INTO t VALUES (1, 2)");
    assert!(is_duplicate_key(&err, "t_id"), "{err:?}");
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

    let err = db.query_error("UPDATE t SET v = 99, id = 2 WHERE id = 1");
    assert!(is_duplicate_key(&err, "t_id"), "{err:?}");

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
fn binder_errors_are_reported() {
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
        (
            "DROP VIEW emp",
            BinderError::UnsupportedDropObject {
                object_type: "VIEW".into(),
            },
        ),
        (
            "DROP TABLE emp, dept",
            unsupported(Unsupported::DropManyObjects),
        ),
        (
            "DROP TABLE emp CASCADE",
            unsupported(Unsupported::DropCascade),
        ),
        (
            "DROP TABLE emp ON dept",
            unsupported(Unsupported::DropTableOn),
        ),
        (
            "DROP TABLE nope",
            BinderError::TableNotFound {
                table: "nope".into(),
            },
        ),
        (
            "DROP INDEX nope",
            BinderError::IndexNotFound {
                index: "nope".into(),
            },
        ),
        (
            "DROP INDEX nope ON nowhere",
            BinderError::TableNotFound {
                table: "nowhere".into(),
            },
        ),
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
            "SELECT $$text$$",
            BinderError::UnsupportedLiteral {
                literal: "$$text$$".into(),
            },
        ),
        (
            "SELECT id FROM emp LIMIT id",
            unsupported(Unsupported::LimitExpression),
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
            "CREATE TABLE s (name CHAR(3))",
            BinderError::UnsupportedType {
                type_name: "CHAR(3)".into(),
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
        let err = db.exec_error("SELECT * FROM t");
        assert!(
            matches!(
                err,
                ExecutorError::Catalog(CatalogError::TupleSchemaMismatch)
                    | ExecutorError::ColumnRead {
                        source: CatalogError::TupleSchemaMismatch,
                        ..
                    }
            ),
            "decoding the short tuple {short:?} fails the query: {err:?}"
        );
    }
}

fn scans_through_an_index(plan: &Plan) -> bool {
    match plan {
        Plan::IndexScan(_) => true,
        Plan::Projection(p) => scans_through_an_index(&p.child),
        Plan::Filter(p) => scans_through_an_index(&p.child),
        _ => false,
    }
}

#[test]
fn drop_table_removes_the_table_and_its_indexes() {
    let db = emp_db();
    db.exec("CREATE INDEX emp_id ON emp (id)");
    db.exec("DROP TABLE emp");
    let err = db
        .try_exec("SELECT * FROM emp")
        .expect_err("the table is gone");
    assert!(err.contains("table not found: emp"), "{err}");
    let err = db.try_exec("DROP TABLE emp").expect_err("dropped already");
    assert!(err.contains("table not found: emp"), "{err}");
    let err = db
        .try_exec("DROP INDEX emp_id")
        .expect_err("dropped with emp");
    assert!(err.contains("index not found: emp_id"), "{err}");

    db.exec("CREATE TABLE emp (id INTEGER)");
    db.exec("CREATE INDEX emp_id ON emp (id)");
    db.exec("INSERT INTO emp VALUES (7)");
    assert_eq!(
        ints(&db.exec("SELECT id FROM emp WHERE id = 7")),
        vec![[7]],
        "a table and index can reuse the dropped names"
    );
}

#[test]
fn drop_index_makes_queries_fall_back_to_a_scan() {
    let db = emp_db();
    db.exec("CREATE INDEX emp_id ON emp (id)");
    let lookup = "SELECT salary FROM emp WHERE id = 3";
    assert!(scans_through_an_index(&db.plan_of(lookup)));

    db.exec("DROP INDEX emp_id");
    assert!(
        !scans_through_an_index(&db.plan_of(lookup)),
        "the planner no longer picks the dropped index"
    );
    assert_eq!(ints(&db.exec(lookup)), vec![[300]]);
    assert!(db.catalog().index_oid("emp", "emp_id").is_err());
}

#[test]
fn drop_index_on_a_table_picks_that_tables_index() {
    let db = emp_db();
    db.exec("CREATE TABLE dept (id INTEGER)");
    db.exec("CREATE INDEX by_id ON emp (id)");
    db.exec("CREATE INDEX by_id ON dept (id)");

    let err = db
        .try_exec("DROP INDEX by_id")
        .expect_err("two tables have an index called by_id");
    assert!(err.contains("ambiguous"), "{err}");

    db.exec("DROP INDEX by_id ON dept");
    assert!(db.catalog().index_oid("dept", "by_id").is_err());
    assert!(db.catalog().index_oid("emp", "by_id").is_ok());
    db.exec("DROP INDEX by_id");
    assert!(db.catalog().index_oid("emp", "by_id").is_err());
}

#[test]
fn drop_if_exists_skips_a_missing_table_or_index() {
    let db = emp_db();
    for sql in [
        "DROP TABLE IF EXISTS nope",
        "DROP INDEX IF EXISTS nope",
        "DROP INDEX IF EXISTS nope ON emp",
        "DROP INDEX IF EXISTS nope ON nowhere",
    ] {
        let results = db.execute(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert_eq!(results.len(), 1, "{sql}");
    }

    db.exec("CREATE INDEX emp_id ON emp (id)");
    db.exec("DROP INDEX IF EXISTS emp_id");
    assert!(db.catalog().index_oid("emp", "emp_id").is_err());
    db.exec("DROP TABLE IF EXISTS emp");
    assert!(db.catalog().table_oid("emp").is_err());
}

/// One column type for the per-type matrix: its SQL name and distinct values in ascending order,
/// each as SQL text and as the value it binds to.
struct TypeCase {
    sql_type: &'static str,
    values: Vec<(&'static str, Value)>,
    indexable: bool,
}

/// Distinct strings in ascending (byte) order, as SQL literals and as values.
fn text_values() -> Vec<(&'static str, Value)> {
    vec![
        ("''", Value::text("")),
        ("'Zed'", Value::text("Zed")),
        ("'apple'", Value::text("apple")),
        ("'it''s naïve'", Value::text("it's naïve")),
    ]
}

fn type_cases() -> Vec<TypeCase> {
    let ts = |text: &str| Value::TIMESTAMP(cmudb::catalog::timestamp::parse(text).unwrap());
    vec![
        TypeCase {
            sql_type: "INT",
            values: vec![
                ("-7", Value::INT(-7)),
                ("0", Value::INT(0)),
                ("2147483647", Value::INT(i32::MAX)),
            ],
            indexable: true,
        },
        TypeCase {
            sql_type: "BIGINT",
            values: vec![
                ("-5000000000", Value::BIGINT(-5_000_000_000)),
                ("1", Value::BIGINT(1)),
                ("9000000000", Value::BIGINT(9_000_000_000)),
            ],
            indexable: true,
        },
        TypeCase {
            sql_type: "FLOAT",
            values: vec![
                ("-1.5", Value::float(-1.5)),
                ("2", Value::float(2.0)),
                ("16777217.0", Value::float(16_777_217.0)),
            ],
            indexable: true,
        },
        TypeCase {
            sql_type: "BOOLEAN",
            values: vec![
                ("false", Value::BOOLEAN(false)),
                ("true", Value::BOOLEAN(true)),
            ],
            indexable: true,
        },
        TypeCase {
            sql_type: "TIMESTAMP",
            values: vec![
                ("TIMESTAMP '1999-12-31 23:59:59'", ts("1999-12-31 23:59:59")),
                ("'2024-01-01'", ts("2024-01-01")),
                (
                    "TIMESTAMP '2024-02-29 12:00:00.5'",
                    ts("2024-02-29 12:00:00.5"),
                ),
            ],
            indexable: true,
        },
        TypeCase {
            sql_type: "TEXT",
            values: text_values(),
            indexable: false,
        },
        TypeCase {
            sql_type: "VARCHAR(12)",
            values: text_values(),
            indexable: true,
        },
    ]
}

fn single_values(rows: Vec<Row>) -> Vec<Option<Value>> {
    rows.into_iter().map(|r| r[0].clone()).collect()
}

#[test]
fn every_column_type_inserts_compares_sorts_groups_aggregates_and_updates() {
    for case in type_cases() {
        let ty = case.sql_type;
        let values = &case.values;
        let (lo_sql, lo) = values[0].clone();
        let (hi_sql, hi) = values[values.len() - 1].clone();
        let db = Db::new(64);
        db.exec(&format!("CREATE TABLE t (k INT, v {ty})"));
        for (k, (sql, _)) in values.iter().enumerate().rev() {
            db.exec(&format!("INSERT INTO t VALUES ({k}, {sql})"));
        }
        db.exec(&format!("INSERT INTO t VALUES (10, {lo_sql}), (11, NULL)"));

        assert_eq!(
            sorted(ints(
                &db.exec(&format!("SELECT k FROM t WHERE v = {lo_sql}"))
            )),
            vec![vec![0], vec![10]],
            "{ty}: WHERE ="
        );
        assert_eq!(
            ints(&db.exec(&format!("SELECT COUNT(*) FROM t WHERE v > {lo_sql}"))),
            vec![[values.len() as i32 - 1]],
            "{ty}: WHERE >"
        );
        let mut expected_order: Vec<Option<Value>> = vec![Some(lo.clone())];
        expected_order.extend(values.iter().map(|(_, v)| Some(v.clone())));
        expected_order.push(None);
        assert_eq!(
            single_values(db.exec("SELECT v FROM t ORDER BY v")),
            expected_order,
            "{ty}: ORDER BY, NULLs last"
        );
        let mut groups = db.exec("SELECT v, COUNT(*) FROM t GROUP BY v");
        groups.sort();
        let mut expected_groups: Vec<Row> = values
            .iter()
            .map(|(_, v)| vec![Some(v.clone()), int(if *v == lo { 2 } else { 1 })])
            .collect();
        expected_groups.push(vec![None, int(1)]);
        expected_groups.sort();
        assert_eq!(groups, expected_groups, "{ty}: GROUP BY");
        assert_eq!(
            db.exec("SELECT MIN(v), MAX(v), COUNT(v), COUNT(*) FROM t"),
            vec![vec![
                Some(lo),
                Some(hi),
                int(values.len() as i32 + 1),
                int(values.len() as i32 + 2)
            ]],
            "{ty}: MIN, MAX and COUNT skip NULL"
        );

        db.exec(&format!("UPDATE t SET v = {hi_sql} WHERE v = {lo_sql}"));
        assert_eq!(
            ints(&db.exec(&format!("SELECT COUNT(*) FROM t WHERE v = {hi_sql}"))),
            vec![[3]],
            "{ty}: UPDATE"
        );
        db.exec("UPDATE t SET v = NULL WHERE k = 11");
        assert_eq!(
            ints(&db.exec("SELECT COUNT(v) FROM t")),
            vec![[values.len() as i32 + 1]],
            "{ty}: UPDATE to NULL"
        );
    }
}

#[test]
fn every_column_type_can_be_indexed_and_looked_up() {
    for case in type_cases().into_iter().filter(|c| c.indexable) {
        let ty = case.sql_type;
        let values = &case.values;
        let db = Db::new(64);
        db.exec(&format!("CREATE TABLE t (k INT, v {ty})"));
        for (k, (sql, _)) in values.iter().enumerate() {
            db.exec(&format!("INSERT INTO t VALUES ({k}, {sql})"));
        }
        db.exec("INSERT INTO t VALUES (99, NULL)");
        db.exec("CREATE INDEX t_v ON t (v)");
        for (k, (sql, _)) in values.iter().enumerate() {
            let query = format!("SELECT k FROM t WHERE v = {sql}");
            assert!(
                plan_contains(&db.plan_of(&query), |p| matches!(p, Plan::IndexScan(_))),
                "{ty}: {query} uses the index"
            );
            assert_eq!(ints(&db.exec(&query)), vec![[k as i32]], "{ty}: {query}");
        }
        let (first_sql, _) = values[0].clone();
        assert!(
            db.try_exec(&format!("INSERT INTO t VALUES (7, {first_sql})"))
                .is_err(),
            "{ty}: the index keeps keys unique"
        );
    }
}

#[test]
fn integer_literals_widen_into_bigint_and_float_columns() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (i INT, b BIGINT, f FLOAT)");
    db.exec("INSERT INTO t VALUES (1, 1, 1), (2, 2147483648, 2.5), (-2147483648, -3, -4)");
    db.exec("INSERT INTO t (f, i) VALUES (16777217, 5)");
    db.exec("INSERT INTO t (i, b) VALUES (6, -9223372036854775808)");
    assert_eq!(
        db.exec("SELECT b FROM t WHERE i = 6"),
        vec![vec![Some(Value::BIGINT(i64::MIN))]]
    );
    assert_eq!(
        db.exec("SELECT * FROM t WHERE i = 5"),
        vec![vec![int(5), None, Some(Value::float(16_777_217.0))]]
    );
    assert_eq!(
        sorted(ints(&db.exec("SELECT i FROM t WHERE f = 1 OR f = 2.5"))),
        vec![vec![1], vec![2]]
    );
    assert_eq!(
        ints(&db.exec("SELECT i FROM t WHERE b = 2147483648")),
        vec![[2]]
    );
    assert_eq!(ints(&db.exec("SELECT i FROM t WHERE i = 1.0")), vec![[1]]);
    assert!(db.exec("SELECT i FROM t WHERE i = 1.5").is_empty());

    db.exec("UPDATE t SET f = i + 1, b = i WHERE i = 1");
    assert_eq!(
        db.exec("SELECT b, f FROM t WHERE i = 1"),
        vec![vec![Some(Value::BIGINT(1)), Some(Value::float(2.0))]]
    );
    db.exec("UPDATE t SET i = b WHERE i = -2147483648");
    assert_eq!(ints(&db.exec("SELECT i FROM t WHERE b = -3")), vec![[-3]]);
    assert_eq!(
        db.exec("SELECT i + b, b * 2, f / 2, -i, -f FROM t WHERE i = 2"),
        vec![vec![
            Some(Value::BIGINT(2_147_483_650)),
            Some(Value::BIGINT(4_294_967_296)),
            Some(Value::float(1.25)),
            int(-2),
            Some(Value::float(-2.5)),
        ]]
    );

    db.exec("CREATE INDEX t_f ON t (f)");
    let query = "SELECT i FROM t WHERE f = 2";
    assert!(plan_contains(&db.plan_of(query), |p| matches!(
        p,
        Plan::IndexScan(_)
    )));
    assert_eq!(ints(&db.exec(query)), vec![[1]]);
}

#[test]
fn sum_of_ints_is_a_bigint_and_avg_is_a_float() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (i INT, b BIGINT)");
    db.exec("INSERT INTO t VALUES (2147483647, 9223372036854775807), (2, 1)");
    assert_eq!(
        db.exec("SELECT SUM(i), AVG(i) FROM t"),
        vec![vec![
            Some(Value::BIGINT(2_147_483_649)),
            Some(Value::float(1_073_741_824.5)),
        ]]
    );
    assert!(matches!(
        db.exec_error("SELECT SUM(b) FROM t"),
        ExecutorError::IntegerOverflow { .. }
    ));
    assert!(matches!(
        db.exec_error("SELECT i + 1 FROM t"),
        ExecutorError::IntegerOverflow { .. }
    ));
}

#[test]
fn timestamps_compare_with_string_literals_and_print_readably() {
    let db = Db::new(64);
    db.exec("CREATE TABLE ev (id INT, at TIMESTAMP)");
    db.exec(
        "INSERT INTO ev VALUES (1, '2024-01-01 09:30:00'), (2, TIMESTAMP '2024-06-15'), \
         (3, '1970-01-01T00:00:00.000001')",
    );
    assert_eq!(
        sorted(ints(&db.exec(
            "SELECT id FROM ev WHERE at >= '2024-01-01' AND '2024-12-31' > at"
        ))),
        vec![vec![1], vec![2]]
    );
    let printed: Vec<String> = db
        .exec("SELECT at FROM ev ORDER BY at")
        .into_iter()
        .map(|r| r[0].as_ref().expect("not NULL").to_string())
        .collect();
    assert_eq!(
        printed,
        [
            "1970-01-01 00:00:00.000001",
            "2024-01-01 09:30:00",
            "2024-06-15 00:00:00"
        ]
    );
}

#[test]
fn type_errors_name_the_column_and_both_types() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (i INT, f FLOAT, ok BOOLEAN, at TIMESTAMP)");
    let cases = [
        (
            "INSERT INTO t VALUES (2147483648, 1, true, NULL)",
            "2147483648 is out of range for INT column `i`",
        ),
        (
            "INSERT INTO t VALUES (1, true, true, NULL)",
            "column `f` is FLOAT, got BOOLEAN",
        ),
        (
            "INSERT INTO t VALUES (1.5, 1, true, NULL)",
            "column `i` is INT, got FLOAT",
        ),
        (
            "INSERT INTO t VALUES (1, 1, true, 1700000000)",
            "column `at` is TIMESTAMP, got INT",
        ),
        (
            "INSERT INTO t VALUES (1, 1, true, '2024-13-01')",
            "invalid TIMESTAMP '2024-13-01': expected YYYY-MM-DD[ HH:MM:SS[.ffffff]]",
        ),
        (
            "INSERT INTO t VALUES (1, 1, true)",
            "INSERT into `t` has 3 values for 4 columns",
        ),
        (
            "INSERT INTO t (i, i) VALUES (1, 2)",
            "column `i` is listed more than once in INSERT",
        ),
        ("UPDATE t SET ok = 1", "column `ok` is BOOLEAN, got INT"),
        ("UPDATE t SET i = f", "column `i` is INT, got FLOAT"),
        (
            "SELECT i FROM t WHERE at > 5",
            "operator `>` does not take TIMESTAMP and INT",
        ),
        (
            "SELECT i + ok FROM t",
            "operator `+` does not take INT and BOOLEAN",
        ),
        (
            "SELECT i FROM t WHERE i AND ok",
            "operator `AND` does not take INT and BOOLEAN",
        ),
        ("SELECT -ok FROM t", "cannot negate a BOOLEAN"),
        (
            "SELECT SUM(ok) FROM t",
            "aggregate `SUM` does not take a BOOLEAN argument",
        ),
        (
            "SELECT AVG(at) FROM t",
            "aggregate `AVG` does not take a TIMESTAMP argument",
        ),
        (
            "SELECT i FROM t WHERE i = 'one'",
            "operator `=` does not take INT and TEXT",
        ),
        (
            "CREATE TABLE z (at TIMESTAMP WITH TIME ZONE)",
            "unsupported type: TIMESTAMP WITH TIME ZONE",
        ),
    ];
    let failures: Vec<String> = cases
        .into_iter()
        .filter_map(|(sql, expected)| match db.query_error(sql) {
            QueryError::Bind(e) if e.to_string() == expected => None,
            other => Some(format!("{sql}: expected {expected:?}, got {other}")),
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        db.exec("SELECT * FROM t").is_empty(),
        "nothing was inserted"
    );
}

#[test]
fn values_columns_widen_to_the_common_numeric_type() {
    let db = Db::new(16);
    assert_eq!(
        db.exec("VALUES (1, 1), (2.5, 3000000000)"),
        vec![
            vec![Some(Value::float(1.0)), Some(Value::BIGINT(1))],
            vec![Some(Value::float(2.5)), Some(Value::BIGINT(3_000_000_000))],
        ]
    );
}

fn distinct_db() -> Db {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (a INTEGER, b INTEGER)");
    db.exec(
        "INSERT INTO t VALUES \
         (1, 10), (1, 10), (1, 20), (2, 10), (2, 10), \
         (NULL, 10), (NULL, 10), (NULL, NULL), (NULL, NULL), (3, NULL)",
    );
    db
}

#[test]
fn select_distinct_removes_duplicate_rows_over_one_or_more_columns() {
    let db = distinct_db();
    assert_eq!(
        sorted_nullable(nullable_ints(&db.exec("SELECT DISTINCT a FROM t"))),
        vec![vec![None], vec![Some(1)], vec![Some(2)], vec![Some(3)]]
    );
    assert_eq!(
        sorted_nullable(nullable_ints(&db.exec("SELECT DISTINCT a, b FROM t"))),
        vec![
            vec![None, None],
            vec![None, Some(10)],
            vec![Some(1), Some(10)],
            vec![Some(1), Some(20)],
            vec![Some(2), Some(10)],
            vec![Some(3), None],
        ]
    );
    assert_eq!(
        sorted_nullable(nullable_ints(&db.exec("SELECT DISTINCT * FROM t"))),
        sorted_nullable(nullable_ints(&db.exec("SELECT DISTINCT a, b FROM t")))
    );
}

#[test]
fn select_distinct_keeps_every_null_in_one_group() {
    let db = emp_db();
    db.exec("INSERT INTO emp VALUES (5, 10, 500, 5.0, NULL), (6, 10, 600, 6.0, true)");
    let mut active: Vec<Option<bool>> = db
        .exec("SELECT DISTINCT active FROM emp")
        .into_iter()
        .map(|row| match row.as_slice() {
            [Some(Value::BOOLEAN(b))] => Some(*b),
            [None] => None,
            other => panic!("expected one BOOLEAN or NULL column, got {other:?}"),
        })
        .collect();
    active.sort();
    assert_eq!(active, vec![None, Some(false), Some(true)]);
}

#[test]
fn select_distinct_sorts_and_limits_the_distinct_rows() {
    let db = distinct_db();
    assert_eq!(
        nullable_ints(&db.exec("SELECT DISTINCT a FROM t ORDER BY a")),
        vec![vec![Some(1)], vec![Some(2)], vec![Some(3)], vec![None]]
    );
    assert_eq!(
        nullable_ints(&db.exec("SELECT DISTINCT a FROM t ORDER BY a DESC LIMIT 3")),
        vec![vec![None], vec![Some(3)], vec![Some(2)]]
    );
    assert_eq!(
        nullable_ints(&db.exec("SELECT DISTINCT b, a FROM t ORDER BY a NULLS FIRST, b LIMIT 4")),
        vec![
            vec![Some(10), None],
            vec![None, None],
            vec![Some(10), Some(1)],
            vec![Some(20), Some(1)],
        ]
    );
    assert_eq!(
        nullable_ints(&db.exec("SELECT DISTINCT a FROM t ORDER BY a LIMIT 0")),
        Vec::<Vec<Option<i32>>>::new()
    );
}

#[test]
fn select_distinct_applies_to_computed_and_aggregated_columns() {
    let db = distinct_db();
    assert_eq!(
        ints(&db.exec("SELECT DISTINCT b * 0 + 1 FROM t WHERE b > 0")),
        vec![[1]]
    );
    let emp = emp_db();
    assert_eq!(
        ints(&emp.exec("SELECT DISTINCT COUNT(*) FROM emp GROUP BY dept ORDER BY COUNT(*)")),
        vec![[1], [2]]
    );
}

#[test]
fn select_all_keeps_duplicate_rows() {
    let db = distinct_db();
    assert_eq!(db.exec("SELECT ALL a FROM t").len(), 10);
}

#[test]
fn select_distinct_plans_an_aggregation_between_the_projection_and_the_sort() {
    let db = distinct_db();
    let Plan::Limit(limit) = db.plan_of("SELECT DISTINCT a FROM t ORDER BY a LIMIT 2") else {
        panic!("expected LIMIT at the top");
    };
    let Plan::ExternalMergeSort(sort) = *limit.child else {
        panic!("expected the sort under LIMIT");
    };
    let Plan::Aggregation(distinct) = *sort.child else {
        panic!("expected DISTINCT's aggregation under the sort");
    };
    assert!(
        distinct.aggregates.is_empty() && distinct.group_bys.len() == 1,
        "DISTINCT groups on its one projected column and computes no aggregates"
    );
    assert!(
        matches!(*distinct.child, Plan::Projection(_)),
        "DISTINCT reads the projection's output"
    );
}

#[test]
fn select_distinct_rejects_distinct_on_and_order_by_keys_outside_the_select_list() {
    let db = distinct_db();
    assert_eq!(
        db.bind_error("SELECT DISTINCT ON (a) a, b FROM t"),
        BinderError::Unsupported(Unsupported::DistinctOn)
    );
    for sql in [
        "SELECT DISTINCT a FROM t ORDER BY b",
        "SELECT DISTINCT a FROM t ORDER BY a + 1",
    ] {
        assert_eq!(
            db.bind_error(sql),
            BinderError::DistinctOrderByNotInSelectList,
            "{sql}"
        );
    }
    assert!(
        db.query_error("SELECT DISTINCT ON (a) a FROM t")
            .to_string()
            .contains("DISTINCT ON"),
        "the error names DISTINCT ON"
    );
}

#[test]
fn identical_aggregate_calls_read_one_aggregation_column() {
    let db = emp_db();
    let Plan::Projection(projection) =
        db.plan_of("SELECT COUNT(*), COUNT(*) + 1, SUM(salary) FROM emp")
    else {
        panic!("expected a projection at the top");
    };
    let Plan::Aggregation(aggregation) = *projection.child else {
        panic!("expected an aggregation under the projection");
    };
    assert_eq!(
        aggregation.aggregates.len(),
        2,
        "COUNT(*) is computed once and SUM once"
    );
    assert_eq!(
        ints(&db.exec("SELECT COUNT(*), COUNT(*) + 1, SUM(salary) FROM emp")),
        vec![[4, 5, 1000]]
    );
    assert_eq!(
        ints(&db.exec(
            "SELECT dept, COUNT(*), COUNT(*) * 10 FROM emp GROUP BY dept ORDER BY COUNT(*) DESC, dept"
        )),
        vec![[10, 2, 20], [20, 1, 10], [30, 1, 10]]
    );
}

#[test]
fn select_distinct_removes_duplicate_window_function_rows() {
    let db = emp_db();
    assert_eq!(
        ints(&db.exec(
            "SELECT DISTINCT dept, COUNT(*) OVER (PARTITION BY dept) FROM emp ORDER BY dept"
        )),
        vec![[10, 2], [20, 1], [30, 1]]
    );
    assert_eq!(
        ints(&db.exec(
            "SELECT DISTINCT dept, COUNT(*) OVER (PARTITION BY dept) FROM emp \
             ORDER BY COUNT(*) OVER (PARTITION BY dept) DESC, dept"
        )),
        vec![[10, 2], [20, 1], [30, 1]]
    );
}

#[test]
fn identical_window_calls_read_one_window_column() {
    let db = emp_db();
    let sql = "SELECT id, RANK() OVER (ORDER BY salary DESC), RANK() OVER (ORDER BY salary DESC) * 10 \
               FROM emp ORDER BY RANK() OVER (ORDER BY salary DESC)";
    assert_eq!(
        ints(&db.exec(sql)),
        vec![[4, 1, 10], [3, 2, 20], [2, 3, 30], [1, 4, 40]]
    );
    assert!(
        plan_contains(&db.plan_of(sql), |plan| matches!(
            plan,
            Plan::WindowFunction(w) if w.aggregates.len() == 1
        )),
        "RANK() over the same window is computed once"
    );
}

fn texts(rows: &[Row]) -> Vec<Vec<Option<String>>> {
    rows.iter()
        .map(|r| {
            r.iter()
                .map(|v| match v {
                    None => None,
                    Some(Value::TEXT(s)) => Some(s.to_string()),
                    Some(other) => panic!("expected TEXT, got {other:?}"),
                })
                .collect()
        })
        .collect()
}

fn some(s: &str) -> Option<String> {
    Some(s.to_string())
}

fn people_db() -> Db {
    let db = Db::new(64);
    db.exec("CREATE TABLE people (id INT, name VARCHAR(10), team TEXT)");
    db.exec("CREATE TABLE teams (name TEXT, floor INT)");
    db.exec(
        "INSERT INTO people VALUES (1, 'ann', 'db'), (2, 'bo', 'web'), (3, 'cy', 'db'), \
         (4, 'dee', NULL), (5, 'O''Hara', 'ops')",
    );
    db.exec("INSERT INTO teams VALUES ('db', 3), ('web', 5), ('hr', 1)");
    db
}

#[test]
fn string_literals_filter_text_columns_in_where_insert_and_update() {
    let db = people_db();
    for (filter, expected) in [
        ("name = 'bo'", vec![vec![2]]),
        ("name <> 'bo'", vec![vec![1], vec![3], vec![4], vec![5]]),
        ("name < 'bo'", vec![vec![1], vec![5]]),
        ("name >= 'cy'", vec![vec![3], vec![4]]),
        ("'db' = team", vec![vec![1], vec![3]]),
        ("team > 'a' AND team < 'p'", vec![vec![1], vec![3], vec![5]]),
        ("name = 'O''Hara'", vec![vec![5]]),
        ("name = 'nobody at all'", vec![]),
    ] {
        assert_eq!(
            sorted(ints(
                &db.exec(&format!("SELECT id FROM people WHERE {filter}"))
            )),
            expected,
            "WHERE {filter}"
        );
    }

    db.exec("UPDATE people SET team = 'platform', name = 'ann b' WHERE name = 'ann'");
    db.exec("UPDATE people SET team = NULL WHERE team = 'web'");
    db.exec("UPDATE people SET team = 'x' WHERE id = 4");
    assert_eq!(
        texts(&db.exec("SELECT name, team FROM people ORDER BY id")),
        vec![
            vec![some("ann b"), some("platform")],
            vec![some("bo"), None],
            vec![some("cy"), some("db")],
            vec![some("dee"), some("x")],
            vec![some("O'Hara"), some("ops")],
        ]
    );
    assert_eq!(
        texts(&db.exec("SELECT 'it''s', '', 'ünï'")),
        vec![vec![some("it's"), some(""), some("ünï")]]
    );
}

#[test]
fn text_columns_sort_group_aggregate_and_join() {
    let db = people_db();
    assert_eq!(
        texts(&db.exec("SELECT name FROM people ORDER BY name DESC")),
        vec![
            vec![some("dee")],
            vec![some("cy")],
            vec![some("bo")],
            vec![some("ann")],
            vec![some("O'Hara")],
        ],
        "ORDER BY compares bytes, so upper case sorts first"
    );
    assert_eq!(
        texts(&db.exec("SELECT team FROM people ORDER BY team NULLS FIRST")),
        vec![
            vec![None],
            vec![some("db")],
            vec![some("db")],
            vec![some("ops")],
            vec![some("web")],
        ]
    );
    assert_eq!(
        texts(&db.exec("SELECT MIN(name), MAX(name), MIN(team), MAX(team) FROM people")),
        vec![vec![some("O'Hara"), some("dee"), some("db"), some("web")]]
    );
    let mut groups = db.exec("SELECT team, COUNT(*) FROM people GROUP BY team");
    groups.sort();
    assert_eq!(
        groups,
        vec![
            vec![None, int(1)],
            vec![text("db"), int(2)],
            vec![text("ops"), int(1)],
            vec![text("web"), int(1)],
        ]
    );
    assert_eq!(
        texts(&db.exec("SELECT DISTINCT team FROM people WHERE team <> 'ops' ORDER BY team")),
        vec![vec![some("db")], vec![some("web")]]
    );

    for sql in [
        "SELECT p.id, t.floor FROM people p JOIN teams t ON p.team = t.name",
        "SELECT p.id, t.floor FROM people p LEFT JOIN teams t ON t.name = p.team",
    ] {
        let hashed = db.plan_of(sql);
        assert!(plan_contains(&hashed, is_hash_join), "{sql}");
        let nested = with_nested_loop_joins(db.plan_of(sql));
        let expected = sorted_nullable(nullable_ints(&db.run_plan(&nested).unwrap()));
        assert_eq!(
            sorted_nullable(nullable_ints(&db.run_plan(&hashed).unwrap())),
            expected,
            "{sql}"
        );
    }
    assert_eq!(
        sorted_nullable(nullable_ints(&db.exec(
            "SELECT p.id, t.floor FROM people p LEFT JOIN teams t ON p.team = t.name"
        ))),
        vec![
            vec![Some(1), Some(3)],
            vec![Some(2), Some(5)],
            vec![Some(3), Some(3)],
            vec![Some(4), None],
            vec![Some(5), None],
        ],
        "a NULL or unmatched text key joins nothing"
    );
}

#[test]
fn varchar_limits_count_characters_and_name_the_column() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (short VARCHAR(3), long TEXT)");
    db.exec("INSERT INTO t VALUES ('ééé', 'abcdef'), ('abc', 'x')");
    let too_long = |chars: usize| {
        format!("value of {chars} characters is too long for VARCHAR(3) column `short`")
    };
    for (sql, expected) in [
        ("INSERT INTO t VALUES ('abcd', NULL)", too_long(4)),
        ("UPDATE t SET short = 'éééé'", too_long(4)),
    ] {
        match db.query_error(sql) {
            QueryError::Bind(e) => assert_eq!(e.to_string(), expected, "{sql}"),
            other => panic!("{sql}: expected a bind error, got {other}"),
        }
    }
    match db.exec_error("UPDATE t SET short = long") {
        ExecutorError::Catalog(CatalogError::ColumnTooLong {
            column,
            max_chars: 3,
            len: 6,
        }) => assert_eq!(column, "short"),
        other => panic!("expected the column limit, got {other:?}"),
    }
    db.exec("UPDATE t SET short = long WHERE long = 'x'");
    assert_eq!(
        texts(&db.exec("SELECT short FROM t ORDER BY short")),
        vec![vec![some("x")], vec![some("ééé")]]
    );
}

#[test]
fn text_type_errors_are_clear() {
    let db = people_db();
    let cases = [
        (
            "SELECT name + 1 FROM people",
            "operator `+` does not take VARCHAR(10) and INT",
        ),
        ("SELECT -team FROM people", "cannot negate a TEXT"),
        (
            "SELECT id FROM people WHERE id = 'one'",
            "operator `=` does not take INT and TEXT",
        ),
        (
            "SELECT SUM(name) FROM people",
            "aggregate `SUM` does not take a VARCHAR(10) argument",
        ),
        (
            "SELECT AVG(team) FROM people",
            "aggregate `AVG` does not take a TEXT argument",
        ),
        (
            "INSERT INTO people VALUES ('6', 'x', 'y')",
            "column `id` is INT, got TEXT",
        ),
        (
            "INSERT INTO people VALUES (6, 7, 'y')",
            "column `name` is VARCHAR(10), got INT",
        ),
        (
            "UPDATE people SET team = id",
            "column `team` is TEXT, got INT",
        ),
        ("CREATE TABLE z (c CHAR(3))", "unsupported type: CHAR(3)"),
        (
            "CREATE TABLE z (c VARCHAR(0))",
            "unsupported type: VARCHAR(0)",
        ),
    ];
    let failures: Vec<String> = cases
        .into_iter()
        .filter_map(|(sql, expected)| match db.query_error(sql) {
            QueryError::Bind(e) if e.to_string() == expected => None,
            other => Some(format!("{sql}: expected {expected:?}, got {other}")),
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn varchar_columns_can_be_indexed_but_unbounded_text_cannot() {
    let db = people_db();
    let err = db
        .try_exec("CREATE INDEX by_team ON people (team)")
        .expect_err("TEXT has no key width");
    assert!(
        err.contains("column `team` is TEXT, which cannot be an index key"),
        "{err}"
    );
    assert!(db.catalog().index_oid("people", "by_team").is_err());

    db.exec("CREATE INDEX by_name ON people (name)");
    let lookup = "SELECT id FROM people WHERE name = 'cy'";
    assert!(plan_contains(&db.plan_of(lookup), |p| matches!(
        p,
        Plan::IndexScan(_)
    )));
    assert_eq!(ints(&db.exec(lookup)), vec![[3]]);
    let longer_than_the_column = "SELECT id FROM people WHERE name = 'a name past ten'";
    assert!(
        !plan_contains(&db.plan_of(longer_than_the_column), |p| matches!(
            p,
            Plan::IndexScan(_)
        )),
        "a string longer than the column has no key in the index"
    );
    assert!(db.exec(longer_than_the_column).is_empty());
    db.exec("UPDATE people SET name = 'cyril' WHERE id = 3");
    assert_eq!(
        ints(&db.exec("SELECT id FROM people WHERE name = 'cyril'")),
        vec![[3]]
    );
    assert!(db.exec(lookup).is_empty());
    assert!(
        db.try_exec("INSERT INTO people VALUES (9, 'bo', NULL)")
            .is_err(),
        "the index keeps names unique"
    );
}

#[test]
fn rows_grow_and_shrink_in_place_and_a_rollback_restores_them() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INT, note TEXT)");
    for id in 0..300 {
        db.exec(&format!("INSERT INTO t VALUES ({id}, 'n{id:03}')"));
    }
    for id in 0..300 {
        db.exec(&format!(
            "UPDATE t SET note = 'note{id:03}' WHERE id = {id}"
        ));
    }
    assert_eq!(
        texts(&db.exec("SELECT note FROM t WHERE id = 7 OR id = 299 ORDER BY id")),
        vec![vec![some("note007")], vec![some("note299")]]
    );

    db.exec("BEGIN");
    db.exec("UPDATE t SET note = 'x'");
    db.exec("UPDATE t SET note = 'longer still, longer than every note so far' WHERE id = 7");
    db.exec("ROLLBACK");
    for id in [0, 7, 150, 299] {
        assert_eq!(
            texts(&db.exec(&format!("SELECT note FROM t WHERE id = {id}"))),
            vec![vec![some(&format!("note{id:03}"))]],
            "rollback writes the old image of row {id} back"
        );
    }
    assert!(db.exec("SELECT id FROM t WHERE note = 'x'").is_empty());
}

#[test]
fn a_row_too_large_for_a_page_is_refused_with_a_clear_error() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INT, note TEXT)");
    let huge = "y".repeat(9000);
    let err = db
        .try_exec(&format!("INSERT INTO t VALUES (1, '{huge}')"))
        .expect_err("9000 bytes do not fit a page");
    assert!(
        err.contains("bytes does not fit in a page, which holds at most"),
        "{err}"
    );
    assert!(db.exec("SELECT id FROM t").is_empty());
}

#[test]
#[ignore = "known gap: a row that grows past its page's free space cannot move to another page (BOR-76)"]
fn an_update_can_grow_every_row_of_a_full_page_past_the_reserve() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INT, note TEXT)");
    for id in 0..200 {
        db.exec(&format!("INSERT INTO t VALUES ({id}, 'n')"));
    }
    let long = "z".repeat(200);
    db.exec(&format!("UPDATE t SET note = '{long}'"));
    assert_eq!(
        ints(&db.exec(&format!("SELECT COUNT(*) FROM t WHERE note = '{long}'"))),
        vec![[200]]
    );
}

#[test]
fn an_update_that_grows_rows_past_their_page_fails_and_changes_nothing() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (id INT, note TEXT)");
    for id in 0..200 {
        db.exec(&format!("INSERT INTO t VALUES ({id}, 'n')"));
    }
    let long = "z".repeat(200);
    let err = db
        .try_exec(&format!("UPDATE t SET note = '{long}'"))
        .expect_err("the first page cannot hold every row at 200 bytes");
    assert!(err.contains("rows cannot move to another page"), "{err}");
    assert_eq!(
        ints(&db.exec("SELECT COUNT(*) FROM t WHERE note = 'n'")),
        vec![[200]]
    );
}
