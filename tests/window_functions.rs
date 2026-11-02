//! Window functions (`agg(...) OVER (PARTITION BY ... ORDER BY ...)` and `RANK()`) reached
//! through SQL.

mod common;

use cmudb::catalog::Value;
use cmudb::parser::{BinderError, Unsupported};
use cmudb::processor::plan::Plan;
use common::*;

fn sales_db() -> Db {
    let db = Db::new(64);
    db.exec("CREATE TABLE sales (dept INTEGER, amount INTEGER)");
    db.exec("INSERT INTO sales VALUES (1, 10), (2, 5), (1, 30), (1, 20), (2, 5), (2, 7)");
    db
}

/// Every window function the executor computes with an INT result, each over
/// `OVER (<window>)`, after `dept, amount`, ordered by `dept, amount` so tied rows are
/// identical. AVG, a FLOAT, is checked by `avg_over_a_window_is_a_float`.
fn every_function_over(db: &Db, window: &str) -> Vec<Vec<i32>> {
    let functions = [
        "SUM(amount)",
        "COUNT(amount)",
        "COUNT(*)",
        "MIN(amount)",
        "MAX(amount)",
        "RANK()",
    ];
    let calls: Vec<String> = functions
        .iter()
        .map(|f| format!("{f} OVER ({window})"))
        .collect();
    ints(&db.exec(&format!(
        "SELECT dept, amount, {} FROM sales ORDER BY dept, amount",
        calls.join(", ")
    )))
}

fn window_node_count(plan: &Plan) -> usize {
    match plan {
        Plan::WindowFunction(p) => 1 + window_node_count(&p.child),
        Plan::Filter(p) => window_node_count(&p.child),
        Plan::Projection(p) => window_node_count(&p.child),
        Plan::ExternalMergeSort(p) => window_node_count(&p.child),
        Plan::Limit(p) => window_node_count(&p.child),
        _ => 0,
    }
}

#[test]
fn empty_over_clause_aggregates_the_whole_table_on_every_row() {
    let db = sales_db();
    let every_row = vec![77, 6, 6, 5, 30, 1];
    let expected: Vec<Vec<i32>> = [[1, 10], [1, 20], [1, 30], [2, 5], [2, 5], [2, 7]]
        .iter()
        .map(|key| key.iter().chain(&every_row).copied().collect())
        .collect();
    assert_eq!(every_function_over(&db, ""), expected);
}

#[test]
fn partition_by_without_order_by_aggregates_each_whole_partition() {
    let db = sales_db();
    assert_eq!(
        every_function_over(&db, "PARTITION BY dept"),
        vec![
            vec![1, 10, 60, 3, 3, 10, 30, 1],
            vec![1, 20, 60, 3, 3, 10, 30, 1],
            vec![1, 30, 60, 3, 3, 10, 30, 1],
            vec![2, 5, 17, 3, 3, 5, 7, 1],
            vec![2, 5, 17, 3, 3, 5, 7, 1],
            vec![2, 7, 17, 3, 3, 5, 7, 1],
        ],
        "without ORDER BY every row of a partition is a peer, so RANK is 1 throughout"
    );
}

#[test]
fn order_by_without_partition_by_runs_over_the_table_and_ties_share_a_rank() {
    let db = sales_db();
    assert_eq!(
        every_function_over(&db, "ORDER BY amount"),
        vec![
            vec![1, 10, 27, 4, 4, 5, 10, 4],
            vec![1, 20, 47, 5, 5, 5, 20, 5],
            vec![1, 30, 77, 6, 6, 5, 30, 6],
            vec![2, 5, 10, 2, 2, 5, 5, 1],
            vec![2, 5, 10, 2, 2, 5, 5, 1],
            vec![2, 7, 17, 3, 3, 5, 7, 3],
        ],
        "the two 5s are peers: both rank 1, both see each other, and 7 ranks 3"
    );
}

#[test]
fn partition_by_and_order_by_run_within_each_partition() {
    let db = sales_db();
    assert_eq!(
        every_function_over(&db, "PARTITION BY dept ORDER BY amount"),
        vec![
            vec![1, 10, 10, 1, 1, 10, 10, 1],
            vec![1, 20, 30, 2, 2, 10, 20, 2],
            vec![1, 30, 60, 3, 3, 10, 30, 3],
            vec![2, 5, 10, 2, 2, 5, 5, 1],
            vec![2, 5, 10, 2, 2, 5, 5, 1],
            vec![2, 7, 17, 3, 3, 5, 7, 3],
        ]
    );
}

#[test]
fn rank_follows_the_window_order_direction() {
    let db = sales_db();
    assert_eq!(
        ints(&db.exec(
            "SELECT amount, RANK() OVER (ORDER BY amount DESC) FROM sales ORDER BY amount DESC"
        )),
        vec![[30, 1], [20, 2], [10, 3], [7, 4], [5, 5], [5, 5]]
    );
}

#[test]
fn window_order_by_honors_nulls_first() {
    let db = sales_db();
    db.exec("INSERT INTO sales VALUES (1, NULL)");
    assert_eq!(
        nullable_ints(&db.exec(
            "SELECT amount, RANK() OVER (PARTITION BY dept ORDER BY amount NULLS FIRST) \
             FROM sales WHERE dept = 1 ORDER BY amount NULLS FIRST"
        )),
        vec![
            vec![None, Some(1)],
            vec![Some(10), Some(2)],
            vec![Some(20), Some(3)],
            vec![Some(30), Some(4)],
        ]
    );
}

#[test]
fn count_of_a_column_skips_nulls_but_count_star_does_not() {
    let db = sales_db();
    db.exec("INSERT INTO sales VALUES (1, NULL)");
    let counts = ints(&db.exec(
        "SELECT dept, COUNT(amount) OVER (PARTITION BY dept), COUNT(*) OVER (PARTITION BY dept) \
         FROM sales WHERE dept = 1",
    ));
    assert_eq!(counts, vec![[1, 3, 4]; 4]);
}

#[test]
fn where_filters_rows_before_the_window_sees_them() {
    let db = sales_db();
    assert_eq!(
        ints(&db.exec("SELECT amount, COUNT(*) OVER () FROM sales WHERE dept = 1 ORDER BY amount")),
        vec![[10, 3], [20, 3], [30, 3]]
    );
}

#[test]
fn window_results_can_be_computed_on_ordered_by_and_limited() {
    let db = sales_db();
    assert_eq!(
        ints(&db.exec(
            "SELECT amount, RANK() OVER (ORDER BY amount) * 10 FROM sales \
             ORDER BY RANK() OVER (ORDER BY amount DESC), amount LIMIT 3"
        )),
        vec![[30, 60], [20, 50], [10, 40]]
    );
}

#[test]
fn calls_over_different_windows_each_read_their_own_window() {
    let db = sales_db();
    let sql = "SELECT dept, amount, RANK() OVER (PARTITION BY dept ORDER BY amount), \
               SUM(amount) OVER (), SUM(amount) OVER (PARTITION BY dept ORDER BY amount) \
               FROM sales ORDER BY dept, amount";
    assert_eq!(
        ints(&db.exec(sql)),
        vec![
            vec![1, 10, 1, 77, 10],
            vec![1, 20, 2, 77, 30],
            vec![1, 30, 3, 77, 60],
            vec![2, 5, 1, 77, 10],
            vec![2, 5, 1, 77, 10],
            vec![2, 7, 3, 77, 17],
        ]
    );
    assert_eq!(
        window_node_count(&db.plan_of(sql)),
        2,
        "the two calls over the same OVER clause share one window node"
    );
}

#[test]
fn window_over_a_join_reads_both_sides() {
    let db = sales_db();
    db.exec("CREATE TABLE depts (id INTEGER, bonus INTEGER)");
    db.exec("INSERT INTO depts VALUES (1, 100), (2, 200)");
    assert_eq!(
        ints(&db.exec(
            "SELECT amount, SUM(amount + bonus) OVER (PARTITION BY id) \
             FROM sales JOIN depts ON sales.dept = depts.id WHERE id = 2 ORDER BY amount"
        )),
        vec![[5, 617], [5, 617], [7, 617]]
    );
}

#[test]
fn window_function_without_from_sees_one_row() {
    let db = Db::new(16);
    assert_eq!(
        ints(&db.exec("SELECT RANK() OVER (), COUNT(*) OVER ()")),
        vec![[1, 1]]
    );
}

#[test]
fn window_frame_clauses_are_rejected() {
    let db = sales_db();
    for frame in [
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW",
        "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        "GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
    ] {
        let err = db.bind_error(&format!(
            "SELECT SUM(amount) OVER (ORDER BY amount {frame}) FROM sales"
        ));
        assert_eq!(
            err,
            BinderError::Unsupported(Unsupported::WindowFrame),
            "{frame}"
        );
        assert_eq!(
            err.to_string(),
            "window frame clauses (ROWS / RANGE / GROUPS) are not supported"
        );
    }
}

#[test]
fn rank_requires_an_over_clause_and_takes_no_arguments() {
    let db = sales_db();
    let rank = || "RANK".to_string();
    let err = db.bind_error("SELECT RANK() FROM sales");
    assert_eq!(
        err,
        BinderError::WindowFunctionWithoutOver { function: rank() }
    );
    assert_eq!(err.to_string(), "RANK() requires an OVER clause");
    assert_eq!(
        db.bind_error("SELECT RANK(amount) OVER () FROM sales"),
        BinderError::WindowFunctionArguments { function: rank() }
    );
}

#[test]
fn window_functions_outside_the_projection_and_order_by_are_rejected() {
    let db = sales_db();
    for sql in [
        "SELECT amount FROM sales WHERE RANK() OVER (ORDER BY amount) = 1",
        "SELECT dept FROM sales GROUP BY RANK() OVER ()",
    ] {
        assert_eq!(
            db.bind_error(sql),
            BinderError::WindowOutsideProjection,
            "{sql}"
        );
    }
}

#[test]
fn nested_window_functions_are_rejected() {
    let db = sales_db();
    for sql in [
        "SELECT SUM(RANK() OVER ()) OVER () FROM sales",
        "SELECT SUM(amount) OVER (ORDER BY RANK() OVER ()) FROM sales",
    ] {
        assert_eq!(
            db.bind_error(sql),
            BinderError::NestedWindowFunction,
            "{sql}"
        );
    }
    assert_eq!(
        db.bind_error("SELECT SUM(COUNT(amount)) OVER () FROM sales"),
        BinderError::NestedAggregate
    );
}

#[test]
fn unsupported_window_function_forms_are_rejected() {
    let db = sales_db();
    let cases = [
        (
            "SELECT SUM(amount) OVER w FROM sales WINDOW w AS (PARTITION BY dept)",
            BinderError::Unsupported(Unsupported::NamedWindow),
        ),
        (
            "SELECT COUNT(DISTINCT amount) OVER () FROM sales",
            BinderError::Unsupported(Unsupported::AggregateDistinct),
        ),
        (
            "SELECT SUM(amount) FILTER (WHERE dept = 1) OVER () FROM sales",
            BinderError::Unsupported(Unsupported::WindowFunctionClauses),
        ),
        (
            "SELECT amount, RANK() OVER (ORDER BY amount) FROM sales \
             QUALIFY RANK() OVER (ORDER BY amount) = 1",
            BinderError::Unsupported(Unsupported::Qualify),
        ),
        (
            "SELECT ROW_NUMBER() OVER () FROM sales",
            BinderError::UnknownWindowFunction {
                function: "ROW_NUMBER".into(),
            },
        ),
    ];
    for (sql, expected) in cases {
        assert_eq!(db.bind_error(sql), expected, "{sql}");
    }
}

#[test]
fn window_functions_in_an_aggregating_select_are_rejected() {
    let db = sales_db();
    assert_eq!(
        db.bind_error("SELECT dept, RANK() OVER (ORDER BY dept) FROM sales GROUP BY dept"),
        BinderError::Unsupported(Unsupported::WindowInAggregatingSelect)
    );
}

#[test]
#[ignore = "known gap: window functions over GROUP BY output are rejected"]
fn window_functions_rank_grouped_rows() {
    let db = sales_db();
    assert_eq!(
        ints(&db.exec(
            "SELECT dept, SUM(amount), RANK() OVER (ORDER BY SUM(amount) DESC) \
             FROM sales GROUP BY dept ORDER BY dept"
        )),
        vec![[1, 60, 1], [2, 17, 2]]
    );
}

#[test]
#[ignore = "known gap: explicit window frames (ROWS / RANGE / GROUPS) are rejected"]
fn rows_frame_computes_a_moving_sum() {
    let db = sales_db();
    assert_eq!(
        ints(&db.exec(
            "SELECT amount, SUM(amount) OVER (ORDER BY amount ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
             FROM sales WHERE dept = 1 ORDER BY amount"
        )),
        vec![[10, 10], [20, 30], [30, 50]]
    );
}

#[test]
#[ignore = "known gap: QUALIFY is rejected"]
fn qualify_filters_on_a_window_result() {
    let db = sales_db();
    assert_eq!(
        ints(&db.exec(
            "SELECT dept, amount FROM sales \
             QUALIFY RANK() OVER (PARTITION BY dept ORDER BY amount DESC) = 1 ORDER BY dept"
        )),
        vec![[1, 30], [2, 7]]
    );
}

#[test]
fn avg_over_a_window_is_a_float() {
    let db = sales_db();
    let third = 17.0 / 3.0;
    for (window, expected) in [
        ("", vec![77.0 / 6.0; 6]),
        (
            "PARTITION BY dept",
            vec![20.0, 20.0, 20.0, third, third, third],
        ),
        (
            "ORDER BY amount",
            vec![6.75, 9.4, 77.0 / 6.0, 5.0, 5.0, third],
        ),
        (
            "PARTITION BY dept ORDER BY amount",
            vec![10.0, 15.0, 20.0, 5.0, 5.0, third],
        ),
    ] {
        let avgs: Vec<Option<Value>> = db
            .exec(&format!(
                "SELECT AVG(amount) OVER ({window}) FROM sales ORDER BY dept, amount"
            ))
            .into_iter()
            .map(|r| r[0])
            .collect();
        let expected: Vec<Option<Value>> = expected
            .into_iter()
            .map(|f| Some(Value::float(f)))
            .collect();
        assert_eq!(avgs, expected, "AVG(amount) OVER ({window})");
    }
}
