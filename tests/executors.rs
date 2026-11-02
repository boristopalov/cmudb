//! Executor tests for plan nodes the planner does not emit yet (hash join,
//! external merge sort, window functions, DROP TABLE/INDEX). Plans are built
//! by hand on top of tables created through SQL.
mod common;

use cmudb::catalog::{DataType, Schema, Value};
use cmudb::processor::plan::{
    AggType, ColIdx, DropIndexPlanNode, DropTablePlanNode, Expression, ExternalMergeSortPlanNode,
    HashJoinPlanNode, JoinType, LimitPlanNode, NullOrdering, Plan, SeqScanPlanNode, SortDirection,
    SortExpr, SortPlanNode, WindowFunctionPlanNode,
};
use common::*;

fn col(col_idx: u32) -> Expression {
    Expression::Column {
        col: ColIdx(col_idx),
        dtype: DataType::INT,
    }
}

fn int_schema(names: &[&str]) -> Schema {
    Schema::new(
        names
            .iter()
            .map(|n| (DataType::INT, n.to_string()))
            .collect(),
    )
}

fn scan(db: &Db, table: &str, names: &[&str]) -> Plan {
    Plan::SeqScan(SeqScanPlanNode {
        table_oid: db.table_oid(table),
        schema: int_schema(names),
    })
}

fn join_schema(left: &[&str], right: &[&str]) -> Schema {
    int_schema(&[left, right].concat())
}

fn two_tables() -> Db {
    let db = Db::new(256);
    db.exec("CREATE TABLE l (k INTEGER, a INTEGER)");
    db.exec("CREATE TABLE r (k INTEGER, b INTEGER)");
    db.exec("INSERT INTO l VALUES (1, 10), (2, 20), (2, 21), (3, 30), (NULL, 99)");
    db.exec("INSERT INTO r VALUES (2, 200), (3, 300), (3, 301), (4, 400), (NULL, 999)");
    db
}

fn hash_join(db: &Db, join_type: JoinType) -> Plan {
    Plan::HashJoin(HashJoinPlanNode {
        schema: join_schema(&["k", "a"], &["k", "b"]),
        left: Box::new(scan(db, "l", &["k", "a"])),
        right: Box::new(scan(db, "r", &["k", "b"])),
        left_exprs: vec![col(0)],
        right_exprs: vec![col(0)],
        join_type,
    })
}

#[test]
fn hash_join_inner_matches_equal_keys_only() {
    let db = two_tables();
    let mut rows = ints(&db.run_plan(&hash_join(&db, JoinType::Inner)).unwrap());
    rows.sort();
    assert_eq!(
        rows,
        vec![
            vec![2, 20, 2, 200],
            vec![2, 21, 2, 200],
            vec![3, 30, 3, 300],
            vec![3, 30, 3, 301],
        ]
    );
}

#[test]
fn hash_join_left_pads_unmatched_rows_with_nulls() {
    let db = two_tables();
    let rows = db.run_plan(&hash_join(&db, JoinType::Left)).unwrap();
    assert_eq!(rows.len(), 6);
    let unmatched: Vec<_> = rows.iter().filter(|r| r[2].is_none()).collect();
    assert_eq!(
        unmatched.len(),
        2,
        "k=1 and k=NULL have no partner: {rows:?}"
    );
    assert!(
        rows.iter().all(|r| r[0].is_some() || r[2].is_none()),
        "NULL keys never match"
    );
    assert!(unmatched.iter().all(|r| r[3].is_none()));
}

#[test]
fn hash_join_handles_empty_inputs() {
    let db = Db::new(256);
    db.exec("CREATE TABLE l (k INTEGER, a INTEGER)");
    db.exec("CREATE TABLE r (k INTEGER, b INTEGER)");
    assert!(
        db.run_plan(&hash_join(&db, JoinType::Inner))
            .unwrap()
            .is_empty()
    );
    db.exec("INSERT INTO l VALUES (1, 1)");
    assert!(
        db.run_plan(&hash_join(&db, JoinType::Inner))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        db.run_plan(&hash_join(&db, JoinType::Left)).unwrap().len(),
        1
    );
}

#[test]
fn external_merge_sort_orders_by_multiple_keys() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (a INTEGER, b INTEGER)");
    db.exec("INSERT INTO t VALUES (3, 1), (1, 2), (2, 3), (1, 1), (NULL, 5)");
    let plan = Plan::ExternalMergeSort(ExternalMergeSortPlanNode {
        schema: int_schema(&["a", "b"]),
        child: Box::new(scan(&db, "t", &["a", "b"])),
        order_by_exprs: vec![SortExpr::ascending(col(0)), SortExpr::ascending(col(1))],
    });
    let rows = db.run_plan(&plan).unwrap();
    let keys: Vec<_> = rows.iter().map(|r| (r[0], r[1])).collect();
    assert_eq!(
        keys,
        vec![
            (int(1), int(1)),
            (int(1), int(2)),
            (int(2), int(3)),
            (int(3), int(1)),
            (None, int(5)),
        ],
        "NULL sorts above every value, so it comes last ascending"
    );
}

#[test]
fn external_merge_sort_of_empty_input_is_empty() {
    let db = Db::new(16);
    db.exec("CREATE TABLE t (a INTEGER)");
    let plan = Plan::ExternalMergeSort(ExternalMergeSortPlanNode {
        schema: int_schema(&["a"]),
        child: Box::new(scan(&db, "t", &["a"])),
        order_by_exprs: vec![SortExpr::ascending(col(0))],
    });
    assert!(db.run_plan(&plan).unwrap().is_empty());
}

#[test]
fn external_merge_sort_orders_mixed_directions_with_nulls_highest() {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (a INTEGER, b INTEGER)");
    db.exec("INSERT INTO t VALUES (3, 1), (1, 2), (2, 3), (1, 1), (NULL, 5)");
    let plan = Plan::ExternalMergeSort(ExternalMergeSortPlanNode {
        schema: int_schema(&["a", "b"]),
        child: Box::new(scan(&db, "t", &["a", "b"])),
        order_by_exprs: vec![SortExpr::descending(col(0)), SortExpr::ascending(col(1))],
    });
    let rows = db.run_plan(&plan).unwrap();
    let keys: Vec<_> = rows.iter().map(|r| (r[0], r[1])).collect();
    assert_eq!(
        keys,
        vec![
            (None, int(5)),
            (int(3), int(1)),
            (int(2), int(3)),
            (int(1), int(1)),
            (int(1), int(2)),
        ],
        "NULL sorts above every value, so it comes first descending"
    );
}

fn sort_keys_with_nulls(direction: SortDirection, nulls: NullOrdering) -> Vec<SortExpr> {
    vec![SortExpr::new(col(0), direction).with_nulls(nulls)]
}

fn nulls_db() -> Db {
    let db = Db::new(64);
    db.exec("CREATE TABLE t (a INTEGER)");
    db.exec("INSERT INTO t VALUES (2), (NULL), (3), (1), (NULL)");
    db
}

const NULL_PLACEMENT_CASES: [(SortDirection, NullOrdering, [Option<i32>; 5]); 4] = [
    (
        SortDirection::Ascending,
        NullOrdering::First,
        [None, None, Some(1), Some(2), Some(3)],
    ),
    (
        SortDirection::Ascending,
        NullOrdering::Last,
        [Some(1), Some(2), Some(3), None, None],
    ),
    (
        SortDirection::Descending,
        NullOrdering::First,
        [None, None, Some(3), Some(2), Some(1)],
    ),
    (
        SortDirection::Descending,
        NullOrdering::Last,
        [Some(3), Some(2), Some(1), None, None],
    ),
];

#[test]
fn sort_places_nulls_first_or_last_in_either_direction() {
    let db = nulls_db();
    for (direction, nulls, expected) in NULL_PLACEMENT_CASES {
        let plan = Plan::Sort(SortPlanNode {
            schema: int_schema(&["a"]),
            child: Box::new(scan(&db, "t", &["a"])),
            order_by_exprs: sort_keys_with_nulls(direction, nulls),
        });
        assert_eq!(
            nullable_ints(&db.run_plan(&plan).unwrap()),
            expected.map(|v| vec![v]),
            "{direction:?} NULLS {nulls:?}"
        );
    }
}

#[test]
fn external_merge_sort_places_nulls_first_or_last_in_either_direction() {
    let db = nulls_db();
    for (direction, nulls, expected) in NULL_PLACEMENT_CASES {
        let plan = Plan::ExternalMergeSort(ExternalMergeSortPlanNode {
            schema: int_schema(&["a"]),
            child: Box::new(scan(&db, "t", &["a"])),
            order_by_exprs: sort_keys_with_nulls(direction, nulls),
        });
        assert_eq!(
            nullable_ints(&db.run_plan(&plan).unwrap()),
            expected.map(|v| vec![v]),
            "{direction:?} NULLS {nulls:?}"
        );
    }
}

#[test]
fn sort_and_limit_nodes_compose() {
    let db = Db::new(16);
    db.exec("CREATE TABLE t (a INTEGER)");
    db.exec("INSERT INTO t VALUES (5), (3), (9), (1)");
    let plan = Plan::Limit(LimitPlanNode {
        schema: int_schema(&["a"]),
        limit: 2,
        child: Box::new(Plan::Sort(SortPlanNode {
            schema: int_schema(&["a"]),
            child: Box::new(scan(&db, "t", &["a"])),
            order_by_exprs: vec![SortExpr::ascending(col(0))],
        })),
    });
    assert_eq!(ints(&db.run_plan(&plan).unwrap()), vec![[1], [3]]);
}

fn sales_db() -> Db {
    let db = Db::new(64);
    db.exec("CREATE TABLE sales (dept INTEGER, amount INTEGER)");
    db.exec("INSERT INTO sales VALUES (1, 10), (2, 5), (1, 30), (1, 20), (2, 5), (2, 7)");
    db
}

fn window(
    db: &Db,
    partition: Option<Vec<Expression>>,
    order: Option<Vec<Expression>>,
    aggs: Vec<AggType>,
) -> Plan {
    let mut names = vec!["dept".to_string(), "amount".to_string()];
    let aggregates: Vec<_> = aggs
        .into_iter()
        .enumerate()
        .map(|(i, a)| {
            names.push(format!("agg{i}"));
            (col(1), a)
        })
        .collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    Plan::WindowFunction(WindowFunctionPlanNode {
        schema: int_schema(&names),
        child: Box::new(scan(db, "sales", &["dept", "amount"])),
        order_by_exprs: order,
        partition_by_exprs: partition,
        aggregates,
        columns: vec![col(0), col(1)],
    })
}

#[test]
fn window_partition_without_order_gives_partition_totals() {
    let db = sales_db();
    let plan = window(
        &db,
        Some(vec![col(0)]),
        None,
        vec![
            AggType::SUM,
            AggType::COUNT,
            AggType::MIN,
            AggType::MAX,
            AggType::AVG,
        ],
    );
    let mut rows = ints(&db.run_plan(&plan).unwrap());
    rows.sort();
    assert_eq!(
        rows,
        vec![
            vec![1, 10, 60, 3, 10, 30, 20],
            vec![1, 20, 60, 3, 10, 30, 20],
            vec![1, 30, 60, 3, 10, 30, 20],
            vec![2, 5, 17, 3, 5, 7, 5],
            vec![2, 5, 17, 3, 5, 7, 5],
            vec![2, 7, 17, 3, 5, 7, 5],
        ]
    );
}

#[test]
fn window_order_by_puts_null_keys_in_the_last_peer_group() {
    let db = sales_db();
    db.exec("INSERT INTO sales VALUES (1, NULL)");
    let plan = window(
        &db,
        Some(vec![col(0)]),
        Some(vec![col(1)]),
        vec![AggType::SUM, AggType::RANK],
    );
    let dept_1: Vec<_> = nullable_ints(&db.run_plan(&plan).unwrap())
        .into_iter()
        .filter(|r| r[0] == Some(1))
        .collect();
    assert_eq!(
        dept_1,
        vec![
            vec![Some(1), Some(10), Some(10), Some(1)],
            vec![Some(1), Some(20), Some(30), Some(2)],
            vec![Some(1), Some(30), Some(60), Some(3)],
            vec![Some(1), None, Some(60), Some(4)],
        ],
        "a NULL order key sorts above every value, so it ranks last and adds nothing to the sum"
    );
}

#[test]
fn window_with_order_by_computes_running_sum_and_rank_with_ties() {
    let db = sales_db();
    let plan = window(
        &db,
        Some(vec![col(0)]),
        Some(vec![col(1)]),
        vec![AggType::SUM, AggType::RANK],
    );
    let rows = ints(&db.run_plan(&plan).unwrap());
    assert_eq!(
        rows,
        vec![
            vec![1, 10, 10, 1],
            vec![1, 20, 30, 2],
            vec![1, 30, 60, 3],
            vec![2, 5, 10, 1],
            vec![2, 5, 10, 1],
            vec![2, 7, 17, 3],
        ]
    );
}

#[test]
fn window_without_partition_is_one_partition() {
    let db = sales_db();
    let plan = window(&db, None, None, vec![AggType::COUNT]);
    let rows = ints(&db.run_plan(&plan).unwrap());
    assert_eq!(rows.len(), 6);
    assert!(rows.iter().all(|r| r[2] == 6));
}

#[test]
fn drop_index_then_drop_table() {
    let db = Db::new(32);
    db.exec("CREATE TABLE t (id INTEGER, v INTEGER)");
    db.exec("CREATE INDEX t_id ON t (id)");
    db.exec("INSERT INTO t VALUES (1, 10)");
    let table_oid = db.table_oid("t");
    let index_oid = db.catalog.index_oid("t", "t_id").unwrap();

    db.run_plan(&Plan::DropIndex(DropIndexPlanNode {
        index_oid,
        if_exists: false,
    }))
    .unwrap();
    assert!(db.catalog.get_index(index_oid).is_err());
    assert!(db.catalog.get_table_indexes(table_oid).unwrap().is_empty());
    assert_eq!(
        ints(&db.exec("SELECT v FROM t WHERE id = 1")),
        vec![[10]],
        "falls back to a sequential scan once the index is gone"
    );
    let again = Plan::DropIndex(DropIndexPlanNode {
        index_oid,
        if_exists: false,
    });
    assert!(db.run_plan(&again).is_err());
    let again = Plan::DropIndex(DropIndexPlanNode {
        index_oid,
        if_exists: true,
    });
    db.run_plan(&again).unwrap();

    db.run_plan(&Plan::DropTable(DropTablePlanNode {
        table_oid,
        if_exists: false,
    }))
    .unwrap();
    assert!(
        db.try_exec("SELECT * FROM t")
            .unwrap_err()
            .contains("table not found")
    );
    let again = Plan::DropTable(DropTablePlanNode {
        table_oid,
        if_exists: false,
    });
    assert!(db.run_plan(&again).is_err());
    let again = Plan::DropTable(DropTablePlanNode {
        table_oid,
        if_exists: true,
    });
    db.run_plan(&again).unwrap();
    db.exec("CREATE TABLE t (id INTEGER)");
}

#[test]
fn values_are_comparable_and_support_arithmetic_across_types() {
    assert_eq!(Value::INT(2) + Value::INT(3), Value::INT(5));
    assert_eq!(
        Value::FLOAT(1.5.into()) * Value::FLOAT(2.0.into()),
        Value::FLOAT(3.0.into())
    );
    assert_eq!(
        Value::FLOAT(4.0.into()) - Value::FLOAT(1.0.into()),
        Value::FLOAT(3.0.into())
    );
    assert_eq!(
        Value::FLOAT(4.0.into()) / Value::FLOAT(2.0.into()),
        Value::FLOAT(2.0.into())
    );
    assert!(Value::INT(1) < Value::INT(2));
    assert_eq!(Value::TIMESTAMP(5).data_type(), DataType::TIMESTAMP);
    assert_eq!(Value::BOOLEAN(true).data_type(), DataType::BOOLEAN);
}
