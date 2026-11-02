//! Shared harness for SQL-level end-to-end tests: a thin wrapper over [`cmudb::Database`]
//! that returns each statement's rows as decoded values.
#![allow(dead_code)]

pub mod seed;

use cmudb::catalog::Value;
use cmudb::concurrency::TxnError;
use cmudb::concurrency::mvcc::MvccError;
use cmudb::parser::BinderError;
use cmudb::processor::executor::ExecutorError;
use cmudb::processor::plan::{
    ColIdx, Expression, HashJoinPlanNode, JoinColumn, JoinPredicate, NestedLoopJoinPlanNode, Op,
    Plan, Side,
};
use cmudb::{Database, QueryError, StatementResult};
use std::ops::Deref;

pub type Row = cmudb::database::Row;

/// A [`Database`] in a scratch directory; derefs to it for everything the wrapper does not cover.
pub struct Db {
    db: Database,
}

impl Deref for Db {
    type Target = Database;

    fn deref(&self) -> &Database {
        &self.db
    }
}

impl Db {
    pub fn new(num_frames: usize) -> Self {
        Self {
            db: Database::scratch(num_frames).expect("temporary database"),
        }
    }

    /// A [`Db::new`] database whose `ORDER BY` spills sorted runs once its input passes
    /// `bytes` of tuple data.
    pub fn with_sort_memory_budget(num_frames: usize, bytes: usize) -> Self {
        Self {
            db: Database::scratch(num_frames)
                .expect("temporary database")
                .with_sort_memory_budget(bytes),
        }
    }

    /// Runs every statement in `sql` and returns the rows of the last one.
    /// DML statements return a single row holding the affected-row count.
    pub fn try_exec(&self, sql: &str) -> Result<Vec<Row>, String> {
        let results = self.db.execute(sql).map_err(|e| e.to_string())?;
        Ok(results.into_iter().last().map(rows_of).unwrap_or_default())
    }

    pub fn exec(&self, sql: &str) -> Vec<Row> {
        self.try_exec(sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    /// The error `sql` fails with, unwrapped from the rollback of its transaction.
    pub fn query_error(&self, sql: &str) -> QueryError {
        match self.db.execute(sql) {
            Ok(results) => panic!("{sql}: expected an error, got {} results", results.len()),
            Err(e) => rollback_cause(e),
        }
    }

    /// The executor error `sql` fails with; panics if it fails any other way.
    pub fn exec_error(&self, sql: &str) -> ExecutorError {
        match self.query_error(sql) {
            QueryError::Execute(e) => e,
            other => panic!("{sql}: expected an executor error, got {other:?}"),
        }
    }

    /// The error `plan` fails with, unwrapped from the rollback of its transaction.
    pub fn plan_error(&self, plan: &Plan) -> QueryError {
        match self.db.execute_plan(plan) {
            Ok(_) => panic!("expected the plan to fail"),
            Err(e) => rollback_cause(e),
        }
    }

    /// The binder error `sql` fails with; panics if it fails any other way.
    pub fn bind_error(&self, sql: &str) -> BinderError {
        match self.query_error(sql) {
            QueryError::Bind(e) => e,
            other => panic!("{sql}: expected a bind error, got {other:?}"),
        }
    }

    /// Runs `plan` in a transaction of its own, committed when the plan succeeds and rolled
    /// back when it fails.
    pub fn run_plan(&self, plan: &Plan) -> Result<Vec<Row>, String> {
        self.db
            .execute_plan(plan)
            .map(rows_of)
            .map_err(|e| e.to_string())
    }

    /// The plan `sql` (a single statement) gets.
    pub fn plan_of(&self, sql: &str) -> Plan {
        let mut plans = self
            .db
            .plan(sql)
            .unwrap_or_else(|e| panic!("plan {sql}: {e}"));
        assert_eq!(plans.len(), 1, "expected one statement in {sql}");
        plans.remove(0)
    }

    pub fn table_oid(&self, name: &str) -> u32 {
        self.db.catalog().table_oid(name).expect("table")
    }
}

/// Whether `wanted` matches `plan` or a node under its filters, projections, aggregations,
/// sorts and limits.
pub fn plan_contains(plan: &Plan, wanted: fn(&Plan) -> bool) -> bool {
    wanted(plan)
        || match plan {
            Plan::Filter(p) => plan_contains(&p.child, wanted),
            Plan::Projection(p) => plan_contains(&p.child, wanted),
            Plan::Aggregation(p) => plan_contains(&p.child, wanted),
            Plan::Sort(p) => plan_contains(&p.child, wanted),
            Plan::ExternalMergeSort(p) => plan_contains(&p.child, wanted),
            Plan::Limit(p) => plan_contains(&p.child, wanted),
            _ => false,
        }
}

/// `plan` with each hash join under its filters, projections, aggregations, sorts and limits
/// replaced by a nested loop join on the same condition, as a reference result for the hash
/// join.
pub fn with_nested_loop_joins(plan: Plan) -> Plan {
    let below = |child: Box<Plan>| Box::new(with_nested_loop_joins(*child));
    match plan {
        Plan::HashJoin(p) => Plan::NestedLoopJoin(NestedLoopJoinPlanNode {
            predicate: hash_join_condition(&p),
            schema: p.schema,
            left: p.left,
            right: p.right,
            join_type: p.join_type,
        }),
        Plan::Filter(mut p) => {
            p.child = below(p.child);
            Plan::Filter(p)
        }
        Plan::Projection(mut p) => {
            p.child = below(p.child);
            Plan::Projection(p)
        }
        Plan::Aggregation(mut p) => {
            p.child = below(p.child);
            Plan::Aggregation(p)
        }
        Plan::Sort(mut p) => {
            p.child = below(p.child);
            Plan::Sort(p)
        }
        Plan::ExternalMergeSort(mut p) => {
            p.child = below(p.child);
            Plan::ExternalMergeSort(p)
        }
        Plan::Limit(mut p) => {
            p.child = below(p.child);
            Plan::Limit(p)
        }
        other => other,
    }
}

/// A hash join's key equalities and residual ANDed into one join condition.
fn hash_join_condition(p: &HashJoinPlanNode) -> JoinPredicate {
    let on = |side: Side| move |col: ColIdx| JoinColumn { side, col };
    let binary = |left: JoinPredicate, right: JoinPredicate, op: Op| Expression::Binary {
        left: Box::new(left),
        right: Box::new(right),
        op,
    };
    p.left_exprs
        .iter()
        .zip(&p.right_exprs)
        .map(|(l, r)| {
            binary(
                l.clone().map_columns(on(Side::Left)),
                r.clone().map_columns(on(Side::Right)),
                Op::Eq,
            )
        })
        .chain(p.residual.clone())
        .reduce(|l, r| binary(l, r, Op::And))
        .expect("a hash join has at least one key")
}

/// The error that made a statement roll back, or `err` itself if it is not a rollback.
pub fn rollback_cause(err: QueryError) -> QueryError {
    match err {
        QueryError::RolledBack { cause, .. } => *cause,
        e => e,
    }
}

/// Whether `err` is a write-write conflict reported by an executor.
pub fn is_write_conflict(err: &QueryError) -> bool {
    matches!(
        err,
        QueryError::Execute(ExecutorError::Mvcc(MvccError::Txn(
            TxnError::WriteConflict(_)
        )))
    )
}

/// Whether `err` is a unique index rejecting a key already taken in index `index`.
pub fn is_duplicate_key(err: &QueryError, index: &str) -> bool {
    matches!(
        err,
        QueryError::Execute(ExecutorError::Mvcc(MvccError::DuplicateKey(name))) if name == index
    )
}

/// A statement's result as rows: a query's rows, one row holding a DML count, or none.
pub fn rows_of(result: StatementResult) -> Vec<Row> {
    match result {
        StatementResult::Rows { rows, .. } => rows,
        StatementResult::Count { rows, .. } => {
            vec![vec![
                int(i32::try_from(rows).expect("count fits in an INT")),
            ]]
        }
        StatementResult::Ddl(_) | StatementResult::Txn(_) => Vec::new(),
    }
}

pub fn int(v: i32) -> Option<Value> {
    Some(Value::INT(v))
}

/// Each row's INT columns; a BIGINT in the INT range reads as an INT, and any other value
/// panics.
pub fn ints(rows: &[Row]) -> Vec<Vec<i32>> {
    rows.iter()
        .map(|r| {
            r.iter()
                .map(|v| match v.and_then(|v| v.as_i64()).map(i32::try_from) {
                    Some(Ok(i)) => i,
                    _ => panic!("expected an INT, or a BIGINT in the INT range, got {v:?}"),
                })
                .collect()
        })
        .collect()
}

/// Each row's INT columns, with NULL as `None`; a BIGINT in the INT range reads as an INT, and
/// any other value panics.
pub fn nullable_ints(rows: &[Row]) -> Vec<Vec<Option<i32>>> {
    rows.iter()
        .map(|r| {
            r.iter()
                .map(|v| match v.map(|v| v.as_i64().map(i32::try_from)) {
                    Some(Some(Ok(i))) => Some(i),
                    None => None,
                    _ => panic!("expected INT or NULL, got {v:?}"),
                })
                .collect()
        })
        .collect()
}
