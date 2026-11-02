//! Shared harness for SQL-level end-to-end tests: a thin wrapper over [`cmudb::Database`]
//! that returns each statement's rows as decoded values.
#![allow(dead_code)]

pub mod seed;

use cmudb::catalog::Value;
use cmudb::processor::plan::Plan;
use cmudb::{Database, StatementResult};
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

    /// Runs every statement in `sql` and returns the rows of the last one.
    /// DML statements return a single row holding the affected-row count.
    pub fn try_exec(&self, sql: &str) -> Result<Vec<Row>, String> {
        let results = self.db.execute(sql).map_err(|e| e.to_string())?;
        Ok(results.into_iter().last().map(rows_of).unwrap_or_default())
    }

    pub fn exec(&self, sql: &str) -> Vec<Row> {
        self.try_exec(sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
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

pub fn ints(rows: &[Row]) -> Vec<Vec<i32>> {
    rows.iter()
        .map(|r| {
            r.iter()
                .map(|v| match v {
                    Some(Value::INT(i)) => *i,
                    other => panic!("expected INT, got {other:?}"),
                })
                .collect()
        })
        .collect()
}

/// Each row's INT columns, with NULL as `None`; panics on any other type.
pub fn nullable_ints(rows: &[Row]) -> Vec<Vec<Option<i32>>> {
    rows.iter()
        .map(|r| {
            r.iter()
                .map(|v| match v {
                    Some(Value::INT(i)) => Some(*i),
                    None => None,
                    other => panic!("expected INT or NULL, got {other:?}"),
                })
                .collect()
        })
        .collect()
}
