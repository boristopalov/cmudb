//! Shared harness for SQL-level end-to-end tests: wires disk manager,
//! buffer pool, catalog, binder, planner and executors together and runs
//! SQL text, returning decoded rows.
#![allow(dead_code)]

use cmudb::buffer_pool::BufferPoolManager;
use cmudb::catalog::{Catalog, Value};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::parser::Binder;
use cmudb::planner::Planner;
use cmudb::processor::executor_factory::create_executor;
use cmudb::processor::plan::Plan;
use cmudb::replacer::ArcReplacer;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::sync::Arc;
use tempfile::TempDir;

pub type Row = Vec<Option<Value>>;

pub struct Db {
    pub catalog: Catalog,
    pub bpm: Arc<BufferPoolManager>,
    _dir: TempDir,
}

impl Db {
    pub fn new(num_frames: usize) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk");
        let bpm = Arc::new(create_buffer_pool_manager(
            num_frames,
            ArcReplacer::new(num_frames),
            DiskScheduler::new(dm),
        ));
        let catalog = Catalog::new(bpm.clone());
        Self {
            catalog,
            bpm,
            _dir: dir,
        }
    }

    /// Runs every statement in `sql` and returns the rows of the last one.
    /// DML statements return a single row holding the affected-row count.
    pub fn try_exec(&self, sql: &str) -> Result<Vec<Row>, String> {
        let ast = Parser::parse_sql(&GenericDialect, sql).map_err(|e| format!("parse: {e}"))?;
        let mut out = Vec::new();
        for stmt in ast {
            let bound = Binder::new(&self.catalog)
                .bind(vec![stmt])
                .map_err(|e| format!("bind: {}", e.0))?;
            for stmt in bound {
                let plan = Planner::new(&self.catalog)
                    .plan(stmt)
                    .map_err(|e| format!("plan: {}", e.0))?;
                out = self.run_plan(&plan)?;
            }
        }
        Ok(out)
    }

    pub fn exec(&self, sql: &str) -> Vec<Row> {
        self.try_exec(sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    pub fn run_plan(&self, plan: &Plan) -> Result<Vec<Row>, String> {
        let mut exec = create_executor(&self.catalog, &self.bpm, plan);
        exec.open().map_err(|e| format!("open: {}", e.0))?;
        let mut rows = Vec::new();
        match plan {
            Plan::Insert(_) => {
                if let Some((t, _)) = exec.next().map_err(|e| format!("exec: {}", e.0))? {
                    let n = i32::from_ne_bytes(t.data.as_slice().try_into().unwrap());
                    rows.push(vec![Some(Value::INT(n))]);
                }
            }
            Plan::Update(_) | Plan::Delete(_) => {
                let mut n = 0;
                while exec.next().map_err(|e| format!("exec: {}", e.0))?.is_some() {
                    n += 1;
                }
                rows.push(vec![Some(Value::INT(n))]);
            }
            Plan::CreateTable(_)
            | Plan::CreateIndex(_)
            | Plan::DropTable(_)
            | Plan::DropIndex(_) => {
                exec.next().map_err(|e| format!("exec: {}", e.0))?;
            }
            _ => {
                let schema = exec.schema().clone();
                while let Some((t, _)) = exec.next().map_err(|e| format!("exec: {}", e.0))? {
                    let row = (0..schema.cols.len())
                        .map(|i| schema.get_value(&t, i).expect("decode"))
                        .collect();
                    rows.push(row);
                }
            }
        }
        exec.close().map_err(|e| format!("close: {}", e.0))?;
        Ok(rows)
    }

    pub fn table_oid(&self, name: &str) -> u32 {
        self.catalog.table_oid(name).expect("table")
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
