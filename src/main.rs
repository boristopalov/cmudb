use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;

use cmudb::catalog::{Catalog, Schema, Value};
use cmudb::concurrency::{IsolationLevel, Transaction, TransactionManager};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::parser::Binder;
use cmudb::planner::Planner;
use cmudb::processor::executor::{Executor, ExecutorContext, RowCursor};
use cmudb::processor::executor_factory::create_executor;
use cmudb::processor::plan::Plan;
use cmudb::replacer::ArcReplacer;

use sqlparser::ast::Statement;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use tempfile::tempdir;

const NUM_FRAMES: usize = 64;

fn main() {
    let db_dir = tempdir().expect("failed to create tempdir for db");
    run_shell(db_dir.path());
}

fn run_shell(db_dir: &Path) {
    let dm = DiskManager::new(db_dir.join("cmudb.data"), db_dir.join("cmudb.log"))
        .expect("failed to create disk manager");
    let scheduler = DiskScheduler::new(dm);
    let replacer = ArcReplacer::new(NUM_FRAMES);
    let bpm = Arc::new(create_buffer_pool_manager(NUM_FRAMES, replacer, scheduler));
    let catalog = Catalog::new(bpm.clone());
    let mut session = Session {
        catalog: &catalog,
        bpm: bpm.clone(),
        txn_mgr: Arc::new(TransactionManager::new(bpm.clone())),
        txn: None,
    };

    let dialect = GenericDialect {};
    let stdin = io::stdin();
    let stdin = stdin.lock();
    let mut stdout = io::stdout();
    let mut buf = String::new();

    println!("cmudb — a toy SQL shell. terminate statements with `;`. type `\\q` to quit.");

    for line in stdin.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        };

        if buf.is_empty() {
            let t = line.trim();
            if t == "\\q" || t == "quit" || t == "exit" {
                break;
            }
            if t.is_empty() {
                prompt(&mut stdout, false);
                continue;
            }
        }

        buf.push_str(&line);
        buf.push('\n');

        if buf.trim_end().ends_with(';') {
            let sql = std::mem::take(&mut buf);
            if let Err(e) = session.run(&dialect, &sql) {
                eprintln!("error: {e}");
            }
            prompt(&mut stdout, false);
        } else {
            prompt(&mut stdout, true);
        }
    }
}

fn prompt(out: &mut io::Stdout, continuation: bool) {
    let p = if continuation { "    -> " } else { "cmudb> " };
    let _ = out.write_all(p.as_bytes());
    let _ = out.flush();
}

/// A REPL session. Statements run in the explicit transaction opened by `BEGIN` if there is
/// one, and otherwise each in a transaction of its own (autocommit).
struct Session<'a> {
    catalog: &'a Catalog,
    bpm: Arc<cmudb::buffer_pool::BufferPoolManager>,
    txn_mgr: Arc<TransactionManager>,
    txn: Option<ExplicitTxn>,
}

/// The transaction opened by `BEGIN`.
enum ExplicitTxn {
    Open(Arc<Transaction>),
    /// A statement failed. Its writes are already rolled back. As in Postgres, every later
    /// statement is refused until the user ends the block with `ROLLBACK` (or `COMMIT`, which
    /// rolls back too), so a partial transaction cannot pass for a whole one.
    Failed,
}

impl Session<'_> {
    /// Runs every statement in `sql`. A failed statement aborts its whole transaction right
    /// away, releasing its rows and its snapshot, since there is no statement-level rollback.
    fn run(&mut self, dialect: &GenericDialect, sql: &str) -> Result<(), String> {
        let ast = Parser::parse_sql(dialect, sql).map_err(|e| format!("parse: {e}"))?;
        for stmt in ast {
            match stmt {
                Statement::StartTransaction { .. } => {
                    if self.txn.is_some() {
                        return Err("a transaction is already in progress".to_string());
                    }
                    self.txn = Some(ExplicitTxn::Open(
                        self.txn_mgr.begin(IsolationLevel::SnapshotIsolation),
                    ));
                    println!("BEGIN");
                }
                Statement::Commit { .. } => match self.txn.take() {
                    Some(ExplicitTxn::Open(txn)) => {
                        self.finish(&txn, true)?;
                        println!("COMMIT");
                    }
                    Some(ExplicitTxn::Failed) => println!("ROLLBACK"),
                    None => return Err("no transaction in progress".to_string()),
                },
                Statement::Rollback { .. } => match self.txn.take() {
                    Some(ExplicitTxn::Open(txn)) => {
                        self.finish(&txn, false)?;
                        println!("ROLLBACK");
                    }
                    Some(ExplicitTxn::Failed) => println!("ROLLBACK"),
                    None => return Err("no transaction in progress".to_string()),
                },
                stmt => {
                    let (txn, autocommit) = match &self.txn {
                        Some(ExplicitTxn::Open(txn)) => (txn.clone(), false),
                        Some(ExplicitTxn::Failed) => {
                            return Err("current transaction is aborted, statements are \
                                        ignored until ROLLBACK"
                                .to_string());
                        }
                        None => (self.txn_mgr.begin(IsolationLevel::SnapshotIsolation), true),
                    };
                    match self.run_statement(&txn, &stmt) {
                        Ok(()) if autocommit => self.finish(&txn, true)?,
                        Ok(()) => {}
                        Err(e) => {
                            self.finish(&txn, false)?;
                            if autocommit {
                                return Err(format!("{e} (rolled back)"));
                            }
                            self.txn = Some(ExplicitTxn::Failed);
                            return Err(format!("{e} (transaction aborted; ROLLBACK to end it)"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn run_statement(&self, txn: &Arc<Transaction>, stmt: &Statement) -> Result<(), String> {
        let bound = Binder::new(self.catalog)
            .bind_statement(stmt)
            .map_err(|e| format!("bind: {}", e.0))?;
        let plan = Planner::new(self.catalog)
            .plan(bound)
            .map_err(|e| format!("plan: {}", e.0))?;
        let ctx = ExecutorContext::new(
            self.catalog,
            self.bpm.clone(),
            txn.clone(),
            self.txn_mgr.clone(),
        );
        let mut exec = create_executor(&ctx, &plan);
        exec.open().map_err(|e| format!("open: {}", e.0))?;
        execute(&plan, &mut *exec)?;
        exec.close().map_err(|e| format!("close: {}", e.0))?;
        Ok(())
    }

    /// Commits or aborts `txn`, then collects garbage. The outcome is settled before
    /// collection runs, so a collection failure is only a warning.
    fn finish(&self, txn: &Transaction, commit: bool) -> Result<(), String> {
        if commit {
            if let Err(e) = self.txn_mgr.commit(txn) {
                self.txn_mgr.abort(txn).map_err(|e| format!("abort: {e}"))?;
                return Err(format!("commit: {e} (transaction rolled back)"));
            }
        } else {
            self.txn_mgr.abort(txn).map_err(|e| format!("abort: {e}"))?;
        }
        if let Err(e) = self.txn_mgr.garbage_collect() {
            eprintln!("warning: garbage collection failed: {e}");
        }
        Ok(())
    }
}

fn execute(plan: &Plan, exec: &mut dyn Executor) -> Result<(), String> {
    match plan {
        Plan::Insert(_) => {
            if let Some((tup, _)) = RowCursor::new(exec)
                .next_row()
                .map_err(|e| format!("exec: {}", e.0))?
            {
                let n = i32::from_ne_bytes(tup.data.as_slice().try_into().unwrap_or([0, 0, 0, 0]));
                println!("INSERT {n}");
            }
        }
        Plan::Update(_) => {
            let mut n = 0;
            while let Some(batch) = exec.next_batch().map_err(|e| format!("exec: {}", e.0))? {
                n += batch.len();
            }
            println!("UPDATE {n}");
        }
        Plan::Delete(_) => {
            let mut n = 0;
            while let Some(batch) = exec.next_batch().map_err(|e| format!("exec: {}", e.0))? {
                n += batch.len();
            }
            println!("DELETE {n}");
        }
        Plan::CreateTable(p) => {
            exec.next_batch().map_err(|e| format!("exec: {}", e.0))?;
            println!("CREATE TABLE {}", p.table_name);
        }
        Plan::CreateIndex(p) => {
            exec.next_batch().map_err(|e| format!("exec: {}", e.0))?;
            println!("CREATE INDEX {}", p.index_name);
        }
        Plan::DropTable(_) => {
            exec.next_batch().map_err(|e| format!("exec: {}", e.0))?;
            println!("DROP TABLE");
        }
        Plan::DropIndex(_) => {
            exec.next_batch().map_err(|e| format!("exec: {}", e.0))?;
            println!("DROP INDEX");
        }
        _ => print_rows(exec)?,
    }
    Ok(())
}

fn print_rows(exec: &mut dyn Executor) -> Result<(), String> {
    let schema: Schema = exec.schema().clone();
    let headers: Vec<String> = schema.cols.iter().map(|c| c.name.clone()).collect();

    let mut rows: Vec<Vec<String>> = Vec::new();
    for row in RowCursor::new(exec) {
        let (tup, _) = row.map_err(|e| format!("exec: {}", e.0))?;
        let mut row = Vec::with_capacity(schema.cols.len());
        for i in 0..schema.cols.len() {
            let cell = match schema.get_value(&tup, i) {
                Ok(Some(v)) => format_value(&v),
                Ok(None) => "NULL".to_string(),
                Err(e) => format!("<err {:?}>", e),
            };
            row.push(cell);
        }
        rows.push(row);
    }

    let widths: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let cell_max = rows.iter().map(|r| r[i].len()).max().unwrap_or(0);
            h.len().max(cell_max)
        })
        .collect();

    let sep = format!(
        "+{}+",
        widths
            .iter()
            .map(|w| "-".repeat(w + 2))
            .collect::<Vec<_>>()
            .join("+")
    );

    println!("{sep}");
    println!(
        "|{}|",
        headers
            .iter()
            .zip(&widths)
            .map(|(h, w)| format!(" {:<width$} ", h, width = w))
            .collect::<Vec<_>>()
            .join("|")
    );
    println!("{sep}");
    for row in &rows {
        println!(
            "|{}|",
            row.iter()
                .zip(&widths)
                .map(|(c, w)| format!(" {:<width$} ", c, width = w))
                .collect::<Vec<_>>()
                .join("|")
        );
    }
    println!("{sep}");
    println!(
        "({} row{})",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    );
    Ok(())
}

fn format_value(v: &Value) -> String {
    match v {
        Value::BOOLEAN(b) => b.to_string(),
        Value::INT(i) => i.to_string(),
        Value::FLOAT(f) => f.into_inner().to_string(),
        Value::TIMESTAMP(t) => t.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_input_line_can_create_a_table_and_then_use_it() {
        let dir = tempdir().expect("tempdir");
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk");
        let bpm = Arc::new(create_buffer_pool_manager(
            16,
            ArcReplacer::new(16),
            DiskScheduler::new(dm),
        ));
        let catalog = Catalog::new(bpm.clone());
        let mut session = Session {
            catalog: &catalog,
            bpm: bpm.clone(),
            txn_mgr: Arc::new(TransactionManager::new(bpm.clone())),
            txn: None,
        };

        let result = session.run(
            &GenericDialect {},
            "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1); SELECT * FROM t;",
        );

        assert_eq!(result, Ok(()));
        let oid = catalog.table_oid("t").expect("table t was created");
        assert_eq!(catalog.get_table(oid).unwrap().heap.iter().count(), 1);
    }
}
