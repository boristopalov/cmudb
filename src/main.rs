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
#[cfg(test)]
use tempfile::TempDir;
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

/// The transaction opened by `BEGIN`, or one whose rollback has yet to succeed.
enum ExplicitTxn {
    Open(Arc<Transaction>),
    /// A statement failed. Its writes are already rolled back. As in Postgres, every later
    /// statement is refused until the user ends the block with `ROLLBACK` (or `COMMIT`, which
    /// rolls back too), so a partial transaction cannot pass for a whole one.
    Failed,
    /// Rolling `txn` back failed (a buffer pool error), so its rows are still locked. The
    /// session keeps it and retries the abort before the next statement, until it succeeds.
    /// `explicit` says whether it came from `BEGIN`, in which case the block then becomes
    /// `Failed` and still needs a `ROLLBACK`.
    AbortPending {
        txn: Arc<Transaction>,
        explicit: bool,
    },
}

/// Why `Session::finish` failed.
enum FinishError {
    /// The transaction is rolled back and its rows released.
    RolledBack(String),
    /// The abort failed too, so the transaction still holds its rows.
    AbortFailed(String),
}

impl Session<'_> {
    /// Runs every statement in `sql`. A failed statement aborts its whole transaction right
    /// away, releasing its rows and its snapshot, since there is no statement-level rollback.
    fn run(&mut self, dialect: &GenericDialect, sql: &str) -> Result<(), String> {
        let ast = Parser::parse_sql(dialect, sql).map_err(|e| format!("parse: {e}"))?;
        for stmt in ast {
            self.retry_pending_abort()?;
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
                Statement::Commit { .. } | Statement::Rollback { .. } => {
                    let commit = matches!(stmt, Statement::Commit { .. });
                    match self.txn.take() {
                        Some(ExplicitTxn::Open(txn)) => {
                            self.finish_or_keep(txn, commit, true)?;
                            println!("{}", if commit { "COMMIT" } else { "ROLLBACK" });
                        }
                        Some(ExplicitTxn::Failed) => println!("ROLLBACK"),
                        Some(pending @ ExplicitTxn::AbortPending { .. }) => {
                            self.txn = Some(pending);
                            return Err("rollback of the transaction is still pending".to_string());
                        }
                        None => return Err("no transaction in progress".to_string()),
                    }
                }
                stmt => {
                    let (txn, autocommit) = match &self.txn {
                        Some(ExplicitTxn::Open(txn)) => (txn.clone(), false),
                        Some(ExplicitTxn::Failed | ExplicitTxn::AbortPending { .. }) => {
                            return Err("current transaction is aborted, statements are \
                                        ignored until ROLLBACK"
                                .to_string());
                        }
                        None => (self.txn_mgr.begin(IsolationLevel::SnapshotIsolation), true),
                    };
                    match self.run_statement(&txn, &stmt) {
                        Ok(()) if autocommit => self.finish_or_keep(txn, true, false)?,
                        Ok(()) => {}
                        Err(e) => {
                            if !autocommit {
                                self.txn = Some(ExplicitTxn::Failed);
                            }
                            self.finish_or_keep(txn, false, !autocommit)?;
                            if autocommit {
                                return Err(format!("{e} (rolled back)"));
                            }
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

    /// Aborts the transaction whose earlier rollback failed. Until that succeeds no other
    /// statement runs, since the transaction still holds its rows.
    fn retry_pending_abort(&mut self) -> Result<(), String> {
        let Some(ExplicitTxn::AbortPending { txn, explicit }) = &self.txn else {
            return Ok(());
        };
        self.txn_mgr
            .abort(txn)
            .map_err(|e| format!("rollback of the earlier transaction failed again: {e}"))?;
        self.txn = explicit.then_some(ExplicitTxn::Failed);
        Ok(())
    }

    /// Runs `finish`, and keeps `txn` in the session as `AbortPending` if its abort failed,
    /// so that its rows are released by a later retry rather than held forever.
    fn finish_or_keep(
        &mut self,
        txn: Arc<Transaction>,
        commit: bool,
        explicit: bool,
    ) -> Result<(), String> {
        match self.finish(&txn, commit) {
            Ok(()) => Ok(()),
            Err(FinishError::RolledBack(e)) => Err(e),
            Err(FinishError::AbortFailed(e)) => {
                self.txn = Some(ExplicitTxn::AbortPending { txn, explicit });
                Err(format!(
                    "{e} (rollback will be retried before the next statement)"
                ))
            }
        }
    }

    /// Commits or aborts `txn`, then collects garbage. The outcome is settled before
    /// collection runs, so a collection failure is only a warning.
    fn finish(&self, txn: &Transaction, commit: bool) -> Result<(), FinishError> {
        if commit {
            if let Err(e) = self.txn_mgr.commit(txn) {
                self.txn_mgr
                    .abort(txn)
                    .map_err(|a| FinishError::AbortFailed(format!("commit: {e}; abort: {a}")))?;
                return Err(FinishError::RolledBack(format!(
                    "commit: {e} (transaction rolled back)"
                )));
            }
        } else {
            self.txn_mgr
                .abort(txn)
                .map_err(|e| FinishError::AbortFailed(format!("abort: {e}")))?;
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

    fn small_pool(frames: usize) -> (Arc<cmudb::buffer_pool::BufferPoolManager>, TempDir) {
        let dir = tempdir().expect("tempdir");
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk");
        let bpm = Arc::new(create_buffer_pool_manager(
            frames,
            ArcReplacer::new(frames),
            DiskScheduler::new(dm),
        ));
        (bpm, dir)
    }

    fn value_of_a(catalog: &Catalog, mgr: &TransactionManager) -> Vec<u8> {
        let table = catalog.get_table(catalog.table_oid("t").unwrap()).unwrap();
        let (_, rid) = table.heap.iter().next().unwrap().unwrap();
        let t = mgr.begin(IsolationLevel::SnapshotIsolation);
        let row = cmudb::concurrency::mvcc::read_visible(&t, mgr, &table.heap, rid)
            .unwrap()
            .unwrap();
        mgr.commit(&t).unwrap();
        row.data
    }

    /// Pinning every frame leaves no room to latch the table page, so both the commit and the
    /// abort after it fail. The session keeps the transaction and releases its row on retry.
    /// The autocommit case commits through `finish_or_keep` directly, since the frames have to
    /// be pinned between the statement and its commit.
    fn commit_and_abort_fail_then_retry(explicit: bool) {
        let (bpm, _dir) = small_pool(4);
        let catalog = Catalog::new(bpm.clone());
        let mgr = Arc::new(TransactionManager::new(bpm.clone()));
        let mut session = Session {
            catalog: &catalog,
            bpm: bpm.clone(),
            txn_mgr: mgr.clone(),
            txn: None,
        };
        let sql = |session: &mut Session, q: &str| session.run(&GenericDialect {}, q);
        sql(
            &mut session,
            "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);",
        )
        .unwrap();
        let before = value_of_a(&catalog, &mgr);

        let update = "UPDATE t SET a = 2;";
        let autocommit_txn = if explicit {
            sql(&mut session, "BEGIN;").unwrap();
            sql(&mut session, update).unwrap();
            None
        } else {
            let txn = mgr.begin(IsolationLevel::SnapshotIsolation);
            let stmt = Parser::parse_sql(&GenericDialect {}, update).unwrap();
            session.run_statement(&txn, &stmt[0]).unwrap();
            Some(txn)
        };
        let pins: Vec<_> = std::iter::from_fn(|| bpm.new_page().ok()).collect();
        let failed = match autocommit_txn {
            None => sql(&mut session, "COMMIT;"),
            Some(txn) => session.finish_or_keep(txn, true, false),
        };
        assert!(failed.is_err(), "commit fails with every frame pinned");
        assert!(
            matches!(session.txn, Some(ExplicitTxn::AbortPending { .. })),
            "the session keeps the transaction whose abort failed"
        );
        assert!(
            sql(&mut session, "SELECT * FROM t;").is_err(),
            "nothing runs while the retried abort keeps failing"
        );
        drop(pins);

        if explicit {
            assert!(
                sql(&mut session, "SELECT * FROM t;").is_err(),
                "after the abort succeeds the block still needs ROLLBACK"
            );
            sql(&mut session, "ROLLBACK;").unwrap();
        } else {
            sql(&mut session, "SELECT * FROM t;").expect("the retried abort succeeds");
        }
        assert!(session.txn.is_none());
        assert_eq!(
            value_of_a(&catalog, &mgr),
            before,
            "the update was rolled back"
        );
        sql(&mut session, "UPDATE t SET a = 3;").expect("the row is writable again");
        assert_eq!(
            mgr.watermark(),
            mgr.last_commit_ts(),
            "no transaction is left running"
        );
    }

    #[test]
    fn failed_explicit_commit_keeps_the_transaction_until_rollback_succeeds() {
        commit_and_abort_fail_then_retry(true);
    }

    #[test]
    fn failed_autocommit_keeps_the_transaction_until_its_abort_succeeds() {
        commit_and_abort_fail_then_retry(false);
    }
}
