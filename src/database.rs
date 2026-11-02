//! Runs SQL text against one database. Each statement is parsed, bound, planned and executed
//! inside the transaction `BEGIN` opened, or else in a transaction of its own (autocommit).

use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Catalog, Schema, Value};
use crate::concurrency::{IsolationLevel, Transaction, TransactionManager, TxnError};
use crate::create_buffer_pool_manager;
use crate::disk::{DiskManager, DiskScheduler};
use crate::parser::{Binder, BinderError};
use crate::planner::{Planner, PlannerError};
use crate::processor::executor::{
    DEFAULT_SORT_MEMORY_BUDGET_BYTES, ExecutorContext, ExecutorError, collect_rows, rows_affected,
};
use crate::processor::executor_factory::create_executor;
use crate::processor::plan::Plan;
use crate::replacer::ArcReplacer;
use parking_lot::Mutex;
use sqlparser::ast::Statement;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserError};
use std::fmt;
use std::io;
use std::sync::Arc;
use tempfile::TempDir;

/// One decoded result row, a value or `None` (SQL NULL) per column.
pub type Row = Vec<Option<Value>>;

/// What one statement produced.
#[derive(Debug)]
pub enum StatementResult {
    /// A query's rows, decoded with `schema`.
    Rows {
        schema: Schema,
        rows: Vec<Row>,
    },
    /// How many rows an INSERT, UPDATE or DELETE changed.
    Count {
        op: DmlOp,
        rows: u64,
    },
    Ddl(DdlOp),
    Txn(TxnOp),
}

/// The statement behind a [`StatementResult::Count`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmlOp {
    Insert,
    Update,
    Delete,
}

/// The schema change behind a [`StatementResult::Ddl`], with the name it created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DdlOp {
    CreateTable(String),
    CreateIndex(String),
    DropTable,
    DropIndex,
}

/// The transaction statement behind a [`StatementResult::Txn`]. A `COMMIT` of a block that
/// already failed rolls it back and reports `Rollback`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnOp {
    Begin,
    Commit,
    Rollback,
}

impl fmt::Display for DmlOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DmlOp::Insert => "INSERT",
            DmlOp::Update => "UPDATE",
            DmlOp::Delete => "DELETE",
        })
    }
}

impl fmt::Display for TxnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            TxnOp::Begin => "BEGIN",
            TxnOp::Commit => "COMMIT",
            TxnOp::Rollback => "ROLLBACK",
        })
    }
}

/// Why running SQL failed.
#[derive(Debug)]
pub enum QueryError {
    Parse(ParserError),
    Bind(BinderError),
    Plan(PlannerError),
    Execute(ExecutorError),
    /// `BEGIN` while a transaction block is open.
    AlreadyInTransaction,
    /// `COMMIT` or `ROLLBACK` with no transaction block open.
    NoTransaction,
    /// A statement in a block that already failed. Only `ROLLBACK` (or `COMMIT`, which rolls
    /// back too) ends the block.
    TransactionAborted,
    /// `COMMIT` or `ROLLBACK` while the abort of a failed transaction is still to be retried.
    RollbackPending,
    /// `BEGIN`, `COMMIT` or `ROLLBACK` passed to [`Database::execute_in`], whose caller owns
    /// the transaction.
    TransactionStatementInExecuteIn,
    /// Committing failed, so the transaction was rolled back instead.
    Commit(TxnError),
    /// A statement failed and its transaction was rolled back. `in_block` says whether the
    /// transaction came from `BEGIN`, in which case the block waits for `ROLLBACK`.
    RolledBack {
        cause: Box<QueryError>,
        in_block: bool,
    },
    /// Rolling the transaction back failed (after `commit` failed, if it is set), so it still
    /// holds its rows. The abort is retried before the next statement.
    AbortFailed {
        commit: Option<TxnError>,
        abort: TxnError,
    },
    /// Retrying the abort of an earlier transaction failed again.
    RetryAbortFailed(TxnError),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryError::Parse(e) => write!(f, "parse: {e}"),
            QueryError::Bind(e) => write!(f, "bind: {}", e.0),
            QueryError::Plan(e) => write!(f, "plan: {}", e.0),
            QueryError::Execute(e) => write!(f, "exec: {}", e.0),
            QueryError::AlreadyInTransaction => f.write_str("a transaction is already in progress"),
            QueryError::NoTransaction => f.write_str("no transaction in progress"),
            QueryError::TransactionAborted => {
                f.write_str("current transaction is aborted, statements are ignored until ROLLBACK")
            }
            QueryError::RollbackPending => {
                f.write_str("rollback of the transaction is still pending")
            }
            QueryError::TransactionStatementInExecuteIn => {
                f.write_str("BEGIN, COMMIT and ROLLBACK cannot run inside a caller's transaction")
            }
            QueryError::Commit(e) => write!(f, "commit: {e} (transaction rolled back)"),
            QueryError::RolledBack {
                cause,
                in_block: false,
            } => write!(f, "{cause} (rolled back)"),
            QueryError::RolledBack {
                cause,
                in_block: true,
            } => write!(f, "{cause} (transaction aborted; ROLLBACK to end it)"),
            QueryError::AbortFailed { commit, abort } => {
                if let Some(commit) = commit {
                    write!(f, "commit: {commit}; ")?;
                }
                write!(
                    f,
                    "abort: {abort} (rollback will be retried before the next statement)"
                )
            }
            QueryError::RetryAbortFailed(e) => {
                write!(f, "rollback of the earlier transaction failed again: {e}")
            }
        }
    }
}

impl std::error::Error for QueryError {}

/// The transaction block `BEGIN` opened, or a transaction whose rollback has yet to succeed.
enum ExplicitTxn {
    Open(Arc<Transaction>),
    /// A statement in the block failed and its writes are rolled back. As in Postgres, every
    /// later statement is refused until `ROLLBACK` (or `COMMIT`, which rolls back too), so a
    /// partial transaction cannot pass for a whole one.
    Failed,
    /// Rolling `txn` back failed (a buffer pool error), so its rows are still locked. The
    /// abort is retried before the next statement until it succeeds. `in_block` says whether
    /// `txn` came from `BEGIN`, in which case the block then becomes `Failed` and still needs
    /// a `ROLLBACK`.
    AbortPending {
        txn: Arc<Transaction>,
        in_block: bool,
    },
}

/// No transaction block is open.
type Session = Option<ExplicitTxn>;

/// A database: its buffer pool, catalog and transaction manager, plus the transaction block
/// that `BEGIN` opened, if any.
///
/// A `Database` is one session. Every caller of [`Database::execute`] shares that block, so
/// two threads that both call it run inside the same `BEGIN`. To give a caller a transaction
/// of its own, begin one on [`Database::txn_mgr`] and run SQL with [`Database::execute_in`].
pub struct Database {
    catalog: Catalog,
    bpm: Arc<BufferPoolManager>,
    txn_mgr: Arc<TransactionManager>,
    session: Mutex<Session>,
    sort_memory_budget_bytes: usize,
    _dir: Option<TempDir>,
}

/// Database file names inside the directory [`Database::scratch`] creates.
const DATA_FILE: &str = "cmudb.data";
const LOG_FILE: &str = "cmudb.log";

/// A buffer pool of `frames` frames over files in a new scratch directory, which is
/// deleted when the returned `TempDir` drops.
pub(crate) fn scratch_pool(frames: usize) -> io::Result<(Arc<BufferPoolManager>, TempDir)> {
    let dir = tempfile::tempdir()?;
    let dm = DiskManager::new(dir.path().join(DATA_FILE), dir.path().join(LOG_FILE))?;
    let bpm = create_buffer_pool_manager(frames, ArcReplacer::new(frames), DiskScheduler::new(dm));
    Ok((Arc::new(bpm), dir))
}

impl Database {
    /// An empty database over `bpm`.
    pub fn new(bpm: Arc<BufferPoolManager>) -> Self {
        Self {
            catalog: Catalog::new(bpm.clone()),
            txn_mgr: Arc::new(TransactionManager::new(bpm.clone())),
            bpm,
            session: Mutex::new(None),
            sort_memory_budget_bytes: DEFAULT_SORT_MEMORY_BUDGET_BYTES,
            _dir: None,
        }
    }

    /// The same database with `ORDER BY` sorting up to `bytes` of tuple data in memory and
    /// spilling sorted runs to scratch pages past that. Defaults to
    /// [`DEFAULT_SORT_MEMORY_BUDGET_BYTES`].
    pub fn with_sort_memory_budget(self, bytes: usize) -> Self {
        Self {
            sort_memory_budget_bytes: bytes,
            ..self
        }
    }

    /// An empty database with a `frames`-frame buffer pool, stored in a new scratch directory
    /// that is deleted when the database drops.
    pub fn scratch(frames: usize) -> io::Result<Self> {
        let (bpm, dir) = scratch_pool(frames)?;
        Ok(Self {
            _dir: Some(dir),
            ..Self::new(bpm)
        })
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn bpm(&self) -> &Arc<BufferPoolManager> {
        &self.bpm
    }

    pub fn txn_mgr(&self) -> &Arc<TransactionManager> {
        &self.txn_mgr
    }

    /// Runs every statement in `sql` and returns their results in order. Stops at the first
    /// statement that fails; the statements before it keep their effects.
    pub fn execute(&self, sql: &str) -> Result<Vec<StatementResult>, QueryError> {
        let mut results = Vec::new();
        self.execute_with(sql, |result| results.push(result))?;
        Ok(results)
    }

    /// Like [`Database::execute`], but hands each statement's result to `on_result` as soon
    /// as that statement finishes. `on_result` runs with the session unlocked, so it can run
    /// more SQL on this database.
    pub fn execute_with(
        &self,
        sql: &str,
        mut on_result: impl FnMut(StatementResult),
    ) -> Result<(), QueryError> {
        for stmt in parse(sql)? {
            let result = self.execute_statement(&mut self.session.lock(), stmt)?;
            on_result(result);
        }
        Ok(())
    }

    fn execute_statement(
        &self,
        session: &mut Session,
        stmt: Statement,
    ) -> Result<StatementResult, QueryError> {
        self.retry_pending_abort(session)?;
        match stmt {
            Statement::StartTransaction { .. } => self.begin_block(session),
            Statement::Commit { .. } => self.end_block(session, true),
            Statement::Rollback { .. } => self.end_block(session, false),
            stmt => self.run_in_session(session, |txn| self.run_statement(txn, &stmt)),
        }
    }

    /// Runs `plan` the way [`Database::execute`] runs a statement: in the open transaction
    /// block, or else in a transaction of its own.
    pub fn execute_plan(&self, plan: &Plan) -> Result<StatementResult, QueryError> {
        let mut session = self.session.lock();
        self.retry_pending_abort(&mut session)?;
        self.run_in_session(&mut session, |txn| {
            self.run_plan(txn, plan).map_err(QueryError::Execute)
        })
    }

    /// Runs every statement in `sql` inside `txn`, which the caller commits or aborts. The
    /// session's transaction block is left alone.
    ///
    /// Fails with [`QueryError::TransactionStatementInExecuteIn`], before running anything,
    /// if `sql` holds a `BEGIN`, `COMMIT` or `ROLLBACK`.
    pub fn execute_in(
        &self,
        txn: &Arc<Transaction>,
        sql: &str,
    ) -> Result<Vec<StatementResult>, QueryError> {
        let statements = parse(sql)?;
        if statements.iter().any(is_transaction_statement) {
            return Err(QueryError::TransactionStatementInExecuteIn);
        }
        statements
            .iter()
            .map(|stmt| self.run_statement(txn, stmt))
            .collect()
    }

    /// The plan of each statement in `sql`, all bound against the catalog as it is now.
    pub fn plan(&self, sql: &str) -> Result<Vec<Plan>, QueryError> {
        parse(sql)?
            .iter()
            .map(|stmt| self.plan_statement(stmt))
            .collect()
    }

    fn plan_statement(&self, stmt: &Statement) -> Result<Plan, QueryError> {
        let bound = Binder::new(&self.catalog)
            .bind_statement(stmt)
            .map_err(QueryError::Bind)?;
        Planner::new(&self.catalog)
            .plan(bound)
            .map_err(QueryError::Plan)
    }

    fn run_statement(
        &self,
        txn: &Arc<Transaction>,
        stmt: &Statement,
    ) -> Result<StatementResult, QueryError> {
        let plan = self.plan_statement(stmt)?;
        self.run_plan(txn, &plan).map_err(QueryError::Execute)
    }

    fn run_plan(
        &self,
        txn: &Arc<Transaction>,
        plan: &Plan,
    ) -> Result<StatementResult, ExecutorError> {
        let ctx = ExecutorContext::new(
            &self.catalog,
            self.bpm.clone(),
            txn.clone(),
            self.txn_mgr.clone(),
        )
        .with_sort_memory_budget(self.sort_memory_budget_bytes);
        let mut exec = create_executor(&ctx, plan);
        exec.open()?;
        let result = match plan {
            Plan::Insert(_) => StatementResult::Count {
                op: DmlOp::Insert,
                rows: rows_affected(plan, &mut *exec)?,
            },
            Plan::Update(_) => StatementResult::Count {
                op: DmlOp::Update,
                rows: rows_affected(plan, &mut *exec)?,
            },
            Plan::Delete(_) => StatementResult::Count {
                op: DmlOp::Delete,
                rows: rows_affected(plan, &mut *exec)?,
            },
            Plan::CreateTable(p) => {
                exec.next_batch()?;
                StatementResult::Ddl(DdlOp::CreateTable(p.table_name.clone()))
            }
            Plan::CreateIndex(p) => {
                exec.next_batch()?;
                StatementResult::Ddl(DdlOp::CreateIndex(p.index_name.clone()))
            }
            Plan::DropTable(_) => {
                exec.next_batch()?;
                StatementResult::Ddl(DdlOp::DropTable)
            }
            Plan::DropIndex(_) => {
                exec.next_batch()?;
                StatementResult::Ddl(DdlOp::DropIndex)
            }
            _ => {
                let schema = exec.schema().clone();
                let rows = collect_rows(&mut *exec)?
                    .into_iter()
                    .map(|(tuple, _)| {
                        (0..schema.cols.len())
                            .map(|i| schema.get_value(&tuple, i))
                            .collect::<Result<Row, _>>()
                    })
                    .collect::<Result<Vec<Row>, _>>()?;
                StatementResult::Rows { schema, rows }
            }
        };
        exec.close()?;
        Ok(result)
    }

    fn begin_block(&self, session: &mut Session) -> Result<StatementResult, QueryError> {
        if session.is_some() {
            return Err(QueryError::AlreadyInTransaction);
        }
        *session = Some(ExplicitTxn::Open(
            self.txn_mgr.begin(IsolationLevel::SnapshotIsolation),
        ));
        Ok(StatementResult::Txn(TxnOp::Begin))
    }

    fn end_block(
        &self,
        session: &mut Session,
        commit: bool,
    ) -> Result<StatementResult, QueryError> {
        match session.take() {
            Some(ExplicitTxn::Open(txn)) => {
                self.finish_or_keep(session, txn, commit, true)?;
                Ok(StatementResult::Txn(if commit {
                    TxnOp::Commit
                } else {
                    TxnOp::Rollback
                }))
            }
            Some(ExplicitTxn::Failed) => Ok(StatementResult::Txn(TxnOp::Rollback)),
            Some(pending @ ExplicitTxn::AbortPending { .. }) => {
                *session = Some(pending);
                Err(QueryError::RollbackPending)
            }
            None => Err(QueryError::NoTransaction),
        }
    }

    /// Runs `body` in the open transaction block, or else in a transaction of its own that
    /// commits when `body` succeeds. A failure aborts the whole transaction right away,
    /// releasing its rows and its snapshot, since there is no statement-level rollback.
    fn run_in_session(
        &self,
        session: &mut Session,
        body: impl FnOnce(&Arc<Transaction>) -> Result<StatementResult, QueryError>,
    ) -> Result<StatementResult, QueryError> {
        let (txn, autocommit) = match session {
            Some(ExplicitTxn::Open(txn)) => (txn.clone(), false),
            Some(ExplicitTxn::Failed | ExplicitTxn::AbortPending { .. }) => {
                return Err(QueryError::TransactionAborted);
            }
            None => (self.txn_mgr.begin(IsolationLevel::SnapshotIsolation), true),
        };
        match body(&txn) {
            Ok(result) => {
                if autocommit {
                    self.finish_or_keep(session, txn, true, false)?;
                }
                Ok(result)
            }
            Err(cause) => {
                if !autocommit {
                    *session = Some(ExplicitTxn::Failed);
                }
                self.finish_or_keep(session, txn, false, !autocommit)?;
                Err(QueryError::RolledBack {
                    cause: Box::new(cause),
                    in_block: !autocommit,
                })
            }
        }
    }

    /// Aborts the transaction whose earlier rollback failed. Until that succeeds no other
    /// statement runs, since the transaction still holds its rows.
    fn retry_pending_abort(&self, session: &mut Session) -> Result<(), QueryError> {
        let Some(ExplicitTxn::AbortPending { txn, in_block }) = session else {
            return Ok(());
        };
        self.txn_mgr
            .abort(txn)
            .map_err(QueryError::RetryAbortFailed)?;
        let in_block = *in_block;
        *session = in_block.then_some(ExplicitTxn::Failed);
        Ok(())
    }

    /// Runs [`Database::finish`], and keeps `txn` in the session as `AbortPending` if its
    /// abort failed, so that a later retry releases its rows.
    fn finish_or_keep(
        &self,
        session: &mut Session,
        txn: Arc<Transaction>,
        commit: bool,
        in_block: bool,
    ) -> Result<(), QueryError> {
        match self.finish(&txn, commit) {
            Err(e @ QueryError::AbortFailed { .. }) => {
                *session = Some(ExplicitTxn::AbortPending { txn, in_block });
                Err(e)
            }
            other => other,
        }
    }

    /// Commits or aborts `txn`, then collects garbage. The outcome is settled before
    /// collection runs, so a collection failure is only logged.
    fn finish(&self, txn: &Transaction, commit: bool) -> Result<(), QueryError> {
        if commit {
            if let Err(commit) = self.txn_mgr.commit(txn) {
                return Err(match self.txn_mgr.abort(txn) {
                    Ok(()) => QueryError::Commit(commit),
                    Err(abort) => QueryError::AbortFailed {
                        commit: Some(commit),
                        abort,
                    },
                });
            }
        } else {
            self.txn_mgr
                .abort(txn)
                .map_err(|abort| QueryError::AbortFailed {
                    commit: None,
                    abort,
                })?;
        }
        if let Err(e) = self.txn_mgr.garbage_collect() {
            log::warn!("garbage collection failed: {e}");
        }
        Ok(())
    }
}

fn parse(sql: &str) -> Result<Vec<Statement>, QueryError> {
    Parser::parse_sql(&GenericDialect, sql).map_err(QueryError::Parse)
}

fn is_transaction_statement(stmt: &Statement) -> bool {
    matches!(
        stmt,
        Statement::StartTransaction { .. } | Statement::Commit { .. } | Statement::Rollback { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concurrency::{TransactionState, mvcc};
    use crate::testing;

    fn ints(result: &StatementResult) -> Vec<Vec<i32>> {
        let StatementResult::Rows { rows, .. } = result else {
            panic!("expected rows, got {result:?}");
        };
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

    #[test]
    fn one_call_can_create_a_table_and_then_use_it() {
        let db = Database::scratch(16).unwrap();

        let results = db
            .execute("CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2); SELECT * FROM t;")
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(matches!(
            &results[0],
            StatementResult::Ddl(DdlOp::CreateTable(name)) if name == "t"
        ));
        assert!(matches!(
            results[1],
            StatementResult::Count {
                op: DmlOp::Insert,
                rows: 2
            }
        ));
        assert_eq!(ints(&results[2]), vec![vec![1], vec![2]]);
    }

    #[test]
    fn update_and_delete_report_how_many_rows_they_changed() {
        let db = Database::scratch(16).unwrap();
        db.execute("CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2), (3);")
            .unwrap();

        let results = db
            .execute("UPDATE t SET a = a + 10 WHERE a > 1; DELETE FROM t WHERE a = 1;")
            .unwrap();

        assert!(matches!(
            results[..],
            [
                StatementResult::Count {
                    op: DmlOp::Update,
                    rows: 2
                },
                StatementResult::Count {
                    op: DmlOp::Delete,
                    rows: 1
                }
            ]
        ));
    }

    #[test]
    fn a_failed_statement_keeps_the_effects_of_the_ones_before_it() {
        let db = Database::scratch(16).unwrap();
        let mut seen = Vec::new();

        let err = db
            .execute_with(
                "CREATE TABLE t (a INTEGER); SELECT * FROM missing; INSERT INTO t VALUES (1);",
                |r| seen.push(r),
            )
            .unwrap_err();

        assert!(matches!(seen[..], [StatementResult::Ddl(_)]), "{seen:?}");
        assert!(
            matches!(&err, QueryError::RolledBack { cause, in_block: false } if matches!(**cause, QueryError::Bind(_))),
            "{err:?}"
        );
        assert_eq!(
            err.to_string(),
            "bind: table not found: missing (rolled back)"
        );
        assert!(db.catalog().table_oid("t").is_ok());
    }

    #[test]
    fn on_result_can_run_more_sql_on_the_same_database() {
        let db = Database::scratch(16).unwrap();
        let mut counts_seen_by_callback = Vec::new();

        db.execute_with(
            "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1), (2);",
            |result| {
                if let StatementResult::Count { .. } = result {
                    let follow_up = db.execute("SELECT * FROM t;").unwrap();
                    counts_seen_by_callback.push(ints(&follow_up[0]).len());
                }
            },
        )
        .unwrap();

        assert_eq!(counts_seen_by_callback, vec![2]);
    }

    #[test]
    fn a_rolled_back_block_leaves_no_rows_behind() {
        let db = Database::scratch(16).unwrap();
        db.execute("CREATE TABLE t (a INTEGER);").unwrap();

        let results = db
            .execute("BEGIN; INSERT INTO t VALUES (1); ROLLBACK; SELECT * FROM t;")
            .unwrap();

        assert!(matches!(results[0], StatementResult::Txn(TxnOp::Begin)));
        assert!(matches!(results[2], StatementResult::Txn(TxnOp::Rollback)));
        assert_eq!(ints(&results[3]), Vec::<Vec<i32>>::new());
    }

    #[test]
    fn a_failed_block_refuses_statements_until_it_ends() {
        let db = Database::scratch(16).unwrap();
        db.execute("CREATE TABLE t (a INTEGER); BEGIN;").unwrap();

        assert!(matches!(
            db.execute("SELECT * FROM missing;").unwrap_err(),
            QueryError::RolledBack { in_block: true, .. }
        ));
        assert!(matches!(
            db.execute("SELECT * FROM t;").unwrap_err(),
            QueryError::TransactionAborted
        ));
        assert!(matches!(
            db.execute("COMMIT;").unwrap()[..],
            [StatementResult::Txn(TxnOp::Rollback)]
        ));
        assert!(db.execute("SELECT * FROM t;").is_ok());
    }

    #[test]
    fn transaction_statements_out_of_place_are_refused() {
        let db = Database::scratch(16).unwrap();
        assert!(matches!(
            db.execute("COMMIT;").unwrap_err(),
            QueryError::NoTransaction
        ));
        db.execute("BEGIN;").unwrap();
        assert!(matches!(
            db.execute("BEGIN;").unwrap_err(),
            QueryError::AlreadyInTransaction
        ));
    }

    #[test]
    fn execute_in_leaves_the_transaction_to_the_caller() {
        let db = Database::scratch(16).unwrap();
        db.execute("CREATE TABLE t (a INTEGER);").unwrap();
        let writer = db.txn_mgr().begin(IsolationLevel::SnapshotIsolation);

        db.execute_in(&writer, "INSERT INTO t VALUES (7);").unwrap();

        assert_eq!(
            ints(&db.execute("SELECT * FROM t;").unwrap()[0]),
            Vec::<Vec<i32>>::new()
        );
        db.txn_mgr().commit(&writer).unwrap();
        assert_eq!(
            ints(&db.execute("SELECT * FROM t;").unwrap()[0]),
            vec![vec![7]]
        );
    }

    #[test]
    fn execute_in_rejects_transaction_statements_and_leaves_the_transaction_running() {
        let db = Database::scratch(16).unwrap();
        db.execute("CREATE TABLE t (a INTEGER);").unwrap();
        let txn = db.txn_mgr().begin(IsolationLevel::SnapshotIsolation);

        for sql in ["BEGIN;", "INSERT INTO t VALUES (1); COMMIT;", "ROLLBACK;"] {
            assert!(
                matches!(
                    db.execute_in(&txn, sql),
                    Err(QueryError::TransactionStatementInExecuteIn)
                ),
                "{sql}"
            );
        }

        assert_eq!(txn.state(), TransactionState::Running);
        assert!(
            ints(&db.execute_in(&txn, "SELECT * FROM t;").unwrap()[0]).is_empty(),
            "nothing in the rejected batch ran"
        );
        db.txn_mgr().commit(&txn).unwrap();
    }

    #[test]
    fn plan_returns_one_plan_per_statement() {
        let db = Database::scratch(16).unwrap();
        db.execute("CREATE TABLE t (a INTEGER);").unwrap();

        let plans = db.plan("SELECT * FROM t; DELETE FROM t;").unwrap();

        assert!(matches!(plans[..], [_, Plan::Delete(_)]));
    }

    fn value_of_a(db: &Database) -> Vec<u8> {
        let catalog = db.catalog();
        let table = catalog.get_table(catalog.table_oid("t").unwrap()).unwrap();
        let (_, rid) = table.heap.iter().next().unwrap().unwrap();
        let t = db.txn_mgr().begin(IsolationLevel::SnapshotIsolation);
        let row = mvcc::read_visible(&t, db.txn_mgr(), &table.heap, rid)
            .unwrap()
            .unwrap();
        db.txn_mgr().commit(&t).unwrap();
        row.data
    }

    /// Pinning every frame leaves no room to latch the table page, so both the commit and the
    /// abort after it fail. The session keeps the transaction and releases its row on retry.
    /// The autocommit case commits through `finish_or_keep` directly, since the frames have to
    /// be pinned between the statement and its commit.
    fn commit_and_abort_fail_then_retry(in_block: bool) {
        let (bpm, _dir) = testing::bpm(4);
        let db = Database::new(bpm.clone());
        db.execute("CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1);")
            .unwrap();
        let before = value_of_a(&db);

        let update = "UPDATE t SET a = 2;";
        let autocommit_txn = if in_block {
            db.execute("BEGIN;").unwrap();
            db.execute(update).unwrap();
            None
        } else {
            let txn = db.txn_mgr().begin(IsolationLevel::SnapshotIsolation);
            db.execute_in(&txn, update).unwrap();
            Some(txn)
        };
        let pins: Vec<_> = std::iter::from_fn(|| bpm.new_page().ok()).collect();
        let failed = match autocommit_txn {
            None => db.execute("COMMIT;").map(drop),
            Some(txn) => db.finish_or_keep(&mut db.session.lock(), txn, true, false),
        };
        assert!(
            matches!(
                failed,
                Err(QueryError::AbortFailed {
                    commit: Some(_),
                    ..
                })
            ),
            "commit fails with every frame pinned: {failed:?}"
        );
        assert!(
            matches!(*db.session.lock(), Some(ExplicitTxn::AbortPending { .. })),
            "the session keeps the transaction whose abort failed"
        );
        assert!(
            matches!(
                db.execute("SELECT * FROM t;"),
                Err(QueryError::RetryAbortFailed(_))
            ),
            "nothing runs while the retried abort keeps failing"
        );
        drop(pins);

        if in_block {
            assert!(
                matches!(
                    db.execute("SELECT * FROM t;"),
                    Err(QueryError::TransactionAborted)
                ),
                "after the abort succeeds the block still needs ROLLBACK"
            );
            db.execute("ROLLBACK;").unwrap();
        } else {
            db.execute("SELECT * FROM t;")
                .expect("the retried abort succeeds");
        }
        assert!(db.session.lock().is_none());
        assert_eq!(value_of_a(&db), before, "the update was rolled back");
        db.execute("UPDATE t SET a = 3;")
            .expect("the row is writable again");
        assert_eq!(
            db.txn_mgr().watermark(),
            db.txn_mgr().last_commit_ts(),
            "no transaction is left running"
        );
    }

    #[test]
    fn failed_block_commit_keeps_the_transaction_until_rollback_succeeds() {
        commit_and_abort_fail_then_retry(true);
    }

    #[test]
    fn failed_autocommit_keeps_the_transaction_until_its_abort_succeeds() {
        commit_and_abort_fail_then_retry(false);
    }
}
