use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;

use sqlparser::ast::{
    Assignment, AssignmentTarget, BinaryOperator, CharLengthUnits, CharacterLength,
    CreateIndex as SqlCreateIndex, CreateTable as SqlCreateTable, DataType as SqlDataType, Delete,
    Distinct, Expr, FromTable, Function, FunctionArg, FunctionArgExpr, FunctionArguments,
    GroupByExpr, Insert, JoinConstraint, JoinOperator, LimitClause, ObjectName, ObjectNamePart,
    ObjectType, OrderByExpr, OrderByKind, Query, Select, SelectItem, SetExpr, Statement,
    TableFactor, TableObject, TableWithJoins, TimezoneInfo, UnaryOperator, Update,
    Value as SqlValue, WildcardAdditionalOptions, WindowSpec, WindowType,
};

use crate::catalog::timestamp::{self, InvalidTimestamp};
use crate::catalog::{Catalog, CoercionError, Column, DataType, Schema, Value};
use crate::processor::plan::{
    AggType, ColIdx, Expression, JoinColumn, JoinPredicate, JoinType, NullOrdering, Op, Side,
    SortDirection, SortExpr,
};

/// Argument bound for `COUNT(*)`. COUNT skips NULL arguments, so a non-NULL constant makes it count
/// every row.
const COUNT_STAR_ARGUMENT: Expression = Expression::Constant(Some(Value::INT(1)));

/// Argument bound for `RANK()`, which reads no value: the window executor ranks rows by their
/// peer group.
const RANK_ARGUMENT: Expression = Expression::Constant(None);

/// Resolves parsed SQL statements against the catalog into bound statements for the planner.
pub struct Binder<'a> {
    pub catalog: &'a Catalog,
    /// Active aggregation context for the SELECT currently being resolved.
    /// `None` outside of an aggregating SELECT.
    agg_ctx: RefCell<Option<AggCtx>>,
    /// When set, `resolve_expr` is currently descending into an aggregate
    /// function's argument and should not perform group-by / aggregate
    /// rewrites — bare column refs are legal there.
    inside_agg_arg: Cell<bool>,
    /// Window function calls of the non-aggregating SELECT whose projection or ORDER BY is
    /// being resolved. `None` everywhere else, which is where window functions are rejected.
    window_ctx: RefCell<Option<WindowCtx>>,
}

/// Bookkeeping accumulated by the binder while resolving an aggregating
/// SELECT. Aggregates discovered inside projection / ORDER BY expressions
/// get appended here, and the original expressions are rewritten to refer
/// to the aggregation node's output columns: `[group_bys..., aggregates...]`.
struct AggCtx {
    /// The raw AST of each GROUP BY expression, used to match projection /
    /// ORDER BY expressions against group-bys via structural equality.
    group_by_exprs: Vec<Expr>,
    /// Resolved GROUP BY expressions — evaluated against the child schema.
    group_bys: Vec<Expression>,
    /// Discovered aggregate function calls, in projection / ORDER BY order.
    aggregates: Vec<(Expression, AggType)>,
}

/// Window function calls found while resolving a SELECT's projection and ORDER BY, grouped by
/// OVER clause. Each call first resolves to the placeholder column `input_width + n`, where `n`
/// counts the calls found before it; [`WindowCtx::output_columns`] maps each placeholder to the
/// call's column in the window output.
struct WindowCtx {
    /// Number of columns in the FROM output, which the window output starts with.
    input_width: usize,
    /// The OVER clause of each entry of `windows`, matched by structural equality so calls over
    /// the same window share one.
    specs: Vec<WindowSpec>,
    windows: Vec<BoundWindow>,
    /// For each call in discovery order, its window and its position among that window's calls.
    calls: Vec<(usize, usize)>,
}

impl WindowCtx {
    fn new(input_width: usize) -> Self {
        Self {
            input_width,
            specs: vec![],
            windows: vec![],
            calls: vec![],
        }
    }

    /// Where each call, in discovery order, lands in the window output: after the FROM columns
    /// and the calls of every earlier window.
    fn output_columns(&self) -> Vec<ColIdx> {
        let mut first_call_col = Vec::with_capacity(self.windows.len());
        let mut next = self.input_width;
        for w in &self.windows {
            first_call_col.push(next);
            next += w.calls.len();
        }
        self.calls
            .iter()
            .map(|&(window, pos)| ColIdx((first_call_col[window] + pos) as u32))
            .collect()
    }

    /// `expr` with each window call placeholder replaced by the call's output column.
    fn place_calls(&self, expr: Expression, output: &[ColIdx]) -> Expression {
        expr.map_columns(|col| match col.index().checked_sub(self.input_width) {
            Some(call) => output[call],
            None => col,
        })
    }
}

/// Why a parsed statement could not be bound against the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinderError {
    /// No table with this name is in the catalog.
    TableNotFound { table: String },
    /// A `table.column` reference names a table the statement does not read.
    TableNotInScope { table: String },
    /// No table in scope has a column with this name.
    ColumnNotFound { column: String },
    /// More than one table in scope has a column with this name.
    AmbiguousColumn { column: String },
    /// A column read by an aggregating SELECT is neither a GROUP BY expression nor inside an
    /// aggregate call.
    UngroupedColumn { column: String },
    /// An aggregate call appears outside a SELECT projection, such as in WHERE.
    AggregateOutsideProjection,
    /// An aggregate call appears inside another aggregate's argument.
    NestedAggregate,
    /// A window function call appears outside a SELECT projection and ORDER BY, such as in WHERE.
    WindowOutsideProjection,
    /// A window function call appears inside a function argument or an OVER clause.
    NestedWindowFunction,
    /// A function the window executor does not compute is called with OVER.
    UnknownWindowFunction { function: String },
    /// A window-only function such as RANK is called without OVER.
    WindowFunctionWithoutOver { function: String },
    /// A window function that reads no argument, such as RANK, is passed one.
    WindowFunctionArguments { function: String },
    /// An aggregate is called without a parenthesized argument list.
    AggregateWithoutArguments { function: String },
    /// An aggregate is called with other than one argument.
    AggregateArgumentCount { function: String, args: usize },
    /// `*` is passed to an aggregate other than COUNT.
    StarArgumentNotCount { function: String },
    /// An aggregate's argument has no type, such as a bare NULL.
    UntypedAggregateArgument { function: String },
    /// A GROUP BY expression has no type, such as a bare NULL.
    UntypedGroupBy,
    /// A SELECT projection is a bare NULL, whose output column has no type.
    BareNullProjection,
    /// A `SELECT DISTINCT` ORDER BY key that is not one of the select list's expressions.
    DistinctOrderByNotInSelectList,
    /// A numeric literal fits neither an INT, a BIGINT nor a finite FLOAT.
    InvalidNumber { literal: String },
    /// A TIMESTAMP literal that is not a valid date and time.
    InvalidTimestamp(InvalidTimestamp),
    /// An aggregate is called on a type it does not take, such as SUM of a BOOLEAN.
    AggregateArgumentType { function: String, arg: DataType },
    /// A binary operator applied to operand types it does not take, such as `TRUE + 1` or a
    /// TIMESTAMP compared with an INT. A `None` side is a bare NULL.
    OperandTypes {
        operator: BinaryOperator,
        left: Option<DataType>,
        right: Option<DataType>,
    },
    /// Unary `-` or `+` applied to a value that is not a number.
    NegateNonNumber { operand: DataType },
    /// An INSERT or UPDATE stores a value of type `found` in column `column` of type
    /// `expected`, with no implicit conversion between the two.
    ColumnType {
        column: String,
        expected: DataType,
        found: DataType,
    },
    /// An INSERT or UPDATE stores a constant the column's type cannot hold, such as
    /// 2147483648 in an INT column.
    ColumnOutOfRange {
        column: String,
        expected: DataType,
        value: Value,
    },
    /// An INSERT or UPDATE stores a string constant of `len` characters in a `VARCHAR(n)` or
    /// `CHAR(n)` column that holds fewer; a CHAR counts no trailing spaces.
    ColumnTooLong {
        column: String,
        expected: DataType,
        len: usize,
    },
    /// An INSERT row has other than one value per target column.
    InsertArity {
        table: String,
        columns: usize,
        values: usize,
    },
    /// An INSERT column list names the same column twice.
    DuplicateInsertColumn { column: String },
    /// A LIMIT literal is not a non-negative integer that fits a `u32`.
    InvalidLimit { limit: String },
    /// A table or index name has more than one part, such as `schema.table`.
    QualifiedName { name: String },
    /// A column type other than INT, BIGINT, FLOAT, BOOLEAN, TIMESTAMP (without time zone), TEXT,
    /// VARCHAR, VARCHAR(n) with `n` from 1 to `u32::MAX` characters, or CHAR and CHAR(n) with `n`
    /// from 1 to [`MAX_CHAR_LENGTH`] characters.
    UnsupportedType { type_name: String },
    /// A binary operator with no bound counterpart, such as `%`.
    UnsupportedOperator { operator: BinaryOperator },
    /// A literal other than a number, a string, a boolean or NULL, such as a dollar-quoted
    /// string.
    UnsupportedLiteral { literal: String },
    /// An expression form the binder does not handle, such as `IN (...)`.
    UnsupportedExpression { expr: String },
    /// No table has an index with this name.
    IndexNotFound { index: String },
    /// `DROP INDEX` without `ON` names an index that more than one table has.
    AmbiguousIndex { index: String },
    /// A DROP of an object other than a table or an index, such as a view.
    UnsupportedDropObject { object_type: String },
    /// A statement shape or clause the binder does not handle.
    Unsupported(Unsupported),
}

impl fmt::Display for BinderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinderError::TableNotFound { table } => write!(f, "table not found: {table}"),
            BinderError::TableNotInScope { table } => write!(f, "table not in scope: {table}"),
            BinderError::ColumnNotFound { column } => write!(f, "column not found: {column}"),
            BinderError::AmbiguousColumn { column } => write!(f, "ambiguous column: {column}"),
            BinderError::UngroupedColumn { column } => write!(
                f,
                "column `{column}` must appear in GROUP BY or be used inside an aggregate function"
            ),
            BinderError::AggregateOutsideProjection => {
                f.write_str("aggregate functions are only allowed in a SELECT projection")
            }
            BinderError::NestedAggregate => {
                f.write_str("nested aggregate functions are not allowed")
            }
            BinderError::WindowOutsideProjection => {
                f.write_str("window functions are only allowed in a SELECT projection or ORDER BY")
            }
            BinderError::NestedWindowFunction => f.write_str(
                "window functions cannot be nested inside a function argument or an OVER clause",
            ),
            BinderError::UnknownWindowFunction { function } => {
                write!(f, "unsupported window function: {function}")
            }
            BinderError::WindowFunctionWithoutOver { function } => {
                write!(f, "{function}() requires an OVER clause")
            }
            BinderError::WindowFunctionArguments { function } => {
                write!(f, "{function}() takes no arguments")
            }
            BinderError::AggregateWithoutArguments { function } => {
                write!(f, "aggregate `{function}` must be called with arguments")
            }
            BinderError::AggregateArgumentCount { function, args } => write!(
                f,
                "aggregate `{function}` expects exactly one argument, got {args}"
            ),
            BinderError::StarArgumentNotCount { function } => {
                write!(
                    f,
                    "`*` is only valid as an argument to COUNT, not {function}"
                )
            }
            BinderError::UntypedAggregateArgument { function } => {
                write!(
                    f,
                    "argument to aggregate `{function}` has no inferable type"
                )
            }
            BinderError::UntypedGroupBy => f.write_str("GROUP BY expression has no inferable type"),
            BinderError::BareNullProjection => {
                f.write_str("bare NULL not allowed in SELECT projection")
            }
            BinderError::InvalidNumber { literal } => {
                write!(f, "invalid numeric literal: {literal}")
            }
            BinderError::InvalidTimestamp(e) => write!(f, "{e}"),
            BinderError::AggregateArgumentType { function, arg } => {
                write!(f, "aggregate `{function}` does not take a {arg} argument")
            }
            BinderError::OperandTypes {
                operator,
                left,
                right,
            } => write!(
                f,
                "operator `{operator}` does not take {} and {}",
                type_name(*left),
                type_name(*right)
            ),
            BinderError::NegateNonNumber { operand } => {
                write!(f, "cannot negate a {operand}")
            }
            BinderError::ColumnType {
                column,
                expected,
                found,
            } => write!(f, "column `{column}` is {expected}, got {found}"),
            BinderError::ColumnOutOfRange {
                column,
                expected,
                value,
            } => write!(
                f,
                "{value} is out of range for {expected} column `{column}`"
            ),
            BinderError::ColumnTooLong {
                column,
                expected,
                len,
            } => write!(
                f,
                "value of {len} characters is too long for {expected} column `{column}`"
            ),
            BinderError::InsertArity {
                table,
                columns,
                values,
            } => write!(
                f,
                "INSERT into `{table}` has {values} values for {columns} columns"
            ),
            BinderError::DuplicateInsertColumn { column } => {
                write!(f, "column `{column}` is listed more than once in INSERT")
            }
            BinderError::DistinctOrderByNotInSelectList => f.write_str(
                "for SELECT DISTINCT, ORDER BY expressions must appear in the select list",
            ),
            BinderError::InvalidLimit { limit } => write!(f, "invalid LIMIT: {limit}"),
            BinderError::QualifiedName { name } => {
                write!(f, "qualified name not supported: {name}")
            }
            BinderError::UnsupportedType { type_name } => {
                write!(f, "unsupported type: {type_name}")
            }
            BinderError::UnsupportedOperator { operator } => {
                write!(f, "unsupported binary operator: {operator}")
            }
            BinderError::UnsupportedLiteral { literal } => {
                write!(f, "unsupported literal: {literal}")
            }
            BinderError::UnsupportedExpression { expr } => {
                write!(f, "unsupported expression: {expr}")
            }
            BinderError::IndexNotFound { index } => write!(f, "index not found: {index}"),
            BinderError::AmbiguousIndex { index } => write!(
                f,
                "index name {index} is ambiguous: more than one table has an index with that \
                 name, so name the table with DROP INDEX {index} ON <table>"
            ),
            BinderError::UnsupportedDropObject { object_type } => {
                write!(f, "DROP {object_type} is not supported")
            }
            BinderError::Unsupported(feature) => write!(f, "{feature}"),
        }
    }
}

impl std::error::Error for BinderError {}

/// A type's name, or `NULL` for a bare NULL.
fn type_name(t: Option<DataType>) -> String {
    t.map_or_else(|| "NULL".to_string(), |t| t.to_string())
}

impl From<Unsupported> for BinderError {
    fn from(feature: Unsupported) -> Self {
        BinderError::Unsupported(feature)
    }
}

/// A statement shape or clause the binder does not handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// A statement other than SELECT, VALUES, INSERT, DELETE, UPDATE, CREATE TABLE,
    /// CREATE INDEX, DROP TABLE or DROP INDEX.
    Statement,
    /// A query body other than SELECT, VALUES or a parenthesized query.
    QueryBody,
    /// A select item other than an unaliased expression or `*`.
    SelectItem,
    /// `SELECT DISTINCT ON (...)`.
    DistinctOn,
    /// A column reference with other than two parts, such as `a.b.c`.
    CompoundIdentifier,
    /// More than one comma-separated table in FROM.
    CommaSeparatedFrom,
    /// A FROM item that is not a plain table, such as a subquery.
    FromTarget,
    /// A join other than INNER or LEFT.
    JoinType,
    /// A join constraint other than `ON <expr>`, such as USING.
    JoinConstraint,
    /// More than one join in FROM.
    MultiWayJoin,
    /// GROUP BY with ROLLUP, CUBE or another modifier.
    GroupByModifiers,
    /// `GROUP BY ALL`.
    GroupByAll,
    /// `ORDER BY ALL`.
    OrderByAll,
    /// `ORDER BY ... INTERPOLATE`.
    OrderByInterpolate,
    /// `ORDER BY ... WITH FILL`.
    OrderByWithFill,
    /// A LIMIT that is not a numeric literal.
    LimitExpression,
    /// LIMIT with OFFSET or BY, or an OFFSET alone.
    LimitClause,
    /// `DISTINCT` inside an aggregate call.
    AggregateDistinct,
    /// An aggregate call with ORDER BY, WITHIN GROUP or another clause.
    AggregateClauses,
    /// A named argument to an aggregate.
    AggregateNamedArgument,
    /// A qualified wildcard such as `t.*` passed to an aggregate.
    AggregateQualifiedWildcard,
    /// An explicit window frame (`ROWS`, `RANGE` or `GROUPS`). The window executor computes
    /// only the default frame: the whole partition without ORDER BY, and the rows up to the
    /// current row's last peer with it.
    WindowFrame,
    /// A named window, `OVER w` or a `WINDOW` clause.
    NamedWindow,
    /// FILTER, IGNORE NULLS or WITHIN GROUP on a window function call.
    WindowFunctionClauses,
    /// A window function call in a SELECT with GROUP BY or aggregate functions.
    WindowInAggregatingSelect,
    /// A `QUALIFY` filter on window function results.
    Qualify,
    /// A `HAVING` filter on groups.
    Having,
    /// `SELECT TOP n`.
    Top,
    /// `SELECT ... INTO <table>`.
    SelectInto,
    /// `PREWHERE`.
    Prewhere,
    /// `START WITH ... CONNECT BY` hierarchical queries.
    ConnectBy,
    /// `CLUSTER BY`.
    ClusterBy,
    /// `DISTRIBUTE BY`.
    DistributeBy,
    /// `SORT BY`.
    SortBy,
    /// `LATERAL VIEW`.
    LateralView,
    /// `* EXCLUDE (...)`.
    WildcardExclude,
    /// `* EXCEPT (...)`.
    WildcardExcept,
    /// `* REPLACE (...)`.
    WildcardReplace,
    /// `* RENAME (...)`.
    WildcardRename,
    /// `* ILIKE '<pattern>'`.
    WildcardIlike,
    /// `* AS <alias>`.
    WildcardAlias,
    /// A `WITH` clause of common table expressions.
    With,
    /// `FETCH FIRST n ROWS`.
    Fetch,
    /// `FOR UPDATE` or `FOR SHARE` row locking.
    LockingClause,
    /// `FOR XML` or `FOR JSON` result formatting.
    ForClause,
    /// `SETTINGS key = value` on a query.
    QuerySettings,
    /// `FORMAT <name>` on a query.
    FormatClause,
    /// A pipe operator (`|>`) after a query.
    PipeOperator,
    /// ORDER BY on a query whose body is not a SELECT, such as VALUES, a parenthesized
    /// query or an INSERT source.
    OrderByWithoutSelect,
    /// LIMIT on a query whose body is not a SELECT, such as VALUES, a parenthesized query or
    /// an INSERT source.
    LimitWithoutSelect,
    /// An INSERT target that is not a table name, such as a table function.
    InsertTarget,
    /// An INSERT with no source rows.
    InsertWithoutSource,
    /// An INSERT whose source is not a VALUES list.
    InsertSource,
    /// A DELETE from other than one table.
    DeleteTableCount,
    /// A DELETE target that is not a plain table.
    DeleteTarget,
    /// An UPDATE target that is not a plain table.
    UpdateTarget,
    /// An UPDATE assignment to a tuple of columns.
    TupleAssignment,
    /// A CREATE INDEX with no index name.
    UnnamedIndex,
    /// An index key that is not a plain column name.
    IndexKeyExpression,
    /// `DROP ... CASCADE`.
    DropCascade,
    /// `DROP ... RESTRICT`.
    DropRestrict,
    /// `DROP ... PURGE`.
    DropPurge,
    /// A DROP with the MySQL keyword for session-scoped tables.
    DropTemporary,
    /// A DROP that names more than one object.
    DropManyObjects,
    /// `DROP TABLE ... ON <table>`.
    DropTableOn,
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Unsupported::Statement => "unsupported statement",
            Unsupported::QueryBody => "unsupported query body",
            Unsupported::SelectItem => "unsupported select item",
            Unsupported::DistinctOn => "SELECT DISTINCT ON (...) is not supported",
            Unsupported::CompoundIdentifier => "only `table.column` references are supported",
            Unsupported::CommaSeparatedFrom => "comma-separated FROM tables not supported",
            Unsupported::FromTarget => "FROM target must be a plain table",
            Unsupported::JoinType => "unsupported join type",
            Unsupported::JoinConstraint => "only ON <expr> join constraints supported",
            Unsupported::MultiWayJoin => "only 2-way joins supported",
            Unsupported::GroupByModifiers => {
                "GROUP BY modifiers (ROLLUP/CUBE/...) are not supported"
            }
            Unsupported::GroupByAll => "GROUP BY ALL is not supported",
            Unsupported::OrderByAll => "ORDER BY ALL not supported",
            Unsupported::OrderByInterpolate => "ORDER BY ... INTERPOLATE is not supported",
            Unsupported::OrderByWithFill => "ORDER BY ... WITH FILL is not supported",
            Unsupported::LimitExpression => "LIMIT must be a numeric literal",
            Unsupported::LimitClause => "unsupported LIMIT clause",
            Unsupported::AggregateDistinct => {
                "DISTINCT inside aggregate functions is not supported"
            }
            Unsupported::AggregateClauses => {
                "aggregate function clauses (ORDER BY / WITHIN GROUP / ...) are not supported"
            }
            Unsupported::AggregateNamedArgument => {
                "named arguments to aggregate functions are not supported"
            }
            Unsupported::AggregateQualifiedWildcard => {
                "qualified wildcards in aggregates are not supported"
            }
            Unsupported::WindowFrame => {
                "window frame clauses (ROWS / RANGE / GROUPS) are not supported"
            }
            Unsupported::NamedWindow => {
                "named windows are not supported; write the window inline in OVER (...)"
            }
            Unsupported::WindowFunctionClauses => {
                "FILTER, IGNORE NULLS and WITHIN GROUP are not supported on window functions"
            }
            Unsupported::WindowInAggregatingSelect => {
                "window functions in a SELECT with GROUP BY or aggregate functions are not supported"
            }
            Unsupported::Qualify => "QUALIFY is not supported",
            Unsupported::Having => "HAVING is not supported",
            Unsupported::Top => "SELECT TOP is not supported; use LIMIT",
            Unsupported::SelectInto => "SELECT ... INTO is not supported",
            Unsupported::Prewhere => "PREWHERE is not supported; use WHERE",
            Unsupported::ConnectBy => "CONNECT BY is not supported",
            Unsupported::ClusterBy => "CLUSTER BY is not supported",
            Unsupported::DistributeBy => "DISTRIBUTE BY is not supported",
            Unsupported::SortBy => "SORT BY is not supported; use ORDER BY",
            Unsupported::LateralView => "LATERAL VIEW is not supported",
            Unsupported::WildcardExclude => "SELECT * EXCLUDE is not supported",
            Unsupported::WildcardExcept => "SELECT * EXCEPT is not supported",
            Unsupported::WildcardReplace => "SELECT * REPLACE is not supported",
            Unsupported::WildcardRename => "SELECT * RENAME is not supported",
            Unsupported::WildcardIlike => "SELECT * ILIKE is not supported",
            Unsupported::WildcardAlias => "SELECT * AS <alias> is not supported",
            Unsupported::With => "WITH (common table expressions) is not supported",
            Unsupported::Fetch => "FETCH is not supported; use LIMIT",
            Unsupported::LockingClause => "FOR UPDATE / FOR SHARE is not supported",
            Unsupported::ForClause => "FOR XML / FOR JSON is not supported",
            Unsupported::QuerySettings => "SETTINGS is not supported",
            Unsupported::FormatClause => "FORMAT is not supported",
            Unsupported::PipeOperator => "pipe operators (|>) are not supported",
            Unsupported::OrderByWithoutSelect => "ORDER BY is only supported on a SELECT",
            Unsupported::LimitWithoutSelect => "LIMIT is only supported on a SELECT",
            Unsupported::InsertTarget => "unsupported INSERT target",
            Unsupported::InsertWithoutSource => "INSERT without source",
            Unsupported::InsertSource => "INSERT requires VALUES",
            Unsupported::DeleteTableCount => "DELETE expects one table",
            Unsupported::DeleteTarget => "DELETE target must be a table",
            Unsupported::UpdateTarget => "UPDATE target must be a table",
            Unsupported::TupleAssignment => "tuple UPDATE assignment not supported",
            Unsupported::UnnamedIndex => "CREATE INDEX without a name",
            Unsupported::IndexKeyExpression => "index columns must be plain identifiers",
            Unsupported::DropCascade => "DROP ... CASCADE is not supported",
            Unsupported::DropRestrict => "DROP ... RESTRICT is not supported",
            Unsupported::DropPurge => "DROP ... PURGE is not supported",
            Unsupported::DropTemporary => "DROP TEMPORARY is not supported",
            Unsupported::DropManyObjects => "DROP of more than one object is not supported",
            Unsupported::DropTableOn => "DROP TABLE ... ON is not supported",
        })
    }
}

impl<'a> Binder<'a> {
    pub fn new(catalog: &'a Catalog) -> Self {
        Self {
            catalog,
            agg_ctx: RefCell::new(None),
            inside_agg_arg: Cell::new(false),
            window_ctx: RefCell::new(None),
        }
    }

    fn resolve_expr(&self, expr: &Expr, scope: &Scope) -> Result<Expression, BinderError> {
        if let Expr::Function(f) = expr
            && let Some(over) = &f.over
        {
            return self.register_window_call(f, over, scope);
        }
        if self.inside_agg_arg.get() {
            reject_nested_aggregate(expr)?;
        } else if let Some(rewritten) = self.rewrite_for_aggregation(expr, scope)? {
            return Ok(rewritten);
        }

        match expr {
            Expr::Nested(inner) => self.resolve_expr(inner, scope),

            Expr::Identifier(id) => {
                let name = &id.value;
                let mut found: Option<Expression> = None;
                for ts in scope.values() {
                    if let Some(idx) = ts.schema.col_idx(name) {
                        if found.is_some() {
                            return Err(BinderError::AmbiguousColumn {
                                column: name.clone(),
                            });
                        }
                        found = Some(ts.column(idx));
                    }
                }
                found.ok_or_else(|| BinderError::ColumnNotFound {
                    column: name.clone(),
                })
            }

            Expr::CompoundIdentifier(parts) => {
                let [t, c] = parts.as_slice() else {
                    return Err(Unsupported::CompoundIdentifier.into());
                };
                let ts = scope
                    .get(&t.value)
                    .ok_or_else(|| BinderError::TableNotInScope {
                        table: t.value.clone(),
                    })?;
                let idx =
                    ts.schema
                        .col_idx(&c.value)
                        .ok_or_else(|| BinderError::ColumnNotFound {
                            column: c.value.clone(),
                        })?;
                Ok(ts.column(idx))
            }

            Expr::BinaryOp {
                left,
                op: sql_op,
                right,
            } => {
                let op = match sql_op {
                    BinaryOperator::Plus => Op::Add,
                    BinaryOperator::Minus => Op::Sub,
                    BinaryOperator::Multiply => Op::Mul,
                    BinaryOperator::Divide => Op::Div,
                    BinaryOperator::Eq => Op::Eq,
                    BinaryOperator::NotEq => Op::NEq,
                    BinaryOperator::Lt => Op::Lt,
                    BinaryOperator::Gt => Op::Gt,
                    BinaryOperator::LtEq => Op::Lte,
                    BinaryOperator::GtEq => Op::Gte,
                    BinaryOperator::And => Op::And,
                    BinaryOperator::Or => Op::Or,
                    other => {
                        return Err(BinderError::UnsupportedOperator {
                            operator: other.clone(),
                        });
                    }
                };
                let (l, r) = if string_literal(left).is_some() && string_literal(right).is_none() {
                    let r = self.resolve_expr(right, scope)?;
                    (self.resolve_expr_expecting(left, scope, r.dtype())?, r)
                } else {
                    let l = self.resolve_expr(left, scope)?;
                    let r = self.resolve_expr_expecting(right, scope, l.dtype())?;
                    (l, r)
                };
                check_operand_types(sql_op, &op, l.dtype(), r.dtype())?;
                let (l, r) = if is_comparison(&op) {
                    match_constant_to_other_side(l, r)
                } else {
                    (l, r)
                };
                Ok(Expression::Binary {
                    left: Box::new(l),
                    right: Box::new(r),
                    op,
                })
            }

            Expr::UnaryOp {
                op: op @ (UnaryOperator::Minus | UnaryOperator::Plus),
                expr: inner,
            } => {
                if let (UnaryOperator::Minus, Expr::Value(v)) = (op, inner.as_ref())
                    && let SqlValue::Number(s, _) = &v.value
                {
                    return Ok(Expression::Constant(Some(number_literal(&format!(
                        "-{s}"
                    ))?)));
                }
                let inner = self.resolve_expr(inner, scope)?;
                if let Some(t) = inner.dtype()
                    && !t.is_numeric()
                {
                    return Err(BinderError::NegateNonNumber { operand: t });
                }
                match op {
                    UnaryOperator::Minus => negate(inner),
                    _ => Ok(inner),
                }
            }

            Expr::TypedString(ts) => match (&ts.data_type, &ts.value.value) {
                (
                    SqlDataType::Timestamp(_, TimezoneInfo::None | TimezoneInfo::WithoutTimeZone),
                    SqlValue::SingleQuotedString(text),
                ) => timestamp_constant(text),
                _ => Err(BinderError::UnsupportedExpression {
                    expr: expr.to_string(),
                }),
            },

            Expr::Value(v) => {
                let val = match &v.value {
                    SqlValue::Null => None,
                    SqlValue::Boolean(b) => Some(Value::BOOLEAN(*b)),
                    SqlValue::Number(s, _) => Some(number_literal(s)?),
                    SqlValue::SingleQuotedString(s) => Some(Value::text(s)),
                    other => {
                        return Err(BinderError::UnsupportedLiteral {
                            literal: other.to_string(),
                        });
                    }
                };
                Ok(Expression::Constant(val))
            }

            Expr::Function(f) if is_window_only_function(&f.name) => {
                Err(BinderError::WindowFunctionWithoutOver {
                    function: f.name.to_string(),
                })
            }

            other => Err(BinderError::UnsupportedExpression {
                expr: other.to_string(),
            }),
        }
    }

    /// Runs `f` as the binder runs an aggregate's argument, where aggregate and window calls are
    /// rejected as nested.
    fn resolve_nested<R>(&self, f: impl FnOnce() -> R) -> R {
        let was_inside = self.inside_agg_arg.replace(true);
        let result = f();
        self.inside_agg_arg.set(was_inside);
        result
    }

    /// Resolves `expr` where a value of type `expected` belongs, such as an INSERT value or the
    /// other side of a comparison. A string literal there is read as that type; elsewhere it is
    /// resolved as [`Binder::resolve_expr`] resolves it.
    fn resolve_expr_expecting(
        &self,
        expr: &Expr,
        scope: &Scope,
        expected: Option<DataType>,
    ) -> Result<Expression, BinderError> {
        match (expected, string_literal(expr)) {
            (Some(DataType::TIMESTAMP), Some(text)) => timestamp_constant(text),
            _ => self.resolve_expr(expr, scope),
        }
    }

    /// Resolves a value an INSERT or UPDATE stores in `col`. A constant is converted to the
    /// column's type here; any other expression has to have a type that converts to it.
    fn bind_column_value(
        &self,
        expr: &Expr,
        scope: &Scope,
        col: &Column,
    ) -> Result<Expression, BinderError> {
        let resolved = self.resolve_expr_expecting(expr, scope, Some(col.dtype))?;
        match resolved {
            Expression::Constant(Some(v)) => v
                .clone()
                .coerce_to(col.dtype)
                .map(|v| Expression::Constant(Some(v)))
                .map_err(|e| match e {
                    CoercionError::WrongType => BinderError::ColumnType {
                        column: col.name.clone(),
                        expected: col.dtype,
                        found: v.data_type(),
                    },
                    CoercionError::OutOfRange => BinderError::ColumnOutOfRange {
                        column: col.name.clone(),
                        expected: col.dtype,
                        value: v,
                    },
                    CoercionError::TooLong { len } => BinderError::ColumnTooLong {
                        column: col.name.clone(),
                        expected: col.dtype,
                        len,
                    },
                }),
            other => match other.dtype() {
                Some(found) if !found.coerces_to(col.dtype) => Err(BinderError::ColumnType {
                    column: col.name.clone(),
                    expected: col.dtype,
                    found,
                }),
                _ => Ok(other),
            },
        }
    }

    /// Inside an aggregating SELECT, maps an expression equal to a GROUP BY expression, or an
    /// aggregate call, to a column of the aggregation output (`[group_bys..., aggregates...]`), and
    /// rejects a column reference that is neither. Returns `None` when `expr` resolves normally.
    fn rewrite_for_aggregation(
        &self,
        expr: &Expr,
        scope: &Scope,
    ) -> Result<Option<Expression>, BinderError> {
        if let Expr::Function(f) = expr
            && let Some(agg_type) = parse_agg_type(&f.name)
        {
            return self.register_aggregate(f, scope, agg_type).map(Some);
        }
        let ctx = self.agg_ctx.borrow();
        let Some(ctx) = ctx.as_ref() else {
            return Ok(None);
        };
        if let Some(idx) = ctx.group_by_exprs.iter().position(|gb| gb == expr) {
            let dtype = ctx.group_bys[idx]
                .dtype()
                .ok_or(BinderError::UntypedGroupBy)?;
            return Ok(Some(Expression::Column {
                col: ColIdx(idx as u32),
                dtype,
            }));
        }
        if let Some(path) = column_path(expr) {
            return Err(BinderError::UngroupedColumn { column: path });
        }
        Ok(None)
    }

    /// Records an aggregate call of the current aggregating SELECT and returns the column of the
    /// aggregation output that will hold its result.
    fn register_aggregate(
        &self,
        f: &Function,
        scope: &Scope,
        agg_type: AggType,
    ) -> Result<Expression, BinderError> {
        if self.agg_ctx.borrow().is_none() {
            return Err(BinderError::AggregateOutsideProjection);
        }
        let arg_expr = self.bind_aggregate_argument(f, scope, &agg_type)?;
        let result_dtype = aggregate_result_type(&agg_type, &arg_expr, &f.name)?;

        let mut ctx_mut = self.agg_ctx.borrow_mut();
        let ctx = ctx_mut
            .as_mut()
            .ok_or(BinderError::AggregateOutsideProjection)?;
        let aggregate = (arg_expr, agg_type);
        let agg_idx = match ctx.aggregates.iter().position(|a| *a == aggregate) {
            Some(existing) => existing,
            None => {
                ctx.aggregates.push(aggregate);
                ctx.aggregates.len() - 1
            }
        };
        let col_idx = ctx.group_bys.len() + agg_idx;
        Ok(Expression::Column {
            col: ColIdx(col_idx as u32),
            dtype: result_dtype,
        })
    }

    /// Binds the single argument of an aggregate call `f`, with `*` standing for every row of a
    /// COUNT.
    fn bind_aggregate_argument(
        &self,
        f: &Function,
        scope: &Scope,
        agg_type: &AggType,
    ) -> Result<Expression, BinderError> {
        let FunctionArguments::List(list) = &f.args else {
            return Err(BinderError::AggregateWithoutArguments {
                function: f.name.to_string(),
            });
        };
        if list.duplicate_treatment.is_some() {
            return Err(Unsupported::AggregateDistinct.into());
        }
        if !list.clauses.is_empty() {
            return Err(Unsupported::AggregateClauses.into());
        }
        if list.args.len() != 1 {
            return Err(BinderError::AggregateArgumentCount {
                function: f.name.to_string(),
                args: list.args.len(),
            });
        }
        let FunctionArg::Unnamed(arg) = &list.args[0] else {
            return Err(Unsupported::AggregateNamedArgument.into());
        };

        match arg {
            FunctionArgExpr::Wildcard => {
                if !matches!(agg_type, AggType::COUNT) {
                    return Err(BinderError::StarArgumentNotCount {
                        function: f.name.to_string(),
                    });
                }
                Ok(COUNT_STAR_ARGUMENT)
            }
            FunctionArgExpr::Expr(e) => self.resolve_nested(|| self.resolve_expr(e, scope)),
            FunctionArgExpr::QualifiedWildcard(_) | FunctionArgExpr::WildcardWithOptions(_) => {
                Err(Unsupported::AggregateQualifiedWildcard.into())
            }
        }
    }

    /// Records a window function call `f OVER over` of the current SELECT and returns the
    /// placeholder column that [`WindowCtx::place_calls`] later moves to the call's result.
    fn register_window_call(
        &self,
        f: &Function,
        over: &WindowType,
        scope: &Scope,
    ) -> Result<Expression, BinderError> {
        if self.inside_agg_arg.get() {
            return Err(BinderError::NestedWindowFunction);
        }
        if self.agg_ctx.borrow().is_some() {
            return Err(Unsupported::WindowInAggregatingSelect.into());
        }
        if self.window_ctx.borrow().is_none() {
            return Err(BinderError::WindowOutsideProjection);
        }
        let agg_type = parse_window_function_type(&f.name).ok_or_else(|| {
            BinderError::UnknownWindowFunction {
                function: f.name.to_string(),
            }
        })?;
        if f.filter.is_some() || f.null_treatment.is_some() || !f.within_group.is_empty() {
            return Err(Unsupported::WindowFunctionClauses.into());
        }
        let arg = match agg_type {
            AggType::RANK => {
                reject_arguments(f)?;
                RANK_ARGUMENT
            }
            _ => self.bind_aggregate_argument(f, scope, &agg_type)?,
        };
        let dtype = aggregate_result_type(&agg_type, &arg, &f.name)?;
        let window = self.window_for(inline_window_spec(over)?, scope)?;

        let mut ctx_mut = self.window_ctx.borrow_mut();
        let ctx = ctx_mut
            .as_mut()
            .ok_or(BinderError::WindowOutsideProjection)?;
        let existing = ctx.calls.iter().position(|&(w, pos)| {
            let call = &ctx.windows[w].calls[pos];
            w == window && call.arg == arg && call.agg_type == agg_type
        });
        let call = match existing {
            Some(call) => call,
            None => {
                ctx.calls.push((window, ctx.windows[window].calls.len()));
                ctx.windows[window].calls.push(BoundWindowCall {
                    arg,
                    agg_type,
                    dtype,
                });
                ctx.calls.len() - 1
            }
        };
        let placeholder = ctx.input_width + call;
        Ok(Expression::Column {
            col: ColIdx(placeholder as u32),
            dtype,
        })
    }

    /// The index of the current SELECT's window over `spec`, binding it on its first use.
    fn window_for(&self, spec: &WindowSpec, scope: &Scope) -> Result<usize, BinderError> {
        let existing = self
            .window_ctx
            .borrow()
            .as_ref()
            .and_then(|ctx| ctx.specs.iter().position(|s| s == spec));
        if let Some(window) = existing {
            return Ok(window);
        }
        let window = self.resolve_nested(|| {
            Ok::<_, BinderError>(BoundWindow {
                partition_by: spec
                    .partition_by
                    .iter()
                    .map(|e| self.resolve_expr(e, scope))
                    .collect::<Result<_, _>>()?,
                order_by: spec
                    .order_by
                    .iter()
                    .map(|oe| self.resolve_sort_expr(oe, scope))
                    .collect::<Result<_, _>>()?,
                calls: vec![],
            })
        })?;
        let mut ctx_mut = self.window_ctx.borrow_mut();
        let ctx = ctx_mut
            .as_mut()
            .ok_or(BinderError::WindowOutsideProjection)?;
        ctx.specs.push(spec.clone());
        ctx.windows.push(window);
        Ok(ctx.windows.len() - 1)
    }

    /// Runs `f` with `ctx` as the active aggregation context, then restores the previous context.
    /// Returns `f`'s result together with the context as `f` left it.
    fn with_agg_ctx<R>(&self, ctx: Option<AggCtx>, f: impl FnOnce() -> R) -> (R, Option<AggCtx>) {
        let prev = self.agg_ctx.replace(ctx);
        let result = f();
        (result, self.agg_ctx.replace(prev))
    }

    fn resolve_select(&self, q: &Query) -> Result<BoundStatement, BinderError> {
        reject_unsupported_query_clauses(q)?;
        match q.body.as_ref() {
            SetExpr::Query(inner) => {
                reject_order_by_and_limit(q)?;
                self.resolve_select(inner)
            }

            SetExpr::Values(values) => {
                reject_order_by_and_limit(q)?;
                let empty = HashMap::new();
                let rows = values
                    .rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|e| self.resolve_expr(e, &empty))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(BoundStatement::Values(BoundValues { rows }))
            }

            SetExpr::Select(s) => {
                reject_unsupported_select_clauses(s)?;
                let distinct = is_distinct(s.distinct.as_ref())?;
                let (scope, join) = self.build_select_scope(&s.from)?;
                let filters = s
                    .selection
                    .as_ref()
                    .map(|e| self.resolve_expr(e, &scope).map(|e| vec![e]))
                    .transpose()?;

                let group_by_exprs_raw = match &s.group_by {
                    GroupByExpr::Expressions(exprs, modifiers) => {
                        if !modifiers.is_empty() {
                            return Err(Unsupported::GroupByModifiers.into());
                        }
                        exprs.clone()
                    }
                    GroupByExpr::All(_) => {
                        return Err(Unsupported::GroupByAll.into());
                    }
                };
                let group_bys: Vec<Expression> = group_by_exprs_raw
                    .iter()
                    .map(|e| self.resolve_expr(e, &scope))
                    .collect::<Result<_, _>>()?;

                let aggregating =
                    !group_by_exprs_raw.is_empty() || projection_has_aggregate(&s.projection);
                let agg_ctx = aggregating.then(|| AggCtx {
                    group_by_exprs: group_by_exprs_raw,
                    group_bys,
                    aggregates: vec![],
                });
                let from_width = scope.values().map(|ts| ts.schema.cols.len()).sum();
                let window_ctx = (!aggregating).then(|| WindowCtx::new(from_width));
                let prev_window_ctx = self.window_ctx.replace(window_ctx);
                let (resolved, agg_ctx) = self.with_agg_ctx(agg_ctx, || {
                    let projections = self.resolve_projections(&s.projection, &scope)?;
                    let sort = self.resolve_order_by(q, &scope)?;
                    Ok::<_, BinderError>((projections, sort))
                });
                let window_ctx = self.window_ctx.replace(prev_window_ctx);
                let (mut projections, mut sort) = resolved?;
                let windows = match window_ctx {
                    Some(ctx) if !ctx.calls.is_empty() => {
                        let output = ctx.output_columns();
                        projections = projections
                            .into_iter()
                            .map(|e| ctx.place_calls(e, &output))
                            .collect();
                        sort = sort.map(|keys| {
                            keys.into_iter()
                                .map(|key| SortExpr {
                                    expr: ctx.place_calls(key.expr, &output),
                                    ..key
                                })
                                .collect()
                        });
                        ctx.windows
                    }
                    _ => vec![],
                };
                let sort = match sort {
                    Some(keys) if distinct => Some(sort_by_select_list(keys, &projections)?),
                    other => other,
                };
                let limit = resolve_limit(q.limit_clause.as_ref())?;
                let (group_bys, aggregates) = agg_ctx
                    .map(|ctx| (ctx.group_bys, ctx.aggregates))
                    .unwrap_or_default();

                Ok(BoundStatement::Select(BoundSelect {
                    scope,
                    join,
                    projections,
                    distinct,
                    filters,
                    group_bys,
                    aggregates,
                    windows,
                    sort,
                    limit,
                }))
            }

            _ => Err(Unsupported::QueryBody.into()),
        }
    }

    fn resolve_order_by(
        &self,
        q: &Query,
        scope: &Scope,
    ) -> Result<Option<Vec<SortExpr>>, BinderError> {
        let Some(ob) = q.order_by.as_ref() else {
            return Ok(None);
        };
        if ob.interpolate.is_some() {
            return Err(Unsupported::OrderByInterpolate.into());
        }
        match &ob.kind {
            OrderByKind::Expressions(exprs) => exprs
                .iter()
                .map(|oe| self.resolve_sort_expr(oe, scope))
                .collect::<Result<_, _>>()
                .map(Some),
            OrderByKind::All(_) => Err(Unsupported::OrderByAll.into()),
        }
    }

    fn resolve_sort_expr(&self, oe: &OrderByExpr, scope: &Scope) -> Result<SortExpr, BinderError> {
        if oe.with_fill.is_some() {
            return Err(Unsupported::OrderByWithFill.into());
        }
        let direction = match oe.options.asc {
            Some(false) => SortDirection::Descending,
            Some(true) | None => SortDirection::Ascending,
        };
        let sort = SortExpr::new(self.resolve_expr(&oe.expr, scope)?, direction);
        Ok(match oe.options.nulls_first {
            Some(true) => sort.with_nulls(NullOrdering::First),
            Some(false) => sort.with_nulls(NullOrdering::Last),
            None => sort,
        })
    }

    /// Builds a scope from the FROM clause: at most two tables, joined by a single
    /// INNER/LEFT JOIN ... ON. Columns resolve to positions in the FROM output, which lists the
    /// left table's columns before the right table's; the ON predicate reads each side's row.
    fn build_select_scope(
        &self,
        from: &[TableWithJoins],
    ) -> Result<(Scope, Option<BoundJoin>), BinderError> {
        if from.is_empty() {
            return Ok((HashMap::new(), None));
        }
        let [twj] = from else {
            return Err(Unsupported::CommaSeparatedFrom.into());
        };

        let mut scope = HashMap::new();
        let (left_alias, left) = self.bind_table_factor(&twj.relation, Side::Left, 0)?;
        let left_cols = left.schema.cols.len();
        scope.insert(left_alias, left);

        let join = match twj.joins.as_slice() {
            [] => None,
            [j] => {
                let (right_alias, right) =
                    self.bind_table_factor(&j.relation, Side::Right, left_cols)?;
                scope.insert(right_alias, right);

                let (join_type, predicate_expr) = match &j.join_operator {
                    JoinOperator::Inner(c) | JoinOperator::Join(c) => (JoinType::Inner, c),
                    JoinOperator::Left(c) | JoinOperator::LeftOuter(c) => (JoinType::Left, c),
                    _ => return Err(Unsupported::JoinType.into()),
                };
                let JoinConstraint::On(expr) = predicate_expr else {
                    return Err(Unsupported::JoinConstraint.into());
                };
                let predicate = join_predicate(self.resolve_expr(expr, &scope)?, left_cols);
                Some(BoundJoin {
                    join_type,
                    predicate,
                })
            }
            _ => return Err(Unsupported::MultiWayJoin.into()),
        };

        Ok((scope, join))
    }

    fn bind_table_factor(
        &self,
        tf: &TableFactor,
        side: Side,
        first_col: usize,
    ) -> Result<(TableAlias, TableScope), BinderError> {
        let TableFactor::Table { name, alias, .. } = tf else {
            return Err(Unsupported::FromTarget.into());
        };
        let table_name = ident_name(name)?;
        let oid = self
            .catalog
            .table_oid(&table_name)
            .map_err(|_| BinderError::TableNotFound {
                table: table_name.to_string(),
            })?;
        let tinfo = self
            .catalog
            .get_table(oid)
            .map_err(|_| BinderError::TableNotFound {
                table: table_name.to_string(),
            })?;
        let key = alias
            .as_ref()
            .map(|a| a.name.value.clone())
            .unwrap_or(table_name);
        Ok((
            key,
            TableScope {
                oid,
                side,
                first_col,
                schema: tinfo.schema.clone(),
            },
        ))
    }

    fn resolve_projections(
        &self,
        items: &[SelectItem],
        scope: &Scope,
    ) -> Result<Vec<Expression>, BinderError> {
        let mut out = Vec::new();
        for item in items {
            match item {
                SelectItem::UnnamedExpr(e) => {
                    let resolved = self.resolve_expr(e, scope)?;
                    if resolved.dtype().is_none() {
                        return Err(BinderError::BareNullProjection);
                    }
                    out.push(resolved);
                }
                SelectItem::Wildcard(options) => {
                    reject_wildcard_options(options)?;
                    let mut tables: Vec<&TableScope> = scope.values().collect();
                    tables.sort_by_key(|ts| ts.first_col);
                    for ts in tables {
                        out.extend((0..ts.schema.cols.len()).map(|i| ts.column(i)));
                    }
                }
                _ => return Err(Unsupported::SelectItem.into()),
            }
        }
        Ok(out)
    }

    fn resolve_insert(&self, stmt: &Insert) -> Result<BoundStatement, BinderError> {
        let TableObject::TableName(name) = &stmt.table else {
            return Err(Unsupported::InsertTarget.into());
        };
        let table_name = ident_name(name)?;
        let scope = self.build_scope(&table_name)?;
        let schema = &scope[&table_name].schema;
        let targets = insert_targets(schema, &stmt.columns)?;

        let source = stmt
            .source
            .as_ref()
            .ok_or(Unsupported::InsertWithoutSource)?;
        reject_unsupported_query_clauses(source)?;
        let SetExpr::Values(values) = source.body.as_ref() else {
            return Err(Unsupported::InsertSource.into());
        };
        reject_order_by_and_limit(source)?;

        let no_input_columns = HashMap::new();
        let rows = values
            .rows
            .iter()
            .map(|row| {
                if row.content.len() != targets.len() {
                    return Err(BinderError::InsertArity {
                        table: table_name.clone(),
                        columns: targets.len(),
                        values: row.content.len(),
                    });
                }
                let mut full_row = vec![Expression::Constant(None); schema.cols.len()];
                for (e, &col) in row.content.iter().zip(&targets) {
                    full_row[col] =
                        self.bind_column_value(e, &no_input_columns, &schema.cols[col])?;
                }
                Ok(full_row)
            })
            .collect::<Result<_, _>>()?;

        Ok(BoundStatement::Insert(BoundInsert { scope, rows }))
    }

    fn resolve_delete(&self, stmt: &Delete) -> Result<BoundStatement, BinderError> {
        let (FromTable::WithFromKeyword(tables) | FromTable::WithoutKeyword(tables)) = &stmt.from;
        let [t] = tables.as_slice() else {
            return Err(Unsupported::DeleteTableCount.into());
        };
        let TableFactor::Table { name, .. } = &t.relation else {
            return Err(Unsupported::DeleteTarget.into());
        };
        let scope = self.build_scope(&ident_name(name)?)?;

        let filters = stmt
            .selection
            .as_ref()
            .map(|e| self.resolve_expr(e, &scope).map(|e| vec![e]))
            .transpose()?;

        Ok(BoundStatement::Delete(BoundDelete { scope, filters }))
    }

    fn resolve_update(&self, stmt: &Update) -> Result<BoundStatement, BinderError> {
        let TableFactor::Table { name, .. } = &stmt.table.relation else {
            return Err(Unsupported::UpdateTarget.into());
        };
        let table_name = ident_name(name)?;
        let scope = self.build_scope(&table_name)?;
        let schema = &scope[&table_name].schema;

        let mut cols = Vec::with_capacity(stmt.assignments.len());
        let mut new_cols = Vec::with_capacity(stmt.assignments.len());
        for Assignment { target, value } in &stmt.assignments {
            let AssignmentTarget::ColumnName(col_name) = target else {
                return Err(Unsupported::TupleAssignment.into());
            };
            let col = ident_name(col_name)?;
            let idx = schema
                .col_idx(&col)
                .ok_or_else(|| BinderError::ColumnNotFound {
                    column: col.clone(),
                })?;
            cols.push(ColIdx(idx as u32));
            new_cols.push(self.bind_column_value(value, &scope, &schema.cols[idx])?);
        }

        let filters = stmt
            .selection
            .as_ref()
            .map(|e| self.resolve_expr(e, &scope).map(|e| vec![e]))
            .transpose()?;

        Ok(BoundStatement::Update(BoundUpdate {
            scope,
            cols,
            new_cols,
            filters,
        }))
    }

    fn resolve_create_table(&self, stmt: &SqlCreateTable) -> Result<BoundStatement, BinderError> {
        let table_name = ident_name(&stmt.name)?;
        let cols = stmt
            .columns
            .iter()
            .map(|c| Ok((sql_to_dtype(&c.data_type)?, c.name.value.clone())))
            .collect::<Result<Vec<_>, BinderError>>()?;
        Ok(BoundStatement::CreateTable(BoundCreateTable {
            table_name,
            schema: Schema::new(cols),
        }))
    }

    fn resolve_create_index(&self, stmt: &SqlCreateIndex) -> Result<BoundStatement, BinderError> {
        let name = stmt
            .name
            .as_ref()
            .ok_or(BinderError::Unsupported(Unsupported::UnnamedIndex))
            .and_then(ident_name)?;
        let table_name = ident_name(&stmt.table_name)?;
        let scope = self.build_scope(&table_name)?;
        let schema = &scope[&table_name].schema;

        let key_cols = stmt
            .columns
            .iter()
            .map(|c| {
                let Expr::Identifier(id) = &c.column.expr else {
                    return Err(Unsupported::IndexKeyExpression.into());
                };
                let col_name = id.value.clone();
                let idx = schema
                    .col_idx(&col_name)
                    .ok_or_else(|| BinderError::ColumnNotFound {
                        column: col_name.clone(),
                    })?;
                Ok((schema.cols[idx].dtype, col_name))
            })
            .collect::<Result<Vec<_>, BinderError>>()?;

        Ok(BoundStatement::CreateIndex(BoundCreateIndex {
            scope,
            name,
            index_schema: Schema::new(key_cols),
        }))
    }

    fn resolve_drop_table(
        &self,
        name: &ObjectName,
        if_exists: bool,
    ) -> Result<BoundStatement, BinderError> {
        let table_name = ident_name(name)?;
        let table_oid = match self.catalog.table_oid(&table_name).ok() {
            None if !if_exists => {
                return Err(BinderError::TableNotFound { table: table_name });
            }
            oid => oid,
        };
        Ok(BoundStatement::DropTable(BoundDropTable {
            table_oid,
            if_exists,
        }))
    }

    /// Binds `DROP INDEX name [ON table]`. Without `ON`, the index is looked up across every
    /// table, and a name that more than one table uses is an error.
    fn resolve_drop_index(
        &self,
        name: &ObjectName,
        table: Option<&ObjectName>,
        if_exists: bool,
    ) -> Result<BoundStatement, BinderError> {
        let index_name = ident_name(name)?;
        let candidates = match table {
            Some(table) => {
                let table_name = ident_name(table)?;
                if self.catalog.table_oid(&table_name).is_err() && !if_exists {
                    return Err(BinderError::TableNotFound { table: table_name });
                }
                self.catalog
                    .index_oid(&table_name, &index_name)
                    .ok()
                    .into_iter()
                    .collect()
            }
            None => self.catalog.index_oids_named(&index_name),
        };
        let index_oid = match candidates.as_slice() {
            [] if if_exists => None,
            [] => return Err(BinderError::IndexNotFound { index: index_name }),
            [oid] => Some(*oid),
            _ => return Err(BinderError::AmbiguousIndex { index: index_name }),
        };
        Ok(BoundStatement::DropIndex(BoundDropIndex {
            index_oid,
            if_exists,
        }))
    }

    /// Binds every statement against the catalog as it is now. A statement that depends on an
    /// earlier one in the same batch (such as an INSERT into a table created just before it) fails
    /// to bind; run statements one at a time with [`Binder::bind_statement`] instead.
    pub fn bind(&self, stmts: Vec<Statement>) -> Result<Vec<BoundStatement>, BinderError> {
        stmts.iter().map(|s| self.bind_statement(s)).collect()
    }

    /// Binds one statement against the catalog as it is now.
    pub fn bind_statement(&self, stmt: &Statement) -> Result<BoundStatement, BinderError> {
        match stmt {
            Statement::Query(q) => self.resolve_select(q),
            Statement::Insert(i) => self.resolve_insert(i),
            Statement::Delete(d) => self.resolve_delete(d),
            Statement::Update(u) => self.resolve_update(u),
            Statement::CreateTable(c) => self.resolve_create_table(c),
            Statement::CreateIndex(c) => self.resolve_create_index(c),
            Statement::Drop {
                object_type,
                if_exists,
                names,
                cascade,
                restrict,
                purge,
                temporary,
                table,
            } => {
                let modifiers = [
                    (Unsupported::DropCascade, *cascade),
                    (Unsupported::DropRestrict, *restrict),
                    (Unsupported::DropPurge, *purge),
                    (Unsupported::DropTemporary, *temporary),
                ];
                if let Some((modifier, _)) = modifiers.iter().find(|(_, set)| *set) {
                    return Err((*modifier).into());
                }
                let [name] = names.as_slice() else {
                    return Err(Unsupported::DropManyObjects.into());
                };
                match object_type {
                    ObjectType::Table if table.is_some() => Err(Unsupported::DropTableOn.into()),
                    ObjectType::Table => self.resolve_drop_table(name, *if_exists),
                    ObjectType::Index => self.resolve_drop_index(name, table.as_ref(), *if_exists),
                    other => Err(BinderError::UnsupportedDropObject {
                        object_type: other.to_string(),
                    }),
                }
            }
            _ => Err(Unsupported::Statement.into()),
        }
    }

    fn build_scope(&self, table_name: &str) -> Result<Scope, BinderError> {
        let oid = self
            .catalog
            .table_oid(table_name)
            .map_err(|_| BinderError::TableNotFound {
                table: table_name.to_string(),
            })?;
        let tinfo = self
            .catalog
            .get_table(oid)
            .map_err(|_| BinderError::TableNotFound {
                table: table_name.to_string(),
            })?;
        let mut scope = HashMap::new();
        scope.insert(
            table_name.to_string(),
            TableScope {
                oid,
                side: Side::Left,
                first_col: 0,
                schema: tinfo.schema.clone(),
            },
        );
        Ok(scope)
    }
}

fn ident_name(name: &ObjectName) -> Result<String, BinderError> {
    match name.0.as_slice() {
        [ObjectNamePart::Identifier(id)] => Ok(id.value.clone()),
        _ => Err(BinderError::QualifiedName {
            name: name.to_string(),
        }),
    }
}

/// Rejects the clauses of `s` the binder does not bind, so a query using one fails instead of
/// running without it.
///
/// Destructures every field of `Select`, so a clause added by a sqlparser upgrade fails to
/// compile here until it is bound or rejected.
fn reject_unsupported_select_clauses(s: &Select) -> Result<(), BinderError> {
    let Select {
        select_token: _,
        optimizer_hints: _,
        distinct: _,
        select_modifiers: _,
        top,
        top_before_distinct: _,
        projection: _,
        exclude: _,
        into,
        from: _,
        lateral_views,
        prewhere,
        selection: _,
        connect_by,
        group_by: _,
        cluster_by,
        distribute_by,
        sort_by,
        having,
        named_window,
        qualify,
        window_before_qualify: _,
        value_table_mode: _,
        flavor: _,
    } = s;
    let unsupported = [
        (having.is_some(), Unsupported::Having),
        (top.is_some(), Unsupported::Top),
        (into.is_some(), Unsupported::SelectInto),
        (prewhere.is_some(), Unsupported::Prewhere),
        (!connect_by.is_empty(), Unsupported::ConnectBy),
        (!cluster_by.is_empty(), Unsupported::ClusterBy),
        (!distribute_by.is_empty(), Unsupported::DistributeBy),
        (!sort_by.is_empty(), Unsupported::SortBy),
        (!lateral_views.is_empty(), Unsupported::LateralView),
        (!named_window.is_empty(), Unsupported::NamedWindow),
        (qualify.is_some(), Unsupported::Qualify),
    ];
    first_unsupported(unsupported)
}

/// Rejects the query-level clauses of `q` the binder does not bind, so a query using one fails
/// instead of running without it. ORDER BY and LIMIT are left to the caller.
///
/// Destructures every field of `Query`, so a clause added by a sqlparser upgrade fails to
/// compile here until it is bound or rejected.
fn reject_unsupported_query_clauses(q: &Query) -> Result<(), BinderError> {
    let Query {
        with,
        body: _,
        order_by: _,
        limit_clause: _,
        fetch,
        locks,
        for_clause,
        settings,
        format_clause,
        pipe_operators,
    } = q;
    let unsupported = [
        (with.is_some(), Unsupported::With),
        (fetch.is_some(), Unsupported::Fetch),
        (!locks.is_empty(), Unsupported::LockingClause),
        (for_clause.is_some(), Unsupported::ForClause),
        (settings.is_some(), Unsupported::QuerySettings),
        (format_clause.is_some(), Unsupported::FormatClause),
        (!pipe_operators.is_empty(), Unsupported::PipeOperator),
    ];
    first_unsupported(unsupported)
}

/// Rejects ORDER BY and LIMIT on `q`, for a query whose body the binder resolves without
/// sorting or limiting it.
fn reject_order_by_and_limit(q: &Query) -> Result<(), BinderError> {
    first_unsupported([
        (q.order_by.is_some(), Unsupported::OrderByWithoutSelect),
        (q.limit_clause.is_some(), Unsupported::LimitWithoutSelect),
    ])
}

/// Returns the first feature whose flag is set as an error, or `Ok` when none is.
fn first_unsupported<const N: usize>(checks: [(bool, Unsupported); N]) -> Result<(), BinderError> {
    match checks.into_iter().find(|&(present, _)| present) {
        Some((_, feature)) => Err(feature.into()),
        None => Ok(()),
    }
}

/// Whether a SELECT removes duplicate rows; `DISTINCT ON (...)` is rejected.
fn is_distinct(distinct: Option<&Distinct>) -> Result<bool, BinderError> {
    match distinct {
        None | Some(Distinct::All) => Ok(false),
        Some(Distinct::Distinct) => Ok(true),
        Some(Distinct::On(_)) => Err(Unsupported::DistinctOn.into()),
    }
}

/// Rewrites ORDER BY keys to read the projection output: each key becomes a reference to the
/// first select-list expression equal to it.
fn sort_by_select_list(
    keys: Vec<SortExpr>,
    projections: &[Expression],
) -> Result<Vec<SortExpr>, BinderError> {
    keys.into_iter()
        .map(|key| {
            let (col, projection) = projections
                .iter()
                .enumerate()
                .find(|(_, p)| **p == key.expr)
                .ok_or(BinderError::DistinctOrderByNotInSelectList)?;
            let expr = Expression::Column {
                col: ColIdx(col as u32),
                dtype: projection.dtype().unwrap_or(DataType::INT),
            };
            Ok(SortExpr { expr, ..key })
        })
        .collect()
}

/// Rejects the options of a `*` select item, so a query using one fails instead of returning
/// every column unchanged.
///
/// Destructures every field of `WildcardAdditionalOptions`, so an option added by a sqlparser
/// upgrade fails to compile here until it is bound or rejected.
fn reject_wildcard_options(options: &WildcardAdditionalOptions) -> Result<(), BinderError> {
    let WildcardAdditionalOptions {
        wildcard_token: _,
        opt_ilike,
        opt_exclude,
        opt_except,
        opt_replace,
        opt_rename,
        opt_alias,
    } = options;
    let unsupported = [
        (opt_exclude.is_some(), Unsupported::WildcardExclude),
        (opt_except.is_some(), Unsupported::WildcardExcept),
        (opt_replace.is_some(), Unsupported::WildcardReplace),
        (opt_rename.is_some(), Unsupported::WildcardRename),
        (opt_ilike.is_some(), Unsupported::WildcardIlike),
        (opt_alias.is_some(), Unsupported::WildcardAlias),
    ];
    match unsupported.into_iter().find(|&(present, _)| present) {
        Some((_, feature)) => Err(feature.into()),
        None => Ok(()),
    }
}

fn resolve_limit(clause: Option<&LimitClause>) -> Result<Option<u32>, BinderError> {
    match clause {
        None => Ok(None),
        Some(LimitClause::LimitOffset {
            limit: None,
            offset: None,
            limit_by,
        }) if limit_by.is_empty() => Ok(None),
        Some(LimitClause::LimitOffset {
            limit: Some(Expr::Value(v)),
            offset: None,
            limit_by,
        }) if limit_by.is_empty() => {
            let SqlValue::Number(s, _) = &v.value else {
                return Err(Unsupported::LimitExpression.into());
            };
            s.parse::<u32>()
                .map(Some)
                .map_err(|_| BinderError::InvalidLimit { limit: s.clone() })
        }
        Some(LimitClause::LimitOffset {
            limit: Some(_),
            offset: None,
            limit_by,
        }) if limit_by.is_empty() => Err(Unsupported::LimitExpression.into()),
        _ => Err(Unsupported::LimitClause.into()),
    }
}

/// Maps a function name to an aggregate kind, case-insensitively. Returns `None` for any other
/// function name.
fn parse_agg_type(name: &ObjectName) -> Option<AggType> {
    let part = name.0.first()?;
    let ObjectNamePart::Identifier(id) = part else {
        return None;
    };
    match id.value.to_ascii_uppercase().as_str() {
        "COUNT" => Some(AggType::COUNT),
        "SUM" => Some(AggType::SUM),
        "AVG" => Some(AggType::AVG),
        "MIN" => Some(AggType::MIN),
        "MAX" => Some(AggType::MAX),
        _ => None,
    }
}

/// Maps a window function name to the function the window executor computes, case-insensitively:
/// every aggregate plus RANK. Returns `None` for any other function name.
fn parse_window_function_type(name: &ObjectName) -> Option<AggType> {
    if is_window_only_function(name) {
        return Some(AggType::RANK);
    }
    parse_agg_type(name)
}

/// True for RANK, the one function the binder accepts only with an OVER clause.
fn is_window_only_function(name: &ObjectName) -> bool {
    matches!(
        name.0.as_slice(),
        [ObjectNamePart::Identifier(id)] if id.value.eq_ignore_ascii_case("RANK")
    )
}

/// Rejects a call `f` that passes any argument, as `RANK(x)` does.
fn reject_arguments(f: &Function) -> Result<(), BinderError> {
    match &f.args {
        FunctionArguments::List(list)
            if list.args.is_empty()
                && list.duplicate_treatment.is_none()
                && list.clauses.is_empty() =>
        {
            Ok(())
        }
        _ => Err(BinderError::WindowFunctionArguments {
            function: f.name.to_string(),
        }),
    }
}

/// The inline `OVER (PARTITION BY ... ORDER BY ...)` clause of a window call, rejecting named
/// windows and explicit frames.
fn inline_window_spec(over: &WindowType) -> Result<&WindowSpec, BinderError> {
    let WindowType::WindowSpec(spec) = over else {
        return Err(Unsupported::NamedWindow.into());
    };
    if spec.window_name.is_some() {
        return Err(Unsupported::NamedWindow.into());
    }
    if spec.window_frame.is_some() {
        return Err(Unsupported::WindowFrame.into());
    }
    Ok(spec)
}

/// True iff any projection item syntactically contains an aggregate function
/// call. Used to decide whether to set up an aggregation context even when
/// the query has no GROUP BY (`SELECT COUNT(*) FROM t`).
fn projection_has_aggregate(items: &[SelectItem]) -> bool {
    items.iter().any(|item| match item {
        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
            expr_has_aggregate(e)
        }
        _ => false,
    })
}

fn reject_nested_aggregate(expr: &Expr) -> Result<(), BinderError> {
    match expr {
        Expr::Function(f) if parse_agg_type(&f.name).is_some() => Err(BinderError::NestedAggregate),
        _ => Ok(()),
    }
}

/// The dotted name of a column reference, or `None` if `expr` is not one.
fn column_path(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Identifier(id) => Some(id.value.clone()),
        Expr::CompoundIdentifier(parts) => Some(
            parts
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
                .join("."),
        ),
        _ => None,
    }
}

/// The type of `agg_type` over `arg` (see [`AggType::result_type`]). COUNT and RANK take any
/// argument, a bare NULL included.
fn aggregate_result_type(
    agg_type: &AggType,
    arg: &Expression,
    function: &ObjectName,
) -> Result<DataType, BinderError> {
    if matches!(agg_type, AggType::COUNT | AggType::RANK) {
        return Ok(DataType::INT);
    }
    let arg_type = arg
        .dtype()
        .ok_or_else(|| BinderError::UntypedAggregateArgument {
            function: function.to_string(),
        })?;
    agg_type
        .result_type(arg_type)
        .ok_or_else(|| BinderError::AggregateArgumentType {
            function: function.to_string(),
            arg: arg_type,
        })
}

fn expr_has_aggregate(e: &Expr) -> bool {
    match e {
        Expr::Function(f) => f.over.is_none() && parse_agg_type(&f.name).is_some(),
        Expr::Nested(inner) => expr_has_aggregate(inner),
        Expr::BinaryOp { left, right, .. } => expr_has_aggregate(left) || expr_has_aggregate(right),
        Expr::UnaryOp { expr, .. } => expr_has_aggregate(expr),
        _ => false,
    }
}

fn sql_to_dtype(t: &SqlDataType) -> Result<DataType, BinderError> {
    match t {
        SqlDataType::Int(_) | SqlDataType::Integer(_) | SqlDataType::Int4(_) => Ok(DataType::INT),
        SqlDataType::BigInt(_) | SqlDataType::Int8(_) => Ok(DataType::BIGINT),
        SqlDataType::Float(_)
        | SqlDataType::Float4
        | SqlDataType::Float8
        | SqlDataType::Real
        | SqlDataType::Double(_)
        | SqlDataType::DoublePrecision => Ok(DataType::FLOAT),
        SqlDataType::Bool | SqlDataType::Boolean => Ok(DataType::BOOLEAN),
        SqlDataType::Timestamp(_, TimezoneInfo::None | TimezoneInfo::WithoutTimeZone) => {
            Ok(DataType::TIMESTAMP)
        }
        SqlDataType::Text
        | SqlDataType::Varchar(None)
        | SqlDataType::CharacterVarying(None)
        | SqlDataType::CharVarying(None) => Ok(DataType::TEXT),
        SqlDataType::Varchar(Some(length))
        | SqlDataType::CharacterVarying(Some(length))
        | SqlDataType::CharVarying(Some(length)) => character_limit(length)
            .map(DataType::VARCHAR)
            .ok_or_else(|| BinderError::UnsupportedType {
                type_name: t.to_string(),
            }),
        SqlDataType::Char(None) | SqlDataType::Character(None) => {
            Ok(DataType::CHAR(DEFAULT_CHAR_LENGTH))
        }
        SqlDataType::Char(Some(length)) | SqlDataType::Character(Some(length)) => {
            character_limit(length)
                .filter(|&n| n <= MAX_CHAR_LENGTH)
                .map(DataType::CHAR)
                .ok_or_else(|| BinderError::UnsupportedType {
                    type_name: t.to_string(),
                })
        }
        other => Err(BinderError::UnsupportedType {
            type_name: other.to_string(),
        }),
    }
}

/// The length of a `CHAR` declared without one.
const DEFAULT_CHAR_LENGTH: u32 = 1;

/// The longest `CHAR(n)`, as in Postgres. A result pads each value to `n` characters, so `n`
/// is bounded where a VARCHAR's limit is not.
pub const MAX_CHAR_LENGTH: u32 = 10_485_760;

/// The character limit of a `VARCHAR(n)` or `CHAR(n)`: `n` when it counts characters and is
/// from 1 to `u32::MAX`.
fn character_limit(length: &CharacterLength) -> Option<u32> {
    match length {
        CharacterLength::IntegerLength {
            length,
            unit: None | Some(CharLengthUnits::Characters),
        } => u32::try_from(*length).ok().filter(|&n| n > 0),
        _ => None,
    }
}

/// The column positions an INSERT fills, in the order its rows list values: every column when
/// the INSERT has no column list.
fn insert_targets(schema: &Schema, columns: &[ObjectName]) -> Result<Vec<usize>, BinderError> {
    if columns.is_empty() {
        return Ok((0..schema.cols.len()).collect());
    }
    let mut targets = Vec::with_capacity(columns.len());
    for name in columns {
        let column = ident_name(name)?;
        let idx = schema
            .col_idx(&column)
            .ok_or_else(|| BinderError::ColumnNotFound {
                column: column.clone(),
            })?;
        if targets.contains(&idx) {
            return Err(BinderError::DuplicateInsertColumn { column });
        }
        targets.push(idx);
    }
    Ok(targets)
}

/// The text of a single-quoted string literal, looking through parentheses.
fn string_literal(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Nested(inner) => string_literal(inner),
        Expr::Value(v) => match &v.value {
            SqlValue::SingleQuotedString(s) => Some(s),
            _ => None,
        },
        _ => None,
    }
}

fn timestamp_constant(text: &str) -> Result<Expression, BinderError> {
    timestamp::parse(text)
        .map(|micros| Expression::Constant(Some(Value::TIMESTAMP(micros))))
        .map_err(BinderError::InvalidTimestamp)
}

/// An integer literal as the narrowest of INT and BIGINT that holds it, or else a FLOAT.
fn number_literal(literal: &str) -> Result<Value, BinderError> {
    if let Ok(i) = literal.parse::<i64>() {
        return Ok(Value::integer(i));
    }
    match literal.parse::<f64>() {
        Ok(f) if f.is_finite() => Ok(Value::float(f)),
        _ => Err(BinderError::InvalidNumber {
            literal: literal.to_string(),
        }),
    }
}

/// `-inner` for a numeric (or NULL) expression. A constant is negated here; any other
/// expression becomes `0 - inner`. `resolve_expr` reads a negative number literal such as
/// `-9223372036854775808` as one literal before calling this.
fn negate(inner: Expression) -> Result<Expression, BinderError> {
    match inner {
        Expression::Constant(None) => Ok(Expression::Constant(None)),
        Expression::Constant(Some(v)) => {
            let negated = match &v {
                Value::FLOAT(f) => Some(Value::float(-f.0)),
                other => other
                    .as_i64()
                    .and_then(i64::checked_neg)
                    .map(Value::integer),
            };
            negated
                .map(|v| Expression::Constant(Some(v)))
                .ok_or_else(|| BinderError::InvalidNumber {
                    literal: format!("-{v}"),
                })
        }
        other => Ok(Expression::Binary {
            left: Box::new(Expression::Constant(Some(Value::INT(0)))),
            right: Box::new(other),
            op: Op::Sub,
        }),
    }
}

fn is_comparison(op: &Op) -> bool {
    matches!(op, Op::Eq | Op::NEq | Op::Lt | Op::Gt | Op::Lte | Op::Gte)
}

/// Fails when `op` does not take operands of these types. A `None` side is a bare NULL, which
/// every operator takes.
fn check_operand_types(
    operator: &BinaryOperator,
    op: &Op,
    left: Option<DataType>,
    right: Option<DataType>,
) -> Result<(), BinderError> {
    let takes = |t: Option<DataType>, ok: fn(DataType) -> bool| t.is_none_or(ok);
    let fits = match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            takes(left, |t| t.is_numeric()) && takes(right, |t| t.is_numeric())
        }
        Op::And | Op::Or => {
            let boolean = |t| t == DataType::BOOLEAN;
            takes(left, boolean) && takes(right, boolean)
        }
        Op::Eq | Op::NEq | Op::Lt | Op::Gt | Op::Lte | Op::Gte => match (left, right) {
            (Some(l), Some(r)) => l.compares_with(r),
            _ => true,
        },
    };
    if fits {
        Ok(())
    } else {
        Err(BinderError::OperandTypes {
            operator: operator.clone(),
            left,
            right,
        })
    }
}

/// In a comparison between a constant and an expression of another type, converts the constant
/// with [`Value::exactly_as`]: a number to the other side's numeric type when it converts
/// exactly, so that `f = 1` on a FLOAT column compares FLOATs and can use an index on `f`, and a
/// string compared with a CHAR to one without trailing spaces, so that `c = 'ab   '` matches
/// the CHAR value `'ab'` and can use an index on `c`.
fn match_constant_to_other_side(l: Expression, r: Expression) -> (Expression, Expression) {
    let exactly = |constant: &Expression, target: Option<DataType>| match (constant, target) {
        (Expression::Constant(Some(v)), Some(t)) => v.clone().exactly_as(t),
        _ => None,
    };
    match (&l, &r) {
        (Expression::Constant(_), Expression::Constant(_)) => (l, r),
        (Expression::Constant(_), other) => match exactly(&l, other.dtype()) {
            Some(v) => (Expression::Constant(Some(v)), r),
            None => (l, r),
        },
        (other, Expression::Constant(_)) => match exactly(&r, other.dtype()) {
            Some(v) => (l, Expression::Constant(Some(v))),
            None => (l, r),
        },
        _ => (l, r),
    }
}

pub enum BoundStatement {
    Select(BoundSelect),
    Values(BoundValues),
    Insert(BoundInsert),
    Delete(BoundDelete),
    Update(BoundUpdate),
    CreateTable(BoundCreateTable),
    CreateIndex(BoundCreateIndex),
    DropTable(BoundDropTable),
    DropIndex(BoundDropIndex),
}

/// A bound SELECT: the tables in scope, an optional two-way join, projections, filters, grouping,
/// ordering and an optional row limit.
pub struct BoundSelect {
    pub scope: Scope,
    pub join: Option<BoundJoin>,
    pub projections: Vec<Expression>,
    /// `SELECT DISTINCT`: duplicate projected rows are removed before ORDER BY and LIMIT.
    pub distinct: bool,
    pub filters: Option<Vec<Expression>>,
    /// GROUP BY expressions (resolved against the child scope). Empty when
    /// the query has no GROUP BY clause. When `aggregates` is also empty,
    /// the planner skips the Aggregation node entirely.
    pub group_bys: Vec<Expression>,
    /// Aggregate function calls discovered while resolving the projection
    /// and ORDER BY. Each is `(argument_expr, agg_type)` and is evaluated
    /// against the child schema by the AggregationExecutor.
    pub aggregates: Vec<(Expression, AggType)>,
    /// Window function calls grouped by OVER clause, in order of first use. Empty unless the
    /// SELECT calls one, and empty whenever `group_bys` or `aggregates` is not. The projection and
    /// ORDER BY read call `j` of window `k` at column `from_width + c + j`, where `from_width` is
    /// the FROM output's width and `c` counts the calls of windows `0..k`.
    pub windows: Vec<BoundWindow>,
    /// ORDER BY keys. They read the aggregation output (or the FROM output, widened by any window
    /// calls, when nothing is aggregated), or the projection output when `distinct` is set.
    pub sort: Option<Vec<SortExpr>>,
    pub limit: Option<u32>,
}

/// One OVER clause of a SELECT and the window function calls over it.
pub struct BoundWindow {
    pub partition_by: Vec<Expression>,
    pub order_by: Vec<SortExpr>,
    pub calls: Vec<BoundWindowCall>,
}

/// A window function call: its argument, evaluated against the FROM output, the function, and
/// the type of its result, which the SELECT's projection and ORDER BY read.
pub struct BoundWindowCall {
    pub arg: Expression,
    pub agg_type: AggType,
    pub dtype: DataType,
}

pub struct BoundJoin {
    pub join_type: JoinType,
    pub predicate: JoinPredicate,
}

pub struct BoundValues {
    pub rows: Vec<Vec<Expression>>,
}

pub struct BoundInsert {
    pub scope: Scope,
    pub rows: Vec<Vec<Expression>>,
}

pub struct BoundUpdate {
    pub scope: Scope,
    pub cols: Vec<ColIdx>,
    pub new_cols: Vec<Expression>,
    pub filters: Option<Vec<Expression>>,
}

pub struct BoundDelete {
    pub scope: Scope,
    pub filters: Option<Vec<Expression>>,
}

pub struct BoundCreateTable {
    pub table_name: String,
    pub schema: Schema,
}

pub struct BoundCreateIndex {
    pub scope: Scope,
    pub name: String,
    pub index_schema: Schema,
}

/// A bound `DROP TABLE`. `table_oid` is `None` when the table does not exist and the statement
/// says `IF EXISTS`.
pub struct BoundDropTable {
    pub table_oid: Option<TableOid>,
    pub if_exists: bool,
}

/// A bound `DROP INDEX`. `index_oid` is `None` when the index does not exist and the statement
/// says `IF EXISTS`.
pub struct BoundDropIndex {
    pub index_oid: Option<u32>,
    pub if_exists: bool,
}

/// The tables a statement can reference, keyed by the alias (or table name) used in the SQL.
pub type Scope = HashMap<TableAlias, TableScope>;
pub struct TableScope {
    pub oid: TableOid,
    pub side: Side,
    /// Position of this table's first column in the statement's FROM output.
    pub first_col: usize,
    pub schema: Schema,
}

impl TableScope {
    /// A reference to this table's column `idx`, positioned in the FROM output.
    fn column(&self, idx: usize) -> Expression {
        Expression::Column {
            col: ColIdx((self.first_col + idx) as u32),
            dtype: self.schema.cols[idx].dtype,
        }
    }
}

pub type TableAlias = String;
pub type TableOid = u32;

/// Rewrites an ON predicate bound over the FROM output into one that reads the left and right
/// rows of the join, given the number of columns the left table contributes.
fn join_predicate(predicate: Expression, left_cols: usize) -> JoinPredicate {
    predicate.map_columns(|col| match col.index().checked_sub(left_cols) {
        None => JoinColumn {
            side: Side::Left,
            col,
        },
        Some(right_idx) => JoinColumn {
            side: Side::Right,
            col: ColIdx(right_idx as u32),
        },
    })
}

#[cfg(test)]
mod tests {

    use sqlparser::dialect::{GenericDialect, RedshiftSqlDialect};
    use sqlparser::parser::Parser;

    use super::{Binder, BinderError, BoundDropIndex, BoundDropTable, BoundStatement, Unsupported};
    use crate::catalog::{Catalog, DataType, IndexType, Schema};
    use crate::processor::plan::{ColIdx, Expression, JoinColumn, JoinPredicate, Side};

    fn catalog_with_tables(tables: &[(&str, &[&str])]) -> (Catalog, tempfile::TempDir) {
        let (catalog, _bpm, dir) = crate::testing::catalog(16);
        for (name, cols) in tables {
            let schema = Schema::new(
                cols.iter()
                    .map(|c| (DataType::INT, c.to_string()))
                    .collect(),
            );
            catalog.create_table(name.to_string(), schema).unwrap();
        }
        (catalog, dir)
    }

    fn bind_one(catalog: &Catalog, sql: &str) -> Result<BoundStatement, BinderError> {
        let mut ast = Parser::parse_sql(&GenericDialect {}, sql).expect("valid SQL");
        assert_eq!(ast.len(), 1, "expected exactly one statement");
        Binder::new(catalog).bind_statement(&ast.remove(0))
    }

    fn column_refs(exprs: &[Expression]) -> Vec<u32> {
        exprs
            .iter()
            .map(|e| match e {
                Expression::Column { col, .. } => col.0,
                _ => panic!("expected only column references"),
            })
            .collect()
    }

    #[test]
    fn select_star_over_a_join_lists_left_columns_before_right_columns() {
        let (catalog, _dir) = catalog_with_tables(&[("l", &["a", "b"]), ("r", &["c"])]);
        for _ in 0..32 {
            let Ok(BoundStatement::Select(sel)) =
                bind_one(&catalog, "SELECT * FROM l JOIN r ON l.a = r.c")
            else {
                panic!("expected a bound SELECT");
            };
            assert_eq!(column_refs(&sel.projections), vec![0, 1, 2]);
        }
    }

    fn bind_error(catalog: &Catalog, sql: &str) -> BinderError {
        match bind_one(catalog, sql) {
            Ok(_) => panic!("expected `{sql}` to fail to bind"),
            Err(e) => e,
        }
    }

    #[test]
    fn wildcard_alias_is_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        let mut ast = Parser::parse_sql(&RedshiftSqlDialect {}, "SELECT * AS x FROM t")
            .expect("Redshift parses a wildcard alias");
        let bound = Binder::new(&catalog).bind_statement(&ast.remove(0));
        assert!(
            matches!(
                bound,
                Err(BinderError::Unsupported(Unsupported::WildcardAlias))
            ),
            "a wildcard alias fails to bind"
        );
    }

    #[test]
    fn join_predicate_reads_each_column_from_its_own_side() {
        let (catalog, _dir) = catalog_with_tables(&[("l", &["a", "b"]), ("r", &["c"])]);
        let Ok(BoundStatement::Select(sel)) =
            bind_one(&catalog, "SELECT * FROM l JOIN r ON l.b = r.c")
        else {
            panic!("expected a bound SELECT");
        };
        let join = sel.join.expect("a bound join");
        let Expression::Binary { left, right, .. } = join.predicate else {
            panic!("expected `l.b = r.c`");
        };
        let side_of = |e: &JoinPredicate| match e {
            Expression::Column { col, .. } => *col,
            _ => panic!("expected a column reference"),
        };
        assert_eq!(
            side_of(&left),
            JoinColumn {
                side: Side::Left,
                col: ColIdx(1)
            }
        );
        assert_eq!(
            side_of(&right),
            JoinColumn {
                side: Side::Right,
                col: ColIdx(0)
            }
        );
    }

    #[test]
    fn insert_values_referencing_a_column_is_a_bind_error() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        assert_eq!(
            bind_error(&catalog, "INSERT INTO t VALUES (a)"),
            BinderError::ColumnNotFound { column: "a".into() }
        );
        assert_eq!(
            bind_error(&catalog, "INSERT INTO t VALUES (t.a + 1)"),
            BinderError::TableNotInScope { table: "t".into() }
        );
    }

    #[test]
    fn aggregate_in_where_is_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        assert_eq!(
            bind_error(&catalog, "SELECT a FROM t WHERE COUNT(a) > 1"),
            BinderError::AggregateOutsideProjection
        );
    }

    #[test]
    fn nested_aggregate_is_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        assert_eq!(
            bind_error(&catalog, "SELECT SUM(COUNT(a)) FROM t"),
            BinderError::NestedAggregate
        );
    }

    #[test]
    fn ungrouped_column_in_aggregating_select_is_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a", "b"])]);
        assert_eq!(
            bind_error(&catalog, "SELECT a, b FROM t GROUP BY a"),
            BinderError::UngroupedColumn { column: "b".into() }
        );
        assert_eq!(
            bind_error(&catalog, "SELECT t.b, COUNT(a) FROM t"),
            BinderError::UngroupedColumn {
                column: "t.b".into()
            }
        );
    }

    #[test]
    fn group_by_and_aggregates_map_to_aggregation_output_columns() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a", "b"])]);
        let Ok(BoundStatement::Select(sel)) =
            bind_one(&catalog, "SELECT a, SUM(b), COUNT(*) FROM t GROUP BY a")
        else {
            panic!("expected a bound SELECT");
        };
        assert_eq!(column_refs(&sel.projections), vec![0, 1, 2]);
        assert_eq!(sel.group_bys.len(), 1);
        assert_eq!(sel.aggregates.len(), 2);
    }

    #[test]
    fn count_star_counts_every_row_through_a_non_null_argument_and_yields_int() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        let Ok(BoundStatement::Select(sel)) = bind_one(&catalog, "SELECT COUNT(*) FROM t") else {
            panic!("expected a bound SELECT");
        };
        let [(arg, _)] = sel.aggregates.as_slice() else {
            panic!("expected one aggregate");
        };
        assert!(matches!(arg, Expression::Constant(Some(_))));
        assert_eq!(sel.projections[0].dtype(), Some(DataType::INT));
        assert_eq!(
            bind_error(&catalog, "SELECT SUM(*) FROM t"),
            BinderError::StarArgumentNotCount {
                function: "SUM".into()
            }
        );
    }

    #[test]
    fn bare_null_projection_is_rejected_but_null_in_arithmetic_is_typed_by_its_sibling() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        assert_eq!(
            bind_error(&catalog, "SELECT NULL FROM t"),
            BinderError::BareNullProjection
        );
        let Ok(BoundStatement::Select(sel)) = bind_one(&catalog, "SELECT a + NULL FROM t") else {
            panic!("expected a bound SELECT");
        };
        assert_eq!(sel.projections[0].dtype(), Some(DataType::INT));
    }

    #[test]
    fn joins_of_more_than_two_tables_are_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("a", &["x"]), ("b", &["y"]), ("c", &["z"])]);
        assert_eq!(
            bind_error(
                &catalog,
                "SELECT * FROM a JOIN b ON a.x = b.y JOIN c ON b.y = c.z",
            ),
            BinderError::Unsupported(Unsupported::MultiWayJoin)
        );
    }

    #[test]
    fn failed_aggregating_select_does_not_leak_into_the_next_statement() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a", "b"])]);
        let binder = Binder::new(&catalog);
        let parse = |sql: &str| {
            Parser::parse_sql(&GenericDialect {}, sql)
                .unwrap()
                .remove(0)
        };
        assert!(
            binder
                .bind_statement(&parse("SELECT a, b FROM t GROUP BY a"))
                .is_err()
        );
        assert!(
            binder.bind_statement(&parse("UPDATE t SET a = b")).is_ok(),
            "a later statement must not see the failed SELECT's GROUP BY"
        );
    }

    #[test]
    fn aggregate_call_shapes_the_binder_does_not_handle_are_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        assert_eq!(
            bind_error(&catalog, "SELECT SUM(a, a) FROM t"),
            BinderError::AggregateArgumentCount {
                function: "SUM".into(),
                args: 2
            }
        );
        assert_eq!(
            bind_error(&catalog, "SELECT COUNT(DISTINCT a) FROM t"),
            BinderError::Unsupported(Unsupported::AggregateDistinct)
        );
        assert_eq!(
            bind_error(&catalog, "SELECT SUM(NULL) FROM t"),
            BinderError::UntypedAggregateArgument {
                function: "SUM".into()
            }
        );
    }

    #[test]
    fn limit_must_be_a_u32_literal() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        assert_eq!(
            bind_error(&catalog, "SELECT a FROM t LIMIT 99999999999"),
            BinderError::InvalidLimit {
                limit: "99999999999".into()
            }
        );
        for limit in ["a", "1 + 1", "-1", "'x'"] {
            assert_eq!(
                bind_error(&catalog, &format!("SELECT a FROM t LIMIT {limit}")),
                BinderError::Unsupported(Unsupported::LimitExpression),
                "LIMIT {limit}"
            );
        }
        assert_eq!(
            bind_error(&catalog, "SELECT a FROM t LIMIT 1 OFFSET 1"),
            BinderError::Unsupported(Unsupported::LimitClause)
        );
    }

    #[test]
    fn binder_errors_display_as_readable_messages() {
        let cases = [
            (
                BinderError::TableNotFound { table: "t".into() },
                "table not found: t",
            ),
            (
                BinderError::AmbiguousColumn {
                    column: "id".into(),
                },
                "ambiguous column: id",
            ),
            (
                BinderError::UngroupedColumn { column: "b".into() },
                "column `b` must appear in GROUP BY or be used inside an aggregate function",
            ),
            (
                BinderError::AggregateArgumentCount {
                    function: "SUM".into(),
                    args: 2,
                },
                "aggregate `SUM` expects exactly one argument, got 2",
            ),
            (
                BinderError::UnsupportedOperator {
                    operator: sqlparser::ast::BinaryOperator::Modulo,
                },
                "unsupported binary operator: %",
            ),
            (
                BinderError::Unsupported(Unsupported::GroupByAll),
                "GROUP BY ALL is not supported",
            ),
        ];
        for (err, message) in cases {
            assert_eq!(err.to_string(), message, "{err:?}");
        }
    }

    #[test]
    fn bind_resolves_every_statement_against_the_current_catalog() {
        let (catalog, _dir) = catalog_with_tables(&[]);
        let ast = Parser::parse_sql(
            &GenericDialect {},
            "CREATE TABLE t (a INTEGER); INSERT INTO t VALUES (1)",
        )
        .unwrap();
        let err = Binder::new(&catalog)
            .bind(ast)
            .err()
            .expect("INSERT cannot see t yet");
        assert_eq!(err, BinderError::TableNotFound { table: "t".into() });
    }

    #[test]
    fn drop_table_binds_to_the_tables_oid_or_to_none_under_if_exists() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        let oid = catalog.table_oid("t").unwrap();
        let Ok(BoundStatement::DropTable(BoundDropTable {
            table_oid: Some(bound),
            if_exists: false,
        })) = bind_one(&catalog, "DROP TABLE t")
        else {
            panic!("expected DROP TABLE t to bind to t's oid");
        };
        assert_eq!(bound, oid);
        let Ok(BoundStatement::DropTable(BoundDropTable {
            table_oid: None,
            if_exists: true,
        })) = bind_one(&catalog, "DROP TABLE IF EXISTS gone")
        else {
            panic!("expected DROP TABLE IF EXISTS gone to bind with no target");
        };
        assert_eq!(
            bind_error(&catalog, "DROP TABLE gone"),
            BinderError::TableNotFound {
                table: "gone".into()
            }
        );
        assert_eq!(
            bind_error(&catalog, "DROP TABLE s.t"),
            BinderError::QualifiedName { name: "s.t".into() }
        );
    }

    #[test]
    fn drop_index_finds_the_index_by_name_across_tables() {
        let (catalog, bpm, _dir) = crate::testing::catalog(16);
        let int_col = |name: &str| Schema::new(vec![(DataType::INT, name.to_string())]);
        let t_oid = catalog.create_table("t".into(), int_col("a")).unwrap();
        catalog.create_table("u".into(), int_col("b")).unwrap();
        let index_oid = catalog
            .create_index(bpm, t_oid, "t_a".into(), &["a"], IndexType::BPlusTree)
            .unwrap();
        for sql in ["DROP INDEX t_a", "DROP INDEX t_a ON t"] {
            let Ok(BoundStatement::DropIndex(BoundDropIndex {
                index_oid: Some(bound),
                if_exists: false,
            })) = bind_one(&catalog, sql)
            else {
                panic!("expected {sql} to bind to t_a's oid");
            };
            assert_eq!(bound, index_oid, "{sql}");
        }
        assert_eq!(
            bind_error(&catalog, "DROP INDEX t_a ON u"),
            BinderError::IndexNotFound {
                index: "t_a".into()
            }
        );
        assert_eq!(
            bind_error(&catalog, "DROP TABLE t ON u"),
            Unsupported::DropTableOn.into()
        );
        let Ok(BoundStatement::DropIndex(BoundDropIndex {
            index_oid: None,
            if_exists: true,
        })) = bind_one(&catalog, "DROP INDEX IF EXISTS t_a ON u")
        else {
            panic!("expected IF EXISTS to bind with no target");
        };
    }
}
