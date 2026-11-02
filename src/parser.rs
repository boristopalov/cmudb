use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;

use ordered_float::OrderedFloat;
use sqlparser::ast::{
    Assignment, AssignmentTarget, BinaryOperator, CreateIndex as SqlCreateIndex,
    CreateTable as SqlCreateTable, DataType as SqlDataType, Delete, Expr, FromTable, Function,
    FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Insert, JoinConstraint,
    JoinOperator, LimitClause, ObjectName, ObjectNamePart, OrderByExpr, OrderByKind, Query,
    SelectItem, SetExpr, Statement, TableFactor, TableObject, TableWithJoins, Update,
    Value as SqlValue,
};

use crate::catalog::{Catalog, DataType, Schema, Value};
use crate::processor::plan::{
    AggType, ColIdx, Expression, JoinColumn, JoinPredicate, JoinType, NullOrdering, Op, Side,
    SortDirection, SortExpr,
};

/// Argument bound for `COUNT(*)`. COUNT skips NULL arguments, so a non-NULL constant makes it count
/// every row.
const COUNT_STAR_ARGUMENT: Expression = Expression::Constant(Some(Value::INT(1)));

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
    /// A numeric literal fits neither an INT nor a FLOAT.
    InvalidNumber { literal: String },
    /// A LIMIT literal is not a non-negative integer that fits a `u32`.
    InvalidLimit { limit: String },
    /// A table or index name has more than one part, such as `schema.table`.
    QualifiedName { name: String },
    /// A column type other than INT, FLOAT, BOOLEAN or TIMESTAMP.
    UnsupportedType { type_name: String },
    /// A binary operator with no bound counterpart, such as `%`.
    UnsupportedOperator { operator: BinaryOperator },
    /// A literal other than a number, a boolean or NULL.
    UnsupportedLiteral { literal: String },
    /// An expression form the binder does not handle, such as `IN (...)`.
    UnsupportedExpression { expr: String },
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
            BinderError::Unsupported(feature) => write!(f, "{feature}"),
        }
    }
}

impl std::error::Error for BinderError {}

impl From<Unsupported> for BinderError {
    fn from(feature: Unsupported) -> Self {
        BinderError::Unsupported(feature)
    }
}

/// A statement shape or clause the binder does not handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// A statement other than SELECT, VALUES, INSERT, DELETE, UPDATE, CREATE TABLE or
    /// CREATE INDEX.
    Statement,
    /// A query body other than SELECT, VALUES or a parenthesized query.
    QueryBody,
    /// A select item other than an unaliased expression or `*`.
    SelectItem,
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
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Unsupported::Statement => "unsupported statement",
            Unsupported::QueryBody => "unsupported query body",
            Unsupported::SelectItem => "unsupported select item",
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
            Unsupported::InsertTarget => "unsupported INSERT target",
            Unsupported::InsertWithoutSource => "INSERT without source",
            Unsupported::InsertSource => "INSERT requires VALUES",
            Unsupported::DeleteTableCount => "DELETE expects one table",
            Unsupported::DeleteTarget => "DELETE target must be a table",
            Unsupported::UpdateTarget => "UPDATE target must be a table",
            Unsupported::TupleAssignment => "tuple UPDATE assignment not supported",
            Unsupported::UnnamedIndex => "CREATE INDEX without a name",
            Unsupported::IndexKeyExpression => "index columns must be plain identifiers",
        })
    }
}

impl<'a> Binder<'a> {
    pub fn new(catalog: &'a Catalog) -> Self {
        Self {
            catalog,
            agg_ctx: RefCell::new(None),
            inside_agg_arg: Cell::new(false),
        }
    }

    fn resolve_expr(&self, expr: &Expr, scope: &Scope) -> Result<Expression, BinderError> {
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

            Expr::BinaryOp { left, op, right } => {
                let op = match op {
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
                let l = self.resolve_expr(left, scope)?;
                let r = self.resolve_expr(right, scope)?;
                Ok(Expression::Binary {
                    left: Box::new(l),
                    right: Box::new(r),
                    op,
                })
            }

            Expr::Value(v) => {
                let val = match &v.value {
                    SqlValue::Null => None,
                    SqlValue::Boolean(b) => Some(Value::BOOLEAN(*b)),
                    SqlValue::Number(s, _) => {
                        if let Ok(i) = s.parse::<i32>() {
                            Some(Value::INT(i))
                        } else if let Ok(f) = s.parse::<f32>() {
                            Some(Value::FLOAT(OrderedFloat(f)))
                        } else {
                            return Err(BinderError::InvalidNumber { literal: s.clone() });
                        }
                    }
                    other => {
                        return Err(BinderError::UnsupportedLiteral {
                            literal: other.to_string(),
                        });
                    }
                };
                Ok(Expression::Constant(val))
            }

            other => Err(BinderError::UnsupportedExpression {
                expr: other.to_string(),
            }),
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

        let arg_expr = match arg {
            FunctionArgExpr::Wildcard => {
                if !matches!(agg_type, AggType::COUNT) {
                    return Err(BinderError::StarArgumentNotCount {
                        function: f.name.to_string(),
                    });
                }
                COUNT_STAR_ARGUMENT
            }
            FunctionArgExpr::Expr(e) => {
                let was_inside = self.inside_agg_arg.replace(true);
                let result = self.resolve_expr(e, scope);
                self.inside_agg_arg.set(was_inside);
                result?
            }
            FunctionArgExpr::QualifiedWildcard(_) | FunctionArgExpr::WildcardWithOptions(_) => {
                return Err(Unsupported::AggregateQualifiedWildcard.into());
            }
        };

        let result_dtype = aggregate_result_type(&agg_type, &arg_expr).ok_or_else(|| {
            BinderError::UntypedAggregateArgument {
                function: f.name.to_string(),
            }
        })?;

        let mut ctx_mut = self.agg_ctx.borrow_mut();
        let ctx = ctx_mut
            .as_mut()
            .ok_or(BinderError::AggregateOutsideProjection)?;
        let col_idx = ctx.group_bys.len() + ctx.aggregates.len();
        ctx.aggregates.push((arg_expr, agg_type));
        Ok(Expression::Column {
            col: ColIdx(col_idx as u32),
            dtype: result_dtype,
        })
    }

    /// Runs `f` with `ctx` as the active aggregation context, then restores the previous context.
    /// Returns `f`'s result together with the context as `f` left it.
    fn with_agg_ctx<R>(&self, ctx: Option<AggCtx>, f: impl FnOnce() -> R) -> (R, Option<AggCtx>) {
        let prev = self.agg_ctx.replace(ctx);
        let result = f();
        (result, self.agg_ctx.replace(prev))
    }

    fn resolve_select(&self, q: &Query) -> Result<BoundStatement, BinderError> {
        match q.body.as_ref() {
            SetExpr::Query(inner) => self.resolve_select(inner),

            SetExpr::Values(values) => {
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
                let (resolved, agg_ctx) = self.with_agg_ctx(agg_ctx, || {
                    let projections = self.resolve_projections(&s.projection, &scope)?;
                    let sort = self.resolve_order_by(q, &scope)?;
                    Ok::<_, BinderError>((projections, sort))
                });
                let (projections, sort) = resolved?;
                let limit = resolve_limit(q.limit_clause.as_ref())?;
                let (group_bys, aggregates) = agg_ctx
                    .map(|ctx| (ctx.group_bys, ctx.aggregates))
                    .unwrap_or_default();

                Ok(BoundStatement::Select(BoundSelect {
                    scope,
                    join,
                    projections,
                    filters,
                    group_bys,
                    aggregates,
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
                SelectItem::Wildcard(_) => {
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
        let scope = self.build_scope(&ident_name(name)?)?;

        let source = stmt
            .source
            .as_ref()
            .ok_or(Unsupported::InsertWithoutSource)?;
        let SetExpr::Values(values) = source.body.as_ref() else {
            return Err(Unsupported::InsertSource.into());
        };

        let no_input_columns = HashMap::new();
        let rows = values
            .rows
            .iter()
            .map(|row| {
                row.content
                    .iter()
                    .map(|e| self.resolve_expr(e, &no_input_columns))
                    .collect()
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
            new_cols.push(self.resolve_expr(value, &scope)?);
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

/// COUNT yields INT; every other aggregate yields its argument's type.
fn aggregate_result_type(agg_type: &AggType, arg: &Expression) -> Option<DataType> {
    match agg_type {
        AggType::COUNT => Some(DataType::INT),
        _ => arg.dtype(),
    }
}

fn expr_has_aggregate(e: &Expr) -> bool {
    match e {
        Expr::Function(f) => parse_agg_type(&f.name).is_some(),
        Expr::Nested(inner) => expr_has_aggregate(inner),
        Expr::BinaryOp { left, right, .. } => expr_has_aggregate(left) || expr_has_aggregate(right),
        _ => false,
    }
}

fn sql_to_dtype(t: &SqlDataType) -> Result<DataType, BinderError> {
    match t {
        SqlDataType::Int(_) | SqlDataType::Integer(_) | SqlDataType::Int4(_) => Ok(DataType::INT),
        SqlDataType::Float(_) | SqlDataType::Float4 | SqlDataType::Real => Ok(DataType::FLOAT),
        SqlDataType::Bool | SqlDataType::Boolean => Ok(DataType::BOOLEAN),
        SqlDataType::Timestamp(_, _) => Ok(DataType::TIMESTAMP),
        other => Err(BinderError::UnsupportedType {
            type_name: other.to_string(),
        }),
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
}

/// A bound SELECT: the tables in scope, an optional two-way join, projections, filters, grouping,
/// ordering and an optional row limit.
pub struct BoundSelect {
    pub scope: Scope,
    pub join: Option<BoundJoin>,
    pub projections: Vec<Expression>,
    pub filters: Option<Vec<Expression>>,
    /// GROUP BY expressions (resolved against the child scope). Empty when
    /// the query has no GROUP BY clause. When `aggregates` is also empty,
    /// the planner skips the Aggregation node entirely.
    pub group_bys: Vec<Expression>,
    /// Aggregate function calls discovered while resolving the projection
    /// and ORDER BY. Each is `(argument_expr, agg_type)` and is evaluated
    /// against the child schema by the AggregationExecutor.
    pub aggregates: Vec<(Expression, AggType)>,
    pub sort: Option<Vec<SortExpr>>,
    pub limit: Option<u32>,
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

    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    use super::{Binder, BinderError, BoundStatement, Unsupported};
    use crate::catalog::{Catalog, DataType, Schema};
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
        assert_eq!(
            bind_error(&catalog, "SELECT a FROM t LIMIT a"),
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
}
