use std::cell::{Cell, RefCell};
use std::collections::HashMap;

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
use crate::processor::plan::{AggType, Expression, JoinType, Op, SortDirection, SortExpr};

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

#[derive(Debug)]
pub struct BinderError(pub String);

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
            reject_aggregate_call(expr, "nested aggregate functions are not allowed")?;
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
                            return Err(BinderError(format!("ambiguous column: {name}")));
                        }
                        found = Some(Expression::Column {
                            tuple_idx: ts.idx,
                            col_idx: idx as u32,
                            dtype: ts.schema.cols[idx].dtype,
                        });
                    }
                }
                found.ok_or_else(|| BinderError(format!("column not found: {name}")))
            }

            Expr::CompoundIdentifier(parts) => {
                let [t, c] = parts.as_slice() else {
                    return Err(BinderError(
                        "only `table.column` references are supported".into(),
                    ));
                };
                let ts = scope
                    .get(&t.value)
                    .ok_or_else(|| BinderError(format!("table not in scope: {}", t.value)))?;
                let idx = ts
                    .schema
                    .col_idx(&c.value)
                    .ok_or_else(|| BinderError(format!("column not found: {}", c.value)))?;
                Ok(Expression::Column {
                    tuple_idx: ts.idx,
                    col_idx: idx as u32,
                    dtype: ts.schema.cols[idx].dtype,
                })
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
                        return Err(BinderError(format!("unsupported binary operator: {other}")));
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
                            return Err(BinderError(format!("invalid numeric literal: {s}")));
                        }
                    }
                    other => return Err(BinderError(format!("unsupported literal: {other}"))),
                };
                Ok(Expression::Constant(val))
            }

            other => Err(BinderError(format!("unsupported expression: {other}"))),
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
                .ok_or_else(|| BinderError("GROUP BY expression has no inferable type".into()))?;
            return Ok(Some(Expression::Column {
                tuple_idx: 0,
                col_idx: idx as u32,
                dtype,
            }));
        }
        if let Some(path) = column_path(expr) {
            return Err(BinderError(format!(
                "column `{path}` must appear in GROUP BY or be used inside an aggregate function"
            )));
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
            return Err(aggregate_outside_projection());
        }
        let FunctionArguments::List(list) = &f.args else {
            return Err(BinderError(format!(
                "aggregate `{}` must be called with arguments",
                f.name
            )));
        };
        if list.duplicate_treatment.is_some() {
            return Err(BinderError(
                "DISTINCT inside aggregate functions is not supported".into(),
            ));
        }
        if !list.clauses.is_empty() {
            return Err(BinderError(
                "aggregate function clauses (ORDER BY / WITHIN GROUP / ...) are not supported"
                    .into(),
            ));
        }
        if list.args.len() != 1 {
            return Err(BinderError(format!(
                "aggregate `{}` expects exactly one argument",
                f.name
            )));
        }
        let FunctionArg::Unnamed(arg) = &list.args[0] else {
            return Err(BinderError(
                "named arguments to aggregate functions are not supported".into(),
            ));
        };

        let arg_expr = match arg {
            FunctionArgExpr::Wildcard => {
                if !matches!(agg_type, AggType::COUNT) {
                    return Err(BinderError(format!(
                        "`*` is only valid as an argument to COUNT, not {}",
                        f.name
                    )));
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
                return Err(BinderError(
                    "qualified wildcards in aggregates are not supported".into(),
                ));
            }
        };

        let result_dtype = aggregate_result_type(&agg_type, &arg_expr).ok_or_else(|| {
            BinderError(format!(
                "argument to aggregate `{}` has no inferable type",
                f.name
            ))
        })?;

        let mut ctx_mut = self.agg_ctx.borrow_mut();
        let ctx = ctx_mut.as_mut().ok_or_else(aggregate_outside_projection)?;
        let col_idx = ctx.group_bys.len() + ctx.aggregates.len();
        ctx.aggregates.push((arg_expr, agg_type));
        Ok(Expression::Column {
            tuple_idx: 0,
            col_idx: col_idx as u32,
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
                            return Err(BinderError(
                                "GROUP BY modifiers (ROLLUP/CUBE/...) are not supported".into(),
                            ));
                        }
                        exprs.clone()
                    }
                    GroupByExpr::All(_) => {
                        return Err(BinderError("GROUP BY ALL is not supported".into()));
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

            _ => Err(BinderError("unsupported query body".into())),
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
            return Err(BinderError(
                "ORDER BY ... INTERPOLATE is not supported".into(),
            ));
        }
        match &ob.kind {
            OrderByKind::Expressions(exprs) => exprs
                .iter()
                .map(|oe| self.resolve_sort_expr(oe, scope))
                .collect::<Result<_, _>>()
                .map(Some),
            OrderByKind::All(_) => Err(BinderError("ORDER BY ALL not supported".into())),
        }
    }

    fn resolve_sort_expr(&self, oe: &OrderByExpr, scope: &Scope) -> Result<SortExpr, BinderError> {
        if oe.options.nulls_first.is_some() {
            return Err(BinderError(
                "ORDER BY ... NULLS FIRST/LAST is not supported".into(),
            ));
        }
        if oe.with_fill.is_some() {
            return Err(BinderError(
                "ORDER BY ... WITH FILL is not supported".into(),
            ));
        }
        let direction = match oe.options.asc {
            Some(false) => SortDirection::Descending,
            Some(true) | None => SortDirection::Ascending,
        };
        Ok(SortExpr {
            expr: self.resolve_expr(&oe.expr, scope)?,
            direction,
        })
    }

    /// Builds a scope from the FROM clause. We allow at most two tables, joined
    /// by a single INNER/LEFT JOIN ... ON. The left table gets tuple_idx 0, the
    /// right gets tuple_idx 1 — matching how the executors thread tuples.
    fn build_select_scope(
        &self,
        from: &[TableWithJoins],
    ) -> Result<(Scope, Option<BoundJoin>), BinderError> {
        if from.is_empty() {
            return Ok((HashMap::new(), None));
        }
        let [twj] = from else {
            return Err(BinderError(
                "comma-separated FROM tables not supported".into(),
            ));
        };

        let mut scope = HashMap::new();
        let (left_alias, left) = self.bind_table_factor(&twj.relation, 0)?;
        scope.insert(left_alias, left);

        let join = match twj.joins.as_slice() {
            [] => None,
            [j] => {
                let (right_alias, right) = self.bind_table_factor(&j.relation, 1)?;
                scope.insert(right_alias, right);

                let (join_type, predicate_expr) = match &j.join_operator {
                    JoinOperator::Inner(c) | JoinOperator::Join(c) => (JoinType::Inner, c),
                    JoinOperator::Left(c) | JoinOperator::LeftOuter(c) => (JoinType::Left, c),
                    _ => return Err(BinderError("unsupported join type".into())),
                };
                let JoinConstraint::On(expr) = predicate_expr else {
                    return Err(BinderError(
                        "only ON <expr> join constraints supported".into(),
                    ));
                };
                let predicate = self.resolve_expr(expr, &scope)?;
                Some(BoundJoin {
                    join_type,
                    predicate,
                })
            }
            _ => return Err(BinderError("only 2-way joins supported".into())),
        };

        Ok((scope, join))
    }

    fn bind_table_factor(
        &self,
        tf: &TableFactor,
        idx: TupleIndex,
    ) -> Result<(TableAlias, TableScope), BinderError> {
        let TableFactor::Table { name, alias, .. } = tf else {
            return Err(BinderError("FROM target must be a plain table".into()));
        };
        let table_name = ident_name(name)?;
        let oid = self
            .catalog
            .table_oid(&table_name)
            .map_err(|_| BinderError(format!("table not found: {table_name}")))?;
        let tinfo = self
            .catalog
            .get_table(oid)
            .map_err(|_| BinderError(format!("table not found: {table_name}")))?;
        let key = alias
            .as_ref()
            .map(|a| a.name.value.clone())
            .unwrap_or(table_name);
        Ok((
            key,
            TableScope {
                oid,
                idx,
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
                        return Err(BinderError(
                            "bare NULL not allowed in SELECT projection".into(),
                        ));
                    }
                    out.push(resolved);
                }
                SelectItem::Wildcard(_) => {
                    let mut tables: Vec<&TableScope> = scope.values().collect();
                    tables.sort_by_key(|ts| ts.idx);
                    for ts in tables {
                        for (i, col) in ts.schema.cols.iter().enumerate() {
                            out.push(Expression::Column {
                                tuple_idx: ts.idx,
                                col_idx: i as u32,
                                dtype: col.dtype,
                            });
                        }
                    }
                }
                _ => return Err(BinderError("unsupported select item".into())),
            }
        }
        Ok(out)
    }

    fn resolve_insert(&self, stmt: &Insert) -> Result<BoundStatement, BinderError> {
        let TableObject::TableName(name) = &stmt.table else {
            return Err(BinderError("unsupported INSERT target".into()));
        };
        let scope = self.build_scope(&ident_name(name)?)?;

        let source = stmt
            .source
            .as_ref()
            .ok_or_else(|| BinderError("INSERT without source".into()))?;
        let SetExpr::Values(values) = source.body.as_ref() else {
            return Err(BinderError("INSERT requires VALUES".into()));
        };

        let rows = values
            .rows
            .iter()
            .map(|row| {
                row.content
                    .iter()
                    .map(|e| self.resolve_expr(e, &scope))
                    .collect()
            })
            .collect::<Result<_, _>>()?;

        Ok(BoundStatement::Insert(BoundInsert { scope, rows }))
    }

    fn resolve_delete(&self, stmt: &Delete) -> Result<BoundStatement, BinderError> {
        let (FromTable::WithFromKeyword(tables) | FromTable::WithoutKeyword(tables)) = &stmt.from;
        let [t] = tables.as_slice() else {
            return Err(BinderError("DELETE expects one table".into()));
        };
        let TableFactor::Table { name, .. } = &t.relation else {
            return Err(BinderError("DELETE target must be a table".into()));
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
            return Err(BinderError("UPDATE target must be a table".into()));
        };
        let table_name = ident_name(name)?;
        let scope = self.build_scope(&table_name)?;
        let schema = &scope[&table_name].schema;

        let mut cols = Vec::with_capacity(stmt.assignments.len());
        let mut new_cols = Vec::with_capacity(stmt.assignments.len());
        for Assignment { target, value } in &stmt.assignments {
            let AssignmentTarget::ColumnName(col_name) = target else {
                return Err(BinderError("tuple UPDATE assignment not supported".into()));
            };
            let col = ident_name(col_name)?;
            let idx = schema
                .col_idx(&col)
                .ok_or_else(|| BinderError(format!("column not found: {col}")))?;
            cols.push(Expression::Column {
                tuple_idx: 0,
                col_idx: idx as u32,
                dtype: schema.cols[idx].dtype,
            });
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
            .ok_or_else(|| BinderError("CREATE INDEX without a name".into()))
            .and_then(ident_name)?;
        let table_name = ident_name(&stmt.table_name)?;
        let scope = self.build_scope(&table_name)?;
        let schema = &scope[&table_name].schema;

        let key_cols = stmt
            .columns
            .iter()
            .map(|c| {
                let Expr::Identifier(id) = &c.column.expr else {
                    return Err(BinderError(
                        "index columns must be plain identifiers".into(),
                    ));
                };
                let col_name = id.value.clone();
                let idx = schema
                    .col_idx(&col_name)
                    .ok_or_else(|| BinderError(format!("column not found: {col_name}")))?;
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
            _ => Err(BinderError("unsupported statement".into())),
        }
    }

    fn build_scope(&self, table_name: &str) -> Result<Scope, BinderError> {
        let oid = self
            .catalog
            .table_oid(table_name)
            .map_err(|_| BinderError(format!("table not found: {table_name}")))?;
        let tinfo = self
            .catalog
            .get_table(oid)
            .map_err(|_| BinderError(format!("table not found: {table_name}")))?;
        let mut scope = HashMap::new();
        scope.insert(
            table_name.to_string(),
            TableScope {
                oid,
                idx: 0,
                schema: tinfo.schema.clone(),
            },
        );
        Ok(scope)
    }
}

fn ident_name(name: &ObjectName) -> Result<String, BinderError> {
    match name.0.as_slice() {
        [ObjectNamePart::Identifier(id)] => Ok(id.value.clone()),
        _ => Err(BinderError(format!("qualified name not supported: {name}"))),
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
                return Err(BinderError("LIMIT must be a numeric literal".into()));
            };
            s.parse::<u32>()
                .map(Some)
                .map_err(|_| BinderError(format!("invalid LIMIT: {s}")))
        }
        _ => Err(BinderError("unsupported LIMIT clause".into())),
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

fn aggregate_outside_projection() -> BinderError {
    BinderError("aggregate functions are only allowed in a SELECT projection".into())
}

fn reject_aggregate_call(expr: &Expr, message: &str) -> Result<(), BinderError> {
    match expr {
        Expr::Function(f) if parse_agg_type(&f.name).is_some() => Err(BinderError(message.into())),
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
        other => Err(BinderError(format!("unsupported type: {other}"))),
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
    pub predicate: Expression,
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
    pub cols: Vec<Expression>,
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
    pub idx: TupleIndex,
    pub schema: Schema,
}
pub type TableAlias = String;
pub type TableOid = u32;
pub type TupleIndex = u8;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    use super::{Binder, BoundStatement};
    use crate::catalog::{Catalog, DataType, Schema};
    use crate::create_buffer_pool_manager;
    use crate::disk::{DiskManager, DiskScheduler};
    use crate::processor::plan::Expression;
    use crate::replacer::ArcReplacer;

    fn catalog_with_tables(tables: &[(&str, &[&str])]) -> (Catalog, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).expect("disk");
        let bpm = Arc::new(create_buffer_pool_manager(
            16,
            ArcReplacer::new(16),
            DiskScheduler::new(dm),
        ));
        let catalog = Catalog::new(bpm);
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

    fn bind_one(catalog: &Catalog, sql: &str) -> Result<BoundStatement, String> {
        let mut ast = Parser::parse_sql(&GenericDialect {}, sql).expect("valid SQL");
        assert_eq!(ast.len(), 1, "expected exactly one statement");
        Binder::new(catalog)
            .bind_statement(&ast.remove(0))
            .map_err(|e| e.0)
    }

    fn column_refs(exprs: &[Expression]) -> Vec<(u8, u32)> {
        exprs
            .iter()
            .map(|e| match e {
                Expression::Column {
                    tuple_idx, col_idx, ..
                } => (*tuple_idx, *col_idx),
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
            assert_eq!(column_refs(&sel.projections), vec![(0, 0), (0, 1), (1, 0)]);
        }
    }

    fn bind_error(catalog: &Catalog, sql: &str) -> String {
        match bind_one(catalog, sql) {
            Ok(_) => panic!("expected `{sql}` to fail to bind"),
            Err(e) => e,
        }
    }

    #[test]
    fn aggregate_in_where_is_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        let err = bind_error(&catalog, "SELECT a FROM t WHERE COUNT(a) > 1");
        assert!(err.contains("only allowed in a SELECT projection"), "{err}");
    }

    #[test]
    fn nested_aggregate_is_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        let err = bind_error(&catalog, "SELECT SUM(COUNT(a)) FROM t");
        assert!(err.contains("nested aggregate"), "{err}");
    }

    #[test]
    fn ungrouped_column_in_aggregating_select_is_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a", "b"])]);
        let err = bind_error(&catalog, "SELECT a, b FROM t GROUP BY a");
        assert!(err.contains("column `b` must appear in GROUP BY"), "{err}");
        let err = bind_error(&catalog, "SELECT t.b, COUNT(a) FROM t");
        assert!(
            err.contains("column `t.b` must appear in GROUP BY"),
            "{err}"
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
        assert_eq!(column_refs(&sel.projections), vec![(0, 0), (0, 1), (0, 2)]);
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
        let err = bind_error(&catalog, "SELECT SUM(*) FROM t");
        assert!(err.contains("only valid as an argument to COUNT"), "{err}");
    }

    #[test]
    fn bare_null_projection_is_rejected_but_null_in_arithmetic_is_typed_by_its_sibling() {
        let (catalog, _dir) = catalog_with_tables(&[("t", &["a"])]);
        let err = bind_error(&catalog, "SELECT NULL FROM t");
        assert!(err.contains("bare NULL"), "{err}");
        let Ok(BoundStatement::Select(sel)) = bind_one(&catalog, "SELECT a + NULL FROM t") else {
            panic!("expected a bound SELECT");
        };
        assert_eq!(sel.projections[0].dtype(), Some(DataType::INT));
    }

    #[test]
    fn joins_of_more_than_two_tables_are_rejected() {
        let (catalog, _dir) = catalog_with_tables(&[("a", &["x"]), ("b", &["y"]), ("c", &["z"])]);
        let err = bind_error(
            &catalog,
            "SELECT * FROM a JOIN b ON a.x = b.y JOIN c ON b.y = c.z",
        );
        assert!(err.contains("only 2-way joins"), "{err}");
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
        assert!(err.0.contains("table not found: t"), "{}", err.0);
    }
}
