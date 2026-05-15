use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use ordered_float::OrderedFloat;
use sqlparser::ast::{
    Assignment, AssignmentTarget, BinaryOperator, CreateIndex as SqlCreateIndex,
    CreateTable as SqlCreateTable, DataType as SqlDataType, Delete, Expr, Function,
    FunctionArg, FunctionArgExpr, FunctionArguments, FromTable, GroupByExpr, Insert,
    JoinConstraint, JoinOperator, LimitClause, ObjectName, ObjectNamePart, OrderByKind, Query,
    SelectItem, SetExpr, Statement, TableFactor, TableObject, TableWithJoins, Update,
    Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::catalog::{Catalog, DataType, Schema, Value};
use crate::processor::plan::{
    AggType, CreateIndexPlanNode, CreateTablePlanNode, DeletePlanNode, Expression, FilterPlanNode,
    InsertPlanNode, JoinType, LimitPlanNode, NestedLoopJoinPlanNode, Op, Plan, ProjectionPlanNode,
    SeqScanPlanNode, SortPlanNode, UpdatePlanNode, ValuesPlanNode,
};

/// a good read: https://craftinginterpreters.com/resolving-and-binding.html
/// most of this was written by AI since this is not the part of dbms i was interested in
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
        // When we're inside an aggregating SELECT (but not currently descending
        // into an aggregate's argument), rewrite:
        //   - any sub-expression that structurally matches a GROUP BY expr to
        //     a Column ref into the aggregation output (col_idx = group-by position).
        //   - any aggregate function call to a Column ref into the aggregation
        //     output (col_idx = group_bys.len() + aggregate position).
        // Bare column refs that don't match a group-by are an error.
        if !self.inside_agg_arg.get() {
            // first, group-by rewrite
            {
                let ctx = self.agg_ctx.borrow();
                if let Some(ctx) = ctx.as_ref() {
                    if let Some(idx) = ctx.group_by_exprs.iter().position(|gb| gb == expr) {
                        let dtype = ctx.group_bys[idx].dtype().ok_or_else(|| {
                            BinderError("GROUP BY expression has no inferable type".into())
                        })?;
                        return Ok(Expression::Column {
                            tuple_idx: 0,
                            col_idx: idx as u32,
                            dtype,
                        });
                    }
                }
            }

            // aggregate function rewrite
            if let Expr::Function(f) = expr {
                if let Some(agg_type) = parse_agg_type(&f.name) {
                    if self.agg_ctx.borrow().is_none() {
                        return Err(BinderError(
                            "aggregate functions are only allowed in a SELECT projection".into(),
                        ));
                    }
                    return self.register_aggregate(f, scope, agg_type);
                }
            }

            // bare column refs without a matching group-by are illegal in
            // aggregating queries.
            if self.agg_ctx.borrow().is_some() {
                match expr {
                    Expr::Identifier(id) => {
                        return Err(BinderError(format!(
                            "column `{}` must appear in GROUP BY or be used inside an aggregate function",
                            id.value
                        )));
                    }
                    Expr::CompoundIdentifier(parts) => {
                        let path = parts
                            .iter()
                            .map(|p| p.value.clone())
                            .collect::<Vec<_>>()
                            .join(".");
                        return Err(BinderError(format!(
                            "column `{path}` must appear in GROUP BY or be used inside an aggregate function"
                        )));
                    }
                    _ => {}
                }
            }
        } else {
            // inside an aggregate's argument: nested aggregates are forbidden.
            if let Expr::Function(f) = expr {
                if parse_agg_type(&f.name).is_some() {
                    return Err(BinderError(
                        "nested aggregate functions are not allowed".into(),
                    ));
                }
            }
        }

        match expr {
            Expr::Nested(inner) => self.resolve_expr(inner, scope),

            // even though sqlparser's Expr::Identifier can represent both
            // table and column names, we don't care about table names because
            // if we call resolve_expr() we will always have a column to resolve,
            // never a table
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

    /// Registers an aggregate function call discovered inside a projection /
    /// ORDER BY expression. Returns a Column reference into the aggregation
    /// node's output that the executor will eventually fill in.
    fn register_aggregate(
        &self,
        f: &Function,
        scope: &Scope,
        agg_type: AggType,
    ) -> Result<Expression, BinderError> {
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

        // The argument expression is evaluated against the *child* tuples
        // (the pre-aggregation rows), so flip off the agg-rewrite mode
        // while we resolve it.
        let arg_expr = match arg {
            FunctionArgExpr::Wildcard => {
                if !matches!(agg_type, AggType::COUNT) {
                    return Err(BinderError(format!(
                        "`*` is only valid as an argument to COUNT, not {}",
                        f.name
                    )));
                }
                // COUNT(*) just counts rows; the actual value is unused by the
                // executor, but we still need an expression that evaluates
                // without errors. A constant works.
                Expression::Constant(Some(Value::INT(1)))
            }
            FunctionArgExpr::Expr(e) => {
                self.inside_agg_arg.set(true);
                let result = self.resolve_expr(e, scope);
                self.inside_agg_arg.set(false);
                result?
            }
            FunctionArgExpr::QualifiedWildcard(_) | FunctionArgExpr::WildcardWithOptions(_) => {
                return Err(BinderError(
                    "qualified wildcards in aggregates are not supported".into(),
                ));
            }
        };

        // Result type: COUNT is always INT; everything else inherits the
        // argument's type (matching how AggState handles them).
        let result_dtype = match agg_type {
            AggType::COUNT => DataType::INT,
            _ => arg_expr.dtype().ok_or_else(|| {
                BinderError(format!(
                    "argument to aggregate `{}` has no inferable type",
                    f.name
                ))
            })?,
        };

        let mut ctx_mut = self.agg_ctx.borrow_mut();
        let ctx = ctx_mut.as_mut().expect("agg_ctx active");
        let col_idx = ctx.group_bys.len() + ctx.aggregates.len();
        ctx.aggregates.push((arg_expr, agg_type));
        Ok(Expression::Column {
            tuple_idx: 0,
            col_idx: col_idx as u32,
            dtype: result_dtype,
        })
    }

    fn resolve_select(&self, q: &Box<Query>) -> Result<BoundStatement, BinderError> {
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

                // WHERE is resolved *before* the aggregation context is set
                // up, because aggregates cannot appear in WHERE (only HAVING).
                let filters = s
                    .selection
                    .as_ref()
                    .map(|e| self.resolve_expr(e, &scope).map(|e| vec![e]))
                    .transpose()?;

                // GROUP BY exprs are resolved against the child scope, *not*
                // through the agg-ctx (they define the rewrite map).
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

                // Activate the aggregation context if anything would produce
                // an Aggregation plan node: either there are GROUP BY exprs
                // or one of the projections contains an aggregate function.
                let aggregating = !group_by_exprs_raw.is_empty()
                    || projection_has_aggregate(&s.projection);
                let prev_ctx = if aggregating {
                    self.agg_ctx.borrow_mut().replace(AggCtx {
                        group_by_exprs: group_by_exprs_raw,
                        group_bys: group_bys.clone(),
                        aggregates: vec![],
                    })
                } else {
                    self.agg_ctx.borrow_mut().take()
                };

                let projections = self.resolve_projections(&s.projection, &scope)?;
                let sort = match q.order_by.as_ref() {
                    None => None,
                    Some(ob) => match &ob.kind {
                        OrderByKind::Expressions(exprs) => Some(
                            exprs
                                .iter()
                                .map(|oe| self.resolve_expr(&oe.expr, &scope))
                                .collect::<Result<_, _>>()?,
                        ),
                        OrderByKind::All(_) => {
                            return Err(BinderError("ORDER BY ALL not supported".into()));
                        }
                    },
                };
                let limit = resolve_limit(q.limit_clause.as_ref())?;

                // Pull out the aggregates registered while resolving the
                // projection / ORDER BY, then restore any outer agg-ctx.
                let (group_bys, aggregates) = if aggregating {
                    let ctx = self
                        .agg_ctx
                        .borrow_mut()
                        .take()
                        .expect("agg_ctx was just installed");
                    *self.agg_ctx.borrow_mut() = prev_ctx;
                    (ctx.group_bys, ctx.aggregates)
                } else {
                    *self.agg_ctx.borrow_mut() = prev_ctx;
                    (vec![], vec![])
                };

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
                    // a bare `NULL` literal in a projection has no inferable
                    // type — reject it. NULLs nested inside arithmetic or
                    // comparisons are fine because the sibling supplies a type.
                    if resolved.dtype().is_none() {
                        return Err(BinderError(
                            "bare NULL not allowed in SELECT projection".into(),
                        ));
                    }
                    out.push(resolved);
                }
                SelectItem::Wildcard(_) => {
                    for ts in scope.values() {
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

    pub fn bind(&self, stmts: Vec<Statement>) -> Result<Vec<BoundStatement>, BinderError> {
        stmts.iter().map(|s| self.resolve(s)).collect()
    }

    fn resolve(&self, stmt: &Statement) -> Result<BoundStatement, BinderError> {
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

/// Maps a function name to an aggregate kind, case-insensitively.
/// Returns `None` for any non-aggregate function call (which the binder then
/// rejects as unsupported — we have no scalar functions yet).
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

fn expr_has_aggregate(e: &Expr) -> bool {
    match e {
        Expr::Function(f) => parse_agg_type(&f.name).is_some(),
        Expr::Nested(inner) => expr_has_aggregate(inner),
        Expr::BinaryOp { left, right, .. } => {
            expr_has_aggregate(left) || expr_has_aggregate(right)
        }
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

// select statements includes:
// the tables included (scope),
// an optional join (we support at most one — 2-way joins only),
// whatever we want to retrieve (projections)
// any filters
// anything we want to sort by
// a limit if there is one
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
    pub sort: Option<Vec<Expression>>,
    pub limit: Option<u32>, // no offset
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

// keeps track of the tables in scope and the aliases that the AST uses
pub type Scope = HashMap<TableAlias, TableScope>;
pub struct TableScope {
    pub oid: TableOid,
    pub idx: TupleIndex,
    pub schema: Schema,
}
pub type TableAlias = String;
pub type TableOid = u32;
pub type TupleIndex = u8;

fn test() {
    let sql = "SELECT a, b, 123, myfunc(b) \
           FROM table_1 t1 \
           LEFT JOIN table_2 t2
           ON t1.a = t2.a
           WHERE a > b AND b < 100 \
           ORDER BY a DESC, b";

    let dialect = GenericDialect {}; // or AnsiDialect, or your own dialect ...

    let ast = Parser::parse_sql(&dialect, sql).unwrap();

    println!("AST: {:?}", ast);
    println!("AST len: {:?}", ast.len());
}

#[test]
fn test_ast() {
    let a = test();
}
