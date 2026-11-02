use std::fmt;
use std::ops::Bound;

use crate::catalog::Value;
use crate::catalog::{Catalog, DataType, Schema};
use crate::parser::{
    BoundCreateIndex, BoundCreateTable, BoundDelete, BoundDropIndex, BoundDropTable, BoundInsert,
    BoundJoin, BoundSelect, BoundStatement, BoundUpdate, BoundValues, BoundWindow, Scope,
    TableScope,
};
use crate::processor::plan::{
    AggType, AggregationPlanNode, ColIdx, ColumnAssignment, CreateIndexPlanNode,
    CreateTablePlanNode, DeletePlanNode, DropIndexPlanNode, DropTablePlanNode, Expression,
    ExternalMergeSortPlanNode, FilterPlanNode, HashJoinPlanNode, IndexScanPlanNode, InsertPlanNode,
    JoinColumn, JoinPredicate, JoinType, LimitPlanNode, NestedIndexJoinPlanNode,
    NestedLoopJoinPlanNode, Op, Plan, ProjectionPlanNode, SeqScanPlanNode, Side, SortExpr,
    UpdatePlanNode, ValuesPlanNode, WindowFunctionPlanNode,
};

/// Turns bound statements into plans, without statistics or a cost model.
///
/// A `col = constant` filter on a table with a single-column index on `col` becomes an
/// `IndexScan`; an inner `left.col = right.col` join whose right table has a single-column
/// index on its join column becomes a `NestedIndexJoin`. Any other join whose condition ANDs
/// at least one `left.col = right.col` of matching types becomes a `HashJoin` on those
/// equalities, with the rest of the condition as its residual. Everything else is a `SeqScan`
/// or a `NestedLoopJoin`. `ORDER BY` is an `ExternalMergeSort`, which sorts in memory until
/// its input passes the executor's sort memory budget and spills sorted runs after that.
/// `SELECT DISTINCT` is an `Aggregation` grouping on every projected column, placed above the
/// projection and below the sort.
/// Window functions run in one `WindowFunction` node per distinct OVER clause, stacked over the
/// filtered FROM output, each passing its input's columns through and appending its calls.
pub struct Planner<'a> {
    catalog: &'a Catalog,
}

/// Why a bound statement could not be planned. Each variant is a bound statement the binder
/// does not produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerError {
    /// A statement that reads one table has `tables` tables in scope.
    NotOneTable {
        statement: &'static str,
        tables: usize,
    },
    /// A join's scope does not hold exactly one left and one right table.
    JoinSides { tables: usize },
    /// A CREATE INDEX key column is not a column of the indexed table.
    UnknownIndexColumn { column: String },
    /// The planned FROM output has `planned` columns, but the binder placed the SELECT's window
    /// call results after `bound` FROM columns.
    WindowInputWidth { bound: usize, planned: usize },
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlannerError::NotOneTable { statement, tables } => {
                write!(f, "{statement} needs exactly one table, got {tables}")
            }
            PlannerError::JoinSides { tables } => write!(
                f,
                "a join needs exactly one left and one right table, got {tables} tables"
            ),
            PlannerError::UnknownIndexColumn { column } => {
                write!(f, "CREATE INDEX references unknown column: {column}")
            }
            PlannerError::WindowInputWidth { bound, planned } => write!(
                f,
                "window calls were bound after {bound} FROM columns, but the planned FROM output has {planned}"
            ),
        }
    }
}

impl std::error::Error for PlannerError {}

impl<'a> Planner<'a> {
    pub fn new(catalog: &'a Catalog) -> Self {
        Self { catalog }
    }

    pub fn plan(&self, stmt: BoundStatement) -> Result<Plan, PlannerError> {
        match stmt {
            BoundStatement::Select(s) => self.plan_select(s),
            BoundStatement::Values(v) => Ok(self.plan_values(v)),
            BoundStatement::Insert(i) => self.plan_insert(i),
            BoundStatement::Delete(d) => self.plan_delete(d),
            BoundStatement::Update(u) => self.plan_update(u),
            BoundStatement::CreateTable(c) => Ok(self.plan_create_table(c)),
            BoundStatement::CreateIndex(c) => self.plan_create_index(c),
            BoundStatement::DropTable(d) => Ok(plan_drop_table(d)),
            BoundStatement::DropIndex(d) => Ok(plan_drop_index(d)),
        }
    }

    fn plan_select(&self, sel: BoundSelect) -> Result<Plan, PlannerError> {
        let BoundSelect {
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
        } = sel;
        let filter_expr = conjunction(filters);

        let (mut plan, mut source_schema, remaining_filter) = match join {
            Some(j) => {
                let (left, right) = join_sides(&scope)?;

                let cols: Vec<(DataType, String)> = left
                    .schema
                    .cols
                    .iter()
                    .chain(right.schema.cols.iter())
                    .map(|c| (c.dtype, c.name.clone()))
                    .collect();
                let s = Schema::new(cols);
                let plan = self.build_join(left, right, j, s.clone());
                (plan, s, filter_expr)
            }
            None if scope.is_empty() => {
                let schema = Schema::new(vec![]);
                let plan = Plan::Values(ValuesPlanNode {
                    schema: schema.clone(),
                    rows: vec![vec![]],
                });
                (plan, schema, filter_expr)
            }
            None => {
                let ts = only_table(&scope, "SELECT")?;
                let (plan, rem) = self.build_scan(ts, filter_expr);
                (plan, ts.schema.clone(), rem)
            }
        };

        if let Some(pred) = remaining_filter {
            plan = Plan::Filter(FilterPlanNode {
                schema: source_schema.clone(),
                predicate: pred,
                child: Box::new(plan),
            });
        }

        if !windows.is_empty() {
            check_window_input_width(&scope, &source_schema)?;
        }
        let mut window_calls = 0;
        for window in windows {
            let window_schema = window_output_schema(&window, &source_schema, window_calls);
            window_calls += window.calls.len();
            plan = window_plan(window, plan, &source_schema, window_schema.clone());
            source_schema = window_schema;
        }

        if !group_bys.is_empty() || !aggregates.is_empty() {
            let agg_schema = aggregation_output_schema(&group_bys, &aggregates, &source_schema);
            plan = Plan::Aggregation(AggregationPlanNode {
                schema: agg_schema.clone(),
                child: Box::new(plan),
                group_bys,
                aggregates,
            });
            source_schema = agg_schema;
        }

        let proj_schema = projection_schema(&projections, &source_schema);
        if distinct {
            plan = project(plan, projections, &proj_schema);
            plan = distinct_rows(plan, &proj_schema);
            plan = sort_rows(plan, &proj_schema, sort);
        } else {
            plan = sort_rows(plan, &source_schema, sort);
            plan = project(plan, projections, &proj_schema);
        }

        if let Some(n) = limit {
            plan = Plan::Limit(LimitPlanNode {
                schema: proj_schema,
                child: Box::new(plan),
                limit: n,
            });
        }

        Ok(plan)
    }

    /// Each VALUES column takes the first type its rows give it, widened to the common numeric
    /// type of every row, so `VALUES (1), (2.5)` is a FLOAT column.
    fn plan_values(&self, v: BoundValues) -> Plan {
        let width = v.rows.first().map_or(0, Vec::len);
        let cols: Vec<(DataType, String)> = (0..width)
            .map(|i| {
                let dtype = v
                    .rows
                    .iter()
                    .filter_map(|row| row.get(i).and_then(Expression::dtype))
                    .reduce(|acc, t| acc.common_numeric(t).unwrap_or(acc))
                    .unwrap_or(DataType::INT);
                (dtype, format!("col{i}"))
            })
            .collect();
        let schema = Schema::new(cols);
        Plan::Values(ValuesPlanNode {
            schema,
            rows: v.rows,
        })
    }

    fn plan_insert(&self, ins: BoundInsert) -> Result<Plan, PlannerError> {
        let ts = only_table(&ins.scope, "INSERT")?;
        let table_schema = ts.schema.clone();
        let values = Plan::Values(ValuesPlanNode {
            schema: table_schema.clone(),
            rows: ins.rows,
        });
        Ok(Plan::Insert(InsertPlanNode {
            schema: table_schema,
            child: Box::new(values),
            table_oid: ts.oid,
        }))
    }

    fn plan_delete(&self, del: BoundDelete) -> Result<Plan, PlannerError> {
        let ts = only_table(&del.scope, "DELETE")?;
        let filter = conjunction(del.filters);
        let (mut scan, remaining) = self.build_scan(ts, filter);
        if let Some(pred) = remaining {
            scan = Plan::Filter(FilterPlanNode {
                schema: ts.schema.clone(),
                predicate: pred,
                child: Box::new(scan),
            });
        }
        Ok(Plan::Delete(DeletePlanNode {
            schema: ts.schema.clone(),
            child: Box::new(scan),
            table_oid: ts.oid,
        }))
    }

    fn plan_update(&self, upd: BoundUpdate) -> Result<Plan, PlannerError> {
        let ts = only_table(&upd.scope, "UPDATE")?;
        let filter = conjunction(upd.filters);
        let (mut scan, remaining) = self.build_scan(ts, filter);
        if let Some(pred) = remaining {
            scan = Plan::Filter(FilterPlanNode {
                schema: ts.schema.clone(),
                predicate: pred,
                child: Box::new(scan),
            });
        }

        let mut assignments = Vec::with_capacity(upd.cols.len());
        for (col, value) in upd.cols.into_iter().zip(upd.new_cols) {
            assignments.push(ColumnAssignment {
                col_idx: col,
                value,
            });
        }

        Ok(Plan::Update(UpdatePlanNode {
            schema: ts.schema.clone(),
            assignments,
            child: Box::new(scan),
            table_oid: ts.oid,
        }))
    }

    fn plan_create_table(&self, c: BoundCreateTable) -> Plan {
        Plan::CreateTable(CreateTablePlanNode {
            table_name: c.table_name,
            schema: c.schema,
            if_not_exists: false,
        })
    }

    fn plan_create_index(&self, c: BoundCreateIndex) -> Result<Plan, PlannerError> {
        let ts = only_table(&c.scope, "CREATE INDEX")?;
        let key_columns = c
            .index_schema
            .cols
            .iter()
            .map(|col| {
                ts.schema
                    .col_idx(&col.name)
                    .map(|i| ColIdx(i as u32))
                    .ok_or_else(|| PlannerError::UnknownIndexColumn {
                        column: col.name.clone(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Plan::CreateIndex(CreateIndexPlanNode {
            index_name: c.name,
            table_oid: ts.oid,
            key_columns,
        }))
    }

    fn build_scan(
        &self,
        ts: &TableScope,
        filter: Option<Expression>,
    ) -> (Plan, Option<Expression>) {
        if let Some((col_idx, value)) = filter.as_ref().and_then(column_equals_constant)
            && value.data_type() == ts.schema.cols[col_idx.index()].dtype
            && let Some(index_oid) = self.index_for_cols(ts.oid, &[col_idx.index()])
        {
            return (
                Plan::IndexScan(IndexScanPlanNode {
                    schema: ts.schema.clone(),
                    start: Bound::Included(vec![value]),
                    end: Bound::Included(vec![value]),
                    index_oid,
                }),
                None,
            );
        }
        let plan = Plan::SeqScan(SeqScanPlanNode {
            table_oid: ts.oid,
            schema: ts.schema.clone(),
        });
        (plan, filter)
    }

    fn build_join(
        &self,
        left_ts: &TableScope,
        right_ts: &TableScope,
        join: BoundJoin,
        out_schema: Schema,
    ) -> Plan {
        let left = Plan::SeqScan(SeqScanPlanNode {
            table_oid: left_ts.oid,
            schema: left_ts.schema.clone(),
        });
        let right = Plan::SeqScan(SeqScanPlanNode {
            table_oid: right_ts.oid,
            schema: right_ts.schema.clone(),
        });

        let index_join = match join.join_type {
            JoinType::Inner => left_right_equi_join(&join.predicate)
                .filter(|&(outer, inner)| {
                    left_ts.schema.cols[outer as usize].dtype
                        == right_ts.schema.cols[inner as usize].dtype
                })
                .and_then(|(outer, inner)| {
                    self.index_for_cols(right_ts.oid, &[inner as usize])
                        .map(|index_oid| (outer, index_oid))
                }),
            JoinType::Left => None,
        };
        if let Some((outer_col, index_oid)) = index_join {
            return Plan::NestedIndexJoin(NestedIndexJoinPlanNode {
                schema: out_schema,
                child: Box::new(left),
                key_predicate: Expression::Column {
                    col: ColIdx(outer_col),
                    dtype: left_ts.schema.cols[outer_col as usize].dtype,
                },
                table_oid: right_ts.oid,
                table_schema: right_ts.schema.clone(),
                index_oid,
                join_type: JoinType::Inner,
            });
        }

        match hash_join_keys(join.predicate, &left_ts.schema, &right_ts.schema) {
            Ok(keys) => Plan::HashJoin(HashJoinPlanNode {
                schema: out_schema,
                left: Box::new(left),
                right: Box::new(right),
                left_exprs: keys.left_exprs,
                right_exprs: keys.right_exprs,
                residual: keys.residual,
                join_type: join.join_type,
            }),
            Err(predicate) => Plan::NestedLoopJoin(NestedLoopJoinPlanNode {
                schema: out_schema,
                left: Box::new(left),
                right: Box::new(right),
                predicate,
                join_type: join.join_type,
            }),
        }
    }

    fn index_for_cols(&self, table_oid: u32, col_idxs: &[usize]) -> Option<u32> {
        self.catalog
            .get_table_indexes(table_oid)
            .ok()?
            .into_iter()
            .find(|info| info.index.indexed_col_idxs == col_idxs)
            .map(|info| info.oid)
    }
}

fn plan_drop_table(d: BoundDropTable) -> Plan {
    Plan::DropTable(DropTablePlanNode {
        table_oid: d.table_oid,
        if_exists: d.if_exists,
    })
}

fn plan_drop_index(d: BoundDropIndex) -> Plan {
    Plan::DropIndex(DropIndexPlanNode {
        index_oid: d.index_oid,
        if_exists: d.if_exists,
    })
}

fn agg_type_name(t: &AggType) -> &'static str {
    match t {
        AggType::COUNT => "count",
        AggType::SUM => "sum",
        AggType::AVG => "avg",
        AggType::MIN => "min",
        AggType::MAX => "max",
        AggType::RANK => "rank",
    }
}

/// ANDs the filters together; `None` when there are none.
fn conjunction<C>(filters: Option<Vec<Expression<C>>>) -> Option<Expression<C>> {
    filters?
        .into_iter()
        .reduce(|left, right| Expression::Binary {
            left: Box::new(left),
            right: Box::new(right),
            op: Op::And,
        })
}

/// The operands of every top-level AND in `expr`, left to right.
fn conjuncts<C>(expr: &Expression<C>) -> Vec<&Expression<C>> {
    match expr {
        Expression::Binary {
            left,
            right,
            op: Op::And,
        } => {
            let mut parts = conjuncts(left);
            parts.extend(conjuncts(right));
            parts
        }
        other => vec![other],
    }
}

/// The equality keys and residual condition of a hash join.
#[derive(Debug)]
struct HashJoinKeys {
    left_exprs: Vec<Expression>,
    right_exprs: Vec<Expression>,
    residual: Option<JoinPredicate>,
}

/// Splits a join condition into hash join keys: each ANDed `left.col = right.col` whose two
/// columns have the same type becomes a key pair, and the other conjuncts become the residual.
/// Returns the condition unchanged when it has no such equality.
///
/// Keys of different types stay in the residual because the hash table compares keys by
/// value and type, while SQL equality compares INT with FLOAT numerically.
fn hash_join_keys(
    predicate: JoinPredicate,
    left: &Schema,
    right: &Schema,
) -> Result<HashJoinKeys, JoinPredicate> {
    let key_columns = |part: &JoinPredicate| {
        left_right_equi_join(part)
            .filter(|&(l, r)| left.cols[l as usize].dtype == right.cols[r as usize].dtype)
    };
    let parts = conjuncts(&predicate);
    if !parts.iter().any(|part| key_columns(part).is_some()) {
        return Err(predicate);
    }
    let column = |schema: &Schema, col: u32| Expression::Column {
        col: ColIdx(col),
        dtype: schema.cols[col as usize].dtype,
    };
    let mut left_exprs = Vec::new();
    let mut right_exprs = Vec::new();
    let mut residual = Vec::new();
    for part in parts {
        match key_columns(part) {
            Some((l, r)) => {
                left_exprs.push(column(left, l));
                right_exprs.push(column(right, r));
            }
            None => residual.push(part.clone()),
        }
    }
    Ok(HashJoinKeys {
        left_exprs,
        right_exprs,
        residual: conjunction(Some(residual)),
    })
}

/// The left and right tables of a two-way join.
fn join_sides(scope: &Scope) -> Result<(&TableScope, &TableScope), PlannerError> {
    let on =
        |side: Side| -> Vec<&TableScope> { scope.values().filter(|ts| ts.side == side).collect() };
    match (on(Side::Left).as_slice(), on(Side::Right).as_slice()) {
        ([left], [right]) => Ok((left, right)),
        _ => Err(PlannerError::JoinSides {
            tables: scope.len(),
        }),
    }
}

fn only_table<'s>(
    scope: &'s Scope,
    statement: &'static str,
) -> Result<&'s TableScope, PlannerError> {
    let mut tables = scope.values();
    match (tables.next(), tables.next()) {
        (Some(ts), None) => Ok(ts),
        _ => Err(PlannerError::NotOneTable {
            statement,
            tables: scope.len(),
        }),
    }
}

/// Sorts `plan`'s rows, laid out as `schema`, by `sort`; no sort without keys.
fn sort_rows(plan: Plan, schema: &Schema, sort: Option<Vec<SortExpr>>) -> Plan {
    match sort {
        Some(order_by_exprs) => Plan::ExternalMergeSort(ExternalMergeSortPlanNode {
            schema: schema.clone(),
            child: Box::new(plan),
            order_by_exprs,
        }),
        None => plan,
    }
}

/// The output layout of `projections` evaluated over rows laid out as `source_schema`.
fn projection_schema(projections: &[Expression], source_schema: &Schema) -> Schema {
    Schema::new(
        projections
            .iter()
            .enumerate()
            .map(|(i, e)| {
                (
                    e.dtype().unwrap_or(DataType::INT),
                    output_column_name(e, source_schema, || format!("col{i}")),
                )
            })
            .collect(),
    )
}

fn project(plan: Plan, expressions: Vec<Expression>, schema: &Schema) -> Plan {
    Plan::Projection(ProjectionPlanNode {
        schema: schema.clone(),
        expressions,
        child: Box::new(plan),
    })
}

/// Removes duplicate rows from `plan`, laid out as `schema`, by grouping on every column with
/// no aggregates. Grouping treats NULLs as equal, so duplicate rows holding NULL collapse too.
fn distinct_rows(plan: Plan, schema: &Schema) -> Plan {
    let group_bys = schema
        .cols
        .iter()
        .enumerate()
        .map(|(i, c)| Expression::Column {
            col: ColIdx(i as u32),
            dtype: c.dtype,
        })
        .collect();
    Plan::Aggregation(AggregationPlanNode {
        schema: schema.clone(),
        child: Box::new(plan),
        group_bys,
        aggregates: vec![],
    })
}

/// The name of the source column a plain column reference reads, else `fallback()`.
fn output_column_name(
    expr: &Expression,
    source_schema: &Schema,
    fallback: impl FnOnce() -> String,
) -> String {
    let Expression::Column { col, .. } = expr else {
        return fallback();
    };
    source_schema
        .cols
        .get(col.index())
        .map(|c| c.name.clone())
        .unwrap_or_else(fallback)
}

/// Checks that `input`, the planned FROM output, is as wide as the FROM output the binder placed
/// window call results after: every column of every table in `scope`.
fn check_window_input_width(scope: &Scope, input: &Schema) -> Result<(), PlannerError> {
    let bound = scope.values().map(|ts| ts.schema.cols.len()).sum();
    let planned = input.cols.len();
    if bound == planned {
        Ok(())
    } else {
        Err(PlannerError::WindowInputWidth { bound, planned })
    }
}

/// The input's columns followed by one column per call of `window`, named after the call's
/// function and its position `first_call..` among the SELECT's window calls.
fn window_output_schema(window: &BoundWindow, input: &Schema, first_call: usize) -> Schema {
    let input_cols = input.cols.iter().map(|c| (c.dtype, c.name.clone()));
    let call_cols = window.calls.iter().enumerate().map(|(i, call)| {
        (
            call.dtype,
            format!("{}_{}", agg_type_name(&call.agg_type), first_call + i),
        )
    });
    Schema::new(input_cols.chain(call_cols).collect())
}

/// A `WindowFunction` node over `child` that passes every column of `input` through and
/// appends `window`'s calls.
fn window_plan(window: BoundWindow, child: Plan, input: &Schema, schema: Schema) -> Plan {
    let columns = input
        .cols
        .iter()
        .enumerate()
        .map(|(i, c)| Expression::Column {
            col: ColIdx(i as u32),
            dtype: c.dtype,
        })
        .collect();
    Plan::WindowFunction(WindowFunctionPlanNode {
        schema,
        child: Box::new(child),
        order_by_exprs: (!window.order_by.is_empty()).then_some(window.order_by),
        partition_by_exprs: (!window.partition_by.is_empty()).then_some(window.partition_by),
        aggregates: window
            .calls
            .into_iter()
            .map(|call| (call.arg, call.agg_type))
            .collect(),
        columns,
    })
}

/// One column per group-by expression followed by one column per aggregate, the row
/// layout `AggregationExecutor` emits.
fn aggregation_output_schema(
    group_bys: &[Expression],
    aggregates: &[(Expression, AggType)],
    source_schema: &Schema,
) -> Schema {
    let group_cols = group_bys.iter().enumerate().map(|(i, gb)| {
        (
            gb.dtype().unwrap_or(DataType::INT),
            output_column_name(gb, source_schema, || format!("g{i}")),
        )
    });
    let agg_cols = aggregates.iter().enumerate().map(|(i, (expr, agg_type))| {
        let dtype = expr
            .dtype()
            .and_then(|arg| agg_type.result_type(arg))
            .unwrap_or(DataType::INT);
        (dtype, format!("{}_{}", agg_type_name(agg_type), i))
    });
    Schema::new(group_cols.chain(agg_cols).collect())
}

/// `(col_idx, value)` when `filter` is `col = value` or `value = col` on the scanned table.
/// An index lookup encodes `value` with its own type, so callers use the index only when
/// `value` has the column's type. The binder converts a numeric constant to the column's type
/// when it converts exactly, so `f = 1` on a FLOAT column has a FLOAT constant here.
fn column_equals_constant(filter: &Expression) -> Option<(ColIdx, Value)> {
    let Expression::Binary {
        left,
        right,
        op: Op::Eq,
    } = filter
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (Expression::Column { col, .. }, Expression::Constant(Some(v)))
        | (Expression::Constant(Some(v)), Expression::Column { col, .. }) => Some((*col, *v)),
        _ => None,
    }
}

/// `(left_col, right_col)` when `predicate` is an equality between a left-input column
/// and a right-input column, in either order.
fn left_right_equi_join(predicate: &JoinPredicate) -> Option<(u32, u32)> {
    let Expression::Binary {
        left,
        right,
        op: Op::Eq,
    } = predicate
    else {
        return None;
    };
    let side_col = |e: &JoinPredicate| match e {
        Expression::Column { col, .. } => Some(*col),
        _ => None,
    };
    match (side_col(left)?, side_col(right)?) {
        (
            JoinColumn {
                side: Side::Left,
                col: outer,
            },
            JoinColumn {
                side: Side::Right,
                col: inner,
            },
        )
        | (
            JoinColumn {
                side: Side::Right,
                col: inner,
            },
            JoinColumn {
                side: Side::Left,
                col: outer,
            },
        ) => Some((outer.0, inner.0)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(col: u32) -> Expression {
        Expression::Column {
            col: ColIdx(col),
            dtype: DataType::INT,
        }
    }

    fn join_column(side: Side, col: u32) -> JoinPredicate {
        Expression::Column {
            col: JoinColumn {
                side,
                col: ColIdx(col),
            },
            dtype: DataType::INT,
        }
    }

    #[test]
    fn aggregation_output_lists_group_bys_before_aggregates() {
        let source = Schema::new(vec![
            (DataType::INT, "dept".to_string()),
            (DataType::FLOAT, "bonus".to_string()),
        ]);
        let bonus = Expression::Column {
            col: ColIdx(1),
            dtype: DataType::FLOAT,
        };
        let schema = aggregation_output_schema(
            &[column(0)],
            &[(bonus.clone(), AggType::SUM), (bonus, AggType::COUNT)],
            &source,
        );
        let cols: Vec<(DataType, &str)> = schema
            .cols
            .iter()
            .map(|c| (c.dtype, c.name.as_str()))
            .collect();
        assert_eq!(
            cols,
            vec![
                (DataType::INT, "dept"),
                (DataType::FLOAT, "sum_0"),
                (DataType::INT, "count_1"),
            ]
        );
    }

    #[test]
    fn output_column_name_reads_the_source_column_at_the_flat_index() {
        let joined = Schema::new(vec![
            (DataType::INT, "l".to_string()),
            (DataType::INT, "r".to_string()),
        ]);
        let fallback = || "fallback".to_string();
        assert_eq!(output_column_name(&column(0), &joined, fallback), "l");
        assert_eq!(output_column_name(&column(1), &joined, fallback), "r");
        assert_eq!(
            output_column_name(&column(2), &joined, fallback),
            "fallback"
        );
    }

    #[test]
    fn multiple_filters_are_combined_with_and() {
        assert!(conjunction::<ColIdx>(None).is_none());
        assert!(conjunction::<ColIdx>(Some(vec![])).is_none());
        let combined = conjunction(Some(vec![column(0), column(1)]));
        assert!(matches!(
            combined,
            Some(Expression::Binary { op: Op::And, .. })
        ));
    }

    #[test]
    fn index_lookups_match_column_equals_constant_in_either_order() {
        let three = || Expression::Constant(Some(Value::INT(3)));
        assert_eq!(
            column_equals_constant(&eq(column(2), three())),
            Some((ColIdx(2), Value::INT(3)))
        );
        assert_eq!(
            column_equals_constant(&eq(three(), column(2))),
            Some((ColIdx(2), Value::INT(3)))
        );
        assert_eq!(
            column_equals_constant(&eq(column(2), Expression::Constant(None))),
            None,
            "NULL is never looked up in an index"
        );
        assert_eq!(
            left_right_equi_join(&eq(join_column(Side::Right, 4), join_column(Side::Left, 1))),
            Some((1, 4))
        );
        assert_eq!(
            left_right_equi_join(&eq(join_column(Side::Left, 4), join_column(Side::Left, 1))),
            None
        );
    }

    fn int_schema(cols: &[&str]) -> Schema {
        Schema::new(
            cols.iter()
                .map(|c| (DataType::INT, c.to_string()))
                .collect(),
        )
    }

    fn table_scope(side: Side, cols: &[&str]) -> TableScope {
        TableScope {
            oid: 0,
            side,
            first_col: 0,
            schema: int_schema(cols),
        }
    }

    fn plan_error(stmt: BoundStatement) -> PlannerError {
        let (catalog, _bpm, _dir) = crate::testing::catalog(16);
        match Planner::new(&catalog).plan(stmt) {
            Ok(_) => panic!("expected the statement to fail to plan"),
            Err(e) => e,
        }
    }

    #[test]
    fn single_table_statements_without_exactly_one_table_are_rejected() {
        let err = plan_error(BoundStatement::Insert(BoundInsert {
            scope: Scope::new(),
            rows: vec![],
        }));
        assert_eq!(
            err,
            PlannerError::NotOneTable {
                statement: "INSERT",
                tables: 0
            }
        );
        assert_eq!(err.to_string(), "INSERT needs exactly one table, got 0");

        let two_tables = Scope::from([
            ("a".to_string(), table_scope(Side::Left, &["x"])),
            ("b".to_string(), table_scope(Side::Left, &["y"])),
        ]);
        assert_eq!(
            plan_error(BoundStatement::Delete(BoundDelete {
                scope: two_tables,
                filters: None,
            })),
            PlannerError::NotOneTable {
                statement: "DELETE",
                tables: 2
            }
        );
    }

    #[test]
    fn window_input_narrower_than_the_bound_from_output_is_rejected() {
        let scope = Scope::from([("a".to_string(), table_scope(Side::Left, &["x", "y"]))]);
        let one_column = Schema::new(vec![(DataType::INT, "x".to_string())]);
        let err = super::check_window_input_width(&scope, &one_column).unwrap_err();
        assert_eq!(
            err,
            PlannerError::WindowInputWidth {
                bound: 2,
                planned: 1
            }
        );
        assert_eq!(
            err.to_string(),
            "window calls were bound after 2 FROM columns, but the planned FROM output has 1"
        );
    }

    #[test]
    fn join_without_one_left_and_one_right_table_is_rejected() {
        let scope = Scope::from([
            ("a".to_string(), table_scope(Side::Left, &["x"])),
            ("b".to_string(), table_scope(Side::Left, &["y"])),
        ]);
        let err = plan_error(BoundStatement::Select(BoundSelect {
            scope,
            join: Some(BoundJoin {
                join_type: JoinType::Inner,
                predicate: Expression::Constant(Some(Value::BOOLEAN(true))),
            }),
            projections: vec![],
            distinct: false,
            filters: None,
            group_bys: vec![],
            aggregates: vec![],
            windows: vec![],
            sort: None,
            limit: None,
        }));
        assert_eq!(err, PlannerError::JoinSides { tables: 2 });
        assert_eq!(
            err.to_string(),
            "a join needs exactly one left and one right table, got 2 tables"
        );
    }

    #[test]
    fn create_index_on_a_column_the_table_lacks_is_rejected() {
        let err = plan_error(BoundStatement::CreateIndex(BoundCreateIndex {
            scope: Scope::from([("t".to_string(), table_scope(Side::Left, &["a"]))]),
            name: "i".to_string(),
            index_schema: int_schema(&["b"]),
        }));
        assert_eq!(
            err,
            PlannerError::UnknownIndexColumn {
                column: "b".to_string()
            }
        );
        assert_eq!(err.to_string(), "CREATE INDEX references unknown column: b");
    }

    fn eq<C>(l: Expression<C>, r: Expression<C>) -> Expression<C> {
        Expression::Binary {
            left: Box::new(l),
            right: Box::new(r),
            op: Op::Eq,
        }
    }

    fn and<C>(l: Expression<C>, r: Expression<C>) -> Expression<C> {
        Expression::Binary {
            left: Box::new(l),
            right: Box::new(r),
            op: Op::And,
        }
    }

    fn typed_join_column(side: Side, col: u32, dtype: DataType) -> JoinPredicate {
        Expression::Column {
            col: JoinColumn {
                side,
                col: ColIdx(col),
            },
            dtype,
        }
    }

    fn key_cols(exprs: &[Expression]) -> Vec<u32> {
        exprs
            .iter()
            .map(|e| match e {
                Expression::Column { col, .. } => col.0,
                _ => panic!("hash join keys are plain columns"),
            })
            .collect()
    }

    #[test]
    fn hash_join_keys_take_every_cross_side_equality_and_keep_the_rest_as_residual() {
        let left = int_schema(&["c0", "c1", "c2"]);
        let right = int_schema(&["c0", "c1", "c2"]);
        let predicate = and(
            and(
                eq(join_column(Side::Left, 0), join_column(Side::Right, 2)),
                eq(join_column(Side::Left, 1), join_column(Side::Left, 2)),
            ),
            and(
                eq(join_column(Side::Right, 1), join_column(Side::Left, 2)),
                Expression::Binary {
                    left: Box::new(join_column(Side::Left, 0)),
                    right: Box::new(join_column(Side::Right, 0)),
                    op: Op::Lt,
                },
            ),
        );
        let keys = hash_join_keys(predicate, &left, &right).expect("two equalities are keys");
        assert_eq!(key_cols(&keys.left_exprs), vec![0, 2]);
        assert_eq!(key_cols(&keys.right_exprs), vec![2, 1]);
        let residual = keys
            .residual
            .expect("the same-side equality and the < stay");
        assert_eq!(conjuncts(&residual).len(), 2);
    }

    #[test]
    fn hash_join_keys_without_residual_when_the_condition_is_only_equalities() {
        let schema = int_schema(&["c0"]);
        let keys = hash_join_keys(
            eq(join_column(Side::Right, 0), join_column(Side::Left, 0)),
            &schema,
            &schema,
        )
        .expect("one equality is a key");
        assert_eq!(key_cols(&keys.left_exprs), vec![0]);
        assert_eq!(key_cols(&keys.right_exprs), vec![0]);
        assert!(keys.residual.is_none());
    }

    #[test]
    fn hash_join_keys_reject_conditions_without_a_same_type_cross_side_equality() {
        let ints = int_schema(&["c0", "c1"]);
        let floats = Schema::new(vec![(DataType::FLOAT, "f".to_string())]);
        let mixed = eq(
            typed_join_column(Side::Left, 0, DataType::INT),
            typed_join_column(Side::Right, 0, DataType::FLOAT),
        );
        assert!(hash_join_keys(mixed, &ints, &floats).is_err());

        let same_side = eq(join_column(Side::Left, 0), join_column(Side::Left, 1));
        assert!(hash_join_keys(same_side, &ints, &ints).is_err());

        let or = Expression::Binary {
            left: Box::new(eq(join_column(Side::Left, 0), join_column(Side::Right, 0))),
            right: Box::new(eq(join_column(Side::Left, 1), join_column(Side::Right, 1))),
            op: Op::Or,
        };
        assert!(
            hash_join_keys(or, &ints, &ints).is_err(),
            "an equality under OR is not a key"
        );
    }
}
