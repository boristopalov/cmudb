use std::ops::Bound;

use crate::catalog::Value;
use crate::catalog::{Catalog, DataType, Schema};
use crate::parser::{
    BoundCreateIndex, BoundCreateTable, BoundDelete, BoundInsert, BoundJoin, BoundSelect,
    BoundStatement, BoundUpdate, BoundValues, Scope, TableScope,
};
use crate::processor::plan::{
    AggType, AggregationPlanNode, ColumnAssignment, CreateIndexPlanNode, CreateTablePlanNode,
    DeletePlanNode, Expression, FilterPlanNode, IndexScanPlanNode, InsertPlanNode, JoinType,
    LEFT_TUPLE, LimitPlanNode, NestedIndexJoinPlanNode, NestedLoopJoinPlanNode, Op, Plan,
    ProjectionPlanNode, RIGHT_TUPLE, SeqScanPlanNode, SortPlanNode, UpdatePlanNode, ValuesPlanNode,
};

/// Turns bound statements into plans, without statistics or a cost model.
///
/// A `col = constant` filter on a table with a single-column index on `col` becomes an
/// `IndexScan`; an inner `left.col = right.col` join whose right table has a single-column
/// index on its join column becomes a `NestedIndexJoin`. Everything else is a `SeqScan`
/// or a `NestedLoopJoin`.
pub struct Planner<'a> {
    catalog: &'a Catalog,
}

#[derive(Debug)]
pub struct PlannerError(pub String);

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
        }
    }

    fn plan_select(&self, sel: BoundSelect) -> Result<Plan, PlannerError> {
        let BoundSelect {
            scope,
            join,
            projections,
            filters,
            group_bys,
            aggregates,
            sort,
            limit,
        } = sel;
        let filter_expr = conjunction(filters);

        let (mut plan, mut source_schema, remaining_filter) = match join {
            Some(j) => {
                let mut sides: Vec<&TableScope> = scope.values().collect();
                sides.sort_by_key(|ts| ts.idx);
                let [left, right] = sides[..] else {
                    return Err(PlannerError(format!(
                        "a join needs exactly two tables, got {}",
                        sides.len()
                    )));
                };

                let cols: Vec<(DataType, String)> = left
                    .schema
                    .cols
                    .iter()
                    .chain(right.schema.cols.iter())
                    .map(|c| (c.dtype, c.name.clone()))
                    .collect();
                let mut s = Schema::new(cols);
                s.join_offset = Some(left.schema.cols.len());
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

        if let Some(order_by_exprs) = sort {
            plan = Plan::Sort(SortPlanNode {
                schema: source_schema.clone(),
                child: Box::new(plan),
                order_by_exprs,
            });
        }

        let proj_cols: Vec<(DataType, String)> = projections
            .iter()
            .enumerate()
            .map(|(i, e)| {
                (
                    e.dtype().unwrap_or(DataType::INT),
                    output_column_name(e, &source_schema, || format!("col{i}")),
                )
            })
            .collect();
        let proj_schema = Schema::new(proj_cols);
        plan = Plan::Projection(ProjectionPlanNode {
            schema: proj_schema.clone(),
            expressions: projections,
            child: Box::new(plan),
        });

        if let Some(n) = limit {
            plan = Plan::Limit(LimitPlanNode {
                schema: proj_schema,
                child: Box::new(plan),
                limit: n,
            });
        }

        Ok(plan)
    }

    fn plan_values(&self, v: BoundValues) -> Plan {
        let schema = if let Some(first) = v.rows.first() {
            let cols: Vec<(DataType, String)> = first
                .iter()
                .enumerate()
                .map(|(i, e)| (e.dtype().unwrap_or(DataType::INT), format!("col{i}")))
                .collect();
            Schema::new(cols)
        } else {
            Schema::new(vec![])
        };
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
        for (col_expr, value) in upd.cols.into_iter().zip(upd.new_cols) {
            let Expression::Column { col_idx, .. } = col_expr else {
                return Err(PlannerError(
                    "UPDATE target must be a column reference".into(),
                ));
            };
            assignments.push(ColumnAssignment { col_idx, value });
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
                    .map(|i| i as u32)
                    .ok_or_else(|| {
                        PlannerError(format!(
                            "CREATE INDEX references unknown column: {}",
                            col.name
                        ))
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
            && value.data_type() == ts.schema.cols[col_idx as usize].dtype
            && let Some(index_oid) = self.index_for_cols(ts.oid, &[col_idx as usize])
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
                    tuple_idx: LEFT_TUPLE,
                    col_idx: outer_col,
                    dtype: left_ts.schema.cols[outer_col as usize].dtype,
                },
                table_oid: right_ts.oid,
                table_schema: right_ts.schema.clone(),
                index_oid,
                join_type: JoinType::Inner,
            });
        }

        Plan::NestedLoopJoin(NestedLoopJoinPlanNode {
            schema: out_schema,
            left: Box::new(left),
            right: Box::new(right),
            predicate: join.predicate,
            join_type: join.join_type,
        })
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
fn conjunction(filters: Option<Vec<Expression>>) -> Option<Expression> {
    filters?
        .into_iter()
        .reduce(|left, right| Expression::Binary {
            left: Box::new(left),
            right: Box::new(right),
            op: Op::And,
        })
}

fn only_table<'s>(scope: &'s Scope, statement: &str) -> Result<&'s TableScope, PlannerError> {
    let mut tables = scope.values();
    match (tables.next(), tables.next()) {
        (Some(ts), None) => Ok(ts),
        _ => Err(PlannerError(format!(
            "{statement} needs exactly one table, got {}",
            scope.len()
        ))),
    }
}

/// The name of the source column a plain column reference reads, else `fallback()`.
fn output_column_name(
    expr: &Expression,
    source_schema: &Schema,
    fallback: impl FnOnce() -> String,
) -> String {
    let Expression::Column {
        tuple_idx, col_idx, ..
    } = expr
    else {
        return fallback();
    };
    let offset = match *tuple_idx {
        RIGHT_TUPLE => source_schema.join_offset.unwrap_or(0),
        _ => 0,
    };
    source_schema
        .cols
        .get(offset + *col_idx as usize)
        .map(|c| c.name.clone())
        .unwrap_or_else(fallback)
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
        let dtype = match agg_type {
            AggType::COUNT => DataType::INT,
            _ => expr.dtype().unwrap_or(DataType::INT),
        };
        (dtype, format!("{}_{}", agg_type_name(agg_type), i))
    });
    Schema::new(group_cols.chain(agg_cols).collect())
}

/// `(col_idx, value)` when `filter` is `col = value` or `value = col` on the scanned table.
/// An index lookup encodes `value` with its own type, so callers use the index only when
/// `value` has the column's type; mixed INT and FLOAT equality goes through a filtered scan.
fn column_equals_constant(filter: &Expression) -> Option<(u32, Value)> {
    let Expression::Binary {
        left,
        right,
        op: Op::Eq,
    } = filter
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (
            Expression::Column {
                tuple_idx: LEFT_TUPLE,
                col_idx,
                ..
            },
            Expression::Constant(Some(v)),
        )
        | (
            Expression::Constant(Some(v)),
            Expression::Column {
                tuple_idx: LEFT_TUPLE,
                col_idx,
                ..
            },
        ) => Some((*col_idx, *v)),
        _ => None,
    }
}

/// `(left_col, right_col)` when `predicate` is an equality between a left-input column
/// and a right-input column, in either order.
fn left_right_equi_join(predicate: &Expression) -> Option<(u32, u32)> {
    let Expression::Binary {
        left,
        right,
        op: Op::Eq,
    } = predicate
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (
            Expression::Column {
                tuple_idx: LEFT_TUPLE,
                col_idx: outer,
                ..
            },
            Expression::Column {
                tuple_idx: RIGHT_TUPLE,
                col_idx: inner,
                ..
            },
        )
        | (
            Expression::Column {
                tuple_idx: RIGHT_TUPLE,
                col_idx: inner,
                ..
            },
            Expression::Column {
                tuple_idx: LEFT_TUPLE,
                col_idx: outer,
                ..
            },
        ) => Some((*outer, *inner)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(tuple_idx: u8, col_idx: u32) -> Expression {
        Expression::Column {
            tuple_idx,
            col_idx,
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
            tuple_idx: LEFT_TUPLE,
            col_idx: 1,
            dtype: DataType::FLOAT,
        };
        let schema = aggregation_output_schema(
            &[column(LEFT_TUPLE, 0)],
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
    fn output_column_name_offsets_right_input_columns_by_the_join_offset() {
        let mut joined = Schema::new(vec![
            (DataType::INT, "l".to_string()),
            (DataType::INT, "r".to_string()),
        ]);
        joined.join_offset = Some(1);
        let fallback = || "fallback".to_string();
        assert_eq!(
            output_column_name(&column(LEFT_TUPLE, 0), &joined, fallback),
            "l"
        );
        assert_eq!(
            output_column_name(&column(RIGHT_TUPLE, 0), &joined, fallback),
            "r"
        );
        assert_eq!(
            output_column_name(&column(RIGHT_TUPLE, 1), &joined, fallback),
            "fallback"
        );
    }

    #[test]
    fn multiple_filters_are_combined_with_and() {
        assert!(conjunction(None).is_none());
        assert!(conjunction(Some(vec![])).is_none());
        let combined = conjunction(Some(vec![column(LEFT_TUPLE, 0), column(LEFT_TUPLE, 1)]));
        assert!(matches!(
            combined,
            Some(Expression::Binary { op: Op::And, .. })
        ));
    }

    #[test]
    fn index_lookups_match_column_equals_constant_in_either_order() {
        let eq = |l: Expression, r: Expression| Expression::Binary {
            left: Box::new(l),
            right: Box::new(r),
            op: Op::Eq,
        };
        let three = || Expression::Constant(Some(Value::INT(3)));
        assert_eq!(
            column_equals_constant(&eq(column(LEFT_TUPLE, 2), three())),
            Some((2, Value::INT(3)))
        );
        assert_eq!(
            column_equals_constant(&eq(three(), column(LEFT_TUPLE, 2))),
            Some((2, Value::INT(3)))
        );
        assert_eq!(
            column_equals_constant(&eq(column(LEFT_TUPLE, 2), Expression::Constant(None))),
            None,
            "NULL is never looked up in an index"
        );
        assert_eq!(
            left_right_equi_join(&eq(column(RIGHT_TUPLE, 4), column(LEFT_TUPLE, 1))),
            Some((1, 4))
        );
        assert_eq!(
            left_right_equi_join(&eq(column(LEFT_TUPLE, 4), column(LEFT_TUPLE, 1))),
            None
        );
    }
}
