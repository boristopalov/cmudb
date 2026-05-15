use std::ops::Bound;

use crate::catalog::{Catalog, DataType, Schema};
use crate::parser::{
    BoundCreateIndex, BoundCreateTable, BoundDelete, BoundInsert, BoundJoin, BoundSelect,
    BoundStatement, BoundUpdate, BoundValues, TableScope,
};
use crate::processor::plan::{
    AggType, AggregationPlanNode, CreateIndexPlanNode, CreateTablePlanNode, DeletePlanNode,
    Expression, FilterPlanNode, IndexScanPlanNode, InsertPlanNode, JoinType, LimitPlanNode,
    NestedIndexJoinPlanNode, NestedLoopJoinPlanNode, Op, Plan, ProjectionPlanNode, SeqScanPlanNode,
    SortPlanNode, UpdatePlanNode, ValuesPlanNode,
};

/// A deliberately minimal planner. No statistics, no cost model.
/// The only real decisions:
///   1) for a single-table scan with a `col = const` filter, use IndexScan
///      when there's a single-column index on `col`.
///   2) for a 2-way INNER join with a `left.col = right.col` predicate,
///      use NestedIndexJoin when the inner side has a single-column index
///      on its join column; otherwise NestedLoopJoin.
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
        let filter_expr = filters.and_then(|mut v| v.pop());

        let (mut plan, mut source_schema, remaining_filter) = match join {
            Some(j) => {
                let mut sides: Vec<&TableScope> = scope.values().collect();
                debug_assert_eq!(sides.len(), 2, "only 2-way joins supported");
                sides.sort_by_key(|ts| ts.idx);
                let (left, right) = (sides[0], sides[1]);

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
                let ts = scope.values().next().expect("non-empty scope");
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

        // If the binder discovered any GROUP BY exprs or aggregate function
        // calls, insert an Aggregation node here. Its output schema is
        //   [ group-by columns... , aggregate-result columns... ]
        // which matches the order the executor encodes tuples in.
        if !group_bys.is_empty() || !aggregates.is_empty() {
            let mut cols: Vec<(DataType, String)> = Vec::with_capacity(
                group_bys.len() + aggregates.len(),
            );
            for (i, gb) in group_bys.iter().enumerate() {
                let dtype = gb.dtype().unwrap_or(DataType::INT);
                // Borrow the source column name when the group-by is a plain
                // column ref; otherwise synthesize a generic name.
                let name = match gb {
                    Expression::Column {
                        tuple_idx, col_idx, ..
                    } => {
                        let join_off = source_schema.join_offset.unwrap_or(0);
                        let idx =
                            *col_idx as usize + if *tuple_idx == 1 { join_off } else { 0 };
                        source_schema
                            .cols
                            .get(idx)
                            .map(|c| c.name.clone())
                            .unwrap_or_else(|| format!("g{i}"))
                    }
                    _ => format!("g{i}"),
                };
                cols.push((dtype, name));
            }
            for (i, (expr, agg_type)) in aggregates.iter().enumerate() {
                let dtype = match agg_type {
                    AggType::COUNT => DataType::INT,
                    _ => expr.dtype().unwrap_or(DataType::INT),
                };
                let name = format!("{}_{}", agg_type_name(agg_type), i);
                cols.push((dtype, name));
            }
            let agg_schema = Schema::new(cols);
            plan = Plan::Aggregation(AggregationPlanNode {
                schema: agg_schema.clone(),
                child: Box::new(plan),
                group_bys,
                aggregates,
            });
            // From here on, projection / sort references resolve against
            // the aggregation's output schema, not the original source.
            source_schema = agg_schema;
        }

        if let Some(order_by_exprs) = sort {
            plan = Plan::Sort(SortPlanNode {
                schema: source_schema.clone(),
                child: Box::new(plan),
                order_by_exprs,
            });
        }

        // Output schema of the projection: column names are carried over from
        // the source for plain column refs, otherwise generated as `colN`.
        let proj_cols: Vec<(DataType, String)> = projections
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let dtype = e.dtype().unwrap_or(DataType::INT);
                let name = match e {
                    Expression::Column {
                        tuple_idx, col_idx, ..
                    } => {
                        let join_off = source_schema.join_offset.unwrap_or(0);
                        let idx = *col_idx as usize + if *tuple_idx == 1 { join_off } else { 0 };
                        source_schema
                            .cols
                            .get(idx)
                            .map(|c| c.name.clone())
                            .unwrap_or_else(|| format!("col{i}"))
                    }
                    _ => format!("col{i}"),
                };
                (dtype, name)
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
        // we can use the first tuple's columns to figure out the schema
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
        let ts = ins.scope.values().next().expect("INSERT scope");
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
        let ts = del.scope.values().next().expect("DELETE scope");
        let filter = del.filters.and_then(|mut v| v.pop());
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
        let ts = upd.scope.values().next().expect("UPDATE scope");
        let filter = upd.filters.and_then(|mut v| v.pop());
        let (mut scan, remaining) = self.build_scan(ts, filter);
        if let Some(pred) = remaining {
            scan = Plan::Filter(FilterPlanNode {
                schema: ts.schema.clone(),
                predicate: pred,
                child: Box::new(scan),
            });
        }

        let mut target_exprs = Vec::with_capacity(upd.cols.len());
        for (col_expr, new_expr) in upd.cols.into_iter().zip(upd.new_cols.into_iter()) {
            let Expression::Column { col_idx, .. } = col_expr else {
                return Err(PlannerError(
                    "UPDATE target must be a column reference".into(),
                ));
            };
            target_exprs.push((col_idx, new_expr));
        }

        Ok(Plan::Update(UpdatePlanNode {
            schema: ts.schema.clone(),
            target_exprs,
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
        let ts = c.scope.values().next().expect("CREATE INDEX scope");
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
        if let Some(Expression::Binary {
            left,
            right,
            op: Op::Eq,
        }) = filter.as_ref()
        {
            let probe = match (left.as_ref(), right.as_ref()) {
                (
                    Expression::Column {
                        tuple_idx: 0,
                        col_idx,
                        ..
                    },
                    Expression::Constant(Some(v)),
                )
                | (
                    Expression::Constant(Some(v)),
                    Expression::Column {
                        tuple_idx: 0,
                        col_idx,
                        ..
                    },
                ) => Some((*col_idx as usize, *v)),
                _ => None,
            };
            if let Some((col_idx, value)) = probe {
                // if we have and index on the join columns, use an index scan
                // otherwise sequential scan
                if let Some(index_oid) = self.index_for_cols(ts.oid, &[col_idx]) {
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
            }
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

        if matches!(join.join_type, JoinType::Inner) {
            if let Expression::Binary {
                left: l,
                right: r,
                op: Op::Eq,
            } = &join.predicate
            {
                let cols = match (l.as_ref(), r.as_ref()) {
                    (
                        Expression::Column {
                            tuple_idx: 0,
                            col_idx: outer,
                            ..
                        },
                        Expression::Column {
                            tuple_idx: 1,
                            col_idx: inner,
                            ..
                        },
                    )
                    | (
                        Expression::Column {
                            tuple_idx: 1,
                            col_idx: inner,
                            ..
                        },
                        Expression::Column {
                            tuple_idx: 0,
                            col_idx: outer,
                            ..
                        },
                    ) => Some((*outer, *inner)),
                    _ => None,
                };
                if let Some((outer_col, inner_col)) = cols {
                    if let Some(index_oid) =
                        self.index_for_cols(right_ts.oid, &[inner_col as usize])
                    {
                        return Plan::NestedIndexJoin(NestedIndexJoinPlanNode {
                            schema: out_schema,
                            child: Box::new(left),
                            key_predicate: Expression::Column {
                                tuple_idx: 0,
                                col_idx: outer_col,
                                dtype: left_ts.schema.cols[outer_col as usize].dtype,
                            },
                            table_oid: right_ts.oid,
                            table_schema: right_ts.schema.clone(),
                            index_oid,
                            join_type: JoinType::Inner,
                        });
                    }
                }
            }
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
