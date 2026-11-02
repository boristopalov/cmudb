use crate::catalog::{Schema, Value};
use crate::processor::executor::{Batch, Executor, ExecutorError, collect_rows};
use crate::processor::executors::sort::sorted_by_exprs;
use crate::processor::executors::{ExprKey, eval_key, not_opened, take_batch};
use crate::processor::plan::{
    AggType, AggregationPlanNode, Arithmetic, Expression, SortExpr, WindowFunctionPlanNode,
};
use crate::table_heap::{RecordId, Tuple};
use std::collections::HashMap;
use std::collections::hash_map::IntoIter;
use std::vec::IntoIter as VecIter;

pub struct AggregationExecutor<'a> {
    plan: &'a AggregationPlanNode,
    child: Box<dyn Executor + 'a>,
    batch_size: usize,
    iter: Option<IntoIter<ExprKey, AggVal>>,
}

impl<'a> AggregationExecutor<'a> {
    pub fn new(
        plan: &'a AggregationPlanNode,
        child: Box<dyn Executor + 'a>,
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            child,
            batch_size,
            iter: None,
        }
    }
}

impl<'a> Executor for AggregationExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let mut agg_hash_table: HashMap<ExprKey, AggVal> = HashMap::new();
        while let Some(batch) = self.child.next_batch()? {
            let schema = self.child.schema();
            for (tuple, _) in batch {
                let groups = eval_key(&self.plan.group_bys, &tuple, schema)?;
                let entry = agg_hash_table.entry(groups).or_insert_with(|| {
                    AggVal(
                        self.plan
                            .aggregates
                            .iter()
                            .map(|(_, agg_type)| AggState::new(agg_type))
                            .collect(),
                    )
                });
                for (state, (expr, _)) in entry.0.iter_mut().zip(self.plan.aggregates.iter()) {
                    state.update(expr.evaluate(&tuple, schema)?)?;
                }
            }
        }
        self.iter = Some(agg_hash_table.into_iter());
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("AggregationExecutor"))?;
        let mut rows = Vec::with_capacity(self.batch_size);
        for (key, val) in iter.by_ref().take(self.batch_size) {
            let mut out: Vec<Option<Value>> = Vec::with_capacity(key.0.len() + val.0.len());
            out.extend(key.0);
            for v in val.0 {
                out.push(v.value()?);
            }
            rows.push((self.plan.schema.encode_tuple(&out)?, RecordId::RESERVED));
        }
        Ok(Batch::from_rows(rows))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

struct AggVal(Vec<AggState>);

enum AggState {
    Max(Option<Value>),
    Min(Option<Value>),
    Sum(Option<Value>),
    Avg { sum: f64, count: i64 },
    Count(i32),
    Rank(i32),
}

impl AggState {
    fn new(agg_type: &AggType) -> Self {
        match agg_type {
            AggType::AVG => Self::Avg { sum: 0.0, count: 0 },
            AggType::MAX => Self::Max(None),
            AggType::MIN => Self::Min(None),
            AggType::SUM => Self::Sum(None),
            AggType::COUNT => Self::Count(0),
            AggType::RANK => Self::Rank(1),
        }
    }

    /// Moves a RANK to `pos` when a new peer group starts; other aggregates ignore it.
    fn update_rank(&mut self, new_peer_group: bool, pos: i32) {
        if let Self::Rank(rank) = self
            && new_peer_group
        {
            *rank = pos;
        }
    }

    /// Folds one input value in. NULLs are skipped, so `COUNT(col)` counts non-NULL values;
    /// RANK ignores values and moves through `update_rank`. SUM adds INTs as BIGINTs and errors
    /// when the total overflows a BIGINT; AVG adds as FLOAT, so its total does not overflow.
    fn update(&mut self, v: Option<Value>) -> Result<(), ExecutorError> {
        let Some(v) = v else { return Ok(()) };
        match self {
            Self::Count(count) => *count += 1,
            Self::Sum(current) => *current = Some(add_to(current.take(), sum_operand(v))?),
            Self::Avg { sum, count } => {
                let operand = v.as_f64().ok_or(ExecutorError::ArithmeticOperands {
                    op: Arithmetic::Add,
                    left: Value::float(*sum),
                    right: v,
                })?;
                *sum += operand;
                *count += 1;
            }
            Self::Max(current) => *current = Some(current.take().map_or(v.clone(), |c| c.max(v))),
            Self::Min(current) => *current = Some(current.take().map_or(v.clone(), |c| c.min(v))),
            Self::Rank(_) => {}
        }
        Ok(())
    }

    /// The aggregate over the values folded in so far. AVG is a FLOAT for every input type.
    fn value(&self) -> Result<Option<Value>, ExecutorError> {
        Ok(match self {
            Self::Sum(v) | Self::Max(v) | Self::Min(v) => v.clone(),
            Self::Count(c) | Self::Rank(c) => Some(Value::INT(*c)),
            Self::Avg { count: 0, .. } => None,
            Self::Avg { sum, count } => Some(Value::float(*sum / *count as f64)),
        })
    }
}

/// The value SUM adds: an INT widened to BIGINT, anything else as it is.
fn sum_operand(v: Value) -> Value {
    match v {
        Value::INT(i) => Value::BIGINT(i64::from(i)),
        other => other,
    }
}

fn add_to(total: Option<Value>, v: Value) -> Result<Value, ExecutorError> {
    match total {
        Some(total) => Arithmetic::Add.apply(total, v),
        None => Ok(v),
    }
}

pub struct WindowFunctionExecutor<'a> {
    plan: &'a WindowFunctionPlanNode,
    child: Box<dyn Executor + 'a>,
    batch_size: usize,
    iter: Option<VecIter<Tuple>>,
}

impl<'a> WindowFunctionExecutor<'a> {
    pub fn new(
        plan: &'a WindowFunctionPlanNode,
        child: Box<dyn Executor + 'a>,
        batch_size: usize,
    ) -> Self {
        Self {
            plan,
            child,
            batch_size,
            iter: None,
        }
    }
}

impl<'a> Executor for WindowFunctionExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let tups: Vec<Tuple> = collect_rows(self.child.as_mut())?
            .into_iter()
            .map(|(t, _)| t)
            .collect();

        let child_schema = self.child.schema();
        let partition_by: &[Expression] = self.plan.partition_by_exprs.as_deref().unwrap_or(&[]);
        let order_by: &[SortExpr] = self.plan.order_by_exprs.as_deref().unwrap_or(&[]);
        let sort_exprs: Vec<SortExpr> = partition_by
            .iter()
            .cloned()
            .map(SortExpr::ascending)
            .chain(order_by.iter().cloned())
            .collect();
        let order_keys: Vec<Expression> = order_by.iter().map(|key| key.expr.clone()).collect();
        let tups = sorted_by_exprs(tups, |t| t, &sort_exprs, child_schema)?;

        let running = self.plan.order_by_exprs.is_some();
        let fresh_aggregates = || -> Vec<AggState> {
            self.plan
                .aggregates
                .iter()
                .map(|(_, agg_type)| AggState::new(agg_type))
                .collect()
        };
        let mut partition_aggregates = fresh_aggregates();
        let mut peer_buffer: Vec<Tuple> = Vec::new();
        let mut output_tuples: Vec<Tuple> = Vec::with_capacity(tups.len());

        let mut curr_rank = 1;
        let mut row_num = 1;
        let mut prev_partition_key: Option<ExprKey> = None;
        let mut prev_order_key: Option<ExprKey> = None;

        for t in tups.iter() {
            let curr_partition_key = eval_key(partition_by, t, child_schema)?;
            let new_partition = prev_partition_key.as_ref() != Some(&curr_partition_key);
            let curr_order_key = eval_key(&order_keys, t, child_schema)?;
            let new_peer_group = new_partition || prev_order_key.as_ref() != Some(&curr_order_key);

            let flush_now = if new_partition {
                prev_partition_key.is_some()
            } else {
                running && new_peer_group && !peer_buffer.is_empty()
            };
            if flush_now {
                flush_peer_group(
                    self.plan,
                    &mut peer_buffer,
                    &mut partition_aggregates,
                    child_schema,
                    &mut output_tuples,
                )?;
            }
            if new_partition {
                partition_aggregates = fresh_aggregates();
                row_num = 1;
            }
            if new_peer_group {
                curr_rank = row_num;
            }
            for state in partition_aggregates.iter_mut() {
                state.update_rank(new_peer_group, curr_rank);
            }

            peer_buffer.push(t.clone());
            row_num += 1;
            prev_partition_key = Some(curr_partition_key);
            prev_order_key = Some(curr_order_key);
        }

        if !peer_buffer.is_empty() {
            flush_peer_group(
                self.plan,
                &mut peer_buffer,
                &mut partition_aggregates,
                child_schema,
                &mut output_tuples,
            )?;
        }

        self.iter = Some(output_tuples.into_iter());
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("WindowFunctionExecutor"))?;
        let mut rows = iter.by_ref().map(|tup| (tup, RecordId::RESERVED));
        Ok(take_batch(&mut rows, self.batch_size))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Folds every buffered peer row into `aggregates`, then emits each of those rows as the
/// plan's `columns` followed by the aggregates' current values, draining `buffer`.
fn flush_peer_group(
    plan: &WindowFunctionPlanNode,
    buffer: &mut Vec<Tuple>,
    aggregates: &mut [AggState],
    child_schema: &Schema,
    output: &mut Vec<Tuple>,
) -> Result<(), ExecutorError> {
    for buf_tup in buffer.iter() {
        for (state, (expr, _)) in aggregates.iter_mut().zip(plan.aggregates.iter()) {
            state.update(expr.evaluate(buf_tup, child_schema)?)?;
        }
    }
    for buf_tup in buffer.drain(..) {
        let mut combined = plan
            .columns
            .iter()
            .map(|e| e.evaluate(&buf_tup, child_schema))
            .collect::<Result<Vec<_>, _>>()?;
        for agg in aggregates.iter() {
            combined.push(agg.value()?);
        }
        output.push(plan.schema.encode_tuple(&combined)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::executor::DEFAULT_BATCH_SIZE;
    use crate::processor::executors::test_support::*;

    #[test]
    fn group_by_evaluation_errors_are_reported_not_grouped_as_null() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[2], &[1]]);
        let plan = AggregationPlanNode {
            schema: int_schema(&["g"]),
            child: Box::new(values_plan(&schema, &[])),
            group_bys: vec![divide_by_zero(0)],
            aggregates: vec![],
        };
        let mut agg = AggregationExecutor::new(&plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        let err = agg.open().unwrap_err();
        assert!(
            matches!(err, ExecutorError::DivisionByZero { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn window_output_is_its_columns_followed_by_its_aggregates() {
        let schema = int_schema(&["dept", "amount"]);
        let rows = values_plan(&schema, &[&[1, 10], &[1, 30], &[2, 5]]);
        let doubled_amount = Expression::Binary {
            left: Box::new(col(1)),
            right: Box::new(Expression::Constant(Some(Value::INT(2)))),
            op: crate::processor::plan::Op::Mul,
        };
        let out_schema = int_schema(&["doubled", "dept_total"]);
        let plan = WindowFunctionPlanNode {
            schema: out_schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            order_by_exprs: None,
            partition_by_exprs: Some(vec![col(0)]),
            aggregates: vec![(col(1), AggType::SUM)],
            columns: vec![doubled_amount],
        };
        let mut window =
            WindowFunctionExecutor::new(&plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        assert_eq!(
            decode_ints(&out_schema, drain(&mut window)),
            vec![vec![20, 40], vec![60, 40], vec![10, 5]]
        );
    }

    #[test]
    fn count_skips_nulls_and_rank_ignores_values() {
        let mut count = AggState::new(&AggType::COUNT);
        count.update(Some(Value::INT(7))).unwrap();
        count.update(None).unwrap();
        count.update(Some(Value::INT(8))).unwrap();
        assert_eq!(count.value().unwrap(), Some(Value::INT(2)));

        let mut rank = AggState::new(&AggType::RANK);
        rank.update(Some(Value::INT(100))).unwrap();
        assert_eq!(rank.value().unwrap(), Some(Value::INT(1)));
        rank.update_rank(false, 5);
        assert_eq!(
            rank.value().unwrap(),
            Some(Value::INT(1)),
            "same peer group keeps its rank"
        );
        rank.update_rank(true, 5);
        assert_eq!(rank.value().unwrap(), Some(Value::INT(5)));
    }

    #[test]
    fn avg_of_float_values_is_a_float() {
        let mut avg = AggState::new(&AggType::AVG);
        avg.update(Some(Value::FLOAT(1.5.into()))).unwrap();
        avg.update(Some(Value::FLOAT(2.0.into()))).unwrap();
        assert_eq!(avg.value().unwrap(), Some(Value::FLOAT(1.75.into())));
    }

    #[test]
    fn sum_of_ints_is_a_bigint_past_the_int_range() {
        let mut sum = AggState::new(&AggType::SUM);
        sum.update(Some(Value::INT(i32::MAX))).unwrap();
        sum.update(Some(Value::INT(1))).unwrap();
        assert_eq!(
            sum.value().unwrap(),
            Some(Value::BIGINT(i64::from(i32::MAX) + 1))
        );
    }

    #[test]
    fn sum_that_overflows_bigint_is_an_error() {
        let mut sum = AggState::new(&AggType::SUM);
        sum.update(Some(Value::BIGINT(i64::MAX))).unwrap();
        assert!(matches!(
            sum.update(Some(Value::INT(1))).unwrap_err(),
            ExecutorError::IntegerOverflow {
                op: Arithmetic::Add,
                left: i64::MAX,
                right: 1
            }
        ));
    }

    #[test]
    fn avg_of_ints_is_a_float() {
        let mut avg = AggState::new(&AggType::AVG);
        avg.update(Some(Value::INT(1))).unwrap();
        avg.update(Some(Value::INT(2))).unwrap();
        assert_eq!(avg.value().unwrap(), Some(Value::float(1.5)));
    }

    #[test]
    fn avg_of_bigints_past_the_bigint_range_is_a_float() {
        let mut avg = AggState::new(&AggType::AVG);
        avg.update(Some(Value::BIGINT(i64::MAX))).unwrap();
        avg.update(Some(Value::BIGINT(1))).unwrap();
        assert_eq!(
            avg.value().unwrap(),
            Some(Value::float((i64::MAX as f64 + 1.0) / 2.0))
        );
    }

    #[test]
    fn avg_of_a_non_number_is_an_error() {
        let mut avg = AggState::new(&AggType::AVG);
        assert!(matches!(
            avg.update(Some(Value::BOOLEAN(true))),
            Err(ExecutorError::ArithmeticOperands { .. })
        ));
    }

    #[test]
    fn avg_of_only_nulls_is_null() {
        let mut avg = AggState::new(&AggType::AVG);
        avg.update(None).unwrap();
        assert_eq!(avg.value().unwrap(), None);
    }

    #[test]
    fn next_before_open_is_an_error() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[1]]);
        let agg_plan = AggregationPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            group_bys: vec![],
            aggregates: vec![],
        };
        let mut agg = AggregationExecutor::new(&agg_plan, values_child(&rows), DEFAULT_BATCH_SIZE);
        let err = agg.next_batch().unwrap_err();
        assert!(
            matches!(
                err,
                ExecutorError::NotOpened {
                    executor: "AggregationExecutor"
                }
            ),
            "{err:?}"
        );
    }
}
