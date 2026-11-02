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
    iter: Option<IntoIter<ExprKey, AggregateState<'a>>>,
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
    /// Groups the whole input. Without GROUP BY there is one group even when the input is
    /// empty, so `SELECT COUNT(*) FROM empty` yields one row holding 0.
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()?;
        let aggregates = &self.plan.aggregates;
        let mut groups: HashMap<ExprKey, AggregateState<'a>> = HashMap::new();
        while let Some(batch) = self.child.next_batch()? {
            let schema = self.child.schema();
            for (tuple, _) in batch {
                let key = eval_key(&self.plan.group_bys, &tuple, schema)?;
                groups
                    .entry(key)
                    .or_insert_with(|| AggregateState::new(aggregates))
                    .update(&tuple, schema)?;
            }
        }
        if self.plan.group_bys.is_empty() && groups.is_empty() {
            groups.insert(ExprKey(Vec::new()), AggregateState::new(aggregates));
        }
        self.iter = Some(groups.into_iter());
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| not_opened("AggregationExecutor"))?;
        let mut rows = Vec::with_capacity(self.batch_size);
        for (key, state) in iter.by_ref().take(self.batch_size) {
            let mut out = key.0;
            state.append_values(&mut out)?;
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

/// The running value of each of a plan's aggregates, folded in one input row at a time.
/// GROUP BY keeps one per group and a window keeps one per partition.
struct AggregateState<'p> {
    aggregates: &'p [(Expression, AggType)],
    states: Vec<AggState>,
}

impl<'p> AggregateState<'p> {
    /// Every aggregate of `aggregates` over no rows.
    fn new(aggregates: &'p [(Expression, AggType)]) -> Self {
        Self {
            aggregates,
            states: aggregates
                .iter()
                .map(|(_, agg_type)| AggState::new(agg_type))
                .collect(),
        }
    }

    /// Folds in the value each aggregate's argument takes on `tuple`.
    fn update(&mut self, tuple: &Tuple, schema: &Schema) -> Result<(), ExecutorError> {
        for (state, (expr, _)) in self.states.iter_mut().zip(self.aggregates) {
            state.update(expr.evaluate(tuple, schema)?)?;
        }
        Ok(())
    }

    /// Moves every RANK to `pos` when a new peer group starts.
    fn update_rank(&mut self, new_peer_group: bool, pos: i32) {
        for state in &mut self.states {
            state.update_rank(new_peer_group, pos);
        }
    }

    /// Appends each aggregate's value over the rows folded in so far to `out`.
    fn append_values(&self, out: &mut Vec<Option<Value>>) -> Result<(), ExecutorError> {
        out.reserve(self.states.len());
        for state in &self.states {
            out.push(state.value()?);
        }
        Ok(())
    }
}

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
        let tuples: Vec<Tuple> = collect_rows(self.child.as_mut())?
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        let output = window_rows(self.plan, tuples, self.child.schema())?;
        self.iter = Some(output.into_iter());
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

/// Sorts `tuples` by partition and then by the window's ORDER BY, and emits each row as the
/// plan's `columns` followed by its aggregates. Without ORDER BY an aggregate covers the
/// whole partition; with it, each row sees the running value up to the end of its peer group.
fn window_rows(
    plan: &WindowFunctionPlanNode,
    tuples: Vec<Tuple>,
    child_schema: &Schema,
) -> Result<Vec<Tuple>, ExecutorError> {
    let partition_by: &[Expression] = plan.partition_by_exprs.as_deref().unwrap_or(&[]);
    let order_by: &[SortExpr] = plan.order_by_exprs.as_deref().unwrap_or(&[]);
    let sort_exprs: Vec<SortExpr> = partition_by
        .iter()
        .cloned()
        .map(SortExpr::ascending)
        .chain(order_by.iter().cloned())
        .collect();
    let order_keys: Vec<Expression> = order_by.iter().map(|key| key.expr.clone()).collect();
    let tuples = sorted_by_exprs(tuples, |t| t, &sort_exprs, child_schema)?;

    let running = plan.order_by_exprs.is_some();
    let mut partition_aggregates = AggregateState::new(&plan.aggregates);
    let mut peer_buffer: Vec<Tuple> = Vec::new();
    let mut output: Vec<Tuple> = Vec::with_capacity(tuples.len());

    let mut curr_rank = 1;
    let mut row_num = 1;
    let mut prev_partition_key: Option<ExprKey> = None;
    let mut prev_order_key: Option<ExprKey> = None;

    for t in tuples {
        let curr_partition_key = eval_key(partition_by, &t, child_schema)?;
        let new_partition = prev_partition_key.as_ref() != Some(&curr_partition_key);
        let curr_order_key = eval_key(&order_keys, &t, child_schema)?;
        let new_peer_group = new_partition || prev_order_key.as_ref() != Some(&curr_order_key);

        let flush_now = if new_partition {
            prev_partition_key.is_some()
        } else {
            running && new_peer_group && !peer_buffer.is_empty()
        };
        if flush_now {
            flush_peer_group(
                plan,
                &mut peer_buffer,
                &mut partition_aggregates,
                child_schema,
                &mut output,
            )?;
        }
        if new_partition {
            partition_aggregates = AggregateState::new(&plan.aggregates);
            row_num = 1;
        }
        if new_peer_group {
            curr_rank = row_num;
        }
        partition_aggregates.update_rank(new_peer_group, curr_rank);

        peer_buffer.push(t);
        row_num += 1;
        prev_partition_key = Some(curr_partition_key);
        prev_order_key = Some(curr_order_key);
    }

    flush_peer_group(
        plan,
        &mut peer_buffer,
        &mut partition_aggregates,
        child_schema,
        &mut output,
    )?;
    Ok(output)
}

/// Folds every buffered peer row into `aggregates`, then emits each of those rows as the
/// plan's `columns` followed by the aggregates' current values, draining `buffer`.
fn flush_peer_group(
    plan: &WindowFunctionPlanNode,
    buffer: &mut Vec<Tuple>,
    aggregates: &mut AggregateState,
    child_schema: &Schema,
    output: &mut Vec<Tuple>,
) -> Result<(), ExecutorError> {
    for buf_tup in buffer.iter() {
        aggregates.update(buf_tup, child_schema)?;
    }
    for buf_tup in buffer.drain(..) {
        let mut combined = plan
            .columns
            .iter()
            .map(|e| e.evaluate(&buf_tup, child_schema))
            .collect::<Result<Vec<_>, _>>()?;
        aggregates.append_values(&mut combined)?;
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
