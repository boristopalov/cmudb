use crate::{
    catalog::{DataType, Schema, Value},
    processor::executor::ExecutorError,
    table_heap::Tuple,
};
use std::ops::Bound;

pub enum Plan {
    SeqScan(SeqScanPlanNode),
    Insert(InsertPlanNode),
    Values(ValuesPlanNode),
}

pub struct SeqScanPlanNode {
    pub table_oid: u32,
    pub schema: Schema,
}

pub struct InsertPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub table_oid: u32,
}

pub struct ValuesPlanNode {
    pub schema: Schema,
    pub rows: Vec<Vec<Expression>>,
}

pub struct UpdatePlanNode {
    pub schema: Schema,
    pub target_exprs: Vec<(u32, Expression)>, // 1st element is the index of the column in the tuple we want to update
    pub child: Box<Plan>,
    pub table_oid: u32,
}

pub struct DeletePlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub table_oid: u32,
}

pub struct IndexScanPlanNode {
    pub schema: Schema,
    pub start: Bound<Vec<Value>>,
    pub end: Bound<Vec<Value>>,
    pub index_oid: u32,
}

pub struct FilterPlanNode {
    pub schema: Schema,
    pub predicate: Expression,
    pub child: Box<Plan>,
}

pub struct ProjectionPlanNode {
    pub expressions: Vec<Expression>,
    pub child: Box<Plan>,
}

/// Example: SUM(a+b), MIN(c) GROUP BY c;
/// group_bys would be len 1 vec![c]
/// aggregates would be len 2 vec![(a+b, SUM), (c, MIN)]
pub struct AggregationPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub group_bys: Vec<Expression>,
    pub aggregates: Vec<(Expression, AggType)>,
}

pub enum AggType {
    MAX,
    MIN,
    SUM,
    AVG,
    COUNT,
    RANK, // not an agg type but whatever
}

pub struct NestedLoopJoinPlanNode {
    pub schema: Schema,
    pub left: Box<Plan>,
    pub right: Box<Plan>,
    pub predicate: Expression,
    pub join_type: JoinType,
}

pub enum JoinType {
    Inner,
    Left,
}

pub struct NestedIndexJoinPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub key_predicate: Expression,
    pub table_oid: u32,
    pub table_schema: Schema, // the schema of the table we are joining on
    pub index_oid: u32,
    pub join_type: JoinType,
}

pub struct HashJoinPlanNode {
    pub schema: Schema,
    pub left: Box<Plan>,
    pub right: Box<Plan>,
    pub left_exprs: Vec<Expression>,
    pub right_exprs: Vec<Expression>,
    pub join_type: JoinType,
}

pub struct ExternalMergeSortPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub order_by_exprs: Vec<Expression>,
}

pub struct SortPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub order_by_exprs: Vec<Expression>,
}

/// Unlike aggregate functions, window functions are "self-contained" and don't collapse columns
/// down to the group bys + aggregates.
/// So we need to know which columns are present ahead of time.
/// This is how window functions work by definition.
/// We could not specify the columns, and stream back the window function columns,
/// plus the child columns, and have a separate projection pull from the window function,
/// but that would be wasteful.
///
/// For example:  `SELECT salary * 2, AVG(salary) OVER (PARTITION BY dept) FROM emp;`
///
/// Without providing columns, we would return salary directly, and a project would pull and return salary * 2.
/// If providing columns, we can directly compute salary * 2 and skip the projection
pub struct WindowFunctionPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub order_by_exprs: Option<Vec<Expression>>,
    pub partition_by_exprs: Option<Vec<Expression>>,
    pub aggregates: Vec<(Expression, AggType)>,
    pub columns: Vec<Expression>,
}

pub struct LimitPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub limit: u32,
}

#[derive(Clone)]
pub enum Expression {
    Constant(Option<Value>),
    Column {
        tuple_idx: u8, // 0 or 1, left valiue or right value, used in joins
        col_idx: u32,
        dtype: DataType,
    },
    Binary {
        left: Box<Expression>,
        right: Box<Expression>,
        op: Op,
    },
}

impl Expression {
    /// Evaluates am expression against a single tuple
    pub fn evaluate(&self, tuple: &Tuple, schema: &Schema) -> Result<Option<Value>, ExecutorError> {
        match self {
            Expression::Binary { left, right, op } => {
                let l = left.evaluate(tuple, schema)?;
                let r = right.evaluate(tuple, schema)?;
                return Ok(eval_binary(op, l, r));
            }
            Expression::Column {
                tuple_idx,
                col_idx,
                dtype,
            } => {
                let c_idx = *col_idx as usize;
                schema
                    .get_value(tuple, c_idx)
                    .map_err(|_| ExecutorError("error when evaluating".to_string()))
            }
            Expression::Constant(v) => Ok(v.clone()),
        }
    }

    /// Evaluates an expression against two tuples
    pub fn evaluate_join(
        &self,
        left_tuple: &Tuple,
        left_schema: &Schema,
        right_tuple: &Tuple,
        right_schema: &Schema,
    ) -> Result<Option<Value>, ExecutorError> {
        match self {
            Expression::Binary { left, right, op } => {
                let l = left.evaluate_join(left_tuple, left_schema, right_tuple, right_schema)?;
                let r = right.evaluate_join(left_tuple, left_schema, right_tuple, right_schema)?;
                Ok(eval_binary(op, l, r))
            }
            Expression::Column {
                tuple_idx, col_idx, ..
            } => {
                let (tup, sch) = match tuple_idx {
                    0 => (left_tuple, left_schema),
                    1 => (right_tuple, right_schema),
                    _ => unreachable!("tuple_idx must be 0 or 1"),
                };
                sch.get_value(tup, *col_idx as usize)
                    .map_err(|_| ExecutorError("error when evaluating".to_string()))
            }
            Expression::Constant(v) => Ok(v.clone()),
        }
    }
}

/// Some invariants that we should take care of in the Binder:
/// - reject different types of l and r
fn eval_binary(op: &Op, l: Option<Value>, r: Option<Value>) -> Option<Value> {
    use Value::*;

    match op {
        // arithmetic and comparison: NULL propagates
        Op::Add => Some(l? + r?),
        Op::Sub => Some(l? - r?),
        Op::Mul => Some(l? * r?),
        Op::Div => Some(l? / r?),

        Op::Eq => Some(BOOLEAN(l? == r?)),
        Op::NEq => Some(BOOLEAN(l? != r?)),
        Op::Lt => Some(BOOLEAN(l? < r?)),
        Op::Gt => Some(BOOLEAN(l? > r?)),
        Op::Lte => Some(BOOLEAN(l? <= r?)),
        Op::Gte => Some(BOOLEAN(l? >= r?)),

        //   TRUE  AND x    = x
        //   FALSE AND _    = FALSE
        //   NULL  AND NULL = NULL
        Op::And => match (l, r) {
            (Some(BOOLEAN(false)), _) | (_, Some(BOOLEAN(false))) => Some(BOOLEAN(false)),
            (Some(BOOLEAN(a)), Some(BOOLEAN(b))) => Some(BOOLEAN(a && b)),
            (None, _) | (_, None) => None,
            _ => unreachable!("type-checked at plan time"),
        },

        //   TRUE  OR _    = TRUE
        //   FALSE OR x    = x
        //   NULL  OR NULL = NULL
        Op::Or => match (l, r) {
            (Some(BOOLEAN(true)), _) | (_, Some(BOOLEAN(true))) => Some(BOOLEAN(true)),
            (Some(BOOLEAN(a)), Some(BOOLEAN(b))) => Some(BOOLEAN(a || b)),
            (None, _) | (_, None) => None,
            _ => unreachable!("type-checked at plan time"),
        },

        Op::Like | Op::In => todo!(),
    }
}

#[derive(Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NEq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
    Like,
    In,
}

// pub struct HashJoinPlanNode {
//     left: Box<Plan>,
//     right: Box<Plan>,
// }
