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

/// The binder should
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

#[derive(Clone)]
pub enum Expression {
    Constant(Value),
    Column {
        tuple_idx: u8, // left valiue or right value, used in joins
        col_idx: u32,
        dtype: DataType,
    },
    Binary {
        left: Box<Expression>,
        right: Box<Expression>,
        op: Op,
    },
}

/// TODO: add nulls and return Option<Value> instead of just Value
impl Expression {
    pub fn evaluate(&self, tuple: &Tuple, schema: &Schema) -> Result<Value, ExecutorError> {
        match self {
            Expression::Binary { left, right, op } => {
                let l = left.evaluate(tuple, schema)?;
                let r = right.evaluate(tuple, schema)?;
                return Ok(eval_binary(op, l, r));
            }
            Expression::Column {
                tuple_idx, // will be used for joins
                col_idx,
                dtype,
            } => {
                let c_idx = *col_idx as usize;
                schema
                    .get_value(tuple, c_idx)
                    .map_err(|_| ExecutorError("error when evaluating".to_string()))
            }
            Expression::Constant(v) => {
                return Ok(v.clone());
            }
        }
    }
}

/// Some invariants that we should take care of in the Binder:
/// - reject different types of l and r
fn eval_binary(op: &Op, l: Value, r: Value) -> Value {
    use Value::*;

    macro_rules! arith {
        ($f:tt) => {
            match (l, r) {
                (INT(a), INT(b))     => INT(a $f b),
                (FLOAT(a), FLOAT(b)) => FLOAT(a $f b),
                _ => unreachable!("type-checked at plan time"),
            }
        };
    }

    match op {
        Op::Add => arith!(+),
        Op::Sub => arith!(-),
        Op::Mul => arith!(*),
        Op::Div => arith!(/),

        Op::Eq => BOOLEAN(l == r),
        Op::NEq => BOOLEAN(l != r),
        Op::Lt => BOOLEAN(l < r),
        Op::Gt => BOOLEAN(l > r),
        Op::Lte => BOOLEAN(l <= r),
        Op::Gte => BOOLEAN(l >= r),

        Op::And => match (l, r) {
            (BOOLEAN(a), BOOLEAN(b)) => BOOLEAN(a && b),
            _ => unreachable!(),
        },
        Op::Or => match (l, r) {
            (BOOLEAN(a), BOOLEAN(b)) => BOOLEAN(a || b),
            _ => unreachable!(),
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
