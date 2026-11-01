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
    Update(UpdatePlanNode),
    Delete(DeletePlanNode),
    IndexScan(IndexScanPlanNode),
    Filter(FilterPlanNode),
    Projection(ProjectionPlanNode),
    Aggregation(AggregationPlanNode),
    NestedLoopJoin(NestedLoopJoinPlanNode),
    NestedIndexJoin(NestedIndexJoinPlanNode),
    HashJoin(HashJoinPlanNode),
    Sort(SortPlanNode),
    ExternalMergeSort(ExternalMergeSortPlanNode),
    Limit(LimitPlanNode),
    WindowFunction(WindowFunctionPlanNode),
    CreateTable(CreateTablePlanNode),
    DropTable(DropTablePlanNode),
    CreateIndex(CreateIndexPlanNode),
    DropIndex(DropIndexPlanNode),
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
    pub assignments: Vec<ColumnAssignment>,
    pub child: Box<Plan>,
    pub table_oid: u32,
}

/// `SET <column> = <value>` for one column of an UPDATE.
pub struct ColumnAssignment {
    pub col_idx: u32,
    pub value: Expression,
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
    pub schema: Schema,
    pub expressions: Vec<Expression>,
    pub child: Box<Plan>,
}

/// `SELECT SUM(a + b), MIN(c) ... GROUP BY c` has `group_bys = [c]` and
/// `aggregates = [(a + b, SUM), (c, MIN)]`.
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
    /// Window-only: the 1-based position of the row's peer group within its partition.
    RANK,
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
    /// Schema of the inner table that is probed through the index.
    pub table_schema: Schema,
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

/// Computes window aggregates without collapsing rows. For
/// `SELECT salary * 2, AVG(salary) OVER (PARTITION BY dept) FROM emp`, `columns` is
/// `[salary * 2]` and `aggregates` is `[(salary, AVG)]`; each output row is the
/// `columns` values followed by one value per aggregate.
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

pub struct CreateTablePlanNode {
    pub table_name: String,
    pub schema: Schema,
    pub if_not_exists: bool,
}

pub struct DropTablePlanNode {
    pub table_oid: u32,
    pub if_exists: bool,
}

pub struct CreateIndexPlanNode {
    pub index_name: String,
    pub table_oid: u32,
    pub key_columns: Vec<u32>,
}

pub struct DropIndexPlanNode {
    pub index_oid: u32,
    pub if_exists: bool,
}

/// `Expression::Column::tuple_idx` of a column read from the left (or only) input.
pub const LEFT_TUPLE: u8 = 0;
/// `Expression::Column::tuple_idx` of a column read from the right input of a join.
pub const RIGHT_TUPLE: u8 = 1;

#[derive(Clone)]
pub enum Expression {
    Constant(Option<Value>),
    Column {
        /// [`LEFT_TUPLE`] or [`RIGHT_TUPLE`].
        tuple_idx: u8,
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
    /// The type this expression evaluates to, or `None` for a NULL constant.
    pub fn dtype(&self) -> Option<DataType> {
        match self {
            Expression::Column { dtype, .. } => Some(*dtype),
            Expression::Constant(Some(v)) => Some(v.data_type()),
            Expression::Constant(None) => None,
            Expression::Binary { left, right, op } => match op {
                Op::Add | Op::Sub | Op::Mul | Op::Div => left.dtype().or_else(|| right.dtype()),
                _ => Some(DataType::BOOLEAN),
            },
        }
    }

    /// Evaluates the expression against one tuple. In a join schema, right-input
    /// columns are read at `schema.join_offset + col_idx`.
    pub fn evaluate(&self, tuple: &Tuple, schema: &Schema) -> Result<Option<Value>, ExecutorError> {
        match self {
            Expression::Binary { left, right, op } => {
                let l = left.evaluate(tuple, schema)?;
                let r = right.evaluate(tuple, schema)?;
                Ok(eval_binary(op, l, r))
            }
            Expression::Column {
                tuple_idx, col_idx, ..
            } => {
                let idx = match *tuple_idx {
                    LEFT_TUPLE => *col_idx as usize,
                    RIGHT_TUPLE => {
                        let offset = schema.join_offset.ok_or_else(|| {
                            ExecutorError(
                                "right-input column evaluated against a non-join schema"
                                    .to_string(),
                            )
                        })?;
                        offset + *col_idx as usize
                    }
                    other => return Err(invalid_tuple_idx(other)),
                };
                schema
                    .get_value(tuple, idx)
                    .map_err(|_| ExecutorError("error when evaluating".to_string()))
            }
            Expression::Constant(v) => Ok(*v),
        }
    }

    /// Evaluates the expression against the two input tuples of a join.
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
                let (tup, sch) = match *tuple_idx {
                    LEFT_TUPLE => (left_tuple, left_schema),
                    RIGHT_TUPLE => (right_tuple, right_schema),
                    other => return Err(invalid_tuple_idx(other)),
                };
                sch.get_value(tup, *col_idx as usize)
                    .map_err(|_| ExecutorError("error when evaluating".to_string()))
            }
            Expression::Constant(v) => Ok(*v),
        }
    }
}

fn invalid_tuple_idx(tuple_idx: u8) -> ExecutorError {
    ExecutorError(format!(
        "column tuple_idx {tuple_idx} is neither LEFT_TUPLE nor RIGHT_TUPLE"
    ))
}

fn eval_binary(op: &Op, l: Option<Value>, r: Option<Value>) -> Option<Value> {
    use Value::*;

    match op {
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

        Op::And => match (l, r) {
            (Some(BOOLEAN(false)), _) | (_, Some(BOOLEAN(false))) => Some(BOOLEAN(false)),
            (Some(BOOLEAN(a)), Some(BOOLEAN(b))) => Some(BOOLEAN(a && b)),
            (None, _) | (_, None) => None,
            _ => unreachable!("type-checked at plan time"),
        },

        Op::Or => match (l, r) {
            (Some(BOOLEAN(true)), _) | (_, Some(BOOLEAN(true))) => Some(BOOLEAN(true)),
            (Some(BOOLEAN(a)), Some(BOOLEAN(b))) => Some(BOOLEAN(a || b)),
            (None, _) | (_, None) => None,
            _ => unreachable!("type-checked at plan time"),
        },
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(v: Option<Value>) -> Expression {
        Expression::Constant(v)
    }

    fn boolean(b: bool) -> Expression {
        constant(Some(Value::BOOLEAN(b)))
    }

    fn null() -> Expression {
        constant(None)
    }

    fn eval(l: Expression, op: Op, r: Expression) -> Result<Option<Value>, ExecutorError> {
        Expression::Binary {
            left: Box::new(l),
            right: Box::new(r),
            op,
        }
        .evaluate(&Tuple { data: vec![] }, &Schema::new(vec![]))
    }

    fn eval_ok(l: Expression, op: Op, r: Expression) -> Option<Value> {
        eval(l, op, r).expect("evaluates")
    }

    #[test]
    fn and_follows_three_valued_logic() {
        let t = Some(Value::BOOLEAN(true));
        let f = Some(Value::BOOLEAN(false));
        assert_eq!(eval_ok(boolean(true), Op::And, boolean(true)), t);
        assert_eq!(eval_ok(boolean(true), Op::And, null()), None);
        assert_eq!(
            eval_ok(null(), Op::And, boolean(false)),
            f,
            "FALSE wins over NULL"
        );
        assert_eq!(
            eval_ok(boolean(false), Op::And, null()),
            f,
            "FALSE wins over NULL"
        );
        assert_eq!(eval_ok(null(), Op::And, null()), None);
    }

    #[test]
    fn or_follows_three_valued_logic() {
        let t = Some(Value::BOOLEAN(true));
        let f = Some(Value::BOOLEAN(false));
        assert_eq!(eval_ok(boolean(false), Op::Or, boolean(false)), f);
        assert_eq!(eval_ok(boolean(false), Op::Or, null()), None);
        assert_eq!(
            eval_ok(null(), Op::Or, boolean(true)),
            t,
            "TRUE wins over NULL"
        );
        assert_eq!(
            eval_ok(boolean(true), Op::Or, null()),
            t,
            "TRUE wins over NULL"
        );
        assert_eq!(eval_ok(null(), Op::Or, null()), None);
    }

    #[test]
    fn arithmetic_and_comparison_propagate_null() {
        let one = || constant(Some(Value::INT(1)));
        for op in [
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Eq,
            Op::NEq,
            Op::Lt,
            Op::Gte,
        ] {
            assert_eq!(eval_ok(one(), op.clone(), null()), None);
            assert_eq!(eval_ok(null(), op, one()), None);
        }
    }

    #[test]
    fn right_input_column_reads_past_the_join_offset() {
        let mut schema = Schema::new(vec![
            (DataType::INT, "l".to_string()),
            (DataType::INT, "r".to_string()),
        ]);
        let tuple = schema
            .encode_tuple(&[Some(Value::INT(1)), Some(Value::INT(2))])
            .unwrap();
        let right_col0 = Expression::Column {
            tuple_idx: RIGHT_TUPLE,
            col_idx: 0,
            dtype: DataType::INT,
        };
        assert!(
            right_col0.evaluate(&tuple, &schema).is_err(),
            "a right-input column needs a join schema"
        );
        schema.join_offset = Some(1);
        assert_eq!(
            right_col0.evaluate(&tuple, &schema).unwrap(),
            Some(Value::INT(2))
        );
    }

    #[test]
    fn column_with_unknown_tuple_idx_is_an_error() {
        let schema = Schema::new(vec![(DataType::INT, "a".to_string())]);
        let tuple = schema.encode_tuple(&[Some(Value::INT(1))]).unwrap();
        let col = Expression::Column {
            tuple_idx: 2,
            col_idx: 0,
            dtype: DataType::INT,
        };
        assert!(col.evaluate(&tuple, &schema).is_err());
        assert!(col.evaluate_join(&tuple, &schema, &tuple, &schema).is_err());
    }
}
