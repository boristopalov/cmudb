use crate::{
    catalog::{DataType, Schema, Value},
    processor::executor::ExecutorError,
    table_heap::Tuple,
};
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::convert::Infallible;
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
    pub col_idx: ColIdx,
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
    pub predicate: JoinPredicate,
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
    pub order_by_exprs: Vec<SortExpr>,
}

pub struct SortPlanNode {
    pub schema: Schema,
    pub child: Box<Plan>,
    pub order_by_exprs: Vec<SortExpr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Where rows whose `ORDER BY` key is NULL go in a sorted result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullOrdering {
    First,
    Last,
}

impl NullOrdering {
    /// The placement used when a query names no `NULLS FIRST` or `NULLS LAST`. As in
    /// PostgreSQL, NULL sorts above every value: last ascending, first descending.
    pub fn default_for(direction: SortDirection) -> Self {
        match direction {
            SortDirection::Ascending => Self::Last,
            SortDirection::Descending => Self::First,
        }
    }
}

/// One `ORDER BY` key: the expression to sort on, which way to sort it and where its
/// NULLs go.
#[derive(Clone)]
pub struct SortExpr {
    pub expr: Expression,
    pub direction: SortDirection,
    pub nulls: NullOrdering,
}

impl SortExpr {
    /// A key sorted in `direction` with NULLs placed by [`NullOrdering::default_for`].
    pub fn new(expr: Expression, direction: SortDirection) -> Self {
        Self {
            expr,
            direction,
            nulls: NullOrdering::default_for(direction),
        }
    }

    pub fn ascending(expr: Expression) -> Self {
        Self::new(expr, SortDirection::Ascending)
    }

    pub fn descending(expr: Expression) -> Self {
        Self::new(expr, SortDirection::Descending)
    }

    /// This key with its NULLs placed at `nulls` instead of the direction's default.
    pub fn with_nulls(self, nulls: NullOrdering) -> Self {
        Self { nulls, ..self }
    }
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
    pub key_columns: Vec<ColIdx>,
}

pub struct DropIndexPlanNode {
    pub index_oid: u32,
    pub if_exists: bool,
}

/// Position of a column in the schema of the input an expression is evaluated against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColIdx(pub u32);

impl ColIdx {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Which input of a join a column comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// A column of one input of a join, as read by a join predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoinColumn {
    pub side: Side,
    pub col: ColIdx,
}

/// A predicate over the left and right input rows of a join.
pub type JoinPredicate = Expression<JoinColumn>;

/// A scalar expression whose column references are `C`: a [`ColIdx`] into one input row, or a
/// [`JoinColumn`] inside a [`JoinPredicate`].
#[derive(Clone, Debug)]
pub enum Expression<C = ColIdx> {
    Constant(Option<Value>),
    Column {
        col: C,
        dtype: DataType,
    },
    Binary {
        left: Box<Expression<C>>,
        right: Box<Expression<C>>,
        op: Op,
    },
}

/// The values an expression's column references `C` read from.
pub trait ColumnSource<C> {
    fn column_value(&self, col: C) -> Result<Option<Value>, ExecutorError>;
}

/// One input row and the schema it is encoded with.
#[derive(Clone, Copy)]
pub struct InputRow<'a> {
    pub tuple: &'a Tuple,
    pub schema: &'a Schema,
}

impl ColumnSource<ColIdx> for InputRow<'_> {
    fn column_value(&self, col: ColIdx) -> Result<Option<Value>, ExecutorError> {
        self.schema
            .get_value(self.tuple, col.index())
            .map_err(|e| ExecutorError(format!("cannot read column {}: {e:?}", col.0)))
    }
}

/// The left and right input rows a join predicate is evaluated against.
#[derive(Clone, Copy)]
pub struct JoinRow<'a> {
    pub left: InputRow<'a>,
    pub right: InputRow<'a>,
}

impl ColumnSource<JoinColumn> for JoinRow<'_> {
    fn column_value(&self, col: JoinColumn) -> Result<Option<Value>, ExecutorError> {
        match col.side {
            Side::Left => self.left.column_value(col.col),
            Side::Right => self.right.column_value(col.col),
        }
    }
}

impl<C: Copy> Expression<C> {
    /// The type this expression evaluates to, or `None` for a NULL constant.
    pub fn dtype(&self) -> Option<DataType> {
        match self {
            Expression::Column { dtype, .. } => Some(*dtype),
            Expression::Constant(Some(v)) => Some(v.data_type()),
            Expression::Constant(None) => None,
            Expression::Binary { left, right, op } => match op {
                Op::Add | Op::Sub | Op::Mul | Op::Div => match (left.dtype(), right.dtype()) {
                    (Some(DataType::FLOAT), _) | (_, Some(DataType::FLOAT)) => {
                        Some(DataType::FLOAT)
                    }
                    (l, r) => l.or(r),
                },
                _ => Some(DataType::BOOLEAN),
            },
        }
    }

    /// Evaluates the expression, reading each column reference from `row`.
    pub fn evaluate_on(&self, row: &impl ColumnSource<C>) -> Result<Option<Value>, ExecutorError> {
        match self {
            Expression::Binary { left, right, op } => {
                let l = left.evaluate_on(row)?;
                let r = right.evaluate_on(row)?;
                eval_binary(op, l, r)
            }
            Expression::Column { col, .. } => row.column_value(*col),
            Expression::Constant(v) => Ok(*v),
        }
    }

    /// The same expression with every column reference replaced by `f(col)`.
    pub fn try_map_columns<D, E>(self, f: &impl Fn(C) -> Result<D, E>) -> Result<Expression<D>, E> {
        Ok(match self {
            Expression::Constant(v) => Expression::Constant(v),
            Expression::Column { col, dtype } => Expression::Column {
                col: f(col)?,
                dtype,
            },
            Expression::Binary { left, right, op } => Expression::Binary {
                left: Box::new(left.try_map_columns(f)?),
                right: Box::new(right.try_map_columns(f)?),
                op,
            },
        })
    }

    /// The same expression with every column reference replaced by `f(col)`.
    pub fn map_columns<D>(self, f: impl Fn(C) -> D) -> Expression<D> {
        match self.try_map_columns(&|col| Ok::<D, Infallible>(f(col))) {
            Ok(mapped) => mapped,
            Err(infallible) => match infallible {},
        }
    }
}

impl Expression {
    /// Evaluates the expression against one input row encoded with `schema`.
    pub fn evaluate(&self, tuple: &Tuple, schema: &Schema) -> Result<Option<Value>, ExecutorError> {
        self.evaluate_on(&InputRow { tuple, schema })
    }
}

fn eval_binary(
    op: &Op,
    l: Option<Value>,
    r: Option<Value>,
) -> Result<Option<Value>, ExecutorError> {
    let cmp = |holds: fn(Ordering) -> bool| -> Result<Option<Value>, ExecutorError> {
        Ok(compare(l, r)?.map(|ord| Value::BOOLEAN(holds(ord))))
    };
    let arith = |op: Arithmetic| -> Result<Option<Value>, ExecutorError> {
        match (l, r) {
            (Some(l), Some(r)) => op.apply(l, r).map(Some),
            _ => Ok(None),
        }
    };
    match op {
        Op::Add => arith(Arithmetic::Add),
        Op::Sub => arith(Arithmetic::Sub),
        Op::Mul => arith(Arithmetic::Mul),
        Op::Div => arith(Arithmetic::Div),
        Op::Eq => cmp(Ordering::is_eq),
        Op::NEq => cmp(Ordering::is_ne),
        Op::Lt => cmp(Ordering::is_lt),
        Op::Gt => cmp(Ordering::is_gt),
        Op::Lte => cmp(Ordering::is_le),
        Op::Gte => cmp(Ordering::is_ge),
        Op::And => match (as_bool(l)?, as_bool(r)?) {
            (Some(false), _) | (_, Some(false)) => Ok(Some(Value::BOOLEAN(false))),
            (Some(true), Some(true)) => Ok(Some(Value::BOOLEAN(true))),
            _ => Ok(None),
        },
        Op::Or => match (as_bool(l)?, as_bool(r)?) {
            (Some(true), _) | (_, Some(true)) => Ok(Some(Value::BOOLEAN(true))),
            (Some(false), Some(false)) => Ok(Some(Value::BOOLEAN(false))),
            _ => Ok(None),
        },
    }
}

fn as_bool(v: Option<Value>) -> Result<Option<bool>, ExecutorError> {
    match v {
        None => Ok(None),
        Some(Value::BOOLEAN(b)) => Ok(Some(b)),
        Some(other) => Err(ExecutorError(format!(
            "expected a BOOLEAN operand, got {other:?}"
        ))),
    }
}

/// Orders two values of the same type, comparing INT with FLOAT numerically.
/// `None` when either side is NULL.
fn compare(l: Option<Value>, r: Option<Value>) -> Result<Option<Ordering>, ExecutorError> {
    let (Some(l), Some(r)) = (l, r) else {
        return Ok(None);
    };
    let ord = match (l, r) {
        (Value::INT(a), Value::FLOAT(b)) => OrderedFloat(a as f32).cmp(&b),
        (Value::FLOAT(a), Value::INT(b)) => a.cmp(&OrderedFloat(b as f32)),
        (a, b) if a.data_type() == b.data_type() => a.cmp(&b),
        (a, b) => {
            return Err(ExecutorError(format!("cannot compare {a:?} with {b:?}")));
        }
    };
    Ok(Some(ord))
}

/// Arithmetic on INT and FLOAT values. INT with INT stays INT and errors on overflow or
/// division by zero; any FLOAT operand makes the result FLOAT.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Arithmetic {
    Add,
    Sub,
    Mul,
    Div,
}

impl Arithmetic {
    pub(crate) fn apply(self, l: Value, r: Value) -> Result<Value, ExecutorError> {
        match (l, r) {
            (Value::INT(a), Value::INT(b)) => self.apply_int(a, b).map(Value::INT),
            (Value::INT(a), Value::FLOAT(b)) => {
                Ok(Value::FLOAT(self.apply_float(a as f32, *b).into()))
            }
            (Value::FLOAT(a), Value::INT(b)) => {
                Ok(Value::FLOAT(self.apply_float(*a, b as f32).into()))
            }
            (Value::FLOAT(a), Value::FLOAT(b)) => Ok(Value::FLOAT(self.apply_float(*a, *b).into())),
            (a, b) => Err(ExecutorError(format!(
                "cannot apply {self:?} to {a:?} and {b:?}"
            ))),
        }
    }

    fn apply_int(self, a: i32, b: i32) -> Result<i32, ExecutorError> {
        let result = match self {
            Arithmetic::Add => a.checked_add(b),
            Arithmetic::Sub => a.checked_sub(b),
            Arithmetic::Mul => a.checked_mul(b),
            Arithmetic::Div if b == 0 => {
                return Err(ExecutorError(format!("division by zero: {a} / 0")));
            }
            Arithmetic::Div => a.checked_div(b),
        };
        result.ok_or_else(|| ExecutorError(format!("integer overflow: {self:?} of {a} and {b}")))
    }

    fn apply_float(self, a: f32, b: f32) -> f32 {
        match self {
            Arithmetic::Add => a + b,
            Arithmetic::Sub => a - b,
            Arithmetic::Mul => a * b,
            Arithmetic::Div => a / b,
        }
    }
}

#[derive(Clone, Debug)]
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

    fn int(v: i32) -> Expression {
        constant(Some(Value::INT(v)))
    }

    fn float(v: f32) -> Expression {
        constant(Some(Value::FLOAT(v.into())))
    }

    #[test]
    fn comparing_int_with_float_compares_numerically() {
        let f = Some(Value::BOOLEAN(false));
        let t = Some(Value::BOOLEAN(true));
        assert_eq!(eval_ok(float(1.5), Op::Gt, int(2)), f);
        assert_eq!(eval_ok(int(2), Op::Lt, float(1.5)), f);
        assert_eq!(eval_ok(int(2), Op::Eq, float(2.0)), t);
        assert_eq!(eval_ok(float(2.5), Op::Gte, int(2)), t);
    }

    #[test]
    fn arithmetic_mixing_int_and_float_yields_float() {
        assert_eq!(
            eval_ok(int(1), Op::Add, float(0.5)),
            Some(Value::FLOAT(1.5.into()))
        );
        assert_eq!(
            eval_ok(float(3.0), Op::Div, int(2)),
            Some(Value::FLOAT(1.5.into()))
        );
        assert_eq!(eval_ok(int(7), Op::Div, int(2)), Some(Value::INT(3)));
    }

    #[test]
    fn dtype_of_mixed_int_and_float_arithmetic_is_float() {
        let add = |l: Expression, r: Expression| Expression::Binary {
            left: Box::new(l),
            right: Box::new(r),
            op: Op::Add,
        };
        assert_eq!(add(int(1), float(0.5)).dtype(), Some(DataType::FLOAT));
        assert_eq!(add(float(0.5), int(1)).dtype(), Some(DataType::FLOAT));
        assert_eq!(add(int(1), int(1)).dtype(), Some(DataType::INT));
        assert_eq!(add(null(), int(1)).dtype(), Some(DataType::INT));
    }

    #[test]
    fn operands_of_incompatible_types_are_an_error() {
        assert!(eval(int(1), Op::Add, boolean(true)).is_err());
        assert!(eval(int(1), Op::Lt, boolean(true)).is_err());
        assert!(eval(boolean(true), Op::And, int(1)).is_err());
        assert!(eval(boolean(false), Op::And, int(1)).is_err());
        assert!(eval(int(1), Op::Or, null()).is_err());
        assert!(eval(boolean(true), Op::Mul, boolean(true)).is_err());
        let ts = || constant(Some(Value::TIMESTAMP(5)));
        assert!(eval(ts(), Op::Add, ts()).is_err());
        assert!(eval(ts(), Op::Eq, int(5)).is_err());
        assert_eq!(eval_ok(ts(), Op::Lte, ts()), Some(Value::BOOLEAN(true)));
    }

    #[test]
    fn integer_division_by_zero_is_an_error() {
        let err = eval(int(1), Op::Div, int(0)).unwrap_err();
        assert!(err.0.contains("division by zero"), "{}", err.0);
    }

    #[test]
    fn integer_overflow_is_an_error() {
        assert!(eval(int(i32::MAX), Op::Add, int(1)).is_err());
        assert!(eval(int(i32::MIN), Op::Sub, int(1)).is_err());
        assert!(eval(int(i32::MAX), Op::Mul, int(2)).is_err());
        assert!(eval(int(i32::MIN), Op::Div, int(-1)).is_err());
    }

    fn int_column(col: u32) -> Expression {
        Expression::Column {
            col: ColIdx(col),
            dtype: DataType::INT,
        }
    }

    #[test]
    fn join_predicate_reads_each_side_from_its_own_row() {
        let schema = Schema::new(vec![
            (DataType::INT, "a".to_string()),
            (DataType::INT, "b".to_string()),
        ]);
        let left = schema
            .encode_tuple(&[Some(Value::INT(1)), Some(Value::INT(2))])
            .unwrap();
        let right = schema
            .encode_tuple(&[Some(Value::INT(10)), Some(Value::INT(20))])
            .unwrap();
        let row = JoinRow {
            left: InputRow {
                tuple: &left,
                schema: &schema,
            },
            right: InputRow {
                tuple: &right,
                schema: &schema,
            },
        };
        let column = |side, col| JoinPredicate::Column {
            col: JoinColumn {
                side,
                col: ColIdx(col),
            },
            dtype: DataType::INT,
        };
        assert_eq!(
            column(Side::Left, 1).evaluate_on(&row).unwrap(),
            Some(Value::INT(2))
        );
        assert_eq!(
            column(Side::Right, 1).evaluate_on(&row).unwrap(),
            Some(Value::INT(20))
        );
    }

    #[test]
    fn column_past_the_end_of_the_input_is_an_error() {
        let schema = Schema::new(vec![(DataType::INT, "a".to_string())]);
        let tuple = schema.encode_tuple(&[Some(Value::INT(1))]).unwrap();
        assert!(int_column(1).evaluate(&tuple, &schema).is_err());
        assert!(
            int_column(0)
                .evaluate(&Tuple { data: vec![] }, &schema)
                .is_err(),
            "a column read from an empty input row is an error"
        );
    }

    #[test]
    fn mapping_columns_keeps_the_expression_shape() {
        let sum = Expression::Binary {
            left: Box::new(int_column(0)),
            right: Box::new(int_column(3)),
            op: Op::Add,
        };
        let mapped: Expression<u32> = sum
            .try_map_columns(&|c: ColIdx| Ok::<_, ()>(c.0 * 10))
            .unwrap();
        let Expression::Binary { left, right, .. } = mapped else {
            panic!("expected a binary expression");
        };
        assert!(matches!(*left, Expression::Column { col: 0, .. }));
        assert!(matches!(*right, Expression::Column { col: 30, .. }));
    }
}
