use crate::catalog::{DataType, Schema, Value};
use crate::processor::executor::{DEFAULT_BATCH_SIZE, Executor, collect_rows};
use crate::processor::executors::{ValuesExecutor, row_values};
use crate::processor::plan::{ColIdx, Expression, Plan, ValuesPlanNode};
use crate::table_heap::Tuple;

pub(crate) fn int_schema(names: &[&str]) -> Schema {
    Schema::new(
        names
            .iter()
            .map(|n| (DataType::INT, n.to_string()))
            .collect(),
    )
}

pub(crate) fn col(col_idx: u32) -> Expression {
    Expression::Column {
        col: ColIdx(col_idx),
        dtype: DataType::INT,
    }
}

pub(crate) fn int_rows(schema: &Schema, rows: &[&[i32]]) -> Vec<Tuple> {
    rows.iter()
        .map(|r| {
            let vals: Vec<Option<Value>> = r.iter().map(|v| Some(Value::INT(*v))).collect();
            schema.encode_tuple(&vals).expect("encode")
        })
        .collect()
}

pub(crate) fn decode_ints(
    schema: &Schema,
    tuples: impl IntoIterator<Item = Tuple>,
) -> Vec<Vec<i32>> {
    tuples
        .into_iter()
        .map(|t| {
            row_values(schema, &t)
                .expect("decode")
                .into_iter()
                .map(|v| match v {
                    Some(Value::INT(i)) => i,
                    other => panic!("expected INT, got {other:?}"),
                })
                .collect()
        })
        .collect()
}

pub(crate) fn values_plan(schema: &Schema, rows: &[&[i32]]) -> Plan {
    Plan::Values(ValuesPlanNode {
        schema: schema.clone(),
        rows: rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|v| Expression::Constant(Some(Value::INT(*v))))
                    .collect()
            })
            .collect(),
    })
}

pub(crate) fn values_child<'a>(plan: &'a Plan) -> Box<dyn Executor + 'a> {
    let Plan::Values(values) = plan else {
        panic!("expected a Values plan");
    };
    Box::new(ValuesExecutor::new(values, DEFAULT_BATCH_SIZE))
}

pub(crate) fn drain(exec: &mut dyn Executor) -> Vec<Tuple> {
    exec.open().expect("open");
    collect_rows(exec)
        .expect("next_batch")
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

pub(crate) fn nullable_int_rows(schema: &Schema, rows: &[&[Option<i32>]]) -> Vec<Tuple> {
    rows.iter()
        .map(|r| {
            let vals: Vec<Option<Value>> = r.iter().map(|v| v.map(Value::INT)).collect();
            schema.encode_tuple(&vals).expect("encode")
        })
        .collect()
}

pub(crate) fn decode_nullable_ints(
    schema: &Schema,
    tuples: impl IntoIterator<Item = Tuple>,
) -> Vec<Option<i32>> {
    tuples
        .into_iter()
        .map(|t| match row_values(schema, &t).expect("decode")[..] {
            [Some(Value::INT(i))] => Some(i),
            [None] => None,
            ref other => panic!("expected one INT or NULL, got {other:?}"),
        })
        .collect()
}

pub(crate) fn divide_by_zero(col_idx: u32) -> Expression {
    Expression::Binary {
        left: Box::new(col(col_idx)),
        right: Box::new(Expression::Constant(Some(Value::INT(0)))),
        op: crate::processor::plan::Op::Div,
    }
}
