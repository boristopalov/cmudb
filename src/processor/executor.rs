use crate::buffer_pool::BufferPoolManager;
use crate::catalog::index_schema::{IndexKey, IndexValue};
use crate::catalog::{Catalog, CatalogError, IndexInfo, Schema, TableInfo, Value};
use crate::index::IndexError;
use crate::processor::plan::{
    AggType, AggregationPlanNode, DeletePlanNode, Expression, ExternalMergeSortPlanNode,
    HashJoinPlanNode, IndexScanPlanNode, InsertPlanNode, JoinType, LimitPlanNode,
    NestedIndexJoinPlanNode, NestedLoopJoinPlanNode, SeqScanPlanNode, SortPlanNode, UpdatePlanNode,
    ValuesPlanNode, WindowFunctionPlanNode,
};
use crate::table_heap::{RecordId, TableHeap, TableHeapError, TableHeapIterator, Tuple};
use std::cmp::Ordering;
use std::collections::hash_map::IntoIter;
use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Bound;
use std::sync::Arc;
use std::vec::IntoIter as VecIter;

#[derive(Debug)]
pub struct ExecutorError(pub String);

pub struct ExecutorContext<'a> {
    pub catalog: &'a Catalog,
}

impl From<CatalogError> for ExecutorError {
    fn from(e: CatalogError) -> Self {
        ExecutorError(format!("catalog error: {:?}", e))
    }
}

impl From<TableHeapError> for ExecutorError {
    fn from(e: TableHeapError) -> Self {
        ExecutorError(format!("table heap error: {:?}", e))
    }
}

impl From<IndexError> for ExecutorError {
    fn from(e: IndexError) -> Self {
        ExecutorError(format!("index error: {:?}", e))
    }
}

/// Executors in the executor tree can be:
/// - leaves, meaning they produce values
/// - unary, meaning they pull from child to get values to mutate
/// - binary (joins)
/// in many cases, close() doesn't do anything; the executor simply going out of scope is enough for rust to clean things up for us
/// the next() method returns only 1 tuple at a time, with no batching, for simplicity.
pub trait Executor {
    fn open(&mut self) -> Result<(), ExecutorError>;
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError>;
    fn schema(&self) -> &Schema; // output schema
    fn close(&mut self) -> Result<(), ExecutorError>;
}

pub struct SeqScanExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a SeqScanPlanNode,
    iter: Option<TableHeapIterator>,
}

impl<'a> SeqScanExecutor<'a> {
    pub fn new(ctx: &'a ExecutorContext<'a>, plan: &'a SeqScanPlanNode) -> Self {
        Self {
            ctx,
            plan,
            iter: None,
        }
    }
}

impl<'a> Executor for SeqScanExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.iter = Some(table.heap.iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self
            .iter
            .as_mut()
            .ok_or_else(|| ExecutorError("SeqScanExecutor::next called before open".to_string()))?;
        iter.next().transpose().map_err(ExecutorError::from)
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct InsertExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a InsertPlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
    done: bool,
}

impl<'a> InsertExecutor<'a> {
    pub fn new(
        ctx: &'a ExecutorContext<'a>,
        plan: &'a InsertPlanNode,
        child: Box<dyn Executor + 'a>,
    ) -> Self {
        Self {
            ctx,
            plan,
            child,
            table: None,
            done: false,
        }
    }
}

impl<'a> Executor for InsertExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.done {
            return Err(ExecutorError("already executed".to_string()));
        }

        // Here we call child.next() in a loop, in case there are multiple values to insert
        // Recall that each child.next() call returns a single (tuple, record_id)
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| ExecutorError("InsertExecutor::next called before open".to_string()))?;

        let mut count: i32 = 0;
        while let Some(res) = self.child.next()? {
            let rid = table.heap.insert(&res.0)?;
            for idx in self.ctx.catalog.get_table_indexes(self.plan.table_oid)? {
                idx.as_ref().index.insert(&res.0, rid)?;
            }
            count += 1;
        }

        self.done = true;

        // This is a synthetic tuple, we need to tell our parent how many items we inserted
        Ok(Some((
            Tuple {
                data: count.to_ne_bytes().to_vec(),
            },
            RecordId::RESERVED,
        )))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

/// Retrieves tuples, one by one, from a list of Vec<Expressions>.
/// corresponds to the VALUES sql keyword.
pub struct ValuesExecutor<'a> {
    plan: &'a ValuesPlanNode,
    cursor: u32,
}

impl<'a> ValuesExecutor<'a> {
    pub fn new(plan: &'a ValuesPlanNode) -> Self {
        Self { plan, cursor: 0 }
    }
}

impl<'a> Executor for ValuesExecutor<'a> {
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        if self.cursor as usize >= self.plan.rows.len() {
            return Ok(None);
        }
        let exprs = &self.plan.rows[self.cursor as usize];

        // exprs represents a single tuple
        // each expr repsents the value in that particular column in the tuple at exprs.index_of(expr)
        let dummy = Tuple { data: vec![] };
        let vals: Vec<Option<Value>> = exprs
            .iter()
            .map(|e| e.evaluate(&dummy, self.schema()))
            .collect::<Result<Vec<Option<Value>>, ExecutorError>>()?;

        let tuple = self.schema().encode_tuple(&vals)?;
        self.cursor += 1;
        Ok(Some((tuple, RecordId::RESERVED)))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct UpdateExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a UpdatePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> Executor for UpdateExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| ExecutorError("UpdateExecutor::next called before open".to_string()))?;

        let Some((tuple_to_update, rid)) = self.child.next()? else {
            return Ok(None);
        };

        // we need to evaluate each expression
        // target_exprs can have some or all columns
        // so we need to update the values in those columns, and keep the values of the old columns,
        // and then return the updated tuple
        // First, populate new_values with the old values
        let mut new_values: Vec<Option<Value>> = Vec::with_capacity(self.schema().cols.len());
        for i in 0..self.schema().cols.len() {
            new_values.push(self.schema().get_value(&tuple_to_update, i)?);
        }

        // update new_values
        for expr in self.plan.target_exprs.iter() {
            let col_to_update_idx = expr.0;
            let new_val = expr.1.evaluate(&tuple_to_update, self.schema())?;
            new_values[col_to_update_idx as usize] = new_val;
        }

        let new_tuple = self.schema().encode_tuple(new_values.as_ref())?;
        let (new_rid, _) = table.heap.update(rid, &new_tuple)?;

        // to update the index we remove first, and then re-insert
        // not ideal :/
        for idx in self.ctx.catalog.get_table_indexes(self.plan.table_oid)? {
            idx.as_ref().index.remove(&tuple_to_update)?;
            idx.as_ref().index.insert(&new_tuple, new_rid)?;
        }
        Ok(Some((new_tuple, new_rid)))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct DeleteExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a DeletePlanNode,
    child: Box<dyn Executor + 'a>,
    table: Option<Arc<TableInfo>>,
}

impl<'a> Executor for DeleteExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.table = Some(table);
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| ExecutorError("DeleteExecutor::next called before open".to_string()))?;

        let Some((tup, rid)) = self.child.next()? else {
            return Ok(None);
        };

        table.heap.delete(&rid)?;
        for idx in self.ctx.catalog.get_table_indexes(self.plan.table_oid)? {
            idx.as_ref().index.remove(&tup)?;
        }
        Ok(Some((tup, rid)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct IndexScanExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a IndexScanPlanNode,
    table: Option<Arc<TableInfo>>,
    iter: Option<Box<dyn Iterator<Item = Result<(IndexKey, IndexValue), IndexError>>>>,
}

impl<'a> Executor for IndexScanExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let info = self.ctx.catalog.get_index(self.plan.index_oid)?;

        // for now we don't support NULLS when scanning indexes
        let iter = info.as_ref().index.scan(
            self.plan.start.as_ref().map(Vec::as_slice),
            self.plan.end.as_ref().map(Vec::as_slice),
        )?;
        self.iter = Some(iter);

        let table = self.ctx.catalog.get_table(info.as_ref().table_oid)?;
        self.table = Some(table);

        Ok(())
    }
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let iter = self.iter.as_mut().ok_or_else(|| {
            ExecutorError("IndexScanExecutor::next called before open".to_string())
        })?;
        let table = self.table.as_ref().ok_or_else(|| {
            ExecutorError("IndexScanExecutor::next called before open".to_string())
        })?;

        let Some(item) = iter.next() else {
            return Ok(None);
        };
        let (_key, IndexValue::IndexValue(rid)) = item?;

        let (rid, tuple) = table.heap.get(rid)?;
        Ok(Some((tuple, rid)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct AggregationExecutor<'a> {
    plan: &'a AggregationPlanNode,
    child: Box<dyn Executor + 'a>,
    iter: Option<IntoIter<ExprKey, AggVal>>,
}

impl<'a> Executor for AggregationExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let mut agg_hash_table: HashMap<ExprKey, AggVal> = HashMap::new();

        // populate the hash table
        // this executor is a pipeline breaker, meaning we need to
        // drain the child completely before we do any work
        while let Some((tuple, _)) = self.child.next()? {
            let schema = self.schema();
            let groups: Vec<Option<Value>> = self
                .plan
                .group_bys
                .iter()
                .map(|expr| expr.evaluate(&tuple, schema).ok()?)
                .collect();
            let hashkey = ExprKey(groups);

            // get the entry, if it exists
            // otherwise, round up all of the aggregates to get a default
            let entry = agg_hash_table.entry(hashkey).or_insert_with(|| {
                AggVal(
                    self.plan
                        .aggregates
                        .iter()
                        .map(|(_, agg_type)| AggState::new(agg_type))
                        .collect(),
                )
            });

            // compute the updated value by, once again, iterating through the aggregates,
            // this time evaluating the actual expressions
            for (state, (expr, _)) in entry.0.iter_mut().zip(self.plan.aggregates.iter()) {
                let v = expr.evaluate(&tuple, schema)?;
                state.update(v);
            }
        }
        self.iter = Some(agg_hash_table.into_iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        // here, next iterates through the hash table
        // each entry in the hash table corresponds to one tuple that we return to our parent
        let Some(iter) = self.iter.as_mut() else {
            return Err(ExecutorError(
                "AggregationExecutor::next called before open".to_string(),
            ));
        };

        let Some((key, val)) = iter.next() else {
            return Ok(None);
        };

        // ok so now we need to zip up key+val into a tuple
        // output tuple has columns for each group by + column for each aggregate
        let mut out: Vec<Option<Value>> = Vec::with_capacity(key.0.len() + val.0.len());
        out.extend(key.0);
        for v in val.0 {
            out.push(v.consume());
        }

        let tuple = self.schema().encode_tuple(&out)?;
        Ok(Some((tuple, RecordId::RESERVED)))
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

#[derive(Eq, PartialEq, Hash)]
struct ExprKey(Vec<Option<Value>>);

struct AggVal(Vec<AggState>);

enum AggState {
    Max(Option<Value>),
    Min(Option<Value>),
    Sum(Option<Value>),
    Avg { sum: Option<Value>, count: i32 },
    Count(i32),
    Rank(i32), // not an aggregate but whatever
}

impl AggState {
    fn new(agg_type: &AggType) -> Self {
        match agg_type {
            AggType::AVG => Self::Avg {
                sum: None,
                count: 0,
            },
            AggType::MAX => Self::Max(None),
            AggType::MIN => Self::Min(None),
            AggType::SUM => Self::Sum(None),
            AggType::COUNT => Self::Count(0),
            AggType::RANK => Self::Rank(1),
        }
    }

    fn update_rank(&mut self, order_changed: bool, pos: i32) {
        match self {
            Self::Rank(r) => {
                if order_changed {
                    *r = pos;
                }
            }
            _ => unreachable!("use update() instead"),
        }
    }

    fn update(&mut self, v: Option<Value>) {
        if let Self::Count(count) = self {
            *count += 1;
            return;
        }

        // skip nulls
        let Some(v) = v else { return };

        match self {
            Self::Sum(current) => {
                *current = Some(current.take().map_or(v, |c| c + v));
            }
            Self::Avg { sum, count } => {
                *sum = Some(sum.take().map_or(v, |s| s + v));
                *count += 1;
            }
            Self::Max(current) => {
                *current = Some(current.take().map_or(v, |c| c.max(v)));
            }
            Self::Min(current) => {
                *current = Some(current.take().map_or(v, |c| c.min(v)));
            }
            Self::Count(_) => unreachable!("blah"),
            Self::Rank(_) => unreachable!("use update_rank() instead"),
        }
    }

    fn consume(self) -> Option<Value> {
        match self {
            Self::Sum(v) | Self::Max(v) | Self::Min(v) => v,
            Self::Count(c) => Some(Value::INT(c)),
            Self::Avg { sum, count } => {
                if count == 0 {
                    None
                } else {
                    sum.map(|s| s / Value::INT(count))
                }
            }
            Self::Rank(r) => Some(Value::INT(r)),
        }
    }

    fn peek(&self) -> Option<Value> {
        match self {
            Self::Sum(v) | Self::Max(v) | Self::Min(v) => v.clone(),
            Self::Count(c) => Some(Value::INT(*c)),
            Self::Avg { sum, count } => {
                if *count == 0 {
                    None
                } else {
                    sum.clone().map(|s| s / Value::INT(*count))
                }
            }
            Self::Rank(r) => Some(Value::INT(*r)),
        }
    }
}

pub struct NestedLoopJoinExecutor<'a> {
    plan: &'a NestedLoopJoinPlanNode,
    left: Box<dyn Executor>,
    right: Box<dyn Executor>,

    right_tuples: Vec<Tuple>,

    right_cursor: usize,
    curr_left: Option<Tuple>,
    left_matched: bool,
}

impl<'a> Executor for NestedLoopJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        while let Some((tup, _)) = self.right.next()? {
            self.right_tuples.push(tup);
        }
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        loop {
            if self.curr_left.is_none() {
                let Some((lt, _)) = self.left.next()? else {
                    return Ok(None);
                };
                self.curr_left = Some(lt);

                // reset state
                // reset back to 0 so we can iterate through right tuples again
                self.right_cursor = 0;
                self.left_matched = false;
            };

            let lt = self.curr_left.as_ref().unwrap();

            while self.right_cursor < self.right_tuples.len() {
                let rt = &self.right_tuples[self.right_cursor];
                self.right_cursor += 1;
                let val = self.plan.predicate.evaluate_join(
                    &lt,
                    self.left.schema(),
                    &rt,
                    self.right.schema(),
                )?;

                // predicate must always evaluate to a boolean
                if matches!(val, Some(Value::BOOLEAN(true))) {
                    self.left_matched = true;
                    // build the combined tuple
                    let mut combined: Vec<Option<Value>> = Vec::with_capacity(
                        self.left.schema().cols.len() + self.right.schema().cols.len(),
                    );

                    for i in 0..self.left.schema().cols.len() {
                        combined.push(self.left.schema().get_value(&lt, i)?);
                    }
                    for i in 0..self.right.schema().cols.len() {
                        combined.push(self.right.schema().get_value(&rt, i)?);
                    }

                    let out = self.schema().encode_tuple(&combined)?;
                    return Ok(Some((out, RecordId::RESERVED)));
                }
            }

            // if this is a left outer join, and there is no match,
            // we still need to make sure to return the left tuple.
            // since there is no match we just populate the "right" tuple with None's
            if matches!(self.plan.join_type, JoinType::Left) && !self.left_matched {
                let lt = self.curr_left.as_ref().unwrap();
                let mut combined = Vec::with_capacity(
                    self.left.schema().cols.len() + self.right.schema().cols.len(),
                );
                for i in 0..self.left.schema().cols.len() {
                    combined.push(self.left.schema().get_value(lt, i)?);
                }
                for _ in 0..self.right.schema().cols.len() {
                    combined.push(None);
                }
                let out = self.schema().encode_tuple(&combined)?;
                self.curr_left = None;
                return Ok(Some((out, RecordId::RESERVED)));
            }
            self.curr_left = None;
        }
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct NestedIndexJoinExecutor<'a> {
    ctx: &'a ExecutorContext<'a>,
    plan: &'a NestedIndexJoinPlanNode,
    child: Box<dyn Executor>,
    inner_table: Option<Arc<TableInfo>>,

    index: Option<Arc<IndexInfo>>,
    // since duplicate keys can be inserted into the index, instead of doing index.search(),
    // we need to do index.scan(), using the same value as both the lower and upper bounds, so that
    // we can scan for all index keys that are equal to the given key, instead of returning the first match like search()
    index_iter: Option<Box<dyn Iterator<Item = Result<(IndexKey, IndexValue), IndexError>>>>,
    curr_left: Option<Tuple>,
    left_matched: bool,
}

impl<'a> Executor for NestedIndexJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let info = self.ctx.catalog.get_index(self.plan.index_oid)?;
        self.index = Some(info);

        let table = self.ctx.catalog.get_table(self.plan.table_oid)?;
        self.inner_table = Some(table);
        Ok(())
    }
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let Some(index) = self.index.as_ref() else {
            return Err(ExecutorError(
                "NestedIndexJoinExecutor::next called before open".to_string(),
            ));
        };

        let Some(table) = self.inner_table.as_ref() else {
            return Err(ExecutorError(
                "NestedIndexJoinExecutor::next called before open".to_string(),
            ));
        };

        loop {
            // very similar to NestedLoopJoinExecutor
            // basically we keep some state on the executor,
            // in this case an index iterator, that we use across next() calls.
            // we also track the current outer (left) key, to account for the scenario
            // where that key has multiple matches inside the index.
            if self.curr_left.is_none() {
                let Some((outer, _)) = self.child.next()? else {
                    return Ok(None);
                };
                let key = self
                    .plan
                    .key_predicate
                    .evaluate(&outer, self.child.schema())?;
                self.index_iter = match key {
                    Some(v) => {
                        let probe = [v];
                        Some(
                            index
                                .index
                                .scan(Bound::Included(&probe[..]), Bound::Included(&probe[..]))?,
                        )
                    }
                    None => None,
                };
                self.curr_left = Some(outer);
                self.left_matched = false;
            }

            // try to pull the next inner match for the current outer tuple
            if let Some(iter) = self.index_iter.as_mut() {
                if let Some(entry) = iter.next() {
                    let (_, IndexValue::IndexValue(rid)) = entry?;
                    let (_, inner_tuple) = table.heap.get(rid)?;
                    let outer = self.curr_left.as_ref().unwrap();

                    let mut combined: Vec<Option<Value>> = Vec::with_capacity(
                        self.child.schema().cols.len() + table.schema.cols.len(),
                    );
                    for i in 0..self.child.schema().cols.len() {
                        combined.push(self.child.schema().get_value(outer, i)?);
                    }
                    for i in 0..table.schema.cols.len() {
                        combined.push(table.schema.get_value(&inner_tuple, i)?);
                    }
                    let out = self.schema().encode_tuple(&combined)?;
                    self.left_matched = true;
                    return Ok(Some((out, RecordId::RESERVED)));
                }
            }

            if matches!(self.plan.join_type, JoinType::Left) && !self.left_matched {
                let outer = self.curr_left.as_ref().unwrap();
                let mut combined: Vec<Option<Value>> =
                    Vec::with_capacity(self.child.schema().cols.len() + table.schema.cols.len());
                for i in 0..self.child.schema().cols.len() {
                    combined.push(self.child.schema().get_value(outer, i)?);
                }
                for _ in 0..table.schema.cols.len() {
                    combined.push(None);
                }
                let out = self.schema().encode_tuple(&combined)?;
                self.curr_left = None;
                self.index_iter = None;
                return Ok(Some((out, RecordId::RESERVED)));
            }

            self.curr_left = None;
            self.index_iter = None;
        }
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct HashJoinExecutor<'a> {
    plan: &'a HashJoinPlanNode,
    left: Box<dyn Executor>,
    right: Box<dyn Executor>,
    bpm: Arc<BufferPoolManager>,
    left_partitions: Vec<TableHeap>,
    right_partitions: Vec<TableHeap>,

    right_partition_iter: Option<TableHeapIterator>,
    // we store the hashed tuples, as well as a boolean flag to indicate if the tuple matched with a right tuple
    // at the end of the join, for any tuples that didn't have a match, we emit them
    // we store this in the hash table, because the left tuples are only accessible when we build the left hash table
    // when there is no match, there is no left tuple, so it's not like other joins where we keep track of the current left
    // tuple and emit it.
    left_hash_table: Option<HashMap<ExprKey, Vec<(Tuple, bool)>>>,
    partition_cursor: usize,
    matches_buffer: VecDeque<Tuple>,
}

impl<'a> Executor for HashJoinExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        // here we populate the partition pages
        // in a real database we would have statistics on each table,
        // and would accordingly create N partitions based on those stats,
        // as well as based on a memory budget that we assign to the join.
        // in this case we just pick a constant 64 for N.
        // according to claude it's normal for a single hash join to take 10s or 100s of MB,
        // or even 1GB. For us: let's assume a memory budget of 10000 pages per partition. we won't
        // hold the database to this number, we will just assume it works.
        // that means we assume each indiv. partition will take up to 40mb (4kb * 10000 pages),
        // so the max size of the input must be 64 * 4kb * 10000 = 2.56gb.
        // The input could be a full table, or some filtered output from another child.
        // which is several million rows for a table with ~15 cols with various dtypes.
        // in our databse it would be even more rows because we only support a few basic data types.
        let n = 64;
        for _ in 0..n {
            let lh = TableHeap::new(self.bpm.clone())?;
            self.left_partitions.push(lh);

            let rh = TableHeap::new(self.bpm.clone())?;
            self.right_partitions.push(rh);
        }

        while let Some((lt, _)) = self.left.next()? {
            let key: Vec<Option<Value>> = self
                .plan
                .left_exprs
                .iter()
                .map(|e| e.evaluate(&lt, self.left.schema()))
                .collect::<Result<_, _>>()?;

            let slot = hash_key(key.as_ref()) % n;
            self.left_partitions[slot as usize].insert(&lt)?;
        }

        while let Some((rt, _)) = self.right.next()? {
            let key: Vec<Option<Value>> = self
                .plan
                .right_exprs
                .iter()
                .map(|e| e.evaluate(&rt, self.right.schema()))
                .collect::<Result<_, _>>()?;

            let slot = hash_key(key.as_ref()) % n;
            self.right_partitions[slot as usize].insert(&rt)?;
        }
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        // now that we have partitioned our data, we need ot join it together,
        // one partition at a time. Like in our other joins, we need to be careful
        // to keep track of where we are in each partition
        // at this point we have built our hash table using the left partition's keys
        // now we can iterate over the right partition and check for any matches
        loop {
            if let Some(out) = self.matches_buffer.pop_back() {
                return Ok(Some((out, RecordId::RESERVED)));
            };

            if self.partition_cursor >= self.left_partitions.len() {
                return Ok(None);
            };

            if self.left_hash_table.is_none() {
                // we need to build the hash table for this partition
                // to do so we iterate through whatever partition we are on,
                // and build using the left partition
                let p = &self.left_partitions[self.partition_cursor];
                let mut new_table: HashMap<ExprKey, Vec<(Tuple, bool)>> = HashMap::new();
                for item in p.iter() {
                    let (tup, _) = item?;
                    let key: ExprKey = ExprKey(
                        self.plan
                            .left_exprs
                            .iter()
                            .map(|e| e.evaluate(&tup, self.left.schema()))
                            .collect::<Result<_, _>>()?,
                    );
                    new_table.entry(key).or_default().push((tup, false));
                }
                self.left_hash_table = Some(new_table);
            }

            if self.right_partition_iter.is_none() {
                self.right_partition_iter =
                    Some(self.right_partitions[self.partition_cursor].iter());
            }
            let iter = self.right_partition_iter.as_mut().unwrap();
            let Some(item) = iter.next() else {
                // we are done with this partition,
                // move on to the next partition
                // for a left join, we need to make sure to emit the unmatched left tuples
                if matches!(self.plan.join_type, JoinType::Left) {
                    for entries in self.left_hash_table.as_ref().unwrap().values() {
                        for (lt, matched) in entries {
                            if !matched {
                                let mut combined: Vec<Option<Value>> = Vec::with_capacity(
                                    self.left.schema().cols.len() + self.right.schema().cols.len(),
                                );
                                for i in 0..self.left.schema().cols.len() {
                                    combined.push(self.left.schema().get_value(&lt, i)?);
                                }
                                for _ in 0..self.right.schema().cols.len() {
                                    combined.push(None);
                                }
                                let out = self.schema().encode_tuple(&combined)?;
                                self.matches_buffer.push_back(out);
                            }
                        }
                    }
                }
                self.right_partition_iter = None;
                self.partition_cursor += 1;
                self.left_hash_table = None;
                continue;
            };

            let (rt, _) = item?;
            let key: ExprKey = ExprKey(
                self.plan
                    .right_exprs
                    .iter()
                    .map(|e| e.evaluate(&rt, self.right.schema()))
                    .collect::<Result<_, _>>()?,
            );

            // probe the hash table for this key
            let tuple_matches = self.left_hash_table.as_mut().unwrap().get_mut(&key);
            if tuple_matches.is_some() {
                // we have a match, we need to yield the combined tuples
                // we need to yield them one by one so we store them in a buffer
                let tuples_to_yield = tuple_matches.unwrap();

                for entry in tuples_to_yield {
                    // mark the tuple as matched
                    entry.1 = true;
                    let mut combined: Vec<Option<Value>> = Vec::with_capacity(
                        self.left.schema().cols.len() + self.right.schema().cols.len(),
                    );
                    for i in 0..self.left.schema().cols.len() {
                        combined.push(self.left.schema().get_value(&entry.0, i)?);
                    }
                    for i in 0..self.right.schema().cols.len() {
                        combined.push(self.right.schema().get_value(&rt, i)?);
                    }

                    // borrow checker does not like self.schema() because we already borrow self as mutable,
                    // due to self.left_hash_table.as_mut() call.
                    // self.plan.schema works because rust sees the self.plan is a different field than self.left_hash_table,
                    // while self.schema() borrows the whole self as immutable
                    let out = self.plan.schema.encode_tuple(&combined)?;
                    self.matches_buffer.push_back(out);
                }
                continue;
            }
        }
    }

    /// TODO: delete the pages allocated for the partitions
    /// see tasks/delete_page.md for claude plan
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct SortExecutor<'a> {
    plan: &'a SortPlanNode,
    child: Box<dyn Executor>,
    sorted: VecIter<(Tuple, RecordId)>,
}

impl<'a> Executor for SortExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        let mut buf: Vec<(Tuple, RecordId)> = Vec::new();
        while let Some(row) = self.child.next()? {
            buf.push(row);
        }
        let schema = &self.plan.schema;
        let order_by = &self.plan.order_by_exprs;
        buf.sort_by(|a, b| cmp_tuples(&a.0, &b.0, order_by, schema));
        self.sorted = buf.into_iter();
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        Ok(self.sorted.next())
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct ExternalMergeSortExecutor<'a> {
    plan: &'a ExternalMergeSortPlanNode,
    child: Box<dyn Executor>,
    bpm: Arc<BufferPoolManager>,
    iter: Option<TableHeapIterator>,
}

fn cmp_tuples(a: &Tuple, b: &Tuple, exprs: &[Expression], schema: &Schema) -> Ordering {
    for expr in exprs {
        let va = expr.evaluate(a, schema).ok().flatten();
        let vb = expr.evaluate(b, schema).ok().flatten();
        match va.cmp(&vb) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

impl<'a> Executor for ExternalMergeSortExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        // the idea here is: we define a constant B, which dictates
        // the number of pages we pull into memory at a time,
        // or in other words the size of each run.
        // the number of runs N == len(tuples) / B.
        // we sort these in memory, and then write them out to disk.
        // after we have drained self.child, we recursively merge 2 runs at a time.
        // to do this, we load 1 page from each run into memory, and use TableHeapIterators
        // to iterate over them. At each iteration we take the smaller of the 2 entries and
        // write out that entry to a 3rd output page.
        // when we've exhausted both iterators, we move on to the next two runs
        // after doing this once for, we'll have N/2 sorted runs
        // we repeat this until the number of sorted runs == 1
        let num_pages = 10_000; // 4kb pages * 10,000 = 40mb chunks at a time
        let target_bytes = num_pages * 4096;
        let mut run_bytes: usize = 0;
        let mut buf: Vec<Tuple> = Vec::new();
        let mut runs: Vec<TableHeap> = Vec::new();
        let child_schema = &self.plan.schema;
        let order_by = &self.plan.order_by_exprs;

        while let Some((tup, _)) = self.child.next()? {
            run_bytes += tup.data.len();
            buf.push(tup);
            if run_bytes >= target_bytes {
                buf.sort_by(|a, b| cmp_tuples(a, b, order_by, child_schema));

                // iterate over tuples and insert into table heap
                let th = TableHeap::new(self.bpm.clone())?;
                for t in buf.iter() {
                    th.insert(&t)?;
                }

                buf.clear();
                run_bytes = 0;
                runs.push(th);
            }
        }

        // flush whatever is left in the buffer
        buf.sort_by(|a, b| cmp_tuples(a, b, order_by, child_schema));

        // iterate over tuples and insert into table heap
        let th = TableHeap::new(self.bpm.clone())?;
        for t in buf.iter() {
            th.insert(&t)?;
        }
        runs.push(th);

        // at this point we have all of our runs, so we can start merging
        while runs.len() > 1 {
            let mut next_runs: Vec<TableHeap> = Vec::new();
            for chunk in runs.chunks(2) {
                let out = TableHeap::new(self.bpm.clone())?;
                if chunk.len() == 2 {
                    // merge the two chunks: hold whichever side's current head
                    // wasn't emitted, and pull a new tuple from the side that was.
                    let mut c1 = chunk[0].iter();
                    let mut c2 = chunk[1].iter();
                    let mut head1 = c1.next().transpose()?.map(|(t, _)| t);
                    let mut head2 = c2.next().transpose()?.map(|(t, _)| t);
                    loop {
                        match (&head1, &head2) {
                            (Some(t1), Some(t2)) => {
                                if cmp_tuples(t1, t2, order_by, child_schema) != Ordering::Greater {
                                    out.insert(head1.as_ref().unwrap())?;
                                    head1 = c1.next().transpose()?.map(|(t, _)| t);
                                } else {
                                    out.insert(head2.as_ref().unwrap())?;
                                    head2 = c2.next().transpose()?.map(|(t, _)| t);
                                }
                            }
                            (Some(_), None) => {
                                out.insert(head1.as_ref().unwrap())?;
                                for t in c1.by_ref() {
                                    let (tup, _) = t?;
                                    out.insert(&tup)?;
                                }
                                break;
                            }
                            (None, Some(_)) => {
                                out.insert(head2.as_ref().unwrap())?;
                                for t in c2.by_ref() {
                                    let (tup, _) = t?;
                                    out.insert(&tup)?;
                                }
                                break;
                            }
                            (None, None) => break,
                        }
                    }
                } else {
                    // only 1 chunk
                    let mut c = chunk[0].iter();
                    while let Some(t) = c.next() {
                        let (tup, _) = t?;
                        out.insert(&tup)?;
                    }
                }
                next_runs.push(out);
            }
            runs = next_runs;
        }

        if runs.len() == 0 {
            return Ok(());
        }

        self.iter = Some(runs[0].iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let Some(iter) = self.iter.as_mut() else {
            return Err(ExecutorError(
                "ExternalMergeSortExecutor: next(): ran before open".to_string(),
            ));
        };

        let t = iter.next();
        if t.is_none() {
            return Ok(None);
        }

        let (tup, _) = t.unwrap()?;
        Ok(Some((tup, RecordId::RESERVED)))
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct LimitExecutor<'a> {
    plan: &'a LimitPlanNode,
    child: Box<dyn Executor>,
    count: u32,
}

impl<'a> Executor for LimitExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let Some((tup, rid)) = self.child.next()? else {
            return Ok(None);
        };
        if self.count >= self.plan.limit {
            return Ok(None);
        };
        self.count += 1;
        Ok(Some((tup, rid)))
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct WindowFunctionExecutor<'a> {
    plan: &'a WindowFunctionPlanNode,
    child: Box<dyn Executor>,
    iter: Option<VecIter<Tuple>>,
}

impl<'a> Executor for WindowFunctionExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        // another pipeline breaker
        // we start by draining the child and sorting the tuples
        // sort is based on both the partition expressions and order by expressions
        // if no partition expressions are provided,
        // it's just a single partition with all the tuples
        // need to make sure to sort first
        let mut tups: Vec<Tuple> = vec![];
        while let Some((t, _)) = self.child.next()? {
            tups.push(t);
        }

        let child_schema = self.child.schema();
        let partition_by_exprs = self.plan.partition_by_exprs.iter().flatten();
        let order_by_exprs = self.plan.order_by_exprs.iter().flatten();
        let sort_exprs: Vec<&Expression> = partition_by_exprs.chain(order_by_exprs).collect();

        // sort by the both the order by exprs and the partition exprs
        // pretty sure this will work...
        tups.sort_by(|a, b| {
            for expr in &sort_exprs {
                let va = expr.evaluate(a, child_schema).ok().flatten();
                let vb = expr.evaluate(b, child_schema).ok().flatten();
                match va.cmp(&vb) {
                    Ordering::Equal => continue,
                    ord => return ord,
                }
            }
            return Ordering::Equal;
        });

        // check if we need to keep running aggregates
        // if ORDER BY is present, we basically have two levels of partitions,
        // the first being PARTITION BY expr, the second being ORDER BY exprs
        // so this tells us what level to use when calculating aggregates
        let running = self.plan.order_by_exprs.is_some();

        let mut partition_aggregates: Vec<AggState> = self
            .plan
            .aggregates
            .iter()
            .map(|agg| AggState::new(&agg.1))
            .collect();

        let mut peer_buffer: Vec<Tuple> = Vec::new();
        let mut output_tuples: Vec<Tuple> = Vec::with_capacity(tups.len());

        let mut curr_rank = 1;
        let mut row_num = 1;
        let mut prev_partition_key: Option<ExprKey> = None;
        let mut prev_order_key: Option<ExprKey> = None;

        for t in tups.iter() {
            let curr_partition_key: ExprKey = ExprKey(
                self.plan
                    .partition_by_exprs
                    .iter()
                    .flatten()
                    .map(|e| e.evaluate(t, child_schema))
                    .collect::<Result<_, _>>()?,
            );
            let new_partition = match &prev_partition_key {
                Some(prev) => &curr_partition_key != prev,
                None => true,
            };

            let curr_order_key: ExprKey = ExprKey(
                self.plan
                    .order_by_exprs
                    .iter()
                    .flatten()
                    .map(|e| e.evaluate(t, child_schema))
                    .collect::<Result<_, _>>()?,
            );
            // a partition boundary is also a peer-group boundary
            let new_order = new_partition
                || match &prev_order_key {
                    Some(prev) => &curr_order_key != prev,
                    None => true,
                };

            if new_partition && prev_partition_key.is_some() {
                // flush the trailing peer group of the previous partition with its final state
                flush_peer_group(
                    &mut peer_buffer,
                    &mut partition_aggregates,
                    &self.plan.aggregates,
                    child_schema,
                    &self.plan.schema,
                    &mut output_tuples,
                )?;
                // reset state for the new partition
                partition_aggregates = self
                    .plan
                    .aggregates
                    .iter()
                    .map(|agg| AggState::new(&agg.1))
                    .collect();
                row_num = 1;
            } else if running && new_order && !peer_buffer.is_empty() {
                // peer group changed within the same partition: flush before buffering new peers
                flush_peer_group(
                    &mut peer_buffer,
                    &mut partition_aggregates,
                    &self.plan.aggregates,
                    child_schema,
                    &self.plan.schema,
                    &mut output_tuples,
                )?;
            }

            if new_order {
                curr_rank = row_num;
            }

            // RANK is updated here, since its value is based on row position within partitions
            for (state, (_, agg_type)) in partition_aggregates
                .iter_mut()
                .zip(self.plan.aggregates.iter())
            {
                if matches!(agg_type, AggType::RANK) {
                    state.update_rank(new_order, curr_rank);
                }
            }

            peer_buffer.push(t.clone());
            row_num += 1;
            prev_partition_key = Some(curr_partition_key);
            prev_order_key = Some(curr_order_key);
        }

        // we need to make sure to push the final partition buffer here
        if !peer_buffer.is_empty() {
            flush_peer_group(
                &mut peer_buffer,
                &mut partition_aggregates,
                &self.plan.aggregates,
                child_schema,
                &self.plan.schema,
                &mut output_tuples,
            )?;
        }

        self.iter = Some(output_tuples.into_iter());
        Ok(())
    }

    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError> {
        let Some(iter) = self.iter.as_mut() else {
            return Err(ExecutorError(
                "WindowFunctionExecutor::next called before open".to_string(),
            ));
        };

        let Some(tup) = iter.next() else {
            return Ok(None);
        };

        Ok(Some((tup, RecordId::RESERVED)))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

fn hash_key(key: &[Option<Value>]) -> u64 {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

/// Zips up input tuples from buffer with provided aggregates, and outputs to a new vector of the combined tuples
fn flush_peer_group(
    buffer: &mut Vec<Tuple>,
    aggregates: &mut [AggState],
    plan_aggs: &[(Expression, AggType)],
    child_schema: &Schema,
    out_schema: &Schema,
    output: &mut Vec<Tuple>,
) -> Result<(), ExecutorError> {
    for buf_tup in buffer.iter() {
        for (state, (expr, agg_type)) in aggregates.iter_mut().zip(plan_aggs.iter()) {
            if !matches!(agg_type, AggType::RANK) {
                let v = expr.evaluate(buf_tup, child_schema)?;
                state.update(v);
            }
        }
    }
    for buf_tup in buffer.drain(..) {
        let mut combined: Vec<Option<Value>> = Vec::with_capacity(out_schema.cols.len());
        for i in 0..child_schema.cols.len() {
            combined.push(child_schema.get_value(&buf_tup, i)?);
        }
        for agg in aggregates.iter() {
            combined.push(agg.peek());
        }
        let out = out_schema.encode_tuple(&combined)?;
        output.push(out);
    }
    Ok(())
}
