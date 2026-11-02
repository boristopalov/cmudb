use crate::catalog::{Schema, Value};
use crate::processor::executor::{Batch, Executor, ExecutorError};
use crate::processor::executors::is_true;
use crate::processor::plan::{FilterPlanNode, LimitPlanNode, ProjectionPlanNode};

pub struct LimitExecutor<'a> {
    plan: &'a LimitPlanNode,
    child: Box<dyn Executor + 'a>,
    count: u32,
}

impl<'a> LimitExecutor<'a> {
    pub fn new(plan: &'a LimitPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self {
            plan,
            child,
            count: 0,
        }
    }
}

impl<'a> Executor for LimitExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()
    }
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        if self.count >= self.plan.limit {
            return Ok(None);
        }
        let Some(mut batch) = self.child.next_batch()? else {
            return Ok(None);
        };
        let remaining = (self.plan.limit - self.count) as usize;
        batch.rows.truncate(remaining);
        self.count += batch.len() as u32;
        Ok(Some(batch))
    }
    fn close(&mut self) -> Result<(), ExecutorError> {
        self.child.close()
    }
    fn schema(&self) -> &Schema {
        &self.plan.schema
    }
}

pub struct FilterExecutor<'a> {
    plan: &'a FilterPlanNode,
    child: Box<dyn Executor + 'a>,
}

impl<'a> FilterExecutor<'a> {
    pub fn new(plan: &'a FilterPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self { plan, child }
    }
}

impl<'a> Executor for FilterExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()
    }

    /// Filters child batches, pulling until at least one row passes, so that an empty batch
    /// is not handed up. A NULL predicate rejects the row.
    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        while let Some(batch) = self.child.next_batch()? {
            let child_schema = self.child.schema();
            let mut rows = Vec::with_capacity(batch.len());
            for (tup, rid) in batch {
                if is_true(self.plan.predicate.evaluate(&tup, child_schema)?) {
                    rows.push((tup, rid));
                }
            }
            if let Some(out) = Batch::from_rows(rows) {
                return Ok(Some(out));
            }
        }
        Ok(None)
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

pub struct ProjectionExecutor<'a> {
    plan: &'a ProjectionPlanNode,
    child: Box<dyn Executor + 'a>,
}

impl<'a> ProjectionExecutor<'a> {
    pub fn new(plan: &'a ProjectionPlanNode, child: Box<dyn Executor + 'a>) -> Self {
        Self { plan, child }
    }
}

impl<'a> Executor for ProjectionExecutor<'a> {
    fn open(&mut self) -> Result<(), ExecutorError> {
        self.child.open()
    }

    fn next_batch(&mut self) -> Result<Option<Batch>, ExecutorError> {
        let Some(batch) = self.child.next_batch()? else {
            return Ok(None);
        };
        let child_schema = self.child.schema();
        let mut rows = Vec::with_capacity(batch.len());
        for (tup, rid) in batch {
            let vals: Vec<Option<Value>> = self
                .plan
                .expressions
                .iter()
                .map(|e| e.evaluate(&tup, child_schema))
                .collect::<Result<_, _>>()?;
            rows.push((self.plan.schema.encode_tuple(&vals)?, rid));
        }
        Ok(Batch::from_rows(rows))
    }

    fn schema(&self) -> &Schema {
        &self.plan.schema
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::executors::test_support::*;

    #[test]
    fn limit_stops_pulling_from_its_child_once_reached() {
        let schema = int_schema(&["a"]);
        let rows = values_plan(&schema, &[&[1], &[2], &[3]]);
        let plan = LimitPlanNode {
            schema: schema.clone(),
            child: Box::new(values_plan(&schema, &[])),
            limit: 2,
        };
        let mut exec = LimitExecutor::new(&plan, values_child(&rows));
        assert_eq!(
            decode_ints(&schema, drain(&mut exec)),
            vec![vec![1], vec![2]]
        );
        assert!(exec.next_batch().unwrap().is_none());
    }
}
