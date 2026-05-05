use crate::buffer_pool::BufferPoolManager;
use crate::catalog::{Catalog, Schema};
use crate::table_heap::{RecordId, Tuple};
use std::sync::Arc;

#[derive(Debug)]
enum ExecutorError {}

pub struct ExecutorContext<'a> {
    pub catalog: &'a Catalog,
}

trait Executor {
    fn init(&mut self) -> Result<(), ExecutorError>;
    fn next(&mut self) -> Result<Option<(Tuple, RecordId)>, ExecutorError>;
    fn schema(&self) -> &Schema;
}
