mod common;

use cmudb::catalog::{Catalog, Schema, Value};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::parser::Binder;
use cmudb::planner::Planner;
use cmudb::processor::executor_factory::create_executor;
use cmudb::processor::plan::Plan;
use cmudb::replacer::ArcReplacer;
use common::{Db, ints};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::sync::Arc;
use tempfile::tempdir;

struct TestContext<'a> {
    planner: Planner<'a>,
    binder: Binder<'a>,
}

fn new_test_context<'a>(catalog: &'a Catalog) -> TestContext<'a> {
    let binder = Binder::new(catalog);
    let planner = Planner::new(catalog);
    TestContext { planner, binder }
}

#[test]
fn create_insert_select_round_trip() {
    let db = Db::new(64);
    db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER);");
    assert_eq!(
        ints(&db.exec("INSERT INTO users (id, age) VALUES (1, 25), (2, 50);")),
        vec![[2]]
    );
    assert_eq!(
        ints(&db.exec("SELECT * FROM users;")),
        vec![[1, 25], [2, 50]]
    );
}

#[test]
fn scan_driven_sql_update_visits_each_original_row_once() {
    use cmudb::catalog::DataType;

    for row_count in [8, 700] {
        let dir = tempdir().unwrap();
        let dm = DiskManager::new(dir.path().join("db"), dir.path().join("log")).unwrap();
        let bpm = Arc::new(create_buffer_pool_manager(
            1,
            ArcReplacer::new(1),
            DiskScheduler::new(dm),
        ));
        let catalog = Catalog::new(bpm.clone());
        let context = new_test_context(&catalog);
        let execute_sql = |sql: &str| {
            let ast = Parser::parse_sql(&GenericDialect, sql).unwrap();
            let statements = context.binder.bind(ast).unwrap();
            let mut result = Vec::new();
            for statement in statements {
                let plan = context.planner.plan(statement).unwrap();
                let mut exec = create_executor(&catalog, &bpm, &plan);
                exec.open().unwrap();
                let produces_single_status_row =
                    matches!(plan, Plan::Insert(_) | Plan::CreateTable(_));
                if produces_single_status_row {
                    result.extend(exec.next().unwrap());
                } else {
                    while let Some(row) = exec.next().unwrap() {
                        result.push(row);
                        assert!(
                            result.len() <= row_count,
                            "statement revisited appended replacement rows: {sql}"
                        );
                    }
                    assert!(exec.next().unwrap().is_none());
                }
                exec.close().unwrap();
            }
            result
        };
        execute_sql("CREATE TABLE items (id INTEGER, value INTEGER)");
        let values = (0..row_count)
            .map(|id| format!("({id}, 0)"))
            .collect::<Vec<_>>()
            .join(", ");
        let inserted = execute_sql(&format!("INSERT INTO items VALUES {values}"));
        assert_eq!(
            i32::from_ne_bytes(inserted[0].0.data.as_slice().try_into().unwrap()),
            row_count as i32
        );
        let original_rids: Vec<_> = execute_sql("SELECT * FROM items")
            .into_iter()
            .map(|(_, rid)| rid)
            .collect();
        let updated = execute_sql("UPDATE items SET value = value + 1");
        assert_eq!(updated.len(), row_count);
        let schema = Schema::new(vec![
            (DataType::INT, "id".to_string()),
            (DataType::INT, "value".to_string()),
        ]);
        let final_rows = execute_sql("SELECT * FROM items");
        assert_eq!(final_rows.len(), row_count);
        for (id, (tuple, rid)) in final_rows.iter().enumerate() {
            assert_eq!(
                schema.get_value(tuple, 0).unwrap(),
                Some(Value::INT(id as i32))
            );
            assert_eq!(schema.get_value(tuple, 1).unwrap(), Some(Value::INT(1)));
            assert!(!original_rids.contains(rid));
        }
    }
}
