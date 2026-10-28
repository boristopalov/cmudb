use cmudb::catalog::{Catalog, Schema, Value};
use cmudb::create_buffer_pool_manager;
use cmudb::disk::{DiskManager, DiskScheduler};
use cmudb::parser::Binder;
use cmudb::planner::Planner;
use cmudb::processor::executor::Executor;
use cmudb::processor::executor_factory::create_executor;
use cmudb::processor::plan::Plan;
use cmudb::replacer::ArcReplacer;
use env_logger::Env;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::sync::Arc;
use tempfile::tempdir;

fn init_logger() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = env_logger::Builder::from_env(Env::default().default_filter_or("debug")).try_init();
    });
}

struct TestContext<'a> {
    planner: Planner<'a>,
    binder: Binder<'a>,
}

fn new_test_context<'a>(catalog: &'a Catalog) -> TestContext<'a> {
    let binder = Binder::new(&catalog);
    let planner = Planner::new(&catalog);
    TestContext { planner, binder }
}

#[test]
fn test_e2e_flow() {
    // wires up the whole dbms and executes several queries against the db
    let num_frames: usize = 64;
    let dir = tempdir().expect("failed to create tempdir for db");
    let dm = DiskManager::new(dir.path().join("cmudb.data"), dir.path().join("cmudb.log"))
        .expect("failed to create disk manager");
    let scheduler = DiskScheduler::new(dm);
    let replacer = ArcReplacer::new(num_frames);
    let bpm = Arc::new(create_buffer_pool_manager(num_frames, replacer, scheduler));
    let catalog = Catalog::new(bpm.clone());
    let context = new_test_context(&catalog);

    let dialect = GenericDialect;
    let create_sql = "CREATE TABLE users ( \
        id INTEGER PRIMARY KEY, \
        age INTEGER \
    );";
    let create_ast = Parser::parse_sql(&dialect, create_sql)
        .map_err(|e| format!("parse: {e}"))
        .unwrap();
    let create_bound = context
        .binder
        .bind(create_ast)
        .map_err(|e| format!("bind: {}", e.0))
        .unwrap();
    for stmt in create_bound {
        let plan = context
            .planner
            .plan(stmt)
            .map_err(|e| format!("plan: {}", e.0))
            .unwrap();
        let mut exec = create_executor(&catalog, &bpm, &plan);
        exec.open().map_err(|e| format!("open: {}", e.0)).unwrap();
        execute(&plan, &mut *exec).unwrap();
        exec.close().map_err(|e| format!("close: {}", e.0)).unwrap();
    }

    let insert_sql = "INSERT INTO users (id, age) VALUES (1, 25), (2, 50);";
    let insert_ast = Parser::parse_sql(&dialect, insert_sql)
        .map_err(|e| format!("parse: {e}"))
        .unwrap();
    let insert_bound = context
        .binder
        .bind(insert_ast)
        .map_err(|e| format!("bind: {}", e.0))
        .unwrap();
    for stmt in insert_bound {
        let plan = context
            .planner
            .plan(stmt)
            .map_err(|e| format!("plan: {}", e.0))
            .unwrap();
        let mut exec = create_executor(&catalog, &bpm, &plan);
        exec.open().map_err(|e| format!("open: {}", e.0)).unwrap();
        execute(&plan, &mut *exec).unwrap();
        exec.close().map_err(|e| format!("close: {}", e.0)).unwrap();
    }

    let query_sql = "SELECT * FROM users;";
    let query_ast = Parser::parse_sql(&dialect, query_sql)
        .map_err(|e| format!("parse: {e}"))
        .unwrap();
    let query_bound = context
        .binder
        .bind(query_ast)
        .map_err(|e| format!("bind: {}", e.0))
        .unwrap();
    for stmt in query_bound {
        let plan = context
            .planner
            .plan(stmt)
            .map_err(|e| format!("plan: {}", e.0))
            .unwrap();
        let mut exec = create_executor(&catalog, &bpm, &plan);
        exec.open().map_err(|e| format!("open: {}", e.0)).unwrap();
        execute(&plan, &mut *exec).unwrap();
        exec.close().map_err(|e| format!("close: {}", e.0)).unwrap();
    }
}

fn execute(plan: &Plan, exec: &mut dyn Executor) -> Result<(), String> {
    match plan {
        Plan::Insert(_) => {
            if let Some((tup, _)) = exec.next().map_err(|e| format!("exec: {}", e.0))? {
                let n = i32::from_ne_bytes(tup.data.as_slice().try_into().unwrap_or([0, 0, 0, 0]));
                println!("INSERT {n}");
            }
        }
        Plan::Update(_) => {
            let mut n = 0;
            while exec.next().map_err(|e| format!("exec: {}", e.0))?.is_some() {
                n += 1;
            }
            println!("UPDATE {n}");
        }
        Plan::Delete(_) => {
            let mut n = 0;
            while exec.next().map_err(|e| format!("exec: {}", e.0))?.is_some() {
                n += 1;
            }
            println!("DELETE {n}");
        }
        Plan::CreateTable(p) => {
            exec.next().map_err(|e| format!("exec: {}", e.0))?;
            println!("CREATE TABLE {}", p.table_name);
        }
        Plan::CreateIndex(p) => {
            exec.next().map_err(|e| format!("exec: {}", e.0))?;
            println!("CREATE INDEX {}", p.index_name);
        }
        Plan::DropTable(_) => {
            exec.next().map_err(|e| format!("exec: {}", e.0))?;
            println!("DROP TABLE");
        }
        Plan::DropIndex(_) => {
            exec.next().map_err(|e| format!("exec: {}", e.0))?;
            println!("DROP INDEX");
        }
        _ => print_rows(exec).unwrap(),
    }
    Ok(())
}

fn print_rows(exec: &mut dyn Executor) -> Result<(), String> {
    let schema: Schema = exec.schema().clone();
    let headers: Vec<String> = schema.cols.iter().map(|c| c.name.clone()).collect();

    let mut rows: Vec<Vec<String>> = Vec::new();
    while let Some((tup, _)) = exec.next().map_err(|e| format!("exec: {}", e.0))? {
        let mut row = Vec::with_capacity(schema.cols.len());
        for i in 0..schema.cols.len() {
            let cell = match schema.get_value(&tup, i) {
                Ok(Some(v)) => format_value(&v),
                Ok(None) => "NULL".to_string(),
                Err(e) => format!("<err {:?}>", e),
            };
            row.push(cell);
        }
        rows.push(row);
    }

    let widths: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let cell_max = rows.iter().map(|r| r[i].len()).max().unwrap_or(0);
            h.len().max(cell_max)
        })
        .collect();

    let sep = format!(
        "+{}+",
        widths
            .iter()
            .map(|w| "-".repeat(w + 2))
            .collect::<Vec<_>>()
            .join("+")
    );

    println!("{sep}");
    println!(
        "|{}|",
        headers
            .iter()
            .zip(&widths)
            .map(|(h, w)| format!(" {:<width$} ", h, width = w))
            .collect::<Vec<_>>()
            .join("|")
    );
    println!("{sep}");
    for row in &rows {
        println!(
            "|{}|",
            row.iter()
                .zip(&widths)
                .map(|(c, w)| format!(" {:<width$} ", c, width = w))
                .collect::<Vec<_>>()
                .join("|")
        );
    }
    println!("{sep}");
    println!(
        "({} row{})",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    );
    Ok(())
}

fn format_value(v: &Value) -> String {
    match v {
        Value::BOOLEAN(b) => b.to_string(),
        Value::INT(i) => i.to_string(),
        Value::FLOAT(f) => f.into_inner().to_string(),
        Value::TIMESTAMP(t) => t.to_string(),
    }
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
                // INSERT/DDL executors produce one result and are not streams.
                if matches!(plan, Plan::Insert(_) | Plan::CreateTable(_)) {
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
