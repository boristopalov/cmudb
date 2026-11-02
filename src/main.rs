use std::io::{self, BufRead, Write};

use cmudb::catalog::{Schema, Value};
use cmudb::database::Row;
use cmudb::{Database, DdlOp, StatementResult};

const NUM_FRAMES: usize = 64;

fn main() {
    let db = Database::scratch(NUM_FRAMES).expect("failed to create a temporary database");
    run_shell(&db);
}

fn run_shell(db: &Database) {
    let stdin = io::stdin();
    let stdin = stdin.lock();
    let mut stdout = io::stdout();
    let mut buf = String::new();

    println!("cmudb — a toy SQL shell. terminate statements with `;`. type `\\q` to quit.");

    for line in stdin.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        };

        if buf.is_empty() {
            let t = line.trim();
            if t == "\\q" || t == "quit" || t == "exit" {
                break;
            }
            if t.is_empty() {
                prompt(&mut stdout, false);
                continue;
            }
        }

        buf.push_str(&line);
        buf.push('\n');

        if buf.trim_end().ends_with(';') {
            let sql = std::mem::take(&mut buf);
            if let Err(e) = db.execute_with(&sql, print_result) {
                eprintln!("error: {e}");
            }
            prompt(&mut stdout, false);
        } else {
            prompt(&mut stdout, true);
        }
    }
}

fn prompt(out: &mut io::Stdout, continuation: bool) {
    let p = if continuation { "    -> " } else { "cmudb> " };
    let _ = out.write_all(p.as_bytes());
    let _ = out.flush();
}

fn print_result(result: StatementResult) {
    match result {
        StatementResult::Rows { schema, rows } => print_rows(&schema, &rows),
        StatementResult::Count { op, rows } => println!("{op} {rows}"),
        StatementResult::Ddl(DdlOp::CreateTable(name)) => println!("CREATE TABLE {name}"),
        StatementResult::Ddl(DdlOp::CreateIndex(name)) => println!("CREATE INDEX {name}"),
        StatementResult::Ddl(DdlOp::DropTable) => println!("DROP TABLE"),
        StatementResult::Ddl(DdlOp::DropIndex) => println!("DROP INDEX"),
        StatementResult::Txn(op) => println!("{op}"),
    }
}

fn print_rows(schema: &Schema, values: &[Row]) {
    let headers: Vec<String> = schema.cols.iter().map(|c| c.name.clone()).collect();
    let rows: Vec<Vec<String>> = values
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| v.as_ref().map_or_else(|| "NULL".to_string(), format_value))
                .collect()
        })
        .collect();

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
}

fn format_value(v: &Value) -> String {
    match v {
        Value::BOOLEAN(b) => b.to_string(),
        Value::INT(i) => i.to_string(),
        Value::FLOAT(f) => f.into_inner().to_string(),
        Value::TIMESTAMP(t) => t.to_string(),
    }
}
