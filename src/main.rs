use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use cmudb::catalog::{Schema, Value};
use cmudb::database::Row;
use cmudb::{Database, DdlOp, StatementResult};

const NUM_FRAMES: usize = 64;

const USAGE: &str = "usage: cmudb [DIR]\n\n\
Opens the database stored in directory DIR, creating it if it is missing, and keeps it\n\
there across runs. Without DIR, runs on a temporary database that is deleted on exit.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let db = match args.as_slice() {
        [] => Database::scratch(NUM_FRAMES).map_err(|e| e.to_string()),
        [flag] if flag == "-h" || flag == "--help" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        [dir] => Database::open(dir, NUM_FRAMES)
            .map_err(|e| format!("cannot open the database in {dir}: {e}")),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let db = match db {
        Ok(db) => db,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    run_shell(&db);
    match db.close() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: writing the database to disk failed: {e}");
            ExitCode::FAILURE
        }
    }
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
            let cell_max = rows.iter().map(|r| r[i].chars().count()).max().unwrap_or(0);
            h.chars().count().max(cell_max)
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
    v.to_string()
}
