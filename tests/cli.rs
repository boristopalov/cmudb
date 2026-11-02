//! End-to-end tests for the `cmudb` SQL shell binary: SQL goes in on stdin,
//! results come out on stdout and errors on stderr.
use std::io::Write;
use std::process::{Command, Stdio};

const CONTINUATION_PROMPT: &str = "    -> ";

fn run_shell(input: &str) -> (String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cmudb"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cmudb");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("wait for cmudb");
    assert!(out.status.success(), "shell exited with {:?}", out.status);
    (
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

#[test]
fn shell_runs_ddl_dml_and_prints_a_table() {
    let (out, err) = run_shell(
        "CREATE TABLE t (id INTEGER, ok BOOLEAN, f FLOAT);\n\
         CREATE INDEX t_id ON t (id);\n\
         INSERT INTO t VALUES (1, true, 1.5), (22, NULL, 2.0);\n\
         \n\
         SELECT *\n  FROM t\n  WHERE id > 0;\n\
         UPDATE t SET f = 3.5 WHERE id = 22;\n\
         DELETE FROM t WHERE id = 1;\n\
         SELECT id FROM t;\n\
         \\q\n\
         SELECT 'never reached';\n",
    );
    assert_eq!(err, "", "unexpected stderr");
    for expected in [
        "CREATE TABLE t",
        "CREATE INDEX t_id",
        "INSERT 2",
        "| id | ok   | f   |",
        "| 1  | true | 1.5 |",
        "| 22 | NULL | 2   |",
        "(2 rows)",
        "UPDATE 1",
        "DELETE 1",
        "(1 row)",
        CONTINUATION_PROMPT,
    ] {
        assert!(out.contains(expected), "missing {expected:?} in:\n{out}");
    }
    assert!(!out.contains("never reached"));
}

#[test]
fn shell_reports_errors_and_keeps_going() {
    let (out, err) = run_shell("SELEC 1;\nSELECT * FROM missing;\nSELECT 1 + 1;\nquit\n");
    assert!(err.contains("error: parse:"), "{err}");
    assert!(
        err.contains("error: bind: table not found: missing"),
        "{err}"
    );
    assert!(out.contains("| 2    |"), "{out}");
}

#[test]
fn shell_exits_cleanly_at_end_of_input() {
    let (out, _) = run_shell(
        "CREATE TABLE t (a INTEGER);\nCREATE INDEX t_a ON t (a);\nDROP INDEX t_a;\nDROP TABLE t;\nexit\n",
    );
    assert!(out.contains("CREATE TABLE t"));
    assert!(out.contains("DROP INDEX"), "{out}");
    assert!(out.contains("DROP TABLE"), "{out}");
    let (out, _) = run_shell("");
    assert!(out.starts_with("cmudb"));
}
