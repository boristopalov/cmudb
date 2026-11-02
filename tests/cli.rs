//! End-to-end tests for the `cmudb` SQL shell binary: SQL goes in on stdin,
//! results come out on stdout and errors on stderr.
use std::io::Write;
use std::process::{Command, Stdio};

const CONTINUATION_PROMPT: &str = "    -> ";

fn run_shell(input: &str) -> (String, String) {
    run_shell_with(&[], input)
}

fn run_shell_with(args: &[&std::ffi::OsStr], input: &str) -> (String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cmudb"))
        .args(args)
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

#[test]
fn shell_prints_bigints_floats_and_timestamps_readably() {
    let (out, err) = run_shell(
        "CREATE TABLE ev (n BIGINT, x FLOAT, at TIMESTAMP);\n\
         INSERT INTO ev VALUES (9000000000, 16777217, '2024-02-29 12:00:00.5');\n\
         SELECT * FROM ev;\n",
    );
    assert_eq!(err, "", "unexpected stderr");
    assert!(
        out.contains("| 9000000000 | 16777217 | 2024-02-29 12:00:00.5 |"),
        "row not printed readably in:\n{out}"
    );
}

#[test]
fn a_database_directory_keeps_its_data_from_one_shell_to_the_next() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("mydb");
    let args = [db.as_os_str()];

    let (out, err) = run_shell_with(
        &args,
        "CREATE TABLE t (id INTEGER, ok BOOLEAN);\n\
         CREATE INDEX t_id ON t (id);\n\
         INSERT INTO t VALUES (1, true), (2, false);\n\
         BEGIN;\n\
         INSERT INTO t VALUES (3, true);\n\
         COMMIT;\n\
         BEGIN;\n\
         INSERT INTO t VALUES (4, true);\n\
         \\q\n",
    );
    assert_eq!(err, "", "unexpected stderr");
    assert!(out.contains("COMMIT"), "{out}");
    assert!(db.join(cmudb::database::DATA_FILE).is_file());

    let (out, err) = run_shell_with(&args, "SELECT * FROM t;\nSELECT id FROM t WHERE id = 3;\n");
    assert_eq!(err, "", "unexpected stderr");
    for expected in [
        "| 1  | true  |",
        "| 2  | false |",
        "| 3  | true  |",
        "(3 rows)",
        "(1 row)",
    ] {
        assert!(out.contains(expected), "missing {expected:?} in:\n{out}");
    }
    assert!(
        !out.contains("| 4 "),
        "the block left open at \\q was rolled back:\n{out}"
    );
}

#[test]
fn the_shell_rejects_extra_arguments_and_a_foreign_data_file() {
    let out = Command::new(env!("CARGO_BIN_EXE_cmudb"))
        .args(["a", "b"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8(out.stderr)
            .unwrap()
            .contains("usage: cmudb [DIR]")
    );

    let out = Command::new(env!("CARGO_BIN_EXE_cmudb"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(
        String::from_utf8(out.stdout)
            .unwrap()
            .contains("usage: cmudb [DIR]")
    );

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(cmudb::database::DATA_FILE), [1u8; 100]).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_cmudb"))
        .arg(dir.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains("cannot open the database in"), "{err}");
}
