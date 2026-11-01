# Contributing

## Checks

CI runs these on every pull request; run them locally before pushing:

```sh
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
python3 scripts/check_comments.py
cargo test
cargo llvm-cov --fail-under-lines 80   # needs cargo-llvm-cov
```

Unit tests live next to the code in `#[cfg(test)] mod tests`. Integration and
end-to-end tests live in `tests/`; `tests/common` runs SQL text through the
whole engine and `tests/cli.rs` drives the `cmudb` shell binary.

## Comments

The code has to explain itself. Anything a reader would otherwise need to keep
in their head belongs in the code, where the compiler or the test suite checks
it:

| Instead of a comment that says...     | write                                         |
|---------------------------------------|-----------------------------------------------|
| what a block does                     | a function with that name                     |
| what a magic number means             | a named constant                              |
| "this is never X" / "caller must ..." | a type that rules X out, or `assert!`         |
| "this can fail when ..."              | a `Result` with a specific error variant      |
| "known limitation / workaround / TODO"| a fix, or a typed error and an `#[ignore]`d test that names the gap |
| why a test is shaped the way it is    | a descriptive test name and assert messages   |

`scripts/check_comments.py` enforces this in CI:

- No `//` line comments and no `/* */` block comments in `src/` or `tests/`.
- Doc comments (`///`, `//!`) are allowed to say what an item does, returns or
  fails with. They may not carry hidden knowledge, so words such as TODO,
  FIXME, hack, workaround, "for now", temporary, invariant, assume, must,
  never, always, careful and "note that" are rejected.
- `// SAFETY:` is allowed directly above an `unsafe` block, to say why it is
  sound.

If you find yourself wanting to write a comment, change the code instead. If
the knowledge cannot be expressed in code, raise it in the pull request rather
than leaving it in the source.
