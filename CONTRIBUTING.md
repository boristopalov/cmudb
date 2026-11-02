# Contributing

## Checks

Run every check CI runs with one command before pushing:

```sh
just ci
```

It runs `cargo fmt --all --check`, clippy on all targets and on the benches,
`scripts/check_comments.py`, and `cargo test`. Tests install no logger, so
`RUST_LOG` has no effect on them. CI's lint and test jobs call the same
`just lint` and `just test` recipes, so a green `just ci` means green lint and
test jobs; CI's coverage job also runs `just coverage`. Install `just` with `cargo install just` (or `pip install rust-just`).

`just coverage` runs CI's 80% line-coverage gate and needs `cargo-llvm-cov`.

The toolchain is pinned in `rust-toolchain.toml`, and rustup installs it on
first use. Warnings fail `just lint`, `just ci` and CI (clippy runs with
`-D warnings`), but not local `cargo build` or `cargo test`.

The randomized soak tests take their seed from `CMUDB_SEED` (a fixed default
when unset) and print it when they fail; rerun with `CMUDB_SEED=<seed>` to
replay the same operations in each thread (thread interleaving still varies).

CI type-checks the benchmarks in `benches/` but never runs them. To run each bench once
without timing it, including the checks it makes on its results:

```sh
cargo bench --features bench -- --test
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

- No `//` line comments and no `/* */` block comments in `src/`, `tests/` or `benches/`.
- Doc comments (`///`, `//!`) are allowed to say what an item does, returns or
  fails with. They may not carry hidden knowledge, so words such as TODO,
  FIXME, hack, workaround, "for now", temporary, invariant, assume, must,
  never, always, careful and "note that" are rejected.
- `// SAFETY:` is allowed directly above an `unsafe` block, to say why it is
  sound.

If you find yourself wanting to write a comment, change the code instead. If
the knowledge cannot be expressed in code, raise it in the pull request rather
than leaving it in the source.
