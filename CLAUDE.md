Read CONTRIBUTING.md before changing code: it lists the checks CI runs and the
comment policy (no hidden-knowledge comments; put invariants, workarounds and
limitations into code, errors or tests instead).

Read ARCHITECTURE.md for the layer map, the latching rules, how to add an
executor and where each kind of test goes. MVCC is described in docs/mvcc.md.

## Commands

Run `just ci` before pushing: it runs every check CI runs (fmt, clippy, the
comment policy and the tests), and CI calls the same recipes.

```sh
just ci                                      # fmt, clippy, comment policy, tests (RUST_LOG=warn)
cargo run                                    # SQL shell on a fresh temporary database
cargo test --test sql_e2e                    # one integration test file
CMUDB_SEED=<seed> cargo test --test buffer_pool_integration   # replay a failed soak run
just coverage                                # CI's 80% coverage gate; needs cargo-llvm-cov
```

## Invariants

Details and the reasons are in ARCHITECTURE.md.

- Page guards (`PageRef`, `PageMut`) are the pin and the latch. Keep them in a block
  scope, and never hold one across a `next_batch` return.
- Latches are not reentrant: never fetch a page your thread already holds.
- Outside the B+ tree, hold one page guard at a time. Inside it, latch top down, and a
  sibling only while holding its parent.
- Lock order: index `unique_latch` (by oid), then page latches, then the transaction
  manager's version map.
- `next_batch` never returns an empty batch or more rows than its batch size.
- Known gaps are typed errors plus `#[ignore = "known gap: …"]` tests, not comments.
