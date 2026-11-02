# Every check CI runs, so `just` or `just ci` before pushing matches CI.
# CI calls the `lint`, `test` and `coverage` recipes below directly.

# Run fmt, clippy, the comment policy and the tests, as CI does.
ci: lint test

# fmt check, clippy with warnings as errors, and the comment policy.
lint:
    cargo fmt --all --check
    cargo clippy --all-targets -- -D warnings
    cargo clippy --benches --features bench -- -D warnings
    python3 scripts/check_comments.py

# Unit, integration, end-to-end and doc tests.
test:
    cargo test

# Line coverage with CI's 80% gate; needs cargo-llvm-cov. Extra args go to cargo llvm-cov.
coverage *args:
    cargo llvm-cov --fail-under-lines 80 {{args}}
