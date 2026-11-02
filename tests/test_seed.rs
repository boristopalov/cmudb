mod common;

use common::seed::{DEFAULT_SEED, SEED_ENV, parse_seed, stream_rng};
use rand::Rng;

#[test]
fn unset_seed_uses_the_fixed_default() {
    assert_eq!(parse_seed(None), Ok(DEFAULT_SEED));
}

#[test]
fn seed_parses_decimal_u64_with_surrounding_whitespace() {
    assert_eq!(parse_seed(Some(" 42\n")), Ok(42));
    assert_eq!(parse_seed(Some("18446744073709551615")), Ok(u64::MAX));
}

#[test]
fn non_numeric_seed_is_rejected_with_the_variable_name() {
    let err = parse_seed(Some("abc")).expect_err("abc is not a seed");
    assert!(
        err.contains(SEED_ENV),
        "error should name {SEED_ENV}: {err}"
    );
    assert!(
        parse_seed(Some("-1")).is_err(),
        "negative seeds are rejected"
    );
    assert!(parse_seed(Some("")).is_err(), "an empty seed is rejected");
}

fn first_draws(seed: u64, stream: u64) -> Vec<u64> {
    let mut rng = stream_rng(seed, stream);
    (0..8).map(|_| rng.random()).collect()
}

#[test]
fn same_seed_and_stream_replay_the_same_sequence() {
    assert_eq!(first_draws(7, 3), first_draws(7, 3));
}

#[test]
fn different_streams_or_seeds_give_different_sequences() {
    assert_ne!(first_draws(7, 0), first_draws(7, 1));
    assert_ne!(first_draws(7, 0), first_draws(8, 0));
}
