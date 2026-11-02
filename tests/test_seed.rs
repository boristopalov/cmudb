mod common;

use common::seed::{SEED_ENV, Stream, parse_seed, stream_rng, with_seed};
use rand::Rng;

#[test]
fn unset_seed_picks_a_fresh_random_seed_each_time() {
    let first = parse_seed(None).expect("unset seed is valid");
    let second = parse_seed(None).expect("unset seed is valid");
    assert_ne!(first, second, "two unset reads should draw different seeds");
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

fn first_draws(seed: u64, stream: Stream) -> Vec<u64> {
    let mut rng = stream_rng(seed, stream);
    (0..8).map(|_| rng.random()).collect()
}

#[test]
fn same_seed_and_stream_replay_the_same_sequence() {
    assert_eq!(
        first_draws(7, Stream::Worker(3)),
        first_draws(7, Stream::Worker(3))
    );
}

#[test]
fn different_streams_or_seeds_give_different_sequences() {
    assert_ne!(
        first_draws(7, Stream::Setup),
        first_draws(7, Stream::Worker(0))
    );
    assert_ne!(
        first_draws(7, Stream::Worker(0)),
        first_draws(7, Stream::Worker(1))
    );
    assert_ne!(first_draws(7, Stream::Setup), first_draws(8, Stream::Setup));
}

#[test]
fn with_seed_passes_the_seed_and_returns_the_test_result() {
    let seen = with_seed(|seed| seed);
    if let Ok(text) = std::env::var(SEED_ENV) {
        let expected: u64 = text.trim().parse().expect("valid CMUDB_SEED");
        assert_eq!(seen, expected, "with_seed uses CMUDB_SEED when it is set");
    }
    assert_eq!(
        with_seed(|_| "result"),
        "result",
        "with_seed returns the test's value"
    );
}

#[test]
fn with_seed_lets_a_test_panic_through() {
    let outcome = std::panic::catch_unwind(|| with_seed(|_| panic!("soak failure")));
    assert!(
        outcome.is_err(),
        "a panic inside with_seed reaches the caller"
    );
}
