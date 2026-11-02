//! Seeds for randomized tests. The seed comes from `CMUDB_SEED`, so a failing
//! run can be replayed with the seed it prints.

use rand::SeedableRng;
use rand::rngs::StdRng;
use std::env::VarError;

pub const SEED_ENV: &str = "CMUDB_SEED";
pub const DEFAULT_SEED: u64 = 0x00C0_FFEE;

const STREAM_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Parses a `CMUDB_SEED` value; `None` (variable unset) gives `DEFAULT_SEED`.
pub fn parse_seed(value: Option<&str>) -> Result<u64, String> {
    match value {
        None => Ok(DEFAULT_SEED),
        Some(text) => text
            .trim()
            .parse()
            .map_err(|e| format!("{SEED_ENV}={text:?} is not a u64: {e}")),
    }
}

/// Prints the seed to stderr when dropped during a panic.
struct SeedReport {
    seed: u64,
}

impl Drop for SeedReport {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "randomized test failed with seed {seed}; replay with {SEED_ENV}={seed}",
                seed = self.seed
            );
        }
    }
}

/// Reads the run's seed from `CMUDB_SEED` (or `DEFAULT_SEED`), runs `test` with it,
/// and prints the seed to stderr if `test` panics.
/// Panics when the variable is set to something other than a `u64`.
pub fn with_seed<R>(test: impl FnOnce(u64) -> R) -> R {
    let value = match std::env::var(SEED_ENV) {
        Ok(text) => Some(text),
        Err(VarError::NotPresent) => None,
        Err(VarError::NotUnicode(raw)) => panic!("{SEED_ENV}={raw:?} is not valid UTF-8"),
    };
    let seed = parse_seed(value.as_deref()).unwrap_or_else(|e| panic!("{e}"));
    let _report = SeedReport { seed };
    test(seed)
}

/// One independent random sequence within a seeded run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Setup,
    Worker(usize),
}

impl Stream {
    fn index(self) -> u64 {
        match self {
            Stream::Setup => 0,
            Stream::Worker(worker) => worker as u64 + 1,
        }
    }
}

/// Returns the RNG for one stream of the run. Equal seeds and streams give equal sequences,
/// and distinct streams of one seed give distinct sequences.
pub fn stream_rng(seed: u64, stream: Stream) -> StdRng {
    StdRng::seed_from_u64(seed ^ stream.index().wrapping_add(1).wrapping_mul(STREAM_MIX))
}
