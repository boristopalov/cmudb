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

/// Prints the seed to stderr when the test panics while this guard is alive.
pub struct SeedReport {
    seed: u64,
}

impl SeedReport {
    pub fn seed(&self) -> u64 {
        self.seed
    }
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

/// Reads the run's seed from `CMUDB_SEED` and returns a guard that prints it on failure.
/// Panics when the variable is set to something other than a `u64`.
pub fn seed_from_env() -> SeedReport {
    let value = match std::env::var(SEED_ENV) {
        Ok(text) => Some(text),
        Err(VarError::NotPresent) => None,
        Err(VarError::NotUnicode(raw)) => panic!("{SEED_ENV}={raw:?} is not valid UTF-8"),
    };
    let seed = parse_seed(value.as_deref()).unwrap_or_else(|e| panic!("{e}"));
    SeedReport { seed }
}

/// Returns an RNG for one numbered stream of the run, such as one worker thread.
/// Equal seeds and stream numbers give equal sequences.
pub fn stream_rng(seed: u64, stream: u64) -> StdRng {
    StdRng::seed_from_u64(seed ^ stream.wrapping_add(1).wrapping_mul(STREAM_MIX))
}
