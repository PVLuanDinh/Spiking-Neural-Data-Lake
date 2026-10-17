//! R0.4 parity: every claimed kit cell (outputs + post-state, per section, per seed) against the Phase 0
//! fixtures. The copy here must be byte-identical to the kit's (sha256 asserted first).
use sndl_core::rng::parity::{UNCLAIMED, run, sha256_hex};

const FIX: &str = include_str!("fixtures_rng.json");
const SHA: &str = "1515bc47f7e3c598f5faa52f96ba212f6e635170a07b56f20dcf42ef0f381997";

#[test]
fn every_claimed_cell_matches_cpython_and_numpy() {
    assert_eq!(
        sha256_hex(FIX.as_bytes()),
        SHA,
        "fixture copy drifted from the kit's"
    );
    let r = run(FIX);
    println!(
        "claimed {} failed {} unclaimed {} (OPEN sections {UNCLAIMED:?}): {:?}",
        r.claimed,
        r.failed.len(),
        r.unclaimed.len(),
        r.unclaimed
    );
    assert_eq!(r.claimed, 254);
    assert_eq!(r.unclaimed.len(), 39);
    assert!(r.failed.is_empty(), "failing cells: {:?}", r.failed);
}
