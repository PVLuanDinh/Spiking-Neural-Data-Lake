//! R0.4 parity: every claimed kit cell (outputs + post-state, per section, per seed) against the Phase 0
//! fixtures. The copy here must be the kit's bytes (sha256 asserted first, either checkout line ending).
use sndl_core::rng::parity::{KIT_SHA, UNCLAIMED, is_kit_fixture, run, sha256_hex};

const FIX: &str = include_str!("fixtures_rng.json");

#[test]
fn every_claimed_cell_matches_cpython_and_numpy() {
    assert!(
        is_kit_fixture(FIX.as_bytes()),
        "fixture copy drifted from the kit's: sha256 {} != {KIT_SHA}",
        sha256_hex(FIX.as_bytes())
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
