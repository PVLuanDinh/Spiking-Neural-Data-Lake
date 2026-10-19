//! sndl-core: the Rust home of SNDL engine components (plan section 11: lake, brain, reservoir,
//! plasticity, readout, sleep, rng). R0.4 ships `rng`; Phase 2 ships `pfc` (the prefrontal decision
//! race). No PyO3 here: the binding crate is `sndl-python`.
#![forbid(unsafe_code)]

pub mod prefrontal_executive;
pub mod rng;
