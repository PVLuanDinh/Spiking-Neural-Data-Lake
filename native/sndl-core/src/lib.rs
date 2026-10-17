//! sndl-core: the Rust home of SNDL engine components (plan section 11: lake, brain, reservoir,
//! plasticity, readout, sleep, rng). Stage R0.4 ships `rng` only; no PyO3 here (the binding crate
//! `sndl-python` arrives with the first L-stage port).
#![forbid(unsafe_code)]

pub mod rng;
