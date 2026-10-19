//! sndl-python: the ONE thin PyO3 binding over `sndl-core` (plan section 9, polars' `crates/polars-python`
//! shape). No logic lives here: a `#[pyfunction]` converts its arguments, runs the core inside [`enter`] and
//! converts the result. Built as `snn_data_lake._native` by `tools/native_build.py`.
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sndl_core::prefrontal_executive::{Params, decide_spiking, has_evidence};
use sndl_core::rng::mt19937::N;
use sndl_core::rng::pyrandom::{PyRandom, State};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// polars' `enter_polars` shape: run `f` detached from the interpreter, catch a panic, and map both a panic
/// and an `Err` to `RuntimeError`. LOAD-BEARING (Phase 0 item 8b): an uncaught panic is `PanicException`, a
/// `BaseException` that walks through `except Exception` and ends the daemon's tick loop. It only works under
/// `panic = "unwind"` (the workspace release profile); `tests/test_native_binding.py` asserts the behaviour.
fn enter<T, F>(py: Python<'_>, f: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> Result<T, String> + Send,
{
    match py.detach(|| catch_unwind(AssertUnwindSafe(f))) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".into());
            Err(PyRuntimeError::new_err(format!("rust panic: {msg}")))
        }
    }
}

/// Item 8b probe, the one entry point that panics on purpose: `x < 0` panics inside [`enter`], and the caller
/// must see `RuntimeError`, never a dead process. `x >= 0` returns `2 * x`.
#[pyfunction]
fn panic_probe(py: Python<'_>, x: i64) -> PyResult<i64> {
    enter(py, || {
        assert!(x >= 0, "negative input {x}");
        Ok(2 * x)
    })
}

/// `PrefrontalExecutive._decide_spiking` (Phase 2): one spiking winner-take-all decision on the CALLER'S CPython
/// `random.Random` stream. `internal` is `random.getstate()[1]` (624 words + position) and comes back in the same
/// shape for `setstate`, so the Python object's stream continues exactly where the race left it. `ints` =
/// `(n_dec, n_inh, t_max)`, `floats` = `(md, g_ext, b0, w_plus, w_inh, w_ie, w_cross, trace_decay, theta,
/// noise_amp, init_v)`, `dec_bias` = `_dec_bias` (at least `n_dec * len(evidence)` entries when there is evidence;
/// no evidence of any length is `(0, 1)` with no draw and no length check).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn pfc_decide(
    py: Python<'_>,
    evidence: Vec<f64>,
    ints: (usize, usize, usize),
    floats: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64),
    dec_bias: Vec<f64>,
    internal: Vec<u32>,
) -> PyResult<(usize, usize, Py<PyTuple>)> {
    let (winner, rt, st) = enter(py, || {
        if internal.len() != N + 1 || internal[N] as usize > N {
            return Err(format!(
                "internal state must be {} ints (624 words + position <= 624), got {}",
                N + 1,
                internal.len()
            ));
        }
        if has_evidence(&evidence) && dec_bias.len() < ints.0 * evidence.len() {
            return Err(format!(
                "dec_bias has {} entries, the race needs n_dec * n_options = {} * {}",
                dec_bias.len(),
                ints.0,
                evidence.len()
            ));
        }
        let mut words = [0u32; N];
        words.copy_from_slice(&internal[..N]);
        let mut rng = PyRandom::from_u128(0);
        rng.setstate(&State {
            words,
            pos: internal[N] as usize,
            gauss_next: None,
        });
        let (md, g_ext, b0, w_plus, w_inh, w_ie, w_cross, trace_decay, theta, noise_amp, init_v) =
            floats;
        let p = Params {
            n_dec: ints.0,
            n_inh: ints.1,
            t_max: ints.2,
            md,
            g_ext,
            b0,
            w_plus,
            w_inh,
            w_ie,
            w_cross,
            trace_decay,
            theta,
            noise_amp,
            init_v,
            dec_bias,
        };
        let (winner, rt) = decide_spiking(&p, &evidence, &mut rng);
        let st = rng.getstate();
        let mut internal = st.words.to_vec();
        internal.push(st.pos as u32);
        Ok((winner, rt, internal))
    })?;
    let internal = PyTuple::new(py, st)?;
    Ok((winner, rt, internal.unbind()))
}

#[pymodule(gil_used = false)]
mod _native {
    #[pymodule_export]
    use super::panic_probe;
    #[pymodule_export]
    use super::pfc_decide;
}
