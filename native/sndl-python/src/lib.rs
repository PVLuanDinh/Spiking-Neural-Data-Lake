//! sndl-python: the ONE thin PyO3 binding over `sndl-core` (plan section 9, polars' `crates/polars-python`
//! shape). No logic lives here: a `#[pyfunction]` converts its arguments, runs the core inside [`enter`] and
//! converts the result. Built as `snn_data_lake._native` by `tools/native_build.py`.
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
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

#[pymodule(gil_used = false)]
mod _native {
    #[pymodule_export]
    use super::panic_probe;
}
