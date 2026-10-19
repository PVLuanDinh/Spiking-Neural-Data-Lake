"""The binding crate's contract (`native/sndl-python`, module `snn_data_lake._native`; Phase 0
items 5 and 8 of `sndl-rust-port-phase0-report-2026-10-17.md`). Each assert is a Phase 0 red seen
firing: a `gil_used = true` twin re-enables the GIL on import (item 5); a Rust panic NOT wrapped in
`catch_unwind` is a `PanicException`, a `BaseException` that walks through `except Exception` and
kills a tick loop, and under `panic = "abort"` the process dies with 0xC0000409 (item 8b); asking
for abi3 on 3.14t silently builds a version-specific wheel, so the build is judged by its tag, not
by a build error (item 8a).

The extension is built and placed by `python -m snn_data_lake.infra.native_build`; this file only
judges what was placed.
"""

import sys
import sysconfig

from snn_data_lake import _native


def test_gil_stays_off_after_import():
    assert sys._is_gil_enabled() is False, "the binding re-enabled the GIL (gil_used must be false)"


def test_panic_is_a_runtime_error_inside_an_except_exception_loop():
    seen, survived = [], 0
    for x in (1, -1, 2):  # the daemon's shape: one bad call must not end the loop
        try:
            assert _native.panic_probe(x) == 2 * x
            survived += 1
        except Exception as e:  # noqa: BLE001 -- the point: a panic must be an Exception
            seen.append(e)
    assert survived == 2 and len(seen) == 1, (survived, seen)
    assert type(seen[0]) is RuntimeError and "rust panic" in str(seen[0]), repr(seen[0])


def test_extension_is_version_specific_not_abi3():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")  # e.g. .cp314t-win_amd64.pyd
    free_threaded = f"{sys.version_info.major}{sys.version_info.minor}t"  # abi3: a bare .pyd
    assert free_threaded in suffix and _native.__file__.endswith(suffix), (_native.__file__, suffix)
