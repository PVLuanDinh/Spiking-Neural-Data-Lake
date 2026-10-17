#!/usr/bin/env python3
"""native_ratchet -- src files importing a C/C++-backed library may only go DOWN (R0.3 meter).

Owner directive 2026-10-17: replace the C/C++-backed libraries with Rust, library by library. This
gate is the meter for that arc: per library, how many DISTINCT `src/` files import it. It asks for a
NUMBER, not a verdict, and has no per-file allowlist (same design as `p4c_ratchet`).

The importer walk is `dependency_ledger._imported`, not a second one. That walk drops stdlib modules,
and zlib/struct are stdlib C modules, so they are counted by running the SAME walk with the stdlib
filter switched off for the call.

    python tools/native_ratchet.py                   # report
    python tools/native_ratchet.py --assert          # red if any library rose / is not in the baseline
    python tools/native_ratchet.py --update-baseline # re-derive after a fall; refuses a rise
    python tools/native_ratchet.py --self-check      # constructed rise -> red, fall -> green
"""

import json
import pathlib
import sys
import types
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import dependency_ledger as dl  # noqa: E402

from snn_data_lake.gates import rust_facts  # noqa: E402

BASELINE_PATH = ROOT / "tools" / "native_ratchet_baseline.json"
NATIVE_ROOT = ROOT
THIRD_PARTY = (
    "numpy",
    "torch",
    "scipy",
    "scikit-learn",
    "pyarrow",
    "pygenn",
    "mujoco",
    "brian2",
)
STDLIB_C = ("zlib", "struct")
LIBS = THIRD_PARTY + STDLIB_C


def _walk():
    """`dependency_ledger._imported` over src/ with stdlib kept, so zlib/struct are visible."""
    with mock.patch.object(dl, "sys", types.SimpleNamespace(stdlib_module_names=frozenset())):
        return dl._imported(("src",))


def census(imported=None):
    """({lib: sorted distinct files}, sorted files importing >= 1 third-party library)."""
    hits = _walk() if imported is None else imported
    zones = {lib: sorted({site[0] for site in hits.get(lib, [])}) for lib in LIBS}
    union = sorted({f for lib in THIRD_PARTY for f in zones[lib]})
    return zones, union


def counts(zones):
    return {lib: len(files) for lib, files in zones.items()}


def compare(live, baseline):
    """[(lib, live, allowed)] for each library above its baseline; allowed None = not in the baseline."""
    return [
        (lib, n, baseline.get(lib))
        for lib, n in sorted(live.items())
        if baseline.get(lib) is None or n > baseline[lib]
    ]


def native_line(root):
    """The Rust half: native/ status, failing closed (RustFactsUnavailable propagates) with .rs present."""
    return rust_facts.native_status(root)[0]


def _load_baseline():
    """{lib: allowed} from the baseline: one JSON string of counts in LIBS order. The verify line runs
    ruff over the .json as Python, so only a lone string (a docstring) passes B018 and the formatter."""
    if not BASELINE_PATH.exists():
        return None
    return dict(
        zip(
            LIBS,
            map(int, json.loads(BASELINE_PATH.read_text(encoding="utf-8-sig")).split()),
            strict=False,
        )
    )


def _report(zones, union, baseline):
    allowed = baseline or {}
    print(native_line(NATIVE_ROOT))
    print("rust blind spots: " + "; ".join(rust_facts.BLIND_SPOTS))
    print(f"{'library':<14}{'files':>7}{'baseline':>10}")
    for lib, n in counts(zones).items():
        print(f"{lib:<14}{n:>7}{allowed.get(lib, '-'):>10}")
    print(f"{len(union)} src files import >= 1 third-party C/C++ library")


def self_check():
    base = {lib: 5 for lib in LIBS}
    print("native_ratchet --self-check")
    steady = compare(base, base)
    print(f"[control ] unmutated counts == baseline -> {len(steady)} findings (must be 0)")
    rise = compare(dict(base, numpy=6), base)
    print(f"[RED     ] numpy 5 -> 6 -> {rise} (must name numpy)")
    new = compare(dict(base), {k: v for k, v in base.items() if k != "torch"})
    print(f"[RED     ] torch missing from baseline -> {new} (must name torch)")
    fall = compare(dict(base, numpy=4), base)
    print(
        f"[GREEN   ] numpy 5 -> 4 -> {fall} (must be empty; update the baseline to lock the fall in)"
    )
    ok = steady == [] and rise == [("numpy", 6, 5)] and new == [("torch", 5, None)] and fall == []
    print("self-check OK" if ok else "BROKEN: a constructed case did not behave")
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-check" in argv:
        return self_check()
    try:
        zones, union = census()
        baseline = _load_baseline()
        _report(zones, union, baseline)
    except rust_facts.RustFactsUnavailable as e:
        print(f"GATE FAIL: native/ has .rs files but rust_facts is unavailable: {e}")
        return 1
    live = counts(zones)
    held = baseline or {}

    if "--update-baseline" in argv:
        rises = compare(live, held) if baseline else []
        if rises:
            print(
                "REFUSED: --update-baseline may only ratchet DOWN. These libraries ROSE or are new:"
            )
            for lib, n, allowed in rises:
                print(f"  {lib}: {n} > baseline {allowed}")
            return 1
        BASELINE_PATH.write_text(
            json.dumps(" ".join(str(live[lib]) for lib in LIBS)) + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {BASELINE_PATH.relative_to(ROOT).as_posix()}  union={len(union)}")
        return 0

    if "--assert" not in argv:
        print("(report only -- pass --assert to ratchet, --self-check to prove it can fire)")
        return 0
    if baseline is None:
        print(
            f"GATE FAIL: no baseline at {BASELINE_PATH.relative_to(ROOT).as_posix()}; run --update-baseline"
        )
        return 1
    findings = compare(live, held)
    for lib, n, allowed in findings:
        print(f"GATE FAIL: {lib}: {n} files > baseline {allowed}")
    if findings:
        return 1
    print("gate OK: no library above its baseline (importers may only fall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
