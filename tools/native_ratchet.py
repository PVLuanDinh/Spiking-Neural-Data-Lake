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
COLS = (*LIBS, "union")
PKG_ROOT = "src/snn_data_lake/"
#: provisional until G8 boundaries settle: a report-only view, not a rule.
COMPONENTS = {
    "lake": ("lakehouse", "substrate"),
    "reservoir": ("reservoir",),
    "brain/grower": ("engine", "growth", "faculties", "brainstem"),
    "readouts": ("readout",),
    "sleep": ("sleep",),
}


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


def _package(path):
    head = path.removeprefix(PKG_ROOT)
    return head.split("/")[0] if "/" in head else "(root)"


def live_counts(zones, union):
    """Flat {col: n} over the whole tree plus {'<pkg>:<col>': n} for each package importing anything."""
    live = {**counts(zones), "union": len(union)}
    by_pkg = {}
    for lib, files in zones.items():
        for f in files:
            by_pkg.setdefault(_package(f), {}).setdefault(lib, set()).add(f)
    for pkg, libs in by_pkg.items():
        for col in COLS:
            files = (
                libs.get(col, set())
                if col != "union"
                else set().union(*(libs.get(t, set()) for t in THIRD_PARTY))
            )
            live[f"{pkg}:{col}"] = len(files)
    return live


def compare(live, baseline):
    """[(key, live, allowed)] above baseline; allowed None = key absent (red unless a package row of 0)."""
    return [
        (k, n, baseline.get(k))
        for k, n in sorted(live.items())
        if (baseline.get(k) is None and (n or ":" not in k))
        or (baseline.get(k) is not None and n > baseline[k])
    ]


def encode(live):
    """Whitespace-free one-line form `ALL=<csv>;<pkg>=<csv>;...` (see _load_baseline for why)."""
    pkgs = sorted({k.split(":")[0] for k in live if ":" in k})
    row = lambda pre: ",".join(str(live.get(pre + c, 0)) for c in COLS)  # noqa: E731
    return ";".join([f"ALL={row('')}", *(f"{p}={row(p + ':')}" for p in pkgs)])


def decode(text):
    live = {}
    for seg in text.split(";"):
        name, csv = seg.split("=")
        for col, n in zip(COLS, csv.split(","), strict=False):
            live[col if name == "ALL" else f"{name}:{col}"] = int(n)
    return live


def component_view(live):
    """Report-only component -> packages table."""
    out = ["components (provisional until G8 boundaries settle; a view, not a rule):"]
    pkgs = {k.split(":")[0] for k in live if ":" in k}
    for comp, members in {
        **COMPONENTS,
        "unassigned": tuple(sorted(pkgs - {m for v in COMPONENTS.values() for m in v})),
    }.items():
        union = sum(live.get(f"{m}:union", 0) for m in members)
        out.append(f"  {comp:<14}{union:>5} files  <- {', '.join(members) or '-'}")
    return out


def native_line(root):
    """The Rust half: native/ status, failing closed (RustFactsUnavailable propagates) with .rs present."""
    return rust_facts.native_status(root)[0]


def _load_baseline():
    """{key: allowed} from the baseline: ONE whitespace-free JSON string. The verify line runs ruff
    over the .json as Python, so only a lone string (a docstring) passes B018 and the formatter, and
    only a string with no whitespace passes E501 at this length."""
    if not BASELINE_PATH.exists():
        return None
    return decode(json.loads(BASELINE_PATH.read_text(encoding="utf-8-sig")))


def _report(zones, union, baseline):
    allowed = baseline or {}
    live = live_counts(zones, union)
    print(native_line(NATIVE_ROOT))
    print("rust blind spots: " + "; ".join(rust_facts.BLIND_SPOTS))
    print(f"{'library':<14}{'files':>7}{'baseline':>10}")
    for lib in LIBS:
        print(f"{lib:<14}{live[lib]:>7}{allowed.get(lib, '-'):>10}")
    print(f"{len(union)} src files import >= 1 third-party C/C++ library")
    for pkg in sorted({k.split(":")[0] for k in live if ":" in k}):
        row = " ".join(f"{c}={live[f'{pkg}:{c}']}" for c in COLS if live[f"{pkg}:{c}"])
        print(f"  pkg {pkg:<14}{row}")
    print("\n".join(component_view(live)))


def self_check():
    base = {col: 5 for col in COLS}
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
    pkg = compare({**base, "sleep:numpy": 3}, {**base, "sleep:numpy": 2})
    print(f"[RED     ] one package 2 -> 3, library total flat -> {pkg} (must name sleep:numpy)")
    ok = (
        pkg == [("sleep:numpy", 3, 2)]
        and steady == []
        and rise == [("numpy", 6, 5)]
        and new == [("torch", 5, None)]
        and fall == []
    )
    print("self-check OK" if ok else "BROKEN: a constructed case did not behave")
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-check" in argv:
        return self_check()
    try:
        zones, union = census()
        baseline = _load_baseline()
        live = live_counts(zones, union)
        _report(zones, union, baseline)
    except rust_facts.RustFactsUnavailable as e:
        print(f"GATE FAIL: native/ has .rs files but rust_facts is unavailable: {e}")
        return 1
    held = baseline or {}

    if "--update-baseline" in argv:
        first_pkgs = not any(
            ":" in k for k in held
        )  # the one-time derivation of the per-package rows
        rises = (
            [r for r in compare(live, held) if not (first_pkgs and ":" in r[0])] if baseline else []
        )
        if rises:
            print(
                "REFUSED: --update-baseline may only ratchet DOWN. These libraries ROSE or are new:"
            )
            for lib, n, allowed in rises:
                print(f"  {lib}: {n} > baseline {allowed}")
            return 1
        BASELINE_PATH.write_text(
            json.dumps(encode(live)) + "\n",
            encoding="utf-8",
        )
        print(
            f"baseline written: {BASELINE_PATH.relative_to(ROOT).as_posix()}  union={live['union']}"
        )
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
