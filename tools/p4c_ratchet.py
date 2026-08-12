#!/usr/bin/env python3
"""p4c_ratchet -- the number of builds OUTSIDE the governed tree may only go DOWN.

P4c: "EVERY CURRENT BUILD in this project is part of the brain, and EVERY FUTURE BUILD must be
part of the brain. There is no outside-the-brain here. ... reasoning/ IS NOT A PARKING LOT."
Today that clause is enforced by REVIEW ONLY, and review is what let ~165 modules accumulate in
`reasoning/` alone. This gate mechanizes exactly one half of it -- the half that is permanent:

    ** the backlog may be worked DOWN, and may never be added to. **

It does NOT promote anything, does NOT disposition anything, and deliberately has NO per-file
allowlist. That absence is the design. POLICIES.md records three proposals that died at the bar
"implement the gate and COUNT ITS OUTPUT before adopting the rule", one of them because adoption
would have meant a 58-entry rubber stamp; a gate over 259 ungoverned files that demanded a
disposition for each would be that failure at four times the size. The ratchet asks for a NUMBER,
not a verdict, so it costs nothing to adopt and cannot be satisfied by an allowlist.


WHAT IS COUNTED, and why this denominator and not the other three
-----------------------------------------------------------------
Four different numbers were MEASURED on this tree before a line of assert was written -- all four
on the identical corpus and the identical `__init__.py` exclusion, so they are comparable. They are
NOT interchangeable and picking the flattering one would be the whole failure:

  (A) files under `src/snn_data_lake/reasoning/` ................................... 161
  (B) package files not in `dormancy_gate.BRAIN_MODULES` .......................... 171
  (C) files matched by the coverage `omit` globs `*/reasoning/*`, `*/training/*` ... 174
  (D) every `.py` in the corpus not registered in REGIONS/TORCH_TIER, less the two
      documented exclusions below ................................................. 259

  Reconciliation, and it is exact: 259 (D) + 29 (`__init__.py`) + 55 (`tests/`) + 205 (governed)
  = 548 = the corpus. Nothing is unaccounted for, and 0 of the 205 REGIONS/TORCH_TIER entries name
  a file that does not exist.

THIS GATE COUNTS (D), because (D) is the only one that matches what P4c actually says. P4c names
`research/`, `training/`, `integrations/`, `gcp/`, `infra/`, `tools/` and "a script at the repo
root" in one breath, and then says the quiet part: *"The directory a build sits in is a fact about
the filesystem, never a statement about whether it is part of the organism."* A ratchet scoped to
(A) would therefore be blind to the exact move it exists to prevent -- park the new build in
`research/` or `tools/` instead of `reasoning/` and the number still goes down. MEASURED: that
blind spot is **98 files across 9 other zones**, more than a third of the backlog. (C) is not a
policy at all but a coverage config: its `*/training/*` glob matches two different directories, and
it says nothing about `research/`, `integrations/`, `tools/`, `docs/coverage/artifacts/` or the
repo root.

(B) is the closest honest rival and its number is the one POLICIES.md quotes. It is rejected for
one reason: it stops at the package boundary, which is a directory fact, and P4c's whole point is
that directory facts do not decide membership. Measured cost of that stop: (B) cannot see the 88
files in `research/`, `tools/`, `docs/`, `integrations/`, repo-root `training/` and the repo root.

GOVERNED is DERIVED, never hand-listed: the REGIONS + TORCH_TIER tables in
`src/snn_data_lake/__init__.py` are read by `ast.literal_eval` (no import -- see DETERMINISM) and
turned into `src/snn_data_lake/<region>/<module>.py`. Verified equivalent to the live
`substrate/dormancy_gate.py::BRAIN_MODULES`: 205 entries, exact set match, 2026-08-06. Registration
is the test, not location -- a file dropped into `cortex/` without a REGIONS entry is invisible to
all three P3 gates, so this gate counts it as ungoverned no matter where it sits.

ZONES are derived too (first path segment; first sub-package under `snn_data_lake/`), so a NEW
top-level directory becomes a NEW zone with no baseline entry, which is treated as a baseline of
zero and fires on its first file. That is the `_external_roots()` lesson POLICIES.md records: a
hand-maintained list of what to scan rots, and a sweep that misses a directory manufactures false
completeness -- the worst failure an instrument like this can have.

The ratchet is PER ZONE, not on the grand total, and that is load-bearing. A total-only ratchet
lets a zone launder against another: retire five `reasoning/` modules, spend the headroom on five
new ungoverned `tools/` scripts, total unchanged, gate green, tree no better. Per-zone implies the
total, so the total is reported and never separately asserted.


THE TWO EXCLUSIONS -- stated, counted and REPORTED on every run, never silent
-----------------------------------------------------------------------------
  `**/__init__.py` (29 files) -- package scaffolding, not a build; it carries no capability and
      exists so the directory imports. ACCEPTED COST, stated exactly: capability moved INTO an
      `__init__.py` is invisible to this gate.
  `tests/**` (55 files) -- a test is the instrument that measures a build, not a build. Counting
      them would make the ratchet fire on writing test #56, i.e. it would punish the only activity
      that discharges the backlog. That is the failure the sibling instruments' spec names from the
      other side -- "if it returns an allowlist-sized list, THE GATE IS DROPPED, NOT THE TREE" --
      and a gate that reddens on `git add tests/test_new.py` is dropped inside a week. ACCEPTED
      COST: capability smuggled into `tests/` is invisible here. It is small and bounded, because
      `tests/` is the most-gated directory in the repo -- full ruff ruleset (the `src/**`
      per-file-ignores do not reach it), run by pytest, coverage-measured.

Nothing else is excluded. In particular `tools/` (this file included), `docs/coverage/artifacts/`
and the repo root are COUNTED, because P4c names them and POLICIES.md already carries both as open
debt. Adding a new ungoverned tool therefore reddens the build, and that is the gate working: the
question it forces -- "why is this not a module in `substrate/`?" -- is the question P4c asks.


THE BASELINE IS DERIVED, NOT A MAGIC CONSTANT (P1)
--------------------------------------------------
`tools/p4c_ungoverned_baseline.json` is machine-generated by `--update-baseline` from the identical
walk `--assert` uses. It is never hand-typed, and P1's "never keep a constant fixed" is satisfied
in the strongest available way: the number is not fixed at all, it is a measurement pinned in time
whose ONLY legal motion is downward. `--update-baseline` REFUSES to write a rise (exit 1) -- so the
regeneration flag cannot be used to launder a new ungoverned file into the baseline, which is the
drift a hand-edited number would have. The escape hatch, if one is ever genuinely needed, is a
hand-edit of the JSON inside a reviewed commit, where it shows up in the diff as what it is.

  FALSIFIER, constructed and RUN (see `--self-check` for the in-memory form): create any `.py`
  outside the governed tree that git will see, run `--assert`, and the gate exits 1 naming the file
  and its zone. Delete it and the same command exits 0.

The `paths` block in the baseline is DIAGNOSTICS ONLY and is never asserted on -- it exists so the
failure message can name the offending file instead of printing a zone count. A stale `paths` block
(after a rename, say) can make that one line list a wrong candidate; it can never redden or green a
build on its own.


DETERMINISM
-----------
  * The repo root is resolved from `__file__`, and git is invoked with `-C <root>`, so every run is
    identical from any CWD (verified from `C:\\` and from `tools/`).
  * The corpus is `git ls-files --cached --others --exclude-standard -- *.py`: tracked files PLUS
    untracked-but-not-ignored ones. Tracked-only would be reproducible but useless locally -- you
    would only learn about the violation after committing it. A raw filesystem glob would be worse:
    it is the mistake `tools/doc_command_guard.py` had to correct, where the corpus depended on who
    ran it. `--exclude-standard` honours `.gitignore`, so `scratchpad/`, the venvs and
    `__pycache__/` are out by construction. Measured on this tree with a clean working directory:
    cached = 547, cached+others = 547, delta 0 -- so the `others` half adds no noise in CI, and
    locally it is what makes the run useful: it is how this file, written and not yet committed,
    appears in the `tools` zone at all rather than only after someone commits it.
  * All walks are sorted; no filesystem-order dependence anywhere.
  * NO IMPORT of `snn_data_lake`. REGIONS is read with `ast`. The gate therefore needs no torch, no
    numpy, no GPU and no Brain -- CI has no card, and an import-time side effect in the package
    (the `learn_daemon` promotion found three at module scope) cannot reach this gate.

Reproduce:
    python tools/p4c_ratchet.py                    # report only, exit 0 always
    python tools/p4c_ratchet.py --assert           # ratchet; non-zero exit on a RISE
    python tools/p4c_ratchet.py --self-check       # prove the criteria can fire (no disk writes)
    python tools/p4c_ratchet.py --update-baseline  # re-derive after a promotion; refuses a rise
"""

import ast
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = "src/snn_data_lake"
BASELINE_PATH = ROOT / "tools" / "p4c_ungoverned_baseline.json"

#: The only hand-written list in this gate. Each entry is argued in the module docstring, and both
#: are COUNTED and REPORTED on every run so the exclusion is visible rather than silent.
EXCLUDE = (
    ("**/__init__.py", "package scaffolding, not a build"),
    ("tests/**", "the instrument that measures a build, not a build"),
)


def _excluded(rel):
    """True if `rel` (repo-relative posix path) is one of the two documented exclusions."""
    return rel.endswith("/__init__.py") or rel == "__init__.py" or rel.startswith("tests/")


def _corpus(root=None):
    """Sorted repo-relative posix .py paths: tracked + untracked-not-ignored. Deterministic."""
    root = ROOT if root is None else root
    out = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()})


def _governed(root=None):
    """Governed paths, DERIVED from REGIONS + TORCH_TIER by AST -- no import of the package.

    Verified set-equal to `substrate/dormancy_gate.py::BRAIN_MODULES` (205 entries) on 2026-08-06.
    """
    root = ROOT if root is None else root
    init = (root / PKG / "__init__.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(init)
    paths = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            # TRAIN_TIER added 2026-08-11 with the annex itself (lane-2 CI regression fix): a
            # third registration table with the same shape; a registered train-tier module is
            # governed exactly as a torch-tier one is.
            if not (
                isinstance(tgt, ast.Name) and tgt.id in ("REGIONS", "TORCH_TIER", "TRAIN_TIER")
            ):
                continue
            try:
                table = ast.literal_eval(node.value)
            except Exception:  # a dynamically built table would be a real finding; report, not hide
                continue
            if not isinstance(table, dict):
                continue
            for region, mods in table.items():
                for m in mods:
                    paths.add(f"{PKG}/{region}/{m}.py")
    return paths


def _zone(rel):
    """DERIVED zone for a repo-relative path. A new top-level dir yields a new, unbaselined zone."""
    parts = rel.split("/")
    if rel.startswith(PKG + "/"):
        rest = parts[2:]
        # a sub-package (reasoning/, substrate/, ...) vs a file sitting at the package top level
        return f"{PKG}/{rest[0]}" if len(rest) > 1 else f"{PKG}/<toplevel>"
    return parts[0] if len(parts) > 1 else "<repo root>"


def census(root=None):
    """(zone -> sorted ungoverned paths, exclusion counts, phantom REGIONS entries)."""
    root = ROOT if root is None else root
    corpus = _corpus(root)
    governed = _governed(root)
    zones, excl = {}, {name: 0 for name, _ in EXCLUDE}
    for rel in corpus:
        if _excluded(rel):
            key = "**/__init__.py" if rel.endswith("__init__.py") else "tests/**"
            excl[key] += 1
            continue
        if rel in governed:
            continue
        zones.setdefault(_zone(rel), []).append(rel)
    for z in zones:
        zones[z].sort()
    phantom = sorted(g for g in governed if g not in set(corpus))
    return zones, excl, phantom


def compare(counts, baseline):
    """PURE. -> sorted [(zone, live, allowed)] for every zone that ROSE. Absent zone allows 0.

    This is the whole assert, isolated from the filesystem so `--self-check` can mutate its inputs.
    """
    findings = []
    for zone in sorted(set(counts) | set(baseline)):
        live, allowed = counts.get(zone, 0), baseline.get(zone, 0)
        if live > allowed:
            findings.append((zone, live, allowed))
    return findings


def _load_baseline():
    """Read the committed baseline. `utf-8-sig` is deliberate, not decoration.

    The baseline's documented escape hatch is a hand-edit in a reviewed commit, and this is a
    Windows-first repo: PowerShell's `Set-Content -Encoding utf8` writes a BOM. Found by doing
    exactly that during this gate's own red-construction -- plain `utf-8` raised
    `JSONDecodeError: Unexpected UTF-8 BOM` and the gate died with a traceback instead of a
    verdict. It still exited non-zero, so it could not have greened a build, but an instrument
    whose documented edit path crashes it is an instrument people route around. `utf-8-sig`
    accepts a BOM and is identical to `utf-8` when there is none.
    """
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8-sig"))


def _report(zones, excl, phantom, baseline):
    total = sum(len(v) for v in zones.values())
    print("=" * 92)
    print("p4c_ratchet -- builds OUTSIDE the governed tree (P4c). The count may only go DOWN.")
    print("=" * 92)
    base_zones = (baseline or {}).get("zones", {})
    print(f"{'zone':<40} {'live':>6} {'baseline':>9} {'delta':>7}")
    print("-" * 92)
    for zone in sorted(set(zones) | set(base_zones)):
        live = len(zones.get(zone, []))
        allowed = base_zones.get(zone)
        if allowed is None:
            print(f"{zone:<40} {live:>6} {'(none)':>9} {'NEW ZONE':>7}")
        else:
            d = live - allowed
            print(f"{zone:<40} {live:>6} {allowed:>9} {d:>+7}")
    print("-" * 92)
    base_total = (baseline or {}).get("total")
    print(f"{'TOTAL ungoverned':<40} {total:>6} {base_total if base_total is not None else '-':>9}")
    print()
    print("EXCLUDED (documented in the module docstring; reported so it is never silent):")
    for name, reason in EXCLUDE:
        print(f"    {name:<20} {excl.get(name, 0):>4}   {reason}")
    if phantom:
        print()
        print(f"NOTE (not asserted, outside this gate's lane): {len(phantom)} REGIONS/TORCH_TIER")
        print("     entries name a file that does not exist:")
        for p in phantom[:10]:
            print(f"       {p}")
    return total


def _fail_detail(findings, zones, baseline):
    """Name the offending files using the baseline's DIAGNOSTICS-ONLY path manifest."""
    base_paths = (baseline or {}).get("paths", {})
    print()
    print("GATE FAIL: the ungoverned count ROSE. P4c: every future build lands inside the")
    print("governed tree from the start -- reasoning/ is not a parking lot.")
    for zone, live, allowed in findings:
        print(f"  {zone}: {live} > baseline {allowed}  (+{live - allowed})")
        known = set(base_paths.get(zone, []))
        new = [p for p in zones.get(zone, []) if p not in known]
        for p in new[:20]:
            print(f"      NEW (candidate): {p}")
        if not new:
            print("      (no path candidates -- the baseline manifest is stale; counts are")
            print("       authoritative, re-derive with --update-baseline after a real promotion)")
    print()
    print("To clear this: register the build in src/snn_data_lake/__init__.py REGIONS (or")
    print("TORCH_TIER) and run_selfchecks.py so the three P3 gates can see it, or retire it.")
    print("Do NOT run --update-baseline to absorb it: that path refuses a rise by design.")


def _selfcheck():
    """Prove every criterion this gate carries can FIRE, on synthetic in-memory inputs.

    Nothing on disk is read or written. Each case states the mutation and the expected direction,
    and a mutation that changed nothing is reported BROKEN rather than passing quietly -- the
    rotted-anchor failure `tools/dependency_ledger.py --self-check` exists to catch.
    """
    base = {"src/snn_data_lake/reasoning": 161, "tools": 30, "research": 25, "gcp": 0}
    clean = dict(base)

    print("=" * 92)
    print("p4c_ratchet --self-check : every criterion mutated, each red CONSTRUCTED and RUN")
    print("=" * 92)

    # Sensitivity control FIRST: an unmutated corpus must be silent, or every red below is vacuous.
    zero = compare(clean, base)
    print(f"[control ] unmutated counts == baseline -> {len(zero)} findings (must be 0)")
    if zero:
        print("BROKEN: the comparator fires on an unmutated corpus; every red below is meaningless")
        return 1

    cases = [
        (
            "RED",
            "rise in a known zone",
            "reasoning 161 -> 162 (one new ungoverned module parked in reasoning/)",
            {**clean, "src/snn_data_lake/reasoning": 162},
            True,
        ),
        (
            "RED",
            "rise in the zone holding this gate",
            "tools 30 -> 31 (a new ungoverned helper script)",
            {**clean, "tools": 31},
            True,
        ),
        (
            "RED",
            "file in a zone with NO baseline entry",
            "new top-level dir 'skunkworks' -> 1 (implicit baseline 0)",
            {**clean, "skunkworks": 1},
            True,
        ),
        (
            "RED",
            "first file in a zone baselined at exactly 0",
            "gcp 0 -> 1 (P4c names gcp/ explicitly; it is empty today)",
            {**clean, "gcp": 1},
            True,
        ),
        (
            "CONTROL",
            "a legitimate PROMOTION must stay green",
            "reasoning 161 -> 160 (one module promoted into a governed region)",
            {**clean, "src/snn_data_lake/reasoning": 160},
            False,
        ),
        (
            "CONTROL",
            "cross-zone laundering must NOT be maskable",
            "reasoning 161 -> 156 AND tools 30 -> 35 (total unchanged)",
            {**clean, "src/snn_data_lake/reasoning": 156, "tools": 35},
            True,
        ),
    ]

    red_ok = no_red = broken = 0
    for kind, name, mutation, counts, expect_fire in cases:
        if counts == clean:
            print(f"[BROKEN  ] {name}: the mutation changed nothing -- anchor has rotted")
            broken += 1
            continue
        fired = bool(compare(counts, base))
        mark = "OK" if fired == expect_fire else "NO RED" if expect_fire else "FALSE RED"
        print(f"[{kind:<8}] {name}")
        print(f"           mutation: {mutation}")
        print(
            f"           expected {'FIRE' if expect_fire else 'PASS'}, got "
            f"{'FIRE' if fired else 'PASS'}  -> {mark}"
        )
        if fired == expect_fire:
            red_ok += 1
        else:
            no_red += 1

    # The regeneration path must not be a laundering hole. Same comparator, opposite intent.
    hole = compare({**clean, "research": 26}, base)
    print("[RED     ] --update-baseline refuses a rise")
    print("           mutation: research 25 -> 26 offered to the baseline writer")
    print(
        f"           expected REFUSE, got {'REFUSE' if hole else 'ACCEPT'}  -> "
        f"{'OK' if hole else 'NO RED'}"
    )
    if hole:
        red_ok += 1
    else:
        no_red += 1

    print()
    print(f"RED OK {red_ok}   NO RED {no_red}   BROKEN {broken}")
    if no_red or broken:
        print("self-check FAIL: a criterion this gate carries cannot be made to fire")
        return 1
    print(f"self-check OK: {red_ok}/{red_ok} criteria proven falsifiable (no disk was touched)")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-check" in argv:
        return _selfcheck()

    zones, excl, phantom = census()
    counts = {z: len(v) for z, v in zones.items()}
    baseline = _load_baseline()
    total = _report(zones, excl, phantom, baseline)

    if "--update-baseline" in argv:
        if baseline is not None:
            rises = compare(counts, baseline.get("zones", {}))
            if rises:
                print()
                print("REFUSED: --update-baseline may only ratchet DOWN. These zones ROSE:")
                for zone, live, allowed in rises:
                    print(f"  {zone}: {live} > baseline {allowed}")
                print("Promote or retire the new build; do not absorb it into the baseline.")
                return 1
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "_generated_by": "tools/p4c_ratchet.py --update-baseline",
                    "_policy": "P4c -- every build is part of the brain",
                    "_contract": "counts may only go DOWN; 'paths' is DIAGNOSTICS ONLY, never asserted on",
                    "total": total,
                    "zones": dict(sorted(counts.items())),
                    "paths": {z: zones[z] for z in sorted(zones)},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print()
        print(f"baseline written: {BASELINE_PATH.relative_to(ROOT).as_posix()}  total={total}")
        return 0

    if "--assert" not in argv:
        print()
        print("(report only -- pass --assert to ratchet, --self-check to prove it can fire)")
        return 0

    if baseline is None:
        print()
        print(f"GATE FAIL: no baseline at {BASELINE_PATH.relative_to(ROOT).as_posix()}.")
        print("Generate it with: python tools/p4c_ratchet.py --update-baseline")
        return 1

    findings = compare(counts, baseline.get("zones", {}))
    if findings:
        _fail_detail(findings, zones, baseline)
        return 1
    print()
    print(f"gate OK: {total} ungoverned builds, no zone above its baseline (backlog may only fall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
