#!/usr/bin/env python3
"""promotion_sweep -- classify every module OUTSIDE the governed tier, against every policy P0-P8.

Owner directive 2026-08-04: every file in this project must be usable and promoted into main `src/`
the moment its build lands. `reasoning/` (inside src/ but outside all three P3 gates), plus
`research/`, `training/` and `integrations/`, are where dormancy hides -- they are ruff-excluded,
outside the coverage denominator, and invisible to `dormancy_gate` / `method_dormancy_gate` /
`import_manifest`.

THIS SCRIPT PROMOTES NOTHING. It counts first, because POLICIES.md's standing invariant is
"implement the gate before adopting the rule ... its output counted, before the policy lands", and
three prior proposals died at exactly that bar (0 failures / 47 UNCERTAIN; 58 failures where 1 was
claimed; 474 violations across 126 files). A blind 227-module promotion would BE that failure: it
would force 227 P3 dispositions at once and the result would be an allowlist, not a work list.

WHAT IT CLASSIFIES, one column per policy that bears on a promotion:

  LOAD-BEARING (P3 cross-tier)  is this module imported by a GOVERNED module? `import_manifest`
                                already proves the governed tiers reach into these through bare
                                `except Exception:` swallows. A load-bearing module is not optional
                                -- deleting it breaks the brain while every gate stays green.
  SELF-CHECK (P3)               does it have a `main()`? `run_selfchecks` can only execute modules
                                that have one, so this decides whether promotion is even mechanical.
  DUP RISK (P2)                 does a governed module already share its stem or its declared job?
                                P2: exactly one mechanism survives, so a promotion that duplicates
                                must ADOPT or MERGE, never land additively.
  ZERO-DEP DEBT (P7)            does it advertise stdlib purity, or hand-roll what a declared
                                dependency does? Promotion is a REDESIGN, not a move.
  CONSTANTS (P1)                count of numeric literals in comparison operators outside asserts
                                and `main()` -- a cheap proxy for magic constants that arbitrate.
  TIER (P5)                     does it import torch / require CUDA? Decides train-tier vs core,
                                and the suite must stay runnable without a GPU.
  BACKPROP (e-prop invariant)   does it call `.backward()` or name BPTT? Those may be promoted but
                                may NOT enter a live learning path.

Reproduce:
    python tools/promotion_sweep.py            # full report
    python tools/promotion_sweep.py --summary  # counts only
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "snn_data_lake"

#: Directories outside the repo's package tree. These are genuinely separate top-level dirs.
OUT_OF_TREE_DIRS = ("research", "training", "integrations", "gcp", "infra")


def _out_of_gate_dirs():
    """src/ subpackages that are NOT declared regions -- DERIVED, never hand-listed.

    The first version of this file hardcoded `("reasoning",)` and silently missed `lakehouse` (3
    modules) and `src/snn_data_lake/training` (8), so it reported 216 when the real figure was
    higher. That is the `_external_roots()` lesson POLICIES.md already records: a hand-maintained
    list of what to scan rots, and a sweep that misses a directory manufactures false completeness
    -- the worst failure an instrument like this can have, because the gap looks like a clean bill.

    Deriving it means a new ungoverned subpackage shows up here the day it is created.
    """
    init = (SRC / "__init__.py").read_text(encoding="utf-8", errors="replace")
    declared = set(re.findall(r'^\s*"([a-z_]+)":\s*\[', init, re.M))
    out = []
    for p in sorted(SRC.iterdir()):
        if not p.is_dir() or p.name.startswith("__"):
            continue
        if p.name not in declared:
            out.append(p.name)
    return tuple(out)


ZERO_DEP_MARKERS = (
    "pure standard library",
    "pure-stdlib",
    "stdlib only",
    "stdlib-only",
    "zero-dep",
    "zero dependency",
    "zero-dependency",
    "no numpy",
    "without numpy",
    "no tensors",
)

BACKPROP_MARKERS = (".backward(", "bptt", "backprop", "autograd")


def _governed():
    """(regions_map, governed_stems) from __init__.py REGIONS + TORCH_TIER."""
    init = (SRC / "__init__.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(init)
    regions, stems = {}, set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not (isinstance(tgt, ast.Name) and tgt.id in ("REGIONS", "TORCH_TIER")):
                continue
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(val, dict):
                for region, mods in val.items():
                    for m in mods:
                        regions[m] = region
                        stems.add(m)
            elif isinstance(val, (list, tuple, set)):
                for m in val:
                    if isinstance(m, str):
                        stems.add(m)
    return regions, stems


def _iter_targets():
    for d in _out_of_gate_dirs():
        base = SRC / d
        if base.is_dir():
            for p in sorted(base.rglob("*.py")):
                if p.name != "__init__.py":
                    yield p, f"src/{d}"
    for d in OUT_OF_TREE_DIRS:
        base = ROOT / d
        if base.is_dir():
            for p in sorted(base.rglob("*.py")):
                if p.name != "__init__.py":
                    yield p, d


def _governed_files():
    """Every .py under a governed region dir (i.e. NOT reasoning/)."""
    out = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC)
        if rel.parts and rel.parts[0] in _out_of_gate_dirs():
            continue
        out.append(p)
    return out


def _importers_of(stems, governed_files):
    """stem -> [governed file:line] that import it. Includes lazy/in-function imports."""
    hits = {s: [] for s in stems}
    pat = re.compile(
        r"(?:from|import)\s+[\w.]*\b(" + "|".join(re.escape(s) for s in stems) + r")\b"
    )
    for p in governed_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            ls = line.strip()
            if not (ls.startswith("from ") or ls.startswith("import ") or " import " in ls):
                continue
            m = pat.search(line)
            if m:
                hits[m.group(1)].append(f"{p.relative_to(ROOT).as_posix()}:{i}")
    return hits


def _magic_constants(tree):
    """Numeric literals in comparisons, excluding asserts and main()/__main__ bodies."""
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            skip.update(id(x) for x in ast.walk(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            skip.update(id(x) for x in ast.walk(node))
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            lhs = node.test.left
            if isinstance(lhs, ast.Name) and lhs.id == "__name__":
                skip.update(id(x) for x in ast.walk(node))
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and id(node) not in skip:
            for c in [node.left, *node.comparators]:
                if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)):
                    if c.value not in (0, 1, -1, 0.0, 1.0, 2):  # structural, not calibration
                        n += 1
    return n


def classify():
    regions, governed_stems = _governed()
    targets = list(_iter_targets())
    stems = {p.stem for p, _ in targets}
    gfiles = _governed_files()
    importers = _importers_of(stems, gfiles)

    rows = []
    for path, where in targets:
        try:
            txt = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(txt)
        except Exception:
            continue
        low = txt.lower()
        has_main = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"
            for n in ast.walk(tree)
        )
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "stem": path.stem,
                "where": where,
                "load_bearing": importers.get(path.stem, []),
                "has_main": has_main,
                "dup": path.stem in governed_stems,
                "zero_dep": [m for m in ZERO_DEP_MARKERS if m in low],
                "constants": _magic_constants(tree),
                "torch": ("import torch" in low or "from torch" in low),
                "cuda": ("cuda" in low),
                "backprop": [m for m in BACKPROP_MARKERS if m in low],
                "loc": txt.count("\n") + 1,
            }
        )
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rows = classify()

    lb = [r for r in rows if r["load_bearing"]]
    dup = [r for r in rows if r["dup"]]
    zd = [r for r in rows if r["zero_dep"]]
    bp = [r for r in rows if r["backprop"]]
    no_main = [r for r in rows if not r["has_main"]]
    torch_t = [r for r in rows if r["torch"]]

    print("=" * 86)
    print("promotion_sweep -- every module outside the governed tier, against P0-P8")
    print("=" * 86)
    by_where = {}
    for r in rows:
        by_where[r["where"]] = by_where.get(r["where"], 0) + 1
    print(f"total out-of-gate modules: {len(rows)}")
    print(
        "REMINDER (docs/POLICIES.md, e-prop SCOPE): promoting a module holding a "
        ".backward()+.step() hazard symbol triggers the reachability-hole revisit."
    )
    for k in sorted(by_where):
        print(f"    {k:<16} {by_where[k]}")
    print()
    print(
        f"  LOAD-BEARING (imported by governed code)  {len(lb):>4}   <- promotion is NOT optional"
    )
    print(f"  no main() -- cannot register as a check    {len(no_main):>4}   <- needs one written")
    print(f"  P2 DUP RISK (stem exists in governed tier) {len(dup):>4}")
    print(f"  P7 zero-dep debt (advertises purity)       {len(zd):>4}")
    print(f"  P5 torch tier                              {len(torch_t):>4}")
    print(
        f"  e-prop: names backprop/BPTT/.backward()    {len(bp):>4}   <- may NOT enter live learning"
    )
    print(f"  total LOC to bring under lint+coverage     {sum(r['loc'] for r in rows):>6}")
    print()

    if "--summary" in argv:
        return 0

    print("--- LOAD-BEARING: governed code already imports these (promote FIRST) ---")
    for r in sorted(lb, key=lambda r: -len(r["load_bearing"])):
        flags = []
        if r["dup"]:
            flags.append("P2-DUP")
        if r["zero_dep"]:
            flags.append("P7-DEBT")
        if r["backprop"]:
            flags.append("EPROP")
        if not r["has_main"]:
            flags.append("NO-MAIN")
        print(f"  {r['path']:<62} {len(r['load_bearing'])} importer(s)  {' '.join(flags)}")
        for site in r["load_bearing"][:3]:
            print(f"        <- {site}")
    print()

    print("--- P2 DUPLICATION RISK: a governed module already has this stem ---")
    for r in dup:
        print(f"  {r['path']:<62} stem '{r['stem']}' also governed")
    if not dup:
        print("  (none)")
    print()

    print("--- e-PROP HAZARD: promoting these must not put backprop on a live path ---")
    for r in bp[:15]:
        print(f"  {r['path']:<62} {','.join(r['backprop'])}")
    print()

    print("--- TOP P1 RISK: most numeric literals in comparisons (magic-constant proxy) ---")
    for r in sorted(rows, key=lambda r: -r["constants"])[:10]:
        print(f"  {r['path']:<62} {r['constants']} literal comparisons")
    print()
    print("NOTE: this script promotes NOTHING. It is the count that decides the plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
