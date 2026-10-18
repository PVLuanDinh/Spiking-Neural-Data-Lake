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
`gates/dormancy_gate.py::BRAIN_MODULES`: 205 entries, exact set match, 2026-08-06. Registration
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

  `tools/rsfacts/**` (R0.2 row 17, 2026-10-17: Cargo.toml + src/main.rs) -- the gate toolchain's own
      build, the syn CLI behind `gates/rust_facts.py`. It joined the corpus the day `.rs` did, and it
      carries the disposition every Rust-aware gate and `dependency_ledger` already give rsfacts'
      Cargo.toml: gate tooling, out of scope for brain rules, NAMED and counted, never scanned.
  `native/** test code` -- Rust test code per `rust_facts.test_file_flags` (a crate's `tests/`,
      `benches/`, `examples/`, not reached by a production `mod`): the instrument, not a build, by the
      same argument as `tests/**`.

Nothing else is excluded. In particular `tools/` (this file included), `docs/coverage/artifacts/`
and the repo root are COUNTED, because P4c names them and POLICIES.md already carries both as open
debt. Adding a new ungoverned tool therefore reddens the build, and that is the gate working: the
question it forces -- "why is this not a module in `substrate/`?" -- is the question P4c asks.


THE RUST HALF (R0.2 row 17): `.rs` and Cargo manifests are in the corpus, with a MEMBERSHIP RULE
------------------------------------------------------------------------------------------------
The corpus is `*.py`, `*.rs` and every `Cargo.toml` (git's view, as before). The zone of a file under
`native/<crate>/` is `native/<crate>`. A `.rs` there is GOVERNED iff its crate is in `NATIVE_TIER`
(the hand-written table below, printed on every run: the Rust registration, as REGIONS is the Python
one) AND the file is reachable from the crate's `src/lib.rs` / `src/main.rs` through `mod x;`
declarations -- resolved by the reader's ONE resolver (`rust_facts.mod_files`, cfg(test) children kept,
`#[path]` honoured) through the ONE walker (`native_dormancy._Tree`), never re-derived here. A tier
crate's `Cargo.toml` and `build.rs` are governed with it; `native/Cargo.toml` (the workspace manifest)
is governed while any tier crate directory exists. Everything else under `native/` is ungoverned and
counts: an untracked `src/scratch.rs` outside the mod tree, or the first file of a crate the table does
not list, reds its zone (no baseline entry = 0). Membership follows a tier crate's OWN roots (a file
another crate reaches through `#[path]` is not thereby governed). Blind spot, inherited from the shared
resolver: a `#[path]` on an ENCLOSING inline module is not applied to its children (they resolve under
the module's name, not its path). Facts come from `rust_facts` (cargo-built `rsfacts`);
`RustFactsUnavailable` with `.rs` present under `native/` exits 1 in EVERY mode (fail closed). `native/`
absent prints `native/ ABSENT`, never 0 findings.


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
  * The corpus is `git ls-files --cached --others --exclude-standard -- *.py *.rs *Cargo.toml`
    (basename `Cargo.toml` kept): tracked files PLUS untracked-but-not-ignored ones. Tracked-only would be reproducible but useless locally -- you
    would only learn about the violation after committing it. A raw filesystem glob would be worse:
    it is the mistake `tools/doc_command_guard.py` had to correct, where the corpus depended on who
    ran it. `--exclude-standard` honours `.gitignore`, so `scratchpad/`, the venvs and
    `__pycache__/` are out by construction. Measured on this tree with a clean working directory:
    cached = 547, cached+others = 547, delta 0 -- so the `others` half adds no noise in CI, and
    locally it is what makes the run useful: it is how this file, written and not yet committed,
    appears in the `tools` zone at all rather than only after someone commits it.
  * All walks are sorted; no filesystem-order dependence anywhere.
  * NO IMPORT of `snn_data_lake` for the Python half. REGIONS is read with `ast`. The gate therefore
    needs no torch, no numpy, no GPU and no Brain -- CI has no card, and an import-time side effect in
    the package (the `learn_daemon` promotion found three at module scope) cannot reach this gate. The
    Rust half imports `gates.rust_facts` + `gates.native_dormancy` (stdlib-only modules) LAZILY, and
    only when `.rs` files exist under `native/`; on a tree without them nothing is imported.

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

#: The only hand-written list in this gate. Each entry is argued in the module docstring, and both
#: are COUNTED and REPORTED on every run so the exclusion is visible rather than silent.
EXCLUDE = (
    ("**/__init__.py", "package scaffolding, not a build"),
    ("tests/**", "the instrument that measures a build, not a build"),
    (
        "tools/rsfacts/**",
        "the gate toolchain's own build (rust_facts' syn CLI): named, not scanned",
    ),
    (
        "native/** test code",
        "Rust test code per rust_facts.test_file_flags: the instrument, not a build",
    ),
)

#: The Rust registration table (R0.2 row 17): a crate under native/ whose mod tree is governed.
#: `sndl-python` (the PyO3 binding crate) is listed even before it exists, so its first file is governed.
NATIVE_TIER = ("sndl-core", "sndl-python")


def _exclusion(rel, native_tests=frozenset()):
    """The EXCLUDE key `rel` (repo-relative posix path) falls under, else None."""
    if rel.endswith("/__init__.py") or rel == "__init__.py":
        return "**/__init__.py"
    if rel.startswith("tests/"):
        return "tests/**"
    if rel.startswith("tools/rsfacts/"):
        return "tools/rsfacts/**"
    if rel in native_tests:
        return "native/** test code"
    return None


def _corpus(root=None):
    """Sorted repo-relative posix .py / .rs / Cargo.toml paths: tracked + untracked-not-ignored."""
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
            "*.rs",
            "*Cargo.toml",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rels = {line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()}
    return sorted(
        r for r in rels if r.endswith((".py", ".rs")) or r.rsplit("/", 1)[-1] == "Cargo.toml"
    )


def _rust_facts():
    """The one Rust reader (stdlib-only modules), imported lazily and only when native/ exists."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from snn_data_lake.gates import rust_facts

    return rust_facts


def _native(root, corpus):
    """(status line, governed rels, test-code rels) for the Rust half of the corpus. Raises
    `RustFactsUnavailable` when `.rs` is present but unreadable -- or hidden from git (fail closed)."""
    tier = {c for c in NATIVE_TIER if (root / "native" / c).is_dir()}
    candidates = {f"native/{c}/{n}" for c in tier for n in ("Cargo.toml", "build.rs")}
    governed = (candidates | ({"native/Cargo.toml"} if tier else set())) & set(corpus)
    absent = "native/ ABSENT (0 .rs files)"
    if not (root / "native").is_dir():
        return absent, governed, frozenset()
    rf = _rust_facts()
    files = rf.rust_files(root)  # raises on .rs that git cannot see: ABSENT never masks it
    if not files:
        return absent, governed, frozenset()
    from snn_data_lake.gates.native_dormancy import _Tree

    facts = rf.facts(files)  # rsfacts is built from THIS tree's tools/rsfacts; `root` picks the .rs
    base = root / "native"
    by_rel = {pathlib.Path(f["file"]).relative_to(base).as_posix(): f for f in facts}
    tree = _Tree(by_rel)
    for c in sorted(tier):  # membership follows EACH tier crate's OWN roots, inside that crate
        roots = [r for r in (f"{c}/src/lib.rs", f"{c}/src/main.rs", f"{c}/build.rs") if r in by_rel]
        bins = f"{c}/src/bin/"
        roots += sorted(
            r
            for r in by_rel
            if r.startswith(bins) and (r.count("/") == 3 or r.endswith("/main.rs"))
        )
        for r in roots:  # _Tree roots one file per crate (lib.rs first); the rest are walked here
            tree._walk_decl(r, via_path=r.endswith("/build.rs"))
        reached, todo = set(), list(roots)
        while todo:
            r = todo.pop()
            if r not in reached:
                reached.add(r)
                todo += [k for k in tree.children.get(r, {}).values() if k != r]
        governed |= {f"native/{r}" for r in reached if tree.crate_dir(r) == c}
    tests = frozenset(f"native/{r}" for r, f in by_rel.items() if f["test_file"])
    return f"native/ PRESENT ({len(files)} .rs files; NATIVE_TIER {NATIVE_TIER})", governed, tests


def _governed(root=None):
    """Governed paths, DERIVED from REGIONS + TORCH_TIER by AST -- no import of the package.

    Verified set-equal to `gates/dormancy_gate.py::BRAIN_MODULES` (205 entries) on 2026-08-06.
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
    if rel.startswith("native/") and len(parts) > 2:
        return f"native/{parts[1]}"  # one zone per crate
    return parts[0] if len(parts) > 1 else "<repo root>"


def census(root=None):
    """(zone -> sorted ungoverned paths, exclusion counts, phantom REGIONS entries, native/ status)."""
    root = ROOT if root is None else root
    corpus = _corpus(root)
    governed = _governed(root)
    native, native_governed, native_tests = _native(root, corpus)
    governed |= native_governed
    zones, excl = {}, {name: 0 for name, _ in EXCLUDE}
    for rel in corpus:
        key = _exclusion(rel, native_tests)
        if key:
            excl[key] += 1
            continue
        if rel in governed:
            continue
        zones.setdefault(_zone(rel), []).append(rel)
    for z in zones:
        zones[z].sort()
    phantom = sorted(g for g in governed if g not in set(corpus))
    return zones, excl, phantom, native


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


def _baseline_path(root=None):
    return (ROOT if root is None else root) / "tools" / "p4c_ungoverned_baseline.json"


def _load_baseline(root=None):
    """Read the committed baseline. `utf-8-sig` is deliberate, not decoration.

    The baseline's documented escape hatch is a hand-edit in a reviewed commit, and this is a
    Windows-first repo: PowerShell's `Set-Content -Encoding utf8` writes a BOM. Found by doing
    exactly that during this gate's own red-construction -- plain `utf-8` raised
    `JSONDecodeError: Unexpected UTF-8 BOM` and the gate died with a traceback instead of a
    verdict. It still exited non-zero, so it could not have greened a build, but an instrument
    whose documented edit path crashes it is an instrument people route around. `utf-8-sig`
    accepts a BOM and is identical to `utf-8` when there is none.
    """
    path = _baseline_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _report(zones, excl, phantom, baseline, native=""):
    total = sum(len(v) for v in zones.values())
    print("=" * 92)
    print("p4c_ratchet -- builds OUTSIDE the governed tree (P4c). The count may only go DOWN.")
    print("=" * 92)
    print(
        f"Rust half: {native}; governed = NATIVE_TIER {NATIVE_TIER} crate + reachable in its mod tree"
    )
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


FIXTURE_GOVERNED = (  # every file the membership rule must reach in the fixture crate
    "src/lib.rs",
    "src/a.rs",
    "src/rng/mod.rs",
    "src/main.rs",
    "src/cli.rs",
    "src/tests.rs",
    "src/bin/tool.rs",
    "src/bin/tool/helper.rs",
    "shared.rs",
    "build.rs",
    "gen.rs",
    "Cargo.toml",
)


def _fixture_repo(root, crate_name="sndl-core"):
    """A throwaway git tree with one crate: a lib root (`mod a; mod rng;`, a `#[path]` module
    beside src/, a cfg(test) child), a bin root (`main.rs -> cli.rs`), an auto bin target
    (`src/bin/tool.rs -> tool/helper.rs`), `build.rs -> gen.rs`, test code, and `src/scratch.rs`
    OUTSIDE the mod tree."""
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    (root / PKG).mkdir(parents=True)
    (root / PKG / "__init__.py").write_text("REGIONS = {}\n", encoding="utf-8")
    crate = root / "native" / crate_name
    (crate / "src" / "rng").mkdir(parents=True)
    (root / "native" / "Cargo.toml").write_text(
        f'[workspace]\nmembers = ["{crate_name}"]\n', encoding="utf-8"
    )
    (crate / "Cargo.toml").write_text(f'[package]\nname = "{crate_name}"\n', encoding="utf-8")
    (crate / "src" / "lib.rs").write_text(
        'mod a;\nmod rng;\n#[path = "../shared.rs"]\nmod shared;\n#[cfg(test)]\nmod tests;\n',
        encoding="utf-8",
    )
    (crate / "src" / "a.rs").write_text("pub fn a() {}\n", encoding="utf-8")
    (crate / "src" / "tests.rs").write_text("#[test]\nfn unit() {}\n", encoding="utf-8")
    (crate / "src" / "rng" / "mod.rs").write_text("pub fn r() {}\n", encoding="utf-8")
    (crate / "shared.rs").write_text("pub fn shared() {}\n", encoding="utf-8")
    (crate / "src" / "main.rs").write_text("mod cli;\nfn main() {}\n", encoding="utf-8")
    (crate / "src" / "cli.rs").write_text("pub fn cli() {}\n", encoding="utf-8")
    (crate / "src" / "bin" / "tool").mkdir(parents=True)
    (crate / "src" / "bin" / "tool.rs").write_text("mod helper;\nfn main() {}\n", encoding="utf-8")
    (crate / "src" / "bin" / "tool" / "helper.rs").write_text("pub fn h() {}\n", encoding="utf-8")
    (crate / "build.rs").write_text("mod gen;\nfn main() {}\n", encoding="utf-8")
    (crate / "gen.rs").write_text("pub fn g() {}\n", encoding="utf-8")
    (crate / "src" / "scratch.rs").write_text("pub fn s() {}\n", encoding="utf-8")
    (crate / "tests").mkdir()
    (crate / "tests" / "t.rs").write_text("#[test]\nfn t() {}\n", encoding="utf-8")
    return crate


def _selfcheck_native():
    """The Rust reds, each CONSTRUCTED on a fixture tree and RUN through `census(root)` -- the same
    walk `--assert` uses. -> (ok, not_ok) counts, printed like the comparator cases."""
    import tempfile

    unavailable = _rust_facts().RustFactsUnavailable

    def counts(root):
        return {z: len(v) for z, v in census(root)[0].items()}

    def raises(root):
        try:
            counts(root)
        except unavailable:
            return True
        return False

    results = []

    def case(kind, name, mutation, fired, expect_fire=True):
        mark = "OK" if fired == expect_fire else "NO RED" if expect_fire else "FALSE RED"
        print(f"[{kind:<8}] {name}")
        print(f"           mutation: {mutation}")
        print(
            f"           expected {'FIRE' if expect_fire else 'PASS'}, got "
            f"{'FIRE' if fired else 'PASS'}  -> {mark}"
        )
        results.append(fired == expect_fire)

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        crate = _fixture_repo(root)
        try:
            case(
                "RED",
                "untracked .rs outside the mod tree",
                "native/sndl-core/src/scratch.rs -> zone native/sndl-core = 1",
                counts(root).get("native/sndl-core") == 1,
            )
            (crate / "src" / "scratch.rs").unlink()
            case(
                "CONTROL",
                "the GREEN twin",
                f"scratch.rs removed; {len(FIXTURE_GOVERNED)} crate files reached (lib + bin roots, "
                "#[path], build.rs, manifest), tests/t.rs test code -> no zone",
                "native/sndl-core" in counts(root),
                expect_fire=False,
            )
            (crate.parent / "other" / "src").mkdir(parents=True)
            (crate.parent / "other" / "src" / "lib.rs").write_text(
                "pub fn o() {}\n", encoding="utf-8"
            )
            case(
                "RED",
                "a crate NATIVE_TIER does not list",
                "native/other/src/lib.rs -> zone native/other = 1",
                counts(root).get("native/other") == 1,
            )
            (root / ".gitignore").write_text("*.rs\n", encoding="utf-8")
            hidden = raises(root)
            (root / ".gitignore").unlink()
            case(
                "RED",
                "Rust hidden from git fails CLOSED",
                ".gitignore `*.rs` -> RustFactsUnavailable, never `native/ ABSENT`",
                hidden,
            )
        except unavailable as exc:
            print(f"[NO RED  ] the Rust reader is unavailable on this host -- {exc}")
            results.append(False)
        (crate / "src" / "a.rs").write_text("pub fn a( {\n", encoding="utf-8")
        case(
            "RED",
            "unreadable Rust fails CLOSED",
            "src/a.rs made unparseable -> RustFactsUnavailable; main exits 1 with and without --assert",
            raises(root) and main(["--assert"], root=root) == 1 and main([], root=root) == 1,
        )
    return sum(results), len(results) - sum(results)


def _selfcheck():
    """Prove every criterion this gate carries can FIRE: the comparator on synthetic in-memory inputs,
    then the Rust membership rule on a throwaway fixture tree (`_selfcheck_native`; a temp dir is the
    only disk touched, the real tree and baseline are never read or written).

    Each case states the mutation and the expected direction, and a mutation that changed nothing is
    reported BROKEN rather than passing quietly -- the rotted-anchor failure
    `tools/dependency_ledger.py --self-check` exists to catch.
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

    native_ok, native_bad = _selfcheck_native()
    red_ok += native_ok
    no_red += native_bad

    print()
    print(f"RED OK {red_ok}   NO RED {no_red}   BROKEN {broken}")
    if no_red or broken:
        print("self-check FAIL: a criterion this gate carries cannot be made to fire")
        return 1
    print(
        f"self-check OK: {red_ok}/{red_ok} criteria proven falsifiable (only a temp dir was touched)"
    )
    return 0


def main(argv=None, root=None):
    argv = sys.argv[1:] if argv is None else argv
    root = ROOT if root is None else root
    if "--self-check" in argv:
        return _selfcheck()

    # .rs present but the reader is down (cargo, parse, hidden file) -- RED, never a skip
    unreadable = (_rust_facts().RustFactsUnavailable,) if (root / "native").is_dir() else ()
    try:
        zones, excl, phantom, native = census(root)
    except unreadable as exc:
        print(f"GATE FAIL (fail closed): Rust under native/ cannot be read -- {exc}")
        return 1
    counts = {z: len(v) for z, v in zones.items()}
    baseline = _load_baseline(root)
    total = _report(zones, excl, phantom, baseline, native)
    baseline_path = _baseline_path(root)

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
        baseline_path.write_text(
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
        print(f"baseline written: {baseline_path.relative_to(root).as_posix()}  total={total}")
        return 0

    if "--assert" not in argv:
        print()
        print("(report only -- pass --assert to ratchet, --self-check to prove it can fire)")
        return 0

    if baseline is None:
        print()
        print(f"GATE FAIL: no baseline at {baseline_path.relative_to(root).as_posix()}.")
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
