"""P5b gate -- every native Rust module cites a pre-registration recording all six P5b criteria with numbers.

Rule enforced: docs/POLICIES.md P5 "P5b" (Gate specification): "a pre-registration check that every module under
the native crate cites a pre-registration recording all six criteria with their numbers, with a planted-missing
red." Template for shape: gates/null_scale_gate.py (Red records, check(root=...), --assert / --self-check /
--census, a closed GRANDFATHERED table).

# distinct-from: rust_facts.py -- reused here for the one `.rs` walk (`rust_files`); native_dormancy.py -- its
# `_write` builds the constructed trees; null_scale_gate.py -- same CLI shape, different corpus (this reads
# `//!` headers and the cited record, that reads evidence-file NULL-RECORD lines).

A module (a `.rs` under a crate's `src/`) cites its pre-registration with exactly one line in its leading `//!`
block:  //! P5b-PREREG: <repo-relative path>
The cited file (any repo file) records, one line each (k = 1..6), then names every module citing it:

    P5b-<k> <label>: <VERDICT> -- <evidence>     labels: hot, python-bound, coarse-boundary, charged,
                                                  python-fix-first, parity-oracle
    P5b-MODULES: <native-relative path>, ...

What it reds: R-UNCITED, R-DUP-CITE, R-MISSING, R-CRITERION, R-NONUMBER, R-UNMET, R-UNLISTED, R-STALE-GRANDFATHER,
R-GRANDFATHER-OPEN, R-HISTORY (and R-ZERO / R-RUSTFACTS: a present `native/` that yields no modules or cannot be
listed). What it cannot do, and says so on every run: judge whether a criterion's evidence is TRUE -- a digit
and a MET verdict are what it counts; the pre-registration's reviewer decides the rest.

  python -m snn_data_lake.gates.p5b_gate [--assert | --self-check | --census]

Bare invocation is --assert (the run_selfchecks.py convention).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from snn_data_lake.gates import native_dormancy, rust_facts

REPO = Path(__file__).resolve().parents[3]

NATIVE = "native"
CUTOFF = "20e1e742f6d1d9deb19dee0b443e6737592a1200"  # origin/main the day P5b's gate was specified
RETRO = "specs/022-unified-brain-arc/evidence/p5b-retroactive-record.md"
LABELS = {
    1: "hot",
    2: "python-bound",
    3: "coarse-boundary",
    4: "charged",
    5: "python-fix-first",
    6: "parity-oracle",
}
CITE_RE = re.compile(r"^//!\s*P5b-PREREG:\s*(\S+)\s*$")
CRITERION_RE = re.compile(r"^\s*(?:[-*]\s+)?P5b-([1-9])\s+([a-z-]+):\s*(\S+)\s*--\s*(.*?)\s*$")
MODULES_RE = re.compile(r"^\s*(?:[-*]\s+)?P5b-MODULES:\s*(.*?)\s*$")
PATH_RE = re.compile(r"[\w./-]+\.md")

# Shrink-only: the eleven native modules that landed before the gate (the 20e1e742 cutoff), recorded
# retroactively. An entry leaves in the same commit its module gains a real `//! P5b-PREREG:` citation or is
# deleted; a module added after the cutoff can never enter (R-GRANDFATHER-OPEN).
_WHY = f"landed before P5b's gate; recorded retroactively in {RETRO}"
GRANDFATHERED: dict[str, str] = {
    rel: _WHY
    for rel in (
        "sndl-core/src/lib.rs",
        "sndl-core/src/prefrontal_executive.rs",
        "sndl-core/src/rng/mod.rs",
        "sndl-core/src/rng/mt19937.rs",
        "sndl-core/src/rng/pyrandom.rs",
        "sndl-core/src/rng/lcg.rs",
        "sndl-core/src/rng/legacy.rs",
        "sndl-core/src/rng/parity.rs",
        "sndl-core/src/rng/pcg64.rs",
        "sndl-core/src/rng/seedseq.rs",
        "sndl-python/src/lib.rs",
    )
}


@dataclass(frozen=True)
class Red:
    code: str
    where: str
    message: str


@dataclass
class Report:
    reds: list[Red] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    absent: bool = False

    def codes(self) -> set[str]:
        return {r.code for r in self.reds}


def cutoff_modules(root: Path = REPO) -> set[str] | None:
    """Native-relative paths of every file under native/ at the cutoff commit. None when git cannot answer
    (no git, unknown commit, shallow clone): the caller reds R-HISTORY, never skips."""
    try:
        out = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", CUTOFF, "--", f"{NATIVE}/"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode:
        return None
    return {
        line[len(NATIVE) + 1 :] for line in out.stdout.splitlines() if line.startswith(f"{NATIVE}/")
    }


def citations(text: str) -> list[str]:
    """The `//! P5b-PREREG:` paths in the file's leading `//!` block."""
    found = []
    for line in text.splitlines():
        if not line.strip().startswith("//!"):
            break
        if m := CITE_RE.match(line.strip()):
            found.append(m.group(1))
    return found


def _record(path: Path) -> tuple[dict[int, list[tuple[str, str]]], set[str]]:
    """{k: [(verdict, evidence)]} for the six criterion lines, and the module names the record lists."""
    criteria: dict[int, list[tuple[str, str]]] = {}
    listed: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := CRITERION_RE.match(line):
            k = int(m.group(1))
            if k in LABELS and LABELS[k] == m.group(2):
                criteria.setdefault(k, []).append((m.group(3), m.group(4)))
        elif m := MODULES_RE.match(line):
            listed |= {p.strip() for p in m.group(1).split(",") if p.strip()}
    return criteria, listed


def _check_record(root: Path, rel: str, cited: str) -> list[Red]:
    target = root / cited
    if not target.is_file():
        return [Red("R-MISSING", rel, f"cites {cited}, which does not exist")]
    criteria, listed = _record(target)
    reds = []
    for k, label in LABELS.items():
        rows = criteria.get(k, [])
        if not rows:
            reds.append(Red("R-CRITERION", rel, f"{cited} lacks criterion P5b-{k} {label}"))
        elif len(rows) > 1:
            reds.append(
                Red("R-CRITERION", rel, f"{cited} records P5b-{k} {label} {len(rows)} times")
            )
        for verdict, evidence in rows[:1]:
            if not re.search(r"\d", evidence):
                reds.append(
                    Red("R-NONUMBER", rel, f"{cited} P5b-{k} {label} evidence has no number")
                )
            if verdict != "MET":
                reds.append(
                    Red("R-UNMET", rel, f"{cited} P5b-{k} {label} verdict is {verdict}, not MET")
                )
    if rel not in listed:
        reds.append(Red("R-UNLISTED", rel, f"{cited} P5b-MODULES does not name this module"))
    return reds


def _universe(root: Path) -> tuple[list[str], list[str], list[str]] | None:
    """(modules, test files, out-of-scope files), native-relative; None when native/ is absent."""
    if not (root / NATIVE).exists():
        return None
    mods, tests, other = [], [], []
    for f in rust_facts.rust_files(root, NATIVE):
        rel = f.relative_to(root / NATIVE).as_posix()
        dirs = rel.split("/")[:-1]
        (mods if "src" in dirs else tests if "tests" in dirs else other).append(rel)
    return sorted(mods), sorted(tests), sorted(other)


def check(root: Path = REPO, grandfathered=GRANDFATHERED, cutoff=cutoff_modules) -> Report:
    try:
        found = _universe(root)
    except rust_facts.RustFactsUnavailable as exc:
        return Report([Red("R-RUSTFACTS", NATIVE, str(exc))])
    if found is None:
        return Report(absent=True)
    mods, tests, _ = found
    reds: list[Red] = []
    if not mods:
        reds.append(
            Red("R-ZERO", NATIVE, "native/ is present and no module under a crate src/ was found")
        )
    cited_n = 0
    for rel in mods:
        text = (root / NATIVE / rel).read_text(encoding="utf-8", errors="replace")
        cites = citations(text)
        cited_n += bool(cites)
        if len(cites) > 1:
            reds.append(Red("R-DUP-CITE", rel, f"{len(cites)} P5b-PREREG lines: {cites}"))
        if cites:
            reds += _check_record(root, rel, cites[0])
        elif rel not in grandfathered:
            reds.append(Red("R-UNCITED", rel, "no `//! P5b-PREREG:` line and not GRANDFATHERED"))
        if cites and rel in grandfathered:
            reds.append(
                Red(
                    "R-STALE-GRANDFATHER",
                    rel,
                    "cites a pre-registration; delete the GRANDFATHERED entry",
                )
            )
    for rel, why in grandfathered.items():
        if rel not in mods:
            reds.append(
                Red("R-STALE-GRANDFATHER", rel, "GRANDFATHERED entry names no existing .rs module")
            )
        for named in PATH_RE.findall(why):
            if not (root / named).is_file():
                reds.append(
                    Red(
                        "R-MISSING",
                        rel,
                        f"GRANDFATHERED reason names {named}, which does not exist",
                    )
                )
    then = cutoff(root)
    if then is None:
        reds.append(
            Red(
                "R-HISTORY", CUTOFF, "history unavailable: cannot read native/ at the cutoff commit"
            )
        )
    else:
        reds += [
            Red(
                "R-GRANDFATHER-OPEN",
                rel,
                f"GRANDFATHERED but not a file under native/ at {CUTOFF[:8]}",
            )
            for rel in grandfathered
            if rel not in then
        ]
    counts = {
        "modules": len(mods),
        "cited": cited_n,
        "grandfathered": len(set(grandfathered) & set(mods)),
        "test files": len(tests),
        "reds": len(reds),
    }
    return Report(reds, counts)


def count_line(report: Report) -> str:
    if report.absent:
        return "native/ ABSENT"
    c = report.counts
    return (
        f"modules {c.get('modules', 0)}, cited {c.get('cited', 0)}, grandfathered {c.get('grandfathered', 0)}, "
        f"test files {c.get('test files', 0)}, reds {len(report.reds)}"
    )


def census(root: Path = REPO, grandfathered=GRANDFATHERED) -> list[tuple[str, str]]:
    found = _universe(root)
    if found is None:
        return []
    mods, tests, other = found
    rows = []
    for rel in mods:
        cites = citations((root / NATIVE / rel).read_text(encoding="utf-8", errors="replace"))
        rows.append(
            (rel, "CITED" if cites else "GRANDFATHERED" if rel in grandfathered else "UNCITED")
        )
    rows += [(rel, "TEST-FILE") for rel in tests] + [(rel, "OUT-OF-SCOPE") for rel in other]
    return sorted(rows)


def render(report: Report) -> str:
    out = [
        "P5b GATE -- every native module cites a pre-registration recording all six criteria with numbers"
    ]
    out.append(f"  {count_line(report)}")
    out.append(
        "  LIMIT: the gate counts a MET verdict and a digit per criterion; it cannot judge whether the evidence "
        "is true. That stays review-enforced."
    )
    out += [f"  RED {r.code} {r.where}: {r.message}" for r in report.reds]
    return "\n".join(out)


_EV = "specs/s/evidence/p.md"
_MOD = "c/src/lib.rs"
_FILE = f"{NATIVE}/{_MOD}"
_CITE = f"//! P5b-PREREG: {_EV}\n"


def _record_text(modules: str = _MOD, drop: int = 0, **over: str) -> str:
    lines = {
        k: f"P5b-{k} {label}: MET -- measured {k}0 ms over 5 seeds" for k, label in LABELS.items()
    }
    lines |= {int(k): v for k, v in over.items()}
    lines.pop(drop, None)
    return "\n".join([*lines.values(), f"P5b-MODULES: {modules}", ""])


def _planted() -> dict[str, tuple[dict[str, str], dict]]:
    """code -> (tree, check kwargs) for one input that must red with that code. Trees are repo-relative."""
    ok = {_FILE: _CITE, _EV: _record_text()}
    seen = lambda *rels: lambda root: set(rels)  # noqa: E731
    return {
        "R-UNCITED": ({_FILE: "//! none\n"}, {}),
        "R-DUP-CITE": ({**ok, _FILE: _CITE + _CITE}, {}),
        "R-MISSING": ({_FILE: _CITE}, {}),
        "R-CRITERION": ({**ok, _EV: _record_text(drop=3)}, {}),
        "R-NONUMBER": ({**ok, _EV: _record_text(**{"1": "P5b-1 hot: MET -- measured often"})}, {}),
        "R-UNMET": ({**ok, _EV: _record_text(**{"2": "P5b-2 python-bound: UNMET -- 3 ms"})}, {}),
        "R-UNLISTED": ({**ok, _EV: _record_text(modules="c/src/other.rs")}, {}),
        "R-STALE-GRANDFATHER": (
            {_FILE: "//! none\n"},
            {
                "grandfathered": {_MOD: _WHY, "c/src/gone.rs": _WHY},
                "cutoff": seen(_MOD, "c/src/gone.rs"),
            },
        ),
        "R-GRANDFATHER-OPEN": (
            {_FILE: "//! none\n"},
            {"grandfathered": {_MOD: _WHY}, "cutoff": seen()},
        ),
        "R-HISTORY": (ok, {"cutoff": lambda root: None}),
        "R-ZERO": ({f"{NATIVE}/c/README": ""}, {}),
    }


def self_check() -> int:
    """Plant each red in a constructed tree; fail unless the gate reds with that code, and stays green on the
    clean tree. A gate seen only green is indistinguishable from one that cannot see."""
    clean = {_FILE: _CITE, _EV: _record_text(), f"{NATIVE}/c/tests/t.rs": "// test\n"}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (tree, kwargs) in [("clean", (clean, {})), *_planted().items()]:
            root = Path(tmp) / name
            native_dormancy._write(root, tree)
            codes = check(
                root, **{"grandfathered": {}, "cutoff": lambda r: set(), **kwargs}
            ).codes()
            if name == "clean" and codes:
                failures.append(f"clean tree reds: {sorted(codes)}")
            elif name != "clean" and name not in codes:
                failures.append(f"planted {name} did not red (got {sorted(codes)})")
    for failure in failures:
        print(f"p5b_gate self-check FAILED: {failure}")
    if not failures:
        print("p5b_gate self-check OK: every planted red fired, the clean tree stayed green")
    return 1 if failures else 0


def main(argv: list[str] | None = None, root: Path = REPO) -> int:
    ap = argparse.ArgumentParser(description="P5b gate")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ap.add_argument("--self-check", dest="self_check", action="store_true")
    ap.add_argument("--census", action="store_true", help="one line per .rs: path, status")
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()
    if args.census:
        for rel, status in census(root):
            print(f"{NATIVE}/{rel}  {status}")
        return 0
    report = check(root)
    print(render(report))
    if report.reds:
        print(f"p5b_gate ASSERT FAILED: {len(report.reds)} red(s)")
        return 1
    print("p5b_gate assert OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
