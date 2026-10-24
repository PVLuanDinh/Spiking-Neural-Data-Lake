"""P5b gate -- every native Rust module cites a pre-registration recording all six P5b criteria with numbers.

Rule enforced: docs/POLICIES.md P5 "P5b", whose Mechanization line names this gate: every module under the native
crate cites a pre-registration recording all six criteria with their numbers, with a planted-missing red.
Template for shape: gates/null_scale_gate.py (Red records, check(root=...), --assert / --self-check /
--census, a closed GRANDFATHERED table).

# distinct-from: rust_facts.py -- reused here for the one `.rs` walk (`rust_files`, `crate_dirs`); `_tracked` below
# lists the NON-`.rs` files (Cargo.toml, foreign sources) repo-wide, which rust_files cannot; native_dormancy.py -- its
# `_write` builds the constructed trees; null_scale_gate.py -- same CLI shape, different corpus (this reads
# `//!` headers and the cited record, that reads evidence-file NULL-RECORD lines).

A module cites its pre-registration with exactly one line in its leading `//!` block:
    //! P5b-PREREG: <repo-relative path>
The cited record (a reviewable document, below) holds one line each (k = 1..6), then names every module citing it:

    P5b-<k> <label>: <VERDICT> -- <evidence>     labels: hot, python-bound, coarse-boundary, charged,
                                                  python-fix-first, parity-oracle
    P5b-MODULES: <native-relative path>, ...

A module is every `.rs` the crates can compile: any `.rs` under native/ except `<crate root>/{tests,benches,
examples}/` (a crate root is a directory holding a Cargo.toml), so `build.rs`, `#[path]`, `include!` and `[lib]
path` targets outside `src/` count and a `tests/` deeper inside `src/` is a module. A Cargo.toml `[lib]` or
`[[bin]]` target, or a `[package] build` path, that resolves to an excluded file is R-TARGET-EXCLUDED. The cited record must be a reviewable
document: under `specs/`, a `.md`, outside `native/`, not git-ignored. The record names each grandfathered
module by its full native-relative path as a whole token (backtick, space, comma, `|` or line edge each side).

What it reds: R-UNCITED, R-DUP-CITE, R-MISSING, R-RECORD-PLACE, R-CRITERION, R-NONUMBER, R-UNMET, R-UNLISTED,
R-TARGET-EXCLUDED, R-STALE-GRANDFATHER (a stale entry, or one whose record never names the module), R-GRANDFATHER-OPEN,
R-GRANDFATHER-DRIFT (a grandfathered module whose content differs from its blob at the cutoff), R-HISTORY (and
R-ZERO / R-RUSTFACTS: a present `native/` that yields no modules, cannot be listed, lists a `.rs` gone from disk
or has an unreadable or wrongly typed Cargo.toml). Native code that would escape the universe reds too (arc-2
panel A-M2): R-CRATE-OUTSIDE (a Cargo.toml outside native/ not in CARGO_EXEMPT; a path dependency, workspace
member/default-member, `[patch]` or `[replace]` path or a `[package] build` path leaving native/, or any
`.cargo/config[.toml]`; an exempt crate that depends on `pyo3` or
declares a cdylib/dylib/staticlib crate-type), R-STALE-EXEMPT (shrink-only: an exempt entry naming no tracked
file), R-FOREIGN-SOURCE (a tracked compiled source or prebuilt native binary anywhere in the repo) and
R-PATH-ESCAPE (a `#[path = "lit"]` / `include!("lit")` in a module that resolves outside native/ or into an
excluded dir of its crate, or an `include!` whose argument is not one string literal). What it cannot do, and says so on every run: judge whether a criterion's
evidence is TRUE -- a MET verdict and a number (a digit left after issue numbers, dates, shas and path tokens are
stripped) are what it counts; the pre-registration's reviewer decides the rest. The drift check hashes the
working copy with git hash-object, so it assumes the checkout's eol config: a Windows clone with autocrlf
switched off after checkout reds R-GRANDFATHER-DRIFT falsely (CI is LF).

Not checked, and said so in LIMIT: registry and git dependencies (P7's, `dependency_ledger`); a ctypes/cffi load
of a system library at runtime; `include_str!` / `include_bytes!` (data, not code); the `#[path]`/`include!` scan
is textual (a commented-out attribute reds, fail closed) and resolves against the file's own directory (Rust's
base for a `#[path]` inside a nested inline module differs).

  python -m snn_data_lake.gates.p5b_gate [--assert | --self-check | --census]

Bare invocation is --assert (the run_selfchecks.py convention).
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

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
ISSUE_RE = re.compile(r"#\d+")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
SHA_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
EXT_RE = re.compile(r"\.[A-Za-z]\w*$")
NON_MODULE_DIRS = {"tests", "benches", "examples"}
DEP_KINDS = ("dependencies", "dev-dependencies", "build-dependencies")
CARGO_CONFIGS = ("config", "config.toml")
NATIVE_CRATE_TYPES = {"cdylib", "dylib", "staticlib"}
FOREIGN_EXT = frozenset(
    ".c .cc .cpp .cxx .h .hh .hpp .hxx .cu .cuh .s .asm .m .mm .f .f90 "
    ".pyd .so .dll .dylib .a .lib .o .obj".split()
)  # compared lower-cased, so `.S` is `.s`
PATH_ATTR_RE = re.compile(r'#\s*!?\s*\[[^\]]*?\bpath\s*=\s*r?#*"([^"]*)"')
INCLUDE_RE = re.compile(r"\binclude\s*!\s*")
INCLUDE_LIT_RE = re.compile(r'[(\[{]\s*"((?:[^"\\]|\\.)*)"\s*,?\s*[)\]}]')
GLOB_RE = re.compile(r"[*?\[]")

# Shrink-only: the eleven native modules that landed before the gate (the 20e1e742 cutoff), recorded
# retroactively. An entry leaves in the same commit its module gains a real `//! P5b-PREREG:` citation or is
# deleted; a module added after the cutoff can never enter (R-GRANDFATHER-OPEN).
_WHY = f"landed before P5b's gate; recorded retroactively in {RETRO}"
_PFC = "sndl-core/src/prefrontal_executive.rs"
_PFC_WHY = (
    f"{_WHY}; the PFC kernel landed UNDER P5b's text (B2 5978bbcc) and criterion 5 was measured "
    "insufficient after the fact (66f2daac, specs/022-unified-brain-arc/evidence/p5b-pfc-retro-bench.md)"
)
GRANDFATHERED: dict[str, str] = {
    rel: _PFC_WHY if rel == _PFC else _WHY
    for rel in (
        "sndl-core/src/lib.rs",
        _PFC,
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

# Shrink-only: a directory holding a Cargo.toml outside native/ that is not linked into the extension. An entry
# whose directory has no tracked Cargo.toml reds R-STALE-EXEMPT. The exemption covers the crate EXISTING outside
# native/; a native crate LINKING it (a path dependency) still reds, and so does the exempt crate gaining
# pyo3 or a cdylib/dylib/staticlib crate-type.
CARGO_EXEMPT: dict[str, str] = {
    "tools/rsfacts": (
        "gate tooling: the `syn` parser rust_facts.py builds and runs; never linked into snn_data_lake._native"
    ),
}
# Closed and empty: no compiled non-Rust source or prebuilt native binary is tracked, and none may be added.
FOREIGN_EXEMPT: dict[str, str] = {}


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


def cutoff_modules(root: Path = REPO) -> dict[str, str] | None:
    """{native-relative path: blob sha} of every file under native/ at the cutoff commit. None when git cannot
    answer (no git, unknown commit, shallow clone): the caller reds R-HISTORY, never skips."""
    try:
        out = subprocess.run(
            ["git", "ls-tree", "-r", CUTOFF, "--", f"{NATIVE}/"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode:
        return None
    rows = (line.split("\t", 1) for line in out.stdout.splitlines() if "\t" in line)
    return {
        path[len(NATIVE) + 1 :]: meta.split()[2]
        for meta, path in rows
        if path.startswith(f"{NATIVE}/")
    }


def current_blobs(root: Path, rels: list[str]) -> dict[str, str] | None:
    """{native-relative path: blob sha of the file as it is on disk now} (uncommitted edits included).
    None when git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "hash-object", "--", *(f"{NATIVE}/{rel}" for rel in rels)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    shas = out.stdout.split()
    if out.returncode or len(shas) != len(rels):
        return None
    return dict(zip(rels, shas))


def _ignored(root: Path, rel: str) -> bool:
    """True when git ignores `rel`; also True when git cannot say (fail closed)."""
    try:
        return (
            subprocess.run(["git", "check-ignore", "-q", rel], cwd=root, timeout=60).returncode != 1
        )
    except (OSError, subprocess.SubprocessError):
        return True


def _names(rel: str, text: str) -> bool:
    """The record names `rel` by its full native-relative path as a whole token (never by file name)."""
    return re.search(rf"(?<![^\s`,|]){re.escape(rel)}(?![^\s`,|])", text, re.M) is not None


def _has_number(evidence: str) -> bool:
    """A digit survives once issue numbers, dates, shas and path tokens (citations, not numbers) are removed."""
    for noise in (ISSUE_RE, DATE_RE, SHA_RE):
        evidence = noise.sub(" ", evidence)
    return any(
        re.search(r"\d", t) for t in evidence.split() if "/" not in t and not EXT_RE.search(t)
    )


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
    parts = PurePosixPath(cited).parts
    if parts[:1] != ("specs",) or not cited.endswith(".md") or ".." in parts:
        return [Red("R-RECORD-PLACE", rel, f"{cited} is not a specs/**.md record outside native/")]
    target = root / cited
    if not target.is_file():
        return [Red("R-MISSING", rel, f"cites {cited}, which does not exist")]
    if _ignored(root, cited):
        return [Red("R-RECORD-PLACE", rel, f"{cited} is git-ignored (or git cannot say)")]
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
            if not _has_number(evidence):
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


def _excluded(crates: set[Path], f: Path) -> bool:
    """`f` sits under tests/ benches/ examples/ of its NEAREST crate root (judged against the nearest only: a crate
    nested under a workspace's examples/ or tests/ is still a crate, and its src/ holds modules; rv-p5b-d #5032
    item 2)."""
    subs = [f.relative_to(c).parts for c in crates if c in f.parents]
    near = min(subs, key=len) if subs else ()
    return len(near) > 1 and near[0] in NON_MODULE_DIRS


def _universe(root: Path) -> tuple[list[str], list[str], set[Path]] | None:
    """(modules, test files, crate roots), native-relative paths; None when native/ is absent. build.rs is a
    module."""
    if not (root / NATIVE).exists():
        return None
    files = rust_facts.rust_files(root, NATIVE)
    crates = rust_facts.crate_dirs(files)
    mods, tests = [], []
    for f in files:
        rel = f.relative_to(root / NATIVE).as_posix()
        (tests if _excluded(crates, f) else mods).append(rel)
    return sorted(mods), sorted(tests), crates


def _inside(rel: str) -> bool:
    return rel == NATIVE or rel.startswith(f"{NATIVE}/")


def _resolve(base: str, lit: str) -> str | None:
    """`lit` resolved against the repo-relative directory `base`, as a normalised repo-relative path; None when
    it is absolute or climbs above the repo root."""
    lit = lit.replace("\\", "/")
    if lit.startswith("/") or re.match(r"[A-Za-z]:", lit):
        return None
    hit = posixpath.normpath(posixpath.join(base, lit))
    return None if hit == ".." or hit.startswith("../") else hit


def _leaves(base: str, lit: str) -> bool:
    hit = _resolve(base, lit)
    return hit is None or not _inside(hit)


def _dep_tables(doc: dict, bad: list[str]) -> list[tuple[str, dict]]:
    """(label, {name: spec}) for every dependency-shaped table in a manifest: the three kinds at top level and
    under `[target.<cfg>]`, `[workspace.dependencies]`, `[patch.<registry>]` and `[replace]`. A value of the
    wrong TOML type is appended to `bad` (never a crash)."""
    found: list[tuple[str, dict]] = []

    def table(parent: dict, key: str, label: str) -> dict:
        got = parent.get(key, {})
        if isinstance(got, dict):
            return got
        bad.append(f"[{label}] is not a table")
        return {}

    def kinds(parent: dict, prefix: str) -> None:
        found.extend((f"{prefix}{k}", table(parent, k, f"{prefix}{k}")) for k in DEP_KINDS)

    kinds(doc, "")
    found.append(
        (
            "workspace.dependencies",
            table(table(doc, "workspace", "workspace"), "dependencies", "workspace.dependencies"),
        )
    )
    for cfg, body in table(doc, "target", "target").items():
        if isinstance(body, dict):
            kinds(body, f"target.{cfg}.")
        else:
            bad.append(f"[target.{cfg}] is not a table")
    for registry, body in table(doc, "patch", "patch").items():
        if isinstance(body, dict):
            found.append((f"patch.{registry}", body))
        else:
            bad.append(f"[patch.{registry}] is not a table")
    found.append(("replace", table(doc, "replace", "replace")))
    return found


def _manifest_reds(rel: str, here: str, doc: dict) -> list[Red]:
    """R-CRATE-OUTSIDE for a path dependency, workspace member or patch path that leaves native/; R-RUSTFACTS
    for a value of the wrong TOML type. `rel` is the manifest's repo-relative path, `here` its directory."""
    bad: list[str] = []
    reds = []
    for label, deps in _dep_tables(doc, bad):
        for name, spec in deps.items():
            if isinstance(spec, str):
                continue  # a bare version requirement
            if not isinstance(spec, dict):
                bad.append(f"{label}.{name}")
                continue
            path = spec.get("path")
            if path is not None and not isinstance(path, str):
                bad.append(f"{label}.{name}.path")
            elif path is not None and _leaves(here, path):
                reds.append(
                    Red("R-CRATE-OUTSIDE", rel, f"[{label}] {name} path leaves native/: {path!r}")
                )
    ws = doc.get("workspace")
    for key in ("members", "default-members") if isinstance(ws, dict) else ():
        members = ws.get(key, [])
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            bad.append(f"workspace.{key}")
            continue
        for member in members:
            # a glob is judged by its literal prefix: `../*` is the repo root, `d/*` is native/<here>/d
            if _leaves(here, GLOB_RE.split(member)[0] or "."):
                reds.append(
                    Red("R-CRATE-OUTSIDE", rel, f"[workspace] {key} leaves native/: {member!r}")
                )
    return reds + [Red("R-RUSTFACTS", rel, f"{b} has the wrong TOML type") for b in bad]


def _target_reds(root: Path, excluded: set[str]) -> list[Red]:
    """R-TARGET-EXCLUDED for a Cargo.toml `[lib]`/`[[bin]]` target, or any target, whose `path` is an excluded
    file (an example/test/bench target may sit in its own excluded directory); the dependency, member and patch
    paths of the same manifest go through `_manifest_reds`."""
    reds = []
    for manifest in sorted((root / NATIVE).rglob("Cargo.toml")):
        here = manifest.parent.relative_to(root / NATIVE).as_posix()
        if "target" in manifest.relative_to(root / NATIVE).parts:
            continue
        try:
            doc = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            reds.append(Red("R-RUSTFACTS", f"{NATIVE}/{here}/Cargo.toml", f"unreadable: {exc}"))
            continue
        for kind in ("lib", "bin", "example", "test", "bench"):
            targets = doc.get(kind, [])
            targets = [targets] if isinstance(targets, dict) else targets
            if not isinstance(targets, list) or not all(isinstance(t, dict) for t in targets):
                # never a crash on a manifest Cargo itself would reject (rv-p5b-d #5032 hole a)
                reds.append(
                    Red("R-RUSTFACTS", f"{NATIVE}/{here}/Cargo.toml", f"[{kind}] is not a table")
                )
                continue
            for target in targets:
                path = target.get("path")
                if not isinstance(path, str):
                    continue
                hit = os.path.normpath(os.path.join(here, path)).replace(os.sep, "/")
                if os.path.isabs(path) or path.startswith(("/", "\\")) or hit.startswith(".."):
                    # a target compiled from outside native/ never enters the universe (hole b)
                    reds.append(
                        Red(
                            "R-TARGET-EXCLUDED",
                            hit,
                            f"{here}/Cargo.toml [{kind}] path leaves native/: {path!r}",
                        )
                    )
                elif kind in ("lib", "bin") and hit in excluded:
                    reds.append(
                        Red("R-TARGET-EXCLUDED", hit, f"{here}/Cargo.toml [{kind}] path = {path!r}")
                    )
        rel = manifest.relative_to(root).as_posix()
        pkg = doc.get("package", {})
        build = pkg.get("build") if isinstance(pkg, dict) else None
        if not isinstance(pkg, dict) or not isinstance(build, (str, bool, type(None))):
            reds.append(
                Red("R-RUSTFACTS", rel, "[package] or package.build has the wrong TOML type")
            )
        elif isinstance(build, str):
            # Cargo runs a build script from any path the manifest names: the same two escapes as a target
            hit = _resolve(f"{NATIVE}/{here}", build)
            if hit is None or not _inside(hit):
                reds.append(
                    Red("R-CRATE-OUTSIDE", rel, f"[package] build leaves native/: {build!r}")
                )
            elif (under := hit[len(NATIVE) + 1 :]) in excluded:
                reds.append(Red("R-TARGET-EXCLUDED", under, f"{rel} [package] build = {build!r}"))
        reds += _manifest_reds(rel, f"{NATIVE}/{here}", doc)
    return reds


def _tracked(root: Path) -> list[str]:
    """Every file git lists (tracked, plus untracked-but-not-ignored), repo-relative: a local untracked `.c` or
    `.so` reds before it is ever added (fail closed). Git failing is RustFactsUnavailable: the caller reds,
    never skips."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise rust_facts.RustFactsUnavailable(f"git ls-files failed under {root}: {exc}") from exc
    return [p for p in out.split("\0") if p]


def _exempt_reds(root: Path, rel: str) -> list[Red]:
    """An exempt crate must not be able to ship native code: R-CRATE-OUTSIDE when it names pyo3 or builds a
    cdylib/dylib/staticlib."""
    try:
        doc = tomllib.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [Red("R-RUSTFACTS", rel, f"unreadable: {exc}")]
    bad: list[str] = []
    named = any(
        n == "pyo3" or (isinstance(s, dict) and s.get("package") == "pyo3")
        for _, deps in _dep_tables(doc, bad)
        for n, s in deps.items()
    )
    lib = doc.get("lib")
    types = [lib.get(k) for k in ("crate-type", "crate_type")] if isinstance(lib, dict) else []
    builds = any(
        isinstance(t, list) and NATIVE_CRATE_TYPES & {c for c in t if isinstance(c, str)}
        for t in types
    )
    reds = [Red("R-RUSTFACTS", rel, f"{b} has the wrong TOML type") for b in bad]
    if named or builds:
        reds.append(Red("R-CRATE-OUTSIDE", rel, "exempt crate can build a native library"))
    return reds


def _outside_reds(
    root: Path, tracked: list[str], cargo_exempt: dict[str, str], foreign_exempt: dict[str, str]
) -> list[Red]:
    """R-CRATE-OUTSIDE / R-STALE-EXEMPT for every Cargo.toml in the repo, R-FOREIGN-SOURCE for every compiled
    non-Rust source or prebuilt native binary."""
    reds = [
        Red("R-CRATE-OUTSIDE", p, "cargo config can patch, override paths or add link args")
        for p in tracked
        if PurePosixPath(p).parent.name == ".cargo" and PurePosixPath(p).name in CARGO_CONFIGS
    ]
    manifests = {
        p: PurePosixPath(p).parent.as_posix()
        for p in tracked
        if PurePosixPath(p).name == "Cargo.toml"
    }
    for p, d in sorted(manifests.items()):
        if d in cargo_exempt:
            reds += _exempt_reds(root, p)
        elif not _inside(d):
            reds.append(
                Red("R-CRATE-OUTSIDE", p, "Cargo.toml outside native/ and not in CARGO_EXEMPT")
            )
    reds += [
        Red("R-STALE-EXEMPT", d, "CARGO_EXEMPT entry has no tracked Cargo.toml")
        for d in cargo_exempt
        if d not in manifests.values()
    ]
    foreign = {p for p in tracked if PurePosixPath(p).suffix.lower() in FOREIGN_EXT}
    reds += [
        Red("R-FOREIGN-SOURCE", p, "compiled non-Rust source or prebuilt native binary")
        for p in sorted(foreign - set(foreign_exempt))
    ]
    reds += [
        Red("R-STALE-EXEMPT", p, "FOREIGN_EXEMPT entry names no tracked file")
        for p in foreign_exempt
        if p not in foreign
    ]
    return reds


def _escape_reds(root: Path, rel: str, text: str, crates: set[Path]) -> list[Red]:
    """R-PATH-ESCAPE for each `#[path = "lit"]` / `include!("lit")` in module `rel` (native-relative) that
    resolves outside native/ or under an excluded dir of its nearest crate, and for an `include!` whose
    argument is not one plain string literal (unverifiable, fail closed). Textual: include_str!/include_bytes!
    are data and not scanned."""
    base = posixpath.dirname(f"{NATIVE}/{rel}")
    lits = [m.group(1) for m in PATH_ATTR_RE.finditer(text)]
    reds = []
    for m in INCLUDE_RE.finditer(text):
        lit = INCLUDE_LIT_RE.match(text, m.end())
        if lit:
            lits.append(lit.group(1))
        else:
            reds.append(Red("R-PATH-ESCAPE", rel, "include! argument is not one string literal"))
    for lit in lits:
        hit = _resolve(base, lit)
        if hit is None or not _inside(hit):
            reds.append(Red("R-PATH-ESCAPE", rel, f"{lit!r} leaves native/"))
            continue
        target = root / hit
        if _excluded(crates | rust_facts.crate_dirs([target]), target):
            reds.append(Red("R-PATH-ESCAPE", rel, f"{lit!r} resolves into an excluded dir: {hit}"))
    return reds


def check(
    root: Path = REPO,
    grandfathered=GRANDFATHERED,
    cutoff=cutoff_modules,
    blobs=current_blobs,
    cargo_exempt=CARGO_EXEMPT,
    foreign_exempt=FOREIGN_EXEMPT,
) -> Report:
    try:
        found = _universe(root)
        tracked = _tracked(root) if found else []
    except rust_facts.RustFactsUnavailable as exc:
        return Report([Red("R-RUSTFACTS", NATIVE, str(exc))])
    if found is None:
        return Report(absent=True)
    mods, tests, crates = found
    reds: list[Red] = _target_reds(root, set(tests))
    reds += _outside_reds(root, tracked, cargo_exempt, foreign_exempt)
    if not mods:
        reds.append(Red("R-ZERO", NATIVE, "native/ is present and no module was found"))
    cited_n = 0
    for rel in mods:
        try:
            text = (root / NATIVE / rel).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            reds.append(Red("R-RUSTFACTS", rel, f"git lists it but it cannot be read: {exc}"))
            continue
        reds += _escape_reds(root, rel, text, crates)
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
            elif named == RETRO and not _names(
                rel, (root / named).read_text(encoding="utf-8", errors="replace")
            ):
                reds.append(Red("R-STALE-GRANDFATHER", rel, f"{named} never names this module"))
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
        pinned = sorted(rel for rel in grandfathered if rel in mods and rel in then)
        now = blobs(root, pinned) if pinned else {}
        if now is None:
            reds.append(
                Red(
                    "R-HISTORY",
                    NATIVE,
                    "history unavailable: cannot hash the grandfathered modules",
                )
            )
        else:
            reds += [
                Red(
                    "R-GRANDFATHER-DRIFT",
                    rel,
                    f"GRANDFATHERED module differs from its {CUTOFF[:8]} content: cite a pre-registration",
                )
                for rel in pinned
                if now[rel] != then[rel]
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
    try:
        found = _universe(root)
    except rust_facts.RustFactsUnavailable as exc:
        return [(NATIVE, f"R-RUSTFACTS {exc}")]
    if found is None:
        return []
    mods, tests, _ = found
    rows = []
    for rel in mods:
        try:
            text = (root / NATIVE / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            rows.append((rel, "R-RUSTFACTS unreadable"))
            continue
        rows.append(
            (
                rel,
                "CITED"
                if citations(text)
                else "GRANDFATHERED"
                if rel in grandfathered
                else "UNCITED",
            )
        )
    rows += [(rel, "TEST-FILE") for rel in tests]
    return sorted(rows)


def render(report: Report) -> str:
    out = [
        "P5b GATE -- every native module cites a pre-registration recording all six criteria with numbers"
    ]
    out.append(f"  {count_line(report)}")
    out.append(
        "  LIMIT: the gate counts a MET verdict and a number per criterion (a digit left after issue numbers, "
        "dates, shas and path tokens are stripped); it cannot judge whether the evidence is true. That stays "
        "review-enforced. Ids that carry digits (T018, B2, PFC-NEUTER-1023) count as numbers; evidence written "
        "only as ratios with `/` reds R-NONUMBER (fail closed); nothing checks that a record predates its module. "
        "An [[example]]/[[test]]/[[bench]] target inside its own directory is not a module, even as a cdylib. "
        "Registry and git dependencies are P7's (dependency_ledger), not checked here; a ctypes/cffi load of a "
        "system library at runtime is not seen; the #[path]/include! scan is textual (a commented-out attribute "
        "reds, fail closed), resolves against the file's own directory (Rust's base for a #[path] inside a "
        "nested inline module differs), and does not scan include_str!/include_bytes! (data, not code); "
        "the extension build itself is pinned in infra/native_build.py (MANIFEST = "
        "native/sndl-python/Cargo.toml) and this gate does not read that file; a .cargo/config in a directory "
        "above the repo or in CARGO_HOME is not seen."
    )
    out += [f"  RED {r.code} {r.where}: {r.message}" for r in report.reds]
    return "\n".join(out)


_EV = "specs/s/evidence/p.md"
_MOD = "c/src/lib.rs"
_FILE = f"{NATIVE}/{_MOD}"
_CITE = f"//! P5b-PREREG: {_EV}\n"
_OUT_DEP = '[dependencies]\nk = { path = "../../tools/k" }\n'


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
    seen = lambda *rels: lambda root: {rel: "a" for rel in rels}  # noqa: E731
    pinned = {RETRO: f"{_MOD}\n"}
    return {
        "R-UNCITED": ({_FILE: "//! none\n"}, {}),
        "R-UNCITED:outside-src": ({**ok, f"{NATIVE}/c/kernels/hot.rs": "//! none\n"}, {}),
        "R-DUP-CITE": ({**ok, _FILE: _CITE + _CITE}, {}),
        "R-MISSING": ({_FILE: _CITE}, {}),
        "R-RECORD-PLACE": ({_FILE: f"//! P5b-PREREG: {_FILE}\n"}, {}),
        "R-RECORD-PLACE:ignored": ({**ok, ".gitignore": f"{_EV}\n"}, {}),
        "R-CRITERION": ({**ok, _EV: _record_text(drop=3)}, {}),
        "R-NONUMBER": ({**ok, _EV: _record_text(**{"1": "P5b-1 hot: MET -- measured often"})}, {}),
        "R-NONUMBER:citation": (
            {**ok, _EV: _record_text(**{"1": "P5b-1 hot: MET -- see #3847"})},
            {},
        ),
        "R-UNMET": ({**ok, _EV: _record_text(**{"2": "P5b-2 python-bound: UNMET -- 3 ms"})}, {}),
        "R-UNLISTED": ({**ok, _EV: _record_text(modules="c/src/other.rs")}, {}),
        "R-STALE-GRANDFATHER": (
            {_FILE: "//! none\n"},
            {
                "grandfathered": {_MOD: _WHY, "c/src/gone.rs": _WHY},
                "cutoff": seen(_MOD, "c/src/gone.rs"),
            },
        ),
        "R-STALE-GRANDFATHER:unnamed": (
            {_FILE: "//! none\n", RETRO: "nothing\n"},
            {"grandfathered": {_MOD: _WHY}, "cutoff": seen(_MOD)},
        ),
        "R-STALE-GRANDFATHER:cross-name": (
            {_FILE: "//! none\n", RETRO: f"x{_MOD}\n"},
            {"grandfathered": {_MOD: _WHY}, "cutoff": seen(_MOD)},
        ),
        "R-TARGET-EXCLUDED": (
            {
                **ok,
                f"{NATIVE}/c/Cargo.toml": '[lib]\npath = "tests/k.rs"\n',
                f"{NATIVE}/c/tests/k.rs": "#[pyfunction]\nfn f() {}\n",
            },
            {},
        ),
        "R-UNCITED:build-rs": ({**ok, f"{NATIVE}/c/build.rs": "fn main() {}\n"}, {}),
        "R-CRATE-OUTSIDE": ({**ok, "tools/k/Cargo.toml": ""}, {}),
        "R-CRATE-OUTSIDE:path-dep": ({**ok, f"{NATIVE}/c/Cargo.toml": _OUT_DEP}, {}),
        "R-CRATE-OUTSIDE:member": (
            {**ok, f"{NATIVE}/Cargo.toml": '[workspace]\nmembers = ["../tools/k"]\n'},
            {},
        ),
        "R-CRATE-OUTSIDE:patch": (
            {**ok, f"{NATIVE}/c/Cargo.toml": '[patch.crates-io]\nk = { path = "../../tools/k" }\n'},
            {},
        ),
        "R-CRATE-OUTSIDE:exempt-pyo3": (
            {**ok, "tools/k/Cargo.toml": '[dependencies]\npyo3 = "0.30"\n'},
            {"cargo_exempt": {"tools/k": "tooling"}},
        ),
        "R-CRATE-OUTSIDE:exempt-cdylib": (
            {**ok, "tools/k/Cargo.toml": '[lib]\ncrate-type = ["cdylib"]\n'},
            {"cargo_exempt": {"tools/k": "tooling"}},
        ),
        "R-CRATE-OUTSIDE:build": (
            {**ok, f"{NATIVE}/c/Cargo.toml": '[package]\nbuild = "../../tools/b.rs"\n'},
            {},
        ),
        "R-CRATE-OUTSIDE:cargo-config": ({**ok, ".cargo/config.toml": "[net]\n"}, {}),
        "R-TARGET-EXCLUDED:build": (
            {
                **ok,
                f"{NATIVE}/c/Cargo.toml": '[package]\nbuild = "tests/b.rs"\n',
                f"{NATIVE}/c/tests/b.rs": "fn main() {}\n",
            },
            {},
        ),
        "R-RUSTFACTS:build-type": (
            {**ok, f"{NATIVE}/c/Cargo.toml": "[package]\nbuild = 3\n"},
            {},
        ),
        "R-STALE-EXEMPT": (ok, {"cargo_exempt": {"tools/k": "tooling"}}),
        "R-RUSTFACTS:dep-type": (
            {**ok, f"{NATIVE}/c/Cargo.toml": "[dependencies]\nk = { path = 3 }\n"},
            {},
        ),
        "R-FOREIGN-SOURCE": ({**ok, f"{NATIVE}/c/src/k.c": "int k;\n"}, {}),
        "R-FOREIGN-SOURCE:outside": ({**ok, "tools/k/k.cu": "x\n"}, {}),
        "R-FOREIGN-SOURCE:binary": ({**ok, "python/k.pyd": "x\n"}, {}),
        "R-PATH-ESCAPE": (
            {
                **ok,
                _FILE: _CITE + '#[path = "../tests/k.rs"]\nmod k;\n',
                f"{NATIVE}/c/tests/k.rs": "// t\n",
            },
            {},
        ),
        "R-PATH-ESCAPE:include": (
            {**ok, _FILE: _CITE + 'include!("../../../../tools/k.rs");\n'},
            {},
        ),
        "R-PATH-ESCAPE:non-literal": (
            {**ok, _FILE: _CITE + 'include!(concat!(env!("OUT_DIR"), "/k.rs"));\n'},
            {},
        ),
        "R-RUSTFACTS:gone": (ok, {"gone": [_FILE]}),
        "R-GRANDFATHER-OPEN": (
            {_FILE: "//! none\n"},
            {"grandfathered": {_MOD: _WHY}, "cutoff": seen()},
        ),
        "R-GRANDFATHER-DRIFT": (
            {_FILE: "//! none\nmod hot {}\n", **pinned},
            {
                "grandfathered": {_MOD: _WHY},
                "cutoff": seen(_MOD),
                "blobs": lambda root, rels: {rel: "b" for rel in rels},
            },
        ),
        "R-HISTORY": (ok, {"cutoff": lambda root: None}),
        "R-HISTORY:hash": (
            {_FILE: "//! none\n", **pinned},
            {
                "grandfathered": {_MOD: _WHY},
                "cutoff": seen(_MOD),
                "blobs": lambda root, rels: None,
            },
        ),
        "R-ZERO": ({f"{NATIVE}/c/README": ""}, {}),
    }


def self_check() -> int:
    """Plant each red in a constructed tree; fail unless the gate reds with that code, and stays green on the
    clean tree. A gate seen only green is indistinguishable from one that cannot see."""
    clean = {_FILE: _CITE, _EV: _record_text(), f"{NATIVE}/c/tests/t.rs": "// test\n"}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (tree, kwargs) in [("clean", (clean, {})), *_planted().items()]:
            root = Path(tmp) / name.replace(":", "-")
            native_dormancy._write(root, tree)
            kwargs = dict(kwargs)
            for rel in kwargs.pop("gone", ()):
                subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
                (root / rel).unlink()
            codes = check(
                root,
                **{"grandfathered": {}, "cutoff": lambda r: {}, "cargo_exempt": {}, **kwargs},
            ).codes()
            want = name.split(":")[0]
            if name == "clean" and codes:
                failures.append(f"clean tree reds: {sorted(codes)}")
            elif name != "clean" and want not in codes:
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
