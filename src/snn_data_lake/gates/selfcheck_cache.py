"""Self-check carry-forward cache (owner-authorized build, 2026-08-21; envelope v3 the same
day): a battery may CARRY a module's PASS verdict when EVERY input that could have changed
the outcome is byte-identical to the inputs that produced that verdict on THIS host -- a
hash-verified re-statement of "these exact bytes already passed here", never a skip.

WHY: the sequential battery re-runs all ~331 self-checks per commit while a typical commit
touches a handful of files. Parallel battery forms are REFUTED on this host (the pilot,
twice; sequential is binding), so the honest accelerator shrinks WORK instead of
overlapping it: a verdict carries only when nothing that could have changed the outcome --
the module's source, every in-package source it can statically reach, the data files and
environment variables it names, the rosters it imports, the tree it scans, the runner's
logic or the interpreter -- has changed a byte.

PRIOR ART, read before building (testmon's determining-affected-tests + hidden-test-
dependencies; Bazel's hermeticity rule): coverage-block selection is FINER but traces
nothing across subprocess boundaries -- and every self-check here IS a subprocess -- so
the import-closure key is the structurally right granularity for this battery: coarser,
over-invalidating, never carrying wrongly for statically-reachable code. Bazel's lesson
is the envelope: an UNDECLARED input is the classic cache poison, so every input a check
could see is either keyed to the modules that can see it or held in one global envelope.

THE KEY (all of it, or no carry) -- v3, ENVELOPE PRECISION (2026-08-21):
  v2 put every package __init__, the runner's bytes and every tools/ artifact into ONE
  global envelope, so the per-task rhythm (a REGIONS row, a runner tier row, a regenerated
  tools/wiring_manifest.nir) invalidated all ~331 verdicts on EVERY task -- measured ~20
  min, the cache helping only on an unchanged tree. v3 keeps every one of those inputs
  but keys each to the modules whose verdict can depend on it, DERIVED from the closure's
  own source -- never a hand-maintained map:
  * CLOSURE rows: the module's file + every in-package file its static import closure
    reaches (dormancy_gate.import_graph() -- the ONE walk, reused; ast-level, so lazy
    imports are included).
  * ROSTER SPLIT: the root __init__ and run_selfchecks.py are hashed as AST with their
    pure string-roster literals (REGIONS / TORCH_TIER / TRAIN_TIER; CORE / TORCH_GATED /
    ...) blanked. That NORMALIZED hash is GLOBAL (any logic edit invalidates everything);
    the roster CONTENT hash goes to every module whose closure imports ANY name from the
    root package -- `from snn_data_lake import REGIONS` and `from snn_data_lake import
    arm_bridge` alike, because every flat alias resolves through the roster-derived
    `_MOVED` map, so a row REMOVED or moved breaks that import and must flip the key (the
    review built that poison against the first draft, which keyed only the three roster
    NAMES). A row append therefore invalidates the root-package importers and nothing
    else; the root __init__ is lazy (alias finder + __getattr__), so the rest of the tree
    is untouched by it.
  * DATA INPUTS with derived readers: every file under tools/ (non-.py) and models/,
    .github/workflows/*.yml, docs/*.md, root *.md, pyproject.toml, graphify-out/*.json and
    run_selfchecks.py -- a module whose closure names the file's basename in a
    NON-DOCSTRING string constant is a reader and carries that file's hash. A tools/ or
    models/ artifact that NO module names stays GLOBAL (fail-closed: a read under a
    computed name would be invisible). models/ and graphify-out/ are gitignored, which is
    exactly why they must be keyed here: no tree surface sees them.
  * LITERAL RELPATH READS (B4 C4, 2026-10-20): a NON-DOCSTRING string that is, whole, a
    repo-relative path (a directory component; `/` or `\\`; no `..`) naming an EXISTING file
    keys that file's bytes to every module whose closure holds the string, as a data row --
    unless the file is already a closure row. The closing review found the live stale carry:
    instruments.eligible_circuit_census reads 8 src files by literal relpath (its DRIVERS'
    `_read`, every RECORDS citation's line-count check) under a 1-file closure, so a brain.py
    edit pushing a cited line out of range carried the old PASS. Chosen over "such a module
    is a scanner" because both are correct for its verdict and this one is smaller: it moves
    ONE module's class (eligible_circuit_census, plain -> data) and adds a row to two more
    non-scanners -- capability_bar (infra/cockpit.py, one of its TRACKED_SURFACES: a SECOND
    live stale carry, closed) and doc_budget_gate (docs/POLICIES.md, already global) -- where
    the scanner rule would re-run eligible_circuit_census and capability_bar on every tree
    change. Scanners gain the rows too, redundantly (their tree row already holds every
    tracked file). Its `_callee_kind` also parses the modules the DRIVERS import, by a
    COMPUTED path -- outside this rule (blind spot 4) and outside the verdict: CLASS /
    FACTORY / UNRESOLVED all count as object handles (measured: permuting the three leaves
    main() rc 0; one injected undisposed site reds it), so only the printed sites_by_kind
    can go stale.
  * ENVIRONMENT: every NORD_* variable is GLOBAL (read under computed names); any other
    variable a closure reads by literal name -- `os.environ.get("NMNIST_SEED")`,
    `os.getenv(...)`, `os.environ[...]` -- is keyed to that reader with its value. The
    runner passes the whole process environment to every child, so an unnamed variable is
    an undeclared input; the review found 48 of them with 5 registered checks reading at
    module load (the N-MNIST seed and sizes among them).
  * TREE SCANNERS (every listing, since B4 C2, 2026-10-20): a module whose closure lists a
    directory -- a _SCAN_CALLS call (os.walk / fwalk / listdir / scandir / glob / iglob /
    rglob / iterdir / pkgutil.iter_modules / walk_packages), resolved through the file's
    import aliases (ast.walk and the file's own functions are not listings), or such a lister
    taken as a value -- or names git in a non-docstring string (any `git` argv since B4 C3:
    `git diff --name-only`, `git status --short`, `git show`; plus the _GIT_TREE_READS forms
    `ls-files`, `ls-tree`, `--porcelain` however git itself is spelled), reads files no
    import closure names. No root
    is exempt: a listing of substrate.home() or of an environment-named directory scans like
    any other. The landing rule keyed only roots derived from `__file__` IN THE SAME FILE,
    and the Opus B3 review measured the hole: an impact_analysis.py override left
    replication_guard, native_ratchet and rust_facts UNCHANGED although each walks the
    package. The carryfix replaced it with a root RESOLVER (home() and env-read roots
    exempt, through every bind of a name); the B4 closing review found it changed no
    module's class on the real tree and failed open on constructed shapes -- an env value
    seeded from `__file__` (S2), `os.walk(os.environ["PYTHONPATH"])` (S3, live-shaped: the
    runner's _ENV sets PYTHONPATH=src), a rebound `home.home` (S4) or `globals()["home"]`
    (S5), a home() snapshot of the tree (S6), a split `..` (S8) -- so it was DELETED. The
    exemption it bought is ZERO modules at deletion (173 plain / 206 scanners before and
    after); a future home()-lister only loses its carry -- slower, never wrong. The dormancy /
    method-dormancy / import-manifest / replication / transport gates, the atlas, the
    atlas-node gate, brain_geometry, self_mod_harness, nightly_skills, and by closure
    everything that imports them (the brain itself imports the dormancy gate). Their key
    carries the sha256 of the WHOLE tracked + untracked-unignored tree (`git ls-files -co
    --exclude-standard`), so they re-run on any tree change, and only then. v2 had NO row
    for this surface at all: an edit outside a gate's import closure (the method-dormancy
    gate scans modules it never imports) could carry its stale PASS. It was safe only
    because the global envelope changed on every task anyway. A closure containing
    gates.rust_facts also carries the `native:` row below, which folds in tools/rsfacts/**
    (the fixture crate its self-check parses; a tools/ SUBdirectory no data row lists).
  * RUST SURFACE (R0.2 row 13, 2026-10-17): a module whose closure READS the extension
    `snn_data_lake._native` (any import spelling, derived per file like the scanner rule; or
    the graph node itself) carries a `native:` row -- sha256 over (path, sha256) of every
    `.rs`, Cargo.toml, Cargo.lock, build.rs and rust-toolchain(.toml) under native/ (target/
    skipped) and at the repo root, plus every other file git lists under native/ (tracked or
    untracked-unignored: runtime data the extension reads; every file when git cannot list) --
    and an `ext:` row, the loaded extension's bytes (the find_spec origin, else the canonical
    src/snn_data_lake/_native<suffix> maturin develop fills). Before this row `_native` resolved
    to a missing .py, hashed as the constant "ABSENT", and no Cargo file was listed anywhere: a
    Rust edit carried a STALE GREEN. The extension is gitignored, so no tree surface sees it; a
    module that never reads `_native` gets no such row (key invariance, red (m)). native/
    missing prints ABSENT.
    The import graph keys .py files only, so the reader flag is DERIVED per file (absolute
    and root-relative spellings, the roster form, and an `__all__` naming the extension); a
    package __init__ that re-exports the extension lifts both rows into the GLOBAL envelope
    (it is in no closure, and every star importer loads what its `__all__` names).
  * GLOBAL envelope: cache version, interpreter tag (free-threaded marked), every NORD_*
    variable, the two normalized AST hashes above, every non-root package __init__
    (executed on import of any module in the region, present in no closure), the five
    brainstem/ safety-envelope files + SECURITY.md + docs/POLICIES.md (the governance
    surface -- an unapproved brainstem edit must never carry the stale PASS of the gate
    built to catch it), and the unreferenced tools/ + models/ artifacts.
MEASURED AT LANDING (2026-08-21, the T037 task diff: a new module, five edited modules, a
REGIONS row, a runner row, a regenerated wiring_manifest.nir; over the 331 registered
MODULES -- the self-check's own census covers the whole 405-module import graph, a
different denominator): 101 of 331 keys invalidated -- 81 tree scanners, 18 root-package
importers, 2 runner-file readers, 0 by closure alone -- against 331 of 331 under v2; 230
verdicts carried. The first draft measured 83; the review's two widenings (flat-alias
importers, attribute-rooted scanners) raised it to 101, honestly, because each one closed
a constructed stale carry. The 81 scanners are the floor for any tree change and include
the brain-level checks; what that floor costs in minutes is what the per-line wall-clock
exists to show, battery by battery -- the runner's RERUN lines name each invalidation's
class.
Only PASS verdicts are cached; a FAIL always re-runs; record() re-validates every input at
record time (a PASS is attributed to the bytes that ran, not to hashes read minutes
earlier). Each row keeps the live wall-clock.

The store lives at the REAL user home (~/.snn_data_lake/selfcheck_cache.json),
DELIBERATELY not substrate.home(): the battery exports NORD_HOME to a throwaway temp dir,
and a cache living there would die with every run. On Windows that path follows
USERPROFILE, not HOME -- a probe that redirected HOME wrote three unrun PASS rows into the
real store, which is why every self-check instance below uses a temp store_path.

DISCLOSED BLIND SPOTS (banner() prints them at every carrying run):
  1. STRING-DISPATCH imports (importlib with a computed name) are invisible to the static
     closure -- the dormancy gate's own known limit, inherited knowingly.
  2. SHARED-HOME SEQUENCE COUPLING: all self-check subprocesses share one persistent
     NORD_HOME for the run, so a carried module's ABSENCE changes what home-state a
     downstream check sees this run (the hidden-test-dependency shape). Bounded -- that
     home already persists across batteries -- but real. A roster row also reorders the
     run; same shape.
  3. Host state outside the envelope (installed package versions, GPU state) is unhashed.
  4. (v3) A data file or environment variable read under a COMPUTED name is the
     string-dispatch shape for data: invisible to the literal derivation. Bounded by the
     fail-closed rule for tools/ and models/ (unreferenced => global) and by NORD_* being
     global; docs/, .github/, graphify-out/ and non-NORD variables carry no such floor.
     The same holds for a file path built from PARTS (`os.path.join(root, "src", ...)`,
     a tuple of components), an absolute path, a path inside a longer string, and a path
     derived from another read (eligible_circuit_census._callee_kind) -- only a whole
     literal relpath string is a LITERAL RELPATH READ above, and it is resolved against the
     REPO ROOT only: a whole package-relative ('engine/brain.py') or src-relative
     ('snn_data_lake/engine/brain.py') literal is unkeyed (B4 re-review D1; none live at landing).
  5. (v3) Gitignored files are outside the tree surface a scanner is keyed on. The ones
     the tree reads by name are keyed as data inputs (models/*, graphify-out/*.json --
     connectome's self-check asserts on graph.json, found by the review); any other
     gitignored path read by literal name elsewhere is unkeyed. None known at landing.
  6. (v3, B4 C2 2026-10-20) The scanner derivation is SYNTACTIC, not fail-closed: every
     listing call it RECOGNISES scans, whatever its root, but recognition is by name through
     the file's import aliases, so a lister reached another way is invisible: STRING
     DISPATCH -- `getattr(os, "walk")(pkg)`, `getattr(Path(d), name)()` (the closing review's
     S1; stale under every rule it measured, and pinned stale by an xfail(strict) in
     tests/test_selfcheck_cache_unflagged_scanners.py so a fix must update this line) -- and a
     tree listed by a subprocess that is not git or is git without a literal `git` argv (find,
     rg, a git path read from a variable with none of the _GIT_TREE_READS forms). Red (n)
     pins the three walkers the Opus B3 review measured carrying; the derivation pins in
     _SYNTHETIC hold every rule, incl. the closing review's constructed stale shapes S2-S8.
THE PROTOCOL those blind spots buy: carries are a LOCAL-battery accelerator only (CI,
act containers and exit-certification batteries run FULL -- --carry is opt-in and nothing
in .github/ passes it), and ANY FAIL in a carrying battery is re-run FULL before it is
diagnosed -- a carried-run red is not evidence until the full run confirms it.

  python -m snn_data_lake.gates.selfcheck_cache   # self-check (the P8 reds)
"""

# distinct-from: gates/dormancy_gate.py -- owns THE import graph; this REUSES it via
#   import_graph() to hash closures, never re-deriving reachability.
# distinct-from: tools/p4c_ratchet.py -- counts modules per zone; this hashes them.
# distinct-from: gates/policy_manifest.py -- certifies POLICY text; this caches
#   self-check verdicts. All sites read.

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys

from snn_data_lake.gates import rust_facts as _rf  # stdlib-only reader; owns the crate paths

_CACHE_VERSION = 3  # v3: envelope precision -- rosters split out, data inputs, named
# environment variables and the tree surface keyed to their DERIVED readers. v2 (review
# BLOCKERs) had grown the governance surface into one global envelope; every input v2
# hashed is still hashed here, now keyed to the modules whose verdict can see it.
_PKG = "snn_data_lake"
_PKG_PREFIX = _PKG + "."
_NATIVE = _rf.NATIVE_MODULE
_NATIVE_LEAF = _NATIVE.rpartition(".")[2]  # "_native": the extension has no .py of its own
_NATIVE_NAMES = frozenset(
    {"Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain", "rust-toolchain.toml"}
)
_SCAN_CALLS = frozenset(
    {"walk", "fwalk", "listdir", "scandir", "glob", "iglob", "rglob", "iterdir"}
    | {"iter_modules", "walk_packages"}  # pkgutil: a package's modules, loaded outside any closure
)
_LISTING_MODULES = frozenset({"os", "glob", "pathlib", "pkgutil"})  # a lister taken as a value
_GIT_TREE_READS = ("ls-files", "ls-tree", "--porcelain")  # argv strings that read the tree via git
# B4 C3: ANY git argv is a scan -- `git diff --name-only`, `git status --short`, `git show`
# read the tree or its history beyond these three forms (review S7); a string that IS the
# program name, or starts a shell command line with it. `.match` anchors at the START: git
# later in a shell line or behind a path prefix is not seen (B4 re-review D2, in the banner)
_GIT_ARGV = re.compile(r"git(?:\.exe)?(?:\s|$)")
# B4 C4: a whole non-docstring string that IS a repo-relative path (a directory component, no
# `..`, no spaces) names a file the module may READ without importing it -- its bytes key the
# reader as a data row when the file exists (eligible_circuit_census reads src/**/*.py so)
_RELPATH = re.compile(r"[\w.\-]+(?:[\\/][\w.\-]+)+")
# No listing root is excluded (B4 C2, 2026-10-20): the root resolver that once exempted
# substrate.home() and environment-read roots was DELETED -- it moved zero modules on the real
# tree and failed open on constructed shapes (S2-S5 below), one of them live-shaped.
# red (n)'s subjects, shared with tests/test_selfcheck_cache_unflagged_scanners.py: the three
# walkers the Opus B3 review measured carrying (parameter-rooted walks, `git ls-files`), the
# rust_facts reader that already flipped, and the two files whose override must flip them
_WALKERS = ("gates.replication_guard", "gates.native_ratchet", "gates.rust_facts")
_RUST_FACTS_READER = "gates.native_dormancy"
_IMPACT_FILE = ("src", _PKG, "reasoning", "impact_analysis.py")
_LIB_RS_FILE = (_rf.NATIVE_DIR, "sndl-core", "src", "lib.rs")
# red (o)'s subject (B4 C4): a module reading src files by literal relpath, and one it reads
_RELPATH_READER = "instruments.eligible_circuit_census"
_BRAIN_FILE = ("src", _PKG, "engine", "brain.py")
_NULL_RE = re.compile(r"(?!x)x")  # the derivation pins name no data artifact
_RUST_FACTS = _rf.__name__
# (directory, accepted extensions | None = every file, minus .py) -- DATA INPUTS keyed to
# their derived readers.
_DATA_DIRS = (
    ("tools", None),
    ("models", None),
    (os.path.join(".github", "workflows"), (".yml", ".yaml")),
    ("docs", (".md",)),
    ("graphify-out", (".json",)),
    ("", (".md", ".toml")),
)
_FAIL_CLOSED_DIRS = ("tools", "models")  # unreferenced artifacts here stay GLOBAL
# The roster literals the splitter is EXPECTED to blank; a new one must be adjudicated here,
# never inherited (the self-check pins both tuples).
_EXPECTED_ROSTERS = {
    # G8 S1a adjudicated the three LEGACY_* tables (the frozen pre-G8 labels, byte-identical
    # copies of the S0-head tables): string rosters by construction, blanked like their originals
    "root": (
        "REGIONS",
        "TORCH_TIER",
        "TRAIN_TIER",
        "LEGACY_REGIONS",
        "LEGACY_TORCH_TIER",
        "LEGACY_TRAIN_TIER",
    ),
    # B3 adjudicated TIMING_DEFERRED (2026-10-19): a str roster of CORE modules that only moves a
    # module between the --defer-timing and --timing-only runs; no module's behaviour depends on it
    "runner": (
        "CORE",
        "RESEARCH",
        "TORCH_GATED",
        "NENGO_GATED",
        "TRAIN_GATED",
        "ARTIFACT_GATED",
        "TIMING_DEFERRED",
    ),
}
_FACTS_MEMO = {}  # (path, size, mtime_ns, markers) -> facts; shared across instances


def _repo_root():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def _cache_path():
    return os.path.join(os.path.expanduser("~"), ".snn_data_lake", "selfcheck_cache.json")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def _module_file(dotted):
    """snn_data_lake.<region>.<mod> -> source file path (package dirs -> __init__.py)."""
    root = os.path.join(_repo_root(), "src", _PKG)
    rel = dotted[len(_PKG_PREFIX) :] if dotted.startswith(_PKG_PREFIX) else dotted
    base = os.path.join(root, rel.replace(".", os.sep))
    if os.path.isdir(base):
        return os.path.join(base, "__init__.py")
    return base + ".py"


def _is_strlist(node):
    return isinstance(node, ast.List) and all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts
    )


def _is_roster(node):
    """A pure string-roster literal: a list of str constants, or a dict of str keys to
    such lists (REGIONS / TORCH_TIER / TRAIN_TIER; the runner's tier lists)."""
    if _is_strlist(node):
        return True
    return isinstance(node, ast.Dict) and all(
        isinstance(k, ast.Constant) and isinstance(k.value, str) and _is_strlist(v)
        for k, v in zip(node.keys, node.values)
    )


def split_rosters(text):
    """(normalized_hash, roster_hash, roster_names): the module's AST hashed with every
    top-level roster literal blanked, the hash of those literals alone, and their names."""
    tree = ast.parse(text)
    names, dumps = [], []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_roster(node.value)
        ):
            names.append(node.targets[0].id)
            dumps.append(f"{node.targets[0].id}={ast.dump(node.value)}")
            node.value = ast.Constant("<ROSTER>")
    return _sha256_text(ast.dump(tree)), _sha256_text("|".join(dumps)), tuple(names)


def _env_name(node):
    """The literal variable name of an os.environ.get / os.getenv / os.environ[...] /
    os.environ.setdefault read, else None."""
    if isinstance(node, ast.Call):
        f = node.func
        attr = f.attr if isinstance(f, ast.Attribute) else None
        via_environ = isinstance(f, ast.Attribute) and (
            (isinstance(f.value, ast.Attribute) and f.value.attr == "environ")
            or (isinstance(f.value, ast.Name) and f.value.id == "environ")  # from os import environ
        )
        if (attr == "getenv" or (via_environ and attr in ("get", "setdefault"))) and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                return a.value
    elif isinstance(node, ast.Subscript):
        v, s = node.value, node.slice
        if (isinstance(v, ast.Attribute) and v.attr == "environ") or (
            isinstance(v, ast.Name) and v.id == "environ"
        ):
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                return s.value
    return None


_UNPARSEABLE = {
    "mentions": frozenset(),
    "paths": frozenset(),
    "roster": frozenset(),
    "envs": frozenset(),
    "scans": True,
    "native": True,
}


def _facts(text, pkg_parts, marker_re):
    """What one source file can READ beyond its imports: the data-input basenames it names
    in non-docstring string constants, the names it imports from the root package, the
    environment variables it reads by literal name, and whether it lists a directory -- a
    tree scanner. EVERY listing is a scan, whatever its root (B4 C2, 2026-10-20): a
    _SCAN_CALLS call resolved through the file's import aliases (ast.walk and a bare call to a
    function this file defines are not listings; a method call on any receiver is), a lister
    taken as a VALUE (`ls = os.listdir`), and a non-docstring string naming a git tree read
    (_GIT_TREE_READS). Unparseable => scanner."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return dict(_UNPARSEABLE)
    aliases = {}  # local name -> (module, original name | None for `import module [as name]`)
    defined = set()  # functions defined here (methods excluded): a bare call to one is no listing
    methods = set()
    callees = set()  # id() of every Call.func: a lister referenced anywhere else is a scan

    def alias(name, value):
        # the same local name imported two ways (a function-local `import ast as os` beside
        # the module's `import os`) is UNKNOWN: it resolves to no module, so it exempts nothing
        aliases[name] = value if aliases.get(name, value) == value else ("?", "?")

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            callees.add(id(n.func))
        if isinstance(n, ast.Import):
            for a in n.names:
                top = a.name.partition(".")[0]
                alias(a.asname or top, (top, None))
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                alias(a.asname or a.name, (n.module or "", a.name))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(n.name)
        elif isinstance(n, ast.ClassDef):
            methods.update(
                b.name for b in n.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    defined -= methods  # a method named glob/walk never cancels an imported lister

    def module_of(expr):
        """The imported module a dotted callee / receiver chain is rooted at, else None."""
        while isinstance(expr, ast.Attribute):
            expr = expr.value
        if isinstance(expr, ast.Name) and expr.id in aliases and aliases[expr.id][1] is None:
            return aliases[expr.id][0]
        return None

    def resolve(expr):
        """(module, name) a callee or a referenced attribute / name stands for: the module
        is None for a method of a non-module receiver and for the file's own functions."""
        if isinstance(expr, ast.Attribute):
            return module_of(expr.value), expr.attr
        fid = getattr(expr, "id", None)
        if fid in aliases:
            return aliases[fid]
        return None, (None if fid in defined else fid)

    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                doc_ids.add(id(body[0].value))
    mentions, roster, envs, scans, native = set(), set(), set(), False, False
    paths = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in doc_ids:
                mentions.update(marker_re.findall(node.value))
                if _RELPATH.fullmatch(node.value):
                    parts = [p for p in re.split(r"[\\/]", node.value) if p != "."]
                    if ".." not in parts:
                        paths.add("/".join(parts))
                scans = (
                    scans
                    or any(g in node.value for g in _GIT_TREE_READS)
                    or _GIT_ARGV.match(node.value) is not None
                )
        elif isinstance(node, ast.ImportFrom):
            from_root = (node.module == _PKG and not node.level) or (
                node.module is None and node.level == pkg_parts
            )
            if from_root:
                roster.update(a.name for a in node.names)
            # absolute, or relative from the file's own depth (`from .._native import f`
            # two packages down): the root-relative spelling python_native_refs resolves
            native = native or (
                node.module == _NATIVE or (node.level == pkg_parts and node.module == _NATIVE_LEAF)
            )
        elif isinstance(node, ast.Import):
            native = native or any(a.name == _NATIVE for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # a package `__all__` naming the extension loads it for every star importer
            # (list or tuple literal, plain or annotated assignment)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and isinstance(
                node.value, (ast.List, ast.Tuple)
            ):
                native = native or any(
                    isinstance(e, ast.Constant) and e.value == _NATIVE_LEAF for e in node.value.elts
                )
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == _PKG:
                roster.add(node.attr)
        if isinstance(node, (ast.Attribute, ast.Name)) and id(node) not in callees:
            # a lister taken as a value (`ls = os.listdir`, `map(os.listdir, dirs)`): a
            # listing this file cannot classify -- a scan
            mod, name = resolve(node)
            scans = scans or (name in _SCAN_CALLS and mod in _LISTING_MODULES)
        env_name = _env_name(node)
        if env_name is not None:
            envs.add(env_name)
        if isinstance(node, ast.Call):
            mod, name = resolve(node.func)
            # every listing scans, whatever its root; ast.walk is the one non-listing walk
            scans = scans or (name in _SCAN_CALLS and mod != "ast")
    return {
        "mentions": frozenset(mentions),
        "paths": frozenset(paths),
        "roster": frozenset(roster),
        "envs": frozenset(envs),
        "scans": scans,
        "native": native or _NATIVE_LEAF in roster,
    }


# distinct-from: infra/brain_native_selfbuild.py -- its DataIngestSource is a self-build
#   "extension point" (a plug-in slot for data ingestion); this locates the compiled
#   `_native` extension MODULE FILE whose bytes the carry key must hash. Read.
def _native_ext():
    """The compiled extension's file, else the canonical path `maturin develop` fills (hashed
    as ABSENT until it exists, so an override on it -- and the real file later -- flips)."""
    try:
        spec = importlib.util.find_spec(_NATIVE)
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin and os.path.isfile(spec.origin):
        return spec.origin
    return os.path.join(
        _repo_root(), "src", _PKG, _NATIVE_LEAF + importlib.machinery.EXTENSION_SUFFIXES[0]
    )


class SelfcheckCache:
    """consult(mod) -> cached row | None; record(mod, when, secs) after a live PASS."""

    def __init__(
        self,
        file_hash_overrides=None,
        file_text_overrides=None,
        env=None,
        store_path=None,
        no_git=False,
        graph_overrides=None,
        native_root=None,
        rsfacts_root=None,
    ):
        # `file_hash_overrides` ({path: fake_hash}), `file_text_overrides` ({path: source}),
        # `env`, `store_path`, `no_git`, `graph_overrides` ({dotted: {dotted dep}} merged
        # into the import graph), `native_root` (a fixture native/ tree) and `rsfacts_root`
        # (a fixture tools/rsfacts) are the TESTING SEAMS for
        # the self-check's reds: they prove an input change flips exactly the keys
        # it should WITHOUT mutating the tree, the process env or the REAL store (a record()
        # from a probe is a PASS nobody ran -- found by doing exactly that once), and they
        # reach the git-less branch on a host that has git. Never set otherwise.
        self._store_path = store_path or _cache_path()
        self._no_git = bool(no_git)
        self._graph_overrides = {k: set(v) for k, v in (graph_overrides or {}).items()}
        self._native_root = native_root or os.path.join(_repo_root(), _rf.NATIVE_DIR)
        # beside native/ by default, so a fixture native_root brings its own (absent) rsfacts
        self._rsfacts_root = rsfacts_root or os.path.join(
            os.path.dirname(self._native_root), *_rf.RSFACTS.parts
        )
        self._native = None  # (row, files under native/) memo, like _tree
        # every path is normpath'ed at the boundary: git prints forward slashes, os.walk
        # prints os.sep, and an override keyed one way must match a lookup made the other
        self._hash_overrides = {
            os.path.normpath(k): v for k, v in (file_hash_overrides or {}).items()
        }
        self._text_overrides = {
            os.path.normpath(k): v for k, v in (file_text_overrides or {}).items()
        }
        self._env = dict(os.environ if env is None else env)
        self._graph = None
        self._file_hashes = {}
        self._closure_keys = {}
        self._describe = {}
        self._tree = None
        self.tree_surface = 0
        root = _repo_root()
        self._root_init = _module_file(_PKG + ".__init__")
        self._runner = os.path.join(root, "run_selfchecks.py")
        self._data = self._data_inputs()
        markers = sorted({os.path.basename(p) for p in self._data} | {"run_selfchecks"})
        self._marker_re = re.compile("|".join(re.escape(m) for m in markers))
        self._refresh_derived()
        self._store = self._load()

    # -- hashing ---------------------------------------------------------------------
    def _read_text(self, path):
        path = os.path.normpath(path)
        if path in self._text_overrides:
            return self._text_overrides[path]
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            return f.read()

    def _hash_file(self, path):
        """sha256 of a file, memoised per (size, mtime_ns): every read re-stats, so a file
        rewritten mid-run is re-hashed and record() needs no memo purge -- a same-size,
        same-mtime rewrite is the residual, below the frozen-tree rule."""
        path = os.path.normpath(path)
        if path in self._text_overrides:
            return _sha256_text(self._text_overrides[path])
        if path in self._hash_overrides:
            return self._hash_overrides[path]
        try:
            st = os.stat(path)
        except OSError:
            return "ABSENT"
        stamp = (st.st_size, st.st_mtime_ns)
        memo = self._file_hashes.get(path)
        if memo is None or memo[0] != stamp:
            memo = self._file_hashes[path] = (stamp, _sha256_file(path))
        return memo[1]

    def _facts_for(self, path):
        path = os.path.normpath(path)
        rel = os.path.relpath(path, os.path.join(_repo_root(), "src"))
        pkg_parts = max(1, len(rel.split(os.sep)) - 1)  # package components above the file
        if path in self._text_overrides or not os.path.exists(path):
            text = self._text_overrides.get(path, "")
            return _facts(text, pkg_parts, self._marker_re)
        st = os.stat(path)
        memo_key = (path, st.st_size, st.st_mtime_ns, self._marker_re.pattern)
        if memo_key not in _FACTS_MEMO:
            _FACTS_MEMO[memo_key] = _facts(self._read_text(path), pkg_parts, self._marker_re)
        return _FACTS_MEMO[memo_key]

    def _data_inputs(self):
        root = _repo_root()
        out = []
        for d, exts in _DATA_DIRS:
            full = os.path.join(root, d)
            if os.path.isdir(full):
                for f in sorted(os.listdir(full)):
                    p = os.path.join(full, f)
                    ok = (not f.endswith(".py")) if exts is None else f.endswith(exts)
                    if ok and os.path.isfile(p):
                        out.append(p)
        out.append(self._runner)
        always_global = {
            os.path.join(root, "SECURITY.md"),
            os.path.join(root, "docs", "POLICIES.md"),
        }
        return [p for p in out if p not in always_global]

    # distinct-from: gates/envelope_hash_gate.py -- that gate VERIFIES the five
    #   brainstem files against an approved baseline; this merely lists governance-surface
    #   paths (reusing safety_envelope.repo_relative(), the ONE path list) so a change to
    #   any of them invalidates the whole cache. Consumer of the list, not a second
    #   verifier. Read.
    def _global_files(self):
        """The GLOBAL envelope's file rows: SECURITY.md, POLICIES.md, the five brainstem
        files, every NON-root package __init__ (executed on every import of the region, in
        no closure) and the tools/ + models/ artifacts no module names (fail-closed)."""
        root = _repo_root()
        out = [os.path.join(root, "SECURITY.md"), os.path.join(root, "docs", "POLICIES.md")]
        from snn_data_lake.gates.safety_envelope import repo_relative

        out += [os.path.join(root, p) for p in repo_relative()]
        pkg = os.path.join(root, "src", _PKG)
        for r, _d, files in os.walk(pkg):
            if "__init__.py" in files and r != pkg:
                out.append(os.path.join(r, "__init__.py"))
        named = set()
        for r, _d, files in os.walk(pkg):
            for f in files:
                if f.endswith(".py"):
                    named |= self._facts_for(os.path.join(r, f))["mentions"]
        self.unreferenced_artifacts = [
            p
            for p in self._data
            if os.path.basename(os.path.dirname(p)) in _FAIL_CLOSED_DIRS
            and os.path.basename(p) not in named
        ]
        return sorted(out + self.unreferenced_artifacts)

    def _refresh_derived(self):
        """(Re)compute everything derived from the tree: the two roster splits and the
        global envelope. Called at construction and again inside record()."""
        self._norm_root, self._roster_hash, self._roster_names = split_rosters(
            self._read_text(self._root_init)
        )
        self._norm_runner, _, self._runner_rosters = split_rosters(self._read_text(self._runner))
        self.envelope = self._compute_envelope()

    def _compute_envelope(self):
        root = _repo_root()
        gil = getattr(sys, "_is_gil_enabled", lambda: True)()
        nord = "|".join(f"{k}={v}" for k, v in sorted(self._env.items()) if k.startswith("NORD_"))
        gov = self._global_files()
        self.absent_inputs = [p for p in gov if not os.path.exists(p)]
        parts = [
            f"cache_v{_CACHE_VERSION}",
            f"py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            + ("" if gil else "t"),
            "runner_ast:" + self._norm_runner,
            "root_init_ast:" + self._norm_root,
            "nord:" + nord,
        ] + [os.path.relpath(p, root) + ":" + self._hash_file(p) for p in gov]
        # a package __init__ (root or region; in no closure) that re-exports the extension
        # makes the Rust surface a GLOBAL input: every verdict in that region loads it
        inits = [self._root_init] + [p for p in gov if p.endswith("__init__.py")]
        if any(self._facts_for(p)["native"] for p in inits):
            parts += [self._native_hash(), self._ext_row()]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _import_graph(self):
        if self._graph is None:
            from snn_data_lake.gates.dormancy_gate import import_graph

            self._graph = dict(import_graph())
            for node, deps in self._graph_overrides.items():
                self._graph[node] = set(self._graph.get(node, ())) | deps
        return self._graph

    def _closure(self, mod):
        g = self._import_graph()
        start = _PKG_PREFIX + mod
        seen, frontier = {start}, [start]
        while frontier:
            nxt = []
            for node in frontier:
                for dep in g.get(node, ()):
                    if dep not in seen:
                        seen.add(dep)
                        nxt.append(dep)
            frontier = nxt
        return seen

    def _tree_hash(self):
        """sha256 over (path, file sha256) of every tracked + untracked-unignored file --
        the surface a tree scanner reads. Without git it is a per-instance NONCE, so a
        scanner never carries rather than carrying on an unverified surface."""
        if self._tree is None:
            root = _repo_root()
            listed = self._git_files(root)
            if listed is None:
                self._tree = "TREE-SURFACE-UNAVAILABLE:" + os.urandom(8).hex()
                return self._tree
            rows = []
            for rel in sorted(listed):
                p = os.path.join(root, rel)
                if os.path.isfile(p):
                    rows.append(rel + ":" + self._hash_file(p))
            self.tree_surface = len(rows)
            self._tree = _sha256_text("\n".join(rows))
        return self._tree

    def _git_files(self, root):
        """git's view of `root` (tracked + untracked-unignored, root-relative, forward
        slashes), or None when git cannot list it (no_git seam, no git, not a checkout)."""
        if self._no_git:
            return None
        try:
            raw = subprocess.run(
                ["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z"],
                capture_output=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        return [p for p in raw.decode("utf-8", "surrogateescape").split("\0") if p]

    def _native_hash(self):
        """ "native:<sha256>" over (path, sha256) of the Rust surface, or "native:ABSENT" when
        nothing exists. Roots: native/ and tools/rsfacts/ (the fixture crate rust_facts parses),
        target/ skipped under both. Rows: every .rs / manifest / toolchain file, plus every
        other file git lists there -- tracked or untracked-unignored, the `_tree_hash` view
        (runtime data the extension reads; Codex B1 r02b-cache 1) -- and the manifest names
        beside native/ at the repo root. Without git's list EVERY file keys (fail closed). The
        .rs files are counted from disk, not git: the key must move on bytes git cannot see."""
        if self._native is None:
            rows, n_rs = [], 0
            for root, label in ((self._native_root, "native/"), (self._rsfacts_root, "rsfacts/")):
                listed = self._git_files(root)  # None = unavailable, every file keys
                # case-folded: an ignorecase checkout keeps git's spelling of a case-only rename
                listed = None if listed is None else {p.lower() for p in listed}
                for r, dirs, files in os.walk(root):
                    dirs[:] = sorted(d for d in dirs if d != "target")
                    for f in sorted(files):
                        p = os.path.join(r, f)
                        rel = os.path.relpath(p, root).replace(os.sep, "/")
                        if (
                            f.endswith(".rs")
                            or f in _NATIVE_NAMES
                            or listed is None
                            or rel.lower() in listed
                        ):
                            rows.append(label + rel + ":" + self._hash_file(p))
                            # native/ only: n_rs is the "native/ PRESENT (n .rs files)" status
                            n_rs += f.endswith(".rs") and label == "native/"
            for f in sorted(_NATIVE_NAMES):
                p = os.path.join(os.path.dirname(self._native_root), f)
                if os.path.isfile(p):
                    rows.append(f + ":" + self._hash_file(p))
            row = "native:" + (_sha256_text("\n".join(rows)) if rows else "ABSENT")
            self._native = (row, n_rs)
        return self._native[0]

    def _ext_row(self):
        ext = _native_ext()
        return "ext:" + os.path.basename(ext) + ":" + self._hash_file(ext)

    def describe(self, mod):
        """The key's composition for one module: closure size, data inputs and env vars
        carried, whether it imports from the root package or scans the tree. Attribution,
        so a carry -- and a rerun -- is explainable."""
        self.closure_key(mod)
        return self._describe[mod]

    def why(self, mod):
        """One word naming the class that keys this module beyond its closure."""
        d = self.describe(mod)
        if d["scans"]:
            return "scanner"
        if d["native"]:
            return "native"
        if d["roster"]:
            return "root-import"
        if d["data"] or d["envs"]:
            return "data:" + ",".join(d["data"] + ["$" + e for e in d["envs"]])
        return "closure"

    def closure_key(self, mod):
        """The carry key for one MODULES entry ('faculties.percept_surprise'): sha256 over the
        envelope + the closure's (dotted, file-hash) rows + the DERIVED data rows."""
        if mod in self._closure_keys:
            return self._closure_keys[mod]
        root = _repo_root()
        rows, mentions, roster, envs, scans, native = [], set(), set(), set(), False, False
        paths, files = set(), set()
        for d in sorted(self._closure(mod)):
            if d == _NATIVE:  # no .py: its bytes are the native + ext rows below
                native = True
                continue
            native = native or d == _RUST_FACTS  # reads native/**.rs and tools/rsfacts/**
            path = _module_file(d)
            files.add(os.path.normpath(path))
            rows.append(f"{d}={self._hash_file(path)}")
            facts = self._facts_for(path)
            mentions |= facts["mentions"]
            paths |= facts["paths"]
            roster |= facts["roster"]
            envs |= facts["envs"]
            scans = scans or facts["scans"]
            native = native or facts["native"]
        data = {os.path.normpath(p) for p in self._data if os.path.basename(p) in mentions}
        if self._runner in self._data and "run_selfchecks" in mentions:
            data.add(os.path.normpath(self._runner))
        # B4 C4: a file the closure names by literal repo-relative path and that exists is
        # read without an import (its bytes are no closure row); a file already a row is not
        named = (os.path.normpath(os.path.join(root, *s.split("/"))) for s in paths)
        data |= {p for p in named if p not in files and os.path.isfile(p)}
        data_rows = [os.path.relpath(p, root) + ":" + self._hash_file(p) for p in sorted(data)]
        named_envs = sorted(e for e in envs if not e.startswith("NORD_"))
        data_rows += [f"${e}={self._env.get(e, '<unset>')}" for e in named_envs]
        reads_roster = bool(roster)
        if reads_roster:
            data_rows.append("roster:" + self._roster_hash)
        if scans:
            data_rows.append("tree:" + self._tree_hash())
        if native:
            data_rows += [self._native_hash(), self._ext_row()]
        key = hashlib.sha256(
            (self.envelope + "||" + "||".join(rows) + "||" + "||".join(data_rows)).encode()
        ).hexdigest()
        self._closure_keys[mod] = key
        self._describe[mod] = {
            "closure": len(rows),
            "data": sorted(os.path.relpath(p, root) for p in set(data)),
            "envs": named_envs,
            "roster": reads_roster,
            "scans": scans,
            "native": native and (f"{self._native[1]} .rs files" if self._native[1] else "ABSENT"),
            "ext": native and _native_ext(),
        }
        return key

    # -- store ------------------------------------------------------------------------
    def _load(self):
        try:
            with open(self._store_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self):
        path = self._store_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=0)

    def consult(self, mod):
        """The carried row iff a PASS is stored under the CURRENT closure key, else None.
        A mismatched key, a FAIL verdict, or no row all mean: run live."""
        row = self._store.get(mod)
        if not row or row.get("verdict") != "PASS":
            return None
        if row.get("key") != self.closure_key(mod):
            return None
        return row

    def record(self, mod, when, secs=0.0):
        """Record a live PASS under a FRESHLY-validated key (review MAJOR: the memoised
        hashes could be a whole battery old, so a mid-run edit let record() attribute a
        PASS to bytes that never ran -- and that row becomes a wrong carry the moment the
        tree returns to those bytes). Every input the key reads is re-validated at record
        time: file hashes are stat-validated on every read (_hash_file), the tree surface
        is re-listed, the roster splits and the envelope are recomputed. The residual race
        window is milliseconds, with the frozen-tree rule as the primary defence and this
        as the mechanism that does not trust it. `secs` is the live wall-clock, so a later
        carry can say what it saved."""
        self._closure_keys.pop(mod, None)
        self._describe.pop(mod, None)
        self._tree = None  # re-listed + re-stat'ed; only changed files re-hash
        self._native = None
        self._refresh_derived()
        self._store[mod] = {
            "key": self.closure_key(mod),
            "verdict": "PASS",
            "when": when,
            "secs": round(float(secs), 2),
        }
        self._save()

    def banner(self):
        absent = getattr(self, "absent_inputs", [])
        absent_note = (
            " ENVELOPE INPUTS ABSENT (each hashed as the literal ABSENT -- loud, not "
            f"silent): {', '.join(os.path.basename(p) for p in absent)}."
            if absent
            else ""
        )
        return (
            f"CARRY-FORWARD ON (envelope v{_CACHE_VERSION} {self.envelope[:12]}..., "
            f"{len(self._store)} cached verdicts): a PASS carries only when the module's "
            "ENTIRE static import closure + the data files / literal-relpath files / named env "
            "vars / root-package "
            "roster / tree surface / Rust surface (native/**, Cargo manifests, the loaded "
            "extension; for _native readers) its closure reads (DERIVED per module) + the global "
            "envelope (runner and root __init__ logic, region __init__s, brainstem safety "
            "files, SECURITY.md, POLICIES.md, NORD_* env, interpreter) are byte-identical. "
            f"BLIND to {banner_blind_spots()}.{absent_note}"
        )


def banner_blind_spots():
    return (
        "string-dispatch imports; shared-home sequence coupling; unhashed host state; "
        "computed-name data/env reads (tools/, models/ and NORD_* fail closed); gitignored "
        "inputs read by name outside models/ and graphify-out/; a lister not named through "
        "the file's imports (string dispatch: getattr(os, 'walk')); a tree listed by a "
        "subprocess other than git (find, rg), or by git where no argv string STARTS with 'git' "
        "('cd src && git status', '/usr/bin/git ...', a git path in a variable); "
        "gitignored non-Rust files under native/ (when git lists it); "
        "directory symlinks under native/; "
        "Rust build inputs outside native/ and the manifest names "
        "(.cargo/config.toml, RUSTFLAGS, the toolchain binary) (carries = local batteries "
        "only; any carried-run FAIL reruns FULL first)"
    )


def _insert_after(text, anchor, snippet):
    i = text.index(anchor) + len(anchor)
    return text[:i] + snippet + text[i:]


def plain_probe(cache):
    """(desc, plain modules, probe): the census and the key-invariance control subject. desc
    is describe() for every in-package module the ONE import graph knows whose file exists
    (no runner import -- run_selfchecks.py probes torch at import; the graph is the census);
    a PLAIN module carries no derived class at all (scans / roster / data / envs / native);
    the probe is the first plain module whose closure spans more than one file --
    engine.signal_ring when it qualifies -- or None. The self-check and the pytest reds must
    share this census: a forked copy drifts on the next derived class."""
    mods = sorted(
        d[len(_PKG_PREFIX) :]
        for d in cache._import_graph()
        if d.startswith(_PKG_PREFIX) and os.path.isfile(_module_file(d))
    )
    desc = {m: cache.describe(m) for m in mods}
    plain = [
        m
        for m in desc
        if not any(desc[m][k] for k in ("scans", "roster", "data", "envs", "native"))
    ]
    first = ["engine.signal_ring"] if "engine.signal_ring" in plain else []
    probe = next((m for m in first + plain if len(cache._closure(m)) > 1), None)
    return desc, plain, probe


_HOME_IMPORT = "import os\nfrom snn_data_lake.substrate.home import home\n"
_SYNTHETIC = {  # unit reds for the derivation itself, pinned on sources that cannot rot
    "attr_file_scan": (
        "import os\ndef files(pkg):\n    d = os.path.dirname(pkg.__file__)\n"
        "    return os.listdir(d)\n",
        "scans",
        True,
    ),
    "return_rooted_scan": (
        "import os\ndef _root():\n    return os.path.dirname(__file__)\nroot = _root()\n"
        "for f in os.listdir(root):\n    pass\n",
        "scans",
        True,
    ),
    "ast_walk_not_scan": (
        "import ast\ntree = ast.parse('x')\nfor n in ast.walk(tree):\n    pass\n",
        "scans",
        False,
    ),
    # carryfix (Opus B3 M1): a parameter / keyword / cwd / git-listed root is a scan
    "parameter_root_scan": (
        "import os\ndef sites(pkg_dir):\n    for r, d, f in os.walk(pkg_dir):\n        pass\n",
        "scans",
        True,
    ),
    "parameter_rglob_scan": (
        "from pathlib import Path\ndef n(src):\n    return sorted(Path(src).rglob('*.py'))\n",
        "scans",
        True,
    ),
    "keyword_root_scan": ("import os\ndef s(d):\n    os.walk(top=d)\n", "scans", True),
    "annassign_root_scan": (
        "import os\ndef s(p):\n    d: str = p\n    os.listdir(d)\n",
        "scans",
        True,
    ),
    "comprehension_root_scan": (
        "import os\ndef s(ps):\n    return [os.listdir(p) for p in ps]\n",
        "scans",
        True,
    ),
    "cwd_glob_scan": ("import glob\nnames = glob.glob('*.py')\n", "scans", True),
    "no_root_listdir_scan": ("import os\nnames = os.listdir()\n", "scans", True),
    "ls_files_scan": (
        'import subprocess\nsubprocess.run(["git", "ls-files", "-co"], check=True)\n',
        "scans",
        True,
    ),
    "ls_files_docstring_not_scan": ('"""keyed like git ls-files"""\nimport os\n', "scans", False),
    # the polish review's fail-open shapes (correctness / security / coverage, 2026-10-20):
    # an excluded call INSIDE a tree root, name collisions, aliases -- each pinned as a scan
    "env_default_file_scan": (
        "import os\nos.walk(os.environ.get('R', os.path.dirname(__file__)))\n",
        "scans",
        True,
    ),
    "env_or_file_scan": (
        "import os\nR = os.environ.get('R') or os.path.dirname(__file__)\nos.listdir(R)\n",
        "scans",
        True,
    ),
    "home_joined_file_scan": (
        _HOME_IMPORT + "os.listdir(os.path.join(home(), os.path.dirname(__file__)))\n",
        "scans",
        True,
    ),
    "parameter_collision_scan": (
        _HOME_IMPORT + "def a():\n    root = home()\ndef b(root):\n    os.walk(root)\n",
        "scans",
        True,
    ),
    "rebind_scan": (
        _HOME_IMPORT + "d = home()\nd = os.path.dirname(__file__)\nos.listdir(d)\n",
        "scans",
        True,
    ),
    "two_returns_scan": (
        _HOME_IMPORT + "def r(x):\n    if x:\n        return home()\n    return "
        "os.path.dirname(__file__)\nos.listdir(r(1))\n",
        "scans",
        True,
    ),
    "self_attribute_scan": (
        _HOME_IMPORT + "class C:\n    def __init__(s, pkg):\n        s.store = home()\n"
        "        s.pkg = pkg\n    def f(s):\n        return list(s.pkg.glob('*'))\n",
        "scans",
        True,
    ),
    "imported_root_mixed_scan": (
        _HOME_IMPORT + "from snn_data_lake.paths import PKG_DIR\nh = home()\n"
        "os.listdir(os.path.join(PKG_DIR, h))\n",
        "scans",
        True,
    ),
    "module_attribute_root_scan": (
        "import os\nfrom snn_data_lake.gates import rust_facts\nos.walk(rust_facts.ROOT)\n",
        "scans",
        True,
    ),
    "unknown_call_scan": ("import os, tempfile\nos.listdir(tempfile.mkdtemp())\n", "scans", True),
    "method_named_home_scan": (
        "import os\nclass C:\n    def home(self):\n        return '/x'\nos.walk(C().home())\n",
        "scans",
        True,
    ),
    "os_alias_walk_scan": ("import os as o\ndef f(p):\n    o.walk(p)\n", "scans", True),
    "from_os_walk_scan": ("from os import walk\ndef f(p):\n    walk(p)\n", "scans", True),
    "from_os_listdir_alias_scan": (
        "from os import listdir as ls\ndef f(p):\n    ls(p)\n",
        "scans",
        True,
    ),
    "pathlib_walk_scan": (
        "from pathlib import Path\nfor r, d, f in Path(__file__).parent.walk():\n    pass\n",
        "scans",
        True,
    ),
    "mixed_roots_scan": (
        _HOME_IMPORT + "import glob\ndef f(p):\n    glob.glob(home(), root_dir=p)\n",
        "scans",
        True,
    ),
    # the non-listings that must survive: a bare call to the file's own function
    "local_def_walk_not_scan": (
        "def walk(a, depth):\n    walk(a, depth + 1)\nwalk(1, 0)\n",
        "scans",
        False,
    ),
    "method_walk_scan": ("def f(drive, ep):\n    return drive.walk(ep)\n", "scans", True),
    "nested_def_walk_not_scan": (
        "def census():\n    def walk(a, depth):\n        walk(a, depth + 1)\n    walk(1, 0)\n",
        "scans",
        False,
    ),
    # polish round 2 (correctness / security / coverage mutants, 2026-10-20)
    "lambda_param_scan": (
        _HOME_IMPORT + "d = home()\nf = lambda d: os.listdir(d)\n",
        "scans",
        True,
    ),
    "kwonly_param_scan": (
        _HOME_IMPORT + "d = home()\ndef f(*, d):\n    os.listdir(d)\n",
        "scans",
        True,
    ),
    "posonly_param_scan": (
        _HOME_IMPORT + "d = home()\ndef f(d, /):\n    os.listdir(d)\n",
        "scans",
        True,
    ),
    "kwargs_param_scan": (
        _HOME_IMPORT + "d = home()\ndef f(**d):\n    os.listdir(d['x'])\n",
        "scans",
        True,
    ),
    "augassign_tree_rebind_scan": (
        _HOME_IMPORT + "d = home()\nd += os.path.dirname(__file__)\nos.listdir(d)\n",
        "scans",
        True,
    ),
    "for_tree_rebind_scan": (
        _HOME_IMPORT + "d = home()\nfor d in [os.path.dirname(__file__)]:\n    os.listdir(d)\n",
        "scans",
        True,
    ),
    "comprehension_tree_rebind_scan": (
        _HOME_IMPORT + "d = home()\nx = [os.listdir(d) for d in [os.path.dirname(__file__)]]\n",
        "scans",
        True,
    ),
    "walrus_tree_rebind_scan": (
        _HOME_IMPORT + "d = home()\nif (d := os.path.dirname(__file__)):\n    os.listdir(d)\n",
        "scans",
        True,
    ),
    "tuple_tree_rebind_scan": (
        _HOME_IMPORT + "d = home()\nd, e = os.path.dirname(__file__), 1\nos.listdir(d)\n",
        "scans",
        True,
    ),
    "foreign_home_scan": (
        "import os\nfrom mypkg.util import home\nos.listdir(home())\n",
        "scans",
        True,
    ),
    "substrate_other_module_home_scan": (
        "import os\nfrom snn_data_lake.substrate import paths as hm\nos.listdir(hm.home())\n",
        "scans",
        True,
    ),
    "shadowed_wrapper_scan": (
        _HOME_IMPORT + "def sorted(x):\n    return os.path.dirname(__file__)\n"
        "os.listdir(sorted(home()))\n",
        "scans",
        True,
    ),
    "no_arg_wrapper_scan": (
        _HOME_IMPORT
        + "import glob\nfrom pathlib import Path\nglob.glob(home(), root_dir=Path())\n",
        "scans",
        True,
    ),
    "module_non_wrapper_call_scan": (
        _HOME_IMPORT + "import shutil\nos.listdir(shutil.copytree(home(), home()))\n",
        "scans",
        True,
    ),
    "method_in_root_scan": (
        _HOME_IMPORT + "def f(p):\n    os.listdir(os.path.join(home(), p.strip()))\n",
        "scans",
        True,
    ),
    "bare_return_scan": (
        _HOME_IMPORT
        + "def r(x):\n    if x:\n        return\n    return home()\nos.listdir(r(1))\n",
        "scans",
        True,
    ),
    "fallthrough_return_scan": (
        _HOME_IMPORT + "def r(x):\n    if x:\n        return home()\nos.listdir(r(0))\n",
        "scans",
        True,
    ),
    "generator_yield_param_scan": (
        _HOME_IMPORT + "def roots(p):\n    yield p\n    return home()\ndef g(x):\n"
        "    for r in roots(x):\n        os.walk(r)\n",
        "scans",
        True,
    ),
    "env_default_cwd_scan": ("import os\nos.walk(os.environ.get('X', '.'))\n", "scans", True),
    "env_or_cwd_scan": ("import os\nR = os.environ.get('W') or '.'\nos.walk(R)\n", "scans", True),
    "list_mixed_constant_scan": (
        _HOME_IMPORT + "for r in [home(), 'src']:\n    os.walk(r)\n",
        "scans",
        True,
    ),
    "ifexp_constant_scan": (
        _HOME_IMPORT + "os.walk(home() if os.environ.get('A') else 'src')\n",
        "scans",
        True,
    ),
    "import_fallback_scan": (
        _HOME_IMPORT + "try:\n    from .cfg import ROOT\nexcept ImportError:\n    ROOT = home()\n"
        "os.walk(ROOT)\n",
        "scans",
        True,
    ),
    "except_as_rebind_scan": (
        _HOME_IMPORT + "def a():\n    d = home()\ndef b():\n    try:\n        pass\n"
        "    except Exception as d:\n        os.listdir(d)\n",
        "scans",
        True,
    ),
    "match_capture_scan": (
        _HOME_IMPORT + "d = home()\ndef f(v):\n    match v:\n        case str(d):\n"
        "            os.listdir(d)\n",
        "scans",
        True,
    ),
    "class_name_scan": (
        _HOME_IMPORT + "def a():\n    C = home()\nclass C:\n    pass\nos.listdir(C)\n",
        "scans",
        True,
    ),
    "star_import_scan": (
        _HOME_IMPORT + "from snn_data_lake.paths import *\nROOT = home()\nos.walk(ROOT)\n",
        "scans",
        True,
    ),
    "method_shadow_glob_scan": (
        "from glob import glob\nclass C:\n    def glob(self, p):\n        pass\nglob('src/**/*.py')\n",
        "scans",
        True,
    ),
    "method_shadow_listdir_scan": (
        "from os import listdir\nclass S:\n    def listdir(self):\n        pass\n"
        "def f(p):\n    return listdir(p)\n",
        "scans",
        True,
    ),
    "lister_assigned_scan": (
        "import os\ndef f(p):\n    ls = os.listdir\n    return ls(p)\n",
        "scans",
        True,
    ),
    "lister_mapped_scan": (
        "import os\ndef f(ps):\n    return list(map(os.listdir, ps))\n",
        "scans",
        True,
    ),
    "from_import_lister_value_scan": (
        "from os import listdir\ndef f(ps):\n    return list(map(listdir, ps))\n",
        "scans",
        True,
    ),
    "fwalk_scan": ("import os\ndef f(p):\n    os.fwalk(p)\n", "scans", True),
    "pkgutil_walk_packages_scan": (
        "import pkgutil\ndef f(pkg):\n    pkgutil.walk_packages(pkg.__path__)\n",
        "scans",
        True,
    ),
    "home_parent_scan": (_HOME_IMPORT + "os.walk(home().parent.parent)\n", "scans", True),
    "home_dotdot_scan": (_HOME_IMPORT + "os.walk(os.path.join(home(), '..'))\n", "scans", True),
    "git_porcelain_scan": (
        'import subprocess\nsubprocess.run(["git", "status", "--porcelain"])\n',
        "scans",
        True,
    ),
    "git_ls_tree_scan": (
        'import subprocess\nsubprocess.run(["git", "ls-tree", "-r", "HEAD"])\n',
        "scans",
        True,
    ),
    # polish round 3 (peer + correctness / security / coverage, 2026-10-20)
    "local_alias_shadow_scan": (
        "import os\nlist(os.walk(os.path.dirname(__file__)))\ndef helper():\n"
        "    import ast as os\n    return os.walk\n",
        "scans",
        True,
    ),
    "fake_environ_scan": (
        "import os\nclass Config:\n    pass\ncfg = Config()\n"
        "cfg.environ = {'ROOT': os.path.dirname(__file__)}\nos.listdir(cfg.environ['ROOT'])\n",
        "scans",
        True,
    ),
    "env_keyword_default_scan": (
        "import os\nos.listdir(os.getenv('ZZZ_D', default='.'))\n",
        "scans",
        True,
    ),
    "list_append_scan": (
        _HOME_IMPORT + "roots = [home()]\nroots.append(os.path.dirname(__file__))\n"
        "for r in roots:\n    os.listdir(r)\n",
        "scans",
        True,
    ),
    "subscript_assign_scan": (
        _HOME_IMPORT
        + "roots = [home()]\nroots[0] = os.path.dirname(__file__)\nos.listdir(roots[0])\n",
        "scans",
        True,
    ),
    "dict_assign_scan": (
        _HOME_IMPORT + "d = {'a': home()}\nd['a'] = __file__\nos.listdir(d['a'])\n",
        "scans",
        True,
    ),
    "dirname_home_scan": (_HOME_IMPORT + "os.listdir(os.path.dirname(home()))\n", "scans", True),
    "basename_home_scan": (_HOME_IMPORT + "os.listdir(os.path.basename(home()))\n", "scans", True),
    "absolute_join_scan": (
        _HOME_IMPORT + "os.listdir(os.path.join(home(), '/mnt/d/AI-project'))\n",
        "scans",
        True,
    ),
    "drive_join_scan": (
        _HOME_IMPORT
        + "from pathlib import Path\nlist((Path(home()) / 'D:/AI-project').iterdir())\n",
        "scans",
        True,
    ),
    "slice_home_scan": (_HOME_IMPORT + "os.walk(home()[:-15])\n", "scans", True),
    "replace_home_scan": (_HOME_IMPORT + "os.walk(home().replace('home', 'src'))\n", "scans", True),
    "home_name_scan": (
        _HOME_IMPORT + "from pathlib import Path\nos.listdir(Path(home()).name)\n",
        "scans",
        True,
    ),
    "home_relative_to_scan": (
        _HOME_IMPORT + "from pathlib import Path\nos.listdir(Path(home()).relative_to(home()))\n",
        "scans",
        True,
    ),
    "home_parents_scan": (_HOME_IMPORT + "os.walk(home().parents[1])\n", "scans", True),
    "home_rebound_scan": (_HOME_IMPORT + "home = os.getcwd\nos.listdir(home())\n", "scans", True),
    "home_redefined_scan": (
        _HOME_IMPORT + "def home():\n    return '.'\nos.listdir(home())\n",
        "scans",
        True,
    ),
    "foreign_join_scan": (
        _HOME_IMPORT + "from snn_data_lake.util import join\nos.listdir(join(home()))\n",
        "scans",
        True,
    ),
    "shlex_join_scan": (
        _HOME_IMPORT + "import shlex\nos.listdir(shlex.join([home()]))\n",
        "scans",
        True,
    ),
    "ifexp_body_constant_scan": (
        _HOME_IMPORT + "d = 'data' if os.sep else home()\nos.listdir(d)\n",
        "scans",
        True,
    ),
    "tuple_mixed_constant_scan": (
        _HOME_IMPORT + "for d in ('data', home()):\n    os.listdir(d)\n",
        "scans",
        True,
    ),
    "set_mixed_constant_scan": (
        _HOME_IMPORT + "for d in {'data', home()}:\n    os.listdir(d)\n",
        "scans",
        True,
    ),
    "glob_value_scan": ("import glob\nx = list(map(glob.glob, ['src/**']))\n", "scans", True),
    "from_glob_value_scan": (
        "from glob import glob\nx = list(map(glob, ['src/**']))\n",
        "scans",
        True,
    ),
    "pathlib_value_scan": ("import pathlib\nls = pathlib.Path.iterdir\n", "scans", True),
    "pkgutil_value_scan": (
        "import pkgutil\nit = pkgutil.walk_packages\nit(['src'])\n",
        "scans",
        True,
    ),
    "star_import_method_shadow_scan": (
        "from glob import *\nclass C:\n    def glob(self, p):\n        pass\nglob('src/**/*.py')\n",
        "scans",
        True,
    ),
    "async_method_shadow_scan": (
        "from os import listdir\nclass S:\n    async def listdir(self):\n        pass\n"
        "def f(p):\n    return listdir(p)\n",
        "scans",
        True,
    ),
    "iter_modules_scan": (
        "import pkgutil\nfor m in pkgutil.iter_modules(['src']):\n    pass\n",
        "scans",
        True,
    ),
    "scandir_scan": ("import os\nfor e in os.scandir('src'):\n    pass\n", "scans", True),
    "iglob_scan": ("import glob\nfor p in glob.iglob('src/**'):\n    pass\n", "scans", True),
    "iterdir_scan": (
        "from pathlib import Path\nfor p in Path('src').iterdir():\n    pass\n",
        "scans",
        True,
    ),
    "yield_from_param_scan": (
        _HOME_IMPORT + "def g(p):\n    yield from p\n    return home()\n"
        "def f(x):\n    for d in g(x):\n        os.listdir(d)\n",
        "scans",
        True,
    ),
    "match_star_scan": (
        _HOME_IMPORT
        + "d = home()\ndef f(v):\n    match v:\n        case [*d]:\n            os.listdir(d)\n",
        "scans",
        True,
    ),
    "match_mapping_rest_scan": (
        _HOME_IMPORT
        + "d = home()\ndef f(v):\n    match v:\n        case {**d}:\n            os.listdir(d)\n",
        "scans",
        True,
    ),
    "vararg_scan": (
        _HOME_IMPORT + "d = home()\ndef f(*d):\n    return os.listdir(d)\n",
        "scans",
        True,
    ),
    "starred_target_scan": (
        _HOME_IMPORT + "import sys\nd = home()\n_, *d = sys.argv\nos.listdir(d)\n",
        "scans",
        True,
    ),
    "ast_walk_value_not_scan": ("import ast\nx = list(map(ast.walk, []))\n", "scans", False),
    "from_ast_walk_not_scan": (
        "from ast import walk\nimport ast\nfor n in walk(ast.parse('x')):\n    pass\n",
        "scans",
        False,
    ),
    # B4 C2: the closing review's constructed stale shapes the deleted root resolver passed
    # (S2-S5 also pass a home()/env-literal-only exclusion; S6, S8 only its wrappers) -- each a scan
    "s2_env_seeded_from_file_scan": (
        "import os\nos.environ.setdefault('ZZZ_S2', os.path.dirname(__file__))\n"
        "os.listdir(os.environ['ZZZ_S2'])\n",
        "scans",
        True,
    ),
    "s3_env_pythonpath_walk_scan": (
        "import os\nfor r, d, f in os.walk(os.environ['PYTHONPATH']):\n    pass\n",
        "scans",
        True,
    ),
    "s4_home_attribute_rebound_scan": (
        "import os\nfrom snn_data_lake.substrate import home\n"
        "home.home = lambda: os.path.dirname(__file__)\nos.listdir(home.home())\n",
        "scans",
        True,
    ),
    "s5_globals_home_rebound_scan": (
        _HOME_IMPORT + "globals()['home'] = os.getcwd\nos.listdir(home())\n",
        "scans",
        True,
    ),
    "s6_home_snapshot_of_tree_scan": (
        _HOME_IMPORT + "import shutil\nsnap = os.path.join(home(), 'snap')\n"
        "shutil.copytree(os.path.dirname(__file__), snap)\nos.walk(snap)\n",
        "scans",
        True,
    ),
    # B4 C3: any git argv (review S7) -- and the names that merely contain "git" stay quiet
    "s7_git_diff_name_only_scan": (
        "import subprocess\nout = subprocess.run(['git', 'diff', '--name-only'], "
        "capture_output=True).stdout\n",
        "scans",
        True,
    ),
    "s7_git_status_short_shell_scan": (
        "import subprocess\nout = subprocess.check_output('git status --short', shell=True)\n",
        "scans",
        True,
    ),
    "git_named_paths_not_scan": (
        "IGNORED = ('.git', '.gitignore', 'gitattributes', 'github', '.github/workflows')\n",
        "scans",
        False,
    ),
    "s8_home_split_dotdot_scan": (
        _HOME_IMPORT + "os.listdir(home() + '/.' + './x')\n",
        "scans",
        True,
    ),
    "env_reads": (
        'import os\nSEED = int(os.environ.get("ZZZ_SEED", 0))\nX = os.getenv("ZZZ_X")\n'
        'Y = os.environ["ZZZ_Y"]\nos.environ.setdefault("ZZZ_Z", "1")\n',
        "envs",
        frozenset({"ZZZ_SEED", "ZZZ_X", "ZZZ_Y", "ZZZ_Z"}),
    ),
    # B4 C4: a whole-string repo-relative path is a literal read; prose, `..` and bare names are not
    "relpath_reads": (
        "A = 'src/pkg/mod.py'\nB = 'docs\\\\x.md'\nC = ('see src/x.py', '../e/x.py', 'x.py', 'a/../b')\n",
        "paths",
        frozenset({"src/pkg/mod.py", "docs/x.md"}),
    ),
    "native_from_import": ("from snn_data_lake._native import f\n", "native", True),
    "native_relative": ("from ._native import f\n", "native", True),  # pkg_parts 1 = root-relative
    "native_other_package": ("from .other._native import f\n", "native", False),
    "native_import": ("import snn_data_lake._native\n", "native", True),
    "native_from_root": ("from snn_data_lake import _native\n", "native", True),
    "native_not_read": ("import os\n", "native", False),
}


def self_check() -> int:
    """The P8 reds, each constructed and fired, and their KEY-INVARIANCE controls: an
    input change must flip the keys that can see it and must NOT flip the keys that
    cannot. v2 proved only the flip half, which a cache that invalidates everything
    passes trivially -- exactly what v2 did on every task. Every instance here uses a
    TEMP store: nothing reads or writes the real one.
    (0) DERIVATION -- synthetic sources pin the scanner / env rules;
    (a) DEPENDENCY INVALIDATION -- a closure dependency's hash flips the module's key;
    (b) POISON CONTROL -- a stored PASS under a STALE key must NOT carry while the same
        row under the current key must, consulted from a FRESH instance over the shared
        temp store (the control is a second key derivation, not the planter's memo);
    (c) ENVELOPE INVALIDATION -- a changed NORD_* env flips every key;
    (d) FAIL never carries, even under a matching key;
    (e) SAFETY SURFACE -- a brainstem envelope file change flips every key;
    (f) ROSTER SPLIT, root __init__ -- a REGIONS row append flips every root-package
        importer (a flat-alias importer included) and leaves a plain module alone; any
        non-roster edit flips every key; the blanked literals are the pinned tuples;
    (g) ROSTER SPLIT, runner -- a tier-list row append flips nothing plain; an _ENV edit
        flips every key;
    (h) DATA INPUT -- an artifact's change flips its reader and not a plain module;
    (i) TREE SCANNER -- a change to a tracked file NO closure names flips a scanner's key
        and not a plain module's (when git is available);
    (j) FAIL-CLOSED -- an unreferenced tools/ or models/ artifact change flips every key;
    (k) GIT-LESS -- through the no_git seam: a scanner's PASS recorded under one instance's
        nonce is refused by a fresh instance over the SAME store, and the two keys differ
        (the red fires if the nonce ever degenerates into a constant);
    (l) NAMED ENV -- a variable a module reads by name flips that module and not a plain
        one;
    (m) RUST SURFACE -- with `{probe: {_native}}` injected into the graph and a fixture
        native/ tree: a `.rs` byte change, a Cargo.toml change and an extension-bytes change
        each flip the probe's key; the same flips leave the uninjected plain probe alone;
        the real tree's status (ABSENT today) is printed distinctly;
    (n) UNFLAGGED TREE-WALKERS -- a parameter-rooted walk (replication_guard, native_ratchet)
        and a `git ls-files` lister (rust_facts) flip on a file no closure names / a .rs
        edit, and the plain probe holds under both;
    (o) LITERAL RELPATH READS (B4 C4) -- an engine/brain.py change flips
        eligible_circuit_census (it reads brain.py by literal relpath, never imports it) and
        leaves the plain probe alone."""
    # distinct-from: reasoning/analogy_bench.py -- its controls() are NEGATIVE ANALOGY
    #   cases a structure mapper must fail to recover; these are cache-KEY invariance
    #   checks (a key that must not move). Read.
    # distinct-from: instruments/port_charge_probe.py -- margin_frozen() is the charge
    #   probe's not-consuming tell over arbitrate margins; no relation to hash keys. Read.
    import shutil
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="selfcheck_cache_")
    try:
        return _self_check(os.path.join(tmpdir, "store.json"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _self_check(store):
    env = {"NORD_GPU": "1"}
    root = _repo_root()

    def fail(msg):
        print("selfcheck_cache self-check FAILED: " + msg)
        return 1

    def make(**kw):
        kw.setdefault("env", env)
        return SelfcheckCache(store_path=store, **kw)

    fired, held = [], []

    # (0) the derivation, pinned on synthetic sources
    for name, (src, field, want) in _SYNTHETIC.items():
        got = _facts(src, 1, _NULL_RE)[field]
        if got != want:
            return fail(f"derivation red {name}: {field} = {got!r}, expected {want!r}")
    n_scan_pins = sum(1 for _s, field, _w in _SYNTHETIC.values() if field == "scans")
    fired.append(
        f"{n_scan_pins} scanner-derivation pins (every recognised listing scans whatever its root:"
        " parameter / keyword / imported / aliased / cwd / home() / env-dir roots, the closing"
        " review's S2-S8, ls-files, any git argv; ast.walk and the file's own functions are no listing)"
        " + env-name pins"
    )

    base = make()
    if base._roster_names != _EXPECTED_ROSTERS["root"]:
        return fail(
            f"root __init__ rosters {base._roster_names} != pinned {_EXPECTED_ROSTERS['root']}"
            " -- a newly blanked literal must be adjudicated here, not inherited"
        )
    if base._runner_rosters != _EXPECTED_ROSTERS["runner"]:
        return fail(
            f"runner rosters {base._runner_rosters} != pinned {_EXPECTED_ROSTERS['runner']}"
        )

    desc, plain, probe = plain_probe(base)
    mods = sorted(desc)
    scanners = [m for m in mods if desc[m]["scans"]]
    root_importers = [m for m in mods if desc[m]["roster"]]
    data_readers = [m for m in mods if desc[m]["data"]]
    env_readers = [m for m in mods if desc[m]["envs"]]
    if not plain or not scanners or not root_importers or not data_readers or not env_readers:
        return fail(
            f"the derivation left a class empty (plain {len(plain)}, scanners {len(scanners)}, "
            f"root importers {len(root_importers)}, data readers {len(data_readers)}, env "
            f"readers {len(env_readers)}) -- a cache whose every module is a scanner carries "
            "nothing on a row append"
        )
    if probe is None:
        return fail("no plain module with a multi-file closure to probe")
    k0 = base.closure_key(probe)

    def unchanged(fork, what):
        """Key-invariance check: the plain probe's key must NOT move under `fork`."""
        if fork.closure_key(probe) != k0:
            return fail(f"{what} flipped a plain module's key")
        held.append(what)
        return None

    # (a) dependency invalidation: a DEP's hash (not the module's own) must move the key
    dep = next(_module_file(d) for d in sorted(base._closure(probe)) if d != _PKG_PREFIX + probe)
    if make(file_hash_overrides={dep: "0" * 64}).closure_key(probe) == k0:
        return fail("a changed DEPENDENCY did not flip the key")
    fired.append("dependency change")

    # (b) poison control, both directions, consulted from a FRESH instance over the store
    base._store = {probe: {"key": "STALE" + k0[5:], "verdict": "PASS", "when": "x"}}
    base._save()
    if make().consult(probe) is not None:
        return fail("a stale-key PASS was carried (poison)")
    base._store = {probe: {"key": k0, "verdict": "PASS", "when": "x"}}
    base._save()
    if make().consult(probe) is None:
        return fail("a current-key PASS did not carry from a fresh instance")
    fired.append("stale-key poison refused / current-key carry accepted (fresh instance)")

    # (c) envelope invalidation: different NORD_* env -> different key
    if make(env={"NORD_GPU": "0"}).closure_key(probe) == k0:
        return fail("a changed NORD_* env did not flip the key")
    fired.append("NORD_* env")

    # (d) FAIL never carries
    base._store = {probe: {"key": k0, "verdict": "FAIL", "when": "x"}}
    base._save()
    if make().consult(probe) is not None:
        return fail("a FAIL verdict was carried")
    fired.append("FAIL never carries")

    # (e) THE SAFETY-SURFACE RED: a brainstem envelope file -- data the envelope_hash_gate
    # reads, invisible to every import closure -- must flip the ENVELOPE and every key
    from snn_data_lake.gates.safety_envelope import repo_relative

    brainstem_file = os.path.join(root, sorted(repo_relative())[0])
    tampered = make(file_hash_overrides={brainstem_file: "f" * 64})
    if tampered.envelope == base.envelope or tampered.closure_key(probe) == k0:
        return fail("a brainstem safety-file change did not invalidate the cache")
    fired.append("brainstem safety file")

    # (f) ROSTER SPLIT, root __init__: a REGIONS row append flips every root-package
    # importer -- a flat-alias importer that names no roster included -- and leaves the
    # plain probe alone; any non-roster edit flips everything
    init_text = base._read_text(base._root_init)
    # line-anchored (G8 F17): `LEGACY_REGIONS = {` also contains the bare substring
    anchor = re.search(r"(?m)^REGIONS = \{", init_text)
    if anchor is None:
        return fail("the REGIONS anchor for the roster reds is missing")
    i = init_text.index("[", anchor.start()) + 1
    row_fork = make(
        file_text_overrides={base._root_init: init_text[:i] + '"zzz_probe_v3", ' + init_text[i:]}
    )
    if row_fork._roster_names != base._roster_names:
        return fail("a REGIONS row append changed which names are rosters")
    alias_importers = [
        m
        for m in root_importers
        if not desc[m]["scans"]
        and not any(
            n in _EXPECTED_ROSTERS["root"]
            for d in base._closure(m)
            for n in base._facts_for(_module_file(d))["roster"]
        )
    ]
    readers = [next((m for m in root_importers if not desc[m]["scans"]), root_importers[0])]
    if alias_importers:
        readers.append(alias_importers[0])
    readers = list(dict.fromkeys(readers))
    for reader in readers:
        if row_fork.closure_key(reader) == base.closure_key(reader):
            return fail(f"a REGIONS row append did not flip root-package importer {reader}")
    fired.append(
        f"REGIONS row -> {', '.join(readers)} flipped"
        + ("" if alias_importers else " (no non-scanner flat-alias importer on this tree)")
    )
    if (r := unchanged(row_fork, "a REGIONS row append")) is not None:
        return r
    logic_fork = make(file_text_overrides={base._root_init: init_text + "\nZZZ_PROBE_V3 = 1\n"})
    if logic_fork.envelope == base.envelope or logic_fork.closure_key(probe) == k0:
        return fail("a non-roster root __init__ edit did not invalidate the cache")
    fired.append("root __init__ logic edit")

    # (g) ROSTER SPLIT, runner: a tier row append flips nothing plain; an _ENV edit flips all
    runner_text = base._read_text(base._runner)
    if "CORE = [" not in runner_text or '"PYTHONUTF8": "1"' not in runner_text:
        return fail("runner anchors for the roster reds are missing")
    tier_fork = make(
        file_text_overrides={
            base._runner: _insert_after(runner_text, "CORE = [", '\n    "zzz.probe_v3",')
        }
    )
    if (r := unchanged(tier_fork, "a runner tier-row append")) is not None:
        return r
    env_fork = make(
        file_text_overrides={
            base._runner: runner_text.replace('"PYTHONUTF8": "1"', '"PYTHONUTF8": "0"', 1)
        }
    )
    if env_fork.envelope == base.envelope or env_fork.closure_key(probe) == k0:
        return fail("a runner _ENV edit did not invalidate the cache")
    fired.append("runner _ENV edit")

    # (h) DATA INPUT: an artifact's change flips its reader and not the plain probe
    data_reader = next((m for m in data_readers if not desc[m]["scans"]), data_readers[0])
    artifact = os.path.join(root, desc[data_reader]["data"][0])
    art_fork = make(file_hash_overrides={artifact: "a" * 64})
    if art_fork.closure_key(data_reader) == base.closure_key(data_reader):
        return fail(
            f"a change to {os.path.relpath(artifact, root)} did not flip reader {data_reader}"
        )
    fired.append(f"{os.path.basename(artifact)} -> {data_reader} flipped")
    if artifact not in base.unreferenced_artifacts:
        if (r := unchanged(art_fork, "a data-artifact change")) is not None:
            return r

    # (l) NAMED ENV: a variable a module reads by name flips it and not the plain probe
    env_reader = next((m for m in env_readers if not desc[m]["scans"]), env_readers[0])
    var = desc[env_reader]["envs"][0]
    var_fork = make(env={**env, var: "zzz_probe_v3"})
    if var_fork.closure_key(env_reader) == base.closure_key(env_reader):
        return fail(f"a change to ${var} did not flip its reader {env_reader}")
    fired.append(f"${var} -> {env_reader} flipped")
    if (r := unchanged(var_fork, "a named env-var change")) is not None:
        return r

    # (i) TREE SCANNER: a tracked file NO closure names flips a scanner and not the probe
    scanner = "gates.dormancy_gate" if "gates.dormancy_gate" in scanners else scanners[0]
    base.closure_key(scanner)
    surface = base.tree_surface
    if surface > 0:
        markers = {os.path.basename(p) for p in base._data}
        unnamed = None
        for r_, _d, files in os.walk(os.path.join(root, "specs")):
            for f in sorted(files):
                if f.endswith(".md") and f not in markers:
                    unnamed = os.path.join(r_, f)
                    break
            if unnamed:
                break
        if unnamed is None:
            return fail("no specs/ file to use as the tree-surface probe")
        tree_fork = make(file_hash_overrides={unnamed: "b" * 64})
        if tree_fork.closure_key(scanner) == base.closure_key(scanner):
            return fail(
                f"a change to {os.path.relpath(unnamed, root)} did not flip scanner {scanner}"
            )
        fired.append(f"{os.path.relpath(unnamed, root)} -> {scanner} flipped")
        if (r := unchanged(tree_fork, "a tree-surface change")) is not None:
            return r
    else:
        fired.append("red (i) UNCONSTRUCTIBLE here (no git: tree surface unavailable)")

    # (k) GIT-LESS, through the seam: a scanner's PASS recorded under one instance's nonce
    # must be refused by a fresh instance over the SAME store, and the keys must differ
    planter = make(no_git=True)
    planted_key = planter.closure_key(scanner)
    planter._store = {scanner: {"key": planted_key, "verdict": "PASS", "when": "x"}}
    planter._save()
    fresh = make(no_git=True)
    if fresh.closure_key(scanner) == planted_key or fresh.consult(scanner) is not None:
        return fail(
            "without git a scanner's PASS carried across instances -- the tree-surface "
            "nonce degenerated into a constant"
        )
    fired.append("git-less nonce refuses a scanner's PASS across instances")

    # (j) FAIL-CLOSED: an unreferenced tools/models artifact change flips every key
    if base.unreferenced_artifacts:
        orphan = base.unreferenced_artifacts[0]
        if make(file_hash_overrides={orphan: "c" * 64}).closure_key(probe) == k0:
            return fail(
                f"an unreferenced artifact ({os.path.basename(orphan)}) change did not "
                "invalidate the cache"
            )
        fired.append(
            f"{len(base.unreferenced_artifacts)} unreferenced tools/models artifact(s) held GLOBAL "
            f"({os.path.basename(orphan)} flipped everything)"
        )
    else:
        fired.append(
            "red (j) UNCONSTRUCTIBLE on this tree: every tools/models artifact has a named reader"
        )

    # (m) RUST SURFACE: inject the extension dependency, point native_root at a fixture
    # crate (tools/rsfacts derived beside it), flip each Rust input in turn; the plain probe
    # must not move under any of them, and a cargo target/ artifact must not move the reader
    native_root = os.path.join(os.path.dirname(store), "native")
    crate = os.path.join(native_root, "sndl-core")
    os.makedirs(os.path.join(crate, "src"))
    lib = os.path.join(crate, "src", "lib.rs")
    manifest = os.path.join(crate, "Cargo.toml")
    params = os.path.join(crate, "params.json")  # stand-in for non-Rust data under native/
    rsfacts = os.path.join(os.path.dirname(store), *_rf.RSFACTS.parts)
    os.makedirs(os.path.join(rsfacts, "target"))
    rsfacts_src = os.path.join(rsfacts, "fixture.rs")
    rsfacts_target = os.path.join(rsfacts, "target", "built.bin")
    with open(lib, "w", encoding="utf-8") as f:
        f.write("pub fn f() {}\n")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write('[package]\nname = "sndl-core"\n')
    with open(params, "w", encoding="utf-8") as f:
        f.write("{}\n")
    with open(rsfacts_src, "w", encoding="utf-8") as f:
        f.write("pub fn g() {}\n")
    with open(rsfacts_target, "wb") as f:
        f.write(b"\0")
    inj = {_PKG_PREFIX + probe: {_NATIVE}}
    reader = make(graph_overrides=inj, native_root=native_root)
    kn = reader.closure_key(probe)
    if kn == k0 or reader.describe(probe)["native"] in (False, "ABSENT"):
        return fail(f"the injected _native dependency was not keyed: {reader.describe(probe)}")
    # the PRODUCTION path: no graph node (the import graph keys .py files only) -- the
    # probe's own source gains an `_native` import and must be keyed by derivation alone
    probe_file = _module_file(_PKG_PREFIX + probe)
    src = base._read_text(probe_file) + f"\nfrom {_NATIVE} import zzz_probe_v3\n"
    derived = make(file_text_overrides={probe_file: src}, native_root=native_root)
    kd = derived.closure_key(probe)
    kd_flipped = make(
        file_text_overrides={probe_file: src},
        native_root=native_root,
        file_hash_overrides={lib: "d" * 64},
    ).closure_key(probe)
    if not derived.describe(probe)["native"] or kd_flipped == kd:
        return fail("a DERIVED _native reader (no graph node) was not keyed to the Rust surface")
    ext = _native_ext()
    flips = (
        (lib, "d" * 64, ".rs bytes"),
        (manifest, "e" * 64, "Cargo.toml"),
        (params, "9" * 64, "native/ data file"),
        (rsfacts_src, "7" * 64, "tools/rsfacts source"),
        (ext, "f" * 64, "extension bytes"),
    )
    for path, fake, what in flips:
        fork = make(graph_overrides=inj, native_root=native_root, file_hash_overrides={path: fake})
        if fork.closure_key(probe) == kn:
            return fail(f"a {what} change did not flip the _native reader's key (STALE GREEN)")
    built = make(
        graph_overrides=inj, native_root=native_root, file_hash_overrides={rsfacts_target: "8" * 64}
    )
    if built.closure_key(probe) != kn:
        return fail("a tools/rsfacts/target build artifact flipped the _native reader's key")
    all_flips = {path: fake for path, fake, _ in flips}
    control = make(native_root=native_root, file_hash_overrides=all_flips)
    if (
        r := unchanged(control, "a .rs + Cargo.toml + data + rsfacts + extension change")
    ) is not None:
        return r
    real = make(graph_overrides=inj).describe(probe)["native"]
    fired.append(
        f"Rust surface (.rs, Cargo.toml, data, tools/rsfacts, extension) -> {probe} flipped (graph "
        f"node and derived import alike; rsfacts target/ held); real tree "
        f"{'native/ ABSENT (0 .rs files)' if real == 'ABSENT' else 'native/ PRESENT (' + real + ')'}"
        f", ext {os.path.basename(ext)}"
        f"{'' if os.path.isfile(ext) else ' ABSENT'}"
    )

    # (n) UNFLAGGED TREE-WALKERS (carryfix, Opus B3 M1): a listing whose root enters as a
    # parameter (replication_guard's os.walk(pkg_dir), native_ratchet's src.rglob) or through
    # `git ls-files` (rust_facts) is keyed to the tree; the Rust surface flips every rust_facts
    # reader; the plain probe moves under neither. Like (i): constructible only with git (without
    # it every scanner key is a per-instance nonce and the flip would be vacuous)
    for m in _WALKERS + (_RUST_FACTS_READER,):
        if m not in desc:
            return fail(f"{m} left the graph -- red (n) lost its subject; re-pin it")
        if not desc[m]["scans"]:
            return fail(f"{m} lists the tree yet is not a scanner (stale PASS would carry)")
    impact = os.path.join(root, *_IMPACT_FILE)
    lib_rs = os.path.join(root, *_LIB_RS_FILE)
    if surface > 0 and os.path.isfile(lib_rs):
        walk_fork = make(file_hash_overrides={impact: "0" * 64})
        for m in _WALKERS[:2]:
            if walk_fork.closure_key(m) == base.closure_key(m):
                return fail(f"a change to impact_analysis.py did not flip walker {m}")
        if (r := unchanged(walk_fork, "an impact_analysis.py change")) is not None:
            return r
        rs_fork = make(file_hash_overrides={lib_rs: "0" * 64})
        for m in (_WALKERS[2], _RUST_FACTS_READER):
            if rs_fork.closure_key(m) == base.closure_key(m):
                return fail(f"a lib.rs change did not flip rust_facts reader {m}")
        if (r := unchanged(rs_fork, "a lib.rs change")) is not None:
            return r
        fired.append(
            "impact_analysis.py -> replication_guard, native_ratchet flipped; lib.rs -> rust_facts, "
            "native_dormancy flipped (parameter-rooted / ls-files walkers)"
        )
    else:
        fired.append(
            "red (n) UNCONSTRUCTIBLE here ("
            + ("no git: tree surface unavailable" if surface == 0 else "native/ lib.rs absent")
            + ")"
        )

    # (o) LITERAL RELPATH READS (B4 C4): eligible_circuit_census reads src files by literal
    # relpath under a 1-file closure -- a brain.py change must flip it and not the plain probe
    if _RELPATH_READER not in desc:
        return fail(f"{_RELPATH_READER} left the graph -- red (o) lost its subject; re-pin it")
    brain_mod = _PKG_PREFIX + ".".join(_BRAIN_FILE[2:])[: -len(".py")]
    if brain_mod in base._closure(_RELPATH_READER):
        return fail(f"red (o) needs a NON-import read: {brain_mod} joined the reader's closure")
    read_fork = make(file_hash_overrides={os.path.join(root, *_BRAIN_FILE): "0" * 64})
    if read_fork.closure_key(_RELPATH_READER) == base.closure_key(_RELPATH_READER):
        return fail(f"a brain.py change did not flip literal-relpath reader {_RELPATH_READER}")
    if brain_mod not in base._closure(probe):
        if (r := unchanged(read_fork, "a brain.py change")) is not None:
            return r
    fired.append(
        f"engine/brain.py -> {_RELPATH_READER} flipped (a literal-relpath read of "
        f"{len(desc[_RELPATH_READER]['data'])} files under a {desc[_RELPATH_READER]['closure']}"
        "-file closure)"
    )

    surface_note = (
        f"the {surface}-file tracked surface" if surface else "an UNAVAILABLE tracked surface"
    )
    print(
        f"selfcheck_cache self-check OK (envelope v{_CACHE_VERSION} {base.envelope[:12]}...): "
        f"{len(mods)} graph modules classified -- {len(plain)} plain, {len(scanners)} tree "
        f"scanners (re-run on any change to {surface_note}), {len(root_importers)} root-package "
        f"importers, {len(data_readers)} data readers, {len(env_readers)} named-env readers; "
        f"rosters split out: root __init__ {base._roster_names}, runner {base._runner_rosters}. "
        f"Reds fired: {'; '.join(fired)}. Key-invariance held: {', '.join(held)} each left plain "
        f"module {probe} ({len(base._closure(probe))}-file closure) UNCHANGED. "
        f"Blind spots: {banner_blind_spots()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(self_check())
