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
  * ENVIRONMENT: every NORD_* variable is GLOBAL (read under computed names); any other
    variable a closure reads by literal name -- `os.environ.get("NMNIST_SEED")`,
    `os.getenv(...)`, `os.environ[...]` -- is keyed to that reader with its value. The
    runner passes the whole process environment to every child, so an unnamed variable is
    an undeclared input; the review found 48 of them with 5 registered checks reading at
    module load (the N-MNIST seed and sizes among them).
  * TREE SCANNERS: a module whose closure lists a directory ROOTED AT `__file__` --
    os.walk / listdir / scandir / glob / iglob / rglob / iterdir whose path derives from
    `__file__` (bare or as `pkg.__file__`) through the file's own assignments, loop targets
    and return values: os.walk(_PKG_DIR), a glob over src/, brain_atlas's
    listdir(dirname(pkg.__file__)) -- reads files no import closure names. The dormancy /
    method-dormancy / import-manifest / replication / transport gates, the atlas, the
    atlas-node gate, brain_geometry, self_mod_harness, nightly_skills, and by closure
    everything that imports them (the brain itself imports the dormancy gate). Their key
    carries the sha256 of the WHOLE tracked + untracked-unignored tree (`git ls-files -co
    --exclude-standard`), so they re-run on any tree change, and only then. v2 had NO row
    for this surface at all: an edit outside a gate's import closure (the method-dormancy
    gate scans modules it never imports) could carry its stale PASS. It was safe only
    because the global envelope changed on every task anyway.
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
  5. (v3) Gitignored files are outside the tree surface a scanner is keyed on. The ones
     the tree reads by name are keyed as data inputs (models/*, graphify-out/*.json --
     connectome's self-check asserts on graph.json, found by the review); any other
     gitignored path read by literal name elsewhere is unkeyed. None known at landing.
  6. (v3) The scanner derivation roots a listing at `__file__` PER FILE: a root that
     enters from ANOTHER module (an imported `_PKG_DIR`, a helper imported from
     self_mod_harness) is not seen in that file. Counted at landing: 0 module-class
     changes from it -- every such file already reaches a direct scanner through its own
     closure. Each rooting rule is counted too: the return-value rule classifies exactly
     one file (this one, whose package walk is rooted through _repo_root()); a parameter-
     binding rule was built, measured at 0 files and DELETED (P8: a null is removed, not
     parked). The self-check pins the surviving rules on synthetic sources so they cannot
     rot silently. A listing of home() or a dataset dir is deliberately NOT a scan.
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

from snn_data_lake.gates.rust_facts import NATIVE_MODULE as _NATIVE  # stdlib-only reader

_CACHE_VERSION = 3  # v3: envelope precision -- rosters split out, data inputs, named
# environment variables and the tree surface keyed to their DERIVED readers. v2 (review
# BLOCKERs) had grown the governance surface into one global envelope; every input v2
# hashed is still hashed here, now keyed to the modules whose verdict can see it.
_PKG = "snn_data_lake"
_PKG_PREFIX = _PKG + "."
_NATIVE_LEAF = _NATIVE.rpartition(".")[2]  # "_native": the extension has no .py of its own
_NATIVE_NAMES = frozenset(
    {"Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain", "rust-toolchain.toml"}
)
_SCAN_CALLS = frozenset({"walk", "listdir", "scandir", "glob", "iglob", "rglob", "iterdir"})
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
    "runner": ("CORE", "RESEARCH", "TORCH_GATED", "NENGO_GATED", "TRAIN_GATED", "ARTIFACT_GATED"),
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


def _names(node):
    """Names an expression references; `x.__file__` counts as `__file__` (the attribute
    form a package-relative listing uses -- brain_atlas roots on `pkg.__file__`)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and n.attr == "__file__":
            out.add("__file__")
    return out


def _env_name(node):
    """The literal variable name of an os.environ.get / os.getenv / os.environ[...] /
    os.environ.setdefault read, else None."""
    if isinstance(node, ast.Call):
        f = node.func
        attr = f.attr if isinstance(f, ast.Attribute) else None
        via_environ = (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "environ"
        )
        if (attr == "getenv" or (via_environ and attr in ("get", "setdefault"))) and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                return a.value
    elif isinstance(node, ast.Subscript):
        v, s = node.value, node.slice
        if isinstance(v, ast.Attribute) and v.attr == "environ":
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                return s.value
    return None


_UNPARSEABLE = {
    "mentions": frozenset(),
    "roster": frozenset(),
    "envs": frozenset(),
    "scans": True,
    "native": True,
}


def _facts(text, pkg_parts, marker_re):
    """What one source file can READ beyond its imports: the data-input basenames it names
    in non-docstring string constants, the names it imports from the root package, the
    environment variables it reads by literal name, and whether it lists a directory
    ROOTED AT `__file__` (a tree scanner: the listing's path derives from `__file__`
    through this file's own assignments, loop targets and return values -- a home() or
    dataset listing is not a tree input). Unparseable => scanner (fail-closed)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return dict(_UNPARSEABLE)
    rooted = {"__file__"}
    binds = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and n.value is not None:
            binds.append((n.targets if isinstance(n, ast.Assign) else [n.target], n.value))
        elif isinstance(n, (ast.For, ast.comprehension)):
            binds.append(([n.target], n.iter))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for r in ast.walk(n):  # a function RETURNING a rooted value roots its name
                if isinstance(r, ast.Return) and r.value is not None:
                    binds.append(([ast.Name(id=n.name)], r.value))
    changed = True
    while changed:  # fixpoint: a name bound from a rooted expression is rooted
        changed = False
        for targets, value in binds:
            if _names(value) & rooted:
                for t in targets:
                    for nm in _names(t) - rooted:
                        rooted.add(nm)
                        changed = True
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
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in doc_ids:
                mentions.update(marker_re.findall(node.value))
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
        env_name = _env_name(node)
        if env_name is not None:
            envs.add(env_name)
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            is_os_walk = isinstance(f, ast.Attribute) and getattr(f.value, "id", None) == "os"
            if name in _SCAN_CALLS and (name != "walk" or is_os_walk):  # ast.walk is not a listing
                used = set().union(
                    *(_names(a) for a in node.args), *(_names(k.value) for k in node.keywords)
                )
                if isinstance(f, ast.Attribute):
                    used |= _names(f.value)
                if used & rooted:
                    scans = True
    return {
        "mentions": frozenset(mentions),
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
    ):
        # `file_hash_overrides` ({path: fake_hash}), `file_text_overrides` ({path: source}),
        # `env`, `store_path`, `no_git`, `graph_overrides` ({dotted: {dotted dep}} merged
        # into the import graph) and `native_root` (a fixture native/ tree) are the TESTING
        # SEAMS for the self-check's reds: they prove an input change flips exactly the keys
        # it should WITHOUT mutating the tree, the process env or the REAL store (a record()
        # from a probe is a PASS nobody ran -- found by doing exactly that once), and they
        # reach the git-less branch on a host that has git. Never set otherwise.
        self._store_path = store_path or _cache_path()
        self._no_git = bool(no_git)
        self._graph_overrides = {k: set(v) for k, v in (graph_overrides or {}).items()}
        self._native_root = native_root or os.path.join(_repo_root(), "native")
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
            raw = None
            if not self._no_git:
                try:
                    raw = subprocess.run(
                        ["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z"],
                        capture_output=True,
                        check=True,
                    ).stdout
                except (OSError, subprocess.CalledProcessError):
                    raw = None
            if raw is None:
                self._tree = "TREE-SURFACE-UNAVAILABLE:" + os.urandom(8).hex()
                return self._tree
            rows = []
            for rel in sorted(p for p in raw.decode("utf-8", "surrogateescape").split("\0") if p):
                p = os.path.join(root, rel)
                if os.path.isfile(p):
                    rows.append(rel + ":" + self._hash_file(p))
            self.tree_surface = len(rows)
            self._tree = _sha256_text("\n".join(rows))
        return self._tree

    def _native_hash(self):
        """ "native:<sha256>" over (path, sha256) of every Rust input under native/ (target/
        skipped) and every other file git lists there -- tracked or untracked-unignored, the
        `_tree_hash` view (runtime data the extension reads; Codex B1 r02b-cache 1 -- without
        git's list EVERY file under native/ keys, fail closed), plus the manifest / toolchain
        names beside it (the repo root), or
        "native:ABSENT" when nothing exists; memoised with the count of .rs files under native/
        (disk view, not git's: the key must move on bytes git cannot see)."""
        if self._native is None:
            listed = None  # git's view of native/; None = unavailable, every file keys
            if not self._no_git:
                try:
                    raw = subprocess.run(
                        [
                            "git",
                            "-C",
                            self._native_root,
                            "ls-files",
                            "-co",
                            "--exclude-standard",
                            "-z",
                        ],
                        capture_output=True,
                        check=True,
                    ).stdout
                    # case-folded: an ignorecase checkout keeps git's spelling of a case-only rename
                    listed = {
                        p.lower() for p in raw.decode("utf-8", "surrogateescape").split("\0") if p
                    }
                except (OSError, subprocess.CalledProcessError):
                    pass
            rows, n_rs = [], 0
            for r, dirs, files in os.walk(self._native_root):
                dirs[:] = sorted(d for d in dirs if d != "target")
                for f in sorted(files):
                    p = os.path.join(r, f)
                    rel = os.path.relpath(p, self._native_root).replace(os.sep, "/")
                    if (
                        f.endswith(".rs")
                        or f in _NATIVE_NAMES
                        or listed is None
                        or rel.lower() in listed
                    ):
                        rows.append("native/" + rel + ":" + self._hash_file(p))
                        n_rs += f.endswith(".rs")
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
        for d in sorted(self._closure(mod)):
            if d == _NATIVE:  # no .py: its bytes are the native + ext rows below
                native = True
                continue
            path = _module_file(d)
            rows.append(f"{d}={self._hash_file(path)}")
            facts = self._facts_for(path)
            mentions |= facts["mentions"]
            roster |= facts["roster"]
            envs |= facts["envs"]
            scans = scans or facts["scans"]
            native = native or facts["native"]
        data = [p for p in self._data if os.path.basename(p) in mentions]
        if self._runner in self._data and "run_selfchecks" in mentions:
            data.append(self._runner)
        data_rows = [os.path.relpath(p, root) + ":" + self._hash_file(p) for p in sorted(set(data))]
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
            "ENTIRE static import closure + the data files / named env vars / root-package "
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
        "inputs read by name outside models/ and graphify-out/; a listing root imported "
        "from another module; gitignored non-Rust files under native/ (when git lists it); "
        "directory symlinks under native/; "
        "Rust build inputs outside native/ and the manifest names "
        "(.cargo/config.toml, RUSTFLAGS, the toolchain binary) (carries = local batteries "
        "only; any carried-run FAIL reruns FULL first)"
    )


def _insert_after(text, anchor, snippet):
    i = text.index(anchor) + len(anchor)
    return text[:i] + snippet + text[i:]


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
    "home_listing_not_scan": (
        "import os\nfrom snn_data_lake.substrate.home import home\nnames = os.listdir(home())\n",
        "scans",
        False,
    ),
    "ast_walk_not_scan": (
        "import ast\ntree = ast.parse('x')\nfor n in ast.walk(tree):\n    pass\n",
        "scans",
        False,
    ),
    "env_reads": (
        'import os\nSEED = int(os.environ.get("ZZZ_SEED", 0))\nX = os.getenv("ZZZ_X")\n'
        'Y = os.environ["ZZZ_Y"]\nos.environ.setdefault("ZZZ_Z", "1")\n',
        "envs",
        frozenset({"ZZZ_SEED", "ZZZ_X", "ZZZ_Y", "ZZZ_Z"}),
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
        the real tree's status (ABSENT today) is printed distinctly."""
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
    null_re = re.compile(r"(?!x)x")
    for name, (src, field, want) in _SYNTHETIC.items():
        got = _facts(src, 1, null_re)[field]
        if got != want:
            return fail(f"derivation red {name}: {field} = {got!r}, expected {want!r}")
    fired.append(
        "derivation pins (attribute __file__, return-rooted, home()/ast.walk excluded, env names)"
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

    # The universe: every in-package module the ONE import graph knows (no runner import --
    # run_selfchecks.py probes torch and asserts tiers at import; the graph is the census).
    mods = sorted(
        d[len(_PKG_PREFIX) :]
        for d in base._import_graph()
        if d.startswith(_PKG_PREFIX) and os.path.isfile(_module_file(d))
    )
    desc = {m: base.describe(m) for m in mods}

    plain = [
        m
        for m in mods
        if not any(desc[m][k] for k in ("scans", "roster", "data", "envs", "native"))
    ]
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
    first = "engine.signal_ring" if "engine.signal_ring" in plain else plain[0]
    for probe in [first] + plain:
        if len(base._closure(probe)) > 1:
            break
    else:
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
    # crate, flip each Rust input in turn; the plain probe must not move under any of them
    native_root = os.path.join(os.path.dirname(store), "native")
    crate = os.path.join(native_root, "sndl-core")
    os.makedirs(os.path.join(crate, "src"))
    lib = os.path.join(crate, "src", "lib.rs")
    manifest = os.path.join(crate, "Cargo.toml")
    params = os.path.join(crate, "params.json")  # stand-in for non-Rust data under native/
    with open(lib, "w", encoding="utf-8") as f:
        f.write("pub fn f() {}\n")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write('[package]\nname = "sndl-core"\n')
    with open(params, "w", encoding="utf-8") as f:
        f.write("{}\n")
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
        (ext, "f" * 64, "extension bytes"),
    )
    for path, fake, what in flips:
        fork = make(graph_overrides=inj, native_root=native_root, file_hash_overrides={path: fake})
        if fork.closure_key(probe) == kn:
            return fail(f"a {what} change did not flip the _native reader's key (STALE GREEN)")
    all_flips = {path: fake for path, fake, _ in flips}
    control = make(native_root=native_root, file_hash_overrides=all_flips)
    if (r := unchanged(control, "a .rs + Cargo.toml + data + extension change")) is not None:
        return r
    real = make(graph_overrides=inj).describe(probe)["native"]
    fired.append(
        f"Rust surface (.rs, Cargo.toml, data, extension) -> {probe} flipped (graph node and derived "
        f"import alike); real tree "
        f"{'native/ ABSENT (0 .rs files)' if real == 'ABSENT' else 'native/ PRESENT (' + real + ')'}"
        f", ext {os.path.basename(ext)}"
        f"{'' if os.path.isfile(ext) else ' ABSENT'}"
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
