"""move_gate -- G8's G-R8 instrument: the gate census, its diff against a pre-registered delta, and
the reds that prove it can see (specs/022 evidence g8-reorganization.md 2.6; landed at S1a).

WHY. G8 moves modules into engine-function packages and must change no gate's outcome beyond what
each stage pre-registers: a pure move has an EMPTY delta, every other stage declares its delta
before its first gate run, and this instrument decides whether the tree kept to it.

THE CENSUS (`--census OUT [--root TREE]`). Every C1 gate runs as a subprocess against TREE
(`PYTHONPATH=<TREE>/src`, cwd TREE, one fresh temporary NORD_HOME per run, sequential) and leaves:
  * gates  -- each gate's exit code and printed lines, durations and line numbers stripped
              (neither is an outcome, and a move re-wraps the import lines a report cites);
  * probes -- the outcome SETS a gate computes but does not print, read through the gates' OWN
              functions (never re-derived): the dormancy reachable set and EXEMPT verdicts, the
              import-manifest targets and broken list, replication sites, the region-charge and
              atlas-node censuses, the policy audit's per-command verdicts (its `run_audit`) and
              R-b0's source parity (C4, `check_r_b0`);
  * names  -- the NAME-valued registries of C1, read statically from source, never imported;
  * static -- the live and LEGACY_* tables, the rule census's dispositions and C6's invariants.
Every gate is located by its BASENAME in the tree's own live tables, so a gate that moved is still
the gate that runs; this module imports nothing of the package it measures (stdlib only), so it
censuses any tree, one from before it existed included (the pre-S1a census, 2.6).

THE DIFF (`--diff PRE POST --declared DELTA`). Each side's module references are mapped through
LEGACY_* (its live package -> the frozen pre-G8 label), so a pure move reads identical, and the
remaining difference must EQUAL the declared delta change for change: an undeclared change and a
declared change that did not happen are both red (rc 1). C6's invariants are never declarable:
POST's LEGACY_* must equal PRE's (PRE's live tables when PRE predates them), no module's depth
changes, and every static violation list stays empty.

THE REDS (`--self-check --mover MOVER`), built in a temporary copy of the tree, never the checkout:
(1) C1's planted EXEMPT-row deletion reds the diff, beside the control that an unplanted re-census
diffs EMPTY; (2) F5's join red: MOVER (2.5's out-of-tree operator artifact) moves one module in the
copy, `insula/interoceptive_integration` to its frozen 2.12 destination, and both region censuses
stay identical with the label/path joins in place, while the same move with the path taken from
LEGACY_* (the draft's form: module_path() replaced by the label) loses reachability. Without MOVER
the self-check REFUSES (exit 2) rather than skip. The BARE invocation (run_selfchecks, pytest)
builds the diff engine's own reds over constructed censuses: no subprocess and no copy.

`--export REV|WORKTREE DIR` makes a census-ready tree: `git archive REV` (WORKTREE: HEAD's archive
with the working tree's changes laid over it) with each file's line endings as this checkout holds
them (the envelope hashes are of those bytes), then an index-only `git init` so the git-listing
gates see it. With no commit the policy manifest reports its freshness conjunct NOT checked
(disclosed), the same on both sides of a diff. One module by design: 2.4 registers exactly one G8
instrument, and a second module would widen S1a's declared delta; hence its length.

  python -m snn_data_lake.gates.move_gate                    # bare: the diff engine's reds
  python -m snn_data_lake.gates.move_gate --census OUT [--root TREE]
  python -m snn_data_lake.gates.move_gate --diff PRE POST --declared DELTA [--observed OUT]
  python -m snn_data_lake.gates.move_gate --self-check --mover MOVER [--root TREE]
  python -m snn_data_lake.gates.move_gate --export REV|WORKTREE DIR [--root REPO]
"""

# distinct-from: substrate/method_dormancy_gate.py -- its _imports() lists one file's in-package
#   imports for its own scan inside the RUNNING package; this module never imports the package it
#   measures (the census tree is an export, not the running package) and reads trees by AST.
#   Also read: substrate/dormancy_gate.py (reachable), substrate/atlas_node_gate.py
#   (node_references) -- both REUSED here through probes, never re-derived; __init__.py
#   (_AliasLoader); reasoning/self_improve.py (_self_pkg_relpath) -- neither is a gate census.

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from collections import Counter
from pathlib import Path

PKG = "snn_data_lake"
REPO = Path(__file__).resolve().parents[3]
SCHEMA = "g8.move_gate.census.v1"
MARKER = ".g8_temp_copy"  # a self-check copy; the mover refuses any tree without it
TABLES = ("REGIONS", "TORCH_TIER", "TRAIN_TIER")
UNGOVERNED = ("reasoning", "training", "lakehouse")
PROBE_TAG = "G8PROBE "
DOC = "specs/022-unified-brain-arc/evidence/g8-reorganization.md"
F5_MODULE = "insula/interoceptive_integration.py"  # F5's planted move (row 265: faculties)

#: C1's gate runs: (key, gate basename or tools/ script, arguments). A basename is resolved in the
#: TREE's live tables at census time (the join), a top-level module at the package root.
GATE_RUNS = (
    ("dormancy_gate", "dormancy_gate", ("--assert",)),
    ("method_dormancy_gate", "method_dormancy_gate", ("--assert",)),
    ("region_charge_gate", "region_charge_gate", ("--assert",)),
    ("growth_coverage_gate", "growth_coverage_gate", ("--assert",)),
    ("manifest_gate", "manifest_gate", ("--assert",)),
    ("determinism_gate", "determinism_gate", ("--assert",)),
    ("replay_gate", "replay_gate", ("--assert",)),
    ("envelope_hash_gate", "envelope_hash_gate", ("--assert",)),
    ("policy_manifest", "policy_manifest", ("--assert",)),
    ("gradient_claim_gate", "gradient_claim_gate", ("--assert",)),
    ("eprop_transport_gate", "eprop_transport_gate", ("--assert",)),
    ("import_manifest", "import_manifest", ("--assert",)),
    ("replication_guard", "replication_guard", ("--assert",)),
    ("capability_bar", "capability_bar", ("--assert",)),
    ("immortal_retention", "immortal_retention", ("--assert",)),
    ("conclusion_protocol", "conclusion_protocol", ("--assert",)),
    ("atlas_node_gate", "atlas_node_gate", ("--assert",)),
    ("doc_budget_gate", "doc_budget_gate", ("--assert",)),
    ("rule_composer", "rule_composer", ()),
    ("brain_atlas", "brain_atlas", ()),
    ("p4c_ratchet", "tools/p4c_ratchet.py", ("--assert",)),
    ("p6_gate", "tools/p6_gate.py", ("--assert",)),
    ("citation_guard", "tools/citation_guard.py", ()),
    ("doc_command_guard", "tools/doc_command_guard.py", ("--assert",)),
)
#: C1 compares only these lines of a gate whose printout is wider than its census.
KEEP = {"citation_guard": ("STRICT", "BROAD")}

#: Probes: (gate basename, code). `__MOD__` becomes the basename's dotted path in the tree.
_P = 'import importlib, json\ng = importlib.import_module("__MOD__")\n'
PROBES = {
    "dormancy": (
        "dormancy_gate",
        _P + "c = g.coverage()\n"
        "o = {k: c[k] for k in ('total', 'reachable', 'exempt', 'dormant', 'stale_exempt',"
        " 'conditional_imports')}\n"
        "o['phantom_exempt'] = c.get('phantom_exempt')\n"
        "o['open_dispositions'] = [list(x) for x in c['open_dispositions']]\n"
        "o['reachable_set'] = sorted(g.reachable())\n"
        "o['exempt_verdicts'] = sorted(f'{k}={v[0]}' for k, v in g.EXEMPT.items())\n",
    ),
    "import_manifest": (
        "import_manifest",
        _P + "m = g.manifest()\no = {'targets': sorted(m), 'broken': sorted(g.broken(m))}\n",
    ),
    "replication": (
        "replication_guard",
        _P + "a = g.audit()\n"
        "o = {'sites': {k: sorted(v) for k, v in a['sites'].items()},"
        " 'replicas': sorted(map(list, a['replicas'])), 'missing': sorted(map(list, a['missing']))}\n",
    ),
    "region_charge": ("region_charge_gate", _P + "o = g.census()\n"),
    "atlas_node": (
        "atlas_node_gate",
        _P + "o = {'census': g.census(), 'references': g.node_references()}\n",
    ),
    "policy_audit": (
        "policy_manifest",
        _P + "res, cmds = g.run_audit(g.cross_check(g.derive()))\n"
        "o = {'results': sorted(f'{r.policy_id}={r.verdict}' for r in res),"
        " 'commands': sorted(' '.join(c['argv']) + ' => ' + c['verdict'] for c in cmds)}\n",
    ),
    "r_b0": ("recurrent_pool", _P + "o = g.check_r_b0()\n"),
}
_EMIT = 'print("' + PROBE_TAG + '" + json.dumps(o, sort_keys=True, default=repr))\n'
#: F5's draft form: module_path() replaced by the frozen label (the path taken from LEGACY_*).
DRAFT_FORM = (
    "import snn_data_lake as _S\n"
    "_lab = {m: k for t in (_S.LEGACY_REGIONS, _S.LEGACY_TORCH_TIER, _S.LEGACY_TRAIN_TIER)"
    " for k, ms in t.items() for m in ms}\n"
    '_S.module_path = lambda m: f"snn_data_lake.{_lab[m]}.{m}"\n'
)

_TIME = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|secs|seconds)\b")
_LINENO = re.compile(r"(?<=[\w@\])]):\d+\b")
_MODREF = re.compile(
    r"(?<![\w.])((?:snn_data_lake[./])?)([a-z_][a-z0-9_]*)([./])([a-z_][a-z0-9_]*)"
)
_DOTTED = re.compile(r"snn_data_lake(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_RUNNER_TIERS = ("CORE", "RESEARCH", "TORCH_GATED", "NENGO_GATED", "TRAIN_GATED", "ARTIFACT_GATED")


# ---- the tree's tables and files, read by AST ------------------------------------------------
def _assign(path: Path, name: str):
    """The value node of a module-level `name = ...` in `path`, or None."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        tgt = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(tgt, ast.Name) and tgt.id == name:
            return node.value
    return None


def tables(root: Path) -> dict:
    """{'live': {TABLE: {pkg: [mod]}}, 'legacy': {TABLE: ...} or {}} from the tree's root __init__."""
    init = Path(root) / "src" / PKG / "__init__.py"
    out = {"live": {}, "legacy": {}}
    for t in TABLES:
        for side, name in (("live", t), ("legacy", "LEGACY_" + t)):
            node = _assign(init, name)
            if node is not None:
                out[side][t] = ast.literal_eval(node)
    return out


def _pkg_of(live: dict, base: str) -> str | None:
    for t in TABLES:
        for pkg, mods in live.get(t, {}).items():
            if base in mods:
                return pkg
    return None


def _dotted(root: Path, tabs: dict, base: str) -> str:
    """The tree's current dotted path for a basename (live tables, else the package root)."""
    pkg = _pkg_of(tabs["live"], base)
    if pkg is not None:
        return f"{PKG}.{pkg}.{base}"
    if (Path(root) / "src" / PKG / f"{base}.py").is_file():
        return f"{PKG}.{base}"
    raise SystemExit(f"move_gate: {base} is in no live table and not at the package root of {root}")


def _file_of(root: Path, tabs: dict, base: str) -> Path:
    return Path(root, "src", *_dotted(root, tabs, base).split(".")).with_suffix(".py")


# ---- running a gate ----------------------------------------------------------------------------
def _run(argv: list, root: Path, home: Path) -> tuple:
    home.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=str(Path(root) / "src"),
        NORD_HOME=str(home),
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        p = subprocess.run(
            argv,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "timeout", "", "TIMEOUT after 1800 s"


def _norm(line: str, subs: list) -> str:
    for old, new in subs:
        line = line.replace(old, new)
    return _LINENO.sub("", _TIME.sub("<t>", line)).rstrip()


def _subs(root: Path, home: Path) -> list:
    out = []
    for p, tag in ((Path(root).resolve(), "<root>"), (Path(home).resolve(), "<home>")):
        for s in {str(p), str(p).replace("\\", "/"), str(p).replace("\\", "\\\\")}:
            out.append((s, tag))
    out.append((sys.executable, "<python>"))
    return sorted(out, key=lambda x: -len(x[0]))


def _live_home_stamp() -> dict:
    """The default home's files (size, mtime): a census must leave the live lake untouched."""
    d = Path(os.path.expanduser("~")) / ".snn_data_lake"
    if not d.is_dir():
        return {}
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir() if p.is_file()}


def _git(args: list, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.safecrlf=false", *args],
        cwd=cwd,
        capture_output=True,
        timeout=600,
    )


def _git_status(root: Path) -> bytes | None:
    try:
        p = _git(["status", "--porcelain", "--untracked-files=all"], root)
    except OSError:
        return None
    return p.stdout if p.returncode == 0 else None


def census(root: Path, only=None, draft: bool = False, log=print) -> dict:
    """The census of one tree (see the module docstring). `only` limits the gate runs and probes
    (the self-check's partial censuses); names and static are always read (cheap, no subprocess)."""
    root = Path(root).resolve()
    tabs = tables(root)
    stamp, status = _live_home_stamp(), _git_status(root)
    homes = Path(tempfile.mkdtemp(prefix="g8census_"))
    out = {"schema": SCHEMA, "root": str(root), "gates": {}, "probes": {}}
    try:
        for key, target, args in GATE_RUNS:
            if only is not None and key not in only:
                continue
            if target.startswith("tools/"):
                argv = [sys.executable, str(root / target), *args]
            else:
                argv = [sys.executable, "-m", _dotted(root, tabs, target), *args]
            t0 = time.perf_counter()
            rc, so, se = _run(argv, root, homes / key)
            subs = _subs(root, homes / key)
            lines = [_norm(x, subs) for x in so.splitlines()]
            if key in KEEP:
                lines = [x for x in lines if x.strip().startswith(KEEP[key])]
            out["gates"][key] = {
                "rc": rc,
                "lines": [x for x in lines if x.strip()],
                "stderr_tail": [_norm(x, subs) for x in se.splitlines()[-5:]],
            }
            log(f"  census {key:<22} rc={rc} [{time.perf_counter() - t0:.1f}s]")
        for key, (target, code) in PROBES.items():
            if only is not None and key not in only:
                continue
            src = (DRAFT_FORM if draft else "") + code.replace(
                "__MOD__", _dotted(root, tabs, target)
            )
            home = homes / ("probe_" + key)
            rc, so, se = _run([sys.executable, "-c", src + _EMIT], root, home)
            tagged = [x for x in so.splitlines() if x.startswith(PROBE_TAG)]
            if rc != 0 or len(tagged) != 1:
                raise SystemExit(f"move_gate: probe {key} failed rc={rc}: {(so + se)[-1500:]}")
            out["probes"][key] = json.loads(_norm(tagged[0][len(PROBE_TAG) :], _subs(root, home)))
            log(f"  probe  {key:<22} ok")
    finally:
        shutil.rmtree(homes, ignore_errors=True)
    out["names"] = names(root, tabs)
    out["static"] = static(root, tabs)
    if _live_home_stamp() != stamp:
        raise SystemExit(
            "move_gate: the DEFAULT home changed during the census -- a gate wrote the live lake"
        )
    if status is not None and _git_status(root) != status:
        raise SystemExit(f"move_gate: the census changed {root}'s working tree")
    return out


# ---- NAME-valued registries (C1, read RAW from source) ----------------------------------------
def _strs(node) -> list:
    """Every name an expression spells: string constants, f-string constant joins (the
    `f"{_PKG}.cortex.brain"` form, read as `cortex.brain`), and any other element as source text."""
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value for v in node.values if isinstance(v, ast.Constant)).lstrip(".")]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [s for e in node.elts for s in _strs(e)]
    if isinstance(node, ast.Dict):
        return [s for k in node.keys for s in _strs(k)]
    return [ast.unparse(node)]


#: registry -> (gate basename or repo path, assignment name, how to read it)
_REG = {
    "dormancy.EXEMPT": ("dormancy_gate", "EXEMPT", "keys"),
    "dormancy._LIVE_ROOTS": ("dormancy_gate", "_LIVE_ROOTS", "strs"),
    "dormancy._ENTRYPOINTS": ("dormancy_gate", "_ENTRYPOINTS", "strs"),
    "method_dormancy.EXEMPT_METHOD": ("method_dormancy_gate", "EXEMPT_METHOD", "keys"),
    "import_manifest.GOVERNED": ("import_manifest", "GOVERNED", "strs"),
    "replication.CANON": ("replication_guard", "CANON", "items"),
    "manifest_gate.REGISTRANTS": ("manifest_gate", "REGISTRANTS", "first"),
    "brian2.MIGRATED_CALL_SITES": ("brian2_backend", "MIGRATED_CALL_SITES", "strs"),
    "brian2.CUSTOM_KERNELS": ("brian2_backend", "CUSTOM_KERNELS", "strs"),
    "brian2.MODULE_NAME": ("brian2_backend", "MODULE_NAME", "strs"),
    "growth_coverage._PLACEMENT_MODULES": ("growth_coverage_gate", "_PLACEMENT_MODULES", "strs"),
    "safety_envelope.ENVELOPE": ("safety_envelope", "ENVELOPE", "strs"),
    "self_mod_harness._EXTRA_PROTECTED": ("self_mod_harness", "_EXTRA_PROTECTED", "strs"),
    "godel_machine._SAFETY_KERNEL": (
        "src/snn_data_lake/reasoning/godel_machine.py",
        "_SAFETY_KERNEL",
        "strs",
    ),
}


def _read_reg(root: Path, tabs: dict, where: str, name: str, how: str) -> list:
    path = Path(root) / where if "/" in where else _file_of(root, tabs, where)
    node = _assign(path, name)
    if node is None:
        return [f"<missing {name} in {where}>"]
    if how == "keys" and isinstance(node, ast.Dict):
        return [s for k in node.keys for s in _strs(k)]
    if how == "items" and isinstance(node, ast.Dict):
        return [f"{ast.literal_eval(k)}={_strs(v)[0]}" for k, v in zip(node.keys, node.values)]
    if how == "first" and isinstance(node, (ast.List, ast.Tuple)):
        return [_strs(e.elts[0])[0] for e in node.elts if isinstance(e, (ast.Tuple, ast.List))]
    return _strs(node)


def _parse(path: Path):
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def _imports_of(path: Path) -> list:
    out = []
    for n in ast.walk(_parse(path) or ast.Module(body=[], type_ignores=[])):
        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith(PKG):
            out += [f"{n.module}:{a.name}" for a in n.names]
        elif isinstance(n, ast.Import):
            out += [a.name for a in n.names if a.name.startswith(PKG)]
    return out


def _code_strings(path: Path) -> list:
    tree = _parse(path) or ast.Module(body=[], type_ignores=[])
    return [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _json_field(path: Path, field: str) -> list:
    try:
        return sorted(json.loads(Path(path).read_text(encoding="utf-8"))[field])
    except (OSError, KeyError, ValueError) as exc:
        return [f"<unreadable: {type(exc).__name__}>"]


def names(root: Path, tabs: dict) -> dict:
    """C1's NAME-valued registries, each a sorted list of the strings it holds, read RAW."""
    root = Path(root)
    out = {k: sorted(_read_reg(root, tabs, *v)) for k, v in _REG.items()}
    runner = root / "run_selfchecks.py"
    out["run_selfchecks"] = sorted(s for t in _RUNNER_TIERS for s in _strs(_assign(runner, t)))
    known = {
        (pkg, m)
        for side in ("live", "legacy")
        for t in tabs[side].values()
        for pkg, ms in t.items()
        for m in ms
    }
    out["selfcheck_cache.subjects"] = sorted(
        {
            s
            for s in _code_strings(_file_of(root, tabs, "selfcheck_cache"))
            if re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", s) and tuple(s.split(".")) in known
        }
    )
    pin = root / "tests" / "pin" / "capture.py"
    out["pin.capture"] = sorted({s for s in _code_strings(pin) if _DOTTED.fullmatch(s)})
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        out["pyproject.scripts"] = sorted(
            f"{k}={v}" for k, v in pyproject["project"]["scripts"].items()
        )
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        out["pyproject.scripts"] = [f"<unreadable: {type(exc).__name__}>"]
    ci = root / ".github" / "workflows" / "ci.yml"
    ci_text = ci.read_text(encoding="utf-8") if ci.is_file() else ""
    out["ci.m_targets"] = sorted(re.findall(r"-m\s+(snn_data_lake(?:\.\w+)+)", ci_text))
    spec = root / "integrations" / "packaging" / "snn-ui.spec"
    out["snn_ui_spec"] = (
        sorted(_DOTTED.findall(spec.read_text(encoding="utf-8"))) if spec.is_file() else []
    )
    out["snn_ui_main.imports"] = sorted(
        _imports_of(root / "integrations" / "packaging" / "snn_ui_main.py")
    )
    out["learn_daemon_shim.imports"] = sorted(_imports_of(root / "learn_daemon.py"))
    out["envelope_hashes"] = _json_field(root / "tools" / "envelope_hashes.json", "files")
    out["rule_census.keys"] = _json_field(root / "tools" / "rule_census.json", "rules")
    for t in TABLES:
        live = tabs["live"].get(t, {})
        out[f"{t}.keys"] = sorted(live)
        out[t] = sorted(f"{pkg}.{m}" for pkg, ms in live.items() for m in ms)
    return out


# ---- C6's static invariants --------------------------------------------------------------------
def static(root: Path, tabs: dict) -> dict:
    src = Path(root) / "src" / PKG
    files = sorted(
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if p.name != "__init__.py" and "__pycache__" not in p.parts
    )
    bases = Counter(f.rsplit("/", 1)[-1][:-3] for f in files)
    live, legacy = tabs["live"], tabs["legacy"] or tabs["live"]
    governed = {m: pkg for t in live.values() for pkg, ms in t.items() for m in ms}
    label = {m: pkg for t in legacy.values() for pkg, ms in t.items() for m in ms}
    live_keys = {k for t in live.values() for k in t}
    moved_away = [m for m, p in governed.items() if m in label and label[m] != p]
    inv = {
        "basename_collisions": sorted(b for b, c in bases.items() if c > 1),
        "governed_without_file": sorted(
            f"{p}.{m}" for m, p in governed.items() if not (src / p / f"{m}.py").is_file()
        ),
        "governed_in_ungoverned": sorted(
            f"{p}.{m}" for m, p in governed.items() if p in UNGOVERNED
        ),
        "legacy_py_paths": sorted(
            f"{label[m]}/{m}.py" for m in moved_away if (src / label[m] / f"{m}.py").exists()
        ),
        "legacy_pkg_dirs": sorted(
            {k for t in legacy.values() for k in t if k not in live_keys and (src / k).exists()}
        ),
        "tables_not_literal": sorted(t for t in TABLES if t not in live),
    }
    try:
        rows = json.loads((Path(root) / "tools" / "rule_census.json").read_text(encoding="utf-8"))
        rules = {
            k: (v.get("disposition") if isinstance(v, dict) else v)
            for k, v in rows["rules"].items()
        }
    except (OSError, KeyError, ValueError) as exc:
        rules = {"<unreadable>": type(exc).__name__}
    return {
        "live_tables": live,
        "legacy_tables": tabs["legacy"],
        "module_depth": {f.rsplit("/", 1)[-1][:-3]: f.count("/") for f in files},
        "rule_dispositions": rules,
        "invariants": inv,
    }


# ---- the diff ----------------------------------------------------------------------------------
def pairmap(c: dict) -> dict:
    """(live package, module) -> frozen label, for every module whose package differs from it."""
    st = c["static"]
    legacy = st["legacy_tables"] or st["live_tables"]
    label = {m: k for t in legacy.values() for k, ms in t.items() for m in ms}
    return {
        (k, m): label[m]
        for t in st["live_tables"].values()
        for k, ms in t.items()
        for m in ms
        if m in label and label[m] != k
    }


def map_text(s: str, pm: dict) -> str:
    """Every exact (package, module) reference in `s` rewritten to its frozen label."""
    if not pm or not isinstance(s, str):
        return s

    def sub(mo):
        lab = pm.get((mo.group(2), mo.group(4)))
        return f"{mo.group(1)}{lab}{mo.group(3)}{mo.group(4)}" if lab else mo.group(0)

    return _MODREF.sub(sub, s)


def _flat(obj, prefix: str = "") -> dict:
    """Nested dicts -> {'a.b': leaf}; a leaf is a list (compared as a multiset) or a scalar."""
    if isinstance(obj, dict) and obj:
        out = {}
        for k in sorted(obj):
            out.update(_flat(obj[k], f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: obj}


def _leaf(v, pm):
    if isinstance(v, list):
        return Counter(
            map_text(x if isinstance(x, str) else json.dumps(x, sort_keys=True), pm) for x in v
        )
    return map_text(v, pm) if isinstance(v, str) else v


def _section(a: dict, b: dict, pa: dict, pb: dict) -> dict:
    """Changes between two flat-mappable sections, keyed by flattened path."""
    fa, fb = _flat(a or {}), _flat(b or {})
    fa = {map_text(k, pa): v for k, v in fa.items()}
    fb = {map_text(k, pb): v for k, v in fb.items()}
    out = {}
    for key in sorted(set(fa) | set(fb)):
        va, vb = _leaf(fa.get(key, []), pa), _leaf(fb.get(key, []), pb)
        if isinstance(va, Counter) and isinstance(vb, Counter):
            minus, plus = sorted((va - vb).elements()), sorted((vb - va).elements())
            ch = {k: v for k, v in (("-", minus), ("+", plus)) if v}
        else:
            ch = {} if va == vb else {"from": fa.get(key), "to": fb.get(key)}
        if ch:
            out[key] = ch
    return out


def observed(pre: dict, post: dict) -> dict:
    """The difference between two censuses, module references mapped through LEGACY_* (S1a: both
    classes, the draft form of C1; F3(b) makes the NAME class raw-against-expected at S1b)."""
    pa, pb = pairmap(pre), pairmap(post)
    gates = {}
    for key in sorted(set(pre["gates"]) | set(post["gates"])):
        a = pre["gates"].get(key, {"rc": None, "lines": []})
        b = post["gates"].get(key, {"rc": None, "lines": []})
        la = Counter(map_text(x, pa) for x in a["lines"])
        lb = Counter(map_text(x, pb) for x in b["lines"])
        ch = {"rc": {"from": a["rc"], "to": b["rc"]}} if a["rc"] != b["rc"] else {}
        ch.update(
            {
                k: v
                for k, v in (
                    ("-", sorted((la - lb).elements())),
                    ("+", sorted((lb - la).elements())),
                )
                if v
            }
        )
        if ch:
            gates[key] = ch
    probes = {}
    for key in sorted(set(pre["probes"]) | set(post["probes"])):
        ch = _section(pre["probes"].get(key), post["probes"].get(key), pa, pb)
        if ch:
            probes[key] = ch
    depth_a, depth_b = pre["static"]["module_depth"], post["static"]["module_depth"]
    static = _section(
        {
            "module_depth": sorted(f"{m}={d}" for m, d in depth_a.items()),
            "rule_dispositions": sorted(
                f"{k}={v}" for k, v in pre["static"]["rule_dispositions"].items()
            ),
        },
        {
            "module_depth": sorted(f"{m}={d}" for m, d in depth_b.items()),
            "rule_dispositions": sorted(
                f"{k}={v}" for k, v in post["static"]["rule_dispositions"].items()
            ),
        },
        pa,
        pb,
    )
    out = {
        "gates": gates,
        "probes": probes,
        "names": _section(pre["names"], post["names"], pa, pb),
        "static": static,
    }
    return {k: v for k, v in out.items() if v}


def invariant_problems(pre: dict, post: dict) -> list:
    """C6, never declarable: LEGACY_* frozen, depth unchanged, every violation list empty."""
    out = []
    frozen = pre["static"]["legacy_tables"] or pre["static"]["live_tables"]
    if post["static"]["legacy_tables"] and post["static"]["legacy_tables"] != frozen:
        out.append("C6: POST's LEGACY_* tables are not AST-equal to PRE's frozen tables")
    if not post["static"]["legacy_tables"] and pre["static"]["legacy_tables"]:
        out.append("C6: POST lost the LEGACY_* tables PRE carries")
    da, db = pre["static"]["module_depth"], post["static"]["module_depth"]
    moved = sorted(m for m in set(da) & set(db) if da[m] != db[m])
    if moved:
        out.append(f"C6: module depth changed for {moved}")
    for side, c in (("PRE", pre), ("POST", post)):
        for name, bad in c["static"]["invariants"].items():
            if bad:
                out.append(f"C6: {side} {name} = {bad}")
    return out


def _canon(delta: dict) -> dict:
    out = {}
    for section in ("gates", "probes", "names", "static"):
        for key, changes in (delta.get(section) or {}).items():
            for path, ch in changes.items():
                if path in ("-", "+", "rc"):  # a gate's own change set
                    path, ch = "", {path: ch}
                for kind, v in ch.items():
                    val = sorted(v) if kind in ("-", "+") else v
                    out[f"{section}|{key}|{path}|{kind}"] = val
    return out


def compare(obs: dict, declared: dict) -> list:
    """The observed diff must EQUAL the declared delta: both directions are problems."""
    a, b = _canon(obs), _canon(declared)
    probs = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            probs.append(f"{k}: observed {a.get(k)!r} != declared {b.get(k)!r}")
    return probs


def diff(pre: dict, post: dict, declared: dict) -> tuple:
    obs = observed(pre, post)
    return obs, invariant_problems(pre, post) + compare(obs, declared)


# distinct-from: basal_ganglia/self_build_loop.py -- _restore_firewalled puts firewalled bytes back
#   from a snapshot; nothing here restores. The nearest tree copy, self_mod_harness's inline
#   copytree, copies the repo minus an ignore-pattern list for a sandboxed edit; this copies exactly
#   the git-visible tree (ls-files -co --exclude-standard) so a copy's census equals an export's.
#   Also read: substrate/selfcheck_cache.py (self_check), cortex/attention_schema.py (main),
#   cortex/eprop_kernel.py (_TinyNet), cortex/language_readout_slda.py (predict) -- each its own
#   module's self-check or readout, none a gate census, diff or tree export.
# ---- trees: a census-ready export, a self-check copy ------------------------------------------
def _worktree_files(repo: Path) -> list:
    out = _git(["ls-files", "-co", "--exclude-standard", "-z"], repo).stdout.decode("utf-8")
    return [f for f in out.split("\0") if f]


def copy_tree(repo: Path, dest: Path, marker: bool) -> Path:
    """The working tree's tracked + untracked-unignored files, byte for byte, into `dest`."""
    for f in _worktree_files(repo):
        s = Path(repo) / f
        if s.is_file():
            (Path(dest) / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, Path(dest) / f)
    if marker:
        (Path(dest) / MARKER).write_text(
            "move_gate self-check copy -- never a checkout\n", encoding="utf-8"
        )
    return Path(dest)


# distinct-from: cortex/rose_language.py -- _dotted derives one module's own dotted path; this
#   extracts a whole revision for a census. Also read: __init__.py (require_free_threaded),
#   cortex/snn_moe_classifier.py (_new_rng), cortex/theta_gamma_code.py (_new_population),
#   brainstem/so101_arm.py (probe_contact, read only) -- none an export.
def _archive(rev: str, dest: Path, repo: Path) -> None:
    p = _git(["-c", "core.autocrlf=false", "archive", "--format=tar", rev], repo)
    if p.returncode != 0:
        raise SystemExit(f"move_gate: git archive {rev} failed: {p.stderr.decode()[-500:]}")
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
        tf.extractall(dest, filter="data")
    for rec in _git(["ls-files", "--eol", "-z"], repo).stdout.decode("utf-8").split("\0"):
        meta, _, path = rec.partition("\t")
        f = dest / path
        if meta.split()[:2] == ["i/lf", "w/crlf"] and f.is_file():
            f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


def export(rev: str, dest: Path, repo: Path) -> Path:
    """A census-ready tree (see the module docstring): REV's files with this checkout's line
    endings (`git ls-files --eol`: i/lf w/crlf files are re-ended CRLF), then an index-only git.
    WORKTREE is HEAD's export with the working tree's changes (git status) laid over it byte for
    byte, so both sides of a stage's diff come from ONE export procedure and differ only in the
    files the stage touched (a mixed-EOL file the checkout holds reads the same on both sides)."""
    dest, repo = Path(dest), Path(repo)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"move_gate: {dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    _archive("HEAD" if rev == "WORKTREE" else rev, dest, repo)
    if rev == "WORKTREE":
        recs = _git(["status", "--porcelain", "-z", "--untracked-files=all"], repo).stdout
        recs = recs.decode("utf-8").split("\0")
        while recs:
            rec = recs.pop(0)
            if not rec:
                continue
            if rec[0] in "RC":
                recs.pop(0)  # a rename's second record is the old path
            src, dst = repo / rec[3:], dest / rec[3:]
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            elif dst.exists():
                dst.unlink()
    for args in (["init", "-q"], ["add", "-A", "-f"]):
        p = _git(args, dest)
        if p.returncode != 0:
            raise SystemExit(
                f"move_gate: git {args[0]} in {dest} failed: {p.stderr.decode()[-500:]}"
            )
    return dest


# ---- the reds ----------------------------------------------------------------------------------
def _mini(lines=("x 1",), live=None, legacy=None, depth=None, inv=None) -> dict:
    return {
        "gates": {"g": {"rc": 0, "lines": list(lines)}},
        "probes": {},
        "names": {},
        "static": {
            "live_tables": live or {"REGIONS": {"cortex": ["lif_neuron"]}},
            "legacy_tables": legacy or {},
            "module_depth": depth or {"lif_neuron": 1},
            "rule_dispositions": {},
            "invariants": inv or {},
        },
    }


def constructed_reds() -> list:
    """The diff engine's reds over constructed censuses (the bare self-check): each must fire,
    and each control must stay green. Returns the failures."""
    fails = []

    def expect(ok, what):
        if not ok:
            fails.append(what)

    expect(_norm("wall-clock : 2.252s", []) == "wall-clock : <t>", "a duration was not stripped")
    expect(
        _norm("<- cortex/brain.py:3688", []) == "<- cortex/brain.py",
        "a line number was not stripped",
    )
    expect(
        _norm("floor holds: 8 >= 8", []) == "floor holds: 8 >= 8",
        "a count was stripped as a position",
    )
    a, b = _mini(), _mini(("x 1", "y 2"))
    expect(diff(a, _mini(), {})[1] == [], "control: identical censuses did not diff EMPTY")
    obs, probs = diff(a, b, {})
    expect(bool(probs), "red: an undeclared line did not red")
    expect(diff(a, b, obs)[1] == [], "control: a declared delta equal to the change was not green")
    expect(bool(diff(a, a, obs)[1]), "red: a declared change that did not happen did not red")
    c = _mini()
    c["gates"]["g"]["rc"] = 1
    expect(bool(diff(a, c, {})[1]), "red: an exit-code change did not red")
    frozen = {"REGIONS": {"cortex": ["lif_neuron"]}}
    pre = _mini(("reached snn_data_lake.cortex.lif_neuron",), legacy=frozen)
    post = _mini(
        ("reached snn_data_lake.growth.lif_neuron",),
        live={"REGIONS": {"growth": ["lif_neuron"]}},
        legacy=frozen,
    )
    expect(
        diff(pre, post, {})[1] == [], "control: a pure move did not read identical through LEGACY_*"
    )
    edited = _mini(legacy={"REGIONS": {"cortex": ["lif_neuron", "planted"]}})
    probs = diff(pre, edited, observed(pre, edited))[1]
    expect(any(p.startswith("C6") for p in probs), "red: an edited LEGACY_* table did not red (C6)")
    probs = diff(a, _mini(inv={"basename_collisions": ["x"]}), {})[1]
    expect(any(p.startswith("C6") for p in probs), "red: a static violation did not red (C6)")
    probs = diff(a, _mini(depth={"lif_neuron": 2}), {})[1]
    expect(any("depth" in p for p in probs), "red: a depth change did not red (C6)")
    return fails


def _delete_exempt_row(root: Path, key: str) -> None:
    path = _file_of(root, tables(root), "dormancy_gate")
    node = _assign(path, "EXEMPT")
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            with open(path, encoding="utf-8", newline="") as f:
                lines = f.read().splitlines(keepends=True)
            del lines[k.lineno - 1 : v.end_lineno]
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("".join(lines))
            return
    raise SystemExit(f"move_gate: no EXEMPT row {key!r} to plant a deletion of")


def tree_reds(root: Path, mover: Path, log=print) -> list:
    """S1a's planted reds, each in a fresh temporary copy of `root` (never the checkout)."""
    fails = []
    work = Path(tempfile.mkdtemp(prefix="g8selfcheck_"))
    try:
        t1 = copy_tree(root, work / "c1", marker=True)
        only = {"dormancy_gate", "dormancy"}
        base = census(t1, only=only, log=log)
        if diff(base, census(t1, only=only, log=log), {})[1]:
            fails.append("C1 control: an unplanted re-census did not diff EMPTY")
        _delete_exempt_row(t1, "move_gate")
        obs, probs = diff(base, census(t1, only=only, log=log), {})
        fired = obs.get("gates", {}).get("dormancy_gate", {}).get("rc") and obs.get(
            "names", {}
        ).get("dormancy.EXEMPT")
        if not (probs and fired):
            fails.append(f"C1 red: the planted EXEMPT-row deletion did not red: {probs[:3]}")
        else:
            log(f"  RED C1 fired ({len(probs)} undeclared changes): {probs[0][:200]}")
        t2 = copy_tree(root, work / "f5", marker=True)
        keys = {"region_charge", "atlas_node"}
        before = census(t2, only=keys, log=log)["probes"]
        p = subprocess.run(
            [sys.executable, str(mover), "move", str(t2 / DOC), str(t2), F5_MODULE],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        src = t2 / "src" / PKG
        moved = (
            not (src / F5_MODULE).exists() and (src / "faculties" / Path(F5_MODULE).name).exists()
        )
        if p.returncode != 0 or not moved:
            tail = (p.stdout + p.stderr)[-800:]
            return fails + [f"F5: the mover did not move {F5_MODULE}: rc={p.returncode} {tail}"]
        log(f"  mover: {p.stdout.strip()[:300]}")
        joined = census(t2, only=keys, log=log)["probes"]
        draft = census(t2, only=keys, draft=True, log=log)["probes"]
        if joined != before:
            fails.append(
                f"F5 control: the joined move changed a region census: {_section(before, joined, {}, {})}"
            )
        lost = _section(before, draft, {}, {})
        if not lost:
            fails.append(
                "F5 red: the move with the path taken from LEGACY_* kept every region census"
            )
        else:
            log(f"  RED F5 fired (draft form): {json.dumps(lost, sort_keys=True)[:400]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="move_gate", description="G8's G-R8 census, diff and reds")
    ap.add_argument("--census", metavar="OUT")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--diff", nargs=2, metavar=("PRE", "POST"))
    ap.add_argument("--declared", metavar="DELTA")
    ap.add_argument("--observed", metavar="OUT")
    ap.add_argument("--self-check", dest="self_check", action="store_true")
    ap.add_argument("--mover", metavar="MOVER")
    ap.add_argument("--export", nargs=2, metavar=("REV", "DIR"))
    a = ap.parse_args(argv)
    if a.census:
        t0 = time.perf_counter()
        c = census(Path(a.root))
        Path(a.census).write_text(json.dumps(c, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"census of {a.root}: {len(c['gates'])} gates, {len(c['probes'])} probes, "
            f"{len(c['names'])} registries [{time.perf_counter() - t0:.1f}s] -> {a.census}"
        )
        return 0
    if a.diff:
        if not a.declared:
            ap.error("--diff needs --declared DELTA (an empty delta is the file {})")

        def load(p):
            return json.loads(Path(p).read_text(encoding="utf-8"))

        obs, probs = diff(load(a.diff[0]), load(a.diff[1]), load(a.declared))
        if a.observed:
            Path(a.observed).write_text(
                json.dumps(obs, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
        if probs:
            print(
                f"MOVE GATE RED: {len(probs)} difference(s) between the diff and the declared delta:"
            )
            for p in probs:
                print("  " + p)
            return 1
        print(
            f"move_gate OK: the diff equals the declared delta ({len(_canon(obs))} declared change(s))"
        )
        return 0
    if a.export:
        export(a.export[0], Path(a.export[1]), Path(a.root))
        print(
            f"exported {a.export[0]} of {a.root} -> {a.export[1]} (checkout line endings, index-only git)"
        )
        return 0
    fails = constructed_reds()
    if a.self_check:
        if not a.mover:
            print(
                "move_gate --self-check REFUSED: F5's red needs the out-of-tree mover (--mover MOVER); "
                "a red that cannot be built is refused, never skipped"
            )
            return 2
        fails += tree_reds(Path(a.root), Path(a.mover))
    if fails:
        print("move_gate self-check FAILED:")
        for f in fails:
            print("  " + f)
        return 1
    scope = (
        "C1's planted EXEMPT-row deletion and F5's join red fired in temporary copies"
        if a.self_check
        else "bare: the diff engine's reds; --self-check --mover builds the tree reds"
    )
    print(f"move_gate self-check OK: every constructed red fired and every control held ({scope})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
