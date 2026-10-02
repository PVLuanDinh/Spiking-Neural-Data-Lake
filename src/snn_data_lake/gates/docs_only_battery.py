"""docs_only_battery -- the battery for an arc whose diff touches only Markdown (owner, 2026-10-02:
"add the docs-only battery rule"; the runtime reader map chosen in chat over a static grep version).

WHY: the full battery is ~2 h 15 min per pushed arc whatever the diff, and pytest is ~71 min of it. A
docs-only diff can still red a test -- gates and tests READ Markdown (policy_manifest parses POLICIES.md;
38 .py files name a .md, measured 2026-10-02) -- so "run the four doc gates" is unsound. What is sound is
to run the tests that actually read the changed files, and to know which those are from RUNTIME
evidence, not from a grep that misses reads made through helpers, globs and subprocesses.

THREE PARTS, one module:
  * THE INSTRUMENT (HOOK_SOURCE). An audit hook, activated in every gate-venv Python process by a .pth
    line (`--install-hook`) ONLY while SNDL_DOC_AUDIT is set -- inert otherwise. It appends JSON lines
    to $SNDL_DOC_AUDIT/<pid>.jsonl: a `start` per process, a `read` per in-repo .md opened, a `list`
    per in-repo directory listed, a `spawn` per subprocess.Popen. Each record carries the pytest test
    FILE that was running. Children get the right id because the hook writes SNDL_DOC_AUDIT_TEST (and
    the audit variables) into a Popen `env` dict at spawn time: a dict copied from os.environ at import
    time (run_selfchecks._ENV is one) carries a STALE PYTEST_CURRENT_TEST.
  * THE MAP (`--build-map LOGDIR`): tools/doc_reader_map.json from one full pytest under the instrument,
    built during the arc's full battery and committed with its record. It is valid only for the code it
    was measured on: `code_hash` is sha256 over `git ls-files -s` minus *.md, the record and the map.
  * THE BATTERY (`--run OUTDIR`): refuses a diff with any non-.md path (R1) or a map for other code
    (R2); else runs policy_manifest --assert (halts) and --certify, run_selfchecks.py IN FULL, the
    battery gates with --assert, and pytest on the SELECTED test files (--no-cov; a collection-time
    read of a changed file runs the full pytest WITH coverage), then writes a `battery` block into
    tools/policy_certification.json.

WHY SELF-CHECKS STAY FULL: gates/selfcheck_cache.py is the incumbent self-check selector (P2: one
mechanism), and its --carry is barred from certification batteries. Only pytest is mapped.

NAMED BLIND SPOTS (the arc's pre-registration, specs/022 evidence/docs-only-battery.md): an in-process
cache filled by one test file and read by another shows the read under the first only; a Python child
started without `site` (-S/-I) carries no hook (the map counts python spawns against child starts); a
non-Python child reading Markdown is invisible except `git ls-files`/`git grep`, recorded as a scan.

Bare (run_selfchecks registers it with no argv) runs the self-check: the R1/R2 refusals and the
selection rules on constructed inputs, and the instrument's positive control, negative control and
spawn injection in real child processes. No side effects outside a temp dir.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

# distinct-from: gates/selfcheck_cache.py -- selects SELF-CHECKS by static import-closure + data-input
#   keys and carries verdicts; this selects PYTEST FILES by runtime-observed Markdown reads and carries
#   nothing (every selected file re-runs). Also read: gates/move_gate.py (census_mode classifies a tree's
#   git mode; its gate list is the G8 census, not a battery runner) -- term overlaps only.
ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = ROOT / "tools" / "doc_reader_map.json"
CERT_PATH = ROOT / "tools" / "policy_certification.json"
PTH_NAME = "sndl_doc_audit.pth"
MAP_VERSION = 1
# The record and the map are written AFTER the battery that measured the code; neither can change a verdict.
HASH_EXEMPT = frozenset({"tools/policy_certification.json", "tools/doc_reader_map.json"})
# A directory listing whose first non-plumbing caller is one of these top-level packages is the import
# system or the test runner walking the tree (measured 2026-10-02: every collection-time listing came
# from importlib's finder or pytest), never a reader of a document. READS are never dropped.
INFRA_CALLERS = frozenset(
    {
        "importlib",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "zipimport",
        "pkgutil",
        "site",
        "_pytest",
        "pluggy",
        "pytest",
        "pytest_cov",
        "coverage",
    }
)
# The battery's --assert gates (plan A.4's literal block, as run by every battery since 2026-09-29), plus
# doc_budget_gate, which run_selfchecks runs bare (printing, not failing).
BATTERY_GATES = (
    "dormancy_gate",
    "method_dormancy_gate",
    "region_charge_gate",
    "growth_coverage_gate",
    "manifest_gate",
    "determinism_gate",
    "replay_gate",
    "envelope_hash_gate",
    "policy_manifest",
    "gradient_claim_gate",
    "eprop_transport_gate",
    "import_manifest",
    "replication_guard",
    "capability_bar",
    "immortal_retention",
    "conclusion_protocol",
    "doc_budget_gate",
)

HOOK_SOURCE = r"""
def _sndl_doc_audit():
    import json, os, sys, threading
    out_dir = os.environ.get("SNDL_DOC_AUDIT")
    root = os.environ.get("SNDL_DOC_AUDIT_ROOT")
    if not out_dir or not root or getattr(sys, "_sndl_doc_audit_on", False):
        return
    sys._sndl_doc_audit_on = True  # site-packages can be reached twice: the .pth then runs twice
    root = os.path.normcase(os.path.realpath(root))
    out = os.path.join(out_dir, "%d.jsonl" % os.getpid())
    tl = threading.local()
    lock = threading.Lock()
    magic = set("*?[")
    import itertools
    spawn_no = itertools.count(1)

    def test_id():
        t = os.environ.get("SNDL_DOC_AUDIT_TEST") or os.environ.get("PYTEST_CURRENT_TEST") or ""
        return t.split("::", 1)[0].replace("\\", "/")

    def emit(rec):
        tl.busy = True
        try:
            with lock, open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        finally:
            tl.busy = False

    plumbing = ("os", "ntpath", "posixpath", "genericpath", "glob", "fnmatch", "pathlib", "shutil")

    # distinct-from: brainstem/so101_arm.py -- its close() is a lifecycle no-op; this names the module
    #   that asked for a directory listing (term overlap on "caller" only).
    def caller():
        # the first module above the filesystem plumbing that asked for the listing
        f = sys._getframe(2)
        while f is not None:
            n = f.f_globals.get("__name__", "")
            if n.split(".")[0] not in plumbing:
                return n
            f = f.f_back
        return ""

    def rel(p):
        try:
            real =os.path.realpath(os.path.join(os.getcwd(), os.fsdecode(p)))
        except Exception:
            return None
        key = os.path.normcase(real)
        if key == root:
            return "."
        if key.startswith(root + os.sep):
            return real[len(root) + 1:].replace("\\", "/")  # the file's real case, as git names it
        return None

    def hook(event, args):
        if getattr(tl, "busy", False):
            return
        if event == "open":
            p = args[0]
            if p is None or isinstance(p, int):
                return
            try:
                s = os.fsdecode(p)
            except Exception:
                return
            if s.lower().endswith(".md"):
                r = rel(s)
                if r is not None:
                    emit({"k": "read", "p": r, "t": test_id()})
        elif event in ("os.listdir", "os.scandir"):
            p = args[0] if args else None
            if isinstance(p, int):
                return
            r = rel("." if p is None else p)
            if r is not None:
                emit({"k": "list", "p": r, "t": test_id(), "c": caller()})
        elif event == "glob.glob" or event == "glob.glob/2":
            try:
                pat = os.fsdecode(args[0])
            except Exception:
                return
            parts = pat.replace("\\", "/").split("/")
            fixed = []
            for part in parts:
                if magic & set(part):
                    break
                fixed.append(part)
            if len(fixed) == len(parts):
                return  # no wildcard: an existence check (pytest's testpaths), not a listing
            base = "/".join(fixed) or "."
            if len(args) > 2 and args[2] is not None:
                base = os.path.join(os.fsdecode(args[2]), base)
            r = rel(base)
            if r is not None:
                k = "listr" if "**" in pat else "list"
                emit({"k": k, "p": r, "t": test_id(), "c": caller()})
        elif event == "subprocess.Popen":
            argv = args[1]
            env = args[3] if len(args) > 3 else None
            t = os.environ.get("SNDL_DOC_AUDIT_TEST") or os.environ.get("PYTEST_CURRENT_TEST")
            # `env` is the CALLER's own dict (Popen audits it uncopied), so a write persists in a
            # module-level dict such as run_selfchecks._ENV: re-stamp the CURRENT test on every spawn,
            # or the first test file to spawn through it would own every later spawn's reads. A dict
            # naming a DIFFERENT audit dir is a nested instrument run: leave it alone.
            # A per-spawn token pairs this spawn with the child's `start` record: on Windows the venv's
            # python.exe is a launcher, so the child's ppid names the launcher, not this process.
            tok = "%d.%d" % (os.getpid(), next(spawn_no))
            if isinstance(env, dict) and env.get("SNDL_DOC_AUDIT") in (None, out_dir):
                env["SNDL_DOC_AUDIT"] = out_dir
                env["SNDL_DOC_AUDIT_ROOT"] = root
                env["SNDL_DOC_AUDIT_TOK"] = tok
                if t:
                    env["SNDL_DOC_AUDIT_TEST"] = t
                else:
                    env.pop("SNDL_DOC_AUDIT_TEST", None)
            elif env is None:
                os.environ["SNDL_DOC_AUDIT_TOK"] = tok  # the child inherits this process's environment
            else:
                tok = None  # a nested instrument run's own dict: not this audit's child
            if isinstance(argv, (list, tuple)):
                a = [os.fsdecode(x) if isinstance(x, (str, bytes, os.PathLike)) else str(x) for x in argv]
            else:
                a = [str(argv)]
            emit({"k": "spawn", "a": a, "t": test_id(), "tok": tok})

    emit({"k": "start", "argv": list(sys.orig_argv), "ppid": os.getppid(), "t": test_id(),
          "tok": os.environ.get("SNDL_DOC_AUDIT_TOK")})
    sys.addaudithook(hook)


_sndl_doc_audit()
del _sndl_doc_audit
"""


# --- the instrument's installation -------------------------------------------------------------


def pth_line() -> str:
    """The one .pth line: inert unless SNDL_DOC_AUDIT is set (no import beyond os otherwise)."""
    return f'import os; exec({HOOK_SOURCE!r}) if os.environ.get("SNDL_DOC_AUDIT") else None\n'


def pth_path() -> Path:
    return Path(sysconfig.get_paths()["purelib"]) / PTH_NAME


def install_hook() -> Path:
    p = pth_path()
    p.write_text(pth_line(), encoding="utf-8")
    return p


def uninstall_hook() -> bool:
    p = pth_path()
    if p.exists():
        p.unlink()
        return True
    return False


# --- the map --------------------------------------------------------------------------------------


def _git(*args: str, root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def code_hash(ls_files_stage: str) -> str:
    """sha256 over `git ls-files -s` lines whose path is not Markdown and not HASH_EXEMPT."""
    keep = []
    for line in ls_files_stage.splitlines():
        path = line.split("\t", 1)[-1]
        if path.lower().endswith(".md") or path in HASH_EXEMPT:
            continue
        keep.append(line)
    return hashlib.sha256("\n".join(keep).encode("utf-8")).hexdigest()


# distinct-from: reasoning/arc_diagnostic.py -- a verdict line from measured numbers; this tokenizes a
#   spawn command line. Also read: brainstem/reflex.py (reflex step), cerebellum/spiking_jepa.py
#   (encoder), gates/gradient_claim_gate.py (baseline loader), reasoning/spike_chat_lake.py (chat
#   clause) -- term overlaps only.
def _tokens(argv: list[str]) -> list[str]:
    """On Windows the Popen audit event carries ONE command-line string (measured 2026-10-02), not a
    list, and it may embed quoted `-c` code -- so only the executable is parsed exactly (quoted or not)
    and the rest is split on whitespace; never raises."""
    if len(argv) != 1:
        return argv
    s = argv[0].lstrip()
    if s.startswith('"') and '"' in s[1:]:
        end = s.index('"', 1)
        return [s[1:end], *s[end + 1 :].split()]
    return s.split()


def _is_git_scan(argv: list[str]) -> bool:
    toks = _tokens(argv)
    return (
        bool(toks)
        and os.path.basename(toks[0]).lower() in ("git", "git.exe")
        and any(a in ("ls-files", "grep") for a in toks)
    )


def _is_python(argv: list[str]) -> bool:
    toks = _tokens(argv)
    return bool(toks) and os.path.basename(toks[0]).lower().startswith("python")


def _empty() -> dict:
    return {"read": set(), "list": set(), "listr": set(), "git_scan": False}


def build_map(records: list[dict], head: str, chash: str) -> dict:
    """Fold instrument records into the map. Records without a test id in a pytest process are
    collection-time reads (pytest_all); records without a test id elsewhere are outside pytest and
    unused (self-checks and gates always run in full)."""
    tests: dict[str, dict] = {}
    pytest_all: dict[str, set] = {"read": set(), "list": set(), "listr": set()}
    starts = spawns_py = dropped = 0
    is_pytest: dict[str, bool] = {}
    started_toks = set()
    unpaired: list[str] = []
    for rec in records:
        if rec.get("k") == "start":
            starts += 1
            is_pytest[rec["_pid"]] = "pytest" in " ".join(rec.get("argv", []))
            if rec.get("tok"):
                started_toks.add(rec["tok"])
    for rec in records:
        k = rec.get("k")
        t = rec.get("t") or ""
        if k == "spawn":
            if _is_python(rec.get("a", [])):
                spawns_py += 1
                if rec.get("tok") and rec["tok"] not in started_toks:
                    unpaired.append(
                        " ".join(_tokens(rec.get("a", []))[1:4]) + f"  [{t or 'no test'}]"
                    )
            if t and _is_git_scan(rec.get("a", [])):
                tests.setdefault(t, _empty())["git_scan"] = True
            continue
        if k not in ("read", "list", "listr"):
            continue
        if k != "read" and str(rec.get("c", "")).split(".")[0] in INFRA_CALLERS:
            dropped += 1  # an import finder or pytest's own walk lists dirs; it reads no document
            continue
        if t:
            tests.setdefault(t, _empty())[k].add(rec["p"])
        elif is_pytest.get(rec["_pid"]):
            pytest_all[k].add(rec["p"])
    return {
        "version": MAP_VERSION,
        "head": head,
        "code_hash": chash,
        "built": _now(),
        "coverage": {
            "process_starts": starts,
            "python_spawns_seen_by_parents": spawns_py,
            "listings_dropped_as_infrastructure": dropped,
            # each a python child whose hook never started: a NAMED blind spot (no hook = no reads seen)
            "python_spawns_without_child_start": sorted(unpaired),
        },
        "pytest_all": {k: sorted(v) for k, v in pytest_all.items()},
        "tests": {
            t: {
                "read": sorted(v["read"]),
                "list": sorted(v["list"]),
                "listr": sorted(v["listr"]),
                "git_scan": v["git_scan"],
            }
            for t, v in sorted(tests.items())
        },
    }


def load_records(log_dir: Path) -> list[dict]:
    out = []
    for f in sorted(Path(log_dir).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["_pid"] = f.stem
                out.append(rec)
    return out


# --- selection and refusals -----------------------------------------------------------------------


class Refused(Exception):
    """A docs-only battery may not run on this diff or this map."""


def check_changed(changes: list[tuple[str, str]]) -> None:
    """R1: every changed path must be Markdown."""
    if not changes:
        raise Refused("R1: the diff origin/main..HEAD is empty -- nothing to verify")
    bad = sorted(p for _, p in changes if not p.lower().endswith(".md"))
    if bad:
        raise Refused(f"R1: non-Markdown paths changed -- run the full battery: {bad[:10]}")


def check_map(m: dict | None, chash: str) -> None:
    """R2: the map must exist, be this version, and describe this code."""
    if not m:
        raise Refused(
            "R2: no tools/doc_reader_map.json -- run the full battery (instrumented) first"
        )
    if m.get("version") != MAP_VERSION:
        raise Refused(f"R2: map version {m.get('version')} != {MAP_VERSION}")
    if m.get("code_hash") != chash:
        raise Refused(
            "R2: the map was measured on other code (map head "
            f"{str(m.get('head'))[:10]}) -- run the full battery, which rebuilds it"
        )


def _ancestors(path: str) -> list[str]:
    parts = path.split("/")[:-1]
    return ["."] + ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _touches(entry: dict, status: str, path: str) -> bool:
    if path in entry["read"]:
        return True
    if status in ("A", "D"):
        parent = "/".join(path.split("/")[:-1]) or "."
        if parent in entry["list"] or any(a in entry["listr"] for a in _ancestors(path)):
            return True
        if entry.get("git_scan"):
            return True
    return False


# distinct-from: readout/language_readout_slda.py -- its _split is a train/test data split; this
#   selects test FILES and lowercases repo PATHS so a Windows normcased record (seen 2026-10-02) still
#   matches git's spelling. Also read: sensorimotor/corpus_stream.py (word tokens),
#   cortex/self_improvement_suite.py, growth/arealization.py, instruments/pool_selection.py -- term
#   overlaps only.
def _folded(entry: dict) -> dict:
    return {k: ({p.lower() for p in v} if k != "git_scan" else v) for k, v in entry.items()}


def select(m: dict, changes: list[tuple[str, str]]) -> tuple[bool, list[str]]:
    """(run the FULL pytest?, the test files to run). Statuses are git's A / M / D (renames split)."""
    ch = [(s, p.lower()) for s, p in changes]
    full = any(_touches(_folded({**m["pytest_all"], "git_scan": False}), s, p) for s, p in ch)
    files = sorted(
        t for t, e in m["tests"].items() if any(_touches(_folded(e), s, p) for s, p in ch)
    )
    return full, files


def changed_paths(base: str = "origin/main", head: str = "HEAD") -> list[tuple[str, str]]:
    out = []
    for line in _git("diff", "--name-status", "--no-renames", base, head).splitlines():
        status, path = line.split("\t", 1)
        out.append((status[0], path))
    return out


# --- the battery ----------------------------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def run(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    home = out_dir / "home"
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "NORD_PERSIST": "0",
        "NORD_HOME": str(home),
        "PYTHONPATH": str(ROOT / "src"),
    }
    for v in ("SNDL_DOC_AUDIT", "SNDL_DOC_AUDIT_ROOT", "SNDL_DOC_AUDIT_TEST"):
        env.pop(v, None)
    head = _git("rev-parse", "HEAD").strip()
    (out_dir / "started.txt").write_text(f"{_now()}\n{head}\n", encoding="utf-8")
    try:
        changes = changed_paths()
        check_changed(changes)
        m = json.loads(MAP_PATH.read_text(encoding="utf-8")) if MAP_PATH.exists() else None
        check_map(m, code_hash(_git("ls-files", "-s")))
    except Refused as e:
        print(f"REFUSED {e}")
        (out_dir / "verdict.txt").write_text("REFUSED\n", encoding="utf-8")
        return 2
    full, files = select(m, changes)
    print(
        f"changed {len(changes)} Markdown file(s); pytest: "
        + ("FULL (a collection-time reader)" if full else f"{len(files)} selected file(s)")
    )
    py = sys.executable

    def stage(name: str, argv: list[str]) -> int:
        with open(out_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            rc = subprocess.run(
                argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT
            ).returncode
        print(f"{name} rc={rc}", flush=True)
        return rc

    if stage("record_assert", [py, "-m", "snn_data_lake.gates.policy_manifest", "--assert"]):
        print("RECORD-ASSERT RED: battery halted")
        (out_dir / "verdict.txt").write_text("HALTED\n", encoding="utf-8")
        return 1
    rcs = [stage("certify", [py, "-m", "snn_data_lake.gates.policy_manifest", "--certify"])]
    rcs.append(stage("selfchecks", [py, "run_selfchecks.py"]))
    for g in BATTERY_GATES:
        rcs.append(stage(g, [py, "-m", f"snn_data_lake.gates.{g}", "--assert"]))
    if full:
        rcs.append(stage("pytest", [py, "-m", "pytest", "-p", "no:cacheprovider"]))
    elif files:
        rcs.append(
            stage("pytest", [py, "-m", "pytest", "-p", "no:cacheprovider", "--no-cov", *files])
        )
    else:
        print("pytest: no test file reads the changed Markdown -- skipped (the map says so)")
    green = not any(rcs)
    record = json.loads(CERT_PATH.read_text(encoding="utf-8"))
    record["battery"] = {
        "kind": "docs-only",
        "verdict": "GREEN" if green else "RED",
        "changed": [f"{s} {p}" for s, p in changes],
        "pytest": "full" if full else files,
        "map_head": m["head"],
        "map_code_hash": m["code_hash"],
    }
    CERT_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "verdict.txt").write_text(("GREEN" if green else "RED") + "\n", encoding="utf-8")
    print("VERDICT", "GREEN" if green else "RED")
    return 0 if green else 1


# --- self-check -----------------------------------------------------------------------------------


def _child(code: str, env: dict) -> None:
    # -S: no `site`, so no installed .pth hook -- the child runs THIS module's HOOK_SOURCE, not
    # whatever version was installed (without it a mutant of the hook passed, seen 2026-10-02).
    subprocess.run([sys.executable, "-S", "-c", code], env=env, check=True, timeout=60)


def _selfcheck_rules() -> None:
    # R1: a non-Markdown path refuses; an empty diff refuses; a Markdown-only diff passes.
    for bad in ([("M", "docs/a.md"), ("M", "src/x.py")], []):
        try:
            check_changed(bad)
        except Refused:
            pass
        else:
            raise AssertionError(f"R1 did not refuse {bad}")
    check_changed([("M", "docs/a.md"), ("A", "specs/b/C.MD")])
    print("  RED    R1 refuses a non-.md path and an empty diff; a .md-only diff passes")

    # code_hash ignores Markdown and the two exempt files, and sees any other byte.
    base = (
        "100644 aaa 0\tsrc/x.py\n100644 bbb 0\tdocs/a.md\n100644 ccc 0\ttools/doc_reader_map.json"
    )
    assert code_hash(base) == code_hash(base.replace("bbb", "zzz").replace("ccc", "yyy"))
    assert code_hash(base) != code_hash(base.replace("aaa", "abc"))

    # R2: a missing map, a wrong version and a map for other code all refuse.
    good = {"version": MAP_VERSION, "code_hash": "h", "head": "x"}
    check_map(good, "h")
    for bad_map, h in ((None, "h"), ({**good, "version": 0}, "h"), (good, "other")):
        try:
            check_map(bad_map, h)
        except Refused:
            pass
        else:
            raise AssertionError(f"R2 did not refuse {bad_map!r} against {h}")
    print("  RED    R2 refuses a missing map, a wrong version and a map for other code")

    # Selection: a reader of the changed file; a lister of its directory on ADD only; a recursive
    # glob over an ancestor; a git scanner on ADD; a collection-time read forces the full pytest.
    e = {"read": [], "list": [], "listr": [], "git_scan": False}
    m = {
        "pytest_all": {"read": ["docs/POLICIES.md"], "list": [], "listr": []},
        "tests": {
            "tests/test_reader.py": {**e, "read": ["docs/a.md"]},
            "tests/test_lister.py": {**e, "list": ["specs/ev"]},
            "tests/test_globber.py": {**e, "listr": ["specs"]},
            "tests/test_scanner.py": {**e, "git_scan": True},
            "tests/test_none.py": {**e, "read": ["docs/other.md"]},
        },
    }
    assert select(m, [("M", "docs/a.md")]) == (False, ["tests/test_reader.py"])
    assert select(m, [("M", "specs/ev/x.md")]) == (False, [])
    assert select(m, [("A", "specs/ev/x.md")]) == (
        False,
        ["tests/test_globber.py", "tests/test_lister.py", "tests/test_scanner.py"],
    )
    assert select(m, [("M", "docs/POLICIES.md")])[0] is True
    assert select(m, [("M", "DOCS/A.md")]) == (False, ["tests/test_reader.py"])  # case-insensitive
    # The spawn classifiers, on the Windows command-line string and on a list.
    assert _is_python(['"C:\\a b\\python.exe" -c pass']) and _is_python(["/usr/bin/python3", "-c"])
    assert _is_git_scan(['"C:\\Program Files\\Git\\cmd\\git.exe" ls-files -z'])
    assert not _is_git_scan(["git", "status"]) and not _is_python(["git", "ls-files"])
    assert _is_python(['python.exe -c "print(\'x)"'])  # unbalanced quotes inside -c code
    print(
        "  RED    selection: reader on modify; lister/recursive glob/git scan on add; collection read -> full"
    )


def _selfcheck_instrument() -> None:
    # In real child processes: H+ a read is recorded with its test file; H- inert without the variable;
    # HS a child that spawns with an env dict LACKING the audit variables still arms the grandchild,
    # under the parent's test id.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        logs.mkdir()
        repo = tmp / "repo"
        (repo / "docs").mkdir(parents=True)
        md = repo / "docs" / "k.md"
        md.write_text("x", encoding="utf-8")
        base_env = {
            k: v for k, v in os.environ.items() if not k.startswith(("SNDL_DOC_AUDIT", "PYTEST_"))
        }
        on = {
            **base_env,
            "SNDL_DOC_AUDIT": str(logs),
            "SNDL_DOC_AUDIT_ROOT": str(repo),
            "PYTEST_CURRENT_TEST": "tests/test_k.py::test_k (call)",
            # explicit: under an instrumented battery the PARENT's hook would otherwise fill it with
            # the running pytest file (seen 2026-10-02: this check red only under audit)
            "SNDL_DOC_AUDIT_TEST": "tests/test_k.py::test_k (call)",
        }
        reader = HOOK_SOURCE + f"\nopen({str(md)!r}).read()\n"
        _child(reader, on)
        reads = [r for r in load_records(logs) if r["k"] == "read"]
        assert [(r["p"], r["t"]) for r in reads] == [("docs/k.md", "tests/test_k.py")], reads
        n_files = len(list(logs.iterdir()))
        _child(reader, base_env)
        assert len(list(logs.iterdir())) == n_files, "H-: the hook logged without SNDL_DOC_AUDIT"
        parent = (
            HOOK_SOURCE
            + "\nimport os, subprocess, sys\n"
            + "env = {k: v for k, v in os.environ.items() "
            + "if not k.startswith(('SNDL_DOC_AUDIT', 'PYTEST_'))}\n"
            + f"subprocess.run([sys.executable, '-S', '-c', {reader!r}], env=env, check=True)\n"
        )
        for f in logs.iterdir():
            f.unlink()
        _child(parent, on)
        recs = load_records(logs)
        reads = [r for r in recs if r["k"] == "read"]
        spawns = [r for r in recs if r["k"] == "spawn"]
        assert len(reads) == 1 and reads[0]["t"] == "tests/test_k.py", recs
        assert len(spawns) == 1 and reads[0]["_pid"] != spawns[0]["_pid"], recs
        mp = build_map(recs, "h", "c")
        assert mp["tests"]["tests/test_k.py"]["read"] == ["docs/k.md"], mp
        cov = mp["coverage"]
        assert (cov["process_starts"], cov["python_spawns_seen_by_parents"]) == (2, 1), mp
        assert cov["python_spawns_without_child_start"] == [], mp  # the token paired the spawn
        # CV red: a python child started WITHOUT the hook (-S -I: no site, no .pth, no explicit hook)
        # leaves its parent's spawn unpaired, and the map names it.
        for f in logs.iterdir():
            f.unlink()
        blind = HOOK_SOURCE + (
            "\nimport subprocess, sys\n"
            "subprocess.run([sys.executable, '-S', '-I', '-c', 'pass'], check=True)\n"
        )
        _child(blind, on)
        cov = build_map(load_records(logs), "h", "c")["coverage"]
        assert len(cov["python_spawns_without_child_start"]) == 1, cov
        # SHARED DICT (advisor, 2026-10-02): one module-level env dict, two spawns under two test ids --
        # each child logs under its OWN test file (setdefault pinned the first id; reachable red).
        for f in logs.iterdir():
            f.unlink()
        shared = (
            HOOK_SOURCE
            + "\nimport os, subprocess, sys\n"
            + "D = {k: v for k, v in os.environ.items() if not k.startswith(('SNDL_DOC_AUDIT_TEST',))}\n"
            + "os.environ.pop('SNDL_DOC_AUDIT_TEST', None)\n"
            + "for t in ('tests/test_a.py::x (call)', 'tests/test_b.py::y (call)'):\n"
            + "    os.environ['PYTEST_CURRENT_TEST'] = t\n"
            + f"    subprocess.run([sys.executable, '-S', '-c', {reader!r}], env=D, check=True)\n"
        )
        _child(shared, on)
        got = sorted(r["t"] for r in load_records(logs) if r["k"] == "read")
        assert got == ["tests/test_a.py", "tests/test_b.py"], got
        # Globs: a literal path is an existence check (no record); `*` lists one dir; `**` is recursive.
        for f in logs.iterdir():
            f.unlink()
        globber = (
            HOOK_SOURCE
            + f"\nimport glob, os\nos.chdir({str(repo)!r})\n"
            + "glob.glob('docs')\nglob.glob('docs/*.md')\nglob.glob('docs/**/*.md', recursive=True)\n"
        )
        _child(globber, on)
        got = sorted((r["k"], r["p"]) for r in load_records(logs) if r["k"] in ("list", "listr"))
        assert ("list", "docs") in got and ("listr", "docs") in got, got
        assert all(p != "." or k == "list" for k, p in got), got  # no recursive walk of the root
    # Infrastructure listings are dropped from the map; a test's own listing is kept; reads always kept.
    recs = [
        {"k": "start", "argv": ["python", "-m", "pytest"], "_pid": "1"},
        {"k": "list", "p": "src/x", "t": "", "c": "_frozen_importlib_external", "_pid": "1"},
        {"k": "listr", "p": ".", "t": "", "c": "_pytest.main", "_pid": "1"},
        {"k": "list", "p": "docs", "t": "tests/test_a.py", "c": "tests.test_a", "_pid": "1"},
        {"k": "read", "p": "docs/POLICIES.md", "t": "", "c": "_pytest.x", "_pid": "1"},
    ]
    mp = build_map(recs, "h", "c")
    assert mp["pytest_all"] == {"read": ["docs/POLICIES.md"], "list": [], "listr": []}, mp
    assert mp["tests"]["tests/test_a.py"]["list"] == ["docs"], mp
    assert mp["coverage"]["listings_dropped_as_infrastructure"] == 2, mp
    print(
        "  RED    instrument: H+ read recorded with its test file; H- silent without the variable; "
        "HS a stripped env dict is re-armed and the grandchild logs under the parent's test; "
        "globs by kind; import/pytest listings dropped, reads kept"
    )


def _selfcheck() -> None:
    _selfcheck_rules()
    _selfcheck_instrument()
    print("docs_only_battery self-check OK")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "--self-check":
        _selfcheck()
        return 0
    cmd = argv[0]
    if cmd == "--install-hook":
        print(f"installed {install_hook()}")
        return 0
    if cmd == "--uninstall-hook":
        print("removed" if uninstall_hook() else "not installed")
        return 0
    if cmd == "--build-map" and len(argv) == 2:
        # The .pth embeds HOOK_SOURCE as it was at install: a hook edited since then measured the run
        # with OLD code (seen 2026-10-02: the probe after the case fix still logged lowercase).
        installed = pth_path().read_text(encoding="utf-8") if pth_path().exists() else ""
        if installed != pth_line():
            print(
                "REFUSED: the installed hook is missing or stale -- run --install-hook, then re-run pytest"
            )
            return 2
        recs = load_records(Path(argv[1]))
        m = build_map(recs, _git("rev-parse", "HEAD").strip(), code_hash(_git("ls-files", "-s")))
        MAP_PATH.write_text(
            json.dumps(m, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        cov = m["coverage"]
        print(
            f"map: {sum(1 for e in m['tests'].values() if e['read'])} test files READ Markdown, "
            f"{len(m['tests'])} with any record (reads, listings, git scans); collection-time reads "
            f"{len(m['pytest_all']['read'])}; process starts {cov['process_starts']}, python "
            f"spawns seen by parents {cov['python_spawns_seen_by_parents']}, of them without a child "
            f"start {len(cov['python_spawns_without_child_start'])}"
        )
        for line in cov["python_spawns_without_child_start"][:20]:
            print(f"  BLIND (no hook in the child): {line}")
        return 0
    if cmd == "--select":
        m = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        full, files = select(m, changed_paths(*argv[1:3]) if len(argv) > 1 else changed_paths())
        print("FULL pytest" if full else ("\n".join(files) or "(no test file selected)"))
        return 0
    if cmd == "--run" and len(argv) == 2:
        return run(Path(argv[1]))
    print(__doc__)
    print(
        "usage: [--self-check] | --install-hook | --uninstall-hook | --build-map LOGDIR | "
        "--select [BASE HEAD] | --run OUTDIR"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
