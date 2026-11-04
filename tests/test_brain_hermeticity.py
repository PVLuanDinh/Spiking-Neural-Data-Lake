"""T120 rung 3: the hermeticity census. A temp-home Brain keeps its learned bytes in its home.

One fresh process builds one Brain on a temp NORD_HOME under a `sys.addaudithook` recorder installed
before the package is imported. It records every read `open` (builtin `open`, `io.open_code`,
`os.open`) and `sqlite3.connect` under the REPO or the LIVE home (computed with NORD_HOME unset),
and every write -- an `open` or connect for writing, or an event in `_WRITES` -- wherever it points.
`os.replace` raises `os.rename`; on Windows `shutil.copy2` and a cross-volume `shutil.move` raise
`_winapi.CopyFile2` instead of `shutil.copyfile`. Each recorded path is classified:

  * CODE -- the package's source and extension modules read under `src/`, and Python bytecode in a
    `__pycache__/` directory (and that directory) outside the live home;
  * ASSET -- a read-only repo input on the closed `ASSETS` list below, each with its reason;
  * anything else is LEARNED STATE outside the home, and reds: a repo file off the list, any write
    outside the home and outside CODE, and any open of the live home.

A write outside the repo, the home and the live home counts only if a FILE is still there when the
process exits (a directory holds no learned bytes; a file written into it is recorded on its own):
the Brain's working-memory store is a `mkstemp` file it removes on `close`, and `tempfile` probes
its directory with a file it removes at once. Those are reported as `scratch`, not hidden.

A test, not a CI gate (P4c: the census output is counted before any rule lands). The planted red
re-adds the pre-T120 repo read of the self-build learner, plus a repo write through every recording
call and sqlite connect form (plain, URI, unparsable), a write then a read of one path (a write is
never downgraded), read-only, in-memory and empty connects, a read and a write in a stand-in live
home, a left-behind and a scratch write in a private temp directory outside the home, and (POSIX)
`dir_fd`-relative calls that must be skipped.

LIMITS (policy-enforcer #5476 m1): the hook sees Python-level calls only. File I/O done inside a C
extension -- torch, h5py, the `_native` extension, sqlite's own journal -- raises no audit event and
is invisible here. The run covers one Brain build, 20 ticks, one propose and close; paths a daemon
reaches later (sleep, the NORD_SELFMOD wake self-build) are outside it. A removal, rename, mkdir or
link made relative to an open directory (`dir_fd=`, as POSIX `shutil.rmtree` does) is skipped, not
resolved; an `os.open(..., dir_fd=)` (its audit event carries no dir_fd) is resolved against the
child's cwd, the repo, and can red falsely. A `sqlite3.connect("file:...")` without `uri=True` opens
a plain file (`file` with an alternate stream on Windows, the whole string elsewhere), which the
census reads as the URI's path.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ENCODER = (
    "a frozen perception encoder or pack (gitignored; trained offline, never written by a Brain)"
)
# Closed allowlist, one entry per file: repo-relative path -> why the Brain may read it.
ASSETS = {
    **{
        f"models/{name}": _ENCODER
        for name in (
            "eprop_nmnist.pt",
            "eprop_shd.pt",
            "nmnist_pack.npz",
            "shd_pack.npz",
            "pc_cortex_nmnist.json",
        )
    },
    "tools/lake_append_baseline.json": "the committed per-tag append budget "
    "`substrate/lake_engine` checks appends against (a host measurement, not learning)",
    "graphify-out/graph.json": "the generated import graph "
    "`connectome._import_graph_is_untyped` reads as a fair control (gitignored, rebuilt by "
    "graphify, never written by a Brain)",
}

_CHILD = r"""
import json, os, re, sys
from urllib.parse import parse_qs, unquote, urlsplit

home, root, live = (os.path.normcase(os.path.abspath(p)) for p in sys.argv[1:4])
plant = sys.argv[4]  # "" for the census; for the planted run, a private directory outside the home
seen = {}
_W = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND
# event -> (argument positions it writes, its dir_fd positions); a link's or copy's source is read
_WRITES = {"os.rename": ((0, 1), (2, 3)), "os.remove": ((0,), (1,)), "os.rmdir": ((0,), (1,)),
           "os.mkdir": ((0,), (2,)), "os.truncate": ((0,), ()), "os.link": ((1,), (2, 3)),
           "os.symlink": ((1,), (2,)), "shutil.copyfile": ((1,), ()), "shutil.copytree": ((1,), ()),
           "shutil.rmtree": ((0,), (1,)), "_winapi.CopyFile2": ((1,), ())}


def key(path):
    p = os.path.normcase(os.path.abspath(path))
    for base, tag in ((home, None), (root, ""), (live, "<live>/")):
        if p == base or p.startswith(base + os.sep):
            return None if tag is None else tag + os.path.relpath(p, base).replace(os.sep, "/")
    return "<out>" + p.replace(os.sep, "/")


def note(path, wrote):
    k = key(path)
    if k is None:
        return
    if wrote:
        seen[k] = True  # sticky: events from different threads arrive in any order
    elif not k.startswith("<out>"):
        seen.setdefault(k, False)


# (path, wrote) of what `sqlite3.connect(db)` opens; None for an in-memory database or an empty
# path (anonymous, root or host-only URI)
def sqlite_target(db):
    mode = ""
    if db.startswith("file:"):
        try:
            u = urlsplit(db)
        except ValueError:  # unparsable (an unclosed "["): fail closed, as a write
            return db, True
        db, mode = unquote(u.path), parse_qs(u.query).get("mode", [""])[-1]
        if db.startswith("/"):
            db = "/" + db.lstrip("/")
            if re.match(r"/[A-Za-z]:", db):
                db = db[1:]
    if db in ("", "/", ":memory:") or mode == "memory":
        return None
    return db, mode != "ro"


def _path(a):
    return os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else None


def hook(event, args):
    if event in _WRITES:
        paths, fds = _WRITES[event]
        if any(i < len(args) and isinstance(args[i], int) and args[i] >= 0 for i in fds):
            return  # relative to an open directory: not resolvable here (see LIMITS)
        for i in paths:
            p = _path(args[i]) if i < len(args) else None
            if p:
                note(p, True)
        return
    if event not in ("open", "sqlite3.connect") or not args or not (path := _path(args[0])):
        return
    if event == "sqlite3.connect":
        target = sqlite_target(path)
        if target:
            note(*target)
        return
    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    note(path, any(c in mode for c in "wax+") or bool(flags & _W))


sys.addaudithook(hook)
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.substrate.home import assert_not_live_home, home as _home

assert_not_live_home(_home())
if plant:
    from snn_data_lake.infra import brain_native_selfbuild as m

    real = m.BrainNativeSelfBuildProposer._load

    def planted(self):
        try:
            open(os.path.join(self._repo, "data", "selfbuild_state.json"), "rb").close()
        except OSError:
            pass
        real(self)

    m.BrainNativeSelfBuildProposer._load = planted
    absent = os.path.join(root, "models", "_t120_absent")  # never created: the event fires first
    assert not os.path.lexists(absent), absent  # else the probes below would act on real files
    import shutil
    import sqlite3

    def a(name):
        return os.path.join(absent, name)

    for probe in (lambda: open(a("a"), "w"),
                  lambda: os.open(a("b"), os.O_WRONLY | os.O_CREAT),
                  lambda: os.replace(a("src"), a("r")),
                  lambda: os.remove(a("rm")),
                  lambda: os.rmdir(a("rd")),
                  lambda: os.mkdir(a("md")),
                  lambda: os.truncate(a("tr"), 0),
                  lambda: os.link(a("ls"), a("ld")),
                  lambda: os.symlink(a("ss"), a("sd")),
                  lambda: shutil.copyfile(a("cs"), a("cd")),
                  lambda: shutil.copy2(a("ks"), a("kd")),
                  lambda: shutil.copytree(a("ts"), a("td")),
                  lambda: shutil.rmtree(a("rt")),
                  lambda: sys.audit("_winapi.CopyFile2", a("ws"), a("wd"), 0),
                  lambda: open(a("wr"), "w"),
                  lambda: open(a("wr"), "rb"),
                  lambda: sqlite3.connect(a("p.db")),
                  lambda: sqlite3.connect(f"file:{absent}/c.db?mode=rwc", uri=True),
                  lambda: sqlite3.connect(f"file:///{absent}/d.db?mode=rwc", uri=True),
                  lambda: sqlite3.connect(f"file://localhost/{absent}/e.db?mode=rwc", uri=True),
                  lambda: sqlite3.connect(f"file:{absent}/f%2Edb?mode=rwc", uri=True),
                  lambda: sqlite3.connect(f"file:{root}/models/_t120_absent.db?mode=ro", uri=True),
                  lambda: sqlite3.connect(f"file:{absent}/g.db?mode=ro#x", uri=True),
                  lambda: sqlite3.connect(f"file://[{absent}/v.db?mode=rwc", uri=True),
                  lambda: sqlite3.connect("file://h", uri=True),
                  lambda: sqlite3.connect("file:", uri=True).close(),
                  lambda: sqlite3.connect(":memory:").close(),
                  lambda: sqlite3.connect("file::memory:?cache=shared", uri=True).close(),
                  lambda: sqlite3.connect("file:x.db?mode=memory", uri=True).close(),
                  lambda: open(os.path.join(live, "_t120_absent", "r"), "rb"),
                  lambda: open(os.path.join(live, "_t120_absent", "a"), "w"),
                  lambda: open(os.path.join(plant, "_t120_absent", "a"), "w"),
                  lambda: open(os.path.join(plant, "left"), "w").close()):
        try:
            probe()
        except (OSError, sqlite3.Error):
            pass
    if os.mkdir in os.supports_dir_fd:  # POSIX: names relative to an open directory, skipped
        fd = os.open(plant, os.O_RDONLY)
        for probe in (lambda: os.mkdir("_t120_fd", dir_fd=fd),
                      lambda: os.remove("_t120_fd_rm", dir_fd=fd)):
            try:
                probe()
            except OSError:
                pass
        os.close(fd)
from snn_data_lake.engine.brain import make_brain

b = make_brain(seed=0, persist=True)
for _ in range(20):
    b.tick()
b.propose_improvements(budget=1)
s = b.state()
b.close()

scratch = sorted(k for k in seen if k.startswith("<out>") and not os.path.isfile(k[len("<out>") :]))
for k in scratch:
    del seen[k]
print("CENSUS " + json.dumps({"opened": seen, "scratch": scratch, "tier": s.get("perception_tier"),
                              "audio": s.get("perceiving_audio")}))
"""


def _child_env(home):
    """The census child's environment: the brain on `home`, with the battery's doc-audit hook OFF.

    The arc battery sets SNDL_DOC_AUDIT; its .pth hook (gates/docs_only_battery) appends
    `<audit>/<pid>.jsonl` from every process, and re-injects the variable into a child whose
    environment lacks it. That is the battery's instrument, not the brain, and the census redded it
    as a write outside the home (engine arc 3 battery 1). An EMPTY value is the hook's own opt-out
    (docs_only_battery's tests use it): not re-injected, and the .pth line stays inert."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SNDL_DOC_AUDIT")}
    env.update(PYTHONUTF8="1", NORD_HOME=home, SNDL_DOC_AUDIT="")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (os.path.join(_REPO, "src"), env.get("PYTHONPATH", "")) if p
    )
    return env


def _child(root, live, plant):
    """Run `_CHILD` on a fresh temp home (asserted not live) against `root`."""
    home = tempfile.mkdtemp(prefix="snn_t120_census_")
    from snn_data_lake.substrate.home import assert_not_live_home

    assert_not_live_home(home)
    return subprocess.run(
        [sys.executable, "-c", _CHILD, home, root, live, plant],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(home),
        cwd=_REPO,
        timeout=900,
    )


def _census(plant=""):
    """The census of one child Brain; `plant` is a private directory for the planted run."""
    from snn_data_lake.substrate.home import _default_live_home, _resolve

    # The planted run never names the operator's home: its live home is a path inside `plant`.
    live = os.path.join(plant, "live") if plant else _resolve(_default_live_home())
    p = _child(_REPO, live, plant)
    for line in p.stdout.splitlines():
        if line.startswith("CENSUS "):
            return json.loads(line[len("CENSUS ") :])
    raise AssertionError(f"no census (rc={p.returncode})\n{p.stderr[-2000:]}")


# Python's bytecode, the `<name>.pyc.<id>` temp file its atomic write replaces it from, and the
# `__pycache__` directory itself.
_PYC = re.compile(r"/__pycache__(/[^/]+\.pyc(\.\d+)?)?$")


def _classify(opened):
    """{key: (kind, why)} for every path the census recorded."""
    out = {}
    for k, wrote in sorted(opened.items()):
        if k.startswith("<live>/"):
            out[k] = ("STATE", "the live home")
        elif _PYC.search(k):
            out[k] = ("CODE", None)
        elif k.startswith("src/") and k.endswith((".py", ".pyc", ".pyd", ".so")):
            out[k] = ("STATE", "a write into src/") if wrote else ("CODE", None)
        elif wrote:
            out[k] = ("STATE", "a write outside the home")
        elif k in ASSETS:
            out[k] = ("ASSET", ASSETS[k])
        else:
            out[k] = ("STATE", "not on the asset allowlist")
    return out


def _absent_assets(prefix=""):
    return [
        k for k in ASSETS if k.startswith(prefix) and not os.path.exists(os.path.join(_REPO, k))
    ]


def _stale(opened, absent):
    """Allowlisted assets present on disk that no Brain opened (an absent one cannot be opened)."""
    return [k for k in ASSETS if k not in absent and k not in opened]


@pytest.fixture(scope="module")
def census():
    return _census()


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_a_temp_home_brain_writes_no_learned_state_outside_its_home(census):
    kinds = _classify(census["opened"])
    print(
        json.dumps(
            {
                "tier": census["tier"],
                "audio": census["audio"],
                "files": kinds,
                "scratch": census["scratch"],
            },
            indent=1,
        )
    )
    reds = {k: v[1] for k, v in kinds.items() if v[0] == "STATE"}
    assert not reds, reds


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_the_census_runs_the_perception_tier_it_claims(census):
    absent = _absent_assets("models/")
    if absent:
        pytest.skip(f"perception tier not checked, absent on disk: {absent}")
    assert census["tier"] == "eprop-snn", census["tier"]


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_every_allowlisted_asset_is_still_read(census):
    """The closed list stays closed: an entry no Brain opens any more is removed, not kept. An
    asset absent on disk (the gitignored `models/` on a CI checkout) is skipped, by name."""
    absent = _absent_assets()
    stale = _stale(census["opened"], absent)
    assert not stale, stale
    if absent:
        pytest.skip(f"staleness not checked, absent on disk: {absent}")


def test_a_checkout_without_the_encoders_skips_only_their_staleness_check():
    models = [k for k in ASSETS if k.startswith("models/")]
    opened = {k: False for k in ASSETS if k not in models}
    assert _stale(opened, absent=models) == []
    assert _stale(opened, absent=[]) == models


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_the_census_reds_a_planted_repo_state_read_and_writes(tmp_path):
    out = str(tmp_path)
    planted = _census(out)
    kinds = _classify(planted["opened"])
    write = ("STATE", "a write outside the home")
    assert kinds.get("data/selfbuild_state.json") == ("STATE", "not on the asset allowlist"), kinds
    written = "a b src r rm rd md tr ld sd cd kd td rt wd wr p.db c.db d.db e.db f.db".split()
    assert {n: kinds.get(f"models/_t120_absent/{n}") for n in written} == dict.fromkeys(
        written, write
    )
    sources = "ls ss cs ks ts ws".split()  # a link's or a copy's source is not written
    for source in sources:
        assert kinds.get(f"models/_t120_absent/{source}") != write, kinds
    # a read-only connect is a read: off the list, but not a write
    for read in ("models/_t120_absent.db", "models/_t120_absent/g.db"):
        assert kinds.get(read) == ("STATE", "not on the asset allowlist"), kinds
    assert not {":memory:", "x.db", ".", ""} & set(kinds), kinds
    assert [v for k, v in kinds.items() if "[" in k] == [write], kinds  # the unparsable URI
    assert not [k for k in kinds if k.endswith(("_t120_fd", "_t120_fd_rm"))], kinds
    assert kinds.get("<live>/_t120_absent/r") == ("STATE", "the live home"), kinds
    assert kinds.get("<live>/_t120_absent/a") == ("STATE", "the live home"), kinds
    left = "<out>" + os.path.normcase(os.path.join(out, "left")).replace(os.sep, "/")
    assert kinds.get(left) == write, kinds
    assert [k for k in planted["scratch"] if k.endswith("/_t120_absent/a")], planted["scratch"]


@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_the_planted_probes_refuse_to_run_over_an_existing_directory(tmp_path):
    """The probes act on `<root>/models/_t120_absent`; if it exists, the child stops before any."""
    keep = tmp_path / "models" / "_t120_absent" / "rt" / "keep"
    keep.parent.mkdir(parents=True)
    keep.write_text("x", encoding="ascii")
    p = _child(str(tmp_path), str(tmp_path / "live"), str(tmp_path / "out"))
    assert p.returncode != 0 and "AssertionError" in p.stderr, p.stderr[-2000:]
    assert "CENSUS" not in p.stdout and keep.exists()


def test_the_classifier_rules():
    assert _classify(
        {
            "src/snn_data_lake/x.py": False,
            "src/snn_data_lake/x.json": False,
            "src/snn_data_lake/y.py": True,
            "src/snn_data_lake/__pycache__/y.cpython-314.pyc": True,
            "src/snn_data_lake/__pycache__/y.cpython-314.pyc.2405": True,
            "src/snn_data_lake/__pycache__/learned.json": True,
            "src/snn_data_lake/__pycache__": True,
            "src/snn_data_lake/__pycache__/y.cpython-314.pyc.tmp": True,
            "<out>c:/venv/lib/__pycache__/z.cpython-314.pyc": True,
            "<out>c:/temp/x.bin": True,
            "<live>/brain_lake.bin": False,
            "models/eprop_shd.pt": False,
            "models/learned.pt": False,
            "tools/lake_append_baseline.json": False,
            "tools/lake_append_baseline.json.bak": False,
        }
    ) == {
        "src/snn_data_lake/x.py": ("CODE", None),
        "src/snn_data_lake/x.json": ("STATE", "not on the asset allowlist"),
        "src/snn_data_lake/y.py": ("STATE", "a write into src/"),
        "src/snn_data_lake/__pycache__/y.cpython-314.pyc": ("CODE", None),
        "src/snn_data_lake/__pycache__/y.cpython-314.pyc.2405": ("CODE", None),
        "src/snn_data_lake/__pycache__/learned.json": ("STATE", "a write outside the home"),
        "src/snn_data_lake/__pycache__": ("CODE", None),
        "src/snn_data_lake/__pycache__/y.cpython-314.pyc.tmp": (
            "STATE",
            "a write outside the home",
        ),
        "<out>c:/venv/lib/__pycache__/z.cpython-314.pyc": ("CODE", None),
        "<out>c:/temp/x.bin": ("STATE", "a write outside the home"),
        "<live>/brain_lake.bin": ("STATE", "the live home"),
        "models/eprop_shd.pt": ("ASSET", ASSETS["models/eprop_shd.pt"]),
        "models/learned.pt": ("STATE", "not on the asset allowlist"),
        "tools/lake_append_baseline.json": ("ASSET", ASSETS["tools/lake_append_baseline.json"]),
        "tools/lake_append_baseline.json.bak": ("STATE", "not on the asset allowlist"),
    }


def test_the_census_child_runs_without_the_battery_doc_audit_hook(monkeypatch, tmp_path):
    """Engine arc 3 battery 1 red: the doc-audit hook wrote <audit>/<pid>.jsonl from the census
    child. The child gets the hook's opt-out (empty), never the battery's audit directory."""
    monkeypatch.setenv("SNDL_DOC_AUDIT", str(tmp_path))
    monkeypatch.setenv("SNDL_DOC_AUDIT_ROOT", str(tmp_path))
    env = _child_env(str(tmp_path / "home"))
    assert env["SNDL_DOC_AUDIT"] == ""
    assert [k for k in env if k.startswith("SNDL_DOC_AUDIT")] == ["SNDL_DOC_AUDIT"]
    assert env["NORD_HOME"] == str(tmp_path / "home")
