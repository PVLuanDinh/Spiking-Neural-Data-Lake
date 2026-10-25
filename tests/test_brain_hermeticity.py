"""T120 rung 3: the hermeticity census. A temp-home Brain keeps every learned byte in its home.

One fresh process builds one Brain on a temp NORD_HOME under a `sys.addaudithook` recorder of every
`open` (builtin `open`, `io.open_code`, `os.open`) and `sqlite3.connect` event, installed before the
package is imported. Each file under the REPO that the process opens is classified:

  * CODE -- the package's own source and bytecode under `src/` (and Python's `__pycache__` writes);
  * ASSET -- a read-only input on the closed `ASSETS` list below, each with its reason;
  * anything else is LEARNED STATE outside the home, and reds, as does any write outside CODE.

A test, not a CI gate (P4c: the census output is counted before any rule lands). The planted red
re-adds the pre-T120 repo read of the self-build learner, plus a write through `open`, `os.open` and
`sqlite3.connect`, and a read-only connect. Blind spot: a native library's own file I/O (h5py, for
one) raises no audit event.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Closed allowlist: repo-relative path (or a prefix ending "/") -> why the Brain may read it.
ASSETS = {
    "models/": "the frozen perception encoders and packs (gitignored; trained offline, never "
    "written by a Brain)",
    "tools/lake_append_baseline.json": "the committed per-tag append budget "
    "`substrate/lake_engine` checks appends against (a host measurement, not learning)",
    "graphify-out/graph.json": "the generated import graph "
    "`connectome._import_graph_is_untyped` reads as a fair control (gitignored, rebuilt by "
    "graphify, never written by a Brain)",
}

_CHILD = r"""
import json, os, sys

home, root, plant = sys.argv[1], os.path.normcase(os.path.abspath(sys.argv[2])), sys.argv[3]
seen = {}
_W = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND


def note(path, wrote):
    p = os.path.normcase(os.path.abspath(path))
    if p.startswith(root + os.sep):
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        seen[rel] = seen.get(rel, False) or wrote


def hook(event, args):
    if event not in ("open", "sqlite3.connect") or not args:
        return
    if not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    if event == "sqlite3.connect":
        query = ""
        if path.startswith("file:"):
            path, _, query = path[len("file:") :].partition("?")
        note(path, "mode=ro" not in query)
        return
    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    note(path, any(c in mode for c in "wax+") or bool(flags & _W))


sys.addaudithook(hook)
os.environ["NORD_HOME"] = home
from snn_data_lake.substrate.home import assert_not_live_home, home as _home

assert_not_live_home(_home())
if plant == "plant":
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
    import sqlite3

    for write in (lambda: open(os.path.join(absent, "a"), "w"),
                  lambda: os.open(os.path.join(absent, "b"), os.O_WRONLY | os.O_CREAT),
                  lambda: sqlite3.connect(f"file:{absent}/c.db?mode=rwc", uri=True),
                  lambda: sqlite3.connect(f"file:{root}/models/_t120_absent.db?mode=ro", uri=True)):
        try:
            write()
        except (OSError, sqlite3.Error):
            pass
from snn_data_lake.engine.brain import make_brain

b = make_brain(seed=0, persist=True)
for _ in range(20):
    b.tick()
b.propose_improvements(budget=1)
s = b.state()
b.close()
print("CENSUS " + json.dumps({"opened": seen, "tier": s.get("perception_tier"),
                              "audio": s.get("perceiving_audio")}))
"""


def _census(plant=""):
    home = tempfile.mkdtemp(prefix="snn_t120_census_")
    from snn_data_lake.substrate.home import assert_not_live_home

    assert_not_live_home(home)
    env = dict(os.environ, PYTHONUTF8="1", NORD_HOME=home)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (os.path.join(_REPO, "src"), env.get("PYTHONPATH", "")) if p
    )
    p = subprocess.run(
        [sys.executable, "-c", _CHILD, home, _REPO, plant],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=_REPO,
        timeout=900,
    )
    for line in p.stdout.splitlines():
        if line.startswith("CENSUS "):
            return json.loads(line[len("CENSUS ") :])
    raise AssertionError(f"no census (rc={p.returncode})\n{p.stderr[-2000:]}")


def _matches(key, rel):
    return rel == key or (key.endswith("/") and rel.startswith(key))


def _classify(opened):
    """{rel: (kind, why)} for every repo file the Brain opened."""
    out = {}
    for rel, wrote in sorted(opened.items()):
        pyc = "/__pycache__/" in rel
        if rel.startswith("src/") and (rel.endswith((".py", ".pyc", ".pyd", ".so")) or pyc):
            out[rel] = ("CODE", None) if (not wrote or pyc) else ("STATE", "a write into src/")
            continue
        why = next((w for k, w in ASSETS.items() if _matches(k, rel)), None)
        if why and not wrote:
            out[rel] = ("ASSET", why)
        else:
            out[rel] = (
                "STATE",
                "a write outside the home" if wrote else "not on the asset allowlist",
            )
    return out


@pytest.fixture(scope="module")
def census():
    return _census()


def test_a_temp_home_brain_opens_no_learned_state_under_the_repo(census):
    kinds = _classify(census["opened"])
    print(json.dumps({"tier": census["tier"], "audio": census["audio"], "files": kinds}, indent=1))
    reds = {k: v[1] for k, v in kinds.items() if v[0] == "STATE"}
    assert not reds, reds


def test_every_allowlisted_asset_is_still_read(census):
    """The closed list stays closed: an entry no Brain opens any more is removed, not kept."""
    opened = census["opened"]
    stale = [k for k in ASSETS if not any(_matches(k, r) for r in opened)]
    assert not stale, stale


def test_the_census_reds_a_planted_repo_state_read_and_writes():
    kinds = _classify(_census("plant")["opened"])
    assert kinds.get("data/selfbuild_state.json") == ("STATE", "not on the asset allowlist"), kinds
    for name in ("a", "b", "c.db"):
        assert kinds.get(f"models/_t120_absent/{name}") == ("STATE", "a write outside the home")
    assert kinds.get("models/_t120_absent.db") == (
        "ASSET",
        ASSETS["models/"],
    )  # a read-only connect


def test_the_classifier_rules():
    assert _classify(
        {
            "src/snn_data_lake/x.py": False,
            "src/snn_data_lake/x.json": False,
            "src/snn_data_lake/y.py": True,
            "src/snn_data_lake/__pycache__/y.cpython-314.pyc": True,
            "models/eprop_shd.pt": False,
            "tools/lake_append_baseline.json": False,
            "tools/lake_append_baseline.json.bak": False,
        }
    ) == {
        "src/snn_data_lake/x.py": ("CODE", None),
        "src/snn_data_lake/x.json": ("STATE", "not on the asset allowlist"),
        "src/snn_data_lake/y.py": ("STATE", "a write into src/"),
        "src/snn_data_lake/__pycache__/y.cpython-314.pyc": ("CODE", None),
        "models/eprop_shd.pt": ("ASSET", ASSETS["models/"]),
        "tools/lake_append_baseline.json": ("ASSET", ASSETS["tools/lake_append_baseline.json"]),
        "tools/lake_append_baseline.json.bak": ("STATE", "not on the asset allowlist"),
    }
