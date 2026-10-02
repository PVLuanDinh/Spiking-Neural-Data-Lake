"""The brain's PROTOTYPES of what it perceived must survive a restart -- proven across a real
process boundary (specs/022 audit s8 rank 2b, 2026-09-02).

`growth/growing_learner.GrowingLearner` had no persistence path anywhere in the tree: the
arm loop's context prototypes and the vision/audio prototype learners re-grew from zero on every
daemon relaunch. Since rank 2a the neurogenesis grower's categories ARE the arm learner's prototype
ids, so a re-grown learner re-keys the vocabulary the persisted NG-SNN readout was fit on.

WHY A NEW PROCESS AND NOT A NEW OBJECT. P4b: *"it persists" is a claim about a NEW process, not
about the object in front of you*. This spawns a real interpreter over the same NORD_HOME.

THE CONTROLS ARE THE TEST. Five things together:
  * the parent actually recruited (>= 2 arm prototypes AND >= 1 vision + 1 audio prototype --
    the real perception packs must be armed, which they are from the repo root; a 0-prototype
    learner would make a byte-equal comparison vacuous);
  * the lake actually holds a `growing_learners.json` named-state record (counted by BinLog
    replay through the lake's own record parser, never by the writer's own state);
  * a child over an EMPTY home does NOT come back with prototypes -- the negative control that
    separates "restored" from "any brain looks like this";
  * a parent KILLED without close() (os._exit after a night's flush) still comes back with what
    the night flushed -- the sleep-tick caller has its own red, not only close();
  * a record of the WRONG width is REFUSED whole (no half-restore).

Record: specs/022-unified-brain-arc/evidence/s2b-growing-learner-lake-record.md.
"""

import json
import os
import subprocess
import sys
import tempfile
import zlib

import pytest

pytestmark = pytest.mark.slow

_PARENT = r"""
import json, os, sys, zlib
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=7, persist=True)
arm = b.brain.arm.learner
n = 0
while arm.n_units < 2 and n < 400:
    b.tick(); n += 1
def crc(l):
    if l is None:
        return None
    return zlib.crc32(json.dumps(l.state_dict()["prototypes"]).encode())
vis, aud = b._vis_learner, b._aud_learner
out = {"ticks": n, "arm_units": arm.n_units, "arm_crc": crc(arm),
       "vis_units": (vis.n_units if vis else 0), "vis_crc": crc(vis),
       "aud_units": (aud.n_units if aud else 0), "aud_crc": crc(aud)}
b.close()
print("PARENT " + json.dumps(out))
"""

_PARENT_KILLED = r"""
import json, os, sys, zlib
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=7, persist=True)
arm = b.brain.arm.learner
flushed = {"crc": None, "n": 0}
real = b._flush_prototypes
def spy():
    ok = real()
    if ok:
        flushed["crc"] = zlib.crc32(json.dumps(arm.state_dict()["prototypes"]).encode())
        flushed["n"] = arm.n_units
    return ok
b._flush_prototypes = spy  # the night's flush, observed; close() never runs below
t = 0
while flushed["n"] < 2 and t < 900:
    b.tick(); t += 1
out = {"ticks": t, "flushed_units": flushed["n"], "flushed_crc": flushed["crc"],
       "arm_units_now": arm.n_units}
print("PARENTK " + json.dumps(out)); sys.stdout.flush()
os._exit(0)  # a KILL: no close(), no atexit, nothing flushes after this line
"""

_CHILD = r"""
import json, os, sys, zlib
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=7, persist=True)
arm = b.brain.arm.learner
def crc(l):
    if l is None:
        return None
    return zlib.crc32(json.dumps(l.state_dict()["prototypes"]).encode())
vis, aud = b._vis_learner, b._aud_learner
out = {"arm_units": arm.n_units, "arm_crc": crc(arm),
       "vis_units": (vis.n_units if vis else 0), "vis_crc": crc(vis),
       "aud_units": (aud.n_units if aud else 0), "aud_crc": crc(aud),
       "real_units": int(b.state()["real_perception_units"])}
b.close()
print("CHILD " + json.dumps(out))
"""


def _run(code, home, tag):
    env = dict(os.environ, PYTHONUTF8="1", NORD_HOME=home)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (os.path.join(os.getcwd(), "src"), env.get("PYTHONPATH", "")) if p
    )
    p = subprocess.run(
        [sys.executable, "-c", code, home], capture_output=True, text=True, env=env, timeout=900
    )
    for line in p.stdout.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1 :])
    raise AssertionError(f"{tag} produced no result (rc={p.returncode})\n{p.stderr[-2000:]}")


def _record_count(home):
    """Named-state records for the learners, counted through the lake's OWN record parser."""
    from snn_data_lake.substrate.home import _T_STATE, BinLog, _unpack_state

    path = os.path.join(home, "brain_lake.bin")
    if not os.path.exists(path):
        return 0
    n = 0
    for payload in BinLog(path).replay():
        if payload[:1] != _T_STATE:
            continue
        try:
            name = _unpack_state(payload)[0]
        except Exception:
            continue
        if name in (b"growing_learners.json", "growing_learners.json"):
            n += 1
    return n


def test_prototypes_survive_a_new_process_and_an_empty_home_does_not():
    torch = pytest.importorskip("torch")  # noqa: F841  (the brain's growth tier)
    home = tempfile.mkdtemp(prefix="snn_proto_restart_")
    parent = _run(_PARENT, home, "PARENT")
    assert parent["arm_units"] >= 2, f"the parent recruited nothing to restore: {parent}"
    assert parent["vis_units"] >= 1 and parent["aud_units"] >= 1, (
        "the real perception packs are not armed from this cwd (models/nmnist_pack.npz, "
        f"models/shd_pack.npz) -- a 0-prototype comparison would be vacuous: {parent}"
    )
    assert _record_count(home) >= 1, "the lake holds no growing_learners.json named-state record"
    child = _run(_CHILD, home, "CHILD")
    assert child["arm_units"] == parent["arm_units"] and child["arm_crc"] == parent["arm_crc"], (
        f"the arm learner did not come back byte-equal: parent {parent} child {child}"
    )
    for k in ("vis", "aud"):
        assert (
            child[f"{k}_units"] == parent[f"{k}_units"] and child[f"{k}_crc"] == parent[f"{k}_crc"]
        ), f"{k} learner differs: parent {parent} child {child}"
    assert child["real_units"] == parent["vis_units"] + parent["aud_units"], (
        "real_perception_units at tick 0 must be what the lake holds"
    )
    # NEGATIVE CONTROL: an empty home restores nothing
    empty = tempfile.mkdtemp(prefix="snn_proto_empty_")
    ctrl = _run(_CHILD, empty, "CHILD")
    assert ctrl["arm_units"] == 0 and ctrl["vis_units"] == 0 and ctrl["aud_units"] == 0, ctrl


def test_a_killed_parent_keeps_what_the_night_flushed():
    """The sleep-tick caller's own red: a parent that dies WITHOUT close() (os._exit) after a
    night's flush comes back with exactly what that night wrote -- the daemon's crash/kill
    relaunch, not only its clean commit-relaunch, is covered."""
    pytest.importorskip("torch")
    home = tempfile.mkdtemp(prefix="snn_proto_killed_")
    parent = _run(_PARENT_KILLED, home, "PARENTK")
    assert parent["flushed_units"] >= 2 and parent["flushed_crc"] is not None, (
        f"no night flushed >= 2 prototypes within the budget: {parent}"
    )
    child = _run(_CHILD, home, "CHILD")
    assert child["arm_units"] == parent["flushed_units"], f"parent {parent} child {child}"
    assert child["arm_crc"] == parent["flushed_crc"], (
        f"the child must hold the LAST FLUSHED prototypes byte-equal: {parent} vs {child}"
    )


def test_a_wrong_width_record_is_refused_whole():
    from snn_data_lake.growth.growing_learner import GrowingLearner

    g = GrowingLearner(dim=4)
    good = {
        "fmt": g.STATE_FORMAT,
        "vigilance": 0.85,
        "lr": 0.2,
        "dim": 4,
        "prototypes": [[0.1] * 4],
    }
    bad_width = dict(good, prototypes=[[0.1] * 4, [0.2] * 3])
    assert not g.load_state_dict(bad_width) and g.n_units == 0
    assert not g.load_state_dict(dict(good, fmt="other")) and g.n_units == 0
    assert not g.load_state_dict(dict(good, prototypes=[[float("nan")] * 4])) and g.n_units == 0
    assert not g.load_state_dict(dict(good, lr=2.0)) and g.n_units == 0
    assert g.load_state_dict(good) and g.n_units == 1 and g.dim == 4
    assert (g.vigilance, g.lr) == (0.85, 0.2), "every record field is adopted"
    # a learner that already recruited is never overwritten (its ids are in use)
    assert not g.load_state_dict(good) and g.n_units == 1
    assert g.state_dict()["prototypes"] == [[0.1] * 4]
    # the record's dim wins for an EMPTY learner, then grow_dim still widens it
    h = GrowingLearner(dim=2)
    assert h.load_state_dict(good) and h.dim == 4
    h.grow_dim(6)
    assert h.dim == 6 and len(h.prototypes[0]) == 6
    zlib.crc32(json.dumps(h.state_dict()).encode())  # JSON-able end to end


def test_a_width_mismatched_record_never_reaches_a_wake_tick():
    """The panel's hazard: an arm record of another width restored into the arm would raise out
    of EVERY wake tick; a wider vision record would be fed forever and learn nothing. Both are
    now refused or dropped, COUNTED and NAMED on state(), and the brain ticks."""
    pytest.importorskip("torch")
    from snn_data_lake.growth.growing_learner import GrowingLearner
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        from snn_data_lake.engine.brain import PROTOTYPES_STATE, make_brain
        from snn_data_lake.substrate.home import save_state

        narrow = GrowingLearner(dim=32)
        narrow.observe([0.3] * 32)
        wide = GrowingLearner(dim=512)
        wide.observe([0.2] * 512)
        save_state(PROTOTYPES_STATE, {"arm": narrow.state_dict(), "vis": wide.state_dict()})
        b = make_brain(seed=0, persist=True)
        st = b.state()
        assert "refused" in str(st["prototypes_restore"]) and "arm" in str(st["prototypes_restore"])
        assert b.brain.arm.learner.n_units == 0, "a width-mismatched arm record must be refused"
        modes = [b.tick() for _ in range(40)]
        assert modes.count("awake") > 0
        st = b.state()
        if b._perceiving_real:  # the wide vision record is dropped at the first real percept
            assert st["prototype_faults"] >= 1 and "dropped" in str(st["prototypes_restore"])
            assert b._vis_learner is not None and b._vis_learner.dim != 512
        assert b.brain.arm.learner.n_units >= 1, "the arm learner recruits normally afterwards"
