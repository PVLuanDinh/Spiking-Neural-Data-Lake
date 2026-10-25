"""P4b: "A restart is not amnesia." The three live learned stores the prototype record left out --
the arm's Q-table over its context-unit ids (`arm.q`), the affect integrator's emotion SOM
(`n_emotions`) and the lifelong consolidator's learner (`lifelong_units`) -- come back from the ONE
lake in a NEW process (policy audit pa-p4bd-3, 2026-10-24).

Same record as the percept prototypes (`growing_learners.json`, `_T_STATE`), same refusal
discipline: a record of the wrong shape or width is REFUSED, the store stays empty (never
half-restored) and the restore note says so. A non-persisting brain reads and writes no record.

Every home here is a fresh temp dir that passes `assert_not_live_home`; the live lake and the
operator home are never touched.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

# What both sides report about the three stores (`b`, `arm` bound by the script around it).
_SNAPSHOT = r"""
out = {"q": json.dumps({str(k): v for k, v in sorted(arm.q.items())}, sort_keys=True),
       "som": b._affect_int._som.state_dict()["prototypes"],
       "ll": b._ll.learner.state_dict()["prototypes"], "ll_labels": list(b._ll.labels),
       "ll_i": b._ll_i, "ll_mem": json.dumps(b._ll.memory.buf),
       "ll_rng": json.dumps(b._ll.memory.rng.getstate())}
"""

# The parent LEARNS each store through the store's own learning path, then close() flushes.
_PARENT = (
    r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=7, persist=True)
arm = b.brain.arm
n = 0
while (arm.learner.n_units < 2 or not arm.q) and n < 400:
    b.tick(); n += 1
for v, a in ((0.9, 0.9), (-0.9, 0.8), (0.0, -0.9), (-0.8, -0.8)):
    b._affect_int.observe(v, a)
b._ll.learn_task(0, b._ll_tasks[0]["train"]); b._ll_i = 1
"""
    + _SNAPSHOT
    + r"""
out["ticks"] = n
b.close()
print("PARENT " + json.dumps(out))
"""
)

_CHILD = (
    r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=7, persist=(sys.argv[2] == "1"))
arm = b.brain.arm
"""
    + _SNAPSHOT
    + r"""
s = b.state()
out.update(n_emotions=s["n_emotions"], lifelong_units=s["lifelong_units"],
           note=b._prototypes_restore_note)
b.close()
print("CHILD " + json.dumps(out))
"""
)

# Writes one hand-made `growing_learners.json` record (argv[2], JSON) into the home's lake.
_SEED = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import PROTOTYPES_STATE
from snn_data_lake.substrate.home import save_state
print("SEED " + json.dumps(bool(save_state(PROTOTYPES_STATE, json.loads(sys.argv[2])))))
"""


def _home(tag):
    from snn_data_lake.substrate.home import assert_not_live_home

    path = tempfile.mkdtemp(prefix=f"snn_p4b_{tag}_")
    assert_not_live_home(path)
    return path


def _run(code, home, tag, *args):
    env = dict(os.environ, PYTHONUTF8="1", NORD_HOME=home)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (os.path.join(os.getcwd(), "src"), env.get("PYTHONPATH", "")) if p
    )
    p = subprocess.run(
        [sys.executable, "-c", code, home, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    for line in p.stdout.splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1 :])
    raise AssertionError(f"{tag} produced no result (rc={p.returncode})\n{p.stderr[-2000:]}")


def _record_count(home):
    """`growing_learners.json` records, counted through the lake's OWN record parser."""
    from snn_data_lake.engine.brain import PROTOTYPES_STATE
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
        if name in (PROTOTYPES_STATE.encode(), PROTOTYPES_STATE):
            n += 1
    return n


def _learner(dim):
    from snn_data_lake.growth.growing_learner import GrowingLearner

    return {
        "fmt": GrowingLearner.STATE_FORMAT,
        "vigilance": 0.9,
        "lr": 0.2,
        "dim": dim,
        "prototypes": [[0.5] * dim],
    }


def _lifelong(learner, i):
    from snn_data_lake.sleep.lifelong_consolidation import LifelongConsolidator

    n = len(learner["prototypes"])
    return {
        "fmt": LifelongConsolidator.STATE_FORMAT,
        "learner": learner,
        "labels": [0] * n,
        "utility": [1.0] * n,
        "i": i,
    }


def _grown_consolidator():
    from snn_data_lake.sleep.lifelong_consolidation import D, LifelongConsolidator, _make_tasks

    c = LifelongConsolidator(dim=D, budget=6, seed=7)
    c.learn_task(0, _make_tasks(seed=7)[0]["train"])
    return c


def _grown_affect():
    from snn_data_lake.faculties.affect_integrator import AffectIntegrator

    a = AffectIntegrator(seed=7)
    a.observe(0.9, 0.9)
    return a


@pytest.fixture(scope="module")
def restarted():
    pytest.importorskip("torch")  # the brain's growth tier
    home = _home("restart")
    parent = _run(_PARENT, home, "PARENT")
    assert parent["q"] != "{}" and parent["som"] and parent["ll"], (
        f"the parent learned nothing to restore -- a comparison would be vacuous: {parent}"
    )
    return parent, _run(_CHILD, home, "CHILD", "1"), home


@pytest.mark.slow
def test_arm_q_table_survives_a_new_process(restarted):
    parent, child, _ = restarted
    assert child["q"] == parent["q"], "arm.q did not come back equal"


@pytest.mark.slow
def test_affect_som_survives_a_new_process(restarted):
    parent, child, _ = restarted
    assert child["som"] == parent["som"], "the affect integrator's SOM did not come back equal"


@pytest.mark.slow
def test_lifelong_learner_survives_a_new_process(restarted):
    parent, child, _ = restarted
    assert (child["ll"], child["ll_labels"], child["ll_i"]) == (
        parent["ll"],
        parent["ll_labels"],
        parent["ll_i"],
    ), "the lifelong consolidator's learner did not come back equal"


@pytest.mark.slow
def test_lifelong_replay_memory_is_rebuilt_in_a_new_process(restarted):
    """A-B1: without the replay memory the next task learns with no rehearsal and task 0 is
    pruned away. The memory is rebuilt from the tasks the restored index says were learned."""
    parent, child, _ = restarted
    assert parent["ll_mem"] != "[]", "the parent committed no exemplar -- a vacuous comparison"
    assert (child["ll_mem"], child["ll_rng"]) == (parent["ll_mem"], parent["ll_rng"])


@pytest.mark.slow
def test_consumers_read_the_restored_state_at_tick_0(restarted):
    parent, child, _ = restarted
    assert child["n_emotions"] == len(parent["som"]) > 0
    assert child["lifelong_units"] == len(parent["ll"]) > 0
    assert child["note"] is None, child["note"]


@pytest.mark.slow
def test_a_wrong_width_record_is_refused_and_the_store_stays_empty():
    pytest.importorskip("torch")
    from snn_data_lake.engine.autonomous_loop import ACTIONS, Q_STATE_FMT

    rec = {
        "arm": None,
        "vis": None,
        "aud": None,
        # a well-formed row for a unit the (empty) arm learner does not hold
        "q": {"fmt": Q_STATE_FMT, "rows": {"0": dict.fromkeys(ACTIONS, 1.0)}},
        "affect": _learner(3),  # the SOM's population code is wider than 3
        "lifelong": _lifelong(_learner(2), i=1),
    }
    home = _home("refuse")
    assert _run(_SEED, home, "SEED", json.dumps(rec))
    child = _run(_CHILD, home, "CHILD", "1")
    assert child["q"] == "{}" and child["som"] == [] and child["ll"] == [], child
    assert (child["ll_i"], child["n_emotions"], child["lifelong_units"]) == (0, 0, 0), child
    note = child["note"] or ""
    assert note.startswith("refused:") and all(k in note for k in ("q", "affect", "lifelong")), note


@pytest.mark.slow
def test_persist_false_reads_and_writes_no_record(restarted):
    """The determinism path: a non-persisting brain over a home that HOLDS a record restores
    nothing and writes nothing -- the same tick-0 state as over an empty home."""
    home = restarted[2]
    before = _record_count(home)
    over_record = _run(_CHILD, home, "CHILD", "0")
    over_empty = _run(_CHILD, _home("nopersist_empty"), "CHILD", "0")
    assert _record_count(home) == before, "a non-persisting brain wrote a record"
    assert over_record == over_empty, (over_record, over_empty)


# A parent whose ONLY learned state is one emotion: no tick, so no arm/percept/lifelong unit.
_PARENT_AFFECT = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=7, persist=True)
b._affect_int.observe(0.9, 0.9)
out = {"others": [b.brain.arm.learner.n_units, b._ll.n_units, b._vis_learner is None,
                  b._aud_learner is None],
       "som": b._affect_int._som.state_dict()["prototypes"]}
b.close()
print("PARENTA " + json.dumps(out))
"""


@pytest.mark.slow
def test_a_brain_that_grew_only_emotions_still_writes_and_restores_them():
    pytest.importorskip("torch")
    home = _home("affect_only")
    parent = _run(_PARENT_AFFECT, home, "PARENTA")
    assert parent["others"] == [0, 0, True, True] and parent["som"], parent
    assert _record_count(home) >= 1, "the flush gate dropped a brain that grew only emotions"
    child = _run(_CHILD, home, "CHILD", "1")
    assert child["som"] == parent["som"] and child["n_emotions"] == len(parent["som"]), child


@pytest.mark.slow
def test_one_refused_store_does_not_block_the_others():
    """Mixed record: the emotion map restores; the lifelong store (right width, task index out of
    range) is refused whole; the note names exactly that store."""
    pytest.importorskip("torch")
    affect = _grown_affect().state_dict()
    rec = {
        "arm": None,
        "vis": None,
        "aud": None,
        "affect": affect,
        "lifelong": dict(_grown_consolidator().state_dict(), i=-1),
    }
    home = _home("mixed")
    assert _run(_SEED, home, "SEED", json.dumps(rec))
    child = _run(_CHILD, home, "CHILD", "1")
    assert child["note"] == "refused: ['lifelong']", child["note"]
    assert child["som"] == affect["prototypes"] and child["n_emotions"] == len(affect["prototypes"])
    assert (child["ll"], child["ll_labels"], child["ll_i"], child["lifelong_units"]) == (
        [],
        [],
        0,
        0,
    )


# A restarted brain that learns in one store (`argv[2]`: "affect" or "lifelong") and flushes.
_RELEARN = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import make_brain
b = make_brain(seed=7, persist=True)
if sys.argv[2] == "affect":
    b._affect_int.observe(-0.9, -0.9)
else:
    b._ll.learn_task(0, b._ll_tasks[0]["train"]); b._ll_i = 1
b.close()
print("RELEARN " + json.dumps({"note": b._prototypes_restore_note}))
"""

# A new process runs the sleep's consolidation (superseded _T_STATE history is dropped), then reads.
_CONSOLIDATED = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.engine.brain import PROTOTYPES_STATE
from snn_data_lake.substrate.home import KnowledgeLog, knowledge_path, load_state
kl = KnowledgeLog(knowledge_path())
kl.consolidate(list(kl.replay()))
print("READ " + json.dumps(load_state(PROTOTYPES_STATE)))
"""


def _refused_lifelong_home(tag):
    rec = {
        "arm": None,
        "vis": None,
        "aud": None,
        "affect": _grown_affect().state_dict(),
        "lifelong": dict(_grown_consolidator().state_dict(), i=-1),  # refused at restore
    }
    home = _home(tag)
    assert _run(_SEED, home, "SEED", json.dumps(rec))
    return home, rec


@pytest.mark.slow
def test_a_refused_record_survives_the_next_flush_and_consolidation():
    """A-M1 / B MINOR-3: the refused lifelong record is carried forward unchanged while its store
    has learned nothing afresh -- a refusal is never a permanent loss at the first sleep."""
    pytest.importorskip("torch")
    home, rec = _refused_lifelong_home("carry")
    out = _run(_RELEARN, home, "RELEARN", "affect")
    assert out["note"] == "refused: ['lifelong']", out  # a restored store is never carried
    got = _run(_CONSOLIDATED, home, "READ")
    assert got["affect"] != rec["affect"], "the flush wrote nothing -- a vacuous comparison"
    assert got["lifelong"] == rec["lifelong"]
    assert _record_count(home) == 1


@pytest.mark.slow
def test_a_refused_record_is_superseded_only_once_its_store_relearns_and_says_so():
    pytest.importorskip("torch")
    home, rec = _refused_lifelong_home("supersede")
    out = _run(_RELEARN, home, "RELEARN", "lifelong")
    assert "superseded: ['lifelong']" in out["note"], out
    got = _run(_CONSOLIDATED, home, "READ")
    assert got["lifelong"]["i"] == 1 and got["lifelong"] != rec["lifelong"]


@pytest.mark.slow
def test_a_refused_affect_record_is_superseded_once_the_som_relearns():
    pytest.importorskip("torch")
    rec = {"arm": None, "vis": None, "aud": None, "affect": _learner(3)}  # wrong width: refused
    home = _home("supersede_affect")
    assert _run(_SEED, home, "SEED", json.dumps(rec))
    out = _run(_RELEARN, home, "RELEARN", "affect")
    assert out["note"] == "refused: ['affect']; superseded: ['affect']", out
    got = _run(_CONSOLIDATED, home, "READ")
    assert got["affect"] != rec["affect"] and got["affect"]["prototypes"]


@pytest.mark.slow
def test_records_a_raising_restore_never_reached_are_carried():
    """The arm's malformed width raises before any store is read: every sub-record is carried,
    including a lifelong record the restore would have accepted."""
    pytest.importorskip("torch")
    ll = dict(_grown_consolidator().state_dict(), i=1)
    rec = {"arm": {"dim": "x"}, "vis": None, "aud": None, "lifelong": ll}
    home = _home("raised")
    assert _run(_SEED, home, "SEED", json.dumps(rec))
    out = _run(_RELEARN, home, "RELEARN", "affect")
    assert out["note"].startswith("restore raised:"), out
    got = _run(_CONSOLIDATED, home, "READ")
    assert got["lifelong"] == ll and got["arm"] == {"dim": "x"}


@pytest.mark.slow
def test_a_non_dict_record_does_not_block_construction():
    pytest.importorskip("torch")
    home = _home("nondict")
    assert _run(_SEED, home, "SEED", json.dumps([1]))
    child = _run(_CHILD, home, "CHILD", "1")
    assert (child["note"] or "").startswith("restore raised:"), child


def test_affect_codec_round_trips_and_refuses_untouched():
    from snn_data_lake.faculties.affect_integrator import AffectIntegrator

    a = _grown_affect()
    d = a.state_dict()
    b = AffectIntegrator(seed=7)
    assert not b.load_state_dict("not a record") and b.n_emotions == 0
    assert not b.load_state_dict(dict(d, dim=3)) and b.n_emotions == 0
    assert b.load_state_dict(d) and b.signature() == a.signature()
    assert not b.load_state_dict(d), "an integrator that already grew is never overwritten"


def test_lifelong_codec_round_trips_and_refuses_untouched():
    from snn_data_lake.sleep.lifelong_consolidation import LifelongConsolidator

    c = _grown_consolidator()
    d = c.state_dict()
    n = c.n_units
    assert n > 0
    bad = [
        dict(d, fmt="other"),
        dict(d, labels=d["labels"][:-1]),
        dict(d, utility=[float("nan")] * n),
        {k: v for k, v in d.items() if k != "labels"},
        dict(d, learner=_learner(2), labels=[0], utility=[1.0]),  # aligned: only the width differs
        dict(d, labels=[float("inf")] * n),
        dict(d, learner=dict(d["learner"], fmt="other")),  # refused inside GrowingLearner
    ]
    for rec in bad:
        e = LifelongConsolidator(dim=c.dim, budget=6, seed=7)
        assert not e.load_state_dict(rec), rec
        assert (e.n_units, e.labels, e.utility) == (0, [], []), rec
    e = LifelongConsolidator(dim=c.dim, budget=6, seed=7)
    assert e.load_state_dict(d) and e.state_dict() == d
    assert not e.load_state_dict(d), "a consolidator that already grew is never overwritten"


def test_q_codec_round_trips_and_refuses_untouched():
    from snn_data_lake.engine.autonomous_loop import ACTIONS, AutonomousArm

    def arm_with_units():
        a = AutonomousArm(seed=0)
        dim = a.learner.dim
        for i in range(2):  # two orthogonal patterns -> two units
            a.learner.observe([1.0 if j == i else 0.0 for j in range(dim)])
        assert a.learner.n_units == 2
        return a

    src = arm_with_units()
    src.q = {u: {act: 0.1 * u + k for k, act in enumerate(ACTIONS)} for u in range(2)}
    d = src.q_state_dict()
    row = d["rows"]["0"]
    bad = [
        dict(d, fmt="other"),
        dict(d, rows={"0": {k: v for k, v in row.items() if k != ACTIONS[0]}}),
        dict(d, rows={"0": dict(row, EXTRA=0.0)}),
        dict(d, rows={"0": dict(row, **{ACTIONS[0]: float("nan")})}),
        dict(d, rows={"x": row}),
        dict(d, rows={"2": row}),  # a unit the learner does not hold
        dict(d, rows=[row]),
    ]
    for rec in bad:
        a = arm_with_units()
        assert not a.load_q_state_dict(rec) and a.q == {}, rec
    a = arm_with_units()
    assert a.load_q_state_dict(d) and a.q == src.q
    assert not a.load_q_state_dict(d), "a table that already holds rows is never overwritten"
