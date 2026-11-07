"""P4b: a restart of the lifelong consolidator is not amnesia (arc-2 panel A-B1 / B MAJOR-3).

The restore brings back the learner, labels, utilities and task index, and the replay memory is
rebuilt from the task streams it was filled from. The probe is rv-arc2-a's `ll_restart.py`: learn
task 0, round-trip the state through JSON (the lake), continue in a fresh consolidator. Without the
replay memory the old units decay below the budget and are pruned (task-0 retention 0.000 on seeds
0-4); with it the restarted run is the uninterrupted run.
"""

import json

import pytest

from snn_data_lake.sleep.lifelong_consolidation import D, LifelongConsolidator, _make_tasks

SEEDS = range(5)


def _restarted(seed, tasks, rebuild):
    b = LifelongConsolidator(dim=D, budget=6, seed=seed)  # the Brain's knobs
    b.learn_task(0, tasks[0]["train"])
    c = LifelongConsolidator(dim=D, budget=6, seed=seed)  # the restarted process
    assert c.load_state_dict(json.loads(json.dumps(b.state_dict())))
    if rebuild:
        c.rebuild_memory([t["train"] for t in tasks[:1]])
    for i in range(1, len(tasks)):
        c.learn_task(i, tasks[i]["train"])
    return c


def _uninterrupted(seed, tasks):
    a = LifelongConsolidator(dim=D, budget=6, seed=seed)
    for i, t in enumerate(tasks):
        a.learn_task(i, t["train"])
    return a


@pytest.mark.parametrize("seed", SEEDS)
def test_the_probe_sees_the_amnesia_without_the_replay_memory(seed):
    """Known positive for the instrument: a restart with an empty replay memory loses task 0."""
    tasks = _make_tasks(seed=seed)
    a, c = _uninterrupted(seed, tasks), _restarted(seed, tasks, rebuild=False)
    assert a.accuracy(tasks[0]["test"]) > c.accuracy(tasks[0]["test"])


@pytest.mark.parametrize("seed", SEEDS)
def test_a_restart_with_the_rebuilt_memory_is_the_uninterrupted_run(seed):
    tasks = _make_tasks(seed=seed)
    a, c = _uninterrupted(seed, tasks), _restarted(seed, tasks, rebuild=True)
    assert c.accuracy(tasks[0]["test"]) == a.accuracy(tasks[0]["test"])
    assert c.state_dict() == a.state_dict()
    assert c.memory.buf == a.memory.buf and c.memory.rng.getstate() == a.memory.rng.getstate()


def test_the_rebuilt_memory_is_the_one_learn_task_built():
    tasks = _make_tasks(seed=3)
    a = LifelongConsolidator(dim=D, budget=6, seed=3)
    for i in range(2):
        a.learn_task(i, tasks[i]["train"])
    r = LifelongConsolidator(dim=D, budget=6, seed=3)
    r.rebuild_memory([t["train"] for t in tasks[:2]])
    assert r.memory.buf == a.memory.buf and r.memory.rng.getstate() == a.memory.rng.getstate()
    assert r.n_units == 0, "rebuilding the memory learns nothing"


# ---- D7 commit 1 (D-365): P-LL's utility moves onto the ONE LeakySurvival primitive (A1-A8) ----
# Byte-identity pins below were captured on BASE 2f5a9c22 (before the port) by a throwaway
# script over the same calls; the refactor must reproduce them exactly. Each pin carries its own
# red: a planted change of the arithmetic (task_decay 0.15 -> 0.16) must move it, so a pin that
# cannot fail is caught here.
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

PLL_PIN = {
    "magnitude": "8afb5d68a255f38ebc82f848cb2219a0f9aef3a03347611fd155dad691e9321b",
    "random": "6f7c94c0336bef258fd8215a60b16bb7c31effa31142a5799bd813606d5978eb",
}
PLL_DECAY016_PIN = "0a6119340965bac69e38fad4ecf86beb3fb1bfa7ae81cdf794af095b38c9e2c1"
PLL_STATE_JSON_SEED0 = "badb52a750497ba56d5f99fc2b85ab2562583c7a17801cb6f2e057892831f884"
WHOLE_PIN = {
    0: "3b015557b0da4f734a599cbb3a15fefe9493bcc2e7c75d1b48163667322dae2a",
    1: "ed0108cbc74d65dd212d458a70de86d27ca3a54651fc043b9eb7fe1c4cc00c1c",
}


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _pll_record(seed, mode, decay=0.15):
    tasks = _make_tasks(seed=seed)
    c = LifelongConsolidator(dim=D, budget=6, seed=seed, prune_mode=mode, task_decay=decay)
    accs = []
    for i, t in enumerate(tasks):
        c.learn_task(i, t["train"])
        accs.append([c.accuracy(x["test"]) for x in tasks[: i + 1]])
    rank = sorted(range(c.n_units), key=lambda i: (c.utility[i], i))
    return {"state": c.state_dict(), "accs": accs, "rank": rank}


@pytest.mark.parametrize("mode", ["magnitude", "random"])
def test_pll_ranking_and_record_byte_identical_after_port(mode):
    """PREREG test 9: P-LL's ranking, accuracies and state record over 5 seeds equal the base
    capture."""
    assert _sha([_pll_record(s, mode) for s in range(5)]) == PLL_PIN[mode]


def test_pll_pin_red_a_planted_decay_change_moves_it():
    """The pin's own red: the same run with task_decay 0.16 must NOT reproduce the 0.15 pin."""
    got = _sha([_pll_record(s, "magnitude", 0.16) for s in range(5)])
    assert got == PLL_DECAY016_PIN and got != PLL_PIN["magnitude"]


def test_pll_lake_record_bytes_identical_and_pre_d7_record_restores():
    """A3.C3: the record P-LL writes (state_dict -> JSON) is byte-identical to the base build's,
    so the base (pre-D7) bytes ARE this record; restoring them gives the same record and the same
    ranking."""
    rec = _pll_record(0, "magnitude")
    raw = json.dumps(rec["state"])
    assert hashlib.sha256(raw.encode()).hexdigest() == PLL_STATE_JSON_SEED0
    c = LifelongConsolidator(dim=D, budget=6, seed=0)
    assert c.load_state_dict(json.loads(raw))
    assert json.dumps(c.state_dict()) == raw
    assert sorted(range(c.n_units), key=lambda i: (c.utility[i], i)) == rec["rank"]
    bad = json.loads(raw)
    bad["utility"][0] = float("nan")
    assert not LifelongConsolidator(dim=D, budget=6, seed=0).load_state_dict(bad), (
        "finiteness refusal kept"
    )


def _whole_digest(seed):
    from snn_data_lake.faculties.wholistic_loop import WholisticLoop

    # exactly the two runs Brain birth takes (engine/brain.py:547-556), here OUTSIDE its
    # swallowing try
    w = WholisticLoop(hard=True, seed=seed).run()
    s = WholisticLoop(
        hard=True, metacog=False, consolidation=False, criticality=True, seed=seed
    ).run()
    return hashlib.sha256(
        (repr(sorted(w.items())) + "|" + repr(sorted(s.items()))).encode()
    ).hexdigest()


@pytest.mark.parametrize("seed", [0, 1])
def test_wholistic_loop_byte_identical_after_utility_port(seed):
    """PREREG test 18 (A3.B1, A5): both whole result dicts Brain birth stores are unchanged by
    the port."""
    assert _whole_digest(seed) == WHOLE_PIN[seed]


def test_utility_in_place_write_raises():
    """PREREG test 19 (A5.B1a): `utility` is an immutable view; in-place and rebinding writes
    raise."""
    tasks = _make_tasks(seed=0)
    c = LifelongConsolidator(dim=D, budget=6, seed=0)
    c.learn_task(0, tasks[0]["train"])
    before = c.state_dict()
    with pytest.raises(AttributeError):
        c.utility.append(1.0)
    with pytest.raises(TypeError):
        c.utility[0] += 1.0
    with pytest.raises(AttributeError):
        c.utility = []
    assert c.state_dict() == before, "a refused write must change nothing"


def test_leaky_survival_operations():
    """The primitive's four operations, each the arithmetic P-LL ran inline (append 1.0, += 1.0,
    * decay, reindex by keep), and the finiteness refusal on load."""
    from snn_data_lake.sleep.lifelong_consolidation import LeakySurvival

    s = LeakySurvival()
    s.add_unit(1.0)
    s.add_unit(1.0)
    s.credit(1, 1.0)
    s.decay(0.15)
    assert s.values() == (1.0 * 0.15, 2.0 * 0.15)
    s.reindex([1])
    assert s.values() == (2.0 * 0.15,) and len(s) == 1
    with pytest.raises(ValueError):
        LeakySurvival([1.0, float("inf")])


_SURVIVOR = re.compile(
    r"\.utility\s*=[^=]|\.utility\[[^\]]*\]\s*[-+*/]?=[^=]|\.utility\.(append|extend|insert|pop|remove|clear)\b"
)
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "snn_data_lake"
)


def _survivor_hits():
    hits = []
    for root, _dirs, files in os.walk(_SRC):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                with open(path, encoding="utf-8") as fh:
                    for no, line in enumerate(fh, 1):
                        if _SURVIVOR.search(line):
                            hits.append(f"{os.path.relpath(path, _SRC)}:{no}")
    return hits


def test_survivor_pattern_fires_on_known_positives():
    """A5.B1b: the detector's known positives (the base's writer shapes) and its known negatives."""
    for line in (
        "self.con.utility = [u * TASK_DECAY for u in self.con.utility]",
        "self.utility.append(1.0)",
        "self.utility[uid] += 1.0",
        "self.utility = [self.utility[i] for i in keep]",
    ):
        assert _SURVIVOR.search(line), line
    for line in (
        "self._utility = utility",
        "self._utility.decay(0.15)",
        "x == c.utility",
        "def utility(self):",
    ):
        assert not _SURVIVOR.search(line), line


def test_no_utility_writer_survives_outside_the_primitive():
    """A5.B1b / A6 (P2 count the survivors): no write to `.utility` anywhere under src after the
    port."""
    assert _survivor_hits() == []


def _birth_competence(monkeypatch, home):
    from snn_data_lake.engine.brain import make_brain

    home.mkdir()
    monkeypatch.setenv("NORD_HOME", str(home))  # a throwaway home, never the live lake
    return make_brain(seed=7)._integration_competence


def test_birth_constructs_wholistic_loop_on_leaky_survival(monkeypatch, tmp_path):
    """PREREG test 20 (A9, replacing A7's torch-purity check): Brain birth runs WholisticLoop
    inside a swallowing try (engine/brain.py integration block, 0.0 on ANY failure), so the
    faculty must be shown PRESENT at birth and built on the one LeakySurvival primitive; a
    planted break first hit inside the loop must flip the birth reading to the swallowed 0.0,
    and a broken constructor must fail birth loudly (A10)."""
    from snn_data_lake.faculties.wholistic_loop import WholisticLoop
    from snn_data_lake.sleep.lifelong_consolidation import LeakySurvival

    assert _birth_competence(monkeypatch, tmp_path / "h1") > 0.0  # 1: the birth block ran the loop
    assert type(WholisticLoop(hard=True, seed=7).con._utility) is LeakySurvival  # 2

    def _broken(self, *args, **kwargs):
        raise ImportError("planted: LeakySurvival broken")

    with monkeypatch.context() as m:  # 3a (A10): a break first hit INSIDE the loop is swallowed
        m.setattr(LeakySurvival, "decay", _broken)
        assert _birth_competence(m, tmp_path / "h2") == 0.0
    with monkeypatch.context() as m:  # 3b (A10): a broken constructor fails birth LOUDLY (:475)
        m.setattr(LeakySurvival, "__init__", _broken)
        with pytest.raises(ImportError):
            _birth_competence(m, tmp_path / "h3")
