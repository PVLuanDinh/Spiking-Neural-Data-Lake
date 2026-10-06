r"""G2 / T110's taught campaign (P3): the canonical incumbent through G0's channel, the presized nonlinear control,
the explicit-history control, the arms, checks B3-B8, the G-R2 reading and the campaign driver.

Pre-registration: specs/022-unified-brain-arc/evidence/g2-taught-problems.md (frozen at 5e3c2f6e; 2.4, 2.5,
2.8, 2.9, 2.11-2.13, 2.15). The generators are `delayed_cue` (P1), the contract `taught_problems` (P2).

The channel (2.5): probe and context turns go through the READ (`route_text`, pure); teaching goes through the
correction (`learn_turn(text, None, feedback=label)` == `LiveBrain.correct`). Context turns are read, never
taught. B8 proves the factory incumbent equal to `make_brain(seed, persist=False).brain._ixn` under
`brain.correct` / `brain.brain.route`, under `throwaway_home()`.

The presized control (2.4): a FROZEN front-end in front of the incumbent's own class. z = W phi (phi the
learner's 52-dim PSP-of-rank code, W H x 52 ~ N(0, 1) from `random.Random(seed * 7919 + H)`), x_j = max(0, z_j -
m_j) with m_j unit j's median drive over the problem's distinct stimulus texts (TEACH texts only for C1-d, so no
held-out text enters the front-end), read by `SpikingInteractionLearner(n_in=H, t_post=6)` with the canonical
stack, taught by `interact(x, label)`. Disclosed: a different front-end on the same teaching sequence, not the
same channel; it proves that A layer of size H solves a cell, never that a small one does. The NGSNN feature
mode (growth/ng_snn.py: LIF rate of the constant current A = W f + b over T steps, its own multi-scale draw)
is the P2 adopt-or-control challenger, read on the dev seeds (2.4, 2.13).

future_d (Codex #1): no learner arm reads it; `taught_problems.Problem.episodes` refuses it and `preflight`
refuses an arm list that names it.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.taught_campaign            # self-check
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.taught_campaign --run OUT   # campaign

The self-check teaches only on exposed scratch seed 8900 at a short schedule and builds no Brain.
"""

import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

from snn_data_lake.instruments import delayed_cue as dc
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments import taught_problems as tp

# distinct-from: instruments/carried_state_driver.py -- G1(c)'s campaign driver over the reservoir pool; this one
#   drives the taught chat learner on G2's problems. instruments/inseparable_mapping.py::construction_check --
#   (ii)'s A1-A5 on 7000-7007, read beside (never re-run as) this campaign's (ii) rows. Sites read 2026-10-04.

EPOCHS = 400  # [PROJECTION, 2.7]: re-derived by the dated pre-data amendment (dev seeds 8000-8003)
H = 64  # [PROJECTION, 2.4]: re-derived on GREET / GOAL by the pre-data amendment (2.13's rule)
# The dated pre-data amendment (2.13's second freeze step) that fixes EPOCHS, H and the NGSNN disposition. None
# until the lead commits it; while None, preflight refuses every confirmatory / reserve seed (cp1 #7).
AMENDMENT = None
DEV_SEEDS = (8000, 8001, 8002, 8003)
CONFIRMATORY_SEEDS = tuple(range(8200, 8208))
RESERVE_SEEDS = tuple(range(8300, 8308))
RED_SEED = 8500  # the planted-reds block 8500-8599
SCRATCH_SEEDS = tuple(range(8900, 9000))
LIVE_SEEDS = frozenset([*range(300, 362), 6000, 6001, *range(6100, 6108), *range(7000, 7008)])
LIVE_HOME = Path(os.path.expanduser("~")) / ".snn_data_lake"
REPO = Path(__file__).resolve().parents[3]


# ---- the canonical learners ---------------------------------------------------------------------------------
learner = (
    tp.canonical_incumbent
)  # the ONE factory (P2); B8 below proves it equal to the Brain's learner


# ---- the front-ends -----------------------------------------------------------------------------------------
class Frozen:
    """A frozen front-end + the incumbent's class over it. `features(text)` -> the H-dim stimulus."""

    def __init__(self, features, h, seed, lr=None):
        self.features, self.unit = features, learner(seed, n_in=h, lr=lr)

    def route_text(self, text):
        return self.unit.respond(self.features(text))

    def learn_turn(self, text, _firewall, feedback):
        return self.unit.interact(self.features(text), feedback)


def stimulus_texts(problem, seed, history=False):
    """The problem's distinct stimulus texts the front-end centres on: probe (or history) texts; for C1-d /
    A_c1d the TEACH split only."""
    if problem.kind == "combo":
        return sorted({dc.combo_text(problem.family, it) for it in problem.items(seed, "teach")})
    if problem.kind in ("i", "u", "us"):
        make = dc.history_text if history else dc.probe_text
        return sorted({make(it) for it in dc.ITEMS_I})
    texts = dc.TEXTS_AII if problem.kind == "aii" else im.TEXTS
    return sorted(texts.values())


def presized_expansion(problem, h, seed, texts=None, *, history=False, lr=None):
    """The centred frozen expansion (2.4). `texts` (G5's non-adapted arm) defaults byte-identically to the
    problem's distinct stimulus texts."""
    rng = random.Random(seed * 7919 + h)
    w = [[rng.gauss(0.0, 1.0) for _ in range(im.N_IN)] for _ in range(h)]
    texts = stimulus_texts(problem, seed, history) if texts is None else list(texts)
    drives = [[dc.dot(row, im.code(t)) for t in texts] for row in w]
    medians = [statistics.median(d) for d in drives]

    def features(text):
        phi = im.code(text)
        return [max(0.0, dc.dot(row, phi) - m) for row, m in zip(w, medians, strict=True)]

    return Frozen(features, h, seed, lr=lr)


def ngsnn_expansion(problem, h, seed, *, history=False, lr=None):
    """NGSNN's feature mode as the frozen front-end (the P2 challenger, 2.4): its own multi-scale draw on its
    own CPU generator (P units per scale, as its candidate pool), its LIF-rate nonlinearity over T steps, the
    learner's code as f. Disclosed adapter: one fixed draw, no growth, no analytical readout."""
    import torch

    from snn_data_lake.growth.ng_snn import NGSNN

    net = NGSNN(device="cpu", seed=seed)
    per = math.ceil(h / len(net.lambdas))
    ws, bs = [], []
    for lam in net.lambdas:
        ws.append((torch.rand(per, im.N_IN, generator=net.g) * 2 - 1) * lam)
        bs.append((torch.rand(per, generator=net.g) * 2 - 1) * lam)
    w, b = torch.cat(ws)[:h], torch.cat(bs)[:h]

    def features(text):
        f = torch.tensor([im.code(text)], dtype=torch.float32)
        return [float(v) for v in net._rates(f, w, b)[0]]

    return Frozen(features, h, seed, lr=lr)


# ---- the readers (one interface: route(turns) -> answer; teach(turns, label)) -----------------------------
class Reader:
    """Reads the probe (or the explicit history) through `route_text`; teaches the probe through `learn_turn`."""

    def __init__(self, unit, history=False):
        self.unit, self.history = unit, history

    def text(self, turns):
        return (
            " ".join(t.text for t in turns[-dc.HISTORY_WINDOW :])
            if self.history
            else turns[-1].text
        )

    def route(self, turns):
        return self.unit.route_text(self.text(turns))

    def teach(self, turns, label):
        return self.unit.learn_turn(self.text(turns), None, feedback=label)


def make_reader(arm, problem, seed, *, h=H, lr=None, window=dc.HISTORY_WINDOW):
    """incumbent | history | presized | presized_history | ngsnn | ngsnn_history."""
    if arm == "incumbent":
        return Reader(learner(seed, lr=lr))
    if arm == "history":
        reader = Reader(learner(seed, lr=lr), history=True)
        reader.text = lambda turns: " ".join(t.text for t in turns[-window:])
        return reader
    history = arm.endswith("_history")
    if arm.startswith("presized"):
        return Reader(presized_expansion(problem, h, seed, history=history, lr=lr), history=history)
    if arm.startswith("ngsnn"):
        return Reader(ngsnn_expansion(problem, h, seed, history=history, lr=lr), history=history)
    raise ValueError(f"unknown arm {arm!r}")


# distinct-from: substrate/chat_memory.py -- a chat turn's retrieval cue words
# distinct-from: infra/region_wiring.py -- the thread-count gate
# distinct-from: reasoning/test_prototype.py -- a do-nothing control on a corrupted cue
# distinct-from: sensorimotor/nl_interface.py -- a word-boundary cue matcher
# distinct-from: substrate/episodic_recall.py -- coincidence ranking of stored episodes. Sites read 2026-10-04.
# B6's red: the explicit-history reader with the cue turn replaced by the constant word CUE[0]
def constant_cue_reader(problem, seed):
    reader = make_reader("history", problem, seed)
    reader.text = lambda turns: " ".join(
        [dc.CUE[0], *(t.text for t in turns[-dc.HISTORY_WINDOW + 1 :])]
    )
    return reader


def _shuffled_labels(seed):
    """Each epoch's labels re-arranged, never equal to the truth (inseparable_mapping._shuffled's form)."""
    rng = random.Random(seed + 1_000_003)

    def taught(true):
        if len(set(true)) < 2:
            return list(true)
        while True:
            labels = list(true)
            rng.shuffle(labels)
            if labels != true:
                return labels

    return taught


def attempt(
    problem,
    reader,
    seed,
    epochs,
    *,
    judge_future=False,
    stop_when_stable=False,
    shuffled=False,
    on_epoch_end=None,
):
    """One E-epoch teaching attempt: per episode, context turns are READ and the probe is taught; at each epoch
    end a learning-OFF validation read. Returns epoch-end solved flags and accuracies, the max accuracy, the
    first solved epoch end, and (judge_future) one final future read."""
    taught = _shuffled_labels(seed) if shuffled else None
    solved_ends, accuracy, read_indices = [], [], []
    for k, epoch in enumerate(problem.teach_epochs(seed, epochs)):
        true = [episode[-1].label for episode in epoch]
        labels = taught(true) if taught else true
        for episode, label in zip(epoch, labels, strict=True):
            for i in range(len(episode) - 1):
                reader.route(episode[: i + 1])  # context turns arrive through the read
            reader.teach(episode, label)
        # each epoch-end read is its own read index (U_S draws fresh labels per read; cp1 #9)
        records = []
        reads = tp.read_split(
            problem, seed, "validation", reader.route, read_index=k, records=records
        )
        read_indices.append(k)
        solved_ends.append(tp.solved(problem, seed, "validation", reads))
        accuracy.append(sum(a == b for a, b in reads) / len(reads))
        if on_epoch_end:
            on_epoch_end(k, reader, records)  # the ACTUAL read's records (B3 judges these; cp1 #10)
        if stop_when_stable and tp.solves_stably(solved_ends):
            break
    out = {
        "solved_ends": solved_ends,
        "accuracy_ends": accuracy,
        "max_accuracy": max(accuracy),
        "first_solved_end": next((i for i, s in enumerate(solved_ends) if s), None),
        "stable": tp.solves_stably(solved_ends),
        "never": tp.never_solves(solved_ends),
        "epochs_run": len(solved_ends),
        "validation_read_indices": read_indices,
    }
    if judge_future:
        reads = tp.read_split(problem, seed, "future", reader.route)
        out["future_solved"] = tp.solved(problem, seed, "future", reads)
        out["future_accuracy"] = sum(a == b for a, b in reads) / len(reads)
    return out


# ---- B3: state identity on a live reader ------------------------------------------------------------------------
# B3 on a live reader over an attempt: every epoch-end read judged on its ACTUAL stream (cp1 #10)
def b3_on_stream(reader, problem, seed, epochs):
    verdicts = []
    attempt(
        problem,
        reader,
        seed,
        epochs,
        on_epoch_end=lambda k, r, records: verdicts.append(tp.check_b3_stream(records)["ok"]),
    )
    return all(verdicts)


# ---- the arms (2.9, 2.15.1) ---------------------------------------------------------------------------------------
# (arm_id, problem_id, reader arm, expectation, judge_future)
ARMS = (
    ("inc.S0C0", "i.S0C0", "incumbent", "solves", False),
    ("inc.S1C0", "i.S1C0", "incumbent", "never", False),
    ("inc.S0C1", "i.S0C1", "incumbent", "never", False),
    ("inc.S1C1", "i.S1C1", "incumbent", "never", False),
    ("hist.S0C0", "i.S0C0", "history", "solves", False),
    ("hist.S1C0", "i.S1C0", "history", "solves", False),
    ("hist.S1C1", "i.S1C1", "history", "never", False),
    ("pre.S0C1", "i.S0C1", "presized", "solves", False),
    ("pre.S1C1h", "i.S1C1", "presized_history", "solves", False),
    ("pre.S1C0", "i.S1C0", "presized", "never", False),
    ("inc.aii", "aii", "incumbent", "solves", False),
    ("inc.ac1d", "ac1d", "incumbent", "solves", True),
    ("inc.c1d", "c1d", "incumbent", "never", True),
    ("pre.c1d", "c1d", "presized", "solves", True),
)
ARMS_II = (
    ("inc.ii.xor", "ii.xor", "incumbent", "never", False),
    ("inc.ii.twin", "ii.twin", "incumbent", "solves", False),
    ("pre.ii.xor", "ii.xor", "presized", "solves", False),
    ("pre.ii.twin", "ii.twin", "presized", "solves", False),
)


def met(expectation, result, judge_future):
    if expectation == "solves":
        return result["stable"] and (not judge_future or result["future_solved"])
    return result["never"] and (not judge_future or not result["future_solved"])


def run_arm(arm, seed, epochs=EPOCHS, h=H):
    arm_id, pid, kind, expectation, judge_future = arm
    problem = tp.PROBLEMS[pid]
    reader = make_reader(kind, problem, seed, h=h)
    b3 = []
    hook = None
    if kind == "incumbent" and pid in ("i.S1C0", "i.S1C1"):

        def hook(k, r, records):
            b3.append(tp.check_b3_stream(records)["ok"])

    result = attempt(
        problem,
        reader,
        seed,
        epochs,
        judge_future=judge_future,
        stop_when_stable=(expectation == "solves"),
        on_epoch_end=hook,
    )
    result.update(
        {
            "arm": arm_id,
            "seed": seed,
            "expectation": expectation,
            "met": met(expectation, result, judge_future),
        }
    )
    if b3:
        result["b3_every_read"] = all(b3)
    return result


# ---- B8: channel identity ------------------------------------------------------------------------------------------
def b8_twin(seed, epochs, *, bare=False):
    """B8: one `make_brain(seed, persist=False)` under `throwaway_home()` taught S0C0 through `brain.correct` and
    read through `brain.brain.route`, against the factory incumbent (or the bare class, the red) on the same
    sequence: weights and every validation read equal at every epoch end. The live home's mtime must not move."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import require_isolated_home, throwaway_home

    problem = tp.PROBLEMS["i.S0C0"]
    before = _live_mtime()
    factory = im._incumbent(seed) if bare else learner(seed)
    equal = True
    with throwaway_home():
        require_isolated_home()  # ruling 16: the ONE shared guard, before any Brain exists
        brain = make_brain(seed=seed, persist=False)
        for epoch in problem.teach_epochs(seed, epochs):
            for episode in epoch:
                probe = episode[-1]
                brain.correct(probe.text, probe.label)
                factory.learn_turn(probe.text, None, feedback=probe.label)
            reads_b = tp.read_split(
                problem, seed, "validation", lambda t: brain.brain.route(t[-1].text)
            )
            reads_f = tp.read_split(
                problem, seed, "validation", lambda t: factory.route_text(t[-1].text)
            )
            equal &= reads_b == reads_f and brain.brain._ixn.weights() == factory.weights()
    return {
        "ok": equal and _live_mtime() == before,
        "conditions": {"twin_equal": equal, "live_home_untouched": _live_mtime() == before},
    }


def _live_mtime():
    """The live home's stamp, REUSED from the move gate's census guard (P2: one guard): every top-level
    file's (size, mtime_ns)."""
    from snn_data_lake.gates.move_gate import _live_home_stamp

    return _live_home_stamp()


def _assert_isolated(before):
    """Halt if the live home's stamp moved since `before` (called AFTER the campaign's last write)."""
    after = _live_mtime()
    if after != before:
        raise RuntimeError("isolation failure: the live home changed; campaign halted")


# ---- the G-R2 reading (2.9, 2.15.1; first row that applies decides) -----------------------------------------------
def reading(r):
    """A pure function of the campaign's flags `r` (each True/False; missing (ii) rows mean NOT RUN)."""
    if not r["b8_ok"] or not r["reds_fired"]:
        return "HALT: shared instrument (B8 or a red did not fire)"
    if not r["construction_ok"]:
        return "RED: construction wrong (the generator)"
    if not r["b3_ok"] or r["presized_solves_s1c0"]:
        return "HALT: a reader carries hidden state"
    if r["incumbent_solves_s1_or_c1"] or r["history_solves_s1c1"] or r["incumbent_solves_c1d"]:
        return "RED: construction wrong (a capacity- or state-cell solve)"
    if r["incumbent_fails_s0c0"]:
        return "RED: the channel does not teach (to the owner)"
    if r["history_fails_s1c0"]:
        return "RED: the delayed-cue explicit-history control fails"
    if r["presized_fails_capacity"]:
        return "RED: the presized nonlinear positive control fails"
    if r["c1d_presized_fails"]:
        return "C1-d REPRESENTABLE, NOT SHOWN LEARNABLE; (i) PASS on its other cells"
    if r["a_fails"]:
        return "PASS for (i) and C1-d; task A not offered (to the lead)"
    return "PASS: S0C0 positive; S1C0 STATE; S0C1 CAPACITY; S1C1 STATE + CAPACITY; C1-d CAPACITY held-out"


# distinct-from: faculties/access_consciousness.py -- an ignition-threshold access predicate
# distinct-from: faculties/recursive_tom.py -- T082's per-seed reading of its own rows
# distinct-from: faculties/vsa_reasoning.py -- a one-spike-per-dimension predicate
# distinct-from: gates/move_gate.py -- the move census diff's reds
# distinct-from: infra/brain_telemetry.py -- the telemetry contract self-check. Sites read 2026-10-04.
# (ii)'s separate reading (2.9; cp1 #6), over complete per-seed rows of ARMS_II, or NOT RUN. Clause 1 is G1(c)'s
# A3 on the bare learner; a canonical-learner XOR solve here is a DISAGREEMENT reported beside it (the float
# near-tie carried opening, to the owner), never a RED against G1(c)'s record. Clauses 2 and 3 are G2's own.
def reading_ii(rows_ii, seeds):
    if rows_ii is None:
        return "NOT RUN: G1(c)'s construct-stage PASS was not supplied"
    for arm in ARMS_II:
        if sorted(r.get("seed") for r in rows_ii.get(arm[0], [])) != sorted(seeds):
            raise ValueError(f"(ii) arm {arm[0]} lacks a result per seed")

    def all_met(arm_id):
        return all(r["met"] for r in rows_ii[arm_id])

    if not all_met("inc.ii.xor"):
        return "DISAGREEMENT with G1(c)'s A3 (a canonical-learner XOR solve): the near-tie opening, to the owner"
    if not all_met("inc.ii.twin"):
        return "RED: (ii)'s separable twin is not learned (clause 2)"
    if not (all_met("pre.ii.xor") and all_met("pre.ii.twin")):
        return "RED: (ii)'s presized nonlinear positive control fails (clause 3)"
    return (
        "PASS (ii): twin learned, presized solves XOR and twin; clause 1 reported beside G1(c)'s A3"
    )


# G1(c)'s construct-stage evidence: verdict True on exactly the check seeds 7000-7007
def g1c_construct_passed(record):
    return (
        isinstance(record, dict)
        and record.get("verdict") is True
        and sorted(record.get("seeds", ())) == list(range(7000, 7008))
    )


PASS_FLAGS = {
    "b8_ok": True,
    "reds_fired": True,
    "construction_ok": True,
    "b3_ok": True,
    "presized_solves_s1c0": False,
    "incumbent_solves_s1_or_c1": False,
    "history_solves_s1c1": False,
    "incumbent_solves_c1d": False,
    "incumbent_fails_s0c0": False,
    "history_fails_s1c0": False,
    "presized_fails_capacity": False,
    "c1d_presized_fails": False,
    "a_fails": False,
}


def reading_reds():
    """One constructed input per reading row: flipping that row's flag must produce that row's verdict."""
    expected = {
        "b8_ok": "HALT: shared",
        "construction_ok": "RED: construction wrong (the generator)",
        "b3_ok": "HALT: a reader",
        "incumbent_solves_c1d": "RED: construction wrong (a capacity",
        "incumbent_fails_s0c0": "RED: the channel",
        "history_fails_s1c0": "RED: the delayed-cue",
        "presized_fails_capacity": "RED: the presized",
        "c1d_presized_fails": "C1-d REPRESENTABLE",
        "a_fails": "PASS for (i) and C1-d; task A",
    }
    fired = {}
    for flag, prefix in expected.items():
        flags = dict(PASS_FLAGS)
        flags[flag] = not flags[flag]
        fired[flag] = reading(flags).startswith(prefix)
    fired["all_pass"] = reading(PASS_FLAGS).startswith("PASS: S0C0")
    return {"fired": fired, "all_fired": all(fired.values())}


# distinct-from: connectome.py -- a BFS reachability query
# distinct-from: engine/autonomous_loop.py -- an end-effector position from a body state
# distinct-from: engine/brain.py -- episode recall from working memory and the lake
# distinct-from: faculties/acetylcholine.py -- an ACh write/recall headline
# distinct-from: faculties/allocentric_map.py -- a landmark sighting. Sites read 2026-10-04.
# every mandatory arm must carry exactly one result per campaign seed (cp1 #5); partial evidence raises
def flags_from(rows, construction_ok, b8_ok, reds_fired, *, seeds, arms=ARMS):
    for arm in arms:
        got = sorted(row.get("seed") for row in rows.get(arm[0], []))
        if got != sorted(seeds):
            raise ValueError(f"arm {arm[0]}: results for seeds {got}, expected {sorted(seeds)}")

    def all_met(arm_id):
        return all(row["met"] for row in rows[arm_id])

    b3 = all(
        row.get("b3_every_read", True)
        for arm in ("inc.S1C0", "inc.S1C1")
        for row in rows.get(arm, [])
    )
    return {
        "b8_ok": b8_ok,
        "reds_fired": reds_fired,
        "construction_ok": construction_ok,
        "b3_ok": b3,
        "presized_solves_s1c0": not all_met("pre.S1C0"),
        "incumbent_solves_s1_or_c1": not all(
            all_met(a) for a in ("inc.S1C0", "inc.S0C1", "inc.S1C1")
        ),
        "history_solves_s1c1": not all_met("hist.S1C1"),
        "incumbent_solves_c1d": not all_met("inc.c1d"),
        "incumbent_fails_s0c0": not all_met("inc.S0C0"),
        "history_fails_s1c0": not (all_met("hist.S1C0") and all_met("hist.S0C0")),
        "presized_fails_capacity": not (all_met("pre.S0C1") and all_met("pre.S1C1h")),
        "c1d_presized_fails": not all_met("pre.c1d"),
        "a_fails": not (all_met("inc.aii") and all_met("inc.ac1d")),
    }


# ---- the campaign's reds (2.8) ---------------------------------------------------------------------------------------
def campaign_reds(seed=RED_SEED, epochs=EPOCHS, h=H):
    """Each check's planted red on the reds block: a stand-in reader whose result must trip the check."""
    s0c0, s1c0, s0c1, s1c1 = (tp.PROBLEMS[f"i.{c}"] for c in ("S0C0", "S1C0", "S0C1", "S1C1"))
    fired = {
        "b3_history_stands_in": not b3_on_stream(
            make_reader("history", s1c0, seed), s1c0, seed, 20
        ),
        "b4_presized_on_s0c1": not attempt(
            s0c1, make_reader("presized", s0c1, seed, h=h), seed, epochs
        )["never"],
        "b5_lr_zero": not attempt(s0c0, make_reader("incumbent", s0c0, seed, lr=0.0), seed, epochs)[
            "stable"
        ],
        "b5_shuffled": not attempt(
            s0c0, make_reader("incumbent", s0c0, seed), seed, epochs, shuffled=True
        )["stable"],
        "b6_window_1": not attempt(
            s1c0, make_reader("history", s1c0, seed, window=1), seed, epochs
        )["stable"],
        # cp1 #15: the cue replaced by a constant word in the history text (B6's registered perturbation)
        "b6_constant_cue": not attempt(s1c0, constant_cue_reader(s1c0, seed), seed, epochs)[
            "stable"
        ],
        # cp1 #15: B8's bare-class twin, staged with the rest so a non-firing red halts before the arms
        "b8_bare_class": not b8_twin(seed, 3, bare=True)["conditions"]["twin_equal"],
        "b6_presized_history_on_s1c1": not attempt(
            s1c1, make_reader("presized_history", s1c1, seed, h=h), seed, epochs
        )["never"],
        "b7_identity_front_end": not attempt(
            s0c1, make_reader("incumbent", s0c1, seed), seed, epochs
        )["stable"],
        "b7_shuffled": not attempt(
            s0c1, make_reader("presized", s0c1, seed, h=h), seed, epochs, shuffled=True
        )["stable"],
        "b7_history_on_s1c0_plain": not attempt(
            s1c0, make_reader("history", s1c0, seed), seed, epochs
        )["never"],
        **{f"reading_{k}": v for k, v in reading_reds()["fired"].items()},
    }
    return {"fired": fired, "all_fired": all(fired.values())}


# ---- the dev measurement path (2.13's pre-data amendment; cp1 #7) ------------------------------------------------------
# distinct-from: instruments/binding_instrument.py -- T080's per-arm print line
# distinct-from: sleep/developmental_pruning.py -- a softmax delta-rule epoch
# distinct-from: faculties/global_workspace.py -- a running-mean baseline rule
# distinct-from: faculties/interoceptive_integration.py -- per-channel running baselines
# distinct-from: growth/similarity_sized_growth.py -- nearest-expert routing. Sites read 2026-10-04.
SAME_ITEMS_CAPACITY = (
    ("presized", "i.S0C1"),
    ("presized_history", "i.S1C1"),
    ("presized", "ii.xor"),
    ("presized", "ii.twin"),
)
POSITIVES = (
    ("incumbent", "i.S0C0"),
    ("history", "i.S1C0"),
    ("history", "i.S0C0"),
    ("incumbent", "aii"),
    ("incumbent", "ac1d"),
    ("incumbent", "ii.twin"),
)


def _dev_row(stage, kind, pid, seed, epochs, h, *, solves=True):
    problem = tp.PROBLEMS[pid]
    future = problem.kind == "combo"
    r = attempt(
        problem,
        make_reader(kind, problem, seed, h=h),
        seed,
        epochs,
        judge_future=future,
        stop_when_stable=solves,
    )
    return {
        "stage": stage,
        "arm": kind,
        "problem": pid,
        "seed": seed,
        "h": h,
        "epochs": epochs,
        "stable": r["stable"],
        "never": r["never"],
        "max_acc": r["max_accuracy"],
        "to_stable": r["epochs_run"] - tp.STABLE if r["stable"] else None,
        "future_solved": r.get("future_solved"),
        "future_acc": r.get("future_accuracy"),
    }


# 2.13's dev readings on dev seeds only, one JSON line per run. R1: the smallest H solving every same-items
# capacity cell on every seed within EPOCHS / 2 (H = 2 x it). R6: the presized control on C1-d (validation AND
# future) at every H. R2: E = max(EPOCHS, 2 x the largest epochs-to-stable of any positive, rounded up to 50).
# R3: the explicit-history proof. R4: NGSNN's feature mode on every capacity cell (adopt iff B7 on every seed).
# R5: A_ii and A_c1d through the incumbent. Returns the derived values.
def dev_measurement(out_path, seeds=DEV_SEEDS, h_grid=(16, 32, 64, 128), max_epochs=4000):
    if not set(seeds) <= set(DEV_SEEDS):
        raise PermissionError(f"dev measurement on dev seeds {DEV_SEEDS} only: {seeds}")

    def log(row):
        with open(out_path, "a", encoding="ascii") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def ok(row, within):
        return row["stable"] and row["to_stable"] <= within and row["future_solved"] in (None, True)

    smallest = None
    for h in h_grid:
        rows = [
            log(_dev_row("R1", k, p, s, EPOCHS, h)) for k, p in SAME_ITEMS_CAPACITY for s in seeds
        ]
        if all(ok(r, EPOCHS // 2) for r in rows):
            smallest = h
            break
    h_final = 2 * smallest if smallest else None
    hc = h_final or H
    c1d = {
        h: all(ok(log(_dev_row("R6", "presized", "c1d", s, EPOCHS, h)), EPOCHS) for s in seeds)
        for h in h_grid
    }
    largest = 0
    for k, p in POSITIVES + SAME_ITEMS_CAPACITY:
        for s in seeds:
            row = log(_dev_row("R2", k, p, s, max_epochs, hc))
            largest = max(largest, row["to_stable"] or 0)
    e_final = max(EPOCHS, int(math.ceil(2 * largest / 50.0) * 50))
    history = all(
        log(_dev_row("R3", "history", "i.S1C0", s, e_final, hc))["stable"]
        and log(_dev_row("R3", "history", "i.S0C0", s, e_final, hc))["stable"]
        and log(_dev_row("R3", "history", "i.S1C1", s, e_final, hc, solves=False))["never"]
        for s in seeds
    )
    ngsnn = all(
        ok(log(_dev_row("R4", k.replace("presized", "ngsnn"), p, s, e_final, hc)), e_final)
        for k, p in SAME_ITEMS_CAPACITY + (("presized", "c1d"),)
        for s in seeds
    ) and all(
        log(_dev_row("R4", "ngsnn", "i.S1C0", s, e_final, hc, solves=False))["never"] for s in seeds
    )
    a_rows = all(
        ok(log(_dev_row("R5", "incumbent", p, s, e_final, hc)), e_final)
        for p in ("aii", "ac1d")
        for s in seeds
    )
    derived = {
        "H": h_final,
        "E": e_final,
        "history_proof": history,
        "ngsnn_adopt": ngsnn,
        "c1d_learned_at": [h for h, v in c1d.items() if v],
        "a_rows_ok": a_rows,
        "largest": largest,
    }
    log({"stage": "derived", **derived})
    return derived


# ---- the driver -------------------------------------------------------------------------------------------------------
def preflight(seeds, out_dir, arms=ARMS, *, require_clean=True):
    """Refusals before any read: a live seed, a seed outside G2's block, OUT inside the repo, an arm list naming
    future_d, a dirty tree."""
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise PermissionError(f"seeds must be non-empty and distinct: {seeds}")
    for block in (
        CONFIRMATORY_SEEDS,
        RESERVE_SEEDS,
    ):  # a confirmatory read is the WHOLE registered block (cp1 #5)
        if set(seeds) & set(block) and sorted(seeds) != list(block):
            raise PermissionError(
                f"seeds {sorted(seeds)} touch {block[0]}-{block[-1]} but are not that exact block"
            )
        if set(seeds) & set(block) and AMENDMENT is None:
            raise PermissionError(
                "confirmation refused: the dated pre-data amendment (2.13) is not entered"
            )
    bad = sorted(set(seeds) & LIVE_SEEDS)
    if bad:
        raise PermissionError(f"live seeds refused: {bad}")
    outside = [s for s in seeds if not 8000 <= s <= 8999]
    if outside:
        raise PermissionError(f"seeds outside G2's block 8000-8999: {outside}")
    out = Path(out_dir).resolve()
    if out == REPO or REPO in out.parents:
        raise PermissionError(f"OUT {out} is inside the repo")
    live = LIVE_HOME.resolve()
    if out == live or live in out.parents:
        raise PermissionError(f"OUT {out} is the live home or inside it")
    if any("future_d" in str(field) for arm in arms for field in arm):
        raise PermissionError("an arm names future_d (G6(d)'s untouched set)")
    if require_clean:
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise PermissionError("dirty tree refused")


# distinct-from: instruments/report_order_probe.py -- T081's apparatus dry run
# distinct-from: faculties/recursive_tom.py -- T082's per-seed printer
# distinct-from: gates/policy_manifest.py -- in_ci / in_selfchecks derivation
# distinct-from: instruments/binding_instrument.py -- T080's apparatus dry run
# distinct-from: instruments/everyday_battery.py -- the C0 battery's port-faithfulness check. Sites read 2026-10-04.
# The campaign (2.9's order): construction (no learner) -> B8 per seed -> reds -> arms; (ii) ONLY on G1(c)'s
# construct-stage evidence `g1c_construct` (verdict True on 7000-7007; cp1 #6), else (ii) reads NOT RUN;
# evidence supplied but not passing is refused. Writes OUT/g2_campaign.json.
def run_campaign(seeds, out_dir, *, epochs=EPOCHS, h=H, g1c_construct=None, require_clean=True):
    if g1c_construct is not None and not g1c_construct_passed(g1c_construct):
        raise PermissionError(
            "G1(c)'s construct-stage evidence does not show a PASS on 7000-7007: (ii) refused"
        )
    include_ii = g1c_construct is not None
    # distinct-from: brainstem/so101_arm.py -- the arm's E-STOP sidecar write
    # distinct-from: faculties/agency_monitor.py -- T079's own campaign runner
    # distinct-from: faculties/metacognitive_monitor.py -- T077's own campaign runner
    # distinct-from: faculties/recursive_tom.py -- T082's own campaign runner
    # distinct-from: instruments/attention_schema.py -- T078's own campaign runner
    # distinct-from: instruments/binding_instrument.py -- T080's own campaign runner ("a shared runner would be
    #   a fifth mechanism, not one"). Sites read 2026-10-04.
    arms = ARMS + (ARMS_II if include_ii else ())
    preflight(seeds, out_dir, arms, require_clean=require_clean)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    before = _live_mtime()
    record = {"seeds": list(seeds), "epochs": epochs, "h": h}

    def conclude(verdict):
        # a failed prerequisite concludes HERE: no later stage builds a learner, teaches or reads (cp1 #2)
        record["verdict"] = verdict
        return _record_campaign(out_dir, record, before)

    construction = dc.construction_check(seeds)
    p2 = {
        "b9": tp.check_b9()["ok"],
        "b11": tp.check_b11()["ok"],
        "b12": all(tp.check_b12(learner(s), tp.PROBLEMS["i.S1C1"], s)["ok"] for s in seeds),
    }
    record.update(
        construction_ok=construction["verdict"] and all(p2.values()),
        p2=p2,
        witnesses=construction.get("witnesses", {}).get("cases"),
        lp=dc.lp_measurements(),
    )  # D2 beside the witness margins (cp1 #20)
    if not record["construction_ok"]:
        return conclude("RED: construction wrong (the generator)")
    record["b8"] = {}
    for s in seeds:
        record["b8"][s] = b8_twin(s, epochs)["ok"]
        if not record["b8"][s]:
            return conclude(f"HALT: shared instrument (B8 failed on seed {s})")
    record["reds"] = {
        "p1": dc.construction_reds()["all_fired"],
        "p2": tp.construction_reds(learner)["all_fired"],
    }
    if not (record["reds"]["p1"] and record["reds"]["p2"]):
        return conclude("HALT: shared instrument (a P1/P2 red did not fire)")
    record["reds"]["campaign"] = campaign_reds(epochs=epochs, h=h)
    if not record["reds"]["campaign"]["all_fired"]:
        return conclude("HALT: shared instrument (a campaign red did not fire)")
    rows = {arm[0]: [run_arm(arm, s, epochs, h) for s in seeds] for arm in arms}
    flags = flags_from(rows, True, True, True, seeds=seeds, arms=ARMS)
    rows_ii = {arm[0]: rows[arm[0]] for arm in ARMS_II} if include_ii else None
    record.update(
        rows=rows, flags=flags, verdict=reading(flags), verdict_ii=reading_ii(rows_ii, seeds)
    )
    return _record_campaign(out_dir, record, before)


def _record_campaign(out_dir, record, before):
    Path(out_dir, "g2_campaign.json").write_text(
        json.dumps(record, indent=1, default=str), encoding="ascii"
    )
    _assert_isolated(before)  # AFTER the last write: an isolation failure halts, never reports
    return record


def main():
    """The fast self-check (scratch seed 8900, 30 epochs, no Brain): the reading and its reds, preflight's
    refusals, the front-ends' determinism, and the incumbent solving S0C0 while never solving S1C0."""
    assert reading_reds()["all_fired"]
    for seeds, out in (((300,), "x"), ((8000,), str(REPO / "out")), ((12000,), "x")):
        try:
            preflight(seeds, out, require_clean=False)
            raise AssertionError(f"preflight accepted {seeds} {out}")
        except PermissionError:
            pass
    s0c0, s1c0 = tp.PROBLEMS["i.S0C0"], tp.PROBLEMS["i.S1C0"]
    a = presized_expansion(tp.PROBLEMS["i.S0C1"], 16, 8900).features("oak fox")
    assert a == presized_expansion(tp.PROBLEMS["i.S0C1"], 16, 8900).features("oak fox")
    pos = attempt(s0c0, make_reader("incumbent", s0c0, 8900), 8900, 30)
    neg = attempt(s1c0, make_reader("incumbent", s1c0, 8900), 8900, 30)
    assert pos["stable"] and neg["never"] and neg["max_accuracy"] <= 0.5, (pos, neg)
    print(
        f"self-check OK (scratch seed 8900, 30 epochs): reading reds all fired; preflight refused live, "
        f"in-repo and out-of-block runs; S0C0 first solved at epoch end {pos['first_solved_end']}; S1C0 max "
        f"accuracy {neg['max_accuracy']:.2f}"
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run":
        result = run_campaign(CONFIRMATORY_SEEDS, sys.argv[2])
        print(result["verdict"])
    else:
        main()
