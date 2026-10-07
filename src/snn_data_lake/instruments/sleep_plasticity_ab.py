r"""R3 / T104 P1 -- the 2026-09-27 sleep-plasticity arms (F1-F4) on the flat incumbent, one campaign.

The frozen pre-registration is specs/022-unified-brain-arc/evidence/r3-replay-growth-composition.md
(P0 f83c7c74 + the dated amendment 384fa78b); its sections 2.2-2.16 are the contract and this module only
implements them. Families (2.3): F1 synaptic (ON / OFF / R / BF / Rec / ORACLE / A-ONLY, plus the K1 twins),
F2 structural (S-ON / S-OFF), F3 attractor (AT-ON / AT-OFF, and the dev-only AT-NONE positive control),
F4 budget (SHORT / MED / EXT / MASSED). F5 (staged) waits on G6's runner and F6 / F7 on R2's results; they are
NOT here.

HOW ONE ARM-SEED RUNS (2.2). Two fresh processes on one scratch home (NORD_HOME, NORD_PERSIST=1), never the
live lake (board PROTOCOL 4b; K14 snapshots it before and after every child):
  bout      make_brain(seed=s) -> PS(s) (sessions A then B taught through learn_turn, R = 3 reps, one seeded
            stream) -> digest D0 -> the arm's pre-bout action -> digest D1 -> _enter_sleep -> n x (per-tick
            action, _sleep_tick) -> _enter_wake -> in-process held-out E -> close
  evaluate  make_brain(seed=s) on the same home (the restore path) -> route_text on every held-out line,
            learning off (weights + rng digested before and after, K4) -> close
The compute ledger is PRESENTATIONS: every call of the learner's `_argmax_label` during the bout (one forward
drive of one stored stimulus) is counted by an instance-level wrapper, an instrument independent of the arms.

Implementation choices the frozen text leaves open, stated here (none changes a contrast):
  * the line order of every teaching rep, the ORACLE re-teach and the A/B split all come from ONE stream,
    random.Random(s * 7919 + 3), drawn in a fixed order before any arm-specific action;
  * a per-tick instrument action (fwd / Rec draws) runs BEFORE that tick's _sleep_tick, so it precedes the
    tick's Benna-Fusi rest exactly as the incumbent replay deposit does (nothing between them touches the
    learner);
  * Rec draws with the learner's own rng, as the incumbent replay does; fwd draws with a twin cloned from it;
  * K5's red ("delete interaction_learner.json") deletes the scratch home's brain_lake.bin: since P4b the
    checkpoint is a blob inside the one lake, not a sidecar file.

Decisions: `verdict` is the statistic wrapper of P-8 (ruling 9 for superiority contrasts, ruling 9a for
non-inferiority bars, the 2^8 sign-flip rate enumerated under the ACTUAL verdict rule, P-8 / Codex 21);
the four decision tables are pure functions (K11), and `census` derives the test count from the IDs read
(P-6 / Codex 27). Statements of NOT MET never name an explanation (X-19).

One file over the 800-line soft ceiling, deliberately: one frozen contract, one harness -- the arm specs,
the two child processes, the checks that read their records and the tables that read the checks share one
record schema, and splitting them would put that schema in two places.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.sleep_plasticity_ab       (offline self-check)
  ... sleep_plasticity_ab run    --family F1 --set dev --out DIR    (arm-seeds, sequential, one process at a time)
  ... sleep_plasticity_ab decide --family F1 --set dev --out DIR    (checks, contrasts, table, census)
  ... sleep_plasticity_ab k7     --out DIR                          (the turnover detector's positive control)
  ... sleep_plasticity_ab red    --name K1 --out DIR                (one brain-level constructed red; must fire)
"""

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from fractions import Fraction
from typing import NamedTuple

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast, paired_t

S = 8  # confirmatory seeds per contrast (Part F)
K = 8  # presentations per sleep tick (the incumbent's own replay(k=8))
SEED_BLOCK = (16000, 16999)  # R3's block (board SEEDS.md)
SEED_SETS = {
    "dev": tuple(range(16000, 16008)),  # EXPOSED dev / rehearsal
    "F1": tuple(range(16100, 16108)),
    "F2": tuple(range(16110, 16118)),
    "F3": tuple(range(16120, 16128)),
    "F4": tuple(range(16130, 16138)),
}
RED_SEEDS = tuple(range(16900, 17000))  # constructed reds and positive controls only (2.10)
BUDGETS = {"SHORT": 10, "MED": 40, "EXT": 160}  # sleep ticks (2.3)
REPS = 3  # teaching reps per session (2.2)
EPS_A = Fraction(1, 8)  # one A item (2.4's NI margin)
# F3's constructed wake stores W (2.3; the dated amendment 2026-10-04): W = 30 put the attractor at ceiling on
# dev (AT-OFF and AT-NONE 36/36 on 8/8), so W is chosen by the frozen dev ladder below and written here by a
# dated note BEFORE any F3 confirmatory seed. None = not yet frozen: an F3 arm without an explicit W refuses.
F3_W_LADDER = (1, 2, 3, 6, 12)
F3_STORES = 1  # frozen 2026-10-06 by the dev ladder (the dated amendment in the pre-registration)
CHILD_TIMEOUT_S = 900
LIVE_LAKE = os.path.join(os.path.expanduser("~"), ".snn_data_lake")


class Arm(NamedTuple):
    caps: tuple  # (replay, benna_fusi, structural)
    action: str  # per-tick: "none" | "fwd" | "rec"
    budget: str  # key of BUDGETS
    ps: str = "AB"  # "AB" | "A" (A-ONLY's PS_A)
    pre: str = "none"  # pre-bout: "none" | "oracle" | "massed" | "stores"
    attractor: str = (
        "learn"  # "learn" | "nolearn" (AT-OFF) | "none" (AT-NONE, dev positive control)
    )
    d1_group: str = ""  # arms whose bout-start digests D1 must be equal (P-2)


ON_CAPS, OFF_CAPS = (True, True, False), (False, False, False)
FAMILIES = {
    "F1": {
        "ON": Arm(ON_CAPS, "none", "MED", d1_group="F1"),
        "ON-TWIN": Arm(ON_CAPS, "none", "MED", d1_group="F1"),
        "OFF": Arm(OFF_CAPS, "fwd", "MED", d1_group="F1"),
        "R": Arm((True, False, False), "none", "MED", d1_group="F1"),
        "BF": Arm((False, True, False), "fwd", "MED", d1_group="F1"),
        "Rec": Arm((False, True, False), "rec", "MED", d1_group="F1"),
        "ORACLE": Arm(OFF_CAPS, "fwd", "MED", pre="oracle"),
        "A-ONLY": Arm(OFF_CAPS, "fwd", "MED", ps="A"),
        "A-ONLY-TWIN": Arm(OFF_CAPS, "fwd", "MED", ps="A"),
    },
    "F2": {
        "S-ON": Arm((True, True, True), "none", "MED", d1_group="F2"),
        "S-OFF": Arm(ON_CAPS, "none", "MED", d1_group="F2"),
    },
    "F3": {
        "AT-ON": Arm(ON_CAPS, "none", "MED", pre="stores", d1_group="F3"),
        "AT-OFF": Arm(ON_CAPS, "none", "MED", pre="stores", attractor="nolearn", d1_group="F3"),
        "AT-NONE": Arm(ON_CAPS, "none", "MED", pre="stores", attractor="none", d1_group="F3"),
    },
    "F4": {
        "SHORT": Arm(ON_CAPS, "none", "SHORT", d1_group="F4"),
        "MED": Arm(ON_CAPS, "none", "MED", d1_group="F4"),
        "EXT": Arm(ON_CAPS, "none", "EXT", d1_group="F4"),
        "MASSED": Arm((False, True, False), "fwd", "EXT", pre="massed"),
    },
}
DEV_ONLY_ARMS = {("F3", "AT-NONE")}  # run on the dev set only (2.6's historical positive control)
PLASTICITY_FREE = {"OFF", "ORACLE", "A-ONLY", "A-ONLY-TWIN"}  # every cap off and fwd only (K6)


# ---------------------------------------------------------------------------------------- safety (K14)
# distinct-from: gates/move_gate.py::_live_home_stamp -- that census stamp is private and hardwired to
#   the default home's top-level files; K14 (P-3) needs the SAME stamp as a function of a PATH so its
#   red can run on a decoy directory, never on the live lake, and it walks the whole tree.
# distinct-from: engine/live_brain.py -- spike_snapshot reads per-region spike rates; also read:
#   engine/report_channel.py::snapshot (a workspace report record), faculties/agency_monitor.py::snapshot
#   (per-channel status) -- term overlaps only; none stamps files on disk.
def lake_snapshot(path):
    """mtime_ns + size of every file under `path` (an absent path snapshots as empty)."""
    snap = {}
    for root, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            snap[os.path.relpath(p, path)] = (st.st_mtime_ns, st.st_size)
    return snap


def live_lake_untouched(path, before):
    """K14 (P-3): (ok, changed) -- every file under `path` reads the same mtime and size as `before`."""
    after = lake_snapshot(path)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    return not changed, changed


def require_scratch_home():
    """Children refuse to build a Brain unless NORD_HOME passes the ONE shared live-home guard
    (substrate/home.require_isolated_home: resolved, junction-following, equality and containment in
    either direction; Codex cp2 #1) and the arm runs persistent without self-modification."""
    from snn_data_lake.substrate.home import require_isolated_home

    home = require_isolated_home(live_home=LIVE_LAKE)
    if os.environ.get("NORD_PERSIST") != "1" or os.environ.get("NORD_SELFMOD"):
        raise RuntimeError("refused: arms run with NORD_PERSIST=1 and no NORD_SELFMOD")
    return home


def guard_out(out):
    """Codex cp2 #1: validate the RESOLVED output directory BEFORE any filesystem mutation. The guard needs
    an existing path, so it is applied to the nearest existing ancestor; the missing remainder is plain names
    under it, which cannot alias anything. Refuses the live lake, anything inside it, and anything containing
    it (an ancestor that contains the live home refuses too -- over-refusal is the safe side)."""
    from snn_data_lake.substrate.home import assert_not_live_home

    probe = os.path.abspath(str(out))
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    assert_not_live_home(probe, live_home=LIVE_LAKE)
    return probe


# ---------------------------------------------------------------------------------- the pre-sleep state
def split_and_order(seed, offset=0):
    """The A / B split and every teaching order, from ONE stream fixed before any teaching (2.2)."""
    from snn_data_lake.readout.interaction_learning import _CONV_INTENTS, _CONV_TRAIN

    rng = random.Random(seed * 7919 + 3 + offset)
    intents = list(_CONV_INTENTS)
    rng.shuffle(intents)
    a, b = intents[:2], intents[2:]

    def reps(session, n):
        out = []
        for _ in range(n):
            lines = [(line, lab) for lab in session for line in _CONV_TRAIN[lab]]
            rng.shuffle(lines)
            out.append(lines)
        return out

    teach_a, teach_b = reps(a, REPS), reps(b, REPS)
    oracle = reps(a, 1)[0]  # drawn AFTER both sessions, so it never moves PS(s)
    return a, b, teach_a, teach_b, oracle


def heldout(intents):
    from snn_data_lake.readout.interaction_learning import _CONV_TEST

    return [(line, lab) for lab in intents for line in _CONV_TEST[lab]]


def _digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def state_digest(b):
    """PreSleepDigest: the learner's whole G0(c) continuation state (weights, ring, rng, the plasticity stack
    it carries) plus `_rewirer is None` (2.2)."""
    return _digest({"ixn": b.brain._ixn.state_dict(), "rewirer_none": b._rewirer is None})


# distinct-from: faculties/quality_diversity.py -- run() is a QD search loop; also read:
#   plasticity/competitive_learning.py::reset_activity (clears traces), reasoning/reservoir_eprop.py::train_arm
#   (trains weights), reasoning/spike_chat_lake.py::_kb_reply (triple retrieval) -- term overlaps only; these
#   two only HASH the interaction learner's state for the checks.
def weights_digest(ixn):
    """The learner's synaptic weights alone (K1's post-bout twin equality, K6's plasticity-free arms)."""
    return _digest({"w": ixn.weights()})


def eval_digest(ixn):
    """Weights AND the learner rng (K4: an evaluation must move neither)."""
    return _digest({"w": ixn.weights(), "rng": repr(ixn.rng.getstate())})


def score(ixn, items):
    """Per-item correctness of route_text (pure inference, no plasticity, no rng)."""
    return [int(ixn.route_text(line) == lab) for line, lab in items]


def margin(ixn, items):
    """P-1 secondary (never decisive): mean drive of the correct label minus the best other, in the learner's
    membrane units. Pure read of `_drive` (no plasticity, no rng); Codex cp2 #12."""
    from snn_data_lake.readout.interaction_learning import text_features

    out = []
    for line, lab in items:
        code = ixn.encode(text_features(line, ixn.n_in))
        d = [ixn._drive(n, code) for n in ixn.neurons]
        k = ixn.responses.index(lab)
        out.append(d[k] - max(v for i, v in enumerate(d) if i != k))
    return sum(out) / len(out)


def chrono(b):
    """The K8 observables: wake chronology, the grower's wake counters and the arm's motor tick count."""
    neuro = getattr(b, "neuro", None)
    arm = getattr(b.brain, "arm", None)
    return {
        "step": b._step,
        "nights": b._nights,
        "wake_ticks": b._wake_ticks,
        "observed": getattr(neuro, "observed", None),
        "live_n": getattr(neuro, "live_n", None),
        "arm_ticks": getattr(arm, "_tick_t", None),
    }


# --------------------------------------------------------------------------------------- the bout child
# distinct-from: growth/brian2_backend.py -- its child runner simulates Brian2 networks; also read:
#   engine/learn_daemon.py (one-daemon lock), gates/gradient_claim_gate.py (run-age record),
#   brainstem/arm_config.py + so101_arm.py (joint geometry) -- term overlaps only; none runs a sleep arm.
def run_bout(family, arm_name, seed, budget=None, red=None, stores=None):
    """One arm's PS(s) + bout + in-process evaluation, in THIS process (the `bout` child). Returns a record."""
    require_scratch_home()
    arm = FAMILIES[family][arm_name]
    n = BUDGETS[arm.budget] if budget is None else int(budget)
    w = F3_STORES if stores is None else int(stores)
    if arm.pre == "stores" and w is None:
        raise RuntimeError(
            "refused: F3's store count W is not frozen (run the dev ladder, then the dated note)"
        )
    wall = {}
    t0 = time.perf_counter()
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.readout.interaction_learning import _CONV_TEST, _utterance_key

    wall["import"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = make_brain(seed=seed)
    ixn = b.brain._ixn
    wall["build"] = time.perf_counter() - t0
    a, bb, teach_a, teach_b, oracle = split_and_order(seed, offset=1 if red == "K1" else 0)
    if red == "K3":  # plant one held-out line of a B intent into session B's first rep
        teach_b[0].append((_CONV_TEST[bb[0]][0], bb[0]))
    t0 = time.perf_counter()
    taught = []
    for rep in teach_a + (teach_b if arm.ps == "AB" else []):
        for line, lab in rep:
            ixn.learn_turn(line, firewall_intent=lab)
            taught.append(_utterance_key(line))
    wall["teach"] = time.perf_counter() - t0
    d0 = state_digest(b)

    log = {"presentations": 0, "drawn": [], "k8a": [], "draws": 0, "in_guard": False}
    real_argmax, real_interact, real_replay = ixn._argmax_label, ixn.interact, ixn.replay

    def counted_argmax(code):
        log["presentations"] += 1
        return real_argmax(code)

    def logged_interact(stimulus, feedback, _record=True, _key=None):
        if not _record:
            log["drawn"].append(_key)
            return guarded(real_interact, stimulus, feedback, _record, _key)
        return real_interact(stimulus, feedback, _record=_record, _key=_key)

    # K8a per draw (Codex cp2 #10): each incumbent replay, fwd, Rec and MASSED draw has its own
    # chronology snapshot, so a change one draw makes and a later draw undoes is still caught.
    def guarded(fn, *args):
        if log["in_guard"]:
            return fn(*args)
        log["in_guard"] = True
        try:
            before = chrono(b)
            out = fn(*args)
            log["draws"] += 1
            if red == "K8a" and log["draws"] <= 2:  # the red: +1 on one draw, -1 on the next
                b._step += 1 if log["draws"] == 1 else -1
            if red == "K8a-obs" and log["draws"] == 1:  # the red: the grower is fed one percept
                from snn_data_lake.growth.neurogenesis_engine import _mode_features

                b.neuro.observe(_mode_features(0, 5), 0)
            after = chrono(b)
            if after != before:
                log["k8a"].append({"before": before, "after": after})
            return out
        finally:
            log["in_guard"] = False

    def logged_replay(k=8):
        return real_replay(k)  # each of its draws reaches logged_interact, which checks it

    if arm.pre == "oracle":
        for line, lab in oracle:
            ixn.learn_turn(line, firewall_intent=lab)
    elif arm.pre == "stores":
        for i in range(w):
            b.attractor.store(b._patterns[i % 3], reps=1)
        b._stored_any = True
    elif arm.pre == "massed":  # EXT's n x K draws, delivered at WAKE through the full deposit path
        twin = random.Random()
        twin.setstate(ixn.rng.getstate())
        ring = list(ixn._replay)

        for _ in range(BUDGETS["EXT"] * K):
            stim, lab, key = ring[twin.randrange(len(ring))]
            guarded(ixn.interact, list(stim), lab, False, key)
    d1 = state_digest(b)

    for cap, on in zip(("replay", "benna_fusi", "structural"), arm.caps):
        if not on:
            b.neuter_sleep(cap)
    att_calls = []
    real_sleep = b.attractor.sleep

    def att_sleep(seed, episodes=3, learn=True, record=None):
        if red == "K13":
            episodes -= 1
        att_calls.append([seed, episodes])
        if arm.attractor == "none":
            return None
        return real_sleep(seed, episodes=episodes, learn=(arm.attractor == "learn"), record=record)

    b.attractor.sleep = att_sleep
    ixn._argmax_label, ixn.interact, ixn.replay = counted_argmax, logged_interact, logged_replay
    twin = random.Random()
    twin.setstate(ixn.rng.getstate())
    b_pool = [e for e in ixn._replay if e[1] in bb]
    weights_d1 = weights_digest(ixn)
    tick = {"presentations": [], "deposits": [], "turnover": [], "cacc": []}
    c_before = chrono(b)
    t0 = time.perf_counter()
    b._enter_sleep()
    wall["enter_sleep"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t in range(n):
        p0, dep0 = log["presentations"], b._plasticity.consolidation_calls()
        ring = ixn._replay
        if arm.action == "fwd":
            for j in range(K):
                stim = ring[twin.randrange(len(ring))][0]
                if red == "K2" and t % 10 == 0 and j == 0:
                    continue  # the red: one fwd draw dropped per 10 ticks
                if red == "K6" and t == 0 and j == 0:
                    ixn.interact(list(stim), ixn.respond(stim), _record=False)  # the red
                    continue
                guarded(ixn.respond, stim)
        elif arm.action == "rec":
            for _ in range(K):
                stim, lab, key = b_pool[ixn.rng.randrange(len(b_pool))]
                ixn.interact(list(stim), lab, _record=False, _key=key)  # guarded per draw
        b._sleep_tick()
        tick["presentations"].append(log["presentations"] - p0)
        tick["deposits"].append(b._plasticity.consolidation_calls() - dep0)
        tick["turnover"].append((b._structural_report or {}).get("turnover"))
        tick["cacc"].append(
            round(b._consolidation_acc * 36) if b._stored_any else None
        )  # 3 patterns x 12 cues (P-5)
    wall["bout"] = time.perf_counter() - t0
    # DELETE the instance wrappers (never re-assign the bound originals): a bound method left in the
    # learner's __dict__ is refused by the G0(c) checkpoint encoder, so the save at _enter_wake would be
    # refused and the restore would read a fresh learner (measured on the first smoke run; K5 caught it).
    for name in ("_argmax_label", "interact", "replay"):
        del ixn.__dict__[name]
    del b.attractor.__dict__["sleep"]
    b._enter_wake()
    if red == "K8b":
        b._enter_wake()
    c_after = chrono(b)
    items = heldout(a) + heldout(bb)
    inproc = score(ixn, items)
    record = {
        "family": family,
        "arm": arm_name,
        "seed": seed,
        "budget": n,
        "stores": w if arm.pre == "stores" else None,
        "red": red,
        "a": a,
        "b": bb,
        "d0": d0,
        "d1": d1,
        "weights_d1": weights_d1,
        "weights_post": weights_digest(ixn),
        "taught": taught,
        "ring_keys": [e[2] for e in ixn._replay],
        "heldout_keys": [_utterance_key(line) for line, _ in heldout(a) + heldout(bb)],
        "drawn": log["drawn"],
        "tick": tick,
        "stored_any": bool(b._stored_any),
        "att_calls": att_calls,
        "k8a": log["k8a"],
        "chrono_before": c_before,
        "chrono_after": c_after,
        "inproc": inproc,
        "n_a": len(heldout(a)),
        "wall": wall,
    }
    t0 = time.perf_counter()
    b.close()
    wall["close"] = time.perf_counter() - t0
    return record


def run_evaluate(seed, red=None):
    """The fresh-process restore + held-out evaluation, learning off (the `evaluate` child)."""
    home = require_scratch_home()
    if red == "K5":  # the checkpoint is gone before the restore (a scratch home only)
        lake = os.path.join(home, "brain_lake.bin")
        if os.path.exists(lake):
            os.remove(lake)
    t0 = time.perf_counter()
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=seed)
    ixn = b.brain._ixn
    a, bb, *_ = split_and_order(seed, offset=1 if red == "K1" else 0)  # the bout's own split
    items = heldout(a) + heldout(bb)
    before = eval_digest(ixn)
    if red == "K4":  # the red: evaluate through interact (a learning path)
        from snn_data_lake.readout.interaction_learning import text_features

        got = [int(ixn.interact(text_features(line, ixn.n_in), lab) == lab) for line, lab in items]
    else:
        got = score(ixn, items)
    after = eval_digest(ixn)
    mar = margin(ixn, items)
    b.close()
    return {
        "restored": got,
        "margin": mar,
        "eval_before": before,
        "eval_after": after,
        "eval_s": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------------- the parent orchestrator
def _child(args, home, timeout=CHILD_TIMEOUT_S, err_path=None):
    env = dict(os.environ)
    env.update({"NORD_HOME": home, "NORD_PERSIST": "1", "PYTHONUTF8": "1"})
    env.pop("NORD_SELFMOD", None)
    cmd = [sys.executable, "-m", "snn_data_lake.instruments.sleep_plasticity_ab", *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if (
        err_path and proc.stderr
    ):  # every refusal the brain prints (e.g. a refused learner save) is kept
        with open(err_path, "a", encoding="utf-8") as fh:
            fh.write(f"--- {args[0]}\n{proc.stderr}")
    if proc.returncode != 0:
        raise RuntimeError(f"child {args[0]} failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# distinct-from: gates/move_gate.py::_rl1_run -- runs its private embedded R-L1 script once per arm; also
#   read: instruments/everyday_battery.py::run_seed (one in-process scratch brain, persist=False),
#   faculties/agency_monitor.py, evidence_accumulation.py, global_workspace.py (in-process arms) -- none
#   runs TWO processes on one PERSISTENT home, which is the restore path R-R3 reads (bout, then evaluate).
def run_arm(out, family, arm_name, seed, budget=None, red=None, stores=None):
    """One arm-seed end to end: a guarded output dir, a throwaway home, the bout child, the evaluate child,
    K14 around both. Codex cp2 #1: `guard_out` runs BEFORE any mkdir / write / delete, and the home comes
    from the shared `throwaway_home` (guarded before NORD_HOME is set, resolved, removed after)."""
    from snn_data_lake.substrate.home import throwaway_home

    guard_out(out)
    tag = (
        f"{family}_{arm_name}_{seed}"
        + (f"_n{budget}" if budget else "")
        + (f"_w{stores}" if stores is not None else "")
        + (f"_red-{red}" if red else "")
    )
    path = os.path.join(out, "records", tag + ".json")
    if os.path.exists(path):  # a completed cell is never re-run (a TIMEOUT re-runs the SAME seed)
        with open(path, encoding="ascii") as fh:
            return json.load(fh)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    homes = os.path.join(out, "homes")
    os.makedirs(homes, exist_ok=True)
    live_before = lake_snapshot(LIVE_LAKE)
    bout_args = ["bout", family, arm_name, str(seed), str(budget or ""), red or ""]
    bout_args.append("" if stores is None else str(stores))
    err = path[: -len(".json")] + ".stderr.txt"
    with throwaway_home(live_home=LIVE_LAKE, parent=homes) as home:
        rec = _child(bout_args, home, err_path=err)
        rec.update(_child(["evaluate", str(seed), red or ""], home, err_path=err))
    rec["stderr_lines"] = 0
    if os.path.exists(err):
        with open(err, encoding="utf-8") as fh:
            rec["stderr_lines"] = sum(1 for _ in fh)
    ok, changed = live_lake_untouched(LIVE_LAKE, live_before)
    rec["live_lake_untouched"] = ok
    rec["live_lake_changed"] = changed
    with open(path, "w", encoding="ascii") as fh:
        json.dump(rec, fh)
    if not ok:
        raise RuntimeError(f"K14 RED: the live lake changed during {tag}: {changed[:5]}")
    return rec


# distinct-from: interaction_learning.py -- g0_campaign / g0b_campaign run G0's OWN seed/arm registry in
#   subprocesses; also read: binding_instrument.py::t080_seed (one seed's binder reading),
#   reentrant_workspace.py::_deliver (broadcast), brainstem arm_config / serial_arm (joint tokens) -- none
#   maps an R3 family to its seed set or lists the arms its reading requires.
def check_set(family, set_name):
    """Codex cp2 #5: a family reads only its OWN confirmatory set or the shared EXPOSED dev set."""
    if set_name not in ("dev", family):
        raise ValueError(
            f"refused: family {family} may use set 'dev' or '{family}', not '{set_name}'"
        )
    return SEED_SETS[set_name]


def registered_arms(family, set_name):
    """Every arm a family's reading REQUIRES, twins included (Codex cp2 #7); dev-only arms on dev only."""
    return [a for a in FAMILIES[family] if set_name == "dev" or (family, a) not in DEV_ONLY_ARMS]


def run_family(out, family, set_name, arms=None):
    seeds = check_set(family, set_name)
    names = arms or registered_arms(family, set_name)
    for seed in seeds:
        for name in names:
            t0 = time.perf_counter()
            run_arm(out, family, name, seed)
            print(f"{family} {name} {seed} {time.perf_counter() - t0:.1f}s", flush=True)


# ------------------------------------------------------------------------------------- the statistic (P-8)
class Verdict(NamedTuple):
    name: str
    kind: str  # "sup" | "ni" | "ctl"
    diffs: tuple
    met: bool
    degenerate: bool
    t: float
    fpr: float  # operative: 2^8 flips under the ACTUAL verdict rule
    ref_fpr: float  # reference: paired_contrast.contrast's own rate (the incumbent t-test rule)
    mean: float
    sd: float
    lower95: float
    mde: float


def _met(diffs, kind):
    if len(set(diffs)) <= 1:  # degenerate: ruling 9 (superiority) / 9a (non-inferiority)
        return kind == "ni" and diffs[0] > 0
    return paired_t(diffs) > T_BAR


def verdict(name, diffs, seeds, kind="sup"):
    """One contrast. Halts on n != 8 or a seed outside R3's block (K12); never averages a partial list."""
    if len(diffs) != S or len(seeds) != S:
        raise ValueError(
            f"{name}: {len(diffs)} values / {len(seeds)} seeds, the contrast needs exactly {S}"
        )
    if any(type(s) is not int or not SEED_BLOCK[0] <= s <= SEED_BLOCK[1] for s in seeds):
        raise ValueError(f"{name}: a seed outside R3's block {SEED_BLOCK}: {list(seeds)}")
    if (
        len(set(seeds)) != S
    ):  # Codex cp2 #5: eight entries from fewer independent seeds is not S = 8
        raise ValueError(f"{name}: repeated seeds {list(seeds)}")
    d = tuple(float(x) for x in diffs)
    flips = sum(
        _met(tuple(-x if (m >> i) & 1 else x for i, x in enumerate(d)), kind) for m in range(1 << S)
    )
    mean = sum(d) / S
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (S - 1))
    return Verdict(
        name, kind, d, _met(d, kind), len(set(d)) <= 1, paired_t(d), flips / float(1 << S),
        contrast(name, d).fpr, mean, sd, mean - T_BAR * sd / math.sqrt(S), T_BAR * sd / math.sqrt(S),
    )  # fmt: skip


# ----------------------------------------------------------------------------- decision tables (K11, pure)
def table_f1(checks_ok, c_sens, c_if, c1, c1_ni, c2):
    if not checks_ok:
        return "HALT (instrument defect, no verdict)"
    if not c_sens:
        return "REFUSED (blind harness: C-SENS NOT MET)"
    if not c1:
        return "NULL: sleep synaptic plasticity not shown on held-out at the medium budget"
    if not c1_ni:
        return "INTERFERENCE RED (C1 MET, C1-NI NOT MET)"
    if not c_if:
        return "MET; no interference was present to fix (C-IF NOT MET), C2 reported, not read"
    if c2:
        return "MET: old-experience replay carries the gain (C2 MET)"
    return "MET; no benefit over recency-only replay demonstrated (C2 NOT MET)"


def table_f2(checks_ok, k7_sighted, turnover_all_positive, c5, c5_ni, restore):
    """restore: True / False / None (None = the r2.v1 restart reading is still pending, X-15)."""
    if not checks_ok:
        return "HALT (instrument defect, no verdict)"
    if not k7_sighted:
        return "REFUSED (blind turnover detector)"
    if not turnover_all_positive:
        return "NOT READ (turnover 0 on some confirmatory seed)"
    if not c5:
        return "NULL"
    if not c5_ni:
        return "INTERFERENCE RED (C5 MET, its NI NOT MET)"
    if restore is False:
        return "NOT CLAIMED (gain not restorable)"
    if restore is None:
        return "MET; the restart reading waits on the accepted r2.v1 carrier"
    return "MET"


def table_f3(checks_ok, stored_any, c6):
    if not checks_ok:
        return "HALT (instrument defect, no verdict)"
    if not stored_any:
        return "NOT READ (no stored patterns)"
    if not c6:
        return "NULL"
    return (
        "MET: attractor sleep learning raises in-process pattern completion of its stored patterns"
    )


def table_f4(checks_ok, c7, c7_ni, c8, c8_ni, c9):
    if not checks_ok:
        return "HALT (instrument defect, no verdict)"
    if not c7:
        return "no budget step shown (MED vs SHORT NOT MET)"
    if not c7_ni:
        return "INTERFERENCE RED at MED (C7 MET, its NI NOT MET)"
    if not c8:
        return "MED credited over SHORT; the EXT step is not shown"
    if not c8_ni:
        return "MED credited; INTERFERENCE RED at EXT (C8 MET, its NI NOT MET)"
    if c9:
        return "EXT credited: a sleep-duration benefit (C9 MET)"
    return "EXT credited; no sleep-duration benefit over massed wake demonstrated (C9 NOT MET)"


TABLES = {"F1": table_f1, "F2": table_f2, "F3": table_f3, "F4": table_f4}
# K11: one constructed input per row of every table (the row's text prefix), each must be reached.
TABLE_REACH = {
    "F1": [
        ((False, 1, 1, 1, 1, 1), "HALT"), ((True, 0, 1, 1, 1, 1), "REFUSED"), ((True, 1, 1, 0, 1, 1), "NULL"),
        ((True, 1, 1, 1, 0, 1), "INTERFERENCE RED"), ((True, 1, 0, 1, 1, 1), "MET; no interference"),
        ((True, 1, 1, 1, 1, 1), "MET: old"), ((True, 1, 1, 1, 1, 0), "MET; no benefit"),
    ],
    "F2": [
        ((False, 1, 1, 1, 1, True), "HALT"), ((True, 0, 1, 1, 1, True), "REFUSED"),
        ((True, 1, 0, 1, 1, True), "NOT READ"), ((True, 1, 1, 0, 1, True), "NULL"),
        ((True, 1, 1, 1, 0, True), "INTERFERENCE RED"), ((True, 1, 1, 1, 1, False), "NOT CLAIMED"),
        ((True, 1, 1, 1, 1, None), "MET; the restart"), ((True, 1, 1, 1, 1, True), "MET"),
    ],
    "F3": [((False, 1, 1), "HALT"), ((True, 0, 1), "NOT READ"), ((True, 1, 0), "NULL"), ((True, 1, 1), "MET")],
    "F4": [
        ((False, 1, 1, 1, 1, 1), "HALT"), ((True, 0, 1, 1, 1, 1), "no budget"),
        ((True, 1, 0, 1, 1, 1), "INTERFERENCE RED at MED"), ((True, 1, 1, 0, 1, 1), "MED credited over"),
        ((True, 1, 1, 1, 0, 1), "MED credited; INTERFERENCE"), ((True, 1, 1, 1, 1, 1), "EXT credited: a"),
        ((True, 1, 1, 1, 1, 0), "EXT credited; no"),
    ],
}  # fmt: skip


def table_reach_ok():
    """K11: every row of every table has a constructed input that reaches it (exact-match on the prefix). The
    row INVENTORY is mechanical -- one row per `return` in the table's source -- so a row left without an
    input is a miss, not a silence."""
    import inspect

    misses = []
    for fam, fn in TABLES.items():
        n_rows = inspect.getsource(fn).count("return ")
        if len(TABLE_REACH.get(fam, ())) != n_rows:
            misses.append(
                (
                    fam,
                    f"{n_rows} rows, {len(TABLE_REACH.get(fam, ()))} constructed inputs",
                    None,
                    None,
                )
            )
    for fam, rows in TABLE_REACH.items():
        reached = [TABLES[fam](*args) for args, _ in rows]
        for (args, prefix), got in zip(rows, reached):
            if not got.startswith(prefix):
                misses.append((fam, args, prefix, got))
        if len(set(reached)) != len(rows):
            misses.append((fam, "two inputs reached one row", None, None))
    return not misses, misses


# ------------------------------------------------------------------------------- record checks (K1-K8, K13)
N_A, N_B = 8, 12  # the evaluation's fixed item counts (2.2): E_A 8 items, E_B 12, E_all 20
N_ALL = N_A + N_B


def endpoint_shape_ok(r):
    """Codex lens-r3-amend: a record's endpoints are 8 + 12 BINARY outcomes, in-process and restored, or
    nothing downstream may divide by an item count taken from one arm."""

    def binary(v):
        return (
            isinstance(v, list)
            and len(v) == N_ALL
            and all(type(x) is int and x in (0, 1) for x in v)
        )

    return r.get("n_a") == N_A and binary(r.get("restored")) and binary(r.get("inproc"))


def _e(rec, which, key="restored"):
    """Endpoint COUNTS (exact integers): all / A / B."""
    v, na = rec[key], rec["n_a"]
    return {"all": sum(v), "A": sum(v[:na]), "B": sum(v[na:])}[which]


def check_records(family, recs):
    """The record-level checks over one family's arm-seeds; returns {check: [failures]} (empty = green)."""
    fails = {k: [] for k in ("K1", "K2", "K3", "K4", "K5", "K6", "K8a", "K8b", "K13", "K14", "KE")}
    by_seed = {}
    for r in recs:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r
    for seed, arms in by_seed.items():
        d0 = {a: r["d0"] for a, r in arms.items() if FAMILIES[family][a].ps == "AB"}
        if len(set(d0.values())) > 1:
            fails["K1"].append((seed, "D0", d0))
        for one, two in (("ON", "ON-TWIN"), ("A-ONLY", "A-ONLY-TWIN")):
            if one in arms and two in arms:
                x, y = arms[one], arms[two]
                if (x["d0"], x["weights_post"], x["restored"]) != (
                    y["d0"],
                    y["weights_post"],
                    y["restored"],
                ):
                    fails["K1"].append((seed, f"{one} twin"))
        groups = {}
        for a, r in arms.items():
            g = FAMILIES[family][a].d1_group
            if g:
                groups.setdefault(g, set()).add(r["d1"])
        fails["K1"] += [(seed, "D1", g) for g, v in groups.items() if len(v) > 1]
        # KE (Codex lens-r3-amend): one ordered held-out identity per seed across every arm
        if len({tuple(r["heldout_keys"]) for r in arms.values()}) > 1:
            fails["KE"].append((seed, "held-out identity"))
    for r in recs:
        tag = (r["arm"], r["seed"])
        if not endpoint_shape_ok(r):
            fails["KE"].append(tag)
        if r["tick"]["presentations"] != [K] * r["budget"]:
            fails["K2"].append(tag)
        held = set(r["heldout_keys"])
        if held & (set(r["taught"]) | set(r["ring_keys"]) | set(r["drawn"])):
            fails["K3"].append(tag)
        if r["eval_before"] != r["eval_after"]:
            fails["K4"].append(tag)
        if (
            family != "F3" and r["inproc"] != r["restored"]
        ):  # P-5: F3 reads in-process (Codex cp2 #13)
            fails["K5"].append(tag)
        if r["arm"] in PLASTICITY_FREE and r["weights_d1"] != r["weights_post"]:
            fails["K6"].append(tag)
        if r["k8a"]:
            fails["K8a"].append(tag)
        cb, ca = r["chrono_before"], r["chrono_after"]
        # K8b (X-17), with `_step` ASSERTED unchanged rather than only printed (Codex cp2 #10; measured
        # unchanged on all 144 dev arm-seeds, so the stricter reading halts nothing valid)
        moved = (ca["nights"] - cb["nights"], ca["wake_ticks"], ca["arm_ticks"], ca["step"])
        if moved != (1, cb["wake_ticks"], cb["arm_ticks"], cb["step"]):
            fails["K8b"].append(tag)
        if not r.get("live_lake_untouched", False):
            fails["K14"].append(tag)
    if family == "F3":
        by = {}
        for r in recs:
            by.setdefault(r["seed"], []).append(r["att_calls"])
        fails["K13"] = [s for s, calls in by.items() if any(c != calls[0] for c in calls)]
    return fails


# --------------------------------------------------------------------------------- contrasts per family
def _d(by, seeds, arm, ctrl, which, eps=0):
    """Paired differences arm - control on an endpoint, as exact Fractions over the FIXED item count
    (KE has already halted any record whose shape differs; this refuses again rather than divide by one
    arm's length)."""
    n_items = {"all": N_ALL, "A": N_A, "B": N_B}[which]
    out = []
    for s in seeds:
        x, y = by[s][arm], by[s][ctrl]
        if not (endpoint_shape_ok(x) and endpoint_shape_ok(y)):
            raise ValueError(f"{arm} vs {ctrl} seed {s}: endpoint shape is not {N_A}+{N_B} binary")
        out.append(Fraction(_e(x, which) - _e(y, which), n_items) + eps)
    return out


def _cacc(r):
    vals = [v for v in r["tick"]["cacc"] if v is not None]
    return Fraction(sum(vals), 36 * len(vals)) if vals else Fraction(0)


def contrasts(family, recs, seeds):
    by = {}
    for r in recs:
        by.setdefault(r["seed"], {})[r["arm"]] = r
    out = []

    def add(name, kind, diffs):
        out.append(verdict(name, diffs, seeds, kind))

    if family == "F1":
        add("C-SENS", "ctl", _d(by, seeds, "ORACLE", "OFF", "A"))
        add("C-IF", "ctl", _d(by, seeds, "A-ONLY", "OFF", "A"))
        add("C1", "sup", _d(by, seeds, "ON", "OFF", "all"))
        add("C1-NI", "ni", _d(by, seeds, "ON", "OFF", "A", EPS_A))
        add("C2", "sup", _d(by, seeds, "ON", "Rec", "all"))
        add("C3", "sup", _d(by, seeds, "R", "OFF", "all"))
        add("C4", "sup", _d(by, seeds, "BF", "OFF", "all"))
    elif family == "F2":
        add("C5", "sup", _d(by, seeds, "S-ON", "S-OFF", "all"))
        add("C5-NI", "ni", _d(by, seeds, "S-ON", "S-OFF", "A", EPS_A))
    elif family == "F3":
        add("C6", "sup", [_cacc(by[s]["AT-ON"]) - _cacc(by[s]["AT-OFF"]) for s in seeds])
    elif family == "F4":
        add("C7", "sup", _d(by, seeds, "MED", "SHORT", "all"))
        add("C7-NI", "ni", _d(by, seeds, "MED", "SHORT", "A", EPS_A))
        add("C8", "sup", _d(by, seeds, "EXT", "MED", "all"))
        add("C8-NI", "ni", _d(by, seeds, "EXT", "MED", "A", EPS_A))
        add("C9", "sup", _d(by, seeds, "EXT", "MASSED", "all"))
    return out


# distinct-from: agency_monitor.py::_t079_refusals -- named joint/count refusals for one recording; also
#   read: everyday_battery.py::Fix (a recorded cell), prior_drift_harness.py::drift_records (per-statistic
#   drift), interaction_learning.py::g0b_campaign (G0's runner) -- none checks an arm x seed record SET.
def completeness(family, recs, seeds, set_name):
    """Codex cp2 #7: exactly one record for every registered arm x seed (both twins included) and nothing
    else; a missing, duplicated or foreign cell is listed. Run BEFORE any check or contrast."""
    want = {(a, s) for a in registered_arms(family, set_name) for s in seeds}
    got = [(r["arm"], r["seed"]) for r in recs]
    missing = sorted(want - set(got))
    dupes = sorted({c for c in got if got.count(c) > 1})
    foreign = sorted(set(got) - want)
    return (
        [("missing", c) for c in missing]
        + [("duplicate", c) for c in dupes]
        + [("foreign", c) for c in foreign]
    )


def decide_family(family, recs, seeds, k7_sighted=False, restore=None, set_name=None):
    incomplete = completeness(family, recs, seeds, set_name or family)
    if incomplete:
        return {
            "checks": {"COMPLETE": incomplete},
            "checks_ok": False,
            "contrasts": {},
            "row": "HALT (instrument defect, no verdict): the record set is incomplete",
        }
    fails = check_records(family, recs)
    checks_ok = not any(fails.values())
    if (
        not checks_ok
    ):  # a red check halts BEFORE any contrast is built (malformed input never divides)
        return {
            "checks": fails,
            "checks_ok": False,
            "contrasts": {},
            "row": "HALT (instrument defect, no verdict)",
        }
    v = {c.name: c for c in contrasts(family, recs, seeds)}
    m = {k: c.met for k, c in v.items()}
    if family == "F1":
        row = table_f1(checks_ok, m["C-SENS"], m["C-IF"], m["C1"], m["C1-NI"], m["C2"])
    elif family == "F2":
        on = [r for r in recs if r["arm"] == "S-ON"]
        pos = len(on) == S and all(
            sum(x for x in r["tick"]["turnover"] if x) / max(1, len(r["tick"]["turnover"])) > 0
            for r in on
        )
        row = table_f2(checks_ok, k7_sighted, pos, m["C5"], m["C5-NI"], restore)
    elif family == "F3":
        stored = all(r["stored_any"] for r in recs)
        row = table_f3(checks_ok, stored, m["C6"])
    else:
        row = table_f4(checks_ok, m["C7"], m["C7-NI"], m["C8"], m["C8-NI"], m["C9"])
    return {"checks": fails, "checks_ok": checks_ok, "contrasts": v, "row": row}


PRIMARY = {
    "F1": ("C1", "C1-NI"),
    "F2": ("C5", "C5-NI"),
    "F3": ("C6",),
    "F4": ("C7", "C7-NI", "C8", "C8-NI", "C9"),
}


def census(read):
    """P-6 / Codex 27: the count is derived from the test IDs actually read, by category."""
    cats = {"superiority": [], "positive controls": [], "NI bars": []}
    for c in read:
        key = (
            "NI bars"
            if c.kind == "ni"
            else "positive controls"
            if c.kind == "ctl"
            else "superiority"
        )
        cats[key].append(
            f"{c.name}={'MET' if c.met else 'NOT MET'}" + (" (degenerate)" if c.degenerate else "")
        )
    lines = [f"{k} read: {len(v)} {v}" for k, v in cats.items()]
    lines.append(
        f"tests read in all: {sum(len(v) for v in cats.values())} ('18' names only C1-C18)"
    )
    return lines


def claims_report(family, contrasts, recs):
    """P-6 (Codex cp2 #12): the claims made through the family's ONE declared primary (PRIMARY, read here),
    beside the tests read; plus the endpoint quantum and the secondary correct-label margin."""
    prim = PRIMARY[family]
    made = [n for n in prim if n in contrasts and contrasts[n].met]
    lines = [
        f"primary met: {len(made)} of {len(prim)} {list(prim)} -> {made}; claim "
        + ("STANDS" if len(made) == len(prim) else "NOT MADE")
        + " (a family claims only through its whole primary)",
        f"endpoint quantum E_all = 1/{len(recs[0]['restored'])}",
        "secondary correct-label margin (mean over seeds, per arm): "
        + str(
            {
                a: round(
                    sum(r["margin"] for r in recs if r["arm"] == a)
                    / sum(r["arm"] == a for r in recs),
                    4,
                )
                for a in sorted({r["arm"] for r in recs})
            }
        ),
    ]
    return lines


# --------------------------------------------------------------------- K7: the turnover detector, sighted
def run_k7(seed):
    """The A/A stable-sign schedule must read exactly 0.0 and a planted sign flip > 0 (in this process)."""
    require_scratch_home()
    from snn_data_lake.engine.brain import make_brain

    out = {}
    for planted in (False, True):
        b = make_brain(seed=seed, persist=False)  # the two schedules must not restore each other
        ixn = b.brain._ixn
        a, bb, teach_a, teach_b, _ = split_and_order(seed)
        for rep in teach_a + teach_b:
            for line, lab in rep:
                ixn.learn_turn(line, firewall_intent=lab)
        b.neuter_sleep("replay").neuter_sleep("benna_fusi")
        b._sleep_tick()  # first sight records birth signs
        if planted:
            i, c = next(
                (i, c) for i, n in enumerate(ixn.neurons) for c, w in enumerate(n.w) if w > 1e-6
            )
            ixn.neurons[i].w[c] = -abs(ixn.neurons[i].w[c])
        b._sleep_tick()
        out["planted" if planted else "aa"] = (b._structural_report or {}).get("turnover")
        b.close()
    out["sighted"] = out["aa"] == 0.0 and (out["planted"] or 0) > 0
    return out


# ------------------------------------------------- F3's W ladder (the dated amendment 2026-10-04, dev only)
# distinct-from: report_order_probe.py -- t081_pin_reading re-derives a committed pin's mean over seeds;
#   also read: binding_instrument.py::t080_pin_reading and recursive_tom.py::t082_pin_reading (the same
#   pin idiom), access_consciousness.py::ignited (an ignition predicate), self_improvement_suite.py::
#   _rung_metaplast (L8's reach) -- none selects a constructed-workload size on a positive control.
def ladder_qualifies(counts_per_seed):
    """A rung qualifies iff AT-NONE reads below 35/36 on ALL 8 dev seeds; each seed's reading is the mean of
    its per-tick correct counts (of 36) over the bout, compared exactly as a Fraction."""
    return len(counts_per_seed) == S and all(
        Fraction(sum(c), 36 * len(c)) < Fraction(35, 36) for c in counts_per_seed
    )


def run_ladder(out):
    """AT-NONE only, dev seeds only, rungs in ascending order; stops at the first qualifying W. Reads no
    AT-ON / AT-OFF cell, so C6 is never tuned on its own outcome. None = no rung qualifies -> F3 REFUSED."""
    rungs = []
    for w in F3_W_LADDER:
        counts = []
        for seed in SEED_SETS["dev"]:
            r = run_arm(out, "F3", "AT-NONE", seed, stores=w)
            counts.append([v for v in r["tick"]["cacc"] if v is not None])
        ok = ladder_qualifies(counts)
        means = [round(sum(c) / (36 * len(c)), 4) for c in counts]
        rungs.append({"w": w, "means": means, "qualifies": ok})
        print(f"ladder W={w}: {means} qualifies={ok}", flush=True)
        if ok:
            return {"w": w, "rungs": rungs}
    return {"w": None, "rungs": rungs}


# ------------------------------------------------------------------------------------------- self-check
def main():
    """Offline self-check (no engine import): the statistic, the tables, the census, the K14 decoy red."""
    import tempfile

    seeds = SEED_SETS["dev"]
    pos = verdict("constant positive", [Fraction(1, 20)] * 8, seeds)
    assert pos.degenerate and not pos.met and pos.fpr == 8 / 256 and pos.ref_fpr == 0.0, pos
    zero = verdict("constant zero", [0] * 8, seeds)
    assert zero.degenerate and not zero.met and zero.fpr == 0.0, zero
    ni = verdict("NI lossless", [EPS_A] * 8, seeds, kind="ni")
    assert ni.met and ni.fpr == 9 / 256, (
        ni
    )  # all-positive (degenerate, > 0) + the eight 7+/1- flips
    ni0 = verdict("NI one item lost", [Fraction(0)] * 8, seeds, kind="ni")
    assert not ni0.met, ni0  # INTERFERENCE RED by 2.4's boundary sentence
    up = verdict("all positive", [0.2, 0.3, 0.25, 0.4, 0.15, 0.35, 0.3, 0.2], seeds)
    assert up.met and up.fpr == up.ref_fpr == contrast("x", up.diffs).fpr, up
    for bad in (([0.1] * 7, seeds[:7]), ([0.1] * 8, (300,) + seeds[1:])):  # K12 reds
        try:
            verdict("red", *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"K12 red did not fire: {bad}")
    ok, misses = table_reach_ok()
    assert ok, misses
    lines = census([pos, ni, verdict("C-SENS", [0.1, 0.2] * 4, seeds, kind="ctl")])
    assert lines[-1].startswith("tests read in all: 3"), lines
    with (
        tempfile.TemporaryDirectory() as decoy
    ):  # K14's red: a decoy path, never the live lake (P-3)
        with open(os.path.join(decoy, "lake.bin"), "w") as fh:
            fh.write("x")
        before = lake_snapshot(decoy)
        assert live_lake_untouched(decoy, before)[0]
        time.sleep(0.01)
        with open(os.path.join(decoy, "lake.bin"), "a") as fh:
            fh.write("y")
        assert not live_lake_untouched(decoy, before)[0], "K14 red did not fire"
    print(
        f"self-check OK: degenerate superiority NOT MET (operative FPR 8/256, reference {pos.ref_fpr}); "
        f"NI lossless MET (FPR 9/256); NI d'=0 NOT MET; K12 refuses n=7 and seed 300; every table row reached "
        f"({sum(len(v) for v in TABLE_REACH.values())} rows); census derived; K14 decoy red fires"
    )


def _cli(argv):
    cmd = argv[0] if argv else ""
    if cmd == "bout":
        family, arm, seed, budget, red, stores = argv[1:7]
        rec = run_bout(
            family,
            arm,
            int(seed),
            int(budget) if budget else None,
            red or None,
            int(stores) if stores else None,
        )
        print(json.dumps(rec))
    elif cmd == "evaluate":
        print(json.dumps(run_evaluate(int(argv[1]), argv[2] or None)))
    elif cmd == "k7child":
        print(json.dumps(run_k7(int(argv[1]))))
    else:
        import argparse

        p = argparse.ArgumentParser(prog="sleep_plasticity_ab")
        p.add_argument("cmd", choices=["run", "decide", "k7", "red", "ladder"])
        p.add_argument("--family", choices=sorted(FAMILIES))
        p.add_argument("--set", default="dev", choices=sorted(SEED_SETS))
        p.add_argument("--out", required=True)
        p.add_argument("--arms", nargs="*")
        p.add_argument("--name")
        a = p.parse_args(argv)
        _main_cmd(a)


RECORD_KEYS = frozenset(
    (
        "family arm seed budget stores red d0 d1 weights_d1 weights_post taught ring_keys heldout_keys "
        "drawn tick stored_any att_calls k8a chrono_before chrono_after inproc restored n_a wall "
        "eval_before eval_after eval_s margin live_lake_untouched"
    ).split()
)


# distinct-from: substrate/home.py::load_state -- reads a JSON state blob out of the one lake; this reads
#   the instrument's own record FILES from its output directory (never the lake). Site read.
def _load(out, family, seeds):
    """Only regular `<family>_*.json` record files (never the kept `.stderr.txt` logs, Codex cp2 #11), each
    schema-checked; a malformed record raises rather than being skipped."""
    recs = []
    d = os.path.join(out, "records")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        p = os.path.join(d, f)
        regular = f.endswith(".json") and "_red-" not in f and os.path.isfile(p)
        if f.startswith(family + "_") and regular:
            with open(p, encoding="ascii") as fh:
                r = json.load(fh)
            missing = RECORD_KEYS - set(r)
            if missing or r["family"] != family:
                raise ValueError(f"malformed record {f}: missing {sorted(missing)}")
            registered = BUDGETS[FAMILIES[family][r["arm"]].budget]
            if (
                r["seed"] in seeds
                and (family, r["arm"]) not in DEV_ONLY_ARMS
                and r["budget"] == registered
                and r.get("stores") == (F3_STORES if family == "F3" else None)
            ):
                recs.append(r)  # a smoke / red cell at another budget never enters a contrast
    return recs


def _main_cmd(a):
    from snn_data_lake.substrate.home import throwaway_home

    guard_out(a.out)  # Codex cp2 #1: before the first mkdir
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "run":
        run_family(a.out, a.family, a.set, a.arms)
    elif a.cmd == "k7":
        homes = os.path.join(a.out, "homes")
        os.makedirs(homes, exist_ok=True)
        before = lake_snapshot(LIVE_LAKE)
        with throwaway_home(live_home=LIVE_LAKE, parent=homes) as home:
            res = _child(["k7child", str(RED_SEEDS[0])], home)
        res["live_lake_untouched"] = live_lake_untouched(LIVE_LAKE, before)[0]
        with open(os.path.join(a.out, "k7.json"), "w", encoding="ascii") as fh:
            json.dump(res, fh)
        print(res)
    elif a.cmd == "ladder":
        res = run_ladder(a.out)
        with open(os.path.join(a.out, "f3_ladder.json"), "w", encoding="ascii") as fh:
            json.dump(res, fh)
        print(json.dumps(res))
    elif a.cmd == "red":
        fam, arm, budget = {
            "K1": ("F1", "ON", 2), "K2": ("F1", "OFF", 10), "K3": ("F1", "ON", 2), "K4": ("F1", "ON", 2),
            "K5": ("F1", "ON", 2), "K6": ("F1", "OFF", 2), "K8a": ("F1", "ON", 2), "K8b": ("F1", "ON", 2),
            "K8a-obs": ("F1", "ON", 2), "K13": ("F3", "AT-OFF", 2),
        }[a.name]  # fmt: skip
        seed = RED_SEEDS[1]
        w = F3_W_LADDER[0] if fam == "F3" else None  # a red needs stores, never the frozen W
        base = run_arm(a.out, fam, arm if fam != "F3" else "AT-ON", seed, budget=budget, stores=w)
        redrec = run_arm(a.out, fam, arm, seed, budget=budget, red=a.name, stores=w)
        twin = (
            dict(redrec, arm="ON-TWIN") if a.name == "K1" else redrec
        )  # the split-offset build as ON's twin
        fails = check_records(fam, [base, twin] if a.name in ("K1", "K13") else [redrec])
        check = a.name.split("-")[0]  # K8a-obs is read by K8a
        fired = bool(fails[check])
        print(f"red {a.name}: {'FIRED' if fired else 'DID NOT FIRE'} {fails[check]}")
        if not fired:
            raise SystemExit(1)
    elif a.cmd == "decide":
        seeds = check_set(a.family, a.set)
        recs = _load(a.out, a.family, seeds)
        k7 = None
        kp = os.path.join(a.out, "k7.json")
        if os.path.exists(kp):
            with open(kp, encoding="ascii") as fh:
                k7 = json.load(fh).get("sighted")
        res = decide_family(a.family, recs, seeds, k7_sighted=bool(k7), set_name=a.set)
        for name, c in res["contrasts"].items():
            print(
                f"{name:6s} {c.kind:3s} met={c.met} deg={c.degenerate} t={c.t:.3f} fpr={c.fpr:.4f} "
                f"(ref {c.ref_fpr:.4f}) mean={c.mean:.4f} sd={c.sd:.4f} lb95={c.lower95:.4f} mde={c.mde:.4f}"
            )
        print("checks:", {k: len(v) for k, v in res["checks"].items()})
        if a.set == "dev":  # Codex cp2 #5: a dev reading is never printed as a claim
            print("DEV READING (EXPOSED seeds, never a verdict) -- table row:", res["row"])
        else:
            print("table:", res["row"], "(set", a.set + ")")
        for line in census(res["contrasts"].values()) + claims_report(
            a.family, res["contrasts"], recs
        ):
            print(line)
        walls = {}
        for r in recs:
            for k, v in dict(r["wall"], restore_and_evaluate=r["eval_s"]).items():
                walls.setdefault(k, []).append(v)
        print("wall mean s:", {k: round(sum(v) / len(v), 3) for k, v in walls.items()})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli(sys.argv[1:])
    else:
        main()
