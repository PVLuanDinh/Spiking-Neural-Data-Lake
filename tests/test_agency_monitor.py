"""The reds for T079 -- specs/022 Phase 4 / C4, the agency instrument
(evidence/t079-agency.md 2.4).

WRITTEN AND COMMITTED **RED** against the 2.9 step-2 skeleton: every failure is an
ASSERTION on a constructed input, never a collection ImportError (the shape T077's panel
named and T078 repeated). The instruments go green at 2.9 step 4, when each red is ALSO
reddened IN FACT by its own mutated scorer.

  R1 THE FLOOR IS A FLOOR -- a constructed COUPLED channel (fb = 0.9 u + noise, 600
     passages) reads its mean controllability ABOVE its shuffle floor; a DECOUPLED
     channel (fb independent of u) reads AT or BELOW it. MUTATION: a floor drawn from the
     UN-shuffled sequence, which makes the decoupled channel read "above".
  R2 THE MACHINE'S HYSTERESIS -- on the module's own `_experiment` construction
     (healthy -> decoupled -> recovered) a single LOW reading while 'ok' does not flip to
     'lost' and a single HIGH reading while 'lost' does not flip to 'ok'. MUTATION:
     delay = recover_n = 0, which flips on one reading.
  R3 AN UNEXERCISED CHANNEL IS UNSCORED, NEVER LOST -- u == 0 on every passage reads
     `scored` False, never 'lost', and (r) REFUSES the recording; the four exercised
     joints of the same recording read `scored` True. MUTATION: var_floor := 0, which
     scores the dead channel and lets (r) read.
  R4 THE PARSER -- `intended_deltas` reads +/-6 on the named joint and 0 on the other
     four; ENGAGE / RELEASE / HOLD / HOME / EMERGENCY_STOP and MOVETO read all 0; a
     ROTATE token naming a joint outside `arm_config.ROTATABLE` is REFUSED, not parsed;
     `live_arm._joint_obs_delta` agrees on shoulder_lift. MUTATION: a parser that accepts
     any joint name.
  R5 THE PER-JOINT FOLD -- `proprioception_deltas` wraps EACH joint (359.9 -> 0.1 folds
     to +0.2 deg, the T037 seam red, per joint) and `proprioception_delta` equals its mean
     BIT-FOR-BIT on 200 random triples. MUTATION: the wrap dropped.
  R6 THE DISSOCIATION ON A CONSTRUCTED BODY -- five synthetic channels through the
     SHIPPED `AgencyMonitor`, one decoupled after passage 300 by an additive
     U(-2, 2) x command-scale disturbance: (a1) MET, (a2) MET on the four, (a3) MET; the
     twin with NO decoupling reads (a1) NOT MET and (a3) NOT MET. MUTATION: c_j replaced
     by 1 - c_j on the shoved arm, which inverts (a1).

Every red also asserts `t079_reds()[name]`, the module's own self-check line: 2.4 requires
the mutation to be applied IN THE TEST *and* in `t079_reds()`, so a gate run that never
executes pytest still runs every red. At this commit `t079_reds()` returns the skeleton's
six Falses, which is one of the assertions that makes this file red.

No brain, no lake, no network: R1-R6 all run on constructed inputs. The live reading (a
scratch brain, seeds 0-7, 1,200 ticks x 2 arms) is NOT here -- the pin is written once by
`--t079-campaign`.

  set PYTHONUTF8=1 & PYTHONPATH=src
  .venv-lake/Scripts/python.exe -m pytest tests/test_agency_monitor.py -q --no-cov
"""

import random

import pytest

from snn_data_lake.brainstem import arm_config, live_arm
from snn_data_lake.cortex.agency_monitor import (
    T079_DEADBAND,
    T079_JOINTS,
    T079_SHOVE_JOINT,
    T079_VAR_FLOOR,
    T079_WARMUP,
    AgencyMonitor,
    _experiment,
    t079_contrasts,
    t079_floor,
    t079_read,
    t079_reds,
)
from snn_data_lake.substrate.afferent_boundary import proprioception_delta
from snn_data_lake.substrate.efferent_boundary import intended_deltas, proprioception_deltas

#: The reds' shuffle count. BUILDER-INTRODUCED and disclosed: 2.1 fixes n_shuffle = 2,000
#: for the CAMPAIGN's floors, and `t079_reds()` runs on every run_selfchecks pass where
#: 2,000 draws x 5 joints x 17 recordings would dominate the gate. 2.4 fixes no draw count
#: for the reds themselves.
REDS_N_SHUFFLE = 200

#: R1's width, VERBATIM from 2.4 ("600 passages").
R1_PASSAGES = 600

#: R6's width. BUILDER-INTRODUCED and disclosed: 2.4 fixes the decoupling point (passage
#: 300) and the disturbance (U(-2, 2) x command scale) but not n. 900 leaves 600 decoupled
#: passages, which is what the state machine needs to reach 'lost' -- the EMA (alpha 0.02,
#: ~50-observation memory) has to decay THROUGH the coupled history before `delay` = 48
#: consecutive low readings can even start counting.
R6_PASSAGES = 900
R6_DECOUPLE_AT = 300
R6_AMP = 2.0
R6_SEEDS = tuple(range(8))


@pytest.fixture(scope="module")
def reds():
    """`t079_reds()` once per module -- the in-command control every red also asserts."""
    return t079_reds(n_shuffle=REDS_N_SHUFFLE)


# ---------------------------------------------------------------------------------------
# Constructed inputs. These live here at 2.9 step 3 and MOVE INTO the module at step 4,
# where `t079_reds()` needs them so a gate run that never executes pytest still runs the
# reds on the same streams (T078's shape).
# ---------------------------------------------------------------------------------------
def _rows_from_channels(u_cols, ds_cols, monitor=None):
    """Feed constructed per-joint channels through the SHIPPED `AgencyMonitor` and return
    rows in the recorder's own shape (2.1). `monitor` is the mutation hook."""
    mon = AgencyMonitor(seed=0) if monitor is None else monitor
    rows = []
    for p in range(len(u_cols[0])):
        u = [float(col[p]) for col in u_cols]
        ds = [float(col[p]) for col in ds_cols]
        for k, j in enumerate(T079_JOINTS):
            mon.observe(j, u[k], ds[k])
        snaps = [mon.snapshot(j) for j in T079_JOINTS]
        rows.append(
            {
                "t": p,
                "kind": "awake",
                "awake": True,
                "action": None,
                "u": u,
                "ds": ds,
                "ctrl": [s["ctrl"] for s in snaps],
                "status": [s["status"] for s in snaps],
                "n": [s["n"] for s in snaps],
                "latched": False,
                "tap_errors": 0,
            }
        )
    return rows


def _coupled_decoupled_channels(n, seed, gain=0.9, noise=0.05):
    """R1's two channels, in the five-joint frame: joint 0 COUPLED (fb = gain u + noise),
    joint 1 DECOUPLED (fb independent of u), joints 2-4 coupled filler."""
    rng = random.Random(seed)
    u_cols, ds_cols = [], []
    for k in range(len(T079_JOINTS)):
        u = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        if k == 1:
            ds = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        else:
            ds = [gain * c + noise * rng.uniform(-1.0, 1.0) for c in u]
        u_cols.append(u)
        ds_cols.append(ds)
    return u_cols, ds_cols


def _dead_channel_channels(n, seed):
    """R3's recording: joint 0 is NEVER commanded (u == 0 on every passage, its feedback
    still moving); the other four are exercised and coupled."""
    rng = random.Random(seed)
    u_cols, ds_cols = [], []
    for k in range(len(T079_JOINTS)):
        if k == 0:
            u_cols.append([0.0] * n)
            ds_cols.append([rng.uniform(-1.0, 1.0) for _ in range(n)])
            continue
        u = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        u_cols.append(u)
        ds_cols.append([0.9 * c + 0.05 * rng.uniform(-1.0, 1.0) for c in u])
    return u_cols, ds_cols


def _body_channels(seed, shove=None, amp=R6_AMP, n=R6_PASSAGES, at=R6_DECOUPLE_AT):
    """R6's constructed body: five channels commanded in {-step, 0, +step} with the servo
    following at gain 0.93, and (when `shove` names one) that joint carrying an additive
    U(-amp, amp) x command-scale disturbance from passage `at` on."""
    rng = random.Random(1000 + seed)
    step = arm_config.DEFAULT_STEP_DEG
    u_cols, ds_cols = [], []
    for j in T079_JOINTS:
        u = [rng.choice((-step, 0.0, 0.0, 0.0, step)) for _ in range(n)]
        ds = [0.93 * c + 0.02 * step * rng.uniform(-1.0, 1.0) for c in u]
        if shove == j:
            ds = [
                d + (amp * step * rng.uniform(-1.0, 1.0) if p >= at else 0.0)
                for p, d in enumerate(ds)
            ]
        u_cols.append(u)
        ds_cols.append(ds)
    return u_cols, ds_cols


def _arm(seed, shove, n_shuffle):
    """One arm of R6: the constructed body, read and floored, in `t079_contrasts`' shape."""
    u_cols, ds_cols = _body_channels(seed, shove=shove)
    rows = _rows_from_channels(u_cols, ds_cols)
    return {"read": t079_read(rows), "floor": t079_floor(rows, seed, n_shuffle=n_shuffle)}


# ---------------------------------------------------------------------------------------
# R1 -- THE FLOOR IS A FLOOR
# ---------------------------------------------------------------------------------------
def test_r1_the_shuffle_floor_is_a_floor(reds):
    u_cols, ds_cols = _coupled_decoupled_channels(R1_PASSAGES, seed=11)
    rows = _rows_from_channels(u_cols, ds_cols)
    read = t079_read(rows)
    floor = t079_floor(rows, 11, n_shuffle=REDS_N_SHUFFLE)

    coupled, decoupled = T079_JOINTS[0], T079_JOINTS[1]
    c_up = read["joints"][coupled]["c"]
    f_up = floor["joints"][coupled]["floor"]
    c_dn = read["joints"][decoupled]["c"]
    f_dn = floor["joints"][decoupled]["floor"]

    assert floor["readable"], "a floor drawn on a scored recording must be readable"
    assert c_up > f_up, (
        f"a COUPLED channel must read above its cause-blind twin's rate "
        f"(c {c_up:.4f} vs floor {f_up:.4f})"
    )
    assert c_dn <= f_dn, (
        f"a DECOUPLED channel must read at or below its floor (c {c_dn:.4f} vs floor {f_dn:.4f})"
    )
    assert reds["R1"], "t079_reds() must run R1 and its mutation in-command"


# ---------------------------------------------------------------------------------------
# R2 -- THE MACHINE'S HYSTERESIS (the module's own _experiment construction, reused)
# ---------------------------------------------------------------------------------------
def test_r2_the_state_machine_keeps_its_hysteresis_both_ways(reds):
    r = _experiment(0)
    assert r["first_high_while_lost_status"] == "lost", (
        "a single HIGH reading while 'lost' must not flip the channel to 'ok'"
    )
    assert r["high_at_flip"] >= r["recover_n"], (
        f"revival needed {r['high_at_flip']} consecutive highs, recover_n {r['recover_n']}"
    )
    assert r["reach_ok_through_decouple"], "the still-coupled channel must stay 'ok' throughout"
    assert reds["R2"], "t079_reds() must run R2 and its no-hysteresis mutation in-command"


# ---------------------------------------------------------------------------------------
# R3 -- AN UNEXERCISED CHANNEL IS UNSCORED, NEVER LOST
# ---------------------------------------------------------------------------------------
def test_r3_an_unexercised_channel_is_unscored_never_lost(reds):
    u_cols, ds_cols = _dead_channel_channels(R1_PASSAGES, seed=13)
    rows = _rows_from_channels(u_cols, ds_cols)
    read = t079_read(rows)
    dead = T079_JOINTS[0]

    assert read["joints"][dead]["n"] >= T079_WARMUP, "the dead channel is still OBSERVED"
    assert not read["joints"][dead]["scored"], (
        f"u == 0 on every passage must leave the channel UNSCORED "
        f"(var_c {read['joints'][dead]['var_c']:.3e} vs var_floor {T079_VAR_FLOOR:.1e})"
    )
    assert read["joints"][dead]["status"] != "lost", (
        "an unexercised channel is unscored, NEVER lost -- you cannot lose what you never drove"
    )
    assert not read["readable"], "(r) must REFUSE a recording with an unscored joint"
    for j in T079_JOINTS[1:]:
        assert read["joints"][j]["scored"], f"the exercised joint {j} must be scored"
    assert reds["R3"], "t079_reds() must run R3 and its var_floor := 0 mutation in-command"


# ---------------------------------------------------------------------------------------
# R4 -- THE PARSER
# ---------------------------------------------------------------------------------------
def test_r4_the_token_parser_refuses_a_joint_outside_rotatable(reds):
    plus = intended_deltas(arm_config.command_for("elbow_flex", 6.0))
    minus = intended_deltas(arm_config.command_for("elbow_flex", -6.0))
    assert plus["elbow_flex"] == 6.0, f"+6deg on elbow_flex, got {plus['elbow_flex']!r}"
    assert minus["elbow_flex"] == -6.0, f"-6deg on elbow_flex, got {minus['elbow_flex']!r}"
    for j in T079_JOINTS:
        if j != "elbow_flex":
            assert plus[j] == 0.0 and minus[j] == 0.0, f"{j} was not commanded"

    for token in ("ENGAGE", "RELEASE", "HOLD", "HOME", "EMERGENCY_STOP", None, ""):
        d = intended_deltas(token)
        assert set(d) == set(T079_JOINTS) and all(v == 0.0 for v in d.values()), (
            f"{token!r} commands no joint delta, got {d!r}"
        )

    # MOVETO is the ABSOLUTE verb: its argument is a setpoint, not a commanded delta.
    moveto = intended_deltas(arm_config.moveto_for("elbow_flex", 30.0))
    assert all(v == 0.0 for v in moveto.values()), f"MOVETO is not a delta, got {moveto!r}"

    # a ROTATE token naming a joint outside ROTATABLE is REFUSED, not parsed
    for foreign in ("JOINT_GRIPPER_ROTATE(+6deg)", "JOINT_TAIL_WAG_ROTATE(+6deg)"):
        d = intended_deltas(foreign)
        assert set(d) == set(T079_JOINTS), f"{foreign} minted a channel: {sorted(d)}"
        assert all(v == 0.0 for v in d.values()), f"{foreign} was parsed, not refused: {d!r}"

    # ONE parser (P2): live_arm's one-joint spelling agrees on shoulder_lift
    for deg in (-15.0, -6.0, 6.0, 15.0):
        tok = arm_config.command_for("shoulder_lift", deg)
        assert live_arm._joint_obs_delta(tok) == intended_deltas(tok)["shoulder_lift"] == deg, (
            f"the two parsers disagree on {tok}"
        )
    assert live_arm._joint_obs_delta("ENGAGE") == 0.0
    assert reds["R4"], "t079_reds() must run R4 and its parse-any-name mutation in-command"


# ---------------------------------------------------------------------------------------
# R5 -- THE PER-JOINT FOLD
# ---------------------------------------------------------------------------------------
def test_r5_the_per_joint_fold_wraps_and_its_mean_is_the_scalar(reds):
    # the T037 seam, per joint: 359.9 -> 0.1 is a +0.2 DEGREE move, not a -359.8 one
    seam = proprioception_deltas([359.9] * 6, [0.1] * 6)
    assert seam is not None, "a readable body must fold"
    for i, d in enumerate(seam):
        assert abs(d * 180.0 - 0.2) < 1e-9, (
            f"joint {i}: 359.9 -> 0.1 must fold to +0.2 deg, got {d * 180.0:+.6f} deg"
        )

    # the scalar IS the mean of the list, bit-for-bit, on 200 random triples
    rng = random.Random(5)
    for _ in range(200):
        before = [rng.uniform(-720.0, 720.0) for _ in range(6)]
        after = [rng.uniform(-720.0, 720.0) for _ in range(6)]
        per = proprioception_deltas(before, after)
        assert per is not None and len(per) == 6
        assert sum(per) / len(per) == proprioception_delta(before, after), (
            "proprioception_delta must BE the mean of proprioception_deltas, bit-for-bit"
        )

    # a body that cannot be read folds to None, never to a row of zeros
    assert proprioception_deltas(None, None) is None
    assert proprioception_deltas([1.0], [2.0]) is None
    assert reds["R5"], "t079_reds() must run R5 and its no-wrap mutation in-command"


# ---------------------------------------------------------------------------------------
# R6 -- THE DISSOCIATION ON A CONSTRUCTED BODY
# ---------------------------------------------------------------------------------------
def test_r6_the_dissociation_reads_on_a_constructed_body(reds):
    intact = [_arm(s, None, REDS_N_SHUFFLE) for s in R6_SEEDS]
    shoved = [_arm(s, T079_SHOVE_JOINT, REDS_N_SHUFFLE) for s in R6_SEEDS]
    got = t079_contrasts(intact, shoved)

    assert got["a1"].met, f"(a1) the perturbed channel must collapse: t {got['a1'].t:.3f}"
    for j, c in got["a2"].items():
        assert c.met, f"(a2) {j} must be preserved within the deadband {T079_DEADBAND}: t {c.t:.3f}"
    assert got["a3"]["met"], f"(a3) the status machine must agree exactly: {got['a3']}"

    # THE TWIN WITH NO DECOUPLING: identical construction, the causal link left intact
    twin = [_arm(s, None, REDS_N_SHUFFLE) for s in R6_SEEDS]
    null = t079_contrasts(intact, twin)
    assert not null["a1"].met, "(a1) must NOT fire when nothing was decoupled"
    assert not null["a3"]["met"], "(a3) must NOT fire when nothing was decoupled"
    assert reds["R6"], "t079_reds() must run R6 and its blind-scorer mutation in-command"
