"""The reds for T079 -- specs/022 Phase 4 / C4, the agency instrument
(evidence/t079-agency.md 2.4).

WRITTEN AND COMMITTED **RED** at 4725249 against the 2.9 step-2 skeleton (6F/6, every
failure an ASSERTION on a constructed input and not a collection ImportError -- the shape
T077's panel named, evidence 3.5 item 10). This is the step-4 revision: the instruments are
green, the four constructed-stream builders have MOVED INTO the module (where `t079_reds()`
needs them, so a gate run that never executes pytest still runs every red on the same
stream), and each red now also RUNS the mutated scorer that reddens it.

  R1 THE FLOOR IS A FLOOR -- a constructed COUPLED channel (fb = 0.9 u + noise, 600
     passages) reads its mean controllability ABOVE its shuffle floor; a DECOUPLED
     channel (fb independent of u) reads AT or BELOW it. MUTATION: the floor drawn from the
     UN-shuffled sequence, which makes every draw equal the reading and collapses the
     coupled channel's margin to exactly zero -- and, by the same exactness, is the batched
     torch recurrence's POSITIVE CONTROL against the shipped `AgencyMonitor`.
  R2 THE MACHINE'S HYSTERESIS -- on the module's own `_experiment` construction
     (healthy -> decoupled -> recovered) a single LOW reading while 'ok' does not flip to
     'lost' and a single HIGH reading while 'lost' does not flip to 'ok'. MUTATION:
     delay = recover_n = 0, which flips on one reading in both directions.
  R3 AN UNEXERCISED CHANNEL IS UNSCORED, NEVER LOST -- u == 0 on every passage reads
     `scored` False, never 'lost', and (r) REFUSES the recording; the four exercised
     joints of the same recording read `scored` True. MUTATION: var_floor := 0, which
     scores the dead channel, lets its status reach 'lost', and lets (r) read.
  R4 THE PARSER -- `intended_deltas` reads +/-6 on the named joint and 0 on the other
     four; ENGAGE / RELEASE / HOLD / HOME / EMERGENCY_STOP and MOVETO read all 0; a
     ROTATE token naming a joint outside `arm_config.ROTATABLE` is REFUSED, not parsed;
     `live_arm._joint_obs_delta` agrees on shoulder_lift. MUTATION: a parser that trusts
     the regex instead of the vocabulary and mints a channel for the foreign joint.
  R5 THE PER-JOINT FOLD -- `proprioception_deltas` wraps EACH joint (359.9 -> 0.1 folds
     to +0.2 deg, the T037 seam red, per joint) and `proprioception_delta` equals its mean
     BIT-FOR-BIT on 200 random triples. MUTATION: the wrap dropped.
  R6 THE DISSOCIATION ON A CONSTRUCTED BODY -- five synthetic channels through the
     SHIPPED `AgencyMonitor`, one decoupled after passage 300 by an additive
     U(-2, 2) x command-scale disturbance: (a1) MET, (a2) MET on the four, (a3) MET; an
     independent no-decoupling twin reads (a1) NOT MET and (a3) NOT MET. MUTATION: c_j
     replaced by 1 - c_j on the shoved arm.

TWO MUTATIONS REDDEN A DIFFERENT CONJUNCT THAN 2.4 PREDICTS, and each is asserted where it
fires rather than where the sentence says, with the discrepancy stated -- never a silent
swap:
  * R1's `floor_unpermuted` -- 2.4 says the decoupled channel then "reads above". It does
    not: with the identity permutation every draw equals the reading EXACTLY, so the
    decoupled channel reads AT its floor, which still satisfies "at or below". What fails is
    the COUPLED channel's strict `c > floor`. (T078's own R1 words this correctly: "the
    deterministic control's margin over it vanishes".)
  * R6's `blind_scorer` -- 2.4 says 1 - c "inverts (a1)". It cannot on a body whose intact
    coupling is near 1: (a1)'s mutated difference is c_intact + c_shoved - 1, negative only
    where the two sum below 1, and c_intact reads ~0.999 on this construction (+0.945 ..
    +1.000 on the real body, 1.3). It reddens R6's reading through (a2), whose mutated
    difference is (1 - c_k) - c_k + DEADBAND and goes decisively negative.

Every red also asserts `t079_reds()[name]`, the module's own self-check line: 2.4 requires
the mutation to be applied IN THE TEST *and* in `t079_reds()`, so the suite and the gate can
never disagree about which reds ran.

No brain, no lake, no network: R1-R6 all run on constructed inputs. The live reading (a
scratch brain, seeds 0-7, 1,200 ticks x 2 arms) is NOT here -- the pin is written once by
`--t079-campaign`.

  set PYTHONUTF8=1 & PYTHONPATH=src
  .venv-lake/Scripts/python.exe -m pytest tests/test_agency_monitor.py -q --no-cov
"""

import random

import pytest

from snn_data_lake.brainstem import arm_config
from snn_data_lake.faculties.agency_monitor import (
    T079_DEADBAND,
    T079_JOINTS,
    T079_R1_PASSAGES,
    T079_REDS_N_SHUFFLE,
    T079_SEEDS,
    T079_SHOVE_JOINT,
    T079_VAR_FLOOR,
    T079_WARMUP,
    _experiment,
    _t079_mutated_scorer,
    _t079_refusals,
    t079_arm,
    t079_channel_rows,
    t079_contrasts,
    t079_coupled_decoupled,
    t079_dead_channel,
    t079_floor,
    t079_hysteresis_probe,
    t079_read,
    t079_reds,
)
from snn_data_lake.sensorimotor import live_arm
from snn_data_lake.sensorimotor.afferent_boundary import proprioception_delta
from snn_data_lake.sensorimotor.efferent_boundary import intended_deltas, proprioception_deltas


@pytest.fixture(scope="module")
def reds():
    """`t079_reds()` once per module -- the in-command control every red also asserts."""
    return t079_reds(n_shuffle=T079_REDS_N_SHUFFLE)


# ---------------------------------------------------------------------------------------
# R1 -- THE FLOOR IS A FLOOR
# ---------------------------------------------------------------------------------------
def test_r1_the_shuffle_floor_is_a_floor(reds):
    u_cols, ds_cols = t079_coupled_decoupled(T079_R1_PASSAGES, seed=11)
    rows = t079_channel_rows(u_cols, ds_cols)
    read = t079_read(rows)
    floor = t079_floor(rows, 11, n_shuffle=T079_REDS_N_SHUFFLE)

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

    # THE MUTATION, RUN: the floor drawn UN-permuted. It is also the positive control on the
    # batched torch recurrence -- with the identity permutation every draw must reproduce the
    # shipped AgencyMonitor's own c_j, which is asserted here and not assumed.
    mutant = _t079_mutated_scorer("floor_unpermuted")(rows, 11, n_shuffle=T079_REDS_N_SHUFFLE)
    assert abs(mutant["joints"][coupled]["floor"] - c_up) < 1e-9, (
        f"the batched recurrence must reproduce the shipped machine exactly "
        f"({mutant['joints'][coupled]['floor']!r} vs {c_up!r})"
    )
    assert not (c_up > mutant["joints"][coupled]["floor"]), (
        "an un-permuted floor must collapse the coupled channel's margin to zero"
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

    # the same trajectory, instrumented for the OTHER direction 2.4 names
    h = t079_hysteresis_probe(0)
    assert h["first_low_while_ok_status"] == "ok", (
        "a single LOW reading while 'ok' must not flip the channel to 'lost'"
    )
    assert h["lows_at_flip"] > h["delay"], (
        f"'lost' needed {h['lows_at_flip']} consecutive lows, delay {h['delay']}"
    )
    assert h["first_high_while_lost_status"] == "lost"
    assert h["highs_at_flip"] >= h["recover_n"]

    # THE MUTATION, RUN: delay = recover_n = 0 -- the window and the dead-band both gone
    mutant = t079_hysteresis_probe(0, monitor=_t079_mutated_scorer("no_hysteresis")())
    assert not (
        mutant["first_low_while_ok_status"] == "ok"
        and mutant["first_high_while_lost_status"] == "lost"
    ), "with delay = recover_n = 0 a single reading must flip the machine in both directions"
    assert reds["R2"], "t079_reds() must run R2 and its no-hysteresis mutation in-command"


# ---------------------------------------------------------------------------------------
# R3 -- AN UNEXERCISED CHANNEL IS UNSCORED, NEVER LOST
# ---------------------------------------------------------------------------------------
def test_r3_an_unexercised_channel_is_unscored_never_lost(reds):
    u_cols, ds_cols = t079_dead_channel(T079_R1_PASSAGES, seed=13)
    read = t079_read(t079_channel_rows(u_cols, ds_cols))
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

    # THE MUTATION, RUN: var_floor := 0 in both the machine and the read
    mutant = _t079_mutated_scorer("no_floor")(u_cols, ds_cols)
    assert not (not mutant["readable"] and not mutant["joints"][dead]["scored"]), (
        "with var_floor 0 the never-commanded channel must score and (r) must read"
    )
    assert mutant["joints"][dead]["status"] == "lost", (
        "and the machine must then declare a channel 'lost' that was never driven at all"
    )
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

    # THE MUTATION, RUN: a parser that trusts the regex instead of the vocabulary
    mutant = _t079_mutated_scorer("parse_any_name")("JOINT_GRIPPER_ROTATE(+6deg)")
    assert not (set(mutant) == set(T079_JOINTS) and all(v == 0.0 for v in mutant.values())), (
        "a parser that skips the ROTATABLE membership check must mint the foreign channel"
    )
    assert reds["R4"], "t079_reds() must run R4 and its parse-any-name mutation in-command"


# ---------------------------------------------------------------------------------------
# R5 -- THE PER-JOINT FOLD
# ---------------------------------------------------------------------------------------
def test_r5_the_per_joint_fold_wraps_and_its_mean_is_the_scalar(reds):
    # the T037 seam, per joint: 359.9 -> 0.1 is a +0.2 DEGREE move, not a -359.8 one
    before, after = [359.9] * 6, [0.1] * 6
    seam = proprioception_deltas(before, after)
    assert seam is not None, "a readable body must fold"
    for i, d in enumerate(seam):
        assert abs(d * 180.0 - 0.2) < 1e-9, (
            f"joint {i}: 359.9 -> 0.1 must fold to +0.2 deg, got {d * 180.0:+.6f} deg"
        )

    # the scalar IS the mean of the list, bit-for-bit, on 200 random triples
    rng = random.Random(5)
    for _ in range(200):
        b = [rng.uniform(-720.0, 720.0) for _ in range(6)]
        a = [rng.uniform(-720.0, 720.0) for _ in range(6)]
        per = proprioception_deltas(b, a)
        assert per is not None and len(per) == 6
        assert sum(per) / len(per) == proprioception_delta(b, a), (
            "proprioception_delta must BE the mean of proprioception_deltas, bit-for-bit"
        )

    # a body that cannot be read folds to None, never to a row of zeros
    assert proprioception_deltas(None, None) is None
    assert proprioception_deltas([1.0], [2.0]) is None

    # THE MUTATION, RUN: the wrap dropped -- the T037 seam returns
    mutant = _t079_mutated_scorer("no_wrap")(before, after)
    assert not all(abs(d * 180.0 - 0.2) < 1e-9 for d in mutant), (
        f"an unwrapped fold must read the seam as a sweep, got {mutant[0] * 180.0:+.3f} deg"
    )
    assert reds["R5"], "t079_reds() must run R5 and its no-wrap mutation in-command"


# ---------------------------------------------------------------------------------------
# R6 -- THE DISSOCIATION ON A CONSTRUCTED BODY
# ---------------------------------------------------------------------------------------
def test_r6_the_dissociation_reads_on_a_constructed_body(reds):
    intact = [t079_arm(s, None) for s in T079_SEEDS]
    shoved = [t079_arm(s, T079_SHOVE_JOINT) for s in T079_SEEDS]
    got = t079_contrasts(intact, shoved)

    assert got["a1"].met, f"(a1) the perturbed channel must collapse: t {got['a1'].t:.3f}"
    for j, c in got["a2"].items():
        assert c.met, f"(a2) {j} must be preserved within the deadband {T079_DEADBAND}: t {c.t:.3f}"
    assert got["a3"]["met"], f"(a3) the status machine must agree exactly: {got['a3']}"
    assert got["g0"] is None, "no cause-blind twin was drawn, so (g0) must read None, not 0.0"

    # THE TWIN WITH NO DECOUPLING: the same body, an INDEPENDENT noise realisation, the
    # causal link left intact. (a1) and (a3) must both stay silent.
    twin = [t079_arm(s, None, stream=1) for s in T079_SEEDS]
    null = t079_contrasts(intact, twin)
    assert not null["a1"].met, f"(a1) must NOT fire with no decoupling: t {null['a1'].t:.3f}"
    assert not null["a3"]["met"], "(a3) must NOT fire when nothing was decoupled"

    # THE MUTATION, RUN: c_j -> 1 - c_j on the shoved arm. It reddens R6's reading through
    # (a2), not (a1) -- see this file's header for the measured reason.
    mutant = t079_contrasts(intact, shoved, score=_t079_mutated_scorer("blind_scorer"))
    assert not (
        mutant["a1"].met and all(c.met for c in mutant["a2"].values()) and mutant["a3"]["met"]
    ), "a scorer blind to the direction of the collapse must not read the dissociation"
    assert not any(c.met for c in mutant["a2"].values()), (
        "and it must fail (a2) on every preserved joint, which is where it fires"
    )
    assert reds["R6"], "t079_reds() must run R6 and its blind-scorer mutation in-command"


def test_r_every_awake_tick_carries_a_passage_and_extra_passages_are_counted():
    """2.10: (r) refuses an awake tick with NO passage and READS a tick with two (a postural
    recovery), counting the extra -- the campaign-width seed-0 run carried 30 of them."""
    u_cols, ds_cols = t079_coupled_decoupled(T079_R1_PASSAGES, seed=5)
    rows = t079_channel_rows(u_cols, ds_cols)
    base = t079_read(rows)
    assert (
        base["readable"] and base["extra_passages"] == 0 and base["observed_ticks"] == base["awake"]
    )

    # two passages on one awake tick (the recovery): readable, extra 1, no refusal named
    two = [dict(r) for r in rows]
    two[300]["awake_ticks"] = two[299]["awake_ticks"]
    for r in two[301:]:
        r["awake_ticks"] -= 1
    read = t079_read(two)
    assert read["readable"], _t079_refusals(read, 0, "intact")
    assert read["extra_passages"] == 1 and read["passages"] == read["awake"] + 1
    assert not _t079_refusals(read, 0, "intact")

    # an awake tick with NO passage: refused, and the refusal names the count
    hole = [dict(r) for r in rows]
    hole[-1]["awake_ticks"] += 1
    read = t079_read(hole)
    assert not read["readable"]
    named = _t079_refusals(read, 0, "intact")
    assert any("carried NO passage" in x for x in named), named
