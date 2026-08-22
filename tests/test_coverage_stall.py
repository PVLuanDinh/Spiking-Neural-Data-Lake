"""The coverage objective's stall (owner-routed fix, 2026-08-22): three mechanisms, each a red.

1. The ground-recovery reflex used to fire a FIXED +shoulder_lift ('one +A step is the minimal
   motion that raises the end effector' -- true of the retired 2-link joint A, false on the SO-101,
   where +shoulder_lift points DOWN). Measured at the recovery site on seed 2: its 189 firings moved
   the end effector by at most ONE MICRON in either direction and not one of them cleared the GROUND
   collision on the tick it fired; five of eight seeds spent 456-577 of 600 ticks in the floor. The
   tap now records the height change coincident with every MOTOR command it passes, and the reflex
   fires the token the body said climbs. No body swap can re-create the defect.
2. Exploration noise was sized by a running MEAN of |r|, which converges to whatever the CURRENT
   action pays -- under a settled HOLD (-0.10) sigma fell to 0.005 against a 0.29 value margin and
   the flip probability was 0 on every tick. A running PEAK of |r_ext| fixed that on a COLD body
   and starved on the live one (a restored ledger earns no fresh cell, and near HOME nothing
   collides, so the peak saw only the stillness term: ZERO new cells in 400 daemon ticks). The
   scale is now what the objective COULD pay -- the [-1, 1] contract, ENFORCED here and counted.
3. Cortisol was fed the raw negative reward, so the steady -0.10 stillness term pinned it at its cap
   and the 0.6 caution bias on HOLD became the lock's margin. It is now fed the BODY'S EVENT. The
   intermediate form the first draft shipped -- the baseline-relative `err` -- is pinned here as a
   red in its own right: it is zero-mean, so it measured reward VARIANCE rather than adversity.

Run: pytest tests/test_coverage_stall.py
"""

import pytest

from snn_data_lake.brainstem import autonomous_loop as al
from snn_data_lake.brainstem.autonomous_loop import ACTIONS, AutonomousArm, _ee_m

CTX = dict(visual=0.55, temperature=24.0, audio=0.3, distance=0.6)
LIFT_UP = al._A_NEG  # -shoulder_lift raises the gripper from the floor on this body (measured)
LIFT_DOWN = al._A_POS


def _run(a, ticks, reward_fn=None):
    under = 0
    for _ in range(ticks):
        a.step(reward_fn=reward_fn or a.coverage_reward, **CTX)
        under += int(a.arm.state().get("collision") == "GROUND")
    return under


def _sunk(seed):
    """A body driven into the floor through step() by the token the old reflex fired."""
    a = AutonomousArm(seed=seed)
    select = a._select
    a._select = lambda *args, **kw: LIFT_DOWN
    for _ in range(40):
        a.step(reward_fn=a.coverage_reward, **CTX)
        if a.arm.state().get("collision") == "GROUND":
            a._select = select  # the real policy from here on
            return a
    raise AssertionError("could not sink the body")


# ------------------------------------------------------------------------- 1. the compass


def test_the_recovery_reflex_follows_the_body_not_a_fixed_sign():
    a = AutonomousArm(seed=0)
    # an untried token is tried first (optimism: unknown beats known-bad) ...
    assert a._recovery_command() == al._MOTOR[0]
    # ... and once the body has answered for every token, the one that climbs wins
    a._climb = {t: 0.0 for t in al._MOTOR}
    a._climb[LIFT_DOWN] = -0.02  # what the old fixed token measured on the stall pose
    a._climb[LIFT_UP] = +0.04
    assert a._recovery_command() == LIFT_UP
    a._climb["JOINT_ELBOW_FLEX_ROTATE(-6deg)"] = +0.05
    assert a._recovery_command() == "JOINT_ELBOW_FLEX_ROTATE(-6deg)", "the compass is the map"


def test_the_compass_records_only_what_the_reflex_can_fire_and_only_when_the_body_can_move():
    """The first sample IS the measurement and later ones move it at the loop's EMA rate; a
    non-motor token, a latched arm and a body that reports no height all teach it nothing --
    the last one COUNTED, because a frozen compass must not look like a body that never climbed."""
    a = AutonomousArm(seed=0)
    z0 = _ee_m(a.arm.state())[2]
    st = a.apply_observed(LIFT_UP)
    dz1 = _ee_m(st)[2] - z0
    assert a._climb[LIFT_UP] == pytest.approx(dz1), "the first sample IS the measurement"
    z1 = _ee_m(st)[2]
    st = a.apply_observed(LIFT_UP)
    dz2 = _ee_m(st)[2] - z1
    assert a._climb[LIFT_UP] == pytest.approx((1.0 - a._int_tau) * dz1 + a._int_tau * dz2)
    # a token the reflex can never fire is not evidence for it (and bounds the map at len(_MOTOR))
    a.apply_observed("HOLD")
    a.apply_observed("ENGAGE")
    assert set(a._climb) <= set(al._MOTOR), "the compass took a token the reflex cannot fire"
    # a latched arm refuses motion, so its zeros must not dilute the map
    a.apply_observed("EMERGENCY_STOP")
    frozen = dict(a._climb)
    for _ in range(20):
        a.apply_observed(LIFT_UP)
    assert a.arm.state()["stopped"] and a._climb == frozen, "a latched arm taught the compass"
    # a body that reports no height teaches nothing, and says so
    b = AutonomousArm(seed=0)
    b.arm.state = lambda: {"theta_deg": [0.0] * 6, "stopped": False}
    b.apply_observed(LIFT_UP)
    assert LIFT_UP not in b._climb and b._climb_errors == 1


def test_a_fallen_body_gets_back_out_of_the_floor():
    """The red, constructed: the floor stops being ABSORBING. A body sunk with +shoulder_lift
    spends under half of the next 120 ticks underground with the learned compass, and the map ends
    ordering the old token BELOW the one that climbs. The control is the old reflex (a fixed
    +shoulder_lift) on the same sunk body, which stays under for more than twice as long. Asserted
    as a fraction of the window rather than the pose at one instant: the recovered body oscillates
    in and out of contact, so the final tick is a coin-flip and the residency is the property."""
    a = _sunk(2)
    under = _run(a, 120)
    assert under < 60, f"the floor is still absorbing: {under} of 120 ticks underground"
    assert a._recoveries >= 1 and LIFT_DOWN in a._climb, "the reflex never fired"
    measured = [m for m in al._MOTOR if m in a._climb]
    assert a._climb[LIFT_DOWN] < max(a._climb[m] for m in measured), (
        "the body's answer: the old token is not the one that climbs"
    )
    ctrl = _sunk(2)
    ctrl._recovery_command = lambda: LIFT_DOWN  # the reflex as it was until 2026-08-22
    ctrl_under = _run(ctrl, 120)
    assert ctrl_under > 2 * under, (
        f"the control: the fixed token must keep the body under ({ctrl_under} vs {under} of 120)"
    )


# ------------------------------------------------------------------- 2. the exploration scale


def test_exploration_does_not_anneal_under_a_settled_action():
    """The two SCALES are measured on a LIVE arm -- 400 ticks of the coverage objective -- and then
    both are applied to the value row the stall actually produced (HOLD at its -0.10 fixed point
    after 400 visits; the 12 motor values poisoned to -0.40 by a contact episode, 37 visits each).
    The retired MECHANISM cannot cross that margin; the shipped one can. The control is the
    mechanism measured from the same reward stream, not a smaller number typed in."""
    a = AutonomousArm(seed=0)
    retired = 0.0  # the EMA of |r| that sized the noise until 2026-08-22

    def ema(r):
        nonlocal retired
        retired = r if retired <= 0.0 else (1.0 - a._int_tau) * retired + a._int_tau * r

    # phase 1: the objective, so the peak reaches the real stake the task pays
    for _ in range(300):
        before = len(a._covered)
        a.step(reward_fn=a.coverage_reward, **CTX)
        ema(1.0 if len(a._covered) > before else 0.10)
    stake = al._REWARD_MAX
    # phase 2: an action SETTLES -- this is the regime where the retired scale collapsed toward
    # whatever that action pays, and where the shipped one must not
    for _ in range(400):
        a.step(reward_fn=lambda u, act: -0.10, **CTX)
        ema(0.10)
    assert stake > 5 * retired > 0.0, (
        f"the scales must diverge once an action settles: stake {stake} vs retired {retired}"
    )
    unit = 0
    a.q[unit] = {act: -0.40 for act in ACTIONS}
    a.q[unit]["HOLD"] = -0.10
    for act in ACTIONS:
        a._visits[(unit, act)] = 400 if act == "HOLD" else 37
    scores = [a.q[unit][act] for act in ACTIONS]
    hold = ACTIONS.index("HOLD")

    def flips(scale):
        al._REWARD_MAX = scale
        return sum(
            max(range(len(ACTIONS)), key=lambda i: scores[i] + a._explore_noise(unit, scores)[i])
            != hold
            for _ in range(2000)
        )

    try:
        assert flips(retired) == 0, "the control: the retired mechanism could not cross the margin"
        assert flips(stake) > 0, "R-max optimism must keep sampling a poisoned action"
    finally:
        al._REWARD_MAX = stake


def test_the_exploration_scale_is_the_enforced_reward_contract():
    """Optimism is sized by what the objective COULD pay (`_REWARD_MAX`), not by what it has paid:
    an OBSERVED scale cannot bootstrap, because the arm must explore to sample the reward that
    would license the exploration. The contract is ENFORCED here rather than assumed -- an
    out-of-range reward is clamped and COUNTED -- which is what makes the bound real."""
    a = AutonomousArm(seed=0)
    for _ in range(5):
        a.step(reward_fn=lambda u, act: 50.0, **CTX)
    assert a._reward_clamps == 5, a._reward_clamps
    assert not hasattr(a, "_r_peak") and not hasattr(a, "_r_mag"), "superseded scales must retire"
    b = AutonomousArm(seed=0)
    for _ in range(5):
        b.step(reward_fn=lambda u, act: 0.25, **CTX)
    assert b._reward_clamps == 0
    # sigma is the contract bound over the state-action count, on both arms alike: a body whose
    # objective happens to pay small rewards must not explore less than one whose objective is loud
    unit = next(iter(b.q))
    b._visits[(unit, "HOLD")] = 0
    scores = [b.q[unit][act] for act in ACTIONS]
    draws = [abs(b._explore_noise(unit, scores)[ACTIONS.index("HOLD")]) for _ in range(400)]
    assert max(draws) > 0.5 * al._REWARD_MAX, (
        f"an untried action must carry full optimism: {max(draws)}"
    )


def test_the_contract_bound_is_where_the_retired_mean_converges_on_a_loud_objective():
    """A curiosity-off probe whose objective pays the full stake is unchanged by the swap: the
    retired running mean of |r| converges to exactly the bound optimism now uses, so the fixed
    +-1 probes in this module's self-check see the same sigma they always did. The divergence is
    entirely in the QUIET regimes -- a settled action, or a restored body that earns neither a
    fresh cell nor a collision -- which is where both observed scales starved."""
    a = AutonomousArm(seed=0)
    retired = 0.0
    for _ in range(30):
        a.step(reward_fn=lambda u, act: 1.0 if act == "RELEASE" else -1.0, **CTX)
        retired = 1.0 if retired <= 0.0 else (1.0 - a._int_tau) * retired + a._int_tau * 1.0
    assert a.curiosity is None, "this coincidence is claimed for the curiosity-OFF probes"
    assert retired == pytest.approx(al._REWARD_MAX)


# ------------------------------------------------------------------------ 3. the stress signal


def test_stress_is_the_bodys_event_not_the_objectives_running_opinion():
    """The four regime reds the feed was chosen against. R1: a steady shaping cost must not pin it
    (the incumbent read 1.0). R2: sustained real danger must not read calm (the first draft's
    baseline-relative `err` read 0.019 after 500 ticks in the floor). R3: an all-positive regime
    must not read stressed (`err` read 0.482 -- it is zero-mean, so it measured VARIANCE)."""
    # R1 and R3 hold the body STILL (a pinned HOLD never leaves the home pose, so nothing collides):
    # these two measure the FEED, and a body that wanders into the floor would raise stress for a
    # legitimate reason and red them for the wrong one.
    steady = AutonomousArm(seed=0)
    steady._select = lambda *args, **kw: "HOLD"
    _run(steady, 400, reward_fn=lambda u, act: -0.10)
    assert not steady.arm.state().get("collision"), "R1's body must not collide"
    assert steady.cort.level < 0.5, f"a steady shaping cost pinned cortisol at {steady.cort.level}"

    good = AutonomousArm(seed=0)
    good._select = lambda *args, **kw: "HOLD"
    _run(good, 300, reward_fn=lambda u, act: 1.0 if act == "HOLD" else 0.2)
    assert not good.arm.state().get("collision"), "R3's body must not collide"
    assert good.cort.level < 0.1, f"an all-positive regime read as stress ({good.cort.level})"

    # The danger regime is DRIVEN, not left to the policy: pinning both the selection and the
    # recovery to the sinking token keeps the body in the floor whatever the other two legs do, so
    # this red is attributable to the stress feed alone (it cross-fired on all three otherwise).
    sunk = _sunk(2)
    sunk._recovery_command = lambda: LIFT_DOWN
    sunk._select = lambda *args, **kw: LIFT_DOWN
    under = _run(sunk, 200)
    assert under > 100, f"the danger regime did not stay dangerous ({under}/200)"
    assert sunk.cort.level > 0.9 and sunk.cort.caution_bias() > 0.5, (
        f"sustained danger read calm ({sunk.cort.level})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
