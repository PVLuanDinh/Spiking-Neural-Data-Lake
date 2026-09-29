"""The embodiment contract for the efference-copy boundary (specs/022 T037).

Three claims, each a regression pin on the tap around the safety envelope:
  1. PASSAGE ACCOUNTING: every AutonomousArm.step is exactly one envelope passage while
     unlatched, an EMERGENCY_STOP is a passage (a stop is a passage), and a LATCHED body
     still counts passages while its joints stay frozen. Sleep needs no separate arm
     claim here: the daemon's sleep phase simply stops calling step(), and one physics
     step per apply() with the caller's tick as the time base means nothing else CAN
     advance the envelope -- what this file pins is that step() and the tick move 1:1.
  2. THE LATCH SCHEDULE: E-STOP latches, and by DEFAULT the latch holds across later
     ticks even when the danger is gone -- ten independent checks encode that freeze as
     a design invariant, and the owner-authorised self-clear (autonomous_loop step 1b)
     is OPT-IN via NORD_ESTOP_SELF_CLEAR=1, off here. What this file adds on top of the
     existing freeze checks: the envelope tick KEEPS COUNTING through the latch -- a
     latched passage is still a passage. (The DURABLE latch across process death is
     T037a's separate, owner-authorised envelope commit -- not tested here.)
  3. OBSERVATION ONLY: neuter the tap (observe -> no-op) and the action sequence and
     joint trajectory are IDENTICAL. The boundary reads the envelope; it must not be
     able to perturb it.

The efference-copy CHARGE (cut the copy -> forward-model error rises) lives in
sensorimotor/efferent_boundary.py's own self-check; this file owns the embodiment contract.
"""

import math
import random

import pytest

from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.engine.autonomous_loop import AutonomousArm

SAFE = {"visual": [0.6] * 4, "temperature": [0.2], "audio": [0.2] * 4, "distance": CONTACT * 3.0}
DANGER = dict(SAFE, distance=0.0)


def test_envelope_tick_counts_every_passage_and_latch_schedule():
    bot = AutonomousArm(seed=0)
    eb = bot.efferent
    for i in range(1, 7):
        bot.step(**SAFE)
        assert eb.envelope_tick == i, "one step, one envelope passage -- exactly"
    assert eb.last_copy is not None and eb.last_copy["u"] is not None
    assert eb.last_copy["err"] is None or math.isfinite(eb.last_copy["err"])

    # an EMERGENCY_STOP is a passage, observed under its own token
    out = bot.step(**DANGER)
    assert out["reflex"] is True and bot.arm.state()["stopped"] is True
    assert eb.envelope_tick == 7 and eb.last_copy["action"] == "EMERGENCY_STOP"

    # latched: joints frozen through further danger, passages STILL counted
    frozen = bot.arm.state()["theta_deg"]
    for _ in range(3):
        bot.step(**DANGER)
    assert bot.arm.state()["stopped"] is True
    assert bot.arm.state()["theta_deg"] == frozen, "a latched arm moved"
    assert eb.envelope_tick == 10, "a stop is a passage; latched stops count too"

    # DEFAULT latch persistence: the self-clear is opt-in (NORD_ESTOP_SELF_CLEAR=1) and
    # off here, so a safe step later the freeze still holds -- and the tick still counts
    # it, because the step's chosen action still passes through apply() and is refused
    # INSIDE the envelope, not before it.
    bot.step(**SAFE)
    assert bot.arm.state()["stopped"] is True, "the default freeze holds through safe ticks"
    assert bot.arm.state()["theta_deg"] == frozen
    assert eb.envelope_tick == 11, "a refused passage is still a passage"


@pytest.mark.parametrize("agency", [False, True])
def test_tap_is_observation_only_identical_trajectory_when_neutered(agency):
    # T079 (2026-09-15, its panel fold): the per-joint agency feed runs INSIDE the tap's guarded
    # block after `efferent.observe`, and `AutonomousArm(seed=0)` defaults to agency_monitor=False,
    # so the green this test carried at T079's build never exercised that branch. The True arm
    # runs the feed; its red is a monitor whose observe reached the caller's arm (a divergence).
    def run(neuter):
        bot = AutonomousArm(seed=0, agency_monitor=agency)
        if neuter:
            bot.efferent.observe = lambda *a, **k: None
        rng = random.Random(7)
        acts, thetas = [], []
        for _ in range(25):
            out = bot.step(
                visual=[rng.random() for _ in range(4)],
                temperature=[rng.random()],
                audio=[rng.random() for _ in range(4)],
                distance=CONTACT * 3.0,
            )
            acts.append(out["action"])
            thetas.append(tuple(bot.arm.state()["theta_deg"]))
        return acts, thetas

    a_live, t_live = run(False)
    a_neut, t_neut = run(True)
    assert a_live == a_neut, "the tap altered the brain's chosen actions"
    assert t_live == t_neut, "the tap perturbed the body's trajectory"
