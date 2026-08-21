"""Afferent lane, simulated (specs/022 T036, SYNTHESIS L1.5): the ONE transduction-in
boundary between raw sensor readings and the brain's stimulus contract, with 014's
idle-sentinel fix applied AT THE BOUNDARY and per-modality manifest rows.

THE MODALITY INVENTORY, pinned (each row's stage and disposition stated once, here):
  * PROPRIOCEPTION -- S0, DECLARED AND LIVE (`proprio_out` -> brain.proprio_in): the
    arm's joint angles folded to one normalized scalar riding the temperature band. The
    fold LIVED INLINE in Brain._safe_ctx until this task; it is extracted here so the
    declared anatomy and the runtime transduction are the same code (P2: one mechanism).
  * DISTANCE / TEMPERATURE (the ranging pair) -- CONTRACT DECLARED, CHARGE DATED, and
    that gap is stated LOUDLY rather than implied away (the review measured it): NO live
    producer routes these readings through this boundary today -- the daemon's ctx pins
    distance SAFE by construction and the serial lane is awaiting hardware -- so
    transduce_distance / transduce_temperature are the CONTRACT the simulated feed and
    the hardware lane adopt when they land (their §6 dispositions carry this date). The
    live sentinel protection that runs today remains 014's guard inside
    AutonomousArm.step, untouched. Presence is not participation; these two rules are
    presence, disclosed as such, until their producer arrives.
  * INTEROCEPTION -- S0 PRODUCER TRACT DECLARED, TERMINUS STILL OPEN, stated exactly:
    T026 declared medulla arousal_out + hypothalamus drive_deficit_out -> insula
    body_in, and insula's `interoception_out` remains EDGE-LESS with the insula module's
    dormancy `wire` row OPEN (its coalition consumer is bound to the S0 exit, T039).
    The LIVE interoceptive channels the arm's coalitions carry (T031) read the arm's
    OWN medullary organ, not the insula tract. This boundary declares NO second
    interoceptive row (P2); an earlier draft of this paragraph claimed the tract
    terminated at T031's coalition channels, and the review measured that edge absent.
  * VISION / AUDITION -- S1, DISPOSITIONED NOT DECLARED (the T028 precedent: an edge
    lands WITH its consumer): the real-percept attention streams exist
    (Brain._attend_real_stream) but their transduction contract is S1's work; declaring
    rows today would be anatomy ahead of the boundary that gives them meaning. Recorded
    in SYNTHESIS section 6 with this task's date.
  * CHEMOSENSATION -- OUT OF SCOPE on PROVENANCE: no sensor exists, simulated or real;
    a declared row would certify a modality with no source of truth.

THE IDLE SENTINEL (014 E-2, the measured hazard): the ESP32 firmware sends distance -1
when nothing is within 3 m, and -1.0 is FINITE and BELOW CONTACT -- the naive wire
turned 'nothing detected' into 'contact', binding fear to empty rooms. The arm's step()
carries the amygdala-side guard since 014; THIS boundary makes the reading unambiguous
IN TYPE -- (metres, present), with nan metres for missing so the fail-closed default
survives a consumer that ignores the flag (see transduce_distance for the measured
disarm a bare-None draft shipped). Temperature reuses esp32_encoder_path.normalize's
finite-input rule whole -- that normalizer is the incumbent, NOT re-implemented -- and
adds exactly TWO halves the incumbent cannot carry, both named: (1) its distance01
CLAMPS negatives to 0.0 = contact; (2) its temp path maps nan/inf to 1.0 = MAX HEAT
(the `<= 0` sentinel test is False for both), and this boundary returns missing for
non-finite instead -- two answers exist for those inputs and the divergence is a
deliberate, disclosed guard, not a fork. The SIMULATED FEED (make_sensor_dataset /
signal_loop's jittered prototypes, both 014 lineage) is the transduction-in incumbent
for every modality until real hardware lands; esp32 serial and csv replay remain
dormancy-covered per SYNTHESIS section 6's dated dispositions.

# distinct-from: substrate/esp32_encoder_path.py -- the 014 normalizer/encoder this
#   boundary REUSES for temperature and extends for the distance sentinel; hardware
#   parsing stays there. distinct-from: substrate/make_sensor_dataset.py -- the 014
#   simulated GESTURE dataset generator; a data tool, not a boundary. distinct-from:
#   brainstem/signal_loop.py -- the reflex/serial event loop; its simulated_stream is a
#   feed this boundary can sit in front of, not a transduction contract. distinct-from:
#   substrate/episode_driver.py -- the TRIAL apparatus behind this boundary; it takes
#   already-transduced stimuli. distinct-from: insula/interoceptive_integration.py --
#   the interoceptive tract terminus; see the inventory above. All sites read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.afferent_boundary
"""

import math

from snn_data_lake.substrate.esp32_encoder_path import normalize as _esp32_normalize


def transduce_distance(distance_m):
    """The distance modality's boundary rule -- 014 E-2's SPLIT PRESERVED IN TYPE.

    Returns (metres, present):
      * real reading      -> (raw metres float, True)
      * sentinel / absent -> (nan, False)
      * non-finite input  -> (nan, False)

    WHY nan AND NOT None FOR MISSING (the review's safety BLOCKER, measured): the arm's
    incumbent guard short-circuits on `distance is not None`, so a first draft that
    returned bare None turned 'sensor said nothing' into NO REFLEX AT ALL -- the arm
    ENGAGED on a reading the incumbent E-STOPs for. 014 split the sentinel handling
    deliberately (fail-CLOSED on the E-STOP, fail-OPEN only on the amygdala pairing);
    this type preserves that split: a lazy consumer that forwards `metres` and ignores
    `present` hands the arm nan, and the arm's non-finite branch FAIL-CLOSES
    (EMERGENCY_STOP) -- the safe outcome is the default by construction, and only the
    `present` flag carries the no-US distinction the amygdala half needs.

    RAW METRES, with the direction stated the way the tracked contract test states it
    (tests/test_distance_units_contract.py exists precisely to stop this record
    drifting): CONTACT is DEFINED in metres, and substituting the normalized value
    CHANGES the trip point exactly 3x -- the arm would trip at three times the distance,
    stopping MORE often, never less; the defect is WRONG BEHAVIOR against a defined
    safety constant, not a loosening. (An earlier draft of this docstring, and its
    SYNTHESIS copy, carried the drifted 'loosens' direction the test refutes --
    corrected against the test's own header.)"""
    if distance_m is None:
        return (float("nan"), False)
    d = float(distance_m)
    if not math.isfinite(d) or d < 0.0:
        return (float("nan"), False)
    return (d, True)


def transduce_temperature(temp_k):
    """The temperature modality's boundary rule: esp32_encoder_path.normalize's OWN
    sentinel semantics (<= 0 K -> missing), reused whole -- one normalizer (P2)."""
    if temp_k is None:
        return None
    t = float(temp_k)
    if not math.isfinite(t):
        return None
    _d01, t01 = _esp32_normalize(0.0, t)
    return t01


def proprioception(theta_deg):
    """The proprioceptive fold, extracted VERBATIM from Brain._safe_ctx (the runtime
    transduction and the declared anatomy must be one code): both joints' angles wrapped
    to [0, 360) and averaged to one scalar in [0, 1]. None / short input -> None (a body
    that cannot be read is a missing modality, never a fabricated posture)."""
    if not theta_deg or len(theta_deg) < 2:
        return None
    try:
        prop = ((float(theta_deg[0]) % 360.0) / 360.0 + (float(theta_deg[1]) % 360.0) / 360.0) / 2.0
    except (TypeError, ValueError):
        return None
    return round(min(1.0, max(0.0, prop)), 4)


def declare_ports(manifest):
    """Manifest registrant (specs/022 T036): the S0 afferent rows. `proprio_out` carries
    the folded joint scalar; the consumer edge is declared by the consumer
    (cortex/brain.py's declare_ports: brain.proprio_in -- the temperature band
    Brain._safe_ctx writes every wake tick). Interoception's row is the T026 tract
    (declared there); vision/audition are S1 dispositions; chemosensation out-of-scope
    -- the inventory paragraph in this module's docstring is the pinned record."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "afferent_boundary",
        Port(
            name="proprio_out",
            region="substrate",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="OUT",
            declaring_module="afferent_boundary",
            delivery_policy="all-or-none",
        ),
    )


def main():
    print("=" * 74)
    print("AFFERENT BOUNDARY -- transduction-in with the idle sentinel handled at the door")
    print("=" * 74)

    # (1) THE SENTINEL RULE IN TYPE, and the FAIL-CLOSED default proven END-TO-END on a
    # real arm (the review's own A/B, inverted: a bare-None first draft made the arm
    # ENGAGE on the sentinel the incumbent E-STOPs for -- the boundary contract must
    # never be able to disarm the reflex, even under a lazy consumer that forwards
    # metres and ignores `present`).
    m, present = transduce_distance(-1.0)
    assert not present and math.isnan(m), (
        "the sentinel must be (nan, False) -- missing, fail-closed"
    )
    for raw in (float("nan"), None, float("-inf")):
        m2, p2 = transduce_distance(raw)
        assert not p2 and math.isnan(m2)
    assert transduce_distance(0.02) == (0.02, True), "a real contact passes in RAW metres, present"
    assert transduce_distance(2.5) == (2.5, True)

    from snn_data_lake.brainstem.autonomous_loop import AutonomousArm

    def _reflex_on(dist):
        bot = AutonomousArm(seed=0)
        out = bot.step(visual=[0.5] * 4, temperature=[0.5], audio=[0.5] * 4, distance=dist)
        return bool(out["reflex"]), out["action"]

    for raw in (-1.0, float("nan")):
        fm, fp = transduce_distance(raw)
        raw_reflex, raw_action = _reflex_on(raw)
        fwd_reflex, fwd_action = _reflex_on(fm)  # the LAZY consumer: metres forwarded
        assert raw_reflex and fwd_reflex and fwd_action == "EMERGENCY_STOP" == raw_action, (
            f"FAIL-CLOSED VIOLATED at raw={raw}: incumbent {raw_action} vs "
            f"boundary-forwarded {fwd_action} -- the boundary must never disarm the reflex"
        )
        assert not fp, "and `present` must still carry the no-US distinction for the amygdala half"

    # (2) temperature reuses the 014 normalizer's sentinel whole.
    assert transduce_temperature(-1.0) is None and transduce_temperature(0.0) is None
    assert transduce_temperature(None) is None
    t = transduce_temperature(300.15)
    assert t is not None and 0.0 <= t <= 1.0

    # (3) proprioception: the fold is total, bounded, and honest about a missing body.
    assert proprioception(None) is None and proprioception([10.0]) is None
    p = proprioception([180.0, 90.0])
    assert p == round((0.5 + 0.25) / 2, 4), f"the fold must match _safe_ctx's verbatim, got {p}"
    assert proprioception([720.0, -90.0]) == round((0.0 + 0.75) / 2, 4), "wrap, not clamp"

    # (4) NEUTER A/B EXIT, on the live composite: the proprioceptive loop is what closes
    # the brain's causal circle (percept <- own action). Neuter the boundary (fold dead)
    # and the temperature band collapses to the synthetic rotation's <= 3 distinct
    # values; live, it tracks the arm and varies. Run in one command, same seed.
    import os
    import tempfile

    os.environ.setdefault("NORD_HOME", tempfile.mkdtemp(prefix="afferent_ab_"))
    from snn_data_lake.cortex.brain import make_brain

    def temp_trace(neutered, n=40):
        b = make_brain(seed=0)
        if neutered:
            import snn_data_lake.substrate.afferent_boundary as ab

            orig = ab.proprioception
            ab.proprioception = lambda theta_deg: None  # the NEUTER: no body reading
        vals = []
        try:
            for _ in range(n):
                b.clock.phase = 0.25  # forced wake
                b.tick()
                vals.append(tuple(b._safe_ctx()["temperature"]))
        finally:
            if neutered:
                ab.proprioception = orig
        return set(vals)

    live_vals = temp_trace(False)
    dead_vals = temp_trace(True)
    assert len(dead_vals) <= 3, (
        f"the neutered boundary must collapse to the synthetic rotation, got {len(dead_vals)}"
    )
    assert len(live_vals) > len(dead_vals), (
        f"the live boundary must DRIVE the proprioceptive band ({len(live_vals)} distinct "
        f"vs neutered {len(dead_vals)}) -- else the declared lane carries nothing"
    )

    print(
        f"\nself-check OK: the sentinel is (nan, False) and FAIL-CLOSED end-to-end on a "
        f"real arm (lazy-forwarded missing metres -> EMERGENCY_STOP, matching the "
        f"incumbent, with `present` carrying the no-US half); temperature via the 014 "
        f"normalizer with both divergence halves disclosed; the fold matches _safe_ctx "
        f"verbatim with wrap semantics; proprio neuter A/B: live {len(live_vals)} distinct "
        f"values vs neutered {len(dead_vals)} (the causal loop the DECLARED lane closes). "
        f"The ranging pair is CONTRACT-until-producer, stated in the inventory."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
