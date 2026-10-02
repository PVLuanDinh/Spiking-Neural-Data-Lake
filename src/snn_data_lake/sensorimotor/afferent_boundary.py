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
  * INTEROCEPTION -- S0 TRACT COMPLETE AT THE TYPED LAYER as of T039f/T039j, and this
    paragraph is rewritten rather than amended because every clause of the previous one
    described a state that no longer exists (it said the terminus was still open, that
    `interoception_out` remained EDGE-LESS, that the insula's dormancy `wire` row was
    OPEN, and that its coalition consumer was bound to the S0 exit). What is true now:
    the tract runs medulla `arousal_out` -> thalamus `visceral_in`/`relay_out` ->
    insula `body_in` -> `interoception_out` -> `region_coherence.interoception_in`,
    every edge carrying a measured replay-twice attribution, and the insula's dormancy
    row is RETIRED. The coalition consumer that row once promised is NOT pending -- it
    was measured OUT OF RANGE at T039b (the bus admits on a floor of 0.5505 while
    `feeling_salience` peaks at 0.2798) and REFUSED; the sink that works is threshold
    modulation. The LIVE interoceptive channels the arm's coalitions carry (T031) still
    read the arm's OWN medullary organ, not the insula tract -- that half is unchanged.
    This boundary declares NO second interoceptive row (P2); an earlier draft claimed the
    tract terminated at T031's coalition channels, and the review measured that edge
    absent.
  * VISION / AUDITION -- DECLARED AND CONSUMED as of specs/022 T052-iii (2026-08-27;
    this row said "S1, DISPOSITIONED NOT DECLARED" from T036 until then, honoring the
    T028 edge-lands-WITH-its-consumer precedent -- the consumers now exist and
    delegate): `vision_out`/`audition_out` carry the TRACKED packs' binned event
    frames (shapes pinned as VISION_FRAME_SHAPE/AUDITION_FRAME_SHAPE), and
    Brain._perceive_real/_perceive_audio acquire every frame THROUGH
    transduce_vision_events/transduce_audition_events, so the declared contract and
    the runtime acquisition are one code (the proprioception pattern). CHARGE honesty:
    these rows declare the TRANSDUCTION lane, not a new capability -- the perception
    tiers' measured accuracies predate them and are unchanged (the boundary is
    identity-on-valid); what the rows add is validation at the door and anatomy that
    matches runtime. The door's validation, in full: ndarray + pinned shape + float32
    + finite + nonnegative (each check panel-forced; an invalid frame is a silent
    no-percept-this-tick, and a custom re-binned pack fail-closes EVERY frame -- both
    disclosed at the contract and at brain's counted-claim note).
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

# distinct-from: sensorimotor/esp32_encoder_path.py -- the 014 normalizer/encoder this
#   boundary REUSES for temperature and extends for the distance sentinel; hardware
#   parsing stays there. distinct-from: sensorimotor/make_sensor_dataset.py -- the 014
#   simulated GESTURE dataset generator; a data tool, not a boundary. distinct-from:
#   sensorimotor/signal_loop.py -- the reflex/serial event loop; its simulated_stream is a
#   feed this boundary can sit in front of, not a transduction contract. distinct-from:
#   instruments/episode_driver.py -- the TRIAL apparatus behind this boundary; it takes
#   already-transduced stimuli. distinct-from: faculties/interoceptive_integration.py --
#   the interoceptive tract terminus; see the inventory above. All sites read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.sensorimotor.afferent_boundary
"""

import math

from snn_data_lake.sensorimotor.esp32_encoder_path import normalize as _esp32_normalize


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


# distinct-from: the absolute fold above -- that is T036's SENSORY posture scalar, pinned
#   verbatim to Brain._safe_ctx and WRAPPED at 0/360, which makes it undifferencable: a
#   joint at minus float noise folds 0.5 away from the same joint at plus float noise.
#   T037's efference copy needs joint-space DELTAS, so the delta is wrapped per joint
#   BEFORE normalising and the seam never enters. Reviewed in the T037 panel: differencing
#   the absolute fold manufactured 25% phantom half-range commands.
# distinct-from: reasoning/jepa_scaffold_torch.py -- a latent JEPA trainer; term overlap
#   only (action-conditioned deltas in LEARNED latent space, not a degree-space fold). Read.
# consolidates: sensorimotor/efferent_boundary.py -- its 2.9 step-2 `proprioception_deltas`
#   STUB is deleted in the same change and becomes an import of the function below, so the
#   wrap exists in exactly ONE place (P2). The efferent door keeps the NAME (specs/022 T079
#   2.1 names `efferent_boundary.proprioception_deltas` as the import path every consumer
#   uses) and the sensory door keeps the FOLD, which is where the wrap already lived; the
#   alternative -- defining it in the efferent module -- makes afferent import efferent at
#   module top, and efferent already imports afferent, which is a cycle.
# distinct-from: sensorimotor/live_arm.py::_joint_obs_delta -- a COMMANDED delta parsed from an
#   action TOKEN (one joint, degrees). This is an ACHIEVED delta read off two posture
#   vectors (every joint, normalised). Read; the token parse is routed through
#   `efferent_boundary.intended_deltas` in the same change, so it is one mechanism too.
# distinct-from: plasticity/r_stdp.py, plasticity/plasticity_rule_stack.py -- per-SYNAPSE
#   weight deltas from a reward-gated eligibility trace. distinct-from: substrate/home.py
#   -- a BinLog rebuild. distinct-from: instruments/prior_drift_harness.py -- per-seed drive-class
#   subsets. All four read; the word "delta"/"per-" is the only overlap and none folds an
#   angle.
def proprioception_deltas(theta_before, theta_after):
    """THE DELTA FOLD, per joint: theta_after - theta_before for each SHARED joint, each
    wrapped to [-180, 180) degrees BEFORE normalising by 180 -> one signed value in
    [-1, 1) per joint, in qpos order. `proprioception_delta` below is the MEAN of this
    list -- ONE fold, read two ways (specs/022 T079 2.1), so the wrap exists in exactly
    one place and the scalar can never disagree with the vector about a joint.

    UNITS, said plainly because the per-joint row now travels beside a DEGREE row (the
    efference copy's `u_j`): these values are the degree delta DIVIDED BY 180, the same
    units the scalar has always carried. A 359.9 -> 0.1 move is +0.2 deg and reads
    +0.2/180 here.

    None / short / mismatched input -> None (a body that cannot be read is a missing
    modality, never a fabricated posture -- and never a row of zeros, which a consumer
    would read as "no motion")."""
    if not theta_before or not theta_after:
        return None
    n = min(len(theta_before), len(theta_after))
    if n < 2:
        return None
    try:
        return [
            (((float(a) - float(b)) + 180.0) % 360.0 - 180.0) / 180.0
            for b, a in zip(theta_before[:n], theta_after[:n])
        ]
    except (TypeError, ValueError):
        return None


def proprioception_delta(theta_before, theta_after):
    """The proprioceptive DELTA fold: per-joint angular change theta_after - theta_before,
    each wrapped to [-180, 180) degrees, normalised by 180, averaged over the shared
    joints -> one signed scalar in [-1, 1). Sign is preserved (a negative command folds
    negative); opposite-moving joints can partially cancel in the mean -- disclosed, and
    acceptable for a one-scalar copy because the live loop commands one joint per tick.
    None / short / mismatched input -> None (a body that cannot be read is a missing
    modality, never a fabricated posture).

    MERGED 2026-09-15 (specs/022 T079 2.1): the wrap loop moved up into
    `proprioception_deltas` and this is now its mean. Numerically unchanged -- the old body
    computed `sum(ds) / n` over that same list with that same `n`, so the two are bit-for-bit
    equal, and `tests/test_agency_monitor.py::test_r5_...` asserts exactly that on 200
    random triples rather than leaving it a claim."""
    ds = proprioception_deltas(theta_before, theta_after)
    if ds is None:
        return None
    return sum(ds) / len(ds)


#: The pack event-frame CONTRACT shapes, pinned once (specs/022 T052-iii). These are the
#: shapes the TRACKED packs carry and the e-prop encoders consume -- (T, polarity, H, W)
#: binned N-MNIST events and (T, channels) binned SHD cochlear events. A frame is data;
#: its shape is anatomy, so the contract pins it here where the ports are declared.
#: DISCLOSED NARROWING (the T052-iii honesty panel measured it): T is a PRODUCER
#: parameter (training/make_packs.py's binning), and both encoders happily consume a
#: re-binned pack -- but this contract does not, so a custom SNN_PERCEPTION_PACK /
#: SNN_SHD_PACK with different binning now FAIL-CLOSES every frame (zero percepts,
#: silently -- see the counted-claim note at brain.py's _perceiving_real). The pin is
#: deliberate (an un-pinned shape is no contract) and the cost is said here: re-binned
#: packs need this constant moved WITH them, one commit. Producer<->contract drift is
#: not machine-gated; this comment is the record a drift would falsify.
VISION_FRAME_SHAPE = (15, 2, 34, 34)
AUDITION_FRAME_SHAPE = (100, 700)


# distinct-from: engine/brain.py (_perceive_real/_perceive_audio) -- the CONSUMERS this
#   contract now fronts (they delegate validation here; encoding stays theirs). Read.
# distinct-from: reservoir/kernel_state.py -- rebuilds KERNEL state from an HDF5 frame;
#   'frame' vocabulary only. Site read.
# distinct-from: substrate/arm_view.py -- PNG camera frames for the cockpit; unrelated.
#   Site read.
# distinct-from: faculties/theta_gamma_code.py -- decodes LIF firing EVENTS into theta
#   cycles; 'events' vocabulary only. Site read.
def _transduce_events(frame, want_shape):
    """Shared event-frame rule: (frame, present). None / non-ndarray / wrong-shape /
    wrong-dtype / non-finite / negative -> (None, False) -- FAIL-CLOSED for a percept
    means NO PERCEPT THIS TICK (the consumer's existing skip branch), never a
    fabricated stimulus; the T036 distance precedent's split does not apply because no
    reflex consumes these lanes (representation-only, reflex path untouched by
    construction). The checks, each panel-forced: ndarray EXPLICITLY (a first draft's
    bare `except Exception` rendered CUDA/grad tensors and MemoryErrors alike as
    'missing frame'; a CPU torch tensor even PASSED -- the contract now names its type
    instead of duck-typing it); dtype float32 (the ports declare f4 and the manifest's
    type check is declaration-level -- without this line the declaration was
    decoration); nonnegative (EVENT counts are counts). DISCLOSED LIMITATION: a
    fail-closed skip is invisible to state() -- no skip counter exists; the accuracy
    denominator honestly excludes skips (panel-verified: pointer advances, stream
    recovers, the poisoned index never counts as seen)."""
    import numpy as _np

    if frame is None or not isinstance(frame, _np.ndarray):
        return (None, False)
    if tuple(frame.shape) != want_shape or frame.dtype != _np.float32:
        return (None, False)
    # no try/except here ON PURPOSE: after the ndarray + float32 gates above, isfinite
    # and >= cannot raise -- a defensive except would be an unreachable branch worn as
    # rigor (the first draft's bare `except Exception` was the real hazard, swallowing
    # device/type errors as 'missing frame').
    if not bool(_np.isfinite(frame).all()) or not bool((frame >= 0).all()):
        return (None, False)
    return (frame, True)


def transduce_vision_events(frame):
    """The VISION modality's boundary rule (T052-iii; discharges T036's S1 disposition):
    a binned N-MNIST event frame -> (frame, present). The transduction is the IDENTITY
    on a valid frame -- the e-prop encoder consumes raw binned events, so the contract's
    job is validation (ndarray + shape + float32 + finite + nonnegative -- the full
    list, panel-hardened; see _transduce_events for why each), not transformation; the
    frame the consumer sees IS the frame the pack carries, asserted by the self-check.
    An invalid frame is a silent no-percept-this-tick (disclosed in _transduce_events:
    no skip counter exists)."""
    return _transduce_events(frame, VISION_FRAME_SHAPE)


def transduce_audition_events(frame):
    """The AUDITION modality's boundary rule (T052-iii): a binned SHD cochlear event
    frame -> (frame, present). Identity-on-valid with the SAME full validation list as
    vision (ndarray + shape + float32 + finite + nonnegative)."""
    return _transduce_events(frame, AUDITION_FRAME_SHAPE)


def declare_ports(manifest):
    """Manifest registrant (specs/022 T036): the S0 afferent rows. `proprio_out` carries
    the folded joint scalar; the consumer edge is declared by the consumer
    (engine/brain.py's declare_ports: brain.proprio_in -- the temperature band
    Brain._safe_ctx writes every wake tick). Interoception's row is the T026 tract
    (declared there); vision/audition are S1 dispositions; chemosensation out-of-scope
    -- the inventory paragraph in this module's docstring is the pinned record."""
    from snn_data_lake.engine.wiring_manifest import Port

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
    # T052-iii: the vision/audition afferent rows, landing WITH their consumers (the
    # T028 precedent) -- brain's perception tiers consume the packs TODAY and now do so
    # through these contracts (the delegation in _perceive_real/_perceive_audio).
    manifest.declare_port(
        "afferent_boundary",
        Port(
            name="vision_out",
            region="substrate",
            shape=VISION_FRAME_SHAPE,
            dtype="f4",
            unit="nmnist_event_frame",
            encoding="EVENT",
            kind="OUT",
            declaring_module="afferent_boundary",
            delivery_policy="all-or-none",
        ),
    )
    manifest.declare_port(
        "afferent_boundary",
        Port(
            name="audition_out",
            region="substrate",
            shape=AUDITION_FRAME_SHAPE,
            dtype="f4",
            unit="shd_event_frame",
            encoding="EVENT",
            kind="OUT",
            declaring_module="afferent_boundary",
            delivery_policy="all-or-none",
        ),
    )


def main():
    # T037a: this self-check drives a So101Arm through EMERGENCY_STOP / HOME, which now writes
    # the durable latch sidecar -- under a throwaway home, never the live brain's.
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _main()


def _main():
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

    from snn_data_lake.engine.autonomous_loop import AutonomousArm

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
    from snn_data_lake.engine.brain import make_brain

    def temp_trace(neutered, n=40):
        b = make_brain(seed=0)
        if neutered:
            import snn_data_lake.sensorimotor.afferent_boundary as ab

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

    # ---- T052-iii: the vision/audition event-frame contracts (identity-on-valid,
    # fail-closed-to-no-percept), proven against the REAL tracked pack -- LOUD when the
    # pack is absent (the percept_surprise precedent: a tracked artifact missing is a
    # broken checkout, not an optional extra).
    import pathlib as _pl

    import numpy as _np

    # repo root via the percept_surprise precedent (its :293 resolves the SAME pack this
    # way) -- NOT via gradient_claim_gate._repo_root: importing a gate from this LIVE
    # module made three exempt gate modules statically reachable and fired the
    # stale-EXEMPT guard (caught in-wave, the T036 same-run precedent).
    _repo = _pl.Path(__file__).resolve().parents[3]
    _pack_p = _repo / "models" / "nmnist_pack.npz"
    assert _pack_p.exists(), (
        f"T052-iii contract check CANNOT VERIFY: {_pack_p} absent -- the pack is TRACKED; "
        "regenerate with `python -m training.make_packs`."
    )
    _fr = _np.load(_pack_p)["frames"][0]
    _out, _pres = transduce_vision_events(_fr)
    assert _pres and _out is _fr, (
        "a valid pack frame must pass IDENTITY (the encoder sees the pack's own bytes)"
    )
    for _bad in (None, _fr[0], _fr.reshape(-1)):
        _o2, _p2 = transduce_vision_events(_bad)
        assert _o2 is None and not _p2, "missing/mis-shaped frames must be (None, False)"
    _nanf = _fr.copy()
    _nanf[0, 0, 0, 0] = float("nan")
    _o3, _p3 = transduce_vision_events(_nanf)
    assert _o3 is None and not _p3, "a non-finite frame must be (None, False) -- fail-closed"
    # AUDITION: the FULL battery against ITS OWN tracked pack -- the first draft ran a
    # single synthetic wrong-shape arm while the printed claim covered both modalities
    # (the adversarial panel measured 1 of 6 arms; shd_pack.npz sits tracked in the
    # same tree). Symmetric with vision, plus the panel-forced type/dtype/negativity
    # reds on both lanes.
    _apack_p = _repo / "models" / "shd_pack.npz"
    assert _apack_p.exists(), (
        f"T052-iii contract check CANNOT VERIFY: {_apack_p} absent -- the pack is TRACKED; "
        "regenerate with `python -m training.make_packs`."
    )
    _afr = _np.load(_apack_p)["frames"][0]
    _ao, _ap = transduce_audition_events(_afr)
    assert _ap and _ao is _afr, "a valid SHD pack frame must pass IDENTITY"
    for _abad in (None, _afr[0], _afr.T, _afr.reshape(-1)):
        _o4, _p4 = transduce_audition_events(_abad)
        assert _o4 is None and not _p4, "missing/mis-shaped audition frames must be (None, False)"
    _ananf = _afr.copy()
    _ananf[0, 0] = float("nan")
    assert transduce_audition_events(_ananf) == (None, False), "non-finite audition -> fail-closed"
    assert transduce_vision_events(_fr.astype("float64")) == (None, False), "f8 must red (f4 port)"
    assert transduce_audition_events(_afr.astype("int32")) == (None, False), "i4 must red"
    _negf = _fr.copy()
    _negf[0, 0, 0, 0] = -1.0
    assert transduce_vision_events(_negf) == (None, False), "negative event counts must red"

    class _DuckProbe4D:  # harness-local: a plausible .shape on a non-ndarray (the
        shape = VISION_FRAME_SHAPE  # duck-typing hole a CPU torch tensor walked through)

    assert transduce_vision_events(_DuckProbe4D()) == (None, False), (
        "a non-ndarray with a plausible .shape must red (the type is named, not duck-typed)"
    )

    print(
        f"\nself-check OK: the sentinel is (nan, False) and FAIL-CLOSED end-to-end on a "
        f"real arm (lazy-forwarded missing metres -> EMERGENCY_STOP, matching the "
        f"incumbent, with `present` carrying the no-US half); temperature via the 014 "
        f"normalizer with both divergence halves disclosed; the fold matches _safe_ctx "
        f"verbatim with wrap semantics; proprio neuter A/B: live {len(live_vals)} distinct "
        f"values vs neutered {len(dead_vals)} (the causal loop the DECLARED lane closes). "
        f"The ranging pair is CONTRACT-until-producer, stated in the inventory; the "
        f"vision/audition contracts are identity-on-valid against the real pack, "
        f"fail-closed to no-percept on missing/mis-shaped/non-finite (T052-iii)."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
