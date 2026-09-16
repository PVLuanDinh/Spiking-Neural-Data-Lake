"""Efferent boundary (specs/022 T037): the declared efferent port with an EFFERENCE-COPY
field, wired AROUND the safety envelope -- `So101Arm.apply()` is untouched, this boundary
OBSERVES every envelope passage OF THE LIVE COMPOSITION and hands the copy to the
cerebellar forward model. Scope, stated exactly: every apply() issued through
AutonomousArm.apply_observed -- all four of step()'s sites (main action, EMERGENCY_STOP,
the HOME self-clear, the postural recovery) AND the cockpit's direct efferent route.
Bodies other modules construct for their own self-checks (cpg_oscillator, spiking_pid,
wavefront_planner, demo) are separate So101Arm instances outside this composition and
are not claimed.

CONSUMER DISPOSITION (dated, the T028 pattern): today the trained model and last_copy
are read by this module's own exit and the embodiment tests -- REACHABLE, not yet
faculty-consumed. The spec's consumers are T069 (the cerebellar rule re-fit to this
plant, US3) and T079 (the agency instrument's efference-copy mismatch red, US4); this
lane is their declared substrate, and the P6 conversation claim is theirs to discharge,
not this commit's. Stated here so the lane is never read as already charged.

WHAT THE COPY IS (and why it is not the reafference the arm already keeps): an efference
copy is the brain's record of WHAT IT JUST COMMANDED, paired with what the body then did.
The copy lives in JOINT-SPACE DELTAS through afferent_boundary.proprioception_delta
(each per-joint delta wrapped to [-180, 180) BEFORE normalising -- the T037 review panel
proved that differencing the ABSOLUTE fold, which wraps at 0/360, manufactured phantom
half-range commands on 25% of passages and the whole first-draft charge with them):
    state  s = the PREVIOUS passage's achieved delta   (servo momentum; 0.0 at birth)
    command u = fold_delta(actual BEFORE -> commanded)  (the commanded move, signed)
    next     = fold_delta(actual BEFORE -> actual AFTER) (the achieved move, signed)
The servo LAGS the command (the plant), so next != u and the mapping is genuinely
learnable -- the cerebellar `ForwardModel` (the incumbent, reused whole; its `update` is
the LOCAL delta rule its own GRADIENT_CLAIM declares) learns the servo response online,
and its RESIDUAL is the forward-model error the neuter exit measures: cut the copy
(u = 0) and prediction degrades, measured, or this boundary is decoration.

THE ENVELOPE TICK: `envelope_tick` counts passages through the safety envelope observed
by this boundary -- every apply() the arm's step issues, E-STOP passages included (a stop
is a passage, and a LATCHED body still counts passages while its joints stay frozen).
The embodiment contract (tests/test_efferent_boundary.py) pins step-and-tick moving 1:1,
the latch schedule around it, and that a NEUTERED tap changes nothing about the action
sequence or trajectory -- observation that could perturb the envelope would be a defect.
Sleep quiescence needs no claim here: the daemon's sleep phase stops calling step(), and
one physics step per apply() (the caller's tick is the time base) means nothing else can
advance the envelope.

# distinct-from: brainstem/so101_arm.py -- THE envelope; apply() is its gate and this
#   boundary never wraps, replaces, or conditions it -- it reads state() before and
#   after. distinct-from: cerebellum/forward_model.py -- the incumbent model this REUSES
#   (constructs one, calls predict/residual/update; re-implements nothing).
# distinct-from: cerebellum/jepa_world_model.py -- the latent world model; different
#   state space, its own reafference. distinct-from: substrate/afferent_boundary.py --
#   the SENSORY door; this is the MOTOR door, sharing only the fold (on purpose).
# distinct-from: brainstem/autonomous_loop.py -- the loop that owns the action choice;
#   this boundary observes the chosen action's envelope passage. All sites read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.efferent_boundary
"""

import re

from snn_data_lake.brainstem import arm_config
from snn_data_lake.cerebellum.forward_model import ForwardModel
from snn_data_lake.substrate.afferent_boundary import proprioception_delta

# LMS learning rate, derived from the fold's own bounds rather than copied from the
# incumbent's demo (which runs dt=0.1; this boundary runs dt=1.0, so its 0.05 under-drives
# by 10x -- measured on the first draft: 60 passages barely moved b_hat). Stability for
# delta/LMS needs lr * dt^2 * (s^2 + u^2) < 2; the delta fold pins |s| <= 1 and |u| <= 1,
# so the worst case is lr * 2 -> lr = 0.5 sits at half the bound.
_LR = 0.5

#: The joint names in qpos order -- `arm_config.JOINTS` is the vocabulary, this is its name
#: projection. It exists so `u_j[i]`, `ds_j[i]` and `theta_deg[i]` are the SAME joint: the
#: per-joint copy is indexed exactly like the body's own posture vector, never by a private
#: ordering of its own.
_QPOS_NAMES = tuple(j.name for j in arm_config.JOINTS)

#: `JOINT_<NAME>_ROTATE(<+/-N>deg)` and ONLY that verb. ROTATE is RELATIVE (arm_config's
#: two-verb note: it adds to the current setpoint), so its argument IS a commanded delta;
#: MOVETO is ABSOLUTE, so its argument is a setpoint and a delta parser that read it would
#: report a target ANGLE as a move. The joint name this regex captures is checked against
#: `arm_config.ROTATABLE` membership before it is believed -- arm_config states the rule in
#: its own words ("a site that matches the shape without checking membership here can mint a
#: token for a joint that does not exist"), and the syntax alone is not the vocabulary.
_ROTATE_TOKEN = re.compile(r"JOINT_([A-Z_]+)_ROTATE\(([+-]?\d+(?:\.\d+)?)deg\)")


# distinct-from: substrate/afferent_boundary.py::proprioception_delta -- the SCALAR this
#   function is the per-joint half of; specs/022 T079 2.1 folds the two into one mechanism at
#   2.9 step 4 (the scalar becomes the mean of this list). Read.
# distinct-from: brainstem/autonomous_loop.py::_pfc_bias -- also returns a zero-filled row when
#   its faculty is opt-out, but that row is a top-down PREFERENCE over the ACTION alphabet fed
#   into selection; this is a per-JOINT proprioceptive fold that drives nothing. Read: no
#   quantity, alphabet or consumer is shared.
def proprioception_deltas(theta_before, theta_after):
    """STUB (specs/022 T079, 2.9 step 2) -- returns zeros, so `tests/test_agency_monitor.py`'s
    R5 runs RED against it. The real per-joint fold lands at 2.9 step 4.

    Readability is already the real thing and not stubbed: a missing, short or unreadable body
    reads None, never a fabricated row a consumer would mistake for "no motion"."""
    if not theta_before or not theta_after:
        return None
    n = min(len(theta_before), len(theta_after))
    if n < 2:
        return None
    return [0.0] * n


# distinct-from: brainstem/live_arm.py::_joint_obs_delta -- the ONE-joint spelling of this
#   parse (shoulder_lift only, its own private regex). 2.1 makes that a call into this at 2.9
#   step 4 so exactly one parser survives (P2). Read.
# distinct-from: brainstem/interpreter.py -- VALIDATES a command against the envelope's
#   allow-list and issues it; this reads a already-issued token's commanded magnitude and
#   issues nothing. Read.
def intended_deltas(action):
    """STUB (specs/022 T079, 2.9 step 2) -- returns zeros, so `tests/test_agency_monitor.py`'s
    R4 runs RED against it. The real token parser lands at 2.9 step 4.

    The shape is already fixed: one entry per `arm_config.ROTATABLE` joint, in degrees."""
    return dict.fromkeys(arm_config.ROTATABLE, 0.0)


def _intended_row(action, n):
    """`intended_deltas` projected into qpos order over `n` slots (None when `n` is None)."""
    if n is None:
        return None
    intent = intended_deltas(action)
    return [
        float(intent.get(_QPOS_NAMES[i], 0.0)) if i < len(_QPOS_NAMES) else 0.0 for i in range(n)
    ]


class EfferentBoundary:
    """The efference-copy tap around the envelope. Pure observation plus one LOCAL model
    update per passage; it issues no commands, gates nothing, and cannot touch apply()."""

    def __init__(self, seed=0):
        # a=0, b=0: the model starts ignorant and LEARNS the servo response online --
        # initializing with plant truth would make the neuter exit unmeasurable (a model
        # that never needed the copy cannot miss it).
        self._fm = ForwardModel(a=0.0, b=0.0, dt=1.0, seed=seed)
        self.envelope_tick = 0  # passages observed; the monotonicity subject
        self.last_copy = None  # the most recent efference-copy record
        self.tap_errors = 0  # counted disclosure: tap exceptions swallowed by apply_observed
        self._prev_ds = 0.0  # the previous passage's achieved delta -- the model's state
        self._err_sum = 0.0
        self._err_n = 0

    # distinct-from: cortex/percept_surprise.py -- Hebbian subspace update on percepts;
    #   this is the MOTOR copy's LMS step on the incumbent cerebellar model. Read.
    def observe(self, action, theta_pre, commanded_after, theta_post):
        """ONE envelope passage: count it, form the delta-space copy, score and update
        the forward model. Missing proprioception (any fold None -- a body that cannot
        be read) still COUNTS the passage but carries no model update and leaves the
        held state untouched: a fabricated posture must never train the plant model."""
        self.envelope_tick += 1
        u = proprioception_delta(theta_pre, commanded_after)
        ds = proprioception_delta(theta_pre, theta_post)
        ds_j = proprioception_deltas(theta_pre, theta_post)
        u_j = _intended_row(action, None if ds_j is None else len(ds_j))
        if u is None or ds is None or ds_j is None:
            self.last_copy = {
                "action": action,
                "s": None,
                "next": None,
                "u": None,
                "u_j": None,
                "ds_j": None,
                "err": None,
                "envelope_tick": self.envelope_tick,
            }
            return None
        s = self._prev_ds
        err = abs(self._fm.residual(s, u, ds))
        self._fm.update(s, u, ds, _LR)
        self._prev_ds = ds
        self._err_sum += err
        self._err_n += 1
        self.last_copy = {
            "action": action,
            "s": s,
            "next": ds,
            "u": u,
            # The PER-JOINT copy beside the scalars (specs/022 T079 2.1): `u_j` the commanded
            # delta per joint in DEGREES (the action token's own magnitude), `ds_j` the achieved
            # delta per joint in the SAME units the scalar `u`/`next` are folded in. Both are
            # qpos-ordered, so `u_j[i]` and `ds_j[i]` name one joint. Nothing reads them at 2.9
            # step 2; the agency feed lands at 2.9 step 5.
            "u_j": u_j,
            "ds_j": ds_j,
            "err": err,
            "envelope_tick": self.envelope_tick,
        }
        return err

    def mean_error(self):
        """Running mean |forward-model residual| -- the neuter exit's measured quantity."""
        return (self._err_sum / self._err_n) if self._err_n else None


def declare_ports(manifest):
    """Manifest registrant (specs/022 T037): the efferent lane. `autonomous_loop`
    declares `action_out` (its chosen action token -- symbolic, EVENT-encoded per the
    TYPE HONESTY discipline); this boundary declares `efferent_in` and its subscription
    edge. The cerebellar forward model consumes the copy INSIDE the boundary (the model
    instance lives here), so no out-port is declared for it -- a cross-module edge would
    claim an import cerebellum/forward_model does not make."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "efferent_boundary",
        Port(
            name="efferent_in",
            region="substrate",
            shape=(1,),
            dtype="f4",
            unit="action_token",
            encoding="EVENT",
            kind="IN",
            declaring_module="efferent_boundary",
            join_policy="arbitrate",
        ),
    )
    manifest.declare_edge("autonomous_loop", "action_out", "efferent_boundary", "efferent_in")


def main():
    # T037a: this self-check drives a So101Arm through EMERGENCY_STOP / HOME, which now writes
    # the durable latch sidecar -- under a throwaway home, never the live brain's.
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():
        _main()


def _main():
    print("=" * 74)
    print("EFFERENT BOUNDARY -- efference copy around the envelope; forward-model charge")
    print("=" * 74)

    # A REAL arm drives the whole check: the envelope is passed through apply() exactly
    # as production does (the boundary is already wired into AutonomousArm.step).
    import os
    import tempfile

    os.environ.setdefault("NORD_HOME", tempfile.mkdtemp(prefix="efferent_ab_"))
    import random

    from snn_data_lake.brainstem.autonomous_loop import AutonomousArm

    # The drive PAYS MOTION. A bare loop takes one repeated action, the servo settles on
    # it, u -> 0, and the charge comparison reads 0.0 vs 0.0 -- vacuous (measured on the
    # first run of this check: 40 bare ticks were 40x ENGAGE with 7 nonzero copies; the
    # same drive paying movement took 12 distinct actions with 26 nonzero copies). The
    # known shape "no reward_fn -> no motor actions" strikes again; the reward is part
    # of the instrument here, not part of the claim.
    # distinct-from: cerebellum/wavefront_planner.py -- that executes a waypoint
    #   trajectory through the gate; this is a self-check REWARD paying any motion. Read.
    def _move(unit, action):
        return 1.0 if action != "HOLD" else -1.0

    # (1) THE TAP COUNTS AND THE COPY CARRIES: a real drive through the real envelope,
    # every passage counted, every readable passage yielding a (s, u, next) triple.
    #
    # THREE DELETED CRITERIA, disclosed rather than weakened (P8: deletion on a null),
    # WITH THE MEASURED CAUSE -- the first disclosure draft attributed the nulls to
    # drive properties, and the T037 review panel proved the real cause was the wrap
    # artifact: the first-draft copy differenced the ABSOLUTE 0/360-wrapped fold, which
    # manufactured phantom half-range commands on ~25% of passages. (a) early-vs-late
    # windowing: null EITHER way (windowing measures the stream); (b) online-vs-
    # persistence-twin: null on wrap-free data 5/5 seeds -- deleted; (c) b_hat > 0.25:
    # the floor was PRODUCED ENTIRELY by wrap ticks (wrap-free b_hat is +0.016..+0.038)
    # and its sweep varied the brain seed, the axis that does not move it -- deleted and
    # re-pinned below from a 5-brain-seed x 3-drive-seed grid on the delta-space copy.
    bot = AutonomousArm(seed=0)
    triples = []
    rng = random.Random(4)
    for _ in range(60):
        bot.step(
            visual=[rng.random() for _ in range(4)],
            temperature=[rng.random()],
            audio=[rng.random() for _ in range(4)],
            distance=0.5,
            reward_fn=_move,
        )
        lc = bot.efferent.last_copy
        if lc and lc["err"] is not None:
            triples.append((lc["s"], lc["u"], lc["next"]))
    eb = bot.efferent
    assert eb.envelope_tick >= 60, f"every step passes the envelope, got {eb.envelope_tick}"
    assert len(triples) >= 40, f"a readable body yields copies, got {len(triples)}"

    # (2) NEUTER -> FORWARD-MODEL ERROR CHANGE (the exit), paired on IDENTICAL triples:
    # two fresh models replay the recorded passages online, one seeing the copy
    # (update(s, u, next)) and one with the copy CUT (update(s, 0, next) -- commanded
    # withheld, so b_hat can never learn). All passages compared: in delta space u is
    # nonzero on essentially every tick (the commanded setpoint always differs from the
    # drooped actual), so there is no command-tick subset to select. HONEST MAGNITUDE:
    # the measured gain is REAL but SMALL -- +0.3..+0.9% across the full 5x3 seed grid,
    # direction positive on 15/15 configurations (and 8/8 on the review panel's
    # independent wrap-free re-run). The assert is the direction; the margin is
    # disclosed, not asserted, because at one-tick resolution the servo's response to
    # a single command is a small fraction of its momentum.
    m_live = ForwardModel(a=0.0, b=0.0, dt=1.0, seed=0)
    m_cut = ForwardModel(a=0.0, b=0.0, dt=1.0, seed=0)
    res_live, res_cut = [], []
    for s, u, nxt in triples:
        res_live.append(abs(m_live.update(s, u, nxt, _LR)))
        res_cut.append(abs(m_cut.update(s, 0.0, nxt, _LR)))
    cut_mean = sum(res_cut) / len(res_cut)
    kept_mean = sum(res_live) / len(res_live)
    assert cut_mean > kept_mean, (
        f"CUTTING the efference copy must RAISE the paired forward-model error "
        f"(kept {kept_mean:.5f} vs cut {cut_mean:.5f}) -- else the copy never informed "
        "the prediction and this boundary is decoration"
    )
    # ... and the LEARNING claim, structural on the same pair: the kept model learns the
    # servo's POSITIVE command response. Floor re-pinned on the delta-space copy from the
    # 5-brain-seed x 3-drive-seed grid (b_hat +0.0162..+0.0378; floor = half the observed
    # minimum). The cut model's b_hat is EXACTLY zero -- its update term is
    # lr * e * dt * u with u always 0, so a copy that never arrives can never teach the
    # plant's command response. That zero is the red's reachability proof: break the
    # wire and the learned direction vanishes.
    assert m_live.b_hat > 0.008, (
        f"the kept model must learn the servo's positive command response, b_hat={m_live.b_hat:.4f}"
    )
    assert m_cut.b_hat == 0.0, "a cut copy can never move b_hat -- structural, exact"

    # (3) THE WRAP-SEAM RED (the review panel's constructed input): a joint sitting at
    # MINUS float noise crossing to plus float noise is a nano-move, and the copy must
    # say so. The first draft differenced the absolute 0/360-wrapped fold and this exact
    # input yielded |u| = 0.5 -- half the range, from two nano-degrees of motion.
    eb_seam = EfferentBoundary(seed=0)
    eb_seam.observe("HOLD", [-1e-6] * 6, [1e-6] * 6, [0.0] * 6)
    seam_u = eb_seam.last_copy["u"]
    assert seam_u is not None and abs(seam_u) < 1e-6, (
        f"a nano-degree move across the 0/360 seam must fold to a nano-command, got {seam_u!r}"
    )

    # (4) missing-body honesty: a None fold counts the passage, trains nothing.
    eb2 = EfferentBoundary(seed=0)
    r = eb2.observe("HOLD", None, None, None)
    assert r is None and eb2.envelope_tick == 1 and eb2.last_copy["err"] is None
    assert eb2.mean_error() is None, "no fabricated posture may enter the plant model"

    print(
        f"\nself-check OK: every step passes the envelope (tick {eb.envelope_tick}, "
        f"{len(triples)} copies); cutting the copy raises the paired forward-model error "
        f"{kept_mean:.5f} -> {cut_mean:.5f} (direction asserted; measured gain "
        f"+{(cut_mean - kept_mean) / kept_mean * 100:.1f}%, small and disclosed), and the "
        f"kept model learns the servo's command response (b_hat {m_live.b_hat:+.4f}) while "
        f"the cut model's stays exactly 0.0; the 0/360 seam folds a nano-move to a "
        f"nano-command (|u| {abs(seam_u):.2e}); a missing body counts passages but never "
        f"trains. apply() untouched -- observation only."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
