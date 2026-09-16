r"""
LIVE STATUS, MEASURED 2026-07-28 (v4.80): write-only, and NOT a wire candidate. `observe()` runs every
tick from `brainstem/autonomous_loop.AutonomousArm.step` (guarded by its `agency_monitor=` flag, which
`cortex/brain.py` sets True), and nothing reads `agency()` / `controllability()` / `status()`. The
line cite this row carried (`autonomous_loop.py:1498`) had drifted onto an unrelated comment about
delivering a raw magnitude; naming the SYMBOL instead is what stops it drifting again. The
obvious P3 fix -- consume `route_around` into action selection -- was investigated and REJECTED on the
measurement, not on taste.

Over 600 live ticks exactly ONE channel of seven crossed `warmup`: HOLD, with 129 observations. The
rest got 70, 44, 19, 15, 11 and 6. `agency_signal` took two values (0.0, 0.459), and no channel was
ever reported lost. Wiring `route_around` in that state would scale HOLD's drive by 0.459 and leave
the other six actions untouched -- a spurious penalty on one action, not "route around a dead
effector". It would move a coverage number while making behaviour worse, which is exactly the theater
Policy 0's honesty guard exists to refuse.

The cause is structural and NOT a miscalibrated constant: `warmup=120` is ~2.4x the EMA memory implied
by `alpha=0.02`, which is a principled settle criterion in units of OBSERVATIONS. The live path simply
delivers those observations split across seven channels, so scoring all of them needs roughly 4,000+
ticks. DISPOSITION: P3 exempt-awaiting-sufficient-per-channel-experience. It becomes genuinely useful
on a long-running daemon, and it starts working with no code change -- what it needs is uptime and a
body whose effectors can actually fail.

DISPOSITION SUPERSEDED 2026-09-15 (specs/022 T079, evidence/t079-agency.md 2.6 and 3.4; pin
tools/t079_agency.json): the feed above is RETIRED -- the monitor is now fed the per-joint efference
copy on every envelope passage from `AutonomousArm.apply_observed` (command = the action token's
intended delta per joint, feedback = the achieved delta; one feed, P2) -- and the in-vivo dissociation
read NOT MET: a 12-deg external force on one joint collapses that joint's coupling on 8/8 seeds
((a1) t +13.9998) but two unshoved joints drop by more than the machine's own 0.20 dead-band
((a2) shoulder_lift t +0.8723, wrist_roll t +1.3668) and 4 other
(joint, arm, seed) read 'lost'. P3 disposition: EXEMPT with that record as its reason -- consumed by no
one, measured unable to separate the perturbed channel from the rest on this body; `route_around`
stays unwired; nothing deleted (R1-R6 fire).

Sense-of-agency / per-channel CONTROLLABILITY monitor -- the learned-helplessness
substrate. Asks, for every action channel the brain drives: "do MY commands still
cause their expected sensory change?" and flags the SUSTAINED loss of that
self-causation (deafferentation / learned helplessness) rather than a one-off jolt.

Honest framing (read this first). The atlas already predicts sensory consequences of
action -- cerebellum/forward_model.py runs an efference-copy forward model that flags a
TRANSIENT prediction error when an external force perturbs a reach (one tick, one push).
That is a fast, per-EVENT surprise. It cannot answer the SLOW question this module owns:
has a channel STOPPED being controllable at all -- has the command->feedback coupling
collapsed and STAYED collapsed over a window? A single perturbation and a severed nerve
look identical for one tick; they diverge only when you integrate the coupling over
hundreds of ticks. So this is NOT a re-implementation of the forward model: forward_model
detects a transient external force on an otherwise-controllable limb; AgencyMonitor
detects the persistent DISAPPEARANCE of controllability itself, per channel, with
hysteresis so a brief dropout does not read as helplessness and a single lucky tick does
not read as recovery.

What it computes. Per action channel c it keeps running EMAs (alpha ~ 0.02, i.e. a ~50-
tick memory) of the command c, the feedback f, and the products cf, cc, ff. From those
five EMAs it reads a streaming Pearson magnitude:

    controllability(c) = |E[cf] - E[c]E[f]| / sqrt(var_c * var_f)     (eps-guarded)

= how reliably this channel's commands still produce their expected sensory change,
in [0,1]. High (~1) means "when I command, the world responds as it should"; ~0 means
the feedback has decoupled from the command -- it went to noise, or froze constant, or
the effector/afferent is gone. A channel whose COMMAND is not being exercised (var_c
below a floor) is left UNSCORED: you cannot know a channel is controllable if you never
drive it (this mirrors the forward model staying quiet on an unused limb).

Sustained-loss detection (the helplessness signal, with hysteresis). controllability is
already EMA-smoothed, and on top of that a two-counter state machine adds the window and
the hysteresis the transient forward model deliberately lacks:
  * status starts 'ok'. If controllability(c) < dead_thresh for > delay CONSECUTIVE
    scored ticks -> status[c] = 'lost' (a real, integrated loss of self-causation).
  * RECOVER 'lost' -> 'ok' only after recover_n CONSECUTIVE readings >= high_thresh
    (high_thresh > dead_thresh -> a dead-band so it cannot chatter). A single good tick
    NEVER flips a lost channel back; helplessness is sticky, exactly as learned.
agency() fuses the channels into ONE scalar agency_signal = mean(controllability over
scored channels), and emits a route-around hint: the lost channels (down-weighted to
0.0) plus reroute_to = the most controllable remaining channel.

What it is -- and is not. A bounded, FUNCTIONAL controllability estimator: a felt "sense
of agency" is the interpretation, not a claim of one. The signal is command/feedback
covariance over a window, not an introspective report. It does not itself act; it only
reports which channels the brain can still steer.

Downstream integration (DOCUMENTED here, NOT wired in this file -- do not edit those
modules from here):
  * cortex/self_model.py -- agency_signal would feed self_model.observe(...) as a
    sense-of-agency channel, folding "how much of my world do I still control" into the
    brain's self-episode / confidence stream (registering a new tracked key there is a
    separate change, out of scope for this module).
  * basal_ganglia action-gating -- the per-channel controllability (and the lost set /
    reroute_to hint) would DOWN-WEIGHT selection of uncontrollable actions, so effort is
    spent where commands still have leverage instead of on a dead channel: the direct
    antidote to learned helplessness.

Determinism: per-instance random.Random(seed)
used ONLY for tie-breaks; zlib.crc32 for channel keys (never builtin hash()); ASCII-only
prints; plain-float EMAs updated in a fixed order so two seed=0 runs are byte-identical.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.agency_monitor
"""

import json
import math
import random
import statistics
import sys
import time
import zlib
from pathlib import Path

from snn_data_lake.brainstem import arm_config
from snn_data_lake.substrate._mathutils import clamp01, pearson, q95
from snn_data_lake.substrate.afferent_boundary import proprioception_delta
from snn_data_lake.substrate.efferent_boundary import (
    _ROTATE_TOKEN,
    intended_deltas,
    proprioception_deltas,
)
from snn_data_lake.substrate.paired_contrast import T_BAR, contrast

__version__ = "0.1.0"


#: MERGED 2026-08-15 into `substrate/_mathutils.clamp01` (8 copies -> 1). Alias keeps the local
#: spelling so no call site changes; the MECHANISM is singular.
_clamp01 = clamp01


class AgencyMonitor:
    """Per-channel controllability / sense-of-agency monitor (learned-helplessness substrate).

    observe(channel, cmd, fb) folds one command/feedback pair into that channel's EMAs and
    advances its state machine. controllability(channel) reads the streaming |Pearson| of
    command vs feedback. A channel that stays below dead_thresh for > delay consecutive
    scored ticks becomes 'lost' (sustained loss of self-causation); it recovers to 'ok'
    only after recover_n consecutive readings >= high_thresh (hysteresis -- one good tick
    never revives it). agency() fuses channels into a mean agency_signal plus a route-around
    hint. Deterministic; per-instance random.Random(seed) breaks ties only."""

    def __init__(
        self,
        alpha=0.02,
        dead_thresh=0.35,
        high_thresh=0.55,
        delay=48,
        recover_n=32,
        warmup=None,
        var_floor=1e-4,
        eps=1e-9,
        seed=0,
    ):
        self.alpha = float(alpha)  # EMA rate (~1/alpha tick memory)
        if not 0.0 < self.alpha <= 1.0:
            # `warmup` is derived by DIVIDING by alpha below, so alpha == 0 is a singularity and a
            # negative alpha is an EMA that runs backwards. Refuse at the boundary rather than
            # producing an infinite settle window nothing would ever clear.
            raise ValueError(f"alpha must be in (0, 1] -- got {self.alpha}")
        self.dead_thresh = float(dead_thresh)  # controllability below this = a "low" reading
        self.high_thresh = float(
            high_thresh
        )  # controllability at/above this = a "high" reading (> dead_thresh: dead-band)
        self.delay = int(delay)  # consecutive low readings before declaring 'lost'
        self.recover_n = int(recover_n)  # consecutive high readings before reviving 'lost' -> 'ok'
        # PER-CHANNEL SETTLE WINDOW, DERIVED FROM alpha (P1). This was the literal 120, while the
        # module docstring above states what that number MEANS: "~2.4x the EMA memory implied by
        # alpha=0.02, which is a principled settle criterion in units of OBSERVATIONS". A derived
        # quantity written as a literal is a constant waiting to go stale -- change alpha and the
        # docstring's justification silently stops describing the code. 2.4/0.02 == 120.0, so the
        # default is numerically unchanged (verified: this module's self-check output is
        # byte-identical). An explicit `warmup=` still overrides, for the same reason the other
        # thresholds are arguments.
        self._warmup_ema_multiple = 2.4  # settle criterion in EMA memories (docstring's "~2.4x")
        self.warmup = (
            int(round(self._warmup_ema_multiple / self.alpha)) if warmup is None else int(warmup)
        )
        self.var_floor = float(
            var_floor
        )  # command variance below this -> channel not exercised -> unscored
        self.eps = float(eps)  # denominator guard for the Pearson ratio
        self.rng = random.Random(seed)  # per-instance RNG (tie-breaks only)
        self._ch = {}  # crc32(channel) -> per-channel state dict

    # ---- channel bookkeeping -------------------------------------------------
    def _key(self, channel):
        """Deterministic channel key via zlib.crc32 (never builtin hash())."""
        return zlib.crc32(str(channel).encode("ascii")) & 0xFFFFFFFF

    def _slot(self, channel):
        """Fetch (or lazily create) a channel's state dict."""
        k = self._key(channel)
        s = self._ch.get(k)
        if s is None:
            s = {
                "label": str(channel),
                "n": 0,
                "ec": 0.0,
                "ef": 0.0,
                "ecf": 0.0,
                "ecc": 0.0,
                "eff": 0.0,
                "below": 0,
                "high": 0,
                "status": "ok",
            }
            self._ch[k] = s
        return s

    # ---- fold one command/feedback pair --------------------------------------
    def observe(self, channel, cmd, fb):
        """Update `channel` with one (command, feedback) pair and advance its state.

        Returns the channel's current status ('ok' or 'lost')."""
        s = self._slot(channel)
        c = float(cmd)
        f = float(fb)
        a = self.alpha
        if s["n"] == 0:
            # lazy init to the first sample -> no cold-start bias toward 0
            s["ec"], s["ef"] = c, f
            s["ecf"], s["ecc"], s["eff"] = c * f, c * c, f * f
        else:
            s["ec"] += a * (c - s["ec"])
            s["ef"] += a * (f - s["ef"])
            s["ecf"] += a * (c * f - s["ecf"])
            s["ecc"] += a * (c * c - s["ecc"])
            s["eff"] += a * (f * f - s["eff"])
        s["n"] += 1
        self._advance(s)
        return s["status"]

    # ---- streaming |Pearson| = controllability -------------------------------
    def _ctrl_var(self, s):
        """Return (controllability, var_c) from a channel's five EMAs."""
        var_c = s["ecc"] - s["ec"] * s["ec"]
        var_f = s["eff"] - s["ef"] * s["ef"]
        if var_c < 0.0:
            var_c = 0.0
        if var_f < 0.0:
            var_f = 0.0
        cov = s["ecf"] - s["ec"] * s["ef"]
        denom = math.sqrt(var_c * var_f)
        if denom < self.eps:
            return 0.0, var_c
        return _clamp01(abs(cov) / denom), var_c

    def controllability(self, channel):
        """|Pearson| of command vs feedback for `channel`, in [0,1] (0 if unseen)."""
        s = self._ch.get(self._key(channel))
        if s is None:
            return 0.0
        return self._ctrl_var(s)[0]

    def status(self, channel):
        """Current status of `channel`: 'ok', 'lost', or 'warmup' (not yet scored)."""
        s = self._ch.get(self._key(channel))
        if s is None:
            return "warmup"
        return s["status"] if s["n"] >= self.warmup else "warmup"

    def snapshot(self, channel):
        """Full per-channel state -- status, controllability, and both hysteresis counters."""
        s = self._ch.get(self._key(channel))
        if s is None:
            return {"status": "warmup", "ctrl": 0.0, "below": 0, "high": 0, "n": 0}
        ctrl, _ = self._ctrl_var(s)
        return {
            "status": self.status(channel),
            "ctrl": ctrl,
            "below": s["below"],
            "high": s["high"],
            "n": s["n"],
        }

    # ---- the sustained-loss state machine (window + hysteresis) --------------
    def _advance(self, s):
        """Advance one channel's status from its current controllability reading."""
        if s["n"] < self.warmup:
            return  # still settling the EMAs
        ctrl, var_c = self._ctrl_var(s)
        if var_c < self.var_floor:
            return  # command not exercised -> cannot assess -> hold
        if ctrl < self.dead_thresh:
            s["below"] += 1
            s["high"] = 0
            if s["status"] == "ok" and s["below"] > self.delay:
                s["status"] = "lost"  # sustained loss of self-causation
        else:
            s["below"] = 0
            if ctrl >= self.high_thresh:
                s["high"] += 1
                if s["status"] == "lost" and s["high"] >= self.recover_n:
                    s["status"] = "ok"  # revived only after recover_n consecutive highs
            else:
                s["high"] = 0  # dead-band reading: breaks both streaks

    # ---- fuse channels into one agency signal + route-around hint ------------
    def agency(self):
        """Fuse scored channels into a mean agency_signal plus a route-around hint.

        Returns a dict: agency_signal (mean controllability over scored channels), lost
        (sorted labels currently 'lost'), route_around (per-channel weight -- 0.0 for lost,
        else controllability), reroute_to (most controllable non-lost channel, or None),
        n_scored (channels past warmup)."""
        ctrls = {}
        lost = []
        for s in self._ch.values():
            if s["n"] < self.warmup:
                continue
            ctrl, _ = self._ctrl_var(s)
            ctrls[s["label"]] = ctrl
            if s["status"] == "lost":
                lost.append(s["label"])
        signal = sum(ctrls.values()) / len(ctrls) if ctrls else 0.0
        route = {lbl: (0.0 if lbl in lost else ctrls[lbl]) for lbl in ctrls}
        reroute_to = self._pick_target(ctrls, lost)
        return {
            "agency_signal": round(signal, 6),
            "lost": sorted(lost),
            "route_around": route,
            "reroute_to": reroute_to,
            "n_scored": len(ctrls),
        }

    def _pick_target(self, ctrls, lost):
        """Most controllable non-lost channel; ties broken by the per-instance RNG."""
        cands = [(c, lbl) for lbl, c in ctrls.items() if lbl not in lost]
        if not cands:
            return None
        best = max(c for c, _ in cands)
        top = sorted(lbl for c, lbl in cands if c >= best - 1e-12)
        return top[0] if len(top) == 1 else self.rng.choice(top)


# =============================================================================
# T079 -- THE AGENCY INSTRUMENT (specs/022 Phase 4 / C4;
# specs/022-unified-brain-arc/evidence/t079-agency.md, frozen at 9b04c8c).
#
# 2.9 step 4: THE INSTRUMENTS, GREEN, each red of 2.4 reddened IN FACT by its own mutated
# scorer (`_t079_mutated_scorer`). Nothing here reads a BODY yet: every input is either a
# constructed stream or a finished recording. The feed switch, the recorder and the
# campaign land at 2.9 step 5.
#
# The instrument is the incumbent `AgencyMonitor` above, whole and un-re-implemented
# (2.1): its EMAs, its thresholds and its two-counter state machine ARE the reading. What
# T079 adds is the FEED (a per-joint efference copy instead of the 2026-07-28 measured-
# meaningless action/qmax/reward triple), a cause-blind twin to read the coupling against,
# and the in-vivo dissociation.
#
# distinct-from: substrate/brain_telemetry.py -- the ONE telemetry CONTRACT, a live
#   per-panel view of the brain's own keys. These functions read a finished RECORDING of
#   envelope passages offline and publish nothing. Read.
# distinct-from: cortex/prior_drift_harness.py -- a scratch-POPULATION drift harness whose
#   "reading" is a host-settling measurement with its subject PID beside it. Different
#   subject, different quantity; its paired statistic is the one this module IMPORTS from
#   `substrate/paired_contrast` rather than re-spelling. Read.
# distinct-from: substrate/amdahl_gate.py -- a live/neuter/sham paired reading over tick
#   COST on three scratch brains. This pairs INTACT against SHOVED on the per-joint
#   command-to-outcome coupling; no timing quantity is read at all. Read.
# distinct-from: cortex/h01_ingest.py -- an Avro OCF record reader over a connectome
#   export. Term overlap ("record", "read") only. Read.
# distinct-from: brainstem/arm_config.py::joint_limits -- per-joint LIMITS read off the
#   MuJoCo model (the safety envelope's vocabulary). This reads a per-joint CORRELATION off
#   a recording and touches no limit. Read; arm_config is imported here, never restated.
# =============================================================================

#: Ticks per recording, per seed, per arm (2.2, 2.8).
T079_N_TICKS = 1200
#: The campaign's seeds (2.2; PLAN-FINAL Part F: S = 8).
T079_SEEDS = tuple(range(8))
#: Permutation draws per joint per recording for the cause-blind twin (2.1, 2.8).
T079_N_SHUFFLE = 2000
#: J*, the perturbed channel: the mid-chain joint (2.2).
T079_SHOVE_JOINT = "elbow_flex"
#: The disturbance amplitude, DERIVED from the command scale rather than chosen (2.2):
#: the shove equals twice the step the loop itself commands.
T079_SHOVE_DEG = 2.0 * arm_config.DEFAULT_STEP_DEG
#: The reported ladder of 2.3 / 2.9 step 5, in the same derived units: half the step, the
#: step, twice the step.
T079_LADDER = (0.5 * arm_config.DEFAULT_STEP_DEG, arm_config.DEFAULT_STEP_DEG, T079_SHOVE_DEG)
#: The per-seed percept-stream offset (2.2): `brain._real_ptr = T079_STREAM_OFFSET * seed`.
T079_STREAM_OFFSET = 150
#: The shove RNG base: `random.Random(T079_SHOVE_RNG_BASE + seed)`, so the disturbance
#: draw is per-seed and the loop's own RNG is never touched (2.1).
T079_SHOVE_RNG_BASE = 7000
#: (p)'s determinism twin: two INTACT seed-0 recordings at this width must be byte-identical.
T079_DETERMINISM_TICKS = 100
#: The one bar (2.3): the paired one-sided t over 8 seeds, 7 d.f., alpha = 0.05.
T079_BAR = T_BAR
#: The pin, written ONCE by `--t079-campaign` (2.1, 2.9 step 6).
T079_PIN = Path("tools/t079_agency.json")

#: The instrument's OWN constants, READ off a default monitor rather than re-declared here
#: (P1, and 2.8 records them as read-not-chosen). A literal 0.20 / 120 / 1e-4 in this module
#: would be three constants to keep in agreement with `AgencyMonitor.__init__` instead of one.
_T079_DEFAULTS = AgencyMonitor(seed=0)
#: (a2)'s DEADBAND: the band the machine ITSELF treats as noise, `high_thresh - dead_thresh`
#: (2.3). A drop smaller than this is a drop the state machine would not act on.
T079_DEADBAND = _T079_DEFAULTS.high_thresh - _T079_DEFAULTS.dead_thresh
#: The settle window, in observations (the monitor derives it from alpha).
T079_WARMUP = _T079_DEFAULTS.warmup
#: The command-variance floor below which a channel is UNSCORED (R3, (r)).
T079_VAR_FLOOR = _T079_DEFAULTS.var_floor
#: The five channels: the rotatable joints (the gripper is ENGAGE/RELEASE and has no angle
#: vocabulary, so it is not a per-joint command/feedback channel at all).
T079_JOINTS = tuple(arm_config.ROTATABLE)
#: The same five as JointSpecs, so the recorder reads `u_j`/`ds_j` by the body's own qpos
#: index and a row's column k is T079_JOINTS[k] on every recording.
_T079_SPECS = tuple(next(j for j in arm_config.JOINTS if j.name == n) for n in T079_JOINTS)

#: The dry run's width (2.9 step 5: ONE seed at 300 ticks).
T079_DRY_TICKS = 300
#: 2.8's n_shuffle sensitivity, COMPUTED in the dry run: this many INDEPENDENT floors of the
#: same recording, whose spread IS the floor's Monte-Carlo sd.
T079_FLOOR_REPEATS = 12
#: The seed base those independent floors are drawn at (the generator is keyed on (seed,
#: joint), so varying this and nothing else varies only the draw).
T079_FLOOR_SD_BASE = 900000

#: The REDS' shuffle count. BUILDER-INTRODUCED and disclosed (2.9 step 4): 2.1 fixes
#: n_shuffle = 2,000 for the CAMPAIGN's floors, and `t079_reds()` is executed by every
#: run_selfchecks pass, where 2,000 draws x 5 joints x 25 recordings would dominate the
#: gate. 2.4 fixes no draw count for the reds themselves. Measured cost is printed by the
#: self-check line, so the trade is visible rather than asserted.
T079_REDS_N_SHUFFLE = 200
#: R1's width, VERBATIM from 2.4 ("600 passages").
T079_R1_PASSAGES = 600
#: R6's width and disturbance. 2.4 fixes the decoupling point (passage 300) and the
#: amplitude (U(-2, 2) x command scale) and NOT n; 900 leaves 600 decoupled passages, which
#: is what the machine needs -- the alpha = 0.02 EMA (a ~50-observation memory) has to decay
#: THROUGH the coupled history before `delay` = 48 consecutive low readings can start
#: counting. BUILDER-INTRODUCED, disclosed.
T079_R6_PASSAGES = 900
T079_R6_AT = 300
T079_R6_AMP = 2.0
#: R6's COMMAND DUTY CYCLE: one passage in this many carries a command, the rest are 0.
#: NOT free-floating -- it is 1.3's measured shape for this body (each rotatable joint was
#: commanded on 14 / 27 / 15 / 17 / 16 of 146 passages, roughly one in ten), and it is what
#: decides whether a U(-2, 2) x step disturbance can collapse the coupling at all: the
#: correlation a shove leaves behind is set by the ratio of command variance to disturbance
#: variance, so a body commanded on every passage cannot be dissociated by this amplitude.
T079_R6_DUTY = 10

#: One ASCII line per red, printed beside its boolean so a gate log says WHAT fired rather
#: than only that something did (2.4's wording, compressed).
T079_RED_LINES = {
    "R1": "the shuffle floor is a FLOOR (an un-permuted floor collapses the coupled margin)",
    "R2": "the machine's hysteresis holds both ways (delay = recover_n = 0 reddens it)",
    "R3": "an unexercised channel is UNSCORED, never lost (var_floor := 0 reddens it)",
    "R4": "the token parser refuses a foreign joint (accepting any name reddens it)",
    "R5": "the per-joint fold wraps and its mean IS the scalar (dropping the wrap reddens it)",
    "R6": "the dissociation reads on a constructed body (1 - c on the shoved arm reddens (a2))",
}


# ---- the constructed streams every red runs on (2.4) ---------------------------------
# They live HERE and not only in tests/test_agency_monitor.py because `t079_reds()` needs
# them: a gate run that never executes pytest still has to run every red on the same
# stream the suite runs it on (T077's in-command control; T078's shape).
def t079_channel_rows(u_cols, ds_cols, monitor=None):
    """Feed constructed per-joint channels through the SHIPPED `AgencyMonitor` and return
    rows in the RECORDER's own shape (2.1), so a red and a live recording are read by the
    same `t079_read`. `monitor` is the mutation hook (R2's no-hysteresis twin, R3's
    var_floor := 0 twin); None means the shipped defaults."""
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
                "awake_ticks": p + 1,
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


def t079_coupled_decoupled(n, seed, gain=0.9, noise=0.05):
    """R1's channels in the five-joint frame: joint 0 COUPLED (fb = gain u + noise), joint 1
    DECOUPLED (fb drawn independently of u), joints 2-4 coupled filler."""
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


def t079_dead_channel(n, seed):
    """R3's recording: joint 0 is NEVER commanded (u == 0 on every passage, its feedback
    still moving -- a channel you cannot assess, not a channel you have lost); the other
    four are exercised and coupled."""
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


def t079_body_channels(
    seed, shove=None, amp=T079_R6_AMP, n=T079_R6_PASSAGES, at=T079_R6_AT, stream=0
):
    """R6's constructed body: five channels commanded in {-step, 0, +step} with the servo
    following at gain 0.93, and -- when `shove` names one -- that joint carrying an additive
    U(-amp, amp) x command-scale disturbance from passage `at` onward. `stream` selects an
    INDEPENDENT noise realisation of the same body, which is how R6's no-decoupling twin is
    built (an identical construction would make every paired difference exactly 0.0 and the
    twin's null degenerate rather than measured)."""
    rng = random.Random(1000 + seed + 100003 * stream)
    step = arm_config.DEFAULT_STEP_DEG
    alphabet = (-step, step) + (0.0,) * (T079_R6_DUTY - 2)
    u_cols, ds_cols = [], []
    for j in T079_JOINTS:
        u = [rng.choice(alphabet) for _ in range(n)]
        ds = [0.93 * c + 0.02 * step * rng.uniform(-1.0, 1.0) for c in u]
        if shove == j:
            ds = [
                d + (amp * step * rng.uniform(-1.0, 1.0) if p >= at else 0.0)
                for p, d in enumerate(ds)
            ]
        u_cols.append(u)
        ds_cols.append(ds)
    return u_cols, ds_cols


def _t079_variance(xs):
    """Population variance of the commanded deltas over a window; 0.0 on an empty window."""
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def t079_read(rows, var_floor=T079_VAR_FLOOR, warmup=T079_WARMUP):
    """2.1's per-joint reading of one recording, and 2.3's (r) beside it.

    Per joint: `c` = the MEAN of the instrument's OWN streaming `controllability(j)` over
    the passages at or past `warmup` (not the last value -- a single final reading is one
    EMA sample of a quantity the machine spent the whole recording estimating); `status` as
    the machine left it; `scored` = the joint reached warmup AND its commanded deltas vary
    by at least `var_floor` over that window; `n`, `var_c` and the scored-passage count.

    TWO INSTRUMENTS ON "EXERCISED", DELIBERATELY: `var_c` here is the PLAIN population
    variance of `u_j` over the scored window, read off the recording; the machine's own gate
    is an EMA variance read inside `_advance`. They answer the same question from different
    ends -- the recording's and the estimator's -- and R3 reddens both with one mutation
    (var_floor := 0 scores the dead channel AND lets its status reach 'lost').

    `readable` IS 2.3's (r) for this one recording: every rotatable joint scored, every
    awake tick carrying at least ONE passage (the T037 1:1 step-and-tick contract, read per
    tick; 2.10), `tap_errors` 0 and the body never latched. A recording that fails it is
    REFUSED, never read around. EXTRA passages -- the postural-recovery reflex issues a
    SECOND tapped `apply_observed` on ground contact (30 of 593 awake ticks on the
    campaign-width seed-0 apparatus run, `scratchpad/t079/probe_width.py`) -- are counted as
    `extra_passages`, fed and reported, never a refusal: a command with its outcome is exactly
    what the copy should see.

    `var_floor` and `warmup` are arguments for the same reason `AgencyMonitor`'s thresholds
    are: R3's mutated scorer has to be able to set the floor to zero and READ the difference."""
    joints = {}
    last = rows[-1] if rows else None
    for k, j in enumerate(T079_JOINTS):
        scored_rows = [r for r in rows if r["n"][k] >= warmup]
        ctrls = [float(r["ctrl"][k]) for r in scored_rows]
        var_c = _t079_variance([float(r["u"][k]) for r in scored_rows])
        joints[j] = {
            "c": (sum(ctrls) / len(ctrls)) if ctrls else 0.0,
            "status": None if last is None else last["status"][k],
            "scored": bool(ctrls) and var_c >= var_floor,
            "n": 0 if last is None else int(last["n"][k]),
            "var_c": var_c,
            "scored_passages": len(ctrls),
        }
    passages = len(rows)
    awake = 0 if last is None else int(last.get("awake_ticks", passages))
    observed_ticks = len(
        {
            int(r["awake_ticks"])
            for r in rows
            if r.get("awake", True) and r.get("awake_ticks") is not None
        }
    )
    extra = passages - awake
    latched = sum(1 for r in rows if r.get("latched"))
    tap_errors = 0 if last is None else int(last.get("tap_errors", 0))
    # A passage whose per-joint copy could not be read fed the monitor NOTHING (the feed
    # skips an unreadable body rather than handing it a row of zeros), so its `u`/`ds`
    # columns are placeholders. Counted here and refused by (r): a recording with a hole in
    # it is not a recording with a smaller n.
    unreadable = sum(1 for r in rows if not r.get("copy_ok", True))
    return {
        "readable": bool(rows)
        and all(v["scored"] for v in joints.values())
        and observed_ticks == awake
        and latched == 0
        and tap_errors == 0
        and unreadable == 0,
        "passages": passages,
        "awake": awake,
        "observed_ticks": observed_ticks,
        "extra_passages": extra,
        "latched": latched,
        "tap_errors": tap_errors,
        "copies_unreadable": unreadable,
        "joints": joints,
    }


def _t079_draws(u_col, ds_col, first_scored, n_shuffle, gen, mon, permute=True):
    """THE CAUSE-BLIND TWIN's draws for one joint: `n_shuffle` runs of the SAME EMA
    recurrence `AgencyMonitor.observe` / `_ctrl_var` implement, batched as one
    (n_shuffle x passages) torch recurrence on CPU.

    The recurrence is sequential in PASSAGES (an EMA is), and vectorised across DRAWS -- a
    per-draw Python loop is what 2.1 forbids, not a per-passage one. Feedback is identical
    on every draw (only the command is permuted), so `ef`/`eff` stay scalars and only
    `ec`/`ecf`/`ecc` carry a draw axis.

    `permute=False` is R1's `floor_unpermuted` mutation AND this recurrence's own positive
    control: with the identity permutation every draw must reproduce the shipped machine's
    own `c_j` to floating-point, which is what R1 asserts rather than assumes."""
    import torch

    n = len(u_col)
    u = torch.tensor(u_col, dtype=torch.float64)
    f = torch.tensor(ds_col, dtype=torch.float64)
    idx = torch.arange(first_scored, n, dtype=torch.long)
    cmd = u.unsqueeze(0).repeat(n_shuffle, 1)
    if permute and len(idx) > 1:
        perm = torch.argsort(torch.rand(n_shuffle, len(idx), generator=gen), dim=1)
        cmd[:, idx] = u[idx][perm]

    a = mon.alpha
    ec = ecf = ecc = None
    ef = eff = 0.0
    total = torch.zeros(n_shuffle, dtype=torch.float64)
    counted = 0
    for p in range(n):
        c = cmd[:, p]
        fv = float(f[p])
        if p == 0:
            ec, ecf, ecc = c.clone(), c * fv, c * c
            ef, eff = fv, fv * fv
        else:
            ec = ec + a * (c - ec)
            ecf = ecf + a * (c * fv - ecf)
            ecc = ecc + a * (c * c - ecc)
            ef += a * (fv - ef)
            eff += a * (fv * fv - eff)
        if p >= first_scored:
            var_c = torch.clamp(ecc - ec * ec, min=0.0)
            var_f = max(eff - ef * ef, 0.0)
            cov = ecf - ec * ef
            denom = torch.sqrt(var_c * var_f)
            ratio = torch.clamp(cov.abs() / torch.clamp(denom, min=1e-300), 0.0, 1.0)
            total = total + torch.where(denom < mon.eps, torch.zeros_like(ratio), ratio)
            counted += 1
    return (total / max(counted, 1)).tolist()


def t079_floor(rows, seed, n_shuffle=T079_N_SHUFFLE, warmup=T079_WARMUP, permute=True):
    """2.1's CAUSE-BLIND TWIN, per joint: the same feed with `u_j` PERMUTED across the
    scored passages -- the outcomes identical, the causal link removed, which is word for
    word the dissociation input `conclusion_protocol.CONCLUSION["agency"]` declares.

    `floor` is `_mathutils.q95` of the `n_shuffle` mean-controllability draws and
    `floor_median` sits beside it (2.3 reports both). The RNG is a `torch.Generator` seeded
    per (recording seed, joint); the global RNG is never touched, so two runs of the same
    recording draw the same floor.

    DISCLOSED, and CONSERVATIVE: the recurrence runs over the WHOLE recording and the
    permutation covers only the scored window, exactly as 2.1 words it, so the EMA carries
    real coupling across the warmup boundary and decays through it (~50 observations at
    alpha 0.02, against a scored window of hundreds). That biases the floor UP, which makes
    (g0) harder to meet, never easier."""
    joints = {}
    first = max(int(warmup) - 1, 0)
    ok = bool(rows) and len(rows) > first
    for k, j in enumerate(T079_JOINTS):
        if not ok:
            joints[j] = {"floor": 0.0, "floor_median": 0.0, "draws": 0}
            continue
        import torch

        gen = torch.Generator()
        gen.manual_seed((int(seed) * 1000003 + zlib.crc32(j.encode("ascii"))) & 0x7FFFFFFF)
        draws = _t079_draws(
            [float(r["u"][k]) for r in rows],
            [float(r["ds"][k]) for r in rows],
            first,
            int(n_shuffle),
            gen,
            _T079_DEFAULTS,
            permute=permute,
        )
        joints[j] = {
            "floor": q95(draws),  # q95 SORTS its argument; `draws` is local and dead after
            "floor_median": statistics.median(draws),
            "draws": len(draws),
        }
    return {
        "readable": ok,
        "n_shuffle": int(n_shuffle),
        "seed": int(seed),
        "joints": joints,
    }


def _t079_score(entry, joint, arm):
    """The barred quantity, read from one arm of one seed: the instrument's OWN
    controllability for that joint. `arm` is 'intact' or 'shoved'; the default reading does
    not use it, and R6's `blind_scorer` mutation does."""
    return float(entry["read"]["joints"][joint]["c"])


def t079_contrasts(intact, shoved, deadband=T079_DEADBAND, score=_t079_score, bar=T079_BAR):
    """2.3's conjuncts, each decided by `substrate/paired_contrast.contrast` (the ONE
    statistic; nothing is re-spelled here).

    `intact` and `shoved` are per-seed lists of `{"read": t079_read(...), "floor":
    t079_floor(...)}`, index-paired -- the floor travels WITH its arm because (g0) reads the
    intact arm against its OWN twin, on its own seed and its own stream.

      (g0) per joint: t over seeds of (c_j^intact - floor_j^intact). The positive control
           that is NOT the method: the twin is a different construction of the same
           outcomes, never a re-reading of the statistic.
      (a1) t over seeds of (c_J*^intact - c_J*^shoved).
      (a2) per unshoved joint k: t over seeds of ((c_k^shoved - c_k^intact) + DEADBAND),
           the band the machine itself treats as noise.
      (a3) EXACT COUNTS, no t: `status_J*^shoved` is 'lost' on every seed AND no other
           (joint, arm, seed) is.

    `score` is R6's mutation hook and `deadband` is the reported sensitivity knob of 2.4's
    `deadband_zero` (a constant's sensitivity, explicitly NOT a red)."""
    seeds = range(min(len(intact), len(shoved)))
    floored = bool(len(intact)) and all(intact[s]["floor"]["readable"] for s in seeds)
    # NO TWIN, NO (g0). A caller that skipped the floor gets None, never a contrast read
    # against a floor of 0.0 -- which every c would clear, and which would look like the
    # positive control passing.
    g0 = (
        {
            j: contrast(
                f"(g0) {j}: intact c - its cause-blind floor",
                [
                    score(intact[s], j, "intact") - float(intact[s]["floor"]["joints"][j]["floor"])
                    for s in seeds
                ],
                bar,
            )
            for j in T079_JOINTS
        }
        if floored
        else None
    )
    a1 = contrast(
        f"(a1) {T079_SHOVE_JOINT}: intact c - shoved c",
        [
            score(intact[s], T079_SHOVE_JOINT, "intact")
            - score(shoved[s], T079_SHOVE_JOINT, "shoved")
            for s in seeds
        ],
        bar,
    )
    a2 = {
        j: contrast(
            f"(a2) {j}: (shoved c - intact c) + {deadband:g}",
            [
                score(shoved[s], j, "shoved") - score(intact[s], j, "intact") + deadband
                for s in seeds
            ],
            bar,
        )
        for j in T079_JOINTS
        if j != T079_SHOVE_JOINT
    }
    star_lost = sum(
        1 for s in seeds if shoved[s]["read"]["joints"][T079_SHOVE_JOINT]["status"] == "lost"
    )
    other_lost = 0
    for s in seeds:
        for arm, entries in (("intact", intact), ("shoved", shoved)):
            for j in T079_JOINTS:
                if j == T079_SHOVE_JOINT and arm == "shoved":
                    continue
                if entries[s]["read"]["joints"][j]["status"] == "lost":
                    other_lost += 1
    n_seeds = len(seeds)
    return {
        "g0": g0,
        "a1": a1,
        "a2": a2,
        "a3": {
            "shoved_star_lost": star_lost,
            "other_lost": other_lost,
            "met": bool(n_seeds) and star_lost == n_seeds and other_lost == 0,
        },
        "n_seeds": n_seeds,
    }


# distinct-from: substrate/eprop_transport_gate.py, substrate/gradient_claim_gate.py --
#   census gates that CLASSIFY existing code. Every entry below is a deliberately broken
#   twin of THIS module's own instrument, constructed so a red can be shown to fire rather
#   than argued to (P8 clause 1). Read; nothing here inspects or rewrites any other module.
# distinct-from: cortex/attention_schema.py::_t078_mutated_scorer and
#   cortex/metacognitive_monitor.py's equivalent -- the PATTERN this follows (2.1 names them
#   as the thing to reuse) over DIFFERENT instruments: an MI floor and a decoder band there,
#   a permutation floor over an EMA correlation and a token parser here. One catalog per
#   module is this tree's convention, not a second mechanism. Read.
def _t079_mutated_scorer(kind):
    """2.4's six mutations, RUN and not reasoned. Each is the deliberately broken twin of
    the instrument whose red it must redden, and each is exercised BOTH in
    `tests/test_agency_monitor.py` and in `t079_reds()` below.

      floor_unpermuted (R1) -- the floor is drawn from the UN-shuffled sequence, so every
          draw equals the reading itself and the decoupled channel reads "above" its own
          floor. Doubles as the batched recurrence's positive control (it must reproduce the
          shipped machine's c_j exactly).
      no_hysteresis    (R2) -- delay = recover_n = 0, so ONE low reading flips 'ok' to
          'lost' and ONE high reading flips it back: the window and the dead-band both gone.
      no_floor         (R3) -- var_floor = 0 in BOTH the machine and the read, so a channel
          that was never commanded is scored, is allowed to go 'lost', and (r) reads a
          recording the pre-registration says must be refused.
      parse_any_name   (R4) -- the parser trusts the regex instead of the vocabulary, so a
          ROTATE token naming a joint outside `arm_config.ROTATABLE` mints a channel.
      no_wrap          (R5) -- the per-joint delta is not wrapped to [-180, 180), so the
          T037 seam returns: a 359.9 -> 0.1 nano-move reads as a -359.8 deg sweep.
      blind_scorer     (R6) -- c_j is replaced by 1 - c_j on the SHOVED arm, which inverts
          (a1): a collapse reads as an improvement.

    2.4's `deadband_zero` is NOT here and that is deliberate: 2.4 declares it a REPORTED
    sensitivity of a declared constant, not a red. It is `t079_contrasts(..., deadband=0.0)`,
    called by name where it is reported."""

    def floor_unpermuted(rows, seed, **kw):
        kw["permute"] = False
        return t079_floor(rows, seed, **kw)

    def no_hysteresis(**kw):
        return AgencyMonitor(delay=0, recover_n=0, seed=0, **kw)

    def no_floor(u_cols, ds_cols):
        rows = t079_channel_rows(u_cols, ds_cols, monitor=AgencyMonitor(var_floor=0.0, seed=0))
        return t079_read(rows, var_floor=0.0)

    def parse_any_name(action):
        out = dict.fromkeys(arm_config.ROTATABLE, 0.0)
        m = _ROTATE_TOKEN.fullmatch(action or "")
        if m is None:
            return out
        out[m.group(1)[len("JOINT_") : -len("_ROTATE")].lower()] = float(m.group(2))
        return out

    def no_wrap(theta_before, theta_after):
        if not theta_before or not theta_after:
            return None
        n = min(len(theta_before), len(theta_after))
        if n < 2:
            return None
        return [(float(a) - float(b)) / 180.0 for b, a in zip(theta_before[:n], theta_after[:n])]

    def blind_scorer(entry, joint, arm):
        c = float(entry["read"]["joints"][joint]["c"])
        return (1.0 - c) if arm == "shoved" else c

    return {
        "floor_unpermuted": floor_unpermuted,
        "no_hysteresis": no_hysteresis,
        "no_floor": no_floor,
        "parse_any_name": parse_any_name,
        "no_wrap": no_wrap,
        "blind_scorer": blind_scorer,
    }[kind]


def t079_hysteresis_probe(seed, monitor=None):
    """R2's trajectory through the module's OWN `_experiment` construction (healthy ->
    decoupled -> recovered), instrumented for the two things 2.4 asks about and `_experiment`
    does not return: the status at the FIRST low reading while 'ok', and the low-streak
    length at the flip to 'lost'. The stream, the gain and the phase lengths are
    `_experiment`'s, so the mutated monitor is judged on the same trajectory."""
    mon = AgencyMonitor(seed=seed) if monitor is None else monitor
    rng = random.Random(seed)
    gain, noise = 0.9, 0.05
    out = {
        "first_low_while_ok_status": None,
        "lows_at_flip": None,
        "first_high_while_lost_status": None,
        "highs_at_flip": None,
        "delay": mon.delay,
        "recover_n": mon.recover_n,
    }
    for phase, n in (("heal", 400), ("decouple", 400), ("recover", 500)):
        for _ in range(n):
            c = rng.uniform(-1.0, 1.0)
            fb = (
                rng.uniform(-1.0, 1.0)
                if phase == "decouple"
                else gain * c + noise * rng.uniform(-1.0, 1.0)
            )
            before = mon.snapshot("grasp")
            mon.observe("grasp", c, fb)
            after = mon.snapshot("grasp")
            if after["n"] < mon.warmup:
                continue
            if before["status"] == "ok" and after["ctrl"] < mon.dead_thresh:
                if out["first_low_while_ok_status"] is None:
                    out["first_low_while_ok_status"] = after["status"]
            if (
                before["status"] == "ok"
                and after["status"] == "lost"
                and out["lows_at_flip"] is None
            ):
                out["lows_at_flip"] = before["below"] + 1
            if before["status"] == "lost" and after["ctrl"] >= mon.high_thresh:
                if out["first_high_while_lost_status"] is None:
                    out["first_high_while_lost_status"] = after["status"]
            if (
                before["status"] == "lost"
                and after["status"] == "ok"
                and out["highs_at_flip"] is None
            ):
                out["highs_at_flip"] = after["high"]
    return out


def t079_arm(seed, shove, n_shuffle=T079_REDS_N_SHUFFLE, stream=0, floor=False):
    """One arm of R6's constructed body, read (and optionally floored) in
    `t079_contrasts`' shape.

    `floor` is OFF by default because R6 bars (a1), (a2) and (a3) and none of them reads a
    floor -- (g0) is the INTACT arm's conjunct and R6 does not assert it. Drawing 24 unused
    floors cost 12 of the reds line's 15 seconds, measured; `t079_contrasts` reports
    `g0 = None` rather than a floor of 0.0 when the twin was never drawn, so a skipped
    floor can never be mistaken for a floor of zero."""
    u_cols, ds_cols = t079_body_channels(seed, shove=shove, stream=stream)
    rows = t079_channel_rows(u_cols, ds_cols)
    return {
        "read": t079_read(rows),
        "floor": t079_floor(rows if floor else [], seed, n_shuffle=n_shuffle),
    }


def t079_reds(n_shuffle=T079_REDS_N_SHUFFLE):
    """2.4's six reds as ONE self-check line (each is ALSO a test in
    `tests/test_agency_monitor.py`; this is the in-command control 2.4 requires, so a gate
    run that never executes pytest still runs every one). Each value is True iff the red
    holds on its constructed input AND its mutated scorer REDDENS it IN FACT -- every
    `not (...)` clause below is a broken twin having been RUN."""
    out = {}

    # R1 -- the shuffle floor is a FLOOR. The mutation draws it UN-permuted, which also
    #       serves as the batched recurrence's positive control: on the identity
    #       permutation every draw must reproduce the shipped machine's own c_j.
    u_cols, ds_cols = t079_coupled_decoupled(T079_R1_PASSAGES, seed=11)
    rows = t079_channel_rows(u_cols, ds_cols)
    read = t079_read(rows)
    floor = t079_floor(rows, 11, n_shuffle=n_shuffle)
    m1 = _t079_mutated_scorer("floor_unpermuted")(rows, 11, n_shuffle=n_shuffle)
    up, dn = T079_JOINTS[0], T079_JOINTS[1]
    # WHAT THE MUTATION REDDENS IN FACT, and it is not what 2.4's sentence predicts.
    # 2.4 says the un-permuted floor makes "the decoupled channel read 'above'". Run, it
    # does not: with the identity permutation EVERY draw equals the reading exactly, so
    # q95(draws) == c and the decoupled channel reads AT its floor, which still satisfies
    # "at or below". What collapses is the MARGIN -- the COUPLED channel's strict
    # `c > floor` becomes `c == floor` and fails. That is T078's own R1 shape
    # ("the deterministic control's margin over it vanishes") and it is the clause asserted
    # here; 2.4's predicted target is reported as a deviation, not silently swapped.
    # The same exactness is the batched recurrence's POSITIVE CONTROL: the un-permuted floor
    # must reproduce the shipped machine's c_j, which is asserted rather than assumed.
    out["R1"] = bool(
        floor["readable"]
        and read["joints"][up]["c"] > floor["joints"][up]["floor"]
        and read["joints"][dn]["c"] <= floor["joints"][dn]["floor"]
        and abs(m1["joints"][up]["floor"] - read["joints"][up]["c"]) < 1e-9
        and not (read["joints"][up]["c"] > m1["joints"][up]["floor"])
    )

    # R2 -- the window and the dead-band. The mutation sets delay = recover_n = 0.
    h = t079_hysteresis_probe(0)
    hm = t079_hysteresis_probe(0, monitor=_t079_mutated_scorer("no_hysteresis")())
    out["R2"] = bool(
        h["first_low_while_ok_status"] == "ok"
        and h["lows_at_flip"] is not None
        and h["lows_at_flip"] > h["delay"]
        and h["first_high_while_lost_status"] == "lost"
        and h["highs_at_flip"] is not None
        and h["highs_at_flip"] >= h["recover_n"]
        and not (
            hm["first_low_while_ok_status"] == "ok" and hm["first_high_while_lost_status"] == "lost"
        )
    )

    # R3 -- an unexercised channel is UNSCORED, never lost. The mutation sets var_floor = 0.
    du, dd = t079_dead_channel(T079_R1_PASSAGES, seed=13)
    dead_read = t079_read(t079_channel_rows(du, dd))
    m3 = _t079_mutated_scorer("no_floor")(du, dd)
    dead = T079_JOINTS[0]
    out["R3"] = bool(
        dead_read["joints"][dead]["n"] >= T079_WARMUP
        and not dead_read["joints"][dead]["scored"]
        and dead_read["joints"][dead]["status"] != "lost"
        and not dead_read["readable"]
        and all(dead_read["joints"][j]["scored"] for j in T079_JOINTS[1:])
        and not (not m3["readable"] and not m3["joints"][dead]["scored"])
    )

    # R4 -- the parser refuses a joint outside the vocabulary. The mutation trusts the regex.
    foreign = "JOINT_GRIPPER_ROTATE(+6deg)"
    plus = intended_deltas(arm_config.command_for("elbow_flex", 6.0))
    ref = intended_deltas(foreign)
    m4 = _t079_mutated_scorer("parse_any_name")(foreign)
    out["R4"] = bool(
        plus["elbow_flex"] == 6.0
        and all(v == 0.0 for k, v in plus.items() if k != "elbow_flex")
        and all(v == 0.0 for v in intended_deltas("ENGAGE").values())
        and all(
            v == 0.0 for v in intended_deltas(arm_config.moveto_for("elbow_flex", 30.0)).values()
        )
        and set(ref) == set(T079_JOINTS)
        and all(v == 0.0 for v in ref.values())
        and not (set(m4) == set(T079_JOINTS) and all(v == 0.0 for v in m4.values()))
    )

    # R5 -- the per-joint fold wraps, and the scalar IS its mean. The mutation drops the wrap.
    before, after = [359.9] * 6, [0.1] * 6
    seam = proprioception_deltas(before, after)
    m5 = _t079_mutated_scorer("no_wrap")(before, after)
    rng5 = random.Random(5)
    agree = True
    for _ in range(200):
        b = [rng5.uniform(-720.0, 720.0) for _ in range(6)]
        a = [rng5.uniform(-720.0, 720.0) for _ in range(6)]
        per = proprioception_deltas(b, a)
        if per is None or sum(per) / len(per) != proprioception_delta(b, a):
            agree = False
            break
    out["R5"] = bool(
        seam is not None
        and all(abs(d * 180.0 - 0.2) < 1e-9 for d in seam)
        and agree
        and proprioception_deltas(None, None) is None
        and not all(abs(d * 180.0 - 0.2) < 1e-9 for d in m5)
    )

    # R6 -- the dissociation on a constructed body. The mutation reads 1 - c on the shoved arm.
    intact = [t079_arm(s, None, n_shuffle) for s in T079_SEEDS]
    shoved = [t079_arm(s, T079_SHOVE_JOINT, n_shuffle) for s in T079_SEEDS]
    twin = [t079_arm(s, None, n_shuffle, stream=1) for s in T079_SEEDS]
    got = t079_contrasts(intact, shoved)
    null = t079_contrasts(intact, twin)
    m6 = t079_contrasts(intact, shoved, score=_t079_mutated_scorer("blind_scorer"))
    # WHAT THE MUTATION REDDENS IN FACT, and again it is not 2.4's predicted target.
    # 2.4 says 1 - c on the shoved arm "inverts (a1)". Run, it cannot on a body whose
    # intact coupling is near 1: (a1)'s mutated difference is c_intact - (1 - c_shoved) =
    # c_intact + c_shoved - 1, which is negative ONLY where the two sum below 1, and
    # c_intact alone reads ~0.999 here (and +0.945..+1.000 on the real body, 1.3). The
    # mutation DOES redden R6's reading -- through (a2), whose mutated difference is
    # (1 - c_k) - c_k + DEADBAND and goes decisively negative on every preserved joint. The
    # clause asserted is therefore R6's READING as a whole, which is the object 2.4 bars;
    # the conjunct that actually fails is reported.
    out["R6"] = bool(
        got["a1"].met
        and all(c.met for c in got["a2"].values())
        and got["a3"]["met"]
        and not null["a1"].met
        and not null["a3"]["met"]
        and not (m6["a1"].met and all(c.met for c in m6["a2"].values()) and m6["a3"]["met"])
    )
    return out


def _t079_reds_line(reds, out=print):
    """Print the six booleans beside what each one says (2.9 step 2: main() prints them)."""
    for name in sorted(reds):
        out(f"  {name}: {str(reds[name]):5s}  {T079_RED_LINES[name]}")


def _t079_pearson(a, b):
    """Pearson of two equal-length series, or None when either has NO variance.
    `_mathutils.pearson` carries no zero-denominator guard -- its own module docstring says
    so -- and this is a REPORTED quantity over recordings where a joint can sit still, so
    the guard lives at this one call site rather than becoming a second correlation."""
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = list(a[:n]), list(b[:n])
    ma, mb = sum(a) / n, sum(b) / n
    if all(x == ma for x in a) or all(y == mb for y in b):
        return None
    return pearson(a, b)


def t079_record_brain(seed, n_ticks=T079_N_TICKS, shove=None, amp=0.0):
    """Tick a SCRATCH brain `n_ticks` times and return ONE ROW PER ENVELOPE PASSAGE (2.1).

    The brain is `make_brain(seed, persist=False)` under `throwaway_home()` -- a fresh
    NORD_HOME for the duration, so nothing this recorder does can reach the live lake -- and
    `_real_ptr` is set to `T079_STREAM_OFFSET * seed` right after construction, so each seed
    walks a DIFFERENT window of the percept pack (2.2; the dry run checks two seeds' first
    50 frame labels actually differ and REFUSES if they do not).

    TWO INSTANCE WRAPS, and the second is a DEVIATION from 2.1 that is stated here rather
    than hidden. 2.1 says "efferent.observe wrapped on the INSTANCE to capture the triple
    and the action per passage" -- that wrap is here and does exactly that. But 2.1 also
    asks each row to carry `ctrl` READ AFTER THE FEED, and the feed happens in
    `apply_observed` AFTER `efferent.observe` returns, so a reading taken inside that wrap
    would be one passage stale. `apply_observed` is therefore wrapped too, and the row is
    built after it returns: the triple comes from the inner wrap, the reading from the
    monitor the feed just updated. Both wraps are observation-only.

    The shove, when `shove` names a joint: `So101Arm.inject_disturbance(shove, U(-amp,
    amp))` ONCE after every passage's feed, drawn from `random.Random(T079_SHOVE_RNG_BASE +
    seed)` so the loop's own RNG is untouched, and NEVER while the body is latched (the port
    refuses a latched arm anyway; this does not rely on that). The draw is recorded per row.

    Rows only -- no reading, no floor, no verdict -- and the brain is dropped on return so
    eight seeds never stay alive at once."""
    from snn_data_lake.cortex.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    rows = []
    with throwaway_home():
        b = make_brain(seed=seed, persist=False)
        b._real_ptr = T079_STREAM_OFFSET * int(seed)
        arm = b.brain.arm  # the AutonomousArm
        body = arm.arm  # the So101Arm
        shove_rng = random.Random(T079_SHOVE_RNG_BASE + int(seed))
        cap = {}
        real_observe = arm.efferent.observe
        real_apply = arm.apply_observed

        def wrapped_observe(action, theta_pre, commanded_after, theta_post):
            cap["action"] = action
            cap["theta_pre"] = None if theta_pre is None else list(theta_pre)
            cap["commanded"] = None if commanded_after is None else list(commanded_after)
            cap["theta_post"] = None if theta_post is None else list(theta_post)
            return real_observe(action, theta_pre, commanded_after, theta_post)

        def wrapped_apply(command, *, deliberate=False):
            cap.clear()
            out = real_apply(command, deliberate=deliberate)
            copy = arm.efferent.last_copy or {}
            u_j, ds_j = copy.get("u_j"), copy.get("ds_j")
            ok = u_j is not None and ds_j is not None
            mon = getattr(arm, "_agency", None)
            snaps = [None if mon is None else mon.snapshot(j) for j in T079_JOINTS]
            st = body.state() or {}
            row = {
                "t": None,
                "kind": None,
                "awake": None,
                "awake_ticks": None,
                "passages_this_tick": None,
                "action": None if command is None else str(command),
                "copy_ok": bool(ok),
                "u": [float(u_j[s.qpos_index]) for s in _T079_SPECS] if ok else [0.0] * 5,
                "ds": [float(ds_j[s.qpos_index]) for s in _T079_SPECS] if ok else [0.0] * 5,
                "ctrl": [0.0 if s is None else float(s["ctrl"]) for s in snaps],
                "status": [None if s is None else s["status"] for s in snaps],
                "n": [0 if s is None else int(s["n"]) for s in snaps],
                "latched": bool(st.get("stopped")),
                "latch_source": st.get("latch_source"),
                "tap_errors": int(arm.efferent.tap_errors),
                "envelope_tick": int(arm.efferent.envelope_tick),
                "theta_pre": cap.get("theta_pre"),
                "commanded": cap.get("commanded"),
                "theta_post": cap.get("theta_post"),
                "shoved_deg": 0.0,
            }
            rows.append(row)
            if shove and not body.stopped:
                deg = shove_rng.uniform(-amp, amp)
                body.inject_disturbance(shove, deg)
                row["shoved_deg"] = deg
            return out

        arm.efferent.observe = wrapped_observe
        arm.apply_observed = wrapped_apply

        labels = getattr(b, "_real_labels", None)
        prev_ptr = int(getattr(b, "_real_ptr", 0) or 0)
        n_awake = 0
        for t in range(n_ticks):
            start = len(rows)
            kind = b.tick()
            ptr = int(getattr(b, "_real_ptr", 0) or 0)
            label = None
            if labels is not None and len(labels) and ptr != prev_ptr:
                label = int(labels[(ptr - 1) % len(labels)])
            prev_ptr = ptr
            awake = kind == "awake"
            n_awake += 1 if awake else 0
            for r in rows[start:]:
                r["t"] = t
                r["kind"] = kind
                r["awake"] = awake
                r["awake_ticks"] = n_awake
                r["passages_this_tick"] = len(rows) - start
                r["label"] = label
    return rows


def t079_stream_labels(seed, n_labels=50, max_ticks=260):
    """The first `n_labels` REAL frame labels seed `s` walks, with 2.2's stream offset
    applied. The offset check is an APPARATUS precondition, so it reads the labels the brain
    actually consumed rather than re-deriving them from the pack by arithmetic."""
    from snn_data_lake.cortex.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    out = []
    with throwaway_home():
        b = make_brain(seed=seed, persist=False)
        b._real_ptr = T079_STREAM_OFFSET * int(seed)
        labels = getattr(b, "_real_labels", None)
        prev = int(getattr(b, "_real_ptr", 0) or 0)
        for _ in range(max_ticks):
            if len(out) >= n_labels:
                break
            b.tick()
            ptr = int(getattr(b, "_real_ptr", 0) or 0)
            if labels is not None and len(labels) and ptr != prev:
                out.append(int(labels[(ptr - 1) % len(labels)]))
            prev = ptr
    return out


def t079_determinism(seed=0, n_ticks=T079_DETERMINISM_TICKS):
    """(p): two INTACT recordings of the same seed at the same width, byte-identical on
    EVERY recorded field. Not a sample of fields and not a tolerance -- the rows are
    compared whole, so a nondeterminism anywhere in the recorder or the loop is a REFUSAL
    before any number is read."""
    a = t079_record_brain(seed, n_ticks=n_ticks)
    b = t079_record_brain(seed, n_ticks=n_ticks)
    return {"identical": a == b, "passages": len(a), "twin_passages": len(b), "seed": seed}


def t079_verdict(got, refusals):
    """2.3's verdict rule, decided in 2.3's order and nowhere else.

    REFUSED iff (p), (r) or (g0) fails -- a finding about the APPARATUS or the INSTRUMENT.
    PASS iff (g0) and (a1) and (a2) and (a3). NOT MET iff (g0) is met and any of (a1)-(a3)
    is not -- a finding about this BODY under this disturbance. The two are never merged."""
    if refusals:
        return "REFUSED"
    if got["g0"] is None or not all(c.met for c in got["g0"].values()):
        return "REFUSED"
    ok = got["a1"].met and all(c.met for c in got["a2"].values()) and got["a3"]["met"]
    return "PASS" if ok else "NOT MET"


def _t079_head():
    """The tree's HEAD, for the pin. Unavailable (no git, a tarball) is recorded as such."""
    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else f"unavailable (rc {r.returncode})"
    except Exception as e:  # noqa: BLE001 -- a pin field, never a control-flow decision
        return f"unavailable ({type(e).__name__})"


def _t079_report_block(per_seed, seeds):
    """2.3's REPORTED, never barred: the floors and their medians, the action distribution
    per arm (the policy's own reaction to the disturbance, which "preserved elsewhere" is
    NEVER read on), the same-tick vs lag-1 Pearson per joint, and the incumbent feed's
    n_scored -- which is not recomputed here at all. 2.3 and 1.3 fix it: the tick-level
    action/qmax/reward feed is RETIRED at build time and read 0 on 1.3's probe; recomputing
    it offline would be building the retired mechanism a second time to report a number its
    own measurement already refused."""
    return {
        "floors": {
            j: {
                "intact_q95": [per_seed[s]["intact"]["floor"]["joints"][j]["floor"] for s in seeds],
                "intact_median": [
                    per_seed[s]["intact"]["floor"]["joints"][j]["floor_median"] for s in seeds
                ],
                "shoved_q95": [per_seed[s]["shoved"]["floor"]["joints"][j]["floor"] for s in seeds],
                "shoved_median": [
                    per_seed[s]["shoved"]["floor"]["joints"][j]["floor_median"] for s in seeds
                ],
            }
            for j in T079_JOINTS
        },
        "actions": {
            arm: [per_seed[s][arm]["actions"] for s in seeds] for arm in ("intact", "shoved")
        },
        "coupling": {
            arm: {
                j: {
                    "same_tick": [per_seed[s][arm]["same_tick"][j] for s in seeds],
                    "lag1": [per_seed[s][arm]["lag1"][j] for s in seeds],
                }
                for j in T079_JOINTS
            }
            for arm in ("intact", "shoved")
        },
        "incumbent_n_scored": "retired, 0 on 1.3's probe",
    }


def _t079_arm_extras(rows):
    """The per-arm REPORTED quantities that are read off the rows themselves."""
    counts = {}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    same, lag1 = {}, {}
    for k, j in enumerate(T079_JOINTS):
        u = [float(r["u"][k]) for r in rows]
        ds = [float(r["ds"][k]) for r in rows]
        same[j] = _t079_pearson(u, ds)
        lag1[j] = _t079_pearson(u[:-1], ds[1:])
    return {"actions": dict(sorted(counts.items())), "same_tick": same, "lag1": lag1}


def _t079_refusals(read, seed, arm):
    """(r)'s named refusals for one recording: WHICH joint, WHICH count, per seed and arm."""
    out = []
    for j in T079_JOINTS:
        v = read["joints"][j]
        if not v["scored"]:
            out.append(
                f"seed {seed} {arm}: {j} UNSCORED (n {v['n']}/{T079_WARMUP}, "
                f"var_c {v['var_c']:.3e}/{T079_VAR_FLOOR:.1e})"
            )
    if read.get("observed_ticks", read["passages"]) != read["awake"]:
        missing = read["awake"] - read.get("observed_ticks", 0)
        out.append(
            f"seed {seed} {arm}: {missing} awake ticks carried NO passage "
            f"({read['passages']} passages, {read['awake']} awake)"
        )
    if read["latched"]:
        out.append(f"seed {seed} {arm}: latched on {read['latched']} passages")
    if read["tap_errors"]:
        out.append(f"seed {seed} {arm}: tap_errors {read['tap_errors']}")
    if read["copies_unreadable"]:
        out.append(f"seed {seed} {arm}: {read['copies_unreadable']} unreadable per-joint copies")
    return out


# distinct-from: cortex/attention_schema.py::run_t078_campaign and
#   cortex/metacognitive_monitor.py's campaign -- the PATTERN this follows (2.1 names them as
#   the thing to reuse), over a different instrument, different arms and a different pin.
#   Read; no state, no pin and no statistic is shared beyond `substrate/paired_contrast`,
#   which all three IMPORT.
def run_t079_campaign(seeds=T079_SEEDS, n_ticks=T079_N_TICKS, pin=T079_PIN):
    """The campaign (2.2-2.3): record, read, floor, contrast, pin ONCE. Returns an exit code.

    ORDER IS 2.3's, and it is the order the verdict is DECIDED in:
      (p)  DETERMINISM  the twin recordings byte-identical, else REFUSED -- nothing is read
      (r)  READABLE     every joint scored on both arms of every seed, passages == awake
                        ticks, tap_errors 0, never latched, else REFUSED with the joint named
      (g0) THE ESTIMATOR SEES  the intact coupling clears its cause-blind twin's rate on
                        EVERY joint, else REFUSED ("blind on this body", never a STOP)
      (a1) (a2) (a3)    the dissociation, the preservation and the status machine
    A REFUSAL is a finding about the apparatus or the instrument; a NOT MET is a finding
    about this body under this disturbance. They are never merged."""
    if pin.exists():
        print(f"REFUSED: {pin} exists -- the pin is written once", flush=True)
        return 2
    t0 = time.time()

    labels0, labels1 = t079_stream_labels(0), t079_stream_labels(1)
    if not labels0 or labels0 == labels1:
        print(
            f"REFUSED at the apparatus: the per-seed stream offset does not vary the frames "
            f"(seed 0 {labels0[:10]} vs seed 1 {labels1[:10]})",
            flush=True,
        )
        return 2
    det = t079_determinism(seeds[0] if seeds else 0)
    print(
        f"(p) determinism: identical={det['identical']} on {det['passages']} passages", flush=True
    )
    if not det["identical"]:
        print("REFUSED: two identical-seed recordings differ -- no reading is taken", flush=True)
        return 2

    per_seed, refusals = {}, []
    for s in seeds:
        entry = {}
        for arm, sh, amp in (("intact", None, 0.0), ("shoved", T079_SHOVE_JOINT, T079_SHOVE_DEG)):
            rows = t079_record_brain(s, n_ticks=n_ticks, shove=sh, amp=amp)
            read = t079_read(rows)
            entry[arm] = {
                "read": read,
                "floor": t079_floor(rows, s),
                "rows": rows,
                **_t079_arm_extras(rows),
            }
            refusals.extend(_t079_refusals(read, s, arm))
        per_seed[s] = entry
        print(
            f"seed {s}: intact c "
            + " ".join(f"{j}={entry['intact']['read']['joints'][j]['c']:.3f}" for j in T079_JOINTS)
            + f" | shoved {T079_SHOVE_JOINT}="
            f"{entry['shoved']['read']['joints'][T079_SHOVE_JOINT]['c']:.3f} "
            f"({entry['shoved']['read']['joints'][T079_SHOVE_JOINT]['status']})",
            flush=True,
        )

    intact = [
        {"read": per_seed[s]["intact"]["read"], "floor": per_seed[s]["intact"]["floor"]}
        for s in seeds
    ]
    shoved = [
        {"read": per_seed[s]["shoved"]["read"], "floor": per_seed[s]["shoved"]["floor"]}
        for s in seeds
    ]
    got = t079_contrasts(intact, shoved)
    verdict = t079_verdict(got, refusals)

    print("-" * 78, flush=True)
    for why in refusals:
        print(f"  (r) REFUSAL: {why}", flush=True)
    if got["g0"] is not None:
        for c in got["g0"].values():
            print(f"  {c.name}: t={c.t:.4f} FPR={c.fpr} met={c.met}", flush=True)
    print(
        f"  {got['a1'].name}: t={got['a1'].t:.4f} FPR={got['a1'].fpr} met={got['a1'].met}",
        flush=True,
    )
    for c in got["a2"].values():
        print(f"  {c.name}: t={c.t:.4f} FPR={c.fpr} met={c.met}", flush=True)
    print(f"  (a3) status machine: {got['a3']}", flush=True)
    print(f"T079 VERDICT: {verdict}", flush=True)

    out = {
        "row": "T079",
        "head": _t079_head(),
        "seeds": list(seeds),
        "n_ticks": n_ticks,
        "bar": T079_BAR,
        "n_shuffle": T079_N_SHUFFLE,
        "shove": {
            "joint": T079_SHOVE_JOINT,
            "amp_deg": T079_SHOVE_DEG,
            "ladder": list(T079_LADDER),
        },
        "constants": {
            "warmup": T079_WARMUP,
            "var_floor": T079_VAR_FLOOR,
            "deadband": T079_DEADBAND,
            "stream_offset": T079_STREAM_OFFSET,
            "alpha": _T079_DEFAULTS.alpha,
            "dead_thresh": _T079_DEFAULTS.dead_thresh,
            "high_thresh": _T079_DEFAULTS.high_thresh,
            "delay": _T079_DEFAULTS.delay,
            "recover_n": _T079_DEFAULTS.recover_n,
        },
        "reds": t079_reds(),
        "determinism": det,
        "stream_labels": {"seed0": labels0, "seed1": labels1},
        # EVERY PER-PASSAGE INPUT OF EVERY BARRED CONJUNCT (2.1): the rows themselves, both
        # arms, every seed -- so each of (g0), (a1), (a2), (a3) re-derives offline from this
        # file alone by replaying t079_read / t079_floor / t079_contrasts over them.
        "per_seed": {
            str(s): {
                arm: {
                    "read": per_seed[s][arm]["read"],
                    "floor": per_seed[s][arm]["floor"],
                    "actions": per_seed[s][arm]["actions"],
                    "same_tick": per_seed[s][arm]["same_tick"],
                    "lag1": per_seed[s][arm]["lag1"],
                    "rows": per_seed[s][arm]["rows"],
                }
                for arm in ("intact", "shoved")
            }
            for s in seeds
        },
        "contrasts": {
            "g0": None if got["g0"] is None else {j: c._asdict() for j, c in got["g0"].items()},
            "a1": got["a1"]._asdict(),
            "a2": {j: c._asdict() for j, c in got["a2"].items()},
            "a3": got["a3"],
        },
        "refusals": refusals,
        "reported": _t079_report_block(per_seed, seeds),
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"pinned {pin} in {time.time() - t0:.0f} s", flush=True)
    return 0 if verdict != "REFUSED" else 2


def run_t079_dry_run(out_path, seed=0, n_ticks=T079_DRY_TICKS):
    """2.9 step 5's APPARATUS dry run: ONE seed, 300 ticks, and NOT ONE barred conjunct.

    One seed cannot carry a paired t, so every (a1)-style number here is printed as a single
    seed's difference with "n/a" where the t would go. What it DOES read, all of it reported
    in 3.0 before the campaign:
      * the shove LADDER A = 3 / 6 / 12 deg (0.5x, 1x, 2x DEFAULT_STEP_DEG), one shoved
        recording each -- the DETECTION FLOOR of (a1) and of the status machine;
      * the STREAM-OFFSET precondition: seeds 0 and 1 must walk different frames, REFUSED if
        not (2.2 makes this an apparatus refusal, not a finding);
      * the floor's MONTE-CARLO sd over 12 independent 2,000-draw floors of the SAME intact
        recording -- 2.8's n_shuffle sensitivity, COMPUTED and not promised;
      * the cost, re-measured on the built apparatus (2.8's 61.5 ms/tick was 1.3's probe).

    Writes its JSON to `out_path` and NOTHING under tools/: the pin is the campaign's."""
    out_path = Path(out_path)
    if "tools" in out_path.parts:
        print(f"REFUSED: the dry run never writes under tools/ ({out_path})", flush=True)
        return 2
    t0 = time.time()
    print("=" * 78, flush=True)
    print(f"T079 DRY RUN -- one seed ({seed}), {n_ticks} ticks, NO barred conjunct", flush=True)
    print("=" * 78, flush=True)

    labels0, labels1 = t079_stream_labels(0), t079_stream_labels(1)
    differ = bool(labels0) and labels0 != labels1
    first_diff = next(
        (i for i, (x, y) in enumerate(zip(labels0, labels1, strict=False)) if x != y), None
    )
    print(
        f"stream offset ({T079_STREAM_OFFSET} x seed): seed0 {labels0[:12]} | "
        f"seed1 {labels1[:12]} | differ={differ} first_diff_at={first_diff} "
        f"(n {len(labels0)}/{len(labels1)})",
        flush=True,
    )
    if not differ:
        print("REFUSED at the apparatus: the per-seed offset does not vary the frames", flush=True)
        return 2

    t_rec = time.time()
    intact_rows = t079_record_brain(seed, n_ticks=n_ticks)
    ms_per_tick = (time.time() - t_rec) * 1000.0 / max(n_ticks, 1)
    intact = t079_read(intact_rows)
    intact_floor = t079_floor(intact_rows, seed)
    print("-" * 78, flush=True)
    print(
        f"cost: {ms_per_tick:.1f} ms/tick measured on the BUILT apparatus "
        f"(2.8 carried 61.5 ms from 1.3's probe) -> campaign estimate "
        f"{ms_per_tick * T079_N_TICKS * 2 * len(T079_SEEDS) / 1000.0 / 60.0:.1f} min for "
        f"{len(T079_SEEDS)} seeds x 2 arms x {T079_N_TICKS} ticks",
        flush=True,
    )
    print(
        f"intact: passages {intact['passages']} awake {intact['awake']} latched "
        f"{intact['latched']} tap_errors {intact['tap_errors']} unreadable_copies "
        f"{intact['copies_unreadable']} readable {intact['readable']}",
        flush=True,
    )
    for j in T079_JOINTS:
        v = intact["joints"][j]
        f = intact_floor["joints"][j]
        print(
            f"  {j:<14} c={v['c']:.4f} status={str(v['status']):<7} n={v['n']:<5} "
            f"scored={str(v['scored']):<5} var_c={v['var_c']:.3e} floor={f['floor']:.4f} "
            f"median={f['floor_median']:.4f} scored_passages={v['scored_passages']}",
            flush=True,
        )

    print("-" * 78, flush=True)
    print(
        "shove ladder (one seed -> NO paired t; the t column reads n/a by construction)",
        flush=True,
    )
    ladder = {}
    for amp in T079_LADDER:
        rows = t079_record_brain(seed, n_ticks=n_ticks, shove=T079_SHOVE_JOINT, amp=amp)
        read = t079_read(rows)
        floor = t079_floor(rows, seed)
        ladder[f"{amp:g}"] = {
            "read": read,
            "floor": floor,
            "diffs": {j: intact["joints"][j]["c"] - read["joints"][j]["c"] for j in T079_JOINTS},
            "shoved_deg": [r["shoved_deg"] for r in rows],
        }
        print(f"  A = {amp:g} deg  (t: n/a -- one seed)", flush=True)
        for j in T079_JOINTS:
            v = read["joints"][j]
            d = ladder[f"{amp:g}"]["diffs"][j]
            star = " <- J*" if j == T079_SHOVE_JOINT else ""
            print(
                f"    {j:<14} c={v['c']:.4f} status={str(v['status']):<7} "
                f"floor={floor['joints'][j]['floor']:.4f} "
                f"intact-shoved={d:+.4f}{star}",
                flush=True,
            )
        print(
            f"    (r) readable={read['readable']} passages={read['passages']} "
            f"awake={read['awake']} latched={read['latched']} "
            f"tap_errors={read['tap_errors']}",
            flush=True,
        )

    print("-" * 78, flush=True)
    print(
        f"floor Monte-Carlo sd: {T079_FLOOR_REPEATS} INDEPENDENT {T079_N_SHUFFLE}-draw floors "
        f"of the SAME intact recording (2.8's n_shuffle sensitivity)",
        flush=True,
    )
    sds = {}
    for j in T079_JOINTS:
        sds[j] = []
    for i in range(T079_FLOOR_REPEATS):
        f = t079_floor(intact_rows, T079_FLOOR_SD_BASE + i)
        for j in T079_JOINTS:
            sds[j].append(f["joints"][j]["floor"])
    floor_sd = {}
    for j in T079_JOINTS:
        vals = sds[j]
        sd = statistics.stdev(vals) if len(set(vals)) > 1 else 0.0
        floor_sd[j] = {"mean": statistics.mean(vals), "sd": sd, "min": min(vals), "max": max(vals)}
        print(
            f"  {j:<14} mean={floor_sd[j]['mean']:.4f} sd={sd:.5f} "
            f"min={min(vals):.4f} max={max(vals):.4f}",
            flush=True,
        )

    report = {
        "row": "T079 dry run",
        "head": _t079_head(),
        "seed": seed,
        "n_ticks": n_ticks,
        "ms_per_tick": ms_per_tick,
        "stream": {"seed0": labels0, "seed1": labels1, "differ": differ, "first_diff": first_diff},
        "intact": {"read": intact, "floor": intact_floor, **_t079_arm_extras(intact_rows)},
        "ladder": {
            k: {"read": v["read"], "floor": v["floor"], "diffs": v["diffs"]}
            for k, v in ladder.items()
        },
        "floor_sd": floor_sd,
        "reds": t079_reds(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("-" * 78, flush=True)
    print(
        f"dry run written to {out_path} in {time.time() - t0:.0f} s; nothing under tools/",
        flush=True,
    )
    return 0


# =============================================================================
# Un-riggable self-check: drive real command/feedback dynamics and assert the
# monitor's behavior emerges from them (never from a hardcoded label).
# =============================================================================
def _experiment(seed):
    """Two channels through healthy -> decouple -> recover; measure the whole trajectory."""
    rng = random.Random(seed)
    mon = AgencyMonitor(seed=seed)
    gain, noise = 0.9, 0.05
    l_heal, l_decouple, l_recover = 400, 400, 500

    def cmd():
        return rng.uniform(-1.0, 1.0)

    def heal_fb(c):
        return gain * c + noise * rng.uniform(-1.0, 1.0)

    # ---- phase 1: BOTH channels healthy (feedback tracks command) ----
    for _ in range(l_heal):
        ca, cb = cmd(), cmd()
        mon.observe("reach", ca, heal_fb(ca))
        mon.observe("grasp", cb, heal_fb(cb))
    heal = mon.agency()
    ctrl_reach_heal = mon.controllability("reach")
    ctrl_grasp_heal = mon.controllability("grasp")
    status_reach_heal = mon.status("reach")
    status_grasp_heal = mon.status("grasp")

    # ---- phase 2: 'grasp' DECOUPLES -- command still varies, feedback -> pure noise ----
    reach_ok_through_decouple = True
    for _ in range(l_decouple):
        ca, cb = cmd(), cmd()
        mon.observe("reach", ca, heal_fb(ca))
        mon.observe("grasp", cb, rng.uniform(-1.0, 1.0))  # feedback no longer caused by command
        if mon.status("reach") != "ok":
            reach_ok_through_decouple = False
    lost = mon.agency()
    ctrl_grasp_lost = mon.controllability("grasp")
    status_grasp_lost = mon.status("grasp")
    status_reach_lost = mon.status("reach")

    # ---- phase 3: 'grasp' RECOVERS -- feedback tracks command again ----
    first_high_while_lost_status = None  # status at the FIRST high reading while still 'lost'
    saw_high_while_lost = False
    high_at_flip = None  # value of the consecutive-high counter when it flips to 'ok'
    recovered = False
    for _ in range(l_recover):
        ca, cb = cmd(), cmd()
        mon.observe("reach", ca, heal_fb(ca))
        prev = mon.snapshot("grasp")["status"]
        mon.observe("grasp", cb, heal_fb(cb))
        snap = mon.snapshot("grasp")
        if prev == "lost" and snap["ctrl"] >= mon.high_thresh:
            saw_high_while_lost = True
            if first_high_while_lost_status is None:
                first_high_while_lost_status = prev  # 'lost' -- a single good tick must NOT flip it
        if prev == "lost" and snap["status"] == "ok" and not recovered:
            recovered = True
            high_at_flip = snap["high"]
    status_grasp_recovered = mon.status("grasp")

    return {
        "heal_signal": heal["agency_signal"],
        "lost_signal": lost["agency_signal"],
        "ctrl_reach_heal": round(ctrl_reach_heal, 6),
        "ctrl_grasp_heal": round(ctrl_grasp_heal, 6),
        "ctrl_grasp_lost": round(ctrl_grasp_lost, 6),
        "status_reach_heal": status_reach_heal,
        "status_grasp_heal": status_grasp_heal,
        "status_grasp_lost": status_grasp_lost,
        "status_reach_lost": status_reach_lost,
        "status_grasp_recovered": status_grasp_recovered,
        "reach_ok_through_decouple": reach_ok_through_decouple,
        "heal_lost_set": heal["lost"],
        "lost_lost_set": lost["lost"],
        "lost_reroute_to": lost["reroute_to"],
        "lost_route_grasp_w": lost["route_around"].get("grasp"),
        "saw_high_while_lost": saw_high_while_lost,
        "first_high_while_lost_status": first_high_while_lost_status,
        "high_at_flip": high_at_flip,
        "recovered": recovered,
        "recover_n": mon.recover_n,
    }


def _p(lines, s):
    """Append an ASCII-only line (Windows cp1252 safety)."""
    if any(ord(ch) > 127 for ch in s):
        raise AssertionError("non-ASCII output")
    lines.append(s)


def main():
    # 2.9 step 5 / step 6: the two brain-touching entry points. Both run under
    # `throwaway_home()` INSIDE their recorders, every brain is `persist=False`, and neither
    # is reachable from the gate's bare `python -m` invocation -- which runs the six reds on
    # constructed streams and touches no brain, no lake and no network.
    argv = sys.argv[1:]
    if "--t079-campaign" in argv:  # 2.9 step 6: the campaign, pinned ONCE
        sys.exit(run_t079_campaign())
    if "--t079-dry-run" in argv:  # 2.9 step 5: the apparatus dry run, NO barred conjunct
        i = argv.index("--t079-dry-run")
        if i + 1 >= len(argv):
            print("usage: --t079-dry-run <out.json>  (never a path under tools/)", flush=True)
            sys.exit(2)
        sys.exit(run_t079_dry_run(argv[i + 1]))

    r = _experiment(0)
    lines = []
    _p(lines, "=" * 78)
    _p(lines, "AGENCY MONITOR -- per-channel controllability / learned-helplessness substrate")
    _p(lines, "=" * 78)
    _p(lines, "controllability(c) = |Pearson(command, feedback)| over an EMA window (alpha=0.02);")
    _p(
        lines,
        "sustained collapse -> 'lost' (delay window); revive only after recover_n consec highs",
    )
    _p(lines, "-" * 78)
    _p(lines, "phases: [1] both healthy  [2] 'grasp' decouples (fb->noise)  [3] 'grasp' recovers")
    _p(lines, "-" * 78)
    _p(lines, "                         reach            grasp")
    _p(
        lines,
        f"  [1] healthy   ctrl :   {r['ctrl_reach_heal']:.3f}            {r['ctrl_grasp_heal']:.3f}",
    )
    _p(
        lines,
        f"      status         :   {r['status_reach_heal']:<6}           {r['status_grasp_heal']}",
    )
    _p(lines, f"  [2] decoupled ctrl :   (stays ok)       {r['ctrl_grasp_lost']:.3f}")
    _p(
        lines,
        f"      status         :   {r['status_reach_lost']:<6}           {r['status_grasp_lost']}",
    )
    _p(lines, f"  [3] recovered stat :                    {r['status_grasp_recovered']}")
    _p(lines, "-" * 78)
    _p(
        lines,
        f"(a) controllable    : reach ctrl {r['ctrl_reach_heal']:.3f} high, status '{r['status_reach_heal']}', "
        f"agency_signal {r['heal_signal']:.3f} high, lost set {r['heal_lost_set']}",
    )
    _p(
        lines,
        f"(b) helplessness    : grasp ctrl collapsed to {r['ctrl_grasp_lost']:.3f} -> status "
        f"'{r['status_grasp_lost']}'; agency dropped {r['heal_signal']:.3f}->{r['lost_signal']:.3f}; "
        f"reroute_to '{r['lost_reroute_to']}'",
    )
    _p(
        lines,
        f"(c) hysteresis      : a high reading while lost did NOT flip it "
        f"(first_high status='{r['first_high_while_lost_status']}'); needed {r['high_at_flip']} "
        f"consec highs (>= recover_n {r['recover_n']}) to revive",
    )
    _p(lines, f"(d) determinism     : two seed=0 runs identical={_experiment(0) == r}")
    _p(lines, "=" * 78)
    for line in lines:
        print(line)

    # ---- (a) A CONTROLLABLE channel reads high, stays 'ok', lifts agency ----
    assert r["ctrl_reach_heal"] >= 0.9, (
        f"a channel whose feedback tracks command must read high controllability ({r['ctrl_reach_heal']:.3f})"
    )
    assert r["ctrl_grasp_heal"] >= 0.9, "grasp starts controllable too"
    assert r["status_reach_heal"] == "ok" and r["status_grasp_heal"] == "ok", (
        "healthy channels must be 'ok'"
    )
    assert r["heal_signal"] >= 0.9, (
        f"all-controllable agency_signal must be high ({r['heal_signal']:.3f})"
    )
    assert r["heal_lost_set"] == [], "no channel lost while both are controllable"

    # ---- (b) a DECOUPLED channel -> sustained 'lost' + agency drops ----
    assert r["ctrl_grasp_lost"] < 0.35, (
        f"decoupled feedback must collapse controllability ({r['ctrl_grasp_lost']:.3f})"
    )
    assert r["status_grasp_lost"] == "lost", (
        "sustained decoupling past the delay window must flip status to 'lost'"
    )
    assert r["status_reach_lost"] == "ok" and r["reach_ok_through_decouple"], (
        "the still-controllable channel must stay 'ok' throughout -- loss is PER CHANNEL"
    )
    assert r["lost_signal"] < r["heal_signal"] - 0.3, (
        f"agency_signal must drop when a channel is lost ({r['heal_signal']:.3f}->{r['lost_signal']:.3f})"
    )
    assert r["lost_lost_set"] == ["grasp"] and r["lost_route_grasp_w"] == 0.0, (
        "the route-around hint must name grasp as lost and down-weight it to 0.0"
    )
    assert r["lost_reroute_to"] == "reach", "reroute_to must point at the controllable channel"

    # ---- (c) HYSTERESIS: one good tick does NOT revive; needs recover_n consecutive ----
    assert r["saw_high_while_lost"], (
        "recovery must produce high readings while the channel is still 'lost'"
    )
    assert r["first_high_while_lost_status"] == "lost", (
        "a single good (high) tick must NOT flip 'lost' -> 'ok'"
    )
    assert r["recovered"] and r["high_at_flip"] is not None, "the channel must eventually recover"
    assert r["high_at_flip"] >= r["recover_n"], (
        f"revival required {r['high_at_flip']} consecutive highs (>= recover_n {r['recover_n']}) -- hysteresis"
    )
    assert r["status_grasp_recovered"] == "ok", "grasp must be 'ok' again after sustained recovery"

    # ---- (d) DETERMINISM ----
    assert _experiment(0) == r, "two seed=0 runs must be byte-for-byte identical"

    print(
        f"self-check OK: controllable channel reads {r['ctrl_reach_heal']:.2f} (status ok); a decoupled "
        f"channel collapses to {r['ctrl_grasp_lost']:.2f} -> 'lost' after the delay window,"
    )
    print(
        f"  dropping agency_signal {r['heal_signal']:.2f}->{r['lost_signal']:.2f} and rerouting to "
        f"'{r['lost_reroute_to']}'; revival needed {r['high_at_flip']} consecutive highs "
        f"(recover_n={r['recover_n']}) -- one good tick never flips it; deterministic"
    )

    # ---- T079's six reds (2.4), each holding on its constructed input AND reddened IN FACT
    #      by its own mutated scorer. Every `not (...)` clause inside `t079_reds()` is a
    #      broken twin having been RUN (P8 clause 1: a criterion with no reachable red is
    #      DELETED, not weakened). This line is the in-command control, so a gate run that
    #      never executes pytest still runs all six.
    t0 = time.time()
    reds = t079_reds()
    print("-" * 78)
    print("T079 -- the agency instrument (specs/022 C4): 2.4's six reds")
    print(
        f"  bar {T079_BAR} | joints {list(T079_JOINTS)} | J* {T079_SHOVE_JOINT} | "
        f"A {T079_SHOVE_DEG:g} deg | ladder {[f'{a:g}' for a in T079_LADDER]}"
    )
    print(
        f"  warmup {T079_WARMUP} obs | var_floor {T079_VAR_FLOOR:g} | deadband {T079_DEADBAND:g} | "
        f"n_shuffle {T079_N_SHUFFLE} (reds {T079_REDS_N_SHUFFLE}) | ticks {T079_N_TICKS} | "
        f"stream offset {T079_STREAM_OFFSET} x seed"
    )
    _t079_reds_line(reds)
    assert all(reds.values()), f"T079 reds must fire on their constructed inputs: {reds}"
    print(
        f"  T079 OK: R1-R6 fire on their constructed streams and each is reddened IN FACT by "
        f"its mutated scorer ({sum(reds.values())}/{len(reds)}); no brain, no lake, no "
        f"network [{time.time() - t0:.1f} s]"
    )


if __name__ == "__main__":
    main()
