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

import math
import random
import zlib
from pathlib import Path

from snn_data_lake.brainstem import arm_config
from snn_data_lake.substrate._mathutils import clamp01
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
# 2.9 step 2: THE SKELETON. Every reading below is NULL BY CONSTRUCTION -- `readable`
# False, every boolean False, no brain ticked, no floor drawn -- so the six reds of 2.4
# land RED against it at 2.9 step 3 and turn green at 2.9 step 4. Nothing here reads a
# body yet; the feed switch, the recorder and the campaign land at 2.9 step 5.
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

#: One ASCII line per red, printed beside its boolean so a gate log says WHAT fired rather
#: than only that something did (2.4's wording, compressed).
T079_RED_LINES = {
    "R1": "the shuffle floor is a FLOOR (drawing it un-permuted reddens it)",
    "R2": "the machine's hysteresis holds both ways (delay = recover_n = 0 reddens it)",
    "R3": "an unexercised channel is UNSCORED, never lost (var_floor := 0 reddens it)",
    "R4": "the token parser refuses a foreign joint (accepting any name reddens it)",
    "R5": "the per-joint fold wraps and its mean IS the scalar (dropping the wrap reddens it)",
    "R6": "the dissociation reads on a constructed body (1 - c on the shoved arm reddens it)",
}


def t079_record_brain(seed, n_ticks=T079_N_TICKS, shove=None, amp=0.0):
    """SKELETON (2.9 step 2): no rows, no brain, no lake. The recorder lands at 2.9 step 5.

    Its built form ticks a SCRATCH brain (`throwaway_home()`, `make_brain(seed,
    persist=False)`, `_real_ptr = T079_STREAM_OFFSET * seed`) and returns one row per
    ENVELOPE PASSAGE -- never a verdict, never a contrast."""
    return []


def t079_read(rows):
    """SKELETON (2.9 step 2): `readable` False, every joint unscored, every controllability
    0.0. The real per-joint read lands at 2.9 step 4.

    Its built form reports, per joint: `c` (the MEAN of the instrument's own streaming
    `controllability(j)` over the passages past warmup), `status` at the end of the
    recording, `scored`, and the counts (r) is refused on."""
    return {
        "readable": False,
        "passages": 0,
        "awake": 0,
        "latched": 0,
        "tap_errors": 0,
        "joints": {
            j: {"c": 0.0, "status": None, "scored": False, "n": 0, "var_c": 0.0}
            for j in T079_JOINTS
        },
    }


def t079_floor(rows, seed, n_shuffle=T079_N_SHUFFLE):
    """SKELETON (2.9 step 2): `readable` False, every floor 0.0, zero draws. The real
    cause-blind twin lands at 2.9 step 4.

    Its built form permutes `u_j` across the SCORED passages (the outcomes identical, the
    causal link removed -- the C5 row's own dissociation input), pushes all `n_shuffle` draws
    through the SAME EMA recurrence as one batched torch tensor, and takes the 95th-percentile
    order statistic of the resulting mean-controllability draws."""
    return {
        "readable": False,
        "n_shuffle": int(n_shuffle),
        "seed": int(seed),
        "joints": {j: {"floor": 0.0, "floor_median": 0.0, "draws": 0} for j in T079_JOINTS},
    }


def t079_contrasts(intact, shoved):
    """SKELETON (2.9 step 2): every conjunct NOT MET. The real contrasts land at 2.9 step 4.

    `intact` and `shoved` are per-seed lists of `{"read": t079_read(...), "floor":
    t079_floor(...)}` -- the floor travels WITH its arm because (g0) reads the intact arm
    against its own twin. Its built form returns 2.3's (g0) per joint, (a1) on J*, (a2) per
    unshoved joint and (a3)'s exact counts, each decided by `paired_contrast.contrast`."""
    return {
        "g0": {
            j: contrast(f"(g0) {j}: intact c - its cause-blind floor", [0.0] * 8)
            for j in T079_JOINTS
        },
        "a1": contrast(f"(a1) {T079_SHOVE_JOINT}: intact c - shoved c", [0.0] * 8),
        "a2": {
            j: contrast(f"(a2) {j}: (shoved c - intact c) + deadband", [0.0] * 8)
            for j in T079_JOINTS
            if j != T079_SHOVE_JOINT
        },
        "a3": {"shoved_star_lost": 0, "other_lost": 0, "met": False},
        "n_seeds": 0,
    }


def t079_reds(n_shuffle=T079_N_SHUFFLE):
    """SKELETON (2.9 step 2): 2.4's six reds, all False -- no red has been RUN yet.

    At 2.9 step 3 each is a failing assertion in `tests/test_agency_monitor.py`; at 2.9 step 4
    each is True iff the red holds on its constructed input AND its mutated scorer reddens it
    IN FACT (a red a broken instrument still passes is not a red)."""
    return {f"R{i}": False for i in range(1, 7)}


def _t079_reds_line(reds, out=print):
    """Print the six booleans beside what each one says (2.9 step 2: main() prints them)."""
    for name in sorted(reds):
        out(f"  {name}: {str(reds[name]):5s}  {T079_RED_LINES[name]}")


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

    # ---- T079's six reds (2.4). At 2.9 step 2 they are NULL READINGS and are PRINTED, not
    #      asserted: the skeleton exists so tests/test_agency_monitor.py can commit them RED
    #      at 2.9 step 3. The assert lands at 2.9 step 4, when each is reddened IN FACT by its
    #      mutated scorer. Printing False here is the honest state of the row, not a failure.
    print("-" * 78)
    print("T079 -- the agency instrument (specs/022 C4): 2.4's six reds")
    print(
        f"  bar {T079_BAR} | joints {list(T079_JOINTS)} | J* {T079_SHOVE_JOINT} | "
        f"A {T079_SHOVE_DEG:g} deg | ladder {[f'{a:g}' for a in T079_LADDER]}"
    )
    print(
        f"  warmup {T079_WARMUP} obs | var_floor {T079_VAR_FLOOR:g} | deadband {T079_DEADBAND:g} | "
        f"n_shuffle {T079_N_SHUFFLE} | ticks {T079_N_TICKS} | stream offset {T079_STREAM_OFFSET} x seed"
    )
    _t079_reds_line(t079_reds())
    print(
        "  2.9 step 2 SKELETON: every reading above is NULL by construction (no brain, no "
        "floor, no feed). The reds go green at 2.9 step 4."
    )


if __name__ == "__main__":
    main()
