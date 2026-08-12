r"""
Executive-control adapters over the COMPETENCE vector of a developmental / NG-SNN training loop.

What lives here after the 2026-08-12 retirement pass (WorkspaceBroadcast and ExecutivePlanner
DELETED -- the RETIREMENT RECORD below is the durable account):

  * DopamineModulator  wraps midbrain.dopamine_nuclei.DopamineNuclei -- turns a reward-prediction
    error over the loop's own COMPETENCE GAIN into a bounded learning-rate multiplier (dopamine
    gates plasticity: a positive surprise -> bigger learning step; a negative surprise -> smaller).
  * CompetenceDynamics + goal_value -- the competence-domain forward model and objective, PUBLIC
    since the retirement: they are the unique substance the retired ExecutivePlanner wrapper
    injected into cortex.prefrontal_executive.PrefrontalExecutive.plan, and consumers now inject
    them into plan() themselves (hippocampus/developmental_loop_brain composes them directly,
    write=False; a live consumer can hand CompetenceDynamics to
    cortex/live_brain.deliberate_ahead as its `predictor` the same way).

Deterministic (per-instance seeds; each wrapped module is seeded from the same seed). ASCII-only.
Stdlib only -- no torch, no GPU, so it is REGIONS tier, not TORCH_TIER (P5: the suite stays
runnable without a card).

BASENAME COLLISION -- WHY THIS FILE IS NAMED WHAT IT IS (resolved at the 2026-08-12 promotion,
#10 lane 4f, promoted+renamed from `reasoning/brain_adapters/executive_control.py`).
  `substrate/dormancy_gate.py:43-50` builds `_CANON` keyed by module BASENAME, and
  `substrate/dormancy_gate.py:79` keys `EXEMPT` the same way; `__init__.py:428` builds the flat
  back-compat alias map `_MOVED` by basename too. `executive_control` was ALREADY registered as
  `cortex.executive_control` (`__init__.py:142`, `run_selfchecks.py:110`) -- a completely different
  faculty (ACC conflict monitoring + EVC + a sequence/tree planner). Promoting this file under
  that basename would have collided in `_CANON` -- LOUDLY, not silently: `brain_atlas.py:81`
  asserts basename uniqueness across the tiers and brain_atlas runs in gate command #1 (an
  earlier draft here claimed the drop would be silent; specs/010 defect C corrected that). The
  file was therefore RENAMED at promotion: `competence_executive`, free tree-wide, and it names
  the actual job -- executive control over a competence vector. specs/010's own region mandate
  (`neuromodulation/dopamine_modulator.py`) was scoped to a DopamineModulator-ONLY split; landing
  the WHOLE file diverges from it deliberately, on a measurement: at neuromodulation/ the p6
  ratchet reds (undeclared_import_pairs 11 -> 12, new pair midbrain<->neuromodulation), while at
  cortex/ all five p6 criteria are bit-identical to baseline, and the split would have rewritten
  the governed, p6-pinned consumer -- the exact bundling specs/010's own correction forbids.
  Nothing in this file hardcodes its own path or name -- see `_self_dotted()`.

P2 -- what already exists, checked before proposing promotion rather than after:
  * WorkspaceBroadcast -- RETIRED 2026-08-12 (the pass specs/010 mandated; deleted in the same
    commit that rewrote its one consumer). It was `GlobalWorkspace.submit()` + the zero-arg read
    plus ONE expression (`ignited = current == content`), and constructing its own workspace made
    it one of the private, mutually invisible instances `tools/p6_gate.py` counts. Everything it
    owed was ported FIRST: the ignition VERDICT lives on
    `brainstem/autonomous_loop._workspace_broadcast` as `_ws_percept_ignited` (consumed by
    `substrate/brain_telemetry`, the same surface that reads `_ignitions`); the tenant contract
    was fixed at 4b24c1da (submit-and-read -- a tenant of the one bus must not drive its clock;
    the measured 20/20 (content, hold) divergence, the sub-threshold 3/20, and the criterion's
    own 4/20 reachable red are all recorded in that commit and CHANGELOG 2026-08-12h);
    `hippocampus/developmental_loop_brain` now constructs `GlobalWorkspace` directly as the OWNER
    of its private offline bus (submit + step + verdict inline, legitimate owner semantics). The
    2026-08-06 live-bus CHARGE measurement (the ignition verdict flips at the rheobase when the
    arm's producer is neutered; harness and neuter-sensitivity control in the 4b24c1da history)
    stays credited to the CHANNEL -- ws= injection -- and the channel survives as
    `GlobalWorkspace` itself plus `live_brain.workspace`, the handle a faculty is given instead
    of building a private bus. Nothing measured was lost in the deletion.
  * DopamineModulator is NOT a duplicate, but half of it is. `brainstem/autonomous_loop.py:1800-
    1826` already runs RPE -> `DopamineNuclei.da_level` on the live tick -- and stores the result as
    TELEMETRY (`self._da_level`) while the learning rate is scaled by `cort.learn_rate_scale(r)`
    (cortisol), not by dopamine. So "dopamine level -> bounded learning-rate multiplier" is done
    nowhere governed. Honest inferiority to record: the governed path derives its RPE from a LEARNED
    predictor (`ValenceLearner.valence(pattern)`), this one from a running mean. If this ever goes
    live, P2 says adopt the learned predictor and delete the running mean.
  * ExecutivePlanner -- RETIRED 2026-08-12, same commit. It was never a mechanism duplicate
    (`PrefrontalExecutive.plan` was always the mechanism, and
    `cortex/executive_control.plan_sequence` is a deeper search over a different domain); what it
    held was the competence-domain forward model + objective, and those are now the PUBLIC
    `CompetenceDynamics` + `goal_value` that consumers inject into plan() directly. The class
    shell carried only a per-instance seed and a `last_rt` attribute. Its write=False
    stateless-planning capability was ported to `cortex/live_brain.deliberate_ahead(write=)` at
    d0984e3b before the deletion, with its own self-check pin.

P7 (dependencies): nothing here was compromised by the retired zero-dependency rule, and the
docstring boast that claimed otherwise ("RESEARCH-tier code: it lives outside src/") was deleted --
it was also factually wrong, since this file is inside `src/`. numpy was MEASURED on the shapes this
module actually runs, not assumed. Protocol stated so the figures can be RE-DERIVED rather than
inherited: `timeit`, 300,000 calls per row, n = 10, `.venv-lake` CPython 3.11.15, numpy 2.4.6.
A RANGE is quoted per row because the cost depends on which numpy formulation is written, and a
single figure hid that:

  `goal_value`                stdlib 0.257 us/call
                               numpy  2.857 (`a.mean() + a.min()`) .. 4.338 (`np.mean(a)+np.min(a)`)
  `CompetenceDynamics.predict` stdlib 0.150 us/call
                               numpy  0.434 (copy + scalar write) .. 0.914 (rebuilt from the tuple)

numpy is therefore 11x-17x slower on the goal value and 2.9x-6.1x slower on the predictor. Array
dispatch dominates at n=10, so stdlib is correct ON THE MERITS here, not on purity; the dependency
would buy nothing. This changes if the competence vector ever becomes wide (thousands of skills),
and then it should be re-measured rather than argued. (An earlier draft of this paragraph quoted
0.272/5.143 us and 0.363/0.688 us. The two STDLIB figures reproduce; neither numpy figure does --
5.143 us is above every one of the four formulations measured above, and 0.688 us falls between two
of them, so the quoted 18.9x and 1.9x ratios were not reproducible in either direction. Corrected
against measurement rather than restated.)

DOMAIN PRECONDITION (disclosed because it is what makes one former assert vacuous): competence
entries are fractions in [0, 1]. On that domain "study the weakest candidate" is a THEOREM about
`CompetenceDynamics` + `goal_value`, not a testable claim -- 0 reds in 500,000 random in-domain
trials (re-run and reproduced). It is FALSE outside the domain (competence > 1 makes the headroom
negative) and it is VOID for a degenerate `gain == 0`, where every candidate ties at exactly zero
gain and the rule has nothing left to discriminate on; both boundaries are asserted in the
self-check instead.

Integration shapes passed to PrefrontalExecutive.plan (the main integration risk -- taken from the
module's OWN plan_scene / goal_value demo, not guessed):
  state      = the length-10 competence vector as an immutable tuple (predictor returns a fresh one).
  options    = the candidate digit indices (ints), used both as WTA options AND as predictor commands.
  predictor  = an OBJECT with .predict(state, command) (NOT a bare callable -- plan() calls
               predictor.predict(s, command)); returns the predicted next competence tuple.
  goal_value = a bare function goal_value(state) -> scalar (plan() calls it directly).
  plan(...)  returns (goal_index, mean_rt); goal_index indexes into `options`, so the chosen digit is
               options[goal_index].

Run the self-check with the invocation this file PRINTS for itself (derived from __file__, so a
move cannot leave a stale command behind in this docstring).
"""

import importlib.util
import os

from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.midbrain.dopamine_nuclei import DopamineNuclei


# ---------------------------------------------------------------------------------------------
# 0. Self-location -- DERIVED, never a hardcoded string
# ---------------------------------------------------------------------------------------------
def _self_dotted():
    """This file's dotted import path, derived from __spec__ / __file__ rather than written down.

    The previous docstring hardcoded `research/brain_adapters/executive_control.py`, which stopped
    being true the moment the file moved into `src/` -- and nothing failed, because a path in prose
    cannot go red. Deriving it means the printed invocation follows the file, and the self-check can
    ASSERT that the derived name is importable, so a bad move fails loudly instead of quietly."""
    spec = globals().get("__spec__")
    name = getattr(spec, "name", None) if spec is not None else None
    if name:
        return name
    directory, base = os.path.split(os.path.abspath(__file__))
    parts = [os.path.splitext(base)[0]]
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        directory, base = os.path.split(directory)
        parts.append(base)
    return ".".join(reversed(parts))


def _self_src_root():
    """The directory that must be on PYTHONPATH for `_self_dotted()` to import. Derived likewise."""
    directory = os.path.dirname(os.path.abspath(__file__))
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        directory = os.path.dirname(directory)
    return directory


# ---------------------------------------------------------------------------------------------
# 1. DopamineModulator -- RPE over competence gain -> bounded learning-rate multiplier
# ---------------------------------------------------------------------------------------------
class DopamineModulator:
    """Wrap the midbrain dopamine nuclei to gate the loop's learning rate by reward-prediction error.

    modulate(competence_gain, expected_gain=None) computes an RPE = competence_gain - baseline (an
    explicit expected_gain when given, else an internal running mean of past gains), feeds it to the
    nucleus's da_level(rpe) (dopamine level in [0, 1]), and maps that level to a bounded learning-rate
    multiplier. Higher dopamine (positive surprise) -> a bigger learning step; lower dopamine (negative
    surprise) -> a smaller step. Deterministic given seed.

    Learning-rate map (documented, bounded):  lr_gain = LR_MIN + (LR_MAX - LR_MIN) * da_level
    Because da_level is clipped to [0, 1] at `midbrain/dopamine_nuclei.py:96`, lr_gain is an affine
    image of [0, 1] onto [LR_MIN, LR_MAX] -- it saturates at both ends and cannot leave them. With
    the nucleus defaults (tonic_baseline = 0.2) a zero RPE rests at lr_gain 0.800; a strong positive
    surprise saturates da_level and reaches exactly LR_MAX. (The previous docstring said 0.725 for
    the resting value, which was arithmetically wrong -- 0.5 + 1.5*0.2 = 0.8. The self-check now
    derives that number from the nucleus's own tonic_baseline so it cannot drift again.)

    P1 -- ADJUDICATED AT THE 2026-08-12 PROMOTION (#10 lane 4f): the pair stays FIXED, and NO
    documented P1 exception carries it. The exception-(c) claim an earlier draft made here was
    measured false twice over: (c) requires default-off with zero live callers, and the consumer's
    dopamine fold defaults ON (dlb's NORD_DOPA=1) -- and the pair is not inert: on the consumer's
    default path (which executes inside `python run_selfchecks.py` via its TRAIN_GATED entry)
    modulate() returns exactly LR_MAX, exactly tonic_lr, exactly LR_MIN and one interior value
    across its four calls, so the pair sits AT a bound on three calls of four and a learned form
    would charge. Conversion is nevertheless DECLINED, on two constraints nameable today (neither
    is a promise about tomorrow): (1) P2, not P1 -- every constructible running-baseline form for
    this pair is built on `_gain_sum`/`_gain_n`, the hand-rolled running-mean RPE baseline
    specs/010 finding G already flags as a THIRD RPE mechanism entering the governed tier beside
    autonomous_loop's ValenceLearner RPE and DopamineNuclei's own TD critic; discharging P1 by
    deepening an open P2 finding is not a discharge. (2) The conversion would vacate criteria
    silently: the self-check reads LR_MIN/LR_MAX as CLASS attributes in five asserts plus the
    closing print, and a conversion keeping them as fallbacks leaves those reads green against
    numbers the mechanism no longer uses (control must follow the read path); all six read sites
    must be rewritten in the same commit as any future conversion. They are not vacuous today:
    removing the nucleus's clip through the `nucleus=` port drives lr_sat_hi to ~1.2e6, so the
    saturation asserts carry a reachable red on the composition WITH the clip. Disclosed, not
    defended: LR_MAX*LR_MIN == 1.0 exactly (one quantity, an octave either side of no-modulation),
    but the affine map never uses that symmetry -- zero RPE rests on 0.8, not 1.0. Also disclosed:
    the exact ceiling hit is a property of the consumer's 4-cycle mechanics check (a freshly
    studied digit scores 12/12 on its per-digit slice against a zero baseline, rpe exactly 1.0);
    at DLB_RUN=1 scale it need not recur, and the x0.75-x3.00 figure in the consumer's own
    annotation has not been reproduced by anyone. The consumer's fold multiplies its readout lr by
    this value (`_mechanics_check`'s `(1.0 + 0.5*na) * da_gain` on the default path; `main()`'s
    DOPA-guarded fold behind DLB_RUN=1), so how far a positive surprise may amplify learning is
    decided by LR_MAX."""

    LR_MIN = 0.5
    LR_MAX = 2.0

    def __init__(self, seed=0, nucleus=None):
        """`nucleus` accepts an ALREADY-LIVE DopamineNuclei (P6: the live tick owns one at
        `brainstem/autonomous_loop.py:1802`). Default None keeps the previous private construction,
        so this is a new port, not a behaviour change."""
        self.seed = seed
        self._nuc = DopamineNuclei(seed=seed) if nucleus is None else nucleus
        self.shared_nucleus = nucleus is not None
        self._gain_sum = 0.0  # running-mean baseline of observed competence gains
        self._gain_n = 0
        self.da = self._nuc.tonic_baseline  # last dopamine level (for logging); starts tonic

    @property
    def tonic_lr(self):
        """The resting learning-rate multiplier at zero RPE, derived from the nucleus's own tonic."""
        return self.LR_MIN + (self.LR_MAX - self.LR_MIN) * self._nuc.tonic_baseline

    def _baseline(self):
        """Running mean of past competence gains (0.0 before any gain has been observed)."""
        return self._gain_sum / self._gain_n if self._gain_n else 0.0

    def modulate(self, competence_gain, expected_gain=None):
        """RPE = competence_gain - (expected_gain or running-mean baseline) -> bounded lr multiplier.

        The RPE is fed to the real nucleus's da_level(rpe); the resulting dopamine level in [0, 1] maps
        linearly to a learning-rate multiplier in [LR_MIN, LR_MAX]. The observed gain always updates the
        internal running-mean baseline so future RPEs are measured against realised progress."""
        baseline = self._baseline() if expected_gain is None else float(expected_gain)
        rpe = float(competence_gain) - baseline
        level = self._nuc.da_level(rpe)  # real dopamine level in [0, 1]
        self.da = level
        # update the running-mean baseline with the realised gain (so RPE stays meaningful over time)
        self._gain_sum += float(competence_gain)
        self._gain_n += 1
        return self.LR_MIN + (self.LR_MAX - self.LR_MIN) * level


# ---------------------------------------------------------------------------------------------
# 2. CompetenceDynamics + goal_value -- the competence-domain forward model and objective
#    (public since the 2026-08-12 retirement: consumers inject them into
#    PrefrontalExecutive.plan themselves; see the RETIREMENT RECORD in the module docstring)
# ---------------------------------------------------------------------------------------------
class CompetenceDynamics:
    """The forward model injected into PrefrontalExecutive.plan -- an OBJECT with .predict(state, cmd).

    plan() calls predictor.predict(s, command); it does NOT call a bare function, so the competence
    model must be wrapped in an object exposing .predict. Studying digit `command` raises that digit's
    competence toward 1.0 by an amount proportional to its current HEADROOM (1 - competence[command]) --
    bigger expected gain where the agent is weak-but-learnable; every other digit is unchanged. Returns
    the predicted next competence as a tuple of the SAME length (immutable, so rollouts do not alias).

    P1 on `gain`: it is a dimensionless fraction of a normalized headroom (exception (a)), and -- the
    part worth measuring rather than asserting -- the planned digit is INVARIANT to it. Measured over
    gain in {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0} the executive plans the same digit every time; only
    the spiking reaction time changes (11.04 steps at 0.5, 200.0 at 0.05, where the race hits its cap).
    A constant that provably cannot change the output is not arbitrating behaviour. The self-check
    re-runs that sweep rather than trusting this paragraph."""

    def __init__(self, gain=0.5):
        self.gain = gain  # fraction of the remaining headroom closed per study step

    def predict(self, state, command):
        nxt = list(state)
        c = nxt[command]
        nxt[command] = c + self.gain * (
            1.0 - c
        )  # close a fraction of the headroom of the studied digit
        return tuple(nxt)


def goal_value(state):
    """The scalar the executive MAXIMISES over a planned rollout terminal state.

    goal_value(state) = mean(state) + min(state): the mean term rewards raising EVERYONE, and the min
    (floor) term additionally rewards fixing the WEAKEST digit. Studying a weak-but-learnable digit
    raises both terms the most (it has the most headroom AND it is the floor), so the executive plans
    toward the weakest learnable digit rather than a digit that is already strong (little headroom) or
    already the floor of a saturated set."""
    n = len(state)
    return sum(state) / n + min(state)


# ---------------------------------------------------------------------------------------------
# self-check
#
# P8 note -- two asserts were DELETED here rather than weakened, each because a red was searched for
# and not found:
#   1. `LR_MIN <= lr_gain <= LR_MAX`. `da_level` clips to [0, 1] unconditionally
#      (`midbrain/dopamine_nuclei.py:96`) and lr_gain is an affine image of that interval, so no
#      input reaches the assert with an out-of-range value. Replaced by the SATURATION endpoints,
#      which do discriminate between inputs: a huge +RPE must land exactly on LR_MAX and a huge -RPE
#      exactly on LR_MIN, while a middling RPE lands on neither.
#   2. `max(gains, key=gains.get) == weakest`. On the documented domain (competence in [0, 1]) this
#      is a theorem, not a test: 0 reds in 500,000 random trials over n in [2, 12], arbitrary
#      candidate subsets and gain in {0.05, 0.2, 0.5, 0.9, 1.0}. Replaced by the two BOUNDARY inputs
#      that leave it -- competence above 1, which FLIPS it (negative headroom), and a degenerate
#      gain of 0, which VOIDS it (every candidate ties at exactly zero gain) -- which turns a
#      tautology into a disclosure of the precondition.
# The surviving planner assert (`chosen == weakest`) keeps a reachable red and now ships with the
# control that proves it: on a near-tied landscape the same harness MISSES the weakest on 31 of 40
# seeds, while on the clear landscape it misses on 0 of 40.
# ---------------------------------------------------------------------------------------------
def main():
    dotted, src_root = _self_dotted(), _self_src_root()
    print("=" * 74)
    print("EXECUTIVE CONTROL ADAPTERS -- dopamine RPE / competence-domain PFC planning")
    print(f"module: {dotted}")
    print(f"run:    PYTHONPATH={os.path.basename(src_root)} python -m {dotted}")
    print("=" * 74)

    # the derived name must actually import -- a stale hardcoded path could never fail; this can.
    assert importlib.util.find_spec(dotted) is not None, (
        f"self-derived module path {dotted!r} does not import -- the derivation is wrong"
    )

    # ---- (a) DopamineModulator: positive surprise -> higher lr than negative surprise --------------
    # Use an explicit expected_gain baseline so the two surprises are isolated from each other.
    dm = DopamineModulator(seed=0)
    lr_pos = dm.modulate(
        competence_gain=0.9, expected_gain=0.4
    )  # realised >> expected -> +RPE -> burst
    da_pos = dm.da
    lr_neg = dm.modulate(
        competence_gain=0.1, expected_gain=0.4
    )  # realised << expected -> -RPE -> dip
    da_neg = dm.da
    # running-mean baseline path (no expected_gain given): still returns a bounded lr
    dm2 = DopamineModulator(seed=0)
    for g in (0.3, 0.35, 0.32):
        dm2.modulate(g)  # build a ~0.32 running-mean baseline
    lr_running = dm2.modulate(0.9)  # a positive surprise vs that baseline
    # saturation ends + the resting point, all derived from the nucleus's own tonic baseline
    lr_sat_hi = DopamineModulator(seed=0).modulate(1e6, 0.0)
    lr_sat_lo = DopamineModulator(seed=0).modulate(-1e6, 0.0)
    dm_rest = DopamineModulator(seed=0)
    lr_rest = dm_rest.modulate(0.0, 0.0)
    print("-" * 74)
    print("(a) dopamine-modulated learning rate:")
    print(
        f"    positive surprise (gain 0.9 vs exp 0.4): da_level {da_pos:.3f} -> lr_gain {lr_pos:.3f}"
    )
    print(
        f"    negative surprise (gain 0.1 vs exp 0.4): da_level {da_neg:.3f} -> lr_gain {lr_neg:.3f}"
    )
    print(f"    running-mean baseline path -> positive surprise lr_gain {lr_running:.3f}")
    print(
        f"    saturation: +RPE -> {lr_sat_hi!r} (LR_MAX), -RPE -> {lr_sat_lo!r} (LR_MIN); "
        f"zero-RPE rest -> {lr_rest!r}"
    )

    assert lr_pos > lr_neg, (
        f"a positive competence surprise must raise lr above a negative one ({lr_pos} vs {lr_neg})"
    )
    # RECORDED RED for the line above: the ordering must follow the RPE, not the call order. Feeding
    # the SAME two gains in the opposite order reverses the inequality (measured 0.5 then 1.4).
    ctl = DopamineModulator(seed=0)
    red_first, red_second = ctl.modulate(0.1, 0.4), ctl.modulate(0.9, 0.4)
    assert red_first < red_second, (
        f"control: swapping the two gains must reverse the ordering, got {red_first} vs {red_second}"
    )

    assert lr_sat_hi == DopamineModulator.LR_MAX, (
        f"a saturating positive RPE must land exactly on LR_MAX, got {lr_sat_hi}"
    )
    assert lr_sat_lo == DopamineModulator.LR_MIN, (
        f"a saturating negative RPE must land exactly on LR_MIN, got {lr_sat_lo}"
    )
    assert DopamineModulator.LR_MIN < lr_pos < DopamineModulator.LR_MAX, (
        f"an UNSATURATED surprise must land strictly inside the ends, got {lr_pos}"
    )
    assert lr_rest == dm_rest.tonic_lr, (
        f"a zero RPE must rest on the nucleus's own tonic baseline ({dm_rest.tonic_lr}), got {lr_rest}"
    )

    assert lr_running > DopamineModulator.LR_MIN, (
        "a positive surprise vs the running mean must lift lr above the floor"
    )
    # RECORDED RED for the line above: build the running mean from HIGH gains, then report a low one
    # -- the RPE goes negative, da_level clips to 0 and lr lands exactly on the floor.
    dm_red = DopamineModulator(seed=0)
    for g in (0.9, 0.9, 0.9):
        dm_red.modulate(g)
    lr_floor = dm_red.modulate(0.0)
    assert lr_floor == DopamineModulator.LR_MIN, (
        f"control: a negative surprise vs a high running mean must hit the floor, got {lr_floor}"
    )

    # ---- (b) competence-domain planning: PrefrontalExecutive.plan over the PUBLIC domain objects --
    # (was (c); the old section (b) died with WorkspaceBroadcast -- its criteria are recorded at the
    # retirement, and the tenant-contract clock criterion lives in the 4b24c1da history)
    def _plan(comp_vec, cands, seed=0, gain=0.5):
        """The retired ExecutivePlanner's composition, inline at its criteria: a FRESH executive
        per call (n_options must equal len(cands); autonomous_loop pins its own executive to
        len(ACTIONS)), the public domain objects injected, write=False -- a planning QUERY never
        writes the executive's working-memory goal. Returns (chosen_digit, mean_rt). The one
        production consumer (hippocampus/developmental_loop_brain) composes the same call at its
        own site; this helper exists so every criterion below exercises the composition shape."""
        ex = PrefrontalExecutive(n_options=len(cands), seed=seed)
        idx, rt = ex.plan(
            tuple(float(x) for x in comp_vec),
            list(cands),
            CompetenceDynamics(gain=gain),
            goal_value,
            horizon=1,
            write=False,
        )
        return list(cands)[idx], rt

    # Competence vector: digit 3 is clearly the weakest (most headroom); others are already strong.
    competence = [0.80, 0.75, 0.82, 0.15, 0.78, 0.79, 0.83, 0.77, 0.81, 0.76]
    candidates = [0, 3, 6, 9]  # digit 3 is the weak-but-learnable candidate
    weakest = min(candidates, key=lambda d: competence[d])
    # show goal_value gains so the plan target is not a black box
    dyn = CompetenceDynamics(gain=0.5)
    base_gv = goal_value(tuple(competence))
    gains = {d: goal_value(dyn.predict(tuple(competence), d)) - base_gv for d in candidates}
    chosen, chosen_rt = _plan(competence, candidates)
    print("-" * 74)
    print("(b) prefrontal planning of the next study objective:")
    print(f"    competence {competence}")
    print(
        f"    candidates {candidates}; goal_value gain per candidate "
        f"{ {d: round(g, 4) for d, g in gains.items()} }"
    )
    print(
        f"    weakest-but-learnable digit = {weakest}; executive PLANNED digit = {chosen} "
        f"(mean rt {chosen_rt:.1f} steps)"
    )

    assert chosen == weakest, (
        f"the executive must plan to study the weakest-but-learnable digit {weakest}, got {chosen}"
    )

    # SENSITIVITY CONTROL (P8 item 2): the assert above is not automatic. Shrink the gap between the
    # two candidates' rollout values -- plan() shifts values by a fixed +0.1 offset before the race,
    # so a small gap is a near-tie in drift space -- and the SAME harness starts missing.
    tied = [0.8615, 0.2126, 0.4615, 0.179, 0.5669, 0.083, 0.7167, 0.6774, 0.0878, 0.3164]
    tied_cands = [4, 8]
    tied_weakest = min(tied_cands, key=lambda d: tied[d])
    n_seeds = 40
    tied_miss = sum(
        _plan(tied, tied_cands, seed=s)[0] != tied_weakest for s in range(n_seeds)
    )
    clear_miss = sum(
        _plan(competence, candidates, seed=s)[0] != weakest for s in range(n_seeds)
    )
    print(
        f"    sensitivity control over {n_seeds} seeds: near-tied landscape misses {tied_miss}, "
        f"clear landscape misses {clear_miss}"
    )
    assert clear_miss == 0, (
        f"the clear landscape must be planned correctly under every seed, got {clear_miss} misses"
    )
    assert 0 < tied_miss < n_seeds, (
        "the harness must be SENSITIVE: a near-tied landscape has to produce both hits and misses, "
        f"else 'chosen == weakest' proves nothing (got {tied_miss}/{n_seeds})"
    )

    # BOUNDARY inputs that make 'study the weakest' FALSE -- these replace the deleted tautology and
    # disclose the precondition that made it one.
    # competence ABOVE 1: headroom is negative, so studying the weaker one LOSES more. The pair is
    # not arbitrary -- at exactly [1.2, 1.6] the two gains TIE at -0.15 and the rule survives by the
    # tie-break, which is why the red needs a gap that the floor term cannot cancel.
    oob = [1.25, 1.65]
    oob_base = goal_value(tuple(oob))
    oob_gains = {d: goal_value(dyn.predict(tuple(oob), d)) - oob_base for d in (0, 1)}
    assert max(oob_gains, key=oob_gains.get) != min((0, 1), key=lambda d: oob[d]), (
        f"competence above 1 must break the weakest-first rule (gains {oob_gains}) -- the rule holds "
        "only on the documented [0, 1] domain"
    )
    # gain == 0 does NOT flip the rule, it VOIDS it: every candidate ties at exactly zero gain, so
    # there is no signal left to rank on. This check used to read
    # `max(flat_gains, key=flat_gains.get) != weakest` over the single state [0.8, 0.1]. On an exact
    # tie `max` returns the FIRST key, so that assert was decided by dict-insertion order and not by
    # the mechanism: it was green for [0.8, 0.1] and RED for [0.1, 0.8] with nothing about
    # `CompetenceDynamics` changed (measured, both orders). A criterion whose colour is set by a
    # tie-break is replaced, not kept. The form below asserts the tie itself, is evaluated in BOTH
    # state orders so no tie-break can satisfy it, and reddens on any gain != 0.
    flat = CompetenceDynamics(gain=0.0)  # degenerate forward model: every candidate gains nothing
    flat_gains = {}
    for flat_state in ([0.8, 0.1], [0.1, 0.8]):
        flat_base = goal_value(tuple(flat_state))
        flat_gains[tuple(flat_state)] = {
            d: goal_value(flat.predict(tuple(flat_state), d)) - flat_base for d in (0, 1)
        }
    assert all(set(g.values()) == {0.0} for g in flat_gains.values()), (
        "a zero-gain forward model must leave the weakest-first rule with NO signal to rank on -- "
        f"every candidate gain must be exactly 0.0 in either state order (gains {flat_gains})"
    )
    print(
        f"    boundary: competence>1 gains {oob_gains} (FLIPS the weakest-first rule); "
        f"gain=0 gains {flat_gains} (VOIDS it -- an exact tie in either state order)"
    )

    # P1 discharge for CompetenceDynamics.gain: the planned digit is INVARIANT to it; only the
    # spiking reaction time moves. Re-measured here rather than asserted from the docstring.
    gain_sweep = {}
    for g in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):  # exactly the set the docstring names
        g_digit, g_rt = _plan(competence, candidates, gain=g)
        gain_sweep[g] = (g_digit, round(g_rt, 2))
    print(f"    gain sweep (digit, mean_rt): {gain_sweep}")
    assert len({d for d, _ in gain_sweep.values()}) == 1, (
        f"the planned digit must be invariant to the forward model's gain, got {gain_sweep}"
    )
    assert len({rt for _, rt in gain_sweep.values()}) > 1, (
        f"control: gain must still REACH the rollout (distinct reaction times), got {gain_sweep}"
    )

    # determinism: same seed reproduces the same plan -- AND the seed is actually WIRED into the
    # spiking race (otherwise 'deterministic' would be trivially true because the seed does nothing).
    assert _plan(competence, candidates)[0] == chosen, (
        "same seed must reproduce the planned objective"
    )
    rts = set()
    for s in range(8):
        rts.add(_plan(competence, candidates, seed=s)[1])
    assert len(rts) > 1, (
        f"control: the seed must reach the spiking race (distinct mean reaction times), got {rts}"
    )

    print("=" * 74)
    print(
        f"self-check OK: dopamine lr {lr_pos:.3f}(+) > {lr_neg:.3f}(-), saturating on "
        f"[{DopamineModulator.LR_MIN}, {DopamineModulator.LR_MAX}];"
    )
    print(
        f"  prefrontal executive plans the weakest-but-learnable digit {chosen} (control: "
        f"{tied_miss}/{n_seeds} misses on a near-tied landscape); deterministic, seed wired"
    )


if __name__ == "__main__":
    main()
