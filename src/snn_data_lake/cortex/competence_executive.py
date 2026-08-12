r"""
Executive-control adapters over the COMPETENCE vector of a developmental / NG-SNN training loop.

Three thin adapters over shipped `snn_data_lake.*` modules that give the loop an EXECUTIVE control
layer built from real brain parts instead of hand-rolled scaffolding:

  * DopamineModulator  wraps midbrain.dopamine_nuclei.DopamineNuclei -- turns a reward-prediction
    error over the loop's own COMPETENCE GAIN into a bounded learning-rate multiplier (dopamine
    gates plasticity: a positive surprise -> bigger learning step; a negative surprise -> smaller).
  * WorkspaceBroadcast wraps thalamus.global_workspace.GlobalWorkspace -- posts the current
    objective / competence summary into the single-slot spiking workspace so a strong-enough goal
    IGNITES and becomes globally available (coordination + logging; NOT an accuracy lever).
  * ExecutivePlanner  wraps cortex.prefrontal_executive.PrefrontalExecutive -- PLANS the next digit
    to study by rolling a simple competence-dynamics forward model through the spiking WTA decision
    attractor, picking the digit that most raises a competence goal_value (weak-but-learnable first).

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
  * WorkspaceBroadcast IS a duplicate. `GlobalWorkspace.submit()/step()/current` is the entire
    mechanism; this wrapper adds one expression (`ignited = current == content`). It also creates a
    PRIVATE workspace, which is the defect `tools/p6_gate.py`'s private_instance_modules criterion
    counts (de-lined 2026-08-12: the old `:50` cite had drifted, this file's own argument). It is kept
    only because `hippocampus/developmental_loop_brain.py` imports the name (LAZILY inside main()
    since its 2026-08-12 promotion) and this wave may not edit consumers; it is redesigned here to
    ACCEPT an injected shared workspace so it no longer MANDATES a private one. Retire it once
    that import is repointed.
    THE COST OF THAT REDESIGN, DISCLOSED BECAUSE IT MOVED THE INSTRUMENT THE WRONG WAY. P6's census
    is RE-DERIVED here rather than quoted, which is what `docs/POLICIES.md` instructs
    ("Re-derive with it rather than quoting a figure"). **The tree-wide totals this paragraph used
    to carry are DELETED rather than updated, because they were re-derived twice inside a single
    session on 2026-08-06 and moved both times**: the sentence here read "31 instantiation sites
    across 9 modules, 16 of them private" (itself a correction of POLICIES.md's 26 / 11); measured
    at the start of that session it was 40 / 9 with 16 private, and measured an hour later 47 / 9
    with 23 private -- with **every one of those increments coming from OTHER modules** being edited
    concurrently, none from this file. A tree-wide count written down here is therefore not stale
    from age, it is stale from the tree having more than one author, which no re-derivation cadence
    fixes. What is durable and is what a reader needs from this paragraph: THIS FILE holds **9** of
    them -- the wrapper's own `_ws`, plus `probe`, `wb_hi`, `wb_lo`, `wb_at`, `wb_under`,
    `shared_ws`, `share_a`, `share_b` in the self-check
    (named rather than cited by line, because writing a line number here is the drift this file
    already refuses for its own module path; `python tools/p6_gate.py` prints all nine with their
    current lines). That is up from 3 at HEAD, and every one of the six new ones is a self-check
    FIXTURE: p6_gate excludes only the two BUS modules' self-checks, so a faculty's test fixtures
    are counted as production private buses. They are not collapsible -- each ignition case needs a
    FRESH
    workspace or it inherits the previous case's ignited content, and reusing one to shrink a count
    would be gaming the instrument rather than fixing anything. p6_gate's own rule is "a rising READ
    count against a flat WRITE count is progress; a rising write count is not" -- this file's WRITE
    count went 3 -> 7 with faculty-to-faculty reads flat at 0 of 9. Net: the `ws=` parameter is a
    real channel, and the self-check demonstrating it made the measured P6 number worse. Both halves
    belong in the promotion decision -- and a THIRD half was added 2026-08-06: the channel was
    finally driven against the LIVE shared bus and its CHARGE measured (the wrapper's own ignition
    verdict flips at the rheobase when the arm's producer is neutered), together with the reason it
    was then not yet shipped live (`broadcast()` stepped a clock it did not own -- the contract
    defect FIXED 2026-08-12: submit-and-read, step only on the private bus the wrapper owns). Both
    measurements, with their harnesses and their control, are recorded on `WorkspaceBroadcast`
    itself rather than here.
  * DopamineModulator is NOT a duplicate, but half of it is. `brainstem/autonomous_loop.py:1800-
    1826` already runs RPE -> `DopamineNuclei.da_level` on the live tick -- and stores the result as
    TELEMETRY (`self._da_level`) while the learning rate is scaled by `cort.learn_rate_scale(r)`
    (cortisol), not by dopamine. So "dopamine level -> bounded learning-rate multiplier" is done
    nowhere governed. Honest inferiority to record: the governed path derives its RPE from a LEARNED
    predictor (`ValenceLearner.valence(pattern)`), this one from a running mean. If this ever goes
    live, P2 says adopt the learned predictor and delete the running mean.
  * ExecutivePlanner is NOT a duplicate. `PrefrontalExecutive.plan` is the mechanism; the unique
    part here is the competence-domain forward model + goal_value injected into it.
    `cortex/executive_control.plan_sequence` searches action SEQUENCES over developer-supplied
    value/transition estimates -- a deeper search over a different domain, not this job (the
    2026-08-12 rename retired the basename ambiguity between that module and this one).

P7 (dependencies): nothing here was compromised by the retired zero-dependency rule, and the
docstring boast that claimed otherwise ("RESEARCH-tier code: it lives outside src/") was deleted --
it was also factually wrong, since this file is inside `src/`. numpy was MEASURED on the shapes this
module actually runs, not assumed. Protocol stated so the figures can be RE-DERIVED rather than
inherited: `timeit`, 300,000 calls per row, n = 10, `.venv-lake` CPython 3.11.15, numpy 2.4.6.
A RANGE is quoted per row because the cost depends on which numpy formulation is written, and a
single figure hid that:

  `_goal_value`                stdlib 0.257 us/call
                               numpy  2.857 (`a.mean() + a.min()`) .. 4.338 (`np.mean(a)+np.min(a)`)
  `_CompetenceDynamics.predict` stdlib 0.150 us/call
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
`_CompetenceDynamics` + `_goal_value`, not a testable claim -- 0 reds in 500,000 random in-domain
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
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace


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
# 2. WorkspaceBroadcast -- ignite the current goal into the single-slot spiking workspace
# ---------------------------------------------------------------------------------------------
class WorkspaceBroadcast:
    """Wrap the spiking global workspace to broadcast the loop's current objective / competence summary.

    broadcast(source, content, salience) submits the content into this tick's competition at the given
    salience (the external drive current on that content's LIF pool) and returns the bus's current
    broadcast (submit-and-READ, the contract fixed 2026-08-12). On a PRIVATE workspace (ws=None) it
    also runs the one settle-to-ignition cycle itself -- this wrapper owns that bus's clock -- so a
    supra-threshold salience ignites into sustained firing and is broadcast while a sub-threshold one
    stays local (returns None). On a SHARED workspace it NEVER steps: the owner's tick decides, and
    read() reports the verdict after it. This is coordination + logging (the "conscious" broadcast of
    the current goal), NOT an accuracy lever. self.ignited records whether the last submitted content
    actually seized the slot.

    P2/P6, stated plainly because it decides this class's future: this is a duplicate of
    `GlobalWorkspace` itself plus one expression, and constructing its own workspace makes it one of
    the private, mutually invisible instances `tools/p6_gate.py` counts. Passing `ws` shares an
    existing workspace instead -- that is the construction-path change P6 says must come before any
    faculty can read another's broadcast. It is a channel, not yet a wire, and this class should be
    retired outright once its one importer is repointed at `GlobalWorkspace`. Retirement is
    deferred, not forgotten: see the last paragraph of this docstring for the four ports it owes
    first.

    THE CHANNEL CHARGES -- MEASURED 2026-08-06 ON THE LIVE SHARED BUS, not asserted. Until now the
    paragraph above said only that the channel existed; P6's honesty guard asks for the other half
    (neuter faculty A, show faculty B's OWN output changes), so it was run. Harness:
    `make_brain(seed=0, refit_every=10, persist=False)`, clock.phase forced to 0.25, 60 wake ticks,
    then `WorkspaceBroadcast(seed=0, ws=brain.workspace)` with the identity asserted
    (`wrapper._ws is brain.workspace` -> True). The producer neuter drops ONLY the arm's sources
    ('perception', 'salience') and leaves every other submitter reaching the bus, so the wrapper's
    OWN write is not silenced by its own control. Three saliences, expressed against the bus's own
    rheobase:

        salience      producer LIVE                 producer NEUTERED
        thr * 0.5     0            ignited=False    None            ignited=False
        thr * 1.0     0            ignited=False    study_digit_7   ignited=True
        thr * 1.6     study_digit_7 ignited=True    study_digit_7   ignited=True

    All three outputs differ, and the middle row is the one that carries the claim: AT the rheobase
    the wrapper's own ignition VERDICT flips, because the arm's perception is holding the slot with
    `self_boost + hysteresis` behind it and denies an at-threshold challenger. Remove the producer
    and the identical submission ignites. That is P6 condition (b) satisfied by the handle alone,
    with no new code -- the wrapper consumes another faculty's output whether or not it reads a
    value. Harness sensitivity control, run in the same pass: the neutered bus reports
    `broadcast()=None hold=0.0` against the live bus's `broadcast()=0 hold=0.476...`, so the neuter
    demonstrably lands.

    AND THE SAME MEASUREMENT FOUND THE CONTRACT DEFECT, WHICH IS NOW FIXED (2026-08-12).
    `broadcast()` used to call `self._ws.step()` unconditionally, and `step()` is the tick
    operation the bus OWNER runs (`brainstem/autonomous_loop.py` on the live bus). On a private
    workspace that was legitimate -- the wrapper IS that bus's only clock. On the SHARED one it
    advanced a competition the wrapper does not drive. Measured over 20 arm ticks with the wrapper
    writing every 5th tick, against an identical control that writes nothing: at `thr * 1.6` the
    arm's own (content, hold) differed on 20 of 20 ticks -- its percept never regained the slot --
    and even at a SUB-threshold `thr * 0.5`, which cannot ignite anything, it still differed on
    3 of 20 purely because the out-of-band `step()` decays every attractor. THE FIX: `broadcast()`
    is now SUBMIT-AND-READ -- it submits into the current tick's competition and reads the bus via
    the zero-arg `GlobalWorkspace.broadcast()` reader; it steps ONLY a private workspace (ws=None),
    where this wrapper is the clock's owner rather than a tenant, so the private semantics every
    prior assert pins are unchanged. On a shared bus the ignition verdict is therefore available
    from `read()` AFTER the owner's step, never at submit time. The self-check pins the tenant
    discipline with the criterion this measurement licensed: a tenant wrapper must add NOTHING to
    the bus beyond its submission (its owner's tick sequence is bit-identical to direct submits);
    reintroducing the step reds it deterministically -- the 20/20 divergence above IS that red.

    NOTHING CONSTRUCTS THIS CLASS ON A PRODUCTION PATH -- re-derived at the 2026-08-12 promotion
    rather than carried over. Tree-wide, `WorkspaceBroadcast(` appears at TEN construction sites:
    eight in this module's own `main()` (seven assigned plus one bare call inside a control
    assert) and two in `hippocampus/developmental_loop_brain.py` -- one in its main() behind a
    lazy import and one in its bounded _mechanics_check, both offline (dlb is EXEMPT-permanent
    train-tier, not a live root, so neither site is on the tick). The previous text said "nine
    sites: eight here and one in dlb"; the total was one short and the split predated dlb's
    second site. A separate NINE is also correct and is a DIFFERENT census: `python
    tools/p6_gate.py` prints nine workspace-class INSTANTIATION sites for this file
    (GlobalWorkspace + WorkspaceBroadcast, bound assignments only) -- its _BindVisitor records a
    construction only from an assignment target, so the bare call in the control assert is
    invisible to it. Cite the gate for the module count it asserts on, not for a
    construction-site count. So there is still no live caller to hand the shared handle to, and
    manufacturing one would be the force-wire P6 forbids. The disposition is unchanged and rests
    on a measurement: the channel is real, it charges, its live use needed the `step()` contract
    fixed first (done 2026-08-12, this file), and the class is a retire candidate. RETIREMENT IS A
    SEPARATE CHANGE, and
    specs/010 names what it owes before the class may go (this module's promotion-plan entry,
    MINIMUM SAFE PLAN): port `ignited` into brainstem/autonomous_loop's _workspace_broadcast;
    port ExecutivePlanner's write=False stateless planning (cortex/live_brain's deliberate_ahead
    gate-writes the persistent working-memory goal; PrefrontalExecutive.plan defaults
    write=True); port _CompetenceDynamics and _goal_value; and rewrite developmental_loop_brain,
    which constructs all three classes. Bundling any of that into a move is what turns a rename
    into red gates."""

    def __init__(self, seed=0, ws=None):
        self.seed = seed
        self._ws = GlobalWorkspace(seed=seed) if ws is None else ws
        self.shared = ws is not None  # False = a private workspace (the P6 defect), True = shared
        self.ignited = False  # did the last submission ignite + get broadcast?
        self._last_content = None  # what read() judges ignition against

    @property
    def ignition_threshold(self):
        """The wrapped workspace's own rheobase -- what a salience must clear to ignite.

        Exposed so callers (and this module's self-check) can express saliences RELATIVE to the
        threshold that actually decides, instead of hardcoding numbers that silently stop meaning
        'clearly above' / 'clearly below' if the workspace is retuned (P1)."""
        return self._ws.ignition_threshold

    # distinct-from: thalamus/global_workspace.py -- no mechanism lives here: pure delegation to
    # the ONE bus (submit + the zero-arg broadcast reader + step only on a bus this wrapper OWNS);
    # the wrapper itself is the recorded P2 retire candidate, kept only until its importer repoints.
    def broadcast(self, source, content, salience):
        """Submit content at `salience` into the current tick's competition, then READ the bus.

        SUBMIT-AND-READ (contract fixed 2026-08-12; it was submit-and-STEP): a tenant of the one
        shared bus must not drive its clock, so on a SHARED workspace this NEVER calls step() --
        the owner's tick decides the competition, the return value here is the bus's PRE-step
        content, and read() reports the verdict after the owner steps. On a PRIVATE workspace
        (ws=None) this wrapper is the bus's only clock owner, so it still runs the one
        settle-to-ignition cycle itself -- the previous semantics, unchanged: sub-threshold
        salience (below ignition_threshold) returns None and does not ignite; a supra-threshold
        salience ignites and is globally broadcast."""
        self._ws.submit(source, content, salience)
        self._last_content = content
        if not self.shared:
            self._ws.step()  # private bus only: the wrapper IS the owner of this clock
        return self.read()

    def read(self):
        """Read the bus WITHOUT advancing it (GlobalWorkspace's zero-arg reader); update ignited.

        On a shared bus, call this after the OWNER's step to learn whether the last submission
        seized the slot; ignition can never be observed at submit time there, because the
        competition has not run yet."""
        current = self._ws.broadcast()
        self.ignited = current is not None and current == self._last_content
        return current


# ---------------------------------------------------------------------------------------------
# 3. ExecutivePlanner -- plan the next study objective over competence dynamics
# ---------------------------------------------------------------------------------------------
class _CompetenceDynamics:
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


def _goal_value(state):
    """The scalar the executive MAXIMISES over a planned rollout terminal state.

    goal_value(state) = mean(state) + min(state): the mean term rewards raising EVERYONE, and the min
    (floor) term additionally rewards fixing the WEAKEST digit. Studying a weak-but-learnable digit
    raises both terms the most (it has the most headroom AND it is the floor), so the executive plans
    toward the weakest learnable digit rather than a digit that is already strong (little headroom) or
    already the floor of a saturated set."""
    n = len(state)
    return sum(state) / n + min(state)


class ExecutivePlanner:
    """Wrap the prefrontal executive to PLAN the next digit to study over competence dynamics.

    plan_objective(competence_vec, candidates, horizon=1) builds the length-10 competence vector as the
    plan state, the candidate digits as both the WTA options and the forward-model commands, injects a
    _CompetenceDynamics forward model and the mean+min goal_value, and runs PrefrontalExecutive.plan --
    the spiking winner-take-all decision attractor over the rollout values. plan() returns (goal_index,
    mean_rt) where goal_index indexes into `candidates`, so the chosen digit is candidates[goal_index].
    write=False so planning does not overwrite the executive's persistent working-memory goal here.
    Deterministic given seed.

    The executive is built per call because `n_options` must equal len(candidates), which varies per
    call -- so unlike the nucleus and the workspace this one cannot take an injected shared instance
    (`brainstem/autonomous_loop.py:289` fixes n_options to len(ACTIONS)). Recorded so the asymmetry
    reads as a constraint rather than an oversight."""

    def __init__(self, seed=0, gain=0.5):
        self.seed = seed
        self._dynamics = _CompetenceDynamics(
            gain=gain
        )  # the injected .predict(state, cmd) forward model
        self.last_rt = None  # mean spiking reaction time of the last plan

    def plan_objective(self, competence_vec, candidates, horizon=1):
        """Plan the next digit to study; return the chosen digit (an element of `candidates`)."""
        # fresh executive per call: n_options is the number of WTA options == number of candidate digits
        executive = PrefrontalExecutive(n_options=len(candidates), seed=self.seed)
        state = tuple(float(x) for x in competence_vec)  # length-10 competence tuple (immutable)
        options = list(candidates)  # candidate digit indices (WTA options + commands)
        goal_index, mean_rt = executive.plan(
            state, options, self._dynamics, _goal_value, horizon=horizon, write=False
        )
        self.last_rt = mean_rt
        return options[goal_index]  # map the WTA index back to the digit


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
    print("EXECUTIVE CONTROL ADAPTERS -- dopamine RPE / global-workspace broadcast / PFC planning")
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

    # ---- (b) WorkspaceBroadcast: a goal above the workspace's own rheobase ignites, below stays local
    # P1: the two saliences are expressed RELATIVE to the workspace's own ignition_threshold rather
    # than hardcoded, so "clearly above" / "clearly below" survive a retune of the workspace.
    probe = WorkspaceBroadcast(seed=0)
    thr = probe.ignition_threshold
    sal_hi, sal_lo = min(1.0, thr * 1.6), thr * 0.5
    wb_hi = WorkspaceBroadcast(seed=0)
    hi = wb_hi.broadcast(
        "executive", "study_digit_7", salience=sal_hi
    )  # supra-threshold -> ignites
    hi_ign = wb_hi.ignited
    wb_lo = WorkspaceBroadcast(seed=0)
    lo = wb_lo.broadcast("executive", "study_digit_7", salience=sal_lo)  # sub-threshold -> local
    lo_ign = wb_lo.ignited
    # the boundary itself: AT the rheobase it ignites, just under it does not (the reachable red)
    wb_at = WorkspaceBroadcast(seed=0)
    wb_at.broadcast("executive", "study_digit_7", salience=thr)
    wb_under = WorkspaceBroadcast(seed=0)
    wb_under.broadcast("executive", "study_digit_7", salience=thr * 0.9)
    print("-" * 74)
    print("(b) global-workspace broadcast of the current goal:")
    print(f"    ignition_threshold (rheobase) {thr}")
    print(f"    salience {sal_hi:.4f} (thr*1.6) -> broadcast {hi!r} (ignited={hi_ign})")
    print(f"    salience {sal_lo:.4f} (thr*0.5) -> broadcast {lo!r} (ignited={lo_ign})")
    print(
        f"    boundary: at thr ignited={wb_at.ignited}; at thr*0.9 ignited={wb_under.ignited}"
        f" (private workspace: shared={wb_hi.shared})"
    )

    assert hi == "study_digit_7" and hi_ign, "a supra-threshold goal must ignite and be broadcast"
    assert lo is None and not lo_ign, (
        "a sub-threshold goal must stay local (no ignition, no broadcast)"
    )
    assert wb_at.ignited and not wb_under.ignited, (
        "ignition must track the workspace's OWN rheobase: at threshold yes, just under it no "
        f"(got {wb_at.ignited} / {wb_under.ignited})"
    )

    # P6 channel: two adapters over ONE workspace instead of two private ones -- under the FIXED
    # contract (submit-and-read, 2026-08-12): tenants submit and read; only the bus's OWNER steps.
    shared_ws = GlobalWorkspace(seed=0)
    share_a = WorkspaceBroadcast(seed=0, ws=shared_ws)
    share_b = WorkspaceBroadcast(seed=0, ws=shared_ws)
    pre = share_a.broadcast("executive", "study_digit_7", salience=sal_hi)
    assert pre is None and not share_a.ignited, (
        "a tenant must not be able to ignite anything BY ITSELF: before the owner's step the shared "
        f"bus must still be empty (got {pre!r}, ignited={share_a.ignited}) -- red here means "
        "broadcast() is stepping a shared clock again"
    )
    shared_ws.step()  # the OWNER's tick -- the one competition this cycle
    seen_by_a = share_a.read()
    seen_by_b = share_b.broadcast("monitor", "competence_report", salience=sal_lo)
    print(
        f"    shared instance: A ignited {share_a.ignited} (after the OWNER's step; pre-step read "
        f"{pre!r}); B (sub-threshold) sees A's content {seen_by_b!r} -- shared={share_b.shared}"
    )
    assert share_a.shared and share_b.shared, "an injected workspace must report shared=True"
    assert seen_by_a == "study_digit_7" and share_a.ignited, (
        "after the owner's step the tenant's supra-threshold submission must have ignited "
        f"(got {seen_by_a!r}, ignited={share_a.ignited})"
    )
    assert seen_by_b == "study_digit_7", (
        "on a SHARED workspace B must see the content A ignited; on a private one it would see None "
        f"(got {seen_by_b!r})"
    )
    assert WorkspaceBroadcast(seed=0).broadcast("monitor", "competence_report", sal_lo) is None, (
        "control: the same sub-threshold submission on a PRIVATE workspace must see nothing"
    )

    # THE CLOCK CRITERION (pins the 2026-08-12 contract fix; the measured defect is its red): a
    # tenant wrapper must add NOTHING to the bus beyond its submission. Two identical owner buses,
    # one written through the wrapper every 5th tick, one given the SAME submissions directly --
    # the owner's per-tick (content, hold) sequence must be BIT-IDENTICAL. The pair, NOT content
    # alone: constructed before trusting (the P8 probe), the old submit-and-step contract leaves
    # CONTENT identical on this harness (0/20) while (content, hold) diverges 4/20 -- the
    # out-of-band step() decays the attractor CHARGE first, which is exactly how the live-bus
    # measurement in the class docstring saw it (hold moved on every divergent tick). A
    # content-only criterion here would have NO reachable red and would be deleted, not kept.
    def _owner_run(through_wrapper):
        bus = GlobalWorkspace(seed=0)
        tenant = WorkspaceBroadcast(seed=0, ws=bus)
        seq = []
        for t in range(20):
            bus.submit("perception", "percept", sal_hi)
            if t % 5 == 0:
                if through_wrapper:
                    tenant.broadcast("executive", "study_digit_7", sal_hi)
                else:
                    bus.submit("executive", "study_digit_7", sal_hi)
            seq.append((bus.step(), bus.broadcast_hold))
        return seq

    wrapped_seq, direct_seq = _owner_run(True), _owner_run(False)
    assert wrapped_seq == direct_seq, (
        "the tenant wrapper perturbed the owner's bus beyond its submission -- broadcast() is "
        "driving a clock it does not own again (reachable red: the old submit-and-step contract "
        "diverges this (content, hold) sequence on 4 of 20 ticks)"
    )
    print(
        f"    clock criterion: tenant-wrapped vs direct-submit owner (content, hold) sequences "
        f"identical over 20 ticks ({sum(1 for c, _h in wrapped_seq if c is not None)} non-empty)"
    )

    # ---- (c) ExecutivePlanner: plans to study the weakest-but-learnable digit ----------------------
    # Competence vector: digit 3 is clearly the weakest (most headroom); others are already strong.
    competence = [0.80, 0.75, 0.82, 0.15, 0.78, 0.79, 0.83, 0.77, 0.81, 0.76]
    candidates = [0, 3, 6, 9]  # digit 3 is the weak-but-learnable candidate
    weakest = min(candidates, key=lambda d: competence[d])
    # show goal_value gains so the plan target is not a black box
    dyn = _CompetenceDynamics(gain=0.5)
    base_gv = _goal_value(tuple(competence))
    gains = {d: _goal_value(dyn.predict(tuple(competence), d)) - base_gv for d in candidates}
    ep = ExecutivePlanner(seed=0)
    chosen = ep.plan_objective(competence, candidates, horizon=1)
    print("-" * 74)
    print("(c) prefrontal planning of the next study objective:")
    print(f"    competence {competence}")
    print(
        f"    candidates {candidates}; goal_value gain per candidate "
        f"{ {d: round(g, 4) for d, g in gains.items()} }"
    )
    print(
        f"    weakest-but-learnable digit = {weakest}; executive PLANNED digit = {chosen} "
        f"(mean rt {ep.last_rt:.1f} steps)"
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
        ExecutivePlanner(seed=s).plan_objective(tied, tied_cands) != tied_weakest
        for s in range(n_seeds)
    )
    clear_miss = sum(
        ExecutivePlanner(seed=s).plan_objective(competence, candidates) != weakest
        for s in range(n_seeds)
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
    oob_base = _goal_value(tuple(oob))
    oob_gains = {d: _goal_value(dyn.predict(tuple(oob), d)) - oob_base for d in (0, 1)}
    assert max(oob_gains, key=oob_gains.get) != min((0, 1), key=lambda d: oob[d]), (
        f"competence above 1 must break the weakest-first rule (gains {oob_gains}) -- the rule holds "
        "only on the documented [0, 1] domain"
    )
    # gain == 0 does NOT flip the rule, it VOIDS it: every candidate ties at exactly zero gain, so
    # there is no signal left to rank on. This check used to read
    # `max(flat_gains, key=flat_gains.get) != weakest` over the single state [0.8, 0.1]. On an exact
    # tie `max` returns the FIRST key, so that assert was decided by dict-insertion order and not by
    # the mechanism: it was green for [0.8, 0.1] and RED for [0.1, 0.8] with nothing about
    # `_CompetenceDynamics` changed (measured, both orders). A criterion whose colour is set by a
    # tie-break is replaced, not kept. The form below asserts the tie itself, is evaluated in BOTH
    # state orders so no tie-break can satisfy it, and reddens on any gain != 0.
    flat = _CompetenceDynamics(gain=0.0)  # degenerate forward model: every candidate gains nothing
    flat_gains = {}
    for flat_state in ([0.8, 0.1], [0.1, 0.8]):
        flat_base = _goal_value(tuple(flat_state))
        flat_gains[tuple(flat_state)] = {
            d: _goal_value(flat.predict(tuple(flat_state), d)) - flat_base for d in (0, 1)
        }
    assert all(set(g.values()) == {0.0} for g in flat_gains.values()), (
        "a zero-gain forward model must leave the weakest-first rule with NO signal to rank on -- "
        f"every candidate gain must be exactly 0.0 in either state order (gains {flat_gains})"
    )
    print(
        f"    boundary: competence>1 gains {oob_gains} (FLIPS the weakest-first rule); "
        f"gain=0 gains {flat_gains} (VOIDS it -- an exact tie in either state order)"
    )

    # P1 discharge for _CompetenceDynamics.gain: the planned digit is INVARIANT to it; only the
    # spiking reaction time moves. Re-measured here rather than asserted from the docstring.
    gain_sweep = {}
    for g in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):  # exactly the set the docstring names
        epg = ExecutivePlanner(seed=0, gain=g)
        gain_sweep[g] = (epg.plan_objective(competence, candidates), round(epg.last_rt, 2))
    print(f"    gain sweep (digit, mean_rt): {gain_sweep}")
    assert len({d for d, _ in gain_sweep.values()}) == 1, (
        f"the planned digit must be invariant to the forward model's gain, got {gain_sweep}"
    )
    assert len({rt for _, rt in gain_sweep.values()}) > 1, (
        f"control: gain must still REACH the rollout (distinct reaction times), got {gain_sweep}"
    )

    # determinism: same seed reproduces the same plan -- AND the seed is actually WIRED into the
    # spiking race (otherwise 'deterministic' would be trivially true because the seed does nothing).
    assert ExecutivePlanner(seed=0).plan_objective(competence, candidates) == chosen, (
        "same seed must reproduce the planned objective"
    )
    rts = set()
    for s in range(8):
        eps = ExecutivePlanner(seed=s)
        eps.plan_objective(competence, candidates)
        rts.add(eps.last_rt)
    assert len(rts) > 1, (
        f"control: the seed must reach the spiking race (distinct mean reaction times), got {rts}"
    )

    print("=" * 74)
    print(
        f"self-check OK: dopamine lr {lr_pos:.3f}(+) > {lr_neg:.3f}(-), saturating on "
        f"[{DopamineModulator.LR_MIN}, {DopamineModulator.LR_MAX}]; workspace ignites at its own "
        f"rheobase {thr} and not under it;"
    )
    print(
        f"  prefrontal executive plans the weakest-but-learnable digit {chosen} (control: "
        f"{tied_miss}/{n_seeds} misses on a near-tied landscape); deterministic, seed wired"
    )


if __name__ == "__main__":
    main()
