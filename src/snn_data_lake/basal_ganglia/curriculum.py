"""The goal curriculum (specs/022 T076, plan Phase 1 W1): propose()/observe() over the arm's
own coverage lattice, with an |LP| bandit ADOPTED rather than rebuilt.

WHY THIS EXISTS. The arm explores under `coverage_reward`, which pays for reaching a workspace
cell it has not reached. That objective is dense and honest but undirected: nothing chooses WHICH
unreached region to go after next. T076 asks for the chooser -- a curriculum that proposes a goal
region, observes how that went, and lets learning progress decide where to look next.

WHAT IS ADOPTED, PORTED, AND DELIBERATELY NOT BUILT. P2 says exactly one mechanism per job
survives, so this module writes as little as it can:

  * THE BANDIT IS ADOPTED, NOT WRITTEN. `neuromodulation/intrinsic_motivation.CuriosityDrive`
    already IS an |LP| bandit over regions. T076's wording ("|LP| bandit over goal-space regions")
    reads as a licence to write one; writing one would have been a third implementation. This
    module holds a CuriosityDrive and calls it.
    TWO CORRECTIONS THE PANEL FORCED HERE, both of which had made the adoption look better
    supported than it was. (1) This paragraph said `most_curious` ranks over
    `learning_progress_abs`. IT DOES NOT. `most_curious` ranks on `curiosity_value`, which is the
    RECTIFIED `learning_progress` (max(0, slow - fast)) scaled by novelty plus a novelty bonus.
    `learning_progress_abs` is read only by `record_lp` and `snr()` -- the FLOOR, never the
    ranking. That distinction is not cosmetic: the premise probe originally measured its
    separation on `learning_progress_abs`, i.e. on a quantity the bandit never consults, and it
    had to be re-measured on `curiosity_value` (see the evidence file's corrected section). "A
    metric name is not what it measures" is a lesson this tree has paid for before. (2) The claim
    "with 12+ live call sites" was quoted, never re-derived, and is wrong: `learning_progress_abs`
    has FOUR occurrences in tracked source, of which exactly ONE is an external production call
    (`basal_ganglia/learning_progress.py`). `most_curious` does have many callers (see the
    distinct-from block below), but that is a different symbol and the sentence conflated them.
    The P2 argument for adopting rather than writing stands on the mechanism being present at
    all; it never needed a count, so the count is dropped rather than repaired.
  * THE RATE DRIVE, NOT THE SPIKING ONE. `cortex/autotelic_goals.py` -- the module W1 was told to
    port from -- builds on `SpikingCuriosityDrive`. `basal_ganglia/learning_progress.py` records
    that its own four-regime check measured that drive INVALID as an |LP| basis (its pools are
    calibrated at one reference error, so on a flat stream the rates do not converge) and adopted
    the rate `CuriosityDrive` instead. Copying the port source's kernel choice would have
    re-introduced a defect this tree already dispositioned.
  * THE BUFFER IS IMPORTED, NOT COPIED. `AchievedOutcomeBuffer` is generic over hashable states,
    and a coverage cell is a hashable integer triple, so the port is an import. Copying it would
    have been the additive failure P2 names.
  * HER IS NOT BUILT, AND THAT IS A P3 DECISION RATHER THAN AN OMISSION. `autotelic_goals`'
    hindsight relabelling produces (state, goal, action) labels for a goal-conditioned readout.
    No such readout consumes relabelled arm transitions today, so building the relabeller here
    would produce labels nothing reads -- dormant on arrival (P3 shape 2, "return value
    discarded"). It lands with the consumer or not at all.

WHAT THE PREMISE PROBE ESTABLISHED, AND HOW MUCH WEAKER IT IS THAN FIRST REPORTED
(evidence/w1-goal-curriculum.md carries the full corrected account). The mechanism only works if
|LP| SEPARATES regions at the scale the live arm produces. Measured over 8 seeds x 3000 ticks, on
`curiosity_value` -- the quantity `most_curious` actually ranks on -- restricted to cells with at
least 5 real observations, and against a genuine 200-draw marginal-preserving label shuffle:
F 0.16-2.13, clearing its own null on 6 of 8 seeds against a bar of 6 of 8 that was fixed before
the run. It passes by nothing, and seeds 5 and 7 do not clear their null at all.

THE FIRST WRITE-UP OF THIS PARAGRAPH CLAIMED 8 OF 8 AT F 9.7-22.7, and that is kept here as a
warning rather than deleted. Three errors inflated it ~15x, all found by an adversarial panel:
the probe sampled EVERY cell every 25th wake row whether or not it had been visited, and a
once-visited cell's |LP| is identically 0 forever by the drive's initialization -- so 74-90% of the
series entering the ratio were CONSTANT and deflated its denominator; the statistic was computed on
`learning_progress_abs`, which the bandit never reads; and the null was reported as 200 draws when
the harness ran 20. Sampling on visit only collapses F from ~15 to ~1.1. The transferable half:
a zero-variance series is usually not a measurement of zero, it is the ABSENCE of a measurement
wearing one, and it lands in the denominator.

THE GOAL SPACE IS THE OBJECTIVE'S OWN LATTICE, WHICH IS WHAT KEEPS IT HONEST. A goal here is an
integer cell triple from `coverage_reward`'s own grid, not an invented target -- that module's
docstring refuses a fabricated goal as "the same class of theatre as a panel with no signal
behind it". Integer cells also make the buffer's exact-equality dedup work; on continuous
end-effector floats it silently fails (measured: three near-identical floats dedup to three).

FR-027's JOIN: the objective is CARRIED, never replaced. `observe()` is fed `coverage_reward`'s
own return value and derives its error from it, so the |LP| this bandit reads is learning progress
ON the incumbent objective rather than a second objective standing beside it.

THE EXPLORATION FLOOR IS DERIVED, NOT SET (P1). T076: "the exploration floor DERIVED from measured
|LP| signal-to-noise on the goal store (20% is the literature reference, never the value)." The
floor here is `1 / (1 + snr)` where snr is the between-region over within-region variance of the
|LP| histories the goal store is already keeping -- so it is measured on the goal store, it is
dimensionless, and it moves: no separation (snr 0) explores everything, strong separation exploits.
A constant would have been the P1 violation this clause exists to prevent.
"""

# distinct-from: cortex/autotelic_goals.py -- the PORT SOURCE. Its sampler draws UNIFORMLY from the
#   achieved-outcome buffer and its competence-progress lives only in `sagg_riac_practice`, an
#   OFFLINE replay harness over pre-measured curves that never runs an environment. This is the
#   online chooser that module does not contain; its buffer is imported, not reimplemented.
# distinct-from: neuromodulation/intrinsic_motivation.py::CuriosityDrive -- the |LP| bandit this
#   module ADOPTS and calls. The bandit ranks regions; this decides what a region IS on this body,
#   feeds it the objective's error, and persists the result.
# distinct-from: basal_ganglia/learning_progress.py::LearningProgress -- the per-REGION |LP| facade
#   over the same drive, keyed on anatomical regions and fed percept surprise. This keys on
#   workspace cells and is fed the coverage objective. Same drive, different partition and input.
# distinct-from: neuromodulation/spiking_curiosity.py::SpikingCurriculum -- an |LP|-driven
#   STAGE-ADVANCE gate (scalar in, bool out: has the learner earned the next stage). This selects
#   WHICH region to pursue. Different question, no shared output.
# distinct-from: cortex/reasoning_curriculum.py -- a transitive-inference ENVIRONMENT plus its
#   learner, scored by held-out accuracy over symbolic orderings. No common benchmark exists that
#   could compare them, which P2 requires before either could retire the other. All sites read.
# distinct-from: cortex/quality_diversity.py::MAPElites._select_parent -- THE CLOSEST INCUMBENT IN
#   THE TREE, and this list did not name it until a panel found it. It is the same three lines this
#   module runs: `most_curious` over sorted integer LATTICE CELLS, then `observe` on that cell. The
#   distinction is real but it is a distinction of DOMAIN, not of mechanism: its cells are a
#   BEHAVIOUR-descriptor lattice over evolved policy parameters, its error term is
#   `_unfilled_neighbor_frac` (an archive-coverage gradient computed from the archive itself), and
#   it selects a PARENT TO MUTATE inside an offline evolutionary loop with no environment tick.
#   This one's cells are the live arm's workspace, its error is the incumbent objective's own
#   return, and it selects a GOAL for a running body. Retiring either for the other would require a
#   benchmark on which both can be scored, and none exists -- so P2 is satisfied by adjudication,
#   not by a merge. Recorded plainly because the search that missed it searched by NAME
#   ("curriculum", "goal") when the thing that identifies an incumbent is the shared CALL.
# distinct-from: basal_ganglia/brain_decision.py -- same package, and it answers a question that
#   sounds identical ("which objective next, and explore or exploit?"). Three separations, each
#   load-bearing: it selects among ~10 DIGIT CLASSES in the developmental loop, not workspace
#   regions; it ranks on a FUSION of latent-Q value, curiosity and a metacognitive controller
#   rather than on |LP| alone; and it builds on `SpikingCuriosityDrive`, the kernel
#   `basal_ganglia/learning_progress.py` recorded as INVALID as an |LP| basis. It is the richer
#   selector for a discrete class set; this is the |LP| chooser for a continuous body's lattice.
# distinct-from: cortex/nightly_skills.py::_pick and hippocampus/skill_builder.py -- both are
#   `most_curious(sorted(names))` over SKILL/TASK NAME sets during sleep. Same primitive, symbolic
#   candidate space, no body and no objective join. Sites read.
# distinct-from: brainstem/autonomous_loop.AutonomousArm.develop(stages, curriculum, ...) -- the
#   arm's OWN self-paced stage loop. It advances through a caller-supplied STAGE LIST on a
#   competence criterion; it does not choose a region and has no goal space. The two compose (a
#   stage could bound which cells are candidates); neither replaces the other.

from __future__ import annotations

import math
import pathlib
import sys

#: A cell needs this many recorded |LP| samples before it can contribute to the SNR estimate. At
#: two, the within-region variance of a constant pair is exactly 0 and the ratio reports infinity
#: -- an artifact of sampling, not a separation. Measured in the premise probe, where exactly that
#: happened at 300 ticks.
#: P1 DISPOSITION, corrected: this was filed as "a sample-size floor, not a behaviour threshold
#: (P1 case (a))". Case (a) is a fixed point on a NORMALIZED OUTPUT, which this is not -- it
#: decides which regions enter snr(), hence floor(), hence explore-vs-exploit, so it does arbitrate
#: behaviour. The honest filing is that it is a STATISTICAL ADMISSIBILITY floor: below three
#: samples the estimator it guards is undefined rather than small, and no measurement can set it
#: because the quantity does not exist there. It is defensible on that ground and on no other; if a
#: future estimator is well-defined at two, this constant goes.
MIN_LP_SAMPLES = 3

#: The achievement function every goal row created here points back to -- the incumbent
#: objective, named once so observe() and record_outcome() cannot disagree about it.
_ACHIEVEMENT_REF = "brainstem.autonomous_loop.AutonomousArm.coverage_reward"


def snr_from_histories(histories):
    """Between-region variance of mean |LP| over the mean within-region variance.

    The signal-to-noise T076 names, computed on the goal store's own `lp_history` lists. Returns
    None when it is undefined (fewer than two regions with enough samples) rather than a number
    that looks like an answer -- the caller must decide what to do with "not yet measurable", and
    here that decision is to explore.

    WRITTEN IN PLAIN FLOAT ARITHMETIC RATHER THAN `statistics`, AND THAT IS A MEASURED CHOICE, not
    a style preference. `statistics.pvariance`/`fmean` compute in exact rational arithmetic, which
    is the right default for a library and the wrong one here: at the measured production scale
    (743 distinct cells, the seed-7 figure from the premise probe) this function cost 17.9 ms per
    call and `propose()` calls it EVERY time through `floor()`. Wired at one proposal per tick that
    is ~26 s against a ~13 s tick budget -- it would have doubled the tick cost, which is the very
    rule the self-check block below cites (an observer must not become the load). The float form is
    0.65 ms, a 27x reduction, and the two agree to 4.3e-19 absolute here with ZERO mismatches over
    300 randomized shapes spanning empty / single-region / sub-threshold inputs and 1e-9 to 1e6
    scales. The remaining last-bit difference is why W-R3 was re-run against cached streams after
    this change rather than assumed unaffected.

    THE WITHIN-REGION TERM IS AN UNWEIGHTED MEAN OF PER-REGION VARIANCES, which is deliberate and
    is NOT the pooled within-group variance an ANOVA-style F would use: a region sitting at exactly
    MIN_LP_SAMPLES counts as much as one with thousands of samples. That biases the estimate toward
    small-sample regions. It is kept because this quantity's only consumer is an exploration FLOOR
    -- a heuristic that wants "do the regions look separated at all", where treating a barely
    sampled region as a full vote is the conservative direction (it inflates within-variance, which
    LOWERS the SNR and RAISES exploration). Changing it is a live option, but it would move every
    number the premise probe and W-R3 already reported, so it is disclosed rather than swapped
    mid-arc.
    """
    means, withins = [], []
    for h in histories:
        n = len(h)
        if n < MIN_LP_SAMPLES:
            continue
        m = math.fsum(h) / n
        means.append(m)
        withins.append(math.fsum((x - m) ** 2 for x in h) / n)
    if len(means) < 2:
        return None
    k = len(means)
    gm = math.fsum(means) / k
    between = math.fsum((m - gm) ** 2 for m in means) / k
    within = math.fsum(withins) / k
    if within == 0.0:
        return math.inf if between > 0.0 else 0.0
    return between / within


def floor_from_snr(snr):
    """The exploration floor, DERIVED (P1). 1/(1+snr): no measurable separation explores
    everything, strong separation exploits. Dimensionless, monotone, and bounded in (0, 1].

    `None` (not yet measurable) returns 1.0 -- explore -- rather than a default constant. That is
    the honest reading of an unmeasured SNR, and it makes the cold-start behaviour a consequence of
    the derivation instead of a second hidden parameter.
    """
    if snr is None:
        return 1.0
    if snr == math.inf:
        return 0.0
    # TOTALITY, and the NaN case is the one that matters. This is a public function, and
    # `snr_from_histories` provably cannot emit a negative or a NaN from finite input (both terms
    # are sums of squares), so nothing in this module reaches these branches today -- they exist
    # because a future caller deriving an SNR some other way would otherwise get silence instead of
    # an error. NaN is the dangerous one: `rng.random() < nan` is False, so a NaN floor would make
    # propose() take the exploit branch FOREVER with no exception anywhere -- a mechanism silently
    # pinned by a value that never announces itself. -1.0 would divide by zero, and any snr < 0
    # returns a "floor" above 1 or below 0, violating this docstring's own bound.
    if not (snr == snr) or snr < 0.0:  # NaN is the only value unequal to itself
        raise ValueError(f"an SNR must be non-negative and not NaN, got {snr!r}")
    return 1.0 / (1.0 + snr)


class GoalCurriculum:
    """propose() a goal region, observe() how it went. The bandit and the buffer are borrowed.

    Not thread-safe, and deliberately owns no lake handle of its own: a GoalStore is passed IN,
    because that store is single-writer by contract and two concurrently-open views on one lake
    share a BinLog and have previously collided on goal ids.
    """

    def __init__(self, seed=0, store=None):
        import random

        from snn_data_lake.cortex.autotelic_goals import AchievedOutcomeBuffer
        from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive

        self._drive = CuriosityDrive(seed=seed)  # ADOPTED: the |LP| bandit, not a new one
        # PORTED by import. Its maxlen is left at its own default ON PURPOSE and is NOT exposed
        # here: all_seen() is the unbounded reached-set, while maxlen bounds only the `_seen`
        # recency deque this module never reads. Measured -- maxlen=8 then 200 observations gives
        # len(seen()) == 200. A `maxlen=` parameter on this class would therefore be a knob that
        # changes nothing a caller can observe, and worse, a false affordance: someone bounding the
        # candidate pool with it would be wrong by the whole ratio.
        self._buf = AchievedOutcomeBuffer()
        self._rng = random.Random(seed)  # per-instance, never module-level (determinism contract)
        self._store = store
        self._gid = {}  # cell -> goal_id, so a region maps to one store row for its whole life
        self._proposed = None
        self.n_observed = 0
        self.n_proposed = 0

    # ---- the interface T076 names -------------------------------------------------------

    def observe(self, cell, reward):
        """Record what the objective paid for being in `cell`.

        `reward` is `coverage_reward`'s own return in [-1, 1]; the error the drive learns on is
        (1 - reward)/2 in [0, 1]. That is FR-027's JOIN in one line: the incumbent objective is
        carried and the |LP| is computed ON it, rather than a second objective being invented
        beside it.
        """
        # THE OBJECTIVE'S RANGE IS CHECKED, matching the precedent set by the sibling facade over
        # this same drive (basal_ganglia/learning_progress.LearningProgress.observe validates its
        # error in [0, 1] and raises). Without it one out-of-contract reward silently poisons the
        # EMA pair for a long time -- measured, observe(cell, 500.0) leaves fast == slow == -249.5,
        # and at alpha 0.2 / 0.04 that takes hundreds of samples to wash out. An objective outside
        # coverage_reward's own stated [-1, 1] contract is a caller bug, and a caller bug that
        # degrades a statistic quietly is the worst shape available.
        r = float(reward)
        if not (-1.0 <= r <= 1.0):
            raise ValueError(
                f"reward must be in coverage_reward's stated [-1, 1] contract, got {r!r}"
            )
        # THE CELL'S SHAPE IS CHECKED HERE rather than where it breaks. The store branch below
        # indexes cell[0..2] and the buffer/sorted() path does not, so a malformed cell either
        # raised IndexError deep in the store branch, or -- worse -- was accepted storeless and
        # then blew up later inside sorted() in seen(), at a call site with nothing pointing back
        # to the bad observe(). Measured, all three: (5,) -> IndexError; ("x",) -> ValueError from
        # float(); a (0,0,0) and an ("x",) in one instance -> TypeError inside seen().
        if not (isinstance(cell, tuple) and len(cell) == 3):
            raise ValueError(f"a goal region is a 3-tuple workspace cell, got {cell!r}")
        err = (1.0 - r) / 2.0
        self._drive.observe(cell, err)
        self._buf.add(cell)
        self.n_observed += 1
        if self._store is not None:
            gid = self._gid.get(cell)
            if gid is None:
                # A cell at the origin is a legal region but a zero-norm embedding, which the
                # store refuses; the trailing 1.0 makes every embedding non-degenerate.
                # DISCLOSED CONSEQUENCE, since these rows are shared lake state: the constant pad
                # is a fixed 4th coordinate, so under GoalStore.query()'s cosine it inflates the
                # similarity of cells NEAR THE ORIGIN (where the pad dominates the norm) relative
                # to distant ones. Nothing queries these rows today and this module never calls
                # query(), so there is no live effect -- but a future consumer would inherit the
                # bias silently, and it is cheaper to say so here than to rediscover it there.
                gid = self._store.create(self._embed(cell), _ACHIEVEMENT_REF)
                self._gid[cell] = gid
            self._store.record_lp(gid, self._drive.learning_progress_abs(cell))

    def propose(self):
        """Return the goal region to pursue next, or None before anything has been observed.

        Exploit the |LP|-most-curious region, or explore uniformly with probability `floor()`.
        Both branches draw only from regions the arm has ACTUALLY REACHED -- a goal is never
        fabricated, which is the constraint `coverage_reward`'s docstring imposes.
        """
        seen = self.seen()
        if not seen:
            return None
        self.n_proposed += 1
        if self._rng.random() < self.floor():
            self._proposed = self._rng.choice(seen)
        else:
            self._proposed = self._drive.most_curious(seen)
        return self._proposed

    def record_outcome(self, cell, outcome):
        """Note how a proposal turned out, creating the region's store row if it has none.

        Returns True if it was recorded, False if there is no store.

        THIS METHOD WAS WRITTEN TWICE WRONG BEFORE THIS, in opposite directions, and both failures
        were about the SAME case: a goal that was proposed and NEVER REACHED.
          * First it did `if cell in self._gid` and silently no-oped otherwise -- discarding
            exactly the informative outcome, because `_gid` is populated only by observe()'s store
            branch and an unreached cell was never observed.
          * Then it RAISED on that case, which a re-panel pointed out is no better: the natural
            caller loop is propose() -> act -> record_outcome(cell, reached-or-not), so raising
            hard-blocks every failure branch. The docstring's own escape hatch ("observe() it")
            was worse still -- it asks the caller to fabricate an observation for a cell the arm
            was never in, contradicting observe()'s contract AND this module's own "a goal is
            never fabricated" invariant.
        Creating the row on demand is what lets the method do its job. It mirrors observe()'s own
        lazy create, and a row with an empty lp_history is harmless to snr(): MIN_LP_SAMPLES
        filters it out, so an unreached goal contributes an OUTCOME without contributing a
        measurement it never made.
        """
        if self._store is None:
            return False
        gid = self._gid.get(cell)
        if gid is None:
            gid = self._gid[cell] = self._store.create(self._embed(cell), _ACHIEVEMENT_REF)
        self._store.record_outcome(gid, outcome)
        return True

    # ---- the derived quantities ----------------------------------------------------------

    def _embed(self, cell):
        """The store embedding for a region. ONE definition, because observe() and
        record_outcome() both create rows and a drift between them would put two different
        geometries in one shared record class."""
        return [float(cell[0]), float(cell[1]), float(cell[2]), 1.0]

    def seen(self):
        """The regions actually reached, sorted for determinism (a set's order is not stable
        across runs, and a bandit fed an unstable candidate order is not reproducible).

        Calls the buffer's own `all_seen()`. An earlier draft reached into its private `_set`
        because no public enumerator existed (`sample()` and `coverage()` both REQUIRE an `allowed`
        pool, and there is no allowed pool here -- the reached cells ARE the candidate set), and
        left a comment saying an accessor was "the better fix the day a second caller needs one".
        The panel's answer was that this module IS that day, so the accessor was added rather than
        the reach documented. The set is UNBOUNDED by the buffer's construction and that is wanted
        here: the premise probe measured 211 to 743 distinct cells per seed, so a candidate list
        that silently stopped growing would quietly narrow the bandit's choices.
        """
        return sorted(self._buf.all_seen())

    def snr(self):
        """|LP| signal-to-noise measured ON THE GOAL STORE, per T076."""
        if self._store is None:
            return None
        return snr_from_histories([g["lp_history"] for g in self._store.goals().values()])

    def floor(self):
        """The exploration floor, derived from snr(). Never a constant (P1)."""
        return floor_from_snr(self.snr())

    def state(self):
        return {
            # WITHOUT A STORE THE BANDIT NEVER FIRES, and that has to be visible rather than
            # inferred. T076 measures the SNR on the goal store, so a storeless curriculum has no
            # measurable SNR, floor() is 1.0, and propose() takes the explore branch every time --
            # a uniform sampler wearing a curriculum's name. W-R3 measured exactly that: the
            # proposal was invariant to the objective on 8 of 8 seeds because the run had no store.
            "bandit_active": self._store is not None,
            "regions": len(self.seen()),
            "observed": self.n_observed,
            "proposed": self.n_proposed,
            "snr": self.snr(),
            "floor": self.floor(),
            "last_goal": self._proposed,
        }


# ---------------------------------------------------------------------------------------
# Self-check. The BARE path (what run_selfchecks runs) constructs NO brain and touches no
# lake: it drives the derived quantities on synthetic regions, each criterion beside a
# constructed red. perf_baseline's rule -- anything that observes the brain must be cheap
# enough not to become the load.
# ---------------------------------------------------------------------------------------


def _emit(line):
    assert line.isascii(), "ASCII-only output"
    print(line)


def _wiring_check():
    # 1 RED -- the SNR must refuse to answer when it cannot. Fewer than two regions with enough
    #   samples is "not measurable", and a number there would be invented.
    assert snr_from_histories([]) is None, "no regions is not an SNR"
    assert snr_from_histories([[0.1, 0.2, 0.3]]) is None, "one region is not a separation"
    assert snr_from_histories([[0.1, 0.2], [0.5, 0.6]]) is None, (
        f"{MIN_LP_SAMPLES} samples are required per region; at 2 a constant pair has zero "
        "within-variance and the ratio reports infinity, which is sampling, not separation"
    )
    # 2 GREEN/RED PAIR -- separated regions must score above unseparated ones, in one command.
    sep = snr_from_histories([[0.10, 0.11, 0.12], [0.90, 0.91, 0.92]])
    flat = snr_from_histories([[0.50, 0.51, 0.52], [0.50, 0.51, 0.52]])
    assert sep is not None and flat is not None, "both arms must be measurable"
    assert sep > flat, f"separated regions must out-score identical ones ({sep} vs {flat})"
    assert flat == 0.0, f"identical regions have zero between-variance ({flat})"
    # The MAGNITUDE, not just the ordering. Without this the ratio can be INVERTED
    # (within/between) and still pass the two asserts above, because the flat arm is 0.0 under
    # both spellings -- a mutation run found exactly that survivor. Widely separated regions must
    # score ABOVE 1: between-variance exceeds within-variance, which is what "signal to noise"
    # means and what the floor's derivation assumes.
    assert sep > 1.0, (
        f"separation must exceed unity for clearly separated regions ({sep}) -- at or below 1 the "
        "ratio is inverted or the statistic is not signal-over-noise, and floor_from_snr would "
        "read exploitation as exploration"
    )
    # 3 -- the floor is DERIVED and monotone, and its endpoints are the honest ones (P1).
    assert floor_from_snr(None) == 1.0, "an unmeasurable SNR explores, it does not pick a constant"
    assert floor_from_snr(0.0) == 1.0, "no separation explores everything"
    assert floor_from_snr(math.inf) == 0.0, "perfect separation exploits"
    lo, hi = floor_from_snr(1.0), floor_from_snr(9.0)
    assert lo > hi, f"the floor must FALL as separation rises ({lo} then {hi})"
    assert abs(lo - 0.5) < 1e-12 and abs(hi - 0.1) < 1e-12, (
        f"1/(1+snr) is the stated derivation ({lo}, {hi}) -- a different curve here would make the "
        "docstring's claim false"
    )
    # 4 RED -- propose() must never invent a region. Before anything is observed there is no goal.
    c = GoalCurriculum(seed=0)
    assert c.propose() is None, "a curriculum that has observed nothing cannot propose a goal"
    assert c.seen() == [], "nothing observed, nothing seen"
    # 5 -- observed regions are the only candidates, and the objective's sign is carried through.
    for cell, r in (((0, 0, 0), 1.0), ((1, 0, 0), -1.0), ((2, 0, 0), 0.0)):
        c.observe(cell, r)
    assert c.seen() == [(0, 0, 0), (1, 0, 0), (2, 0, 0)], (
        f"seen must be the reached cells ({c.seen()})"
    )
    for _ in range(20):
        assert c.propose() in c.seen(), "every proposal must be a region the arm actually reached"
    # 6 RED -- the objective's value must REACH the drive. Constructed carefully, because the
    #   obvious version is vacuous: |LP| is |slow - fast|, which is ZERO on any flat stream
    #   whatever its level, so feeding +1 forever and -1 forever gives 0.0 == 0.0 and would pass
    #   a discarding observe(). The arms must therefore differ in CHANGE, not in level.
    ramp, flat = GoalCurriculum(seed=1), GoalCurriculum(seed=1)
    probe = (7, 7, 7)  # a real 3-cell: observe() validates the shape, and this criterion must
    #   exercise the SAME contract every caller does rather than a looser one.
    for i in range(30):
        ramp.observe(probe, -1.0 + 2.0 * i / 29.0)  # reward rises, so the error falls
        flat.observe(probe, 0.0)  # nothing changes
    lp_ramp = ramp._drive.learning_progress_abs(probe)
    lp_flat = flat._drive.learning_progress_abs(probe)
    assert lp_flat == 0.0, f"a flat objective has no learning progress ({lp_flat})"
    assert lp_ramp > lp_flat, (
        f"a CHANGING objective must produce |LP| where a flat one does not ({lp_ramp} vs "
        f"{lp_flat}) -- equal here means observe() is discarding the reward and the JOIN is "
        "decorative"
    )
    # 7 -- no store means no persistence attempt and no crash; but it ALSO means the bandit is
    #   unreachable, and that must be stated rather than discovered. W-R3 was first measured on a
    #   storeless curriculum and came back invariant to the objective on 8 of 8 seeds -- not
    #   because the mechanism was inert, but because floor() is 1.0 without an SNR to measure, so
    #   propose() explored every time and never consulted the drive. The measurement was right and
    #   the configuration was wrong; this criterion is what makes that legible at the call site.
    assert c.snr() is None and c.floor() == 1.0, "without a store there is no measured SNR"
    assert c.state()["bandit_active"] is False, (
        "a storeless curriculum must SAY its bandit cannot fire -- silently degrading to a uniform "
        "sampler is the 'constant wearing a signal's clothes' shape P3 names"
    )
    assert c.state()["regions"] == 3, "state must report what it has"
    # 8 RED -- THE ARMED CONFIGURATION, which every criterion above misses. 1-7 all run storeless,
    #   and storeless is precisely the mode where the bandit CANNOT fire: a suite that only ever
    #   exercises the degraded path is a control sharing a stage with the defect. This is the
    #   configuration callers use, and it is the one W-R3 measures. Kept cheap and lake-free by
    #   pointing the store at a temp file -- no brain, and never the live lake.
    import tempfile

    from snn_data_lake.basal_ganglia.learning_progress import GoalStore
    from snn_data_lake.substrate import home as _home

    # home._log memoizes BinLogs by (path, dedup), so a store opened in a temp dir leaves a cache
    # entry pointing at a directory TemporaryDirectory is about to delete. TWO CORRECTIONS a
    # re-panel forced here, and the first is the one that mattered:
    #   (1) the cleanup was a bare statement after the assertions, so ANY failing criterion skipped
    #       it -- reproduced by breaking criterion 9, which left exactly one cache entry whose path
    #       no longer existed. A red self-check is when the process most needs to stay clean for
    #       whatever runs next, so the cleanup belongs in a finally.
    #   (2) it was `_LOG_CACHE.clear()`, copied from learning_progress's check, which wipes the
    #       WHOLE shared cache. run_selfchecks runs every module in ONE process, so that silently
    #       evicts other modules' still-valid warm BinLogs and forces a reconstruction home.py's
    #       own docstring measures in seconds on a realistic lake. Only this block's keys are
    #       dropped now.
    tmp_keys = []
    try:
        with tempfile.TemporaryDirectory(prefix="curriculum_selfcheck_") as d:
            goal_path = str(pathlib.Path(d) / "goals.bin")
            tmp_keys = [(goal_path, True), (goal_path, False)]
            armed = GoalCurriculum(seed=2, store=GoalStore(path=goal_path))
            assert armed.state()["bandit_active"] is True, (
                "a curriculum WITH a store has a live bandit"
            )
            # two regions whose |LP| genuinely differs: one region's objective RAMPS, the other is flat.
            for i in range(12):
                armed.observe((0, 0, 0), -1.0 + 2.0 * i / 11.0)
                armed.observe((1, 0, 0), 0.0)
            snr = armed.snr()
            assert snr is not None, "with a store and enough samples the SNR must be MEASURABLE"
            assert armed.floor() < 1.0, (
                f"a measured separation must pull the floor OFF its explore-everything endpoint "
                f"({armed.floor()} at snr {snr}) -- a floor pinned at 1.0 means propose() never "
                "consults the bandit, which is the storeless behaviour wearing a store"
            )
            # and the bandit must actually be REACHED, and reached CORRECTLY. The first draft
            # asserted `most_curious(seen) in seen`, which the panel correctly called VACUOUS: it is a
            # max() over a non-empty list, so it holds whatever the ranking does. The criterion that
            # is not vacuous names WHICH region must win -- the ramping one, whose objective is still
            # changing, over the flat one, whose |LP| is zero by construction.
            armed._rng.seed(7)
            picks = [armed.propose() for _ in range(200)]
            assert set(picks) <= set(armed.seen()), "proposals stay inside the reached regions"
            ramping, still = (0, 0, 0), (1, 0, 0)
            lp_r = armed._drive.learning_progress_abs(ramping)
            lp_s = armed._drive.learning_progress_abs(still)
            assert lp_r > lp_s, f"the ramping region must carry more |LP| ({lp_r} vs {lp_s})"
            assert armed._drive.most_curious([ramping, still]) == ramping, (
                "the bandit must PREFER the region with higher |LP| -- 'the answer is one of the "
                "candidates' is true of any max() and proves nothing about the ranking"
            )
            # 9 RED -- record_outcome's contract, which nothing else here covered and which this
            #   criterion has now caught TWICE as it changed. The case that matters is a goal that
            #   was proposed and NEVER REACHED: the first version dropped it silently, the second
            #   raised on it (which hard-blocks the natural propose -> act -> record loop on every
            #   failure branch), and the third creates the row. The red is that the unreached
            #   outcome must LAND SOMEWHERE, because a curriculum that can only record its
            #   successes is learning from a filtered sample.
            assert armed.record_outcome(ramping, "reached") is True, "an observed region records"
            before = len(armed._store.goals())
            snr_before = armed.snr()
            assert armed.record_outcome((41, 41, 41), "proposed, never reached") is True, (
                "an UNREACHED goal's outcome must be recordable -- it is the informative case, and "
                "neither dropping it nor raising on it records anything"
            )
            assert len(armed._store.goals()) == before + 1, (
                "the unreached goal must get its own row, or the outcome landed nowhere"
            )
            # and creating that row must not disturb the measurement: its lp_history is empty, so
            # MIN_LP_SAMPLES filters it out of the SNR entirely.
            assert armed.snr() == snr_before, (
                f"a row with no |LP| samples must not move the SNR ({armed.snr()} vs {snr_before}) "
                "-- an unreached goal contributes an OUTCOME, never a measurement it never made"
            )
            assert GoalCurriculum(seed=0).record_outcome((0, 0, 0), "x") is False, (
                "with no store there is nowhere to record, and that is reported, not raised"
            )
    finally:
        for k in tmp_keys:
            _home._LOG_CACHE.pop(k, None)
    _emit(
        "curriculum wiring OK: 9 criteria proven falsifiable (SNR refuses when unmeasurable, "
        "separated out-scores flat, the floor is derived and monotone, no goal before an "
        "observation, proposals only from reached regions, the objective reaches the drive, "
        "a storeless curriculum says its bandit is inactive, an ARMED one measures an SNR, pulls "
        "its floor off the explore endpoint, PREFERS the higher-|LP| region, and RECORDS the outcome "
        "of a goal it never reached without letting that row disturb the SNR) -- no brain, and "
        "never the live lake"
    )


def main(argv=None):
    _wiring_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
