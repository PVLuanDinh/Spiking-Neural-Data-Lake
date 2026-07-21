r"""
Self-paced developmental curriculum -- STARTING SMALL, then graduating on the agent's OWN
mastery. An ordered sequence of stages of INCREASING difficulty; the agent AUTONOMOUSLY
advances to the next stage when it has MASTERED the current one, where mastery is read off the
agent's OWN LEARNING PROGRESS (it graduates itself when its learning-progress dries up), NOT on
a fixed clock. This is the engine of cumulative learning: master easy -> unlock harder -> ...
always training at the FRONTIER of competence.

Honest framing (read this first). This is ONE primitive toward cumulative learning -- a small,
deterministic self-pacing gate -- NOT a claim of general developmental intelligence. It holds no
world model and does not itself learn; it CONSUMES a scalar learning-progress signal (produced
elsewhere: neuromodulation/intrinsic_motivation.py CuriosityDrive.learning_progress) and decides
only WHEN to move to the next stage. It is the scheduler, not the learner.

Why gate on learning progress (and not a clock). Learning progress = the RATE at which a
context's prediction error is falling; it is HIGH while the agent is still improving and ~0 once
the stage is mastered (error low and flat) OR unlearnable (error high and flat). Reading mastery
off THIS signal means the agent spends exactly as long on each stage as IT needs -- a fast learner
graduates early, a slow learner stays longer -- rather than every stage getting a fixed budget. The
curriculum sits at the boundary between "still learning" (keep training) and "learned" (move on),
which is precisely the frontier where curriculum learning helps.

The advance rule (deterministic). Per stage the curriculum keeps a streak of CONSECUTIVE observe()
calls whose learning_progress stayed <= mastery_threshold. A high learning_progress RESETS that
streak (still learning -> do not advance). Once the streak reaches `patience` the stage is judged
LEARNED (sustained low progress, not a one-off dip) and the agent advances to the next, harder
stage; mastered increments and the streak resets. When the last stage is mastered the curriculum
is done and the stage index clamps at the hardest.

Why starting small transfers (the un-riggable self-check). A synthetic staged learner refines a
SHARED competence toward per-stage targets; a hard target starts FAR (out of the learnability
frontier) so a learner thrown straight at it gets ~no gradient and stays stuck, whereas the
self-paced curriculum masters nearby stepping-stone targets that walk the shared competence into
reach of the hard one -- so within the SAME step budget the curriculum reaches far higher final
competence. That is Elman's "importance of starting small" made mechanical.

Refs: Elman, "Learning and development in neural networks: the importance of starting small"
(Cognition, 1993); Bengio, Louradour, Collobert & Weston, "Curriculum Learning" (ICML 2009);
Oudeyer & Kaplan, "What is intrinsic motivation? A typology of computational approaches" (2007)
and developmental-robotics intrinsic-motivation work (self-paced exploration at the frontier).

Deps: random, math, substrate._mathutils.mean (numpy/torch allowed project-wide); ASCII-only prints, deterministic
(per-instance random.Random(seed); the mastery/advance rule is fully deterministic).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.curriculum
"""

import math
import random

from snn_data_lake.substrate._mathutils import mean


class Curriculum:
    """A self-paced developmental curriculum over n_stages of increasing difficulty.

    observe(lp) feeds the CURRENT stage's learning-progress and returns True on the tick it
    ADVANCES to the next stage; advancing requires learning_progress <= mastery_threshold for
    `patience` CONSECUTIVE calls (sustained low -> learned, not a one-off dip), and any high
    learning_progress resets that streak (still learning -> hold). stage is the current index
    (clamped at n_stages-1 when done), mastered counts completed stages, done is all-mastered,
    difficulty() is stage/(n_stages-1) in [0,1]. Deterministic; per-instance random.Random(seed)."""

    def __init__(self, n_stages, mastery_threshold=0.02, patience=8, seed=0):
        self.n_stages = max(1, int(n_stages))  # stages indexed 0..n_stages-1
        self.mastery_threshold = float(
            mastery_threshold
        )  # learning_progress at/below this counts as "not learning"
        self.patience = max(
            1, int(patience)
        )  # consecutive low-LP calls needed to judge a stage LEARNED
        self.rng = random.Random(seed)  # per-instance rng (interleaved-review picks / determinism)
        self._mastered = 0  # number of stages completed (0..n_stages)
        self._low_streak = 0  # consecutive observe() calls with LP <= threshold this stage

    def observe(self, learning_progress):
        """Feed the CURRENT stage's learning-progress for this tick. Returns True on the tick the
        curriculum ADVANCES to the next stage, else False. A high learning_progress RESETS the
        patience streak (still learning -> never advance); `patience` consecutive low readings judge
        the stage LEARNED and advance it. No-op once done (returns False)."""
        if self.done:
            return False
        lp = float(learning_progress)
        if lp <= self.mastery_threshold:
            self._low_streak += 1  # sustained low -> progress toward graduation
        else:
            self._low_streak = 0  # still learning -> reset patience (no premature advance)
        if self._low_streak >= self.patience:
            self._mastered += 1  # this stage is mastered -> advance to the next (harder) one
            self._low_streak = 0
            return True
        return False

    @property
    def stage(self):
        """Current stage index in 0..n_stages-1 (clamped at the hardest once done)."""
        return min(self._mastered, self.n_stages - 1)

    @property
    def mastered(self):
        """Number of stages completed so far (0..n_stages)."""
        return self._mastered

    @property
    def done(self):
        """True once every stage has been mastered."""
        return self._mastered >= self.n_stages

    def difficulty(self):
        """Current-stage difficulty in [0,1] = stage/(n_stages-1); 0.0 for a single-stage curriculum."""
        if self.n_stages <= 1:
            return 0.0
        return self.stage / (self.n_stages - 1)

    def review_stage(self):
        """Return a random already-mastered stage index for interleaved REVIEW (spaced practice
        against forgetting), or None if nothing is mastered yet. Uses the per-instance rng ->
        deterministic given the seed. Optional; the core advance rule does not consume it."""
        if self._mastered == 0:
            return None
        return self.rng.randrange(self._mastered)


# =====================================================================================================
# Self-check: four un-riggable, deterministic controls. Every LP trace below is fixed (or drawn from a
# seeded rng), so the tail numbers reproduce byte-for-byte on every run.
# =====================================================================================================

# A synthetic STAGED learner sharing ONE competence w in [0,1] across increasing-difficulty targets.
_TARGETS = (0.25, 0.50, 0.75, 1.00)  # increasing-difficulty milestones of the SHARED competence w
_REACH = 0.30  # frontier of learnability: a target much farther than this yields ~no gradient
_LR = 0.30  # base learning rate (fraction of the in-reach gap closed per tick)


def _learner_step(w, target):
    """One frontier-gated gradient step of the shared competence w toward `target`. The effective gain
    decays as exp(-(gap/_REACH)**2): ~1 when the target is within reach, ~0 when it is far -- a stylized
    vanishing-gradient / sparse-reward frontier (progress is only possible NEAR current competence).
    Returns the new w. This is why a far ("hard") target cannot be learned cold, but a nearby one can."""
    gap = target - w
    gain = math.exp(-((gap / _REACH) ** 2))
    return w + _LR * gain * gap


def _run_curriculum_learner(budget, seed):
    """Self-paced curriculum learner: train the SHARED competence on the CURRENT stage's target, and
    advance to the next (harder) target when the Curriculum detects mastery (learning progress dried
    up). The competence w PERSISTS across stages -> easy stages walk w forward into reach of the hard
    one (transfer). Feeds the per-tick error REDUCTION as the learning-progress signal. Returns
    (final competence w, stages mastered, steps used)."""
    cur = Curriculum(len(_TARGETS), seed=seed)
    w = 0.0
    steps = 0
    while steps < budget and not cur.done:
        steps += 1
        target = _TARGETS[cur.stage]
        before = abs(target - w)
        w = _learner_step(w, target)
        lp = max(
            0.0, before - abs(target - w)
        )  # learning progress = error reduction achieved this tick
        cur.observe(lp)
    return w, cur.mastered, steps


def _run_jump_learner(budget):
    """Jump-to-hardest control: the SAME learner and SAME budget, spent ENTIRELY on the hardest target
    with no curriculum. The hardest target starts out of reach (gap 1.0 >> _REACH) so the gradient is
    ~0 and the competence barely moves. Returns the final competence w."""
    w = 0.0
    for _ in range(budget):
        w = _learner_step(w, _TARGETS[-1])
    return w


def _stages_reached(w, tol=0.05):
    """How many milestone targets the competence w has actually attained (within tol)."""
    return sum(1 for t in _TARGETS if w >= t - tol)


def _advance_step(hi_steps, hi=0.3, lo=0.0, n_stages=4, seed=0):
    """Feed a Curriculum an LP trace that is HIGH for `hi_steps` ticks (still learning) then LOW
    (mastered); return the 1-indexed tick at which it ADVANCES out of stage 0. Proves self-pacing: the
    advance tracks WHEN the LP dries up, not a fixed clock (a longer hi_steps -> a proportionally later
    advance). While LP is high it must NOT advance -- so this only returns during the low tail."""
    cur = Curriculum(n_stages, seed=seed)
    t = 0
    for _ in range(hi_steps):
        t += 1
        if cur.observe(hi):
            return t  # (never happens while LP is high -> no premature advance)
    while True:
        t += 1
        if cur.observe(lo):
            return t


def main():
    T, patience = 20, 8

    # ---- (1) SELF-PACED (the key property): advance time TRACKS the learner's LP-dryup, not a clock ----
    fast_adv = _advance_step(hi_steps=T)  # LP high for T then 0   -> advances ~T + patience
    slow_adv = _advance_step(
        hi_steps=2 * T
    )  # slower learner: LP high 2T -> advances ~2T + patience

    # ---- (2) NO PREMATURE ADVANCE ----
    c_high = Curriculum(3, seed=0)
    for _ in range(100):
        c_high.observe(0.3)  # LP stays above threshold -> a still-learning stage is never skipped
    high_mastered = c_high.mastered

    c_dip = Curriculum(3, seed=0)
    dip_trace = (
        [0.3] * 5 + [0.0] + [0.3] * 20
    )  # a single one-off dip below threshold, then learning resumes
    dip_advanced = False
    for x in dip_trace:
        dip_advanced = c_dip.observe(x) or dip_advanced
    dip_mastered = c_dip.mastered

    c_diff = Curriculum(5, seed=0)
    d_start = c_diff.difficulty()  # fresh -> stage 0 -> difficulty 0.0
    for _ in range(500):
        c_diff.observe(0.0)  # master every stage
    d_end = c_diff.difficulty()  # done -> hardest stage -> difficulty 1.0

    # ---- (3) CURRICULUM-HELPS via TRANSFER (the HARD, un-riggable control) ----
    budget = 300
    cur_w, cur_mastered, cur_steps = _run_curriculum_learner(budget, seed=0)
    jump_w = _run_jump_learner(budget)
    cur_stages = _stages_reached(cur_w)
    jump_stages = _stages_reached(jump_w)

    # ---- (4) DETERMINISM: two seed=s curricula fed the same LP trace agree exactly ----
    srng = random.Random(1234)
    noisy = [0.3 + srng.uniform(-0.05, 0.05) for _ in range(10)] + [
        0.0
    ] * 30  # seeded noisy LP trace (masters 3)
    da, db = Curriculum(5, seed=7), Curriculum(5, seed=7)
    seq_a = [da.observe(x) for x in noisy]
    seq_b = [db.observe(x) for x in noisy]
    review_a = [da.review_stage() for _ in range(12)]
    review_b = [db.review_stage() for _ in range(12)]
    learner_repro = _run_curriculum_learner(budget, seed=0) == _run_curriculum_learner(
        budget, seed=0
    )

    print("=" * 78)
    print(
        "CURRICULUM -- self-paced developmental curriculum (starting small; graduates on mastery)"
    )
    print("=" * 78)
    print(
        f"n_stages=4  mastery_threshold=0.02  patience={patience}  (mastery = LP <= thresh for `patience` ticks)"
    )
    print("-" * 78)
    print(
        f"(1) self-paced  : fast learner (LP high {T}) advances at tick {fast_adv}; "
        f"slow learner (LP high {2 * T}) at tick {slow_adv}  (later by {slow_adv - fast_adv} = T)"
    )
    print(
        f"(2) no-premature: 100 high-LP ticks -> mastered={high_mastered}; a single dip -> mastered={dip_mastered}; "
        f"difficulty {d_start:.2f} -> {d_end:.2f} across the curriculum"
    )
    print(
        f"(3) transfer    : curriculum reaches competence {cur_w:.3f} (mastered {cur_mastered}/4, {cur_stages} "
        f"milestones, {cur_steps} steps) vs jump-to-hard {jump_w:.3f} ({jump_stages} milestones)"
    )
    print(
        f"(4) determinism : identical advance seq={seq_a == seq_b}  review seq={review_a == review_b}  "
        f"learner reproducible={learner_repro}"
    )
    print("=" * 78)

    # ---- (1) SELF-PACED asserts (advance tracks LP-dryup, NOT a fixed schedule) ----
    assert fast_adv == T + patience, (
        "a fast learner must advance exactly `patience` ticks after its LP dries up"
    )
    assert slow_adv == 2 * T + patience, (
        "a slower learner (LP high 2x as long) must advance at 2T + patience"
    )
    assert slow_adv - fast_adv == T, (
        "the advance time must TRACK the LP-dryup (slow later by exactly T), not a clock"
    )
    assert fast_adv > T, (
        "while LP is HIGH the curriculum must NOT advance (no premature graduation)"
    )

    # ---- (2) NO PREMATURE ADVANCE asserts ----
    assert high_mastered == 0, (
        "a still-learning stage (LP above threshold) must NEVER be mastered/skipped"
    )
    assert not dip_advanced and dip_mastered == 0, (
        "a single one-off LP dip must NOT advance (patience holds)"
    )
    assert d_start == 0.0 and d_end == 1.0, (
        "difficulty must rise 0 -> 1 as the curriculum is mastered end to end"
    )

    # ---- (3) TRANSFER -- the HARD, un-riggable assert ----
    assert cur_w > 0.9, "the self-paced curriculum must reach HIGH final competence within budget"
    assert jump_w < 0.1, (
        "jump-to-hardest must stay stuck (the hard target starts out of the learnability frontier)"
    )
    assert cur_w > 5.0 * jump_w, (
        f"curriculum competence {cur_w:.3f} must FAR exceed jump-to-hard {jump_w:.3f} (transfer via starting small)"
    )
    assert cur_mastered > jump_stages and cur_stages > jump_stages, (
        "the curriculum learner must master MORE milestones than the jump-to-hardest control within the same budget"
    )

    # ---- (4) DETERMINISM asserts ----
    assert seq_a == seq_b, (
        "two Curriculum(seed=7) fed the same LP trace must give identical advance sequences"
    )
    assert review_a == review_b, "seeded interleaved-review picks must be reproducible"
    assert learner_repro, (
        "the seeded curriculum learner must reproduce its (competence, mastered, steps) exactly"
    )

    print(
        f"self-check OK: self-paced -- fast learner graduates at tick {fast_adv}, slow at {slow_adv} "
        f"(later by exactly T={T}, tracking LP-dryup not a clock);"
    )
    print(
        "  no premature advance (100 high-LP ticks + a one-off dip both keep mastered=0; difficulty 0->1);"
    )
    print(
        f"  TRANSFER (hard): starting-small curriculum reaches competence {cur_w:.3f} mastering {cur_mastered}/4 "
        f"vs jump-to-hard {jump_w:.3f} ({jump_stages}/4) on the same {budget}-step budget "
        f"(mean advance tick {mean([fast_adv, slow_adv]):.1f}); deterministic"
    )


if __name__ == "__main__":
    main()
