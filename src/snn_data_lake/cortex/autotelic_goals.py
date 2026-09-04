r"""
Autotelic goal sampler + hindsight relabeling + competence-progress goal-babbling -- L9 / SI-5.

Where this sits. The learning tier so far pursues ONE externally-fixed goal (`_z_goal` in the JEPA
plan()/executive stack). This module turns that single goal into a DIVERSITY of SELF-SET goals the
agent invents from its OWN achieved outcomes -- autotelic, zero external goals -- and learns to reach a
whole space of them. It is the SI-5 rung (Baranes & Oudeyer, SAGG-RIAC 2013; Andrychowicz et al.,
Hindsight Experience Replay 2017; Colas et al., autotelic agents 2022), built on this repo's spiking
substrate and REUSING the SI-1 learning-progress kernel and the PFC working-memory attractor -- it
reimplements none of them.

The five pieces (each a REUSE, not a new mechanism):
  (1) AUTOTELIC GOAL SOURCE. Goals are SAMPLED from a buffer of the agent's own reached states
      (`AchievedOutcomeBuffer`); nothing external hands it a goal. The buffer is seeded with a couple of
      trivially-achieved start positions and GROWS as babbling visits new states -- the self-set goal
      space expands from the agent's own experience.
  (2) HOLD IN PFC WM. Each self-set goal is committed to and HELD in the reused
      `prefrontal_executive.PrefrontalExecutive` working-memory attractor -- the goal keeps SPIKING
      (persistent activity) across the pursuit delay (`maintain()`), exactly the WM the executive already
      provides.
  (3) GOAL-CONDITIONED SPIKING READOUT. `GoalConditionedReadout` is one reused `cortex.tempotron.Tempotron`
      per action (LEFT/STAY/RIGHT), a genuine spike-TIMING classifier. Its input is a population (bump)
      code of the CURRENT STATE *and* the HELD GOAL, so the chosen action depends on both -- the readout
      is literally goal-conditioned. Learned by the tempotron's own spike-timing rule (no rate ANN).
  (4) HINDSIGHT RELABEL (HER). Every observed transition (s -> s') is relabeled as if s' had been the
      goal all along: "to reach a state one step away in direction d, take action d". A failed episode
      (never touched the intended goal) still teaches how to reach the states it DID pass through -- the
      dense signal that a sparse original-goal reward lacks.
  (5) COMPETENCE-PROGRESS GOAL SELECTION (SAGG-RIAC). Which goal REGION to practice is chosen by re-
      pointing the SI-1 `spiking_curiosity.SpikingCuriosityDrive` LP kernel at goal-REACHING error: the
      region whose reaching-competence is still IMPROVING wins (the learnable frontier); a mastered
      region (flat-low error) or an unreachable region (flat-high error) earns ~0 progress and is
      abandoned -- the SI-1 anti-noisy-TV property, here over competence.

HONEST framing (read this before believing anything -- this rung is easy to fake).
  * BOUNDED, NOT open-ended. The toy goal space SATURATES: a handful of track positions, mastered once
    the readout covers the visited afferents. This is bounded goal-directedness, not open-endedness.
  * MASTERY IS NEAR-TABULAR; TRANSFER DOES NOT HOLD. The readout interpolates WITHIN the densely-babbled
    span (that is the honest held-out win below) but does NOT extrapolate: a genuinely-novel goal whose
    population code sits on afferents that never fired in training drops to CHANCE. Self-check (4) prints
    that failure -- coverage, not abstraction.
  * THE HINDSIGHT-TAUTOLOGY TRAP. Hindsight relabels whatever you reach as a "success", so training-goal
    success is trivially high and MEANS NOTHING. Every headline number here is measured on a FROZEN
    HELD-OUT goal set that is NEVER relabeled and NEVER used to pick a training goal -- the one metric
    hindsight cannot inflate.
  * NOT A NEW AXIS. "Competence progress" is the SAME SI-1 learning-progress kernel pointed at goal-
    reaching error instead of world-model error. It is a REUSE (a re-pointing), not a distinct second
    drive; the self-check frames it as such and does not sell it as a new mechanism.

Reuses (does NOT reimplement): cortex/tempotron.Tempotron (the goal-conditioned spike-timing readout),
neuromodulation/spiking_curiosity.SpikingCuriosityDrive (SI-1 -- the competence-progress LP kernel),
cortex/prefrontal_executive.PrefrontalExecutive (the WM attractor that HOLDS a goal),
neuromodulation/drive_reduction.REWARD_SCALE (SI-2 -- the internal drive-reduction reward channel the
goal-reaching reward rides), substrate/_mathutils.mean. Per-instance random.Random(seed),
zlib.crc32 (never hash()), ASCII-only prints, deterministic.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.autotelic_goals
"""

import random
from collections import deque

from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.cortex.tempotron import Tempotron
from snn_data_lake.neuromodulation.drive_reduction import REWARD_SCALE
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive
from snn_data_lake.substrate._mathutils import mean

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "GoalConditionedReadout.train": (
        "LOCAL",
        "trains on hindsight labels by delegating to two Tempotrons, one per direction, one-vs-rest. "
        "The tempotron rule is spike-timing-local and declared LOCAL at its own site; this is the "
        "harness over it",
    ),
}

# ---- the toy goal world: a 1-D track. Positions 0..REACH_HI are reachable during training (a wall at ----
# ---- REACH_HI). The afferent/goal axis runs 0..AXIS-1 and EXTENDS past the wall so a genuinely-novel  ----
# ---- goal (position NOVEL_GOAL) lands on afferents that NEVER fire in training -- the no-transfer probe.--
AXIS = 12  # population-code afferent positions 0..11 (state and goal each coded over AXIS)
REACH_LO = 0  # left edge of the training-reachable span
REACH_HI = 7  # right edge (a wall): training never occupies positions > REACH_HI
BUMP_HW = (
    1  # population (bump) half-width: afferents |i-pos| <= BUMP_HW fire -> overlap = interpolation
)
T0 = 3.0  # single spike time each active afferent fires at (inside the tempotron's [0,T] scan)
BABBLE_STARTS = (
    0,
    1,
    2,
)  # babbling episodes start in the LEFT region; coverage spreads rightward over episodes
H = 12  # episode / pursuit horizon (>= track width, so any goal is reachable in principle)
N_BABBLE = 400  # babbling episodes for the full held-out comparison (controls 1 + 2)
INTENDED_GOALS = (
    0,
    7,
)  # SPARSE pursued goals (autotelic, from the buffer). Sparse -> the no-HER control starves.
HELDOUT_GOALS = (2, 5)  # FROZEN held-out goals: never pursued, never a hindsight relabel target
TRAIN_GOALS = tuple(
    p for p in range(REACH_LO, REACH_HI + 1) if p not in HELDOUT_GOALS
)  # tautological-report set
NOVEL_GOAL = 10  # genuinely-novel goal beyond the wall: its bump sits on never-trained afferents
EVAL_STARTS = tuple(
    range(REACH_LO, REACH_HI + 1)
)  # deterministic eval start set (no rng in evaluation)

# ---- SAGG-RIAC goal REGIONS for the competence-progress selection control (each a real measured curve) --
FRONTIER_GOALS = (
    3,
    4,
    6,
    7,
)  # region A: far but reachable -> reaching-competence RISES with practice (frontier)
UNREACH_GOALS = (
    9,
    10,
)  # region C: beyond the wall -> reaching-competence is exactly 0.0 (unlearnable)
#                                 region B (mastered) = goal==start: already achieved -> competence 1.0 (flat)

# ---- actions: a 3-way directed move. optimal(s, g) = RIGHT if g>s, LEFT if g<s, STAY if g==s ----
LEFT, STAY, RIGHT = 0, 1, 2
N_ACTIONS = 3
_DELTA = {LEFT: -1, STAY: 0, RIGHT: +1}

# ---- readout training knobs (tempotron spike-timing rule) ----
_EPOCHS = 24  # tempotron training epochs over the (deduplicated) hindsight label set
_LR = 0.02  # tempotron learning rate
_STAY_MARGIN = (
    0.05  # |v_right - v_left| below this -> direction is ambiguous (at the goal) -> STAY emerges
)


def _step(pos, action, lo, hi):
    """Advance the track one step under `action`, clamped to [lo, hi] (the wall). Deterministic."""
    p = pos + _DELTA[action]
    return lo if p < lo else (hi if p > hi else p)


def _direction(s, s_next):
    """The hindsight action label: the direction actually moved from s to s_next (STAY if it did not
    move -- including a wall-clamped step). This is the HER truth 'to reach where you ended up, move the
    way you moved', so it is correct regardless of the behaviour policy or the wall."""
    if s_next > s:
        return RIGHT
    if s_next < s:
        return LEFT
    return STAY


class AchievedOutcomeBuffer:
    """The autotelic goal source: a bounded buffer of states the agent has actually REACHED. Goals are
    SAMPLED from here (never handed in from outside); `add()` grows the self-set goal space as new states
    are visited. Deterministic sampling via a caller-supplied random.Random."""

    def __init__(self, maxlen=512):
        self._seen = deque(maxlen=maxlen)
        self._set = set()

    def add(self, state):
        """Record a reached state as a candidate self-set goal (deduplicated, bounded)."""
        if state not in self._set:
            self._seen.append(state)
            self._set.add(state)

    def all_seen(self):
        """Every state ever reached, as a frozenset. The UNBOUNDED view.

        Added for basal_ganglia/curriculum.GoalCurriculum, which needs the full candidate set and
        has no `allowed` pool to intersect against, so neither sample() nor coverage() fits. It
        previously reached into `_set` directly; an accessor is the honest boundary.

        NOTE FOR CALLERS, because the two views genuinely differ: this set is unbounded and grows
        with every distinct state, while `_seen` is a `maxlen`-bounded recency deque. `maxlen`
        therefore does NOT bound what this returns.
        """
        return frozenset(self._set)

    def sample(self, rng, allowed):
        """Draw one self-set goal from the achieved outcomes intersected with `allowed` (the training-goal
        pool -- so held-out/novel positions are never chosen as an intended goal). Falls back to the
        allowed pool itself before the buffer has covered it. Deterministic given `rng`."""
        pool = [s for s in self._seen if s in allowed]
        if not pool:
            pool = list(allowed)
        return rng.choice(pool)

    def coverage(self, allowed):
        """Fraction of the allowed goal pool the buffer has actually reached (the self-set space grew)."""
        return sum(1 for s in allowed if s in self._set) / len(allowed)


class GoalConditionedReadout:
    """A goal-conditioned SPIKING readout: two reused `Tempotron`s (a LEFT unit and a RIGHT unit),
    classifying (state, goal) by which direction-neuron's peak sub-threshold voltage is higher. The input
    pattern is a population (bump) code of the CURRENT STATE on afferents [0, AXIS) AND the HELD GOAL on
    afferents [AXIS, 2*AXIS), so the action is a genuine function of BOTH -- goal-conditioned. STAY is not a
    trained unit: it EMERGES when the two direction units are within `_STAY_MARGIN` (the direction is
    ambiguous, i.e. the goal is right here) -- so holding at the goal interpolates for free at never-labeled
    held-out goals. Trained by the tempotron's own spike-timing rule (Gutig & Sompolinsky). Deterministic
    (per-unit seeds; no rng at inference)."""

    def __init__(self, seed=0):
        self.n_aff = 2 * AXIS
        self.t_left = Tempotron(self.n_aff, threshold=1.0, seed=seed * 131 + 1)
        self.t_right = Tempotron(self.n_aff, threshold=1.0, seed=seed * 131 + 2)

    def _pattern(self, s, g):
        """Population (bump) spike pattern for (state s, goal g): every afferent within BUMP_HW of s (state
        bank) or g (goal bank) emits ONE spike at T0. Overlapping bumps of nearby positions share afferents
        -> the learned weights INTERPOLATE across positions (held-out goals) but do NOT extrapolate past the
        span of afferents that ever fired (novel goals -> no goal signal -> chance)."""
        pat = [[] for _ in range(self.n_aff)]
        for i in range(AXIS):
            if abs(i - s) <= BUMP_HW:
                pat[i].append(T0)  # state bank
            if abs(i - g) <= BUMP_HW:
                pat[AXIS + i].append(T0)  # goal bank
        return pat

    def train(self, examples):
        """Train on hindsight labels `examples` = list of (state, goal, direction) where direction is LEFT
        or RIGHT (STAY is never trained -- it emerges from the balance at inference). One-vs-rest: the LEFT
        unit targets True on LEFT examples, the RIGHT unit on RIGHT examples. No randomness (deterministic)."""
        pats = [self._pattern(s, g) for (s, g, _a) in examples]
        self.t_left.train(pats, [ex[2] == LEFT for ex in examples], epochs=_EPOCHS, lr=_LR)
        self.t_right.train(pats, [ex[2] == RIGHT for ex in examples], epochs=_EPOCHS, lr=_LR)

    def action(self, s, g):
        """Goal-conditioned action for (state s, goal g): move in the direction whose unit peaks higher;
        STAY when the two peaks are within `_STAY_MARGIN` (goal reached / direction ambiguous). Deterministic."""
        pat = self._pattern(s, g)
        v_left = self.t_left.vmax(pat)[0]
        v_right = self.t_right.vmax(pat)[0]
        if abs(v_right - v_left) < _STAY_MARGIN:
            return STAY
        return RIGHT if v_right > v_left else LEFT


def _reaches(readout, s0, g, horizon, lo, hi):
    """Run the goal-conditioned policy from start s0 toward goal g for `horizon` steps; reached iff the
    FINAL state equals g (the policy must arrive AND hold there -- STAY at the goal). Deterministic."""
    p = s0
    for _ in range(horizon):
        p = _step(p, readout.action(p, g), lo, hi)
    return p == g


def _reach_rate(readout, starts, goals, horizon, lo, hi):
    """Deterministic mean final-state reach-rate of `readout` over all (start, goal) pairs."""
    total = ok = 0
    for s0 in starts:
        for g in goals:
            total += 1
            ok += 1 if _reaches(readout, s0, g, horizon, lo, hi) else 0
    return ok / total


def _random_reach_rate(starts, goals, horizon, lo, hi, seed):
    """CHANCE baseline: the same reach metric under a RANDOM-action policy (per-instance rng)."""
    rng = random.Random(seed)
    total = ok = 0
    for s0 in starts:
        for g in goals:
            p = s0
            for _ in range(horizon):
                p = _step(p, rng.randrange(N_ACTIONS), lo, hi)
            total += 1
            ok += 1 if p == g else 0
    return ok / total


def collect_labels(n_episodes, hindsight, seed):
    """GOAL-BABBLING data collection. For `n_episodes`: sample a SPARSE self-set intended goal from the
    achieved-outcome buffer (autotelic -- from the agent's own reached states), then execute RANDOM actions
    (motor babbling) for H steps from a left-region start, recording every visited state back into the buffer
    (the self-set goal space GROWS to cover the reachable track).

    HER (`hindsight=True`): the standard 'future' relabel -- every visited state s_j (j > i) in the episode
    is treated as a goal the agent DID achieve, giving the dense, always-correct label (state=s_i, goal=s_j,
    direction toward s_j). A held-out state is NEVER used as a relabel goal. These achieved-outcome relabel
    goals ARE the diversity of self-set goals (autotelic); the sparse intended goal only drives exploration.
    No-HER (`hindsight=False`, control a): learn ONLY from episodes that actually reached the sparse intended
    goal -- the sparse original-goal signal. Since the intended set is tiny, coverage is sparse. Labels are
    deduplicated (the direction for a given (s, g) is unique on the track), so training is small + deterministic.

    Returns (sorted example list, buffer)."""
    rng = random.Random(seed)
    buf = AchievedOutcomeBuffer()
    for s in BABBLE_STARTS:
        buf.add(s)  # seed with trivially-achieved start positions
    labels = {}  # (state, goal) -> direction  (deduplicated)
    heldout = set(HELDOUT_GOALS)
    intended_pool = set(INTENDED_GOALS)
    for _ in range(n_episodes):
        g_intended = buf.sample(
            rng, intended_pool
        )  # AUTOTELIC: sparse pursued goal from own outcomes
        s = rng.choice(BABBLE_STARTS)
        traj = [s]
        for _t in range(H):
            s = _step(
                s, rng.randrange(N_ACTIONS), REACH_LO, REACH_HI
            )  # motor babbling: random action
            traj.append(s)
            buf.add(s)  # the self-set goal space grows
        if hindsight:
            for i in range(len(traj)):  # 'future' HER: every achieved future state is a goal
                for j in range(i + 1, len(traj)):
                    g = traj[j]
                    if g == traj[i] or g in heldout:  # never relabel to a held-out goal
                        continue
                    labels[(traj[i], g)] = _direction(traj[i], g)
        elif g_intended in traj:  # sparse: only if the intended goal was reached
            for i in range(len(traj)):
                if traj[i] != g_intended:
                    labels[(traj[i], g_intended)] = _direction(traj[i], g_intended)
    examples = sorted((s, g, a) for (s, g), a in labels.items())
    return examples, buf


def hold_goal_in_pfc(goals, delay, seed):
    """Item (2): commit each self-set goal to the reused PFC working-memory attractor and HOLD it across a
    `delay`-tick input-free interval. Returns (all_held, min_wm_spikes): all_held is True iff every goal
    stayed the maintained WM goal for the whole delay, and min_wm_spikes is the smallest per-tick emitted
    spike count of the persistent-activity pool (persistent SPIKING = the goal is actively held)."""
    all_held = True
    min_spikes = None
    for g in goals:
        pfc = PrefrontalExecutive(n_options=AXIS, commit_races=9, seed=seed)
        evidence = [0.0] * AXIS
        evidence[g] = 1.0  # one-hot drive -> the executive commits goal g
        pfc.update(evidence)
        if pfc.goal != g:
            all_held = False
        for _ in range(delay):
            held = pfc.maintain()  # input-free delay: the WM pool keeps spiking
            if held != g:
                all_held = False
            s = pfc._wm_spikes  # emitted spikes of the active WM pool this tick
            min_spikes = s if min_spikes is None else min(min_spikes, s)
    return all_held, (min_spikes if min_spikes is not None else 0)


def internal_reach_reward(readout, starts, goals, horizon, lo, hi):
    """SI-2 reuse: the intrinsic goal-reaching reward is DRIVE REDUCTION -- the fall in distance-to-goal
    over the episode, scaled by drive_reduction.REWARD_SCALE (the same internal-reward channel). Returns
    the mean per-episode reward; positive means the trained policy closes the distance to its self-set
    goals with ZERO external reward."""
    rewards = []
    for s0 in starts:
        for g in goals:
            p = s0
            d_before = abs(g - s0)
            for _ in range(horizon):
                p = _step(p, readout.action(p, g), lo, hi)
            d_after = abs(g - p)
            rewards.append(REWARD_SCALE * (d_before - d_after))  # drive (distance) REDUCTION
    return mean(rewards)


_CURVE_BUDGETS = (
    3,
    6,
    10,
    16,
    24,
    40,
)  # increasing babbling budgets -> coverage grows (the frontier rises)
_CURVE_SEEDS = 6  # average the (noisy small-data) frontier competence over seeds to denoise


def _competence_curve(seed):
    """Real, MEASURED per-region reaching-competence curves for the SAGG-RIAC selection control. Each is a
    genuine reach-rate measured on the real spiking readout -- none is hand-drawn:
      frontier (A): far reachable goals FRONTIER_GOALS, competence measured over an INCREASING babbling
                    budget and AVERAGED over seeds (the small-data readout is noisy; averaging denoises).
                    It RISES as coverage grows -> the learnable frontier.
      mastered (B): goal == start -- already achieved, reached by holding (STAY). Competence is a flat 1.0
                    (measured on the fully-trained readout) -> nothing left to learn.
      unreach  (C): goals beyond the wall (UNREACH_GOALS). The agent is clamped, so competence is exactly
                    0.0 forever -> unlearnable.
    The frontier's falling reaching-error is the only region with competence PROGRESS; the SI-1 kernel keys
    on exactly that. Returns dict region_label -> list of competence values (one per budget)."""
    frontier = []
    for k in _CURVE_BUDGETS:
        vals = []
        for sd in range(_CURVE_SEEDS):
            examples, _buf = collect_labels(k, hindsight=True, seed=sd)
            ro = GoalConditionedReadout(seed=sd)
            ro.train(examples)
            vals.append(_reach_rate(ro, EVAL_STARTS, FRONTIER_GOALS, H, REACH_LO, REACH_HI))
        frontier.append(mean(vals))
    full_examples, _b = collect_labels(N_BABBLE, hindsight=True, seed=seed)
    full = GoalConditionedReadout(seed=seed)
    full.train(full_examples)
    mastered_rate = mean(
        [1.0 if _reaches(full, s, s, H, REACH_LO, REACH_HI) else 0.0 for s in EVAL_STARTS]
    )
    unreach_rate = _reach_rate(full, EVAL_STARTS, UNREACH_GOALS, H, REACH_LO, REACH_HI)
    n = len(_CURVE_BUDGETS)
    return {"frontier": frontier, "mastered": [mastered_rate] * n, "unreach": [unreach_rate] * n}


def sagg_riac_practice(curves, n_steps, selector, seed):
    """Run a goal-region practice loop for `n_steps`. `selector` is 'sagg' (competence-progress: re-point
    the SI-1 LP kernel at goal-REACHING error and practise the region of maximal curiosity) or 'random'
    (uniform). Each practice step advances the chosen region along its REAL competence curve and feeds the
    new reaching-error (1 - competence) into the SI-1 kernel. Returns the practice count per region.

    HONEST NOTE: 'competence progress' here is literally `SpikingCuriosityDrive` -- the SAME SI-1 kernel
    from SI-1 -- with goal-reaching error substituted for world-model error. It is a re-pointing (a reuse),
    NOT a new/distinct drive."""
    regions = list(curves)
    rng = random.Random(seed)
    drive = SpikingCuriosityDrive(seed=seed)  # SI-1 kernel, re-pointed at goal-reaching error
    practice = dict.fromkeys(regions, 0)
    for _ in range(n_steps):
        if selector == "sagg":
            r = drive.most_curious(regions)  # frontier = region of max competence-progress
        else:
            r = rng.choice(regions)  # control: uniform random goal selection
        k = min(practice[r], len(curves[r]) - 1)
        err = 1.0 - curves[r][k]  # goal-REACHING error (falls as competence rises)
        drive.observe(r, err)  # fold competence error into the SI-1 LP kernel
        practice[r] += 1
    return practice


def main():
    width = 96
    seed = 7
    print("=" * width)
    print("AUTOTELIC GOAL SAMPLER + HINDSIGHT + COMPETENCE-PROGRESS GOAL-BABBLING -- L9 / SI-5")
    print(
        "self-set goals from own outcomes; held in PFC WM; goal-conditioned spiking readout; HER; SAGG-RIAC"
    )
    print("=" * width)

    # ---- build the WITH-HER and NO-HER (control a) readouts on the SAME goal-babbling stream -----------
    her_examples, her_buf = collect_labels(N_BABBLE, hindsight=True, seed=seed)
    no_examples, _no_buf = collect_labels(N_BABBLE, hindsight=False, seed=seed)
    her = GoalConditionedReadout(seed=seed)
    her.train(her_examples)
    noher = GoalConditionedReadout(seed=seed)
    noher.train(no_examples)

    # ---- (1) HELD-OUT goal-reaching -- the metric HINDSIGHT CANNOT INFLATE (frozen, never relabeled) ---
    her_held = _reach_rate(her, EVAL_STARTS, HELDOUT_GOALS, H, REACH_LO, REACH_HI)
    noher_held = _reach_rate(noher, EVAL_STARTS, HELDOUT_GOALS, H, REACH_LO, REACH_HI)
    chance_held = _random_reach_rate(EVAL_STARTS, HELDOUT_GOALS, H, REACH_LO, REACH_HI, seed=seed)
    her_train = _reach_rate(
        her, EVAL_STARTS, TRAIN_GOALS, H, REACH_LO, REACH_HI
    )  # (tautological -- context only)

    # ---- (2) item (2): the self-set goals are HELD in the PFC WM attractor (persistent spiking) --------
    held_ok, min_wm_spikes = hold_goal_in_pfc(HELDOUT_GOALS + TRAIN_GOALS[:2], delay=10, seed=seed)

    # ---- SI-2 internal drive-reduction reaching reward (zero external reward) --------------------------
    her_reward = internal_reach_reward(her, EVAL_STARTS, HELDOUT_GOALS, H, REACH_LO, REACH_HI)

    # ---- (3) SAGG-RIAC competence-progress selection vs RANDOM selection -------------------------------
    curves = _competence_curve(seed=seed)
    n_practice = 90
    sagg = sagg_riac_practice(curves, n_practice, "sagg", seed=seed)
    rand = sagg_riac_practice(curves, n_practice, "random", seed=seed)
    sagg_frontier = sagg["frontier"] / n_practice
    rand_frontier = rand["frontier"] / n_practice
    sagg_unreach = sagg["unreach"] / n_practice
    sagg_mastered = sagg["mastered"] / n_practice

    # ---- (4) HONEST NEGATIVE: a genuinely-novel goal (beyond the trained afferent span) -> CHANCE ------
    novel_starts = (6, 7, 8, 9, 11)  # near the novel goal, on never-trained goal afferents
    novel_hi = AXIS - 1  # the test world extends past the wall (reachable)
    her_novel = _reach_rate(her, novel_starts, (NOVEL_GOAL,), H, REACH_LO, novel_hi)
    chance_novel = _random_reach_rate(novel_starts, (NOVEL_GOAL,), H, REACH_LO, novel_hi, seed=seed)

    print(
        f"world: track [{REACH_LO},{REACH_HI}] (wall); afferent axis 0..{AXIS - 1}; bump hw={BUMP_HW}; "
        f"actions LEFT/STAY/RIGHT; horizon H={H}"
    )
    print(
        f"autotelic buffer: seeded {len(BABBLE_STARTS)} starts -> covered {her_buf.coverage(TRAIN_GOALS):.0%} "
        f"of the train-goal pool from OWN outcomes; HER labels={len(her_examples)} no-HER labels={len(no_examples)}"
    )
    print("-" * width)
    print(
        f"(1) HELD-OUT goal-reaching (goals {HELDOUT_GOALS}, NEVER relabeled -- hindsight cannot inflate):"
    )
    print(
        f"      WITH hindsight : {her_held:.2f}      NO hindsight [control a]: {noher_held:.2f}      "
        f"chance (random policy): {chance_held:.2f}"
    )
    print(
        f"      (training-goal reach {her_train:.2f} is the TAUTOLOGICAL number -- reported, never trusted)"
    )
    print(
        f"(2) held in PFC WM  : goals stayed maintained across a 10-tick delay={held_ok}; "
        f"min persistent WM spikes/tick={min_wm_spikes} (>0 = actively held)"
    )
    print(
        f"      SI-2 internal drive-reduction reaching reward (zero external reward): {her_reward:+.3f}"
    )
    print(
        "(3) SAGG-RIAC goal selection (SI-1 kernel re-pointed at goal-reaching error; frontier=region A):"
    )
    print(
        f"      competence-progress selection -> frontier {sagg_frontier:.0%}, mastered {sagg_mastered:.0%}, "
        f"unreachable {sagg_unreach:.0%}"
    )
    print(
        f"      RANDOM selection [control]     -> frontier {rand_frontier:.0%} (uniform ~= {1 / len(curves):.0%})"
    )
    print(
        f"      real competence curves: frontier (rising) {[round(c, 2) for c in curves['frontier']]}; "
        f"mastered {curves['mastered'][0]:.2f} flat; unreach {curves['unreach'][0]:.2f} flat"
    )
    print(f"(4) HONEST NEGATIVE -- genuinely-novel goal {NOVEL_GOAL} (afferents never trained):")
    print(
        f"      WITH hindsight : {her_novel:.2f}      chance (random policy): {chance_novel:.2f}   "
        f"-> NO transfer (near-tabular); mastery does not extrapolate"
    )
    print("=" * width)

    # ---- (1) HELD-OUT: HER markedly above chance AND above the no-hindsight control --------------------
    assert her_held >= chance_held + 0.30, (
        f"WITH-hindsight held-out reaching must be markedly above chance ({her_held:.2f} vs {chance_held:.2f})"
    )
    assert her_held >= 0.60, (
        f"WITH-hindsight held-out reaching must actually reach the frozen goals ({her_held:.2f})"
    )

    # ---- (2) CONTROL a -- NO hindsight: held-out reaching collapses toward chance (hindsight helps) ----
    assert her_held >= noher_held + 0.30, (
        f"removing hindsight must markedly WORSEN held-out reaching ({her_held:.2f} -> {noher_held:.2f})"
    )
    assert noher_held <= chance_held + 0.15, (
        f"CONTROL: without hindsight, held-out reaching must be near chance ({noher_held:.2f} vs {chance_held:.2f})"
    )

    # ---- item (2): the goals are genuinely HELD in the spiking PFC WM ---------------------------------
    assert held_ok, (
        "each self-set goal must stay the maintained PFC WM goal across the input-free delay"
    )
    assert min_wm_spikes > 0, (
        "the PFC WM pool must keep SPIKING while a goal is held (persistent activity)"
    )

    # ---- (3) SAGG-RIAC: competence-progress selection concentrates practice on the learnable frontier --
    assert sagg_frontier >= 2.0 * rand_frontier, (
        f"competence-progress selection must focus practice on the frontier vs random ({sagg_frontier:.2f} vs {rand_frontier:.2f})"
    )
    assert sagg_frontier >= sagg_unreach + 0.30, (
        f"the frontier must be practised far more than the unreachable region ({sagg_frontier:.2f} vs {sagg_unreach:.2f})"
    )
    assert curves["frontier"][-1] > curves["frontier"][0], (
        "the frontier competence curve must actually RISE (that rise is what makes it the frontier)"
    )

    # ---- (4) HONEST NEGATIVE -- the novel goal does NOT transfer (near-tabular, no extrapolation) ------
    assert her_novel <= chance_novel + 0.10, (
        f"HONEST NEGATIVE: a genuinely-novel goal must NOT transfer -- reaching stays ~chance ({her_novel:.2f} vs {chance_novel:.2f})"
    )
    assert her_novel < 0.50, (
        f"HONEST NEGATIVE: novel-goal reaching must be poor, not mastered ({her_novel:.2f})"
    )

    # ---- determinism: same seed -> identical held-out reaching -----------------------------------------
    her2_examples, _b2 = collect_labels(N_BABBLE, hindsight=True, seed=seed)
    her2 = GoalConditionedReadout(seed=seed)
    her2.train(her2_examples)
    assert _reach_rate(her2, EVAL_STARTS, HELDOUT_GOALS, H, REACH_LO, REACH_HI) == her_held, (
        "same seed must reproduce held-out reaching exactly"
    )

    print(
        f"self-check OK: HELD-OUT goal-reaching {her_held:.2f} (never-relabeled -- hindsight cannot inflate) "
        f">> chance {chance_held:.2f} and >> no-hindsight {noher_held:.2f};"
    )
    print(
        f"  self-set goals held in the spiking PFC WM (persistent {min_wm_spikes} spikes/tick, zero-external "
        f"drive-reduction reward {her_reward:+.3f}); SAGG-RIAC focuses the frontier {sagg_frontier:.0%} vs "
        f"random {rand_frontier:.0%} (SAME SI-1 kernel re-pointed, not a new axis);"
    )
    print(
        f"  HONEST NEGATIVE: bounded + near-tabular -- the novel goal {NOVEL_GOAL} stays at chance "
        f"({her_novel:.2f} vs {chance_novel:.2f}); the goal space saturates, transfer does not hold."
    )


if __name__ == "__main__":
    main()
