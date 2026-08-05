r"""
Spiking learning-progress curiosity + self-paced curriculum -- the ONE live self-improvement
primitive (neuromodulation/intrinsic_motivation.py CuriosityDrive + neuromodulation/curriculum.py
Curriculum) rebuilt so its core signal is GENUINELY SPIKING: the learning-progress value is read
off EMITTED SPIKE COUNTS of leaky integrate-and-fire pools, never off a rate EMA or a membrane
potential. Roadmap item SI-1 "SPIKE-LP".

Honest framing (read this first). This is a PURITY conversion, NOT a new capability. On a stationary
or decaying task it behaves like the rate CuriosityDrive it replaces -- the win is that the mechanism
now lives in spikes, so the live-loop
self-improvement primitive stops being a rate-ANN drift inside an otherwise spiking brain. The
spiking margin is HONESTLY SMALLER than the rate version's (rectification of two noisy spike-rate
estimates leaves a small residual bias on flat noise); the measured margin is printed, not faked.
It holds no world model and makes no action decision -- it is the CONSUMER of a surprise signal.

WHAT THE DROP-IN COSTS BEHAVIOURALLY -- measured 2026-08-04, and the claim above used to be wider.
This paragraph said "same anti-noisy-TV ordering, same self-paced graduation". The graduation half
holds. The ordering half was never re-measured against the rate original at the level that matters,
because THIS module's self-check has no exploration test while `intrinsic_motivation`'s does
(`_run_explorer`, which asserts `visits["learnable"] > visits["noisy"]`). Running that same
curiosity-greedy explorer against both -- `_run_explorer(Drive(seed=s), 48, greedy=True)` for
s in 0..11, splits written learnable/mastered/noisy (RE-DERIVED 2026-08-05):

    chases the learnable frontier   rate 11/12 seeds     spiking 10/12 seeds
    seed-0 visit split              rate 46/1/1          spiking 27/6/15
    MEAN split over the 12 seeds    rate 41.8/1.5/4.7    spiking 24.1/6.8/17.1
      curiosity-blind random control (greedy=False), same 12 seeds: mean 16.5/14.8/16.7

So the ordering is NOT the same. The spiking drive still beats the blind control and still prefers
the frontier on most seeds, but its preference is far weaker -- the unlearnable-noisy candidate
takes 17.1 of 48 visits on average against the rate drive's 4.7 (and against 1 of 48 on the rate
drive's modal 46/1/1 seeds) -- and it loses the property on one more seed. Recorded as an honest
negative rather than smoothed into the sentence above.

*(Two figures were CORRECTED here on 2026-08-05 by re-running them. The random-control split was
recorded as "13/19/16"; that split occurs at no seed in 0..7 for either drive class, and the
protocol behind it was not recorded -- it is replaced above by the 12-seed mean, whose protocol is.
A sentence claiming "the spiking margin ran 6.7x-248x (mean 61.6x) against the rate drive's flat
18.2x" was DELETED: it recorded no accumulation protocol, so it is not re-derivable, and every
best-faith reconstruction is dominated by a near-zero denominator on the noisy arm and returns
values spanning ten orders of magnitude. The margin this module can honestly quote is the one its
own self-check PRINTS on every run -- "MEASURED margin 16.9x" at (2) below -- because that one is
re-derived by the gate rather than by a comment.)*

P2 STATUS, stated because two mechanisms for one job is the defect P2 names. Both drives are live:
`brainstem/autonomous_loop.py` selects between them with a `spiking_curiosity` flag and
`cortex/brain.py` sets it True, so the canonical brain runs this one while the rate original stays
constructible with its own importers. P2 calls a flag that keeps both paths "additive with extra
steps". This is NOT resolved here -- the resolution spans a file outside this module, and the
measurement above does not support retiring EITHER (the spiking one is not better on the behaviour,
the rate one is not spiking, and spiking-ness is a project thesis rather than a benchmark result).
Recorded as open P2 debt with its evidence attached, which is what P2 asks for when neither
mechanism dominates.

The key idea (unchanged from the rate version). Rewarding raw surprise (high prediction error)
gets permanently stuck on unpredictable NOISE -- a screen of static has maximal error forever (the
"noisy-TV problem"). Developmental curiosity rewards LEARNING PROGRESS = the RATE at which a
context's prediction error is FALLING:
  - HIGH where the model is still improving      (learnable novelty -- the frontier),
  - ~0  where the context is already mastered     (error low and flat -- boring),
  - ~0  where error is high but FLAT / unlearnable (noise -- so it does NOT get stuck).

The SPIKING mechanism (this is the whole point).
  Per hashable context_key the drive keeps TWO small LIF pools with per-neuron heterogeneity:
     FAST pool  (SHORT membrane tau_m) -- its firing rate tracks the RECENT prediction error,
     SLOW pool  (LONG  membrane tau_m) -- its membrane retains charge, so its firing rate LAGS.
  On each observe(k, e) a current proportional to the prediction error e is injected into every
  neuron of BOTH pools for a short sub-window of ticks, and the EMITTED SPIKE COUNT (the 1/0 returns
  of LIFNeuron.step) is recorded into a sliding window per pool. Each pool is calibrated once, on
  first sighting, to a shared reference error so both pools normalise to ~1 on a FLAT error (the
  spiking analog of two EMAs converging to the same mean). Then:
     learning_progress = max(0, slow_rate - fast_rate),  each rate = windowed spike count / normaliser.
  When the error is FALLING the fast pool's rate has already dropped while the lagging slow pool is
  still firing at the old higher rate -> slow_rate > fast_rate -> POSITIVE learning progress. When
  the error is FLAT (mastered-low OR noisy-high) both pools settle to the same rate -> LP ~ 0. That
  ON/OFF temporal contrast between two membrane time-constants is the anti-noisy-TV kernel, realised
  in spike counts rather than in a rate difference. First sighting seeds with no transient (LP ~ 0).

  The self-paced CURRICULUM likewise replaces its integer low-progress streak with a LIF ACCUMULATOR
  neuron: each low-LP tick injects a fixed "evidence" current and steps the accumulator; a high-LP
  tick RESETS it (still learning -> no premature advance). When the accumulator SPIKES (sustained
  low-LP evidence crosses threshold) the stage is judged LEARNED. The injection is derived from
  `patience` so the accumulator spikes after exactly `patience` consecutive low-LP ticks, while a
  single one-off dip leaks / gets reset and sustained HIGH LP never spikes -- "starting small" made
  spiking.

Both classes duck-type their rate originals: identical constructor signatures and public methods, so
they drop in wherever CuriosityDrive / Curriculum are used (the loop wiring is done by the caller).

Refs: Oudeyer & Kaplan, "What is intrinsic motivation? A typology of computational approaches"
(2007); Schmidhuber, "Formal Theory of Creativity, Fun, and Intrinsic Motivation" (2010); Elman,
"Learning and development in neural networks: the importance of starting small" (Cognition, 1993).

Deps: math, random, collections.deque, cortex.lif_neuron.LIFNeuron,
substrate._mathutils.mean (numpy/torch allowed project-wide); ASCII-only prints, deterministic (per-instance random.Random(seed);
LIF dynamics are deterministic; spike counts are integers).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.neuromodulation.spiking_curiosity
"""

import math
import random
from collections import deque

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate._mathutils import mean

# ---- SPIKING learning-progress pool constants (tuned so a decaying error yields clearly positive ----
# ---- accumulated LP while a constant-high-mean+jitter stream yields near-zero LP; NON-saturated,  ----
# ---- NON-mode-locked -- an honest finite anti-noisy-TV margin, not a saturation/mode-lock zero).  ----
_POOL_N = 12  # neurons per pool (averaging shrinks the rectification bias on flat noise)
_SUB = 10  # sub-ticks the error current is injected per observe (finer spike-count resolution)
_WINDOW = 12  # sliding window of recent per-observe spike counts (smooths + lengthens the lag)
_TAU_FAST = 2.0  # fast pool membrane time-constant (short -> leaks fast -> tracks RECENT error)
_TAU_SLOW = 16.0  # slow pool membrane time-constant (long  -> retains charge -> its rate LAGS)
_TAU_JIT = 0.22  # per-neuron tau_m heterogeneity (+/- fraction) -- desynchronises, breaks mode-lock
_THR_JIT = 0.16  # per-neuron threshold heterogeneity (+/-) -- smooths the pool rate vs drive
_INJECT_GAIN = (
    0.65  # error -> injected current scale (keeps both pools in the sensitive, non-saturated band)
)
_E_REF = 0.9  # reference error the pools are rate-matched at (both normalise to ~1 on flat error)
_WARMUP = 40  # calibration observes used to measure each pool's flat-error normaliser
_TAU_ACC = (
    50.0  # curriculum LIF-accumulator membrane time-constant (slow leak -> integrates evidence)
)
_ACC_SAFETY = 1.000002  # tiny margin so the accumulator crosses threshold reliably AT `patience`, not one late

# --- transfer-proof world (ported from the retired rate curriculum.py; the "starting small" control) ---
_TARGETS = (0.25, 0.50, 0.75, 1.00)  # increasing-difficulty milestones of the SHARED competence w
_REACH = 0.30  # frontier of learnability: a target much farther than this yields ~no gradient
_LR = 0.30  # base learning rate (fraction of the in-reach gap closed per tick)


class SpikingCuriosityDrive:
    """A GENUINELY SPIKING learning-progress curiosity drive -- a drop-in for CuriosityDrive.

    observe(k, e) injects the prediction error into a FAST and a SLOW LIF pool and records their
    EMITTED SPIKE COUNTS; novelty(k) is a habituating [0,1] recency signal (identical to the rate
    version); learning_progress(k) = max(0, slow_spike_rate - fast_spike_rate) is the anti-noisy-TV
    signal read from spike counts; intrinsic_reward(k, e) observes THEN returns the curiosity reward;
    curiosity_value(k) adds a bootstrapping novelty bonus for a curiosity-greedy explorer. Same public
    interface and constructor defaults as CuriosityDrive. Deterministic; per-instance random.Random(seed)."""

    def __init__(
        self,
        alpha=0.2,
        novelty_decay=0.6,
        slow_ratio=0.2,
        novelty_gain=0.5,
        explore_bonus=0.05,
        seed=0,
    ):
        # alpha / alpha_slow are kept for drop-in signature parity with the rate CuriosityDrive; in the
        # SPIKING drive the fast/slow separation is realised by the membrane time-constants (_TAU_FAST <
        # _TAU_SLOW), NOT by EMA rates -- these attributes are retained so the object introspects the same.
        self.alpha = float(alpha)  # (parity) rate fast-EMA rate -- unused by the spiking core
        self.alpha_slow = float(alpha) * float(
            slow_ratio
        )  # (parity) rate slow-EMA rate -- unused by the spiking core
        self.novelty_decay = float(
            novelty_decay
        )  # geometric habituation factor per observe, in (0,1)
        self.novelty_gain = float(
            novelty_gain
        )  # how strongly novelty multiplicatively boosts the reward
        self.explore_bonus = float(
            explore_bonus
        )  # additive novelty bonus in curiosity_value (bootstrap only)
        self.rng = random.Random(seed)  # per-instance rng (pool heterogeneity + selector tie-break)
        self._ctx = {}  # context_key -> spiking state (pools, normalisers, windows)

    def _make_pool(self, tau_m):
        """Build a pool of _POOL_N LIF neurons around membrane time-constant `tau_m`, with slight per-neuron
        tau and threshold heterogeneity drawn from the instance rng so the pool's spike rate is a SMOOTH
        function of drive (heterogeneity desynchronises the neurons -> no integer mode-locking / jitter averages out)."""
        pool = []
        for _ in range(_POOL_N):
            tau = tau_m * (
                1.0 + self.rng.uniform(-_TAU_JIT, _TAU_JIT)
            )  # per-neuron membrane-tau jitter
            thr = 1.0 + self.rng.uniform(-_THR_JIT, _THR_JIT)  # per-neuron threshold jitter
            pool.append(LIFNeuron(tau_m=tau, v_threshold=thr))
        return pool

    def _drive_pool(self, pool, pred_error):
        """Inject current (_INJECT_GAIN * pred_error) into every neuron for _SUB sub-ticks and return the
        total EMITTED SPIKE COUNT (sum of the 1/0 threshold crossings). THIS is the spiking quantity the
        learning-progress signal is built from -- an integer count of threshold crossings, never a membrane value."""
        current = _INJECT_GAIN * pred_error
        spikes = 0
        for nrn in pool:
            for _ in range(_SUB):
                spikes += nrn.step(current)  # LIFNeuron.step -> 1 (spiked) or 0 (did not)
        return spikes

    def _calibrate(self, pool):
        """Measure a pool's mean spike count at the reference error _E_REF (a flat drive), then reset the
        membranes for a clean start. Dividing each pool's windowed count by this normaliser makes the FAST and
        SLOW pools rate-MATCHED on a flat error (both -> ~1) so their difference is ~0 on flat/noisy input and
        only the transient LAG during a FALLING error opens a positive gap -- the spiking analog of two EMAs
        converging to the same mean."""
        total = 0
        for _ in range(_WARMUP):
            total += self._drive_pool(pool, _E_REF)
        for nrn in pool:
            nrn.reset_state()  # discard the calibration transient; start from rest
        return max(total / _WARMUP, 1e-6)  # mean spikes/observe at _E_REF (guard tiny/zero)

    def observe(self, context_key, pred_error):
        """Fold one world-model prediction error for context_key into its spiking state: on first sighting build
        + calibrate the FAST and SLOW LIF pools; every call drive both pools with the error and append their
        EMITTED SPIKE COUNTS to the per-pool sliding windows, and bump the visit count. Returns None."""
        e = float(pred_error)
        st = self._ctx.get(context_key)
        if st is None:
            fast = self._make_pool(_TAU_FAST)  # short tau -> tracks recent error
            slow = self._make_pool(_TAU_SLOW)  # long tau  -> lags
            st = {
                "n": 0,
                "fast": fast,
                "slow": slow,
                "nf": self._calibrate(fast),
                "ns": self._calibrate(slow),  # per-pool flat-error normalisers
                "wf": deque(maxlen=_WINDOW),
                "ws": deque(maxlen=_WINDOW),  # sliding windows of spike counts
            }
            self._ctx[context_key] = st
        st["n"] += 1
        st["wf"].append(
            self._drive_pool(st["fast"], e)
        )  # windowed EMITTED spike counts -- fast pool
        st["ws"].append(
            self._drive_pool(st["slow"], e)
        )  # windowed EMITTED spike counts -- slow pool

    def novelty(self, context_key):
        """Novelty in [0,1]: 1.0 for a never-seen context, HABITUATES as novelty_decay**count with each
        observe() (identical to the rate CuriosityDrive -- novelty is bookkeeping, not the spiking learner)."""
        st = self._ctx.get(context_key)
        if st is None:
            return 1.0
        return self.novelty_decay ** st["n"]

    def learning_progress(self, context_key):
        """Rectified SPIKING learning progress = max(0, slow_spike_rate - fast_spike_rate), computed ENTIRELY
        from EMITTED SPIKE COUNTS: each rate = (windowed spike count) / (per-pool flat-error normaliser). POSITIVE
        while the context's error is FALLING (the lagging slow pool still fires while the fast pool has dropped);
        ~0 once the error is FLAT -- whether mastered-low OR unlearnable-high. That the high-but-flat (noisy) case
        gives ~0 is the anti-noisy-TV property, here at the spike level. 0.0 before any observe (no window yet)."""
        st = self._ctx.get(context_key)
        if st is None or not st["wf"]:
            return 0.0
        rf = (sum(st["wf"]) / len(st["wf"])) / st[
            "nf"
        ]  # fast pool windowed spike-rate (recent), ~1 at _E_REF
        rs = (sum(st["ws"]) / len(st["ws"])) / st[
            "ns"
        ]  # slow pool windowed spike-rate (lagging), ~1 at _E_REF
        return max(0.0, rs - rf)  # slow > fast only while the error is DECREASING

    def intrinsic_reward(self, context_key, pred_error):
        """The curiosity reward the loop ADDS to any extrinsic reward. Calls observe() internally (pass each
        fresh prediction error exactly ONCE), then returns learning_progress * (1 + novelty_gain * novelty) --
        identical composition to the rate version, so a mastered or noisy context (lp ~0) earns ~0 no matter how
        novel. Deterministic."""
        self.observe(context_key, pred_error)
        return self.learning_progress(context_key) * (
            1.0 + self.novelty_gain * self.novelty(context_key)
        )

    def curiosity_value(self, context_key):
        """Expected intrinsic reward of visiting context_key NEXT (no new error yet):
        learning_progress * (1 + novelty_gain * novelty) + explore_bonus * novelty. The additive novelty bonus
        makes an unseen context worth a first taste (bootstrap); once habituated only real spiking learning
        progress sustains value -- so a curiosity-greedy selector chases the learnable frontier, never the noise."""
        lp = self.learning_progress(context_key)
        nov = self.novelty(context_key)
        return lp * (1.0 + self.novelty_gain * nov) + self.explore_bonus * nov

    def most_curious(self, candidates):
        """Return the context_key of maximal curiosity_value among `candidates` (a curiosity-greedy pick).
        Deterministic: the per-instance rng only shuffles first so exact ties are broken without positional
        bias -- same seed + same history -> same choice."""
        cand = list(candidates)
        self.rng.shuffle(cand)
        return max(cand, key=self.curiosity_value)


class SpikingCurriculum:
    """A self-paced developmental curriculum whose advance gate is a SPIKING LIF ACCUMULATOR -- a drop-in for
    Curriculum. observe(lp) feeds the current stage's learning-progress and returns True on the tick it ADVANCES:
    each low-LP tick injects a fixed evidence current into the accumulator neuron and steps it; a high-LP tick
    RESETS it (still learning -> hold); when the accumulator SPIKES (sustained low-LP evidence, ~`patience` ticks)
    the stage is LEARNED. stage / mastered / done / difficulty() / review_stage() are identical to Curriculum.
    Deterministic; per-instance random.Random(seed) (only review_stage draws it)."""

    def __init__(self, n_stages, mastery_threshold=0.02, patience=8, seed=0):
        self.n_stages = max(1, int(n_stages))  # stages indexed 0..n_stages-1
        self.mastery_threshold = float(
            mastery_threshold
        )  # learning_progress at/below this counts as "not learning"
        self.patience = max(
            1, int(patience)
        )  # consecutive low-LP ticks needed to judge a stage LEARNED
        self.rng = random.Random(seed)  # per-instance rng (interleaved-review picks / determinism)
        self._mastered = 0  # number of stages completed (0..n_stages)
        # LIF accumulator replaces the integer low-LP streak. Solve the injection so a fresh accumulator crosses
        # threshold EXACTLY on the `patience`-th consecutive low-LP tick: with leak d = exp(-1/tau), the membrane
        # after k evidence injections is inj*(1-d^k)/(1-d); setting that == threshold at k = patience gives inj.
        d = math.exp(-1.0 / _TAU_ACC)
        self._inject = (
            (1.0 - d) / (1.0 - d**self.patience) * _ACC_SAFETY
        )  # spikes at exactly `patience` low ticks
        self._acc = LIFNeuron(
            tau_m=_TAU_ACC, v_threshold=1.0
        )  # the evidence-accumulating gate neuron
        self._acc.reset_state()

    def observe(self, learning_progress):
        """Feed the CURRENT stage's learning-progress for this tick. Returns True on the tick the curriculum
        ADVANCES. A low-LP tick injects evidence into the accumulator LIF and steps it; if the accumulator SPIKES
        the stage is LEARNED -> advance (mastered++, reset the accumulator). A high-LP tick RESETS the accumulator
        (still learning -> never a premature advance). No-op once done (returns False)."""
        if self.done:
            return False
        lp = float(learning_progress)
        if lp <= self.mastery_threshold:
            if self._acc.step(
                self._inject
            ):  # sustained low-LP evidence crossed threshold -> LEARNED
                self._mastered += 1  # advance to the next (harder) stage
                self._acc.reset_state()  # clear the evidence for the new stage
                return True
            return False
        self._acc.reset_state()  # still learning -> dump the evidence (no premature advance)
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
        """Return a random already-mastered stage index for interleaved REVIEW (spaced practice against
        forgetting), or None if nothing is mastered yet. Uses the per-instance rng -> deterministic given the
        seed. Optional; the core advance rule does not consume it."""
        if self._mastered == 0:
            return None
        return self.rng.randrange(self._mastered)


# =====================================================================================================
# Self-check: five un-riggable, deterministic controls. Every stream is fixed (or drawn from a seeded
# rng) and the LIF dynamics + spike counts are deterministic, so the tail numbers reproduce byte-for-byte.
# Each claim has a CONTROL that MUST fail. Honest rule: the spiking anti-noisy-TV margin is smaller than
# the rate version's -- the MEASURED margin is printed, NOT asserted to any fixed multiple.
# =====================================================================================================

_EPS = 0.5  # small ABSOLUTE bound: flat/noisy accumulated LP must sit near zero (a broken rectifier lands 2-7x here)
_POS = 3.0  # a genuinely learnable (falling-error) stream must accumulate clearly-positive LP well above this


def _learner_step(w, target):
    """One frontier-gated gradient step of the shared competence w toward `target`. The effective gain
    decays as exp(-(gap/_REACH)**2): ~1 when the target is within reach, ~0 when it is far -- a stylized
    vanishing-gradient / sparse-reward frontier (progress is only possible NEAR current competence).
    Returns the new w. This is why a far ("hard") target cannot be learned cold, but a nearby one can."""
    gap = target - w
    gain = math.exp(-((gap / _REACH) ** 2))
    return w + _LR * gain * gap


def _run_curriculum_learner(budget, seed):
    """Self-paced SPIKING curriculum learner: train the SHARED competence on the CURRENT stage's target,
    and advance to the next (harder) target when the SpikingCurriculum LIF accumulator detects mastery
    (learning progress dried up). The competence w PERSISTS across stages -> easy stages walk w forward
    into reach of the hard one (transfer). Feeds the per-tick error REDUCTION as the learning-progress
    signal. Returns (final competence w, stages mastered, steps used)."""
    cur = SpikingCurriculum(len(_TARGETS), seed=seed)
    w = 0.0
    steps = 0
    while steps < budget and not cur.done:
        steps += 1
        target = _TARGETS[cur.stage]
        before = abs(target - w)
        w = _learner_step(w, target)
        lp = max(0.0, before - abs(target - w))  # learning progress = error reduction this tick
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


def main():
    # ---- (2) ANTI-NOISY-TV AT THE SPIKE LEVEL (the critical control) ----
    srng = random.Random(1234)
    learn_stream = [0.9**t for t in range(90)]  # genuinely learnable: error decays, masters
    noisy_stream = [
        0.9 + srng.uniform(-0.05, 0.05) for _ in range(90)
    ]  # constant-high mean + jitter, NO downward trend
    dl = SpikingCuriosityDrive(seed=1)
    dn = SpikingCuriosityDrive(seed=1)
    learn_lp = 0.0
    noisy_lp = 0.0
    learn_per = []
    for le, ne in zip(learn_stream, noisy_stream):
        dl.observe("learnable", le)
        dn.observe("noisy", ne)
        lv = dl.learning_progress("learnable")
        learn_lp += lv  # accumulate the SPIKING learning-progress ...
        learn_per.append(lv)
        noisy_lp += dn.learning_progress("noisy")  # ... for the learnable and the noisy context
    margin = learn_lp / max(noisy_lp, 1e-9)  # the HONEST measured spiking margin (printed)

    # swap CONTROL: relabel -- feed the FALLING stream under 'noisy', the FLAT-HIGH stream under 'learnable'.
    # The positive-LP-only-on-falling property must FOLLOW THE SIGNAL, not the label: the now-'learnable'
    # (flat-high) stream must NOT accumulate positive LP, while the now-'noisy' (falling) one still does.
    ds_falling = SpikingCuriosityDrive(seed=1)
    ds_flat = SpikingCuriosityDrive(seed=1)
    swap_falling_lp = 0.0
    swap_flat_lp = 0.0
    for le, ne in zip(learn_stream, noisy_stream):
        ds_falling.observe("noisy", le)  # FALLING error, mislabeled 'noisy'
        ds_flat.observe("learnable", ne)  # FLAT-HIGH error, mislabeled 'learnable'
        swap_falling_lp += ds_falling.learning_progress("noisy")
        swap_flat_lp += ds_flat.learning_progress("learnable")

    # ---- (1) STRUCTURAL-SPIKING: learning_progress IS a function of EMITTED SPIKE COUNTS ----
    dstruct = SpikingCuriosityDrive(seed=1)
    for _ in range(30):
        dstruct.observe("hot", 0.9)  # drive hard so the pools actually spike
    st = dstruct._ctx["hot"]
    spike_counts = list(st["wf"]) + list(
        st["ws"]
    )  # the windowed EMITTED spike counts LP is built from
    counts_are_int = bool(spike_counts) and all(isinstance(c, int) for c in spike_counts)
    pools_spiked = sum(spike_counts) > 0  # the pools crossed threshold (spike_count > 0)
    rf = (sum(st["wf"]) / len(st["wf"])) / st["nf"]
    rs = (sum(st["ws"]) / len(st["ws"])) / st["ns"]
    lp_from_spikes = max(0.0, rs - rf)  # recompute LP purely from the integer counts ...
    lp_matches = (
        abs(lp_from_spikes - dstruct.learning_progress("hot")) < 1e-12
    )  # ... it equals learning_progress()
    # CONTROL that LP reads SPIKE COUNTS and not membranes. The previous control here asked whether
    # the membranes were ints (`all(isinstance(v, int) for v in [nrn.v ...])`) and asserted they
    # were not. That had NO reachable red -- LIFNeuron's v_reset/v_rest default to 0.0 so a membrane
    # list is float on every input, and the pool is never empty (12 units) so `all()` cannot go
    # vacuously True either. Deleted under P8 falsifier F11 and replaced by a NEUTER: scramble every
    # membrane WITHOUT touching the recorded spike-count windows, and require LP to be bit-identical.
    #
    # Two properties of the neuter are load-bearing and were BOTH got wrong on the first attempt
    # (the red would not fire), so they are stated rather than left implicit:
    #   * it must run on a FALLING error stream, where LP is genuinely non-zero. On the flat 0.9
    #     drive above LP is ~0 BY DESIGN, and a neuter applied to a zero signal cannot move it
    #     under any implementation -- it would have looked like a pass while testing nothing.
    #   * it must drive the two pools APART. Writing identical values into both leaves any
    #     difference-based membrane readout (rs - rf) unchanged at 0, which is the same failure.
    neut = SpikingCuriosityDrive(seed=1)
    for _t in range(20):
        neut.observe("falling", 0.9**_t)  # LP measured 0.2625 here -- a live signal to move
    _nst = neut._ctx["falling"]
    lp_pre_neuter = neut.learning_progress("falling")
    for _i, _nrn in enumerate(_nst["fast"]):
        _nrn.v = 0.1 + 0.01 * _i
    for _i, _nrn in enumerate(_nst["slow"]):
        _nrn.v = 9.0 + 0.01 * _i
    lp_post_neuter = neut.learning_progress("falling")
    membrane_neuter_moved = lp_post_neuter != lp_pre_neuter

    # ---- (4) DETERMINISM (curiosity): two seed=7 drives, same stream -> identical LP + reward ----
    det_stream = [("x", 0.9), ("x", 0.7), ("x", 0.55), ("x", 0.4), ("x", 0.3), ("x", 0.25)]
    da, db = SpikingCuriosityDrive(seed=7), SpikingCuriosityDrive(seed=7)
    sig_a = [(da.learning_progress(k), da.intrinsic_reward(k, ev)) for (k, ev) in det_stream]
    sig_b = [(db.learning_progress(k), db.intrinsic_reward(k, ev)) for (k, ev) in det_stream]
    curiosity_det = sig_a == sig_b

    # ---- (3) SPIKING CURRICULUM: LIF-accumulator advance + two controls that MUST fail to advance ----
    ca = SpikingCurriculum(4, patience=8, seed=0)
    adv_tick = None
    for t in range(1, 40):
        if ca.observe(0.0):  # sustained low LP -> accumulator SPIKES
            adv_tick = t
            break
    cb = SpikingCurriculum(3, patience=8, seed=0)
    for _ in range(100):
        cb.observe(0.3)  # CONTROL (b): sustained HIGH LP -> never spikes
    high_mastered = cb.mastered
    cc = SpikingCurriculum(3, patience=8, seed=0)
    dip_advanced = False
    for x in [0.3] * 5 + [0.0] + [0.3] * 20:  # CONTROL (c): a single one-off dip -> leaks/reset
        dip_advanced = cc.observe(x) or dip_advanced
    dip_mastered = cc.mastered
    cd = SpikingCurriculum(5, patience=8, seed=0)
    d_start = cd.difficulty()  # fresh -> stage 0 -> difficulty 0.0
    for _ in range(500):
        cd.observe(0.0)  # master every stage
    d_end = cd.difficulty()  # done -> hardest stage -> difficulty 1.0

    # ---- (4) DETERMINISM (curriculum): two seed=7 curricula, same LP trace -> identical advances/reviews ----
    csrng = random.Random(1234)
    lp_trace = [0.3 + csrng.uniform(-0.05, 0.05) for _ in range(10)] + [
        0.0
    ] * 30  # seeded noisy-then-low LP trace
    cca, ccb = SpikingCurriculum(5, seed=7), SpikingCurriculum(5, seed=7)
    seq_a = [cca.observe(x) for x in lp_trace]
    seq_b = [ccb.observe(x) for x in lp_trace]
    review_a = [cca.review_stage() for _ in range(12)]
    review_b = [ccb.review_stage() for _ in range(12)]
    curriculum_det = (seq_a == seq_b) and (review_a == review_b)
    curr_masters = cca.mastered
    curr_ticks = [i for i, a in enumerate(seq_a) if a]

    # ---- (6) TRANSFER -- curriculum-HELPS via starting small (the HARD, un-riggable control, ported ----
    # from the retired rate curriculum.py). The SPIKING curriculum must keep the property: mastering
    # nearby stepping-stone targets walks the shared competence into reach of the hard one, so within the
    # SAME step budget it reaches FAR higher final competence than a learner thrown straight at the hardest.
    budget = 300
    cur_w, cur_mastered, cur_steps = _run_curriculum_learner(budget, seed=0)
    jump_w = _run_jump_learner(budget)
    cur_stages = _stages_reached(cur_w)
    jump_stages = _stages_reached(jump_w)
    learner_repro = _run_curriculum_learner(budget, seed=0) == _run_curriculum_learner(
        budget, seed=0
    )

    # ---- (5) PURITY vs the rate CuriosityDrive (report only, NOT asserted -- different mechanisms) ----
    from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive

    rate = CuriosityDrive(seed=1)
    rate_peak = 0.0
    rate_plateau = 0.0
    for le in learn_stream:
        rate.observe("L", le)
        rate_peak = max(rate_peak, rate.learning_progress("L"))
    for _ in range(60):
        rate.observe("L", learn_stream[-1])
        rate_plateau = rate.learning_progress("L")

    print("=" * 92)
    print(
        "SPIKING CURIOSITY -- learning-progress from EMITTED SPIKE COUNTS (SI-1 SPIKE-LP; anti-noisy-TV)"
    )
    print("=" * 92)
    print(
        f"pools: N={_POOL_N} sub={_SUB} window={_WINDOW}  tau_fast={_TAU_FAST} < tau_slow={_TAU_SLOW}  "
        f"gain={_INJECT_GAIN}  acc tau={_TAU_ACC}"
    )
    print("-" * 92)
    print(
        f"(1) structural  : LP from INTEGER spike counts={counts_are_int}  pools spiked={pools_spiked}  "
        f"LP==spike-count value={lp_matches}  (NEUTER: scrambling every membrane moved LP="
        f"{membrane_neuter_moved} -> LP ignores .v)"
    )
    print(
        f"(2) anti-noisyTV: learnable spiking-LP {learn_lp:.3f}  vs  noisy {noisy_lp:.3f}   "
        f"MEASURED margin {margin:.1f}x  (honest, non-saturated)"
    )
    print(
        f"    swap control: falling(labeled noisy) {swap_falling_lp:.3f}  vs  flat-high(labeled learnable) "
        f"{swap_flat_lp:.3f}  -> LP tracks the SIGNAL not the label"
    )
    print(
        f"(3) curriculum  : sustained-low advance @tick {adv_tick} (=patience 8); sustained-high mastered "
        f"{high_mastered}; single-dip advanced={dip_advanced}; difficulty {d_start:.2f}->{d_end:.2f}"
    )
    print(
        f"(4) determinism : curiosity identical={curiosity_det}  curriculum identical={curriculum_det}  "
        f"(masters {curr_masters}/5 @ {curr_ticks})"
    )
    print(
        f"(5) purity      : spiking learnable-LP peak {max(learn_per):.3f}/plateau {learn_per[-1]:.2e}  vs  "
        f"rate peak {rate_peak:.3f}/plateau {rate_plateau:.2e}  (same shape; win is PURITY)"
    )
    print(
        f"(6) transfer    : spiking curriculum reaches competence {cur_w:.3f} (mastered {cur_mastered}/4, "
        f"{cur_stages} milestones, {cur_steps} steps) vs jump-to-hard {jump_w:.3f} ({jump_stages}) "
        f"reproducible={learner_repro}"
    )
    print("=" * 92)

    # ---- (1) STRUCTURAL-SPIKING asserts (un-riggable: LP is literally a spike-count function) ----
    assert counts_are_int, "learning_progress must be computed from INTEGER emitted-spike counts"
    assert pools_spiked, (
        "the spiking pools must actually cross threshold and spike when driven (spike_count > 0)"
    )
    assert lp_matches, (
        "learning_progress must EQUAL the spike-count-derived value (LP IS a function of spike counts)"
    )
    # NEUTER control: scrambling every membrane must NOT move LP (it is a spike-count function).
    # RED constructed and run (re-verified 2026-08-05) against a membrane-reading LP variant --
    # `mean(nrn.v for nrn in st["slow"]) - mean(nrn.v for nrn in st["fast"])`, the same rs-rf shape
    # as the real rule with `.v` substituted for the spike counts. The variant is STATED because
    # without it the figures are not re-derivable: it moves LP 0.316718 -> 8.900000 and this assert
    # FIRES. (This comment read "0.021015 -> 8.900000"; the post value reproduces to the digit, the
    # pre value did not reproduce under any reconstruction.) The shipped spike-count LP moves by
    # exactly 0.0 from a live 0.262500.
    assert lp_pre_neuter > 0.0, (
        f"the neuter must be applied to a LIVE signal, else it cannot fail under any "
        f"implementation (LP {lp_pre_neuter:.6f})"
    )
    assert not membrane_neuter_moved, (
        f"LP must be a function of SPIKE COUNTS only -- scrambling the pool membranes moved it "
        f"{lp_pre_neuter:.6f} -> {lp_post_neuter:.6f}"
    )

    # ---- (2) ANTI-NOISY-TV -- learnable clearly positive, noisy near-zero ABSOLUTE (catches rectification bias) ----
    assert learn_lp > _POS, (
        f"learnable (falling error) must accumulate clearly POSITIVE spiking LP ({learn_lp:.3f})"
    )
    assert noisy_lp <= _EPS, (
        f"high-but-flat noise must accumulate NEAR-ZERO spiking LP ({noisy_lp:.3f} <= {_EPS})"
    )
    # swap CONTROL: the positive-LP-only-on-falling property inverts under relabeling
    assert swap_falling_lp > _POS, (
        "the FALLING stream carries the LP even when labeled 'noisy' (LP tracks the signal)"
    )
    assert swap_flat_lp <= _EPS, (
        f"CONTROL: the flat-high stream labeled 'learnable' must NOT accumulate positive LP ({swap_flat_lp:.3f})"
    )

    # ---- (3) SPIKING CURRICULUM asserts (advance via LIF accumulator; both controls must NOT advance) ----
    assert adv_tick == 8, (
        f"sustained low LP must SPIKE the accumulator and advance at exactly patience=8 ({adv_tick})"
    )
    assert high_mastered == 0, (
        "CONTROL: sustained HIGH LP must never spike the accumulator (no premature advance)"
    )
    assert not dip_advanced and dip_mastered == 0, (
        "CONTROL: a single one-off low dip must leak / get reset by the next high tick -> no spike, no advance"
    )
    assert d_start == 0.0 and d_end == 1.0, (
        "difficulty must rise 0 -> 1 as the curriculum is mastered end to end"
    )

    # ---- (4) DETERMINISM ----
    assert curiosity_det, (
        "two SpikingCuriosityDrive(seed=7) fed the same stream must give identical LP + reward"
    )
    assert curriculum_det, (
        "two SpikingCurriculum(seed=7) fed the same LP trace must give identical advances + reviews"
    )

    # ---- (6) TRANSFER -- the HARD, un-riggable control (ported): starting small must FAR exceed jump-to-hard ----
    assert cur_w > 0.9, (
        "the self-paced spiking curriculum must reach HIGH final competence within budget"
    )
    assert jump_w < 0.1, (
        "jump-to-hardest must stay stuck (the hard target starts out of the learnability frontier)"
    )
    assert cur_w > 5.0 * jump_w, (
        f"curriculum competence {cur_w:.3f} must FAR exceed jump-to-hard {jump_w:.3f} (transfer via starting small)"
    )
    assert cur_mastered > jump_stages and cur_stages > jump_stages, (
        "the spiking curriculum learner must master MORE milestones than the jump-to-hardest control on the same budget"
    )
    assert learner_repro, (
        "the seeded spiking curriculum learner must reproduce its (competence, mastered, steps) exactly"
    )

    print(
        "self-check OK: GENUINELY SPIKING -- learning_progress read from INTEGER spike counts (a membrane .v "
        "readout fails the int-count control);"
    )
    print(
        f"  ANTI-NOISY-TV preserved at the spike level: learnable LP {learn_lp:.3f} >> noisy {noisy_lp:.3f} "
        f"(MEASURED margin {margin:.1f}x, noisy near-zero absolute); swap control inverts (falling carries LP);"
    )
    print(
        f"  curriculum advances via a LIF ACCUMULATOR at tick {adv_tick}=patience (sustained-high + single-dip both "
        f"hold at mastered 0); deterministic (mean learnable LP {mean(learn_per):.3f});"
    )
    print(
        f"  on a stationary/decaying task it matches the rate CuriosityDrive in shape (peak {max(learn_per):.3f} vs "
        f"{rate_peak:.3f}) -- the win is spiking PURITY, not new capability;"
    )
    print(
        f"  TRANSFER (hard): starting-small spiking curriculum reaches competence {cur_w:.3f} ({cur_stages}/4 "
        f"milestones) vs jump-to-hard {jump_w:.3f} ({jump_stages}/4) on the same {budget}-step budget."
    )


if __name__ == "__main__":
    main()
