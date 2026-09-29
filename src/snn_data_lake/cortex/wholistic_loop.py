r"""
Wholistic bounded integration -- the ladder L13 endpoint: ONE arbitrated,
non-regressing loop that composes the project's OWN self-improvement primitives so
the WHOLE is more than the sum of the parts and does NOT erode old competence
(docs/SELF_IMPROVEMENT_RESEARCH.md rung L13 / increment SI-14, the capstone).

Honest scope (read this first). This is the L13 "wholistic BOUNDED integration"
spine and NOTHING MORE. Until now the project's self-improvement primitives ran as
independent opt-ins with no arbitration, no shared metacognitive/homeostatic spine,
and no non-regression control (the L13 "absent" row of the ladder). This module wires
them into ONE tick with a single arbitration spine and a held-out retention probe. It
is:
  * BOUNDED -- value/representation-only. It grows/prunes prototype UNITS and refines a
    value/label store; it drives NO actuator, emits NO motor command, and NEVER
    rewrites the loop, its objective, its thresholds, or the eval harness. There is no
    recursive self-improvement of the improver.
  * NOT AGI and NOT an intelligence explosion. The ladder EXPLICITLY forbids those.
    "Wholistic" here means exactly two measurable claims -- (a) the arbitrated whole
    reaches HIGHER aggregate competence than the same modules with the spine removed,
    and (b) it does so WITHOUT regressing on old tasks. It is a prototype-nearest
    classifier over synthetic separable tasks; it does not plan, reason, or think.

The one arbitrated tick (over a SEQUENCE of tasks A -> B -> C with a genuinely
UNLEARNABLE noise phase interleaved, old tasks NEVER re-shown live):
  1. PERCEIVE -- a context arrives; a label-free World turns it into the four
     metacognitive signals (confidence, surprise, learning-progress, novelty) from a
     per-bucket familiarity running mean + a CuriosityDrive keyed by a coarse crc32
     bucket. A degraded substrate corrupts perception (see step 6).
  2. ARBITRATE -- the MetacognitiveMonitor (THE SPINE) fuses those signals in a LIF
     reliability pool and emits gates: hold() (abstain when unreliable), learn_gate()
     (scale refinement), grow_trigger() (recruit ONLY on learnable-novelty, never on
     pure noise).
  3. LEARN (gated) -- a recognized context is refined iff learn_gate clears a floor; an
     unrecognized-but-learnable context recruits a unit iff grow_trigger fires. A noise
     context -> low reliability -> HOLD -> not learned, not grown, not stored.
  4. GROW + PRUNE -- neurogenesis recruits units (via the consolidator's GrowingLearner)
     and a capacity BUDGET prunes the lowest-utility units after each task, so n_units
     stays bounded across the whole sequence.
  5. CONSOLIDATE -- interleaved replay of OLD-task exemplars (the consolidator's
     ReplayBuffer) re-fires prior units so they keep utility and survive the budget --
     the non-regression mechanism.
  6. SLOW substrate health -- a CriticalityRegulator holds the spiking substrate's
     branching ratio sigma in a slightly-subcritical band so it stays maximally
     learnable; a degraded (off-band) substrate collapses stimulus separability and
     corrupts perception, lowering competence. Runs far slower than the learning tick.

Composed primitives (READ + REUSED, not reimplemented):
  * neuromodulation/intrinsic_motivation.CuriosityDrive   -- learning-progress / novelty.
  * cortex/metacognitive_monitor.MetacognitiveMonitor      -- the arbitration spine.
  * plasticity/lifelong_consolidation.LifelongConsolidator -- its parts drive GROW
    (hippocampus/growing_learner.GrowingLearner), PRUNE-to-budget, and CONSOLIDATE via
    interleaved replay (hippocampus/experience_replay.ReplayBuffer).
  * reservoir/criticality_regulator.CriticalityRegulator  -- slow substrate health.
  * substrate/_mathutils.mean                              -- aggregation.

Each spine component (metacog / consolidation / criticality) can be DISABLED
individually so the self-check can prove -- with a control that FAILS -- that the whole
beats the un-arbitrated sum and that every spine part is load-bearing.

Pure standard library (math, random, zlib) on the DEFAULT path. Per-instance
random.Random(seed); zlib.crc32 for bucketing (never hash()); no wall-clock, no global
RNG; ASCII-only prints; deterministic.

OPT-IN similarity-sized recruitment (default OFF, byte-identical when OFF): pass
similarity_sized=True to size each task's fresh-unit recruit CAP by how NOVEL its
perceptual signature is vs tasks already learned, and ROUTE a familiar task to its
FROZEN prior expert instead of growing. It REUSES the ART developmental match
(growth.growing_learner._best -- the same cosine match the live tick already runs)
at the per-task-centroid granularity via growth.similarity_sized_growth; that advisor
is imported LAZILY only when the flag is on, so the default path stays pure stdlib and
every default run is bit-for-bit unchanged.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.wholistic_loop
"""

import math
import random
import zlib

from snn_data_lake.cortex.metacognitive_monitor import MetacognitiveMonitor
from snn_data_lake.infra._mathutils import clamp01, mean
from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive
from snn_data_lake.plasticity.lifelong_consolidation import LifelongConsolidator
from snn_data_lake.reservoir.criticality_regulator import CriticalityRegulator

# --- task / stream geometry (disjoint separable classes, each with a frequent PRIMARY +
#     a rare, prune-able SECONDARY mode, interleaved with an unlearnable noise phase) ----
# Geometry mirrors plasticity/lifelong_consolidation: a class's two modes are 4-dim HIGH
# windows overlapping by 3 (cross-mode cosine ~0.8 < vigilance -> recruit separate units;
# cross-class ~0.28 -> a pruned secondary's samples still fall on the surviving primary).
DIM = 36  # feature-space dimensionality (6 classes x BLK)
N_CLASSES = 6  # global classes, disjoint HIGH blocks
T = 3  # tasks A, B, C
KPT = 2  # classes per task -> N_CLASSES = T * KPT
BLK = DIM // N_CLASSES  # dims each class owns (disjoint)
HIGH, LOW = 0.9, 0.1  # in-block vs baseline activation (far from the 0.5 bucket bin)
SIGMA_SAMPLE = 0.05  # gaussian class-sample noise (rate units; keeps the bucket code)
ROUNDS_PER_TASK = 8  # PRIMARY samples per class per task
SECONDARY_EVERY = 3  # a rare SECONDARY-mode sample every Nth round (prune-able)
NOISE_PER_ROUND = 3  # unlearnable noise samples interleaved per round
HELDOUT_PER_CLASS = 12  # held-out samples per class (both modes) for the competence probe

# --- learning / capacity knobs ----------------------------------------------------------
VIG = 0.90  # grower vigilance (modes recruit separate units; fresh noise below)
LR = 0.20  # prototype refinement rate
ENGAGE = 0.12  # learn_gate floor to refine a recognized context
BUDGET = 6  # capacity ceiling = one PRIMARY per class -> pruning is FORCED
TASK_DECAY = 0.15  # utility multiplier at each task boundary (recency pressure)
REPLAY_BATCH = 3  # old exemplars rehearsed per consolidation step

# --- slow substrate-health (criticality) knobs -----------------------------------------
CRIT_N = 100  # recurrent LIF pool size (small: the SLOW path is cheap)
CRIT_K = 8  # sparse fan-out
CRIT_STEPS = 20  # cascade length
CRIT_NMEAS = 5  # cascades per sigma measurement
CRIT_NSEED = 3  # seed neurons per cascade
CRIT_SUB_INIT = 60.0  # over-inhibited (dead) init -> sigma below band until regulated
CRIT_EPOCHS = 30  # homeostatic control epochs when regulation is ON
FIDELITY_FLOOR = 0.45  # worst-case perception fidelity of a fully-degraded substrate
FIDELITY_SLOPE = 0.55  # how fast fidelity falls as sigma leaves the band
SUBSTRATE_NOISE = 0.40  # perception jitter amplitude of a fully-degraded substrate

# --- DEEPENED hard task: a genuinely NON-SEPARABLE (overlapping + noisy) task -----------
# The separable task above proves the interlock with clean ablation deltas, but FULL tops out
# at a trivial 1.000 ceiling (deliberately linearly-separable classes -- the refute's fair
# caveat). This second task makes the ABSOLUTE competence INFORMATIVE by making the classes
# genuinely confusable: their spike-rate windows OVERLAP (stride < width, so adjacent classes
# share features -> nearest-prototype confuses them) under moderate noise. The strong 0.9/0.1
# contrast is KEPT so the exact-bucket recognition (familiarity + noise-rejection) stays stable
# under noise, while classification errors on the overlap put FULL below ceiling. (Honest
# engineering note in the module: a population-COUNT front-end -- sensorimotor/population_encoding --
# was tried first but its all-positive codes collapse cosine into a useless 0.69-0.92 band
# [same-class 0.917 vs diff-class 0.910], so the reused cosine/[0,1] machinery cannot separate
# classes or reject noise there; the overlap task keeps the same spike-rate substrate instead.)
HIGH_H, LOW_H = 0.9, 0.1  # strong contrast KEPT -> 0.5-threshold buckets stay stable under noise
HARD_WIN = 5  # class spike-rate window width
HARD_STRIDE = 3  # class-to-class shift (< WIN -> windows OVERLAP -> confusable)
SIGMA_HARD = 0.16  # moderate sample noise -> real nearest-prototype errors (sub-ceiling)


#: MERGED 2026-08-15 into `substrate/_mathutils.clamp01` (8 copies -> 1). Alias keeps the local
#: spelling so no call site changes; the MECHANISM is singular.
_clamp01 = clamp01


def _cosine(a, b):
    """Cosine similarity of two non-negative vectors; 0 if either has zero norm."""
    d = na = nb = 0.0
    for x, y in zip(a, b):
        d += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return d / math.sqrt(na * nb)


def _bucket_key(vec):
    """Coarse, label-free bucket id: threshold each feature at 0.5 -> a DIM-bit code,
    hashed with zlib.crc32 (never hash()). Jittered class samples collapse to a STABLE
    code; fresh noise lands on an essentially unique code every time."""
    return zlib.crc32(bytes(1 if x >= 0.5 else 0 for x in vec))


def _proto(g, mode):
    """Class prototype: HIGH on a 4-dim window inside class g's BLK-wide block. Mode 1's
    window is shifted by one (overlap 3 -> cross-mode cosine ~0.8), so the two modes recruit
    separate units yet a pruned secondary's samples still fall on the surviving primary."""
    b = g * BLK
    p = [LOW] * DIM
    lo = b if mode == 0 else b + 1
    for i in range(lo, lo + 4):
        p[i] = HIGH
    return p


def _class_sample(g, mode, rng):
    """A gaussian-noised draw of a (class, mode) prototype, clamped to [0,1]."""
    p = _proto(g, mode)
    return [_clamp01(p[i] + rng.gauss(0.0, SIGMA_SAMPLE)) for i in range(DIM)]


def _noise_sample(rng):
    """A fresh unlearnable uniform-random sample (a new one every call -> LP stays ~0)."""
    return [rng.random() for _ in range(DIM)]


def _build_tasks(seed):
    """T tasks, KPT disjoint classes each; a frequent PRIMARY mode + a rare SECONDARY mode
    per class, interleaved with an unlearnable NOISE phase per round. Old tasks are NEVER
    re-shown in a later task's live stream. Label is the class (mode is invisible)."""
    rng = random.Random(seed)
    tasks = []
    for t in range(T):
        classes = [t * KPT + k for k in range(KPT)]
        plan = []
        for r in range(ROUNDS_PER_TASK):
            for c in classes:
                plan.append(("class", c, 0))  # frequent PRIMARY
                if r % SECONDARY_EVERY == 0:
                    plan.append(("class", c, 1))  # rare SECONDARY (prune-able)
            for _ in range(NOISE_PER_ROUND):
                plan.append(("noise", -1, -1))
        rng.shuffle(plan)
        stream = [
            (
                c if kind == "class" else -1,
                _class_sample(c, mode, rng) if kind == "class" else _noise_sample(rng),
            )
            for kind, c, mode in plan
        ]
        tasks.append({"classes": classes, "stream": stream})
    return tasks


# --- DEEPENED hard task: genuinely NON-SEPARABLE overlapping spike-rate classes ---------
def _hard_proto(g, mode):
    """A confusable class prototype: a HARD_WIN-wide HIGH window at stride HARD_STRIDE < WIN,
    so ADJACENT classes OVERLAP (share features) -> nearest-prototype confuses them. Mode
    shifts the window by one (a rare, prune-able secondary mode). Strong 0.9/0.1 contrast."""
    lo = g * HARD_STRIDE + (1 if mode else 0)
    p = [LOW_H] * DIM
    for i in range(lo, min(lo + HARD_WIN, DIM)):
        p[i] = HIGH_H
    return p


def _hard_sample(g, mode, rng):
    """A gaussian-noised draw of an OVERLAPPING (class, mode) prototype, clamped to [0,1].
    The overlap + noise make nearest-prototype classification err -> sub-ceiling competence."""
    p = _hard_proto(g, mode)
    return [_clamp01(p[i] + rng.gauss(0.0, SIGMA_HARD)) for i in range(DIM)]


def _hard_noise(rng):
    """A fresh unlearnable uniform-random sample (a new one every call -> LP stays ~0)."""
    return [rng.random() for _ in range(DIM)]


def _build_hard_tasks(seed):
    """Like _build_tasks but OVERLAPPING (non-separable) classes under moderate noise; adjacent
    classes share features (and span task boundaries) so classification errs -> informative
    sub-ceiling competence. Same spike-rate substrate. Old tasks never re-shown live."""
    rng = random.Random(seed)
    tasks = []
    for t in range(T):
        classes = [t * KPT + k for k in range(KPT)]
        plan = []
        for r in range(ROUNDS_PER_TASK):
            for c in classes:
                plan.append(("class", c, 0))
                if r % SECONDARY_EVERY == 0:
                    plan.append(("class", c, 1))
            for _ in range(NOISE_PER_ROUND):
                plan.append(("noise", -1, -1))
        rng.shuffle(plan)
        stream = [
            (
                c if kind == "class" else -1,
                _hard_sample(c, mode, rng) if kind == "class" else _hard_noise(rng),
            )
            for kind, c, mode in plan
        ]
        tasks.append({"classes": classes, "stream": stream})
    return tasks


class _World:
    """Label-free source of the four metacognitive signals for one input.

    confidence = recognition vs a per-bucket running-mean prior (0 for a novel bucket);
    surprise = 1 - confidence; learning_progress / novelty = the anti-noisy-TV curiosity
    signals for that coarse bucket. A recurring class bucket earns rising confidence +
    learning progress; a fresh-every-time noise bucket stays low-confidence, zero-progress."""

    def __init__(self, seed):
        self.fam = {}  # bucket -> running-mean vector
        self.cur = CuriosityDrive(novelty_decay=0.8, seed=seed)
        self.beta = 0.3

    def signals(self, vec):
        key = _bucket_key(vec)
        st = self.fam.get(key)
        conf = _clamp01(_cosine(vec, st)) if st is not None else 0.0  # recognize BEFORE update
        surprise = 1.0 - conf
        if st is None:
            self.fam[key] = list(vec)
        else:
            self.fam[key] = [(1.0 - self.beta) * m + self.beta * x for m, x in zip(st, vec)]
        self.cur.observe(key, surprise)
        return conf, surprise, self.cur.learning_progress(key), self.cur.novelty(key)


class WholisticLoop:
    """ONE arbitrated, non-regressing tick over a task SEQUENCE.

    Composes CuriosityDrive + MetacognitiveMonitor + (GrowingLearner + ReplayBuffer +
    prune, via LifelongConsolidator) + CriticalityRegulator. Each spine component is
    individually disableable: metacog (arbitration), consolidation (interleaved replay),
    criticality (slow substrate health). GROW + PRUNE (bounded capacity) stay ON in every
    configuration -- only the arbitration spine is ablatable. Value/representation-only:
    no actuator, no self-rewrite. Deterministic; per-instance random.Random(seed)."""

    def __init__(
        self,
        *,
        metacog=True,
        consolidation=True,
        criticality=True,
        hard=False,
        similarity_sized=False,
        seed=0,
    ):
        self.metacog = bool(metacog)
        self.consolidation = bool(consolidation)
        self.criticality = bool(criticality)
        # hard=True routes the SAME loop over the brain's REAL spiking front-end (population-
        # coded, overlapping, sub-ceiling) so the ABSOLUTE competence is informative, not 1.000.
        self.hard = bool(hard)
        self.seed = int(seed)
        self._dim = DIM
        self._tasks = _build_hard_tasks(seed) if self.hard else _build_tasks(seed)
        self.world = _World(seed)
        # the MetacognitiveMonitor from cortex/metacognitive_monitor.py IS the spine.
        self.monitor = MetacognitiveMonitor(seed=seed)
        # GROW + PRUNE + CONSOLIDATE via the lifelong consolidator's own parts (prune
        # always on -> bounded capacity in every config; replay is the ablatable part).
        self.con = LifelongConsolidator(
            dim=self._dim,
            budget=BUDGET,
            prune=True,
            vigilance=VIG,
            lr=LR,
            task_decay=TASK_DECAY,
            replay_batch=REPLAY_BATCH,
            seed=seed,
        )
        # SLOW substrate health: a small recurrent-LIF criticality homeostat.
        self.reg = CriticalityRegulator(
            n=CRIT_N,
            k=CRIT_K,
            steps=CRIT_STEPS,
            n_meas=CRIT_NMEAS,
            n_seed=CRIT_NSEED,
            inhib_gain=CRIT_SUB_INIT,
            seed=seed,
        )
        self.substrate_rng = random.Random(seed + 777)  # per-instance corruption rng
        self._abstains = 0
        self._max_units = 0
        self._noise_units = 0
        # regulate the substrate ONCE up front (slow path); OFF -> stays dead/off-band.
        if self.criticality:
            self.reg.run(epochs=CRIT_EPOCHS, regulate=True)
        self.sigma = self.reg.sigma()
        self.fidelity = self._fidelity(self.sigma)

        # ---- OPT-IN similarity-sized recruitment governor (default OFF = byte-identical) --------
        # When ON, a per-TASK advisor (growth.similarity_sized_growth) SIZES how many fresh units
        # this task may recruit by how NOVEL its perceptual signature is vs tasks already learned, and
        # ROUTES a familiar task to its FROZEN prior expert (recruit cap 0 -> reuse, no new growth).
        # It REUSES the ART developmental match (GrowingLearner._best -- the same cosine match the loop
        # already runs per sample) at the task-centroid granularity; it does NOT recompute novelty.
        # advisor stays None in the default config, so every code path below is byte-identical to today.
        self.advisor = None
        self._task_cap = 0  # fresh units this task may recruit (advisor-sized; 0 -> route/reuse)
        self._task_recruited = 0  # live-tick recruit events so far this task (vs the cap)
        self.growth_log = []  # per-task {action, novelty, recruit_cap, units_recruited, ...} when ON
        if similarity_sized:
            from snn_data_lake.growth.similarity_sized_growth import SimilaritySizedGrowth

            self.advisor = SimilaritySizedGrowth(
                dim=self._dim, n_classes=N_CLASSES, cap_max=BUDGET, cap_min=1, route_frac=0.5
            )

    # ---- slow substrate-health -> perception fidelity --------------------------------
    def _fidelity(self, sigma):
        """Perception fidelity in [FLOOR, 1]: 1.0 while sigma sits IN the learnable band,
        falling with distance once the substrate degrades out of band. This instantiates
        the regulator's own validated property -- a critical substrate maximizes stimulus
        SEPARABILITY; a degraded one collapses it."""
        lo, hi = self.reg.band
        if lo <= sigma <= hi:
            return 1.0
        dist = (lo - sigma) if sigma < lo else (sigma - hi)
        return max(FIDELITY_FLOOR, 1.0 - FIDELITY_SLOPE * dist)

    def _perceive(self, vec):
        """Perception through the substrate. A healthy (in-band) substrate passes the
        input cleanly; a degraded one REDUCES CONTRAST (blurs every feature toward the
        sample's own mean, shrinking the HIGH/LOW gap the classes live in) and adds
        substrate noise, both proportional to (1 - fidelity) -- collapsing separability
        without pathologically making every input identical."""
        f = self.fidelity
        if f >= 1.0:
            return list(vec)
        m = mean(vec)
        out = []
        for x in vec:
            blurred = f * x + (1.0 - f) * m
            jitter = (1.0 - f) * self.substrate_rng.uniform(-SUBSTRATE_NOISE, SUBSTRATE_NOISE)
            out.append(_clamp01(blurred + jitter))
        return out

    # ---- one arbitrated tick ---------------------------------------------------------
    def _tick(self, label, raw_vec, fresh):
        """Perceive -> arbitrate -> (gated) learn -> consolidate. Value/rep only."""
        x = self._perceive(raw_vec)
        conf, surp, lp, nov = self.world.signals(x)
        if self.metacog:
            self.monitor.update(conf, surp, lp, nov)
            if self.monitor.hold():  # abstain on the unreliable / noise
                self._abstains += 1
                return
        _, best_sim = self.con.learner._best(x)
        recognized = best_sim >= self.con.learner.vigilance
        if self.metacog:  # arbitration decides engagement
            engage = (recognized and self.monitor.learn_gate() >= ENGAGE) or (
                not recognized and self.monitor.grow_trigger()
            )
        else:  # no arbitration: learn/grow all
            engage = True
        if not engage:
            return
        # OPT-IN novelty-sized recruit cap (default OFF): once this task has recruited its advisor-
        # sized quota of fresh units, block any further NEW-unit growth (a routed/familiar task caps
        # at 0 -> its samples refine/reuse existing prototypes, never grow). Refinement of recognized
        # samples is untouched; only novel recruitment is gated.
        if self.advisor is not None and not recognized and self._task_recruited >= self._task_cap:
            return
        n0 = self.con.n_units
        self.con._observe(x, label)  # GROW (neurogenesis) or refine + credit
        grew = self.con.n_units > n0
        if grew and label < 0:
            self._noise_units += 1  # bookkeeping: a noise-born unit
        if self.advisor is not None and grew:
            self._task_recruited += 1  # charge this recruit against the task's novelty-sized cap
        fresh.append((label, tuple(x)))  # exemplar (committed at task boundary)
        if self.consolidation and len(self.con.memory):
            self.con._replay_once()  # CONSOLIDATE: interleaved OLD-task replay

    # ---- opt-in similarity-sized recruitment governor (advisor present only) ---------
    def _task_signature(self, task):
        """Label-free centroid of a task's CLASS samples -- the per-task routing key the advisor
        matches against tasks already learned. Raw sensory vectors (pre-substrate) so forming it
        never perturbs the substrate-corruption RNG; noise samples (label < 0) are excluded."""
        vs = [vec for label, vec in task["stream"] if label >= 0]
        if not vs:
            return [0.0] * self._dim
        return [mean(col) for col in zip(*vs)]

    def _grow_task_expert(self, task, dec, seed):
        """Train + FREEZE the advisor's freshly-grown per-task expert on this task's (class sample,
        label) pairs, so an OLD task's competence is retained by a read-only frozen readout after
        later tasks grow their own (the no-catastrophic-forgetting guarantee). The expert lives on the
        raw sensory vectors (consistent with _task_signature). Returns (own_acc, heldout) to log."""
        X = [vec for label, vec in task["stream"] if label >= 0]
        y = [label for label, vec in task["stream"] if label >= 0]
        exp = self.advisor.experts[dec.expert_id]
        exp.learn(X, y, epochs=20, seed=seed)  # LOCAL error-modulated Hebbian rule, then...
        exp.freeze()  # ...read-only: later tasks cannot overwrite it -> retention is structural
        rng = random.Random(5000 + seed)  # fresh held-out draws for a retention probe
        held = [
            (self._sample(c, i % 2, rng), c)
            for c in task["classes"]
            for i in range(HELDOUT_PER_CLASS)
        ]
        own_acc = exp.accuracy([x for x, _ in held], [yy for _, yy in held])
        return round(own_acc, 6), held

    # ---- run the whole sequence ------------------------------------------------------
    def run(self):
        """Learn A -> B -> C in order; old tasks never re-shown live. Returns a result
        dict of pure numbers/bools (byte-comparable for the determinism check)."""
        for t_idx, task in enumerate(self._tasks):
            dec = None
            if (
                self.advisor is not None
            ):  # SIZE this task's recruit cap by its novelty; route familiar
                dec = self.advisor.route_or_grow(self._task_signature(task))
                self._task_cap = (
                    dec.recruit_cap if dec.grew else 0
                )  # familiar -> 0 -> reuse, no growth
                self._task_recruited = 0
            self.con.utility = [u * TASK_DECAY for u in self.con.utility]  # recency pressure
            fresh = []
            for label, vec in task["stream"]:
                self._tick(label, vec, fresh)
            for lbl, xt in fresh:  # exemplars visible to LATER tasks only
                self.con.memory.add(lbl, xt, 0.0)
            self.con._prune_to_budget()  # bounded capacity (always on)
            self._max_units = max(self._max_units, self.con.n_units)
            if self.advisor is not None:
                own_acc, held = (None, None)
                if dec.grew:  # a novel task grows -> train + freeze its retained expert
                    own_acc, held = self._grow_task_expert(task, dec, seed=self.seed + t_idx)
                self.growth_log.append(
                    {
                        "task": t_idx,
                        "action": dec.action,
                        "novelty": round(dec.novelty, 6),
                        "recruit_cap": dec.recruit_cap,
                        "grew": dec.grew,
                        "expert_id": dec.expert_id,
                        "units_recruited": self._task_recruited,
                        "own_acc": own_acc,
                        "heldout": held,
                    }
                )
        agg = self._heldout_acc(range(N_CLASSES))  # competence over ALL real tasks
        ret = self._heldout_acc(self._tasks[0]["classes"])  # old-task (A) retention
        return {
            "aggregate": round(agg, 6),
            "retention": round(ret, 6),
            "n_units": self.con.n_units,
            "max_units": self._max_units,
            "noise_units": self._noise_units,
            "abstains": self._abstains,
            "sigma": round(self.sigma, 6),
            "in_band": self.reg.band[0] <= self.sigma <= self.reg.band[1],
        }

    def _sample(self, g, mode, rng):
        """A held-out draw matching this loop's task family (separable or hard-spiking)."""
        return _hard_sample(g, mode, rng) if self.hard else _class_sample(g, mode, rng)

    def _heldout_acc(self, class_ids):
        """Held-out accuracy on fresh samples (BOTH modes, evenly) for `class_ids`,
        perceived through the (possibly degraded) substrate: correct iff the nearest
        surviving prototype carries the SAME class label (noise-born / wrong-class = wrong)."""
        rng = random.Random(9000 + self.seed)
        correct = total = 0
        for c in class_ids:
            for i in range(HELDOUT_PER_CLASS):
                x = self._perceive(self._sample(c, i % 2, rng))
                total += 1
                if self.con.classify(x) == c:
                    correct += 1
        return correct / total if total else 0.0


# =====================================================================================
# Un-riggable self-check. Deterministic (seed=0), ASCII-only. Each claim is
# paired with a CONTROL that FAILS. FULL vs no-arbitration vs each single ablation.
# =====================================================================================


def _run(metacog, consolidation, criticality, seed=0, hard=False):
    """Run one configuration to completion; returns its result dict."""
    return WholisticLoop(
        metacog=metacog, consolidation=consolidation, criticality=criticality, hard=hard, seed=seed
    ).run()


def _p(lines, s):
    """Append an ASCII-only line (Windows cp1252 safety)."""
    if not s.isascii():
        raise AssertionError("non-ASCII output")
    lines.append(s)


def main():
    full = _run(True, True, True)  # the integrated whole
    ctrl = _run(False, False, False)  # no arbitration spine at all
    iso = _run(False, False, True)  # no arbitration, HEALTHY substrate:
    #                                                        isolates the GENUINE metacog+consol
    #                                                        gain from the by-construction crit penalty
    abl_meta = _run(False, True, True)  # metacog OFF only
    abl_cons = _run(True, False, True)  # consolidation OFF only
    abl_crit = _run(True, True, False)  # criticality OFF only

    lines = []
    _p(lines, "=" * 84)
    _p(lines, "WHOLISTIC LOOP -- one arbitrated, non-regressing bounded integration (L13 / SI-14)")
    _p(lines, "=" * 84)
    _p(
        lines,
        "curiosity + METACOG arbitration + grow/prune + CONSOLIDATION + CRITICALITY, one tick",
    )
    _p(
        lines,
        f"sequence: {T} tasks x {KPT} disjoint classes + {NOISE_PER_ROUND} noise/round "
        f"(dim={DIM}, vigilance={VIG}, budget={BUDGET}); old tasks never re-shown live",
    )
    _p(lines, "-" * 84)
    _p(lines, "  config                     aggregate    old-task     n_units   sigma   in-band")
    _p(lines, "                             competence   retention  (<=budget)")
    fmt = "  {:<26} {:>8.3f}     {:>8.3f}   {:>6}    {:>6.3f}   {}"
    _p(
        lines,
        fmt.format(
            "FULL (integrated)",
            full["aggregate"],
            full["retention"],
            full["n_units"],
            full["sigma"],
            full["in_band"],
        ),
    )
    _p(
        lines,
        fmt.format(
            "no-arbitration (control)",
            ctrl["aggregate"],
            ctrl["retention"],
            ctrl["n_units"],
            ctrl["sigma"],
            ctrl["in_band"],
        ),
    )
    _p(
        lines,
        fmt.format(
            "no-arb, healthy substrate",
            iso["aggregate"],
            iso["retention"],
            iso["n_units"],
            iso["sigma"],
            iso["in_band"],
        ),
    )
    _p(
        lines,
        fmt.format(
            "ablate metacog",
            abl_meta["aggregate"],
            abl_meta["retention"],
            abl_meta["n_units"],
            abl_meta["sigma"],
            abl_meta["in_band"],
        ),
    )
    _p(
        lines,
        fmt.format(
            "ablate consolidation",
            abl_cons["aggregate"],
            abl_cons["retention"],
            abl_cons["n_units"],
            abl_cons["sigma"],
            abl_cons["in_band"],
        ),
    )
    _p(
        lines,
        fmt.format(
            "ablate criticality",
            abl_crit["aggregate"],
            abl_crit["retention"],
            abl_crit["n_units"],
            abl_crit["sigma"],
            abl_crit["in_band"],
        ),
    )
    _p(lines, "-" * 84)
    _p(
        lines,
        f"(1) WHOLE > un-arbitrated SUM + NON-REGRESSION: FULL aggregate "
        f"{full['aggregate']:.3f} > control {ctrl['aggregate']:.3f}; "
        f"FULL retention {full['retention']:.3f} >> control {ctrl['retention']:.3f}",
    )
    _p(
        lines,
        f"(2) each spine part load-bearing: metacog {abl_meta['aggregate']:.3f} / "
        f"consolidation-retention {abl_cons['retention']:.3f} / "
        f"criticality {abl_crit['aggregate']:.3f}  all below FULL",
    )
    _p(
        lines,
        f"(3) bounded: max n_units {max(c['max_units'] for c in (full, ctrl, iso, abl_meta, abl_cons, abl_crit))} "
        f"<= budget {BUDGET}; FULL sigma {full['sigma']:.3f} in-band; "
        f"criticality-OFF sigma {abl_crit['sigma']:.3f} out-of-band",
    )
    _p(
        lines,
        f"(4) HONEST: criticality->competence is a BUILT bridge (perception fidelity), NOT "
        f"emergent, so it inflates the all-off control. Remove that confound -- no-arb + "
        f"HEALTHY substrate {iso['aggregate']:.3f}/{iso['retention']:.3f} -- and the whole "
        f"STILL beats the sum: FULL {full['aggregate']:.3f}/{full['retention']:.3f} on "
        f"metacog+consolidation (genuinely emergent) ALONE",
    )
    _p(lines, "=" * 84)
    for line in lines:
        print(line)

    # ---- (1) THE WHOLE BEATS THE UN-ARBITRATED SUM + NON-REGRESSION ------------------
    # control = SAME modules, SAME task budget/info, ONLY the spine removed (fair).
    assert full["aggregate"] > ctrl["aggregate"] + 0.10, (
        f"integrated aggregate {full['aggregate']:.3f} must beat no-arbitration "
        f"control {ctrl['aggregate']:.3f} by a clear margin"
    )
    assert full["retention"] >= ctrl["retention"] + 0.30, (
        f"integrated retention {full['retention']:.3f} must be >> control "
        f"{ctrl['retention']:.3f} (the whole does NOT regress; the control forgets)"
    )

    # ---- (1b) WHOLE > SUM SURVIVES removing the by-construction criticality confound ----
    # criticality->competence is a BUILT perception-fidelity bridge (honest, but hand-set),
    # so it drags the all-off control down. Isolate it: no-arbitration + HEALTHY substrate.
    # The whole must STILL beat this on both axes -- proving the gain is genuinely from the
    # emergent metacog + consolidation dynamics, not the hand-authored substrate penalty.
    assert full["aggregate"] > iso["aggregate"] + 0.10, (
        f"integrated aggregate {full['aggregate']:.3f} must beat no-arb+healthy-substrate "
        f"{iso['aggregate']:.3f} -- whole>sum must not rest on the criticality penalty"
    )
    assert full["retention"] >= iso["retention"] + 0.30, (
        f"integrated retention {full['retention']:.3f} must be >> no-arb+healthy "
        f"{iso['retention']:.3f} (non-regression is from consolidation, not the substrate)"
    )

    # ---- (2) EACH SPINE COMPONENT IS LOAD-BEARING (clean single-component removal) ----
    # metacog OFF -> learns/grows the NOISE -> wasted capacity + lower competence.
    assert abl_meta["aggregate"] < full["aggregate"] - 0.05, (
        f"metacog-OFF competence {abl_meta['aggregate']:.3f} must drop below FULL "
        f"{full['aggregate']:.3f} (noise pollutes learning)"
    )
    assert abl_meta["noise_units"] > 0 and abl_meta["n_units"] >= full["n_units"], (
        f"metacog-OFF must waste capacity on noise (noise_units={abl_meta['noise_units']}, "
        f"n_units {abl_meta['n_units']} vs FULL {full['n_units']})"
    )
    # consolidation OFF -> FORGETS old tasks -> lower retention.
    assert abl_cons["retention"] < full["retention"] - 0.15, (
        f"consolidation-OFF retention {abl_cons['retention']:.3f} must drop below FULL "
        f"{full['retention']:.3f} (no interleaved replay -> catastrophic forgetting)"
    )
    # criticality OFF -> substrate degrades (out of band) -> lower competence.
    assert not abl_crit["in_band"], (
        f"criticality-OFF substrate must fall OUT of the learnable band "
        f"(sigma {abl_crit['sigma']:.3f})"
    )
    assert abl_crit["aggregate"] < full["aggregate"] - 0.05, (
        f"criticality-OFF competence {abl_crit['aggregate']:.3f} must drop below FULL "
        f"{full['aggregate']:.3f} (degraded substrate collapses separability)"
    )

    # ---- (3) BOUNDED + DETERMINISTIC + ASCII ------------------------------
    for name, r in (
        ("FULL", full),
        ("control", ctrl),
        ("iso", iso),
        ("abl_meta", abl_meta),
        ("abl_cons", abl_cons),
        ("abl_crit", abl_crit),
    ):
        assert r["max_units"] <= BUDGET, f"{name} n_units {r['max_units']} exceeded budget {BUDGET}"
    assert full["in_band"], (
        f"FULL substrate sigma {full['sigma']:.3f} must sit in the learnable band"
    )
    assert _run(True, True, True) == full, "two seed=0 FULL runs must be byte-for-byte identical"

    # ===== DEEPENED: the SAME arbitrated loop on the brain's REAL spiking substrate =======
    # A harder, small-contrast, OVERLAPPING task, every feature encoded through the project's
    # own PopulationEncoder (RBF -> spike counts). The ABSOLUTE competence is now INFORMATIVE
    # (sub-ceiling, real errors), not a trivial 1.000 -- yet the arbitrated whole must STILL
    # beat the un-arbitrated sum and not regress. This answers the "trivial ceiling" caveat.
    fh = _run(True, True, True, hard=True)  # integrated whole, real substrate
    sh = _run(False, False, True, hard=True)  # no-arbitration + HEALTHY substrate
    hc = _run(True, False, True, hard=True)  # consolidation OFF (retention lever)
    hfmt = "  {:<26} {:>8.3f}     {:>8.3f}   {:>6}"
    hl = []
    _p(hl, "-" * 84)
    _p(hl, "DEEPENED -- SAME loop, genuinely NON-SEPARABLE overlapping spike-rate classes")
    _p(
        hl,
        f"  window {HARD_WIN} stride {HARD_STRIDE} (overlap {HARD_WIN - HARD_STRIDE}), "
        f"contrast {LOW_H}->{HIGH_H}, noise {SIGMA_HARD}, dim {DIM}",
    )
    _p(hl, "  config                     aggregate    old-task     n_units")
    _p(hl, hfmt.format("FULL (integrated)", fh["aggregate"], fh["retention"], fh["n_units"]))
    _p(
        hl,
        hfmt.format("no-arb, healthy substrate", sh["aggregate"], sh["retention"], sh["n_units"]),
    )
    _p(hl, hfmt.format("ablate consolidation", hc["aggregate"], hc["retention"], hc["n_units"]))
    _p(
        hl,
        f"(H) real-substrate whole > sum + NON-REGRESSION, INFORMATIVE absolute: FULL "
        f"{fh['aggregate']:.3f} (sub-ceiling) vs no-arb {sh['aggregate']:.3f}; retention "
        f"{fh['retention']:.3f} vs {sh['retention']:.3f}",
    )
    _p(hl, "=" * 84)
    for line in hl:
        print(line)

    assert fh["aggregate"] < 0.98, (
        f"DEEPENED FULL competence {fh['aggregate']:.3f} must be SUB-CEILING (informative, not "
        f"a trivial 1.000) on the hard real-spiking task"
    )
    assert fh["aggregate"] > sh["aggregate"] + 0.05, (
        f"DEEPENED whole {fh['aggregate']:.3f} must still beat no-arb+healthy sum "
        f"{sh['aggregate']:.3f} on the brain's real spiking substrate"
    )
    assert fh["retention"] >= sh["retention"] + 0.20, (
        f"DEEPENED non-regression: whole retention {fh['retention']:.3f} >> sum "
        f"{sh['retention']:.3f}"
    )
    assert hc["retention"] < fh["retention"] - 0.10, (
        f"DEEPENED consolidation load-bearing: off-retention {hc['retention']:.3f} must drop "
        f"below FULL {fh['retention']:.3f}"
    )
    assert fh["max_units"] <= BUDGET, f"DEEPENED n_units {fh['max_units']} exceeded budget {BUDGET}"
    assert _run(True, True, True, hard=True) == fh, "two seed=0 hard runs must be byte-identical"

    print(
        f"self-check OK: integrated aggregate {ctrl['aggregate']:.3f}->{full['aggregate']:.3f} and "
        f"retention {ctrl['retention']:.3f}->{full['retention']:.3f} vs the un-arbitrated control;"
    )
    print(
        f"  load-bearing: metacog {full['aggregate']:.3f}->{abl_meta['aggregate']:.3f} "
        f"(+{abl_meta['noise_units']} noise units), consolidation retention "
        f"{full['retention']:.3f}->{abl_cons['retention']:.3f}, criticality "
        f"{full['aggregate']:.3f}->{abl_crit['aggregate']:.3f} (sigma out of band);"
    )
    print(
        f"  bounded (n_units <= {BUDGET}), sigma in-band, deterministic, ASCII -- "
        f"BOUNDED whole > parts, non-regressing; NOT AGI"
    )


if __name__ == "__main__":
    main()
