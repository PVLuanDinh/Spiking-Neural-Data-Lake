"""
Lifelong consolidation: the GROW <-> PRUNE <-> CONSOLIDATE continual-learning loop
with a non-regression (anti-catastrophic-forgetting) probe.

WHAT THIS IS (honest). This is a genuine LIFELONG-LEARNING / anti-catastrophic-
forgetting mechanism -- a real *bounded* continual-learning increment. It learns a
SEQUENCE of tasks (A, B, C, ...) with disjoint classes and keeps OLD-task accuracy
alive using three brain-inspired parts the project already owns:

  * GROW   (neurogenesis)  -- growth.growing_learner.GrowingLearner recruits a
                              new prototype unit for a novel pattern, refines the
                              nearest unit on a familiar one.
  * CONSOLIDATE (replay)   -- hippocampus.experience_replay.ReplayBuffer stores task
                              exemplars; during a NEW task we INTERLEAVE replay of
                              OLD-task exemplars with new learning. Re-firing old
                              units refreshes their utility and keeps their prototypes
                              sharp. This interleaved rehearsal IS the non-regression
                              mechanism.
  * PRUNE  (developmental) -- under a fixed capacity BUDGET we drop the LOWEST-utility
                              units, faithful to plasticity.developmental_pruning
                              (keep-strong beats random at matched sparsity). Utility
                              is a use-decayed activity trace ("use it or lose it").

The failure mode this fixes is real: under a nightly-pruning capacity budget, an old
task's units go stale (never fire during later tasks), fall to the bottom of the
utility ranking, and get pruned away -- the knowledge is silently reallocated. Replay
rehearses those units so they stay useful and survive the budget. That is exactly the
"nightly pruning + replay + dual fast/slow memory" continual-learning core of
arXiv:2504.20109 ("Personalized AGI via Neuroscience-Inspired Continuous Learning"),
realized here from the project's own parts.

WHAT THIS IS NOT. This is NOT AGI. The paper's "AGI" label is aspirational/theoretical;
what is buildable and verified here is the bounded continual-learning SUBSTRATE
(self-improvement ladder L6 grow/prune capacity control + L11 non-regression retention),
not general intelligence. It is a prototype-nearest classifier over synthetic separable
tasks -- no planning, no reasoning, no world model. The streams are synthetic.

Reuses (does NOT reimplement): GrowingLearner (grow + cosine match + refine),
ReplayBuffer (fixed-capacity ring + seeded SWR-style sampling), _mathutils.mean. The
tiny magnitude-prune is reimplemented faithfully over UNITS (developmental_pruning
prunes over synapses), same "drop the smallest score" rule.

Determinism: per-instance random.Random(seed); no global RNG, no wall-clock, no hash().
ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.lifelong_consolidation
"""

import random

from snn_data_lake.growth.growing_learner import GrowingLearner
from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.infra._mathutils import mean

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "LifelongConsolidator.learn_task": (
        "LOCAL",
        "learns one task by growing/refining on its stream and interleaving old-task replay. Replay "
        "of a local rule is still that local rule; the anti-forgetting mechanism is rehearsal, not a "
        "gradient",
    ),
}


class LifelongConsolidator:
    """Continual learner over a sequence of tasks: grow <-> prune <-> consolidate.

    Per task: GROW/refine prototype units on the task stream + store exemplars in a
    lifelong replay memory. During the task: optionally INTERLEAVE replay of OLD-task
    exemplars (consolidation / non-regression). After the task: optionally PRUNE the
    lowest-utility units down to `budget` (bounded capacity). Utility is a use-decayed
    activity trace: it decays every new step and is bumped when a unit fires, so units
    that stop firing (an old task with no rehearsal) sink to the bottom of the ranking.

    Knobs
      budget      : max number of units kept after each task (capacity ceiling).
      replay      : interleave old-task replay during new learning (the retention fix).
      prune       : enforce the budget by dropping low-utility units.
      prune_mode  : "magnitude" (drop lowest utility) or "random" (fair control).
      vigilance   : GrowingLearner novelty threshold (cosine).
      lr          : GrowingLearner prototype refinement rate.
      task_decay  : utility multiplier applied ONCE at the start of each new task
                    (0<task_decay<1). Within a task, credits just accumulate, so a
                    unit's utility = its firing count decayed by how many task
                    boundaries have passed since it last fired. An old task with no
                    rehearsal decays toward zero across boundaries and gets pruned;
                    replay re-fires it each task so it survives -- that contrast is
                    the whole non-regression story.
      replay_batch: old exemplars rehearsed per new sample when replay is on.
      epochs      : passes over each task's training stream.
    """

    def __init__(
        self,
        dim,
        budget,
        *,
        replay=True,
        prune=True,
        prune_mode="magnitude",
        vigilance=0.9,
        lr=0.2,
        task_decay=0.15,
        replay_batch=2,
        epochs=2,
        seed=0,
    ):
        self.dim = dim
        self.budget = budget
        self.replay_on = replay
        self.prune_on = prune
        self.prune_mode = prune_mode
        self.task_decay = task_decay
        self.replay_batch = replay_batch
        self.epochs = epochs
        self.seed = seed
        self.learner = GrowingLearner(vigilance=vigilance, lr=lr, dim=dim)
        self.labels = []  # class label per unit (aligned to prototypes)
        self.utility = []  # use-decayed activity trace per unit
        self.memory = ReplayBuffer(capacity=4096, seed=seed + 1)  # lifelong exemplars
        self.rng = random.Random(seed + 2)  # random-prune RNG

    @property
    def n_units(self):
        return self.learner.n_units

    # ---- learning primitives ------------------------------------------------
    def _observe(self, x, y):
        """Grow a new unit for a novel pattern, else refine + credit the winner.

        Credit accumulates within a task (the task-boundary decay in learn_task
        supplies the recency / 'use it or lose it' pressure), so a unit's utility is
        its firing count discounted by how many tasks ago it last fired."""
        uid, grew = self.learner.observe(x)
        if grew:
            self.labels.append(y)
            self.utility.append(1.0)  # new unit starts with unit credit
        else:
            self.utility[uid] += 1.0  # familiar hit -> refresh (use it)
        return uid

    def _prune_to_budget(self):
        """Drop units above the budget by the LOWEST utility (magnitude) or at RANDOM.

        Faithful to developmental_pruning's magnitude-vs-random comparison, one level
        up: we score UNITS by utility (accumulated activity) and cut the weakest, then
        keep the survivors' prototypes / labels / utilities in lock-step."""
        n = self.n_units
        if not self.prune_on or n <= self.budget:
            return
        n_remove = n - self.budget
        if self.prune_mode == "random":
            victims = set(self.rng.sample(range(n), n_remove))
        else:
            ranked = sorted(range(n), key=lambda i: (self.utility[i], i))
            victims = set(ranked[:n_remove])
        keep = [i for i in range(n) if i not in victims]
        self.learner.prototypes = [self.learner.prototypes[i] for i in keep]
        self.labels = [self.labels[i] for i in keep]
        self.utility = [self.utility[i] for i in keep]

    def _replay_once(self):
        """Rehearse a batch of OLD exemplars: re-fire + refine their units (no decay).

        This is the interleaved SWR-style consolidation -- it refreshes the utility of
        units for tasks not currently streaming, so they survive the pruning budget."""
        for label, xt, _ in self.memory.sample(self.replay_batch):
            self._observe(list(xt), label)

    def learn_task(self, task_id, train):
        """Learn one task: grow/refine on its stream, interleave old-task replay,
        commit its exemplars to lifelong memory, then prune to the budget."""
        self.utility = [u * self.task_decay for u in self.utility]  # recency pressure
        order_rng = random.Random(self.seed + 100 + task_id)
        fresh = []
        for _ in range(self.epochs):
            stream = train[:]
            order_rng.shuffle(stream)
            for x, y in stream:
                self._observe(x, y)  # learn the new sample (decays + credits)
                if self.replay_on and len(self.memory):
                    self._replay_once()  # rehearse old tasks (non-regression)
                fresh.append((y, tuple(x)))
        for y, xt in fresh:  # exemplars visible to LATER tasks only
            self.memory.add(y, xt, 0.0)
        self._prune_to_budget()  # grow <-> prune under the budget

    # ---- readout ------------------------------------------------------------
    def classify(self, x):
        """Nearest surviving prototype's class label (-1 if no units)."""
        uid = self.learner.classify(x)
        return self.labels[uid] if uid >= 0 else -1

    def accuracy(self, data):
        """Held-out accuracy on [(x, y), ...]; also used for old-task retention."""
        return mean([1.0 if self.classify(x) == y else 0.0 for x, y in data])


# ---- synthetic continual-learning benchmark --------------------------------
D = 36  # input dim = 6 classes x BLK
BLK = 6  # dims owned by each global class (disjoint across classes)
HIGH, LOW = 0.9, 0.1  # in-block vs baseline activation
SIGMA = 0.06  # gaussian sample noise (rate units)
T = 3  # number of tasks (A, B, C)
KPT = 2  # classes per task -> 6 disjoint classes
MODES = 2  # sub-prototypes per class: a PRIMARY + a redundant SECONDARY
TRAIN_PER_MODE = (12, 4)  # primary is frequent, secondary is rare (prune-able)
TEST_PER_MODE = (10, 10)  # test both modes evenly (fair retention measurement)
BUDGET = 6  # capacity ceiling: 1 primary per class (12 units grow, 3 tasks)


def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _proto(global_class, mode):
    """Class prototype: HIGH on a 4-dim window inside the class's 6-dim block.

    The two modes' windows overlap by 3 dims (shift by one), so within a class the two
    modes are close (cosine ~0.8) while different classes are far (disjoint blocks,
    cosine ~0.28). Pruning a class's redundant SECONDARY mode therefore does not lose
    the class -- its samples still fall on the surviving PRIMARY mode, not another
    class. Ordering: cross-class (0.28) < cross-mode (0.80) < vigilance (0.90) <
    within-mode (0.98)."""
    b = global_class * BLK
    p = [LOW] * D
    lo = b if mode == 0 else b + 1
    for i in range(lo, lo + 4):
        p[i] = HIGH
    return p


def _make_tasks(seed):
    """T tasks x KPT disjoint classes; each class = a frequent PRIMARY + rare SECONDARY
    noisy Gaussian sub-cluster.

    Returns [{'classes': [...], 'train': [(x, y)], 'test': [(x, y)]}, ...]. Labels are
    globally unique across tasks so old-task forgetting is genuinely measurable."""
    rng = random.Random(seed)

    def sample(g, mode):
        p = _proto(g, mode)
        return [_clip01(p[i] + rng.gauss(0.0, SIGMA)) for i in range(D)]

    tasks = []
    for t in range(T):
        classes = [t * KPT + k for k in range(KPT)]
        train, test = [], []
        for g in classes:
            for mode in range(MODES):
                for _ in range(TRAIN_PER_MODE[mode]):
                    train.append((sample(g, mode), g))
                for _ in range(TEST_PER_MODE[mode]):
                    test.append((sample(g, mode), g))
        tasks.append({"classes": classes, "train": train, "test": test})
    return tasks


def run_sequence(tasks, **knobs):
    """Learn every task in order; snapshot (task, current-acc, A-retention, n_units)
    after each. Returns (consolidator, history)."""
    con = LifelongConsolidator(dim=D, budget=BUDGET, **knobs)
    a_test = tasks[0]["test"]
    history = []
    for t, task in enumerate(tasks):
        con.learn_task(t, task["train"])
        history.append((t, con.accuracy(task["test"]), con.accuracy(a_test), con.n_units))
    return con, history


def _all_test(tasks):
    data = []
    for task in tasks:
        data += task["test"]
    return data


def _digest(tasks):
    """Compact deterministic fingerprint of a full run (for the determinism check)."""
    on, _ = run_sequence(tasks, replay=True, prune=True, prune_mode="magnitude", seed=0)
    off, _ = run_sequence(tasks, replay=False, prune=True, prune_mode="magnitude", seed=0)
    rnd, _ = run_sequence(tasks, replay=True, prune=True, prune_mode="random", seed=0)
    allt = _all_test(tasks)
    return (
        round(on.accuracy(tasks[0]["test"]), 6),
        round(off.accuracy(tasks[0]["test"]), 6),
        round(on.accuracy(allt), 6),
        round(rnd.accuracy(allt), 6),
        on.n_units,
        off.n_units,
        rnd.n_units,
    )


def main():

    tasks = _make_tasks(seed=0)
    all_test = _all_test(tasks)

    # === (1) NON-REGRESSION: interleaved replay stops catastrophic forgetting =====
    con_on, hist_on = run_sequence(tasks, replay=True, prune=True, prune_mode="magnitude", seed=0)
    con_off, hist_off = run_sequence(
        tasks, replay=False, prune=True, prune_mode="magnitude", seed=0
    )
    retA_on = con_on.accuracy(tasks[0]["test"])
    retA_off = con_off.accuracy(tasks[0]["test"])
    curr_on = con_on.accuracy(tasks[-1]["test"])
    curr_off = con_off.accuracy(tasks[-1]["test"])
    ret_margin = retA_on - retA_off

    # === (2) BOUNDED CAPACITY: grow <-> prune vs grow-only ========================
    con_grow, _ = run_sequence(tasks, replay=True, prune=False, seed=0)
    acc_pruned = con_on.accuracy(all_test)
    acc_grow = con_grow.accuracy(all_test)
    bounded = all(n <= BUDGET for _, _, _, n in hist_on)

    # === (3) UTILITY PRUNE beats RANDOM prune at the SAME budget ==================
    # Magnitude is deterministic (drops the rare secondaries, keeps one primary per
    # class). Random is high-variance with so few units, so -- like developmental_
    # pruning averages over many synapses -- we average the random control over
    # RANDOM_SEEDS runs; on average it wipes whole classes and scores lower.
    RANDOM_SEEDS = 8
    acc_mag = con_on.accuracy(all_test)
    rnd_accs = [
        run_sequence(tasks, replay=True, prune=True, prune_mode="random", seed=s)[0].accuracy(
            all_test
        )
        for s in range(RANDOM_SEEDS)
    ]
    acc_rnd = mean(rnd_accs)

    # === report table (replay-on vs replay-off) ===================================
    lines = []
    lines.append("=" * 66)
    lines.append("LIFELONG CONSOLIDATION  (grow <-> prune <-> consolidate)")
    lines.append("=" * 66)
    lines.append(
        f"tasks={T} x {KPT} disjoint classes  dim={D}  budget={BUDGET}  "
        f"prune=magnitude  task_decay={con_on.task_decay}"
    )
    lines.append("-" * 66)
    lines.append("replay ON  (interleaved rehearsal of old-task exemplars)")
    lines.append(f"  {'task':<6}{'current-acc':>14}{'A-retention':>14}{'n_units':>10}")
    for t, cur, reta, n in hist_on:
        lines.append(f"  {chr(ord('A') + t):<6}{cur:>13.1%}{reta:>14.1%}{n:>10}")
    lines.append("replay OFF (control: no rehearsal -> catastrophic forgetting)")
    lines.append(f"  {'task':<6}{'current-acc':>14}{'A-retention':>14}{'n_units':>10}")
    for t, cur, reta, n in hist_off:
        lines.append(f"  {chr(ord('A') + t):<6}{cur:>13.1%}{reta:>14.1%}{n:>10}")
    lines.append("-" * 66)
    lines.append(
        f"(1) A-retention   replay-ON {retA_on:.1%}  >>  replay-OFF "
        f"{retA_off:.1%}   (margin {ret_margin:+.1%})"
    )
    lines.append(
        f"    current task  replay-ON {curr_on:.1%}      replay-OFF "
        f"{curr_off:.1%}   (both still learned)"
    )
    lines.append(
        f"(2) capacity      pruned n_units<=budget: {bounded}  "
        f"(final {con_on.n_units})   grow-only n_units={con_grow.n_units}>"
        f"{BUDGET}"
    )
    lines.append(
        f"    accuracy      pruned {acc_pruned:.1%}  vs  grow-only "
        f"{acc_grow:.1%}  (prune does not tank acc)"
    )
    lines.append(
        f"(3) prune rule    magnitude {acc_mag:.1%}  >  random "
        f"{acc_rnd:.1%}  (mean of {RANDOM_SEEDS}; utility beats random @ budget)"
    )
    lines.append("=" * 66)

    # === deterministic, un-riggable asserts (each with a control that FAILS) ======
    # (1) non-regression headline + both controls behave as claimed.
    assert curr_on >= 0.80, f"replay-on must still learn the current task ({curr_on:.2f})"
    assert curr_off >= 0.80, f"replay-off must still learn the current task ({curr_off:.2f})"
    assert retA_on >= 0.80, f"replay-on old-task A retention too low ({retA_on:.2f})"
    assert retA_off <= 0.55, f"CONTROL replay-off should CATASTROPHICALLY FORGET A ({retA_off:.2f})"
    assert ret_margin >= 0.30, (
        f"retention gain from replay must be clear (>=0.30), got {ret_margin:+.2f}"
    )

    # (2) bounded capacity: prune holds the budget, grow-only blows past it.
    assert bounded, "pruned n_units must stay <= budget after every task"
    assert con_on.n_units <= BUDGET, f"final pruned n_units {con_on.n_units} > {BUDGET}"
    assert con_grow.n_units > BUDGET, (
        f"CONTROL grow-only must exceed budget ({con_grow.n_units} <= {BUDGET})"
    )
    assert acc_pruned >= acc_grow - 0.05, (
        f"pruning must not tank accuracy ({acc_pruned:.2f} vs grow-only {acc_grow:.2f})"
    )

    # (3) utility prune is not random.
    assert acc_mag > acc_rnd + 0.05, (
        f"magnitude prune ({acc_mag:.2f}) must beat mean random prune ({acc_rnd:.2f})"
    )

    # (4) determinism + ASCII.
    assert _digest(tasks) == _digest(tasks), "same seed must give identical runs"
    for ln in lines:
        assert ln.isascii(), "all output must be ASCII"

    for ln in lines:
        print(ln)
    print(
        f"self-check OK: replay retention {retA_on:.0%}>>{retA_off:.0%} "
        f"(margin {ret_margin:+.0%}); capacity<= {BUDGET} vs grow-only "
        f"{con_grow.n_units}; magnitude {acc_mag:.0%}>random {acc_rnd:.0%}; "
        f"deterministic, ASCII"
    )


if __name__ == "__main__":
    main()
