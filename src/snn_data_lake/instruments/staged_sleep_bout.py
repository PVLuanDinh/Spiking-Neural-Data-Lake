r"""G6 / T114 part (e): the staged sleep-bout RUNNER (R3's interface), its participation census and its
fabrication guard. Evaluation (learning OFF) is R3's; persistence and the interrupted-vs-uninterrupted check
are R2's; this module only executes a bout and reports it.

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md, frozen at P0 (df19f61e):
section 2.7 (the runner, its carriers, G6's own (e) instruments), 2.11 (C16, C17, C19, C20's declaration),
2.13 P3b; the ruling that the BUILD does not wait on R3's verdicts (DEPENDENCIES ruling 2).

  run_bout(brain, stages, *, rng_seed) -> BoutReport
    stages: ordered [(name in {"nrem", "rem", "flat"}, {"presentations": int, "structural_candidates": int | None})]
    "nrem" -- ONE presentation = one eligible stored interaction (the learner's own replay ring; held-out
              entries never eligible) re-driven through the REAL learner with its recorded label:
              `interact(code, label, _record=False)`, exactly the live sleep replay's unit (T058).
    "rem"  -- integration/refinement over items ALREADY replayed in this bout: ONE presentation = a SYNTHETIC
              recombination (the elementwise mean of two already-replayed codes), refused by the fabrication
              guard if it carries any observed/label/reward flag, driven FORWARD only (no deposit, no reward)
              and followed by the learner's own label-free Benna-Fusi rest (`relax`) on the row it activates.
              A synthetic item never becomes a lake event, ring entry, label or held-out read (R3's K10).
    "flat" -- the equal-compute control: nrem's unit, one stage carrying the summed presentations.
  The order is a parameter (the stage-shuffle control). Every draw comes from ONE runner RNG seeded by
  `rng_seed`; no draw is conditional on wall time or another RNG, so R2's redo reproduces the bout from its
  carriers (`bout_rng_draws` counted). Pause reasons: budget | plateau | interference | wake_request (the
  last three only when the caller supplies `validate` / `wake_request`). The runner never ticks the brain,
  never advances wake chronology and never issues a motor command (asserted on `brain._step`).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.staged_sleep_bout   (self-check, brain-free)
"""

import random
from dataclasses import asdict, dataclass, field

STAGE_NAMES = ("nrem", "rem", "flat")
PAUSES = ("budget", "plateau", "interference", "wake_request")
PLATEAU_CHECKS = (
    3  # sustained plateau = no validation gain over this many consecutive checks (session call)
)

# distinct-from: readout/interaction_learning.py::replay -- the live per-tick replay of k ring draws on the
#   learner's OWN rng; this runner executes a STAGED bout of registered presentations on one runner RNG and
#   reports carriers for R2's redo. It calls the learner's own `interact` / `respond` and the stack's own
#   `relax`; no learning rule is re-spelled here. Site read.


class FabricationRefused(Exception):
    """A synthetic item carried an observed / label / reward flag (C16)."""


@dataclass
class Item:
    code: list
    label: object = None
    reward: object = None
    observed: bool = False
    synthetic: bool = False


def guard(item):
    """THE fabrication guard: a SYNTHETIC item may carry no observed flag, no label and no reward."""
    if item.synthetic and (item.observed or item.label is not None or item.reward is not None):
        raise FabricationRefused(
            f"synthetic item flagged observed={item.observed} label={item.label!r}"
        )
    return item


class CountingRng:
    """ONE runner RNG; every draw counted (R2's `bout_rng_draws` slot)."""

    def __init__(self, seed):
        self._r = random.Random(f"g6.bout:{seed}")
        self.draws = 0

    def randrange(self, n):
        self.draws += 1
        return self._r.randrange(n)


@dataclass
class StageReport:
    name: str
    budget: int
    presentations: int = 0
    deposits: int = 0
    consolidate_calls: int = 0
    structural_edits: int = 0
    synthetic: int = 0
    refused: int = 0
    starved: int = 0  # budget NOT spent because the stage had nothing to draw (empty ring / < 2 replayed)
    populations: list = field(
        default_factory=list
    )  # participation census: populations whose weights changed


@dataclass
class BoutReport:
    stages: list
    pause_reason: str
    bout_rng_draws: int
    carriers: list  # per presentation: {stage_index, stage_name, presentations_spent, pending, bout_rng_draws}

    def total_presentations(self):
        return sum(s.presentations for s in self.stages)

    def to_dict(self):
        return asdict(self)


# ---- populations for the participation census (the legacy learned circuits; born units join via `extra`) ----
def populations(brain, extra=None):
    """{label: flat weight snapshot} of every learned population the census reads."""
    out = {"legacy:ixn": [w for n in brain.brain._ixn.neurons for w in n.w]}
    att = getattr(brain, "attractor", None)
    if att is not None:
        out["legacy:attractor"] = [w for row in att.W for w in row]
    neuro = getattr(brain, "neuro", None)
    model = getattr(neuro, "_model", None) if neuro is not None else None
    if model is not None and getattr(model, "W", None) is not None:
        out["legacy:grower"] = [float(x) for x in model.W.reshape(-1).tolist()]
    for k, read in (extra or {}).items():
        out[k] = list(read())
    return out


def changed(before, after):
    return sorted(k for k in after if k not in before or before[k] != after[k])


def participation_red(report, solved_populations):
    """C17: red when no measured change happened OUTSIDE the solved problem's populations."""
    moved = {p for s in report.stages for p in s.populations}
    return not (moved - set(solved_populations))


def check_compute(staged, flat):
    """C19: a staged bout and its flat control must report the same total REALIZED presentations (a stage
    that starved spent less than its budget; `StageReport.starved` names it)."""
    return staged.total_presentations() == flat.total_presentations()


def stages_total(stages):
    return sum(int(b["presentations"]) for _, b in stages)


def flat_of(stages):
    """The equal-compute control of a staged list: one 'flat' stage with the summed presentations."""
    return [("flat", {"presentations": stages_total(stages), "structural_candidates": None})]


def _eligible_ring(ixn):
    """The replay ring minus held-out entries (a key carrying split == 'heldout' is never replay-eligible)."""
    out = []
    for code, fb, key in ixn._replay:
        split = key.get("split") if isinstance(key, dict) else None
        if split != "heldout" and isinstance(fb, str):
            out.append((code, fb, key))
    return out


def run_bout(brain, stages, *, rng_seed, validate=None, wake_request=None, extra_populations=None):
    """Execute one bout; see the module docstring. `validate()` -> a score (higher is better) read by the
    caller with learning OFF; `wake_request()` -> bool."""
    for name, b in stages:
        if name not in STAGE_NAMES or int(b.get("presentations", -1)) < 0:
            raise ValueError(f"bad stage {name!r} {b!r}; names are {STAGE_NAMES}")
    ixn = brain.brain._ixn
    stack = getattr(brain, "_plasticity", None)
    rng = CountingRng(rng_seed)
    step0 = brain._step
    ring = _eligible_ring(ixn)
    replayed = []
    reports, carriers = [], []
    pause = "budget"
    best = validate() if validate else None
    start = best
    flat_checks = 0
    deposits = [0]
    o_dep = ixn._deposit

    def dep(*a, **k):
        deposits[0] += 1
        return o_dep(*a, **k)

    ixn._deposit = dep
    try:
        for si, (name, b) in enumerate(stages):
            rep = StageReport(name=name, budget=int(b["presentations"]))
            before = populations(brain, extra_populations)
            c0 = stack.consolidation_calls() if stack is not None else 0
            d0 = deposits[0]
            for _ in range(rep.budget):
                if wake_request is not None and wake_request():
                    pause = "wake_request"
                    break
                if name in ("nrem", "flat"):
                    if not ring:
                        break
                    code, fb, key = ring[rng.randrange(len(ring))]
                    item = guard(Item(list(code), label=fb, observed=True))
                    ixn.interact(item.code, item.label, _record=False, _key=key)
                    replayed.append(list(code))
                else:
                    if len(replayed) < 2:
                        break
                    a = replayed[rng.randrange(len(replayed))]
                    c = replayed[rng.randrange(len(replayed))]
                    try:
                        item = guard(Item([(x + y) / 2.0 for x, y in zip(a, c)], synthetic=True))
                    except FabricationRefused:
                        rep.refused += 1
                        continue
                    rep.synthetic += 1
                    li = ixn._argmax_label(
                        ixn.encode(item.code)
                    )  # forward only: no deposit, no reward
                    if stack is not None:
                        settled = stack.relax(li, 1)
                        if settled is not None:
                            ixn.neurons[li].w = list(settled)
                rep.presentations += 1
                carriers.append(
                    {
                        "stage_index": si,
                        "stage_name": name,
                        "presentations_spent": rep.presentations,
                        "pending": [],
                        "bout_rng_draws": rng.draws,
                    }
                )
                if validate is not None:
                    score = validate()
                    if start is not None and score < start - 1e-12:
                        pause = "interference"
                        break
                    if best is None or score > best + 1e-12:
                        best, flat_checks = score, 0
                    else:
                        flat_checks += 1
                        if flat_checks >= PLATEAU_CHECKS:
                            pause = "plateau"
                            break
            if pause == "budget":
                rep.starved = rep.budget - rep.presentations
            rep.deposits = deposits[0] - d0
            rep.consolidate_calls = (stack.consolidation_calls() if stack is not None else 0) - c0
            rep.populations = changed(before, populations(brain, extra_populations))
            reports.append(rep)
            if pause != "budget":
                break
    finally:
        ixn._deposit = o_dep
    if brain._step != step0:
        raise AssertionError("the runner advanced wake chronology")
    if carriers:
        carriers[-1]["pause_reason"] = pause
    return BoutReport(
        stages=reports, pause_reason=pause, bout_rng_draws=rng.draws, carriers=carriers
    )


# R2's carrier format {class, where, boundary, owner, omission_red} (2.7; C20 is run by R2's R2-c6)
CARRIERS = [
    {
        "class": "bout progress",
        "where": "R2 sleep progress event (r2.v1)",
        "boundary": "end of each learner-affecting sleep tick",
        "owner": "R2 (slot) / G6 (value)",
        "omission_red": f"redo cannot place the bout at its {k}",
        "field": k,
    }
    for k in ("stage_index", "stage_name", "presentations_spent", "pending")
] + [
    {
        "class": "runner rng",
        "where": "R2 sleep progress event",
        "boundary": "end of each learner-affecting sleep tick",
        "owner": "R2 (slot) / G6 (value)",
        "omission_red": "redo draws a different item",
        "field": "bout_rng_draws",
    },
    {
        "class": "pause",
        "where": "R2 sleep progress event (bout's last tick only)",
        "boundary": "bout end",
        "owner": "R2 (slot) / G6 (value)",
        "omission_red": "a resumed bout re-enters a paused stage",
        "field": "pause_reason",
    },
]


# ================================================================================================
# self-check (brain-free): a stand-in learner exercising the runner's accounting
# ================================================================================================
class _Neuron:
    def __init__(self, n):
        self.w = [0.0] * n


class _Learner:
    def __init__(self, n=4, labels=("A", "B")):
        self.responses = list(labels)
        self.neurons = [_Neuron(n) for _ in labels]
        self._replay = []

    def encode(self, s):
        return list(s)

    def _argmax_label(self, code):
        scores = [sum(w * x for w, x in zip(nr.w, code)) for nr in self.neurons]
        return scores.index(max(scores))

    def _deposit(self, li, code, d):
        self.neurons[li].w = [w + d * x for w, x in zip(self.neurons[li].w, code)]

    def interact(self, stimulus, feedback, _record=True, _key=None):
        self._deposit(self.responses.index(feedback), stimulus, 0.1)


class _Stack:
    def __init__(self, ixn):
        self.ixn, self.calls = ixn, 0

    def consolidation_calls(self):
        return self.calls

    def relax(self, i, steps):
        self.calls += 1
        return [w * 0.99 for w in self.ixn.neurons[i].w]


class _Brain:
    def __init__(self):
        class _LB:
            pass

        self.brain = _LB()
        self.brain._ixn = _Learner()
        self._plasticity = _Stack(self.brain._ixn)
        self._step = 0
        self.brain._ixn._replay = [
            ([1, 0, 0, 0], "A", None),
            ([0, 1, 0, 0], "B", None),
            ([0, 0, 1, 0], "A", {"split": "heldout"}),
        ]


def main():
    stages = [
        ("nrem", {"presentations": 6, "structural_candidates": None}),
        ("rem", {"presentations": 4, "structural_candidates": None}),
        ("nrem", {"presentations": 2, "structural_candidates": None}),
    ]
    b1, b2 = _Brain(), _Brain()
    r1 = run_bout(b1, stages, rng_seed=7)
    r2 = run_bout(b2, stages, rng_seed=7)
    assert r1.to_dict() == r2.to_dict() and r1.bout_rng_draws == r2.bout_rng_draws, (
        "one RNG -> redo-exact"
    )
    assert r1.total_presentations() == 12 and r1.pause_reason == "budget"
    assert r1.stages[0].deposits == 6 and r1.stages[1].deposits == 0 and r1.stages[1].synthetic == 4
    assert r1.stages[1].consolidate_calls == 4, (
        "rem = forward + the learner's own rest, never a deposit"
    )
    assert len(_eligible_ring(b1.brain._ixn)) == 2, (
        "held-out ring entries are never replay-eligible"
    )

    flat = run_bout(_Brain(), flat_of(stages), rng_seed=7)
    assert check_compute(r1, flat), "C19: flat carries the summed presentations"
    planted = run_bout(_Brain(), flat_of(stages[:2]) + [("flat", {"presentations": 3})], rng_seed=7)
    assert not check_compute(r1, planted), (
        "C19: one extra presentation must be refused as compute-unmatched"
    )
    # R3 refutation R1: a rem-first order starves (nothing replayed yet), spends less, and must be refused
    rem_first = run_bout(_Brain(), [("rem", {"presentations": 4}), ("nrem", {"presentations": 4})], rng_seed=7)
    assert rem_first.stages[0].starved == 4 and rem_first.stages[1].starved == 0, rem_first.stages
    assert not check_compute(rem_first, run_bout(_Brain(), flat_of([("rem", {"presentations": 4}), ("nrem", {"presentations": 4})]), rng_seed=7))

    try:
        guard(Item([0.5] * 4, synthetic=True, observed=True))
        raise AssertionError("C16: a synthetic item flagged observed must be refused")
    except FabricationRefused:
        pass
    guard(Item([0.5] * 4, label="A", observed=True))  # a real item with its label passes

    assert participation_red(r1, {"legacy:ixn"}), "C17: touching only the solved populations reds"
    assert not participation_red(r1, {"born:p1"})

    seq = iter([0.5, 0.6, 0.4])
    r3 = run_bout(_Brain(), stages, rng_seed=7, validate=lambda: next(seq))
    assert r3.pause_reason == "interference", r3.pause_reason
    r4 = run_bout(_Brain(), stages, rng_seed=7, wake_request=lambda: True)
    assert r4.pause_reason == "wake_request" and r4.total_presentations() == 0
    assert r1.carriers[-1]["pause_reason"] == "budget" and {c["field"] for c in CARRIERS} >= {
        "stage_index",
        "stage_name",
        "presentations_spent",
        "pending",
        "bout_rng_draws",
        "pause_reason",
    }
    print(
        "staged_sleep_bout self-check OK: one RNG redo-exact (draws equal); nrem deposits, rem synthetic + rest "
        "only; held-out ring entries excluded; C19 flat equal and a planted extra presentation refused; C16 "
        "synthetic-observed refused; C17 solved-only participation reds; pauses interference + wake_request; "
        "carriers declared"
    )


if __name__ == "__main__":
    main()
