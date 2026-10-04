r"""G2 / T110's problem contract (P2): the `Problem` record G3, G5, G6 and G9 import, its splits as episodes,
the learning-OFF read, the solved predicate and the growth-budget fields.

Pre-registration: specs/022-unified-brain-arc/evidence/g2-taught-problems.md (frozen at 5e3c2f6e; 2.6, 2.7,
2.15). The generators and their construction checks are `delayed_cue` (P1); the campaign is
`taught_campaign` (P3).

The contract (2.6):
  Problem.episodes(seed, split)    split in teach-epoch / validation / future (/ future_d for C1-d and A_c1d):
                                   a list of episodes, each a tuple of Turn(text, role, label); the episode
                                   is THE boundary unit; labels ride on probes only
  Problem.teach_epochs(seed, E)    E epochs from ONE `random.Random(seed)` stream (as (ii)'s `_schedule`)
  teaching_sequence(problem, seed, E)  the recorded per-episode (index, probe text, label) list an attempt
                                   teaches (G5's time-shifted control shifts it)
  solved(problem, split, reads)    every probe of the read answered correctly; the read must carry the split's
                                   full presentation count; an empty read raises (B9)
  budget(problem, spends)          52 for a first problem, else BETA x the median POSITIVE spend of solved
                                   problems (zero-spend solves never enter); B11 is a COST-BUDGET pacing floor
                                   (budget >= the teach-code count), never a sufficiency claim (2.15.6)

Splits (2.6, 2.15.1): validation and future are 4 passes over the items, ordered by `random.Random(seed +
7_777_777)` and `random.Random(seed + 8_888_888)` (future_d: + 9_999_999). For the same-items problems they
are never-taught PRESENTATIONS of the same items (a stateless reader's future read repeats its validation
read, disclosed); for C1-d and A_c1d they are disjoint distractor-combo sets (novel combinations).
**future_d is G6(d)'s untouched set**: `episodes(seed, "future_d")` raises unless the caller declares a
learner-free construction check (B13 / B14 only; Codex #1).

Learning OFF (B12): a read routes text through `route_text` only -- no `correct` / `learn_turn` / `interact`,
no `tick`; it freezes plasticity, never state (context turns arrive through reads).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.taught_problems

Deterministic; ASCII-only prints; builds no Brain, touches no lake.
"""

import json
import random
import statistics
from collections import namedtuple

from snn_data_lake.instruments import delayed_cue as dc
from snn_data_lake.instruments import inseparable_mapping as im

# distinct-from: instruments/inseparable_mapping.py -- (ii)'s generator and A1-A5, wrapped here as a Problem,
#   never copied; instruments/delayed_cue.py -- the generators this contract poses. Sites read 2026-10-04.

Turn = namedtuple("Turn", "text role label")
PASSES = 4
STABLE = im.STABLE  # A4's stability: `solved` at each of the last 10 epoch-end reads
SPLIT_OFFSETS = {"validation": 7_777_777, "future": 8_888_888, "future_d": 9_999_999}
FIRST_BUDGET = im.N_IN  # 52 grown units: the frozen code's width (2.7, G3's value adopted)
BETA = None  # [PROJECTION 2]: set from G3's dev seeds by the dated pre-data amendment, never here by hand
RETRY_K = 3  # fresh budgets per problem lifetime, from new teaching only, never from sleep (2.7)


class Problem:
    """One G2 problem. `kind` selects the generator; `cell` the labelling for (i); `family` for C1-d / A_c1d."""

    def __init__(self, pid, kind, *, cell=None, family=None, labelling=None, delay_turns=0):
        self.id, self.kind, self.cell, self.family = pid, kind, cell, family
        self.labelling = labelling
        self.intents = dc.LABELS
        self.delay_turns = int(delay_turns)

    # ---- items and turns --------------------------------------------------------------------------------
    def items(self, seed, split):
        """The split's items for LEARNER-facing callers; future_d is refused (cp1 #11)."""
        if split == "future_d":
            raise PermissionError("future_d is G6(d)'s untouched set: learner-free construction checks only")
        return self._items(seed, split)

    def _items(self, seed, split):
        if self.kind == "combo":
            parts, _ = dc.combo_partition(self.family, seed)
            return dc.combo_items(parts[split])
        if self.kind in ("i", "u", "us"):
            return dc.ITEMS_I
        return im.UV  # (ii) and A_ii

    def _episode(self, item, label):
        if self.kind in ("i", "u", "us"):
            *context, probe = dc.episode_i(item)
            return tuple(Turn(t, r, None) for t, r in context) + (Turn(probe[0], "probe", label),)
        if self.kind == "combo":
            return (Turn(dc.combo_text(self.family, item), "probe", label),)
        texts = dc.TEXTS_AII if self.kind == "aii" else im.TEXTS
        return (Turn(texts[item], "probe", label),)

    def label(self, seed, item, pass_k=0, split="teach", read_index=0):
        if self.kind == "i":
            return dc.label_i(self.cell, item)
        if self.kind == "combo":
            return dc.combo_label(self.family, item)
        if self.kind == "u":
            return dc.u_label(seed, dc.ITEMS_I.index(item), pass_k)
        if self.kind == "aii":
            return dc.label_aii(*item)
        return self.labelling(*item)

    # ---- splits -----------------------------------------------------------------------------------------
    def teach_epochs(self, seed, epochs):
        """E epochs, each the teach items once in `random.Random(seed).sample` order (one stream)."""
        rng = random.Random(seed)
        items = self.items(seed, "teach")
        # U_S teaches from its OWN stream, independent of every read's stream (2.15.6; cp1 #8)
        us_teach = random.Random(f"g2.US-teach:{seed}") if self.kind == "us" else None
        for k in range(int(epochs)):
            order = rng.sample(items, len(items))
            if us_teach is not None:
                yield [self._episode(it, dc.LABELS[us_teach.getrandbits(1)]) for it in order]
            else:
                yield [self._episode(it, self.label(seed, it, pass_k=k)) for it in order]

    # learner-facing held-out read (PASSES passes, offset-stream order); future_d refused, cp1 #11
    def episodes(self, seed, split, *, read_index=0):
        if split == "future_d":
            raise PermissionError("future_d is G6(d)'s untouched set: learner-free construction checks only")
        return self._episodes(seed, split, read_index)

    # the ONLY future_d path: learner-free construction checks and G6(d)'s own re-solve; read_split never uses it
    def construction_episodes(self, seed, split):
        return self._episodes(seed, split, 0)

    def _episodes(self, seed, split, read_index):
        if split not in SPLIT_OFFSETS:
            raise ValueError(f"split {split!r}: teaching goes through teach_epochs")
        if self.kind != "combo" and split == "future_d":
            raise ValueError(f"{self.id} has no future_d split")
        rng = random.Random(seed + SPLIT_OFFSETS[split])
        items = self._items(seed, split if self.kind == "combo" else "teach")
        out, presentation = [], 0
        us = dc.us_labels(seed, split, read_index, PASSES * len(items)) if self.kind == "us" else None
        for k in range(PASSES):
            for it in rng.sample(items, len(items)):
                label = us[presentation] if us else self.label(seed, it, pass_k=k)
                out.append(self._episode(it, label))
                presentation += 1
        return out

    def presentations(self, seed, split):
        return PASSES * len(self._items(seed, split if self.kind == "combo" else "teach"))

    # ---- budget fields (2.7, 2.15.6) -----------------------------------------------------------------
    def n_teach_codes(self, seed=8000):
        return len(self.items(seed, "teach"))

    def n_union_codes(self):
        return len(dc.combo_items(dc.COMBOS)) if self.kind == "combo" else len(self.items(8000, "teach"))


PROBLEMS = {
    **{f"i.{cell}": Problem(f"i.{cell}", "i", cell=cell, delay_turns=1) for cell in dc.CELLS},
    "c1d": Problem("c1d", "combo", family="c1d"),
    "ac1d": Problem("ac1d", "combo", family="ac1d"),
    "ii.xor": Problem("ii.xor", "ii", labelling=im.label_xor),
    "ii.twin": Problem("ii.twin", "ii", labelling=im.label_twin),
    "aii": Problem("aii", "aii"),
    "U": Problem("U", "u", delay_turns=1),
    "U_S": Problem("U_S", "us", delay_turns=1),
}


def teaching_sequence(problem, seed, epochs):
    """The recorded per-episode (index, probe text, label) list an attempt teaches."""
    sequence = []
    for epoch in problem.teach_epochs(seed, epochs):
        for episode in epoch:
            probe = episode[-1]
            sequence.append((len(sequence), probe.text, probe.label))
    return sequence


# ---- the solved predicate (2.6; B9) -----------------------------------------------------------------------
def solved(problem, seed, split, reads):
    """True iff every probe of the read is answered correctly through the real read. `reads` is a list of
    (answer, label); it must carry the split's full presentation count; an empty read raises."""
    if not reads:
        raise ValueError("an empty read is refused")
    expected = problem.presentations(seed, split)
    if len(reads) != expected:
        raise ValueError(f"{problem.id} {split}: {len(reads)} presentations, expected {expected}")
    return all(answer == label for answer, label in reads)


def solves_stably(epoch_end_solved, stable=STABLE):
    """'Solves' in a control clause: `solved` at each of the last `stable` epoch-end reads."""
    return len(epoch_end_solved) >= stable and all(epoch_end_solved[-stable:])


def never_solves(epoch_end_solved):
    return not any(epoch_end_solved)


# distinct-from: readout/growable_readout.py -- a zero fan-out weight extension
# distinct-from: readout/laminar_local_readout.py -- an N-MNIST train/test loader
# distinct-from: readout/language_readout_slda.py -- a train/test split of language pairs. Sites read 2026-10-04.
def read_split(problem, seed, split, route, *, read_index=0, records=None):
    """One learning-OFF read: `route(turns_so_far)` answers each probe; context turns are routed too (a
    stateful reader steps its state on reads). The reader sees LABEL-FREE turns (cp1 #4): the targets stay
    here, in the judging instrument. Returns [(answer, label)] for the probes."""
    reads = []
    for episode in problem.episodes(seed, split, read_index=read_index):
        seen = []
        for turn in episode:
            seen.append(Turn(turn.text, turn.role, None))
            answer = route(tuple(seen))
            if turn.role == "probe":
                reads.append((answer, turn.label))
                if records is not None:  # (context texts, probe text, answer) of the ACTUAL read (cp1 #10)
                    records.append((tuple(t.text for t in seen[:-1]), turn.text, answer))
    return reads


# distinct-from: faculties/reentrant_workspace.py -- a self-check specialist's argmax; this judges a reader.
# B3 on the actual validation stream (cp1 #10): one answer per probe text over every cue value and pass
def check_b3_stream(records):
    answers = {}
    for _context, probe, answer in records:
        answers.setdefault(probe, set()).add(answer)
    same = bool(records) and all(len(a) == 1 for a in answers.values())
    return {"ok": same, "conditions": {"same_answer_every_cue_and_pass": same}}


def check_b9(predicate=None):
    """B9: a planted all-right read is solved, one planted wrong answer is not, an empty read raises."""
    predicate = predicate or solved
    problem = PROBLEMS["i.S0C1"]
    n = problem.presentations(8000, "validation")
    right = [(dc.GREET, dc.GREET)] * n
    one_wrong = right[:-1] + [(dc.GOAL, dc.GREET)]
    try:
        predicate(problem, 8000, "validation", [])
        empty_raises = False
    except ValueError:
        empty_raises = True
    conditions = {
        "all_right_solved": predicate(problem, 8000, "validation", right) is True,
        "one_wrong_not_solved": predicate(problem, 8000, "validation", one_wrong) is False,
        "empty_read_raises": empty_raises,
    }
    return {"ok": all(conditions.values()), "conditions": conditions}


# ---- the budget (2.7, 2.15.6; B11) ------------------------------------------------------------------------
def budget(problem, solved_spends, beta=None):
    """FIRST_BUDGET while no POSITIVE-spend solve exists; else beta x the median positive spend. A zero-spend
    solve never enters the baseline (found in the debate: it would set every later budget to 0)."""
    positive = [s for s in solved_spends if s > 0]
    if not positive:
        return FIRST_BUDGET
    beta = BETA if beta is None else beta
    if beta is None:
        raise ValueError("BETA is unset until the dated pre-data amendment enters G3's value")
    return beta * statistics.median(positive)


def _budget_unfixed(problem, solved_spends, beta=2):
    # the red: the debate's unfixed rule (median over EVERY solved spend, zeros included)
    if not solved_spends:
        return FIRST_BUDGET
    return beta * statistics.median(solved_spends)


# the budget a problem OPENS with (2.7; cp1 #12): the registered formula, refused below the teach-code floor
def open_budget(problem, solved_spends, beta=None, seed=8000):
    value = budget(problem, solved_spends, beta)
    floor = problem.n_teach_codes(seed)
    if value < floor:
        raise ValueError(f"{problem.id}: budget {value} below its cost floor {floor} (teach codes); refused")
    return value


def check_b11(rule=None, opener=None):
    """B11, a COST-BUDGET pacing floor (2.15.6): every G2 problem opens at or above its teach-code count, a
    zero-spend solve does not enter the baseline, and an undersized positive-baseline budget is REFUSED at
    opening (cp1 #12: C1-d under [0, 0, 12] with beta 2 gives 24 < 32). No sufficiency is claimed."""
    rule, opener = rule or budget, opener or open_budget
    floors = {pid: p.n_teach_codes() for pid, p in PROBLEMS.items() if p.kind not in ("u", "us")}
    first = all(opener(PROBLEMS[pid], []) >= n for pid, n in floors.items())
    # two zero-spend solves and one of 12: the baseline must be 12 (positive spends only), never 0
    zero_excluded = rule(PROBLEMS["c1d"], [0, 0, 12], beta=2) == 2 * 12
    try:
        opener(PROBLEMS["c1d"], [0, 0, 12], beta=2)
        undersized_refused = False
    except ValueError:
        undersized_refused = True
    conditions = {"first_budget_at_least_teach_codes": first, "zero_spend_excluded": zero_excluded,
                  "undersized_budget_refused": undersized_refused}
    return {"ok": all(conditions.values()), "conditions": conditions, "floors": floors}


# ---- learning OFF (B12) -------------------------------------------------------------------------------------
def learner_state(unit):
    """The learner's full state as bytes: its G0(c) checkpoint tree (weights, ring, stack, rng) plus
    `_authority`, which the checkpoint excludes by design."""
    return json.dumps(
        {"tree": unit.state_dict(), "authority": dict(unit._authority)}, sort_keys=True
    ).encode("ascii")


def check_b12(unit, problem, seed, read=None):
    """B12: a validation read leaves the learner's full state byte-identical. `read(unit)` builds the red."""
    before = learner_state(unit)
    if read is None:
        read_split(problem, seed, "validation", lambda turns: unit.route_text(turns[-1].text))
    else:
        read(unit)
    same = learner_state(unit) == before
    return {"ok": same, "conditions": {"state_byte_identical": same}}


def _fires(result, clause):
    return (not result["ok"]) and (not result["conditions"][clause])


def construction_reds(unit_factory):
    """P2's reds. `unit_factory(seed)` builds the canonical incumbent (taught_campaign's factory)."""
    problem = PROBLEMS["i.S0C1"]

    def skips_last(p, s, split, reads):
        if not reads:
            raise ValueError("empty")
        return all(a == b for a, b in reads[:-1])

    def correct_once(unit):
        unit.learn_turn(dc.probe_text((0, 0, 1)), None, feedback=dc.GOAL)

    try:
        problem.episodes(8900, "future_d")
        future_d_refused = False
    except (PermissionError, ValueError):
        future_d_refused = True
    c1d_future_d_refused = False
    try:
        PROBLEMS["c1d"].episodes(8900, "future_d")
    except PermissionError:
        c1d_future_d_refused = True
    fired = {
        "b9_reader_skipping_one_presentation": _fires(check_b9(skips_last), "one_wrong_not_solved"),
        "b11_unfixed_zero_spend_rule": _fires(check_b11(_budget_unfixed), "zero_spend_excluded"),
        "b11_opener_without_the_floor": _fires(
            check_b11(opener=lambda p, s, beta=None: budget(p, s, beta)), "undersized_budget_refused"
        ),
        "b12_validation_calls_correct": _fires(
            check_b12(unit_factory(8900), problem, 8900, read=correct_once), "state_byte_identical"
        ),
        "b15_learner_read_of_future_d": future_d_refused and c1d_future_d_refused,
    }
    return {"fired": fired, "all_fired": all(fired.values())}


# distinct-from: engine/live_brain.py -- _reasoning_reply, a fall-through to the reasoning brain; reasoning/
#   dishbrain_learning.py -- a Hebbian rally update; growth/brian2_backend.py -- a child-only Brian2 runner;
#   gates/replication_guard.py -- counts Brain / LiveBrain / AutonomousArm class definitions, none defined here.
#   Sites read 2026-10-04.
def canonical_incumbent(seed, n_in=im.N_IN, lr=None):
    """The incumbent learner as G2 teaches it (2.5): the class as the live brain builds it plus the stack
    `Brain.__init__` attaches (n_neurons 9, the brain's seed); taught_campaign's B8 twin proves it equal to the
    Brain's own learner. `n_in` serves the frozen front-ends (the same class over H inputs); `lr` the lr-0 red."""
    from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    options = {} if lr is None else {"lr": lr}
    unit = SpikingInteractionLearner(
        list(im._IXN_INTENTS), n_in=n_in, seed=seed, t_post=im.T_POST, **options
    )
    unit.stack = PlasticityRuleStack(n_neurons=len(unit.responses), seed=seed)
    return unit


def main():
    """The fast self-check: B9, B11, B12 on dev seed 8000, every P2 red on exposed seed 8900, and the
    presentation counts of every problem."""
    assert check_b9()["ok"]
    assert check_b11()["ok"]
    assert check_b12(canonical_incumbent(8000), PROBLEMS["i.S0C1"], 8000)["ok"]
    reds = construction_reds(canonical_incumbent)
    assert reds["all_fired"], reds["fired"]
    counts = {pid: p.presentations(8000, "validation") for pid, p in PROBLEMS.items()}
    print(f"self-check OK: B9, B11, B12; {sum(reds['fired'].values())} of {len(reds['fired'])} reds fired; "
          f"validation presentations {counts}")


if __name__ == "__main__":
    main()
