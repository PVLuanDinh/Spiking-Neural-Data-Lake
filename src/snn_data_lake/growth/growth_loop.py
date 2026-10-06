"""specs/022 T111 (G3) P3: the grow-attempt loop, the budget pause, the problem ledger and the unit journal.

Pre-registration: specs/022-unified-brain-arc/evidence/g3-growth-loop.md 2.4, 2.6, 2.9 C8/C13/C18, 2.12a.
Owner 2026-09-25 (POLICIES P4d): grow only while a problem's solved predicate fails; stop, attempt, repeat; pause
at a budget relative to a persisted running baseline (P1); at the limit report UNSOLVED, stop growing, keep the
problem OPEN to retry after sleep or new teaching; journal units into the lake when grown.

- THE LOOP. Grow one whole batch -> attempt (learning ON, E epochs) -> solved? stop : repeat, until the gate
  refuses the next batch. LITERAL grows before every attempt (the owner's rule); CERTIFICATE-GATED grows only
  while the learner's diagnosis licenses growth (CAPACITY / STATE / STATE+CAPACITY) and otherwise attempts
  without growth, at most A_max = 3 consecutive times, ending LEARNING_RULE_DEFICIT (a separator exists) or
  UNRESOLVED (2.4). DATA grows nothing and attempts nothing (CONTRADICTORY). Every terminal state is a ledger
  row; none reports solved; every one but SOLVED keeps the problem OPEN.
- THE PAUSE is GrowthStop's problem key (one stop rule): grow only while unsolved and the next whole batch fits
  the allowance (D5). A retry with no new teaching draws no fresh budget: it attempts once, without growth.
- THE BUDGET: G3 owns the mechanism, G2's Problem record the VALUES (BudgetValues). The first positive-spend
  problem gets `first` (52, the frozen code's width); afterwards beta x median(spend of SOLVED problems with
  POSITIVE spend), read from the ledger. New teaching draws one FRESH budget (spend under it starts at 0; no
  carry-over of an unspent remainder), at most k_max per problem lifetime.
- THE LEDGER and THE UNIT JOURNAL are event rows on GoalStore's IMMORTAL _T_GOAL class ("problem" and "unit";
  G6's sleep kinds pass through untouched). Every problem row carries a per-problem `seq`; a gap refuses the
  fold (C13: a dropped row is never silently re-derived). A "unit" row is written at admission, BEFORE the
  learner receives the unit (write-ahead), and carries the weights themselves.
- ISOLATION (2.12a): the ledger takes an EXPLICIT lake path and refuses the live home.

A source returns units, or (units, evaluations) when it searches (2.6).
The learner is an adapter with diagnose() / admit(units) / attempt(epochs); the taught learner's adapter lands
with the grown-segment port after R2's P1.
"""

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from snn_data_lake.growth.growth_stop import GrowthStop
from snn_data_lake.growth.learning_progress import GoalStore
from snn_data_lake.substrate.home import LiveHomeRefused, assert_not_live_home

#: diagnoses that license growth (2.4); every other kind attempts without growth or not at all
GROWTH_KINDS = ("CAPACITY", "STATE", "STATE+CAPACITY")
A_MAX = 3  # consecutive no-growth attempts per budget (2.6 D3), a disclosed session call
ENCODER = "text_features n_in=52"


class LedgerError(RuntimeError):
    """A refused ledger operation: a missing or live path, a seq gap, a starved budget, an exhausted retry."""


@dataclass(frozen=True)
class BudgetValues:
    first: float  # the first positive-spend problem's budget (G2: 52, the frozen code's width)
    beta: float  # the multiple of the running baseline (G2's value, fixed on dev seeds)
    k_max: int = 3  # fresh budgets per problem lifetime (O2)


@dataclass
class ProblemRecord:
    problem_id: str
    state: str = "OPEN"
    open: bool = True
    budget: float = 0.0
    budgets_drawn: int = 0
    spend_units: int = 0  # every unit this problem ever admitted
    spend_since_teach: int = 0  # units admitted under the CURRENT budget (a fresh budget resets it)
    spend_evals: int = 0
    runs_this_budget: int = 0  # loop runs started under the CURRENT budget; a second one is a RETRY
    attempts: int = 0
    baseline_at_open: float = None
    n_codes: int = 0
    history: list = field(default_factory=list, repr=False, compare=False)


def retention_ok(before, after, f_bound):
    """C11's factual check: A's drop in items (before - after, learning OFF) is within F_bound (D7)."""
    return (before - after) <= f_bound


def _refuse_path(path):
    """2.12a: an EXPLICIT lake path, never None (GoalStore would resolve None to the home lake), and
    never inside the live home -- through THE shared guard (R1, lead ruling 16), which raises
    LiveHomeRefused; no row-local live-home check (P2)."""
    if path is None:
        raise LedgerError(
            "the ledger needs an explicit lake path (2.12a): None would resolve to the home lake"
        )
    # the guard resolves strictly (the path must exist): an EXISTING lake file is guarded itself (realpath
    # follows a symlink or junction, so an alias of the live lake is refused -- Codex cp5 #1); a fresh lake
    # file does not exist yet, so its DIRECTORY is guarded and the file name kept
    path = Path(path)
    if path.exists():
        return Path(assert_not_live_home(str(path)))
    return Path(assert_not_live_home(str(path.parent))) / path.name


class Ledger:
    """The problem ledger and the unit journal over GoalStore's _T_GOAL rows at an explicit lake path."""

    def __init__(self, path, values):
        self._path = _refuse_path(path)
        self.values = values
        self._store = GoalStore(str(self._path))

    # ---- rows ----------------------------------------------------------------------------------------
    def _append_raw(self, row):
        self._store.append_event(row)

    def events(self):
        """Rows of kinds this ledger does not fold (G6's sleep rows, any later owner's), in lake order."""
        return [r for r in self._store.events() if r.get("kind") not in ("problem", "unit")]

    def units(self):
        return [r for r in self._store.events() if r.get("kind") == "unit"]

    def _problem_rows(self, problem_id):
        return [
            r
            for r in self._store.events()
            if r.get("kind") == "problem" and r.get("problem_id") == problem_id
        ]

    def _write(self, problem_id, op, **fields):
        row = {
            "kind": "problem",
            "problem_id": problem_id,
            "seq": len(self._problem_rows(problem_id)),
            "op": op,
        }
        row.update(fields)
        self._append_raw(row)

    # ---- the fold ------------------------------------------------------------------------------------
    def ledger(self, problem_id):
        rows = self._problem_rows(problem_id)
        if not rows:
            raise KeyError(f"unknown problem {problem_id!r}")
        rec = ProblemRecord(problem_id)
        for expect, row in enumerate(rows):
            if row.get("seq") != expect:
                raise LedgerError(
                    f"problem {problem_id!r}: ledger row seq {row.get('seq')} where {expect} was due"
                )
            op = row["op"]
            if op == "open":
                rec.budget, rec.baseline_at_open = float(row["budget"]), row["baseline"]
                rec.n_codes, rec.budgets_drawn = int(row["n_codes"]), 1
            elif op == "teach":
                rec.budgets_drawn += 1
                rec.spend_since_teach = (
                    0  # one FRESH budget per new teaching episode, no carry-over
                )
                rec.runs_this_budget = 0
            elif op == "run":
                rec.runs_this_budget += 1
            elif op == "spend":
                rec.spend_units += int(row["units"])
                rec.spend_since_teach += int(row["units"])
                rec.spend_evals += int(row["evals"])
            elif op == "attempt":
                rec.attempts += 1
                if row["solved"]:
                    rec.state, rec.open = "SOLVED", False
            elif op == "state":
                rec.state = row["state"]
            rec.history.append(op)
        return rec

    def problem_ids(self):
        seen = []
        for r in self._store.events():
            if r.get("kind") == "problem" and r.get("op") == "open" and r["problem_id"] not in seen:
                seen.append(r["problem_id"])
        return seen

    def baseline(self):
        """median(spend of SOLVED problems with POSITIVE spend), or None before the first one (P1)."""
        spends = []
        for pid in self.problem_ids():
            rec = self.ledger(pid)
            if rec.state == "SOLVED" and rec.spend_units > 0:
                spends.append(rec.spend_units)
        return statistics.median(spends) if spends else None

    # ---- the API -------------------------------------------------------------------------------------
    def open_problem(self, problem_id, n_codes):
        if problem_id in self.problem_ids():
            return self.ledger(problem_id)
        base = self.baseline()
        budget = self.values.first if base is None else self.values.beta * base
        if budget < n_codes:
            raise LedgerError(
                f"problem {problem_id!r}: budget {budget} < {n_codes} distinct taught codes -- a construction/budget "
                "mismatch reported to G2 and the lead, never run with a starved budget (D5)"
            )
        self._write(problem_id, "open", budget=budget, baseline=base, n_codes=int(n_codes))
        return self.ledger(problem_id)

    def record_spend(self, problem_id, units, evals):
        self._write(problem_id, "spend", units=int(units), evals=int(evals))

    def record_run(self, problem_id):
        self._write(problem_id, "run")

    def record_attempt(self, problem_id, solved, epochs=None):
        self._write(problem_id, "attempt", solved=bool(solved), epochs=epochs)

    def set_state(self, problem_id, state):
        self._write(problem_id, "state", state=state)

    def new_teaching(self, problem_id):
        """A new teaching episode of the same problem draws one fresh budget, at most k_max per lifetime."""
        rec = self.ledger(problem_id)
        if rec.budgets_drawn >= self.values.k_max:
            raise LedgerError(
                f"problem {problem_id!r}: {self.values.k_max} budgets drawn; growth exhausted, OPEN"
            )
        self._write(problem_id, "teach")
        if rec.state != "SOLVED":
            self.set_state(problem_id, "OPEN")

    def known_unit(self, unit_id):
        """True for EVERY id ever born -- the birth rows are IMMORTAL (_T_GOAL 0x0B), pruned units included."""
        return any(r["unit_id"] == int(unit_id) for r in self.units())

    def record_units(self, units, problem_id, batch_id):
        """One batch of birth rows; a duplicate id (within the batch or already journaled) refuses the WHOLE
        batch before any write (Codex cp5 #5)."""
        ids = [int(u.unit_id) for u in units]
        known = {r["unit_id"] for r in self.units()}
        dup = sorted({i for i in ids if ids.count(i) > 1} | (set(ids) & known))
        if dup:
            raise LedgerError(f"problem {problem_id!r}: unit ids {dup} are already born or repeated in the batch")
        for u in units:
            self._record_unit(u, problem_id, batch_id)

    def _record_unit(self, unit, problem_id, batch_id):
        """The IMMORTAL birth row, written BEFORE the learner receives the unit (write-ahead)."""
        body = {
            "unit_id": int(unit.unit_id),
            "w_in": [float(x) for x in unit.w_in],
            "threshold": float(unit.threshold),
            "window": int(unit.window),
            "sign": int(unit.sign),
            "out": [float(x) for x in unit.out],
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        self._append_raw(
            {
                "kind": "unit",
                **body,
                "problem_id": problem_id,
                "batch_id": int(batch_id),
                "origin": "born",
                "encoder": ENCODER,
                "digest": digest,
            }
        )

    def record_event(self, row):
        """The PUBLIC entry for another owner's row on this lake (G6's sleep_bout / sleep_verdict /
        sleep_mask_prepared / sleep_mask_commit): one writer per lake (P2), no private call and no
        second GoalStore. The ledger's own kinds ("problem", "unit") are refused here -- they have
        their own API -- and GoalStore.append_event refuses goal kinds and NaN."""
        if not isinstance(row, dict) or row.get("kind") in ("problem", "unit"):
            raise ValueError(f"record_event takes another owner's row, got {row!r}")
        self._append_raw(dict(row))

    def record_sleep_save(self, payload):
        """G6's API: one "sleep_save" row on this class (the fold passes it through untouched)."""
        self._append_raw({"kind": "sleep_save", **dict(payload)})


# distinct-from: growth/function_preserving_growth.py -- its `_eligibility_seed` points a Net2Net/GradMax unit at
#   the MEAN input energy of a window (a disclosed unproven heuristic, no target); this trains a candidate on the
#   multiclass RESIDUAL through a fixed random feedback vector. Disclosed for the panel as a P2 question: if G3's
#   primary family deletes the trace-trained source (P8 clause 5), neither survives as a residual-directed seed.
# distinct-from: faculties/homeostatic_drives.py, engine/brain.py, faculties/wholistic_loop.py,
#   growth/growth_stop.py -- read 2026-10-04 (the duplicate guard's hits): a drive-preference softmax, a table
#   checksum, a metacognitive-signal source and a stop criterion; none trains a candidate unit. The softmax is
#   REUSED from infra/_mathutils.py, not rewritten.
def trace_train(w, threshold, window, tau, items, eta, epochs, seed):
    """THE TRACE-TRAINED CANDIDATE SOURCE (2.5; moved here from candidate_lifecycle by cp1 #19: it is the
    PROBLEM consumer's source, so it lives with that consumer, and the live lifecycle no longer
    references grown_code): one capacity candidate's input row trained on the multiclass
    residual with an eligibility trace and a FIXED RANDOM feedback vector -- never a transposed forward weight
    (the e-prop invariant; 2.9 C7).

    `items` = [(ranks, y_onehot, p)] with p = softmax(v / T) of the learner's CURRENT label drives at the
    registered T = 1 drive unit (D4), fixed while the candidate trains (it has no output synapses before
    admission). Per item, e_i = PSP(t_ref - rank_i) at the candidate's membrane, t_ref = its first-spike step
    (the window end when it does not spike), and L = sum_c B_c (y_c - p_c) with B drawn once at the candidate's
    birth from `seed`. The update is the pre-registered dW_i = -eta * L * e_i; B is drawn from a symmetric law
    (B and -B equally likely), so the sign convention does not change the rule's distribution. Returns the
    trained row as a tuple."""
    import math
    import random

    from snn_data_lake.growth.grown_code import GrownUnit, latency

    rng = random.Random(seed)
    n_labels = len(items[0][1])
    feedback = [rng.gauss(0.0, 1.0) for _ in range(n_labels)]  # B: fixed at birth, never learned
    w = list(w)
    for _ in range(int(epochs)):
        for ranks, y, p in items:
            unit = GrownUnit(0, tuple(w), threshold, window, 1, ())
            t = latency(unit, ranks, tau)
            t_ref = window if t is None else t
            e = [math.exp(-(t_ref - r) / tau) if t_ref > r else 0.0 for r in ranks]
            L = sum(b * (yc - pc) for b, yc, pc in zip(feedback, y, p))
            for i in range(len(w)):
                w[i] += -eta * L * e[i]
    return tuple(w)


def label_probabilities(drives):
    """p = softmax(drives / T) at the registered T = 1 drive unit (D4), via the shared helper."""
    from snn_data_lake.infra._mathutils import softmax

    return softmax(list(drives))


class GrowthLoop:
    """One posed problem's grow-attempt loop (2.6)."""

    def __init__(
        self,
        learner,
        problem_id,
        ledger,
        source,
        batch,
        gated=True,
        diagnosis=True,
        epochs=400,
        a_max=A_MAX,
        n_codes=4,
        gate=None,
        solved_sink=None,
    ):
        self.learner, self.problem_id, self.ledger = learner, problem_id, ledger
        if int(batch) <= 0:
            raise ValueError(f"batch must be positive, got {batch!r}")
        self.source, self.batch, self.gated, self.diagnosis = (
            source,
            int(batch),
            bool(gated),
            bool(diagnosis),
        )
        self.epochs, self.a_max, self.n_codes = int(epochs), int(a_max), int(n_codes)
        self.gate = gate if gate is not None else GrowthStop()
        self.solved_sink = (
            solved_sink  # called once per OPEN -> SOLVED transition (G6's scheduler gate)
        )
        self._batches = 0

    def _may_grow(self, rec):
        self.gate.observe_problem(
            self.problem_id,
            solved=rec.state == "SOLVED",
            spend=rec.spend_since_teach,
            budget=rec.budget,
            batch=self.batch,
        )
        return self.gate.should_grow_problem(self.problem_id)

    def _grow(self):
        """True when one whole batch was admitted; False when the source had none to give (Codex cp5 #4)."""
        out = self.source(self.batch)
        # a searching source returns (units, evaluations) so the matched-search-budget controls are paid in
        # the same currency (2.6; Codex cp5 #7); a plain list evaluated nothing it can report
        units, evals = out if isinstance(out, tuple) else (out, 0)
        if not units:
            if evals:
                self.ledger.record_spend(self.problem_id, 0, evals=evals)
            return False
        if len(units) != self.batch:  # the gate cleared THIS size; anything else could cross the budget
            raise LedgerError(
                f"problem {self.problem_id!r}: the source returned {len(units)} units for a batch of {self.batch}"
            )
        self.ledger.record_units(units, self.problem_id, self._batches)  # write-ahead, before the learner
        self.ledger.record_spend(self.problem_id, len(units), evals=evals)
        self.learner.admit(units)
        self._batches += 1
        return True

    def _attempt(self):
        solved = bool(self.learner.attempt(self.epochs))
        self.ledger.record_attempt(self.problem_id, solved, self.epochs)
        if solved and self.solved_sink is not None:
            self.solved_sink(self.ledger.ledger(self.problem_id))
        return solved

    def run(self):
        rec = self.ledger.open_problem(self.problem_id, self.n_codes)
        if not rec.open:
            return rec
        if self.diagnosis and self.learner.diagnose() == "DATA":
            self.ledger.set_state(self.problem_id, "CONTRADICTORY")
            return self.ledger.ledger(self.problem_id)
        self.ledger.record_run(self.problem_id)
        if rec.runs_this_budget > 0:
            # a retry (a run already started under this budget, spent or not): no new teaching, so no fresh
            # budget -- attempts only, growth frozen (2.6 retry rule; Codex cp5 #3)
            if self._attempt():  # a solve on the retry stands (Codex cp5 #2)
                return self.ledger.ledger(self.problem_id)
            return self._paused()
        idle = 0
        while True:
            rec = self.ledger.ledger(self.problem_id)
            kind = self.learner.diagnose() if (self.gated and self.diagnosis) else "CAPACITY"
            if kind == "DATA":
                self.ledger.set_state(self.problem_id, "CONTRADICTORY")
                return self.ledger.ledger(self.problem_id)
            if kind in GROWTH_KINDS:
                if not self._may_grow(rec):
                    return self._paused()
                if not self._grow():
                    return self._paused()
                idle = 0
            else:
                if idle >= self.a_max:
                    terminal = "LEARNING_RULE_DEFICIT" if kind == "SEPARABLE" else "UNRESOLVED"
                    self.ledger.set_state(self.problem_id, terminal)
                    return self.ledger.ledger(self.problem_id)
                idle += 1
            if self._attempt():
                return self.ledger.ledger(self.problem_id)

    def _paused(self):
        self.ledger.set_state(self.problem_id, "UNSOLVED")
        return self.ledger.ledger(self.problem_id)


def main():
    """Self-check on a scripted learner in a throwaway lake: the pause (C8), the attempt cap (C18), the ledger
    rebuild and its dropped-row red (C13), and the live-lake refusal (2.12a)."""
    import tempfile

    from snn_data_lake.growth.grown_code import GrownUnit

    class Scripted:
        def __init__(self, kind, solve_after=None):
            self.kind, self.solve_after, self.n = kind, solve_after, 0

        def diagnose(self):
            return self.kind

        def admit(self, units):
            self.n += len(units)

        def attempt(self, epochs):
            return self.solve_after is not None and self.n >= self.solve_after

    ids = iter(range(10**6))

    def source(n):
        return [
            GrownUnit(next(ids), (0.0,), 1.0, 12, -1 if i == n - 1 and n > 1 else 1, (0.0,))
            for i in range(n)
        ]

    values = BudgetValues(first=52, beta=1.5)
    with tempfile.TemporaryDirectory() as d:
        lake = Path(d) / "lake.bin"
        paused = GrowthLoop(Scripted("CAPACITY"), "u", Ledger(lake, values), source, 9).run()
        assert paused.state == "UNSOLVED" and paused.open and paused.spend_units == 45, paused
        capped = GrowthLoop(Scripted("SEPARABLE"), "s", Ledger(lake, values), source, 9).run()
        assert (
            capped.state == "LEARNING_RULE_DEFICIT"
            and capped.attempts == A_MAX
            and capped.spend_units == 0
        )
        solved = GrowthLoop(Scripted("CAPACITY", 18), "x", Ledger(lake, values), source, 9).run()
        assert solved.state == "SOLVED" and Ledger(lake, values).ledger("x") == solved
        ledger = Ledger(lake, values)
        ledger._append_raw(
            {"kind": "problem", "problem_id": "x", "seq": 99, "op": "attempt", "solved": False}
        )
        try:
            Ledger(lake, values).ledger("x")
            raise AssertionError("red: a seq gap must refuse the fold")
        except LedgerError:
            pass
    try:
        Ledger(Path.home() / ".snn_data_lake" / "brain_lake.bin", values)
        raise AssertionError("red: the live lake must be refused")
    except LiveHomeRefused:
        pass
    print(
        "growth_loop: pause at 45/52 (UNSOLVED, OPEN), attempt cap -> LEARNING_RULE_DEFICIT, ledger rebuilt "
        "equal, seq-gap red fires, live lake refused"
    )


if __name__ == "__main__":
    main()
