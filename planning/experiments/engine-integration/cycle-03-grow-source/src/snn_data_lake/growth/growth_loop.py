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
The learner is an adapter with diagnose() / admit(units, labels, ticket) / attempt(epochs); the taught learner's adapter lands
with the grown-segment port after R2's P1.
"""

import copy
import hashlib
import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from snn_data_lake.gates.unit_label_gate import LABEL_FIELDS, require_admissible
from snn_data_lake.growth.grown_code import apply_unit_row
from snn_data_lake.growth.growth_stop import GrowthStop
from snn_data_lake.growth.learning_progress import GoalStore
from snn_data_lake.growth.placement import place_labeled
from snn_data_lake.substrate.home import LiveHomeRefused, assert_not_live_home

GRADIENT_CLAIM = {
    "trace_train": (
        "LOCAL",
        "one capacity candidate's input row, each synapse updated from its OWN PSP eligibility trace times ONE "
        "broadcast scalar L = sum_c B_c (y_c - p_c), with B a FIXED random vector drawn at birth from the seed "
        "and never learned (2.5, 2.9 C7). No error crosses a layer, no forward weight is transported or "
        "transposed, and the pre-registration claims no gradient (it states the update, never that it tracks "
        "one), so there is no cosine to owe; its alignment with the true gradient is neither measured nor "
        "claimed. Re-classify as a claimant, and owe the cosine > 0.5 bar, the day any text says it tracks one",
    ),
}

#: diagnoses that license growth (2.4); every other kind attempts without growth or not at all
GROWTH_KINDS = ("CAPACITY", "STATE", "STATE+CAPACITY")
A_MAX = 3  # consecutive no-growth attempts per budget (2.6 D3), a disclosed session call
ENCODER = "text_features n_in=52"


class LedgerError(RuntimeError):
    """A refused ledger operation: a missing or live path, a seq gap, a starved budget, an exhausted retry."""


def _int_id(value, what):
    """An id is a non-negative int, never coerced (a 1.5, True or "7" journaled as 1/1/7 while the learner
    holds the original would split the journal's id from the unit's; G7's gate refuses the same types)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{what} must be a non-negative int, got {value!r}")
    return value


@dataclass(frozen=True)
class BudgetValues:
    first: float  # the first positive-spend problem's budget (G2: 52, the frozen code's width)
    beta: float | None  # None until G3's dated dev amendment; the first problem needs no beta
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
    evidence: dict = None  # the terminal row's diagnosis evidence (the separating witness), if the adapter gave any
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
    # the guard resolves strictly (the path must exist). The LEXICAL leaf is resolved first, non-strictly, so
    # a symlink or junction is followed even when DANGLING (a link to the live lake file that does not exist
    # yet: exists() is False, and BinLog would create the live lake through it -- Codex panel BLOCKER). An
    # existing target is guarded itself; a fresh one has only its DIRECTORY guarded and the name kept.
    target = Path(os.path.realpath(path))
    if target.exists():
        return Path(assert_not_live_home(str(target)))
    return Path(assert_not_live_home(str(target.parent))) / target.name


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
                rec.evidence = row.get("evidence", rec.evidence)
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
        if base is None:
            budget = self.values.first
        elif self.values.beta is None:
            # G3 2.8: SEQUENCE alone needs the source-owned beta. Preserve G2's
            # named refusal BEFORE a new ledger row, never substitute first.
            from snn_data_lake.instruments.taught_problems import budget as problem_budget

            budget = problem_budget(None, (base,))
        else:
            budget = self.values.beta * base
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

    def record_attempt(self, problem_id, solved, epochs=None, *, exposure=None):
        extra = {} if exposure is None else {"exposure": copy.deepcopy(exposure)}
        self._write(problem_id, "attempt", solved=bool(solved), epochs=epochs, **extra)

    def set_state(self, problem_id, state, evidence=None):
        extra = {} if evidence is None else {"evidence": evidence}
        self._write(problem_id, "state", state=state, **extra)

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
        # ponytail: a full scan of the birth rows per call (fixture size); cache the id set if live births get large
        unit_id = _int_id(unit_id, "unit_id")
        return any(r["unit_id"] == unit_id for r in self.units())

    def record_units(self, units, labels, problem_id, batch_id):
        """One batch of birth rows, each carrying its unit's G7 LABEL (the journal is the one birth record that
        `classify_restored` and `check_saved` read). A duplicate id (within the batch or already journaled), a
        label that is not this problem's/batch's or a unit sign that is not its label's refuses the WHOLE batch
        before any write (Codex cp5 #5)."""
        batch_id = _int_id(batch_id, "batch_id")
        ids = [_int_id(u.unit_id, "unit_id") for u in units]
        known = {r["unit_id"] for r in self.units()}
        dup = sorted({i for i in ids if ids.count(i) > 1} | (set(ids) & known))
        if dup:
            raise LedgerError(
                f"problem {problem_id!r}: unit ids {dup} are already born or repeated in the batch"
            )
        by_id = {lab["unit_id"]: lab for lab in labels}
        if sorted(by_id) != sorted(ids) or len(by_id) != len(labels):
            raise LedgerError(
                f"problem {problem_id!r}: the label records do not name exactly the batch's unit ids"
            )
        for u in units:
            lab = by_id[u.unit_id]
            if (
                lab["problem_id"] != problem_id
                or lab["batch_id"] != batch_id
                or lab["sign"] != u.sign
            ):
                raise LedgerError(
                    f"unit {u.unit_id}: its label (problem, batch, sign) is not this batch's / this unit's"
                )
        rows = [self._birth_row(u, by_id[u.unit_id], problem_id, batch_id) for u in units]
        for row in rows:  # every row serialises BEFORE any is appended (the lake is immortal)
            json.dumps(row, allow_nan=False)
        for row in rows:
            self._append_raw(row)

    def unsettled(self, problem_id):
        """Births journaled for `problem_id` minus the admissions its spend rows charged: nonzero after a crash
        between the birth rows and the spend row (amendment (d)); no reconciler exists, so growth refuses."""
        born = sum(1 for r in self.units() if r["problem_id"] == problem_id)
        return born - self.ledger(problem_id).spend_units

    def next_batch_id(self):
        """One past the highest batch id in the immortal birth rows: monotone across loop instances,
        problems and restarts, so `labels_by(records, "batch_id")` never merges two columns."""
        return max((r["batch_id"] for r in self.units()), default=-1) + 1

    def birth_records(self):
        """{unit_id: birth row} over every unit ever born (immortal, pruned included): classify_restored's journal."""
        return {r["unit_id"]: r for r in self.units()}

    def first_sign(self):
        """{unit_id: the sign written at birth}: check_saved's Dale reference."""
        return {r["unit_id"]: r["sign"] for r in self.units()}

    def _birth_row(self, unit, label, problem_id, batch_id):
        """The IMMORTAL birth row, appended BEFORE the learner receives the unit (write-ahead)."""
        body = {
            "unit_id": int(unit.unit_id),
            "w_in": [float(x) for x in unit.w_in],
            "threshold": float(unit.threshold),
            "window": int(unit.window),
            "sign": int(unit.sign),
            "out": [float(x) for x in unit.out],
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        # the label fields come from the label record (LABEL_FIELDS is G7's tuple, imported, never copied);
        # position is a tuple in the record and a list after the lake's JSON, so it is stored as the list
        born = {k: label[k] for k in LABEL_FIELDS}
        born["position"] = list(label["position"])
        return {
            "kind": "unit",
            **body,
            **born,
            "problem_id": problem_id,
            "batch_id": int(batch_id),
            "origin": "born",
            "encoder": ENCODER,
            "digest": digest,
        }

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


#: G3's seed block (SEEDS.md): every seed the loop hands a randomness stream stays inside it
SEED_BLOCK = (9000, 10000)


def make_admission(*, site, centre_um, density_per_mm2, seed):
    """The admission a GrowthLoop needs: G7's `place_labeled` with the campaign's site, centre and density (no
    defaults here: G4's Part B measures the density), each unit's role read off its Dale sign (E +1, I -1) so a
    role/sign disagreement cannot exist, then G7's gate. Returns admission(units, problem_id, batch_id, *,
    resolve_problem, known_unit) -> (labels, ticket). The labels are created and gated in ONE function, which is
    what G7's creation census requires; the ticket is consumed by `grown_code.admit_rows`. `seed` has no default
    and must lie in G3's block; the per-batch stream is the block-relative sum modulo the block width, so it
    never leaves the block (a default 0 + batch 300 once reached a protected seed)."""
    lo, hi = SEED_BLOCK
    if isinstance(seed, bool) or not isinstance(seed, int) or not lo <= seed < hi:
        raise ValueError(f"seed {seed!r} is outside G3's block {lo}-{hi - 1}")

    def admission(units, problem_id, batch_id, *, resolve_problem, known_unit):
        labels = place_labeled(
            len(units),
            unit_ids=[u.unit_id for u in units],
            region="cortex",  # a LITERAL: the growth-coverage census attributes the call by it; only cortex has a measured Dale sign (O3)
            site=site,
            problem_id=problem_id,
            batch_id=batch_id,
            seed=lo + (seed - lo + batch_id) % (hi - lo),
            roles=["E" if u.sign > 0 else "I" for u in units],
            centre_um=centre_um,
            density_per_mm2=density_per_mm2,
        )
        ticket = require_admissible(labels, resolve_problem=resolve_problem, known_unit=known_unit)
        return labels, ticket

    return admission


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
        *,
        admission,
        resolve_problem,
    ):
        if isinstance(learner, CanonicalGrowthAdapter):
            CanonicalGrowthAdapter.require_ready(learner)
            CanonicalGrowthAdapter.require_ready(learner, ledger=ledger)
            CanonicalGrowthAdapter.check_loop_binding(learner, ledger, source, batch, epochs)
        self.learner, self.problem_id, self.ledger = learner, problem_id, ledger
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError(f"batch must be a positive int, got {batch!r}")
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
        # the admission (labels + G7's gate, `make_admission`) and the problem resolver are REQUIRED: no grown
        # unit can reach the learner without a ticket (the 2026-10-04 amendment)
        self.admission, self.resolve_problem = admission, resolve_problem
        self._ran = False

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
        if self.ledger.unsettled(self.problem_id):
            raise LedgerError(
                f"problem {self.problem_id!r}: {self.ledger.unsettled(self.problem_id)} unsettled birth row(s), none "
                "charged (a crash between them); growth refuses until the reconciler lands"
            )
        batch_id = self.ledger.next_batch_id()
        out = self.source(self.batch)
        # a searching source returns (units, evaluations) so the matched-search-budget controls are paid in
        # the same currency (2.6; Codex cp5 #7); a plain list evaluated nothing it can report
        units, evals = out if isinstance(out, tuple) else (out, 0)
        try:
            if (
                units and len(units) != self.batch
            ):  # the gate cleared THIS size; anything else could cross the budget
                raise LedgerError(
                    f"problem {self.problem_id!r}: the source returned {len(units)} units for a batch of {self.batch}"
                )
            if units:
                # the WHOLE admission is dry-run against the learner's incumbent segment: grown_code's
                # refusals (a reused id, non-finite weights, a bad window, Dale) fire BEFORE any birth row
                segment = tuple(self.learner.segment)
                for u in units:
                    segment = apply_unit_row(segment, {"op": "admit", "unit": u})
                labels, ticket = self.admission(
                    units,
                    self.problem_id,
                    batch_id,
                    resolve_problem=self.resolve_problem,
                    known_unit=self.ledger.known_unit,
                )
                labels = list(labels)
                self.ledger.record_units(units, labels, self.problem_id, batch_id)  # write-ahead
        except (LedgerError, ValueError):  # UnitLabelRefused is a ValueError
            self._pay_search(evals)  # a refused batch still paid its search (G7 refutation 1)
            raise
        if not units:
            self._pay_search(evals)
            return False
        self._mark_run()
        self.ledger.record_spend(self.problem_id, len(units), evals=evals)
        self._record_search()
        before = tuple(self.learner.segment)
        self.learner.admit(
            units, labels, ticket
        )  # a conforming adapter appends through grown_code.admit_rows (none exists in src yet: the learner port waits on R2's P1)
        if (
            not ticket.consumed
        ):  # detection after the fact: the loop cannot un-admit (2026-10-06 amendment (e))
            raise LedgerError(
                f"problem {self.problem_id!r}: the adapter admitted a batch without consuming its ticket"
            )
        # G7 finding 4: a spent ticket proves the labels were gated, not that the segment grew by THESE units
        old_ids = {u.unit_id for u in before}
        after = tuple(self.learner.segment)
        added = sorted((u for u in after if u.unit_id not in old_ids), key=lambda u: u.unit_id)
        kept = tuple(u for u in after if u.unit_id in old_ids)
        if added != sorted(units, key=lambda u: u.unit_id) or kept != before:
            raise LedgerError(
                f"problem {self.problem_id!r}: the adapter's segment is not the incumbent plus exactly the "
                "admitted units (an extra, substituted or altered unit); the loop cannot un-admit"
            )
        return True

    def _mark_run(self):
        """The run row is written at a run's FIRST journal write, never at its start: a run that journals
        nothing (a refusal before any write) is no run, so the next one is not a retry (2.6)."""
        if not self._ran:
            self.ledger.record_run(self.problem_id)
            self._ran = True

    def _pay_search(self, evals):
        if evals:
            self._mark_run()
            self.ledger.record_spend(self.problem_id, 0, evals=evals)
            self._record_search()

    def _record_search(self):
        if type(self.source) is CanonicalGrowthSource and hasattr(self.source, "last_receipt"):
            self.ledger.record_event(self.source.last_receipt)
            del self.source.last_receipt

    def _attempt(self):
        solved = bool(self.learner.attempt(self.epochs))
        self._mark_run()
        exposure = getattr(self.learner, "last_attempt", None)
        actual = self.epochs if exposure is None else exposure["epochs"]
        self.ledger.record_attempt(self.problem_id, solved, actual, exposure=exposure)
        if solved and self.solved_sink is not None:
            self.solved_sink(self.ledger.ledger(self.problem_id))
        return solved

    def run(self):
        from snn_data_lake.growth.batch_reconciler import reconcile_batches

        reconcile_batches(self.ledger, self.problem_id)
        rec = self.ledger.open_problem(self.problem_id, self.n_codes)
        if not rec.open:
            return rec
        if self.diagnosis and self.learner.diagnose() == "DATA":
            self.ledger.set_state(self.problem_id, "CONTRADICTORY")
            return self.ledger.ledger(self.problem_id)
        self._ran = False
        if (
            rec.runs_this_budget > 0
        ):  # read BEFORE any write of this run: `rec` is the replayed ledger
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
                    # the adapter's separating witness (weights + verified reads), journaled with the
                    # terminal decision (2.6; Codex cp5 #9); an adapter without evidence() records none
                    evidence = getattr(self.learner, "evidence", lambda: None)()
                    self.ledger.set_state(self.problem_id, terminal, evidence)
                    return self.ledger.ledger(self.problem_id)
                idle += 1
            if self._attempt():
                return self.ledger.ledger(self.problem_id)

    def _paused(self):
        self.ledger.set_state(self.problem_id, "UNSOLVED")
        return self.ledger.ledger(self.problem_id)


def _cycle_recipe(value):
    """Explicit engineering recipe; no defaults, global calibration or learned values."""
    import math

    keys = {
        "schema",
        "batch_size",
        "candidate_pool",
        "input_sigma",
        "threshold_min",
        "threshold_max",
        "window",
        "t_post_g",
        "trace_epochs",
        "trace_eta",
        "feed_norm",
        "return_weight",
        "return_delay",
        "max_epochs",
        "stable_epochs",
        "selection",
    }
    if type(value) is not dict or set(value) not in (keys, keys | {"feedback_gate"}):
        raise LedgerError("G9_CYCLE_RECIPE: exact explicit recipe required")
    v = copy.deepcopy(value)
    if "feedback_gate" in v and v["feedback_gate"] != "corrective":
        raise LedgerError("G9_CYCLE_RECIPE: unsupported explicit feedback gate")
    if v["schema"] != "g9.engine.cycle.v1" or v["selection"] != "all-active/no-prune":
        raise LedgerError("G9_CYCLE_RECIPE: unsupported cycle/selection")
    for key in ("batch_size", "candidate_pool", "window", "max_epochs", "stable_epochs"):
        if type(v[key]) is not int or v[key] <= 0:
            raise LedgerError(f"G9_CYCLE_RECIPE: {key} must be a positive exact int")
    for key in ("trace_epochs", "return_delay"):
        if type(v[key]) is not int or v[key] < 0:
            raise LedgerError(f"G9_CYCLE_RECIPE: {key} must be a nonnegative exact int")
    for key in (
        "input_sigma",
        "threshold_min",
        "threshold_max",
        "t_post_g",
        "trace_eta",
        "feed_norm",
        "return_weight",
    ):
        if type(v[key]) not in (int, float) or not math.isfinite(v[key]) or v[key] <= 0:
            raise LedgerError(f"G9_CYCLE_RECIPE: {key} must be finite and positive")
    from snn_data_lake.instruments import inseparable_mapping as im
    from snn_data_lake.instruments import taught_problems as tp

    if (
        v["candidate_pool"] < v["batch_size"]
        or v["threshold_min"] >= v["threshold_max"]
        or v["t_post_g"] <= v["window"]
        or v["max_epochs"] != im.TEACH_EPOCHS
        or v["stable_epochs"] != tp.STABLE
    ):
        raise LedgerError("G9_CYCLE_RECIPE: candidate/time/stable-10 contract")
    return v


class CanonicalGrowthSource:
    """Actual latency candidates, using the incumbent local fit and lifecycle selector.

    A stream is derived from durable batch identity, so a fresh process draws the
    same next pool. Search can return empty and pause; it never pads with dead
    units. LP witnesses are diagnostics and never enter this learned state.
    """

    def __init__(self, adapter, ledger):
        self.adapter, self.ledger = adapter, ledger

    def __call__(self, count):
        import random

        import torch

        from snn_data_lake.growth.candidate_lifecycle import Pool, contribution, select_block
        from snn_data_lake.growth.grown_code import GrownUnit, segment_psp
        from snn_data_lake.instruments import inseparable_mapping as im
        from snn_data_lake.readout.posed_problem import chat_grown_drives

        a, ledger = self.adapter, self.ledger
        binding, actor = a._require_engine()
        recipe = json.loads(binding["config"])["cycle"]
        a.check_loop_binding(ledger, self, count, recipe["max_epochs"])
        owner = binding["owner"]
        with owner._r2_lock:
            if actor.selection is not None:
                raise LedgerError("G9_SELECTION_FROZEN: growth after selection")
            owner._g9_publish_context(actor)
            lo, hi = SEED_BLOCK
            stream = lo + (a.seed - lo + ledger.next_batch_id()) % (hi - lo)
            rng = random.Random(stream)
            ranks = [im.ranks(text) for text in a._texts]
            residual, items = [], []
            for code, target in zip(ranks, a._labels, strict=True):
                drives = [a.learner._drive(n, code) for n in a.learner.neurons]
                grown = chat_grown_drives(a.learner, code)
                p = label_probabilities([x + y for x, y in zip(drives, grown, strict=True)])
                y = [float(label == target) for label in a.learner.responses]
                residual.append([yc - pc for yc, pc in zip(y, p, strict=True)])
                items.append((code, y, p))
            candidates, scores = [], []
            for i in range(recipe["candidate_pool"]):
                row = tuple(rng.gauss(0.0, recipe["input_sigma"]) for _ in range(im.N_IN))
                threshold = rng.uniform(recipe["threshold_min"], recipe["threshold_max"])
                feedback_seed = lo + (stream - lo + i) % (hi - lo)
                row = trace_train(
                    row,
                    threshold,
                    recipe["window"],
                    a.learner.tau,
                    items,
                    recipe["trace_eta"],
                    recipe["trace_epochs"],
                    feedback_seed,
                )
                u = GrownUnit(
                    i, row, threshold, recipe["window"], 1, (0.0,) * len(a.learner.responses)
                )
                feature = [
                    segment_psp((u,), code, a.learner.tau, recipe["t_post_g"])[0] for code in ranks
                ]
                candidates.append(u)
                scores.append(contribution(feature, residual))
            pool = Pool(
                torch.tensor([u.w_in for u in candidates], dtype=torch.float64),
                torch.tensor([u.threshold for u in candidates], dtype=torch.float64),
                torch.tensor(scores, dtype=torch.float64),
                torch.zeros(len(candidates), dtype=torch.bool),
                torch.zeros(len(candidates), dtype=torch.bool),
            )
            chosen = select_block(pool, count, count, len(actor.units))
            next_id = max(ledger.birth_records(), default=-1) + 1
            selected = []
            if len(chosen) == count:
                for j, i in enumerate(chosen):
                    c = candidates[i]
                    unit_id = next_id + j
                    sign = -1 if unit_id % 9 == 8 else 1
                    selected.append(GrownUnit(unit_id, c.w_in, c.threshold, c.window, sign, c.out))
            self.last_receipt = {
                "kind": "engine_source",
                "problem_id": a.problem.id,
                "batch_id": ledger.next_batch_id(),
                "stream_seed": stream,
                "evaluations": len(candidates),
                "selected_source_indices": chosen,
                "selected_scores": [scores[i] for i in chosen],
                "admitted_ids": [u.unit_id for u in selected],
                "complete_batch": len(selected) == count,
            }
            return selected, len(candidates)


class CanonicalGrowthAdapter:
    """Canonical diagnosis plus explicitly bound, same-writer posed operations.

    G3 2.6 admission and A14 retention are ordinary APIs. The deciding loop
    remains closed until the independent G6 mask/progress consumer is built.
    """

    _MISSING = (
        "grown-reader/admission",
        "retention_select",
        "actual grown-source/admission binding",
        "cold restore/Brain integration",
    )

    def __init__(self, learner, problem, *, seed):
        # Check the namespace BEFORE any Problem access or schedule RNG.
        if type(seed) is not int or not SEED_BLOCK[0] <= seed < SEED_BLOCK[1]:
            raise ValueError("canonical seed must be an exact int in SEED_BLOCK")

        from snn_data_lake.instruments import inseparable_mapping as im
        from snn_data_lake.instruments import taught_problems as tp
        from snn_data_lake.instruments.taught_campaign import Reader
        from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack
        from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

        if type(learner) is not SpikingInteractionLearner:
            raise ValueError("canonical learner class required")
        if type(learner.stack) is not PlasticityRuleStack:
            raise ValueError("canonical plasticity stack class required")
        if learner.seed != seed:
            raise ValueError("canonical seed must match learner.seed")
        if learner.n_in != im.N_IN:
            raise ValueError("canonical input width required")
        if learner.t_post != im.T_POST:
            raise ValueError("canonical post time required")
        if tuple(learner.responses) != tuple(im._IXN_INTENTS):
            raise ValueError("canonical response order required")
        if type(problem) is not tp.Problem or not any(
            problem is tp.PROBLEMS[pid] for pid in ("ii.xor", "ii.twin", "aii")
        ):
            raise ValueError("registered ii.xor/ii.twin/aii problem required")

        reader = Reader(learner, history=False)
        epochs = list(problem.teach_epochs(seed, 1))
        if len(epochs) != 1:
            raise ValueError("one canonical construction epoch required")
        epoch = epochs[0]
        texts = tuple(reader.text(episode) for episode in epoch)
        labels = tuple(episode[-1].label for episode in epoch)
        if (
            not texts
            or len(texts) != len(labels)
            or len(texts) != problem.n_teach_codes(seed)
            or len(set(texts)) != len(texts)
            or any(label not in learner.responses for label in labels)
        ):
            raise ValueError("complete canonical construction item set required")
        self.learner, self.problem, self.seed = learner, problem, seed
        self._reader, self._texts, self._labels = reader, texts, labels
        self.last_diagnosis = None

    @classmethod
    def for_brain(cls, owner, problem, *, factory, teach_schedule, projections, cycle=None):
        """Bind the actual canonical owner, never a lookalike or a second learner."""
        adapter = cls(owner.brain._ixn, problem, seed=factory["seed"])
        adapter.bind_engine(owner, factory, teach_schedule, projections, cycle=cycle)
        return adapter

    @classmethod
    def prepare_cycle(cls, owner, problem, *, ledger, recipe):
        """Build an explicit first-budget integration binding, or resume its frozen source.

        These initialization values are experimental inputs, never measurements.
        The existing Brain/actor and the immortal lake remain the only learned state.
        """
        import numpy as np

        from snn_data_lake.growth.grown_code import GrownUnit
        from snn_data_lake.growth.grown_feed import random_feed
        from snn_data_lake.instruments import taught_problems as tp
        from snn_data_lake.instruments.taught_problems import Turn
        from snn_data_lake.readout.posed_problem import unpack_engine_binding
        from snn_data_lake.substrate import home

        recipe = _cycle_recipe(recipe)
        if type(ledger) is not Ledger or ledger._path != Path(
            os.path.realpath(home.knowledge_path())
        ):
            raise LedgerError("G9_CYCLE_LEDGER: actual isolated owner lake required")
        if problem.id not in ("aii", "ii.xor"):
            raise LedgerError("G9_CYCLE_PROBLEM: first integration supports aii/ii.xor only")
        if problem.id in owner._g9_actors:
            factory = copy.deepcopy(owner._g9_actors[problem.id].factory)
            config = unpack_engine_binding(factory.get("engine_binding", {}))
            if config.get("cycle") != recipe:
                raise LedgerError("G9_ENGINE_CONFIG_DRIFT: restored cycle recipe differs")
            schedule = tuple(
                tuple(tuple(Turn(*t) for t in ep) for ep in epoch) for epoch in config["schedule"]
            )
            projections = config["projections"]
        else:
            expected_old = set() if problem.id == "aii" else {"aii"}
            if set(owner._g9_actors) != expected_old or any(
                pid not in ledger.problem_ids() or ledger.ledger(pid).state != "SOLVED"
                for pid in expected_old
            ):
                raise LedgerError(
                    "G9_CYCLE_CURRICULUM: learn aii before ii.xor in this integration"
                )
            if ledger.baseline() is not None:
                # No implicit BETA or first-budget fallback for later sequences.
                raise LedgerError(
                    "G9_CYCLE_SEQUENCE: positive-spend sequence needs source-owned calibration"
                )
            if ledger.values.first != tp.FIRST_BUDGET:
                raise LedgerError("G9_CYCLE_BUDGET: source-owned first budget required")
            count = tp.FIRST_BUDGET
            next_id = max(ledger.birth_records(), default=-1) + 1
            future = [
                GrownUnit(
                    i,
                    (0.0,) * 52,
                    recipe["threshold_min"],
                    recipe["window"],
                    -1 if i % 9 == 8 else 1,
                    (0.0,) * len(owner.brain._ixn.responses),
                )
                for i in range(next_id, next_id + count)
            ]
            columns = np.array([[u.sign * recipe["feed_norm"] / 16 for u in future]] * 256)
            feed = random_feed(columns, future, owner.brain._ixn.seed)
            projections = {
                str(u.unit_id): {
                    "feed": feed[:, j].tolist(),
                    "return": [
                        {
                            "source": label,
                            "sign": 1,
                            "weight": recipe["return_weight"],
                            "delay": recipe["return_delay"],
                        }
                        for label in owner.brain._ixn.responses
                    ],
                }
                for j, u in enumerate(future)
            }
            reserved = (count // recipe["batch_size"] + A_MAX) * recipe["max_epochs"]
            schedule = tuple(
                tuple(tuple(ep) for ep in epoch)
                for epoch in problem.teach_epochs(owner.brain._ixn.seed, reserved)
            )
            factory = {
                "problem_id": problem.id,
                "labels": list(owner.brain._ixn.responses),
                "seed": owner.brain._ixn.seed,
                "arm": "RSTDP_BRAIN",
                "population_origin": "born",
                "units": [],
                "birth_rows": [],
                "w_feed": [[] for _ in range(256)],
                "t_post_g": float(recipe["t_post_g"]),
                "source_signs": {label: 1 for label in owner.brain._ixn.responses},
                "attachments": [],
                "out_eligible": [],
            }
        adapter = cls.for_brain(
            owner,
            problem,
            factory=factory,
            teach_schedule=schedule,
            projections=projections,
            cycle=recipe,
        )
        adapter._cycle_ledger = ledger
        adapter.source = CanonicalGrowthSource(adapter, ledger)
        return adapter

    def check_loop_binding(self, ledger, source, batch, epochs):
        binding, _ = self._require_engine()
        recipe = json.loads(binding["config"]).get("cycle")
        if recipe is None:
            raise LedgerError("G9_A_UNREADY: actual cycle source is unbound")
        _cycle_recipe(recipe)
        if (
            getattr(self, "_cycle_ledger", None) is not ledger
            or type(source) is not CanonicalGrowthSource
            or source.adapter is not self
            or source.ledger is not ledger
            or getattr(self, "source", None) is not source
            or batch != recipe["batch_size"]
            or epochs != recipe["max_epochs"]
        ):
            raise LedgerError("G9_CYCLE_BINDING: exact source/ledger/batch/exposure required")

    def bind_engine(self, owner, factory, teach_schedule, projections, *, cycle=None):
        """Freeze supplied schedules/routing in the existing typed factory carrier.

        No schedule or projection is generated here. Births remain G3/G7 rows
        written before installation, through the existing lake writer.
        """
        from snn_data_lake.engine.brain import Brain
        from snn_data_lake.instruments import taught_problems as tp
        from snn_data_lake.instruments.joint_mask_search import RetentionSets
        from snn_data_lake.readout.posed_problem import (
            _need,
            pack_engine_binding,
            posed_source,
            unpack_engine_binding,
        )

        _need(
            type(owner) is Brain and owner.brain._ixn is self.learner,
            "actual canonical Brain owner required",
        )
        _need(not hasattr(self, "_engine_binding"), "engine adapter binds once")
        _need(
            factory["problem_id"] == self.problem.id and factory["seed"] == self.seed,
            "factory/problem/seed identity",
        )
        _need(
            factory["population_origin"] == "born",
            "growth adapter requires admitted population origin",
        )
        _need(
            type(teach_schedule) is tuple
            and teach_schedule
            and all(type(epoch) is tuple and epoch for epoch in teach_schedule),
            "pre-frozen teaching schedule required",
        )
        _need(
            all(
                type(episode) is tuple and episode and all(type(t) is tp.Turn for t in episode)
                for epoch in teach_schedule
                for episode in epoch
            ),
            "registered teaching turns required",
        )
        _need(factory["arm"] != "RSTDP_BRAIN_SHUF", "shuffle schedule remains unbound")
        _need(
            type(projections) is dict
            and all(type(k) is str and k.isascii() and k.isdecimal() for k in projections),
            "explicit future-unit projections required",
        )
        for entry in projections.values():
            _need(
                type(entry) is dict
                and set(entry) == {"feed", "return"}
                and type(entry["feed"]) is list
                and len(entry["feed"]) == 256
                and type(entry["return"]) is list,
                "G9_POOL256_WIDTH: explicit projection column required",
            )
            _need(
                all(
                    type(row) is dict and set(row) == {"source", "sign", "weight", "delay"}
                    for row in entry["return"]
                ),
                "explicit return descriptors required",
            )
        config = {
            "schedule": [
                [[[t.text, t.role, t.label] for t in episode] for episode in epoch]
                for epoch in teach_schedule
            ],
            "projections": copy.deepcopy(projections),
        }
        if cycle is not None:
            config["cycle"] = _cycle_recipe(cycle)
        config_source = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        bound_factory = copy.deepcopy(factory)
        if "engine_binding" in bound_factory:
            _need(
                unpack_engine_binding(bound_factory["engine_binding"]) == config,
                "G9_ENGINE_CONFIG_DRIFT: frozen binding differs",
            )
        else:
            bound_factory["engine_binding"] = (
                pack_engine_binding(config) if cycle is not None else config
            )
        posed_source({"op": "bind", "problem_id": self.problem.id, "factory": bound_factory})
        with owner._r2_lock:
            _need(owner._r2_fault is None, "common all-learning latch is closed")
            # A14 freezes each OLD task's validation/future order before search.
            retention = {
                pid: RetentionSets.for_problem(tp.PROBLEMS[pid], actor.factory["seed"])
                for pid, actor in owner._g9_actors.items()
                if pid != self.problem.id
            }
            if self.problem.id in owner._g9_actors:
                _need(
                    owner._g9_actors[self.problem.id].factory == bound_factory,
                    "G9_ENGINE_CONFIG_DRIFT: restored factory differs",
                )
            else:
                owner.bind_part_d_reader(bound_factory)
            self._engine_binding = {
                "owner": owner,
                "schedule": teach_schedule,
                "config": config_source,
                "sha256": hashlib.sha256(config_source.encode()).hexdigest(),
                "carrier": copy.deepcopy(bound_factory["engine_binding"]),
                "cycle": copy.deepcopy(cycle),
                "retention": retention,
            }

    def _require_engine(self):
        if not hasattr(self, "_engine_binding"):
            CanonicalGrowthAdapter.require_ready(self)
        binding = self._engine_binding
        owner = binding["owner"]
        if owner._r2_fault is not None or owner.brain._ixn is not self.learner:
            raise LedgerError("G9 common owner/latch binding is closed")
        if hashlib.sha256(binding["config"].encode()).hexdigest() != binding["sha256"]:
            raise LedgerError("G9_ENGINE_CONFIG_DRIFT: binding source changed")
        actor = owner._g9_actors[self.problem.id]
        if actor.factory.get("engine_binding") != binding["carrier"]:
            raise LedgerError("G9_ENGINE_CONFIG_DRIFT: actual actor binding changed")
        if actor.factory["engine_binding"].get("cycle") != binding["cycle"]:
            raise LedgerError("G9_ENGINE_CONFIG_DRIFT: active cycle recipe changed")
        return binding, actor

    def _retention_read(self, pid, episode):
        """Learning OFF on actual old circuits; full caller snapshot is restored by A14.

        This transient read never appends a durable event that could survive the
        snapshot rollback. Labels remain solely in this judging consumer.
        """
        binding, _ = self._require_engine()
        owner = binding["owner"]
        actor = owner._g9_actors[pid]
        cycle = binding["cycle"] is not None
        turns = []
        result = []
        for turn in episode:
            turns.append([turn.text, turn.role, None])
            value = {
                "op": "engine_read" if cycle else "read",
                "problem_id": pid,
                "turns": list(turns),
                "shifted_label": None,
                "neuron_mask": [True] * len(actor.reader.learner.neurons),
                "return_mask": [True] * (0 if actor.port is None else len(actor.port.rows)),
                "return_neutered": False,
            }
            event_id = f"g9.retention.off:{actor.reads}"
            answer = (
                owner._g9_engine_effect(self.learner, owner._g9_actors, value, event_id)
                if cycle
                else actor._effect(value, event_id=event_id)
            )
            if turn.role == "probe":
                result.append((answer, turn.label))
        return tuple(result)

    def read_retention(self, problem_id, *, meter=None):
        """Actual old skills, with complete rollback and durable final-read reservation."""
        binding, _ = self._require_engine()
        cycle = binding["cycle"] is not None
        if meter is not None:
            from snn_data_lake.instruments.joint_mask_search import Meter, SelectionLeak

            if type(meter) is not Meter or not meter.frozen:
                raise SelectionLeak("retention_final read before the winner was frozen")
            # The consumed-once flag lives only in this adapter's RetentionSets, which a
            # rebind or restart recreates; no durable record of consumption (keyed to the
            # frozen experiment/winner and old task) exists in the carrier/journal yet.
            # Until it does, the untouched final judgment is not callable from a Brain.
            if not cycle:
                raise LedgerError(
                    "G9_RETENTION_FINAL_UNRECORDED: durable consumption record not built; "
                    "the one-shot final read is refused on the Brain-bound adapter"
                )
        with binding["owner"]._r2_lock:
            binding, _ = self._require_engine()
            actor = binding["owner"]._g9_actors[problem_id]
            if problem_id == self.problem.id and cycle:
                sets = binding.setdefault("current_sets", self._current_sets())
            else:
                sets = binding["retention"][problem_id]
            owner = binding["owner"]

            def closing(fn):
                # An unjournaled read relies on COMPLETE restoration. A snapshot/restore
                # failure leaves the old actor in an unknown state, so it closes the SAME
                # R2 all-learning latch (checkpoints and effects refuse) and propagates.
                def run(*a):
                    try:
                        return fn(*a)
                    except BaseException as exc:
                        owner._r2_close("g9_retention_snapshot_restore", exc)
                        raise

                return run

            args = {
                "snapshot": closing(owner._g9_snapshot if cycle else actor.snapshot),
                "restore": closing(owner._g9_restore if cycle else actor.rollback),
                "read_off": lambda episode: self._retention_read(problem_id, episode),
            }
            if meter is not None:
                owner.claim_retention(self.problem.id, problem_id, meter=meter)
                return sets.read_final(meter, **args)
            return sets.read_select(**args)

    def _current_sets(self):
        from snn_data_lake.instruments.joint_mask_search import RetentionSets

        return RetentionSets.for_problem(self.problem, self.seed)

    @staticmethod
    def _correct(reads):
        return all(answer == target for episode in reads for answer, target in episode)

    @property
    def n(self):
        return len(self.segment)

    def _full_mask(self, mask):
        if type(mask) is not tuple or len(mask) != self.n or any(b is not True for b in mask):
            raise LedgerError(
                "G9_CYCLE_SELECTION: only actual all-active/no-prune control is bound"
            )

    def cost(self, mask):
        self._full_mask(mask)
        _, actor = self._require_engine()
        return (
            self.n
            + sum(len(u.w_in) + len(u.out) for u in actor.units)
            + 256 * self.n
            + len(actor.factory["attachments"])
        )

    def evaluate(self, mask):
        """One learned-state feasibility read for the explicit no-prune integration control."""
        self._full_mask(mask)
        binding, _ = self._require_engine()
        current = self.read_retention(self.problem.id)
        old = [self.read_retention(pid) for pid in binding["retention"]]
        feasible = self._correct(current) and all(self._correct(rows) for rows in old)
        score = sum(a == b for episode in current for a, b in episode)
        return feasible, score

    def select_cycle(self):
        from snn_data_lake.instruments.joint_mask_search import Meter

        self.require_ready(ledger=self._cycle_ledger)
        if self._cycle_ledger.ledger(self.problem.id).state != "SOLVED":
            raise LedgerError("G9_CYCLE_INFEASIBLE: stable acquisition has not been recorded")
        binding, actor = self._require_engine()
        meter = Meter(self, budget=1)
        if actor.selection is not None:
            meter.freeze(tuple(actor.selection["winner"]))
            self._cycle_meter = meter
            return meter, copy.deepcopy(actor.selection)
        full = tuple([True] * self.n)
        feasible, _ = meter.evaluate(full)
        if not feasible:
            raise LedgerError("G9_CYCLE_INFEASIBLE: current solve/old-skill retention failed")
        meter.freeze(full)
        receipt = binding["owner"].freeze_selection(self.problem.id, meter=meter)
        self._cycle_meter = meter
        return meter, receipt

    def judge(self, mask):
        self._full_mask(mask)
        meter = getattr(self, "_cycle_meter", None)
        if meter is None or meter.p is not self or meter.winner != mask:
            raise LedgerError("G9_CYCLE_SELECTION: actual frozen deciding meter required")
        binding, _ = self._require_engine()
        current = self.read_retention(self.problem.id, meter=meter)
        old = {pid: self.read_retention(pid, meter=meter) for pid in binding["retention"]}
        self.final_reads = {self.problem.id: current, **old}
        return self._correct(current), all(self._correct(rows) for rows in old.values())

    def diagnose(self):
        from snn_data_lake.growth import capacity_certificate

        actual = hasattr(self, "_engine_binding") and "cycle" in json.loads(
            self._engine_binding["config"]
        )
        if actual:
            binding, actor = self._require_engine()
            binding["owner"]._g9_publish_context(actor)
        self.last_diagnosis = capacity_certificate.diagnose(
            self.learner,
            self._texts,
            self._labels,
            history_texts=None,
            units=self.segment if actual else (),
            **({"actual_context": True} if actual else {}),
        )
        return self.last_diagnosis.kind

    def evidence(self):
        import copy

        diagnosis = self.last_diagnosis
        if diagnosis is None:
            raise LedgerError("canonical diagnosis evidence is absent")
        witness = None
        if diagnosis.witness is not None:
            witness = {
                "path": diagnosis.witness.path,
                "lam": [
                    [item, competitor, value.numerator, value.denominator]
                    for (item, competitor), value in sorted(diagnosis.witness.lam.items())
                ],
            }
        separator = None
        if diagnosis.separator is not None:
            separator = {
                "weights": [list(row) for row in diagnosis.separator.weights],
                "reads": list(diagnosis.separator.reads),
            }
            if hasattr(diagnosis.separator, "grown_outputs"):
                separator["grown_outputs"] = copy.deepcopy(diagnosis.separator.grown_outputs)
        return {
            "kind": diagnosis.kind,
            "k": diagnosis.k,
            "k_history": diagnosis.k_history,
            "log": copy.deepcopy(diagnosis.log),
            "witness": witness,
            "separator": separator,
            "scope": "canonical-actual-context"
            if hasattr(self, "source")
            else "canonical-frozen-code-only",
            "missing": list(self.readiness()),
        }

    def readiness(self, *, ledger=None, sequence=False):
        """G3 2.6/2.8: first-problem budget is ready; SEQUENCE requires beta.

        Independent grown/retention/restore debts continue to refuse effects.
        An actual positive-spend baseline also requires beta, even when a caller
        omits the sequence-family flag.
        """
        from snn_data_lake.instruments import taught_problems as tp

        needs_beta = sequence or (ledger is not None and ledger.baseline() is not None)
        beta = tp.BETA if ledger is None or ledger.values.beta is None else ledger.values.beta
        missing = CanonicalGrowthAdapter._MISSING
        if hasattr(self, "_engine_binding"):
            self._require_engine()
            # Ordinary ports are bound; the automatic search/source and its
            # retention feasibility/mask consumer are the next deciding slice.
            missing = (
                "retention_select",
                "actual grown-source/admission binding",
                "G6 deciding mask/progress carrier",
                "retention final read (G9_RETENTION_FINAL_UNRECORDED) until durable consumption exists",
            )
            if (
                "cycle" in json.loads(self._engine_binding["config"])
                and type(getattr(self, "source", None)) is CanonicalGrowthSource
                and getattr(self, "_cycle_ledger", None) is self.source.ledger
            ):
                self.check_loop_binding(
                    self._cycle_ledger,
                    self.source,
                    json.loads(self._engine_binding["config"])["cycle"]["batch_size"],
                    json.loads(self._engine_binding["config"])["cycle"]["max_epochs"],
                )
                missing = ()
                # Continuing/judging an already granted budget is not opening
                # another positive-spend sequence problem.
                if (
                    not sequence
                    and ledger is self._cycle_ledger
                    and self.problem.id in ledger.problem_ids()
                ):
                    needs_beta = False
        if needs_beta and beta is None:
            missing += ("source-owned BETA baseline",)
        return missing

    def require_ready(self, *, ledger=None, sequence=False):
        missing = CanonicalGrowthAdapter.readiness(self, ledger=ledger, sequence=sequence)
        if missing:
            raise LedgerError("G9_A_UNREADY: " + ", ".join(missing))

    @property
    def segment(self):
        _, actor = self._require_engine()
        return tuple(actor.units)

    def admit(self, units, labels, ticket):
        """G3/G7: actual ticket and durable births precede actor installation."""
        from snn_data_lake.readout.posed_problem import _need, unit_rows

        binding, actor = self._require_engine()
        owner = binding["owner"]
        with owner._r2_lock:
            binding, actor = self._require_engine()
            config = json.loads(binding["config"])
            new = tuple(sorted(units, key=lambda u: u.unit_id))
            _need(
                new
                and len(new) == len(labels)
                and {r["unit_id"] for r in labels} == {u.unit_id for u in new},
                "exact actual G7 batch identities required",
            )
            complete = tuple((*actor.units, *new))
            born = owner._g9_birth_records()
            _need(
                all(u.unit_id in born for u in complete),
                "G9_DURABLE_BIRTH_REQUIRED: birth before actor installation",
            )
            births = [born[u.unit_id] for u in complete]
            _need(
                all(r["problem_id"] == self.problem.id for r in births),
                "different posing birth owner",
            )
            feed = copy.deepcopy(actor.feed)
            attachments = copy.deepcopy(actor.factory["attachments"])
            for unit in new:
                entry = config["projections"].get(str(unit.unit_id))
                _need(entry is not None, "G9_PROJECTION_REQUIRED: source-owned column absent")
                for row, weight in zip(feed, entry["feed"], strict=True):
                    row.append(weight)
                attachments += [
                    {**row, "target": unit.unit_id, "receipt": binding["sha256"]}
                    for row in entry["return"]
                ]
            owner._g9_apply_source(
                {
                    "op": "admit",
                    "problem_id": self.problem.id,
                    "units": unit_rows(complete),
                    "birth_rows": births,
                    "w_feed": feed,
                    "attachments": attachments,
                },
                ticket=ticket,
            )

    def attempt(self, epochs):
        """Resume the durable teaching cursor; never redraw an epoch after restart."""
        from snn_data_lake.instruments import taught_problems as tp

        binding, actor = self._require_engine()
        if "cycle" in json.loads(binding["config"]):
            return self._cycle_attempt(epochs)
        with binding["owner"]._r2_lock:
            binding, actor = self._require_engine()
            if type(epochs) is not int or epochs <= 0:
                raise LedgerError("attempt needs a positive exact epoch count")
            schedule = binding["schedule"]
            completed, offset = 0, actor.teaches
            while completed < len(schedule) and offset >= len(schedule[completed]):
                offset -= len(schedule[completed])
                completed += 1
            if completed + epochs > len(schedule):
                raise LedgerError("attempt would redraw or exceed its pre-frozen teaching schedule")
            for epoch in schedule[completed : completed + epochs]:
                for episode in epoch[offset:]:
                    binding["owner"].learn_part_d_episode(
                        self.problem.id,
                        episode,
                        neuron_mask=[True] * len(actor.reader.learner.neurons),
                        return_mask=[True] * (0 if actor.port is None else len(actor.port.rows)),
                        return_neutered=False,
                    )
                offset = 0
            reads = tp.read_split(
                self.problem,
                self.seed,
                "validation",
                lambda episode: binding["owner"].read_part_d_episode(
                    self.problem.id,
                    episode,
                    neuron_mask=[True] * len(actor.reader.learner.neurons),
                    return_mask=[True] * (0 if actor.port is None else len(actor.port.rows)),
                    return_neutered=False,
                ),
            )
            return tp.solved(self.problem, self.seed, "validation", reads)

    def _cycle_attempt(self, epochs):
        """E=400 maximum, stable-10 at epoch ends, with durable actual exposure."""
        from snn_data_lake.instruments import taught_problems as tp

        binding, actor = self._require_engine()
        recipe = json.loads(binding["config"])["cycle"]
        if type(epochs) is not int or epochs != recipe["max_epochs"]:
            raise LedgerError("G9_CYCLE_EXPOSURE: registered maximum epoch count required")
        ledger, owner = self._cycle_ledger, binding["owner"]
        with owner._r2_lock:
            if actor.selection is not None:
                raise LedgerError("G9_SELECTION_FROZEN: teaching after selection")
            events = [r for r in ledger.events() if r.get("problem_id") == self.problem.id]
            starts = [r for r in events if r.get("kind") == "engine_attempt_start"]
            finishes = {r["attempt_id"]: r for r in events if r.get("kind") == "engine_attempt_end"}
            recorded = {
                r.get("exposure", {}).get("attempt_id")
                for r in ledger._problem_rows(self.problem.id)
                if r.get("op") == "attempt"
            }
            pending = starts[-1] if starts and starts[-1]["attempt_id"] not in recorded else None
            if pending is None:
                pending = {
                    "kind": "engine_attempt_start",
                    "problem_id": self.problem.id,
                    "attempt_id": len(starts),
                    "teaches": actor.teaches,
                    "max_epochs": epochs,
                    "shared_deposits_before": self.learner.stack._global_t,
                }
                ledger.record_event(pending)
            attempt_id = pending["attempt_id"]
            schedule = binding["schedule"]
            size = len(schedule[0])
            if any(len(epoch) != size for epoch in schedule):
                raise LedgerError("G9_CYCLE_SCHEDULE: constant registered epoch width required")
            if pending["teaches"] % size:
                raise LedgerError("G9_CYCLE_CURSOR: new attempt must start at an epoch boundary")
            start_epoch = pending["teaches"] // size
            history = [
                r
                for r in events
                if r.get("kind") == "engine_epoch" and r["attempt_id"] == attempt_id
            ]
            stable = 0
            expected_targets = [
                ep[-1].label for ep in self.problem.episodes(self.seed, "validation")
            ]
            for j, row in enumerate(history, 1):
                flat = row["reads"]
                if (
                    row["teaches"] != pending["teaches"] + j * size
                    or row["epoch"] != start_epoch + j
                    or row["teaches"] > actor.teaches
                    or [pair[1] for pair in flat] != expected_targets
                    or row["correct"] != sum(a == b for a, b in flat)
                    or row["presentations"] != len(flat)
                ):
                    raise LedgerError("G9_CYCLE_HISTORY: missing or inconsistent actual epoch")
                solved = tp.solved(self.problem, self.seed, "validation", flat)
                stable = stable + 1 if solved else 0
                if row["stable_run"] != stable:
                    raise LedgerError(
                        "G9_CYCLE_HISTORY: stability does not match the recorded reads"
                    )
            if len(history) > epochs:
                raise LedgerError("G9_CYCLE_HISTORY: attempt exceeds its registered exposure")
            if attempt_id in finishes:
                self.last_attempt = copy.deepcopy(finishes[attempt_id]["exposure"])
                if (
                    self.last_attempt["teaches_after"] != actor.teaches
                    or self.last_attempt["epochs"] != len(history)
                    or self.last_attempt["stable_run"] != stable
                    or bool(finishes[attempt_id]["solved"]) != (stable >= recipe["stable_epochs"])
                ):
                    raise LedgerError(
                        "G9_CYCLE_HISTORY: completed attempt differs from the actual cursor"
                    )
                return bool(finishes[attempt_id]["solved"])
            while True:
                completed, offset = divmod(actor.teaches, size)
                # A crash after the last teach but before its validation must
                # finish that same epoch, before any new teaching is allowed.
                if (
                    completed > start_epoch
                    and offset == 0
                    and (not history or history[-1]["teaches"] != actor.teaches)
                ):
                    reads = self.read_retention(self.problem.id)
                    flat = [pair for episode in reads for pair in episode]
                    solved = tp.solved(self.problem, self.seed, "validation", flat)
                    stable = stable + 1 if solved else 0
                    row = {
                        "kind": "engine_epoch",
                        "problem_id": self.problem.id,
                        "attempt_id": attempt_id,
                        "epoch": completed,
                        "teaches": actor.teaches,
                        "correct": sum(a == b for a, b in flat),
                        "presentations": len(flat),
                        "stable_run": stable,
                        "shared_deposits": self.learner.stack._global_t,
                        "reads": [list(pair) for pair in flat],
                    }
                    ledger.record_event(row)
                    history.append(row)
                if stable >= recipe["stable_epochs"] or completed - start_epoch >= epochs:
                    break
                if completed >= len(schedule):
                    raise LedgerError(
                        "attempt would redraw or exceed its pre-frozen teaching schedule"
                    )
                for episode in schedule[completed][offset:]:
                    owner.learn_engine_episode(
                        self.problem.id,
                        episode,
                        neuron_mask=[True] * len(actor.reader.learner.neurons),
                        return_mask=[True] * (0 if actor.port is None else len(actor.port.rows)),
                        return_neutered=False,
                    )
            self.last_attempt = {
                "attempt_id": attempt_id,
                "epochs": actor.teaches // size - start_epoch,
                "teach_episodes": actor.teaches - pending["teaches"],
                "teaches_before": pending["teaches"],
                "teaches_after": actor.teaches,
                "validation_presentations": sum(r["presentations"] for r in history),
                "stable_run": stable,
                "shared_plasticity_deposits": (
                    self.learner.stack._global_t - pending["shared_deposits_before"]
                    if "shared_deposits_before" in pending
                    else None
                ),
            }
            success = stable >= recipe["stable_epochs"]
            ledger.record_event(
                {
                    "kind": "engine_attempt_end",
                    "problem_id": self.problem.id,
                    "attempt_id": attempt_id,
                    "solved": success,
                    "exposure": self.last_attempt,
                }
            )
            return success


def main():
    """Self-check on a scripted learner in a throwaway lake: the pause (C8), the attempt cap (C18), the ledger
    rebuild and its dropped-row red (C13), and the live-lake refusal (2.12a)."""
    import tempfile

    from snn_data_lake.growth.grown_code import GrownUnit, admit_rows

    class Scripted:
        def __init__(self, kind, solve_after=None):
            self.kind, self.solve_after, self.n, self.segment = kind, solve_after, 0, ()

        def diagnose(self):
            return self.kind

        def admit(self, units, labels, ticket):
            self.n += len(units)
            self.segment = admit_rows(self.segment, units, labels, ticket)

        def attempt(self, epochs):
            return self.solve_after is not None and self.n >= self.solve_after

    ids = iter(range(10**6))

    def source(n):
        return [
            GrownUnit(next(ids), (0.0,), 1.0, 12, -1 if i == n - 1 and n > 1 else 1, (0.0,))
            for i in range(n)
        ]

    values = BudgetValues(first=52, beta=1.5)
    admission = make_admission(
        site=(0.5 / 14, 1.5 / 4), centre_um=(0.0, 0.0), density_per_mm2=2.7e4, seed=9000
    )  # fixture values

    def Loop(*a, **k):
        return GrowthLoop(*a, admission=admission, resolve_problem=lambda p: True, **k)

    with tempfile.TemporaryDirectory() as d:
        lake = Path(d) / "lake.bin"
        paused = Loop(Scripted("CAPACITY"), "u", Ledger(lake, values), source, 9).run()
        assert paused.state == "UNSOLVED" and paused.open and paused.spend_units == 45, paused
        capped = Loop(Scripted("SEPARABLE"), "s", Ledger(lake, values), source, 9).run()
        assert (
            capped.state == "LEARNING_RULE_DEFICIT"
            and capped.attempts == A_MAX
            and capped.spend_units == 0
        )
        solved = Loop(Scripted("CAPACITY", 18), "x", Ledger(lake, values), source, 9).run()
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
    # a DECOY live home, never the owner's real one (live-lake freeze): the guard reads _default_live_home
    from snn_data_lake.substrate import home

    with tempfile.TemporaryDirectory() as decoy:
        real, home._default_live_home = home._default_live_home, lambda: decoy
        try:
            Ledger(Path(decoy) / "brain_lake.bin", values)
            raise AssertionError("red: the live lake must be refused")
        except LiveHomeRefused:
            pass
        finally:
            home._default_live_home = real
    print(
        "growth_loop: pause at 45/52 (UNSOLVED, OPEN), attempt cap -> LEARNING_RULE_DEFICIT, ledger rebuilt "
        "equal, seq-gap red fires, live lake refused"
    )


if __name__ == "__main__":
    main()
