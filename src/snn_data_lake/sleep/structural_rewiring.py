"""
Structural rewiring on the two-clock design -- specs/022 T048 (FR-019's structural
palette element): DEEP-R budget + MSP counters + the turnover-rate instrument, run on
the SLOW clock (the sleep tick), never the fast per-interaction clock.

THE WIRE WAS MEASURED AND REFUSED (2026-08-26, the module's own review panel -- three
lenses, probes attached; runtime evidence outranks static reading). The intended live
substrate was the IXN intent neurons' 52-wide rows, wired at the sleep tick. Measured
there: (a) the substrate is ALL-EXCITATORY at birth (0 of 468 negative) and produces
ZERO sign crossings on either live path -- 206 autonomous structural cycles and 1500
teach() interactions moved 468/468 weights without one identity violation; (b) the
caller's own Benna-Fusi relax rebinds every row from the consolidation cascade's deep
store IMMEDIATELY above the pass, restoring the old sign before rewire() could read a
violation (warm cascade = any brain that has learned; the wire test was green only on
a cold one); (c) an inhibitory regen is dragged back across zero by the cascade within
2 fast-clock steps, so the MSP inhibitory half was dead on arrival there; (d) a regen
at the row-minimum lands at 80-87%% of a typical weight on that tight distribution --
not DEEP-R's near-zero reactivation. Wiring against those measurements would be
force-wiring (P6's named defect).

REFUSED A SECOND TIME AT T054-ii (2026-08-31), on the tissue that was supposed to be its
home -- and the new reason is worth more than the re-dating. T054's cerebellar
microcircuit DOES supply the sign-structured substrate T048 lacked (PF(+) rows against an
MLI(-) slot, identities read from the weights), and the trigger is STILL unreachable,
because NOTHING WRITES pf_w: measured before any wire was written, 2200 driven steps
through the pn relay give PF sign crossings = 0 and PF weights changed = 0 of 512. The
tissue carries no PF plasticity -- FR-019's cerebellar rule (PF-Purkinje depression under
the olive's teaching signal) is a later task's build. So the condition this pass actually
needs was never "sign-structured rows": it is A WRITER THAT MOVES THEM ACROSS ZERO, which
neither substrate has had. RE-DATED to the first mechanism that writes pf_w, with a
checkable criterion instead of a promise -- a nonzero crossing count on that same probe.
The composition
conflict with _BennaFusiConsolidation (two writers, the deep store re-imposing sign)
is EXACTLY T051's red case (two composed rules whose naive sum is wrong vs either
alone) -- MODELED at T051b-ii at the COMPONENT grain (rule_composer's harness:
this module's prune x the bare cascade's diffuse step, composed; the wrapper,
relax(), and the live wire stay undriven -- the wire is T054's, not resolved here).

WHAT IT DOES. Over a population's weight rows (the harness substrate today; T054's
tissue when the wire lands):

  DEEP-R BUDGET (adopted mechanism, Bellec et al. rewiring under a fixed connection
  count): every live connection carries a SIGN IDENTITY fixed at first sight; a weight
  the fast clock drives ACROSS ZERO has violated its identity and is PRUNED (zeroed,
  slot vacant). The population's TOTAL live count is the budget, derived from the
  substrate at first sight (P1: a baseline, never a picked number). Every cycle the
  deficit (budget - live) is REGENERATED into vacant slots -- the prune-and-regenerate
  pairing IS the conservation invariant (every vacancy answered by one regen at the
  conserved budget), not a per-slot correspondence (the first draft's index-join over
  a shuffled vacancy list dressed a count as an attribution; the panel called it).
  Regens seed at the target row's own minimum live magnitude (self-derived; the
  population minimum when the row has no live weight; MEASURED on the tight IXN
  distribution at 80-87%% of a typical weight -- NOT DEEP-R's near-zero reactivation,
  a named deviation from the adopted mechanism).

  MSP COUNTERS (adopted vocabulary, Butz & van Ooyen's Model of Structural Plasticity;
  homeostasis toward an activity setpoint): on THIS substrate the counters steer the
  regenerated connection's SIGN -- a row BELOW the population's own median activity
  regenerates excitatory (raising future drive toward setpoint), a row above it
  regenerates inhibitory. Activity is READ from retained state that already exists
  (the stack's _BCMMetaGate sliding theta; the public read route lands WITH the T054
  wire -- re-deriving an activity EMA here would be the P2 failure). TARGET steering
  -- choosing WHICH row receives a regen -- was built first and MEASURED STRUCTURALLY
  VACUOUS on a dense-at-birth substrate (this module's own MSP arm read steered ==
  uniform, 0.50 vs 0.50: with every slot live at first sight, each cycle's vacancies
  exactly equal its deficit, so the regen set is FORCED and ordering chooses nothing).
  Deleted, not weakened; re-opens only with a substrate that carries capacity slack.

  TURNOVER INSTRUMENT (the structural-health row): per cycle, turnover =
  replacements / live-budget, with own-prunes, external vacancies and re-adoptions
  reported SEPARATELY (a turnover another mechanism caused must be attributable). A
  healthy substrate under a stable regime turns over ~0; one whose fast clock keeps
  violating identities turns over persistently -- the instrument reports, it never
  decides (the deciding gate is T047's, at a different grain). No cockpit renderer
  exists yet (the T047 precedent: state()-field surfacing only) -- the state()
  surfacing LANDED WITH THE T058 WIRE (2026-09-01): Brain.state()'s
  "sleep_consolidation.structural" carries this report() per sleep tick, with the
  rewire ValueError refusals counted beside it.

TWO CLOCKS, stated: the FAST clock (R-STDP deposit + stack stabilize, per interaction)
moves magnitudes; the SLOW clock (this module, per structural cycle) moves STRUCTURE.
This module never runs on a tick path and owns no learning rate.

READ/WRITE SETS (FR-018): reads the rows' weights + the population's activity vector
(the sign steer uses the population median; the seed magnitude falls back to the
population minimum); writes weight rows ONLY (zeroing pruned slots, seeding
regenerated ones) -- note the measured caveat: on a substrate governed by a
consolidation cascade, a weight row is NOT the state of record and a structural write
must also reach the cascade (the conflict above, modeled at T051b-ii). THE FIX LANDED
AT T058 (2026-09-01), scope exactly as T054-ii's measurement narrowed it: on the IXN
stack -- the one substrate where a cascade wraps the rows --
`PlasticityRuleStack.adopt_structural(i, weights, changed)` re-anchors each CHANGED
slot's cascade at equilibrium (a pruned/regenerated synapse is a NEW synapse; its
consolidation history dies with the slot), called by Brain._sleep_tick's slow-clock
block after every rewire, so the next relax() settles TOWARD the structural edit
instead of silently erasing it. Inapplicable on the cerebellar tissue, whose pf_w IS
its own state of record. THE WIRE ITSELF also landed at T058: Brain._sleep_tick runs
this module per sleep tick over the IXN rows, MSP-steered by the stack's own meta
theta (`PlasticityRuleStack.meta_theta()`, the public read this paragraph long
scheduled). Honest live-regime expectation, unchanged from T048's measurement: a
healthy all-positive-birth substrate produces sign crossings RARELY, so live turnover
~0 is the mechanism holding, not dormancy -- the wire's reachable red is the
constructed sign-flip in tests/test_sleep_capabilities.py (prune + budget conservation
+ cascade adoption each asserted), per the house rule that a gate landing green ships
the reconstruction of the defect it was built for. FR-018's neuter attribution did NOT
wait for this wire: it landed at T054-ii on the tissue's own elements
(faculties/microcircuit.py's per-element A/B).

WHAT THE ADOPTION OWES (P2/P8 discipline; every arm in main(), driven PAST the first
event per the T047 lesson -- a harness that breaks at the first firing certifies a
duty cycle, not a mechanism):
  A/A       -- stable signs, 50 cycles: weights BYTE-IDENTICAL, turnover exactly 0.0
               every cycle.
  RED       -- a constructed sign flip prunes THAT slot and regenerates exactly one,
               budget conserved EXACTLY over 200 cycles of sustained drift.
  MSP       -- under sustained flip pressure, constructed below-setpoint rows
               regenerate EXCITATORY and above-setpoint rows INHIBITORY (sign
               steering, the dense-substrate form); CONTROL (not the method): no
               activity -> the seeded coin. Target steering: measured vacuous
               here, deleted (the module docstring carries the measurement).
  INSTRUMENT -- k constructed flips read exactly k/budget; the no-flip control reads
               exactly 0.0 (a dead instrument and a quiet substrate differ by the
               constructed arm, not by assumption).
  REFUSALS  -- NaN weights, non-finite or mismatched activity: ValueError, raised
               BEFORE any mutation (the panel constructed an aborted pass that had
               already zeroed three slots); the wire must count these, never swallow
               them -- HONORED at T058: Brain._sleep_tick catches ValueError into
               _structural_refusals (surfaced in state()) before its generic guard.

  PYTHONPATH=src python -m snn_data_lake.sleep.structural_rewiring
"""

import math
import random as _random

# distinct-from: sleep/developmental_pruning.py -- an OFFLINE benchmark sparsifier
#   for a linear readout: prune-only, no regeneration, no conserved budget, no live
#   wire; this module is the LIVE substrate's slow-clock rewiring. Site read.
# distinct-from: growth/neurogenesis_engine.py -- grows/prunes UNITS (whole neurons) on
#   the refit clock; this rewires CONNECTIONS under a fixed unit set. Site read.
# distinct-from: growth/growth_stop.py -- T047 gates whether unit growth CONTINUES;
#   this module neither grows nor gates units. Site read.
# distinct-from: plasticity/homeostatic_scaling.py, plasticity/plasticity_rule_stack.py
#   -- magnitude stabilization on the FAST clock; this module reads the stack's theta
#   and moves STRUCTURE only. Sites read.
# distinct-from: engine/brain.py's attractor sub-eps zeroing -- magnitude hygiene on
#   the ATTRACTOR's weights (a different substrate), unbudgeted and unpaired; this
#   module's prune trigger is IDENTITY VIOLATION (sign flip) with a conserved budget.
#   Site read.

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "StructuralRewiring.rewire": (
        "LOCAL",
        "zeroes sign-flipped slots and seeds vacant ones from the row's own minimum "
        "live magnitude (population minimum as the empty-row fallback); reads the "
        "rows' weights + the POPULATION's activity vector (the sign steer compares a "
        "row against the population median -- the panel corrected the first claim's "
        "'reads only the row's own'); no error propagation, no weight transport, no "
        "learning rate -- structural replacement, not gradient descent",
    ),
}


class StructuralRewiring:
    """The slow-clock structural pass. Construct once, call rewire(rows, activity) per
    sleep cycle; rows are mutated IN PLACE (the caller's substrate is the substrate).

    SINGLE-WRITER by design: one population, one rewirer, on one clock."""

    def __init__(self, seed=0):
        self._rng = _random.Random(seed)
        self._signs = None  # per (row, col): +1/-1 identity, 0 = vacant; set at first sight
        self._budget = None  # TOTAL live connections at first sight (the DEEP-R budget)
        self._cycles = 0
        self._pairs_total = 0
        self._last = {"turnover": 0.0, "pruned": 0, "regenerated": 0, "budget": 0}
        self._regen_signs = {}  # row -> (n_pos, n_neg) regen counts (MSP attribution, bounded)

    STATE_FORMAT = "structural_rewiring.v1"

    def state_dict(self):
        """Existing structural topology and exact RNG position, with detached int-key entries."""
        import copy
        from snn_data_lake.readout.interaction_learning import _encode_state

        state = {
            "fmt": self.STATE_FORMAT, "rng": [self._rng.getstate()[0], list(self._rng.getstate()[1]), self._rng.getstate()[2]],
            "signs": copy.deepcopy(self._signs), "budget": self._budget,
            "cycles": self._cycles, "pairs_total": self._pairs_total,
            "last": dict(self._last),
            "regen_signs": [[r, list(v)] for r, v in sorted(self._regen_signs.items())],
        }
        _encode_state(state)  # canonical finite/type validation; no second codec
        return state

    def load_state_dict(self, state):
        """Validate all fields and duplicate int keys BEFORE touching the real carrier."""
        import copy
        from snn_data_lake.readout.interaction_learning import _encode_state

        fields = {"fmt", "rng", "signs", "budget", "cycles", "pairs_total", "last", "regen_signs"}
        if type(state) is not dict or set(state) != fields or state["fmt"] != self.STATE_FORMAT:
            raise ValueError("malformed structural checkpoint")
        _encode_state(state)
        raw_rng = state["rng"]
        if type(raw_rng) is not list or len(raw_rng) != 3 or type(raw_rng[1]) is not list:
            raise ValueError("malformed structural RNG carrier")
        if type(raw_rng[0]) is not int or any(type(n) is not int for n in raw_rng[1]):
            raise ValueError("invalid structural RNG integers")
        if raw_rng[2] is not None and type(raw_rng[2]) is not float:
            raise ValueError("invalid structural Gaussian cache")
        rng = (raw_rng[0], tuple(raw_rng[1]), raw_rng[2])
        probe = _random.Random(0)
        try:
            probe.setstate(rng)
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError("invalid structural RNG") from exc
        signs, budget = state["signs"], state["budget"]
        if signs is None:
            if budget is not None:
                raise ValueError("unborn structural budget must be absent")
        else:
            if type(signs) is not list or any(type(row) is not list for row in signs):
                raise ValueError("malformed structural topology")
            if any(type(s) is not int or s not in (-1, 0, 1) for row in signs for s in row):
                raise ValueError("invalid structural sign")
            if type(budget) is not int or not 0 <= budget <= sum(map(len, signs)):
                raise ValueError("invalid structural budget")
        for key in ("cycles", "pairs_total"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError("invalid structural counter")
        last = state["last"]
        base = {"turnover", "pruned", "regenerated", "budget"}
        if type(last) is not dict or set(last) not in (
            base, base | {"external_vacancies", "readopted"},
        ):
            raise ValueError("malformed structural report")
        if type(last["turnover"]) is not float or not math.isfinite(last["turnover"]):
            raise ValueError("invalid structural turnover")
        for key in set(last) - {"turnover"}:
            if type(last[key]) is not int or last[key] < 0:
                raise ValueError("invalid structural report count")
        if type(state["regen_signs"]) is not list:
            raise ValueError("malformed structural sign entries")
        regen = {}
        for entry in state["regen_signs"]:
            if type(entry) is not list or len(entry) != 2:
                raise ValueError("malformed structural sign entry")
            row, counts = entry
            if type(row) is not int or row < 0 or row in regen:
                raise ValueError("duplicate or invalid structural row")
            if signs is None or row >= len(signs):
                raise ValueError("structural regen row outside topology")
            if type(counts) is not list or len(counts) != 2:
                raise ValueError("malformed structural sign counts")
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError("invalid structural sign count")
            regen[row] = tuple(counts)
        if sum(sum(v) for v in regen.values()) != state["pairs_total"]:
            raise ValueError("structural lifetime sign counts disagree")
        if signs is None and (state["cycles"] or state["pairs_total"] or regen):
            raise ValueError("unborn structural state carries completed cycles")
        self._rng.setstate(rng)
        self._signs = copy.deepcopy(signs)
        self._budget = budget
        self._cycles = state["cycles"]
        self._pairs_total = state["pairs_total"]
        self._last = dict(last)
        self._regen_signs = regen

    # ---- the slow-clock pass -----------------------------------------------------------------
    def rewire(self, rows, activity=None):
        """One structural cycle over `rows` (list of mutable weight lists). `activity`:
        one scalar per row (the stack's theta), or None (no MSP steering: regen targets
        draw uniformly -- honest degradation, not a fake counter). Returns the report
        dict {turnover, pruned, external_vacancies, readopted, regenerated, budget} --
        COUNTS ONLY, no per-site pairs (this line long promised a "pairs" key the
        report never carried; caught by T051b-ii's arm consuming the report). Refuses
        malformed input (non-finite weights, length-mismatched activity) with
        ValueError."""
        acts_f = None
        if activity is not None:
            if len(activity) != len(rows):
                raise ValueError(f"activity length {len(activity)} != rows {len(rows)}")
            # validated (type + finiteness) BEFORE any mutation -- the panel constructed
            # a str activity raising AFTER three slots were already zeroed (an aborted
            # pass that mutated), and NaN activity silently steering every regen
            # inhibitory (nan < med is always False).
            acts_f = [float(a) for a in activity]
            for r, a in enumerate(acts_f):
                if not math.isfinite(a):
                    raise ValueError(f"non-finite activity at row {r}: {activity[r]!r}")
        for r, row in enumerate(rows):
            for c, w in enumerate(row):
                if not math.isfinite(float(w)):
                    raise ValueError(f"non-finite weight at row {r} col {c}: {w!r}")
        if self._signs is None:  # FIRST SIGHT: the substrate defines its own budget
            self._signs = [[_sgn(w) for w in row] for row in rows]
            self._budget = sum(1 for row in self._signs for s in row if s != 0)
        # 1) PRUNE: identity violations (the DEEP-R trigger) -- a live slot whose weight
        #    crossed zero against its birth sign. External zeroings (another mechanism
        #    emptied a live slot) vacate WITHOUT counting as this module's prune; the
        #    deficit regen below still restores the budget, and the report names both.
        pruned, external, adopted = [], 0, 0
        for r, row in enumerate(rows):
            for c, w in enumerate(row):
                s = self._signs[r][c]
                if s == 0:
                    if w != 0.0:
                        # external ZERO->LIVE: another mechanism repopulated a vacant
                        # slot. RE-ADOPT at the current sign -- the slot IS live, and
                        # its birth identity is whatever the external writer set (the
                        # panel measured the unguarded form diverging live-substrate
                        # vs live-signs monotonically on any non-dense substrate).
                        self._signs[r][c] = _sgn(w)
                        adopted += 1
                    continue
                if w == 0.0:
                    self._signs[r][c] = 0
                    external += 1
                elif _sgn(w) != s:
                    row[c] = 0.0
                    self._signs[r][c] = 0
                    pruned.append((r, c))
        # 2) REGENERATE the deficit, MSP-steered, budget-conserving.
        live = sum(1 for row in self._signs for s in row if s != 0)
        deficit = max(0, (self._budget or 0) - live)
        vacant = [(r, c) for r, row in enumerate(self._signs) for c, s in enumerate(row) if s == 0]
        regen = []
        if deficit and vacant:
            self._rng.shuffle(vacant)  # seeded order; on a dense substrate the SET is forced anyway
            # MSP sign steering needs a DISCRIMINATING activity signal: a zero-spread
            # population (every value equal -- e.g. the stack's uniform resting theta)
            # carries no rank information, and the first draft's strict < median then
            # made EVERY regen inhibitory (measured 0/200 excitatory on the live
            # resting distribution). Zero spread -> the seeded coin, stated.
            med = None
            if acts_f is not None and max(acts_f) > min(acts_f):
                med = _median(acts_f)
            for k in range(min(deficit, len(vacant))):
                r, c = vacant[k]
                mag = _seed_magnitude(rows, r)
                if med is None:
                    sign = 1 if self._rng.random() < 0.5 else -1
                else:
                    sign = 1 if acts_f[r] < med else -1
                rows[r][c] = sign * mag
                self._signs[r][c] = sign
                regen.append((r, c))
                pos, neg = self._regen_signs.get(r, (0, 0))
                self._regen_signs[r] = (pos + (sign > 0), neg + (sign < 0))
        self._cycles += 1
        self._pairs_total += len(regen)
        turnover = (len(regen) / self._budget) if self._budget else 0.0
        # the "pairing" is the CONSERVATION INVARIANT -- every vacancy this cycle is
        # answered by exactly one regen at the conserved budget -- not a per-slot
        # correspondence (the first draft's index-join over a shuffled vacancy list
        # dressed a count as an attribution; the panel called it).
        self._last = {
            "turnover": turnover,
            "pruned": len(pruned),
            "external_vacancies": external,
            "readopted": adopted,
            "regenerated": len(regen),
            "budget": int(self._budget or 0),
        }
        return dict(self._last)

    # ---- the instrument ----------------------------------------------------------------------
    def report(self):
        """The structural-health row: the LAST cycle's turnover + counters (INCLUDING
        external_vacancies/readopted -- the first draft dropped them here, so a surface
        reading turnover>0 with zero own prunes could not attribute it) + lifetime
        totals. Reporting only -- nothing here decides anything (T047's gate decides,
        at the unit grain). Single capture of _last: the free-threading runtime showed
        61%% torn multi-field reads when a poller raced rewire()'s wholesale replace."""
        last = self._last
        return {
            "turnover": last["turnover"],
            "pruned": last["pruned"],
            "external_vacancies": last.get("external_vacancies", 0),
            "readopted": last.get("readopted", 0),
            "regenerated": last["regenerated"],
            "budget": last["budget"],
            "cycles": self._cycles,
            "pairs_total": self._pairs_total,
        }

    # distinct-from: percept_surprise.py -- a novelty signal for its own self-check. Site read.
    # distinct-from: neurogenesis_engine.py -- per-UNIT placement (depth, layer, Dale sign)
    #   maintained by the grower; this counts regenerated CONNECTION sign draws. Site read.
    # distinct-from: self_mod_harness.py, selfcheck_cache.py -- a stub repo and a name
    #   walker; the matcher fired on generic list/per-row words. Sites read.
    def regen_signs(self):
        """Per-row (n_excitatory, n_inhibitory) regen counts -- the MSP sign-steering
        attribution. Bounded counters, not per-event lists (the first draft's lists
        grew ~10 entries/cycle forever on an always-on daemon); a former per-row
        regen_counts() accessor is DELETED -- it had zero call sites and a docstring
        claiming a consumer that never existed (P3's shape, caught by the panel)."""
        return dict(self._regen_signs)


def _sgn(w):
    return 1 if w > 0.0 else (-1 if w < 0.0 else 0)


def _seed_magnitude(rows, r):
    """A regenerated slot's magnitude: the target row's own minimum live |w| (the
    smallest weight the substrate itself sustains), else the population minimum, else
    0.0 -- a regen into an all-vacant population is a no-op stated as one."""
    own = [abs(w) for w in rows[r] if w != 0.0]
    if own:
        return min(own)
    pop = [abs(w) for row in rows for w in row if w != 0.0]
    return min(pop) if pop else 0.0


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return float(s[m]) if n % 2 else float((s[m - 1] + s[m]) / 2.0)


# ---- self-check ----------------------------------------------------------------------------
def _fresh_rows(seed, n_rows=6, n_cols=52):
    rng = _random.Random(seed)
    return [
        [rng.uniform(0.1, 1.0) * (1 if rng.random() < 0.5 else -1) for _ in range(n_cols)]
        for _ in range(n_rows)
    ]


def main():
    print("=" * 96)
    print(
        "structural_rewiring -- specs/022 T048: DEEP-R budget + MSP counters + turnover (slow clock)"
    )
    print("=" * 96)

    # 1) A/A: stable signs, 50 cycles -- byte-identical weights, turnover 0.0 every cycle.
    rows = _fresh_rows(0)
    frozen = [list(r) for r in rows]
    sr = StructuralRewiring(seed=0)
    for _ in range(50):
        rep = sr.rewire(rows)
        assert rep["turnover"] == 0.0 and rep["pruned"] == 0 and rep["regenerated"] == 0
    assert rows == frozen, "a stable substrate was mutated by the pass (A/A must be exact)"
    print("  green  A/A: 50 cycles on stable signs -- weights byte-identical, turnover exactly 0.0")

    # 2) RED + budget conservation PAST the first event (the T047 lesson): sustained
    #    sign-flip drift for 200 cycles; every flip pruned+paired, total live EXACTLY
    #    at budget after every cycle.
    for seed in range(5):
        rng = _random.Random(1000 + seed)
        rows = _fresh_rows(seed)
        sr = StructuralRewiring(seed=seed)
        budget = None
        for cycle in range(200):
            # the fast clock drives one random live weight through zero (identity violation)
            r = rng.randrange(len(rows))
            live_ix = [c for c, w in enumerate(rows[r]) if w != 0.0]
            if live_ix:
                c = rng.choice(live_ix)
                rows[r][c] = -rows[r][c]  # a sign flip the slow clock must catch
            rep = sr.rewire(rows)
            if budget is None:
                budget = rep["budget"]
            live = sum(1 for row in rows for w in row if w != 0.0)
            assert live == budget, f"budget leak at cycle {cycle}: live {live} != budget {budget}"
            assert rep["regenerated"] == rep["pruned"], (
                f"unpaired cycle {cycle}: {rep['pruned']} pruned vs {rep['regenerated']} regenerated"
            )
        assert sr.report()["pairs_total"] >= 150, (
            "sustained drift produced almost no pairs -- the trigger is dead"
        )
    print(
        "  RED    sustained sign-flip drift, 5 seeds x 200 cycles: every flip pruned AND paired, live == budget after EVERY cycle"
    )

    # 3) MSP homeostatic SIGN steering + its control. Sustained drift with constructed
    #    activity: the three LOW-activity rows must regenerate EXCITATORY (+1) and the
    #    three HIGH rows INHIBITORY (-1). CONTROL (not the method): no activity -- the
    #    seeded draw sits near half. (TARGET steering was built first and MEASURED
    #    VACUOUS here -- steered == uniform, 0.50 vs 0.50: on a dense-at-birth
    #    substrate each cycle's vacancies equal its deficit, the regen SET is forced,
    #    and ordering chooses nothing. Deleted, not weakened; this sign arm is the
    #    dense-substrate form with a real degree of freedom.)
    pos_lo_ctrl = []
    for seed in range(10):
        for arm, act in (("steered", [0.01, 0.01, 0.01, 0.9, 0.9, 0.9]), ("none", None)):
            rng = _random.Random(2000 + seed)
            rows = _fresh_rows(seed)
            sr = StructuralRewiring(seed=seed)
            for _ in range(300):
                r = rng.randrange(len(rows))
                live_ix = [c for c, w in enumerate(rows[r]) if w != 0.0]
                if live_ix:
                    c = rng.choice(live_ix)
                    rows[r][c] = -rows[r][c]
                sr.rewire(rows, activity=act)
            signs = sr.regen_signs()
            lo = [signs.get(r, (0, 0)) for r in (0, 1, 2)]
            hi = [signs.get(r, (0, 0)) for r in (3, 4, 5)]
            frac_pos_lo = sum(p for p, _n in lo) / max(1, sum(p + n for p, n in lo))
            frac_pos_hi = sum(p for p, _n in hi) / max(1, sum(p + n for p, n in hi))
            if arm == "steered":
                assert frac_pos_lo == 1.0 and frac_pos_hi == 0.0, (
                    f"seed {seed}: sign steering leaked (lo +{frac_pos_lo:.2f}, hi +{frac_pos_hi:.2f})"
                )
            else:
                pos_lo_ctrl.append(frac_pos_lo)
    m_ctrl = _median(pos_lo_ctrl)
    assert 0.3 < m_ctrl < 0.7, f"the no-activity control drifted from the seeded draw: {m_ctrl:.2f}"
    print(
        f"  green  MSP sign steering: below-setpoint rows regenerate +1 at 100%, above-setpoint at 0% "
        f"(10 seeds x 300 cycles); no-activity CONTROL draws +1 at {m_ctrl:.0%} (seeded coin)"
    )

    # 4) Instrument red + dead-instrument control: k constructed flips read EXACTLY
    #    k/budget; zero flips read exactly 0.0.
    rows = _fresh_rows(3)
    sr = StructuralRewiring(seed=3)
    sr.rewire(rows)  # first sight
    budget = sr.report()["budget"]
    for c in (0, 1, 2):
        rows[0][c] = -rows[0][c]
    rep = sr.rewire(rows)
    assert rep["pruned"] == 3 and rep["turnover"] == 3 / budget, (
        f"instrument misread: {rep['pruned']} pruned, turnover {rep['turnover']} != {3 / budget}"
    )
    rep2 = sr.rewire(rows)
    assert rep2["turnover"] == 0.0, "the quiet cycle after the burst must read exactly 0.0"
    print(
        f"  green  instrument: 3 constructed flips read exactly 3/{budget}; the quiet next cycle reads exactly 0.0"
    )

    # 5) External vacancies (another mechanism zeroed a live slot) restore to budget
    #    WITHOUT counting as this module's prunes -- the attribution stays honest.
    rows = _fresh_rows(4)
    sr = StructuralRewiring(seed=4)
    sr.rewire(rows)
    rows[2][5] = 0.0  # an external zeroing, not a sign flip
    rep = sr.rewire(rows)
    assert rep["pruned"] == 0 and rep["external_vacancies"] == 1 and rep["regenerated"] == 1
    live = sum(1 for row in rows for w in row if w != 0.0)
    assert live == rep["budget"], "external vacancy not restored to budget"
    print("  green  external vacancy: restored to budget, attributed as external (0 own prunes)")

    # 5b) External ZERO->LIVE re-adoption (the panel's divergence find): on a NON-dense
    #    substrate, an external writer repopulating a vacant slot is RE-ADOPTED at its
    #    current sign -- live-substrate and live-signs stay equal instead of diverging
    #    monotonically (measured pre-fix: +1 connection per cycle, pruned reading 0).
    rows = _fresh_rows(6, n_rows=4, n_cols=6)
    for c in (0, 1):
        rows[0][c] = 0.0  # sparse at first sight: 2 permanent vacancies, budget = 22
    sr = StructuralRewiring(seed=6)
    sr.rewire(rows)
    budget = sr.report()["budget"]
    rows[0][0] = 0.7  # an external mechanism repopulates a vacant slot
    for _ in range(5):
        rep = sr.rewire(rows)
    live_substrate = sum(1 for row in rows for w in row if w != 0.0)
    live_signs = sum(1 for row in sr._signs for s in row if s != 0)
    assert live_substrate == live_signs == budget + 1, (
        f"zero->live divergence: substrate {live_substrate}, signs {live_signs}, budget {budget}"
    )
    assert sr.report()["readopted"] == 0 and rep["pruned"] == 0, "re-adoption misattributed"
    print(
        "  green  zero->live re-adoption: substrate == signs after an external repopulation (no monotone divergence)"
    )

    # 5c) Degenerate activity (zero spread -- e.g. a resting-uniform theta population)
    #    carries no rank information: the sign steer must fall back to the seeded coin,
    #    NEVER the all-inhibitory collapse (measured pre-fix: 0/200 excitatory, because
    #    strict < median is False when every value equals the median).
    rng = _random.Random(7)
    rows = _fresh_rows(7)
    sr = StructuralRewiring(seed=7)
    for _ in range(200):
        r = rng.randrange(len(rows))
        live_ix = [c for c, w in enumerate(rows[r]) if w != 0.0]
        if live_ix:
            c = rng.choice(live_ix)
            rows[r][c] = -rows[r][c]
        sr.rewire(rows, activity=[0.05] * len(rows))  # the stack's uniform resting theta
    allsigns = sr.regen_signs()
    pos = sum(p for p, _n in allsigns.values())
    tot = sum(p + n for p, n in allsigns.values())
    assert tot > 50 and 0.3 < pos / tot < 0.7, (
        f"degenerate activity did not fall back to the coin: {pos}/{tot} excitatory"
    )
    print(
        f"  green  degenerate (zero-spread) activity -> the seeded coin: {pos}/{tot} excitatory (pre-fix: all-inhibitory)"
    )

    # 6) Refusals: NaN weight, mismatched activity.
    sr = StructuralRewiring(seed=5)
    rows = _fresh_rows(5)
    try:
        sr.rewire([[float("nan")] * 3])
        raise AssertionError("accepted a NaN weight")
    except ValueError:
        pass
    try:
        sr.rewire(rows, activity=[0.5])
        raise AssertionError("accepted a mismatched activity vector")
    except ValueError:
        pass
    print(
        "  green  refusals: NaN weight + mismatched activity (ValueError; the wire counts, never raises into sleep)"
    )

    print()
    print(
        "DISCLOSURES: DEEP-R and MSP are ADOPTED mechanisms (Bellec et al.; Butz & van "
        "Ooyen) on the slow clock; THE LIVE WIRE IS PARKED ON MEASUREMENT (module "
        "docstring: zero sign crossings on both live paths, the relax rebind erasing "
        "the trigger, inhibitory washout in 2 steps, seed ratio 0.80-0.87 of typical) "
        "AND REFUSED A SECOND TIME at T054-ii on the cerebellar tissue built to host it "
        "-- the sign structure arrived, the trigger did not: nothing writes those rows "
        "(0 crossings, 0 of 512 weights changed over 2200 driven steps). UNDATED now, "
        "with a checkable landing criterion instead of a date: the first mechanism that "
        "WRITES pf_w, shown by a nonzero crossing count. FR-018's neuter A/B did NOT "
        "wait for the wire -- it landed at T054-ii on the tissue's own elements. "
        "MSP TARGET steering: built first, MEASURED VACUOUS on dense-at-"
        "birth substrates (steered == uniform 0.50/0.50; vacancies == deficit, the "
        "regen set forced) -- deleted, re-opens only with capacity slack. The shipped "
        "sign steer is RANK-relative (row vs population median), not setpoint-"
        "homeostatic: a uniformly-too-quiet population still draws the coin -- said "
        "plainly. The budget is a FLOOR (deficits regenerate; an external mechanism "
        "adding connections above budget is re-adopted, never trimmed -- trimming "
        "another rule's work is the worse failure). Element-type vocabulary (axonal "
        "vs dendritic) is not modelled. Per-instance state (identities, budget, "
        "counters) is process-local -- on a restart first sight re-derives from the "
        "substrate; the instrument's lifetime counters reset (a pending violation at "
        "shutdown re-derives as a birth identity). The attractor's sub-eps zeroing is "
        "a DIFFERENT substrate; the external-vacancy/re-adoption paths above are the "
        "composition seam if they ever meet."
    )
    print(
        "self-check OK: A/A exact + budget conservation past the first event (5x200) + MSP "
        "sign steering vs the seeded-coin control + exact-turnover instrument with quiet "
        "control + external-vacancy attribution + refusals"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
