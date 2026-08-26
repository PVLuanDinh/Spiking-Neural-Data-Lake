"""
Growth stopping criterion -- specs/022 T047, P8 invention #4 (docs/INVENTION_REGISTER.md
row 4; FR-023): per-region growth gates derived from |LP| + charge, with an arm
independent of the executive path.

WHAT IT DECIDES. Per region: may the NEXT refit grow new units, or has this region grown
past its measured optimum? The verdict derives from exactly two evidence streams, both
observed by this gate itself, never handed in by the caller at decision time:

  CHARGE -- the region's own refit trajectory (n_units, held-out acc), fed per refit
  via observe_refit(). Growth charges while added units buy held-out accuracy;
  a run of growth refits that buys nothing beyond the trajectory's own noise scale is
  growth past the optimum. The optimum is MEASURED (the region's own best-acc refit),
  never declared. This is the region-level form of the engine's own l_max=None warrant:
  "the honest limit is where a new unit can no longer earn its place, which is a
  property of the data, not a number chosen ahead of time."

  |LP| -- the T046a facade (the ADOPTED CuriosityDrive readout), read through the
  injected reader. A SUSTAINED EXCURSION of |LP| above the region's own noise floor
  (trailing median + MAD, the last PATIENCE readings all above it) vetoes a pending
  stop and re-opens a held one: the error stream moving at this refit cadence is the
  world-changed signal. HONEST SCOPE (the panel refuted the first draft's gloss): the
  arm claims MOTION, never learning -- T046a measured |LP| unable to distinguish a
  trend from a matched-marginal shuffle (the shuffle scores HIGHER), and a bare
  cur-above-median test was measured a 51.9% coin flip on the live zero-inflated
  channel and ANTI-correlated with competence (a settling |LP| decays monotonically).
  Self-relative on purpose -- T046a measured the live channel ~55x below its
  harness's validated scale, so any fixed threshold is calibrated on the wrong
  distribution; the MAD floor and PATIENCE-consecutive requirement are the same null
  discipline the charge arm carries.

EXECUTIVE-INDEPENDENT (the register's reason: "a gate routed through the executive would
let the guard's caller constrain it" -- the threat()-constrained-by-its-caller incident).
Made structural two ways, plus one wiring fact stated as a fact: (1) should_grow(region)
accepts NO tuning input -- no thresholds, no overrides, no context object; the refusal
is a TypeError, constructed in the self-check; (2) both evidence streams are the gate's
OWN accumulated observations (the |LP| reader is injected at CONSTRUCTION, an anatomy
decision, not at decision time -- though the route it freezes reads the brain's mutable
|LP| facade at decision time, disclosed, not hidden); (3) the wire consults it on the
SLEEP path and main() drives the criterion with no brain at all -- the independent arm
FR-023 names. (3) is a wiring fact, not an independence proof: the |LP| evidence it
consults was accumulated on the wake path the executive helps steer.

THE STOP IS STICKY, AND STOP IS NOT FOREVER. A held stop re-opens ONLY on evidence: a
real new optimum, a paired-mean gain, or a sustained |LP| excursion -- never on the
gate's own frozen measure-only rows (the panel's central find: the first draft's
net-units guard was satisfied by its own freeze after exactly 2*PATIENCE refits,
turning every stop into a 1-in-7 duty cycle on unbounded growth -- 33 units past the
optimum at 200 refits, on the family the harness certified hardest).

WHAT AN INVENTED METHOD OWES (P8), all in main():
  RED       -- a fixed-step overgrowth run trips the gate post-optimum AND the stop
               HOLDS under continued driving (zero growth refits after the first stop,
               10 seeds x 100 refits -- the duty-cycle red the first harness lacked
               because it break-ed at the first stop).
  CONTROL   -- the fixed growth schedule IS the ungated engine (growth_gate=None, the
               incumbent behaviour) -- the OFF arm of the real A/B, not a list of
               literals (the first draft's 'control' was `[True]*24`, unfailable).
  ATTRIBUTION -- decisions(region): every verdict with its reason; the |LP| ablation
               (None-reader vs excursion-reader changes verdicts 10/10) is the lp_in
               edge's charge attribution.
  DISCLOSURE -- both NDP source papers name stopping as UNSOLVED; this is invention,
               not adoption. PATIENCE=3 is PICKED (the first draft's 'matches the
               engine's holdout discipline' derived nothing -- that discipline has no
               run-of-three; the {2,3,4,5} sweep shows stability, a derivation is
               OWED); the noise scale is the trailing window's own median|step|
               (self-relative, P1), converted to the null SE sigma*sqrt(2/PATIENCE)
               (the first draft's noise/sqrt(PATIENCE) was ~1.5x too small -- a 52%
               growth duty cycle on the flat family).
  NULL      -- if gated growth never beats fixed-schedule growth, the criterion is
               DELETED. THE null measurement is the real two-arm engine A/B (gate on
               vs off, same seeds), asserts armed -- the synthetic families cannot
               carry it: their fixed arm is arithmetic on the same ground-truth curve,
               so the comparison is entailed, not measured.

  PYTHONPATH=src python -m snn_data_lake.cortex.growth_stop
"""

import math
import random as _random

# distinct-from: substrate/growth_coverage_gate.py -- CENSUSES which regions have a live
#   growth call site (a repo instrument); this gate DECIDES whether a censused site may
#   keep growing (a brain faculty). Site read.
# distinct-from: cortex/neurogenesis_engine.py -- the GROWER this gate stops; its
#   earn-your-place prune is per-UNIT admission, this is per-REGION continuation. Site read.
# distinct-from: basal_ganglia/learning_progress.py -- the |LP| PRODUCER this gate reads
#   through injection; nothing re-derived. Site read (T046a's dated consumer is this file).
# distinct-from: cortex/growth_program.py -- ADJUDICATED (the panel's P2 find): its
#   _DIVISION_BUDGET bounds per-NODE embryonic division; this gate polices per-REGION
#   refit growth on (units, held-out acc) rows the division machinery does not produce.
#   Different grain, both survive; T042's "superseded by T047" promise was at the wrong
#   grain and is RE-ADJUDICATED in growth_program's own docstring (supersession owed to
#   the task composing region-wide growth live). Site read.

PATIENCE = 3  # consecutive non-improving GROWTH refits before a stop; PICKED -- see DISCLOSURE
# Evidence retention: every window the criterion reads fits in 3*PATIENCE rows; keep 4x
# that (sized FROM the criterion's own need, not tuned) so the noise scale reflects the
# RECENT regime on an always-on daemon -- the panel measured the unbounded first draft at
# 13.6ms/verdict and ~10MB of history at 50k refits, with the null scale dominated by
# ancient refits.
_WINDOW = 4 * (3 * PATIENCE)

# The T044 lesson pre-applied (class-qualified key, declared the day the symbol lands):
GRADIENT_CLAIM = {
    "GrowthStop.observe_refit": (
        "NOT_A_RULE",
        "appends one (n_units, acc) evidence row to the gate's per-region history -- no "
        "weights, no plasticity, no error propagation; the verdict over that history is "
        "a comparison, not a learning rule",
    ),
}


class GrowthStop:
    """Per-region growth gate. Feed observe_refit() per refit; ask should_grow(region).

    SINGLE-CONSUMER PER REGION by design: one grower per region feeds one gate (the
    engine's refit loop); the gate is not a shared bus."""

    def __init__(self, lp_reader=None):
        # lp_reader: the T046a LearningProgress facade (or None until the brain's lazy
        # |LP| exists). INJECTED AT CONSTRUCTION -- anatomy, not a per-decision input.
        # None is honest: with no fed |LP| channel there is NO plateau evidence, and the
        # |LP| arm can neither veto nor support a stop (absence != zero, T046a's contract).
        self._lp = lp_reader
        self._hist = {}  # region -> [(n_units, acc)], trailing _WINDOW rows
        self._lp_hist = {}  # region -> [|LP| at each refit], trailing _WINDOW
        self._decisions = {}  # region -> [(fed_count, verdict, reason)], trailing _WINDOW
        self._fed = {}  # region -> MONOTONE fed-row count (the cursor; survives the trim)

    # ---- evidence ----------------------------------------------------------------------------
    # distinct-from: basal_ganglia/learning_progress.py -- its observe() feeds ERROR levels
    #   into the |LP| EMAs; this feeds REFIT OUTCOMES (units, acc) into the charge history.
    #   Different stream, and this gate READS that module's output, never re-derives it.
    # distinct-from: basal_ganglia/r_stdp.py, brain_atlas.py, brainstem/signal_loop.py --
    #   eligibility measurement, region lookup, unknowns store; the matcher fired on
    #   generic feed/observe verbs. Sites read.
    def observe_refit(self, region, n_units, heldout_acc):
        """One refit's outcome for `region`, in the engine's own trajectory vocabulary
        (tick, n_units, acc) minus the tick. Refuses non-finite / out-of-range inputs: a
        poisoned trajectory row would decide growth silently. No `pruned` parameter ON
        PURPOSE: the first draft carried one and the criterion never read it -- a dead
        input advertising evidence the verdict cannot feel (the T045 dead-sigma lesson)."""
        u = int(n_units)
        a = float(heldout_acc)
        if u < 0 or not (math.isfinite(a) and 0.0 <= a <= 1.0):
            raise ValueError(
                f"refit row must be (units>=0, acc in [0,1]), got {(n_units, heldout_acc)!r}"
            )
        h = self._hist.setdefault(region, [])
        h.append((u, a))
        del h[:-_WINDOW]  # trailing-window retention; reason at _WINDOW
        self._fed[region] = self._fed.get(region, 0) + 1
        # record the |LP| seen AT this refit into the gate's own trailing history --
        # the self-relative comparison base. An unfed channel records nothing.
        lp_now = self._read_lp(region)
        if lp_now is not None:
            lh = self._lp_hist.setdefault(region, [])
            lh.append(lp_now)
            del lh[:-_WINDOW]

    def _read_lp(self, region):
        if self._lp is None:
            return None
        try:
            v = float(self._lp.lp(region))
        except Exception:
            return None  # unfed channel: absence is a fact, not a zero
        return v if math.isfinite(v) else None

    # ---- the criterion -----------------------------------------------------------------------
    def should_grow(self, region):
        """THE VERDICT. True = the next refit may grow; False = this region's growth is
        gated off, and the stop HOLDS until charging evidence re-opens it.

        Takes the region name and NOTHING else -- the executive-independence contract.
        An unobserved region grows: the gate cannot stop what it has never measured.
        Idempotent per refit: repeated calls at unchanged history REPLACE the last log
        entry instead of appending (the panel: a diagnostic poll must not inflate the
        attribution log the register cites)."""
        hist = self._hist.get(region)
        log = self._decisions.setdefault(region, [])
        verdict, reason = self._decide(hist, self._lp_hist.get(region, []), log)
        # keyed by the MONOTONE fed count, never len(history): the trimmed history caps
        # at _WINDOW, and a len-keyed entry would collide forever once it does -- the
        # dedup then replaces the last slot for the rest of the run and the log freezes
        # on the first _WINDOW verdicts (caught by this harness's own |LP| rate arm).
        entry = (self._fed.get(region, 0), verdict, reason)
        if log and log[-1][0] == entry[0]:
            log[-1] = entry
        else:
            log.append(entry)
            del log[:-_WINDOW]  # bounded, like the evidence windows
        return verdict

    # distinct-from: cortex/prefrontal_executive.py -- its decide-then-gate writes the
    #   WORKING-MEMORY goal by evidence margin; nothing about growth. Site read.
    # distinct-from: cortex/reasoning_curriculum.py -- decide() there scores premises off
    #   a grown codebook; its growth is the learner's own, ungated by design. Site read.
    # distinct-from: reasoning/ar_spiking_lm_serve.py, reasoning/arc_agi/brain_player.py,
    #   reasoning/arc_agi/rule_inducer.py -- a lazy import and two action-selection races;
    #   the matcher fired on generic verbs. Sites read.
    def _decide(self, hist, lp_hist, log):
        """The criterion. STATELESS over (history, log): the same inputs give the same
        verdict -- no hidden counter to inflate on polling.

        OPEN state -- CHARGING (growth still paying) at a window endpoint iff EITHER a
        refit in the last PATIENCE beat the then-running-best by MORE than the noise
        scale (a real new optimum, not a noise re-max -- the first draft compared
        against the raw max and the saturated regime's noise maximum defeated it, 8 of
        10 seeds), OR the paired-mean test shows net gain above the mean-difference's
        own null scale sigma*sqrt(2/PATIENCE), sigma = noise/0.954 (the panel measured
        the first draft's noise/sqrt(PATIENCE) margin ~1.5x too SMALL -- median|step|
        of iid noise is 0.954 sigma, and the null SE of a difference of two
        PATIENCE-means is sigma*sqrt(2/PATIENCE) -- giving ~20-25%% false-charging per
        endpoint and a 52%% growth duty cycle on the flat family). STOP requires
        NOT-charging at PATIENCE consecutive endpoints, growth actually present in the
        window, and no |LP| excursion.

        STOPPED state -- THE STOP IS STICKY (the panel's central find: the first draft
        re-opened on `units[-1] <= units[-1-2*PATIENCE]`, which the gate's OWN frozen
        measure-only rows satisfy after exactly 2*PATIENCE refits -- a self-erasing
        guard that turned every stop into a 1-in-7 duty cycle on unbounded growth,
        measured 45 units / 33 past the optimum at 200 refits). A held stop re-opens
        ONLY on charging evidence at the latest endpoint: a real new optimum, a
        paired-mean gain, or a sustained |LP| excursion. Frozen rows are the gate's own
        doing and can never count as 'nothing to gate'. Asymmetry disclosed: re-open is
        single-endpoint (fast), stop is PATIENCE-consecutive (slow) -- conservative
        toward growth."""
        if not hist:
            return True, "unobserved region: no evidence to stop on"
        need = 2 * PATIENCE + PATIENCE  # deepest consecutive endpoint still needs a full window
        if len(hist) < need:
            return True, f"insufficient history ({len(hist)} refits < {need})"
        accs = [a for _u, a in hist]
        units = [u for u, _a in hist]
        steps = [abs(accs[i] - accs[i - 1]) for i in range(1, len(accs))]
        noise = _median(steps)
        stopped = bool(log) and log[-1][1] is False
        if stopped:
            # Re-open ONLY on the two noise-robust evidence classes. The paired-mean
            # test is deliberately NOT a re-open trigger: at frozen units it measures
            # pure holdout noise, and at its 1-sigma bar it false-re-opened ~16% per
            # refit -- the redesign's own duty-cycle red caught it leaking 5-15 growth
            # refits per 100 (a sustained real gain at frozen units beats the old
            # optimum soon enough, which IS trigger one).
            why = self._new_optimum(accs, len(hist), noise)
            if why is not None:
                return True, f"re-opened: {why}"
            lp_why = self._lp_excursion(lp_hist)
            if lp_why is not None:
                return True, f"re-opened: {lp_why}"
            return False, (
                "stop holds: no real new optimum and no |LP| excursion "
                f"since the stop (noise scale {noise:.4f})"
            )
        # OPEN: growth must actually be present in the window for a stop to be about
        # anything. Blind spot DISCLOSED, not hidden: net units cannot see grow-one/
        # prune-one churn (admissions at flat net count) -- the panel constructed flat
        # units + flat acc reading 'nothing to gate' 40/40; policing admissions needs an
        # evidence stream the (units, acc) vocabulary does not carry.
        if units[-1] <= units[-1 - 2 * PATIENCE]:
            return True, "no net growth over the window: nothing to gate"
        for k in range(PATIENCE):  # PATIENCE consecutive endpoints must all read not-charging
            upto = len(hist) - k
            why = self._charging_at(accs, upto, noise)
            if why is not None:
                return True, why
        lp_why = self._lp_excursion(lp_hist)
        if lp_why is not None:
            return True, f"stop vetoed: {lp_why}"
        best_i = max(range(len(accs)), key=lambda i: accs[i])
        return False, (
            f"grown past the measured optimum (best acc {accs[best_i]:.3f} at u={units[best_i]}): "
            f"{PATIENCE} consecutive endpoints with no real new optimum and paired-mean gain below "
            f"the null scale (noise {noise:.4f}), growth present, no |LP| excursion"
        )

    @staticmethod
    def _charging_at(accs, upto, noise):
        """Charging evidence on the prefix accs[:upto]; returns the reason string, or None
        if NOT charging at this endpoint."""
        if upto < 2 * PATIENCE:
            # unreachable through _decide (need = 3*PATIENCE); FAIL-OPEN loudly rather
            # than let an empty `prior` slice make sum([])/PATIENCE fabricate a gain
            # (the panel constructed 'charging: +0.5000 > margin 0.0000' this way).
            return f"window underflow ({upto} < {2 * PATIENCE}): refusing to judge, growth allowed"
        why = GrowthStop._new_optimum(accs, upto, noise)
        if why is not None:
            return why
        recent = accs[upto - PATIENCE : upto]
        prior = accs[upto - 2 * PATIENCE : upto - PATIENCE]
        gain = sum(recent) / PATIENCE - sum(prior) / PATIENCE
        sigma = noise / 0.954  # median|step| -> per-sample sd, iid null
        margin = sigma * math.sqrt(2.0 / PATIENCE)
        if gain > margin:
            return f"charging: paired-mean gain {gain:+.4f} > null scale {margin:.4f} over the last {2 * PATIENCE} refits"
        return None

    @staticmethod
    def _new_optimum(accs, upto, noise):
        """A refit in the last PATIENCE of accs[:upto] beating the then-running best by
        MORE than the noise scale -- the noise-immune evidence class (on bounded holdout
        noise a flat stream cannot fake it: beating a running max by a further full
        noise scale sits outside the noise's own support). Returns the reason, or None."""
        for i in range(max(0, upto - PATIENCE), upto):
            prev_best = max(accs[:i]) if i else -1.0
            if accs[i] > prev_best + noise:
                return f"charging: refit {i + 1} beat the running best by more than the noise scale {noise:.4f} (a real new optimum)"
        return None

    @staticmethod
    def _lp_excursion(lp_hist):
        """The |LP| arm: an EDGE-TRIGGERED SUSTAINED EXCURSION of the region's |LP|
        above its own order-statistic floor -- the last PATIENCE readings each above the
        trailing PATIENCE/(PATIENCE+1) quantile, with the reading before the run NOT
        above it (a rising edge, so one excursion fires once, never per-refit for its
        whole duration). DERIVED, distribution-free bar: under an exchangeable
        stationary stream, P(all PATIENCE recent readings exceed the trailing upper
        1/(PATIENCE+1) tail) <= (1/(PATIENCE+1))**PATIENCE ~ 1.6%% at PATIENCE=3 --
        immune to the zero-inflation that collapsed the first redesign's median+MAD
        floor to 0.0 (half the live channel's readings are exact zeros, so median AND
        MAD are zero and every nonzero reading cleared the 'floor': measured 57.8%%).
        HONEST SCOPE: the arm claims the error stream MOVING at this refit cadence,
        never 'learning' (T046a measured |LP| unable to distinguish a trend from a
        matched-marginal shuffle -- the shuffle scores HIGHER; a regression is
        capacity-relevant motion too, which is why the rectified readout is
        deliberately NOT used here). The very first draft's bare `cur > median` was a
        measured 50/50 coin flip and ANTI-correlated with competence motion (a
        settling |LP| decays monotonically and can never sit above its own median).
        Returns the reason, or None."""
        if len(lp_hist) < 2 * PATIENCE + 1:
            return None  # not enough |LP| evidence for an excursion claim (absence != veto)
        trailing = sorted(lp_hist[:-PATIENCE])
        n = len(trailing)
        q_ix = max(0, math.ceil(n * PATIENCE / (PATIENCE + 1)) - 1)
        floor = trailing[q_ix]
        recent = lp_hist[-PATIENCE:]
        edge_prev = lp_hist[-(PATIENCE + 1)]
        if all(v > floor for v in recent) and not edge_prev > floor:
            return (
                f"|LP| excursion (rising edge): last {PATIENCE} readings all above the trailing "
                f"{PATIENCE}/{PATIENCE + 1} quantile ({min(recent):.4f} > {floor:.4f}) -- the error stream is moving"
            )
        return None

    # distinct-from: self_improve.py -- the brain's self-model graph census. Site read.
    # distinct-from: brain_decision.py -- a monitor gate pass-through for logging. Site read.
    # distinct-from: live_brain.py -- the per-region SPIKE snapshot. Site read.
    # distinct-from: play_brain.py -- per-color occupancy fractions. Site read.
    # distinct-from: arc_solver.py -- the per-task primitive pool. Site read.
    # distinct-from: growth_program.py -- adjudicated at module top (different GRAIN). Site read.
    # distinct-from: brain_geometry.py, doc_budget_gate.py, dormancy_gate.py -- module
    #   placement, a doc gate's red census, an import census; the matcher fired on
    #   generic count words over a one-line accessor. Sites read.
    def observed(self, region):
        """How many refit rows this gate has EVER been fed for `region` -- the brain's
        incremental-feed cursor (feed trajectory[observed():], never re-feed). A
        monotone counter, deliberately NOT len(history): the evidence windows trim to
        _WINDOW rows and a cursor that shrank with them would re-feed old rows."""
        return self._fed.get(region, 0)

    # distinct-from: region_wiring.py -- worker-count bound. Site read.
    # distinct-from: pragmatics.py -- utterance surface form. Site read.
    # distinct-from: gradient_claim_gate.py -- per-state census counts; the matcher fired
    #   on generic region/state words over a one-line key listing. Sites read.
    def regions(self):
        """The regions this gate has decided for -- the state() surface iterates THIS,
        never a hardcoded list."""
        return sorted(self._decisions)

    # ---- attribution -------------------------------------------------------------------------
    def decisions(self, region):
        """Per-region verdict log [(fed_count, verdict, reason)], trailing _WINDOW -- the
        P8 attribution."""
        return list(self._decisions.get(region, []))


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return float(s[m]) if n % 2 else float((s[m - 1] + s[m]) / 2.0)


# ---- manifest ------------------------------------------------------------------------------
def declare_ports(manifest):
    """T047: THE dated consumer of learning_progress.lp_out (T046a declared it EDGE-LESS
    with this task named -- the T030 precedent; the edge lands here, WITH its consumer).
    The gate's verdict itself is an in-process call on the sleep path, not an inter-faculty
    stream, so no OUT port is declared for it -- declaring one would be anatomy ahead of
    any consumer."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "growth_stop",
        Port(
            name="lp_in",
            region="cortex",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="IN",
            declaring_module="growth_stop",
            # typed-merge, stated honestly (the panel: the first rationale described a
            # multi-region merge the code never performs): TODAY this is an in-process
            # single-region read (_LPRoute -> brain._lp.lp("cortex")); typed-merge is
            # the declared CONTRACT because contributions are region-KEYED by the
            # consumer's own signature (observe/lp take region), and a sum would erase
            # FR-023's per-region attribution the day a second region lands. The edge's
            # charge attribution is the |LP|-ablation arm in main() (verdicts change
            # between a None reader and an excursion reader).
            join_policy="typed-merge",
        ),
    )
    manifest.declare_edge("learning_progress", "lp_out", "growth_stop", "lp_in")


# ---- self-check ----------------------------------------------------------------------------
def _true_acc(kind, u, u_star=12):
    """Ground-truth acc(u) for the regime families (noise added by the caller)."""
    if kind == "improving":
        return 0.2 + 0.6 * (1 - math.exp(-u / 8.0))
    if kind == "overgrowth":  # peak at u_star then decline: growth past the optimum HURTS
        return 0.8 - 0.012 * abs(u - u_star) - (0.0 if u <= u_star else 0.018 * (u - u_star))
    if kind == "saturated":  # flat after u_star: growth buys nothing
        return 0.2 + 0.6 * (1 - math.exp(-min(u, u_star) / 4.0))
    if kind == "noise":  # trendless: no signal either way
        return 0.5
    if kind == "slow":  # genuinely improving, near the discrimination floor
        return 0.3 + 0.001 * u
    raise ValueError(kind)


def _drive(kind, seed, gate, n_refits, u_star=12):
    """THE PRODUCTION LOOP SHAPE (the panel's central instrument -- the first harness
    break-ed at the first stop and never saw the duty cycle): verdict BEFORE each refit;
    True -> the fixed-step grower adds one unit; False -> the engine's measure-only row
    (frozen units, fresh acc). Returns (unit_trace, grow_flags)."""
    rng = _random.Random(seed)
    u = 0
    trace, grew = [], []
    for _ in range(n_refits):
        allow = gate.should_grow("cortex")
        if allow:
            u += 1
        a = min(1.0, max(0.0, _true_acc(kind, u, u_star) + rng.uniform(-0.01, 0.01)))
        gate.observe_refit("cortex", u, a)
        trace.append(u)
        grew.append(allow)
    return trace, grew


class _FlatLP:
    """A stub |LP| reader pinned flat -- competence NOT moving (never an excursion)."""

    def __init__(self, value=0.001):
        self._v = value

    def lp(self, region):
        return self._v


class _StationaryLP:
    """Stationary noisy |LP| -- the live channel's shape (T046a: zero-inflated, tiny).
    The first draft's bare cur>median veto read 51.9% on this; the excursion arm must
    stay near zero."""

    def __init__(self, seed):
        self._rng = _random.Random(seed)

    def lp(self, region):
        return 0.0 if self._rng.random() < 0.5 else self._rng.random() * 0.01


class _ExcursionLP:
    """Quiet floor, then a SUSTAINED excursion from `at` -- the world-changed signal the
    veto exists for."""

    def __init__(self, seed, at=14):
        self._rng = _random.Random(seed)
        self._n = 0
        self._at = at

    def lp(self, region):
        self._n += 1
        base = self._rng.random() * 0.005
        return base + (0.05 if self._n >= self._at else 0.0)


def main():
    global PATIENCE  # the sweep arm swaps it and restores it
    print("=" * 96)
    print("growth_stop -- specs/022 T047: |LP|+charge growth gates (P8 invention #4)")
    print("=" * 96)

    # 1) RED + THE DUTY-CYCLE RED, on the production loop shape. The overgrowth family
    #    must trip the gate after its measured optimum, and the stop must HOLD: zero
    #    growth refits after the first stop (the first draft re-opened on its own frozen
    #    rows -- a 1-in-7 duty cycle, 33 units past the optimum at 200 refits).
    first_stops, post_stop_grows = [], []
    for seed in range(10):
        gate = GrowthStop(lp_reader=_FlatLP())
        trace, grew = _drive("overgrowth", seed, gate, n_refits=100)
        stops = [i for i, g in enumerate(grew) if not g]
        assert stops, f"RED unreachable at seed {seed}: overgrowth never tripped the gate"
        first = stops[0]
        assert 12 < trace[first] + 1 <= 12 + 2 * PATIENCE + 2, (
            f"seed {seed}: first stop at u={trace[first]}, outside the post-optimum window"
        )
        first_stops.append(first + 1)
        post_stop_grows.append(sum(grew[first:]))
    assert sum(post_stop_grows) == 0, (
        f"the stop does not hold: {post_stop_grows} growth refits after the first stop "
        "(10 seeds x 100 refits) -- the duty-cycle failure"
    )
    print(
        f"  RED    overgrowth trips at refit {sorted(set(first_stops))} (10 seeds; optimum u=12), "
        f"then HOLDS: 0 growth refits in the remaining ~{100 - max(first_stops)} (was a 1-in-7 duty cycle pre-fix)"
    )

    # 2) Regimes on the production loop, 10 seeds each, at BOTH horizons (the panel: the
    #    first harness pinned 'improving never stops' at n=24 only -- at 40 the
    #    exponential plateaus and stopping there is the design WORKING). saturated and
    #    noise must stop AND hold (growth at flat acc is growth buying nothing); slow
    #    (+0.001/refit, inside the noise) documents the measured discrimination floor.
    # improving's hard no-stop band ends where the exponential's TRUE per-window gain
    # falls to the criterion's own null scale (~refit 20 for this family); past that the
    # tail is plateau-approach and stopping there is the boundary, not a false stop --
    # measured in the plateau note below rather than asserted either way.
    for kind, horizon, want in (
        ("improving", 20, "no-stop"),
        ("saturated", 60, "stop-and-hold"),
        ("noise", 60, "stop-and-hold"),
    ):
        held, stopped_at = [], []
        for seed in range(10):
            gate = GrowthStop(lp_reader=_FlatLP())
            trace, grew = _drive(kind, seed, gate, n_refits=horizon)
            stops = [i for i, g in enumerate(grew) if not g]
            if want == "no-stop":
                assert not stops, (
                    f"{kind}@{horizon}: stopped at refit {stops[0] + 1} during genuine improvement"
                )
            else:
                assert stops, (
                    f"{kind}@{horizon}: never stopped (growth bought nothing for {horizon} refits)"
                )
                stopped_at.append(stops[0] + 1)
                held.append(sum(grew[stops[0] :]))
        if want == "no-stop":
            print(
                f"  green  regime {kind:10s}@{horizon}: 10/10 seeds grow throughout (pre-plateau)"
            )
        else:
            frac = sum(held) / max(1, sum(len(range(s, 60)) for s in stopped_at))
            assert frac <= 0.10, f"{kind}: post-stop growth duty {frac:.1%} -- the stop leaks"
            print(
                f"  green  regime {kind:10s}@{horizon}: stop 10/10 (median refit {sorted(stopped_at)[5]}), "
                f"post-stop growth duty {frac:.1%} (bound 10%)"
            )
    # the plateau horizon, DISCLOSED with its measurement rather than hidden by the
    # short harness: at n=40 the improving exponential's marginal gain falls inside the
    # noise and the gate stops -- growth buying nothing anymore IS the design.
    plate = []
    for seed in range(10):
        gate = GrowthStop(lp_reader=_FlatLP())
        trace, grew = _drive("improving", seed, gate, n_refits=40)
        stops = [i for i, g in enumerate(grew) if not g]
        if stops:
            plate.append(stops[0] + 1)
    print(
        f"  note   improving@40: stops in {len(plate)}/10 seeds (refits {sorted(set(plate))}) -- the "
        "exponential's plateau, where marginal gain sits inside the noise; measured, not hidden"
    )
    # the discrimination floor, measured and DISCLOSED: a true +0.001/refit slope (a
    # real +0.06 over 60 refits) is inside the +/-0.01 noise at this window and reads
    # as a plateau. The criterion cannot see below its noise scale; declaring the floor
    # is the honest form (the alternative -- an improving family chosen far from the
    # boundary -- implies a discrimination the method does not have).
    slow_stops = 0
    for seed in range(10):
        gate = GrowthStop(lp_reader=_FlatLP())
        _trace, grew = _drive("slow", seed, gate, n_refits=60)
        slow_stops += any(not g for g in grew)
    print(
        f"  note   discrimination floor: true +0.001 acc/refit stops {slow_stops}/10 seeds -- gains "
        "inside the trajectory's own noise scale are indistinguishable from a plateau, BY CONSTRUCTION"
    )

    # 3) The |LP| arm, on honest channels + THE ABLATION (its charge attribution).
    #    (a) stationary/zero-inflated (the live shape): the excursion test must be quiet
    #        (the first draft's bare cur>median was a 51.9% coin flip here);
    #    (b) a sustained excursion re-opens a held stop -- the world-changed signal;
    #    (c) ablation None-vs-excursion: the |LP| term must CHANGE verdicts, or the arm
    #        is decoration (P8: delete on a null).
    veto_like = 0
    checks = 0
    for seed in range(10):
        gate = GrowthStop(lp_reader=_StationaryLP(seed))
        _trace, grew = _drive("overgrowth", seed, gate, n_refits=100)
        reasons = [r for _i, _v, r in gate.decisions("cortex")]
        # count FIRES only ("re-opened:"/"stop vetoed:" prefixes) -- the held-stop reason
        # NAMES the absent excursion, and a bare substring match counted it 100%
        veto_like += sum(
            1 for r in reasons if r.startswith(("re-opened: |LP|", "stop vetoed: |LP|"))
        )
        checks += len(reasons)
    rate = veto_like / max(1, checks)
    assert rate <= 0.02, (
        f"stationary |LP| fires the excursion test at {rate:.1%} -- still a coin flip"
    )
    print(
        f"  green  |LP| stationary (live shape): excursion rate {rate:.2%} over {checks} verdicts (bar 2%; bare cur>median measured 51.9%)"
    )
    reopened = 0
    for seed in range(10):
        gate = GrowthStop(lp_reader=_ExcursionLP(seed, at=30))
        _trace, grew = _drive("overgrowth", seed, gate, n_refits=60)
        stops = [i for i, g in enumerate(grew) if not g]
        if stops and any(grew[i] for i in range(stops[0], 60)):
            reopened += 1
            reasons = [r for _i, _v, r in gate.decisions("cortex")]
            assert any(r.startswith("re-opened: |LP|") for r in reasons), (
                "re-open without the excursion FIRE marker in attribution (the held-stop "
                "reason mentions the absent excursion; a substring match counted that)"
            )
    assert reopened == 10, f"a sustained |LP| excursion re-opened only {reopened}/10 held stops"
    none_verdicts, exc_verdicts = [], []
    for seed in range(10):
        g_none = GrowthStop(lp_reader=None)
        g_exc = GrowthStop(lp_reader=_ExcursionLP(seed, at=30))
        _t1, grew_none = _drive("overgrowth", seed, g_none, n_refits=60)
        _t2, grew_exc = _drive("overgrowth", seed, g_exc, n_refits=60)
        none_verdicts.append(tuple(grew_none))
        exc_verdicts.append(tuple(grew_exc))
    differing = sum(1 for a, b in zip(none_verdicts, exc_verdicts, strict=True) if a != b)
    assert differing == 10, (
        f"|LP| ablation: verdicts identical with and without the reader on {10 - differing}/10 seeds "
        "-- the arm is decoration (P8: delete)"
    )
    print(
        "  green  |LP| excursion re-opens 10/10 held stops; ablation None-vs-excursion changes verdicts 10/10 (the arm is LIVE -- the lp_in edge's charge attribution)"
    )

    # 4) Refusals (input validation, labelled as such -- not criterion reds).
    gate = GrowthStop()
    try:
        gate.should_grow("cortex", 0.9)  # an attempted caller threshold
        raise AssertionError(
            "should_grow accepted a second argument -- the guard is caller-constrainable"
        )
    except TypeError:
        pass
    for bad in ((5, float("nan")), (5, 1.5), (-1, 0.5)):
        try:
            gate.observe_refit("cortex", *bad)
            raise AssertionError(f"accepted poisoned row {bad!r}")
        except ValueError:
            pass
    print(
        "  green  refusals: caller-constraint (TypeError) + poisoned rows (ValueError) -- validation, not criterion reds"
    )

    # 5) Re-open on a REAL new optimum (sticky-stop semantics: frozen rows never re-open;
    #    evidence does).
    gate = GrowthStop(lp_reader=_FlatLP())
    _trace, grew = _drive("overgrowth", 0, gate, n_refits=40)
    assert not gate.should_grow("cortex"), "precondition: gate is holding a stop"
    gate.observe_refit("cortex", _trace[-1], 0.95)  # beats the old optimum by >> noise
    assert gate.should_grow("cortex"), "a refit beating the old optimum must re-open the gate"
    print("  green  re-open: a REAL new optimum re-opens a held stop (frozen rows alone never do)")

    # 6) Idempotence + PATIENCE sweep. Repeated polls at unchanged history must neither
    #    flip the verdict nor inflate the attribution log. PATIENCE is PICKED (the
    #    panel: 'three consecutive is a run' restates the value; no derivation exists) --
    #    the sweep shows the criterion's behaviour is stable in it, which is evidence of
    #    non-brittleness, not a derivation.
    gate = GrowthStop(lp_reader=_FlatLP())
    _drive("overgrowth", 0, gate, n_refits=40)
    n_log = len(gate.decisions("cortex"))
    v0 = gate.should_grow("cortex")
    for _ in range(5):
        assert gate.should_grow("cortex") == v0
    assert len(gate.decisions("cortex")) == n_log, "polling inflated the attribution log"
    print("  green  idempotent: 6 polls at unchanged history -- same verdict, log not inflated")
    _saved = PATIENCE
    sweep = {}
    for p in (2, 3, 4, 5):
        PATIENCE = p
        stops = []
        for seed in range(5):
            g = GrowthStop(lp_reader=_FlatLP())
            _t, gr = _drive("overgrowth", seed, g, n_refits=60)
            s = [i for i, x in enumerate(gr) if not x]
            stops.append((s[0] + 1) if s else None)
        sweep[p] = stops
    PATIENCE = _saved
    print(
        f"  note   PATIENCE sweep (PICKED, derivation owed): first-stop refits {sweep} -- ordering stable, trip moves with the window as expected"
    )

    # 7) THE NULL CLAUSE -- the real-engine two-arm A/B (torch tier; the synthetic
    #    families cannot carry it: their fixed arm is arithmetic on the same curve, so
    #    the comparison is entailed, not measured -- the panel's find). Gate ON vs OFF,
    #    same seeds. WIN = fewer units at non-inferior accuracy. NULL = delete.
    try:
        import torch  # noqa: F401

        from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove

        d_units, d_accs, any_stop = [], [], 0
        for seed in (1, 2, 3, 7, 11, 13):
            eng_off = NeurogenesisSelfImprove(seed=seed, refit_every=20, device="cpu")
            for _ in range(1600):
                eng_off.tick()
            gate = GrowthStop(lp_reader=_FlatLP())

            def _g(traj, _gate=gate):
                for _t, u, a in list(traj)[_gate.observed("cortex") :]:
                    _gate.observe_refit("cortex", int(u), float(a))
                return _gate.should_grow("cortex")

            eng_on = NeurogenesisSelfImprove(
                seed=seed, refit_every=20, device="cpu", growth_gate=_g
            )
            for _ in range(1600):
                eng_on.tick()
            d_units.append(eng_on.grown_units - eng_off.grown_units)
            d_accs.append(eng_on.predict_accuracy - eng_off.predict_accuracy)
            any_stop += any(not v for _i, v, _r in gate.decisions("cortex"))
        assert any_stop > 0, (
            "NULL-red precondition: the gate never stopped on the live engine (6 seeds x 80 refits)"
        )
        assert sum(d_units) <= 0, (
            f"NULL TRIPPED: gated growth used MORE units than fixed ({d_units}, sum {sum(d_units):+d}) -- delete the criterion"
        )
        assert _median(d_accs) >= -0.05, (
            f"NULL TRIPPED: gated accuracy collapsed vs fixed (deltas {[f'{d:+.3f}' for d in d_accs]}) -- delete the criterion"
        )
        print(
            f"  green  REAL two-arm A/B (6 seeds x 1600 ticks): unit deltas {d_units} (sum {sum(d_units):+d}), "
            f"acc deltas median {_median(d_accs):+.4f}, stops on {any_stop}/6 seeds -- the NULL is armed and did not trip"
        )
    except ImportError:
        print(
            "  note   real-engine A/B SKIPPED (no torch in this tier); the synthetic arms above carry the criterion reds, the A/B carries the null"
        )

    print()
    print(
        "DISCLOSURES: both NDP source papers name growth stopping as UNSOLVED -- this is "
        "invention (P8), and the NULL clause (the real two-arm A/B above) remains its "
        "standing deletion licence. PATIENCE=3 is PICKED (the sweep shows stability; a "
        "derivation from the trajectory's own autocorrelation is OWED, not claimed). "
        "Blind spots, measured: (a) grow-one/prune-one churn at flat net units reads "
        "'nothing to gate' -- admissions are invisible in the (units, acc) vocabulary; "
        "(b) true gains inside the noise scale read as plateaus (the discrimination "
        "floor above); (c) the |LP| arm claims MOTION, never learning direction (T046a: "
        "|LP| cannot distinguish trend from shuffle). Re-open is single-endpoint (fast) "
        "vs the PATIENCE-consecutive stop (slow) -- asymmetric toward growth, on "
        "purpose. The gate's evidence rides the engine's persisted trajectory tail, so "
        "a daemon relaunch resumes the SAME verdict state (P4b)."
    )
    print(
        "self-check OK: overgrowth red + duty-cycle hold + regimes at two horizons + measured "
        "discrimination floor + |LP| stationary/excursion/ablation + refusals + evidence re-open + "
        "idempotence + PATIENCE sweep + the armed real-engine null"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
