"""
Learning progress + the goal store -- specs/022 T046a: the |LP| third factor and the
lake-resident Goal records, discharging T028's dated |LP| disposition.

WHAT |LP| IS. Absolute learning progress (the Oudeyer/Kaplan signal): per region, a
COMPETENCE level (1 - normalized prediction error) tracked by TWO EWMAs on different
timescales; |LP| = |fast - slow| -- large while competence is MOVING (learning OR
regressing; the absolute value is the point, a regression is progress-shaped news), small
on any plateau (good or bad). It is a LEVEL channel over change: flat error reads ~0,
improving reads >0, worsening reads >0, trendless noise is filtered by the slow EWMA --
the four regimes its self-check pins.

THE DISCHARGED DISPOSITION (SYNTHESIS section 6, T028): `percept_surprise.surprise_out`'s
third intended consumer is |learning-progress| gating, "discharged at T046a ... whose own
task text names this disposition" -- THE CONSUMER DECLARES THE EDGE, so this module's
declare_ports carries `percept_surprise.surprise_out -> learning_progress.surprise_in`,
and the live wire feeds the brain's real per-tick surprise scalar into the cortex channel
(the ONE live error channel today; other regions' error channels join as their producers
land, dated to their own tasks, and the REGIONS argument stays honest: lp() reports only
channels that have actually been fed). The lp_out port is declared EDGE-LESS with its
dated consumer named (T047's growth gates) -- the T030 broadcast_out precedent; declaring
that edge today would be anatomy ahead of its consumer.

THE GOAL STORE (data-model "Goal (S1+)"): records of (embedding, achievement_function_ref,
|LP| history, outcomes), IN THE ONE LAKE under the NEW record tag _T_GOAL (home.py's tag
registry owns the byte; SYNTHESIS section 3.5 names the goal class as owing a migration
disposition -- v1 is READ-AS-IS, and the round-trip red runs over records this store
actually wrote to a scratch lake, closed, and re-read in a fresh process-shaped view).
P4b: the lake is the record; the vector index (normalized-embedding cosine top-k) is a
DERIVED CACHE rebuilt from replay -- the index-is-cache red asserts a dropped index
rebuilt from the lake answers queries identically. `achievement_function_ref` is a STORED
REFERENCE (a dotted name), deliberately never resolved or executed here: executing a
lake-carried ref is a dispatch surface T047+ must open with its own review, not a side
effect of storage -- disclosed, not smoothed.

  PYTHONPATH=src python -m snn_data_lake.basal_ganglia.learning_progress
"""

import json
import math

from snn_data_lake.neuromodulation.intrinsic_motivation import CuriosityDrive
from snn_data_lake.substrate.home import _T_GOAL, _log


class LearningProgress:
    """Per-REGION |LP| facade over an EXISTING learning-progress mechanism -- adopted, and
    the adoption MEASURED, twice, by this module's own review history.

    P2, first: the first draft carried a THIRD two-EWMA tracker; the duplicate-check
    surfaced both existing producers (`SpikingCuriosityDrive`, live in the brain, and its
    rate sibling `CuriosityDrive` -- a flag-selected coexisting pair recorded as OPEN P2
    debt in spiking_curiosity's own "P2 STATUS" docstring paragraph, with the measurement
    that supports retiring neither). The spiking one was adopted first on P1 grounds --
    and this module's OWN four-regime check then measured it INVALID as an |LP| basis: its
    pools are calibrated at one reference error (0.9), and on a FLAT stream at other
    levels the rates do not converge (|rs-rf| = +0.557 at flat 0.5; the fast pool fully
    sub-threshold), so |.| there conflates error LEVEL with progress -- and the rectified
    live reward inherits the same bias (re-derivable in one probe: observe a flat 0.5
    stream x300 and read the windowed rates; the fix is its own task, outside this one).
    The EMA drive is immune BY STRUCTURE: both EMAs converge to the same value on a flat
    stream at every level (measured: difference exactly 0.0 at e=0.1/0.5/0.9). THIS class
    therefore adopts `CuriosityDrive` with regions as context keys, adding nothing but
    validation, the |.| readout the drive gained at T046a, and the region-keyed surface.
    The task text's 'competence EWMA' is delivered by the drive's own fast/slow error EMAs
    (competence = 1 - error is a sign flip the |.| erases). Channels appear ONLY when fed:
    an unfed region raises, never a silent 0.0 -- absence and zero-progress differ."""

    def __init__(self):
        # No seed parameter ON PURPOSE (T046a panel): the adopted drive's rng feeds only
        # its selector tie-break, which NO path this facade exercises touches -- measured,
        # |LP| identical to 9 decimals across seeds. A parameter the readouts cannot feel
        # would advertise a determinism control that does not exist (the dead-sigma lesson).
        self._drive = CuriosityDrive(seed=0)  # THE mechanism, adopted whole
        self._n = {}

    def observe(self, region, error_level):
        """Feed one error observation for `region` (error in [0, 1] -- the predictor port's
        level01 unit). Non-finite or out-of-range input is REFUSED: a NaN would poison the
        pools silently, and an out-of-range 'error' is a unit bug upstream, not a datum."""
        e = float(error_level)
        if not (math.isfinite(e) and 0.0 <= e <= 1.0):
            raise ValueError(f"error_level must be finite in [0,1], got {error_level!r}")
        self._drive.observe(region, e)
        self._n[region] = self._n.get(region, 0) + 1
        return self.lp(region)

    def consume_surprise(self, surprise):
        """THE discharged T028 consumer: the brain's per-tick percept surprise (level01)
        feeds the cortex channel -- surprise IS the normalized prediction error of the one
        live predictor port. Returns the updated cortex |LP|."""
        return self.observe("cortex", surprise)

    # distinct-from: neuromodulation/intrinsic_motivation.py -- the ADOPTED drive this method
    #   READS (learning_progress_abs); nothing re-derived. Site read (this module's own dep).
    # distinct-from: cortex/placement.py -- region-keyed PRIORS; unrelated job. Site read.
    # distinct-from: neuromodulation/locus_coeruleus.py -- LC pool readout window; unrelated.
    #   Site read; the matcher fired on generic words.
    def lp(self, region):
        """The adopted drive's |slow_EMA minus fast_EMA| readout for a fed region; KeyError
        for an unfed one (absence is a fact)."""
        if region not in self._n:
            raise KeyError(f"region {region!r} has no fed error channel")
        return self._drive.learning_progress_abs(region)

    def rectified(self, region):
        """The reward-side rectified readout of the SAME rates (max(0, slow - fast)) --
        exposed so a consumer never re-derives it beside the |.| form."""
        if region not in self._n:
            raise KeyError(f"region {region!r} has no fed error channel")
        return self._drive.learning_progress(region)

    def lp_all(self):
        """{region: |LP|} over FED channels only."""
        return {r: self.lp(r) for r in self._n}

    def observations(self, region):
        return self._n.get(region, 0)


class GoalStore:
    """Goal records in THE ONE LAKE (_T_GOAL), with a derived cosine index (a CACHE).

    Records are append-only event rows sharing one goal_id: a CREATE row carries the
    embedding + achievement_function_ref; LP and OUTCOME rows append history. The
    materialized view (goals()) folds rows in replay order, so the lake is the record and
    every in-memory structure is reconstructible from it (P4b). Identity is LAKE-DERIVED
    (goal_id = count of CREATE rows already in the lake) -- the T031 coalition lesson:
    identity from lake state, never process state."""

    #: SINGLE-WRITER, stated (T046a panel: the first draft stated no writer model at all,
    #: and two concurrently-open views on one lake -- which home._log's cache makes SHARE a
    #: BinLog -- collided on goal ids and silently destroyed a goal). Exactly ONE GoalStore
    #: may write a given lake at a time; concurrent readers are fine (replay is a stream).
    #: The fold is FIRST-WINS on a duplicate id and _next is monotone, so even a violated
    #: precondition degrades to a refused fold, never an overwrite.
    def __init__(self, path=None):
        self._bin = _log(path, dedup=False)  # event rows may legitimately repeat; no dedup
        self._goals = {}
        self._index = []  # [(goal_id, unit-embedding)] -- DERIVED, rebuilt on replay
        self._next = 0  # monotone id allocator, maintained by _fold (never a count: the
        # panel measured count-based ids re-issuing an occupied id forever after one gap)
        self._replay()

    # ---- lake I/O ------------------------------------------------------------------------
    # distinct-from: substrate/binlog.py -- BinLog owns the FRAME layer (length/CRC/torn-tail
    #   recovery), REUSED via self._bin; these two methods are only the _T_GOAL BODY codec.
    #   Site read.
    # JSON body rows (sort_keys -> deterministic bytes), the _T_COALITION precedent. The
    # first draft used pack_fact and its OWN round-trip red caught the miss: pack_fact is
    # the FACT-schema codec (cue/answer/triple/src) and silently DROPS every other field --
    # a goal came back as {'cue': [], 'answer': ''}. "It persists" is a claim about a new
    # process, and only the red proved it false.
    def _append(self, row):
        self._bin.append(_T_GOAL + json.dumps(row, sort_keys=True).encode("utf-8"))

    def _replay(self):
        self._goals, self._index, self._next = {}, [], 0
        for payload in self._bin.replay():
            if payload[:1] != _T_GOAL:
                continue
            try:
                # THE WHOLE FOLD is inside the guard (T046a panel CRITICAL: the first
                # draft guarded only json.loads while claiming "never crashes the view",
                # and 8 of 12 constructed hostile-but-valid-JSON bodies bricked the
                # constructor FOREVER on an IMMORTAL-retention lake). A malformed row is
                # skipped, never fatal; a poisoned lake record must not cost the view.
                row = json.loads(payload[1:].decode("utf-8"))
                if isinstance(row, dict):
                    self._fold(row)
            except Exception:
                continue

    def _fold(self, row):
        kind = row.get("kind")
        if kind == "create":
            gid = int(row["goal_id"])
            if isinstance(row["goal_id"], bool) or gid in self._goals:
                return  # FIRST-WINS: a duplicate gid CREATE never overwrites (panel: the
                # overwrite silently destroyed a goal's history AND left a stale index row)
            emb = [float(v) for v in row["embedding"]]
            self._goals[gid] = {
                "goal_id": gid,
                "embedding": emb,
                "achievement_function_ref": str(row["achievement_function_ref"]),
                "lp_history": [],
                "outcomes": [],
            }
            self._next = max(self._next, gid + 1)  # monotone: a gap never re-issues an id
            norm = math.sqrt(sum(v * v for v in emb)) or 1.0
            self._index.append((gid, [v / norm for v in emb]))
        elif (
            kind == "lp"
            and not isinstance(row.get("goal_id"), bool)
            and int(row.get("goal_id", -1)) in self._goals
        ):
            self._goals[int(row["goal_id"])]["lp_history"].append(float(row["lp"]))
        elif (
            kind == "outcome"
            and not isinstance(row.get("goal_id"), bool)
            and int(row.get("goal_id", -1)) in self._goals
        ):
            self._goals[int(row["goal_id"])]["outcomes"].append(str(row["outcome"]))

    # ---- the API -------------------------------------------------------------------------
    def create(self, embedding, achievement_function_ref):
        emb = [float(v) for v in embedding]
        if not emb or not all(math.isfinite(v) for v in emb):
            raise ValueError("embedding must be a non-empty finite vector")
        if not any(v != 0.0 for v in emb):
            raise ValueError(
                "zero-norm embedding refused: its cosine against every query is 0, so the "
                "goal would be permanently unreachable by similarity"
            )
        gid = self._next  # monotone allocator (see _fold); never a live count
        row = {
            "kind": "create",
            "goal_id": gid,
            "embedding": emb,
            "achievement_function_ref": str(achievement_function_ref),
        }
        self._append(row)
        self._fold(row)
        return gid

    def _check_gid(self, goal_id):
        if isinstance(goal_id, bool):
            raise TypeError("goal_id must be an int, not a bool (True would target goal 1)")
        gid = int(goal_id)
        if gid not in self._goals:
            raise KeyError(f"unknown goal_id {goal_id}")
        return gid

    def record_lp(self, goal_id, lp):
        gid = self._check_gid(goal_id)
        v = float(lp)
        if not math.isfinite(v):
            # a NaN here would write the literal token 'NaN' -- NOT valid JSON -- into an
            # IMMORTAL lake record, and nan != nan breaks the store's own round-trip red
            raise ValueError(f"lp must be finite, got {lp!r}")
        row = {"kind": "lp", "goal_id": gid, "lp": v}
        self._append(row)
        self._fold(row)

    def record_outcome(self, goal_id, outcome):
        gid = self._check_gid(goal_id)
        row = {"kind": "outcome", "goal_id": gid, "outcome": str(outcome)}
        self._append(row)
        self._fold(row)

    def goals(self):
        return dict(self._goals)

    # distinct-from: reasoning/semantic_memory.py -- its top-k is VSA-hypervector cosine
    #   over FACT text (a different record class in a different space); nothing shared but
    #   the words "top-k". Site read.
    # distinct-from: cortex/learned_attention.py -- per-band bias cosine; unrelated. Read.
    # distinct-from: cortex/prefrontal_executive.py -- a goal STACK index (which goal is
    #   held), not a similarity index; unrelated job. Read.
    def query(self, embedding, k=3):
        """Cosine top-k over the DERIVED index: [(goal_id, similarity)], best first. The
        index is a cache -- the index-is-cache red drops it and rebuilds from the lake.
        k=3 is a caller-overridable default with no tuned meaning (the T028 review's k
        lesson: say so rather than let a reader assume calibration); the harness always
        passes k explicitly. Refuses a non-finite query (the panel measured a NaN query
        returning all-NaN scores in comparison-accident order, sold as 'best first') and a
        bool/negative k (scored[:-1] silently returns n-1 plausible-looking results)."""
        q = [float(v) for v in embedding]
        if not all(math.isfinite(v) for v in q):
            raise ValueError("query embedding must be finite")
        if isinstance(k, bool) or int(k) < 0:
            raise ValueError(f"k must be a non-negative int, got {k!r}")
        qn = math.sqrt(sum(v * v for v in q)) or 1.0
        q = [v / qn for v in q]
        scored = []
        for gid, u in self._index:
            if len(u) != len(q):
                continue  # dimension-mismatched goals are not comparable, not an error
            scored.append((gid, sum(a * b for a, b in zip(q, u, strict=True))))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[: int(k)]


def declare_ports(manifest):
    """Manifest registrant (T046a). IN `surprise_in` consumes the predictor port -- THE
    T028 dated disposition's discharge, the consumer declaring the edge
    `percept_surprise.surprise_out -> learning_progress.surprise_in`. OUT `lp_out` is the
    |LP| third factor, declared EDGE-LESS with its dated consumer named: T047's growth
    stopping criterion (`cortex/growth_stop.py`) -- declaring that edge today would be
    anatomy ahead of its consumer (the T030 broadcast_out precedent)."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "learning_progress",
        Port(
            name="surprise_in",
            region="basal_ganglia",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="IN",
            declaring_module="learning_progress",
            # typed-merge, NOT sum (panel B6): contributions are keyed BY REGION on the
            # consumer side (observe(region, error) -- separate per-region channels), and
            # two producers SUMMED into one [0,1] scalar would exceed observe()'s own
            # domain guard by construction. The sibling brain.salience_surprise_in stays
            # "sum" because its consumer genuinely accumulates one salience stream.
            join_policy="typed-merge",
        ),
    )
    manifest.declare_port(
        "learning_progress",
        Port(
            name="lp_out",
            region="basal_ganglia",
            shape=(1,),
            dtype="f4",
            unit="level01",
            encoding="SCALAR_DENSE",
            kind="OUT",
            declaring_module="learning_progress",
            delivery_policy="all-or-none",
        ),
    )
    manifest.declare_edge("percept_surprise", "surprise_out", "learning_progress", "surprise_in")


def main():
    import tempfile
    from pathlib import Path

    print("=" * 96)
    print("LEARNING PROGRESS + GOAL STORE -- |LP| third factor + lake-resident goals (T046a)")
    print("=" * 96)

    # 1. |LP| FOUR REGIMES (the level-channel rule), through THE adopted mechanism:
    # flat EXACTLY 0; improving >0; worsening >0 with the DISSOCIATION that motivates the
    # new readout; trendless noise small. Statistic choice, honest: the MOVING regimes
    # report their PEAK |LP| during the trend (the first draft read the TERMINAL value and
    # the improving stream's 214-tick floor plateau had -- correctly -- decayed it to
    # 0.0000: mastered IS flat, the design working); the STEADY regimes report terminal.
    # The worsening ramp uses a DIFFERENT slope on purpose: with mirror-image slopes the
    # two moving peaks agree to 1 ULP (|slow-fast| is slope-sign-symmetric), so equal
    # slopes deliver one measurement dressed as two (panel).
    lp = LearningProgress()
    for _ in range(300):
        lp.observe("flat", 0.5)
    flat = lp.lp("flat")
    improving = 0.0
    rect_improving_peak = 0.0
    for i in range(300):
        improving = max(improving, lp.observe("improving", max(0.05, 0.9 - i * 0.01)))
        rect_improving_peak = max(rect_improving_peak, lp.rectified("improving"))
    worsening = 0.0
    rect_at_worst_peak = 0.0
    for i in range(300):
        w = lp.observe("worsening", min(0.95, 0.1 + i * 0.008))
        if w > worsening:
            worsening = w
            rect_at_worst_peak = lp.rectified("worsening")
    import random as _random

    # noise reported as the MAX over 10 seeds, not one draw: the panel swept 40 seeds and
    # measured seed 7 a favourable draw (0.0088 vs a 0.0754 worst case -- an 11x margin
    # advertised where the swept margin is 1.3x). The bar must clear the worst draw.
    noise = 0.0
    for s in range(10):
        rng = _random.Random(s)
        key = f"noise{s}"
        for _ in range(300):
            lp.observe(key, 0.3 + 0.4 * rng.random())
        noise = max(noise, lp.lp(key))
    assert improving > 0.0 and worsening > 0.0, (
        f"a moving competence read no |LP| (improving {improving:.4f}, worsening {worsening:.4f})"
    )
    # EXACTLY zero, not a ratio bar: the exact zero IS the structural property the adoption
    # argument rests on (the panel measured the first draft's 0.25x ratio inert -- 0.0 < x
    # is true for any positive x, a margin advertised but never tested).
    assert flat == 0.0, f"flat error did not read exactly 0 ({flat!r})"
    assert noise < 0.5 * min(improving, worsening), (
        f"trendless noise (worst of 10 seeds) read as progress ({noise:.4f} vs "
        f"{min(improving, worsening):.4f})"
    )
    # THE DISSOCIATION that motivates the |.| readout: at the WORSENING peak the rectified
    # form reads exactly 0 (error rising -> fast pool above slow -> max(0, rs-rf) clamps),
    # while during IMPROVEMENT both readouts move together.
    assert rect_at_worst_peak == 0.0 and worsening > 0.0, (
        f"the DISSOCIATION failed: rectified {rect_at_worst_peak} vs |.| {worsening:.4f} "
        "at the regression peak -- the two readouts are not separating"
    )
    assert rect_improving_peak > 0.0, "the rectified readout missed the improvement too"
    print(
        f"  green  |LP| four regimes (error EMAs): flat {flat:.4f}, improving peak "
        f"{improving:.4f}, worsening peak {worsening:.4f} (rectified reads "
        f"{rect_at_worst_peak} there -- the dissociation IS the new readout's reason), "
        f"noise {noise:.4f}"
    )

    # Absence is a fact: an unfed region raises, never a silent 0.0.
    try:
        lp.lp("cerebellum")
        raise AssertionError("an unfed region read a value")
    except KeyError:
        pass
    for bad in (float("nan"), -0.1, 1.5):
        try:
            lp.observe("cortex", bad)
            raise AssertionError(f"bad error_level {bad!r} accepted")
        except ValueError:
            pass
    print("  RED    unfed region raises (absence != zero); NaN / out-of-range refused")

    # 2. THE DISCHARGED CONSUMER -- scoped to what it MEASURES (the panel deleted the
    # "learning trend" claim per P8): this arm proves the WIRE carries the stream into the
    # drive's state (a moving stream reads a positive |LP| PEAK; a pinned stream reads
    # EXACTLY 0 -- the structural control, asserted as an equality, because the first
    # draft's 10x-ratio bar degenerated to >1e-11). WHAT |LP| CANNOT DO, measured and
    # stated rather than implied away: it does not distinguish a monotone trend from a
    # matched-marginal SHUFFLE of the same values (the panel measured the shuffled stream
    # scoring HIGHER on both terminal and peak -- |slow-fast| detects displacement between
    # timescales, which both produce; the noise regime's low reading reflects its narrower
    # marginal, not trend detection). Trend-vs-shuffle discrimination is the rectified
    # readout's and T047's gate design's burden, not this wire's.
    lp_live, lp_ctrl = LearningProgress(), LearningProgress()
    live_peak = 0.0
    for i in range(300):
        live_peak = max(live_peak, lp_live.consume_surprise(max(0.05, 0.8 - i * 0.005)))
        lp_ctrl.consume_surprise(0.0)  # pinned: a dead predictor port
    assert lp_ctrl.lp("cortex") == 0.0, "the pinned control moved -- harness nondeterminism"
    assert live_peak > 0.05, (
        f"the wire did not carry the moving stream into the drive (peak {live_peak:.4f})"
    )
    print(
        f"  green  T028 disposition discharged (the WIRE carries the stream): moving-stream "
        f"|LP| peak {live_peak:.4f} vs pinned control exactly {lp_ctrl.lp('cortex')} -- "
        "trend-vs-shuffle discrimination is NOT claimed (measured impossible for |.|)"
    )

    # 3. GOAL STORE on a SCRATCH lake: create/lp/outcome; the round-trip red (SYNTHESIS
    # 3.5: records of this tag survive close + fresh-view replay, dict-equal); the
    # index-is-cache red (drop the index, rebuild from the lake, identical answers);
    # lake-derived identity across the reopen.
    with tempfile.TemporaryDirectory() as td:
        lake = str(Path(td) / "scratch_goal_lake.bin")
        gs = GoalStore(path=lake)
        g0 = gs.create([1.0, 0.0, 0.0, 0.0], "snn_data_lake.cortex.brain.predict_accuracy")
        g1 = gs.create([0.0, 1.0, 0.0, 0.0], "snn_data_lake.cortex.brain.reasoning_accuracy")
        gs.record_lp(g0, 0.12)
        gs.record_lp(g0, 0.31)
        gs.record_outcome(g0, "kept")
        gs.record_outcome(g1, "rolled-back")
        before = gs.goals()
        q_before = gs.query([0.9, 0.1, 0.0, 0.0], k=2)
        del gs
        # THE PROCESS BOUNDARY, made real (panel: home._log memoizes by (path, dedup), so
        # `del gs` alone left `fresh` sharing the SAME BinLog object -- the red could not
        # distinguish disk from a live in-memory instance, the exact distinction P4b's
        # "after the writer is gone" guard exists for). Clearing the cache forces a genuine
        # from-disk reopen; a subprocess replay was also verified once by the panel.
        from snn_data_lake.substrate import home as _home

        _home._LOG_CACHE.clear()
        fresh = GoalStore(path=lake)  # a genuine from-disk view: replay only
        assert fresh.goals() == before, "round-trip FAILED: the lake did not carry the goals"
        assert fresh.query([0.9, 0.1, 0.0, 0.0], k=2) == q_before, (
            "the rebuilt index answers differently -- the index is not a pure cache"
        )
        g2 = fresh.create([0.0, 0.0, 1.0, 0.0], "ref.later")
        assert g2 == 2, f"lake-derived identity broke across reopen (got {g2})"
        try:
            fresh.record_lp(99, 0.5)
            raise AssertionError("an unknown goal_id was accepted")
        except KeyError:
            pass
        assert fresh.query([1.0, 0.0], k=2) == [], "a dim-mismatched query matched something"
        del fresh
    print(
        "  RED    goal store: round-trip over live-written _T_GOAL records (v1 READ-AS-IS "
        "disposition), index-is-cache rebuild identical, lake-derived ids, refusals"
    )

    print()
    print(
        "DISCLOSURES: cortex is the ONE fed |LP| channel today (the T028 predictor port; "
        "other regions' error channels land with their producers). lp_out is declared "
        "EDGE-LESS; T047's growth stopping criterion is the dated consumer. "
        "achievement_function_ref is STORED, never resolved or executed here -- opening "
        "that dispatch surface is T047+'s reviewed step, not a storage side effect. "
        "GoalStore has NO live writer today: its callers are this main() only (both "
        "dormancy gates are structurally blind to a class-level sink -- per-module and "
        "learn-verb scoped), goals are dated S1+ in data-model, and NO task id in specs/022 "
        "tasks.md names the wiring yet -- said plainly, never dressed as dated. LIVE "
        "OPERATING MAGNITUDE: the real percept-surprise channel is zero-inflated (panel: "
        "~51% exact zeros over 400 production percepts; live |LP| ~0.0035 vs this "
        "harness's validated moving scale ~0.19, a ~55x gap) -- T047's thresholds must be "
        "calibrated on the LIVE channel, never on this harness's ramps."
    )
    print("self-check OK: |LP| four regimes + discharged consumer A/B + goal-store reds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
