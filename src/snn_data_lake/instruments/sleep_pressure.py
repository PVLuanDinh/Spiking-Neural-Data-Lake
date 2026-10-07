r"""G6 / T114 part (c): learning-load sleep pressure as a second INPUT to the one scheduler -- the HARNESS
that tests it (never a second scheduler).

Pre-registration: specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md, frozen at P0 (df19f61e):
section 2.5 (pressure, budget matching by rescheduling, the arms, the workload, the endpoint), 2.11 (C11,
C12) and 2.16's A6 (the shuffled arm obeys the rescheduling rule) and A7 (per-circuit load, cold start,
the median, C23, C24).

PRESSURE (A7). L_c = sum over WAKE ticks since the last `_enter_wake` of |delta w| over plastic circuit c's
synapses, divided by its synapse count (sleep-tick plasticity excluded). B_c = c's own running mean of L_c
per completed wake period (THE one running-baseline rule, `GlobalWorkspace.next_baseline`, imported). A
circuit is ELIGIBLE after K_MIN = 2 completed wake periods since its admission with B_c > 0; until then it
contributes nothing (never 0/0, never x/0). Pressure = the MEDIAN over eligible circuits of L_c / B_c; sleep
is requested when it exceeds 1. No eligible circuit -> no request ever; the clock alone schedules. The
carrier is `sleep_pressure.json`'s dict (`PressureMeter.state()`).

THE HARNESS (2.5). A run is a fixed number of clock cycles. The clock's own gate sequence (read off a COPY
of the brain's clock, which nothing here perturbs) gives each cycle's scheduled onset and night length. A
request during a cycle's wake ADVANCES that cycle's onset to the request tick; the advanced night REPLACES
the scheduled one and lasts that cycle's own night length; a request never adds or removes a night, and a
second request in a cycle, or one while asleep, is absorbed. So every arm has identical total sleep ticks
AND onset count (C12 refuses a run whose budgets differ). The brain is driven by its OWN `Brain.tick` with
`clock.consolidation_gate` answered from the plan for the run's duration (the gate's only tick caller), so
no tick logic is replicated here; the live input to `CircadianClock` lands only after the deciding test
(P4, C13: harness schedule == live gate on a fixed pressure trace).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.sleep_pressure   (self-check, brain-free)
"""

import contextlib
import copy
import random
import statistics
from dataclasses import dataclass, field

from snn_data_lake.faculties.global_workspace import GlobalWorkspace
from snn_data_lake.instruments.paired_contrast import contrast
from snn_data_lake.instruments.sleep_pruner_census import N_SEEDS

# distinct-from: faculties/global_workspace.py::next_baseline -- REUSED here (imported), not re-spelled.
#   distinct-from: engine/brain.py::Brain.tick -- REUSED: the harness calls it with the gate answered from
#   the plan; nothing of the tick is replicated. distinct-from: sleep/hippocampal_memory.py::consolidate --
#   a wake->sleep consolidation pass over digits. distinct-from: substrate/signal_calibrator.py -- a persisted
#   running QUANTILE of one live signal, not a per-circuit load ratio. distinct-from:
#   faculties/spiking_grid_can.py::_run_circuits -- path-integration circuits on a grid sheet. Sites read.

K_MIN = (
    2  # A7: completed wake periods before a circuit's baseline is valid (disclosed session choice)
)
ARMS = ("clock", "pressure", "shuffled", "solve", "oracle")
_next_baseline = GlobalWorkspace.next_baseline


# ================================================================================================
# the pressure meter (A7)
# ================================================================================================
class PressureMeter:
    """Per-circuit wake load relative to the circuit's own persisted baseline. `circuits` maps a name to a
    zero-argument callable returning the circuit's current flat weight list (read, never written)."""

    def __init__(self, circuits=None, state=None, tick=0):
        self.circuits = dict(circuits or {})
        self.c = {}  # name -> {"base", "n", "admitted", "load", "prev"}
        self.tick = int(tick)  # the cursor is not an A7 carrier field: the restorer supplies it
        for name, row in (state or {}).items():
            self.c[name] = {
                "base": float(row["baseline_load_per_wake"]),
                "n": int(row["n_wake_periods"]),
                "admitted": int(row["admission_tick"]),
                "load": float(row["load_since_last_wake_onset"]),
                "prev": None,
            }
        for name in (
            self.circuits
        ):  # a snapshot at construction/restore: the first wake tick's learning counts
            self._snapshot(name)

    def _snapshot(self, name):
        row = self._row(name)
        row["prev"] = [float(x) for x in self.circuits[name]()]

    def admit(self, name, read):
        """A circuit admitted mid-run (a born population) starts its own count here (A7)."""
        self.circuits[name] = read
        self.c[name] = {"base": 0.0, "n": 0, "admitted": self.tick, "load": 0.0, "prev": None}
        self._snapshot(name)

    def _row(self, name):
        if name not in self.c:
            self.c[name] = {"base": 0.0, "n": 0, "admitted": self.tick, "load": 0.0, "prev": None}
        return self.c[name]

    def observe_wake(self):
        """Call once per WAKE tick, after the tick's learning."""
        for name, read in self.circuits.items():
            row = self._row(name)
            w = [float(x) for x in read()]
            if row["prev"] is not None and len(row["prev"]) == len(w) and w:
                row["load"] += sum(abs(a - b) for a, b in zip(w, row["prev"])) / len(w)
            row["prev"] = w
        self.tick += 1

    def observe_sleep(self):
        """Sleep ticks move weights but never count as load; each re-snapshots the circuits, so the next wake
        diff starts from the post-sleep weights and the first wake tick's own learning is counted."""
        for name in self.circuits:
            self._snapshot(name)
        self.tick += 1

    def wake_period_done(self):
        """At sleep onset: fold the finished wake period's load into each circuit's running mean."""
        for row in self.c.values():
            row["n"] += 1
            row["base"] = _next_baseline(row["base"], row["load"], row["n"])
            row["load"] = 0.0

    def eligible(self):
        return sorted(k for k, r in self.c.items() if r["n"] >= K_MIN and r["base"] > 0.0)

    def pressure(self):
        ratios = [self.c[k]["load"] / self.c[k]["base"] for k in self.eligible()]
        return statistics.median(ratios) if ratios else None

    def request(self):
        p = self.pressure()
        return p is not None and p > 1.0

    def state(self):
        """The `sleep_pressure.json` carrier (A7's fields)."""
        return {
            k: {
                "baseline_load_per_wake": r["base"],
                "n_wake_periods": r["n"],
                "admission_tick": r["admitted"],
                "load_since_last_wake_onset": r["load"],
            }
            for k, r in sorted(self.c.items())
        }


# ================================================================================================
# the clock's cycles and budget-matched rescheduling (2.5, A6, C12)
# ================================================================================================
@dataclass
class Cycle:
    wake_start: int
    onset: int  # scheduled sleep onset
    night: int  # this cycle's own night length (ticks)


def cycles_from_gate(gate):
    """Split a clock gate sequence (True = asleep) into cycles: each = one wake run then one night. A leading
    night (a run that starts asleep) is cycle 0 with an empty wake; a trailing wake with no night is not a
    cycle (its ticks stay awake in every arm)."""
    out, t, n = [], 0, len(gate)
    while t < n:
        ws = t
        while t < n and not gate[t]:
            t += 1
        on = t
        while t < n and gate[t]:
            t += 1
        if t == on:
            break
        out.append(Cycle(ws, on, t - on))
    return out


class Rescheduler:
    """The ONLINE plan: tick -> asleep under the clock's cycles, requests advancing onsets (2.5)."""

    def __init__(self, cycles, total):
        self.cy = [Cycle(c.wake_start, c.onset, c.night) for c in cycles]
        self.total = total
        self.advanced = {}  # cycle index -> offset (scheduled onset - request tick)
        self.absorbed = 0

    def _cycle_at(self, t):
        for i, c in enumerate(self.cy):
            nxt = self.cy[i + 1].wake_start if i + 1 < len(self.cy) else self.total
            if c.wake_start <= t < nxt:
                return i
        return None

    def asleep(self, t):
        i = self._cycle_at(t)
        if i is None:
            return False
        c = self.cy[i]
        return c.onset <= t < c.onset + c.night

    def request(self, t):
        """A request at wake tick t advances its cycle's onset to t, at most once per cycle."""
        i = self._cycle_at(t)
        if i is None or self.asleep(t) or i in self.advanced or t >= self.cy[i].onset:
            self.absorbed += 1
            return False
        self.advanced[i] = self.cy[i].onset - t
        self.cy[i].onset = t
        return True

    def plan(self):
        return [self.asleep(t) for t in range(self.total)]


def shuffled_plan(cycles, total, advanced, seed):
    """A6: the pressure arm's k advanced cycles and k offsets; choose k of the N cycles uniformly without
    replacement (own RNG stream, same seed) and give them the offsets in uniformly permuted order. An offset
    longer than its target cycle's wake is REFUSED (never clamped)."""
    rng = random.Random(f"g6.shuffle:{seed}")
    offsets = [advanced[i] for i in sorted(advanced)]
    eligible = [i for i, c in enumerate(cycles) if c.onset > c.wake_start]
    picks = rng.sample(eligible, len(offsets))
    rng.shuffle(offsets)
    r = Rescheduler(cycles, total)
    for i, off in zip(sorted(picks), offsets):
        c = r.cy[i]
        if off > c.onset - c.wake_start:
            raise ValueError(
                f"A6: offset {off} exceeds cycle {i}'s wake ({c.onset - c.wake_start})"
            )
        r.advanced[i] = off
        c.onset -= off
    return r.plan()


def budget(plan):
    """(total sleep ticks, onset count) of a plan."""
    onsets = sum(1 for t, s in enumerate(plan) if s and (t == 0 or not plan[t - 1]))
    return sum(1 for s in plan if s), onsets


class BudgetMismatch(Exception):
    """An arm's (sleep ticks, onsets) differs from the clock arm's: the run is refused (C12)."""


def check_budgets(plans):
    """C12: every arm's plan must carry the clock arm's (sleep ticks, onsets); a mismatch REFUSES the run."""
    ref = budget(plans["clock"])
    bad = {a: budget(p) for a, p in plans.items() if budget(p) != ref}
    return (not bad), ref, bad


def _beats(a, b, seeds):
    c = contrast(f"{a} > {b}", [s[a] - s[b] for s in seeds])
    return c._replace(met=False) if c.degenerate else c  # a degenerate superiority is NOT MET (A9)


def decide_c(seeds):
    """`seeds`: per-seed {arm: held-out retention rate} for clock, oracle, pressure, shuffled (solve optional).
    C11 first: if the schedule ORACLE does not beat clock-only the workload cannot show a timing benefit and (c)
    is REFUSED. Then the primary: pressure > shuffled-pressure (pressure-SPECIFIC); pressure > clock is reported."""
    if len(seeds) != N_SEEDS:
        return "REFUSED", f"{len(seeds)} seeds read; the bar is the S = {N_SEEDS} one", {}
    need = ("clock", "oracle", "pressure", "shuffled")
    if any(a not in s for s in seeds for a in need):
        return "REFUSED", "an arm is missing on some seed", {}
    con = {
        k: _beats(*k.split(">"), seeds)
        for k in ("oracle>clock", "pressure>shuffled", "pressure>clock")
    }
    if "solve" in seeds[0]:
        con.update({f"solve>{b}": _beats("solve", b, seeds) for b in ("clock", "pressure")})
    if not con["oracle>clock"].met:
        return (
            "REFUSED",
            "C11: the oracle does not beat clock-only; the workload shows no timing benefit",
            con,
        )
    if con["pressure>shuffled"].met:
        return (
            "PRESSURE-SPECIFIC",
            "pressure beats its shuffled twin; the live input may wire after C13",
            con,
        )
    return (
        "NULL",
        "pressure does not beat its shuffled twin; the input is deleted on this null",
        con,
    )


# ================================================================================================
# driving a brain: its own Brain.tick, the gate answered from the plan
# ================================================================================================
def clock_gate(brain, total):
    """The clock-only gate for the next `total` ticks, read off a COPY of the brain's clock and light schedule
    (the copy's RNG draws are the same ones the live clock will make, so the gate is the clock arm's)."""
    clk = copy.deepcopy(brain.clock)
    out = []
    for t in range(brain._step, brain._step + total):
        clk.step(brain._schedule(t))
        out.append(bool(clk.consolidation_gate()))
    return out


@contextlib.contextmanager
def gate_from(brain, decide):
    """For the block's duration `brain.clock.consolidation_gate()` answers `decide()` (Brain.tick's only
    gate call); restored exactly after."""
    had = "consolidation_gate" in vars(brain.clock)
    old = vars(brain.clock).get("consolidation_gate")
    brain.clock.consolidation_gate = lambda phase=None: bool(decide())
    try:
        yield
    finally:
        if had:
            brain.clock.consolidation_gate = old
        else:
            del brain.clock.consolidation_gate


def _flat(w):
    if w is None:
        return []
    if hasattr(w, "detach"):  # a torch tensor
        return w.detach().flatten().tolist()
    return [x for row in w for x in row] if w and isinstance(w[0], (list, tuple)) else list(w)


def attractor_circuit(brain):
    """`AttractorNet.W` (2.4's legacy list, third member)."""
    return lambda: _flat(brain.attractor.W)


def grower_circuit(brain):
    """The NG-SNN grower's input weights (second member), read through the engine each call: None until the
    grower has a model, so it is ADMITTED mid-run (A7), never assumed present."""
    return lambda: _flat(getattr(getattr(brain.neuro, "_model", None), "W", None))


def grower_ready(brain):
    return getattr(getattr(brain.neuro, "_model", None), "W", None) is not None


def ixn_circuit(brain):
    """The interaction learner's rows as one plastic circuit (2.4's legacy list, first member)."""
    return lambda: [w for n in brain.brain._ixn.neurons for w in n.w]


# ================================================================================================
# the workload (2.5): uneven load -- planted bursts, then conflicting material -- through the
# interaction learner; every arm receives the SAME teaching in the SAME order (a queue on wake ticks),
# only its placement relative to sleep differs. Constants are disclosed session choices.
# ================================================================================================
BURST_EVERY = 4  # a burst in every 4th clock cycle, from cycle 2
BURST_ITEMS = 4  # new items per burst
BURST_REPS = 3  # each taught 3 times at the burst tick
CONFLICT_PER_TICK = 2  # conflicting items per later wake tick, queued
HORIZON = 50  # ticks after a burst its retention is read (about two clock cycles)


class BurstWorkload:
    """`workload(brain, t)` for `run_pressure_arm`. Teaches with `interact` (the ring is fed, as TEACH does)
    and reads retention with `brain.brain.route` (pure inference: learning OFF, no event, no RNG)."""

    def __init__(self, seed, cycles, labels, total):
        rng = random.Random(f"g6.burst:{seed}")
        vocab = [f"w{rng.randrange(10**6):06d}" for _ in range(400)]
        self.items = {}  # burst tick -> [(text, label)]: what the burst taught (the retention set)
        self.conflict = {}  # burst tick -> the same texts with one word swapped and a different label
        for ci in range(2, len(cycles), BURST_EVERY):
            t0 = cycles[ci].wake_start
            items = [
                (" ".join(rng.sample(vocab, 3)), rng.choice(labels)) for _ in range(BURST_ITEMS)
            ]
            self.items[t0] = items
            out = []
            for text, lab in items:
                words = text.split()
                words[rng.randrange(3)] = rng.choice(vocab)
                out.append((" ".join(words), rng.choice([x for x in labels if x != lab])))
            self.conflict[t0] = out
        self.order = sorted(self.items)
        self.taught = set()
        self.queue = []  # conflicting items waiting for wake ticks, delivered from burst tick + 2 on
        self.reads = []  # (due tick, burst tick)
        self.final_at = (
            total - 26
        )  # the last clock cycle's wake: the end-of-run read of every burst
        self.final_done = False

    def ends(self):
        """The oracle's request ticks: the tick right after each planted burst."""
        return [t0 + 1 for t0 in self.order]

    @staticmethod
    def _read(brain, items):
        return sum(1 for text, lab in items if brain.brain.route(text) == lab), len(items)

    def __call__(self, brain, t):
        from snn_data_lake.readout.interaction_learning import text_features

        ixn = brain.brain._ixn
        for t0 in self.order:
            if t0 <= t and t0 not in self.taught:  # a burst is taught at its first wake tick >= t0
                self.taught.add(t0)
                for _ in range(BURST_REPS):
                    for text, lab in self.items[t0]:
                        ixn.interact(text_features(text, ixn.n_in), lab)
                self.queue.extend((t0 + 2, it) for it in self.conflict[t0])
                self.reads.append((t0 + HORIZON, t0))
        ready = [it for due, it in self.queue if due <= t][:CONFLICT_PER_TICK]
        for text, lab in ready:
            ixn.interact(text_features(text, ixn.n_in), lab)
        for it in ready:
            self.queue = [q for q in self.queue if q[1] is not it]
        c = n = 0
        due = [p for p in self.reads if p[0] <= t]
        if due:
            got = self._read(brain, [it for _, t0 in due for it in self.items[t0]])
            c, n = c + got[0], n + got[1]
            self.reads = [p for p in self.reads if p[0] > t]
        if not self.final_done and t >= self.final_at:
            self.final_done = True
            got = self._read(brain, [it for t0 in sorted(self.taught) for it in self.items[t0]])
            c, n = c + got[0], n + got[1]
        return (c, n) if n else None


@dataclass
class PressureRun:
    seed: int
    arm: str
    plan: list = field(default_factory=list)
    advanced: dict = field(default_factory=dict)
    absorbed: int = 0
    retention: list = field(
        default_factory=list
    )  # [tick, correct, total] at each registered horizon
    carrier: dict = field(default_factory=dict)


def run_pressure_arm(
    brain, arm, total, workload, *, requests=None, seed=0, shuffle_from=None, extra_circuits=None
):
    """Drive `brain` for `total` ticks under `arm`. `workload(brain, t)` teaches on wake ticks and returns a
    learning-OFF held-out read [correct, total] at its registered horizons (None otherwise). `requests` =
    the request ticks of the 'oracle' / 'solve' arms; `shuffle_from` = the pressure arm's `advanced`."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms are {ARMS}")
    if arm == "solve" and requests is None:
        raise ValueError(
            "the solve arm needs the solved_sink request ticks (G3's loop); it is never the oracle"
        )
    clock_plan = clock_gate(brain, total)
    cycles = cycles_from_gate(clock_plan)
    run = PressureRun(seed=seed, arm=arm)
    sched = None if arm == "shuffled" else Rescheduler(cycles, total)
    fixed = shuffled_plan(cycles, total, shuffle_from or {}, seed) if arm == "shuffled" else None
    meter = None
    if (
        arm == "pressure"
    ):  # 2.4's legacy list: the ixn rows and AttractorNet.W now, the grower once it exists
        meter = PressureMeter(
            {
                "ixn": ixn_circuit(brain),
                "attractor": attractor_circuit(brain),
                **(extra_circuits or {}),
            }
        )
    asked = set(requests or ())
    t_now = [0]

    def decide():
        t = t_now[0]
        if sched is None:
            return fixed[t]
        if arm in ("oracle", "solve") and t in asked:
            sched.request(t)
        elif meter is not None and not sched.asleep(t) and meter.request():
            sched.request(t)
        return sched.asleep(t)

    with gate_from(brain, decide):
        for t in range(total):
            t_now[0] = t
            was = brain._was_asleep
            mode = brain.tick()
            asleep = mode == "asleep"
            if not asleep:
                got = workload(brain, t)  # the tick's teaching happens BEFORE the meter reads (A7)
                if got is not None:
                    run.retention.append([t, int(got[0]), int(got[1])])
            if meter is not None:
                if asleep and not was:
                    meter.wake_period_done()
                if asleep:
                    meter.observe_sleep()
                else:
                    if "grower" not in meter.circuits and grower_ready(brain):
                        meter.admit("grower", grower_circuit(brain))
                    meter.observe_wake()
            run.plan.append(asleep)
    ok, ref, bad = check_budgets(
        {"clock": clock_plan, arm: run.plan}
    )  # C12 at run time, not only in main()
    if not ok:
        raise BudgetMismatch(f"arm {arm!r} budget {bad[arm]} != the clock arm's {ref}")
    if sched is not None:
        run.advanced, run.absorbed = dict(sched.advanced), sched.absorbed
    run.carrier = meter.state() if meter is not None else {}
    return run


def run_seed_arm(seed, arm, *, total=900, shuffle_from=None, solve_requests=None):
    """One (seed, arm) on a fresh scratch brain with the burst workload; the caller owns a throwaway
    NORD_HOME (refused if it resolves to the live home). Returns (PressureRun, pooled [correct, total])."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.instruments.sleep_pruner_census import admit_seed
    from snn_data_lake.substrate.home import require_isolated_home

    admit_seed(seed)
    require_isolated_home()
    b = make_brain(seed=seed)
    cycles = cycles_from_gate(clock_gate(b, total))
    wl = BurstWorkload(seed, cycles, list(b.brain._ixn.responses), total)
    # the oracle's requests are the planted burst ends; the solve arm's are G3's solved_sink events, never the oracle's
    req = wl.ends() if arm == "oracle" else solve_requests if arm == "solve" else None
    run = run_pressure_arm(b, arm, total, wl, requests=req, seed=seed, shuffle_from=shuffle_from)
    return run, [sum(r[1] for r in run.retention), sum(r[2] for r in run.retention)]


# ================================================================================================
# self-check (brain-free)
# ================================================================================================
def main():
    gate = ([False] * 6 + [True] * 4) * 5  # 5 cycles: 6 wake, 4 night
    cyc = cycles_from_gate(gate)
    assert [(c.wake_start, c.onset, c.night) for c in cyc][:2] == [(0, 6, 4), (10, 16, 4)]
    r = Rescheduler(cyc, len(gate))
    assert (
        r.request(13) and not r.request(14) and not r.request(17)
    )  # once per cycle; asleep absorbed
    plan = r.plan()
    assert budget(plan) == budget(gate) == (20, 5), budget(plan)
    assert plan[13:17] == [True] * 4 and plan[17:20] == [False] * 3

    ok, ref, _ = check_budgets({"clock": gate, "pressure": plan})
    assert ok and ref == (20, 5)
    extra = list(gate)
    extra[5] = True  # C12's planted red: one extra sleep tick
    ok, _, bad = check_budgets({"clock": gate, "planted": extra})
    assert not ok and "planted" in bad

    base = [
        {
            "clock": 0.50 + 0.01 * i,
            "oracle": 0.60 + 0.01 * i,
            "pressure": 0.58 + 0.01 * i,
            "shuffled": 0.50 + 0.01 * i,
        }
        for i in range(8)
    ]
    jit = [
        0.001 * ((i * 5) % 7 - 3) for i in range(8)
    ]  # nonzero spread so no contrast is degenerate
    base = [{a: v + j for a, v in s_.items()} for s_, j in zip(base, jit)]
    assert decide_c(base)[0] == "PRESSURE-SPECIFIC"
    flat_oracle = [dict(s_, oracle=s_["clock"] + j) for s_, j in zip(base, jit)]
    assert decide_c(flat_oracle)[0] == "REFUSED", (
        "C11 red: an oracle no better than clock refuses (c)"
    )
    twin = [dict(s_, shuffled=s_["pressure"] - j) for s_, j in zip(base, jit)]
    assert decide_c(twin)[0] == "NULL", "a shuffled twin as good as pressure is a null"
    sh = shuffled_plan(cyc, len(gate), r.advanced, seed=1)
    assert budget(sh) == budget(gate), "A6: the shuffle keeps the budget by construction"
    try:
        shuffled_plan(cyc, len(gate), {0: 7}, seed=1)
        raise AssertionError("A6: an offset longer than a wake must refuse")
    except ValueError:
        pass

    w = {"a": [0.0, 0.0], "z": [1.0, 1.0]}
    m = PressureMeter({"a": lambda: w["a"], "z": lambda: w["z"]})
    for _period in range(K_MIN):
        assert m.pressure() is None, "A7: no pressure before K_MIN completed wake periods"
        for _step in range(3):
            w["a"] = [w["a"][0] + 0.1, w["a"][1]]
            m.observe_wake()
        m.wake_period_done()
    assert m.eligible() == ["a"], m.eligible()  # 'z' never moved: B = 0, never eligible (C23)
    big = {"b": [0.0]}
    m.admit("b", lambda: big["b"])
    m.observe_wake()
    big["b"] = [1e6]
    m.observe_wake()
    assert "b" not in m.eligible() and m.pressure() < 1e3, (
        "C24: an admitted circuit is silent before K_MIN"
    )
    s = m.state()
    m2 = PressureMeter(state=s)
    assert m2.state() == s and m2.eligible() == m.eligible(), "the carrier round-trips"
    print(
        "sleep_pressure self-check OK: rescheduling keeps (sleep ticks, onsets) = (20, 5); C12 refuses a "
        "planted extra tick; A6 shuffle budget-equal and refuses an over-long offset; A7 K_MIN cold start; "
        "C23 a zero-baseline circuit never eligible; C24 an admitted circuit silent before K_MIN; carrier "
        "round-trips"
    )


if __name__ == "__main__":
    main()
