"""
region_wiring -- the always-on daemon's FACULTY-WIRING REGISTRY, and the project's PARALLEL GATE.

The daemon runs Brain.tick() (which transitively charges many faculties on the wake/sleep path) but some
faculties are DORMANT (never reached) or PARTIAL (a signal computed then ignored -- an attention mask that
masks nothing, a plan with no reward, a recall that is write-only). This registry is the explicit plug
point: each wired faculty registers a `charge(brain, ctx)` adapter the daemon runs EVERY cycle, so a dormant
faculty becomes live WITHOUT editing the daemon loop.

FACULTY-LEVEL PARALLELISM (added 2026-08-05, free-threaded CPython 3.14). An SNN is event-driven and its
neuron states are independent WITHIN a timestep, so faculties that do not read each other on the same tick
can compute concurrently. Which faculties those are is **DERIVED, never guessed** --
`same_tick_dependency_map()` reads `connectome.EDGES` and reports the regions coupled within one tick.
Measured on this tree: **81 edges, 0 of them delay-0**, so the same-tick region-dependency graph is EMPTY
and every region is trivially independent per tick (the same fact `tools/p6_gate.py` reports as 0
faculty-to-faculty reads -- a P6 defect, and simultaneously the property that makes this safe today).
The derivation is re-run on every call, so the day a delay-0 edge is added the coupled regions drop back to
serial automatically instead of silently racing.

Two things are NEVER parallelised, and both are derived from the connectome rather than named by hand:
the region owning `connectome.GATE_NODE` and any region with a direct edge to `connectome.ACTUATOR`.
That resolves to `brainstem` -- the arm, the safety envelope and MuJoCo `mjData`, which
`substrate/learn_daemon.py` already documents as single-threaded (mj_forward racing mj_step on one mjData
is undefined behaviour). The arm is physically sequential; this module must never make it look otherwise.

THE GATE IS MEASURED, NOT HARDCODED (P1). `pick_thread_count(serial_estimate_s)` compares the caller's
own running-EMA estimate of its serial cost against the **measured** dispatch overhead of this machine
(`parallel_overhead_s`, calibrated lazily on first use, cached per thread count) and returns 1 -- run
serial -- unless a real net win survives. No width constant, no thread constant, appears in the decision.
`thalamus/global_workspace.py` and `thalamus/connectome_router.py` gate their own parallel kernels
through this same function.

STATUS, STATED PLAINLY: **the registry is EMPTY in production and always has been**, so `charge_all()`
still charges nothing and its parallel path is SCAFFOLDING (permitted by P0's honesty guard: build the
channel, fill it, measure it, credit it only at the third step). The mechanism is measured against
SYNTHETIC faculties on a 28-core free-threaded 3.14.6 box -- 8 x 5 us of work 1.09x (the gate keeps it
serial-ish), 8 x 50 us 2.02x, 32 x 50 us 5.22x, 128 x 50 us 6.28x, 32 x 500 us 7.12x, 32 x 5 ms 6.64x,
byte-identical results and registration key order at every point. Those numbers are a fact about the
MECHANISM, not a claim about the brain, which registers zero faculties today and therefore gets zero
speedup from any of it.

This docstring used to claim "the daemon imports `wired_faculties` (which registers every adapter) at
boot" -- there is no `wired_faculties` module in the tree, `learn_daemon.py` imports only this module, and
nothing anywhere calls `wire()` outside the self-check below (which deliberately clears `_WIRED` first, so
it proves the mechanism and says nothing about production). So `charge_all()` iterates an empty list every
cycle and the daemon logs `adapters=0`. The plug point works; it has never been plugged. Registering a
faculty here is real P3 work, and claiming otherwise was the same species of defect this registry exists
to expose.

Once registrations exist, they are picked up automatically: learn_daemon exits when git HEAD changes and
the launcher relaunches the new build, so the wired set tracks the code with no daemon edit per faculty.

HONEST COVERAGE: coverage() reports the EXPLICITLY-wired faculties per region against the brain-atlas region
map. It is NOT a claim that every module is exercised (Brain.tick charges more transitively; the coverage
audit measures that). It makes the wiring GAP visible + measurable, so "use every faculty" is tracked, not
faked.

Determinism / robustness (CLAUDE.md): charge_all() guards each faculty in its own try/except so one bad
faculty never stops the loop or blocks the others; pure registry, no wall-clock / no hash; ASCII-only.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.region_wiring
"""

import os
import threading
import time

# Registry of (name, region, charge_fn, value, parallel_safe). Module-global so `wire()` calls made at
# import time by whatever registers adapters accumulate into one set; the daemon reads it via
# charge_all()/coverage().
#
# This comment named `wired_faculties` as that registrar until 2026-08-04. THERE IS NO
# `wired_faculties` MODULE -- the docstring above says so in full and this line went on asserting
# the opposite twenty lines later, which is precisely the "fixing one copy is not fixing it" drift
# POLICIES.md records. The honest state is unchanged and is stated once, up there: nothing calls
# wire() outside the self-check, and filling this registry is open P3 work.
_WIRED = []


def wire(name, region, charge_fn, value="med", parallel_safe=False):
    """Register a faculty's per-cycle charge adapter. `charge_fn(brain, ctx)` returns a small result
    (dict / str / number) describing what it charged, for the coverage log. `value` in {high,med,low}.
    Returns charge_fn so it can be used as a decorator. Idempotent by (name): re-registering a name
    REPLACES it (so a reload / re-import does not double-register).

    `parallel_safe` (default False) is the faculty's OWN declaration that its adapter touches no state
    another faculty writes on the same cycle. It is necessary but NOT sufficient: `charge_all()` also
    requires the connectome-derived `same_tick_dependency_map()` to agree that the faculty's region is
    uncoupled this tick and off the actuator path. Default-False is deliberate -- an adapter that has
    not thought about concurrency must never be run concurrently, and the safe direction of a wrong
    answer here is 'slower', not 'raced'."""
    global _WIRED
    _WIRED = [w for w in _WIRED if w[0] != str(name)]
    _WIRED.append((str(name), str(region), charge_fn, str(value), bool(parallel_safe)))
    return charge_fn


def wired():
    """The registered faculties as [(name, region, value)] -- introspection for reports/tests."""
    return [(n, r, v) for (n, r, _fn, v, _p) in _WIRED]


def wired_detail():
    """As `wired()` plus each faculty's own `parallel_safe` declaration: [(name, region, value, safe)].
    Kept separate so the long-standing 3-tuple shape of `wired()` does not change under callers."""
    return [(n, r, v, p) for (n, r, _fn, v, p) in _WIRED]


# ==================================================================================================
# THE SAME-TICK DEPENDENCY MAP -- derived from the connectome, never guessed.
# ==================================================================================================
def same_tick_dependency_map(edges=None, nodes=None, gate_node=None, actuator=None):
    """DERIVE which brain regions can influence each other WITHIN a single tick.

    A connectome edge carries its source's rate `delay` ticks later, so an edge with `delay >= 1`
    creates NO same-tick dependency -- the target reads the PAST, which is exactly why an SNN
    timestep parallelises. Only a `delay <= 0` edge couples two regions within one tick. Regions
    joined (transitively, in either direction) by delay-0 edges form a CONFLICT GROUP and must be
    charged serially with respect to each other.

    Separately and unconditionally SERIAL: the region owning `gate_node` and any region with a direct
    edge to `actuator`. That is the arm's final common path -- MuJoCo `mjData` is not thread-safe and
    the safety envelope is physically sequential. Derived here rather than spelled out so it cannot
    drift away from `connectome.GATE_NODE` / `connectome.ACTUATOR`.

    `edges`/`nodes`/`gate_node`/`actuator` default to the live connectome and are injectable so the
    self-check can construct the RED (a real delay-0 edge) without editing `connectome.py`.

    Returns {n_edges, n_zero_delay, zero_delay_edges, conflict_groups, serial_regions,
             independent_regions}."""
    if edges is None or nodes is None:
        from snn_data_lake import connectome

        edges = connectome.EDGES if edges is None else edges
        nodes = connectome.NODES if nodes is None else nodes
        gate_node = connectome.GATE_NODE if gate_node is None else gate_node
        actuator = connectome.ACTUATOR if actuator is None else actuator

    def region_of(node_id):
        entry = nodes.get(node_id)
        return entry[0] if entry else None

    # --- (1) same-tick coupling: union-find over the delay-0 subgraph -----------------------------
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # deterministic: lexicographically smaller root wins

    zero_delay = []
    for e in edges:
        if int(e.get("delay", 1)) > 0:
            continue
        s_reg, t_reg = region_of(e["source"]), region_of(e["target"])
        zero_delay.append((e["source"], e["target"]))
        if s_reg and t_reg and s_reg != t_reg:
            union(s_reg, t_reg)
    groups = {}
    for reg in parent:
        groups.setdefault(find(reg), set()).add(reg)
    conflict_groups = sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])

    # --- (2) the actuator / safety path is serial unconditionally --------------------------------
    serial = set()
    gate_region = region_of(gate_node)
    if gate_region:
        serial.add(gate_region)
    for e in edges:
        if e["target"] == actuator:
            r = region_of(e["source"])
            if r:
                serial.add(r)

    all_regions = sorted({r for r in (region_of(n) for n in nodes) if r})
    coupled = {r for g in conflict_groups for r in g}
    return {
        "n_edges": len(edges),
        "n_zero_delay": len(zero_delay),
        "zero_delay_edges": sorted(zero_delay),
        "conflict_groups": conflict_groups,
        "serial_regions": sorted(serial),
        "independent_regions": [r for r in all_regions if r not in coupled and r not in serial],
    }


def parallel_regions(regions, dep_map=None):
    """Filter `regions` to those that may be charged CONCURRENTLY with each other, per the derived
    map: not on the actuator/safety path, and not sharing a same-tick conflict group with any other
    region in the set. Order-preserving. A region unknown to the connectome is treated as coupled
    (unknown -> serial), because 'we could not derive it' must never read as 'it is safe'."""
    dep = same_tick_dependency_map() if dep_map is None else dep_map
    serial = set(dep["serial_regions"])
    known = set(dep["independent_regions"]) | {r for g in dep["conflict_groups"] for r in g}
    group_of = {}
    for i, g in enumerate(dep["conflict_groups"]):
        for r in g:
            group_of[r] = i
    present = [r for r in regions]
    seen_groups = {}
    for r in present:
        if r in group_of:
            seen_groups[group_of[r]] = seen_groups.get(group_of[r], 0) + 1
    return [
        r
        for r in present
        if r not in serial and r in known and seen_groups.get(group_of.get(r, -1), 0) <= 1
    ]


# ==================================================================================================
# THE PARALLEL GATE -- measured on this machine, never a hardcoded width or thread count (P1).
# ==================================================================================================
# Explicit documented cap, not an arbiter: never ask for more workers than the machine has cores, and
# never more than 16. The 16 is a CAP on the search, not a threshold that decides anything -- the
# decision is always `pick_thread_count`'s measured net-win comparison, which returns 1 (serial) at
# every width where parallel does not pay. Measured justification for the cap on this machine (28
# cores): charge_all scaling flattens past 16 threads (64 faculties x 1000 us -> 7.65x at nt=8,
# 11.28x at nt=16), and both spiking kernels peak at nt<=8.
_MAX_THREADS_CAP = 16
_CAL_LOCK = threading.Lock()
_CAL = {}  # ("pool"|"threads", n_threads) -> measured dispatch overhead, seconds
_POOL = [
    None
]  # lazily created ThreadPoolExecutor; NEVER at module scope (run_selfchecks is 1 proc)


def max_threads():
    """The upper bound on workers this machine will be asked for. `min(cpu_count, cap)`, so a 2-core
    CI box never spawns 16 threads. Env override `NORD_PARALLEL_THREADS` clamps it further (0 or 1
    disables parallelism entirely, which is the honest off-switch for a bisect)."""
    n = min(int(os.cpu_count() or 1), _MAX_THREADS_CAP)
    raw = os.environ.get("NORD_PARALLEL_THREADS")
    if raw:
        try:
            n = min(n, int(raw))
        except ValueError:
            pass
    return max(1, n)


def _pool():
    """The shared, lazily-created worker pool for INDEPENDENT (barrier-free) work only. Barrier-based
    kernels must use dedicated threads: a barrier across pool tasks deadlocks if the pool is saturated,
    and that is a hang, not a slowdown.

    Sized at `max_threads()` ONCE, deliberately. An earlier version took the caller's requested width
    and sized the pool to whichever request happened to arrive first -- so a warm-up at 2 threads
    capped every later 16-thread request at 2 workers, and the measured speedup collapsed from 8.9x
    to 1.8x with nothing in the output saying why. Concurrency is limited by submitting exactly `nt`
    chunk tasks, never by starving the pool."""
    with _CAL_LOCK:
        if _POOL[0] is None:
            from concurrent.futures import ThreadPoolExecutor

            _POOL[0] = ThreadPoolExecutor(
                max_workers=max(2, max_threads()), thread_name_prefix="snn-charge"
            )
        return _POOL[0]


def parallel_overhead_s(n_threads, mode="pool"):
    """MEASURED cost, in seconds, on THIS machine, for `mode`:

      "pool"    -- dispatching and joining `n_threads` no-op tasks on the warm shared pool;
      "threads" -- creating and joining `n_threads` dedicated threads (the one-time setup cost);
      "barrier" -- ONE barrier round across `n_threads` already-running threads.

    Calibrated once per (mode, n_threads) and cached; lazy, never at import. `"barrier"` is measured
    separately because it is the term that decides whether a lock-step kernel pays at all, and it is
    NOT proportional to the setup cost: on a 28-core box thread setup runs ~300 us flat while a
    barrier round runs ~13 us at 2 threads and ~106 us at 8. A gate that models a per-micro-step
    kernel as if it paid setup only will over-thread it badly -- measured, and it is why this mode
    exists."""
    nt = max(1, int(n_threads))
    key = (mode, nt)
    hit = _CAL.get(key)
    if hit is not None:
        return hit
    if mode == "pool":
        ex = _pool()
        list(ex.map(lambda i: None, range(nt)))  # warm the workers before timing them

        def once():
            list(ex.map(lambda i: None, range(nt)))

        reps = 5
    elif mode == "barrier":
        rounds = 40

        def once():
            bar = threading.Barrier(nt)

            def spin():
                for _ in range(rounds):
                    bar.wait()

            ths = [threading.Thread(target=spin) for _ in range(nt - 1)]
            for t in ths:
                t.start()
            spin()
            for t in ths:
                t.join()

        reps = 3
    else:

        def once():
            ths = [threading.Thread(target=int) for _ in range(nt - 1)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()

        reps = 5

    once()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        best = min(best, time.perf_counter() - t0)
    if mode == "barrier":
        best = max(0.0, best - parallel_overhead_s(nt, "threads")) / 40  # net of thread setup
    with _CAL_LOCK:
        _CAL[key] = best
    return best


def parallel_cost_s(n_threads, mode="pool", sync_rounds=0):
    """The TOTAL measured overhead of running a kernel on `n_threads`: the one-time dispatch cost
    plus `sync_rounds` barrier rounds. A kernel that fans out once and joins passes `sync_rounds=0`;
    a lock-step kernel passes the number of barriers it actually executes."""
    nt = max(1, int(n_threads))
    cost = parallel_overhead_s(nt, mode)
    r = max(0, int(sync_rounds))
    if r:
        cost += r * parallel_overhead_s(nt, "barrier")
    return cost


def realized_efficiency(serial_estimate_s, elapsed_s, n_threads, mode="pool", sync_rounds=0):
    """Back-solve the fraction of IDEAL parallel savings a kernel actually achieved, from the model
    `elapsed = serial * (1 - eff * (1 - 1/nt)) + cost(nt)`. Returns a value in [0, 1].

    Perfect scaling is an assumption, and on real kernels it is wrong: measured here, the workspace
    settle achieves ~0.66 of ideal at 2 threads and ~0.77-0.80 at 4-8 (a barrier round between
    WORKING threads costs more than the same barrier between idle ones), the router's LIF update
    ~0.5-0.75 (memory-bandwidth bound), and `charge_all` ~0.9. Feeding that measurement
    back is what turns the gate from a predictor into a controller -- it is the same
    running-baseline discipline P1 asks for everywhere else, applied to the gate itself rather than
    to a threshold. Without it this gate chose 2 threads at 200 pools for a measured 0.91x."""
    try:
        est, el = float(serial_estimate_s), float(elapsed_s)
    except (TypeError, ValueError):
        return 1.0
    nt = int(n_threads)
    if nt < 2 or not (est > 0.0) or not (el > 0.0):
        return 1.0
    ideal = est * (1.0 - 1.0 / nt)
    if ideal <= 0.0:
        return 1.0
    eff = (est + parallel_cost_s(nt, mode, sync_rounds) - el) / ideal
    return 0.0 if eff < 0.0 else (1.0 if eff > 1.0 else eff)


def pick_thread_count(serial_estimate_s, mode="pool", sync_rounds=0, efficiency=1.0, cap=None):
    """THE GATE. Return the thread count with the largest MEASURED net win, or 1 to run serial.

    Perfect-scaling model: `nt` threads save `serial * (1 - 1/nt)` and cost
    `parallel_cost_s(nt, mode, sync_rounds)`. Both terms are measured -- the caller's own running-EMA
    estimate of its serial cost, and this machine's dispatch and barrier floors. Whichever `nt`
    maximises the difference wins, and if no candidate makes the difference positive the answer is 1
    and the caller runs serial. That is what makes the crossover DERIVED: change the machine, or the
    work, and the same expression moves.

    `sync_rounds` is not optional bookkeeping -- it is the difference between a gate that works and
    one that does harm. Modelling the workspace's 121-barrier settle as a single dispatch made this
    gate choose 16 threads where 4 was right, and turned a measured 2.0x win at 500 pools into a
    measured 0.41x loss. The caller knows its own synchronisation structure; the gate must be told.

    Zero/empty/one, checked in the self-check: a zero, negative or non-finite estimate returns 1; a
    cap below 2 returns 1; a single-core machine returns 1."""
    try:
        est = float(serial_estimate_s)
    except (TypeError, ValueError):
        return 1
    if not (est > 0.0) or est == float("inf"):
        return 1
    top = max_threads() if cap is None else min(max_threads(), int(cap))
    if top < 2:
        return 1
    candidates = []
    nt = 2
    while nt <= top:
        candidates.append(nt)
        nt *= 2
    if top not in candidates:
        candidates.append(top)
    try:
        eff = float(efficiency)
    except (TypeError, ValueError):
        eff = 1.0
    eff = 0.0 if eff < 0.0 else (1.0 if eff > 1.0 else eff)
    best_nt, best_gain = 1, 0.0
    for nt in candidates:
        gain = est * eff * (1.0 - 1.0 / nt) - parallel_cost_s(nt, mode, sync_rounds)
        if gain > best_gain:
            best_nt, best_gain = nt, gain
    return best_nt


def blocks(n, k):
    """Partition range(n) into at most `k` CONTIGUOUS, non-empty [start, stop) blocks, largest first.
    Contiguity is what makes a partitioned reduction reproducible: chunk order == index order, so a
    left-fold over the chunks in order visits elements in exactly the serial order."""
    n = max(0, int(n))
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    step, rem = divmod(n, k)
    out, a = [], 0
    for i in range(k):
        b = a + step + (1 if i < rem else 0)
        if b > a:
            out.append((a, b))
        a = b
    return out


# ==================================================================================================
_CHARGE_EMA = [0.0]  # running mean of the SERIAL cost of one charge_all(), seconds (P1 baseline)
_CHARGE_EFF = [1.0]  # running mean of the fraction of IDEAL parallel savings actually achieved
_EMA_ALPHA = 0.25


def charge_ema_s():
    """The running-EMA estimate of one serial charge_all(), seconds. 0.0 until a serial cycle has
    been measured -- which is why the FIRST cycle always runs serial: the gate refuses to guess."""
    return _CHARGE_EMA[0]


def _run_one(fn, brain, ctx):
    try:
        return fn(brain, ctx)
    except Exception as e:  # noqa: BLE001 -- a bad faculty must never kill the always-on loop
        return f"ERR {type(e).__name__}: {e}"


def charge_all(brain, ctx=None, parallel=None):
    """Charge EVERY wired faculty this cycle. Each is guarded in its own try/except: a failure is recorded
    as an 'ERR ...' string, never raised, so one bad faculty never stops the always-on loop or blocks the
    others. Returns {name: result_or_error}.

    Faculties run CONCURRENTLY when all four hold, and serially otherwise:
      1. the faculty declared `parallel_safe=True` at `wire()` time;
      2. `same_tick_dependency_map()` says its region is not on the actuator/safety path and shares no
         same-tick conflict group with another region in this cycle's set;
      3. `pick_thread_count()` finds a measured net win against this machine's dispatch floor, using
         the running EMA of this function's own serial cost (so cycle 1 is always serial);
      4. at least two faculties qualify -- one faculty is never worth a thread.
    Faculties that fail 1 or 2 still run, on the calling thread, AFTER the concurrent set joins.

    DETERMINISM: results are written into per-faculty slots (disjoint -- no shared container is
    mutated by two threads) and the returned dict is built in REGISTRATION order. Thread completion
    order cannot reach the caller. Ordering does not depend on `parallel` being on or off, so the
    serial and parallel paths return equal dicts with equal key order."""
    ctx = {} if ctx is None else ctx
    entries = list(_WIRED)
    if not entries:
        return {}

    declared = [i for i, w in enumerate(entries) if w[4]]
    if declared:
        allowed = set(parallel_regions([entries[i][1] for i in declared]))
        conc = [i for i in declared if entries[i][1] in allowed]
    else:
        conc = []

    nt = 1
    if len(conc) >= 2:
        if parallel is None:
            nt = min(pick_thread_count(_CHARGE_EMA[0], efficiency=_CHARGE_EFF[0]), len(conc))
        elif parallel:
            nt = min(max_threads(), len(conc))
    use_parallel = nt >= 2 and len(conc) >= 2

    slots = [None] * len(entries)
    if use_parallel:
        ex = _pool()
        est = _CHARGE_EMA[0]

        def _run_chunk(ab):  # exactly `nt` chunk tasks -> exactly `nt` workers busy
            for j in range(ab[0], ab[1]):
                i = conc[j]
                slots[i] = _run_one(entries[i][2], brain, ctx)

        t0 = time.perf_counter()
        list(ex.map(_run_chunk, blocks(len(conc), nt)))
        conc_set = set(conc)
        for i in range(len(entries)):  # everything not eligible runs HERE, on the calling thread
            if i not in conc_set:
                slots[i] = _run_one(entries[i][2], brain, ctx)
        if est > 0.0:  # feed the ACHIEVED scaling back into the gate
            eff = realized_efficiency(est, time.perf_counter() - t0, nt)
            _CHARGE_EFF[0] = (1.0 - _EMA_ALPHA) * _CHARGE_EFF[0] + _EMA_ALPHA * eff
    else:
        t0 = time.perf_counter()
        for i, w in enumerate(entries):
            slots[i] = _run_one(w[2], brain, ctx)
        dt = time.perf_counter() - t0
        _CHARGE_EMA[0] = (
            dt if _CHARGE_EMA[0] <= 0.0 else ((1.0 - _EMA_ALPHA) * _CHARGE_EMA[0] + _EMA_ALPHA * dt)
        )
    return {entries[i][0]: slots[i] for i in range(len(entries))}  # REGISTRATION order, always


def coverage():
    """Honest coverage of the EXPLICIT wirings vs the brain-atlas region map. Returns:
      {regions_total, regions_wired, n_wired_faculties, region_faculty_counts, unwired_regions}.
    This counts explicitly-wired faculties only -- NOT total brain coverage (Brain.tick charges more
    transitively). It surfaces the visible wiring gap so progress toward 'every faculty' is measurable."""
    regions_total = []
    try:
        # the anatomical region -> module map: the FROZEN pre-G8 labels since G8 S1a (F5), so
        # G8's engine-function packages never count as unwired regions
        from snn_data_lake import LEGACY_REGIONS

        regions_total = sorted(LEGACY_REGIONS)
    except Exception:
        regions_total = []
    counts = {}
    for _n, region, _fn, _v, _p in _WIRED:
        counts[region] = counts.get(region, 0) + 1
    return {
        "regions_total": len(regions_total),
        "regions_wired": len([r for r in regions_total if counts.get(r)]),
        "n_wired_faculties": len(_WIRED),
        "region_faculty_counts": counts,
        "unwired_regions": [r for r in regions_total if not counts.get(r)],
    }


def live_coverage():
    """The AUTHORITATIVE wiring coverage (Policy 3), delegating to gates.dormancy_gate: how many
    brain modules are REACHABLE from a live path/entrypoint, how many are EXEMPT (with a reason), and
    how many are (silently) DORMANT -- which is 0 unless the dormancy gate is failing. This is distinct
    from coverage() above, which counts only the EXPLICIT per-cycle charge adapters registered here (an
    optional mechanism for genuinely-periodic faculties -- NOT a way to fake liveness for a dormant
    module; the honest way to make a faculty live is to WIRE it into the tick so it charges behaviour).
    Returns {total, reachable, exempt, dormant}."""
    from snn_data_lake.gates import dormancy_gate

    c = dormancy_gate.coverage()
    return {
        "total": c["total"],
        "reachable": c["reachable"],
        "exempt": c["exempt"],
        "dormant": len(c["dormant"]),
    }


# ================================================================================================
def main():
    global _WIRED
    saved = _WIRED
    _WIRED = []  # isolate the self-check from any real registrations
    try:
        lc = live_coverage()  # the AUTHORITATIVE Policy-3 coverage (delegates to the dormancy gate)
        assert lc["dormant"] == 0 and lc["reachable"] > 0 and lc["total"] > lc["reachable"], lc

        # ---- the DERIVED same-tick dependency map -------------------------------------------
        dep = same_tick_dependency_map()
        assert dep["n_edges"] > 0, dep
        # Derived, not asserted: the live connectome has no delay-0 edge, so nothing is coupled.
        assert dep["n_zero_delay"] == 0 and dep["conflict_groups"] == [], (
            f"live connectome gained a same-tick edge; re-derive before parallelising: {dep}"
        )
        # The arm's final common path is SERIAL, and it is derived from GATE_NODE/ACTUATOR.
        assert "brainstem" in dep["serial_regions"], (
            f"the connectome gate node's region must never be parallel-eligible: {dep}"
        )
        assert "brainstem" not in dep["independent_regions"], dep
        assert "cortex" in dep["independent_regions"], dep

        # REACHABLE RED for the coupling branch: inject a real delay-0 edge and prove it is CAUGHT.
        red_nodes = {
            "a1": ("cortex", "a"),
            "b1": ("hippocampus", "b"),
            "c1": ("thalamus", "c"),
            "m1": ("brainstem", "gate"),
            "act": (None, "effector"),
        }
        red_edges = [
            {"source": "a1", "target": "b1", "sign": "exc", "delay": 0},  # SAME-TICK -> couples
            {"source": "b1", "target": "c1", "sign": "exc", "delay": 0},  # transitive -> one group
            {"source": "c1", "target": "a1", "sign": "exc", "delay": 1},  # delayed -> no coupling
            {"source": "m1", "target": "act", "sign": "exc", "delay": 1},
        ]
        red = same_tick_dependency_map(red_edges, red_nodes, "m1", "act")
        assert red["n_zero_delay"] == 2, red
        assert red["conflict_groups"] == [["cortex", "hippocampus", "thalamus"]], (
            f"a delay-0 chain must transitively couple all three regions: {red}"
        )
        assert red["independent_regions"] == [], red
        assert red["serial_regions"] == ["brainstem"], red
        # ...and the filter must actually DEMOTE the coupled pair while keeping a lone member.
        assert parallel_regions(["cortex", "hippocampus"], red) == [], (
            "two regions in one same-tick conflict group must NOT run concurrently"
        )
        assert parallel_regions(["cortex"], red) == ["cortex"], (
            "a lone member of a conflict group has nobody to race with -- it stays eligible"
        )
        assert parallel_regions(["brainstem", "cortex"], red) == ["cortex"], (
            "the actuator/safety path must be filtered out unconditionally"
        )
        assert parallel_regions(["not_a_region"], red) == [], (
            "a region the connectome does not know is treated as coupled (unknown -> serial)"
        )
        # The gate-node rule and the direct-actuator-edge rule BOTH cover brainstem on the live
        # connectome, so a map that lost the gate-node rule still looks correct there. This case
        # isolates it: a gate node with no actuator edge at all, where only that rule can protect it.
        solo_gate = same_tick_dependency_map(
            [{"source": "x1", "target": "g1", "sign": "exc", "delay": 1}],
            {"g1": ("brainstem", "gate"), "x1": ("cortex", "x"), "act": (None, "eff")},
            "g1",
            "act",
        )
        assert solo_gate["serial_regions"] == ["brainstem"], (
            f"the GATE NODE's region must be serial on its own account, not only because something "
            f"happens to project to the actuator: {solo_gate}"
        )
        assert parallel_regions(["brainstem", "cortex"], solo_gate) == ["cortex"], solo_gate

        # ---- the MEASURED parallel gate ------------------------------------------------------
        # zero / empty / one (P1: check any new baseline at its boundaries)
        assert pick_thread_count(0.0) == 1, "no measured work -> serial"
        assert pick_thread_count(-1.0) == 1, "a negative estimate -> serial"
        assert pick_thread_count(None) == 1, "an unmeasurable estimate -> serial"
        assert pick_thread_count(float("inf")) == 1, "a non-finite estimate -> serial"
        assert pick_thread_count(1.0, cap=1) == 1, "a one-worker cap -> serial"
        assert blocks(0, 4) == [] and blocks(1, 4) == [(0, 1)], blocks(1, 4)
        assert blocks(7, 3) == [(0, 3), (3, 5), (5, 7)], blocks(7, 3)
        assert sum(b - a for a, b in blocks(97, 8)) == 97, "a partition must cover every index once"
        floor = parallel_overhead_s(2, "pool")
        assert floor > 0.0, "the dispatch floor must be MEASURED, not assumed zero"
        # RED for the gate: work far below the measured floor must NOT be parallelised...
        assert pick_thread_count(floor * 0.05) == 1, (
            f"work well under the measured dispatch floor ({floor * 1e6:.1f} us) must stay serial"
        )
        # ...and work far above it MUST be, or the gate is a permanent 'no' wearing a measurement.
        assert pick_thread_count(floor * 5000.0) >= 2, (
            "work thousands of times the dispatch floor must select >= 2 threads"
        )
        # sync_rounds is load-bearing: the same work, declared as a lock-step kernel, must be
        # threaded LESS. Without this the gate over-threads every barrier kernel in the tree.
        big = floor * 5000.0
        assert pick_thread_count(big, sync_rounds=122) <= pick_thread_count(big), (
            "declaring barrier rounds must never INCREASE the chosen thread count"
        )
        assert parallel_cost_s(4, "threads", 122) > parallel_cost_s(4, "threads", 0), (
            "122 barrier rounds must cost measurably more than a bare fan-out"
        )
        assert parallel_cost_s(4, "threads", 0) == parallel_overhead_s(4, "threads"), (
            "zero sync rounds must add nothing"
        )
        assert parallel_overhead_s(8, "barrier") > 0.0, "a barrier round must be MEASURED, not free"
        # realized-efficiency feedback: a run that achieved nothing must drive the estimate down and
        # the next decision back to serial. Zero/one boundaries included.
        assert realized_efficiency(1.0, 1.0, 1) == 1.0, "nt<2 has no parallel efficiency to measure"
        assert realized_efficiency(0.0, 1.0, 4) == 1.0, "no serial estimate -> no measurement"
        assert realized_efficiency(1.0, 0.0, 4) == 1.0, "no elapsed time -> no measurement"
        assert realized_efficiency(1.0, 99.0, 4) == 0.0, "a run slower than serial has 0 efficiency"
        ideal_el = 1.0 * (1.0 - (1.0 - 1.0 / 4)) + parallel_cost_s(4, "pool", 0)
        assert abs(realized_efficiency(1.0, ideal_el, 4) - 1.0) < 1e-9, (
            "a perfectly-scaling run must back-solve to efficiency 1.0"
        )
        assert 0.4 < realized_efficiency(1.0, 0.55 + parallel_cost_s(4, "pool", 0), 4) < 0.7, (
            realized_efficiency(1.0, 0.55 + parallel_cost_s(4, "pool", 0), 4)
        )
        assert pick_thread_count(big, efficiency=0.0) == 1, (
            "a kernel measured to gain NOTHING from threads must be returned to serial"
        )
        assert pick_thread_count(big, efficiency=0.02) <= pick_thread_count(big, efficiency=1.0), (
            "lower measured efficiency must never select MORE threads"
        )

        # ---- charge_all: parallel path is byte-identical to serial, in registration order -----
        _WIRED.clear()
        order = []
        tids = [None] * 24  # slot i is written only by faculty i -- disjoint, no shared mutation

        for i in range(24):

            def slow(b, c, i=i):
                x = 0.0
                for j in range(4000):  # real Python work -- sleep would fake the scaling
                    x += j * 0.5
                tids[i] = threading.get_ident()
                return {"i": i, "x": x}

            wire(f"p{i:02d}", "cortex", slow, parallel_safe=True)
            order.append(f"p{i:02d}")
        # These two report the THREAD they ran on, because that is the property under test. Asserting
        # on a return value cannot tell a brainstem faculty that stayed home from one that was handed
        # to a worker: both return the same thing. mjData is not thread-safe, so 'it ran on the tick
        # thread' is the actual safety claim and it is the one asserted.
        here = threading.get_ident()
        wire("unsafe", "cortex", lambda b, c: threading.get_ident())  # parallel_safe defaults False
        wire("motor", "brainstem", lambda b, c: threading.get_ident(), parallel_safe=True)
        wire("boom", "cortex", lambda b, c: 1 / 0, parallel_safe=True)
        ser = charge_all(None, {"tick": 1}, parallel=False)
        assert all(t == here for t in tids), "the serial path must run every faculty on this thread"
        par = charge_all(None, {"tick": 1}, parallel=True)
        assert list(ser) == list(par), (
            f"key order must be REGISTRATION order on both paths: {list(ser)[:3]} vs {list(par)[:3]}"
        )
        assert list(par)[: len(order)] == order, list(par)[: len(order)]
        assert {k: v for k, v in ser.items() if k.startswith("p")} == {
            k: v for k, v in par.items() if k.startswith("p")
        }, "parallel-safe faculties must return byte-identical results on both paths"
        assert par["motor"] == here, (
            "a brainstem (actuator/safety-path) faculty must run on the CALLING thread even when it "
            "declares itself parallel-safe -- MuJoCo mjData is not thread-safe"
        )
        assert par["unsafe"] == here, (
            "a faculty that did not declare parallel_safe must run on the calling thread"
        )
        assert ser["motor"] == here and ser["unsafe"] == here, ser
        # The pool must be able to SERVE the width the gate picks. Sizing it to whichever request
        # arrived first is a pure-throughput bug -- every result stays correct, every assert above
        # still passes, and the measured speedup silently collapsed from 6.3x to 1.8x. Correctness
        # asserts cannot see this class of defect, so it is pinned structurally.
        assert _pool()._max_workers >= max_threads(), (
            f"the shared pool must be sized for the widest fan-out the gate can choose "
            f"({_pool()._max_workers} workers < cap {max_threads()})"
        )
        assert len(set(tids)) >= 2, (
            f"a parallel charge must actually spread across threads, got {len(set(tids))} distinct"
        )
        n_offloaded = sum(1 for t in tids if t != here)
        assert n_offloaded >= 2, (
            f"the parallel path must actually offload work, or every assert above is vacuous "
            f"(only {n_offloaded} of {len(tids)} faculties left the calling thread)"
        )
        assert par["boom"].startswith("ERR ZeroDivisionError"), (
            f"a faculty that raises must be GUARDED on the parallel path too: {par['boom']!r}"
        )
        # the auto gate: EMA is 0 on a cold registry, so the first auto cycle MUST be serial
        _CHARGE_EMA[0] = 0.0
        auto1 = charge_all(None, {"tick": 2})
        assert _CHARGE_EMA[0] > 0.0, "a serial cycle must record its own cost (the P1 baseline)"
        assert auto1 == ser, auto1
        assert charge_all(None, {"tick": 3}) == ser, "the auto-gated cycle must agree with serial"

        _WIRED.clear()
        wire("dummy_ok", "cortex", lambda b, c: {"charged": True, "seen": c.get("tick", 0)})
        wire("dummy_bad", "cortex", lambda b, c: (_ for _ in ()).throw(RuntimeError("boom")))
        wire("dummy_amyg", "amygdala", lambda b, c: 1.0)
        # re-registering a name REPLACES it (reload-safe: no double registration)
        wire("dummy_ok", "cortex", lambda b, c: {"charged": True, "v2": True})
        assert len(wired()) == 3, f"re-wiring a name must REPLACE, not duplicate: {wired()!r}"

        res = charge_all(brain=None, ctx={"tick": 7})
        assert res["dummy_ok"] == {"charged": True, "v2": True}, res["dummy_ok"]
        assert res["dummy_amyg"] == 1.0
        # a failing faculty is GUARDED (recorded, not raised) so the others still charge
        assert res["dummy_bad"].startswith("ERR RuntimeError: boom"), res["dummy_bad"]

        cov = coverage()
        assert cov["region_faculty_counts"].get("cortex") == 2, cov  # dummy_ok + dummy_bad
        assert cov["region_faculty_counts"].get("amygdala") == 1, cov
        assert cov["n_wired_faculties"] == 3, cov

        print("region_wiring self-check OK:")
        print(
            f"  same-tick dependency map (DERIVED from connectome): {dep['n_edges']} edges, "
            f"{dep['n_zero_delay']} delay-0 -> {len(dep['conflict_groups'])} conflict groups; "
            f"serial (actuator/safety) {dep['serial_regions']}; "
            f"{len(dep['independent_regions'])} regions independent per tick"
        )
        print(
            f"  parallel gate: measured dispatch floor nt=2 {parallel_overhead_s(2) * 1e6:.1f} us, "
            f"nt={max_threads()} {parallel_overhead_s(max_threads()) * 1e6:.1f} us; "
            f"cold EMA -> serial, {floor * 5000 * 1e3:.0f} ms of work -> "
            f"{pick_thread_count(floor * 5000.0)} threads"
        )
        print("  charge_all parallel == serial, byte-identical, in registration order")
        print(
            f"  live_coverage (Policy-3, via dormancy_gate): wired {lc['reachable']}/{lc['total']} "
            f"exempt {lc['exempt']} dormant {lc['dormant']}"
        )
        print(
            "  charge_all charged",
            len(res),
            "faculties; one failing faculty was GUARDED, not raised",
        )
        print("  re-wiring a name REPLACES it (reload-safe)")
        print(
            f"  coverage(): {cov['regions_wired']}/{cov['regions_total']} regions have >=1 explicit "
            f"wiring; unwired: {cov['unwired_regions']}"
        )
        return 0
    finally:
        _WIRED = saved


if __name__ == "__main__":
    import sys

    sys.exit(main())
