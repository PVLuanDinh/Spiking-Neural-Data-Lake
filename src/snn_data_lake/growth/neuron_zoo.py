"""
Neuron Zoo -- ONE heterogeneous point-neuron population behind a single uniform contract.

Every neuron model in this repo lived in its own module with its own constructor and its own
step signature; a pool was always HOMOGENEOUS (all LIF) because there was no common way to mix
them. This module is that common way. It does NOT re-implement any dynamics -- it IMPORTS the
existing (largely dead) neuron modules and puts a thin adapter in front of each so they all obey
one contract:

    neuron.step(current) -> 0 or 1          # one timestep, one input current, one spike bit
    neuron.reset_state()                     # back to the model's rest state

    make_neuron(kind, seed) -> neuron        # build one cell of a named kind
    sample_population(n, seed, kinds=...) -> [neuron, ...]   # build a mixed pool
    Population(cells).step(currents) -> [0/1, ...]           # advance the WHOLE pool one timestep

The zoo REUSES (import + adapt, never rewrite) the repo's own neuron models:
  * growth.lif_neuron.LIFNeuron        -- leaky integrate-and-fire      (native step/reset_state)
  * growth.alif_neuron.ALIFNeuron      -- adaptive-threshold LIF (LSNN) (native step/reset_state)
  * growth.autapse_neuron.AutapseNeuron-- one neuron, delayed self-loops(native step/reset_state)
  * growth.dendritic_neuron.DendriticNeuron -- two-compartment BAC cell (2-arg step -> adapter)
  * growth.adex_neuron.AdEx            -- adaptive exponential I&F      (units + reset() -> adapter)
  * growth.resonate_and_fire.ResonateFire -- frequency-selective        (units + reset() -> adapter)
  * growth.srm_neuron.SRMNeuron        -- kernel/soft-reset spike model (event API -> adapter)

DEFAULT-SAFE population = the six always-finite point/reduced models above (SRM is a
kernel model included in the safe set). Two BIOPHYSICAL models integrate forward-Euler ODEs that can
diverge, so they are OPT-IN ONLY and NaN-GUARDED (a non-finite drive or a blown-up state is clamped /
skipped, never propagated, never raised) and are EXCLUDED from the default population:
  * growth.hodgkin_huxley.HodgkinHuxley-- conductance-based biophysical (V trace -> spike + NaN guard)
  * growth.izhikevich_neuron           -- 2-var reduced (batch-only module -> stepping wrapper + guard)

The DEFAULT population kind set is ("lif",) only, so a pool built with the defaults is homogeneous LIF
and BYTE-IDENTICAL to today's direct LIFNeuron pools -- adopting the zoo changes nothing until a caller
explicitly asks for a heterogeneous `kinds=...`. Heterogeneity is opt-in.

SAFETY / SCOPE: this module is intent-recognition/substrate only -- it builds neurons and returns spike
bits. It adds NO actuator route. A heterogeneous READOUT can, however, change which intent wins an
argmax, so wiring it into the interaction-learning intent readout is ARM-ADJACENT and must be E-STOP
re-verified by the main agent (see the integration-hook spec returned with this build).

Deterministic: per-instance random.Random(seed); no module-level seeding, no
wall-clock. ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.neuron_zoo
"""

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

from snn_data_lake.growth.adex_neuron import AdEx
from snn_data_lake.growth.alif_neuron import ALIFNeuron
from snn_data_lake.growth.autapse_neuron import AutapseNeuron
from snn_data_lake.growth.dendritic_neuron import DendriticNeuron
from snn_data_lake.growth.hodgkin_huxley import HodgkinHuxley
from snn_data_lake.growth.izhikevich_neuron import REGIMES as _IZHI_REGIMES
from snn_data_lake.growth.lif_neuron import LIFNeuron
from snn_data_lake.growth.resonate_and_fire import ResonateFire
from snn_data_lake.growth.srm_neuron import SRMNeuron

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_train": (
        "NOT_A_RULE",
        "drives ONE neuron with a current stream from rest and returns its 0/1 spike train. A "
        "simulation of membrane dynamics; no weight exists to update. Caught purely on the name",
    ),
}

# --- input-scale constants: the zoo presents ONE "current" in LIF units (thresholds ~1.0). Models that
#     work in other native units (pA for AdEx, arbitrary drive for the resonator, uA/cm^2 for HH, the
#     Izhikevich current scale) get that common current mapped into their operating range by a fixed
#     per-kind scale INSIDE the adapter. Pure re-scaling; it re-uses the model's own dynamics untouched.
_ADEX_SCALE = 1500.0  # LIF-unit current -> pA (AdEx rheobase ~ a few hundred pA)
_RES_SCALE = 130.0  # LIF-unit current -> resonator drive (fixed-point above threshold)
_HH_SCALE = 45.0  # LIF-unit current -> uA/cm^2 (HH rheobase ~ a few uA/cm^2)
_IZHI_SCALE = 55.0  # LIF-unit current -> Izhikevich current scale (~10 in the source module)


def _finite(x):
    """True iff x is a real, finite number (the NaN/inf gate for the biophysical adapters)."""
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# --------------------------------------------------------------------------------------------------
# Thin adapters: each wraps ONE existing model and exposes step(current)->0/1 + reset_state().
# The dynamics live in the imported module; the adapter only translates the interface (and, for the
# two biophysical models, guards against non-finite states). Every adapter stores the wrapped model
# on `.raw` so a caller can still reach the underlying object.
# --------------------------------------------------------------------------------------------------
class _DendriticCell:
    """DendriticNeuron behind the uniform contract: the common current drives the BASAL compartment
    (apical held at 0 -> the basal-drive regime), the somatic 0/1 spike is returned unchanged."""

    kind = "dendritic"

    def __init__(self, seed=0):
        self.raw = DendriticNeuron()  # deterministic; no internal randomness
        self.reset_state()

    def reset_state(self):
        self.raw.reset_state()

    def step(self, current):
        return self.raw.step(current, 0.0)  # basal = common current, apical = 0


class _AdExCell:
    """AdEx (physical-unit adaptive exponential I&F) behind the uniform contract: rescale the common
    current into pA and rename reset() -> reset_state(). Native step(I)->0/1 kept as-is."""

    kind = "adex"

    def __init__(self, seed=0, scale=_ADEX_SCALE):
        self.raw = AdEx()  # deterministic fixed-step forward Euler
        self.scale = scale
        self.reset_state()

    def reset_state(self):
        self.raw.reset()

    def step(self, current):
        if not _finite(current):
            return 0
        return self.raw.step(current * self.scale)


class _ResonateCell:
    """ResonateFire (frequency-selective resonator) behind the uniform contract: rescale the common
    current into the resonator's drive range and rename reset() -> reset_state()."""

    kind = "resonate"

    def __init__(self, seed=0, scale=_RES_SCALE):
        self.raw = ResonateFire(10.0)  # 10 Hz-tuned; deterministic
        self.scale = scale
        self.reset_state()

    def reset_state(self):
        self.raw.reset()

    def step(self, current):
        if not _finite(current):
            return 0
        return self.raw.step(current * self.scale)


class _SRMCell:
    """SRMNeuron (kernel + soft-reset spike-response model) behind the uniform contract. The source
    exposes an event-time API (step(t, in_weights)); this adapter advances an internal integer clock
    by one tick per step() and injects the common current as a single weighted input, returning the
    0/1 fire bit. Reuses the module's eps/eta kernels untouched."""

    kind = "srm"

    def __init__(self, seed=0):
        self.raw = SRMNeuron()
        self.reset_state()

    def reset_state(self):
        self.raw.inputs = []
        self.raw.fires = []
        self._t = 0

    def step(self, current):
        self._t += 1
        fired = self.raw.step(self._t, (float(current),) if _finite(current) else ())
        return 1 if fired else 0


class _HodgkinHuxleyCell:
    """HodgkinHuxley (conductance-based biophysical) behind the uniform contract. HH.step returns the
    membrane voltage V (mV), not a spike, so this adapter emits 1 on an upward crossing of a fixed
    detection level and 0 otherwise. OPT-IN + NaN-GUARDED: a non-finite drive is skipped, and a
    blown-up membrane is clamped back to rest -- a non-finite value never escapes and never raises."""

    kind = "hh"
    _LEVEL = 0.0  # mV: an upward crossing of 0 mV counts as one AP

    def __init__(self, seed=0, scale=_HH_SCALE):
        self.raw = HodgkinHuxley(seed=seed)
        self.scale = scale
        self.reset_state()

    def reset_state(self):
        self.raw.reset()
        self._prev = self.raw.v

    def step(self, current):
        if not _finite(current):  # skip a non-finite drive; hold state
            return 0
        try:
            v = self.raw.step(current * self.scale)
        except (OverflowError, ValueError):  # the gate ODEs (math.exp) can overflow on a wild V
            self.raw.reset()
            self._prev = self.raw.v
            return 0
        if (
            not _finite(v) or abs(v) > 1.0e4
        ):  # divergence guard: a non-finite OR absurd (>1e4 mV) V ->
            self.raw.reset()  #   clamp back to rest so it cannot poison the next step
            self._prev = self.raw.v
            return 0
        spike = 1 if (self._prev < self._LEVEL <= v) else 0
        self._prev = v
        return spike


class _IzhikevichCell:
    """Izhikevich 2-variable reduced model behind the uniform contract. izhikevich_neuron exposes only
    a batch simulate() (no single-step primitive), so this adapter is a THIN STEPPING WRAPPER around
    that module's canonical REGIMES parameter sets and its identical update equations -- not a new
    model. OPT-IN + NaN-GUARDED: a non-finite drive is skipped and a diverged state is clamped to rest,
    so the quadratic v-update can never leak a non-finite value or raise."""

    kind = "izhikevich"

    def __init__(self, seed=0, scale=_IZHI_SCALE, regime="regular-spiking", dt=0.5):
        self.a, self.b, self.c, self.d = _IZHI_REGIMES[
            regime
        ]  # reuse the module's canonical params
        self.scale = scale
        self.dt = dt
        self.reset_state()

    def reset_state(self):
        self.v = -65.0
        self.u = self.b * -65.0

    def step(self, current):
        if not _finite(current):
            return 0
        if not (_finite(self.v) and _finite(self.u)):  # a prior diverged state -> clamp
            self.reset_state()
            return 0
        spike = 0
        if self.v >= 30.0:  # spike recorded from the previous overshoot, then reset
            spike = 1
            self.v = self.c
            self.u += self.d
        i = current * self.scale
        try:
            self.v += self.dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i)
            self.u += self.dt * (self.a * (self.b * self.v - self.u))
        except (OverflowError, ValueError):  # quadratic v-update can overflow -> clamp to rest
            self.reset_state()
            return spike
        if not _finite(self.v) or abs(self.v) > 1.0e6:  # divergence guard on the quadratic update
            self.reset_state()
        return spike


# native models already satisfy step(current)->0/1 + reset_state(); no adapter needed
def _make_lif(seed=0):
    return LIFNeuron()  # default params -> the byte-identical anchor


def _make_alif(seed=0):
    return ALIFNeuron()


def _make_autapse(seed=0):
    return AutapseNeuron(delays=(1, 2, 3), seed=seed)  # seed -> its random self-feedback weights


# --------------------------------------------------------------------------------------------------
# kind sets
# --------------------------------------------------------------------------------------------------
DEFAULT_KINDS = ("lif",)  # a default pool is homogeneous LIF = byte-identical to today
SAFE_KINDS = (
    "lif",
    "alif",
    "autapse",
    "dendritic",
    "adex",
    "resonate",
    "srm",
)  # always-finite, default-safe
OPTIN_KINDS = (
    "hh",
    "izhikevich",
)  # biophysical ODEs: opt-in only, NaN-guarded, excluded by default
ALL_KINDS = SAFE_KINDS + OPTIN_KINDS


class NeuronZoo:
    """A registry of heterogeneous point/reduced neuron models behind one uniform step()/reset_state()
    contract. make(kind, seed) builds one; sample(n, seed, kinds) builds a mixed pool. The default
    kind set is homogeneous LIF, so the default pool is byte-identical to a direct LIFNeuron pool."""

    def __init__(self):
        self._builders = {
            "lif": _make_lif,
            "alif": _make_alif,
            "autapse": _make_autapse,
            "dendritic": _DendriticCell,
            "adex": _AdExCell,
            "resonate": _ResonateCell,
            "srm": _SRMCell,
            "hh": _HodgkinHuxleyCell,
            "izhikevich": _IzhikevichCell,
        }

    def kinds(self):
        """The registered kind names (safe + opt-in)."""
        return tuple(self._builders)

    def make(self, kind, seed=0):
        """Build one neuron of the named kind. Every returned object exposes step(current)->0/1 and
        reset_state()."""
        try:
            builder = self._builders[kind]
        except KeyError:
            raise ValueError(
                f"unknown neuron kind {kind!r}; known: {sorted(self._builders)}"
            ) from None
        return builder(seed=seed)

    def sample(self, n, seed=0, kinds=DEFAULT_KINDS):
        """Build a pool of n neurons, cycling deterministically through `kinds` (round-robin). With the
        default kinds=('lif',) the pool is homogeneous LIF -> byte-identical to a direct LIFNeuron pool.
        Each neuron gets a distinct sub-seed derived from `seed` (matters for the kinds that use RNG)."""
        if n < 0:
            raise ValueError(f"population size must be >= 0 (got {n})")
        if not kinds:
            raise ValueError("kinds must be a non-empty sequence of kind names")
        rng = random.Random(seed)  # reserved for future jitter; determinism anchor
        pool = []
        for i in range(n):
            kind = kinds[i % len(kinds)]  # round-robin -> balanced, deterministic mix
            pool.append(self.make(kind, seed=seed * 1000 + i + rng.randint(0, 0)))
        return pool


# module-level default zoo + convenience API (what the integration hook imports)
_ZOO = NeuronZoo()


def make_neuron(kind, seed=0):
    """Build one neuron of the named kind (step(current)->0/1 + reset_state()). See NeuronZoo.make."""
    return _ZOO.make(kind, seed=seed)


def sample_population(n, seed=0, kinds=DEFAULT_KINDS):
    """Build a mixed pool of n neurons. Default kinds=('lif',) -> homogeneous, byte-identical to a
    direct LIFNeuron pool. See NeuronZoo.sample."""
    return _ZOO.sample(n, seed=seed, kinds=kinds)


# --------------------------------------------------------------------------------------------------
# THE POPULATION UPDATE -- and the measured gate that decides whether to run it across cores.
#
# A pool could be BUILT here (sample_population) but never STEPPED as a population: every caller
# wrote its own `for cell in pool` loop. That loop is the one embarrassingly-parallel part of this
# brain -- within a timestep each neuron integrates its OWN membrane from its OWN state, so the
# neurons of a population are mutually independent and can run on separate cores. The free-threaded
# build (3.14.6, Py_GIL_DISABLED=1) is what makes that a real option: MEASURED here, pure-Python
# work scales 3.80x on 4 threads and 8.06x on 28, which the GIL made impossible regardless of how
# the loop was written.
#
# WHETHER IT PAYS IS A SEPARATE QUESTION, AND IT IS NOT A QUESTION ABOUT WIDTH. Measured on this
# tree (.venv-lake, 3.14.6 free-threaded, 28 logical CPUs, median of repeated real steps, each
# sweep in a FRESH process -- see the contamination note below):
#
#   pool                              us/neuron   crossover N   best measured
#   homogeneous LIF (DEFAULT_KINDS)      0.10       ~10,000      1.85x @ 10k, 2.76x @ 1,000,000
#   mixed SAFE_KINDS (7 kinds)           0.85        ~1,000      1.26x @ 1k,  2.79x @ 100,000
#   AdEx/Izhikevich 40-substep cells     6.1          ~12-24     4.02x @ 240, 10.03x @ 10,000
#
# The crossover moves by ~400x in N (24 vs 10,000) across neuron kinds this project already uses,
# because what actually decides it is the WORK PER STEP (N x us/neuron) against the measured
# thread-dispatch floor (35.8 us at 4 workers, 65.0 us at 8, 113.9 us at 14 -- medians). A width
# constant would therefore be wrong in BOTH directions: it would thread a 100,000-cell LIF pool
# that loses, and skip a 240-cell AdEx pool that wins 4x. So there is NO width threshold in this
# file. The gate RACES the two paths on the real steps it is already making, on the real pool, and
# keeps whichever measured fastest (P1: derived from a measurement, never a hardcoded arbiter).
#
# THE FIRST VERSION OF THAT TABLE WAS INFLATED AND THE CAUSE IS WORTH KEEPING. It was measured in
# one process that had already allocated 1,000,000-neuron pools earlier in the same run, and the
# resulting heap pressure raised the SERIAL baseline for the fat mix from 6.1 to 14.5 us/neuron --
# so the speedup it reported at 240 cells was 6.61x where a fresh process measures 4.02x. The
# parallel side was not measured wrong; the thing it was divided by was. Re-derived per-sweep in
# separate interpreters, which is why the numbers above are lower at small N and HIGHER at large N
# (10.03x, not 6.93x, at 10,000).
#
# The race is safe to lose: both paths are BYTE-IDENTICAL, so the choice can only change how long a
# step takes, never what it returns. And near the crossover a wrong pick costs almost nothing by
# definition -- the decision is only ambiguous exactly where it does not matter.
#
# DECOMPOSITION -- this is load-bearing and the naive version is much worse than serial. Threads
# must own DISJOINT state: each gets its own slice of the pool and builds its OWN result list.
# MEASURED, homogeneous LIF at N=1,000,000: threads indexing one SHARED pool list and writing one
# SHARED output list ran 0.12x at 8 workers (8x SLOWER than serial, and it got worse with more
# threads -- 0.04x at 14); the same work with disjoint slices ran 1.34x. Same neurons, same thread
# count, ~11x apart. Do not reintroduce a shared output buffer.
#
# THE PARTITION IS AN INDEX RANGE, NOT A CACHED LIST OF CELLS, and that is a CORRECTNESS choice
# found by adversarial review rather than a style one. The first version cached `cells[lo:hi]`
# inside the committed plan, which made the plan a SECOND copy of the pool: replacing a neuron in
# `Population.cells` afterwards left the parallel path stepping the OLD object while the serial
# path stepped the new one. MEASURED as a live divergence -- after committing to 4 threads, a
# sentinel swapped into index 10 returned 0 through the committed path and 1 through the serial
# loop, i.e. the two paths were no longer byte-identical, which is the one property this whole
# mechanism rests on. Slicing per step instead re-reads the pool every time, so a mutated
# population is visible to both paths by construction. It is not a trade: re-slicing was measured
# against the cached form at 7 (mix, N, p) points from 24 to 100,000 cells and is free --
# -18.6% to +4.1%, i.e. inside run-to-run noise and usually FASTER, because one slice per thread
# per step is nothing beside N neuron steps. This is NOT the shared-list pathology above: that was
# per-NEURON indexing into a shared list inside the inner loop; this is a single slice per thread
# that hands each worker its own private list, exactly as before.
# --------------------------------------------------------------------------------------------------
_PROBE_STEPS = 3  # timed real steps per candidate before committing. A SAMPLE SIZE, not a
# threshold: it sets how precisely the race is measured, and arbitrates no behaviour of its own.


def _worker_ladder():
    """Candidate thread counts to race, derived from the machine (never a hardcoded width).

    Powers of two up to the CPU count, plus the CPU count itself. A 1-CPU machine yields an empty
    ladder, so the gate has no parallel candidate to race and stays serial forever."""
    cpus = os.cpu_count() or 1
    ladder = []
    p = 2
    while p < cpus:
        ladder.append(p)
        p *= 2
    if cpus > 1:
        ladder.append(cpus)
    return tuple(ladder)


def _partition(n, p):
    """Split n indices into at most p contiguous, DISJOINT (lo, hi) ranges in index order.

    Ranges, NOT cached cell slices -- see the DECOMPOSITION note above: a plan that caches cell
    objects becomes a stale second copy of the pool the moment a caller replaces a neuron.
    Empty ranges are dropped, so a partition never dispatches a thread with nothing to do (p > n)."""
    edges = [(n * k) // p for k in range(p + 1)]
    return tuple((edges[k], edges[k + 1]) for k in range(p) if edges[k] < edges[k + 1])


def _step_serial(cells, currents):
    """Advance every neuron one timestep, in index order. The reference result."""
    return [nu.step(c) for nu, c in zip(cells, currents)]


def _step_parallel(cells, currents, plan):
    """Advance every neuron one timestep across threads. BYTE-IDENTICAL to _step_serial.

    Identity holds because the population update carries no cross-neuron coupling within a step:
    each neuron reads only its own state and its own current, and each thread writes only its own
    result list. `Executor.map` yields chunks in submission order, so the concatenation below is in
    neuron-index order -- there is no reduction anywhere in this function, and therefore no
    float-addition-order hazard to reorder. (`plan` is the (executor, parts) chosen by the gate.)

    Each worker takes its OWN private slice of `cells` for its range, so the disjoint-state
    decomposition is unchanged -- the slice is simply taken per step rather than cached, which is
    what keeps this path reading the same pool the serial path reads."""
    ex, parts = plan

    def work(part):
        lo, hi = part
        return [nu.step(c) for nu, c in zip(cells[lo:hi], currents[lo:hi])]

    out = []
    for chunk in ex.map(work, parts):
        out.extend(chunk)
    return out


class _PopulationGate:
    """Races serial against each candidate thread count on REAL steps, then commits to the winner.

    Calibration consumes real population steps -- it never double-steps the pool, because stepping
    a neuron MUTATES it and a re-run would advance the simulation twice. Every probe step is a
    genuine step whose result is returned to the caller; only the timing is kept.

    Commits to serial without racing when the population cannot be split (n < 2), when the machine
    offers no second core, or when the pool ALIASES the same neuron object more than once -- that
    last one is a correctness gate, not a performance one: two threads stepping one shared object
    is a data race, and even serially the result depends on how many times that object is stepped,
    so such a pool has no byte-identical parallel form at all."""

    def __init__(self, cells):
        self._cands = None  # [None (serial), p1, p2, ...] -- built lazily on first step
        self._plans = {}  # candidate -> (executor, parts)
        self._times = {}  # candidate -> [elapsed, ...]
        self._order = []  # probe schedule, consumed left to right
        self._n = len(cells)
        self.plan = None  # committed plan; None == serial
        self.committed = True
        # Set here as well as in record(), so a gate that commits AT CONSTRUCTION and never races
        # still answers the same questions as one that did. They were previously assigned only in
        # record(), so `gate.workers` raised AttributeError on exactly the pools that skip the race
        # (n < 2, single-core, aliased) -- a reporting surface that works only on the happy path.
        self.workers = None  # committed thread count; None == serial
        self.speedup = 1.0  # measured serial/parallel ratio; 1.0 when no race was run
        ladder = _worker_ladder()
        if len(cells) < 2:
            self.reason = "population too small to split"
        elif not ladder:
            self.reason = "no second core"
        elif len({id(c) for c in cells}) != len(cells):
            self.reason = "pool aliases the same neuron object (no byte-identical parallel form)"
        else:
            self.committed = False
            self.reason = "calibrating"
            self._cands = [None] + [p for p in ladder if p <= len(cells)]
            self._order = [c for c in self._cands for _ in range(_PROBE_STEPS)]

    def next_candidate(self):
        """The candidate this step must be timed on, and its plan. Call once per calibration step."""
        cand = self._order[0]
        if cand is None:
            return None, None
        plan = self._plans.get(cand)
        if plan is None:
            ex = ThreadPoolExecutor(max_workers=cand, thread_name_prefix="snn-pop")
            plan = (ex, _partition(self._n, cand))
            self._plans[cand] = plan
        return cand, plan

    def record(self, cand, elapsed):
        """Log one probe timing; commit to the fastest candidate once the schedule is exhausted."""
        self._times.setdefault(cand, []).append(elapsed)
        self._order.pop(0)
        if self._order:
            return

        # median of each candidate's probe steps -> the winner. Median, not min: a min rewards a
        # lucky scheduling window, and the question is what this pool costs in the steady state.
        def med(vals):
            s = sorted(vals)
            return s[len(s) // 2]

        # `self._cands` starts with None (serial), and min() keeps the FIRST of equal keys, so an
        # exact tie resolves to serial. Deliberate: threads have to EARN the population, not merely
        # match it.
        best = min(self._cands, key=lambda c: med(self._times[c]))
        ser = med(self._times[None])
        self.plan = self._plans.get(best)
        self.committed = True
        self.speedup = ser / med(self._times[best]) if med(self._times[best]) > 0.0 else 1.0
        self.workers = best
        self.reason = (
            f"serial {ser * 1e6:.1f} us/step"
            if best is None
            else f"{best} threads, {self.speedup:.2f}x over serial ({ser * 1e6:.1f} us/step)"
        )
        for cand_i in list(self._plans):  # release the losers
            if cand_i != best:
                self._plans.pop(cand_i)[0].shutdown(wait=False)

    def close(self):
        """Shut down the retained executor (run_selfchecks runs part of the suite in ONE process)."""
        for plan in list(self._plans.values()):
            if plan is not None:
                plan[0].shutdown(wait=False)
        self._plans.clear()
        self.plan = None


class Population:
    """A pool of heterogeneous neurons advanced as ONE population update.

    `step(currents)` advances every neuron by one timestep and returns the population's 0/1 spike
    vector in neuron-index order. Whether that runs serially or across cores is decided by racing
    the two on this pool's own real steps (see _PopulationGate); both paths are byte-identical, so
    the decision is a pure performance one and cannot reach the returned spikes.

    Owns its cells so the partition can be cached safely -- a module-level cache keyed on the pool
    object would be a correctness hazard, since `id()` is reused after garbage collection. `cells`
    is the ONE list both paths read: the committed plan holds index RANGES, never cell objects, so
    replacing a neuron in `cells` is seen by the serial and parallel paths alike. A caller that
    wants the pool it passed in to stay authoritative should read `Population.cells` back (this is
    what `HeterogeneousLayer` does) -- the constructor copies the sequence it is given.

    THREE LIMITATIONS, STATED RATHER THAN HIDDEN, because a thread pool that quietly turns itself
    off is the defect class this project hunts:
      * `step()` is single-caller. It parallelises WITHIN one population update; it does not make a
        population safe to step from two threads at once, which would double-advance every neuron
        and race the calibration bookkeeping regardless of any locking here.
      * `close()` releases the worker threads and leaves the population running SERIALLY from then
        on. Results stay correct -- both paths are byte-identical -- but a closed population is a
        slow one, and it will not re-race itself.
      * the pool's LENGTH is fixed at construction (`n`), so growing or shrinking `cells` in place
        makes `step()` reject the new currents vector rather than silently step the wrong count.
        Replacing a neuron is supported; resizing means building a new `Population`."""

    def __init__(self, cells):
        self.cells = list(cells)
        self.n = len(self.cells)
        self.gate = _PopulationGate(self.cells)

    def step(self, currents):
        """Advance the whole population one timestep. `currents` must hold exactly n drives."""
        if len(currents) != self.n:
            raise ValueError(
                f"currents must hold exactly one drive per neuron (got {len(currents)} for {self.n})"
            )
        if self.n == 0:
            return []
        gate = self.gate
        cells = self.cells
        if gate.committed:
            plan = gate.plan
            return (
                _step_serial(cells, currents)
                if plan is None
                else _step_parallel(cells, currents, plan)
            )
        cand, plan = gate.next_candidate()
        t0 = time.perf_counter()
        out = (
            _step_serial(cells, currents) if plan is None else _step_parallel(cells, currents, plan)
        )
        gate.record(cand, time.perf_counter() - t0)
        return out

    def close(self):
        """Release any worker threads this population retained."""
        self.gate.close()


# --------------------------------------------------------------------------------------------------
# self-check (deterministic asserts, ASCII-only)
# --------------------------------------------------------------------------------------------------
def _train(neuron, currents):
    """Drive one neuron with a current stream from rest; return its 0/1 spike train."""
    neuron.reset_state()
    return [neuron.step(c) for c in currents]


def main():
    drive_c, T = 0.2, 80
    currents = [drive_c] * T

    # 1. HETEROGENEITY: one common drive into a MIXED, default-safe pool -> DIFFERENT firing patterns.
    het = sample_population(len(SAFE_KINDS) * 2, seed=3, kinds=SAFE_KINDS)
    trains = [_train(nu, currents) for nu in het]
    counts = [sum(tr) for tr in trains]
    all_bits_binary = all(s in (0, 1) for tr in trains for s in tr)
    distinct_counts = sorted(set(counts))
    # per-kind representative spike counts (first cell of each kind, in SAFE_KINDS order)
    per_kind = {}
    for k in SAFE_KINDS:
        nu = make_neuron(k, seed=3)
        per_kind[k] = sum(_train(nu, currents))

    # 2. BYTE-IDENTICAL HOMOGENEOUS LIF: default pool must equal a direct LIFNeuron pool, bit for bit.
    homo = sample_population(6, seed=3)  # default kinds = ('lif',)
    homo_trains = [_train(nu, currents) for nu in homo]
    direct_trains = [_train(LIFNeuron(), currents) for _ in range(6)]
    byte_identical = homo_trains == direct_trains
    homo_uniform = len(set(sum(tr) for tr in homo_trains)) == 1  # homogeneous -> all identical

    # 3. NaN GUARD (opt-in biophysical): pathological drives must never raise, never return non-0/1,
    #    never leave the state non-finite; the cell recovers and fires again on a normal drive.
    pathological = [float("nan"), float("inf"), -float("inf"), 1e30, -1e30]
    guard_ok = True
    hh_finite = izhi_finite = True
    for kind in OPTIN_KINDS:
        cell = make_neuron(kind, seed=1)
        cell.reset_state()
        for _ in range(20):  # warm it up on a normal drive first
            cell.step(0.3)
        for bad in pathological:
            s = cell.step(bad)
            if s not in (0, 1):
                guard_ok = False
        # state stayed finite through the abuse
        if kind == "hh":
            hh_finite = _finite(cell.raw.v)
        else:
            izhi_finite = _finite(cell.v) and _finite(cell.u)
        # recovers: still steps to valid 0/1 bits on a normal drive afterwards
        post = [cell.step(0.3) for _ in range(30)]
        if any(s not in (0, 1) for s in post):
            guard_ok = False
    # opt-in cells CAN spike under a benign drive (so they are live, not merely silent-guarded)
    hh_spikes = sum(_train(make_neuron("hh", seed=1), [0.4] * 400))
    izhi_spikes = sum(_train(make_neuron("izhikevich", seed=1), [0.5] * 400))

    # 4. DETERMINISM: same seed + same drive -> byte-identical pool trains.
    het_again = [
        _train(nu, currents)
        for nu in sample_population(len(SAFE_KINDS) * 2, seed=3, kinds=SAFE_KINDS)
    ]
    deterministic = het_again == trains

    # 5. POPULATION UPDATE: the parallel path must be BYTE-IDENTICAL to the serial one across MANY
    #    steps -- comparing a single step would not notice a thread that corrupted a neuron's state,
    #    because the damage only shows on the NEXT step. 24 steps of a 7-kind pool are compared bit
    #    for bit, which is a claim about the neurons' internal state as much as about the output.
    #
    #    THE DRIVE IS GRADED BY INDEX ON PURPOSE, and the first version of this check was WRONG
    #    because it was not. A near-uniform weak drive (0.15 + 0.001*((i+t)%11)) left almost every
    #    cell silent, so the spike vector was essentially all zeros -- and a PERMUTATION of an
    #    all-zero vector equals itself. MEASURED: the misordered-concatenation red below did not
    #    fire under that drive, i.e. the identity assert was passing vacuously. The drive now spans
    #    0.05..0.65 across the pool so neighbouring neurons fire at visibly different rates, and
    #    pop_n is coprime with both the partition count and the 7-kind round robin so no partition
    #    boundary lands on a repeating pattern.
    pop_n, pop_t = 143, 24
    stream = [[0.05 + 0.02 * ((i + t) % 31) for i in range(pop_n)] for t in range(pop_t)]

    def _fresh():
        return sample_population(pop_n, seed=9, kinds=SAFE_KINDS)

    ser_out = [_step_serial(_p, cur) for _p in [_fresh()] for cur in stream]
    ex = ThreadPoolExecutor(max_workers=4)
    par_pool = _fresh()
    par_plan = (ex, _partition(len(par_pool), 4))
    par_out = [_step_parallel(par_pool, cur, par_plan) for cur in stream]
    par_identical = par_out == ser_out

    # REACHABLE RED for that assert. The decomposition's real failure mode is chunks concatenated
    # out of neuron-index order -- the one way a correct-looking parallel population step silently
    # returns a permuted spike vector. Build exactly that defect and RUN it: the check must go red.
    def _step_parallel_misordered(cells, currents, plan):
        ex_, parts = plan

        def work(part):
            lo, hi = part
            return [nu.step(c) for nu, c in zip(cells[lo:hi], currents[lo:hi])]

        out = []
        for chunk in reversed(list(ex_.map(work, parts))):  # THE DEFECT: wrong concatenation order
            out.extend(chunk)
        return out

    red_pool = _fresh()
    red_plan = (ex, _partition(len(red_pool), 4))
    red_out = [_step_parallel_misordered(red_pool, cur, red_plan) for cur in stream]
    red_fires = red_out != ser_out  # the identity assert IS falsifiable

    # the GATED path (whatever the race picks on this machine) must also match, bit for bit
    gated = Population(_fresh())
    gate_out = [gated.step(cur) for cur in stream]
    gate_identical = gate_out == ser_out
    gate_committed = gated.gate.committed  # the race must terminate within the probe schedule

    # THE COMMITTED PLAN MUST NOT BE A SECOND COPY OF THE POOL. Found by adversarial review: an
    # earlier form cached `cells[lo:hi]` inside the plan, so once the race committed to threads a
    # neuron REPLACED in `Population.cells` was still stepped as its old object -- the parallel and
    # serial paths silently stopped agreeing, which is the single property this mechanism rests on.
    #
    # The plan is INSTALLED here rather than raced for, and that is the point: this pool is mid-
    # grain and well below its crossover, so on this machine the race correctly picks serial -- and
    # a serial path cannot exhibit the defect, which would have made the check pass vacuously
    # everywhere. (It did, on the first run: the assert went red only because the reference twin
    # below caught it.) Forcing the parallel branch is what makes the property testable on any
    # machine instead of only on one wide enough for threads to win.
    mut_n, mut_i = 64, 32

    class _AlwaysFire:  # a sentinel that cannot be confused with a real cell's output
        def step(self, c):
            return 1

        def reset_state(self):
            pass

    def _forced_parallel(pool):
        pop = Population(pool)
        pop.gate.plan = (ex, _partition(mut_n, 4))  # bypass the race: exercise the PARALLEL branch
        pop.gate.committed = True
        return pop

    mutated = _forced_parallel(sample_population(mut_n, seed=4, kinds=SAFE_KINDS))
    twin = _forced_parallel(sample_population(mut_n, seed=4, kinds=SAFE_KINDS))
    for _ in range(6):  # identical history, so the two differ only by the swap
        mutated.step([0.35] * mut_n)
        twin.step([0.35] * mut_n)
    mutated.cells[mut_i] = _AlwaysFire()
    mut_out = mutated.step([0.0] * mut_n)
    twin_out = twin.step([0.0] * mut_n)  # the SAME pool, unswapped -- the attribution control
    # the 1 must come from the sentinel, not from a cell that would have fired anyway
    swap_seen = mut_out[mut_i] == 1 and twin_out[mut_i] == 0 and mutated.gate.plan is not None

    # REACHABLE RED: rebuild the cached-slice plan that used to ship and run the same swap through
    # it. It must MISS the sentinel -- proving the check above can fail, and that the fix is what
    # makes it pass rather than the parallel branch never having been taken.
    stale_pool = sample_population(mut_n, seed=4, kinds=SAFE_KINDS)
    stale_parts = tuple(
        (stale_pool[lo:hi], lo, hi) for lo, hi in _partition(mut_n, 4)
    )  # THE DEFECT: cell objects captured at plan time

    def _step_stale(currents):
        out = []
        for chunk in ex.map(
            lambda pt: [nu.step(c) for nu, c in zip(pt[0], currents[pt[1] : pt[2]])], stale_parts
        ):
            out.extend(chunk)
        return out

    for _ in range(6):
        _step_stale([0.35] * mut_n)
    stale_pool[mut_i] = _AlwaysFire()
    stale_out = _step_stale([0.0] * mut_n)
    stale_red_fires = stale_out[mut_i] != 1  # the cached form does NOT see the swap

    # baselines at zero / one, and the two pools that have NO byte-identical parallel form
    zero_ok = Population([]).step([]) == []
    one = Population(sample_population(1, seed=9))
    one_ok = one.step([0.5]) in ([0], [1]) and one.gate.plan is None
    # a gate that commits AT CONSTRUCTION still answers workers/speedup (they used to be set only
    # by record(), so these attributes existed only on pools that raced)
    never_raced_reports = one.gate.workers is None and one.gate.speedup == 1.0
    _dup = make_neuron("lif", seed=2)
    aliased = Population([_dup, _dup, _dup, _dup])
    alias_serial = aliased.gate.committed and aliased.gate.plan is None
    try:
        Population(sample_population(4, seed=1)).step([0.1, 0.2])
        rejects_bad_len = False
    except ValueError:
        rejects_bad_len = True

    print("=" * 74)
    print("NEURON ZOO -- one heterogeneous point-neuron population, one uniform step() contract")
    print("=" * 74)
    print(f"registered kinds : {list(_ZOO.kinds())}")
    print(f"safe (default)   : {list(SAFE_KINDS)}")
    print(f"opt-in (guarded) : {list(OPTIN_KINDS)}  (excluded from the default population)")
    print(
        "per-kind spikes  : "
        + ", ".join(f"{k}={per_kind[k]}" for k in SAFE_KINDS)
        + f"  (I={drive_c}, T={T})"
    )
    print(
        f"heterogeneity    : {len(distinct_counts)} distinct spike-counts across the mixed pool {distinct_counts}"
    )
    print(f"homogeneous LIF  : zoo pool == direct LIFNeuron pool : {byte_identical}")
    print(
        f"NaN guard        : opt-in cells survived {pathological} -> still 0/1, state finite : {guard_ok}"
    )
    print(
        f"opt-in live      : hh {hh_spikes} spikes, izhikevich {izhi_spikes} spikes under a benign drive"
    )
    print(f"deterministic    : same seed -> identical pool trains : {deterministic}")
    print(
        f"population step  : {pop_n} cells x {pop_t} steps, parallel == serial bit for bit : "
        f"{par_identical}  (misordered-concat red fires : {red_fires})"
    )
    print(f"gate (raced)     : {gated.gate.reason}")
    print(
        f"pool not copied  : neuron swapped into a COMMITTED pool is stepped : {swap_seen}  "
        f"(cached-slice red fires : {stale_red_fires})"
    )
    print("=" * 74)

    # ---- asserts ----
    assert all_bits_binary, "every neuron.step must return a 0/1 spike bit"
    assert len(distinct_counts) >= 3, (
        f"a mixed pool must show heterogeneous firing ({distinct_counts})"
    )
    assert per_kind["lif"] != per_kind["alif"], (
        "adaptation (ALIF) must differ from plain LIF firing"
    )
    assert not all(tr == trains[0] for tr in trains), (
        "a heterogeneous pool must not fire one identical train"
    )
    assert byte_identical, (
        "homogeneous-LIF zoo pool must be BYTE-IDENTICAL to a direct LIFNeuron pool"
    )
    assert homo_uniform, "a homogeneous LIF pool must fire one identical count across cells"
    assert guard_ok, (
        "opt-in biophysical cells must be NaN-guarded (never non-0/1, never raise, recover)"
    )
    assert hh_finite and izhi_finite, "opt-in cell state must stay finite through non-finite drives"
    assert hh_spikes > 0 and izhi_spikes > 0, (
        "opt-in cells must be able to spike under a benign drive"
    )
    assert deterministic, "same seed + drive -> identical pool trains (deterministic)"
    # -- population update / parallelism --
    assert par_identical, (
        f"the parallel population step must be BYTE-IDENTICAL to the serial one over {pop_t} steps"
    )
    assert red_fires, (
        "the byte-identity check must be FALSIFIABLE: a parallel step whose chunks are concatenated "
        "out of index order must be detected, or the assert above proves nothing"
    )
    assert gate_identical, (
        "the raced gate must return exactly the serial result, whichever path it picked"
    )
    assert gate_committed, (
        f"the race must terminate within its own probe schedule ({pop_t} steps was not enough)"
    )
    assert zero_ok, "an empty population must step to an empty spike vector, not raise"
    assert one_ok, "a 1-neuron population must step, and must never be split across threads"
    assert alias_serial, (
        "a pool that aliases one neuron object has no byte-identical parallel form and must be "
        "pinned to serial"
    )
    assert rejects_bad_len, "step() must refuse a currents vector that is not one drive per neuron"
    assert swap_seen, (
        "the committed plan must hold index RANGES, not cell objects: a neuron replaced in "
        "Population.cells must be stepped by the committed path, or the parallel and serial paths "
        "have silently stopped reading the same pool"
    )
    assert stale_red_fires, (
        "that check must be FALSIFIABLE: the cached-slice plan it replaced must MISS the swapped "
        "neuron, or passing proves nothing about which form is in use"
    )
    assert never_raced_reports, (
        "a gate that commits at construction must still report workers/speedup, not raise "
        "AttributeError on the pools that never race"
    )
    gated.close()
    one.close()
    aliased.close()
    ex.shutdown()  # `mutated`/`twin` share this executor via their installed plans

    print(
        f"self-check OK: {len(SAFE_KINDS)} safe + {len(OPTIN_KINDS)} opt-in kinds; heterogeneous pool "
        f"{distinct_counts} distinct counts; homogeneous LIF byte-identical; HH/Izhikevich NaN-guarded "
        f"(hh {hh_spikes}, izhi {izhi_spikes} spikes); deterministic; population step parallel == "
        f"serial over {pop_n}x{pop_t} (red fires), swapped cell seen by the committed plan "
        f"(cached-slice red fires), gate: {gated.gate.reason}"
    )


if __name__ == "__main__":
    main()
