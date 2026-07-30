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

The zoo REUSES (import + adapt, never rewrite) the repo's own neuron models:
  * cortex.lif_neuron.LIFNeuron        -- leaky integrate-and-fire      (native step/reset_state)
  * cortex.alif_neuron.ALIFNeuron      -- adaptive-threshold LIF (LSNN) (native step/reset_state)
  * cortex.autapse_neuron.AutapseNeuron-- one neuron, delayed self-loops(native step/reset_state)
  * cortex.dendritic_neuron.DendriticNeuron -- two-compartment BAC cell (2-arg step -> adapter)
  * reasoning.adex_neuron.AdEx         -- adaptive exponential I&F      (units + reset() -> adapter)
  * reasoning.resonate_and_fire.ResonateFire -- frequency-selective     (units + reset() -> adapter)
  * reasoning.srm_neuron.SRMNeuron     -- kernel/soft-reset spike model (event API -> adapter)

DEFAULT-SAFE population = the six always-finite point/reduced models above (SRM is a
kernel model included in the safe set). Two BIOPHYSICAL models integrate forward-Euler ODEs that can
diverge, so they are OPT-IN ONLY and NaN-GUARDED (a non-finite drive or a blown-up state is clamped /
skipped, never propagated, never raised) and are EXCLUDED from the default population:
  * cortex.hodgkin_huxley.HodgkinHuxley-- conductance-based biophysical (V trace -> spike + NaN guard)
  * cortex.izhikevich_neuron           -- 2-var reduced (batch-only module -> stepping wrapper + guard)

The DEFAULT population kind set is ("lif",) only, so a pool built with the defaults is homogeneous LIF
and BYTE-IDENTICAL to today's direct LIFNeuron pools -- adopting the zoo changes nothing until a caller
explicitly asks for a heterogeneous `kinds=...`. Heterogeneity is opt-in.

SAFETY / SCOPE: this module is intent-recognition/substrate only -- it builds neurons and returns spike
bits. It adds NO actuator route. A heterogeneous READOUT can, however, change which intent wins an
argmax, so wiring it into the interaction-learning intent readout is ARM-ADJACENT and must be E-STOP
re-verified by the main agent (see the integration-hook spec returned with this build).

Deterministic: per-instance random.Random(seed); no module-level seeding, no
wall-clock. ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.neuron_zoo
"""

import math
import random

from snn_data_lake.cortex.alif_neuron import ALIFNeuron
from snn_data_lake.cortex.autapse_neuron import AutapseNeuron
from snn_data_lake.cortex.dendritic_neuron import DendriticNeuron
from snn_data_lake.cortex.hodgkin_huxley import HodgkinHuxley
from snn_data_lake.cortex.izhikevich_neuron import REGIMES as _IZHI_REGIMES
from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.reasoning.adex_neuron import AdEx
from snn_data_lake.reasoning.resonate_and_fire import ResonateFire
from snn_data_lake.reasoning.srm_neuron import SRMNeuron

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

    print(
        f"self-check OK: {len(SAFE_KINDS)} safe + {len(OPTIN_KINDS)} opt-in kinds; heterogeneous pool "
        f"{distinct_counts} distinct counts; homogeneous LIF byte-identical; HH/Izhikevich NaN-guarded "
        f"(hh {hh_spikes}, izhi {izhi_spikes} spikes); deterministic"
    )


if __name__ == "__main__":
    main()
