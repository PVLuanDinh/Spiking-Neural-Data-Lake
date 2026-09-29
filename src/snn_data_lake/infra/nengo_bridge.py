"""
Nengo bridge -- run the brain's NEF / spiking PRIMITIVES on the Nengo simulator (deployable to NengoLoihi).

HONEST framing (read this first):
  * This bridges the project's Neural-Engineering-Framework (NEF) and LIF spiking PRIMITIVES to Nengo, so
    the brain's substrate can run on `nengo.Simulator` and is DEPLOYABLE to NengoLoihi neuromorphic
    hardware. Loihi silicon itself is a deploy TARGET -- it is NOT exercised or verified here.
  * It is NOT a re-implementation of the ~200 brain-region modules in Nengo. It is the GENERIC vector /
    spike <-> nengo.Ensemble bridge that ANY module routes a computation THROUGH (a module hands the
    bridge a vector + a function; the bridge encodes it in a spiking population, decodes f(x), and can run
    the whole thing on Nengo / export it to a neuromorphic chip). Verified faithful on the NEF (represent /
    transform) and LIF (rate) primitives -- the substrate the rest sits on.
  * The NEF is a function APPROXIMATOR. Fidelity is within a tolerance, never exact. A finite population
    decodes with a small residual error (the classic NEF ~1/N scaling); we assert tolerance bands, not
    bit-equality (the Nengo sim is numpy/float, seeded for reproducibility, not deterministic to the bit).

Ground truth it MATCHES (reused, not re-implemented):
  * instruments/nef_compute.py     -- the in-tree NEF (represent x, transform x -> x^2; promoted 2026-08-12,
    was reasoning/ and before that research/). Nengo is configured
    with the SAME LIF constants (tau_rc=0.02, tau_ref=0.002), intercepts and max-rates so the Nengo decode
    is COMPARABLE to the zero-dep decode: same framework, different engine.
  * cortex/lif_neuron.LIFNeuron -- the canonical discrete-time LIF. `lif_to_nengo` builds a Nengo LIF whose
    firing rate tracks this neuron's rate across an input-current sweep.
  * sensorimotor/population_encoding, connectome.py -- `to_nengo_network` exports a population front-end (or a
    recurrent line-attractor loop) as an equivalent nengo.Network that runs and emits spikes.

GATED TIER (like the repo's torch tier): `import nengo` is guarded in try/except so this file still IMPORTS
with nengo absent -- the bridge functions then raise a clear RuntimeError and the self-check prints SKIP and
exits 0. The CORE never imports this module.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.infra.nengo_bridge   # self-check (needs nengo)
"""

import math

try:
    import nengo
    import numpy as np

    # Disable Nengo's on-disk decoder cache (~/.nengo/cache, shared across ALL venvs by the same user).
    # It pickles solved decoders keyed only by the problem -- NOT by numpy version -- so a cache entry
    # written by one env's numpy (e.g. .venv-nengo's numpy 2.x, module path `numpy._core`) crashes with
    # ModuleNotFoundError when read back under another env's numpy (.venv-cu128's 1.26, path `numpy.core`).
    # Decoders are cheap and deterministically re-solved from the seed, so disabling the cache makes the
    # bridge deterministic + env-robust with no effect on any decode/assertion. See docs -- gated tier.
    nengo.rc["decoder_cache"]["enabled"] = "False"
    _NENGO_OK = True
    _IMPORT_ERR = None
except ImportError as _e:  # gated: import stays clean when the neuromorphic stack is absent
    nengo = None
    np = None
    _NENGO_OK = False
    _IMPORT_ERR = _e

# Nengo's LIF defaults ARE the project's NEF constants (instruments/nef_compute: TAU_RC, T_REF).
TAU_RC = 0.02  # membrane time constant (s)   -- Nengo default == nef_compute.TAU_RC
TAU_REF = 0.002  # refractory period (s)        -- Nengo default == nef_compute.T_REF


_UNSET = (
    object()
)  # sentinel: run(func=_UNSET) uses the transform()-stored func; func=None forces identity


def _require_nengo():
    if not _NENGO_OK:
        raise RuntimeError(
            "nengo_bridge is a GATED tier and needs `nengo` (pip install nengo, e.g. into .venv-cu128). "
            "It is deliberately import-guarded so the core stays clean; install nengo to use the "
            "bridge or deploy to NengoLoihi."
        ) from _IMPORT_ERR


# ----------------------------------------------------------------------------------------------------------
# 1. NEF COMPUTE BRIDGE -- represent / transform / (recurrent) dynamics on a nengo.Ensemble
# ----------------------------------------------------------------------------------------------------------
class NengoNEF:
    """NEF principles 1-3 on a Nengo ensemble -- the generic vector-computation substrate any brain module
    offloads to (and thus deploys to Loihi):

        represent(vec)          principle 1: drive the population to encode `vec`, decode it back (identity)
        transform(func)         principle 2: store the decoded function f(x) the output connection computes
        run(vec, func, t)       run the spiking Nengo sim and return the decoded result (f(x))

    Configured to MATCH instruments/nef_compute (same LIF tau_rc/tau_ref, intercepts in [-0.95, 0.95],
    max-rates in [100, 300] Hz, radius) so the Nengo decode error is COMPARABLE to the zero-dep NEF's own.
    """

    def __init__(
        self,
        dims=1,
        n_neurons=80,
        seed=0,
        radius=1.0,
        max_rates=(100.0, 300.0),
        intercepts=(-0.95, 0.95),
        tau_rc=TAU_RC,
        tau_ref=TAU_REF,
        dt=0.001,
    ):
        _require_nengo()
        self.dims = dims
        self.n_neurons = n_neurons
        self.seed = seed  # per-instance seed (no import-time / global seeding)
        self.radius = radius
        self.dt = dt
        self._nt = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
        self._max = nengo.dists.Uniform(*max_rates)
        self._int = nengo.dists.Uniform(*intercepts)
        self._func = None  # set by transform()

    def _ensemble(self):
        return nengo.Ensemble(
            self.n_neurons,
            self.dims,
            radius=self.radius,
            neuron_type=self._nt,
            max_rates=self._max,
            intercepts=self._int,
        )

    def transform(self, func):
        """NEF principle 2: store the function the decoded output connection computes. Chainable."""
        self._func = func
        return self

    def represent(self, vec, t=0.3, settle=0.5):
        """NEF principle 1: encode `vec` in the spiking population and decode identity back."""
        return self.run(vec, func=None, t=t, settle=settle)

    def run(self, vec, func=_UNSET, t=0.3, settle=0.5):
        """Drive the ensemble with `vec`, decode `func(x)` off a spiking Nengo run, and return the
        steady-state decoded value (mean over the last `settle` fraction). `func` unset -> the
        transform()-stored function; func=None -> identity (represent)."""
        decoded, _spikes = self.run_full(vec, func=func, t=t, settle=settle)
        return decoded

    def run_full(self, vec, func=_UNSET, t=0.3, settle=0.5):
        """As run(), also returning the spike-event count -> (decoded, spikes)."""
        _require_nengo()
        func = (
            self._func if func is _UNSET else func
        )  # _UNSET -> stored transform; None -> identity
        v = np.atleast_1d(np.asarray(vec, dtype=float))
        sample = func(v) if func is not None else v
        odim = int(np.atleast_1d(np.asarray(sample, dtype=float)).size)
        with nengo.Network(seed=self.seed) as net:
            stim = nengo.Node(v)  # constant input vector
            ens = self._ensemble()
            out = nengo.Node(size_in=odim)  # passthrough that collects the decode
            nengo.Connection(stim, ens, synapse=None)
            if func is None:
                nengo.Connection(ens, out, synapse=0.01)
            else:
                nengo.Connection(ens, out, function=func, synapse=0.01)
            p_out = nengo.Probe(out, synapse=0.02)  # low-pass the decoded output (steady state)
            p_spk = nengo.Probe(ens.neurons)  # raw spikes
        with nengo.Simulator(net, dt=self.dt, progress_bar=False) as sim:
            sim.run(t)
        data = sim.data[p_out]
        k = max(1, int(len(data) * settle))
        decoded = data[-k:].mean(axis=0)
        spikes = int(np.count_nonzero(sim.data[p_spk]))
        decoded = float(decoded[0]) if odim == 1 else decoded.tolist()
        return decoded, spikes


# ----------------------------------------------------------------------------------------------------------
# 2. LIF EXPORT -- a Nengo LIF configured to MATCH cortex/lif_neuron.LIFNeuron
# ----------------------------------------------------------------------------------------------------------
class _NengoLIFMatch:
    """A Nengo LIF neuron matched to the brain's discrete-time LIFNeuron(tau_m, v_threshold).

    Mapping: the brain neuron steps  v <- v * exp(-1/tau_m) + current  (dt = 1 step, v_reset = v_rest = 0).
    Over a unit step the exact-leak Euler map with a constant per-step increment `current` equals a
    continuous LIF (tau_rc = tau_m * dt seconds) driven by  J = current / (1 - exp(-1/tau_m)) / v_threshold.
    So a brain per-step current maps to a Nengo injected current via `current_for`, and the matched Nengo LIF
    fires at the same rate (see the self-check's tolerance band). A refractory of R brain steps maps to
    tau_ref = R * dt.
    """

    def __init__(self, tau_m, v_threshold, dt, tau_ref, seed):
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.dt = dt
        self.tau_ref = tau_ref
        self.seed = seed
        self.tau_rc = tau_m * dt
        self.decay = math.exp(-1.0 / tau_m)
        self.neuron_type = nengo.LIF(tau_rc=self.tau_rc, tau_ref=tau_ref)

    def current_for(self, brain_current):
        """Map the brain LIFNeuron's per-step input increment to the equivalent Nengo injected current J."""
        return brain_current / (1.0 - self.decay) / self.v_threshold

    def rate(self, brain_current):
        """Analytic Nengo LIF steady-state rate (Hz) for a brain per-step current. NOTE: the closed-form
        rate over-estimates the discrete brain neuron at high rates (integer inter-spike-interval
        quantization); `sim_rate` -- which discretizes at dt like the brain does -- is the faithful match."""
        j = self.current_for(brain_current)
        if j <= 1.0:
            return 0.0
        return float(self.neuron_type.rates(np.array([j]), np.array([1.0]), np.array([0.0]))[0])

    def sim_rate(self, brain_current, t=3.0):
        """Empirical firing rate (Hz) of a single matched Nengo LIF neuron with current injected directly,
        run on nengo.Simulator -- 'the Nengo LIF ensemble's firing rate'."""
        j = self.current_for(brain_current)
        with nengo.Network(seed=self.seed) as net:
            stim = nengo.Node(j)
            e = nengo.Ensemble(
                1,
                1,
                neuron_type=self.neuron_type,
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([0.0]),
                encoders=nengo.dists.Choice([[1]]),
            )
            nengo.Connection(
                stim, e.neurons, synapse=None
            )  # inject current straight into the neuron
            ps = nengo.Probe(e.neurons)
        with nengo.Simulator(net, dt=self.dt, progress_bar=False) as sim:
            sim.run(t)
        return int(np.count_nonzero(sim.data[ps])) / t


def lif_to_nengo(tau_m=20.0, v_threshold=1.0, dt=0.001, tau_ref=0.0, seed=0):
    """Build a Nengo LIF configured to MATCH cortex/lif_neuron.LIFNeuron(tau_m, v_threshold). Returns a
    helper exposing `.neuron_type` (drop into a nengo.Ensemble), `.current_for`, `.rate` (analytic) and
    `.sim_rate` (empirical, run on the simulator). Defaults mirror LIFNeuron's (v_threshold=1, no refractory)."""
    _require_nengo()
    return _NengoLIFMatch(tau_m, v_threshold, dt, tau_ref, seed)


# ----------------------------------------------------------------------------------------------------------
# 3. CIRCUIT EXPORT -- a small brain spiking circuit as an equivalent nengo.Network
# ----------------------------------------------------------------------------------------------------------
def to_nengo_network(
    value=0.5,
    n_neurons=100,
    lo=-1.0,
    hi=1.0,
    seed=0,
    tau_rc=TAU_RC,
    tau_ref=TAU_REF,
    recurrent=False,
):
    """Export a small brain spiking circuit to an equivalent nengo.Network that RUNS on nengo.Simulator and
    emits spikes.

      default (recurrent=False): a POPULATION FRONT-END -- a spiking population representing a scalar in
        [lo, hi] (the Nengo analogue of sensorimotor/population_encoding; both encode a scalar robustly across a
        population, though Nengo tiles with LIF tuning curves rather than the RBF Gaussian fields).
      recurrent=True: a NEF line-attractor INTEGRATOR -- the connectome recurrent loop of
        research/nef_integrator (identity feedback) that HOLDS / integrates its state.

    Returns (network, probes) where probes = {'decoded', 'spikes', 'offset'}; feed to `simulate()`."""
    _require_nengo()
    radius = (hi - lo) / 2.0
    mid = (hi + lo) / 2.0
    x = float(value) - mid  # center the value into [-radius, radius]
    nt = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(
            n_neurons,
            1,
            radius=radius,
            neuron_type=nt,
            max_rates=nengo.dists.Uniform(100, 300),
            intercepts=nengo.dists.Uniform(-0.95, 0.95),
        )
        if recurrent:
            syn = 0.1  # recurrent synapse == integration time constant
            nengo.Connection(nengo.Node(x), ens, transform=syn, synapse=syn)
            nengo.Connection(
                ens, ens, synapse=syn
            )  # identity feedback -> line attractor (holds/integrates)
        else:
            nengo.Connection(nengo.Node(x), ens, synapse=None)
        p_dec = nengo.Probe(ens, synapse=0.02)
        p_spk = nengo.Probe(ens.neurons)
    return net, {"decoded": p_dec, "spikes": p_spk, "offset": mid}


def simulate(net, probes, t=0.3, dt=0.001, settle=0.5):
    """Run an exported network on nengo.Simulator for `t` seconds; return (decoded_scalar, spike_count)."""
    _require_nengo()
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(t)
    dec = sim.data[probes["decoded"]]
    k = max(1, int(len(dec) * settle))
    decoded = float(dec[-k:].mean(axis=0)[0]) + probes.get("offset", 0.0)
    spikes = int(np.count_nonzero(sim.data[probes["spikes"]]))
    return decoded, spikes


# ----------------------------------------------------------------------------------------------------------
# 4. WHOLE-BRAIN OBSERVATION -- expose the ENTIRE brain (every region + the torch tiers) to Nengo
# ----------------------------------------------------------------------------------------------------------
def brain_signals(brain):
    """The numeric observable surface of a brain: every scalar in brain.state() (bool -> 0/1; each `*_tier`
    string -> 1.0 when a non-stdlib tier is active else 0.0). This surface spans ALL tiers -- the
    spiking regions AND the torch tiers via their EXPORTED signals (perception_tier, pc_cortex_tier,
    nengo_tier, grown/reasoning units, access index, novelty, ...). Returns an ordered {name: float}.
    HONEST: a torch organ exports a derived scalar / tier-active flag here, NOT its raw internal spikes."""
    out = {}
    for k, v in brain.state().items():
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, str) and k.endswith("_tier"):
            out[k] = (
                0.0 if v == "stdlib" else 1.0
            )  # torch/nengo/pc tier ACTIVE as an observable channel
    return out


def to_nengo_brain(brain, tick=True, n_neurons=30):
    """Build a nengo.Network that OBSERVES THE WHOLE BRAIN. A live-tap Node reads brain.state() each sim step
    (ticking the brain once per step when tick=True) and every numeric signal is REPRESENTED in its own
    spiking NEF Ensemble, with a decoded Probe and a neuron (spike) Probe. Running nengo.Simulator then records
    every organ's signal over time as Nengo spikes -- torch tiers INCLUDED, via their exported scalars / tier
    flags. Deployable to NengoLoihi as a whole-brain monitoring layer. Returns (net, dec_probes, spk_probes,
    keys, scales).

    HONEST framing: Nengo does NOT run the brain's computation (the brain is custom stdlib + torch). It
    OBSERVES and re-encodes the brain's OUTPUT signals into spiking populations. The spiking regions
    and the pc_cortex serve export real activity; the torch organs (SpikeGPT language center, neurogenesis,
    ng_snn, surrogate perception) export a derived scalar / tier-active flag, not their internal spikes --
    observing those requires entering the torch process. This is the whole-brain OBSERVABILITY seam."""
    _require_nengo()
    sig0 = brain_signals(brain)
    keys = list(sig0)
    if not keys:
        raise RuntimeError("brain.state() exposed no numeric signals to observe")
    scales = [
        max(1.0, 1.5 * abs(sig0[k])) for k in keys
    ]  # per-signal NEF radius (bounds the signal)
    net = nengo.Network(label="whole-brain observation", seed=0)
    dec_probes, spk_probes = {}, {}
    with net:

        def _tap(t):
            if tick:
                brain.tick()
            s = brain_signals(brain)
            return [s.get(k, 0.0) for k in keys]

        drive = nengo.Node(_tap, size_out=len(keys))
        for i, k in enumerate(keys):
            ens = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, radius=scales[i], seed=0)
            nengo.Connection(drive[i], ens, synapse=0.01)
            dec_probes[k] = nengo.Probe(ens, synapse=0.02)
            spk_probes[k] = nengo.Probe(ens.neurons)
    return net, dec_probes, spk_probes, keys, scales


def observe_brain(brain, t=0.05, dt=0.001, tick=True):
    """Run the whole-brain observation network on nengo.Simulator for `t` s. Returns ({name: mean decoded
    signal}, total spikes across all ensembles, keys). The brain ticks once per sim step when tick=True."""
    _require_nengo()
    net, dec_probes, spk_probes, keys, _ = to_nengo_brain(brain, tick=tick)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(t)
    decoded = {}
    for k in keys:
        d = sim.data[dec_probes[k]]
        decoded[k] = float(d[-max(1, len(d) // 2) :].mean())
    total_spikes = int(sum(np.count_nonzero(sim.data[spk_probes[k]]) for k in keys))
    return decoded, total_spikes, keys


def _ascii_spark(vals, width=48):
    """An ASCII sparkline of a value sequence (ASCII-only ramp, so Windows cp1252 can't crash on it)."""
    ramp = " .:-=+*#%@"
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    return "".join(
        ramp[int((vals[int(j * n / width)] - lo) / rng * (len(ramp) - 1))] for j in range(width)
    )


def launch(
    t=0.1,
    dt=0.001,
    trace=("brain_alive", "access_index", "novelty_live", "wake_ticks", "reflex_armed"),
    quiet=False,
):
    """Launch the WHOLE SNDL brain LIVE under nengo.Simulator (console script: `snn-nengo-launch`).

    Builds Brain(seed=0), wraps every numeric brain.state() signal in its own spiking NEF population, and steps
    the nengo sim TICK-BY-TICK -- printing a live per-tick trace of a few signals and, at the end, an ASCII
    sparkline of each traced signal over the run plus the total spikes. This is the v3.35 whole-brain OBSERVATION
    layer as an entrypoint; needs nengo (deployable to NengoLoihi). Observation-only -- no actuator route.
    `quiet` suppresses the live trace (used by the self-check smoke). Returns 0 on success, 1 if nengo absent."""
    if not _NENGO_OK:
        print(
            "snn-nengo-launch: nengo not installed (gated tier). pip install nengo (e.g. into .venv-nengo / "
            ".venv-cu128) to launch the brain under Nengo; the core is unaffected."
        )
        return 1
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0)  # Policy 4: the ONE sanctioned brain-construction path
    net, dec_probes, spk_probes, keys, _ = to_nengo_brain(b, tick=True)
    traced = [k for k in trace if k in keys] or keys[:5]
    history = {k: [] for k in traced}
    steps = max(1, int(t / dt))
    if not quiet:
        print("=" * 74)
        print(
            f"snn-nengo-launch: SNDL brain LIVE under nengo.Simulator ({len(keys)} signals, {steps} ticks)"
        )
        print("=" * 74)
    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        every = max(1, steps // 20)
        for i in range(steps):
            sim.step()  # one nengo dt = one brain tick (Node ticks it)
            for k in traced:
                history[k].append(float(sim.data[dec_probes[k]][-1][0]))
            if not quiet and i % every == 0:
                cells = "  ".join(f"{k[:11]}={history[k][-1]:+.2f}" for k in traced)
                print(f"  t={i * dt * 1000:5.0f}ms  {cells}")
    total_spikes = int(sum(np.count_nonzero(sim.data[spk_probes[k]]) for k in keys))
    b.close()
    if not quiet:
        print("-" * 74)
        print("per-signal trace over the run (ASCII sparkline, low..high):")
        for k in traced:
            h = history[k]
            print(f"  {k:16s} |{_ascii_spark(h)}| {min(h):+.2f}..{max(h):+.2f}")
        print("-" * 74)
        print(
            f"DONE: {len(keys)} brain signals observed as spiking NEF populations under Nengo; {total_spikes} "
            f"spikes. Observation-only, no actuator route; deployable to NengoLoihi."
        )
        print("=" * 74)
    return 0


# ----------------------------------------------------------------------------------------------------------
# self-check (un-riggable where nengo allows; tolerance bands, not bit-equality)
# ----------------------------------------------------------------------------------------------------------
def _rmse(pairs):
    return (sum((a - b) ** 2 for a, b in pairs) / len(pairs)) ** 0.5


def main():
    if not _NENGO_OK:
        print(
            "nengo_bridge self-check SKIP: nengo not installed (gated tier). "
            "pip install nengo to run the bridge / deploy to NengoLoihi; the core is unaffected."
        )
        return 0

    import sys
    from pathlib import Path

    # Repo root on sys.path. The comment here read "-> import research.*" until 2026-08-04, which
    # was stale: the import on the very next line goes to `snn_data_lake.reasoning.nef_compute`,
    # the FOLDED reasoning tier, as its own trailing comment already said. Nothing in this module
    # imports `research.*`. Corrected while retiring `research/brain_adapters/` -- a stale comment
    # naming a directory being deleted is exactly how a safe deletion gets mistaken for a breaking
    # one, and vice versa.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import snn_data_lake.instruments.nef_compute as nef  # the NEF ground truth (PROMOTED to cortex/ 2026-08-12)
    from snn_data_lake.growth.lif_neuron import LIFNeuron  # the discrete brain LIF

    print("=" * 72)
    print("NENGO BRIDGE -- brain NEF / LIF primitives on the Nengo simulator (Loihi-deployable)")
    print("=" * 72)

    # --- (1) NEF FIDELITY: Nengo decodes x^2 with error comparable to the zero-dep nef_compute ------------
    n = 150
    xs = [round(-0.9 + 1.8 * j / 8, 3) for j in range(9)]  # test points off the training grid
    sq = lambda x: x * x  # noqa: E731  (the nonlinear NEF target)

    zdep = nef.NEFEnsemble(n=n, seed=0)  # zero-dep NEF, same target
    d_sq = zdep.solve_decoders(sq)
    zdep_sq_rmse = _rmse([(zdep.decode(d_sq, x), sq(x)) for x in xs])
    d_id = zdep.solve_decoders(lambda x: x)
    zdep_id_rmse = _rmse([(zdep.decode(d_id, x), x) for x in xs])

    bridge = NengoNEF(dims=1, n_neurons=n, seed=0).transform(sq)
    ng_sq_pairs, ng_id_pairs, total_spikes = [], [], 0
    for x in xs:
        dec_sq, spk = bridge.run_full(x, func=sq, t=0.3)
        dec_id, _ = bridge.run_full(x, func=None, t=0.3)
        ng_sq_pairs.append((dec_sq, sq(x)))
        ng_id_pairs.append((dec_id, x))
        total_spikes += spk
    ng_sq_rmse = _rmse(ng_sq_pairs)
    ng_id_rmse = _rmse(ng_id_pairs)

    ctrl = NengoNEF(dims=1, n_neurons=1, seed=0)  # CONTROL: 1 neuron cannot decode x^2
    ctrl_rmse = _rmse([(ctrl.run(x, func=sq, t=0.3), sq(x)) for x in xs])

    print("(1) NEF FIDELITY  (approximator: within tolerance, never exact)")
    print(f"    represent x : zero-dep RMSE {zdep_id_rmse:.4f}   Nengo RMSE {ng_id_rmse:.4f}")
    print(
        f"    decode  x^2 : zero-dep RMSE {zdep_sq_rmse:.4f}   Nengo RMSE {ng_sq_rmse:.4f}  "
        f"(same NEF, different engine)"
    )
    print(f"    CONTROL 1-neuron decode x^2 : RMSE {ctrl_rmse:.4f}  (>> {n}-neuron -> fails)")

    # --- (2) LIF MATCH: matched Nengo LIF rate tracks the brain LIFNeuron rate across a current sweep ------
    tau_m, dt, N = 20.0, 0.001, 3000
    match = lif_to_nengo(tau_m=tau_m, v_threshold=1.0, dt=dt, tau_ref=0.0, seed=0)
    mistuned = lif_to_nengo(
        tau_m=tau_m * 2.5, v_threshold=1.0, dt=dt, tau_ref=0.0, seed=0
    )  # CONTROL
    currents = [0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
    diffs, ctrl_diffs, rows = [], [], []
    for c in currents:
        brain_hz = sum(LIFNeuron(tau_m=tau_m, v_threshold=1.0).run([c] * N)) / (N * dt)
        nengo_hz = match.sim_rate(c, t=N * dt)
        mis_hz = mistuned.sim_rate(c, t=N * dt)
        diffs.append(abs(brain_hz - nengo_hz))
        ctrl_diffs.append(abs(brain_hz - mis_hz))
        rows.append((c, brain_hz, nengo_hz, mis_hz))
    max_diff = max(diffs)
    ctrl_max_diff = max(ctrl_diffs)

    print("(2) LIF MATCH  (tau_m=20 -> tau_rc=0.02s; rate over a current sweep)")
    print("      current   brainHz   nengoHz   mistunedHz")
    for c, b, ng, mis in rows:
        print(f"       {c:.2f}    {b:7.1f}   {ng:7.1f}   {mis:7.1f}")
    print(
        f"    max abs rate diff (matched) {max_diff:.2f} Hz   CONTROL (mistuned tau) {ctrl_max_diff:.2f} Hz"
    )

    # --- (3) EXPORT RUNS: an exported brain circuit runs on nengo.Simulator and emits spikes --------------
    net_ff, pr_ff = to_nengo_network(value=0.5, n_neurons=120, lo=-1.0, hi=1.0, seed=0)
    dec_ff, spk_ff = simulate(net_ff, pr_ff, t=0.3)
    net_rc, pr_rc = to_nengo_network(
        value=0.4, n_neurons=120, lo=-1.0, hi=1.0, seed=0, recurrent=True
    )
    dec_rc, spk_rc = simulate(net_rc, pr_rc, t=0.3)

    print("(3) EXPORT RUNS  (brain circuit -> nengo.Network -> nengo.Simulator)")
    print(f"    population front-end : decoded {dec_ff:+.3f} (target 0.500), spikes {spk_ff}")
    print(
        f"    recurrent integrator : decoded {dec_rc:+.3f}, spikes {spk_rc}  (line-attractor loop holds)"
    )

    # --- (4) WHOLE-BRAIN OBSERVATION: the ENTIRE brain (incl torch tiers via exported signals) under Nengo ---
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0)  # Policy 4: the ONE sanctioned brain-construction path
    tier_keys = [k for k in brain_signals(b) if k.endswith("_tier")]  # torch/nengo/pc tier channels
    decoded_b, brain_spikes, bkeys = observe_brain(b, t=0.05)
    b.close()
    print(
        "(4) WHOLE-BRAIN OBSERVATION  (every numeric brain.state() signal -> its own spiking NEF ensemble)"
    )
    print(
        f"    observed {len(bkeys)} brain signals as Nengo spiking populations (total spikes {brain_spikes}); "
        f"tier channels incl torch tiers: {tier_keys}"
    )
    print("=" * 72)

    # ---- asserts: tolerance bands (the Nengo sim is float/seeded, not bit-deterministic) ----
    # (1) fidelity: Nengo decodes x^2 within tolerance AND comparably to the zero-dep NEF; control fails
    assert zdep_sq_rmse < 0.05, f"zero-dep NEF must decode x^2 tightly ({zdep_sq_rmse:.4f})"
    assert ng_sq_rmse < 0.10, f"Nengo NEF must decode x^2 within tolerance ({ng_sq_rmse:.4f})"
    assert ng_sq_rmse < zdep_sq_rmse + 0.08, (
        f"Nengo decode error must be COMPARABLE to the zero-dep NEF ({ng_sq_rmse:.4f} vs {zdep_sq_rmse:.4f})"
    )
    assert ng_id_rmse < 0.10, f"Nengo must represent x within tolerance ({ng_id_rmse:.4f})"
    assert ctrl_rmse > 0.15 and ctrl_rmse > 3 * ng_sq_rmse, (
        f"CONTROL: a 1-neuron ensemble must fail to decode x^2 ({ctrl_rmse:.4f} vs {ng_sq_rmse:.4f})"
    )
    assert total_spikes > 0, "the NEF ensemble must actually spike"

    # (2) LIF match: matched rate tracks the brain neuron within the band; mistuned tau breaks it
    assert max_diff < 10.0, (
        f"matched Nengo LIF rate must track the brain LIFNeuron ({max_diff:.2f} Hz)"
    )
    assert ctrl_max_diff > 3 * max_diff and ctrl_max_diff > 15.0, (
        f"CONTROL: a mistuned tau must FAIL the rate band ({ctrl_max_diff:.2f} vs matched {max_diff:.2f} Hz)"
    )

    # (3) export: the circuit runs, emits spikes, decodes finite
    assert spk_ff > 0, "exported population front-end must emit spikes"
    assert math.isfinite(dec_ff), "exported front-end decode must be finite"
    assert abs(dec_ff - 0.5) < 0.15, (
        f"front-end must represent the input scalar ({dec_ff:.3f} vs 0.5)"
    )
    assert spk_rc > 0 and math.isfinite(dec_rc), (
        "exported recurrent loop must run, spike, decode finite"
    )

    # (4) whole-brain observation: many signals observed as spiking populations; torch-tier channels included
    assert len(bkeys) >= 8, f"the whole-brain observer must expose many signals ({len(bkeys)})"
    assert brain_spikes > 0, "the whole-brain observation ensembles must actually spike"
    assert all(math.isfinite(v) for v in decoded_b.values()), (
        "every observed brain signal must decode finite"
    )
    assert len(tier_keys) >= 1 and all(k in bkeys for k in tier_keys), (
        "the observed set must include the torch/nengo/pc TIER channels (whole-brain observability incl torch)"
    )
    # (5) snn-nengo-launch entrypoint smoke: the console launcher steps a short brain-under-Nengo pass
    assert launch(t=0.006, quiet=True) == 0, (
        "the snn-nengo-launch entrypoint must run a brain under Nengo"
    )

    print(
        f"self-check OK: Nengo NEF decodes x^2 RMSE {ng_sq_rmse:.4f} (comparable to zero-dep "
        f"{zdep_sq_rmse:.4f}), 1-neuron control fails ({ctrl_rmse:.3f}); matched Nengo LIF tracks the brain "
        f"LIFNeuron to {max_diff:.2f} Hz (mistuned control {ctrl_max_diff:.1f} Hz); exported circuit ran on "
        f"nengo.Simulator emitting {spk_ff} spikes, decode {dec_ff:.3f}."
    )
    print(
        f"WHOLE-BRAIN: {len(bkeys)} brain.state() signals (incl the {len(tier_keys)} torch/nengo/pc tier "
        f"channels) observed as spiking NEF populations under nengo.Simulator ({brain_spikes} spikes)."
    )
    print(
        "HONEST: NEF is an APPROXIMATOR -- fidelity within tolerance, never exact. This bridges the NEF/LIF "
        "PRIMITIVES + OBSERVES the whole brain's output signals (torch tiers via exported scalars, not raw "
        "internal spikes); it is NOT a Nengo re-impl of ~200 brain modules. NengoLoihi is a deploy TARGET; "
        "Loihi silicon is not exercised here. Core untouched."
    )
    return 0


if __name__ == "__main__":
    main()
