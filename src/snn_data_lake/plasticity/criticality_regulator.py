"""
Self-organized criticality regulator -- a homeostat that holds a spiking substrate at the EDGE OF CHAOS
so it stays maximally learnable (ladder rung L7).

A recurrent LIF pool has a BRANCHING RATIO sigma = the average number of downstream spikes each spike
triggers. sigma < 1 (subcritical) -> activity dies, the raster goes silent, there is nothing to learn
from; sigma > 1 (supercritical) -> activity saturates, every stimulus evokes the same runaway response,
so nothing is separable; sigma ~= 1 (critical, held slightly SUB at ~0.95-1.02) = the edge of chaos,
where dynamic range and stimulus separability are maximal. Real cortex self-regulates to this band.

This closes the open loop noted in docs/SELF_IMPROVEMENT_RESEARCH.md (L7 / SI-11): the validated homeostats
(homeostatic_scaling, intrinsic_plasticity, inhibitory_stdp) and the branching-ratio measurement
(neuronal_avalanches) both existed but were IMPORTED NOWHERE -- nothing measured the regime from the live
raster and fed it back. Here we do:

  1. a recurrent LIF pool (reuses cortex.lif_neuron.LIFNeuron) with sparse random heterogeneous
     excitatory weights and E/I-balanced, activity-proportional global inhibition; seed a cascade, step
     it, record the integer spike raster. The activity-dependent inhibition makes the excitatory->silent
     transition SECOND-ORDER (a smooth, tunable critical region) rather than a bistable all-or-nothing
     jump -- that is what gives a real edge-of-chaos band to sit in.
  2. MEASURE sigma from the raster (reuses neuronal_avalanches.BranchingNetwork.measured_branching_ratio
     over the unsaturated onset window of seeded cascades).
  3. REGULATE: a slow, slew-rate-limited homeostatic integrator nudges ONE global knob -- the inhibitory
     gain -- by the error (sigma - target): MORE inhibition when sigma runs hot, LESS when it runs cold.
     regulate=False freezes the knob (the control).

HONEST SCOPE. Self-organized criticality / homeostatic regulation holding the spiking substrate slightly
sub-critical (the edge of chaos) = maximally learnable. This is a bounded substrate-HEALTH mechanism: it
keeps the network the OTHER learners write into learnable; it is NOT itself intelligence, NOT AGI, NOT
consciousness. "Criticality" here is nothing more than the MEASURED branching ratio of a small finite
network -- finite-size sigma-CONTROL, not a claim of physical criticality; power-law != criticality; the
regulator earns no task-accuracy by itself (the arm regime is band-linear, the bottleneck is the encoder,
not the regime). It regulates the substrate; it does not think.

Per-instance random.Random(seed); zlib.crc32 for child seeds (never
hash()); deterministic; ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.criticality_regulator   # (run_selfchecks.py)
"""

import random
import zlib

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.cortex.neuronal_avalanches import BranchingNetwork


def _child_seed(base, tag):
    """Deterministic child seed from a base seed + a string tag via crc32 (never hash())."""
    return zlib.crc32(f"{base}:{tag}".encode("ascii"))


class CriticalityRegulator:
    """Recurrent LIF pool whose branching ratio sigma is measured from the live raster and held in a
    slightly-subcritical band by a slow homeostatic integrator on ONE global inhibitory-gain knob.

    The knob (`inhib_gain`) scales E/I-balanced, activity-proportional global inhibition: every neuron
    gets -inhib_gain * (spikes_last_step / n) of inhibitory current, so the more the pool fired, the
    harder it is pushed back. Raising inhib_gain lowers the pool's steady activity and its branching
    ratio -- a smoothly monotone map (the activity-dependence turns the transition second-order, so there
    is a real critical band, not a bistable cliff). run(regulate=True) drives the knob toward `target`;
    regulate=False freezes it. All state is deterministic given `seed`."""

    def __init__(
        self,
        n=240,
        k=12,
        w_max=1.7,
        tau_m=8.0,
        v_threshold=1.0,
        refractory=1,
        inhib_gain=18.0,
        target=0.99,
        band=(0.95, 1.02),
        lr=40.0,
        max_step=5.0,
        seed=0,
        steps=30,
        n_seed=4,
        seed_current=5.0,
        sat_frac=0.6,
        ema=0.6,
        n_meas=12,
    ):
        self.n = n
        self.k = k
        self.v_threshold = v_threshold
        self.inhib_gain = inhib_gain  # THE KNOB: gain of activity-proportional global inhibition
        self.target = target
        self.band = band
        self.lr = lr
        self.max_step = (
            max_step  # slew-rate limit: |knob change| per control epoch (slow homeostasis)
        )
        self.ema = ema
        self.seed = seed
        self.steps = steps
        self.n_seed = n_seed
        self.seed_current = seed_current
        self.sat_cap = max(
            2, int(sat_frac * n)
        )  # ancestor generations >= this are saturated: excluded
        self.n_meas = n_meas  # cascades per (deterministic) sigma measurement
        self.g_lo, self.g_hi = 0.0, 120.0  # knob bounds
        self.sigma_ema = None
        self.sigma_history = []
        self.knob_history = []
        rng = random.Random(seed)
        # sparse random recurrent excitatory connectivity with HETEROGENEOUS weights
        self.out_edges = []
        for _j in range(n):
            targets = rng.sample(range(n), k)
            self.out_edges.append([(i, rng.uniform(0.0, w_max)) for i in targets])
        self.neurons = [
            LIFNeuron(tau_m=tau_m, v_threshold=v_threshold, refractory=refractory) for _ in range(n)
        ]

    # ---- live spiking substrate -------------------------------------------------------------------
    def _seed_units(self, tag):
        """A deterministic distinct set of seed neurons for one cascade (crc32-derived, per instance)."""
        return random.Random(_child_seed(self.seed, tag)).sample(range(self.n), self.n_seed)

    def _cascade(self, seed_units):
        """Run ONE seeded cascade at the CURRENT knob through real LIF membranes.
        Returns (raster, response): raster = per-step integer spike counts A(t); response = per-neuron
        total spike counts (the stimulus-response vector). Every step, activity-proportional global
        inhibition -inhib_gain*A(t-1)/n hits every neuron; a t=0 seed current forces the seed set to
        spike; propagation is recurrent and E/I-balanced."""
        for nu in self.neurons:
            nu.reset_state()
        response = [0] * self.n
        raster = []
        prev_spikes = list(seed_units)  # gen 0 = the seed set (forced to fire below at t=0)
        a_prev = 0
        seed_set = set(seed_units)
        for t in range(self.steps):
            inh = (
                self.inhib_gain * a_prev / self.n
            )  # E/I-balanced global inhibition (activity-proportional)
            cur = [-inh] * self.n
            if t == 0:
                for i in seed_set:
                    cur[i] += self.seed_current  # force the seed neurons over threshold
            else:
                for j in prev_spikes:  # deliver last step's spikes along the synapses
                    for i, w in self.out_edges[j]:
                        cur[i] += w
            spikes = []
            for i in range(self.n):
                if self.neurons[i].step(cur[i]):
                    spikes.append(i)
                    response[i] += 1
            raster.append(len(spikes))
            a_prev = len(spikes)
            prev_spikes = spikes
            if not spikes and t > 0:
                break  # cascade extinguished -> stop early
        return raster, response

    # ---- readouts ---------------------------------------------------------------------------------
    def measure(self, n_cascades=None, tag="meas"):
        """Average (sigma, mean_activity) over a FIXED deterministic battery of seeded cascades at the
        current knob. Deterministic given the knob, so the controller and the evaluator agree exactly.
        sigma reuses BranchingNetwork.measured_branching_ratio over the pooled unsaturated-window trace
        (walk generations from t=0 while the ancestor count is alive and unsaturated -- captures the
        onset expansion, excludes the finite-size collapse); mean_activity = mean fraction of the pool
        spiking per step (0 = silent, ~1 = saturated)."""
        nc = self.n_meas if n_cascades is None else n_cascades
        a_tot = 0
        d_tot = 0
        act = 0.0
        for c in range(nc):
            raster, _ = self._cascade(self._seed_units(f"{tag}:{c}"))
            for t in range(len(raster) - 1):
                a = raster[t]
                if a <= 0 or a >= self.sat_cap:
                    break
                a_tot += a
                d_tot += raster[t + 1]
            act += sum(raster) / (self.n * len(raster))
        trace = [(a_tot, d_tot)] if a_tot else []
        sigma = BranchingNetwork.measured_branching_ratio(trace)
        return sigma, act / nc

    def sigma(self):
        """Stable branching-ratio estimate at the current knob (the fixed measurement battery)."""
        return self.measure()[0]

    def activity(self):
        """Mean pool firing fraction per step (dead ~0, critical low, saturated ~near max)."""
        return self.measure()[1]

    def separability(self, n_stim=6, tag="sep"):
        """Dynamic-range / separability readout: mean pairwise L2 distance between the per-neuron
        spike-count responses to n_stim DISTINCT seed stimuli. Peaks at the edge of chaos -- a dead
        subcritical pool answers every stimulus with ~zeros, a saturated supercritical pool answers every
        stimulus with ~the same maxed-out burst, and BOTH collapse the distances; only a critical pool
        maps distinct inputs to distinct spike patterns."""
        responses = [self._cascade(self._seed_units(f"{tag}:{s}"))[1] for s in range(n_stim)]
        total = 0.0
        pairs = 0
        for a in range(n_stim):
            for b in range(a + 1, n_stim):
                ra, rb = responses[a], responses[b]
                d = sum((ra[i] - rb[i]) ** 2 for i in range(self.n))
                total += d**0.5
                pairs += 1
        return total / pairs if pairs else 0.0

    # ---- the homeostatic control loop -------------------------------------------------------------
    def run(self, epochs=55, regulate=True):
        """Run the control loop for `epochs`. Each epoch: measure sigma from the live raster, update its
        EMA, and (when regulate) nudge the inhibitory-gain knob by lr*(sigma_ema - target) -- MORE
        inhibition when sigma runs hot, LESS when it runs cold -- slew-limited to +/-max_step per epoch
        and clamped to the knob bounds. regulate=False freezes the knob. Returns the final sigma_ema."""
        for _e in range(epochs):
            sigma, _act = self.measure()
            self.sigma_ema = (
                sigma
                if self.sigma_ema is None
                else self.ema * sigma + (1.0 - self.ema) * self.sigma_ema
            )
            self.sigma_history.append(sigma)
            self.knob_history.append(self.inhib_gain)
            if regulate:
                step = self.lr * (self.sigma_ema - self.target)
                step = min(
                    self.max_step, max(-self.max_step, step)
                )  # slow homeostasis: slew-limited
                self.inhib_gain = min(self.g_hi, max(self.g_lo, self.inhib_gain + step))
        return self.sigma_ema


def _in_band(x, lo, hi):
    return lo <= x <= hi


def main():
    lo, hi = 0.95, 1.02  # the slightly-subcritical target band
    sub_init, super_init = 90.0, 0.0  # knob inits: over-inhibited (dead) / un-inhibited (saturated)
    epochs = 55

    print("=" * 82)
    print("CRITICALITY REGULATOR -- self-organized criticality / edge-of-chaos homeostat (L7)")
    print("=" * 82)
    print(
        "pool: recurrent LIF (n=240, k=12 sparse heterogeneous exc, E/I-balanced global inhibition)"
    )
    print(
        f"target band [{lo:.2f}, {hi:.2f}] (slightly sub); knob = inhibitory gain (activity-proportional)"
    )
    print(
        "knob scan on the FIXED substrate (seed=0): more inhibition -> lower sigma (smooth, monotone)"
    )
    print("  inhib_gain    sigma    activity   separability")
    for g in (0.0, 10.0, 16.0, 20.0, 35.0, 90.0):
        r = CriticalityRegulator(inhib_gain=g, seed=0)
        print(f"    {g:6.1f}    {r.sigma():6.3f}    {r.activity():7.4f}   {r.separability():9.2f}")

    # (1) REGULATION TO THE BAND from BOTH a subcritical and a supercritical init -----------------
    reg_sub = CriticalityRegulator(inhib_gain=sub_init, seed=0)
    sub_sigma0, sub_act0 = reg_sub.measure()
    reg_sub.run(epochs=epochs, regulate=True)
    reg_sub_final = reg_sub.sigma()

    reg_super = CriticalityRegulator(inhib_gain=super_init, seed=0)
    super_sigma0, super_act0 = reg_super.measure()
    reg_super.run(epochs=epochs, regulate=True)
    reg_super_final = reg_super.sigma()

    # controls: SAME inits, knob FROZEN (regulate=False) -> must stay out of the band ---------------
    ctl_sub = CriticalityRegulator(inhib_gain=sub_init, seed=0)
    ctl_sub.run(epochs=epochs, regulate=False)
    ctl_sub_sigma, ctl_sub_act = ctl_sub.measure()

    ctl_super = CriticalityRegulator(inhib_gain=super_init, seed=0)
    ctl_super.run(epochs=epochs, regulate=False)
    ctl_super_sigma, ctl_super_act = ctl_super.measure()

    # (2) DYNAMIC RANGE / SEPARABILITY: critical vs the two frozen controls -------------------------
    reg_sep = reg_sub.separability()
    sub_sep = ctl_sub.separability()
    super_sep = ctl_super.separability()

    # (3) DETERMINISM: a fresh identical run reproduces the outcome exactly -------------------------
    det = CriticalityRegulator(inhib_gain=sub_init, seed=0)
    det.run(epochs=epochs, regulate=True)
    det_final, det_knob = det.sigma(), det.inhib_gain

    print("")
    print(
        "  init            init sigma  init act   regulated sigma   frozen-control (sigma / act)   separability"
    )
    print(
        f"  subcritical      {sub_sigma0:7.3f}   {sub_act0:6.3f}      {reg_sub_final:7.3f}"
        f"          {ctl_sub_sigma:6.3f} / {ctl_sub_act:.3f}         reg  {reg_sep:6.2f}"
    )
    print(
        f"  supercritical    {super_sigma0:7.3f}   {super_act0:6.3f}      {reg_super_final:7.3f}"
        f"          {ctl_super_sigma:6.3f} / {ctl_super_act:.3f}         dead {sub_sep:6.2f}  sat {super_sep:6.2f}"
    )
    print(
        f"  regulated knob (inhib_gain): sub-init {sub_init:.0f} -> {reg_sub.inhib_gain:.2f}, "
        f"super-init {super_init:.0f} -> {reg_super.inhib_gain:.2f}"
    )
    print("=" * 82)

    # ---- self-check: deterministic asserts, each claim + a control that FAILS ----------------------
    # (1) regulator pulls BOTH extreme inits into the band; both frozen controls stay OUT.
    assert sub_sigma0 < lo, (
        f"subcritical init must start dead below the band ({sub_sigma0:.3f} < {lo})"
    )
    assert super_act0 > 0.15 and super_sigma0 > hi, (
        f"supercritical init must start saturated above the band ({super_sigma0:.3f}, act {super_act0:.3f})"
    )
    assert _in_band(reg_sub_final, lo, hi), (
        f"regulator must pull the subcritical init INTO the band ({reg_sub_final:.3f})"
    )
    assert _in_band(reg_super_final, lo, hi), (
        f"regulator must pull the supercritical init INTO the band ({reg_super_final:.3f})"
    )
    assert ctl_sub_sigma < lo and ctl_sub_act < 0.02, (
        f"frozen subcritical control must STAY dead below the band ({ctl_sub_sigma:.3f}, act {ctl_sub_act:.3f})"
    )
    assert ctl_super_act > 0.15 and ctl_super_sigma > hi, (
        f"frozen supercritical control must STAY saturated above the band ({ctl_super_sigma:.3f}, act {ctl_super_act:.3f})"
    )

    # (2) the regulated (critical) pool has a far wider dynamic range than either frozen control.
    assert reg_sep > 2.0 * sub_sep, (
        f"critical separability must dwarf the dead control ({reg_sep:.2f} vs {sub_sep:.2f})"
    )
    assert reg_sep > 2.0 * super_sep, (
        f"critical separability must dwarf the saturated control ({reg_sep:.2f} vs {super_sep:.2f})"
    )

    # (3) spiking + determinism + ASCII.
    assert det_final == reg_sub_final and det_knob == reg_sub.inhib_gain, (
        "same seed -> identical regulated outcome"
    )
    assert isinstance(reg_sub.measure()[0], float), "sigma is computed from integer spike cascades"

    print(
        f"self-check OK: subcritical {sub_sigma0:.2f} -> regulated {reg_sub_final:.3f} (band); "
        f"supercritical sat(act {super_act0:.2f}) -> regulated {reg_super_final:.3f} (band); "
        f"frozen controls stay dead {ctl_sub_sigma:.2f}/sat(act {ctl_super_act:.2f}); "
        f"dynamic range {reg_sep:.1f} >> {sub_sep:.1f}/{super_sep:.1f}; deterministic"
    )


if __name__ == "__main__":
    main()
