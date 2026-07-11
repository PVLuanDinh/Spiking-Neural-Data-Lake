"""
Superior colliculus -- a retinotopic salience map that orients a saccade by a SPIKING race.

The superior colliculus holds a topographic map of visual space. Salient stimuli drive local
activity; lateral (surround) inhibition makes the locations COMPETE, and the FIRST location whose
neuron fires wins -> a saccade toward that location. Because each neuron integrates its input to a
fixed firing threshold, a MORE salient target reaches threshold and SPIKES sooner: latency falls
with salience, which is the express-vs-regular-saccade distinction (a chronometric law).

Mechanism (genuinely spiking, deterministic):
  1. A salience map (one value per retinotopic location) arrives from upstream feature/attention
     maps -- this module does NOT decide what is salient; it takes the map as input.
  2. Feedforward contrast: each location is penalized by its surround (the mean salience of every
     OTHER location, scaled by `inhib`), then rectified, giving a per-location INPUT CURRENT:
        drive_i = max(0, salience_i - inhib * mean(salience_{j != i})).
     A location survives only if it EXCEEDS its context -- a flat map cancels to zero current.
  3. LIF neurons: one leaky integrate-and-fire neuron per location. Every step its membrane leaks
     and integrates its input current and any lateral inhibition it received:
        v_i <- max(0, v_i * (1 - leak) + drive_i - inh_i).
     When v_i crosses the firing threshold `theta` it emits a discrete SPIKE and RESETS to v_reset.
  4. Spiking winner-take-all: every spike delivers a lateral INHIBITORY post-synaptic potential
     (`w_inhib`, decaying by `inhib_decay`) onto every OTHER neuron. The FIRST neuron to spike WINS
     the saccade; its spike TIME is the latency. The winner then keeps spiking while its lateral
     inhibition suppresses the losers -- a single committed target, not a blend.

Fair control (un-riggable):
  - a FLAT or ZERO salience map generates NO saccade: with nothing exceeding its surround, every
    input current rectifies to zero, no membrane ever reaches threshold, and no neuron spikes (no
    spurious target). A flat map can carry large total energy and still fire nothing -- it is
    CONTRAST, not energy, that drives a saccade.
  - two competing targets -> the STRONGER one spikes first and wins: the winning location equals
    the argmax of the salience map, and its lateral inhibition out-competes the weaker target.
  - LATENCY decreases as target salience increases (build-to-threshold-and-spike): sweeping a
    single target's salience upward yields strictly earlier first spikes -> shorter latencies.

Honest framing: a reduced, spiking model of collicular target SELECTION and saccade latency on a
synthetic salience map. It models the orienting/selection FUNCTION -- which location the map
commits to and how fast -- not vision, perception, or awareness. A "saccade" here is the first
threshold-crossing spike in a small LIF winner-take-all array, nothing more.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.midbrain.superior_colliculus
"""
import random


class SuperiorColliculus:
    """Retinotopic salience map: contrast current -> LIF neurons + lateral inhibition -> spiking WTA.

    saccade(salience) returns which location (if any) wins and its first-spike latency in steps. No
    location ever wins on a flat/zero map (nothing spikes); on a peaked map the argmax spikes first
    and wins, and higher salience -> earlier first spike -> lower latency."""

    def __init__(self, n=24, inhib=1.0, leak=0.02, theta=3.0, max_steps=200, seed=0,
                 w_inhib=2.0, inhib_decay=0.5, v_reset=0.0, readout_window=12):
        self.n = n
        self.inhib = inhib                  # surround-inhibition strength (>=1 cancels a flat map)
        self.leak = leak                    # LIF membrane leak per step (>0 == genuinely leaky)
        self.theta = theta                  # LIF firing threshold (membrane spikes when it crosses)
        self.max_steps = max_steps          # give up (no saccade) if nothing spikes in this many steps
        self.w_inhib = w_inhib              # lateral inhibitory PSP a spike delivers to every other
        self.inhib_decay = inhib_decay      # per-step decay of the lateral inhibitory synaptic trace
        self.v_reset = v_reset              # membrane potential a neuron resets to after a spike
        self.readout_window = readout_window  # steps to keep integrating after the winner first spikes
        self.rng = random.Random(seed)

    @staticmethod
    def _kwta(values, k):
        """Indices of the k largest values (winner-take-all), tie-break by lowest index.

        Matches basal_ganglia/action_gating._kwta so the two competition circuits agree."""
        order = sorted(range(len(values)), key=lambda i: (values[i], -i), reverse=True)
        return set(order[:k])

    def _lateral(self, salience):
        """Salience -> rectified per-location INPUT CURRENT via surround (lateral) inhibition. A
        location that does not exceed the mean of its surround gets zero drive (never spikes)."""
        n = len(salience)
        total = sum(salience)
        drives = []
        for i in range(n):
            surround = (total - salience[i]) / (n - 1) if n > 1 else 0.0
            drives.append(max(0.0, salience[i] - self.inhib * surround))
        return drives

    def saccade(self, salience):
        """Run the spiking winner-take-all race. One LIF neuron per location integrates its
        contrast-rectified salience current; the FIRST neuron to cross threshold and SPIKE wins the
        saccade, and its spike time is the latency. Each spike resets its own membrane and delivers
        a lateral INHIBITORY post-synaptic potential to every other neuron (the winner suppresses
        its competitors). Returns a dict with the winning target (or None -- no saccade -- if
        nothing ever spikes), the first-spike latency in steps, the per-location input drives, the
        salience argmax for cross-checking, the per-location spike counts, and the discrete spike
        train as (step, location) events."""
        drives = self._lateral(salience)
        n = len(salience)
        v = [0.0] * n                        # LIF membrane potentials (rest = 0)
        inh = [0.0] * n                      # lateral inhibitory synaptic trace onto each neuron
        spikes = [0] * n
        spike_train = []                     # discrete (step, location) spike events
        target, latency = None, None
        stop_at = self.max_steps
        for t in range(1, self.max_steps + 1):
            # leaky integrate-and-fire: leak the membrane, add the input current, subtract lateral
            # inhibition; the membrane cannot fall below rest. Then decay the inhibitory trace.
            for i in range(n):
                vi = v[i] * (1.0 - self.leak) + drives[i] - inh[i]
                v[i] = vi if vi > 0.0 else 0.0
                inh[i] *= self.inhib_decay
            # threshold crossings become discrete spikes (deterministic order: strongest membrane
            # first, tie -> lowest index, matching _kwta so the two competition circuits agree).
            fired = sorted((i for i in range(n) if v[i] >= self.theta), key=lambda i: (-v[i], i))
            for i in fired:
                spikes[i] += 1
                spike_train.append((t, i))
                v[i] = self.v_reset                       # SPIKE -> reset membrane
                for j in range(n):                        # lateral inhibitory synapses to all others
                    if j != i:
                        inh[j] += self.w_inhib
            if fired and target is None:
                target = fired[0]                         # FIRST location to spike wins the saccade
                latency = t
                stop_at = t + self.readout_window         # keep integrating to build the spike raster
            if target is not None and t >= stop_at:
                break
        argmax = next(iter(self._kwta(salience, 1)))
        return {"target": target, "latency": latency, "drives": drives, "argmax": argmax,
                "spikes": spikes, "spike_train": spike_train}

    def single_map(self, loc, salience, background=0.05, jitter=0.0):
        """A map with one target at `loc` over a low uniform background (optional deterministic
        jitter drawn from this instance's rng)."""
        m = [background] * self.n
        m[loc] = salience
        if jitter:
            m = [max(0.0, x + self.rng.uniform(-jitter, jitter)) for x in m]
        return m

    def two_target_map(self, loc_strong, s_strong, loc_weak, s_weak, background=0.05):
        """A map with a strong and a weak target competing over a low uniform background."""
        m = [background] * self.n
        m[loc_strong] = s_strong
        m[loc_weak] = s_weak
        return m


def main():
    import sys

    sc = SuperiorColliculus(seed=0)

    # single salient target -> a saccade to it (the neuron there spikes first)
    one = sc.single_map(loc=7, salience=0.8)
    res_one = sc.saccade(one)

    # flat and zero maps -> NO saccade (nothing spikes -> no spurious target)
    flat = [0.5] * sc.n
    zero = [0.0] * sc.n
    res_flat = sc.saccade(flat)
    res_zero = sc.saccade(zero)

    # two competing targets -> the stronger spikes first and wins (WTA == argmax)
    two = sc.two_target_map(loc_strong=4, s_strong=0.9, loc_weak=17, s_weak=0.6)
    res_two = sc.saccade(two)

    # latency vs salience sweep (build-to-threshold-and-spike: more salient -> earlier first spike)
    sweep = [0.3, 0.5, 0.7, 0.9]
    sweep_res = [sc.saccade(sc.single_map(loc=11, salience=s)) for s in sweep]
    latencies = [r["latency"] for r in sweep_res]

    print("=" * 68)
    print("SUPERIOR COLLICULUS -- retinotopic salience map -> SPIKING WTA saccade")
    print("=" * 68)
    print(f"locations: {sc.n} | threshold: {sc.theta} | leak: {sc.leak} | surround inhib: "
          f"{sc.inhib} | lateral w: {sc.w_inhib} | max steps: {sc.max_steps}")
    print(f"single target @ loc 7 (salience 0.80): first spike -> loc {res_one['target']} "
          f"at step {res_one['latency']} ({res_one['spikes'][res_one['target']]} spikes total)")
    print(f"flat map (all 0.50, high energy)     : saccade -> {res_flat['target']} "
          "(no contrast -> nothing spikes)")
    print(f"zero map (all 0.00)                  : saccade -> {res_zero['target']} "
          "(no drive -> nothing spikes)")
    print("-" * 68)
    print("two targets: strong @ loc 4 (0.90) vs weak @ loc 17 (0.60)")
    print(f"  -> winner loc {res_two['target']} first spike at step {res_two['latency']}  "
          f"(map argmax = loc {res_two['argmax']})")
    print(f"  -> spikes in readout window: winner loc 4 = {res_two['spikes'][4]}, "
          f"loser loc 17 = {res_two['spikes'][17]} (lateral inhibition suppresses the loser)")
    print("-" * 68)
    print("latency falls as salience rises (express vs regular saccade):")
    for s, lat in zip(sweep, latencies):
        print(f"  salience {s:.2f} -> first-spike latency {lat} steps")
    print("=" * 68)

    # ---- un-riggable self-checks (deterministic asserts) --------------------
    # (a) FUNCTION: the highest-salience target wins, matching the argmax of the input map.
    assert res_one["target"] == 7, "a single salient target must be selected"
    assert res_one["target"] == res_one["argmax"], "the winner must be the salience argmax"
    # a randomized battery: across many maps, the first neuron to spike is always the argmax.
    trng = random.Random(12345)
    for _ in range(40):
        m = [trng.uniform(0.02, 0.15) for _ in range(sc.n)]
        peak = trng.randrange(sc.n)
        m[peak] = max(m) + trng.uniform(0.3, 0.9)          # a unique, clearly dominant target
        r = sc.saccade(m)
        assert r["target"] == peak, "the highest-salience location must win the spiking race"
        assert r["target"] == r["argmax"], "the spiking winner must equal the salience argmax"
    # flat / zero maps generate NO saccade (nothing spikes -> no spurious target).
    assert res_flat["target"] is None, "a flat salience map must generate no saccade"
    assert res_zero["target"] is None, "a zero salience map must generate no saccade"
    assert res_flat["spike_train"] == [] and res_zero["spike_train"] == [], \
        "a flat/zero map must produce no spikes at all"
    assert sum(res_flat["spikes"]) == 0 and sum(res_zero["spikes"]) == 0, "no spikes on a flat map"

    # (b) CHRONOMETRIC: first-spike latency strictly decreases as salience increases.
    assert all(lat is not None for lat in latencies), "each real target must produce a spike"
    assert all(a > b for a, b in zip(latencies, latencies[1:])), \
        "first-spike latency must strictly decrease as salience increases"

    # (c) GENUINELY SPIKING: real LIF membranes (leak + integrate + threshold + spike + reset),
    #     the winner is chosen by the FIRST spike, discrete spike events, no numpy/torch.
    assert sc.leak > 0.0, "the membrane must genuinely leak (leaky integrate-and-fire)"
    assert sc.theta > 0.0 and hasattr(sc, "v_reset"), "a firing threshold and a reset must exist"
    # discrete spike events: integer (step, location) pairs, and the winner's FIRST spike == latency.
    train = res_one["spike_train"]
    assert train and all(isinstance(t, int) and isinstance(i, int) for t, i in train), \
        "spikes must be discrete integer (step, location) events"
    winner_first = min(t for t, i in train if i == res_one["target"])
    assert winner_first == res_one["latency"], "latency must be the winner's FIRST spike time"
    # reset is real: after firing, the winner resets and re-integrates to spike again (a train).
    assert res_one["spikes"][res_one["target"]] >= 2, \
        "the winner must spike repeatedly, which is only possible if the membrane resets"
    # lateral inhibition genuinely produces winner-take-all: winner out-spikes the loser, and
    # non-target (background) neurons never spike.
    assert res_two["spikes"][4] > res_two["spikes"][17], \
        "the winner's lateral inhibition must suppress the loser (spiking WTA)"
    assert all(c == 0 for k, c in enumerate(res_two["spikes"]) if k not in (4, 17)), \
        "background neurons (zero drive) must never spike"
    assert "numpy" not in sys.modules and "torch" not in sys.modules, \
        "this module must be pure stdlib (no numpy / torch)"

    # (d) DETERMINISM: same seed -> identical jittered maps and identical saccades (spikes included).
    a = SuperiorColliculus(seed=0)
    b = SuperiorColliculus(seed=0)
    ma = a.single_map(loc=7, salience=0.8, jitter=0.03)
    mb = b.single_map(loc=7, salience=0.8, jitter=0.03)
    assert ma == mb, "same seed must reproduce identical jittered maps"
    assert a.saccade(ma) == b.saccade(mb), "same seed must reproduce identical spiking saccades"

    print(f"self-check OK: single target -> loc {res_one['target']} spikes first (== argmax); "
          f"flat/zero maps -> no spikes / no saccade;")
    print(f"  stronger of two spikes first and wins (loc {res_two['target']} == argmax, "
          f"{res_two['spikes'][4]} vs {res_two['spikes'][17]} spikes via lateral inhibition);")
    print(f"  first-spike latency {latencies[0]}->{latencies[-1]} steps strictly falls as salience "
          f"{sweep[0]:.1f}->{sweep[-1]:.1f} rises; LIF leak/threshold/reset; deterministic")


if __name__ == "__main__":
    main()
