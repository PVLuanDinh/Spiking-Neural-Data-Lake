"""
Izhikevich neuron — rich cortical spiking regimes from a 2-equation model (neuron primitive)
Pure-stdlib.

Every neuron in the repo is LIF: integrate, fire, reset — one regime, no bursting or adaptation.
Izhikevich's model (2003) reproduces the full zoo of cortical firing patterns — regular spiking,
fast spiking, chattering, intrinsically bursting — from just two variables (membrane v, recovery u)
and four parameters (a,b,c,d), at ~13 flops/step. That is the efficiency-vs-richness sweet spot a
neuromorphic project cares about: LIF-cheap, but able to express bursts and spike-frequency
adaptation the temporal codes here rely on. Refs: Izhikevich, "Simple model of spiking neurons",
IEEE Trans. Neural Networks 14 (2003); "Which model to use for cortical spiking neurons?" (2004).

  Run: python izhikevich_neuron.py        # (also run by run_selfchecks.py)
"""


def simulate(a, b, c, d, current=10.0, T=300.0, dt=0.1):
    """Integrate the Izhikevich model; return the list of spike times (ms)."""
    v, u = -65.0, b * -65.0
    spikes = []
    for k in range(int(T / dt)):
        if v >= 30.0:                       # spike: record peak, then reset
            spikes.append(k * dt)
            v, u = c, u + d
        v += dt * (0.04 * v * v + 5 * v + 140 - u + current)
        u += dt * (a * (b * v - u))
    return spikes


def _isi_cv(spikes):
    """Coefficient of variation of inter-spike intervals; high = bursty, low = regular."""
    if len(spikes) < 3:
        return 0.0
    isi = [b - a for a, b in zip(spikes, spikes[1:])]
    m = sum(isi) / len(isi)
    var = sum((x - m) ** 2 for x in isi) / len(isi)
    return (var ** 0.5) / m if m else 0.0


# canonical Izhikevich parameter sets (a, b, c, d)
REGIMES = {
    "regular-spiking":      (0.02, 0.2, -65.0, 8.0),
    "intrinsically-bursting": (0.02, 0.2, -55.0, 4.0),
    "chattering":           (0.02, 0.2, -50.0, 2.0),
    "fast-spiking":         (0.10, 0.2, -65.0, 2.0),
}


def main():
    out = {name: simulate(*p) for name, p in REGIMES.items()}
    counts = {name: len(s) for name, s in out.items()}
    cvs = {name: _isi_cv(s) for name, s in out.items()}

    print("=" * 62)
    print("IZHIKEVICH NEURON — cortical spiking regimes from a 2-eq model")
    print("=" * 62)
    print(f"{'regime':<24}{'spikes/300ms':>13}{'ISI CV':>10}")
    for name in REGIMES:
        print(f"{name:<24}{counts[name]:>13}{cvs[name]:>10.2f}")
    print("=" * 62)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert all(n >= 3 for n in counts.values()), f"every regime must fire under drive ({counts})"
    assert counts["fast-spiking"] > counts["regular-spiking"], \
        f"fast-spiking must fire faster than regular-spiking ({counts['fast-spiking']} vs {counts['regular-spiking']})"
    assert cvs["chattering"] > cvs["regular-spiking"], \
        f"chattering must burst (higher ISI variability) than regular-spiking ({cvs['chattering']:.2f} vs {cvs['regular-spiking']:.2f})"
    assert cvs["intrinsically-bursting"] > cvs["regular-spiking"], \
        "intrinsically-bursting must show burst structure vs regular-spiking"
    assert simulate(*REGIMES["chattering"]) == simulate(*REGIMES["chattering"]), "deterministic"
    print(f"self-check OK: 4 distinct regimes; FS {counts['fast-spiking']} > RS {counts['regular-spiking']} spikes; "
          f"bursting CV CH {cvs['chattering']:.2f} > RS {cvs['regular-spiking']:.2f}; deterministic")


if __name__ == "__main__":
    main()
