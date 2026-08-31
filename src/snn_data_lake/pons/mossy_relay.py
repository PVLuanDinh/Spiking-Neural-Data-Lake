"""
Pontine mossy-fibre relay -- the `pn` node implemented (specs/022 T054-i, the second pons
disposition: the package existed as anatomy prose only; this is its first computing module,
and it SAYS ITS OWN NODE ID, which is what retires the atlas-gate's day-one "pn" phantom row).

WHAT IT IS. The cortico-ponto-cerebellar relay stage the connectome has declared since
T018: `ctx_motor -> pn` (corticopontine) and `pn -> cb_ctx` (pontocerebellar mossy fibres,
excitatory). Pontine nuclei receive a cortical efferent copy and DIVERGE it onto the
cerebellar granule stage as mossy-fibre activity -- a recombining fan-out, not a mirror:
each pontine cell samples a seeded subset of the cortical drive, so the mossy code the
cerebellum sees is a mixed projection of cortex, which is what the granule expansion then
separates (cerebellum/microcircuit.py, the tissue this relay feeds).

STRUCTURE, said honestly per value (P1, re-adjudicated by this module's panel -- the first
draft called these "structure as spec" and attached no measurement to either, so its own
self-check passed unchanged at n_pn=1*n_ctx and k_in=2): n_pn = 2*n_ctx and
k_in = ceil(sqrt(n_ctx)) are CHOSEN STRUCTURAL POINTS, not derivations and not cited
constants -- divergence and sparse sampling are the anatomy, these particular integers are
this build's affordable point on it. What IS held to a measurement is the property the
SPARSE SAMPLING exists to buy -- DISCRIMINATION, with the full-fan-out control as its
reachable red: sparse taps give 59 of 60 distinct mossy codes for distinct cortical
patterns, while a relay whose cells all tap every channel collapses to 2 of 60 (they see
one common near-saturated average). Measured direction, not assumed: the first attempt at
this control predicted the opposite (that full fan-out would SPREAD a one-channel change)
and measured 0 -- saturation, not spread. The THRESHOLD is the LIF's normalized fixed point (v_threshold=1.0 on fan-in-
normalized drive, P1 case (a)); tau_m=10.0 is the reused LIFNeuron default; every weight
is seeded structure, procedurally regenerable from (seed, dims).

Deterministic (per-instance random.Random(seed)); stdlib + the reused LIFNeuron only.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.pons.mossy_relay   # (run by run_selfchecks.py)
"""

import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron

# distinct-from: thalamus/interoceptive_relay.py -- thal_relay is the interoceptive
#   tonic/burst hop into cortex; this is the cortico-CEREBELLAR mossy fan-out. Different
#   declared edges, different transform. Site read.
# distinct-from: thalamus/connectome_router.py -- the router turns EVERY declared edge
#   into a generic delayed LIF projection; this module IS the pn stage's own computation
#   (seeded recombining fan-out), the thing the router's generic projection is not. Read.

NODE_ID = "pn"  # the connectome id this module computes near -- the atlas-gate contract


class MossyRelay:
    """The pontine nuclei stage: cortical drive in -> mossy-fibre spikes out.

    n_ctx cortical channels diverge onto n_pn = 2*n_ctx pontine LIF cells, each sampling
    k_in = ceil(sqrt(n_ctx)) channels with seeded fan-in-normalized weights, so a
    uniformly-active cortex drives each cell to the LIF's normalized threshold scale
    (P1 case (a)). Both integers are CHOSEN structural points, said so in the module
    docstring rather than dressed as derivations; the recombination they exist to buy is
    what the self-check measures, against a full-fan-out control."""

    def __init__(self, n_ctx, seed=0):
        if not (isinstance(n_ctx, int) and n_ctx >= 2):
            raise ValueError(f"n_ctx must be an int >= 2, got {n_ctx!r}")
        self.n_ctx = n_ctx
        self.n_pn = 2 * n_ctx
        self.k_in = max(2, math.ceil(math.sqrt(n_ctx)))
        rng = random.Random(seed)
        self.seed = seed
        # seeded recombining fan-out: each pontine cell's (channel indices, weights);
        # weights positive (the corticopontine edge is declared "exc") and fan-in
        # normalized so sum(w) == 1.0 per cell -- structure, regenerable from (seed, dims)
        self.taps = []
        for _ in range(self.n_pn):
            idx = rng.sample(range(n_ctx), self.k_in)
            raw = [rng.random() + 0.1 for _ in idx]
            s = sum(raw)
            self.taps.append((tuple(idx), tuple(w / s for w in raw)))
        self.cells = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(self.n_pn)]

    def step(self, ctx_drive):
        """One timestep: cortical drive vector (len n_ctx, finite) -> 0/1 mossy spikes (len n_pn)."""
        if len(ctx_drive) != self.n_ctx:
            raise ValueError(
                f"ctx_drive has {len(ctx_drive)} channels, relay declares {self.n_ctx}"
            )
        out = []
        for (idx, w), cell in zip(self.taps, self.cells):
            drive = 0.0
            for i, wi in zip(idx, w):
                x = float(ctx_drive[i])
                if not math.isfinite(x):
                    raise ValueError(f"non-finite cortical drive at channel {i}: {x!r}")
                drive += wi * x
            out.append(cell.step(drive))
        return out


def main():
    line = "=" * 78
    print(line)
    print("pons/mossy_relay -- the pn stage (specs/022 T054-i): cortico-cerebellar fan-out")
    print(line)
    n_ctx = 12
    r1 = MossyRelay(n_ctx, seed=5)
    r2 = MossyRelay(n_ctx, seed=5)
    r3 = MossyRelay(n_ctx, seed=6)
    # (1) REGENERABLE STRUCTURE: same (seed, dims) -> identical taps; a different seed differs
    assert r1.taps == r2.taps, "same seed must regenerate identical relay structure"
    assert r1.taps != r3.taps, "a different seed must produce different structure"
    # (2) FAN-IN NORMALIZATION: every cell's weights sum to 1.0 (within float sum error)
    worst = max(abs(sum(w) - 1.0) for _idx, w in r1.taps)
    assert worst < 1e-12, f"fan-in normalization broke (worst |sum-1| = {worst})"
    # (3) RELAY IS LIVE + DETERMINISTIC: a driven cortex produces mossy spikes, twice identically
    rng = random.Random(11)
    drives = [[1.4 if rng.random() < 0.5 else 0.0 for _ in range(n_ctx)] for _ in range(60)]
    s1 = [r1.step(d) for d in drives]
    s2 = [r2.step(d) for d in drives]
    total = sum(sum(s) for s in s1)
    assert s1 == s2, "same seed + same drive must produce identical spike trains"
    assert total > 0, "a driven relay must emit mossy spikes"
    # (4) RECOMBINATION, not a mirror: two cortical patterns that differ produce mossy
    # patterns that differ (the fan-out carries information), while a single channel's
    # change does NOT flip every cell (locality of the seeded taps)
    a = [1.4] * n_ctx
    b = list(a)
    b[0] = 0.0
    ra = MossyRelay(n_ctx, seed=5)
    rb = MossyRelay(n_ctx, seed=5)
    for _ in range(5):
        sa = ra.step(a)
        sb = rb.step(b)
    diff = sum(1 for x, y in zip(sa, sb) if x != y)
    touched = sum(1 for idx, _w in ra.taps if 0 in idx)
    assert 0 < diff <= touched, (
        f"a one-channel change must move only cells tapping it (moved {diff}, tap {touched})"
    )

    # (4b) DISCRIMINATION, the property the sparse sampling exists to buy, with the
    # FULL-FAN-OUT control as its reachable red (the first draft asserted locality with no
    # control at all -- its panel's finding). Measured, and the direction is the opposite
    # of the one first guessed: with every cell tapping every channel the cells all see
    # the same near-saturated average and COLLAPSE onto ~one code, so distinct cortical
    # patterns stop producing distinct mossy codes. That collapse is the red.
    def _codes(full_fanout, n_pat=60, steps=3):
        rngc = random.Random(17)
        out = []
        for _ in range(n_pat):
            r = MossyRelay(n_ctx, seed=5)
            if full_fanout:
                fw = tuple([1.0 / n_ctx] * n_ctx)
                r.taps = [(tuple(range(n_ctx)), fw) for _ in r.taps]
            pat = [1.4 if rngc.random() < 0.5 else 0.0 for _ in range(n_ctx)]
            for _ in range(steps):
                s = r.step(pat)
            out.append(tuple(s))
        return len(set(out)), len(out)

    d_sparse, n_pat = _codes(False)
    d_full, _ = _codes(True)
    assert d_sparse > 0.9 * n_pat and d_full < 0.2 * n_pat, (
        f"sparse sampling must DISCRIMINATE cortical patterns where full fan-out collapses "
        f"({d_sparse}/{n_pat} distinct mossy codes vs the control's {d_full}/{n_pat})"
    )
    # (5) refusals
    try:
        r1.step([0.0] * (n_ctx - 1))
        raise AssertionError("wrong-width drive must refuse")
    except ValueError as e:
        assert "channels" in str(e)
    try:
        MossyRelay(1)
        raise AssertionError("degenerate n_ctx must refuse")
    except ValueError as e:
        assert "n_ctx" in str(e)
    print(
        f"self-check OK: pn stage regenerable (taps identical under seed {r1.seed}); "
        f"fan-out {n_ctx} ctx -> {r1.n_pn} mossy (k_in {r1.k_in}, fan-in normalized, worst "
        f"|sum-1| {worst:.1e}); driven relay emitted {total} spikes over 60 steps, "
        f"deterministic; one-channel locality {diff} moved / {touched} tapping; "
        f"DISCRIMINATION {d_sparse}/{n_pat} distinct mossy codes vs the full-fan-out "
        f"control's {d_full}/{n_pat} (the sampling's reachable red); refusals by message"
    )


if __name__ == "__main__":
    main()
