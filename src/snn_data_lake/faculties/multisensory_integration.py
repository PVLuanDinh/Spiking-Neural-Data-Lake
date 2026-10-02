"""
Multisensory super-additivity -- a spiking superior-colliculus cell that fuses two
cross-modal cues (e.g. visual + auditory) the way Stein & Meredith's deep-SC neurons do:
COINCIDENT WEAK cues drive a response GREATER than the sum of the two unimodal responses
(multisensory enhancement), the relative gain is LARGEST when the cues are weak and SHRINKS
to sub-additive as either cue strengthens (inverse effectiveness), and cues that are NOT
spatially/temporally coincident produce NO enhancement (the spatial/temporal rule).
Deterministic.

Stein & Meredith (1993, The Merging of the Senses; Meredith & Stein, J Neurophysiol 1986)
found single deep superior-colliculus neurons that receive converging inputs from more than
one sense. Three lawful properties define their multisensory integration:
  1. SUPER-ADDITIVITY   -- a spatially and temporally coincident cross-modal pair evokes MORE
                           impulses than the arithmetic SUM of the two modality-alone responses
                           (a combined response that is not just additive but multiplicative).
  2. INVERSE EFFECTIVENESS -- the proportionate enhancement is GREATEST for WEAK stimuli that
                           are barely effective alone, and it SHRINKS (going sub-additive) as
                           each unimodal input becomes individually strong.
  3. SPATIAL/TEMPORAL RULE -- enhancement requires the inputs to fall within the same receptive
                           field AND within a short temporal window; disparate (non-coincident)
                           inputs summate little or not at all -- no enhancement (or depression).

All three fall out of ONE leaky integrate-and-fire cell, with NO fitting or training:

  * THRESHOLD RECTIFICATION gives super-additivity. A weak unimodal current leaves the membrane at or
    below the spike threshold, so it fires little or nothing -- the spike threshold RECTIFIES a weak
    cue's response toward the floor (0). Two coincident weak currents ADD inside the membrane and push
    it well ABOVE threshold, firing far more than twice the (near-zero) single-cue response:
    response(2c) >> 2 * response(c). It is this EXPANSIVE threshold nonlinearity (near-0 below, firing
    above) -- NOT any special curvature of the F-I curve above rheobase -- that makes the enhancement
    index MS / (unimodal_A + unimodal_B) exceed 1. (Honest correction: the LIF F-I curve above rheobase
    is concave/saturating, not convex; super-additivity here is the rectifying threshold, not F-I shape.)
  * F-I SATURATION gives inverse effectiveness. As the cues strengthen, each unimodal current ALONE
    already drives the cell up toward its refractory-limited ceiling rate, where the F-I curve is
    CONCAVE (compressive). There summing two strong currents can no longer roughly double an already-
    high rate, so response(2c) < 2 * response(c): the index falls THROUGH 1 into sub-additive. Sweeping
    cue strength therefore moves the cell from the THRESHOLD-dominated (expansive -> super-additive)
    regime at weak drive to the SATURATION-dominated (compressive -> sub-additive) regime at strong
    drive -- inverse effectiveness is that expansive-to-compressive transition, not one convex region.
  * The LEAK gives the coincidence rule. The membrane is a leaky integrator: two currents add
    inside the cell only while both are present within one membrane time-constant. If the second
    cue is delayed past the integration window, the first cue's depolarization has LEAKED back
    toward rest before the second arrives, so the two never summate -- the cell responds to each
    separately and MS -> unimodal_A + unimodal_B, index -> 1 (no enhancement). Spatial disparity
    produces the same outcome by the analogous rule (the second cue falls outside the cell's
    receptive field, contributing ~no current); here we model the temporal offset, which is the
    same coincidence detector the leak already implements.

Reuses the repo's canonical spiking primitive (does NOT reimplement a neuron):
  * growth.lif_neuron.LIFNeuron -- the leaky integrate-and-fire cell. Each modality contributes an
    input current; the two currents are SUMMED per timestep and fed to ONE LIFNeuron, so the whole
    phenomenon is genuine spiking membrane dynamics, not an algebraic rule about spike counts.

DISTINCTNESS (this is single-cell threshold ENHANCEMENT, a different mechanism from its neighbors):
  * cortex.cue_integration fuses several noisy population estimates of ONE quantity by inverse-
    variance spike ADDITION (Ma-Pouget PPC): its fused population is optimally PRECISE, but a
    linear sum of Poisson populations is by construction ADDITIVE -- it makes no super-additivity
    claim. THIS cell is deliberately NON-linear: a single threshold unit whose combined response
    EXCEEDS the sum, and it reports the sub-additive regime too. The two are the two classic and
    complementary accounts of multisensory integration (optimal-cue-combination vs SC enhancement).
  * sensorimotor.multimodal_encoder BANDS four modalities into disjoint slots of the N=64 feature vector
    (concatenation, no interaction). THIS makes the modalities INTERACT inside one cell.
  * faculties.superior_colliculus is a WHERE map -- a spiking winner-take-all that SELECTS which of
    many locations to orient to. THIS is a HOW-MUCH cell -- how strongly one location's cross-modal
    pair drives the SC, the enhancement that makes a faint bimodal target detectable. They compose:
    enhancement here raises a weak target's salience; the colliculus map there picks the winner.

HONEST FRAMING / NEGATIVE: this is the canonical Stein-Meredith PHENOMENOLOGY reproduced in a
toy LIF cell, NOT a trained cortical map, not real vision or audition, and not a fit to recorded
SC data. The cues are synthetic scalar currents; "visual" and "auditory" are just two current
sources into one neuron. The three laws are shown to be forced by threshold + saturation + leak
in an untrained integrate-and-fire unit -- an existence/mechanism demonstration of WHY the laws
hold, not a claim to match any particular neuron's tuning. Deterministic (per-instance RNG used
only for the randomized robustness battery; the enhancement measurements carry no randomness).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.multisensory_integration
"""

import random

from snn_data_lake.growth.lif_neuron import LIFNeuron


class MultisensoryCell:
    """One spiking SC cell fed by two cross-modal current sources (visual + auditory).

    A cue of `strength` s delivers a constant current s per timestep over a `dur`-step window.
    respond() runs a fresh LIFNeuron over the summed current trace and returns the spike count;
    enhancement() compares the coincident bimodal response to the two unimodal responses and
    reports the enhancement index MS / (unimodal_visual + unimodal_auditory): index > 1 is
    super-additive, index < 1 is sub-additive, index ~= 1 is no enhancement."""

    def __init__(
        self,
        tau_m=20.0,
        v_threshold=1.0,
        v_reset=0.0,
        refractory=1,
        dur=600,
        onset=20,
        tail=40,
        seed=0,
    ):
        self.tau_m = tau_m  # membrane time constant (sets leak = coincidence window)
        self.v_threshold = v_threshold  # LIF firing threshold (the super-additive non-linearity)
        self.v_reset = v_reset  # membrane potential after a spike
        self.refractory = refractory  # refractory steps -> saturation ceiling 1/(refractory+1)
        self.dur = dur  # steps each cue is active (long window -> smooth F-I)
        self.onset = onset  # step the (visual) cue turns on
        self.tail = tail  # extra quiet steps so the last spike is not clipped
        self.rng = random.Random(seed)  # per-instance RNG; ONLY for the randomized battery

    def _neuron(self):
        """A fresh, reset LIF cell for one independent trial (run() re-initializes its state)."""
        return LIFNeuron(
            tau_m=self.tau_m,
            v_threshold=self.v_threshold,
            v_reset=self.v_reset,
            refractory=self.refractory,
        )

    def respond(self, visual, auditory, offset=0):
        """Total spikes when the visual cue (current `visual`) is on over [onset, onset+dur) and
        the auditory cue (current `auditory`) is on over [onset+offset, onset+offset+dur). The two
        currents are SUMMED per timestep and integrated by ONE LIF cell. offset >= 0 delays the
        auditory cue; offset 0 is perfectly coincident. A modality with strength 0 is absent."""
        if offset < 0:
            raise ValueError("offset must be >= 0 (auditory cue delayed relative to visual)")
        v_end = self.onset + self.dur if visual > 0.0 else 0
        a_start = self.onset + offset
        a_end = a_start + self.dur if auditory > 0.0 else 0
        total = max(v_end, a_end, self.onset) + self.tail
        trace = [0.0] * total
        if visual > 0.0:
            for t in range(self.onset, self.onset + self.dur):
                trace[t] += visual
        if auditory > 0.0:
            for t in range(a_start, a_start + self.dur):
                trace[t] += auditory
        return sum(self._neuron().run(trace))

    def enhancement(self, strength_v, strength_a, offset=0):
        """Measure multisensory enhancement for a cross-modal pair.

        Returns a dict with the two unimodal spike counts (uv, ua), the bimodal spike count (ms),
        their sum (denom = uv + ua), and the enhancement index = ms / denom. index > 1 means the
        bimodal response EXCEEDS the sum of the unimodal responses (super-additive); index < 1 is
        sub-additive; index ~= 1 is purely additive (no enhancement). The index is None when both
        unimodal responses are silent (undefined -- unbounded enhancement)."""
        uv = self.respond(strength_v, 0.0)
        ua = self.respond(0.0, strength_a)
        ms = self.respond(strength_v, strength_a, offset)
        denom = uv + ua
        index = ms / denom if denom else None
        return {"uv": uv, "ua": ua, "ms": ms, "denom": denom, "index": index}


def main():
    cell = MultisensoryCell(seed=0)

    # ---- (1) super-additivity: two WEAK, coincident cues -> MS exceeds the unimodal SUM --------
    weak = 0.055
    e = cell.enhancement(weak, weak, offset=0)

    # ---- (2) inverse effectiveness: enhancement index vs cue strength (weak -> strong) ---------
    sweep = [0.055, 0.07, 0.09, 0.12, 0.17, 0.25, 0.40]
    curve = [cell.enhancement(s, s, offset=0) for s in sweep]
    idx = [c["index"] for c in curve]

    # ---- (3) control: the SAME weak cues, but NON-COINCIDENT (auditory delayed past the window)
    far = cell.dur + 200  # fully separated -> no temporal overlap
    e_nc = cell.enhancement(weak, weak, offset=far)
    # temporal window: enhancement decays smoothly as the auditory cue slides out of coincidence
    offsets = [0, 30, 60, 120, 240, far]
    window = [cell.enhancement(weak, weak, offset=o)["index"] for o in offsets]

    print("=" * 72)
    print("MULTISENSORY INTEGRATION -- spiking superior-colliculus super-additivity")
    print("=" * 72)
    print(
        f"LIF cell: tau_m {cell.tau_m} | threshold {cell.v_threshold} | refractory "
        f"{cell.refractory} (ceiling rate 1/{cell.refractory + 1}) | cue window {cell.dur} steps"
    )
    print("-" * 72)
    print(f"(1) SUPER-ADDITIVITY  weak coincident cues (v={weak}, a={weak}):")
    print(
        f"    unimodal visual = {e['uv']} spikes, unimodal auditory = {e['ua']} spikes, "
        f"sum = {e['denom']}"
    )
    print(
        f"    bimodal (coincident) = {e['ms']} spikes -> enhancement index = "
        f"{e['index']:.2f}  (> 1: MS EXCEEDS the unimodal sum)"
    )
    print("-" * 72)
    print("(2) INVERSE EFFECTIVENESS  enhancement index falls as cue strength rises:")
    for s, c in zip(sweep, curve):
        tag = "super-additive" if c["index"] > 1.0 else "sub-additive"
        print(
            f"    strength {s:.3f}: uni {c['uv']:3d}+{c['ua']:3d}={c['denom']:3d}  bimodal "
            f"{c['ms']:3d}  index {c['index']:.3f}  ({tag})"
        )
    print("-" * 72)
    print("(3) CONTROL -- coincidence rule (same weak cues, auditory cue delayed):")
    for o, w in zip(offsets, window):
        print(f"    offset {o:4d} steps -> index {w:.3f}")
    print(
        f"    coincident index {e['index']:.2f}  vs  non-coincident index {e_nc['index']:.2f} "
        f"(offset {far}): enhancement is GONE when the cues do not coincide"
    )
    print("=" * 72)

    # ---- un-riggable self-checks (deterministic asserts) -------------------------------------
    # (1) SUPER-ADDITIVITY: the coincident bimodal response exceeds the sum of the unimodal ones.
    assert e["denom"] > 0, "both unimodal responses must be non-silent (index well-defined)"
    assert e["ms"] > e["denom"], "coincident weak cues: bimodal spikes must EXCEED the unimodal sum"
    assert e["index"] > 1.0, "super-additivity means enhancement index > 1"

    # (2) INVERSE EFFECTIVENESS: the index decreases MONOTONICALLY with cue strength, starting
    #     super-additive (> 1) and ending sub-additive (< 1). No fitting -- this is the F-I shape.
    assert all(c["denom"] > 0 for c in curve), "every sweep point must have a defined index"
    assert all(a > b for a, b in zip(idx, idx[1:])), (
        f"enhancement index must strictly decrease as cue strength rises: {[round(v, 3) for v in idx]}"
    )
    assert idx[0] > 1.0, "the weakest cues must be super-additive (index > 1)"
    assert idx[-1] < 1.0, "the strongest cues must be sub-additive (index < 1)"

    # (3) COINCIDENCE RULE is LOAD-BEARING: identical cue strengths that are NOT coincident produce
    #     NO enhancement (index ~= 1), strictly less than the coincident case. Uses the LIF leak:
    #     a delayed cue's partner has leaked back to rest, so the two currents never summate.
    assert e_nc["index"] < e["index"], (
        "non-coincident cues must enhance LESS than coincident cues (coincidence is load-bearing)"
    )
    assert abs(e_nc["index"] - 1.0) < 1e-9, (
        "fully non-coincident cues must be purely additive (index == 1, no enhancement)"
    )
    # the temporal window monotonically closes: enhancement decays as the auditory cue slides out.
    assert all(a >= b for a, b in zip(window, window[1:])), (
        f"enhancement must decay monotonically as the cues separate in time: {[round(v, 3) for v in window]}"
    )
    assert window[0] > 1.0 and abs(window[-1] - 1.0) < 1e-9, (
        "the window must run from super-additive (coincident) down to additive (fully offset)"
    )

    # (4) GENUINELY SPIKING: real LIF membrane dynamics (leak + threshold + reset),
    #     discrete integer spike counts.
    n = cell._neuron()
    assert 0.0 < n.decay < 1.0, "the membrane must genuinely leak (leaky integrate-and-fire)"
    assert n.v_threshold > 0.0 and hasattr(n, "v_reset"), "a firing threshold and reset must exist"
    assert all(isinstance(c["ms"], int) and isinstance(c["uv"], int) for c in curve), (
        "responses must be discrete integer spike counts"
    )

    # (5) RANDOMIZED ROBUSTNESS: across many random WEAK coincident pairs, enhancement is
    #     super-additive AND always beats the non-coincident control (per-instance RNG, seeded).
    batt = MultisensoryCell(seed=12345)
    n_super = 0
    for _ in range(30):
        sv = batt.rng.uniform(0.050, 0.090)
        sa = batt.rng.uniform(0.050, 0.090)
        co = batt.enhancement(sv, sa, offset=0)
        nc = batt.enhancement(sv, sa, offset=batt.dur + 200)
        assert co["index"] > 1.0, (
            f"weak coincident cues must be super-additive (v={sv:.3f}, a={sa:.3f})"
        )
        assert co["index"] > nc["index"], "coincident must beat non-coincident for every weak pair"
        n_super += 1

    # (6) DETERMINISM: same construction -> byte-identical measurements (no wall-clock, no global RNG).
    assert MultisensoryCell(seed=0).enhancement(weak, weak) == cell.enhancement(weak, weak), (
        "same parameters must reproduce identical enhancement measurements"
    )

    print(
        f"self-check OK: super-additivity index {e['index']:.2f} (>1); inverse effectiveness "
        f"{idx[0]:.2f}->{idx[-1]:.2f} strictly falls through 1;"
    )
    print(
        f"  non-coincident control index {e_nc['index']:.2f} (== additive, no enhancement) < "
        f"coincident {e['index']:.2f}; temporal window {window[0]:.2f}->{window[-1]:.2f};"
    )
    print(
        f"  {n_super}/30 random weak pairs super-additive and beat the control; LIF leak/threshold/"
        f"reset; deterministic"
    )


if __name__ == "__main__":
    main()
