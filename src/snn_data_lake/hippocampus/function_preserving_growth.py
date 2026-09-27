"""
L6 / SI-10 -- FUNCTION-PRESERVING SPIKING GROWTH (Net2Net split).

Widen a live spiking layer WITHOUT disturbing its function. To add capacity to a
running LIF hidden layer we DUPLICATE a live unit: copy its incoming weights AND
its membrane state (v, refractory counter) EXACTLY, then HALVE the two copies'
OUTGOING weights. The duplicate receives identical input from identical state, so
it emits a BIT-IDENTICAL spike train to its parent forever after. Its two
half-weight fan-outs recombine to the parent's original fan-out:

    w * 0.5            # divide by a power of two -- EXACT in IEEE754
    (w/2) + (w/2) == w # equal-magnitude add of the two halves -- EXACT in IEEE754

so the downstream summed current -- and therefore every downstream spike -- is
BIT-FOR-BIT unchanged. The network grows capacity mid-stream with ZERO functional
disruption. This is the WHOLE honest story: an EPS-INVARIANT (exact output
preservation), proven by exact float equality (max abs diff == 0.0), not a
tolerance.

Downstream currents are summed with math.fsum -- a single correct rounding of the
exact real sum of the addends. Plain left-fold `+` is NOT associative, so
splitting one addend w*s into (w/2)*s + (w/2)*s could re-round differently and
break the last bit; fsum is order/grouping independent, so the exact real sum
(which is unchanged, because w/2 is exact and the two halves add back to w) rounds
to the identical float. That is what makes the invariant literally bit-exact.

DE-SCOPE (honest, per the SI-10 spec): we make NO "GradMax provably accelerates
learning" claim. GradMax-style growth here inserts a brand-new unit with OUTGOING
weights EXACTLY 0.0 -- also function-preserving at insertion (0 * anything adds
nothing) -- and seeds its INCOMING weights from an eligibility-directed HEURISTIC.
That seed is an UNPROVEN heuristic: any speed-up is scale-dependent and is pure
noise at a few units. No learning-speed assertion is made anywhere.

Reuses (does NOT reimplement):
  * cortex.lif_neuron.LIFNeuron      -- the canonical spiking unit + its membrane state
  * infra._mathutils.dot, unit   -- incoming current; heuristic-seed normalization

Deterministic (per-instance random.Random, no global RNG / no wall-clock). Pure
standard library. ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.function_preserving_growth
"""

import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.infra._mathutils import dot, unit

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "SpikingGrowthLayer.add_unit_gradmax": (
        "NOT_A_RULE",
        "STRUCTURAL GROWTH, not a learning rule: it INSERTS a hidden unit with outgoing weights "
        "exactly 0.0 (function-preserving at insertion) and incoming weights from an "
        "eligibility-directed seed. It creates a unit rather than updating existing ones from "
        "experience, and the method's own docstring disclaims any learning claim -- 'the seed is an "
        "UNPROVEN heuristic. No learning-speed claim is made'. Enumerated on the 'grad' in its name",
    ),
}


class SpikingGrowthLayer:
    """A tiny feed-forward spiking net: n_in inputs -> n_hidden LIF units -> n_out
    LIF readouts, dense weights w_in[hidden][in] and w_out[hidden][out]. Supports
    function-preserving widening (Net2Net split) and GradMax-style zero-fanout
    insertion. Hidden units are independent (no lateral wiring), so splitting one
    leaves every other unit's dynamics untouched."""

    def __init__(
        self, n_in=6, n_hidden=4, n_out=2, seed=0, tau_m=10.0, v_threshold=1.0, refractory=0
    ):
        rng = random.Random(seed)
        self.n_in = n_in
        self.n_out = n_out
        self._hp = {"tau_m": tau_m, "v_threshold": v_threshold, "refractory": refractory}
        self.w_in = [[rng.uniform(0.0, 1.0) for _ in range(n_in)] for _ in range(n_hidden)]
        self.w_out = [[rng.uniform(0.3, 0.9) for _ in range(n_out)] for _ in range(n_hidden)]
        self.hidden = [LIFNeuron(**self._hp) for _ in range(n_hidden)]
        self.readout = [LIFNeuron(**self._hp) for _ in range(n_out)]
        self.reset_state()

    @property
    def n_hidden(self):
        return len(self.hidden)

    def reset_state(self):
        for u in self.hidden:
            u.reset_state()
        for k in self.readout:
            k.reset_state()

    def step(self, x):
        """Advance one timestep with input vector x (length n_in). Returns
        (currents[n_out], out_spikes[n_out], hidden_spikes[n_hidden]).

        The downstream current for readout k is math.fsum over hidden units of
        w_out[u][k] * spike[u] -- the exact-real-sum that makes the split invariant
        hold to the last bit."""
        hs = [self.hidden[u].step(dot(self.w_in[u], x)) for u in range(self.n_hidden)]
        currents = [
            math.fsum(self.w_out[u][k] * hs[u] for u in range(self.n_hidden))
            for k in range(self.n_out)
        ]
        ys = [self.readout[k].step(currents[k]) for k in range(self.n_out)]
        return currents, ys, hs

    def run(self, window):
        """Reset state, then step the whole input window. Returns
        (currents_seq, out_spike_seq, hidden_spike_seq), each a list over time."""
        self.reset_state()
        cur, out, hid = [], [], []
        for x in window:
            c, y, h = self.step(x)
            cur.append(c)
            out.append(y)
            hid.append(h)
        return cur, out, hid

    def _clone_neuron(self, src):
        """A new LIFNeuron with identical parameters AND identical membrane state
        (v and the refractory counter) copied EXACTLY from src. Exact state copy is
        what lets a mid-stream duplicate track its parent bit-for-bit."""
        twin = LIFNeuron(**self._hp)
        twin.v = src.v
        twin._refr = src._refr
        return twin

    def duplicate_unit(self, p, factor=0.5, copy_state=True):
        """Net2Net function-preserving widening. Duplicate hidden unit p: copy its
        incoming weights exactly, copy its live membrane state exactly (unless
        copy_state=False), and scale BOTH copies' OUTGOING weights by `factor`.

        factor=0.5 is the UNIQUE function-preserving choice: the duplicate emits a
        bit-identical spike train and (w/2)+(w/2)==w recombines the fan-out exactly,
        so the downstream current is unchanged to the last bit. Any other factor
        perturbs the downstream current (see the control in the self-check).
        Returns the new unit's id."""
        self.w_in.append(list(self.w_in[p]))  # incoming weights: exact copy
        if copy_state:
            self.hidden.append(self._clone_neuron(self.hidden[p]))
        else:
            self.hidden.append(LIFNeuron(**self._hp))  # control: state NOT copied
        parent_out = self.w_out[p]
        self.w_out[p] = [w * factor for w in parent_out]  # halve parent fan-out
        self.w_out.append([w * factor for w in parent_out])  # halve duplicate fan-out
        return self.n_hidden - 1

    def add_unit_gradmax(self, window=None):
        """GradMax-style growth: insert a brand-new hidden unit with OUTGOING
        weights EXACTLY 0.0 (function-preserving at insertion) and INCOMING weights
        from an eligibility-directed HEURISTIC seed derived from `window`.

        HONEST NEGATIVE: the seed is an UNPROVEN heuristic. No learning-speed claim
        is made -- acceleration is scale-dependent and is pure noise at a few units.
        Returns the new unit's id."""
        self.w_in.append(self._eligibility_seed(window))
        self.hidden.append(LIFNeuron(**self._hp))
        self.w_out.append([0.0] * self.n_out)  # zero fan-out -> exact at insertion
        return self.n_hidden - 1

    def _eligibility_seed(self, window):
        """Heuristic in-weight seed: the unit-normalized mean input over `window`
        (a crude 'point the new unit where the input energy is' eligibility trace).
        Returns a fresh in-weight vector. UNPROVEN -- see add_unit_gradmax. Because
        the new unit's fan-out is 0.0, this seed cannot affect the eps-invariant."""
        if not window:
            return [0.0] * self.n_in
        cols = [math.fsum(x[i] for x in window) / len(window) for i in range(self.n_in)]
        return list(unit(cols))


def _max_abs_diff(a, b):
    """Max absolute elementwise difference between two current sequences
    (list-of-list of floats). 0.0 iff bit-for-bit identical."""
    m = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            d = abs(x - y)
            if d > m:
                m = d
    return m


def _make_window(n_steps, n_in, seed):
    """Deterministic input window: n_steps input vectors in [0,1]^n_in."""
    rng = random.Random(seed)
    return [[rng.uniform(0.0, 1.0) for _ in range(n_in)] for _ in range(n_steps)]


def main():
    n_in, n_hid, n_out, n_steps, seed = 6, 4, 2, 24, 0
    window = _make_window(n_steps, n_in, seed=101)

    # refractory>0 keeps the units out of saturation, so at t=mid a unit carries live
    # membrane state (a nonzero refractory counter) -- which is what the mid-stream
    # split must copy exactly. mk() builds identical layers so every branch is comparable.
    def mk():
        return SpikingGrowthLayer(n_in, n_hid, n_out, seed=seed, refractory=3)

    base = mk()
    cur0, out0, hid0 = base.run(window)

    # split the MOST ACTIVE hidden unit so the invariant + control are non-vacuous
    spikes_per_unit = [sum(h[u] for h in hid0) for u in range(n_hid)]
    p = max(range(n_hid), key=lambda u: spikes_per_unit[u])
    assert spikes_per_unit[p] > 0, "need an active unit for a non-vacuous invariant + control"

    print("=" * 66)
    print("FUNCTION-PRESERVING SPIKING GROWTH (Net2Net split) -- L6 / SI-10")
    print("=" * 66)
    print(
        f"layer {n_in}->{n_hid} LIF->{n_out}   window={n_steps} steps   "
        f"split unit p={p} ({spikes_per_unit[p]} spikes)"
    )

    # (1a) EXACT INVARIANT -- fresh re-run after a halving split == baseline, bit-for-bit.
    g = mk()
    g.duplicate_unit(p, factor=0.5)
    cur1, out1, _ = g.run(window)
    diff_split = _max_abs_diff(cur0, cur1)
    assert g.n_hidden == n_hid + 1, "split must add exactly one unit"
    assert diff_split == 0.0, f"halving split must preserve output EXACTLY, got {diff_split!r}"
    assert out1 == out0, "downstream spike train must be identical after a halving split"

    # (1b) MID-STREAM INVARIANT -- split at t=mid COPYING the live membrane state, then
    #      continue; identical to the un-grown net at every step. This is where the exact
    #      membrane-state copy is load-bearing (the fresh re-run in 1a does not exercise it).
    #      The baseline pass also records unit p's pre-step state so we can pick a split time
    #      where the parent genuinely HOLDS live state (nonzero v or an active refractory).
    baseline = mk()
    baseline.reset_state()
    cur_b, live = [], []
    for x in window:
        live.append((baseline.hidden[p].v, baseline.hidden[p]._refr))
        cur_b.append(baseline.step(x)[0])
    cand = [t for t in range(3, n_steps - 3) if live[t][0] != 0.0 or live[t][1] > 0]
    mid = cand[len(cand) // 2] if cand else n_steps // 2
    v_at_split, refr_at_split = live[mid]

    grown = mk()
    grown.reset_state()
    cur_m = []
    for t, x in enumerate(window):
        if t == mid:
            grown.duplicate_unit(p, factor=0.5, copy_state=True)
        cur_m.append(grown.step(x)[0])
    diff_mid = _max_abs_diff(cur_b, cur_m)
    # the split is only a real test of state-copy if the parent HELD live state at t=mid
    assert v_at_split != 0.0 or refr_at_split > 0, "mid-stream split must exercise live state"
    assert diff_mid == 0.0, f"mid-stream halving split must be EXACT, got {diff_mid!r}"

    # informational: the SAME mid-stream split WITHOUT copying membrane state perturbs the
    # output -- the fresh duplicate mis-fires relative to its clamped parent (not asserted; it
    # could legitimately be 0.0 in a saturating regime where initial state is washed out).
    nostate = mk()
    nostate.reset_state()
    cur_ns = []
    for t, x in enumerate(window):
        if t == mid:
            nostate.duplicate_unit(p, factor=0.5, copy_state=False)
        cur_ns.append(nostate.step(x)[0])
    diff_nostate = _max_abs_diff(cur_b, cur_ns)

    # (2) CONTROL THAT MUST FAIL -- split WITHOUT halving (factor=1.0) doubles unit p's
    #     fan-out -> the downstream current changes -> the bit-identical assertion FAILS.
    c = mk()
    c.duplicate_unit(p, factor=1.0)
    cur_c, _, _ = c.run(window)
    diff_nohalve = _max_abs_diff(cur0, cur_c)
    assert diff_nohalve > 0.0, "control: a split without halving MUST perturb the output"

    # (3) GRADMAX-STYLE ADD -- a brand-new unit with OUT-weights == 0.0 leaves the output
    #     bit-identical at insertion (0 * anything contributes nothing to the fsum).
    gm = mk()
    gm.add_unit_gradmax(window=window)
    cur_gm, out_gm, _ = gm.run(window)
    diff_gradmax = _max_abs_diff(cur0, cur_gm)
    assert gm.n_hidden == n_hid + 1, "gradmax add must add exactly one unit"
    assert diff_gradmax == 0.0, f"zero-fanout insertion must preserve output, got {diff_gradmax!r}"
    assert out_gm == out0, "downstream spikes unchanged by a zero-fanout insertion"

    # (5) DETERMINISM -- two builds from the same seed produce identical weights + output.
    d1, d2 = mk(), mk()
    d1.duplicate_unit(p, factor=0.5)
    d2.duplicate_unit(p, factor=0.5)
    assert d1.run(window)[0] == d2.run(window)[0], "same seed -> identical output"
    assert d1.w_out == d2.w_out, "same seed -> identical grown weights"

    print(
        f"1a EXACT (fresh)   : split unit {p}, halve fan-out, re-run -> max abs diff "
        f"= {diff_split!r}  (bit-identical)"
    )
    print(
        f"1b EXACT (stream)  : split at t={mid} copying live state (v={v_at_split:.4f}, "
        f"refr={refr_at_split}) -> max abs diff = {diff_mid!r}"
    )
    print(
        f"   membrane matters: SAME split WITHOUT state copy -> max abs diff = "
        f"{diff_nostate!r} (perturbed; informational)"
    )
    print(
        f"2  CONTROL (fail)  : split WITHOUT halving (factor=1.0) -> max abs diff = "
        f"{diff_nohalve!r} (nonzero -> exact-equality FAILS, as it must)"
    )
    print(
        f"3  GRADMAX add     : new unit, OUT-weights == 0.0 -> max abs diff = "
        f"{diff_gradmax!r} (bit-identical at insertion)"
    )
    print("5  DETERMINISM     : identical grown weights + output across two builds")
    print("-" * 66)
    print("HONEST NEGATIVE: the GradMax in-weight seed is an eligibility-directed")
    print("  UNPROVEN heuristic. NO learning-speed claim is made -- any acceleration")
    print("  is scale-dependent and is pure noise at a few units. The shipped, proven")
    print("  result is the eps-invariant ALONE: exact bit-for-bit output preservation.")
    print("=" * 66)
    print(
        f"self-check OK: halving split exact (diff {diff_split!r} fresh / {diff_mid!r} "
        f"mid-stream), no-halve control fails (diff {diff_nohalve!r}), zero-fanout add "
        f"exact (diff {diff_gradmax!r}), deterministic"
    )


if __name__ == "__main__":
    main()
