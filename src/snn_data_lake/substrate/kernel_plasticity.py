"""THE kernel-side driver of the ONE R-STDP rule -- the first thing in this tree that can write
a weight into a `Synapses` bundle table (specs/022 T063 2.1).

Before this module the kernel could not learn, measured rather than assumed: `compile_bundles`
quantizes weights ONCE at construction, `Synapses` has exactly two methods and no setter, and
`KernelTick.tick` writes only v / refrac / last_spike. No rule in `plasticity/*`, in
`cortex/eprop_kernel` (torch, offline encoders) or behind the 0x09 writer (NOT_A_RULE, dense-only)
can touch a bundle table. T063 asks "does learning move the brain off the seed", and that question
has NO MECHANISM on the kernel route without a learner -- so the learner lands with the row.

WHAT THIS IS NOT. It is not a second R-STDP: `basal_ganglia/r_stdp.py` owns the rule and this
module calls its module-level functions (the P2 merge; `tests/test_r_stdp_kernel_equivalence.py`
feeds one spike/reward sequence through the per-object neuron and through this driver and requires
the same weights to within the quantization step). It is not on the live tick and it does not
discharge `kernel_tick`/`kernel_state`'s "wire" exemptions -- an instrument that ticks a scratch
population is not the live consumer those rows name.

THE THREE INVARIANTS, each with a reachable red in `tests/test_kernel_plasticity.py`:
  1. SIGN IS PRESERVED. A weight moves toward zero and MAY REACH zero, never crosses it (Dale is a
     property of the cell, and SC-004 asserts it survives learning). The red: construct the same
     sequence with `sign_preserving=False` and a flip occurs.
  2. DETERMINISM. The same construction and the same reward sequence give BYTE-EQUAL weights --
     the determinism gate's contract over `kernel_tick` is not weakened by a table that now moves.
  3. FOUR DIAGNOSTIC LINES, never one. EXECUTED (reward calls), ATTEMPTED (edges with nonzero
     eligibility at a reward), EFFECTIVE COUNT and EFFECTIVE L1 (entries actually changed, and by
     how much). A zero is read from all four: `attempted > 0, effective == 0` is "cancelled or
     below quantization", `attempted == 0` is "no eligible spike pairs", and NEITHER is pinning.
     Only a verified `lr = 0` intervention is pinning, and the harness -- not this module -- is
     the thing that applies it.

THE INTRA-TICK ORDER IS FROZEN, because it is the only place the two substrates could diverge:
  (1) POST phase -- every fired unit pairs with its IN-edges against pre-spike times from BEFORE
      this tick (dt > 0, causal, LTP);
  (2) record the fired units' post-spike times;
  (3) PRE phase -- every fired unit pairs with its OUT-edges against post-spike times as of (2)
      (dt < 0, anti-causal, LTD; a post that fired in THIS tick gives dt == 0 and contributes
      nothing, which is exactly what `RSTDPNeuron.pre_spike` does after `post_spike`);
  (4) record the fired units' pre-spike times.
That is `post_spike` then `pre_spike`, per tick, which is the per-object form's own order.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.kernel_plasticity
"""

import sys

from snn_data_lake.basal_ganglia.r_stdp import (
    A_MINUS,
    A_PLUS,
    LR,
    TAU_E,
    TAU_MINUS,
    TAU_PLUS,
    accumulate_anticausal,
    accumulate_causal,
    decay_eligibility,
    reward_deltas,
)

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
#:
GRADIENT_CLAIM = {
    "KernelPlasticity.reward": (
        "LOCAL",
        "the kernel form of RSTDPNeuron.reward, whose own claim this mirrors: a broadcast scalar "
        "reward gates the STORED per-edge eligibility into the quantized bundle weights. The "
        "eligibility is formed locally from pre-before-post spike timing on incident edges and "
        "NOTHING is propagated backward -- no error is transported along the graph, no weight is "
        "read by another synapse's update -- so it owes no cosine under the e-prop invariant's "
        "SCOPE clause. The delta itself is computed by r_stdp.reward_deltas, not here",
    ),
    "KernelPlasticity.update_eligibility": (
        "LOCAL",
        "forms the eligibility trace from THIS tick's spike set alone: a fired unit pairs with "
        "its incident edges' opposite-endpoint spike times via r_stdp.accumulate_causal and "
        "accumulate_anticausal. It is a local coincidence detector with no error signal and no "
        "gradient of any kind; the third factor arrives later, in reward()",
    ),
}


class KernelPlasticity:
    """Eligibility arrays beside every bundle's `qweight`, and the reward gate that writes them.

    The bundle tables the tick reads are re-pointed at VIEWS into one flat int64 array, so a
    weight written here is the weight `KernelTick._deliver_and_rotate` delivers on the next tick.
    A learner that wrote a private copy would be a null wearing a mechanism's name.

    distinct-from: plasticity/structural_rewiring.py::StructuralRewiring.rewire -- mutates DENSE
      weight rows in place on the sleep tick and needs rows[r][c] addressing; a bundle table is
      (post_idx, qweight) arrays keyed (pre, partition, delay). Read 2026-09-09.
    distinct-from: plasticity/meta_plasticity.py::MetaPlasticLearner -- its per-unit R-STDP-shaped
      competitor moves LEARNING RATES from a reward history; it never touches a synapse table and
      has no graph. Read 2026-09-09.
    distinct-from: cortex/interaction_learning.py::SpikingInteractionLearner._record_synapse_delta
      -- NOT_A_RULE: it reads weights another rule already moved and writes them to the lake.
      Read 2026-09-09.
    """

    def __init__(
        self,
        kt,
        lr=LR,
        a_plus=A_PLUS,
        a_minus=A_MINUS,
        tau_plus=TAU_PLUS,
        tau_minus=TAU_MINUS,
        tau_e=TAU_E,
        sign_preserving=True,
    ):
        import numpy as np

        self._np = np
        self.kt = kt
        self.syn = kt.syn
        self.frac_bits = int(kt.syn.frac_bits)
        self.scale = float(1 << self.frac_bits)
        self.lr = float(lr)
        self.a_plus, self.a_minus = float(a_plus), float(a_minus)
        self.tau_plus, self.tau_minus = float(tau_plus), float(tau_minus)
        self.tau_e = float(tau_e)
        self.sign_preserving = bool(sign_preserving)

        # ---- one flat edge table; every bundle's qweight becomes a VIEW into it -------------
        keys = sorted(self.syn.bundles)
        pre, post, qs, spans = [], [], [], []
        at = 0
        for k in keys:
            p, q = self.syn.bundles[k]
            pre.extend([k[0]] * len(p))
            post.extend(p.tolist())
            qs.append(q)
            spans.append((at, at + len(p)))
            at += len(p)
        self.n_edges = at
        self.qflat = np.concatenate(qs).astype(np.int64) if qs else np.zeros(0, dtype=np.int64)
        for k, (lo, hi) in zip(keys, spans, strict=True):
            self.syn.bundles[k] = (self.syn.bundles[k][0], self.qflat[lo:hi])
        self.edge_pre = np.asarray(pre, dtype=np.int64)
        self.edge_post = np.asarray(post, dtype=np.int64)
        self.q_start = self.qflat.copy()

        # ---- CSR indices: a fired unit's out-edges and in-edges, in O(degree) --------------
        n = int(kt.state.n_units)
        self._out_ptr, self._out_idx = _csr(self.edge_pre, n)
        self._in_ptr, self._in_idx = _csr(self.edge_post, n)

        self.elig = np.zeros(self.n_edges, dtype=np.float64)
        self._t_pre = np.full(n, np.nan, dtype=np.float64)
        self._t_post = np.full(n, np.nan, dtype=np.float64)
        self.executed = 0
        self.attempted = 0
        self.effective_count = 0
        self.effective_l1 = 0.0
        self.clamped = 0

    # ---- eligibility ----------------------------------------------------------------------
    def update_eligibility(self, fired, t):
        """Form this tick's eligibility from its spike set, in the frozen order of the header."""
        np = self._np
        fired = np.asarray(fired, dtype=np.int64)
        if fired.size == 0:
            return
        t = float(t)
        t_pre, t_post = self._t_pre, self._t_post
        in_ptr, in_idx = self._in_ptr, self._in_idx
        out_ptr, out_idx = self._out_ptr, self._out_idx
        pre_of, post_of = self.edge_pre, self.edge_post

        # (1) POST phase: causal pairs against pre times from BEFORE this tick
        pairs = []
        for u in fired.tolist():
            for j in in_idx[in_ptr[u] : in_ptr[u + 1]].tolist():
                tp = t_pre[pre_of[j]]
                if tp == tp:  # not NaN
                    pairs.append((j, t - tp))
        accumulate_causal(self.elig, pairs, self.a_plus, self.tau_plus)

        # (2) the fired units' post times are now current
        t_post[fired] = t

        # (3) PRE phase: anti-causal pairs against post times as of (2)
        pairs = []
        for u in fired.tolist():
            for j in out_idx[out_ptr[u] : out_ptr[u + 1]].tolist():
                tq = t_post[post_of[j]]
                if tq == tq:
                    pairs.append((j, tq - t))
        accumulate_anticausal(self.elig, pairs, self.a_minus, self.tau_minus)

        # (4) the fired units' pre times are now current
        t_pre[fired] = t

    #: the header's name for the same call; `observe` is what the harness and the tests read
    observe = update_eligibility

    def decay(self, dt_step=1.0):
        """One timestep of the distal-reward bridge, over the whole edge table."""
        decay_eligibility(self.elig, dt_step, self.tau_e)

    # ---- the reward gate -------------------------------------------------------------------
    def reward(self, d):
        """Gate the stored eligibility into the quantized bundle weights.

        Re-quantization at `frac_bits` is the whole application: a delta below half a step is
        LOST, which is why "effective 0 with attempted > 0" is a real reading and not a bug --
        the diagnostics report it as such rather than calling it pinning.

        SIGN-PRESERVING: the post-update weight is clamped at zero from whichever side it
        started on, so a weight may reach zero and never cross. `q_start` is the construction
        sign, not the previous tick's, so an edge that has reached zero cannot be walked across
        by a later update either."""
        np = self._np
        self.executed += 1
        nz = self.elig != 0.0
        self.attempted += int(np.count_nonzero(nz))
        deltas = reward_deltas(self.elig, float(d), self.lr)
        steps = np.rint(deltas * self.scale).astype(np.int64)
        if not np.any(steps):
            return self.diagnostics()
        before = self.qflat.copy()
        new = before + steps
        if self.sign_preserving:
            pos = self.q_start > 0
            neg = self.q_start < 0
            clamped = np.where(pos & (new < 0), 0, new)
            clamped = np.where(neg & (clamped > 0), 0, clamped)
            self.clamped += int(np.count_nonzero(clamped != new))
            new = clamped
        self.qflat[:] = new
        moved = new != before
        self.effective_count += int(np.count_nonzero(moved))
        self.effective_l1 += float(np.abs(new - before).sum()) / self.scale
        return self.diagnostics()

    # ---- reporting -------------------------------------------------------------------------
    def diagnostics(self):
        """The four lines of 2.5, separately. A zero is never read from one of them."""
        return {
            "executed": self.executed,
            "attempted": self.attempted,
            "effective_count": self.effective_count,
            "effective_l1": self.effective_l1,
        }

    def reset(self):
        """Zero the eligibility and the spike-time memory between trials. Weights are NOT
        touched -- what a trial resets is the state the protocol says it resets."""
        np = self._np
        self.elig[:] = 0.0
        self._t_pre[:] = np.nan
        self._t_post[:] = np.nan

    def weights(self):
        """The dequantized edge weights, aligned with `edge_pre` / `edge_post`."""
        return self.qflat.astype(self._np.float64) / self.scale

    def weight_bytes(self):
        """The canonical byte-form the determinism arm compares."""
        return self.qflat.tobytes()


def _csr(keys, n):
    """(ptr, idx) so that idx[ptr[u]:ptr[u+1]] are the edge positions whose key is unit u."""
    import numpy as np

    counts = np.bincount(keys, minlength=n) if keys.size else np.zeros(n, dtype=np.int64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    idx = np.argsort(keys, kind="stable").astype(np.int64)
    return ptr, idx


# =============================================================================================
# The offline self-check (run_selfchecks CORE) -- a 12-unit ring, no lake, no network
# =============================================================================================
def _ring(lr=LR, sign_preserving=True, n_units=12, frac_bits=16):
    from snn_data_lake.substrate.kernel_state import KernelState
    from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses

    pre = [u for u in range(n_units) for _ in (0, 1)]
    post = [(u + 1 + 4 * k) % n_units for u in range(n_units) for k in (0, 1)]
    weight = [(0.20 if u < 8 else -0.20) for u in range(n_units) for _ in (0, 1)]
    state = KernelState([("kp", n_units)])
    syn = Synapses(
        pre=pre,
        post=post,
        weight=weight,
        delay=[1] * len(pre),
        n_units=n_units,
        p_parts=1,
        frac_bits=frac_bits,
        max_delay=1,
    )
    kt = KernelTick(state, syn, decay=0.9, v_reset=0.0, refrac_ticks=0)
    return KernelPlasticity(kt, lr=lr, sign_preserving=sign_preserving)


def _drive(plast, rewards, ticks=12, n_units=12):
    import numpy as np

    for t in range(1, ticks + 1):
        fired = np.asarray([u for u in range(n_units) if (u + t) % 3 == 0], dtype=np.int64)
        plast.update_eligibility(fired, float(t))
        plast.decay()
        if t in rewards:
            plast.reward(rewards[t])
    return plast


def main():
    import hashlib

    import numpy as np

    print("=" * 78)
    print("kernel plasticity -- the ONE R-STDP rule over a quantized bundle table")
    print("=" * 78)

    # 1. the table is actually written, and the tick sees it
    p = _ring()
    before = p.qflat.copy()
    _drive(p, {4: +1.0, 8: -1.0})
    key = sorted(p.syn.bundles)[0]
    is_view = p.syn.bundles[key][1].base is not None
    moved = not np.array_equal(before, p.qflat)
    d = p.diagnostics()
    print(
        f"1 writes the table    : moved = {moved}, bundle qweight is a view = {is_view}; "
        f"executed {d['executed']} attempted {d['attempted']} effective {d['effective_count']} "
        f"L1 {d['effective_l1']:.6f}"
    )

    # 2. sign preservation, and the RED that removes the clamp
    hard = {2: +1.0, 4: +1.0, 6: -1.0, 8: -1.0, 10: +1.0, 12: -1.0}
    on = _ring(lr=40.0)
    s0 = np.sign(on.qflat.copy())
    _drive(on, hard)
    crossed_on = int(np.count_nonzero((s0 * np.sign(on.qflat)) < 0))
    off = _ring(lr=40.0, sign_preserving=False)
    s1 = np.sign(off.qflat.copy())
    _drive(off, hard)
    crossed_off = int(np.count_nonzero((s1 * np.sign(off.qflat)) < 0))
    print(
        f"2 sign preserved      : clamp ON crossed {crossed_on} (clamped {on.clamped} entries); "
        f"RED clamp OFF crossed {crossed_off}"
    )

    # 3. determinism, and a different reward sequence that must differ
    def sha(rewards):
        q = _ring()
        _drive(q, rewards)
        return hashlib.sha256(q.weight_bytes()).hexdigest()

    a, b = sha({3: +1.0, 6: -1.0, 9: +1.0}), sha({3: +1.0, 6: -1.0, 9: +1.0})
    c = sha({3: -1.0, 6: +1.0, 9: -1.0})
    print(f"3 determinism         : same sequence sha {a[:16]} == {b[:16]} -> {a == b}")
    print(f"  reward sensitivity  : flipped sequence sha {c[:16]} differs -> {a != c}")

    # 4. the pin, and the two zeros that are NOT the pin
    pinned = _drive(_ring(lr=0.0), {4: +1.0, 8: -1.0}).diagnostics()
    tiny = _drive(_ring(lr=1e-12), {4: +1.0}).diagnostics()
    print(
        f"4 the four lines      : lr=0 executed {pinned['executed']} attempted "
        f"{pinned['attempted']} effective {pinned['effective_count']}; lr=1e-12 attempted "
        f"{tiny['attempted']} effective {tiny['effective_count']} (below quantization, NOT pinning)"
    )

    # 5. the equivalence with the per-object form, on a fan-in tree
    worst, clamps = _equivalence()
    step = 1.0 / float(1 << 16)
    print(f"5 P2 equivalence      : max |w_kernel - w_neuron| = {worst:.3e} vs one step {step:.3e}")

    print("=" * 78)
    assert moved, "the learner must write the bundle table"
    assert is_view, "each bundle's qweight must be a VIEW, or the tick reads pre-learning weights"
    assert crossed_on == 0, f"{crossed_on} weights crossed zero under the clamp"
    assert on.clamped > 0, "the clamp must have done something -- else the invariant is vacuous"
    assert crossed_off > 0, "the sign red is UNREACHABLE: removing the clamp changed nothing"
    assert a == b, "same construction and reward sequence must give byte-equal weights"
    assert a != c, "a flipped reward sequence must give different weights"
    assert pinned["executed"] > 0 and pinned["attempted"] > 0
    assert pinned["effective_count"] == 0, "lr = 0 must move nothing"
    assert tiny["attempted"] > 0 and tiny["effective_count"] == 0
    assert clamps == 0, "the equivalence sequence must not clamp"
    assert worst <= step, f"P2 equivalence broken: {worst:.3e} > {step:.3e}"
    print(
        "kernel_plasticity self-check OK: the bundle table is written through a view, sign is "
        "preserved with the red reachable, weights are byte-deterministic and reward-sensitive, "
        "the four diagnostic lines separate a pin from a below-quantization update, and the "
        "kernel form agrees with RSTDPNeuron within one quantization step"
    )
    return 0


def _equivalence(n_post=4, n_syn=3, frac_bits=16):
    """One spike/reward sequence through both forms -- the P2 verification, in-module so the bare
    self-check carries it too (the pytest file runs the same comparison)."""
    import numpy as np

    from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron
    from snn_data_lake.substrate.kernel_state import KernelState
    from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses

    pre0 = n_post
    n_units = n_post + n_post * n_syn
    pre, post, w = [], [], []
    for q in range(n_post):
        for i in range(n_syn):
            pre.append(pre0 + q * n_syn + i)
            post.append(q)
            w.append(0.40 + 0.03 * (q * n_syn + i))
    state = KernelState([("equiv", n_units)])
    syn = Synapses(
        pre=pre,
        post=post,
        weight=w,
        delay=[1] * len(pre),
        n_units=n_units,
        p_parts=1,
        frac_bits=frac_bits,
        max_delay=1,
    )
    kt = KernelTick(state, syn, decay=0.9, v_reset=0.0, refrac_ticks=0)
    plast = KernelPlasticity(kt)
    scale = float(1 << frac_bits)
    start = {
        (int(a), int(b)): float(q) / scale
        for a, b, q in zip(plast.edge_pre, plast.edge_post, plast.qflat, strict=True)
    }
    script = [
        (1, [pre0 + 0, pre0 + 1, pre0 + 5]),
        (2, [0, 1]),
        (3, [pre0 + 3, pre0 + 7]),
        (4, [1, 2]),
        (5, [pre0 + 2, pre0 + 9, pre0 + 11]),
        (6, [0, 3]),
        (7, [pre0 + 4, pre0 + 6, pre0 + 8, pre0 + 10]),
        (8, [2, 3]),
    ]
    rewards = {4: +1.0, 8: -1.0}
    for t, fired in script:
        plast.update_eligibility(np.asarray(sorted(fired), dtype=np.int64), float(t))
        plast.decay()
        if t in rewards:
            plast.reward(rewards[t])

    cells = [RSTDPNeuron(n_syn, seed=q) for q in range(n_post)]
    for q, cell in enumerate(cells):
        for i in range(n_syn):
            cell.w[i] = start[(pre0 + q * n_syn + i, q)]
    for t, fired in script:
        for u in sorted(fired):
            if u < n_post:
                cells[u].post_spike(float(t))
        for u in sorted(fired):
            if u >= n_post:
                q, i = divmod(u - pre0, n_syn)
                cells[q].pre_spike(i, float(t))
        for cell in cells:
            cell.step()
        if t in rewards:
            for cell in cells:
                cell.reward(rewards[t])

    worst = 0.0
    for j in range(plast.n_edges):
        a, b = int(plast.edge_pre[j]), int(plast.edge_post[j])
        q, i = b, (a - pre0) - b * n_syn
        worst = max(worst, abs(float(plast.qflat[j]) / scale - cells[q].w[i]))
    return worst, plast.clamped


if __name__ == "__main__":
    sys.exit(main())
