"""
Benna-Fusi complex synapse -- multi-timescale beaker cascade for graded consolidation.

Self-improvement ladder L11 / increment SI-9. The SPIKING-NATIVE replacement for EWC's
Fisher-information penalty: instead of a rate-level quadratic regulariser that pins whole
weights to an old task, each synapse becomes a CHAIN of coupled scalar "beakers" spanning
a range of timescales, and consolidation falls out of the physics of the chain.

WHAT THIS IS (Benna & Fusi, Nat. Neurosci. 2016 "Computational principles of synaptic
memory consolidation"). Every synaptic weight is a cascade of N variables u_1..u_N with
geometrically GROWING capacities C_k and (adjacent-only) DIFFUSION coupling g_k, so the
levels equilibrate on geometrically SEPARATED timescales (fast -> slow):

    fast beaker u_1 receives the plasticity update each tick (the readout / effective weight);
    value then SEEPS u_1 -> u_2 -> ... -> u_N by linear diffusion:
        C_k * du_k/dt = g_{k-1}(u_{k-1} - u_k) + g_k(u_{k+1} - u_k)      (closed boundary)
    discretised (exact float Euler):  u_k += dt/C_k * (net diffusive flow into k)

The readout tracks the FAST beaker u_1, but at rest u_1 is pulled back toward the large,
slow deep beakers. A memory that had time to seep into the deep beakers is therefore
ANCHORED: later conflicting plasticity flips the free fast beaker (so the new thing is
learned) yet the deep store still biases the resting readout, so the old memory survives
the overwrite -- graded, not catastrophic, forgetting.

HOW THIS DIFFERS from lifelong_consolidation.py (do not confuse them). That module keeps
old-task accuracy alive by interleaved REPLAY over prototype UNITS (a system-level rehearsal
loop). This is a per-SYNAPSE metaplasticity primitive: no replay, no rehearsal, no stored
exemplars -- the resistance is intrinsic to the coupled-beaker dynamics of a single weight.

ATTACHMENT (spiking). The plasticity events are NOT a hand rate rule: they are produced by
the project's SPIKING reward-modulated STDP (basal_ganglia.r_stdp.RSTDPNeuron). Per training
tick the direction of each synapse's update is set by SPIKE TIMING (causal pre-before-post ->
potentiate, anti-causal post-before-pre -> depress) and gated by a reward; the resulting
reward-gated deltas are DEPOSITED into the fast beaker of each synapse's cascade. The cascade
is the metaplasticity layer wrapped around the spiking R-STDP weight -- it is never laundered
onto a rate MlpQ/JEPA head.

HONEST framing (required). This is spiking-ADJACENT METAPLASTICITY: the cascade itself is a
LINEAR DIFFUSION process over scalar auxiliary variables, NOT a neuron model and NOT itself
spiking. LIMIT: at a FEW (here 2) contexts this cannot be demonstrated as forgetting-IMMUNITY
and the horizon is far too short to measure a power law. Ship it as a MECHANISM DEMO. We
report retention at the MEASURED horizon ONLY. We do NOT claim measured power-law forgetting
and we do NOT claim lifelong / immortal memory -- a two-context run cannot establish either.
The bounded (clipped) synaptic efficacy is what makes a plain single synapse forget
catastrophically; the cascade escapes that by keeping the memory in the slow deep beakers.
One more honest caveat: the R-STDP source here is driven by FIXED causal / anti-causal spike
pairings (a deterministic ~+/-0.19 teaching signal), so the spike-timing rule is genuinely
EXERCISED (pre/post spike, exp window, eligibility, reward gate) but is NOT emergent learning
from noisy spike trains -- consistent with the "spiking-ADJACENT" framing. The claim under test
is the CASCADE's consolidation, not the realism of the upstream spike code.

Controls exposed for an un-riggable comparison:
  * collapse to a SINGLE scalar  (n_levels=1)  -- the plasticity-matched fast synapse.
  * ZERO the inter-beaker diffusion (g0=0.0)   -- the cascade-OFF control (same fast lr).

Reuses (does NOT reimplement): basal_ganglia.r_stdp.RSTDPNeuron (the spiking reward-gated
weight update the cascade attaches to) and substrate._mathutils.mean. The diffusion is exact,
deterministic float arithmetic; per-instance random.Random(seed); zlib.crc32 for sub-seeds
(never hash()); no wall-clock, no global RNG. ASCII-only output.

Refs: Benna & Fusi, Nat. Neurosci. 2016; Fusi, Drew & Abbott, Neuron 2005 (cascade model).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.benna_fusi
"""

import random
import zlib

from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron
from snn_data_lake.substrate._mathutils import mean

# ---- experiment constants (deterministic; sequential-interference protocol A -> B -> A) ----
N_SYN = 24  # synapses carrying the mapping
N_LEVELS = 6  # beakers per cascade (fast u_1 .. slow u_6)
C_RATIO = 2.0  # geometric capacity growth  C_k = C_0 * C_RATIO^k
G0 = 0.10  # first (fastest) inter-beaker conductance
G_RATIO = 0.5  # geometric conductance shrink  g_k = G0 * G_RATIO^k (deep tubes narrow)
W_MAX = 1.0  # bounded synaptic efficacy (the bound is what makes a scalar forget)
LR = 0.20  # R-STDP learning rate -> per-tick deposit ~ LR * exp(-1/tau) ~ 0.19
T_A = 300  # A-training ticks (long: let A seep into / CONSOLIDATE the deep beakers)
CRIT = 0.90  # criterion: sign-agreement fraction with the target mapping
T_B_MAX = 150  # cap on B-training ticks
RELAX = 200  # rest/consolidation steps before the A re-test (labile B fades)


def _clip(x, lo, hi):
    if x > hi:
        return hi
    if x < lo:
        return lo
    return x


class BennaFusiSynapse:
    """One weight as a Benna-Fusi cascade of N coupled beakers on separated timescales.

    deposit(delta) injects a plasticity event into the FAST beaker u_1; diffuse(steps) runs
    the linear inter-beaker diffusion (value seeps fast -> slow); .weight reads the fast beaker
    u_1 (the effective synaptic efficacy). n_levels=1 collapses to a single scalar (control a);
    g0=0.0 zeroes the coupling so the cascade is OFF (control b), leaving the fast beaker alone.
    """

    def __init__(
        self, n_levels=N_LEVELS, c_ratio=C_RATIO, g0=G0, g_ratio=G_RATIO, dt=1.0, w_max=W_MAX
    ):
        self.n = n_levels
        self.dt = dt
        self.w_max = w_max
        self.c = [1.0 * (c_ratio**k) for k in range(n_levels)]  # capacities (grow)
        self.g = [g0 * (g_ratio**k) for k in range(n_levels - 1)]  # conductances (shrink)
        self.u = [0.0] * n_levels  # beaker levels

    def deposit(self, delta):
        """A plasticity event enters the fast beaker u_1 (bounded synaptic efficacy)."""
        self.u[0] = _clip(self.u[0] + delta, -self.w_max, self.w_max)

    def diffuse(self, steps=1):
        """Advance the linear inter-beaker diffusion `steps` times (exact float Euler).

        Closed boundary: total capacity-weighted charge is (up to clipping) conserved, so at
        rest the fast beaker relaxes toward the capacity-weighted mean, which the big deep
        beakers dominate -- that is the anchor a consolidated memory leaves behind.
        """
        u, c, g, dt, n = self.u, self.c, self.g, self.dt, self.n
        wmax = self.w_max
        for _ in range(steps):
            flow = [0.0] * n
            for k in range(n - 1):
                f = g[k] * (u[k] - u[k + 1])  # diffusive flow from beaker k to k+1
                flow[k] -= f
                flow[k + 1] += f
            for k in range(n):
                u[k] = _clip(u[k] + dt * flow[k] / c[k], -wmax, wmax)

    @property
    def weight(self):
        """Effective synaptic efficacy = the fast beaker u_1."""
        return self.u[0]

    def deep_mean(self):
        """Mean of the slow (deep) beakers u_2..u_N -- the consolidated store. u_1 for N=1."""
        return mean(self.u[1:]) if self.n > 1 else self.u[0]


class BennaFusiLearner:
    """A spiking R-STDP readout neuron whose per-synapse weights are Benna-Fusi cascades.

    Each training tick, spike TIMING sets the sign of every synapse's update (causal pairing
    -> potentiate, anti-causal -> depress) and a reward gates it; the reward-gated deltas from
    RSTDPNeuron are DEPOSITED into the fast beakers, then the cascades diffuse one step. mode:
    'cascade' (full Benna-Fusi), 'single' (n_levels=1 scalar control), 'diffusion_off' (g0=0).
    """

    def __init__(self, n_syn=N_SYN, mode="cascade", seed=0):
        self.n = n_syn
        self.mode = mode
        rs_seed = zlib.crc32(b"rstdp") ^ seed
        self.rstdp = RSTDPNeuron(n_syn, lr=LR, seed=rs_seed)
        if mode == "single":
            self.syn = [BennaFusiSynapse(n_levels=1) for _ in range(n_syn)]
        elif mode == "diffusion_off":
            self.syn = [BennaFusiSynapse(g0=0.0) for _ in range(n_syn)]
        else:
            self.syn = [BennaFusiSynapse() for _ in range(n_syn)]

    def _rstdp_deltas(self, targets):
        """Spike-timing -> reward-gated per-synapse deltas (the spiking attachment point).

        target +1 -> causal pre(t=0)-before-post(t=1) -> LTP eligibility -> +delta.
        target -1 -> anti-causal post(t=1)-before-pre(t=2) -> LTD eligibility -> -delta.
        A single positive reward gates the per-tick eligibility into signed weight deltas.
        """
        r = self.rstdp
        r.e = [0.0] * r.n  # fresh per-tick eligibility (this tick only)
        r.t_pre = [None] * r.n
        r.t_post = None
        for i, t in enumerate(targets):
            if t > 0:  # causal pairing: pre before post
                r.pre_spike(i, 0.0)
        r.post_spike(1.0)
        for i, t in enumerate(targets):
            if t < 0:  # anti-causal pairing: pre after post
                r.pre_spike(i, 2.0)
        return r.reward(1.0)  # w += lr * e; returns the per-synapse deltas

    def train_tick(self, targets):
        """One tick: spiking deltas -> deposit into fast beakers -> one diffusion step."""
        deltas = self._rstdp_deltas(targets)
        for i, s in enumerate(self.syn):
            s.deposit(deltas[i])
            s.diffuse(1)

    def relax(self, steps=RELAX):
        """Rest: diffuse only (no plasticity). The labile fast beaker settles toward deep."""
        for s in self.syn:
            s.diffuse(steps)

    def weights(self):
        return [s.weight for s in self.syn]

    def accuracy(self, targets):
        """Fraction of synapses whose effective-weight SIGN agrees with the mapping (chance 0.5)."""
        w = self.weights()
        ok = sum(1 for i in range(self.n) if (w[i] > 0.0) == (targets[i] > 0.0))
        return ok / self.n


def _make_tasks(n_syn, seed):
    """Mapping A = a random +/-1 sign pattern; B = its maximal conflict (b = -a)."""
    rng = random.Random(zlib.crc32(b"tasks") ^ seed)
    a = [1 if rng.random() < 0.5 else -1 for _ in range(n_syn)]
    b = [-x for x in a]
    return a, b


def _run(mode, seed=0):
    """Sequential-interference run: train A to consolidation, train B to criterion, re-test A.

    Returns (a_after_A, b_criterion_acc, b_ticks, a_retention_after_B).
    """
    a, b = _make_tasks(N_SYN, seed)
    learner = BennaFusiLearner(mode=mode, seed=seed)
    for _ in range(T_A):
        learner.train_tick(a)
    a_after_a = learner.accuracy(a)
    b_ticks = 0
    while learner.accuracy(b) < CRIT and b_ticks < T_B_MAX:
        learner.train_tick(b)
        b_ticks += 1
    b_crit = learner.accuracy(b)  # measured right after B training (fast beaker holds B)
    learner.relax(RELAX)  # rest: labile B fades, consolidated A re-emerges
    a_ret = learner.accuracy(a)
    return a_after_a, b_crit, b_ticks, a_ret


def main():
    print("=" * 74)
    print("BENNA-FUSI complex synapse -- multi-timescale beaker cascade (L11 / SI-9)")
    print("sequential interference: train A -> train conflicting B -> re-test A")
    print("=" * 74)

    a_a_bf, b_bf, bt_bf, ret_bf = _run("cascade", seed=0)
    a_a_1, b_1, bt_1, ret_1 = _run("single", seed=0)
    a_a_0, b_0, bt_0, ret_0 = _run("diffusion_off", seed=0)

    print("config              A-acc(after A)   B-crit(after B)   A-retention(after B+rest)")
    print(
        f"Benna-Fusi cascade      {a_a_bf:5.3f}            {b_bf:5.3f}  ({bt_bf:3d} B-ticks)     {ret_bf:5.3f}"
    )
    print(
        f"single scalar (N=1)     {a_a_1:5.3f}            {b_1:5.3f}  ({bt_1:3d} B-ticks)     {ret_1:5.3f}"
    )
    print(
        f"diffusion off (g=0)     {a_a_0:5.3f}            {b_0:5.3f}  ({bt_0:3d} B-ticks)     {ret_0:5.3f}"
    )
    print("-" * 74)
    print(
        f"chance retention = 0.500 ; B reached criterion (>= {CRIT:.2f}) on the cascade = {b_bf >= CRIT}"
    )

    # ---- determinism: same seed -> identical outcome ----
    same = _run("cascade", seed=7) == _run("cascade", seed=7)
    print(f"determinism (two seed=7 cascade runs identical) = {same}")
    print("=" * 74)

    # ---- self-check: deterministic asserts, ASCII, un-riggable ----
    # (1) RETENTION: cascade retains A markedly better than the plasticity-matched scalar,
    #     WHILE B still reached its criterion (the free fast beaker learned B).
    assert b_bf >= CRIT, f"B must reach criterion on the cascade (fast beaker learns B): {b_bf:.3f}"
    assert ret_bf >= 0.80, f"cascade must retain A at the measured horizon: {ret_bf:.3f}"
    assert ret_bf >= ret_1 + 0.30, (
        f"cascade must retain A MARKEDLY better than the single scalar: {ret_bf:.3f} vs {ret_1:.3f}"
    )
    # (2) CONTROL a -- plasticity-matched single scalar forgets A catastrophically (~ chance).
    assert ret_1 <= 0.50, f"single scalar must forget A catastrophically (<= chance): {ret_1:.3f}"
    # (3) CONTROL b -- zero diffusion collapses retention BACK to the scalar baseline: it is the
    #     CASCADE (multi-timescale coupling), not merely a low fast lr, that holds A.
    assert ret_0 <= ret_1 + 0.05, (
        f"g=0 must collapse to the single-scalar baseline (cascade, not low lr): {ret_0:.3f} vs {ret_1:.3f}"
    )
    assert ret_bf >= ret_0 + 0.30, (
        f"cascade must beat its own g=0 control (the coupling is the mechanism): {ret_bf:.3f} vs {ret_0:.3f}"
    )
    assert same, "same seed must give an identical outcome (determinism)"

    # (4) HONEST LIMIT (printed, not overclaimed).
    print(
        f"self-check OK: cascade A-retention {ret_bf:.3f} vs single-scalar {ret_1:.3f} vs g=0 "
        f"{ret_0:.3f} (both controls at/below chance), B reached criterion {b_bf:.3f}."
    )
    print(
        "HONEST LIMIT: spiking-ADJACENT metaplasticity (linear diffusion, not a neuron model). "
        "MECHANISM DEMO at the MEASURED 2-context horizon ONLY -- this does NOT show "
        "forgetting-immunity and makes NO power-law / lifelong-memory claim (a two-context run "
        "cannot measure a power law)."
    )


if __name__ == "__main__":
    main()
