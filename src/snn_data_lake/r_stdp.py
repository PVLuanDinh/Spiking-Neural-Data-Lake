"""
Reward-modulated STDP (R-STDP) with a decaying eligibility trace -- the distal-reward rule.

A new member of the plasticity family (additive, opt-in; changes no existing behaviour). This
is the classic Izhikevich (2007) solution to the DISTAL-REWARD problem: spike-timing
coincidences leave a slowly-decaying ELIGIBILITY TRACE at each synapse, so a reward arriving
many timesteps AFTER the causal pre-before-post pairing can still reinforce the CORRECT
synapse -- even though many other spikes happened in between. Mechanism (BrainCog learning
rule #5, R-STDP):

    STDP(dt) is added to the eligibility e, NOT to w:
        dt = t_post - t_pre;  LTP if dt > 0: e += A_plus  * exp(-dt / tau_plus)
                              LTD if dt < 0: e += -A_minus * exp( dt / tau_minus)
    e decays every step:      e *= exp(-dt_step / tau_e)   (tau_e is LONG: the reward bridge)
    delayed reward d gates:   w += lr * d * e              (per synapse; sign of d flips it)

HOW THIS DIFFERS from valence_stdp.py (do not confuse the two): ValenceLearner is a reward-
PREDICTION-ERROR rule over FEATURE PRESENCE with IMMEDIATE reward (w += lr*(reward-predicted)
*fired_now) -- it is NOT spike-timing-dependent and bridges NO temporal delay. R-STDP here
(a) keys on pre/post spike TIMING (the STDP window), and (b) bridges a DELAY between the
pairing and a LATER raw reward d via the decaying eligibility trace. Different on both axes.

Honesty: this improves temporal CREDIT ASSIGNMENT (distal reward) via a synaptic eligibility
trace. It is not cognition/AGI.
ponytail: single-layer, few-synapse rate-timing caricature on synthetic spike trains -- no
deep multi-step credit assignment, no hidden layers. Additive/opt-in new plasticity member.
Refs: Zeng et al., BrainCog, Patterns 2023 (rule #5, R-STDP); Izhikevich, "Solving the distal
reward problem through STDP and dopamine", Cereb. Cortex 2007.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.r_stdp
"""
import math
import random

LR = 0.1
A_PLUS = 1.0
A_MINUS = 1.0
TAU_PLUS = 20.0
TAU_MINUS = 20.0
TAU_E = 1000.0        # long eligibility time constant -- the distal-reward bridge


class RSTDPNeuron:
    """One output neuron fed by n_syn input synapses; each synapse has a weight w and an
    eligibility trace e. STDP (spike timing) writes to e; a delayed reward gates e into w."""

    def __init__(self, n_syn, lr=LR, a_plus=A_PLUS, a_minus=A_MINUS, tau_plus=TAU_PLUS,
                 tau_minus=TAU_MINUS, tau_e=TAU_E, w0=0.5, seed=0):
        rng = random.Random(seed)
        self.n = n_syn
        self.w = [w0 + rng.uniform(-0.05, 0.05) for _ in range(n_syn)]
        self.e = [0.0] * n_syn
        self.t_pre = [None] * n_syn          # last pre-spike time per synapse
        self.t_post = None                   # last post-spike time of the neuron
        self.lr, self.a_plus, self.a_minus = lr, a_plus, a_minus
        self.tau_plus, self.tau_minus, self.tau_e = tau_plus, tau_minus, tau_e

    def pre_spike(self, i, t):
        """Input synapse i fires at time t. Pair with the neuron's most recent POST spike:
        post-before-pre (dt < 0) is anti-causal -> LTD onto this synapse's eligibility."""
        self.t_pre[i] = t
        if self.t_post is not None:
            dt = self.t_post - t             # <= 0 here: post came before this pre
            if dt < 0:
                self.e[i] += -self.a_minus * math.exp(dt / self.tau_minus)

    def post_spike(self, t):
        """The output neuron fires at time t. Pair with each synapse's most recent PRE spike:
        pre-before-post (dt > 0) is causal -> LTP onto that synapse's eligibility."""
        self.t_post = t
        for i in range(self.n):
            tp = self.t_pre[i]
            if tp is not None:
                dt = t - tp                  # > 0 here: pre came before this post
                if dt > 0:
                    self.e[i] += self.a_plus * math.exp(-dt / self.tau_plus)

    def step(self, dt_step=1.0):
        """Advance one timestep: the eligibility trace decays slowly (over tau_e). No reward
        yet -- this is the interval in which OTHER spikes may occur before the reward lands."""
        decay = math.exp(-dt_step / self.tau_e)
        for i in range(self.n):
            self.e[i] *= decay

    def reward(self, d):
        """A (possibly long-delayed) reward/dopamine signal d arrives: gate the stored
        eligibility into the weights. w += lr * d * e for every synapse; d>0 reinforces
        positive-eligibility (causally paired) synapses, d<0 flips the direction. Returns the
        per-synapse weight deltas."""
        deltas = [self.lr * d * self.e[i] for i in range(self.n)]
        for i in range(self.n):
            self.w[i] += deltas[i]
        return deltas


def _run(seed):
    """A fixed multi-synapse protocol used to check determinism (same seed -> same weights)."""
    n = RSTDPNeuron(3, seed=seed)
    n.pre_spike(0, 5.0)
    n.pre_spike(1, 6.0)
    n.post_spike(10.0)
    for _ in range(20):
        n.step()
    n.pre_spike(2, 30.0)
    n.reward(+1.0)
    return n.w


def main():
    print("=" * 70)
    print("R-STDP -- reward-modulated STDP with a decaying eligibility trace")
    print("(distal-reward rule: causal spike pairing reinforced by a LATER reward)")
    print("=" * 70)

    # 1. CAUSAL POTENTIATION: pre-before-post (dt>0) + positive delayed reward -> w up.
    n1 = RSTDPNeuron(1, seed=0)
    w0 = n1.w[0]
    n1.pre_spike(0, 5.0)
    n1.post_spike(10.0)
    for _ in range(5):
        n1.step()
    n1.reward(+1.0)
    causal = n1.w[0] - w0
    print(f"1 causal potentiation : pre-before-post, reward +1 -> dw = {causal:+.4f}")

    # 2. ANTI-CAUSAL DEPRESSION: post-before-pre (dt<0) + positive reward -> w down.
    n2 = RSTDPNeuron(1, seed=0)
    w0 = n2.w[0]
    n2.post_spike(5.0)
    n2.pre_spike(0, 10.0)
    for _ in range(5):
        n2.step()
    n2.reward(+1.0)
    anti = n2.w[0] - w0
    print(f"2 anti-causal depress : post-before-pre, reward +1 -> dw = {anti:+.4f}")

    # 3. DISTAL REWARD (headline): a causal pairing, then a LONG delay, then a distractor
    #    fires near reward time. The causally-paired synapse is still reinforced far more.
    nd = RSTDPNeuron(2, seed=0)
    before = list(nd.w)
    nd.pre_spike(0, 5.0)        # synapse 0: causal pre-before-post pairing
    nd.post_spike(10.0)
    for _ in range(100):        # many timesteps pass; other activity happens
        nd.step()
    nd.pre_spike(1, 112.0)      # synapse 1 (distractor): active near reward, NOT causally paired
    nd.reward(+1.0)
    paired_gain = nd.w[0] - before[0]
    distractor_gain = nd.w[1] - before[1]
    print(f"3 distal reward       : delay=100 steps  paired dw = {paired_gain:+.4f}  "
          f"distractor dw = {distractor_gain:+.4f}")

    # 4. REWARD-SIGN FLIP: the same causal pairing but a NEGATIVE reward -> w down.
    n4 = RSTDPNeuron(1, seed=0)
    w0 = n4.w[0]
    n4.pre_spike(0, 5.0)
    n4.post_spike(10.0)
    for _ in range(5):
        n4.step()
    n4.reward(-1.0)
    neg = n4.w[0] - w0
    print(f"4 reward-sign flip    : causal pairing, reward -1 -> dw = {neg:+.4f}")

    # 5. DETERMINISM: same seed -> identical final weights across two fresh instances.
    w_a, w_b = _run(7), _run(7)
    print(f"5 determinism         : two seed=7 runs identical = {w_a == w_b}")
    print("=" * 70)

    # ---- self-check (deterministic asserts, zero-dep, ASCII) ----
    assert causal > 0.02, f"causal pre-before-post + reward must potentiate ({causal:+.4f})"
    assert anti < -0.02, f"anti-causal post-before-pre + reward must depress ({anti:+.4f})"
    assert paired_gain > distractor_gain + 0.03, \
        f"distal: causally-paired synapse must beat the distractor ({paired_gain:+.4f} vs {distractor_gain:+.4f})"
    assert paired_gain > 0.02, f"the paired synapse must still be reinforced after the delay ({paired_gain:+.4f})"
    assert neg < 0, f"negative reward must flip the causal update downward ({neg:+.4f})"
    assert w_a == w_b, "same seed must give identical final weights (determinism)"
    print(f"self-check OK: causal {causal:+.4f}, anti {anti:+.4f}, distal paired {paired_gain:+.4f} "
          f"beats distractor {distractor_gain:+.4f}, neg-reward {neg:+.4f}; eligibility trace bridges "
          f"the reward delay; additive; deterministic")


if __name__ == "__main__":
    main()
