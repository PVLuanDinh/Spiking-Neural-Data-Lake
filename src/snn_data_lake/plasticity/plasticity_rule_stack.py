"""
PlasticityRuleStack -- a SUPERSET of HomeostaticPlasticityStack that ADDS metaplastic adaptive-LR +
anti-forgetting consolidation around the live R-STDP learning synapses, as a DROP-IN replacement.

WHY THIS EXISTS (a deeper merge, from the same study of the brain). HomeostaticPlasticityStack already
wraps the one real online learning path (reward-modulated spike-timing plasticity: basal_ganglia.r_stdp,
driven by cortex.interaction_learning) in a fixed ordered sandwich -- a short-term dynamic gain BEFORE
the R-STDP deposit, then homeostatic synaptic scaling + intrinsic plasticity AFTER, so the synapses stay
bounded and rate-homeostatic. That stops runaway, but it does NOTHING about two other failures real
cortex solves: (a) the learning RATE is fixed regardless of how strongly the neuron has recently been
driven, and (b) a fresh burst of teaching OVERWRITES what was learned before (catastrophic forgetting).
This stack composes three more of the dead standalone plasticity rules into exactly those two missing
faculties, WITHOUT touching the base sandwich (it subclasses it):

    (0) metaplastic BCM learning-rate gate      [PRE]   -- bcm_plasticity.BCMNeuron's sliding threshold
        theta (<y^2>) is the metaplastic variable: recent high activity RAISES theta, which LOWERS the
        effective learning rate (Abraham-Bear metaplasticity), folded on top of the base short-term gain.
        --> [ the base PRE gain + the live R-STDP weight update happen ] <--
    (1) base homeostatic sandwich               [POST]  -- HomeostaticPlasticityStack.stabilize verbatim
        (short-term gain already applied PRE; homeostatic scaling + intrinsic plasticity here) -> bounded.
    (2) inhibitory-STDP E/I balance term         [POST]  -- inhibitory_stdp.InhibitorySTDP (Vogels 2011)
        develops co-tuned inhibition that GROWS while a neuron runs over-excited; applied as a GENTLE
        multiplicative trim (<= a few %, never a boost, sign-preserving) so it can only reduce runaway.
    (3) Benna-Fusi multi-timescale consolidation [POST]  -- benna_fusi.BennaFusiSynapse: each live synapse
        gets a coupled-beaker cascade; the per-tick weight CHANGE is deposited into the fast beaker and
        seeps to the slow deep beakers. The effective (returned) weight is the fast beaker, which at rest
        is ANCHORED by the consolidated deep store -- a conflicting new-pattern burst flips the fast beaker
        (the new thing is learned) yet the deep store biases the weight back toward the old pattern, so the
        old memory survives the overwrite far better than the un-consolidated base (graded forgetting).

DROP-IN. Same public contract as the base stack -- .pre_gain(neuron_id) -> scalar and
.stabilize(neuron_id, weights, activity) -> weights (plus .learning_step for the full-sandwich self-check)
-- so anywhere a HomeostaticPlasticityStack is attached (web_server's live intent learner;
cortex.live_brain's SpikingInteractionLearner), a PlasticityRuleStack swaps straight in. With its three
extra faculties DISABLED (ei / metaplastic / consolidation = False) it is BYTE-IDENTICAL to the base stack.

REUSE, not reinvention. Every added faculty is a THIN ADAPTER over the real rule module (import + delegate,
no rewrite): bcm_plasticity.BCMNeuron (theta as the metaplastic threshold), inhibitory_stdp.InhibitorySTDP
(the Vogels E/I controller), benna_fusi.BennaFusiSynapse (the coupled-beaker cascade). The base sandwich is
inherited unchanged.

HONESTY / SAFETY. stdlib + numpy/torch allowed project-wide (all reused rules happen to be stdlib).
OPT-IN: the learner takes the stack as an injected duck-typed object; with
no stack (the default) the R-STDP path is BYTE-IDENTICAL to today, and attaching this one changes nothing
until it is enabled. It stabilizes/consolidates PLASTICITY only and adds NO actuator route (the intent-
recognition learning path only; the motor path is untouched). Deterministic: per-instance random.Random /
zlib.crc32 sub-seeds only (BCM init), never builtin hash(); no module-level seeding; every other rule here
is closed-form. ARM-ADJACENT: the Benna-Fusi deep anchor shifts the effective-weight argmax, which can bias
which intent (hence which downstream action) the router selects -- flagged for the main agent's E-STOP
re-verify. ASCII-only output.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.plasticity_rule_stack
"""

import zlib

from snn_data_lake.plasticity.bcm_plasticity import BCMNeuron
from snn_data_lake.plasticity.benna_fusi import BennaFusiSynapse
from snn_data_lake.plasticity.homeostatic_plasticity_stack import HomeostaticPlasticityStack
from snn_data_lake.plasticity.inhibitory_stdp import InhibitorySTDP


# --------------------------------------------------------------------------------------------------
# (0) PRE -- metaplastic BCM learning-rate gate. REUSES bcm_plasticity.BCMNeuron.
# --------------------------------------------------------------------------------------------------
class _BCMMetaGate:
    """A per-neuron metaplastic brake on the effective learning rate, built on the BCM sliding threshold.

    REUSES bcm_plasticity.BCMNeuron: its threshold theta tracks a running <y^2> of the neuron's recent
    drive. That is the metaplastic variable of Abraham & Bear (2008): a neuron that has recently been
    strongly driven RAISES theta, which should make further potentiation HARDER (a lower effective LR);
    a quiet neuron keeps theta low and learns at full rate. `observe` slides theta from the just-seen
    activity (POST, once per learning step); `gate` returns the PRE multiplier -- monotonically DECREASING
    in theta and bounded in [floor, 1.0], so it can only slow, never accelerate or invert, the learning
    signal. Lazily sized to the synapse count on first sight; cold start (no history) -> gate 1.0."""

    name = "bcm_metaplastic_lr_gate"
    reused = "bcm_plasticity.BCMNeuron"

    def __init__(self, n_neurons, beta=1.5, floor=0.3, seed=0, lr=0.05):
        self._n = int(n_neurons)
        self._beta = float(beta)
        self._floor = float(floor)
        self._seed = int(seed)
        self._lr = float(lr)  # small BCM lr -> the STABLE non-trivial fixed point
        self._bcm = [None] * self._n  # one scalar BCMNeuron(1) per neuron, lazily built
        self._theta0 = 0.05  # BCMNeuron's initial theta (gate == 1.0 at rest)

    def _ensure(self, i):
        if self._bcm[i] is None:
            sub = zlib.crc32(b"bcm-meta") ^ (
                self._seed + i
            )  # deterministic sub-seed (never hash())
            self._bcm[i] = BCMNeuron(
                1, lr=self._lr, seed=sub
            )  # scalar unit: theta tracks the drive

    @staticmethod
    def _squash(drive):
        """Map any drive magnitude into BCM's STABLE monotonic input band [0, 0.5) so the threshold theta
        rises monotonically with drive across the whole live range (above ~0.5 raw drive the BCM response
        saturates/collapses -- the squash keeps the metaplastic variable well-behaved and bounded)."""
        d = abs(float(drive))
        return 0.5 * d / (1.0 + d)

    def observe(self, i, drive):
        """Slide neuron i's BCM threshold with the just-seen scalar drive (the neuron's membrane response
        magnitude, squashed into the stable band). theta = <y^2> is the metaplastic modification threshold:
        strong recent drive RAISES it. Delegates to the real BCM update rule (a single-input BCMNeuron)."""
        self._ensure(i)
        self._bcm[i].update([self._squash(drive)])  # REUSE the real BCM rule (slides theta)

    def theta(self, i):
        b = self._bcm[i]
        return self._theta0 if b is None else b.theta

    def gate(self, i):
        """PRE effective-LR multiplier in [floor, 1.0], decreasing in the neuron's BCM threshold theta."""
        b = self._bcm[i]
        if b is None:
            return 1.0
        g = (1.0 + self._beta * self._theta0) / (1.0 + self._beta * b.theta)
        return min(1.0, max(self._floor, g))


# --------------------------------------------------------------------------------------------------
# (2) POST -- inhibitory-STDP E/I balance term. REUSES inhibitory_stdp.InhibitorySTDP (Vogels 2011).
# --------------------------------------------------------------------------------------------------
class _EIBalance:
    """A per-neuron E/I controller that trims runaway excitation using inhibitory spike-timing plasticity.

    REUSES inhibitory_stdp.InhibitorySTDP: each neuron owns a Vogels-Sprekeler controller whose inhibitory
    conductance GROWS while its postsynaptic cell fires above the target rate rho0 and shrinks below it. We
    drive that controller with an excitatory proxy of the neuron's current membrane drive and read the
    developed mean inhibitory weight as an E/I imbalance signal, then apply it as a GENTLE multiplicative
    trim on the live weights: f = 1 - min(max_trim, k * inhibition), always in [1 - max_trim, 1.0]. It can
    only REDUCE (never boost) drive and preserves the relative weight pattern (multiplicative), so it can
    never cause runaway and complements -- rather than fights -- the base sandwich's rate homeostasis.
    Deterministic: the inhibitory input is a fixed rotating single-synapse pattern (no rng)."""

    name = "inhibitory_stdp_ei_balance"
    reused = "inhibitory_stdp.InhibitorySTDP"

    def __init__(self, n_neurons, n_syn=16, rho0=0.05, max_trim=0.05, k=0.5):
        self._ctrl = [InhibitorySTDP(n_syn=n_syn, rho0=rho0) for _ in range(int(n_neurons))]
        self._n_syn = int(n_syn)
        self._max_trim = float(max_trim)
        self._k = float(k)
        self._t = 0

    def apply(self, i, weights, drive):
        """Advance neuron i's iSTDP controller one step (driven by `drive`) and return (trimmed weights,
        developed inhibition). The trim is bounded, multiplicative, and only ever reduces the weights."""
        self._t += 1
        inh = [0] * self._n_syn
        inh[self._t % self._n_syn] = 1  # deterministic inhibitory input (no rng)
        exc = 0.15 + 0.02 * max(0.0, drive)  # excitatory proxy of the neuron's drive
        exc = min(1.0, max(0.0, exc))
        self._ctrl[i].step(exc, inh, learn=True)  # REUSE the real Vogels iSTDP update
        inhibition = self._ctrl[i].mean_w()
        f = 1.0 - min(self._max_trim, self._k * inhibition)
        return [w * f for w in weights], inhibition


# --------------------------------------------------------------------------------------------------
# (3) POST -- Benna-Fusi multi-timescale consolidation. REUSES benna_fusi.BennaFusiSynapse.
# --------------------------------------------------------------------------------------------------
class _BennaFusiConsolidation:
    """A per-synapse coupled-beaker cascade wrapped around each live weight for anti-forgetting.

    REUSES benna_fusi.BennaFusiSynapse: each of a neuron's synapses becomes a chain of beakers on
    geometrically separated timescales. Each learning step we deposit the weight CHANGE into the fast
    beaker and diffuse one step so it seeps toward the slow deep beakers; the effective (returned) weight
    is the fast beaker u_1, which at rest is pulled toward the consolidated deep store. A cascade is
    initialized at EQUILIBRIUM (all beakers = the current weight) so it perturbs nothing until real
    plasticity arrives. `relax` diffuses without depositing (a rest phase) -- the labile fast beaker
    settles toward the deep anchor, which is how a consolidated old memory re-emerges after a conflicting
    burst. Lazily sized on first sight. No rng; exact deterministic float diffusion."""

    name = "benna_fusi_consolidation"
    reused = "benna_fusi.BennaFusiSynapse"

    def __init__(self, n_neurons, w_max=4.0, n_levels=6):
        self._n = int(n_neurons)
        self._w_max = float(w_max)
        self._n_levels = int(n_levels)
        self._cas = [None] * self._n  # per-neuron list[BennaFusiSynapse]
        self._prev = [None] * self._n  # last returned effective weights (delta reference)

    def _ensure(self, i, weights):
        if self._cas[i] is None:
            cas = [BennaFusiSynapse(n_levels=self._n_levels, w_max=self._w_max) for _ in weights]
            for s, wi in zip(cas, weights):
                lvl = min(self._w_max, max(-self._w_max, float(wi)))
                s.u = [lvl] * self._n_levels  # start at equilibrium: flow == 0, perturbs nothing
            self._cas[i] = cas
            self._prev[i] = list(weights)

    def consolidate(self, i, weights):
        """Deposit this step's per-synapse weight change into the fast beakers, diffuse once, and return
        the deep-anchored effective weights (the fast beakers)."""
        self._ensure(i, weights)
        cas, prev = self._cas[i], self._prev[i]
        out = []
        for k, s in enumerate(cas):
            s.deposit(
                weights[k] - prev[k]
            )  # inject this step's plasticity event into the fast beaker
            s.diffuse(1)  # seep fast -> deep
            out.append(s.weight)  # effective weight = fast beaker (deep-anchored at rest)
        self._prev[i] = list(out)
        return out

    def relax(self, i, steps):
        """Rest: diffuse only (no deposit). The labile fast beaker settles toward the deep store; the
        consolidated old memory re-emerges. Returns the settled effective weights."""
        if self._cas[i] is None:
            return None
        cas = self._cas[i]
        for s in cas:
            s.diffuse(int(steps))
        out = [s.weight for s in cas]
        self._prev[i] = list(out)
        return out


# --------------------------------------------------------------------------------------------------
# The rule stack -- SUBCLASSES HomeostaticPlasticityStack and layers the three extra faculties on top.
# --------------------------------------------------------------------------------------------------
class PlasticityRuleStack(HomeostaticPlasticityStack):
    """A drop-in SUPERSET of HomeostaticPlasticityStack: same .pre_gain / .stabilize contract, plus a PRE
    metaplastic BCM learning-rate gate and POST inhibitory-STDP E/I balance + Benna-Fusi consolidation.

    Attach exactly like the base stack (pass as `stack=` to SpikingInteractionLearner, or assign
    `ixn.stack = PlasticityRuleStack(n_neurons=n)`). With ei=metaplastic=consolidation=False it is
    byte-identical to the base stack. Deterministic, no actuator route."""

    def __init__(
        self,
        n_neurons,
        *,
        ei=True,
        metaplastic=True,
        consolidation=True,
        seed=0,
        meta_beta=8.0,
        meta_floor=0.3,
        ei_n_syn=16,
        ei_rho0=0.05,
        ei_max_trim=0.05,
        ei_k=0.5,
        bf_w_max=4.0,
        bf_n_levels=6,
        **base_kwargs,
    ):
        super().__init__(n_neurons, seed=seed, **base_kwargs)
        self._meta = (
            _BCMMetaGate(self.n_neurons, beta=meta_beta, floor=meta_floor, seed=seed)
            if metaplastic
            else None
        )
        self._ei = (
            _EIBalance(self.n_neurons, n_syn=ei_n_syn, rho0=ei_rho0, max_trim=ei_max_trim, k=ei_k)
            if ei
            else None
        )
        self._cons = (
            _BennaFusiConsolidation(self.n_neurons, w_max=bf_w_max, n_levels=bf_n_levels)
            if consolidation
            else None
        )
        # the full order this superset touches its rules each step (base sandwich sits in the middle)
        self.rule_order = tuple(
            [a.name for a in (self._meta,) if a is not None]
            + [self._UPDATE]
            + [a.name for a in (self._ei, self._cons) if a is not None]
        )
        self.rule_reuses = tuple(
            a.reused for a in (self._meta, self._ei, self._cons) if a is not None
        )

    # -- PRE: base short-term gain, then the metaplastic BCM brake ---------------------------------
    def pre_gain(self, neuron_id):
        g = super().pre_gain(neuron_id)
        if self._meta is not None:
            g *= self._meta.gate(int(neuron_id))
        return g

    # -- POST: base sandwich, then E/I trim, then Benna-Fusi consolidation --------------------------
    def stabilize(self, neuron_id, weights, activity):
        i = int(neuron_id)
        w = super().stabilize(i, weights, activity)  # base: bounded + rate-homeostatic
        drive = sum(wi * ai for wi, ai in zip(w, activity))
        if self._ei is not None:
            w, _ = self._ei.apply(i, w, drive)
        if self._cons is not None:
            w = self._cons.consolidate(i, w)  # deep-anchored effective weights
        if self._meta is not None:
            self._meta.observe(i, drive)  # slide theta for the NEXT pre_gain
        return w

    def relax(self, neuron_id, steps):
        """Rest phase: diffuse the consolidation cascades without new plasticity (no-op if consolidation
        is off). Lets the deep store re-assert an old memory after a conflicting burst. Returns the settled
        weights, or None if there is nothing to relax."""
        if self._cons is None:
            return None
        return self._cons.relax(int(neuron_id), int(steps))

    # -- full superset sandwich, for the self-check: metaplastic gate folded into the live update ----
    def learning_step(self, neuron_id, weights, activity, rstdp_update):
        i = int(neuron_id)
        mg = self._meta.gate(i) if self._meta is not None else 1.0

        def _gated(gain, w):
            return rstdp_update(gain * mg, w)  # metaplastic brake on the effective LR

        w, ctx = super().learning_step(i, weights, activity, _gated)
        drive = sum(wi * ai for wi, ai in zip(w, activity))
        if self._ei is not None:
            w, inhibition = self._ei.apply(i, w, drive)
            ctx["ei_inhibition"] = inhibition
        if self._cons is not None:
            w = self._cons.consolidate(i, w)
            drive = sum(wi * ai for wi, ai in zip(w, activity))
        if self._meta is not None:
            self._meta.observe(i, drive)
            ctx["meta_gate"] = mg
            ctx["theta"] = self._meta.theta(i)
        ctx["weights"] = w
        return w, ctx


# --------------------------------------------------------------------------------------------------
# self-check helpers (deterministic, ASCII)
# --------------------------------------------------------------------------------------------------
def _sign_agree(w, target):
    """Fraction of synapses whose weight SIGN agrees with the target sign pattern (chance 0.5)."""
    return sum(1 for wi, ti in zip(w, target) if (wi > 0.0) == (ti > 0.0)) / len(target)


def _signed_deltas(n_in, target_signs, seed):
    """REUSE RSTDPNeuron to produce the per-synapse signed teaching delta for a target sign pattern
    (causal pre-before-post -> +, anti-causal -> -), exactly the benna_fusi spiking teaching signal."""
    from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron

    r = RSTDPNeuron(n_in, lr=0.20, seed=seed)
    r.e = [0.0] * r.n
    r.t_pre = [None] * r.n
    r.t_post = None
    for i, s in enumerate(target_signs):
        if s > 0:
            r.pre_spike(i, 0.0)
    r.post_spike(1.0)
    for i, s in enumerate(target_signs):
        if s < 0:
            r.pre_spike(i, 2.0)
    return list(r.reward(1.0))


def _clip(x, lo, hi):
    return hi if x > hi else lo if x < lo else x


def _retention_run(stack, n_in, a, d_old, d_new, activity, t_a, t_b, relax, w_bound):
    """Sequential-interference protocol on ONE neuron via the .stabilize contract: consolidate old A,
    burst a conflicting new B, then rest. Returns (A-acc after A, B-acc after B, A-retention after rest).

    A per-synapse magnitude clip (identical for both stacks) models the BOUNDED synaptic efficacy that is
    exactly what makes a plain synapse forget catastrophically -- the cascade escapes it via the deep store.
    """
    new = [-x for x in a]
    w = [0.0] * n_in
    for _ in range(t_a):  # A: consolidate the old pattern
        w = stack.stabilize(
            0, [_clip(w[i] + d_old[i], -w_bound, w_bound) for i in range(n_in)], activity
        )
    a_after_a = _sign_agree(w, a)
    for _ in range(t_b):  # B: conflicting new-pattern burst
        w = stack.stabilize(
            0, [_clip(w[i] + d_new[i], -w_bound, w_bound) for i in range(n_in)], activity
        )
    b_after_b = _sign_agree(w, new)
    relax_fn = getattr(stack, "relax", None)  # base stack has no rest phase
    r = relax_fn(0, relax) if relax_fn is not None else None
    if r is not None:
        w = r
    a_ret = _sign_agree(w, a)
    return a_after_a, b_after_b, a_ret


def _retention_selfcheck():
    """Anti-forgetting: the rule stack retains an OLD pattern through a conflicting NEW burst markedly
    better than the base HomeostaticPlasticityStack (which has no consolidation memory)."""
    n_in, seed = 24, 0
    a = [
        1 if (zlib.crc32(bytes([i])) ^ seed) % 2 else -1 for i in range(n_in)
    ]  # deterministic +/-1 pattern
    d_old = _signed_deltas(n_in, a, seed)
    d_new = _signed_deltas(n_in, [-x for x in a], seed)
    activity = [0.05] * n_in  # small drive: keep the base ceiling inactive
    t_a, t_b, relax, w_bound = 60, 40, 400, 1.0

    # high ceiling on BOTH so homeostatic scaling does not couple synapses -> a clean per-synapse test;
    # the cascade shares the same bounded efficacy (bf_w_max == w_bound) as the base, so the ONLY
    # difference is the multi-timescale deep store.
    base = HomeostaticPlasticityStack(1, seed=seed, ceiling_factor=200.0)
    rule = PlasticityRuleStack(1, seed=seed, ceiling_factor=200.0, bf_w_max=w_bound)

    b_aA, b_aB, b_ret = _retention_run(
        base, n_in, a, d_old, d_new, activity, t_a, t_b, relax, w_bound
    )
    r_aA, r_aB, r_ret = _retention_run(
        rule, n_in, a, d_old, d_new, activity, t_a, t_b, relax, w_bound
    )

    print("=" * 78)
    print(
        "ANTI-FORGETTING -- rule stack vs base stack: retain OLD pattern through a NEW-pattern burst"
    )
    print("=" * 78)
    print(
        f"protocol: consolidate A ({t_a} ticks) -> conflicting B burst ({t_b} ticks) -> rest ({relax})"
    )
    print("                    A learned (after A)   B learned (after B)   A retained (after rest)")
    print(
        f"  base stack             {b_aA:5.3f}                {b_aB:5.3f}                 {b_ret:5.3f}"
    )
    print(
        f"  rule stack             {r_aA:5.3f}                {r_aB:5.3f}                 {r_ret:5.3f}"
    )
    print(f"  chance A-retention = 0.500 ; rule improvement over base = {r_ret - b_ret:+.3f}")
    print("=" * 78)

    assert b_aA > 0.9 and r_aA > 0.9, f"both stacks must first LEARN A ({b_aA:.3f}, {r_aA:.3f})"
    assert r_aB > 0.9, f"the rule stack's fast beaker must still LEARN the new B ({r_aB:.3f})"
    assert b_ret <= 0.5 + 1e-9, (
        f"base stack must FORGET A after the conflicting burst (<= chance): {b_ret:.3f}"
    )
    assert r_ret >= 0.80, f"rule stack must RETAIN A at the measured horizon: {r_ret:.3f}"
    assert r_ret >= b_ret + 0.30, (
        f"rule stack must retain A MARKEDLY better than base: {r_ret:.3f} vs {b_ret:.3f}"
    )
    return b_ret, r_ret


def _adaptive_lr_selfcheck():
    """Adaptive LR: the metaplastic BCM gate lowers the effective learning rate for a neuron that has
    recently been strongly driven, while the base stack's PRE gain has no such activity dependence."""
    seed, settle = 0, 8000
    lo_drive, hi_drive = 1.5, 6.0

    def _settled_gate(drive):
        gate = _BCMMetaGate(1, seed=seed)
        for _ in range(settle):
            gate.observe(0, drive)  # sustained recent drive of this magnitude
        return gate.gate(0), gate.theta(0)

    g_lo, th_lo = _settled_gate(lo_drive)
    g_hi, th_hi = _settled_gate(hi_drive)
    g_cold = _BCMMetaGate(1, seed=seed).gate(0)  # no history -> 1.0

    print()
    print("=" * 78)
    print("ADAPTIVE LR -- metaplastic BCM learning-rate gate (recent activity -> effective LR)")
    print("=" * 78)
    print(f"cold start (no history) gate : {g_cold:.3f}  (== 1.0, no brake)")
    print(f"low recent drive  ({lo_drive}) : theta {th_lo:6.3f} -> gate {g_lo:.3f}")
    print(
        f"high recent drive ({hi_drive}) : theta {th_hi:6.3f} -> gate {g_hi:.3f}  (metaplastic brake)"
    )
    print("=" * 78)

    assert g_cold == 1.0, f"cold-start gate must be 1.0 (no metaplastic history): {g_cold:.3f}"
    assert th_hi > th_lo, (
        f"stronger recent drive must RAISE the BCM threshold theta ({th_hi:.3f} > {th_lo:.3f})"
    )
    assert g_hi < g_lo, (
        f"higher recent activity must LOWER the effective LR gate ({g_hi:.3f} < {g_lo:.3f})"
    )
    assert 0.3 <= g_hi <= 1.0 and 0.3 <= g_lo <= 1.0, "the gate stays bounded in [floor, 1.0]"
    return g_lo, g_hi


def _stability_selfcheck():
    """Bounded + rate-homeostatic PRESERVED: run the full superset sandwich over a potentiating stream and
    show the synapses stay bounded and the firing rate stays at the set-point, versus a diverging control."""
    from snn_data_lake.plasticity import homeostatic_plasticity_stack as HPS

    steps, seed, mu, n_in = 300, 1, 0.15, 10
    code, activity, t_post = HPS._drive_code(n_in)
    from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron

    neuron = RSTDPNeuron(
        n_in, lr=0.1, a_plus=1.0, a_minus=1.0, tau_plus=8.0, tau_minus=8.0, w0=0.3, seed=seed
    )
    update = HPS._rstdp_update_fn(neuron, code, t_post)
    stack = PlasticityRuleStack(1, seed=seed)

    w = list(neuron.w)
    drive_tail, rate_tail = [], []
    for t in range(steps):
        w, ctx = stack.learning_step(0, w, activity, update)
        if t >= steps - 50:
            drive_tail.append(sum(wi * ai for wi, ai in zip(w, activity)))
            rate_tail.append(ctx["rate"])
    s_drive = max(drive_tail)
    s_rate = sum(rate_tail) / len(rate_tail)

    # unstabilized control: raw R-STDP potentiation diverges
    c_neuron = RSTDPNeuron(
        n_in, lr=0.1, a_plus=1.0, a_minus=1.0, tau_plus=8.0, tau_minus=8.0, w0=0.3, seed=seed
    )
    c_update = HPS._rstdp_update_fn(c_neuron, code, t_post)
    cw = list(c_neuron.w)
    for _ in range(steps):
        cw = c_update(1.0, cw)
    c_drive = sum(wi * ai for wi, ai in zip(cw, activity))

    print()
    print("=" * 78)
    print("STABILITY PRESERVED -- full superset sandwich stays bounded + rate-homeostatic")
    print("=" * 78)
    print(
        f"stream: {steps} potentiating R-STDP deposits onto one neuron (full rule stack wrapping each)"
    )
    print(
        f"  drive (v)         : rule {s_drive:8.3f} (bounded)    control {c_drive:12.3f} (diverges)"
    )
    print(f"  firing rate -> mu : rule {s_rate:8.3f} (~mu={mu})")
    print("=" * 78)

    assert s_drive < 20.0, f"the superset must keep the synapses BOUNDED (drive {s_drive:.2f})"
    assert c_drive > 3.0 * s_drive, (
        f"the unstabilized control must DIVERGE past the bounded rule ({c_drive:.1f} vs {s_drive:.2f})"
    )
    assert abs(s_rate - mu) < 0.08, f"rate homeostasis must be preserved ({s_rate:.3f} vs mu={mu})"
    return s_drive, c_drive, s_rate


def _equivalence_and_determinism_selfcheck():
    """Drop-in: with all three extra faculties OFF the superset is BYTE-IDENTICAL to the base stack; and
    with them ON the trajectory is deterministic (same seed -> identical weights)."""
    from snn_data_lake.plasticity import homeostatic_plasticity_stack as HPS

    steps, seed, n_in = 80, 3, 10
    code, activity, t_post = HPS._drive_code(n_in)
    from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron

    def _run(stack):
        neuron = RSTDPNeuron(n_in, lr=0.1, w0=0.3, seed=seed)
        update = HPS._rstdp_update_fn(neuron, code, t_post)
        w = list(neuron.w)
        gains = []
        for _ in range(steps):
            gains.append(round(stack.pre_gain(0), 12))
            w, _ = stack.learning_step(0, w, activity, update)
        return w, gains

    base_w, base_g = _run(HomeostaticPlasticityStack(1, seed=seed))
    off_w, off_g = _run(
        PlasticityRuleStack(1, seed=seed, ei=False, metaplastic=False, consolidation=False)
    )
    on1_w, _ = _run(PlasticityRuleStack(1, seed=seed))
    on2_w, _ = _run(PlasticityRuleStack(1, seed=seed))

    equivalent = off_w == base_w and off_g == base_g
    deterministic = on1_w == on2_w
    reuse_ok = True
    r = PlasticityRuleStack(1, seed=seed)
    r.stabilize(0, [0.3] * n_in, activity)  # size the lazy structures
    reuse_ok = (
        isinstance(r._meta._bcm[0], BCMNeuron)
        and isinstance(r._ei._ctrl[0], InhibitorySTDP)
        and isinstance(r._cons._cas[0][0], BennaFusiSynapse)
    )
    import sys

    stdlib_ok = "numpy" not in sys.modules and "torch" not in sys.modules

    print()
    print("=" * 78)
    print("DROP-IN EQUIVALENCE + DETERMINISM + REUSE")
    print("=" * 78)
    print(
        f"extras OFF == base HomeostaticPlasticityStack (byte-identical weights + gains): {equivalent}"
    )
    print(f"extras ON deterministic (same seed -> identical trajectory): {deterministic}")
    print(f"reuses real modules (BCMNeuron / InhibitorySTDP / BennaFusiSynapse): {reuse_ok}")
    print(f"numpy / torch not imported in this self-check: {stdlib_ok}")
    print("=" * 78)

    assert equivalent, "with all extras OFF the superset must be byte-identical to the base stack"
    assert deterministic, "same seed must give a byte-identical trajectory with the extras ON"
    assert reuse_ok, (
        "the extra faculties must REUSE the real BCMNeuron / InhibitorySTDP / BennaFusiSynapse"
    )
    return equivalent, deterministic


def main():
    b_ret, r_ret = _retention_selfcheck()
    g_lo, g_hi = _adaptive_lr_selfcheck()
    s_drive, c_drive, s_rate = _stability_selfcheck()
    equivalent, deterministic = _equivalence_and_determinism_selfcheck()
    print()
    print(
        f"self-check OK: PlasticityRuleStack is a drop-in SUPERSET of HomeostaticPlasticityStack "
        f"(extras OFF byte-identical: {equivalent}) that ADDS a metaplastic BCM adaptive-LR gate "
        f"(recent activity lowers the effective LR: gate {g_lo:.2f} -> {g_hi:.2f}) and anti-forgetting "
        f"(rule retains an old pattern through a conflicting burst {r_ret:.2f} vs the base's forgetting "
        f"{b_ret:.2f}), REUSING BCMNeuron / InhibitorySTDP / BennaFusiSynapse verbatim, while PRESERVING "
        f"the base's bounded (drive {s_drive:.2f} vs a diverging {c_drive:.1f}) + rate-homeostatic "
        f"(rate {s_rate:.2f}~mu) guarantee; deterministic ({deterministic})."
    )


if __name__ == "__main__":
    main()
