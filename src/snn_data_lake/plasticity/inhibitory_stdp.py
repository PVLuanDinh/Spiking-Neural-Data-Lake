"""
Inhibitory STDP (Vogels-Sprekeler) -- additive spike-timing plasticity of INHIBITORY synapses for E/I balance.

The canonical rule of Vogels, Sprekeler, Zenke, Clopath & Gerstner, "Inhibitory Plasticity Balances
Excitation and Inhibition in Sensory Pathways and Memory Networks", Science 334:1569 (2011). A single
postsynaptic LIF neuron gets a FIXED excitatory drive plus N plastic INHIBITORY synapses. Each inhibitory
synapse carries a presynaptic trace; the neuron carries a postsynaptic trace; both jump +1 on a spike and
leak with tau. On a presynaptic spike w += eta*(x_post - alpha); on a postsynaptic spike w += eta*x_pre for
every synapse; w is clamped >= 0. With alpha = 2*rho0*tau the mean weight change is 2*eta*tau*r_pre*(r_post
- rho0), a negative-feedback controller: inhibition GROWS while the neuron fires above rho0 and SHRINKS
while it fires below, so the rate is driven to the target rho0 and excitation is met by co-tuned inhibition
(detailed E/I balance / network stability).

Genuinely DISTINCT from the existing core plasticity family (the headline differentiator):
- homeostatic_scaling.py: slow MULTIPLICATIVE scaling of EXCITATORY weights toward a rate set-point; global,
  NOT spike-timing. iSTDP is SPIKE-TIMING plasticity of INHIBITORY synapses producing detailed balance.
- intrinsic_plasticity.py: adapts the NEURON'S OWN gain/bias (excitability). iSTDP adapts inhibitory SYNAPSE
  weights, leaving the neuron fixed.
- bcm_plasticity / stdp / r_stdp / valence_stdp: EXCITATORY / reward-modulated potentiation. iSTDP is
  inhibitory and balance-seeking, not reward-driven.
So: inhibitory-synapse spike-timing plasticity that drives the postsynaptic rate to a target and balances
excitation with co-tuned inhibition. Additive/opt-in -- a new member of the plasticity family that changes
no existing default. Ref: Vogels et al., Science 334 (2011).

This is a single postsynaptic neuron with small, Poisson-synthetic presynaptic input -- a rate/
timing caricature of the Vogels 2011 rule, NOT a full recurrent E/I network and NOT cognition/AGI. It is
inhibitory plasticity for E/I balance and network stability only. Deterministic (seeded);
additive (no existing behaviour touched).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.plasticity.inhibitory_stdp        # (also run by run_selfchecks.py)
"""

import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron

GRADIENT_CLAIM = {
    "InhibitorySTDP.step": (
        "LOCAL",
        "per-synapse spike-timing update from the synapse's own traces only (pre spike x "
        "the alpha-offset postsynaptic trace, post spike x presynaptic trace) -- no error "
        "transport, no gradient chain; the canonical Vogels-Sprekeler controller",
    ),
    "istdp_synapse_rule": (
        "LOCAL",
        "authors the per-synapse Vogels-Sprekeler sum-of-products declaration (pre x dev, "
        "post x x_pre, both coefficients eta read from the instance; dev = x_post - alpha "
        "pre-collapsed, the espp deviation-factor pattern) through the T050 algebra; "
        "declares, never applies (specs/022 T053-i)",
    ),
}


class InhibitorySTDP:
    """N plastic inhibitory synapses onto one LIF neuron; iSTDP drives the post rate to rho0 (Vogels 2011)."""

    def __init__(self, n_syn=60, rho0=0.05, tau=20.0, eta=0.005, w_init=0.0):
        self.n = n_syn
        self.rho0 = rho0
        self.eta = eta
        self.decay = math.exp(-1.0 / tau)  # synaptic-trace leak per step
        self.alpha = 2.0 * rho0 * tau  # Vogels depression offset -> fixed point at rho0
        self.w = [w_init] * n_syn  # inhibitory conductance magnitudes (clamped >= 0)
        self.x_pre = [0.0] * n_syn  # per-synapse presynaptic traces
        self.x_post = 0.0  # postsynaptic trace
        self.neuron = LIFNeuron(tau_m=10.0, v_threshold=1.0)

    def step(self, exc_drive, inh_spikes, learn=True):
        """One timestep. Net input = exc_drive - sum_i(w[i]*inh_spikes[i]); apply iSTDP; return 1/0 post spike."""
        self.x_post *= self.decay
        inh = 0.0
        for i in range(self.n):  # leak pre traces + gather this step's inhibitory current
            self.x_pre[i] *= self.decay
            if inh_spikes[i]:
                inh += self.w[i]
        post = self.neuron.step(
            exc_drive - inh
        )  # inhibition subtracts from the fixed excitatory drive
        if learn:
            d = self.eta * (
                self.x_post - self.alpha
            )  # pre spike: potentiate by x_post, depress by alpha
            for i in range(self.n):
                if inh_spikes[i]:
                    w = self.w[i] + d
                    self.w[i] = w if w > 0.0 else 0.0
        for i in range(self.n):  # pre spikes bump their own traces (after the update)
            if inh_spikes[i]:
                self.x_pre[i] += 1.0
        if post:
            if learn:
                for i in range(self.n):  # post spike: potentiate every synapse by its pre trace
                    w = self.w[i] + self.eta * self.x_pre[i]
                    self.w[i] = w if w > 0.0 else 0.0
            self.x_post += 1.0
        return post

    def mean_w(self):
        return sum(self.w) / self.n


# distinct-from: espp_rule.py -- espp_synapse_rule authors the ESPP contrastive rule's
#   declaration (deviation factors x lr*sg*e); this authors the Vogels-Sprekeler iSTDP
#   rule's OWN declaration (trace/event factors x eta). Same sanctioned T050 authoring
#   PATTERN (one declaration per rule is the contract), different rule. Site read.
# distinct-from: spiking_jepa.py -- lpl_synapse_rule authors the LPL modulator branch's
#   declaration for jepa instances; different rule, same pattern. Site read.
def istdp_synapse_rule(ctrl):
    """The AUTHORED per-synapse Vogels-Sprekeler rule for `ctrl` (an InhibitorySTDP
    instance): coefficients READ from the instance at authoring time. specs/022 T053-i --
    iSTDP's OWN declaration through the T050 algebra; the algebra/composer siblings'
    EXEMPT expiry is dated to T053's tick-charged composition, which composes THIS rule
    (T053-ii).

    Factor semantics name EXACT evaluation points in step()'s order (the T051b-i
    factor-order lesson): `pre` = this synapse's presynaptic spike indicator (0/1) this
    step; `dev` = x_post - alpha where x_post is the postsynaptic trace AFTER this step's
    decay and BEFORE any post-spike bump (the pre-branch read point; alpha = 2*rho0*tau,
    read from the instance, PRE-COLLAPSED into the deviation exactly as espp's d_* factors
    are -- production computes eta*(x_post - alpha) in that association, so the collapsed
    factor is what makes the per-branch equality BIT-exact); `post` = the postsynaptic
    spike (0/1) the neuron returned this step; `x_pre` = this synapse's presynaptic trace
    AFTER this step's decay AND its own pre-spike bump (the post-branch read point -- the
    bump lands between the two update branches).

    The declaration describes the UNCLAMPED per-branch dw; production applies the two
    branches SEQUENTIALLY, clamping w >= 0 after each. The self-check therefore holds the
    declaration to the APPLIED weight through the exact STAGED clamp identity (per-gate
    ablation of the authored rule reproduces each branch's dw; staging + clamps reproduce
    the observed weight TO THE BIT), and the flat term-sum form is NOT claimed equal to
    the staged application -- the class that separates them is the both-events-AND-
    pre-clamp intersection, structurally rare, floored non-empty AND driven by a
    constructed discriminating event in the self-check. Disclosed: the post-branch clamp
    is structurally INERT in production (eta and x_pre are non-negative, so that add never
    goes negative); it is mirrored for form, and only the pre-branch clamp discriminates.
    Factor producers: `dev`, `x_pre` and `post` are instance-owned (traces + the neuron's
    own spike); `pre` is the CALLER's spike input (step()'s inh_spikes argument) -- the
    RuleSpec producers declaration rides the T053-ii composition, not the authored rule.
    Returns the algebra-authored rule ('istdp.synapse'); always a FRESH algebra (the
    caller-supplied-algebra parameter pattern was deleted at T051b-i for corrupting on
    refusal paths)."""
    if not isinstance(ctrl, InhibitorySTDP):
        raise ValueError(
            f"istdp_synapse_rule declares InhibitorySTDP instances only (got {type(ctrl).__name__})"
        )
    from snn_data_lake.plasticity.eligibility_algebra import EligibilityAlgebra

    eta = float(ctrl.eta)
    terms = [
        (eta, ("pre", "dev")),
        (eta, ("post", "x_pre")),
    ]
    roles = {
        "pre": "presynaptic spike indicator (0/1) this step, this synapse",
        "dev": (
            "x_post - alpha, pre-collapsed: x_post read AFTER this step's decay and BEFORE "
            f"any post-spike bump; alpha = 2*rho0*tau = {float(ctrl.alpha):.6g} from this "
            "instance (the Vogels depression offset -- fixed point at rho0)"
        ),
        "post": "postsynaptic spike (0/1) from the neuron this step",
        "x_pre": (
            "this synapse's presynaptic trace AFTER this step's decay AND its own "
            "pre-spike bump (the post-branch read point)"
        ),
    }
    alg = EligibilityAlgebra(term_budget=len(terms))
    for f in ("pre", "dev", "post", "x_pre"):
        alg.declare_factor(f, roles[f])
    return alg.author("istdp.synapse", terms)


def simulate(
    exc_drive, seed, steps=20000, n_syn=60, pre_rate=0.05, rho0=0.05, w_init=0.0, learn=True
):
    """Drive an InhibitorySTDP with fixed excitation + seeded Poisson inhibitory spike trains.
    Return (tail firing rate over the last quarter, final mean inhibitory weight, the trained object)."""
    rng = random.Random(seed)
    net = InhibitorySTDP(n_syn=n_syn, rho0=rho0, w_init=w_init)
    rnd = rng.random
    tail = steps // 4
    tail_spikes = 0
    for t in range(steps):
        inh_spikes = [
            1 if rnd() < pre_rate else 0 for _ in range(n_syn)
        ]  # seeded Poisson presynaptic input
        post = net.step(exc_drive, inh_spikes, learn=learn)
        if t >= steps - tail:
            tail_spikes += post
    return tail_spikes / tail, net.mean_w(), net


def main():
    rho0 = 0.05
    exc_strong, exc_weak = 0.20, 0.10  # strong drive fires >> rho0; weak drive fires < rho0
    w_prime = 0.02  # weak case starts pre-inhibited, to show weights melt

    # (1) BALANCING: strong drive un-inhibited fires far above rho0; iSTDP grows inhibition down to rho0.
    uninhib_strong, _, _ = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=False)
    rate_strong, wmean_strong, _ = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=True)

    # (2) BIDIRECTIONAL: weak drive fires below rho0; pre-inhibited weights shrink toward 0 (not over-suppressed).
    uninhib_weak, _, _ = simulate(exc_weak, seed=11, rho0=rho0, w_init=0.0, learn=False)
    rate_weak, wmean_weak, _ = simulate(exc_weak, seed=11, rho0=rho0, w_init=w_prime, learn=True)

    # (4) DETERMINISM: a second fresh strong run must reproduce the weights and rate exactly.
    rate_strong2, _, net2 = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=True)
    _, _, net1 = simulate(exc_strong, seed=7, rho0=rho0, w_init=0.0, learn=True)

    print("=" * 72)
    print("INHIBITORY STDP (Vogels 2011) -- spike-timing plasticity of inhibition for E/I balance")
    print("=" * 72)
    print(
        f"target rho0            : {rho0:.3f} spikes/step   (alpha = 2*rho0*tau = {2 * rho0 * 20.0:.2f})"
    )
    print(
        f"strong drive {exc_strong:.2f}      : un-inhibited {uninhib_strong:.4f} -> post-iSTDP {rate_strong:.4f} (-> rho0)"
    )
    print(
        f"weak drive   {exc_weak:.2f}      : un-inhibited {uninhib_weak:.4f} (below rho0), post-iSTDP {rate_weak:.4f}"
    )
    print(
        f"mean inhib weight      : strong {wmean_strong:.4f} (grew from 0)  vs  weak {wmean_weak:.4f} (shrank from {w_prime:.2f})"
    )
    print("=" * 72)

    # ---- self-check: deterministic asserts, ASCII only ----
    assert abs(rate_strong - rho0) < 0.01, (
        f"iSTDP must drive the strong-drive rate to rho0 ({rate_strong:.4f} vs {rho0})"
    )
    assert abs(rate_strong - rho0) < 0.25 * abs(uninhib_strong - rho0), (
        f"post-iSTDP rate must be far closer to rho0 than the un-inhibited rate ({rate_strong:.4f})"
    )
    assert rate_strong < 0.5 * uninhib_strong, (
        f"balanced rate must fall well below the un-inhibited runaway ({rate_strong:.4f} vs {uninhib_strong:.4f})"
    )
    assert wmean_strong > 0.0, (
        f"strong drive must GROW inhibitory weights from 0 ({wmean_strong:.4f})"
    )
    assert uninhib_weak < rho0, (
        f"weak drive must fire below the target ({uninhib_weak:.4f} < {rho0})"
    )
    assert wmean_weak < w_prime, (
        f"weak drive must SHRINK the pre-inhibited weights ({wmean_weak:.4f} < {w_prime})"
    )
    assert wmean_weak < wmean_strong, (
        f"mean inhib weight must be lower under weak than strong drive ({wmean_weak:.4f} < {wmean_strong:.4f})"
    )
    assert net1.w == net2.w and rate_strong == rate_strong2, (
        "same seed -> identical weights and rate"
    )

    # ---- T053-i: the authored declaration held to PRODUCTION step() ---------------------------
    # Sites harvested from a REAL run: bindings derived from PRE-CALL state + inputs + the
    # DECLARATION-TIME parameter pins. The incumbent side is the OBSERVED per-synapse weight.
    # ALL THREE parameter axes are controlled (the first draft controlled eta only and its
    # "never false GREEN" claim was panel-refuted -- alpha/decay were read off the LIVE object,
    # so the identity followed production into a parameter defect): eta rides in rule.terms;
    # alpha and decay are captured AT AUTHORING and the derivation uses the PINS, so production
    # drifting on any axis mismatches -- each axis's corruption arm is driven RED below.
    # Staged-clamp identity: per-gate ablation of the authored rule gives each branch's dw;
    # production staging + production's own clamp predicate must reproduce the observed weight
    # TO THE BIT. The flat term-sum form is NOT asserted equal, and the class that actually
    # separates the two forms is the both-AND-pre-clamp INTERSECTION -- structurally rare
    # (n=1 in the seeded run), so it is floored non-empty AND driven by a constructed event
    # where staged matches production while flat provably does not.
    # distinct-from: espp_rule.py -- its BIT/FULL/DELTA arms hold the espp declaration to
    #   spiking_jepa's train_step; this is the iSTDP-specific STAGED variant (two clamped
    #   branches, one rule) against InhibitorySTDP.step. Same discipline, different rule and
    #   different staging shape. Site read.
    # distinct-from: spiking_jepa.py -- its L331 runner drives the event retina + LIF backbone
    #   (a forward pass recorder, no declaration equivalence); term overlap is prose-level. Read.
    # distinct-from: plasticity_rule_stack.py -- its L221 deposits weight CHANGES into Benna-Fusi
    #   beakers (consolidation), not a declaration-vs-production identity check. Read.
    # distinct-from: espp_rule.py -- different rule, same declaration discipline. Read.
    # distinct-from: spiking_jepa.py -- L331 is a forward-pass recorder; prose overlap. Read.
    # distinct-from: self_improvement_suite.py -- L258 wraps a census; prose overlap. Read.
    def _staged_identity(net, rule, steps, seed, drive, alpha_pin, dec_pin):
        """Per synapse-event of a seeded production run, hold the authored declaration to
        the observed weight, deriving bindings from the DECLARATION-TIME pins (alpha_pin,
        dec_pin) -- never from the live object, so production drifting from the declared
        parameters mismatches instead of being followed. Returns (checked, mismatches,
        class counts incl. the both-and-clamp intersection)."""
        rng = random.Random(seed)
        counts = {"pre_only": 0, "post_only": 0, "both": 0, "clamp": 0, "both_and_clamp": 0}
        checked = mism = 0
        for _t in range(steps):
            inh = [1 if rng.random() < 0.08 else 0 for _ in range(net.n)]
            xp0 = list(net.x_pre)
            xpost0 = net.x_post
            w0 = list(net.w)
            post = net.step(drive, inh, learn=True)
            for i in range(net.n):
                pre_i = 1.0 if inh[i] else 0.0
                post_i = 1.0 if post else 0.0
                if pre_i == 0.0 and post_i == 0.0:
                    continue
                dev = xpost0 * dec_pin - alpha_pin  # production's association, PINNED params
                x_pre_b = xp0[i] * dec_pin + pre_i  # post-branch read point (bump included)
                dw_pre = rule.evaluate(pre=pre_i, dev=dev, post=0.0, x_pre=x_pre_b)
                dw_post = rule.evaluate(pre=0.0, dev=dev, post=post_i, x_pre=x_pre_b)
                w_mid = w0[i] + dw_pre
                clamped = pre_i == 1.0 and not (w_mid > 0.0)
                w_mid = w_mid if w_mid > 0.0 else 0.0  # production's exact predicate
                w_fin = w_mid + dw_post
                w_fin = w_fin if w_fin > 0.0 else 0.0
                checked += 1
                if w_fin != net.w[i]:
                    mism += 1
                if pre_i and post_i:
                    counts["both"] += 1
                    if clamped:
                        counts["both_and_clamp"] += 1
                elif pre_i:
                    counts["pre_only"] += 1
                else:
                    counts["post_only"] += 1
                if clamped:
                    counts["clamp"] += 1
        return checked, mism, counts

    decl_net = InhibitorySTDP(n_syn=8, rho0=0.05)
    rule = istdp_synapse_rule(decl_net)
    # DECLARATION-TIME pins: eta rides in rule.terms; alpha/decay are captured here and the
    # identity derives from THEM (the panel's blocker: reading them off the live object made
    # the identity follow production into an alpha/decay defect -- two axes with no red).
    eta_pin = float(decl_net.eta)
    alpha_pin = float(decl_net.alpha)
    dec_pin = float(decl_net.decay)
    assert rule.terms == ((eta_pin, ("pre", "dev")), (eta_pin, ("post", "x_pre"))), (
        f"authored terms must pin exactly (got {rule.terms})"
    )
    mut = InhibitorySTDP(n_syn=8, rho0=0.05)
    mut.eta = mut.eta * 2.0
    assert istdp_synapse_rule(mut).terms != rule.terms, (
        "a mutated instance must author different terms (the retype red)"
    )
    try:
        istdp_synapse_rule(object())
        raise AssertionError("a non-InhibitorySTDP instance must refuse")
    except ValueError as e:
        assert "InhibitorySTDP instances only" in str(e)

    checked, mism, counts = _staged_identity(
        decl_net, rule, steps=6000, seed=20260831, drive=0.20, alpha_pin=alpha_pin, dec_pin=dec_pin
    )
    assert mism == 0, f"declaration != production on {mism}/{checked} synapse-events"
    # LIVE-site FLOORS per class (one-sided by design -- an increase is not a defect; the
    # exact pin lives on rule.terms above). Every class non-zero by measurement, not hope.
    # clamp=10 is step-count-flat (binds only in the early w~0 regime, measured across a 4x
    # sweep); `both` is RARE but grows slowly with steps (5 -> 6 at 12000 -- a pre spike
    # raises that step's inhibition and tends to VETO the post spike, the E/I
    # anti-correlation this rule exists to produce: rare, NOT invariant).
    assert checked >= 6395 and counts["pre_only"] >= 3795, f"site coverage collapsed: {counts}"
    assert counts["post_only"] >= 2595 and counts["both"] >= 5 and counts["clamp"] >= 10, (
        f"an event class lost coverage: {counts}"
    )
    # the class that separates STAGED from FLAT is the both-AND-pre-clamp intersection --
    # floored non-empty so the staging claim is never validated on zero discriminating events
    assert counts["both_and_clamp"] >= 1, f"staged-vs-flat discriminating class empty: {counts}"
    # CONSTRUCTED staged-vs-flat red: one deterministic event where the pre branch clamps and
    # the post branch fires -- staged must match production, the flat one-shot form must NOT.
    sn = InhibitorySTDP(n_syn=1, rho0=0.05)
    srule = istdp_synapse_rule(sn)
    sn.w[0] = 1e-9  # tiny weight: the pre-branch depression clamps it to zero
    s_dev = sn.x_post * sn.decay - sn.alpha  # x_post=0 -> dev = -alpha < 0
    s_xpre = sn.x_pre[0] * sn.decay + 1.0
    s_post = sn.step(5.0, [1], learn=True)  # huge drive -> post fires; pre spike same step
    assert s_post == 1, "construction needs the post spike to fire"
    s_dw_pre = srule.evaluate(pre=1.0, dev=s_dev, post=0.0, x_pre=s_xpre)
    s_dw_post = srule.evaluate(pre=0.0, dev=s_dev, post=1.0, x_pre=s_xpre)
    s_mid = 1e-9 + s_dw_pre
    s_mid = s_mid if s_mid > 0.0 else 0.0
    s_staged = s_mid + s_dw_post
    s_staged = s_staged if s_staged > 0.0 else 0.0
    s_flat = 1e-9 + (s_dw_pre + s_dw_post)
    s_flat = s_flat if s_flat > 0.0 else 0.0
    assert s_staged == sn.w[0], "staged form must reproduce production on the constructed event"
    assert s_flat != sn.w[0], "the flat one-shot form must FAIL here -- else staging claims nothing"
    # corrupted-production controls IN THE SAME COMMAND, one per parameter axis: production
    # drifting from the declared eta, alpha, or decay must each RED the identity.
    ax_red = {}
    for ax, mutate in (
        ("eta", lambda n: setattr(n, "eta", n.eta * 1.25)),
        ("alpha", lambda n: setattr(n, "alpha", n.alpha * 1.25)),
        ("decay", lambda n: setattr(n, "decay", n.decay * 0.5)),
    ):
        cnet = InhibitorySTDP(n_syn=8, rho0=0.05)
        crule = istdp_synapse_rule(cnet)
        cap, cdp = float(cnet.alpha), float(cnet.decay)
        mutate(cnet)  # corrupt production AFTER authoring; pins stay declaration-time
        _c2, cmism, _cc = _staged_identity(
            cnet, crule, steps=400, seed=20260831, drive=0.20, alpha_pin=cap, dec_pin=cdp
        )
        ax_red[ax] = cmism
        assert cmism > 0, f"corrupted production ({ax}) stayed green -- the identity cannot see"

    print(
        f"self-check OK: un-inhibited {uninhib_strong:.3f} -> iSTDP {rate_strong:.3f} ~ rho0 {rho0:.3f}; "
        f"weights strong {wmean_strong:.3f} > weak {wmean_weak:.3f}; additive; deterministic"
    )
    print(
        f"T053-i declaration OK: staged-clamp identity {checked} synapse-events bit-equal "
        f"(pre_only {counts['pre_only']}, post_only {counts['post_only']}, both {counts['both']}, "
        f"clamp {counts['clamp']}, both-and-clamp {counts['both_and_clamp']} -- the staged-vs-flat "
        f"discriminating class, plus its constructed event where flat FAILS); corrupted-production "
        f"controls RED on all three axes (eta {ax_red['eta']}, alpha {ax_red['alpha']}, "
        f"decay {ax_red['decay']} mismatches); terms + alpha/decay pinned at authoring; refusal red"
    )


if __name__ == "__main__":
    main()
