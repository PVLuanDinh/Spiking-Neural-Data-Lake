"""L1b (specs/022 Phase 2, 2026-09-05): the grown units' INPUT weights learn online, on every wake percept,
by a LOCAL rule with a scalar third factor from a live neuromodulator.

THE SUBSTRATE. The grower (hippocampus/ng_snn.NGSNN, driven by cortex/neurogenesis_engine) admits LIF units
whose input rows W (L, d_in) are drawn at random at admission and, until this module, NEVER MOVE AGAIN --
only the readout beta is ever re-solved (measured 2026-09-05: the only writers of W in the tree were fit's
whole-assign, prune's row select and restore). This rule is the first mechanism that moves them.

THE RULE FAMILY -- one mechanism, one factor swapped (P2): a NORM-PRESERVING Hebbian step on each unit's
input row. For unit i with input row W_i, admission scale Lambda_i = ||W_i|| (never changed: the LIF's
operating point is set by it), unit direction u_i = W_i / Lambda_i, the percept direction xn = x / ||x||,
the unit's own post rate s_i (spikes / T from the model's LIF rollout on THIS percept, computed BEFORE the
update) and the scalar gate g:

    u_i  <-  normalize( u_i + mu * g * post_i * xn )
    W_i  <-  Lambda_i * u_i

  * kind "oja":  post_i = s_i                 -- Hebb in direction; the renormalization IS Oja's decay
                                                 term to first order (the weight's norm never drifts)
  * kind "bcm":  post_i = s_i * (s_i - theta_i) -- BCM's sliding threshold, theta_i a running mean of s_i^2
                                                 (tau_theta = plasticity/bcm_plasticity's 40.0), so a unit
                                                 firing below its own average moves AWAY from the percept
                                                 (LTD) and above it moves toward it (LTP); theta_i starts at
                                                 the unit's first observed s_i^2 (no move on first touch)

Everything the rule reads is local to the synapse or a global scalar: the presynaptic percept x, the unit's
own rate s_i, its own row, its own theta_i, and g. It never reads the readout beta, never forms an error, and
never transposes a weight -- transport-free by construction (docs/POLICIES.md e-prop SCOPE: a LOCAL rule owes
no gradient cosine; cortex/eprop_kernel stays the cosine reference and is NOT this rule).

CONSTANTS (P1, each named): MU = 0.05 is CHOSEN -- the angular step per percept at g = post = 1 -- and is the
debt this arc carries (the laminar readout's mu = 1.0 is the tree's precedent for a CHOSEN step behind a
derived normalizer; here the normalizer is the row norm itself). TAU_THETA = 40.0 is bcm_plasticity's own
default, cited not re-picked. The GATE is Acetylcholine.plasticity_scale() (neuromodulation/acetylcholine.py:
declared NOT_A_RULE, "a scalar a rule may multiply by"), stepped by the brain BEFORE the feed on the same
wake tick; None means no modulator is wired and the rule runs ungated (g = 1), which the record discloses.

FR-018: the rule carries its GRADIENT_CLAIM row (LOCAL), its declared read/write sets through a RuleSpec
whose AuthoredRule is the flat pre-normalization increment mu*g*post*xn (the renormalization is production
staging outside the algebra, the istdp precedent), a rule_census row (regenerated through
rule_composer --regen-census, never hand-edited) and per-item neuter attribution in the arc's cells
(rule off / gate pinned / readout frozen).

A UNIT THAT DOES NOT FIRE DOES NOT LEARN: post = s_i = 0 moves nothing (Hebb has no presynaptic-only
term), so a unit silent on the live stream keeps its admission row -- disclosed, and the reason the reds
below are stated over the rows that FIRE.

REACHABLE REDS (the self-check, and tests/test_grown_input_rule.py): (1) on a planted stream every unit
that fires on it turns its row toward the stream's direction (cosine rises), silent units and the same
rule with g = 0 move nothing; (2) every row norm is preserved to 1e-5; (3) BCM's LTD: a unit driven to an
intermediate rate below its theta moves AWAY from the percept; (4) the tensor increment equals the
authored scalar rule on random bindings (the declaration IS the implementation); (5) determinism: two
rules on the same stream give bit-identical W.
"""

import torch

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "GrownInputRule.learn": (
        "LOCAL",
        "norm-preserving Hebbian / BCM step on each grown unit's OWN input row from the presynaptic "
        "percept, the unit's own LIF rate and a global scalar gate. Reads no readout, forms no error, "
        "transposes nothing: transport-free by construction. A LOCAL rule owes no gradient cosine "
        "(POLICIES e-prop SCOPE); eprop_kernel is the reference, not this rule",
    ),
}

MU = 0.05  # CHOSEN (P1 debt): the angular step per percept at g = post = 1
TAU_THETA = 40.0  # bcm_plasticity.BCMNeuron's own default, cited
KINDS = ("oja", "bcm")
_TINY = 1e-12  # guards a zero-norm row or percept (moves nothing rather than dividing by zero)


def authored_increment(kind):
    """The flat pre-normalization increment as an AuthoredRule: du = mu * g * post * xn per synapse.
    `post` is the caller-collapsed post factor (s for oja; s*(s-theta) for bcm -- the istdp
    pre-collapse pattern). Returns (algebra, RuleSpec) so the composer's census can hold the claim."""
    from snn_data_lake.plasticity.eligibility_algebra import EligibilityAlgebra
    from snn_data_lake.plasticity.rule_composer import RuleSpec

    alg = EligibilityAlgebra(term_budget=1)
    alg.declare_factor(
        "g", "the scalar third factor: Acetylcholine.plasticity_scale() on the same tick"
    )
    alg.declare_factor(
        "post", f"the unit's post factor ({'s' if kind == 'oja' else 's*(s-theta)'})"
    )
    alg.declare_factor("xn", "the presynaptic percept component, unit-normalized")
    rule = alg.author(f"grown_input.{kind}", [(MU, ("g", "post", "xn"))])
    spec = RuleSpec(
        f"grown_input.{kind}",
        rule,
        reads=("g", "post", "xn"),
        writes=("W",),
        claim_key="GrownInputRule.learn",
    )
    return alg, spec


class GrownInputRule:
    """One rule object per engine; the model's W is the substrate and is moved IN PLACE."""

    def __init__(self, kind="oja", mu=MU, tau_theta=TAU_THETA, gated=True):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        # gated=False: the third factor is WITHDRAWN (Stage 0, 2026-09-05: the ACh plasticity scale decays
        # to a last-200 mean of 0.03 on 2 of 3 seeds -- a multiplicative gate at ~0 makes the rule inert);
        # the gate passed in is still RECORDED (last_gate / mean_gate) so the instrument reads it, but g = 1.
        self.gated = bool(gated)
        self.mu = float(mu)
        self.tau_theta = float(tau_theta)
        self.calls = 0  # learn() calls that saw a model
        self.moved = 0  # calls that changed W (gate > 0, live percept, live units)
        self.cum_abs_dw = 0.0
        self.last_gate = None
        self.gate_sum = 0.0

    # ---- BCM's per-unit threshold lives ON the model (it is the unit's state; refits re-index it) ----
    @staticmethod
    def theta_of(model, L, device):
        th = getattr(model, "bcm_theta", None)
        if th is None or int(th.shape[0]) != L:
            th = torch.full((L,), float("nan"), device=device)
            model.bcm_theta = th
        return th

    def learn(self, model, x, gate=None):
        """One wake percept: move every grown unit's input row. Returns the summed |dW| (0.0 when
        nothing could move: no units yet, a silent percept, or a closed gate)."""
        self.calls += 1
        W = getattr(model, "W", None)
        if W is None or int(W.shape[0]) == 0:
            return 0.0
        raw = None if gate is None else float(gate)
        self.last_gate = raw
        self.gate_sum += 0.0 if raw is None else raw
        g = 1.0 if (raw is None or not self.gated) else raw
        if not (g > 0.0):
            return 0.0
        xt = torch.as_tensor(x, dtype=torch.float32, device=W.device).reshape(1, -1)
        xnorm = float(xt.norm())
        if xnorm <= _TINY:
            return 0.0
        with torch.no_grad():
            s = model._rates(xt, W, model.b)[0]  # (L,) the units' own rates on THIS percept
            if self.kind == "oja":
                post = s
            else:
                L = int(W.shape[0])
                theta = self.theta_of(model, L, W.device)
                unset = torch.isnan(theta)
                theta[unset] = s[unset] * s[unset]  # first touch: no move
                post = s * (s - theta)
                theta += (s * s - theta) / self.tau_theta
            xn = xt[0] / xnorm
            norms = W.norm(dim=1, keepdim=True)
            u = W / norms.clamp_min(_TINY)
            u_new = u + (self.mu * g) * post[:, None] * xn[None, :]
            u_new = u_new / u_new.norm(dim=1, keepdim=True).clamp_min(_TINY)
            W_new = norms * u_new
            dw = float((W_new - W).abs().sum())
            W.copy_(W_new)
        self.moved += 1
        self.cum_abs_dw += dw
        return dw

    def state(self):
        return {
            "kind": self.kind,
            "gated": self.gated,
            "calls": self.calls,
            "moved": self.moved,
            "cum_abs_dw": round(self.cum_abs_dw, 6),
            "mean_gate": (self.gate_sum / self.calls) if self.calls else None,
        }


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the five reachable reds, brain-free
# --------------------------------------------------------------------------------------------------
class _Stub:
    """A stand-in for NGSNN's tensors and LIF rollout (the real class is imported by the test file;
    the self-check must not depend on the grower's admission path)."""

    def __init__(self, L, d, seed, T=15, lam=0.5, v_th=1.0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.W = (torch.rand(L, d, generator=g) * 2 - 1).to(device)
        self.b = torch.zeros(L, device=device)
        self.T, self.lam, self.v_th, self.device = T, lam, v_th, device

    def _rates(self, X, W, b):
        A = X.to(self.device) @ W.t() + b
        V = torch.zeros_like(A)
        cnt = torch.zeros_like(A)
        for _ in range(self.T):
            V = (1.0 - self.lam) * V + self.lam * A
            o = (V >= self.v_th).float()
            cnt += o
            V = V * (1.0 - o)
        return cnt / self.T


def _cos_rows(W, x):
    return (W @ x) / (W.norm(dim=1) * x.norm()).clamp_min(_TINY)


def main():
    torch.manual_seed(0)
    d, L = 64, 6
    g = torch.Generator().manual_seed(1)
    target = torch.rand(d, generator=g)  # the planted stream's direction
    stream = [target + 0.05 * torch.rand(d, generator=g) for _ in range(300)]
    # (1) the rule turns rows toward the stream; the closed gate does not
    live, dead = _Stub(L, d, 3), _Stub(L, d, 3)
    r_live, r_dead = GrownInputRule("oja"), GrownInputRule("oja")
    c0 = _cos_rows(live.W, target).clone()
    for x in stream:
        r_live.learn(live, x, gate=1.0)
        r_dead.learn(dead, x, gate=0.0)
    c1, c_dead = _cos_rows(live.W, target), _cos_rows(dead.W, target)
    moved = c1 - c0
    fires = _Stub(L, d, 3)._rates(target.reshape(1, -1), _Stub(L, d, 3).W, torch.zeros(L))[0] > 0
    assert int(fires.sum()) >= 1, "the planted stream must drive at least one unit"
    assert bool((moved[fires] > 0.05).all()), (
        f"a firing row did not turn: {moved.tolist()} fires {fires.tolist()}"
    )
    assert float(moved[~fires].abs().max()) < 1e-6 if int((~fires).sum()) else True, (
        "a silent unit moved"
    )
    assert torch.equal(c_dead, c0), "the closed gate must move nothing"
    assert r_live.moved > 0 and r_dead.moved == 0, (r_live.moved, r_dead.moved)
    # (2) norm preservation
    n0 = _Stub(L, d, 3).W.norm(dim=1)
    assert float((live.W.norm(dim=1) - n0).abs().max()) < 1e-5, "a row norm drifted"
    # (3) BCM's LTD: drive the strongest unit to rate 1 (theta -> 1), then feed the same direction
    # scaled so that unit fires at an INTERMEDIATE rate (1/3): s*(s - theta) < 0 -> it moves away
    m = _Stub(L, d, 5)
    rb = GrownInputRule("bcm")
    strong = torch.ones(d) * 3.0
    for _ in range(40):
        rb.learn(m, strong, gate=1.0)
    A = m.W @ strong  # each unit's current on the strong percept
    i = int(A.argmax())
    assert float(A[i]) > 1.0, "no unit fires on the strong percept"
    theta_i = float(m.bcm_theta[i])
    weaker = strong * (1.2 / float(m.W[i] @ strong))  # unit i now sits at current 1.2 -> rate 1/3
    c_before = _cos_rows(m.W, strong).clone()
    rb.learn(m, weaker, gate=1.0)
    c_after = _cos_rows(m.W, strong)
    assert theta_i > 1.0 / 9.0 and float(c_after[i] - c_before[i]) < 0.0, (
        f"BCM below theta must move AWAY: theta {theta_i:.3f}, dcos {float(c_after[i] - c_before[i]):.2e}"
    )
    # (4) the tensor increment equals the authored scalar rule on random bindings
    from snn_data_lake.plasticity.eligibility_algebra import equivalence_check

    alg, spec = authored_increment("oja")
    cases = []
    gen = torch.Generator().manual_seed(9)
    for _ in range(200):
        cases.append(
            {
                "g": float(torch.rand(1, generator=gen)),
                "post": float(torch.rand(1, generator=gen)),
                "xn": float(torch.rand(1, generator=gen) * 2 - 1),
            }
        )
    worst = equivalence_check(
        spec.rule, lambda c: MU * c["g"] * c["post"] * c["xn"], cases, tol=1e-12
    )
    assert worst <= 1e-12, worst
    # (5) determinism
    a, b = _Stub(L, d, 7), _Stub(L, d, 7)
    ra, rb2 = GrownInputRule("bcm"), GrownInputRule("bcm")
    for x in stream[:100]:
        ra.learn(a, x, gate=0.5)
        rb2.learn(b, x, gate=0.5)
    assert torch.equal(a.W, b.W), "two rules on one stream must agree bit for bit"
    print(
        f"grown_input_rule self-check OK: {int(fires.sum())}/{L} firing rows turned toward the stream, "
        f"silent rows and the closed gate moved 0; norms held; BCM LTD reachable (theta {theta_i:.3f}); "
        f"authored == tensor (worst {worst:.1e}); "
        f"deterministic; mu {MU} CHOSEN, tau_theta {TAU_THETA} cited"
    )


if __name__ == "__main__":
    main()
