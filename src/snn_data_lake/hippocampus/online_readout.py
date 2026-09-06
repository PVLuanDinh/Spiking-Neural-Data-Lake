"""L1c (specs/022 Phase 2, 2026-09-06): the grown readout learns ONLINE, on every wake percept, by a local rule on
the output synapses -- and is re-initialized by the batch solve at every open-gate sleep refit (consolidation).

THE SUBSTRATE. `NGSNN.beta` (L, m) maps the grown units' LIF rates s (1, L) to class scores; today it is solved
analytically inside `fit` at sleep refits and is STALE on every wake tick between them and through every
closed-gate period (the mechanism behind L1b's measured harm, evidence/l1b-synaptic-online.md section 4).

ONE readout, ONE online rule between refits, ONE batch solve at refits (a schedule composition, P2):
`learn(model, s, cat)` moves `model.beta` in place from the pair the live instrument just scored -- the rates
of the PREVIOUS percept and the category that followed it -- so the instrument never reads a pair it learned
from. Step size, one derivation for both candidates (P1): `lr = MU / (||s||^2 + 1)` -- the normalized-LMS form
of cortex/laminar_local_readout._observe_lr (there the +1 is the bias unit's presynaptic energy; here beta has
no bias, so +1 is CHOSEN as the floor that keeps the denominator >= 1 on a silent percept -- disclosed);
MU = 1.0 cited from laminar (CHOSEN there; the P1 debt named).

  * kind "delta": beta -= lr * s^T (softmax(s @ beta) - onehot(cat))      -- laminar's L5 rule, ported
  * kind "sign":  on argmax(s @ beta) != cat: beta[:, cat] += lr*s, beta[:, argmax] -= lr*s   -- FrozenExpert's
                  error-sign perceptron (substrate/similarity_sized_growth), single-sample form

A category id at or past beta's width widens beta with zero columns first (function-preserving on the old
classes' scores; the new class starts at score 0 -- disclosed). Both rules read only s, beta's own columns and
the category: the error is the output layer's own, no weight is transposed, no hidden error is formed --
LOCAL rows, no gradient cosine owed (docs/POLICIES.md e-prop SCOPE). `sign=-1` reverses the update (the
arc's control that is not the method: a readout moved the wrong way must read BELOW the incumbent);
`mu=0` neuters it (the reach control: bit-identical to no rule, including NO widening).

REACHABLE REDS (self-check + tests/test_online_readout.py): (1) on a planted chain the delta rule raises the
engine's live accuracy above a batch-only twin engine on the same stream, and the sign-flipped rule lowers
it; (2) the neutered rule leaves beta bit-identical; (3) widening preserves the old classes' scores exactly;
(4) determinism; (5) the online beta survives the growth record's round trip.
"""

import torch

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "OnlineReadout.learn": (
        "LOCAL",
        "single-sample update of the grown readout's OWN output synapses from the unit rates and the "
        "category that followed (laminar's L5 delta rule / FrozenExpert's error-sign perceptron, ported "
        "onto NGSNN's beta). The error is the output layer's own; nothing is transposed, no hidden error "
        "is formed. A LOCAL rule owes no gradient cosine (POLICIES e-prop SCOPE)",
    ),
}

MU = 1.0  # cited from cortex/laminar_local_readout (CHOSEN there; the P1 debt named)
KINDS = ("delta", "sign")


class OnlineReadout:
    def __init__(self, kind="delta", mu=MU, sign=1.0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.mu = float(mu)
        self.sign = float(sign)
        self.calls = 0
        self.updates = 0  # calls that changed beta (the sign rule only moves on an error)
        self.cum_abs_dbeta = 0.0
        self.widened = 0

    @staticmethod
    def widen(model, m_new):
        """Zero columns for categories beta has never seen (function-preserving on the old classes)."""
        beta = model.beta
        if int(beta.shape[1]) >= m_new:
            return
        pad = torch.zeros(int(beta.shape[0]), m_new - int(beta.shape[1]), device=beta.device)
        model.beta = torch.cat([beta, pad], dim=1)
        model.m = m_new

    def learn(self, model, s, cat):
        """One wake transition: s = the unit rates of the PREVIOUS percept (L,), cat = the category that
        followed. Returns the summed |dbeta| (0.0 when nothing could move)."""
        self.calls += 1
        beta = getattr(model, "beta", None)
        if beta is None or s is None or self.mu == 0.0:
            return 0.0
        cat = int(cat)
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32, device=beta.device).reshape(-1)
            if cat >= int(beta.shape[1]):
                self.widen(model, cat + 1)
                self.widened += 1
                beta = model.beta
            lr = self.sign * self.mu / (float(s @ s) + 1.0)
            scores = s @ beta  # (m,)
            if self.kind == "delta":
                p = torch.softmax(scores, dim=0)
                p[cat] -= 1.0
                dbeta = -lr * torch.outer(s, p)
            else:
                pred = int(scores.argmax())
                if pred == cat:
                    return 0.0
                dbeta = torch.zeros_like(beta)
                dbeta[:, cat] += lr * s
                dbeta[:, pred] -= lr * s
            beta += dbeta
            d = float(dbeta.abs().sum())
        self.updates += 1
        self.cum_abs_dbeta += d
        return d

    def state(self):
        return {
            "kind": self.kind,
            "mu": self.mu,
            "sign": self.sign,
            "calls": self.calls,
            "updates": self.updates,
            "cum_abs_dbeta": round(self.cum_abs_dbeta, 6),
            "widened": self.widened,
        }


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the five reds, brain-free, on the real engine and a planted chain
# --------------------------------------------------------------------------------------------------
def _chain(seed, n=400, k=4, flip_every=None):
    """A first-order chain: class c is followed by (c + 1) % k with probability 0.85 -- succession the
    readout can learn from the previous percept's rates. With `flip_every`, the succession rule
    REVERSES every that many observations (a DRIFTING world): the batch solve on the whole buffer
    goes stale between refits and the online readout's recency is what the red measures. Measured
    before the build (record section 2.0): drifting +0.19 for the online readout over the stale
    batch, stationary -0.04 (the online step is noise when the batch solve is fresh)."""
    g = torch.Generator().manual_seed(seed)
    cats, c = [], 0
    for i in range(n):
        step = 1 if (flip_every is None or (i // flip_every) % 2 == 0) else -1
        c = (
            (c + step) % k
            if float(torch.rand(1, generator=g)) < 0.85
            else int(torch.randint(0, k, (1,), generator=g))
        )
        cats.append(c)
    return cats


def _percept(c, k, d=64):
    feats = [0.0] * d
    feats[c * 8 : c * 8 + 8] = [1.0] * 8
    feats[40 + (k % 5)] = 0.5
    return feats


def _run(seed, readout, refit_every=1, n=600, flip_every=100):
    from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove

    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=refit_every, online_readout=readout
    )
    for i, c in enumerate(_chain(seed, n, flip_every=flip_every)):
        eng.observe(_percept(c, i), c)
        if i % 20 == 19:
            eng.tick()
    return eng


def main():
    # (1) on the DRIFTING chain the delta rule beats the batch-only twin engine; the sign flip reads below
    for seed in (0, 1):
        base = _run(seed, None)
        live = _run(seed, OnlineReadout("delta"))
        flip = _run(seed, OnlineReadout("delta", sign=-1.0))
        b, l, f = (
            base.state()["live_accuracy"],
            live.state()["live_accuracy"],
            flip.state()["live_accuracy"],
        )
        assert l > b + 0.05, (
            f"seed {seed}: online {l:.3f} must beat batch-only {b:.3f} on a drifting chain"
        )
        assert f < b, (
            f"seed {seed}: the sign-flipped readout {f:.3f} must read below the incumbent {b:.3f}"
        )
        assert live.online_readout.updates > 0
    # the disclosed NEGATIVE: on a STATIONARY chain the online step is noise against a fresh batch solve
    sb = _run(0, None, flip_every=None).state()["live_accuracy"]
    sl = _run(0, OnlineReadout("delta"), flip_every=None).state()["live_accuracy"]
    stationary_gap = sl - sb
    # (2) the neutered rule is bit-identical to no rule
    base, neut = _run(2, None), _run(2, OnlineReadout("delta", mu=0.0))
    assert torch.equal(base._model.beta, neut._model.beta) and torch.equal(
        base._model.W, neut._model.W
    )
    assert base.state()["live_accuracy"] == neut.state()["live_accuracy"]
    # (3) widening preserves the old classes' scores exactly
    m = _run(3, OnlineReadout("sign"))._model
    s = torch.rand(int(m.W.shape[0]))
    before = (s @ m.beta).clone()
    OnlineReadout.widen(m, int(m.beta.shape[1]) + 2)
    after = s @ m.beta
    assert torch.allclose(after[: before.shape[0]], before, atol=1e-6) and bool(
        (after[before.shape[0] :] == 0).all()
    )
    # (4) determinism
    a, b2 = _run(4, OnlineReadout("delta")), _run(4, OnlineReadout("delta"))
    assert torch.equal(a._model.beta, b2._model.beta)
    # (5) the online beta survives the growth record's round trip
    from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove

    blob = a.growth_state_bytes()
    fresh = NeurogenesisSelfImprove(seed=99, device="cpu")
    assert fresh.restore_growth_state(blob) and torch.equal(fresh._model.beta, a._model.beta)
    print(
        f"online_readout self-check OK: on the DRIFTING chain delta beats batch-only (seed 1: {l:.3f} vs {b:.3f}), "
        f"the sign flip reads below ({f:.3f}); STATIONARY chain gap {stationary_gap:+.3f} (disclosed: noise "
        f"against a fresh batch solve); neutered bit-identical; widening exact; deterministic; persisted"
    )


if __name__ == "__main__":
    main()
