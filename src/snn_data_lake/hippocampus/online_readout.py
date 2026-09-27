"""L1c (specs/022 Phase 2, 2026-09-06): the grown readout learns ONLINE, on every wake percept, between the batch
solves of the sleep refits -- and is re-initialized by the batch solve at every open-gate refit (consolidation).

THE SUBSTRATE. `NGSNN.beta` (L, m) maps the grown units' LIF rates s (1, L) to class scores; the incumbent solves it
analytically inside `fit` at open-gate sleep refits and it is STALE on every wake tick between them, through every
closed-gate period (the mechanism behind L1b's measured harm, evidence/l1b-synaptic-online.md section 4), and for
as long as a one-category buffer makes `_refit` skip its re-solve (the deployed brain: live accuracy 0.0 against
persistence 0.999, evidence/l1c-online-readout.md).

ONE readout, ONE online rule between refits, ONE batch solve at refits (a schedule composition, P2). `learn(model,
s, cat)` moves `model.beta` in place from the pair the live instrument just scored -- the rates of the PREVIOUS
percept and the category that followed -- so the instrument never reads a pair it learned from. THE RULE (the A/B's
survivor; evidence/l1c-online-readout.md section 4b -- FrozenExpert's error-sign perceptron and a sliding-window
ridge were measured beside it and deleted, P2; the archive and commit 7862d44 keep their code):

    beta -= lr * s^T (softmax(s @ beta) - onehot(cat)),   lr = MU / (||s||^2 + 1)

-- laminar's L5 delta rule (cortex/laminar_local_readout) ported onto NGSNN.beta in the normalized form of its
`_observe_lr`. MU = 1.0 is laminar's value: laminar claims it under P1 exception (a) and records that its
"fully corrective" NLMS derivation was DELETED because NLMS convergence does not transfer to a softmax readout;
the same caveat applies here and the value is load-bearing (the interval-derived 1/20 was inert; the sign-flipped
control is directional only at this step) -- the in-brain step sweep with the sign-flip control at each step is
the owed item (record section 2.1). The +1 keeps the denominator >= 1 on a silent percept (CHOSEN).

A category id at or past beta's width widens beta with zero columns first (function-preserving on the old
classes' scores to float tolerance; the new class starts at score 0 -- disclosed). The rule reads only s, beta's
own columns and the category: the error is the output layer's own, no weight is transposed, no hidden error is
formed -- a LOCAL row, no gradient cosine owed (docs/POLICIES.md e-prop SCOPE). `sign=-1` reverses the update (the
control that is not the method -- directional at this step: -0.52 below the incumbent on 8 of 8 in-brain); `mu=0`
neuters it (the reach control: bit-identical to no rule, including NO widening).

THE DECOUPLED INSTRUMENT (the engine's, record section 3.3): a closed-gate refit scores the held-out slice with
the last OPEN-gate solve's beta, so the growth gate's evidence is what a batch solve extracts from the current
units, while this rule moves the beta the brain predicts with. The instrument is PERSISTED in the growth record
(`batch_beta`, optional) and restored with it; on a blob without it the only beta there is becomes the instrument.

CHARGE, disclosed: `dbeta = -lr * outer(s, p)` is identically zero on a unit that does not fire (s = 0). In the
cells (2-7 grown units) the rule carried Sigma|dbeta| 54-194 per run and +0.14 live accuracy; on the deployed
brain's single silent unit it carries 0.0 -- reachable, not charged (P3), until a unit fires on the stream the
brain lives in.

REACHABLE REDS (self-check + tests/test_online_readout.py), brain-free on the real engine and a drifting chain:
(1) under a growth gate that CLOSES (beta held stale) the rule beats the batch-only twin engine on both seeds and
the sign flip reads below; (2) ungated (the batch re-solved every 20 observations) it gains nothing -- the disclosed
negative that the first writing of this arc read as a refusal; (3) the neutered rule leaves beta bit-identical;
(4) widening preserves the old classes' scores; (5) determinism; (6) the growth record round trip restores the
online beta AND the batch instrument as two different tensors, and a restored engine's closed-gate trajectory
continues identical to a restored batch-only twin's (the instrument survives a restart).
"""

import torch

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "OnlineReadout.learn": (
        "LOCAL",
        "single-sample update of the grown readout's OWN output synapses from the unit rates and the "
        "category that followed (laminar's L5 delta rule ported onto NGSNN.beta). The error is the output "
        "layer's own; nothing is transposed, no hidden error is formed. A LOCAL rule owes no gradient cosine "
        "(POLICIES e-prop SCOPE)",
    ),
}

MU = 1.0  # laminar's value (P1 exception (a) there; the non-transfer caveat above; the step sweep owed)
KINDS = (
    "delta",
)  # the A/B deleted "sign" and "ridge" (P2; record section 4b): refused at construction


class OnlineReadout:
    def __init__(self, kind="delta", mu=MU, sign=1.0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.mu = float(mu)
        self.sign = float(sign)
        self.calls = 0
        self.steps = 0  # calls whose update was non-zero (a silent unit's step is exactly 0)
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
            p = torch.softmax(s @ beta, dim=0)
            p[cat] -= 1.0
            dbeta = -lr * torch.outer(s, p)
            beta += dbeta
            d = float(dbeta.abs().sum())
        if d > 0.0:
            self.steps += 1
        return d

    def state(self):
        return {
            "kind": self.kind,
            "mu": self.mu,
            "sign": self.sign,
            "calls": self.calls,
            "steps": self.steps,
            "widened": self.widened,
        }


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the six reds, brain-free, on the real engine and a drifting chain
# --------------------------------------------------------------------------------------------------
def _chain(seed, n=600, k=4, flip_every=100):
    """A first-order chain (class c -> (c + step) % k with probability 0.85) whose succession rule
    REVERSES every `flip_every` observations: a drifting world, the regime in which a readout's recency
    is worth something. `flip_every=None` gives the stationary chain (the disclosed negative)."""
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


class ClosingGate:
    """A T047-shaped gate that opens for the first `k` refits and then holds CLOSED: the refit scores the
    retained model without re-solving beta (the engine's closed-gate branch) -- beta held stale."""

    def __init__(self, k=3):
        self.k, self.n = k, 0

    def __call__(self, trajectory):
        self.n += 1
        return self.n <= self.k


def _feed(eng, cats, start=0):
    for i, c in enumerate(cats, start):
        eng.observe(_percept(c, i), c)
        if i % 20 == 19:
            eng.tick()


def _run(seed, readout, gate=None, n=600, flip_every=100):
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=1, online_readout=readout, growth_gate=gate
    )
    _feed(eng, _chain(seed, n, flip_every=flip_every))
    return eng


def main():
    torch.set_num_threads(1)
    # (1) beta held stale under a closed gate: the rule beats the batch-only twin; the sign flip reads below
    for seed in (0, 1):
        b = _run(seed, None, ClosingGate(3)).state()["live_accuracy"]
        d = _run(seed, OnlineReadout("delta"), ClosingGate(3)).state()["live_accuracy"]
        f = _run(seed, OnlineReadout("delta", sign=-1.0), ClosingGate(3)).state()["live_accuracy"]
        assert d > b + 0.05, f"closed gate, seed {seed}: batch {b:.3f} delta {d:.3f}"
        assert f < b, f"closed gate, seed {seed}: the sign flip {f:.3f} must read below {b:.3f}"
    # (2) ungated: no gain -- the disclosed negative
    b0 = _run(0, None).state()["live_accuracy"]
    d0 = _run(0, OnlineReadout("delta")).state()["live_accuracy"]
    ungated_gap = d0 - b0
    # (3) the neutered rule is bit-identical to no rule
    base, neut = _run(2, None), _run(2, OnlineReadout("delta", mu=0.0))
    assert torch.equal(base._model.beta, neut._model.beta)
    assert torch.equal(base._model.W, neut._model.W)
    assert base.state()["live_accuracy"] == neut.state()["live_accuracy"]
    # (4) widening preserves the old classes' scores (to float tolerance: a wider matmul)
    m = _run(3, OnlineReadout("delta"))._model
    s = torch.rand(int(m.W.shape[0]))
    before = (s @ m.beta).clone()
    OnlineReadout.widen(m, int(m.beta.shape[1]) + 2)
    after = s @ m.beta
    assert torch.allclose(after[: before.shape[0]], before, atol=1e-6)
    assert bool((after[before.shape[0] :] == 0).all())
    # (5) determinism
    a, b2 = _run(4, OnlineReadout("delta")), _run(4, OnlineReadout("delta"))
    assert torch.equal(a._model.beta, b2._model.beta)
    # (6) the record round trip carries the online beta AND the batch instrument as two tensors, and a
    # restored engine's closed-gate trajectory continues identical to a restored batch-only twin's
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    live = _run(6, OnlineReadout("delta"), ClosingGate(3), n=300)
    twin = _run(6, None, ClosingGate(3), n=300)
    assert not torch.equal(live._model.beta, live._batch_beta), "the online beta never moved"
    fresh = NeurogenesisSelfImprove(
        seed=99, device="cpu", online_readout="delta", growth_gate=ClosingGate(0)
    )
    fresh_twin = NeurogenesisSelfImprove(seed=99, device="cpu", growth_gate=ClosingGate(0))
    assert fresh.restore_growth_state(live.growth_state_bytes())
    assert fresh_twin.restore_growth_state(twin.growth_state_bytes())
    assert torch.equal(fresh._model.beta, live._model.beta)
    assert torch.equal(fresh._batch_beta, live._batch_beta)
    assert torch.equal(fresh._batch_beta, twin._model.beta), "the instrument is not the last solve"
    tail = _chain(6, 400, flip_every=100)[300:]
    _feed(fresh, tail, 300)
    _feed(fresh_twin, tail, 300)
    assert [r[2] for r in fresh.trajectory] == [r[2] for r in fresh_twin.trajectory], (
        "the gate's evidence diverged after a restart"
    )
    print(
        f"online_readout self-check OK: closed gate -- delta {d:.3f} beats batch-only {b:.3f}, the "
        f"flip {f:.3f} below (seed 1); ungated gap {ungated_gap:+.3f} (the disclosed negative); "
        f"neutered bit-identical; widening exact to 1e-6; deterministic; the instrument survives a restart"
    )


if __name__ == "__main__":
    main()
