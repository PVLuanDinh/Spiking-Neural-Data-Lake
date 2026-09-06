"""L1c (specs/022 Phase 2, 2026-09-06): the grown readout learns ONLINE, on every wake percept, between the batch
solves of the sleep refits -- and is re-initialized by the batch solve at every open-gate refit (consolidation).

THE SUBSTRATE. `NGSNN.beta` (L, m) maps the grown units' LIF rates s (1, L) to class scores; the incumbent solves it
analytically inside `fit` at open-gate sleep refits and it is STALE on every wake tick between them, through every
closed-gate period (the mechanism behind L1b's measured harm, evidence/l1b-synaptic-online.md section 4), and for
as long as a one-category buffer makes `_refit` skip its re-solve (the deployed brain: live accuracy 0.0 against
persistence 0.999, evidence/l1c-online-readout.md section 4).

ONE readout, ONE online rule between refits, ONE batch solve at refits (a schedule composition, P2). `learn(model,
s, cat)` moves `model.beta` in place from the pair the live instrument just scored -- the rates of the PREVIOUS
percept and the category that followed -- so the instrument never reads a pair it learned from. Three candidates were measured and the A/B decided (record section 4b): the delta rule LANDED; the perceptron and
the ridge were deleted (P2) and are refused at construction:

  * kind "delta": beta -= lr * s^T (softmax(s @ beta) - onehot(cat)), lr = MU / (||s||^2 + 1) -- laminar's L5
                  rule ported (cortex/laminar_local_readout._observe_lr's normalized form; MU = 1.0 cited from
                  laminar, CHOSEN there -- the P1 debt named; the +1 keeps the denominator >= 1 on a silent percept)
  * kind "sign":  on argmax(s @ beta) != cat: beta[:, cat] += lr*s, beta[:, argmax] -= lr*s -- FrozenExpert's
                  error-sign perceptron (substrate/similarity_sized_growth), single-sample form
  * kind "ridge": the ENGINE'S OWN `NGSNN._solve_beta` (the same activity-regularized ridge the batch uses)
                  re-solved on the last K transitions at every wake percept, moving only beta -- the incumbent
                  solve at a per-sample cadence with forgetting; K = 20 = one refit interval (the online memory
                  is one refit interval; in-brain ~20 fed percepts per refit, measured in the L1b cells). The
                  window is reset when the unit count changes (a refit grew or pruned units).

A category id at or past beta's width widens beta with zero columns first (function-preserving on the old
classes' scores to float tolerance; the new class starts at score 0 -- disclosed). All three read only s, beta's
own columns and the category: the error is the output layer's own, no weight is transposed, no hidden error is
formed -- LOCAL rows, no gradient cosine owed (docs/POLICIES.md e-prop SCOPE). `sign=-1` reverses the LMS update
(the control that is not the method -- directional ONLY at the ported step, record section 2.0c); `mu=0` neuters
an LMS rule (the reach control: bit-identical to no rule, including NO widening); `shuffle_window=True` permutes
the ridge window's labels at every solve (the ridge's control that is not the method).

REACHABLE REDS (self-check + tests/test_online_readout.py), brain-free on the real engine and a drifting chain:
(1) under a growth gate that CLOSES (beta held stale) DELTA and RIDGE beat the batch-only twin engine on both
seeds and the sign flip reads below; (2) ungated (the batch re-solved every 20 observations) RIDGE still beats
batch-only while the LMS rules do not -- the disclosed negative that the first writing of this arc read as a
refusal; (3) the neutered rule leaves beta bit-identical; (4) widening preserves the old classes' scores;
(5) determinism; (6) the online beta survives the growth record's round trip.
"""

import torch

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "OnlineReadout.learn": (
        "LOCAL",
        "single-sample update of the grown readout's OWN output synapses from the unit rates and the "
        "category that followed (laminar's L5 delta rule / FrozenExpert's error-sign perceptron ported "
        "onto NGSNN.beta, or the engine's own ridge re-solved on a sliding window). The error is the output "
        "layer's own; nothing is transposed, no hidden error is formed. A LOCAL rule owes no gradient cosine "
        "(POLICIES e-prop SCOPE)",
    ),
}

MU = 1.0  # cited from cortex/laminar_local_readout (CHOSEN there; the P1 debt named)
K_WINDOW = (
    20  # one refit interval: the online memory (measured ~20 fed percepts per refit in-brain)
)
KINDS = ("delta",)  # the A/B deleted ['sign', 'ridge'] (P2; the record section 4)


class OnlineReadout:
    def __init__(self, kind="delta", mu=MU, sign=1.0, K=K_WINDOW, shuffle_window=False):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.mu = float(mu)
        self.sign = float(sign)
        self.K = int(K)
        self.shuffle_window = bool(shuffle_window)
        self.calls = 0
        self.updates = 0  # calls that changed beta (the sign rule only moves on an error)
        self.cum_abs_dbeta = 0.0
        self.widened = 0
        self.window_resets = 0
        self._win_s = []
        self._win_y = []
        self._shuf = torch.Generator().manual_seed(4242)

    @staticmethod
    def widen(model, m_new):
        """Zero columns for categories beta has never seen (function-preserving on the old classes)."""
        beta = model.beta
        if int(beta.shape[1]) >= m_new:
            return
        pad = torch.zeros(int(beta.shape[0]), m_new - int(beta.shape[1]), device=beta.device)
        model.beta = torch.cat([beta, pad], dim=1)
        model.m = m_new

    def _ridge(self, model, s, cat):
        L = int(s.shape[0])
        if self._win_s and int(self._win_s[-1].shape[0]) != L:  # the unit count changed: a refit
            self._win_s, self._win_y = [], []
            self.window_resets += 1
        self._win_s.append(s.detach().clone())
        self._win_y.append(cat)
        self._win_s, self._win_y = self._win_s[-self.K :], self._win_y[-self.K :]
        if len(self._win_y) < 2:
            return None
        m_new = max(int(model.beta.shape[1]), max(self._win_y) + 1)
        if m_new > int(model.beta.shape[1]):
            self.widen(model, m_new)
            self.widened += 1
        S = torch.stack(self._win_s)
        ys = list(self._win_y)
        if self.shuffle_window:
            perm = torch.randperm(len(ys), generator=self._shuf).tolist()
            ys = [ys[i] for i in perm]
        Y = torch.zeros(len(ys), int(model.beta.shape[1]), device=model.beta.device)
        Y[range(len(ys)), ys] = 1.0
        return model._solve_beta(S.to(model.beta.device), Y)

    def learn(self, model, s, cat):
        """One wake transition: s = the unit rates of the PREVIOUS percept (L,), cat = the category that
        followed. Returns the summed |dbeta| (0.0 when nothing could move)."""
        self.calls += 1
        beta = getattr(model, "beta", None)
        if beta is None or s is None:
            return 0.0
        cat = int(cat)
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32, device=beta.device).reshape(-1)
            if self.kind == "ridge":
                new = self._ridge(model, s, cat)
                if new is None:
                    return 0.0
                beta = model.beta
                dbeta = new - beta
                beta.copy_(new)
            else:
                if self.mu == 0.0:
                    return 0.0
                if cat >= int(beta.shape[1]):
                    self.widen(model, cat + 1)
                    self.widened += 1
                    beta = model.beta
                lr = self.sign * self.mu / (float(s @ s) + 1.0)
                scores = s @ beta
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
            "K": self.K,
            "shuffle_window": self.shuffle_window,
            "calls": self.calls,
            "updates": self.updates,
            "cum_abs_dbeta": round(self.cum_abs_dbeta, 6),
            "widened": self.widened,
            "window_resets": self.window_resets,
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


def _run(seed, readout, gate=None, n=600, flip_every=100):
    from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove

    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=1, online_readout=readout, growth_gate=gate
    )
    for i, c in enumerate(_chain(seed, n, flip_every=flip_every)):
        eng.observe(_percept(c, i), c)
        if i % 20 == 19:
            eng.tick()
    return eng


def main():
    torch.set_num_threads(1)
    # (1) beta held stale under a closed gate: DELTA beats the batch-only twin; the sign flip reads below
    for seed in (0, 1):
        b = _run(seed, None, ClosingGate(3)).state()["live_accuracy"]
        d = _run(seed, OnlineReadout("delta"), ClosingGate(3)).state()["live_accuracy"]
        f = _run(seed, OnlineReadout("delta", sign=-1.0), ClosingGate(3)).state()["live_accuracy"]
        assert d > b + 0.05, f"closed gate, seed {seed}: batch {b:.3f} delta {d:.3f}"
        assert f < b, (
            f"closed gate, seed {seed}: the sign flip {f:.3f} must read below the incumbent {b:.3f}"
        )
    # (2) ungated (the batch re-solved every 20 observations) the delta rule gains nothing -- the disclosed
    # negative the first writing of this arc read as a refusal (record section 2.0c)
    b0 = _run(0, None).state()["live_accuracy"]
    d0 = _run(0, OnlineReadout("delta")).state()["live_accuracy"]
    ungated_gap = d0 - b0
    # (3) the neutered rule is bit-identical to no rule
    base, neut = _run(2, None), _run(2, OnlineReadout("delta", mu=0.0))
    assert torch.equal(base._model.beta, neut._model.beta) and torch.equal(
        base._model.W, neut._model.W
    )
    assert base.state()["live_accuracy"] == neut.state()["live_accuracy"]
    # (4) widening preserves the old classes' scores (to float tolerance: a wider matmul)
    m = _run(3, OnlineReadout("delta"))._model
    s = torch.rand(int(m.W.shape[0]))
    before = (s @ m.beta).clone()
    OnlineReadout.widen(m, int(m.beta.shape[1]) + 2)
    after = s @ m.beta
    assert torch.allclose(after[: before.shape[0]], before, atol=1e-6) and bool(
        (after[before.shape[0] :] == 0).all()
    )
    # (5) determinism
    a, b2 = _run(4, OnlineReadout("delta")), _run(4, OnlineReadout("delta"))
    assert torch.equal(a._model.beta, b2._model.beta)
    # (6) the online beta survives the growth record's round trip; the decoupled instrument restores with it
    from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove

    fresh = NeurogenesisSelfImprove(seed=99, device="cpu")
    assert fresh.restore_growth_state(a.growth_state_bytes()) and torch.equal(
        fresh._model.beta, a._model.beta
    )
    assert torch.equal(fresh._batch_beta, a._model.beta)
    print(
        f"online_readout self-check OK: closed gate -- delta {d:.3f} beats batch-only {b:.3f}, the flip {f:.3f} "
        f"below (seed 1); ungated gap {ungated_gap:+.3f} (the disclosed negative); neutered bit-identical; "
        f"widening exact to 1e-6; deterministic; persisted with the instrument"
    )


if __name__ == "__main__":
    main()
