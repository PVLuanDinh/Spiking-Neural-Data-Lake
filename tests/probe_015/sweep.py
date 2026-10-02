"""The discriminating measurement for specs/015-resolve-weight-transport (C3 / T017-T022).

SCRATCH PROBE. Lives in tests/probe_015/, never promoted to src/, registered in no gate. Kept out
of MlpQ.main() deliberately: main() re-derives on every gate run, and a 4x8x30 sweep there is a
cost every future commit pays to answer a one-time question (research.md D-4).

FOUR FEEDBACK PATHWAYS, differing in ONE vector inside the shipped `dh` step:

    transport         dh[j] = err * W2[a][j] * mask      the incumbent (the violation)
    fixed_random      dh[j] = err * B[a][j] * mask       B drawn once, never moves (vanilla FA)
    converging        dh[j] = err * B[a][j] * mask       B += lr*err*h[j]  (KOLEN-POLLACK)
    permutation_null  dh[j] = err * P[a][j] * mask       P = shuffled W2[a], magnitude preserved

KOLEN-POLLACK, NOT SFDFA, and the naming is load-bearing. SFDFA (Bacho & Chu 2024) estimates
output->hidden weights by propagating directional derivatives as SPIKE GRADES during inference.
MlpQ is a RATE MLP -- no membrane, no threshold, no spike, no time index in update() -- so that
mechanism has nothing to attach to. What transfers is its ESSENTIAL property: feedback that is
ESTIMATED and CONVERGES toward the forward weights rather than being transported or frozen. In a
rate net that is Kolen-Pollack, and MlpQ.update already computes every term: the read-out step
applies `lr * err * h[j]`, so a feedback matrix receiving the SAME increment tracks W2 without ever
reading it. Calling this SFDFA would be the citation defect this lane exists to remove.

REUSES the module's own fixtures and training loop by IMPORT (_make_contexts, _train, _acc,
_xor_action, _linear_action). Reimplementing them would be a second mechanism for one job (P2).

Run:  .venv-lake\\Scripts\\python.exe tests\\probe_015\\sweep.py
"""

import math
import random

from snn_data_lake.faculties.mlp_q import (
    MlpQ,
    _acc,
    _linear_action,
    _make_contexts,
    _xor_action,
)
from tests.probe_015.criterion import COSINE_BAR, cosine, dh

# The module's own fixture constants -- read from main()'s body, not re-chosen here.
LATENT, ACTIONS, N_TRAIN, N_TEST, PASSES, HIDDEN, LR = 2, ("A0", "A1"), 400, 300, 30, 32, 0.1
BSEEDS = tuple(range(1, 9))
VARIANTS = ("transport", "fixed_random", "converging", "permutation_null")
COS_SAMPLES = 40  # held-out contexts the per-epoch cosine averages over


class FeedbackMlpQ(MlpQ):
    """MlpQ with the feedback vector in `dh` swapped. Everything else is the shipped update.

    `decay` is the Kolen-Pollack weight-decay term. It is NOT a new bare constant: it is expressed
    as a MULTIPLE OF lr and defaults to 0.0 (decay-free), so the P1 question "derive it or omit it"
    is answered by measurement rather than by a literal (research.md D-1, open sub-decision).
    """

    def __init__(self, actions, latent_dim, variant, bseed, hidden=32, lr=0.05, seed=0, decay=0.0):
        super().__init__(actions, latent_dim, hidden=hidden, lr=lr, seed=seed)
        self.variant = variant
        self.decay = decay
        rng = random.Random(bseed)  # PER-INSTANCE, never module-level
        self._perm_rng = random.Random(bseed + 10_000)
        scale = math.sqrt(hidden)
        # Same init scale the module's own FA twin uses: uniform(-1,1)/sqrt(hidden).
        self.B = [[rng.uniform(-1.0, 1.0) / scale for _ in range(hidden)] for _ in actions]

    def _feedback(self, i):
        if self.variant == "transport":
            return self.W2[i]
        if self.variant == "permutation_null":
            p = list(self.W2[i])
            self._perm_rng.shuffle(p)
            return p
        return self.B[i]  # fixed_random and converging both read B; only converging moves it

    def update(self, z, a, target):
        i = self._idx(a)
        pre, h = self._forward(z)
        err = target - (sum(w * x for w, x in zip(self.W2[i], h, strict=True)) + self.b2[i])
        lr = self.lr
        d = dh(err, self._feedback(i), pre)

        # read-out step -- identical to the shipped rule
        self.b2[i] += lr * err
        for j in range(self.hidden):
            self.W2[i][j] += lr * err * h[j]

        # KOLEN-POLLACK: the feedback receives the SAME increment, so B tracks W2 without reading
        # it. err and h[j] are already local to this synapse -- nothing new is transported.
        if self.variant == "converging":
            bi = self.B[i]
            for j in range(self.hidden):
                bi[j] += lr * err * h[j] - self.decay * bi[j]

        # shared hidden-layer step -- identical to the shipped rule
        for j in range(self.hidden):
            g = lr * d[j]
            if g == 0.0:
                continue
            self.b1[j] += g
            w1j = self.W1[j]
            for k in range(self.latent_dim):
                w1j[k] += g * z[k]
        return err


def _epoch_cosine(model, contexts):
    """cos(dh_rule, dh_transport) averaged over held-out contexts, at the CURRENT weights.

    Both sides use the same input, the same pre-update weights and the same err -- they differ in
    the feedback vector alone, which is the whole point (contracts/criterion.md).
    """
    tot, n = 0.0, 0
    for z, opt in contexts[:COS_SAMPLES]:
        for i in range(len(ACTIONS)):
            pre, h = model._forward(z)
            err = (1.0 if i == opt else -1.0) - (
                sum(w * x for w, x in zip(model.W2[i], h, strict=True)) + model.b2[i]
            )
            d_rule = dh(err, model._feedback(i), pre)
            d_tran = dh(err, model.W2[i], pre)
            if any(d_tran):
                tot += cosine(d_rule, d_tran)
                n += 1
    return tot / n if n else 0.0


def run(decay=0.0):
    idx_of = {a: i for i, a in enumerate(ACTIONS)}
    nl_train = _make_contexts(N_TRAIN, LATENT, _xor_action, seed=1)
    nl_test = _make_contexts(N_TEST, LATENT, _xor_action, seed=2)
    li_train = _make_contexts(N_TRAIN, LATENT, _linear_action, seed=3)
    li_test = _make_contexts(N_TEST, LATENT, _linear_action, seed=4)

    rows = []
    for variant in VARIANTS:
        for bseed in BSEEDS:
            m = FeedbackMlpQ(
                ACTIONS, LATENT, variant, bseed, hidden=HIDDEN, lr=LR, seed=0, decay=decay
            )
            ml = FeedbackMlpQ(
                ACTIONS, LATENT, variant, bseed, hidden=HIDDEN, lr=LR, seed=0, decay=decay
            )
            for epoch in range(1, PASSES + 1):
                for z, opt in nl_train:
                    for i in range(len(ACTIONS)):
                        m.update(z, i, 1.0 if i == opt else -1.0)
                for z, opt in li_train:
                    for i in range(len(ACTIONS)):
                        ml.update(z, i, 1.0 if i == opt else -1.0)
                rows.append(
                    {
                        "variant": variant,
                        "bseed": bseed,
                        "epoch": epoch,
                        "cosine": _epoch_cosine(m, nl_test),
                        "acc_nonlinear": _acc(m, nl_test, idx_of),
                        "acc_linear": _acc(ml, li_test, idx_of),
                    }
                )
    return rows


def _report(rows, decay):
    n_expected = len(VARIANTS) * len(BSEEDS) * PASSES
    print(f"\n  rows: {len(rows)}  expected {n_expected}  (control: must match exactly)")
    assert len(rows) == n_expected, f"row count {len(rows)} != {n_expected}; the probe is broken"

    # CONTROL: transport must read 1.0 at every epoch. If it does not, no verdict may be read.
    t = [r["cosine"] for r in rows if r["variant"] == "transport"]
    print(f"  transport cosine: min={min(t):.12f} max={max(t):.12f}  (control: must be 1.0)")
    assert all(abs(c - 1.0) < 1e-9 for c in t), "transport is not 1.0; the probe is broken"

    print(f"\n  decay = {decay:.4g} x (as a multiple of lr={LR})")
    print("  " + "=" * 88)
    print(f"  {'variant':<18}{'final |cos| range':<26}{'draws>bar':<12}{'acc XOR':<16}acc linear")
    print("  " + "-" * 88)
    verdict = {}
    for v in VARIANTS:
        fin = [r for r in rows if r["variant"] == v and r["epoch"] == PASSES]
        cs = [abs(r["cosine"]) for r in fin]
        over = sum(1 for c in cs if c > COSINE_BAR)
        an = [r["acc_nonlinear"] for r in fin]
        al = [r["acc_linear"] for r in fin]
        print(
            f"  {v:<18}[{min(cs):.4f}, {max(cs):.4f}]{'':<8}{over} of 8{'':<5}"
            f"[{min(an):.3f}, {max(an):.3f}]  [{min(al):.3f}, {max(al):.3f}]"
        )
        verdict[v] = {
            "over": over,
            "cos": (min(cs), max(cs)),
            "acc_nl": (min(an), max(an)),
            "acc_li": (min(al), max(al)),
            "per_draw": cs,
            "an": an,
            "al": al,
        }

    # TREND: does the converging variant RISE? Compare first third against last third.
    for v in ("fixed_random", "converging"):
        early = [abs(r["cosine"]) for r in rows if r["variant"] == v and r["epoch"] <= PASSES // 3]
        late = [
            abs(r["cosine"]) for r in rows if r["variant"] == v and r["epoch"] > 2 * PASSES // 3
        ]
        print(
            f"  {v:<18}trend: early mean {sum(early) / len(early):.4f} -> "
            f"late mean {sum(late) / len(late):.4f}"
        )
    return verdict


if __name__ == "__main__":
    print("=" * 92)
    print("015 sweep -- cosine ACROSS TRAINING, four feedback pathways")
    print("=" * 92)
    rows = run(decay=0.0)
    _report(rows, 0.0)
