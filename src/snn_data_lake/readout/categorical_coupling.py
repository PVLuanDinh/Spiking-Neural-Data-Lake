"""G5 / T113 Part B (specs/022 evidence/g5-readout-variants.md section 3, frozen at 500d3f58): a CALIBRATION
LAYER on the grown readout's output -- a learned map from the category it predicted to the category that
turned out true. NOT the engine-functional test (record 3.1, D6): it grows no neuron, reads no reservoir and
projects nowhere; a win is never cited for P4d's connection 6. It decides one thing: whether a which-confuses-
which coupling is a property worth carrying into Part D's return projection, where it is re-tested.

NO INCUMBENT (record section 1: `grep -i confusion` over src/ fires on `faculties/affect_core.py`'s matrix and
no readout couples categories).

THE COUNTS. C[i, j] counts scored transitions where the readout predicted i and the truth was j, updated
AFTER the live instrument scores, from the same pair the online rule learns from (prequential). One event
moves through three phases in order -- `decide` (the decision is made), `scored` (the caller has scored it
against the truth), `update` (the truth is counted) -- and every out-of-order call raises (PB4, Codex cp4 #4:
a count before SCORING, not merely before deciding, is the leak).

TWO READINGS OF THE SAME COUNTS (one class, one count matrix, a `mode` -- the arms are measurement modes of
ONE mechanism, not two mechanisms; the losing mode is deleted with its arm, record 3.6):

    "full"  K[i, j] = (C[i, j] + alpha [i == j]) / (sum_j C[i, j] + alpha);  argmax_j sum_i q_i K[i, j]
    "marg"  K_marg = diag((m_j + alpha) / (r_j + alpha)),  r = row sums, m = column sums;  argmax_j q_j K_marg[j, j]

"marg" is COUPLED_MARG, the control that is not the method (D2): a diagonal label-shift correction from the
counts' marginals only -- it cannot move evidence from one category to another; only K's off-diagonal can.
alpha = 1 (CHOSEN, one pseudo-count of "the readout was right"; a P1 debt, record open question 3). With no
counts both readings are the identity and the decision is the incumbent's argmax exactly (PB3 / COUPLED0).

SCOPE (record 3.2): learning is untouched (`OnlineReadout.learn` reads the uncoupled softmax), the growth
gate's batch instrument is untouched, and a SILENT row (no unit fires) keeps `NGSNN.decide`'s fallback --
the coupling only re-reads rows that carry evidence. Until Part B lands the brain never constructs this;
the harness wraps `predict_next` (record 3.2's measurement path).
"""

# distinct-from: faculties/affect_core.py -- `_confusion_section` TABULATES a confusion matrix
#   (confusion[true][pred]) as a separability report and never feeds it back into a decision; this module
#   learns C online and re-reads the readout's decision through it. Read 2026-10-04.
# distinct-from: growth/ng_snn.py -- `marginal_holdout` selects held-out ROWS by the category marginal
#   (an admission window), and `decide` is the bare argmax + silent-row fallback this module wraps,
#   never re-implements. Read 2026-10-04.
# distinct-from: readout/growable_readout.py, readout/interaction_learning.py, gates/perf_baseline.py,
#   instruments/pool_selection.py -- matched by the P2 guard on docstring vocabulary only (a zero-fan-out
#   widen, a checkpoint class-path map, a bytes-per-unit slope, a pool's mechanism checks). Read 2026-10-04.

import torch

ALPHA = 1.0  # CHOSEN (P1 debt): one pseudo-count of "the readout was right"
MODES = ("full", "marg")


class CategoricalCoupling:
    def __init__(self, m, mode="full", alpha=ALPHA, frozen=False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        m = int(m)
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        self.mode = mode
        self.alpha = float(alpha)
        self.frozen = bool(frozen)  # COUPLED0: counts never move, K stays the identity
        self.C = torch.zeros(m, m, dtype=torch.float64)
        self._phase = "idle"  # idle -> decided -> scored -> idle (one event)
        self.decisions = 0
        self.moved = 0  # decisions the coupling changed against the uncoupled argmax

    @property
    def m(self):
        return int(self.C.shape[0])

    def coupling(self):
        """The (m, m) matrix the decision multiplies by: K ("full") or diag(K_marg) ("marg")."""
        a = self.alpha
        if self.mode == "full":
            num = self.C + a * torch.eye(self.m, dtype=torch.float64)
            return num / (self.C.sum(dim=1, keepdim=True) + a)
        r, mcol = self.C.sum(dim=1), self.C.sum(dim=0)
        return torch.diag((mcol + a) / (r + a))

    def decide(self, q):
        """The coupled decision for one row of class probabilities q (m,). Opens the event. A re-decide of
        an event not yet scored replaces its decision (the harness may read the prediction twice)."""
        q = torch.as_tensor(q, dtype=torch.float64).reshape(-1)
        if q.shape[0] != self.m:
            raise ValueError(f"q has {q.shape[0]} classes, the coupling {self.m}")
        if self._phase == "scored":
            raise RuntimeError("decide on a scored event: update must close it first")
        self._phase = "decided"
        self.decisions += 1
        pick = int((q @ self.coupling()).argmax())
        self.moved += int(pick != int(q.argmax()))
        return pick

    def scored(self):
        """The caller has scored the decided event against its truth; only now may it be counted."""
        if self._phase != "decided":
            raise RuntimeError(f"scored() in phase {self._phase!r}: nothing was decided to score")
        self._phase = "scored"

    def update(self, pred, truth):
        """Count one SCORED transition (prequential). Refuses an event that was not decided AND scored
        first (PB4): the counts would see the truth before the decision it informs was scored."""
        if self._phase != "scored":
            raise RuntimeError(
                f"update in phase {self._phase!r}: the counts would see the truth before it is scored"
            )
        self._phase = "idle"
        if self.frozen:
            return
        self.C[int(pred), int(truth)] += 1.0

    def state(self):
        return {
            "mode": self.mode,
            "frozen": self.frozen,
            "m": self.m,
            "counts": int(self.C.sum()),
            "decisions": self.decisions,
            "moved": self.moved,
        }


def main():
    """The module's own cheap reds; the planted PB1/PB2/PB5 contrasts live in instruments/g5_readout_cells."""
    g = torch.Generator().manual_seed(20261003)
    # PB3 reach: frozen counts -> the decision is the bare argmax, both modes
    for mode in MODES:
        cc = CategoricalCoupling(4, mode=mode, frozen=True)
        for _ in range(50):
            q = torch.softmax(torch.randn(4, generator=g), dim=0)
            assert cc.decide(q) == int(q.argmax())
            cc.scored()
            cc.update(int(q.argmax()), 0)
        assert cc.moved == 0 and int(cc.C.sum()) == 0
    # PB4 prequential: a count before the event is decided AND scored raises
    for leak in ("undecided", "unscored"):
        cc = CategoricalCoupling(3)
        if leak == "unscored":
            cc.decide(torch.tensor([1.0, 0.0, 0.0]))
        try:
            cc.update(0, 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"a count before scoring must raise ({leak})")
    # "full" learns a systematic relabelling; "marg" cannot move evidence across classes
    full, marg = CategoricalCoupling(3, "full"), CategoricalCoupling(3, "marg")
    for _ in range(30):  # the readout always says 1 when the truth is 0
        for cc in (full, marg):
            cc.decide(torch.tensor([0.0, 1.0, 0.0]))
            cc.scored()
            cc.update(1, 0)
    q = torch.tensor([0.2, 0.7, 0.1])
    assert full.decide(q) == 0, "full coupling must map the systematic 1 -> 0"
    assert marg.coupling()[0, 1] == 0 and marg.coupling()[1, 0] == 0, "marg is diagonal"
    print(
        "categorical_coupling self-check OK: frozen counts reproduce the bare argmax in both modes; a count "
        "before scoring raises; full coupling recovers a systematic relabelling; marg stays diagonal"
    )


if __name__ == "__main__":
    main()
