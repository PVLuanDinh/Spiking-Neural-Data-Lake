"""G5 / T113 -- the readout-variant cells (specs/022 evidence/g5-readout-variants.md, frozen at 500d3f58).

ONE instrument for the row's in-brain and planted cells. PART A IS DELETED: its 2.7a reachability gate read
PA1 NOT MET -- 1 of 12 frozen configurations met on discovery (11900-11907), 0 replicated (11936-11943) --
so the trace-delta challenger, its schedule control and the gate were removed before any in-brain cell
(P8 F11: a criterion with no reachable red is deleted, not weakened). The record's "Results: 2.7a" section
carries every contrast verbatim; commit 4ecc90d9 keeps the code as it was run. `OnlineReadout("delta")`
stays the incumbent.

WHAT STAYS (the record's deletion scope): the frozen generator `planted_chain` (record 2.6, Codex #11), the
brain-free engine runner `run_planted` under the closing growth gate L1c's reds use, and the reading (record
2.4) -- the machinery Part B's planted checks (PB1/PB2/PB5), the shift's positive control (PA3) and the
in-brain cells reuse.

THE STATISTIC (record 2.4): `paired_contrast.contrast` at T_BAR; MET iff non-degenerate (ruling 9: a
zero-variance contrast reads NOT MET) AND t > T_BAR AND >= 5 seeds non-tied (|d| > 1e-6, a separate named
restriction). The exact FPR is a DIAGNOSTIC, printed, gating nothing (Codex #9).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.g5_readout_cells
"""

import math

import torch

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast
from snn_data_lake.readout.online_readout import ClosingGate, OnlineReadout

TIE_TOL = 1e-6  # |d| <= this is a tie (L1c's harness value)
MIN_NONTIED = 5  # the separate named restriction (record 2.4)
CHAIN_KINDS = ("burst", "drift", "iid")


# --------------------------------------------------------------------------------------------------
# the frozen generator (record 2.6, Codex #11)
# --------------------------------------------------------------------------------------------------
def planted_chain(seed, kind, noise, n=600, k=4, run=20, d=64):
    """[(features, category)] -- one torch.Generator, draws in a fixed order: prototypes, categories,
    noise. burst/drift: fixed-length runs, each new run's category uniform over the k - 1 others; iid:
    every category i.i.d. uniform (successive pairs independent). drift ramps from the previous run's
    prototype to the current one over the first run // 2 events."""
    if kind not in CHAIN_KINDS:
        raise ValueError(f"kind must be one of {CHAIN_KINDS}, got {kind!r}")
    g = torch.Generator().manual_seed(int(seed))
    protos = torch.randn(k, d, generator=g)
    protos = protos / protos.norm(dim=1, keepdim=True)
    cats = []
    if kind == "iid":
        cats = [int(torch.randint(0, k, (1,), generator=g)) for _ in range(n)]
    else:
        c = int(torch.randint(0, k, (1,), generator=g))
        while len(cats) < n:
            cats.extend([c] * run)
            c = (c + 1 + int(torch.randint(0, k - 1, (1,), generator=g))) % k
        cats = cats[:n]
    eps = torch.randn(n, d, generator=g)
    half = max(1, run // 2)
    out = []
    for t, c in enumerate(cats):
        if kind == "drift":
            j = t % run
            c_prev = cats[t - j - 1] if t - j - 1 >= 0 else c
            w = min(1.0, j / half)
            x = (1.0 - w) * protos[c_prev] + w * protos[c] + noise * eps[t]
        else:
            x = protos[c] + noise * eps[t]
        out.append((x.tolist(), c))
    return out


def run_planted(seed, readout, chain):
    """Live next-category accuracy of the real engine on a planted chain, closing gate, brain-free."""
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=1, online_readout=readout, growth_gate=ClosingGate(3)
    )
    for i, (x, c) in enumerate(chain):
        eng.observe(x, c)
        if i % 20 == 19:
            eng.tick()
    return eng


# --------------------------------------------------------------------------------------------------
# the reading (record 2.4)
# --------------------------------------------------------------------------------------------------
def reading(name, diffs):
    """(met, Contrast, extras): MET iff non-degenerate, t > T_BAR and >= MIN_NONTIED non-tied seeds."""
    c = contrast(name, diffs, T_BAR)
    pattern = "".join("0" if abs(x) <= TIE_TOL else ("+" if x > 0 else "-") for x in c.diffs)
    nontied = sum(1 for ch in pattern if ch != "0")
    n = len(c.diffs)
    mean = sum(c.diffs) / n if n else 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in c.diffs) / n) if n else 0.0
    skew = (sum((x - mean) ** 3 for x in c.diffs) / n) / sd**3 if sd > 0 else 0.0
    met = (not c.degenerate) and c.t > T_BAR and nontied >= MIN_NONTIED
    return met, c, {"pattern": pattern, "nontied": nontied, "mean": mean, "skew": skew}


def _line(name, met, c, x):
    return (
        f"  {name:<44} mean {x['mean']:+.4f}  t {c.t:+.3f}  {x['pattern']}  FPR(diag) {c.fpr:.4f}  "
        f"skew {x['skew']:+.2f}  {'MET' if met else 'NOT MET'}{'  [degenerate]' if c.degenerate else ''}"
    )


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the cheap reds (never builds a Brain, never opens a lake)
# --------------------------------------------------------------------------------------------------
def main():
    torch.set_num_threads(1)
    # the runner is deterministic and the online readout reaches the engine
    chain = planted_chain(20261003, "burst", 0.5, n=200)
    a = run_planted(20261003, OnlineReadout("delta"), chain)
    b = run_planted(20261003, OnlineReadout("delta"), chain)
    assert torch.equal(a._model.beta, b._model.beta), "the runner must be deterministic"
    assert a.state()["online_readout"]["steps"] > 0, "the online readout never moved"
    # the generator: deterministic, the three kinds differ, iid has no run structure
    assert planted_chain(20261003, "drift", 1.0, n=60) == planted_chain(
        20261003, "drift", 1.0, n=60
    )
    burst = [c for _, c in planted_chain(20261003, "burst", 1.0, n=100)]
    iid = [c for _, c in planted_chain(20261003, "iid", 1.0, n=100)]
    assert all(burst[i] == burst[i - i % 20] for i in range(100)), "burst runs are fixed length"
    assert sum(iid[i] == iid[i - 1] for i in range(1, 100)) < 60, "iid must not be run-structured"
    # the reading can say NOT MET: degenerate, too few non-tied, below the bar
    assert not reading("deg", [0.1] * 8)[0], "ruling 9: degenerate reads NOT MET"
    assert not reading("ties", [0.3, 0.2, 0.25, 0.4, 0.0, 0.0, 0.0, 0.0])[0], "4 non-tied < 5"
    assert not reading("noise", [0.1, -0.1, 0.12, -0.11, 0.09, -0.1, 0.1, -0.12])[0]
    assert reading("up", [0.2, 0.3, 0.25, 0.4, 0.15, 0.35, 0.3, 0.2])[0]
    print(
        "g5_readout_cells self-check OK: the runner is deterministic and the online readout moves; the frozen "
        "generator is deterministic and iid is not run-structured; the reading says NOT MET on degenerate, "
        "mostly-tied and sign-balanced samples"
    )


if __name__ == "__main__":
    main()
