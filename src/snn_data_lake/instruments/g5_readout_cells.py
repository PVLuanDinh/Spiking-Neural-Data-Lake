"""G5 / T113 -- the readout-variant cells (specs/022 evidence/g5-readout-variants.md, frozen at 500d3f58).

ONE instrument for Part A (the trace-delta challenger against the incumbent `OnlineReadout("delta")`) and,
later, Part B (the categorical coupling). This first build carries what P1 owes BEFORE any in-brain cell:
the 2.7a REACHABILITY GATE, brain-free, on the real engine (`NeurogenesisSelfImprove`) under the closing
growth gate L1c's reds use (`online_readout.ClosingGate(3)`).

THE CANDIDATE LIVES HERE, NOT IN `readout/` (record 2.2). `TraceDelta` is a subclass of `OnlineReadout` the
brain never constructs; only if 2.7a is met AND Part A lands does the rule move into `OnlineReadout` itself
(the P2 merge). On a null this module's candidate and arms are deleted and `readout/online_readout.py` is
never touched. THE RULE, on the same s and cat the incumbent uses:

    e    <- lambda * e + (1 - lambda) * s        # presynaptic eligibility: an EMA of the readout's input rates
    p     = softmax(s @ beta);  p[cat] -= 1       # the output layer's OWN error on the CURRENT prediction
    beta -= sign * MU / (||e||^2 + 1) * outer(e, p)

At lambda = 0, e = s and the update is byte-identical to `OnlineReadout.learn` (PA5, checked below by tensor
equality, not by reading). LIFECYCLE (record 2.2, PA6): e resets to zero whenever the retained model OBJECT
changes (an open-gate refit re-solves beta and swaps the object); it continues across a closed-gate refit
(same object); a widen adds beta columns, not rows; a row-count mismatch RAISES, never broadcasts.

`DeltaSched` (record 2.3, Codex #10) is the delta DIRECTION outer(s, p) at the trace's eventwise scalar step
c_e * ||e||, from a shadow EMA with the same lambda and lifecycle: it carries the trace's adaptive step
schedule and none of its historical direction. Its matching check is IN-ARM (a_t * ||s|| == c_e * ||e|| to
relative 1e-5) -- the full update norm differs per arm through ||p|| and is never compared.

THE GATE (record 2.7a; family frozen, never widened after a look): 12 configurations, kind {drift, burst} x
noise {2.0, 3.0, 4.0} x lambda {exp(-1), exp(-1/3)} (printed 0.368 / 0.717), run length 20, on the frozen
`planted_chain`. contrast(TRACE - DELTA) is read on DISCOVERY 11900-11907; a configuration counts as MET only
if it REPLICATES at the same bar on 11936-11943, run only for configurations met on discovery (lead ruling
L2). PA1 is MET iff one configuration is met on discovery AND replicates; the tie-break is the largest
replication mean gain. contrast(TRACE - DELTA_SCHED) is printed beside, never a bar here. If PA1 is NOT met,
Part A is DELETED before any in-brain cell (P8 F11: a criterion with no reachable red is deleted, not
weakened). The pilot this gate was registered on LOST on every planted stream but one marginal cell
(record 2.6a, carried verbatim there).

THE STATISTIC (record 2.4): `paired_contrast.contrast` at T_BAR; MET iff non-degenerate (ruling 9: a
zero-variance contrast reads NOT MET) AND t > T_BAR AND >= 5 seeds non-tied (|d| > 1e-6, a separate named
restriction). The exact FPR is a DIAGNOSTIC, printed, gating nothing (Codex #9).

Modes
  (bare)    the run_selfchecks path: the cheap reds -- reach (PA5), lifecycle (PA6), determinism (PA7),
            the generator's determinism and shape, DeltaSched's matching, the reading's NOT MET cases.
            Never builds a Brain; never opens a lake.
  --gate    the 2.7a reachability gate (brain-free, one process, minutes).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.g5_readout_cells [--gate]
"""

import math
import sys
import weakref

import torch

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast
from snn_data_lake.readout.online_readout import MU, ClosingGate, OnlineReadout

TIE_TOL = 1e-6  # |d| <= this is a tie (L1c's harness value)
MIN_NONTIED = 5  # the separate named restriction (record 2.4)
SCHED_TOL = 1e-5  # DeltaSched's in-arm matching tolerance (float32)
LAMBDAS = (math.exp(-1.0), math.exp(-1.0 / 3.0))  # printed 0.368 / 0.717
GATE_FAMILY = tuple(
    (kind, noise, lam)
    for kind in ("drift", "burst")
    for noise in (2.0, 3.0, 4.0)
    for lam in LAMBDAS
)
DISCOVERY = tuple(range(11900, 11908))
REPLICATION = tuple(range(11936, 11944))
CHAIN_KINDS = ("burst", "drift", "iid")
ALPHA = 0.05  # the per-contrast nominal one-sided alpha T_BAR stands for (7 d.f.)


# --------------------------------------------------------------------------------------------------
# the candidate and its schedule control
# --------------------------------------------------------------------------------------------------
class _Eligibility:
    """The presynaptic EMA and its lifecycle, shared by TraceDelta and DeltaSched (one implementation)."""

    def _init_trace(self, lam):
        lam = float(lam)
        if not 0.0 <= lam < 1.0:
            raise ValueError(f"lambda must be in [0, 1), got {lam}")
        self.lam = lam
        self._e = None
        self._owner = None  # weakref to the retained model object the trace belongs to
        self.resets = 0

    def _advance(self, model, s):
        owner = self._owner() if self._owner is not None else None
        if owner is not model:  # a new model object: the open-gate refit re-initialized beta
            if self._e is not None:
                self.resets += 1
            self._owner = weakref.ref(model)
            self._e = torch.zeros_like(s)
        if self._e.shape != s.shape:
            raise ValueError(
                f"trace rows {tuple(self._e.shape)} != rate rows {tuple(s.shape)} on the same model: "
                "a trace never broadcasts"
            )
        self._e = self.lam * self._e + (1.0 - self.lam) * s
        return self._e


class TraceDelta(_Eligibility, OnlineReadout):
    """The challenger (record 2.2). lambda = 0 IS the incumbent, bit for bit."""

    def __init__(self, lam, mu=MU, sign=1.0):
        OnlineReadout.__init__(self, "delta", mu=mu, sign=sign)
        self._init_trace(lam)

    def learn(self, model, s, cat):
        self.calls += 1
        beta = getattr(model, "beta", None)
        if beta is None or s is None or self.mu == 0.0:
            return 0.0
        cat = int(cat)
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32, device=beta.device).reshape(-1)
            e = self._advance(model, s)
            if cat >= int(beta.shape[1]):
                self.widen(model, cat + 1)
                self.widened += 1
                beta = model.beta
            lr = self.sign * self.mu / (float(e @ e) + 1.0)
            p = torch.softmax(s @ beta, dim=0)
            p[cat] -= 1.0
            dbeta = -lr * torch.outer(e, p)
            beta += dbeta
            d = float(dbeta.abs().sum())
        if d > 0.0:
            self.steps += 1
        return d

    def state(self):
        return {**OnlineReadout.state(self), "lambda": self.lam, "trace_resets": self.resets}


class DeltaSched(_Eligibility, OnlineReadout):
    """The schedule control (record 2.3, Codex #10): delta's direction at the trace's scalar step."""

    def __init__(self, lam, mu=MU, sign=1.0):
        OnlineReadout.__init__(self, "delta", mu=mu, sign=sign)
        self._init_trace(lam)
        self.max_match_err = 0.0
        self.silent_events = 0  # ||s|| = 0: a_t = 0 by definition, no matching check

    def learn(self, model, s, cat):
        self.calls += 1
        beta = getattr(model, "beta", None)
        if beta is None or s is None or self.mu == 0.0:
            return 0.0
        cat = int(cat)
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float32, device=beta.device).reshape(-1)
            e = self._advance(model, s)
            if cat >= int(beta.shape[1]):
                self.widen(model, cat + 1)
                self.widened += 1
                beta = model.beta
            c_e = self.mu / (float(e @ e) + 1.0)
            s_norm, e_norm = float(s.norm()), float(e.norm())
            if s_norm == 0.0:
                a_t = 0.0
                self.silent_events += 1
            else:
                a_t = c_e * e_norm / s_norm
                want = c_e * e_norm
                if want > 0.0:
                    err = abs(a_t * s_norm - want) / want
                    self.max_match_err = max(self.max_match_err, err)
                    if err > SCHED_TOL:
                        raise RuntimeError(f"DeltaSched matching error {err:.3e} > {SCHED_TOL}")
            p = torch.softmax(s @ beta, dim=0)
            p[cat] -= 1.0
            dbeta = -(self.sign * a_t) * torch.outer(s, p)
            beta += dbeta
            d = float(dbeta.abs().sum())
        if d > 0.0:
            self.steps += 1
        return d


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


def _config_accs(seeds, kind, noise, lam):
    tr, de, sc = [], [], []
    for s in seeds:
        chain = planted_chain(s, kind, noise)
        tr.append(run_planted(s, TraceDelta(lam), chain).state()["live_accuracy"])
        de.append(run_planted(s, OnlineReadout("delta"), chain).state()["live_accuracy"])
        sc.append(run_planted(s, DeltaSched(lam), chain).state()["live_accuracy"])
    return tr, de, sc


def reachability_gate(discovery=DISCOVERY, replication=REPLICATION, family=GATE_FAMILY, log=print):
    """Record 2.7a. Returns {'pa1_met', 'chosen', 'rows'}; prints every contrast."""
    torch.set_num_threads(1)
    rows = []
    for kind, noise, lam in family:
        tr, de, sc = _config_accs(discovery, kind, noise, lam)
        tag = f"{kind} noise {noise} lambda {lam:.3f}"
        met, c, x = reading(f"TRACE - DELTA  [{tag}] disc", [a - b for a, b in zip(tr, de)])
        _, c2, x2 = reading(f"TRACE - DELTA_SCHED  [{tag}] disc", [a - b for a, b in zip(tr, sc)])
        log(_line(c.name, met, c, x))
        log(_line(c2.name, c2.t > T_BAR and not c2.degenerate, c2, x2) + "  (printed, not a bar)")
        row = {"kind": kind, "noise": noise, "lam": lam, "disc_met": met, "disc_mean": x["mean"]}
        if met:
            rtr, rde, rsc = _config_accs(replication, kind, noise, lam)
            rmet, rc, rx = reading(
                f"TRACE - DELTA  [{tag}] REPL", [a - b for a, b in zip(rtr, rde)]
            )
            _, rc2, rx2 = reading(
                f"TRACE - DELTA_SCHED  [{tag}] REPL", [a - b for a, b in zip(rtr, rsc)]
            )
            log(_line(rc.name, rmet, rc, rx))
            log(
                _line(rc2.name, rc2.t > T_BAR and not rc2.degenerate, rc2, rx2)
                + "  (printed, not a bar)"
            )
            row.update(repl_met=rmet, repl_mean=rx["mean"])
        rows.append(row)
    replicated = [r for r in rows if r.get("repl_met")]
    chosen = max(replicated, key=lambda r: r["repl_mean"]) if replicated else None
    m = len(family)
    log(
        f"  false-pass references (record 2.7a; references, not guarantees): independence reference "
        f"1-(1-{ALPHA})^{m} = {1 - (1 - ALPHA) ** m:.2f}; union bound {m}x{ALPHA} = {m * ALPHA:.2f}; "
        f"replicated <= {m}x{ALPHA}^2 = {m * ALPHA**2:.3f} (needs independent discovery/replication data)"
    )
    met_disc = sum(1 for r in rows if r["disc_met"])
    log(
        f"  PA1 {'MET' if chosen else 'NOT MET'}: {met_disc}/{m} met on discovery, {len(replicated)} "
        "replicated"
        + (
            f"; chosen {chosen['kind']} noise {chosen['noise']} lambda {chosen['lam']:.3f}"
            if chosen
            else " -- Part A is DELETED before any in-brain cell (record 2.7a, P8 F11)"
        )
    )
    return {"pa1_met": chosen is not None, "chosen": chosen, "rows": rows}


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the cheap reds (never builds a Brain, never opens a lake)
# --------------------------------------------------------------------------------------------------
def _short(seed, readout, n=200):
    return run_planted(seed, readout, planted_chain(seed, "burst", 0.5, n=n))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    torch.set_num_threads(1)
    if "--gate" in argv:
        reachability_gate()
        return
    # PA5 reach: lambda = 0 is the incumbent, bit for bit (beta, W, live accuracy)
    inc, t0 = _short(20261003, OnlineReadout("delta")), _short(20261003, TraceDelta(0.0))
    assert torch.equal(inc._model.beta, t0._model.beta), "lambda = 0 is not the incumbent"
    assert torch.equal(inc._model.W, t0._model.W)
    assert inc.state()["live_accuracy"] == t0.state()["live_accuracy"]
    # ... and lambda > 0 does move something (the reach is not vacuous)
    t1 = _short(20261003, TraceDelta(LAMBDAS[1]))
    assert not torch.equal(inc._model.beta, t1._model.beta), "the trace never moved beta"
    # PA6 lifecycle: a model swap zeroes e; a row mismatch raises
    tr = TraceDelta(0.5)

    class _M:
        def __init__(self, rows, m=3):
            self.beta = torch.zeros(rows, m)
            self.m = m

    a, b = _M(4), _M(4)
    tr.learn(a, torch.ones(4), 1)
    tr.learn(a, torch.ones(4), 1)
    assert tr.resets == 0 and torch.allclose(tr._e, torch.full((4,), 0.75))
    tr.learn(b, torch.ones(4), 1)
    assert tr.resets == 1 and torch.allclose(tr._e, torch.full((4,), 0.5)), "a swap must zero e"
    try:
        tr.learn(b, torch.ones(5), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a row mismatch on the same model must raise, never broadcast")
    # PA7 determinism
    d1, d2 = _short(20261003, TraceDelta(LAMBDAS[0])), _short(20261003, TraceDelta(LAMBDAS[0]))
    assert torch.equal(d1._model.beta, d2._model.beta)
    # the generator: deterministic, the three kinds differ, iid has no run structure
    assert planted_chain(20261003, "drift", 1.0, n=60) == planted_chain(
        20261003, "drift", 1.0, n=60
    )
    burst = [c for _, c in planted_chain(20261003, "burst", 1.0, n=100)]
    iid = [c for _, c in planted_chain(20261003, "iid", 1.0, n=100)]
    assert all(burst[i] == burst[i - i % 20] for i in range(100)), "burst runs are fixed length"
    assert sum(iid[i] == iid[i - 1] for i in range(1, 100)) < 60, "iid must not be run-structured"
    # DeltaSched's in-arm matching holds and is exercised
    ds = DeltaSched(LAMBDAS[1])
    _short(20261003, ds)
    assert ds.steps > 0 and ds.max_match_err <= SCHED_TOL
    # the reading can say NOT MET: degenerate, too few non-tied, below the bar
    assert not reading("deg", [0.1] * 8)[0], "ruling 9: degenerate reads NOT MET"
    assert not reading("ties", [0.3, 0.2, 0.25, 0.4, 0.0, 0.0, 0.0, 0.0])[0], "4 non-tied < 5"
    assert not reading("noise", [0.1, -0.1, 0.12, -0.11, 0.09, -0.1, 0.1, -0.12])[0]
    assert reading("up", [0.2, 0.3, 0.25, 0.4, 0.15, 0.35, 0.3, 0.2])[0]
    print(
        "g5_readout_cells self-check OK: lambda 0 == incumbent bit for bit, lambda > 0 moves beta; a model "
        "swap zeroes e, a row mismatch raises; deterministic; the frozen generator is deterministic and iid "
        f"is not run-structured; DeltaSched matched to {ds.max_match_err:.1e}; the reading says NOT MET on "
        "degenerate, mostly-tied and sign-balanced samples"
    )


if __name__ == "__main__":
    main()
