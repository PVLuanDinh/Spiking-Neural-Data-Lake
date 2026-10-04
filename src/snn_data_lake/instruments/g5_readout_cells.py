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

PART B'S PLANTED CHECKS (record 3.5, PB1 / PB2 / PB5), brain-free on a SYNTHETIC readout (`planted_readout`:
class probabilities q from a planted error process, no engine), each contrast over 8 seeds:
  PB1 (positive; 11924-11931) a cyclic relabelling 0->1->2->0 of uniform truths: contrast(COUPLED - uncoupled)
      MET AND contrast(COUPLED - COUPLED_MARG) MET (equal marginals: the diagonal control cannot move it).
  PB2 (negative; 11924-11931) non-systematic errors: contrast(COUPLED - uncoupled) NOT met.
  PB5 (the marginal control's own positive; 11944-11951) a pure base-rate miscalibration (skewed truths, a
      logit bias toward one class): contrast(COUPLED_MARG - uncoupled) MET decides; contrast(COUPLED -
      COUPLED_MARG) NOT met is the secondary conjunct.
The fixture parameters (PLANT_*) are fixed in this file BEFORE the registered seeds are run, once.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.g5_readout_cells [--planted]
"""

import math
import sys

import torch

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast
from snn_data_lake.readout.categorical_coupling import CategoricalCoupling
from snn_data_lake.readout.online_readout import ClosingGate, OnlineReadout

TIE_TOL = 1e-6  # |d| <= this is a tie (L1c's harness value)
MIN_NONTIED = 5  # the separate named restriction (record 2.4)
CHAIN_KINDS = ("burst", "drift", "iid")
PB12_SEEDS = tuple(range(11924, 11932))
PB5_SEEDS = tuple(range(11944, 11952))
# the synthetic-readout fixture, fixed before the first registered run (CHOSEN; never tuned after a look)
PLANT_N, PLANT_K = 400, 4
PLANT_STRENGTH = 2.0  # logit of the readout's chosen class, beside N(0, 1) logit noise
PLANT_HIT = 0.85  # PB1: the planted relabelling's rate; PB2: the readout's accuracy
PLANT_PRIOR = (0.55, 0.15, 0.15, 0.15)  # PB5: skewed truths
PLANT_BIAS = 1.0  # PB5: the logit bias toward class k - 1
PLANTS = ("relabel", "iid_errors", "prior")


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
# Part B's planted checks (record 3.5)
# --------------------------------------------------------------------------------------------------
# distinct-from: readout/snn_moe_classifier.py -- `_new_rng` seeds a numpy Generator from `random`;
#   no planted readout. Read 2026-10-04.
# distinct-from: faculties/interoceptive_integration.py -- `record_charge` is a port A/B. Read 2026-10-04.
# distinct-from: infra/cockpit.py -- `_http_get` drives a handler without a socket. Read 2026-10-04.
# distinct-from: instruments/everyday_battery.py -- `spawn_seeds` runs one process per seed. Read 2026-10-04.
# distinct-from: instruments/report_order_probe.py -- `t081_constructed_read` reads a constructed recording;
#   none synthesizes a readout's class probabilities under a planted error process. Read 2026-10-04.
def planted_readout(seed, plant, n=PLANT_N, k=PLANT_K):
    """[(q, truth)] from a synthetic readout with a planted error process (one torch.Generator).
    relabel: the readout names (truth + 1) % 3 for truths 0-2 (k - 1 is answered right) at rate
    PLANT_HIT, else a uniform class; iid_errors: right at rate PLANT_HIT, else a uniform OTHER class;
    prior: truths drawn from PLANT_PRIOR, logits = STRENGTH * onehot(truth) + BIAS on class k - 1."""
    if plant not in PLANTS:
        raise ValueError(f"plant must be one of {PLANTS}, got {plant!r}")
    g = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(n):
        noise = torch.randn(k, generator=g)
        if plant == "prior":
            t = int(torch.multinomial(torch.tensor(PLANT_PRIOR), 1, generator=g))
            logits = noise
            logits[t] += PLANT_STRENGTH
            logits[k - 1] += PLANT_BIAS
        else:
            t = int(torch.randint(0, k, (1,), generator=g))
            hit = float(torch.rand(1, generator=g)) < PLANT_HIT
            if plant == "relabel":
                said = (
                    ((t + 1) % 3 if t < 3 else t)
                    if hit
                    else int(torch.randint(0, k, (1,), generator=g))
                )
            else:
                other = int(torch.randint(0, k - 1, (1,), generator=g))
                said = t if hit else (other if other < t else other + 1)
            logits = noise
            logits[said] += PLANT_STRENGTH
        out.append((torch.softmax(logits, dim=0), t))
    return out


def coupled_accuracies(stream, k=PLANT_K):
    """Prequential accuracy of the uncoupled argmax, COUPLED ("full") and COUPLED_MARG ("marg") on one
    stream. Counts take the READOUT's own argmax as the prediction (record 3.2), after scoring."""
    full, marg = CategoricalCoupling(k, "full"), CategoricalCoupling(k, "marg")
    hits = {"uncoupled": 0, "full": 0, "marg": 0}
    for q, t in stream:
        said = int(q.argmax())
        hits["uncoupled"] += int(said == t)
        hits["full"] += int(full.decide(q) == t)
        hits["marg"] += int(marg.decide(q) == t)
        full.update(said, t)
        marg.update(said, t)
    n = len(stream)
    return {a: h / n for a, h in hits.items()}


def planted_checks(pb12=PB12_SEEDS, pb5=PB5_SEEDS, log=print):
    """Record 3.5: PB1, PB2 and PB5 on their registered seeds. Returns {name: bool}."""

    def contrasts(seeds, plant, pairs):
        accs = [coupled_accuracies(planted_readout(s, plant)) for s in seeds]
        res = {}
        for a, b in pairs:
            met, c, x = reading(f"{plant}: {a} - {b}", [r[a] - r[b] for r in accs])
            log(_line(c.name, met, c, x))
            res[(a, b)] = met
        return res

    r1 = contrasts(pb12, "relabel", [("full", "uncoupled"), ("full", "marg")])
    r2 = contrasts(pb12, "iid_errors", [("full", "uncoupled")])
    r5 = contrasts(pb5, "prior", [("marg", "uncoupled"), ("full", "marg")])
    out = {
        "PB1": r1[("full", "uncoupled")] and r1[("full", "marg")],
        "PB2": not r2[("full", "uncoupled")],
        "PB5": r5[("marg", "uncoupled")] and not r5[("full", "marg")],
    }
    log("  " + "; ".join(f"{k} {'MET' if v else 'NOT MET'}" for k, v in out.items()))
    return out


# --------------------------------------------------------------------------------------------------
# the cells: arms wired onto a live engine (record 2.3 / 3.3; common vocabulary init, Codex #8)
# --------------------------------------------------------------------------------------------------
VACUITY_FLOOR = 0.20  # record 2.3 (CHOSEN): a shift that moves fewer teaching labels is VACUOUS
CELL_TICKS = 800  # the L1c harness shape
DEV_SEEDS = tuple(range(11000, 11008))
CONF_SEEDS = tuple(range(11100, 11108))
RESERVE_SEEDS = tuple(range(11108, 11116))
PA3_SEEDS = tuple(range(11916, 11924))
B_ARMS = ("BATCH", "DELTA", "COUPLED", "COUPLED0", "COUPLED_MARG", "COUPLED_SHUF", "COUPLED_PERM")
DEV_ARMS = ("BATCH", "DELTA", "COUPLED", "COUPLED0", "COUPLED_MARG")
SAME_TOL = 1e-6


class SameRowsHalt(RuntimeError):
    """The arms of one seed did not see the same rows: no contrast may be read (record 2.3)."""


def circular_shift(seq):
    """The shuffled teaching signal (record 2.3): seq circularly shifted by floor(n / 2) events."""
    k = len(seq) // 2
    return list(seq[k:]) + list(seq[:k])


def shift_fraction(seq, shifted):
    """f: the fraction of events whose label the shift changed (VACUOUS if f < VACUITY_FLOOR)."""
    n = len(seq)
    return sum(1 for a, b in zip(seq, shifted) if a != b) / n if n else 0.0


class _Taught:
    """A readout taught a substituted label sequence by EVENT index; scoring keeps the true label."""

    def __init__(self, readout, labels, index):
        self.readout, self.labels, self._index = readout, list(labels), index

    @property
    def widened(self):
        return self.readout.widened

    def learn(self, model, s, cat):
        k = self._index()
        return self.readout.learn(model, s, self.labels[k] if k < len(self.labels) else cat)

    def state(self):
        return self.readout.state()


def instrument_engine(eng, readout, M=None, coupling=None, count_labels=None, teach_labels=None):
    """Wire one arm onto a NeurogenesisSelfImprove. Returns its record: the true categories, the
    column count at every scoring (the vocabulary schedule) and the readout. M: the common vocabulary
    -- every new model object is widened to M columns before its first scoring (None: the deployed
    lazy widen). coupling: the live decision is re-read through it on non-silent rows, and it counts
    (the readout's own decision, count_labels[k] or the truth) AFTER scoring."""
    rec = {"cats": [], "cols": [], "k": 0}
    if teach_labels is not None and readout is not None:
        readout = _Taught(readout, teach_labels, lambda: rec["k"])
    eng.online_readout = readout
    rec["readout"] = readout
    if M is not None:
        real_refit, last = eng._refit, [None]

        def _refit():
            out = real_refit()
            m = eng._model
            if m is not None and m is not last[0] and getattr(m, "beta", None) is not None:
                OnlineReadout.widen(m, int(M))
                last[0] = m
            return out

        eng._refit = _refit
    real_predict, real_observe, said = eng.predict_next, eng.observe, {}
    if coupling is not None:

        def predict_next():
            d = real_predict()
            if d is None:
                return d
            k, m = rec["k"], eng._model
            if said.get("k") == k and said.get("model") is m:
                return said["pick"]
            pend = eng._pending_prediction
            pick, decided = d, False
            if not pend[4]:  # a silent row keeps NGSNN.decide's fallback (record 3.2)
                pick, decided = coupling.decide(torch.softmax(pend[2] @ m.beta, dim=0)), True
            said.update(k=k, model=m, said=d, pick=pick, decided=decided)
            return pick

        eng.predict_next = predict_next

    def observe(features, category):
        k, cat, m = rec["k"], int(category), eng._model
        if m is not None and getattr(m, "beta", None) is not None and eng._prev is not None:
            rec["cols"].append(int(m.beta.shape[1]))
        out = real_observe(features, category)
        if coupling is not None and said.get("k") == k and said.get("decided"):
            coupling.update(said["said"], cat if count_labels is None else count_labels[k])
        rec["cats"].append(cat)
        rec["k"] = k + 1
        return out

    eng.observe = observe
    return rec


def arm_parts(arm, M, seq, seed):
    """(readout, coupling, count_labels, teach_labels) for one arm on one seed's recorded sequence."""
    from snn_data_lake.readout.categorical_coupling import CategoricalCoupling as CC

    if arm == "BATCH":
        return None, None, None, None
    delta = OnlineReadout("delta")
    if arm in ("RECORD", "DELTA"):
        return delta, None, None, None
    if arm == "COUPLED":
        return delta, CC(M, "full"), None, None
    if arm == "COUPLED0":
        return delta, CC(M, "full", frozen=True), None, None
    if arm == "COUPLED_MARG":
        return delta, CC(M, "marg"), None, None
    if arm == "COUPLED_SHUF":
        return delta, CC(M, "full"), circular_shift(seq), None
    if arm == "COUPLED_PERM":
        perm = torch.randperm(M, generator=torch.Generator().manual_seed(int(seed))).tolist()
        return delta, CC(M, "full"), [perm[c] for c in seq], None
    raise ValueError(f"unknown arm {arm!r}")


def _digest(seq):
    import hashlib

    return hashlib.sha256(",".join(str(int(v)) for v in seq).encode()).hexdigest()[:16]


def summarize(eng, rec, seed, arm):
    st = eng.state()
    ro = rec.get("readout")
    beta = None if eng._model is None else eng._model.beta
    return {
        "seed": seed,
        "arm": arm,
        "cats": list(rec["cats"]),
        "cats_hash": _digest(rec["cats"]),
        "cols_hash": _digest(rec["cols"]),
        "widened": 0 if ro is None else int(ro.widened),
        "live_acc": st.get("live_accuracy"),
        "live_pers": st.get("live_persistence"),
        "live_n": st.get("live_n"),
        "traj": [(int(t), int(u), float(a)) for t, u, a in eng.trajectory],
        "beta_hash": None if beta is None else _digest((beta * 1e6).round().flatten().tolist()),
        "units": eng.grown_units,
    }


def same_rows(cells):
    """Record 2.3's check, per seed across arms: identical category sequence, vocabulary schedule and
    live persistence floor; no readout widened inside a cell. Raises SameRowsHalt."""
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c["seed"], []).append(c)
    for seed, cs in by_seed.items():
        ref = cs[0]
        for c in cs:
            if c["widened"]:
                raise SameRowsHalt(
                    f"seed {seed} arm {c['arm']}: the readout widened {c['widened']}x"
                )
            for key in ("cats_hash", "cols_hash"):
                if c[key] != ref[key]:
                    raise SameRowsHalt(f"seed {seed}: {key} {c['arm']} != {ref['arm']}")
            if abs((c["live_pers"] or 0.0) - (ref["live_pers"] or 0.0)) > SAME_TOL:
                raise SameRowsHalt(f"seed {seed}: live persistence {c['arm']} != {ref['arm']}")
    return True


def live_lake_stamp():
    """mtimes of the live home and its brain lake (record 2.6 PA9 / board PROTOCOL 4b)."""
    import os

    base = os.path.join(os.path.expanduser("~"), ".snn_data_lake")
    lake = os.path.join(base, "brain_lake.bin")
    return tuple(os.stat(p).st_mtime_ns if os.path.exists(p) else None for p in (base, lake))


def brain_cell(args):
    """One in-brain cell in a worker process: make_brain(seed) under throwaway_home, the arm wired onto
    its engine, CELL_TICKS ticks. args = (seed, arm, recorded sequence or None for the RECORD stage).
    Until R1's shared guard lands (record 10 row 8) the home is checked to be a fresh temp dir."""
    import os
    import tempfile

    seed, arm, seq = args
    torch.set_num_threads(1)
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home() as h:
        tmp = os.path.realpath(tempfile.gettempdir())
        if os.environ.get("NORD_HOME") != h or not os.path.realpath(h).startswith(tmp):
            raise RuntimeError(
                f"refused: NORD_HOME {os.environ.get('NORD_HOME')!r} is not a temp home"
            )
        from snn_data_lake.engine.brain import make_brain

        b = make_brain(seed=seed, persist=False)
        try:
            M = None if seq is None else 1 + max(seq)
            readout, coupling, count_labels, teach = arm_parts(arm, M, seq, seed)
            rec = instrument_engine(b.neuro, readout, M, coupling, count_labels, teach)
            for _ in range(CELL_TICKS):
                b.tick()
            out = summarize(b.neuro, rec, seed, arm)
            out["unified"] = bool(getattr(b, "_perception_unified", False))
            out["M"] = M
        finally:
            b.close()
    return out


def run_cells(seeds, arms, workers=4, log=print):
    """Stage 1 records each seed's sequence (RECORD, the deployed wiring); stage 2 runs every arm under
    the common vocabulary init on that sequence. The live lake's mtimes must not move."""
    from concurrent.futures import ProcessPoolExecutor

    before = live_lake_stamp()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recorded = {c["seed"]: c for c in ex.map(brain_cell, [(s, "RECORD", None) for s in seeds])}
        jobs = [(s, a, recorded[s]["cats"]) for s in seeds for a in arms]
        cells = list(ex.map(brain_cell, jobs))
    if live_lake_stamp() != before:
        raise RuntimeError(
            "the live lake's mtime moved during the cells -- halt (board PROTOCOL 4b)"
        )
    for c in cells:
        if c["cats_hash"] != recorded[c["seed"]]["cats_hash"]:
            raise SameRowsHalt(f"seed {c['seed']} arm {c['arm']}: rows differ from the recording")
    same_rows(cells)
    for s in seeds:
        seq = recorded[s]["cats"]
        f = shift_fraction(seq, circular_shift(seq))
        vac = "  VACUOUS" if f < VACUITY_FLOOR else ""
        log(f"  seed {s}: M {1 + max(seq)}  events {len(seq)}  shift f {f:.3f}{vac}")
    return recorded, cells


def contrast_cells(cells, a, b, label=""):
    """reading(a - b) over the seeds, asserting each difference pairs ONE seed's two cells (Codex #13)."""
    by = {(c["seed"], c["arm"]): c for c in cells}
    diffs = []
    for s in sorted({c["seed"] for c in cells}):
        ca, cb = by[(s, a)], by[(s, b)]
        if not (ca["seed"] == cb["seed"] == s):
            raise SameRowsHalt(f"a paired difference crossed seeds at {s}")
        diffs.append(ca["live_acc"] - cb["live_acc"])
    return reading(f"{label}{a} - {b}", diffs)


# --------------------------------------------------------------------------------------------------
# the brain-free reds of the cell machinery (record 2.6: PA3, PA4, PA8, PA10)
# --------------------------------------------------------------------------------------------------
def oracle_chain(seed, n=600):
    """PA3's oracle input: the percept is one-hot(category), so the engine's pair is
    one-hot(c_{t-1}) -> c_t, on the frozen burst chain's categories (noise 2.0)."""
    out = []
    for _, c in planted_chain(seed, "burst", 2.0, n=n):
        oh = [0.0] * 64
        oh[c] = 1.0
        out.append((oh, c))
    return out


def run_instrumented(seed, chain, readout, M=None, teach_labels=None):
    """A planted chain through the real engine under the closing gate, one arm wired by
    instrument_engine; returns `summarize`'s record."""
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove

    eng = NeurogenesisSelfImprove(
        seed=seed, device="cpu", refit_every=1, online_readout=None, growth_gate=ClosingGate(3)
    )
    rec = instrument_engine(eng, readout, M, teach_labels=teach_labels)
    for i, (x, c) in enumerate(chain):
        eng.observe(x, c)
        if i % 20 == 19:
            eng.tick()
    return summarize(eng, rec, seed, "")


def pa3(seeds=PA3_SEEDS, log=print):
    """The shift's positive control: ORACLE - BATCH MET and ORACLE - ORACLE_SHUF MET (deciding); the
    secondary ORACLE_SHUF - BATCH NOT met can only add a refusal."""
    rows = []
    for s in seeds:
        ch = oracle_chain(s)
        seq = [c for _, c in ch]
        M = 1 + max(seq)
        shuf = circular_shift(seq)
        rows.append(
            {
                "BATCH": run_instrumented(s, ch, None, M)["live_acc"],
                "ORACLE": run_instrumented(s, ch, OnlineReadout("delta"), M)["live_acc"],
                "ORACLE_SHUF": run_instrumented(s, ch, OnlineReadout("delta"), M, shuf)["live_acc"],
            }
        )
    res = {}
    for a, b in (("ORACLE", "BATCH"), ("ORACLE", "ORACLE_SHUF"), ("ORACLE_SHUF", "BATCH")):
        met, c, x = reading(f"PA3 {a} - {b}", [r[a] - r[b] for r in rows])
        log(_line(c.name, met, c, x))
        res[(a, b)] = met
    ok = res[("ORACLE", "BATCH")] and res[("ORACLE", "ORACLE_SHUF")]
    ok = ok and not res[("ORACLE_SHUF", "BATCH")]
    log(f"  PA3 {'MET' if ok else 'NOT MET'}")
    return ok


def late_category_chain(n=400):
    """PA10's fixture: categories 0-2 in runs of 20 for the first half; category 3 first appears in the
    second half, so a circular shift teaches it EARLY. One-hot percepts (deterministic, no RNG)."""
    cats = [(i // 20) % 3 if i < n // 2 else (3 if (i // 20) % 2 == 0 else 0) for i in range(n)]
    return [([1.0 if j == c else 0.0 for j in range(64)], c) for c in cats]


def pa10():
    """Codex #8's red: an early-category mutation without the common init must HALT the same-rows
    check; with it, the same construction must pass (widened 0, identical schedule)."""
    ch = late_category_chain()
    seq = [c for _, c in ch]
    shuf = circular_shift(seq)
    lazy = [
        run_instrumented(20261003, ch, OnlineReadout("delta")),
        run_instrumented(20261003, ch, OnlineReadout("delta"), None, shuf),
    ]
    lazy[0]["arm"], lazy[1]["arm"] = "DELTA", "DELTA_SHUF"
    try:
        same_rows(lazy)
    except SameRowsHalt:
        halted = True
    else:
        halted = False
    M = 1 + max(seq)
    common = [
        run_instrumented(20261003, ch, OnlineReadout("delta"), M),
        run_instrumented(20261003, ch, OnlineReadout("delta"), M, shuf),
    ]
    common[0]["arm"], common[1]["arm"] = "DELTA", "DELTA_SHUF"
    same_rows(common)  # must not raise
    return halted


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the cheap reds (never builds a Brain, never opens a lake)
# --------------------------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    torch.set_num_threads(1)
    if "--planted" in argv:
        planted_checks()
        return
    if "--pa3" in argv:
        pa3()
        return
    if "--dev" in argv:  # in-brain: run only on the lead's go (board rule)
        _, cells = run_cells(DEV_SEEDS, DEV_ARMS)
        for a, b in (("DELTA", "BATCH"), ("COUPLED", "DELTA"), ("COUPLED_MARG", "DELTA")):
            met, c, x = contrast_cells(cells, a, b, "DEV ")
            print(_line(c.name, met, c, x))
        return
    # PA4: the vacuity detector fires on a known positive (a one-category sequence reads f = 0)
    assert shift_fraction([2] * 40, circular_shift([2] * 40)) == 0.0 < VACUITY_FLOOR
    assert shift_fraction([0] * 20 + [1] * 20, circular_shift([0] * 20 + [1] * 20)) == 1.0
    # PA8: a planted one-label perturbation in one arm's rows must halt the same-rows check
    row = {
        "seed": 1,
        "arm": "A",
        "widened": 0,
        "cats_hash": "x",
        "cols_hash": "y",
        "live_pers": 0.5,
    }
    try:
        same_rows([row, {**row, "arm": "B", "cats_hash": "x2"}])
    except SameRowsHalt:
        pass
    else:
        raise AssertionError("PA8: a perturbed label must halt the same-rows check")
    same_rows([row, {**row, "arm": "B"}])
    # PA10 (Codex #8): an early category without the common init halts; with it, it passes
    assert pa10(), "PA10: the early-category mutation did not halt the same-rows check"
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
    # Part B's synthetic readout (fixture seed only; the registered PB seeds run under --planted)
    rel = coupled_accuracies(planted_readout(20261003, "relabel"))
    assert rel["full"] > rel["uncoupled"] + 0.2 and rel["full"] > rel["marg"] + 0.2, rel
    pri = coupled_accuracies(planted_readout(20261003, "prior"))
    assert pri["marg"] > pri["uncoupled"], pri
    assert (
        planted_readout(20261003, "iid_errors", n=50)[0][1]
        == planted_readout(20261003, "iid_errors", n=50)[0][1]
    )
    print(
        "g5_readout_cells self-check OK: the runner is deterministic and the online readout moves; the frozen "
        "generator is deterministic and iid is not run-structured; the reading says NOT MET on degenerate, "
        "mostly-tied and sign-balanced samples; the planted relabelling is recovered by full coupling and not by "
        "the marginal control, which does correct a planted base-rate miscalibration; PA4 vacuity, PA8 "
        "same-rows halt and PA10 early-category halt (with the common init passing) all fire"
    )


if __name__ == "__main__":
    main()
