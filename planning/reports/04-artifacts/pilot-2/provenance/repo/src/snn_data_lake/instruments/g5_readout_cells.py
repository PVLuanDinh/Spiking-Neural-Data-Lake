"""G5 / T113 -- the readout-variant cells (specs/022 evidence/g5-readout-variants.md, frozen at 500d3f58).

ONE instrument for the row's in-brain and planted cells. PART A IS DELETED: its 2.7a reachability gate read
PA1 NOT MET -- 1 of 12 frozen configurations met on discovery (11900-11907), 0 replicated (11936-11943) --
so the trace-delta challenger, its schedule control and the gate were removed before any in-brain cell
(P8 F11: a criterion with no reachable red is deleted, not weakened). PART B IS DELETED: on the confirmatory
seeds 11100-11107 B-R1 read NOT MET (COUPLED - DELTA mean +0.0021, t +0.444), so `categorical_coupling`, its
arms, its planted checks (PB1/PB2/PB5) and the Part B reading were removed (record 3.6; P8 deletion on a
null). The record's results sections carry every contrast verbatim; commits 4ecc90d9 (Part A) and 0fba900c
(Part B's reading) keep the code as it was run. `OnlineReadout("delta")` stays the incumbent.

WHAT STAYS (the record's deletion scope): the frozen generator `planted_chain` (record 2.6, Codex #11), the
brain-free engine runner `run_planted` under the closing growth gate L1c's reds use, the reading (record
2.4), and the in-brain cell machinery (`run_cells`, `brain_cell`, `same_rows`, the shift's positive control
PA3, PA4/PA8/PA10) that Part D reuses.

THE STATISTIC (record 2.4): `paired_contrast.contrast` at T_BAR; MET iff non-degenerate (ruling 9: a
zero-variance contrast reads NOT MET) AND t > T_BAR AND >= 5 seeds non-tied (|d| > 1e-6, a separate named
restriction). The exact FPR is a DIAGNOSTIC, printed, gating nothing (Codex #9).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.g5_readout_cells [--pa3]
"""

import math
import os
import sys

import torch

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast
from snn_data_lake.readout.online_readout import ClosingGate, OnlineReadout
from snn_data_lake.substrate.home import _default_live_home, lake_snapshot, live_lake_untouched

#: The live home the cells must never touch (record 2.6 PA9 / board PROTOCOL 4b); a test seam.
LIVE_LAKE = _default_live_home()

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_Taught.learn": (
        "NOT_A_RULE",
        "a delegation wrapper: it swaps the teaching label for the event's substituted one (the shifted-"
        "teaching control) and calls the wrapped readout's own learn; it changes no weight and computes "
        "no error or update itself. The rule that runs is OnlineReadout.learn, declared LOCAL there",
    ),
}

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
# the cells: arms wired onto a live engine (record 2.3 / 3.3; common vocabulary init, Codex #8)
# --------------------------------------------------------------------------------------------------
VACUITY_FLOOR = 0.20  # record 2.3 (CHOSEN): a shift that moves fewer teaching labels is VACUOUS
CELL_TICKS = 800  # the L1c harness shape
PA3_SEEDS = tuple(range(11916, 11924))
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


def instrument_engine(eng, readout, M=None, teach_labels=None):
    """Wire one arm onto a NeurogenesisSelfImprove. Returns its record: the true categories, the
    column count at every scoring (the vocabulary schedule) and the readout. M: the common vocabulary
    -- every new model object is widened to M columns before its first scoring (None: the deployed
    lazy widen)."""
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
    real_observe = eng.observe

    def observe(features, category):
        k, cat, m = rec["k"], int(category), eng._model
        if m is not None and getattr(m, "beta", None) is not None and eng._prev is not None:
            rec["cols"].append(int(m.beta.shape[1]))
        out = real_observe(features, category)  # the live instrument scores the event here
        rec["cats"].append(cat)
        rec["k"] = k + 1
        return out

    eng.observe = observe
    return rec


def arm_parts(arm, M, seq, seed):
    """(readout, teach_labels) for one arm on one seed's recorded sequence. Only the incumbent arms remain:
    Part A's trace and Part B's coupling arms were deleted on their nulls."""
    if arm == "BATCH":
        return None, None
    if arm in ("RECORD", "DELTA"):
        return OnlineReadout("delta"), None
    raise ValueError(f"unknown arm {arm!r}")


def tensor_digest(t):
    """The EXACT identity of a tensor (Codex cp4 #11): sha256 over dtype, shape and its raw bytes --
    never a rounded copy, which a sub-rounding perturbation would pass."""
    import hashlib

    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


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
        # the held-out slice's majority and persistence floors per refit (record 2.3; Codex cp4 #12)
        "floors": [(float(m), float(p)) for _, m, p in eng.baselines],
        "beta_hash": None if beta is None else tensor_digest(beta),
        "units": eng.grown_units,
    }


def same_rows(cells):
    """Record 2.3's check, per seed across arms: identical category sequence, vocabulary schedule, live
    persistence floor and every refit's majority and persistence floors (Codex cp4 #12); no readout
    widened inside a cell. Raises SameRowsHalt."""
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
            if (
                not abs((c["live_pers"] or 0.0) - (ref["live_pers"] or 0.0)) <= SAME_TOL
            ):  # NaN refuses
                raise SameRowsHalt(f"seed {seed}: live persistence {c['arm']} != {ref['arm']}")
            fa, fr = c.get("floors", []), ref.get("floors", [])
            if len(fa) != len(fr) or any(
                not abs(x - y) <= SAME_TOL for a, b in zip(fa, fr) for x, y in zip(a, b)
            ):
                raise SameRowsHalt(f"seed {seed}: refit floors {c['arm']} != {ref['arm']}")
    return True


def arm_table(cells, log=print):
    """Record 2.4 / 3.4's printed context: per arm, mean live accuracy, mean live persistence floor and
    their difference -- a contrast that only approaches the floor is read against it (Codex cp4 #12)."""
    arms = sorted({c["arm"] for c in cells})
    for a in arms:
        cs = [c for c in cells if c["arm"] == a]
        acc = sum(c["live_acc"] for c in cs) / len(cs)
        pers = sum(c["live_pers"] for c in cs) / len(cs)
        log(
            f"  {a:<14} live acc {acc:.4f}  live persistence {pers:.4f}  acc - floor {acc - pers:+.4f}"
        )


def check_fresh_home(fresh):
    """Codex cp4 #7 / record 2.6 PA9: NORD_HOME must BE the home throwaway_home yielded, not merely any
    non-live directory (a populated B swapped in for the fresh A would feed a Brain B's state)."""
    from snn_data_lake.substrate.home import LiveHomeRefused, require_isolated_home

    got = require_isolated_home()
    if os.path.normcase(got) != os.path.normcase(fresh):
        raise LiveHomeRefused(
            f"home refused (not the fresh one): NORD_HOME {got} != yielded {fresh}"
        )
    return got


def brain_cell(args):
    """One in-brain cell in a worker process: make_brain(seed) under throwaway_home, the arm wired onto
    its engine, CELL_TICKS ticks. args = (seed, arm, recorded sequence or None for the RECORD stage).
    The home is THE shared live-home guard (record 10 row 8; R1, lead ruling 16): throwaway_home
    resolves and checks its temp dir, and require_isolated_home() re-checks NORD_HOME before any
    lake-binding import; a refusal raises home.LiveHomeRefused and no Brain is built."""
    seed, arm, seq = args
    torch.set_num_threads(1)
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home() as fresh:
        check_fresh_home(fresh)
        from snn_data_lake.engine.brain import make_brain

        b = make_brain(seed=seed, persist=False)
        try:
            M = None if seq is None else 1 + max(seq)
            readout, teach = arm_parts(arm, M, seq, seed)
            rec = instrument_engine(b.neuro, readout, M, teach)
            for _ in range(CELL_TICKS):
                b.tick()
            out = summarize(b.neuro, rec, seed, arm)
            out["unified"] = bool(getattr(b, "_perception_unified", False))
            out["M"] = M
        finally:
            b.close()
    return out


class LiveLakeMoved(RuntimeError):
    """The live lake changed while cells ran: a HALT, never a replaceable instrument failure."""


def _lake_unmoved(before, stage):
    """The one shared check (home.live_lake_untouched: content digest, an unreadable file reads as changed).
    COST: ~6.4 s per call on the 3.3 GB live lake; run_cells snapshots once up front, then checks twice
    per seed (after RECORD, after the arm stage), and once more only on the way out of an exception."""
    ok, changed = live_lake_untouched(LIVE_LAKE, before)
    if not ok:
        raise LiveLakeMoved(
            f"the live lake changed during {stage} ({len(changed)} file(s), e.g. {changed[:2]}) -- halt "
            "(board PROTOCOL 4b; Codex cp4 #13)"
        )


class InstrumentFailure(RuntimeError):
    """An instrument failure (record sec. 4): `unified` False, or rows that differ from the recording. A
    seed that fails this way is replaced WHOLE from the reserve; a result never triggers it."""


def _scored(c):
    """True iff the cell scored predictions: live_n > 0 and both live metrics present (Codex g5_b M3)."""
    return (
        bool(c.get("live_n")) and c.get("live_acc") is not None and c.get("live_pers") is not None
    )


def _run_seed(seed, arms, before, workers, pool, cell):
    """One seed's block: RECORD, then every arm on that sequence; both stages validated (Codex cp4 #9)."""
    with pool(max_workers=workers) as ex:
        rec = ex.submit(cell, (seed, "RECORD", None)).result()
        _lake_unmoved(before, "the RECORD stage")
        if not rec["unified"]:
            raise InstrumentFailure(
                f"seed {seed}: RECORD stage ran on fallback perception (unified False)"
            )
        cells = list(ex.map(cell, [(seed, a, rec["cats"]) for a in arms]))
        _lake_unmoved(before, "the arm stage")
    for c in [rec, *cells]:
        if not _scored(c):
            raise InstrumentFailure(f"seed {seed} arm {c['arm']}: scored no prediction")
    for c in cells:
        if not c["unified"]:
            raise InstrumentFailure(
                f"seed {seed} arm {c['arm']}: fallback perception (unified False)"
            )
        if c["cats_hash"] != rec["cats_hash"]:
            raise InstrumentFailure(f"seed {seed} arm {c['arm']}: rows differ from the recording")
    same_rows(cells)
    return rec, cells


def write_records(path, recorded, cells, replaced):
    """Persist one campaign's per-seed records as json + a sha256 sidecar (`path`.sha256) so a read can be
    re-derived without re-running spent seeds. Returns the digest of the json bytes."""
    import hashlib
    import json

    body = json.dumps(
        {"recorded": recorded, "cells": cells, "replaced": replaced}, sort_keys=True, indent=1
    ).encode()
    digest = hashlib.sha256(body).hexdigest()
    with open(path, "wb") as f:
        f.write(body)
    with open(path + ".sha256", "w") as f:
        f.write(digest)
    return digest


def load_records(path):
    """The records `write_records` wrote; ValueError if the json no longer matches its sha256 sidecar."""
    import hashlib
    import json

    with open(path, "rb") as f:
        body = f.read()
    with open(path + ".sha256") as f:
        want = f.read().strip()
    if hashlib.sha256(body).hexdigest() != want:
        raise ValueError(f"{path}: sha256 does not match its sidecar -- the records were altered")
    return json.loads(body)


def run_cells(
    seeds, arms, workers=4, log=print, pool=None, cell=None, reserve=(), record_path=None
):
    """Per seed: RECORD (the deployed wiring) records the sequence, then every arm runs under the common
    vocabulary init on it. The live lake's mtimes must not move: checked after EACH stage and on the way
    out of an exception (record 2.6 PA9; Codex cp4 #13). A seed whose block fails on the INSTRUMENT
    (crash, `unified` False, same-rows halt) is discarded WHOLE and replaced by the next reserve seed,
    printed beside it (record sec. 4's reserve rule; Codex cp4 #9); a live-lake or live-home violation is
    a halt, never replaced. `pool` / `cell` are test seams. `record_path` persists every seed's record
    and cell (json + sha256, `write_records`) before returning. Returns (recorded, cells, replacements)."""
    from concurrent.futures import ProcessPoolExecutor

    from snn_data_lake.substrate.home import LiveHomeRefused

    pool = pool or ProcessPoolExecutor
    cell = cell or brain_cell
    before = lake_snapshot(LIVE_LAKE)
    spare = list(reserve)
    recorded, cells, replaced = {}, [], []
    for seed in seeds:
        s = seed
        while True:
            try:
                rec, cs = _run_seed(s, arms, before, workers, pool, cell)
                break
            except (LiveLakeMoved, LiveHomeRefused):
                raise
            except BaseException as e:
                _lake_unmoved(
                    before, "a stage that raised"
                )  # a violation outranks the exception it hides (KeyboardInterrupt included)
                if not isinstance(e, Exception) or not spare:
                    raise  # a terminal exception is never an instrument failure to replace
                nxt = spare.pop(0)
                log(
                    f"  seed {s} FAILED on the instrument ({type(e).__name__}: {e}); replaced by {nxt}"
                )
                replaced.append((seed, s, nxt))
                s = nxt
        recorded[s] = rec
        cells.extend(cs)
    for s, rec in recorded.items():
        seq = rec["cats"]
        f = shift_fraction(seq, circular_shift(seq))
        vac = "  VACUOUS" if f < VACUITY_FLOOR else ""
        log(f"  seed {s}: M {1 + max(seq)}  events {len(seq)}  shift f {f:.3f}{vac}")
    if record_path:
        log(
            f"  records {record_path} sha256 {write_records(record_path, recorded, cells, replaced)}"
        )
    return recorded, cells, replaced


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
# Part D D1: the brain producer of the tuning signal and its brain-free positive control PD-P1
# (record: Amendment 2026-10-08 D.3 and the dated note)
# --------------------------------------------------------------------------------------------------
PD_P1_SEEDS = range(
    11500, 11508
)  # note (d): the fresh block proposed for a clean PD-P1; UNRUN, lead go only
PD_OUTCOME = (
    1.0,
    -1.0,
)  # (right, wrong): the teacher's d as +-1, the incumbent's own sign convention
#: PD-P1's planted stream goes THROUGH the pool feature path (lead ruling 2026-10-08, note (g)): three class
#: texts, each presented with one filler word, so the 256-unit T_in = 1 fired mask is the label neurons' input.
PD_CLASS_TEXTS = (
    "greet the user warmly please",
    "what is the capital of france",
    "set a timer for ten minutes",
)
PD_FILLERS = ("now", "today", "quickly", "kindly", "again", "soon", "right", "maybe")
PD_PER_CLASS, PD_HELD = 4, 20


class BrainProducer:
    """The item-9 producer: the harness's own DopamineNuclei behind competence_executive.DopamineModulator
    (the ONE RPE -> modulation step, P2). Input: the one-bit outcome r (+1 right, -1 wrong) and the stimulus key, never a label.
    The modulator computes rpe = r - V(stim_key) and the nucleus's da_level; `to_signed` is the single
    unit conversion (the modulator's output is an unsigned learning-rate multiplier; interact's scalar
    branch needs a signed reward): the deviation from tonic_lr scaled by the modulator's OWN bound on
    that side, so m is in [-1, +1], zero at zero RPE, and symmetric (note (g)). `neutered` delivers r to the nucleus and modulator but pins the
    output to the tonic and never updates V (the path-breaking ablation, D.3(d))."""

    def __init__(self, seed, neutered=False):
        from snn_data_lake.faculties.competence_executive import DopamineModulator
        from snn_data_lake.faculties.dopamine_nuclei import DopamineNuclei

        self.nucleus = DopamineNuclei(seed=seed)
        self.modulator = DopamineModulator(seed=seed, nucleus=self.nucleus)
        self.neutered = bool(neutered)

    @staticmethod
    def to_signed(modulator, lr_gain):
        t = modulator.tonic_lr
        if lr_gain >= t:
            return (lr_gain - t) / (modulator.LR_MAX - t)
        return (lr_gain - t) / (t - modulator.LR_MIN)

    @staticmethod
    def agree(expected, r, rpe_n):
        """The modulator's rpe (r - expected) and the nucleus's TD rpe are one number by construction."""
        assert abs((r - expected) - rpe_n) < 1e-9, f"two RPEs: {r - expected} vs {rpe_n}"

    def factor(self, stim_key, r):
        expected = self.nucleus.v.get(stim_key, 0.0)
        lr = self.modulator.modulate(float(r), expected_gain=expected)
        if self.neutered:
            return self.to_signed(self.modulator, self.modulator.tonic_lr)
        trial = [("ITI", 0.0), (stim_key, 0.0), ("OUT", float(r))]
        self.agree(expected, float(r), self.nucleus.present(trial, learn=True)["OUT"])
        return self.to_signed(self.modulator, lr)


def pd_p1_arm(seed, neutered, epochs=30):
    """Held-out accuracy of the label neurons tuned ONLY by the producer's scalar, on a planted separable text
    stream read through the Part D pool feature path (pd_pool "reset" -> pool_features, T_in = 1 binary mask).
    Brain-free: a pool, no Brain, no lake. Same seed -> same stream for both arms."""
    import random

    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner

    rng = random.Random(seed)
    pool = pd_pool(seed, "reset")
    n = len(PD_CLASS_TEXTS)

    def draw(c):
        return f"{PD_CLASS_TEXTS[c]} {rng.choice(PD_FILLERS)}"

    labels = [f"c{i}" for i in range(n)]
    train = [(draw(c), c) for c in range(n) for _ in range(PD_PER_CLASS)]
    train = [
        (i, t, c) for i, (t, c) in enumerate(train)
    ]  # i = the stimulus key, stable across epochs
    held = [(draw(c), c) for c in range(n) for _ in range(PD_HELD)]
    feats = {t: pool_features(pool, t).tolist() for _, t, _ in train}
    feats.update({t: pool_features(pool, t).tolist() for t, _ in held})
    learner = SpikingInteractionLearner(labels, len(next(iter(feats.values()))), seed=seed)
    prod = BrainProducer(seed, neutered=neutered)
    for _ in range(epochs):
        rng.shuffle(train)
        for i, t, c in train:
            r = PD_OUTCOME[0] if learner.respond(feats[t]) == labels[c] else PD_OUTCOME[1]
            learner.interact(feats[t], float(prod.factor(f"s{i}", r)))
    return sum(learner.respond(feats[t]) == labels[c] for t, c in held) / len(held)


def pd_p1(seeds=None):
    """PD-P1: contrast(BRAIN - PRODNEUTER) over the registered planted seeds; MET means the ablation can fire."""
    diffs = [
        pd_p1_arm(s, False) - pd_p1_arm(s, True) for s in (PD_P1_SEEDS if seeds is None else seeds)
    ]
    return reading("PD-P1 BRAIN - PRODNEUTER", diffs)


#: Part D's pool: G1(b)'s tuned g1 LSM (carried_state_campaign.TUNED), one source; window 0 so the form
#: applies from percept 1 (G1(c) 2.6). Modes: carry / reset are the pool; neuter_* zero the recurrent
#: weights (a stateless LIF expansion under neuter_reset; under neuter_carry the membrane still carries
#: v = v*lam + injected, so it is NOT stateless -- G2 #162, the S1 bound clause applies to neuter_reset only).
PD_POOL_MODES = {
    "carry": ("carried", False),
    "reset": ("reset", False),
    "neuter_carry": ("carried", True),
    "neuter_reset": ("reset", True),
}


def _pd_pool_config():
    from snn_data_lake.instruments.carried_state_campaign import TUNED

    return dict(TUNED)


PD_POOL_CONFIG = _pd_pool_config()


def pd_pool(recording_seed, mode):
    """A KernelLsmPool for a Part D arm. Fixtures and registered reads alike go through here (one path)."""
    if mode not in PD_POOL_MODES:
        raise ValueError(f"mode must be one of {tuple(PD_POOL_MODES)}, got {mode!r}")
    import numpy as np

    from snn_data_lake.reservoir import recurrent_pool as rp

    form, neuter = PD_POOL_MODES[mode]
    pool = rp.make_pool("lsm", PD_POOL_CONFIG, recording_seed, form, window=0)
    if not neuter:
        return pool
    return rp.KernelLsmPool(
        pool.pre,
        np.zeros_like(pool.w),
        pool.config,
        form,
        pool.w_in,
        window=0,
        s_settle=pool.s_settle,
    )


def pool_features(pool, text, *, feed=None):
    """One turn's 256-vector: the pool's mean activity over T_in ticks for the text's 52-code zero-padded
    to the encoder's 64 inputs (the padding is fixed, not tuned). The caller owns episode-boundary resets."""
    import numpy as np

    from snn_data_lake.instruments import inseparable_mapping as im
    from snn_data_lake.reservoir.recurrent_pool import N_IN

    x = np.zeros(N_IN)
    code = im.code(text)
    x[: len(code)] = code
    return pool.percept(x) if feed is None else pool.percept_fed(x, feed)


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the cheap reds (never builds a Brain, never opens a lake)
# --------------------------------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    torch.set_num_threads(1)
    if "--pa3" in argv:
        return 0 if pa3() else 1  # a failed positive control must not exit 0 (Codex g5_b lens-1 m2)
    if "--pd-p1" in argv:  # Part D's producer-ablation positive control; a NOT MET must not exit 0
        if "--lead-go" not in argv:  # the registered block is read once, on the lead's explicit go
            print("REFUSED: --pd-p1 reads the registered PD_P1_SEEDS; it needs --lead-go")
            return 2
        met, c, x = pd_p1()
        print(_line("PD-P1 BRAIN - PRODNEUTER", met, c, x))
        return 0 if met else 1
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
    print(
        "g5_readout_cells self-check OK: the runner is deterministic and the online readout moves; the frozen "
        "generator is deterministic and iid is not run-structured; the reading says NOT MET on degenerate, "
        "mostly-tied and sign-balanced samples; PA4 vacuity, PA8 same-rows halt and PA10 early-category halt (with the common init passing) all fire"
    )


if __name__ == "__main__":
    sys.exit(main())
