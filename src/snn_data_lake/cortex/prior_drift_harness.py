"""THE three-arm drift harness: is the H01 prior beatable, and does learning move the brain off it?

specs/022 T063 / SC-006 / FR-022, composing specs/011's DriftRecord design (SC-004, SC-005,
FR-008/009/010, C5, D1, D4, Scenario 2, T003, T004). The pre-registration is
`specs/022-unified-brain-arc/evidence/t063-prior-drift-harness.md`, sections 1-2.11 FROZEN before
a line of this module existed; this file implements it and adds nothing to it.

FIVE TABLES on ONE unit set, ONE drive, ONE learner, ONE readout -- only the wiring differs:
  SEEDED   T062's H01 c3 population at the pilot's width.
  RANDOM   the control that is NOT the method: a WITHIN-SIGN degree-preserving rewire, the
           per-edge synapse-count multiset reassigned at random inside each sign class, no self-
           or multi-edges, fan-in re-normalised. WHAT IT TESTS, narrowly: wiring information
           BEYOND the retained H01 marginals (degrees, signs, magnitudes), which the two arms
           SHARE. It cannot separate "H01 contributes nothing" from "H01's useful information is
           exactly what both arms share", and every claim here is restricted to that component.
  WRONG    the DIRECTION-REVERSED laminar table (specs/011 T004): every INTER-band edge's pre and
           post swapped, intra-band untouched, polarity re-applied from the NEW source's cell
           sign, fan-in re-normalised. VALID only if its band-pair weight-share matrix is farther
           from SEEDED's than RANDOM's realised matrix is -- measured, never assumed.
  PLANTED  the POSITIVE CONTROL, never an arm of the verdict: each drive class's units wired
           densely onto a disjoint block of non-driven units. ONE predicate: PLANTED beats RANDOM
           at the same t bar. If it does not, the harness is BLIND and NO verdict is read.
  FROZEN   the learner's own control: SEEDED at lr = 0. Red 1's constructed pin, and the twin the
           learner's retention criterion is read against.

THE NUISANCE PROPERTIES, held constant by construction and REFUSED if violated: the unit set and
each unit's cell sign; every unit's outgoing edges carry its own sign (Dale); every unit's total
absolute fan-in is 1; the edge count; the per-edge synapse-count multiset.

DISCLOSED READING of "the weight multiset" (2.2). Weight is `sign(pre) * n_syn / sum_pre' n_syn`,
so a multiset of WEIGHTS and a re-applied fan-in normalisation cannot both be preserved: the
normaliser changes when the edges move. The quantity held constant is therefore the multiset of
per-edge SYNAPSE COUNTS -- the pre-normalisation quantity the weight is computed FROM -- and the
fan-in normalisation is re-applied on top, exactly as 2.2 requires. Stated here rather than left
implied, because it is the one place the frozen text admits two readings.

WHAT THIS ROW DOES NOT CLAIM (2.8): nothing about the LIVE brain (the population is off the live
tick, and an instrument that ticks a scratch population does not discharge `kernel_tick`'s or
`kernel_state`'s "wire" exemptions); nothing about the connectome beyond the c3 soma-to-soma
subgraph T062 measured; nothing about R-STDP being the brain's learner; nothing about specs/011's
placement-side statistics (layer occupancy, layer geometry, the excitatory CELL fraction), which
are a pinned null on the object brain and which the graph-weight statistics below do not measure.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.prior_drift_harness
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.prior_drift_harness --pilot
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.prior_drift_harness --campaign
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

from snn_data_lake.substrate.drift_monitor import psi

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
#:
GRADIENT_CLAIM = {
    "fit_readout": (
        "DIRECT_SOLVE",
        "the task's readout is a least-squares linear map from non-driven spike counts to a "
        "one-hot class, obtained in closed form by numpy.linalg.lstsq. No error is propagated, "
        "no weight of the POPULATION is touched, and nothing is iterated -- the readout is solved, "
        "not trained, which is why it can be refit between blocks without contaminating the "
        "learner's own attribution",
    ),
    "_learner": (
        "NOT_A_RULE",
        "the verdict's LEARNER DISPOSITION formatter: it reads the already-computed t(d_f) "
        "contrast and the arms' effective-change counts and returns KEPT or DELETED with its "
        "reason. It touches no weight, forms no trace and computes no update -- 'learner' here "
        "names the thing being JUDGED, not a rule. Declared rather than renamed, because the "
        "census is right to enumerate the token and a reader is better served by the reason than "
        "by a name chosen to dodge a gate",
    ),
}

REPO = Path(__file__).resolve().parents[3]
PILOT_PIN = REPO / "tools" / "t063_pilot.json"
CAMPAIGN_PIN = REPO / "tools" / "t063_campaign.json"

# ---- CONSTANTS, chosen BEFORE the pilot and disclosed as P1 debt (2.7) ------------------------
T_TRIAL = 50  # ticks per trial
TRIALS_PER_CLASS = 4  # trials per class per block
C_BLOCKS = 5  # learning blocks after one warm-up block
HELD_OUT_PER_CLASS = 1  # held-out trials per class per block, via marginal_holdout
CLASS_FRAC = 0.05  # each drive class is 5 % of the units
K_CANDIDATES = (8, 16, 32)  # powers of two >= 8; the pilot picks the first under the ceiling
K_CEILING = 0.9  # the no-learning RANDOM arm must score BELOW this
P_DRIVE = 0.30  # per-tick firing probability of a driven unit (the Poisson shape)
DRIVE_AMPLITUDE = 1.2  # x threshold, the amplitude h01_ingest's own liveness driver uses
WIDTH_FRACTIONS = (1.0, 0.5, 0.25, 0.125)
CAMPAIGN_HOURS = 6.0  # the projection ceiling W is chosen under
SWAP_PASSES = 10  # double-edge swap attempts per edge, per sign class, in the RANDOM rewire

#: THE HOST CLAUSE, READ AND DISCLOSED (2.7). Measured 2026-09-09 on the daemon at PID 27228,
#: five consecutive 90 s windows of process CPU: 112.000, 122.906, 106.266, 133.641, 117.828 s.
#: Mean 118.5 s per 90 s wall (~132 % of one core on a 28-core box), no monotone trend, range
#: +-11.5 %. The pre-registration's literal test -- two consecutive windows within ~6 % -- is NOT
#: MET and cannot be met by this daemon in ANY state: the swing is its ordinary wake/sleep duty
#: cycle, not a boot transient. What the clause is FOR (no startup burst inflating a timing) is
#: read instead from the absence of a trend across five windows, and every timing number below is
#: a MIN of repeats so the swing costs precision rather than correctness. Recorded, not smoothed.
HOST_NOTE = (
    "daemon PID 27228, LISTEN on 8137, working set 5.50-8.77 GB; five 90 s CPU windows "
    "112.000 / 122.906 / 106.266 / 133.641 / 117.828 s, mean 118.5, no trend, range +-11.5 %. "
    "The two-consecutive-windows-within-6 % clause is NOT MET and is not meetable by this "
    "daemon's steady duty cycle; settling read from the absence of a trend. No battery and no "
    "panel were running. Every timing is a min of repeats."
)

S_SEEDS = 8
T_BAR = 1.895  # paired one-sided t, 7 d.f., alpha = 0.05 (PLAN-FINAL Part F preamble)
STATISTICS = ("band_pair_weight_share", "band_fanin_e_fraction")
N_BANDS = 7


class ArmRefused(ValueError):
    """An arm that violates a nuisance property is refused BEFORE it runs, not explained after."""


class DriftRecord(NamedTuple):
    """specs/011 data-model.md 4, fields verbatim. Sign is EXCLUDED from drift by design."""

    statistic: str
    seeded: tuple
    current: tuple
    distance: float
    direction: int


class ArmResult(NamedTuple):
    """One arm, one seed. `pinned` is TRUE only for a VERIFIED lr = 0 intervention -- it is the
    single thing that licenses the word "pinning" in red 1's message."""

    name: str
    pinned: bool
    accuracy: float
    executed: int
    attempted: int
    effective_count: int
    effective_l1: float
    records: tuple


class Contrast(NamedTuple):
    name: str
    diffs: tuple
    t: float
    fpr: float
    degenerate: bool
    met: bool


# =============================================================================================
# The statistics (2.4) -- NEWLY CHOSEN graph-weight statistics, and what they do not measure
# =============================================================================================
def band_pair_weight_share(layer, pre, post, w):
    """S1: the 7x7 matrix of sum |w| over edges band a -> band b, divided by sum |w| over all
    edges. 49 cells summing to 1. Invariant to a uniform positive rescaling of all weights and to
    redistribution WITHIN a band pair -- which is why drift is never read from a statistic alone
    (2.5's four diagnostic lines exist for exactly that reason)."""
    import numpy as np

    a = np.asarray(layer)[np.asarray(pre)]
    b = np.asarray(layer)[np.asarray(post)]
    mag = np.abs(np.asarray(w, dtype=np.float64))
    cell = np.zeros(N_BANDS * N_BANDS, dtype=np.float64)
    np.add.at(cell, a.astype(np.int64) * N_BANDS + b.astype(np.int64), mag)
    total = float(cell.sum())
    if total <= 0.0:
        return tuple([0.0] * (N_BANDS * N_BANDS))
    return tuple((cell / total).tolist())


def band_fanin_e_fraction(layer, post, w):
    """S2: per band b, sum of POSITIVE w into b over sum |w| into b. Seven scalars."""
    import numpy as np

    b = np.asarray(layer)[np.asarray(post)].astype(np.int64)
    wv = np.asarray(w, dtype=np.float64)
    pos = np.zeros(N_BANDS, dtype=np.float64)
    mag = np.zeros(N_BANDS, dtype=np.float64)
    np.add.at(pos, b, np.maximum(wv, 0.0))
    np.add.at(mag, b, np.abs(wv))
    out = np.divide(pos, mag, out=np.zeros(N_BANDS), where=mag > 0)
    return tuple(out.tolist())


def statistics_of(layer, pre, post, w):
    return {
        "band_pair_weight_share": band_pair_weight_share(layer, pre, post, w),
        "band_fanin_e_fraction": band_fanin_e_fraction(layer, post, w),
    }


def distance(name, current, seeded):
    """S1's distance is the PSI the drift monitor owns (the P2 merge); S2's is the mean absolute
    per-band difference. Sign never enters either."""
    if name == "band_pair_weight_share":
        return float(psi(current, seeded))
    if name == "band_fanin_e_fraction":
        n = len(seeded) or 1
        return float(sum(abs(c - s) for c, s in zip(current, seeded, strict=True)) / n)
    raise ValueError(f"unknown statistic {name!r}")


def drift_records(seeded_stats, current_stats, mid_stats):
    """One DriftRecord per seeded statistic (C5: EVERY arm produces them). `direction` is the sign
    of distance(end) - distance(mid): +1 still moving away from the seed, -1 returning, 0 flat."""
    out = []
    for name in STATISTICS:
        s, c, m = seeded_stats[name], current_stats[name], mid_stats[name]
        d_end = distance(name, c, s)
        d_mid = distance(name, m, s)
        delta = d_end - d_mid
        out.append(DriftRecord(name, tuple(s), tuple(c), d_end, (delta > 0) - (delta < 0)))
    return tuple(out)


def is_seeding_failure(records):
    """FR-009 / C5 / T012: zero distance across ALL statistics is a FAILURE of the seeding
    approach, reported as such -- never as stability and never as success."""
    return bool(records) and all(r.distance == 0.0 for r in records)


# =============================================================================================
# The two reds' readings (2.5)
# =============================================================================================
def red1(result):
    """RED 1 -- never moves off the seed. Returns (fired, message).

    Pinning is attributed ONLY to the verified intervention. A learning arm's zero is reported as
    what it is -- cancelled / below quantization, or no eligible spike pairs -- and red 1 does NOT
    fire on it, because a zero from four other causes is not evidence about seeding."""
    zero_drift = is_seeding_failure(result.records)
    if result.pinned and result.effective_count == 0 and zero_drift:
        return True, (
            "seeding approach failed: PINNED (lr = 0 by construction): "
            f"executed {result.executed}, attempted {result.attempted}, effective 0, drift 0"
        )
    if result.effective_count == 0 and result.attempted == 0:
        return False, (
            f"{result.name}: no eligible spike pairs (executed {result.executed}, attempted 0) "
            "-- NOT pinning"
        )
    if result.effective_count == 0:
        return False, (
            f"{result.name}: updates cancelled or below quantization (executed "
            f"{result.executed}, attempted {result.attempted}, effective 0) -- NOT pinning"
        )
    return False, (
        f"{result.name}: moved (executed {result.executed}, attempted {result.attempted}, "
        f"effective {result.effective_count}, L1 {result.effective_l1:.6f})"
    )


def drift_control_holds(result):
    """Red 1's control: the SEEDED arm at the declared lr must read effective > 0 AND a distance
    above zero on BOTH statistics. A red whose control does not hold measures the harness."""
    return bool(
        result.effective_count > 0
        and result.records
        and all(r.distance > 0.0 for r in result.records)
    )


# =============================================================================================
# The statistic over the seeds (2.6)
# =============================================================================================
def paired_t(diffs):
    """Paired one-sided t over the S seeds; ties are counted, never dropped. 0.0 when the sample
    has no variance -- that case is DEGENERATE and `contrast` decides it explicitly."""
    n = len(diffs)
    if n < 2:
        return 0.0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var <= 0.0:
        return 0.0
    return mean / math.sqrt(var / n)


def permutation_fpr(diffs, bar=T_BAR):
    """The false-positive rate of this test on THIS sample: the fraction of the 2^S sign flips
    whose t exceeds the bar. Printed beside every contrast (Part F's preamble)."""
    n = len(diffs)
    hits = 0
    for mask in range(1 << n):
        flipped = [(-d if (mask >> i) & 1 else d) for i, d in enumerate(diffs)]
        if paired_t(flipped) > bar:
            hits += 1
    return hits / float(1 << n)


def contrast(name, diffs, bar=T_BAR):
    """One contrast, decided BEFORE any number: a zero-variance set of differences is MET iff its
    common value is > 0 (recorded as degenerate) and NOT MET otherwise."""
    diffs = tuple(float(d) for d in diffs)
    degenerate = len(set(diffs)) <= 1
    if degenerate:
        return Contrast(name, diffs, 0.0, 0.0, True, bool(diffs and diffs[0] > 0.0))
    t = paired_t(diffs)
    return Contrast(name, diffs, t, permutation_fpr(diffs, bar), False, t > bar)


# =============================================================================================
# The arms (2.2)
# =============================================================================================
def _copy(art):
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in art.items()}


def population_order(art):
    """The artifact index -> population index map `h01_ingest.population` builds, and its inverse.
    Every arm shares it: the node set, the layers and the agglo ids are nuisance-constant, so the
    population index of a unit is the same in every arm and a drive class drawn once is the same
    subset everywhere."""
    import numpy as np

    order = np.lexsort((art["agglo_id"], art["layer"]))
    pos = np.empty(order.size, dtype=np.int64)
    pos[order] = np.arange(order.size, dtype=np.int64)
    return order, pos


def population_layers(art):
    order, _pos = population_order(art)
    return art["layer"][order]


def arm_seeded(art):
    return _copy(art)


def arm_random(art, seed):
    """The within-sign degree-preserving rewire. Double-edge swaps inside each SOURCE-SIGN class
    keep every unit's in-degree, out-degree and polarity exactly; the synapse-count multiset is
    then permuted inside the same class; fan-in is re-normalised by the weight formula itself."""
    import numpy as np

    out = _copy(art)
    pre = out["pre_idx"].astype(np.int64).copy()
    post = out["post_idx"].astype(np.int64).copy()
    n_syn = out["n_syn"].copy()
    sign = art["sign"]
    rng = np.random.default_rng(int(seed) * 1_000_003 + 17)
    existing = set(zip(pre.tolist(), post.tolist(), strict=True))

    for cls in (+1, -1):
        idx = np.nonzero(sign[pre] == cls)[0]
        m = idx.size
        if m < 2:
            continue
        picks = rng.integers(0, m, size=(SWAP_PASSES * m, 2))
        for u, v in picks.tolist():
            if u == v:
                continue
            i, j = int(idx[u]), int(idx[v])
            a, b = int(pre[i]), int(post[i])
            c, d = int(pre[j]), int(post[j])
            if a == d or c == b:
                continue
            if (a, d) in existing or (c, b) in existing:
                continue
            existing.discard((a, b))
            existing.discard((c, d))
            existing.add((a, d))
            existing.add((c, b))
            post[i], post[j] = d, b
        n_syn[idx] = n_syn[idx][rng.permutation(m)]

    order = np.lexsort((post, pre))
    out["pre_idx"] = pre[order].astype(art["pre_idx"].dtype)
    out["post_idx"] = post[order].astype(art["post_idx"].dtype)
    out["n_syn"] = n_syn[order]
    return out


def arm_wrong(art):
    """The direction-reversed laminar table (specs/011 T004). INTER-band edges have pre and post
    swapped; intra-band edges are untouched. This transposes the realised band-pair matrix exactly
    and swaps each unit's inter-band in- and out-degree; the degree DISTRIBUTION is preserved and
    per-unit degrees are NOT, which is disclosed rather than hidden. Polarity follows the NEW
    source because the weight formula reads `sign(pre)`, and fan-in is re-normalised by it too."""
    import numpy as np

    out = _copy(art)
    lay = art["layer"]
    pre = out["pre_idx"].astype(np.int64).copy()
    post = out["post_idx"].astype(np.int64).copy()
    inter = lay[pre] != lay[post]
    pre_new = np.where(inter, post, pre)
    post_new = np.where(inter, pre, post)
    out["pre_idx"] = pre_new.astype(art["pre_idx"].dtype)
    out["post_idx"] = post_new.astype(art["post_idx"].dtype)
    return out


def arm_planted(art, classes, seed):
    """The POSITIVE CONTROL: each drive class's units wired onto a DISJOINT block of non-driven
    units, under the same nuisance constraints (same units, same signs, same edge count, same
    synapse-count multiset, fan-in re-normalised). It is never an arm of the verdict -- it exists
    to prove the harness can see an advantage at all.

    DISCLOSED: PLANTED may repeat a (pre, post) pair. "No self-edges or multi-edges" is a property
    2.2 states for the RANDOM rewire, not one of the nuisance properties every arm holds constant,
    and without that freedom the positive control would be refused on any table whose edge count
    exceeds sum_c |class c| x |block c| -- i.e. the control would be unbuildable exactly where the
    task is hardest. Sources and targets are walked on coprime strides so both are covered before
    any pair repeats."""
    import numpy as np

    out = _copy(art)
    order, _pos = population_order(art)
    n = int(art["agglo_id"].size)
    n_edges = int(art["pre_idx"].size)
    driven = sorted({int(u) for cls in classes for u in cls})
    free = [u for u in range(n) if u not in set(driven)]
    k = len(classes)
    if not free or k == 0:
        raise ArmRefused("planted needs at least one non-driven unit and one class")

    blocks = [free[i::k] for i in range(k)]
    per = [n_edges // k + (1 if i < n_edges % k else 0) for i in range(k)]
    pre, post = [], []
    for c in range(k):
        src = sorted(int(u) for u in classes[c])
        dst = blocks[c] or free
        for j in range(per[c]):
            pre.append(src[j % len(src)])
            post.append(dst[j % len(dst)])

    rng = np.random.default_rng(int(seed) * 7_919 + 3)
    pre_a = order[np.asarray(pre, dtype=np.int64)]
    post_a = order[np.asarray(post, dtype=np.int64)]
    n_syn = art["n_syn"].copy()[rng.permutation(n_edges)]
    srt = np.lexsort((post_a, pre_a))
    out["pre_idx"] = pre_a[srt].astype(art["pre_idx"].dtype)
    out["post_idx"] = post_a[srt].astype(art["post_idx"].dtype)
    out["n_syn"] = n_syn[srt]
    return out


def check_nuisance(art, base, name, unique_edges=False):
    """Refuse an arm that changed something the comparison holds constant. Called BEFORE the arm
    runs -- a nuisance violation makes the whole contrast uninterpretable, so it is not a warning."""
    import numpy as np

    from snn_data_lake.cortex.h01_ingest import fan_in_shares

    for key in ("agglo_id", "sign", "layer"):
        if not np.array_equal(art[key], base[key]):
            raise ArmRefused(f"{name}: the unit set changed ({key})")
    if art["pre_idx"].size != base["pre_idx"].size:
        raise ArmRefused(f"{name}: edge count {art['pre_idx'].size} != {base['pre_idx'].size}")
    if not np.array_equal(np.sort(art["n_syn"]), np.sort(base["n_syn"])):
        raise ArmRefused(f"{name}: the per-edge synapse-count multiset changed")
    if np.any(art["pre_idx"] == art["post_idx"]):
        raise ArmRefused(f"{name}: a self-edge was created")
    if unique_edges:
        pairs = art["pre_idx"].astype(np.int64) * (art["agglo_id"].size + 1) + art["post_idx"]
        if np.unique(pairs).size != pairs.size:
            raise ArmRefused(f"{name}: a multi-edge was created")
    w = fan_in_shares(art)
    pol = art["sign"][art["pre_idx"]].astype(np.float64)
    if np.any(np.sign(w[w != 0.0]) != pol[w != 0.0]):
        raise ArmRefused(f"{name}: an edge does not carry its source cell's sign (Dale)")
    mag = np.zeros(art["agglo_id"].size, dtype=np.float64)
    np.add.at(mag, art["post_idx"], np.abs(w))
    live = mag > 0
    if np.any(np.abs(mag[live] - 1.0) > 1e-9):
        raise ArmRefused(f"{name}: a unit's total absolute fan-in is not 1")


def wrong_validity(art_wrong, art_random, art_seeded):
    """T004's VALIDITY PRECONDITION, measured on the REALISED tables: the reversed table's
    band-pair weight-share matrix must be farther from SEEDED's, in the same PSI the drift
    statistic uses, than RANDOM's realised matrix is. Returns (valid, d_wrong, d_random)."""
    from snn_data_lake.cortex.h01_ingest import fan_in_shares

    lay = art_seeded["layer"]
    s = band_pair_weight_share(
        lay, art_seeded["pre_idx"], art_seeded["post_idx"], fan_in_shares(art_seeded)
    )
    dw = distance(
        "band_pair_weight_share",
        band_pair_weight_share(
            lay, art_wrong["pre_idx"], art_wrong["post_idx"], fan_in_shares(art_wrong)
        ),
        s,
    )
    dr = distance(
        "band_pair_weight_share",
        band_pair_weight_share(
            lay, art_random["pre_idx"], art_random["post_idx"], fan_in_shares(art_random)
        ),
        s,
    )
    return bool(dw > dr), dw, dr


# =============================================================================================
# Width (2.7) and the task (2.3)
# =============================================================================================
def band_balanced_subgraph(art, frac):
    """The largest band-balanced induced subgraph at `frac` of the units: within EACH band, keep
    the units of highest total degree (ties by index), so no band is emptied and the laminar
    composition is preserved. Edges are the induced ones; node arrays are re-indexed."""
    import numpy as np

    if frac >= 1.0:
        return _copy(art)
    n = int(art["agglo_id"].size)
    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, art["pre_idx"].astype(np.int64), 1)
    np.add.at(deg, art["post_idx"].astype(np.int64), 1)
    keep = []
    for band in range(N_BANDS):
        members = np.nonzero(art["layer"] == band)[0]
        if members.size == 0:
            continue
        take = max(1, int(round(members.size * frac)))
        ranked = sorted(members.tolist(), key=lambda u: (-int(deg[u]), u))
        keep.extend(ranked[:take])
    keep = np.asarray(sorted(keep), dtype=np.int64)
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size, dtype=np.int64)
    sel = (remap[art["pre_idx"].astype(np.int64)] >= 0) & (
        remap[art["post_idx"].astype(np.int64)] >= 0
    )
    out = {
        "agglo_id": art["agglo_id"][keep],
        "sign": art["sign"][keep],
        "layer": art["layer"][keep],
        "pre_idx": remap[art["pre_idx"].astype(np.int64)][sel].astype(art["pre_idx"].dtype),
        "post_idx": remap[art["post_idx"].astype(np.int64)][sel].astype(art["post_idx"].dtype),
        "n_syn": art["n_syn"][sel],
    }
    if "meta" in art:
        out["meta"] = art["meta"]
    return out


def drive_classes(art, k, seed, frac=CLASS_FRAC):
    """K disjoint drive classes, each a fixed `frac` subset of the units, drawn per seed in
    POPULATION index space (identical across arms by construction, 2.6)."""
    import numpy as np

    n = int(art["agglo_id"].size)
    per = max(1, int(n * frac))
    if per * k > n:
        raise ArmRefused(f"{k} classes of {per} units do not fit in {n} units")
    rng = np.random.default_rng(int(seed) * 2_654_435_761 + 11)
    draw = rng.permutation(n)[: per * k]
    return tuple(tuple(sorted(int(u) for u in draw[c * per : (c + 1) * per])) for c in range(k))


def trial_drive(classes, cls, seed, trial_id, n_units, t_trial=T_TRIAL):
    """The deterministic per-seed Poisson-shaped drive of ONE trial: a (t_trial, n_units) boolean
    mask over which units receive the external current on which tick. Generated from (seed,
    trial_id) alone, so every arm of a seed is driven by exactly the same input."""
    import numpy as np

    rng = np.random.default_rng((int(seed) + 1) * 1_000_033 + int(trial_id))
    units = np.asarray(classes[cls], dtype=np.int64)
    mask = rng.random((t_trial, units.size)) < P_DRIVE
    out = np.zeros((t_trial, n_units), dtype=bool)
    out[:, units] = mask
    return out


def fit_readout(features, labels, k):
    """The least-squares linear map from a feature vector to a one-hot class -- solved in closed
    form, never trained (see GRADIENT_CLAIM). Returns the (n_feat + 1, k) coefficient matrix, or
    None when there is nothing to fit."""
    import numpy as np

    if not len(features):
        return None
    x = np.asarray(features, dtype=np.float64)
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    y = np.zeros((x.shape[0], k), dtype=np.float64)
    y[np.arange(x.shape[0]), np.asarray(labels, dtype=np.int64)] = 1.0
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def predict(coef, feature, k):
    import numpy as np

    if coef is None:
        return 0
    x = np.concatenate([np.asarray(feature, dtype=np.float64), [1.0]])
    return int(np.argmax(x @ coef)) % k


def _reset_population(kt, plast):
    """The per-trial reset of 2.3: membranes, refractory state, spike memory, the delay rings and
    the partition accumulators, and the learner's eligibility. Weights are NOT reset -- what a
    trial resets is state, and the weights are the thing under measurement."""
    kt.state.arrays["v"][:] = 0.0
    kt.state.arrays["refrac"][:] = 0
    kt.state.arrays["last_spike"][:] = 0
    for ring in kt._rings:
        for slot in ring:
            slot.clear()
    for acc in kt._acc:
        acc[:] = 0
    if plast is not None:
        plast.reset()


def run_trial(kt, plast, mask, learn, amp_q):
    """One trial: reset, drive for T_trial ticks, collect the spike counts. When `learn`, the
    learner forms eligibility from each tick's spike set and decays it; a held-out trial passes
    `learn=False` and carries no state at all."""
    import numpy as np

    _reset_population(kt, plast)
    n = kt.state.n_units
    counts = np.zeros(n, dtype=np.int64)
    ext = np.zeros(n, dtype=np.int64)
    for tick in range(mask.shape[0]):
        ext[:] = 0
        ext[mask[tick]] = amp_q
        fired = kt.tick(ext)
        counts[fired] += 1
        if learn and plast is not None:
            plast.update_eligibility(fired, float(tick + 1))
            plast.decay()
    return counts


def run_arm(
    art,
    name,
    seed,
    classes,
    lr=None,
    blocks=C_BLOCKS,
    t_trial=T_TRIAL,
    trials_per_class=TRIALS_PER_CLASS,
    held_out=HELD_OUT_PER_CLASS,
    threads=1,
):
    """One arm, one seed, the frozen per-trial protocol of 2.3. Returns an ArmResult.

    Order per trial: reset -> drive -> collect -> PREDICT with the readout fit on all PREVIOUS
    blocks -> reward = +1 if correct else -1 -> the learner updates. Block 0 uses the readout fit
    on a warm-up block run with the learner OFF. Held-out trials run with the learner OFF, deliver
    no reward, form no eligibility and carry no state; the split is `marginal_holdout` per block.
    """
    import numpy as np

    from snn_data_lake.basal_ganglia.r_stdp import LR
    from snn_data_lake.cortex.h01_ingest import population
    from snn_data_lake.hippocampus.ng_snn import marginal_holdout
    from snn_data_lake.substrate.kernel_plasticity import KernelPlasticity

    lr = LR if lr is None else lr
    state, syn, kt = population(art, threads=threads)
    plast = KernelPlasticity(kt, lr=lr)
    layer = population_layers(art)
    amp_q = int(DRIVE_AMPLITUDE * float(state.arrays["thr"][0]) * (1 << plast.frac_bits))

    seeded_stats = statistics_of(layer, plast.edge_pre, plast.edge_post, plast.weights())
    mid_stats = seeded_stats

    k = len(classes)
    driven = np.zeros(state.n_units, dtype=bool)
    for cls in classes:
        driven[np.asarray(cls, dtype=np.int64)] = True
    feat_idx = np.nonzero(~driven)[0]

    pool_x, pool_y = [], []
    coef = None
    accuracy = 0.0
    trial_id = 0
    mid_block = blocks // 2
    for block in range(-1, blocks):
        labels = [c for c in range(k) for _ in range(trials_per_class)]
        rng = np.random.default_rng((int(seed) + 1) * 88_001 + (block + 1))
        perm = rng.permutation(len(labels))
        labels = [labels[i] for i in perm.tolist()]
        _train_idx, val_idx = marginal_holdout(labels, held_out * k)
        val = set(val_idx)
        block_x, block_y = [], []
        correct = hits = 0
        for i, cls in enumerate(labels):
            learn = (block >= 0) and (i not in val)
            mask = trial_drive(classes, cls, seed, trial_id, state.n_units, t_trial)
            trial_id += 1
            counts = run_trial(kt, plast, mask, learn, amp_q)
            feature = counts[feat_idx]
            guess = predict(coef, feature, k)
            if i in val:
                hits += 1
                correct += int(guess == cls)
            else:
                block_x.append(feature)
                block_y.append(cls)
                if learn:
                    plast.reward(+1.0 if guess == cls else -1.0)
        if hits:
            accuracy = correct / float(hits)
        pool_x.extend(block_x)
        pool_y.extend(block_y)
        coef = fit_readout(pool_x, pool_y, k)
        if block == mid_block:
            mid_stats = statistics_of(layer, plast.edge_pre, plast.edge_post, plast.weights())

    current = statistics_of(layer, plast.edge_pre, plast.edge_post, plast.weights())
    d = plast.diagnostics()
    return ArmResult(
        name,
        pinned=(lr == 0.0),
        accuracy=accuracy,
        executed=d["executed"],
        attempted=d["attempted"],
        effective_count=d["effective_count"],
        effective_l1=d["effective_l1"],
        records=drift_records(seeded_stats, current, mid_stats),
    )


def build_arms(art, seed, classes):
    """The five tables of a seed, each checked against the nuisance list before it can run."""
    tables = {
        "seeded": arm_seeded(art),
        "random": arm_random(art, seed),
        "wrong": arm_wrong(art),
        "planted": arm_planted(art, classes, seed),
    }
    tables["frozen"] = arm_seeded(art)
    check_nuisance(tables["seeded"], art, "seeded", unique_edges=True)
    check_nuisance(tables["random"], art, "random", unique_edges=True)
    check_nuisance(tables["wrong"], art, "wrong")
    check_nuisance(tables["planted"], art, "planted")
    return tables


# =============================================================================================
# The verdict (2.6) and the null clauses (2.9)
# =============================================================================================
def verdict(results, wrong_valid):
    """SC-006's five conjuncts, each printed MET / NOT MET / NOT READ with its numbers, plus the
    LEARNER's own disposition, which is independent of the prior's."""
    seeds = sorted(results)
    acc = {arm: [results[s][arm].accuracy for s in seeds] for arm in results[seeds[0]]}
    d_r = [a - b for a, b in zip(acc["seeded"], acc["random"], strict=True)]
    d_w = [a - b for a, b in zip(acc["seeded"], acc["wrong"], strict=True)]
    d_p = [a - b for a, b in zip(acc["planted"], acc["random"], strict=True)]
    d_f = [a - b for a, b in zip(acc["seeded"], acc["frozen"], strict=True)]
    d_rw = [a - b for a, b in zip(acc["random"], acc["wrong"], strict=True)]

    c_p = contrast("planted - random", d_p)
    c_r = contrast("seeded - random", d_r)
    c_w = contrast("seeded - wrong", d_w)
    c_rw = contrast("random - wrong", d_rw)
    c_f = contrast("seeded - frozen", d_f)

    red1_fired = all(red1(results[s]["frozen"])[0] for s in seeds)
    drift_ok = all(drift_control_holds(results[s]["seeded"]) for s in seeds)
    conj = {}
    if not c_p.met:
        conj["c"] = _con("NOT MET", c_p, "the PLANTED control did not beat RANDOM")
        for key in "abde":
            conj[key] = {"state": "NOT READ", "note": "no verdict is read: the harness is BLIND"}
        out = {
            "conjuncts": conj,
            "sc006": "NO VERDICT",
            "contrasts": _contrasts(c_p, c_r, c_w, c_rw, c_f),
            "learner": _learner(c_f, results, seeds),
            "per_seed": _per_seed(results, seeds),
            "wrong_valid": bool(wrong_valid),
        }
        return out

    conj["c"] = _con("MET", c_p, "the harness can see an advantage")
    conj["a"] = {
        "state": "MET" if (drift_ok and red1_fired) else "NOT MET",
        "note": (
            f"SEEDED effective change > 0 and both distances > 0 on every seed = {drift_ok}; "
            f"red 1 fired on the FROZEN pin = {red1_fired}"
        ),
    }
    conj["b"] = _con("MET" if c_r.met else "NOT MET", c_r, "seeded beats random")
    if not wrong_valid:
        conj["d"] = {
            "state": "NOT READ",
            "note": (
                "the WRONG arm is INVALID (its band-pair matrix is not farther from SEEDED's "
                "than RANDOM's realised matrix is) -- red 2 is not read and no underperformance "
                "is attributed to the prior"
            ),
        }
    elif c_rw.met:
        conj["d"] = _con("MET", c_rw, "red 2 FIRED: WRONG underperforms RANDOM")
    else:
        conj["d"] = _con(
            "NOT MET",
            c_rw,
            "underperformance NOT ESTABLISHED -- red 2 is NOT FIRED and nothing is attributed "
            "to the prior",
        )
    conj["e"] = _con("MET" if c_w.met else "NOT MET", c_w, "seeded beats wrong")
    met = all(conj[k]["state"] == "MET" for k in "abcde")
    return {
        "conjuncts": conj,
        "sc006": "MET" if met else "NOT MET",
        "contrasts": _contrasts(c_p, c_r, c_w, c_rw, c_f),
        "learner": _learner(c_f, results, seeds),
        "per_seed": _per_seed(results, seeds),
        "wrong_valid": bool(wrong_valid),
    }


def _con(state, c, note):
    return {
        "state": state,
        "t": c.t,
        "fpr": c.fpr,
        "degenerate": c.degenerate,
        "diffs": list(c.diffs),
        "note": note,
    }


def _contrasts(*cs):
    return {
        c.name: {"t": c.t, "fpr": c.fpr, "degenerate": c.degenerate, "diffs": list(c.diffs)}
        for c in cs
    }


def _learner(c_f, results, seeds):
    """2.9: the learner is KEPT iff its attributed benefit holds -- t(d_f) > bar AND the effective
    change is above zero. A learner that moves weights without benefit is a null (P8 clauses 3 and
    5, and r_stdp's own docstring rejects weight movement as charge)."""
    moved = any(results[s][a].effective_count > 0 for s in seeds for a in results[s])
    if not moved:
        return {
            "disposition": "DELETED",
            "t": c_f.t,
            "fpr": c_f.fpr,
            "note": (
                "the effective change is 0 on EVERY arm: the learner is deleted and red 1's "
                "reading is a HARNESS DEFECT, not a finding"
            ),
        }
    keep = c_f.met
    return {
        "disposition": "KEPT" if keep else "DELETED",
        "t": c_f.t,
        "fpr": c_f.fpr,
        "note": (
            "learning beats its frozen twin on the seeded table"
            if keep
            else "a learner that moves weights without an attributed benefit is a null"
        ),
    }


def _per_seed(results, seeds):
    rows = []
    for s in seeds:
        row = {"seed": s}
        for arm, r in results[s].items():
            row[arm] = {
                "accuracy": r.accuracy,
                "executed": r.executed,
                "attempted": r.attempted,
                "effective_count": r.effective_count,
                "effective_l1": r.effective_l1,
                "drift": {rec.statistic: rec.distance for rec in r.records},
                "direction": {rec.statistic: rec.direction for rec in r.records},
            }
        rows.append(row)
    return rows


def format_verdict(v):
    lines = ["SC-006 verdict: " + v["sc006"], ""]
    labels = {
        "a": "DRIFT (seeded moved, and red 1 fired on the pin)",
        "b": "BEATS RANDOM",
        "c": "NOT BLIND (the PLANTED control)",
        "d": "RED 2 FIRED (wrong underperforms random) and the WRONG arm was VALID",
        "e": "BEATS WRONG",
    }
    for k in "abcde":
        c = v["conjuncts"][k]
        num = ""
        if "t" in c:
            num = f"  t = {c['t']:+.4f}  permutation FPR = {c['fpr']:.4f}"
            if c["degenerate"]:
                num += "  [DEGENERATE: zero-variance differences]"
        lines.append(f"({k}) {labels[k]}: {c['state']}{num}")
        lines.append(f"     {c['note']}")
    lines.append("")
    lo = v["learner"]
    lines.append(
        f"the LEARNER's own criterion t(d_f) = {lo['t']:+.4f} -> {lo['disposition']}  ({lo['note']})"
    )
    lines.append("")
    lines.append(
        "per-seed table (accuracy | executed / attempted / effective / L1 | S1 / S2 drift):"
    )
    for row in v["per_seed"]:
        lines.append(f"  seed {row['seed']}")
        for arm in ("seeded", "random", "wrong", "planted", "frozen"):
            if arm not in row:
                continue
            r = row[arm]
            lines.append(
                f"    {arm:<8} acc {r['accuracy']:.4f} | {r['executed']} / {r['attempted']} / "
                f"{r['effective_count']} / {r['effective_l1']:.6f} | "
                f"{r['drift']['band_pair_weight_share']:.6f} / "
                f"{r['drift']['band_fanin_e_fraction']:.6f}"
            )
    return "\n".join(lines)


# =============================================================================================
# A synthetic artifact in the real format -- the offline self-check's population
# =============================================================================================
def synthetic_artifact(n_units=64, seed=0, out_degree=4):
    """A population in `h01_ingest`'s artifact format: signed units spread over the seven bands, a
    sparse laminar-biased edge set with integer synapse counts. No network, no lake, no artifact
    file -- the self-check must run anywhere, and an instrument validated only against the thing
    it was built to bless is not an instrument (specs/011 D4).

    `out_degree` = 4 (256 edges over 64 units) was CHOSEN by measurement, not by taste: at 2 the
    fixture's non-driven units have a fan-in of ~2, so the class that reaches a unit reaches it
    alone and EVERY arm's feature vector is a perfect class code -- the baseline sits at ceiling
    and a positive control has nothing to beat (measured over a density sweep: t(planted - random)
    = +0.66 at degree 2, +3.53 at 4, +2.82 at 8, +7.48 at 12, K = 4). Fan-in of ~4 dilutes each
    class's contribution enough that the base graph's task is genuinely hard, which is what makes
    PLANTED's advantage a reading rather than a formality. Disclosed as a fixture property; it
    changes nothing about the campaign's population."""
    import numpy as np

    rng = np.random.default_rng(int(seed) * 9_176 + 5)
    layer = np.sort(rng.integers(0, N_BANDS, size=n_units)).astype(np.int8)
    sign = np.where(rng.random(n_units) < 0.7, 1, -1).astype(np.int8)
    agglo = np.arange(1000, 1000 + n_units, dtype=np.int64)

    pairs = set()
    for u in range(n_units):
        #: a laminar bias -- a unit prefers targets in the next band up, which is what the
        #: direction-reversed arm has to have something to destroy
        for _ in range(out_degree):
            for _try in range(20):
                if rng.random() < 0.75:
                    want = (int(layer[u]) + 1) % N_BANDS
                    cand = np.nonzero(layer == want)[0]
                    if cand.size == 0:
                        cand = np.arange(n_units)
                    v = int(cand[rng.integers(0, cand.size)])
                else:
                    v = int(rng.integers(0, n_units))
                if v != u and (u, v) not in pairs:
                    pairs.add((u, v))
                    break
    edges = sorted(pairs)
    pre = np.asarray([a for a, _ in edges], dtype=np.int32)
    post = np.asarray([b for _, b in edges], dtype=np.int32)
    n_syn = rng.integers(1, 6, size=len(edges)).astype(np.int32)
    return {
        "agglo_id": agglo,
        "sign": sign,
        "layer": layer,
        "pre_idx": pre,
        "post_idx": post,
        "n_syn": n_syn,
    }


# =============================================================================================
# The offline self-check (run_selfchecks CORE): both reds, the positive control, the refusals
# =============================================================================================
def selfcheck(n_units=64, k=4, t_trial=20, trials_per_class=6, blocks=3, held_out=2, seeds=S_SEEDS):
    """The bare invocation: the whole pipeline on a synthetic population, under 60 s, no network.

    WHAT IS REAL AND WHAT IS CONSTRUCTED, stated plainly. RED 1 is REAL: the FROZEN arm is run at
    lr = 0 on the synthetic population and its verdict message names the pin the harness applied.
    The PLANTED control is REAL: the contrast is computed from the arms' own accuracies. RED 2's
    PREDICATE is exercised on a CONSTRUCTED accuracy set -- a synthetic table has no reason to
    make a reversed prior genuinely harmful, and a red must be shown to be reachable rather than
    waited for (P8 clause 1) -- while the WRONG arm's own VALIDITY PRECONDITION is measured on the
    synthetic table for real. The campaign is where red 2 is read evidentially.
    """

    t0 = time.perf_counter()
    art = synthetic_artifact(n_units=n_units, seed=0)
    results = {}
    valid = None
    for s in range(seeds):
        classes = drive_classes(art, k, s)
        tables = build_arms(art, s, classes)
        if valid is None:
            valid, d_wrong, d_random = wrong_validity(tables["wrong"], tables["random"], art)
        row = {}
        for name, table in tables.items():
            row[name] = run_arm(
                table,
                name,
                s,
                classes,
                lr=0.0 if name == "frozen" else None,
                blocks=blocks,
                t_trial=t_trial,
                trials_per_class=trials_per_class,
                held_out=held_out,
            )
        results[s] = row

    v = verdict(results, wrong_valid=valid)
    red1_fired = all(red1(results[s]["frozen"])[0] for s in results)
    red1_msg = red1(results[0]["frozen"])[1]
    planted = v["conjuncts"]["c"]["state"] == "MET"

    #: red 2's predicate, on the constructed input that makes it fire, and on one that must NOT
    fire_set = contrast(
        "constructed random - wrong", [0.20, 0.22, 0.18, 0.21, 0.19, 0.23, 0.20, 0.22]
    )
    null_set = contrast("constructed tie", [0.02, -0.03, 0.01, -0.02, 0.00, 0.03, -0.01, 0.02])
    red2 = bool(fire_set.met and not null_set.met)

    #: the refusals (2.2): an arm that violates a nuisance property never runs
    refusals = []
    for label, mutate in (
        ("sign flipped", lambda a: a.update(sign=-a["sign"])),
        (
            "edge dropped",
            lambda a: a.update(
                pre_idx=a["pre_idx"][:-1], post_idx=a["post_idx"][:-1], n_syn=a["n_syn"][:-1]
            ),
        ),
        ("multiset moved", lambda a: a.update(n_syn=a["n_syn"] + 1)),
    ):
        bad = _copy(art)
        mutate(bad)
        try:
            check_nuisance(bad, art, label)
            refusals.append((label, False))
        except ArmRefused:
            refusals.append((label, True))

    seconds = time.perf_counter() - t0
    return {
        "red1": bool(red1_fired),
        "red1_message": red1_msg,
        "red2": red2,
        "planted": bool(planted),
        "wrong_valid": bool(valid),
        "wrong_psi": float(d_wrong),
        "random_psi": float(d_random),
        "drift_control": all(drift_control_holds(results[s]["seeded"]) for s in results),
        "refusals": refusals,
        "verdict": v,
        "seconds": seconds,
        "accuracies": {
            arm: [float(results[s][arm].accuracy) for s in sorted(results)] for arm in results[0]
        },
    }


def main(argv=None):
    from snn_data_lake.substrate.home import throwaway_home

    argv = list(sys.argv[1:] if argv is None else argv)
    with throwaway_home():
        if "--pilot" in argv:
            return run_pilot()
        if "--campaign" in argv:
            return run_campaign()
        return _bare()


def _bare():
    print("=" * 78)
    print("prior drift harness -- seeded / random / wrong, with PLANTED and FROZEN")
    print("=" * 78)
    out = selfcheck()
    print(f"1 red 1 (the pin)     : fired = {out['red1']}")
    print(f"                        {out['red1_message']}")
    print(
        f"2 red 2 (predicate)   : reachable = {out['red2']} (constructed input; see the docstring)"
    )
    print(
        f"3 PLANTED control     : beats RANDOM = {out['planted']}  "
        f"(t = {out['verdict']['contrasts']['planted - random']['t']:+.4f})"
    )
    print(
        f"4 WRONG validity      : valid = {out['wrong_valid']}  PSI(wrong) "
        f"{out['wrong_psi']:.6f} vs PSI(random) {out['random_psi']:.6f}"
    )
    print(f"5 drift control       : SEEDED moved on every seed = {out['drift_control']}")
    print(f"6 nuisance refusals   : {out['refusals']}")
    print(f"7 accuracies          : {json.dumps(out['accuracies'])}")
    print("=" * 78)
    print(format_verdict(out["verdict"]))
    print("=" * 78)
    assert out["red1"], "red 1 must FIRE on the constructed pin"
    assert out["red2"], "red 2's predicate must be reachable and must not fire on a tie"
    assert out["planted"], (
        "the PLANTED positive control must beat RANDOM -- else the harness is blind"
    )
    assert out["drift_control"], "red 1's control failed: the SEEDED arm did not move"
    assert all(ok for _n, ok in out["refusals"]), (
        f"a nuisance refusal did not fire: {out['refusals']}"
    )
    assert out["seconds"] < 60.0, f"the offline self-check took {out['seconds']:.1f}s"
    print(
        f"prior_drift_harness self-check OK in {out['seconds']:.1f}s: red 1 fires on the "
        "constructed pin and names it, red 2's predicate is reachable and does not fire on a tie, "
        "the PLANTED control sees an advantage, the SEEDED arm moves on every seed, and every "
        "nuisance violation is refused before it can run"
    )
    return 0


# =============================================================================================
# The pilot (2.7) and the campaign (2.6)
# =============================================================================================
# distinct-from: substrate/kernel_plasticity.py::KernelPlasticity -- "beside every" there is the
#   eligibility array beside a bundle's qweight; here it is a host reading beside a timing number.
# distinct-from: cortex/h01_ingest.py::_lif_defaults -- imports the LIF constants from their
#   equation owner; no host, no process, no port.
# distinct-from: substrate/doc_budget_gate.py::_selfcheck,
#   hippocampus/spiking_storage_prototype.py::run, hypothalamus/__init__ -- a gate's constructed
#   reds, a prototype's returned criteria numbers and a region docstring; none reads a PROCESS.
#   All five read 2026-09-09.
def _host_condition():
    """The daemon's PID, working set and port state, recorded beside every timing number.

    The pre-registration's host clause also asks for a SETTLED boot burst, read as two consecutive
    90 s CPU windows within ~6 %. That reading is taken OUT OF BAND and recorded in the pin's
    `host_note`: it costs minutes, and on this host the daemon's STEADY state duty-cycles by more
    than 6 % between windows, so settling is read from the ABSENCE OF A TREND across several
    windows rather than from one pair. This function records the part a run can read in a second,
    which is the part that must sit beside every number."""
    import subprocess

    ps = (
        "$c = Get-NetTCPConnection -LocalPort 8137 -State Listen -ErrorAction SilentlyContinue; "
        "if ($c) { $p = Get-Process -Id $c.OwningProcess; "
        "'{0} {1} {2}' -f $c.OwningProcess, $p.WorkingSet64, 'LISTEN' } else { 'none 0 CLOSED' }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        pid, ws, port = out.split()
        return {"daemon_pid": pid, "daemon_working_set": int(ws), "port_8137": port}
    except Exception as exc:  # pragma: no cover - the host probe must never fail the run
        return {
            "daemon_pid": "unknown",
            "daemon_working_set": 0,
            "port_8137": f"probe failed: {exc}",
        }


def _tick_cost(art, classes, reps=3, trials=4, t_trial=T_TRIAL):
    """Seconds per tick at this width WITH THE LEARNER ON and the reward gate fired once per
    trial, so the projection's basis is the work the campaign actually does (Codex defect 11).

    MIN OF `reps`, not a single reading: the host runs a live daemon whose own duty cycle swings
    its CPU by more than 10 % between 90 s windows, and a single timing on a contended box is the
    instrument defect this project hunts. Each rep runs whole TRIALS with the per-trial reset, so
    the activity level the cost depends on is the campaign's, not a 200-tick free run's."""
    import numpy as np

    from snn_data_lake.cortex.h01_ingest import population
    from snn_data_lake.substrate.kernel_plasticity import KernelPlasticity

    state, _syn, kt = population(art)
    plast = KernelPlasticity(kt)
    amp_q = int(DRIVE_AMPLITUDE * float(state.arrays["thr"][0]) * (1 << plast.frac_bits))
    masks = [
        trial_drive(classes, i % len(classes), 0, i, state.n_units, t_trial) for i in range(trials)
    ]
    ext = np.zeros(state.n_units, dtype=np.int64)
    best = None
    for _rep in range(reps):
        t0 = time.perf_counter()
        for mask in masks:
            _reset_population(kt, plast)
            for tick in range(t_trial):
                ext[:] = 0
                ext[mask[tick]] = amp_q
                fired = kt.tick(ext)
                plast.update_eligibility(fired, float(tick + 1))
                plast.decay()
            plast.reward(+1.0)
        dt = (time.perf_counter() - t0) / (trials * t_trial)
        best = dt if best is None else min(best, dt)
    return best


def _campaign_ticks(k, blocks=C_BLOCKS, trials_per_class=TRIALS_PER_CLASS, t_trial=T_TRIAL):
    """5 tables x 8 seeds x (1 + C) blocks x K x trials x T_trial ticks (2.7's own formula)."""
    return 5 * S_SEEDS * (1 + blocks) * k * trials_per_class * t_trial


def run_pilot():
    """K by the 0.9 rule, the drive-only measurement, the tick cost at four widths with the
    learner ON, W by the 6-hour rule, and the WRONG validity precondition -- all recorded BEFORE
    the campaign, in `tools/t063_pilot.json`."""
    from snn_data_lake.cortex.h01_ingest import ARTIFACT, load_artifact

    host = _host_condition()
    print("=" * 78)
    print(f"T063 pilot -- host {json.dumps(host)}")
    print("=" * 78)
    art = load_artifact()
    art_sha = hashlib.sha256(Path(ARTIFACT).read_bytes()).hexdigest()
    print(
        f"artifact {ARTIFACT.name} sha256 {art_sha}  units {art['agglo_id'].size} "
        f"edges {art['pre_idx'].size}"
    )

    # ---- K: the smallest power of two >= 8 at which the NO-LEARNING RANDOM arm scores < 0.9 ----
    k_readings = []
    k_chosen = None
    for k in K_CANDIDATES:
        try:
            classes = drive_classes(art, k, 0)
        except ArmRefused as exc:
            k_readings.append({"k": k, "accuracy": None, "note": str(exc)})
            continue
        table = arm_random(art, 0)
        check_nuisance(table, art, "random", unique_edges=True)
        r = run_arm(table, "random", 0, classes, lr=0.0, blocks=1)
        k_readings.append({"k": k, "accuracy": r.accuracy})
        print(f"K rule: K = {k:>2}  no-learning RANDOM held-out accuracy = {r.accuracy:.4f}")
        if r.accuracy < K_CEILING:
            k_chosen = k
            break
    if k_chosen is None:
        print("PILOT REFUSED: no candidate K put the no-learning RANDOM arm under the ceiling")
        return 1
    classes = drive_classes(art, k_chosen, 0)

    # ---- T003's required measurement: a DRIVE-ONLY population must score at chance -------------
    no_edges = _copy(art)
    for key in ("pre_idx", "post_idx", "n_syn"):
        no_edges[key] = no_edges[key][:0]
    drive_only = run_arm(no_edges, "drive-only", 0, classes, lr=0.0, blocks=1)
    seeded_pilot = run_arm(arm_seeded(art), "seeded", 0, classes, blocks=1)
    chance = 1.0 / k_chosen
    print(
        f"T003 drive-only: accuracy {drive_only.accuracy:.4f} against chance {chance:.4f}; "
        f"the same protocol WITH edges (SEEDED) reads {seeded_pilot.accuracy:.4f}"
    )

    # ---- the WRONG arm's validity precondition, on the realised table --------------------------
    rnd = arm_random(art, 0)
    wrg = arm_wrong(art)
    valid, d_wrong, d_random = wrong_validity(wrg, rnd, art)
    print(
        f"T004 validity: PSI(wrong, seeded) {d_wrong:.6f} vs PSI(random, seeded) {d_random:.6f} "
        f"-> WRONG is {'VALID' if valid else 'INVALID'}"
    )

    # ---- the tick cost at four widths, learner ON, and the 6-hour projection -------------------
    total_ticks = _campaign_ticks(k_chosen)
    widths = []
    for frac in WIDTH_FRACTIONS:
        sub = band_balanced_subgraph(art, frac)
        try:
            sub_classes = drive_classes(sub, k_chosen, 0)
        except ArmRefused as exc:
            widths.append({"frac": frac, "units": int(sub["agglo_id"].size), "note": str(exc)})
            continue
        sec = _tick_cost(sub, sub_classes)
        hours = sec * total_ticks / 3600.0
        widths.append(
            {
                "frac": frac,
                "units": int(sub["agglo_id"].size),
                "edges": int(sub["pre_idx"].size),
                "seconds_per_tick": sec,
                "projected_hours": hours,
            }
        )
        print(
            f"width {frac:<6} units {sub['agglo_id'].size:>6} edges {sub['pre_idx'].size:>6}  "
            f"{sec * 1000:.4f} ms/tick  -> {hours:.3f} h for {total_ticks} ticks"
        )
    under = [w for w in widths if w.get("projected_hours", 1e9) < CAMPAIGN_HOURS]
    if not under:
        print(
            f"PILOT STOP: the projection exceeds {CAMPAIGN_HOURS} h at EVERY width -- the "
            "campaign is not run and the pre-registration's stop clause applies"
        )
        w_chosen = None
    else:
        w_chosen = max(under, key=lambda w: w["frac"])
        print(
            f"W = {w_chosen['frac']} ({w_chosen['units']} units): the LARGEST width whose measured "
            f"cost projects {w_chosen['projected_hours']:.3f} h < {CAMPAIGN_HOURS} h"
        )

    pin = {
        "task": "specs/022 T063 pilot",
        "git_head": _git_head(),
        "harness_sha256": _self_sha(),
        "artifact": str(ARTIFACT.name),
        "artifact_sha256": art_sha,
        "host": host,
        "host_note": HOST_NOTE,
        "constants": {
            "T_trial": T_TRIAL,
            "trials_per_class": TRIALS_PER_CLASS,
            "C_blocks": C_BLOCKS,
            "held_out_per_class": HELD_OUT_PER_CLASS,
            "class_frac": CLASS_FRAC,
            "p_drive": P_DRIVE,
            "drive_amplitude_x_thr": DRIVE_AMPLITUDE,
            "swap_passes": SWAP_PASSES,
            "S_seeds": S_SEEDS,
            "t_bar": T_BAR,
        },
        "K_rule": {"ceiling": K_CEILING, "readings": k_readings, "K": k_chosen},
        "drive_only": {
            "accuracy": drive_only.accuracy,
            "chance": chance,
            "seeded_same_protocol": seeded_pilot.accuracy,
        },
        "wrong_validity": {"valid": bool(valid), "psi_wrong": d_wrong, "psi_random": d_random},
        "campaign_ticks": total_ticks,
        "widths": widths,
        "W": w_chosen,
    }
    PILOT_PIN.parent.mkdir(parents=True, exist_ok=True)
    PILOT_PIN.write_text(json.dumps(pin, indent=2, sort_keys=True), encoding="utf-8")
    print(f"pilot pinned to {PILOT_PIN}")
    return 0 if w_chosen is not None else 2


def run_campaign():
    """Five tables x 8 seeds at the pilot's W, every number into `tools/t063_campaign.json`."""
    from snn_data_lake.cortex.h01_ingest import ARTIFACT, load_artifact

    if not PILOT_PIN.exists():
        print(
            "CAMPAIGN REFUSED: run --pilot first; W and K are read from its pin, never chosen here"
        )
        return 1
    pin = json.loads(PILOT_PIN.read_text(encoding="utf-8"))
    if pin.get("W") is None:
        print("CAMPAIGN REFUSED: the pilot recorded no width under the projection ceiling")
        return 1
    k = int(pin["K_rule"]["K"])
    frac = float(pin["W"]["frac"])
    host = _host_condition()
    print("=" * 78)
    print(f"T063 campaign -- K = {k}, W = {frac}, host {json.dumps(host)}")
    print("=" * 78)

    full = load_artifact()
    art = band_balanced_subgraph(full, frac)
    t0 = time.perf_counter()
    results = {}
    valid = None
    validity = {}
    for s in range(S_SEEDS):
        classes = drive_classes(art, k, s)
        tables = build_arms(art, s, classes)
        v_ok, d_wrong, d_random = wrong_validity(tables["wrong"], tables["random"], art)
        validity[s] = {"valid": bool(v_ok), "psi_wrong": d_wrong, "psi_random": d_random}
        valid = v_ok if valid is None else (valid and v_ok)
        row = {}
        for name, table in tables.items():
            r0 = time.perf_counter()
            row[name] = run_arm(table, name, s, classes, lr=0.0 if name == "frozen" else None)
            print(
                f"seed {s} {name:<8} acc {row[name].accuracy:.4f}  "
                f"exec {row[name].executed} att {row[name].attempted} "
                f"eff {row[name].effective_count} L1 {row[name].effective_l1:.6f}  "
                f"[{time.perf_counter() - r0:.1f}s]"
            )
        results[s] = row
    wall = time.perf_counter() - t0

    v = verdict(results, wrong_valid=valid)
    print("=" * 78)
    print(format_verdict(v))
    print("=" * 78)
    print(f"wall clock {wall:.1f}s ({wall / 3600.0:.3f} h)")

    out = {
        "task": "specs/022 T063 campaign",
        "git_head": _git_head(),
        "harness_sha256": _self_sha(),
        "artifact": str(ARTIFACT.name),
        "artifact_sha256": hashlib.sha256(Path(ARTIFACT).read_bytes()).hexdigest(),
        "host": host,
        "host_note": HOST_NOTE,
        "K": k,
        "W_frac": frac,
        "W_units": int(art["agglo_id"].size),
        "W_edges": int(art["pre_idx"].size),
        "wrong_validity_per_seed": validity,
        "verdict": v,
        "red1_messages": {s: red1(results[s]["frozen"])[1] for s in results},
        "drift_records": {
            s: {a: [r._asdict() for r in results[s][a].records] for a in results[s]}
            for s in results
        },
        "wall_seconds": wall,
    }
    CAMPAIGN_PIN.parent.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_PIN.write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"campaign pinned to {CAMPAIGN_PIN}")
    return 0


def _self_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _git_head():
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
