r"""G1(c) / T109 part (c): carried state through the selected pool -- the campaign module's frozen
constants and pure decision functions.

G1(c) asks one question of the pool G1(b) selected (g1, an LSM): does a pool that CARRIES its state across
percepts predict the next percept better than the same pool reset and settled on each percept
(D = accuracy(Carried) - accuracy(Settled), both through the fixed reference readout), over S = 31 live
recordings? The frozen text is specs/022-unified-brain-arc/evidence/g1c-carried-state.md (pre-registered
before any build; the PLAN-FINAL G workstream's G1 / T109 row, its dated notes and Part F's G-R1 and G-R2).
It inherits g1a-causal-contract.md and g1b-pool-selection.md and may only strengthen them.

This module (arc P1-C) holds what that text freezes and what decides without touching a recording:

  constants          section 2.2 (the tuned configuration), 2.3 (the bar, Delta, the skew bound) and 2.12
                     (every seed set), tied at import to the code they restate
  validate_seed_set  a duplicate, an exposed seed, or a reserve seed in a live set is refused (2.12, 2.11)
  contrast_row       section 2.3's one contrast row: exactly 31 differences, `paired_contrast.contrast` for
                     t and degeneracy, the exact sign-flip rate and its band, the skew disclosure; MET only
                     if non-degenerate AND t > the bar
  oracle_check       section 2.4 (ii): accuracy(f64) - accuracy(a) through the reference readout decides,
                     the grower's reading of the same difference is reported beside it

The campaign stages (record ... report, section 2.13) arrive in P5, added to this module; it imports
`pool_selection`'s step, job and table machinery for them (section 2.14).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.carried_state_campaign

Pure standard library beyond the instruments it restates; deterministic; ASCII-only prints.
"""

import math
import random

from snn_data_lake.instruments import paired_contrast as pc
from snn_data_lake.instruments import pool_selection as ps
from snn_data_lake.reservoir import recurrent_pool as rp

# distinct-from: instruments/pool_selection.py::module -- G1(b)'s selection campaign answers another question (which pool, on tuning seeds, in ten steps); this module only IMPORTS its power_s, SIGMA_PROXY, DELTA and seed constants to tie G1(c)'s frozen text to them, and adds the seed-set refusal, the contrast row and the oracle check that module has none of
# distinct-from: instruments/paired_contrast.py::contrast -- the row CALLS contrast (for t and degeneracy) and permutation_fpr_detail, never reimplements them; it adds 2.3's decision rule (met = non-degenerate AND t > bar, which contrast's own .met is not), the 31-value halt, the skewness and the sign counts

# ---- section 2.12: the seeds ---------------------------------------------------------------------
# the primary's 31 live recordings (2.12), S = power_s(0.0436)["S"]
LIVE_SEEDS = tuple(range(300, 331))
RESERVE_SEEDS = tuple(range(331, 362))  # the repair re-run's live set after inspection (2.11)
IDENTITY_SEED = ps.PILOT_SEED  # R-b8's stream-identity recording, seed 100 (2.6, 2.12)
# R-b1's recording, a G1(b) tuning seed read only by the causality check (2.10)
CAUSALITY_SEED = 203
# 204-207: the migration gate's live seeds (2.9 (ii)), exposed by G1(b)
GATE_SEEDS = ps.SELECT_SEEDS
# 5005-5008: the migration gate's D_task seeds (2.9 (ii)), exposed by G1(b)
GATE_SYNTHETIC_SEEDS = ps.SELECT_TASK_SEEDS
RB6_SEED = 6000  # R-b6 on a synthetic seed no G1(b) record names (2.4 (i)); G1(b)'s was 4242
RB6_RESERVE_SEED = 6001  # the synthetic reserve for R-b6 (2.11)
SHUFFLE_RED_SEEDS = tuple(range(6100, 6108))  # the shuffle's planted order-independent gain (2.10)
G2II_CHECK_SEEDS = tuple(range(7000, 7008))  # G-R2's construction-check seeds (2.8)
G2II_DEV_SEEDS = tuple(range(7100, 7104))  # G2(ii)'s learner development seeds (2.8)
# Exposed by G1(a) and G1(b), never confirmatory (2.12): 0-7, 100-103, 200-211, 4242-4247, 5001-5012.
EXPOSED_SEEDS = frozenset(
    (*range(0, 8), *range(100, 104), *range(200, 212), *range(4242, 4248), *range(5001, 5013))
)

# ---- sections 2.2 and 2.3: the configuration, the bar, the design effect -------------------------
TUNED = {"k": 8, "w": 0.25, "lambda": 0.9, "g": 4, "T_in": 1}  # 2.2: g1 LSM as G1B 3.4 tuned it
T_CRIT = pc.T_BAR_S31  # 2.3: t_quantile(0.95, 30) = 1.6972609 rounded UP at the fourth place
SKEW_BOUND = -1.0  # 2.3: a met contrast whose skewness is below this reads "met, skew-flagged"
N_LIVE = len(LIVE_SEEDS)  # S = 31: the number of differences every contrast row takes (2.3)
DELTA = ps.DELTA  # 2.3: the design effect (0.02), reported against, never an acceptance threshold

assert len(LIVE_SEEDS) == ps.power_s(ps.SIGMA_PROXY)["S"], "S is power_s(0.0436)['S'] (2.12)"
assert len(RESERVE_SEEDS) == len(LIVE_SEEDS), "the reserve repeats the primary's S (2.11)"
assert not set(LIVE_SEEDS) & set(RESERVE_SEEDS), "live and reserve seeds must be disjoint (2.12)"
assert not set(LIVE_SEEDS) & EXPOSED_SEEDS, "a live seed is never an exposed seed (2.12)"
assert not set(RESERVE_SEEDS) & EXPOSED_SEEDS, "a reserve seed is never an exposed seed (2.12)"
_OTHER_FRESH = (RB6_SEED, RB6_RESERVE_SEED, *SHUFFLE_RED_SEEDS, *G2II_CHECK_SEEDS, *G2II_DEV_SEEDS)
assert len(set(_OTHER_FRESH)) == len(_OTHER_FRESH), "the synthetic and G2(ii) seeds are distinct"
assert not set(_OTHER_FRESH) & (EXPOSED_SEEDS | set(LIVE_SEEDS) | set(RESERVE_SEEDS)), (
    "the synthetic and G2(ii) seeds are fresh (2.12)"
)
assert (IDENTITY_SEED, GATE_SEEDS, GATE_SYNTHETIC_SEEDS) == (
    100,
    (204, 205, 206, 207),
    (5005, 5006, 5007, 5008),
), "the identity and gate seeds are G1(b)'s (2.12)"
assert (T_CRIT, DELTA) == (1.6973, 0.02), "the bar and Delta are the frozen numbers (2.3)"
rp.canonical("lsm", TUNED)  # raises ValueError unless TUNED lies in section 2.3's grid of G1B


# distinct-from: sensorimotor/rsnn_encoder.py -- load_weights validates a weights file's shapes (like global_workspace._deliver's tuple and prior_drift_harness.run_campaign's seeds, none checks a seed list); this is the one structural seed-list check
def distinct_seeds(seeds):
    """The seeds as a non-empty tuple of distinct ints, or ValueError: the structural half of every seed
    list this campaign reads (`validate_seed_set` here; `inseparable_mapping`'s A3 / A4 / construction
    check import it). Fold of the P2 panel's F2: an empty list had made A4 vacuously ok, and eight copies
    of one seed had read as eight measurements."""
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("no seeds: a check over an empty seed list proves nothing")
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise ValueError(f"seeds must be ints, got {seeds!r}")
    duplicated = sorted({s for s in seeds if seeds.count(s) > 1})
    if duplicated:
        raise ValueError(f"duplicate seeds {duplicated}")
    return seeds


def validate_seed_set(seeds, *, allow_reserve=False):
    """The seeds as a tuple, or ValueError: a duplicate, an exposed seed (2.12: exposed seeds are never
    confirmatory), or a reserve seed unless `allow_reserve` (2.11: a reserve seed enters a primary
    table only in the repair re-run)."""
    seeds = distinct_seeds(seeds)
    exposed = sorted(set(seeds) & EXPOSED_SEEDS)
    if exposed:
        raise ValueError(f"exposed seeds {exposed} are never confirmatory (g1c 2.12)")
    reserve = sorted(set(seeds) & set(RESERVE_SEEDS))
    if reserve and not allow_reserve:
        raise ValueError(f"reserve seeds {reserve} are not live seeds (g1c 2.11)")
    return seeds


def skewness(values):
    """The sample skewness g1 = m3 / m2**1.5, with the biased moments m_k = mean((x - mean(x))**k);
    0.0 when m2 == 0 (a constant sample, and an empty one). Section 2.3's disclosure statistic."""
    xs = [float(v) for v in values]
    if len(set(xs)) <= 1:
        return 0.0
    n = len(xs)
    mean = math.fsum(xs) / n
    m2 = math.fsum((x - mean) ** 2 for x in xs) / n
    m3 = math.fsum((x - mean) ** 3 for x in xs) / n
    return m3 / m2**1.5 if m2 > 0.0 else 0.0


def sign_counts(values):
    """How many values are positive, negative and exactly zero (reported beside every contrast; sign
    balance is never a decision rule, Part F)."""
    xs = [float(v) for v in values]
    return {
        "positive": sum(x > 0.0 for x in xs),
        "negative": sum(x < 0.0 for x in xs),
        "zero": sum(x == 0.0 for x in xs),
    }


def contrast_row(name, diffs, *, bar=T_CRIT, n_required=N_LIVE):
    """Section 2.3's contrast row over exactly `n_required` per-seed differences, as a plain dict.

    A shorter or longer list (a TIMEOUT, a missing cell) or a non-finite difference raises ValueError:
    the campaign halts, it never averages over what completed. t and the degeneracy come from
    `paired_contrast.contrast`; the exact sign-flip rate and its band population come from
    `permutation_fpr_detail` even for a degenerate row. **met = (not degenerate) and t > bar** -- NOT
    `contrast(...).met`, which reads a zero-variance positive sample as met (kept for its n = 8
    callers). skew_flagged = met and skewness < SKEW_BOUND. sd is the n - 1 sample sd and
    lower_bound = mean - bar * sd / sqrt(n), the one-sided 95% bound reported against delta."""
    try:
        diffs = tuple(float(d) for d in diffs)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: a difference is not a number ({error})") from error
    n = len(diffs)
    if n != n_required or n < 2:
        raise ValueError(
            f"{name}: {n} differences, {n_required} required (a short list halts, never averages "
            "over what completed; g1c 2.3)"
        )
    if not all(map(math.isfinite, diffs)):
        raise ValueError(f"{name}: a difference is NaN or infinite")
    c = pc.contrast(name, diffs, bar)
    fpr, band = pc.permutation_fpr_detail(diffs, bar)
    if c.degenerate:
        mean, sd = diffs[0], 0.0  # every value equal: exact, whatever the rounding of the sum
    else:
        mean = sum(diffs) / n
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    skew = skewness(diffs)
    met = (not c.degenerate) and c.t > bar
    return {
        "name": name,
        "n": n,
        "mean": mean,
        "sd": sd,
        "t": c.t,
        "bar": bar,
        "lower_bound": mean - bar * sd / math.sqrt(n),
        "fpr": fpr,
        "band_population": band,
        "skewness": skew,
        "signs": sign_counts(diffs),
        "degenerate": c.degenerate,
        "met": met,
        "skew_flagged": met and skew < SKEW_BOUND,
        "delta": DELTA,
    }


def _accuracy(table, seed, key, who):
    """table[seed][key] as a finite float, or ValueError('blind harness: oracle incomplete ...')."""
    try:
        value = float(table[seed][key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"blind harness: oracle incomplete -- {who}, seed {seed}, {key!r}: {error!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"blind harness: oracle incomplete -- {who}, seed {seed}, {key!r} is {value}"
        )
    return value


# distinct-from: substrate/minhash.py -- exact_jaccard (like frequency_moment.exact_f2 and snn_mnist_stdp_genn's codegen check) is a self-check ground truth; this is G1(c)'s seed-paired (f64)-(a) contrast over replay accuracies
def oracle_check(readout, grower, *, seeds=LIVE_SEEDS, allow_reserve=False):
    """Section 2.4 (ii): per seed, accuracy(f64) - accuracy(a) through the reference readout, one
    contrast row at the bar; the grower's reading of the same difference beside it.

    `readout` and `grower` map seed -> {"f64": accuracy, "a": accuracy}. A missing seed or key, or a
    value that is None, NaN or infinite, raises ValueError('blind harness: oracle incomplete ...'):
    G-R1 then reads REFUSED. Only the decoder row decides: the grower's rows (G1A read f64 above a
    through the GROWER in 12 of 12 cells) never stand in for the decoder's."""
    # the seed set is validated first (fold of the P1 panel's F1: 30 copies of 300 plus 301 had read n 31, MET):
    # a duplicate or an exposed seed is refused, a reserve seed only with the repair run's authorization (2.11)
    seeds = validate_seed_set(tuple(seeds), allow_reserve=allow_reserve)
    rows = {}
    for who, table in (("reference readout", readout), ("grower", grower)):
        diffs = [_accuracy(table, s, "f64", who) - _accuracy(table, s, "a", who) for s in seeds]
        rows[who] = contrast_row(f"(f64) - (a) through the {who}", diffs)
    decoder = rows["reference readout"]
    return {"decoder": decoder, "grower": rows["grower"], "met": decoder["met"]}


# ---- the self-check's hand fixtures --------------------------------------------------------------
FIXTURE_RNG_SEED = 20261003  # an RNG seed for the fixtures below, not a recording seed
# 2.10's named symmetric zero-mean fixture: 15 pairs of +0.02 / -0.02 and one 0.0 (interleaved so the
# sum is exactly 0.0); non-degenerate, t = 0, and it must read NOT MET (sign balance is no decision rule)
SYMMETRIC_ZERO_MEAN = (0.02, -0.02) * 15 + (0.0,)


def _jittered(center, half, salt):
    """31 differences center + uniform(-half, half), from a fixed RNG (a hand fixture, not a stream)."""
    rng = random.Random(FIXTURE_RNG_SEED + salt)
    return [center + rng.uniform(-half, half) for _ in LIVE_SEEDS]


def _left_tail():
    """A met vector with a long left tail: 30 values near +0.05 and one at -0.30."""
    return [*_jittered(0.05, 0.005, 3)[:30], -0.30]


def _oracle_table(diffs):
    """seed -> {"f64", "a"} whose difference is `diffs` (a constant base accuracy 0.5, so equal
    differences stay equal to the last bit)."""
    return {s: {"f64": 0.5 + d, "a": 0.5} for s, d in zip(LIVE_SEEDS, diffs, strict=True)}


def _raises(function, *args, **kwargs):
    """The ValueError message `function(*args, **kwargs)` raises; AssertionError if it does not raise."""
    try:
        function(*args, **kwargs)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{function.__name__}{args!r} did not raise ValueError")


def main():
    """The self-check: every function above on hand fixtures, each with its red. No argv, no files, no
    cwd: the fixtures are built in memory."""
    # seeds: [300..329, 203] carries an exposed seed; a reserve seed is refused unless allowed
    assert validate_seed_set(LIVE_SEEDS) == LIVE_SEEDS == tuple(range(300, 331))
    assert "exposed" in _raises(validate_seed_set, [*range(300, 330), 203])
    assert "exposed" in _raises(validate_seed_set, [300, 5005])
    assert "reserve" in _raises(validate_seed_set, [300, 331])
    assert validate_seed_set([300, 331], allow_reserve=True) == (300, 331)
    assert "duplicate" in _raises(validate_seed_set, [300, 301, 300])
    # statistics: the formulas on hand values
    assert skewness([1.0, 2.0, 3.0]) == 0.0 and skewness([5.0] * 31) == 0.0 and skewness([]) == 0.0
    assert math.isclose(skewness([0.0, 0.0, 3.0]), 2.0 / 2.0**1.5, rel_tol=1e-12)  # m2 = 2, m3 = 2
    assert sign_counts([1.0, -1.0, 0.0, 0.0, 2.0]) == {"positive": 2, "negative": 1, "zero": 2}
    # contrast_row: 30 (and 32) values halt, a NaN halts; a green row reports every field
    green = _jittered(0.05, 0.01, 1)
    assert "30 differences" in _raises(contrast_row, "short", green[:30])
    assert "32 differences" in _raises(contrast_row, "long", [*green, 0.05])
    assert "NaN" in _raises(contrast_row, "nan", [*green[:30], math.nan])
    row = contrast_row("green", green)
    assert row["met"] and not row["degenerate"] and not row["skew_flagged"] and row["n"] == 31
    assert row["t"] > T_CRIT and row["bar"] == T_CRIT and row["delta"] == DELTA
    assert math.isclose(row["sd"], math.sqrt(sum((d - row["mean"]) ** 2 for d in green) / 30))
    assert math.isclose(row["lower_bound"], row["mean"] - T_CRIT * row["sd"] / math.sqrt(31))
    assert row["signs"] == {"positive": 31, "negative": 0, "zero": 0} and abs(row["skewness"]) < 1.0
    assert 0.0 < row["fpr"] < 0.10, row["fpr"]
    # skew: a met vector with a long left tail is skew_flagged; the same tail shifted below zero is
    # not met, so it is not flagged (skew_flagged = met AND skewness < SKEW_BOUND); a symmetric met
    # vector (green, above) is not flagged
    tail = contrast_row("left tail", _left_tail())
    assert tail["met"] and tail["skewness"] < SKEW_BOUND and tail["skew_flagged"], tail
    shifted = contrast_row("left tail below zero", [d - 0.1 for d in _left_tail()])
    assert shifted["skewness"] < SKEW_BOUND and not shifted["met"], shifted
    assert not shifted["skew_flagged"], shifted
    # the oracle (2.4 (ii)): the grower reads f64 > a, the decoder does not -- the decoder-not-grower red
    reds = oracle_check(
        _oracle_table(_jittered(-0.005, 0.005, 2)), _oracle_table(_jittered(0.05, 0.01, 4))
    )
    assert reds["grower"]["met"] and not reds["decoder"]["met"] and not reds["met"], reds
    assert reds["decoder"]["t"] < 0.0
    # the decoder 0.05 +- 0.01 reads MET, with its exact rate and band printed below
    oracle = oracle_check(
        _oracle_table(_jittered(0.05, 0.01, 5)), _oracle_table(_jittered(0.05, 0.01, 6))
    )
    decoder = oracle["decoder"]
    assert oracle["met"] and decoder["met"] and oracle["grower"]["met"], oracle
    assert 0.0 < decoder["fpr"] < 0.10, decoder["fpr"]
    # sign balance is no decision rule: 15 pairs of +-0.02 and one 0.0 is NOT MET, through both paths
    assert len(SYMMETRIC_ZERO_MEAN) == 31 and sum(SYMMETRIC_ZERO_MEAN) == 0.0
    symmetric = contrast_row("symmetric", SYMMETRIC_ZERO_MEAN)
    assert not symmetric["met"] and not symmetric["degenerate"], symmetric
    assert symmetric["signs"] == {"positive": 15, "negative": 15, "zero": 1}
    assert abs(symmetric["t"]) < 1e-12
    assert not oracle_check(_oracle_table(SYMMETRIC_ZERO_MEAN), _oracle_table(green))["met"]
    # all zero: degenerate, NOT MET, and no flips to count
    zero = oracle_check(_oracle_table([0.0] * 31), _oracle_table(green))["decoder"]
    assert zero["degenerate"] and not zero["met"] and zero["fpr"] == 0.0
    assert zero["band_population"] == 0
    # constant +0.03: contrast()'s own rule says MET (its n = 8 callers' rule); 2.3's rule reads NOT MET
    constant = [0.03] * 31
    assert pc.contrast("constant", constant, T_CRIT).met is True  # the red this rule exists for
    flat = oracle_check(_oracle_table(constant), _oracle_table(green))["decoder"]
    assert flat["degenerate"] and not flat["met"] and not flat["skew_flagged"], flat
    assert 0.0 < flat["fpr"] < 0.5 and flat["sd"] == 0.0 and flat["skewness"] == 0.0, flat
    # one seed missing, a None, a NaN, a missing key: the oracle is incomplete, the harness blind
    complete = _oracle_table(green)
    assert "blind harness: oracle incomplete" in _raises(
        oracle_check, {s: v for s, v in complete.items() if s != 315}, complete
    )
    assert "blind harness: oracle incomplete" in _raises(
        oracle_check, complete, {**complete, 330: {"f64": None, "a": 0.5}}
    )
    assert "blind harness: oracle incomplete" in _raises(
        oracle_check, {**complete, 300: {"f64": math.nan, "a": 0.5}}, complete
    )
    assert "blind harness: oracle incomplete" in _raises(
        oracle_check, complete, {**complete, 301: {"a": 0.5}}
    )
    print(
        "self-check OK: constants tied at import (S = 31, live/reserve/exposed disjoint, TUNED in the "
        f"grid); decoder 0.05 +- 0.01 MET at t {decoder['t']:.4f} > {T_CRIT} (exact FPR "
        f"{decoder['fpr']:.6f}, band {decoder['band_population']} flips); decoder-not-grower red: "
        f"decoder NOT MET at t {reds['decoder']['t']:.4f} while the grower is MET at t "
        f"{reds['grower']['t']:.4f}; symmetric zero-mean NOT MET (t {symmetric['t']:.4f}, FPR "
        f"{symmetric['fpr']:.6f}); all-zero degenerate NOT MET; constant +0.03 degenerate NOT MET "
        f"where contrast() says met (FPR {flat['fpr']:.6f}, band {flat['band_population']}); "
        f"left-tailed met vector skew-flagged (g1 {tail['skewness']:.4f}, t {tail['t']:.4f}); "
        "30 values, exposed, reserve and duplicate seeds, a missing, None or NaN oracle cell all refused"
    )


if __name__ == "__main__":
    main()
