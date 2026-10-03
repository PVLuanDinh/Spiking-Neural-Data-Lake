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

Arc P3b adds section 2.6's within-block time shuffle and section 2.5's reading of the primary:

  shuffle_blocks     the shuffle as a PURE stream transform (percept fields move inside a block, slot fields
                     and sleep rows stay); the campaign writes each shuffled stream to disk and replays it
                     as an ordinary stream, so replay has no new path
  check_shuffle_transform, settled_invariance
                     the transform's checks, and 2.6's Settled invariance over two replay results
  survives, incremental
                     "survives" = contrast(D_shuf) met at T_CRIT; contrast(D - D_shuf) is reported
  g_r1_reading       2.5's table, rows top to bottom, as a pure function
  _check_pool_controls, _check_decoder_decisions
                     2.10's shuffle controls on REAL replays, beside the arithmetic checks: Settled
                     invariance on pool replays with its Carried and skip-a-carrier reds, and "survives"
                     on a planted current-category gain and on R-b6's history gain through the real
                     reference readout (seeds 6100-6107 at Part F's bar)

Arc P4b adds section 2.9's migration gate as two pure functions (the checks of the hosted form itself,
R-b3 / R-b4 / R-b5 with their reds, live in reservoir/recurrent_pool.py):

  within             |d_host - d_port| <= 0.005, INCLUSIVE; None, NaN, an infinity or a non-number is
                     False, never a pass
  migration_reading  "hosted" only when the parity fixtures, BOTH D conjuncts (live D, D_task) and R-b3
                     and R-b4 all hold, else "port"

SIZE: this module is about 1,200 lines, past the 800-line soft ceiling, on purpose: it is the one place
the campaign's frozen decisions and their self-checks are tied to the frozen text (the stages are not
here: they are carried_state_driver's).

The 48-event self-check fixture is shorter than one 256-event block, so its shuffles use blocks of 8
(FIXTURE_BLOCK); every function takes the block size and defaults to SHUFFLE_BLOCK.

The campaign stages (record ... report, section 2.13) are `carried_state_driver`'s (P5): it imports this
module's constants, decisions and transforms and `pool_selection`'s step, job and table machinery
(section 2.14).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.carried_state_campaign

Pure standard library beyond the instruments it restates; deterministic; ASCII-only prints.
"""

import contextlib
import copy
import functools
import itertools
import json
import math
import random
import statistics
import zlib
from collections import Counter

import numpy as np

from snn_data_lake.instruments import memory_diagnostic as md
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


# ---- section 2.6: the within-block time shuffle (a pure stream transform) --------------------------
SHUFFLE_BLOCK = 256  # 2.6: a block is 256 consecutive events by index within one segment (packs are 256-periodic)
SHUFFLE_K = 4  # 2.6: shuffles per recording
# 2.6's TAG: the fixed middle word of SeedSequence([recording_seed, TAG, k]); the ASCII bytes of "SHUF".
# Frozen with the draws the campaign asserts before its first recording; never changed after them.
SHUFFLE_TAG = 0x53485546
# 2.6: a percept's fields move together; the slot's fields stay (|LP| is a property of the time axis). Together
# they are every key memory_diagnostic.record_stream writes to an event plus the "segment" _segments adds (a
# test ties the two sets to that code): an event key in neither set refuses the transform.
PERCEPT_FIELDS = ("x", "c", "source", "attended", "prediction", "fallback")
SLOT_FIELDS = ("t", "tick", "segment", "causal_lp")
_KNOWN_FIELDS = frozenset((*PERCEPT_FIELDS, *SLOT_FIELDS))
_BOUND_KEYS = (
    "seed",
    "config",
    "policy",
    "final",
)  # a shuffled stream keeps the recording's binding keys
FIXTURE_BLOCK = (
    8  # the 48-event fixture is shorter than one SHUFFLE_BLOCK: its shuffles use blocks of 8
)


# distinct-from: instruments/memory_diagnostic.py::_source_digest -- that digests events + sleep to one crc32; this is the canonical JSON string itself, so the multiset comparisons below are exact (a crc would make them probabilistic). reservoir/recurrent_pool.py::crc_json is likewise a crc32
def _canon(value):
    """Canonical JSON: the form memory_diagnostic's source digest and every comparison below use."""
    return json.dumps(value, sort_keys=True)


def _percept(event):
    return {key: event[key] for key in PERCEPT_FIELDS if key in event}


def _slots(event):
    return _canon({key: event[key] for key in SLOT_FIELDS if key in event})


def _pairs(rows, group):
    """The sorted canonical (x, c) pairs of a block."""
    return sorted(_canon([rows[i].get("x"), rows[i].get("c")]) for i in group)


def _percepts(rows, group):
    """The sorted canonical whole percepts of a block."""
    return sorted(_canon(_percept(rows[i])) for i in group)


# distinct-from: instruments/inseparable_mapping.py::_shuffled -- re-draws a taught-label arrangement until it differs, from a random.Random; this is the numpy SeedSequence([recording_seed, TAG, k]) generator 2.6 freezes, whose draws are asserted and never re-drawn
def shuffle_seed(recording_seed, k):
    """The generator of 2.6's permutation draws: default_rng(SeedSequence([recording_seed, TAG, k]))."""
    return np.random.default_rng(np.random.SeedSequence([recording_seed, SHUFFLE_TAG, k]))


def blocks(stream, block=SHUFFLE_BLOCK):
    """2.6's blocks as lists of event indices: within each segment (an event whose segment is None, x_1,
    is in no block) consecutive groups of `block` by index within the segment; a final partial group is
    its own block."""
    if not isinstance(block, int) or isinstance(block, bool) or block < 1:
        raise ValueError(f"the block size must be a positive int, got {block!r}")
    by_segment = {}
    for index, event in enumerate(stream["events"]):
        if event.get("segment") is not None:
            by_segment.setdefault(event["segment"], []).append(index)
    return [
        indices[start : start + block]
        for indices in by_segment.values()
        for start in range(0, len(indices), block)
    ]


def block_permutations(recording_seed, k, sizes):
    """One permutation per block size, drawn in order from `shuffle_seed(recording_seed, k)`; permutation[i]
    is the block position whose percept lands at position i. 2.6: a block of two or more events moves at
    least half its events, ASSERTED on the draw (AssertionError) and never re-drawn; the campaign asserts
    it on its seeds before the first recording."""
    rng = shuffle_seed(recording_seed, k)
    drawn = []
    for size in sizes:
        permutation = rng.permutation(size).tolist()
        moved = sum(permutation[i] != i for i in range(size))
        if size >= 2 and 2 * moved < size:
            raise AssertionError(
                f"shuffle (seed {recording_seed}, k {k}): a block of {size} moves {moved} events, "
                "under half (g1c 2.6; never re-drawn)"
            )
        drawn.append(permutation)
    return drawn


def _pass_problems(events, groups):
    """Why a block is not one contiguous pass (empty if all are). A source pointer is the brain's frame
    cursor at an acquisition and advances by one per acquisition, so a block of consecutive deliveries is
    one contiguous pass exactly when its events are consecutive in the recording and, where they carry
    sources, each modality's pointers are consecutive integers. A reflex tick acquires without delivering
    (memory_diagnostic._segments), so a live gap is refused here, never shuffled across."""
    problems = []
    for group in groups:
        start = group[0]
        if group[-1] - start != len(group) - 1:
            problems.append(f"block at event {start}: events are not consecutive in the recording")
            continue
        sources = [events[i].get("source") for i in group]
        if all(source is None for source in sources):
            continue
        for modality in ("vision", "audio"):
            try:
                pointers = [source[modality]["pointer"] for source in sources]
                gapless = pointers == [pointers[0] + j for j in range(len(pointers))]
            except (KeyError, TypeError):
                gapless = False
            if not gapless:
                problems.append(
                    f"block at event {start}: {modality} pointers are not one contiguous pass"
                )
    return problems


# distinct-from: reasoning/arc_agi/rule_inducer.py::shuffle_transitions -- permutes ACTION labels across a whole transition list as a control; this moves a recorded percept's fields between events inside one 256-event block of a recording, keeps slot fields, sleep rows and the revision binding, and refuses an unknown key or a non-contiguous block
def shuffle_blocks(stream, k, *, block=SHUFFLE_BLOCK):
    """2.6's shuffle k of a recording: a deep copy in which, inside every block, each event carries the
    PERCEPT_FIELDS of another event of that block (a permutation drawn by `block_permutations`) while
    every SLOT_FIELD, every event in no block and every sleep row stays in place. The copy keeps the
    recording's seed, config, policy and final (the revision binding holds) and gains
    "shuffle": {"seed", "block", "k"}; the campaign writes it to disk and replays it as an ordinary
    stream. ValueError refuses an event key in neither field set, an already shuffled stream, a stream
    with no block, and a block that is not one contiguous pass."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError(f"k must be a non-negative int, got {k!r}")
    if "shuffle" in stream:
        raise ValueError("REFUSED: the stream is already shuffled")
    for index, event in enumerate(stream["events"]):
        extra = sorted(set(event) - _KNOWN_FIELDS)
        if extra:
            raise ValueError(
                f"REFUSED: event {index} carries the key {extra[0]!r}, in neither PERCEPT_FIELDS nor "
                "SLOT_FIELDS (g1c 2.6)"
            )
    groups = blocks(stream, block)
    if not groups:
        raise ValueError("REFUSED: no segment events: there is no block to shuffle")
    problems = _pass_problems(stream["events"], groups)
    if problems:
        raise ValueError(f"REFUSED: {problems[0]}")
    shuffled = copy.deepcopy(stream)
    permutations = block_permutations(stream["seed"], k, [len(group) for group in groups])
    for group, permutation in zip(groups, permutations, strict=True):
        held = [_percept(shuffled["events"][i]) for i in group]
        for slot, donor in zip(group, permutation, strict=True):
            event = shuffled["events"][slot]
            for key in PERCEPT_FIELDS:
                event.pop(key, None)
            event.update(held[donor])
    shuffled["shuffle"] = {"seed": stream["seed"], "block": block, "k": k}
    return shuffled


def check_shuffle_transform(stream, shuffled, block=SHUFFLE_BLOCK):
    """The transform's own check, as {"ok", "reasons"} (every failed reason, none raised): every event key
    known; per block the multiset of (x, c) pairs and of whole percepts equal; slot fields, events in no
    block and the whole sleep list equal to the original; the revision-binding keys kept; the shuffle
    declared; the digest different; at least half of each block of two or more moved; and `shuffled` is
    exactly what `shuffle_blocks` returns for its declared k. Only the declared k is ever drawn: a
    neighbouring k is not a registered shuffle, so a stream is never failed for what k + 1 would do
    (that k drives the draw is shown on the fixture, by the self-check and its test)."""
    events, mixed = stream["events"], shuffled.get("events")
    if not isinstance(mixed, list) or len(mixed) != len(events) or "sleep" not in shuffled:
        return {"ok": False, "reasons": ["the shuffled stream lacks the events or the sleep rows"]}
    reasons = []
    for name, rows in (("original", events), ("shuffled", mixed)):
        for index, event in enumerate(rows):
            extra = sorted(set(event) - _KNOWN_FIELDS)
            if extra:
                reasons.append(f"{name} event {index}: unknown key {extra[0]!r}")
                break
    groups = blocks(stream, block)
    in_block = {i for group in groups for i in group}
    if any(_canon(events[i]) != _canon(mixed[i]) for i in range(len(events)) if i not in in_block):
        reasons.append("an event in no block (x_1) changed")
    if any(_slots(a) != _slots(b) for a, b in zip(events, mixed, strict=True)):
        reasons.append("a slot field changed (t, tick, segment and causal_lp stay at the slot)")
    if _canon(stream["sleep"]) != _canon(shuffled["sleep"]):
        reasons.append("the sleep rows differ from the original's")
    if any(stream.get(key) != shuffled.get(key) for key in _BOUND_KEYS):
        reasons.append(
            "revision binding: seed, config, policy or final differ from the recording's"
        )
    declared = shuffled.get("shuffle")
    declares = (
        isinstance(declared, dict)
        and declared.get("seed") == stream.get("seed")
        and declared.get("block") == block
        and isinstance(declared.get("k"), int)
    )
    if not declares:
        reasons.append('the stream does not declare its shuffle {"seed", "block", "k"}')
    for group in groups:
        if _pairs(events, group) != _pairs(mixed, group):
            reasons.append(f"block at event {group[0]}: the multiset of (x, c) pairs changed")
        elif _percepts(events, group) != _percepts(mixed, group):
            reasons.append(f"block at event {group[0]}: the percept fields did not travel together")
        moved = sum(_canon(_percept(events[i])) != _canon(_percept(mixed[i])) for i in group)
        if len(group) >= 2 and 2 * moved < len(group):
            reasons.append(
                f"block at event {group[0]}: {moved} of {len(group)} events moved, under half"
            )
    if md._source_digest(stream) == md._source_digest(shuffled):
        reasons.append("the digest is unchanged: nothing moved")
    if declares:
        try:
            again = shuffle_blocks(stream, declared["k"], block=block)
        except (ValueError, AssertionError) as error:
            reasons.append(f"the transform refuses the original stream: {error}")
        else:
            if _canon(again) != _canon(shuffled):
                reasons.append("not the transform's output for its declared (seed, block, k)")
    return {"ok": not reasons, "reasons": reasons}


def settled_invariance(plain_run, shuffled_run, stream, *, boundary=256, block=SHUFFLE_BLOCK):
    """2.6's Settled invariance over two replay results of one recording, plain and shuffled, as
    {"ok", "per_block", "excluded"}: in every block lying wholly after the adaptation boundary the
    multiset of the runs' per-event phi crc32s is equal while the sequences differ (a block of one event
    cannot differ and is only held to the multiset). Replay delivers every event to the pool once, in
    tick order, so event i (from 0) is the pool's percept number i + 1, and a block lies wholly after
    `boundary` when its first event's delivery count exceeds it; every other block is `excluded`
    (warm-up runs every form as Carried, G1B 2.2). No counted block is not ok: an empty proof proves
    nothing. The red is the same check on a history-dependent phi, whose multisets must differ."""
    count = len(stream["events"])
    for name, run in (("plain", plain_run), ("shuffled", shuffled_run)):
        if len(run.get("phi", ())) != count:
            raise ValueError(
                f"the {name} run has {len(run.get('phi', ()))} phi rows for {count} events"
            )
    per_block, excluded = [], 0
    for group in blocks(stream, block):
        if group[0] + 1 <= boundary:  # delivery count of the block's first event
            excluded += 1
            continue
        before = [plain_run["phi"][i] for i in group]
        after = [shuffled_run["phi"][i] for i in group]
        equal, differs = sorted(before) == sorted(after), before != after
        per_block.append(
            {
                "start": group[0],
                "n": len(group),
                "segment": stream["events"][group[0]]["segment"],
                "multiset_equal": equal,
                "sequence_differs": differs,
                "ok": equal and (differs or len(group) < 2),
            }
        )
    return {
        "ok": bool(per_block) and all(row["ok"] for row in per_block),
        "per_block": per_block,
        "excluded": excluded,
    }


# Sites the duplicate-capability hook flagged for this decision, read; none decides survival of a shuffle:
# distinct-from: faculties/brain_decision.py::_ab_world -- shuffles a per-seed list of learn rates for an A/B world
# distinct-from: faculties/metacognitive_monitor.py::_build_stream -- shuffles a plan of class and noise samples into a stream
# distinct-from: faculties/recursive_tom.py::t082_contrasts -- T082's per-seed conjuncts, a different task's verdict
# distinct-from: instruments/paired_contrast.py::contrast -- this CALLS it (through contrast_row) and adds only 2.3's met rule at the bar and count its caller names
def survives(d_shuf_diffs, *, bar=T_CRIT, n_required=N_LIVE):
    """2.6: D_shuf "survives" the shuffle when contrast(D_shuf) is met at T_CRIT: the 31 per-seed
    differences (each a mean over the K shuffles), through `contrast_row`. The construction controls
    (2.10) take the same decision over 8 synthetic seeds at Part F's bar: `bar` and `n_required`."""
    return bool(contrast_row("D_shuf", d_shuf_diffs, bar=bar, n_required=n_required)["met"])


def incremental(d, d_shuf):
    """2.6's reported "incremental benefit": `contrast_row` over the per-seed d - d_shuf."""
    d, d_shuf = tuple(d), tuple(d_shuf)
    if len(d) != len(d_shuf):
        raise ValueError(f"D has {len(d)} differences and D_shuf {len(d_shuf)}")
    return contrast_row(
        "D - D_shuf (incremental benefit)", [a - b for a, b in zip(d, d_shuf, strict=True)]
    )


# ---- section 2.5: the G-R1 reading ---------------------------------------------------------------


def _read_bool(row, key, who):
    value = row.get(key) if isinstance(row, dict) else None
    if not isinstance(value, bool | np.bool_):
        raise ValueError(
            f"g_r1_reading: {who} must be a contrast_row dict with a bool {key!r}, "
            f"got a {type(row).__name__}"
        )
    return bool(value)


# distinct-from: gates/conclusion_protocol.py -- reads tracked prose for a verdict about experience; this is G1(c)'s 2.5 table, rows over contrast_row dicts and two bools, a pure function with no text
def g_r1_reading(*, blind_harness_ok, primary, stream_history, d_shuf, shuffle_sighted):
    """2.5's table as a pure function, rows read top to bottom, the first that applies deciding:
    {"verdict", "wording", "skew_flagged"}. `blind_harness_ok` and `shuffle_sighted` are bools (2.4's
    (i) and (ii) both held; 2.6's positive control sighted on the surviving host); `primary`,
    `stream_history` and `d_shuf` are `contrast_row` dicts. A row reads only the inputs it needs (an
    input that row never reaches may be None); a needed input that is not a row or a bool raises
    ValueError. A met D_shuf reads RED whether or not the control is sighted, a blind control never
    reads PASS, and a met primary below SKEW_BOUND carries skew_flagged in every row."""
    if not isinstance(blind_harness_ok, bool):
        raise ValueError("g_r1_reading: blind_harness_ok must be a bool")
    primary_met = _read_bool(primary, "met", "primary")
    flagged = _read_bool(primary, "skew_flagged", "primary")

    def reading(verdict, wording=None):
        return {"verdict": verdict, "wording": wording, "skew_flagged": flagged}

    if not blind_harness_ok:  # row 1: 2.4 (i) or (ii) failed
        return reading("REFUSED (blind harness)")
    if not primary_met:  # row 2: the stream-history reading words the null
        met = _read_bool(stream_history, "met", "stream_history")
        return reading("RED", "pool-null" if met else "stream-null")
    if _read_bool(d_shuf, "met", "d_shuf"):  # row 3: the advantage survives the shuffle
        return reading("RED", "survives the shuffle")
    if not isinstance(shuffle_sighted, bool):
        raise ValueError("g_r1_reading: shuffle_sighted must be a bool")
    return reading("PASS" if shuffle_sighted else "NOT READ")  # rows 4 and 5


# ---- section 2.9: the migration gate's reading -----------------------------------------------------
# 2.9 (ii): |mean D_hosted - mean D_port| <= 0.005, on D and (strengthening) on D_task
MIGRATION_TOL = 0.005
assert MIGRATION_TOL == 0.005, "the migration tolerance is the frozen number (2.9)"


# a real, finite number: None, NaN, an infinity, a bool or a string is no measurement and never a pass
def _measured(value):
    is_number = isinstance(value, int | float | np.integer | np.floating)
    return is_number and not isinstance(value, bool | np.bool_) and math.isfinite(float(value))


# Sites the duplicate-capability hook flagged for the two functions below, read; none compares two means:
# distinct-from: gates/eprop_transport_gate.py::resolve_disposition -- reads a finding's declared disposition from a registry
# distinct-from: plasticity/r_stdp.py::Calibration -- a NamedTuple of what a host measured (workers, floors, ns per synapse)
# distinct-from: reasoning/spike_chat_lake.py::_attended_clause -- a chat reply's one-sentence report of the attended content
# distinct-from: faculties/access_consciousness.py::ignited -- whether a content crossed the ignition threshold and held a slot
# distinct-from: faculties/affective_memory.py::store -- encodes one episode with emotion-weighted eviction
# distinct-from: reasoning/stats_bounds.py::within -- a value inside the mean +- k sigma band of its own samples; this is an absolute tolerance between two means
# distinct-from: gates/amdahl_gate.py::within_noise_of_zero -- a timing sham's magnitude-or-band criterion; no two-mean difference
def within(d_host, d_port, tol=MIGRATION_TOL):
    """2.9 (ii)'s conjunct: |d_host - d_port| <= tol, INCLUSIVE at tol. None, NaN, an infinity or a
    non-number on either side is False, never a pass (a gate that cannot read a mean accepts nothing)."""
    if not _measured(tol) or tol < 0:
        raise ValueError(f"within: tol must be a finite non-negative number, got {tol!r}")
    return _measured(d_host) and _measured(d_port) and abs(float(d_host) - float(d_port)) <= tol


def migration_reading(
    parity_ok, d_live_host, d_live_port, d_task_host, d_task_port, r_b3_ok, r_b4_ok
):
    """2.9's disposition as a pure function: "hosted" only when ALL hold -- (i) the parity fixtures
    (`parity_ok`), (ii) BOTH D conjuncts (`within` on the live D means and on the D_task means), and
    (iii) R-b3 and R-b4 on the hosted form -- else "port" (the migration is then an opening). A flag
    counts only as a bool True; a missing or non-bool input, a NaN mean and a False are all "port"."""
    held = all(
        isinstance(flag, bool | np.bool_) and bool(flag) for flag in (parity_ok, r_b3_ok, r_b4_ok)
    )
    both = within(d_live_host, d_live_port) and within(d_task_host, d_task_port)
    return "hosted" if held and both else "port"


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


# ---- the shuffle's hand fixtures and reds ---------------------------------------------------------


# distinct-from: instruments/memory_diagnostic.py::_synthetic_stream -- this CALLS it and adds only the recording's `final` (the current revision), so replay's revision binding holds on the fixture
def _stream_fixture():
    """memory_diagnostic's 48-event hand stream (24 warm-up, 24 unseen-order; the self-check never builds
    a brain) with the current revision as its `final`, so replay's revision binding holds on it."""
    stream = md._synthetic_stream()
    head, dirty = md._revision()
    stream["final"] = {"head": head, "dirty": dirty}
    return stream


def _sourced_fixture():
    """The same stream with every PERCEPT_FIELD an event carries live: event i has the acquisition record
    of frame pointer i in both modalities (consecutive pointers, one contiguous pass), an attended pair,
    a prediction and a fallback flag."""
    stream = _stream_fixture()
    for i, event in enumerate(stream["events"]):
        event["source"] = {
            m: {"modality": m, "pointer": i, "success": True, "index": i % 8, "order": "canonical"}
            for m in ("vision", "audio")
        }
        event["attended"] = {"vision": True, "audio": True}
        event["prediction"] = None if i % 4 == 0 else i % 3
        event["fallback"] = i % 5 == 0
    return stream


def _percept_phi(stream):
    """A Settled-like phi per event: a pure function of the percept (x, c) alone."""
    return [zlib.crc32(_canon([e["x"], e["c"]]).encode()) for e in stream["events"]]


def _history_phi(stream):
    """A Carried-like phi per event: a crc chain over every x delivered so far, so a reorder changes it."""
    phi, chain = [], 0
    for event in stream["events"]:
        chain = zlib.crc32(_canon(event["x"]).encode(), chain)
        phi.append(chain)
    return phi


def _mutant(stream, block, *, fields=PERCEPT_FIELDS, whole_segment=False, identity=False):
    """A red's mutant of `shuffle_blocks` (the same declared shuffle, k 0): `fields` travel instead of
    PERCEPT_FIELDS, or the permutation spans a whole segment, or it is the identity."""
    out = copy.deepcopy(stream)
    out["shuffle"] = {"seed": stream["seed"], "block": block, "k": 0}
    groups = blocks(stream, len(stream["events"]) if whole_segment else block)
    sizes = [len(group) for group in groups]
    perms = (
        [list(range(n)) for n in sizes]
        if identity
        else block_permutations(stream["seed"], 0, sizes)
    )
    for group, perm in zip(groups, perms, strict=True):
        held = [
            {f: stream["events"][i][f] for f in fields if f in stream["events"][i]} for i in group
        ]
        for slot, donor in zip(group, perm, strict=True):
            out["events"][slot].update(copy.deepcopy(held[donor]))
    return out


def _sleep_moved(shuffled):
    out = copy.deepcopy(shuffled)
    out["sleep"][0], out["sleep"][1] = out["sleep"][1], out["sleep"][0]
    return out


def _binding_dropped(shuffled):
    out = copy.deepcopy(shuffled)
    del out["final"]
    return out


# name -> (builds the mutant from the sourced fixture, the reason the check must give for it)
_TRANSFORM_REDS = {
    "cross-block": (
        lambda s: _mutant(s, FIXTURE_BLOCK, whole_segment=True),
        "(x, c) pairs changed",
    ),
    "c-only": (lambda s: _mutant(s, FIXTURE_BLOCK, fields=("c",)), "(x, c) pairs changed"),
    "identity": (lambda s: _mutant(s, FIXTURE_BLOCK, identity=True), "under half"),
    "slot field (tick) moved": (
        lambda s: _mutant(s, FIXTURE_BLOCK, fields=(*PERCEPT_FIELDS, "tick")),
        "a slot field changed",
    ),
    "source stays at the slot": (
        lambda s: _mutant(
            s, FIXTURE_BLOCK, fields=tuple(f for f in PERCEPT_FIELDS if f != "source")
        ),
        "did not travel together",
    ),
    "sleep row moved": (
        lambda s: _sleep_moved(shuffle_blocks(s, 0, block=FIXTURE_BLOCK)),
        "sleep rows",
    ),
    "binding key dropped": (
        lambda s: _binding_dropped(shuffle_blocks(s, 0, block=FIXTURE_BLOCK)),
        "revision binding",
    ),
}


# ---- 2.10's shuffle controls on REAL replays ------------------------------------------------------
# The checks in _check_shuffle on hand phi and hand gains stay as regressions of the arithmetic. These
# run what the campaign runs, the pool and the decoder: a pool made unavailable, or a decoder that
# always predicts 0, fails them.


def _equal_blocks(result):
    return sum(row["multiset_equal"] for row in result["per_block"])


# Sites the duplicate-capability hook flagged for the next two functions, read; none is a pool replay's invariance check:
# distinct-from: instruments/binding_instrument.py::_t080_arm_counts -- counts one arm's scored ticks of a binding recording
# distinct-from: instruments/memory_diagnostic.py::_PoolArm -- the pool arm's bookkeeping; this CALLS it (through replay) and compares two replays of it
# distinct-from: reservoir/recurrent_pool.py::pool_noise_red -- kicks one Carried pool and compares its state chain with an unkicked twin's; no shuffled stream, no Settled form
# distinct-from: brainstem/arm_config.py::ground_free_band -- a joint's range over which the gripper stays off the floor (read only; no brainstem file is touched)
# distinct-from: engine/brain.py::_flush_coverage -- writes the arm's coverage ledger into the lake
def _pool_invariance(stream, shuffled, form, *, block, carrier=None):
    """Settled invariance (2.6) over two REAL pool replays of the fixture, plain and shuffled: the g1
    LSM at TUNED with a window of two blocks, which is also the boundary. `carrier` replays both with
    that carrier skipped on reset (memory_diagnostic._skipped_carrier)."""
    spec = {"candidate": "lsm", "config": TUNED, "form": form, "window": 2 * block}
    with md._skipped_carrier(carrier) if carrier else contextlib.nullcontext():
        plain = md.replay(stream, "pool", pool=spec)
        mixed = md.replay(shuffled, "pool", pool=spec)
    return settled_invariance(plain, mixed, stream, boundary=2 * block, block=block)


def _check_pool_controls(stream, shuffled, block):
    """2.10: on real pool replays Settled keeps every post-boundary block's multiset of phi while the
    sequence moves; Carried must FAIL it (history reaches phi: the red), and so must a Settled whose
    reset skips a carrier (the skip-a-carrier red, each carrier of the pool). Returns the counts."""
    settled = _pool_invariance(stream, shuffled, "settled", block=block)
    assert settled["ok"] and settled["excluded"] == 2 and len(settled["per_block"]) == 4, settled
    assert all(row["sequence_differs"] for row in settled["per_block"]), settled
    carried = _pool_invariance(stream, shuffled, "carried", block=block)
    assert not carried["ok"], f"Carried held Settled invariance: the red did not fire: {carried}"
    skipped = {}
    for carrier in rp.LsmPool.CARRIERS:
        mutant = _pool_invariance(stream, shuffled, "settled", block=block, carrier=carrier)
        assert not mutant["ok"], f"a Settled skipping {carrier!r} held invariance: {mutant}"
        skipped[carrier] = _equal_blocks(mutant)
    return {
        "blocks": len(settled["per_block"]),
        "settled": _equal_blocks(settled),
        "carried": _equal_blocks(carried),
        "skipped": skipped,
    }


def _arm_rows(stream, arm):
    """The feature rows memory_diagnostic's `_feature` gives `arm` (a, e1 or e1-0) on a recording, one
    per event, from the percept and the three before it (zeros before the start), as replay's loop
    supplies them."""
    rows, seen = [], []
    for event in stream["events"]:
        values = {
            "x": event["x"],
            "x_history": [seen[-i] if len(seen) >= i else [0.0] * 64 for i in range(1, 4)],
        }
        rows.append(md._feature(arm, md._View(md.ARMS[arm][1], values)))
        seen.append(event["x"])
    return rows


def _lookahead_rows(stream):
    """RED INSTRUMENT, never an arm and never in a campaign table: row i holds onehot(c_{i+1}), so the
    reference readout, which predicts c_i from row i - 1, is fed the CURRENT category. Its gain over
    arm (a) is order-independent by construction, which is what 2.10's planted gain must be."""
    cats = [event["c"] for event in stream["events"]]
    return [
        md.onehot_32(c) + [0.0] * 32 if c is not None else [0.0] * 64 for c in (*cats[1:], None)
    ]


def _decoder_accuracy(stream, rows):
    """A recording's accuracy through memory_diagnostic's own ReferenceReadout (`_drive_readout`:
    predict c_t from the row before it, then add the pair) at replay's one BLAS thread, scored as the
    campaign's tables score it (`md.score`, `recording_accuracy`)."""
    pairs = list(zip(rows, (event["c"] for event in stream["events"]), strict=True))
    with rp._blas_threads(1):
        guesses = md._drive_readout(md.ReferenceReadout(), pairs)
    return ps.recording_accuracy(md.score(stream, guesses, [False] * len(guesses), floors=False))


# Sites the duplicate-capability hook flagged for the next function, read; none is this gap over feature rows:
# distinct-from: instruments/report_order_probe.py::t081_read -- one recording's per-seed reading of the order probe's fractions
# distinct-from: faculties/agency_monitor.py::_t079_refusals -- names the joints a T079 recording left unscored
# distinct-from: faculties/recursive_tom.py::t082_read -- one recording's per-seed reading of T082's statements
# distinct-from: instruments/pool_selection.py::_contrast -- takes two finished REPLAYS of one synthetic stream and differences their scored accuracy; this runs the reference readout alone over feature rows (a replay cannot carry the lookahead red instrument, and a grower replay costs about a second a stream) and averages the difference over the stream's shuffles, scoring through its recording_accuracy
def _decoder_gaps(recordings, first, second, *, shuffles):
    """Per recording (an rb6 stream of one synthetic seed): accuracy(first rows) - accuracy(second
    rows) through the reference readout, the mean over its `shuffles` within-block shuffles
    (k = 0 .. shuffles - 1); `shuffles` 0 reads the recording unshuffled."""
    gaps = []
    for recording in recordings:
        streams = [shuffle_blocks(recording, k) for k in range(shuffles)] or [recording]
        gaps.append(
            statistics.fmean(
                _decoder_accuracy(s, first(s)) - _decoder_accuracy(s, second(s)) for s in streams
            )
        )
    return gaps


def _check_decoder_decisions(*, history_shuffles=1):
    """2.10's "survives" decision on real decoders, synthetic seeds 6100-6107 at Part F's bar: a gain
    from a decoder fed the CURRENT category (the red instrument above) over arm (a), through the
    reference readout and shuffled with the registered K, survives; R-b6's order-dependent history
    gain (e1) - (e1-0) does not (its unshuffled gain, the sighted half, is met at the same bar and
    reaches RB6_BAR). A decoder that always predicts 0 gives every gain 0 and fails both halves."""
    recordings = [md.rb6_stream(seed) for seed in SHUFFLE_RED_SEEDS]
    arm_a, e1, e1_0 = (functools.partial(_arm_rows, arm=arm) for arm in ("a", "e1", "e1-0"))
    bar, n = pc.T_BAR, len(recordings)
    current = _decoder_gaps(recordings, _lookahead_rows, arm_a, shuffles=SHUFFLE_K)
    current_row = contrast_row("planted current-category gain", current, bar=bar, n_required=n)
    assert survives(current, bar=bar, n_required=n) is True, current_row
    sighted = _decoder_gaps(recordings, e1, e1_0, shuffles=0)
    sighted_row = contrast_row("(e1) - (e1-0), unshuffled", sighted, bar=bar, n_required=n)
    assert sighted_row["met"] and statistics.fmean(sighted) >= md.RB6_BAR, sighted_row
    history = _decoder_gaps(recordings, e1, e1_0, shuffles=history_shuffles)
    history_row = contrast_row("(e1) - (e1-0), shuffled", history, bar=bar, n_required=n)
    assert survives(history, bar=bar, n_required=n) is False, history_row
    assert statistics.fmean(history) < md.RB6_BAR, history_row
    return {
        "current_t": current_row["t"],
        "unshuffled_t": sighted_row["t"],
        "history_t": history_row["t"],
        "history_shuffles": history_shuffles,
    }


def _check_shuffle():
    """The transform, its reds, Settled invariance, "survives" and one real replay, on the fixtures;
    returns the numbers main() prints."""
    block = FIXTURE_BLOCK
    stream = _sourced_fixture()
    shuffled = shuffle_blocks(stream, 0, block=block)
    assert check_shuffle_transform(stream, shuffled, block) == {"ok": True, "reasons": []}
    assert _canon(shuffle_blocks(stream, 0, block=block)) == _canon(shuffled)  # same (seed, k)
    assert md._source_digest(shuffle_blocks(stream, 1, block=block)) != md._source_digest(shuffled)
    for name, (
        build,
        reason,
    ) in _TRANSFORM_REDS.items():  # each red FAILS the check, for its reason
        result = check_shuffle_transform(stream, build(stream), block)
        assert not result["ok"] and any(reason in r for r in result["reasons"]), (name, result)
    identity = check_shuffle_transform(stream, _mutant(stream, block, identity=True), block)
    assert any("digest" in r for r in identity["reasons"]), identity
    unknown = copy.deepcopy(stream)
    unknown["events"][5]["extra"] = 1
    assert "'extra'" in _raises(shuffle_blocks, unknown, 0, block=block)
    jumped = copy.deepcopy(stream)  # a gap in block [8..15]'s vision pointers
    jumped["events"][9]["source"]["vision"]["pointer"] = 100
    assert "contiguous pass" in _raises(shuffle_blocks, jumped, 0, block=block)
    interleaved = copy.deepcopy(
        stream
    )  # a segment that is not one run: block [8..15] not consecutive
    interleaved["events"][10]["segment"] = "unseen-order"
    assert "not consecutive" in _raises(shuffle_blocks, interleaved, 0, block=block)
    refused = []  # the half-moved assertion fires on a drawn identity, and is never re-drawn
    for k in range(40):
        try:
            block_permutations(stream["seed"], k, [2])
        except AssertionError as error:
            assert "under half" in str(error)
            refused.append(k)
    assert refused and len(refused) < 40, refused
    # Settled invariance: delivery counts 1-16 are the window here (events 0-15), 17-48 are after it
    plain_run, shuffled_run = {"phi": _percept_phi(stream)}, {"phi": _percept_phi(shuffled)}
    settled = settled_invariance(plain_run, shuffled_run, stream, boundary=2 * block, block=block)
    assert settled["ok"] and settled["excluded"] == 2
    assert [row["start"] for row in settled["per_block"]] == [16, 24, 32, 40]
    carried = settled_invariance(
        {"phi": _history_phi(stream)},
        {"phi": _history_phi(shuffled)},
        stream,
        boundary=2 * block,
        block=block,
    )
    assert not carried["ok"] and not all(r["multiset_equal"] for r in carried["per_block"]), carried
    pre = {"phi": [*shuffled_run["phi"][:3], 12345, *shuffled_run["phi"][4:]]}  # block [0..7] only
    assert settled_invariance(plain_run, pre, stream, boundary=2 * block, block=block)["ok"]
    late = {"phi": [*shuffled_run["phi"][:27], 12345, *shuffled_run["phi"][28:]]}  # block [24..31]
    assert not settled_invariance(plain_run, late, stream, boundary=2 * block, block=block)["ok"]
    # survives: a planted order-independent gain survives, an order-dependent one the shuffle destroys does not
    gain, destroyed = _jittered(0.05, 0.01, 7), _jittered(-0.002, 0.01, 8)
    assert survives(gain) is True and survives(destroyed) is False
    assert survives([0.03] * 31) is False and incremental(gain, destroyed)["mean"] > 0.0
    assert incremental(gain, gain)["degenerate"] and not incremental(gain, gain)["met"]
    # one real replay: a shuffled stream is an ordinary stream under the revision binding
    plain_replay, shuffled_replay = md.replay(stream, "a"), md.replay(shuffled, "a")
    md._validate_run(shuffled_replay, shuffled, shuffled, "a", "causal")
    assert shuffled_replay["source_digest"] == md._source_digest(shuffled)
    assert shuffled_replay["source_digest"] != plain_replay["source_digest"]
    replayed = settled_invariance(
        plain_replay, shuffled_replay, stream, boundary=2 * block, block=block
    )
    assert replayed["ok"], replayed
    # 2.10's controls on REAL replays: the pool's invariance and its two reds, the decoder's decisions
    pool = _check_pool_controls(stream, shuffled, block)
    decoder = _check_decoder_decisions()
    return {
        "reds": len(_TRANSFORM_REDS),
        "counted": len(settled["per_block"]),
        "excluded": settled["excluded"],
        "carried_equal": sum(r["multiset_equal"] for r in carried["per_block"]),
        "gain": contrast_row("gain", gain)["t"],
        "destroyed": contrast_row("destroyed", destroyed)["t"],
        "refused_k": len(refused),
        "digests": (md._source_digest(stream), md._source_digest(shuffled)),
        "pool": pool,
        "decoder": decoder,
    }


def _check_reading():
    """2.5's table: one fixture per row, every input combination, a mutant that ignores each input."""
    met = contrast_row("met", _jittered(0.05, 0.01, 1))
    unmet = contrast_row("not met", _jittered(-0.005, 0.005, 2))
    flagged = contrast_row("met, left tail", _left_tail())

    def given(ok=True, primary=met, history=unmet, shuf=unmet, sighted=True):
        return {
            "blind_harness_ok": ok,
            "primary": primary,
            "stream_history": history,
            "d_shuf": shuf,
            "shuffle_sighted": sighted,
        }

    def reads(verdict, wording=None, skew=False):
        return {"verdict": verdict, "wording": wording, "skew_flagged": skew}

    rows = {
        "refused": (given(ok=False, shuf=met), reads("REFUSED (blind harness)")),
        "pool-null": (given(primary=unmet, history=met, shuf=met), reads("RED", "pool-null")),
        "stream-null": (given(primary=unmet, shuf=met), reads("RED", "stream-null")),
        "survives, sighted": (given(shuf=met), reads("RED", "survives the shuffle")),
        "survives, blind": (given(shuf=met, sighted=False), reads("RED", "survives the shuffle")),
        "not read": (given(sighted=False), reads("NOT READ")),
        "pass": (given(), reads("PASS")),
        "pass, skew-flagged": (given(primary=flagged), reads("PASS", None, True)),
    }
    for name, (inputs, expected) in rows.items():
        assert g_r1_reading(**inputs) == expected, (name, g_r1_reading(**inputs))
    assert g_r1_reading(**given(ok=False, primary=flagged))["skew_flagged"] is True
    # a reading that ignores one input is caught: that input held at a constant changes a row's output
    for key, constant, row in (
        ("blind_harness_ok", True, "refused"),
        ("primary", met, "stream-null"),
        ("primary", unmet, "pass"),
        ("stream_history", unmet, "pool-null"),
        ("stream_history", met, "stream-null"),
        ("d_shuf", unmet, "survives, blind"),
        ("shuffle_sighted", True, "not read"),
        ("shuffle_sighted", False, "pass"),
    ):
        assert g_r1_reading(**{**rows[row][0], key: constant}) != rows[row][1], (key, row)
    verdicts = Counter()  # every input combination against an independent statement of PASS
    for ok, primary, history, shuf, sighted in itertools.product(
        (True, False), (met, unmet), (met, unmet), (met, unmet), (True, False)
    ):
        verdict = g_r1_reading(**given(ok, primary, history, shuf, sighted))["verdict"]
        verdicts[verdict] += 1
        assert (verdict == "PASS") == (ok and primary is met and shuf is unmet and sighted)
    assert "blind_harness_ok" in _raises(g_r1_reading, **given(ok=1))
    assert "primary" in _raises(g_r1_reading, **given(primary=True))
    assert "d_shuf" in _raises(g_r1_reading, **given(shuf={"t": 3.0}))
    assert "shuffle_sighted" in _raises(g_r1_reading, **given(sighted=None))
    return {"rows": len(rows), "combinations": sum(verdicts.values()), "verdicts": dict(verdicts)}


# distinct-from: instruments/deep_stdp.py::_mechanics_check -- asserts an STDP library's mechanics on an MNIST slice; this holds 2.9's two pure functions and the hosted R-b3 equality (sites read)
# distinct-from: reasoning/spike_chat_lake.py::_wants_repl -- decides between an interactive session and the self-check by stdin; no gate, no tolerance
def _check_migration():
    # 2.9's two pure functions: `within`'s inclusive edge and its 0.0049 / 0.0051 boundaries, NaN / None /
    # infinity / non-number inputs and a refused tolerance; `migration_reading`'s "hosted" on a good row
    # and "port" for every False flag, every non-bool flag, a NaN mean, each D conjunct failing alone
    # (the live one passing while D_task fails is the hosted-Reset machinery red's reading) and a
    # reading that ignores one input; then the hosted R-b3 equality on TUNED, the configuration 2.2 freezes
    assert within(0.005, 0.0) and within(0.0, 0.005) and within(-0.005, 0.0)  # inclusive at tol
    assert within(0.0049, 0.0) and within(0.1049, 0.1) and within(0.0, 0.0049)
    assert not within(0.0051, 0.0) and not within(0.1051, 0.1) and not within(0.0, 0.0051)
    for bad in (math.nan, None, math.inf, -math.inf, "0.0", True, [0.0]):
        assert not within(bad, 0.0) and not within(0.0, bad), bad
    assert not within(math.nan, math.nan) and not within(math.inf, math.inf)
    assert not within(None, None) and not within(np.float64("nan"), np.float64(0.0))
    assert within(np.float64(0.0049), np.float32(0.0)) and not within(np.float64(0.0051), 0.0)
    for tol in (-0.001, math.nan, None, math.inf):
        assert "tol" in _raises(within, 0.0, 0.0, tol), tol
    good = {
        "parity_ok": True,
        "d_live_host": 0.012,
        "d_live_port": 0.010,
        "d_task_host": 0.090,
        "d_task_port": 0.0892,
        "r_b3_ok": True,
        "r_b4_ok": True,
    }
    assert migration_reading(**good) == "hosted"
    edge = dict(good, d_live_host=0.0049, d_live_port=0.0, d_task_host=0.0049, d_task_port=0.0)
    assert migration_reading(**edge) == "hosted"
    over = {"d_live_host": 0.0051, "d_live_port": 0.0, "d_task_host": 0.0051, "d_task_port": 0.0}
    assert migration_reading(**dict(good, **over)) == "port"
    task_fails = dict(good, d_task_host=0.0051, d_task_port=0.0)
    live_fails = dict(good, d_live_host=0.0051, d_live_port=0.0)
    rows = {
        "the live conjunct alone passes while D_task fails": task_fails,
        "D_task passes while the live conjunct fails": live_fails,
        "a NaN live mean": dict(good, d_live_host=math.nan),
        "a missing task mean": dict(good, d_task_port=None),
        "parity False": dict(good, parity_ok=False),
        "R-b3 False": dict(good, r_b3_ok=False),
        "R-b4 False": dict(good, r_b4_ok=False),
        "parity missing": dict(good, parity_ok=None),
        "R-b3 a truthy non-bool": dict(good, r_b3_ok=1),
        "R-b4 a string": dict(good, r_b4_ok="True"),
    }
    for name, row in rows.items():
        assert migration_reading(**row) == "port", name
    assert migration_reading(**dict(good, parity_ok=np.True_)) == "hosted"
    # a reading that ignores one input is caught: that input held at a constant flips a row's verdict
    for key, constant, row in (
        ("parity_ok", True, "parity False"),
        ("r_b3_ok", True, "R-b3 False"),
        ("r_b4_ok", True, "R-b4 False"),
        ("d_live_host", 0.0, "D_task passes while the live conjunct fails"),
        ("d_live_port", 0.0051, "D_task passes while the live conjunct fails"),
        ("d_task_host", 0.0, "the live conjunct alone passes while D_task fails"),
        ("d_task_port", 0.0051, "the live conjunct alone passes while D_task fails"),
    ):
        assert migration_reading(**{**rows[row], key: constant}) == "hosted", (key, row)
    equal = rp.check_r_b3("lsm", TUNED, host="kernel", percepts=64, window=32)
    assert equal["verdict"], equal["numbers"]["equal"]
    return {"rows": len(rows), "equal": equal["numbers"]["equal"]}


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
    shuffle, reading, migration = _check_shuffle(), _check_reading(), _check_migration()
    print(
        "self-check OK: constants tied at import (S = 31, live/reserve/exposed disjoint, TUNED in the "
        f"grid); decoder 0.05 +- 0.01 MET at t {decoder['t']:.4f} > {T_CRIT} (exact FPR "
        f"{decoder['fpr']:.6f}, band {decoder['band_population']} flips); decoder-not-grower red: "
        f"decoder NOT MET at t {reds['decoder']['t']:.4f} while the grower is MET at t "
        f"{reds['grower']['t']:.4f}; symmetric zero-mean NOT MET (t {symmetric['t']:.4f}, FPR "
        f"{symmetric['fpr']:.6f}); all-zero degenerate NOT MET; constant +0.03 degenerate NOT MET "
        f"where contrast() says met (FPR {flat['fpr']:.6f}, band {flat['band_population']}); "
        f"left-tailed met vector skew-flagged (g1 {tail['skewness']:.4f}, t {tail['t']:.4f}); "
        "30 values, exposed, reserve and duplicate seeds, a missing, None or NaN oracle cell all refused; "
        f"shuffle (blocks of {FIXTURE_BLOCK}): {shuffle['reds']} transform reds each fail the check for "
        "their own reason, an unknown key and a non-contiguous block are refused, the half-moved "
        f"assertion fires on {shuffle['refused_k']} of 40 drawn 2-blocks; Settled invariance holds on "
        f"{shuffle['counted']} post-boundary blocks ({shuffle['excluded']} pre-boundary excluded) and "
        f"fails on a Carried-like phi ({shuffle['carried_equal']} blocks equal); survives True at t "
        f"{shuffle['gain']:.4f}, False at t {shuffle['destroyed']:.4f}; a shuffled stream replays under "
        f"the revision binding (digest {shuffle['digests'][0]} -> {shuffle['digests'][1]}); on REAL "
        f"replays of the g1 pool Settled holds {shuffle['pool']['settled']} of "
        f"{shuffle['pool']['blocks']} post-boundary blocks while Carried holds "
        f"{shuffle['pool']['carried']} and a Settled skipping a carrier holds "
        f"{shuffle['pool']['skipped']}; through the reference readout on seeds 6100-6107 at "
        f"{pc.T_BAR} the planted current-category gain survives (t {shuffle['decoder']['current_t']:.4f}) "
        f"and R-b6's history gain (e1) - (e1-0), met unshuffled (t "
        f"{shuffle['decoder']['unshuffled_t']:.4f}), does not when shuffled "
        f"({shuffle['decoder']['history_shuffles']} shuffle, t {shuffle['decoder']['history_t']:.4f}); G-R1: "
        f"{reading['rows']} table fixtures exact, {reading['combinations']} input combinations "
        f"{reading['verdicts']}, an ignore-one-input mutant caught for every input; migration gate: "
        f"within() inclusive at {MIGRATION_TOL} (0.0049 in, 0.0051 out, NaN and None never a pass), "
        f"migration_reading hosted on the good row and edge, port on {migration['rows']} reds (each "
        f"False flag, each D conjunct failing alone, a NaN, a missing input), hosted R-b3 equal on "
        f"TUNED {migration['equal']}"
    )


if __name__ == "__main__":
    main()
