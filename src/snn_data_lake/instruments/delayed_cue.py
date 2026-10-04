r"""G2 / T110's problem generators in the learner's own code, and their construction checks (P1).

Pre-registration: specs/022-unified-brain-arc/evidence/g2-taught-problems.md (frozen at 5e3c2f6e; 2.2, 2.3,
2.8, 2.15). This module GENERATES problems and CHECKS their construction; it teaches nothing. The Problem
contract, splits-as-episodes, the solved predicate and the budget fields are `taught_problems` (P2); the
taught campaign, the presized expansion and the arms are `taught_campaign` (P3).

Families:
  (i)   the delayed-cue 2x2: three-turn episodes CUE[c] -> FILLER -> "P[p] Q[q]", four labellings of the
        same 8 episodes: S0C0 = p, S1C0 = c (needs state), S0C1 = p XOR q (needs capacity), S1C1 = c XOR p
        (both). Labels exist only on the probe.
  C1-d  the held-out capacity cell (2.15.1): one-turn "P Q R1 R2", label p XOR q, 4 x 16 distractor combos;
        splits are disjoint distractor-combo sets, so held-out means novel COMBINATIONS of taught words.
  A_c1d task A for B = C1-d (orchestrator ruling 15): C1-d's shape on its own words, label p (separable).
  A_ii  task A for B = (ii) (2.2): (ii)'s word-order construction on ram/cup/emu/cod, labelled u (separable).
  U     the deterministic contradiction fixture for STATELESS readers (2.15.6): item i at pass k is labelled
        h(seed, i) XOR (k mod 2), so every item carries both labels.
  U_S   the stateful-branch fixture (2.15.6): each read's labels are drawn fresh from its own stream index.

Why the S1 and C1 cells are out of the incumbent's reach (2.3): the plain probe text does not depend on the cue
and the read is a pure function of (weights, text), so a state cell reads at most 0.5 EXACTLY; each relevant bit
sets its own word's (synapse, rank), so phi00 + phi11 = phi01 + phi10 and no linear read answers all four XOR
combinations -- exact on integer ranks only, a MEASUREMENT in the learner's float drive (B4, as A3 is for (ii)).

Checks here, each with a constructed red in `construction_reds`:
  B1   vocabulary and code (buckets distinct, >= 8, disjoint across families; ranks strictly by position; the
       exact empty-bucket filler form; <= T_POST words; firewall kind `reject`)
  B2   (a) the plain probe code does not depend on the cue; (b) additivity per value of every other bit;
       (c) witness weights read the separable cells 8 of 8 through the real `route_text`, XOR at most 6 of 8
  B3   the state-identity reader (a pure check over a reader's answers; its stateful red is P3's)
  B10  U carries both labels on every judged item; B10-S each U_S read draws its own stream index
  B13  C1-d / A_c1d splits: four-way disjoint, sizes >= 8 / 2 / 3 / 3, every distractor word in teach
  B14  the whole-code lookup memorizer solves teach and NOT future / future_d; the hand witness solves every split
  B2-A the A_c1d linear witness reads all 64 items through the real `route_text`

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.delayed_cue

Deterministic; ASCII-only prints; builds no Brain, touches no lake.
"""

import functools
import random
import statistics
import zlib

from snn_data_lake.instruments.inseparable_mapping import (
    BUCKETS_52 as BUCKETS_II,
)
from snn_data_lake.instruments.inseparable_mapping import (
    GOAL,
    GREET,
    N_IN,
    T_POST,
    WITNESS_THRESHOLD,
    code,
    ranks,
)
from snn_data_lake.instruments.inseparable_mapping import TEXTS as TEXTS_II
from snn_data_lake.instruments.inseparable_mapping import _bucket as bucket
from snn_data_lake.instruments.inseparable_mapping import _incumbent as incumbent
from snn_data_lake.instruments.inseparable_mapping import label_xor as label_xor_ii

# distinct-from: growth/autapse_neuron.py::_delayed_match_stream -- one autaptic LIF on a spike stream, last-bit
#   recall; instruments/episode_driver.py -- KernelTick sample/delay/probe trials with a same/different oracle;
#   reasoning/developmental_loop_wm.py -- explicit working memory in the developmental loop. None is a
#   text-channel problem for the taught learner (evidence 1.4). Sites read 2026-10-03.
# distinct-from: faculties/self_model.py -- a ring of the brain's own signals; infra/self_mod_harness.py -- runs
#   run_selfchecks in a sandbox; substrate/chat_memory.py and engine/learn_daemon.py -- ingest (role, text)
#   turns from transcripts; engine/autonomous_loop.py -- the multimodal percept vector. None generates a
#   labelled problem or checks one's construction. Sites read 2026-10-04.

LABELS = (GREET, GOAL)  # bit 0 -> GREET, bit 1 -> GOAL (2.2; the pair (ii) uses)
ADDITIVITY_TOL = 1e-12
MIN_BUCKET = 8  # every word's bucket >= 8, so fillers are the lowest-index empty buckets in every text

# ---- (i) the delayed-cue 2x2 (2.2) -------------------------------------------------------------------
CUE = ("jar", "pen")
FILLER = "box"
P_WORDS = ("oak", "elm")
Q_WORDS = ("fox", "owl")
BUCKETS_I = {"jar": 14, "pen": 19, "box": 38, "oak": 21, "elm": 23, "fox": 50, "owl": 49}
ITEMS_I = tuple((c, p, q) for c in (0, 1) for p in (0, 1) for q in (0, 1))
CELLS = {
    "S0C0": lambda c, p, q: p,
    "S1C0": lambda c, p, q: c,
    "S0C1": lambda c, p, q: p ^ q,
    "S1C1": lambda c, p, q: c ^ p,
}
SEPARABLE_CELLS = ("S0C0", "S1C0")
XOR_CELLS = ("S0C1", "S1C1")
HISTORY_WINDOW = 3


def probe_text(item):
    _, p, q = item
    return f"{P_WORDS[p]} {Q_WORDS[q]}"


def episode_i(item):
    """The three turns of item (c, p, q): (text, role); the label rides on the probe only."""
    c, _, _ = item
    return ((CUE[c], "context"), (FILLER, "context"), (probe_text(item), "probe"))


def history_text(item, window=HISTORY_WINDOW):
    """The explicit-history control's probe input: the episode's last `window` turns concatenated."""
    return " ".join(text for text, _ in episode_i(item)[-window:])


def label_i(cell, item):
    return LABELS[CELLS[cell](*item)]


class ExplicitHistory:
    """The plan's explicit-history control (2.4): the same turn stream, but the reader's probe input is the
    episode's last `window` turns concatenated. It adds exactly the cue (window 3) and no capacity. Teaching
    goes through the wrapped learner's own `learn_turn(text, None, feedback=label)`."""

    def __init__(self, unit, window=HISTORY_WINDOW):
        self.unit, self.window = unit, int(window)

    def text(self, turns):
        return " ".join(text for text, _ in tuple(turns)[-self.window :])

    def route(self, turns):
        return self.unit.route_text(self.text(turns))

    def teach(self, turns, label):
        return self.unit.learn_turn(self.text(turns), None, feedback=label)


# ---- C1-d and A_c1d (2.15.1; rulings 12, 14, 15) ---------------------------------------------------------
# R0 2026-10-04 (dev, no learner): first fit over a fixed sorted list of three-letter words; every bucket
# distinct, >= 8, outside (i), (ii) and A_ii; firewall kind `reject` ("go home" reads `arm`).
COMBO_FAMILIES = {
    "c1d": {
        "P": ("awl", "bat"),
        "Q": ("bee", "bun"),
        "R1": ("cab", "car", "cog", "cud"),
        "R2": ("dip", "dot", "dub", "fir"),
        "label": lambda p, q: p ^ q,
    },
    "ac1d": {
        "P": ("fog", "gel"),
        "Q": ("gin", "gum"),
        "R1": ("hip", "ink", "jaw", "keg"),
        "R2": ("lop", "mob", "mod", "orb"),
        "label": lambda p, q: p,
    },
}
# the R0 buckets: ASSERTED by B1 against the learner's own featurizer, never used to build anything
BUCKETS_C1D = {"awl": 27, "bat": 11, "bee": 29, "bun": 48, "cab": 45, "car": 33, "cog": 8, "cud": 37,
               "dip": 24, "dot": 35, "dub": 25, "fir": 42}
BUCKETS_AC1D = {"fog": 43, "gel": 30, "gin": 18, "gum": 9, "hip": 32, "ink": 44, "jaw": 13, "keg": 22,
                "lop": 34, "mob": 17, "mod": 40, "orb": 47}
COMBOS = tuple((a, b) for a in range(4) for b in range(4))
PQ = ((0, 0), (0, 1), (1, 0), (1, 1))
SPLIT_SIZES = {"teach": 8, "validation": 2, "future": 3, "future_d": 3}  # ruling 12: counts only go up
SPLITS = tuple(SPLIT_SIZES)


def combo_words(family):
    f = COMBO_FAMILIES[family]
    return tuple(f["P"] + f["Q"] + f["R1"] + f["R2"])


def combo_text(family, item):
    p, q, a, b = item
    f = COMBO_FAMILIES[family]
    return f"{f['P'][p]} {f['Q'][q]} {f['R1'][a]} {f['R2'][b]}"


def combo_label(family, item):
    p, q, _, _ = item
    return LABELS[COMBO_FAMILIES[family]["label"](p, q)]


def combo_items(combos):
    return tuple((p, q, a, b) for (a, b) in combos for (p, q) in PQ)


def _covers_teach(teach):
    return {a for a, _ in teach} == set(range(4)) and {b for _, b in teach} == set(range(4))


def combo_partition(family, seed, sizes=SPLIT_SIZES):
    """The per-seed partition of the 16 distractor combos (2.15.1): `random.Random(f"g2.{family}-split:{seed}")`
    shuffles them; a draw whose teach split misses a distractor word is REJECTED and redrawn from the same
    stream, so no read ever drives an untaught synapse. Returns ({split: combos}, draws used)."""
    if sum(sizes.values()) > len(COMBOS):
        raise ValueError(f"split sizes {sizes} exceed {len(COMBOS)} combos")
    rng = random.Random(f"g2.{family}-split:{seed}")
    for draws in range(1, 10_001):
        order = rng.sample(COMBOS, len(COMBOS))
        parts, start = {}, 0
        for name, size in sizes.items():
            parts[name] = tuple(order[start : start + size])
            start += size
        if _covers_teach(parts["teach"]):
            return parts, draws
    raise RuntimeError("no admissible partition in 10,000 draws")


# ---- A_ii (2.2): task A for B = (ii) -------------------------------------------------------------------
WORDS_AII = ("ram", "cup", "emu", "cod")
BUCKETS_AII = {"ram": 15, "cup": 16, "emu": 20, "cod": 26}
TEXTS_AII = {  # u = "cup precedes ram", v = "cod precedes emu" -- (ii)'s construction on disjoint words
    (0, 0): "ram cup emu cod",
    (0, 1): "ram cup cod emu",
    (1, 0): "cup ram emu cod",
    (1, 1): "cup ram cod emu",
}


def label_aii(u, v):
    """A_ii is labelled separably: GREET iff "ram" precedes "cup" (u == 0)."""
    return LABELS[u]


# ---- U and U_S (2.15.6) ---------------------------------------------------------------------------------
def u_label(seed, index, pass_k):
    """U: item `index` at pass `pass_k` is labelled h(seed, index) XOR (pass_k mod 2) -- opposite labels on
    consecutive passes, so a stateless reader is wrong on exactly half of every item's passes."""
    h = zlib.crc32(f"{seed}:{index}".encode("ascii")) & 1
    return LABELS[h ^ (pass_k % 2)]


def us_labels(seed, split, read_index, n):
    """U_S: one fair bit per presentation from the stream of this (seed, split, read_index) alone."""
    rng = random.Random(f"g2.US:{seed}:{split}:{read_index}")
    return [LABELS[rng.getrandbits(1)] for _ in range(n)]


# ---- the witness and decoy units (2.13 P1; G4's T2) -------------------------------------------------------
def dot(w, phi):
    return sum(wi * pi for wi, pi in zip(w, phi, strict=True))


class ThresholdUnit:
    """A hand threshold unit over the frozen 52-dim code: fires iff w . phi > theta."""

    def __init__(self, w, theta):
        self.w, self.theta = tuple(float(x) for x in w), float(theta)

    def fires(self, phi):
        return dot(self.w, phi) > self.theta

    def margin(self, phi):
        return abs(dot(self.w, phi) - self.theta)


def xor_bit(units, phi):
    """The two-unit XOR read: bit 1 iff the first unit fires and the second does not."""
    first, second = units
    return int(first.fires(phi) and not second.fires(phi))


def _indicator(words_and_texts):
    """Weights that read each word's presence as 1.0: 1 / (the word's code value in a text containing it)."""
    w = [0.0] * N_IN
    for word, text in words_and_texts:
        value = code(text)[bucket(word)]
        if value <= 0.0:
            raise ValueError(f"{word!r} carries no PSP in {text!r}")
        w[bucket(word)] = 1.0 / value
    return w


def witness_units(problem):
    """The problem's two hand threshold units over the frozen code (OR, AND; label = OR minus AND), for the
    capacity problems: 'S0C1' (probe code), 'S1C1' (history code), 'c1d', and 'ii'. For (ii) the units are
    `inseparable_mapping.check_a5`'s rule (w = +1 at dog and sky, -1 at cat and sun; thresholds -/+
    WITNESS_THRESHOLD), re-expressed as units and pinned against check_a5's answers by the tests."""
    if problem == "ii":
        # cp1 #21 REFUTED, with this reason: "delegate, never a copy" would need a shared helper INSIDE
        # inseparable_mapping, which is G1(c)'s campaign-recorded instrument (results record 2d714954) -- editing it
        # changes a recorded instrument outside this row. The re-expression is instead PINNED to check_a5 by
        # tests/test_delayed_cue.py (same four answers, margin equal to 1e-12), so any A5 change reds here. The
        # shared helper is the integrator's / G1(c)'s to add when that file next opens.
        w = [0.0] * N_IN
        w[bucket("dog")], w[bucket("sky")] = 1.0, 1.0
        w[bucket("cat")], w[bucket("sun")] = -1.0, -1.0
        return ThresholdUnit(w, -WITNESS_THRESHOLD), ThresholdUnit(w, WITNESS_THRESHOLD)
    if problem == "S0C1":
        w = _indicator([(P_WORDS[1], probe_text((0, 1, 1))), (Q_WORDS[1], probe_text((0, 1, 1)))])
    elif problem == "S1C1":
        w = _indicator([(CUE[1], history_text((1, 1, 0))), (P_WORDS[1], history_text((1, 1, 0)))])
    elif problem == "c1d":
        f = COMBO_FAMILIES["c1d"]
        text = combo_text("c1d", (1, 1, 0, 0))
        w = _indicator([(f["P"][1], text), (f["Q"][1], text)])
    else:
        raise ValueError(f"no witness for {problem!r}")
    return ThresholdUnit(w, 0.5), ThresholdUnit(w, 1.5)


def decoy_units(texts, n, seed):
    """n random threshold units over the same code (G4's T2 dilution control): w ~ N(0, 1) per synapse from
    `random.Random(seed)` (the CALLER's seed), theta the median drive over `texts` (label-free)."""
    rng = random.Random(seed)
    phis = [code(t) for t in texts]
    units = []
    for _ in range(int(n)):
        w = [rng.gauss(0.0, 1.0) for _ in range(N_IN)]
        units.append(ThresholdUnit(w, statistics.median(dot(w, phi) for phi in phis)))
    return units


# ---- B1: vocabulary and code -----------------------------------------------------------------------------
@functools.cache
def _firewall():
    from snn_data_lake.sensorimotor.nl_interface import NLInterface

    return NLInterface()


def firewall_kind(text):
    return _firewall().interpret(text).get("kind")


def check_b1(words, texts, forbidden=(), expected=None):
    """B1 over one family: `words` occupy distinct buckets >= MIN_BUCKET, none in `forbidden`, equal to
    `expected` when given; every text <= T_POST words, made of `words`, each word at rank n - 1 - position
    (ranks strictly by position, read off the learner's own encode), the empty-bucket fillers exactly buckets
    0..(5 - n) at ranks n..5 (D5), and every text's firewall kind `reject` (never `arm` or `lake`)."""
    derived = {word: bucket(word) for word in words}
    by_position, fillers = True, True
    for text in texts:
        split, r = text.split(), ranks(text)
        n = len(split)
        if n > T_POST:
            by_position = fillers = False
            continue
        by_position &= all(r[bucket(word)] == n - 1 - j for j, word in enumerate(split))
        fillers &= all(r[b] == n + b for b in range(T_POST - n))
    conditions = {
        "distinct_buckets": len(set(derived.values())) == len(words),
        "buckets_at_least_8": all(b >= MIN_BUCKET for b in derived.values()),
        "disjoint_from_other_families": not set(derived.values()) & set(forbidden),
        "expected_buckets": expected is None or derived == dict(expected),
        "texts_within_horizon": all(len(t.split()) <= T_POST for t in texts),
        "texts_use_the_vocabulary": all(set(t.split()) <= set(words) for t in texts),
        "ranks_by_position": by_position,
        "fillers_exact": fillers,
        "firewall_reject": all(firewall_kind(t) == "reject" for t in texts),
    }
    return {"ok": all(conditions.values()), "conditions": conditions, "buckets": derived}


def texts_i():
    texts = {text for item in ITEMS_I for text, _ in episode_i(item)}
    return sorted(texts | {history_text(item) for item in ITEMS_I})


def texts_combo(family):
    return [combo_text(family, (p, q, a, b)) for (a, b) in COMBOS for (p, q) in PQ]


def _family_buckets():
    return {
        "i": set(BUCKETS_I.values()),
        "ii": set(BUCKETS_II.values()),
        "aii": set(BUCKETS_AII.values()),
        "c1d": set(BUCKETS_C1D.values()),
        "ac1d": set(BUCKETS_AC1D.values()),
    }


def _others(family):
    return set().union(*(b for name, b in _family_buckets().items() if name != family))


def check_b1_all():
    """B1 on every G2 family. (ii)'s own A1 is G1(c)'s; its buckets enter here only as `forbidden`."""
    words_i = (*CUE, FILLER, *P_WORDS, *Q_WORDS)
    results = {
        "i": check_b1(words_i, texts_i(), _others("i"), BUCKETS_I),
        "c1d": check_b1(combo_words("c1d"), texts_combo("c1d"), _others("c1d"), BUCKETS_C1D),
        "ac1d": check_b1(combo_words("ac1d"), texts_combo("ac1d"), _others("ac1d"), BUCKETS_AC1D),
        "aii": check_b1(WORDS_AII, list(TEXTS_AII.values()), _others("aii"), BUCKETS_AII),
    }
    return {"ok": all(r["ok"] for r in results.values()), "families": results}


# ---- B2: structure on the learner's code -----------------------------------------------------------------
def additivity(text_of):
    """phi00 + phi11 = phi01 + phi10 for the two-bit text function `text_of(x, y)`: integer-rank deviation,
    float-code deviation, and whether the four codes differ (an additivity of identical codes is vacuous)."""
    r = {xy: ranks(text_of(*xy)) for xy in PQ}
    p = {xy: code(text_of(*xy)) for xy in PQ}
    rank_dev = max(abs(r[0, 0][i] + r[1, 1][i] - r[0, 1][i] - r[1, 0][i]) for i in range(N_IN))
    code_dev = max(abs(p[0, 0][i] + p[1, 1][i] - p[0, 1][i] - p[1, 0][i]) for i in range(N_IN))
    distinct = all(p[a] != p[b] for k, a in enumerate(PQ) for b in PQ[k + 1 :])
    return rank_dev, code_dev, distinct


def _separable_witness(greet_bucket, goal_bucket):
    """An incumbent whose weights read one bit: GREET +1 at `greet_bucket` and -1 at `goal_bucket`, GOAL the
    negative, every other neuron 0 (inseparable_mapping's _witness_unit form)."""
    unit = incumbent(0)
    for index, name in enumerate(unit.responses):
        w = [0.0] * N_IN
        if name == GREET:
            w[greet_bucket], w[goal_bucket] = 1.0, -1.0
        elif name == GOAL:
            w[greet_bucket], w[goal_bucket] = -1.0, 1.0
        unit.neurons[index].w = w
    return unit


def _read_count(unit, pairs):
    return sum(unit.route_text(text) == label for text, label in pairs)


def check_b2(probe=probe_text, history=history_text, slot_cell=None):
    """B2 on (i). (a) the plain probe code is identical across cue values; (b) additivity for S0C1 on the probe
    code (each value of c) and S1C1 on the history code (each value of q), exact on ranks, within 1e-12 in
    float; (c) witness weights read S0C0 (probe) and S1C0 (history) 8 of 8 through the real route_text, and
    the XOR cells at most 6 of 8 under the same weights. `probe`, `history` and `slot_cell` (a labelling put
    in a separable slot) build the reds."""
    cue_free = all(code(probe((0, p, q))) == code(probe((1, p, q))) for p in (0, 1) for q in (0, 1))
    devs = []
    for c in (0, 1):
        devs.append(additivity(lambda p, q, c=c: probe((c, p, q))))
    for q in (0, 1):
        devs.append(additivity(lambda c, p, q=q: history((c, p, q))))
    s0 = _separable_witness(bucket(P_WORDS[0]), bucket(P_WORDS[1]))
    s1 = _separable_witness(bucket(CUE[0]), bucket(CUE[1]))
    s0_cell, s1_cell = slot_cell or "S0C0", slot_cell or "S1C0"
    read_s0 = _read_count(s0, [(probe(it), label_i(s0_cell, it)) for it in ITEMS_I])
    read_s1 = _read_count(s1, [(history(it), label_i(s1_cell, it)) for it in ITEMS_I])
    xor_s0 = _read_count(s0, [(probe(it), label_i("S0C1", it)) for it in ITEMS_I])
    xor_s1 = _read_count(s1, [(history(it), label_i("S1C1", it)) for it in ITEMS_I])
    conditions = {
        "probe_code_cue_free": cue_free,
        "ranks_additive": all(d[0] == 0 for d in devs),
        "code_additive": all(d[1] <= ADDITIVITY_TOL for d in devs),
        "codes_distinct": all(d[2] for d in devs),
        "witness_reads_separable_8_of_8": read_s0 == 8 and read_s1 == 8,
        "witness_reads_xor_at_most_6": xor_s0 <= 6 and xor_s1 <= 6,
    }
    return {
        "ok": all(conditions.values()),
        "conditions": conditions,
        "max_code_deviation": max(d[1] for d in devs),
        "witness_reads": {"S0C0": read_s0, "S1C0_history": read_s1, "S0C1": xor_s0, "S1C1": xor_s1},
    }


def check_b2_combo(family="c1d", text_of=None):
    """B2(b)-C1d: for EVERY distractor combo the four core codes are additive (exact on ranks, 1e-12 float),
    so the union of all splits is inseparable (one combo's Farkas witness, 0 elsewhere). `text_of` builds
    the reds."""
    text_of = text_of or (lambda item: combo_text(family, item))
    devs = [additivity(lambda p, q, a=a, b=b: text_of((p, q, a, b))) for (a, b) in COMBOS]
    conditions = {
        "ranks_additive_every_combo": all(d[0] == 0 for d in devs),
        "code_additive_every_combo": all(d[1] <= ADDITIVITY_TOL for d in devs),
        "codes_distinct_every_combo": all(d[2] for d in devs),
    }
    return {"ok": all(conditions.values()), "conditions": conditions,
            "max_code_deviation": max(d[1] for d in devs)}


def float_separability_lp(texts_labels):
    """D2's float measurement: is there (w, b) with y (w . phi + b) >= 1 on every item? Returns 'infeasible'
    (no linear read separates the set in float), 'feasible', or 'not run' (scipy absent)."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return "not run"
    rows = []
    for text, label in texts_labels:
        y = 1.0 if label == GOAL else -1.0
        rows.append([-y * v for v in code(text)] + [-y])
    result = linprog(
        c=[0.0] * (N_IN + 1), A_ub=rows, b_ub=[-1.0] * len(rows),
        bounds=[(None, None)] * (N_IN + 1), method="highs",
    )
    # cp1 #20: HiGHS status 2 is PROVEN infeasibility; any other non-zero status (iteration limit, numerical
    # trouble) is unresolved and reported as such, never read as infeasible
    if result.status == 0:
        return "feasible"
    return "infeasible" if result.status == 2 else f"unresolved:{result.status}"


# D2's float measurement for the record, beside the witness margins (cp1 #20)
LP_CASES = {
    "S0C0": ("i", "S0C0", "probe"), "S1C0_history": ("i", "S1C0", "history"),
    "S0C1": ("i", "S0C1", "probe"), "S1C1_history": ("i", "S1C1", "history"),
    "c1d_union": ("combo", "c1d", None), "ac1d_union": ("combo", "ac1d", None),
}


def lp_measurements():
    out = {}
    for name, (kind, which, text) in LP_CASES.items():
        if kind == "i":
            make = probe_text if text == "probe" else history_text
            pairs = [(make(it), label_i(which, it)) for it in ITEMS_I]
        else:
            pairs = [(combo_text(which, it), combo_label(which, it)) for it in combo_items(COMBOS)]
        out[name] = float_separability_lp(pairs)
    return out


# ---- B3: state identity (pure; the stateful red is the campaign's) -----------------------------------------
def check_b3(answers):
    """B3: `answers` maps each (c, p, q) to a reader's answer on the plain probe; every probe text must get the
    same answer under both cue values (a stateless reader cannot do otherwise)."""
    same = all(answers[(0, p, q)] == answers[(1, p, q)] for p in (0, 1) for q in (0, 1))
    return {"ok": same, "conditions": {"same_answer_both_cues": same}}


# ---- B10, B10-S: the U fixtures -----------------------------------------------------------------------------
PASSES = 4


def check_b10(seed, n_items=len(ITEMS_I), labeller=None):
    """B10: under U's labels every item carries both labels across the PASSES passes of a read (so a stateless
    reader is never solved, exactly), and the labels are balanced. `labeller(seed, i, k)` builds the red."""
    labeller = labeller or u_label
    table = {i: [labeller(seed, i, k) for k in range(PASSES)] for i in range(n_items)}
    both = all(set(v) == set(LABELS) for v in table.values())
    greet = sum(v.count(GREET) for v in table.values())
    conditions = {"every_item_both_labels": both, "balanced": greet * 2 == n_items * PASSES}
    return {"ok": all(conditions.values()), "conditions": conditions}


def check_b10s(seed, read_indices=(0, 1, 2), n=32, split="validation"):
    """B10-S: each U_S read draws its own stream index, so no two reads of a run share a label sequence."""
    reads = [tuple(us_labels(seed, split, r, n)) for r in read_indices]
    distinct = len(set(read_indices)) == len(read_indices) and len(set(reads)) == len(reads)
    return {"ok": distinct, "conditions": {"reads_draw_distinct_streams": distinct}}


# ---- B13, B14: C1-d / A_c1d splits ----------------------------------------------------------------------------
def check_b13(parts, sizes=SPLIT_SIZES):
    seen = [c for name in SPLITS for c in parts.get(name, ())]
    conditions = {
        "four_way_disjoint": len(seen) == len(set(seen)),
        "sizes_at_least_registered": all(len(parts.get(n, ())) >= sizes[n] for n in SPLITS),
        "teach_covers_every_distractor_word": _covers_teach(parts.get("teach", ())),
    }
    return {"ok": all(conditions.values()), "conditions": conditions}


def combo_witness_answer(family):
    """B14's hand positive: c1d through the OR-minus-AND units; ac1d through its linear witness (B2-A)."""
    if family == "c1d":
        units = witness_units("c1d")
        return lambda text: LABELS[xor_bit(units, code(text))]
    p_words = COMBO_FAMILIES["ac1d"]["P"]
    return _separable_witness(bucket(p_words[0]), bucket(p_words[1])).route_text


def check_b14(family, parts, labelled=None):
    """B14: the whole-code lookup MEMORIZER (each teach item's code -> its label; GREET on an unseen code)
    solves teach and NOT future / future_d; the hand witness solves every split. `labelled(item)` replaces the
    generator's labels (the red)."""
    label = labelled or (lambda it: combo_label(family, it))
    items = {name: combo_items(parts[name]) for name in SPLITS}
    memo = {tuple(code(combo_text(family, it))): label(it) for it in items["teach"]}

    def memo_answer(text):
        return memo.get(tuple(code(text)), GREET)

    witness = combo_witness_answer(family)

    def solves(answer, name):
        return all(answer(combo_text(family, it)) == label(it) for it in items[name])

    conditions = {
        "memorizer_solves_teach": solves(memo_answer, "teach"),
        "memorizer_fails_future": not solves(memo_answer, "future"),
        "memorizer_fails_future_d": not solves(memo_answer, "future_d"),
        "witness_solves_every_split": all(solves(witness, n) for n in SPLITS),
    }
    return {"ok": all(conditions.values()), "conditions": conditions}


def check_b2a(labelled=None):
    """B2-A: the A_c1d linear witness reads all 64 items through the real route_text."""
    label = labelled or (lambda it: combo_label("ac1d", it))
    answer = combo_witness_answer("ac1d")
    every = combo_items(COMBOS)
    right = sum(answer(combo_text("ac1d", it)) == label(it) for it in every)
    return {"ok": right == len(every), "conditions": {"witness_reads_all_64": right == len(every)},
            "right": right}


def check_witnesses():
    """The capacity witnesses (2.13 P1): two hand units read every item right on S0C1 (probe code), S1C1
    (history code), C1-d (all 64) and (ii) (check_a5's rule), and report their smallest margin."""
    cases = {
        "S0C1": [(probe_text(it), label_i("S0C1", it)) for it in ITEMS_I],
        "S1C1": [(history_text(it), label_i("S1C1", it)) for it in ITEMS_I],
        "c1d": [(combo_text("c1d", it), combo_label("c1d", it)) for it in combo_items(COMBOS)],
        "ii": [(TEXTS_II[uv], label_xor_ii(*uv)) for uv in PQ],
    }
    out = {}
    for name, pairs in cases.items():
        units = witness_units(name)
        right = sum(LABELS[xor_bit(units, code(t))] == lab for t, lab in pairs)
        margin = min(u.margin(code(t)) for t, _ in pairs for u in units)
        out[name] = {"right": right, "of": len(pairs), "min_margin": margin}
    return {"ok": all(v["right"] == v["of"] for v in out.values()), "cases": out}


def construction_check(seeds=(8000, 8001, 8002, 8003)):
    """Every learner-free construction check of P1; the split- and label-dependent ones on each of `seeds`."""
    result = {"b1": check_b1_all(), "b2": check_b2(), "b2_c1d": check_b2_combo("c1d"),
              "b2a": check_b2a(), "witnesses": check_witnesses()}
    per_seed = {}
    for seed in seeds:
        row = {"b10": check_b10(seed)["ok"], "b10s": check_b10s(seed)["ok"]}
        for family in COMBO_FAMILIES:
            parts, draws = combo_partition(family, seed)
            row[f"b13_{family}"] = check_b13(parts)["ok"]
            row[f"b14_{family}"] = check_b14(family, parts)["ok"]
            row[f"draws_{family}"] = draws
        per_seed[seed] = row
    result["per_seed"] = per_seed
    result["verdict"] = all(result[k]["ok"] for k in ("b1", "b2", "b2_c1d", "b2a", "witnesses")) and all(
        v for row in per_seed.values() for k, v in row.items() if not k.startswith("draws")
    )
    return result


# ---- the constructed reds ---------------------------------------------------------------------------------
LONG_WORD = "interpretation"  # 14 letters: as a first word it outranks the next word (1.24 > 1.23)


# distinct-from: faculties/access_consciousness.py -- a workspace slot-claim probe
# distinct-from: faculties/synfire_chain.py -- a packet-propagation predicate
# distinct-from: gates/amdahl_gate.py -- a pre-registered sham criterion
# distinct-from: gates/manifest_gate.py -- the manifest gate's own reds
# distinct-from: infra/demo.py -- an ordered-firing detector. Sites read 2026-10-04.
# C1-d's text with R1[0] replaced by the core slot's word 0 (`core` 'P' or 'Q'): a planted R/core collision
def collided_text(item, core):
    words = combo_text("c1d", item).split()
    if item[2] == 0:
        words[2] = COMBO_FAMILIES["c1d"][core][0]
    return " ".join(words)


# True: an R/P collision leaves the p/q identity intact on ranks AND code, so the registered R/P wording
# cannot fire on B2
def r_p_collision_keeps_additivity():
    return check_b2_combo("c1d", lambda it: collided_text(it, "P"))["ok"]


def _fires(result, clause):
    """A red fired iff its check failed AND the named clause is the one that failed."""
    return (not result["ok"]) and (not result["conditions"][clause])


def construction_reds(seed=8900):
    """Every P1 red, each a perturbation that must fail its named clause. `seed` is an EXPOSED scratch seed."""
    words_i = (*CUE, FILLER, *P_WORDS, *Q_WORDS)
    c1d_words = combo_words("c1d")
    parts, _ = combo_partition("c1d", seed)
    shared = {**parts, "future_d": parts["future_d"][:-1] + (parts["future"][0],)}
    no_r1_3 = [c for c in COMBOS if c[0] != 3]

    def swapped_pq(item):
        words = combo_text("c1d", item).split()
        return " ".join(words if item[0] == 0 else [words[i] for i in (1, 0, 2, 3)])

    fired = {
        "b1_bucket_below_8": _fires(check_b1(("eel",), ["eel"]), "buckets_at_least_8"),
        "b1_collision": _fires(check_b1(("ash", "plum"), ["ash plum"]), "distinct_buckets"),
        "b1_long_first_word": _fires(
            check_b1((LONG_WORD, "fox"), [f"{LONG_WORD} fox"]), "ranks_by_position"
        ),
        "b1_seven_word_text": _fires(
            check_b1(words_i, ["jar box oak fox elm owl pen"]), "texts_within_horizon"
        ),
        "b1_go_home": _fires(check_b1(("go", "home"), ["go home"]), "firewall_reject"),
        "b1_c1d_word_in_aii_bucket": _fires(
            check_b1((*c1d_words[:-1], "ram"), [combo_text("c1d", (0, 0, 0, 0))], _others("c1d")),
            "disjoint_from_other_families",
        ),
        "b2a_probe_carries_cue": _fires(
            check_b2(probe=lambda it: f"{CUE[it[0]]} {P_WORDS[it[1]]} {Q_WORDS[it[2]]}"),
            "probe_code_cue_free",
        ),
        "b2b_q_position_depends_on_p": _fires(
            check_b2(probe=lambda it: probe_text(it) if it[1] == 0 else f"{Q_WORDS[it[2]]} {P_WORDS[it[1]]}"),
            "ranks_additive",
        ),
        "b2c_xor_in_separable_slot": _fires(check_b2(slot_cell="S0C1"), "witness_reads_separable_8_of_8"),
        "b2_c1d_q_position_depends_on_p": _fires(
            check_b2_combo("c1d", swapped_pq), "ranks_additive_every_combo"
        ),
        # cp1 #14: an R word colliding with a CORE word. Measured 2026-10-04: an R/Q collision breaks the identity
        # in the learner's PSP code (deviation 0.081: a doubled Q[0] bucket moves P's rank with q, and P's rank
        # crosses t_post where the PSP is 0), while the integer ranks stay additive (the empty-bucket ranks shift
        # together). An R/P collision keeps both (rank 0, code 1.1e-16): P sits at the first position, so Q keeps
        # rank 2. B1's collision refusal covers the R/P case, as `r_p_collision_keeps_additivity` pins.
        "b2_c1d_r_q_collision": _fires(
            check_b2_combo("c1d", lambda it: collided_text(it, "Q")), "code_additive_every_combo"
        ),
        "b10_s0c0_labels": _fires(
            check_b10(seed, labeller=lambda s, i, k: LABELS[(i >> 1) & 1]), "every_item_both_labels"
        ),
        "b10s_shared_index": _fires(check_b10s(seed, read_indices=(0, 0)), "reads_draw_distinct_streams"),
        "b13_future_future_d_shared": _fires(check_b13(shared), "four_way_disjoint"),
        "b13_r1_3_missing_from_teach": _fires(
            check_b13({"teach": tuple(no_r1_3[:8]), "validation": (), "future": (), "future_d": ()}),
            "teach_covers_every_distractor_word",
        ),
        "b13_future_d_of_2": _fires(
            check_b13({**parts, "future_d": parts["future_d"][:2]}), "sizes_at_least_registered"
        ),
        "b14_witness_on_unlearnable_labels": _fires(
            check_b14("c1d", parts, labelled=lambda it: LABELS[it[2] & 1]), "witness_solves_every_split"
        ),
        "b2a_xor_labels": _fires(check_b2a(labelled=lambda it: LABELS[it[0] ^ it[1]]), "witness_reads_all_64"),
    }
    return {"fired": fired, "all_fired": all(fired.values())}


def main():
    """The fast self-check (seconds, no learner taught, no Brain): every construction check on dev seed 8000 and
    every constructed red on exposed seed 8900."""
    result = construction_check(seeds=(8000,))
    assert result["verdict"], result
    reds = construction_reds()
    assert reds["all_fired"], reds["fired"]
    w = result["witnesses"]["cases"]
    print(
        "self-check OK: B1 on (i), C1-d, A_c1d, A_ii; B2 max code deviation "
        f"{result['b2']['max_code_deviation']:.3g}, C1-d {result['b2_c1d']['max_code_deviation']:.3g}; "
        f"witness margins S0C1 {w['S0C1']['min_margin']:.4f}, S1C1 {w['S1C1']['min_margin']:.4f}, "
        f"C1-d {w['c1d']['min_margin']:.4f}, (ii) {w['ii']['min_margin']:.4f}; "
        f"seed 8000 partition draws c1d {result['per_seed'][8000]['draws_c1d']}, "
        f"ac1d {result['per_seed'][8000]['draws_ac1d']}; "
        f"{sum(reds['fired'].values())} of {len(reds['fired'])} reds fired"
    )


if __name__ == "__main__":
    main()
