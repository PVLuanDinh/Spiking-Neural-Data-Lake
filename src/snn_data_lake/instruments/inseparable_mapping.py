r"""G2(ii)'s inseparable mapping in the learner's own code, and G-R2's construction check on it.

The plan's G2 / T110 row, item (ii) (PLAN-FINAL), is a mapping IN THE LEARNER'S REAL CODE. Four texts
that differ only in word order, "cat dog sun sky", "cat dog sky sun", "dog cat sun sky" and "dog cat sky
sun", with u = "dog precedes cat" and v = "sky precedes sun", labelled by the XOR of u and v: GREET when
u == v, else GOAL (the plan's A and B, as two real intents of `engine/live_brain._IXN_INTENTS`).

Why the learner cannot separate it. The four words hash to four distinct buckets of 52 (cat 28, dog 41,
sun 31, sky 51), so a text's rank-order spike code is four word spikes at ranks 0-3, and each word's
rank depends on ONE of the two bits (cat and dog on u, sun and sky on v). The learner's drive is linear in
the PSP of that code, so every neuron's score is a(u) + b(v), and the difference of any two neurons'
scores is additive too: phi00 + phi11 = phi01 + phi10. An argmax over such scores can never answer all
four XOR texts (GREET at 00 and 11 needs D(00) + D(11) > 0, GOAL at 01 and 10 needs D(01) + D(10) < 0).
That argument is EXACT arithmetic. The learner sums its drive in floating point, so at a constructed
near-tie a rounding can flip a winner: the G1(c) P2 panel's hand-built setting (dev seed 7100; GREET - GOAL
is -1.74e-16 exactly on all four texts, so an exact argmax reads 2 of 4) reads XOR 4 of 4 through the real
`route_text` (pinned in the tests; a carried opening for the live learner). A3 is therefore a measurement of
the instrument, not a consequence of A2.
A5's two threshold units over the same code do separate it: the inseparability is the readout's, not the
code's.

The separable twin (strengthening; the plan names it, never defines it): the same four texts and the same
code, labelled GREET iff "cat precedes dog" (u == 0). The bare learner reaches it, and a witness weight
setting reads it 4 of 4 through the real `route_text` while reading the XOR labelling 2 of 4.

Construction check (g1c-carried-state.md 2.8, G-R2's clauses 1-2; run first, at the campaign head):
  A1  the four words occupy four distinct buckets of 52, the plan's 28, 41, 31, 51, none in buckets 0-1
  A2  phi00 + phi11 = phi01 + phi10 in the learner's PSP-of-rank code: exactly on the integer ranks,
      within 1e-12 in float
  A3  the bare incumbent never answers all four XOR texts at any step on any check seed, and 0 of 2,000
      random weight settings do (A2 entails this in exact arithmetic only; in the learner's float drive
      both zeros are measurements, see above)
  A4  the bare incumbent learns the twin (4 of 4 at each of the last 10 epoch ends on every check seed,
      no bystander intent winning), and the witness weights read 4 of 4 on the twin, 2 of 4 on XOR
  A5  (witness only) two hand threshold units over the code separate XOR
Each has a constructed red in `construction_reds`; the stream's reds are in `stream_reds`.

Also here: `stream`, the synthetic pool-reading stream (rb6_stream's schema) for G9's reading, and its
`stream_checks`. G1(c)'s campaign (item (3)), G2, G3 and G9 import this module. G2's presized nonlinear
layer and delayed-cue controls are NOT here (T110 builds them); only the generator and the construction
check are. No function here learns anything: the teaching goes through the incumbent's own production
path, `learn_turn(text, None, feedback=label)`, which is what G0's correction path calls.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.inseparable_mapping

Deterministic; ASCII-only prints. The development measurement (learner seeds 7100-7103 only) set
TEACH_EPOCHS; the check seeds 7000-7007 are reachable only through `construction_check`'s default.
"""

import copy
import functools
import json
import math
import random

from snn_data_lake.readout import _IXN_INTENTS
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner, text_features

# distinct-from: instruments/memory_diagnostic.py::rb6_stream -- that builds the R-b6 symbol stream (categories are blocks of a permuted 64); `stream` here only BORROWS its frame (seed, grower config, policy, revision) and fills the events with the 52-dim PSP code of four texts, c_t = label(text_{t-1}), and a "g2ii" synthetic block

# ---- the generator (the plan's G2(ii); g1c-carried-state.md 2.8) ----------------------------------
WORDS = ("cat", "dog", "sun", "sky")
UV = ((0, 0), (0, 1), (1, 0), (1, 1))  # (u, v): u = "dog precedes cat", v = "sky precedes sun"
TEXTS = {
    (0, 0): "cat dog sun sky",
    (0, 1): "cat dog sky sun",
    (1, 0): "dog cat sun sky",
    (1, 1): "dog cat sky sun",
}
# the plan's numbers: ASSERTED by A1 against the learner's own featurizer, never used to build anything
BUCKETS_52 = {"cat": 28, "dog": 41, "sun": 31, "sky": 51}
N_IN = 52  # engine/live_brain.py's SpikingInteractionLearner(..., n_in=52, ..., t_post=6), pinned by the test
T_POST = 6
GREET = "GREET"
GOAL = "GOAL"
assert GREET != GOAL and GREET in _IXN_INTENTS and GOAL in _IXN_INTENTS, "two real intents"
STREAM_INTENTS = (GREET, GOAL)  # a stream category is the index in this pair


def label_xor(u, v):
    """The plan's A/B: GREET iff u == v (cat dog sun sky and dog cat sky sun), else GOAL."""
    return GREET if u == v else GOAL


def label_twin(u, v):
    """The separable twin: GREET iff "cat" precedes "dog" (u == 0), else GOAL."""
    return GREET if u == 0 else GOAL


# ---- the check's frozen numbers -------------------------------------------------------------------
STABLE = 10  # A4: 4 of 4 at each of the last STABLE epoch ends
# The teaching schedule: 2 x the largest development epochs_to_stable, rounded up to a multiple of 50.
# Measured on learner seeds 7100, 7101, 7102, 7103 (twin labelling, max_epochs 4000, STABLE 10):
# epochs_to_stable 122, 170, 112, 124 -> 2 * 170 = 340 -> 350. The first presentation at which all four
# texts read right was step 129, 106, 112, 84 (steps, not epochs). All four seeds read 4 of 4 at every
# epoch end after epoch end 179, 170, 162, 161 respectively, through epoch 4000.
# DATED 2026-10-03 (g1c 2.8 amendment, pre-data): the rule reads the STRONGER stability, the last epoch end not
# reading 4 of 4 within 4000 epochs (179, 170, 162, 161; reproduced by the orchestrator): 2 * 179 = 358 -> 400.
TEACH_EPOCHS = 400
RANDOM_SETTINGS = 2000  # A3: random weight settings (all 9 neurons, U(-1, 1))
RANDOM_SETTINGS_SEED = 20261003  # an RNG seed for those settings, not a recording seed
ADDITIVITY_TOL = 1e-12  # A2: the float additivity bound (the plan read 1.1e-16)
# A5: the two hand units fire above -0.05 and above +0.05 (an arbitrary round pair; the margin it
# leaves is reported, never tuned)
WITNESS_THRESHOLD = 0.05

# ---- the pool-reading stream (g1c 2.8 "Pool reading"; rb6_stream's schema) --------------------------
STREAM_EVENTS = 1200
STREAM_WARM_UP = 256
STREAM_NORM = 1.73  # the live percepts' L2 norm
STREAM_DIM = 64
FEED = "psp52_pad64_norm1.73"
# A frequency must lie within this many binomial sd of its i.i.d. value. 6, not 4 (g1c 2.8 dated note): a failed
# stream check HALTS the campaign, and five such tests on each of 31 seeds put a valid stream's chance halt near 1 % at
# 4 sd (~3e-7 at 6); the permutation-block red these tests carry sits near 15 sd.
SD_BOUND = 6


# ---- the learner's own code --------------------------------------------------------------------------
@functools.cache
def _probe(n_in=N_IN):
    """A bare incumbent used only for its `encode` and `_psp` (a read-only instrument: nothing here
    ever teaches it), so the code is the learner's own and not a re-implementation of it."""
    return SpikingInteractionLearner(list(_IXN_INTENTS), n_in=n_in, seed=0, t_post=T_POST)


def ranks(text, n_in=N_IN):
    """The integer spike ranks of a text: text_features -> the learner's own rank-order `encode`."""
    return _probe(n_in).encode(text_features(text, n_in))


def code(text, n_in=N_IN):
    """The learner's PSP-of-rank code of a text: `_psp(t_post - rank)` per synapse, exactly the factor
    its `_drive` multiplies each weight by."""
    unit = _probe(n_in)
    return [unit._psp(unit.t_post - rank) for rank in ranks(text, n_in)]


def _bucket(word, n_in=N_IN):
    """The one synapse a lone word drives, read off the learner's own featurizer."""
    hit = [i for i, value in enumerate(text_features(word, n_in)) if value]
    assert len(hit) == 1, (word, hit)
    return hit[0]


def _incumbent(seed, lr=None):
    """The bare incumbent as the live brain builds it (all nine intents, n_in 52, t_post 6); `lr` None
    keeps its default."""
    options = {} if lr is None else {"lr": lr}
    return SpikingInteractionLearner(
        list(_IXN_INTENTS), n_in=N_IN, seed=seed, t_post=T_POST, **options
    )


def _read(unit, labelling):
    """(answers matching `labelling` among the four texts, answers outside {GREET, GOAL}), each read
    through the real `route_text`."""
    reads = [unit.route_text(TEXTS[uv]) for uv in UV]
    correct = sum(read == labelling(*uv) for read, uv in zip(reads, UV, strict=True))
    return correct, sum(read not in STREAM_INTENTS for read in reads)


def _schedule(seed, epochs):
    """The one teaching schedule: each epoch presents the four texts once, in the order of
    `random.Random(seed).sample`. A3, A4, the development measurement and every A4 red consume it."""
    rng = random.Random(seed)
    for _ in range(epochs):
        yield rng.sample(UV, 4)


def _shuffled(seed, labelling):
    """A taught-label source for the shuffled red: each epoch's labels are a re-arrangement of the true
    ones, re-drawn until it differs from them, so no epoch is ever taught the true mapping."""
    rng = random.Random(seed + 1_000_003)

    def taught(order):
        true = [labelling(*uv) for uv in order]
        while True:
            labels = list(true)
            rng.shuffle(labels)
            if labels != true:
                return labels

    return taught


def _run(seed, labelling, epochs, *, every_step=True, lr=None, taught=None, stop_when_stable=False):
    """Teach a bare incumbent `labelling` through `learn_turn(text, None, feedback=label)` for `epochs`
    epochs of `_schedule(seed, epochs)`, reading `route_text` on all four texts after EVERY presentation
    (`every_step`) or only at epoch ends. Returns the per-epoch-end correct counts and bystander counts,
    the maximum correct count read, and the first presentation (1-based) at which all four read right.
    `taught(order)` replaces the true labels of an epoch (a red); `stop_when_stable` stops at the first
    run of STABLE consecutive epoch ends reading 4 of 4."""
    unit = _incumbent(seed, lr)
    steps, first_step, best, run_length = 0, None, 0, 0
    ends, bystanders = [], []
    for order in _schedule(seed, epochs):
        labels = [labelling(*uv) for uv in order] if taught is None else taught(order)
        for uv, label in zip(order, labels, strict=True):
            unit.learn_turn(TEXTS[uv], None, feedback=label)
            steps += 1
            if every_step:
                correct, outside = _read(unit, labelling)
                best = max(best, correct)
                if correct == 4 and first_step is None:
                    first_step = steps
        if not every_step:
            correct, outside = _read(unit, labelling)
            best = max(best, correct)
            if correct == 4 and first_step is None:
                first_step = steps
        ends.append(correct)
        bystanders.append(outside)
        run_length = run_length + 1 if correct == 4 else 0
        if stop_when_stable and run_length == STABLE:
            break
    return {
        "epoch_ends": ends,
        "bystander_ends": bystanders,
        "max_correct": best,
        "first_step_4of4": first_step,
        "steps": steps,
    }


def epochs_to_stable(seed, labelling, max_epochs):
    """The number e of epochs completed before the FIRST run of STABLE consecutive epoch ends that all
    read 4 of 4 on `labelling`: the stable run occupies epoch ends e + 1 .. e + STABLE (e = 0: 4 of 4
    from the first epoch end). None if no such run begins within `max_epochs` epochs. Epoch ends only;
    the per-presentation reading is `_run`'s."""
    run = _run(seed, labelling, max_epochs, every_step=False, stop_when_stable=True)
    if len(run["epoch_ends"]) < STABLE or not all(c == 4 for c in run["epoch_ends"][-STABLE:]):
        return None
    return len(run["epoch_ends"]) - STABLE


def _epochs(epochs):
    # None means the module's schedule length; one shorter than the stable tail cannot be judged
    epochs = TEACH_EPOCHS if epochs is None else epochs
    if epochs is None or epochs < STABLE:
        raise ValueError(f"epochs {epochs!r}: at least {STABLE} are needed (TEACH_EPOCHS is unset)")
    return int(epochs)


def _resolve(check_seeds, epochs):
    # The seed constants live in carried_state_campaign; the campaign's separation stage
    # (instruments/carried_state_controls) imports THIS module,
    # so the import is deferred to here: a top-level import in both directions would make the first
    # import of either order-dependent.
    if check_seeds is None:
        from snn_data_lake.instruments.carried_state_campaign import G2II_CHECK_SEEDS

        check_seeds = G2II_CHECK_SEEDS
    return _distinct(check_seeds), _epochs(epochs)


def _distinct(seeds):
    # delegates to carried_state_campaign.distinct_seeds, the one shared check (P2 panel fold); the
    # import is deferred for the reason _resolve gives
    from snn_data_lake.instruments.carried_state_campaign import distinct_seeds

    return distinct_seeds(seeds)


def _dev_seed():
    from snn_data_lake.instruments.carried_state_campaign import G2II_DEV_SEEDS

    return G2II_DEV_SEEDS[0]


# ---- A1, A2 -------------------------------------------------------------------------------------------
def check_a1(texts=TEXTS, buckets=BUCKETS_52, n_in=N_IN, words=WORDS):
    """A1: the words occupy distinct buckets of `n_in`, the plan's `buckets`, none of them in buckets 0-1
    (the two lowest-index empty buckets take ranks 4 and 5 identically in every text); every text is the
    four words once each, the four texts differ, and in each the four words hold ranks 0-3."""
    derived = {word: _bucket(word, n_in) for word in words}
    conditions = {
        "distinct_buckets": len(set(derived.values())) == len(words) == 4,
        "plans_buckets": derived == dict(buckets),
        "outside_buckets_0_1": all(bucket not in (0, 1) for bucket in derived.values()),
        "texts_are_the_words_once": len(texts) == 4
        and all(sorted(text.split()) == sorted(WORDS) for text in texts.values())
        and len(set(texts.values())) == 4,
        "words_hold_ranks_0_3": all(
            sorted(ranks(text, n_in)[b] for b in derived.values()) == [0, 1, 2, 3]
            for text in texts.values()
        ),
    }
    return {"ok": all(conditions.values()), "conditions": conditions, "buckets": derived}


def check_a2(texts=TEXTS):
    """A2: phi00 + phi11 = phi01 + phi10 in the learner's code, exactly on the integer ranks and within
    ADDITIVITY_TOL in float, and the four codes differ (an additivity of identical codes is vacuous)."""
    r = {uv: ranks(texts[uv]) for uv in UV}
    p = {uv: code(texts[uv]) for uv in UV}
    rank_dev = max(abs(r[0, 0][i] + r[1, 1][i] - r[0, 1][i] - r[1, 0][i]) for i in range(N_IN))
    code_dev = max(abs(p[0, 0][i] + p[1, 1][i] - p[0, 1][i] - p[1, 0][i]) for i in range(N_IN))
    distinct = all(p[a] != p[b] for k, a in enumerate(UV) for b in UV[k + 1 :])
    conditions = {
        "ranks_additive": rank_dev == 0,
        "code_additive": code_dev <= ADDITIVITY_TOL,
        "codes_distinct": distinct,
    }
    return {
        "ok": all(conditions.values()),
        "conditions": conditions,
        "max_rank_deviation": rank_dev,
        "max_code_deviation": code_dev,
    }


# ---- the witness and A5 ---------------------------------------------------------------------------------
def _witness_unit():
    """A bare incumbent whose weights are the witness: GREET +1 at cat's bucket and -1 at dog's, GOAL the
    negative, every other neuron 0. Buckets come from the featurizer (BUCKETS_52 builds nothing)."""
    unit = _incumbent(0)
    cat, dog = _bucket("cat"), _bucket("dog")
    for index, name in enumerate(unit.responses):
        weights = [0.0] * N_IN
        if name == GREET:
            weights[cat], weights[dog] = 1.0, -1.0
        elif name == GOAL:
            weights[cat], weights[dog] = -1.0, 1.0
        unit.neurons[index].w = weights
    return unit


def witness_reads():
    """(answers right on the twin, answers right on XOR) of the witness weights through `route_text`."""
    unit = _witness_unit()
    return _read(unit, label_twin)[0], _read(unit, label_xor)[0]


def check_a5(texts=TEXTS):
    """A5 (witness only): two hand threshold units over the code, w.phi > -0.05 and w.phi > +0.05 with w =
    +1 at dog and sky, -1 at cat and sun, answer GOAL iff the first fires and the second does not (an
    s = w.phi in the middle band), and so separate XOR. Reports the smallest margin to a threshold."""
    w = [0.0] * N_IN
    w[_bucket("dog")], w[_bucket("sky")] = 1.0, 1.0
    w[_bucket("cat")], w[_bucket("sun")] = -1.0, -1.0
    answers, margins = {}, []
    for uv in UV:
        s = sum(wi * pi for wi, pi in zip(w, code(texts[uv]), strict=True))
        first, second = s > -WITNESS_THRESHOLD, s > WITNESS_THRESHOLD
        answers[uv] = GOAL if first and not second else GREET
        margins.append(min(abs(s + WITNESS_THRESHOLD), abs(s - WITNESS_THRESHOLD)))
    right = sum(answers[uv] == label_xor(*uv) for uv in UV)
    return {"ok": right == 4, "right": right, "min_margin": min(margins)}


# ---- A3, A4 ---------------------------------------------------------------------------------------------
def _random_hits(settings, labelling):
    """How many of `settings` random weight settings (every neuron U(-1, 1), one fixed RNG) read all
    four texts right on `labelling` through the real `route_text`."""
    rng = random.Random(RANDOM_SETTINGS_SEED)
    unit = _incumbent(0)
    hits = 0
    for _ in range(settings):
        for neuron in unit.neurons:
            neuron.w = [rng.uniform(-1.0, 1.0) for _ in range(N_IN)]
        hits += _read(unit, labelling)[0] == 4
    return hits


def check_a3(seeds, epochs, *, labelling=label_xor, settings=RANDOM_SETTINGS):
    """A3: the bare incumbent taught `labelling` (XOR) for `epochs` epochs never reads more than 3 of 4
    right at any presentation on any of `seeds`, and 0 of `settings` random weight settings read 4 of 4.
    A2's additivity entails both clauses in exact arithmetic only; the learner's float drive admits
    rounding exceptions at constructed near-ties (the module docstring), so both are measurements."""
    seeds = _distinct(seeds)
    runs = {seed: _run(seed, labelling, epochs) for seed in seeds}
    best = max(run["max_correct"] for run in runs.values())
    hits = _random_hits(settings, labelling)
    conditions = {"never_4_of_4": best <= 3, "no_random_setting_4_of_4": hits == 0}
    return {
        "ok": all(conditions.values()),
        "conditions": conditions,
        "max_correct": best,
        "max_correct_by_seed": {seed: run["max_correct"] for seed, run in runs.items()},
        "random_settings": settings,
        "random_4_of_4": hits,
        "presentations": sum(run["steps"] for run in runs.values()),
    }


def check_a4(seeds, epochs, *, labelling=label_twin, lr=None, shuffled=False):
    """A4: the bare incumbent taught `labelling` (the twin) reads 4 of 4 at each of the last STABLE epoch
    ends on every one of `seeds`, with no bystander intent winning there, and the witness weights read 4 of
    4 on `labelling` and 2 of 4 on XOR. `lr` and `shuffled` build the reds."""
    if epochs < STABLE:
        raise ValueError(f"epochs {epochs}: at least {STABLE} are needed")
    seeds = _distinct(seeds)
    rows = {}
    for seed in seeds:
        taught = _shuffled(seed, labelling) if shuffled else None
        run = _run(seed, labelling, epochs, lr=lr, taught=taught)
        tail = run["epoch_ends"][-STABLE:]
        rows[seed] = {
            "tail": tail,
            "stable": all(c == 4 for c in tail),
            "bystanders": sum(run["bystander_ends"][-STABLE:]),
            "first_step_4of4": run["first_step_4of4"],
        }
    witness_slot = _read(_witness_unit(), labelling)[0]
    witness_xor = witness_reads()[1]
    conditions = {
        "stable_on_every_seed": all(row["stable"] for row in rows.values()),
        "no_bystander_in_the_tail": all(row["bystanders"] == 0 for row in rows.values()),
        "witness_4_of_4": witness_slot == 4,
        "witness_2_of_4_on_xor": witness_xor == 2,
    }
    return {
        "ok": all(conditions.values()),
        "conditions": conditions,
        "by_seed": rows,
        "witness_on_slot": witness_slot,
        "witness_on_xor": witness_xor,
    }


def construction_check(check_seeds=None, epochs=None):
    """G-R2's construction check: verdict = A1-A4 all ok; A5 (witness only) is reported beside it. The
    defaults are the campaign's (check seeds 7000-7007, TEACH_EPOCHS); every caller outside the campaign
    passes the development seeds explicitly."""
    seeds, epochs = _resolve(check_seeds, epochs)
    result = {
        "a1": check_a1(),
        "a2": check_a2(),
        "a3": check_a3(seeds, epochs),
        "a4": check_a4(seeds, epochs),
        "a5": check_a5(),
        "seeds": seeds,
        "epochs": epochs,
    }
    result["verdict"] = all(result[key]["ok"] for key in ("a1", "a2", "a3", "a4"))
    return result


def _fires(result, clause):
    """A red fired iff its check failed AND the named clause is the one that failed."""
    return (not result["ok"]) and (not result["conditions"][clause])


def construction_reds(seeds=None, epochs=None):
    """The constructed reds of A1-A4, each a perturbation of the real check that must make it fail on
    its named clause: A1 (the text "cat dog sun sun"; n_in 3; BUCKETS with cat and dog swapped), A2 (text
    11 replaced by "dog sky cat sun"), A3 (label_twin through the XOR counter reaches 4 of 4), A4 (lr 0.0;
    shuffled teaching labels; label_xor in the twin slot). A red needs one failing seed, so the default
    is ONE development seed, never a check seed."""
    seeds = (_dev_seed(),) if seeds is None else tuple(seeds)
    epochs = _epochs(epochs)
    swapped = {**BUCKETS_52, "cat": BUCKETS_52["dog"], "dog": BUCKETS_52["cat"]}
    a3 = check_a3(seeds, epochs, labelling=label_twin, settings=0)
    fired = {
        "a1_duplicate_word_text": _fires(
            check_a1(texts={**TEXTS, (1, 1): "cat dog sun sun"}), "texts_are_the_words_once"
        ),
        "a1_n_in_3": _fires(check_a1(n_in=3), "plans_buckets"),
        "a1_swapped_buckets": _fires(check_a1(buckets=swapped), "plans_buckets"),
        "a2_text_11_replaced": _fires(
            check_a2(texts={**TEXTS, (1, 1): "dog sky cat sun"}), "ranks_additive"
        ),
        "a3_twin_through_the_xor_counter": _fires(a3, "never_4_of_4"),
        "a4_lr_zero": _fires(check_a4(seeds, epochs, lr=0.0), "stable_on_every_seed"),
        "a4_shuffled_labels": _fires(
            check_a4(seeds, epochs, shuffled=True), "stable_on_every_seed"
        ),
        "a4_xor_in_the_twin_slot": _fires(
            check_a4(seeds, epochs, labelling=label_xor), "stable_on_every_seed"
        ),
    }
    return {
        "fired": fired,
        "all_fired": all(fired.values()),
        "a3_red_max_correct": a3["max_correct"],
    }


# ---- the pool-reading stream ------------------------------------------------------------------------------
def _scaled(text):
    """x for a text: its 52-dim PSP code at L2 norm STREAM_NORM, zero-padded to STREAM_DIM."""
    values = code(text)
    scale = STREAM_NORM / math.sqrt(math.fsum(v * v for v in values))
    return [v * scale for v in values] + [0.0] * (STREAM_DIM - N_IN)


def _label_name(labelling):
    return getattr(labelling, "__name__", "").removeprefix("label_")


def _frame(seed):
    """rb6_stream's seed, grower config, policy and revision stamp, which this stream's schema shares
    (replay refuses a policy that is not at one thread, so rb6_stream's own thread pin must run)."""
    from snn_data_lake.instruments import memory_diagnostic as md

    base = md.rb6_stream(seed)
    return {key: base[key] for key in ("seed", "config", "policy", "final")}


def _build_stream(
    seed, labelling, *, frame=None, n_events=STREAM_EVENTS, lag=1, blocks=False, rng_seed=None
):
    """The stream of one (seed, labelling). The keyword arguments are the reds' perturbations of this
    one constructor: `lag` 0, a short `n_events`, texts in permutation `blocks` of 4, another `rng_seed`."""
    frame = copy.deepcopy(_frame(seed) if frame is None else frame)
    rng = random.Random(seed if rng_seed is None else rng_seed)
    order = []  # order[0] is the virtual text_0; event i shows order[i + 1]
    while len(order) < n_events + 1:
        order.extend(rng.sample(range(4), 4) if blocks else [rng.randrange(4)])
    order = order[: n_events + 1]
    xs = {k: _scaled(TEXTS[uv]) for k, uv in enumerate(UV)}
    events, sleep = [], []
    for i in range(n_events):
        previous = UV[order[i + 1 - lag]]
        events.append(
            {
                "t": i + 1,
                "tick": 2 * i + 1,
                "x": list(xs[order[i + 1]]),
                "c": STREAM_INTENTS.index(labelling(*previous)),
                "causal_lp": None,
                "segment": "warm-up" if i < STREAM_WARM_UP else "unseen-order",
            }
        )
        sleep.append({"tick": i + 1, "brain_tick": 2 * i + 2, "event": i + 1, "live_lp": None})
    return {
        "seed": frame["seed"],
        "synthetic": {
            "family": "g2ii",
            "labelling": _label_name(labelling),
            "intents": list(STREAM_INTENTS),
            "feed": FEED,
        },
        "config": frame["config"],
        "policy": frame["policy"],
        "events": events,
        "sleep": sleep,
        "final": frame["final"],
    }


def stream(seed, labelling):
    """G2(ii)'s pool-reading stream for one live seed (g1c 2.8): 1,200 events, texts i.i.d. uniform from
    `random.Random(seed)` (text_0 a virtual first draw), x = the code scaled to norm 1.73 and zero-padded
    to 64, c_t = category(label(text_{t-1})) (the decoder predicts c_t from phi_{t-1}, so each label
    answers the previous presentation), the first 256 events "warm-up" and the rest "unseen-order". The
    XOR and twin streams of one seed share every x byte. No "text" key rides on an event: g1c 2.5's
    within-block shuffle refuses an event key outside its moved and slot sets, so a text is read back
    from x (the four codes are distinct) by `stream_checks`."""
    return _build_stream(seed, labelling)


def _decode(x, references):
    """The (u, v) whose code x equals within 1e-12, or None (not exactly one)."""
    hits = [
        uv
        for uv in UV
        if len(x) == STREAM_DIM and max(abs(a - b) for a, b in zip(x, references[uv])) <= 1e-12
    ]
    return hits[0] if len(hits) == 1 else None


def _within(count, n, p):
    return abs(count / n - p) <= SD_BOUND * math.sqrt(p * (1 - p) / n)


def stream_checks(seed, *, xor=None, twin=None):
    """The checks of one seed's XOR and twin streams (built here unless passed, for the reds): more than
    256 events (and exactly 1,200); every x of norm 1.73 within 1e-12 and zero outside dims 0-51; every x
    a text's code; each text within SD_BOUND (6) binomial sd of 25 %; the lag-1 repeat rate within 6 sd of 25 %
    (strengthening: the i.i.d. witness that catches texts drawn in permutation blocks, where every text
    is exactly 25 %); the xor and twin x events byte-identical; c_t == category(label(text_{t-1})) re-derived
    from the decoded texts and, for t = 1, the first `random.Random(seed)` draw."""
    xor = stream(seed, label_xor) if xor is None else xor
    twin = stream(seed, label_twin) if twin is None else twin
    events = xor["events"]
    n = len(events)
    references = {uv: _scaled(TEXTS[uv]) for uv in UV}
    decoded = [_decode(e["x"], references) for e in events]
    norms = [math.sqrt(math.fsum(v * v for v in e["x"])) for e in events]
    counts = {uv: decoded.count(uv) for uv in UV}
    repeats = sum(a == b for a, b in zip(decoded, decoded[1:], strict=False))
    virtual = UV[random.Random(seed).randrange(4)]

    def answers_previous(source, labelling):
        # each stream's texts are read back from its own x; text_0 is the first RNG draw
        texts = [_decode(e["x"], references) for e in source["events"]]
        previous = [virtual, *texts[:-1]]
        return all(
            prior is not None and e["c"] == STREAM_INTENTS.index(labelling(*prior))
            for e, prior in zip(source["events"], previous, strict=True)
        )

    checks = {
        "beyond_warm_up": n > STREAM_WARM_UP,
        "exactly_1200_events": n == STREAM_EVENTS,
        "x_norm_1_73": all(abs(v - STREAM_NORM) <= 1e-12 for v in norms),
        "x_zero_outside_dims_0_51": all(
            len(e["x"]) == STREAM_DIM and all(v == 0.0 for v in e["x"][N_IN:]) for e in events
        ),
        "x_is_a_text_code": all(d is not None for d in decoded),
        "texts_within_6_sd": n > 0 and all(_within(c, n, 0.25) for c in counts.values()),
        "repeat_rate_within_6_sd": n > 1 and _within(repeats, n - 1, 0.25),
        "xor_twin_x_identical": json.dumps([e["x"] for e in events])
        == json.dumps([e["x"] for e in twin["events"]]),
        "c_is_the_previous_texts_xor_label": answers_previous(xor, label_xor),
        "c_is_the_previous_texts_twin_label": answers_previous(twin, label_twin),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "text_counts": {f"{u}{v}": c for (u, v), c in counts.items()},
        "repeat_rate": repeats / (n - 1) if n > 1 else None,
        "events": n,
    }


# distinct-from: instruments/attention_schema.py::main -- that fires attention_schema's own six reds on its own scorer; this fires four perturbations of THIS stream constructor (lag, length, block order, RNG state) on stream_checks. distinct-from: faculties/gwt_deep.py::_reds -- another module's own constructed reds on its own binding criteria. distinct-from: faculties/reentrant_workspace.py::_reds -- another module's reds on its own claims. Sites read 2026-10-03
def stream_reds(seed):
    """The stream's four constructed reds, each a perturbation of `_build_stream`, each firing only
    if stream_checks fails on its named clause: lag 0, a 256-event stream, texts in permutation
    blocks of 4, a twin drawn from another RNG state."""
    frame = _frame(seed)

    def fires(clause, xor_options, twin_options):
        xor = _build_stream(seed, label_xor, frame=frame, **xor_options)
        twin = _build_stream(seed, label_twin, frame=frame, **twin_options)
        result = stream_checks(seed, xor=xor, twin=twin)
        return (not result["ok"]) and (not result["checks"][clause])

    fired = {
        "lag_0": fires("c_is_the_previous_texts_xor_label", {"lag": 0}, {"lag": 0}),
        "events_256": fires("beyond_warm_up", {"n_events": 256}, {"n_events": 256}),
        "permutation_blocks_of_4": fires(
            "repeat_rate_within_6_sd", {"blocks": True}, {"blocks": True}
        ),
        "twin_from_another_rng_state": fires("xor_twin_x_identical", {}, {"rng_seed": seed + 1}),
    }
    return {"fired": fired, "all_fired": all(fired.values())}


# ---- the self-check -------------------------------------------------------------------------------------
def main():
    """The fast self-check (cwd-independent, a few seconds): A1, A2, A5, the witness, every red, a SMOKE
    A3 and A4 on ONE development seed at TEACH_EPOCHS (a smoke of the real check on one seed, not the
    check seeds), and `stream_checks` and `stream_reds` on one seed. The full construction check is the
    campaign's."""
    assert TEACH_EPOCHS is not None and TEACH_EPOCHS >= STABLE
    dev = _dev_seed()
    a1, a2, a5 = check_a1(), check_a2(), check_a5()
    assert a1["ok"] and a1["buckets"] == BUCKETS_52, a1
    assert a2["ok"] and a2["max_rank_deviation"] == 0, a2
    assert a5["ok"] and a5["min_margin"] > 0.03, a5
    assert witness_reads() == (4, 2), witness_reads()
    a3 = check_a3((dev,), TEACH_EPOCHS)  # SMOKE: one development seed
    a4 = check_a4((dev,), TEACH_EPOCHS)  # SMOKE: one development seed
    assert a3["ok"] and a3["max_correct"] <= 3 and a3["random_4_of_4"] == 0, a3
    assert a4["ok"], a4
    reds = construction_reds()
    assert reds["all_fired"] and reds["a3_red_max_correct"] == 4, reds
    checks = stream_checks(dev)
    assert checks["ok"], checks
    stream_red = stream_reds(dev)
    assert stream_red["all_fired"], stream_red
    print(
        f"self-check OK (smoke on development seed {dev}, TEACH_EPOCHS {TEACH_EPOCHS}): A1 buckets "
        f"{a1['buckets']}; A2 rank deviation {a2['max_rank_deviation']}, code deviation "
        f"{a2['max_code_deviation']:.3g}; A3 max correct {a3['max_correct']} of 4 and "
        f"{a3['random_4_of_4']} of {a3['random_settings']} random settings at 4 of 4; A4 tail "
        f"{a4['by_seed'][dev]['tail']}, witness {a4['witness_on_slot']} of 4 on the twin and "
        f"{a4['witness_on_xor']} of 4 on XOR; A5 margin {a5['min_margin']:.4f}; "
        f"{sum(reds['fired'].values())} of {len(reds['fired'])} construction reds fired and "
        f"{sum(stream_red['fired'].values())} of {len(stream_red['fired'])} stream reds fired; "
        f"stream seed {dev}: {checks['events']} events, text counts {checks['text_counts']}, "
        f"repeat rate {checks['repeat_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
