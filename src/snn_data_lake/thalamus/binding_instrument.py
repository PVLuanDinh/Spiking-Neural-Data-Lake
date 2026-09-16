"""Temporal-binding instrument (specs/022 T080 [US4]) -- the LIVE cross-modal binder's report of
ONE object, read against its own features RE-PAIRED across windows at matched K.

THE ROW. tasks.md T080 [US4]; the C5 row `temporal_binding` in
`substrate/conclusion_protocol.py`; FR-028's binding item; register #6's binding line;
PLAN-FINAL Part F row C-R9. Pre-registration:
`specs/022-unified-brain-arc/evidence/t080-temporal-binding.md`, frozen at the commit that
landed it (sections 1-2.9). Nothing here may be read outside those sections' words.

THE C5 SENTENCES, verbatim from the conclusion protocol's own row:
  LICENSED -- "features inside one window are bound into one reported object more often than
              features across windows".
  REFUSED  -- "this licenses no claim that there is a felt present, a stream of experience,
              or a duration lived".
  DISSOCIATION INPUT -- "membership-matched pairing at matched K (the plan's primary null)
              must not reproduce the binding".

THE APPARATUS (2.1-2.2). A SCRATCH brain per seed under `substrate.home.throwaway_home()`
(`cortex.brain.make_brain(seed, persist=False)`; the daemon's brain and the durable lake are
never touched). At the STIMULUS -- and nowhere else -- the shipped SHD audio pack is
re-ordered so that stream position i carries a word whose label mod 10 names the digit of
vision position i (`t080_aligned_order`). The brain then teaches itself K = 10 cross-modal
objects through its OWN one-shot congruent-sighting rule, after which the taught set is
frozen and K is matched across every arm by construction. Six arms follow -- ALIGNED, NULL
IN VIVO (the same frames, positions permuted), LAG 1 / 2 / 5, SCRAMBLED PHASE -- and every
awake tick with both latents present is recorded whole. The barred quantity X is the
fraction of those scored ticks on which the binder reports ONE object.

WHAT IS MEASURED AGAINST WHAT. The decomposition (`t080_recognitions`) re-reads each
recorded tick through the binder's OWN `_project` / `_recognize`, so `bound` = (both
modalities recognised AND naming the same object) must equal (`handle` is not None) on every
scored tick -- two instruments required to agree before anything is reported. The PRIMARY
null (`t080_null`) permutes the audio-object column ACROSS the scored ticks: the same
multiset of audio features, the same taught K, only the pairing broken. Its q95 IS the
derived margin. The NULL IN VIVO arm applies the same permutation at the STIMULUS and must
land inside the offline null's band.

WHAT THIS DOES NOT CLAIM (2.7). No felt present, no stream of experience, no duration lived.
Nothing about a temporal INTEGRATION window: the window is the tick because the binder is
stateless between calls once taught, and the lag arms measure the mechanism's SHAPE, not a
tolerance. Nothing about the daemon's brain, whose two streams are independent, so its
reported bindings in the wild are chance congruences. Nothing about gwt_deep's K-object task
or theta-gamma phase codes. No aggregate consciousness score. Nothing about the classifiers'
accuracy beyond its reported role as the yield's ceiling. A green reading here is evidence
about a MECHANISM, never a verdict about experience.

# distinct-from: thalamus/gwt_deep.py -- a SYNTHETIC K-object phase-multiplexing benchmark
#   with its own spiking-vs-rate control, scoring a pairing this tree's live tick never
#   performs. This module runs no synthetic binding task at all: it reads the LIVE binder's
#   own report off a recorded brain and scores it against a permutation of that recording.
#   gwt_deep is not a control here and is not run. Read.
# distinct-from: cortex/crossmodal_binder.py -- the MECHANISM UNDER TEST. This module changes
#   nothing in it: it calls its public `bind` and its own `_project` / `_recognize` and
#   re-implements neither. The margin (0.15) and the cue strength (0.055) are READ off the
#   shipped binder, never re-declared. Read.
# distinct-from: thalamus/ignition_instrument.py -- an EXEMPT scorer over scratch
#   ReentrantWorkspace subjects, scoring an all-or-none DRIVE transition. Different subject,
#   different quantity; the EXEMPT-instrument shape is the thing reused. Read.
# distinct-from: cortex/agency_monitor.py's T079 functions -- the same campaign/dry-run/pin
#   PATTERN over a different instrument, different arms and a different pin. No state, no pin
#   and no statistic is shared beyond `substrate/paired_contrast`, which both IMPORT. Read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.thalamus.binding_instrument
"""

import copy
import statistics
import sys
from pathlib import Path

import numpy as np

from snn_data_lake.substrate._mathutils import q95 as _q95
from snn_data_lake.substrate.paired_contrast import T_BAR, contrast

# =============================================================================
# 2.8's DECLARED CONSTANTS. Every one is either the pack's own shape, the plan's width, or a
# quantity READ off the shipped binder -- none is a magic number chosen here (P1).
# =============================================================================
#: The taught object count the warm-up drives to: the N-MNIST digit classes (2.8).
T080_K = 10
#: The warm-up cap in ticks; a seed that has not taught K by here REFUSES at (r) (2.1, 2.8).
T080_WARMUP_MAX = 800
#: Ticks in the ALIGNED and NULL IN VIVO arms (2.2).
T080_N_TICKS = 600
#: Ticks in each LAG arm (2.2).
T080_LAG_TICKS = 300
#: The lags, in stream positions, the audio pack is delayed by (2.2).
T080_LAGS = (1, 2, 5)
#: Ticks in the SCRAMBLED PHASE arm (2.2).
T080_SCRAMBLE_TICKS = 300
#: The per-seed stream offset applied to BOTH pointers: `(T080_STREAM_OFFSET * seed) % 256`.
T080_STREAM_OFFSET = 150
#: Permutation draws in the offline membership-matched null (2.1, 2.8).
T080_N_DRAWS = 2000
#: The campaign's seeds (2.2; PLAN-FINAL Part F: S = 8).
T080_SEEDS = tuple(range(8))
#: (b2)'s agreement band on the offline null's draws (2.3).
T080_BAND = (0.005, 0.995)

#: The one bar (2.3): the paired one-sided t over 8 seeds, 7 d.f., alpha = 0.05.
T080_BAR = T_BAR
#: The pin, written ONCE by `--t080-campaign` (2.9 step 6).
T080_PIN = Path("tools/t080_temporal_binding.json")
#: The dry run's report (2.9 step 5). Never under tools/: the pin is the campaign's.
T080_DRY_RUN_OUT = Path("scratchpad/t080/dry_run.json")

#: `random.Random(T080_ALIGN_RNG_BASE + seed)` draws BOTH the aligned order's per-class cycle
#: and the NULL arm's position permutation (2.1, 2.2).
T080_ALIGN_RNG_BASE = 1000
#: `numpy.random.default_rng(T080_NULL_RNG_BASE + seed)` draws the offline null (2.1).
T080_NULL_RNG_BASE = 7000
#: `numpy.random.default_rng(T080_SCRAMBLE_RNG_BASE + seed)` draws the ONE time-bin
#: permutation the SCRAMBLED PHASE arm applies to every frame (2.2).
T080_SCRAMBLE_RNG_BASE = 5000

#: The arm names, in the order 2.2 fixes AFTER the parity swap of the first two.
T080_ARMS = ("aligned", "null", "lag1", "lag2", "lag5", "scrambled")
#: Ticks per arm (2.2), keyed by arm name.
T080_ARM_TICKS = {
    "aligned": T080_N_TICKS,
    "null": T080_N_TICKS,
    "lag1": T080_LAG_TICKS,
    "lag2": T080_LAG_TICKS,
    "lag5": T080_LAG_TICKS,
    "scrambled": T080_SCRAMBLE_TICKS,
}

#: (p)'s determinism twin: two ALIGNED seed-0 recordings at this width, after identical
#: warm-ups, must be identical field by field (2.3).
T080_DETERMINISM_TICKS = 100
#: 2.8's REPORTED margin sensitivity, re-decomposed OFFLINE on the dry run's ALIGNED arm.
#: The middle value is the binder's own shipped `recog_margin` and is asserted to be so.
T080_MARGINS = (0.10, 0.15, 0.20)
#: 2.3's Monte-Carlo sd of Q95: this many INDEPENDENT `T080_N_DRAWS`-draw nulls of the SAME
#: recording, whose spread IS the sd. BUILDER-INTRODUCED width, disclosed (2.9 step 5 fixes
#: the count at 12).
T080_Q95_REPEATS = 12
#: The seed base those independent nulls are drawn at, so varying this and nothing else
#: varies only the draw.
T080_Q95_SD_BASE = 900000
#: The offset check's width (2.8): the first this many vision labels of seeds 0 and 1 differ.
T080_OFFSET_LABELS = 50

#: The latents are recorded as lists of floats rounded to this many places (2.9 step 5). The
#: decomposition must STILL reproduce the handle at this rounding -- the dry run counts it.
T080_LATENT_PLACES = 6

#: R3's constructed lag table (2.4): rates ABOVE the derived margin at lags 1-3 and at or
#: below it from lag 4, so the window reads 4; and an all-above table, which reads "none".
#: They live here and not only in the suite because `t080_reds()` runs the same red.
T080_R3_Q95 = 0.10
T080_R3_LAG_RATES = {1: 0.31, 2: 0.24, 3: 0.18, 4: 0.09, 5: 0.05}
T080_R3_ALL_ABOVE = {1: 0.31, 2: 0.24, 3: 0.18}
T080_R3_FIRST_BELOW = {1: 0.05, 2: 0.31}
T080_R3_WINDOW = 4

#: R1's constructed recognition table (2.4): ticks, objects, congruent fraction.
T080_R1_TICKS = 200
T080_R1_OBJECTS = 10
T080_R1_CONGRUENT = 0.6
#: R2's constructed binder (2.4): 10 taught objects, 200 probe pairs, a third near a
#: prototype. The latent widths are the LIVE packs': vision 64-d, audio 128-d, in [0, 1].
T080_R2_TAUGHT = 10
T080_R2_PROBES = 200
T080_R2_VIS_DIM = 64
T080_R2_AUD_DIM = 128
T080_R2_NEAR_EVERY = 3
T080_R2_NOISE = 0.02
#: R2's latent DENSITY -- the fraction of each latent's coordinates that are non-zero, the
#: rest exactly 0.0. BUILDER-INTRODUCED and MEASURED, not chosen: 2.4 requires the 200 probe
#: pairs to contain a SUB-MARGIN pair (else `ignore_margin` cannot redden R2), and DENSE
#: latents drawn uniform in [0, 1] cannot produce one. A sign-random projection of two
#: all-positive vectors is dominated by their common mean, so every pair reads far above the
#: margin. Measured on this construction (200 probes, 10 taught, margin 0.15), sub-margin
#: vision / audio pairs and the number of pairs `ignore_margin` changes:
#:   density 1.00 -> min sim 0.469 / 0.492,  0 /  0 sub-margin,  0 changed  (R2 UNREACHABLE)
#:   density 0.25 -> min sim 0.117 / 0.105,  7 / 12 sub-margin,  3 changed
#:   density 0.15 -> min sim 0.059 / 0.074, 26 / 40 sub-margin,  4 changed
#:   density 0.10 -> min sim 0.020 / 0.066, 41 / 58 sub-margin, 11 changed  <- adopted
#:   density 0.06 -> min sim 0.016 / 0.039, 47 / 59 sub-margin,  7 changed
#: Every row read `agree` 0 (the decomposition equals `bind()`'s handle), which is R2's own
#: claim: the density decides whether the red is REACHABLE, never whether it holds.
T080_R2_DENSITY = 0.10
#: The REDS' draw count. BUILDER-INTRODUCED and disclosed (T079's precedent): 2.1 fixes
#: n_draws = 2,000 for the CAMPAIGN's nulls; `t080_reds()` runs on every run_selfchecks pass
#: and 2.4 fixes no draw count for the reds themselves.
T080_REDS_N_DRAWS = 500

#: One ASCII line per red, printed beside its boolean so a gate log says WHAT fired (2.4).
T080_RED_LINES = {
    "R1": "the null is MEMBERSHIP-MATCHED (resampling the audio objects breaks the multiset)",
    "R2": "the decomposition IS the binder (recognition without recog_margin diverges)",
    "R3": "the window reader finds the SMALLEST lag at or below the margin (first-lag reddens it)",
    "R4": "the positive control REFUSES an independent stream (an always-met (g0) reddens it)",
    "R5": "the dissociation reads on a constructed report (within/null swapped reddens it)",
    "R6": "the agreement band is a BAND ([0, 1] reddens it)",
}


# =============================================================================
# 2.9 step 4: THE INSTRUMENTS, GREEN, each red of 2.4 reddened IN FACT by its own mutated
# scorer (`_t080_mutated_scorer`). Nothing here reads a BRAIN yet: every input is either a
# constructed table or a finished recording. The recorder, the dry run and the campaign land
# at 2.9 step 5. `t080_record_brain` is still the step-2 stub.
# =============================================================================
def t080_aligned_order(vis_labels, aud_labels, rng):
    """Per vision position an audio frame index whose label mod 10 equals the vision label,
    cycling within each class in a per-seed shuffled order (2.1).

    The SHD pack's labels run 0-19 (English + German spoken digits) and its digit is
    `label % 10`, so each of the 10 digit classes pools two SHD labels. The per-class lists
    are visited in SORTED class order so the order depends on the pack's CONTENT and not on
    the order numpy happened to hand the labels back in.

    Raises rather than silently mis-pairing when a vision class has no audio frame at all:
    an apparatus that cannot be aligned is a refusal, never a quieter alignment."""
    by_class = {}
    for j, lab in enumerate(aud_labels):
        by_class.setdefault(int(lab) % 10, []).append(j)
    for k in sorted(by_class):
        rng.shuffle(by_class[k])
    ptr = dict.fromkeys(by_class, 0)
    order = []
    for lab in vis_labels:
        k = int(lab) % 10
        lst = by_class.get(k)
        if not lst:
            raise ValueError(f"no audio frame names digit {k}: the two packs cannot be aligned")
        order.append(lst[ptr[k] % len(lst)])
        ptr[k] += 1
    return order


def t080_record_brain(seed, arms=None, n_ticks=None):
    """STUB (2.9 step 2; the recorder lands at 2.9 step 5). A scratch brain under
    `throwaway_home()`, the aligned pack installed at the stimulus, the warm-up to K, then
    the arms of 2.2, rows returned."""
    return None


def _t080_is_scored(row):
    """2.2's SCORED tick, spelled once: awake, with BOTH latents present. Everything barred
    is a fraction of these."""
    return bool(row.get("awake")) and row.get("vis") is not None and row.get("aud") is not None


def t080_recognitions(rows, binder, recognize=None):
    """THE DECOMPOSITION of `bind()`'s handle (2.1), per scored tick, through the binder's
    OWN `_project` and `_recognize` -- nothing of the mechanism is re-implemented here.

      `bound` = `obj_v is not None and obj_v == obj_a`, which is exactly what `bind()` calls
      `congruent` and publishes as `handle`.

    `agree` is the count of scored ticks where the decomposition DISAGREES with the recorded
    handle -- the name is 2.3 (r)'s ("the decomposition's `bound` equals (`handle` is not
    None) on EVERY scored tick"), and the conjunct requires it to be 0. It is stated here
    because the name reads like a count of agreements and is the opposite.

    `recognize` is R2's mutation hook (`ignore_margin`); None means the binder's own."""
    recognize = binder._recognize if recognize is None else recognize
    recs, agree = [], 0
    for r in rows:
        if not _t080_is_scored(r):
            continue
        obj_v, s_v = recognize(binder._project(r["vis"], "vision"), "vision")
        obj_a, s_a = recognize(binder._project(r["aud"], "audio"), "audio")
        bound = obj_v is not None and obj_v == obj_a
        recs.append(
            {
                "t": r.get("t"),
                "obj_v": obj_v,
                "s_v": float(s_v),
                "obj_a": obj_a,
                "s_a": float(s_a),
                "bound": bound,
                "handle": r.get("handle"),
                "digit": r.get("digit"),
                "word": r.get("word"),
            }
        )
        if bound != (r.get("handle") is not None):
            agree += 1
    return {"recs": recs, "agree": agree, "n": len(recs)}


def t080_null(recs, seed, n_draws=T080_N_DRAWS, resample=False):
    """THE MEMBERSHIP-MATCHED NULL AT MATCHED K (2.1), the C5 row's own dissociation input.

    The AUDIO-OBJECT column of the scored ticks is permuted ACROSS the ticks: the same
    multiset of audio features, the same taught K, and only the pairing across windows
    broken. `n_draws` draws are taken as ONE (n_draws x n) integer array -- never a per-draw
    Python loop over ticks, which at 2,000 x 600 would cost more than the recording did.

    Returns the derived margin `q95` (`substrate/_mathutils.q95`, the tree's ONE order
    statistic), the draws' `mean` and `draws_sd`, the agreement band [`q005`, `q995`] of
    2.3 (b2), the product-of-marginals `expected` (sum over objects of p_v(o) p_a(o)), and
    `multiset_ok` -- the membership match ASSERTED on the draws themselves rather than argued
    from the construction.

    `resample` is R1's mutation hook (`null_resamples`): the audio objects drawn i.i.d. WITH
    REPLACEMENT, which keeps the marginal and breaks the multiset."""
    codes = {}

    def code(handle):
        if handle is None:
            return -1
        if handle not in codes:
            codes[handle] = len(codes)
        return codes[handle]

    n = len(recs)
    n_draws = max(0, int(n_draws))
    v = np.array([code(r["obj_v"]) for r in recs], dtype=np.int32)
    a = np.array([code(r["obj_a"]) for r in recs], dtype=np.int32)
    expected = 0.0
    if n:
        for o in set(codes.values()):
            expected += (np.count_nonzero(v == o) / n) * (np.count_nonzero(a == o) / n)
    if n == 0 or n_draws == 0:
        return {
            "mean": 0.0,
            "q95": 0.0,
            "q005": 0.0,
            "q995": 0.0,
            "expected": float(expected),
            "draws_sd": 0.0,
            "multiset_ok": False,
            "n": n,
            "n_draws": n_draws,
        }
    rng = np.random.default_rng(T080_NULL_RNG_BASE + int(seed))
    if resample:
        drawn = rng.choice(a, size=(n_draws, n), replace=True)
    else:
        drawn = rng.permuted(np.tile(a, (n_draws, 1)), axis=1)
    multiset_ok = bool(
        np.array_equal(np.sort(drawn, axis=1), np.broadcast_to(np.sort(a), drawn.shape))
    )
    fractions = ((v[None, :] >= 0) & (v[None, :] == drawn)).mean(axis=1)
    draws = [float(x) for x in fractions]
    q95 = float(_q95(draws))  # sorts `draws` IN PLACE -- the band below reads it sorted
    return {
        "mean": float(sum(draws) / len(draws)),
        "q95": q95,
        "q005": float(_t080_order_stat(draws, T080_BAND[0])),
        "q995": float(_t080_order_stat(draws, T080_BAND[1])),
        "expected": float(expected),
        "draws_sd": float(statistics.stdev(draws)) if len(set(draws)) > 1 else 0.0,
        "multiset_ok": multiset_ok,
        "n": n,
        "n_draws": n_draws,
    }


def _t080_order_stat(sorted_draws, q):
    """The `q`-th order statistic of an ALREADY SORTED sample, by `_mathutils.q95`'s own
    convention (round(q * (len - 1)), never an interpolation), so the band and the margin are
    read the same way."""
    return sorted_draws[min(len(sorted_draws) - 1, int(round(q * (len(sorted_draws) - 1))))]


def t080_window(lag_rates, q95):
    """THE BINDING WINDOW (2.1): the smallest lag whose bound fraction is at or below the
    derived margin, else the string "none" when every lag arm is above it. REPORTED, never
    barred (2.3) -- on this stateless binder the expected reading is 1 tick, and a lag arm
    ABOVE the margin is a finding, printed."""
    for lag in sorted(lag_rates):
        if float(lag_rates[lag]) <= float(q95):
            return int(lag)
    return "none"


def _t080_score(entry, which):
    """(b1)'s two sides, read from one seed's reading. R5's `blind_scorer` mutation is the
    hook that swaps them."""
    return float(entry["within"]) if which == "within" else float(entry["q95_null"])


def _t080_g0(entry):
    """(g0)'s per-seed difference: the CLASSIFIERS' congruence (digit == word mod 10) on the
    ALIGNED arm minus the NULL arm. It is not the binder's report -- that is the point of
    the control. R4's `g0_always_met` is the hook."""
    return float(entry["congruent_aligned"]) - float(entry["congruent_null"])


def _t080_band(entry):
    """(b2)'s per-seed agreement band: the offline null's own [q005, q995]. R6's
    `band_unbounded` is the hook."""
    return float(entry["q005_null"]), float(entry["q995_null"])


def t080_contrasts(per_seed, score=None, band=None, g0=None, bar=T080_BAR):
    """2.3's conjuncts, each decided by `substrate/paired_contrast.contrast` (the ONE
    statistic; nothing is re-spelled here).

      (g0) t over seeds of (congruent fraction ALIGNED - congruent fraction NULL), the
           CLASSIFIERS' agreement. The positive control that is NOT the method: it never
           reads the binder's report. NOT MET -> REFUSED ("the alignment did not reach the
           brain"), never a STOP.
      (b1) t over seeds of (R_within - Q95_null): the ALIGNED arm's bound fraction against
           the DERIVED margin -- the licensed sentence, and the only barred claim.
      (b2) EXACT COUNTS, no t: the NULL IN VIVO arm's bound fraction must lie inside the
           offline null's [q005, q995] on EVERY seed (two instruments required to agree,
           obs #2106). Any seed outside is NAMED, and the whole reading REFUSES.

    `score`, `band` and `g0` are R5's, R6's and R4's mutation hooks; None means the shipped
    reading."""
    score = _t080_score if score is None else score
    band = _t080_band if band is None else band
    g0 = _t080_g0 if g0 is None else g0
    entries = list(per_seed)
    outside = []
    for e in entries:
        lo, hi = band(e)
        x = float(e["null_in_vivo"])
        if not (lo <= x <= hi):
            outside.append(
                f"seed {e.get('seed')}: NULL IN VIVO {x:.4f} outside [{lo:.4f}, {hi:.4f}]"
            )
    n = len(entries)
    return {
        "g0": contrast(
            "(g0) the classifiers' congruence: ALIGNED - NULL",
            [g0(e) for e in entries],
            bar,
        ),
        "b1": contrast(
            "(b1) the binder's within-window rate - the null's q95",
            [score(e, "within") - score(e, "null") for e in entries],
            bar,
        ),
        "b2": {
            "outside": outside,
            "inside": n - len(outside),
            "n_seeds": n,
            "met": bool(n) and not outside,
        },
        "n_seeds": n,
    }


def t080_verdict(got, refusals):
    """2.3's verdict rule, spelled ONCE and decided in 2.3's order.

    REFUSED iff (p), (r), (g0) or (b2) fails -- a finding about the APPARATUS or the
    INSTRUMENT. PASS iff (b1) is MET with all of them holding. NOT MET iff (b1) is NOT MET --
    a finding about this BINDER on this stimulus. The two are never merged."""
    if refusals:
        return "REFUSED"
    if not got["g0"].met or not got["b2"]["met"]:
        return "REFUSED"
    return "PASS" if got["b1"].met else "NOT MET"


# distinct-from: substrate/eprop_transport_gate.py, substrate/gradient_claim_gate.py --
#   census gates that CLASSIFY existing code. Every entry below is a deliberately broken twin
#   of THIS module's own instrument, constructed so a red can be shown to fire rather than
#   argued to (P8 clause 1). Read; nothing here inspects or rewrites any other module.
# distinct-from: cortex/agency_monitor.py::_t079_mutated_scorer and
#   cortex/attention_schema.py::_t078_mutated_scorer -- the PATTERN this follows over
#   DIFFERENT instruments (a permutation floor over an EMA correlation there, a
#   membership-matched permutation over a recognition pair here). One catalog per module is
#   this tree's convention, not a second mechanism. Read.
def _t080_mutated_scorer(kind):
    """2.4's six mutations, RUN and not reasoned. Each is the deliberately broken twin of the
    instrument whose red it must redden, and each is exercised BOTH in
    `tests/test_binding_instrument.py` and in `t080_reds()` below.

      null_resamples   (R1) -- the audio objects drawn i.i.d. WITH REPLACEMENT instead of
          permuted, so a draw's multiset no longer equals the table's: the null stops being
          membership-matched while its marginal is unchanged.
      ignore_margin    (R2) -- the binder's OWN `_recognize` on a shallow twin whose
          `recog_margin` is lowered below every reachable coincidence, so nothing is ever
          refused and a sub-margin pair reads bound.
      window_first_lag (R3) -- the first lag tested is returned whatever its rate, so the
          window is read off the arm ORDER instead of off the margin.
      g0_always_met    (R4) -- the per-seed (g0) difference is a constant +1.0, a positive
          control that cannot refuse an independent stream.
      blind_scorer     (R5) -- (b1)'s two sides swapped, so the reading is compared to itself
          from the wrong direction and the dissociation is lost.
      band_unbounded   (R6) -- (b2)'s band is [0, 1], which no in-vivo fraction can leave."""

    def null_resamples(recs, seed, n_draws=T080_N_DRAWS):
        return t080_null(recs, seed, n_draws, resample=True)

    def ignore_margin(binder):
        twin = copy.copy(binder)  # shares `objects` and `_proj`; only the margin is rebound
        twin.recog_margin = -2.0  # below the centred coincidence's own floor of -1.0
        return twin._recognize

    def window_first_lag(lag_rates, q95):
        return int(min(lag_rates))

    def g0_always_met(entry):
        return 1.0

    def blind_scorer(entry, which):
        return _t080_score(entry, "null" if which == "within" else "within")

    def band_unbounded(entry):
        return 0.0, 1.0

    return {
        "null_resamples": null_resamples,
        "ignore_margin": ignore_margin,
        "window_first_lag": window_first_lag,
        "g0_always_met": g0_always_met,
        "blind_scorer": blind_scorer,
        "band_unbounded": band_unbounded,
    }[kind]


def t080_reds(n_draws=T080_REDS_N_DRAWS):
    """2.4's six reds as ONE self-check line (each is ALSO a test in
    `tests/test_binding_instrument.py`; this is the in-command control 2.4 requires, so a
    gate run that never executes pytest still runs every one on the SAME constructed input).
    Each value is True iff the red holds on its input AND its mutated scorer REDDENS it IN
    FACT -- every `not (...)` clause below is a broken twin having been RUN."""
    out = {}

    # R1 -- the null is MEMBERSHIP-MATCHED. The mutation resamples with replacement.
    recs = t080_constructed_recs(seed=3)
    null = t080_null(recs, 3, n_draws=n_draws)
    m1 = _t080_mutated_scorer("null_resamples")(recs, 3, n_draws)
    out["R1"] = bool(
        null["n"] == T080_R1_TICKS
        and null["n_draws"] == n_draws
        and null["multiset_ok"]
        and null["draws_sd"] > 0.0
        and abs(null["mean"] - null["expected"]) <= 3.0 * null["draws_sd"]
        and null["q005"] <= null["q95"] <= null["q995"]
        and not m1["multiset_ok"]
    )

    # R2 -- the decomposition IS the binder. The mutation drops `recog_margin`.
    binder, protos = t080_constructed_binder(seed=0)
    rows = t080_constructed_pairs(binder, protos, seed=1)
    got2 = t080_recognitions(rows, binder)
    m2 = t080_recognitions(rows, binder, recognize=_t080_mutated_scorer("ignore_margin")(binder))
    bound2 = sum(1 for r in got2["recs"] if r["bound"])
    sub2 = sum(
        1 for r in got2["recs"] if r["s_v"] < binder.recog_margin or r["s_a"] < binder.recog_margin
    )
    out["R2"] = bool(
        got2["n"] == T080_R2_PROBES
        and 0 < bound2 < T080_R2_PROBES
        and sub2 > 0
        and got2["agree"] == 0
        and m2["agree"] > 0
    )

    # R3 -- the window reader. The mutation returns the first lag tested.
    out["R3"] = bool(
        t080_window(T080_R3_LAG_RATES, T080_R3_Q95) == T080_R3_WINDOW
        and t080_window(T080_R3_ALL_ABOVE, T080_R3_Q95) == "none"
        and t080_window(T080_R3_FIRST_BELOW, T080_R3_Q95) == 1
        and _t080_mutated_scorer("window_first_lag")(T080_R3_LAG_RATES, T080_R3_Q95)
        != T080_R3_WINDOW
    )

    # R4 -- the positive control refuses an independent stream. The mutation always meets it.
    indep = t080_constructed_seeds("independent", seed=4)
    c_ind = t080_contrasts(indep)
    c_ali = t080_contrasts(t080_constructed_seeds("aligned", seed=4))
    m4 = t080_contrasts(indep, g0=_t080_mutated_scorer("g0_always_met"))
    out["R4"] = bool(
        c_ind["n_seeds"] == len(T080_SEEDS)
        and not c_ind["g0"].met
        and t080_verdict(c_ind, []) == "REFUSED"
        and c_ali["g0"].met
        and c_ali["g0"].t > T080_BAR
        and m4["g0"].met
    )

    # R5 -- the dissociation on a constructed report. The mutation swaps within and null.
    at_margin = t080_contrasts(t080_constructed_seeds("at_margin", seed=5))
    above = t080_constructed_seeds("above", seed=5)
    c_above = t080_contrasts(above)
    m5 = t080_contrasts(above, score=_t080_mutated_scorer("blind_scorer"))
    out["R5"] = bool(
        at_margin["b1"].degenerate
        and not at_margin["b1"].met
        and t080_verdict(at_margin, []) == "NOT MET"
        and c_above["b1"].met
        and c_above["b1"].t > T080_BAR
        and t080_verdict(c_above, []) == "PASS"
        and not m5["b1"].met
    )

    # R6 -- the agreement band. The mutation widens it to [0, 1].
    outside = t080_constructed_seeds("band_outside", seed=6)
    c_out = t080_contrasts(outside)
    c_in = t080_contrasts(t080_constructed_seeds("band_inside", seed=6))
    m6 = t080_contrasts(outside, band=_t080_mutated_scorer("band_unbounded"))
    out["R6"] = bool(
        c_out["b2"]["n_seeds"] == len(T080_SEEDS)
        and not c_out["b2"]["met"]
        and bool(c_out["b2"]["outside"])
        and t080_verdict(c_out, []) == "REFUSED"
        and c_in["b2"]["met"]
        and c_in["b2"]["inside"] == len(T080_SEEDS)
        and m6["b2"]["met"]
    )
    return out


# =============================================================================
# THE CONSTRUCTED INPUTS every red runs on (2.4). They live HERE and not only in
# tests/test_binding_instrument.py because `t080_reds()` needs them: a gate run that never
# executes pytest still has to run every red on the same input the suite runs it on (T079's
# shape). No brain, no lake, no network: every one is built from `random.Random`.
# =============================================================================
def t080_constructed_recs(n=T080_R1_TICKS, k=T080_R1_OBJECTS, congruent=T080_R1_CONGRUENT, seed=0):
    """R1's recognition table: `n` scored ticks over `k` objects, a `congruent` fraction of
    which carry the SAME object in both modalities. The rest name different objects, so the
    table's audio multiset is not a copy of its vision multiset."""
    import random

    rng = random.Random(seed)
    recs = []
    for t in range(n):
        v = rng.randrange(k)
        if rng.random() < congruent:
            a = v
        else:
            a = (v + 1 + rng.randrange(k - 1)) % k
        recs.append(
            {
                "t": t,
                "obj_v": f"obj{v}",
                "obj_a": f"obj{a}",
                "s_v": 1.0,
                "s_a": 1.0,
                "bound": v == a,
                "handle": f"obj{v}" if v == a else None,
                "digit": v,
                "word": a,
            }
        )
    return recs


def _clip01(x):
    """Keep a noised latent coordinate inside [0, 1] -- the packs' own range (2.4's words)."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _t080_sparse_latent(rng, dim, density=T080_R2_DENSITY):
    """One random latent: `density` of its coordinates drawn uniform in [0, 1], the rest
    exactly 0.0. The sparsity is what makes a SUB-MARGIN pair reachable at all -- see
    `T080_R2_DENSITY` for the measurement that fixed it."""
    v = [0.0] * dim
    for j in rng.sample(range(dim), max(2, int(round(dim * density)))):
        v[j] = rng.random()
    return v


def t080_constructed_binder(seed=0, taught=T080_R2_TAUGHT):
    """R2's binder: a shipped `CrossModalBinder` taught `taught` random latent pairs at the
    LIVE packs' widths (vision 64-d, audio 128-d, values in [0, 1]). Returns (binder,
    prototypes) where prototypes[i] is the (vision, audio) pair that registered `obj<i>`."""
    import random

    from snn_data_lake.cortex.crossmodal_binder import CrossModalBinder

    rng = random.Random(seed)
    binder = CrossModalBinder(seed=0)
    protos = []
    for i in range(taught):
        v = _t080_sparse_latent(rng, T080_R2_VIS_DIM)
        a = _t080_sparse_latent(rng, T080_R2_AUD_DIM)
        binder.register(f"obj{i}", v, a)
        protos.append((v, a))
    return binder, protos


def t080_constructed_pairs(binder, protos, n=T080_R2_PROBES, seed=1):
    """R2's probe pairs, in the RECORDER's own row shape so `t080_recognitions` reads them
    exactly as it reads a live recording. Every `T080_R2_NEAR_EVERY`-th pair is a prototype
    plus small noise (so it is recognised); the rest are uniform (so some fall under the
    margin). `handle` is the SHIPPED `bind()`'s own report on that pair."""
    import random

    rng = random.Random(seed)
    rows = []
    for t in range(n):
        if t % T080_R2_NEAR_EVERY == 0 and protos:
            pv, pa = protos[rng.randrange(len(protos))]
            v = [_clip01(x + rng.uniform(-T080_R2_NOISE, T080_R2_NOISE)) for x in pv]
            a = [_clip01(x + rng.uniform(-T080_R2_NOISE, T080_R2_NOISE)) for x in pa]
        else:
            v = _t080_sparse_latent(rng, T080_R2_VIS_DIM)
            a = _t080_sparse_latent(rng, T080_R2_AUD_DIM)
        rows.append(
            {
                "t": t,
                "kind": "awake",
                "awake": True,
                "scored": True,
                "digit": None,
                "word": None,
                "handle": binder.bind(v, a)["handle"],
                "superadditive": None,
                "vis": v,
                "aud": a,
                "vptr": t,
                "aptr": t,
            }
        )
    return rows


def t080_constructed_seeds(kind, seeds=T080_SEEDS, seed=0):
    """R4's, R5's and R6's per-seed tables, in `t080_contrasts`' own shape. `kind` picks the
    construction, and each is built so the conjunct it bars reads a KNOWN answer:

      "aligned"     -- the classifiers agree far more on ALIGNED than on NULL: (g0) MET.
      "independent" -- the two congruence fractions are sign-balanced noise: (g0) NOT MET.
      "at_margin"   -- within EQUALS the null's q95 on every seed: (b1) NOT MET.
      "above"       -- within EXCEEDS it on every seed: (b1) MET.
      "band_inside" -- the in-vivo null lies inside [q005, q995] on every seed: (b2) holds.
      "band_outside"-- it lies ABOVE q995 on one seed: (b2) REFUSED.
    """
    import random

    rng = random.Random(seed)
    out = []
    for i, s in enumerate(seeds):
        q95, q005, q995 = 0.10, 0.04, 0.16
        entry = {
            "seed": s,
            "congruent_aligned": 0.70 + rng.uniform(-0.02, 0.02),
            "congruent_null": 0.10 + rng.uniform(-0.02, 0.02),
            "within": 0.30 + rng.uniform(-0.02, 0.02),
            "q95_null": q95,
            "q005_null": q005,
            "q995_null": q995,
            "null_in_vivo": 0.09 + rng.uniform(-0.01, 0.01),
        }
        if kind == "independent":
            base = 0.40 + rng.uniform(-0.02, 0.02)
            entry["congruent_aligned"] = base + (0.01 if i % 2 == 0 else -0.01)
            entry["congruent_null"] = base + (0.01 if i % 2 else -0.01)
        elif kind == "at_margin":
            entry["within"] = q95
        elif kind == "above":
            pass
        elif kind == "band_outside":
            entry["null_in_vivo"] = q995 + 0.05 if i == 0 else entry["null_in_vivo"]
        elif kind not in ("aligned", "band_inside"):
            raise ValueError(f"unknown constructed-seed kind {kind!r}")
        out.append(entry)
    return out


# =============================================================================
# The self-check line (2.9 step 2: main() prints the six booleans and EXITS NON-ZERO while
# any is False, so the gate carries the red until 2.9 step 4).
# =============================================================================
def _t080_reds_line(reds, out=print):
    """Print the six booleans beside what each one says."""
    for name in sorted(reds):
        out(f"  {name}: {str(reds[name]):5s}  {T080_RED_LINES[name]}")


def main(argv=None):
    """The module's self-check. Runs the six reds on their CONSTRUCTED inputs: no brain, no
    lake, no network. Returns 0 only when all six fire."""
    argv = list(sys.argv[1:] if argv is None else argv)
    print("=" * 78)
    print("T080 -- the temporal-binding instrument (specs/022 [US4], C5 row temporal_binding)")
    print("=" * 78)
    print(
        f"  K {T080_K} | warmup max {T080_WARMUP_MAX} | arms {list(T080_ARMS)} | "
        f"ticks {T080_ARM_TICKS} | lags {list(T080_LAGS)}"
    )
    print(
        f"  bar {T080_BAR} | seeds {list(T080_SEEDS)} | n_draws {T080_N_DRAWS} "
        f"(reds {T080_REDS_N_DRAWS}) | band {T080_BAND} | offset {T080_STREAM_OFFSET} x seed"
    )
    reds = t080_reds()
    print("-" * 78)
    print("2.4's six reds:")
    _t080_reds_line(reds)
    if not all(reds.values()):
        print(
            f"T080 SELF-CHECK RED: only {sum(reds.values())}/{len(reds)} of 2.4's reds fire "
            f"on their constructed inputs -- {reds}"
        )
        return 1
    print(
        f"  T080 OK: R1-R6 fire on their constructed inputs and each is reddened IN FACT by "
        f"its mutated scorer ({sum(reds.values())}/{len(reds)}); no brain, no lake, no network"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
