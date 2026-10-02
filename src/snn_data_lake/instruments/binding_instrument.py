"""Temporal-binding instrument (specs/022 T080 [US4]) -- the LIVE cross-modal binder's report of
ONE object, read against its own features RE-PAIRED across windows at matched K.

THE ROW. tasks.md T080 [US4]; the C5 row `temporal_binding` in
`gates/conclusion_protocol.py`; FR-028's binding item; register #6's binding line;
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
(`engine.brain.make_brain(seed, persist=False)`; the daemon's brain and the durable lake are
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

# distinct-from: faculties/gwt_deep.py -- a SYNTHETIC K-object phase-multiplexing benchmark
#   with its own spiking-vs-rate control, scoring a pairing this tree's live tick never
#   performs. This module runs no synthetic binding task at all: it reads the LIVE binder's
#   own report off a recorded brain and scores it against a permutation of that recording.
#   gwt_deep is not a control here and is not run. Read.
# distinct-from: faculties/crossmodal_binder.py -- the MECHANISM UNDER TEST. This module changes
#   nothing in it: it calls its public `bind` and its own `_project` / `_recognize` and
#   re-implements neither. The margin (0.15) and the cue strength (0.055) are READ off the
#   shipped binder, never re-declared. Read.
# distinct-from: instruments/ignition_instrument.py -- an EXEMPT scorer over scratch
#   ReentrantWorkspace subjects, scoring an all-or-none DRIVE transition. Different subject,
#   different quantity; the EXEMPT-instrument shape is the thing reused. Read.
# distinct-from: faculties/agency_monitor.py's T079 functions -- the same campaign/dry-run/pin
#   PATTERN over a different instrument, different arms and a different pin. No state, no pin
#   and no statistic is shared beyond `instruments/paired_contrast`, which both IMPORT. Read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.instruments.binding_instrument
"""

import copy
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from snn_data_lake.infra._mathutils import q95 as _q95
from snn_data_lake.instruments.paired_contrast import T_BAR, contrast

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
#: (b2)'s REFUSAL band (2.11, after run 1's numbers, disclosed): the FAMILY-WISE 99 % band over the
#: S seeds, [q(0.005 / S), q(1 - 0.005 / S)] of the same draws -- DERIVED from the seed count, never a
#: literal. Eight per-seed 99 % checks refuse an honest apparatus 1 - 0.99^8 = 7.7 % of the time
#: (6.2 % simulated from run 1's own position nulls); the per-seed band above stays REPORTED.
T080_BAND_FW = (0.005 / len(T080_SEEDS), 1.0 - 0.005 / len(T080_SEEDS))

#: The one bar (2.3): the paired one-sided t over 8 seeds, 7 d.f., alpha = 0.05.
T080_BAR = T_BAR
#: The tree root, so the pin resolves from any working directory (ignition_instrument's idiom).
REPO = Path(__file__).resolve().parents[3]
#: The pin, written ONCE by `--t080-campaign` (2.9 step 6); READ by the self-check below, which
#: re-derives the C5 row's reading from it on every gate run (the T080 panel fold, 2026-09-16).
T080_PIN = REPO / "tools" / "t080_temporal_binding.json"
#: The pin's COMPANION (2.10 (g), T078's npz precedent): every scored tick's raw latents per
#: seed and arm at float32, sha256 recorded in the pin, so (r)'s decomposition re-derives
#: from the files alone. Written once, beside the pin.
T080_LATENTS = Path("tools/t080_latents.npz")
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


def t080_arm_order(seed):
    """2.2's arm order, ALTERNATING by seed parity -- even seeds ALIGNED first, odd seeds
    NULL first -- then LAG 1, 2, 5 and SCRAMBLED PHASE. The brain's other state drifts across
    arms; the binder does not (it is stateless between calls once taught), so the alternation
    is what keeps that drift from being confounded with the arm."""
    head = ("aligned", "null") if int(seed) % 2 == 0 else ("null", "aligned")
    return list(head) + ["lag1", "lag2", "lag5", "scrambled"]


def _t080_record_arm(b, n_ticks):
    """One arm: `n_ticks` ticks of the live brain, one row each, 2.9 step 5's field list
    exactly. The latents are lists of floats rounded to `T080_LATENT_PLACES` -- the
    decomposition must STILL reproduce the handle at that rounding, which the dry run
    counts rather than assumes."""
    rows = []
    for t in range(n_ticks):
        kind = b.tick()
        vis, aud = b._real_percept, b._audio_percept
        rows.append(
            {
                "t": t,
                "kind": kind,
                "awake": kind == "awake",
                "digit": None if b._perceived_digit is None else int(b._perceived_digit),
                "word": None if b._perceived_word is None else int(b._perceived_word),
                "handle": b._xmodal_handle,
                "superadditive": bool(getattr(b, "_superadditive", False)),
                "vis": None if vis is None else [round(float(x), T080_LATENT_PLACES) for x in vis],
                "aud": None if aud is None else [round(float(x), T080_LATENT_PLACES) for x in aud],
                "vptr": int(b._real_ptr),
                "aptr": int(b._audio_ptr),
            }
        )
    return rows


def t080_record_brain(seed, arms=None, n_ticks=None):
    """THE RECORDER (2.1, 2.2). A SCRATCH brain under `throwaway_home()` -- a fresh NORD_HOME
    for the duration, so nothing this recorder does can reach the live lake -- built with
    `make_brain(seed, persist=False)`.

    THE STIMULUS MANIPULATION, and it is the only one: the shipped SHD pack is re-ordered so
    that position i carries a word naming the digit of vision position i
    (`t080_aligned_order`), `_audio_frames` / `_audio_labels` are replaced, and BOTH pointers
    are set to `(T080_STREAM_OFFSET * seed) % 256`. Nothing inside the binder or the brain is
    touched.

    EVERY ARM'S PACK IS BUILT BEFORE ANY ARM RUNS, so the parity order of 2.2 cannot change
    which frames an arm sees:
      ALIGNED   the aligned pack as installed.
      NULL      the aligned pack with its 256 positions permuted under
                `random.Random(T080_ALIGN_RNG_BASE + seed)` -- the same frames, K already
                taught, the pairing broken at the stimulus.
      LAG d     the aligned pack with `_audio_ptr` set d positions behind `_real_ptr` at the
                arm's start.
      SCRAMBLED the aligned pack with every frame's 100 time bins permuted by ONE fixed
                per-seed permutation from `default_rng(T080_SCRAMBLE_RNG_BASE + seed)`.

    The warm-up ticks until `len(_xmodal_known) == T080_K` or `T080_WARMUP_MAX`; the taught
    set is then FROZEN by the brain's own one-shot rule, so K is matched across every arm by
    construction. The brain's `_xmodal` is returned under "binder" and is the ONLY thing that
    outlives the brain -- `t080_seed` drops it once the seed's analysis is done.

    `arms` and `n_ticks` are the twin's hooks ((p) records one ALIGNED arm at 100 ticks)."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    seed = int(seed)
    order_arms = t080_arm_order(seed) if arms is None else list(arms)
    out = {"seed": seed, "arm_order": order_arms, "arms": {}, "timings": {}}
    t0 = time.time()
    with throwaway_home():
        b = make_brain(seed=seed, persist=False)
        out["timings"]["build_s"] = round(time.time() - t0, 3)
        vis_labels = [int(x) for x in b._real_labels]
        aud = b._audio_frames
        aud_labels = [int(x) for x in b._audio_labels]
        order = t080_aligned_order(
            vis_labels, aud_labels, random.Random(T080_ALIGN_RNG_BASE + seed)
        )
        n_pos = len(order)
        aligned_frames = aud[order]
        aligned_labels = np.array([aud_labels[j] for j in order])
        nrng = random.Random(T080_ALIGN_RNG_BASE + seed)
        perm = list(range(n_pos))
        nrng.shuffle(perm)
        null_frames = aud[[order[p] for p in perm]]
        null_labels = np.array([aud_labels[order[p]] for p in perm])
        bins = np.random.default_rng(T080_SCRAMBLE_RNG_BASE + seed).permutation(
            aligned_frames.shape[1]
        )
        packs = {
            "aligned": (aligned_frames, aligned_labels),
            "null": (null_frames, null_labels),
            "lag1": (aligned_frames, aligned_labels),
            "lag2": (aligned_frames, aligned_labels),
            "lag5": (aligned_frames, aligned_labels),
            "scrambled": (aligned_frames[:, bins, :], aligned_labels),
        }
        b._audio_frames, b._audio_labels = aligned_frames, aligned_labels
        b._real_ptr = b._audio_ptr = (T080_STREAM_OFFSET * seed) % n_pos
        out["offset"] = int(b._real_ptr)
        out["positions"] = n_pos

        tw = time.time()
        warm = 0
        while len(b._xmodal_known) < T080_K and warm < T080_WARMUP_MAX:
            b.tick()
            warm += 1
        out["ticks_to_k"] = warm
        out["k"] = len(b._xmodal_known)
        out["taught"] = sorted(b._xmodal_known)
        out["perception_acc"] = _t080_float_or_none(getattr(b, "_perception_acc", None))
        out["audio_acc"] = _t080_float_or_none(getattr(b, "_audio_acc", None))
        out["timings"]["warmup_s"] = round(time.time() - tw, 3)

        out["arm_acc"] = {}
        for name in order_arms:
            frames, labels = packs[name]
            b._audio_frames, b._audio_labels = frames, labels
            if name.startswith("lag"):
                b._audio_ptr = (int(b._real_ptr) - int(name[3:])) % n_pos
            ta = time.time()
            before = _t080_acc_counters(b)
            n = T080_ARM_TICKS[name] if n_ticks is None else int(n_ticks)
            out["arms"][name] = _t080_record_arm(b, n)
            out["arm_acc"][name] = _t080_arm_accuracy(before, _t080_acc_counters(b))
            out["timings"][f"{name}_s"] = round(time.time() - ta, 3)

        out["taught_end"] = sorted(b._xmodal_known)
        out["audio_acc_end"] = _t080_float_or_none(getattr(b, "_audio_acc", None))
        out["perception_acc_end"] = _t080_float_or_none(getattr(b, "_perception_acc", None))
        out["binder"] = b._xmodal
    out["timings"]["total_s"] = round(time.time() - t0, 3)
    return out


def _t080_float_or_none(x):
    """A brain accuracy that has not been computed yet reads None, and None is REPORTED as
    None -- never silently folded to 0.0."""
    return None if x is None else float(x)


# distinct-from: instruments/prior_drift_harness.py::above_chance -- reads an arm's accuracy
#   against the 1/K chance floor with a paired t and a bar. The two below compute NO
#   statistic, take NO bar and decide nothing: they difference the brain's own cumulative
#   hit/seen counters so a per-ARM accuracy exists to report at all. Read.
# distinct-from: faculties/agency_monitor.py::_t079_arm_extras -- the per-arm REPORTED
#   quantities of a DIFFERENT instrument (an action histogram and two Pearsons over joint
#   channels). Same word "arm", different recording, different quantity. Read.
# distinct-from: gates/amdahl_gate.py::red_sham_neuter and
#   gates/immortal_retention.py::unlisted_reclaimers -- a sham-path timing control and a
#   reclaimer census. Term overlap ("arm", "reads") only; neither touches a classifier. Read.
def _t080_acc_counters(b):
    """The brain's OWN classifier hit/seen counters. `_perception_acc` and `_audio_acc` are
    CUMULATIVE over the brain's whole life (`hits / seen`, `engine/brain.py::_perceive_audio`),
    so an arm's accuracy is a DIFFERENCE of these counters, never the published ratio --
    reading the ratio at an arm's end would report the warm-up and every earlier arm too."""
    return {
        "vis_hits": int(getattr(b, "_perc_hits", 0) or 0),
        "vis_seen": int(getattr(b, "_perc_seen", 0) or 0),
        "aud_hits": int(getattr(b, "_audio_hits", 0) or 0),
        "aud_seen": int(getattr(b, "_audio_seen", 0) or 0),
    }


def _t080_arm_accuracy(before, after):
    """2.3's REPORTED per-arm classifier accuracy. The SCRAMBLED arm's audio accuracy beside
    ALIGNED's is what says whether the audio feature survives the loss of its spike phase;
    ALIGNED's is the yield's ceiling."""
    dv_s = after["vis_seen"] - before["vis_seen"]
    da_s = after["aud_seen"] - before["aud_seen"]
    return {
        "vision_seen": dv_s,
        "vision_acc": ((after["vis_hits"] - before["vis_hits"]) / dv_s) if dv_s else None,
        "audio_seen": da_s,
        "audio_acc": ((after["aud_hits"] - before["aud_hits"]) / da_s) if da_s else None,
    }


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
            "q_lo": 0.0,
            "q_hi": 0.0,
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
        # the family-wise ends (2.11): the refusal reads THESE; q005/q995 stay reported
        "q_lo": float(_t080_order_stat(draws, T080_BAND_FW[0])),
        "q_hi": float(_t080_order_stat(draws, T080_BAND_FW[1])),
        "expected": float(expected),
        "draws_sd": float(statistics.stdev(draws)) if len(set(draws)) > 1 else 0.0,
        "multiset_ok": multiset_ok,
        "n": n,
        "n_draws": n_draws,
    }


# distinct-from: infra/_mathutils.py::q95 -- the same round(q * (len - 1)) convention at a
#   PARAMETERISED q over an ALREADY-SORTED sample (q95 sorts and is pinned at 0.95). Four bodies of
#   this convention are live in src/ (T080 panel, 2026-09-16: _mathutils, here, metacognitive_monitor,
#   lake_engine); the lift to one `order_stat(sorted_draws, q)` is deferred to T081, the second
#   consumer of a band -- recorded in the T080 record's 3.5 and the route ledger, not silently.
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
    """(b2)'s agreement band: the offline null's own FAMILY-WISE 99 % band [q_lo, q_hi] over
    the S seeds (2.11; run 1's per-seed [q005, q995] refused an honest apparatus on seed 3 at a
    6-8 % family-wise rate). R6's `band_unbounded` is the hook."""
    return float(entry["q_lo_null"]), float(entry["q_hi_null"])


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
           offline null's band on EVERY seed (two instruments required to agree, obs #2106)
           -- AS AMENDED BY 2.11 (after run 1's numbers, disclosed) the FAMILY-WISE band
           `T080_BAND_FW` derived from `len(T080_SEEDS)`, read via `_t080_band`; 2.3's
           per-seed [q005, q995] stays REPORTED, never barred. Any seed outside is NAMED,
           and the whole reading REFUSES.

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


# distinct-from: gates/eprop_transport_gate.py, gates/gradient_claim_gate.py --
#   census gates that CLASSIFY existing code. Every entry below is a deliberately broken twin
#   of THIS module's own instrument, constructed so a red can be shown to fire rather than
#   argued to (P8 clause 1). Read; nothing here inspects or rewrites any other module.
# distinct-from: faculties/agency_monitor.py::_t079_mutated_scorer and
#   instruments/attention_schema.py::_t078_mutated_scorer -- the PATTERN this follows over
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

    from snn_data_lake.faculties.crossmodal_binder import CrossModalBinder

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
      "band_inside" -- the in-vivo null lies inside the family-wise band [q_lo, q_hi] on every
                       seed: (b2) holds.
      "band_outside"-- it lies ABOVE q_hi on one seed: (b2) REFUSED.
    """
    import random

    rng = random.Random(seed)
    out = []
    for i, s in enumerate(seeds):
        q95, q005, q995 = 0.10, 0.04, 0.16
        q_lo, q_hi = q005 - 0.01, q995 + 0.01  # the family-wise band is wider than the per-seed one
        entry = {
            "seed": s,
            "congruent_aligned": 0.70 + rng.uniform(-0.02, 0.02),
            "congruent_null": 0.10 + rng.uniform(-0.02, 0.02),
            "within": 0.30 + rng.uniform(-0.02, 0.02),
            "q95_null": q95,
            "q005_null": q005,
            "q995_null": q995,
            "q_lo_null": q_lo,
            "q_hi_null": q_hi,
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
# 2.9 step 5: THE PER-SEED READING, the determinism precondition, the apparatus DRY RUN and
# the campaign. Everything below touches a brain; everything above does not.
# =============================================================================
def _t080_arm_counts(rows):
    """One arm's counts (2.2's "every recording also carries..."), all of them over the
    SCORED ticks: `n`, `bound`, `false` (a handle that does not name the perceived digit),
    `congruent` (the CLASSIFIERS' agreement, digit == word mod 10) and the `yield` (bound
    over congruent -- the audio classifier's accuracy is its ceiling).

    `awake_missing` is REPORTED beside them: 2.3 (r)'s second clause, "on every scored awake
    tick both latents are present", is true BY THE DEFINITION of scored, so the quantity that
    can actually say something about the apparatus is how many awake ticks carried no pair.

    `superadditive` is NOT one of 2.2's quantities and is not a second observation: bind()
    sets the SC cell's receptive-field offset to 0 iff `congruent`, so at the declared
    cue_strength 0.055 the enhancement index is the constant 1.769 on a congruent pair and
    exactly 1.0 otherwise (measured per call, 400/400 at the T080 panel fold, 2026-09-16) --
    the field records the SC CELL'S REGIME, never a per-tick fact about the pair, and enters
    no conjunct."""
    scored = [r for r in rows if _t080_is_scored(r)]
    awake = sum(1 for r in rows if r["awake"])
    handled = [r for r in scored if r["handle"] is not None]
    false_ = sum(1 for r in handled if r["digit"] is None or r["handle"] != f"obj{int(r['digit'])}")
    congruent = [
        r
        for r in scored
        if r["digit"] is not None and r["word"] is not None and r["digit"] == r["word"] % 10
    ]
    n = len(scored)
    return {
        "ticks": len(rows),
        "awake": awake,
        "n": n,
        "awake_missing": awake - n,
        "bound": len(handled),
        "bound_frac": (len(handled) / n) if n else 0.0,
        "false": false_,
        "false_frac": (false_ / len(handled)) if handled else 0.0,
        "congruent": len(congruent),
        "congruent_frac": (len(congruent) / n) if n else 0.0,
        "yield": (len(handled) / len(congruent)) if congruent else 0.0,
        "superadditive": sum(1 for r in scored if r["superadditive"]),
    }


def _t080_refusals(reading):
    """2.3 (r)'s named refusals for ONE seed: WHICH clause, WHICH arm, WHICH count.

    (r) has three clauses. K taught within the cap, and the decomposition equal to the
    recorded handle on EVERY scored tick of EVERY arm, are checked here. The middle clause --
    both latents present on every SCORED tick -- is definitionally true (2.2 defines a scored
    tick as an awake tick with both latents), so `awake_missing` is REPORTED per arm instead
    of being dressed up as a refusal the pre-registration did not license."""
    out = []
    if reading["k"] != T080_K:
        out.append(
            f"seed {reading['seed']}: taught {reading['k']} of K={T080_K} objects in "
            f"{reading['ticks_to_k']} ticks (cap {T080_WARMUP_MAX})"
        )
    for name in sorted(reading["arms"]):
        arm = reading["arms"][name]
        if arm["n"] == 0:
            out.append(f"seed {reading['seed']} {name}: NO scored tick in {arm['ticks']}")
        if arm["agree"]:
            out.append(
                f"seed {reading['seed']} {name}: the decomposition disagrees with the "
                f"recorded handle on {arm['agree']} of {arm['n']} scored ticks"
            )
    return out


def t080_seed(seed, arms=None, n_ticks=None, n_draws=T080_N_DRAWS):
    """ONE seed's reading: record, decompose every arm through the brain's OWN binder, null
    the ALIGNED arm offline, read the window off the lag arms, and drop the binder.

    Returns the per-seed entry `t080_contrasts` consumes (`within`, `q95_null`, `q005_null`,
    `q995_null`, `null_in_vivo`, `congruent_aligned`, `congruent_null`) with everything 2.3
    REPORTS beside it."""
    rec = t080_record_brain(seed, arms=arms, n_ticks=n_ticks)
    binder = rec.pop("binder")
    out = {k: rec[k] for k in ("seed", "arm_order", "timings", "offset", "positions")}
    out.update({k: rec[k] for k in ("ticks_to_k", "k", "taught", "taught_end")})
    out["perception_acc"] = rec["perception_acc_end"]
    out["audio_acc"] = rec["audio_acc_end"]
    out["arms"] = {}
    t0 = time.time()
    for name, rows in rec["arms"].items():
        counts = _t080_arm_counts(rows)
        recs = t080_recognitions(rows, binder)
        counts["agree"] = recs["agree"]
        counts["recs"] = recs["recs"]
        out["arms"][name] = counts
        counts["acc"] = rec["arm_acc"][name]
    # the raw latents of every scored tick, in the SAME order as each arm's `recs` (the
    # campaign packs them into the companion npz; nothing barred reads them here)
    out["latents"] = {
        name: {
            "vis": np.asarray([r["vis"] for r in rows if _t080_is_scored(r)], dtype=np.float32),
            "aud": np.asarray([r["aud"] for r in rows if _t080_is_scored(r)], dtype=np.float32),
        }
        for name, rows in rec["arms"].items()
    }
    out["timings"]["decompose_s"] = round(time.time() - t0, 3)

    aligned = out["arms"].get("aligned")
    null = t080_null(aligned["recs"], seed, n_draws=n_draws) if aligned else t080_null([], seed, 0)
    out["null"] = null
    out["within"] = aligned["bound_frac"] if aligned else 0.0
    out["q95_null"] = null["q95"]
    out["q005_null"] = null["q005"]
    out["q995_null"] = null["q995"]
    out["q_lo_null"] = null["q_lo"]
    out["q_hi_null"] = null["q_hi"]
    out["null_in_vivo"] = out["arms"]["null"]["bound_frac"] if "null" in out["arms"] else 0.0
    out["congruent_aligned"] = aligned["congruent_frac"] if aligned else 0.0
    out["congruent_null"] = out["arms"]["null"]["congruent_frac"] if "null" in out["arms"] else 0.0
    out["lag_rates"] = {
        d: out["arms"][f"lag{d}"]["bound_frac"] for d in T080_LAGS if f"lag{d}" in out["arms"]
    }
    out["window"] = t080_window(out["lag_rates"], null["q95"]) if out["lag_rates"] else "none"
    out["refusals"] = _t080_refusals(out)
    # THE TAUGHT PROTOTYPES (2.10 (g)): the registering latents of every object the warm-up
    # taught, so the decomposition of the companion's latents re-derives OFFLINE through a
    # fresh binder -- register these, project the npz rows, recognise -- with no brain at all.
    # IN REGISTRATION ORDER, never sorted: `cleanup` breaks an exact coincidence tie by the
    # codebook's insertion order, so an offline binder must register the objects in the order
    # the warm-up did or it names the other prototype on tied ticks (measured on run 1's pin:
    # 9 of 294 ALIGNED ticks on seed 7 tied at 440/512 between two prototypes).
    out["prototypes"] = {
        h: {"vis": list(o["vision_latent"]), "aud": list(o["audio_latent"])}
        for h, o in binder.objects.items()
    }
    del binder
    return out


def t080_determinism(seed=0, n_ticks=T080_DETERMINISM_TICKS):
    """(p): two ALIGNED recordings of the same seed, at the same width, after identical
    warm-ups, compared FIELD BY FIELD -- not a sample of fields and not a tolerance. A
    nondeterminism anywhere in the recorder, the stimulus install or the brain is a REFUSAL
    before any number is read, and the first differing (row, field) is named."""
    a = t080_record_brain(seed, arms=("aligned",), n_ticks=n_ticks)
    b = t080_record_brain(seed, arms=("aligned",), n_ticks=n_ticks)
    a.pop("binder", None)
    b.pop("binder", None)
    ra, rb = a["arms"]["aligned"], b["arms"]["aligned"]
    diff = None
    if len(ra) != len(rb):
        diff = f"row count {len(ra)} vs {len(rb)}"
    for i, (x, y) in enumerate(zip(ra, rb)):
        if diff:
            break
        if set(x) != set(y):
            diff = f"row {i}: field sets differ ({sorted(set(x) ^ set(y))})"
            break
        for k in sorted(x):
            if x[k] != y[k]:
                diff = f"row {i} field {k}: {x[k]!r} vs {y[k]!r}"
                break
    warm_same = a["ticks_to_k"] == b["ticks_to_k"] and a["taught"] == b["taught"]
    return {
        "identical": diff is None and warm_same,
        "first_diff": diff,
        "rows": len(ra),
        "twin_rows": len(rb),
        "ticks_to_k": [a["ticks_to_k"], b["ticks_to_k"]],
        "taught_equal": a["taught"] == b["taught"],
        "seed": int(seed),
    }


def t080_stream_labels(seed, n_labels=T080_OFFSET_LABELS, max_ticks=260):
    """The first `n_labels` REAL frame labels seed `s` walks with 2.2's offset applied. The
    offset check is an APPARATUS precondition, so it reads the labels the brain actually
    consumed rather than re-deriving them from the pack by arithmetic."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    out = []
    with throwaway_home():
        b = make_brain(seed=int(seed), persist=False)
        labels = b._real_labels
        b._real_ptr = b._audio_ptr = (T080_STREAM_OFFSET * int(seed)) % len(labels)
        prev = int(b._real_ptr)
        for _ in range(max_ticks):
            if len(out) >= n_labels:
                break
            b.tick()
            ptr = int(b._real_ptr)
            if ptr != prev:
                out.append(int(labels[(ptr - 1) % len(labels)]))
            prev = ptr
    return out


def t080_margin_sensitivity(rows, binder, margins=T080_MARGINS):
    """2.8's REPORTED sensitivity of the binding rate to `recog_margin`, computed OFFLINE on
    ONE decomposition: the margin-free twin gives (nearest prototype, coincidence) per
    modality per tick, and each margin is then a re-threshold of those same numbers -- so the
    sweep costs one pass, not three, and the three readings share their projections.

    `rows` are the RECORDER's rows, never a finished decomposition: the sweep re-projects the
    latents, and a decomposition has none.

    ITS OWN POSITIVE CONTROL: at the binder's SHIPPED margin the re-threshold must reproduce
    the default decomposition's bound count exactly, ON A NON-EMPTY recording. `n > 0` is part
    of `control_ok` because the first build of this sweep was handed a decomposition instead
    of rows, read 0 scored ticks, and its control passed on 0 == 0 -- a zero is a claim about
    the instrument before it is a claim about the brain."""
    free = t080_recognitions(rows, binder, recognize=_t080_mutated_scorer("ignore_margin")(binder))
    shipped = t080_recognitions(rows, binder)
    shipped_bound = sum(1 for r in shipped["recs"] if r["bound"])
    sv = sorted(r["s_v"] for r in free["recs"])
    sa = sorted(r["s_a"] for r in free["recs"])
    out = {
        "n": free["n"],
        "shipped_margin": float(binder.recog_margin),
        "margins": {},
        # WHY the sweep can be flat, in its own numbers rather than in a sentence: if the
        # WORST coincidence on the recording already clears the widest margin tested, then
        # `recog_margin` refuses nothing in vivo and the handle is decided entirely by
        # whether the two nearest-prototype winners agree.
        "coincidence": {
            "vision": _t080_spread(sv),
            "audio": _t080_spread(sa),
        },
        "top_agree": sum(
            1 for r in free["recs"] if r["obj_v"] is not None and r["obj_v"] == r["obj_a"]
        ),
    }
    for m in margins:
        bound = sum(
            1
            for r in free["recs"]
            if r["s_v"] >= m
            and r["s_a"] >= m
            and r["obj_v"] is not None
            and r["obj_v"] == r["obj_a"]
        )
        out["margins"][f"{m:g}"] = {
            "bound": bound,
            "frac": (bound / free["n"]) if free["n"] else 0.0,
            "sub_vision": sum(1 for x in sv if x < m),
            "sub_audio": sum(1 for x in sa if x < m),
        }
    key = f"{float(binder.recog_margin):g}"
    out["shipped_bound"] = shipped_bound
    out["control_ok"] = bool(
        free["n"] > 0 and key in out["margins"] and out["margins"][key]["bound"] == shipped_bound
    )
    return out


def _t080_spread(sorted_values):
    """min / median / max of an already-sorted sample, or Nones when it is empty. Reported
    beside the margin sweep so a FLAT sweep is explained by the data rather than argued."""
    if not sorted_values:
        return {"min": None, "median": None, "max": None, "n": 0}
    return {
        "min": sorted_values[0],
        "median": statistics.median(sorted_values),
        "max": sorted_values[-1],
        "n": len(sorted_values),
    }


def t080_q95_sd(recs, repeats=T080_Q95_REPEATS, n_draws=T080_N_DRAWS, base=T080_Q95_SD_BASE):
    """2.3's Monte-Carlo sd of Q95_null: `repeats` INDEPENDENT `n_draws`-draw nulls of the
    SAME recording, whose spread IS the margin's sampling noise. COMPUTED, not promised."""
    vals = [float(t080_null(recs, base + i, n_draws=n_draws)["q95"]) for i in range(repeats)]
    return {
        "q95": vals,
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(set(vals)) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "repeats": repeats,
        "n_draws": n_draws,
    }


def _t080_head():
    """The tree's HEAD, for the pin. Unavailable (no git, a tarball) is recorded as such."""
    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else f"unavailable (rc {r.returncode})"
    except Exception as e:  # noqa: BLE001 -- a pin field, never a control-flow decision
        return f"unavailable ({type(e).__name__})"


# distinct-from: instruments/attention_schema.py -- its `g4_rule` is a BRANCH RULE over per-seed
#   decodes; this computes nothing at all. Read.
# distinct-from: faculties/metacognitive_monitor.py -- `_experiment` RUNS a three-condition
#   comparison; this formats one already-computed arm. Read.
# distinct-from: faculties/autotelic_goals.py -- `_random_reach_rate` is a CHANCE BASELINE under
#   a random policy; no baseline, no policy and no rng here. Read.
# distinct-from: sensorimotor/efferent_boundary.py -- `intended_deltas` PARSES an action token
#   into per-joint degrees; this parses nothing. Read.
# distinct-from: faculties/agency_monitor.py -- its `_t079_arm_extras` computes an action
#   histogram and two Pearsons for a DIFFERENT instrument's arms. This is a printf, and it
#   exists so the dry run and the campaign cannot drift into two report formats. Read.
def _t080_arm_line(name, arm):
    """One ASCII line per arm, the same shape in the dry run and the campaign."""
    acc = arm.get("acc") or {}
    va, aa = acc.get("vision_acc"), acc.get("audio_acc")
    return (
        f"  {name:<10} ticks {arm['ticks']:<5} awake {arm['awake']:<5} scored {arm['n']:<5} "
        f"(missing {arm['awake_missing']:<4}) bound {arm['bound']:<5} = {arm['bound_frac']:.4f} "
        f"| false {arm['false']:<4} ({arm['false_frac']:.3f}) congruent {arm['congruent']:<5} "
        f"({arm['congruent_frac']:.4f}) yield {arm['yield']:.4f} super {arm['superadditive']:<5} "
        f"agree_fail {arm['agree']} | vision acc "
        f"{'n/a' if va is None else format(va, '.4f')} audio acc "
        f"{'n/a' if aa is None else format(aa, '.4f')} (audio seen {acc.get('audio_seen')})"
    )


def _t080_seed_block(r, out=print):
    """Everything 2.3 REPORTS for one seed, printed."""
    out(
        f"seed {r['seed']}: K {r['k']}/{T080_K} in {r['ticks_to_k']} ticks "
        f"(offset {r['offset']} of {r['positions']}); taught {r['taught']}; "
        f"vision acc {r['perception_acc']} audio acc {r['audio_acc']}; "
        f"arm order {r['arm_order']}"
    )
    for name in r["arm_order"]:
        if name in r["arms"]:
            out(_t080_arm_line(name, r["arms"][name]))
    n = r["null"]
    out(
        f"  offline membership-matched null on ALIGNED ({n['n']} ticks x {n['n_draws']} draws): "
        f"mean {n['mean']:.4f} q95 {n['q95']:.4f} band [{n['q005']:.4f}, {n['q995']:.4f}] "
        f"family-wise [{n['q_lo']:.4f}, {n['q_hi']:.4f}] "
        f"sd {n['draws_sd']:.4f} expected(product of marginals) {n['expected']:.4f} "
        f"multiset_ok {n['multiset_ok']}"
    )
    out(
        f"  within {r['within']:.4f} - q95 {r['q95_null']:.4f} = {r['within'] - r['q95_null']:+.4f}"
        f" | NULL IN VIVO {r['null_in_vivo']:.4f} in per-seed band "
        f"{r['q005_null'] <= r['null_in_vivo'] <= r['q995_null']} in family-wise band "
        f"{r['q_lo_null'] <= r['null_in_vivo'] <= r['q_hi_null']}"
        f" | congruent ALIGNED {r['congruent_aligned']:.4f} - NULL {r['congruent_null']:.4f} = "
        f"{r['congruent_aligned'] - r['congruent_null']:+.4f}"
    )
    out(
        f"  lag rates { {d: round(v, 4) for d, v in r['lag_rates'].items()} } vs margin "
        f"{r['q95_null']:.4f} -> window {r['window']}"
    )
    for why in r["refusals"]:
        out(f"  (r) REFUSAL: {why}")


def run_t080_dry_run(out_path=T080_DRY_RUN_OUT, seed=0):
    """2.9 step 5's APPARATUS dry run: ONE seed at the CAMPAIGN's widths, and NOT ONE barred
    conjunct -- one seed cannot carry a paired t, so every (g0)/(b1) number here is printed
    as a single seed's difference with "n/a" where the t would go.

    What it DOES read, all of it reported in 3.0 before the campaign:
      * the STREAM-OFFSET precondition: seeds 0 and 1 must walk different frames;
      * (p)'s determinism twin: two 100-tick ALIGNED recordings after identical warm-ups,
        compared field by field;
      * the six arms at their campaign widths, with the false-binding count, the congruence,
        the yield and the scrambled-phase arm's audio accuracy beside ALIGNED's;
      * the DECOMPOSITION's agreement with the recorded handle per arm -- the rounding of
        the latents to 6 places must not move a single sign, and the count says so;
      * the margin sensitivity at 0.10 / 0.15 / 0.20 with its own positive control;
      * the Monte-Carlo sd of Q95 over 12 independent 2,000-draw nulls;
      * the cost, re-measured on the BUILT apparatus (2.8 carried 55 ms from 1.4's probe).

    Writes its JSON to `out_path` and NOTHING under tools/: the pin is the campaign's."""
    out_path = Path(out_path)
    if "tools" in out_path.parts:
        print(f"REFUSED: the dry run never writes under tools/ ({out_path})", flush=True)
        return 2
    t0 = time.time()
    print("=" * 78, flush=True)
    print(
        f"T080 DRY RUN -- one seed ({seed}) at the campaign widths, NO barred conjunct",
        flush=True,
    )
    print("=" * 78, flush=True)

    labels0, labels1 = t080_stream_labels(0), t080_stream_labels(1)
    differ = bool(labels0) and labels0 != labels1
    first_diff = next((i for i, (x, y) in enumerate(zip(labels0, labels1)) if x != y), None)
    print(
        f"stream offset ({T080_STREAM_OFFSET} x seed, BOTH pointers): "
        f"seed0 {labels0[:12]} | seed1 {labels1[:12]} | differ={differ} "
        f"first_diff_at={first_diff} (n {len(labels0)}/{len(labels1)})",
        flush=True,
    )
    if not differ:
        print("REFUSED at the apparatus: the per-seed offset does not vary the frames", flush=True)
        return 2

    det = t080_determinism(seed)
    print(
        f"(p) determinism twin: identical={det['identical']} on {det['rows']} ALIGNED rows "
        f"(ticks_to_k {det['ticks_to_k']}, taught_equal {det['taught_equal']}, "
        f"first_diff {det['first_diff']})",
        flush=True,
    )

    t_rec = time.time()
    rec = t080_record_brain(seed)
    binder = rec.pop("binder")
    ticks = sum(len(v) for v in rec["arms"].values()) + rec["ticks_to_k"]
    ms_per_tick = (time.time() - t_rec) * 1000.0 / max(ticks, 1)
    reading = {k: rec[k] for k in ("seed", "arm_order", "timings", "offset", "positions")}
    reading.update({k: rec[k] for k in ("ticks_to_k", "k", "taught", "taught_end")})
    reading["perception_acc"] = rec["perception_acc_end"]
    reading["audio_acc"] = rec["audio_acc_end"]
    reading["arms"] = {}
    for name, rows in rec["arms"].items():
        counts = _t080_arm_counts(rows)
        recs = t080_recognitions(rows, binder)
        counts["agree"] = recs["agree"]
        counts["recs"] = recs["recs"]
        counts["acc"] = rec["arm_acc"][name]
        reading["arms"][name] = counts
    aligned = reading["arms"]["aligned"]
    null = t080_null(aligned["recs"], seed)
    reading["null"] = null
    reading["within"] = aligned["bound_frac"]
    reading["q95_null"] = null["q95"]
    reading["q005_null"] = null["q005"]
    reading["q995_null"] = null["q995"]
    reading["q_lo_null"] = null["q_lo"]
    reading["q_hi_null"] = null["q_hi"]
    reading["null_in_vivo"] = reading["arms"]["null"]["bound_frac"]
    reading["congruent_aligned"] = aligned["congruent_frac"]
    reading["congruent_null"] = reading["arms"]["null"]["congruent_frac"]
    reading["lag_rates"] = {d: reading["arms"][f"lag{d}"]["bound_frac"] for d in T080_LAGS}
    reading["window"] = t080_window(reading["lag_rates"], null["q95"])
    reading["refusals"] = _t080_refusals(reading)

    print("-" * 78, flush=True)
    print(
        f"cost: {ms_per_tick:.1f} ms/tick measured on the BUILT apparatus over {ticks} ticks "
        f"(2.8 carried 55 ms from 1.4's probe) -> campaign estimate "
        f"{ms_per_tick * ticks * len(T080_SEEDS) / 1000.0 / 60.0:.1f} min for "
        f"{len(T080_SEEDS)} seeds x {ticks} ticks; timings {rec['timings']}",
        flush=True,
    )
    print("-" * 78, flush=True)
    _t080_seed_block(reading)
    total_agree = sum(a["agree"] for a in reading["arms"].values())
    total_scored = sum(a["n"] for a in reading["arms"].values())
    print(
        f"  DECOMPOSITION vs the recorded handle at {T080_LATENT_PLACES}-place rounding: "
        f"{total_scored - total_agree}/{total_scored} reproduce, {total_agree} disagree",
        flush=True,
    )

    print("-" * 78, flush=True)
    ms = t080_margin_sensitivity(rec["arms"]["aligned"], binder)
    print(
        f"margin sensitivity on the ALIGNED arm ({ms['n']} scored ticks, shipped margin "
        f"{ms['shipped_margin']}; control_ok {ms['control_ok']} -- the shipped margin's "
        f"re-threshold reproduces the decomposition's {ms['shipped_bound']} bound ticks):",
        flush=True,
    )
    for m, v in ms["margins"].items():
        print(
            f"  recog_margin {m:<6} bound {v['bound']:<5} = {v['frac']:.4f} "
            f"(sub-margin ticks: vision {v['sub_vision']}, audio {v['sub_audio']})",
            flush=True,
        )
    cv, ca = ms["coincidence"]["vision"], ms["coincidence"]["audio"]
    print(
        f"  centred coincidence IN VIVO: vision min {cv['min']} median {cv['median']} max "
        f"{cv['max']} | audio min {ca['min']} median {ca['median']} max {ca['max']}; "
        f"nearest-prototype winners agree on {ms['top_agree']} of {ms['n']} scored ticks",
        flush=True,
    )

    print("-" * 78, flush=True)
    sd = t080_q95_sd(aligned["recs"])
    print(
        f"Q95 Monte-Carlo sd: {sd['repeats']} INDEPENDENT {sd['n_draws']}-draw nulls of the "
        f"SAME ALIGNED recording -> mean {sd['mean']:.5f} sd {sd['sd']:.5f} "
        f"min {sd['min']:.5f} max {sd['max']:.5f}",
        flush=True,
    )
    print(f"  q95 draws: {[round(x, 5) for x in sd['q95']]}", flush=True)

    print("-" * 78, flush=True)
    print(
        "NO barred conjunct is read here: (g0) and (b1) are paired t's over 8 seeds and one "
        "seed cannot carry one. The single-seed differences above are printed with t = n/a.",
        flush=True,
    )
    print(
        f"  (g0) one seed: {reading['congruent_aligned'] - reading['congruent_null']:+.4f}  t: n/a",
        flush=True,
    )
    print(f"  (b1) one seed: {reading['within'] - reading['q95_null']:+.4f}  t: n/a", flush=True)
    print(
        f"  (b2) one seed: NULL IN VIVO {reading['null_in_vivo']:.4f} in family-wise "
        f"[{reading['q_lo_null']:.4f}, {reading['q_hi_null']:.4f}] -> "
        f"{reading['q_lo_null'] <= reading['null_in_vivo'] <= reading['q_hi_null']} "
        f"(per-seed [{reading['q005_null']:.4f}, {reading['q995_null']:.4f}] reported)",
        flush=True,
    )

    report = {
        "row": "T080 dry run",
        "head": _t080_head(),
        "seed": int(seed),
        "ms_per_tick": ms_per_tick,
        "ticks": ticks,
        "stream": {"seed0": labels0, "seed1": labels1, "differ": differ, "first_diff": first_diff},
        "determinism": det,
        "reading": reading,
        "decomposition": {"scored": total_scored, "disagree": total_agree},
        "margin_sensitivity": ms,
        "q95_sd": sd,
        "reds": t080_reds(),
        "constants": _t080_constants(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("-" * 78, flush=True)
    print(
        f"dry run written to {out_path} in {time.time() - t0:.0f} s; nothing under tools/",
        flush=True,
    )
    return 0


def _t080_constants():
    """Every declared constant of 2.8, pinned beside the reading it produced."""
    return {
        "K": T080_K,
        "warmup_max": T080_WARMUP_MAX,
        "n_ticks": T080_N_TICKS,
        "lag_ticks": T080_LAG_TICKS,
        "lags": list(T080_LAGS),
        "scramble_ticks": T080_SCRAMBLE_TICKS,
        "stream_offset": T080_STREAM_OFFSET,
        "n_draws": T080_N_DRAWS,
        "seeds": list(T080_SEEDS),
        "band": list(T080_BAND),
        "band_fw": list(T080_BAND_FW),
        "bar": T080_BAR,
        "margins": list(T080_MARGINS),
        "latent_places": T080_LATENT_PLACES,
        "align_rng_base": T080_ALIGN_RNG_BASE,
        "null_rng_base": T080_NULL_RNG_BASE,
        "scramble_rng_base": T080_SCRAMBLE_RNG_BASE,
    }


# distinct-from: faculties/agency_monitor.py::run_t079_campaign and
#   instruments/attention_schema.py::run_t078_campaign -- the PATTERN this follows (record, read,
#   contrast, pin ONCE) over a different instrument, different arms and a different pin.
#   Read; no state, no pin and no statistic is shared beyond `instruments/paired_contrast`.
# distinct-from: agency_monitor.py -- run_t079_campaign is T079's campaign over its own efference feed and pin; read 2026-09-16.
# distinct-from: attention_schema.py -- run_t078_campaign is T078's over the attention gates and its pin; read 2026-09-16.
# distinct-from: metacognitive_monitor.py -- run_t077_campaign is T077's over the metacognition scorer; read 2026-09-16.
# distinct-from: prior_drift_harness.py -- run_campaign is the drift harness's over its priors; read 2026-09-16.
#   Each row's campaign is its own instrument's; a shared runner would be a fifth mechanism, not one.
def run_t080_campaign(seeds=T080_SEEDS, pin=T080_PIN, latents=T080_LATENTS):
    """The campaign (2.2-2.3): record, decompose, null, contrast, pin ONCE. Returns an exit
    code.

    ORDER IS 2.3's, and it is the order the verdict is DECIDED in:
      (p)  DETERMINISM  two ALIGNED seed-0 recordings identical field by field, else REFUSED
                        -- nothing is read
      (r)  READABLE     K taught within the cap on every seed, and the decomposition equal to
                        the recorded handle on EVERY scored tick of EVERY arm, else REFUSED
                        with the seed and the arm named
      (g0) THE STREAM CARRIES WITHIN-WINDOW STRUCTURE  the CLASSIFIERS' congruence clears the
                        bar ALIGNED over NULL, else REFUSED ("the alignment did not reach the
                        brain"), never a STOP
      (b1) (b2)         the licensed sentence at the derived margin, and the two nulls'
                        agreement
    A REFUSAL is a finding about the apparatus or the instrument; a NOT MET is a finding about
    this binder on this stimulus. They are never merged."""
    pin = Path(pin)
    latents = Path(latents)
    if latents.exists():
        print(f"REFUSED: {latents} exists -- the companion is written once", flush=True)
        return 2
    if pin.exists():
        print(f"REFUSED: {pin} exists -- the pin is written once", flush=True)
        return 2
    t0 = time.time()
    print("=" * 78, flush=True)
    print(f"T080 CAMPAIGN -- seeds {list(seeds)}, six arms, the binder's own report", flush=True)
    print("=" * 78, flush=True)

    labels0, labels1 = t080_stream_labels(0), t080_stream_labels(1)
    if not labels0 or labels0 == labels1:
        print(
            f"REFUSED at the apparatus: the per-seed stream offset does not vary the frames "
            f"(seed 0 {labels0[:10]} vs seed 1 {labels1[:10]})",
            flush=True,
        )
        return 2
    det = t080_determinism(seeds[0] if seeds else 0)
    print(
        f"(p) determinism: identical={det['identical']} on {det['rows']} ALIGNED rows "
        f"(first_diff {det['first_diff']})",
        flush=True,
    )
    if not det["identical"]:
        print("REFUSED: two identical-seed recordings differ -- no reading is taken", flush=True)
        return 2

    per_seed, refusals = [], []
    for s in seeds:
        r = t080_seed(s)
        per_seed.append(r)
        refusals.extend(r["refusals"])
        print("-" * 78, flush=True)
        _t080_seed_block(r)

    got = t080_contrasts(per_seed)
    verdict = t080_verdict(got, refusals)
    print("=" * 78, flush=True)
    for why in refusals:
        print(f"  (r) REFUSAL: {why}", flush=True)
    for key in ("g0", "b1"):
        c = got[key]
        print(
            f"  {c.name}: diffs {[round(d, 4) for d in c.diffs]} t={c.t:.4f} FPR={c.fpr} "
            f"degenerate={c.degenerate} met={c.met}",
            flush=True,
        )
    print(f"  (b2) the two nulls agree: {got['b2']}", flush=True)
    print(f"  REPORTED windows per seed: {[r['window'] for r in per_seed]}", flush=True)
    print(
        f"  REPORTED scrambled-phase bound fraction: "
        f"{[round(r['arms']['scrambled']['bound_frac'], 4) for r in per_seed]}",
        flush=True,
    )
    print(f"T080 VERDICT: {verdict}", flush=True)

    # THE COMPANION (2.10 (g)): every scored tick's raw latents per seed and arm, float32, in
    # the SAME order as `per_seed[].arms[arm].recs`, so the decomposition and the in-vivo
    # handle re-derive from the two files alone. Popped from the JSON, hashed into it.
    arrays = {}
    for r in per_seed:
        for arm, la in r.pop("latents", {}).items():
            arrays[f"s{r['seed']}_{arm}_vis"] = la["vis"]
            arrays[f"s{r['seed']}_{arm}_aud"] = la["aud"]
    latents.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(latents, **arrays)
    latents_meta = {
        "path": str(latents),
        "sha256": hashlib.sha256(latents.read_bytes()).hexdigest(),
        "dtype": "float32",
        "keys": sorted(arrays),
        "shapes": {k: list(arrays[k].shape) for k in sorted(arrays)},
        "order": (
            "the scored ticks of each arm in recording order -- the same order as "
            "per_seed[].arms[arm].recs"
        ),
    }
    print(
        f"companion {latents}: {len(arrays)} arrays, sha256 {latents_meta['sha256'][:12]}",
        flush=True,
    )

    out = {
        "row": "T080",
        "head": _t080_head(),
        "seeds": list(seeds),
        "constants": _t080_constants(),
        "reds": t080_reds(),
        "determinism": det,
        "stream_labels": {"seed0": labels0, "seed1": labels1},
        # EVERY PER-TICK INPUT OF EVERY BARRED CONJUNCT: the DECOMPOSITION's own columns
        # (obj_v, obj_a, the two coincidences, bound, the recorded handle, the two classifier
        # labels) for every scored tick of every arm of every seed, so (g0), (b1) and (b2)
        # re-derive offline from this file alone by replaying t080_null / t080_contrasts.
        # The raw LATENTS are deliberately NOT pinned: 192 floats per tick x ~1,200 scored
        # ticks x 8 seeds is a 3.7 M-float artifact, and no conjunct reads them -- the
        # decomposition that does is what is recorded. Stated, not silently omitted.
        "rows_omitted": (
            "raw vis/aud latents (64 + 128 floats per tick) are NOT in this JSON: they are in "
            "the companion npz named under latents_npz (float32, sha256 pinned), in the same "
            "order as each arm's recs; the decomposition's per-tick columns are pinned here "
            "and are what every conjunct reads"
        ),
        "latents_npz": latents_meta,
        "per_seed": per_seed,
        "contrasts": {
            "g0": got["g0"]._asdict(),
            "b1": got["b1"]._asdict(),
            "b2": got["b2"],
        },
        "refusals": refusals,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"pinned {pin} in {time.time() - t0:.0f} s", flush=True)
    return 0 if verdict != "REFUSED" else 2


# =============================================================================
# The self-check line (2.9 step 2: main() prints the six booleans and EXITS NON-ZERO while
# any is False, so the gate carries the red until 2.9 step 4).
# =============================================================================
def _t080_reds_line(reds, out=print):
    """Print the six booleans beside what each one says."""
    for name in sorted(reds):
        out(f"  {name}: {str(reds[name]):5s}  {T080_RED_LINES[name]}")


# distinct-from: reasoning/spike_chat_lake.py::_wants_repl -- decides whether `__main__` is an
#   interactive REPL or the self-check for THAT module. This one has no REPL and always runs
#   its reds; the two flags below only ADD the brain-touching entry points. Read.
# distinct-from: infra/brain_telemetry.py::_main and substrate/home.py's self-check --
#   each module's own `__main__` block, which is this tree's convention (run_selfchecks.py
#   executes exactly that block per module and scores the exit status), not a second
#   dispatcher. Read; nothing here runs another module's self-check.
# distinct-from: gates/selfcheck_cache.py::closure_key and brain_atlas.py::region_of --
#   a cache key over the MODULES table and a module -> region lookup. Term overlap ("module",
#   "main") only; neither runs anything. Read.
def main(argv=None):
    """The module's self-check, plus 2.9's two brain-touching entry points.

    BOTH of those run under `throwaway_home()` INSIDE their recorders, every brain is
    `persist=False`, and NEITHER is reachable from the gate's bare `python -m` invocation --
    which runs the six reds on constructed inputs and touches no brain, no lake and no
    network."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--t080-campaign" in argv:  # 2.9 step 6: the campaign, pinned ONCE
        return run_t080_campaign()
    if "--t080-dry-run" in argv:  # 2.9 step 5: the apparatus dry run, NO barred conjunct
        i = argv.index("--t080-dry-run")
        path = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else None
        return run_t080_dry_run(T080_DRY_RUN_OUT if path is None else path)
    print("=" * 78)
    print("T080 -- the temporal-binding instrument (specs/022 [US4], C5 row temporal_binding)")
    print("=" * 78)
    print(
        f"  K {T080_K} | warmup max {T080_WARMUP_MAX} | arms {list(T080_ARMS)} | "
        f"ticks {T080_ARM_TICKS} | lags {list(T080_LAGS)}"
    )
    print(
        f"  bar {T080_BAR} | seeds {list(T080_SEEDS)} | n_draws {T080_N_DRAWS} "
        f"(reds {T080_REDS_N_DRAWS}) | band {T080_BAND} family-wise {T080_BAND_FW} | "
        f"offset {T080_STREAM_OFFSET} x seed"
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
    # THE CHARGED READING IS RE-DERIVED FROM THE PIN ON EVERY GATE RUN (the T080 panel fold,
    # 2026-09-16): the C5 row's 0.2309 was a hand-typed literal with nothing on the tree tying
    # it to tools/t080_temporal_binding.json. Absence is LOUD (ignition_instrument's baseline
    # discipline), never a skip: the pin is tracked, so a missing file is a deletion or a broken
    # path. The same pass reddens the clause with a mutated derivation, so it is shown reachable.
    derived, reading, mutated = t080_pin_reading()
    print(
        f"  C5 temporal_binding: reading {reading} re-derived from {T080_PIN.name} as {derived} "
        f"(a mutated derivation reads {mutated} and would refuse)"
    )
    if derived != reading or mutated == reading:
        print(
            f"T080 SELF-CHECK RED: C5 reading {reading} != {derived} re-derived from {T080_PIN} "
            f"(mutated {mutated}) -- re-derive or explain before re-recording"
        )
        return 1
    return 0


# distinct-from: amdahl_gate.py -- its repeatability red compares two timing readings' bands; this re-derives
#   a C5 row's charged number from a committed pin. Read 2026-09-16.
# distinct-from: evidence_accumulation.py -- a replay-twice charge A/B over serotonin frames; no pin, no row.
# distinct-from: task_set.py -- a replay-twice charge A/B over film frames; no pin, no row.
# distinct-from: gwt_deep.py -- an F1 over recovered pairs on a synthetic task; no pin, no row.
def t080_pin_reading(pin=None):
    """(derived, recorded, mutated): 2.6's reading -- the mean over seeds of (the ALIGNED arm's
    bound fraction minus its offline null's MEAN), rounded to 4 places -- re-derived from the
    committed pin; the C5 row's stored value; and the same formula with the null's q95 in place
    of its mean (the MUTATED derivation, which must NOT equal the recorded reading -- the clause
    is shown reachable on every run, never assumed). Raises when the pin is absent."""
    from snn_data_lake.gates.conclusion_protocol import CONCLUSION

    pin = T080_PIN if pin is None else Path(pin)
    assert pin.exists(), (
        f"{pin} missing -- the C5 reading's re-derivation cannot run; the pin is tracked, so "
        "absence means deletion or a broken path resolution"
    )
    seeds = json.loads(pin.read_text(encoding="utf-8"))["per_seed"]
    derived = round(sum(e["within"] - e["null"]["mean"] for e in seeds) / len(seeds), 4)
    mutated = round(sum(e["within"] - e["q95_null"] for e in seeds) / len(seeds), 4)
    return derived, CONCLUSION["temporal_binding"][3], mutated


if __name__ == "__main__":
    sys.exit(main())
