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

import sys
from pathlib import Path

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
# 2.9 step 2: THE SKELETON. Every function below is a STUB returning a NULL reading -- an
# empty list, a degenerate contrast, False, or None -- so that 2.9 step 3's R1-R6 run on
# their constructed inputs and fail on their own ASSERTIONS rather than on a missing symbol.
# The constructed-input BUILDERS are real here and not stubbed, for exactly that reason: a
# red that dies on a TypeError has not been shown to be reachable.
# =============================================================================
def t080_aligned_order(vis_labels, aud_labels, rng):
    """STUB (2.9 step 2). Per vision position an audio frame index whose label mod 10 equals
    the vision label, cycling within each class in a per-seed shuffled order (2.1)."""
    return []


def t080_record_brain(seed, arms=None, n_ticks=None):
    """STUB (2.9 step 2). The recorder of 2.1 / 2.2: a scratch brain, the aligned pack
    installed at the stimulus, the warm-up to K, then the arms, rows returned."""
    return None


def t080_recognitions(rows, binder, recognize=None):
    """STUB (2.9 step 2). The decomposition of `bind()`'s handle through the binder's OWN
    `_project` and `_recognize` (2.1), with `agree` counting the ticks where it DISAGREES
    with the recorded handle -- (r) requires 0."""
    return {"recs": [], "agree": 0, "n": 0}


def t080_null(recs, seed, n_draws=T080_N_DRAWS):
    """STUB (2.9 step 2). The membership-matched null at matched K (2.1): the audio-object
    column permuted across the scored ticks, `n_draws` draws, q95 the derived margin."""
    return {
        "mean": 0.0,
        "q95": 0.0,
        "q005": 0.0,
        "q995": 0.0,
        "expected": 0.0,
        "draws_sd": 0.0,
        "multiset_ok": False,
        "n": 0,
        "n_draws": 0,
    }


def t080_window(lag_rates, q95):
    """STUB (2.9 step 2). The binding window (2.1): the smallest lag whose bound fraction is
    at or below the derived margin, else "none"."""
    return None


def t080_contrasts(per_seed, score=None, band=T080_BAND, g0=None):
    """STUB (2.9 step 2). 2.3's conjuncts (g0), (b1) and (b2) through
    `substrate/paired_contrast.contrast`."""
    return {
        "g0": contrast("(g0) the classifiers' congruence: ALIGNED - NULL", ()),
        "b1": contrast("(b1) within - the null's q95", ()),
        "b2": {"outside": [], "inside": 0, "n_seeds": 0, "met": False},
        "n_seeds": 0,
    }


def t080_reds(n_draws=T080_REDS_N_DRAWS):
    """STUB (2.9 step 2). 2.4's six reds as ONE self-check line. All False here BY DESIGN:
    the self-check RUNS RED until 2.9 step 4 greens the instruments."""
    return {"R1": False, "R2": False, "R3": False, "R4": False, "R5": False, "R6": False}


def _t080_mutated_scorer(kind):
    """STUB (2.9 step 2). 2.4's six mutated scorers, each the deliberately broken twin of the
    instrument whose red it must redden. Every one returns its target's NULL reading here, so
    a step-3 test that runs a mutation fails on its assertion and not on a missing name."""

    def null_resamples(recs, seed, n_draws=T080_N_DRAWS):
        return t080_null(recs, seed, n_draws)

    def ignore_margin(binder):
        return lambda code, modality: (None, 0.0)

    def window_first_lag(lag_rates, q95):
        return t080_window(lag_rates, q95)

    def g0_always_met(entry):
        return 0.0

    def blind_scorer(entry, which):
        return 0.0

    def band_unbounded():
        return T080_BAND

    return {
        "null_resamples": null_resamples,
        "ignore_margin": ignore_margin,
        "window_first_lag": window_first_lag,
        "g0_always_met": g0_always_met,
        "blind_scorer": blind_scorer,
        "band_unbounded": band_unbounded,
    }[kind]


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
        v = [rng.random() for _ in range(T080_R2_VIS_DIM)]
        a = [rng.random() for _ in range(T080_R2_AUD_DIM)]
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
            v = [x + rng.uniform(-T080_R2_NOISE, T080_R2_NOISE) for x in pv]
            a = [x + rng.uniform(-T080_R2_NOISE, T080_R2_NOISE) for x in pa]
        else:
            v = [rng.random() for _ in range(T080_R2_VIS_DIM)]
            a = [rng.random() for _ in range(T080_R2_AUD_DIM)]
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
            f"T080 SELF-CHECK RED: {sum(reds.values())}/{len(reds)} reds fire -- the "
            f"instruments are STUBS (2.9 step 2); step 4 greens them"
        )
        return 1
    print(
        f"  T080 OK: R1-R6 fire on their constructed inputs and each is reddened IN FACT by "
        f"its mutated scorer ({sum(reds.values())}/{len(reds)}); no brain, no lake, no network"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
