r"""
T078 PART I -- the attention schema on the live loop: THE TWO GATES FIRST, and nothing else.

THE ROW. specs/022 Phase 4 / C4, tasks.md :163 ("the attention schema"), invention register #8,
SC-009's first half, PLAN-FINAL Part F row C-R7. The pre-registration this file implements is
`specs/022-unified-brain-arc/evidence/t078-attention-schema.md`, sections 1-2, FROZEN at commit
02d677e BEFORE a line of this module existed. Every form below is 2.1's; every bar is 2.3's;
every red is 2.4's. NO NUMBER IN THIS FILE IS A READING -- the campaign writes the readings, once.

WHAT PART I IS. Two gates, their six reds, and ONE recording of a SCRATCH brain. There is NO
attention schema in Part I: no ACS, no report, no third factor, no wire to brain.py. The
algorithm's own build order (docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md 12) is "G3 before any
ACS", and SYNTHESIS.md :65 keeps G3/G4's stop/void wording intact.

  G3 TASK BINDING (a STOP gate). Does the action path consume the attention state one tick later
      on the loop that exists? MI(focus band at t ; committed action at t+1) over consecutive
      AWAKE pairs, against a 2,000-permutation floor, paired over 8 seeds against T_BAR. Its
      PRECONDITION is the estimator's own positive control MI(action_t ; action_{t+1}) -- actions
      are autocorrelated on this loop, so an estimator that cannot read THAT is blind on this
      stream and G3 is REFUSED (a finding about the instrument), never STOPPED (a finding about
      the loop). SCOPED, and the scoping is stated rather than smoothed (2.7): this loop has no
      cued gap and no held goal, so G3 here is NOT the algorithm's G3. A PASS says only that the
      action path consumes the attention state one tick later on this loop.
  G4 CONTENT LEAKAGE (a VOID gate). Is the candidate control state `u_t` content-blind in
      INFORMATION, not merely by naming? A held-out decoder of the frame LABEL from `u_t` (15
      inputs) against its own 200-draw label-shuffle band, beside the SAME decoder at the SAME
      input dimension fed the attended CONTENT (the gated 64-d latent projected to 15). VOID iff
      `u_t` decodes the label above its band AND the matched content decode cannot be told from
      it -- content-blindness would then be a naming convention and every downstream schema
      reading is VOID, not weakened.

WHAT THIS MODULE REUSES rather than rebuilds (P2; checked with graph-exists 2026-09-15):
  * `substrate/paired_contrast` -- `contrast` and `T_BAR = 1.895`, THE seed-paired statistic with
    its exact sign-flip FPR. Imported AT MODULE LEVEL and never through
    `cortex/prior_drift_harness`: the harness is an EXEMPT instrument and a live import of it
    stales five dormancy rows (T077 2.10, which is why the statistic was lifted in the first place).
  * `cortex/metacognitive_monitor` -- the SHAPE of `t077_record_brain` (a scratch brain under
    `throwaway_home()`, per-tick rows, an INSTANCE's method wrapped to capture what it is fed and
    what it returns), of `_t077_mutated_scorer` / `t077_reds` (every red reddened IN FACT by a
    deliberately broken scorer, run, never argued), and the q95 convention of `track_error`
    (`draws[min(len-1, round(0.95*(len-1)))]`) -- which `substrate/lake_engine::AppendBudget._p95_of`
    also spells, so `_q95` is the THIRD spelling of it and not the second; the `# distinct-from:`
    block above `_q95` carries the P2 debt and the home a successor's merge belongs in.
  * `cortex/integration_measure.source_entropy` -- the tree's binary entropy.
  * `substrate/home.throwaway_home` -- the scratch NORD_HOME every recording runs under. THE LIVE
    LAKE IS NEVER TOUCHED and the daemon's brain is never constructed here.
ABSENT from the tree on 2026-09-15 and therefore this row's two new symbols: a mutual-information
estimator with a permutation floor, and a held-out decoder with a label-shuffle band.

DECLARED CONSTANTS, each with its source (2.8 computes the consequence of each; nothing here is a
free parameter introduced by the builder):
  N_MIN_PER_CELL = 10   2.1/2.8 -- pairs per OBSERVED joint cell below which `mi_read` REFUSES.
  N_PERM         = 2000 2.1     -- permutations of y behind every MI floor.
  N_SHUFFLE      = 200  2.1     -- label-permuted decoders behind every decode band.
  N_TEST_MIN     = 50   2.3 (r) -- held-out rows below which a decode is not readable.
  U_DIM          = 15   1.5/A4  -- 2B + L + 5 at B = 4, L = 2; also the MATCHED input dimension
                                   of every barred G4 decode.
  T078_N_TICKS   = 1200 2.2/2.8 -- ticks per seed (about 585 awake, about 580 awake pairs).
  T078_SEEDS     = 0..7 2.2     -- PLAN-FINAL Part F's S = 8.
  DECODER_STEPS / DECODER_LR / DECODER_L2 = 200 / 0.1 / 1e-3   2.1 -- a fixed, pre-registered
                                   optimizer budget whose SENSITIVITY WAS NOT COMPUTED. 2.8 lists
                                   the three under "the consequence of each, computed here" and
                                   substitutes the raw-64 ceiling decode for the computation; that
                                   substitution cannot do the job, because the ceiling runs through
                                   the SAME `_fit_and_score` at the SAME budget as the arms it is
                                   meant to diagnose (`t078_g4` reads all four decodes through one
                                   `decode_heldout`), so it is partly the method (2.5, P8 clause 2)
                                   and cannot separate capacity from budget. The panel fold's dated
                                   item in section 3.5 of the evidence file carries the
                                   re-derivation and its numbers; the pre-registered reading at the
                                   pre-registered budget remains the record.
  PROJ_TAG = b"t078-proj"  2.1 -- `crc32(PROJ_TAG) + seed` seeds the (64, 15) Gaussian projection.
BUILDER-INTRODUCED beyond 2.1/2.8, disclosed here because every constant owes its why:
  DETERMINISM_TICKS = 100  -- 2.2 words the determinism precondition as "two 100-tick recordings";
                             the number is the pre-registration's, the NAME is the builder's.

Host: Python 3.14.6 free-threaded, torch 2.11.0+cu128 where present. `decode_heldout` imports torch
INSIDE the call, so importing this module costs no torch; the decode itself is identical on CPU and
CUDA by seed and the tests run it on CPU. ASCII-only prints (Windows cp1252). Deterministic:
`random.Random(seed)` per call, `zlib.crc32` for persisted seeds, never the builtin `hash()`.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.attention_schema
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.attention_schema --t078-campaign
"""

import hashlib
import json
import math
import random
import statistics
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

from snn_data_lake.substrate.paired_contrast import T_BAR, contrast
from snn_data_lake.substrate.report_channel import snapshot as _report_snapshot

# distinct-from: cortex/metacognitive_monitor.py::track_error -- a TYPE-2 AUROC over one
#   (confidence, outcome) stream against its own label-permutation null. This is the MUTUAL
#   INFORMATION of TWO discrete streams against a permutation floor: a different statistic
#   answering a different question (does the attention state predict the next action, not does
#   confidence track error). Its q95 CONVENTION is reused verbatim; its scorer is not.
# distinct-from: cortex/integration_measure.py::source_entropy -- the Shannon entropy of ONE
#   binary stream. `mi_bits` is the entropy of a JOINT distribution over two streams; the binary
#   entropy itself is IMPORTED from there rather than re-derived (P2).
# distinct-from: neuromodulation/empowerment.py::Empowerment -- reachable-cell counting over
#   world-model rollouts, which its own docstring already declares is "NOT exact mutual
#   information" and which carries no null. This estimates MI on an OBSERVED recording, with a
#   permutation FLOOR and an explicit refusal, and constructs no world model at all.

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it. `_fit_and_score`
#: is inferred CLAIM by the census (it calls `torch.autograd.grad`), so it owes a declaration --
#: the gate reported MISSING_DECLARATION on the commit that added it, which is the registration
#: working, not the gate misfiring.
#:
#: NOT_A_RULE is the honest label and the reason states the gradient rather than hiding it. It DOES
#: compute an exact reverse-mode gradient, of an OFFLINE ANALYSIS DECODER's own convex multinomial
#: logistic loss, over a RECORDING that already exists. No synapse, trace, threshold or weight of
#: the brain is touched by it; it never runs on a tick; its output is an ACCURACY, read against a
#: label-shuffle band, and thrown away. The precedent is `cortex/eprop_kernel::_first_layer_grad`,
#: a gradient-computing function declared NOT_A_RULE because it DIAGNOSES rather than learns, and
#: the sibling is `metacognitive_monitor`'s two NOT_A_RULE rows. EXACT_GRADIENT was considered and
#: NOT taken: in this tree that value means "the independent BPTT reference the e-prop cosine
#: clause compares against" (eprop_kernel::bptt_grads), and this decoder is a reference for
#: nothing -- claiming it would put a measurement probe into the rule census as a bearing row.
GRADIENT_CLAIM = {
    "_fit_and_score": (
        "NOT_A_RULE",
        "fits m multinomial-logistic ANALYSIS decoders by full-batch autograd descent over an "
        "already-recorded (rows, labels) table, to read whether the label is decodable above a "
        "label-shuffle band (G4). An estimator, not a learning rule: it trains no part of the "
        "brain, touches no synapse or trace, never runs on a tick, and its weights are discarded "
        "with the reading. Same disposition and same reason shape as cortex/eprop_kernel::"
        "_first_layer_grad -- a gradient that measures instead of learning",
    ),
}

# ---- 2.1 / 2.8's declared constants -------------------------------------------------------------
N_MIN_PER_CELL = 10  # pairs per OBSERVED joint cell; below it `mi_read` REFUSES (2.1, 2.8)
N_PERM = 2000  # permutations of y behind every MI floor (2.1)
N_SHUFFLE = 200  # label-permuted decoders behind every decode band (2.1)
N_TEST_MIN = 50  # held-out rows below which a decode is not readable (2.3 (r))
U_DIM = 15  # A4's 2B + L + 5 at B = 4, L = 2 -- and the MATCHED G4 input dimension
LATENT_DIM = 64  # the brain's normalized percept latent (brain.py::Brain._perceive_real)
N_BANDS = 4  # LearnedAttention(dim=64, n_bands=4), constructed brain.py:1066
DECODER_STEPS = 200  # full-batch gradient descent steps (2.1)
DECODER_LR = 0.1  # decoder learning rate (2.1)
DECODER_L2 = 1e-3  # decoder L2 penalty (2.1)
PROJ_TAG = b"t078-proj"  # crc32(PROJ_TAG) + seed seeds the (64, 15) Gaussian projection (2.1)
DETERMINISM_TICKS = 100  # the determinism twin's length (2.2); the NAME is the builder's
#: The committed-action label the E-STOP path returns (brainstem/autonomous_loop.py `step`
#: :1895-1919). 2.2 records that it NEVER occurs under `_safe_ctx`, so a nonzero
#: `dropped_estop` in any seed is itself a finding and the campaign prints it per seed.
ESTOP_LABEL = "EMERGENCY_STOP"
#: BUILDER-INTRODUCED, disclosed (2.9 step 4). The reds and the tests run their MI floors at 200
#: draws, not the campaign's 2,000: 2.1 fixes n_perm = 2,000 for the CAMPAIGN's four readings, and
#: `t078_reds()` is executed by every run_selfchecks pass, where 2,000 x 8 reads would dominate the
#: gate. 2.4's n_shuffle = 200 and n_test = 300 are used VERBATIM because R4 asserts on them.
REDS_N_PERM = 200
#: BUILDER-INTRODUCED, disclosed: R2's constructed action streams are 2,000 ticks long. 2.4 fixes
#: neither n nor the alphabet for R2, and the apparatus probe measured ~12 action labels, so a
#: 12-class pair stream spans up to 144 observed joint cells and N_MIN_PER_CELL x 144 = 1,440 pairs
#: is the floor the REFUSAL itself imposes -- a 600-tick R2 would be refused by R3's own rule.
REDS_ACTION_N = 2000
REDS_ACTION_K = 12
T078_N_TICKS = 1200  # ticks per seed (2.2, 2.8)
T078_SEEDS = tuple(range(8))  # PLAN-FINAL Part F: S = 8
T078_BAR = T_BAR  # the ONE bar; paired one-sided t, 7 d.f., alpha = 0.05
T078_PIN = Path("tools/t078_attention_schema.json")  # written ONCE (2.1)
T078_LATENTS = Path("tools/t078_latents.npz")  # the 64-d latents; sha256 carried in the pin

#: A4's `u_t`, in the ORDER 2.1 lists it: gate(4), precision(2), onehot(focus | NONE)(5),
#: focus_strength, ignited, drive_margin, sustain. IMPOVERISHED BY CONSTRUCTION -- the static
#: assert below is what makes "content-blind by TYPE" a check rather than a claim (2.4 R5).
U_FIELDS = (
    "gate0",
    "gate1",
    "gate2",
    "gate3",
    "pc_surprise",
    "pc_dev",
    "focus_is_0",
    "focus_is_1",
    "focus_is_2",
    "focus_is_3",
    "focus_is_none",
    "focus_strength",
    "ignited",
    "drive_margin",
    "sustain",
)

#: Names that would make the vector a content carrier. 2.1's IMPOVERISHED clause, as a static
#: membership check over U_FIELDS -- run at import, not at review time.
U_FORBIDDEN = ("latent", "chan", "salience", "surprise_raw", "reward", "weight")

assert len(U_FIELDS) == U_DIM, f"u_t must carry exactly {U_DIM} fields: {len(U_FIELDS)}"
assert not (set(U_FIELDS) & set(U_FORBIDDEN)), (
    f"u_t is IMPOVERISHED by construction: {sorted(set(U_FIELDS) & set(U_FORBIDDEN))}"
)


# distinct-from: cortex/h01_ingest.py::_ei_band / check_ei -- the tree's other permutation bar, and
#   the nearest thing to `_q95` here. READ before this marker was written: it bars a CORRELATION r
#   between two instruments, takes its bar from `np.percentile` (which INTERPOLATES between order
#   statistics), and RAISES IngestRefused instead of returning a reading. `_q95` is the exact
#   non-interpolating order statistic 2.1 pre-registers -- "the q95 as `track_error` takes it" --
#   over MI draws and decoder accuracies. DISCLOSED P2 debt rather than hidden, and CORRECTED
#   2026-09-15 in the panel fold: this is the THIRD spelling of that non-interpolating order
#   statistic in the tree, not the second. The others are the inline copy in
#   `cortex/metacognitive_monitor::track_error`, which 2.1 names as THE convention, and
#   `substrate/lake_engine::AppendBudget._p95_of`, which landed at 37c3203 on 2026-09-06 -- NINE
#   DAYS before this row's pre-registration -- so this module's own stated trigger ("the moment a
#   THIRD caller appears") was already met when `_q95` was written, and the sentence that said
#   "two spellings" was false at HEAD. The reason the merge is not taken here is ALSO corrected: it
#   changes EXECUTABLE code in two shipped instruments outside this row's brief and is a row of its
#   own, NOT that T077's file was untouchable (5b020cf edited `metacognitive_monitor` in this same
#   push -- one file, 5/3, comment-only, and a T077-arc commit four commits after this disclosure
#   was written). If a successor takes the lift, the home is `substrate/_mathutils` -- the tree's
#   DECLARED home for byte-equivalent duplicated helpers, whose own docstring states the doctrine
#   and which `metacognitive_monitor` already imports -- and not `paired_contrast`, which is the
#   seed-paired statistic plus `T_BAR` and would become a second home for shared helpers. Two
#   hazards to carry with it, both read here: `track_error` sorts `draws` IN PLACE and the next
#   statement reads `shuffled_auroc2` as the median of that same list, so a non-mutating helper
#   moves a pinned T077 number; and `_p95_of` must keep its `sorted(xs)` copy, because an
#   in-place-sorting helper would reorder the list `AppendBudget._recent` holds as a rolling window,
#   which `bar()` passes directly and `end_tick` appends to and trims BY POSITION.
# distinct-from: substrate/determinism_gate.py::measure_frac_bits -- a sweep for the smallest
#   fractional-bit width whose spike trains equal an FP64 reference. A search over widths, not a
#   percentile of a null. Read; unrelated.
# distinct-from: cortex/executive_control.py::search -- beam/tree search over action SEQUENCES.
#   Read; shares only the word "score". distinct-from: cortex/neuron_zoo.py::_worker_ladder (a
#   thread-count race) and substrate/selfcheck_cache.py::_refresh_derived (a roster recompute) --
#   both read, neither computes a percentile, a null or an information quantity.
def _q95(draws):
    """The 95th percentile of the draws, EXACTLY as `track_error` takes it
    (metacognitive_monitor:915): the order statistic at round(0.95 * (len - 1)), never an
    interpolation. ONE convention for every floor and every band in this module."""
    draws.sort()
    return draws[min(len(draws) - 1, int(round(0.95 * (len(draws) - 1))))]


def mi_bits(x, y):
    """The plug-in mutual information in bits of two equal-length discrete label sequences: the
    joint counts over the OBSERVED cells only, with NO smoothing. The plug-in estimate of a finite
    sample is positive even under independence; nothing here corrects that, because the
    permutation floor in `mi_read` is what absorbs it (2.1) and a bias correction would be a second
    mechanism doing the floor's job. graph-exists found no MI estimator in the tree on
    2026-09-15; the three nearest relatives are the `# distinct-from:` lines at the top."""
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    return _mi_from_counts(Counter(zip(x, y, strict=True)), Counter(x), Counter(y), n)


def _mi_from_counts(joint, px, py, n):
    """MI in bits from joint and marginal COUNTS. Split out so the permutation floor can reuse the
    marginals, which a permutation of y leaves unchanged -- the same mechanism, not a second one."""
    total = 0.0
    for (a, b), c in joint.items():
        total += (c / n) * math.log2((c * n) / float(px[a] * py[b]))
    return total


def _mi_read(x, y, n_perm=N_PERM, seed=0, n_min_per_cell=N_MIN_PER_CELL, min_rule="per_cell"):
    """The core of `mi_read`. `n_min_per_cell` is a parameter ONLY so R3's mutation
    (N_MIN_PER_CELL := 0) can be RUN rather than argued -- the public `mi_read` hardcodes the
    declared constant and the campaign never reaches this spelling. `min_rule` is 2.10's: "per_cell"
    (the pre-registered rule, n >= N_MIN_PER_CELL x observed joint cells -- the PRIMARY reading) or
    "per_class" (n >= N_MIN_PER_CELL x |y classes| -- the (g3c) CONTROL only, whose 12 x 12 joint would
    otherwise refuse at every pair count N = 1,200 can produce; its floor is exact at any cell count)."""
    n = len(x)
    px, py = Counter(x), Counter(y)
    joint = Counter(zip(x, y, strict=True)) if n and n == len(y) else Counter()
    cells = len(joint)
    out = {
        "n": n,
        "cells": cells,
        "mi": None,
        "perm_q95": None,
        "perm_median": None,
        "n_perm": 0,
        "readable": False,
        "refusal": None,
    }
    if n == 0 or n != len(y):
        out["refusal"] = f"length mismatch: {n} against {len(y)}"
        return out
    if len(px) < 2 or len(py) < 2:
        out["refusal"] = f"a one-class axis (|x| = {len(px)}, |y| = {len(py)}): MI is 0 by identity"
        return out
    need = n_min_per_cell * (len(py) if min_rule == "per_class" else cells)
    if n < need:
        unit = (
            f"{len(py)} y classes" if min_rule == "per_class" else f"{cells} observed joint cells"
        )
        out["refusal"] = (
            f"{n} pairs over {unit} is below {n_min_per_cell} per {'class' if min_rule == 'per_class' else 'cell'} "
            f"({need} required): the floor cannot resolve this stream"
        )
        return out
    rng = random.Random(seed)
    shuffled = list(y)
    draws = []
    for _ in range(max(int(n_perm), 1)):
        rng.shuffle(shuffled)
        draws.append(_mi_from_counts(Counter(zip(x, shuffled, strict=True)), px, py, n))
    out["mi"] = _mi_from_counts(joint, px, py, n)
    out["perm_median"] = sorted(draws)[len(draws) // 2]
    out["perm_q95"] = _q95(draws)
    out["n_perm"] = len(draws)
    out["readable"] = True
    return out


def mi_read(x, y, n_perm=N_PERM, seed=0, min_rule="per_cell"):
    """`mi_bits` against its OWN permutation floor: the 95th percentile of `mi_bits(x, permuted y)`
    over `n_perm` permutations of y, drawn by a seeded `random.Random` -- the same convention as
    `metacognitive_monitor.track_error`, whose q95 order statistic is reused verbatim.

    Returns {n, cells, mi, perm_q95, perm_median, n_perm, readable, refusal}. REFUSES (readable
    False, NO mi) below `N_MIN_PER_CELL` pairs per OBSERVED joint cell, or when either axis carries
    one class. `n_perm` and `refusal` are beyond 2.1's listed keys and are reported, never barred:
    a refusal that does not say why is a number without a reason."""
    return _mi_read(
        x, y, n_perm=n_perm, seed=seed, n_min_per_cell=N_MIN_PER_CELL, min_rule=min_rule
    )


def _fit_and_score(x_tr, y_tr, x_sc, y_sc, k, seed):
    """`m` multinomial logistic decoders AT ONCE: the weight tensor is (m, d, k), so the 200-draw
    band is ONE batched fit rather than 200 sequential ones (2.1). Full-batch gradient descent,
    DECODER_STEPS steps at DECODER_LR with an L2 penalty of DECODER_L2; the models are independent,
    so summing their losses is exactly the per-model objective.

    `y_tr` is (m, n_train) and `y_sc` is (m, n_score) -- the band's models differ only in their
    TRAIN labels and are scored on the TRUE test labels; R4's mutation is the one caller that
    passes a per-model scoring target. The init is drawn from an explicit CPU `torch.Generator`:
    2.1 words it `torch.manual_seed(seed)`, and a local generator is how "CPU or CUDA identically
    by seed" is actually obtained (the global seed draws DIFFERENT numbers on the two devices),
    without touching process-global RNG state -- which matters on a free-threaded interpreter."""
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    m, d = y_tr.shape[0], x_tr.shape[1]
    w = (0.01 * torch.randn(m, d, k, generator=gen)).requires_grad_(True)
    b = torch.zeros(m, 1, k, requires_grad=True)
    xb = x_tr.unsqueeze(0)
    for _ in range(DECODER_STEPS):
        logp = (xb @ w + b).log_softmax(-1)
        nll = -logp.gather(-1, y_tr.unsqueeze(-1)).squeeze(-1).mean(-1)
        loss = nll.sum() + DECODER_L2 * (w * w).sum()
        gw, gb = torch.autograd.grad(loss, (w, b))
        with torch.no_grad():
            w -= DECODER_LR * gw
            b -= DECODER_LR * gb
    with torch.no_grad():
        pred = (x_sc.unsqueeze(0) @ w + b).argmax(-1)
        return (pred == y_sc).float().mean(-1).tolist()


def _decode(X, y, seed, n_shuffle=N_SHUFFLE, score_train=False):
    """The core of `decode_heldout`. `score_train` is a parameter ONLY so R4's mutation (a decoder
    scored on the half it was fitted on, which collapses the band to 1.0) can be RUN -- the public
    `decode_heldout` never scores on train and the campaign never reaches this spelling."""
    import torch

    n = len(y)
    n_train = n // 2
    ytr_raw, yte_raw = list(y[:n_train]), list(y[n_train:])
    out = {
        "n_train": n_train,
        "n_test": n - n_train,
        "k": 0,
        "acc": None,
        "shuffle_q95": None,
        "shuffle_median": None,
        "readable": False,
        "refusal": None,
    }
    if n != len(X):
        out["refusal"] = f"length mismatch: {len(X)} rows against {n} labels"
        return out
    if out["n_test"] < N_TEST_MIN:
        out["refusal"] = f"{out['n_test']} held-out rows is below N_TEST_MIN = {N_TEST_MIN}"
        return out
    if len(set(ytr_raw)) < 2 or len(set(yte_raw)) < 2:
        out["refusal"] = (
            f"a half with one class (train {len(set(ytr_raw))}, test {len(set(yte_raw))}): "
            f"the decode is undefined"
        )
        return out
    classes = sorted(set(y))
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    out["k"] = k
    x_tr = torch.tensor(X[:n_train], dtype=torch.float32)
    x_te = torch.tensor(X[n_train:], dtype=torch.float32)
    mu = x_tr.mean(0, keepdim=True)  # standardized on the TRAIN half only -- no lookahead
    sd_raw = x_tr.std(0, unbiased=False, keepdim=True)
    sd = sd_raw.clamp_min(1e-6)
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    # A column whose TRAIN-half sd is EXACTLY zero carries no train information by construction,
    # and `clamp_min(1e-6)` turns it into a 1e6 amplifier for any test row on which it fires: the
    # standardized feature reads (1 - 0) / 1e-6 = 1e+06 against a weight that received only the L2
    # gradient (never an NLL one, the column being constant in train), so that row's argmax is
    # decided by the random init. Measured on a constructed 15-d input: 1e+06 exactly, a logit
    # contribution of order 1e4 against the other fourteen columns' 1.24e4 combined, and one
    # held-out row of 100 flipped. Zeroing those columns in BOTH standardized halves is what 2.10
    # (ii)'s "a zero-variance column reads 0" already says; `clamp_min` STAYS for the genuinely
    # small-but-non-zero case (the smallest non-zero train-half sd on the pinned recording is
    # 0.0236, four orders above the floor). `U_DIM` and therefore `d` are unchanged, so
    # `_fit_and_score`'s (m, d, k) init draw is untouched and no pinned number moves -- verified by
    # re-deriving the pin at DECODER_STEPS = 200 on all 8 seeds after this guard landed.
    const = sd_raw == 0.0
    if bool(const.any()):
        x_tr, x_te = x_tr.masked_fill(const, 0.0), x_te.masked_fill(const, 0.0)
    y_tr = torch.tensor([[idx[c] for c in ytr_raw]], dtype=torch.long)
    y_te = torch.tensor([[idx[c] for c in yte_raw]], dtype=torch.long)
    x_score = x_tr if score_train else x_te
    out["acc"] = _fit_and_score(x_tr, y_tr, x_score, y_tr if score_train else y_te, k, seed)[0]
    rng = random.Random(seed)
    perm_rows = []
    for _ in range(max(int(n_shuffle), 1)):
        shuffled = list(ytr_raw)
        rng.shuffle(shuffled)
        perm_rows.append([idx[c] for c in shuffled])
    y_perm = torch.tensor(perm_rows, dtype=torch.long)
    # the band is scored on the TRUE test labels; the mutation scores each model on the PERMUTED
    # labels it was fitted to, which is exactly what makes the band collapse toward 1.0
    band_target = y_perm if score_train else y_te.expand(y_perm.shape[0], -1)
    draws = _fit_and_score(x_tr, y_perm, x_score, band_target, k, seed)
    out["shuffle_median"] = sorted(draws)[len(draws) // 2]
    out["shuffle_q95"] = _q95(draws)
    out["readable"] = True
    return out


def decode_heldout(X, y, seed, n_shuffle=N_SHUFFLE):
    """A multinomial logistic decoder (torch) trained on the FIRST half of the rows in tick order
    and scored on the SECOND -- prequential halves, no lookahead, standardized on the train half.

    Returns {n_train, n_test, k, acc, shuffle_q95, shuffle_median, readable, refusal}. The band is
    `n_shuffle` decoders trained on label-permuted train halves and scored on the TRUE test labels,
    fitted as ONE batched (n_shuffle, d, k) tensor. NOT readable when the held-out half is below
    `N_TEST_MIN` rows or either half carries fewer than two classes. `k` and `refusal` are beyond
    2.1's listed keys and are reported, never barred."""
    return _decode(X, y, seed, n_shuffle=n_shuffle, score_train=False)


def control_state(brain):
    """A4's candidate `u_t`: the 15 fields of `U_FIELDS`, in that order, read from the live signals
    of the pre-registration's 1.2 and from NOTHING else.

      gate(4)            `brain._learned_attn.gate`         LearnedAttention.gate, :181-184
      precision(2)       `_pc_surprise` and `pc_dev`        brain.py:2406-2408 and :2482-2485's form
      onehot(focus)(5)   `brain._attn_focus`                brain.py:1571; the fifth slot is NONE,
                                                            reachable while it is still the :1049 init
      focus_strength     the LIF winner's rate              brain.py:1571
      ignited            `_access_state["ignited"]`         brain.py:3154-3158
      drive_margin       `brain.brain.arm._decision_margin` autonomous_loop.py:1219
      sustain            the workspace snapshot's strength  report_channel.snapshot :98-111

    CONTENT-BLIND BY TYPE: a pure function of those fields, so two brains whose control fields are
    equal and whose latents differ read a bit-identical vector (R5). Every field is read through
    `getattr` with a declared default, which is what lets R5 run on a duck-typed stand-in instead
    of constructing a brain."""
    la = getattr(brain, "_learned_attn", None)
    raw_gate = list(getattr(la, "gate", None) or []) if la is not None else []
    gate = [float(g) for g in raw_gate[:N_BANDS]]
    gate += [0.0] * (N_BANDS - len(gate))
    raw = float(getattr(brain, "_pc_surprise", 0.0) or 0.0)
    ema = float(getattr(brain, "_pc_surprise_ema", 0.0) or 0.0)
    pc_dev = min(1.0, max(0.0, raw - ema) / (ema + 1e-6)) if ema > 0.0 else 0.0
    focus = getattr(brain, "_attn_focus", None)
    onehot = [0.0] * (N_BANDS + 1)
    strength = 0.0
    if focus is None:
        onehot[N_BANDS] = 1.0
    else:
        band, strength = int(focus[0]), float(focus[1])
        onehot[band if 0 <= band < N_BANDS else N_BANDS] = 1.0
    ignited = 1.0 if (getattr(brain, "_access_state", None) or {}).get("ignited") else 0.0
    arm = getattr(getattr(brain, "brain", None), "arm", None)
    margin = float(getattr(arm, "_decision_margin", 0.0) or 0.0)
    sustain = float(_report_snapshot(getattr(brain, "workspace", None))["strength"])
    u = gate + [raw, pc_dev] + onehot + [strength, ignited, margin, sustain]
    assert len(u) == U_DIM, f"u_t must carry exactly {U_DIM} fields, got {len(u)}"
    return u


def _g4_reading(g4a, g4b):
    """2.3's branch map, shared by `g4_rule` and by R6's mutated twin so that the mutation changes
    ONLY the operands of (g4b) and nothing else."""
    if not g4a.met:
        return "PASS"
    return "PASS-WITH-LEAK" if g4b.met else "VOID"


def g4_rule(u_reads, c_reads):
    """2.3's G4 branches from the per-seed decodes. ONE implementation, read by R6 and by the
    campaign alike (P2).

      (g4a) t(u_acc - u_shuffle_q95) > the bar  -- the 15-d `u_t` decode above its own band
      (g4b) t(c15_acc - u_acc)       > the bar  -- the MATCHED 15-d content decode above it
      VOID           iff (g4a) MET and (g4b) NOT MET -- content-blindness is a naming convention
      PASS           iff (g4a) NOT MET           -- the vector decodes nothing above chance
      PASS-WITH-LEAK iff (g4a) MET and (g4b) MET -- it leaks, far below the content"""
    a = contrast("(g4a) u_t decodes content", [u["acc"] - u["shuffle_q95"] for u in u_reads])
    b = contrast(
        "(g4b) content decodes better",
        [c["acc"] - u["acc"] for c, u in zip(c_reads, u_reads, strict=True)],
    )
    return {"g4a": a, "g4b": b, "reading": _g4_reading(a, b)}


# ============================== 2.1 / 2.2: the recorder and the campaign =========================
# A SCRATCH brain per seed under a throwaway NORD_HOME (the live lake is never touched and the
# daemon's brain is never constructed), N_TICKS ticks, the attention state captured by wrapping the
# INSTANCE's `_learned_attn.attend` so both what it is FED and what it RETURNS are recorded -- the
# shape `metacognitive_monitor.t077_record_brain` uses for the monitor's `update`, reused rather
# than reinvented (2.1's P2 list). Seeds run sequentially in ONE process because `throwaway_home()`
# sets a process-wide env var and is not thread-safe (2.8). The pin is written ONCE.
# distinct-from: cortex/metacognitive_monitor.py::t077_record_brain -- records the METACOGNITIVE
#   monitor's five inputs and its unrounded read, and carries a LESIONED arm. This records the
#   ATTENTION control state, the committed action and the frame label, and Part I has NO lesion
#   and NO second arm at all: one recording, two gates read off it.
# distinct-from: substrate/amdahl_gate.py::live_reading -- a TIMING ceiling over three brains with
#   a neutered code path. No timing is read here and no code path is neutered.
def t078_record_brain(seed, n_ticks=T078_N_TICKS):
    """Tick a SCRATCH brain `n_ticks` times and return the per-tick rows of 2.1.

    `_learned_attn.attend` is wrapped on the INSTANCE, so every call's input latent, gated output,
    gate and focus are captured without touching brain.py. DISCLOSED, because the brain calls it
    up to TWICE per tick (`_attend_real_stream` for vision then for audio, brain.py:1603-1608) and
    the two halves of the brain's own record then disagree: `_attn_real` is `va or aa`, the FIRST
    successful call, while `_attn_focus`, `la.gate` and `la.focus` are all last-call state. This
    recorder takes the LAST call's latent/gated/gate/focus, so the recorded vector is consistent
    with `control_state`'s own reads, records `attend_calls` per tick, and records `chan` BOTH from
    `_attn_real` (1.2's arm sensory channel) and as the last call's four band means, so the
    campaign can COUNT the ticks on which the two differ instead of adjudicating between them.

    DISCLOSED, 2026-09-15 (the panel fold), because the staleness is the RECORDER's and not the
    packer's: `cap` is built ONCE above the tick loop and is never reset, so a tick on which
    `attend` is not called keeps the LAST attending tick's capture. Exactly three recorded fields
    are tainted by that -- `gated`, `gate` and `chan_from_gated` (which is derived from `gated`).
    Measured on the committed pin: on every post-first-attend asleep row of all eight seeds, `gate`
    is identical to the previous row's (604/604, 607/607, 602/602, 608/608, 611/611, 602/602,
    602/602, 611/611). `focus_band`, `focus_strength` and `chan` are NOT tainted -- they read the
    brain's `_attn_focus` and `_attn_real` directly, and `chan`'s `from_gated` FALLBACK was taken on
    0 of those rows on every seed -- and `cap["latent"]` / `cap["focus"]` are captured but never
    consumed into a row at all. The LIVE RISK this leaves open: `t078_g4` filters on a truthy
    `gated` and `t078_pairs` filters nothing, so an awake tick with `attend_calls == 0` arriving
    AFTER the first attend would enter a G3 pair and a G4 row on the previous tick's latent -- the
    stale pairing T077 3.2's amendment forbids, and no gate catches it today. It did not arise on
    the pinned recording: the one zero-attend awake tick per seed is the tick immediately BEFORE
    that seed's first attend, so it is outside the npz and its `gate` is already the empty vector.

    The frame LABEL is recovered as `_real_labels[(_real_ptr - 1) % n]` on the ticks where
    `_real_ptr` ADVANCED (1.2), and is None otherwise -- never carried forward from a past tick.
    Returns rows only; the brain is dropped so eight seeds do not stay alive at once."""
    from snn_data_lake.cortex.brain import make_brain

    b = make_brain(seed=seed, persist=False)
    la = getattr(b, "_learned_attn", None)
    cap = {"calls": 0, "latent": None, "gated": None, "gate": None, "focus": None}
    if la is not None:
        real_attend = la.attend

        def wrapped(features, goal=None):
            gated = real_attend(features, goal=goal)
            cap["calls"] += 1
            cap["latent"] = [float(v) for v in features]
            cap["gated"] = [float(v) for v in gated]
            cap["gate"] = [float(g) for g in la.gate]
            cap["focus"] = la.focus
            return gated

        la.attend = wrapped
    rows = []
    prev_ptr = int(getattr(b, "_real_ptr", 0) or 0)
    for t in range(n_ticks):
        before = cap["calls"]
        kind = b.tick()
        res = b._last_res or {}
        ptr = int(getattr(b, "_real_ptr", 0) or 0)
        labels = getattr(b, "_real_labels", None)
        label = None
        if labels is not None and len(labels) and ptr != prev_ptr:
            label = int(labels[(ptr - 1) % len(labels)])
        prev_ptr = ptr
        calls = cap["calls"] - before
        gated = list(cap["gated"] or [])
        span = max(1, len(gated) // N_BANDS) if gated else 1
        from_gated = [
            min(1.0, max(0.0, sum(gated[i * span : (i + 1) * span]) / span)) for i in range(N_BANDS)
        ]
        attn = getattr(b, "_attn_real", None) or {}
        chan = [float(v) for v in (attn.get("channel") or from_gated)]
        focus = getattr(b, "_attn_focus", None)
        u = control_state(b)
        rows.append(
            {
                "t": t,
                "kind": kind,
                "awake": kind == "awake",
                "attend_calls": calls,
                "u": [round(v, 6) for v in u],
                "gate": [round(v, 6) for v in (cap["gate"] or [0.0] * N_BANDS)],
                "focus_band": None if focus is None else int(focus[0]),
                "focus_strength": 0.0 if focus is None else round(float(focus[1]), 6),
                "ignited": bool((getattr(b, "_access_state", None) or {}).get("ignited", False)),
                "sustain": round(
                    float(_report_snapshot(getattr(b, "workspace", None))["strength"]), 6
                ),
                "drive_margin": round(
                    float(getattr(getattr(b.brain, "arm", None), "_decision_margin", 0.0) or 0.0), 6
                ),
                "pc_surprise": round(float(getattr(b, "_pc_surprise", 0.0) or 0.0), 6),
                "action": None if res.get("action") is None else str(res.get("action")),
                "digit": getattr(b, "_perceived_digit", None),
                "label": label,
                "chan": [round(v, 6) for v in chan],
                "chan_from_gated": [round(v, 6) for v in from_gated],
                # the 64-d gated latent: the ONE field the pin does not carry (it goes to the
                # companion .npz, float32 per seed, sha256 in the pin -- 2.1)
                "gated": [round(v, 4) for v in gated],
            }
        )
    return rows


def t078_pairs(rows):
    """2.2's G3 pairs: CONSECUTIVE awake ticks (t, t+1), and nothing else.

    Two drops, both COUNTED rather than silently skipped (T077 3.2's no-stale-pairing amendment):
    a link from one awake tick to the next awake tick that is NOT adjacent has a sleep bout between
    them and is dropped; a pair either of whose ticks committed EMERGENCY_STOP is dropped (2.2
    records that it never occurs under `_safe_ctx`, so a nonzero count here is itself a finding).
    Returns {pairs, dropped_sleep, dropped_estop, n_awake}."""
    awake = [i for i, r in enumerate(rows) if r["awake"]]
    pairs, dropped_sleep, dropped_estop = [], 0, 0
    for i, j in zip(awake, awake[1:], strict=False):
        if j != i + 1:
            dropped_sleep += 1
            continue
        if rows[i]["action"] == ESTOP_LABEL or rows[j]["action"] == ESTOP_LABEL:
            dropped_estop += 1
            continue
        pairs.append((rows[i], rows[j]))
    return {
        "pairs": pairs,
        "dropped_sleep": dropped_sleep,
        "dropped_estop": dropped_estop,
        "n_awake": len(awake),
    }


def _gate16(pairs):
    """2.2's REPORTED gate16 code: the four gate values, each split at ITS OWN seed median over the
    pairs of this seed, packed into one of 16 cells. A per-seed median rather than a fixed
    threshold, so no magic constant enters (P1); it is reported, never barred."""
    if not pairs:
        return []
    meds = [statistics.median([a["gate"][b] for a, _ in pairs]) for b in range(N_BANDS)]
    return [sum(1 << b for b in range(N_BANDS) if a["gate"][b] > meds[b]) for a, _ in pairs]


def t078_g3(rows, seed, n_perm=N_PERM):
    """2.2's four G3 readings, ALL over the same pairs (2.2: "the same pairs serve every G3
    reading"). Returns {primary, control, gate16, same_tick, n_pairs, cells, dropped_*}.

      primary  BARRED   MI(focus_t ; action_t+1)     -- the gate itself
      control  BARRED   MI(action_t ; action_t+1)    -- the estimator's positive control; a failure
                                                        here REFUSES G3, it never STOPS it
      gate16   reported MI(gate16_t ; action_t+1)
      same_tick reported MI(focus_t ; action_t)      -- separates prediction from co-occurrence"""
    p = t078_pairs(rows)
    pairs = p["pairs"]
    x_focus = [a["focus_band"] for a, _ in pairs]
    y_next = [b["action"] for _, b in pairs]
    x_action = [a["action"] for a, _ in pairs]
    primary = mi_read(x_focus, y_next, n_perm=n_perm, seed=seed)
    return {
        "n_pairs": len(pairs),
        "dropped_sleep": p["dropped_sleep"],
        "dropped_estop": p["dropped_estop"],
        "n_awake": p["n_awake"],
        "primary": primary,
        # 2.10: the CONTROL's readability is per action CLASS (its 12 x 12 joint would refuse at
        # every pair count N = 1,200 can produce under the per-cell rule); the primary keeps 2.1's
        "control": mi_read(x_action, y_next, n_perm=n_perm, seed=seed, min_rule="per_class"),
        "gate16": mi_read(_gate16(pairs), y_next, n_perm=n_perm, seed=seed),
        "same_tick": mi_read(x_focus, x_action, n_perm=n_perm, seed=seed),
        "cells": primary["cells"],
        "distinct_actions": len({a for a in y_next if a is not None}),
    }


def t078_projection(seed):
    """2.1's fixed Gaussian random projection of the 64-d gated latent down to the MATCHED 15
    inputs, seeded `crc32(b"t078-proj") + seed` -- `zlib.crc32`, never the builtin `hash()`, so it
    is stable across processes. A (LATENT_DIM, U_DIM) tensor; the capacity match is what makes
    (g4b) a comparison of INFORMATION rather than of input width."""
    import torch

    g = torch.Generator().manual_seed(int(zlib.crc32(PROJ_TAG)) + int(seed))
    return torch.randn(LATENT_DIM, U_DIM, generator=g)


def t078_g4(rows, seed, n_shuffle=N_SHUFFLE):
    """2.2's four G4 decodes, over the awake ticks on which a frame was PERCEIVED (the label
    exists). Target = the frame's LABEL; the perceived digit is a second target, reported.

      u        BARRED (g4a)  the 15-d `u_t`
      c15      BARRED (g4b)  the gated 64-d latent projected to 15 -- the MATCHED arm
      raw64    reported      the raw gated latent: the CEILING, never barred (2.1)
      chan     reported      the arm's 4-d sensory channel
    Every decode carries its own `n_shuffle`-draw label-shuffle band."""
    import torch

    seen = [r for r in rows if r["awake"] and r["label"] is not None and r["gated"]]
    y = [r["label"] for r in seen]
    digits = [r["digit"] for r in seen]
    u_rows = [r["u"] for r in seen]
    raw = [r["gated"] for r in seen]
    chan = [r["chan"] for r in seen]
    proj = t078_projection(seed)
    c15 = (torch.tensor(raw, dtype=torch.float32) @ proj).tolist() if raw else []
    out = {
        "n_rows": len(seen),
        "n_classes": len(set(y)),
        "u": decode_heldout(u_rows, y, seed=seed, n_shuffle=n_shuffle),
        "c15": decode_heldout(c15, y, seed=seed, n_shuffle=n_shuffle),
        "raw64": decode_heldout(raw, y, seed=seed, n_shuffle=n_shuffle),
        "chan": decode_heldout(chan, y, seed=seed, n_shuffle=n_shuffle),
        "u_digit": decode_heldout(u_rows, digits, seed=seed, n_shuffle=n_shuffle),
    }
    return out


def t078_pack_latents(rows):
    """The companion .npz's per-seed arrays: the 64-d gated latents of the ticks that CARRY one, as
    a (m, LATENT_DIM) float32 array, beside the (m,) int32 array of their tick indices -- so the npz
    re-joins the pin's rows by `t`. `seed{s}_ticks` is therefore "the ticks that carry a latent",
    NOT "the ticks that attended".

    THE MECHANISM, corrected 2026-09-15 in the panel fold and MEASURED on the committed pin rather
    than argued. An asleep tick does NOT record an empty `gated`: `t078_record_brain`'s `cap` dict
    is built once before the tick loop and is never reset, so a tick on which `attend` is not called
    keeps the LAST attending tick's `gated` and `gate`. The npz proves it -- per seed it keeps
    1,191-1,199 of 1,200 ticks, and the count of consecutive byte-identical latent rows equals the
    ASLEEP count exactly on every seed while the number of DISTINCT rows equals the AWAKE count
    exactly. So about half of each seed's packed rows (50.2%-51.0%, 50.6% over all 9,577) is one
    byte-duplicate per asleep tick. The ONLY rows carrying an empty `gated` are those before the
    FIRST `attend` call of a run, and those are exactly the ticks missing from the npz on all eight
    seeds -- they are the rows the first dry run's ragged `np.asarray` tripped on (2026-09-15 14:48,
    200 ticks; the builder's constructed-recording test had given every row a latent, asleep ones
    included, so it never saw the empty ones -- 2.9 step 5's red, added after)."""
    import numpy as np

    kept = [(r["t"], r["gated"]) for r in rows if r.get("gated")]
    if not kept:
        return np.zeros((0, LATENT_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return (
        np.asarray([g for _, g in kept], dtype=np.float32),
        np.asarray([t for t, _ in kept], dtype=np.int32),
    )


def t078_determinism(seed, n_ticks=DETERMINISM_TICKS):
    """2.2's DETERMINISM PRECONDITION (G5 on this loop, cheap): two `n_ticks` recordings of the
    same seed must be byte-identical on (focus band, gate, action, perceived digit, label). A
    difference REFUSES the campaign -- nothing below is readable on a non-deterministic recording,
    and that is the INSTRUMENT's fault, not a finding about the loop.

    SCOPE OF THE KEY, disclosed here so the next reader of this symbol cannot mistake it for the
    prose above. `t078_record_brain` emits 18 fields per row; this key compares FIVE, which is
    2.2's frozen wording and is NOT changed here. Those five pin `u_t`'s `gate0`-`gate3` directly
    and its five `focus_is_*` slots derivatively (`control_state` and the row read the same
    `_attn_focus`), leaving `pc_surprise`, `pc_dev`, `focus_strength`, `ignited`, `drive_margin`
    and `sustain`, plus `awake`/`kind` -- which `t078_pairs` consumes for EVERY G3 pair and for
    its sleep-bout drop count -- and `attend_calls`, `t`, `chan`, `chan_from_gated` and the 64-d
    `gated` latent, i.e. none of the G4 content arm, outside it. MEASURED, not argued: a
    constructed recording whose un-keyed fields are all corrupted is key-identical to its clean
    twin while the full rows differ, and corrupting any ONE keyed field fires (p) on all five --
    so (p) has a reachable red and what is narrower than its stated warrant is its SCOPE.
    "(p) MET" in the record therefore certifies the five, not the eighteen. The successor that
    re-runs G3 pre-registers the key over the FULL recorded row and writes the compared field list
    beside `reading.determinism` in its own pin (the fold's opening in 3.6)."""
    key = lambda r: (  # noqa: E731 - one line, used twice, never exported
        r["focus_band"],
        tuple(r["gate"]),
        r["action"],
        r["digit"],
        r["label"],
    )
    a = [key(r) for r in t078_record_brain(seed, n_ticks)]
    b = [key(r) for r in t078_record_brain(seed, n_ticks)]
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    return {"identical": a == b, "n": len(a), "first_diff": first, "seed": seed}


# ============================== 2.4's constructed inputs (no brain, no lake) ======================
# Every red below runs on one of these. They live in the MODULE rather than in the test because
# `t078_reds()` is the in-command control the module's own self-check asserts, and a constructed
# input that exists only under pytest is unavailable to a gate run -- which is why T077's
# `t077_tracker_stream` sits in `metacognitive_monitor` for the same reason.
# distinct-from: cortex/live_brain.py::LiveBrain.run -- drives the REAL brain over a stream of
#   context dicts and returns live states. Read: these constructors tick nothing and construct no
#   brain; `t078_stub_brain` is a duck-typed namespace carrying the seven attributes
#   `control_state` reads, so R5 runs with no Brain, no lake and no torch.
# distinct-from: cortex/self_model.py::SelfModel.observe -- folds one tick of REAL brain signals
#   into an EMA-tracked model. Read: `control_state` keeps no state, folds nothing and returns a
#   pure function of the fields it is handed on that tick.
# distinct-from: substrate/amdahl_gate.py::_count_lif_in -- a bounded attribute walk counting
#   LIFNeurons reachable from one brain. Read: shares only the phrase "one brain"; no census here.
# distinct-from: cortex/snn_moe_classifier.py::_new_rng -- derives a numpy Generator's seed from
#   the module-global `random` stream to keep a consumer's determinism contract. Read: every RNG
#   here is a per-call `random.Random(seed)` or a per-call `torch.Generator`, and NOTHING in this
#   module reads or writes a process-global random stream.
# distinct-from: cerebellum/wavefront_planner.py::_limits -- per-joint actuator ranges from the
#   arm model. Read; unrelated to anything in this file.
def t078_label_stream(n, k, seed):
    """n i.i.d. labels drawn uniformly from range(k) by a seeded `random.Random`."""
    rng = random.Random(seed)
    return [rng.randrange(k) for _ in range(n)]


def t078_markov_stream(n, k, p_self, seed):
    """n labels from a k-state chain that repeats the previous label with probability `p_self` and
    otherwise jumps uniformly to one of the other k - 1 states -- autocorrelated BY CONSTRUCTION,
    which is the property R2's positive control has to be able to read."""
    rng = random.Random(seed)
    out = [rng.randrange(k)]
    for _ in range(n - 1):
        if rng.random() < p_self:
            out.append(out[-1])
        else:
            out.append((out[-1] + 1 + rng.randrange(k - 1)) % k)
    return out


def t078_cell_stream(n_pairs, n_cells, n_x=N_BANDS, n_y=12):
    """An (x, y) pair stream occupying EXACTLY `n_cells` observed joint cells, by cycling the first
    `n_cells` of the n_x by n_y grid. R3's shape: the cell count is pinned and the PAIR count is
    the only thing that moves, so the refusal is attributable to the ratio and to nothing else."""
    grid = [(i, j) for i in range(n_x) for j in range(n_y)][:n_cells]
    pairs = [grid[t % len(grid)] for t in range(n_pairs)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def t078_planted_rows(n, d, k, seed, planted=True):
    """n rows of d uniform-noise features with the label planted in coordinate 0 as y / (k - 1) --
    an ORDERED one-dimensional separation a linear multinomial decoder can reach. With
    planted=False the labels are drawn independently of the rows, so nothing predicts them."""
    rng = random.Random(seed)
    y = [rng.randrange(k) for _ in range(n)]
    rows = [[rng.random() for _ in range(d)] for _ in range(n)]
    if planted:
        for i in range(n):
            rows[i][0] = y[i] / float(k - 1)
    return rows, y


def t078_g4_recording(seed, kind, n=200, k=10):
    """A constructed (u_rows, content_rows, labels) recording for R6.

    "void": `u_t` IS a one-hot of the label (a near-perfect decode) while the content is a LOSSY
    one-hot of label // 2 (five distinguishable groups), so (g4a) is MET and (g4b) -- content minus
    u_t -- is NOT: the VOID branch. Deliberately NOT a tie, so that swapping (g4b)'s operands
    (2.4's mutation) FLIPS the reading instead of leaving a degenerate contrast degenerate.
    "pass": `u_t` is pure noise, so (g4a) is NOT MET: the PASS branch."""
    rng = random.Random(seed)
    y = [rng.randrange(k) for _ in range(n)]
    u_rows, c_rows = [], []
    for i in range(n):
        if kind == "void":
            u = [rng.random() * 1e-3 for _ in range(U_DIM)]
            u[y[i] % U_DIM] += 1.0
        else:
            u = [rng.random() for _ in range(U_DIM)]
        c = [rng.random() * 1e-3 for _ in range(U_DIM)]
        c[(y[i] // 2) % U_DIM] += 1.0
        u_rows.append(u)
        c_rows.append(c)
    return u_rows, c_rows, y


def t078_stub_brain(gate, latent_seed, pc=0.4, ema=0.2, focus=(2, 0.05), ignited=False, margin=1.0):
    """The attribute shape `control_state` reads, and nothing else: a duck-typed stand-in so R5
    runs without constructing a brain. Its `_real_percept` / `_attn_real` carry CONTENT that varies
    with `latent_seed` and that `control_state` must IGNORE -- which is the whole of R5."""
    import types

    rng = random.Random(latent_seed)
    latent = [rng.random() for _ in range(LATENT_DIM)]
    span = max(1, LATENT_DIM // N_BANDS)
    chan = [sum(latent[b * span : (b + 1) * span]) / span for b in range(N_BANDS)]
    return types.SimpleNamespace(
        _learned_attn=types.SimpleNamespace(gate=list(gate), focus=focus),
        _attn_focus=focus,
        _pc_surprise=pc,
        _pc_surprise_ema=ema,
        _access_state={"ignited": ignited},
        _real_percept=latent,
        _attn_real={"channel": chan, "band": focus[0], "focus": focus[1]},
        workspace=types.SimpleNamespace(broadcast=lambda: "digit-3", broadcast_hold=0.7),
        brain=types.SimpleNamespace(arm=types.SimpleNamespace(_decision_margin=margin)),
    )


def _t078_mutated_scorer(kind):
    """2.4's six mutations, RUN and not reasoned (P8: a reachable red; T077 3.5 item 10). Each is
    the deliberately broken twin of the instrument whose red it must redden, and each is exercised
    BOTH in `tests/test_attention_schema.py` and in `t078_reds()` below.

      floor_unpermuted (R1) -- the floor is drawn on UN-permuted y, so every draw equals the
          reading: the i.i.d. stream then reads exactly AT its "floor" and the deterministic
          control's margin over it vanishes.
      self_score       (R2) -- the estimator scores y against ITSELF, so the i.i.d. action stream
          reads its own entropy instead of reading at its floor: the control becomes an identity,
          which is the defect that withdrew T077's oracle.
      no_min           (R3) -- N_MIN_PER_CELL := 0, so the probe's own 145-pairs-over-39-cells
          shape stops refusing and a number appears where the pre-registration says none can.
      train_scored     (R4) -- every decoder is scored on the half it was FITTED on, the band's
          models on their own permuted labels, so the band collapses toward 1.0.
      content_leak     (R5) -- chan[0] is appended to `u_t`, so the vector carries a coordinate of
          the attended content: the two brains stop being bit-identical and the length guard trips.
      g4_swap          (R6) -- (g4b)'s operands are swapped, so the VOID construction reads
          PASS-WITH-LEAK instead.
      blind_min_rule   (R3, 2.10's TWIN) -- the scorer DROPS `min_rule` and reads everything under
          the per-CELL floor, so the twin (the same 145 pairs read as a 12-class CONTROL) refuses
          and R3's `ctrl_ok` conjunct fails. ADDED IN THE PANEL FOLD, 2026-09-15, and disclosed as
          post-number: 2.10's twin conjunct landed in the SAME commit as the `min_rule`
          implementation (455236d), so it was never in the tree in a failing state, and none of the
          six mutations above reddens it -- `no_min`, `floor_unpermuted` and `self_score` all leave
          the twin readable. It is an ADDITIONAL conjunct of R3, never a substitution for `ctrl_ok`
          (substituting takes R3 False). R3's boolean is unchanged."""

    def floor_unpermuted(x, y, **kw):
        r = mi_read(x, y, **kw)
        if r["readable"]:
            r["perm_q95"] = r["mi"]
            r["perm_median"] = r["mi"]
        return r

    def self_score(x, y, **kw):
        return mi_read(y, y, **kw)

    def no_min(x, y, **kw):
        return _mi_read(x, y, n_min_per_cell=0, **kw)

    # distinct-from: cortex/laminar_local_readout.py::observe -- one LOCAL WEIGHT UPDATE on a
    #   sample. distinct-from: plasticity/eligibility_algebra.py::canonical -- like-term folding of
    #   a plasticity RULE expression. distinct-from: plasticity/plasticity_rule_stack.py
    #   ::_equivalence_and_determinism_selfcheck -- a stack drop-in equivalence check.
    # distinct-from: reasoning/arc_mint.py::induce_rule -- MDL induction of an attribute->colour
    #   map. distinct-from: reasoning/raven_bench.py::_synth_task -- NVSA rule sampling for an RPM.
    #   All five READ; every one is a LEARNING or INDUCTION rule. The twin below learns nothing and
    #   induces nothing: it is 2.4's catalog shape, a deliberately broken `mi_read` that DISCARDS
    #   its `min_rule` argument so 2.10's per-class readability twin can be reddened in fact.
    def blind_min_rule(x, y, **kw):
        kw.pop("min_rule", None)  # the per-CLASS rule is ignored; everything reads per-CELL
        return mi_read(x, y, **kw)

    def train_scored(X, y, seed, n_shuffle=N_SHUFFLE):
        return _decode(X, y, seed, n_shuffle=n_shuffle, score_train=True)

    def content_leak(brain):
        chan = (getattr(brain, "_attn_real", None) or {}).get("channel") or [0.0] * N_BANDS
        return control_state(brain) + [float(chan[0])]

    def g4_swap(u_reads, c_reads):
        a = contrast("(g4a) u_t decodes content", [u["acc"] - u["shuffle_q95"] for u in u_reads])
        b = contrast(
            "(g4b) SWAPPED operands",
            [u["acc"] - c["acc"] for c, u in zip(c_reads, u_reads, strict=True)],
        )
        return {"g4a": a, "g4b": b, "reading": _g4_reading(a, b)}

    return {
        "floor_unpermuted": floor_unpermuted,
        "self_score": self_score,
        "no_min": no_min,
        "train_scored": train_scored,
        "content_leak": content_leak,
        "g4_swap": g4_swap,
        "blind_min_rule": blind_min_rule,
    }[kind]


def t078_reds(n_perm=REDS_N_PERM, n_shuffle=N_SHUFFLE, seeds=T078_SEEDS):
    """2.4's six reds as ONE self-check line (each is ALSO a test in
    `tests/test_attention_schema.py`; this is the in-command control 2.4 requires). Every value is
    True iff the red holds on its constructed input AND its mutated scorer REDDENS it in fact --
    a red a broken instrument still passes is not a red (T077's panel, C1)."""
    out = {}

    # R1 -- the floor absorbs the plug-in bias; the mutation draws it on UN-permuted y
    x = t078_label_stream(600, N_BANDS, seed=1)
    y = t078_label_stream(600, 12, seed=2)
    iid = mi_read(x, y, n_perm=n_perm, seed=0)
    mapped = [(v * 3) % 12 for v in x]
    ctrl = mi_read(x, mapped, n_perm=n_perm, seed=0)
    m1 = _t078_mutated_scorer("floor_unpermuted")(x, mapped, n_perm=n_perm, seed=0)
    out["R1"] = bool(
        iid["readable"]
        and iid["mi"] == mi_bits(x, y)
        and iid["mi"] <= iid["perm_q95"]
        and ctrl["readable"]
        and ctrl["mi"] - ctrl["perm_q95"] > 1.0
        and not (m1["mi"] - m1["perm_q95"] > 1.0)
    )

    # R2 -- the positive control is not an identity; the mutation scores y against itself
    a_iid = t078_label_stream(REDS_ACTION_N, REDS_ACTION_K, seed=3)
    r_iid = mi_read(a_iid[:-1], a_iid[1:], n_perm=n_perm, seed=0)
    a_mk = t078_markov_stream(REDS_ACTION_N, REDS_ACTION_K, 0.7, seed=4)
    r_mk = mi_read(a_mk[:-1], a_mk[1:], n_perm=n_perm, seed=0)
    m2 = _t078_mutated_scorer("self_score")(a_iid[:-1], a_iid[1:], n_perm=n_perm, seed=0)
    out["R2"] = bool(
        r_iid["readable"]
        and r_iid["mi"] <= r_iid["perm_q95"]
        and r_mk["readable"]
        and r_mk["mi"] > r_mk["perm_q95"]
        and not (m2["readable"] and m2["mi"] <= m2["perm_q95"])
    )

    # R3 -- the refusal fires on the probe's own shape, AND 2.10's per-CLASS twin reads the same
    #       145 pairs; TWO mutations, one per half: N_MIN_PER_CELL := 0 reddens the per-cell
    #       refusal, a `min_rule`-blind scorer reddens the twin
    cx, cy = t078_cell_stream(145, 39)
    refused = mi_read(cx, cy, n_perm=n_perm, seed=0)
    ox, oy = t078_cell_stream(600, 39)
    readable = mi_read(ox, oy, n_perm=n_perm, seed=0)
    one_class = mi_read([0] * 600, oy, n_perm=n_perm, seed=0)
    m3 = _t078_mutated_scorer("no_min")(cx, cy, n_perm=n_perm, seed=0)
    # 2.10's twin: the same 145 pairs read as a 12-class CONTROL are readable under the per-class rule
    ctrl_ok = mi_read(cx, cy, n_perm=n_perm, seed=0, min_rule="per_class")
    # ... and its OWN mutation, added in the panel fold: a scorer blind to `min_rule` reads the same
    # 145 pairs under the per-CELL floor and REFUSES, so the `ctrl_ok` conjunct is reddened in fact.
    # An ADDITIONAL conjunct, never a substitution -- substituting the blind read takes R3 False.
    m3b = _t078_mutated_scorer("blind_min_rule")(
        cx, cy, n_perm=n_perm, seed=0, min_rule="per_class"
    )
    out["R3"] = bool(
        refused["readable"] is False
        and refused["mi"] is None
        and refused["refusal"]
        and readable["readable"] is True
        and readable["cells"] == 39
        and one_class["readable"] is False
        and ctrl_ok["readable"] is True
        and m3["readable"] is True
        and not m3b["readable"]  # the mutation must break `ctrl_ok["readable"] is True`
    )

    # R4 -- the decoder's band; the mutation scores every model on its own TRAIN half
    rows, ry = t078_planted_rows(600, U_DIM, 10, seed=5, planted=True)
    planted = decode_heldout(rows, ry, seed=0, n_shuffle=n_shuffle)
    flat_rows, flat_y = t078_planted_rows(600, U_DIM, 10, seed=6, planted=False)
    unplanted = decode_heldout(flat_rows, flat_y, seed=0, n_shuffle=n_shuffle)
    m4 = _t078_mutated_scorer("train_scored")(rows, ry, 0, n_shuffle=n_shuffle)
    out["R4"] = bool(
        planted["readable"]
        and planted["n_test"] == 300
        and planted["shuffle_q95"] < 0.2
        and planted["acc"] > planted["shuffle_q95"]
        and unplanted["readable"]
        and unplanted["acc"] <= unplanted["shuffle_q95"]
        and not (m4["shuffle_q95"] < 0.2)
    )

    # R5 -- content-blind by TYPE plus the vacuity clause; the mutation appends chan[0]
    same_gate = (0.11, 0.12, 0.13, 0.14)
    b1 = t078_stub_brain(gate=same_gate, latent_seed=1)
    b2 = t078_stub_brain(gate=same_gate, latent_seed=2)
    b3 = t078_stub_brain(gate=(0.90, 0.02, 0.03, 0.04), latent_seed=1)
    u1, u2, u3 = control_state(b1), control_state(b2), control_state(b3)
    leak = _t078_mutated_scorer("content_leak")
    out["R5"] = bool(
        len(u1) == U_DIM
        and not (set(U_FIELDS) & set(U_FORBIDDEN))
        and u1 == u2
        and u1 != u3
        and not (leak(b1) == leak(b2) and len(leak(b1)) == U_DIM)
    )

    # R6 -- the void rule can fire and the pass branch can too; the mutation swaps (g4b)
    void_u, void_c, pass_u, pass_c = [], [], [], []
    for s in seeds:
        u_rows, c_rows, labels = t078_g4_recording(s, "void")
        void_u.append(decode_heldout(u_rows, labels, seed=s, n_shuffle=n_shuffle))
        void_c.append(decode_heldout(c_rows, labels, seed=s, n_shuffle=n_shuffle))
        u_rows, c_rows, labels = t078_g4_recording(s, "pass")
        pass_u.append(decode_heldout(u_rows, labels, seed=s, n_shuffle=n_shuffle))
        pass_c.append(decode_heldout(c_rows, labels, seed=s, n_shuffle=n_shuffle))
    void = g4_rule(void_u, void_c)
    swapped = _t078_mutated_scorer("g4_swap")(void_u, void_c)
    noise = g4_rule(pass_u, pass_c)
    out["R6"] = bool(
        all(r["readable"] for r in void_u + void_c + pass_u + pass_c)
        and void["reading"] == "VOID"
        and noise["reading"] == "PASS"
        and swapped["reading"] != "VOID"
    )
    return out


def _line(c):
    """One contrast as an ASCII line: its verdict, its t and its EXACT sign-flip FPR (PLAN-FINAL
    Part F's preamble requires the FPR printed beside every contrast)."""
    verdict = "MET" if c.met else "NOT MET"
    deg = " DEGENERATE" if c.degenerate else ""
    return f"{c.name}: {verdict}{deg} t={c.t:.4f} FPR={c.fpr}"


# distinct-from: substrate/paired_contrast.py::contrast -- decides MET/degenerate over a list of
#   differences it is HANDED, and is the function this one calls. It has no notion of a reading, a
#   readability or a refusal, and cannot: by the time diffs reach it the fabrication has happened.
#   This is the guard in front of it -- whether a contrast may be BUILT at all. Read, imported,
#   never re-implemented. distinct-from: hippocampus/online_readout.py::ClosingGate (a refit-count
#   gate that holds beta stale after k refits), substrate/signal_ring.py::publish (a single-writer
#   ring append whose "never drops ITS OWN write" is about slot reuse), cerebellum/hjepa_serve.py::
#   _validate_stack (layer shape chaining, raises) and cerebellum/gain_adaptation.py (online LMS
#   gain recalibration). All four READ: none decides whether a REPORTED statistic may be computed
#   from readings that refused, and none returns a refusal record in place of a number.
def _reported_mi(name, key, seeds, ss):
    """A REPORTED MI contrast that never fabricates a number for a refusal.

    `reported.gate16` and `reported.same_tick` were each built over
    `(mi or 0.0) - (perm_q95 or 0.0)`, which turns a REFUSED reading into a 0.0 DIFF. On the
    landed campaign `gate16` refused on every seed, so the surface carried a contrast of eight
    fabricated zeros; the sharper case is PARTIAL refusal, where one refused seed injects one 0.0
    among real diffs and the contrast is not even degenerate, so it carries no flag at all.
    `mi_read`'s own rule is that a refusal which does not say why is a number without a reason.

    Gated on EVERY seed readable, mirroring 2.9's (r) rule -- never on "no seed readable", and
    never by contrasting the readable remainder, which would silently change the paired sample
    size the bar is pre-registered over. A refused entry carries NO `t`, NO `fpr` and NO `met`, so
    no downstream reader can lift a number that does not exist; it carries the refusing seeds and
    `mi_read`'s own refusal sentence for each, which is the reason the surface owed.

    POST-NUMBER (2026-09-15, the panel fold): `tools/t078_attention_schema.json` was pinned ONCE
    before this shape existed and retains the eight-zero `reported.gate16` contrast. No BARRED
    reading is touched -- gate16 and same_tick are pre-registered "reported, never barred", enter
    no conjunct, and the STOP x PASS verdict is computed without either."""
    refused = {
        str(s): x["g3"][key]["refusal"]
        for s, x in zip(seeds, ss, strict=True)
        if not x["g3"][key]["readable"]
    }
    if refused:
        return {"name": name, "refused": True, "n_refused": len(refused), "refusals": refused}
    return contrast(name, [x["g3"][key]["mi"] - x["g3"][key]["perm_q95"] for x in ss])._asdict()


# distinct-from: cortex/metacognitive_monitor.py::run_t077_campaign -- the PIN-ONCE campaign shape
#   2.1 names for reuse, read before this marker was written. That one records a metacognitive
#   monitor's five inputs over an intact and a LESIONED arm plus a chat arm, and scores a type-2
#   AUROC. This one records the ATTENTION state over ONE arm (Part I has no lesion), scores an MI
#   floor and a decoder band, and decides a STOP/VOID gate. Different instrument, different
#   conjuncts, different pin; the `contrast` they share is IMPORTED from substrate/paired_contrast
#   by both, which is the single mechanism.
# distinct-from: substrate/policy_manifest.py::execute -- runs a gate SUBPROCESS and maps its exit
#   code to CANNOT_RUN / FAIL. Read: no subprocess here, and a REFUSED is decided from a reading,
#   not from an exit code. distinct-from: substrate/gradient_claim_gate.py::_write_measure_artifact
#   -- writes a staleness-checkable measurement artifact. Read: the shared idea is "pin the run",
#   and this pin is written ONCE and refuses to overwrite, which that one deliberately does not do.
# distinct-from: substrate/signal_ring.py::publish (a single-writer ring append) and
#   hippocampus/episodic_recall.py::_flip (a spike-code corruption probe). Both read; neither
#   records a brain, contrasts seeds, or writes a pin.
def run_t078_campaign(seeds=T078_SEEDS, n_ticks=T078_N_TICKS, pin=T078_PIN, latents=T078_LATENTS):
    """Part I's campaign (2.2-2.3): record, read, contrast, pin ONCE. Returns an exit code.

    ORDER IS 2.3's, and it is the order verdicts are DECIDED in, not a presentation choice:
      (p)   DETERMINISM   the twin recordings byte-identical, else REFUSED -- no reading of any kind
      (r)   READABLE      every seed's G3 primary and control readable, every G4 decode readable
      (g3c) THE ESTIMATOR SEES  MI(action_t; action_t+1) above its floor, else G3 REFUSED
      (g3)  TASK BINDING  MI(focus_t; action_t+1) above its floor -> PASS, else STOP
      (g4a) u_t DECODES CONTENT / (g4b) CONTENT DECODES BETTER -> PASS / PASS-WITH-LEAK / VOID
    A REFUSAL is a finding about the INSTRUMENT and a STOP is a finding about the LOOP; the two are
    never merged. The pin carries every per-tick field except the 64-d gated latents, which go to
    the companion .npz (float32 per seed) whose sha256 the pin carries -- so every reading here
    re-derives offline from the pin alone (T077 3.5 item 23)."""
    import numpy as np

    if pin.exists():
        print(
            f"REFUSED: {pin} exists -- the pin is written once (delete it by hand only with the "
            f"record)",
            flush=True,
        )
        return 2
    t0 = time.time()
    det = t078_determinism(seeds[0] if seeds else 0)
    print(f"(p) determinism twin: {det}", flush=True)
    if not det["identical"]:
        print(
            "REFUSED at (p): two same-seed recordings are NOT byte-identical -- the instrument's "
            "own fault; nothing below is readable on a non-deterministic recording",
            flush=True,
        )
        return 2
    per_seed, lat = {}, {}
    for s in seeds:
        rows = t078_record_brain(s, n_ticks)
        g3 = t078_g3(rows, s)
        g4 = t078_g4(rows, s)
        arr, ticks = t078_pack_latents(rows)
        lat[f"seed{s}"], lat[f"seed{s}_ticks"] = arr, ticks
        per_seed[s] = {
            "g3": g3,
            "g4": g4,
            "n_ticks": len(rows),
            "n_awake": g3["n_awake"],
            # the DISCLOSURES the recorder's docstring names, COUNTED rather than adjudicated
            "ticks_two_attend_calls": sum(1 for r in rows if r["attend_calls"] > 1),
            "ticks_chan_disagrees": sum(1 for r in rows if r["chan"] != r["chan_from_gated"]),
            "ticks_ignited": sum(1 for r in rows if r["ignited"]),
            "rows": [{k: v for k, v in r.items() if k != "gated"} for r in rows],
        }
        print(
            f"seed {s}: awake {g3['n_awake']} pairs {g3['n_pairs']} (sleep-dropped "
            f"{g3['dropped_sleep']}, estop-dropped {g3['dropped_estop']}) cells {g3['cells']} "
            f"actions {g3['distinct_actions']} | primary readable={g3['primary']['readable']} "
            f"mi={g3['primary']['mi']} q95={g3['primary']['perm_q95']} | control "
            f"readable={g3['control']['readable']} mi={g3['control']['mi']} "
            f"q95={g3['control']['perm_q95']} | G4 rows {g4['n_rows']} classes {g4['n_classes']} "
            f"u_acc={g4['u']['acc']} band={g4['u']['shuffle_q95']} c15_acc={g4['c15']['acc']} "
            f"raw64_acc={g4['raw64']['acc']} chan_acc={g4['chan']['acc']} "
            f"[{time.time() - t0:.0f} s]",
            flush=True,
        )
    ss = [per_seed[s] for s in seeds]
    g3_readable = all(x["g3"]["primary"]["readable"] and x["g3"]["control"]["readable"] for x in ss)
    g4_readable = all(x["g4"][k]["readable"] for x in ss for k in ("u", "c15"))
    reading = {
        "determinism": det,
        "r_g3_readable": g3_readable,
        "r_g4_readable": g4_readable,
        "g3": "REFUSED",
        "g4": "REFUSED",
    }
    if not (g3_readable and g4_readable):
        reading["refusal"] = (
            "REFUSED at (r): a reading is not readable on every seed -- the instrument's own "
            "fault. G3 refusals: "
            # TAGGED BY ARM, over BOTH arms `g3_readable` conjoins: the two refuse under DIFFERENT
            # rules (2.10 reads the control per action CLASS, the primary per observed CELL), so a
            # control-only refusal is reachable and the single-arm list rendered it as `[]` -- a
            # refusal naming nothing, against `mi_read`'s own rule. 2.6 separates the (g3c) control
            # branch from (r), so WHICH arm refused is part of the fault's identity.
            + str(
                [
                    (k, x["g3"][k]["refusal"])
                    for x in ss
                    for k in ("primary", "control")
                    if not x["g3"][k]["readable"]
                ]
            )
            + "; G4 refusals: "
            + str(
                [
                    x["g4"][k]["refusal"]
                    for x in ss
                    for k in ("u", "c15")
                    if not x["g4"][k]["readable"]
                ]
            )
        )
        print(reading["refusal"], flush=True)
    else:
        g3c = contrast(
            "(g3c) the estimator sees",
            [x["g3"]["control"]["mi"] - x["g3"]["control"]["perm_q95"] for x in ss],
        )
        g3 = contrast(
            "(g3) task binding",
            [x["g3"]["primary"]["mi"] - x["g3"]["primary"]["perm_q95"] for x in ss],
        )
        g4 = g4_rule([x["g4"]["u"] for x in ss], [x["g4"]["c15"] for x in ss])
        reading["g3c"] = g3c._asdict()
        reading["g3_contrast"] = g3._asdict()
        reading["g4a"] = g4["g4a"]._asdict()
        reading["g4b"] = g4["g4b"]._asdict()
        reading["g3"] = "REFUSED" if not g3c.met else ("PASS" if g3.met else "STOP")
        reading["g4"] = g4["reading"]
        reading["leak_points"] = statistics.mean(
            x["g4"]["u"]["acc"] - x["g4"]["u"]["shuffle_q95"] for x in ss
        )
        reading["reported"] = {
            # BOTH MI readings go through `_reported_mi`: a refusal becomes a REFUSAL record, never
            # a 0.0 diff. same_tick was readable on all 8 seeds of the landed run, so it carried the
            # identical `or 0.0` coercion one refusal away from fabricating a number.
            "gate16": _reported_mi("gate16 (reported)", "gate16", seeds, ss),
            "same_tick": _reported_mi("same-tick focus/action (reported)", "same_tick", seeds, ss),
            "raw64_ceiling": contrast(
                "raw-64 content ceiling (reported)",
                [x["g4"]["raw64"]["acc"] - x["g4"]["u"]["acc"] for x in ss],
            )._asdict(),
            "chan": contrast(
                "chan decode (reported)",
                [x["g4"]["chan"]["acc"] - x["g4"]["chan"]["shuffle_q95"] for x in ss],
            )._asdict(),
        }
        print("-" * 78, flush=True)
        print(_line(g3c), flush=True)
        print(_line(g3), flush=True)
        print(_line(g4["g4a"]), flush=True)
        print(_line(g4["g4b"]), flush=True)
        for name, c in reading["reported"].items():
            if c.get("refused"):
                seed0, why = next(iter(c["refusals"].items()))
                print(
                    f"  reported {name}: REFUSED on {c['n_refused']}/{len(ss)} seeds "
                    f"(seed {seed0}: {why})",
                    flush=True,
                )
            else:
                # the DEGENERATE marker `_line` appends and this loop used to drop
                deg = " DEGENERATE" if c["degenerate"] else ""
                print(
                    f"  reported {name}: t={c['t']:.4f}{deg} FPR={c['fpr']} met={c['met']}",
                    flush=True,
                )
        print(
            f"PART I VERDICT: G3 {reading['g3']} x G4 {reading['g4']} "
            f"(leak {reading['leak_points']:+.4f} accuracy points)",
            flush=True,
        )
    latents.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(latents, **lat)
    reading["latents_sha256"] = hashlib.sha256(latents.read_bytes()).hexdigest()
    out = {
        "row": "T078 Part I",
        "n_ticks": n_ticks,
        "seeds": list(seeds),
        "bar": T078_BAR,
        "n_perm": N_PERM,
        "n_shuffle": N_SHUFFLE,
        "n_min_per_cell": N_MIN_PER_CELL,
        "u_fields": list(U_FIELDS),
        "latents": str(latents),
        "latents_sha256": reading["latents_sha256"],
        "reds": t078_reds(),
        "per_seed": per_seed,
        "reading": reading,
        "elapsed_s": round(time.time() - t0, 1),
    }
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"pinned {pin} (+ {latents}) in {time.time() - t0:.0f} s", flush=True)
    return 0


#: The one-line description of each red, printed beside its boolean so a gate run's log says WHAT
#: fired rather than only that something did. 2.4's wording, compressed to one ASCII line each.
RED_LINES = {
    "R1": "the permutation floor absorbs the plug-in bias (an un-permuted floor reddens it)",
    "R2": "MI(a_t; a_t+1) is a CONTROL, not an identity (scoring y against itself reddens it)",
    "R3": (
        "145 pairs over 39 cells REFUSE per-CELL and READ per-CLASS (2.10's twin), 600 do not "
        "refuse (N_MIN_PER_CELL := 0 reddens the per-cell half, a min_rule-BLIND scorer the twin)"
    ),
    "R4": "the decode beats its 200-shuffle band (scoring on the TRAIN half reddens it)",
    "R5": "u_t is content-blind by TYPE and not constant (appending chan[0] reddens it)",
    "R6": "the VOID rule can fire and PASS can too (swapping (g4b)'s operands reddens it)",
}


# distinct-from: cortex/metacognitive_monitor.py::t077_reds and its main() reds line -- the PATTERN
#   this self-check mirrors, and 2.1 names it as the thing to reuse. Read: that line asserts T077's
#   R0-R5 over a metacognitive monitor's replay and its type-2 AUROC; this asserts T078's R1-R6
#   over an MI floor, a decoder band and a control vector. Same shape, different instrument,
#   different reds, no shared state -- one self-check per module is the tree's own convention, not
#   a second mechanism.
# distinct-from: substrate/paired_contrast.py::main -- the reds of the STATISTIC (it must be able
#   to say NOT MET, its FPR must be exact). Read; that module is IMPORTED here, never re-checked.
# distinct-from: thalamus/gwt_deep.py::_reds -- constructs the input that makes each BINDING
#   criterion fail. distinct-from: substrate/selfcheck_cache.py::self_check -- cache key-invariance
#   reds. distinct-from: substrate/learn_daemon.py::_selfcheck -- the daemon's non-driving checks.
#   All three read; each fires its own module's reds, and none touches a permutation floor, a
#   decoder band or a control state.
def main():
    """The self-check: 2.4's six reds, each holding on its constructed input AND reddened IN FACT
    by its own mutated scorer. No brain, no lake, no network -- the campaign is `--t078-campaign`
    and lands in 2.9 step 5."""
    if "--t078-campaign" in sys.argv[1:]:  # 2.9 step 6: the campaign, scratch home, pinned ONCE
        from snn_data_lake.substrate.home import throwaway_home

        with throwaway_home():
            sys.exit(run_t078_campaign())
    t0 = time.time()
    reds = t078_reds()
    print("=" * 78)
    print("T078 PART I -- the attention schema on the live loop: the gates first")
    print("=" * 78)
    print(
        f"bar {T078_BAR} | u_t {U_DIM} fields | N_MIN_PER_CELL {N_MIN_PER_CELL} | "
        f"campaign n_perm {N_PERM} | n_shuffle {N_SHUFFLE} | seeds {list(T078_SEEDS)} | "
        f"ticks {T078_N_TICKS}"
    )
    print(
        f"reds run at n_perm {REDS_N_PERM} (2.1's 2,000 is the CAMPAIGN's), n_shuffle {N_SHUFFLE}"
    )
    print("-" * 78)
    for name in sorted(reds):
        print(f"  {name}: {str(reds[name]):5s}  {RED_LINES[name]}")
    print("-" * 78)

    # ---- 2.4's SIX REDS. Each is also a test in tests/test_attention_schema.py; this line is the
    #      in-command control, so a gate run that never executes pytest still runs them. Every
    #      `not (...)` clause inside t078_reds() is a MUTATED scorer having been RUN (P8 clause 1:
    #      a criterion with no reachable red is DELETED, not weakened).
    assert all(reds.values()), f"T078 reds must fire on their constructed inputs: {reds}"

    print(
        f"self-check OK: R1-R6 fire on their constructed streams and each is reddened IN FACT by "
        f"its mutated scorer ({sum(reds.values())}/{len(reds)}); no brain, no lake, no network "
        f"[{time.time() - t0:.0f} s]"
    )
    print(
        "  PART I adds NO wire: there is no attention schema in this module, brain.py's "
        "_attend_real_stream keeps goal=None, and the recorder + campaign land in 2.9 step 5"
    )


if __name__ == "__main__":
    main()
