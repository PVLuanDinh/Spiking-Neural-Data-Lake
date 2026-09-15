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
    (`draws[min(len-1, round(0.95*(len-1)))]`).
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
  DECODER_STEPS / DECODER_LR / DECODER_L2 = 200 / 0.1 / 1e-3   2.1 -- fixed; the raw-64 ceiling
                                   decode is what shows whether capacity, not information, limits
                                   the matched decode.
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

from pathlib import Path

from snn_data_lake.substrate.paired_contrast import T_BAR

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


def mi_bits(x, y):
    """PART I STUB (2.9 step 2). The plug-in mutual information in bits of two equal-length
    discrete label sequences, over the OBSERVED joint cells, with no smoothing -- the permutation
    floor absorbs the plug-in bias. Returns a null reading until 2.9 step 4 lands."""
    return 0.0


def mi_read(x, y, n_perm=N_PERM, seed=0):
    """PART I STUB (2.9 step 2). `mi_bits` against its own permutation floor: the 95th percentile
    of `mi_bits(x, permuted y)` over `n_perm` permutations of y, drawn by a seeded
    `random.Random` -- the same convention as `metacognitive_monitor.track_error`. REFUSES below
    `N_MIN_PER_CELL` pairs per observed joint cell, or on a one-class axis. Returns a null
    reading until 2.9 step 4 lands."""
    return {"readable": False, "mi": None, "perm_q95": None, "refusal": "stub"}


def decode_heldout(X, y, seed, n_shuffle=N_SHUFFLE):
    """PART I STUB (2.9 step 2). A multinomial logistic decoder trained on the FIRST half of the
    rows in tick order and scored on the SECOND (prequential halves, no lookahead), against a
    band of `n_shuffle` decoders trained on label-permuted train halves and scored on the TRUE
    test labels. Returns a null reading until 2.9 step 4 lands."""
    return {"readable": False, "acc": None, "shuffle_q95": None}


def control_state(brain):
    """PART I STUB (2.9 step 2). A4's candidate `u_t` -- the 15 fields of `U_FIELDS`, in that
    order, read from the live signals of the pre-registration's 1.2 and from nothing else.
    Returns the 15 zeros until 2.9 step 4 lands."""
    return [0.0] * U_DIM


def t078_record_brain(seed, n_ticks=T078_N_TICKS):
    """PART I STUB (2.9 step 2). One SCRATCH brain's per-tick recording (2.1). Built in 2.9
    step 5; it is deliberately absent here so that nothing in this commit can touch a brain."""
    raise NotImplementedError("t078_record_brain lands in 2.9 step 5 (the recorder)")


def t078_reds():
    """2.4's six reds as ONE self-check line (each is also a test in
    `tests/test_attention_schema.py`). Every value is False against the stubs: the reds are
    committed RED, and they go True in 2.9 step 4 only when each is reddened IN FACT by its own
    mutated scorer."""
    return {"R1": False, "R2": False, "R3": False, "R4": False, "R5": False, "R6": False}


def run_t078_campaign(seeds=T078_SEEDS, n_ticks=T078_N_TICKS, pin=T078_PIN):
    """PART I STUB (2.9 step 2). The Part I campaign (2.2-2.3). Built in 2.9 step 5 and RUN only
    after the Codex design read folds as 2.10 -- this commit refuses rather than records."""
    print(
        f"REFUSED: run_t078_campaign is a Part I stub (2.9 step 2); the recorder lands in step 5 "
        f"[seeds={list(seeds)} n_ticks={n_ticks} pin={pin}]",
        flush=True,
    )
    return 2


def main():
    """THE STUB SELF-CHECK. It asserts NOTHING: at 2.9 step 2 every red is False by construction,
    and a self-check that went red here would redden CI on a commit whose whole purpose is to give
    step 3's tests a real symbol to fail against. The asserts arrive with the instruments in
    step 4, together with the mutated scorers that make each red reachable."""
    reds = t078_reds()
    print("=" * 78)
    print("T078 PART I -- the attention schema on the live loop: the gates first (SKELETON)")
    print("=" * 78)
    print(
        f"bar {T078_BAR} | u_t {U_DIM} fields | N_MIN_PER_CELL {N_MIN_PER_CELL} | "
        f"n_perm {N_PERM} | n_shuffle {N_SHUFFLE} | seeds {list(T078_SEEDS)} | ticks {T078_N_TICKS}"
    )
    print("-" * 78)
    for name in sorted(reds):
        print(f"  {name}: {reds[name]}  (stub -- the instrument lands in 2.9 step 4)")
    print("-" * 78)
    print(f"T078 Part I stubs: {sum(reds.values())}/{len(reds)} reds fire, as expected at step 2")


if __name__ == "__main__":
    main()
