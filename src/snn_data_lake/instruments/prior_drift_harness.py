"""THE three-arm drift harness: is the H01 prior beatable, and does learning move the brain off it?

specs/022 T063 / SC-006 / FR-022, composing specs/011's DriftRecord design (SC-004, SC-005,
FR-008/009/010, C5, D1, D4, Scenario 2, T003, T004). The pre-registration is
`specs/022-unified-brain-arc/evidence/t063-prior-drift-harness.md`, sections 1-2.11 FROZEN before
a line of this module existed; this file implements it and adds nothing to it.

DISPOSITION 2026-09-09 (T063's record, section 3) -- THIS HARNESS NO LONGER HAS A LEARNER. The
campaign pinned in `tools/t063_campaign.json` fired 2.9's null clauses. The learner's own
criterion FAILED: t(d_f) = -13.0154, the FROZEN twin ahead of the learning arm on every one of the
8 seeds, so `substrate/kernel_plasticity` was DELETED on the disposition it was built under ("a
learner that moves weights without an attributed benefit is a null"). What that costs, stated
plainly rather than worked around: DRIFT IS NOT MEASURABLE by this instrument any more -- nothing
in the tree can write a weight into a `Synapses` bundle table -- so conjunct (a) reads NOT READ,
SC-006 can never be called MET here again, and specs/011's FR-008 / FR-009 remain OPEN. What
SURVIVES is everything that never needed a learner: the five tables, the task instrument, the
nuisance refusals, the WRONG arm's validity precondition, the PLANTED positive control, and the
prior's contrasts (b) (c) (d) (e). The DriftRecord machinery and the four diagnostic lines survive
too, and now serve as the MEASUREMENT that no learner ran -- four zeros and two zero distances per
arm, read rather than assumed from the absence of an import. The campaign is NOT re-run and NOT
edited: it is the record of the learner's run at its own head and harness hash.

APPENDED 2026-09-10 (T063 panel of 2026-09-09), the paragraph above kept VERBATIM -- WHAT
"SURVIVES" AND "STAND" MEAN, MEASURED. On `data/h01/h01_c3_graph.npz` at the pinned K = 8, D = 4,
W = 1.0 over seeds 0-7 this no-learner harness reads (c) NOT MET at t = +0.8143 (permutation FPR
0.0469) -- per-arm means seeded 0.8750, random 0.8281, wrong 0.8281, planted 0.8750, with FROZEN
identical to SEEDED and d_f = 0 on all 8 seeds, so the identity control holds and the reading is
on the pinned path. `verdict()` therefore prints SC-006 = NO VERDICT with (a) (b) (d) (e) NOT
READ, "no verdict is read: the harness is BLIND". So what SURVIVES the deletion is RETAINED AS A
MECHANISM, and on this population at this capacity the instrument REFUSES rather than reads --
2.2's pre-registered disposition firing, not a new defect. A successor cannot re-read (b) (c) (d)
(e) on H01 with this instrument as-is and must re-derive the capacity guard under a NEW
pre-registration. The bare self-check's green is the 64-unit fixture at `out_degree = 8`, which
does not cover the artifact. REPRODUCTION, and it is not the CLI: direct calls to
`band_balanced_subgraph` / `drive_classes` / `build_arms` / `wrong_validity` / `run_arm` /
`verdict` at `_pin_stages()[-1]`'s constants with NORD_HOME on a scratch dir -- NEVER
`--campaign`, which rewrites `tools/t063_campaign.json` and would destroy the record of the
learner's run. The pinned campaign's own (c) = +9.2601 was read WITH the learner and stands.

FIVE TABLES on ONE unit set, ONE drive, NO learner, ONE readout -- only the wiring differs:
  SEEDED   T062's H01 c3 population at the pilot's width.
  RANDOM   the control that is NOT the method: a WITHIN-SIGN degree-preserving rewire, the
           per-edge synapse-count multiset reassigned at random inside each sign class, no self-
           or multi-edges, fan-in re-normalised. WHAT IT TESTS, narrowly: wiring information
           BEYOND the retained H01 marginals (degrees, signs, magnitudes), which the two arms
           SHARE. It cannot separate "H01 contributes nothing" from "H01's useful information is
           exactly what both arms share", and every claim here is restricted to that component.
  WRONG    the BAND-PERMUTED-TARGETS table (specs/011 T004, RE-DECIDED by amendment 2.12): every
           edge keeps its pre and its synapse count and gets a post in band pi(band(post)) at the
           same within-band degree rank, pi reversing cortical depth. VALID only if its band-pair
           weight-share matrix is farther from SEEDED's than RANDOM's realised matrix is --
           measured, never assumed. Its predecessor, the DIRECTION-REVERSED laminar table, was
           RETIRED on that measurement: PSI(reversed, seeded) = 0.032389 against
           PSI(random, seeded) = 2.281271, because 0.726520 of the band-pair mass is DIAGONAL and
           a transpose cannot move it. Its readings stay in `tools/t063_pilot.json`'s first stage.
  PLANTED  the POSITIVE CONTROL, never an arm of the verdict: each drive class's units wired
           densely onto a disjoint block of non-driven units. ONE predicate: PLANTED beats RANDOM
           at the same t bar. If it does not, the harness is BLIND and NO verdict is read.
  FROZEN   WAS the learner's own control -- SEEDED at lr = 0, red 1's constructed pin and the twin
           the retention criterion was read against. With no learner it is SEEDED run a second
           time, so d_f must be identically 0 and the contrast is kept as an IDENTITY control: a
           nonzero reading is a DETERMINISM defect in this harness, never a learning signal.

THE TASK'S READOUT sees the non-driven units' spike counts ONLY through a FIXED per-seed
orthonormalised Gaussian projection to D dimensions (2.12(a)). D -- the readout's CAPACITY -- is
chosen by the pilot as the LARGEST of {4, 8, 16, 32, 64, 128} at which the no-learning RANDOM arm
scores below 0.9, and the row is REFUSED ON MEASUREMENT if no D qualifies. K is FIXED at 8. The
guard used to be K itself and selected NOTHING: over ~13.5k non-driven units against ~24 training
trials, the least-squares readout interpolated exactly and EVERY wiring read 1.0000.

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

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.prior_drift_harness
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.prior_drift_harness --pilot
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.prior_drift_harness --campaign
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

from snn_data_lake.instruments.paired_contrast import (  # noqa: F401  (re-exported: LIFTED 2026-09-15, T077 2.10)
    T_BAR,
    Contrast,
    contrast,
    paired_t,
    permutation_fpr,
)
from snn_data_lake.substrate.drift_monitor import psi

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
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
        "by a name chosen to dodge a gate. DATED 2026-09-09: the judgement was EXECUTED and the "
        "subject is gone -- the formatter now reports the DELETION and its date, and derives no "
        "criterion from a run that has no learner to judge",
    ),
    "no_learner_holds": (
        "NOT_A_RULE",
        "the ASSERTION THAT NO RULE RAN, and the census is right to enumerate it: the name carries "
        "the learning verb precisely because the thing it names is a learner's absence. It reads "
        "an already-computed ArmResult -- four diagnostic counters and two DriftRecord distances "
        "-- and returns True iff all six are zero. It touches no weight, forms no trace, computes "
        "no update and has no gradient of any kind. Declared rather than renamed, for the same "
        "reason `_learner` above is: a name chosen to dodge a gate tells a later reader less than "
        "the reason does (2026-09-09)",
    ),
}

#: The one statement of what this instrument can no longer measure, quoted by red 1's refusal, by
#: conjunct (a) and by the learner's disposition, so the three cannot drift apart.
NO_LEARNER = (
    "no learner in the tree -- substrate/kernel_plasticity DELETED on its own criterion "
    "2026-09-09 (T063's record, section 3): drift is not measurable; FR-008/FR-009 (specs/011) "
    "remain OPEN"
)

#: The four diagnostic lines of 2.5 as a no-learner arm reads them. Written ONCE so an arm cannot
#: report a shape the refusal does not describe.
NO_LEARNER_DIAGNOSTICS = {
    "executed": 0,
    "attempted": 0,
    "effective_count": 0,
    "effective_l1": 0.0,
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
#: 2.12(a). The guard MOVED: it was the class count K (the smallest power of two >= 8 at which the
#: no-learning RANDOM arm fell under 0.9) and that rule selected NOTHING -- RANDOM read 1.0000 at
#: K = 8 and at K = 16, and K = 32 does not fit (32 x 5 % > 100 %). K is now FIXED at 8 and the
#: guard is the READOUT'S CAPACITY: D, the dimension of a fixed per-seed orthonormalised random
#: projection of the non-driven units' spike counts. The task's DEFINITION is unchanged.
K_FIXED = 8
D_CANDIDATES = (4, 8, 16, 32, 64, 128)
D_CEILING = 0.9  # the no-learning RANDOM arm must score BELOW this at the chosen D
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
    "HISTORICAL host-clause reading (2026-09-09, PID 27228). The settling reading for any run is "
    "the in-band `settling` block beside it, which governs. "
    "CPU WINDOWS TAKEN ON DAEMON PID 27228 (not necessarily the PID a later run records -- the "
    "daemon on this host RESTARTS: 33396, then 27228, then 32604 were observed holding 8137 "
    "inside about forty minutes, so `host.daemon_pid` beside a number and this note's PID may "
    "differ, and that is a fact about the daemon rather than a mismatch to reconcile). LISTEN on "
    "8137 throughout, working set 5.50-8.77 GB; five 90 s CPU windows 112.000 / 122.906 / "
    "106.266 / 133.641 / 117.828 s, mean 118.5, no trend, range +-11.5 %. The "
    "two-consecutive-windows-within-6 % clause is NOT MET and is not meetable by this daemon's "
    "steady duty cycle; settling read from the absence of a trend. No battery and no panel were "
    "running. Every timing is a min of repeats."
)

S_SEEDS = 8
# T_BAR, Contrast, paired_t, permutation_fpr and contrast were LIFTED 2026-09-15 (specs/022 T077 2.10) into
# instruments/paired_contrast.py, verbatim, and are imported at the top of this module so every name still
# resolves here; the reason is in that module's docstring (a live module importing this EXEMPT instrument
# stales five dormancy rows). Nothing else in this file changed.
STATISTICS = ("band_pair_weight_share", "band_fanin_e_fraction")
N_BANDS = 7

#: 2.12(b). pi reverses CORTICAL DEPTH over `h01_ingest.BANDS` = (Layer 1 .. Layer 6, White
#: matter): Layer 1 <-> Layer 6, Layer 2 <-> Layer 5, Layer 3 <-> Layer 4, White matter fixed.
BAND_PERMUTATION = (5, 4, 3, 2, 1, 0, 6)

#: 2.12(c). SETTLING_WINDOWS is the amendment's FLOOR -- "at least three windows" -- below which
#: there is no reading at all. SETTLING_MEASURED is how many a RUN actually takes, and it is not
#: the floor on purpose: over n windows of a settled duty cycle exactly 2 of the n! orderings are
#: monotone, so a three-window reading rejects a SETTLED host one time in three (measured: the
#: 2026-09-09 re-run read 127.047 / 121.172 / 109.969 on one live daemon, no restart, and the
#: clause said NOT settled because those three came out descending). At five it is 2/120 = 1.7 %.
SETTLING_WINDOWS = 3
SETTLING_MEASURED = 5
SETTLING_SECONDS = 90


class ArmRefused(ValueError):
    """An arm that violates a nuisance property is refused BEFORE it runs, not explained after."""


class DriftRecord(NamedTuple):
    """specs/011 data-model.md 4, fields verbatim. Sign is EXCLUDED from drift by design."""

    statistic: str
    seeded: tuple
    current: tuple
    distance: float
    direction: int


# distinct-from: substrate/kernel_plasticity.py::KernelPlasticity.diagnostics -- returned a LIVE
#   learner's four counters as it accumulated them; this is the immutable per-arm record a scored
#   run returns, and after 2026-09-09 there is no learner to accumulate anything.
# distinct-from: gates/amdahl_gate.py::red_repeatability_broken_arm -- a deliberately PADDED
#   second timing reading, built to reach a criterion that had no failing arm; "arm" there is a
#   reading of one timed pair, not a wiring of a population.
# distinct-from: gates/replay_gate.py::run_gate -- returns four arms' outcomes where arms 2-4
#   are arm 1's reds; the arms differ by an INJECTED DEFECT, not by a table under comparison.
# distinct-from: sensorimotor/arm_bridge.py::send_outcome and brainstem/so101_arm.py::collision -- the
#   ROBOT arm's UDP reward datagram and its contact authority; "arm" there is hardware.
#   All five sites read 2026-09-09 before this note was written.
class ArmResult(NamedTuple):
    """The scored record of one wiring at one seed, plus 2.5's four diagnostic lines.

    `pinned` was a field here until 2026-09-09 and is GONE with the learner. It was TRUE only for
    a VERIFIED lr = 0 intervention -- the single thing that licensed the word "pinning" in red 1's
    message -- and with no learner no such intervention can be applied to any wiring. A field that
    can never be True is not a reading, and keeping one would let the refusal below claim a pin
    nobody applied. The four diagnostic lines STAY: they carry the MEASUREMENT that no learner
    ran, rather than leaving it inferred from the absence of an import."""

    name: str
    accuracy: float
    executed: int
    attempted: int
    effective_count: int
    effective_l1: float
    records: tuple
    #: 2.12(a): the D the readout actually saw, carried so "the readout has capacity D" is a
    #: reading off the result rather than a claim about the call site.
    feature_dim: int = 0


# (`Contrast` lives in instruments/paired_contrast.py since 2026-09-15; imported above.)


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
    """RED 1's SLOT, and what it reads after 2026-09-09: the NO-LEARNER REFUSAL. Returns
    (fired, message), and `fired` is now always False -- there is no drift verdict to read.

    Red 1 asked whether the brain never moves off the seed, and its whole discipline was
    ATTRIBUTION: pinning was claimed only for a VERIFIED lr = 0 intervention, while a learning
    arm's zero was reported as cancellation, quantization or absent pairs (Codex defect 4). With
    the learner deleted the question has no subject -- there is no intervention to verify and no
    learning arm to contrast it against -- so every arm's zero has ONE cause, and the honest
    reading is the harness defect 2.9 itself names: "if effective change is 0 on every arm the
    learner is DELETED and red 1's reading is a harness defect, not a finding".

    The message still carries all four diagnostic lines, because a zero is read from four lines
    and never from one, and it must NEVER contain the word PINNED: claiming a pin nobody applied
    is the exact attribution error red 1 existed to prevent."""
    return False, (
        f"harness defect: no learner ran ({result.name}): executed {result.executed}, "
        f"attempted {result.attempted}, effective {result.effective_count}, "
        f"L1 {result.effective_l1:.6f} -- {NO_LEARNER}"
    )


# distinct-from: gates/amdahl_gate.py::red_sham_neuter -- RED (c), a stand-in on a ZERO-share
#   path whose bar is "within noise of zero" over a resampled band; that is a magnitude criterion
#   on a TIMING, judged against a reported band. This is an EXACT-zero predicate over six recorded
#   quantities of one arm and admits no band at all.
# distinct-from: cortex/h01_ingest.py::_disclose -- prints five ingest numbers and explicitly never
#   filters on them; it decides nothing. This decides, and its decision gates a conjunct.
# distinct-from: basal_ganglia/r_stdp.py::Calibration -- a NamedTuple of what a host measured for
#   the parallel crossover; a record, not a predicate.
# distinct-from: substrate/signal_ring.py::attach -- cursors a new consumer at the current head;
#   "sees nothing before it attached" is a ring's history rule, not a zero reading.
# distinct-from: cortex/interaction_learning.py::_live_brain_demo -- talks to the live brain and
#   shows a learner improving; it is a demo of a learner, and its subject is exactly the thing
#   this predicate asserts is ABSENT. All five sites read 2026-09-09 before this note was written.
def no_learner_holds(result):
    """RED 1's CONTROL, inverted and MEASURED. The learning control was "the SEEDED arm at the
    declared lr reads effective > 0 and a distance above zero on BOTH statistics"; with no learner
    the reading that must hold is its exact complement -- all four diagnostic lines at zero AND
    both distances exactly zero -- and it is taken per arm FROM THE RUN rather than assumed from
    the absence of an import (runtime evidence outranks static reading).

    Its red is reachable six ways: any nonzero diagnostic line, or any nonzero distance, breaks
    it. An EMPTY record set breaks it too, for the reason red 1 refused to fire on one: there is
    nothing to read the zero FROM. The distance half is `is_seeding_failure` itself -- specs/011's
    FR-009 / C5 / T012 predicate, CALLED and not re-written, so the tree states "every distance is
    zero" exactly once."""
    return bool(
        result.executed == 0
        and result.attempted == 0
        and result.effective_count == 0
        and result.effective_l1 == 0.0
        and is_seeding_failure(result.records)
    )


# =============================================================================================
# The statistic over the seeds (2.6) -- `paired_t`, `permutation_fpr`, `contrast`: LIFTED 2026-09-15 into
# instruments/paired_contrast.py (T077 2.10), verbatim; imported at the top of this module.
# =============================================================================================


#: THE FLOOR RULE, LIFTED (P2). It was written once, in prose, inside `synthetic_artifact`'s
#: docstring -- "12 and 16 are REJECTED though their t is larger: both base arms sit at chance
#: (0.25 = 1/K) there, and a positive control that beats a floor is a formality, not a reading" --
#: where it governed the FIXTURE only, and the same docstring told the reader it "changes nothing
#: about the campaign's population". The campaign was never exempt from it. ONE function now states
#: the rule, that docstring cites it, and it reads the fixture and the campaign alike.
# distinct-from: instruments/nef_compute.py::_run_hint -- DERIVES A COMMAND STRING from `__spec__.name` so a self-check hint cannot go stale when the file moves; it reads no number and has no criterion.
# distinct-from: reasoning/self_improve.py::_self_pkg_relpath -- DERIVES THIS MODULE'S OWN PATH from `__file__` so a PROTECTED-set assert has an independent second side; a filesystem walk, not a statistic.
# distinct-from: substrate/self_mod_harness.py::_repo_root -- WALKS UP TO THE REPO ROOT by marker files; an anchored directory search, with no accuracy, no chance level and no bar.
#   All three sites read 2026-09-10 before this marker was written. What follows derives an arm's
#   distance from the 1/K chance floor and reads it with the row's own paired t and its own bar.
def above_chance(accs, k, bar=T_BAR):
    """Is this arm's accuracy distinguishable from the 1/K floor, by the harness's OWN statistic?

    `accs` is one accuracy per seed; `k` is the number of drive classes, so chance is 1/k. The
    reading is `contrast` on (accuracy - chance) over the seeds -- the same paired one-sided t and
    the same permutation FPR every conjunct is read with, against the same bar.

    IT REPORTS AND IT NEVER REFUSES. Nothing here changes a conjunct, a verdict or a threshold, and
    it is deliberately not a gate: a guard that suppressed a reading taken near the floor would
    have suppressed the pinned campaign's own t(d_f) = -13.0154, the number that deleted the
    learner, and a guard that suppresses a negative is worse than the omission it repairs. What it
    buys is that a reader of any verdict can see WHERE the arms were when a contrast between them
    was read (T063 panel of 2026-09-09, finding F22)."""
    accs = [float(a) for a in accs]
    chance = 1.0 / float(k)
    c = contrast("arm - chance", [a - chance for a in accs], bar)
    return {
        "mean": sum(accs) / float(len(accs)) if accs else 0.0,
        "chance": chance,
        "t": c.t,
        "fpr": c.fpr,
        "degenerate": c.degenerate,
        "met": bool(c.met),
    }


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


# distinct-from: substrate/kernel_tick.py::Synapses.empty -- the empty COMPILED table (zero call
#   sites); this returns an ARTIFACT dict for `population()` to compile, the units untouched.
# distinct-from: cortex/h01_ingest.py::build_arrays -- builds the node and edge arrays FROM the
#   release's cells; this drops the edges of an artifact already built.
# distinct-from: gates/determinism_gate.py::_bench_net -- a disclosed synthetic bench with
#   random edges; nothing is cut. distinct-from: substrate/brain_geometry.py::_connectome_pairs --
#   region->region tracts of the connectome, not a unit population. All four sites read 2026-09-10.
def arm_no_edges(art):
    """The population with EVERY edge cut -- T003's drive-only twin, and (specs/022 T070) the
    integration instrument's PARTITIONED arm, "the same parts with the measured channel cut". Lifted
    2026-09-10 from `run_pilot`'s inline block so the two callers share ONE constructor (P2)."""
    out = _copy(art)
    for key in ("pre_idx", "post_idx", "n_syn"):
        out[key] = out[key][:0]
    return out


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


def within_band_degree_rank(art):
    """Each unit's rank INSIDE its own band by IN-degree, ties by artifact index; 0 is the highest.
    In-degree and not total degree, because the quantity `arm_wrong` remaps is a TARGET."""
    import numpy as np

    n = int(art["agglo_id"].size)
    indeg = np.zeros(n, dtype=np.int64)
    np.add.at(indeg, art["post_idx"].astype(np.int64), 1)
    rank = np.zeros(n, dtype=np.int64)
    for band in range(N_BANDS):
        members = np.nonzero(art["layer"] == band)[0]
        if members.size == 0:
            continue
        ordered = members[np.lexsort((members, -indeg[members]))]
        rank[ordered] = np.arange(ordered.size, dtype=np.int64)
    return rank


# distinct-from: h01_ingest.py -- `drive_train` drives 1 % of units for N ticks and hashes the run.
# distinct-from: kernel_tick.py -- `_bench_kernel` times a synthetic network to derive P.
# distinct-from: placement.py -- `_place_grn` draws POSITIONS for a non-cortical region's units.
# distinct-from: prefrontal_executive.py -- `retention_dead` reads one memory against its floor.
# distinct-from: schema_induction.py -- `_confident` RANKS entities as candidate tails for a fact.
#   None of the five maps a unit to another unit; this is a pure index permutation over the
#   laminar bands, and it has no dynamics, no drive and no scoring in it. All read 2026-09-09.
def band_rank_map(art):
    """The UNIT-level half of `arm_wrong`, separated so it is readable on its own: for every unit u
    of band b, the unit of band pi(b) holding the same within-band degree RANK. Returns
    (target, slot, ordered, tgt_band) -- the map, the rank slot each unit took in its target band,
    each band's units in rank order, and each unit's target band.

    "The same within-band degree rank" between bands of different size (Layer 2 has 4,709 units
    against Layer 6's 1,040) is realised as the same rank QUANTILE: rank r of n_b maps to
    floor(r * n_pi(b) / n_b). It is the only reading of "same rank" that is total and MONOTONE
    across unequal sets -- and monotone is the property that makes this a laminar remap rather than
    a second scramble. Where the target band is smaller the map is many-to-one, which is why this
    arm permits repeated (pre, post) pairs and is not checked for multi-edges."""
    import numpy as np

    lay = np.asarray(art["layer"]).astype(np.int64)
    rank = within_band_degree_rank(art)
    ordered = []
    for band in range(N_BANDS):
        members = np.nonzero(lay == band)[0]
        ordered.append(members[np.argsort(rank[members], kind="stable")])

    tgt_band = np.asarray(BAND_PERMUTATION, dtype=np.int64)[lay]
    src_n = np.asarray([max(1, ordered[b].size) for b in range(N_BANDS)], dtype=np.int64)
    tgt_n = np.asarray([ordered[b].size for b in range(N_BANDS)], dtype=np.int64)
    slot = (rank * tgt_n[tgt_band]) // src_n[lay]
    slot = np.minimum(slot, np.maximum(tgt_n[tgt_band] - 1, 0))

    target = np.arange(lay.size, dtype=np.int64)
    for band in range(N_BANDS):
        sel = tgt_band == band
        if ordered[band].size == 0 or not bool(sel.any()):
            continue  # an empty mirrored band: the unit keeps its own place, counted as a fallback
        target[sel] = ordered[band][slot[sel]]
    return target, slot, ordered, tgt_band


def arm_wrong(art):
    """2.12(b) -- the BAND-PERMUTED-TARGETS table (T004 re-decided). Every edge KEEPS its pre and
    its synapse count; its post becomes the unit of band `pi(band(post))` holding the same
    within-band degree rank, `pi` reversing cortical depth. Polarity is re-applied from the pre --
    which this arm never moves, so `sign(pre)` and every unit's OUT-degree are untouched -- and
    fan-in is re-normalised by the weight formula. Edge count and synapse-count multiset preserved.

    WHY IT REPLACED THE DIRECTION-REVERSED TABLE. A reversal transposes the realised band-pair
    matrix, and 0.726520 of that matrix's mass sits on the DIAGONAL, which a transpose cannot
    touch: the pilot measured PSI(reversed, seeded) = 0.032389 against PSI(random, seeded) =
    2.281271, so 2.2's validity precondition failed by 70x -- structurally, not unluckily. A band
    PERMUTATION moves the diagonal itself (Layer 2's 4,709 units' intra-band mass lands on Layer
    6's 1,040), so it is far from SEEDED wherever band masses differ. It falsifies the claim red 2
    is FOR: that the laminar IDENTITY of a synapse's target carries information.

    TWO READINGS, DISCLOSED because the amendment's text admits them. (1) "the same within-band
    degree rank" between bands of different size is realised as the same rank QUANTILE -- rank r of
    n_b maps to floor(r * n_pi(b) / n_b) -- the only total, monotone reading of "same rank" across
    unequal sets; where the target band is smaller the map is many-to-one, which is why this arm
    permits repeated (pre, post) pairs and is not checked for multi-edges. (2) A remap that would
    make an edge a SELF-edge walks forward inside the target band to the next unit; if the target
    band is empty, or holds only that one unit, the ORIGINAL post is kept. `band_mass_report`
    counts every edge that did not land in pi(band(post)) for exactly that reason."""
    import numpy as np

    out = _copy(art)
    pre = art["pre_idx"].astype(np.int64)
    post = art["post_idx"].astype(np.int64)
    target, slot, ordered, tgt_band = band_rank_map(art)
    new_post = target[post]

    for i in np.nonzero(new_post == pre)[0].tolist():
        pool = ordered[int(tgt_band[int(post[i])])]
        new_post[i] = post[i]
        for step in range(1, int(pool.size)):
            cand = int(pool[(int(slot[int(post[i])]) + step) % int(pool.size)])
            if cand != int(pre[i]):
                new_post[i] = cand
                break

    out["pre_idx"] = pre.astype(art["pre_idx"].dtype)
    out["post_idx"] = new_post.astype(art["post_idx"].dtype)
    out["n_syn"] = art["n_syn"].copy()
    return out


def band_mass_report(art_seeded, art_wrong):
    """The numbers BEHIND the validity precondition: what the seeded band-pair weight-share matrix
    carries on its diagonal, what fraction of its edges are inter-band, its total asymmetry (the
    quantity that CAPS any transpose-based corruption), the PSI a transpose would have reached, how
    many edges the permutation could not move off their own band, and the PSI the precondition
    actually reads. A precondition that fails owes the asymmetry it measured (2.2); a precondition
    that passes owes it too, or the pass is unattributable."""
    import numpy as np

    from snn_data_lake.cortex.h01_ingest import fan_in_shares

    lay = art_seeded["layer"]
    flat = band_pair_weight_share(
        lay, art_seeded["pre_idx"], art_seeded["post_idx"], fan_in_shares(art_seeded)
    )
    s = np.asarray(flat, dtype=np.float64).reshape(N_BANDS, N_BANDS)
    w = band_pair_weight_share(
        lay, art_wrong["pre_idx"], art_wrong["post_idx"], fan_in_shares(art_wrong)
    )
    diag = float(np.trace(s))
    inter = float(
        (np.asarray(lay)[art_seeded["pre_idx"]] != np.asarray(lay)[art_seeded["post_idx"]]).mean()
    )
    want = np.asarray(BAND_PERMUTATION, dtype=np.int64)[
        np.asarray(lay)[art_seeded["post_idx"]].astype(np.int64)
    ]
    got = np.asarray(lay)[art_wrong["post_idx"]].astype(np.int64)
    return {
        "diagonal_mass": diag,
        "off_diagonal_mass": float(s.sum() - diag),
        "inter_band_edge_fraction": inter,
        "asymmetry": float(np.abs(s - s.T).sum() / 2.0),
        "psi_transpose_vs_seeded": float(
            distance("band_pair_weight_share", tuple(s.T.reshape(-1).tolist()), flat)
        ),
        "off_band_fallback_edges": int((got != want).sum()),
        "psi_wrong_vs_seeded": float(distance("band_pair_weight_share", w, flat)),
    }


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


# distinct-from: cerebellum/spiking_jepa.py::_sparse_init -- the nearest neighbour, and it is a
#   different object: a SPARSE random WEIGHT INITIALISER for a LIF layer, k = SPARSITY*fan_in
#   non-zeros per row drawn U(0.5, 1) and scaled by gain * v_threshold / k so the layer sits at the
#   rate fixed point, with the connectivity mask kept so plasticity stays sparse. It is not
#   orthonormal, it is not a feature map, and it is TRAINED afterwards. What is built here is a
#   DENSE, ORTHONORMAL, FROZEN measurement basis whose only job is to bound the readout's capacity.
# distinct-from: hippocampus/grid_place_map.py::grid_spikes -- a deterministic rate-to-SPIKE
#   sampler over a grid population (Bernoulli draws per cell per tick), not a linear map of counts.
# distinct-from: hippocampus/growable_readout.py::_restore_adam -- optimizer-moment surgery when a
#   torch readout gains a unit; no projection and no draw.
# distinct-from: cortex/neurogenesis_engine.py::grown_model, cortex/placement.py::_place_grn -- a
#   retained-model accessor and a uniform POSITION draw for a non-cortical region. All five read
#   2026-09-09 before this note was written.
def random_projection(n_feat, d, seed):
    """2.12(a) -- the FIXED per-seed random projection of the NON-DRIVEN units' spike counts to D
    dimensions: a Gaussian matrix ORTHONORMALISED (the thin QR's Q, so Q^T Q = I), drawn from the
    seed alone. It is therefore identical for every arm of a seed -- the driven set, and so
    `n_feat`, is a nuisance constant across the arms by construction (2.2).

    WHY IT REPLACED THE RAW COUNT VECTOR. With ~13.5k non-driven units against ~24 training trials
    per block the least-squares readout was massively over-parameterised, so EVERY wiring -- seeded,
    marginal-matched random, reversed -- interpolated the training block exactly and read 1.0000 on
    the held-out trials at every K and every class fraction the pilot swept. D is the readout's
    CAPACITY and it is the quantity the amendment's guard now bounds. A projection to more
    dimensions than there are features has no orthonormal basis and is REFUSED, never padded --
    padding would hand the ladder a rank-deficient D that reads like a measurement."""
    import numpy as np

    n_feat, d = int(n_feat), int(d)
    if d < 1:
        raise ArmRefused(f"a projection to {d} dimensions is not a readout")
    if d > n_feat:
        raise ArmRefused(
            f"a projection to {d} dimensions has no orthonormal basis in {n_feat} features"
        )
    rng = np.random.default_rng(int(seed) * 5_915_587 + 29)
    q, _r = np.linalg.qr(rng.standard_normal((n_feat, d)))
    return np.ascontiguousarray(q[:, :d])


def choose_d(readings):
    """2.12(a)'s capacity guard, applied to the ladder BEFORE any arm is compared: the LARGEST D at
    which the no-learning RANDOM arm scores below the ceiling. `None` means NO D qualifies, and
    that is not a smaller number to fall back on -- it is the row REFUSED ON MEASUREMENT, with the
    finding that this population admits no readout capacity at which its wiring can be separated
    from a marginal-matched null by this protocol. A D with no reading (one that does not fit the
    feature count) never qualifies."""
    under = [
        int(r["d"])
        for r in readings
        if r.get("accuracy") is not None and float(r["accuracy"]) < D_CEILING
    ]
    return max(under) if under else None


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


# distinct-from: substrate/kernel_plasticity.py::KernelPlasticity.__init__ -- built the same three
#   arrays as a SIDE EFFECT of re-pointing every bundle's qweight at a writable view, so a learner
#   could move the table the tick reads. That module is DELETED (2026-09-09); this reads and never
#   re-points, and the tree now has no path that can write a bundle weight at all.
# distinct-from: substrate/lake_engine.py -- the guard's inventory cites a block record's out-edges
#   as `(pre_local, post, delay) -> f4 weight` at L769. GREPPED 2026-09-09 and it is NOT IN THE
#   TREE: lake_engine.py is 561 lines and neither `pre_local` nor that docstring exists anywhere in
#   src. The overlap is with a stale index entry, not with code.
# distinct-from: cortex/heterogeneous_microcircuit.py::run_circuit -- drives a circuit and returns
#   per-neuron SPIKE COUNTS; a dynamics run, not a weight read.
# distinct-from: cortex/predictive_coding.py::set_state and reasoning/resume_learning.py::_forward
#   -- force causes / clear error membranes, and integrate a LIF readout for one window. Both are
#   MEMBRANE state; a synapse table is neither. All four live sites read 2026-09-09.
def edge_table(syn):
    """(edge_pre, edge_post, weight) read out of a compiled `Synapses` bundle table, in the
    bundle-key order the tick itself stores.

    The statistics of 2.4 need an edge list with weights and nothing else. They used to take it
    off `KernelPlasticity.edge_pre / edge_post / weights()`, which assembled exactly these three
    arrays -- but only on its way to re-pointing every bundle at a writable view. With the learner
    deleted the read is all that is left, and it is stated as a read: no view, no copy handed
    back, nothing in this tree that can move the numbers it returns."""
    import numpy as np

    scale = float(1 << int(syn.frac_bits))
    pre, post, qs = [], [], []
    for key in sorted(syn.bundles):
        p, q = syn.bundles[key]
        pre.extend([key[0]] * len(p))
        post.extend(p.tolist())
        qs.append(q)
    flat = np.concatenate(qs).astype(np.int64) if qs else np.zeros(0, dtype=np.int64)
    return (
        np.asarray(pre, dtype=np.int64),
        np.asarray(post, dtype=np.int64),
        flat.astype(np.float64) / scale,
    )


def _reset_population(kt):
    """The per-trial reset of 2.3: membranes, refractory state, spike memory, the delay rings and
    the partition accumulators. The learner's eligibility used to be reset here too, and there is
    none to reset. Weights are NOT reset -- what a trial resets is state, and the weights are the
    thing under measurement."""
    kt.state.arrays["v"][:] = 0.0
    kt.state.arrays["refrac"][:] = 0
    kt.state.arrays["last_spike"][:] = 0
    for ring in kt._rings:
        for slot in ring:
            slot.clear()
    for acc in kt._acc:
        acc[:] = 0


def run_trial(kt, mask, amp_q):
    """One trial: reset, drive for T_trial ticks, collect the spike counts.

    `learn` and the learner it gated are GONE (2026-09-09). Every trial is now what a HELD-OUT
    trial always was -- drive and count, no eligibility, no reward, no state carried -- so the
    training and held-out halves of a block differ only in whether the readout is FIT on them."""
    import numpy as np

    _reset_population(kt)
    n = kt.state.n_units
    counts = np.zeros(n, dtype=np.int64)
    ext = np.zeros(n, dtype=np.int64)
    for tick in range(mask.shape[0]):
        ext[:] = 0
        ext[mask[tick]] = amp_q
        fired = kt.tick(ext)
        counts[fired] += 1
    return counts


# distinct-from: live_arm.py -- the ROBOT arm's stream-to-actuator loop; "arm" there is hardware.
# distinct-from: so101_arm.py -- a contact counterfactual on a never-stepped body; also hardware.
# distinct-from: hjepa_arm_torch.py -- a coarse ACTION batch for the robot arm's trajectory.
# distinct-from: developmental_pruning.py -- runs a prune-and-retrain SCHEDULE and returns a curve.
# distinct-from: meta_plasticity.py -- runs a non-stationary REACH and returns a learning curve.
#   "Arm" below is an ARM OF A COMPARISON: one wiring of one unit set, scored once under a frozen
#   per-trial protocol whose learner is the SUBJECT of the measurement, not its method. All five
#   sites read 2026-09-09 before this note was written.
def run_arm(
    art,
    name,
    seed,
    classes,
    *,
    d,
    blocks=C_BLOCKS,
    t_trial=T_TRIAL,
    trials_per_class=TRIALS_PER_CLASS,
    held_out=HELD_OUT_PER_CLASS,
    threads=1,
):
    """One arm, one seed, the per-trial protocol of 2.3 with its learner removed. -> ArmResult.

    Order per trial: reset -> drive -> collect -> PREDICT with the readout fit on all PREVIOUS
    blocks. Block 0 uses the readout fit on a warm-up block; the split is `marginal_holdout` per
    block. The reward step and the weight update that followed it are GONE (2026-09-09), and with
    them the `lr` parameter: with no learner an `lr` would be a knob that moves nothing, which is
    precisely the kind of call site that lets a deleted mechanism look alive.

    THE STATISTICS ARE STILL TAKEN THREE TIMES -- at construction, at the middle block and at the
    end. Nothing can move them, and that is the point: the DriftRecords are the MEASUREMENT that
    the table did not move, not an assumption from the absence of an import.

    `d` (2.12(a), required, no default): the readout sees the NON-DRIVEN units' spike counts
    through a fixed per-seed orthonormal projection to D dimensions and never sees the raw counts.
    It has no default on purpose -- a default would let a call site silently restore the saturated
    instrument the amendment retired.
    """
    import numpy as np

    from snn_data_lake.cortex.h01_ingest import population
    from snn_data_lake.hippocampus.ng_snn import marginal_holdout

    state, syn, kt = population(art, threads=threads)
    layer = population_layers(art)
    amp_q = int(DRIVE_AMPLITUDE * float(state.arrays["thr"][0]) * (1 << int(syn.frac_bits)))

    seeded_stats = statistics_of(layer, *edge_table(syn))
    mid_stats = seeded_stats

    k = len(classes)
    driven = np.zeros(state.n_units, dtype=bool)
    for cls in classes:
        driven[np.asarray(cls, dtype=np.int64)] = True
    feat_idx = np.nonzero(~driven)[0]
    proj = random_projection(feat_idx.size, d, seed)

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
            mask = trial_drive(classes, cls, seed, trial_id, state.n_units, t_trial)
            trial_id += 1
            counts = run_trial(kt, mask, amp_q)
            feature = counts[feat_idx].astype(np.float64) @ proj
            guess = predict(coef, feature, k)
            if i in val:
                hits += 1
                correct += int(guess == cls)
            else:
                block_x.append(feature)
                block_y.append(cls)
        if hits:
            accuracy = correct / float(hits)
        pool_x.extend(block_x)
        pool_y.extend(block_y)
        coef = fit_readout(pool_x, pool_y, k)
        if block == mid_block:
            mid_stats = statistics_of(layer, *edge_table(syn))

    current = statistics_of(layer, *edge_table(syn))
    return ArmResult(
        name,
        accuracy=accuracy,
        records=drift_records(seeded_stats, current, mid_stats),
        feature_dim=int(proj.shape[1]),
        **NO_LEARNER_DIAGNOSTICS,
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
def verdict(results, wrong_valid, k=K_FIXED):
    """SC-006's five conjuncts, each printed MET / NOT MET / NOT READ with its numbers, plus the
    LEARNER's own disposition, which is independent of the prior's.

    `k` is the number of drive classes, and it is here for ONE reason: a contrast is a DIFFERENCE
    between two arms, and a difference says nothing about wiring unless at least one of the arms
    was doing better than guessing. Every arm therefore carries `arms_vs_chance` -- its mean, the
    1/k floor and `above_chance`'s reading of it -- and every contrast carries `wiring_contrast`,
    which reads NOT INTERPRETABLE when BOTH of its arms fail that check. Both are REPORTS: no
    conjunct state, no verdict word and no bar depends on either (T063 panel of 2026-09-09,
    finding F22)."""
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

    vs_chance = {arm: above_chance(acc[arm], k) for arm in acc}
    cs = (c_p, c_r, c_w, c_rw, c_f)
    #: the two arms each contrast is BETWEEN, named here rather than parsed back out of its label
    pairs = {
        c_p.name: ("planted", "random"),
        c_r.name: ("seeded", "random"),
        c_w.name: ("seeded", "wrong"),
        c_rw.name: ("random", "wrong"),
        c_f.name: ("seeded", "frozen"),
    }

    #: 2026-09-09. Conjunct (a) is NOT READ on every run from here on: drift needs a learner and
    #: there is none. `red1` returns the no-learner refusal (fired is always False) and the note
    #: below carries it verbatim, so the conjunct says WHY it cannot be read instead of failing
    #: quietly as NOT MET -- "the instrument cannot measure this" and "the measurement came out
    #: negative" are different facts, and SC-006's own reading depends on which.
    no_learner_note = red1(results[seeds[0]]["frozen"])[1]
    conj = {}
    if not c_p.met:
        conj["c"] = _con("NOT MET", c_p, "the PLANTED control did not beat RANDOM")
        for key in "abde":
            conj[key] = {"state": "NOT READ", "note": "no verdict is read: the harness is BLIND"}
        out = {
            "conjuncts": conj,
            "sc006": "NO VERDICT",
            "contrasts": _contrasts(cs, pairs, vs_chance),
            "arms_vs_chance": vs_chance,
            "learner": _learner(c_f, results, seeds),
            "per_seed": _per_seed(results, seeds),
            "wrong_valid": bool(wrong_valid),
        }
        return out

    conj["c"] = _con("MET", c_p, "the harness can see an advantage")
    conj["a"] = {"state": "NOT READ", "note": no_learner_note}
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
    #: MET is no longer one of SC-006's readings here. With (a) NOT READ the criterion is NOT MET
    #: whenever a conjunct this instrument CAN still read fails, and NOT READ when every readable
    #: conjunct holds and only the missing drift half stands between the row and its criterion.
    #: Folding the second into the first would hide which of the two happened.
    readable = all(conj[k]["state"] == "MET" for k in "bcde")
    return {
        "conjuncts": conj,
        "sc006": "NOT READ" if readable else "NOT MET",
        "contrasts": _contrasts(cs, pairs, vs_chance),
        "arms_vs_chance": vs_chance,
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


NOT_A_WIRING_CONTRAST = (
    "NOT INTERPRETABLE AS A WIRING CONTRAST (both arms fail the above-chance check)"
)


def _contrasts(cs, pairs, vs_chance):
    return {
        c.name: {
            "t": c.t,
            "fpr": c.fpr,
            "degenerate": c.degenerate,
            "diffs": list(c.diffs),
            "wiring_contrast": _wiring_contrast(pairs[c.name], vs_chance),
        }
        for c in cs
    }


# distinct-from: substrate/wiring_manifest.py::PortManifest.warnings -- reports a DECLARED CONNECTOME PORT that no edge connects (its "two arms" are the IN and OUT directions of that static census); it reads no accuracy, no seed and no bar. Site read 2026-09-10.
def _wiring_contrast(pair, vs_chance):
    """The mark on ONE contrast: a difference between two arms that were both at the floor is a
    difference between two guesses, and it says so and stops there. It decides nothing."""
    a, b = pair
    if not vs_chance[a]["met"] and not vs_chance[b]["met"]:
        return NOT_A_WIRING_CONTRAST
    return "INTERPRETABLE"


def _learner(c_f, results, seeds):
    """2.9's learner criterion, EXECUTED 2026-09-09 and not re-derivable here.

    The rule was: KEPT iff t(d_f) > bar AND the effective change is above zero; DELETED otherwise,
    because a learner that moves weights without an attributed benefit is a null (P8 clauses 3 and
    5, and r_stdp's own docstring rejects weight movement as charge). The campaign read
    t(d_f) = -13.0154 with the FROZEN twin ahead of the learning arm on every seed, and the
    learner was DELETED on it.

    What this reports now is that EXECUTED disposition and its date -- never a fresh verdict. A
    no-learner run's own t(d_f) is a mechanical zero (FROZEN is SEEDED run twice), and returning
    KEPT or DELETED off it would be a criterion evaluated on the absence of its subject. The
    contrast is still carried, because zero is the reading the identity control needs."""
    moved = any(results[s][a].effective_count > 0 for s in seeds for a in results[s])
    return {
        "disposition": "DELETED",
        "t": c_f.t,
        "fpr": c_f.fpr,
        "any_arm_moved": bool(moved),
        "note": (
            f"{NO_LEARNER}. The disposition was executed on the pinned campaign "
            f"(tools/t063_campaign.json), not on this run: t(d_f) = -13.0154, the frozen twin "
            f"ahead of the learning arm on every seed. Any arm moved on THIS run = {moved}, "
            "which must be False -- a True here would mean a learner came back unannounced"
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
        "a": "DRIFT (learning moved the seeded table) -- NO INSTRUMENT since 2026-09-09",
        "b": "BEATS RANDOM",
        "c": "NOT BLIND (the PLANTED control)",
        "d": "RED 2 FIRED (wrong underperforms random) and the WRONG arm was VALID",
        "e": "BEATS WRONG",
    }
    #: which contrast each conjunct is read from, so the wiring mark prints beside its own t
    conjunct_contrast = {
        "b": "seeded - random",
        "c": "planted - random",
        "d": "random - wrong",
        "e": "seeded - wrong",
    }
    for k in "abcde":
        c = v["conjuncts"][k]
        num = ""
        if "t" in c:
            num = f"  t = {c['t']:+.4f}  permutation FPR = {c['fpr']:.4f}"
            if c["degenerate"]:
                num += "  [DEGENERATE: zero-variance differences]"
            cn = v["contrasts"].get(conjunct_contrast.get(k, ""))
            if cn is not None:
                num += f"  [{cn['wiring_contrast']}]"
        lines.append(f"({k}) {labels[k]}: {c['state']}{num}")
        lines.append(f"     {c['note']}")
    lines.append("")
    #: WHERE THE ARMS WERE when those contrasts were read. A difference between two arms says
    #: nothing about wiring unless at least one of them beat the 1/K floor (finding F22).
    lines.append("each arm against chance = 1/K (reported, never a refusal):")
    for arm, a in v["arms_vs_chance"].items():
        lines.append(
            f"  {arm:<8} mean {a['mean']:.4f}  chance {a['chance']:.4f}  {arm} - chance  "
            f"t = {a['t']:+.4f}  permutation FPR = {a['fpr']:.4f}  above chance = {a['met']}"
        )
    lines.append("")
    lo = v["learner"]
    lines.append(
        f"the LEARNER: {lo['disposition']}  (this run's identity contrast t(d_f) = {lo['t']:+.4f}, "
        f"a mechanical zero)"
    )
    lines.append(f"     {lo['note']}")
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
def synthetic_artifact(n_units=64, seed=0, out_degree=8):
    """A population in `h01_ingest`'s artifact format: signed units spread over the seven bands, a
    sparse laminar-biased edge set with integer synapse counts. No network, no lake, no artifact
    file -- the self-check must run anywhere, and an instrument validated only against the thing
    it was built to bless is not an instrument (specs/011 D4).

    `out_degree` is CHOSEN by measurement, not by taste, under one rule: the SMALLEST density at
    which the PLANTED positive control clears the t bar while the base graph's task is still
    genuinely hard. At a low density the fixture's non-driven units have a fan-in of ~2, so the
    class that reaches a unit reaches it alone and EVERY arm's feature vector is a perfect class
    code -- the baseline sits at ceiling and a positive control has nothing to beat.

    IT WAS 4, RE-MEASURED TO 8 ON 2026-09-09 WHEN THE LEARNER WAS DELETED, and both sweeps are
    kept because the difference between them is the finding. WITH the learner, K = 4:
    t(planted - random) = +0.66 at degree 2, +3.53 at 4, +2.82 at 8, +7.48 at 12 -- 4 was the
    smallest that cleared. WITHOUT it, same fixture and the same self-check settings:
    +0.6831 at 2 (random 0.9062), +0.7977 at 4 (random 0.8281), +4.4063 at 8 (random 0.5469),
    +6.8739 at 12 (random 0.2812), +11.0625 at 16 (random 0.2500). Degree 4 NO LONGER REACHES the
    control -- the learner had been supplying part of PLANTED's margin -- and 8 is the smallest
    that does. 12 and 16 are REJECTED though their t is larger: both base arms sit at chance
    (0.25 = 1/K) there, and a positive control that beats a floor is a formality, not a reading.
    Disclosed as a fixture property; it changes nothing about the campaign's population, whose own
    PLANTED contrast was read at t = +9.2601 with the learner and is pinned, not re-run.

    APPENDED 2026-09-10 (T063 panel of 2026-09-09), the paragraph above kept VERBATIM: the closing
    clause is TOO BROAD as written. What it changes nothing about is the PINNED READING -- t =
    +9.2601, taken WITH the learner, not re-run. The out_degree-4 collapse was NOT confined to the
    fixture: the same collapse is present on the CAMPAIGN'S OWN POPULATION, where this positive
    control does not clear its bar without a learner (+0.8143 against the fixture's +0.7977). And
    the floor rule stated three sentences up -- "a positive control that beats a floor is a
    formality, not a reading" -- is not a fixture-only rule and the campaign was never EXEMPT from
    it: the campaign's (c) was read against a RANDOM arm at 1.6x chance, between the rejected
    cells' 1.1x and the accepted cell's 2.2x. Read as a limit on the warrant, not as a
    re-reading of the pin. THE RULE NO LONGER LIVES IN THIS DOCSTRING: it was lifted into the
    module-level `above_chance`, `verdict` reads every arm through it, and this paragraph now
    CITES it rather than being its only statement (P2 -- one mechanism, one place)."""
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
def selfcheck(
    n_units=64, k=4, d=8, t_trial=20, trials_per_class=6, blocks=3, held_out=2, seeds=S_SEEDS
):
    """The bare invocation: the whole pipeline on a synthetic population, under 60 s, no network.

    WHAT IS REAL AND WHAT IS CONSTRUCTED, stated plainly. The NO-LEARNER READING is REAL and is
    taken from the run: every arm on every seed must report four zero diagnostic lines and zero
    distance on both statistics, measured per arm by `no_learner_holds` rather than inferred from
    the absence of an import. The IDENTITY control is REAL: with no learner the FROZEN arm must
    reproduce SEEDED's accuracy exactly on every seed, and a difference would be a determinism
    defect. The PLANTED control is REAL: the contrast is computed from the arms' own accuracies.
    RED 2's PREDICATE is exercised on a CONSTRUCTED accuracy set -- a synthetic table has no
    reason to make a corrupted prior genuinely harmful, and a red must be shown to be reachable
    rather than waited for (P8 clause 1) -- while the WRONG arm's own VALIDITY PRECONDITION is
    measured on the synthetic table for real.

    WHAT IS GONE: red 1. Its subject was a verified lr = 0 pin standing against a learning arm,
    and there is no learning arm. Its slot reads the no-learner refusal, and the control on THAT
    is that the refusal never says PINNED.
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
                d=d,
                blocks=blocks,
                t_trial=t_trial,
                trials_per_class=trials_per_class,
                held_out=held_out,
            )
        results[s] = row

    v = verdict(results, wrong_valid=valid, k=k)
    no_learner_msg = red1(results[0]["frozen"])[1]
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

    #: 2.12(a): the projection is orthonormal and REFUSES a D it has no basis for
    import numpy as np

    q = random_projection(32, 8, seed=0)
    proj_ok = bool(np.allclose(q.T @ q, np.eye(8), atol=1e-9))
    proj_ok = proj_ok and bool(np.array_equal(q, random_projection(32, 8, seed=0)))
    try:
        random_projection(4, 8, seed=0)
        proj_refuses = False
    except ArmRefused:
        proj_refuses = True

    #: 2.12(a): the ladder takes the LARGEST D under the ceiling, and REFUSES the row when none is
    d_guard = bool(
        choose_d([{"d": 4, "accuracy": 0.3}, {"d": 8, "accuracy": 0.8}, {"d": 16, "accuracy": 1.0}])
        == 8
        and choose_d([{"d": x, "accuracy": 1.0} for x in D_CANDIDATES]) is None
    )

    #: 2.12(c): the settling clause fires on a trend and holds on this daemon's duty cycle
    settle_ok = bool(
        is_settled([112.000, 122.906, 106.266, 133.641, 117.828])
        and not is_settled([100.0, 110.0, 120.0])
        and not is_settled([120.0, 110.0, 100.0])
        and not is_settled([100.0, 110.0])
        #: the reading that PASSED before the guard: the daemon died under the probe
        and not is_settled([-515.156, 0.0, 0.0])
        and not is_settled([0.0, 0.0, 0.0])
        and settling_report([("1", 112.0), ("1", 122.9), ("1", 106.3)])["settled"]
        and not settling_report([("1", 112.0), ("2", 122.9), ("2", 106.3)])["settled"]
        and not settling_report([("1", -515.156), ("none", 0.0), ("none", 0.0)])["settled"]
    )

    #: 2.12: the campaign guard refuses on no D and on an invalid WRONG arm
    good = {"D_rule": {"D": 8, "satisfied": True}, "wrong_validity": {"valid": True}, "W": {}}
    no_d = {"D_rule": {"D": None, "satisfied": False}, "wrong_validity": {"valid": True}, "W": {}}
    bad_w = {"D_rule": {"D": 8, "satisfied": True}, "wrong_validity": {"valid": False}, "W": {}}
    guard_ok = bool(
        campaign_guard(good)[0] and not campaign_guard(no_d)[0] and not campaign_guard(bad_w)[0]
    )

    #: 2.12's third disclosed reading: PLANTED may repeat a pair, and nothing else may
    planted_tbl = arm_planted(art, drive_classes(art, k, 0), 0)
    check_nuisance(planted_tbl, art, "planted")
    try:
        check_nuisance(planted_tbl, art, "planted", unique_edges=True)
        planted_repeats = False
    except ArmRefused:
        planted_repeats = True

    seconds = time.perf_counter() - t0
    return {
        #: the no-learner reading, MEASURED on every arm of every seed -- not on the frozen one
        #: alone, because "nothing in the tree can write a bundle weight" is a claim about the
        #: whole run and a per-arm zero is the only evidence for it
        "no_learner": all(no_learner_holds(results[s][a]) for s in results for a in results[s]),
        "no_learner_message": no_learner_msg,
        #: the identity control: with no learner FROZEN is SEEDED run twice
        "frozen_identity": all(
            results[s]["frozen"].accuracy == results[s]["seeded"].accuracy for s in results
        ),
        "red2": red2,
        "planted": bool(planted),
        "wrong_valid": bool(valid),
        "wrong_psi": float(d_wrong),
        "random_psi": float(d_random),
        "wrong_band_masses": band_mass_report(art, arm_wrong(art)),
        "projection": bool(proj_ok),
        "projection_refuses_overcapacity": bool(proj_refuses),
        "d_guard": d_guard,
        "settling_rule": settle_ok,
        "campaign_guard": guard_ok,
        "planted_repeats_a_pair": bool(planted_repeats),
        "feature_dim": int(results[0]["seeded"].feature_dim),
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
    print(f"1 NO-LEARNER reading  : every arm four zero lines and zero drift = {out['no_learner']}")
    print(f"                        {out['no_learner_message']}")
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
    print(
        f"5 identity control    : FROZEN reproduces SEEDED on every seed = {out['frozen_identity']}"
    )
    print(f"6 nuisance refusals   : {out['refusals']}")
    print(f"7 accuracies          : {json.dumps(out['accuracies'])}")
    m = out["wrong_band_masses"]
    print(
        f"8 WRONG band masses   : diagonal {m['diagonal_mass']:.6f} off-diagonal "
        f"{m['off_diagonal_mass']:.6f} inter-band edges {m['inter_band_edge_fraction']:.6f} "
        f"asymmetry {m['asymmetry']:.6f} PSI(transpose) {m['psi_transpose_vs_seeded']:.6f} "
        f"off-band fallbacks {m['off_band_fallback_edges']}"
    )
    print(
        f"9 2.12(a) projection  : orthonormal + fixed = {out['projection']}, refuses D > n_feat = "
        f"{out['projection_refuses_overcapacity']}, feature dim seen by the readout = "
        f"{out['feature_dim']}"
    )
    print(f"10 2.12(a) D guard    : largest-under-ceiling + REFUSES on none = {out['d_guard']}")
    print(f"11 2.12(c) settling   : trend fires, duty cycle does not = {out['settling_rule']}")
    print(f"12 campaign guard     : refuses no-D and an invalid WRONG = {out['campaign_guard']}")
    print(f"13 PLANTED repeats    : permitted for PLANTED alone = {out['planted_repeats_a_pair']}")
    print("=" * 78)
    print(format_verdict(out["verdict"]))
    print("=" * 78)
    assert out["no_learner"], (
        "an arm reported a learner's diagnostics or a nonzero distance -- with "
        "substrate/kernel_plasticity deleted nothing in this tree can move a bundle weight"
    )
    assert "PINNED" not in out["no_learner_message"], (
        "the no-learner refusal must never claim a pin nobody applied"
    )
    assert out["frozen_identity"], (
        "FROZEN did not reproduce SEEDED: with no learner the two are the same run, so a "
        "difference is a DETERMINISM defect in this harness"
    )
    assert out["red2"], "red 2's predicate must be reachable and must not fire on a tie"
    assert out["planted"], (
        "the PLANTED positive control must beat RANDOM -- else the harness is blind"
    )
    assert all(ok for _n, ok in out["refusals"]), (
        f"a nuisance refusal did not fire: {out['refusals']}"
    )
    assert out["projection"], "the projection must be orthonormal and fixed per seed"
    assert out["projection_refuses_overcapacity"], "a projection to D > n_feat must be REFUSED"
    assert out["d_guard"], "the D ladder must take the LARGEST D and refuse the row when none is"
    assert out["settling_rule"], "the settling clause must fire on a trend and hold on a duty cycle"
    assert out["campaign_guard"], "the campaign guard must refuse no-D and an invalid WRONG arm"
    assert out["planted_repeats_a_pair"], (
        "PLANTED must be allowed to repeat a (pre, post) pair, and nothing else may"
    )
    assert out["seconds"] < 60.0, f"the offline self-check took {out['seconds']:.1f}s"
    print(
        f"prior_drift_harness self-check OK in {out['seconds']:.1f}s: every arm reads four zero "
        "diagnostic lines and zero drift on both statistics, the refusal names the deleted learner "
        "and never says PINNED, FROZEN reproduces SEEDED exactly, red 2's predicate is reachable "
        "and does not fire on a tie, the PLANTED control sees an advantage, every nuisance "
        "violation is refused before it can run, the readout sees only an orthonormal "
        "D-dimensional projection, the D ladder refuses the row when no D is under the ceiling, "
        "the settling clause fires on a trend and not on this daemon's duty cycle, and the "
        "campaign guard refuses both a missing D and an invalid WRONG arm"
    )
    return 0


# =============================================================================================
# The pilot (2.7) and the campaign (2.6)
# =============================================================================================
# distinct-from: cortex/h01_ingest.py::_lif_defaults -- imports the LIF constants from their
#   equation owner; no host, no process, no port.
# distinct-from: gates/doc_budget_gate.py::_selfcheck,
#   hippocampus/spiking_storage_prototype.py::run, hypothalamus/__init__ -- a gate's constructed
#   reds, a prototype's returned criteria numbers and a region docstring; none reads a PROCESS.
#   All five read 2026-09-09.
# distinct-from: microcircuit.py -- its `sign_identities` READS a per-Purkinje +1/-1 row from the
#   tissue's own weights; the prose overlap is "READ from" in capitals. This reads a PROCESS's CPU
#   counter off the operating system and decides whether a HOST has settled. Site read 2026-09-09.
def is_settled(windows):
    """2.12(c) -- THE SETTLING CLAUSE, read as the ABSENCE OF A MONOTONE TREND across at least
    three consecutive 90 s CPU windows.

    2.7's literal test (two consecutive windows within ~6 %) is NOT MET and is NOT MEETABLE by this
    daemon in any state: five consecutive windows read 112.0 / 122.9 / 106.3 / 133.6 / 117.8 CPU-s
    per 90 s, which is its ordinary wake/sleep duty cycle rather than a boot transient. What the
    clause is FOR -- no startup burst inflating a timing -- survives that reading, because a burst
    that has not settled shows as a TREND. Fewer than three windows is not a reading and is never
    settled: two windows cannot distinguish a trend from a duty cycle, which is the exact failure
    the original pair test had.

    AND EVERY WINDOW MUST BE STRICTLY POSITIVE. Measured defect, 2026-09-09: the first campaign run
    read [-515.156, 0.0, 0.0] and this function returned TRUE, because a run of zeros after a
    negative is not monotone. The daemon had RESTARTED under the probe, so the CPU counter was
    being differenced against a dead process. A negative CPU delta over 90 s is impossible and a
    zero is impossible for a daemon that holds a drive loop -- both mean the probe lost its
    subject, and a reading of nothing must never read as a settled host. The positive control
    [112.000, 122.906, 106.266] is unchanged by the guard, which is the check that the guard did
    not simply kill the measurement it was added to protect."""
    w = [float(x) for x in windows]
    if len(w) < SETTLING_WINDOWS:
        return False
    if any(x <= 0.0 for x in w):
        return False
    pairs = list(zip(w[:-1], w[1:], strict=True))
    rising = all(b > a for a, b in pairs)
    falling = all(b < a for a, b in pairs)
    return not (rising or falling)


# distinct-from: amdahl_gate.py -- `red_repeatability_broken_arm` PADS a rig's second reading to
#   make a criterion fail on purpose; it constructs a red, it does not read a host.
# distinct-from: so101_arm.py -- `collision` is the robot arm's contact authority inside the
#   SAFETY ENVELOPE. Both sites read 2026-09-09; neither measures a process. This is a host
#   reading: which daemon each CPU window measured, and whether it stayed the same daemon.
def settling_report(samples, seconds=SETTLING_SECONDS):
    """The settling reading with its SUBJECT recorded beside it: `samples` is one (pid, cpu_delta)
    per window. The daemon on this host RESTARTS -- three different PIDs were observed holding 8137
    inside forty minutes, and a fourth swap corrupted the first campaign run's reading -- so WHICH
    process each window measured is part of the reading, not context around it. A run whose subject
    changed or was lost is NOT settled, whatever its numbers look like.

    APPENDED 2026-09-10 (T063 panel of 2026-09-09), the paragraph above kept VERBATIM: "the daemon
    on this host RESTARTS" names the FACT correctly and the arc's record attached a false CAUSE to
    it ("a supervisor cycle of roughly 15 minutes"). There is no timer anywhere. `learn_daemon.bat`
    relaunches only when the daemon EXITS, 8 s later, and `substrate/learn_daemon.py` has exactly
    one `os._exit(0)`, reachable only from `_head() != start_head` -- so a clean exit is
    DEFINITIONALLY a new commit at HEAD, and a session that commits during a probe restarts the
    subject itself. The guard's logic and its red are UNCHANGED: a probe whose subject changed is
    not a reading, whatever moved the subject."""
    pids = [str(p) for p, _d in samples]
    windows = [float(d) for _p, d in samples]
    live = {p for p in pids if p != "none"}
    restarted = bool(len(live) > 1 or "none" in pids or not pids)
    return {
        "cpu_seconds_per_window": windows,
        "daemon_pid_per_window": pids,
        "daemon_restarted": restarted,
        "window_seconds": int(seconds),
        "settled": bool(is_settled(windows) and not restarted),
        "rule": (
            "2.12(c): settled = every window strictly positive, ONE live daemon throughout, and "
            f"NO monotone trend across at least {SETTLING_WINDOWS} windows of {SETTLING_SECONDS} s"
            f"; a run takes {SETTLING_MEASURED} windows, because at the floor of "
            f"{SETTLING_WINDOWS} the monotone test rejects a SETTLED host one time in three "
            "(2 of n! orderings are monotone; 2/120 at five)"
        ),
    }


def _cpu_windows(n=SETTLING_MEASURED, seconds=SETTLING_SECONDS):
    """`n` consecutive `seconds`-long windows of the daemon's own process CPU as (pid, CPU-seconds)
    pairs. Taken IN BAND (2.12(c)) so the settling reading sits BESIDE the timing numbers it
    qualifies instead of in a prose note a later reader has to trust.

    THE LISTENER IS RE-RESOLVED AT BOTH ENDS OF EVERY WINDOW. The first version resolved it ONCE
    and then differenced a cached handle, so when the daemon restarted mid-probe it kept
    subtracting from a corpse and emitted [-515.156, 0.0, 0.0]. Now a window whose owner changed or
    vanished reports `none 0` and `settling_report` marks the whole reading unsettled."""
    import subprocess

    ps = (
        "for ($i = 0; $i -lt " + str(int(n)) + "; $i++) { "
        "$c = Get-NetTCPConnection -LocalPort 8137 -State Listen -ErrorAction SilentlyContinue; "
        "$p = if ($c) { Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue } "
        "else { $null }; "
        "if (-not $p) { 'none 0'; Start-Sleep -Seconds " + str(int(seconds)) + "; continue }; "
        "$a = $p.TotalProcessorTime.TotalSeconds; Start-Sleep -Seconds "
        + str(int(seconds))
        + "; $c2 = Get-NetTCPConnection -LocalPort 8137 -State Listen "
        "-ErrorAction SilentlyContinue; "
        "$p2 = if ($c2 -and $c2.OwningProcess -eq $c.OwningProcess) { "
        "Get-Process -Id $c2.OwningProcess -ErrorAction SilentlyContinue } else { $null }; "
        "if (-not $p2) { 'none 0'; continue }; "
        "'{0} {1:F3}' -f $c.OwningProcess, ($p2.TotalProcessorTime.TotalSeconds - $a) }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=int(n) * int(seconds) + 120,
        ).stdout.split()
        return [(out[i], float(out[i + 1])) for i in range(0, len(out) - 1, 2)]
    except Exception:  # pragma: no cover - the host probe must never fail the run
        return []


def _settling(n=SETTLING_MEASURED, seconds=SETTLING_SECONDS):
    return settling_report(_cpu_windows(n, seconds), seconds)


def _host_condition():
    """The daemon's PID, working set and port state, recorded beside every timing number.

    The host clause's SETTLING half is `_settling` / `is_settled` (2.12(c)), taken IN BAND and
    recorded beside every timing number -- it used to be an out-of-band note in `HOST_NOTE`, which
    left a later reader trusting prose. This function records the part a run can read in a second;
    the two are written side by side into every stage of the pin."""
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

    DATED CORRECTION 2026-09-09: THE LEARNER IS GONE, so the first sentence describes what this
    measured UNTIL the disposition and not what it measures now. The basis is the tick and the
    per-trial reset alone. Disclosed here rather than left as an unexplained shrink -- Codex
    defect 11's point was that a projection excluding the learner UNDER-COUNTS the campaign, so
    any width chosen against this number is chosen against a cheaper workload than the pinned
    campaign's, and the pinned campaign's own W is NOT re-derived from it.

    MIN OF `reps`, not a single reading: the host runs a live daemon whose own duty cycle swings
    its CPU by more than 10 % between 90 s windows, and a single timing on a contended box is the
    instrument defect this project hunts. Each rep runs whole TRIALS with the per-trial reset, so
    the activity level the cost depends on is the campaign's, not a 200-tick free run's."""
    import numpy as np

    from snn_data_lake.cortex.h01_ingest import population

    state, syn, kt = population(art)
    amp_q = int(DRIVE_AMPLITUDE * float(state.arrays["thr"][0]) * (1 << int(syn.frac_bits)))
    masks = [
        trial_drive(classes, i % len(classes), 0, i, state.n_units, t_trial) for i in range(trials)
    ]
    ext = np.zeros(state.n_units, dtype=np.int64)
    best = None
    for _rep in range(reps):
        t0 = time.perf_counter()
        for mask in masks:
            _reset_population(kt)
            for tick in range(t_trial):
                ext[:] = 0
                ext[mask[tick]] = amp_q
                kt.tick(ext)
        dt = (time.perf_counter() - t0) / (trials * t_trial)
        best = dt if best is None else min(best, dt)
    return best


def _campaign_ticks(k, blocks=C_BLOCKS, trials_per_class=TRIALS_PER_CLASS, t_trial=T_TRIAL):
    """5 tables x 8 seeds x (1 + C) blocks x K x trials x T_trial ticks (2.7's own formula)."""
    return 5 * S_SEEDS * (1 + blocks) * k * trials_per_class * t_trial


def campaign_guard(stage):
    """2.12's authorization, made mechanical: the campaign runs IFF the ladder selected a D AND the
    WRONG arm is VALID AND a width came in under the projection ceiling. Returns (ok, reason).

    A refusal is a FINDING carrying numbers, never a blocked chore. No D means this population
    admits no readout capacity at which its wiring separates from a marginal-matched null under
    this protocol; an invalid WRONG arm means red 2 has no reachable reading on this connectome."""
    d_rule = stage.get("D_rule") or {}
    valid = (stage.get("wrong_validity") or {}).get("valid")
    if not d_rule.get("satisfied") or d_rule.get("D") is None:
        return False, (
            "ROW REFUSED ON MEASUREMENT: no D in the ladder put the no-learning RANDOM arm under "
            f"the {D_CEILING} ceiling, so this population admits no readout capacity at which its "
            "wiring can be separated from a marginal-matched null by this protocol"
        )
    if not valid:
        return False, (
            "CAMPAIGN REFUSED: the WRONG arm is INVALID -- its band-pair weight-share matrix is "
            "not farther from SEEDED's than RANDOM's realised matrix is, so red 2 has no reachable "
            "reading and 2.12 authorizes the campaign only on a VALID wrong arm"
        )
    if stage.get("W") is None:
        return False, "CAMPAIGN REFUSED: no width came in under the projection ceiling"
    return True, "the ladder selected a D, the WRONG arm is valid, and a width fits the ceiling"


# distinct-from: lake_bridge.py -- `iter_records` streams durable payloads; home.py -- the
#   _T_SELFBUILD append-only sidecar; lake_engine.py -- the retention census. All three are the
#   BRAIN'S DURABLE LAKE. This reads a JSON evidence pin in `tools/`, which is never a lake record.
# distinct-from: theory_of_mind.py -- `place` updates observers' beliefs; arm_config.py -- the
#   robot arm's calibration constants. Prose-level overlap only. Five sites read 2026-09-09.
def _pin_stages():
    """Every pilot stage ever written, earliest first. The pre-amendment pilot was written as a
    FLAT document; it is lifted into the list under its own stage key and never erased -- 2.12
    keeps its readings in the record beside the re-run's, so the pin GROWS and is not overwritten.
    """
    if not PILOT_PIN.exists():
        return []
    doc = json.loads(PILOT_PIN.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "stages" in doc:
        return list(doc["stages"])
    doc = dict(doc)
    doc.setdefault("stage", "2026-09-09 pre-amendment -- 2.7's K rule (UNSATISFIED)")
    return [doc]


PILOT_STAGE = "2026-09-09 amended -- 2.12's D ladder and the band-permuted WRONG arm"


def run_pilot():
    """2.12's pilot: the D ladder at K = 8, the drive-only control re-measured at the chosen D, the
    WRONG arm's validity precondition with the band masses behind it, the tick cost with the
    learner ON, W by the 6-hour rule, and the settling reading beside every timing. APPENDED to
    `tools/t063_pilot.json` as a new stage; the pre-amendment stage is never erased."""
    from snn_data_lake.cortex.h01_ingest import ARTIFACT, load_artifact

    host = _host_condition()
    print("=" * 78)
    print(f"T063 pilot (2.12) -- host {json.dumps(host)}")
    print(f"settling: taking {SETTLING_MEASURED} x {SETTLING_SECONDS} s CPU windows...")
    settling = _settling()
    print(
        f"settling: windows {settling['cpu_seconds_per_window']} CPU-s on PIDs "
        f"{settling['daemon_pid_per_window']} -> settled = {settling['settled']} "
        f"(daemon_restarted = {settling['daemon_restarted']})"
    )
    print("=" * 78)
    art = load_artifact()
    art_sha = hashlib.sha256(Path(ARTIFACT).read_bytes()).hexdigest()
    print(
        f"artifact {ARTIFACT.name} sha256 {art_sha}  units {art['agglo_id'].size} "
        f"edges {art['pre_idx'].size}"
    )

    # ---- 2.12(a): D = the LARGEST of the ladder at which no-learning RANDOM scores < 0.9 --------
    k = K_FIXED
    classes = drive_classes(art, k, 0)
    table_random = arm_random(art, 0)
    check_nuisance(table_random, art, "random", unique_edges=True)
    n_feat = int(art["agglo_id"].size) - sum(len(c) for c in classes)
    d_readings = []
    for d in D_CANDIDATES:
        if d > n_feat:
            d_readings.append({"d": d, "accuracy": None, "note": f"D > {n_feat} features"})
            print(f"D ladder: D = {d:>3}  DOES NOT FIT ({n_feat} non-driven units)")
            continue
        r = run_arm(table_random, "random", 0, classes, d=d, blocks=1)
        d_readings.append({"d": d, "accuracy": r.accuracy})
        print(f"D ladder: D = {d:>3}  no-learning RANDOM held-out accuracy = {r.accuracy:.4f}")
    d_chosen = choose_d(d_readings)
    if d_chosen is None:
        print(
            f"D RULE UNSATISFIED: no D in {D_CANDIDATES} put the no-learning RANDOM arm under the "
            f"{D_CEILING} ceiling. THE ROW IS REFUSED ON MEASUREMENT."
        )
    else:
        print(f"D = {d_chosen}: the LARGEST D at which RANDOM sits under the {D_CEILING} ceiling")
    d_basis = d_chosen if d_chosen is not None else min(D_CANDIDATES)

    # ---- T003's required measurement, RE-MEASURED at the chosen D ------------------------------
    drive_only = run_arm(arm_no_edges(art), "drive-only", 0, classes, d=d_basis, blocks=1)
    seeded_pilot = run_arm(arm_seeded(art), "seeded", 0, classes, d=d_basis, blocks=1)
    chance = 1.0 / k
    print(
        f"T003 drive-only at D = {d_basis}: accuracy {drive_only.accuracy:.4f} against chance "
        f"{chance:.4f}; the same protocol WITH edges (SEEDED) reads {seeded_pilot.accuracy:.4f}"
    )

    # ---- 2.12(b): the WRONG arm's validity precondition, and the band masses behind it ---------
    wrg = arm_wrong(art)
    check_nuisance(wrg, art, "wrong")
    valid, psi_wrong, psi_random = wrong_validity(wrg, table_random, art)
    masses = band_mass_report(art, wrg)
    print(
        f"T004 validity: PSI(wrong, seeded) {psi_wrong:.6f} vs PSI(random, seeded) "
        f"{psi_random:.6f} -> WRONG is {'VALID' if valid else 'INVALID'}"
    )
    print(
        f"           band masses: diagonal {masses['diagonal_mass']:.6f} off-diagonal "
        f"{masses['off_diagonal_mass']:.6f} inter-band edges "
        f"{masses['inter_band_edge_fraction']:.6f} asymmetry {masses['asymmetry']:.6f} "
        f"PSI(transpose) {masses['psi_transpose_vs_seeded']:.6f} off-band fallbacks "
        f"{masses['off_band_fallback_edges']}"
    )

    # ---- the tick cost per width, learner ON, and the 6-hour projection ------------------------
    total_ticks = _campaign_ticks(k)
    widths = []
    for frac in WIDTH_FRACTIONS:
        sub = band_balanced_subgraph(art, frac)
        try:
            sub_classes = drive_classes(sub, k, 0)
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
                "settled": settling["settled"],
                "cpu_seconds_per_window": settling["cpu_seconds_per_window"],
            }
        )
        print(
            f"width {frac:<6} units {sub['agglo_id'].size:>6} edges {sub['pre_idx'].size:>6}  "
            f"{sec * 1000:.4f} ms/tick  -> {hours:.3f} h for {total_ticks} ticks  "
            f"[settled {settling['settled']}]"
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

    stage = {
        "stage": PILOT_STAGE,
        "task": "specs/022 T063 pilot",
        "git_head": _git_head(),
        "harness_sha256": _self_sha(),
        "artifact": str(ARTIFACT.name),
        "artifact_sha256": art_sha,
        "host": host,
        "host_note": HOST_NOTE,
        "settling": settling,
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
            "K": K_FIXED,
            "band_permutation": list(BAND_PERMUTATION),
        },
        "D_rule": {
            "ceiling": D_CEILING,
            "candidates": list(D_CANDIDATES),
            "readings": d_readings,
            "D": d_chosen,
            "satisfied": d_chosen is not None,
            "D_basis": d_basis,
            "n_non_driven_features": n_feat,
            "note": (
                "SATISFIED: D is the LARGEST candidate at which the no-learning RANDOM arm sits "
                "under the ceiling"
                if d_chosen is not None
                else (
                    "UNSATISFIED: no candidate D put the no-learning RANDOM arm under the "
                    "ceiling. The row is REFUSED ON MEASUREMENT -- this population admits no "
                    "readout capacity at which its wiring separates from a marginal-matched null "
                    "under this protocol. The items below were measured on D_basis as a workload "
                    "basis only"
                )
            ),
        },
        "drive_only": {
            "accuracy": drive_only.accuracy,
            "chance": chance,
            "D": d_basis,
            "seeded_same_protocol": seeded_pilot.accuracy,
        },
        "wrong_validity": {
            "valid": bool(valid),
            "psi_wrong": psi_wrong,
            "psi_random": psi_random,
            "band_masses": masses,
        },
        "campaign_ticks": total_ticks,
        "widths": widths,
        "W": w_chosen,
    }
    stage["campaign_authorized"], stage["campaign_reason"] = campaign_guard(stage)
    stages = [s for s in _pin_stages() if s.get("stage") != PILOT_STAGE] + [stage]
    PILOT_PIN.parent.mkdir(parents=True, exist_ok=True)
    PILOT_PIN.write_text(
        json.dumps({"task": "specs/022 T063 pilot", "stages": stages}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"pilot stage {len(stages)} pinned to {PILOT_PIN}")
    print(f"campaign authorized = {stage['campaign_authorized']}: {stage['campaign_reason']}")
    if not stage["campaign_authorized"]:
        return 3
    return 0 if w_chosen is not None else 2


def run_campaign():
    """Five tables x 8 seeds at the pilot's W, every number into `tools/t063_campaign.json`."""
    from snn_data_lake.cortex.h01_ingest import ARTIFACT, load_artifact

    if not PILOT_PIN.exists():
        print(
            "CAMPAIGN REFUSED: run --pilot first; W and D are read from its pin, never chosen here"
        )
        return 1
    stages = _pin_stages()
    if not stages:
        print("CAMPAIGN REFUSED: the pin holds no stage")
        return 1
    pin = stages[-1]
    ok, reason = campaign_guard(pin)
    if not ok:
        print(reason)
        return 1
    k = int(pin["constants"]["K"])
    d = int(pin["D_rule"]["D"])
    frac = float(pin["W"]["frac"])
    host = _host_condition()
    print("=" * 78)
    print(f"T063 campaign -- K = {k}, D = {d}, W = {frac}, host {json.dumps(host)}")
    print(f"settling: taking {SETTLING_MEASURED} x {SETTLING_SECONDS} s CPU windows...")
    settling = _settling()
    print(
        f"settling: windows {settling['cpu_seconds_per_window']} CPU-s on PIDs "
        f"{settling['daemon_pid_per_window']} -> settled = {settling['settled']} "
        f"(daemon_restarted = {settling['daemon_restarted']})"
    )
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
        validity[s] = {
            "valid": bool(v_ok),
            "psi_wrong": d_wrong,
            "psi_random": d_random,
            "band_masses": band_mass_report(art, tables["wrong"]),
        }
        valid = v_ok if valid is None else (valid and v_ok)
        row = {}
        for name, table in tables.items():
            r0 = time.perf_counter()
            row[name] = run_arm(table, name, s, classes, d=d)
            print(
                f"seed {s} {name:<8} acc {row[name].accuracy:.4f}  "
                f"exec {row[name].executed} att {row[name].attempted} "
                f"eff {row[name].effective_count} L1 {row[name].effective_l1:.6f}  "
                f"[{time.perf_counter() - r0:.1f}s]"
            )
        results[s] = row
    wall = time.perf_counter() - t0
    #: the host is read again AFTER the arms: on this box the daemon restarts, and a run whose
    #: daemon changed under it is a fact the record has to carry rather than a start-of-run
    #: snapshot that quietly stops being true.
    host_end = _host_condition()

    v = verdict(results, wrong_valid=valid, k=k)
    print("=" * 78)
    print(format_verdict(v))
    print("=" * 78)
    print(f"wall clock {wall:.1f}s ({wall / 3600.0:.3f} h)")
    print(
        f"host at end {json.dumps(host_end)}  daemon changed during the run = "
        f"{host.get('daemon_pid') != host_end.get('daemon_pid')}"
    )

    out = {
        "task": "specs/022 T063 campaign",
        "git_head": _git_head(),
        "harness_sha256": _self_sha(),
        "artifact": str(ARTIFACT.name),
        "artifact_sha256": hashlib.sha256(Path(ARTIFACT).read_bytes()).hexdigest(),
        "host": host,
        "host_end": host_end,
        "host_daemon_changed": host.get("daemon_pid") != host_end.get("daemon_pid"),
        "host_note": HOST_NOTE,
        "settling": settling,
        "pilot_stage": pin.get("stage"),
        "K": k,
        "D": d,
        "W_frac": frac,
        "W_units": int(art["agglo_id"].size),
        "W_edges": int(art["pre_idx"].size),
        "wrong_validity_per_seed": validity,
        "diagnostics": {
            s: {
                a: {
                    "executed": results[s][a].executed,
                    "attempted": results[s][a].attempted,
                    "effective_count": results[s][a].effective_count,
                    "effective_l1": results[s][a].effective_l1,
                    "feature_dim": results[s][a].feature_dim,
                }
                for a in results[s]
            }
            for s in results
        },
        "verdict": v,
        #: RENAMED from `red1_messages` on 2026-09-09. The key held red 1's reading, and red 1's
        #: subject is deleted; a key whose name says "red 1 fired or did not" over a no-learner
        #: refusal is a name that is not what it measures. The PINNED campaign is not edited and
        #: keeps its own `red1_messages` -- the two pins differ because the instrument did.
        "no_learner_refusals": {s: red1(results[s]["frozen"])[1] for s in results},
        "no_learner_holds": {
            s: {a: no_learner_holds(results[s][a]) for a in results[s]} for s in results
        },
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
