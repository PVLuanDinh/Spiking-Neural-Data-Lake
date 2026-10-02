r"""
Self-improvement SUITE -- compose every self-improvement RUNG into ONE program.

WHAT THIS IS (read this first, honestly). The brain has grown a whole LADDER of
self-improvement rungs, each shipped as its own module with its own heavyweight,
un-riggable __main__ self-check (controls, honest-negatives, the full quantitative
proof). Those live as disconnected demo files that the self-check gate runs one at a
time. (This read "the 198-check gate" until 2026-08-04. The gate has not had 198 checks
for a long time -- an unreproducible count in a docstring is a stale citation that
happens to have no filename in it, so the number is dropped rather than re-quoted at a
figure that will rot again.) This module is the missing COMPOSITION layer: a single deterministic
`census(seed=0) -> dict` that exercises the REAL code path of each rung at REDUCED
SCALE, in-process, under ONE seed -- so the unified Brain can run and report them all
as ONE program instead of sixteen separate demos.

WHAT THIS IS NOT (do not overclaim). This is a LIGHTWEIGHT LIVE CENSUS, not a proof.
Every rung here runs on small budgets/steps chosen so the whole census finishes in a
couple of seconds. The reduced-scale numbers are NOT the headline results and MUST NOT
be presented as such: the FULL quantitative verification (the controls that must fail,
the honest negatives, the seed sweeps, the wide-margin thresholds) lives in each
module's own self-check, run by the gate. The single defensible claim of THIS file is
COMPOSITION: that all eight rungs load, run in one process under one seed, and return a
finite number without any one rung's failure taking down the brain (each is guarded).

The one EXACT invariant that must hold even at reduced scale: rung 5's
function-preserving growth is a bit-for-bit eps-invariant, so `growth_exact_diff` must
be exactly 0.0 here too (the self-check asserts it).

THE EIGHT RUNGS (each a REAL, guarded exercise of the actual module):
  1. L5  generalizing_novelty.GeneralizingNovelty  -> novelty_generalization (distinct SimHash buckets)
  2. L8  meta_plasticity.MetaPlasticLearner        -> metaplast_gain (post-flip full vs both-off)
  3. L11 benna_fusi.BennaFusiLearner               -> consolidation_retention (A-retention after A->B->rest)
  4. L9  drive_reduction.DriveReductionLearner      -> drive_regulation (mean total drive, lower=better)
  5. L6  function_preserving_growth.SpikingGrowthLayer -> growth_exact_diff (must be 0.0)
  6. L11 sleep_consolidation.run                    -> sleep_retention_gap (sleep - no-sleep ret_a)
  7. L9  autotelic_goals (collect_labels/readout)   -> goal_reach (held-out goal-reaching rate)
  8. L12 quality_diversity.QualityDiversityArchive  -> qd_coverage (behavior cells filled)

VALUE / REPRESENTATION ONLY. This module imports NOTHING from brainstem/ or the arm and
emits no motor command; every rung is a pure value/representation exercise. The __main__
self-check asserts that by scanning this file's own import lines.

Determinism & style invariants (per CLAUDE.md): per-instance random.Random(seed);
no global RNG; no wall-clock inside census() (the __main__ runtime line uses perf_counter for
DISPLAY ONLY and is never part of the returned census); ASCII-only prints; deterministic.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.self_improvement_suite
"""

import math
import random
import time

from snn_data_lake.cortex.autotelic_goals import (
    EVAL_STARTS,
    HELDOUT_GOALS,
    REACH_HI,
    REACH_LO,
    GoalConditionedReadout,
    H,
    _reach_rate,
    collect_labels,
)
from snn_data_lake.cortex.quality_diversity import (
    DPOS,
    K_NEURONS,
    QualityDiversityArchive,
    behavior_descriptor,
)
from snn_data_lake.faculties.drive_reduction import (
    DriveReductionLearner,
    NeedPool,
    _regulate,
)
from snn_data_lake.faculties.generalizing_novelty import GeneralizingNovelty
from snn_data_lake.growth.function_preserving_growth import (
    SpikingGrowthLayer,
    _max_abs_diff,
)
from snn_data_lake.growth.lif_neuron import LIFNeuron
from snn_data_lake.plasticity.benna_fusi import BennaFusiLearner
from snn_data_lake.plasticity.meta_plasticity import MetaPlasticLearner
from snn_data_lake.sensorimotor.population_encoding import PopulationEncoder
from snn_data_lake.sleep.sleep_consolidation import run as sleep_run

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_rung_metaplast": (
        "NOT_A_RULE",
        "runs one experiment rung -- the non-stationary reach at a small budget with both metaplastic "
        "mechanisms on. An experiment harness reporting a number",
    ),
}

# The failure sentinel: chosen OUT of the legitimate range of every rung metric (distinct
# buckets 1..8, gains/gaps in [-1,1], retentions/rates in [0,1], drive >= 0, coverage >= 0),
# so a real value can never be mistaken for a failure and vice-versa.
SENTINEL = -999.0

# Reduced-scale census budgets (small on purpose -- this is a LIVE census, NOT the full proof).
_NOV_DIM = 64  # raster width (matches the LIF raster the real self-check uses)
_NOV_BITS = 16  # SimHash bits
_MP_WARMUP = 20  # meta-plasticity: short warmup ...
_MP_EPOCH = 10  # ... short epoch ...
_MP_EPOCHS = 2  # ... few flips
_MP_WIN = 8  # short re-adaptation window
_BF_A = 150  # benna-fusi: A-training ticks (enough for A to seep into the deep beakers)
_BF_B = 15  # B-training ticks (the conflicting overwrite)
_BF_RELAX = 50  # rest ticks before the A re-test
_DR_CELLS = 8  # drive-reduction NEED pool cells (small -> cheap spiking readout)
_DR_STEPS = 20  # LIF integration steps per NEED-pool rate readout
_DR_EPISODES = 3  # short training episodes ...
_DR_EPSTEPS = 8  # ... few steps each
_DR_EVAL = 120  # eval rollout steps (no LIF -> cheap); tail after warmup=50 is the metric
_GROW_WIN = 12  # growth: input-window length
_GOAL_EPISODES = 6  # autotelic: FEW babbling episodes
_QD_BUDGET = 60  # quality-diversity: small evaluation budget
_QD_INIT = 20  # random seed policies before mutation begins


def _rung_novelty(seed):
    """L5 -- feed ~8 rasters (4 bases + a 1-bit near-duplicate of each) through the real
    SimHash pseudo-count and report how many DISTINCT buckets they occupy. A pure exercise
    of GeneralizingNovelty.bonus()/.bucket(); the number is finite by construction."""
    rng = random.Random(seed)
    nov = GeneralizingNovelty(n_bits=_NOV_BITS, dim=_NOV_DIM, seed=seed)
    rasters = []
    for _ in range(4):
        base = [rng.randint(0, 1) for _ in range(_NOV_DIM)]
        dup = list(base)
        dup[rng.randrange(_NOV_DIM)] ^= 1  # a near-duplicate (single-bit flip)
        rasters.append(base)
        rasters.append(dup)
    buckets = set()
    for r in rasters:
        nov.bonus(r)  # observe (increment the bucket count)
        buckets.add(nov.bucket(r))
    return float(len(buckets))


def _rung_metaplast(seed):
    """L8 -- run the real non-stationary reach at a SMALL budget with both metaplastic
    mechanisms on (full) vs both off, and report the post-flip competence delta."""
    full = MetaPlasticLearner(lr_gate=True, bcm_meta=True, seed=seed).run(
        warmup=_MP_WARMUP, epoch=_MP_EPOCH, n_epochs=_MP_EPOCHS, adapt_window=_MP_WIN
    )
    both_off = MetaPlasticLearner(lr_gate=False, bcm_meta=False, seed=seed).run(
        warmup=_MP_WARMUP, epoch=_MP_EPOCH, n_epochs=_MP_EPOCHS, adapt_window=_MP_WIN
    )
    return float(full["post_acc"] - both_off["post_acc"])


def _rung_consolidation(seed):
    """L11 -- a short sequential-interference run on the real Benna-Fusi cascade:
    train mapping A, train the conflicting B, rest, re-test A. Report A-retention."""
    rng = random.Random(seed)
    a = [1 if rng.random() < 0.5 else -1 for _ in range(24)]
    b = [-x for x in a]
    learner = BennaFusiLearner(mode="cascade", seed=seed)
    for _ in range(_BF_A):
        learner.train_tick(a)
    for _ in range(_BF_B):
        learner.train_tick(b)
    learner.relax(_BF_RELAX)
    return float(learner.accuracy(a))


def _rung_drive(seed):
    """L9 -- a short regulate episode on the real drive-reduction learner over a LIF NEED
    pool; report the mean total drive over an eval rollout (lower == better regulation)."""
    pool = NeedPool(n_cells=_DR_CELLS, steps=_DR_STEPS, seed=seed)
    learner = DriveReductionLearner(
        pool, reward_kind="reduction", elig_kind="spiking", seed=seed
    ).train(_DR_EPISODES, _DR_EPSTEPS)
    return float(_regulate(learner.policy, [1.0, 2.0, 3.0], _DR_EVAL))


def _rung_growth(seed):
    """L6 -- run a window through the real spiking growth layer, duplicate a unit with a
    halved fan-out, re-run, and report the max-abs current difference. The eps-invariant
    says this is EXACTLY 0.0 even at reduced scale."""
    rng = random.Random(seed)
    window = [[rng.uniform(0.0, 1.0) for _ in range(6)] for _ in range(_GROW_WIN)]
    base = SpikingGrowthLayer(seed=seed)
    cur0, _out0, _hid0 = base.run(window)
    grown = SpikingGrowthLayer(seed=seed)  # identical weights (same seed)
    grown.duplicate_unit(0, factor=0.5)  # function-preserving widen
    cur1, _out1, _hid1 = grown.run(window)
    return float(_max_abs_diff(cur0, cur1))


def _rung_sleep(seed):
    """L11 -- the real live wake/quiescent sleep-consolidation experiment with the sleep
    knob ON vs OFF; report the old-skill retention gap (sleep - no-sleep)."""
    sl = sleep_run(seed, sleep=True)
    ns = sleep_run(seed, sleep=False)
    return float(sl["ret_a"] - ns["ret_a"])


def _rung_goal(seed):
    """L9 -- FEW autotelic goal-babbling episodes with hindsight relabel, train the real
    goal-conditioned spiking readout, and report the held-out goal-reaching rate (the
    frozen, never-relabeled metric hindsight cannot inflate)."""
    examples, _buf = collect_labels(_GOAL_EPISODES, hindsight=True, seed=seed)
    readout = GoalConditionedReadout(seed=seed)
    readout.train(examples)
    return float(_reach_rate(readout, EVAL_STARTS, HELDOUT_GOALS, H, REACH_LO, REACH_HI))


def _rung_qd(seed):
    """L12 -- a small MAP-Elites illumination run over the real spiking-policy quality-
    diversity archive; report how many BEHAVIOR cells were filled (coverage)."""
    enc = PopulationEncoder(n=DPOS, lo=0.0, hi=1.0, max_spikes=10)
    neurons = [LIFNeuron(tau_m=10.0, v_threshold=1.0) for _ in range(K_NEURONS)]

    def beh(
        policy, _enc=enc, _neu=neurons
    ):  # policy -> behavior descriptor (rollout settling point)
        return behavior_descriptor(policy, _enc, _neu)

    def obj(policy):  # within-niche quality (prefer the up-right corner)
        x, y = beh(policy)
        return x + y

    archive = QualityDiversityArchive(beh, obj, beh, seed=seed)
    archive.run(_QD_BUDGET, n_init=_QD_INIT)
    return float(len(archive.behavior_seen))


# Ordered rung registry: metric name -> the guarded exercise that produces it.
_RUNGS = (
    ("novelty_generalization", _rung_novelty),
    ("metaplast_gain", _rung_metaplast),
    ("consolidation_retention", _rung_consolidation),
    ("drive_regulation", _rung_drive),
    ("growth_exact_diff", _rung_growth),
    ("sleep_retention_gap", _rung_sleep),
    ("goal_reach", _rung_goal),
    ("qd_coverage", _rung_qd),
)


def census(seed=0):
    """Run every self-improvement rung ONCE at reduced scale under a single seed and return
    a metric dict. Each rung is GUARDED: any exception -> the SENTINEL, so one rung can never
    take down the brain. Deterministic (per-instance random.Random(seed) throughout)."""
    seed = int(seed)
    out = {}
    for name, fn in _RUNGS:
        try:
            out[name] = float(fn(seed))
        except Exception:  # noqa: BLE001 -- one rung must never break the brain
            out[name] = SENTINEL
    return out


class SelfImprovementSuite:
    """Thin wrapper around census() so the unified Brain can hold and run the whole
    self-improvement ladder as one object. census() returns the reduced-scale metric dict."""

    def __init__(self, seed=0):
        self.seed = int(seed)

    def census(self):
        """The reduced-scale live census of every self-improvement rung (see census())."""
        return census(self.seed)


def main():
    t0 = time.perf_counter()  # DISPLAY-ONLY wall-clock (never part of the census)
    result = census(0)
    elapsed = time.perf_counter() - t0

    print("=" * 78)
    print("SELF-IMPROVEMENT SUITE -- one deterministic census over every rung (reduced scale)")
    print("=" * 78)
    print("HONEST: this is a LIGHTWEIGHT LIVE census -- each rung's REAL code path at reduced")
    print("scale, proving they COMPOSE into one program under one seed. The reduced-scale")
    print("numbers are NOT headline results; the full quantitative proof (controls, honest-")
    print("negatives) lives in each module's own __main__, run by the self-check gate.")
    print("-" * 78)
    for name, _fn in _RUNGS:
        print(f"  {name:26s} = {result[name]!r}")
    print("-" * 78)

    # (1) COMPOSITION: every rung returned a FINITE real number and none is the failure
    #     sentinel -> all eight self-improvement rungs load + run in one process under one seed.
    for name, _fn in _RUNGS:
        v = result[name]
        assert isinstance(v, float) and math.isfinite(v), (
            f"{name} must be a finite real number ({v!r})"
        )
        assert v != SENTINEL, f"{name} hit the failure sentinel -- the rung did not compose/run"

    # (2) EXACT INVARIANT: function-preserving growth is bit-for-bit even at reduced scale.
    assert result["growth_exact_diff"] == 0.0, (
        f"growth must preserve output EXACTLY (diff {result['growth_exact_diff']!r} != 0.0)"
    )

    # (3) DETERMINISM: a second census under the same seed is byte-identical.
    assert census(0) == result, "census(0) must be byte-identical across calls (deterministic)"

    # (4) NO-ACTUATOR: this module's own import lines pull nothing from the actuator / arm /
    #     safety-envelope package -- a value/representation-only suite (scanned, un-riggable).
    with open(__file__, encoding="utf-8") as fh:
        import_lines = [
            ln
            for ln in fh.read().splitlines()
            if ln.startswith("import ") or ln.startswith("from ")
        ]
    forbidden = ("brainstem", "serial_arm", "so101_arm", "arm_config")
    hits = [tok for tok in forbidden if any(tok in ln for ln in import_lines)]
    assert hits == [], f"suite must import NOTHING from brainstem/arm (found {hits})"

    print(f"(1) COMPOSITION : all {len(_RUNGS)} rungs returned a finite non-sentinel number")
    print("(2) EXACT       : growth_exact_diff == 0.0 (eps-invariant holds at reduced scale)")
    print("(3) DETERMINISM : census(0) == census(0) byte-identical")
    print(f"(4) NO-ACTUATOR : forbidden-import hits={hits} (value/representation only)")
    print(f"total census runtime = {elapsed:.3f}s (display only; not part of the census)")
    print("=" * 78)
    print(
        f"suite OK: {len(_RUNGS)} self-improvement rungs compose into ONE deterministic census "
        f"under one seed; growth exact (0.0); no actuator; runtime {elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
