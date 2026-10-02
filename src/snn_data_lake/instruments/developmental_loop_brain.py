r"""
BRAIN-INTEGRATED DEVELOPMENTAL LOOP (Slice 1) -- the deep developmental loop (v3.54) with its
hand-rolled parts replaced by the brain's REAL subsystems, wired through thin adapters.

  perception    : conv1 latency-STDP (unsupervised, frozen)          [instruments/deep_stdp_latency, promoted 2026-08-11]
  memory        : HippocampalMemory  = real ReplayBuffer + SleepConsolidator   [hippocampus/, promoted 2026-08-11]
  readout       : GrowableReadout    = MLP grown function-preservingly (Net2Net) [hippocampus/, promoted 2026-08-11]
  decision      : SpikingCuriosity + SpikingCurriculum + BrainDecision (LatentQ value + Metacog; trains and
                  monitors -- the planner SELECTS since the 2026-08-12 EXEC-bypass adjudication below) [basal_ganglia/]
  neurogenesis  : readout.grow() fired by the MetacognitiveMonitor grow_trigger  [the brain grows its own readout neurons]
  neuromod      : LocusCoeruleus (readout lr gain)                    [neuromodulation]
  executive     : PrefrontalExecutive.plan over CompetenceDynamics + goal_value (the competence-domain
                  objects public in cortex/competence_executive since the 2026-08-12 retirement of the
                  ExecutivePlanner/WorkspaceBroadcast wrappers) + DopamineModulator; the goal broadcast
                  is a GlobalWorkspace this loop OWNS (private offline bus: submit + step + verdict).
                  All imported lazily inside main() and _mechanics_check().

vs v3.54 (developmental_loop_deep.py, RETIRED 2026-08-06 -- superseded by this file, which carries
its control flow; its reference number survives as DEEP_LOOP_REF below): the loop's AUTONOMY + accuracy path is now driven by the actual
brain modules (hippocampal replay/consolidation, basal-ganglia value, metacognitive control, function-
preserving neurogenesis) instead of a python replay dict + a plain nn.Linear + curiosity alone. Feature
learning stays UNSUPERVISED (frozen conv1); only the readout is supervised (the accepted thesis tradeoff).

HONEST GOAL: this is an ARCHITECTURAL-FAITHFULNESS upgrade, not an accuracy lever -- accuracy is readout-
capped (~95% joint, ~93.5% continual, v3.53/54). The honest test is whether routing the loop through the
real brain subsystems HOLDS ~93.5% or COSTS accuracy. MEASURED RESULT (RTX 5070, MNIST 10000/2000):
  - hippocampal memory (real ReplayBuffer + SleepConsolidator) + brain decision (LatentQ value + Metacog)
    HOLD accuracy: the loop reaches 93.3% (-0.2% vs the v3.54 deep loop) -- faithfulness for FREE. These two
    subsystems are the default.
  - the growable MLP readout (NORD_GROWABLE=1) all-plastic FORGETS (crashes to 78.6% at ~7 classes). FIX (v3.57,
    NORD_FREEZE=1 default): GROW-AND-FREEZE -- progressive neurogenesis + parameter isolation. Each new digit
    grows a fresh hidden block and FREEZES the prior units' incoming weights (+ zeroes their Adam moments), so a
    new digit CANNOT overwrite old feature detectors. This re-enables neurogenesis: 78.6% -> 88.8% (no crash,
    stable curve, grew 32->192 hidden). HONEST: 88.8% still trails the plain linear 93.3% -- MNIST digits share
    features, so a shared readout is more parameter-efficient than per-digit isolated blocks; freeze fixes
    forgetting but at a representation-efficiency cost. Neurogenesis is now VIABLE, not the accuracy winner.
    The default readout stays linear (93.3%); growable+freeze is the working opt-in.

SLEEP CONSOLIDATION (NORD_SLEEP_STEPS>0, an interleaved balanced-all-digit replay pass at reduced lr -- the
CLS anti-forgetting mechanism) was TESTED as a fix for the MLP and does NOT rescue it: MLP no-sleep 78.6%,
heavy sleep 73.7% (overfits the finite ~200/digit replay buffer WORSE), light sleep 78.7% (no change). The
MLP fails by OVERFITTING the finite replay, not by lack of interleaving, so more replay cannot help. Note the
DEFAULT loop already embodies working sleep-consolidation: the per-cycle BALANCED REPLAY (linear readout) IS
continual interleaved consolidation, which is exactly why it holds 93.3% without an extra sleep phase. Sleep
is thus kept OFF by default (the linear+replay default is the working consolidation); the explicit sleep pass
is an opt-in knob that only overfits the MLP.
PROMOTED 2026-08-12 (#10 lane 4e): reasoning/ -> hippocampus/, the region whose mechanism
delivers the module's identity -- continual learning without catastrophic forgetting via
balanced replay (the docstring above records the finding: the per-cycle balanced replay IS
working sleep-consolidation), and the region that already holds both modules this loop
uniquely consumes (hippocampal_memory, growable_readout). Registered in the TRAIN_TIER
annex + run_selfchecks TRAIN_GATED (torchvision at module top via load_mnist -- the
recorded deliberate shape; REGIONS membership would break the alias round-trip on a
torchvision-less box, the lane-2 CI regression). Full run reproduced AT THE NEW PATH
before landing: 93.3% final overall, -0.2% vs DEEP_LOOP_REF -- the move preserved the
module, not merely relocated it.

E-PROP ADJUDICATION, carried forward verbatim from specs/010 (its premise re-verified at
promotion): the readout trains by Adam + .backward() and the dopamine multiplier SCALES
that path -- permitted because "That path is TRAIN-TIER OFFLINE, not a live learning path
... no _LIVE_ROOT or _ENTRYPOINT in gates/dormancy_gate.py reaches it" (re-measured:
dlb's whole import closure sits outside reachable()). BINDING CONSTRAINT: this module must
never be wired to a live LEARNING consumer; if it ever is, that consumer takes the
governed `lr_scale` three-factor parameter (plasticity/valence_stdp.py,
faculties/cortisol.py), never Adam -- and making this module reachable converts the
permitted offline trainer into an e-prop violation that nothing would announce.

EXEC-BYPASS P2 ADJUDICATED 2026-08-12 (#10 lane 4f follow-up): the PLANNER is the one
selection mechanism, and the NORD_EXEC flag + the fused decision.choose() branch are
DELETED, not gated. Measured before deletion (independent harness, positive control in
the same runs): at the shipped config (15 cycles, 30 seeds) planner vs fused chooser is
a genuine tie (0.5917 vs 0.5894, higher on 12/30 vs 13/30) while the same harness
separates a worse policy at 29/30 (curiosity-only, -2.1pp) -- a real null, not a dead
harness. Settled on MECHANISM: the planner duplicates no governed mechanism, while the
chooser's value channel carries faculties/brain_decision's recorded fusion null,
re-confirmed here by a permutation null (structure preserved, content destroyed:
-0.11pp, 7/15; the constant-pin null alone is confounded and was not relied on).
BrainDecision stays constructed: observe()/reliability()/grow_trigger() keep their
sites, and on the default path observe() trains a table nothing reads (bit-identical
A/B over 3 seeds -- suppressing all 15 observe() calls changes no output while the
table's |W|+|b| moves 0.469 -> 1.099). That is presence-not-participation, named here
and in this module's dormancy EXEMPT entry rather than laundered; choose() keeps
exactly ONE external call site, the mechanics-check split criterion (ASSERT 4), which
gate command #1 executes on every smoke run.

Self-check shape (the deep_stdp DEEPSTDP_RUN precedent): the bare __main__ is a bounded
MECHANICS check (~10-30 s: wiring asserts with reachable reds, NO accuracy claims -- any
training-scale reduction reds the 0.85 bar, measured: 0.220 at NORD_TRAIN=1000, so the
accuracy claims live with the scale that earns them). DLB_RUN=1 = the full 5.8-min
measurement with both accuracy asserts intact.

Run (from repo root; GPU strongly recommended):
    DLB_RUN=1 NORD_GPU=1 PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.instruments.developmental_loop_brain
Scale knobs: NORD_CYCLES(15) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(300) NORD_REPLAY(200)
             NORD_SGD(300) NORD_CAP(1) NORD_TRAIN(10000) NORD_TEST(2000) NORD_HIDDEN(64) NORD_GROW_K(4)
             NORD_HIDDEN_CAP(160) NORD_GPU
Deps: numpy torch torchvision brian2 ([train]/[gpu] extras; sklearn not needed here).
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.faculties.brain_decision import BrainDecision
from snn_data_lake.faculties.locus_coeruleus import LocusCoeruleus
from snn_data_lake.faculties.spiking_curiosity import (
    SpikingCuriosityDrive,
    SpikingCurriculum,
)
from snn_data_lake.infra.device import enable, resolve_device
from snn_data_lake.instruments.deep_stdp_latency import ConvLatencyStdp
from snn_data_lake.instruments.snn_mnist_stdp import load_mnist
from snn_data_lake.readout.growable_readout import GrowableReadout, LaminarColumns
from snn_data_lake.sleep.hippocampal_memory import HippocampalMemory

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_LinearReadout.train_step": (
        "APPROXIMATES_GRADIENT",
        "explicit autograd: `F.cross_entropy(self.clf(X), y).backward()` then `self.opt.step()`. It "
        "has no docstring, so this was read off the BODY rather than a claim. PROVISIONAL, and the "
        "criterion it turns on is spec FR-002h -- LIVE-PATH REACHABILITY, which THIS MODULE's own "
        "docstring already adjudicates from specs/010: the Adam path is permitted because it is "
        "TRAIN-TIER OFFLINE and no _LIVE_ROOT or _ENTRYPOINT reaches it. Re-measured 2026-08-17 with "
        "controls: this module is UNREACHABLE, so the permission holds today. The amendment LANDED "
        "2026-08-18 and states that criterion as a general clause, so this exit condition is met -- "
        "the over-flag is KEPT anyway, deliberately, and no longer waits on anything, "
        "because the same docstring warns that becoming reachable converts this into a violation "
        "'that nothing would announce' -- and Standing Invariant 3 makes the confident NEGATIVE the "
        "dangerous verdict",
    ),
}

# competence_executive symbols (DopamineModulator, CompetenceDynamics, goal_value) plus
# PrefrontalExecutive and GlobalWorkspace are imported LAZILY inside main() and
# _mechanics_check() -- the kb_ranking shape: used exclusively inside those functions. The
# 2026-08-12 retirement pass replaced the ExecutivePlanner/WorkspaceBroadcast wrappers with
# direct composition: _plan_objective below injects the public domain objects into
# PrefrontalExecutive.plan (write=False -- an offline planning query never writes the
# executive's working-memory goal), and the goal broadcast submits to a GlobalWorkspace this
# loop constructs and OWNS (private offline bus, so stepping it is owner semantics -- the
# tenant contract fixed at 4b24c1da does not apply to a bus's own clock).


def _plan_objective(comp_vec, candidates, seed=0):
    """PLAN the next study objective: a fresh PrefrontalExecutive per call (n_options must equal
    len(candidates)), the competence-domain forward model + objective injected, write=False.
    The composition the retired ExecutivePlanner wrapper held, now owned by its one consumer."""
    from snn_data_lake.cortex.competence_executive import CompetenceDynamics, goal_value
    from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive

    executive = PrefrontalExecutive(n_options=len(candidates), seed=seed)
    idx, _rt = executive.plan(
        tuple(float(x) for x in comp_vec),
        list(candidates),
        CompetenceDynamics(),
        goal_value,
        horizon=1,
        write=False,
    )
    return list(candidates)[idx]


class _LinearReadout:
    """Diagnostic drop-in matching GrowableReadout's interface but a PLAIN continual nn.Linear (the v3.54
    winner). Lets NORD_GROWABLE=0 isolate whether the growable-MLP readout is the accuracy cost while KEEPING
    the hippocampal memory + brain-decision modules wired. grow() is a no-op (linear has no hidden to grow).

    P2 debt, recorded not laundered (AUDIT_BACKLOG M21): this class is a verbatim copy of
    reasoning/_loop_common.LinearReadout, and _cache_features below is the per-image loop
    _loop_common replaced with a ~135x batched version. NOT adopted here because both
    superseders are still reasoning-tier -- adopting them would create two new
    governed->reasoning module-top imports, the exact shape this promotion exists to remove.
    M21 is owed by whichever lane promotes _loop_common + fast_features."""

    def __init__(self, in_dim, n_classes=10, seed=0, lr=0.01):
        # A constructor may not re-seed the process-global RNG: `run_selfchecks.py` runs part of the
        # suite in ONE process and `tests/test_selfchecks.py` runs every CORE `__main__` in the
        # pytest process, so a bare `torch.manual_seed(seed)` here shifted the stream for every
        # module built afterwards. Same defect the Determinism note in this region's own
        # `growable_readout.py` records removing on 2026-08-04 -- it survived here because no gate
        # could see a constructor, and this module module-top-imports that very file at :110.
        # `nn.Linear` has no `generator=` parameter, so the seed stays and the RNG is FORKED
        # instead: `self.clf`'s weights are byte-identical to the unbracketed form, and the global
        # stream is unchanged. `devices=[]` forks CPU only -- the draw below is a CPU draw, and
        # forking CUDA state would force CUDA initialization at construction time.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            dev = resolve_device()
            self.clf = nn.Linear(in_dim, n_classes).to(dev)
        self.opt = torch.optim.Adam(self.clf.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(self, X, y, lr, steps=1):
        for g in self.opt.param_groups:
            g["lr"] = lr
        for _ in range(steps):
            self.opt.zero_grad()
            F.cross_entropy(self.clf(X), y).backward()
            self.opt.step()

    def predict(self, X):
        with torch.no_grad():
            return self.clf(X)

    def grow(self, k=1, freeze_old=False):
        # linear: no hidden layer to grow. `freeze_old` accepted so the signature matches the
        # duck-typed call at the FREEZE branch (specs/007 T021's disposition: the stub stays
        # because grow()/n_hidden are part of the readout interface the loop calls; the former
        # signature mismatch could never fire only because all growth selectors default to 0).
        return

    @property
    def n_hidden(self):
        return 0


VOTE_LOOP_BAR = (
    0.85  # developmental_loop.py STDP-prototype + neuron-label VOTE readout (best 88.8%)
)
# The v3.54 deep-loop result. Its producing module (developmental_loop_deep.py) was RETIRED
# 2026-08-06 under P2; this constant is where the measurement now lives, so it is a citation of a
# recorded result rather than a number re-derivable in-tree.
DEEP_LOOP_REF = 0.935  # v3.54 deep loop: conv1 + continual LINEAR readout
BASE_LR = 0.01


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _cache_features(conv, X, dev, tag):
    rows = []
    for i, img in enumerate(X):
        rows.append(conv.features(img).reshape(-1))
        if (i + 1) % 2000 == 0:
            print(f"  cache {tag}: {i + 1}/{len(X)} features")
    return torch.stack(rows, dim=0).to(dev)


def _accuracy(readout, feats, labels, idx):
    if len(idx) == 0:
        return 0.0
    with torch.no_grad():
        pred = readout.predict(feats[idx]).argmax(dim=1)
        return (pred == labels[idx]).float().mean().item()


# distinct-from: instruments/deep_stdp.py, instruments/deep_stdp_latency.py -- the SAME settled
# gate-tier PATTERN (bare __main__ = bounded integrity check, env flag = the full
# measurement), deliberately followed, but each _mechanics_check asserts ITS OWN module's
# wiring (this one: replay store/sample, planner-vs-fused selection, the LC*DA lr fold,
# this loop's feature cache). A wiring check cannot be shared across modules whose wiring
# differs; the shared thing is the SHAPE, which is the precedent, not a mechanism.
def _mechanics_check():
    """Bounded MECHANICS + STRUCTURE check (the deep_stdp_latency genn shape): asserts the
    WIRING with reachable reds and claims NOTHING about accuracy at scale -- measured, any
    training-scale reduction reds the 0.85 bar (0.220 at NORD_TRAIN=1000), so the accuracy
    claims live behind DLB_RUN=1 with the scale that earns them.

    Sizing discipline (the measured silent trap this check exists to dodge): at reduced
    NORD_TRAIN the full loop's replay store slice runs past the per-digit index and every
    cycle silently reports '+ 0 hippocampal-replay feats'. Here PER + REPLAY is asserted
    under the per-digit row count, and ASSERT 1 makes an empty replay a hard red."""
    from snn_data_lake.cortex.competence_executive import DopamineModulator
    from snn_data_lake.faculties.global_workspace import GlobalWorkspace

    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    TRAIN_N, TEST_N, CYC_M, PER_M, REPLAY_M, SGD_M = 300, 120, 4, 12, 8, 40

    print("=" * 76)
    print("developmental_loop_brain -- bounded mechanics check (DLB_RUN=1 = the full 5.8-min run)")
    print("=" * 76)
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)
    conv = ConvLatencyStdp(12, 15)
    for img in Xtr[:200]:
        conv.train_image(img)
    Ftr = _cache_features(conv, Xtr, dev, "train")
    Fte = _cache_features(conv, Xte, dev, "test")
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    # ASSERT 5 -- the feature cache is WIRED. Red: break _cache_features' reshape/stack.
    assert Ftr.shape == (TRAIN_N, conv.feat_dim()), (
        f"feature cache shape {tuple(Ftr.shape)} != ({TRAIN_N}, {conv.feat_dim()})"
    )
    assert bool(torch.isfinite(Ftr).all()) and float(Ftr.min()) >= 0.0, (
        "conv1 features must be finite and non-negative (spike-derived)"
    )

    readout = _LinearReadout(Ftr.shape[1], n_classes=10, seed=0)
    memory = HippocampalMemory(seed=0)
    decision = BrainDecision(n_actions=10, state_dim=10, seed=0)
    curiosity = SpikingCuriosityDrive(seed=0)
    lc = LocusCoeruleus()
    dopamine = DopamineModulator(seed=0)
    workspace = GlobalWorkspace(seed=0)  # this check OWNS this private bus: submit + step below

    # sizing guard: replay slice must be NON-EMPTY by construction (the silent-trap dodge).
    for d in range(10):
        assert len(tr_by[d]) >= PER_M + REPLAY_M, (
            f"digit {d} has {len(tr_by[d])} rows < PER+REPLAY {PER_M + REPLAY_M} -- "
            f"the replay store would be silently empty; resize the check"
        )

    # ASSERT 4 -- the planner SELECTS rather than decorates: on a constructed state where
    # lookahead and fused-curiosity disagree, the planner and the fused chooser must pick
    # DIFFERENT objectives. Red: make plan_objective return a constant / the fused pick.
    # Since the 2026-08-12 EXEC-bypass adjudication this is the ONLY external choose() call
    # site tree-wide, and gate command #1 executes it on every smoke run -- the fused chooser
    # survives as this criterion's control, not as a selection path.
    comp_c = [0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    cands_c = list(range(10))
    cur_c = [1.0 if d == 5 else 0.01 for d in cands_c]  # curiosity screams digit 5
    pick_exec = _plan_objective(comp_c, cands_c)
    pick_fused = decision.choose(cands_c, comp_c, cur_c)
    assert pick_exec != pick_fused, (
        f"the planner and the fused chooser picked the same objective ({pick_exec}) on a "
        f"state constructed to split them -- the planner is decorative, not selecting"
    )

    w0 = readout.clf.weight.detach().clone()
    gains, da_gain = [], 1.0
    learned, competence = [], {d: 0.0 for d in range(10)}
    for cyc in range(1, CYC_M + 1):
        unseen = [d for d in range(10) if d not in learned]
        cands = unseen if unseen else list(range(10))
        comp_vec = [competence[d] for d in range(10)]
        objective = _plan_objective(comp_vec, cands)
        is_new = objective not in learned
        workspace.submit("executive", f"study_digit_{objective}", 1.0 - competence[objective])
        workspace.step()  # owner tick of the check's private bus (the one competition per cycle)
        na = lc.step(1.0 if is_new else (1.0 - competence[objective]))
        gain = (1.0 + 0.5 * na) * da_gain
        gains.append(gain)
        comp_before = dict(competence)
        obj_rows = tr_by[objective][:PER_M]
        replay_rows = memory.sample([d for d in learned if d != objective], REPLAY_M)
        if cyc >= 2:
            # ASSERT 1 -- hippocampal replay is REAL once something is stored. Red: neuter
            # HippocampalMemory.sample to return [] (and the sizing trap above produces
            # this red naturally, which is why the assert is worth having).
            assert len(replay_rows) > 0, (
                "replay empty after digits were stored -- store/sample dead"
            )
        batch_rows = obj_rows + replay_rows
        assert len(batch_rows) == len(obj_rows) + len(replay_rows)
        idx = torch.tensor(batch_rows, dtype=torch.long, device=dev)
        readout.train_step(Ftr[idx], ytr[idx], lr=BASE_LR * gain, steps=SGD_M)
        if is_new:
            memory.store(tr_by[objective][PER_M : PER_M + REPLAY_M], objective)
            learned.append(objective)
        for d in learned:
            competence[d] = _accuracy(readout, Fte, yte, te_by[d])
        lp_gain = competence[objective] - comp_before[objective]
        curiosity.observe(objective, 1.0 - competence[objective])
        decision.observe(
            objective,
            [comp_before[d] for d in range(10)],
            [competence[d] for d in range(10)],
            curiosity.learning_progress(objective),
            curiosity.novelty(objective),
        )
        da_gain = dopamine.modulate(lp_gain)

    # ASSERT 2 -- the readout actually LEARNS. Red: neuter train_step / lr=0.
    assert not torch.equal(w0, readout.clf.weight.detach()), (
        "readout weights did not move across the mechanics cycles"
    )
    # ASSERT 3 -- the neuromodulatory fold is NOT a constant (pins the gain line, the
    # e-prop-hazard site). Red: neuter LocusCoeruleus.step or DopamineModulator.modulate
    # to a constant return.
    assert len({round(g, 6) for g in gains}) > 1, (
        f"lr gain was constant across cycles ({gains[0]:.4f}) -- LC/dopamine fold dead"
    )
    # ASSERT 6 -- pooled accuracy over the STUDIED digits clears a bar with a real red:
    # neutered train_step measures ~0.10 (10-way chance), the trained check must clear a
    # margin above it. Bar set from measurement at these knobs, halfway to the observed
    # value -- NOT an accuracy claim about the loop (those live behind DLB_RUN=1).
    pooled = [r for d in learned for r in te_by[d]]
    overall = _accuracy(readout, Fte, yte, pooled)
    print(
        f"mechanics: studied {sorted(learned)}, pooled acc {overall:.3f} (bar 0.30; "
        f"neutered-readout red ~0.10), gains {[round(g, 2) for g in gains]}, "
        f"replay live, planner selects (exec {pick_exec} != fused {pick_fused})"
    )
    assert overall > 0.30, (
        f"pooled accuracy {overall:.3f} <= 0.30 on studied digits -- wiring broken "
        f"(neutered red measures ~0.10; this is a wiring bar, not an accuracy claim)"
    )
    print(
        "mechanics check OK: cache wired, replay real, readout learns, neuromod fold live, "
        "planner selects; accuracy claims live behind DLB_RUN=1"
    )
    return 0


def main():
    if os.environ.get("DLB_RUN") != "1":
        return _mechanics_check()
    # lazy, governed-to-governed since the 2026-08-12 lane-4f promotion (see the module-top note).
    from snn_data_lake.cortex.competence_executive import DopamineModulator
    from snn_data_lake.faculties.global_workspace import GlobalWorkspace

    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)

    CYCLES = _envi("NORD_CYCLES", 15)
    C1 = _envi("NORD_C1", 30)
    T_LAT = _envi("NORD_T", 15)
    L1_EPOCHS = _envi("NORD_L1_EPOCHS", 3)
    PER = _envi("NORD_PER", 300)
    REPLAY = _envi("NORD_REPLAY", 200)
    SGD = _envi("NORD_SGD", 300)
    CAP = _envi("NORD_CAP", 1)
    TRAIN_N = _envi("NORD_TRAIN", 10000)
    TEST_N = _envi("NORD_TEST", 2000)
    HIDDEN = _envi("NORD_HIDDEN", 64)
    GROW_K = _envi("NORD_GROW_K", 4)
    HIDDEN_CAP = _envi("NORD_HIDDEN_CAP", 160)
    # SLEEP consolidation (anti-forgetting): after the objective-skewed LEARN, replay a BALANCED interleaved
    # batch of ALL learned digits at reduced lr (hippocampal replay -> neocortical consolidation, CLS).
    SLEEP_EVERY = _envi("NORD_SLEEP_EVERY", 1)  # run a sleep pass every N cycles
    SLEEP_PER = _envi(
        "NORD_SLEEP_PER", 200
    )  # balanced replay feats per learned digit in the sleep pass
    SLEEP_STEPS = _envi(
        "NORD_SLEEP_STEPS", 0
    )  # consolidation SGD steps (0 = sleep OFF, the default -- see below)
    SLEEP_LR_SCALE = float(
        os.environ.get("NORD_SLEEP_LR", "0.5")
    )  # reduced plasticity during sleep

    print("=" * 92)
    print(
        "BRAIN-INTEGRATED DEVELOPMENTAL LOOP (Slice 1) -- real hippocampus + basal-ganglia + metacog + neurogenesis"
    )
    print(
        "  conv1 (unsupervised) -> GrowableReadout ; memory=HippocampalMemory ; decision=curiosity+curriculum+BrainDecision"
    )
    print(
        f"device={dev}  cycles={CYCLES}  C1={C1}  per={PER}  replay={REPLAY}  sgd={SGD}  "
        f"hidden={HIDDEN}(grow +{GROW_K}, cap {HIDDEN_CAP})  train={TRAIN_N}  test={TEST_N}"
    )
    print("=" * 92)

    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    # ---- STAGE 1: UNSUPERVISED PERCEPTION (conv1, frozen) ----
    torch.manual_seed(0)
    print(
        f"\n[perception] pre-training conv1 latency-STDP (UNSUPERVISED, {L1_EPOCHS} epochs, no labels)..."
    )
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        for i, img in enumerate(Xtr):
            conv.train_image(img)
            if (i + 1) % 2000 == 0:
                print(f"  conv1 ep{ep + 1}/{L1_EPOCHS}: {i + 1}/{len(Xtr)} images")
    diversity = conv.kernel_diversity()
    print(
        f"[perception] conv1 FROZEN; feat_dim={conv.feat_dim()}; kernel diversity={diversity:.3f} "
        f"({'no collapse' if diversity < 0.6 else 'WARNING collapse'})"
    )

    print("\n[cache] pre-computing frozen conv1 features (train + test, once)...")
    Ftr = _cache_features(conv, Xtr, dev, "train")
    Fte = _cache_features(conv, Xte, dev, "test")
    ytr = Ytr.long().to(dev)
    yte = Yte.long().to(dev)
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    # ---- the brain subsystems ----
    # Readout: DEFAULT plain linear (93.3%, holds the deep loop). NORD_GROWABLE=1 opts into the GrowableReadout
    # (MLP + function-preserving neurogenesis) -- an HONEST NEGATIVE here: the growth is function-preserving and
    # works standalone, but a continually-trained MLP readout FORGETS catastrophically across digits (crashes to
    # 78.6% at ~7 classes) where the linear readout does not. The neurogenesis mechanism is sound; MLP-continual-
    # forgetting is the cost. Kept opt-in for reproducibility.
    GROWABLE = _envi("NORD_GROWABLE", 0)
    # NORD_LAMINAR deleted 2026-08-12 at the laminar merge: LaminarReadout was RETIRED with
    # its file (the v3.59 failure mode -- partial L2/3 freeze under a drifting shared L4,
    # 63.7% -- and an e-prop violator; record kept in readout/growable_readout's merge
    # note). LaminarColumns (the fix) stays available below under NORD_COLUMNS.
    COLUMNS = _envi(
        "NORD_COLUMNS", 0
    )  # per-digit ISOLATED cortical columns (progressive nets): each new digit
    #                                        gets its OWN full L4+L2/3 column, whole old columns frozen -> frozen
    #                                        features always see stable (their own) input. Fixes v3.59's L4 drift.
    FREEZE = _envi(
        "NORD_FREEZE", 1
    )  # (with GROWABLE/COLUMNS=1) grow-and-FREEZE progressive neurogenesis.
    if COLUMNS:
        readout = LaminarColumns(
            D, n_classes=10, n23=HIDDEN, seed=0
        )  # per-digit isolated columns + shared L5
    elif GROWABLE:
        readout = GrowableReadout(
            D, n_classes=10, hidden=HIDDEN, seed=0
        )  # flat MLP + function-preserving neurogenesis
    else:
        readout = _LinearReadout(
            D, n_classes=10, seed=0
        )  # plain continual linear (v3.54 winner, 93.3%)
    memory = HippocampalMemory(seed=0)  # real ReplayBuffer + SleepConsolidator
    decision = BrainDecision(
        n_actions=10, state_dim=10, seed=0
    )  # LatentQ value + MetacognitiveMonitor
    curiosity = SpikingCuriosityDrive(seed=0)
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
    lc = LocusCoeruleus()
    # ---- Slice 2: executive planning + dopamine RPE neuromod + global-workspace broadcast ----
    # NORD_EXEC deleted 2026-08-12 (EXEC-bypass P2 adjudication, module docstring): the planner
    # IS the selection path -- no flag, no fused fallback.
    DOPA = _envi("NORD_DOPA", 1)  # dopamine RPE modulates the readout lr alongside LC
    WS = _envi("NORD_WS", 1)  # global_workspace broadcasts the current goal (coordination)
    dopamine = DopamineModulator(seed=0)
    workspace = GlobalWorkspace(seed=0)  # the loop OWNS this private offline bus
    da_gain = 1.0  # dopamine lr multiplier (from the previous cycle's RPE)

    learned = []
    competence = {d: 0.0 for d in range(10)}
    consec = {d: 0 for d in range(10)}
    curve = []
    grow_events = 0

    for cyc in range(1, CYCLES + 1):
        # ---- DECISION: phase 1 explore unseen; phase 2 revisit under BrainDecision (curiosity + LatentQ
        #      value fused by the metacognitive explore/exploit gate). NORD_CAP anti-fixation guard kept. ----
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        if not unseen:
            allowed = [d for d in candidates if consec[d] < CAP]
            candidates = allowed or candidates
        comp_vec = [competence[d] for d in range(10)]
        # EXECUTIVE: prefrontal_executive plans the objective by lookahead over competence dynamics
        # (goal_value = mean + min competence, predictor raises headroom-proportionally). The ONE
        # selection mechanism since the 2026-08-12 EXEC-bypass adjudication.
        objective = _plan_objective(comp_vec, candidates)
        consec = {d: (consec[d] + 1 if d == objective else 0) for d in range(10)}
        is_new = objective not in learned
        # GLOBAL WORKSPACE: broadcast the current goal (salience = how far from mastered). Coordination/faithfulness.
        # This loop constructed the bus, so stepping it is OWNER semantics (the one competition per cycle).
        # PREDICATE CHANGE disclosed (2026-08-12 review pass): the retired wrapper's broadcast()
        # returned the bus's top CONTENT -- truthy even when a STALE earlier cycle's goal still
        # held the slot -- while ws_ignited below reports whether THIS cycle's goal seized it,
        # which is what the variable's name and the cycle print always claimed. The two diverge
        # exactly on stale-hold cycles (low-salience revisits after a strong earlier goal).
        if WS:
            goal_content = f"study_digit_{objective}"
            workspace.submit("executive", goal_content, 1.0 - competence[objective])
            ws_ignited = workspace.step() == goal_content
        else:
            ws_ignited = None
        print(
            f"\nCYCLE {cyc}  ->  OBJECTIVE: digit {objective}  "
            f"[{'explore (novel)' if is_new else 'revisit'}; planner-selected, "
            f"reliability={decision.reliability():.2f}, novelty={curiosity.novelty(objective):.2f}"
            f"{'; workspace IGNITED' if ws_ignited else ''}]"
        )

        # ---- NEUROMODULATION: locus-coeruleus NA on salience -> bounded readout lr gain ----
        salience = 1.0 if is_new else (1.0 - competence[objective])
        na = lc.step(salience)
        # P1 exception, stated with its measurement (owned at the 2026-08-12 promotion): the
        # 0.5 fixes how far noradrenaline may amplify the readout lr. It is NOT inert --
        # measured live at full scale, the combined gain ranged x0.75 to x3.00 across the 15
        # cycles (the mechanics check asserts non-constancy). The lane-4f adjudication
        # (2026-08-12, in cortex/competence_executive's own P1 paragraph) settled the
        # LR_MIN/LR_MAX side: kept fixed, NO exception claimed, conversion declined on P2
        # grounds. This file is the governed caller of that pair; both P1 records now sit in
        # the governed tier.
        gain = (1.0 + 0.5 * na) * (
            da_gain if DOPA else 1.0
        )  # LC noradrenaline x dopamine RPE (prev cycle)

        # ---- LEARN: continual readout SGD on the objective's cached features + BALANCED REPLAY sampled by
        #      the real HippocampalMemory (ReplayBuffer) of previously-learned digits. ----
        comp_before = dict(competence)
        n_train = PER if is_new else max(REPLAY, PER // 4)
        obj_rows = tr_by[objective][:n_train]
        replay_rows = memory.sample([d for d in learned if d != objective], REPLAY)
        batch_rows = obj_rows + replay_rows
        idx = torch.tensor(batch_rows, dtype=torch.long, device=dev)
        t0 = time.time()
        readout.train_step(Ftr[idx], ytr[idx], lr=BASE_LR * gain, steps=SGD)
        dt = time.time() - t0
        if is_new:
            memory.store(
                tr_by[objective][n_train : n_train + REPLAY], objective
            )  # hippocampus remembers exemplars
            learned.append(objective)

        # ---- SLEEP CONSOLIDATION (the anti-forgetting fix): the per-cycle LEARN above is objective-SKEWED, which a
        #      plastic MLP readout overfits -> catastrophic forgetting. A sleep pass replays a BALANCED, INTERLEAVED
        #      batch of ALL learned digits equally (no objective skew) at reduced lr -- hippocampal replay driving
        #      neocortical consolidation (Complementary Learning Systems). This is what stops the MLP crash. ----
        if learned and SLEEP_STEPS and (cyc % SLEEP_EVERY == 0):
            sleep_rows = memory.sample(
                learned, SLEEP_PER
            )  # balanced across ALL learned (incl objective)
            if sleep_rows:
                sidx = torch.tensor(sleep_rows, dtype=torch.long, device=dev)
                readout.train_step(
                    Ftr[sidx], ytr[sidx], lr=BASE_LR * SLEEP_LR_SCALE, steps=SLEEP_STEPS
                )
                print(
                    f"  [sleep]         consolidated {len(learned)} digits, {len(sleep_rows)} balanced replay feats "
                    f"x{SLEEP_STEPS} steps @ lr {BASE_LR * SLEEP_LR_SCALE:.4f} (interleaved, anti-forgetting)"
                )
        print(
            f"  [neuromod]      LC NA {na:.2f} ({lc.mode()}) salience {salience:.2f} -> readout lr gain x{gain:.2f}"
        )
        print(
            f"  [readout]       {SGD} steps on {len(obj_rows)} digit-{objective} + {len(replay_rows)} hippocampal-replay feats ({dt:.1f}s)"
        )

        # ---- MEASURE competence on the held-out TEST cache (train/test never mix) ----
        for d in learned:
            competence[d] = _accuracy(readout, Fte, yte, te_by[d])
        pooled = [r for d in learned for r in te_by[d]]
        overall = _accuracy(readout, Fte, yte, pooled)
        curve.append(overall)

        # ---- DECIDE: feed the real modules the outcome. curiosity/curriculum (spiking learning-progress),
        #      BrainDecision (LatentQ value update + metacog), then NEUROGENESIS on the metacog grow_trigger. ----
        curiosity.observe(objective, 1.0 - competence[objective])
        lp = curiosity.learning_progress(objective)
        curriculum.observe(lp)
        comp_after_vec = [competence[d] for d in range(10)]
        decision.observe(
            objective,
            [comp_before[d] for d in range(10)],
            comp_after_vec,
            lp,
            curiosity.novelty(objective),
        )
        # DOPAMINE: reward-prediction-error on the realised competence gain sets NEXT cycle's lr multiplier.
        if DOPA:
            da_gain = dopamine.modulate(competence[objective] - comp_before[objective])
        grew = ""
        if (GROWABLE or COLUMNS) and FREEZE:
            # PROGRESSIVE NEUROGENESIS + FREEZE: after learning this new digit, freeze its units and grow a fresh
            # block for the NEXT digit -> each digit owns a protected sub-network, new cannot overwrite old.
            if is_new and readout.n_hidden < HIDDEN_CAP:
                readout.grow(GROW_K, freeze_old=True)
                grow_events += 1
                grew = f" [+neurogenesis+FREEZE: hidden -> {readout.n_hidden}, old frozen]"
        elif GROWABLE:
            # non-freeze growable (the honest-negative version): metacog-triggered growth, all-plastic -> forgets.
            if decision.grow_trigger() and readout.n_hidden < HIDDEN_CAP:
                readout.grow(GROW_K)
                grow_events += 1
                grew = f" [+neurogenesis: readout hidden -> {readout.n_hidden}]"
        print(
            f"  [self-improve]  competence over {sorted(learned)} = {overall:.1%}  (digit {objective}: "
            f"{competence[objective]:.1%}, lp={lp:.3f}); curriculum {curriculum.mastered}/10{grew}"
        )

    # ---- consolidation pass (real SleepConsolidator over the learned digits) ----
    consol = memory.consolidate(learned)

    final_overall = curve[-1] if curve else 0.0
    gap_deep = final_overall - DEEP_LOOP_REF
    print("\n" + "=" * 92)
    print(
        f"BRAIN-INTEGRATED LOOP COMPLETE: autonomously learned {sorted(learned)} over {CYCLES} cycles via the "
        f"REAL brain subsystems;\n  perception unsupervised, readout grew {grow_events}x to hidden={readout.n_hidden} "
        f"(function-preserving neurogenesis); curriculum mastered {curriculum.mastered}/10."
    )
    print("-" * 92)
    print(f"  final OVERALL held-out test accuracy         : {final_overall:.1%}")
    print("  accuracy CURVE (per cycle)                   : " + " ".join(f"{a:.2f}" for a in curve))
    print(
        f"  hippocampal consolidation (SleepConsolidator): retention={consol.get('retention')}, "
        f"value_margin={consol.get('value_margin')}, skills={consol.get('n_skills')}"
        f"  [fixed-configuration readout, NOT a measured retention: printed bit-identical at "
        f"NORD_TRAIN=1000 and 10000 while the loop's accuracy moved 0.220 -> 0.933]"
    )
    print("-" * 92)
    print(f"  reference VOTE loop (STDP prototypes + vote) : ~{VOTE_LOOP_BAR:.0%} (best 88.8%)")
    print(f"  reference DEEP loop (conv1 + linear, v3.54)  : {DEEP_LOOP_REF:.1%}")
    print(
        f"  brain-integrated (this) vs deep loop         : {gap_deep:+.1%} "
        f"({'holds/gains -- faithfulness free' if gap_deep >= -0.01 else 'costs accuracy (honest)'})"
    )
    print("=" * 92)

    # self-check: routing through the real brain subsystems must still BEAT the vote loop (>0.85) + chance;
    # the signed gap to the v3.54 deep loop (0.935) is the honest, reported result -- we do NOT assert the
    # brain modules add accuracy (they are an architectural-faithfulness upgrade; holding ~93.5% IS success).
    assert final_overall > 0.10, f"brain loop at/below chance: {final_overall:.2f}"
    assert final_overall > VOTE_LOOP_BAR, (
        f"brain-integrated loop {final_overall:.3f} did not beat the vote-loop bar {VOTE_LOOP_BAR:.2f} -- "
        "report the honest number, do not lower the bar."
    )
    verdict = (
        "holds/gains vs the deep loop"
        if gap_deep >= -0.01
        else f"costs {-gap_deep:.1%} vs the deep loop"
    )
    print(
        f"self-check OK: brain-integrated loop {final_overall:.1%} > vote-loop ~{VOTE_LOOP_BAR:.0%} + chance; "
        f"neurogenesis grew readout {grow_events}x; {verdict} (honest)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
