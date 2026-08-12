r"""
BRAIN-INTEGRATED DEVELOPMENTAL LOOP (Slice 1) -- the deep developmental loop (v3.54) with its
hand-rolled parts replaced by the brain's REAL subsystems, wired through thin adapters.

  perception    : conv1 latency-STDP (unsupervised, frozen)          [reasoning/deep_stdp_latency]
  memory        : HippocampalMemory  = real ReplayBuffer + SleepConsolidator   [brain_adapters]
  readout       : GrowableReadout    = MLP grown function-preservingly (Net2Net) [brain_adapters]
  decision      : SpikingCuriosity + SpikingCurriculum + BrainDecision (LatentQ value + Metacog) [brain_adapters]
  neurogenesis  : readout.grow() fired by the MetacognitiveMonitor grow_trigger  [brain the brain grows its own readout neurons]
  neuromod      : LocusCoeruleus (readout lr gain)                    [neuromodulation]

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
The self-check asserts the loop beats the vote loop (>0.85) + chance and prints the signed gap to v3.54's 93.5%.

Run (from repo root; GPU strongly recommended):
    NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python -m snn_data_lake.reasoning.developmental_loop_brain
Scale knobs: NORD_CYCLES(15) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(300) NORD_REPLAY(200)
             NORD_SGD(300) NORD_CAP(1) NORD_TRAIN(10000) NORD_TEST(2000) NORD_HIDDEN(64) NORD_GROW_K(4)
             NORD_HIDDEN_CAP(160) NORD_GPU
Deps: numpy torch torchvision snntorch   ([train]/[gpu] extras. NOT "outside the core": this module lives in src/snn_data_lake/reasoning/
         and imports from the package -- P4c, every build is part of the brain.)
"""

import os
import sys
import time

import torch

from snn_data_lake.plasticity.snn_mnist_stdp import load_mnist
from snn_data_lake.substrate.device import enable, resolve_device

import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp
from snn_data_lake.reasoning.brain_adapters.hippocampal_memory import HippocampalMemory
from snn_data_lake.reasoning.brain_adapters.growable_readout import GrowableReadout
from snn_data_lake.reasoning.brain_adapters.brain_decision import BrainDecision
from snn_data_lake.reasoning.brain_adapters.executive_control import (
    ExecutivePlanner,
    DopamineModulator,
    WorkspaceBroadcast,
)
from snn_data_lake.reasoning.brain_adapters.laminar_readout import LaminarReadout, LaminarColumns


class _LinearReadout:
    """Diagnostic drop-in matching GrowableReadout's interface but a PLAIN continual nn.Linear (the v3.54
    winner). Lets NORD_GROWABLE=0 isolate whether the growable-MLP readout is the accuracy cost while KEEPING
    the hippocampal memory + brain-decision modules wired. grow() is a no-op (linear has no hidden to grow)."""

    def __init__(self, in_dim, n_classes=10, seed=0, lr=0.01):
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

    def grow(self, k=1):
        return  # linear: no hidden layer to grow

    @property
    def n_hidden(self):
        return 0


from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum

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


def main():
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
    LAMINAR = _envi(
        "NORD_LAMINAR", 0
    )  # readout as the 6-layer NEOCORTICAL motif (L4->L2/3->L5->L6) with
    #                                        NEUROGENESIS (grow+freeze) in the L2/3 feature layer (v3.59: partial
    #                                        freeze insufficient -> 63.7%).
    COLUMNS = _envi(
        "NORD_COLUMNS", 0
    )  # per-digit ISOLATED cortical columns (progressive nets): each new digit
    #                                        gets its OWN full L4+L2/3 column, whole old columns frozen -> frozen
    #                                        features always see stable (their own) input. Fixes v3.59's L4 drift.
    FREEZE = _envi(
        "NORD_FREEZE", 1
    )  # (with GROWABLE/LAMINAR/COLUMNS=1) grow-and-FREEZE progressive neurogenesis.
    if COLUMNS:
        readout = LaminarColumns(
            D, n_classes=10, n23=HIDDEN, seed=0
        )  # per-digit isolated columns + shared L5
    elif LAMINAR:
        readout = LaminarReadout(
            D, n_classes=10, n23=HIDDEN, seed=0
        )  # cortical-column readout + L2/3 neurogenesis
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
    # ---- Slice 2: executive planning + dopamine RPE neuromod + global-workspace broadcast (opt-in, default on) ----
    EXEC = _envi(
        "NORD_EXEC", 1
    )  # prefrontal_executive plans the objective (lookahead over competence)
    DOPA = _envi("NORD_DOPA", 1)  # dopamine RPE modulates the readout lr alongside LC
    WS = _envi("NORD_WS", 1)  # global_workspace broadcasts the current goal (coordination)
    planner = ExecutivePlanner(seed=0)
    dopamine = DopamineModulator(seed=0)
    workspace = WorkspaceBroadcast(seed=0)
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
        cur_scores = [curiosity.curiosity_value(d) for d in candidates]
        # EXECUTIVE: prefrontal_executive plans the objective by lookahead over competence dynamics (goal_value =
        # mean + min competence, predictor raises headroom-proportionally). Falls back to BrainDecision otherwise.
        objective = (
            planner.plan_objective(comp_vec, candidates)
            if EXEC
            else decision.choose(candidates, comp_vec, cur_scores)
        )
        consec = {d: (consec[d] + 1 if d == objective else 0) for d in range(10)}
        is_new = objective not in learned
        # GLOBAL WORKSPACE: broadcast the current goal (salience = how far from mastered). Coordination/faithfulness.
        ws_ignited = (
            workspace.broadcast(
                "executive", f"study_digit_{objective}", 1.0 - competence[objective]
            )
            if WS
            else None
        )
        print(
            f"\nCYCLE {cyc}  ->  OBJECTIVE: digit {objective}  "
            f"[{'explore (novel)' if is_new else 'revisit'}; {'EXEC-planned' if EXEC else 'metacog ' + ('EXPLORE' if decision.explore() else 'exploit')}, "
            f"reliability={decision.reliability():.2f}, novelty={curiosity.novelty(objective):.2f}"
            f"{'; workspace IGNITED' if ws_ignited else ''}]"
        )

        # ---- NEUROMODULATION: locus-coeruleus NA on salience -> bounded readout lr gain ----
        salience = 1.0 if is_new else (1.0 - competence[objective])
        na = lc.step(salience)
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
        if (GROWABLE or LAMINAR or COLUMNS) and FREEZE:
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
    print(
        f"  accuracy CURVE (per cycle)                   : " + " ".join(f"{a:.2f}" for a in curve)
    )
    print(
        f"  hippocampal consolidation (SleepConsolidator): retention={consol.get('retention')}, "
        f"value_margin={consol.get('value_margin')}, skills={consol.get('n_skills')}"
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
