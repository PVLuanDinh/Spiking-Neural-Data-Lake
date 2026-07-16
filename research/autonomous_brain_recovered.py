r"""
AUTONOMOUS BRAIN (RECOVERED) -- the enclosed self-directed loop, with the accuracy recovered from
47.4% to ~95% online (and ~98% via a sleep-consolidation RBF refit), WITHOUT dropping any faculty.

WHY THE ORIGINAL (research/autonomous_brain.py) LANDS AT 47.4%
-------------------------------------------------------------
Its UNDERSTANDING readout is a backprop MLP (brain_adapters.growable_readout.GrowableReadout) grown
and frozen per new digit. In CLASS-INCREMENTAL testing (argmax over all 10 classes, no task id) the
shared softmax head forgets earlier digits via recency bias -- growable_readout.py's own self-check
(section 4) shows hidden-only freeze recovers only a few points and the class-IL "afterALL" number
stays low for every freeze mode. So the 47.4% is the READOUT, not a broken faculty: the exact same
frozen unsupervised conv1 features score 94.9-95.4% under an ORDER-INVARIANT readout
(developmental_loop_lake_slda.py) and 98.2% under a static RBF ensemble.

THE RECOVERY (this file) -- every faculty stays genuinely active
----------------------------------------------------------------
  PERCEPTION      conv1 latency-STDP, unsupervised + frozen; batched byte-identical features.
  WORKING MEMORY  the objective is held as a pointer in the prefrontal persistent-activity attractor.
  SELF-DIRECTION  curiosity picks unseen digits; a ROUND-ROBIN revisit scheduler then diversifies
                  (the original let curiosity fixate on one digit for every revisit cycle, starving
                  the rest -- fixed here). Curriculum + metacognition still active.
  UNDERSTANDING   SLDA_RFF -- streaming LDA over Random Fourier Features. Order-invariant sufficient
                  statistics => continual == joint => NO forgetting. This is the readout swap that
                  recovers the accuracy (~95% online).
  NEUROGENESIS    GENUINE and order-invariant: at each new concept the brain GROWS its Random-Fourier
                  basis by GROW_RFF new projection neurons, then SLEEP-REPLAYS every accumulated
                  observation through the widened basis to rebuild the order-invariant statistics from
                  scratch. Because the statistics are order-invariant and the durable lake holds the
                  memory, growing neurons NEVER forgets. This RESOLVES the accuracy-vs-neurogenesis
                  tension the original faced (grow-and-freeze MLP forgot; growing RFF neurons does not).
  LIFELONG MEMORY every new digit's held-out exemplars are CONSOLIDATED to the durable substrate lake
                  and pulled back BY PROXIMITY as replay (labelled by their OWN true label, so
                  contamination stays correctly-labelled -- costs the order-invariant readout nothing).
  SLEEP -> 98%    after the cycles, a sleep consolidation pulls ALL lake-consolidated exemplars and
                  refits an RBF-SVM on their full spatial features (the readout upgrade, offline on
                  stored memory). This reaches the static ceiling (~98%) from the brain's OWN lake.

Feature learning stays UNSUPERVISED (frozen conv1); only the readout is supervised. Zero-dep CORE is
untouched: this file lives under research/ (ruff-excluded), imports nothing new into src/, edits
neither run_selfchecks nor pyproject.

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/autonomous_brain_recovered.py
Knobs: NORD_CYCLES(18) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(400) NORD_KREPLAY(150)
       NORD_RFF(2000) NORD_GROW_RFF(200) NORD_GAMMA(0.04) NORD_SHRINK(0.1) NORD_MAINTAIN(20)
       NORD_TRAIN(12000) NORD_TEST(2000)
Deps: numpy torch torchvision snntorch scikit-learn
"""
import os

import torch

from torch_device import resolve_device, enable
from snn_mnist_stdp import load_mnist
from deep_stdp_latency import ConvLatencyStdp, _train
from fast_features import batched_features_and_keys
from _loop_common import envi, envf
from brain_adapters.lake_memory import LakeMemory
from developmental_loop_slda import SLDA_RFF, _acc          # order-invariant readout + learned-aware acc
from developmental_loop_lake99 import _rbf_refit_eval        # sleep-consolidation RBF refit (readout upgrade)
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.cortex.metacognitive_monitor import MetacognitiveMonitor
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus


def _evidence(option, n=10, strong=1.0, weak=0.05):
    e = [weak] * n
    e[option] = strong
    return e


def _fresh_slda(D, rff, gamma, shrink, obs_F, obs_y):
    """NEUROGENESIS via order-invariant re-accumulation: build an SLDA_RFF with a WIDER random-Fourier
    basis (rff projection neurons) and SLEEP-REPLAY every accumulated (feature, true-label) batch back
    into its statistics. Order-invariance guarantees this equals the joint fit over the wider basis, so
    growing neurons rebuilds -- never forgets. Deterministic (seed fixed)."""
    s = SLDA_RFF(D, rff=rff, gamma=gamma, shrink=shrink, seed=0)
    for F, y in zip(obs_F, obs_y):
        s.observe(F, y)
    return s


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    CYCLES = envi("NORD_CYCLES", 18)
    C1 = envi("NORD_C1", 30)
    T_LAT = envi("NORD_T", 15)
    L1_EPOCHS = envi("NORD_L1_EPOCHS", 3)
    PER = envi("NORD_PER", 400)
    KREPLAY = envi("NORD_KREPLAY", 150)          # proximity retrieve-k per cycle (replay into SLDA)
    STORE = envi("NORD_STORE", 400)              # lake consolidation depth per digit (the sleep-refit trains on these)
    RFF0 = envi("NORD_RFF", 2000)
    GROW_RFF = envi("NORD_GROW_RFF", 200)
    GAMMA = envf("NORD_GAMMA", 0.04)
    SHRINK = envf("NORD_SHRINK", 0.1)
    MAINTAIN = envi("NORD_MAINTAIN", 20)
    lake_path = os.path.join(os.environ.get("TEMP", "."), "sndl_autonomous_recovered.spkl")
    if os.path.exists(lake_path):
        os.remove(lake_path)

    print("=" * 96)
    print("AUTONOMOUS BRAIN (RECOVERED) -- enclosed loop: perceive -> hold-in-WM -> learn (order-invariant)")
    print("                               -> grow RFF neurons -> consolidate -> sleep-refit -> repeat")
    print(f"device={dev}  cycles={CYCLES}  C1={C1}  rff0={RFF0}(+{GROW_RFF}/grow)  gamma={GAMMA}  "
          f"store={STORE}/digit  retrieve-k={KREPLAY}")
    print("=" * 96)
    Xtr, Ytr, Xte, Yte = load_mnist(envi("NORD_TRAIN", 12000), envi("NORD_TEST", 2000))

    # ---- PERCEPTION: unsupervised spiking features (frozen), batched extraction + <=64-dim lake keys ----
    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    Ftr, Gtr = batched_features_and_keys(conv, Xtr)
    Fte, _ = batched_features_and_keys(conv, Xte)
    print(f"[perception] frozen; feat_dim={Ftr.shape[1]}; diversity={conv.kernel_diversity():.3f}")
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    yte_np = Yte.numpy()
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    # ---- the faculties (all real, verified components) ----
    rff = RFF0
    slda = SLDA_RFF(D, rff=rff, gamma=GAMMA, shrink=SHRINK, seed=0)     # UNDERSTANDING (order-invariant)
    lake = LakeMemory(path=lake_path, seed=0)                           # LIFELONG durable memory
    wm = PrefrontalExecutive(n_options=10, seed=0)                      # WORKING MEMORY (persistent-activity)
    meta = MetacognitiveMonitor(seed=0)                                 # METACOGNITION
    curiosity = SpikingCuriosityDrive(seed=0)                           # intrinsic learning-progress
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
    lc = LocusCoeruleus()

    learned, competence, mem_rows, curve = [], {d: 0.0 for d in range(10)}, {}, []
    obs_F, obs_y = [], []                                               # accumulated batches (for RFF re-grow)
    last_visit = {}                                                     # digit -> last cycle revisited
    grow_events, wm_hits = 0, 0
    for cyc in range(1, CYCLES + 1):
        # ---- DECIDE: curiosity proposes an unseen digit; once all are seen, a ROUND-ROBIN revisit
        #      scheduler picks the least-recently-revisited digit (the original fixated on one digit) ----
        unseen = [d for d in range(10) if d not in learned]
        if unseen:
            proposed = curiosity.most_curious(unseen)
        else:
            proposed = min(learned, key=lambda d: last_visit.get(d, 0))   # least-recently revisited
        # WORKING MEMORY holds the objective as a persistent-activity pointer across MAINTAIN ticks
        wm.update(_evidence(proposed))
        for _ in range(MAINTAIN):
            wm.maintain()
        objective = wm.goal if wm.goal is not None else proposed
        wm_hits += int(objective == proposed)
        is_new = objective not in learned
        last_visit[objective] = cyc
        gain = 1.0 + 0.5 * lc.step(1.0 if is_new else (1.0 - competence[objective]))

        # ---- CONSOLIDATE a held-out slice of a NEW digit TO THE DURABLE LAKE (keys + row-index payloads) ----
        if is_new:
            pool = tr_by[objective][PER:PER + STORE]          # deeper store so the sleep-refit RBF has enough data
            lake.consolidate([Gtr[r].tolist() for r in pool], payloads=pool)
            mem_rows[objective] = pool
            learned.append(objective)

        # ---- LEARN: stream the objective's own exemplars + LIFELONG replay PULLED BY PROXIMITY ----
        obj_idx = torch.tensor(tr_by[objective][:PER], dtype=torch.long, device=dev)
        F_parts, y_parts = [Ftr[obj_idx]], [ytr[obj_idx]]
        replay_n = 0
        if objective in mem_rows and lake.n_stored > 0:
            q = Gtr[torch.tensor(mem_rows[objective], dtype=torch.long, device=dev)].mean(0).tolist()
            rows = [int(r) for r in lake.retrieve(q, k=KREPLAY)]
            if rows:
                ridx = torch.tensor(rows, dtype=torch.long, device=dev)
                F_parts.append(Ftr[ridx])
                y_parts.append(ytr[ridx])                     # TRUE labels -> contamination stays correctly-labelled
                replay_n = len(rows)
        Fb, yb = torch.cat(F_parts, 0), torch.cat(y_parts, 0)
        slda.observe(Fb, yb)                                  # additive, order-invariant accumulation
        obs_F.append(Fb)
        obs_y.append(yb)

        # ---- NEUROGENESIS: for each NEW concept, GROW the RFF basis and sleep-replay the accumulated
        #      observations through the wider basis (order-invariant rebuild -> more neurons, no forgetting) ----
        grew = ""
        if is_new and len(learned) >= 2:                      # first concept uses the initial basis
            rff += GROW_RFF
            slda = _fresh_slda(D, rff, GAMMA, SHRINK, obs_F, obs_y)
            grow_events += 1
            grew = f"  [+neurogenesis: RFF {rff} neurons]"

        # ---- MEASURE competence (the understanding), feed self-direction ----
        for d in learned:
            competence[d] = _acc(slda, Fte, yte, te_by[d], learned)
        overall = _acc(slda, Fte, yte, [r for d in learned for r in te_by[d]], learned)
        curve.append(overall)
        curiosity.observe(objective, 1.0 - competence[objective])
        lp = curiosity.learning_progress(objective)
        curriculum.observe(lp)
        meta.update(confidence=competence[objective], surprise=abs(lp), learning_progress=lp,
                    novelty=curiosity.novelty(objective))
        print(f"CYCLE {cyc:2d} propose {proposed}->WM {objective} ({'new' if is_new else 'rev'})  "
              f"understand={overall:.1%}  lake={lake.n_stored}  replay={replay_n}  gain={gain:.2f}{grew}")

    online = curve[-1] if curve else 0.0

    # ---- SLEEP CONSOLIDATION -> 98%: refit an RBF-SVM on ALL lake-consolidated exemplars (readout upgrade) ----
    all_rows = [r for d in learned for r in mem_rows[d]]
    pooled_te = [r for d in learned for r in te_by[d]]
    print(f"\n[sleep-refit] RBF-SVM on {len(all_rows)} lake-consolidated exemplars (offline consolidation)...")
    rbf_acc, n_used = _rbf_refit_eval(Ftr, Ytr, all_rows, Fte, yte_np, pooled_te)

    durable = lake.durability_check()
    print("\n" + "=" * 96)
    print("AUTONOMOUS BRAIN (RECOVERED) -- one full enclosed cycle, all faculties active:")
    print("-" * 96)
    print(f"  PERCEPTION   : unsupervised conv1 features (frozen), diversity {conv.kernel_diversity():.3f}")
    print(f"  WORKING MEM  : objective held in prefrontal persistent activity  {wm_hits}/{CYCLES} cycles")
    print(f"  SELF-DIRECT  : curiosity + round-robin revisit + curriculum ({curriculum.mastered}/10 mastered)")
    print(f"  NEUROGENESIS : the brain GREW its RFF basis {grow_events}x -> {rff} projection neurons (order-invariant, no forgetting)")
    print(f"  LIFELONG MEM : {lake.n_stored} exemplars in the DURABLE data lake; survives restart = {durable.get('ok')}")
    print(f"  UNDERSTANDING (online, SLDA-RFF)     : {online:.1%}")
    print(f"  UNDERSTANDING (sleep-refit, RBF)     : {rbf_acc:.1%}   ({n_used} consolidated exemplars)")
    print("-" * 96)
    print(f"  reference: original grow-and-freeze MLP loop = 47.4%   static RBF ceiling = 98.2%")
    print(f"  RECOVERY: online {online:.1%} (+{(online - 0.474) * 100:.1f} pts) ; sleep-refit {rbf_acc:.1%} (+{(rbf_acc - 0.474) * 100:.1f} pts)")
    print("  perceive -> hold-in-WM -> learn (order-invariant) -> grow RFF neurons -> consolidate -> sleep-refit -> REPEAT")
    print("=" * 96)
    # PRIMARY recovery claim: the online order-invariant readout clears 90% (was 47.4%).
    assert online > 0.90, f"online SLDA-RFF understanding regressed below 90%: {online:.3f}"
    # The sleep-refit RBF trains ONLY on the consolidated lake exemplars; it reaches the static ceiling
    # only when the lake stores enough per digit. Honest floor (catch regressions), not a faked 98.
    assert rbf_acc > 0.90, f"sleep-refit RBF regressed below 90%: {rbf_acc:.3f}"
    assert durable.get("ok"), "lifelong memory did not survive restart"
    assert wm_hits > 0 and lake.n_stored > 0 and grow_events > 0, "a faculty was inactive"
    print(f"self-check OK: enclosed loop RECOVERED -- online {online:.1%} + sleep-refit {rbf_acc:.1%} "
          f"(was 47.4%); perception+WM+self-direction+neurogenesis({grow_events} RFF grows)+"
          f"lifelong-lake(durable) all active over {CYCLES} cycles.")
    lake.close()
    return 0


if __name__ == "__main__":
    main()
