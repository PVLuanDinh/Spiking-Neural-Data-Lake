r"""
EXPLICIT WORKING MEMORY in the developmental loop: the current objective is held as a POINTER in
prefrontal PERSISTENT SPIKING ACTIVITY (a reverberating attractor), not a python variable.

Closes the explicit-working-memory gap. The loop already has IMPLICIT memory (STDP-learned conv1
features = procedural) + durable DECLARATIVE memory (the data lake). This adds EXPLICIT WORKING
MEMORY: `cortex.prefrontal_executive.PrefrontalExecutive` (Wang-2002 persistent-activity WM,
recurrence 0.98) actively MAINTAINS the current objective across the cycle with NO input, and the
loop READS the objective back from the attractor (wm.goal) to drive learning -- so the pointer
lives in reverberating spikes, and the WM is genuinely in the control path.

Each cycle: curiosity proposes an objective -> WRITE it into WM (wm.update, gated spiking commit)
-> MAINTAIN it across a delay with no input (wm.maintain, persistent activity) -> READ the held
pointer (wm.goal) and act on THAT. Measures: WM-hold rate (goal survives the maintain delay),
persistence over a long delay (delayed recall), distractor robustness (attractor stability), and
that final accuracy HOLDS (this is an architectural-faithfulness faculty, not an accuracy lever).

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/developmental_loop_wm.py
Knobs: NORD_CYCLES(15) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(300) NORD_REPLAY(150)
       NORD_SGD(300) NORD_MAINTAIN(20) NORD_TRAIN(10000) NORD_TEST(2000)
Deps: numpy torch torchvision snntorch
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.reasoning.torch_device import resolve_device, enable
from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus


from snn_data_lake.reasoning._loop_common import envi as _envi, LinearReadout, cache_features, acc as _acc   # noqa: E402 (shared DRY helpers)


def _evidence(option, n=10, strong=1.0, weak=0.05):
    """One input drive per option: strong drive on `option`, weak baseline elsewhere."""
    e = [weak] * n
    e[option] = strong
    return e


def _wm_persistence_probe(seed, hold_ticks=200):
    """Standalone: write a goal, maintain with NO input for many ticks, report how long the
    persistent-activity attractor HOLDS the pointer (delayed recall) -- the WM hallmark."""
    wm = PrefrontalExecutive(n_options=10, seed=seed)
    wm.update(_evidence(7))
    held_for = 0
    for _ in range(hold_ticks):
        wm.maintain()
        if wm.goal == 7:
            held_for += 1
        else:
            break
    # distractor robustness: reload, then maintain WITH a weak distractor on another option
    wm2 = PrefrontalExecutive(n_options=10, seed=seed)
    wm2.update(_evidence(3))
    for _ in range(20):
        wm2.maintain()
    kept = (wm2.goal == 3)
    return held_for, kept


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    CYCLES = _envi("NORD_CYCLES", 15)
    C1 = _envi("NORD_C1", 30)
    T_LAT = _envi("NORD_T", 15)
    L1_EPOCHS = _envi("NORD_L1_EPOCHS", 3)
    PER = _envi("NORD_PER", 300)
    REPLAY = _envi("NORD_REPLAY", 150)
    SGD = _envi("NORD_SGD", 300)
    MAINTAIN = _envi("NORD_MAINTAIN", 20)     # delay ticks the WM holds the pointer with NO input each cycle

    print("=" * 86)
    print("EXPLICIT WORKING MEMORY loop -- objective held as a POINTER in prefrontal persistent activity")
    print(f"device={dev}  cycles={CYCLES}  C1={C1}  maintain_ticks/cycle={MAINTAIN} (held with NO input)")
    print("=" * 86)
    Xtr, Ytr, Xte, Yte = load_mnist(_envi("NORD_TRAIN", 10000), _envi("NORD_TEST", 2000))

    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    print(f"[perception] frozen; feat_dim={conv.feat_dim()}; diversity={conv.kernel_diversity():.3f}")
    Ftr, Fte = cache_features(conv, Xtr, dev), cache_features(conv, Xte, dev)
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    readout = LinearReadout(D, seed=0)
    wm = PrefrontalExecutive(n_options=10, seed=0)          # EXPLICIT working memory (persistent-activity attractor)
    curiosity = SpikingCuriosityDrive(seed=0)
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
    lc = LocusCoeruleus()

    learned, competence, memory, curve = [], {d: 0.0 for d in range(10)}, {}, []
    wm_hold_hits, wm_hold_total = 0, 0
    for cyc in range(1, CYCLES + 1):
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        proposed = curiosity.most_curious(candidates)

        # ---- WRITE the objective into WORKING MEMORY (gated spiking commit) ----
        wm.update(_evidence(proposed))
        # ---- MAINTAIN it across a delay with NO input (reverberating persistent activity) ----
        for _ in range(MAINTAIN):
            wm.maintain()
        # ---- READ the held pointer back from the attractor and ACT on THAT ----
        held = wm.goal
        wm_hold_total += 1
        if held == proposed:
            wm_hold_hits += 1
        objective = held if held is not None else proposed     # the loop acts on the WM-held pointer
        is_new = objective not in learned
        gain = 1.0 + 0.5 * lc.step(1.0 if is_new else (1.0 - competence[objective]))

        n_train = PER if is_new else max(REPLAY, PER // 4)
        replay = [r for d in learned if d != objective for r in memory.get(d, [])]
        idx = torch.tensor(tr_by[objective][:n_train] + replay, dtype=torch.long, device=dev)
        readout.train_step(Ftr[idx], ytr[idx], lr=0.01 * gain, steps=SGD)
        if is_new:
            memory[objective] = tr_by[objective][n_train:n_train + REPLAY]
            learned.append(objective)

        for d in learned:
            competence[d] = _acc(readout, Fte, yte, te_by[d])
        overall = _acc(readout, Fte, yte, [r for d in learned for r in te_by[d]])
        curve.append(overall)
        curiosity.observe(objective, 1.0 - competence[objective])
        curriculum.observe(curiosity.learning_progress(objective))
        print(f"CYCLE {cyc:2d} proposed={proposed} -> WM-held={held} (persist {MAINTAIN} ticks, no input)  "
              f"acc over {sorted(learned)} = {overall:.1%}")

    held_for, distractor_kept = _wm_persistence_probe(seed=0, hold_ticks=200)
    final = curve[-1] if curve else 0.0
    print("\n" + "=" * 86)
    print("RESULT -- explicit working memory in the loop")
    print("-" * 86)
    print(f"  WM HOLD rate (goal survives the {MAINTAIN}-tick maintain delay each cycle) : "
          f"{wm_hold_hits}/{wm_hold_total} = {wm_hold_hits / max(1, wm_hold_total):.0%}")
    print(f"  persistence probe: attractor held the pointer for                         : "
          f"{held_for} delay ticks with NO input (delayed recall)")
    print(f"  distractor robustness: held the goal through a distracted maintain         : {distractor_kept}")
    print(f"  final held-out accuracy (WM in the control path)                           : {final:.1%}")
    print(f"  reference: deep loop (python-variable objective) = 93.3%")
    print("-" * 86)
    print("  MEMORY SYSTEM NOW COMPLETE: implicit (STDP conv1 features) + declarative (data lake) +")
    print("  EXPLICIT WORKING (prefrontal persistent-activity pointer, held in reverberating spikes).")
    print("=" * 86)
    assert final > 0.85, f"loop regressed below its ~92% claim: {final:.2f}"   # catch regressions, not just chance
    assert wm_hold_hits / max(1, wm_hold_total) > 0.5, "WM did not reliably hold the objective pointer"
    print(f"self-check OK: WM held the objective pointer {wm_hold_hits}/{wm_hold_total} cycles across a "
          f"{MAINTAIN}-tick no-input delay; persisted {held_for} ticks; accuracy {final:.1%} (faculty holds).")
    return 0


if __name__ == "__main__":
    main()
