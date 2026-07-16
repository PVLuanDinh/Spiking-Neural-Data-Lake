r"""
LAKE LOOP -> toward 99%: fix the proximity KEY + add an RBF SLEEP-REFIT on the lake replay.

developmental_loop_lake.py hit 90.2% because (a) it keyed proximity retrieval on a COARSE 30-dim
global feature (contaminated replay, -3.1%) and (b) it used a continual LINEAR head (the 93->98
linear-vs-RBF gap). This file closes both, on the same durable lake:

  1. RICHER KEY: the proximity key is a fixed RANDOM PROJECTION of the full 1080-dim conv1 SPATIAL
     feature down to a <=64-dim vector (sigmoid-mapped to [0,1]) -- label-free, deterministic,
     distance-preserving (Johnson-Lindenstrauss) -- so content-addressable retrieval pulls cleaner,
     less-contaminated replay than the crude global-max key.
  2. RBF SLEEP-REFIT: the continual linear head still runs online with proximity replay, but a
     "sleep" consolidation periodically pulls ALL lake-consolidated exemplars and REFITS an RBF-SVM
     on their full spatial features -- the readout UPGRADE (imports the static ~98% RBF into the loop).

Reports both the fast continual-linear head AND the RBF sleep-refit head, vs the 90.2% baseline and
the 98.2% static ceiling. Feature learning stays unsupervised; only the readouts are supervised.

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/developmental_loop_lake99.py
Knobs: NORD_CYCLES(15) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(300) NORD_REPLAY(300)
       NORD_KREPLAY(100) NORD_SGD(300) NORD_KEYDIM(48) NORD_TRAIN(12000) NORD_TEST(2000)
Deps: numpy torch torchvision snntorch scikit-learn
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from torch_device import resolve_device, enable
from snn_mnist_stdp import load_mnist
from deep_stdp_latency import ConvLatencyStdp, _train
from brain_adapters.lake_memory import LakeMemory
from _loop_common import envi as _envi, envf as _envf, LinearReadout, cache_features, acc as _acc






def _rbf_refit_eval(Ftr, Ytr, rows, Fte, yte_np, pooled_te):
    """SLEEP consolidation: refit an RBF-SVM on ALL lake-consolidated exemplars' full spatial
    features, evaluate on the held-out test. This is the readout UPGRADE (linear -> RBF)."""
    Xc = Ftr[rows].cpu().numpy()
    yc = Ytr[rows].numpy()
    sc = StandardScaler().fit(Xc)
    rbf = SVC(kernel="rbf", gamma="scale", decision_function_shape="ovr", random_state=0)
    rbf.fit(sc.transform(Xc), yc)
    pred = rbf.predict(sc.transform(Fte[pooled_te].cpu().numpy()))
    return float((pred == yte_np[pooled_te]).mean()), len(rows)


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    CYCLES = _envi("NORD_CYCLES", 15)
    C1 = _envi("NORD_C1", 30)
    T_LAT = _envi("NORD_T", 15)
    L1_EPOCHS = _envi("NORD_L1_EPOCHS", 3)
    PER = _envi("NORD_PER", 300)
    REPLAY = _envi("NORD_REPLAY", 300)          # exemplars consolidated per digit (the RBF sleep-refit trains on these)
    KREPLAY = _envi("NORD_KREPLAY", 100)        # proximity-replay pulled per class per cycle (for the linear head)
    SGD = _envi("NORD_SGD", 300)
    KEYDIM = _envi("NORD_KEYDIM", 48)           # richer proximity key dim (<=64)
    TRAIN_N = _envi("NORD_TRAIN", 12000)
    TEST_N = _envi("NORD_TEST", 2000)
    lake_path = os.path.join(os.environ.get("TEMP", "."), "sndl_lake99.spkl")

    print("=" * 88)
    print("LAKE LOOP -> 99: richer projected proximity KEY + RBF SLEEP-REFIT on the lake replay")
    print(f"device={dev}  cycles={CYCLES}  C1={C1}  key_dim={KEYDIM}  consolidate/digit={REPLAY}  "
          f"proximity_k={KREPLAY}")
    print("=" * 88)
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    print(f"[perception] frozen; feat_dim={conv.feat_dim()}; diversity={conv.kernel_diversity():.3f}")

    print("[cache] conv1 SPATIAL features + richer projected KEYS...")
    Ftr = cache_features(conv, Xtr, dev)
    Fte = cache_features(conv, Xte, dev)
    D = Ftr.shape[1]
    # RICHER KEY: fixed random projection of the full spatial feature -> <=64-dim, sigmoid to [0,1].
    torch.manual_seed(12345)
    R = (torch.randn(D, KEYDIM, device=dev) * 0.12)
    Gtr = torch.sigmoid(Ftr @ R)                # [Ntr, KEYDIM] in (0,1) -- label-free richer proximity key
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    yte_np = Yte.numpy()
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    readout = LinearReadout(D, seed=0)
    lake = LakeMemory(path=lake_path, seed=0)
    from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
    from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus
    curiosity = SpikingCuriosityDrive(seed=0)
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
    lc = LocusCoeruleus()

    learned, competence, mem_rows, curve = [], {d: 0.0 for d in range(10)}, {}, []
    for cyc in range(1, CYCLES + 1):
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        objective = curiosity.most_curious(candidates)
        is_new = objective not in learned
        gain = 1.0 + 0.5 * lc.step(1.0 if is_new else (1.0 - competence[objective]))

        # replay pulled BY PROXIMITY using the richer projected key
        replay_rows = []
        for d in [x for x in learned if x != objective]:
            q = Gtr[mem_rows[d]].mean(0)
            replay_rows.extend(int(r) for r in lake.retrieve(q.tolist(), k=KREPLAY))
        n_train = PER if is_new else max(KREPLAY, PER // 4)
        idx = torch.tensor(tr_by[objective][:n_train] + replay_rows, dtype=torch.long, device=dev)
        readout.train_step(Ftr[idx], ytr[idx], lr=0.01 * gain, steps=SGD)

        if is_new:
            pool = tr_by[objective][n_train:n_train + REPLAY]
            lake.consolidate([Gtr[r].tolist() for r in pool], payloads=pool)   # richer key -> durable lake
            mem_rows[objective] = pool
            learned.append(objective)

        for d in learned:
            competence[d] = _acc(readout, Fte, yte, te_by[d])
        overall = _acc(readout, Fte, yte, [r for d in learned for r in te_by[d]])
        curve.append(overall)
        curiosity.observe(objective, 1.0 - competence[objective])
        curriculum.observe(curiosity.learning_progress(objective))
        print(f"CYCLE {cyc:2d} obj={objective} {'new' if is_new else 'rev'}  lake={lake.n_stored}  "
              f"proximity_replay={len(replay_rows)}  linear acc over {sorted(learned)} = {overall:.1%}")

    # ---- SLEEP CONSOLIDATION: refit an RBF-SVM on ALL lake-consolidated exemplars (readout upgrade) ----
    all_rows = [r for d in learned for r in mem_rows[d]]
    pooled_te = [r for d in learned for r in te_by[d]]
    print(f"\n[sleep-refit] RBF-SVM on {len(all_rows)} lake-consolidated exemplars (readout upgrade)...")
    rbf_acc, n_used = _rbf_refit_eval(Ftr, Ytr, all_rows, Fte, yte_np, pooled_te)
    lin_final = curve[-1] if curve else 0.0
    durable = lake.durability_check()

    print("\n" + "=" * 88)
    print("RESULT (proximity-replay lake loop, toward 99%)")
    print("-" * 88)
    print(f"  continual LINEAR head (fast, proximity replay)   : {lin_final:.1%}")
    print(f"  RBF SLEEP-REFIT head (consolidated, {n_used} exemplars) : {rbf_acc:.1%}")
    print("-" * 88)
    print(f"  baseline lake loop (crude key, linear only)      : 90.2%")
    print(f"  static ceiling (RBF ensemble, 30k)               : 98.2%")
    print(f"  richer-key linear delta vs 90.2%                 : {lin_final - 0.902:+.1%}")
    print(f"  RBF sleep-refit delta vs baseline 90.2%          : {rbf_acc - 0.902:+.1%}")
    print(f"  DURABILITY (lake survives close+reopen)          : {durable.get('ok')}")
    print("=" * 88)
    assert lin_final > 0.85 and rbf_acc > 0.85, "a readout regressed below its ~90%+ claim"   # catch regressions
    assert durable.get("ok"), "lake memory did not survive reopen"
    print(f"self-check OK: richer-key linear {lin_final:.1%}, RBF sleep-refit {rbf_acc:.1%} "
          f"(baseline 90.2%); {n_used} exemplars durable in the lake.")
    return 0


if __name__ == "__main__":
    main()
