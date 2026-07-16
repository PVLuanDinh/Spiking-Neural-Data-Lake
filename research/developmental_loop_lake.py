r"""
LAKE-BACKED DEVELOPMENTAL LOOP -- the loop consolidates its long-term memory to the substrate
DATA LAKE (durable) and PULLS replay from it BY PROXIMITY (content-addressable), instead of a
random in-RAM ring.

vs developmental_loop_deep.py (RAM ring, random replay): here every learned digit's exemplars are
CONSOLIDATED into the durable spike lake via `brain_adapters.lake_memory.LakeMemory`
(StreamingHub .spkl + ProximityIndex over Victor-Purpura spike-train distance). Each cycle, replay
is RETRIEVED BY PROXIMITY -- query the lake with a class-representative key, get back the nearest
stored exemplars (content-addressable recall) -- the drop-in for random sampling. The memory is
DURABLE (survives close+reopen), not a bounded RAM ring.

Key detail: the ProximityIndex metric is defined over <=64 channels, so the PROXIMITY KEY is the
compact C1-dim (<=64) GLOBAL conv1 feature (one latency value per feature map). The PAYLOAD is the
exemplar's train-row-index, which points at the full spatial feature actually replayed. So the lake
holds compact spiking keys; retrieval returns row-indices; the loop replays the full features.

Feature learning stays UNSUPERVISED (frozen conv1). Only the linear readout is supervised.
The honest test: does content-addressable proximity replay from the durable lake HOLD the deep
loop's ~93.3% (vs random RAM replay), while proving the memory is durable?

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/developmental_loop_lake.py
Knobs: NORD_CYCLES(15) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(300) NORD_REPLAY(150)
       NORD_SGD(300) NORD_TRAIN(10000) NORD_TEST(2000)
Deps: numpy torch torchvision snntorch  (research/GPU tier; core stays zero-dep)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_device import resolve_device, enable
from snn_mnist_stdp import load_mnist
from deep_stdp_latency import ConvLatencyStdp, _train
from brain_adapters.lake_memory import LakeMemory


from _loop_common import envi as _envi, LinearReadout, cache_features, acc as _acc   # noqa: E402 (shared DRY helpers)


def _global_key(conv, img):
    """Compact C1-dim (<=64) proximity KEY: one global latency value per conv1 feature map.
    fired[C1,h,w] -> (t_lat - fired)/t_lat, global-max over space -> length C1 in [0,1]."""
    st = conv._dog_latency(img)
    fired, _ = conv._forward(st)
    val = torch.where(fired < conv.t_lat, (conv.t_lat - fired).float() / conv.t_lat,
                      torch.zeros_like(fired, dtype=torch.float32))
    return val.amax(dim=(1, 2))            # [C1]




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
    TRAIN_N = _envi("NORD_TRAIN", 10000)
    TEST_N = _envi("NORD_TEST", 2000)
    scratch = os.environ.get("TEMP", ".")
    lake_path = os.path.join(scratch, "sndl_devloop_lake.spkl")

    print("=" * 84)
    print("LAKE-BACKED DEVELOPMENTAL LOOP -- consolidate to the durable data lake, replay BY PROXIMITY")
    print(f"device={dev}  cycles={CYCLES}  C1={C1}  per={PER}  replay(proximity-k)={REPLAY}  lake={lake_path}")
    print("=" * 84)
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    print(f"[perception] frozen; feat_dim={conv.feat_dim()}; diversity={conv.kernel_diversity():.3f}")

    print("[cache] conv1 SPATIAL features (train+test) + compact C1 global KEYS...")
    Ftr = cache_features(conv, Xtr, dev)
    Fte = cache_features(conv, Xte, dev)
    Gtr = torch.stack([_global_key(conv, img) for img in Xtr], 0)     # [Ntr, C1] proximity keys
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    readout = LinearReadout(D, seed=0)
    lake = LakeMemory(path=lake_path, seed=0)            # DURABLE substrate lake + proximity index
    from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
    from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus
    curiosity = SpikingCuriosityDrive(seed=0)
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)
    lc = LocusCoeruleus()

    learned, competence, mem_rows = [], {d: 0.0 for d in range(10)}, {}
    curve = []
    for cyc in range(1, CYCLES + 1):
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        objective = curiosity.most_curious(candidates)
        is_new = objective not in learned
        salience = 1.0 if is_new else (1.0 - competence[objective])
        gain = 1.0 + 0.5 * lc.step(salience)

        # ---- REPLAY PULLED BY PROXIMITY from the durable lake (content-addressable recall) ----
        replay_rows = []
        for d in [x for x in learned if x != objective]:
            q = Gtr[mem_rows[d]].mean(0)                              # class-representative KEY for digit d
            got = lake.retrieve(q.tolist(), k=REPLAY)                 # nearest stored exemplars from the lake
            replay_rows.extend(int(r) for r in got)

        n_train = PER if is_new else max(REPLAY, PER // 4)
        obj_rows = tr_by[objective][:n_train]
        idx = torch.tensor(obj_rows + replay_rows, dtype=torch.long, device=dev)
        readout.train_step(Ftr[idx], ytr[idx], lr=0.01 * gain, steps=SGD)

        # ---- CONSOLIDATE this digit's held-out exemplars TO THE DATA LAKE (durable) ----
        if is_new:
            pool = tr_by[objective][n_train:n_train + REPLAY]
            lake.consolidate([Gtr[r].tolist() for r in pool], payloads=pool)   # keys -> durable lake + index
            mem_rows[objective] = pool
            learned.append(objective)

        for d in learned:
            competence[d] = _acc(readout, Fte, yte, te_by[d])
        overall = _acc(readout, Fte, yte, [r for d in learned for r in te_by[d]])
        curve.append(overall)
        curiosity.observe(objective, 1.0 - competence[objective])
        curriculum.observe(curiosity.learning_progress(objective))
        print(f"CYCLE {cyc:2d} obj={objective} {'new' if is_new else 'rev'}  "
              f"lake_stored={lake.n_stored}  proximity_replay={len(replay_rows)}  "
              f"acc over {sorted(learned)} = {overall:.1%}")

    # ---- durability proof: the consolidated memory survives a lake close+reopen ----
    durable = lake.durability_check()
    final = curve[-1] if curve else 0.0
    print("\n" + "=" * 84)
    print(f"LAKE LOOP COMPLETE: learned {sorted(learned)}; final held-out accuracy {final:.1%}")
    print(f"  long-term memory: {lake.n_stored} exemplars CONSOLIDATED to the durable data lake ({lake_path})")
    print(f"  replay each cycle was PULLED BY PROXIMITY (content-addressable), not random RAM sampling")
    print(f"  DURABILITY (close+reopen the .spkl, memory survives): {durable}")
    print(f"  reference: developmental_loop_deep.py (RAM ring, random replay) = 93.3%")
    print("=" * 84)
    assert final > 0.85, f"lake loop regressed below its ~90% claim: {final:.2f}"   # catch regressions, not just chance
    assert durable, "consolidated memory did NOT survive the lake reopen"
    print(f"self-check OK: lake-backed loop {final:.1%} (proximity replay from the durable lake); "
          f"{lake.n_stored} exemplars durable across restart; feature learning unsupervised.")
    return 0


if __name__ == "__main__":
    main()
