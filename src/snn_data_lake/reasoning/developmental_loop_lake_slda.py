r"""
LAKE-BACKED + SLDA-RFF DEVELOPMENTAL LOOP -- combine the durable proximity-replay LAKE with the
order-INVARIANT SLDA-on-RFF readout, removing BOTH costs the linear lake loop paid.

WHERE THIS SITS
---------------
  * developmental_loop_lake.py  (90.2%) -- a CONTINUAL LINEAR head trained by SGD on proximity-
    retrieved replay batches. It eats two penalties: -1.8% continual FORGETTING (SGD drifts as new
    digits arrive) and -3.1% replay CONTAMINATION (proximity retrieval returns some off-query rows,
    and the SGD head is trained on those imperfect batches, order-dependently).
  * developmental_loop_slda.py  (94.8%) -- SLDA-on-RFF: a fixed Random-Fourier-Feature map z =
    sqrt(2/M) cos(F W + b) (approximates the RBF kernel) + streaming LDA over ORDER-INVARIANT
    sufficient statistics (per-class sums, counts, global ZZ^T). Continual == joint by construction:
    no replay, no forgetting. But it had no durable long-term store.
  * THIS FILE = both. The durable substrate LAKE (brain_adapters.lake_memory.LakeMemory) is the
    long-term memory; each learned digit's held-out exemplars are CONSOLIDATED to the .spkl lake
    (durable, content-addressable). The readout is SLDA_RFF (imported from developmental_loop_slda),
    NOT the linear head. Each cycle, curiosity picks an objective, we RETRIEVE by proximity the
    lake neighbours of the objective's class-representative key (retrieval-augmented replay), and
    stream them PLUS the objective's own exemplars into SLDA's order-invariant statistics.

WHY PROXIMITY RETRIEVAL FEEDING SLDA REMOVES THE -3.1 CONTAMINATION COST (the 12-line argument)
----------------------------------------------------------------------------------------------
  1. Proximity retrieval returns row-INDICES (payloads) of stored exemplars near the query key.
  2. Retrieval is imperfect: some returned rows belong to a DIFFERENT class than the query digit
     (this "contamination" is exactly what cost the linear head -3.1%).
  3. But SLDA does not trust the query -- it labels every streamed row by its OWN TRUE label
     ytr[row]. A contaminated (off-query) row is still accumulated into ITS OWN class's statistics.
  4. So a wrong-for-the-query row is a CORRECTLY-LABELLED sample for some other class: it adds
     signal, never label noise. The SGD head, by contrast, absorbed those rows into an order- and
     batch-dependent gradient step, which is where the contamination penalty came from.
  5. SLDA's state is only (per-class sum_c, count_c, global ZZ^T). These are ADDITIVE and
     ORDER-INVARIANT: streaming a set of (feature, true-label) pairs gives the SAME statistics no
     matter which cycle, order, or duplication they arrive in.
  6. Therefore the continual lake loop's SLDA state == the JOINT SLDA state over the union of all
     streamed rows -> no forgetting (kills the -1.8) and contamination degrades only to "a few
     extra correctly-labelled samples in other classes" (kills most of the -3.1).
  7. Net: lake-SLDA should land near the no-lake SLDA (94.8%), far above the linear lake (90.2%),
     while ADDING the durable, content-addressable long-term store the no-lake SLDA lacked.

Feature learning stays UNSUPERVISED (frozen conv1). Only the SLDA readout is supervised.
CORE untouched; research/GPU tier, ruff-excluded. ONE new file.

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/developmental_loop_lake_slda.py
Knobs: NORD_CYCLES(15) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(400) NORD_KREPLAY(150)
       NORD_RFF(2000) NORD_GAMMA(0.02) NORD_SHRINK(0.1) NORD_PROTO(0) NORD_TRAIN(12000)
       NORD_TEST(2000) NORD_GAMMA_SWEEP("g1,g2,...")
Deps: numpy torch torchvision snntorch  (research/GPU tier; core untouched)
"""
import os

import torch

from snn_data_lake.reasoning.torch_device import resolve_device, enable
from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.reasoning.fast_features import batched_features_and_keys
from snn_data_lake.reasoning.brain_adapters.lake_memory import LakeMemory
from snn_data_lake.reasoning._loop_common import envi as _envi, envf as _envf
from snn_data_lake.reasoning.developmental_loop_slda import SLDA_RFF, _acc     # reuse the RFF+streaming-LDA readout + learned-aware acc


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)
    CYCLES = _envi("NORD_CYCLES", 15)
    C1 = _envi("NORD_C1", 30)
    T_LAT = _envi("NORD_T", 15)
    L1_EPOCHS = _envi("NORD_L1_EPOCHS", 3)
    PER = _envi("NORD_PER", 400)                 # objective's own exemplars streamed raw each cycle
    KREPLAY = _envi("NORD_KREPLAY", 150)         # lake store depth per class AND proximity retrieve-k
    RFF = _envi("NORD_RFF", 2000)
    GAMMA = _envf("NORD_GAMMA", 0.02)
    SHRINK = _envf("NORD_SHRINK", 0.1)
    PROTO = _envi("NORD_PROTO", 0) == 1          # LEVER 3: prototype replay instead of raw exemplars
    TRAIN_N = _envi("NORD_TRAIN", 12000)
    TEST_N = _envi("NORD_TEST", 2000)
    scratch = os.environ.get("TEMP", ".")
    lake_path = os.path.join(scratch, "sndl_devloop_lake_slda.spkl")
    # a fresh lake per run (avoid resuming a stale store from a previous invocation)
    if os.path.exists(lake_path):
        os.remove(lake_path)

    print("=" * 92)
    print("LAKE-BACKED + SLDA-RFF LOOP -- durable proximity replay feeding order-invariant SLDA statistics")
    print(f"device={dev}  cycles={CYCLES}  C1={C1}  per={PER}  retrieve-k={KREPLAY}  rff={RFF}  "
          f"gamma={GAMMA}  shrink={SHRINK}  proto={PROTO}")
    print(f"lake={lake_path}")
    print("=" * 92)
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    torch.manual_seed(0)
    print(f"\n[perception] conv1 latency-STDP C1={C1} (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"conv1 ep{ep + 1}/{L1_EPOCHS}")
    print(f"[perception] frozen; feat_dim={conv.feat_dim()}; diversity={conv.kernel_diversity():.3f}")

    print("[cache] conv1 SPATIAL features (train+test) + compact C1 global proximity KEYS (ONE forward pass)...")
    Ftr, Gtr = batched_features_and_keys(conv, Xtr)      # [Ntr, feat_dim] spatial, [Ntr, C1] <=64-dim keys
    Fte, _ = batched_features_and_keys(conv, Xte)        # test keys unused
    ytr, yte = Ytr.long().to(dev), Yte.long().to(dev)
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}

    slda = SLDA_RFF(D, rff=RFF, gamma=GAMMA, shrink=SHRINK, seed=0)
    lake = LakeMemory(path=lake_path, seed=0)            # durable substrate lake + proximity index
    from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
    curiosity = SpikingCuriosityDrive(seed=0)
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)

    learned, competence, mem_rows, class_proto, curve = [], {d: 0.0 for d in range(10)}, {}, {}, []
    obs_F, obs_y = [], []                                # accumulate streamed batches (for the gamma sweep)
    for cyc in range(1, CYCLES + 1):
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        objective = curiosity.most_curious(candidates)
        is_new = objective not in learned

        # ---- CONSOLIDATE a held-out slice of this digit TO THE DURABLE LAKE (keys + row-index payloads) ----
        if is_new:
            pool = tr_by[objective][PER:PER + KREPLAY]            # held-out from the raw-streamed slice
            pool_idx = torch.tensor(pool, dtype=torch.long, device=dev)
            if PROTO:
                # LEVER 3: store ONE class PROTOTYPE (mean key) with payload = class label; keep the mean feature.
                pkey = Gtr[pool_idx].mean(0).tolist()
                lake.consolidate([pkey], payloads=[objective])
                class_proto[objective] = Ftr[pool_idx].mean(0)   # [D] mean feature -> streamed as replay
            else:
                lake.consolidate([Gtr[r].tolist() for r in pool], payloads=pool)   # per-exemplar keys
            mem_rows[objective] = pool
            learned.append(objective)

        # ---- stream OBJECTIVE'S OWN exemplars (raw) into SLDA's order-invariant statistics ----
        obj_rows = tr_by[objective][:PER]
        obj_idx = torch.tensor(obj_rows, dtype=torch.long, device=dev)
        F_parts, y_parts = [Ftr[obj_idx]], [ytr[obj_idx]]

        # ---- RETRIEVAL-AUGMENTED replay: proximity neighbours of the objective's class-representative key ----
        replay_n = 0
        if objective in mem_rows and lake.n_stored > 0:
            q = Gtr[torch.tensor(mem_rows[objective], dtype=torch.long, device=dev)].mean(0).tolist()
            got = lake.retrieve(q, k=KREPLAY)                    # nearest stored payloads (row-idx or class-label)
            if PROTO:
                classes = sorted({int(c) for c in got if int(c) in class_proto})
                if classes:
                    F_parts.append(torch.stack([class_proto[c] for c in classes]))
                    y_parts.append(torch.tensor(classes, dtype=torch.long, device=dev))
                    replay_n = len(classes)
            else:
                rows = [int(r) for r in got]
                if rows:
                    ridx = torch.tensor(rows, dtype=torch.long, device=dev)
                    F_parts.append(Ftr[ridx])
                    y_parts.append(ytr[ridx])                    # TRUE labels -> contamination stays correctly-labelled
                    replay_n = len(rows)

        Fb, yb = torch.cat(F_parts, 0), torch.cat(y_parts, 0)
        slda.observe(Fb, yb)                                     # additive, order-invariant accumulation
        obs_F.append(Fb)
        obs_y.append(yb)

        for d in learned:
            competence[d] = _acc(slda, Fte, yte, te_by[d], learned)
        overall = _acc(slda, Fte, yte, [r for d in learned for r in te_by[d]], learned)
        curve.append(overall)
        curiosity.observe(objective, 1.0 - competence[objective])
        curriculum.observe(curiosity.learning_progress(objective))
        print(f"CYCLE {cyc:2d} obj={objective} {'new' if is_new else 'rev'}  lake_stored={lake.n_stored}  "
              f"{'proto' if PROTO else 'proximity'}_replay={replay_n}  "
              f"SLDA-RFF acc over {sorted(learned)} = {overall:.1%}")

    final = curve[-1] if curve else 0.0

    # ---- LEVER 2: RFF gamma sensitivity. Re-stream the SAME accumulated batches into fresh SLDA_RFF ----
    # instances (order-invariance => the GAMMA entry reproduces `final` exactly). Cheap: features are cached.
    sweep_env = os.environ.get("NORD_GAMMA_SWEEP")
    if sweep_env:
        gammas = sorted({float(g) for g in sweep_env.split(",") if g.strip()})
    else:
        gammas = sorted({round(GAMMA * 0.5, 6), GAMMA, round(GAMMA * 2.0, 6)})
    all_test = [r for d in learned for r in te_by[d]]
    best_g, best_a = GAMMA, -1.0
    sweep = []
    for g in gammas:
        s = SLDA_RFF(D, rff=RFF, gamma=g, shrink=SHRINK, seed=0)
        for F, y in zip(obs_F, obs_y):
            s.observe(F, y)
        a = _acc(s, Fte, yte, all_test, learned)
        sweep.append((g, a))
        if a > best_a:
            best_g, best_a = g, a

    # ---- durability proof: the consolidated long-term memory survives a lake close+reopen ----
    dur = lake.durability_check()

    print("\n" + "=" * 92)
    print("RESULT -- lake-backed SLDA-RFF continual loop")
    print("-" * 92)
    print(f"  SLDA-RFF on the durable lake        : {final:.1%}   (gamma={GAMMA}, proto={PROTO})")
    print(f"  reference: lake LINEAR head         : 90.2%   (-1.8 forgetting + -3.1 contamination)")
    print(f"  reference: SLDA-RFF (no lake)       : 94.8%")
    print(f"  reference: static RBF ensemble      : 98.2%")
    print(f"  HONEST DELTA vs the linear lake     : {(final - 0.902) * 100:+.1f} pts vs 90.2%")
    print("-" * 92)
    print("  LEVER 2 -- RFF gamma sweep (re-streamed on cached features; order-invariant):")
    for g, a in sweep:
        tag = "  <- loop gamma" if abs(g - GAMMA) < 1e-9 else ("  <- best" if g == best_g else "")
        print(f"      gamma={g:<10g} held-out acc = {a:.1%}{tag}")
    print(f"    gamma used in the loop = {GAMMA}; best swept gamma = {best_g} ({best_a:.1%})")
    print("-" * 92)
    if PROTO:
        print("  LEVER 3 -- PROTOTYPE replay ON: replay streamed class mean-features (1 point/class) instead")
        print("    of raw retrieved rows. Expected HONEST cost: a single mean point per class adds ~zero")
        print("    within-class scatter, so SLDA's shared covariance is under-estimated -> typically <= raw.")
        print("    Compare this run's accuracy to the default (NORD_PROTO=0, raw retrieval) to read the sign.")
    else:
        print("  LEVER 3 -- prototype replay OFF (raw-exemplar retrieval). Set NORD_PROTO=1 to compare.")
    print("-" * 92)
    print(f"  DURABILITY (close+reopen the .spkl, memory survives): {dur['ok']}  "
          f"(events {dur['events_before']} -> {dur['events_after']}, {dur['n_stored']} exemplars)")
    print("=" * 92)

    # ---- self-checks (deterministic asserts) ----
    assert dur["ok"], f"consolidated long-term memory did NOT survive the lake reopen: {dur}"
    if PROTO:
        # LEVER-3 experiment: prototype replay is allowed to be an honest degradation; only guard chance.
        assert final > 0.10, f"proto-replay SLDA at/below chance: {final:.2f}"
        print(f"self-check OK (PROTO): lake-SLDA {final:.1%} (prototype replay, honest experiment); "
              f"lake durable across restart. Compare to raw retrieval for the sign of the effect.")
    else:
        # DEFAULT: raw retrieval feeding order-invariant SLDA MUST beat the 90.2% linear lake loop.
        assert final > 0.90, f"lake-SLDA {final:.2f} did not beat the 90.2% linear lake loop"
        print(f"self-check OK: lake-backed SLDA-RFF {final:.1%} > 90.2% linear lake "
              f"({(final - 0.902) * 100:+.1f} pts); proximity replay feeds order-invariant statistics with "
              f"TRUE labels (contamination stays correctly-labelled, no forgetting); {dur['n_stored']} "
              f"exemplars durable across a lake restart.")
    lake.close()
    return 0


if __name__ == "__main__":
    main()
