"""
Two-STAGE unsupervised conv-STDP + GLOBAL pooling + SVM readout on MNIST -- the
Kheradpisheh-2018 ~98% recipe, and an honest test of whether it beats the
single-stage conv1 ceiling (~95%, v3.53).  (torch/GPU tier, ruff-excluded; the
CORE is untouched -- this file imports nothing from src/ and edits
nothing there.)

WHY THIS FILE EXISTS (vs the earlier FAILED 2-stage)
----------------------------------------------------
research/deep_stdp_2stage.py stacked a second conv-STDP stage but did TWO things
that threw away the win:
  1. it SPATIALLY max-pooled conv2 down to a small ~4x4 grid, then FLATTENED that
     grid (conv2.feat_dim = C2 * ph * ph) -- discarding resolution AND coupling the
     feature to absolute position; and
  2. it read out with the weak neuron-label majority VOTE.
Result (v3.53): conv1+conv2 = 92.1% linear < conv1 alone = 95.1%. Depth HURT.

Kheradpisheh's ~98.4% does the OPPOSITE at the readout:
  * conv2 learns COMPOSITE object-part features (large receptive field, MANY maps);
  * GLOBAL pooling -- ONE value per conv2 map -> a compact, TRANSLATION-INVARIANT
    feature vector of length C2 (NOT a flattened spatial grid); and
  * an SVM classifier on those frozen unsupervised features.
GLOBAL pooling + SVM is the key difference. This file reproduces exactly THAT and
reports the honest delta, whatever it is.

WHAT THIS FILE DOES (feature learning stays 100% unsupervised)
--------------------------------------------------------------
The conv feature learning is byte-for-byte the SAME unsupervised spike-timing
machinery already built and validated: DoG ON/OFF -> intensity-bucketed latency ->
non-leaky IF -> first-spike two-axis WTA -> multiplicative soft-bound STDP (stage 1
= ConvLatencyStdp; stage 2 = ConvLatencyStdp2 over the stage-1 first-spike volley).
NO labels and NO backprop touch ANY conv kernel. The ONLY supervised step is the
SVM readout fit on the frozen features -- exactly the Kheradpisheh protocol (an SVM
trained on features an unsupervised net produced; the features never see a label).

GLOBAL POOLING (the one change) -- three feature sets are extracted:
  (a) conv1 SPATIAL   : the existing conv1.features() -- spatial max-pool of the
                        stage-1 first-spike-latency map, flattened + /max. This is
                        the ~95.1% baseline (v3.53). len = C1 * ph1 * ph1.
  (b) conv1 GLOBAL    : the SAME stage-1 first-spike-latency map [C1,h1,h1], but
                        GLOBAL-max-pooled over BOTH spatial dims -> length-C1 vector
                        (one value per conv1 map; translation invariant).
  (c) conv1+conv2 GLOBAL : cascade the frozen conv1 volley into frozen conv2, take
                        the stage-2 first-spike-latency map [C2,h2,h2], GLOBAL-max-
                        pool over BOTH spatial dims -> length-C2 vector. This is the
                        Kheradpisheh feature: one composite-part activation per map.
The latency value is (t_lat - fired)/t_lat where the neuron fired (earlier spike =
stronger = larger value) and 0 where it never fired; GLOBAL-max over the spatial
dims keeps each map's single strongest (earliest) activation anywhere in the image.

READOUT: for each feature set, StandardScaler (fit on TRAIN only) -> then BOTH a
LinearSVC and an RBF SVC(kernel='rbf'). Each SVM is fit ONLY on the train matrix +
train labels and scored ONLY on the held-out test matrix + test labels -- no
leakage (the scaler and both SVMs see train exclusively at fit time). RBF is what
usually earns Kheradpisheh's last few points, so both are reported.

HONEST COMPARISON: main() prints a table of test accuracy for each feature set x
{LinearSVC, RBF-SVC}. The question: does conv1+conv2 GLOBAL + SVM BEAT conv1
SPATIAL + SVM (~95%) and approach ~98%? The delta is reported as-is; it is NOT
asserted (a tie or a drop is a valid, honest result). What IS asserted only guards
the PIPELINE: conv1-SPATIAL + SVM must beat chance AND land in the ~0.90+ ballpark
(if the frozen 95%-baseline features + an SVM cannot clear that, the extraction or
scaling is broken -- fix the mechanism, do not lower the bar). We deliberately do
NOT assert 2-stage > conv1.

Run (defaults; conv training dominates, SVM fit on cached features is fast; a smoke
run finishes in ~30-60 min on GPU):
    NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/deep_stdp_2stage_svm.py
Scale knobs : NORD_C1(30) NORD_C2(150) NORD_T(15) NORD_L1_EPOCHS(4) NORD_L2_EPOCHS(4)
              NORD_POOL1(2) NORD_K2 NORD_TRAIN(15000) NORD_TEST(3000) NORD_GPU
Deps: numpy torch torchvision snntorch scikit-learn  (research/GPU tier; core untouched)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from snn_data_lake.reasoning.torch_device import resolve_device, enable
from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.reasoning.deep_stdp_2stage import ConvLatencyStdp2, _stage1_volley, _train2


def _env(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


# ------------------------------------------------------------------------------
# GLOBAL-POOLED feature vectors (the one change vs the failed 2-stage).
# All compute on the resolved (GPU) device; callers move to CPU numpy for sklearn.
# ------------------------------------------------------------------------------
def _conv1_both(conv1, image):
    """ONE frozen conv1 forward -> (spatial_flat, global_vec).

    spatial_flat : reproduces conv1.features() EXACTLY (the ~95.1% baseline) --
                   the stage-1 first-spike-latency map, spatial max-pooled by
                   conv1.pool, flattened, rescaled to [0,1]. len = C1*ph1*ph1.
    global_vec   : the SAME latency map GLOBAL-max-pooled over BOTH spatial dims
                   -> length-C1 (one translation-invariant value per conv1 map).
    """
    st = conv1._dog_latency(image)
    fired, _ = conv1._forward(st)                                  # [C1,h1,h1] long
    val = torch.where(fired < conv1.t_lat,
                      (conv1.t_lat - fired).float() / conv1.t_lat,
                      torch.zeros_like(fired, dtype=torch.float32))  # [C1,h1,h1]
    # (b) GLOBAL pooling: max over the two spatial dims -> [C1]
    global_vec = val.amax(dim=(1, 2))
    # (a) SPATIAL baseline: identical recipe to conv1.features()
    pooled = F.max_pool2d(val.view(1, conv1.C, conv1.hout, conv1.hout), conv1.pool)
    spatial = pooled.reshape(-1)
    spatial = spatial / spatial.max().clamp_min(1e-6)
    return spatial, global_vec


def _conv2_global(conv1, conv2, image):
    """Cascade frozen conv1 -> frozen conv2 -> stage-2 first-spike-latency map
    [C2,h2,h2] -> GLOBAL-max-pool over BOTH spatial dims -> length-C2 vector
    (one composite-part activation per conv2 map; translation invariant). This is
    the Kheradpisheh feature. Reuses _stage1_volley + conv2._forward2 unchanged."""
    st = _stage1_volley(conv1, image, conv2.pool1)                # [C1,H,W] volley
    fired, _ = conv2._forward2(st)                                # [C2,h2,h2] long
    val = torch.where(fired < conv2.t_lat,
                      (conv2.t_lat - fired).float() / conv2.t_lat,
                      torch.zeros_like(fired, dtype=torch.float32))  # [C2,h2,h2]
    return val.amax(dim=(1, 2))                                   # [C2]


def _extract(conv1, conv2, X, tag):
    """One pass over X -> three feature MATRICES as CPU float32 numpy arrays:
    (conv1_spatial [N, C1*ph1*ph1], conv1_global [N, C1], conv2_global [N, C2]).
    Each per-image vector is computed on GPU then moved to CPU numpy (.cpu().numpy())
    -- the only place torch tensors cross to numpy for sklearn."""
    s_rows, g1_rows, g2_rows = [], [], []
    for i, img in enumerate(X):
        spatial, g1 = _conv1_both(conv1, img)
        g2 = _conv2_global(conv1, conv2, img)
        s_rows.append(spatial.detach().cpu().numpy())
        g1_rows.append(g1.detach().cpu().numpy())
        g2_rows.append(g2.detach().cpu().numpy())
        if (i + 1) % 2000 == 0:
            print(f"  extract {tag}: {i + 1}/{len(X)} images")
    stack = lambda rows: np.stack(rows).astype(np.float32)
    return stack(s_rows), stack(g1_rows), stack(g2_rows)


def _svm_eval(Xtr_f, ytr, Xte_f, yte):
    """StandardScaler (fit on TRAIN only) -> LinearSVC AND RBF SVC. Both fit ONLY
    on train, scored ONLY on the held-out test set. Returns (acc_linear, acc_rbf).
    No leakage: scaler and both SVMs never see test at fit time."""
    scaler = StandardScaler().fit(Xtr_f)                          # fit on TRAIN only
    Xtr_s = scaler.transform(Xtr_f)
    Xte_s = scaler.transform(Xte_f)                               # test only transformed
    lin = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=0)
    lin.fit(Xtr_s, ytr)
    acc_lin = float(lin.score(Xte_s, yte))
    rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
    rbf.fit(Xtr_s, ytr)
    acc_rbf = float(rbf.score(Xte_s, yte))
    return acc_lin, acc_rbf


def main():
    enable(resolve_device())
    torch.manual_seed(0)                       # deterministic conv weight init
    C1 = _env("NORD_C1", 30)
    C2 = _env("NORD_C2", 150)
    T_LAT = _env("NORD_T", 15)
    TRAIN_N = _env("NORD_TRAIN", 15000)
    TEST_N = _env("NORD_TEST", 3000)
    POOL1 = _env("NORD_POOL1", 2)
    L1_EPOCHS = _env("NORD_L1_EPOCHS", 4)
    L2_EPOCHS = _env("NORD_L2_EPOCHS", 4)

    print("=" * 74)
    print("2-STAGE CONV-STDP + GLOBAL POOL + SVM  (MNIST, Kheradpisheh-2018 ~98% recipe)")
    print("=" * 74)
    print(f"C1={C1}  C2={C2}  latency-steps T={T_LAT}  train={TRAIN_N}  test={TEST_N}")
    print(f"pool1={POOL1}  L1_epochs={L1_EPOCHS}  L2_epochs={L2_EPOCHS}  k2={_env('NORD_K2', 5)}")
    print("feature learning: 100% UNSUPERVISED spike-timing STDP (no labels, no backprop)")
    print("readout: SVM on FROZEN features (Kheradpisheh protocol); GLOBAL pool = the key change\n")
    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)
    ytr = Ytr.detach().cpu().numpy()
    yte = Yte.detach().cpu().numpy()

    # ---------- STAGE 1: unsupervised conv-STDP (freeze) ----------
    print(f"\n[stage1] training conv-latency-STDP (unsupervised, {L1_EPOCHS} epochs)...")
    conv1 = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv1, Xtr, f"stage1 ep{ep + 1}/{L1_EPOCHS}")
        print(f"  [stage1] epoch {ep + 1}/{L1_EPOCHS} kernel diversity (cos-sim) = "
              f"{conv1.kernel_diversity():.3f}")
    div1 = conv1.kernel_diversity()
    print(f"[stage1] frozen; diversity={div1:.3f} (expect < 0.4)  "
          f"hout1={conv1.hout}  spatial feat_dim={conv1.feat_dim()}  global feat_dim={C1}")

    # ---------- STAGE 2: unsupervised conv-STDP over the stage-1 volley (freeze) ----------
    print(f"\n[stage2] training conv-latency-STDP on the stage-1 volley "
          f"(unsupervised, {L2_EPOCHS} epochs)...")
    conv2 = ConvLatencyStdp2(C1, C2, T_LAT, conv1.hout, POOL1)
    print(f"  stage-2 geometry: input {C1}x{conv2.h_in}x{conv2.h_in} "
          f"-> conv(k2={conv2.k2}) -> {C2}x{conv2.hout}x{conv2.hout}  (global feat_dim={C2})")
    assert conv2.hout >= 1, (
        f"stage-2 output spatial size = {conv2.hout} < 1: k2={conv2.k2} too large for the "
        f"pooled stage-1 size h_in={conv2.h_in}. Lower NORD_K2 or NORD_POOL1.")
    for ep in range(L2_EPOCHS):
        _train2(conv2, conv1, Xtr, f"stage2 ep{ep + 1}/{L2_EPOCHS}")
        print(f"  [stage2] epoch {ep + 1}/{L2_EPOCHS} kernel diversity (cos-sim) = "
              f"{conv2.kernel_diversity():.3f}")
    div2 = conv2.kernel_diversity()
    print(f"[stage2] frozen; diversity={div2:.3f}")

    # ---------- extract the three frozen feature MATRICES (train + test) ----------
    print("\n[features] extracting frozen feature matrices (train)...")
    s_tr, g1_tr, g2_tr = _extract(conv1, conv2, Xtr, "train")
    print("[features] extracting frozen feature matrices (test)...")
    s_te, g1_te, g2_te = _extract(conv1, conv2, Xte, "test")
    print(f"[features] shapes: conv1-spatial {s_tr.shape} | conv1-global {g1_tr.shape} | "
          f"conv1+conv2-global {g2_tr.shape}")

    # ---------- SVM readout on each feature set (fit train, score test) ----------
    print("\n[svm] fitting LinearSVC + RBF-SVC per feature set (fit on TRAIN, score on TEST)...")
    lin_a, rbf_a = _svm_eval(s_tr, ytr, s_te, yte)      # (a) conv1 SPATIAL (~95% baseline)
    lin_b, rbf_b = _svm_eval(g1_tr, ytr, g1_te, yte)    # (b) conv1 GLOBAL
    lin_c, rbf_c = _svm_eval(g2_tr, ytr, g2_te, yte)    # (c) conv1+conv2 GLOBAL (Kheradpisheh)

    best_a = max(lin_a, rbf_a)
    best_c = max(lin_c, rbf_c)
    delta = best_c - best_a

    # ---------- honest table ----------
    print("\n" + "=" * 74)
    print("RESULT  (unsupervised STDP features, SVM readout; chance = 10%)")
    print("-" * 74)
    print(f"  {'feature set':<34}{'LinearSVC':>12}{'RBF-SVC':>12}")
    print("-" * 74)
    print(f"  {'(a) conv1 SPATIAL  (~95% base)':<34}{lin_a:>11.1%}{rbf_a:>12.1%}")
    print(f"  {'(b) conv1 GLOBAL':<34}{lin_b:>11.1%}{rbf_b:>12.1%}")
    print(f"  {'(c) conv1+conv2 GLOBAL':<34}{lin_c:>11.1%}{rbf_c:>12.1%}")
    print("-" * 74)
    print(f"  best (a) conv1 SPATIAL          : {best_a:.1%}")
    print(f"  best (c) conv1+conv2 GLOBAL     : {best_c:.1%}")
    print(f"  delta (c minus a)              : {delta:+.1%}   "
          f"({'2-STAGE GLOBAL BEATS conv1' if delta > 0 else ('TIE' if delta == 0 else 'conv1 STILL WINS')})")
    print(f"  approaching ~98% Kheradpisheh?  : best 2-stage = {best_c:.1%}")
    print(f"  stage-1 diversity {div1:.3f} | stage-2 diversity {div2:.3f}  (lower = more specialised)")
    print("=" * 74)

    # self-check: guard the PIPELINE only -- conv1-SPATIAL + SVM must beat chance AND
    # reproduce the ~0.90+ ballpark of the frozen 95%-baseline features. We do NOT
    # assert 2-stage > conv1 (the delta above is reported honestly, whatever its sign).
    assert best_a > 0.5, f"conv1-spatial + SVM failed to beat chance: {best_a:.2f}"
    assert best_a >= 0.85, (
        f"conv1-spatial + SVM = {best_a:.2f} -- below the ~0.90+ ballpark of the frozen "
        "95%-baseline features. The feature extraction or scaling is broken (fix the "
        "pipeline, do NOT lower the bar).")
    print(f"self-check OK: conv1-spatial + SVM = {best_a:.1%} (>10% chance, in the ~0.90+ "
          f"ballpark); 2-stage-global delta {delta:+.1%} reported honestly.")


if __name__ == "__main__":
    main()
