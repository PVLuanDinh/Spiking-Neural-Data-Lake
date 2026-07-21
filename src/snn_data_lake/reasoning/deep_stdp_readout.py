r"""
LINEAR READOUT on deep unsupervised spiking features -- the decisive test of the
deepening arc.  (torch/GPU tier, ruff-excluded; core untouched.)

WHY THIS FILE EXISTS
--------------------
v3.49-51 showed the deep unsupervised conv-STDP hierarchy learns genuinely DIVERSE
spiking features (no collapse), yet accuracy TIES then HURTS under the spiking
neuron-label VOTE readout (v3.51: pixels 58.0% -> conv1 53.9% -> conv1+conv2 48.9%).
The hypothesis: the vote (nearest-prototype) is too weak to use the abstract pooled
features, so depth looks useless when it is really the READOUT that is the cap.
Kheradpisheh 2018 reaches ~98% by putting an SVM on the spiking features, not a vote.

This file runs that test: a LINEAR readout (a torch nn.Linear + cross-entropy, i.e.
softmax/multinomial-logistic regression -- the SAME hypothesis class as a linear SVM)
on three frozen feature sources -- raw pixels, stage-1 conv features, stage-1+stage-2
conv features -- and reports test accuracy for each. Feature learning stays FULLY
UNSUPERVISED (the conv STDP saw no labels); only the linear readout is supervised, on
frozen features -- the standard Kheradpisheh protocol. If linear-readout accuracy RISES
with depth (pixels < conv1 < conv2) while the vote FELL, then depth added linearly-
separable class information the vote could not extract, and the readout -- not the
depth -- was the bottleneck. If it does NOT rise, depth genuinely does not help even
the right readout. Either way is an honest, reported result.

Run (from the repo root; a few minutes on GPU):
    NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/deep_stdp_readout.py
Scale knobs: NORD_C1 NORD_C2 NORD_TRAIN NORD_TEST NORD_T NORD_L1_EPOCHS NORD_L2_EPOCHS
             NORD_POOL1 NORD_READEP (linear-readout epochs)
Deps: numpy torch torchvision snntorch   (research/GPU tier; core untouched)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.reasoning.torch_device import resolve_device, enable
from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp, _train
from snn_data_lake.reasoning.deep_stdp_2stage import ConvLatencyStdp2, _train2


def _env(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


def _stack_features(feat_fn, X):
    """Apply a per-image feature function and stack into a [N, D] matrix on the
    resolved device. feat_fn(img) -> a 1-D feature tensor."""
    rows = [feat_fn(img).reshape(-1) for img in X]
    return torch.stack(rows, dim=0)


def linear_readout(Xtr, ytr, Xte, yte, epochs, seed=0):
    """A LINEAR classifier (nn.Linear + cross-entropy = softmax regression, the linear-
    SVM hypothesis class) trained SUPERVISED on frozen features. Returns test accuracy.
    Full-batch Adam -- the feature matrices are small and this keeps it deterministic."""
    torch.manual_seed(seed)
    dev = resolve_device()
    Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    Xte, yte = Xte.to(dev), yte.to(dev)
    clf = nn.Linear(Xtr.shape[1], 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(Xte).argmax(dim=1) == yte).float().mean().item()
    return acc


def main():
    enable(resolve_device())
    C1 = _env("NORD_C1", 30)
    C2 = _env("NORD_C2", 60)
    TRAIN_N = _env("NORD_TRAIN", 8000)
    TEST_N = _env("NORD_TEST", 2000)
    T_LAT = _env("NORD_T", 15)
    POOL1 = _env("NORD_POOL1", 2)
    L1_EPOCHS = _env("NORD_L1_EPOCHS", 3)
    L2_EPOCHS = _env("NORD_L2_EPOCHS", 3)
    READEP = _env("NORD_READEP", 300)

    print("=" * 70)
    print("LINEAR READOUT on deep UNSUPERVISED spiking features  (MNIST)")
    print("=" * 70)
    print(f"C1={C1}  C2={C2}  T={T_LAT}  train={TRAIN_N}  test={TEST_N}  "
          f"L1ep={L1_EPOCHS}  L2ep={L2_EPOCHS}  readout_epochs={READEP}")
    print("features are UNSUPERVISED (conv STDP, no labels); ONLY the linear readout "
          "sees labels (Kheradpisheh protocol)\n")
    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    # ---- train the unsupervised feature extractors (no labels) ----
    torch.manual_seed(0)
    print(f"\ntraining conv-latency-STDP stage 1 (unsupervised, {L1_EPOCHS} epochs)...")
    conv1 = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv1, Xtr, f"stage1 ep{ep + 1}/{L1_EPOCHS}")
    print(f"  stage-1 kernel diversity (cos-sim) = {conv1.kernel_diversity():.3f}")

    print(f"\ntraining conv-latency-STDP stage 2 on the stage-1 volley (unsupervised, {L2_EPOCHS} epochs)...")
    conv2 = ConvLatencyStdp2(C1, C2, T_LAT, conv1.hout, POOL1)
    for ep in range(L2_EPOCHS):
        _train2(conv2, conv1, Xtr, f"stage2 ep{ep + 1}/{L2_EPOCHS}")
    print(f"  stage-2 kernel diversity (cos-sim) = {conv2.kernel_diversity():.3f}")

    # ---- build feature matrices for the three sources ----
    print("\nextracting feature matrices (pixels / conv1 / conv1+conv2)...")
    ytr, yte = Ytr.long(), Yte.long()
    sources = {}
    sources["pixels (784)"] = (
        torch.stack([img.reshape(-1) for img in Xtr]).to(resolve_device()),
        torch.stack([img.reshape(-1) for img in Xte]).to(resolve_device()),
    )
    sources[f"conv1 ({conv1.feat_dim()})"] = (
        _stack_features(conv1.features, Xtr),
        _stack_features(conv1.features, Xte),
    )
    sources[f"conv1+conv2 ({conv2.feat_dim()})"] = (
        _stack_features(lambda im: conv2.features2(im, conv1), Xtr),
        _stack_features(lambda im: conv2.features2(im, conv1), Xte),
    )

    # ---- linear readout on each ----
    print(f"training LINEAR readouts ({READEP} epochs each)...\n")
    accs = {}
    for name, (Xtr_f, Xte_f) in sources.items():
        accs[name] = linear_readout(Xtr_f, ytr, Xte_f, yte, READEP)

    names = list(sources.keys())
    apx, a1, a2 = accs[names[0]], accs[names[1]], accs[names[2]]
    print("=" * 70)
    print("RESULT  (LINEAR readout on frozen features; chance = 10%)")
    print("-" * 70)
    for name in names:
        print(f"  {name:28s} : {accs[name]:.1%}")
    print("-" * 70)
    print(f"  delta conv1      vs pixels : {a1 - apx:+.1%}")
    print(f"  delta conv1+conv2 vs pixels: {a2 - apx:+.1%}")
    print(f"  delta conv1+conv2 vs conv1 : {a2 - a1:+.1%}")
    rises = a2 > apx and a1 >= apx - 0.005
    print(f"  verdict: depth {'HELPS a linear readout (readout WAS the cap)' if a2 > apx else 'does NOT help even a linear readout'}")
    print("  (compare v3.51 VOTE readout at this scale: pixels 58.0 -> conv1 53.9 -> conv1+conv2 48.9, i.e. depth HURT)")
    print("=" * 70)

    # self-check: every linear readout must beat chance (guards the readout + features);
    # the pixel baseline must be sane. We do NOT assert depth helps -- the signed deltas
    # are the honest, reported result (depth helping OR not both answer the question).
    assert apx >= 0.5, f"pixel linear readout failed to beat chance: {apx:.2f}"
    for name in names:
        assert accs[name] >= 0.10, f"{name} linear readout at/below chance: {accs[name]:.2f}"
    verdict = "readout WAS the cap" if a2 > apx else "depth does not help even a linear readout"
    print(f"self-check OK: pixel linear {apx:.1%} sane; all readouts > chance; "
          f"depth-vs-pixels {a2 - apx:+.1%} -> {verdict} (honest).")


if __name__ == "__main__":
    main()
