r"""D8 -- INFORMATION PROBE: how much ordered context survives in the 88-dim feature?

The readout is linear on _encode_ctx's 88-dim feature. So the ceiling on what the learner can ever
condition its next-character prediction on is exactly what a LINEAR map can recover from that
feature. This probe trains an optimal linear decoder (converged full-batch delta rule -- the same
single-layer local rule, no backprop) to read back each of the 8 context characters from the
feature, and reports top-1 accuracy per position.

If position -1 (the immediately preceding character) is not cleanly recoverable, the feature cannot
even support a BIGRAM, and no amount of corpus can help.

This is the source of the "each position 72-84% decodable" evidence in the "falsified by ablation" table
in docs/DEPTH_CEILINGS.md.

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88), so the
decodability evidence cited an instrument absent from every clone -- the defect tools/citation_guard.py
now gates. The corpus is resolved by tools/lang_scale.py (this module imports its reader), so the same
--corpus flag applies and an absent corpus SKIPs with a message. BLiMP is not needed here.

Usage:
  python tools/lang_probe.py [--corpus FILE]
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.environ.setdefault("NORD_PERSIST", "0")

from lang_scale import (  # noqa: E402
    BatchEncoder,
    contexts_from_ids,
    encode_stream,
    fold_ids,
    missing_inputs,
    pop_opt,
    read_corpus,
    set_inputs,
)

from snn_data_lake.faculties import language_acquisition as LA  # noqa: E402

V = LA.V
N = 100_000


def fit_linear_softmax(feats, targets, epochs=200, lr=2.0):
    """Converged single-layer delta rule (the shipped local rule, full-batch). Returns W."""
    W = np.zeros((V, feats.shape[1]))
    oh = np.zeros((len(targets), V))
    oh[np.arange(len(targets)), targets] = 1.0
    for _ in range(epochs):
        z = feats @ W.T
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        W += lr * ((oh - p).T @ feats) / len(targets)
    return W


def top1(W, feats, targets):
    return float((np.argmax(feats @ W.T, axis=1) == targets).mean())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    set_inputs(pop_opt(argv, "--corpus", None), None)
    # This probe reads only a prefix: no held-out slice and no BLiMP, so it is runnable on a
    # truncated corpus that lang_scale.py would (correctly) refuse.
    why = missing_inputs(need_heldout=False, need_blimp=False)
    if why:
        print(f"SKIP lang_probe: {why}")
        return 0
    text = read_corpus(N + 200_000)
    ids = fold_ids(text)
    la = LA.LanguageAcquisition(seed=0)
    enc = BatchEncoder(la)

    tr = ids[:N]
    te = ids[N : N + 100_000]
    f_tr = encode_stream(enc, tr)
    f_te = encode_stream(enc, te)
    c_tr = contexts_from_ids(tr, ctx=8)
    c_te = contexts_from_ids(te, ctx=8)

    print("=" * 92)
    print("D8  LINEAR DECODABILITY OF THE CONTEXT FROM THE 88-DIM FEATURE")
    print("=" * 92)
    print(f"  {N} train / {len(te)} test contexts; optimal linear decoder per position")
    print(
        f"  chance = 1/34 = {100.0 / V:.1f}%;  'most frequent char' baseline shown for reference\n"
    )
    print(f"    {'position':>10} {'test top-1 %':>14} {'majority %':>12}")
    for p in range(8):
        y_tr = c_tr[:, p]
        y_te = c_te[:, p]
        W = fit_linear_softmax(f_tr, y_tr)
        maj = float((y_te == np.bincount(y_tr, minlength=V).argmax()).mean())
        lbl = f"-{8 - p}"
        print(f"    {lbl:>10} {100 * top1(W, f_te, y_te):14.1f} {100 * maj:12.1f}", flush=True)

    # and the thing that actually matters: next-character prediction from the same feature
    print()
    y_tr = ids[1 : N + 1]
    y_te = ids[N + 1 : N + 100_001]
    W = fit_linear_softmax(f_tr, y_tr)
    z = f_te @ W.T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    nll = -np.log(p[np.arange(len(y_te)), y_te] + 1e-12).mean()
    print(
        f"  next-char from this feature (optimal linear): test PPL {math.exp(nll):.3f}, "
        f"top-1 {100 * top1(W, f_te, y_te):.1f}%"
    )
    uni = np.bincount(ids[:N], minlength=V) / N
    unll = -np.log(uni[y_te] + 1e-12).mean()
    print(f"  unigram character model (same text)        : test PPL {math.exp(unll):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
