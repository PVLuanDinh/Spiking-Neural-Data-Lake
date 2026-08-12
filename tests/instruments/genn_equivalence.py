"""specs/012 equivalence harness: the GPU port vs a numpy replica of the oracle, toy scale.

Proves the pipeline-lag derivations in the module docstring (input one kernel-step behind,
winner spike two behind, LTP reading prevTrace2, thetaPlusEff folding) by MEASUREMENT:
per-image winner-count vectors, theta, and the full weight matrix must match a float32
numpy replica of FastStdpNetwork.run() on random toy input — and a second GPU run must
match the first bitwise (cross-run determinism).

The replica mirrors reasoning/snn_mnist_stdp_fast.py:100-136 statement-for-statement,
parameterised over dims (the oracle class bakes module-global M/N_IN/T). ASCII only.

Run (GeNN box):
  set CUDA_PATH=D:\\AI-project\\cuda-local  (+ its bin on PATH, vcvars active)
  .venv-lake\\Scripts\\python tests\\instruments\\genn_equivalence.py
(Lived at specs/012-finish-genn-port/ until 2026-08-11; moved because a spec-bundle .py is a
build the p4c ratchet counts, while tests/** is the instrument category it excludes.)
"""

import sys

import numpy as np
import torch

from snn_data_lake.plasticity.snn_mnist_stdp import BETA, THETA_DECAY, THETA_PLUS
from snn_data_lake.plasticity.snn_mnist_stdp_genn import build_model, pack_schedule
from snn_data_lake.reasoning.snn_mnist_stdp_fast import FAST_LR, FAST_XTAR, PRE_DECAY

M_TOY = 8
N_TOY = 16
T_TOY = 10
N_IMG = 4
THRESH = 2.0
W_NORM_TOY = 6.0
WMAX_TOY = 1.0


def replica_run(W, theta, times, idx, learn):
    """FastStdpNetwork.run(), numpy float32, dims-parameterised. Returns counts."""
    mem = np.zeros(M_TOY, dtype=np.float32)
    x_pre = np.zeros(N_TOY, dtype=np.float32)
    counts = np.zeros(M_TOY, dtype=np.float32)
    for t in range(T_TOY):
        x_pre *= np.float32(PRE_DECAY)
        sel = times == t
        if sel.any():
            fi = idx[sel]
            x_pre[fi] += np.float32(1.0)
            mem = mem * np.float32(BETA) + W[:, fi].sum(axis=1)
        else:
            mem = mem * np.float32(BETA)
        eff = mem - theta
        w = int(eff.argmax())
        if eff[w] > THRESH:
            counts[w] += 1
            if learn:
                W[w] += np.float32(FAST_LR) * (x_pre - np.float32(FAST_XTAR))
                np.clip(W[w], 0.0, WMAX_TOY, out=W[w])
                theta[w] += np.float32(THETA_PLUS)
            mem = np.zeros(M_TOY, dtype=np.float32)
        if learn:
            theta *= np.float32(THETA_DECAY)
    if learn:
        sums = np.maximum(W.sum(axis=1), 1e-6)
        W *= (W_NORM_TOY / sums)[:, None]
    return counts


def gpu_run(spike_times, starts, ends, w_init, run_tag):
    model, ssa, exc, syn = build_model(
        N_TOY, M_TOY, spike_times, THRESH, model_name="equiv_toy_" + run_tag
    )
    model.build()
    model.load()
    syn.vars["g"].view[:] = w_init.T.reshape(-1)
    syn.push_var_to_device("g")
    syn.set_dynamic_param_value("lr", FAST_LR)
    exc.set_dynamic_param_value("learnFlag", 1.0)

    all_counts, all_w, all_theta = [], [], []
    for k in range(N_IMG):
        ssa.vars["startSpike"].view[:] = starts[k]
        ssa.vars["endSpike"].view[:] = ends[k]
        ssa.push_var_to_device("startSpike")
        ssa.push_var_to_device("endSpike")
        # T+3 kernels: WTA decisions at boundaries 2..T+1, kernel T+2 is the bare flush
        # step that lets the t=T-1 winner's post-learn LTP land (mirrors _run_pass).
        for s in range(T_TOY + 3):
            if 2 <= s <= T_TOY + 1:
                model.custom_update("wta1")
                model.custom_update("wta2")
                model.custom_update("wta3")
            model.step_time()
        exc.pull_var_from_device("SpikeCount")
        counts = exc.vars["SpikeCount"].view.copy()
        syn.pull_var_from_device("g")
        w = syn.vars["g"].view.reshape(N_TOY, M_TOY).T.copy()
        sums = np.maximum(w.sum(axis=1), 1e-6)
        w *= (W_NORM_TOY / sums)[:, None]
        syn.vars["g"].view[:] = w.T.reshape(-1)
        syn.push_var_to_device("g")
        exc.pull_var_from_device("theta")
        all_counts.append(counts)
        all_w.append(w.copy())
        all_theta.append(exc.vars["theta"].view.copy())
        model.custom_update("img_reset")
    return all_counts, all_w, all_theta


def main():
    rng = torch.Generator().manual_seed(7)
    images = []
    for _ in range(N_IMG):
        n_sp = int(torch.randint(6, 18, (1,), generator=rng))
        tm = torch.randint(0, T_TOY, (n_sp,), generator=rng)
        ix = torch.randint(0, N_TOY, (n_sp,), generator=rng)
        # dedupe (t, pixel) pairs: the oracle's encoder never emits duplicates (consecutive
        # distinct burst times per pixel) and the SSA cannot represent them — a duplicate
        # bumps the CPU trace once but would smear across two GPU kernels (measured).
        pairs = sorted(set(zip(tm.tolist(), ix.tolist(), strict=True)))
        tm = torch.tensor([a for a, _ in pairs], dtype=torch.long)
        ix = torch.tensor([b for _, b in pairs], dtype=torch.long)
        images.append((tm, ix))

    w0 = (torch.rand(M_TOY, N_TOY, generator=rng) * 0.3).numpy().astype(np.float32)
    w0 *= (W_NORM_TOY / np.maximum(w0.sum(axis=1), 1e-6))[:, None]

    # replica
    W = w0.copy()
    theta = np.zeros(M_TOY, dtype=np.float32)
    rep_counts, rep_w, rep_theta = [], [], []
    for tm, ix in images:
        c = replica_run(W, theta, tm.numpy(), ix.numpy(), learn=True)
        rep_counts.append(c)
        rep_w.append(W.copy())
        rep_theta.append(theta.copy())

    spike_times, starts, ends = pack_schedule(images, N_TOY, T_TOY + 3)
    g_counts, g_w, g_theta = gpu_run(spike_times, starts, ends, w0.copy(), "a")
    g_counts2, _, _ = gpu_run(spike_times, starts, ends, w0.copy(), "b")

    ok = True
    for k in range(N_IMG):
        if not np.array_equal(rep_counts[k], g_counts[k]):
            print(f"img {k}: COUNTS DIVERGE  replica={rep_counts[k]}  gpu={g_counts[k]}")
            ok = False
        if not np.allclose(rep_w[k], g_w[k], atol=1e-4):
            print(f"img {k}: WEIGHTS DIVERGE  max|dW|={np.abs(rep_w[k] - g_w[k]).max():.6f}")
            ok = False
        if not np.allclose(rep_theta[k], g_theta[k], atol=1e-5):
            print(f"img {k}: THETA DIVERGES  replica={rep_theta[k]}  gpu={g_theta[k]}")
            ok = False
        if not np.array_equal(g_counts[k], g_counts2[k]):
            print(f"img {k}: GPU NOT DETERMINISTIC ACROSS RUNS")
            ok = False
    if ok:
        print(f"EQUIVALENCE OK: {N_IMG} images, counts exact, dW<=1e-4, theta<=1e-5, 2x bitwise")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
