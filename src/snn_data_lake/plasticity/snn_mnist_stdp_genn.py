"""
GeNN GPU STDP-MNIST with CUSTOM plasticity — the FINISHED port (specs/012, mandated build).

# distinct-from: reasoning/snn_mnist_stdp_genn.py -- that path IS this module before its
#   2026-08-07 git mv into plasticity/ (specs/012 US4); the graph lagged the move. One file,
#   one mechanism, moved then finished.
# distinct-from: cortex/lif_neuron.py -- that is the composable CPU LIF (Python object, one
#   neuron, step()). This module code-GENERATES C++ for GeNN's CUDA kernels mirroring the
#   ORACLE's population semantics (multiplicative beta decay + global hard-WTA + full-pop
#   reset), which no cortex/ neuron object implements or should: the CPU mechanism for this
#   job is reasoning/snn_mnist_stdp_fast.FastStdpNetwork, and it stays the oracle.
# distinct-from: cortex/alif_neuron.py -- same distinction; its adaptive threshold is a
#   per-neuron CPU object; the theta here is the oracle's population homeostasis in C++.

The CPU-verified v0.17 rule (burst encoding + x_tar LTP, 79.0% measured on this box at
M=300/6k — see specs/012-finish-genn-port/evidence.md) running on the RTX 5070 via GeNN 5
codegen. The exact rule math lives in `snn_mnist_stdp_fast.py` (the oracle); this module is
its GPU mirror, EXACT by construction:

  1. SpikeSourceArray = deterministic anchor. The whole dataset's burst-latency spike
     schedule (shared encoder `encode_latency`, REUSED not reimplemented) is uploaded once
     with absolute per-image time offsets; identical inputs -> identical trains.
  2. Hard GLOBAL winner-take-all, mirrored exactly. The oracle's "lateral inhibition" is a
     per-step argmax over (V - theta) with a FULL-population membrane reset on fire — not
     expressible with built-in neuron/synapse models. Implemented as three custom updates
     per step (REDUCE_MAX of the excess; REDUCE_MAX of an index-break score so exact ties
     resolve to the LOWEST index like torch argmax; flag + global V reset), with the winner
     emitting a TRUE spike so the plasticity codegen below fires event-driven.
  3. The burst+x_tar rule as GeNN C++ (the tree's only GPU plasticity codegen). GeNN's
     pipeline runs input delivery one kernel-step behind the CPU loop and the winner spike
     two behind, so the LTP reads `prevTrace2` — a two-step pre-trace history kept in
     pre_dynamics — which IS the oracle's x_pre(t) at the decision step, exactly (verified
     empirically by tests/instruments/genn_equivalence.py (at specs/012-finish-genn-port/
     until 2026-08-11): winner sequences and
     weights match a numpy replica of the oracle at float32 tolerance).
  4. theta homeostasis: bump folded through one decay (thetaPlusEff = THETA_PLUS *
     THETA_DECAY) so the GPU recurrence equals the CPU's bump-then-decay ordering exactly
     (identity at the default THETA_DECAY = 1.0).

LAZY IMPORTS, DELIBERATE: pygenn and the oracle are imported inside functions, never at
module scope. Two measured reasons. (a) importing pygenn RE-ENABLES THE GIL process-wide
(its extension predates free-threading declarations) — at module scope that tax would land
on every `import snn_data_lake` on a GeNN box; lazily it lands only on processes that
actually build models (and run_selfchecks isolates per-module subprocesses). (b) the
package's back-compat alias layer imports region modules during package init, so a
module-scope reach into `reasoning/` re-enters this module while partially initialised —
the circular import was OBSERVED, not theorised.

ENVIRONMENT (measured 2026-08-07, this box — the old "cannot run on the dev box" claim is
RETIRED): pygenn 5.4.0 builds from source against the FREE-THREADED ABI (cp314t) and runs
CUDA models on the RTX 5070 (sm_120). The box's CUDA 13.3 install is INCOMPLETE (no crt/,
no nvvm/cicc, no cudart DLL); a complete local toolkit is assembled at
D:\\AI-project\\cuda-local from official NVIDIA PyPI wheels version-matched to nvcc V13.3.73.
A GPU run therefore needs: CUDA_PATH=D:\\AI-project\\cuda-local, that toolkit's bin on PATH,
and MSVC active (vcvars64) — set PATH before calling vcvars in cmd.

GATE SAFETY: the self-check NEVER compiles or trains. `GENN_RUN=1` is the explicit opt-in
for the GPU pipeline (minutes of nvcc + training). `GENN_LR0=1` runs the untrained control.
Config via the oracle's own env names (NORD_M/NORD_TRAIN/NORD_TEST/...); GENN_SEED seeds
the weight init (default 0).
"""

import importlib.util
import os
import time
from types import SimpleNamespace

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "run_mnist": (
        "LOCAL",
        "the full unsupervised STDP pipeline -- encode, build, train, assign, evaluate -- mirroring "
        "the FAST arm. The learning is spike-timing-dependent plasticity: pairwise, synapse-local, "
        "with no labels entering the weight update and no gradient anywhere",
    ),
    "_run_pass": (
        "NOT_A_RULE",
        "one forward pass over an image range returning the per-image spike-count matrix. Inference, "
        "not learning",
    ),
}

# ---- The rule's C++ (the codegen), importable on EVERY box (no pygenn needed) -------------
# post_spike is event-driven on the winner's true spike. prevTrace2 is x_pre(t) at the
# CPU decision step t (two-step history; derivation in the module docstring, verified by
# genn_equivalence.py). lr is a DYNAMIC param (GeNN 5.4 retired scalar EGPs — the canary in
# main() caught this) so eval/control runs reuse the SAME compiled model with lr = 0
# instead of forcing a recompile.
GENN_PRE_DYNAMICS = "prevTrace2 = prevTrace1; prevTrace1 = preTrace; preTrace *= preDecay;"
GENN_PRE_SPIKE = "preTrace += 1.0;"
# per-SYNAPSE codes (GeNN 5: *_syn_code strings are the ones with g access; the plain
# pre/post_spike_code strings above/below are per-NEURON var updates). Delivery is explicit
# in GeNN — a WU model with no addToPost delivers NOTHING (measured: silent zero-V network).
GENN_PRE_SPIKE_SYN = "addToPost(g);"
GENN_POST_SPIKE = "g += lr * (prevTrace2 - xtar); g = fmin(fmax(g, 0.0), wmax);"

# WTA custom-update C++ (per-step, in invocation order). `id` is GeNN's element index;
# score = (n - id) so REDUCE_MAX picks the LOWEST index among exact ties (torch argmax).
GENN_CU_REDUCE_EFF = "maxEff = V - theta;"
GENN_CU_REDUCE_SCORE = (
    "const scalar eff = V - theta;"
    " maxScore = (eff == maxEff && eff > thresh) ? (n - (scalar)id) : 0.0;"
)
GENN_CU_PICK_FLAG = (
    "const scalar eff = V - theta;"
    " if (maxScore > 0.0) {"
    " if ((n - (scalar)id) == maxScore && eff == maxEff && eff > thresh) { winnerFlag = 1.0; }"
    " V = 0.0; }"
)

GENN_NEURON_SIM = "V = V * beta + Isyn; if (learnFlag > 0.5) { theta *= thetaDecay; }"
GENN_NEURON_THRESHOLD = "winnerFlag > 0.5"
GENN_NEURON_RESET = (
    "winnerFlag = 0.0; SpikeCount += 1.0; if (learnFlag > 0.5) { theta += thetaPlusEff; }"
)

_ORACLE_CACHE = None


def oracle():
    """Lazy namespace over the oracle's constants + helpers (reasoning tier, needs torch).
    Lazy for the two reasons in the module docstring."""
    global _ORACLE_CACHE
    if _ORACLE_CACHE is None:
        import numpy as np
        import torch

        from snn_data_lake.reasoning import snn_mnist_stdp as base
        from snn_data_lake.reasoning import snn_mnist_stdp_fast as fast

        _ORACLE_CACHE = SimpleNamespace(
            np=np,
            torch=torch,
            M=base.M,
            N_IN=base.N_IN,
            T=base.T,
            TRAIN_N=base.TRAIN_N,
            TEST_N=base.TEST_N,
            BETA=base.BETA,
            THETA_PLUS=base.THETA_PLUS,
            THETA_DECAY=base.THETA_DECAY,
            W_NORM=base.W_NORM,
            WMAX=base.WMAX,
            load_mnist=base.load_mnist,
            FAST_LR=fast.FAST_LR,
            FAST_THRESH=fast.FAST_THRESH,
            FAST_XTAR=fast.FAST_XTAR,
            PRE_DECAY=fast.PRE_DECAY,
            encode_latency=fast.encode_latency,
        )
    return _ORACLE_CACHE


def rule_params():
    """The rule's GeNN params, from the oracle's constants (the one source of truth)."""
    o = oracle()
    return {"xtar": o.FAST_XTAR, "wmax": o.WMAX, "preDecay": o.PRE_DECAY}


def burst_xtar_rule():
    """The v0.17 rule as a GeNN custom weight-update model. 'Burst' lives in the input
    encoding, not the rule. This is exactly `self.W[w] += FAST_LR * (x_pre - FAST_XTAR)`
    from snn_mnist_stdp_fast.py, in C++, event-driven on the winner's spike."""
    from pygenn import create_weight_update_model

    return create_weight_update_model(
        "burst_xtar_ltp",
        params=["xtar", "wmax", "preDecay", "lr"],
        vars=[("g", "scalar")],
        pre_vars=[("preTrace", "scalar"), ("prevTrace1", "scalar"), ("prevTrace2", "scalar")],
        pre_dynamics_code=GENN_PRE_DYNAMICS,
        pre_spike_code=GENN_PRE_SPIKE,
        pre_spike_syn_code=GENN_PRE_SPIKE_SYN,
        post_spike_syn_code=GENN_POST_SPIKE,
    )


def _wta_neuron():
    from pygenn import VarAccess, create_neuron_model

    return create_neuron_model(
        "wta_lif",
        params=["beta", "thetaDecay", "thetaPlusEff", "learnFlag"],
        vars=[
            ("V", "scalar"),
            ("theta", "scalar"),
            ("winnerFlag", "scalar"),
            ("SpikeCount", "scalar"),
            ("maxEff", "scalar", VarAccess.READ_ONLY_SHARED_NEURON),
            ("maxScore", "scalar", VarAccess.READ_ONLY_SHARED_NEURON),
        ],
        sim_code=GENN_NEURON_SIM,
        threshold_condition_code=GENN_NEURON_THRESHOLD,
        reset_code=GENN_NEURON_RESET,
    )


def build_model(n_in, m, spike_times, thresh, model_name="stdp_mnist"):
    """The complete WTA learner as a GeNN model. `spike_times` is the FULL absolute-time
    schedule (float array) for the SpikeSourceArray; per-image startSpike/endSpike windows
    are pushed by the host loop. Returns (model, ssa, exc, syn)."""
    import numpy as np
    from pygenn import (
        GeNNModel,
        VarAccessMode,
        create_custom_update_model,
        create_var_ref,
        create_wu_pre_var_ref,
        init_postsynaptic,
        init_weight_update,
    )

    o = oracle()
    model = GeNNModel("float", model_name)
    model.dt = 1.0
    # LIFETIME, MEASURED THE HARD WAY (deterministic 0xC0000005, bisected 2026-08-07): GeNN's
    # C++ keeps RAW POINTERS to custom model definitions. A snippet passed as a temporary
    # (e.g. `init_weight_update(burst_xtar_rule(), ...)`) is GC'd by Python while GeNN still
    # dereferences it inside add_synapse_population — access violation. EVERY custom model
    # object below is therefore bound to a local AND pinned on the returned model for the
    # model's whole life.
    model._snn_keepalive = keep = []

    ssa = model.add_neuron_population(
        "In",
        n_in,
        "SpikeSourceArray",
        {},
        {"startSpike": np.zeros(n_in, dtype=np.int32), "endSpike": np.zeros(n_in, dtype=np.int32)},
    )
    ssa.extra_global_params["spikeTimes"].set_init_values(spike_times)

    nm = _wta_neuron()
    keep.append(nm)
    exc = model.add_neuron_population(
        "Exc",
        m,
        nm,
        {
            "beta": o.BETA,
            "thetaDecay": o.THETA_DECAY,
            "thetaPlusEff": o.THETA_PLUS * o.THETA_DECAY,
            "learnFlag": 1.0,
        },
        {
            "V": 0.0,
            "theta": 0.0,
            "winnerFlag": 0.0,
            "SpikeCount": 0.0,
            "maxEff": 0.0,
            "maxScore": 0.0,
        },
    )
    exc.set_param_dynamic("learnFlag")

    wu = burst_xtar_rule()
    keep.append(wu)
    syn = model.add_synapse_population(
        "InExc",
        "DENSE",
        ssa,
        exc,
        init_weight_update(
            wu,
            {**rule_params(), "lr": o.FAST_LR},
            {"g": 0.0},
            {"preTrace": 0.0, "prevTrace1": 0.0, "prevTrace2": 0.0},
        ),
        init_postsynaptic("DeltaCurr"),
    )
    syn.set_wu_param_dynamic("lr")

    reduce_eff = create_custom_update_model(
        "reduce_eff",
        var_refs=[
            ("V", "scalar", VarAccessMode.READ_ONLY),
            ("theta", "scalar", VarAccessMode.READ_ONLY),
            ("maxEff", "scalar", VarAccessMode.REDUCE_MAX),
        ],
        update_code=GENN_CU_REDUCE_EFF,
    )
    reduce_score = create_custom_update_model(
        "reduce_score",
        params=["thresh", "n"],
        var_refs=[
            ("V", "scalar", VarAccessMode.READ_ONLY),
            ("theta", "scalar", VarAccessMode.READ_ONLY),
            ("maxEff", "scalar", VarAccessMode.READ_ONLY),
            ("maxScore", "scalar", VarAccessMode.REDUCE_MAX),
        ],
        update_code=GENN_CU_REDUCE_SCORE,
    )
    pick_flag = create_custom_update_model(
        "pick_flag",
        params=["thresh", "n"],
        var_refs=[
            ("V", "scalar", VarAccessMode.READ_WRITE),
            ("theta", "scalar", VarAccessMode.READ_ONLY),
            ("maxEff", "scalar", VarAccessMode.READ_ONLY),
            ("maxScore", "scalar", VarAccessMode.READ_ONLY),
            ("winnerFlag", "scalar", VarAccessMode.READ_WRITE),
        ],
        update_code=GENN_CU_PICK_FLAG,
    )
    reset_neuron = create_custom_update_model(
        "reset_neuron",
        var_refs=[
            ("V", "scalar", VarAccessMode.READ_WRITE),
            ("winnerFlag", "scalar", VarAccessMode.READ_WRITE),
            ("SpikeCount", "scalar", VarAccessMode.READ_WRITE),
        ],
        update_code="V = 0.0; winnerFlag = 0.0; SpikeCount = 0.0;",
    )
    reset_pre = create_custom_update_model(
        "reset_pre",
        var_refs=[
            ("preTrace", "scalar", VarAccessMode.READ_WRITE),
            ("prevTrace1", "scalar", VarAccessMode.READ_WRITE),
            ("prevTrace2", "scalar", VarAccessMode.READ_WRITE),
        ],
        update_code="preTrace = 0.0; prevTrace1 = 0.0; prevTrace2 = 0.0;",
    )

    keep.extend([reduce_eff, reduce_score, pick_flag, reset_neuron, reset_pre])
    ex = {"V": create_var_ref(exc, "V"), "theta": create_var_ref(exc, "theta")}
    model.add_custom_update(
        "cu_eff", "wta1", reduce_eff, {}, {}, {**ex, "maxEff": create_var_ref(exc, "maxEff")}
    )
    model.add_custom_update(
        "cu_score",
        "wta2",
        reduce_score,
        {"thresh": thresh, "n": float(m)},
        {},
        {
            **ex,
            "maxEff": create_var_ref(exc, "maxEff"),
            "maxScore": create_var_ref(exc, "maxScore"),
        },
    )
    model.add_custom_update(
        "cu_flag",
        "wta3",
        pick_flag,
        {"thresh": thresh, "n": float(m)},
        {},
        {
            **ex,
            "maxEff": create_var_ref(exc, "maxEff"),
            "maxScore": create_var_ref(exc, "maxScore"),
            "winnerFlag": create_var_ref(exc, "winnerFlag"),
        },
    )
    model.add_custom_update(
        "cu_img_n",
        "img_reset",
        reset_neuron,
        {},
        {},
        {
            "V": create_var_ref(exc, "V"),
            "winnerFlag": create_var_ref(exc, "winnerFlag"),
            "SpikeCount": create_var_ref(exc, "SpikeCount"),
        },
    )
    model.add_custom_update(
        "cu_img_p",
        "img_reset",
        reset_pre,
        {},
        {},
        {
            "preTrace": create_wu_pre_var_ref(syn, "preTrace"),
            "prevTrace1": create_wu_pre_var_ref(syn, "prevTrace1"),
            "prevTrace2": create_wu_pre_var_ref(syn, "prevTrace2"),
        },
    )
    return model, ssa, exc, syn


def pack_schedule(encodings, n_in, steps_per_image):
    """Concatenate per-image (times, idx) encodings into one absolute-time SSA schedule.
    Returns (spike_times, starts, ends): starts/ends are [n_images, n_in] int32 windows
    into spike_times; image k's spikes live at absolute times k*steps_per_image + t."""
    import numpy as np

    n_img = len(encodings)
    per_pixel = [[] for _ in range(n_in)]
    starts = np.zeros((n_img, n_in), dtype=np.int32)
    ends = np.zeros((n_img, n_in), dtype=np.int32)
    cursor = 0
    all_times = []
    for k, (tm, ix) in enumerate(encodings):
        base = float(k * steps_per_image)
        for p in range(n_in):
            per_pixel[p].clear()
        for t_val, p in zip(tm.tolist(), ix.tolist()):
            per_pixel[p].append(base + float(t_val))
        for p in range(n_in):
            starts[k, p] = cursor
            times_p = sorted(per_pixel[p])
            # The SSA emits at most ONE spike per neuron per kernel step, so a duplicate
            # (pixel, time) pair cannot be represented — it would SMEAR to the next step,
            # silently diverging from the oracle (which bumps the trace once and delivers
            # the weight twice for a duplicate; measured in the equivalence harness).
            # encode_latency emits consecutive DISTINCT times per pixel, so real data never
            # trips this; refuse loudly rather than mis-run if some future encoder does.
            if any(a == b for a, b in zip(times_p, times_p[1:])):
                raise ValueError(
                    f"image {k}: pixel {p} has duplicate spike times {times_p} — "
                    "not representable by SpikeSourceArray (one spike/neuron/step)"
                )
            all_times.extend(times_p)
            cursor += len(times_p)
            ends[k, p] = cursor
    return np.asarray(all_times, dtype=np.float32), starts, ends


def _run_pass(model, ssa, exc, syn, starts, ends, k_range, m, n_in, steps, learn, w_norm):
    """One pass over images [k_range]: returns per-image SpikeCount matrix. Mirrors
    FastStdpNetwork.run + the per-image row normalisation when learning."""
    import numpy as np

    counts = np.zeros((len(k_range), m), dtype=np.float32)
    g_view = syn.vars["g"].view
    for row, k in enumerate(k_range):
        ssa.vars["startSpike"].view[:] = starts[k]
        ssa.vars["endSpike"].view[:] = ends[k]
        ssa.push_var_to_device("startSpike")
        ssa.push_var_to_device("endSpike")
        # Kernels 0..steps-1 where steps = T+3: decisions for CPU steps 0..T-1 run at the
        # boundaries before kernels 2..T+1; the winner for t spikes at kernel t+2 and its
        # post-learn LTP executes at the START of kernel t+3, so the t=T-1 winner needs
        # kernel T+2 — a bare flush step with NO WTA. Measured: without it the last-step
        # winner's LTP silently never lands while its SpikeCount and theta bump DO — a
        # half-applied update the equivalence harness caught as a reshaped weight row.
        for s in range(steps):
            if 2 <= s <= steps - 2:
                model.custom_update("wta1")
                model.custom_update("wta2")
                model.custom_update("wta3")
            model.step_time()
        exc.pull_var_from_device("SpikeCount")
        counts[row] = exc.vars["SpikeCount"].view[:]
        if learn:
            syn.pull_var_from_device("g")
            w = g_view.reshape(n_in, m)  # GeNN dense layout: [pre][post]
            sums = np.maximum(w.sum(axis=0), 1e-6)
            w *= w_norm / sums
            syn.push_var_to_device("g")
        model.custom_update("img_reset")
    return counts


def _phase_knobs(exc, syn, lr, learn):
    # distinct-from: cortex/interaction_learning.py -- not a learner: flips the compiled
    # model's two DYNAMIC params (lr, learnFlag) between the train/assign/eval phases.
    syn.set_dynamic_param_value("lr", lr)
    exc.set_dynamic_param_value("learnFlag", 1.0 if learn else 0.0)


def run_mnist():
    """The full pipeline: encode -> build -> train -> assign -> evaluate, mirroring
    snn_mnist_stdp_fast.main()'s FAST arm. Prints acc= / train= lines (the contract)."""
    o = oracle()
    np = o.np
    seed = int(os.environ.get("GENN_SEED", "0"))
    lr0 = os.environ.get("GENN_LR0") == "1"
    o.torch.manual_seed(seed)
    print("=" * 64)
    print("GeNN STDP-MNIST (custom burst+x_tar plasticity) - specs/012")
    print("=" * 64)
    print(
        f"M={o.M}  T={o.T}  train={o.TRAIN_N}  test={o.TEST_N}  thresh={o.FAST_THRESH}  lr0={lr0}"
    )

    Xtr, Ytr, Xte, Yte = o.load_mnist(o.TRAIN_N, o.TEST_N)
    n_asg = min(o.TRAIN_N, 6000)

    t0 = time.perf_counter()
    enc_tr = [o.encode_latency(img) for img in Xtr]
    enc_te = [o.encode_latency(img) for img in Xte]
    # determinism check: the deterministic-anchor property (identical input -> identical train)
    d_tm, d_ix = o.encode_latency(Xtr[0])
    det = bool(o.torch.equal(d_tm, enc_tr[0][0]) and o.torch.equal(d_ix, enc_tr[0][1]))
    print(f"encode={time.perf_counter() - t0:.1f}s  determinism_encode={'OK' if det else 'FAIL'}")

    steps = o.T + 3  # T input steps + 2 pipeline kernels + 1 post-learn flush (see _run_pass)
    # one schedule: train pass | assign pass (same train images again) | test pass
    seq = enc_tr + enc_tr[:n_asg] + enc_te
    spike_times, starts, ends = pack_schedule(seq, o.N_IN, steps)
    print(f"schedule: {len(seq)} images, {len(spike_times)} spikes")

    model, ssa, exc, syn = build_model(o.N_IN, o.M, spike_times, o.FAST_THRESH)
    t0 = time.perf_counter()
    model.build()
    model.load()
    print(f"build+load (nvcc)={time.perf_counter() - t0:.1f}s")

    # init weights EXACTLY as the oracle: torch.rand(M, N_IN)*0.3, row-normalised.
    w0 = o.torch.rand(o.M, o.N_IN) * 0.3
    w0 *= o.W_NORM / w0.sum(dim=1, keepdim=True).clamp_min(1e-6)
    syn.vars["g"].view[:] = w0.numpy().T.reshape(-1)  # [post][pre] -> GeNN [pre][post]
    syn.push_var_to_device("g")

    n_tr, n_te = len(enc_tr), len(enc_te)
    _phase_knobs(exc, syn, 0.0 if lr0 else o.FAST_LR, learn=not lr0)
    t0 = time.perf_counter()
    _run_pass(
        model,
        ssa,
        exc,
        syn,
        starts,
        ends,
        range(n_tr),
        o.M,
        o.N_IN,
        steps,
        learn=not lr0,
        w_norm=o.W_NORM,
    )
    train_s = time.perf_counter() - t0
    print(f"train={train_s:.1f}s")

    _phase_knobs(exc, syn, 0.0, learn=False)
    t0 = time.perf_counter()
    resp = _run_pass(
        model,
        ssa,
        exc,
        syn,
        starts,
        ends,
        range(n_tr, n_tr + n_asg),
        o.M,
        o.N_IN,
        steps,
        learn=False,
        w_norm=o.W_NORM,
    )
    per_class = np.zeros((o.M, 10), dtype=np.float32)
    for row in range(n_asg):
        per_class[:, int(Ytr[row])] += resp[row]
    assignment = per_class.argmax(axis=1)

    counts_te = _run_pass(
        model,
        ssa,
        exc,
        syn,
        starts,
        ends,
        range(n_tr + n_asg, n_tr + n_asg + n_te),
        o.M,
        o.N_IN,
        steps,
        learn=False,
        w_norm=o.W_NORM,
    )
    correct = 0
    for row in range(n_te):
        scores = np.zeros(10, dtype=np.float32)
        for cl in range(10):
            mask = assignment == cl
            if mask.any():
                scores[cl] = counts_te[row][mask].sum()
        correct += int(scores.argmax()) == int(Yte[row])
    acc = correct / n_te
    print(f"assign+eval={time.perf_counter() - t0:.1f}s")
    print(f"acc={acc:.4f}")
    print(f"train={train_s:.1f}s")
    return acc


def _selfcheck_codegen():
    # distinct-from: reasoning/web_ingest_gate.py -- that gates WEB FETCHES (host allowlist);
    #   this asserts C++ codegen string integrity. Sole overlap is the word "check".
    """The falsifiable no-toolchain check: the codegen text carries the oracle's rule. All 7
    asserts have a REACHABLE red — mutate any constant and the matching assert fires
    (demonstrated verbatim in evidence.md #red-demo).

    A previous 8th assert compared rule_params() to the oracle constants it is BUILT FROM —
    a derivation that round-trips, red unreachable, so per P8 it is DELETED, not weakened.
    The hazard it pretended to cover (build_model feeding GeNN different values) is closed
    structurally: rule_params() is the ONE source init_weight_update consumes."""
    assert "lr * (prevTrace2 - xtar)" in GENN_POST_SPIKE, "LTP form lost from codegen"
    assert "fmin(fmax(g, 0.0), wmax)" in GENN_POST_SPIKE, "weight clamp lost from codegen"
    assert "preTrace *= preDecay" in GENN_PRE_DYNAMICS, "trace decay lost from codegen"
    assert "prevTrace2 = prevTrace1; prevTrace1 = preTrace" in GENN_PRE_DYNAMICS, (
        "two-step trace history lost - LTP would read the wrong step's trace"
    )
    assert "preTrace += 1.0" in GENN_PRE_SPIKE, "trace bump lost from codegen"
    assert "addToPost(g)" in GENN_PRE_SPIKE_SYN, "synaptic delivery lost — a silent network"
    assert "V = V * beta + Isyn" in GENN_NEURON_SIM, "oracle membrane update lost"
    print("self-check OK: codegen integrity (7 asserts, every red reachable)")


def main():
    have_genn = importlib.util.find_spec("pygenn") is not None
    if not have_genn:
        print("=" * 60)
        print("GeNN STDP-MNIST (custom plasticity) - toolchain not present here")
        print("=" * 60)
        print(
            "  pygenn not installed. Needs the [genn] extra (source install), C++ compiler, CUDA."
        )
        print("  Rule + encoding are CPU-verified in: python snn_mnist_stdp_fast.py")
        _selfcheck_codegen()
        return
    if os.environ.get("GENN_RUN") != "1":
        # Gate safety: model.build() is an nvcc compile taking minutes and needing CUDA_PATH;
        # it must NEVER run inside run_selfchecks. The GPU path is an explicit opt-in.
        print(
            "GeNN toolchain present (pygenn importable). GPU build/run is opt-in: set GENN_RUN=1."
        )
        _selfcheck_codegen()
        try:
            oracle()
            have_oracle = True
        except Exception:  # noqa: BLE001
            have_oracle = False
        if have_oracle:
            import numpy as np

            # API-drift canary: construct the full model graph (no nvcc, no load, no training).
            build_model(784, 8, np.zeros(1, dtype=np.float32), 2.0, model_name="stdp_canary")
            print("self-check OK: model graph constructs against installed pygenn")
        return
    run_mnist()


if __name__ == "__main__":
    main()
