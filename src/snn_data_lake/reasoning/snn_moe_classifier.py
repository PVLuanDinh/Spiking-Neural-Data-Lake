r"""
Spike-driven MoE classifier -- Nord's firing-rate routing, ported into this brain.

Ported from Project Nord's SpikeDrivenMoE (NORD-V5/nord_core_700m.py:438). The three
faithful pieces:
  1. firing-rate routing  -- route by the MEAN SPIKE RATE of input feature-clusters,
     aggregated per expert. NO learned router network (Nord keeps only a tiny per-expert
     bias). This is the storage win: a learned router would need N*n_experts weights;
     firing-rate routing needs n_experts biases.
  2. top-k sparse experts -- only top_k of n_experts compute per sample. This is the
     compute win: expert work drops by n_experts/top_k.
  3. load balance         -- Nord spreads usage with an aux GRADIENT loss; a local rule
     cannot backprop one, so this uses the homeostatic equivalent: nudge each expert's
     bias toward an even share against a RUNNING EMA of that expert's own usage. That is
     the running-baseline form P1 asks for, and main() now A/B's it against its own
     homeostat-OFF control instead of asserting a threshold at it.

LEARNING RULE -- e-prop shape, NOT backprop. Experts learn by a local three-factor delta:

    dW[e][c][i] = LR * gate_e * (target_c - p_c) * spikecount_i / STEPS

eligibility (pre-synaptic spike count) x broadcast learning signal (target - prob) x the
routing gate. This module contains no .backward(), no BPTT and no weight transport, which
matters because it IS on a live learning path: cortex/recognizer_bank.py's MoeRecognizer
trains it at construction and calls forward() on every recognition.

Reuses cortex/snn_classifier.py for the data + spike rates (no duplication) and
substrate/_mathutils.softmax (P2: one implementation of a formula, not two).

-----------------------------------------------------------------------------------------
P7 -- what the RETIRED zero-dependency rule cost this module, measured, and what was done
-----------------------------------------------------------------------------------------
Two separate compromises, re-judged on measurement (.venv-lake, this module's own workload:
4 classes x 40 train / 20 test samples, 8 epochs) rather than on what the docstring boasted.

  HEADLINE, module against module, same data, same seed, best of 3, in-process:
        stdlib build (pre-2026-08-04)   train 0.701 s   eval 0.040 s
        this build                      train 0.097 s   eval 0.005 s   -> 7.2x / 8.0x

  RESTORED (numpy) -- the expert bank and its forward pass:
    The bank is now one (n_experts, n_classes, N) array and an expert's forward pass is one
    matvec `W_e @ counts`. Under the stdlib rule it was a nested list driven by a triple loop
    over (timestep x active input x class), fed by a per-step list of active input indices --
    recomputing per TIMESTEP a quantity that depends only on the per-input spike COUNT. That
    list existed solely because there was no array to sum into. Split out in an isolated
    bench so the two causes of the headline are not conflated:
        original triple loop        train 0.353 s
        counts-only, still stdlib   train 0.164 s   (2.2x -- purely algorithmic, no numpy)
        counts + numpy matvec       train 0.132 s   (2.7x -- numpy on the expert bank)

  RESTORED (numpy) -- the Poisson draw, the rest of the headline and the hottest function:
    `rng.random((STEPS, N)) < rates` replaces 1280 scalar random.random() calls per sample.
        stdlib scalar loop              66.6 us/sample
        numpy, Generator per call       14.7 us/sample  (4.5x, including 7.9 us construction)
        numpy, Generator reused          6.8 us/sample  (9.8x -- what train()/evaluate() get)
    The Generator is SEEDED FROM the module-global `random` (see _new_rng), which preserves
    the live consumer's determinism contract exactly: MoeRecognizer seeds and restores that
    global stream around train()/forward() so its decision stays a pure function of its input.
    What changed is the STREAM, not the contract -- the spike trains differ from the stdlib
    build, so every downstream float moves.

  CONSUMER DELTA, measured rather than assumed, because a re-seeded spike train moves every
    number downstream and a 4-decimal match would not have been a null. Paired A/B against
    cortex/recognizer_bank.py with the stdlib build force-loaded, 3 bank seeds x 4 OOD seeds
    x 64 uniform-random draws = 768 paired trials:
        OOD rejection   stdlib 673/768 = 0.8763    numpy 660/768 = 0.8594
        paired t over the 12 cells = -0.79 (n.s.); 5 cells up, 6 down, 1 tie
        in-distribution recognition  30/30 both builds
        worst single cell   stdlib 50/64 (0.781)   numpy 51/64 (0.797)   floor 0.70
    So the 0.844 -> 0.797 that shows up in the bank's default self-check cell is inside the
    build-to-build spread -- the stdlib build owns a worse cell of its own -- not a regression.

  NOT restored, on the merits: route() ranks and softmaxes TOP_K=2 scores. numpy on a
    2-element vector is slower than the builtin sort, so the dependency would buy nothing
    there and the stdlib form stays.

This module prints its own dotted path in main(), derived from __spec__ at runtime, so
re-homing the file cannot leave a stale `python -m ...` instruction behind.
"""

import math
import random

import numpy as np

from snn_data_lake.cortex.snn_classifier import (
    OFF_RATE,
    ON_RATE,
    STEPS,
    N,
    dataset,
    templates,
)
from snn_data_lake.substrate._mathutils import softmax

# random.seed(0) MOVED into main() 2026-08-04 (specs/010, C33). It was an IMPORT-TIME global
# reseed: importing this module silently clobbered the importer's RNG stream. Unlike the ten
# vestigial cases, this file DOES use the global random API, so the seed could not simply be
# deleted -- the self-check's determinism rides on it. Moving it into main() keeps the
# self-check deterministic while removing the side effect on every importer.

# ---- MoE config -------------------------------------------------------------------------
# P1 disposition, stated explicitly because promotion carries these across a tier boundary.
#
# SHAPE, not behaviour-arbitration. These say how many units exist, not where a decision
# flips, so P1 -- which governs values that arbitrate behaviour -- does not reach them:
N_EXPERTS = 8
TOP_K = 2  # experts active per sample (sparse routing)
N_CLUSTERS = 16  # input feature-clusters
CLUSTERS_PER_EXPERT = N_CLUSTERS // N_EXPERTS
# BUDGETS of the training loop (how long / how big a step), not decision thresholds:
LR = 0.5
EPOCHS = 8
# P1 exception (a) AND (b). ROUTE_TEMP scales a MEAN FIRING RATE, which counts/STEPS has
# already normalized to [0,1], so it is a dimensionless fixed point on a normalized output
# -- exception (a). A running-baseline temperature is additionally blocked by exception (b):
# the one live consumer persists only (experts, expert_bias) out of train()'s four returns
# and discards `usage`, so a running baseline would not be reproducible at recognize() time
# from what that consumer actually keeps.
ROUTE_TEMP = 0.5  # Nord's moe_route_temperature
# P1 exception (a). Step sizes on quantities already normalized to [0,1] -- a usage SHARE and
# an EMA of it. The value they drive, expert_bias, is NOT a fixed constant: it is exactly the
# running-baseline-relative homeostatic control P1 asks for, moving against
# target_share = TOP_K/N_EXPERTS, which is derived rather than written down.
LB_RATE = 0.05  # homeostatic load-balance strength (bias nudging)
USAGE_EMA = 0.05  # EMA rate of the per-expert usage share (was an unnamed inline literal)
# SELF-CHECK ONLY -- no consumer reaches these; the live consumer supplies its own data.
TRAIN_PER_CLASS = 40
TEST_PER_CLASS = 20

# Cluster geometry, precomputed once. bincount over these indices reproduces the old
# `cid = i % N_CLUSTERS` accumulation in the same order, so cluster rates stay bit-identical
# to the stdlib build; it also keeps working when N is not a multiple of N_CLUSTERS, which a
# bare reshape would not.
_CLUSTER_OF = np.arange(N) % N_CLUSTERS
_CLUSTER_SIZE = np.maximum(np.bincount(_CLUSTER_OF, minlength=N_CLUSTERS), 1).astype(float)
_EXPERT_SPAN = N_EXPERTS * CLUSTERS_PER_EXPERT  # clusters the expert bank actually consumes


def _new_rng():
    """A numpy Generator seeded FROM the module-global `random`.

    This one line is the hinge that keeps the live consumer's determinism contract while the
    draw itself runs in numpy: recognizer_bank's MoeRecognizer seeds and restores the global
    stream around train() and forward(), so deriving the Generator's seed from that stream
    makes the spike train a pure function of the seed the consumer set -- exactly what the
    scalar random.random() loop gave it, with a different stream.
    """
    return np.random.default_rng(random.getrandbits(128))


def sample_spikes(sample, rng=None):
    """One Poisson realisation of the input: the per-input spike COUNT over STEPS.

    Returns counts only. The stdlib build also returned a per-step list of active input
    indices; the only thing that ever consumed it summed it straight back into counts.
    """
    if len(sample) != N:
        raise ValueError(f"sample must carry {N} inputs (snn_classifier.N), got {len(sample)}")
    if rng is None:
        rng = _new_rng()
    # `!= 0`, not `> 0`: the stdlib build's `ON_RATE if px else OFF_RATE` used Python
    # truthiness, so a negative input took ON_RATE. Kept identical rather than tightened --
    # both live callers pass binarised 0/1, but a silent divergence on an untested input is
    # exactly the kind of thing a rewrite is supposed to not introduce.
    rates = np.where(np.asarray(sample) != 0, ON_RATE, OFF_RATE)
    return (rng.random((STEPS, N)) < rates).sum(axis=0).astype(float)


# ---- 1. firing-rate routing (no learned router) ------------------------------------------
def expert_scores(counts, expert_bias):
    """Nord's _compute_expert_scores: cluster firing rates -> one score per expert.
    cluster of input i = i % N_CLUSTERS; expert score = mean of its clusters + its bias."""
    cluster_rate = np.bincount(_CLUSTER_OF, weights=counts / STEPS, minlength=N_CLUSTERS)
    cluster_rate /= _CLUSTER_SIZE
    per_expert = cluster_rate[:_EXPERT_SPAN].reshape(N_EXPERTS, CLUSTERS_PER_EXPERT).mean(axis=1)
    return per_expert / max(ROUTE_TEMP, 0.01) + expert_bias


def route(scores):
    """top-k experts + softmax gate weights over their scores (Nord's forward).
    Deliberately stdlib: see the P7 note -- numpy on TOP_K=2 elements is slower."""
    top = sorted(range(N_EXPERTS), key=lambda e: -scores[e])[:TOP_K]
    m = max(scores[e] for e in top)
    ex = [math.exp(scores[e] - m) for e in top]
    s = sum(ex)
    return {e: ex[j] / s for j, e in enumerate(top)}  # {expert_idx: gate_weight}


# ---- 2. sparse expert forward ------------------------------------------------------------
def expert_forward(W_e, counts):
    """Class currents from ONE expert over the shared spike train, plus its SynOps.

    Still spike-driven: a synaptic op happens only when an input fires, so an input
    contributes its weight once per spike and `W_e @ counts` IS the sum the old
    per-timestep loop accumulated -- at STEPS-fold less work. SynOps is therefore an exact
    count (total spikes x n_classes), not an estimate.
    """
    return W_e @ counts, int(counts.sum()) * W_e.shape[0]


def forward(experts, expert_bias, sample, n_classes, rng=None):
    """route -> run only top-k experts -> gate-weighted class currents.
    Returns (probs, gates, counts, expert_synops, route_synops)."""
    counts = sample_spikes(sample, rng)
    route_synops = int(counts.sum())  # routing = summing spikes into clusters
    gates = route(expert_scores(counts, expert_bias))
    combined = np.zeros(n_classes)
    esynops = 0
    for e, g in gates.items():
        cur, s = expert_forward(experts[e], counts)
        combined += g * cur
        esynops += s
    return softmax(combined), gates, counts, esynops, route_synops


# ---- 3. train (local delta rule on selected experts + load balance) -----------------------
def train(train_data, n_classes, lb_rate=LB_RATE, report=True):
    """Local three-factor delta rule on the SELECTED experts + homeostatic load balance.

    `lb_rate` is a parameter, not a hidden constant, so main() can run the homeostat's own
    OFF control (lb_rate=0.0). That control is what gives the load-balance criterion a
    reachable red (P8): with the nudging disabled the two runs coincide and the assert fails.
    """
    experts = np.zeros((N_EXPERTS, n_classes, N))
    expert_bias = np.zeros(N_EXPERTS)
    usage = np.full(N_EXPERTS, 1.0 / N_EXPERTS)  # EMA usage share
    target_share = TOP_K / N_EXPERTS
    target = np.zeros(n_classes)
    rng = _new_rng()  # one Generator for the whole run: the 9.8x path, same seeding contract
    total_synops = 0
    for epoch in range(EPOCHS):
        correct = 0
        for sample, label in train_data:
            probs, gates, counts, esynops, _ = forward(experts, expert_bias, sample, n_classes, rng)
            total_synops += esynops
            if max(range(n_classes), key=lambda c: probs[c]) == label:
                correct += 1
            target[:] = 0.0
            target[label] = 1.0
            err = target - np.asarray(probs)
            # update ONLY the selected experts, credit scaled by the gate weight
            for e, g in gates.items():
                experts[e] += np.outer(LR * g * err / STEPS, counts)
            # load balance: EMA usage, nudge bias toward an even share (homeostasis)
            hit = np.zeros(N_EXPERTS)
            hit[list(gates)] = 1.0
            usage += USAGE_EMA * (hit - usage)
            expert_bias += lb_rate * (target_share - usage)
        if report:
            print(f"  epoch {epoch + 1}/{EPOCHS}  train_acc={correct / len(train_data):.1%}")
    return experts, expert_bias, usage, total_synops


def evaluate(experts, expert_bias, data, n_classes):
    """Returns (accuracy, expert_synops, per-expert selection COUNT over `data`)."""
    correct, esynops = 0, 0
    used = np.zeros(N_EXPERTS)
    rng = _new_rng()
    for sample, label in data:
        probs, gates, _counts, s, _ = forward(experts, expert_bias, sample, n_classes, rng)
        esynops += s
        for e in gates:
            used[e] += 1
        if max(range(n_classes), key=lambda c: probs[c]) == label:
            correct += 1
    return correct / len(data), esynops, used


def usage_entropy(share):
    """Shannon entropy of an expert usage share, normalized so 1.0 = perfectly even.
    Dimensionless by construction (a share sums to 1), so it needs no calibration."""
    return -sum(p * math.log(p) for p in share if p > 0) / math.log(N_EXPERTS)


def _share(used):
    total = used.sum()
    return used / total if total else used


def _fmt_share(share):
    return "  ".join(f"e{e}:{s:.0%}" for e, s in enumerate(share))


def main():
    random.seed(0)  # deterministic self-check; scoped to main() so importing is side-effect free
    temps = templates()
    C = len(temps)
    train_data = dataset(TRAIN_PER_CLASS, temps)
    test_data = dataset(TEST_PER_CLASS, temps)
    dotted = __spec__.name if __spec__ else __name__  # derived: cannot go stale when re-homed

    print("=" * 60)
    print("SPIKE-DRIVEN MoE CLASSIFIER  (Nord firing-rate routing, ported)")
    print(f"module: {dotted}")
    print("=" * 60)
    print(f"experts={N_EXPERTS}  top_k={TOP_K}  clusters={N_CLUSTERS}  classes={C}  inputs N={N}")
    print("routing: firing-rate (NO learned router net)\n")

    # The homeostat-OFF control shares this RNG state so the ONLY difference between the two
    # runs is lb_rate. Without the snapshot the control would also see different spikes and
    # the comparison below would not be a control at all.
    train_state = random.getstate()
    experts, expert_bias, _usage, _ = train(train_data, C)
    random.setstate(train_state)
    ctl_experts, ctl_bias, _cu, _cs = train(train_data, C, lb_rate=0.0, report=False)

    eval_state = random.getstate()
    acc, moe_synops, used = evaluate(experts, expert_bias, test_data, C)
    random.setstate(eval_state)
    _ctl_acc, _ctl_syn, ctl_used = evaluate(ctl_experts, ctl_bias, test_data, C)

    print()
    print(f"TEST ACCURACY : {acc:.1%}   (chance = {1 / C:.0%})\n")

    # --- compute: top-k sparse vs running ALL experts -------------------------------------
    # Both figures are exact spike counts (spikes x classes x experts-that-ran); an expert's
    # cost does not depend on WHICH expert it is, so the all-experts number is arithmetic on
    # the measured one. Reported, never asserted: `moe < moe * N_EXPERTS // TOP_K` has no
    # reachable red, and P8 says such a criterion is DELETED rather than weakened.
    dense_all_synops = moe_synops * N_EXPERTS // TOP_K
    print("COMPUTE on test set (lower = less power):")
    print(f"  MoE SynOps (top-{TOP_K})   : {moe_synops:,}")
    print(f"  all-{N_EXPERTS}-experts SynOps : {dense_all_synops:,}")
    print(f"  reduction              : {N_EXPERTS / TOP_K:.1f}x  (= n_experts/top_k)")
    print()

    # --- storage: firing-rate router vs a learned router network --------------------------
    # Two parameter counts over module constants. Also reported, never asserted, for the same
    # P8 reason: 8 < 512 cannot go red for any input.
    fr_router = N_EXPERTS  # just the per-expert biases
    learned_router = N * N_EXPERTS  # a real router: N -> n_experts
    print("ROUTER STORAGE (lower = less space):")
    print(f"  firing-rate router : {fr_router} params (biases only)")
    print(f"  learned router     : {learned_router} params (N x n_experts)")
    print(f"  reduction          : {learned_router / fr_router:.0f}x  (no router net)")
    print()

    # --- load balance, against its own OFF control (P8: a control that is not the method) --
    share, ctl_share = _share(used), _share(ctl_used)
    h_on, h_off = usage_entropy(share), usage_entropy(ctl_share)
    print("LOAD BALANCE (expert usage share on test set):")
    print(f"  homeostat ON  : {_fmt_share(share)}")
    print(f"  homeostat OFF : {_fmt_share(ctl_share)}   <- control, lb_rate=0")
    print(f"  usage entropy : ON={h_on:.4f}  OFF={h_off:.4f}  (1.0 = perfectly even)")
    print("=" * 60)

    # ---- self-check --------------------------------------------------------------------
    # P8 pass, 2026-08-04: three of the four former asserts had NO reachable red and were
    # DELETED rather than weakened --
    #   moe_synops < dense_all_synops   : arithmetic on a single measured number
    #   fr_router  < learned_router     : 8 < 512, two module constants
    #   distinct experts used >= TOP_K  : route() returns exactly TOP_K experts, always
    # The two that survive have a red something can actually reach: shuffle the labels and
    # accuracy collapses to chance; set lb_rate=0 in BOTH runs and the entropy gap closes.
    # acc floor: P1 exception (a) -- a fixed point on a normalized output (accuracy in [0,1],
    # chance 1/C = 25%), inside a self-check that no consumer reaches.
    assert acc >= 0.85, f"MoE did not learn: acc={acc:.2f}"
    assert h_on > h_off, (
        "homeostatic load balance did not spread expert usage further than its own "
        f"homeostat-OFF control (usage entropy ON={h_on:.4f} vs OFF={h_off:.4f})"
    )
    print("self-check OK: acc>=85%; load balance beats its homeostat-OFF control on usage entropy")


if __name__ == "__main__":
    main()
