"""
Two-STAGE latency conv-STDP on MNIST -- a genuinely DEEP unsupervised spiking
hierarchy (Kheradpisheh-2018 SDNN style, pool-between-conv-layers), and an honest
1-vs-2-vs-3-layer head-to-head.  (torch/GPU tier, ruff-excluded; core untouched.)

WHY THIS FILE EXISTS
--------------------
reasoning/deep_stdp_latency.py built the WORKING single conv-STDP stage (the
diversity-creating DoG + latency + first-spike two-axis WTA + multiplicative
soft-bound STDP machinery). It is a 2-layer network (one conv stage + FC). This
file STACKS a SECOND latency conv-STDP stage on top of that frozen first stage,
making a 3-layer deep unsupervised spiking model (conv1 -> pool -> conv2 -> pool
-> FC), and asks the honest question: does a second unsupervised conv stage buy
anything over one stage or over none?

FULLY UNSUPERVISED: NO labels and NO backprop write ANY weight in ANY conv OR FC
(prototype) learning. Every conv kernel and prototype is set purely by spike-
timing competition. Labels touch the pipeline ONLY at the very end, in the
neuron-label majority vote that reads out accuracy -- the identical readout used
by snn_mnist_stdp / deep_stdp / deep_stdp_latency.

THE STAGE-1 -> STAGE-2 CASCADE (the whole point of this file)
------------------------------------------------------------
The key insight: stage 1's per-map FIRST-SPIKE-TIME map is ITSELF a latency code
(earlier spike = stronger feature), so stage 2 needs NO DoG / no re-encoding --
it consumes stage-1's fired-time volley directly, exactly the way stage 1 consumed
the DoG latency volley.

  1. STAGE-1 VOLLEY: freeze conv1, run conv1._forward(conv1._dog_latency(image))
     -> `fired` [C1, hout1, hout1] (long), the per-map first-spike TIME in
     [0, t_lat-1], with t_lat == "never fired". This is already a C1-channel
     latency volley.
  2. POOL BETWEEN STAGES (Kheradpisheh pool-between-conv): spatially MIN-pool that
     time map by NORD_POOL1 (default 2) -- MIN, not max, because EARLIER = stronger,
     so the earliest spike in each window survives; a window fires "never" only if
     ALL its cells never fired (sentinel t_lat is the max value, so min-pool keeps
     it only when nothing else fired). Yields a [C1, H, W] volley, H = hout1//pool1.
  3. STAGE-2 CONV-IF: C2 shared [C2, C1, k2, k2] kernels integrate the pooled
     C1-channel volley with the SAME non-leaky IF as stage 1 -- step t in
     range(t_lat), add the inputs whose spike-time == t to a cumulative volley,
     conv2d the cumulative volley, first threshold (cthresh2) crossing per
     (map,y,x) is its stage-2 first-spike time. never-fired inputs (time == t_lat)
     never enter the cumulative volley.
  4. STAGE-2 WTA + STDP: the SAME first-spike two-axis WTA (_compete: intra-map one
     fire + inter-map neighbourhood claim, <= k_winner distinct-map winners) and the
     SAME multiplicative soft-bound STDP (dw = +-a * w * (1 - w)) as stage 1, now
     over the C1-channel patch spike_time[:, y:y+k2, x:x+k2] of the stage-1 volley.
     Stage 1 stays FROZEN throughout stage-2 training.
  5. STAGE-2 FEATURES: frozen conv1 + conv2 -> stage-2 first-spike-LATENCY map
     ((t_lat - fired2)/t_lat, never -> 0) -> spatial max-pool -> flatten -> /max.
     That vector feeds the FC-STDP prototype layer (the 3-layer readout).

HONEST 1-vs-2-vs-3 HARNESS
--------------------------
main() trains, on the SAME MNIST subset with torch.manual_seed(0) reseeded before
each config:
  - 1-layer : FC-STDP on raw 784 pixels                       (OneLayerNet)
  - 2-layer : conv1 -> pool -> FC-STDP                         (TwoLayerNet, reused)
  - 3-layer : conv1 -> pool -> conv2 -> pool -> FC-STDP        (ThreeLayerNet, new)
and prints all three test accuracies, BOTH conv-layer diversities, and the signed
deltas (2 vs 1, 3 vs 1, 3 vs 2) with a verdict. The deltas are NOT asserted in any
direction -- a tie or a drop is a valid, reported result (adding depth to an
unsupervised STDP hierarchy is NOT guaranteed to help, and this file reports the
truth). What IS asserted, exactly as in stage 1:
  - the 1-layer baseline beats chance (>= 0.5), guarding the shared recipe; and
  - BOTH conv stages are DIVERSE: mean off-diagonal pairwise cosine similarity of
    each stage's kernels < 0.6. Diversity is THE acceptance criterion -- if either
    stage collapsed (near-identical kernels -> non-discriminative / all-zero
    features), fix the mechanism, do NOT lower the bar.

HONEST simplification vs full Kheradpisheh 2018: still intensity-BUCKETED latency
(via stage 1), FIXED per-stage thresholds (not per-map adaptive), two conv+pool
stages (not the deeper SDNN), and no separate temporal-pooling layer. Expect
specialised, non-collapsed maps in BOTH stages and an honest signed depth delta on
a small subset -- the two-stage cascade mechanism, not paper-scale numbers, is the
deliverable.

Run (defaults finish a smoke run in a few minutes on GPU; run from the repo root):
    NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python -m snn_data_lake.reasoning.deep_stdp_2stage
Scale knobs : NORD_C1 NORD_C2 NORD_M NORD_TRAIN NORD_TEST NORD_T NORD_GPU
Stage knobs : NORD_L1_EPOCHS NORD_L2_EPOCHS NORD_POOL1
Stage-2 calibration (optional): NORD_K2 NORD_CTHRESH2 NORD_KWINNER2 NORD_RINH2
                                NORD_APLUS2 NORD_AMINUS2
Deps: numpy torch torchvision snntorch   ([train]/[gpu] extras. NOT "outside the core": this module lives in src/snn_data_lake/reasoning/
         and imports from the package -- P4c, every build is part of the brain.)
"""

import os

import torch
import torch.nn.functional as F

from snn_data_lake.substrate.device import resolve_device, enable
from snn_data_lake.plasticity.snn_mnist_stdp import load_mnist
from snn_data_lake.plasticity.deep_stdp import FcStdp, OneLayerNet, assign_labels, evaluate
from snn_data_lake.reasoning.deep_stdp_latency import (
    ConvLatencyStdp,
    TwoLayerNet,
    _train,
    A_PLUS,
    A_MINUS,
)

K2 = 5  # stage-2 conv kernel size (C1 input channels, valid conv, stride 1)


def _env(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


def _envf(key, default):
    v = os.environ.get(key)
    return float(v) if v else default


def _stage1_volley(conv1, image, pool1):
    """Frozen stage-1 first-spike-TIME map, spatially MIN-pooled (earliest spike
    kept; never == t_lat survives only if every cell in the window never fired).
    Returns a [C1, H, W] long spike-time volley (H == conv1.hout // pool1) -- the
    latency input to stage 2. NO DoG / re-encoding: stage 1's time map IS a latency
    code."""
    st = conv1._dog_latency(image)
    fired, _ = conv1._forward(st)  # [C1, hout1, hout1] long, t_lat=never
    if pool1 > 1:
        # MIN-pool via negated MAX-pool: earlier (smaller) time wins; t_lat sentinel
        # (the largest value) is kept only when nothing in the window fired.
        neg = -fired.float().view(1, conv1.C, conv1.hout, conv1.hout)
        pooled = -F.max_pool2d(neg, pool1)
        fired = pooled[0].round().long()  # [C1, H, W]
    return fired


class ConvLatencyStdp2:
    """Stage 2: C2 shared [C2, C1, k2, k2] conv-IF feature maps over the stage-1
    first-spike-time volley (NOT DoG). Same non-leaky IF + first-spike two-axis WTA
    + multiplicative soft-bound STDP as stage 1, at most k_winner distinct-map
    updates per image. Stage 1 is frozen during stage-2 training."""

    def __init__(
        self, C1, C2, t_lat, hout1, pool1, k2=None, cthresh2=None, k_winner=None, r_inh=None
    ):
        self.C1 = C1
        self.C2 = C2
        self.t_lat = t_lat  # latency steps (== NORD_T)
        self.pool1 = pool1  # stage-1 -> stage-2 pool factor
        self.h_in = hout1 // pool1  # pooled stage-1 spatial size
        self.k2 = _env("NORD_K2", K2) if k2 is None else k2
        self.hout = self.h_in - self.k2 + 1  # stage-2 valid-conv output size
        self.cthresh2 = _envf("NORD_CTHRESH2", 5.0) if cthresh2 is None else cthresh2
        self.k_winner = _env("NORD_KWINNER2", 5) if k_winner is None else k_winner
        self.r_inh = _env("NORD_RINH2", 1) if r_inh is None else r_inh
        self.a_plus = _envf("NORD_APLUS2", A_PLUS)
        self.a_minus = _envf("NORD_AMINUS2", A_MINUS)
        # shared kernels [C2, C1, k2, k2] init ~Normal(0.8, 0.05) clamped [0,1].
        # GPU-safe: torch.normal(float mean, float std, size=) IGNORES
        # set_default_device (a torch quirk) and lands on CPU, so pin EVERY created
        # tensor to the resolved device -- this exact bug already bit stage 1.
        self.device = resolve_device()
        self.Wc2 = (
            torch.normal(0.8, 0.05, size=(C2, C1, self.k2, self.k2))
            .clamp_(0.0, 1.0)
            .to(self.device)
        )
        self.pool = max(1, self.hout // 4)  # spatial max-pool to ~4x4

    # -- forward: non-leaky IF over the C1-channel stage-1 volley, first-spike time -
    def _forward2(self, spike_time):
        """Integrate the [C1,H,W] stage-1 spike-time volley. Returns (first-spike-
        time [C2,hout,hout] with t_lat == never, final potential [C2,hout,hout]).
        Non-leaky IF => potential is the conv of the CUMULATIVE input volley, so it
        is monotonic and the first threshold crossing is well defined. never-fired
        inputs (time == t_lat) never enter the cumulative volley."""
        weight = self.Wc2  # [C2,C1,k2,k2]
        cum = torch.zeros(self.C1, self.h_in, self.h_in, device=self.device)
        fired = torch.full(
            (self.C2, self.hout, self.hout), self.t_lat, dtype=torch.long, device=self.device
        )
        pot = torch.zeros(self.C2, self.hout, self.hout, device=self.device)
        for t in range(self.t_lat):
            vol = (spike_time == t).float()  # [C1,H,W]
            cum = cum + vol
            pot = F.conv2d(cum.unsqueeze(0), weight)[0]  # [C2,hout,hout]
            newly = (pot >= self.cthresh2) & (fired == self.t_lat)
            if newly.any():
                fired = torch.where(newly, torch.full_like(fired, t), fired)
        return fired, pot

    # -- competition: first-spike two-axis WTA (identical rule to stage 1) ---------
    def _compete(self, fired, pot):
        """Globally earliest winners under intra-map (one per map) and inter-map (one
        map owns a neighbourhood of radius r_inh) inhibition; <= k_winner distinct-map
        winners per image. Returns a list of (f, y, x)."""
        mask = fired < self.t_lat
        if not bool(mask.any()):
            return []
        idx = mask.nonzero(as_tuple=False)  # [N,3] f,y,x
        times = fired[mask].float()
        pots = pot[mask]
        order = torch.argsort(times * 1e6 - pots)  # earliest, tie-break stronger
        idx = idx[order].tolist()
        winners = []
        maps_won = set()
        claimed = set()
        r = self.r_inh
        for f, y, x in idx:
            if len(winners) >= self.k_winner:
                break
            if f in maps_won:  # intra-map WTA
                continue
            if (y, x) in claimed:  # inter-map WTA
                continue
            winners.append((f, y, x))
            maps_won.add(f)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    claimed.add((y + dy, x + dx))
        return winners

    # -- multiplicative soft-bound STDP over the C1-channel patch (as stage 1) -----
    def _stdp(self, f, y, x, spike_time, post_time):
        """Update shared kernel f [C1,k2,k2] toward its winning input patch. Pre
        inputs (across all C1 stage-1 maps) that fired at/before the post spike
        potentiate; the rest depress; w*(1-w) soft bound drives each weight toward 0
        or 1 (no renormalisation)."""
        patch = spike_time[:, y : y + self.k2, x : x + self.k2]  # [C1,k2,k2] times
        pre_before = (patch <= post_time) & (patch < self.t_lat)  # actually fired
        w = self.Wc2[f]
        dw = torch.where(pre_before, self.a_plus * w * (1.0 - w), -self.a_minus * w * (1.0 - w))
        self.Wc2[f] = (w + dw).clamp_(0.0, 1.0)

    def train_image2(self, image, conv1):
        """One unsupervised stage-2 update: frozen stage-1 pooled volley -> forward2
        -> compete -> STDP the winners."""
        st = _stage1_volley(conv1, image, self.pool1)
        fired, pot = self._forward2(st)
        for f, y, x in self._compete(fired, pot):
            self._stdp(f, y, x, st, int(fired[f, y, x].item()))

    # -- frozen feature extraction for the FC prototype layer ----------------------
    def features2(self, image, conv1):
        """Frozen conv1 + conv2 -> stage-2 first-spike-LATENCY map (earlier = stronger,
        never = 0) -> spatial max-pool -> flatten -> rescale to [0,1] (FC input)."""
        st = _stage1_volley(conv1, image, self.pool1)
        fired, _ = self._forward2(st)
        val = torch.where(
            fired < self.t_lat,
            (self.t_lat - fired).float() / self.t_lat,
            torch.zeros_like(fired, dtype=torch.float32),
        )
        pooled = F.max_pool2d(val.view(1, self.C2, self.hout, self.hout), self.pool)
        feat = pooled.reshape(-1)
        return feat / feat.max().clamp_min(1e-6)

    def feat_dim(self):
        ph = self.hout // self.pool
        return self.C2 * ph * ph

    def kernel_diversity(self):
        """Mean off-diagonal pairwise cosine similarity of the C2 flattened
        [C1*k2*k2] kernels. LOW == stage 2 specialised (did not collapse)."""
        flat = self.Wc2.reshape(self.C2, -1)
        n = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        s = n @ n.t()
        off = (s.sum() - torch.diagonal(s).sum()) / (self.C2 * (self.C2 - 1))
        return float(off.item())


class ThreeLayerNet:
    """Deep: frozen conv1 -> pool -> frozen conv2 -> pool -> FC-STDP prototypes."""

    def __init__(self, conv1, conv2, m):
        self.conv1 = conv1  # trained + frozen
        self.conv2 = conv2  # trained + frozen
        self.fc = FcStdp(conv2.feat_dim(), m)

    def train_image(self, image):
        self.fc.run(self.conv2.features2(image, self.conv1), learn=True)

    def run(self, image, learn=False):
        return self.fc.run(self.conv2.features2(image, self.conv1), learn)


def _train2(conv2, conv1, X, tag):
    """Stage-2 conv training loop (needs the frozen conv1 for the cascade)."""
    for i, img in enumerate(X):
        conv2.train_image2(img, conv1)
        if (i + 1) % 2000 == 0:
            print(f"  {tag}: {i + 1}/{len(X)} images")


def main():
    enable(resolve_device())
    C1 = _env("NORD_C1", 30)
    C2 = _env("NORD_C2", 60)
    M = _env("NORD_M", 100)
    TRAIN_N = _env("NORD_TRAIN", 8000)
    TEST_N = _env("NORD_TEST", 2000)
    T_LAT = _env("NORD_T", 15)
    POOL1 = _env("NORD_POOL1", 2)
    L1_EPOCHS = _env("NORD_L1_EPOCHS", 3)
    L2_EPOCHS = _env("NORD_L2_EPOCHS", 3)

    print("=" * 70)
    print("TWO-STAGE LATENCY CONV-STDP  1 vs 2 vs 3 LAYER  (MNIST, Kheradpisheh SDNN)")
    print("=" * 70)
    print(f"C1={C1}  C2={C2}  M={M}  latency-steps T={T_LAT}  train={TRAIN_N}  test={TEST_N}")
    print(
        f"pool1={POOL1}  L1_epochs={L1_EPOCHS}  L2_epochs={L2_EPOCHS}  "
        f"k2={_env('NORD_K2', K2)}  cthresh2={_envf('NORD_CTHRESH2', 5.0)}"
    )
    print(
        "conv1(DoG->latency->IF WTA STDP) -> pool -> conv2(volley->IF WTA STDP) -> pool -> FC-STDP"
    )
    print("(NO labels, NO backprop; labels only in the final neuron-label vote)\n")
    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)
    n_assign = min(TRAIN_N, 6000)

    # ---------- 1-layer baseline: FC-STDP on raw pixels ----------
    torch.manual_seed(0)
    print("\n[1-layer] training FC-STDP prototypes on raw pixels...")
    net1 = OneLayerNet(M)
    _train(net1, Xtr, "1-layer")
    print("[1-layer] neuron-label vote + evaluate...")
    a1 = assign_labels(net1, Xtr[:n_assign], Ytr[:n_assign], M)
    acc1 = evaluate(net1, Xte, Yte, a1, M)

    # ---------- 2-layer: conv1 -> pool -> FC-STDP ----------
    torch.manual_seed(0)
    print(f"\n[2-layer] training conv-latency-STDP stage 1 (unsupervised, {L1_EPOCHS} epochs)...")
    conv1 = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv1, Xtr, f"stage1 ep{ep + 1}/{L1_EPOCHS}")
        print(
            f"  [stage1] epoch {ep + 1}/{L1_EPOCHS} kernel diversity (cos-sim) = "
            f"{conv1.kernel_diversity():.3f}"
        )
    div1 = conv1.kernel_diversity()
    print(f"[2-layer] stage 1 frozen; stage-1 feature dim = {conv1.feat_dim()}")
    print(f"[2-layer] STAGE-1 KERNEL DIVERSITY (mean pairwise cos-sim) = {div1:.3f}")
    print("[2-layer] training FC-STDP on pooled stage-1 first-spike features...")
    net2 = TwoLayerNet(conv1, M)
    _train(net2, Xtr, "2-layer FC")
    print("[2-layer] neuron-label vote + evaluate...")
    a2 = assign_labels(net2, Xtr[:n_assign], Ytr[:n_assign], M)
    acc2 = evaluate(net2, Xte, Yte, a2, M)

    # ---------- 3-layer: conv1 -> pool -> conv2 -> pool -> FC-STDP ----------
    # Reuse the SAME trained+frozen stage 1; stack a second frozen conv stage on top.
    torch.manual_seed(0)
    print(
        f"\n[3-layer] training conv-latency-STDP stage 2 on the stage-1 volley "
        f"(unsupervised, {L2_EPOCHS} epochs)..."
    )
    conv2 = ConvLatencyStdp2(C1, C2, T_LAT, conv1.hout, POOL1)
    print(
        f"  stage-2 input volley = {C1}x{conv2.h_in}x{conv2.h_in}  "
        f"-> conv -> {C2}x{conv2.hout}x{conv2.hout}"
    )
    for ep in range(L2_EPOCHS):
        _train2(conv2, conv1, Xtr, f"stage2 ep{ep + 1}/{L2_EPOCHS}")
        print(
            f"  [stage2] epoch {ep + 1}/{L2_EPOCHS} kernel diversity (cos-sim) = "
            f"{conv2.kernel_diversity():.3f}"
        )
    div2 = conv2.kernel_diversity()
    print(f"[3-layer] stage 2 frozen; stage-2 feature dim = {conv2.feat_dim()}")
    print(f"[3-layer] STAGE-2 KERNEL DIVERSITY (mean pairwise cos-sim) = {div2:.3f}")
    print("[3-layer] training FC-STDP on pooled stage-2 first-spike features...")
    net3 = ThreeLayerNet(conv1, conv2, M)
    _train(net3, Xtr, "3-layer FC")
    print("[3-layer] neuron-label vote + evaluate...")
    a3 = assign_labels(net3, Xtr[:n_assign], Ytr[:n_assign], M)
    acc3 = evaluate(net3, Xte, Yte, a3, M)

    # ---------- honest 1 vs 2 vs 3 ----------
    d21 = acc2 - acc1
    d31 = acc3 - acc1
    d32 = acc3 - acc2
    print("\n" + "=" * 70)
    print("RESULT  (unsupervised STDP, neuron-label-vote readout; chance = 10%)")
    print("-" * 70)
    print(f"  1-layer  (FC-STDP on 784 pixels)                        : {acc1:.1%}")
    print(f"  2-layer  (conv1 -> pool -> FC-STDP)                     : {acc2:.1%}")
    print(f"  3-layer  (conv1 -> pool -> conv2 -> pool -> FC-STDP)    : {acc3:.1%}")
    print("-" * 70)
    print(f"  delta 2 vs 1 (one conv stage helps?)                   : {d21:+.1%}")
    print(f"  delta 3 vs 1 (deep hierarchy vs pixels?)               : {d31:+.1%}")
    print(f"  delta 3 vs 2 (second conv stage helps?)                : {d32:+.1%}")

    def _verdict(d):
        return "HELPS" if d > 0 else ("TIE" if d == 0 else "HURTS")

    print(
        f"  verdict  2 vs 1 = {_verdict(d21)}   3 vs 1 = {_verdict(d31)}   3 vs 2 = {_verdict(d32)}"
    )
    print(f"  stage-1 kernel diversity (cos-sim, lower=better)        : {div1:.3f}")
    print(f"  stage-2 kernel diversity (cos-sim, lower=better)        : {div2:.3f}")
    print("=" * 70)

    # self-checks: the baseline must beat chance (guards the shared recipe), and BOTH
    # conv stages must be DIVERSE (mean pairwise cos-sim < 0.6). Diversity is THE
    # acceptance criterion: a collapsed stage has near-identical kernels -> non-
    # discriminative (or all-zero) features. We deliberately do NOT assert any depth
    # delta -- a tie or a drop is a valid, honestly reported result (a second
    # unsupervised conv stage is NOT guaranteed to help).
    assert acc1 >= 0.5, f"1-layer baseline failed to beat chance: acc={acc1:.2f}"
    assert div1 < 0.6, (
        f"stage-1 maps COLLAPSED: mean pairwise cos-sim={div1:.3f} (need < 0.6). "
        "Fix the stage-1 WTA/STDP mechanism (do NOT lower the bar)."
    )
    assert div2 < 0.6, (
        f"stage-2 maps COLLAPSED: mean pairwise cos-sim={div2:.3f} (need < 0.6). "
        "The second conv stage did not separate its maps -- fix the cascade/WTA/STDP "
        "or the cthresh2 calibration (do NOT lower the bar)."
    )
    print(
        f"self-check OK: 1-layer acc={acc1:.1%} (>10% chance); BOTH conv stages "
        f"DIVERSE (stage1 {div1:.3f}, stage2 {div2:.3f} < 0.6, no collapse); "
        f"deltas 2v1 {d21:+.1%} / 3v1 {d31:+.1%} / 3v2 {d32:+.1%} (honest)."
    )


if __name__ == "__main__":
    main()
