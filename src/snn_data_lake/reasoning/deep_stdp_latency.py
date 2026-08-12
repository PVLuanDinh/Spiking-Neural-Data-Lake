"""
Latency-coded convolutional STDP on MNIST -- the Kheradpisheh-2018 machinery
that PREVENTS map collapse.  (torch/GPU tier, ruff-excluded; core untouched.)

WHY THIS FILE EXISTS
--------------------
reasoning/deep_stdp.py built a SIMPLIFIED deep unsupervised conv-STDP and it
COLLAPSED to chance: RATE (Poisson) coding + a single greedy most-spikes winner
+ ADDITIVE STDP drove all C conv maps to near-identical filters (pairwise cosine
similarity ~0.96), so the pooled features carried no class information. This file
replaces those four choices with the FULL diversity-creating machinery from
Kheradpisheh et al. 2018 ("STDP-based spiking deep convolutional neural networks
for object recognition"), the version that reaches paper-level unsupervised MNIST.

FULLY UNSUPERVISED: NO labels and NO backprop write ANY weight in the feature
(conv) OR prototype (FC) learning. Every synapse is set purely by spike-timing
competition. Labels touch the pipeline ONLY at the very end, in the neuron-label
majority vote that reads out accuracy (each output neuron -> the class it fires
most for) -- the identical readout used by snn_mnist_stdp / deep_stdp.

THE FOUR ANTI-COLLAPSE MECHANISMS (this is the whole point)
----------------------------------------------------------
1. DoG PREPROCESSING (ON + OFF center-surround). A zero-sum 7x7 Difference-of-
   Gaussians (center sigma=1.0, surround sigma=2.0) is convolved over each image.
   ON-center = ReLU(DoG), OFF-center = ReLU(-DoG); small responses are zeroed ->
   two SPARSE contrast (edge-saliency) channels. Flat regions -> ~0 (zero-sum).

2. LATENCY / RANK-ORDER CODING. Each DoG contrast value becomes a single spike
   TIME over T_LAT steps: the STRONGER the contrast the EARLIER it fires; zero
   contrast never fires; every input neuron fires AT MOST ONCE. The image is thus
   a temporally-ordered volley, most-salient-first (intensity-bucketed rank order).

3. CONV INTEGRATE-and-FIRE LAYER + FIRST-SPIKE TWO-AXIS WTA (the anti-collapse
   core). C shared kernels (default 30, k=5, over the 2 DoG channels, valid conv)
   integrate the input volley step by step (pure IF, non-leaky). A conv neuron
   fires when its potential FIRST crosses threshold. Per image the winners are the
   globally EARLIEST-firing neurons (tie-break by potential), selected under TWO
   inhibitions: (a) INTRA-MAP -- each map fires only ONCE per image (its earliest/
   strongest location; all other positions of that map are suppressed); (b) INTER-
   MAP -- once a map wins near a position, NO OTHER map may win in that
   neighbourhood (one map owns a location). At most k_winner distinct-map winners
   per image. This double WTA forces different maps to learn DIFFERENT features.

4. MULTIPLICATIVE SOFT-BOUND STDP. For a winning conv neuron with post spike-time
   t*, each presynaptic input in its k x k patch either fired BEFORE t* (pre-before-
   post -> potentiate) or fired after / never (-> depress):
        dw = +a_plus  * w * (1 - w)      (pre fired at time <= t*)
        dw = -a_minus * w * (1 - w)      (pre fired after t*, or never)
   with a_plus=0.004, a_minus=0.003, weights init ~Normal(0.8, 0.05) clamped [0,1].
   The w*(1-w) soft bound pushes each weight toward 0 or 1 -> sharp, distinct,
   stable binary features. NO renormalisation (the soft bound handles stability).
   Additive STDP (the old file) has no such attractor and was part of the collapse.
   Layer 1 is trained GREEDILY for several epochs (NORD_L1_EPOCHS, default 3) to
   convergence, as Kheradpisheh does: with these small rates a single pass over a
   6000-image subset only reaches ~0.67 cos-sim, whereas ~2-3 epochs sharpen the
   kernels to ~0.35-0.40 (frac binary ~0.9). The FC prototype layer and the
   1-layer baseline stay single-pass (they converge in one pass, as in deep_stdp).

Then: freeze layer 1, forward each image -> per-map FIRST-SPIKE-LATENCY map
(earlier spike = stronger feature; never-fired = 0) -> spatial MAX-POOL -> flatten
-> feed the FC-STDP prototype layer (M neurons) + neuron-label vote, exactly like
deep_stdp.py's layer 2 (FcStdp/assign_labels/evaluate are reused verbatim).

HONEST 1-vs-2 HARNESS
---------------------
main() trains, on the SAME MNIST subset with torch.manual_seed(0) reseeded before
each config, a "1-layer" baseline (FC-STDP on raw 784 pixels) and a "2-layer" deep
model (DoG -> latency -> conv-STDP -> pool -> FC-STDP), then prints both test
accuracies + the signed delta + verdict. The delta is NOT asserted in either
direction (a negative delta is a valid, reported result). What IS asserted:
  - the 1-layer baseline beats chance (>= 0.5), guarding the shared recipe; and
  - the layer-1 kernels are DIVERSE: mean off-diagonal pairwise cosine similarity
    of the C kernels < 0.6 (the old file was ~0.96). Diversity is the acceptance
    criterion here -- if it is still high the WTA/STDP is wrong, not the bar.

HONEST simplification vs full Kheradpisheh 2018: this is a SINGLE conv+pool stage
(not the multi-layer SDNN), intensity-BUCKETED latency (not strict per-neuron rank
order), a FIXED conv threshold (not a per-map adaptive/aggregation threshold), and
no separate temporal-pooling layer. Expect specialised, non-collapsed maps and a
sane baseline on a small subset -- the mechanisms, not paper-scale numbers, are
the deliverable.

Run (defaults finish a smoke run in a few minutes on GPU):
    NORD_GPU=1 PYTHONPATH=src ../.venv-cu128/Scripts/python -m snn_data_lake.reasoning.deep_stdp_latency
Scale knobs: NORD_C NORD_M NORD_TRAIN NORD_TEST NORD_GPU NORD_T
Calibration knobs (optional): NORD_L1_EPOCHS NORD_CTHRESH NORD_KWINNER NORD_RINH
                              NORD_APLUS NORD_AMINUS
Deps: numpy torch torchvision snntorch   ([train]/[gpu] extras. NOT "outside the core": this module lives in src/snn_data_lake/reasoning/
         and imports from the package -- P4c, every build is part of the brain.)
"""

import os

import torch
import torch.nn.functional as F

from snn_data_lake.substrate.device import resolve_device, enable
from snn_data_lake.plasticity.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.deep_stdp import FcStdp, OneLayerNet, assign_labels, evaluate

K = 5  # conv kernel size (2 input channels, valid conv, stride 1)
A_PLUS = 0.004  # multiplicative STDP potentiation rate (pre-before-post)
A_MINUS = 0.003  # multiplicative STDP depression rate (pre-after / never)
DOG_SIZE = 7  # Difference-of-Gaussians window
DOG_C_SIGMA = 1.0  # center Gaussian sigma
DOG_S_SIGMA = 2.0  # surround Gaussian sigma
DOG_THRESH = 0.10  # zero DoG responses below this fraction of the per-image max


def _env(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


def _envf(key, default):
    v = os.environ.get(key)
    return float(v) if v else default


def _dog_kernel(size, c_sigma, s_sigma):
    """Zero-sum Difference-of-Gaussians (center minus surround). Lands on the
    default device (GPU-safe under set_default_device)."""
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r2 = xx * xx + yy * yy
    center = torch.exp(-r2 / (2.0 * c_sigma * c_sigma))
    center = center / center.sum().clamp_min(1e-6)
    surround = torch.exp(-r2 / (2.0 * s_sigma * s_sigma))
    surround = surround / surround.sum().clamp_min(1e-6)
    dog = center - surround
    return dog - dog.mean()  # exact zero-sum -> flat regions give 0


class ConvLatencyStdp:
    """Layer 1: C shared k x k conv-IF feature maps over the 2-channel DoG latency
    volley. Unsupervised first-spike two-axis WTA + multiplicative soft-bound STDP,
    at most k_winner distinct-map updates per image."""

    def __init__(self, C, t_lat, k=K, cthresh=None, k_winner=None, r_inh=None):
        self.C = C
        self.k = k
        self.t_lat = t_lat  # latency steps (== NORD_T)
        self.hout = 28 - k + 1  # valid conv, stride 1 -> 24 for k=5
        self.cthresh = _envf("NORD_CTHRESH", 5.0) if cthresh is None else cthresh
        self.k_winner = _env("NORD_KWINNER", 5) if k_winner is None else k_winner
        self.r_inh = _env("NORD_RINH", 2) if r_inh is None else r_inh
        self.a_plus = _envf("NORD_APLUS", A_PLUS)
        self.a_minus = _envf("NORD_AMINUS", A_MINUS)
        # shared kernels [C, 2, k, k] init ~Normal(0.8, 0.05) clamped [0,1] -> the
        # multiplicative soft bound sharpens them toward distinct binary features.
        # GPU-safe: torch.normal(float mean, float std, size=) IGNORES set_default_device
        # (a torch quirk) and lands on CPU, so pin every tensor to the resolved device.
        self.device = resolve_device()
        self.Wc = torch.normal(0.8, 0.05, size=(C, 2, k, k)).clamp_(0.0, 1.0).to(self.device)
        self.dog = (
            _dog_kernel(DOG_SIZE, DOG_C_SIGMA, DOG_S_SIGMA)
            .view(1, 1, DOG_SIZE, DOG_SIZE)
            .to(self.device)
        )
        self.pool = max(1, self.hout // 6)  # spatial max-pool to ~6x6

    # -- mechanism 1 + 2: DoG ON/OFF -> intensity-bucketed latency spike times ----
    def _dog_latency(self, image):
        """Return spike-time tensor [2,28,28] (long): time in [0, t_lat-1] per input
        neuron, or t_lat (== "never") where there is no contrast. Stronger contrast
        fires EARLIER; each input fires at most once."""
        img = image.to(self.device).reshape(1, 1, 28, 28)
        resp = F.conv2d(img, self.dog, padding=DOG_SIZE // 2)[0, 0]  # [28,28]
        on = F.relu(resp)
        off = F.relu(-resp)
        combined = torch.stack((on, off), dim=0)  # [2,28,28]
        vmax = combined.max()
        st = torch.full((2, 28, 28), self.t_lat, dtype=torch.long, device=self.device)
        if float(vmax) > 0:
            thr = DOG_THRESH * vmax
            active = combined > thr  # sparse mask
            norm = (combined / vmax).clamp(0.0, 1.0)
            # strongest contrast -> time 0 ; weakest active -> time t_lat-1
            times = torch.round((1.0 - norm) * (self.t_lat - 1)).long()
            st = torch.where(active, times, st)
        return st

    # -- mechanism 3 (forward half): non-leaky IF, first-spike time per neuron -----
    def _forward(self, spike_time):
        """Integrate the latency volley. Returns (first-spike-time [C,hout,hout] with
        t_lat == never, final potential map [C,hout,hout]). Non-leaky IF => potential
        is the conv of the CUMULATIVE input volley, so it is monotonic and the first
        threshold crossing is well defined."""
        weight = self.Wc  # [C,2,k,k]
        cum = torch.zeros(2, 28, 28, device=self.device)
        fired = torch.full(
            (self.C, self.hout, self.hout), self.t_lat, dtype=torch.long, device=self.device
        )
        pot = torch.zeros(self.C, self.hout, self.hout, device=self.device)
        for t in range(self.t_lat):
            vol = (spike_time == t).float()  # [2,28,28]
            cum = cum + vol
            pot = F.conv2d(cum.unsqueeze(0), weight)[0]  # [C,hout,hout]
            newly = (pot >= self.cthresh) & (fired == self.t_lat)
            if newly.any():
                fired = torch.where(newly, torch.full_like(fired, t), fired)
        return fired, pot

    # -- mechanism 3 (competition half): first-spike two-axis WTA ------------------
    def _compete(self, fired, pot):
        """Pick the globally earliest winners under intra-map (one per map) and
        inter-map (one map owns a neighbourhood of radius r_inh) inhibition; at most
        k_winner distinct-map winners per image. Returns a list of (f, y, x)."""
        mask = fired < self.t_lat
        if not bool(mask.any()):
            return []
        idx = mask.nonzero(as_tuple=False)  # [N,3] f,y,x
        times = fired[mask].float()
        pots = pot[mask]
        # order: earliest spike first, break ties by stronger potential
        order = torch.argsort(times * 1e6 - pots)
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
            for dy in range(-r, r + 1):  # claim its patch
                for dx in range(-r, r + 1):
                    claimed.add((y + dy, x + dx))
        return winners

    # -- mechanism 4: multiplicative soft-bound STDP ------------------------------
    def _stdp(self, f, y, x, spike_time, post_time):
        """Update shared kernel f toward the winning input patch. Pre inputs that
        fired at/before the post spike potentiate; the rest depress; w*(1-w) soft
        bound drives each weight toward 0 or 1 (no renormalisation)."""
        patch = spike_time[:, y : y + self.k, x : x + self.k]  # [2,k,k] times
        pre_before = (patch <= post_time) & (patch < self.t_lat)  # actually fired
        w = self.Wc[f]
        dw = torch.where(pre_before, self.a_plus * w * (1.0 - w), -self.a_minus * w * (1.0 - w))
        self.Wc[f] = (w + dw).clamp_(0.0, 1.0)

    def train_image(self, image):
        st = self._dog_latency(image)
        fired, pot = self._forward(st)
        for f, y, x in self._compete(fired, pot):
            self._stdp(f, y, x, st, int(fired[f, y, x].item()))

    # -- frozen feature extraction for layer 2 ------------------------------------
    def features(self, image):
        """Frozen forward -> first-spike-LATENCY map (earlier spike = stronger, never
        = 0) -> spatial max-pool -> flatten -> rescale to [0,1] (layer-2 FC input)."""
        st = self._dog_latency(image)
        fired, _ = self._forward(st)
        val = torch.where(
            fired < self.t_lat,
            (self.t_lat - fired).float() / self.t_lat,
            torch.zeros_like(fired, dtype=torch.float32),
        )
        pooled = F.max_pool2d(val.view(1, self.C, self.hout, self.hout), self.pool)
        feat = pooled.reshape(-1)
        return feat / feat.max().clamp_min(1e-6)

    def feat_dim(self):
        ph = self.hout // self.pool
        return self.C * ph * ph

    def kernel_diversity(self):
        """Mean off-diagonal pairwise cosine similarity of the C flattened kernels.
        LOW == the maps specialised (did not collapse). The old file was ~0.96."""
        flat = self.Wc.reshape(self.C, -1)
        n = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
        s = n @ n.t()
        off = (s.sum() - torch.diagonal(s).sum()) / (self.C * (self.C - 1))
        return float(off.item())


class TwoLayerNet:
    """Deep: frozen conv-latency-STDP layer 1 -> pooled features -> FC-STDP prototypes."""

    def __init__(self, conv, m):
        self.conv = conv  # already trained + frozen
        self.fc = FcStdp(conv.feat_dim(), m)

    def train_image(self, image):
        self.fc.run(self.conv.features(image), learn=True)

    def run(self, image, learn=False):
        return self.fc.run(self.conv.features(image), learn)


def _train(net, X, tag):
    for i, img in enumerate(X):
        net.train_image(img)
        if (i + 1) % 2000 == 0:
            print(f"  {tag}: {i + 1}/{len(X)} images")


def main():
    enable(resolve_device())
    C = _env("NORD_C", 30)
    M = _env("NORD_M", 100)
    TRAIN_N = _env("NORD_TRAIN", 6000)
    TEST_N = _env("NORD_TEST", 2000)
    T_LAT = _env("NORD_T", 15)

    print("=" * 66)
    print("LATENCY CONV-STDP vs SINGLE-LAYER  (MNIST, Kheradpisheh-2018 machinery)")
    print("=" * 66)
    print(f"C={C}  M={M}  k={K}  latency-steps T={T_LAT}  train={TRAIN_N}  test={TEST_N}")
    print(
        f"conv-thresh={_envf('NORD_CTHRESH', 5.0)}  k_winner={_env('NORD_KWINNER', 5)}  "
        f"r_inh={_env('NORD_RINH', 2)}  a+={_envf('NORD_APLUS', A_PLUS)}  "
        f"a-={_envf('NORD_AMINUS', A_MINUS)}"
    )
    print(
        "DoG ON/OFF -> latency rank-order -> conv-IF first-spike 2-axis WTA -> mult soft-bound STDP"
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

    # ---------- 2-layer deep: DoG -> latency -> conv-STDP -> pool -> FC-STDP ------
    torch.manual_seed(0)
    L1_EPOCHS = _env("NORD_L1_EPOCHS", 3)
    print(f"\n[2-layer] training conv-latency-STDP layer 1 (unsupervised, {L1_EPOCHS} epochs)...")
    conv = ConvLatencyStdp(C, T_LAT)
    for ep in range(L1_EPOCHS):
        _train(conv, Xtr, f"2-layer L1 ep{ep + 1}/{L1_EPOCHS}")
        print(
            f"  [L1] epoch {ep + 1}/{L1_EPOCHS} kernel diversity (cos-sim) = "
            f"{conv.kernel_diversity():.3f}"
        )
    diversity = conv.kernel_diversity()
    print(f"[2-layer] layer-1 frozen; feature dim = {conv.feat_dim()}")
    print(f"[2-layer] LAYER-1 KERNEL DIVERSITY (mean pairwise cos-sim) = {diversity:.3f}")
    print("[2-layer] training FC-STDP layer 2 on pooled first-spike features...")
    net2 = TwoLayerNet(conv, M)
    _train(net2, Xtr, "2-layer L2")
    print("[2-layer] neuron-label vote + evaluate...")
    a2 = assign_labels(net2, Xtr[:n_assign], Ytr[:n_assign], M)
    acc2 = evaluate(net2, Xte, Yte, a2, M)

    # ---------- honest head-to-head ----------
    delta = acc2 - acc1
    print("\n" + "=" * 66)
    print("RESULT  (unsupervised STDP, neuron-label-vote readout; chance = 10%)")
    print("-" * 66)
    print(f"  1-layer  (FC-STDP on 784 pixels)                : {acc1:.1%}")
    print(f"  2-layer  (DoG -> latency -> conv-STDP -> FC-STDP) : {acc2:.1%}")
    print(f"  delta    (2-layer minus 1-layer)                : {delta:+.1%}")
    verdict = "DEEP HELPS" if delta > 0 else ("TIE" if delta == 0 else "DEEP HURTS")
    print(f"  verdict                                          : {verdict}")
    print(f"  layer-1 kernel diversity (cos-sim, lower=better)  : {diversity:.3f}")
    print("=" * 66)

    # self-checks: the baseline must beat chance (guards the shared recipe), and the
    # conv maps must be DIVERSE (mean pairwise cos-sim < 0.6). Diversity is THE
    # acceptance criterion: the old rate-coded single-greedy-winner additive-STDP
    # build collapsed to ~0.96 cos-sim (identical filters -> non-discriminative
    # features -> chance). We deliberately do NOT assert 2-layer > 1-layer; the
    # signed delta is reported as-is whatever its sign.
    assert acc1 >= 0.5, f"1-layer baseline failed to beat chance: acc={acc1:.2f}"
    assert diversity < 0.6, (
        f"layer-1 maps COLLAPSED: mean pairwise cos-sim={diversity:.3f} (need < 0.6). "
        "The two-axis WTA or multiplicative STDP is not separating the maps -- fix "
        "the mechanism (do NOT lower the bar)."
    )
    print(
        f"self-check OK: 1-layer acc={acc1:.1%} (>10% chance); kernels DIVERSE "
        f"(cos-sim {diversity:.3f} < 0.6, no collapse); 2-layer delta {delta:+.1%} (honest)."
    )


if __name__ == "__main__":
    main()
