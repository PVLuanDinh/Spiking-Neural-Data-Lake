"""
Deep vs single-layer UNSUPERVISED STDP on MNIST (Torch/GPU tier, Brian2 LIF equation).

The honest question: does a DEEP unsupervised spiking feature hierarchy
(conv-STDP -> pool -> prototype-STDP) beat a SINGLE unsupervised prototype
layer on raw pixels? Both learners are trained with the SAME recipe, on the
SAME MNIST subset + seed, and scored the SAME way, so the delta is a fair
head-to-head. A NEGATIVE delta is a valid result and is reported as-is (the
self-check never asserts 2-layer > 1-layer).

UNSUPERVISED end to end: NO labels and NO backprop are used to write any
weight. Every synapse (conv kernels AND prototype weights) is set purely by
spike-timing correlation (STDP) with winner-take-all competition and adaptive-
threshold homeostasis. Labels touch the pipeline ONLY at the very end, in the
neuron-label vote that reads out accuracy (each output neuron is assigned the
class it fires most for) -- exactly the readout used by snn_mnist_stdp.

Greedy layer-wise design (Kheradpisheh 2018 style, simplified):
  1. CONV-STDP layer 1 : C shared 5x5 kernels slide over the 28x28 rate-coded
     Poisson input spikes. Each timestep -> conv currents -> leaky membrane ->
     per-unit threshold spikes. Once per image, each map picks its single most
     active (most-spiking) above-threshold location and does ONE STDP step,
     pulling that map's SHARED kernel toward the presynaptic input-trace patch
     under the winning location. Per-position lateral inhibition lets at most
     one map claim a given location; per-map adaptive-threshold homeostasis
     (theta bump) forces the maps to specialise on different local features.
     Weights are shared across space -> translation-invariant edge/stroke
     detectors. Kernels are sum-normalised (homeostasis) after every image.
  2. FEATURES : freeze layer 1, forward each image -> per-map spike-count map ->
     spatial MAX-POOL (to ~6x6) -> flatten -> rescale to [0,1] intensities.
  3. PROTO-STDP layer 2 : a fully-connected STDP prototype learner (M neurons,
     hard k-WTA=1 + adaptive threshold) over the layer-1 feature vectors -- the
     identical mechanics as the single-layer baseline, just on features instead
     of pixels.

Honest simplification vs full Kheradpisheh 2018: we use RATE coding (Poisson
spikes) and a most-spikes winner, not temporal LATENCY coding with a
first-spike / rank-order winner; competition is a single greedy winner per map
per image rather than a full spatial lateral-inhibition field; and the layer-1
threshold is a proportionally-scaled reuse of the flat-layer THRESH rather than
a per-map STDP-competition threshold. Expect modest accuracy on a small subset,
not paper-level numbers.

MEASURED RESULT (RTX 5070, MNIST 6000/2000, C=16, M=100 -- honest NEGATIVE):
  1-layer FC-STDP on pixels      63.6%
  2-layer conv-STDP -> FC-STDP    8.8%  (= chance; delta -54.9%, "DEEP HURTS")
The simplified conv layer COLLAPSES: its 16 maps converge to near-identical
filters (pairwise cosine similarity ~0.96) so the pooled features carry no class
information (threshold-independent, so it is structural map-collapse, not firing
calibration). This is the well-known weakness of naive rate-coded single-winner
conv-STDP; the fix is the fuller machinery named above (DoG + latency coding +
first-spike winner + a diversity-forcing inter-map inhibition field), a real
research build, not a knob. Kept as a documented negative + a clean baseline
scaffold for that future work. The developmental loop keeps its single-layer
STDP learner (accuracy lever = neuron SCALE, ~88% -> ~95% at 6400).

Run (defaults finish a smoke run in a few minutes on GPU):
    NORD_GPU=1 PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.instruments.deep_stdp
    (command corrected 2026-08-11 at promotion: it named the retired .venv-cu128 env and the
    pre-promotion path)
Scale knobs: NORD_C NORD_M NORD_TRAIN NORD_TEST NORD_GPU NORD_CTHRESH
Deps: numpy torch torchvision brian2   ([train]/[gpu] extras. NOT "outside the core": this module lives in src/snn_data_lake/reasoning/
         and imports from the package -- P4c, every build is part of the brain.)
"""

import os

import torch
import torch.nn.functional as F

from snn_data_lake.instruments.snn_mnist_stdp import (
    BETA,
    LR,
    MAX_RATE,
    N_IN,
    TAU_PRE,
    THETA_DECAY,
    THETA_PLUS,
    THRESH,
    W_NORM,
    WMAX,
    T,
    load_mnist,
)
from snn_data_lake.substrate.brian2_backend import BrianLIF
from snn_data_lake.substrate.device import enable, resolve_device

K = 5  # conv kernel size (single input channel, valid conv, stride 1)

# _env/_envf were a line-for-line second copy of the hub's own helpers until the 2026-08-11
# promotion (the P2-shape duplication its ledger row flagged); imported from the one home now.
from snn_data_lake.instruments.snn_mnist_stdp import _env, _envf  # noqa: E402

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "ConvStdp.train_image": (
        "LOCAL",
        "unsupervised WTA-STDP: each map's most-active above-threshold location wins and its weights "
        "move by spike timing. Unsupervised and local",
    ),
    "OneLayerNet.train_image": (
        "LOCAL",
        "drives the STDP layer over one image; inherits the claim of the rule it applies",
    ),
    "TwoLayerNet.train_image": (
        "LOCAL",
        "drives the stacked STDP layers over one image; each layer learns by its own local rule, "
        "never by an error passed between them",
    ),
    "_train": (
        "LOCAL",
        "the harness that runs train_image across the set; carries the same claim as the rule it "
        "drives",
    ),
    "FcStdp.run": (
        "LOCAL",
        "rate-codes the input and simulates T steps, applying STDP on the resulting spikes",
    ),
    "FcStdp._normalise": (
        "LOCAL",
        "synaptic scaling to a fixed incoming budget -- homeostatic, computed per neuron from its own "
        "weights",
    ),
}


class FcStdp:
    """Fully-connected STDP prototype layer (mirrors snn_mnist_stdp.StdpNetwork,
    parameterised by input dim so the SAME mechanics serve pixels or features).
    Input(n_in Poisson) -> Prototypes(m, Brian2-equation LIF) with STDP + hard k-WTA=1."""

    def __init__(self, n_in, m):
        self.n_in = n_in
        self.m = m
        self.W = torch.rand(m, n_in) * 0.3  # lands on default device (GPU-safe)
        self._normalise()
        # BrianLIF with unreachable threshold: pure leaky integration, then our
        # own adaptive-threshold + WTA (same trick as snn_mnist_stdp).
        self.lif = BrianLIF(beta=BETA, threshold=1e9, reset_mechanism="none")
        self.pre_decay = torch.exp(torch.tensor(-1.0 / TAU_PRE))
        self.theta = torch.zeros(m)  # per-neuron adaptive threshold

    def _normalise(self):
        s = self.W.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.W *= W_NORM / s

    def run(self, vec, learn):
        """Rate-code the [n_in] intensity vector and simulate T steps. Returns the
        prototype spike-count vector [m]. STDP + k-WTA=1 only when learn=True."""
        vec = vec.to(self.W.device).reshape(-1)  # GPU-safe: match network device
        mem = self.lif.init_leaky()
        x_pre = torch.zeros(self.n_in)
        counts = torch.zeros(self.m)
        rates = vec * MAX_RATE
        for _ in range(T):
            s_in = (torch.rand(self.n_in) < rates).float()  # Poisson input spikes
            x_pre = x_pre * self.pre_decay + s_in  # pre-synaptic trace
            cur = self.W @ s_in
            _, mem = self.lif(cur, mem)  # leaky integration
            eff = mem - self.theta
            cand = (eff > THRESH).nonzero(as_tuple=True)[0]
            if cand.numel() > 0:
                wi = int(cand[torch.topk(eff[cand], 1).indices].item())  # k-WTA=1
                counts[wi] += 1
                if learn:
                    self.W[wi] += LR * x_pre  # STDP potentiation
                    self.W[wi].clamp_(0.0, WMAX)
                    self.theta[wi] += THETA_PLUS  # raise its own bar
                mem = torch.zeros(self.m)  # lateral inhibition reset
            if learn:
                self.theta *= THETA_DECAY
        if learn:
            self._normalise()
        return counts


class ConvStdp:
    """Layer 1: C shared 5x5 conv-STDP feature maps over the Poisson input spikes.
    Unsupervised WTA-STDP, one update per map per image toward the winning patch."""

    def __init__(self, C, k=K, cthresh=None):
        self.C = C
        self.k = k
        self.hout = 28 - k + 1  # valid conv, stride 1 -> 24 for k=5
        self.cthresh = THRESH if cthresh is None else cthresh
        # Fan-in-proportional homeostasis: scale the kernel-sum target by the
        # patch/full-image input-count ratio so each conv weight has ~the same
        # magnitude as a flat-layer weight -> the reused THRESH stays meaningful.
        self.cwnorm = W_NORM * (k * k) / N_IN
        self.Wc = torch.rand(C, k * k) * 0.3  # shared kernels, one row per map
        self._normalise()
        self.lif = BrianLIF(beta=BETA, threshold=1e9, reset_mechanism="none")
        self.pre_decay = torch.exp(torch.tensor(-1.0 / TAU_PRE))
        self.theta = torch.zeros(C)  # per-map adaptive threshold
        self.pool = max(1, self.hout // 6)  # max-pool to ~6x6 (keeps layer-2
        #                                          FC in its calibrated fan-in regime)

    def _normalise(self):
        s = self.Wc.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.Wc *= self.cwnorm / s

    def _forward(self, image):
        """Simulate one image for T steps. Returns (spike-count map [C,hout,hout],
        final presynaptic input trace [28,28])."""
        img = image.to(self.Wc.device).reshape(28, 28)  # GPU-safe
        weight = self.Wc.view(self.C, 1, self.k, self.k)  # [C,1,k,k] for conv2d
        mem = self.lif.init_leaky()
        x_pre = torch.zeros(28, 28)
        cnt = torch.zeros(self.C, self.hout, self.hout)
        rates = img * MAX_RATE
        for _ in range(T):
            s_in = (torch.rand(28, 28) < rates).float()  # Poisson input spikes
            x_pre = x_pre * self.pre_decay + s_in  # pre-synaptic trace
            cur = F.conv2d(s_in.view(1, 1, 28, 28), weight)[0]  # [C,hout,hout]
            _, mem = self.lif(cur, mem)  # leaky integration
            eff = mem - self.theta.view(self.C, 1, 1)
            sp = eff > self.cthresh
            cnt += sp.to(cnt.dtype)
            mem = mem * (~sp).to(mem.dtype)  # soft-reset spiked units
        return cnt, x_pre

    def train_image(self, image):
        """Unsupervised WTA-STDP: each map's most-active above-threshold location
        wins (per-position lateral inhibition), one STDP step toward its input
        patch, adaptive-threshold homeostasis, kernel renormalisation."""
        cnt, x_pre = self._forward(image)
        flat = cnt.view(self.C, -1)
        best_val, best_idx = flat.max(dim=1)  # per map: its strongest location
        order = torch.argsort(best_val, descending=True)
        taken = set()
        for m in order.tolist():
            if best_val[m].item() <= 0:  # never crossed threshold -> skip
                continue
            pos = int(best_idx[m].item())
            if pos in taken:  # at most one map wins a position
                continue
            taken.add(pos)
            y, x = divmod(pos, self.hout)  # output loc -> input patch top-left
            patch = x_pre[y : y + self.k, x : x + self.k].reshape(-1)
            self.Wc[m] += LR * patch  # STDP toward the local input trace
            self.Wc[m].clamp_(0.0, WMAX)
            self.theta[m] += THETA_PLUS  # this map raises its own bar
        self.theta *= THETA_DECAY
        self._normalise()

    def features(self, image):
        """Frozen forward -> spike-count map -> spatial max-pool -> flatten ->
        rescale to [0,1] intensities (the layer-2 FC input vector)."""
        cnt, _ = self._forward(image)
        pooled = F.max_pool2d(cnt.view(1, self.C, self.hout, self.hout), self.pool)
        feat = pooled.reshape(-1)
        return feat / feat.max().clamp_min(1e-6)

    def feat_dim(self):
        ph = self.hout // self.pool
        return self.C * ph * ph


class OneLayerNet:
    """Baseline: single FC-STDP prototype layer directly on the 784 raw pixels."""

    def __init__(self, m):
        self.fc = FcStdp(N_IN, m)

    def train_image(self, image):
        self.fc.run(image, learn=True)

    def run(self, image, learn=False):
        return self.fc.run(image, learn)


class TwoLayerNet:
    """Deep: frozen conv-STDP layer 1 -> pooled features -> FC-STDP prototypes."""

    def __init__(self, conv, m):
        self.conv = conv  # already trained + frozen
        self.fc = FcStdp(conv.feat_dim(), m)

    def train_image(self, image):
        self.fc.run(self.conv.features(image), learn=True)

    def run(self, image, learn=False):
        return self.fc.run(self.conv.features(image), learn)


def assign_labels(net, X, Y, m):
    """Neuron-label voting (same as snn_mnist_stdp): after unsupervised training,
    label each output neuron by the class it responds to most strongly."""
    resp = torch.zeros(m, 10)
    per_class = torch.zeros(10)
    for img, y in zip(X, Y):
        resp[:, int(y)] += net.run(img, learn=False)
        per_class[int(y)] += 1
    resp /= per_class.clamp_min(1)
    return resp.argmax(dim=1)  # neuron -> assigned class


def evaluate(net, X, Y, assignment, m):
    correct = 0
    for img, y in zip(X, Y):
        counts = net.run(img, learn=False)
        scores = torch.zeros(10)
        for c in range(10):
            mask = assignment == c
            if mask.any():
                scores[c] = counts[mask].sum()  # predicted = best-voting class
        if int(scores.argmax().item()) == int(y):
            correct += 1
    return correct / len(Y)


def _train(net, X, tag):
    for i, img in enumerate(X):
        net.train_image(img)
        if (i + 1) % 2000 == 0:
            print(f"  {tag}: {i + 1}/{len(X)} images")


def _mechanics_check():
    """The GATE-TIER integrity check (genn precedent: the bare __main__ never runs the 8-minute
    benchmark; DEEPSTDP_RUN=1 opts into it). Asserts the lineage LIBRARY's mechanics on a small
    REAL-MNIST slice and claims NOTHING about accuracy -- the benchmark's measured claims (the
    1-layer >=55% bar and the ConvStdp collapse negative) belong exclusively to the full run.
    Reachable reds: neuter the STDP update and the weights-moved assert fails; drop _normalise
    and the row-sum assert fails; raise THRESH to 1e9 and the fires assert fails."""
    torch.manual_seed(0)
    X, Y, Xte, Yte = load_mnist(64, 32)
    net = OneLayerNet(16)
    w0 = net.fc.W.clone()
    for img in X:
        net.run(img, learn=True)
    assert not torch.equal(w0, net.fc.W), "STDP moved no weight (learning neutered)"
    s = net.fc.W.sum(dim=1)
    assert torch.allclose(s, torch.full_like(s, W_NORM), rtol=1e-3), (
        f"homeostatic normalise broken (row sums {s.min():.2f}..{s.max():.2f} vs {W_NORM})"
    )
    counts = net.run(X[0], learn=False)
    assert counts.sum() > 0, "no prototype fires on a trained image"
    a = assign_labels(net, X, Y, 16)
    acc = evaluate(net, Xte, Yte, a, 16)
    assert 0.0 <= acc <= 1.0, f"evaluate returned a non-probability ({acc})"
    print(
        f"mechanics OK: STDP moves weights, normalise holds ({W_NORM}), k-WTA fires, "
        f"assign/evaluate round-trip (slice acc {acc:.2f} -- NOT the benchmark; "
        f"DEEPSTDP_RUN=1 runs the full 6000/2000 measurement, ~8 min GPU)"
    )
    return 0


def main():
    if os.environ.get("DEEPSTDP_RUN") != "1":
        return _mechanics_check()
    enable(resolve_device())
    C = _env("NORD_C", 16)
    M = _env("NORD_M", 100)
    TRAIN_N = _env("NORD_TRAIN", 6000)
    TEST_N = _env("NORD_TEST", 2000)
    CTHRESH = _envf("NORD_CTHRESH", THRESH)

    print("=" * 62)
    print("DEEP vs SINGLE-LAYER UNSUPERVISED STDP  (MNIST, Brian2 equation)")
    print("=" * 62)
    print(f"C={C}  M={M}  T={T}  k={K}  train={TRAIN_N}  test={TEST_N}")
    print(f"conv-thresh={CTHRESH}  (NO labels, NO backprop; labels only in vote)\n")
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

    # ---------- 2-layer deep: conv-STDP -> pool -> FC-STDP ----------
    torch.manual_seed(0)
    print("\n[2-layer] training conv-STDP layer 1 (unsupervised)...")
    conv = ConvStdp(C, cthresh=CTHRESH)
    _train(conv, Xtr, "2-layer L1")
    print(f"[2-layer] layer-1 frozen; feature dim = {conv.feat_dim()}")
    print("[2-layer] training FC-STDP layer 2 on pooled features...")
    net2 = TwoLayerNet(conv, M)
    _train(net2, Xtr, "2-layer L2")
    print("[2-layer] neuron-label vote + evaluate...")
    a2 = assign_labels(net2, Xtr[:n_assign], Ytr[:n_assign], M)
    acc2 = evaluate(net2, Xte, Yte, a2, M)

    # ---------- honest head-to-head ----------
    delta = acc2 - acc1
    print("\n" + "=" * 62)
    print("RESULT  (unsupervised STDP, neuron-label-vote readout; chance = 10%)")
    print("-" * 62)
    print(f"  1-layer  (FC-STDP on 784 pixels)          : {acc1:.1%}")
    print(f"  2-layer  (conv-STDP -> pool -> FC-STDP)    : {acc2:.1%}")
    print(f"  delta    (2-layer minus 1-layer)          : {delta:+.1%}")
    verdict = "DEEP HELPS" if delta > 0 else ("TIE" if delta == 0 else "DEEP HURTS")
    print(f"  verdict                                    : {verdict}")
    print("=" * 62)

    # self-check: the 1-layer baseline must beat chance (guards the shared recipe).
    # We deliberately DO NOT assert anything about 2-layer -- the MEASURED honest
    # result is that this naive rate-coded, single-greedy-winner conv-STDP COLLAPSES
    # to chance (~8.8%): the 16 maps converge to near-identical filters (pairwise
    # cos-sim ~0.96) and the pooled features are non-discriminative (all-zero at the
    # reused THRESH; identical-and-uninformative when the conv threshold is lowered).
    # Threshold-independent -> structural, not calibration. Making deep unsupervised
    # conv-STDP actually help needs the FULL Kheradpisheh machinery (DoG preprocessing
    # + latency/rank-order coding + first-spike winner + an inter-map lateral-inhibition
    # field that FORCES feature diversity), which this simplified build omits on purpose.
    assert acc1 >= 0.5, f"1-layer baseline failed to beat chance: acc={acc1:.2f}"
    if acc2 < 0.2:
        print("self-check OK (documented NEGATIVE): naive conv-STDP layer collapses to")
        print("chance -- map diversity + latency coding are the missing machinery.")
    else:
        print(f"self-check OK: 1-layer sane; 2-layer acc={acc2:.1%}, delta {delta:+.1%} (honest)")


if __name__ == "__main__":
    main()
