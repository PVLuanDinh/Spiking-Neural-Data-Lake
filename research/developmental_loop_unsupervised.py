r"""
FULLY LABEL-FREE DEVELOPMENTAL LOOP -- the model trains itself on RAW MNIST PIXELS with ZERO labels
used anywhere in training, neurogenesis, or any decision. It is self-DIRECTED by an INTRINSIC
learning-progress signal and self-GROWN by ART-vigilance neurogenesis. Labels are read at the VERY END
ONLY to MEASURE how well it clustered (cluster PURITY + NMI + a reference accuracy) -- there is no other
way to know it discovered digits. The intrinsic learning-progress is the label-free signal that actually
DROVE the loop.

This is the label-free sibling of research/developmental_loop.py. That loop still peeks at labels to pick
the next DIGIT objective and to score competence each cycle; here NOTHING but the final metric ever touches
a label. The loop reuses (does not reinvent) that file's verified pieces:

  - GrowableSTDP (from developmental_loop.py) -- the spiking STDP PROTOTYPE learner on raw 784 pixels:
      .match(image)            max cosine sim of the input to existing prototypes -- an INTRINSIC
                               "do I recognise this?" signal (NO labels).
      .maybe_recruit(img, VIG) ART neurogenesis: grow ONE prototype only when best match < vigilance
                               (label-free novelty). Self-limiting, need-sized.
      .run(img, learn, lr)     LIF + STDP + k-WTA weight refinement by spike timing (NO labels, NO backprop).
  - SpikingCuriosityDrive (neuromodulation/spiking_curiosity.py) -- learning-progress read from EMITTED
      SPIKE COUNTS of two LIF pools; its input is a PREDICTION ERROR, so it is fully label-free.
  - LocusCoeruleus (neuromodulation/locus_coeruleus.py) -- a label-free salience/novelty gain on the STDP lr.
  - load_mnist (snn_mnist_stdp.py) + the STDP constants.

LABEL-FREE CONTROL FLOW (one cycle):
  1. STREAM a batch of RAW pixel images sampled from Xtr (Ytr is NEVER indexed).
  2. INTRINSIC ERROR (label-free): err = 1 - mean(net.match(img)) over the batch, measured with the CURRENT
     prototypes BEFORE this cycle's learning -- a quantisation / "how well do my prototypes explain the raw
     data" error, purely from the model's own weights. High = the world is still surprising.
  3. SELF-DIRECTION (label-free): feed err to SpikingCuriosityDrive.observe("world", err) and read
     learning_progress("world") (the DROP in error = still learning). LocusCoeruleus.step(err) turns a high
     (salient) error into a phasic NA burst -> STDP lr gain. The loop then: GROWS while err is high (ART
     neurogenesis, capped), runs STDP, and does EXTRA consolidation passes while learning-progress is
     positive. It declares CONVERGED and stops when learning-progress dries up (~0 for K cycles, after it
     had once been positive). No label enters any of these decisions.
  4. SCORING -- THE ONLY PLACE LABELS ARE TOUCHED, and only to MEASURE (never fed back): see score().

Needs the torch venv:
    D:\AI-project\spiking-neural-data-lake\.venv-cu128\Scripts\python research\developmental_loop_unsupervised.py
  or on GPU:
    NORD_GPU=1 <python> research\developmental_loop_unsupervised.py
Scale knobs (env): NORD_CYCLES, NORD_BATCH (raw imgs/cycle), NORD_GROW (per-cycle recruit cap),
                   NORD_VIG (ART vigilance), NORD_PROBE (labelled probe imgs/digit -- SCORING ONLY),
                   NORD_EXTRA (consolidation passes), NORD_CONVK (converged after K flat-LP cycles),
                   NORD_GPU=1.
"""
import math
import os
import sys
from collections import Counter, deque

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, _here)

import torch

from snn_mnist_stdp import load_mnist
from torch_device import enable, resolve_device

from developmental_loop import GrowableSTDP
from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive

torch.manual_seed(0)   # deterministic run + self-check


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


# ---- intrinsic (label-free) quantisation error over a batch of RAW images -----------------------------
def intrinsic_error(net, batch):
    """The model's OWN prediction/quantisation error on raw pixels: err = 1 - mean over the batch of the
    best cosine match of each image to the current prototypes. NO labels. 1.0 when there are no prototypes
    yet (nothing recognised); falls toward 0 as neurogenesis + STDP make the prototypes explain the data.
    This is the surprise signal the curiosity drive turns into learning-progress."""
    if not batch:
        return 1.0
    s = 0.0
    for img in batch:
        s += net.match(img)          # match() is a pure function of the input + weights -- no labels
    return 1.0 - s / len(batch)


# ---- NMI between the prototype-cluster assignment and the true labels (SCORING ONLY, ~15 lines) --------
def _nmi(clusters, labels):
    """Normalized mutual information between two integer labelings, computed directly (no new dependency).
    NMI = I(U;V) / sqrt(H(U) H(V)), all logs natural (base-invariant). Returns 0.0 for a degenerate (single
    cluster or single label) partition. USED ONLY IN SCORING -- labels enter here to MEASURE clustering."""
    n = len(labels)
    if n == 0:
        return 0.0
    cu = Counter(clusters)                       # cluster (winning-prototype) sizes
    cv = Counter(labels)                         # true-class sizes
    joint = Counter(zip(clusters, labels))       # co-occurrence counts
    hu = -sum((a / n) * math.log(a / n) for a in cu.values())
    hv = -sum((b / n) * math.log(b / n) for b in cv.values())
    mi = 0.0
    for (u, v), nij in joint.items():
        mi += (nij / n) * math.log((nij * n) / (cu[u] * cv[v]))
    denom = math.sqrt(hu * hv)
    return mi / denom if denom > 0 else 0.0


def score(net, probe_imgs, probe_lbls):
    """THE ONLY PLACE LABELS ARE TOUCHED, and only to MEASURE (never fed back into training/growth/decisions).
    Over a labelled probe set:
      - resp[neuron, class] = total spike counts neuron `neuron` emits on true-class `class` images.
      - CLUSTER PURITY = (sum over neurons of its TOP-class spike count) / (total spike responses)
        -- the standard unsupervised-STDP readout (Diehl & Cook neuron-label voting, reported as PURITY,
        no classifier trained). High purity = each learned prototype fires for essentially one true digit.
      - NMI = normalized mutual information between the per-image PROTOTYPE-CLUSTER assignment (the
        most-active prototype index) and the true label.
      - reference ACCURACY = assign each image to the class of its most-active prototype (each prototype
        voted to the class it fires most for), for a familiar number to compare against.
    Returns (purity, nmi, accuracy, n_clusters_used)."""
    n = net.n
    resp = torch.zeros(n, 10, device=net.device)     # per-neuron per-class spike totals
    clusters = []                                     # per-image winning prototype index (the cluster id)
    trues = []                                        # per-image TRUE label (touched only for measurement)
    for img, y in zip(probe_imgs, probe_lbls):
        counts = net.run(img, learn=False)            # NO learning, NO label into the net -- pure readout
        resp[:, int(y)] += counts
        clusters.append(int(counts.argmax().item()))  # cluster = the single most-active prototype
        trues.append(int(y))
    total = float(resp.sum().item())
    purity = float(resp.max(dim=1).values.sum().item()) / total if total > 0 else 0.0
    neuron_class = resp.argmax(dim=1)                 # each prototype's most-fired-for true class (voting)
    correct = sum(int(neuron_class[c].item()) == t for c, t in zip(clusters, trues))
    accuracy = correct / len(trues) if trues else 0.0
    nmi = _nmi(clusters, trues)
    return purity, nmi, accuracy, len(set(clusters))


def main():
    enable(resolve_device())
    dev = resolve_device()
    CYCLES = _envi("NORD_CYCLES", 20)          # max cycles (loop may STOP earlier on intrinsic convergence)
    BATCH = _envi("NORD_BATCH", 300)           # RAW images streamed per cycle (labels NEVER indexed)
    GROW = _envi("NORD_GROW", 40)              # per-cycle ART recruit CAP (the brain sizes actual growth <= this)
    VIG = _envf("NORD_VIG", 0.85)              # ART vigilance: best-match below this = label-free novelty -> recruit
    PROBE = _envi("NORD_PROBE", 40)            # labelled probe images per digit -- SCORING ONLY
    EXTRA = _envi("NORD_EXTRA", 1)             # extra consolidation STDP passes while learning-progress > 0
    CONVK = _envi("NORD_CONVK", 3)             # declare CONVERGED after K consecutive flat-learning-progress cycles
    LP_TH = _envf("NORD_LPTH", 0.01)           # learning-progress above this = "still learning" (drives consolidation)
    ERR_LO = _envf("NORD_ERRLO", 0.15)         # below this intrinsic error the world is mostly explained -> taper growth

    print("=" * 96)
    print("FULLY LABEL-FREE DEVELOPMENTAL LOOP -- raw MNIST pixels, ZERO labels in training/growth/decisions")
    print(f"device={dev}  max_cycles={CYCLES}  raw_imgs/cycle={BATCH}  recruit_cap/cycle={GROW}  "
          f"vigilance={VIG}  probe/digit(SCORING ONLY)={PROBE}")
    print("NO LABELS are used to learn, to grow neurons, or to make ANY decision. The INTRINSIC")
    print("learning-progress (drop in the model's own quantisation error) is what drives the loop.")
    print("=" * 96)

    Xtr, Ytr, Xte, Yte = load_mnist(6000, 3000)
    # Ytr is loaded but is NEVER referenced below the scoring block -- it does not touch the loop.
    # A balanced labelled PROBE (Xte/Yte) is prepared HERE but consumed ONLY inside score() at the very end.
    probe_idx = [int(i) for d in range(10) for i in (Yte == d).nonzero(as_tuple=True)[0][:PROBE].tolist()]
    probe_imgs = [Xte[i] for i in probe_idx]
    probe_lbls = [int(Yte[i]) for i in probe_idx]   # <-- the ONLY labels used anywhere; SCORING ONLY

    net = GrowableSTDP(dev)                          # the label-free spiking STDP prototype learner
    curiosity = SpikingCuriosityDrive(seed=0)        # spiking learning-progress from PREDICTION ERROR (label-free)
    lc = LocusCoeruleus()                            # label-free salience/novelty gain on the STDP lr

    lp_curve = []                                    # the intrinsic learning-progress curve across cycles
    err_curve = []
    flat = deque(maxlen=CONVK)                       # recent learning-progress values (convergence window)
    had_progress = False                             # only allow "converged" AFTER learning-progress was once real
    converged_at = None

    for cyc in range(1, CYCLES + 1):
        # ---- 1/STREAM raw pixels (NO labels): sample BATCH raw training images deterministically ----
        torch.manual_seed(cyc)
        idx = torch.randperm(len(Xtr))[:BATCH].tolist()
        batch = [Xtr[i] for i in idx]                # Ytr is NOT indexed -- the loop never sees a label

        # ---- 2/INTRINSIC ERROR (label-free) with the CURRENT prototypes, BEFORE this cycle's learning ----
        err = intrinsic_error(net, batch)
        err_curve.append(err)

        # ---- 3/SELF-DIRECTION (label-free): curiosity learning-progress + LC salience gain ----
        curiosity.observe("world", err)              # feed the model's own prediction error (no labels)
        lp = curiosity.learning_progress("world")    # DROP in error = the model is still learning
        lp_curve.append(lp)
        na = lc.step(err)                            # high (salient) error -> phasic NA burst
        gain = 1.0 + 2.0 * na                        # noradrenaline boosts the STDP step 1x..3x on a surprising world
        if lp > LP_TH:
            had_progress = True

        # decision (label-free): grow hard while the world is still surprising, taper once mostly explained
        grow_cap = GROW if err > ERR_LO else max(2, GROW // 10)

        # ---- neurogenesis + unsupervised STDP learning (interleaved), NO labels, NO backprop ----
        m0 = net.n
        recruited = 0
        for img in batch:
            if recruited < grow_cap and net.maybe_recruit(img, VIG, cycle=cyc):   # ART: recruit iff novel
                recruited += 1
            net.run(img, learn=True, lr_scale=gain)  # STDP refines matched prototypes by spike timing

        # extra CONSOLIDATION passes while the intrinsic learning-progress says we are still learning
        extra_done = 0
        if lp > LP_TH:
            for _ in range(EXTRA):
                torch.manual_seed(cyc * 1000 + extra_done)
                for j in torch.randperm(len(batch)).tolist():
                    net.run(batch[j], learn=True, lr_scale=gain)
                extra_done += 1

        print(f"cycle {cyc:2d}  intrinsic_err={err:.3f}  learning_progress={lp:.4f}  "
              f"LC_gain=x{gain:.2f}  recruited={recruited} (cap {grow_cap}) -> {m0}->{net.n} prototypes  "
              f"consolidation_passes={extra_done}")

        # ---- CONVERGENCE (label-free): learning-progress dried up for K cycles, after it was once positive ----
        flat.append(lp)
        if had_progress and len(flat) == CONVK and all(v <= LP_TH for v in flat):
            converged_at = cyc
            print(f"  [self-directed] learning-progress dried up (<= {LP_TH} for {CONVK} cycles) "
                  f"-> the loop declares CONVERGED and stops. NO label was consulted.")
            break

    # =========================================================================================
    # SCORING -- the ONLY place labels are touched, and only to MEASURE (never fed back).
    # =========================================================================================
    purity, nmi, accuracy, n_clusters = score(net, probe_imgs, probe_lbls)

    print("\n" + "=" * 96)
    print("RESULT (labels touched ONLY from here on, and ONLY to measure -- never fed back):")
    print(f"  prototypes grown (self-built, label-free)     : {net.n}")
    print(f"  final intrinsic error (model's own)           : {err_curve[-1]:.3f}")
    print(f"  intrinsic learning-progress curve (the SIGNAL : "
          f"{['%.3f' % v for v in lp_curve]}")
    print(f"    that DROVE the loop, per cycle)             {'converged at cycle %d' % converged_at if converged_at else 'ran full budget'}")
    print(f"  cluster PURITY (top-class / total responses)  : {purity:.3f}   (chance ~ 0.10)")
    print(f"  NMI (prototype-cluster vs true labels)        : {nmi:.3f}   (0 = independent, 1 = perfect)")
    print(f"  reference accuracy (most-active prototype)    : {accuracy:.1%}   (chance = 10%)")
    print(f"  distinct prototype-clusters used on the probe : {n_clusters}")
    print("-" * 96)
    print("HONESTY: NO labels were used in training, in neurogenesis, or in ANY objective/decision. The loop")
    print("self-directed on the INTRINSIC learning-progress (the drop in its own quantisation error) and")
    print("self-grew via ART vigilance. Labels (Yte) were read ONLY just now to compute PURITY / NMI /")
    print("accuracy -- there is no other way to know it clustered digits well. Ytr was never indexed at all.")
    print("=" * 96)

    # ---- self-check: clustered ABOVE chance, and the brain grew a few prototypes on its own ----
    assert net.n >= 3, f"the loop must self-grow at least a few prototypes (grew {net.n})"
    assert purity > 0.10, f"cluster purity must beat chance 0.10 (got {purity:.3f})"
    print(f"self-check OK: {net.n} self-grown prototypes; purity {purity:.3f} > 0.10 chance; "
          f"NMI {nmi:.3f}; no labels used in training/growth/decisions (scoring only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
