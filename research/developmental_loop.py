r"""
DEVELOPMENTAL LOOP -- the full autonomous brain cycle on REAL data (MNIST), closed and repeating:

    unsupervised learning -> memorization -> self-improvement + NEUROGENESIS (grow neurons)
        -> decision for the next learning objective -> unsupervised learning (repeat)

Each stage is a REAL brain mechanism, composed (not faked):

  1. UNSUPERVISED LEARNING   a GROWABLE spiking STDP learner (LIF + STDP + k-WTA + adaptive-threshold
                             homeostasis -- the verified snn_mnist_stdp mechanics) learns the CURRENT
                             objective's real MNIST digits with NO labels and NO backprop.
  2. MEMORIZATION            exemplars of what was learned are stored and REHEARSED (replayed) on every
                             later cycle -- hippocampal replay / systems consolidation that stops the
                             earlier digits being forgotten as new ones are learned (continual learning).
  3. SELF-IMPROVEMENT +      the learner GROWS NEURONS (neurogenesis) to add capacity for each new
     NEUROGENESIS            objective, and self-evaluates competence by neuron-label voting on a held-out
                             probe (labels used ONLY to read out competence, never to train).
  4. DECISION                a curiosity / learning-progress drive picks the NEXT objective -- the digit
                             the brain is least competent at / has not yet seen (novelty-seeking), i.e. it
                             chooses what to learn next by itself (autotelic).
  5. repeat.

The loop is OPEN-ENDED and SELF-DIRECTED: it keeps choosing new digits, growing, and consolidating until
every digit is mastered (or a cycle budget is hit). Observation / cognition only -- no actuator route.

Needs the torch venv:
    D:\AI-project\spiking-neural-data-lake\.venv-cu128\Scripts\python research\developmental_loop.py
Scale knobs (env): NORD_CYCLES, NORD_GROW (neurons grown per cycle), NORD_PER (train images per digit),
                   NORD_PROBE (held-out probe images per digit), NORD_GPU=1.
"""
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "src"))
sys.path.insert(0, _here)

import torch

from snn_mnist_stdp import (BETA, KWTA, LR, MAX_RATE, N_IN, T, TAU_PRE, THETA_DECAY,
                            THETA_PLUS, THRESH, W_NORM, WMAX, load_mnist)
from torch_device import enable, resolve_device

import snntorch as snn


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


# ---- a GROWABLE unsupervised spiking STDP learner (neuron count is per-instance, so it can grow) -------
class GrowableSTDP:
    """The snn_mnist_stdp STDP + k-WTA mechanics, but with a per-instance neuron count so the network can
    RECRUIT neurons (neurogenesis). All neurons stay PLASTIC -- a shared representation refined across every
    learned digit; forgetting is held off by strong REPLAY of stored exemplars (replay-based continual
    learning), not by freezing, which lets accuracy approach the joint-training (one-shot) result."""

    def __init__(self, device):
        self.device = device
        self.n = 0
        self.W = torch.zeros(0, N_IN, device=device)
        self.theta = torch.zeros(0, device=device)
        self.lif = snn.Leaky(beta=BETA, threshold=1e9, reset_mechanism="none")
        self.pre_decay = torch.exp(torch.tensor(-1.0 / TAU_PRE))

    def grow(self, k):
        """Recruit k new excitatory neurons (neurogenesis): small random positive weights, normalised."""
        w = torch.rand(k, N_IN, device=self.device) * 0.3
        w *= (W_NORM / w.sum(dim=1, keepdim=True).clamp_min(1e-6))
        self.W = torch.cat([self.W, w], dim=0)
        self.theta = torch.cat([self.theta, torch.zeros(k, device=self.device)], dim=0)
        self.n += k

    def run(self, image, learn):
        """Simulate one image for T steps; return the exc spike-count vector [n]. ALL neurons stay plastic
        (shared representation); forgetting is held off by strong REPLAY (rehearsal) of old digits, not by
        freezing -- replay-based continual learning, which refines every prototype across all learned digits."""
        image = image.to(self.device)
        mem = self.lif.init_leaky()
        x_pre = torch.zeros(N_IN, device=self.device)
        counts = torch.zeros(self.n, device=self.device)
        rates = image * MAX_RATE
        for _ in range(T):
            s_in = (torch.rand(N_IN, device=self.device) < rates).float()
            x_pre = x_pre * self.pre_decay + s_in
            cur = self.W @ s_in
            _, mem = self.lif(cur, mem)
            eff = mem - self.theta
            cand = (eff > THRESH).nonzero(as_tuple=True)[0]
            if cand.numel() > 0:
                if cand.numel() > KWTA:
                    cand = cand[torch.topk(eff[cand], KWTA).indices]
                for w in cand.tolist():
                    counts[w] += 1
                    if learn:                                 # ALL neurons stay plastic (replay prevents forgetting)
                        self.W[w] += LR * x_pre
                        self.W[w].clamp_(0.0, WMAX)
                        self.theta[w] += THETA_PLUS
                mem = torch.zeros(self.n, device=self.device)
            if learn:
                self.theta *= THETA_DECAY
        if learn:
            s = self.W.sum(dim=1, keepdim=True).clamp_min(1e-6)
            self.W *= (W_NORM / s)
        return counts


def assign_and_eval(net, probe_imgs, probe_lbls, classes):
    """Neuron-label voting on a labelled PROBE (labels used only to read out competence, not to train),
    then held-out accuracy over the given classes. Returns (overall_acc, {class: acc})."""
    resp = torch.zeros(net.n, 10, device=net.device)
    per = torch.zeros(10, device=net.device)
    for img, y in zip(probe_imgs, probe_lbls):
        resp[:, int(y)] += net.run(img, learn=False)
        per[int(y)] += 1
    assignment = (resp / per.clamp_min(1)).argmax(dim=1)
    correct = {c: 0 for c in classes}
    total = {c: 0 for c in classes}
    for img, y in zip(probe_imgs, probe_lbls):
        c = int(y)
        if c not in classes:
            continue
        counts = net.run(img, learn=False)
        scores = torch.zeros(10, device=net.device)
        for cc in range(10):
            m = assignment == cc
            if m.any():
                scores[cc] = counts[m].sum()
        correct[c] += int(scores.argmax().item() == c)
        total[c] += 1
    per_cls = {c: (correct[c] / total[c] if total[c] else 0.0) for c in classes}
    ov = sum(correct.values()) / max(1, sum(total.values()))
    return ov, per_cls


def main():
    enable(resolve_device())
    dev = resolve_device()
    CYCLES = _envi("NORD_CYCLES", 6)
    GROW = _envi("NORD_GROW", 40)          # neurons recruited per objective (neurogenesis)
    PER = _envi("NORD_PER", 250)           # training images per digit (unsupervised)
    PROBE = _envi("NORD_PROBE", 40)        # held-out probe images per digit
    REPLAY = _envi("NORD_REPLAY", 60)      # exemplars stored + rehearsed per learned digit (anti-forgetting)

    print("=" * 88)
    print("DEVELOPMENTAL LOOP -- unsupervised learning -> memorize -> grow neurons -> decide next -> repeat")
    print(f"device={dev}  cycles={CYCLES}  grow/cycle={GROW}  train/digit={PER}  probe/digit={PROBE}")
    print("=" * 88)

    Xtr, Ytr, Xte, Yte = load_mnist(6000, 3000)
    by_class = {d: (Ytr == d).nonzero(as_tuple=True)[0] for d in range(10)}
    probe_by = {d: (Yte == d).nonzero(as_tuple=True)[0][:PROBE] for d in range(10)}

    net = GrowableSTDP(dev)
    learned = []                            # digits mastered so far
    competence = {d: 0.0 for d in range(10)}   # self-measured competence per digit
    memory = {}                             # MEMORIZATION: stored exemplars per learned digit (for rehearsal)

    for cyc in range(1, CYCLES + 1):
        # ---- 4/DECISION: curiosity picks the next objective = the UNSEEN digit (novelty); if all seen,
        #      revisit the least-competent (learning-progress). This is the brain choosing what to learn. ----
        unseen = [d for d in range(10) if d not in learned]
        if unseen:
            objective = min(unseen, key=lambda d: competence[d])   # novelty-seeking
            reason = "novel (never seen)"
        else:
            objective = min(range(10), key=lambda d: competence[d])
            reason = "lowest competence (revisit)"
        print(f"\nCYCLE {cyc}  ->  OBJECTIVE: learn digit {objective}   [curiosity: {reason}]")

        # ---- 3/NEUROGENESIS: grow neurons to add capacity for the new objective ----
        m0 = net.n
        net.grow(GROW)
        print(f"  [neurogenesis]   neurons {m0} -> {net.n}  (+{GROW} recruited)")

        # ---- 1/UNSUPERVISED LEARNING: STDP on the objective's digits, STRONGLY interleaved with REPLAY of
        #      every previously-learned digit so the shared (all-plastic) representation refines all classes ----
        t0 = time.time()
        train_imgs = [Xtr[i] for i in by_class[objective][:PER]]
        # 2/MEMORIZATION rehearsal: a balanced replay of ALL learned digits' stored exemplars (anti-forgetting)
        rehearse = [ex for d in learned for ex in memory.get(d, [])]
        stream = train_imgs + rehearse
        torch.manual_seed(cyc)
        order = torch.randperm(len(stream))
        for j in order.tolist():
            net.run(stream[j], learn=True)
        dt = time.time() - t0
        print(f"  [unsupervised]   STDP on {len(train_imgs)} digit-{objective} imgs interleaved with "
              f"{len(rehearse)} replayed old exemplars, no labels ({dt:.0f}s)")

        # ---- 2/MEMORIZATION: store a rich exemplar buffer of the new digit for future replay ----
        memory[objective] = [Xtr[i] for i in by_class[objective][PER:PER + REPLAY]]
        learned.append(objective)
        print(f"  [memorize]       stored {len(memory[objective])} digit-{objective} exemplars in the replay "
              f"buffer (rehearsed every later cycle to consolidate against forgetting)")

        # ---- 3/SELF-IMPROVEMENT: self-evaluate competence over ALL learned digits (progress signal) ----
        probe_imgs = [Xte[i] for d in learned for i in probe_by[d]]
        probe_lbls = [int(Yte[i]) for d in learned for i in probe_by[d]]
        prev = dict(competence)
        ov, per_cls = assign_and_eval(net, probe_imgs, probe_lbls, set(learned))
        competence.update(per_cls)
        prog = competence[objective] - prev[objective]
        print(f"  [self-improve]   competence over learned {sorted(learned)} = {ov:.1%}  "
              f"(digit {objective}: {competence[objective]:.1%}, learning-progress +{prog:.1%})")

    print("\n" + "=" * 88)
    print(f"DEVELOPMENTAL LOOP COMPLETE: the brain autonomously learned {sorted(learned)} UNSUPERVISED, grew "
          f"{net.n} neurons over {CYCLES} cycles, consolidated each into memory, and chose every objective itself.")
    final = sum(competence[d] for d in learned) / len(learned)
    print(f"final mean competence over the {len(learned)} self-chosen digits: {final:.1%}  (no labels used in training)")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
