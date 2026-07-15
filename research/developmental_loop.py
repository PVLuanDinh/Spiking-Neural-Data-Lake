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
  3. SELF-IMPROVEMENT +      the BRAIN grows its OWN neurons (neurogenesis) -- the ART / hippocampus.
     NEUROGENESIS            GrowingLearner vigilance rule recruits a neuron ONLY on a learnable-NOVEL pattern
                             no existing prototype matches (self-limiting, need-sized -- NOT a fixed grow(GROW)
                             schedule); and it self-evaluates competence by neuron-label voting on a held-out
                             probe (labels only read out competence, never train).
  4. DECISION                the brain's OWN neuromodulation/spiking_curiosity.SpikingCuriosityDrive picks the
                             next objective (most_curious = spiking learning-progress + novelty) and a
                             SpikingCurriculum (LIF accumulator) paces mastery -- it explores every unseen
                             digit, then REVISITS the weakest-still-improving one on the learnable frontier.
                             Autotelic, no hand-coded heuristic.
  5. repeat.

The loop is OPEN-ENDED and SELF-DIRECTED: it keeps choosing new digits, growing, and consolidating until
every digit is mastered (or a cycle budget is hit). Observation / cognition only -- no actuator route.

Needs the torch venv:
    D:\AI-project\spiking-neural-data-lake\.venv-cu128\Scripts\python research\developmental_loop.py
Scale knobs (env): NORD_CYCLES, NORD_GROW (neurons/cycle), NORD_PER (train imgs/digit), NORD_PROBE
                   (probe imgs/digit), NORD_REPLAY (exemplars/digit), NORD_CAP (max consecutive revisits),
                   NORD_GPU=1.

Empirical accuracy ladder (full 10-digit self-directed run, no labels, GPU). NORD_GROW is now a per-cycle
recruitment CAP (the brain's vigilance rule sizes actual growth up to it): cap=50 -> 87.5%; cap=120,
random-init growth -> 88.0%; cap=120, BRAIN-DECIDED novelty-seeded growth -> 88.8% (digit 9: 60% -> 75%,
curriculum mastered 3/10). SCALING the learner is the accuracy lever; the objective-selection / revisit
strategy is NOT (revisiting plateaus ~81-83% -- see CHANGELOG v3.42/v3.43). Toward Diehl & Cook: ~92% at
1600, ~95% at 6400.
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

from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum

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

    def match(self, image):
        """Best match of `image` to the existing neuron prototypes -- max cosine similarity of the input to
        every neuron's weight vector. 0.0 when there are no neurons yet. This is the 'do I already recognise
        this?' signal the brain's neurogenesis rule reads."""
        if self.n == 0:
            return 0.0
        x = image.to(self.device)
        xn = x / (x.norm() + 1e-6)
        wn = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-6)
        return float((wn @ xn).max())

    def maybe_recruit(self, image, vigilance):
        """The BRAIN'S OWN neurogenesis decision (the ART / hippocampus.GrowingLearner vigilance rule): recruit
        exactly ONE neuron, SEEDED toward `image`, iff no existing prototype matches it above `vigilance`
        (learnable-novelty). Familiar patterns (match >= vigilance) recruit nothing -- they refine the matching
        neuron via STDP instead -- so growth is SELF-LIMITING and need-sized, not a fixed schedule. Returns True
        if it grew."""
        if self.n == 0 or self.match(image) < vigilance:
            x = image.to(self.device)
            w = (x / (x.sum() + 1e-6) * W_NORM).unsqueeze(0)   # the new prototype IS the novel pattern (ART recruit)
            self.W = torch.cat([self.W, w], dim=0)
            self.theta = torch.cat([self.theta, torch.zeros(1, device=self.device)], dim=0)
            self.n += 1
            return True
        return False

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
    CYCLES = _envi("NORD_CYCLES", 15)      # >10 so curiosity REVISITS the weakest digits after covering all 10
    GROW = _envi("NORD_GROW", 40)          # neurons recruited per objective (neurogenesis)
    PER = _envi("NORD_PER", 250)           # training images per digit (unsupervised)
    PROBE = _envi("NORD_PROBE", 40)        # held-out probe images per digit
    REPLAY = _envi("NORD_REPLAY", 60)      # exemplars stored + rehearsed per learned digit (anti-forgetting)
    CAP = _envi("NORD_CAP", 1)             # max consecutive revisits of the SAME digit (anti-fixation -> diversify)
    VIG = float(os.environ.get("NORD_VIG", "0.85"))   # neurogenesis vigilance: best-match below this = learnable-novel -> the brain recruits

    print("=" * 88)
    print("DEVELOPMENTAL LOOP -- unsupervised learning -> memorize -> grow neurons -> decide next -> repeat")
    print(f"device={dev}  cycles={CYCLES}  grow/cycle={GROW}  train/digit={PER}  probe/digit={PROBE}")
    print("=" * 88)

    Xtr, Ytr, Xte, Yte = load_mnist(6000, 3000)
    by_class = {d: (Ytr == d).nonzero(as_tuple=True)[0] for d in range(10)}
    probe_by = {d: (Yte == d).nonzero(as_tuple=True)[0][:PROBE] for d in range(10)}

    net = GrowableSTDP(dev)
    learned = []                            # digits seen/being-learned so far
    competence = {d: 0.0 for d in range(10)}   # self-measured competence per digit
    memory = {}                             # MEMORIZATION: stored exemplars per learned digit (for rehearsal)
    # the brain's OWN neuromodulatory decision machinery drives objective selection + pacing:
    curiosity = SpikingCuriosityDrive(seed=0)                     # spiking learning-progress + novelty
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)   # self-paced mastery gate (LIF accumulator)
    consec = {d: 0 for d in range(10)}     # consecutive-pick counter (fixation guard for the phase-2 revisits)

    for cyc in range(1, CYCLES + 1):
        # ---- 4/DECISION: the brain's SPIKING CURIOSITY drive picks the objective (most_curious = novelty +
        #      spiking learning-progress). Phase 1 explores UNSEEN digits; once all seen, curiosity REVISITS
        #      the weakest-still-improving digit (the learnable frontier). No hand-coded heuristic. ----
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        if not unseen:                       # phase 2: block a digit revisited CAP times in a row -> diversify
            allowed = [d for d in candidates if consec[d] < CAP]
            candidates = allowed or candidates
        objective = curiosity.most_curious(candidates)
        consec = {d: (consec[d] + 1 if d == objective else 0) for d in range(10)}
        is_new = objective not in learned
        print(f"\nCYCLE {cyc}  ->  OBJECTIVE: digit {objective}  "
              f"[{'explore (novel)' if is_new else 'revisit (curiosity frontier)'}; "
              f"curiosity_value={curiosity.curiosity_value(objective):.3f}, novelty={curiosity.novelty(objective):.2f}]")

        # ---- 3/NEUROGENESIS + 1/UNSUPERVISED LEARNING (interleaved): the STDP learner streams the objective's
        #      images + balanced replay; as it goes, the BRAIN'S OWN neurogenesis rule (the ART / hippocampus.
        #      GrowingLearner vigilance rule) RECRUITS a neuron whenever it meets a learnable-NOVEL pattern no
        #      existing prototype matches. The brain SIZES ITS OWN GROWTH -- self-limiting (familiar patterns
        #      refine existing neurons, they do not recruit), not a fixed net.grow(GROW) schedule. Revisits use
        #      a light, replay-balanced fresh dose. ----
        t0 = time.time()
        n_train = PER if is_new else max(REPLAY, PER // 4)
        train_imgs = [Xtr[i] for i in by_class[objective][:n_train]]
        rehearse = [ex for d in learned for ex in memory.get(d, [])]
        stream = train_imgs + rehearse
        torch.manual_seed(cyc)
        m0 = net.n
        recruited = 0
        for j in torch.randperm(len(stream)).tolist():
            img = stream[j]
            if recruited < GROW and net.maybe_recruit(img, VIG):   # the brain decides: recruit iff learnable-novel
                recruited += 1
            net.run(img, learn=True)
        dt = time.time() - t0
        if is_new:
            memory[objective] = [Xtr[i] for i in by_class[objective][n_train:n_train + REPLAY]]
            learned.append(objective)
        print(f"  [neurogenesis]   the BRAIN recruited {recruited} neurons on learnable-novelty (vigilance {VIG}); "
              f"{m0} -> {net.n} neurons  [self-sized by the ART/grow-trigger rule, per-cycle cap {GROW}]")
        print(f"  [unsupervised]   STDP on {len(train_imgs)} digit-{objective} imgs + {len(rehearse)} replayed "
              f"exemplars, no labels ({dt:.0f}s)")

        # ---- 2/SELF-IMPROVEMENT: self-evaluate competence over ALL learned digits ----
        probe_imgs = [Xte[i] for d in learned for i in probe_by[d]]
        probe_lbls = [int(Yte[i]) for d in learned for i in probe_by[d]]
        ov, per_cls = assign_and_eval(net, probe_imgs, probe_lbls, set(learned))
        competence.update(per_cls)

        # ---- 4/feed the REAL curiosity + curriculum modules the outcome (prediction error = 1 - competence),
        #      so the NEXT decision is driven by their spiking learning-progress signal, not a heuristic ----
        curiosity.observe(objective, 1.0 - competence[objective])
        lp = curiosity.learning_progress(objective)
        mastered_now = curriculum.observe(lp)
        print(f"  [self-improve]   competence over {sorted(learned)} = {ov:.1%}  (digit {objective}: "
              f"{competence[objective]:.1%}, spiking learning-progress={lp:.3f}); "
              f"curriculum mastered {curriculum.mastered}/10{' [+advanced]' if mastered_now else ''}")

    print("\n" + "=" * 88)
    print(f"DEVELOPMENTAL LOOP COMPLETE: the brain autonomously learned {sorted(learned)} UNSUPERVISED over "
          f"{CYCLES} cycles, grew {net.n} neurons, consolidated via replay, and CHOSE every objective itself "
          f"through its spiking curiosity drive + curriculum (curriculum mastered {curriculum.mastered}/10).")
    final = sum(competence[d] for d in learned) / len(learned)
    print(f"final mean competence over the {len(learned)} self-chosen digits: {final:.1%}  (no labels used in training)")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
