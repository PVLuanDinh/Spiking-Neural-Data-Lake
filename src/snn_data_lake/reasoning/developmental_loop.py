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
     (+ PRUNE, + LC)         no existing prototype matches (self-limiting, need-sized -- NOT a fixed grow(GROW)
                             schedule), and PRUNES dead never-winning neurons (plasticity/developmental_pruning:
                             the brain grows AND prunes). The locus-coeruleus (neuromodulation) boosts the STDP
                             learning gain 1x-3x on a salient/novel objective (attend harder to the unknown).
                             It self-evaluates competence by neuron-label voting (labels only read out competence).
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

SOFT k-WTA + PRUNING (v3.47, opt-in NORD_INH / NORD_PRUNE) -- HONEST NEGATIVES, kept for reproducibility:
graded lateral inhibition (a continuously-adjustable global coupling) looked strong on a 3-digit smoke
(94-96%) but that was a LOW-CLASS-COUNT artifact: at the full 10 classes it does NOT beat hard top-1 WTA --
inh=5 -> 68.5%, inh=15 -> 76.2% (soft peak), inh=50 -> 71.2%, all below hard-WTA's 88.8%. Developmental
dead-pruning also fails to pay off: naive pruning COLLAPSES the net (inh=5 -> 52 neurons / 29.8%); a
newborn-grace critical period fixes the collapse (~150-260 neurons / 58.2%) but pruning still HURTS accuracy
(low-activity neurons carry useful class-vote mass). Default stays HARD-WTA, no pruning (88.8%).
"""
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
pass  # migrated: sys.path bootstrap removed (now an importable package)
pass  # migrated: sys.path bootstrap removed (now an importable package)

import torch

from snn_data_lake.reasoning.snn_mnist_stdp import (BETA, KWTA, LR, MAX_RATE, N_IN, T, TAU_PRE, THETA_DECAY,
                            THETA_PLUS, THRESH, W_NORM, WMAX, load_mnist)
from snn_data_lake.reasoning.torch_device import enable, resolve_device

from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus
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

    def __init__(self, device, inh=None, alpha=1.0):
        self.device = device
        self.alpha = float(alpha)              # criticality metric slope (surrogate-derivative g'(u-Vth), arXiv 2311.16141)
        self.last_crit = None                  # per-neuron criticality of the most recent run() (near-threshold-ness)
        # inh=None -> HARD top-1 k-WTA + full lateral reset (the verified snn_mnist_stdp mechanic, best acc
        # 88.8% at 10-class scale). A float turns on experimental SOFT k-WTA (graded global coupling): kept
        # as an opt-in because it does NOT beat hard-WTA at 10 classes (peaks ~76% at inh=15) -- see docstring.
        self.inh = None if inh is None else float(inh)
        self.n = 0
        self.W = torch.zeros(0, N_IN, device=device)
        self.theta = torch.zeros(0, device=device)
        self.birth = torch.zeros(0, device=device)   # cycle each neuron was recruited (newborn-protection critical period)
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

    def maybe_recruit(self, image, vigilance, cycle=0):
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
            self.birth = torch.cat([self.birth, torch.full((1,), float(cycle), device=self.device)], dim=0)
            self.n += 1
            return True
        return False

    def prune_dead(self, activity, min_spikes=1, cycle=0, grace=1, criticality=None):
        """Developmental PRUNING (the brain grows AND prunes -- plasticity/developmental_pruning +
        lifelong_consolidation). NEWBORN PROTECTION: a neuron recruited within the last `grace` cycles is
        never pruned (a developmental critical period) -- otherwise pruning kills freshly-recruited neurons
        before they have had a cycle to win, fighting neurogenesis. Returns the number pruned (keeps >=1).

        Selection metric (ablation):
          - criticality=None  -> ACTIVITY rule: drop neurons whose spike total < `min_spikes` (v3.46/47 default;
            fails here because low-activity != useless under k-WTA).
          - criticality given  -> CRITICALITY rule (arXiv 2311.16141): drop the SAME NUMBER the activity rule
            would (K = #dead-and-old), but pick the K LEAST-CRITICAL old neurons (lowest near-threshold score),
            keeping barely-excitable-but-informative sparse coders the activity rule wrongly deletes. Pure
            metric swap -> net-size trajectory identical, only the SELECTION differs."""
        newborn = (cycle - self.birth) < grace              # protected: too young to prune
        dead = (activity < min_spikes) & (~newborn)         # what the activity rule would drop
        k = int(dead.sum().item())
        if criticality is None or k == 0:
            keep = (activity >= min_spikes) | newborn       # ACTIVITY rule (or nothing to prune)
        else:                                               # CRITICALITY rule: drop the k least-critical OLD neurons
            score = criticality.clone()
            score[newborn] = float("inf")                   # newborns never selected for pruning
            drop_idx = torch.topk(score, k, largest=False).indices
            keep = torch.ones(self.n, dtype=torch.bool, device=self.device)
            keep[drop_idx] = False
        pruned = int((~keep).sum().item())
        if pruned and keep.any():
            self.W = self.W[keep]
            self.theta = self.theta[keep]
            self.birth = self.birth[keep]
            self.n = int(keep.sum().item())
        return pruned

    def run(self, image, learn, lr_scale=1.0):
        """Simulate one image for T steps; return the exc spike-count vector [n]. ALL neurons stay plastic
        (shared representation); forgetting is held off by strong REPLAY (rehearsal) of old digits, not by
        freezing -- replay-based continual learning. `lr_scale` multiplies the STDP step (the locus-coeruleus
        noradrenaline gain: a salient/novel input drives a bigger, more decisive weight change)."""
        image = image.to(self.device)
        mem = self.lif.init_leaky()
        x_pre = torch.zeros(N_IN, device=self.device)
        counts = torch.zeros(self.n, device=self.device)
        crit = torch.zeros(self.n, device=self.device)     # CRITICALITY (arXiv 2311.16141): sum_t g'(eff - THRESH),
        rates = image * MAX_RATE                            #   g'(x)=1/(1+alpha^2 x^2) peaks when membrane ~ threshold
        for _ in range(T):
            s_in = (torch.rand(N_IN, device=self.device) < rates).float()
            x_pre = x_pre * self.pre_decay + s_in
            cur = self.W @ s_in
            _, mem = self.lif(cur, mem)
            eff = mem - self.theta
            d = eff - THRESH                                # membrane-to-threshold distance (before any WTA reset)
            crit += 1.0 / (1.0 + (self.alpha * d) ** 2)     # near-threshold = high criticality (barely-excitable)
            if self.inh is None:
                # HARD top-1 k-WTA (the verified snn_mnist_stdp mechanic, KWTA=1): the single strongest neuron
                # that clears threshold fires, then a FULL lateral-inhibition reset silences everyone that step.
                cand = (eff > THRESH).nonzero(as_tuple=True)[0]
                if cand.numel() > 0:
                    w = int(cand[torch.topk(eff[cand], min(KWTA, cand.numel())).indices][0])
                    counts[w] += 1
                    if learn:
                        self.W[w] += (LR * lr_scale) * x_pre
                        self.W[w].clamp_(0.0, WMAX)
                        self.theta[w] += THETA_PLUS
                    mem = torch.zeros(self.n, device=self.device)
            else:
                fired = eff > THRESH                       # SOFT k-WTA: the whole supra-threshold POPULATION fires
                nf = int(fired.sum().item())               #   (graded competition, not a single hard winner)
                if nf > 0:
                    idx = fired.nonzero(as_tuple=True)[0]
                    counts[idx] += 1
                    if learn:                              # vectorised STDP for the co-firing population (all plastic)
                        self.W[idx] += (LR * lr_scale) * x_pre.unsqueeze(0)   # LR scaled by the LC noradrenaline gain
                        self.W[idx].clamp_(0.0, WMAX)
                        self.theta[idx] += THETA_PLUS
                    # graded ALL-BUT-SELF lateral inhibition (a continuously-adjustable GLOBAL coupling): each neuron's
                    # membrane drops by inh * (spikes of the OTHERS) -- SMOOTH suppression by rank, not a hard reset.
                    # Spreads activity across a population; makes pruning non-collapsing (with newborn grace) but does
                    # NOT beat hard-WTA on accuracy at 10 classes -- kept opt-in as an honest negative.
                    mem = (mem - self.inh * (nf - fired.float())).clamp_min(0.0)
            if learn:
                self.theta *= THETA_DECAY
        if learn:
            s = self.W.sum(dim=1, keepdim=True).clamp_min(1e-6)
            self.W *= (W_NORM / s)
        self.last_crit = crit
        return counts


def assign_and_eval(net, probe_imgs, probe_lbls, classes):
    """Neuron-label voting on a labelled PROBE (labels used only to read out competence, not to train),
    then HELD-OUT accuracy. The probe is split into DISJOINT halves: neuron labels are assigned on the
    first half, accuracy is scored on the SECOND half -- otherwise assigning and scoring on the same
    images optimistically biases competence (a real bug fixed in v3.72). Returns (overall_acc,
    {class: acc}, activity)."""
    half = len(probe_imgs) // 2
    assign_imgs, assign_lbls = probe_imgs[:half], probe_lbls[:half]
    eval_imgs, eval_lbls = probe_imgs[half:], probe_lbls[half:]
    if not eval_imgs:                                          # too few probe images -> fall back (no split possible)
        assign_imgs, assign_lbls = probe_imgs, probe_lbls
        eval_imgs, eval_lbls = probe_imgs, probe_lbls
    resp = torch.zeros(net.n, 10, device=net.device)
    per = torch.zeros(10, device=net.device)
    activity = torch.zeros(net.n, device=net.device)          # per-neuron spike totals -> feeds pruning
    for img, y in zip(assign_imgs, assign_lbls):              # ASSIGN neuron labels on the first half
        counts = net.run(img, learn=False)
        resp[:, int(y)] += counts
        activity += counts
        per[int(y)] += 1
    assignment = (resp / per.clamp_min(1)).argmax(dim=1)
    correct = {c: 0 for c in classes}
    total = {c: 0 for c in classes}
    for img, y in zip(eval_imgs, eval_lbls):                  # SCORE on the disjoint second half
        c = int(y)
        if c not in classes:
            continue
        counts = net.run(img, learn=False)
        activity += counts
        scores = torch.zeros(10, device=net.device)
        for cc in range(10):
            m = assignment == cc
            if m.any():
                scores[cc] = counts[m].sum()
        correct[c] += int(scores.argmax().item() == c)
        total[c] += 1
    per_cls = {c: (correct[c] / total[c] if total[c] else 0.0) for c in classes}
    ov = sum(correct.values()) / max(1, sum(total.values()))
    return ov, per_cls, activity


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
    PRUNE = _envi("NORD_PRUNE", 0)         # opt-in developmental pruning (HONEST: hurts here regardless -- see below)
    INH = float(os.environ["NORD_INH"]) if os.environ.get("NORD_INH") else None   # None -> HARD top-1 WTA (default, best 88.8%); a float -> experimental SOFT k-WTA
    CRIT = _envi("NORD_CRIT", 0)           # prune selection metric: 0 -> activity (v3.47), 1 -> CRITICALITY (arXiv 2311.16141)
    ALPHA = float(os.environ.get("NORD_ALPHA", "1.0"))   # criticality metric slope g'(x)=1/(1+(alpha*x)^2)

    print("=" * 88)
    print("DEVELOPMENTAL LOOP -- unsupervised learning -> memorize -> grow neurons -> decide next -> repeat")
    print(f"device={dev}  cycles={CYCLES}  grow/cycle={GROW}  train/digit={PER}  probe/digit={PROBE}")
    print("=" * 88)

    Xtr, Ytr, Xte, Yte = load_mnist(6000, 3000)
    by_class = {d: (Ytr == d).nonzero(as_tuple=True)[0] for d in range(10)}
    probe_by = {d: (Yte == d).nonzero(as_tuple=True)[0][:PROBE] for d in range(10)}

    net = GrowableSTDP(dev, inh=INH, alpha=ALPHA)
    learned = []                            # digits seen/being-learned so far
    competence = {d: 0.0 for d in range(10)}   # self-measured competence per digit
    memory = {}                             # MEMORIZATION: stored exemplars per learned digit (for rehearsal)
    # the brain's OWN neuromodulatory decision machinery drives objective selection + pacing:
    curiosity = SpikingCuriosityDrive(seed=0)                     # spiking learning-progress + novelty
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)   # self-paced mastery gate (LIF accumulator)
    lc = LocusCoeruleus()                   # noradrenaline: salient/novel input -> phasic burst -> higher learning gain
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
        # NEUROMODULATION -- locus coeruleus: a NOVEL / hard objective is SALIENT -> phasic NA burst -> higher
        # learning gain (the brain attends harder to what it does not yet know). gain scales the STDP step.
        salience = 1.0 if is_new else (1.0 - competence[objective])
        na = lc.step(salience)                 # NA firing rate in [0,1] (phasic burst on a salient/novel event)
        gain = 1.0 + 2.0 * na                  # noradrenaline boosts the STDP step 1x..3x on salient/novel input

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
            if recruited < GROW and net.maybe_recruit(img, VIG, cycle=cyc):   # the brain decides: recruit iff learnable-novel
                recruited += 1
            net.run(img, learn=True, lr_scale=gain)                # LC noradrenaline gain modulates the STDP step
        dt = time.time() - t0
        if is_new:
            memory[objective] = [Xtr[i] for i in by_class[objective][n_train:n_train + REPLAY]]
            learned.append(objective)
        print(f"  [neurogenesis]   the BRAIN recruited {recruited} neurons on learnable-novelty (vigilance {VIG}); "
              f"{m0} -> {net.n} neurons  [self-sized by the ART/grow-trigger rule, per-cycle cap {GROW}]")
        print(f"  [neuromod]       locus-coeruleus NA {na:.2f} ({lc.mode()}) on salience {salience:.2f} "
              f"-> STDP step gain x{gain:.2f}")
        print(f"  [unsupervised]   STDP on {len(train_imgs)} digit-{objective} imgs + {len(rehearse)} replayed "
              f"exemplars, no labels ({dt:.0f}s)")

        # ---- 3b/PRUNE (opt-in, NORD_PRUNE=1): the brain grows AND prunes (plasticity/developmental_pruning) --
        #      drop neurons dead over the training distribution, with a NEWBORN-GRACE critical period (a neuron
        #      recruited this cycle is never pruned). HONEST, MEASURED: the grace gate stops the pruning COLLAPSE
        #      (soft-kWTA inh=5 + naive prune died to 52 neurons / 29.8%; + grace stabilises ~150-260 neurons /
        #      58.2%), but pruning STILL HURTS accuracy vs no-prune (inh=15: 76.2% -> 58.2%; hard-WTA 88.8% ->
        #      lower) -- in this replay-STDP vote, low-activity neurons still carry useful class-vote mass, so
        #      removing them removes signal. Conclusion: developmental dead-pruning is NOT a win for this system
        #      (dense-synapse mechanism, not sparse-prototype). Left OFF by default. ----
        if PRUNE:
            pact = torch.zeros(net.n, device=dev)
            pcrit = torch.zeros(net.n, device=dev)
            for img in stream[:min(len(stream), 300)]:
                pact += net.run(img, learn=False)
                pcrit += net.last_crit                       # accumulate criticality over the same probe pass
            crit_arg = pcrit if CRIT else None               # NORD_CRIT=1 -> rank pruning by criticality, else activity
            pruned = net.prune_dead(pact, min_spikes=1, cycle=cyc, grace=1, criticality=crit_arg)
            if pruned:
                metric = "least-critical" if CRIT else "dead"
                print(f"  [prune]          removed {pruned} {metric} neurons -> {net.n} live (NORD_PRUNE on)")

        # ---- 2/SELF-IMPROVEMENT: self-evaluate competence over ALL learned digits ----
        probe_imgs = [Xte[i] for d in learned for i in probe_by[d]]
        probe_lbls = [int(Yte[i]) for d in learned for i in probe_by[d]]
        ov, per_cls, _ = assign_and_eval(net, probe_imgs, probe_lbls, set(learned))
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
