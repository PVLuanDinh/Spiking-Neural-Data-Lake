r"""
DEEP DEVELOPMENTAL LOOP -- the v3.53 winner (unsupervised conv1 latency-STDP features +
a LINEAR readout, 95.1% static joint) wired INTO the autonomous developmental loop, so the
loop's curiosity / curriculum / replay drive a CONTINUAL linear readout on frozen features.

WHAT THIS FILE IS (and what it replaces)
----------------------------------------
research/developmental_loop.py runs the full autonomous cycle -- spiking curiosity picks the
next objective digit, the STDP learner grows neurons (neurogenesis), replay consolidates old
digits, competence feeds the next decision -- but its ACCURACY path is the STDP-prototype +
neuron-label VOTE readout, which plateaus at ~88% (best 88.8%). research/deep_stdp_readout.py
(v3.53) showed the CAP was the readout, not the features: a LINEAR readout on the SAME frozen
unsupervised conv1 features reaches 95.1% (static, joint over all digits at once).

This file keeps the loop's AUTONOMY machinery verbatim (SpikingCuriosityDrive, SpikingCurriculum,
LocusCoeruleus -- objective selection, self-pacing, neuromodulated gain, replay-based continual
learning) but SWAPS the accuracy mechanism: the learner is now a single CONTINUAL nn.Linear over
CACHED conv1 features, trained ONLINE across cycles (never re-initialised) with a balanced replay
of previously-learned digits as the anti-forgetting mechanism. The curiosity/curriculum/replay
CONTROL FLOW is unchanged; only the thing being trained changed (STDP prototypes -> linear head).

THE ACCEPTED THESIS TRADEOFF (honest)
-------------------------------------
FEATURE LEARNING STAYS FULLY UNSUPERVISED: the conv1 latency-STDP extractor is pre-trained with
NO labels and NO backprop, then FROZEN (the Kheradpisheh protocol). ONLY the linear readout is
supervised -- it sees labels, on frozen features. This is the accepted design of the deepening
arc (v3.49-53), not a regression: the spiking substrate is unsupervised; the thin linear readout
on top is the standard supervised probe.

THE HONEST OPEN QUESTION THIS ANSWERS
-------------------------------------
Does a CONTINUAL, self-directed, replay-based linear readout reach the STATIC joint 95.1%, or
does continual learning (learning one digit at a time, revisiting, fighting catastrophic
forgetting in the linear head with replay) COST a few points versus training the readout jointly
over all digits at once? Either answer is a valid, reported result. This file does NOT rig the
number: the self-check asserts the continual readout BEATS the vote-loop's ~0.85 (a real
improvement) and beats chance, but does NOT assert it hits 95.1% -- the signed gap to the static
joint is printed honestly, whatever its sign.

HOW THE CONTINUAL READOUT + REPLAY + CACHING WORK (as coded)
------------------------------------------------------------
  - CACHING (the speed win): the conv1 extractor is frozen, so its feature vectors never change.
    We run the conv ONCE over the whole train + test pool and cache the [N, feat_dim] matrices.
    Every later cycle is then just cheap cached-feature SGD -- no conv re-runs per cycle.
  - CONTINUAL readout: a single nn.Linear(feat_dim, 10) with ONE persistent Adam optimizer. It is
    never reset between cycles -- weights and Adam momentum accumulate, so knowledge is carried
    forward (continual learning), which is exactly what exposes forgetting when a new digit arrives.
  - REPLAY: as each digit is first learned, a fixed pool of NORD_REPLAY of its cached train-feature
    rows is stored. Every LEARN step trains on the objective digit's features PLUS a balanced replay
    of every previously-learned digit's stored features -- rehearsal is the loop's anti-forgetting
    mechanism (systems consolidation), the same role replay plays in developmental_loop.py.

Train and test feature caches are kept strictly separate: the readout is trained ONLY on the train
cache; competence is measured ONLY on the held-out test cache. No train/test row ever mixes.

Needs the torch venv (GPU strongly recommended; the conv is pre-trained ONCE, per-cycle is fast):
    NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/developmental_loop_deep.py
Scale knobs (env): NORD_CYCLES (15, >10 so phase-2 revisits) NORD_C1 (30) NORD_T (15)
                   NORD_L1_EPOCHS (3) NORD_PER (300) NORD_REPLAY (200) NORD_SGD (300)
                   NORD_CAP (1, anti-fixation) NORD_TRAIN (10000) NORD_TEST (2000) NORD_GPU
Deps: numpy torch torchvision snntorch   (research/GPU tier; core untouched)
"""
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
pass  # migrated: sys.path bootstrap removed (now an importable package)
pass  # migrated: sys.path bootstrap removed (now an importable package)

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_data_lake.reasoning.snn_mnist_stdp import load_mnist
from snn_data_lake.reasoning.torch_device import enable, resolve_device

from snn_data_lake.reasoning.deep_stdp_latency import ConvLatencyStdp

from snn_data_lake.neuromodulation.locus_coeruleus import LocusCoeruleus
from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum

# reference accuracies to compare against (printed, and used as the honest acceptance bar):
VOTE_LOOP_BAR = 0.85    # developmental_loop.py STDP-prototype + neuron-label VOTE readout (best 88.8%)
STATIC_JOINT = 0.951    # deep_stdp_readout.py (v3.53) LINEAR readout on frozen conv1, joint over all digits
BASE_LR = 0.01          # linear-readout Adam lr (matches deep_stdp_readout.linear_readout)
WEIGHT_DECAY = 1e-4


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _cache_features(conv, X, dev, tag):
    """Run the FROZEN conv1 extractor ONCE over the pool and stack the [N, feat_dim] feature
    matrix on the resolved device. Features never change (conv is frozen), so caching here means
    every later cycle is cheap cached-feature SGD -- the main speed win. GPU-safe: conv.features()
    already pins to resolve_device(); we .to(dev) the stack to be certain no CPU tensor leaks in."""
    rows = []
    for i, img in enumerate(X):
        rows.append(conv.features(img).reshape(-1))
        if (i + 1) % 2000 == 0:
            print(f"  cache {tag}: {i + 1}/{len(X)} features")
    return torch.stack(rows, dim=0).to(dev)


def _accuracy(clf, feats, labels, idx):
    """Held-out linear-readout accuracy over the given cached-feature rows (test cache only)."""
    if len(idx) == 0:
        return 0.0
    with torch.no_grad():
        pred = clf(feats[idx]).argmax(dim=1)
        return (pred == labels[idx]).float().mean().item()


def main():
    enable(resolve_device())
    dev = resolve_device()
    torch.manual_seed(0)

    CYCLES = _envi("NORD_CYCLES", 15)      # >10 so curiosity REVISITS the weakest digits after covering all 10
    C1 = _envi("NORD_C1", 30)              # conv1 feature maps
    T_LAT = _envi("NORD_T", 15)            # latency steps
    L1_EPOCHS = _envi("NORD_L1_EPOCHS", 3)  # unsupervised conv1 pre-training epochs
    PER = _envi("NORD_PER", 300)           # objective-digit train features per cycle
    REPLAY = _envi("NORD_REPLAY", 200)     # replayed exemplars stored + rehearsed per learned digit (anti-forgetting).
    #                                        200/300 verified: 93.5% (-1.6% vs joint); 80/150 gave 90.7% (-4.4%) --
    #                                        more rehearsal narrows the continual-vs-joint forgetting gap.
    SGD = _envi("NORD_SGD", 300)           # Adam steps per cycle on the objective + replay batch
    CAP = _envi("NORD_CAP", 1)             # max consecutive revisits of the SAME digit (anti-fixation -> diversify)
    TRAIN_N = _envi("NORD_TRAIN", 10000)
    TEST_N = _envi("NORD_TEST", 2000)

    print("=" * 90)
    print("DEEP DEVELOPMENTAL LOOP -- unsupervised conv1 features (frozen) + CONTINUAL linear readout")
    print("  autonomy machinery (curiosity/curriculum/replay) preserved; accuracy path = conv1 + linear")
    print(f"device={dev}  cycles={CYCLES}  C1={C1}  T={T_LAT}  L1_epochs={L1_EPOCHS}  per={PER}  "
          f"replay={REPLAY}  sgd={SGD}  cap={CAP}  train={TRAIN_N}  test={TEST_N}")
    print("=" * 90)

    print("loading MNIST...")
    Xtr, Ytr, Xte, Yte = load_mnist(TRAIN_N, TEST_N)

    # ---- STAGE 1: UNSUPERVISED PERCEPTION (pre-train conv1, no labels, then FREEZE) ----
    torch.manual_seed(0)
    print(f"\n[stage 1] pre-training conv1 latency-STDP (UNSUPERVISED, {L1_EPOCHS} epochs, no labels)...")
    conv = ConvLatencyStdp(C1, T_LAT)
    for ep in range(L1_EPOCHS):
        for i, img in enumerate(Xtr):
            conv.train_image(img)
            if (i + 1) % 2000 == 0:
                print(f"  conv1 ep{ep + 1}/{L1_EPOCHS}: {i + 1}/{len(Xtr)} images")
        print(f"  [conv1] epoch {ep + 1}/{L1_EPOCHS} kernel diversity (cos-sim) = {conv.kernel_diversity():.3f}")
    diversity = conv.kernel_diversity()
    print(f"[stage 1] conv1 FROZEN; feat_dim = {conv.feat_dim()}; kernel diversity = {diversity:.3f} "
          f"({'no collapse' if diversity < 0.6 else 'WARNING: collapse (>=0.6)'})")

    # ---- CACHE conv1 features ONCE for the whole train + test pool (the speed win) ----
    print("\n[cache] pre-computing frozen conv1 features for train + test pools (once)...")
    Ftr = _cache_features(conv, Xtr, dev, "train")
    Fte = _cache_features(conv, Xte, dev, "test")
    ytr = Ytr.long().to(dev)
    yte = Yte.long().to(dev)
    D = Ftr.shape[1]
    tr_by = {d: (Ytr == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    te_by = {d: (Yte == d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
    print(f"[cache] train cache {tuple(Ftr.shape)}  test cache {tuple(Fte.shape)}  (train/test kept separate)")

    # ---- STAGE 2: the CONTINUAL linear readout (single persistent head, never re-initialised) ----
    clf = nn.Linear(D, 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    # ---- the loop's OWN autonomy machinery (verbatim from developmental_loop.py) ----
    curiosity = SpikingCuriosityDrive(seed=0)                         # spiking learning-progress + novelty
    curriculum = SpikingCurriculum(n_stages=10, patience=2, seed=0)   # self-paced mastery gate (LIF accumulator)
    lc = LocusCoeruleus()                                             # noradrenaline: salient/novel -> higher gain
    learned = []                                                     # digits chosen/learned so far
    competence = {d: 0.0 for d in range(10)}                        # self-measured competence per digit (test cache)
    replay_pool = {}                                                # REPLAY: stored cached-feature rows per learned digit
    consec = {d: 0 for d in range(10)}                             # consecutive-pick counter (phase-2 fixation guard)
    curve = []                                                      # overall test accuracy per cycle (the accuracy CURVE)

    for cyc in range(1, CYCLES + 1):
        # ---- DECISION: spiking curiosity picks the objective. Phase 1 explores UNSEEN digits;
        #      once all seen, phase 2 REVISITS the weakest-still-improving digit on the learnable
        #      frontier, with the NORD_CAP anti-fixation guard (copied verbatim). ----
        unseen = [d for d in range(10) if d not in learned]
        candidates = unseen if unseen else list(range(10))
        if not unseen:                                             # phase 2: block a digit revisited CAP times in a row
            allowed = [d for d in candidates if consec[d] < CAP]
            candidates = allowed or candidates
        objective = curiosity.most_curious(candidates)
        consec = {d: (consec[d] + 1 if d == objective else 0) for d in range(10)}
        is_new = objective not in learned
        print(f"\nCYCLE {cyc}  ->  OBJECTIVE: digit {objective}  "
              f"[{'explore (novel)' if is_new else 'revisit (curiosity frontier)'}; "
              f"curiosity_value={curiosity.curiosity_value(objective):.3f}, novelty={curiosity.novelty(objective):.2f}]")

        # ---- NEUROMODULATION: a novel / hard objective is SALIENT -> locus-coeruleus phasic NA burst
        #      -> a (bounded, stable) boost to the readout learning rate. Faithful to the loop's neuromod. ----
        salience = 1.0 if is_new else (1.0 - competence[objective])
        na = lc.step(salience)                                     # NA firing rate in [0,1]
        gain = 1.0 + 0.5 * na                                      # gentle 1x..1.5x lr boost (kept stable, not 3x)
        for g in opt.param_groups:
            g["lr"] = BASE_LR * gain

        # ---- LEARN: continual SGD on the objective digit's cached features + a BALANCED REPLAY of every
        #      previously-learned digit's stored cached features (rehearsal = anti-forgetting). ----
        n_train = PER if is_new else max(REPLAY, PER // 4)         # revisits use a lighter dose (mirror the loop)
        obj_rows = tr_by[objective][:n_train]
        others = [d for d in learned if d != objective]
        replay_rows = [r for d in others for r in replay_pool.get(d, [])]
        batch_rows = obj_rows + replay_rows
        idx = torch.tensor(batch_rows, dtype=torch.long, device=dev)
        Xb, yb = Ftr[idx], ytr[idx]
        t0 = time.time()
        clf.train()
        for _ in range(SGD):                                      # full-batch Adam on the objective + replay batch
            opt.zero_grad()
            loss = F.cross_entropy(clf(Xb), yb)
            loss.backward()
            opt.step()
        clf.eval()
        dt = time.time() - t0
        if is_new:                                                # store this digit's replay pool (held-out from the training rows)
            replay_pool[objective] = tr_by[objective][n_train:n_train + REPLAY]
            learned.append(objective)
        print(f"  [neuromod]         locus-coeruleus NA {na:.2f} ({lc.mode()}) on salience {salience:.2f} "
              f"-> readout lr gain x{gain:.2f}")
        print(f"  [continual-readout] {SGD} Adam steps on {len(obj_rows)} digit-{objective} feats + "
              f"{len(replay_rows)} replayed feats (loss {float(loss):.3f}, {dt:.1f}s)")

        # ---- MEASURE competence: held-out linear-readout accuracy per learned digit on the TEST cache
        #      (the loop's self-evaluation). Overall (pooled) + per-digit. Train/test never mix. ----
        for d in learned:
            competence[d] = _accuracy(clf, Fte, yte, te_by[d])
        pooled_test_idx = torch.tensor([r for d in learned for r in te_by[d]], dtype=torch.long, device=dev)
        overall = _accuracy(clf, Fte, yte, pooled_test_idx.tolist())
        curve.append(overall)

        # ---- DECIDE: feed the REAL curiosity + curriculum modules the outcome (pred error = 1 - competence),
        #      so the NEXT objective is driven by their spiking learning-progress signal, not a heuristic. ----
        curiosity.observe(objective, 1.0 - competence[objective])
        lp = curiosity.learning_progress(objective)
        mastered_now = curriculum.observe(lp)
        print(f"  [self-improve]      competence over {sorted(learned)} = {overall:.1%}  (digit {objective}: "
              f"{competence[objective]:.1%}, spiking learning-progress={lp:.3f}); "
              f"curriculum mastered {curriculum.mastered}/10{' [+advanced]' if mastered_now else ''}")

    # ---- SUMMARY: final competence + the accuracy CURVE + honest comparison to the two references ----
    final_mean = sum(competence[d] for d in learned) / max(1, len(learned))
    final_overall = curve[-1] if curve else 0.0
    gap_joint = final_overall - STATIC_JOINT
    print("\n" + "=" * 90)
    print(f"DEEP DEVELOPMENTAL LOOP COMPLETE: the loop autonomously chose + learned {sorted(learned)} over "
          f"{CYCLES} cycles;\n  perception UNSUPERVISED (frozen conv1), only the CONTINUAL linear readout supervised; "
          f"curriculum mastered {curriculum.mastered}/10.")
    print("-" * 90)
    print(f"  final mean competence over the {len(learned)} self-chosen digits : {final_mean:.1%}")
    print(f"  final OVERALL held-out test accuracy (all learned digits)      : {final_overall:.1%}")
    print(f"  accuracy CURVE (overall per cycle) : " + " ".join(f"{a:.2f}" for a in curve))
    print("-" * 90)
    print(f"  reference VOTE developmental loop (STDP prototypes + vote)     : ~{VOTE_LOOP_BAR:.0%} (best 88.8%)")
    print(f"  reference STATIC joint linear readout (conv1, all digits once) : {STATIC_JOINT:.1%}")
    print(f"  continual (this) vs static joint                               : {gap_joint:+.1%} "
          f"({'continual reaches joint' if gap_joint >= 0 else 'continual learning costs accuracy'})")
    print("=" * 90)

    # self-check: the continual readout must be a REAL improvement over the vote loop (> ~0.85) AND beat
    # chance (guards the pipeline). We do NOT assert it hits the static joint 95.1% -- the signed gap is the
    # honest, reported result (continual learning reaching OR falling short of the joint both answer the
    # question). Do not lower these bars to make it pass; if the honest number is below, report it.
    assert final_overall > 0.10, f"continual readout at/below chance: {final_overall:.2f}"
    assert final_overall > VOTE_LOOP_BAR, (
        f"continual readout {final_overall:.3f} did NOT beat the vote-loop bar {VOTE_LOOP_BAR:.2f} -- "
        "the deep readout should improve on the STDP-vote path; report the honest number, do not lower the bar.")
    verdict = "reaches the static joint" if gap_joint >= 0 else f"costs {-gap_joint:.1%} vs the static joint"
    print(f"self-check OK: continual deep readout {final_overall:.1%} > vote-loop ~{VOTE_LOOP_BAR:.0%} and > chance; "
          f"conv1 unsupervised (diversity {diversity:.3f}); continual {verdict} (honest).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
