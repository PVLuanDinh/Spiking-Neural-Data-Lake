r"""
LANGUAGE SELF-BUILD -- the brain builds a language skill BY ITSELF, through autonomous
self-supervised SPIKING learning, forming cell assemblies at three levels.

This is the Garagnani-Pulvermuller distributed-cell-assembly idea made autonomous: no labels
anywhere; the teaching signal at every level is the token that ACTUALLY comes next in a real
corpus (supervision-by-the-future). Learning is reward-modulated spike-timing plasticity (R-STDP),
never a gradient -- honouring the project's standing spiking-purity rule.

THREE LEVELS (each an assembly pool + a spiking learning rule, all reused from the repo):
  PHONETICS  cell assemblies = R-STDP response units over the 39 ARPABET phones; context =
             rank-order spike code of the previous phonemes; self-supervised next-phoneme.
  SEMANTICS  cell assemblies over the most frequent words (a bounded vocab -- honest scale note
             below); self-supervised next-word (spiking distributional semantics = Hebbian
             co-occurrence read out by spike timing).
  SYNTAX     order sensitivity: the same spike-timing readout on ORDER-PRESERVING context vs a
             SHUFFLED-order context. **MEASURED NULL, 2026-08-04** -- swept over training seeds 0..5
             the ordered-minus-shuffled delta is +0.21/+0.34 on two seeds and negative on four
             (mean -0.30 pts), so the sign is chosen by the seed. The two numbers are still computed
             and printed, labelled as a null; the assert that rode on the shipped seed is DELETED
             (P8: a criterion the seed decides is not a measurement). Long-range/compositional
             syntax was already the stated ceiling; local order is now measured to be one too.

AUTONOMY: a SpikingCuriosityDrive picks which level to study next by spike-count learning-progress
(the frontier, not noise); a SpikingCurriculum graduates PHONETICS->SEMANTICS->SYNTAX when a level's
progress dries up. Learned phoneme/word assemblies are CONSOLIDATED to the durable substrate LAKE
(content-addressable, survives restart) -- the emergent skill is stored, not hand-coded.

HONEST SCOPE (do not oversell): phonetics is the high-confidence win; semantics is spiking
distributional co-occurrence over a bounded vocab (word neighbourhoods, not compositional meaning);
syntax is LOCAL order only. Cross-level binding (theta-gamma phase + global-workspace ignition) is
the NEXT slice, not built here. Next-token prediction is BELOW n-gram count baselines (a single-layer
spike-timing readout underperforms -- the readout-is-the-lever ceiling seen across this project) and
far below a transformer LM -- the value is the emergent, self-supervised, spiking assembly-formation
loop with measurable learning curves and un-riggable controls, NOT competence or SOTA perplexity.

  PYTHONPATH=src python -m snn_data_lake.cortex.language_self_build
  Knobs: NORD_CTX(3) NORD_NIN(48) NORD_CYCLES(12) NORD_PER(400) NORD_VOCAB(60) NORD_LIMIT(0=all)
         NORD_EVAL_N(2000; 0 = the full held-out tail)

PROMOTED reasoning/ -> cortex/ 2026-08-11 (#10 family settlement C6, executed BEFORE C4/C5 --
the recorded order was corrected by the measured imports: skill_builder and
language_readout_slda both import THIS module at module top, so it goes first). This module is
LIVE on the daemon's sleep tick (py-spy-proven: brain.tick -> _enter_sleep ->
nightly_skills.night -> Domain.study -> _positional_code; the study frame lived in
meta_skill_loop when spied, absorbed into nightly_skills at its 2026-08-11 retirement).
GATE-RUNTIME NOTE, per the
owner's not-toy-scale directive: the corpus, the mechanism, the controls and every assert are
UNCHANGED and real-scale; the only bound added at promotion is NORD_EVAL_N (default 2000) on
the HELD-OUT EVAL SAMPLE -- the 324 s pre-promotion runtime was ~1.1M spiking respond() calls
across six full-tail (~91k-pair) evaluations whose asserts' margins are decided at n=2000
within ~±1% binomial noise. NORD_EVAL_N=0 restores the full tail.
Deps: reuses src/ spiking primitives + substrate/proximity_replay (was
reasoning/brain_adapters/lake_memory until its 2026-08-11 promotion+rename).
"""

import os
import random
import tempfile

from snn_data_lake.neuromodulation.spiking_curiosity import SpikingCuriosityDrive, SpikingCurriculum
from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner
from snn_data_lake.sensorimotor.corpus_stream import (
    ARPABET,
    load_cmudict,
    load_text,
    phoneme_stream,
    word_stream,
)
from snn_data_lake.substrate.proximity_replay import LakeMemory

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_train_eval": (
        "LOCAL",
        "trains a SpikingInteractionLearner online on (context, next-token) pairs, self-supervised. "
        "The underlying rule is R-STDP, declared LOCAL at its own site",
    ),
}


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _positional_code(tokens, vindex, V, ctx):
    """A clean POSITIONAL one-hot context code: length V*ctx, most-recent token right-aligned to the
    last slot and given the largest value (recency), so rank-order coding fires the recent context
    first. No hashing -> no collisions; ORDER is encoded by the slot, so shuffling the tokens changes
    the code (the syntax control). OOV tokens contribute nothing (never faked)."""
    vec = [0.0] * (V * ctx)
    toks = list(tokens)[-ctx:]
    m = len(toks)
    for j, tok in enumerate(toks):  # j=0 oldest .. m-1 most recent
        idx = vindex.get(tok)
        if idx is None:
            continue
        slot = ctx - m + j  # right-align: most recent -> slot ctx-1
        vec[idx + slot * V] = float(slot + 1)  # recency weight -> recent fires earliest
    return vec


# distinct-from: snn_mnist_stdp_genn.py -- not a new train/eval mechanism; this module's own
# _train_eval gaining one eval-sample bound parameter at its 2026-08-11 promotion.
def _train_eval(
    pairs, responses, ctx, per, seed=0, shuffle_labels=False, shuffle_ctx=False, eval_n=0
):
    """Train a SpikingInteractionLearner online on (context_tokens, next_token) pairs, self-supervised
    (the label IS the real next token). Returns (first_third_acc, last_third_acc, final_acc, learner).
    Held-out accuracy is measured on a disjoint tail. shuffle_labels/shuffle_ctx drive the controls."""
    rset = list(responses)
    ridx = {r: i for i, r in enumerate(rset)}
    V = len(rset)
    n_in = V * ctx
    learner = SpikingInteractionLearner(rset, n_in=n_in, seed=seed, t_post=max(4.0, n_in / 4.0))
    rng = random.Random(seed + 7)
    usable = [(c, nx) for c, nx in pairs if nx in ridx]
    if len(usable) < 30:
        return 0.0, 0.0, 0.0, learner
    cut = int(len(usable) * 0.8)
    train, test = usable[:cut], usable[cut:]
    train = train[:per] if per else train
    # eval_n bounds the HELD-OUT EVAL SAMPLE only (contiguous head of the tail -- deterministic;
    # corpus order, no cherry-picking). 0 = the full tail. Added at the 2026-08-11 promotion so
    # the gate runs the same real corpus/mechanism/asserts without six ~91k-pair evaluations.
    if eval_n:
        test = test[:eval_n]

    def stim(ctxt):
        toks = list(ctxt)
        if shuffle_ctx:
            rng.shuffle(toks)
        return _positional_code(toks, ridx, V, ctx)

    # progress probe: held-out accuracy after the FIRST third vs the LAST third of training
    def heldout_acc():
        if not test:
            return 0.0
        ok = sum(learner.respond(stim(c)) == nx for c, nx in test)
        return ok / len(test)

    third = max(1, len(train) // 3)
    first_acc = last_acc = 0.0
    for i, (cx, nxt) in enumerate(train):
        label = rng.choice(rset) if shuffle_labels else nxt  # control: teach random labels
        learner.interact(stim(cx), label)
        if i == third:
            first_acc = heldout_acc()
    last_acc = heldout_acc()
    return first_acc, last_acc, last_acc, learner


def _cos_dist(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return 1.0 - dot / (na * nb)


def _assembly_separation(learner):
    """Inter/intra assembly separation ratio over the learned response units. Each unit's weight
    vector is its cell assembly's tuning; measure mean COSINE distance BETWEEN distinct assemblies vs
    a within-assembly jitter baseline. Ratio >> 1 => the assemblies acquired distinct, separated
    tunings (they are not all the same vector).

    Read the scale honestly: the intra baseline is a +/-0.01 jitter of the unit's OWN vector, so it
    is ~1e-3 and the ratio lands in the tens (measured: 56.9 phonetics, 52.1 semantics). The
    `> 1.2` assert in main() therefore has a reachable red only in the degenerate case where every
    assembly ends up the SAME vector (inter -> 0); it is a not-all-identical check, not a
    calibrated separation threshold, and it should not be read as one.

    (The `responses` parameter was dropped 2026-08-04: it was never read.)"""
    W = [list(n.w) for n in learner.neurons]
    inter = [_cos_dist(W[i], W[j]) for i in range(len(W)) for j in range(i + 1, len(W))]
    rng = random.Random(0)
    intra = [_cos_dist(w, [x + rng.uniform(-0.01, 0.01) for x in w]) for w in W]
    mi = sum(inter) / max(1, len(inter))
    ma = sum(intra) / max(1, len(intra))
    return mi / ma if ma > 1e-9 else float("inf")


def _frequent(words, k):
    from collections import Counter

    return [w for w, _ in Counter(words).most_common(k)]


def main():
    CTX = _envi("NORD_CTX", 3)
    PER = _envi("NORD_PER", 400)
    VOCAB = _envi("NORD_VOCAB", 60)
    LIMIT = _envi("NORD_LIMIT", 0) or None
    EVAL_N = _envi(
        "NORD_EVAL_N", 2000
    )  # held-out eval sample bound; 0 = full tail (pre-promotion behaviour)
    # tempfile.gettempdir(), not os.environ.get("TEMP", ".") -- the Windows-only TEMP variable is
    # unset on Linux/macOS CI so the "." fallback wrote into the repo checkout; the same defect
    # already fixed in lake99 / autonomous_brain, fixed here at the promotion.
    lake_path = os.path.join(tempfile.gettempdir(), "sndl_language_assemblies.spkl")
    if os.path.exists(lake_path):
        os.remove(lake_path)

    print("=" * 94)
    print(
        "LANGUAGE SELF-BUILD -- autonomous self-supervised spiking assemblies: phonetics/semantics/syntax"
    )
    print(
        f"ctx={CTX}  per={PER}  vocab={VOCAB}  n_in=V*ctx  (labels never used; next-token IS the signal)"
    )
    print("=" * 94)

    cmudict, dict_off = load_cmudict()
    words, text_off = load_text()
    print(
        f"[data] {len(words):,} word tokens ({'offline sample' if text_off else 'real corpus'}); "
        f"cmudict {len(cmudict):,} ({'offline' if dict_off else 'real'})"
    )

    ph_pairs = list(phoneme_stream(words, cmudict, k=CTX, limit=LIMIT))
    wd_pairs = list(word_stream(words, k=CTX, limit=LIMIT))
    freq_words = _frequent([nx for _, nx in wd_pairs], VOCAB)

    # ---- AUTONOMY: curiosity picks the level to study; curriculum graduates on progress dry-up ----
    curiosity = SpikingCuriosityDrive(seed=0)
    curriculum = SpikingCurriculum(n_stages=3, patience=2, seed=0)
    lake = LakeMemory(path=lake_path, seed=0)

    print("\n[PHONETICS] self-supervised next-phoneme (39 ARPABET assemblies), R-STDP...")
    ph_first, ph_last, ph_final, ph_learn = _train_eval(
        ph_pairs, ARPABET, CTX, PER, seed=0, eval_n=EVAL_N
    )
    ph_shuf = _train_eval(ph_pairs, ARPABET, CTX, PER, seed=0, shuffle_labels=True, eval_n=EVAL_N)[
        1
    ]
    ph_bigram = _bigram_baseline(ph_pairs)
    ph_sep = _assembly_separation(ph_learn)
    curiosity.observe("phonetics", 1.0 - ph_final)
    print(
        f"  next-phoneme held-out: first-third {ph_first:.1%} -> last-third {ph_last:.1%}  "
        f"(bigram {ph_bigram:.1%}, shuffled-teach control {ph_shuf:.1%})"
    )
    # "van Rossum" was WRONG here until 2026-08-04: _assembly_separation computes COSINE distance
    # (_cos_dist) over the response units' weight vectors and never touches a spike-train metric.
    # The label is the measurement's name, so a wrong one is a wrong report, not a typo.
    print(f"  assembly separation (inter/intra cosine): {ph_sep:.2f}")

    print(
        f"\n[SEMANTICS] self-supervised next-word over the {VOCAB} most frequent words, R-STDP..."
    )
    wd_first, wd_last, wd_final, wd_learn = _train_eval(
        wd_pairs, freq_words, CTX, PER, seed=1, eval_n=EVAL_N
    )
    wd_shuf = _train_eval(
        wd_pairs, freq_words, CTX, PER, seed=1, shuffle_labels=True, eval_n=EVAL_N
    )[1]
    wd_uni = _unigram_baseline(wd_pairs, freq_words)
    wd_sep = _assembly_separation(wd_learn)
    curiosity.observe("semantics", 1.0 - wd_final)
    print(
        f"  next-word held-out: first-third {wd_first:.1%} -> last-third {wd_last:.1%}  "
        f"(unigram {wd_uni:.1%}, shuffled-teach control {wd_shuf:.1%})"
    )
    print(f"  assembly separation: {wd_sep:.2f}  (vocab bounded at {VOCAB} -- honest scale limit)")

    print("\n[SYNTAX] order sensitivity: next-word with ORDER-PRESERVING vs SHUFFLED context...")
    _, syn_ordered, _, _ = _train_eval(wd_pairs, freq_words, CTX, PER, seed=2, eval_n=EVAL_N)
    _, syn_shuffled, _, _ = _train_eval(
        wd_pairs, freq_words, CTX, PER, seed=2, shuffle_ctx=True, eval_n=EVAL_N
    )
    print(
        f"  next-word acc: ordered context {syn_ordered:.1%}  vs  shuffled-order context {syn_shuffled:.1%}"
    )
    print(
        f"  order-sensitivity delta: {(syn_ordered - syn_shuffled) * 100:+.1f} pts "
        f"-- NULL: the sign is seed-dependent (see the seed sweep below), so this is NOT evidence "
        f"that the readout uses word order"
    )
    print(
        "  seed sweep 0..5 (measured 2026-08-04, PER=400): deltas -0.77 -0.34 +0.21 -0.17 -1.04 "
        "+0.34 pts -> 2 of 6 positive, mean -0.30 pts. The shipped seed=2 is one of the two that"
    )
    print(
        "  land positive. Reported as a null rather than a finding; see the SYNTAX note in main()."
    )

    # ---- CONSOLIDATE the learned assemblies to the durable lake (content-addressable, durable) ----
    for i, n in enumerate(ph_learn.neurons):
        key = list(n.w)[:60]
        lake.consolidate([key], payloads=[i])
    dur = lake.durability_check()
    curriculum.observe(curiosity.learning_progress("phonetics"))

    print("\n" + "=" * 94)
    print("LANGUAGE SELF-BUILD -- the skill built ITSELF (autonomous, self-supervised, spiking):")
    print("-" * 94)
    print(
        f"  PHONETICS : next-phoneme {ph_first:.1%} -> {ph_last:.1%}  (bigram {ph_bigram:.1%}, "
        f"shuffle {ph_shuf:.1%}); assembly sep {ph_sep:.2f}"
    )
    print(
        f"  SEMANTICS : next-word {wd_first:.1%} -> {wd_last:.1%}  (unigram {wd_uni:.1%}, "
        f"shuffle {wd_shuf:.1%}); assembly sep {wd_sep:.2f}"
    )
    print(
        f"  SYNTAX    : order-sensitivity {(syn_ordered - syn_shuffled) * 100:+.1f} pts "
        f"(ordered {syn_ordered:.1%} vs shuffled {syn_shuffled:.1%}) -- NULL, sign is seed-dependent"
    )
    print(
        f"  LAKE      : {lake.n_stored} phoneme assemblies consolidated; durable across restart = {dur['ok']}"
    )
    print("-" * 94)
    print("  honest scope: the local-order (syntax) result is a NULL, not a win; cross-level")
    print("  theta-gamma/workspace binding is the NEXT slice.")
    print("=" * 94)

    # ---- HONEST NEGATIVE (printed, not asserted): the single-layer spiking readout does NOT beat the
    #      n-gram count baselines (phonetics 7% vs bigram 20%; semantics near-unigram). This is the
    #      readout-is-the-lever ceiling seen across this project: rate/count readouts win on accuracy;
    #      a pure spike-timing readout underperforms. The WIN here is the autonomous, self-supervised,
    #      real-corpus ASSEMBLY-FORMATION loop (learning above the shuffle control + separated
    #      assemblies), NOT competence -- do not oversell it as beating n-grams. ----
    print(
        f"\n  HONEST: the spiking readout does NOT beat n-grams (phonetics {ph_last:.1%} vs bigram "
        f"{ph_bigram:.1%}; semantics {wd_last:.1%} vs unigram {wd_uni:.1%}) -- the readout-is-the-lever ceiling."
    )

    # ---- un-riggable self-checks: assemblies FORMED + learned above the shuffle control + durable ----
    assert ph_last > ph_first, (
        f"phonetics learning curve did not rise ({ph_first:.3f}->{ph_last:.3f})"
    )
    assert ph_last > ph_shuf + 0.02, (
        f"phonetics no better than the shuffled-teaching control ({ph_last:.3f} vs {ph_shuf:.3f})"
    )
    assert ph_sep > 1.2, f"phoneme assemblies not separated (inter/intra cos {ph_sep:.2f} <= 1.2)"
    assert wd_sep > 1.2, f"word assemblies not separated ({wd_sep:.2f} <= 1.2)"
    assert wd_last > wd_shuf, (
        f"semantics no better than shuffled-teaching control ({wd_last:.3f} vs {wd_shuf:.3f})"
    )
    # SYNTAX: the `assert syn_ordered >= syn_shuffled` that stood here is DELETED, not weakened.
    # Measured 2026-08-04 by sweeping the training seed 0..5 at the shipped PER=400 (every other
    # parameter unchanged), ordered-minus-shuffled next-word accuracy in points:
    #     seed 0 -0.77 | seed 1 -0.34 | seed 2 +0.21 | seed 3 -0.17 | seed 4 -1.04 | seed 5 +0.34
    # 2 of 6 positive, mean -0.30 pts. The assert passed only because the shipped seed=2 is one of
    # the two that land positive -- it goes RED on four of six seeds, which makes it a statement
    # about the seed rather than about word order. P8: a criterion whose verdict is chosen by the
    # seed is not a measurement, and the disposition for a null is deletion, not a lower threshold.
    # The two numbers are still COMPUTED and PRINTED above, labelled as the null they are: that is
    # the honest record. Re-establishing a real order-sensitivity claim needs a criterion that
    # survives a seed sweep (e.g. a paired sign test over >=10 seeds), which is open work.
    assert dur["ok"], "consolidated language assemblies did not survive the lake restart"
    print(
        f"self-check OK: assemblies FORMED + separated (phon {ph_sep:.1f}, sem {wd_sep:.1f}); phonetics "
        f"learned above the shuffle control ({ph_first:.1%}->{ph_last:.1%} vs shuffle {ph_shuf:.1%}); "
        f"SYNTAX order-sensitivity is a NULL (seed-dependent sign, mean -0.30 pts over seeds 0..5); "
        f"{lake.n_stored} assemblies durable -- the skill built itself, no "
        f"labels. Competence below n-grams (honest spiking-readout ceiling)."
    )
    lake.close()
    return 0


def _bigram_baseline(pairs):
    """Most-frequent-next-token given the immediate previous token; held-out on the same 20% tail."""
    from collections import Counter, defaultdict

    cut = int(len(pairs) * 0.8)
    train, test = pairs[:cut], pairs[cut:]
    tbl = defaultdict(Counter)
    for c, nx in train:
        tbl[c[-1] if c else None][nx] += 1
    if not test:
        return 0.0
    ok = 0
    for c, nx in test:
        prev = c[-1] if c else None
        if tbl[prev]:
            ok += tbl[prev].most_common(1)[0][0] == nx
    return ok / len(test)


def _unigram_baseline(pairs, responses):
    from collections import Counter

    rset = set(responses)
    cut = int(len(pairs) * 0.8)
    train, test = pairs[:cut], [(c, nx) for c, nx in pairs[cut:] if nx in rset]
    cnt = Counter(nx for _, nx in train if nx in rset)
    if not test or not cnt:
        return 0.0
    top = cnt.most_common(1)[0][0]
    return sum(nx == top for _, nx in test) / len(test)


if __name__ == "__main__":
    main()
