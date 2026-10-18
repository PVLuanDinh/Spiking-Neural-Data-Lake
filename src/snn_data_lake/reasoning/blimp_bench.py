r"""BLiMP scoring harness -- the honest language yardstick.

BLiMP (Warstadt et al. 2020, "BLiMP: The Benchmark of Linguistic Minimal Pairs for English",
TACL; github.com/alexwarstadt/blimp) is 67 paradigms x 1000 MINIMAL PAIRS. Each pair differs in
exactly one grammatical feature: one sentence is acceptable, one is not. A model is scored by
whether it assigns the acceptable sentence a higher score than the unacceptable one.

Why this benchmark and not a chat demo: it is AUTOMATIC and needs NO LLM JUDGE. It cannot be gamed
by fluent-sounding output, and it cannot be passed by a template bank, because the two sentences of a
pair share almost every word -- only the grammar differs.

    PYTHONPATH=src python -m snn_data_lake.reasoning.blimp_bench

Data: data/lang/blimp/*.jsonl (67 files). Download once:
    curl -sL -o blimp.zip https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip
and extract blimp-master/data/*.jsonl into data/lang/blimp/.

HONEST ANCHORS for reading any score:
  - chance is 50.0% (the self-check asserts a random scorer lands near it -- the control that must fail)
  - human agreement is 88.6% (Warstadt et al. 2020, Table 3)
  - the published 61.2% 5-gram anchor is a WORD-level model on 83M words, and GPT-2 large's 86.0% is
    word-level too (same paper, Table 3). NEITHER is the right yardstick for the CHARACTER-level models
    served here. The measured char-level bar on this repo's own exposure is 53.0% (a char 5-gram,
    held-out char PPL 4.56); see docs/DEPTH_CEILINGS.md.
  So: <=53% means "at or below a char n-gram", >=80% means "real syntax".

TWO SCORINGS, BOTH REPORTED. A full-sentence UNNORMALIZED sum-logprob carries a LENGTH BIAS: BLiMP's
grammatical member is usually the longer one, and a weak char LM pays a roughly constant cost per
character, so the sum systematically prefers the shorter (ungrammatical) member. Sentence length ALONE
scores far off chance here (see the length control), which is the size of the leak. main() therefore
prints EVERY model under BOTH sum-logprob and length-normalized (per-character) scoring, side by side.
Switching silently to the normalized one would make every number this bench printed before incomparable
without saying so; printing both keeps the record readable and puts the bias at the point of measurement.

Representation-only: imports no brainstem, routes no actuator.
"""

import importlib
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA = os.path.join(_ROOT, "data", "lang", "blimp")


def paradigms(root=None):
    """Return sorted paradigm names available on disk (empty if not downloaded)."""
    d = root or DATA
    if not os.path.isdir(d):
        return []
    return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl"))


def load_pairs(name, root=None, limit=None):
    """Return [(good, bad), ...] for one paradigm."""
    p = os.path.join(root or DATA, name + ".jsonl")
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append((r["sentence_good"], r["sentence_bad"]))
            if limit and len(out) >= limit:
                break
    return out


def score(scorer, per_paradigm=None, root=None, only=None, verbose=False):
    """Score a sentence-scorer over BLiMP.

    scorer(sentence) -> float, higher = more acceptable. Ties count as WRONG (a scorer that returns
    a constant must land at 0%, not 50% -- we never award credit for having no opinion).

    Returns {'accuracy','correct','total','by_paradigm','n_paradigms'}.
    """
    names = only or paradigms(root)
    by, correct, total = {}, 0, 0
    for name in names:
        pairs = load_pairs(name, root=root, limit=per_paradigm)
        c = 0
        for good, bad in pairs:
            try:
                if scorer(good) > scorer(bad):
                    c += 1
            except Exception:
                pass
        by[name] = (c / len(pairs)) if pairs else 0.0
        correct += c
        total += len(pairs)
        if verbose:
            print(f"  {name:45s} {100 * by[name]:5.1f}%")
    return {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "by_paradigm": by,
        "n_paradigms": len(names),
    }


def random_scorer(sentence, _rng=random.Random(0)):  # noqa: B008 -- ONE seeded Random bound at def time is the point: the control must be reproducible across calls
    """Control: must land near 50%."""
    return _rng.random()


def constant_scorer(sentence):
    """Control: no opinion. Ties count as wrong, so this must score 0%."""
    return 0.0


def length_scorer(sentence):
    """Control: pure surface heuristic. Reveals how much of a score is length leakage, not syntax."""
    return -len(sentence)


# Sentinel a sentence the served model could not score. Any real char-LM log-prob is orders of
# magnitude above it, so an unscorable sentence always LOSES its comparison instead of winning one.
UNSCORABLE = -1e18


def per_char(logprob):
    """Length-normalized rescoring: sum-logprob / characters. The other half of the pair reported.

    This removes the surface length term the unnormalized sum inherits (see the module docstring).
    Note what it does to the length control: -len(s)/len(s) is the SAME value for every sentence, so
    normalizing turns the length heuristic into a constant -- which is the mechanism, made visible.

    UNSCORABLE passes through UNNORMALIZED, and that is the honesty point: UNSCORABLE/len(s) would be
    a bigger number for a LONGER sentence, so a pair whose two sentences the model both failed to
    score would be decided by length -- credit awarded for having no opinion, which score() forbids.
    Passing it through keeps such a pair a tie, and ties count as wrong."""

    def scorer(sentence):
        lp = logprob(sentence)
        return lp if lp <= UNSCORABLE else lp / max(1, len(sentence))

    return scorer


def cached_logprob(serve):
    """Wrap serve.logprob in a per-sentence cache so BOTH scorings cost ONE forward pass per sentence.

    Honest-unavailable is preserved: a None logprob becomes a sentinel that always loses its comparison,
    never a fabricated number. The cache is a plain dict keyed by the sentence -- nothing is persisted,
    so no builtin hash() value is ever written down.

    The sentinel is UNSCORABLE and is NOT overridable. It used to be an `unscorable=` parameter, which
    made two sources of truth for one value: `per_char` recognises the sentinel by testing
    `lp <= UNSCORABLE` against the module constant, so a caller passing anything larger (say -1e9) got
    a sentinel per_char could not see, which then divided it by sentence length -- resurrecting exactly
    the length bug the constant exists to prevent, silently and only for that caller."""
    cache = {}

    def logprob(sentence):
        if sentence not in cache:
            lp = serve.logprob(sentence)
            cache[sentence] = UNSCORABLE if lp is None else lp
        return cache[sentence]

    return logprob


# The frozen char-LM serve this bench scores (the e-prop serves were retired by del-reasoning 1c). It is an ORDER-SENSITIVE
# full-sentence AR log-prob models (unlike the order-invariant SLDA-RFF language head).
SERVES = (
    ("AR spik-RWKV", "ar_spiking_lm_serve", "SpikingLMServe"),
)


def frozen_serves():
    """Return [(label, serve_or_None), ...] for every entry of SERVES.

    Unavailable serves are RETURNED, not dropped: models/ is gitignored, so a fresh clone has no
    artifacts and the table must print an explicit n/a instead of quietly showing fewer rows."""
    out = []
    for label, mod, cls in SERVES:
        try:
            m = importlib.import_module("snn_data_lake.reasoning." + mod)
            out.append((label, getattr(m, cls)()))
        except Exception:
            out.append((label, None))
    return out


def main():
    ok = True
    names = paradigms()
    if not names:
        print(f"SKIP blimp_bench: no BLiMP data at {DATA} (see module docstring for the download)")
        return 0

    print(f"blimp_bench: {len(names)} paradigms at {DATA}")
    # Pairs per paradigm. 25 keeps the self-check affordable now that it scores the real LM twice
    # each, and it is the sample size the numbers recorded in docs/DEPTH_CEILINGS.md were measured at,
    # so the printed table is directly comparable to the record. Full runs use per_paradigm=None.
    n = 25

    r = score(random_scorer, per_paradigm=n)
    print(f"  control random  : {100 * r['accuracy']:5.1f}%  ({r['correct']}/{r['total']})")
    if not (0.42 <= r["accuracy"] <= 0.58):
        print("  FAIL: random control did not land near chance -- the harness is biased")
        ok = False

    r = score(constant_scorer, per_paradigm=n)
    print(f"  control constant: {100 * r['accuracy']:5.1f}%  (ties must score 0, not 50)")
    if r["accuracy"] != 0.0:
        print("  FAIL: a no-opinion scorer earned credit")
        ok = False

    r = score(length_scorer, per_paradigm=n)
    len_acc, n_pairs = r["accuracy"], r["total"]
    print(f"  control length  : {100 * len_acc:5.1f}%  (surface-length leakage baseline)")
    if r["accuracy"] > 0.75:
        print("  FAIL: sentence length alone solves BLiMP; the loaded pairs are not minimal")
        ok = False

    # An oracle that can see which sentence is the good one must score 100%.
    goods = set()
    for name in names[:5]:
        for g, _b in load_pairs(name, limit=n):
            goods.add(g)
    r = score(lambda s: 1.0 if s in goods else 0.0, per_paradigm=n, only=names[:5])
    print(f"  oracle          : {100 * r['accuracy']:5.1f}%")
    if r["accuracy"] < 0.999:
        print("  FAIL: oracle did not score 100% -- the harness cannot register a correct answer")
        ok = False

    # OPT-IN real scorers: every frozen order-sensitive char LM, scored BOTH ways. These are
    # MEASUREMENTS, not controls -- no pass/fail assertion, and no model is preferred over another
    # (each gets its own row, so a weaker one cannot hide behind a stronger one's number). A model
    # whose artifact is absent prints n/a: models/ is gitignored, and nothing is ever fabricated.
    print(f"  real scorers, BOTH scorings ({n_pairs} pairs = {len(names)} x {n}):")
    print(f"    {'model':<15}{'sum-logprob':>12}{'per-char':>10}{'delta':>8}")
    print(
        f"    {'length only':<15}{100 * len_acc:>11.1f}%{'const':>10}{'--':>8}"
        "   surface length, no model"
    )
    for label, serve in frozen_serves():
        if serve is None or not serve.available():
            print(f"    {label:<15}{'n/a':>12}{'n/a':>10}{'--':>8}   no frozen artifact")
            continue
        logprob = cached_logprob(serve)
        a_sum = score(logprob, per_paradigm=n)["accuracy"]
        a_pc = score(per_char(logprob), per_paradigm=n)["accuracy"]
        print(
            f"    {label:<15}{100 * a_sum:>11.1f}%{100 * a_pc:>9.1f}%{100 * (a_pc - a_sum):>+8.1f}"
        )
    print(
        "    read BOTH: the length-only control is far off chance, so the sum-logprob column carries a"
    )
    print(
        "    length bias in unknown proportion; per-char removes the surface term (and makes the length"
    )
    print(
        "    control a constant, hence 'const'). Neither column is 'the' number -- char-level bar 53.0%."
    )

    print("blimp_bench self-check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
