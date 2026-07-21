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
  - a 5-gram LM trained on 83M words scores ~61%; GPT-2 large scores ~86% (same paper, Table 3)
  So: <=55% means "no grammatical knowledge", ~61% means "n-gram level", >=80% means "real syntax".

Representation-only: imports no brainstem, routes no actuator.
"""

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


def random_scorer(sentence, _rng=random.Random(0)):
    """Control: must land near 50%."""
    return _rng.random()


def constant_scorer(sentence):
    """Control: no opinion. Ties count as wrong, so this must score 0%."""
    return 0.0


def length_scorer(sentence):
    """Control: pure surface heuristic. Reveals how much of a score is length leakage, not syntax."""
    return -len(sentence)


def main():
    ok = True
    names = paradigms()
    if not names:
        print(f"SKIP blimp_bench: no BLiMP data at {DATA} (see module docstring for the download)")
        return 0

    print(f"blimp_bench: {len(names)} paradigms at {DATA}")
    n = 50  # per paradigm, keeps the self-check fast; full runs use per_paradigm=None

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
    print(f"  control length  : {100 * r['accuracy']:5.1f}%  (surface-length leakage baseline)")
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

    # OPT-IN real scorer: the order-sensitive AR Spiking-RWKV LM (the actual BLiMP lever). This is a
    # MEASUREMENT, not a control -- no pass/fail assertion. It stays honestly dormant until a frozen
    # artifact (models/ar_spiking_lm.npz) exists; moving BLiMP off the ~57% floor needs the real-scale
    # GPU train (training/ar_spiking_lm.py). Never fabricates: unavailable -> a printed SKIP, no number.
    try:
        serve, label = None, "e-prop spik-LM"
        try:                                         # PREFER the deep-ALIF long-range LM (W4b architectural upgrade)
            from snn_data_lake.reasoning.eprop_lm_alif_serve import EpropAlifLMServe
            _alif = EpropAlifLMServe()
            if _alif.available():
                serve, label = _alif, "e-prop ALIF-LM"
        except Exception:
            pass
        if serve is None:                            # else the flat single-layer e-prop LM
            from snn_data_lake.reasoning.eprop_lm_serve import EpropLMServe
            serve = EpropLMServe()
        if serve.available():
            def _eprop_scorer(s):                    # one forward pass per sentence; None -> -inf, never faked
                lp = serve.logprob(s)
                return lp if lp is not None else -1e18
            r = score(_eprop_scorer, per_paradigm=n)
            print(f"  {label:<15}: {100 * r['accuracy']:5.1f}%  ({r['correct']}/{r['total']}) [brain-native, order-sensitive logprob]")
        else:
            print("  e-prop spik-LM  :  n/a  (no frozen artifact -- needs the real-scale e-prop GPU train)")
    except Exception as exc:
        print(f"  e-prop spik-LM  :  n/a  ({type(exc).__name__}: serve unavailable)")

    print("blimp_bench self-check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
