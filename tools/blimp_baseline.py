r"""Honest BLiMP baseline for the brain's language faculty AS IT STANDS TODAY (pre-build measurement).

Scorers, weakest to strongest existing mechanism:

  1. acquired-LM  -- faculties/language_acquisition.py, the online locally-plastic character predictor
                     (the "learn like a human, no backprop" learner). Sentence score = total
                     log-likelihood, computed READ-ONLY (no weight update during scoring, or the
                     benchmark would teach the model its own test).
  2. rose         -- faculties/rose_language.py comprehend(): accept/reject/gap over grounded structure.
  3. char-bigram  -- a plain bigram model fitted on the same exposure text. Reference point: if the
                     spiking learner cannot beat a bigram table, its language knowledge is n-gram or less.

Whatever these score is the honest "before" for the fluent-language ceiling, and it is the number
docs/DEPTH_CEILINGS.md reports under "Ceiling 2 -- fluent open-domain language / Before".

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88), so the "before"
number's instrument did not survive a clone -- the defect tools/citation_guard.py now gates. The BLiMP
root is an explicit path and an absent download SKIPs cleanly (exit 0) rather than failing.

Usage:
  python tools/blimp_baseline.py [N_PER_PARADIGM] [--blimp DIR]

  N_PER_PARADIGM  pairs per paradigm (default 50; the doc's table uses 25).
  --blimp DIR     BLiMP *.jsonl directory (default: <repo>/data/lang/blimp).
                  Download: https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("NORD_PERSIST", "0")

from snn_data_lake.reasoning.blimp_bench import load_pairs, paradigms, score  # noqa: E402


def pop_opt(argv, name, default):
    """Remove `--name VALUE` from argv in place and return VALUE, else default."""
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def build_acquired_lm():
    """Expose the online learner to text, then freeze it for read-only scoring."""
    # RESIDUE REMOVED 2026-08-04: this imported numpy and then did `_ = np  # numpy is a hard dep of
    # this path; surfaced explicitly`. Nothing in this function used it. That line is a zero-
    # dependency-era artifact -- announcing a dependency was the ritual when having one was a cost
    # -- and P7 (2026-08-04) abolished the rule it belonged to. An import whose only consumer is a
    # discard is dead code, and dead code that exists to make a point about dependencies is exactly
    # the residue P7 names.
    from snn_data_lake.faculties import language_acquisition as LA

    lm = LA.load_lm(seed=0, expose_rounds=3)

    def sentence_logprob(sentence):
        ids = [LA._STOI[LA._fold(c)] for c in sentence]
        ctx = [LA._STOI[" "]] * lm.ctx
        total = 0.0
        for t in ids:
            latent = lm._encode_ctx(ctx)  # forward only
            probs = LA._softmax(lm._logits(latent))  # no _learn_step -> W is never updated
            total += math.log(float(probs[t]) + 1e-12)
            ctx = ctx[1:] + [t]
        return total / max(1, len(ids))  # length-normalised, per BLiMP's LM method

    return sentence_logprob


def build_rose_scorer():
    from snn_data_lake.faculties.rose_language import RoseLanguage

    rose = RoseLanguage(seed=0)

    def s(sentence):
        try:
            out = rose.comprehend(sentence)
        except Exception:
            return 0.0
        if isinstance(out, dict):
            for k in ("score", "confidence", "goodness"):
                if isinstance(out.get(k), (int, float)):
                    return float(out[k])
            return {"accept": 1.0, "gap": 0.0, "reject": -1.0}.get(out.get("verdict"), 0.0)
        return 0.0

    return s


def build_bigram():
    """Reference model: a character bigram fitted on the same exposure text the learner saw."""
    from snn_data_lake.faculties import language_acquisition as LA

    text = ""
    for attr in ("SEED_TEXT", "CORPUS", "_TEXT", "TEXT"):
        v = getattr(LA, attr, None)
        if isinstance(v, str) and len(v) > 200:
            text = v
            break
    if not text:  # fall back to the same source load_lm uses, whatever it is
        text = " ".join(getattr(LA, "SENTENCES", []) or [])
    counts, uni = {}, {}
    prev = " "
    for ch in text.lower():
        c = LA._fold(ch)
        counts[(prev, c)] = counts.get((prev, c), 0) + 1
        uni[prev] = uni.get(prev, 0) + 1
        prev = c
    V = len(LA.VOCAB)

    def s(sentence):
        prev, total = " ", 0.0
        for ch in sentence.lower():
            c = LA._fold(ch)
            total += math.log((counts.get((prev, c), 0) + 1.0) / (uni.get(prev, 0) + V))
            prev = c
        return total / max(1, len(sentence))

    return s, len(text)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    blimp = pop_opt(argv, "--blimp", None)
    n = int(argv[0]) if argv else 50
    names = paradigms(blimp)
    # An absent dataset is not a failure. This used to `return 1`, which reddens a caller that was
    # only asking "is this reproducible here?" -- the answer is "not without the download".
    if not names:
        print(f"SKIP blimp_baseline: no BLiMP paradigms under {blimp or 'data/lang/blimp'}")
        print(
            "  download: https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip -> data/lang/blimp/"
        )
        return 0
    total_pairs = sum(len(load_pairs(x, root=blimp, limit=n)) for x in names)
    print(
        f"BLiMP baseline of the CURRENT brain -- {len(names)} paradigms x {n} pairs = {total_pairs}\n"
    )
    print("  anchors: chance 50.0% | 5-gram LM 61.2% | GPT-2 large 86.0% | human 88.6%\n")

    bigram, ntext = build_bigram()
    for name, mk in (
        (f"char-bigram (on {ntext} chars of exposure)", lambda: bigram),
        ("acquired-LM (language_acquisition)", build_acquired_lm),
        ("rose comprehend", build_rose_scorer),
    ):
        try:
            r = score(mk(), per_paradigm=n, root=blimp)
            print(f"  {name:44s}: {100 * r['accuracy']:5.1f}%  ({r['correct']}/{r['total']})")
        except Exception as e:
            print(f"  {name:44s}: ERROR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
