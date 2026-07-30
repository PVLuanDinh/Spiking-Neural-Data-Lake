r"""ar_lm_ngram_control -- a char n-gram fitted on the SAME exposure the spiking LM trains on.

This is the control that makes the spiking LM's BLiMP number interpretable, and it is the source of the
53.0% / held-out char PPL 4.56 bar that docs/DEPTH_CEILINGS.md calls "the best number in the table".
Both models are order-sensitive and both see the same text, so:

  spiking ~ n-gram   -> the spiking LM has learned char statistics and nothing beyond them; the limit is
                        what a local char model can express, and more of the same data will not move it.
  spiking > n-gram   -> the recurrence is buying something a fixed-window count model cannot.
  spiking < n-gram   -> the spiking model has not even reached the statistics of its own training data,
                        so the bottleneck is optimisation/capacity, not the benchmark.

Held-out char perplexity is reported on the SAME byte-disjoint slice the trainer selects on (proved
disjoint by tools/ar_lm_disjoint.py), so the two models' PPLs are directly comparable. Model selection is
on held-out PPL, never on BLiMP.

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88), so a published
number's control was unavailable to anyone who cloned the repo -- the defect tools/citation_guard.py now
gates. Both inputs are explicit paths and each is SKIPped with a message when absent.

Usage:
  python tools/ar_lm_ngram_control.py [FIT_CHARS] [PER_PARADIGM] [--corpus FILE] [--blimp DIR]

  FIT_CHARS      chars of exposure to fit on (default 10,000,000; the appendix reports 12e6).
  PER_PARADIGM   BLiMP pairs per paradigm (default 25 -> 67 x 25 = 1675 pairs).
  --corpus FILE  WikiText-103 plain text (default: <repo>/data/lang/wikitext/train0.txt).
  --blimp DIR    BLiMP *.jsonl directory   (default: <repo>/data/lang/blimp).
"""

import math
import os
import sys
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "training"))

import ar_spiking_lm as trainer  # noqa: E402

from snn_data_lake.reasoning.blimp_bench import length_scorer, paradigms, score  # noqa: E402

MAX_ORDER = 5  # char 5-gram: the order the published 61.2% anchor uses
ALPHA = 1.0  # recursive add-alpha (Dirichlet) backoff -- no free parameter tuned on BLiMP


def pop_opt(argv, name, default):
    """Remove `--name VALUE` from argv in place and return VALUE, else default."""
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


class CharNgram:
    """Counts + recursive add-alpha backoff. p_n(w|ctx) = (c(ctx,w) + A*p_(n-1)) / (c(ctx) + A)."""

    def __init__(self, order=MAX_ORDER):
        self.order = order
        self.ctx = [defaultdict(lambda: defaultdict(int)) for _ in range(order + 1)]
        self.tot = [defaultdict(int) for _ in range(order + 1)]
        self.vocab = set()
        self.V = 1

    def fit(self, text):
        self.vocab = set(text)
        n = len(text)
        for k in range(self.order + 1):
            cmap, tmap = self.ctx[k], self.tot[k]
            for i in range(k, n):
                c = text[i - k : i]
                cmap[c][text[i]] += 1
                tmap[c] += 1
            print(f"    order {k}: {len(tmap)} contexts", flush=True)
        self.V = max(1, len(self.vocab))
        return self

    def logp(self, ch, hist):
        p = 1.0 / self.V  # uniform base
        for k in range(0, self.order + 1):
            c = hist[len(hist) - k :] if k else ""
            if k and len(hist) < k:
                break
            t = self.tot[k].get(c, 0)
            if t == 0:
                continue
            num = self.ctx[k][c].get(ch, 0) + ALPHA * p
            p = num / (t + ALPHA)
        return math.log(max(p, 1e-300))

    def sent_logprob(self, s):
        tot, hist = 0.0, " "  # implicit space start-token, same convention as the serve
        for ch in s:
            tot += self.logp(ch, hist)
            hist = (hist + ch)[-self.order :]
        return tot

    def ppl(self, text):
        tot, hist, n = 0.0, " ", 0
        for ch in text:
            tot += self.logp(ch, hist)
            hist = (hist + ch)[-self.order :]
            n += 1
        return math.exp(-tot / max(1, n))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    corpus = pop_opt(argv, "--corpus", trainer.CORPUS_TXT)
    blimp = pop_opt(argv, "--blimp", None)
    fit_chars = int(float(argv[0])) if argv else 10_000_000
    per = int(argv[1]) if len(argv) > 1 else 25

    # Both inputs are gitignored local-only artifacts. Without these guards a clone would either
    # traceback out of the corpus reader or score BLiMP over zero pairs and print 0.0% as a result.
    if not os.path.exists(corpus):
        print(f"SKIP ar_lm_ngram_control: no corpus at {corpus}")
        print(
            "  WikiText-103: HF parquet wikitext-103-raw-v1/train-00000-of-00002.parquet -> data/lang/wikitext/"
        )
        return 0
    if not paradigms(blimp):
        print(f"SKIP ar_lm_ngram_control: no BLiMP paradigms under {blimp or 'data/lang/blimp'}")
        print(
            "  download: https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip -> data/lang/blimp/"
        )
        return 0

    print("=" * 100)
    print(f"CHAR {MAX_ORDER}-GRAM CONTROL on the same exposure (fit {fit_chars} chars)")
    print("anchors: chance 50.0 | length-only ~30 | published char/word 5-gram on 83M words 61.2")
    print("=" * 100)

    # the SAME byte-addressed reader, so the control sees the same text and the same held-out slice
    text = trainer.wikitext_prefix(int(fit_chars * 1.05), path=corpus)[:fit_chars]
    held = trainer.wikitext_heldout(path=corpus)[:200_000]
    print(f"fit on {len(text)} chars (vocab {len(set(text))}); held-out {len(held)} chars")

    t0 = time.time()
    ng = CharNgram().fit(text)
    print(f"fitted in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    p = ng.ppl(held)
    print(f"HELD-OUT char PPL : {p:.4f}   ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    r = score(ng.sent_logprob, per_paradigm=per, root=blimp)
    print(
        f"BLiMP  {MAX_ORDER}-gram    : {100 * r['accuracy']:5.1f}%  "
        f"({r['correct']}/{r['total']})  {time.time() - t0:.0f}s",
        flush=True,
    )
    rl = score(length_scorer, per_paradigm=per, root=blimp)
    print(f"BLiMP  length ctrl: {100 * rl['accuracy']:5.1f}%")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
