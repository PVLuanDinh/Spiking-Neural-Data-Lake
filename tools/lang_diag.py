r"""MECHANISM DIAGNOSIS -- why does the local-rule character learner sit near chance on BLiMP?

Seven targeted probes, one per candidate cause named in the brief. Each one is a measurement, not
an argument. Nothing in src/ is modified; the only variant knob is `decay`, which reproduces the
hard-coded 1e-5 by default.

  D1  homeostatic decay      -- is `W *= (1 - 1e-5)` erasing what is learned at scale?
  D2  representational level -- where does the acquired LM sit on the n-gram ladder?
  D3  vocabulary folding     -- does the 34-char fold destroy the distinctions BLiMP tests?
  D4  context window         -- is ctx=8 characters the binding constraint?
  D5  decision margins       -- how strong an opinion does the model even have per pair?
  D6  feature injectivity    -- can the 88-dim feature tell two contexts apart at all?
  D7  capacity ceiling       -- the BEST linear readout this feature can support, fit to convergence

These seven probes are the "falsified by ablation" table in docs/DEPTH_CEILINGS.md ("The mechanism,
established by measurement").

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88), so the ablation
table cited an instrument absent from every clone -- the defect tools/citation_guard.py now gates. Inputs
are resolved by tools/lang_scale.py (this module imports its readers), so the same --corpus / --blimp
flags apply and an absent input SKIPs with a message.

Usage:
  python tools/lang_diag.py [--corpus FILE] [--blimp DIR]
"""

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.environ.setdefault("NORD_PERSIST", "0")

import lang_scale  # noqa: E402
from lang_scale import (  # noqa: E402
    BLIMP_PER_PARADIGM,
    BatchEncoder,
    apply_local_rule,
    encode_stream,
    fit_ngrams,
    fold_ids,
    logprobs_readonly,
    missing_inputs,
    ngram_logprob_seq,
    pop_opt,
    read_corpus,
    read_heldout,
    set_inputs,
)

from snn_data_lake.cortex import language_acquisition as LA  # noqa: E402
from snn_data_lake.reasoning.blimp_bench import load_pairs, paradigms, score  # noqa: E402

V = LA.V
TRAIN_N = 1_000_000
SWEEP_N = 300_000


def rule(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


class Bench:
    """BLiMP sentences + their (W-independent) latents, precomputed once per context length."""

    def __init__(self, ctx=8):
        la = LA.LanguageAcquisition(seed=0, ctx=ctx)
        self.enc = BatchEncoder(la)
        self.names = paradigms(lang_scale.BLIMP_DIR)
        self.pairs = []
        sents = []
        for nm in self.names:
            for g, b in load_pairs(nm, root=lang_scale.BLIMP_DIR, limit=BLIMP_PER_PARADIGM):
                self.pairs.append((g, b))
                sents.append(g)
                sents.append(b)
        self.sents = list(dict.fromkeys(sents))
        self.sent_ids = [fold_ids(s) for s in self.sents]
        self.offs, cur = [], 0
        for si in self.sent_ids:
            self.offs.append((cur, cur + len(si)))
            cur += len(si)
        self.all_ids = np.concatenate(self.sent_ids)
        self.lat = np.empty((cur, self.enc.d_in + self.enc.d_r2))
        for (a, b), si in zip(self.offs, self.sent_ids, strict=False):
            self.lat[a:b] = encode_stream(self.enc, si)

    def table(self, W):
        lp = logprobs_readonly(W, self.lat, self.all_ids)  # READ-ONLY
        return {
            s: float(lp[a:b].sum()) / max(1, b - a)
            for s, (a, b) in zip(self.sents, self.offs, strict=False)
        }

    def acc(self, W):
        t = self.table(W)
        return (
            100.0
            * score(lambda s: t[s], per_paradigm=BLIMP_PER_PARADIGM, root=lang_scale.BLIMP_DIR)[
                "accuracy"
            ]
        )


def train(bench, ids, decay=1e-5, ctx=8):
    la = LA.LanguageAcquisition(seed=0, ctx=ctx)
    carry = None
    for i in range(0, len(ids), 200_000):
        part = ids[i : i + 200_000]
        lat = encode_stream(bench.enc, part, carry)
        apply_local_rule(la.W, lat, part, la.lr, decay=decay)
        carry = part[-ctx:] if len(part) >= ctx else carry
    return la


def ppl(W, lat, ids):
    return math.exp(-logprobs_readonly(W, lat, ids).mean())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    set_inputs(pop_opt(argv, "--corpus", None), pop_opt(argv, "--blimp", None))
    why = missing_inputs()
    if why:
        print(f"SKIP lang_diag: {why}")
        return 0
    t0 = time.perf_counter()
    text = read_corpus(TRAIN_N)
    ids = fold_ids(text)
    heldout = read_heldout()
    ho_ids = fold_ids(heldout)
    bench = Bench(ctx=8)
    ho_lat = encode_stream(bench.enc, ho_ids)
    print(
        f"setup: {len(ids)} exposure chars, {len(ho_ids)} held-out chars, "
        f"{len(bench.sents)} BLiMP sentences  [{time.perf_counter() - t0:.1f}s]"
    )

    # ---------------------------------------------------------------------------------- D1 decay
    rule("D1  HOMEOSTATIC DECAY  -- W *= (1 - 1e-5) after every character")
    print("  A contribution made N characters ago survives with weight (1-1e-5)^N:")
    for n in (1e4, 1e5, 1e6, 1e7):
        print(f"    after {n:10.0e} chars : {(1 - 1e-5) ** n:12.3e} of it remains")
    print(f"  => effective memory time constant = 1/1e-5 = {1 / 1e-5:,.0f} characters.")
    print("     Past ~1e5 chars of exposure the readout is a function of only the RECENT window,")
    print("     so more corpus cannot accumulate.  Ablation (same text, decay switched off):\n")
    base = train(bench, ids, decay=1e-5)
    nodec = train(bench, ids, decay=0.0)
    print(f"    {'variant':28s} {'BLiMP %':>9} {'held-out PPL':>14}")
    print(
        f"    {'decay 1e-5 (as shipped)':28s} {bench.acc(base.W):9.1f} {ppl(base.W, ho_lat, ho_ids):14.2f}"
    )
    print(
        f"    {'decay 0 (ablated)':28s} {bench.acc(nodec.W):9.1f} {ppl(nodec.W, ho_lat, ho_ids):14.2f}"
    )

    # ------------------------------------------------------------------------- D2 n-gram ladder
    rule("D2  REPRESENTATIONAL LEVEL -- held-out PPL of the acquired LM vs the n-gram ladder")
    models = fit_ngrams(ids, 5)
    print(f"    {'model':34s} {'held-out PPL':>14} {'free parameters':>18}")
    print(f"    {'uniform over 34 chars':34s} {float(V):14.2f} {0:18,d}")
    for k in range(1, 6):
        p = math.exp(-ngram_logprob_seq(models, ho_ids, k) / len(ho_ids))
        print(
            f"    {'char ' + str(k) + '-gram (same 1e6 text)':34s} {p:14.2f} {len(models[k - 1][1]):18,d}"
        )
    print(
        f"    {'ACQUIRED LM (local rule)':34s} {ppl(base.W, ho_lat, ho_ids):14.2f} "
        f"{base.W.size:18,d}"
    )

    # ------------------------------------------------------------------------- D3 vocab folding
    rule("D3  VOCABULARY FOLDING -- do the BLiMP minimal pairs survive the 34-char fold?")
    same = 0
    for g, b in bench.pairs:
        if np.array_equal(fold_ids(g), fold_ids(b)):
            same += 1
    n = len(bench.pairs)
    print(
        f"    pairs collapsed to an identical character string : {same}/{n} ({100.0 * same / n:.2f}%)"
    )
    print(
        f"    pairs still distinguishable after folding        : {n - same}/{n} "
        f"({100.0 * (n - same) / n:.2f}%)"
    )
    print("    => the fold is NOT the cap: virtually every pair is still separable in principle.")

    # ------------------------------------------------------------------------ D4 context window
    rule("D4  CONTEXT WINDOW -- vary ctx, hold text and rule fixed (300k chars)")
    ids_s = ids[:SWEEP_N]
    print(f"    {'ctx (chars)':>12} {'BLiMP %':>9} {'held-out PPL':>14}")
    for c in (2, 4, 8, 16, 32):
        bn = Bench(ctx=c)
        ho_l = encode_stream(bn.enc, ho_ids)
        la_c = train(bn, ids_s, ctx=c)
        print(f"    {c:12d} {bn.acc(la_c.W):9.1f} {ppl(la_c.W, ho_l, ho_ids):14.2f}", flush=True)

    # ----------------------------------------------------------------------------- D5 margins
    rule("D5  DECISION MARGINS -- how large an opinion does each model have per minimal pair?")
    acq_t = bench.table(base.W)
    ng_t = {
        s: ngram_logprob_seq(models, si, 5) / max(1, len(si))
        for s, si in zip(bench.sents, bench.sent_ids, strict=False)
    }
    for nm, tb in (("acquired LM", acq_t), ("char 5-gram", ng_t)):
        m = np.array([abs(tb[g] - tb[b]) for g, b in bench.pairs])
        print(
            f"    {nm:14s} |score(good)-score(bad)| : median {np.median(m):.5f}  "
            f"mean {m.mean():.5f}  p90 {np.percentile(m, 90):.5f}"
        )

    # ------------------------------------------------------------------------ D6 injectivity
    rule("D6  FEATURE INJECTIVITY -- can the 88-dim feature even tell two contexts apart?")
    ctxm = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.full(8, LA._STOI[" "], dtype=np.int64), ids[:200_000]]), 8
    )[:200_000]
    uc = len(np.unique(ctxm, axis=0))
    lat = encode_stream(bench.enc, ids[:200_000])
    ul = len(np.unique(lat.round(12), axis=0))
    print(f"    distinct 8-char contexts in 200k chars : {uc:,}")
    print(f"    distinct 88-dim features they produce  : {ul:,}")
    print(f"    collision rate                         : {100.0 * (1 - ul / uc):.2f}%")
    print(
        f"    but the readout is LINEAR on 88 dims -> at most {bench.enc.d_in + bench.enc.d_r2} "
        f"independent directions,"
    )
    print(
        f"    i.e. {V} x {bench.enc.d_in + bench.enc.d_r2} = {V * (bench.enc.d_in + bench.enc.d_r2):,} "
        f"free parameters TOTAL, no matter how much text is seen."
    )

    # --------------------------------------------------------------------------- D7 ceiling
    rule("D7  CAPACITY CEILING -- the BEST linear readout this feature admits, fit to convergence")
    print("    (same single-layer delta rule, but full-batch and run to convergence with no decay:")
    print("     the convex optimum of the shipped architecture -- an upper bound on the rule.)")
    lat_tr = encode_stream(bench.enc, ids[:SWEEP_N])
    tgt = ids[:SWEEP_N]
    W = np.zeros((V, lat_tr.shape[1]))
    onehot = np.zeros((len(tgt), V))
    onehot[np.arange(len(tgt)), tgt] = 1.0
    for ep in range(1, 401):
        z = lat_tr @ W.T
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        g = (onehot - p).T @ lat_tr / len(tgt)
        W += 2.0 * g
        if ep in (1, 50, 100, 200, 400):
            nll = -np.log(p[np.arange(len(tgt)), tgt] + 1e-12).mean()
            print(
                f"      epoch {ep:4d}  train NLL {nll:.4f}  train PPL {math.exp(nll):7.3f}  "
                f"BLiMP {bench.acc(W):5.1f}%  held-out PPL {ppl(W, ho_lat, ho_ids):7.3f}",
                flush=True,
            )
    print(f"\n    optimal-linear-readout BLiMP = {bench.acc(W):.1f}%  vs chance 50.0%")
    print(f"    online local rule as shipped = {bench.acc(base.W):.1f}%")
    print("    => if these agree, the LEARNING RULE is not the cap; the FEATURE+LINEAR READOUT is.")

    print(f"\n[total {time.perf_counter() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
