r"""CORPUS-SCALE EXPERIMENT for the online, locally-plastic character learner.

Question: does cortex/language_acquisition.py::LanguageAcquisition actually ACQUIRE grammar when it
is given a REAL corpus instead of its 2,428-character built-in CORPUS -- and where does it plateau?

The learner is exposed ONLINE to increasing prefixes of WikiText-103 (real text). At each scale we
measure, all on the SAME text:
  * BLiMP accuracy of the acquired LM, scored READ-ONLY (no weight update during scoring)
  * BLiMP accuracy of a char-bigram and a char-5-gram fitted on the identical prefix
  * held-out per-character perplexity (a disjoint slice of the corpus, never exposed)

NOTHING in src/ is modified. This script only READS the learner.

-- WHY THIS IS FAST, AND WHY IT IS STILL THE REAL LEARNER ------------------------------------------
The naive path runs at 280 chars/sec (measured, scratchpad/prof_encode.py), i.e. ~10 h for 1e7 chars.
Two exact observations make it ~300x faster WITHOUT changing the model:

  (1) LanguageAcquisition._encode_ctx(ids) is a PURE DETERMINISTIC FUNCTION of the 8 context ids.
      The predictive-coding network's generative weights w1/w2 are never learned (_pc.learn() is
      never called anywhere), _pc.reset() clears all state before every infer(), and infer() only
      touches r1/r2/membranes. So the latent depends on nothing but the context characters.
  (2) Therefore the ENTIRE LATENT SEQUENCE for a text is fixed by the text alone -- it does not
      depend on the readout W. So the latents can be precomputed in a vectorised numpy batch, and
      the LOCAL plasticity rule (dW = lr * outer(err, latent), then the homeostatic decay) is then
      replayed sequentially, character by character, in exactly the original order.

The vectorised settling loop is NO LONGER duplicated here: it was promoted into src and is now
PredictiveCodingNetwork.infer_batch (same LIF membrane update order, same reset-by-subtraction, same
6 settle steps + 1 lr=0 readout step). All that remains in this file is the LA-specific input
encoding (the position-tagged grapheme bag).
`--verify` checks it against the untouched original: latent agreement AND end-to-end equality of the
learned W after exposing the same text through both paths. Run --verify before believing any number.

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88), so the
scale table in docs/DEPTH_CEILINGS.md ("The mechanism, established by measurement") cited an instrument
absent from every clone -- the defect tools/citation_guard.py now gates. Both inputs are explicit paths
(flag or env) and an absent input SKIPs with a message instead of reporting a number from no data.
tools/lang_diag.py and tools/lang_probe.py import this module's readers, so setting the paths here sets
them for all three.

Usage:
  python tools/lang_scale.py --verify
  python tools/lang_scale.py 1e3 1e4 1e5 1e6 1e7 [--corpus FILE] [--blimp DIR]

  --corpus FILE  WikiText-103 plain text (default $NORD_LANG_CORPUS, else
                 <repo>/data/lang/wikitext/train0.txt). Must be >= 200,100,000 bytes: the held-out
                 slice is read at byte 200,000,000 and a short file would silently hold out nothing.
  --blimp DIR    BLiMP *.jsonl directory (default $NORD_BLIMP_DIR, else <repo>/data/lang/blimp).
"""

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("NORD_PERSIST", "0")

from snn_data_lake.cortex import language_acquisition as LA  # noqa: E402
from snn_data_lake.reasoning.blimp_bench import load_pairs, paradigms, score  # noqa: E402

# Module-level because tools/lang_diag.py and tools/lang_probe.py import read_corpus/read_heldout from
# here and must see the same inputs; set_inputs() below is how their CLIs point all three at one corpus.
CORPUS_TXT = os.environ.get("NORD_LANG_CORPUS") or os.path.join(
    _ROOT, "data", "lang", "wikitext", "train0.txt"
)
BLIMP_DIR = os.environ.get("NORD_BLIMP_DIR") or None  # None -> blimp_bench.DATA

V = LA.V
SPACE = LA._STOI[" "]
HELDOUT_BYTE_OFFSET = 200_000_000  # disjoint from every exposure prefix we use
HELDOUT_CHARS = 100_000
BLIMP_PER_PARADIGM = 25  # 67 x 25 = 1675 pairs -- same size as the v3.x baseline
NGRAM_MAX_ORDER = 5
NGRAM_MAX_FIT = 10_000_000  # past this the n-gram count tables exceed RAM; printed as --
ALPHA = 1.0  # recursive add-alpha backoff (Dirichlet interpolation)


# ---------------------------------------------------------------------------------------------
# character folding -- an exact vectorised replica of LA._fold (lower(), then non-vocab -> space)
# ---------------------------------------------------------------------------------------------
def _build_fold_lut():
    lut = np.full(65537, SPACE, dtype=np.int64)
    for cp in range(65536):
        ch = chr(cp).lower()  # may return >1 char; then it is not in _STOI -> stays space
        if ch in LA._STOI:
            lut[cp] = LA._STOI[ch]
    return lut


_FOLD_LUT = _build_fold_lut()


def fold_ids(s):
    """text -> int64 array of folded vocab ids, one per character. Matches LA._fold exactly."""
    cp = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
    return _FOLD_LUT[np.minimum(cp, 65536)]


# ---------------------------------------------------------------------------------------------
# batched predictive-coding encoder -- the LA-specific input bag; the settling loop is src's
# ---------------------------------------------------------------------------------------------
class BatchEncoder:
    """Batched, exact replica of LanguageAcquisition._encode_ctx for many contexts at once.

    Holds a REFERENCE to the learner's own predictive-coding network and defers the settling to
    PredictiveCodingNetwork.infer_batch. That is sound because LA never calls _pc.learn(), so w1/w2
    are frozen for the lifetime of the learner (see the module docstring)."""

    def __init__(self, la):
        self._pc = la._pc
        self.d_in = self._pc.d_in
        self.d_r2 = self._pc.d_r2
        self.ctx = la.ctx
        # bucket index (c*7 + p*13) % d_in, precomputed for every (char id, position) pair
        self.bucket = np.array(
            [[(c * 7 + p * 13) % self.d_in for p in range(self.ctx)] for c in range(V)],
            dtype=np.int64,
        )

    def _inputs(self, ctx_ids):
        """(B, ctx) ids -> (B, d_in) normalised position-tagged grapheme bag (LA._encode_ctx head)."""
        b = ctx_ids.shape[0]
        x = np.zeros((b, self.d_in), dtype=np.float64)
        pos = np.arange(self.ctx)
        idx = self.bucket[ctx_ids, pos]  # (B, ctx)
        np.add.at(x, (np.repeat(np.arange(b), self.ctx), idx.ravel()), 1.0)
        m = x.max(axis=1)
        m[m == 0.0] = 1.0  # `max(x) or 1.0`
        return x / m[:, None]

    def encode(self, ctx_ids):
        """(B, ctx) ids -> (B, d_in + d_r2) latents, identical to stacking _encode_ctx row by row."""
        x = self._inputs(ctx_ids)
        _, r2 = self._pc.infer_batch(x, steps=6, lr=0.05)  # LA calls infer(steps=6, lr=0.05)
        return np.concatenate([x, r2], axis=1)


def contexts_from_ids(ids, carry=None, ctx=8):
    """ids (N,) -> (N, ctx) context matrix; context at t is the ctx chars BEFORE t."""
    if carry is None:
        carry = np.full(ctx, SPACE, dtype=np.int64)
    padded = np.concatenate([carry, ids])
    return np.lib.stride_tricks.sliding_window_view(padded, ctx)[: len(ids)]


def encode_stream(enc, ids, carry=None, batch=16384):
    out = np.empty((len(ids), enc.d_in + enc.d_r2), dtype=np.float64)
    ctxm = contexts_from_ids(ids, carry, enc.ctx)
    for i in range(0, len(ids), batch):
        out[i : i + batch] = enc.encode(ctxm[i : i + batch])
    return out


# ---------------------------------------------------------------------------------------------
# the LOCAL plasticity rule, replayed sequentially over precomputed latents (exact, in order)
# ---------------------------------------------------------------------------------------------
def apply_local_rule(W, latents, targets, lr, decay=1e-5):
    """Literal replay of LanguageAcquisition._learn_step over a precomputed latent sequence.

    `decay` is the homeostatic term; 1e-5 is the value hard-coded in src and is the default, so the
    default call is byte-for-byte the original rule. The diagnostics vary it to test its effect."""
    keep = 1.0 - decay
    tot = 0.0
    for t in range(len(targets)):
        lat = latents[t]
        z = W @ lat
        z = z - z.max()
        e = np.exp(z)
        probs = e / (e.sum() + 1e-12)
        tid = targets[t]
        tot += -math.log(probs[tid] + 1e-12)
        err = -probs
        err[tid] += 1.0
        W += lr * np.outer(err, lat)
        W *= keep
    return tot / max(1, len(targets))


def logprobs_readonly(W, latents, targets):
    """Read-only per-character log p(target). No weight update anywhere -- the scoring path."""
    z = latents @ W.T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / (e.sum(axis=1, keepdims=True) + 1e-12)
    return np.log(p[np.arange(len(targets)), targets] + 1e-12)


# ---------------------------------------------------------------------------------------------
# n-gram references, fitted on the SAME text, recursive add-alpha backoff to uniform
# ---------------------------------------------------------------------------------------------
def fit_ngrams(ids, max_order=NGRAM_MAX_ORDER):
    models = []
    for k in range(1, max_order + 1):
        if k == 1:
            ctxk = np.zeros(len(ids), dtype=np.int64)
            full = ids.astype(np.int64)
        else:
            win = np.lib.stride_tricks.sliding_window_view(ids, k)
            powers = (V ** np.arange(k - 2, -1, -1)).astype(np.int64)
            ctxk = win[:, : k - 1] @ powers
            full = ctxk * V + win[:, k - 1]
        u, c = np.unique(full, return_counts=True)
        fc = dict(zip(u.tolist(), c.tolist(), strict=False))
        u2, c2 = np.unique(ctxk, return_counts=True)
        cc = dict(zip(u2.tolist(), c2.tolist(), strict=False))
        models.append((cc, fc))
    return models


def ngram_logprob_seq(models, seq_ids, order):
    """Sum of log p(c_t | history) with recursive add-alpha backoff; history padded with spaces."""
    hist = [SPACE] * (order - 1)
    total = 0.0
    for c in seq_ids:
        p = 1.0 / V
        for k in range(1, order + 1):
            cc, fc = models[k - 1]
            if k == 1:
                key = 0
            else:
                key = 0
                for h in hist[len(hist) - (k - 1) :]:
                    key = key * V + h
            tot = cc.get(key, 0)
            cnt = fc.get(key * V + int(c), 0)
            p = (cnt + ALPHA * p) / (tot + ALPHA)
        total += math.log(p)
        if order > 1:
            hist = hist[1:] + [int(c)]
    return total


# ---------------------------------------------------------------------------------------------
# verification: the vectorised path must reproduce the untouched original
# ---------------------------------------------------------------------------------------------
def verify(text):
    print("=" * 96)
    print("VERIFY -- vectorised path vs the untouched src/ implementation")
    print("=" * 96)
    la = LA.LanguageAcquisition(seed=0)
    enc = BatchEncoder(la)

    # (a) latent agreement on real contexts
    ids = fold_ids(text[:4000])
    ctxm = contexts_from_ids(ids)
    ref = np.array([la._encode_ctx(list(map(int, row))) for row in ctxm[:2000]])
    got = enc.encode(ctxm[:2000])
    d = np.abs(ref - got)
    print(f"  latents compared        : {ref.shape[0]} contexts x {ref.shape[1]} dims")
    print(f"  max abs difference      : {d.max():.3e}")
    print(f"  rows bitwise-identical  : {int((d.max(axis=1) == 0).sum())}/{ref.shape[0]}")

    # (b) end-to-end: expose the same text through both paths, compare the learned readout W
    n = 3000
    ids_e = fold_ids(text[:n])
    ref_la = LA.LanguageAcquisition(seed=0)
    t = time.perf_counter()
    ref_surprise = ref_la.expose(text[:n])
    t_ref = time.perf_counter() - t

    fast_la = LA.LanguageAcquisition(seed=0)
    t = time.perf_counter()
    lat = encode_stream(enc, ids_e)
    fast_surprise = apply_local_rule(fast_la.W, lat, ids_e, fast_la.lr)
    t_fast = time.perf_counter() - t

    dW = np.abs(ref_la.W - fast_la.W)
    print(f"  exposed both paths to   : {n} chars of real corpus")
    print(f"  mean surprise ref/fast  : {ref_surprise:.10f} / {fast_surprise:.10f}")
    print(f"  max abs dW              : {dW.max():.3e}   (max |W| = {np.abs(ref_la.W).max():.3f})")
    print(f"  wall clock ref/fast     : {t_ref:.2f}s / {t_fast:.2f}s   ({t_ref / t_fast:.1f}x)")
    ok = d.max() < 1e-9 and dW.max() < 1e-9
    print(
        f"  VERDICT                 : {'OK -- vectorised path reproduces the original' if ok else 'MISMATCH'}"
    )
    print()
    return ok


# ---------------------------------------------------------------------------------------------
def pop_opt(argv, name, default):
    """Remove `--name VALUE` from argv in place and return VALUE, else default."""
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def set_inputs(corpus=None, blimp=None):
    """Point this module (and anything importing its readers) at explicit input paths."""
    global CORPUS_TXT, BLIMP_DIR
    if corpus:
        CORPUS_TXT = corpus
    if blimp:
        BLIMP_DIR = blimp


def missing_inputs(need_heldout=True, need_blimp=True):
    """Return a one-line reason the measurement cannot run here, or None if it can.

    Explicit because both inputs are gitignored local-only artifacts. Without this the corpus reader
    returns a SHORT string from a missing seek and the BLiMP scorer returns 0.0% over zero pairs --
    both of which look like results. A constant where a computed value belongs is this repo's
    recurring silent-degradation shape; refusing to print one is the point of the guard.
    """
    if not os.path.exists(CORPUS_TXT):
        return (
            f"no corpus at {CORPUS_TXT} -- WikiText-103 parquet "
            "wikitext-103-raw-v1/train-00000-of-00002.parquet -> data/lang/wikitext/"
        )
    if need_heldout:
        size = os.path.getsize(CORPUS_TXT)
        need = HELDOUT_BYTE_OFFSET + HELDOUT_CHARS
        if size < need:
            return (
                f"corpus {CORPUS_TXT} is {size} bytes, but the held-out slice starts at byte "
                f"{HELDOUT_BYTE_OFFSET} ({need} needed) -- a shorter file holds out nothing"
            )
    if need_blimp and not paradigms(BLIMP_DIR):
        return (
            f"no BLiMP paradigms under {BLIMP_DIR or 'data/lang/blimp'} -- "
            "https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip -> data/lang/blimp/"
        )
    return None


def read_corpus(n_chars, path=None):
    with open(path or CORPUS_TXT, "rb") as f:
        raw = f.read(int(n_chars * 1.6) + 4096)
    return raw.decode("utf-8", errors="ignore")[:n_chars]


def read_heldout(path=None):
    with open(path or CORPUS_TXT, "rb") as f:
        f.seek(HELDOUT_BYTE_OFFSET)
        raw = f.read(HELDOUT_CHARS * 4 + 4096)
    s = raw.decode("utf-8", errors="ignore")
    return s[64 : 64 + HELDOUT_CHARS]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    set_inputs(pop_opt(argv, "--corpus", None), pop_opt(argv, "--blimp", None))
    args = [a for a in argv if a != "--verify"]
    do_verify = "--verify" in argv
    why = missing_inputs()
    if why:
        print(f"SKIP lang_scale: {why}")
        return 0
    scales = [int(float(a)) for a in args] or [int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)]

    text = read_corpus(max(scales))
    print(f"corpus: WikiText-103 raw (train shard 0), {len(text)} chars read for exposure")

    if do_verify:
        if not verify(text):
            return 1
        if not args:
            return 0

    heldout = read_heldout()
    ho_ids = fold_ids(heldout)
    print(f"held-out: {len(ho_ids)} chars from byte offset {HELDOUT_BYTE_OFFSET} (never exposed)\n")

    la = LA.LanguageAcquisition(seed=0)
    enc = BatchEncoder(la)

    # -- precompute the fixed latents (they do not depend on W, so this is done once) -----------
    t0 = time.perf_counter()
    names = paradigms(BLIMP_DIR)
    sents = []
    for nm in names:
        for g, b in load_pairs(nm, root=BLIMP_DIR, limit=BLIMP_PER_PARADIGM):
            sents.append(g)
            sents.append(b)
    sents = list(dict.fromkeys(sents))
    sent_ids = [fold_ids(s) for s in sents]
    offs, cur = [], 0
    for si in sent_ids:
        offs.append((cur, cur + len(si)))
        cur += len(si)
    all_ids = np.concatenate(sent_ids)
    blimp_lat = np.empty((cur, enc.d_in + enc.d_r2), dtype=np.float64)
    for (a, b), si in zip(offs, sent_ids, strict=False):
        blimp_lat[a:b] = encode_stream(enc, si)
    ho_lat = encode_stream(enc, ho_ids)
    print(
        f"precomputed latents: {cur} BLiMP chars ({len(sents)} sentences) + {len(ho_ids)} held-out"
        f"  [{time.perf_counter() - t0:.1f}s]\n"
    )

    def blimp_acc(W):
        lp = logprobs_readonly(W, blimp_lat, all_ids)  # READ-ONLY: W is never updated here
        table = {}
        for s, (a, b) in zip(sents, offs, strict=False):
            table[s] = float(lp[a:b].sum()) / max(1, b - a)
        return score(lambda s: table[s], per_paradigm=BLIMP_PER_PARADIGM, root=BLIMP_DIR)[
            "accuracy"
        ]

    def ngram_blimp_acc(models, order):
        table = {
            s: ngram_logprob_seq(models, si, order) / max(1, len(si))
            for s, si in zip(sents, sent_ids, strict=False)
        }
        return score(lambda s: table[s], per_paradigm=BLIMP_PER_PARADIGM, root=BLIMP_DIR)[
            "accuracy"
        ]

    hdr = (
        f"{'chars_seen':>12} | {'acq-LM':>7} | {'bigram':>7} | {'5-gram':>7} | "
        f"{'acq PPL':>9} | {'2g PPL':>8} | {'5g PPL':>8} | {'expose s':>9}"
    )
    print("=" * len(hdr))
    print("BLiMP accuracy (%) and held-out per-character perplexity vs corpus exposure")
    print("anchors: chance 50.0 | 5-gram word LM 61.2 | GPT-2 large 86.0 | human 88.6")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    seen = 0
    carry = None
    for target in scales:
        chunk = text[seen:target]
        ids = fold_ids(chunk)
        t = time.perf_counter()
        for i in range(0, len(ids), 200_000):  # chunked so latents fit comfortably in RAM
            part = ids[i : i + 200_000]
            lat = encode_stream(enc, part, carry)
            apply_local_rule(la.W, lat, part, la.lr)
            carry = part[-la.ctx :] if len(part) >= la.ctx else carry
        dt = time.perf_counter() - t
        seen = target
        la.n_seen += len(ids)

        acq = 100.0 * blimp_acc(la.W)
        acq_ppl = math.exp(-logprobs_readonly(la.W, ho_lat, ho_ids).mean())
        if seen <= NGRAM_MAX_FIT:
            models = fit_ngrams(fold_ids(text[:seen]))
            bi = f"{100.0 * ngram_blimp_acc(models, 2):7.1f}"
            fi = f"{100.0 * ngram_blimp_acc(models, 5):7.1f}"
            bp = f"{math.exp(-ngram_logprob_seq(models, ho_ids, 2) / len(ho_ids)):8.2f}"
            fp = f"{math.exp(-ngram_logprob_seq(models, ho_ids, 5) / len(ho_ids)):8.2f}"
            del models
        else:  # n-gram count tables do not fit in RAM past 1e7
            bi = fi = f"{'--':>7}"
            bp = fp = f"{'--':>8}"
        print(
            f"{seen:12d} | {acq:7.1f} | {bi} | {fi} | {acq_ppl:9.2f} | {bp} | {fp} | {dt:9.1f}",
            flush=True,
        )

    print("-" * len(hdr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
