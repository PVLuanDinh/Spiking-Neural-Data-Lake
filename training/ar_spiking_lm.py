r"""training/ar_spiking_lm.py -- offline [train] tier for the SpikeGPT-style Spiking-RWKV language model.

WHY (the gap). Fluent open-domain LANGUAGE is stuck ~chance on BLiMP because the ROSE E-stage head
(reasoning/language_readout_slda.SLDA_RFF_np) is ORDER-INVARIANT and structurally blind to BLiMP's
order / agreement minimal-pair contrasts. The fix is a genuinely AUTOREGRESSIVE, ORDER-SENSITIVE
spiking sequence model scored by full-sentence log-probability. This file TRAINS that model; the frozen
weights it exports are served forward-only, in numpy, by reasoning/ar_spiking_lm_serve.py.

ARCHITECTURE (SpikeGPT, Zhu et al. arXiv:2302.13939 -- WikiText-2 test PPL 18.01, first SNN text
generation). Char-level. Token embedding -> per layer {token-shift + Spiking-RWKV token-mixer (a
LIF-gated LINEAR recurrence: O(T), O(1)/token state, NO softmax attention) + Spiking squared-ReLU
channel-mixer}, each mixer followed by a LIF spiking layer with multi-timescale decays (theta-over-
gamma) and a per-channel positional weight decay -> linear head. The forward math here is BYTE-FOR-BYTE
the same recurrence reasoning/ar_spiking_lm_serve.py runs in numpy, so a model trained here serves there.

TRAIN (this is the OPTIONAL torch [train] tier -- torch + GPU allowed, .venv-cu128 on the RTX 5070).
BPTT + a fast-sigmoid surrogate gradient (the SuperSpike / snntorch.surrogate.fast_sigmoid family,
hand-rolled here so the trainer needs ONLY torch and the forward Heaviside is byte-exact for numpy
parity) over the Gutenberg char stream from reasoning/corpus_stream.py (load_text -> word tokens joined
into a char stream). Exports FROZEN weights + metadata to a numpy .npz (models/ar_spiking_lm.npz --
the SAME path reasoning/ar_spiking_lm_serve.DEFAULT_ARTIFACT loads; the two MUST stay equal).

CONTRACT (mirrors training/surrogate_snn.py). Lazy imports of the corpus (no network at import time),
NO import from brainstem/ or the zero-dep core, NO import-time side effects (training runs only under an
explicit call / __main__), exports a forward-servable frozen artifact. GPU training is stochastic
(expected for the torch tier); the SERVE path over the frozen weights is deterministic. ASCII-only
output; zlib.crc32 (never builtin hash()) for any hashing.

HONEST SCALE (BabyLM 2025, aclanthology 2025.babylm-main.16). Moving BLiMP off its ~57% floor needs an
order-sensitive AR LM at ~GPT-2-small scale (~100M params, 10-100M words) to reach ~70-80% (human
88.6%). This file is the ARCHITECTURE + the train->freeze pipeline. The __main__ self-check is a TINY,
CPU-runnable end-to-end PROOF (small dim/layers, a few KB of in-file text, few steps) that the pipeline
WORKS and exports a reloadable artifact -- NOT a BLiMP-competitive model. A toy model sits at n-gram
~60%; the real result is a SEPARATE real-scale GPU run. This file claims no BLiMP number it cannot back.

  PYTHONUTF8=1 .venv-cu128/Scripts/python training/ar_spiking_lm.py          # tiny CPU/GPU self-check
  PYTHONUTF8=1 .venv-cu128/Scripts/python training/ar_spiking_lm.py --demo   # small real-corpus train+save
  PYTHONUTF8=1 .venv-cu128/Scripts/python training/ar_spiking_lm.py --full   # REAL-scale GPU train (wikitext)
"""
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_ARTIFACT = os.path.join(_ROOT, "models", "ar_spiking_lm.npz")

FORMAT = "spiking_lm.v1"          # frozen-artifact schema tag (serve + trainer must agree)
EPS = 1e-5                        # layer-norm epsilon (matches numpy serve)
THR = 1.0                        # LIF spike threshold
SLOPE = 25.0                     # fast-sigmoid surrogate steepness

_LAYER_PARAMS = (
    "tln_g", "tln_b", "t_mix_k", "t_mix_v", "t_mix_r",
    "t_Wk", "t_Wv", "t_Wr", "t_Wo", "t_decay",
    "cln_g", "cln_b", "c_mix_k", "c_mix_r", "c_Wk", "c_Wv", "c_Wr",
)


def default_betas(layers):
    """Multi-timescale LIF decays -- a per-layer ramp (theta-over-gamma hierarchy). Kept in lockstep
    with reasoning/ar_spiking_lm_serve.default_betas so a trained model serves identically."""
    span = max(1, layers - 1)
    t_beta = [round(0.50 + 0.45 * l / span, 4) for l in range(layers)]
    c_beta = [round(0.60 + 0.30 * l / span, 4) for l in range(layers)]
    return t_beta, c_beta


class _FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward (byte-exact for numpy parity) + fast-sigmoid surrogate gradient on the
    backward. This is the SuperSpike / snntorch.surrogate.fast_sigmoid surrogate, hand-rolled so the
    trainer depends on torch alone."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + ctx.slope * x.abs()) ** 2
        return grad_out * surrogate, None


def _spike(x, slope=SLOPE):
    return _FastSigmoidSpike.apply(x, slope)


class _Block(nn.Module):
    """One SpikeGPT-style layer: token-shift + Spiking-RWKV token-mixer, then token-shift + spiking
    squared-ReLU channel-mixer, each with a residual add. All params are explicit so export names line
    up with the numpy serve schema."""

    def __init__(self, dim, d_ff, t_beta, c_beta):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff
        self.t_beta = float(t_beta)
        self.c_beta = float(c_beta)
        self.tln_g = nn.Parameter(torch.ones(dim))
        self.tln_b = nn.Parameter(torch.zeros(dim))
        self.t_mix_k = nn.Parameter(torch.rand(dim))
        self.t_mix_v = nn.Parameter(torch.rand(dim))
        self.t_mix_r = nn.Parameter(torch.rand(dim))
        self.t_Wk = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.t_Wv = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.t_Wr = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.t_Wo = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.t_decay = nn.Parameter(torch.randn(dim) * 0.1)
        self.cln_g = nn.Parameter(torch.ones(dim))
        self.cln_b = nn.Parameter(torch.zeros(dim))
        self.c_mix_k = nn.Parameter(torch.rand(dim))
        self.c_mix_r = nn.Parameter(torch.rand(dim))
        self.c_Wk = nn.Parameter(torch.randn(dim, d_ff) * 0.1)
        self.c_Wv = nn.Parameter(torch.randn(d_ff, dim) * 0.1)
        self.c_Wr = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def _ln(self, x, g, b):
        return F.layer_norm(x, (self.dim,), g, b, EPS)

    @staticmethod
    def _shift(x):
        # token-shift for [B, T, D]: row t <- x[t-1], row 0 == 0
        return F.pad(x, (0, 0, 1, 0))[:, :-1, :]

    def _timemix(self, x):
        b, t, dim = x.shape
        xn = self._ln(x, self.tln_g, self.tln_b)
        xp = self._shift(xn)
        xk = xn * self.t_mix_k + xp * (1.0 - self.t_mix_k)
        xv = xn * self.t_mix_v + xp * (1.0 - self.t_mix_v)
        xr = xn * self.t_mix_r + xp * (1.0 - self.t_mix_r)
        k = xk @ self.t_Wk
        v = xv @ self.t_Wv
        r = torch.sigmoid(xr @ self.t_Wr)
        gate = torch.sigmoid(k)                          # bounded input gate (stable, parity-safe)
        decay = torch.sigmoid(self.t_decay)              # per-channel positional weight decay in (0,1)
        s = torch.zeros(b, dim, device=x.device)
        wkv = []
        for step in range(t):                            # LIF-gated linear recurrence, O(1)/token
            s = decay * s + gate[:, step] * v[:, step]
            wkv.append(s)
        out = (r * torch.stack(wkv, 1)) @ self.t_Wo
        return self._fire(out, self.t_beta)

    def _channelmix(self, x):
        xn = self._ln(x, self.cln_g, self.cln_b)
        xp = self._shift(xn)
        xk = xn * self.c_mix_k + xp * (1.0 - self.c_mix_k)
        xr = xn * self.c_mix_r + xp * (1.0 - self.c_mix_r)
        kk = torch.relu(xk @ self.c_Wk)
        kk = kk * kk                                     # squared-ReLU channel activation (RWKV RFFN)
        kv = kk @ self.c_Wv
        r = torch.sigmoid(xr @ self.c_Wr)
        return self._fire(r * kv, self.c_beta)

    def _fire(self, cur, beta):
        b, t, dim = cur.shape
        mem = torch.zeros(b, dim, device=cur.device)
        spk = []
        for step in range(t):                            # leaky integrate-and-fire, subtractive reset
            mem = beta * mem + cur[:, step]
            fire = _spike(mem - THR)
            spk.append(fire)
            mem = mem - fire * THR
        return torch.stack(spk, 1)

    def forward(self, x):
        x = x + self._timemix(x)
        x = x + self._channelmix(x)
        return x


class SpikingLM(nn.Module):
    """Char-level Spiking-RWKV LM. forward(ids[B,T]) -> logits[B,T,V]. Causal (token-shift + recurrences
    look only backward), so logits[:, t] predict token t+1 from tokens 0..t."""

    def __init__(self, vocab, dim, layers, d_ff=None, seed=0):
        super().__init__()
        torch.manual_seed(int(seed))
        self.vocab = list(vocab)
        self.vsize = len(self.vocab)
        self.dim = int(dim)
        self.layers = int(layers)
        self.d_ff = int(d_ff) if d_ff else 4 * int(dim)
        t_beta, c_beta = default_betas(self.layers)
        self.emb = nn.Parameter(torch.randn(self.vsize, self.dim) * 0.1)
        self.blocks = nn.ModuleList(
            [_Block(self.dim, self.d_ff, t_beta[l], c_beta[l]) for l in range(self.layers)]
        )
        self.ln_out_g = nn.Parameter(torch.ones(self.dim))
        self.ln_out_b = nn.Parameter(torch.zeros(self.dim))
        self.head_W = nn.Parameter(torch.randn(self.dim, self.vsize) * 0.1)
        self.head_b = nn.Parameter(torch.zeros(self.vsize))

    def forward(self, ids):
        x = self.emb[ids]
        for blk in self.blocks:
            x = blk(x)
        xo = F.layer_norm(x, (self.dim,), self.ln_out_g, self.ln_out_b, EPS)
        return xo @ self.head_W + self.head_b


def export_npz(model, path, trained=True):
    """Freeze the trained model to a numpy .npz with the schema reasoning/ar_spiking_lm_serve.py reloads."""
    import numpy as np
    w = {"emb": model.emb.detach().cpu().numpy()}
    for l, blk in enumerate(model.blocks):
        for name in _LAYER_PARAMS:
            w["%s.%d" % (name, l)] = getattr(blk, name).detach().cpu().numpy()
    w["ln_out_g"] = model.ln_out_g.detach().cpu().numpy()
    w["ln_out_b"] = model.ln_out_b.detach().cpu().numpy()
    w["head_W"] = model.head_W.detach().cpu().numpy()
    w["head_b"] = model.head_b.detach().cpu().numpy()
    t_beta = [blk.t_beta for blk in model.blocks]
    c_beta = [blk.c_beta for blk in model.blocks]
    meta = {
        "format": FORMAT, "vocab": model.vocab, "dim": model.dim, "layers": model.layers,
        "d_ff": model.d_ff, "thr": THR, "eps": EPS, "t_beta": t_beta, "c_beta": c_beta,
        "trained": bool(trained),
        "note": "SpikeGPT-style Spiking-RWKV char LM; forward-served by reasoning/ar_spiking_lm_serve.py",
    }
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta)), **w)
    return path


def train(corpus, dim=64, layers=2, ctx=64, steps=400, seed=0, out_path=None,
          batch=16, lr=3e-3, d_ff=None, log_every=50, heldout=None, eval_every=0):
    """Train the Spiking-RWKV LM by BPTT + fast-sigmoid surrogate on a CHAR corpus, then export the
    frozen artifact. `corpus` is a string; the vocab is its distinct characters. Returns out_path.

    `heldout` (optional) is text DISJOINT from `corpus`; every eval_every steps its per-character
    perplexity is printed and the artifact is checkpointed, so a long run is selected on held-out
    PPL and survives an interruption."""
    import time

    import numpy as np
    torch.manual_seed(int(seed))
    if not isinstance(corpus, str):
        corpus = "".join(corpus)
    vocab = sorted(set(corpus))
    if len(vocab) < 2:
        raise ValueError("corpus needs at least 2 distinct characters")
    stoi = {c: i for i, c in enumerate(vocab)}
    # LUT encode via numpy when the corpus is ASCII: a list comprehension over a 1e8-char corpus
    # costs ~1 GiB of pointers and a minute of wall clock. The ids are IDENTICAL either way -- this
    # is plumbing, not math. Non-ASCII corpora take the slow exact path rather than a lossy encode.
    if corpus.isascii():
        lut = np.zeros(128, dtype=np.int64)
        for c, i in stoi.items():
            lut[ord(c)] = i
        ids = torch.from_numpy(lut[np.frombuffer(corpus.encode("ascii"), dtype=np.uint8)])
    else:
        ids = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)
    n = ids.numel()
    if n < ctx + 2:
        raise ValueError("corpus too short (%d chars) for ctx=%d" % (n, ctx))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpikingLM(vocab, dim, layers, d_ff=d_ff, seed=seed).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if log_every:
        print("train: device=%s vocab=%d dim=%d layers=%d ctx=%d steps=%d params=%d"
              % (device, len(vocab), dim, layers, ctx, steps, n_params))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Warmup + cosine decay rather than a fixed lr. Not decoration: over thousands of steps a fixed
    # 1.5e-3 leaves the model oscillating around its optimum, and the held-out PPL this run is
    # SELECTED on is the thing that pays for it. `steps` sets the period, so nothing is hand-tuned.
    warm = max(1, int(0.02 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm
        else 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * (s - warm) / max(1, steps - warm))),
    )
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(int(seed))
    model.train()
    last_loss = float("nan")
    out_path = out_path or DEFAULT_ARTIFACT
    best_ppl = float("inf")
    t0 = time.time()
    for step in range(steps):
        starts = rng.randint(0, n - ctx - 1, size=batch)
        xb = torch.stack([ids[s:s + ctx] for s in starts]).to(device)
        yb = torch.stack([ids[s + 1:s + ctx + 1] for s in starts]).to(device)
        logits = model(xb)
        loss = loss_fn(logits.reshape(-1, model.vsize), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()                                  # BPTT through the unrolled recurrences
        opt.step()
        sched.step()
        last_loss = float(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            el = time.time() - t0
            print("  step %d/%d  loss %.4f  ppl(train) %.2f  %.0fs  %.0f tok/s"
                  % (step, steps, last_loss, min(1e9, math.exp(min(20.0, last_loss))), el,
                     batch * ctx * (step + 1) / max(1e-9, el)), flush=True)
        # HELD-OUT selection: checkpoint only when PPL on the byte-disjoint slice IMPROVES.
        if heldout and eval_every and (step % eval_every == eval_every - 1 or step == steps - 1):
            ppl = eval_ppl(model, heldout, ctx=ctx, device=device)
            mark = ""
            if ppl < best_ppl:
                best_ppl, mark = ppl, "  <- best, checkpointed"
                export_npz(model, out_path, trained=True)
            print("    step %d  HELD-OUT char PPL %.4f%s" % (step, ppl, mark), flush=True)

    if not (heldout and eval_every):
        export_npz(model, out_path, trained=True)
    elif best_ppl == float("inf"):
        export_npz(model, out_path, trained=True)
    if log_every:
        print("saved frozen artifact -> %s (final train loss %.4f, best held-out PPL %s)"
              % (out_path, last_loss, "n/a" if best_ppl == float("inf") else "%.4f" % best_ppl))
    return out_path


def _char_corpus_from_stream(max_chars=200000):
    """A real char stream from reasoning/corpus_stream.load_text (Gutenberg / bundled sample offline),
    word tokens joined by spaces. Imported lazily so this module stays importable without src on path."""
    src = os.path.join(_ROOT, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from snn_data_lake.reasoning.corpus_stream import load_text
    words, offline = load_text()
    text = " ".join(words)
    return text[:max_chars], offline


# ---------------------------------------------------------------------------------------------
# REAL-SCALE corpus -- byte-addressed, with a HARD held-out disjointness guard.
#
# WHY THE GUARD IS BYTE-ADDRESSED. scratchpad/lang_word_big_out.VOID.txt is a voided run in this
# repo: its reader took a CHARACTER budget, consumed int(n*1.6)+4096 BYTES to fill it, and at 2.5e8
# chars that byte count swallowed the "held-out" slice at byte 200,000,000. The perplexity of 6.2 it
# printed was the model being scored on text it had already read. An estimated exposure cannot be
# audited, so this reader takes a BYTE budget and seeks by BYTE: the exposed interval is exactly
# [0, exposure_bytes) and the held-out interval is exactly [HELDOUT_BYTE_OFFSET, +HELDOUT_BYTES).
# Disjointness is then a property of two integers, not of a ratio that might drift.
# ---------------------------------------------------------------------------------------------
CORPUS_TXT = os.path.join(_ROOT, "data", "lang", "wikitext", "train0.txt")
HELDOUT_BYTE_OFFSET = 200_000_000  # same offset scratchpad/lang_scale.py holds out
HELDOUT_BYTES = 400_000


def _ascii_only(raw):
    """Decode a byte block and keep only ASCII (Windows cp1252-safe printing downstream)."""
    txt = raw.decode("utf-8", errors="ignore")
    return "".join(c for c in txt if ord(c) < 128)


def wikitext_prefix(exposure_bytes, path=None):
    """Return the ASCII-filtered text of the FIRST `exposure_bytes` bytes of the corpus.

    Hard-fails if the exposure would reach HELDOUT_BYTE_OFFSET -- refusing to run beats printing a
    flattering contaminated number (the VOID lesson above)."""
    p = path or CORPUS_TXT
    exposure_bytes = int(exposure_bytes)
    if exposure_bytes >= HELDOUT_BYTE_OFFSET:
        raise SystemExit(
            "REFUSING TO RUN: exposure of %d bytes reaches the held-out slice at byte %d. "
            "Max safe exposure is %d bytes." % (exposure_bytes, HELDOUT_BYTE_OFFSET, HELDOUT_BYTE_OFFSET - 1)
        )
    if not os.path.exists(p):
        raise SystemExit("corpus not found: %s" % p)
    with open(p, "rb") as fh:
        raw = fh.read(exposure_bytes)
        consumed = fh.tell()
    assert consumed <= exposure_bytes, "reader over-read its byte budget"
    return _ascii_only(raw)


def wikitext_heldout(path=None, n_bytes=HELDOUT_BYTES):
    """Return ASCII text from bytes [HELDOUT_BYTE_OFFSET, +n_bytes) -- never inside any exposure
    wikitext_prefix() will hand out, because that function hard-fails at the offset."""
    p = path or CORPUS_TXT
    if not os.path.exists(p):
        raise SystemExit("corpus not found: %s" % p)
    with open(p, "rb") as fh:
        fh.seek(HELDOUT_BYTE_OFFSET)
        raw = fh.read(int(n_bytes))
    return _ascii_only(raw)


def eval_ppl(model, text, ctx=256, batch=32, max_windows=256, device=None):
    """Per-character perplexity of `model` over `text`, in non-overlapping ctx-length windows.
    Chars outside the model's vocab are mapped to space, exactly as the numpy serve does."""
    device = device or next(model.parameters()).device
    stoi = {c: i for i, c in enumerate(model.vocab)}
    unk = stoi.get(" ", 0)
    ids = [stoi.get(c, unk) for c in text]
    wins = [ids[i : i + ctx + 1] for i in range(0, len(ids) - ctx - 1, ctx)][:max_windows]
    if not wins:
        return float("nan")
    lossf = nn.CrossEntropyLoss(reduction="sum")
    tot, cnt = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            chunk = torch.tensor(wins[i : i + batch], dtype=torch.long, device=device)
            logits = model(chunk[:, :-1])
            tot += float(lossf(logits.reshape(-1, model.vsize), chunk[:, 1:].reshape(-1)))
            cnt += chunk[:, 1:].numel()
    model.train()
    return float(torch.exp(torch.tensor(tot / cnt)))


def full(exposure_bytes=120_000_000, dim=512, layers=4, ctx=256, steps=6000, batch=48,
         lr=1.5e-3, seed=0, out_path=None, log_every=100, eval_every=500):
    """REAL-scale GPU train on the wikitext char prefix -> models/ar_spiking_lm.npz.

    Reports held-out char perplexity on the byte-disjoint slice. Model selection is on held-out
    PPL, never on BLiMP -- BLiMP is scored ONCE, afterwards, by the frozen artifact."""
    torch.backends.cuda.matmul.allow_tf32 = True  # training-time only; exported weights are exact
    torch.backends.cudnn.allow_tf32 = True
    text = wikitext_prefix(exposure_bytes)
    held = wikitext_heldout()
    print("full: exposure bytes [0, %d) -> %d ASCII chars, vocab %d" % (exposure_bytes, len(text), len(set(text))))
    print("full: held-out bytes [%d, %d) -> %d ASCII chars (DISJOINT by byte interval)"
          % (HELDOUT_BYTE_OFFSET, HELDOUT_BYTE_OFFSET + HELDOUT_BYTES, len(held)))
    out = train(text, dim=dim, layers=layers, ctx=ctx, steps=steps, seed=seed,
                out_path=out_path or DEFAULT_ARTIFACT, batch=batch, lr=lr, log_every=log_every,
                heldout=held, eval_every=eval_every)
    print("HONEST: this is one run at the scale one card and one session allow. BLiMP-competitive")
    print("grammar needs ~GPT-2-small on 10-100M WORDS; report the size trained and the number got.")
    return out


def main():
    """Small demo: train on the real corpus_stream char stream and save to models/ar_spiking_lm.npz."""
    text, offline = _char_corpus_from_stream()
    print("demo: char corpus %d chars, vocab %d (%s)"
          % (len(text), len(set(text)), "offline sample" if offline else "real corpus"))
    out = train(text, dim=64, layers=2, ctx=64, steps=400, seed=0, out_path=DEFAULT_ARTIFACT)
    print("HONEST: small demo train -- BLiMP-moving competence needs a real-scale GPU run (see docstring).")
    return out


_TINY_TEXT = (
    "the spiking network reads the sentence one character at a time and it never looks ahead "
    "so the order of the letters changes the score that the model gives to the whole sentence "
    "a cat sat on the mat while the dog ran across the yard in the bright morning sun and then "
    "she sells sea shells by the sea shore and the shells shine softly in the pale evening light "
) * 3


def _selfcheck():
    """Fast end-to-end pipeline proof: train a TINY model on tiny in-file text, export to a temp .npz,
    assert the artifact exists and reloads. NOT a competent model -- a proof the pipeline works."""
    import tempfile
    print("=" * 92)
    print("SPIKING-RWKV LM TRAINER -- tiny end-to-end pipeline proof (train -> freeze -> reload)")
    print("=" * 92)
    workdir = tempfile.mkdtemp()
    art = os.path.join(workdir, "tiny_spiking_lm.npz")
    out = train(_TINY_TEXT, dim=24, layers=2, ctx=48, steps=120, seed=0,
                out_path=art, batch=16, lr=4e-3, log_every=40)
    assert os.path.exists(out), "trainer must write the frozen artifact"

    import numpy as np
    data = np.load(out, allow_pickle=False)
    assert "emb" in data.files and "head_W" in data.files, "artifact missing core weights"
    meta = json.loads(str(data["meta"]))
    assert meta["format"] == FORMAT and meta["trained"] is True, "artifact metadata wrong"
    assert len(meta["vocab"]) >= 2 and len(meta["t_beta"]) == meta["layers"], "artifact meta inconsistent"
    print("reload OK: %d arrays, vocab=%d, dim=%d, layers=%d"
          % (len(data.files), len(meta["vocab"]), meta["dim"], meta["layers"]))

    try:
        os.remove(out)
        os.rmdir(workdir)
    except OSError:
        pass

    print("-" * 92)
    print("self-check OK: trained a TINY Spiking-RWKV LM and exported a reloadable frozen artifact.")
    print("The forward math is byte-identical to reasoning/ar_spiking_lm_serve.py, which serves it in numpy")
    print("(forward-only) and scores full-sentence AR log-prob -- the ORDER-SENSITIVE BLiMP lever the")
    print("order-invariant SLDA-RFF lacks. HONEST: this is a CPU-runnable PIPELINE proof, not a BLiMP")
    print("result. Moving BLiMP off its ~57 pct floor needs a REAL-SCALE GPU run (~GPT-2-small, 10-100M")
    print("words); a toy model sits at n-gram ~60 pct.")
    print("=" * 92)
    return 0


def _flag(name, default, cast=int):
    """Read --name=VALUE from argv (no argparse: this file stays importable with zero side effects)."""
    for a in sys.argv[1:]:
        if a.startswith("--%s=" % name):
            return cast(a.split("=", 1)[1])
    return default


if __name__ == "__main__":
    if "--demo" in sys.argv:
        main()
        sys.exit(0)
    if "--full" in sys.argv:
        full(
            exposure_bytes=_flag("bytes", 120_000_000),
            dim=_flag("dim", 512),
            layers=_flag("layers", 4),
            ctx=_flag("ctx", 256),
            steps=_flag("steps", 6000),
            batch=_flag("batch", 48),
            lr=_flag("lr", 1.5e-3, float),
            seed=_flag("seed", 0),
            eval_every=_flag("eval_every", 500),
            out_path=_flag("out", None, str),
        )
        sys.exit(0)
    sys.exit(_selfcheck())
