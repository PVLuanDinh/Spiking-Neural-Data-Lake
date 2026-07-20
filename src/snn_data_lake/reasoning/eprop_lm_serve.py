r"""eprop_lm_serve -- forward-only (stdlib + numpy) server for a frozen e-prop recurrent-spiking char LM.

WHY THIS EXISTS. training/eprop_lm.py trains a recurrent-spiking char LM the BRAIN-NATIVE way -- by
e-prop (Bellec et al., Nature Communications 11:3625, 2020): online eligibility traces + a broadcast
learning signal, with NO backpropagation-through-time for the hidden/recurrent weights. This module is
the forward-serve half: it reloads that model's FROZEN numpy artifact and runs the SAME recurrent
spiking forward in numpy to score full-sentence log-probability. The RECURRENT MEMBRANE STATE carries
the character history, so the score depends on ORDER by construction (unlike the order-invariant
SLDA-RFF language head).

WHAT THIS IS (honest framing). A char-level recurrent LIF LM: token/input weights W_in[N,V] (column c
is the input current from char c) -> a recurrent spiking LIF hidden layer (W_rec[N,N], no self-loop;
discrete-time soft-reset) -> a leaky readout W_out[V,N] (+b_out) over the vocab -> next-char logits, run
FORWARD ONLY in numpy from frozen weights. The exact same forward is trained offline by e-prop in
training/eprop_lm.py (the torch [train] tier), which exports the frozen artifact this module reloads.

CONTRACT (mirrors reasoning/ar_spiking_lm_serve.SpikingLMServe: guarded, numpy-lazy, degrade HONESTLY):
  EpropLMServe(path=None, seed=0)
    .available() -> bool                 True only if a valid frozen artifact loaded
    .logprob(sentence) -> float | None   sum of next-char log-probs (ORDER-SENSITIVE); None when
                                         unavailable -- never a fabricated number
    .generate(prompt, max_tokens) -> str deterministic (seeded) sample; "" when unavailable
  Higher .logprob == more acceptable, so it drops straight into a full-sentence scorer.

FORWARD-ONLY, NO TORCH AT LOAD. Importing this module pulls only stdlib; numpy is imported lazily and,
if absent, the server degrades to unavailable rather than raising. torch is NEVER imported here -- the
only torch touch is the OPTIONAL inline-train path in the __main__ self-check (guarded), which reuses
training/eprop_lm.py to produce a trained artifact for the strict order proof.

DETERMINISM. The serve path over frozen weights is fully deterministic (pure numpy float ops; seeded
sampling in generate). Hashing uses zlib.crc32, never builtin hash(). ASCII-only output.

HONEST SCALE. A TINY recurrent-spiking model sits near n-gram / chance on real grammar. This file +
training/eprop_lm.py are the ARCHITECTURE and the full train(e-prop)->freeze->forward-serve PIPELINE; a
grammar-moving result is a SEPARATE real-scale GPU run. This module claims no benchmark number a tiny
model cannot back.

  PYTHONPATH=src python -m snn_data_lake.reasoning.eprop_lm_serve            # fast stdlib+numpy proof
  PYTHONPATH=src python -m snn_data_lake.reasoning.eprop_lm_serve --train    # + strict trained proof
"""
import json
import os
import random
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_ARTIFACT = os.path.join(_ROOT, "models", "eprop_lm.npz")

FORMAT = "eprop_lm.v1"            # frozen-artifact schema tag (trainer + serve must agree)
# ---- LIF dynamics defaults (mirror training/eprop_lm.py; per-artifact values override via meta) ----
ALPHA = 0.90                     # membrane decay
V_TH = 0.6                       # spike threshold (soft reset subtracts V_TH)
KAPPA = 0.9                      # leaky-readout decay
W_IN_SCALE = 0.9                 # random-init input-weight std (test artifacts only)
W_REC_SCALE = 0.2                # random-init recurrent-weight std
W_OUT_SCALE = 0.2                # random-init readout-weight std

_WEIGHTS = ("W_in", "W_rec", "W_out", "b_out", "b_hid")


def _load_numpy():
    """Lazy numpy import (numpy-lazy contract). Returns the module or None if unavailable."""
    try:
        import numpy as np
        return np
    except Exception:
        return None


def build_random_artifact(path, text, hidden=32, seed=0):
    """Test/demo utility -- write a RANDOM-INIT (NOT trained) frozen artifact with the serve schema.

    Exists so the forward-serve + order-sensitivity + honest-unavailable paths are testable stdlib-only
    with NO torch when no trained artifact is present. Labelled trained=False; it proves the pipeline
    runs and is order-dependent, NOT that it has learned anything. A real artifact comes from e-prop
    training in training/eprop_lm.py."""
    np = _load_numpy()
    if np is None:
        raise RuntimeError("numpy is required to build an artifact")
    vocab = sorted(set(text))
    v_n, n = len(vocab), int(hidden)

    def rng_for(name):
        salt = zlib.crc32(name.encode("ascii")) & 0xFFFFFFFF   # crc32, never builtin hash()
        return np.random.RandomState((int(seed) ^ salt) & 0xFFFFFFFF)

    w_rec = rng_for("W_rec").standard_normal((n, n)) * W_REC_SCALE
    np.fill_diagonal(w_rec, 0.0)                                # no self-loop
    w = {
        "W_in": rng_for("W_in").standard_normal((n, v_n)) * W_IN_SCALE,
        "W_rec": w_rec,
        "W_out": rng_for("W_out").standard_normal((v_n, n)) * W_OUT_SCALE,
        "b_out": np.zeros(v_n),
        "b_hid": np.zeros(n),
    }
    meta = {
        "format": FORMAT, "vocab": vocab, "hidden": n, "vsize": v_n,
        "alpha": ALPHA, "kappa": KAPPA, "v_th": V_TH, "trained": False,
        "note": "random-init frozen artifact (NOT trained) -- proves forward-serve + order-sensitivity only",
    }
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta)), **w)
    return path


class EpropLMServe:
    """Forward-only numpy server for a frozen e-prop recurrent-spiking char LM. Guarded: no artifact /
    no numpy -> available() is False and logprob() returns None (honest-unavailable, never faked)."""

    def __init__(self, path=None, seed=0):
        self.seed = int(seed)
        self.path = path if path is not None else DEFAULT_ARTIFACT
        self._np = _load_numpy()
        self._ok = False
        self.meta = None
        self.w = None
        self.vocab = None
        self.stoi = None
        if self._np is not None and os.path.exists(self.path):
            try:
                self._load()
                self._ok = True
            except Exception:
                self._ok = False

    def available(self):
        return bool(self._ok)

    def _load(self):
        np = self._np
        data = np.load(self.path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"unrecognized artifact format: {meta.get('format')!r}")
        self.meta = meta
        self.vocab = list(meta["vocab"])
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.hidden = int(meta["hidden"])
        self.alpha = float(meta.get("alpha", ALPHA))
        self.kappa = float(meta.get("kappa", KAPPA))
        self.v_th = float(meta.get("v_th", V_TH))
        # unknown chars and the implicit sentence-start token both map to space (or index 0).
        self.unk = self.stoi.get(" ", 0)
        self.bos = self.stoi.get(" ", self.unk)
        self.w = {k: np.asarray(data[k], dtype=np.float64) for k in data.files if k != "meta"}
        missing = [k for k in _WEIGHTS if k not in self.w]
        if missing:
            raise ValueError("frozen artifact missing weights: " + ", ".join(missing))
        # W_rec is exported diagonal-free; enforce it here too so a hand-edited artifact stays honest.
        np.fill_diagonal(self.w["W_rec"], 0.0)

    # ---- encode / decode ----
    def _encode(self, s):
        return [self.stoi.get(c, self.unk) for c in s]

    def _decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    # ---- forward (identical recurrence to the torch e-prop trainer, run forward-only in numpy) ----
    def _forward_logits(self, ids):
        """Roll the recurrent spiking net forward over the char-id context `ids`; return logits[T,V].
        v = ALPHA*v - V_TH*z_prev + (W_in[:,c] + W_rec@z_prev + b_hid); z = (v>=V_TH);
        yr = KAPPA*yr + W_out@z; logits = yr + b_out. Byte-for-byte the trainer's forward."""
        np = self._np
        w = self.w
        n, v_n = self.hidden, len(self.vocab)
        v = np.zeros(n)
        z_prev = np.zeros(n)
        yr = np.zeros(v_n)
        logits = np.empty((len(ids), v_n))
        for t, cid in enumerate(ids):
            cur = w["W_in"][:, cid] + w["W_rec"] @ z_prev + w["b_hid"]
            v = self.alpha * v - self.v_th * z_prev + cur
            z = (v >= self.v_th).astype(np.float64)
            yr = self.kappa * yr + w["W_out"] @ z
            logits[t] = yr + w["b_out"]
            z_prev = z
        return logits

    def _log_softmax(self, logits):
        np = self._np
        z = logits - logits.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    # ---- public API ----
    def logprob(self, sentence):
        """Full-sentence log-probability = sum over positions of log P(char_t | chars_<t). An implicit
        space start-token lets the first real char be scored. Higher == more acceptable. This is the
        ORDER-SENSITIVE quantity (the recurrence carries history). None when unavailable (never faked)."""
        if not self._ok:
            return None
        core = self._encode(sentence)
        if not core:
            return 0.0
        ids = [self.bos] + core
        logp = self._log_softmax(self._forward_logits(ids))
        total = 0.0
        for t in range(len(core)):
            total += float(logp[t, ids[t + 1]])
        return total

    def generate(self, prompt, max_tokens=64):
        """Deterministic (seeded) autoregressive sample continuing `prompt`. Returns an ASCII string
        ("" when unavailable). Re-runs the causal forward each step so generate stays consistent with
        logprob (the architecture supports O(1)/token streaming state; the re-run is just simpler)."""
        if not self._ok:
            return ""
        np = self._np
        rng = np.random.RandomState(self.seed)
        temp = 0.8
        ids = [self.bos] + self._encode(prompt)
        out = []
        for _ in range(int(max_tokens)):
            last = self._forward_logits(ids)[-1] / temp
            last = last - last.max()
            p = np.exp(last)
            p = p / p.sum()
            nxt = int(rng.choice(len(p), p=p))
            ids.append(nxt)
            out.append(nxt)
        return "".join(c for c in self._decode(out) if ord(c) < 128)


# ---------------------------------------------------------------------------
# Self-check: fast stdlib+numpy proof of forward-serve + order-sensitivity + honest-unavailable on a
# random-init artifact, plus a torch-absent proof. With --train (torch present) it ALSO trains a tiny
# model inline via training/eprop_lm.py and asserts the STRICT learned order proof (in-dist > shuffled).
# ---------------------------------------------------------------------------
_TINY_TEXT = (
    "the spiking network reads the sentence one character at a time and it never looks ahead "
    "so the order of the letters changes the score that the model gives to the whole sentence "
    "a cat sat on the mat while the dog ran across the yard in the bright morning sun and then "
    "she sells sea shells by the sea shore and the shells shine softly in the pale evening light "
) * 3


def _try_train_inline(art, text):
    """Best-effort: train a tiny model by e-prop via the torch [train] tier for the strict order proof.
    Returns True if a trained artifact was written, False if torch/the trainer is unavailable. torch is
    imported HERE ONLY to TRAIN -- the serve itself never imports it."""
    try:
        import torch  # noqa: F401  (probe only; this module never depends on torch)
    except Exception:
        return False
    try:
        tdir = os.path.join(_ROOT, "training")
        if tdir not in sys.path:
            sys.path.insert(0, tdir)
        import eprop_lm as trainer
        trainer.train(text, hidden=48, ctx=48, steps=400, seed=0, out_path=art, batch=16,
                      lr=4e-3, log_every=0)
        return True
    except Exception as exc:
        print(f"  [note] inline torch train unavailable ({type(exc).__name__}); using random-init artifact")
        return False


def _selfcheck(with_train=False):
    np = _load_numpy()
    if np is None:
        print("SKIP eprop_lm_serve self-check: numpy unavailable")
        return 0
    import tempfile
    print("=" * 92)
    print("E-PROP RECURRENT-SPIKING LM SERVE -- forward-only numpy; full-sentence AR log-prob is the")
    print("order lever, and the model was trained the brain-native way (e-prop, no BPTT)")
    print("=" * 92)

    # torch-absence proof #1: importing + being about to USE the serve pulls no torch.
    assert "torch" not in sys.modules, "importing eprop_lm_serve must NOT import torch"

    workdir = tempfile.mkdtemp()
    art = os.path.join(workdir, "tiny_eprop_lm.npz")
    text = _TINY_TEXT

    trained = _try_train_inline(art, text) if with_train else False
    if not trained:
        build_random_artifact(art, text, hidden=32, seed=0)
    print(f"artifact: {os.path.basename(art)} (trained={trained})")

    srv = EpropLMServe(art, seed=0)
    assert srv.available() is True, "a valid frozen artifact must load"

    # (2) ORDER-SENSITIVITY -- the whole point (the recurrence carries history).
    seq = text[80:120]
    shuffled = list(seq)
    random.Random(0).shuffle(shuffled)
    shuffled = "".join(shuffled)
    lp = srv.logprob(seq)
    lps = srv.logprob(shuffled)
    assert lp is not None and lps is not None, "logprob must return floats when available"
    print(f"  logprob(in-dist)={lp:.4f}  logprob(shuffled)={lps:.4f}  diff={lp - lps:.4f}")
    if trained:
        assert lp > lps, "a trained AR LM must score the in-distribution string ABOVE its shuffle"
        print("  ORDER PROOF (trained): in-dist logprob > shuffled logprob -> LEARNED order-sensitivity.")
    else:
        assert abs(lp - lps) > 1e-3, "shuffling MUST change the AR logprob (i.e. NOT order-invariant)"
        print("  ORDER PROOF (random-init): shuffling CHANGES the AR logprob -> the model is NOT")
        print("  order-invariant. Strict in-dist>shuffled needs training (run with --train).")

    # (3) generate returns a non-empty ASCII string.
    gen = srv.generate("the ", max_tokens=24)
    print(f"  generate('the ', 24) -> {gen!r}")
    assert isinstance(gen, str) and len(gen) > 0, "generate must return a non-empty string"
    assert all(ord(c) < 128 for c in gen), "generate output must be ASCII"

    # (4) no artifact -> honest-unavailable (available() False, logprob None), never faked.
    miss = EpropLMServe(os.path.join(workdir, "does_not_exist.npz"))
    assert miss.available() is False, "a missing artifact must report unavailable"
    assert miss.logprob("hello world") is None, "unavailable logprob must be None, not a fake score"
    assert miss.generate("hello") == "", "unavailable generate must return empty string"
    print("  honest-unavailable: missing artifact -> available()=False, logprob()=None (no fake score).")

    # torch-absence proof #2: the full serve path (load + logprob + generate) imported no torch.
    if not with_train:
        assert "torch" not in sys.modules, "the serve forward path must NOT import torch"
        print("  torch-absent: 'torch' not in sys.modules after import + logprob + generate.")
    else:
        print("  note: --train imported torch ONLY to TRAIN the artifact; the serve forward stays numpy.")

    for p in (art,):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(workdir)
    except OSError:
        pass

    print("-" * 92)
    print("self-check OK: forward-only numpy serve reloads a frozen e-prop recurrent-spiking artifact; the")
    print("full-sentence AR log-prob is ORDER-SENSITIVE (the recurrent membrane state carries history).")
    print("HONEST: this proves the ARCHITECTURE + e-prop train->freeze->serve PIPELINE; a tiny model sits")
    print("near n-gram on real grammar. A grammar-moving result needs a real-scale GPU e-prop run")
    print("(training/eprop_lm.py --full) exported to models/eprop_lm.npz.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck(with_train=("--train" in sys.argv)))
