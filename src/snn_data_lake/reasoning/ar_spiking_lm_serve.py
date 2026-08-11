r"""ar_spiking_lm_serve -- forward-only (stdlib + numpy) server for a frozen Spiking-RWKV language model.

WHY THIS EXISTS (the gap it closes). The ROSE E-stage language head
(cortex/language_readout_slda.SLDA_RFF_np, promoted 2026-08-11) is ORDER-INVARIANT by construction (its own docstring:
"order-invariant continual==joint"): it decodes a fixed positional context code, so it is structurally
blind to BLiMP's order / agreement minimal-pair contrasts and caps ~57% decoupled from perplexity.
The fix is a genuinely AUTOREGRESSIVE, ORDER-SENSITIVE sequence model scored by FULL-SENTENCE
log-probability. This module is the forward-serve half of that fix.

WHAT THIS IS (honest framing). A SpikeGPT-style Spiking-RWKV char-level LM (Zhu et al., "SpikeGPT",
arXiv:2302.13939 -- the proven spiking-native autoregressive LM, WikiText-2 test PPL 18.01), run
FORWARD ONLY in numpy from FROZEN weights. Architecture: token embedding -> per layer {token-shift +
Spiking-RWKV token-mixer (a LIF-gated LINEAR recurrence -- O(T), O(1)/token state, NO softmax
attention) + Spiking squared-ReLU channel-mixer} -> spiking layers with multi-timescale LIF decays
(theta-over-gamma) and a per-channel positional weight decay -> linear head. The exact SAME forward math
is trained offline with BPTT + fast-sigmoid surrogate gradients by training/ar_spiking_lm.py (the torch
[train] tier), which exports the frozen numpy artifact this module reloads.

CONTRACT (mirrors SLDA_RFF_np: guarded, numpy-lazy, degrade HONESTLY -- never fake a score):
  SpikingLMServe(path=None, seed=0)
    .available() -> bool                 True only if a valid frozen artifact loaded
    .logprob(sentence) -> float | None   sum of next-token log-probs (THE order-sensitive BLiMP lever);
                                         None when unavailable -- never a fabricated number
    .generate(prompt, max_tokens) -> str deterministic (seeded) sample; "" when unavailable
  Higher .logprob == more acceptable, so it drops straight into reasoning/blimp_bench.score(scorer).

FORWARD-ONLY, NO TORCH AT LOAD. Importing this module pulls only stdlib; numpy is imported lazily and,
if absent, the server degrades to unavailable rather than raising. torch is NEVER imported here -- the
only torch touch is the OPTIONAL inline-train path in the __main__ self-check (guarded), which reuses
training/ar_spiking_lm.py to produce a trained artifact for the strict order proof.

DETERMINISM. The serve path over frozen weights is fully deterministic (pure numpy float ops; seeded
sampling in generate). Hashing uses zlib.crc32, never builtin hash(). ASCII-only output.

HONEST SCALE (BabyLM 2025, aclanthology 2025.babylm-main.16). Getting BLiMP off the ~57% floor needs an
order-sensitive AR LM at ~GPT-2-small scale (~100M params, 10-100M words) to reach ~70-80% (human
88.6%). A TOY model sits at n-gram ~60%. This file + training/ar_spiking_lm.py are the ARCHITECTURE and
the full train->freeze->forward-serve PIPELINE; the BLiMP-moving result is a SEPARATE real-scale GPU
run. This module claims no BLiMP number a tiny model cannot back.

  PYTHONPATH=src python -m snn_data_lake.reasoning.ar_spiking_lm_serve            # fast stdlib+numpy proof
  PYTHONPATH=src python -m snn_data_lake.reasoning.ar_spiking_lm_serve --train    # + strict trained proof
"""

import json
import os
import random
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_ARTIFACT = os.path.join(_ROOT, "models", "ar_spiking_lm.npz")

FORMAT = "spiking_lm.v1"  # frozen-artifact schema tag (trainer + serve must agree)
EPS = 1e-5  # layer-norm epsilon (matches torch F.layer_norm default)
THR = 1.0  # LIF spike threshold

# Training-provenance keys the trainer records inside the artifact (meta["provenance"]) so the
# artifact testifies to how it was made -- corpus size, planned steps, the CHECKPOINTED step and its
# held-out PPL -- instead of those numbers living only in a gitignored training log. Artifacts
# written before this schema (and the random-init test artifacts) carry no "provenance" key and are
# served with every field "unknown" -- they must keep loading, never crash. docs/TRAINING.md
# documents the schema; training/ar_spiking_lm.export_npz writes it.
PROVENANCE_KEYS = (
    "corpus_chars",
    "ctx",
    "steps_planned",
    "checkpoint_step",
    "heldout_ppl",
    "heldout_chars",
    "heldout_max_windows",
    "seed",
    "batch",
    "lr",
)

# The per-layer parameter names carried in the .npz, stored as "<name>.<layer_index>".
_LAYER_PARAMS = (
    "tln_g",
    "tln_b",
    "t_mix_k",
    "t_mix_v",
    "t_mix_r",
    "t_Wk",
    "t_Wv",
    "t_Wr",
    "t_Wo",
    "t_decay",
    "cln_g",
    "cln_b",
    "c_mix_k",
    "c_mix_r",
    "c_Wk",
    "c_Wv",
    "c_Wr",
)


def _load_numpy():
    """Lazy numpy import (numpy-lazy contract). Returns the module or None if unavailable."""
    try:
        import numpy as np

        return np
    except Exception:
        return None


def default_betas(layers):
    """Multi-timescale LIF decays: a per-layer ramp so deeper layers integrate over longer windows
    (a theta-over-gamma hierarchy). Returns (t_beta[layers], c_beta[layers])."""
    span = max(1, layers - 1)
    t_beta = [round(0.50 + 0.45 * l / span, 4) for l in range(layers)]
    c_beta = [round(0.60 + 0.30 * l / span, 4) for l in range(layers)]
    return t_beta, c_beta


def build_random_artifact(path, text, dim=24, layers=2, d_ff=None, seed=0):
    """Test/demo utility -- write a RANDOM-INIT (NOT trained) frozen artifact with the serve schema.

    This exists so the forward-serve + order-sensitivity + honest-unavailable paths are testable
    stdlib-only, with NO torch, when no trained artifact is present. It is labelled trained=False in the
    metadata; it proves the pipeline runs and is order-dependent, NOT that it has learned anything.
    A real, competent artifact comes from training/ar_spiking_lm.py (BPTT surrogate training)."""
    np = _load_numpy()
    if np is None:
        raise RuntimeError("numpy is required to build an artifact")
    vocab = sorted(set(text))
    v_n, d, l_n = len(vocab), int(dim), int(layers)
    ff = int(d_ff) if d_ff else 4 * d

    def rng_for(name):
        salt = zlib.crc32(name.encode("ascii")) & 0xFFFFFFFF  # crc32, never builtin hash()
        return np.random.RandomState((int(seed) ^ salt) & 0xFFFFFFFF)

    w = {"emb": rng_for("emb").standard_normal((v_n, d)) * 0.1}
    for l in range(l_n):
        r = rng_for(f"layer.{l}")
        w[f"tln_g.{l}"] = np.ones(d)
        w[f"tln_b.{l}"] = np.zeros(d)
        w[f"t_mix_k.{l}"] = r.random_sample(d)
        w[f"t_mix_v.{l}"] = r.random_sample(d)
        w[f"t_mix_r.{l}"] = r.random_sample(d)
        w[f"t_Wk.{l}"] = r.standard_normal((d, d)) * 0.1
        w[f"t_Wv.{l}"] = r.standard_normal((d, d)) * 0.1
        w[f"t_Wr.{l}"] = r.standard_normal((d, d)) * 0.1
        w[f"t_Wo.{l}"] = r.standard_normal((d, d)) * 0.1
        w[f"t_decay.{l}"] = r.standard_normal(d) * 0.1
        w[f"cln_g.{l}"] = np.ones(d)
        w[f"cln_b.{l}"] = np.zeros(d)
        w[f"c_mix_k.{l}"] = r.random_sample(d)
        w[f"c_mix_r.{l}"] = r.random_sample(d)
        w[f"c_Wk.{l}"] = r.standard_normal((d, ff)) * 0.1
        w[f"c_Wv.{l}"] = r.standard_normal((ff, d)) * 0.1
        w[f"c_Wr.{l}"] = r.standard_normal((d, d)) * 0.1
    w["ln_out_g"] = np.ones(d)
    w["ln_out_b"] = np.zeros(d)
    w["head_W"] = rng_for("head").standard_normal((d, v_n)) * 0.1
    w["head_b"] = np.zeros(v_n)

    t_beta, c_beta = default_betas(l_n)
    meta = {
        "format": FORMAT,
        "vocab": vocab,
        "dim": d,
        "layers": l_n,
        "d_ff": ff,
        "thr": THR,
        "eps": EPS,
        "t_beta": t_beta,
        "c_beta": c_beta,
        "trained": False,
        "note": "random-init frozen artifact (NOT trained) -- proves forward-serve + order-sensitivity only",
    }
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta)), **w)
    return path


class SpikingLMServe:
    """Forward-only numpy server for a frozen Spiking-RWKV LM. Guarded: no artifact / no numpy ->
    available() is False and logprob() returns None (honest-unavailable, never a fabricated score)."""

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
        self.dim = int(meta["dim"])
        self.layers = int(meta["layers"])
        self.thr = float(meta.get("thr", THR))
        self.eps = float(meta.get("eps", EPS))
        self.t_beta = [float(x) for x in meta["t_beta"]]
        self.c_beta = [float(x) for x in meta["c_beta"]]
        # unknown chars and the implicit sentence-start token both map to space (or index 0).
        self.unk = self.stoi.get(" ", 0)
        self.bos = self.stoi.get(" ", self.unk)
        self.w = {k: np.asarray(data[k], dtype=np.float64) for k in data.files if k != "meta"}
        # fail loudly if a required weight is missing (guarded load, not a silent half-model)
        need = ["emb", "ln_out_g", "ln_out_b", "head_W", "head_b"]
        for l in range(self.layers):
            need += [f"{n}.{l}" for n in _LAYER_PARAMS]
        missing = [k for k in need if k not in self.w]
        if missing:
            raise ValueError("frozen artifact missing weights: " + ", ".join(missing[:6]))

    def provenance(self):
        """How the loaded artifact was trained, from ITS OWN metadata -- never from a training log.

        Returns a dict with every PROVENANCE_KEYS entry; a field the artifact does not carry (older
        exports, random-init test artifacts, or a trainer run with no held-out eval) reads
        "unknown" rather than raising -- backward compatibility is part of the contract. Returns
        None when no artifact is loaded (honest-unavailable, like logprob).

        READ `checkpoint_step` CAREFULLY: it is the step whose weights these ARE -- the checkpoint
        that was scored and saved -- NOT the last step the training run logged. A run interrupted
        past its best checkpoint logs further steps that never made it into the artifact; quoting
        those for this file would be provenance of a model that does not exist."""
        if not self._ok:
            return None
        raw = self.meta.get("provenance") or {}
        return {k: (raw[k] if raw.get(k) is not None else "unknown") for k in PROVENANCE_KEYS}

    def provenance_status(self):
        """Whether this artifact's provenance was EVER recorded -- a different question from whether a
        field is filled in, and the one a reader needs first.

        `provenance()` answers "unknown" for both a field the trainer left empty and a file written
        before the schema existed. Those are not the same: the first is a gap that a re-run fills, the
        second is PERMANENT for that file, because every PROVENANCE_KEY is a training-run value
        (corpus size, planned steps, checkpoint step, PPL, seed, batch, lr) and NOT ONE of them is
        derivable from the weights. Without the original checkpoint there is nothing to re-export from,
        and retraining would produce a different model -- so a pre-schema artifact's training numbers
        can only ever be sourced to a log, never verified against the file.

        Returns None when no artifact is loaded, "recorded" when the artifact carries the block, and
        "unrecorded" otherwise. `models/ar_spiking_lm.npz` (the shipped real-scale LM) is "unrecorded":
        it was exported before the schema landed, which is why docs quoting its step count mark that
        number as log-sourced rather than artifact-verified."""
        if not self._ok:
            return None
        return "recorded" if self.meta.get("provenance") else "unrecorded"

    # ---- encode / decode ----
    def _encode(self, s):
        return [self.stoi.get(c, self.unk) for c in s]

    def _decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    # ---- forward (identical math to the torch trainer, run forward-only) ----
    def _sigmoid(self, x):
        return 1.0 / (1.0 + self._np.exp(-x))

    def _ln(self, x, g, b):
        np = self._np
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)  # population variance == torch F.layer_norm
        return (x - mu) / np.sqrt(var + self.eps) * g + b

    def _shift(self, x):
        np = self._np
        xp = np.zeros_like(x)
        xp[1:] = x[:-1]  # token-shift: row t <- x[t-1], row 0 == 0
        return xp

    def _timemix(self, x, l):
        w = self.w
        xn = self._ln(x, w[f"tln_g.{l}"], w[f"tln_b.{l}"])
        xp = self._shift(xn)
        mk, mv, mr = w[f"t_mix_k.{l}"], w[f"t_mix_v.{l}"], w[f"t_mix_r.{l}"]
        xk = xn * mk + xp * (1.0 - mk)
        xv = xn * mv + xp * (1.0 - mv)
        xr = xn * mr + xp * (1.0 - mr)
        k = xk @ w[f"t_Wk.{l}"]
        v = xv @ w[f"t_Wv.{l}"]
        r = self._sigmoid(xr @ w[f"t_Wr.{l}"])
        gate = self._sigmoid(k)  # bounded input gate (stable, numpy-parity-safe)
        decay = self._sigmoid(w[f"t_decay.{l}"])  # per-channel positional weight decay in (0,1)
        n, dim = x.shape
        wkv = self._np.empty_like(v)
        s = self._np.zeros(dim)
        for t in range(n):  # LIF-gated linear recurrence, O(1)/token state
            s = decay * s + gate[t] * v[t]
            wkv[t] = s
        out = (r * wkv) @ w[f"t_Wo.{l}"]
        return self._lif(out, self.t_beta[l])

    def _channelmix(self, x, l):
        np = self._np
        w = self.w
        xn = self._ln(x, w[f"cln_g.{l}"], w[f"cln_b.{l}"])
        xp = self._shift(xn)
        mk, mr = w[f"c_mix_k.{l}"], w[f"c_mix_r.{l}"]
        xk = xn * mk + xp * (1.0 - mk)
        xr = xn * mr + xp * (1.0 - mr)
        kk = np.maximum(xk @ w[f"c_Wk.{l}"], 0.0)
        kk = kk * kk  # squared-ReLU channel activation (RWKV RFFN)
        kv = kk @ w[f"c_Wv.{l}"]
        r = self._sigmoid(xr @ w[f"c_Wr.{l}"])
        return self._lif(r * kv, self.c_beta[l])

    def _lif(self, cur, beta):
        np = self._np
        n, dim = cur.shape
        spk = np.empty_like(cur)
        mem = np.zeros(dim)
        for t in range(n):  # leaky integrate-and-fire, subtractive reset
            mem = beta * mem + cur[t]
            fire = (mem > self.thr).astype(np.float64)
            spk[t] = fire
            mem = mem - fire * self.thr
        return spk

    def _forward_logits(self, ids):
        np = self._np
        x = self.w["emb"][np.asarray(ids, dtype=np.int64)]
        for l in range(self.layers):
            x = x + self._timemix(x, l)
            x = x + self._channelmix(x, l)
        xo = self._ln(x, self.w["ln_out_g"], self.w["ln_out_b"])
        return xo @ self.w["head_W"] + self.w["head_b"]

    def _log_softmax(self, logits):
        np = self._np
        z = logits - logits.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    # ---- public API ----
    def logprob(self, sentence):
        """Full-sentence log-probability = sum over positions of log P(char_t | chars_<t). An implicit
        space start-token lets the first real char be scored. Higher == more acceptable. This is the
        ORDER-SENSITIVE quantity BLiMP needs. Returns None when unavailable (never a fabricated score)."""
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
        ("" when unavailable). Re-runs the causal forward each step; the architecture supports O(1)/token
        streaming state, but the simple re-run keeps generate provably consistent with logprob."""
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
# Self-check: fast stdlib+numpy proof of the forward-serve + order-sensitivity + honest-unavailable
# paths on a random-init frozen artifact. With --train (and torch present) it ALSO trains a tiny model
# inline via training/ar_spiking_lm.py and asserts the STRICT learned order proof (in-dist > shuffled).
# ---------------------------------------------------------------------------
_TINY_TEXT = (
    "the spiking network reads the sentence one character at a time and it never looks ahead "
    "so the order of the letters changes the score that the model gives to the whole sentence "
    "a cat sat on the mat while the dog ran across the yard in the bright morning sun and then "
    "she sells sea shells by the sea shore and the shells shine softly in the pale evening light "
) * 3


def _try_train_inline(art, text):
    """Best-effort: train a tiny model via the torch [train] tier for the strict order proof.
    Returns True if a trained artifact was written, False if torch/the trainer is unavailable."""
    try:
        import torch  # noqa: F401  (probe only; this module never depends on torch)
    except Exception:
        return False
    # sys.path is RESTORED, not left mutated (fixed 2026-08-05). This used to `sys.path.insert(0,
    # <repo>/training)` and never remove it, so one self-check permanently re-pointed the import
    # search order of the WHOLE process -- the module-scope-side-effect defect P4c records against
    # the daemon. Invisible under `run_selfchecks.run_all()` (subprocess per module), live under
    # `run_one_in_process()` (the frozen-app path, one process), where `training/` would then
    # shadow same-named top-level modules for every module that ran afterwards.
    # Removing it after the import is safe: `training/ar_spiking_lm.py` imports only stdlib, torch
    # and `snn_data_lake.*` at module scope (verified 2026-08-05), and the module object is cached
    # in `sys.modules` -- `trainer.train()` below never re-searches the path.
    tdir = os.path.join(_ROOT, "training")
    _inserted = tdir not in sys.path
    if _inserted:
        sys.path.insert(0, tdir)
    try:
        import ar_spiking_lm as trainer

        trainer.train(text, dim=24, layers=2, ctx=48, steps=200, seed=0, out_path=art, log_every=0)
        return True
    except Exception as exc:
        print(
            f"  [note] inline torch train unavailable ({type(exc).__name__}); using random-init artifact"
        )
        return False
    finally:
        if _inserted:
            try:
                sys.path.remove(tdir)
            except ValueError:  # someone else already removed it -- nothing to undo
                pass


def _selfcheck(with_train=False):
    np = _load_numpy()
    if np is None:
        print("SKIP ar_spiking_lm_serve self-check: numpy unavailable")
        return 0
    import tempfile

    print("=" * 92)
    print(
        "SPIKING-RWKV LM SERVE -- forward-only numpy; full-sentence AR log-prob is the order lever"
    )
    print("=" * 92)
    workdir = tempfile.mkdtemp()
    art = os.path.join(workdir, "tiny_spiking_lm.npz")
    text = _TINY_TEXT

    trained = _try_train_inline(art, text) if with_train else False
    if not trained:
        build_random_artifact(art, text, dim=24, layers=2, seed=0)
    print(f"artifact: {os.path.basename(art)} (trained={trained})")

    srv = SpikingLMServe(art, seed=0)
    assert srv.available() is True, "a valid frozen artifact must load"

    # (1b) PROVENANCE travels IN the artifact -- and a pre-schema artifact still loads, reporting
    # every field "unknown" instead of crashing (backward compatibility is part of the contract).
    prov = srv.provenance()
    assert prov is not None and set(prov) == set(PROVENANCE_KEYS), (
        "provenance() must return every schema key"
    )
    if trained:
        assert prov["corpus_chars"] == len(text), "trained artifact must carry its corpus size"
        assert prov["checkpoint_step"] != "unknown", "trained artifact must carry its checkpoint"
        print(
            "  provenance (from the artifact itself): corpus_chars=%s steps_planned=%s "
            "checkpoint_step=%s heldout_ppl=%s"
            % (
                prov["corpus_chars"],
                prov["steps_planned"],
                prov["checkpoint_step"],
                prov["heldout_ppl"],
            )
        )
    else:
        assert all(v == "unknown" for v in prov.values()), (
            "an artifact without meta['provenance'] must serve every field as 'unknown'"
        )
        print("  provenance: pre-schema artifact -> every field 'unknown' (loads, never crashes)")

    # (2) ORDER-SENSITIVITY -- the whole point vs the order-invariant SLDA-RFF.
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
        print(
            "  ORDER PROOF (trained): in-dist logprob > shuffled logprob -> LEARNED order-sensitivity."
        )
    else:
        assert abs(lp - lps) > 1e-3, (
            "shuffling MUST change the AR logprob (i.e. NOT order-invariant)"
        )
        print("  ORDER PROOF (random-init): shuffling CHANGES the AR logprob -> the model is NOT")
        print(
            "  order-invariant (structurally UNLIKE SLDA-RFF). Strict in-dist>shuffled needs training."
        )

    # (3) generate returns a non-empty ASCII string.
    gen = srv.generate("the ", max_tokens=24)
    print(f"  generate('the ', 24) -> {gen!r}")
    assert isinstance(gen, str) and len(gen) > 0, "generate must return a non-empty string"
    assert all(ord(c) < 128 for c in gen), "generate output must be ASCII"

    # (4) no artifact -> honest-unavailable (available() False, logprob None), never faked.
    miss = SpikingLMServe(os.path.join(workdir, "does_not_exist.npz"))
    assert miss.available() is False, "a missing artifact must report unavailable"
    assert miss.logprob("hello world") is None, "unavailable logprob must be None, not a fake score"
    assert miss.generate("hello") == "", "unavailable generate must return empty string"
    assert miss.provenance() is None, "unavailable provenance must be None, not fabricated fields"
    print(
        "  honest-unavailable: missing artifact -> available()=False, logprob()=None (no fake score)."
    )

    try:
        os.remove(art)
    except OSError:
        pass
    try:
        os.rmdir(workdir)
    except OSError:
        pass

    print("-" * 92)
    print("self-check OK: forward-only numpy serve reloads a frozen Spiking-RWKV artifact; the")
    print(
        "full-sentence AR log-prob is ORDER-SENSITIVE -- the BLiMP lever the order-invariant SLDA-RFF"
    )
    print(
        "structurally lacks. HONEST: this proves the ARCHITECTURE + serve PIPELINE only. Moving BLiMP"
    )
    print(
        "off its ~57 pct floor needs a REAL-SCALE train (training/ar_spiking_lm.py, GPU, ~GPT-2-small,"
    )
    print(
        "10-100M words) exported to models/ar_spiking_lm.npz -- the SAME path DEFAULT_ARTIFACT loads;"
    )
    print("a toy model sits at n-gram ~60 pct.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck(with_train=("--train" in sys.argv)))
