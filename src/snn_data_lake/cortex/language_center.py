"""Serve the REAL pretrained SpikeGPT-216M spiking language model on the GPU.

This wraps the genuine, published SpikeGPT-216M checkpoint -- an RWKV-style spiking
language model (18 layers, 768-dim, ~216M params) pretrained on ~5B tokens of
OpenWebText, released under BSD-2 / Apache-2.0 upstream. It is a BASE completion
model (no instruction tuning): it continues text, it does not chat or follow
instructions on its own. Served here on the RTX 5070 via the vendored SpikeGPT
inference code (``vendor/SpikeGPT/src``); no model math is reimplemented -- the
weights and the forward/generate loop are the upstream project's.

Role in the wider system: this is the brain's language-generation organ. It is
fluent but ungrounded on its own; the caller supplies a prompt built from the
spiking brain's real state, so the words are anchored to that state rather than
invented. Torch and the vendored SpikeGPT package import lazily inside __init__
so that importing this module (and calling ``available()``) never needs torch.

    NORD_GPU / typical use (run on the GPU box):
        PYTHONPATH=src python -m snn_data_lake.cortex.language_center
"""

import importlib.util
import os
import sys
import types


class SpikeGPTServe:
    """Load SpikeGPT-216M once and generate text continuations from a prompt."""

    N_LAYER = 18
    N_EMBD = 768
    CTX_LEN = 1024
    VOCAB_SIZE = 50277

    def __init__(self, root=None, device="cuda", float_mode="fp32", seed=0):
        self.root = self._resolve_root(root)
        self.seed = int(seed)
        vend = os.path.join(self.root, "vendor", "SpikeGPT")

        os.environ["RWKV_JIT_ON"] = "0"
        os.environ["RWKV_RUN_DEVICE"] = device
        os.environ["RWKV_FLOAT_MODE"] = float_mode
        if vend not in sys.path:
            sys.path.insert(0, vend)

        import numpy as np
        import torch
        from src.model_run import RWKV_RNN
        from src.utils import TOKENIZER

        self._np = np
        self._torch = torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        os.environ["RWKV_RUN_DEVICE"] = device

        model_name = os.path.join(self.root, "models", "spikegpt", "SpikeGPT-216M")
        args = types.SimpleNamespace(
            RUN_DEVICE=device,
            FLOAT_MODE=float_mode,
            MODEL_NAME=model_name,  # RWKV_RNN appends '.pth'
            n_layer=self.N_LAYER,
            n_embd=self.N_EMBD,
            ctx_len=self.CTX_LEN,
            vocab_size=self.VOCAB_SIZE,
            head_qk=0,
            pre_ffn=0,
            grad_cp=0,
            my_pos_emb=0,
        )
        self._model = RWKV_RNN(args)

        tok_path = os.path.join(vend, "20B_tokenizer.json")
        self._tok = TOKENIZER([tok_path, tok_path], UNKNOWN_CHAR=None)

    @staticmethod
    def _resolve_root(root=None):
        if root:
            return os.path.abspath(root)
        # this file lives at src/snn_data_lake/cortex/language_center.py; the project root is FOUR
        # directories up (cortex -> snn_data_lake -> src -> root) so vendor/SpikeGPT + models/ resolve.
        here = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))

    @staticmethod
    def available(root=None):
        """True iff torch is importable and the weights + vendored code are present."""
        root = SpikeGPTServe._resolve_root(root)
        try:
            if importlib.util.find_spec("torch") is None:
                return False
        except Exception:
            return False
        weights = os.path.join(root, "models", "spikegpt", "SpikeGPT-216M.pth")
        model_run = os.path.join(root, "vendor", "SpikeGPT", "src", "model_run.py")
        return os.path.exists(weights) and os.path.exists(model_run)

    def generate(self, prompt, max_tokens=32, temperature=0.5, top_p=0.5, stop=None,
                 greedy=False, repetition_penalty=1.3, min_p=0.05, best_of=1):
        """Return ONLY the newly generated continuation of ``prompt`` (never the prompt).

        Decoding refinements (all in-house, no vendored sampler): ``repetition_penalty`` damps the
        logits of tokens already emitted in the recent window (kills loops / "I don't sleep... I
        don't sleep"); ``min_p`` drops tail tokens below min_p * peak-prob (cleaner than top_p alone);
        ``top_p`` nucleus-truncates on top of that; ``best_of`` generates that many candidates (varied
        seeds) and returns the most coherent by a repetition/alphabetic score. greedy/temperature<=0
        -> deterministic argmax (best_of ignored). ``stop`` cuts at the first substring hit. Any
        failure returns "" so the caller can fall back. Deterministic for fixed seed + params."""
        try:
            torch = self._torch
            model = self._model
            tok = self._tok
            use_greedy = greedy or temperature <= 0
            prompt_ids = tok.tokenizer.encode(prompt)
            if not prompt_ids:
                return ""
            start = len(prompt_ids)

            def _pick(out, ctx):
                logits = out.float().clone()
                logits[0] = -1e9                                  # disable <|endoftext|>
                window = ctx[-96:]                                # FREQUENCY-scaled repetition penalty:
                for tid in set(window):                           # a token seen c times is hit penalty^c,
                    pen = repetition_penalty ** window.count(tid)  # so hard loops (dep, dep, dep) collapse
                    v = logits[tid]
                    logits[tid] = v / pen if v > 0 else v * pen
                if use_greedy:
                    return int(torch.argmax(logits).item())
                probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
                if min_p > 0:                                     # min-p: keep >= min_p * peak
                    probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
                if 0 < top_p < 1:                                 # nucleus on top of min-p
                    sp, si = torch.sort(probs, descending=True)
                    cut = torch.cumsum(sp, dim=-1) > top_p
                    cut[0] = False                                # always keep the top token
                    sp = torch.where(cut, torch.zeros_like(sp), sp)
                    probs = torch.zeros_like(probs).scatter_(0, si, sp)
                total = probs.sum()
                if total <= 0:
                    return int(torch.argmax(logits).item())
                return int(torch.multinomial(probs / total, 1).item())

            def _run(seed):
                torch.manual_seed(seed)
                self._np.random.seed(seed)
                ctx = list(prompt_ids)
                state = mem1 = mem2 = out = None
                n = len(ctx)
                for i in range(n):                                # preprocess the prompt
                    x = ctx[: i + 1]
                    if i == n - 1:
                        out, state, mem1, mem2 = model.forward(x, state, mem1, mem2)
                    else:
                        state, mem1, mem2 = model.forward(x, state, mem1, mem2, preprocess_only=True)
                for _ in range(max_tokens):
                    ttt = _pick(out, ctx)
                    if ttt == 0:                                  # endoftext id
                        break
                    ctx.append(ttt)
                    if stop:
                        so_far = tok.tokenizer.decode(ctx[start:])
                        hits = [so_far.find(s) for s in stop if so_far.find(s) != -1]
                        if hits:
                            return so_far[: min(hits)].strip()
                    out, state, mem1, mem2 = model.forward(ctx[-self.CTX_LEN:], state, mem1, mem2)
                return tok.tokenizer.decode(ctx[start:]).strip()

            if best_of <= 1 or use_greedy:                        # greedy is deterministic -> no re-rank
                return _run(self.seed)
            cands = [_run(self.seed + i) for i in range(best_of)]
            return max(cands, key=self._score)                    # most coherent candidate
        except Exception:
            return ""

    @staticmethod
    def _score(text):
        """Higher = more coherent: rewards distinct words + alphabetic content + a full-ish length,
        so best_of re-ranking prefers the least loopy / least degenerate candidate."""
        words = [w.lower() for w in text.split()]
        if not words:
            return -1.0
        distinct = len(set(words)) / len(words)                   # 1.0 = no repeats, low = loopy
        alpha = sum(c.isalpha() for c in text) / max(1, len(text))
        length = min(len(words), 24) / 24.0
        return distinct * 1.5 + alpha + length * 0.5

    def close(self):
        """Best-effort free of the model and CUDA cache."""
        try:
            self._model = None
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass


def _selfcheck():
    if not SpikeGPTServe.available():
        print("SpikeGPTServe: torch/weights/vendor missing -- skipping self-check")
        return
    serve = SpikeGPTServe(seed=0)
    prompt = "The meaning of life is"
    gen = serve.generate(prompt, max_tokens=12, greedy=True)
    assert isinstance(gen, str) and gen, "expected a non-empty generated string"
    assert prompt not in gen, "generation must not echo the prompt"
    gen2 = serve.generate(prompt, max_tokens=12, greedy=True)
    assert gen == gen2, "greedy generation must be deterministic"
    serve.close()
    print("PROMPT: " + prompt)
    print("GEN: " + gen)
    print("self-check OK")


if __name__ == "__main__":
    _selfcheck()
