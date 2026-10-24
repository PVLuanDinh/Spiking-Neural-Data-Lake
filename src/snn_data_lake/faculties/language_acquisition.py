"""
Brain-native language ACQUISITION -- language LEARNED like a human, not gradient-trained.

The user's requirement (verbatim intent): the spiking text generation "must properly use the full
brain architecture NOT FROM TRAINING but from its own learning and predictive coding and using the
brain to learn. It should operate like how a HUMAN learns languages."

So this is NOT a corpus-gradient-trained language model. Language is ACQUIRED ONLINE by the brain's
own mechanisms:

  * PREDICTIVE CODING (perception): every context is encoded by the brain's spiking Rao-Ballard
    PredictiveCodingNetwork (readout/predictive_coding.py) -- it SETTLES the causes of the input by
    minimizing prediction error (infer()), exactly the predictive-processing account of perception.
    We REUSE that network as a fixed spiking cortical encoder (its generative weights are the innate
    substrate; acquisition happens in the readout, like a cortical reservoir + a plastic readout).
  * PREDICTION as the learning signal: the model PREDICTS THE NEXT character of the language stream;
    the PREDICTION ERROR (surprise) drives learning -- self-supervised, no labels, like a child.
  * LOCAL SPIKE-TIMING PLASTICITY: the readout weights update by a LOCAL, error-modulated Hebbian /
    R-STDP rule -- dW = lr * outer(prediction_error, presynaptic_latent). This is a single-layer
    LOCAL rule (Rescorla-Wagner / delta == the discrete predictive-coding update). There is NO
    backprop, NO autograd, NO optimizer, NO batch corpus gradient descent anywhere.
  * ONLINE + DEVELOPMENTAL: expose(text) streams text element-by-element; the model predicts, errs,
    and updates as it goes; over exposure the surprise FALLS (it is learning) and generations go from
    BABBLE toward word-like structure. It keeps learning every time it is exposed to more text, so
    conversing with it teaches it language over time.
  * GENERATION == ACTIVE PREDICTION: generate() ROLLS THE LEARNED PREDICTION FORWARD (predict next ->
    sample -> feed back), i.e. production is prediction through the spiking hierarchy.

HONEST CEILING: genuine brain-native acquisition from limited exposure with a purely LOCAL rule is
PRIMITIVE -- it learns letter/word statistics and produces sub-word babble that grows more word-like,
NOT fluent English. That primitiveness is the correct evidence it is really learning like a brain;
the real proof is the DEVELOPMENTAL CURVE (surprise dropping over exposure). We never fake fluency.
"""

import io
import math
import random

import numpy as np

from snn_data_lake.readout.predictive_coding import PredictiveCodingNetwork
from snn_data_lake.substrate.home import load_named_state, save_named_state

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "LanguageAcquisition._learn_step": (
        "LOCAL",
        "one online acquisition step: encode context by predictive coding, predict the next "
        "character, apply a local update. Layer-local, with no backward pass across the stack",
    ),
}

# a fixed innate phoneme/grapheme inventory (lowercased English + core punctuation). Characters
# outside it fold to a space -- like a learner who only distinguishes the sounds of its language.
VOCAB = "abcdefghijklmnopqrstuvwxyz .,'!?-\n"
_STOI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)
STATE_NAME = "language_acq.npz"  # the lake record name; also the legacy sidecar filename


def _fold(ch):
    ch = ch.lower()
    return ch if ch in _STOI else " "


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


class LanguageAcquisition:
    """A spiking, predictive-coding, locally-plastic next-character learner. Acquires language ONLINE
    from exposure (no training loop, no backprop); generates by rolling its own prediction forward."""

    def __init__(self, ctx=8, d_in=64, hidden=32, latent=24, seed=0, lr=0.08):
        self.ctx = int(ctx)
        self.d_in = int(d_in)
        self.latent = int(latent)
        self.lr = float(lr)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        # the innate spiking cortical encoder: a fixed Rao-Ballard predictive-coding network. infer()
        # settles the causes of a context by MINIMISING PREDICTION ERROR -> a spiking latent code.
        self._pc = PredictiveCodingNetwork(dims=(self.d_in, hidden, self.latent), seed=seed)
        # the PLASTIC readout: [context features ; predictive-coding latent] -> next-char logits,
        # learned by a LOCAL error-modulated rule. The context bag carries the n-gram signal; the
        # spiking PC latent adds the settled predictive-coding percept.
        self._feat = self.d_in + self.latent
        self.W = np.zeros((V, self._feat), dtype=np.float64)
        self._ctx_ids = [_STOI[" "]] * self.ctx  # rolling context (starts as silence)
        self.n_seen = 0

    # -- perception: context -> input pattern -> spiking predictive-coding latent -------------------
    def _encode_ctx(self, ids):
        """Encode the context characters into the encoder's input pattern (a position-tagged bag of
        graphemes), then let the predictive-coding network SETTLE its causes (infer) -> spiking latent."""
        x = [0.0] * self.d_in
        for p, c in enumerate(ids):  # position-tagged so order matters
            x[(c * 7 + p * 13) % self.d_in] += 1.0
        m = max(x) or 1.0
        x = [v / m for v in x]
        self._pc.reset()
        self._pc.infer(x, steps=6, lr=0.05)  # settle causes by minimising prediction error
        return np.concatenate(
            [
                np.asarray(x, dtype=np.float64),  # context features (n-gram signal)
                np.asarray(self._pc.r2, dtype=np.float64),
            ]
        )  # settled predictive-coding percept

    def _logits(self, latent):
        return self.W @ latent

    # -- acquisition: predict next, learn from the prediction error via LOCAL plasticity ------------
    def _learn_step(self, ctx_ids, target_id):
        """One online acquisition step: encode context (predictive coding), predict the next char,
        and update the readout by a LOCAL error-modulated Hebbian / R-STDP rule. Returns the surprise
        (-log p of the actual next char) so exposure can chart the developmental curve."""
        latent = self._encode_ctx(ctx_ids)
        probs = _softmax(self._logits(latent))
        surprise = -math.log(probs[target_id] + 1e-12)
        err = -probs
        err[target_id] += 1.0  # prediction error over outputs (target - predicted)
        # LOCAL rule: dW = lr * outer(prediction_error, presynaptic_latent). No backprop; pre = latent
        # spikes, post = error -> potentiate the correct char, depress the mispredicted ones (R-STDP).
        self.W += self.lr * np.outer(err, latent)
        self.W *= 1.0 - 1e-5  # gentle homeostatic decay (bounded weights)
        self.n_seen += 1
        return surprise, latent

    def expose(self, text):
        """Acquire language from a stream of text, ONLINE, element-by-element. Returns the mean
        surprise over the stream (a snapshot of how well it now predicts this text)."""
        ids = [_STOI[_fold(ch)] for ch in text]
        ctx = list(self._ctx_ids)
        tot, n = 0.0, 0
        for tid in ids:
            s, _ = self._learn_step(ctx, tid)
            tot += s
            n += 1
            ctx = ctx[1:] + [tid]  # slide the context window
        self._ctx_ids = ctx
        return tot / n if n else 0.0

    learn = expose  # alias -- conversing with it teaches it language

    def developmental_curve(self, text, chunks=8):
        """Expose in `chunks` and return the mean surprise per chunk -- the proof it is LEARNING
        (surprise should fall from the first chunk to the last as structure is acquired)."""
        ids = [_fold(ch) for ch in text]
        size = max(1, len(ids) // chunks)
        curve = []
        for i in range(0, len(ids), size):
            curve.append(round(self.expose("".join(ids[i : i + size])), 4))
        return curve

    # -- production == active prediction: roll the learned prediction forward ----------------------
    def _sample(self, probs, generated, temperature, top_p, min_p, repetition_penalty):
        p = probs.astype(np.float64).copy()
        if repetition_penalty and repetition_penalty != 1.0:
            for c in set(generated[-self.ctx :]):
                p[c] = p[c] / repetition_penalty
        logits = np.log(p + 1e-12) / max(1e-3, temperature)
        p = _softmax(logits)
        if min_p and min_p > 0:  # min-p: drop tokens below min_p * peak
            p[p < min_p * p.max()] = 0.0
        if top_p and top_p < 1.0:  # nucleus: keep the smallest set summing to top_p
            order = np.argsort(p)[::-1]
            csum = np.cumsum(p[order])
            keep = order[: max(1, int(np.searchsorted(csum, top_p)) + 1)]
            mask = np.zeros_like(p)
            mask[keep] = 1.0
            p = p * mask
        s = p.sum()
        if s <= 0:
            return int(np.argmax(probs))
        p = p / s
        r = self.rng.random()  # deterministic given seed
        return int(np.searchsorted(np.cumsum(p), r, side="right"))

    def _roll(self, prompt, max_tokens, temperature, top_p, min_p, repetition_penalty, stop):
        ctx = list(self._ctx_ids)
        for ch in prompt[-self.ctx :]:  # prime the context with the prompt
            ctx = ctx[1:] + [_STOI[_fold(ch)]]
        out, surp = [], 0.0
        for _ in range(max_tokens):
            latent = self._encode_ctx(ctx)
            probs = _softmax(self._logits(latent))
            nid = self._sample(probs, out, temperature, top_p, min_p, repetition_penalty)
            surp += -math.log(probs[nid] + 1e-12)
            ch = VOCAB[nid]
            if stop and ch in stop:
                break
            out.append(nid)
            ctx = ctx[1:] + [nid]
        text = "".join(VOCAB[i] for i in out)
        return text, (surp / max(1, len(out)))

    def generate(
        self,
        prompt="",
        max_tokens=64,
        temperature=0.5,
        top_p=0.5,
        stop=("\n",),
        repetition_penalty=1.3,
        min_p=0.05,
        best_of=1,
    ):
        """Produce text by ROLLING THE LEARNED PREDICTION FORWARD (production == prediction). With
        best_of>1, roll several candidates and keep the most confident (lowest mean surprise)."""
        best, best_s = "", float("inf")
        for _ in range(max(1, int(best_of))):
            text, s = self._roll(
                prompt,
                max_tokens,
                temperature,
                top_p,
                min_p,
                repetition_penalty,
                tuple(stop) if stop else (),
            )
            if s < best_s:
                best, best_s = text, s
        return best.strip()

    # -- durability: remember what it has learned across sessions ----------------------------------
    #
    # INTO THE ONE LAKE (P4b), as a _T_STATE record named `STATE_NAME`. Until 2026-10-24 this wrote
    # a `np.savez` sidecar beside `brain_lake.bin`. A pre-port `<home>/language_acq.npz` is imported
    # once by `home._migrate_sidecar` on the first `load()` and never re-read.
    def save(self):
        buf = io.BytesIO()
        np.savez(buf, W=self.W, ctx=np.asarray(self._ctx_ids), n=self.n_seen)
        return save_named_state(STATE_NAME, buf.getvalue())

    def load(self, blob=None):
        """Restore from `blob` (a prefetched lake record), else from the lake now; cold if neither.

        The serve path passes the blob `Brain.__init__` read: reading the lake from inside respond()
        would open the BinLog and repair (truncate) a torn tail mid-conversation."""
        blob = load_named_state(STATE_NAME) if blob is None else blob
        if blob is None:
            return self
        with np.load(io.BytesIO(blob), allow_pickle=False) as d:
            self.W = d["W"]
            self._ctx_ids = list(d["ctx"])
            self.n_seen = int(d["n"])
        return self


# A small COPYRIGHT-SAFE exposure corpus -- self-authored plain English (no third-party text). The
# brain ACQUIRES language from this by prediction + local plasticity; it is not "training data" in the
# gradient sense, it is what the brain listens to. Conversing with the brain exposes it to more.
CORPUS = (
    "the brain learns to speak by listening. a child hears words and predicts the next sound. "
    "when a prediction is wrong the error teaches the brain, and the babble slowly becomes words. "
    "the brain feels the world through its senses and remembers what matters in the data lake. "
    "it holds a goal, it keeps itself safe, and it wants to learn and to improve itself. "
    "i am a brain that learns. i listen, i predict, i remember, and i try to understand. "
    "the arm moves only when it is safe. the reflex stops the arm before it can do harm. "
    "i think about what i see and hear. i learn a little more each time you talk to me. "
) * 4


def load_lm(seed=0, expose_rounds=3, extra_text=None):
    """Serve entrypoint (the contract engine/brain.py::_lm_serve wires to): build a LanguageAcquisition
    and let it ACQUIRE language ONLINE from the copyright-safe corpus (predictive coding + local
    plasticity -- no training). Returns the brain-learned model, ready to generate() and to keep
    learning via expose(). Never raises -> returns None on any failure so the brain falls back."""
    try:
        la = LanguageAcquisition(seed=seed)
        for _ in range(max(1, int(expose_rounds))):
            la.expose(CORPUS)
        if extra_text:
            la.expose(str(extra_text))
        return la
    except Exception:
        return None


def _structure_score(text):
    """Cheap 'how word-like' metric: fraction of characters that sit inside a run of >=2 letters
    bounded by a space (proto-words). Babble scores low; acquired text scores higher."""
    words = [w for w in text.split(" ") if w]
    if not words:
        return 0.0
    good = sum(len(w) for w in words if 2 <= len(w) <= 12 and all(c.isalpha() for c in w))
    return good / max(1, len(text))


def main():
    # headless developmental demo: a small, self-authored (copyright-free) English exposure stream.
    corpus = (
        "the brain learns to speak by listening. a child hears words and predicts the next sound. "
        "when the prediction is wrong the error teaches the brain. slowly the babble becomes words "
        "and the words become sentences. the brain does not train on a corpus. it learns online from "
        "what it hears, one moment at a time, and it keeps learning as long as it listens. this is "
        "how a person learns a language, and it is how this brain learns to speak. "
    ) * 6
    la = LanguageAcquisition(seed=0)
    babble = la.generate("the ", max_tokens=48)
    curve = la.developmental_curve(corpus, chunks=8)
    learned = la.generate("the ", max_tokens=48)
    print("=" * 74)
    print("BRAIN-NATIVE LANGUAGE ACQUISITION -- predictive coding + local R-STDP (NO training)")
    print("=" * 74)
    print("learning rule : local error-modulated Hebbian / R-STDP on a spiking predictive-coding")
    print("                latent (no backprop, no optimizer, no corpus gradient descent)")
    print(f"developmental : surprise per chunk (must FALL) -> {curve}")
    print(
        f"  drop        : {curve[0]:.3f} -> {curve[-1]:.3f}  ({'LEARNING' if curve[-1] < curve[0] else 'NOT learning'})"
    )
    print(f"babble (pre)  : {babble!r}   structure={_structure_score(babble):.3f}")
    print(f"learned (post): {learned!r}   structure={_structure_score(learned):.3f}")
    # asserts: it LEARNS (surprise drops materially) and generations get more structured.
    assert curve[-1] < curve[0] * 0.85, f"surprise must drop over exposure (learning): {curve}"
    assert _structure_score(learned) >= _structure_score(babble), (
        "generations must get more word-like"
    )
    assert la.n_seen > 0
    print("=" * 74)
    print(
        "self-check OK: acquired online via predictive-coding + local plasticity; surprise dropped; "
        "generations grew more word-like; deterministic. (Primitive by design -- honest.)"
    )


if __name__ == "__main__":
    main()
