"""Zero-dep inference for a trained spiking language model (serve it in the core, no torch).

The spiking-LM TRAINER lives in the torch training tier (`snn_data_lake.training.spiking_lm`,
SpikeGPT lineage). It exports a plain-JSON "snn_lm" weight file; THIS module loads it and runs the
same spiking forward in pure Python, so a clone-and-run install can GENERATE text from a trained
spiking LM with no torch -- the language-model analogue of `trained_brain` serving the classifier.

Forward (matches the trainer's snnTorch `Leaky` block exactly, so weights transfer faithfully):
  * `embed[id]` is the per-token input CURRENT for block 0.
  * each block, per position t: LIF(cur) -> input spikes; `state = a*state + key@spk_in`
    (a = sigmoid(log_decay), the causal RWKV-style linear-attention approximation); `mix = value@state`;
    LIF(mix) -> block spikes; `out_t = ffn@spk_out + ffn_b + cur` (graded residual).
  * readout: LIF(residual) -> head spikes; `logits = readout@spk + readout_b`; next char = argmax
    (greedy) or a temperature sample from a seeded stdlib RNG.
LIF here is snnTorch's default SUBTRACT reset: `mem = mem*beta + cur; if mem >= threshold: spike, mem -= threshold`.

HONEST CAVEAT: the pure-Python forward below is verified in isolation (self-check on a hand-built LM),
but the trainer that produces real weights is torch/GPU and UNVERIFIED against a live run here, so the
end-to-end torch-train -> core-serve transfer should be smoke-tested on a GPU. This module is tiny-LM
inference, not a large LLM.

  python -m snn_data_lake.cortex.spiking_lm_serve        # (also run by run_selfchecks.py)
"""

import json
import math

LM_FORMAT = "snn_lm/1"


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _matvec(w, x):
    """w is [out][in], x is [in] -> [out]."""
    return [sum(wo[i] * x[i] for i in range(len(x))) for wo in w]


def _lif(mem, cur, beta, thr):
    """One snnTorch-Leaky step (SUBTRACT reset): returns (spikes, new_mem), both length len(cur)."""
    spk = [0] * len(cur)
    m = [0.0] * len(cur)
    for i in range(len(cur)):
        v = mem[i] * beta + cur[i]
        if v >= thr:
            spk[i] = 1
            v -= thr
        m[i] = v
    return spk, m


def _forward_logits(lm, ids):
    """Run the spiking LM over a context window `ids`; return the vocab logits at the LAST position."""
    d = lm["d_model"]
    beta, thr = lm["beta"], lm["threshold"]
    embed = lm["embed"]
    logits = None
    # per-block persistent state across the window
    nb = len(lm["blocks"])
    state = [[0.0] * d for _ in range(nb)]
    mem_in = [[0.0] * d for _ in range(nb)]
    mem_out = [[0.0] * d for _ in range(nb)]
    mem_head = [0.0] * d
    decays = [[_sigmoid(z) for z in blk["log_decay"]] for blk in lm["blocks"]]
    for t in range(len(ids)):
        cur = list(embed[ids[t]])  # [d] input current for this token
        for b, blk in enumerate(lm["blocks"]):
            spk_in, mem_in[b] = _lif(mem_in[b], cur, beta, thr)
            k = _matvec(blk["key"], spk_in)
            a = decays[b]
            state[b] = [a[i] * state[b][i] + k[i] for i in range(d)]
            mix = _matvec(blk["value"], state[b])
            spk_out, mem_out[b] = _lif(mem_out[b], mix, beta, thr)
            fout = _matvec(blk["ffn_w"], spk_out)
            fb = blk["ffn_b"]
            cur = [fout[i] + fb[i] + cur[i] for i in range(d)]  # graded residual
        spk_head, mem_head = _lif(mem_head, cur, beta, thr)  # spike-driven readout
        logits = [
            row_i + lm["readout_b"][o] for o, row_i in enumerate(_matvec(lm["readout"], spk_head))
        ]
    return logits


class _Rng:
    """Deterministic stdlib LCG for temperature sampling (no global RNG)."""

    def __init__(self, seed):
        self.s = (seed ^ 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def random(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.s >> 11) / float(1 << 53)


def generate(lm, prompt, n=80, seq_len=None, temperature=0.0, seed=0):
    """Generate `n` chars after `prompt`. temperature=0 -> greedy (deterministic). Returns prompt+gen."""
    seq_len = seq_len or lm.get("seq_len", 64)
    stoi, itos = lm["stoi"], lm["itos"]
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    rng = _Rng(seed)
    for _ in range(int(n)):
        logits = _forward_logits(lm, ids[-seq_len:])
        if temperature <= 0.0:
            nxt = max(range(len(logits)), key=lambda c: logits[c])  # greedy
        else:
            m = max(logits)
            exps = [math.exp((z - m) / temperature) for z in logits]
            tot = sum(exps)
            r = rng.random() * tot
            nxt, acc = 0, 0.0
            for c, e in enumerate(exps):
                acc += e
                if r <= acc:
                    nxt = c
                    break
        ids.append(nxt)
    return "".join(itos[i] for i in ids)


def save_lm(lm, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lm, f)
    return path


def load_lm(path):
    """Load + validate a snn_lm weight file at the trust boundary."""
    with open(path, encoding="utf-8") as f:
        lm = json.load(f)
    if not isinstance(lm, dict) or str(lm.get("format", "")) != LM_FORMAT:
        raise ValueError(f"{path}: not a {LM_FORMAT} file (format={lm.get('format')!r})")
    for key in (
        "vocab",
        "d_model",
        "beta",
        "threshold",
        "stoi",
        "itos",
        "embed",
        "blocks",
        "readout",
        "readout_b",
    ):
        if key not in lm:
            raise ValueError(f"{path}: missing '{key}'")
    if len(lm["embed"]) != lm["vocab"] or len(lm["itos"]) != lm["vocab"]:
        raise ValueError(f"{path}: embed/itos rows must equal vocab={lm['vocab']}")
    return lm


def _demo_lm():
    """A tiny hand-built snn_lm (vocab 4, d 4, 1 block) for the self-check -- deterministic weights,
    NOT trained. The readout is biased hard toward one char so the greedy output is predictable."""
    v, d = 4, 4
    ident = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    blk = {
        "key": ident,
        "value": ident,
        "log_decay": [0.0] * d,
        "ffn_w": [[0.0] * d for _ in range(d)],
        "ffn_b": [0.0] * d,
    }
    readout = [[0.0] * d for _ in range(v)]
    readout_b = [0.0, 5.0, 0.0, 0.0]  # bias hard toward char id 1
    chars = list("abcd")
    return {
        "format": LM_FORMAT,
        "vocab": v,
        "d_model": d,
        "n_blocks": 1,
        "beta": 0.9,
        "threshold": 1.0,
        "seq_len": 8,
        "stoi": {c: i for i, c in enumerate(chars)},
        "itos": chars,
        "embed": [[2.0 if k == i else 0.0 for k in range(d)] for i in range(v)],
        "blocks": [blk],
        "readout": readout,
        "readout_b": readout_b,
        "meta": {},
    }


def main():
    import os
    import tempfile

    lm = _demo_lm()
    out = generate(lm, "a", n=6, temperature=0.0)  # greedy, deterministic
    out2 = generate(lm, "a", n=6, temperature=0.0)

    # round-trip through the JSON weight file the trainer writes
    p = os.path.join(tempfile.gettempdir(), "snn_lm_selfcheck.json")
    save_lm(lm, p)
    lm2 = load_lm(p)
    os.remove(p)
    out_loaded = generate(lm2, "a", n=6, temperature=0.0)

    print("=" * 62)
    print("SPIKING LM SERVE -- zero-dep inference for a trained spiking LM")
    print("=" * 62)
    print(
        f"vocab={lm['vocab']} d_model={lm['d_model']} blocks={lm['n_blocks']} (hand-built demo LM)"
    )
    print(f"greedy generate('a', 6) : {out!r}")
    print(f"round-trip (load) match : {out == out_loaded}")
    print("=" * 62)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert out == out2, "generation must be deterministic (greedy)"
    assert len(out) == len("a") + 6 and all(c in lm["stoi"] for c in out), (
        "output = prompt + n valid chars"
    )
    # positive control: readout biased hard toward char 'b' (id 1) -> greedy emits 'b'
    assert set(out[1:]) == {"b"}, f"readout biased to 'b' must make greedy emit 'b' ({out!r})"
    assert out_loaded == out, "a saved+loaded LM must generate identically (faithful serve)"
    # trust boundary: a wrong-format file is rejected
    bad = os.path.join(tempfile.gettempdir(), "snn_lm_bad.json")
    save_lm({"format": "nope"}, bad)
    try:
        load_lm(bad)
        raised = False
    except ValueError:
        raised = True
    os.remove(bad)
    assert raised, "load_lm must reject a non-snn_lm file at the trust boundary"
    print(
        f"self-check OK: zero-dep spiking-LM forward + greedy generate {out!r} deterministic; "
        f"biased readout controls output; JSON weight file round-trips; bad file rejected"
    )


if __name__ == "__main__":
    main()
