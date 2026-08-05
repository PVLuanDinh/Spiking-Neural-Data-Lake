"""TRAINING-TIER spiking language-model TRAINER (SpikeGPT lineage) -- run via ``snn-train --lm``.

=======================  HONEST FRAMING -- READ THIS FIRST  =======================
This is the torch TRAINING tier of the spiking LM: a char-level transformer whose
neurons are LIF spiking units (snnTorch `Leaky` + `surrogate.fast_sigmoid()`
gradients), trained from scratch by surrogate-gradient BPTT. It is in the SpikeGPT /
SpikingBERT lineage (a transformer BUILT OUT OF spiking neurons), NOT a large trained
LLM. Its EXPORT BRIDGE (`export_lm` below) writes a plain-JSON ``snn_lm/1`` weight
file that the ZERO-DEP core module ``snn_data_lake.cortex.spiking_lm_serve`` loads and
serves with no torch -- the language-model analogue of `training.mnist` exporting a
`trained_brain` the core serves (`verify_transfer` below shows the round trip).

It is NOT related to AirLLM. AirLLM runs a large DENSE (non-spiking) transformer by
streaming its layers off disk; this module is the OPPOSITE -- a small transformer
made of spiking neurons and trained from scratch. They share nothing but the phrase
"language model".

torch and snntorch are NOT installed on this box and there is no GPU, so NONE of the
torch paths below have been run here -- this is an UNVERIFIED SCAFFOLD, architecture-
correct BY BEST EFFORT only. There are NO measured numbers in this file: no loss, no
perplexity, no sample text is claimed. The torch code ONLY runs when torch is present
(on the user's machine), where the numbers it prints are REAL, never fabricated here.

What it provides: the spiking-LM ARCHITECTURE (embedding -> spiking blocks ->
spike-driven readout, unrolled in time, surrogate-grad trained) plus the GROWTH
levers: a real-corpus loader (`load_corpus`), checkpoint/RESUME (`save_checkpoint` /
`load_checkpoint`), the growth metric (`perplexity`), free config knobs (d_model /
n_blocks / seq_len), and the zero-dep export. What it does NOT do: a real spiking LLM
needs large-scale training this scaffold does not do -- see the `# NOTE:` on
`SpikingLMConfig` for the honest scale ceiling. Treat every `# NOTE:` / `# VERIFY:`
as a research choice a real build would ablate and tune.

Run (needs torch + snntorch):
    pip install ".[gpu]"                          # torch + snntorch
    snn-train --lm --corpus my.txt --steps 500 --checkpoint lm.pt --export-lm my_lm.json
    python -m snn_data_lake.training.spiking_lm   # shape demo + a tiny REAL train

With no torch installed the module still imports, and `main()` prints the shape + a
clean "needs torch" note and exits 0. Ruff-excluded (training tier), NOT in CI.
"""
import json
import math
import sys
from dataclasses import dataclass

# torch + snntorch imported at module top, but GUARDED: this box has no torch, and the
# module must still IMPORT and run `main()` (which prints a clean note). When torch is
# absent, `nn` is None and the model classes below fall back to an `object` base so their
# BODIES still define -- they are only ever INSTANTIATED on the torch path, guarded in main().
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    import snntorch as snn
    from snntorch import surrogate
    _TORCH_IMPORT_ERROR = None
except ImportError as _exc:  # torch / snntorch absent -> keep module importable
    torch = None
    nn = None
    F = None  # noqa: N812
    snn = None
    surrogate = None
    _TORCH_IMPORT_ERROR = _exc

# nn.Module when torch is present, else plain `object` so `class ...(_Module)` still defines.
_Module = nn.Module if nn is not None else object


# ------------------------------- toy corpus + tokenizer -------------------------------
# A few sentences embedded as a plain string -- NO downloads, NO external data. Long enough
# to slice many length-`seq_len` training windows. This is a TOY: it only teaches the model
# character statistics of this paragraph; it is not a corpus for a real language model.
TOY_CORPUS = (
    "the spiking neuron fires when its membrane crosses the threshold, then it resets. "
    "a spiking language model reads one character at a time and predicts the next character. "
    "leaky integrate and fire neurons carry state through time, so the past shapes the future. "
    "surrogate gradients let the network learn by backpropagation through the spikes. "
    "small models learn small patterns; large models need large data and long training. "
    "this scaffold shows the architecture of an all spiking transformer on a tiny char task. "
)


class CharTokenizer:
    """Deterministic char <-> id map over the sorted unique chars of `text`. Pure stdlib
    (no torch), so `main()` can report the vocab even when torch is absent."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


@dataclass
class SpikingLMConfig:
    """Small-by-design config. ALL knobs freely settable (CLI: --d-model / --blocks / --seq-len).

    # NOTE: THE SCALE LEVERS -- this scaffold grows TOWARD (not to) SpikeGPT scale:
    #   * d_model / n_blocks: width + depth. 96x2 is the toy default; SpikeGPT-46M is ~512x12+.
    #   * seq_len: context + BPTT unroll length. BPTT memory/compute grow LINEARLY with it; for
    #     long sequences use TRUNCATED BPTT (detach the recurrent state every k steps) or
    #     gradient checkpointing -- neither is implemented here yet, both are the standard
    #     next steps before raising seq_len past a few hundred.
    #   * a REAL corpus: `snn-train --lm --corpus file.txt` (see `load_corpus`) instead of the
    #     built-in toy paragraph.
    #   * models bigger than VRAM: the training tier's `--offload` path (training/offload.py)
    #     shows the gradient-checkpoint + CPU-optimizer-state pattern to borrow.
    #   Honest ceiling: a char tokenizer + single-file corpus + full BPTT trains SMALL models;
    #   real SpikeGPT scale needs tokenized corpora, truncated BPTT, and multi-GPU -- outside
    #   this scaffold.
    """
    vocab_size: int
    d_model: int = 96      # NOTE: 64-128 is the intended small range; 96 is a middle pick.
    n_blocks: int = 2      # NOTE: 2 spiking blocks -- depth is a tuning knob.
    seq_len: int = 64      # NOTE: unroll/context length; BPTT cost grows with this.
    beta: float = 0.9      # LIF membrane decay (snnTorch `Leaky` beta). VERIFY: not tuned here.


def analytic_param_count(cfg):
    """Parameter count computed from the config WITHOUT torch, so `main()` can print a size
    even on this torch-less box. VERIFY: this formula must stay in sync with the layers built
    in `SpikingBlock` / `SpikingLM` below -- update both together if the architecture changes.
      embedding : V*d
      per block : key(d*d) + value(d*d) + ffn(d*d + d bias) + log_decay(d)  = 3*d*d + 2*d
      readout   : d*V + V
    (LIF `Leaky` layers add no learned params with default learn_beta=False.)"""
    d, v, n = cfg.d_model, cfg.vocab_size, cfg.n_blocks
    embed = v * d
    per_block = 3 * d * d + 2 * d
    readout = d * v + v
    return embed + n * per_block + readout


# ----------------------------------- spiking block -----------------------------------
class SpikingBlock(_Module):
    """One SpikeGPT-lineage spiking block: a SPIKING token-mixer + a spike-driven feed-forward.

    Token mixing (the "attention" slot) is a CAUSAL, per-channel EXPONENTIALLY-WEIGHTED running
    summary of PAST input spikes:

        state_t = a * state_{t-1} + Key(spike_in_t)          # a = sigmoid(log_decay) in (0,1)
        mix_t   = Value(state_t)

    # NOTE: This APPROXIMATES linear attention / the RWKV-style recurrence SpikeGPT uses -- it is
    #   a decaying running average of past tokens, NOT softmax dot-product (QK^T) attention. It was
    #   chosen as the SIMPLEST DEFENSIBLE spiking mixer: causal by construction (only sums the past),
    #   O(T) memory, no pairwise attention matrix, and it is genuinely SPIKE-DRIVEN (its input is the
    #   binary `spike_in`). A real SpikeGPT uses a more elaborate spiking RWKV time-mixing block with
    #   token-shift and per-channel time-decay/first tokens -- VERIFY: this is the reduced form.
    """

    def __init__(self, d_model, beta):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()          # surrogate gradient (called at build time, torch path only)
        self.lif_in = snn.Leaky(beta=beta, spike_grad=spike_grad)     # current -> input spikes (binary)
        self.key = nn.Linear(d_model, d_model, bias=False)           # spike -> mixer key
        self.value = nn.Linear(d_model, d_model, bias=False)         # state -> mixed current
        # per-channel decay of the running state; sigmoid keeps it in (0,1). VERIFY: init at 0 -> a=0.5.
        self.log_decay = nn.Parameter(torch.zeros(d_model))
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)   # mixed current -> block spikes
        self.ffn = nn.Linear(d_model, d_model)                       # spike-driven feed-forward projection

    def forward(self, x):
        """x: [batch, seq_len, d_model] input CURRENT stream. Returns the same-shaped output
        current stream (a residual stream between blocks)."""
        a = torch.sigmoid(self.log_decay)              # [d] per-channel decay in (0,1)
        state = torch.zeros(x.size(0), x.size(2), device=x.device)   # running mixer state [batch, d]
        mem_in = self.lif_in.init_leaky()
        mem_out = self.lif_out.init_leaky()
        out = []
        for t in range(x.size(1)):                     # UNROLL over the sequence in time (BPTT)
            cur = x[:, t]                              # [batch, d] input current at position t
            spk_in, mem_in = self.lif_in(cur, mem_in)  # binary input spikes
            state = a * state + self.key(spk_in)       # causal exp-weighted summary of PAST spikes (linear-attn approx)
            mix = self.value(state)                    # read the mixed state out as a current
            spk_out, mem_out = self.lif_out(mix, mem_out)          # block output spikes (binary)
            # NOTE: graded residual current -- like SpikeGPT's non-spiking residual stream, this pathway
            #   carries graded (non-binary) values so gradients + identity flow through depth. VERIFY:
            #   the LIF layers are the spiking nonlinearities; the residual stream itself is graded.
            out.append(self.ffn(spk_out) + cur)
        return torch.stack(out, dim=1)                 # [batch, seq_len, d]


# ---------------------------------- the spiking LM -----------------------------------
class SpikingLM(_Module):
    """Char embedding -> stack of SpikingBlocks -> spike-driven vocab readout.

    Embedding acts as the per-token INPUT CURRENT injected into block 0 (dense embed -> current is a
    standard, defensible input-encoding for a spiking net). The readout is SPIKE-DRIVEN: the final
    residual stream is passed through one more LIF (`lif_head`) and the vocab logits are read from
    those head spikes, so predictions come from spikes -- not from a raw dense activation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)       # id -> input current [.,.,d]
        self.blocks = nn.ModuleList(
            [SpikingBlock(cfg.d_model, cfg.beta) for _ in range(cfg.n_blocks)]
        )
        self.lif_head = snn.Leaky(beta=cfg.beta, spike_grad=surrogate.fast_sigmoid())
        self.readout = nn.Linear(cfg.d_model, cfg.vocab_size)        # head spikes -> per-position logits

    def forward(self, idx):
        """idx: [batch, seq_len] long char ids -> logits [batch, seq_len, vocab]."""
        h = self.embed(idx)                            # [B,T,d] per-token input current
        for blk in self.blocks:
            h = blk(h)                                 # [B,T,d] residual current stream
        mem = self.lif_head.init_leaky()
        logits = []
        for t in range(h.size(1)):                     # spike-driven readout, unrolled in time
            spk, mem = self.lif_head(h[:, t], mem)     # head spikes (binary)
            logits.append(self.readout(spk))           # [B, vocab] logits at position t
        return torch.stack(logits, dim=1)              # [B,T,vocab]


# --------------------------------- train / generate ----------------------------------
def train(model, tok, corpus, steps=50, seq_len=64, lr=3e-3, device="cpu", seed=0, batch_size=16):
    """Surrogate-gradient BPTT on next-char prediction: CrossEntropy over the shifted corpus,
    Adam, optional AMP on CUDA (mirrors src/snn_data_lake/training/mnist.py). Returns the last
    loss. RUNS ONLY when torch is present; the numbers it produces are REAL (a real few-step
    train), never fabricated in this file."""
    torch.manual_seed(int(seed))
    use_amp = str(device).startswith("cuda")           # AMP/fp16 on CUDA, no-op on CPU
    amp_dev = "cuda" if use_amp else "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(amp_dev, enabled=use_amp)
    data = torch.tensor(tok.encode(corpus), dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    last = float("nan")
    hi = max(1, len(data) - seq_len - 1)
    for _ in range(int(steps)):
        ix = torch.randint(0, hi, (batch_size,), generator=gen, device=device).tolist()  # random windows
        xb = torch.stack([data[i:i + seq_len] for i in ix])          # [B,T] input chars
        yb = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix])  # [B,T] next-char targets
        opt.zero_grad()
        with torch.amp.autocast(amp_dev, enabled=use_amp):
            logits = model(xb)                                       # [B,T,V]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        scaler.scale(loss).backward()                               # BPTT through the spiking unroll
        scaler.step(opt)
        scaler.update()
        last = float(loss.detach().cpu())
    return last


def generate(model, tok, prompt, n=120, seq_len=64, device="cpu", temperature=1.0, seed=0):
    """Autoregressive char sampling from the spiking model: encode `prompt`, then repeatedly take
    the last-position logits, temperature-softmax, sample the next char, and append. Returns the
    prompt + `n` generated chars. RUNS ONLY when torch is present."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    ids = tok.encode(prompt) or [0]
    with torch.no_grad():
        for _ in range(int(n)):
            window = ids[-seq_len:]                                  # last seq_len chars = context
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(x)[0, -1]                                 # [V] logits at the last position
            probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))   # sample next char
            ids.append(nxt)
    return tok.decode(ids)


# ----------------------------- growth tooling (the levers) ----------------------------
def load_corpus(path=None):
    """Return the training corpus TEXT: a real UTF-8 file at `path`, or the built-in toy
    string when `path` is None. Pure stdlib (no torch), so the CLI can build the tokenizer
    before touching any torch code.

    # NOTE: growth lever #1 -- point `snn-train --lm --corpus` at real text (a book, a code
    #   dump, a wiki slice) instead of the 6-sentence toy. The char tokenizer adapts to
    #   whatever characters the file contains.
    """
    if path is None:
        return TOY_CORPUS
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if len(text) < 2:
        raise ValueError(f"corpus file {path!r} has fewer than 2 characters -- nothing to model")
    return text


def perplexity(model, tok, corpus, seq_len=64, device="cpu"):
    """exp(mean next-char cross-entropy in nats) of `model` over `corpus` -- THE growth metric.

    Slides non-overlapping length-`seq_len` windows over the encoded corpus, sums the
    CrossEntropy of every next-char prediction, and returns exp(total_nats / n_tokens).
    Lower is better; a uniform-random model over V chars scores ppl == V. Windows are
    independent (LIF/mixer state resets per window, same as the training windows). RUNS
    ONLY when torch is present; the number is REAL (measured), never fabricated here."""
    model.eval()
    data = torch.tensor(tok.encode(corpus), dtype=torch.long, device=device)
    if data.numel() < 2:
        raise ValueError("corpus too short for perplexity (need >= 2 chars)")
    total_nats, n_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, int(data.numel()) - 1, int(seq_len)):
            end = min(i + int(seq_len), int(data.numel()) - 1)
            xb = data[i:end].unsqueeze(0)                # [1, w] input chars
            yb = data[i + 1:end + 1]                     # [w] next-char targets
            logits = model(xb)[0]                        # [w, V] per-position logits
            total_nats += float(F.cross_entropy(logits, yb, reduction="sum").detach().cpu())
            n_tokens += int(yb.numel())
    return math.exp(total_nats / max(1, n_tokens))


# --------------------- checkpointing (torch.save / torch.load) ------------------------
def save_checkpoint(model, cfg, path):
    """Persist the torch model (state_dict + config) so long runs can RESUME instead of
    restarting -- growth lever #2 (mirrors training/mnist.py's checkpoint format)."""
    torch.save(
        {
            "format": "snn_training.SpikingLM/1",
            "state_dict": model.state_dict(),
            "cfg": {"vocab_size": cfg.vocab_size, "d_model": cfg.d_model,
                    "n_blocks": cfg.n_blocks, "seq_len": cfg.seq_len, "beta": cfg.beta},
        },
        path,
    )
    return path


def load_checkpoint(path):
    """Rebuild (model, cfg) from a checkpoint written by `save_checkpoint`. Resume with the
    SAME corpus: the char->id map comes from the corpus tokenizer, so the CLI cross-checks
    only the vocab SIZE -- a different corpus with a coincidentally equal vocab would silently
    remap characters. VERIFY: keep corpus + checkpoint paired."""
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("format") != "snn_training.SpikingLM/1":
        raise ValueError(f"{path}: not a snn_training.SpikingLM/1 checkpoint")
    cfg = SpikingLMConfig(**ckpt["cfg"])
    model = SpikingLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg


# ------- THE BRIDGE: export torch weights into the zero-dep spiking_lm_serve core ------
def export_lm(model, tok, path, cfg, meta=None):
    """Export the trained spiking LM as a zero-dep ``snn_lm/1`` JSON at `path`.

    ``snn_data_lake.cortex.spiking_lm_serve.load_lm`` loads it and ``generate`` runs the SAME
    spiking forward in pure Python. The serve's LIF is snnTorch's DEFAULT subtract-reset at
    threshold 1.0 -- exactly what this trainer's ``snn.Leaky(beta=...)`` layers use -- so the
    dynamics match and the learned weights transfer faithfully. ``nn.Linear.weight`` is
    [out][in], so every matrix below maps 1:1 into the serve layout with NO transpose:
      embed [V][D], per block key/value/ffn_w [D][D] + log_decay/ffn_b [D],
      readout [V][D], readout_b [V].
    All shapes are ASSERTED before writing. Returns the lm dict."""
    v, d, nb = cfg.vocab_size, cfg.d_model, cfg.n_blocks
    embed = model.embed.weight.detach().cpu().tolist()               # [V][D] id -> input current
    readout = model.readout.weight.detach().cpu().tolist()           # [V][D] head spikes -> logits
    readout_b = model.readout.bias.detach().cpu().tolist()           # [V]
    assert len(embed) == v and all(len(r) == d for r in embed), f"embed must be [{v}][{d}]"
    assert len(readout) == v and all(len(r) == d for r in readout), f"readout must be [{v}][{d}]"
    assert len(readout_b) == v, f"readout_b must be [{v}]"
    blocks = []
    for blk in model.blocks:
        b = {
            "key": blk.key.weight.detach().cpu().tolist(),           # [D][D] spikes -> mixer key
            "value": blk.value.weight.detach().cpu().tolist(),       # [D][D] state -> mixed current
            "log_decay": blk.log_decay.detach().cpu().tolist(),      # [D] per-channel decay logit
            "ffn_w": blk.ffn.weight.detach().cpu().tolist(),         # [D][D] spike-driven FFN
            "ffn_b": blk.ffn.bias.detach().cpu().tolist(),           # [D]
        }
        for name in ("key", "value", "ffn_w"):
            assert len(b[name]) == d and all(len(r) == d for r in b[name]), \
                f"block {len(blocks)} {name} must be [{d}][{d}]"
        for name in ("log_decay", "ffn_b"):
            assert len(b[name]) == d, f"block {len(blocks)} {name} must be [{d}]"
        blocks.append(b)
    assert len(blocks) == nb, f"expected {nb} blocks, exported {len(blocks)}"
    itos = [tok.itos[i] for i in range(v)]               # chars in id order (serve indexes by id)
    m = {"source": "snn_data_lake.training.spiking_lm surrogate-gradient trainer",
         "task": "char-lm"}
    if meta:
        m.update(meta)
    lm = {"format": "snn_lm/1", "vocab": v, "d_model": d, "n_blocks": nb,
          "beta": cfg.beta, "threshold": 1.0,            # snnTorch Leaky default: subtract-reset at 1.0
          "seq_len": cfg.seq_len, "stoi": dict(tok.stoi), "itos": itos,
          "embed": embed, "blocks": blocks, "readout": readout, "readout_b": readout_b,
          "meta": m}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lm, f)
    return lm


def verify_transfer(path, prompt="the ", n=60):
    """Show the export transfers into the core: load the JSON at `path` with the ZERO-DEP
    ``snn_data_lake.cortex.spiking_lm_serve`` and greedy-generate `n` chars after `prompt` -- the
    trained LM visibly running with NO torch. Returns the generated string (prompt + text).

    HONEST: greedy core-served text can differ slightly from the torch model's output
    (pure-Python float paths vs torch kernels; and the torch `generate` above SAMPLES while
    this is greedy). The check proves the weights LOAD and RUN in the core -- it is not a
    bitwise-equality claim."""
    from snn_data_lake.cortex.spiking_lm_serve import generate as core_generate
    from snn_data_lake.cortex.spiking_lm_serve import load_lm
    lm = load_lm(path)
    return core_generate(lm, prompt, n=n, temperature=0.0)


# ---------------------------------------- main ---------------------------------------
def main():
    """Print the spiking-LM SHAPE (torch-free), then -- only if torch is installed -- build the
    model, run a few REAL training steps, and sample. On this torch-less box it prints a clean
    'needs torch' note and exits 0 (no traceback, no fabricated numbers)."""
    tok = CharTokenizer(TOY_CORPUS)
    cfg = SpikingLMConfig(vocab_size=tok.vocab_size)
    params = analytic_param_count(cfg)

    print("=" * 68)
    print("TINY SPIKING LANGUAGE MODEL (SpikeGPT / SpikingBERT lineage) -- SCAFFOLD")
    print("=" * 68)
    print("HONEST: best-effort research scaffold, NOT a trained LLM, NOT AirLLM.")
    print("        AirLLM = big DENSE transformer streamed off disk; this = small")
    print("        transformer of SPIKING neurons trained from scratch on a char toy.")
    print("        Unverified: torch is absent here, so nothing below was run/measured.")
    print("-" * 68)
    print("architecture : char-embedding -> {n} spiking blocks -> spike-driven vocab readout"
          .format(n=cfg.n_blocks))
    print("  block      : LIF(in) -> causal exp-decay spike token-mixer (RWKV/linear-attn approx)")
    print("               -> LIF(out) -> spike-driven feed-forward + graded residual")
    print("  neurons    : snntorch.Leaky, surrogate.fast_sigmoid() gradient, unrolled BPTT")
    print("-" * 68)
    print("vocab size   : {v} chars".format(v=cfg.vocab_size))
    print("d_model      : {d}".format(d=cfg.d_model))
    print("blocks       : {n}".format(n=cfg.n_blocks))
    print("seq_len      : {s}".format(s=cfg.seq_len))
    print("params       : {p:,} (analytic; exact count printed below when torch is present)"
          .format(p=params))
    print("corpus       : {c} chars (embedded toy string, no downloads)".format(c=len(TOY_CORPUS)))
    print("-" * 68)
    print('training + export need torch: run them via `snn-train --lm` (pip install ".[gpu]").')

    if _TORCH_IMPORT_ERROR is not None:
        print('needs torch: pip install ".[gpu]"  (installs torch + snntorch), then run:')
        print("  snn-train --lm --export-lm my_lm.json         # train + zero-dep export via the CLI")
        print("  python -m snn_data_lake.training.spiking_lm   # this shape demo + a tiny real train")
        print("=" * 68)
        return 0

    # ---- torch IS present: build, run a few REAL steps, sample. Numbers here are measured. ----
    from snn_data_lake.substrate.device import (
        resolve_device,  # lazy; the ONE device selector (Policy 2, v4.78)
    )
    device = resolve_device()
    torch.manual_seed(0)
    model = SpikingLM(cfg)
    exact = sum(p.numel() for p in model.parameters())
    print("torch present -> exact params: {e:,} on device={d}".format(e=exact, d=device))
    demo_steps = 30
    loss = train(model, tok, TOY_CORPUS, steps=demo_steps, seq_len=cfg.seq_len, device=device)
    print("ran {s} REAL training steps; final loss: {l:.4f} (tiny demo, not a converged model)"
          .format(s=demo_steps, l=loss))
    sample = generate(model, tok, prompt="the ", n=120, seq_len=cfg.seq_len, device=device)
    print("sample (untrained-scale toy, expect gibberish):")
    print("  " + repr(sample))
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
