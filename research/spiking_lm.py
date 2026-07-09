"""TINY from-scratch SPIKING language-model scaffold (SpikeGPT / SpikingBERT lineage).

=======================  HONEST FRAMING -- READ THIS FIRST  =======================
This is a BEST-EFFORT, RESEARCH-SCALE SCAFFOLD of an ALL-SPIKING language model: a
char-level transformer whose neurons are LIF spiking units (snnTorch `Leaky` +
`surrogate.fast_sigmoid()` gradients), trained from scratch by surrogate-gradient
BPTT. It is in the SpikeGPT / SpikingBERT lineage (a transformer BUILT OUT OF
spiking neurons), NOT a large trained LLM.

It is NOT related to AirLLM. AirLLM runs a large DENSE (non-spiking) transformer by
streaming its layers off disk; this module is the OPPOSITE -- a small transformer
made of spiking neurons and trained from scratch on a toy corpus. They share nothing
but the phrase "language model".

torch and snntorch are NOT installed on this box and there is no GPU, so NONE of this
has been run or imported here -- it is architecture-correct-BY-BEST-EFFORT only. There
are NO measured numbers in this file: no loss, no accuracy, no sample text is claimed.
The `main()` below deliberately prints the model SHAPE (params/blocks/vocab) and a note
that training and generation require torch + a GPU and are left to the user; the code
that would actually train and print numbers ONLY runs when torch is present (on the
user's machine), where those numbers are REAL, not fabricated here.

What it demonstrates: the spiking-LM ARCHITECTURE on a char-level toy (embedding ->
spiking blocks -> spike-driven readout, unrolled in time, surrogate-grad trained). What
it does NOT do: a real spiking LLM needs large-scale training this scaffold does not do.
Treat every design decision flagged `# NOTE:` / `# VERIFY:` below as a research choice
that a real build would ablate and tune -- especially the spiking-attention approximation.

Run (needs torch + snntorch; from repo root so `torch_device` on research/ resolves):
    PYTHONPATH=src .venv-gpu/Scripts/python research/spiking_lm.py

With no torch installed it prints the shape + a clean "needs torch" note and exits 0.
Ruff-excluded, runs from research/, NOT in CI.
"""
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
    """Small-by-design config. seq_len 64, d_model 96, 2 blocks -- a research toy, not scale."""
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
    print("training + generation need torch + a GPU and are LEFT TO THE USER.")

    if _TORCH_IMPORT_ERROR is not None:
        print("needs torch: pip install '.[gpu]'  (installs torch + snntorch), then re-run:")
        print("  PYTHONPATH=src .venv-gpu/Scripts/python research/spiking_lm.py")
        print("=" * 68)
        return 0

    # ---- torch IS present: build, run a few REAL steps, sample. Numbers here are measured. ----
    from torch_device import enable, resolve_device  # noqa: PLC0415  (lazy; research device-select)
    device = enable(resolve_device())
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
