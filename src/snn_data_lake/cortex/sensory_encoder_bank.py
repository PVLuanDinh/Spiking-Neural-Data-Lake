"""
SensoryEncoderBank -- a mixture-of-encoders sensory front-end behind ONE contract (pure stdlib).

The live perception/intent path (cortex.interaction_learning.SpikingInteractionLearner.encode) turns a
stimulus into a SPIKE TRAIN with exactly one operation: Thorpe rank-order latency coding
(rank_order_coding.rank_code -- 0 = fires first = largest value). That single fixed encoder is the whole
sensory front-end. This module MERGES the repo's several dead/standalone spike-encoding primitives into one
composable bank so the front-end can be swapped or stacked WITHOUT touching the live learner -- and its
DEFAULT is byte-identical to today, so wiring it in changes nothing until a non-default config is chosen.

MECHANISM: a small library of composable encoding STAGES (each a deterministic vector -> vector map that
reuses one existing primitive) selected by a config. A config names either a built-in preset or an ordered
pipeline of stage names; encode() runs the pipeline. The DEFAULT preset is a single rank-code stage, so
`SensoryEncoderBank().encode(x)` returns EXACTLY `rank_code(list(x))` -- OFF == today.

REUSED verbatim (import + thin adapter glue only; the primitives are NOT rewritten):
  - rank_order_coding.rank_code            -> stage "rank"          (Thorpe TTFS latency; the DEFAULT)
  - theta_gamma_code.ThetaGammaCode        -> stage "ttfs_phase"    (gamma-slot theta-PHASE / TTFS code)
  - divisive_normalization.DivisiveNormalization -> stage "divnorm" (Carandini-Heeger pool gain control)
  - lca_sparse_coding.LCA                   -> stage "lca"           (lateral-inhibition sparse code)
  - spike_pooling.max_pool_1d               -> stage "maxpool"       (anti-low-pass spike downsample)
  - spike_slicer.SpikeSlicer.density_change -> stage "density_edge"  (spike-density change / edge signal)
  - neural_sampling.BoltzmannSampler        -> stage "boltzmann"     (stochastic-spiking marginal code)
EXCLUDED on purpose: cortex.population_vector (an arm-adjacent population-vector DECODER, not an encoder --
this bank is intent/perception encoding only; it adds NO actuator route).

Additive / opt-in / deterministic: imports only stdlib + the reused primitives; each stage is a pure map
seeded via random.Random(seed) (no module-level seed, no builtin hash); no numpy/torch. A stage may change
width (maxpool downsamples, lca -> n_atoms); the width-preserving stages (rank, divnorm+rank, ttfs_phase,
boltzmann, density_edge) are drop-in replacements for the live n_in-wide rank code.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.sensory_encoder_bank
"""
import math
import random
import sys

from snn_data_lake.cortex.divisive_normalization import DivisiveNormalization
from snn_data_lake.cortex.lca_sparse_coding import LCA
from snn_data_lake.cortex.neural_sampling import BoltzmannSampler, marginals
from snn_data_lake.cortex.rank_order_coding import rank_code
from snn_data_lake.cortex.spike_pooling import max_pool_1d
from snn_data_lake.cortex.spike_slicer import SpikeSlicer
from snn_data_lake.cortex.theta_gamma_code import ThetaGammaCode

# ---- stages: each is a deterministic (vector, params) -> vector map wrapping one reused primitive -------

def _stage_rank(x, p):
    """DEFAULT stage. Thorpe rank-order latency: descending rank per input (0 = largest = fires first).

    Byte-identical to the live SpikingInteractionLearner.encode(): rank_code(list(x))."""
    return rank_code(list(x))


def _stage_divnorm(x, p):
    """Carandini-Heeger divisive pool gain control on nonnegative drives. Order-preserving (monotonic),
    so following it with "rank" leaves the rank code UNCHANGED for nonnegative input -- gain-invariant."""
    dn = DivisiveNormalization(sigma=p["sigma"], n=p["dn_exp"])
    return dn.normalize([v if v > 0.0 else 0.0 for v in x])


def _stage_ttfs_phase(x, p):
    """Time-to-first-spike / theta-phase code via ThetaGammaCode: order dims by value (rank), place them in
    gamma slots, and read each dim's theta PHASE (2*pi*rank/n). Higher value -> earlier phase (fires first)."""
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i], reverse=True)   # descending value == rank order (stable)
    tgc = ThetaGammaCode(capacity=max(n, 1), seed=p["seed"])     # capacity==n -> every dim keeps a slot
    tgc.encode_sequence(order)                                   # items are dim indices, in rank order
    return [tgc.slot_phase(tgc.slot_of(i)) for i in range(n)]    # per-dim theta phase == TTFS latency


def _unit(v):
    """Unit-normalise a vector (dictionary-atom glue; not part of the reused LCA algorithm)."""
    nrm = sum(c * c for c in v) ** 0.5 or 1.0
    return [c / nrm for c in v]


def _stage_lca(x, p):
    """LCA lateral-inhibition sparse code over a deterministic seeded dictionary of n_atoms atoms."""
    n = len(x)
    m = p["n_atoms"] if p["n_atoms"] else n
    rng = random.Random(p["seed"] * 7919 + 17)                  # per-instance seeded dictionary (deterministic)
    phi = [_unit([rng.uniform(-1.0, 1.0) for _ in range(n)]) for _ in range(m)]
    return LCA(phi, lam=p["lam"]).encode(list(x), steps=p["lca_steps"])


def _stage_maxpool(x, p):
    """Anti-low-pass spike max-pool downsample (keeps the peak of each window)."""
    return max_pool_1d(list(x), k=p["pool_k"], stride=p["pool_stride"])


def _stage_density_edge(x, p):
    """Spike-density CHANGE signal (SpikeSlicer): abs successive difference, first frame 0 -- an edge code."""
    return SpikeSlicer().density_change(list(x))


def _stage_boltzmann(x, p):
    """Stochastic-spiking marginal code: an independent (zero-weight) Boltzmann net whose biases are the
    scaled drives; return the per-unit spike marginal P(z_i=1) ~ sigmoid(gain*x_i). Deterministic per seed."""
    n = len(x)
    weights = [[0.0] * n for _ in range(n)]                     # independent units (valid symmetric, zero diag)
    biases = [p["bz_gain"] * float(v) for v in x]
    sampler = BoltzmannSampler(weights, biases, seed=p["seed"])
    return marginals(sampler.sample(p["bz_samples"], burn_in=p["bz_burn"], thin=n))


_STAGES = {
    "rank": _stage_rank,
    "divnorm": _stage_divnorm,
    "ttfs_phase": _stage_ttfs_phase,
    "lca": _stage_lca,
    "maxpool": _stage_maxpool,
    "density_edge": _stage_density_edge,
    "boltzmann": _stage_boltzmann,
}

# built-in presets: a named encoder == an ordered pipeline of stage names.
_PRESETS = {
    "rank": ["rank"],              # <-- DEFAULT: byte-identical to the live rank_code encode()
    "default": ["rank"],
    "divnorm_rank": ["divnorm", "rank"],
    "ttfs_phase": ["ttfs_phase"],
    "lca_sparse": ["lca"],
    "maxpool": ["maxpool"],
    "density_edge": ["density_edge"],
    "boltzmann": ["boltzmann"],
}

_DEFAULT_PARAMS = {
    "sigma": 0.5, "dn_exp": 2.0,          # divisive normalization
    "n_atoms": None, "lam": 0.08, "lca_steps": 80,   # LCA (n_atoms None -> input width)
    "pool_k": 2, "pool_stride": None,     # max-pool
    "bz_gain": 2.0, "bz_samples": 32, "bz_burn": 50,  # neural sampling
}


class SensoryEncoderBank:
    """Composable spike-encoding front-end behind one contract: encode(stimulus) -> spike_vector.

    config -- a preset name (see list_encoders()), a list of stage names (a custom pipeline), or None for the
    DEFAULT ("rank"), which is byte-identical to rank_order_coding.rank_code(list(stimulus)). Extra keyword
    params override the stage hyper-parameters. Deterministic per `seed`; pure stdlib; opt-in (nothing runs
    until you call encode). Reuses the repo's dead encoding primitives; adds NO actuator route."""

    DEFAULT = "rank"

    def __init__(self, config=None, seed=0, **params):
        self.seed = int(seed)
        self.params = dict(_DEFAULT_PARAMS)
        self.params.update(params)
        self.params["seed"] = self.seed        # seed is authoritative (drives every seeded stage)
        self.config = self.DEFAULT if config is None else config
        self.stage_names, self.name = self._resolve(self.config)
        self.pipeline = [_STAGES[s] for s in self.stage_names]

    @staticmethod
    def _resolve(config):
        """Return (ordered stage-name list, display name) for a preset name or an explicit pipeline list."""
        if isinstance(config, str):
            if config not in _PRESETS:
                raise ValueError(f"unknown encoder preset {config!r}; choose from {sorted(_PRESETS)}")
            return list(_PRESETS[config]), config
        names = list(config)
        for s in names:
            if s not in _STAGES:
                raise ValueError(f"unknown encoding stage {s!r}; choose from {sorted(_STAGES)}")
        return names, "+".join(names)

    def encode(self, stimulus):
        """Encode a stimulus (a vector of scalars) into a fixed-width spike vector by running the pipeline.

        With the DEFAULT config this returns EXACTLY rank_code(list(stimulus)) -- OFF == today's live path."""
        vec = list(stimulus)
        for stage in self.pipeline:
            vec = stage(vec, self.params)
        return vec

    def code_width(self, n_in):
        """Fixed output width this encoder produces for a stimulus of dimension n_in (deterministic)."""
        return len(self.encode([0.0] * int(n_in)))

    @classmethod
    def list_encoders(cls):
        """Names of the built-in presets (sorted)."""
        return sorted(_PRESETS)

    @classmethod
    def list_stages(cls):
        """Names of the composable stages (sorted)."""
        return sorted(_STAGES)


def _finite(v):
    """True iff every entry is a finite real number (a valid spike-code element)."""
    return all(isinstance(c, (int, float)) and math.isfinite(c) for c in v)


def main():
    x = [0.20, 0.90, 0.50, 0.10, 0.70, 0.30, 0.80, 0.40]      # synthetic 8-d stimulus (distinct values)
    n = len(x)

    # ---- 1. DEFAULT is byte-identical to the live rank_code encode() (OFF == today) ----
    base = rank_code(list(x))
    default_none = SensoryEncoderBank().encode(x)
    default_rank = SensoryEncoderBank("rank").encode(x)
    default_word = SensoryEncoderBank("default").encode(x)
    byte_identical = default_none == base and default_rank == base and default_word == base

    # ---- 2. every encoder produces a VALID FIXED-WIDTH code; widths are deterministic ----
    presets = SensoryEncoderBank.list_encoders()
    widths, valid = {}, True
    for name in presets:
        bank = SensoryEncoderBank(name)
        code = bank.encode(x)
        w = bank.code_width(n)
        widths[name] = len(code)
        valid = valid and isinstance(code, list) and len(code) == w and _finite(code)

    # ---- 3. DETERMINISM: two fresh banks (same config+seed) agree; repeat calls agree ----
    deterministic = True
    for name in presets:
        a = SensoryEncoderBank(name, seed=3).encode(x)
        b = SensoryEncoderBank(name, seed=3).encode(x)
        c = SensoryEncoderBank(name, seed=3)
        deterministic = deterministic and a == b and c.encode(x) == c.encode(x)

    # ---- 4. width-preserving encoders keep the live n_in width (drop-in for the rank code) ----
    keep_width = {"rank", "divnorm_rank", "ttfs_phase", "boltzmann", "density_edge"}
    width_preserved = all(widths[k] == n for k in keep_width)
    pooled_w = widths["maxpool"]                                   # (n-k)//stride + 1 = (8-2)//2 + 1 = 4

    # ---- 5. divisive normalization is order-preserving: divnorm_rank == rank for nonnegative input ----
    divnorm_rank = SensoryEncoderBank("divnorm_rank").encode(x)
    gain_invariant = divnorm_rank == base

    # ---- 6. TTFS/phase: highest-value dim fires first (phase 0); phases distinct and rank-ordered ----
    phase = SensoryEncoderBank("ttfs_phase").encode(x)
    argmax = max(range(n), key=lambda i: x[i])
    ttfs_first = phase[argmax] == min(phase)
    order = sorted(range(n), key=lambda i: x[i], reverse=True)
    ttfs_ordered = all(phase[order[i]] < phase[order[i + 1]] for i in range(n - 1))

    # ---- 7. COMPOSABILITY: an explicit stage pipeline == the equivalent preset ----
    piped = SensoryEncoderBank(["divnorm", "rank"]).encode(x)
    composable = piped == divnorm_rank

    # ---- 8. live-command-library wiring: the DEFAULT encodes a real 64-d command == rank_code (no drift) ----
    from snn_data_lake.brainstem.signal_loop import build_default_library
    library = build_default_library()
    cmd = library[list(library)[0]]
    lib_default = SensoryEncoderBank().encode(cmd)
    lib_ok = lib_default == rank_code(list(cmd)) and len(lib_default) == len(cmd)

    # ---- 9. pure stdlib: numpy/torch not imported anywhere on this path ----
    no_np_torch = ("numpy" not in sys.modules) and ("torch" not in sys.modules)

    print("=" * 74)
    print("SENSORY ENCODER BANK -- composable spike front-end (DEFAULT == live rank code)")
    print("=" * 74)
    print(f"stimulus (n={n})            : {x}")
    print(f"DEFAULT encode == rank_code: {default_none} (byte-identical={byte_identical})")
    print(f"presets                    : {presets}")
    print("code widths per encoder    :")
    for name in presets:
        print(f"    {name:<14s} -> width {widths[name]}")
    print(f"divnorm_rank == rank code  : {gain_invariant} (order-preserving gain control)")
    print(f"ttfs_phase (per-dim phase) : {[round(pv, 3) for pv in phase]}")
    print(f"    highest-value dim {argmax} fires first (phase {phase[argmax]:.3f}); rank-ordered={ttfs_ordered}")
    print(f"maxpool width              : {n} -> {pooled_w} (anti-low-pass downsample)")
    print(f"composable pipeline        : ['divnorm','rank'] == preset 'divnorm_rank' -> {composable}")
    print(f"live library wiring        : DEFAULT on a real 64-d command == rank_code -> {lib_ok}")
    print("=" * 74)

    # ---- self-check: deterministic asserts, ASCII-only, zero-dep ----
    assert byte_identical, "DEFAULT encoder must be byte-identical to rank_code(list(stimulus)) -- OFF == today"
    assert valid, "every encoder must produce a valid fixed-width finite code matching code_width(n)"
    assert deterministic, "same config+seed must give identical codes across fresh instances and repeat calls"
    assert width_preserved, f"width-preserving encoders must keep n={n} ({[(k, widths[k]) for k in keep_width]})"
    assert pooled_w == (n - 2) // 2 + 1, f"maxpool must downsample deterministically ({pooled_w})"
    assert gain_invariant, "divisive normalization is order-preserving: divnorm_rank must equal the rank code"
    assert ttfs_first and ttfs_ordered, "TTFS/phase: highest value must fire first and phases must be rank-ordered"
    assert len(set(phase)) == n, "distinct-valued dims must map to distinct theta phases"
    assert composable, "an explicit stage pipeline must equal the equivalent preset (composability)"
    assert lib_ok, "the DEFAULT must encode a real live command identically to rank_code (no drift)"
    assert no_np_torch, "must be pure stdlib: numpy/torch must not be imported"
    print(f"self-check OK: DEFAULT byte-identical to rank_code (base {base}); {len(presets)} encoders all "
          f"valid+fixed-width+deterministic; divnorm_rank==rank (gain-invariant); TTFS fires-first+ordered; "
          f"pipeline composable; live 64-d command wired identically; pure stdlib")


if __name__ == "__main__":
    main()
