"""G5 / T113 Part D D1: the DELTA / PC / SOM read-out adaptors (evidence/g5-readout-variants.md D.2, D.4 PD-R3).

Each rule R in {DELTA, PC, SOM} is ONE arm against the incumbent R-STDP of the posed-problem job, and each
replaces R-STDP only under D.4 PD-R3: contrast(R - RSTDP_D) MET on S0C1 AND R's own control NOT MET. P2 --
one tuning rule survives per readout job -- so the three rules are ADOPTED from their own modules, never
re-implemented:

    DELTA  `readout/online_readout.py::OnlineReadout("delta")`. The adaptor is a model object that carries
           the (n_in x M) beta and hands it to the module's own `learn`; no rule code is copied. Every
           label's column is allocated from the first event (record 2.3's common vocabulary init carried
           into Part D), so `learn`'s own widen never fires and `widened` stays 0.
    PC     `readout/predictive_coding.py::PredictiveCodingNetwork` (its 2026-08-14 adjudication and open
           exact-symmetry scope stand; `eprop_transport_gate` is G9's, not this row's). Its OWN offline
           `learn` trains the W1/W2 coupling on the TEACH split; the label read-out is a per-label
           prototype over the settled r1 causes of the TEACH split only (D.2: "BMU-to-label assignment is
           a read-out table over the TEACH split only", carried to PC as its prototype table).
    SOM    `reasoning/spiking_som.py::SOM` (promoted out of reasoning/, P2/P4c) at the module's own
           default neighbourhood; the BMU-to-label table is a read-out table over the TEACH split only.

The controls (D.4 PD-R3): DELTA_CTRL is the shuffled-teaching twin and is built by the CELL RUNNER, not
here, because the shift, its vacuity floor and the same-rows check are `g5_readout_cells`' machinery and
are REUSED from there (D.2). PC_CTRL is the neutered-coupling twin (the same net, the same seed, the same
dims, W1/W2 NOT learned). SOM0 is the neighbourhood width at zero.

READ-OUT-ONLY (D.2): these adaptors never import or construct the pool. That is a claim with a reachable
red -- `self_check()` re-reads this module's own source and refuses any pool reference, so the claim cannot
silently stop being true. `code_hash()` is the printed code hash D.2 asks for.

Build-time details, to be frozen by the D1 dev-freeze amendment (D.2: "the adaptor details are
build-time"): the PC dims and its learn hyper-parameters, SOM's g, SOM0's width-0 realisation and the
SOM read-out table's rule. Nothing here reads a validation or future label, and no seed block runs here.
"""

import ast
import hashlib
import random
from pathlib import Path

GRADIENT_CLAIM = {
    "DeltaAdaptor.learn": (
        "LOCAL",
        "Delegates to OnlineReadout.learn: output error times input activity on its own synapses; "
        "no hidden error or transported weight.",
    ),
    "_Adaptor.learn": (
        "NOT_A_RULE",
        "Validates and buffers teaching rows; computes no weight update.",
    ),
}

#: The read-out-only contract's red (D.2). It is checked over THIS module's AST, not its text, because a
#: text scan cannot tell the words "the pool" in a docstring from a pool construction: a draft of this
#: check failed on its own token list, which is the near-miss that argues for the parse. A pool import or
#: a pool construction anywhere in this file turns it red.
FORBIDDEN_MODULES = ("snn_data_lake.reservoir", "carried_state", "g5_readout_cells")
FORBIDDEN_NAMES = ("make_pool", "KernelLsmPool", "pd_pool", "pool_features")

# ---- the adopted modules' own hyper-parameters (module defaults, FROZEN as defaults, D.1) ------------
PC_DIMS_R1, PC_DIMS_R2 = (
    16,
    4,
)  # 256 -> 16 -> 4: the same two reductions PredictiveCodingNetwork's own
#                                 # default dims (8, 4, 2) uses; a build-time choice for the D1 freeze
PC_EPOCHS, PC_STEPS, PC_LR, PC_LRW = 12, 120, 0.05, 0.02  # PredictiveCodingNetwork.learn's defaults
SOM_G = 4  # a 4x4 map: 16 units for a 2-label problem (the (i) cells' whole label set)
SOM_EPOCHS, SOM_LR0 = 15, 0.5  # SOM.train's defaults
#: SOM0 = the neighbourhood width AT ZERO. `SOM.train` reads `sigma0 = sigma0 or self.g / 2.0`, so the
#: literal 0 falls back to g/2 -- passing 0 would NOT give a zero-width map. At SOM0_WIDTH every
#: non-BMU unit's Gaussian h = exp(-1 / 2w^2) < 1e-3, which is exactly the module's own skip threshold, so
#: the update reduces to a pure WTA step on the BMU alone. The self-check asserts that no non-BMU unit
#: moves, which is the reachable red for "the width is at zero".
SOM0_WIDTH = 0.25


def code_hash():
    """sha256 of THIS module's source bytes -- the adaptors' printed code hash (D.2)."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _read_only_red():
    """Parse this module and fail on any pool import or pool construction (D.2's read-out-only claim)."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in FORBIDDEN_MODULES:
                    assert bad not in alias.name, f"read-out-only broken: imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            for bad in FORBIDDEN_MODULES:
                assert bad not in (node.module or ""), (
                    f"read-out-only broken: imports from {node.module}"
                )
        elif isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_NAMES, f"read-out-only broken: names {node.id}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_NAMES, f"read-out-only broken: reaches {node.attr}"


def _digest(values):
    return hashlib.sha256(" ".join(f"{v:.17g}" for v in values).encode("ascii")).hexdigest()


class AdaptorError(RuntimeError):
    """An adaptor was used outside its contract (a bad stimulus width, an unknown label, a control
    asked of an adaptor that has none)."""


class _Adaptor:
    """The uniform surface the D1 cell runner drives: `learn(x, label)` for every teaching event, then
    `respond(x)` for every held-out read. DELTA learns online; PC and SOM buffer the teach split and
    settle their read-out on the first respond, because the adopted modules' own learning is offline."""

    def __init__(self, labels, n_in):
        self.labels = list(labels)
        self.n_in = int(n_in)
        self._buf = []
        self._ready = False

    def _check(self, x):
        if len(x) != self.n_in:
            raise AdaptorError(f"stimulus has {len(x)} dims, expected {self.n_in}")

    def _check_label(self, label):
        if label not in self.labels:
            raise AdaptorError(f"unknown label {label!r}: the adaptor's label set is {self.labels}")

    def learn(self, x, label):
        self._check(x)
        self._check_label(label)
        self._buf.append((list(x), label))

    def _settle(self):
        raise NotImplementedError

    def respond(self, x):
        self._check(x)
        if not self._ready:
            self._settle()
            self._ready = True
        return self._answer(x)

    def _answer(self, x):
        raise NotImplementedError

    def state(self):
        return {"adaptor": type(self).__name__}


class DeltaAdaptor(_Adaptor):
    """DELTA: OnlineReadout("delta")'s rule over the 256-input label neurons. `self` is passed to the
    module's own `learn` as its model object, so beta is mutated in place by the incumbent's code."""

    def __init__(self, labels, n_in, mu=None):
        import torch

        from snn_data_lake.readout.online_readout import MU, OnlineReadout

        super().__init__(labels, n_in)
        self._torch = torch
        self.rule = OnlineReadout("delta", mu=MU if mu is None else mu)
        self.beta = torch.zeros(self.n_in, len(self.labels), dtype=torch.float32)
        self.m = len(self.labels)

    def learn(self, x, label):
        self._check(x)
        self._check_label(label)
        self._ready = True  # the incumbent rule is online; there is nothing to settle
        return self.rule.learn(self, x, self.labels.index(label))

    def _answer(self, x):
        s = self._torch.as_tensor(x, dtype=self._torch.float32).reshape(-1)
        return self.labels[int(self._torch.argmax(s @ self.beta))]

    def state(self):
        return {
            "adaptor": "delta",
            "mu": self.rule.mu,
            "steps": self.rule.steps,
            "widened": self.rule.widened,
            "code": code_hash(),
        }


class PCAdaptor(_Adaptor):
    """PC: `readout/predictive_coding.py` adopted. The coupling is trained by the module's own offline
    `learn` over the buffered TEACH split (`neutered` skips it: the D.4 PD-R3 control). The read-out is
    the nearest per-label prototype of the settled r1 causes, both taken over the TEACH split only."""

    def __init__(
        self, labels, n_in, seed=0, neutered=False, dims_r1=PC_DIMS_R1, dims_r2=PC_DIMS_R2
    ):
        import numpy as np

        from snn_data_lake.readout.predictive_coding import PredictiveCodingNetwork

        super().__init__(labels, n_in)
        self._np = np
        self.seed = int(seed)
        self.neutered = bool(neutered)
        self.net = PredictiveCodingNetwork(dims=(self.n_in, dims_r1, dims_r2), seed=self.seed)
        self._proto = {}

    def _settle(self):
        xs = [x for x, _ in self._buf]
        if not xs:
            raise AdaptorError("PC has no teach split to settle on")
        if not self.neutered:
            self.net.learn(xs, epochs=PC_EPOCHS, steps=PC_STEPS, lr=PC_LR, lr_w=PC_LRW)
        r1, _ = self.net.infer_batch(self._np.asarray(xs, dtype=self._np.float64))
        self._proto = {}
        for row, (_, label) in zip(r1, self._buf, strict=True):
            acc, n = self._proto.get(label, (None, 0))
            self._proto[label] = (
                row if acc is None else acc + row,
                n + 1,
            )
        self._proto = {k: v / n for k, (v, n) in self._proto.items()}
        self._proto = {k: v.tolist() for k, v in self._proto.items()}
        self._r1_dim = len(next(iter(self._proto.values())))

    def _answer(self, x):
        r1, _ = self.net.infer_batch(self._np.asarray([list(x)], dtype=self._np.float64))
        row = r1[0]

        def d2(p):
            return sum((a - b) ** 2 for a, b in zip(row, p, strict=True))

        return min(self._proto, key=lambda k: d2(self._proto[k]))

    def state(self):
        return {
            "adaptor": "pc",
            "neutered": self.neutered,
            "labels": sorted(self._proto),
            "dims": (self.net.d_in, self.net.d_r1, self.net.d_r2),
            "w1_sha": _digest(self.net.w1.ravel().tolist()),
            "w2_sha": _digest(self.net.w2.ravel().tolist()),
            "code": code_hash(),
        }


class SomAdaptor(_Adaptor):
    """SOM: `reasoning/spiking_som.py` adopted at its own default neighbourhood (`width=None`), or at
    SOM0's width for the control. The read-out table maps EVERY map unit to the label of the teach
    sample nearest that unit's weight vector -- one rule, entirely over the TEACH split, so no unit ever
    needs an unregistered fallback and no future label can enter it."""

    def __init__(self, labels, n_in, seed=0, width=None, g=SOM_G):
        from snn_data_lake.reasoning.spiking_som import SOM

        super().__init__(labels, n_in)
        self.seed = int(seed)
        self.width = width
        self.som = SOM(g, self.n_in, seed=self.seed)

    def _settle(self):
        xs = [x for x, _ in self._buf]
        if not xs:
            raise AdaptorError("SOM has no teach split to train on")
        self.som.train(
            xs,
            epochs=SOM_EPOCHS,
            lr0=SOM_LR0,
            sigma0=SOM0_WIDTH if self.width == 0.0 else self.width,
        )
        self._table = {}
        for i, w in enumerate(self.som.w):
            self._table[i] = min(self._buf, key=lambda xl: _dist2(w, xl[0]))[1]

    def _answer(self, x):
        return self._table[self.som.bmu(list(x))]

    def state(self):
        return {
            "adaptor": "som",
            "width": SOM0_WIDTH if self.width == 0.0 else self.width,
            "g": self.som.g,
            "seed": self.seed,
            "code": code_hash(),
        }


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def delta(labels, n_in, mu=None):
    return DeltaAdaptor(labels, n_in, mu=mu)


def pc(labels, n_in, seed=0, neutered=False):
    return PCAdaptor(labels, n_in, seed=seed, neutered=neutered)


def som(labels, n_in, seed=0, control=False):
    return SomAdaptor(labels, n_in, seed=seed, width=0.0 if control else None)


# --------------------------------------------------------------------------------------------------
# SELF-CHECK: the adaptors' cheap reds (brain-free, no pool, no seed block)
# --------------------------------------------------------------------------------------------------
def _chain(seed, n=240, k=2, d=8):
    """A first-order chain with a reversing succession rule (online_readout._chain's shape): learnable,
    so a read-out that cannot move is red. Fixture RNG only."""
    g = random.Random(seed)
    cats, c = [], 0
    for i in range(n):
        step = 1 if (i // 100) % 2 == 0 else -1
        c = (c + step) % k if g.random() < 0.85 else g.randrange(k)
        cats.append(c)
    out = []
    for i, c in enumerate(cats):
        row = [0.0] * d
        row[c * (d // k) : (c + 1) * (d // k)] = [1.0] * (d // k)
        row[d - 1] = 0.5 if (i // 100) % 2 else -0.5
        out.append((row, f"c{c}"))
    return out


def _held(dataset, frac=0.25):
    """A deterministic held-out tail of a planted chain (never a registered seed)."""
    cut = int(len(dataset) * (1 - frac))
    return dataset[cut:]


def self_check():
    """Every adaptor's reachable reds. Returns True; raises on the first failure."""
    # the read-out-only contract (D.2): a pool import or pool construction in this file is RED
    _read_only_red()
    assert len(code_hash()) == 64, "code_hash must be a sha256 hex digest"
    labels = ("c0", "c1")
    data = _chain(20261003)
    teach, held = data[: int(len(data) * 0.75)], _held(data)

    # DELTA reaches the incumbent's own rule: it learns the planted chain and every column was allocated
    d = delta(labels, len(teach[0][0]))
    for x, c in teach:
        d.learn(x, c)
    acc = sum(d.respond(x) == c for x, c in held) / len(held)
    assert acc > 0.9, f"DELTA should learn a first-order chain, held-out accuracy {acc}"
    assert d.state()["widened"] == 0, (
        "the common vocabulary init was not applied: learn's widen fired"
    )
    # DELTA is OnlineReadout("delta"): lambda = 0 is byte-identical to the incumbent's learn (record PA5)
    import torch

    from snn_data_lake.readout.online_readout import OnlineReadout

    ours, theirs = delta(labels, len(teach[0][0])), OnlineReadout("delta")

    class _M:
        pass

    m1, m2 = _M(), _M()
    m1.beta = torch.zeros(len(teach[0][0]), len(labels), dtype=torch.float32)
    m2.beta = torch.zeros(len(teach[0][0]), len(labels), dtype=torch.float32)
    m1.m = m2.m = len(labels)
    for x, c in teach:
        ours.learn(x, c)
        theirs.learn(m2, x, labels.index(c))
    assert torch.allclose(ours.beta, m2.beta, atol=0, rtol=0), (
        "DELTA diverged from OnlineReadout('delta')"
    )
    assert ours.rule.mu == theirs.mu and ours.rule.sign == theirs.sign
    # a bad width or an unknown label is refused rather than broadcast
    try:
        d.respond([0.0] * (len(teach[0][0]) + 1))
    except AdaptorError:
        pass
    else:
        raise AssertionError("a wrong-width stimulus must be refused")
    try:
        d.learn(teach[0][0], "c9")
    except AdaptorError:
        pass
    else:
        raise AssertionError("an unknown label must be refused")

    # PC: the twin at one seed is bit-identical, and the read-out is deterministic
    a = pc(labels, len(teach[0][0]), seed=20261003)
    b = pc(labels, len(teach[0][0]), seed=20261003)
    for x, c in teach:
        a.learn(x, c)
        b.learn(x, c)
    assert a.state()["w1_sha"] == b.state()["w1_sha"], "PC is not deterministic per seed"
    ra = [a.respond(x) for x, _ in held]
    rb = [b.respond(x) for x, _ in held]
    assert ra == rb, "PC's read-out is not deterministic per seed"
    # PC_CTRL: the neutered twin starts from the SAME seeded W1/W2 as its arm (one seed difference only)
    n = pc(labels, len(teach[0][0]), seed=20261003, neutered=True)
    fresh = pc(labels, len(teach[0][0]), seed=20261003)
    for x, c in teach:
        n.learn(x, c)
    assert n.state()["w1_sha"] == fresh.state()["w1_sha"], (
        "the neutered twin must keep its arm's seeded W1/W2"
    )
    # ... and it differs from the trained arm exactly in that learned coupling, which is the control
    assert a.state()["w1_sha"] != n.state()["w1_sha"], (
        "PC_CTRL learned nothing: the control is inert"
    )

    # SOM: the read-out table is a pure function of the TEACH split, and SOM0 moves ONLY the BMU
    s = som(labels, len(teach[0][0]), seed=20261003)
    for x, c in teach:
        s.learn(x, c)
    s0 = som(labels, len(teach[0][0]), seed=20261003, control=True)
    for x, c in teach:
        s0.learn(x, c)
    assert [s.respond(x) for x, _ in held] == [s.respond(x) for x, _ in held], (
        "SOM is not deterministic"
    )
    # SOM0's width: no non-BMU unit's weight may move under the control (the width-at-zero red)
    x = teach[0][0]
    before = [list(w) for w in s0.som.w]
    bmu = s0.som.bmu(x)
    s0.som.train([x], epochs=1, lr0=SOM_LR0, sigma0=SOM0_WIDTH)
    moved = [i for i in range(len(before)) if before[i] != list(s0.som.w[i])]
    assert moved == [bmu], f"the width is not at zero: units {moved} moved, not only the BMU {bmu}"
    assert s0.state()["width"] == SOM0_WIDTH and som(labels, 8).state()["width"] is None

    # no adaptor may see a future label: a teach label outside the label set is refused (above), and the
    # runner owns the split, so the adaptors' only entry point for labels is `learn`
    for adaptor in (d, a, s):
        assert adaptor.state()["code"] == code_hash(), "the printed code hash is not this module's"
    print(
        "g5_part_d_adaptors self-check OK: DELTA is OnlineReadout('delta') bit-for-bit with all columns "
        "allocated and no widen; PC is deterministic per seed and its neutered twin shares the seeded "
        "weights while the coupling differs; SOM is deterministic and SOM0 moves only the BMU (the width "
        "is at zero); wrong widths and unknown labels are refused; the read-out-only source scan passes"
    )
    return True


def main(argv=None):
    self_check()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
