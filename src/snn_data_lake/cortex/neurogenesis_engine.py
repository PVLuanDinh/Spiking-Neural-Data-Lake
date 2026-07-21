"""
Neurogenesis-driven self-improvement engine for the chat daemon (torch tier).

The brain improves itself by GROWING neurons to PREDICT ITS OWN NEXT PERCEPTION -- a
self-supervised forward model over its perceptual substrate. "Growing == improving" is
made literal and measurable: an NG-SNN (hippocampus/ng_snn.py, contribution-grown LIF
neurogenesis) is refit as experience accumulates; each new "granule cell" is admitted only
if it functionally reduces the next-perception prediction residual, and self-improvement is
read off as next-context prediction accuracy rising above chance as the pool grows.

Mechanism:
  * K context MODES, each a distinct SAFE multimodal band (distinct visual/temperature/audio
    levels; distance fixed at a SAFE value so nothing is ever in a danger band). A fixed,
    seeded, DOUBLY-STOCHASTIC Markov matrix over the K modes makes the next mode
    predictable-but-nontrivial (a permuted preferred next-mode with prob `peak`, else spread)
    -> the stationary distribution is uniform (chance = 1/K) yet next-mode carries structure.
  * tick() advances the chain one step, encodes the CURRENT mode via encode_multimodal into a
    [0,1]^64 feature vector, and records the transition (features_t -> mode_{t+1}) into a
    rolling buffer.
  * Every `refit_every` ticks a FRESH NG-SNN is fit on the buffer (features_t -> next mode id).
    It GROWS neurons (contribution criterion) to predict the next perception. predict_accuracy
    is measured on a HELD-OUT recent slice never used in the fit. Learning-progress /
    self-improvement = that held-out accuracy rising above chance as the model grows and the
    transition structure is learned.

Safety: PERCEPTION / next-perception self-model ONLY. This engine never constructs a motor
command, never calls ArmSim / serial_arm / reflex, and has no actuator route -- it sits BESIDE
the live brain's gated-arm + E-STOP path and cannot touch it. `distance` is pinned to a SAFE
value (CONTACT * 3.0), well outside any reflex band, purely as a perceptual channel.

Runs in the torch tier (.venv-cu128); deterministic via random.Random(seed) (Markov chain) and
torch.Generator(seed) inside NG-SNN. Device = cuda if available else cpu.

  PYTHONUTF8=1 PYTHONPATH=src .venv-cu128/Scripts/python.exe -m snn_data_lake.cortex.neurogenesis_engine
"""

import random

import torch

from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.hippocampus.ng_snn import NGSNN

SAFE_DISTANCE = CONTACT * 3.0  # 0.12 m -- perception-only channel, never in a reflex/danger band


def _mode_features(mode_id, k_modes, rng=None, jitter=0.0):
    """SAFE multimodal BAND for a context mode -> [0,1]^64 feature vector. Distinct
    visual/temperature/audio levels per mode; a small seeded within-band jitter turns each mode
    into a CLOUD rather than a point, so separating the next-mode genuinely needs capacity (few
    neurons underfit, more neurons generalize) -- that is what makes GROWING == improving. Distance
    is pinned SAFE (perception only, no motor route). encode_multimodal clamps to [0,1]."""
    base = (mode_id + 0.5) / k_modes  # 0.1,0.3,0.5,0.7,0.9 for K=5

    def j():
        return rng.uniform(-jitter, jitter) if (rng is not None and jitter) else 0.0

    return encode_multimodal(
        visual=base + j(),
        temperature=(1.0 - base) + j(),  # inverted -> richer per-mode signature
        audio=((mode_id * 2) % k_modes + 0.5) / k_modes + j(),  # permuted band
        distance=SAFE_DISTANCE,
    )


def _markov_matrix(k_modes, peak, seed):
    """Seeded DOUBLY-STOCHASTIC KxK transition matrix: from mode i go to a permuted preferred
    next-mode with prob `peak`, else uniform over the rest. A permutation preferred => every
    column also sums to 1 => uniform stationary distribution => chance stays 1/K, while a high
    `peak` makes the next mode predictable well above chance."""
    rng = random.Random(seed)
    perm = list(range(k_modes))
    rng.shuffle(perm)
    rest = (1.0 - peak) / (k_modes - 1)
    matrix = []
    for i in range(k_modes):
        row = [rest] * k_modes
        row[perm[i]] = peak
        matrix.append(row)
    return matrix


class NeurogenesisSelfImprove:
    """Self-improvement by neurogenesis: grow NG-SNN neurons to predict the brain's own next
    perception. PERCEPTION-ONLY -- no arm, no motor command, no actuator route (see module
    docstring). Read state via .goal/.ticks/.grown_units/.predict_accuracy/.n_modes/.state()."""

    def __init__(
        self,
        k_modes=5,
        refit_every=20,
        buffer_cap=1000,
        peak=0.75,
        jitter=0.12,
        l_max=120,
        eps_tol=0.28,
        T=15,
        seed=7,
        shuffle_transitions=False,
        device=None,
    ):
        self.k_modes = int(k_modes)
        self.refit_every = int(refit_every)
        self.buffer_cap = int(buffer_cap)
        self.peak = float(peak)
        self.jitter = float(jitter)
        self.seed = int(seed)
        self.shuffle_transitions = bool(shuffle_transitions)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._l_max, self._eps_tol, self._T = int(l_max), float(eps_tol), int(T)
        self._chain = random.Random(seed)  # mode transitions + jitter RNG
        self._matrix = _markov_matrix(self.k_modes, self.peak, seed)  # transition structure
        self.current_mode = self._chain.randrange(self.k_modes)
        self._buf_x = []  # features_t
        self._buf_y = []  # mode_{t+1}
        self.ticks = 0
        self._n_units = 0
        self._acc = 1.0 / self.k_modes  # chance until the first fit
        self.trajectory = []  # (tick, n_units, acc) per refit

    # ---- context stream ----------------------------------------------------------------------
    def _next_mode(self, cur):
        """Draw mode_{t+1}. Structured: sample row `cur` of the Markov matrix. Control
        (shuffle_transitions): draw uniformly, INDEPENDENT of cur -> no learnable structure."""
        if self.shuffle_transitions:
            return self._chain.randrange(self.k_modes)
        r = self._chain.random()
        acc = 0.0
        for j, p in enumerate(self._matrix[cur]):
            acc += p
            if r < acc:
                return j
        return self.k_modes - 1

    def tick(self):
        """Advance the chain one step; record (features of the current mode -> next mode id);
        refit the NG-SNN when due. Returns the new current mode id."""
        feats = _mode_features(self.current_mode, self.k_modes, self._chain, self.jitter)  # feat_t
        nxt = self._next_mode(self.current_mode)  # mode_{t+1} (next perception)
        self._buf_x.append(feats)
        self._buf_y.append(nxt)
        if len(self._buf_x) > self.buffer_cap:
            self._buf_x.pop(0)
            self._buf_y.pop(0)
        self.current_mode = nxt
        self.ticks += 1
        if self.ticks % self.refit_every == 0:
            self._refit()
        return nxt

    # ---- neurogenesis / self-improvement -----------------------------------------------------
    def _refit(self):
        """Grow a fresh NG-SNN on the buffer (features_t -> next mode) and measure next-mode
        prediction accuracy on a HELD-OUT recent slice (never used in the fit)."""
        n = len(self._buf_x)
        holdout = min(60, n // 3)
        if holdout < 3 or (n - holdout) < self.k_modes:
            return
        x = torch.tensor(self._buf_x, dtype=torch.float32)
        y = torch.tensor(self._buf_y, dtype=torch.long)
        x_tr, y_tr = x[: n - holdout], y[: n - holdout]
        x_te, y_te = x[n - holdout :], y[n - holdout :]
        model = NGSNN(
            l_max=self._l_max, eps_tol=self._eps_tol, T=self._T, device=self.device, seed=self.seed
        )
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        acc = (pred == y_te.to(model.device)).float().mean().item()
        self._n_units = model.n_units
        self._acc = acc
        self.trajectory.append((self.ticks, model.n_units, acc))

    # ---- read-only state ---------------------------------------------------------------------
    @property
    def goal(self):
        return "self-improvement"

    @property
    def grown_units(self):
        return self._n_units

    @property
    def predict_accuracy(self):
        return self._acc

    @property
    def n_modes(self):
        return self.k_modes

    def state(self):
        return {
            "goal": self.goal,
            "ticks": self.ticks,
            "grown_units": self.grown_units,
            "predict_accuracy": self.predict_accuracy,
            "n_modes": self.n_modes,
        }


# -------------------------------------------------------------------------------------------------
# __main__ un-riggable self-check: each claim + a CONTROL that MUST fail.
# -------------------------------------------------------------------------------------------------
def _run(engine, n_ticks):
    for _ in range(n_ticks):
        engine.tick()
    return engine


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(
        k_modes=5,
        refit_every=20,
        buffer_cap=1000,
        peak=0.75,
        jitter=0.12,
        l_max=120,
        eps_tol=0.28,
        T=15,
    )
    chance = 1.0 / cfg["k_modes"]

    # claim 1: SELF-IMPROVEMENT -- held-out next-mode accuracy RISES from ~chance as it grows.
    # early/late are averaged over first-2 / last-3 refits (the held-out slice is small -> noisy).
    eng = _run(NeurogenesisSelfImprove(seed=7, **cfg), 600)
    traj = eng.trajectory
    accs = [ac for (_, _, ac) in traj]
    early_acc = sum(accs[:2]) / 2  # first refits (small buffer, few neurons)
    late_acc = sum(accs[-3:]) / 3  # last refits (600 ticks of experience)
    grown = eng.grown_units
    c1_pass = (
        (late_acc >= 0.5) and (early_acc <= 0.40) and (late_acc - early_acc >= 0.15) and (grown > 0)
    )

    # claim 2: CONTROL THAT MUST FAIL -- shuffled (structure-free) transitions stay at chance.
    ctrl = _run(NeurogenesisSelfImprove(seed=7, shuffle_transitions=True, **cfg), 600)
    ctrl_acc = ctrl.trajectory[-1][2]
    ctrl_grew = ctrl.grown_units > 0  # it DID grow, but cannot generalize
    c2_pass = (ctrl_acc < 0.35) and ctrl_grew and (late_acc - ctrl_acc >= 0.20)

    # claim 3: GENUINELY NEUROGENIC -- more distinct experience -> more (but bounded) neurons.
    units_100 = next(u for (t, u, _) in traj if t >= 100)
    units_600 = traj[-1][1]
    c3_pass = (units_600 > units_100) and (units_600 < cfg["l_max"])

    # claim 4: DETERMINISTIC -- two seed=7 engines give identical grown_units + accuracy trajos.
    a = _run(NeurogenesisSelfImprove(seed=7, **cfg), 600)
    b = _run(NeurogenesisSelfImprove(seed=7, **cfg), 600)
    units_a = [u for (_, u, _) in a.trajectory]
    units_b = [u for (_, u, _) in b.trajectory]
    accs_a = [ac for (_, _, ac) in a.trajectory]
    accs_b = [ac for (_, _, ac) in b.trajectory]
    accs_match = all(abs(x - y) < 1e-6 for x, y in zip(accs_a, accs_b, strict=True))
    c4_pass = (units_a == units_b) and accs_match

    # ---- ASCII report ---------------------------------------------------------------------------
    rows = [
        (
            1,
            "held-out next-mode acc RISES as it grows",
            f"early {early_acc:.2f} -> late {late_acc:.2f} ({grown}u)",
            c1_pass,
        ),
        (
            2,
            "shuffled control stays near chance",
            f"ctrl {ctrl_acc:.2f} vs structured {late_acc:.2f}",
            c2_pass,
        ),
        (
            3,
            "neurogenic: more experience -> more units",
            f"t100 {units_100}u -> t600 {units_600}u (<{cfg['l_max']})",
            c3_pass,
        ),
        (
            4,
            "deterministic trajectory (seed=7)",
            f"units {units_a == units_b} + accs match",
            c4_pass,
        ),
    ]
    line = "+" + "-" * 4 + "+" + "-" * 42 + "+" + "-" * 34 + "+" + "-" * 6 + "+"
    print("=" * 90)
    print(f"NEUROGENESIS SELF-IMPROVE | dev={dev} K={cfg['k_modes']} chance={chance:.2f}")
    print(line)
    print(f"| {'#':>2} | {'claim':<40} | {'measured':<32} | {'ok':<4} |")
    print(line)
    for i, claim, meas, ok in rows:
        print(f"| {i:>2} | {claim:<40} | {meas:<32} | {str(ok):<4} |")
    print(line)
    print("trajectory (tick, grown_units, predict_accuracy):")
    print("  " + "  ".join(f"[{t},{u}u,{ac:.2f}]" for (t, u, ac) in traj))
    print(
        "CONTROL (must FAIL): shuffled transitions -> no learnable structure -> stays at chance "
        "even after the same growth."
    )
    print("=" * 90)

    # ---- asserts --------------------------------------------------------------------------------
    assert c1_pass, f"claim1: early {early_acc:.2f} -> late {late_acc:.2f}, grown {grown}"
    assert late_acc >= 0.5, f"late held-out accuracy must clear 0.5 (got {late_acc:.2f})"
    assert grown > 0, "neurogenesis must have grown at least one neuron"
    assert c2_pass, f"claim2: shuffled acc {ctrl_acc:.2f} must stay near chance {chance:.2f}"
    assert ctrl_acc < 0.35, f"CONTROL: no-structure must NOT beat chance (got {ctrl_acc:.2f})"
    assert c3_pass, f"claim3: t100 {units_100}u -> t600 {units_600}u (l_max {cfg['l_max']})"
    assert units_600 > units_100, "more distinct experience must grow more neurons"
    assert units_600 < cfg["l_max"], "growth must stay bounded (not explode to l_max)"
    assert c4_pass, "claim4: two seed=7 engines must give identical grown_units + accuracy"

    print("self-check OK: the brain self-improves by GROWING NG-SNN neurons to predict its own")
    print("next perception; held-out accuracy rises above chance as it grows (shuffled control")
    print("stays at chance); a genuine neurogenesis-driven self-improvement loop, perception-only")
    print("+ safe. HONEST NOTE: over a FIXED Markov structure the accuracy plateaus once the")
    print("transitions are learned -- open-ended improvement needs a growing/richer world (see")
    print("docs/SELF_IMPROVEMENT_RESEARCH.md).")


if __name__ == "__main__":
    main()
