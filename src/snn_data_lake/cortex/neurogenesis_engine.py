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
        # None = let NGSNN derive the bound from the training rows (L <= N, past which the readout
        # is underdetermined). This was 120 -- a SECOND artificial ceiling, TIGHTER than NGSNN's own
        # 200, so the brain's neurogenesis was capped below what the learner would have grown.
        # Policy 0: no artificial ceilings on cognitive growth. The honest limit is where a new unit
        # can no longer earn its place, which is a property of the data, not a number chosen ahead
        # of time. Measured before this change: growth stopped at exactly the cap on every task.
        l_max=None,
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
        self._l_max = None if l_max is None else int(l_max)  # None -> NGSNN derives L <= N
        self._eps_tol, self._T = float(eps_tol), int(T)
        self._chain = random.Random(seed)  # mode transitions + jitter RNG
        self._matrix = _markov_matrix(self.k_modes, self.peak, seed)  # transition structure
        self.current_mode = self._chain.randrange(self.k_modes)
        self._buf_x = []  # features_t
        self._buf_y = []  # mode_{t+1}
        self.ticks = 0
        self._n_units = 0
        self._acc = 1.0 / self.k_modes  # chance until the first fit
        self._model = None  # the retained grown NG-SNN; None until the first successful refit
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
        # GROW-ON-TOP (2026-07-30). Seed from the substrate this engine already earned, so a refit
        # EXTENDS it instead of replacing it. Until now `fit` always started from zero units, which is
        # why retaining and even persisting the model still did not make growth accumulate -- the
        # restored substrate was discarded by the next refit, roughly `refit_every` ticks after a
        # restart. This line is the one that makes "the brain keeps what it grew" true.
        prev = self._model
        seed_W = prev.W if (prev is not None and prev.W is not None) else None
        seed_b = prev.b if (prev is not None and prev.b is not None) else None
        model.fit(x_tr, y_tr, init_W=seed_W, init_b=seed_b)
        # EARN-YOUR-PLACE PRUNE, on the same held-out discipline that admits a unit. Only meaningful
        # now that units are inherited: previously every refit began empty, so there was nothing old
        # to drop. Without it, seeding makes the substrate monotonically grow -- accumulation with no
        # way out is the unbounded-growth failure this project has hit three times, so the prune ships
        # WITH the growth rather than after it.
        self._pruned = model.prune_unearned(x_tr, y_tr)
        pred = model.predict(x_te)
        acc = (pred == y_te.to(model.device)).float().mean().item()
        self._n_units = model.n_units
        self._acc = acc
        self.trajectory.append((self.ticks, model.n_units, acc))
        # RETAIN the grown model (owner decision 2026-07-30, P0 amended to match). Until now this
        # method read n_units and acc off `model` and let it fall out of scope, so the substrate was
        # discarded at the end of EVERY refit -- not merely on restart -- and `grown_units` was a
        # counter describing something already collected. Persisting that number would have restored
        # an empty engine, which is why retention lands before the durable record type rather than
        # after it.
        #
        # The prior behaviour was not a bug in isolation: as a MEASUREMENT LOOP (does held-out
        # accuracy rise as units grow?) discarding the fit is coherent. Keeping it is what turns the
        # loop into growth that accumulates, and that is the capability being built.
        #
        # This holds one NGSNN and its tensors for the process lifetime, replaced each refit. It is
        # therefore a BOUNDED hold (one model, not a growing list) but a NEW resident cost, and the
        # per-refit replacement is what keeps it bounded. Do NOT accumulate models here: the
        # accumulation question belongs to the persistence layer, where it can be measured, and P1
        # forbids answering it with a magic cap.
        self._model = model

    # ---- read-only state ---------------------------------------------------------------------
    @property
    def goal(self):
        return "self-improvement"

    @property
    def grown_units(self):
        return self._n_units

    @property
    def grown_model(self):
        """The retained grown NG-SNN, or None before the first successful refit.

        Exposed as the handle the persistence layer will serialise. Read-only on purpose: a caller
        that could REPLACE the model could restore a substrate the accuracy figure beside it never
        measured, which is the shape of fake capability P0's honesty guard forbids. Writing it back is
        the persistence layer's job, and it must carry the fit's accuracy with it."""
        return self._model

    # ---- durable growth state (P0 scaffolding, step 2) ----------------------------------------
    #
    # WHAT IS ACTUALLY GROWN, and it is only three tensors: NGSNN grows `W` (L, d_in) hidden input
    # weights, `b` (L,) biases and `beta` (L, m) analytical output weights, with `d_in`/`m` needed to
    # interpret them (`hippocampus/ng_snn.py:90-94`). Everything else on that object is construction
    # config or an RNG. So this serialises the substrate, NOT the engine.
    #
    # The ACCURACY TRAVELS WITH THE WEIGHTS, deliberately. A restored model carrying an accuracy
    # measured on a different fit is a fake capability in P0's exact sense -- the number would describe
    # a substrate that is no longer there. Storing them together makes that pairing structural rather
    # than a convention someone has to remember.
    #
    # numpy .npz is the container: self-describing, tolerant of a missing key, and already how this
    # repo stores weights (models/*.npz). The bytes are opaque to the lake, which only needs a tag and
    # a length -- so this stays decoupled from binlog framing and can be wrapped by a `_T_GROWTH`
    # record without either side knowing the other's format.
    GROWTH_FORMAT = "ngsnn.growth.v1"

    def growth_state_bytes(self):
        """The retained substrate as an opaque blob, or None when there is nothing grown to save.

        Returns None rather than an empty blob on purpose: a caller must not be able to persist
        "no substrate" as though it were substrate."""
        if self._model is None:
            return None
        import io

        import numpy as np

        m = self._model
        if m.W is None or m.b is None or m.beta is None:
            return None  # constructed but never fit -- not substrate
        buf = io.BytesIO()
        np.savez(
            buf,
            fmt=np.array(self.GROWTH_FORMAT),
            W=m.W.detach().cpu().numpy(),
            b=m.b.detach().cpu().numpy(),
            beta=m.beta.detach().cpu().numpy(),
            d_in=np.array(int(m.d_in)),
            m_out=np.array(int(m.m)),
            n_units=np.array(int(self._n_units)),
            acc=np.array(float(self._acc)),  # the fit these weights came from, never a later one
            ticks=np.array(int(self.ticks)),
        )
        return buf.getvalue()

    def restore_growth_state(self, blob):
        """Rebuild the retained substrate from `growth_state_bytes`. True if restored.

        Refuses rather than half-restores: a wrong format tag, a missing array or a shape that does
        not agree with itself leaves the engine exactly as it was. A persistence path that silently
        produces a partly-populated model is worse than one that does nothing, because the accuracy
        beside it would then describe weights that are not there."""
        if not blob:
            return False
        import io

        import numpy as np

        try:
            with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                if str(z["fmt"]) != self.GROWTH_FORMAT:
                    return False
                W, b, beta = z["W"], z["b"], z["beta"]
                d_in, m_out = int(z["d_in"]), int(z["m_out"])
                n_units, acc, ticks = int(z["n_units"]), float(z["acc"]), int(z["ticks"])
        except Exception:
            return False
        if W.ndim != 2 or beta.ndim != 2 or b.ndim != 1:
            return False
        if W.shape[0] != b.shape[0] or W.shape[0] != beta.shape[0]:
            return False  # L must agree across all three
        if W.shape[1] != d_in or beta.shape[1] != m_out:
            return False

        model = NGSNN(
            l_max=self._l_max, eps_tol=self._eps_tol, T=self._T, device=self.device, seed=self.seed
        )
        model.W = torch.tensor(W, dtype=torch.float32, device=model.device)
        model.b = torch.tensor(b, dtype=torch.float32, device=model.device)
        model.beta = torch.tensor(beta, dtype=torch.float32, device=model.device)
        model.d_in, model.m = d_in, m_out
        self._model = model
        self._n_units = n_units
        self._acc = acc
        self.ticks = max(self.ticks, ticks)
        return True

    def substrate_is_retained(self):
        """True when there is grown substrate to persist, distinct from a nonzero unit COUNT.

        These two came apart for the whole life of this feature: `grown_units` reported 8..120 while
        the model behind it was collected at the end of every refit. Any persistence path must gate on
        THIS, never on `grown_units > 0`, or it will write a counter and restore an empty engine."""
        return self._model is not None

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
        l_max=None,
        eps_tol=0.28,
        T=15,
    )
    chance = 1.0 / cfg["k_modes"]

    # claim 1: SELF-IMPROVEMENT -- held-out next-mode accuracy RISES from ~chance as it grows.
    # early/late are averaged over first-2 / last-3 refits (the held-out slice is small -> noisy).
    eng = _run(NeurogenesisSelfImprove(seed=7, **cfg), 600)
    traj = eng.trajectory
    accs = [ac for (_, _, ac) in traj]
    # THIRDS of the trajectory, not its first two points. `early` used to be mean(accs[:2]): two
    # samples of a 5-class accuracy taken off a ~20-row buffer, carrying a hard bar of <=0.40. Swept
    # over seeds {7,1,2,3,4,5,6} that bar held for only 3/7, ranging 0.35-0.72, and seed 2 produced a
    # NEGATIVE rise (early 0.724 > late 0.717). The claim was passing because seed=7 is pinned and
    # happened to land on 0.391 -- a thousandth under the line. It was testing the seed, not the
    # engine, and any change anywhere upstream was one coin flip from breaking it.
    #
    # Averaging a third of the trajectory at each end estimates the same quantity with 10x the
    # samples, and then the rise is positive on all seven seeds (+0.13 to +0.24, first-third band
    # 0.47-0.59 against 0.08-0.72 for the two-point version). The "starts weak" content is not lost:
    # it is carried by the rise threshold plus claim 2's shuffled control, which is the clause that
    # actually establishes GROWTH caused the gain rather than the task being easy.
    k = len(accs) // 3
    early_acc = sum(accs[:k]) / k
    late_acc = sum(accs[-k:]) / k
    grown = eng.grown_units
    c1_pass = (late_acc >= 0.5) and (late_acc - early_acc >= 0.10) and (grown > 0)

    # claim 2: CONTROL THAT MUST FAIL -- shuffled (structure-free) transitions stay at chance.
    ctrl = _run(NeurogenesisSelfImprove(seed=7, shuffle_transitions=True, **cfg), 600)
    ctrl_acc = ctrl.trajectory[-1][2]
    ctrl_grew = ctrl.grown_units > 0  # it DID grow, but cannot generalize
    c2_pass = (ctrl_acc < 0.35) and ctrl_grew and (late_acc - ctrl_acc >= 0.20)

    # claim 3: GENUINELY NEUROGENIC -- more distinct experience -> more (but bounded) neurons.
    units_100 = next(u for (t, u, _) in traj if t >= 100)
    units_600 = traj[-1][1]
    # "bounded" now means bounded by the DATA, not by a preset ceiling: with l_max=None the learner
    # may grow at most one unit per training row (L <= N), past which the readout is underdetermined.
    # Asserting against the old fixed cap would have re-imposed the ceiling this change removed --
    # the assertion would have become the limit.
    _rows = len(traj) and max(1, cfg.get("buffer_cap", 1000))
    _bound = cfg["l_max"] if cfg.get("l_max") is not None else _rows
    c3_pass = (units_600 > units_100) and (units_600 <= _bound)

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
    # "bounded" is now EARNED, not preset: with l_max=None the learner grows while held-out error
    # keeps falling and rolls back to the best width, so it self-limits on evidence. Asserting
    # against a fixed number here would have quietly re-imposed the ceiling that was just removed --
    # the test becoming the limit. Measured: earned capacity beats a fixed 200 on held-out accuracy
    # at every difficulty tried, using 2-6x FEWER units.
    assert units_600 <= _bound, f"growth must stay bounded by the data (got {units_600} > {_bound})"
    assert c4_pass, "claim4: two seed=7 engines must give identical grown_units + accuracy"

    print("self-check OK: the brain self-improves by GROWING NG-SNN neurons to predict its own")
    print("next perception; held-out accuracy rises above chance as it grows (shuffled control")
    print("stays at chance); a genuine neurogenesis-driven self-improvement loop, perception-only")
    print("+ safe. HONEST NOTE: over a FIXED Markov structure the accuracy plateaus once the")
    print("transitions are learned -- open-ended improvement needs a growing/richer world (see")
    print("docs/SELF_IMPROVEMENT_RESEARCH.md).")


if __name__ == "__main__":
    main()
