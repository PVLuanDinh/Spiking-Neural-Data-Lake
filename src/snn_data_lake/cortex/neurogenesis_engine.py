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
command, never calls So101Arm / serial_arm / reflex, and has no actuator route -- it sits BESIDE
the live brain's gated-arm + E-STOP path and cannot touch it. `distance` is pinned to a SAFE
value (CONTACT * 3.0), well outside any reflex band, purely as a perceptual channel.
(Named `ArmSim` here until 2026-08-04; that class does not exist anywhere in `src/` -- the live
body is `brainstem/so101_arm.So101Arm` and every motion passes its `apply()` gate. CLAUDE.md
carried the same dead name and corrected it the same day; a safety-scope sentence naming a file
that is not there sends a reviewer looking for the wrong gate.)

Runs in the torch tier; deterministic via random.Random(seed) (Markov chain) and torch.Generator(seed)
inside NG-SNN. DEVICE IS MEASURED, NOT ASSUMED (P5a, 2026-08-05): this line read "cuda if available
else cpu", which is exactly what `_refit` used to do -- it sent a ~60,000-element fit to a card whose
measured crossover on this machine is 262,144. The refit now hands its real element count to
`substrate/device.select`, and the env named here was `.venv-cu128`, retired the same day.

  PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.cortex.neurogenesis_engine
"""

import random

import torch

from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.cortex.genomic_bottleneck import (
    CandidatePool,
    GenomicMap,
    feature_tags,
    unit_tags,
)
from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.hippocampus.ng_snn import NGSNN
from snn_data_lake.substrate.device import crossover, decisions, select

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "NeurogenesisSelfImprove._refit": (
        "DIRECT_SOLVE",
        "grows a fresh NG-SNN on the buffer and measures next-mode prediction accuracy. It drives "
        "hippocampus/ng_snn::fit, whose greedy matching pursuit with an analytical beta per block "
        "POLICIES.md adjudicates as a DIRECT SOLVE -- non-local, but propagating no gradient and "
        "transporting no weight. It inherits that disposition rather than LOCAL, because LOCAL would "
        "assert a locality the adjudication explicitly denies",
    ),
}

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
        # T047: the growth stopping criterion (cortex/growth_stop.py) -- a ONE-ARG callable
        # receiving THIS engine's trajectory (the system under test reaches the criterion
        # through its parameter -- the red_reachability zero-arg red) and returning bool,
        # consulted at each refit. None = ungated (the incumbent fixed schedule, which is
        # exactly the criterion's CONTROL arm). The callable's verdict logic is the gate's
        # own; this engine can consult it, never constrain it.
        growth_gate=None,
        # FR-033: where a candidate unit's INITIAL fan-in comes from. "random" = the incumbent
        # pool (candidate placements still drawn before the fit, so bookkeeping is identical
        # across arms); "genome" = cortex/genomic_bottleneck's map, refitted every refit on the
        # earned substrate, read for every candidate whose tag is in the fitted span; "genome_
        # shuffled" = the same map fitted on tag-SHUFFLED units -- the control that keeps the
        # map's magnitude and span and destroys its tag->weight association. The production
        # default is decided by genomic_bottleneck.py's --ab entrypoint's paired measurement, never picked.
        init_pool="random",
    ):
        self.k_modes = int(k_modes)
        self.refit_every = int(refit_every)
        self.buffer_cap = int(buffer_cap)
        self.peak = float(peak)
        self.jitter = float(jitter)
        self.seed = int(seed)
        self.shuffle_transitions = bool(shuffle_transitions)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._growth_gate = growth_gate
        if init_pool not in ("random", "genome", "genome_shuffled"):
            raise ValueError(f"init_pool {init_pool!r} not in random / genome / genome_shuffled")
        self.init_pool = str(init_pool)
        self.pool_stats = {}  # the last refit's candidate-pool census (FR-033 attribution)
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
        #: [(depth, layer, sign)] index-aligned with `self._model.W` rows -- every grown unit's
        #: cortical home. Maintained across BOTH growth (new units appended) and PRUNE (survivors
        #: reindexed by the model's `last_keep`), because a placement that stops tracking its unit
        #: is worse than no placement: it looks like structure and describes a different neuron.
        self._placement = []
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
        # T047: consult the growth gate. CLOSED -> measurement continues, growth does not:
        # score the RETAINED substrate on the fresh held-out slice and record the refit
        # row, because the gate re-opens on its own evidence (a new measured optimum, |LP|
        # motion) and that evidence IS this stream -- a gate that stopped the measurement
        # would weld itself shut. An engine that has never grown ignores a closed gate:
        # the criterion has no evidence about a substrate that does not exist.
        if (
            self._growth_gate is not None
            and self._model is not None
            and not self._growth_gate(self.trajectory)
        ):
            pred = self._model.predict(x_te)
            acc = (pred == y_te.to(self._model.device)).float().mean().item()
            self._n_units = self._model.n_units
            self._acc = acc
            self.trajectory.append((self.ticks, self._model.n_units, acc))
            return
        # MEASURED ROUTING, not availability routing (P5a, 2026-08-05). `self.device` is
        # "cuda if a card exists" -- decided by `torch.cuda.is_available()` at construction, before
        # anything knows how big the work is. This refit's element count is not a guess: it is
        # exactly `x_tr.numel()` (rows in the buffer x the 64-wide multimodal feature), so it is
        # handed to `substrate.device.select`, which routes on the crossover THIS machine measured
        # rather than on the card being plugged in.
        #
        # HONEST ABOUT TODAY'S EFFECT: with buffer_cap=1000 and 64 features the widest refit this
        # engine can ever run is 64,000 elements, and the crossover measured here is 262,144 -- so
        # `select` answers 'cpu', every time, and the visible change is that a refit which used to
        # run on the card now runs on the CPU. That is the correct answer at this width and NOT a
        # speed claim; what is bought is that the answer is measured and flips on its own when the
        # buffer or the feature width grows past the crossover.
        #
        # An engine explicitly pinned to CPU stays pinned -- `explicit` is `select`'s own override
        # slot, so the constructor's `device="cpu"` still wins and there is no second override path.
        # NGSNN moves seeded weights with `init_W.to(self.device)`, so a refit landing on a
        # different device from the retained model is safe rather than merely unlikely.
        pin = self.device if self.device == "cpu" else None
        dev = select("neurogenesis_refit", n=int(x_tr.numel()), explicit=pin)
        model = NGSNN(
            l_max=self._l_max, eps_tol=self._eps_tol, T=self._T, device=dev, seed=self.seed
        )
        # GROW-ON-TOP (2026-07-30). Seed from the substrate this engine already earned, so a refit
        # EXTENDS it instead of replacing it. Until now `fit` always started from zero units, which is
        # why retaining and even persisting the model still did not make growth accumulate -- the
        # restored substrate was discarded by the next refit, roughly `refit_every` ticks after a
        # restart. This line is the one that makes "the brain keeps what it grew" true.
        prev = self._model
        seed_W = prev.W if (prev is not None and prev.W is not None) else None
        seed_b = prev.b if (prev is not None and prev.b is not None) else None
        pool, genome_info = self._candidates(int(x_tr.shape[1]))
        model.fit(x_tr, y_tr, init_W=seed_W, init_b=seed_b, candidates=pool)
        # EARN-YOUR-PLACE PRUNE, on the same held-out discipline that admits a unit. Only meaningful
        # now that units are inherited: previously every refit began empty, so there was nothing old
        # to drop. Without it, seeding makes the substrate monotonically grow -- accumulation with no
        # way out is the unbounded-growth failure this project has hit three times, so the prune ships
        # WITH the growth rather than after it.
        # PLACEMENT, part 1 of 2: the units this refit ADDED keep the cortical home that was drawn
        # for them BEFORE the fit (FR-033: the placement is the candidate's tag, and under the
        # genome pool it generated the unit's fan-in). `fit(init_W=...)` keeps the inherited units
        # as the LEADING rows, so index identity survives growth and the new units are exactly the
        # tail; `model.n_seeded` says how many rows were inherited (0 on a cold start or a width
        # mismatch, where the substrate was replaced and so is its placement). A provenance that
        # does not cover the tail exactly is refused: a placement that describes a different
        # neuron is worse than none.
        n_before = len(self._placement)
        n_after_fit = int(model.n_units)
        prov = list(model.last_provenance)
        if len(prov) != n_after_fit - int(model.n_seeded):
            raise RuntimeError(
                f"placement provenance misaligned: {len(prov)} entries for "
                f"{n_after_fit - int(model.n_seeded)} admitted units"
            )
        self._placement = self._placement[: int(model.n_seeded)] + [p for p, _src in prov]
        self.pool_stats = {
            "mode": self.init_pool,
            **pool.stats,
            "admitted": max(0, n_after_fit - n_before),
            "admitted_genome": sum(1 for _p, src in prov if src == "genome"),
            **genome_info,
        }

        self._pruned = model.prune_unearned(x_tr, y_tr)
        # PLACEMENT, part 2 of 2: FOLLOW THE SURVIVORS. `prune_unearned` drops columns, so without
        # this every placement after the first prune would describe a DIFFERENT neuron -- a Dale
        # sign assigned at birth would silently migrate, which is the integrity failure this whole
        # structure exists to prevent. The count alone (`self._pruned`) cannot do it; the survivor
        # mask can.
        keep = getattr(model, "last_keep", None)
        if keep is not None and len(self._placement) >= max(keep, default=-1) + 1:
            self._placement = [self._placement[i] for i in keep]
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
    # REUSE, not a second mechanism: both accessors return what `cortex/placement.place` produced
    # and this engine maintains. The assignment rule lives there and is imported, never reimplemented.
    #
    # distinct-from: brainstem/autonomous_loop.py and cortex/brain.py -- their "grown units" are the
    #   arm's PROTOTYPE recruits (context->action, a Q-table over them), a different growth mechanism
    #   entirely; these are NG-SNN hidden units. Worth naming: the phrase means two things in this
    #   tree. distinct-from: substrate/brain_geometry.py -- places this package's MODULES in 3D.
    #   distinct-from: substrate/home.py -- drops superseded lake records. All sites read.
    @property
    def placement(self):
        """Per-unit (depth, layer, sign), aligned with the substrate rows it describes.

        Empty until a refit grows something. `main()` asserts its length equals `grown_units`: a
        list that drifts out of step with the weights is the failure this tracking prevents, and it
        is invisible from either side alone.
        """
        return list(self._placement)

    def layer_census(self):
        """Counts per cortical layer over what has been grown."""
        out = {}
        for _d, layer, _s in self._placement:
            out[layer] = out.get(layer, 0) + 1
        return out

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
    # a length -- so this stays decoupled from binlog framing and can be carried by the lake's
    # growth records (since specs/022 T010-ii, home.save_growth decomposes the npz into a
    # `_T_GROWTH_DELTA` chain; the opaque-blob contract at THIS boundary is unchanged) without
    # either side knowing the other's format.
    GROWTH_FORMAT = "ngsnn.growth.v1"

    # distinct-from: hjepa_arm_torch.py -- coarse-transition windows, unrelated. Site read.
    # distinct-from: hjepa_torch.py -- same family, unrelated. Site read.
    # distinct-from: method_dormancy_gate.py -- self-check line filter. Site read.
    # distinct-from: replay_gate.py -- snapshot-resume proof; the matcher fired on the
    #   word trajectory over a two-line slice helper. Site read.
    @staticmethod
    def _traj_tail(trajectory):
        """The persisted-trajectory tail: the growth gate's own evidence window sizes it
        (the gate is the one consumer of restored rows; imported, not re-picked)."""
        from snn_data_lake.cortex.growth_stop import _WINDOW

        return trajectory[-_WINDOW:]

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
        # PLACEMENT rides with the substrate. Added as OPTIONAL arrays rather than by bumping
        # GROWTH_FORMAT: `restore_growth_state` refuses on a tag mismatch, so a bump would discard
        # every _T_GROWTH record already in the lake -- 73,490 of them, 508 MB, at the last census.
        # Old blobs therefore still restore (placement is DERIVED on the way in), and an older build
        # reading a new blob simply ignores arrays it does not name. Additive both directions.
        from snn_data_lake.cortex.placement import LAYER_NAMES

        placed = self._placement[: int(self._n_units)]
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
            # float64, not float32: depth is a python float, and storing it narrower makes the
            # round-trip lossy in a way only exact equality catches -- the probe that found this
            # reported 23 placements restored against 23 written and still unequal. Eight bytes a
            # unit is the right price for a restart that returns the SAME brain, not a rounded one.
            place_depth=np.array([p[0] for p in placed], dtype=np.float64),
            place_layer=np.array([LAYER_NAMES.index(p[1]) for p in placed], dtype=np.int16),
            place_sign=np.array([p[2] for p in placed], dtype=np.int8),
            # T047: the trailing refit trajectory rides with the substrate -- OPTIONAL
            # arrays, same additive contract as placement (no format bump). The panel's
            # P4b find: the substrate persisted while the evidence that GOVERNS its
            # growth did not, so every daemon relaunch (~20/session) reset the growth
            # gate to its 9-refit insufficient-history horizon -- restart amnesia for
            # the gate, on a brain whose restarts are routine. Tail length = the gate's
            # own evidence window (its need sizes it; imported, not re-picked).
            traj_ticks=np.array(
                [t for t, _u, _a in self._traj_tail(self.trajectory)], dtype=np.int64
            ),
            traj_units=np.array(
                [u for _t, u, _a in self._traj_tail(self.trajectory)], dtype=np.int64
            ),
            traj_acc=np.array(
                [a for _t, _u, a in self._traj_tail(self.trajectory)], dtype=np.float64
            ),
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
                # optional -- absent in every blob written before placement existed
                z_depth = z["place_depth"] if "place_depth" in z else None
                z_layer = z["place_layer"] if "place_layer" in z else None
                z_sign = z["place_sign"] if "place_sign" in z else None
                # optional -- absent in every blob written before T047's gate evidence
                z_tt = z["traj_ticks"] if "traj_ticks" in z else None
                z_tu = z["traj_units"] if "traj_units" in z else None
                z_ta = z["traj_acc"] if "traj_acc" in z else None
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

        # PLACEMENT comes back with the substrate, or is DERIVED when the blob predates it. Deriving
        # rather than leaving it empty is deliberate: a restored brain whose units have weights but
        # no laminar identity is half-restored, which this method's own docstring refuses to do. The
        # derived placement is deterministic in `seed`, so the same old blob always reconstitutes the
        # same structure -- it is a first assignment, not a guess that drifts per restart.
        from snn_data_lake.cortex.placement import LAYER_NAMES, place

        placed = None
        try:
            depth, layer_ix, sign = z_depth, z_layer, z_sign
            if depth is not None and len(depth) == n_units:
                placed = [
                    (float(depth[i]), LAYER_NAMES[int(layer_ix[i])], int(sign[i]))
                    for i in range(n_units)
                ]
        except Exception:
            placed = None  # a malformed placement is DERIVED afresh, never half-applied
        self._placement = placed if placed is not None else place(n_units, seed=self.seed)
        # T047: the gate's evidence comes back with the substrate it governs (P4b -- a
        # restart is not amnesia). A malformed tail is DROPPED whole, never half-applied;
        # a pre-T047 blob restores with an empty trajectory, exactly today's behaviour.
        try:
            if z_tt is not None and len(z_tt) == len(z_tu) == len(z_ta):
                self.trajectory = [
                    (int(z_tt[i]), int(z_tu[i]), float(z_ta[i])) for i in range(len(z_tt))
                ]
        except Exception:
            pass
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
            "init_pool": self.init_pool,
            "pool_stats": dict(self.pool_stats),
        }

    # ---- FR-033: candidate placement + the genome pool ----------------------------------------
    def _place_candidates(self, n_cand, step, lam):
        """Placements for one candidate pool, drawn BEFORE the fit so the placement that generates
        a unit's fan-in is the one the admitted unit keeps. Deterministic in the engine seed, the
        tick, the step and the scale; this grower's region is cortex (the H01 measured arm)."""
        from snn_data_lake.cortex.placement import place

        seed = (
            self.seed * 1_000_003 + self.ticks * 7_919 + step * 131 + int(lam * 1000)
        ) & 0xFFFFFFFF
        return place(n_cand, seed=seed)

    def _candidates(self, d):
        """The candidate pool for this refit and a census of the genome behind it. A genome needs
        >= 2 earned, placement-aligned units to fit; before that every mode is the random pool
        and `genome_fit` says so. `genome_shuffled` fits the SAME map on tag-shuffled units --
        the control that keeps the map's magnitude and span while destroying its tag->weight
        association -- so a genome gain that survives the shuffle was never about identity."""
        genome, info = None, {"genome_fit": False}
        prev = self._model
        if self.init_pool != "random" and prev is not None and prev.W is not None:
            n = int(prev.W.shape[0])
            # a restored substrate whose width differs from the live one cannot seed the map
            # (GenomicMap.fit would refuse the shape before NGSNN.fit could refuse the seed)
            if n >= 2 and len(self._placement) >= n and int(prev.W.shape[1]) == d:
                utags = unit_tags(self._placement[:n], "cortex")
                if self.init_pool == "genome_shuffled":
                    perm = random.Random(self.seed * 31 + self.ticks).sample(range(n), n)
                    utags = utags[perm]
                genome = GenomicMap()
                rmse = genome.fit(prev.W.detach().cpu().numpy(), feature_tags(d), utags)
                info = {
                    "genome_fit": True,
                    "genome_rmse": round(float(rmse), 6),
                    "rank_u": genome.rank_u,
                    "rank_f": genome.rank_f,
                    "fit_units": n,
                }
        ftags = feature_tags(d) if genome is not None else None
        return CandidatePool(self._place_candidates, "cortex", genome=genome, ftags=ftags), info


# -------------------------------------------------------------------------------------------------
# CROSS-PROCESS RESTART PROOF (P4b: "it persists" is a claim about a NEW PROCESS)
# -------------------------------------------------------------------------------------------------
# docs/POLICIES.md names this as the open P0 work on the neural-growth avenue: retention is built,
# `growth_state_bytes` / `restore_growth_state` are called on two live paths into the ONE lake under
# `_T_GROWTH` -- and NO CROSS-PROCESS RESTART HAD EVER BEEN DEMONSTRATED. The existing durability
# test round-trips a raw `np.zeros(4).tobytes()` blob to prove three lake tenants do not eat each
# other; it says nothing about a FITTED MODEL. Until now, both call sites lived in `cortex/brain.py`
# and neither serialisation method had a single assertion anywhere in the tree.
#
# P4b's honesty guard is explicit about what would settle it: "prove it by reading the lake back from
# disk after the writer is gone, not by inspecting the writer's own state." So the parent fits an
# engine, writes it through `substrate.home.save_growth` into an ISOLATED NORD_HOME, and exits that
# object's scope; a genuinely SEPARATE interpreter then boots, calls `load_growth()` +
# `restore_growth_state()`, and prints the fingerprint of what it restored. The parent compares.
# In-process restore is deliberately NOT the check: it cannot fail for the reason that matters.
_PROBE_FLAG = "--restart-probe"


def growth_fingerprint(engine):
    """crc32 over the grown tensors + the counters that interpret them (never builtin hash()).

    Covers W / b / beta byte-exactly plus n_units and the accuracy those weights were fit at, so a
    restore that drops a tensor, transposes one, or pairs the weights with a different fit's accuracy
    changes the fingerprint. Returns None when there is no substrate."""
    import zlib

    m = engine.grown_model
    if m is None or m.W is None or m.b is None or m.beta is None:
        return None
    acc = 0
    for t in (m.W, m.b, m.beta):
        acc = zlib.crc32(t.detach().cpu().numpy().tobytes(), acc)
    tail = (
        f"|{int(engine.grown_units)}|{float(engine.predict_accuracy):.9f}|{int(m.d_in)}|{int(m.m)}"
    )
    return zlib.crc32(tail.encode("ascii"), acc)


def _restart_probe(seed, **cfg):
    """Run in a NEW PROCESS: restore the grown substrate from the lake `NORD_HOME` points at and
    print `RESTORED <fingerprint> <n_units> <acc>` (or `RESTORED none ...`). ASCII, one line."""
    from snn_data_lake.substrate.home import load_growth

    eng = NeurogenesisSelfImprove(seed=seed, **cfg)
    ok = eng.restore_growth_state(load_growth())
    fp = growth_fingerprint(eng)
    print(
        f"RESTORED {ok} {fp if fp is not None else 'none'} "
        f"{eng.grown_units} {eng.predict_accuracy:.9f}"
    )


def _cross_process_restart(seed, cfg, ticks=140):
    """Fit an engine, persist it into an isolated lake, and read it back FROM A NEW INTERPRETER.

    Returns a dict of the parent-side and child-side fingerprints plus the lake size, so the caller
    can assert they agree AND that the lake was genuinely written."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    work = tempfile.mkdtemp(prefix="snn_growth_restart_")
    prior = os.environ.get("NORD_HOME")
    try:
        os.environ["NORD_HOME"] = work
        from snn_data_lake.substrate.home import knowledge_path, save_growth

        eng = _run(NeurogenesisSelfImprove(seed=seed, **cfg), ticks)
        retained = eng.substrate_is_retained()
        blob = eng.growth_state_bytes()
        parent_fp = growth_fingerprint(eng)
        parent_units, parent_acc = eng.grown_units, eng.predict_accuracy
        wrote = bool(save_growth(blob))
        lake_bytes = os.path.getsize(knowledge_path()) if os.path.exists(knowledge_path()) else 0
        # WHICH TAG the save landed under, measured off the isolated lake rather than asserted from
        # knowledge of save_growth's internals: since specs/022 T010-ii an npz blob lands as a
        # _T_GROWTH_DELTA chain record (FULL here -- fresh lake), and this count is what keeps the
        # printed claim below from outliving the encoding the way its "_T_GROWTH" wording did.
        from snn_data_lake.substrate.home import _T_GROWTH_DELTA, BinLog

        chain_records = sum(
            1 for p in BinLog(knowledge_path()).replay() if p[:1] == _T_GROWTH_DELTA
        )
        del eng  # the writer is gone before anything reads the lake back

        # A GENUINELY SEPARATE INTERPRETER. Same NORD_HOME, nothing else shared -- no module state,
        # no live object, no warm import. This is the only configuration in which "it persists" is
        # a statement about persistence rather than about the object still being in RAM.
        env = dict(os.environ)
        env["NORD_HOME"] = work
        env["PYTHONUTF8"] = "1"
        # PYTHONPATH is DERIVED, not inherited (fixed 2026-08-05). Inheriting it made this probe
        # depend on how its PARENT happened to be launched: `run_selfchecks.py` sets PYTHONPATH=src
        # for every child, so the probe passed there, while `tests/test_selfchecks.py` runs the same
        # self-check IN PROCESS under pytest -- which puts src on sys.path but NOT in the environment
        # -- so the child died with `ModuleNotFoundError: No module named 'snn_data_lake'`, printed
        # no RESTORED line, and was scored as `child_units = -1`, i.e. "the substrate did not
        # survive a restart". The failure looked like a P4b persistence regression and was actually
        # an unimportable child; the lake round-trip was verified intact throughout.
        #
        # That distinction matters more than the fix: a probe whose NEGATIVE result can be produced
        # by its own harness failing cannot tell "persistence is broken" from "I could not start".
        # The package root is derived from this file (cortex/ -> snn_data_lake/ -> src/) so the child
        # can import regardless of the parent's launcher, and the child now reports a start failure
        # distinctly rather than silently scoring as a persistence failure.
        _src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = (
            _src_root + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else _src_root
        )
        env["NORD_GROWTH_PROBE_CFG"] = json.dumps({"seed": seed, **cfg})
        proc = subprocess.run(
            [sys.executable, "-m", "snn_data_lake.cortex.neurogenesis_engine", _PROBE_FLAG],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        line = ""
        for ln in (proc.stdout or "").splitlines():
            if ln.startswith("RESTORED "):
                line = ln.strip()
        parts = line.split()
        # A child that never STARTED is not evidence about persistence. Distinguish the two loudly:
        # without this, an unimportable child prints no RESTORED line, scores child_units = -1, and
        # claim 5 reports "the grown substrate did NOT survive a new process" -- a false negative
        # about the brain caused by a broken harness. That is the inverse of the vacuous-assert
        # defect this project keeps finding, and it is just as misleading.
        if not line and proc.returncode != 0:
            raise AssertionError(
                "the restart PROBE failed to start, which says nothing about persistence: child "
                f"rc={proc.returncode}. This is a harness fault, not a P4b result. stderr tail:\n"
                f"{(proc.stderr or '')[-500:]}"
            )
        return {
            "retained": retained,
            "blob_bytes": len(blob) if blob else 0,
            "wrote": wrote,
            "lake_bytes": lake_bytes,
            "chain_records": chain_records,
            "parent_fp": parent_fp,
            "parent_units": parent_units,
            "parent_acc": parent_acc,
            "child_rc": proc.returncode,
            "child_line": line or (proc.stderr or "")[-300:],
            "child_ok": len(parts) == 5 and parts[1] == "True",
            "child_fp": int(parts[2]) if len(parts) == 5 and parts[2] != "none" else None,
            "child_units": int(parts[3]) if len(parts) == 5 else -1,
            "child_acc": float(parts[4]) if len(parts) == 5 else float("nan"),
        }
    finally:
        if prior is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prior
        shutil.rmtree(work, ignore_errors=True)


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
    # P5a: snapshot the routing decision THIS run's last refit made, before claims 2-5 build further
    # engines and overwrite it -- `decisions()` is keyed by workload and is last-write-wins by design.
    # Read live rather than assumed: claim 6 first failed against a width derived from `buffer_cap`,
    # because the entry it read belonged to claim 5's 140-tick restart probe (n=6,016, not 60,160).
    _route, _route_rows = decisions().get("neurogenesis_refit"), len(eng._buf_x)
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

    # claim 5: THE GROWN SUBSTRATE SURVIVES A NEW PROCESS (P4b). Fit, persist into an isolated lake,
    # drop the writer, and read it back from a separate interpreter -- the only form in which
    # "it persists" is a claim about persistence.
    rs = _cross_process_restart(seed=7, cfg=cfg, ticks=140)
    c5_pass = (
        rs["retained"]
        and rs["blob_bytes"] > 0
        and rs["wrote"]
        and rs["child_ok"]
        and rs["child_fp"] == rs["parent_fp"]
        and rs["child_units"] == rs["parent_units"]
    )

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
        (
            5,
            "grown substrate survives a NEW PROCESS",
            f"{rs['parent_units']}u crc {rs['parent_fp']} -> child {rs['child_units']}u",
            c5_pass,
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
    # claim 5 -- P4b's honesty guard, executed rather than asserted in prose.
    assert rs["retained"] and rs["blob_bytes"] > 0, (
        f"nothing was grown to persist (retained={rs['retained']}, blob={rs['blob_bytes']} B) -- "
        "the restart proof is vacuous without substrate"
    )
    assert rs["wrote"] and rs["lake_bytes"] > 0, (
        f"the grown substrate must reach the ONE lake on disk (wrote={rs['wrote']}, "
        f"lake={rs['lake_bytes']} B)"
    )
    assert rs["chain_records"] > 0, (
        "the save landed under NO _T_GROWTH_DELTA record -- the T010-ii chain cutover did not "
        "reach this path (a legacy fallback firing here would pass every other assert)"
    )
    assert rs["child_rc"] == 0 and rs["child_ok"], (
        f"a NEW PROCESS must restore the grown substrate from the lake "
        f"(rc={rs['child_rc']}, line={rs['child_line']!r})"
    )
    assert rs["child_fp"] == rs["parent_fp"] and rs["child_units"] == rs["parent_units"], (
        f"the restored substrate must be the SAME substrate: parent crc {rs['parent_fp']} "
        f"({rs['parent_units']}u, acc {rs['parent_acc']:.6f}) vs child crc {rs['child_fp']} "
        f"({rs['child_units']}u, acc {rs['child_acc']:.6f})"
    )
    assert c5_pass, "claim5: the grown substrate must survive a new process"

    # claim 6: THE REFIT IS WIDTH-ROUTED, NOT AVAILABILITY-ROUTED (P5a, 2026-08-05). This reads the
    # decision the refits ABOVE actually made -- it does not re-derive one -- so it fails if `_refit`
    # stops routing, if it routes on a placeholder width, or if the answer stops following the
    # measurement. The expected width is derived from cfg and the feature encoder, never written down.
    _feat = len(_mode_features(0, cfg["k_modes"]))
    _expect_n = (_route_rows - min(60, _route_rows // 3)) * _feat
    _rt = _route
    assert _rt is not None, (
        "the refit must route through substrate.device.select(); nothing recorded"
    )
    assert _rt["n"] == _expect_n, (
        f"the routed width must be the fit tensor's own element count {_expect_n}, got {_rt['n']} -- "
        "a router given the wrong n decides the wrong thing while looking perfectly healthy"
    )
    _xo = crossover()
    assert _rt["device"] == ("cuda" if (_xo is not None and _rt["n"] >= _xo) else "cpu"), (
        f"refit at n={_rt['n']} routed to {_rt['device']} against measured crossover {_xo} -- "
        "that is availability routing wearing the router's clothes"
    )
    # That the answer FLIPS above the crossover is asserted where its red lives, in
    # substrate/device.py's own self-check; repeating it here would be a second copy of one
    # criterion, and the copy in the module that cannot break it is the one with no reachable red.
    print(
        f"routing: refit n={_rt['n']} -> {_rt['device']} ({_rt['why']}); crossover {_xo}. "
        "NOT a speedup claim -- at this width CPU is the measured answer, and the point is that it "
        "is measured and flips on its own when the buffer or the feature width grows."
    )

    print(
        f"restart proof: {rs['parent_units']} grown units -> {rs['blob_bytes']} B npz -> "
        f"{rs['chain_records']} _T_GROWTH_DELTA chain record(s) in a {rs['lake_bytes']} B lake "
        f"(measured, not assumed -- the lake can be SMALLER than the blob because the chain sheds "
        f"the zip container) -> a SEPARATE interpreter restored crc {rs['child_fp']} "
        f"({rs['child_units']}u, acc {rs['child_acc']:.4f}), byte-equal to the writer's."
    )
    # PLACEMENT ALIGNMENT. Asserted here because it is invisible from either side alone: the weights
    # look fine, the placement list looks fine, and only their correspondence can be wrong. It goes
    # wrong at a PRUNE -- `prune_unearned` drops columns, so without reindexing by the survivor mask
    # every entry after the first prune would describe a DIFFERENT neuron and a Dale sign assigned
    # at birth would silently migrate. Measured over 600 ticks of this same run: 20 growth events,
    # 8 prunes, alignment held at every one.
    _placed = eng.placement
    assert len(_placed) == eng.grown_units, (
        f"placement/substrate MISALIGNED: {eng.grown_units} units but {len(_placed)} placements -- "
        "a placement that stops tracking its unit is worse than none, because it looks like "
        "structure while describing a different neuron"
    )
    _l1_exc = [p for p in _placed if p[1] == "Layer 1" and p[2] == 1]
    assert not _l1_exc, f"Dale broken after growth/prune churn: excitatory in Layer 1 {_l1_exc}"
    _census = " ".join(
        f"{k.replace('Layer ', 'L')}:{v}" for k, v in sorted(eng.layer_census().items())
    )
    print(
        f"placement: {len(_placed)} units carry a cortical depth, layer and immutable Dale sign "
        f"[{_census}] -- aligned through growth AND prune, and round-tripped in the blob above."
    )
    print("self-check OK: the brain self-improves by GROWING NG-SNN neurons to predict its own")
    print("next perception; held-out accuracy rises above chance as it grows (shuffled control")
    print("stays at chance); a genuine neurogenesis-driven self-improvement loop, perception-only")
    print("+ safe. HONEST NOTE: over a FIXED Markov structure the accuracy plateaus once the")
    print("transitions are learned -- open-ended improvement needs a growing/richer world (see")
    print("docs/SELF_IMPROVEMENT_RESEARCH.md).")


if __name__ == "__main__":
    import sys

    if _PROBE_FLAG in sys.argv:
        # The CHILD half of the cross-process restart proof (see _cross_process_restart). Never
        # reached by run_selfchecks, which invokes this module with no arguments.
        import json
        import os

        _cfg = json.loads(os.environ.get("NORD_GROWTH_PROBE_CFG") or "{}")
        _restart_probe(int(_cfg.pop("seed", 7)), **_cfg)
    else:
        main()
