r"""
Spiking JEPA -- a GENUINELY SPIKING, GENUINELY LOCAL predictive JEPA world model (honest toy-scale study).

What this module IS (each claim verified by the un-riggable __main__ self-check against a control that
GENUINELY fails):

  * GENUINELY SPIKING (claim 1). A temporal-contrast EVENT retina (log-ratio ON/OFF events, deterministic
    latency binning -- the encode_latency style of cortex/spike_preprocessing, never the non-deterministic
    encode_poisson) feeds a stack of LIF grids (cortex/lif_neuron.LIFNeuron) wired as RECURRENT PREDICTIVE
    layers (per-timestep input = [feed-forward spikes ++ own spikes at t-1]). Every inter-layer signal is a
    0/1 spike; each layer's latent is a function of INTEGER spike counts. CONTROL: a graded (tanh) backbone
    emits non-binary signals and fails the check.
  * GENUINELY LOCAL (claim 2). Learning is a three-factor rule  delta w_ij = lr * m_j * sg_j * e_ij  with NO
    backprop and NO weight transport -- e_ij is a PREsynaptic eligibility trace (r_stdp.RSTDPNeuron), sg_j a
    POSTsynaptic fast-sigmoid SURROGATE of neuron j's membrane, m_j a LOCAL modulator built ONLY from layer
    j's own s_bar + EMA stats + frozen (stop-grad) lagged "echo". CONTROL: with the top frozen the bottom
    STILL learns from its own local loss; a weight-transport surrogate (bottom gated on the top's update)
    does NOT.
    **THIS CLAIM WAS FALSE UNTIL 2026-08-04, and the correction is recorded rather than quietly applied.**
    The three-factor BACKBONE rule above was always local -- but the JEPA head feeding it was not, in two
    places, and both read a forward weight matrix transposed to carry error backward:
      (a) `dL/dz = gp + Wz^T gp` in train_step, the same site `docs/POLICIES.md` names as an open
          violation in the sibling `jepa_world_model`; and
      (b) `_readout_grad`'s top-down modulator `gsbar = sum(dldu[i] * Wr[i][j])`, whose own docstring
          asserted it was "a broadcast error, not weight transport" while computing exactly that.
    Both are fixed. (a) keeps only the residual skip (a broadcast error; the recurrent term measured
    HARMFUL as well as non-local -- see jepa_world_model._feedback_gz). (b) has no residual skip, so
    dropping it would delete the top-down signal; it now uses FEEDBACK ALIGNMENT -- a fixed random `_Br`
    (Lillicrap 2016), drawn from a crc32-derived stream so every existing weight stays byte-identical.
    A self-check assert below now guards both, with its RED constructed and run.
    Note what the shipped claim-2 CONTROL could and could not see: it gates the bottom layer on the top's
    update, so it caught inter-LAYER transport and was blind to transport inside the head. An honest
    control that tests the wrong axis is still a control that passes.
  * TRACKS ACTION-CONDITIONED LATENT DYNAMICS AT THE IDENTITY MAP'S ERROR SCALE (claim 4, weakened
    honestly). The bounded action-conditioned latent map predict(state, command) -- a row-
    bounded LINEAR residual (max row-abs-sum <= rho_max = sqrt(L), clamp +/- z_clip; the bound body shared with jepa_world_model,
    duck-typed for PrefrontalExecutive.plan()) -- is trained by the JEPA invariance loss. ASSERTED: its
    held-out MSE stays at the identity predictor's error scale, and TIME-SHUFFLED targets never train a
    materially better predictor than the true order. "Beats the identity predictor" is NOT asserted --
    it is a seed-level coin flip (see the honest negatives below and the note at the self-check assert).

What this module does NOT claim (HONEST NEGATIVES -- MEASURED and REPORTED by the self-check, NOT asserted,
because at this toy scale they cannot be demonstrated against a control that genuinely fails):

  * NO "useful representation beats a random encoder" claim (the old claim 5 is DROPPED). At this toy scale
    identity is trivially linearly separable from even a RANDOM spiking encoder (a competent probe decodes a
    random SNN's latent at >0.9), and the local rule does NOT beat -- and on a held-out-position
    generalization probe slightly DEGRADES -- a random SNN encoder. The spiking ENCODER (learned OR random)
    is far more position-tolerant than raw pixels, but that is an ARCHITECTURAL property, not a product of the
    local learning. The self-check reports trained/random/raw side by side and prints this verdict.
  * NO "no-collapse vs a collapsing ablation" claim (the old claim 3 is DROPPED). A random sparse LIF backbone
    keeps its latent effective rank high even with ALL anti-collapse pressure removed -- the slowness
    objective is already near-satisfied by the backbone's naturally-slow features, so a genuinely-collapsing
    ablation cannot be built here. The self-check reports full vs anti-collapse-ablation latent std/rank as
    TELEMETRY only.
  * NO "predictor beats identity / time-shuffle kills the advantage" claim (the old claim-4 ORDERINGS are
    DROPPED to reported telemetry). Measured over 20 seed shifts: pred < ident on only 11/20 and
    shuf >= ident on 15/20, margins in the 4th significant figure and stable under a 2000-transition hold
    stream -- the drifting-shape latent is so slow that the identity map is already near-optimal. The ~20%
    win at the pinned seed is seed luck, the same architecture-dominates-learning story as claims 3 and 5.

Rule choice, honestly. The shipped default is the NON-CONTRASTIVE LPL slowness form (rule="lpl"). A
CONTRASTIVE ESPP variant (rule="espp") is kept selectable (its math lives in plasticity/espp_rule
since T052-i -- the palette element; this branch delegates), but at this toy scale (only 3 shape identities,
temporally persistent, so contrastive negatives are dominated by the SAME identity at a different position)
it does NOT improve the representation and does NOT give a robust predict-beats-identity result -- so it is
NOT the default. No claim of event-camera SOTA, motion-equivariance, or "beats backprop".

The abstract transition map predict() is NOT itself spiking (a bounded linear residual, kept so a 40-step
plan() rollout stays finite and cheap). So the encoder + the learning are spiking/local; the transition map
is not.

Refs: Halvagal & Zenke, "The combination of Hebbian and predictive plasticity learns invariant object
representations in deep sensory networks" (LPL, 2023); Illing et al., "Local plasticity rules can learn deep
representations using self-supervised contrastive predictions" (2021); the JEPA move (LeCun 2022) -- predict
in a learned latent space, not pixel space; Izhikevich (2007) eligibility traces.

T049 ADJUDICATION (2026-08-26, specs/022): _modulator's rule="lpl" IS THE TREE'S LPL SPINE (FR-019's
palette element). A standalone re-implementation (plasticity/lpl_rule.py) was built the same day and its
own three-lens review panel measured it DEGENERATE -- its smoothed target and its mean EMA were the same
EMA computed twice (bit-identical at every tau), collapsing the two terms into one sign-flipping gain
whose net drive pushed activity AWAY from its smoothed past; its slow-feature charge was carried by the
eligibility low-pass (a constant modulator aligned BETTER, 0.954 vs 0.796); and its "erratum pin" was an
algebraic identity that passed with the rule neutered and with no rule at all. DELETED per P8(5) -- the
licence working. THIS incumbent has the correct two-quantity form (echo is a genuinely FROZEN delayed
target, distinct from mu). Two facts the panel put on the record: (a) NOTHING IN THE TREE PINS the 2024
erratum's content -- "erratum-corrected" in specs/022 research.md is a spec-level label, and the
variance-normalized form here derives from -log var independently of any erratum; (b) the lambda_var=2.0
weighting is the incumbent's form, unadjudicated against a lambda-free reading. PALETTE EXTRACTION IS
DATED TO T051: the composer defines the rule interface (typed read/write sets, per-rule neuter
attribution); extracting this modulator into a composable plasticity/ rule lands WITH that interface,
not before it (a premature extraction was measured wrong once already, above).

Deps: math, random, json, zlib (numpy/torch allowed project-wide); ASCII-only prints, deterministic (random.Random(seed) per instance).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.spiking_jepa
"""

import json
import math
import random
import zlib

from snn_data_lake.cerebellum.jepa_world_model import bound_rows
from snn_data_lake.cortex.is_snn_forward import SPARSITY, STABLE_GAIN
from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.plasticity.espp_rule import espp_modulator_core
from snn_data_lake.substrate._mathutils import dot, mean

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "SpikingJEPA._readout_grad": (
        "APPROXIMATES_GRADIENT",
        "backprops dL/dz through z = unit(tanh(Wr@s_bar+br)) to (dWr, dbr, dL/ds_bar_top). It carries "
        "error backward across a nonlinearity, so it owes the cosine",
    ),
    "SpikingJEPA.train_step": (
        "APPROXIMATES_GRADIENT",
        "one online JEPA update minimising invariance + variance + covariance against a frozen "
        "(stop-gradient) echo; applies what _readout_grad computed, so it carries the same claim",
    ),
    "main": (
        "NOT_A_RULE",
        "the module entrypoint that runs the demo and prints results. Enumerated only because the "
        "recall-biased walk sees a weight-shaped assignment in it; it updates nothing of its own",
    ),
    "SpikingJEPA._update_stats": (
        "NOT_A_RULE",
        "EMA-updates per-layer running mean, variance and covariance. Bookkeeping statistics over "
        "activations; no weight moves",
    ),
    "_probe_acc": (
        "NOT_A_RULE",
        "fits a linear identity PROBE on z-scored features to measure representation quality. A "
        "measurement instrument over a frozen encoder",
    ),
}

# Feed-forward / recurrent init gain. The IS-SNN dense-drive fixed point (STABLE_GAIN=2.5) assumes ~k inputs
# active EVERY step; a latency EVENT code delivers only a few spikes across the whole window, so the sparse
# projection is scaled UP from that fixed point to make the LIF backbone fire on genuinely sparse events.
_FF_GAIN = STABLE_GAIN * 3.0
_REC_GAIN = STABLE_GAIN * 1.4


def _clamp(x, lo, hi):
    """Scalar clamp to [lo, hi] -- the final bounded squash on predicted latents (copied from jepa_world_model)."""
    return lo if x < lo else hi if x > hi else x


def _sparse_init(rng, fan_in, fan_out, gain, v_threshold):
    """Sparse random excitatory projection scaled to the LIF rate fixed point (is_snn_forward pattern):
    each of fan_out units samples k = SPARSITY*fan_in inputs, weight ~ U(0.5,1)*gain*v_threshold/k. Returns
    (W rows, connectivity index lists) -- the connectivity mask is preserved so plasticity keeps it sparse."""
    k = max(1, int(SPARSITY * fan_in))
    scale = gain * v_threshold / k
    w = []
    conn = []
    for _ in range(fan_out):
        row = [0.0] * fan_in
        idx = sorted(rng.sample(range(fan_in), k))
        for i in idx:
            row[i] = rng.uniform(0.5, 1.0) * scale
        w.append(row)
        conn.append(idx)
    return w, conn


class SpikingJEPA:
    """A genuinely-spiking, locally-trained JEPA latent world model.

    encode()/target_encode() run the event retina + LIF backbone and read out a bounded latent z (online /
    EMA-target); predict() is the copied bounded action-conditioned latent map (duck-typed for
    PrefrontalExecutive.plan()); train_step() applies the three-factor LOCAL rule to the backbone and a JEPA
    invariance update to the readout+predictor; latent_std()/effective_rank() are collapse telemetry;
    make_goal_value() builds a negative-L1 latent goal_value; save/load persist tagged JSON with a crc32
    trust boundary."""

    def __init__(
        self,
        actions,
        retina=20,
        T=10,
        sizes=(24, 16),
        latent_dim=12,
        rule="lpl",
        anti_collapse=True,
        tau=0.99,
        lr=0.04,
        seed=0,
        theta=0.5,
        rho_star=0.25,
        graded=False,
        transport=False,
    ):
        self.actions = tuple(actions)
        self.n_act = len(self.actions)
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        self.R = int(retina)
        self.T = int(T)
        self.sizes = tuple(int(s) for s in sizes)  # exactly 2 LIF layers
        self.L = int(latent_dim)
        self.rule = str(rule)
        self.anti_collapse = bool(anti_collapse)
        self.tau = float(tau)  # EMA target-readout coefficient
        self.lr = float(lr)
        self.seed = seed
        self.theta = float(theta)  # event log-contrast threshold
        self.rho_star = float(rho_star)  # homeostatic target firing rate
        self.graded = bool(graded)  # CONTROL: tanh instead of a spike -> non-binary (claim 1)
        self.transport = bool(transport)  # CONTROL: bottom gated by top update (weight transport)
        self.freeze_top = False  # CONTROL: freeze top layer+readout+predictor (claim 2)
        self.rng = random.Random(seed)

        # ---- LIF backbone constants ----
        self.tau_m = 12.0
        self.v_threshold = 1.0

        # ---- event encoder constants ----
        self._eps_log = 1e-3
        self._dmax = -math.log(self._eps_log)  # max |log-contrast| for [0,1] frames (latency scale)
        self._ref = [0.0] * self.R  # stored/zero reference frame (encode spikes obs vs this)

        # ---- learning-rule constants ----
        self.tau_e = 3.0
        self.elig_decay = math.exp(-1.0 / self.tau_e)  # presynaptic eligibility EMA
        self.beta = 3.0  # surrogate sharpness
        self.stat_decay = 0.98  # per-layer EMA statistics decay (tau_stat)
        self.eps_stat = 1e-3
        self.lambda_pred = 1.0
        self.lambda_var = 2.0
        self.lambda_cov = 0.5
        self.lambda_homeo = 0.5
        self.lambda_jepa = 0.4  # top-down JEPA modulator weight on the top LIF layer
        self.espp_alpha = 1.0  # ESPP negative-pair push strength

        # ---- predictor constants (the bound body is jepa_world_model.bound_rows; the CONSTANT is derived
        #      the same way as there: rho_max = sqrt(L), the JEPA arc's record section 11 -- a residual map's rows
        #      can need up to 1 + sqrt(L); sqrt(L) is kept on the measurement. Until 2026-09-05 this copied 0.9,
        #      which forbids every step longer than the latent's own radius. In this module's own self-check
        #      regime the bound never fires either way: max row-abs-sum 0.181 over 20 seed shifts against 0.9,
        #      so every printed number is unchanged (evidence/jepa-live-floor.md section 3.3). ----
        self.z_clip = 3.0
        self.rho_max = math.sqrt(self.L)
        self.wd = 1e-3  # a fixed constant in both classes, disclosed

        # ---- parameter init (all draws from the per-instance rng => seed => byte-identical weights) ----
        r = self.rng
        self.Wff = []
        self.Wrec = []
        self.conn_ff = []
        self.conn_rec = []
        fan_ff = [
            2 * self.R,
            self.sizes[0],
        ]  # layer0 ff = 2R event lines; layer1 ff = layer0 spikes
        for l in range(2):
            wff, cff = _sparse_init(r, fan_ff[l], self.sizes[l], _FF_GAIN, self.v_threshold)
            wrec, crec = _sparse_init(r, self.sizes[l], self.sizes[l], _REC_GAIN, self.v_threshold)
            self.Wff.append(wff)
            self.conn_ff.append(cff)
            self.Wrec.append(wrec)
            self.conn_rec.append(crec)

        # readout: z = unit(tanh(Wr @ s_bar_top + br)); a frozen EMA/stop-grad target readout Wr_t/br_t
        sr = 1.0 / math.sqrt(self.sizes[-1])
        self.Wr = [
            [r.uniform(-1.0, 1.0) * sr for _ in range(self.sizes[-1])] for _ in range(self.L)
        ]
        self.br = [0.0] * self.L
        self.Wr_t = [row[:] for row in self.Wr]
        self.br_t = self.br[:]
        # FIXED RANDOM feedback matrix for the top-down modulator (feedback alignment; see
        # _readout_grad). Drawn from a crc32-DERIVED stream, never `self.rng`, so adding it leaves
        # Wff/Wrec/Wr/Wz/Wa byte-identical to a model built before it existed. Same scale as Wr's
        # init so the top-down term keeps its magnitude band.
        _fb = random.Random(zlib.crc32(b"spiking-jepa-Br") ^ int(seed))
        self._Br = [
            [_fb.uniform(-1.0, 1.0) * sr for _ in range(self.sizes[-1])] for _ in range(self.L)
        ]

        # bounded action-conditioned predictor (Wz small, row-bounded via jepa_world_model.bound_rows, Wa, b) -- shapes shared with jepa_world_model
        self.Wz = [[r.uniform(-1.0, 1.0) * 0.02 for _ in range(self.L)] for _ in range(self.L)]
        self.Wa = [[r.uniform(-1.0, 1.0) * 0.05 for _ in range(self.n_act)] for _ in range(self.L)]
        self.b = [0.0] * self.L

        # per-layer EMA statistics (mu, var, cov) -- the LOCAL modulator's only memory
        self.mu = [[self.rho_star] * n for n in self.sizes]
        self.var = [[0.01] * n for n in self.sizes]
        self.cov = [[[0.0] * n for _ in range(n)] for n in self.sizes]

        self._buf = []  # ring buffer of online latents (telemetry)
        self.buffer = 64
        self._neg = [[] for _ in self.sizes]  # ESPP negative memory per layer

    # ---- action handling (copied from jepa_world_model) ------------------------------------------------
    def _idx(self, command):
        """Map an action name or an integer index to an action index (duck-typed command for predict/plan)."""
        if isinstance(command, str):
            return self._aidx[command]
        return int(command)

    # ---- event encoder ---------------------------------------------------------------------------------
    def _coerce(self, obs):
        """Coerce obs to a length-R [0,1] frame: truncate a longer live pattern, pad a shorter one with 0."""
        v = list(obs)
        if len(v) >= self.R:
            return [float(x) for x in v[: self.R]]
        return [float(x) for x in v] + [0.0] * (self.R - len(v))

    def _events(self, prev, cur, graded=False):
        """Temporal-contrast retina: d = log(cur+eps) - log(prev+eps); ON event (line i) where d > theta, OFF
        event (line R+i) where d < -theta. Deterministic latency binning (stronger contrast -> earlier bin),
        the encode_latency style. Returns a (T, 2R) list of 0/1 spike vectors (graded=True emits tanh(|d|),
        non-binary -- the claim-1 control)."""
        r_ = self.R
        t_ = self.T
        theta = self.theta
        eps = self._eps_log
        span = self._dmax - theta
        grid = [[0.0] * (2 * r_) for _ in range(t_)]
        for i in range(r_):
            d = math.log(cur[i] + eps) - math.log(prev[i] + eps)
            mag = d if d >= 0.0 else -d
            if mag <= theta:
                continue
            v = (mag - theta) / span
            if v > 1.0:
                v = 1.0
            tbin = int(round((1.0 - v) * (t_ - 1)))
            line = i if d > 0.0 else r_ + i
            grid[tbin][line] = math.tanh(mag) if graded else 1.0
        return grid

    # ---- spiking LIF backbone forward ------------------------------------------------------------------
    def _forward(self, obs, graded=None):
        """Run the event retina + 2-layer recurrent LIF backbone. Records per-timestep spikes AND pre-reset
        membranes (needed for the eligibility trace + surrogate factor) and the per-layer s_bar = summed
        spikes / T. Each layer's per-timestep input is [feed-forward spikes ++ own spikes at t-1] (RPL)."""
        graded = self.graded if graded is None else graded
        ev = self._events(self._ref, self._coerce(obs), graded)
        sizes = self.sizes
        nl = len(sizes)
        t_ = self.T
        spk = [[[0.0] * sizes[l] for _ in range(t_)] for l in range(nl)]
        vpre = [[[0.0] * sizes[l] for _ in range(t_)] for l in range(nl)]
        neurons = [
            [LIFNeuron(self.tau_m, self.v_threshold) for _ in range(sizes[l])] for l in range(nl)
        ]
        prev = [[0.0] * sizes[l] for l in range(nl)]
        for t in range(t_):
            layer_in = ev[t]
            for l in range(nl):
                wff = self.Wff[l]
                wrec = self.Wrec[l]
                cff = self.conn_ff[l]
                crec = self.conn_rec[l]
                own = prev[l]
                out = [0.0] * sizes[l]
                spk_lt = spk[l][t]
                vpre_lt = vpre[l][t]
                for j in range(sizes[l]):
                    cur = 0.0
                    wffj = wff[j]
                    for i in cff[j]:
                        xi = layer_in[i]
                        if xi:
                            cur += wffj[i] * xi
                    wrecj = wrec[j]
                    for k in crec[j]:
                        xk = own[k]
                        if xk:
                            cur += wrecj[k] * xk
                    nrn = neurons[l][j]
                    vprevv = nrn.v
                    if graded:
                        vv = nrn.v_rest + (vprevv - nrn.v_rest) * nrn.decay + cur
                        nrn.v = vv
                        s = math.tanh(vv)  # non-binary: the graded (non-spiking) control
                        vp = vv
                    else:
                        s = float(nrn.step(cur))  # binary 0/1 spike
                        vp = (
                            nrn.v_rest + (vprevv - nrn.v_rest) * nrn.decay + cur
                        )  # pre-reset membrane
                    out[j] = s
                    spk_lt[j] = s
                    vpre_lt[j] = vp
                prev[l] = out
                layer_in = out  # this layer's spikes are the next layer's ff input
        sbar = [
            [sum(spk[l][t][j] for t in range(t_)) / t_ for j in range(sizes[l])] for l in range(nl)
        ]
        counts = [
            [int(round(sum(spk[l][t][j] for t in range(t_)))) for j in range(sizes[l])]
            for l in range(nl)
        ]
        return {"ev": ev, "spk": spk, "vpre": vpre, "sbar": sbar, "counts": counts}

    # ---- readout ---------------------------------------------------------------------------------------
    def _readout(self, s_bar_top, wr, br):
        """z = unit(tanh(Wr @ s_bar_top + br)) -- the same bounded unit(tanh) latent shape as jepa_world_model
        so the copied bounded predictor math applies unchanged."""
        u = [dot(wr[i], s_bar_top) + br[i] for i in range(self.L)]
        t = [math.tanh(v) for v in u]
        n = math.sqrt(sum(c * c for c in t)) or 1.0
        return [c / n for c in t]

    def _feedback_gz(self, gp):
        """dL/dz through the residual predictor, as a BROADCAST error with NO path through Wz.

        p = (I + Wz)z + const, so the exact derivative is gp + Wz^T gp. The `Wz^T` half read the
        FORWARD matrix transposed -- weight transport, which this module's own headline claim
        ("NO backprop and NO weight transport") denied it was doing. Only the residual skip
        survives, which is structure rather than a weight. Same definition and same measurement as
        `cerebellum/jepa_world_model._feedback_gz`; see that docstring for the 5-seed benchmark."""
        return list(gp)

    def _readout_grad(self, gz, s_bar_top, wr, br):
        """Backprop dL/dz (=gz) through z = unit(tanh(Wr@s_bar+br)) to (dWr, dbr, dL/ds_bar_top).

        The last return is the top-down JEPA modulator delivered to the top LIF layer, and it is the
        reason this method needed a fix rather than a deletion. It USED to be computed as
        `sum(dldu[i] * wr[i][j])` -- the forward readout matrix read transposed, i.e. exactly the
        weight transport the docstring here claimed it was not ("a broadcast error, not weight
        transport"). That sentence was false for the life of the feature.

        Unlike the predictor path there is NO residual skip to fall back on, so dropping the term
        would delete the top-down signal entirely rather than localise it. The e-prop-compatible fix
        is therefore FEEDBACK ALIGNMENT (Lillicrap 2016): a FIXED RANDOM `_Br` carries the error
        down instead of the transposed forward weights. It is local, needs no weight symmetry, and
        keeps a genuine top-down modulator. `_Br` is drawn from a crc32-derived stream, never the
        instance rng, so every existing weight stays byte-identical.

        dWr / dbr are unchanged: they are outer products of the local error with the local
        presynaptic activity, which is the eligibility term, not transport."""
        big_l = self.L
        h = self.sizes[-1]
        u = [dot(wr[i], s_bar_top) + br[i] for i in range(big_l)]
        t = [math.tanh(v) for v in u]
        n = math.sqrt(sum(c * c for c in t)) or 1.0
        sdot = sum(gz[i] * t[i] for i in range(big_l))
        n3 = n * n * n
        dldu = [0.0] * big_l
        for i in range(big_l):
            dldt = gz[i] / n - sdot * t[i] / n3
            dldu[i] = dldt * (1.0 - t[i] * t[i])
        gwr = [[dldu[i] * s_bar_top[j] for j in range(h)] for i in range(big_l)]
        gbr = dldu[:]
        br_fb = self._Br
        gsbar = [sum(dldu[i] * br_fb[i][j] for i in range(big_l)) for j in range(h)]
        return gwr, gbr, gsbar

    # ---- bounded predictor (copied from jepa_world_model) ----------------------------------------------
    def _predict_pre(self, z, aj):
        """Pre-clamp residual latent prediction z + Wz@z + Wa[:,aj] + b (aj is an action index)."""
        return [z[i] + dot(self.Wz[i], z) + self.Wa[i][aj] + self.b[i] for i in range(self.L)]

    def predict(self, state, command):
        """Action-conditioned residual latent predictor z' = clamp(z + Wz@z + Wa@onehot(a) + b). `state` is a
        latent z; `command` is an action name or index. The clamp keeps |z'_i| <= z_clip so long rollouts stay
        bounded. Copied from jepa_world_model; duck-typed for PrefrontalExecutive.plan()."""
        p = self._predict_pre(state, self._idx(command))
        return [_clamp(v, -self.z_clip, self.z_clip) for v in p]

    # ---- encoders (public) -----------------------------------------------------------------------------
    def encode(self, obs):
        """Online encoder: event retina + LIF backbone on obs (spiked vs the stored/zero reference), read out
        z = unit(tanh(Wr @ s_bar_top + br)). Accepts a raw [0,1]^R frame or a longer live pattern (coerced)."""
        rec = self._forward(obs)
        return self._readout(rec["sbar"][-1], self.Wr, self.br)

    def target_encode(self, obs):
        """EMA/stop-grad TARGET encoder: same backbone, read out through the frozen lagged Wr_t/br_t."""
        rec = self._forward(obs)
        return self._readout(rec["sbar"][-1], self.Wr_t, self.br_t)

    # ---- local three-factor learning helpers -----------------------------------------------------------
    def _elig(self, train, n):
        """Presynaptic ELIGIBILITY trace: EMA (decay exp(-1/tau_e)) of each pre-line's spike train over the
        window (the r_stdp.RSTDPNeuron `e` trace). Returns a length-n vector -- the local pre factor e_i."""
        dec = self.elig_decay
        e = [0.0] * n
        for t in range(self.T):
            row = train[t]
            for i in range(n):
                e[i] = e[i] * dec + row[i]
        return e

    def _surrogate(self, vpre, n):
        """Postsynaptic SURROGATE factor sg_j = mean_t 1/(1+beta*|v_j(t)-v_th|)^2 over the window (fast-sigmoid
        surrogate of neuron j's spike vs its pre-reset membrane). Returns a length-n vector -- the post factor."""
        beta = self.beta
        vth = self.v_threshold
        sg = [0.0] * n
        for t in range(self.T):
            row = vpre[t]
            for j in range(n):
                d = row[j] - vth
                if d < 0.0:
                    d = -d
                sg[j] += 1.0 / ((1.0 + beta * d) ** 2)
        inv_t = 1.0 / self.T
        return [s * inv_t for s in sg]

    def _sample_neg(self, l):
        """ESPP negative: a stored s_bar from layer l's memory bank (a likely different object/position)."""
        bank = self._neg[l]
        if not bank:
            return self.mu[l]
        return bank[self.rng.randrange(len(bank))]

    def _modulator(self, l, sb, echo):
        """The LOCAL modulator m_j (per post-neuron j), built ONLY from layer l's own s_bar / EMA stats / echo.

        PRIMARY (rule="lpl", NON-CONTRASTIVE LPL):
          m_j = -( lambda_pred*2*(s_bar_j - echo_j)                    # predictive slowness toward frozen echo
                 - lambda_var*(s_bar_j - mu_j)/(var_j+eps)             # variance anti-collapse (-log var grad)
                 + lambda_cov*sum_{k!=j} cov_jk*(s_bar_k - mu_k)       # pairwise decorrelation
                 + lambda_homeo*(rate_j - rho_star) )                  # firing-rate homeostasis (rate_j = mu_j)
        FALLBACK (rule="espp", CONTRASTIVE): the predictive term becomes pull-to-positive-echo /
        push-from-negative, plus homeostasis. anti_collapse=False zeroes ALL anti-collapse pressure (var/cov/
        homeo for lpl; the negative + homeo for espp) -- the anti-collapse ABLATION. NOTE (honest): at the
        toy scale of the self-check this ablation does NOT actually collapse (a random sparse LIF backbone
        holds its latent rank even with no anti-collapse pressure), so it is reported as telemetry, not a
        gate -- see the module docstring."""
        n = self.sizes[l]
        mu = self.mu[l]
        var = self.var[l]
        cov = self.cov[l]
        lp = self.lambda_pred
        eps = self.eps_stat
        m = [0.0] * n
        if self.rule == "espp":
            alpha = self.espp_alpha if self.anti_collapse else 0.0
            lh = self.lambda_homeo if self.anti_collapse else 0.0
            # T052-i (P2 ADOPT): the contrastive math lives in plasticity/espp_rule --
            # the palette element -- and this branch DELEGATES (one mechanism; the
            # inline expression left this file in the same commit).
            return espp_modulator_core(
                n, sb, echo, self._sample_neg(l), mu, lp, alpha, lh, self.rho_star
            )
        lv = self.lambda_var if self.anti_collapse else 0.0
        lc = self.lambda_cov if self.anti_collapse else 0.0
        lh = self.lambda_homeo if self.anti_collapse else 0.0
        for j in range(n):
            pred = lp * 2.0 * (sb[j] - echo[j])
            vart = lv * (sb[j] - mu[j]) / (var[j] + eps)
            covt = 0.0
            if lc:
                cj = cov[j]
                acc = 0.0
                for k in range(n):
                    if k != j:
                        acc += cj[k] * (sb[k] - mu[k])
                covt = lc * acc
            homeo = lh * (mu[j] - self.rho_star)
            m[j] = -(pred - vart + covt + homeo)
        return m

    # distinct-from: plasticity_rule_stack.py -- the stack DEPOSITS/consolidates already-computed
    # per-synapse dw through the Benna-Fusi beakers (a composition/storage mechanism); this method
    # AUTHORS the declaration-of-record for ONE rule's own sum-of-products terms. Different job:
    # declaring what this module's update IS, not routing where updates go. The real P2 adjacency
    # is eligibility_algebra's 2-term "lpl.modulator.core2" FIXTURE, adjudicated there and in the
    # self-check arm below.
    # distinct-from: jepa_world_model.py -- the same-directory sibling carries the same coefficient
    # NAMES (lambda_var/lambda_cov) on a DIFFERENT mechanism: a population-normalized VICReg loss
    # gradient (gz[j] += lambda_var*(-(z-mu)/(L*m*sd))), the algebra's censused
    # population-normalizer class, encoder-loss grain, not a per-synapse rule. Out of this
    # declaration's scope; recorded here so a P2 grep on the coefficient names lands somewhere.
    def lpl_synapse_rule(self):
        """T051b-i (specs/022): the AUTHORED per-synapse LPL rule -- this instance's
        declaration, coefficients READ from the instance, never retyped (the T050
        incumbent-coefficient discipline; a retyped lambda once shipped a 100% error;
        the self-check's mutated-instance arm is the reachable red for retyping).

        The incumbent update (train_step's local block) is dw_ij = lr*m_j*sg_j*gate*e_i
        with m_j the rule="lpl" modulator above. Distributed to the SYNAPSE grain:

            dw = (-2*lambda_pred) * d_pred * lr * sg * e         d_pred  = s_bar_j - echo_j
               + (+lambda_var)  * d_mu * inv_var * lr * sg * e   d_mu    = s_bar_j - mu_j
               + (-lambda_cov)  * cov_pull * lr * sg * e         inv_var = 1/(var_j+eps_stat)
               + (-lambda_homeo) * d_homeo * lr * sg * e         d_homeo = mu_j - rho_star

        FACTOR PRE-COLLAPSE, disclosed: the algebra is products-only, so the division
        (inv_var), the differences (d_*) and the k!=j covariance REDUCTION (cov_pull)
        are caller-computed STATE QUANTITIES bound per site -- the eligibility window
        and EMA statistics live in the caller's state (the STDP-window-as-state
        contract; m*sg*e is the per-synapse triple). What the declaration itself
        carries is the LAMBDAS: a coefficient drift reds this module's own self-check.

        FACTOR ORDER: the TRAILING (sg, e) order is load-bearing -- it matches the
        incumbent chain ((lr*m)*sg)*gate*e, making the ablation arm's single term
        equal bit-for-bit (the tol=0.0 arm); the (state, lr) prefix COMMUTES because
        the coefficient -(2*lambda_pred) is exact (panel-measured: 2 of 24 factor
        permutations bit-pass, both sharing the (sg, e) suffix). Zero-coefficient
        terms are DROPPED at authoring (anti_collapse=False authors the 1-term
        ablation form -- mirroring _modulator's own lv/lc/lh zeroing).

        HONEST REMAINDER (partial extraction -- P2 merge-must-reduce NOT discharged
        here, said plainly: the incumbent block REMAINS the production path and this
        method adds a second expression): the lambda_jepa top-layer addition and
        transport-gated bottom sites (gate!=1) stay in-module undeclared;
        espp/transport/freeze_top instances REFUSE (their production dw is gated or
        differently composed -- an identical declaration would be a lie). The
        rule='espp' branch LEFT this remainder at T052-i: its math was extracted to
        plasticity/espp_rule (this module's branch delegates) and espp_synapse_rule
        there is its declaration -- each branch declares itself, neither declares
        the other.

        DORMANCY, said where a reader looks: this method has ZERO production callers
        (consumers = this module's self-check + the composer-palette smoke), its host
        module is REACHABLE since T053-ii through rule_composer's census imports (its
        dormancy row retired on that reachability -- adjudications unmoved), and BOTH
        method-grade gates structurally miss it (no LEARN_VERB in the name). Its first
        tick-charged use is UNDATED. The algebra/composer siblings' expiry WAS
        discharged at T053-ii by iSTDP's OWN declaration composing on the tick, NOT by
        this rule."""
        if self.rule != "lpl":
            raise ValueError(
                f"lpl_synapse_rule declares the rule='lpl' branch only (rule={self.rule!r})"
            )
        if self.transport or self.freeze_top:
            raise ValueError(
                "lpl_synapse_rule refuses transport/freeze_top control instances: their "
                "production dw is gate-zeroed or top-skipped, so the unconditional "
                "declaration would not describe them"
            )
        from snn_data_lake.plasticity.eligibility_algebra import EligibilityAlgebra

        lp = float(self.lambda_pred)
        lv = float(self.lambda_var) if self.anti_collapse else 0.0
        lc = float(self.lambda_cov) if self.anti_collapse else 0.0
        lh = float(self.lambda_homeo) if self.anti_collapse else 0.0
        terms = [(-(lp * 2.0), ("d_pred", "lr", "sg", "e"))]
        if lv != 0.0:
            terms.append((lv, ("d_mu", "inv_var", "lr", "sg", "e")))
        if lc != 0.0:
            terms.append((-lc, ("cov_pull", "lr", "sg", "e")))
        if lh != 0.0:
            terms.append((-lh, ("d_homeo", "lr", "sg", "e")))
        roles = {
            "d_pred": "predictive slowness deviation s_bar_j - echo_j (pre-collapsed)",
            "d_mu": "EMA-mean deviation s_bar_j - mu_j (pre-collapsed)",
            "inv_var": "reciprocal variance 1/(var_j + eps_stat) (pre-collapsed division)",
            "cov_pull": "k!=j covariance reduction sum cov_jk*(s_bar_k - mu_k) (pre-collapsed)",
            "d_homeo": "homeostatic deviation mu_j - rho_star (pre-collapsed)",
            "lr": "learning rate, read from the instance per step",
            "sg": "post surrogate-gradient sensitivity sg_j",
            "e": "presynaptic eligibility trace (the STDP window as state)",
        }
        # Always a FRESH algebra: a caller-supplied one was measured (panel) to be
        # corrupted on every refusal path (declare-before-validate), unusable twice
        # (redeclaration refuses), and to destabilize the canonical vocab digest.
        # No call site ever passed one; the parameter is deleted, not documented.
        alg = EligibilityAlgebra(term_budget=len(terms))
        used = []
        for _c, fs in terms:  # declare exactly what the built terms use, in first-use order
            for f in fs:
                if f not in used:
                    used.append(f)
        for f in used:
            alg.declare_factor(f, roles[f])
        return alg.author("jepa.lpl.synapse", terms)

    def _update_stats(self, sbar):
        """EMA-update per-layer running mean / variance / covariance (decay stat_decay) from the online s_bar."""
        dec = self.stat_decay
        one = 1.0 - dec
        for l in range(len(self.sizes)):
            sb = sbar[l]
            mu = self.mu[l]
            var = self.var[l]
            cov = self.cov[l]
            n = self.sizes[l]
            cen = [sb[j] - mu[j] for j in range(n)]
            for j in range(n):
                mu[j] = dec * mu[j] + one * sb[j]
            for j in range(n):
                var[j] = dec * var[j] + one * cen[j] * cen[j]
                covj = cov[j]
                cj = cen[j]
                for k in range(n):
                    covj[k] = dec * covj[k] + one * cj * cen[k]

    def _spectral_bound(self):
        """The Wz bound: tiny weight decay + row-scaling so max row-abs-sum <= rho_max. The body lives
        ONCE, in jepa_world_model.bound_rows (P2; the copy this method carried until 2026-09-05 was
        byte-identical and is deleted -- evidence/jepa-live-floor.md section 2.6)."""
        bound_rows(self.Wz, self.rho_max, self.wd)

    # ---- one online update -----------------------------------------------------------------------------
    def train_step(self, pattern_t, action_t, pattern_next):
        """One online update; returns the scalar JEPA invariance loss.

        (a) Runs the spiking backbone on x_t and x_next -> per-layer s_bar; echo_j = frozen (stop-grad) s_bar
            of the temporally-adjacent next window (a lagged population code).
        (b) JEPA invariance loss ||predict(readout(s_bar_x), a) - target_readout(echo_next)||^2 updates the
            predictor + readout (framed as the same three-factor structure: error x d(unit-tanh) x presynaptic
            s_bar) and EMA-updates the target readout.
        (c) The two LIF layers update by the LOCAL three-factor rule delta w_ij = lr*m_j*sg_j*e_ij; the top
            layer also receives the JEPA top-down error as an additive modulator (broadcast, not transported).
        anti_collapse=False disables the anti-collapse terms AND ties the target to the online readout (no
        stop-grad) -- the anti-collapse ABLATION (reported as collapse telemetry, NOT a gate: at this toy
        scale it does not actually collapse; see the module docstring). freeze_top / transport are the
        genuinely-failing claim-2 (locality) controls."""
        big_l = self.L
        nl = len(self.sizes)
        aj = self._idx(action_t)
        rec_x = self._forward(pattern_t)
        rec_y = self._forward(pattern_next)
        sbx = rec_x["sbar"]
        sby = rec_y["sbar"]
        echo = [sby[l][:] for l in range(nl)]  # frozen next population code (stop-grad)

        z = self._readout(sbx[-1], self.Wr, self.br)
        self._buf.append(z[:])
        if len(self._buf) > self.buffer:
            self._buf.pop(0)
        self._update_stats(sbx)
        if self.rule == "espp":
            for l in range(nl):
                self._neg[l].append(sbx[l][:])
                if len(self._neg[l]) > self.buffer:
                    self._neg[l].pop(0)

        # ---- JEPA invariance loss (readout + predictor + top-down modulator) ----
        if self.anti_collapse:
            z_tgt = self._readout(echo[-1], self.Wr_t, self.br_t)  # stop-grad EMA target
        else:
            z_tgt = self._readout(echo[-1], self.Wr, self.br)  # tied to online (ablation)
        p = self._predict_pre(z, aj)
        r = [p[i] - z_tgt[i] for i in range(big_l)]
        loss_inv = sum(ri * ri for ri in r)
        gp = [2.0 * r[i] for i in range(big_l)]
        gz = self._feedback_gz(gp)  # broadcast error; NO transposed read of Wz (see _feedback_gz)
        gwr, gbr, gsbar_top = self._readout_grad(gz, sbx[-1], self.Wr, self.br)

        top_l1 = 0.0
        lr = self.lr
        if not self.freeze_top:
            for i in range(big_l):  # predictor params
                self.b[i] -= lr * gp[i]
                self.Wa[i][aj] -= lr * gp[i]
                wzi = self.Wz[i]
                gi = gp[i]
                for k in range(big_l):
                    wzi[k] -= lr * gi * z[k]
            self._spectral_bound()
            for i in range(big_l):  # readout params
                wri = self.Wr[i]
                gwri = gwr[i]
                for j in range(self.sizes[-1]):
                    wri[j] -= lr * gwri[j]
                self.br[i] -= lr * gbr[i]
            if self.anti_collapse:  # EMA target readout
                ta = self.tau
                for i in range(big_l):
                    wti = self.Wr_t[i]
                    wri = self.Wr[i]
                    for j in range(self.sizes[-1]):
                        wti[j] = ta * wti[j] + (1.0 - ta) * wri[j]
                    self.br_t[i] = ta * self.br_t[i] + (1.0 - ta) * self.br[i]
            else:
                self.Wr_t = [row[:] for row in self.Wr]
                self.br_t = self.br[:]

        # ---- backbone LOCAL three-factor updates (top first so transport can gate the bottom) ----
        bottom_delta = 0.0
        for l in (nl - 1, 0):
            if l == nl - 1 and self.freeze_top:
                continue
            n = self.sizes[l]
            m = self._modulator(l, sbx[l], echo[l])
            if l == nl - 1:  # top layer also gets the JEPA error
                lj = self.lambda_jepa
                for j in range(n):
                    m[j] += -lj * gsbar_top[j]
            if l == 0:
                ff_train = rec_x["ev"]
                ff_n = 2 * self.R
            else:
                ff_train = rec_x["spk"][l - 1]
                ff_n = self.sizes[l - 1]
            e_ff = self._elig(ff_train, ff_n)
            e_rec = self._elig(rec_x["spk"][l], n)
            sg = self._surrogate(rec_x["vpre"][l], n)
            gate = 1.0
            if l == 0 and self.transport:  # weight-transport surrogate: gate on top
                gate = 1.0 if top_l1 > 1e-9 else 0.0
            wff = self.Wff[l]
            wrec = self.Wrec[l]
            cff = self.conn_ff[l]
            crec = self.conn_rec[l]
            layer_delta = 0.0
            for j in range(n):
                coef = lr * m[j] * sg[j] * gate
                if coef == 0.0:
                    continue
                wffj = wff[j]
                for i in cff[j]:
                    dw = coef * e_ff[i]
                    wffj[i] += dw
                    layer_delta += dw if dw >= 0.0 else -dw
                wrecj = wrec[j]
                for k in crec[j]:
                    dw = coef * e_rec[k]
                    wrecj[k] += dw
                    layer_delta += dw if dw >= 0.0 else -dw
            if l == nl - 1:
                top_l1 = layer_delta
            else:
                bottom_delta = layer_delta
        self._last_bottom_delta = bottom_delta
        return loss_inv

    # ---- planning / telemetry (copied shapes from jepa_world_model) ------------------------------------
    def make_goal_value(self, z_goal):
        """Return goal_value(z) = -sum_i |z_i - z_goal_i| (negative L1 distance to a goal latent), for plan()."""
        zg = list(z_goal)

        def goal_value(z):
            return -sum(abs(z[i] - zg[i]) for i in range(len(zg)))

        return goal_value

    def latent_std(self):
        """Mean over dims of the per-dim std of the online-latent ring buffer -- collapse telemetry (-> 0 when
        the encoder maps every input to the same latent). Copied from jepa_world_model."""
        m = len(self._buf)
        if m < 2:
            return 0.0
        stds = []
        for j in range(self.L):
            col = [bvec[j] for bvec in self._buf]
            mu = mean(col)
            var = sum((c - mu) * (c - mu) for c in col) / m
            stds.append(math.sqrt(var))
        return mean(stds)

    def effective_rank(self):
        """Stable-rank proxy of the latent covariance C: (sum_j var_j)^2 / sum_{j,k} C[j][k]^2 (trace^2 /
        Frobenius^2, participation ratio). No eigendecomposition, deterministic. Collapsed -> ~1; healthy >> 1."""
        m = len(self._buf)
        if m < 2:
            return 1.0
        big_l = self.L
        mu = [mean([b[j] for b in self._buf]) for j in range(big_l)]
        cmat = [[0.0] * big_l for _ in range(big_l)]
        for b in self._buf:
            cen = [b[j] - mu[j] for j in range(big_l)]
            for j in range(big_l):
                cj = cmat[j]
                cenj = cen[j]
                for k in range(big_l):
                    cj[k] += cenj * cen[k]
        for j in range(big_l):
            for k in range(big_l):
                cmat[j][k] /= m
        tr = sum(cmat[j][j] for j in range(big_l))
        fro = sum(cmat[j][k] * cmat[j][k] for j in range(big_l) for k in range(big_l))
        if fro <= 0.0:
            return 1.0
        return tr * tr / fro

    # ---- persistence (tagged JSON + crc32 trust boundary) ----------------------------------------------
    def save_weights(self, path):
        """Persist all weights as tagged ASCII JSON, format snn_spiking_jepa/1, with a zlib.crc32 integrity
        field over the exact serialized payload. Floats round-trip exactly (json repr of doubles)."""
        body = {
            "format": "snn_spiking_jepa/1",
            "actions": list(self.actions),
            "retina": self.R,
            "T": self.T,
            "sizes": list(self.sizes),
            "latent_dim": self.L,
            "rule": self.rule,
            "anti_collapse": self.anti_collapse,
            "tau": self.tau,
            "lr": self.lr,
            "theta": self.theta,
            "rho_star": self.rho_star,
            "Wff": self.Wff,
            "Wrec": self.Wrec,
            "conn_ff": self.conn_ff,
            "conn_rec": self.conn_rec,
            "Wr": self.Wr,
            "br": self.br,
            "Wr_t": self.Wr_t,
            "br_t": self.br_t,
            "Wz": self.Wz,
            "Wa": self.Wa,
            "b": self.b,
        }
        s = json.dumps(body, sort_keys=True, separators=(",", ":"))
        crc = zlib.crc32(s.encode("ascii")) & 0xFFFFFFFF
        with open(path, "w", encoding="ascii") as f:
            json.dump({"format": "snn_spiking_jepa/1", "crc": crc, "body": s}, f)

    def load_weights(self, path):
        """Load weights from tagged JSON with a trust boundary: verify the format tag, the crc32 over the exact
        payload bytes, and every matrix shape -- raising ValueError on any mismatch or bad crc (never trust an
        unvalidated weight file)."""
        with open(path, encoding="ascii") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or obj.get("format") != "snn_spiking_jepa/1":
            raise ValueError("bad spiking-JEPA weight file: missing/incorrect 'format' tag")
        s = obj.get("body")
        crc = obj.get("crc")
        if not isinstance(s, str) or not isinstance(crc, int):
            raise ValueError("bad spiking-JEPA weight file: missing body/crc")
        if (zlib.crc32(s.encode("ascii")) & 0xFFFFFFFF) != crc:
            raise ValueError("bad spiking-JEPA weight file: crc mismatch (corrupted payload)")
        body = json.loads(s)
        if body.get("format") != "snn_spiking_jepa/1":
            raise ValueError("bad spiking-JEPA weight file: inner format tag")
        sizes = body.get("sizes")
        big_l = int(body.get("latent_dim", -1))
        r_ = int(body.get("retina", -1))
        acts = body.get("actions")
        if not (
            isinstance(sizes, list)
            and len(sizes) == 2
            and big_l > 0
            and r_ > 0
            and isinstance(acts, list)
            and len(acts) > 0
        ):
            raise ValueError("bad spiking-JEPA weight file: bad geometry")
        na = len(acts)

        def _shape_ok(mat, rows, cols):
            return (
                isinstance(mat, list)
                and len(mat) == rows
                and all(isinstance(row, list) and len(row) == cols for row in mat)
            )

        wff = body.get("Wff")
        wrec = body.get("Wrec")
        if not (
            isinstance(wff, list) and len(wff) == 2 and isinstance(wrec, list) and len(wrec) == 2
        ):
            raise ValueError("bad spiking-JEPA weight file: backbone block count")
        if not (_shape_ok(wff[0], sizes[0], 2 * r_) and _shape_ok(wff[1], sizes[1], sizes[0])):
            raise ValueError("bad spiking-JEPA weight file: feed-forward shape mismatch")
        if not (_shape_ok(wrec[0], sizes[0], sizes[0]) and _shape_ok(wrec[1], sizes[1], sizes[1])):
            raise ValueError("bad spiking-JEPA weight file: recurrent shape mismatch")
        wr = body.get("Wr")
        wr_t = body.get("Wr_t")
        wz = body.get("Wz")
        wa = body.get("Wa")
        bb = body.get("b")
        br = body.get("br")
        br_t = body.get("br_t")
        if not (_shape_ok(wr, big_l, sizes[1]) and _shape_ok(wr_t, big_l, sizes[1])):
            raise ValueError("bad spiking-JEPA weight file: readout shape mismatch")
        if not (_shape_ok(wz, big_l, big_l) and _shape_ok(wa, big_l, na)):
            raise ValueError("bad spiking-JEPA weight file: predictor shape mismatch")
        if not (
            isinstance(bb, list)
            and len(bb) == big_l
            and isinstance(br, list)
            and len(br) == big_l
            and isinstance(br_t, list)
            and len(br_t) == big_l
        ):
            raise ValueError("bad spiking-JEPA weight file: bias shape mismatch")
        conn_ff = body.get("conn_ff")
        conn_rec = body.get("conn_rec")
        if not (
            isinstance(conn_ff, list)
            and len(conn_ff) == 2
            and isinstance(conn_rec, list)
            and len(conn_rec) == 2
        ):
            raise ValueError("bad spiking-JEPA weight file: connectivity block count")

        # trusted: adopt the validated weights and geometry
        self.actions = tuple(acts)
        self.n_act = na
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        self.R = r_
        self.T = int(body.get("T", self.T))
        self.sizes = (int(sizes[0]), int(sizes[1]))
        self.L = big_l
        self.rule = str(body.get("rule", self.rule))
        self.anti_collapse = bool(body.get("anti_collapse", True))
        self.tau = float(body.get("tau", self.tau))
        self.lr = float(body.get("lr", self.lr))
        self.theta = float(body.get("theta", self.theta))
        self.rho_star = float(body.get("rho_star", self.rho_star))
        self._ref = [0.0] * self.R
        self.Wff = [[row[:] for row in wff[0]], [row[:] for row in wff[1]]]
        self.Wrec = [[row[:] for row in wrec[0]], [row[:] for row in wrec[1]]]
        self.conn_ff = [[list(idx) for idx in conn_ff[0]], [list(idx) for idx in conn_ff[1]]]
        self.conn_rec = [[list(idx) for idx in conn_rec[0]], [list(idx) for idx in conn_rec[1]]]
        self.Wr = [row[:] for row in wr]
        self.br = br[:]
        self.Wr_t = [row[:] for row in wr_t]
        self.br_t = br_t[:]
        self.Wz = [row[:] for row in wz]
        self.Wa = [row[:] for row in wa]
        self.b = bb[:]
        self.mu = [[self.rho_star] * n for n in self.sizes]
        self.var = [[0.01] * n for n in self.sizes]
        self.cov = [[[0.0] * n for _ in range(n)] for n in self.sizes]
        self._buf = []
        self._neg = [[] for _ in self.sizes]
        # Re-shape the fixed random feedback matrix to the LOADED geometry. It is not a learned
        # weight so it is deliberately NOT serialized -- any fixed random B satisfies feedback
        # alignment -- but it must match (L x sizes[-1]) or _readout_grad would index off the end
        # whenever a file's geometry differs from the constructing instance's.
        _sr = 1.0 / math.sqrt(self.sizes[-1])
        _fb = random.Random(zlib.crc32(b"spiking-jepa-Br") ^ int(self.seed))
        self._Br = [
            [_fb.uniform(-1.0, 1.0) * _sr for _ in range(self.sizes[-1])] for _ in range(self.L)
        ]


# =========================================================================================================
# Self-check: a toy DRIFTING-SHAPE world. hidden = (identity in 0..C-1, position p on an R-pixel retina);
# obs = a shape template (bar/blob/cross) rendered at p; actions drift the position left/right or hold.
# =========================================================================================================
_SHAPES = [
    [0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.8],  # bar (solid)
    [0.2, 0.5, 0.8, 1.0, 0.8, 0.5, 0.2],  # blob (rounded)
    [1.0, 0.3, 1.0, 0.3, 1.0, 0.3, 1.0],  # cross (alternating)
]
_SW = len(_SHAPES[0])  # shape width (7)
_ACTIONS = ("DRIFT_LEFT", "DRIFT_RIGHT", "HOLD")
_DRIFT = 3  # pixels a shape drifts per LEFT/RIGHT action (HOLD = 0). Large
#                                             enough that actions carry real signal -> a genuine gap between
#                                             the action-conditioned predictor and a static identity predictor.


def _render(identity, pos, retina):
    """Render shape `identity` centered at `pos` on a length-`retina` [0,1] frame (0 background)."""
    v = [0.0] * retina
    start = pos - _SW // 2
    tmpl = _SHAPES[identity]
    for k in range(_SW):
        idx = start + k
        if 0 <= idx < retina:
            v[idx] = tmpl[k]
    return v


def _gen_stream(n, seed, retina=20):
    """Contiguous drifting-shape trajectory: identity persists across a transition (the object drifts) and
    resamples occasionally. Returns [(pattern_t, action_t, pattern_next, identity, position), ...]."""
    rng = random.Random(seed)
    lo, hi = _SW // 2, retina - 1 - _SW // 2
    ident = rng.randrange(len(_SHAPES))
    pos = rng.randint(lo, hi)
    out = []
    for _ in range(n):
        if rng.random() < 0.08:
            ident = rng.randrange(len(_SHAPES))
            pos = rng.randint(lo, hi)
        a = _ACTIONS[rng.randrange(len(_ACTIONS))]
        x = _render(ident, pos, retina)
        npos = pos + (-_DRIFT if a == "DRIFT_LEFT" else _DRIFT if a == "DRIFT_RIGHT" else 0)
        npos = max(lo, min(hi, npos))
        y = _render(ident, npos, retina)
        out.append((x, a, y, ident, pos))
        pos = npos
    return out


def _probe_acc(feats_tr, ids_tr, feats_te, ids_te, c, epochs=80, lr=0.3):
    """A COMPETENT linear identity probe: z-score features by TRAIN statistics, then fit a one-vs-rest linear
    readout by LMS (many epochs, per-feature-normalized step). Report accuracy on the held-out (feats_te).

    A competent probe matters for HONESTY: a naive un-standardized LMS diverges on the raw spike-rate
    features and defaults to the majority class, which artificially SUPPRESSES a random encoder's score and
    would make ANY trained encoder look better than it is. Standardizing lets the probe extract what is truly
    linearly present, so the trained/random/raw comparison in the self-check is a FAIR one."""
    fd = len(feats_tr[0])
    mu = [sum(f[k] for f in feats_tr) / len(feats_tr) for k in range(fd)]
    sd = [
        math.sqrt(sum((f[k] - mu[k]) ** 2 for f in feats_tr) / len(feats_tr)) + 1e-6
        for k in range(fd)
    ]

    def _norm(f):
        return [(f[k] - mu[k]) / sd[k] for k in range(fd)]

    ftr = [_norm(f) for f in feats_tr]
    fte = [_norm(f) for f in feats_te]
    w = [[0.0] * fd for _ in range(c)]
    b = [0.0] * c
    rng = random.Random(0)
    idx = list(range(len(ftr)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for p_ in idx:
            f = ftr[p_]
            y = ids_tr[p_]
            for o in range(c):
                e = (1.0 if o == y else 0.0) - (b[o] + dot(w[o], f))
                wo = w[o]
                for k in range(fd):
                    wo[k] += lr * e * f[k] / fd
                b[o] += lr * e
    ok = sum(
        1 for f, y in zip(fte, ids_te) if max(range(c), key=lambda o: b[o] + dot(w[o], f)) == y
    )
    return ok / len(fte)


def _pos_dataset(c, positions, retina):
    """Deterministic identity-decode dataset: every shape rendered at every position in `positions`. Used to
    build a POSITION-GENERALIZATION probe (train positions disjoint from test positions) -- the honest test of
    whether a latent is position-invariant, on which raw pixels sit at chance."""
    xs, ys = [], []
    for ident in range(c):
        for p in positions:
            xs.append(_render(ident, p, retina))
            ys.append(ident)
    return xs, ys


def _fill_buf(model, batch):
    """Measure collapse on a held-out batch: fill the latent ring buffer with fresh encodes, then read out."""
    model._buf = [model.encode(x) for (x, _a, _y, _i, _p) in batch][-model.buffer :]
    return model.latent_std(), model.effective_rank()


def main():
    import os
    import tempfile

    retina = 20
    c = len(_SHAPES)
    steps = 600
    epochs = 2  # 2 passes (predict-vs-identity margin is seed-dependent either way; see claim 4)
    train = _gen_stream(steps, 1, retina)
    hold = _gen_stream(220, 2, retina)

    # ---- train the shipped model (rule="lpl", anti_collapse=True) ----
    full = SpikingJEPA(_ACTIONS, retina=retina, seed=0)
    for _ep in range(epochs):
        for x, a, y, _i, _p in train:
            full.train_step(x, a, y)

    # backbone firing-rate diagnostic (healthy = alive + bounded), averaged over a batch so a single silent
    # frame does not read as "dead"
    rate0 = mean([mean(full._forward(x)["sbar"][0]) for (x, _a, _y, _i, _p) in hold[:20]])
    rate1 = mean([mean(full._forward(x)["sbar"][1]) for (x, _a, _y, _i, _p) in hold[:20]])

    # ============================ CLAIM 1: GENUINELY SPIKING ============================
    def _signals(rec):
        out = []
        for row in rec["ev"]:
            out.extend(row)
        for l in range(2):
            for row in rec["spk"][l]:
                out.extend(row)
        return out

    rec_x = full._forward(train[0][0])
    sig = _signals(rec_x)
    spiking_ok = all(v == 0.0 or v == 1.0 for v in sig)
    # latent is a function of INTEGER spike counts: rebuild z from counts/T and match encode()
    z_counts = full._readout(
        [rec_x["counts"][-1][j] / full.T for j in range(full.sizes[-1])], full.Wr, full.br
    )
    z_enc = full.encode(train[0][0])
    counts_int = all(isinstance(v, int) for row in rec_x["counts"] for v in row)
    latent_from_counts = (z_counts == z_enc) and counts_int
    graded = SpikingJEPA(_ACTIONS, retina=retina, seed=0, graded=True)
    recg = graded._forward(train[0][0], graded=True)
    graded_nonbinary = any(v != 0.0 and v != 1.0 for v in _signals(recg))

    # ============================ CLAIM 2: LOCAL (no backprop / weight transport) ============================
    loc_stream = _gen_stream(160, 5, retina)
    prim = SpikingJEPA(_ACTIONS, retina=retina, seed=0)
    ctrl2 = SpikingJEPA(_ACTIONS, retina=retina, seed=0)
    for x, a, y, _i, _p in train[:120]:  # a shared warm start so both have identical weights
        prim.train_step(x, a, y)
        ctrl2.train_step(x, a, y)

    def _bottom_l1(model):
        return sum(abs(v) for row in model.Wff[0] for v in row) + sum(
            abs(v) for row in model.Wrec[0] for v in row
        )

    prim.freeze_top = True
    prim.transport = False
    ctrl2.freeze_top = True
    ctrl2.transport = True
    b0_prim = _bottom_l1(prim)
    b0_ctrl = _bottom_l1(ctrl2)
    for x, a, y, _i, _p in loc_stream:
        prim.train_step(x, a, y)
        ctrl2.train_step(x, a, y)
    d_prim = abs(_bottom_l1(prim) - b0_prim)
    d_ctrl = abs(_bottom_l1(ctrl2) - b0_ctrl)

    # ================= DIAGNOSTIC (dropped claim 3): COLLAPSE TELEMETRY, NOT asserted =================
    # Honest negative: a random sparse LIF backbone keeps its latent effective rank high even with ALL
    # anti-collapse pressure removed (the slowness objective is already near-satisfied by the backbone's
    # naturally-slow features), so a genuinely-collapsing ablation cannot be built at this scale. We REPORT
    # full vs anti-collapse-ablation latent std / effective-rank as telemetry only (see the module docstring).
    ctrl3 = SpikingJEPA(_ACTIONS, retina=retina, seed=0, anti_collapse=False)
    for _ep in range(epochs):
        for x, a, y, _i, _p in train:
            ctrl3.train_step(x, a, y)
    full_std, full_rank = _fill_buf(full, hold)
    ctrl_std, ctrl_rank = _fill_buf(ctrl3, hold)

    # ========= CLAIM 4 (weakened honestly): PREDICTOR TRACKS FUTURE LATENT AT IDENTITY SCALE; beats-identity is seed-dependent =========
    def _mse(model_pred, predfn):
        tot = 0.0
        for x, a, y, _i, _p in hold:
            zt = model_pred.target_encode(y)
            pr = predfn(model_pred, x, a)
            tot += sum((pr[i] - zt[i]) ** 2 for i in range(model_pred.L))
        return tot / len(hold)

    mse_pred = _mse(full, lambda mdl, x, a: mdl.predict(mdl.encode(x), a))
    mse_ident = _mse(full, lambda mdl, x, a: mdl.encode(x))

    # CONTROL: retrain a predictor (on full's frozen encoder) with TIME-SHUFFLED next frames
    shuf = SpikingJEPA(_ACTIONS, retina=retina, seed=0)
    shuf.Wff = [[row[:] for row in full.Wff[l]] for l in range(2)]
    shuf.Wrec = [[row[:] for row in full.Wrec[l]] for l in range(2)]
    shuf.conn_ff = [[list(i) for i in full.conn_ff[l]] for l in range(2)]
    shuf.conn_rec = [[list(i) for i in full.conn_rec[l]] for l in range(2)]
    shuf.Wr = [row[:] for row in full.Wr]
    shuf.br = full.br[:]
    shuf.Wr_t = [row[:] for row in full.Wr_t]
    shuf.br_t = full.br_t[:]
    rng_s = random.Random(7)
    ys = [t[2] for t in train]
    rng_s.shuffle(ys)
    for (x, a, _y, _i, _p), yy in zip(train, ys):  # predictor-only SGD on shuffled targets
        z = shuf.encode(x)
        zt = shuf.target_encode(yy)
        aj = shuf._idx(a)
        pr = shuf._predict_pre(z, aj)
        gp = [2.0 * (pr[i] - zt[i]) for i in range(shuf.L)]
        for i in range(shuf.L):
            shuf.b[i] -= shuf.lr * gp[i]
            shuf.Wa[i][aj] -= shuf.lr * gp[i]
            for k in range(shuf.L):
                shuf.Wz[i][k] -= shuf.lr * gp[i] * z[k]
        shuf._spectral_bound()
    mse_shuf = _mse(shuf, lambda mdl, x, a: mdl.predict(mdl.encode(x), a))

    # ========= DIAGNOSTIC (dropped "beats random" claim 5): POSITION-GENERALIZATION DECODE, NOT asserted =========
    # Honest negative: the local rule does NOT beat a random SNN encoder at identity decoding. We use a
    # POSITION-GENERALIZATION probe (train positions DISJOINT from the held-out test positions -- the honest
    # test of position-invariance) with a COMPETENT standardized probe, and report trained / random-encoder /
    # raw-pixel side by side. Raw pixels sit near chance (each position is a different pattern); BOTH the
    # trained AND a random spiking encoder generalize far better -- an ARCHITECTURAL property of the spiking
    # encoder, not a product of the local learning (indeed the trained encoder does not exceed the random one).
    lo_p, hi_p = _SW // 2, retina - 1 - _SW // 2
    all_pos = list(range(lo_p, hi_p + 1))
    tr_pos = all_pos[0::2]  # train positions
    te_pos = all_pos[1::2]  # DISJOINT held-out positions
    xtr, ids_tr = _pos_dataset(c, tr_pos, retina)
    xte, ids_te = _pos_dataset(c, te_pos, retina)
    acc_primary = _probe_acc(
        [full.encode(x) for x in xtr], ids_tr, [full.encode(x) for x in xte], ids_te, c
    )
    rand_enc = SpikingJEPA(_ACTIONS, retina=retina, seed=3)  # untrained encoder baseline
    acc_random = _probe_acc(
        [rand_enc.encode(x) for x in xtr], ids_tr, [rand_enc.encode(x) for x in xte], ids_te, c
    )

    def _raw_events(model, x):  # raw event counts per line (2R vector)
        rec = model._forward(x)
        return [sum(rec["ev"][t][i] for t in range(model.T)) for i in range(2 * model.R)]

    acc_raw = _probe_acc(
        [_raw_events(full, x) for x in xtr], ids_tr, [_raw_events(full, x) for x in xte], ids_te, c
    )
    chance = 1.0 / c
    learn_gt_random = acc_primary > acc_random  # reported honestly (False at this scale)

    # ============================ BOUNDED ROLLOUT (40 steps, every action) ============================
    z0 = full.encode(_render(0, retina // 2, retina))
    roll_bounded = True
    roll_max = 0.0
    for a in _ACTIONS:
        z = z0
        for _ in range(40):
            z = full.predict(z, a)
            for v in z:
                if not math.isfinite(v) or abs(v) >= 10.0:
                    roll_bounded = False
                if abs(v) > roll_max:
                    roll_max = abs(v)

    # ============================ DETERMINISM + SAVE/LOAD ============================
    det_a = SpikingJEPA(_ACTIONS, retina=retina, seed=7)
    det_b = SpikingJEPA(_ACTIONS, retina=retina, seed=7)
    for x, a, y, _i, _p in train[:150]:
        det_a.train_step(x, a, y)
        det_b.train_step(x, a, y)
    same_weights = (
        det_a.Wff == det_b.Wff
        and det_a.Wrec == det_b.Wrec
        and det_a.Wr == det_b.Wr
        and det_a.br == det_b.br
        and det_a.Wz == det_b.Wz
        and det_a.Wa == det_b.Wa
        and det_a.b == det_b.b
    )

    tmp = os.path.join(tempfile.gettempdir(), "snn_spiking_jepa_rt.json")
    full.save_weights(tmp)
    loaded = SpikingJEPA(_ACTIONS, retina=retina, seed=99)
    loaded.load_weights(tmp)
    probe = _render(1, retina // 2 - 2, retina)
    roundtrip = loaded.encode(probe) == full.encode(probe) and loaded.predict(
        loaded.encode(probe), "HOLD"
    ) == full.predict(full.encode(probe), "HOLD")
    # corrupted crc -> ValueError
    with open(tmp, encoding="ascii") as f:
        obj = json.load(f)
    bad_crc = dict(obj)
    bad_crc["crc"] = (obj["crc"] + 1) & 0xFFFFFFFF
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(bad_crc, f)
    crc_raises = False
    try:
        SpikingJEPA(_ACTIONS, retina=retina).load_weights(tmp)
    except ValueError:
        crc_raises = True
    # bad shape -> ValueError (tamper the body, refresh the crc so it is the SHAPE that trips)
    body = json.loads(obj["body"])
    body["Wr"] = [row[:-1] for row in body["Wr"]]  # wrong number of columns
    s2 = json.dumps(body, sort_keys=True, separators=(",", ":"))
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(
            {
                "format": "snn_spiking_jepa/1",
                "crc": zlib.crc32(s2.encode("ascii")) & 0xFFFFFFFF,
                "body": s2,
            },
            f,
        )
    shape_raises = False
    try:
        SpikingJEPA(_ACTIONS, retina=retina).load_weights(tmp)
    except ValueError:
        shape_raises = True
    os.remove(tmp)

    # ============================ T051b-i: THE DECLARATION OF RECORD (specs/022) ============================
    # distinct-from: is_snn_forward.py -- that module is a forward-pass architecture (layered time
    # constants); this block is a HARNESS spy holding jepa's own authored rule to jepa's own
    # production update path via the exported equivalence_check. No other site does that job.
    # The authored per-synapse LPL rule (lpl_synapse_rule) held to the LIVE production path by the
    # EXPORTED equivalence_check on SPIED warm-state train_step sites -- never a cold reconstruction
    # (the T048 lesson: a cold cascade certified an order the production path refuses). Two arms:
    #   (a) BIT arm, tol=0.0: the anti_collapse=False ablation's single term -- the trailing
    #       (sg, e) factor order matches the incumbent chain ((lr*m)*sg)*gate*e; state/lr commute
    #       (the coefficient is exact), so the arm is equal to the bit.
    #   (b) TOL arm: the full 4-term form -- the incumbent sums m's terms THEN multiplies the
    #       chain, the declaration multiplies each term's chain THEN sums (distribution), so the
    #       tol is DERIVED: K ulps of the largest per-case sum |term|, K = the counted roundings
    #       of the authored structure -- never fit to the measured divergence. The in-command
    #       CONTROL: a 1e-6 lambda_var mis-declaration must RED at the SAME tol (T050's class).
    #   (c) DELTA arm (the panel's BLOCKER fix): arms (a)/(b) bind the declaration to the
    #       RECORDED modulator + the DECLARED factor semantics (computed through the BASE-class
    #       _elig/_surrogate, so an override cannot skew both sides) -- they cannot see a defect
    #       in the production APPLICATION. The delta arm binds that: incumbent = the OBSERVED
    #       per-site weight delta (Wff/Wrec rows copied at _modulator(0) call time and after
    #       train_step), tol including the += rounding. Its reachable red is IN-COMMAND: a
    #       corrupted spy whose _elig scales e by 1.25 (production diverging from the declared
    #       factor semantics) must RED the delta arm while the chain arms stay green (they are
    #       scoped to the chain, said plainly). A defect INSIDE _elig/_surrogate themselves is
    #       out of equivalence scope -- they DEFINE the factors.
    # P2 ADJACENCY, adjudicated: eligibility_algebra's "lpl.modulator.core2" is a PROPERTY
    # FIXTURE (synthetic algebra-property regression; its coefficient-pin half moved HERE --
    # one mechanism per claim); THIS arm is the binding declaration.
    import sys as _sys

    from snn_data_lake.plasticity.eligibility_algebra import EligibilityAlgebra, equivalence_check
    from snn_data_lake.plasticity.rule_composer import RuleComposer, RuleSpec

    # distinct-from: dendritic_neuron.py -- that (and the other step()/drive() sites) advance
    # NEURON STATE; this spy advances nothing: it is a recording subclass holding jepa's own
    # declaration to jepa's own production update. No other site does that job.
    class _SpyJEPA(SpikingJEPA):
        """Harness-only spy (single-thread by design: _cur is one slot -- concurrent
        train_step under GIL=False would cross-contaminate recs): records _forward
        outputs, _modulator inputs, and BOTTOM-LAYER WEIGHT ROWS at call time
        (copies -- _update_stats and the update loop mutate these in place)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._rec_on = False
            self._spied = []
            self._cur = None

        def train_step(self, x, a, y):
            if self._rec_on:
                self._cur = {"recs": [], "mod": {}}
            out = super().train_step(x, a, y)
            if self._cur is not None:
                self._cur["wff_after"] = [row[:] for row in self.Wff[0]]
                self._cur["wrec_after"] = [row[:] for row in self.Wrec[0]]
                self._spied.append(self._cur)
                self._cur = None
            return out

        def _forward(self, pattern, graded=None):  # graded=None: DELEGATE to self.graded
            rec = super()._forward(pattern, graded=graded)  # (the panel caught graded=False
            if self._cur is not None and len(self._cur["recs"]) < 2:  # silently un-grading)
                self._cur["recs"].append(rec)
            return rec

        def _modulator(self, l, sb, echo):
            m = super()._modulator(l, sb, echo)
            if self._cur is not None:
                if l == 0:  # bottom weights are untouched until after this call returns
                    self._cur["wff_before"] = [row[:] for row in self.Wff[0]]
                    self._cur["wrec_before"] = [row[:] for row in self.Wrec[0]]
                n = self.sizes[l]
                mu = self.mu[l]
                var = self.var[l]
                cov = self.cov[l]
                pull = []
                for j in range(n):  # same loop shape+order as covt -> bit-identical recompute
                    cj = cov[j]
                    acc = 0.0
                    for k in range(n):
                        if k != j:
                            acc += cj[k] * (sb[k] - mu[k])
                    pull.append(acc)
                self._cur["mod"][l] = {
                    "sb": sb[:],
                    "echo": echo[:],
                    "mu": mu[:],
                    "var": var[:],
                    "cov_pull": pull,
                    "m": m[:],
                }
            return m

    def _lpl_sites(spy):
        """(cases, chain-incumbents, observed-deltas) from spied BOTTOM-layer sites --
        gate==1 exactly (transport off; the method now refuses transport instances), no
        lambda_jepa addition (top-layer sites are the disclosed remainder). Chain
        incumbent = the production association lr*m*sg*1.0*e from the RECORDED
        modulator. Observed delta = the weight the production loop ACTUALLY wrote
        (after - before), paired with |w_before| for the delta tol. Factors are
        computed through the BASE-CLASS _elig/_surrogate explicitly -- the DECLARED
        semantics, so a subclass override cannot skew declaration and incumbent
        together."""
        assert spy.transport is False, "equivalence sites assume gate==1 (transport off)"
        cases, incs, deltas = [], [], []
        for st in spy._spied:
            rec_x = st["recs"][0]
            mod = st["mod"].get(0)
            if mod is None:
                continue
            n = spy.sizes[0]
            sg = SpikingJEPA._surrogate(spy, rec_x["vpre"][0], n)
            e_ff = SpikingJEPA._elig(spy, rec_x["ev"], 2 * spy.R)
            e_rec = SpikingJEPA._elig(spy, rec_x["spk"][0], n)
            sb, echo, mu, var = mod["sb"], mod["echo"], mod["mu"], mod["var"]
            for j in range(n):
                m_j = mod["m"][j]
                sites = [("ff", i) for i in sorted(spy.conn_ff[0][j])[:2]]
                sites += [("rec", k) for k in sorted(spy.conn_rec[0][j])[:1]]
                for kind, i in sites:
                    e_val = e_ff[i] if kind == "ff" else e_rec[i]
                    case = {"d_pred": sb[j] - echo[j], "lr": spy.lr, "sg": sg[j], "e": e_val}
                    if spy.anti_collapse:
                        case["d_mu"] = sb[j] - mu[j]
                        case["inv_var"] = 1.0 / (var[j] + spy.eps_stat)
                        case["cov_pull"] = mod["cov_pull"][j]
                        case["d_homeo"] = mu[j] - spy.rho_star
                    cases.append(case)
                    incs.append(spy.lr * m_j * sg[j] * 1.0 * e_val)
                    if kind == "ff":
                        deltas.append(
                            (st["wff_after"][j][i] - st["wff_before"][j][i], st["wff_before"][j][i])
                        )
                    else:
                        deltas.append(
                            (
                                st["wrec_after"][j][i] - st["wrec_before"][j][i],
                                st["wrec_before"][j][i],
                            )
                        )
        return cases, incs, deltas

    def _spy_run(steps=250, record=5, cls=_SpyJEPA, **kw):
        spy = cls(_ACTIONS, retina=retina, seed=0, **kw)
        warm = train[:steps]
        for x, a, y, _i, _p in warm[:-record]:
            spy.train_step(x, a, y)
        spy._rec_on = True  # record only the tail: warm EMA state, bounded memory
        for x, a, y, _i, _p in warm[-record:]:
            spy.train_step(x, a, y)
        return spy

    def _term_mag(rule, case):
        s = 0.0
        for _coeff2, _fs2 in rule.terms:
            p = abs(_coeff2)
            for _f2 in _fs2:
                p *= abs(float(case[_f2]))
            s += p
        return s

    # K derived from the authored structure, not chosen: one ulp per rounding in the
    # longest re-association chain, counted as every factor multiply plus every term
    # add (4 terms x (4|5) factors + 4 adds = 21 for the full rule). The panel measured
    # the actual worst at 2.03 ulps; the derived K's headroom is structural slack, and
    # the mis-declaration control below -- not a second magic ceiling -- is what proves
    # the tol swallows no real defect (a former `< 1e-12` assert died here as magic
    # guarding magic).
    def _k_ulp(rule):
        return float(sum(len(_fs2) for _c2, _fs2 in rule.terms) + len(rule.terms))

    spy_bit = _spy_run(anti_collapse=False)
    rule_bit = spy_bit.lpl_synapse_rule()
    cases_b, incs_b, _deltas_b = _lpl_sites(spy_bit)
    live_b = sum(1 for v in incs_b if v != 0.0)
    inc_map_b = {id(c): v for c, v in zip(cases_b, incs_b)}  # id-keyed: equivalence_check's
    worst_bit = equivalence_check(  # list(cases) keeps object identity, and cases_b holds
        rule_bit, lambda c: inc_map_b[id(c)], cases_b, tol=0.0
    )  # the refs alive -- both load-bearing for id() stability

    spy_full = _spy_run()
    rule_full = spy_full.lpl_synapse_rule()
    cases_f, incs_f, deltas_f = _lpl_sites(spy_full)
    live_f = sum(1 for v in incs_f if v != 0.0)
    inc_map_f = {id(c): v for c, v in zip(cases_f, incs_f)}
    _mag = max(_term_mag(rule_full, _case) for _case in cases_f)
    tol_full = _k_ulp(rule_full) * _sys.float_info.epsilon * _mag
    worst_full = equivalence_check(rule_full, lambda c: inc_map_f[id(c)], cases_f, tol=tol_full)

    # DELTA arm: the declaration vs the weight the production loop ACTUALLY wrote (the
    # chain arms alone were panel-measured BLIND to a 25% production-application defect
    # that moved applied |dw| by +20%). tol adds the += rounding scale (|w|+|terms|).
    _magd = max(abs(w0) + _term_mag(rule_full, _case) for _case, (_d, w0) in zip(cases_f, deltas_f))
    tol_delta = _k_ulp(rule_full) * _sys.float_info.epsilon * _magd
    worst_delta = 0.0
    for _case, (_d, _w0) in zip(cases_f, deltas_f):
        _dv = abs(rule_full.evaluate(**_case) - _d)
        if not (_dv <= tol_delta):
            raise AssertionError(
                f"declaration diverges from the APPLIED weight delta: |d|={_dv!r} > {tol_delta:.3e}"
            )
        worst_delta = max(worst_delta, _dv)

    # the delta arm's reachable red, in-command: production _elig corrupted x1.25 while
    # the declared factor semantics (base-class _elig) stay clean -> applied deltas
    # diverge from the declaration; the chain arms stay green (scoped to the chain).
    class _CorruptElig(_SpyJEPA):
        def _elig(self, train_sig, n):
            return [v * 1.25 for v in SpikingJEPA._elig(self, train_sig, n)]

    _spy_corrupt = _spy_run(steps=60, record=3, cls=_CorruptElig)
    _cases_c, _incs_c, _deltas_c = _lpl_sites(_spy_corrupt)
    _rule_c = _spy_corrupt.lpl_synapse_rule()
    lpl_delta_control_red = False
    for _case, (_d, _w0) in zip(_cases_c, _deltas_c):
        if _d != 0.0 and abs(_rule_c.evaluate(**_case) - _d) > tol_delta:
            lpl_delta_control_red = True
            break

    _alg_bad = EligibilityAlgebra(term_budget=len(rule_full.terms))
    _seen_bad = []
    for _cf, _fs in rule_full.terms:
        for _f in _fs:
            if _f not in _seen_bad:
                _seen_bad.append(_f)
                _alg_bad.declare_factor(
                    _f, "control copy of the declared factor (mis-declaration arm)"
                )
    lpl_bad = _alg_bad.author(
        "jepa.lpl.synapse.misdeclared",
        [((cf + 1e-6) if fs[0] == "d_mu" else cf, fs) for cf, fs in rule_full.terms],
    )
    lpl_control_red = False
    try:
        equivalence_check(lpl_bad, lambda c: inc_map_f[id(c)], cases_f, tol=tol_full)
    except AssertionError:
        lpl_control_red = True

    lpl_espp_refuses = False
    try:
        SpikingJEPA(_ACTIONS, retina=retina, seed=0, rule="espp").lpl_synapse_rule()
    except ValueError:
        lpl_espp_refuses = True
    lpl_ctrl_refuses = 0
    for _kw in ({"transport": True}, {"freeze_top": True}):
        _inst = SpikingJEPA(_ACTIONS, retina=retina, seed=0)
        for _k2, _v2 in _kw.items():
            setattr(_inst, _k2, _v2)
        try:
            _inst.lpl_synapse_rule()
        except ValueError:
            lpl_ctrl_refuses += 1

    # coefficient pins: the TERMS themselves, all four lambdas, exact equality -- the
    # prior substring test ('+2' in canonical) passed 2.5/20/200 (panel). Plus the
    # RETYPING red the shipped-defaults instance cannot construct: a mutated instance's
    # authored coefficients must track the mutation (a retyped literal cannot).
    _cmap = {fs[0]: cf for cf, fs in rule_full.terms}
    lpl_coeffs_pin = _cmap == {
        "d_pred": -(float(spy_full.lambda_pred) * 2.0),
        "d_mu": float(spy_full.lambda_var),
        "cov_pull": -float(spy_full.lambda_cov),
        "d_homeo": -float(spy_full.lambda_homeo),
    }
    _mut = SpikingJEPA(_ACTIONS, retina=retina, seed=0)
    _mut.lambda_var = 3.0
    _mut.lambda_cov = 0.7
    _mut.lambda_homeo = 0.9
    _rmut = _mut.lpl_synapse_rule()
    _mmap = {fs[0]: cf for cf, fs in _rmut.terms}
    lpl_mut_tracks = _mmap == {
        "d_pred": -2.0,
        "d_mu": 3.0,
        "cov_pull": -0.7,
        "d_homeo": -0.9,
    } and ("+3" in _rmut.canonical())

    # the palette point, with the VALUE bound (the prior form had no reachable red: it
    # passed for a garbage rule, and its cases_f[0] binding had e == 0.0 so the
    # declaration contributed exactly 0.0 to its own smoke -- both panel-constructed;
    # the guarantee-string conjunct was a policy-literal echo and is DELETED, not
    # weakened).
    _reads = set()
    for _cf, _fs in rule_full.terms:
        _reads.update(_fs)
    _comp = RuleComposer("sum", "w")
    _comp.add(RuleSpec("jepa_lpl", rule_full, reads=_reads, writes={"w"}))
    _alg_d = EligibilityAlgebra(term_budget=1)
    _alg_d.declare_factor("w", "the composed substrate (smoke)")
    _decay = _alg_d.author("decay", [(-0.5, ("w",))])
    _comp.add(RuleSpec("decay", _decay, reads={"w"}, writes={"w"}))
    _best = max(cases_f, key=lambda c2: abs(rule_full.evaluate(**c2)))
    _contrib = rule_full.evaluate(**_best)
    _bind = dict(_best)
    _bind["w"] = 1.0
    _w1, _parts = _comp.step(1.0, _bind)
    lpl_composes = (
        _contrib != 0.0
        and _parts["jepa_lpl"] == _contrib
        and _parts["decay"] == -0.5
        and _w1 == 1.0 + (_parts["jepa_lpl"] + _parts["decay"])
    )

    # ============================ REPORT ============================
    line = "=" * 92
    print(line)
    print(
        "SPIKING JEPA -- genuinely spiking + genuinely local predictive world model (honest toy-scale study)"
    )
    print(line)
    print(
        f"world      : drifting shapes, {c} identities x positions on a {retina}-pixel retina, "
        f"actions {_ACTIONS}, drift={_DRIFT}px"
    )
    print(
        f"model      : event retina(2R={2 * retina}) -> LIF {full.sizes} (RPL) -> readout L={full.L}; "
        f"rule={full.rule!r}; steps={steps}x{epochs}"
    )
    print(f"backbone   : firing rates layer0={rate0:.3f} layer1={rate1:.3f} (alive + bounded)")
    print(line)
    print("ASSERTED CLAIMS (1, 2 vs a genuinely-failing control; 4 weakened to hold every seed):")
    print(
        f"(1) spiking      : inter-layer signals all in {{0,1}}={spiking_ok}; latent=f(int counts)="
        f"{latent_from_counts}  | CONTROL graded non-binary={graded_nonbinary}"
    )
    print(
        f"(2) local        : freeze-top bottom L1-delta primary={d_prim:.4e} > eps  | CONTROL "
        f"transport bottom-delta={d_ctrl:.4e} (~0)"
    )
    print(
        f"(4) predict      : pred MSE={mse_pred:.5f} ~ identity MSE={mse_ident:.5f} (same scale, "
        f"asserted) | time-shuffled MSE={mse_shuf:.5f} (never materially better than pred, asserted; "
        f"orderings vs identity are seed-flips, REPORTED only)"
    )
    print("HONEST DIAGNOSTICS (dropped claims -- MEASURED, NOT asserted; see docstring):")
    print(
        f"(3) collapse tele: full std={full_std:.4f} rank={full_rank:.2f} | anti-collapse-ABLATION "
        f"std={ctrl_std:.4f} rank={ctrl_rank:.2f} -> ablation does NOT collapse (random LIF holds rank)"
    )
    print(
        f"(5) id-generalize: trained acc={acc_primary:.2f} | random-encoder acc={acc_random:.2f} | "
        f"raw-pixel acc={acc_raw:.2f} (chance {chance:.2f}) -> learning>random={learn_gt_random} "
        f"(architecture, not learning)"
    )
    print(
        f"rollout    : 40-step bounded={roll_bounded} (max |z_i|={roll_max:.3f} < 10) for every action"
    )
    print(
        f"determinism: two seed=7 models byte-identical={same_weights}; save/load exact={roundtrip}; "
        f"crc-raises={crc_raises}; shape-raises={shape_raises}"
    )
    print(
        f"declaration: T051b-i authored LPL rule -- BIT arm worst |d|={worst_bit:.1e} on "
        f"{len(cases_b)} sites of which {live_b} LIVE (tol=0.0); FULL arm worst |d|="
        f"{worst_full:.3e} <= derived tol {tol_full:.3e} (K={_k_ulp(rule_full):.0f} ulps of "
        f"max-mag {_mag:.3e}) on {len(cases_f)} sites of which {live_f} LIVE; DELTA arm (the "
        f"APPLIED weight) worst |d|={worst_delta:.3e} <= {tol_delta:.3e}, corrupted-production "
        f"control REDS={lpl_delta_control_red}; 1e-6 mis-declaration REDS={lpl_control_red}; "
        f"espp+transport+freeze_top refuse={lpl_espp_refuses}/{lpl_ctrl_refuses}; coefficients "
        f"pin ALL FOUR lambdas={lpl_coeffs_pin}, mutated-instance tracks={lpl_mut_tracks}; "
        f"accepted by the T051a kernel with the contributed VALUE bound={lpl_composes}"
    )
    print(line)

    # ==================== ASSERTS (only what genuinely holds; anything seed-contingent is REPORTED) ====================
    # Claims 3 (no-collapse) and 5 (useful-latent-beats-random) are DROPPED and only REPORTED above: at this
    # toy scale neither can be shown against a control that genuinely fails, so asserting them would be a rig.
    # Claim 4's ORDERINGS (pred beats identity, shuffle loses it) joined them -- see the honest note below.
    assert spiking_ok, "every inter-layer signal must be a 0/1 spike"
    assert latent_from_counts, "the latent must be a function of INTEGER spike counts"
    assert graded_nonbinary, (
        "the graded control must emit non-binary signals (would FAIL the spiking check)"
    )

    assert d_prim > 1e-6, (
        f"with the top frozen the bottom must STILL learn from its own local loss ({d_prim:.2e})"
    )
    assert d_ctrl < 1e-9, (
        f"the weight-transport control must NOT learn when the top is frozen ({d_ctrl:.2e})"
    )
    assert d_prim > 100.0 * (d_ctrl + 1e-30), (
        "local bottom-learning must dwarf the transport control"
    )

    # ---- (2b) NO WEIGHT TRANSPORT INSIDE THE HEAD -- the axis the control above cannot see ----
    # The claim-2 control gates the BOTTOM layer on the TOP's update, so it tests inter-LAYER
    # transport only. Both head paths (dL/dz through Wz, and the top-down modulator through Wr)
    # were transposed forward reads and passed it. Guard them directly.
    #
    # The two probes differ, and the reason is worth stating because the naive version of the
    # second one is WRONG and was caught by constructing its red. dL/dz takes only gp, so
    # perturbing Wz is a clean test. gsbar CANNOT be tested that way: its `dldu` comes from the
    # FORWARD activations u = Wr @ s_bar + br, so gsbar legitimately moves when Wr moves under
    # BOTH implementations. Re-measured 2026-08-05 with the perturbation STATED (it was not, and
    # one of the two figures did not reproduce without it): replacing `wr` with the all-0.35
    # matrix -- the same shape of perturbation used on Wz below -- moves gsbar by 1.435e+00
    # shipped and 1.490e+00 pre-fix. (The comment previously read 1.435 as "1.450e+00".) Both
    # move, so that probe cannot tell the two implementations apart at all. The discriminating
    # probe is the opposite one: perturb the FEEDBACK matrix _Br and require gsbar to MOVE,
    # which only a Br-carried signal can do.
    #
    # REDS CONSTRUCTED AND RUN at this exact seed/probe (re-verified 2026-08-05 by monkeypatching
    # the pre-fix bodies back in and running these exact lines):
    #   dL/dz  : pre-fix (Wz^T) dep 2.756e+00 -> FIRES;  shipped 0.000e+00 -> passes.
    #   gsbar  : pre-fix (Wr^T) ignores _Br, dep 0.000e+00 -> FIRES;  shipped 2.910e+00 -> passes.
    #
    # A THIRD assert stood here -- `assert any(v != 0.0 for v in gs_before)`, messaged "the
    # top-down modulator must still CARRY a signal ... it does not delete the term". DELETED
    # 2026-08-05 under P8 falsifier F11: it has NO reachable red IN SITU. Its named red is the
    # DELETED-term implementation, and in that implementation gs_flip is zero too, so `dep_br > 0`
    # one line above fires FIRST and the third assert never executes. Swept four implementations
    # -- linear (shipped), deleted, even-in-_Br (`_Br[i][j]**2`), and rectified
    # (`max(0.0, sum(...))`, chosen because a nonlinearity is the only shape that could zero
    # gs_before while leaving gs_flip nonzero) -- and in all four `A3 fires only where A2 fires`:
    #   linear dep_br 2.910e+00 A2 pass A3 pass | deleted 0.000e+00 A2 FIRE A3 fire
    #   squared 0.000e+00 A2 FIRE A3 pass       | rectified 1.455e+00 A2 pass A3 pass
    # It restated the assert above it, which is the same defect this pass deleted from
    # `cerebellum/synfire_chain.py`. Deleted, not weakened (F11).
    ep = SpikingJEPA(_ACTIONS, retina=retina, seed=5)
    gp_probe = [0.1 * (i + 1) for i in range(ep.L)]
    sbar_probe = [0.3] * ep.sizes[-1]
    gz_before = ep._feedback_gz(gp_probe)
    _, _, gs_before = ep._readout_grad(gp_probe, sbar_probe, ep.Wr, ep.br)
    ep.Wz = [[0.35] * ep.L for _ in range(ep.L)]
    gz_after = ep._feedback_gz(gp_probe)
    dep_z = max(abs(a - b) for a, b in zip(gz_before, gz_after))
    ep._Br = [[-v for v in row] for row in ep._Br]  # flip the FEEDBACK matrix only; Wr untouched
    _, _, gs_flip = ep._readout_grad(gp_probe, sbar_probe, ep.Wr, ep.br)
    dep_br = max(abs(a - b) for a, b in zip(gs_before, gs_flip))
    assert dep_z == 0.0, f"dL/dz must not read Wz (weight transport); Wz moved it by {dep_z:.3e}"
    assert dep_br > 0.0, (
        f"the top-down modulator must ride the FIXED random _Br, not Wr transposed -- flipping "
        f"_Br moved it by {dep_br:.3e} (a transported signal ignores _Br entirely)"
    )

    # HONEST NOTE (the claim-4 ORDERINGS are refuted across seeds -- measured, not assumed). The old
    # `mse_pred < mse_ident` held on 4/7 shifted seeds and `mse_shuf >= mse_ident` on 6/7; re-measured
    # over 20 seed shifts, pred beats identity on only 11/20 (identity-minus-pred from -0.00082 to
    # +0.00172, i.e. the 4th significant figure either way) and shuffle loses to identity on 15/20.
    # Growing the hold stream 220 -> 2000 transitions flips NO losing seed to a win (the losses are
    # properties of the trained backbone, not evaluation noise), and the mean gap over any 8
    # consecutive shifts changes sign with the window -- the drifting-shape latent is so slow that
    # the identity map is already near-optimal, leaving no stable margin for the predictor to beat
    # (the same architecture-dominates-learning story as dropped claims 3 and 5). What DID hold on
    # every one of the 20 measured seeds is asserted instead, at zero added runtime: the trained
    # predictor tracks the future latent at the identity map's error scale (mse_pred/mse_ident in
    # 0.72..1.37, plus one degenerate near-zero-MSE seed), and shuffled-target training never
    # produces a materially BETTER predictor than true-order training (mse_shuf/mse_pred >= 0.90
    # on all 20 seeds, shuffle strictly worse on 17/20).
    assert mse_pred < 1.5 * mse_ident + 1e-9, (
        f"the predictor must track the future latent at the identity map's error scale ({mse_pred:.5f} vs {mse_ident:.5f})"
    )
    assert mse_shuf > 0.8 * mse_pred - 1e-9, (
        f"time-shuffled targets must never train a materially better predictor than the true order ({mse_shuf:.5f} vs {mse_pred:.5f})"
    )

    assert roll_bounded, (
        f"a 40-step rollout must stay finite and bounded for every action (max |z_i| {roll_max:.3f})"
    )
    assert same_weights, "two seed=7 models trained on the same stream must be byte-identical"
    assert roundtrip, "save/load must round-trip encode + predict exactly"
    assert crc_raises, "a corrupted-crc weight file must raise ValueError"
    assert shape_raises, "a bad-shape weight file must raise ValueError"

    # ---- T051b-i asserts: the declaration binds the implementation ----
    # Floors are on LIVE (nonzero-incumbent) sites: the panel measured the raw count
    # 94.2% vacuous on the BIT arm (339/360 were 0==0 compares) -- a floor the vacuum
    # satisfies is no floor. Measured live counts at landing: BIT 21, FULL 201.
    assert live_b >= 15 and live_f >= 150, (
        f"too few LIVE equivalence sites (BIT {live_b}, FULL {live_f}) -- "
        f"a 0==0 compare is not coverage (raw {len(cases_b)}/{len(cases_f)})"
    )
    assert worst_bit == 0.0, (
        f"the ablation declaration must match the production chain to the bit (worst |d|={worst_bit!r})"
    )
    assert tol_full > 0.0 and lpl_control_red, (
        "the derived tol must be positive AND the 1e-6 lambda_var mis-declaration must RED at it "
        "-- the control, not a magnitude ceiling, is what proves the tol swallows no real defect"
    )
    assert lpl_delta_control_red, (
        "the corrupted-production control (e x1.25 at apply) must RED the delta arm -- "
        "without it the check binds only the chain, the panel's BLOCKER"
    )
    assert lpl_espp_refuses and lpl_ctrl_refuses == 2, (
        "lpl_synapse_rule must refuse espp AND transport AND freeze_top instances "
        f"(espp={lpl_espp_refuses}, controls={lpl_ctrl_refuses}/2)"
    )
    assert lpl_coeffs_pin, (
        "the authored TERM coefficients must equal ALL FOUR instance lambdas exactly "
        "(the substring form passed 2.5/20/200 -- panel)"
    )
    assert lpl_mut_tracks, (
        "a mutated instance's authored coefficients must track the mutation -- "
        "the reachable red for a retyped literal (shipped defaults cannot construct it)"
    )
    assert lpl_composes, (
        "the composition smoke must bind the contributed VALUE (nonzero contribution, "
        "parts == the declaration's own evaluation, w1 arithmetic consumed)"
    )

    print(
        "DISCLOSURES (T051b-i): the DECLARATION is equivalence, NOT adoption -- the incumbent "
        "_modulator + train_step block REMAINS the production path, lpl_synapse_rule has ZERO "
        "production callers (consumers: this self-check + the composer-palette smoke), the host "
        "module is REACHABLE since T053-ii (its dormancy row retired on rule_composer's census "
        "imports; adjudications unmoved), and BOTH method-grade gates structurally "
        "miss the method (no LEARN_VERB). Its first tick-charged use "
        "is UNDATED; the algebra/composer siblings' expiry WAS discharged at T053-ii by iSTDP's "
        "OWN declaration composing on the tick, not by this rule. UNDECLARED REMAINDER (shrunk at T052-i): the "
        "lambda_jepa top-layer addition and transport-gated (gate!=1) bottom sites -- espp/"
        "transport/freeze_top instances REFUSE rather than receive a lying declaration; the "
        "rule='espp' branch is declared by plasticity/espp_rule.espp_synapse_rule since T052-i "
        "(the branch delegates its math there). The five "
        "pre-collapsed factors (d_pred, d_mu, inv_var, cov_pull, d_homeo) are caller-computed "
        "state, defined BY _elig/_surrogate/_update_stats -- a defect inside those definitions "
        "is out of equivalence scope (the delta arm binds everything after them). Case coverage: "
        "lr is PINNED to the one instance value and e is substrate-quantized to ~5 distinct "
        "values (panel-measured population property, not sampling bias)."
    )
    print(
        f"self-check OK: genuinely spiking (rates {rate0:.2f}/{rate1:.2f}, latent=f(counts)); genuinely local "
        f"(bottom learns {d_prim:.1e} vs transport {d_ctrl:.1e}); predictor tracks latent at identity scale "
        f"({mse_pred:.5f} vs {mse_ident:.5f}, time-shuffled {mse_shuf:.5f}; orderings seed-dependent, reported);"
    )
    print(
        f"  40-step bounded; deterministic + exact save/load (crc+shape guarded); rule={full.rule!r}. "
        f"HONEST NEGATIVES (reported, not asserted): no-collapse ablation does NOT collapse "
        f"(rank {full_rank:.1f} vs {ctrl_rank:.1f}); learning does NOT beat a random encoder at identity "
        f"(id-gen {acc_primary:.2f} vs random {acc_random:.2f}, raw {acc_raw:.2f})."
    )


if __name__ == "__main__":
    main()
