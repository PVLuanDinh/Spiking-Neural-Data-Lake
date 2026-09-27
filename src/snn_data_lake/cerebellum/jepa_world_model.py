r"""
Latent world model -- a MINIMAL DETERMINISTIC LINEAR JEPA (Joint-Embedding Predictive Architecture).

Honest framing (read this first). This is a *minimal* JEPA: a SINGLE linear encoder z = unit(tanh(We@x + be)),
a LINEAR action-conditioned residual predictor z' = z + Wz@z + Wa@onehot(a) + b, an EMA / stop-gradient
TARGET encoder, and a hand-rolled VICReg anti-collapse objective (variance hinge + off-diagonal covariance).
latent_dim is 16; there is NO latent-variable z for multimodality, NO transformer, NO masking of image
patches. It demonstrates the JEPA *mechanism* -- learn to PREDICT in a learned representation space rather
than in pixel/observation space, using (encoder + predictor + EMA/stop-grad target + a latent-space loss +
an explicit anti-collapse regularizer) -- and it drops, duck-typed, into PrefrontalExecutive.plan() as its
`predictor`. It is NOT a frontier ViT world model and makes no such claim.

Why JEPA and not the sibling forward_model.py. forward_model.ForwardModel predicts the next sensory state
in the RAW INPUT space: s_hat = s + dt*(a_hat*s + b_hat*u), a scalar reafference/exafference comparator. THIS
module predicts the next state in a LEARNED LATENT space z' = f_theta(z, a), which is exactly the JEPA move
(LeCun): pixel/observation-space prediction wastes capacity modelling unpredictable detail, so predict an
abstract representation instead. The consequence, demonstrated in main(), is that a 16-D latent captures the
>=2 hidden factors of an action-conditioned world that a 1-D scalar forward model provably cannot separate.

Mechanism (rate-level, deterministic, per-instance rng):
  1. Online encoder  z  = unit(tanh(We @ pattern + be))            (a point on the unit sphere in R^latent_dim)
  2. Target encoder  z~ = unit(tanh(Wt @ pattern + bt))            (EMA of the online encoder; stop-gradient)
  3. Predictor       z' = clamp( z + Wz@z + Wa@onehot(a) + b )     (residual latent dynamics; bounded rollout)
  4. Invariance loss ||predict(encode(x_t), a_t) - sg(target_encode(x_{t+1}))||^2 -- the JEPA objective:
     predict the (stop-gradient) target latent of the NEXT observation from the current latent + the action.
  5. Anti-collapse (VICReg, hand-rolled over a ring buffer B of recent online latents, per dim j):
       L_var = mean_j max(0, gamma - sqrt(var_j + eps))            (gamma = 0.5/sqrt(L): a REACHABLE per-dim
                                                                    std target on the unit sphere; weight L)
       L_cov = sum_{i!=j} cov_ij^2 / latent_dim                    (decorrelate the latent dimensions)
     Closed-form gradients of L_var, L_cov w.r.t. We, be are backpropagated by hand through unit(tanh(.)).
  6. EMA target       Wt <- tau*Wt + (1-tau)*We,  bt likewise      (slow target = the stop-grad asymmetry)

Rollout bound. The residual predictor z' = z + Wz@z + ... could diverge over the long rollouts plan()
performs (horizon up to 40). predict() applies a final clamp so every |z'_i| stays bounded -- main() asserts
a 40-step rollout stays finite and bounded for every action. Wz is initialized small, decayed, and
row-scaled every step so its max row-abs-sum stays <= sqrt(latent_dim). That admits every coordinate
permutation of latent states and the successor map the planted world needs (measured -- the record's
section 9); it does NOT admit every rotation: the predictor is RESIDUAL, so the map is I + Wz and the
residual of an orthogonal Q is Q - I, whose rows can need L1 up to 1 + sqrt(latent_dim). Measured on the
planted world, no row-scaling at all was identical to sqrt(L) on two of three seeds and differed in the
fourth decimal on the third -- the bound is not binding there. (Until 2026-09-05 it was 0.9,
"contractive", which forbade every step longer than the latent's own radius.)

THE COLLAPSE ARC (specs/022 `evidence/jepa-collapse.md`, 2026-09-05; owner Q1(a)). At the previous defaults
(lambda_var 1.0, gamma 1.0, rho_max 0.9) the model the ARM constructs collapsed its latent on every context
the body sees (twelve of twelve in-brain models: target-latent spread x0.005-0.12 of a fresh model's) and its
prediction error could not tell a deterministic succession from a per-cycle permutation of it -- brain-free,
on a planted 3-bit successor world a tabular oracle reads to 0.0 and a plain linear predictor on the raw
pattern separates at 0.05, the model predicted the ORDERED world WORSE than the shuffled one (3 of 3 seeds).
Three necessary conditions, none sufficient alone or in pairs (per-item attribution in the record): a force of
L (the hinge's 1/L cancelled), a target of 0.5/sqrt(L) the latent can reach (gamma 1.0 never was, so the hinge
never switched off), and a bound of sqrt(L) that lets the predictor step between two separated states. With
all three the model reads the planted succession on 3 of 3 seeds by 12 cycles (ordered/shuffled error
0.069 / 0.022 / 0.238) with the latent interior (x0.61-0.70 of fresh). main() below used to run lambda_var
20 / lambda_cov 2 "to visibly hold variance" while the arm ran 1.0 -- it certified a configuration the live
model never had; it now runs the defaults (one configuration, P2). `latent_std()` over the last 32 ONLINE
latents reads low on a fixed or near-constant stimulus for a healthy encoder too -- it is collapse telemetry
only where the input varies; the two live-model self-checks that read it on a FIXED stimulus are wiring
checks, labelled so. `tests/test_jepa_collapse.py` carries the reds (green only after this change).

E-PROP COMPLIANCE (closed 2026-08-04; this module was the tree's named open violation). Every backward
path here is now a BROADCAST error signal: no weight matrix is ever read transposed to carry error, and
no credit is propagated through time. `docs/POLICIES.md` named ONE site (the encoder path); building the
fix found a SECOND the policy never named -- the 2-step rollout term's `gw = gp2 + Wz^T gp2`, which was
both weight transport and a 2-step BPTT unroll through the predictor's own recurrence. Both now share ONE
definition, `_feedback_gz`, which keeps only the residual skip (structure, not a weight). Measured over
1200 train / 350 held-out transitions, 5 seeds: held-out invariance MSE fell to **0.627x** the transport
incumbent, better on **5 of 5** seeds. The rollout consistency term itself was re-checked and KEPT -- it
still earns its place (removing it is 1.51x worse on 5 of 5 seeds); only its credit path was the defect.
`_feedback_gz` carries the full measurement, including the null on the fix the policy prescribed.

Fair control (un-riggable). anti_collapse=False is the ABLATION: it disables BOTH the VICReg terms
(lambda_var, lambda_cov treated as 0) AND the EMA/stop-grad asymmetry (the target encoder is tied to the
online encoder every step and the invariance target receives no stop-grad protection), so the control is
GUARANTEED to be able to collapse -- and main() shows it does (its per-dim latent std goes to ~0), while the
full model preserves latent variance.

Refs: LeCun, "A Path Towards Autonomous Machine Intelligence" (2022, the JEPA proposal); Assran et al.,
"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" / I-JEPA (2023);
Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance Regularization" (2022).

Deps: math, random, json (numpy/torch allowed project-wide); ASCII-only prints, deterministic (random.Random(seed) per instance).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.jepa_world_model
"""

import json
import math
import random

from snn_data_lake.substrate._mathutils import dot, mean, unit

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "JepaWorldModel.train_step": (
        "APPROXIMATES_GRADIENT",
        "one online JEPA update minimising invariance + variance + covariance terms; owes the "
        "cosine. Its weight-TRANSPORT half was retired 2026-08-04 -- `_feedback_gz()` is now the ONE "
        "backward definition and returns the identity/skip half alone, a broadcast error signal of "
        "the shape e-prop asks for",
    ),
    "_fit_linear": (
        "LOCAL",
        "plain LMS with no matrix inverse: a delta rule on its own inputs and targets. No gradient "
        "is propagated through a network and no weight is transported, so it claims none",
    ),
}


def bound_rows(W, rho_max, wd):
    """THE predictor bound, one mechanism for both latent world models (P2): a tiny weight decay on every
    entry of the residual map W, then row-scaling so its max row-abs-sum stays <= rho_max. In place.

    Extracted 2026-09-05 (evidence/jepa-live-floor.md section 2.6, commit A) from two byte-identical
    bodies -- JepaWorldModel.train_step's inline block and cerebellum/spiking_jepa.SpikingJEPA's copied
    _spectral_bound -- with the operation order kept exactly (decay, rowmax, scale), so both callers'
    self-check numbers repeat to the digit. rho_max's DERIVATION lives on the caller (see
    JepaWorldModel.__init__: sqrt(L), on the measurement, not the 1 + sqrt(L) a residual map can need);
    wd = 1e-3 stays a fixed constant in both classes, disclosed."""
    decay = 1.0 - wd
    for row in W:
        for k in range(len(row)):
            row[k] *= decay
    rowmax = max(sum(abs(v) for v in row) for row in W)
    if rowmax > rho_max:
        f = rho_max / rowmax
        for row in W:
            for k in range(len(row)):
                row[k] *= f


def _clamp(x, lo, hi):
    """Scalar clamp to [lo, hi] -- the final bounded squash on predicted latents."""
    return lo if x < lo else hi if x > hi else x


class JepaWorldModel:
    """A minimal deterministic linear JEPA latent world model.

    encode()/target_encode() are the online / EMA-target encoders (both unit(tanh(W@x + b))); predict() is the
    action-conditioned residual latent predictor (duck-typed for PrefrontalExecutive.plan()); train_step() runs
    one online update of the JEPA invariance loss + VICReg anti-collapse and EMA-updates the target; latent_std()
    is collapse telemetry; make_goal_value() builds a negative-L1 latent goal_value for plan(); save/load persist
    the weights as tagged JSON with a trust-boundary check."""

    def __init__(
        self,
        actions,
        in_dim=64,
        latent_dim=16,
        tau=0.99,
        lr=0.05,
        lambda_inv=1.0,
        lambda_var=None,
        lambda_cov=1.0,
        buffer=32,
        anti_collapse=True,
        seed=0,
        reward_shaping=False,
        lambda_reward=1.0,
    ):
        self.actions = tuple(actions)  # ordered action names, e.g. ("ENGAGE", ...)
        self.n_act = len(self.actions)
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        self.in_dim = int(in_dim)
        self.L = int(latent_dim)
        self.tau = float(tau)  # EMA target coefficient (~1 -> slow target)
        self.lr = float(lr)
        self.lambda_inv = float(lambda_inv)  # weight of the JEPA invariance (prediction) loss
        # THE COLLAPSE ARC (specs/022 evidence/jepa-collapse.md, 2026-09-05). The variance hinge's
        # gradient carries a 1/L from averaging the per-dimension hinge, so at the old default of
        # 1.0 the anti-collapse force was 1/(L*m) of the invariance gradient and the latent the arm
        # runs on collapsed (spread x0.005-0.11 of a fresh model's, twelve of twelve in-brain
        # models). lambda_var = L cancels that 1/L and puts the force at VICReg's own O(1/m) scale.
        # Derived from the latent width, not tuned: measured alone it opens the latent but does not
        # buy order; only with the reachable target and the freed predictor below does the model
        # read a deterministic succession (3 of 3 seeds). Callers may still pass a value.
        self.lambda_var = float(self.L) if lambda_var is None else float(lambda_var)
        self.lambda_cov = float(lambda_cov)  # weight of the VICReg covariance decorrelation
        self.buffer = int(buffer)  # ring-buffer size for the VICReg batch statistics
        self.anti_collapse = bool(
            anti_collapse
        )  # False == ablation: no VICReg + target tied to online
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance rng; identical seed -> identical weights

        # ---- fixed internal constants (not exposed on the contract) ----
        # VICReg target std (hinge threshold). A unit-norm latent's per-dimension std is at most
        # 1/sqrt(L) (sum_j var_j <= 1), so the old 1.0 was NEVER reachable and the hinge never
        # switched off -- a constant expansion force with no equilibrium, whose only fixed points
        # were collapse (weak force) or expansion to the ceiling against a lagging EMA target
        # (strong force). Half the isotropic ceiling is an interior target on the normalized
        # output's own scale (P1: a fixed point on a normalized output). Measured: with it the
        # hinge switches off (|g_var|/|g_inv| -> 0.00) and the encoder settles.
        self.gamma = 0.5 / math.sqrt(self.L)
        self.eps = 1e-4  # VICReg variance floor
        self.z_clip = 3.0  # predicted-latent clamp -> bounded rollout
        # max row-abs-sum of Wz. The old 0.9 ("contractive") forbade every latent step longer than
        # the latent's own radius, so once the latent was open the successor map between two
        # separated states was unrepresentable and the model predicted a deterministic world WORSE
        # than a shuffled one. sqrt(L) admits every coordinate permutation of latent states (a
        # permutation's residual rows have L1 <= 2) and, measured, the successor map the planted
        # world needs (record section 9); it does NOT admit every rotation -- the map is residual,
        # I + Wz, and the residual Q - I of an orthogonal Q can need row-L1 up to 1 + sqrt(L). On the
        # planted world no row-scaling at all measured identical to sqrt(L) on two of three seeds
        # (record section 6): the bound is not binding there, which is why it stays at sqrt(L) and
        # not the derivation's 1 + sqrt(L). Rollout boundedness never rested on this: predict()
        # clamps every output at z_clip, which main()'s 40-step rollout assert continues to verify.
        self.rho_max = math.sqrt(self.L)
        self.wd = 1e-3  # tiny weight decay on Wz each step
        self.lambda_roll = 0.3  # weight of the optional 2-step rollout consistency term
        self.use_rollout = True  # fight teacher-forcing drift with a 2-step term

        # ---- parameter init (all draws from the per-instance rng, so seed => byte-identical weights) ----
        se = 1.0 / math.sqrt(self.in_dim)  # encoder scale (fan-in)
        self.We = [
            [self.rng.uniform(-1.0, 1.0) * se for _ in range(self.in_dim)] for _ in range(self.L)
        ]
        self.be = [0.0] * self.L
        self.Wz = [
            [self.rng.uniform(-1.0, 1.0) * 0.02 for _ in range(self.L)] for _ in range(self.L)
        ]  # SMALL
        self.Wa = [
            [self.rng.uniform(-1.0, 1.0) * 0.05 for _ in range(self.n_act)] for _ in range(self.L)
        ]
        self.b = [0.0] * self.L
        # EMA target starts as a copy of the online encoder (standard); ablation re-ties it every step
        self.Wt = [row[:] for row in self.We]
        self.bt = self.be[:]

        # OPTIONAL reward-shaping head (task-aligns the latent). Pure SSL (invariance + VICReg) can COLLAPSE a
        # small set of repeating contexts to one point (the online arm latent measured ~0.003 apart across
        # distinct contexts), so no downstream value head can separate them -- the encoder, not the head, is the
        # bottleneck (v3.18 finding). A reward-predictive auxiliary head r_hat[a] = Wr[a].z + br[a] adds a weak
        # SUPERVISED gradient that pulls the latent to encode reward/task structure, so contexts with different
        # reward separate. This is "task-shaped SSL" (auxiliary reward prediction; cf. UNREAL), NOT pure SSL --
        # honestly: it consumes the reward signal. Drawn ONLY when enabled (so the rng stream, hence every
        # weight, is byte-identical to a plain model when off); a served/frozen deep encoder skips it entirely.
        self.reward_shaping = bool(reward_shaping)
        self.lambda_reward = float(lambda_reward)
        self.Wr = None
        self.br = None
        if self.reward_shaping:
            sr = 1.0 / math.sqrt(self.L)  # reward-head fan-in scale (over the latent)
            self.Wr = [
                [self.rng.uniform(-1.0, 1.0) * sr for _ in range(self.L)] for _ in range(self.n_act)
            ]
            self.br = [0.0] * self.n_act

        self._buf = []  # ring buffer of recent online latents (VICReg + telemetry)
        self._prev = None  # (prev online latent, prev action index) for 2-step term
        # OPTIONAL pre-encoder HIDDEN LAYER [W1, b1] (a DEEPER encoder trained in the torch tier and imported via
        # load_weights(snn_jepa/2)). When set, encode() becomes a 2-layer MLP and the deep encoder is SERVED and
        # FROZEN (only the predictor keeps adapting online); a freshly-constructed model is always linear (None).
        self.enc_hidden = None  # online encoder's hidden layer [W1, b1], or None (linear)
        self.tgt_hidden = None  # EMA target encoder's hidden layer [W1t, b1t], or None

    # ---- action handling -------------------------------------------------------------------------------
    def _idx(self, command):
        """Map an action name or an integer index to an action index (duck-typed command for predict/plan)."""
        if isinstance(command, str):
            return self._aidx[command]
        return int(command)

    # ---- encoders --------------------------------------------------------------------------------------
    @staticmethod
    def _hidden(pattern, hid):
        """Apply an optional pre-encoder hidden layer tanh(W1 @ pattern + b1); identity when hid is None."""
        if hid is None:
            return pattern
        W1, b1 = hid
        return [math.tanh(dot(W1[j], pattern) + b1[j]) for j in range(len(b1))]

    def encode(self, pattern):
        """Online encoder z = unit(tanh(We @ h + be)), h = optional hidden layer of `pattern` (identity if linear).
        Returns a length-latent_dim unit vector. A 2-layer MLP when a snn_jepa/2 deep encoder is loaded."""
        h = self._hidden(pattern, self.enc_hidden)
        return unit([math.tanh(dot(self.We[i], h) + self.be[i]) for i in range(self.L)])

    def target_encode(self, pattern):
        """EMA target encoder (same form as encode via Wt/bt + its own hidden layer); stop-gradient JEPA target."""
        h = self._hidden(pattern, self.tgt_hidden)
        return unit([math.tanh(dot(self.Wt[i], h) + self.bt[i]) for i in range(self.L)])

    # ---- predictor -------------------------------------------------------------------------------------
    def _predict_pre(self, z, aj):
        """Pre-clamp residual latent prediction z + Wz@z + Wa[:,aj] + b (aj is an action index)."""
        return [z[i] + dot(self.Wz[i], z) + self.Wa[i][aj] + self.b[i] for i in range(self.L)]

    def predict(self, state, command):
        """Action-conditioned residual latent predictor z' = clamp(z + Wz@z + Wa@onehot(a) + b).

        `state` is a latent vector z (length latent_dim), `command` is an action name or index. The clamp keeps
        every |z'_i| <= z_clip so long rollouts stay bounded. Duck-typed for PrefrontalExecutive.plan()."""
        p = self._predict_pre(state, self._idx(command))
        return [_clamp(v, -self.z_clip, self.z_clip) for v in p]

    # ---- one online JEPA update ------------------------------------------------------------------------
    def train_step(self, pattern_t, action_t, pattern_next, reward=None):
        """One online JEPA update; returns the scalar total loss.

        Loss = lambda_inv * invariance + lambda_var * L_var + lambda_cov * L_cov (+ a small 2-step rollout term).
        invariance = ||predict(encode(x_t), a_t) - sg(target_encode(x_{t+1}))||^2. Updates We, be, Wz, Wa, b by
        closed-form gradients (backprop through unit(tanh(.)) by hand), row-scales Wz so its max row-abs-sum stays
        <= rho_max = sqrt(latent_dim), then
        EMA-updates the target encoder. Pushes encode(x_t) into the VICReg ring buffer. In the ablation
        (anti_collapse=False) VICReg is off and the target is tied to the online encoder with no stop-grad.

        `reward` (opt-in reward-shaping only): the scalar reward the taken action a_t achieved for x_t. When
        reward_shaping is on and reward is given, a reward-predictive auxiliary head r_hat[a_t] = Wr[a_t].z +
        br[a_t] contributes lambda_reward * d/dz ||r_hat[a_t]-reward||^2 into the SAME encoder gradient, so the
        latent learns to encode reward/task structure (task-alignment). reward=None (or reward_shaping off) leaves
        every existing update byte-identical."""
        L, D = self.L, self.in_dim
        x, y = pattern_t, pattern_next
        aj = self._idx(action_t)
        # a loaded snn_jepa/2 deep encoder is FROZEN (trained in the torch tier); only the predictor adapts online.
        frozen = self.enc_hidden is not None
        xin = self._hidden(
            x, self.enc_hidden
        )  # output-layer input (x if linear, else the hidden activations)

        # -- forward: online encode of x, keeping pre-activation h, tanh t, and norm n for the backprop --
        h = [dot(self.We[i], xin) + self.be[i] for i in range(L)]
        t = [math.tanh(v) for v in h]
        n = math.sqrt(sum(c * c for c in t)) or 1.0
        z = [c / n for c in t]  # online latent = unit(tanh(...))

        self._buf.append(z[:])  # telemetry + VICReg batch (ring buffer)
        if len(self._buf) > self.buffer:
            self._buf.pop(0)

        # -- target latent of the next observation --
        ty = ny = None
        if self.anti_collapse:
            zt = self.target_encode(y)  # EMA target, stop-gradient
        else:
            # ablation: target tied to the ONLINE encoder, and NO stop-grad (gradient flows into it too) so the
            # two encodings are pulled together with nothing holding variance up -> guaranteed collapse.
            yin = self._hidden(y, self.enc_hidden)
            hy = [dot(self.We[i], yin) + self.be[i] for i in range(L)]
            ty = [math.tanh(v) for v in hy]
            ny = math.sqrt(sum(c * c for c in ty)) or 1.0
            zt = [c / ny for c in ty]

        # -- prediction + invariance residual (pre-clamp; clamp is inactive in the small-latent training regime) --
        p = self._predict_pre(z, aj)
        r = [p[i] - zt[i] for i in range(L)]
        loss_inv = sum(ri * ri for ri in r)

        # -- gradient accumulators --
        gWe = [[0.0] * D for _ in range(L)]
        gbe = [0.0] * L
        gWz = [[0.0] * L for _ in range(L)]
        gWa = [[0.0] * self.n_act for _ in range(L)]
        gb = [0.0] * L

        # invariance gradient into the predictor params:  dL/dp = 2*lambda_inv*r
        gp = [2.0 * self.lambda_inv * r[i] for i in range(L)]
        for i in range(L):
            gb[i] += gp[i]
            gWa[i][aj] += gp[i]  # onehot picks column aj
            gi = gp[i]
            row = gWz[i]
            for k in range(L):
                row[k] += gi * z[k]
        # -- ENCODER-side gradients (VICReg + backprop into We/be). SKIPPED when the encoder is a FROZEN deep
        #    model loaded from the torch tier -- then only the predictor adapts online. --
        Lvar = 0.0
        Lcov = 0.0
        if not frozen:
            gz = self._feedback_gz(gp)

            # -- VICReg gradients into z; the buffer supplies the batch statistics --
            if self.anti_collapse and len(self._buf) >= 2:
                m = len(self._buf)
                mu = [mean([bvec[j] for bvec in self._buf]) for j in range(L)]
                cen = [[bvec[j] - mu[j] for j in range(L)] for bvec in self._buf]
                var = [sum(cen[p_][j] * cen[p_][j] for p_ in range(m)) / m for j in range(L)]
                # variance hinge: keep each dim's batch std up towards gamma (blocks collapse to a constant)
                for j in range(L):
                    sd = math.sqrt(var[j] + self.eps)
                    Lvar += max(0.0, self.gamma - sd)
                    if (
                        self.gamma - sd > 0.0
                    ):  # hinge active -> push var up (move z away from the mean)
                        gz[j] += self.lambda_var * (-(z[j] - mu[j]) / (L * m * sd))
                Lvar /= L
                # covariance: decorrelate the off-diagonal latent dimensions
                cov = [
                    [sum(cen[p_][k] * cen[p_][j] for p_ in range(m)) / m for j in range(L)]
                    for k in range(L)
                ]
                for k in range(L):
                    gk = 0.0
                    for j in range(L):
                        if j != k:
                            Lcov += cov[k][j] * cov[k][j]
                            gk += cov[k][j] * (z[j] - mu[j])
                    gz[k] += self.lambda_cov * (4.0 / (L * m)) * gk
                Lcov /= L

            # -- REWARD-SHAPING (opt-in): a reward-predictive head r_hat[aj]=Wr[aj].z+br[aj] toward the OBSERVED
            #    reward pulls the latent to encode task/reward structure. Its dL/dz folds into the SAME gz (so the
            #    existing unit(tanh) backprop handles it); the head itself is updated by SGD. Task-aligns z so a
            #    downstream value head can separate contexts the pure-SSL objective would collapse. --
            if self.reward_shaping and reward is not None and self.Wr is not None:
                rerr = (dot(self.Wr[aj], z) + self.br[aj]) - reward  # r_hat[aj] - reward
                gr = 2.0 * self.lambda_reward * rerr  # dL_r/d r_hat
                wraj = self.Wr[aj]
                for k in range(L):
                    gz[k] += gr * wraj[k]  # dL_r/dz = gr * Wr[aj] -> into the shared gz
                for k in range(L):  # SGD the head (uses the pre-update z)
                    wraj[k] -= self.lr * gr * z[k]
                self.br[aj] -= self.lr * gr

            # -- backprop the combined gz through z = unit(tanh(h)) into We, be --
            self._encoder_backprop(gz, t, n, xin, gWe, gbe)

            # -- ablation only: also backprop -gp through the (untied, no-stop-grad) target encode(y) --
            if not self.anti_collapse:
                self._encoder_backprop([-gp[i] for i in range(L)], ty, ny, yin, gWe, gbe)

        # -- optional 2-step rollout consistency term (predictor params only; robust, skipped on first call) --
        Lroll = 0.0
        if self.use_rollout and self._prev is not None:
            z0, a0 = self._prev
            w = [
                z0[i] + dot(self.Wz[i], z0) + self.Wa[i][a0] + self.b[i] for i in range(L)
            ]  # inner predict
            p2 = [
                w[i] + dot(self.Wz[i], w) + self.Wa[i][aj] + self.b[i] for i in range(L)
            ]  # outer predict
            r2 = [
                p2[i] - zt[i] for i in range(L)
            ]  # 2-step (a0 then a_t) should reach the same next latent
            Lroll = sum(v * v for v in r2)
            gp2 = [2.0 * self.lambda_roll * r2[i] for i in range(L)]
            for i in range(L):  # outer layer
                gb[i] += gp2[i]
                gWa[i][aj] += gp2[i]
                for k in range(L):
                    gWz[i][k] += gp2[i] * w[k]
            # dL/dw, through the SAME backward definition as the encoder path (see _feedback_gz).
            # This site used to spell out `gp2 + Wz^T gp2` inline -- the same weight transport as
            # the encoder path AND a 2-step BPTT unroll through the predictor's own recurrence,
            # which the e-prop invariant forbids twice over. One definition now serves both.
            gw = self._feedback_gz(gp2)
            for i in range(L):  # inner layer
                gb[i] += gw[i]
                gWa[i][a0] += gw[i]
                for k in range(L):
                    gWz[i][k] += gw[i] * z0[k]

        # -- SGD update (predictor always; the encoder only when it is not a frozen deep model) --
        lr = self.lr
        for i in range(L):
            if not frozen:
                wei = self.We[i]
                gwi = gWe[i]
                for k in range(D):
                    wei[k] -= lr * gwi[k]
                self.be[i] -= lr * gbe[i]
            wzi = self.Wz[i]
            gzi = gWz[i]
            for k in range(L):
                wzi[k] -= lr * gzi[k]
            wai = self.Wa[i]
            gai = gWa[i]
            for j in range(self.n_act):
                wai[j] -= lr * gai[j]
            self.b[i] -= lr * gb[i]

        # -- bound on Wz: small weight decay + row-scaling so max row-abs-sum stays <= rho_max = sqrt(L)
        #    (see __init__); the ONE body, shared with spiking_jepa, is the module-level bound_rows --
        bound_rows(self.Wz, self.rho_max, self.wd)

        # -- target encoder update (skipped for a frozen deep encoder -- its loaded target stays fixed):
        #    EMA (full) or hard tie to online (ablation guarantees target == online) --
        if not frozen:
            if self.anti_collapse:
                tau = self.tau
                for i in range(L):
                    for k in range(D):
                        self.Wt[i][k] = tau * self.Wt[i][k] + (1.0 - tau) * self.We[i][k]
                    self.bt[i] = tau * self.bt[i] + (1.0 - tau) * self.be[i]
            else:
                self.Wt = [row[:] for row in self.We]
                self.bt = self.be[:]

        self._prev = (z[:], aj)  # detached latent for the next 2-step term
        return (
            self.lambda_inv * loss_inv
            + self.lambda_var * Lvar
            + self.lambda_cov * Lcov
            + (self.lambda_roll * Lroll if self.use_rollout else 0.0)
        )

    def _feedback_gz(self, gp):
        """Carry the invariance error back from the prediction p to the latent z -- as a BROADCAST
        learning signal, with NO path through the predictor's weights.

        WHAT THIS FIXES (the e-prop cross-cutting invariant, closed 2026-08-04). p = (I + Wz)z +
        const, so the exact derivative is dL/dz = gp + Wz^T gp. This method used to compute exactly
        that, and the `Wz^T gp` half read the FORWARD weight matrix transposed to carry error
        backward -- weight transport, on the live wake tick (`cortex/brain.py:232` builds the brain
        with `jepa=True`, so `train_step` runs every tick through `autonomous_loop`). `docs/POLICIES.md`
        named this site as an open violation with its benchmark specified.

        What survives is the IDENTITY half alone. That half is not a weight: it is the residual skip
        in `p = z + ...`, so `gp` reaching `z` directly is a broadcast error signal of exactly the
        shape e-prop asks for (a local eligibility times a broadcast learning signal). The predictor's
        own updates (Wz, Wa, b) are untouched -- they consume `gp` directly and were never transport.

        MEASURED, and the measurement is why the fix is this and not the one that was prescribed.
        POLICIES.md prescribed FEEDBACK ALIGNMENT (a fixed random Bz replacing the transpose). That
        was built and benchmarked first, over 1200 train / 350 held-out transitions on this module's
        own world, 5 seeds.

        RE-DERIVED 2026-08-05, and the protocol is stated because without it none of this is
        checkable -- the previous version of this table recorded FA figures whose Bz construction
        was nowhere in the tree, and its drop-the-term row (0.654) CONTRADICTED this module's own
        header docstring (0.627x) a hundred lines above. Protocol: `_gen_stream(1200, seed=1, ...)`
        train and `_gen_stream(350, seed=2, ...)` held out -- main()'s own two streams -- model
        seeds 0..4, held-out invariance MSE = mean over the stream of
        sum_i (predict(encode(x),a)[i] - target_encode(y)[i])^2; Bz drawn as
        `random.Random(1234).uniform(-1,1) * 0.02` per entry, the same scale band as Wz's init:

            held-out invariance MSE, ratio to the transport incumbent
              THIS (drop the recurrent term)       : 0.627 mean  (0.571 .. 0.717)   5 of 5 seeds better
              feedback alignment (fixed random Bz) : 0.620 mean  (0.554 .. 0.728)   5 of 5 seeds better
              cosine(FA gz, transport gz)          : 0.990 mean  (>> the 0.5 e-prop bar)
              cosine(THIS gz, transport gz)        : 0.991 mean  (>> the 0.5 e-prop bar)

        FA and dropping the term land within 1.2% of each other while both are ~37% better than
        transport, and six independent Bz draws at seed 0 span 0.704-0.724 against this method's
        0.717 -- i.e. the shipped method sits INSIDE the spread of the random matrix's own draws.
        The random matrix contributes nothing measurable; the entire gain is the removal of the
        transport term, not feedback alignment. Adding Bz would have been a new matrix, a new seed
        stream and a new constant bought with a null (P2: a merge must REDUCE; P8 guard (d): report
        the null). Recorded rather than shipped. *(An earlier version of this note said FA won "on
        only 2 of 5 seeds"; under the stated Bz it wins on 4 of 5, by a margin far inside the
        seed-to-seed spread. The conclusion is unchanged and the honest statement of it is the
        six-draw spread above, which does not depend on which side of a coin-flip FA lands.)*

        Both paths still beat the identity predictor on every seed, and the `anti_collapse=False`
        ablation still collapses -- ctrl_std 0.0070 vs full 0.1219, which is what this module's own
        self-check prints on every run. (This read "0.0071 vs 0.1225", which matches neither the
        shipped module nor the pre-fix one -- re-measured, the pre-fix pair is 0.0078 / 0.1403.)
        So the module's fair control is intact.
        """
        return list(gp)

    def prediction_error(self, pattern_t, action_t, pattern_next):
        """The invariance (prediction) loss for ONE transition WITHOUT updating -- the world-model 'surprise'
        that a curiosity / intrinsic-motivation drive consumes. Read-only: it mutates no weight, buffer, or
        state (unlike train_step). error = ||predict(encode(x_t), a_t) - target_encode(x_{t+1})||^2."""
        aj = self._idx(action_t)
        z = self.encode(pattern_t)
        zt = self.target_encode(pattern_next)
        p = self._predict_pre(z, aj)
        return sum((p[i] - zt[i]) ** 2 for i in range(self.L))

    def _encoder_backprop(self, gz, t, n, x, gWe, gbe):
        """Chain dL/dz (=gz) through z = unit(tanh(h)) to accumulate dL/dWe, dL/dbe.

        unit Jacobian: dz_i/dt_j = delta_ij/n - t_i t_j / n^3 ; tanh': dt_j/dh_j = 1 - t_j^2 ; h = We@x + be."""
        L, D = self.L, self.in_dim
        sdot = sum(gz[i] * t[i] for i in range(L))  # gz . t (the unit-normalization coupling term)
        n3 = n * n * n
        for j in range(L):
            dLdt = gz[j] / n - sdot * t[j] / n3
            dLdh = dLdt * (1.0 - t[j] * t[j])
            gbe[j] += dLdh
            row = gWe[j]
            for k in range(D):
                row[k] += dLdh * x[k]

    # ---- planning / telemetry / persistence ------------------------------------------------------------
    def make_goal_value(self, z_goal):
        """Return goal_value(z) = -sum_i |z_i - z_goal_i| (negative L1 distance to a goal latent), for plan()."""
        zg = list(z_goal)

        def goal_value(z):
            return -sum(abs(z[i] - zg[i]) for i in range(len(zg)))

        return goal_value

    def latent_std(self):
        """Mean over dimensions of the per-dimension std of the last `buffer` CONSECUTIVE online latents.

        Collapse telemetry where the input varies (a collapsed encoder maps every input to the same
        latent, so every dimension's std -> 0). On a near-constant stream it is NOT a collapse gauge: it
        reads the stimulus's consecutive variation plus the hinge's drift between near-duplicates --
        measured in-brain 2026-09-05 (specs/022 evidence/jepa-live-floor.md): an untrained twin read
        0.01-0.07 over 32 consecutive ticks, the trained model 0.025-0.058, and the trained model over
        DISTINCT patterns 0.036-0.045 (below the 0.05 floor on every seed). The live criterion that
        replaced the floor there is the trained/fresh spread ratio over distinct patterns (>= x0.10; its
        red is the ablation's x0.01-0.08) -- read RED on 1 of 5 seeds in-brain (seed 3: x0.062 against the
        ablation's x0.056), so that criterion is OPEN; the 0.05 floor stays this module's self-check bar
        on its varying synthetic stream and the in-tree planted world."""
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

    def save_weights(self, path):
        """Persist the weights as tagged JSON. A linear model writes `snn_jepa/1`; a model carrying a hidden
        encoder (a deep encoder imported from the torch tier) writes `snn_jepa/2` with the encoder as an ordered
        list of [W, b] layers `enc_layers`/`tgt_layers`. Floats round-trip exactly.

        P4b STATUS -- a SECOND DURABLE STORE that is not on the wake tick. RE-COUNTED 2026-08-05,
        and the previous count was wrong in the direction that mattered. These weights ARE learned
        state, and P4b says learned state belongs in `brain_lake.bin` under a record tag, "not a
        sidecar file, not a JSON blob".

        This note said "a tree-wide search found ZERO callers of `save_weights`/`load_weights`
        outside this module's own `main()`, so nothing live writes a sidecar today and there is no
        violation to fix". `save_weights` has zero external callers; `load_weights` has one --
        `reasoning/jepa_scaffold_torch.py::verify_transfer` constructs a `JepaWorldModel` and calls
        `core.load_weights(path)` on an `snn_jepa/2` export, reached from that module's own
        `train`/export flow. So the path this note described as a future hazard -- "the documented
        purpose of `snn_jepa/2` is importing a torch-tier-trained deep encoder into the brain" -- is
        BUILT AND EXERCISED, on the train tier rather than the wake tick.

        That changes the disposition from "nothing to fix" to "open P4b work with a named owner":
        the brain's learned encoder round-trips through a JSON file beside the lake, and only the
        tier it runs on keeps it off the live path. Closing it means a `_T_*` record tag in
        `substrate/home.py`, which is outside this module -- recorded here so the next reader finds
        it before wiring the import into the wake tick rather than after."""
        obj = {
            "actions": list(self.actions),
            "in_dim": self.in_dim,
            "latent_dim": self.L,
            "tau": self.tau,
            "lr": self.lr,
            "lambda_inv": self.lambda_inv,
            "lambda_var": self.lambda_var,
            "lambda_cov": self.lambda_cov,
            "buffer": self.buffer,
            "anti_collapse": self.anti_collapse,
            "Wz": self.Wz,
            "Wa": self.Wa,
            "b": self.b,
        }
        if self.enc_hidden is None:
            obj.update(
                {"format": "snn_jepa/1", "We": self.We, "be": self.be, "Wt": self.Wt, "bt": self.bt}
            )
        else:
            th = self.tgt_hidden if self.tgt_hidden is not None else self.enc_hidden
            obj.update(
                {
                    "format": "snn_jepa/2",
                    "hidden": len(self.enc_hidden[1]),
                    "enc_layers": [[self.enc_hidden[0], self.enc_hidden[1]], [self.We, self.be]],
                    "tgt_layers": [[th[0], th[1]], [self.Wt, self.bt]],
                }
            )
        with open(path, "w", encoding="ascii") as f:
            json.dump(obj, f)

    def load_weights(self, path):
        """Load weights from tagged JSON with a trust-boundary check. Accepts `snn_jepa/1` (linear encoder) and
        `snn_jepa/2` (a DEEP encoder as `enc_layers=[[W1,b1],[W2,b2]]`, served + frozen -- only the predictor
        adapts online afterwards). Validates the format tag and every shape, raising ValueError on any mismatch
        (never trust an unvalidated weight file). Missing hyper-parameter metadata defaults to the instance's."""
        with open(path, encoding="ascii") as f:
            obj = json.load(f)
        fmt = obj.get("format") if isinstance(obj, dict) else None
        if fmt not in ("snn_jepa/1", "snn_jepa/2"):
            raise ValueError(
                "bad JEPA weight file: missing/incorrect 'format' tag (expected snn_jepa/1 or /2)"
            )
        L = int(obj.get("latent_dim", -1))
        D = int(obj.get("in_dim", -1))
        na = len(obj.get("actions", []))
        Wz, Wa, b = obj.get("Wz"), obj.get("Wa"), obj.get("b")

        def _shape_ok(M, rows, cols):
            return (
                isinstance(M, list)
                and len(M) == rows
                and all(isinstance(row, list) and len(row) == cols for row in M)
            )

        if not (L > 0 and D > 0 and na > 0):
            raise ValueError("bad JEPA weight file: non-positive dimensions")
        if not (
            _shape_ok(Wz, L, L) and _shape_ok(Wa, L, na) and isinstance(b, list) and len(b) == L
        ):
            raise ValueError("bad JEPA weight file: predictor shape mismatch")

        if fmt == "snn_jepa/1":
            We, be, Wt, bt = obj.get("We"), obj.get("be"), obj.get("Wt"), obj.get("bt")
            if not (_shape_ok(We, L, D) and isinstance(be, list) and len(be) == L):
                raise ValueError("bad JEPA weight file: encoder shape mismatch")
            if Wt is None or bt is None:  # a torch /1 export may ship only the online encoder
                Wt, bt = [row[:] for row in We], be[:]
            elif not (_shape_ok(Wt, L, D) and isinstance(bt, list) and len(bt) == L):
                raise ValueError("bad JEPA weight file: target-encoder shape mismatch")
            enc_hidden = tgt_hidden = None
        else:  # snn_jepa/2: a 2-layer deep encoder
            enc_layers, tgt_layers = obj.get("enc_layers"), obj.get("tgt_layers")
            if not (isinstance(enc_layers, list) and len(enc_layers) == 2):
                raise ValueError(
                    "bad JEPA weight file: snn_jepa/2 needs enc_layers=[[W1,b1],[W2,b2]]"
                )
            (W1, b1), (We, be) = enc_layers[0], enc_layers[1]
            H = len(b1) if isinstance(b1, list) else -1
            if not (
                _shape_ok(W1, H, D)
                and _shape_ok(We, L, H)
                and isinstance(be, list)
                and len(be) == L
            ):
                raise ValueError("bad JEPA weight file: deep-encoder shape mismatch")
            if tgt_layers is None:  # default the target to a copy of the online deep encoder
                W1t, b1t, Wt, bt = [r[:] for r in W1], b1[:], [r[:] for r in We], be[:]
            else:
                (W1t, b1t), (Wt, bt) = tgt_layers[0], tgt_layers[1]
                if not (_shape_ok(W1t, H, D) and _shape_ok(Wt, L, H)):
                    raise ValueError("bad JEPA weight file: target deep-encoder shape mismatch")
            enc_hidden, tgt_hidden = [W1, b1], [W1t, b1t]

        # trusted: adopt the validated weights and geometry (metadata defaults to the instance's when absent)
        self.actions = tuple(obj["actions"])
        self.n_act = na
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        self.in_dim, self.L = D, L
        # gamma and rho_max are DERIVED from L in __init__; re-derive them for the loaded geometry so
        # a file of another width does not carry the constructor's width (2026-09-05, the collapse arc).
        # lambda_var is read from the file below: an old snn_jepa/1 file restores its 1.0 beside these.
        self.gamma = 0.5 / math.sqrt(self.L)
        self.rho_max = math.sqrt(self.L)
        self.tau = float(obj.get("tau", self.tau))
        self.lr = float(obj.get("lr", self.lr))
        self.lambda_inv = float(obj.get("lambda_inv", self.lambda_inv))
        self.lambda_var = float(obj.get("lambda_var", self.lambda_var))
        self.lambda_cov = float(obj.get("lambda_cov", self.lambda_cov))
        self.buffer = int(obj.get("buffer", self.buffer))
        self.anti_collapse = bool(obj.get("anti_collapse", True))
        self.We = [row[:] for row in We]
        self.be = be[:]
        self.Wz = [row[:] for row in Wz]
        self.Wa = [row[:] for row in Wa]
        self.b = b[:]
        self.Wt = [row[:] for row in Wt]
        self.bt = bt[:]
        self.enc_hidden = [[r[:] for r in enc_hidden[0]], enc_hidden[1][:]] if enc_hidden else None
        self.tgt_hidden = [[r[:] for r in tgt_hidden[0]], tgt_hidden[1][:]] if tgt_hidden else None
        self._buf = []
        self._prev = None


# =========================================================================================================
# Self-check: a synthetic ACTION-CONDITIONED world. A hidden latent u in R^2 evolves under 3 discrete actions
# with distinct band-structured dynamics; observations are a banded [0,1]^in_dim RBF rendering of u + noise.
# =========================================================================================================

# Balanced 2-factor dynamics (sum of each column is 0 -> a mean-zero random walk that explores [0,1]^2).
# RELEASE and ENGAGE move u0 IDENTICALLY (+d) but u1 OPPOSITELY -- a scalar readout of u0 cannot
# tell them apart, yet the 2-D latent can. HOLD moves u0 the other way (-2d) to balance the walk.
_EFFECTS = {
    "RELEASE": (+1.0, -1.0),
    "ENGAGE": (+1.0, +1.0),
    "HOLD": (-2.0, 0.0),
}


def _world_step(u, action, d):
    """Advance the hidden 2-factor latent u under `action`, clamped to [0, 1]^2."""
    e0, e1 = _EFFECTS[action]
    return [_clamp(u[0] + e0 * d, 0.0, 1.0), _clamp(u[1] + e1 * d, 0.0, 1.0)]


def _render(u, encoders, rng, noise):
    """Banded [0,1]^in_dim rendering of u: concatenate each factor's RBF population activations, add noise."""
    pat = []
    for i in range(len(u)):
        pat.extend(encoders[i].activations(u[i]))  # continuous Gaussian activations in [0,1]
    return [_clamp(v + rng.uniform(-noise, noise), 0.0, 1.0) for v in pat]


def _band0_centroid(pattern, centers, band):
    """A 1-D scalar readout of factor 0 only: the activation-weighted centroid of band 0 (tracks u0, blind to
    u1). This is the scalar bottleneck the ForwardModel baseline is fit on."""
    b = pattern[:band]
    tot = sum(b)
    return sum(c * v for c, v in zip(centers, b)) / tot if tot > 0 else 0.0


def _gen_stream(n, seed, encoders, d, noise, actions, start=(0.5, 0.5)):
    """Generate a contiguous trajectory of (pattern_t, action_t, pattern_next) triples plus the hidden u's."""
    rng = random.Random(seed)
    u = list(start)
    triples, hidden = [], []
    for _ in range(n):
        a = actions[rng.randrange(len(actions))]
        x = _render(u, encoders, rng, noise)
        un = _world_step(u, a, d)
        y = _render(un, encoders, rng, noise)
        triples.append((x, a, y))
        hidden.append((u[:], a, un[:]))
        u = un
    return triples, hidden


def _fit_linear(feats, targets, out_dim, epochs, lr, seed):
    """Fit W (out_dim x feat_dim) mapping feats -> targets by plain LMS (no matrix inverse)."""
    fd = len(feats[0])
    W = [[0.0] * fd for _ in range(out_dim)]
    rng = random.Random(seed)
    idx = list(range(len(feats)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for p_ in idx:
            f = feats[p_]
            tgt = targets[p_]
            for o in range(out_dim):
                pred = dot(W[o], f)
                e = tgt[o] - pred
                wo = W[o]
                for k in range(fd):
                    wo[k] += lr * e * f[k]
    return W


def _readout_mse(W, feats, targets, out_dim):
    """Mean squared error of a fitted linear readout W over (feats, targets)."""
    tot = 0.0
    for f, tgt in zip(feats, targets):
        for o in range(out_dim):
            e = tgt[o] - dot(W[o], f)
            tot += e * e
    return tot / len(feats)


def main():
    from snn_data_lake.cerebellum.forward_model import ForwardModel
    from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
    from snn_data_lake.sensorimotor.population_encoding import PopulationEncoder

    IN_DIM, K, LATENT = 64, 2, 16
    BAND = IN_DIM // K  # 32 neurons per factor
    D, NOISE = 0.20, 0.02
    ACTIONS = ("ENGAGE", "RELEASE", "HOLD")
    OPEN = 1  # index of RELEASE (the goal action) in ACTIONS
    STEPS, LR = 1200, 0.08
    # Both models run the module DEFAULTS for the VICReg weights (one configuration, P2). Until
    # 2026-09-05 this self-check overrode them to 20 / 2 "to visibly hold variance" while the arm
    # ran the default 1.0 -- it certified a configuration the live model never had (the collapse
    # arc, evidence/jepa-collapse.md). The defaults now derive from the latent width.
    encoders = [
        PopulationEncoder(n=BAND, lo=0.0, hi=1.0, max_spikes=10, overlap=1.0) for _ in range(K)
    ]
    centers0 = encoders[0].centers

    # ---- train the full JEPA and the fair control on the SAME seeded stream / steps / architecture ----
    train, train_hidden = _gen_stream(
        STEPS, 1, encoders, D, NOISE, ACTIONS
    )  # keep hidden (reused below)
    full = JepaWorldModel(
        ACTIONS,
        in_dim=IN_DIM,
        latent_dim=LATENT,
        lr=LR,
        anti_collapse=True,
        seed=0,
    )
    ctrl = JepaWorldModel(
        ACTIONS,
        in_dim=IN_DIM,
        latent_dim=LATENT,
        lr=LR,
        anti_collapse=False,
        seed=0,
    )
    for x, a, y in train:
        full.train_step(x, a, y)
    for x, a, y in train:
        ctrl.train_step(x, a, y)
    full_std, ctrl_std = full.latent_std(), ctrl.latent_std()
    # by construction the ablation ties the target encoder to the online encoder every step
    tied = ctrl.Wt == ctrl.We and ctrl.bt == ctrl.be

    # ---- (2) BASELINES on HELD-OUT episodes of the same world ----
    hold, hidden = _gen_stream(350, 2, encoders, D, NOISE, ACTIONS)
    # (2a) JEPA next-latent MSE beats an IDENTITY predictor (predict(z,a)=z), in the SAME latent space
    err_j = [
        sum(
            (full.predict(full.encode(x), a)[i] - full.target_encode(y)[i]) ** 2
            for i in range(LATENT)
        )
        for (x, a, y) in hold
    ]
    err_id = [
        sum((full.encode(x)[i] - full.target_encode(y)[i]) ** 2 for i in range(LATENT))
        for (x, a, y) in hold
    ]
    mse_j_lat, mse_id_lat = mean(err_j), mean(err_id)

    # (2b) JEPA beats a 1-D linear ForwardModel on a scalar projection -- fair common target: predict next u.
    CMD = {
        "RELEASE": 1.0,
        "ENGAGE": 1.0,
        "HOLD": -1.0,
    }  # OPEN/CLOSE share a command: the scalar
    fm = ForwardModel(a=0.0, b=0.0, dt=1.0, seed=0)  # readout of u0 cannot distinguish them
    for _ep in range(4):
        for x, a, y in train:
            fm.update(
                _band0_centroid(x, centers0, BAND),
                CMD[a],
                _band0_centroid(y, centers0, BAND),
                lr=0.15,
            )
    # JEPA feature -> u readout, and ForwardModel feature -> u readout, both LMS-fit on the training stream
    jf = [full.predict(full.encode(x), a) for (x, a, _y) in train]
    ff = [[fm.predict(_band0_centroid(x, centers0, BAND), CMD[a]), 1.0] for (x, a, _y) in train]
    ut = [un for (_u, _a, un) in hidden]  # held-out hidden next-state targets
    utr = [un for (_u, _a, un) in train_hidden]  # training hidden next-state targets
    Wj = _fit_linear(jf, utr, 2, 12, 0.05, seed=3)
    Wf = _fit_linear(ff, utr, 2, 12, 0.05, seed=3)
    jf_h = [full.predict(full.encode(x), a) for (x, a, _y) in hold]
    ff_h = [[fm.predict(_band0_centroid(x, centers0, BAND), CMD[a]), 1.0] for (x, a, _y) in hold]
    mse_j_u = _readout_mse(Wj, jf_h, ut, 2)
    mse_f_u = _readout_mse(Wf, ff_h, ut, 2)

    # ---- (3) BOUNDED ROLLOUT: 40 steps from an encoded state, for every action, stays finite and bounded ----
    z0 = full.encode(_render([0.5, 0.5], encoders, random.Random(9), NOISE))
    roll_bounded, roll_max = True, 0.0
    for a in ACTIONS:
        z = z0
        for _ in range(40):
            z = full.predict(z, a)
            for v in z:
                if not math.isfinite(v) or abs(v) >= 10.0:
                    roll_bounded = False
                roll_max = max(roll_max, abs(v))

    # ---- (4) LATENT > SCALAR via plan(): a goal that depends on BOTH hidden factors ----
    # Start (0.40, 0.50); goal (u0 high, u1 low) -> reachable only by RELEASE (u0 up, u1 down). CLOSE is a
    # decoy (u0 up, u1 UP) that a u0-only scalar cannot distinguish from OPEN.
    rng_g = random.Random(21)
    start_pat = _render([0.40, 0.50], encoders, rng_g, NOISE)
    goal_pat = _render([0.85, 0.10], encoders, rng_g, NOISE)
    z_start, z_goal = full.encode(start_pat), full.encode(goal_pat)
    gv = full.make_goal_value(z_goal)
    HZ = 4

    def roll_val(pred, s0, cmd, horizon, value):  # terminal value of a horizon-step rollout
        s = s0
        for _ in range(horizon):
            s = pred.predict(s, cmd)
        return value(s)

    jepa_val = {a: roll_val(full, z_start, i, HZ, gv) for i, a in enumerate(ACTIONS)}
    jepa_best = max(range(len(ACTIONS)), key=lambda i: jepa_val[ACTIONS[i]])
    jepa_gap = jepa_val["RELEASE"] - jepa_val["ENGAGE"]

    # the scalar ForwardModel: state = u0 centroid, goal = u0 centroid of the goal; OPEN and CLOSE share a
    # command, so its rollout values for them are PROVABLY identical -> it cannot represent the task.
    s_start = _band0_centroid(start_pat, centers0, BAND)
    s_goal = _band0_centroid(goal_pat, centers0, BAND)
    scal_gv = lambda s: -((s - s_goal) ** 2)
    fm_val = {a: roll_val(fm, s_start, CMD[a], HZ, scal_gv) for a in ACTIONS}
    fm_gap = abs(fm_val["RELEASE"] - fm_val["ENGAGE"])

    # drive PrefrontalExecutive.plan() with the JEPA as its (duck-typed) predictor
    pfc = PrefrontalExecutive(n_options=len(ACTIONS), seed=5)
    plan_goal, _rt = pfc.plan(z_start, [0, 1, 2], full, gv, horizon=HZ)
    plan_goal2, _rt2 = PrefrontalExecutive(n_options=len(ACTIONS), seed=5).plan(
        z_start, [0, 1, 2], full, gv, horizon=HZ
    )
    plan_det = plan_goal == plan_goal2

    # ---- DETERMINISM: two seed=7 models on the same stream -> byte-identical weights ----
    det_a = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, lr=LR, seed=7)
    det_b = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, lr=LR, seed=7)
    for x, a, y in train[:200]:
        det_a.train_step(x, a, y)
        det_b.train_step(x, a, y)
    same_weights = (
        det_a.We == det_b.We
        and det_a.be == det_b.be
        and det_a.Wz == det_b.Wz
        and det_a.Wa == det_b.Wa
        and det_a.b == det_b.b
        and det_a.Wt == det_b.Wt
        and det_a.bt == det_b.bt
    )

    # ---- SAVE / LOAD round-trip (exact) ----
    import os
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "snn_jepa_roundtrip.json")
    full.save_weights(tmp)
    loaded = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, seed=99)
    loaded.load_weights(tmp)
    probe = _render([0.3, 0.7], encoders, random.Random(4), NOISE)
    roundtrip = (
        loaded.We == full.We
        and loaded.Wz == full.Wz
        and loaded.Wa == full.Wa
        and loaded.b == full.b
        and loaded.Wt == full.Wt
        and loaded.encode(probe) == full.encode(probe)
        and loaded.predict(loaded.encode(probe), "HOLD") == full.predict(full.encode(probe), "HOLD")
    )
    os.remove(tmp)

    # ---- HIDDEN ENCODER (snn_jepa/2): a served DEEP encoder represents structure a single linear layer can't ----
    # XOR is not separable by one (tanh) layer; a hidden layer (OR/AND -> XOR) makes it separable. The OUTPUT
    # layer is identical with and without the hidden pre-layer, so this isolates the hidden layer's contribution.
    xor_in = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    xor_lab = [[0.0], [1.0], [1.0], [0.0]]
    W1x, b1x = [[8.0, 8.0], [8.0, 8.0]], [-4.0, -12.0]  # hidden: h0 ~ OR(x0,x1), h1 ~ AND(x0,x1)
    Wex, bex = (
        [[1.0, -1.0], [0.0, 1.0]],
        [0.0, 0.0],
    )  # output: latent ~ [OR-AND, AND] -> XOR lives in z0
    deep = JepaWorldModel(ACTIONS, in_dim=2, latent_dim=2, seed=0)
    deep.enc_hidden, deep.tgt_hidden = [W1x, b1x], [[r[:] for r in W1x], b1x[:]]
    deep.We, deep.be, deep.Wt, deep.bt = Wex, bex, [r[:] for r in Wex], bex[:]
    p2 = os.path.join(tempfile.gettempdir(), "snn_jepa2_roundtrip.json")
    deep.save_weights(p2)  # writes snn_jepa/2 (the deep encoder as enc_layers)
    served = JepaWorldModel(ACTIONS, in_dim=2, latent_dim=2, seed=1)
    served.load_weights(p2)  # round-trip a deep encoder
    os.remove(p2)
    deep_lat = [served.encode(x) for x in xor_in]
    v2_roundtrip = deep_lat == [deep.encode(x) for x in xor_in]
    lin = JepaWorldModel(
        ACTIONS, in_dim=2, latent_dim=2, seed=2
    )  # ablation: SAME output layer, NO hidden layer
    lin.We, lin.be = Wex, bex
    lin_lat = [lin.encode(x) for x in xor_in]

    def _xor_acc(latents):  # fit a linear readout latent -> XOR label, count correct
        w = _fit_linear(latents, xor_lab, 1, 400, 0.1, seed=0)
        return sum(1 for lt, lb in zip(latents, xor_lab) if (dot(w[0], lt) > 0.5) == (lb[0] > 0.5))

    deep_acc, lin_acc = _xor_acc(deep_lat), _xor_acc(lin_lat)
    served_We0 = [r[:] for r in served.We]  # a loaded deep encoder must be FROZEN under train_step
    for _ in range(30):
        served.train_step(xor_in[1], "HOLD", xor_in[2])
    enc_frozen = served.We == served_We0 and served.enc_hidden[0] == W1x

    print("=" * 78)
    print(
        "JEPA WORLD MODEL -- minimal linear latent world model (encoder + predictor + EMA target + VICReg)"
    )
    print("=" * 78)
    print(
        f"world       : hidden u in R^{K}, {len(ACTIONS)} actions {ACTIONS}, obs [0,1]^{IN_DIM} banded RBF"
    )
    print(f"train       : {STEPS} steps, latent_dim={LATENT}, lr={LR}, tau={full.tau}")
    print(
        f"(1) collapse: full latent std {full_std:.4f} vs control (ablation) {ctrl_std:.4f} "
        f"[target tied to online in control: {tied}]"
    )
    print(
        f"(2a) latent : JEPA next-latent MSE {mse_j_lat:.4f} < identity-predictor MSE {mse_id_lat:.4f}"
    )
    print(
        f"(2b) vs 1-D : JEPA->u MSE {mse_j_u:.4f} < scalar ForwardModel->u MSE {mse_f_u:.4f} (common target u)"
    )
    print(
        f"(3) rollout : 40-step rollout bounded={roll_bounded} (max |z_i| {roll_max:.3f} < 10) for all actions"
    )
    print(
        "(3b) e-prop : backward signal is a BROADCAST error -- independent of Wz "
        "(no weight transport, no BPTT)"
    )
    print(f"(4) plan    : JEPA terminal values { {a: round(v, 3) for a, v in jepa_val.items()} }")
    print(
        f"              JEPA best action {ACTIONS[jepa_best]} (gap OPEN-CLOSE {jepa_gap:+.4f}); "
        f"scalar OPEN==CLOSE gap {fm_gap:.2e} (indifferent)"
    )
    print(
        f"              plan() with JEPA commits option {plan_goal} = {ACTIONS[plan_goal]} "
        f"(deterministic: {plan_det})"
    )
    print(
        f"determinism : two seed=7 models byte-identical: {same_weights}; save/load round-trip: {roundtrip}"
    )
    print(
        f"(5) hidden  : deep 2-layer encoder separates XOR {deep_acc}/4 vs single-layer control {lin_acc}/4; "
        f"snn_jepa/2 round-trip={v2_roundtrip}, frozen under train_step={enc_frozen}"
    )
    print("=" * 78)

    # ---- (1) FAIR CONTROL: the ablation collapses; the full model preserves latent variance ----
    assert tied, (
        "with anti_collapse=False the target encoder must be tied to the online encoder every step"
    )
    assert ctrl_std < 0.05 or ctrl_std < 0.30 * full_std, (
        f"the control (no VICReg, no EMA/stop-grad asymmetry) must collapse (std {ctrl_std:.4f} vs full {full_std:.4f})"
    )
    assert full_std > 0.05 and full_std > 3.0 * ctrl_std, (
        f"the full model must preserve latent variance well above the collapsed control ({full_std:.4f} vs {ctrl_std:.4f})"
    )

    # ---- (2) BASELINES (not shuffled targets): beat identity and the 1-D ForwardModel ----
    assert mse_j_lat < mse_id_lat, (
        f"JEPA next-latent prediction must beat the identity predictor ({mse_j_lat:.4f} vs {mse_id_lat:.4f})"
    )
    assert mse_j_u < mse_f_u, (
        f"JEPA must predict the 2-factor next-state better than the 1-D scalar ForwardModel ({mse_j_u:.4f} vs {mse_f_u:.4f})"
    )

    # ---- (3) BOUNDED ROLLOUT: the spectral bound + clamp keep a 40-step rollout finite and bounded ----
    assert roll_bounded, (
        f"a 40-step rollout must stay finite and bounded for every action (max |z_i| {roll_max:.3f})"
    )

    # ---- (3b) E-PROP: no backward path may read the forward weights (no transport, no BPTT) ------
    # The RED is constructed and RUN, not asserted into existence: perturb Wz -- the only weight a
    # transposed backward path could read -- and require the backward signal to be UNCHANGED. Measured
    # against the pre-fix `gp + Wz^T gp` form at this exact seed/probe: eprop_dep = 4.870e+00, assert
    # FIRES. Against the shipped broadcast form: 0.000e+00, passes. A broadcast signal cannot move.
    ep = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, seed=11)
    gp_probe = [0.1 * (i + 1) for i in range(LATENT)]
    gz_before = ep._feedback_gz(gp_probe)
    ep.Wz = [[0.35 for _ in range(LATENT)] for _ in range(LATENT)]  # a large, deliberate change
    gz_after = ep._feedback_gz(gp_probe)
    eprop_dep = max(abs(a - b) for a, b in zip(gz_before, gz_after))
    assert eprop_dep == 0.0, (
        f"e-prop: the backward signal must NOT depend on the forward weights (weight transport); "
        f"perturbing Wz moved it by {eprop_dep:.2e}"
    )

    # ---- (4) LATENT > SCALAR: JEPA reaches the 2-factor goal; the scalar model is provably indifferent ----
    assert jepa_best == OPEN, (
        f"the JEPA plan must pick RELEASE, the only action reaching the (high u0, low u1) goal (got {ACTIONS[jepa_best]})"
    )
    assert jepa_gap > 1e-3, (
        f"the JEPA must clearly prefer OPEN over the CLOSE decoy in latent space (gap {jepa_gap:+.4f})"
    )
    assert fm_gap < 1e-9, (
        f"the 1-D scalar model must be provably indifferent between OPEN and CLOSE (gap {fm_gap:.2e})"
    )
    assert plan_goal == OPEN and plan_det, (
        f"PrefrontalExecutive.plan() driven by the JEPA must deterministically commit RELEASE (got {plan_goal})"
    )

    # ---- (5) HIDDEN ENCODER (snn_jepa/2): a served deep encoder represents XOR a single layer cannot ----
    assert deep_acc == 4, (
        f"a hidden (2-layer) deep encoder must make XOR linearly separable ({deep_acc}/4)"
    )
    assert lin_acc < 4, (
        f"the single-layer control must NOT separate XOR (fair ablation) ({lin_acc}/4)"
    )
    assert v2_roundtrip, "snn_jepa/2 save/load must round-trip the deep encoder exactly"
    assert enc_frozen, (
        "a loaded deep encoder must stay FROZEN under train_step (only the predictor adapts)"
    )

    # ---- (6) REWARD-SHAPING: task-align the latent where pure SSL collapses a few repeating contexts ----
    # Pure SSL (invariance + VICReg) drives a small set of repeating contexts TOGETHER (the online arm latent
    # measured ~0 apart across distinct contexts -> no value head can separate them, the v3.18 finding). A
    # reward-predictive auxiliary head pulls them APART by reward structure. Fair control: SAME seed / stream /
    # steps -- the ONLY difference is reward_shaping. 3 distinct banded contexts, optimal action = the band.
    _band = IN_DIM // 3
    _rsx = [[1.0 if (i // _band) == d else 0.05 for i in range(IN_DIM)] for d in range(3)]
    _rstr = [(_rsx[d], d) for d in range(3)]

    def _rs_run(shaped):
        j = JepaWorldModel(
            ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, lr=LR, seed=1, reward_shaping=shaped
        )
        rng = random.Random(4)
        prev = None
        for _ in range(600):  # online, OBSERVED-reward-only (like the arm)
            p, opt = _rstr[rng.randrange(3)]
            a = rng.randrange(len(ACTIONS))
            r = 1.0 if a == opt else -1.0
            if prev is not None:
                j.train_step(prev[0], prev[1], p, reward=prev[2])
            prev = (p, a, r)
        zs = [j.encode(p) for p, _ in _rstr]
        sp = max(
            math.sqrt(sum((zs[i][k] - zs[m][k]) ** 2 for k in range(LATENT)))
            for i in range(3)
            for m in range(i + 1, 3)
        )
        wr = _fit_linear(
            [j.encode(p) for p, _ in _rstr],
            [[1.0 if a == o else -1.0 for a in range(len(ACTIONS))] for _, o in _rstr],
            len(ACTIONS),
            300,
            0.1,
            seed=0,
        )
        acc = sum(
            1
            for p, o in _rstr
            if max(range(len(ACTIONS)), key=lambda a: dot(wr[a], j.encode(p))) == o
        ) / len(_rstr)
        return sp, acc, j

    rs_plain_sp, rs_plain_acc, _jp = _rs_run(False)
    rs_shaped_sp, rs_shaped_acc, _js = _rs_run(True)
    rs_off_none = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, seed=3).Wr is None
    # byte-identity when OFF: the reward head is drawn ONLY when enabled, so a plain model's weights are unchanged
    rs_bytesame = (
        JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, seed=3).We
        == JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, seed=3).We
    )
    # determinism of the reward-shaped path
    _jd = _rs_run(True)[2]
    rs_det = _js.We == _jd.We and _js.Wr == _jd.Wr

    print(
        f"(6) reward  : pure-SSL few-context spread {rs_plain_sp:.3f} (readout {rs_plain_acc:.2f}) -> "
        f"reward-shaped {rs_shaped_sp:.3f} (readout {rs_shaped_acc:.2f}); head off => Wr None {rs_off_none}"
    )
    print("=" * 78)

    # ---- (6) REWARD-SHAPING: the reward head task-aligns the few-context latent and separates it
    #      MORE than pure SSL does. Until 2026-09-05 this block also asserted `rs_plain_sp < 0.1`
    #      ("pure SSL must collapse a few repeating contexts") and `rs_shaped_acc > rs_plain_acc + 0.3`.
    #      The arc's panel corrected the first draft of this comment: the spread criterion did NOT go
    #      red (pure SSL at the new defaults still maps these three repeating contexts within 0.088 of
    #      each other, printed above -- a standing limitation of the defaults on a FEW-context stream,
    #      distinct from the varying-input collapse the arc fixed); it was removed because it asserted
    #      the defect's PRESENCE, which no shipped criterion may require. The "+0.3" readout contrast
    #      is what went red (both readouts now recover the rule, 1.00 vs 1.00). What survives: the head
    #      separates by a margin over plain SSL, and a readout recovers the rule from its latent -- the
    #      latter a REGRESSION GUARD (the plain arm reads 1.00 too; it does not discriminate the head).
    assert rs_shaped_sp > rs_plain_sp + 0.3, (
        f"reward-shaping must separate the few contexts MORE than pure SSL "
        f"(spread {rs_shaped_sp:.3f} vs {rs_plain_sp:.3f})"
    )
    assert rs_shaped_sp > 0.4, (
        f"reward-shaping must separate the contexts (spread {rs_shaped_sp:.3f})"
    )
    assert rs_shaped_acc > 0.9, (  # regression guard; the plain arm also reads 1.00 at the defaults
        f"a linear readout must recover the rule from the reward-shaped latent ({rs_shaped_acc:.2f})"
    )
    assert rs_off_none and rs_bytesame, (
        "reward_shaping OFF must draw no reward head (Wr None) and leave weights byte-identical"
    )
    assert rs_det, "the reward-shaped training path must be deterministic (same seed + stream)"

    # ---- DETERMINISM + persistence ----
    assert same_weights, (
        "two JepaWorldModel(seed=7) trained on the same stream must have byte-identical weights"
    )
    assert roundtrip, "save/load must round-trip the weights and predictions exactly"

    print(
        f"self-check OK: control collapses (std {ctrl_std:.3f}) while full preserves variance ({full_std:.3f}); "
        f"JEPA beats identity ({mse_j_lat:.3f}<{mse_id_lat:.3f}) and 1-D scalar ({mse_j_u:.3f}<{mse_f_u:.3f});"
    )
    print(
        f"  40-step rollout bounded (max |z_i| {roll_max:.2f}); latent>scalar via plan() -> commits "
        f"{ACTIONS[plan_goal]} (scalar OPEN==CLOSE gap {fm_gap:.1e}); deterministic + exact save/load;"
    )
    print(
        f"  hidden encoder (snn_jepa/2) separates XOR {deep_acc}/4 vs single-layer {lin_acc}/4 (served + frozen);"
    )
    print(
        f"  reward-shaping task-aligns few-context latents (spread {rs_plain_sp:.3f}->{rs_shaped_sp:.3f}, "
        f"readout {rs_plain_acc:.2f}->{rs_shaped_acc:.2f}) beyond pure SSL's"
    )


if __name__ == "__main__":
    main()
