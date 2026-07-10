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
       L_var = mean_j max(0, gamma - sqrt(var_j + eps))            (gamma=1.0: keep each dim's batch std up)
       L_cov = sum_{i!=j} cov_ij^2 / latent_dim                    (decorrelate the latent dimensions)
     Closed-form gradients of L_var, L_cov w.r.t. We, be are backpropagated by hand through unit(tanh(.)).
  6. EMA target       Wt <- tau*Wt + (1-tau)*We,  bt likewise      (slow target = the stop-grad asymmetry)

Spectral / rollout bound. The residual predictor z' = z + Wz@z + ... can diverge over the long rollouts
plan() performs (horizon up to 40). Wz is initialized small, decayed, and row-scaled every step so its
operator-norm proxy (max row-abs-sum) stays < 1 (contractive), and predict() applies a final clamp so every
|z'_i| stays bounded -- main() asserts a 40-step rollout stays finite and bounded for every action.

Fair control (un-riggable). anti_collapse=False is the ABLATION: it disables BOTH the VICReg terms
(lambda_var, lambda_cov treated as 0) AND the EMA/stop-grad asymmetry (the target encoder is tied to the
online encoder every step and the invariance target receives no stop-grad protection), so the control is
GUARANTEED to be able to collapse -- and main() shows it does (its per-dim latent std goes to ~0), while the
full model preserves latent variance.

Refs: LeCun, "A Path Towards Autonomous Machine Intelligence" (2022, the JEPA proposal); Assran et al.,
"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" / I-JEPA (2023);
Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance Regularization" (2022).

Pure stdlib (math, random, json), ASCII-only prints, deterministic (random.Random(seed) per instance).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.jepa_world_model
"""
import json
import math
import random

from snn_data_lake.substrate._mathutils import dot, mean, unit


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

    def __init__(self, actions, in_dim=64, latent_dim=16, tau=0.99, lr=0.05,
                 lambda_inv=1.0, lambda_var=1.0, lambda_cov=1.0, buffer=32,
                 anti_collapse=True, seed=0):
        self.actions = tuple(actions)                 # ordered action names, e.g. ("GRIPPER_CLOSE", ...)
        self.n_act = len(self.actions)
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        self.in_dim = int(in_dim)
        self.L = int(latent_dim)
        self.tau = float(tau)                          # EMA target coefficient (~1 -> slow target)
        self.lr = float(lr)
        self.lambda_inv = float(lambda_inv)            # weight of the JEPA invariance (prediction) loss
        self.lambda_var = float(lambda_var)            # weight of the VICReg variance hinge
        self.lambda_cov = float(lambda_cov)            # weight of the VICReg covariance decorrelation
        self.buffer = int(buffer)                      # ring-buffer size for the VICReg batch statistics
        self.anti_collapse = bool(anti_collapse)       # False == ablation: no VICReg + target tied to online
        self.seed = seed
        self.rng = random.Random(seed)                 # per-instance rng; identical seed -> identical weights

        # ---- fixed internal constants (not exposed on the contract) ----
        self.gamma = 1.0                               # VICReg target std (hinge threshold)
        self.eps = 1e-4                                # VICReg variance floor
        self.z_clip = 3.0                              # predicted-latent clamp -> bounded rollout
        self.rho_max = 0.9                             # max row-abs-sum of Wz (contractive spectral proxy)
        self.wd = 1e-3                                 # tiny weight decay on Wz each step
        self.lambda_roll = 0.3                         # weight of the optional 2-step rollout consistency term
        self.use_rollout = True                        # fight teacher-forcing drift with a 2-step term

        # ---- parameter init (all draws from the per-instance rng, so seed => byte-identical weights) ----
        se = 1.0 / math.sqrt(self.in_dim)              # encoder scale (fan-in)
        self.We = [[self.rng.uniform(-1.0, 1.0) * se for _ in range(self.in_dim)] for _ in range(self.L)]
        self.be = [0.0] * self.L
        self.Wz = [[self.rng.uniform(-1.0, 1.0) * 0.02 for _ in range(self.L)] for _ in range(self.L)]  # SMALL
        self.Wa = [[self.rng.uniform(-1.0, 1.0) * 0.05 for _ in range(self.n_act)] for _ in range(self.L)]
        self.b = [0.0] * self.L
        # EMA target starts as a copy of the online encoder (standard); ablation re-ties it every step
        self.Wt = [row[:] for row in self.We]
        self.bt = self.be[:]

        self._buf = []                                 # ring buffer of recent online latents (VICReg + telemetry)
        self._prev = None                              # (prev online latent, prev action index) for 2-step term

    # ---- action handling -------------------------------------------------------------------------------
    def _idx(self, command):
        """Map an action name or an integer index to an action index (duck-typed command for predict/plan)."""
        if isinstance(command, str):
            return self._aidx[command]
        return int(command)

    # ---- encoders --------------------------------------------------------------------------------------
    def encode(self, pattern):
        """Online encoder: z = unit(tanh(We @ pattern + be)). Returns a length-latent_dim unit vector."""
        return unit([math.tanh(dot(self.We[i], pattern) + self.be[i]) for i in range(self.L)])

    def target_encode(self, pattern):
        """EMA target encoder (Wt, bt); same unit(tanh(.)) form; used with a stop-gradient as the JEPA target."""
        return unit([math.tanh(dot(self.Wt[i], pattern) + self.bt[i]) for i in range(self.L)])

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
    def train_step(self, pattern_t, action_t, pattern_next):
        """One online JEPA update; returns the scalar total loss.

        Loss = lambda_inv * invariance + lambda_var * L_var + lambda_cov * L_cov (+ a small 2-step rollout term).
        invariance = ||predict(encode(x_t), a_t) - sg(target_encode(x_{t+1}))||^2. Updates We, be, Wz, Wa, b by
        closed-form gradients (backprop through unit(tanh(.)) by hand), row-scales Wz to stay contractive, then
        EMA-updates the target encoder. Pushes encode(x_t) into the VICReg ring buffer. In the ablation
        (anti_collapse=False) VICReg is off and the target is tied to the online encoder with no stop-grad."""
        L, D = self.L, self.in_dim
        x, y = pattern_t, pattern_next
        aj = self._idx(action_t)

        # -- forward: online encode of x, keeping pre-activation h, tanh t, and norm n for the backprop --
        h = [dot(self.We[i], x) + self.be[i] for i in range(L)]
        t = [math.tanh(v) for v in h]
        n = math.sqrt(sum(c * c for c in t)) or 1.0
        z = [c / n for c in t]                          # online latent = unit(tanh(...))

        self._buf.append(z[:])                          # telemetry + VICReg batch (ring buffer)
        if len(self._buf) > self.buffer:
            self._buf.pop(0)

        # -- target latent of the next observation --
        if self.anti_collapse:
            zt = self.target_encode(y)                  # EMA target, stop-gradient
            ty = ny = None
        else:
            # ablation: target tied to the ONLINE encoder, and NO stop-grad (gradient flows into it too) so the
            # two encodings are pulled together with nothing holding variance up -> guaranteed collapse.
            hy = [dot(self.We[i], y) + self.be[i] for i in range(L)]
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
            gWa[i][aj] += gp[i]                          # onehot picks column aj
            gi = gp[i]
            row = gWz[i]
            for k in range(L):
                row[k] += gi * z[k]
        # invariance gradient into z:  dL/dz = gp + Wz^T gp   (from p = (I + Wz) z + const)
        gz = [0.0] * L
        for k in range(L):
            s = gp[k]
            for i in range(L):
                s += gp[i] * self.Wz[i][k]
            gz[k] = s

        # -- VICReg gradients into z (full model only); buffer supplies the batch statistics --
        Lvar = 0.0
        Lcov = 0.0
        if self.anti_collapse and len(self._buf) >= 2:
            m = len(self._buf)
            mu = [mean([bvec[j] for bvec in self._buf]) for j in range(L)]
            cen = [[bvec[j] - mu[j] for j in range(L)] for bvec in self._buf]
            var = [sum(cen[p_][j] * cen[p_][j] for p_ in range(m)) / m for j in range(L)]
            # variance hinge: keep each dim's batch std up towards gamma (blocks collapse to a constant)
            for j in range(L):
                sd = math.sqrt(var[j] + self.eps)
                Lvar += max(0.0, self.gamma - sd)
                if self.gamma - sd > 0.0:               # hinge active -> push var up (move z away from the mean)
                    gz[j] += self.lambda_var * (-(z[j] - mu[j]) / (L * m * sd))
            Lvar /= L
            # covariance: decorrelate the off-diagonal latent dimensions
            cov = [[sum(cen[p_][k] * cen[p_][j] for p_ in range(m)) / m for j in range(L)] for k in range(L)]
            for k in range(L):
                gk = 0.0
                for j in range(L):
                    if j != k:
                        Lcov += cov[k][j] * cov[k][j]
                        gk += cov[k][j] * (z[j] - mu[j])
                gz[k] += self.lambda_cov * (4.0 / (L * m)) * gk
            Lcov /= L

        # -- backprop the combined gz through z = unit(tanh(h)) into We, be --
        self._encoder_backprop(gz, t, n, x, gWe, gbe)

        # -- ablation only: also backprop -gp through the (untied, no-stop-grad) target encode(y) --
        if not self.anti_collapse:
            self._encoder_backprop([-gp[i] for i in range(L)], ty, ny, y, gWe, gbe)

        # -- optional 2-step rollout consistency term (predictor params only; robust, skipped on first call) --
        Lroll = 0.0
        if self.use_rollout and self._prev is not None:
            z0, a0 = self._prev
            w = [z0[i] + dot(self.Wz[i], z0) + self.Wa[i][a0] + self.b[i] for i in range(L)]     # inner predict
            p2 = [w[i] + dot(self.Wz[i], w) + self.Wa[i][aj] + self.b[i] for i in range(L)]      # outer predict
            r2 = [p2[i] - zt[i] for i in range(L)]      # 2-step (a0 then a_t) should reach the same next latent
            Lroll = sum(v * v for v in r2)
            gp2 = [2.0 * self.lambda_roll * r2[i] for i in range(L)]
            for i in range(L):                          # outer layer
                gb[i] += gp2[i]
                gWa[i][aj] += gp2[i]
                for k in range(L):
                    gWz[i][k] += gp2[i] * w[k]
            gw = [0.0] * L                              # dL/dw = gp2 + Wz^T gp2
            for k in range(L):
                s = gp2[k]
                for i in range(L):
                    s += gp2[i] * self.Wz[i][k]
                gw[k] = s
            for i in range(L):                          # inner layer
                gb[i] += gw[i]
                gWa[i][a0] += gw[i]
                for k in range(L):
                    gWz[i][k] += gw[i] * z0[k]

        # -- SGD update --
        lr = self.lr
        for i in range(L):
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

        # -- spectral bound on Wz: small weight decay + row-scaling so max row-abs-sum stays < 1 (contractive) --
        for i in range(L):
            for k in range(L):
                self.Wz[i][k] *= (1.0 - self.wd)
        rowmax = max(sum(abs(self.Wz[i][k]) for k in range(L)) for i in range(L))
        if rowmax > self.rho_max:
            f = self.rho_max / rowmax
            for i in range(L):
                for k in range(L):
                    self.Wz[i][k] *= f

        # -- target encoder update: EMA (full) or hard tie to online (ablation guarantees target == online) --
        if self.anti_collapse:
            tau = self.tau
            for i in range(L):
                for k in range(D):
                    self.Wt[i][k] = tau * self.Wt[i][k] + (1.0 - tau) * self.We[i][k]
                self.bt[i] = tau * self.bt[i] + (1.0 - tau) * self.be[i]
        else:
            self.Wt = [row[:] for row in self.We]
            self.bt = self.be[:]

        self._prev = (z[:], aj)                          # detached latent for the next 2-step term
        return self.lambda_inv * loss_inv + self.lambda_var * Lvar + self.lambda_cov * Lcov \
            + (self.lambda_roll * Lroll if self.use_rollout else 0.0)

    def _encoder_backprop(self, gz, t, n, x, gWe, gbe):
        """Chain dL/dz (=gz) through z = unit(tanh(h)) to accumulate dL/dWe, dL/dbe.

        unit Jacobian: dz_i/dt_j = delta_ij/n - t_i t_j / n^3 ; tanh': dt_j/dh_j = 1 - t_j^2 ; h = We@x + be."""
        L, D = self.L, self.in_dim
        sdot = sum(gz[i] * t[i] for i in range(L))      # gz . t (the unit-normalization coupling term)
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
        """Mean over dimensions of the per-dimension std of the current buffer of online latents.

        This is the collapse telemetry: a collapsed encoder maps every input to the same latent, so every
        dimension's batch std -> 0 and this number -> 0."""
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
        """Persist the weights as tagged JSON ({"format": "snn_jepa/1", ...}). Floats round-trip exactly."""
        obj = {
            "format": "snn_jepa/1",
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
            "We": self.We, "be": self.be,
            "Wz": self.Wz, "Wa": self.Wa, "b": self.b,
            "Wt": self.Wt, "bt": self.bt,
        }
        with open(path, "w", encoding="ascii") as f:
            json.dump(obj, f)

    def load_weights(self, path):
        """Load weights from tagged JSON with a trust-boundary check: validate the format tag and every shape,
        raising ValueError on any mismatch (never trust an unvalidated weight file)."""
        with open(path, encoding="ascii") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or obj.get("format") != "snn_jepa/1":
            raise ValueError("bad JEPA weight file: missing or incorrect 'format' tag (expected snn_jepa/1)")
        L = int(obj.get("latent_dim", -1))
        D = int(obj.get("in_dim", -1))
        na = len(obj.get("actions", []))
        We, be = obj.get("We"), obj.get("be")
        Wz, Wa, b = obj.get("Wz"), obj.get("Wa"), obj.get("b")
        Wt, bt = obj.get("Wt"), obj.get("bt")

        def _shape_ok(M, rows, cols):
            return isinstance(M, list) and len(M) == rows and all(
                isinstance(row, list) and len(row) == cols for row in M)

        if not (L > 0 and D > 0 and na > 0):
            raise ValueError("bad JEPA weight file: non-positive dimensions")
        if not (_shape_ok(We, L, D) and _shape_ok(Wz, L, L) and _shape_ok(Wa, L, na)):
            raise ValueError("bad JEPA weight file: matrix shape mismatch")
        if not all(isinstance(v, list) and len(v) == L for v in (be, b)):
            raise ValueError("bad JEPA weight file: bias length mismatch")
        # the EMA target encoder is OPTIONAL: a torch export (research/jepa_scaffold_torch.py) ships only
        # the online encoder + predictor, so default the target to a copy of the online encoder (target
        # starts tied to online -- correct for serving/inference, which never uses train_step's EMA target).
        if Wt is None or bt is None:
            Wt, bt = [row[:] for row in We], be[:]
        elif not (_shape_ok(Wt, L, D) and isinstance(bt, list) and len(bt) == L):
            raise ValueError("bad JEPA weight file: target-encoder shape mismatch")
        # trusted: adopt the validated weights and geometry
        self.actions = tuple(obj["actions"])
        self.n_act = na
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        self.in_dim, self.L = D, L
        self.tau = float(obj["tau"])
        self.lr = float(obj["lr"])
        self.lambda_inv = float(obj["lambda_inv"])
        self.lambda_var = float(obj["lambda_var"])
        self.lambda_cov = float(obj["lambda_cov"])
        self.buffer = int(obj["buffer"])
        self.anti_collapse = bool(obj["anti_collapse"])
        self.We = [row[:] for row in We]
        self.be = be[:]
        self.Wz = [row[:] for row in Wz]
        self.Wa = [row[:] for row in Wa]
        self.b = b[:]
        self.Wt = [row[:] for row in Wt]
        self.bt = bt[:]
        self._buf = []
        self._prev = None


# =========================================================================================================
# Self-check: a synthetic ACTION-CONDITIONED world. A hidden latent u in R^2 evolves under 3 discrete actions
# with distinct band-structured dynamics; observations are a banded [0,1]^in_dim RBF rendering of u + noise.
# =========================================================================================================

# Balanced 2-factor dynamics (sum of each column is 0 -> a mean-zero random walk that explores [0,1]^2).
# GRIPPER_OPEN and GRIPPER_CLOSE move u0 IDENTICALLY (+d) but u1 OPPOSITELY -- a scalar readout of u0 cannot
# tell them apart, yet the 2-D latent can. HOLD moves u0 the other way (-2d) to balance the walk.
_EFFECTS = {
    "GRIPPER_OPEN": (+1.0, -1.0),
    "GRIPPER_CLOSE": (+1.0, +1.0),
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
        pat.extend(encoders[i].activations(u[i]))       # continuous Gaussian activations in [0,1]
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
    """Fit W (out_dim x feat_dim) mapping feats -> targets by plain LMS (no matrix inverse, pure stdlib)."""
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
    from snn_data_lake.cortex.population_encoding import PopulationEncoder
    from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive

    IN_DIM, K, LATENT = 64, 2, 16
    BAND = IN_DIM // K                                   # 32 neurons per factor
    D, NOISE = 0.20, 0.02
    ACTIONS = ("GRIPPER_CLOSE", "GRIPPER_OPEN", "HOLD")
    OPEN = 1                                             # index of GRIPPER_OPEN (the goal action) in ACTIONS
    STEPS, LR = 1200, 0.08
    LV, LC = 20.0, 2.0                                   # VICReg weights (default is 1.0; the unit-sphere latent
    #                                                      caps per-dim energy, so the anti-collapse pressure is
    #                                                      scaled up here for both models to visibly hold variance)
    encoders = [PopulationEncoder(n=BAND, lo=0.0, hi=1.0, max_spikes=10, overlap=1.0) for _ in range(K)]
    centers0 = encoders[0].centers

    # ---- train the full JEPA and the fair control on the SAME seeded stream / steps / architecture ----
    train, _ = _gen_stream(STEPS, 1, encoders, D, NOISE, ACTIONS)
    full = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, lr=LR,
                          lambda_var=LV, lambda_cov=LC, anti_collapse=True, seed=0)
    ctrl = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, lr=LR,
                          lambda_var=LV, lambda_cov=LC, anti_collapse=False, seed=0)
    for (x, a, y) in train:
        full.train_step(x, a, y)
    for (x, a, y) in train:
        ctrl.train_step(x, a, y)
    full_std, ctrl_std = full.latent_std(), ctrl.latent_std()
    # by construction the ablation ties the target encoder to the online encoder every step
    tied = (ctrl.Wt == ctrl.We and ctrl.bt == ctrl.be)

    # ---- (2) BASELINES on HELD-OUT episodes of the same world ----
    hold, hidden = _gen_stream(350, 2, encoders, D, NOISE, ACTIONS)
    # (2a) JEPA next-latent MSE beats an IDENTITY predictor (predict(z,a)=z), in the SAME latent space
    err_j = [sum((full.predict(full.encode(x), a)[i] - full.target_encode(y)[i]) ** 2 for i in range(LATENT))
             for (x, a, y) in hold]
    err_id = [sum((full.encode(x)[i] - full.target_encode(y)[i]) ** 2 for i in range(LATENT))
              for (x, a, y) in hold]
    mse_j_lat, mse_id_lat = mean(err_j), mean(err_id)

    # (2b) JEPA beats a 1-D linear ForwardModel on a scalar projection -- fair common target: predict next u.
    CMD = {"GRIPPER_OPEN": 1.0, "GRIPPER_CLOSE": 1.0, "HOLD": -1.0}   # OPEN/CLOSE share a command: the scalar
    fm = ForwardModel(a=0.0, b=0.0, dt=1.0, seed=0)                   # readout of u0 cannot distinguish them
    for _ep in range(4):
        for (x, a, y) in train:
            fm.update(_band0_centroid(x, centers0, BAND), CMD[a], _band0_centroid(y, centers0, BAND), lr=0.15)
    # JEPA feature -> u readout, and ForwardModel feature -> u readout, both LMS-fit on the training stream
    _, train_hidden = _gen_stream(STEPS, 1, encoders, D, NOISE, ACTIONS)   # same seed -> same trajectory
    jf = [full.predict(full.encode(x), a) for (x, a, _y) in train]
    ff = [[fm.predict(_band0_centroid(x, centers0, BAND), CMD[a]), 1.0] for (x, a, _y) in train]
    ut = [un for (_u, _a, un) in hidden]                             # held-out hidden next-state targets
    utr = [un for (_u, _a, un) in train_hidden]                      # training hidden next-state targets
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
    # Start (0.40, 0.50); goal (u0 high, u1 low) -> reachable only by GRIPPER_OPEN (u0 up, u1 down). CLOSE is a
    # decoy (u0 up, u1 UP) that a u0-only scalar cannot distinguish from OPEN.
    rng_g = random.Random(21)
    start_pat = _render([0.40, 0.50], encoders, rng_g, NOISE)
    goal_pat = _render([0.85, 0.10], encoders, rng_g, NOISE)
    z_start, z_goal = full.encode(start_pat), full.encode(goal_pat)
    gv = full.make_goal_value(z_goal)
    HZ = 4

    def roll_val(pred, s0, cmd, horizon, value):        # terminal value of a horizon-step rollout
        s = s0
        for _ in range(horizon):
            s = pred.predict(s, cmd)
        return value(s)

    jepa_val = {a: roll_val(full, z_start, i, HZ, gv) for i, a in enumerate(ACTIONS)}
    jepa_best = max(range(len(ACTIONS)), key=lambda i: jepa_val[ACTIONS[i]])
    jepa_gap = jepa_val["GRIPPER_OPEN"] - jepa_val["GRIPPER_CLOSE"]

    # the scalar ForwardModel: state = u0 centroid, goal = u0 centroid of the goal; OPEN and CLOSE share a
    # command, so its rollout values for them are PROVABLY identical -> it cannot represent the task.
    s_start = _band0_centroid(start_pat, centers0, BAND)
    s_goal = _band0_centroid(goal_pat, centers0, BAND)
    scal_gv = (lambda s: -(s - s_goal) ** 2)
    fm_val = {a: roll_val(fm, s_start, CMD[a], HZ, scal_gv) for a in ACTIONS}
    fm_gap = abs(fm_val["GRIPPER_OPEN"] - fm_val["GRIPPER_CLOSE"])

    # drive PrefrontalExecutive.plan() with the JEPA as its (duck-typed) predictor
    pfc = PrefrontalExecutive(n_options=len(ACTIONS), seed=5)
    plan_goal, _rt = pfc.plan(z_start, [0, 1, 2], full, gv, horizon=HZ)
    plan_goal2, _rt2 = PrefrontalExecutive(n_options=len(ACTIONS), seed=5).plan(z_start, [0, 1, 2], full, gv,
                                                                                horizon=HZ)
    plan_det = (plan_goal == plan_goal2)

    # ---- DETERMINISM: two seed=7 models on the same stream -> byte-identical weights ----
    det_a = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, lr=LR, seed=7)
    det_b = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, lr=LR, seed=7)
    for (x, a, y) in train[:200]:
        det_a.train_step(x, a, y)
        det_b.train_step(x, a, y)
    same_weights = (det_a.We == det_b.We and det_a.be == det_b.be and det_a.Wz == det_b.Wz
                    and det_a.Wa == det_b.Wa and det_a.b == det_b.b and det_a.Wt == det_b.Wt
                    and det_a.bt == det_b.bt)

    # ---- SAVE / LOAD round-trip (exact) ----
    import os
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "snn_jepa_roundtrip.json")
    full.save_weights(tmp)
    loaded = JepaWorldModel(ACTIONS, in_dim=IN_DIM, latent_dim=LATENT, seed=99)
    loaded.load_weights(tmp)
    probe = _render([0.3, 0.7], encoders, random.Random(4), NOISE)
    roundtrip = (loaded.We == full.We and loaded.Wz == full.Wz and loaded.Wa == full.Wa
                 and loaded.b == full.b and loaded.Wt == full.Wt
                 and loaded.encode(probe) == full.encode(probe)
                 and loaded.predict(loaded.encode(probe), "HOLD") == full.predict(full.encode(probe), "HOLD"))
    os.remove(tmp)

    print("=" * 78)
    print("JEPA WORLD MODEL -- minimal linear latent world model (encoder + predictor + EMA target + VICReg)")
    print("=" * 78)
    print(f"world       : hidden u in R^{K}, {len(ACTIONS)} actions {ACTIONS}, obs [0,1]^{IN_DIM} banded RBF")
    print(f"train       : {STEPS} steps, latent_dim={LATENT}, lr={LR}, tau={full.tau}")
    print(f"(1) collapse: full latent std {full_std:.4f} vs control (ablation) {ctrl_std:.4f} "
          f"[target tied to online in control: {tied}]")
    print(f"(2a) latent : JEPA next-latent MSE {mse_j_lat:.4f} < identity-predictor MSE {mse_id_lat:.4f}")
    print(f"(2b) vs 1-D : JEPA->u MSE {mse_j_u:.4f} < scalar ForwardModel->u MSE {mse_f_u:.4f} (common target u)")
    print(f"(3) rollout : 40-step rollout bounded={roll_bounded} (max |z_i| {roll_max:.3f} < 10) for all actions")
    print(f"(4) plan    : JEPA terminal values { {a: round(v, 3) for a, v in jepa_val.items()} }")
    print(f"              JEPA best action {ACTIONS[jepa_best]} (gap OPEN-CLOSE {jepa_gap:+.4f}); "
          f"scalar OPEN==CLOSE gap {fm_gap:.2e} (indifferent)")
    print(f"              plan() with JEPA commits option {plan_goal} = {ACTIONS[plan_goal]} "
          f"(deterministic: {plan_det})")
    print(f"determinism : two seed=7 models byte-identical: {same_weights}; save/load round-trip: {roundtrip}")
    print("=" * 78)

    # ---- (1) FAIR CONTROL: the ablation collapses; the full model preserves latent variance ----
    assert tied, "with anti_collapse=False the target encoder must be tied to the online encoder every step"
    assert ctrl_std < 0.05 or ctrl_std < 0.30 * full_std, \
        f"the control (no VICReg, no EMA/stop-grad asymmetry) must collapse (std {ctrl_std:.4f} vs full {full_std:.4f})"
    assert full_std > 0.05 and full_std > 3.0 * ctrl_std, \
        f"the full model must preserve latent variance well above the collapsed control ({full_std:.4f} vs {ctrl_std:.4f})"

    # ---- (2) BASELINES (not shuffled targets): beat identity and the 1-D ForwardModel ----
    assert mse_j_lat < mse_id_lat, \
        f"JEPA next-latent prediction must beat the identity predictor ({mse_j_lat:.4f} vs {mse_id_lat:.4f})"
    assert mse_j_u < mse_f_u, \
        f"JEPA must predict the 2-factor next-state better than the 1-D scalar ForwardModel ({mse_j_u:.4f} vs {mse_f_u:.4f})"

    # ---- (3) BOUNDED ROLLOUT: the spectral bound + clamp keep a 40-step rollout finite and bounded ----
    assert roll_bounded, f"a 40-step rollout must stay finite and bounded for every action (max |z_i| {roll_max:.3f})"

    # ---- (4) LATENT > SCALAR: JEPA reaches the 2-factor goal; the scalar model is provably indifferent ----
    assert jepa_best == OPEN, \
        f"the JEPA plan must pick GRIPPER_OPEN, the only action reaching the (high u0, low u1) goal (got {ACTIONS[jepa_best]})"
    assert jepa_gap > 1e-3, \
        f"the JEPA must clearly prefer OPEN over the CLOSE decoy in latent space (gap {jepa_gap:+.4f})"
    assert fm_gap < 1e-9, \
        f"the 1-D scalar model must be provably indifferent between OPEN and CLOSE (gap {fm_gap:.2e})"
    assert plan_goal == OPEN and plan_det, \
        f"PrefrontalExecutive.plan() driven by the JEPA must deterministically commit GRIPPER_OPEN (got {plan_goal})"

    # ---- DETERMINISM + persistence ----
    assert same_weights, "two JepaWorldModel(seed=7) trained on the same stream must have byte-identical weights"
    assert roundtrip, "save/load must round-trip the weights and predictions exactly"

    print(f"self-check OK: control collapses (std {ctrl_std:.3f}) while full preserves variance ({full_std:.3f}); "
          f"JEPA beats identity ({mse_j_lat:.3f}<{mse_id_lat:.3f}) and 1-D scalar ({mse_j_u:.3f}<{mse_f_u:.3f});")
    print(f"  40-step rollout bounded (max |z_i| {roll_max:.2f}); latent>scalar via plan() -> commits "
          f"{ACTIONS[plan_goal]} (scalar OPEN==CLOSE gap {fm_gap:.1e}); deterministic + exact save/load")


if __name__ == "__main__":
    main()
