r"""
Shared NONLINEAR (one-hidden-layer MLP) action-value over the JEPA latent -- the task-aligned
NONLINEAR counterpart of the LINEAR basal_ganglia/latent_q.LatentQ.

WHY BOTH EXIST (Policy 2/4 note -- this pair is a MEASURED tradeoff, not un-retired bloat). MlpQ was
once described as a "strict upgrade" that made LatentQ retirable. Its own self-check refutes that: on a
NON-linear (XOR) rule MlpQ wins decisively (0.993 vs 0.467 ~ chance), but on a LINEARLY-separable rule
it lands very slightly BEHIND the linear Q (0.983 vs 0.993) -- the extra hidden capacity costs a little
where a linear map is already exact. So gate (3) is a NO-REGRESSION tolerance (within 0.05), NOT `>=`,
and neither model dominates the other everywhere. Both stay, selected per task; and note BOTH are
opt-in and default OFF in AutonomousArm (latent_q=False, mlp_q=False) -- the live substrate is the
tabular per-context q, because the head does not separate the encoder's few live contexts on-robot
(the 'near-collapse' explanation was retracted 2026-09-05: specs/022 evidence/jepa-live-floor.md section 5).

Honest framing (read this first). LatentQ is a SHARED LINEAR action-value Q(z, a) = dot(W[a], z) + b[a]:
ONE weight set across all contexts, so a semi-gradient update from one latent transfers along the
latent geometry. But a linear map has a hard ceiling -- it can only separate action-values that are a
LINEAR function of z. An action-value whose optimal action depends NON-LINEARLY on z (e.g. an XOR /
quadrant rule on the sign pattern of two latent dims) is PROVABLY not representable by any linear Q:
no single hyperplane separates the XOR regions, so a linear head is stuck at chance. This module lifts
that ceiling with the smallest thing that clears it -- ONE hidden layer:

    h       = relu(W1 @ z + b1)             # SHARED nonlinear feature layer (W1: hidden x latent_dim)
    Q(z, a) = dot(W2[a], h) + b2[a]         # per-action linear read-out over the shared features

ONE weight set (W1, b1 shared hidden layer; W2, b2 per-action read-out) is shared across ALL contexts,
exactly like LatentQ -- so it keeps the transfer -- but the hidden ReLU layer gives it the nonlinear
features a linear Q lacks. This is textbook NONLINEAR value-function approximation (Sutton & Barto,
ch. 9-10: a nonlinear function approximator for the action-value); the hidden layer is the ONLY
structural change from LatentQ, and the API is identical so it drops into the same wiring.

e-PROP WEIGHT TRANSPORT: **RESOLVED 2026-08-14** (specs/015-resolve-weight-transport). The history
below is kept because the WAY it was resolved is the useful part. (Merge note: lane 017 recorded
this site as an OPEN QUESTION in the policy adjudication and corrected a stale citation here; lane
015 then resolved it. Both contributions are kept below -- the 017 half is the citation paragraph,
the 015 half is the resolution.)

THE VIOLATION, as it stood (audit 2026-08-04). The `dh` line read the FORWARD read-out weights
`W2[a]` transposed to carry error backward into the shared hidden layer. That is WEIGHT TRANSPORT --
structurally the same SHAPE `docs/POLICIES.md` once ruled a real violation at
`cerebellum/jepa_world_model.py`. Hand-writing it in stdlib rather than calling autograd changed
nothing about that ("absence of `.backward()` is NOT evidence of locality"). It was not on the live
tick -- `AutonomousArm` defaults `mlp_q=False` -- but it was one constructor argument away, so it
was latent, not a non-issue.

**THE JEPA SITE IS CLOSED (2026-08-04) and is carried STRUCK THROUGH in that file's e-prop
adjudication -- search the `~~`-wrapped row, NOT a line number.** An earlier pass here wrote
`POLICIES.md:1564` for it and that cite was ALREADY STALE WHEN WRITTEN, because the same lane's
earlier commit had inserted ten lines above it. So the sentence above borrows that site's SHAPE, not
its open status; until 2026-08-13 it read "records at", which pointed a reader at a closed row as if
it were corroboration.

THE RESOLUTION: KOLEN-POLLACK converging feedback (Kolen & Pollack 1994). A separate matrix `B`
carries the error, and `B` receives the SAME increment as the forward read-out, so it TRACKS `W2`
without ever reading it. `err` and `h[j]` are already local to the synapse the read-out step
updates, so nothing is transported. Measured over 8 draws x 30 passes: the learned `B` aligns to
`|cos| in [0.8785, 0.9121]` against the transport gradient on **8 of 8** draws, while the FROZEN
random feedback the earlier analysis tested reaches only [0.1604, 0.3379] and clears the bar on
**0 of 8**. Held-out accuracy is unchanged (XOR 0.993, linear 0.983) -- **no capability was lost**.

NOT SFDFA, and the distinction matters. The owner-proposed Spiking Forward DFA (Bacho & Chu 2024)
estimates output->hidden weights by propagating directional derivatives as SPIKE GRADES during
inference. `MlpQ` is a RATE MLP -- no membrane, no threshold, no spike, no time index in `update` --
so SFDFA's actual contribution has nothing here to attach to. What transferred is its ESSENTIAL
property: feedback that is ESTIMATED and CONVERGES rather than transported or frozen. Naming the
adopted rule "SFDFA" would be a citation defect; it is Kolen-Pollack.

MEASURED, so the fix is a decision rather than a guess. POLICIES prescribes FEEDBACK ALIGNMENT
(Lillicrap 2016) for exactly this shape: replace `W2[a][j]` in `dh` with a FIXED RANDOM feedback
matrix `B[a][j]`, which is local and needs no weight symmetry. The FA twin is a SCRATCH benchmark and
is deliberately not shipped beside `MlpQ` (P2: a second live head for one job is the additive failure),
so its construction is stated here in full and is the only thing a re-derivation needs: `B` is drawn
ONCE per instance from `random.Random(bseed)` as `uniform(-1,1)/sqrt(hidden)` -- the same scale `W2`
uses -- and the ONE changed line is `dh[j] = err*B[i][j]*relu'(pre[j])`; nothing else differs.
Fixtures are main()'s own (400 train / 300 held-out, hidden=32, lr=0.1, 30 passes, model seed 0).
Held-out accuracy, FA swept over EIGHT feedback draws (bseed 1..8) because a single draw is a coin
flip, not a measurement -- the discipline POLICIES already applied to jepa's six `Bz` draws:

    rule                 weight transport   feedback alignment      linear LatentQ
    XOR (non-linear)          0.993          0.990  [0.980-0.997]        0.467
    linearly separable        0.983          0.993  [0.990-0.993]        0.993

The transport and LatentQ columns are main()'s own printed numbers, so they are re-derived on every
gate run. (Corrected 2026-08-04b: this table shipped `0.990/0.990/0.990` on the linear row, which the
self-check twelve lines below contradicts in its own output -- MlpQ measures 0.983 and LatentQ 0.993
there, and the docstring above has said `0.983 vs 0.993` since the module was written. Two cells wrong
in a table whose point was to be a measurement.)

So feedback alignment CLEARS both of this module's shipped gates (>0.85 held-out, >0.30 over the
linear control) on every draw -- it costs ~0.3 points on XOR and GAINS 1.0 on the linear rule. It is
NOT adopted here, and the reason is a measurement rather than caution: the cross-cutting invariant
asks for **cosine > 0.5** against an independent gradient, and the FA hidden-gradient direction misses
that bar in BOTH directions -- swept over the same eight draws, **|cosine| exceeds 0.5 in 0 of 8**.
(The INTERVAL this line used to quote, [-0.472, +0.279], is from specs/015 and the module no longer
produces it; today it prints [-0.4366, +0.1844]. The PROPERTY -- 0 of 8 -- is what the argument
rests on and is what main() re-runs live, so the interval is not restated here.) That sweep is re-run live in main() rather than quoted, because the
quantity is essentially cos(W2[a], B[a]) under the relu mask (`err` cancels), so it is a property of
the DRAW and a single value reported as "the" mean cosine is unreproducible. Swapping would therefore
trade one failed clause of the invariant for another, which is not a fix. Recorded as OPEN work -- the
honest resolutions are (a) accept FA and amend what the cosine clause means for a non-gradient
alignment rule, or (b) retire this head under P2 in favour of the linear `LatentQ`, which has no
hidden layer and therefore no backward pass at all. Deciding that is a policy call, not an edit.

THAT COSINE CANNOT DISCRIMINATE, AND THE REFUTATION ABOVE THEREFORE DOES NOT STAND ON IT
(recorded 2026-08-14, specs/015-resolve-weight-transport, BEFORE any repair so the repair cannot be
retro-justified). The paragraph above already states the quantity: "essentially cos(W2[a], B[a])
under the relu mask (`err` cancels)". `err` is a scalar multiplying both sides, so it divides out,
and what remains is the angle between a FORWARD WEIGHT VECTOR and a FEEDBACK VECTOR. For a FIXED
RANDOM `B` that is near zero BY CONSTRUCTION at any reasonable width -- two independent vectors in
R^32 are near-orthogonal in expectation. So the criterion asks "is the feedback parallel to the
forward weights", which is the WEIGHT SYMMETRY feedback alignment exists to deny. It never looks at
what either rule DOES. An instrument that no alignment rule could ever pass is not evidence against
alignment rules.

THE CONTRAST THAT PROVES IT, and it is in this project's own record: at the jepa site the
same-NAMED criterion measured 0.971 -- far ABOVE the 0.5 bar -- because there both sides were
computed through one shared forward pass, so the identity/skip term dominated both. One criterion
name, two different quantities, opposite verdicts. Establishing what a cosine is computed OVER is
the question, not what it is called. (Same defect class the eprop_kernel lane found, where 81.2% of
an all-parameter cosine comes from the readout block. CORRECTED 2026-08-18 in the same change as
the eprop_kernel site, because a figure living in two places gets fixed in one: the share was 90.7%
before specs/020 T048, and those terms were NOT 'identical by construction' -- that claim was
measurably false (W_out cosine 0.986830, allclose False) and only became true once the reference
was reconciled to the kappa-filtered form the shipped encoders run.)

WHAT THAT PARAGRAPH DID NOT CLAIM: that feedback alignment works here. Only that the number did not
settle it, in either direction. What the shipped gates DID show stands: FA cleared both on every
draw. The measurement that settled it -- alignment read ACROSS TRAINING against a feedback that can
MOVE -- is now done, and check (5) below re-derives it on every gate run.

ONE PREDICTION WAS WRONG AND IS RECORDED AS SUCH. The spec predicted the converging feedback would
"rise monotonically in trend from ~0". It does not: `B` is at 0.92 after ONE epoch and drifts
slightly DOWN thereafter (0.9231 -> 0.8985 over 30 passes). It starts from the same random draw as
the frozen control (~0.18) and aligns within the first pass, because epoch 1 applies `lr*err*h[j]`
to `B` on every one of 800 updates while `W2` takes the identical increment. Alignment is
ESTABLISHED almost immediately and then TRACKED -- not developed over training. The direction of the
prediction was right; the model of the mechanism was not.

NO DECAY TERM, and that is a MEASURED choice. Textbook Kolen-Pollack pairs the shared increment with
weight decay, which would be a new constant (P1). Swept: decay-free scored 8 of 8, decay = lr/10
scored 7 of 8 with a far wider spread, and decay = lr destroyed alignment outright at 0 of 8. So P1
is satisfied by OMISSION rather than derivation -- this rule adds no constant at all.

The learning rule is the same semi-gradient delta step toward a scalar `target` that the arm and
LatentQ use (q += lr*(r - q)), backpropagated BY HAND (no autograd / torch) through the hidden layer:

    err     = target - Q(z, a)                     # error on the taken action ONLY
    b2[a]  += lr * err ;  W2[a][j] += lr * err * h[j]
    dh[j]   = err * W2[a][j] * relu'(pre[j])       # relu'(p) = 1 if p > 0 else 0
    b1[j]  += lr * dh[j] ; W1[j][i] += lr * dh[j] * z[i]

update() returns err. Only the taken action's read-out W2[a] receives the output error, but the SHARED
hidden layer W1 is reshaped by the backprop of EVERY update -- so learning about one context reshapes
the features every other context reads, which is what carries the transfer (and, as in LatentQ, the
interference: the shared representation is one object).

Same ceiling caveat as LatentQ, honestly stated: an MLP head can only recover structure that is
PRESENT in z. If the latent itself does not carry the information the task needs, no head -- linear or
nonlinear -- can invent it, and the nonlinearity must move into the ENCODER instead (a deeper JEPA;
cf. cerebellum/jepa_world_model's snn_jepa/2 hidden encoder). It is still ONE primitive: a nonlinear
function approximator plus a delta rule, a strict drop-in upgrade of the linear LatentQ.

Distinctness. This is the VALUE table (which action is worth most in a context), not the SELECTION
circuit (action_gating.py's Go/NoGo) nor the DECISION-by-integration model (evidence_accumulation.py).
values(z) returns a dict keyed by action name, the same shape as the tabular q[unit] row and LatentQ's
values(z), so it wires in without touching the selection stage.

stdlib + numpy/torch allowed project-wide (uses random; math.sqrt for the fan-in init; dot reused from substrate._mathutils), ASCII-only
prints, deterministic (random.Random(seed) per instance -- identical seed => byte-identical weights).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.basal_ganglia.mlp_q
"""

import math
import random

from snn_data_lake.substrate._mathutils import dot

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "MlpQ.update": (
        "APPROXIMATES_GRADIENT",
        "semi-gradient delta step backpropped BY HAND through the shared hidden layer -- it owes the "
        "cosine, and it is the case SC-003 turns on: the FROZEN feedback-alignment control misses "
        "the bar on 0 of 8 draws while the shipped Kolen-Pollack path clears it at +0.9031. "
        "CORRECTED 2026-08-18: this read 'falsified the INSTRUMENT rather than the rule', which "
        "inverts the verdict -- gradient_claim_gate::verdict_for keys INSTRUMENT_FALSIFIED solely "
        "on the coherence PRODUCT reaching the bar, and this arm's product is ~0.0001, so it is "
        "RULE_JUDGED. The measurement IS degenerate here (it reduces to asking whether feedback is "
        "parallel to the forward weights, which feedback alignment exists to deny) -- but that is a "
        "PROSE adjudication, not a gate verdict",
    ),
    "_train": (
        "APPROXIMATES_GRADIENT",
        "the harness that drives MlpQ.update over (z, opt) contexts; carries the same claim as the "
        "rule it applies",
    ),
}


class MlpQ:
    """Shared NONLINEAR action-value Q(z, a) = dot(W2[a], relu(W1@z + b1)) + b2[a] over the JEPA latent z.

    ONE weight set (W1, b1 = the shared hidden layer; W2, b2 = the per-action read-out) is shared across
    every context, so a semi-gradient update from one latent transfers to all -- but the hidden ReLU layer
    lets it separate action-values that are NON-LINEAR in z, which the linear LatentQ provably cannot.
    Identical API to LatentQ (values / value / greedy / update) so it drops into the same wiring."""

    def __init__(self, actions, latent_dim, hidden=32, lr=0.05, seed=0):
        self.actions = tuple(actions)  # ordered action names
        self.n = len(self.actions)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.seed = seed
        self.rng = random.Random(seed)  # per-instance rng; same seed => byte-identical weights
        self._aidx = {a: i for i, a in enumerate(self.actions)}
        # SHARED hidden layer: W1 (hidden x latent_dim) small-random (fan-in scale), b1 (hidden) = 0
        s1 = 1.0 / math.sqrt(self.latent_dim)
        self.W1 = [
            [self.rng.uniform(-1.0, 1.0) * s1 for _ in range(self.latent_dim)]
            for _ in range(self.hidden)
        ]
        self.b1 = [0.0] * self.hidden
        # per-action read-out: W2 (n_actions x hidden) small-random, b2 (n_actions) = 0 (LatentQ's zero-bias init)
        s2 = 1.0 / math.sqrt(self.hidden)
        self.W2 = [
            [self.rng.uniform(-1.0, 1.0) * s2 for _ in range(self.hidden)] for _ in range(self.n)
        ]
        self.b2 = [0.0] * self.n
        # FEEDBACK matrix B (n_actions x hidden), same shape and same init scale as W2. It carries
        # error back into the shared hidden layer INSTEAD of the transposed forward weights, and it
        # is LEARNED from strictly local quantities -- see `update`. Drawn from the same per-instance
        # rng, so a fixed seed still gives byte-identical weights.
        self.B = [
            [self.rng.uniform(-1.0, 1.0) * s2 for _ in range(self.hidden)] for _ in range(self.n)
        ]

    def _idx(self, a):
        """Map an action name or an integer index to an action index (duck-typed, like LatentQ / the JEPA)."""
        if isinstance(a, str):
            return self._aidx[a]
        return int(a)

    def _forward(self, z):
        """Shared hidden layer forward pass: pre = W1@z + b1, h = relu(pre). Returns (pre, h)."""
        pre = [dot(self.W1[j], z) + self.b1[j] for j in range(self.hidden)]
        h = [p if p > 0.0 else 0.0 for p in pre]  # relu
        return pre, h

    def values(self, z):
        """All-action value row over latent z as {action_name: Q(z, a)} (same shape as LatentQ.values)."""
        _pre, h = self._forward(z)
        return {self.actions[i]: dot(self.W2[i], h) + self.b2[i] for i in range(self.n)}

    def value(self, z, a):
        """Q(z, a) for one action (a = action name or integer index)."""
        i = self._idx(a)
        _pre, h = self._forward(z)
        return dot(self.W2[i], h) + self.b2[i]

    def greedy(self, z):
        """argmax action name over Q(z, .); deterministic tie-break to the LOWEST index (strict >)."""
        _pre, h = self._forward(z)
        best_i = 0
        best_v = dot(self.W2[0], h) + self.b2[0]
        for i in range(1, self.n):
            v = dot(self.W2[i], h) + self.b2[i]
            if v > best_v:  # strict > keeps the lowest index on ties
                best_v, best_i = v, i
        return self.actions[best_i]

    def update(self, z, a, target):
        """Semi-gradient delta step of Q(z, a) toward scalar `target`; returns err.

        err = target - Q(z, a) is applied to the taken action ONLY, then backpropped BY HAND through the
        shared hidden layer: the read-out W2[a]/b2[a] get the output error; dh[j] = err*W2[a][j]*relu'(pre[j])
        flows into the SHARED W1/b1. Because W1 is shared, this one step reshapes the features every context
        reads -- that is the transfer (and the interference). dh uses the PRE-update W2 (the true gradient)."""
        i = self._idx(a)
        pre, h = self._forward(z)
        err = target - (dot(self.W2[i], h) + self.b2[i])
        lr = self.lr
        w2i = self.W2[i]
        bi = self.B[i]
        # hidden-activation gradient dh[j] from the FEEDBACK weights B (relu' gates dead units to 0).
        # NOT the transposed forward weights: B is a separate matrix that never reads W2. The weight
        # transport this line used to perform was RETIRED 2026-08-14 (specs/015).
        dh = [err * bi[j] if pre[j] > 0.0 else 0.0 for j in range(self.hidden)]
        # read-out update (taken action only): b2[a] += lr*err ; W2[a][j] += lr*err*h[j]
        self.b2[i] += lr * err
        for j in range(self.hidden):
            w2i[j] += lr * err * h[j]
            # KOLEN-POLLACK (Kolen & Pollack 1994): the feedback receives the SAME increment as the
            # forward read-out, so B TRACKS W2 without ever reading it. `err` and `h[j]` are already
            # local to this synapse -- the line above computes both -- so nothing is transported.
            # NO DECAY TERM, and that is a MEASURED choice, not an omission: decay-free scored
            # |cos| in [0.8785, 0.9121] on 8 of 8 draws, decay=lr/10 scored 7 of 8 with a far wider
            # spread, and decay=lr destroyed alignment outright at 0 of 8. P1 is satisfied by
            # OMISSION -- this rule introduces no new constant at all.
            bi[j] += lr * err * h[j]
        # shared hidden-layer update: b1[j] += lr*dh[j] ; W1[j][k] += lr*dh[j]*z[k]
        for j in range(self.hidden):
            g = lr * dh[j]
            if g == 0.0:  # dead relu unit (dh[j]==0) -> no change; skip is exact
                continue
            self.b1[j] += g
            w1j = self.W1[j]
            for k in range(self.latent_dim):
                w1j[k] += g * z[k]
        return err


# =========================================================================================================
# Self-check. The whole point is that a NONLINEAR head recovers structure a LINEAR one PROVABLY cannot, so
# the anchors train MlpQ and the linear LatentQ on the SAME data and compare held-out accuracy on a DISJOINT
# split of latents neither model ever saw during training.
#
# Rules (functions OF the latent z, a unit vector like JEPA encode()):
#   NON-LINEAR (XOR): optimal action = XOR of the sign pattern of two latent dims -> the classic XOR, which
#       no hyperplane separates. A linear Q is stuck at chance; a one-hidden-layer MLP recovers it.
#   LINEAR (control): optimal action = which side of the z0 = z1 hyperplane z is on -> a single linear
#       functional threshold, so a linear Q CAN learn it -- the MLP must at least match it, not regress.
# =========================================================================================================


def _xor_action(z):
    """NON-LINEAR rule: optimal action = XOR of the sign pattern of the first two latent dims.
    action 0 when sign(z0)==sign(z1) (quadrants I, III), action 1 otherwise (II, IV) -- the classic XOR,
    PROVABLY not linearly separable, so no linear Q(z, a) = dot(W[a], z) + b[a] can represent it."""
    return 0 if (z[0] >= 0.0) == (z[1] >= 0.0) else 1


def _linear_action(z):
    """LINEARLY-separable control rule: optimal action = which side of the z0 = z1 hyperplane z is on.
    action 0 when (z0 - z1) >= 0 else 1 -- a single linear-functional threshold, so a linear Q CAN learn it."""
    return 0 if (z[0] - z[1]) >= 0.0 else 1


def _make_contexts(n, latent_dim, rule, seed):
    """n (z, optimal_action) pairs; z is a unit vector in R^latent_dim (matches JEPA encode())."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        v = [rng.gauss(0.0, 1.0) for _ in range(latent_dim)]
        nrm = math.sqrt(sum(c * c for c in v)) or 1.0
        z = [c / nrm for c in v]  # L2-normalize -> unit vector (like substrate._mathutils.unit)
        out.append((z, rule(z)))
    return out


def _acc(model, contexts, idx_of):
    """Fraction of held-out contexts whose greedy() equals the optimal action index."""
    ok = 0
    for z, opt in contexts:
        if idx_of[model.greedy(z)] == opt:
            ok += 1
    return ok / len(contexts)


def _train(model, contexts, n_actions, passes):
    """Train a shared action-value on (z, opt) contexts: reward +1 for the optimal action, -1 else."""
    for _ in range(passes):
        for z, opt in contexts:
            for i in range(n_actions):
                model.update(
                    z, i, 1.0 if i == opt else -1.0
                )  # reward = target (bootstrap-free), like the arm


def main():
    from snn_data_lake.basal_ganglia.latent_q import LatentQ

    LATENT = 2
    ACTIONS = ("A0", "A1")
    idx_of = {a: i for i, a in enumerate(ACTIONS)}
    N_TRAIN, N_TEST, PASSES, HIDDEN, LR = 400, 300, 30, 32, 0.1

    # ---- (1)+(2) NON-LINEAR (XOR) rule: MlpQ recovers it, the linear LatentQ provably cannot ----
    nl_train = _make_contexts(N_TRAIN, LATENT, _xor_action, seed=1)
    nl_test = _make_contexts(
        N_TEST, LATENT, _xor_action, seed=2
    )  # DISJOINT held-out (different rng stream)
    mlp = MlpQ(ACTIONS, LATENT, hidden=HIDDEN, lr=LR, seed=0)
    lin = LatentQ(ACTIONS, LATENT, lr=LR, seed=0)
    _train(mlp, nl_train, len(ACTIONS), PASSES)
    _train(lin, nl_train, len(ACTIONS), PASSES)
    mlp_nl = _acc(mlp, nl_test, idx_of)
    lin_nl = _acc(lin, nl_test, idx_of)
    mlp_nl_train = _acc(mlp, nl_train, idx_of)

    # ---- (3) LINEARLY-separable control rule: MlpQ must NOT REGRESS vs LatentQ (gated within 0.05).
    # Measured honestly: MlpQ is very slightly BEHIND the linear Q here (the extra hidden capacity costs
    # a little on a rule a linear model fits exactly), so this is a capacity TRADEOFF, not a strict win. ----
    li_train = _make_contexts(N_TRAIN, LATENT, _linear_action, seed=3)
    li_test = _make_contexts(N_TEST, LATENT, _linear_action, seed=4)
    mlp_l = MlpQ(ACTIONS, LATENT, hidden=HIDDEN, lr=LR, seed=0)
    lin_l = LatentQ(ACTIONS, LATENT, lr=LR, seed=0)
    _train(mlp_l, li_train, len(ACTIONS), PASSES)
    _train(lin_l, li_train, len(ACTIONS), PASSES)
    mlp_lin = _acc(mlp_l, li_test, idx_of)
    lat_lin = _acc(lin_l, li_test, idx_of)

    # ---- (4) determinism: two MlpQ(seed=11) on the same stream -> identical weights + greedy ----
    da = MlpQ(ACTIONS, LATENT, hidden=HIDDEN, lr=LR, seed=11)
    db = MlpQ(ACTIONS, LATENT, hidden=HIDDEN, lr=LR, seed=11)
    _train(da, nl_train[:150], len(ACTIONS), 3)
    _train(db, nl_train[:150], len(ACTIONS), 3)
    same_w = da.W1 == db.W1 and da.b1 == db.b1 and da.W2 == db.W2 and da.b2 == db.b2
    same_greedy = all(da.greedy(z) == db.greedy(z) for z, _o in nl_test[:60])

    print("=" * 78)
    print(
        "MLP Q -- shared NONLINEAR one-hidden-layer action-value over the JEPA latent (vs linear LatentQ)"
    )
    print("=" * 78)
    print(
        f"latent_dim={LATENT} | actions={ACTIONS} | hidden={HIDDEN} | train={N_TRAIN} test={N_TEST} "
        f"| passes={PASSES} lr={LR}"
    )
    print(
        "rule (1/2): NON-LINEAR XOR of the sign pattern of two latent dims (no hyperplane separates it)"
    )
    print(
        f"(1) MlpQ    held-out accuracy : {mlp_nl:.3f}   (train {mlp_nl_train:.3f}) -> LEARNS the XOR rule"
    )
    print(
        f"(2) LatentQ held-out accuracy : {lin_nl:.3f}   (linear -> stuck near chance {1.0 / len(ACTIONS):.3f})"
    )
    print(
        f"    => nonlinear head recovers structure the linear Q cannot ({mlp_nl:.3f} >> {lin_nl:.3f})"
    )
    print(
        "rule (3): LINEARLY-separable (which side of z0 = z1) -- MlpQ must NOT regress vs LatentQ"
    )
    _delta = mlp_lin - lat_lin
    print(
        f"(3) linear  rule held-out     : MlpQ {mlp_lin:.3f}  vs  LatentQ {lat_lin:.3f} "
        f"(delta {_delta:+.3f}) -> NO REGRESSION (gate: within 0.05), not a strict upgrade"
    )
    print(
        f"(4) determin: two seed=11 MlpQ identical weights={same_w}, identical greedy={same_greedy}"
    )
    print("=" * 78)

    # ---- un-riggable anchors (deterministic asserts) --------------------------------------------
    # (1) NON-LINEAR: the MLP head recovers the XOR rule on held-out latents it never trained on.
    assert mlp_nl > 0.85, (
        f"MlpQ must learn the non-linear (XOR) rule on held-out latents (>0.85), got {mlp_nl:.3f}"
    )
    # (2) FAIR CONTROL: a linear Q provably cannot separate XOR -> stuck near chance; the MLP beats it by a wide margin.
    assert lin_nl < 0.65, f"linear LatentQ must NOT learn XOR (near chance <0.65), got {lin_nl:.3f}"
    assert mlp_nl > lin_nl + 0.30, (
        f"nonlinear head must beat the linear Q on the XOR rule by >0.30 ({mlp_nl:.3f} vs {lin_nl:.3f})"
    )
    # (3) TRANSFER / NO-REGRESSION: on a LINEARLY-separable rule the MLP stays within 0.05 of the linear Q.
    # This is deliberately a tolerance, not `>=`: the MLP does not dominate everywhere, and the gate says so.
    assert mlp_lin > 0.85, (
        f"MlpQ must also generalize on the linear rule (>0.85), got {mlp_lin:.3f}"
    )
    assert mlp_lin >= lat_lin - 0.05, (
        f"MlpQ must not regress vs LatentQ on the linearly-separable rule ({mlp_lin:.3f} vs {lat_lin:.3f})"
    )
    # (4) DETERMINISM: identical seed + stream => byte-identical weights and identical greedy choices.
    assert same_w, "two MlpQ(seed=11) on the same stream must have byte-identical weights"
    assert same_greedy, "two MlpQ(seed=11) must make identical greedy choices"

    # ---- (5) the e-prop cosine the docstring's OPEN violation rests on, MEASURED not quoted -------
    # The docstring declines the feedback-alignment fix because FA's hidden-gradient direction misses
    # the cross-cutting invariant's cosine > 0.5 bar. That number is a property of the random feedback
    # DRAW (err cancels in a cosine, so this is cos(W2[a], B[a]) under the relu mask), so one draw
    # quoted in prose is unreproducible. Swept over eight draws here instead, on the model main()
    # already trained -- no second live head is constructed (P2).
    def _dh(model, z, i, target, feedback):
        """The hidden-layer error direction update() computes, with `feedback` in place of W2[a]."""
        pre, h = model._forward(z)
        err = target - (dot(model.W2[i], h) + model.b2[i])
        return [err * feedback[j] if pre[j] > 0.0 else 0.0 for j in range(model.hidden)]

    def _cos(a, b):
        na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
        return None if na <= 0.0 or nb <= 0.0 else sum(x * y for x, y in zip(a, b)) / (na * nb)

    def _mean_cos(feedback_of):
        vals = []
        for z, opt in nl_test[:200]:
            for i in range(len(ACTIONS)):
                t = 1.0 if i == opt else -1.0
                c = _cos(_dh(mlp, z, i, t, mlp.W2[i]), _dh(mlp, z, i, t, feedback_of(i)))
                if c is not None:
                    vals.append(c)
        return sum(vals) / len(vals)

    s2 = 1.0 / math.sqrt(HIDDEN)
    fa_cos = []
    for bseed in range(1, 9):  # per-instance Random, never module-scope seeding
        brng = random.Random(bseed)
        B = [[brng.uniform(-1.0, 1.0) * s2 for _ in range(HIDDEN)] for _ in range(len(ACTIONS))]
        fa_cos.append(_mean_cos(lambda i, B=B: B[i]))
    worst = max(abs(c) for c in fa_cos)
    # POSITIVE CONTROL: feedback == the forward weight IS the transport gradient, so the harness
    # must read exactly 1.0. If it does not, nothing else here measures what it claims.
    transport_cos = _mean_cos(lambda i: mlp.W2[i])
    assert abs(transport_cos - 1.0) < 1e-12, (
        f"positive control: feedback == the forward weight IS the transport gradient, so this must "
        f"read 1.0 -- got {transport_cos!r}; the cosine harness is not measuring what it claims"
    )
    # THE SHIPPED RULE's alignment. `mlp.B` has been trained by _train above, so this is the
    # Kolen-Pollack feedback AFTER convergence -- the quantity the e-prop invariant actually asks
    # about. The FROZEN draws above are the control that is not the method itself: they use the
    # same harness, the same init scale and the same inputs, and differ ONLY in being unable to
    # move. Three-way separation is the point -- transport 1.0, shipped ~0.9, frozen ~0.2.
    shipped_cos = _mean_cos(lambda i: mlp.B[i])
    assert abs(shipped_cos) > 0.5, (
        f"the shipped feedback must clear the e-prop cosine bar: got {shipped_cos:.4f}. Measured "
        f"2026-08-14 over 8 draws x 30 passes it lands in [0.8785, 0.9121] on 8 of 8 "
        f"(specs/015-resolve-weight-transport/evidence/trajectory.md)"
    )
    assert abs(shipped_cos) > worst, (
        f"a LEARNED feedback must beat a FROZEN one on this criterion, else convergence bought "
        f"nothing: shipped {shipped_cos:.4f} vs frozen worst {worst:.4f}"
    )
    print(
        f"(5) e-prop cosine vs the transport gradient over 200 held-out inputs: SHIPPED "
        f"(Kolen-Pollack, learned B) {shipped_cos:+.4f} -> clears the >0.5 bar; FROZEN control over "
        f"8 draws [{min(fa_cos):+.4f}, {max(fa_cos):+.4f}], |cos| > 0.5 in "
        f"{sum(1 for c in fa_cos if abs(c) > 0.5)} of 8; transport identity {transport_cos:.3f}"
    )

    print(
        f"self-check OK: MlpQ learns the XOR rule on held-out latents ({mlp_nl:.3f}) while the linear "
        f"LatentQ cannot ({lin_nl:.3f} ~ chance);"
    )
    print(
        f"  no regression on the linear rule (MlpQ {mlp_lin:.3f} vs LatentQ {lat_lin:.3f}, "
        f"delta {_delta:+.3f}, gated within 0.05 -- a capacity TRADEOFF, not a strict upgrade); "
        "identical seed => byte-identical weights + greedy"
    )


if __name__ == "__main__":
    main()
