r"""
PC ABDUCTION -- spiking ABDUCTION (inference to the best explanation) + Pearl COUNTERFACTUAL over the
lake's spiking predictive-coding generative model. The live
reasoning path only had transitive-isa DEDUCTION (reasoning/derivation.py BFS) and single-hop link
prediction; it had NO abduction and NO causal/counterfactual reasoning. This module adds both, built
DIRECTLY on the repo's spiking generative model (sensorimotor/pc_cortex_serve.py::PcCortexServe -- a served
deep spiking Rao-Ballard coder whose weights come from spiking training, NOT a rate-ANN backprop head).
Representation / cognition ONLY: no actuator route, never imports brainstem.

Literature grounding (the mechanisms implemented here, not decoration):
  * ABDUCTION = Bayesian MODEL SELECTION (Koerding et al. 2007, PLoS ONE). Choosing the hidden cause is
    picking the causal structure with the highest posterior given the evidence. Under a Gaussian
    generative likelihood p(x | cause z) proportional to exp(-||x - g(z)||^2 / 2 sigma^2), the negative
    log-likelihood of a candidate cause is its prediction-error energy, so the posterior over a bounded
    candidate set is softmax(-beta * F) with F the free energy and beta an inverse-temperature/precision.
  * FREE-ENERGY / predictive-coding scoring (Rao-Ballard 1999; Friston free-energy). The generative model
    scores a candidate cause by its SETTLED prediction-error energy F. For a GIVEN (clamped) top cause z,
    the spiking network settles to its top-down-consistent fixed point: every inter-level error population
    falls silent (its LIF error neurons stop spiking) and the ONLY residual is the input-level mismatch
    e_0 = x - g(z), whose squared energy the input error population spike-encodes. So the settled free
    energy of a clamped cause reduces to F(z) = ||x - g(z)||^2, read off the repo's spiking generative
    pass g = PcCortexServe.generate (top-down W_1 . W_2 ... W_L). Abduction = argmin_z F(z).
  * PEARL do-calculus / Neural Causal Models (Xia & Bareinboim 2022, NeurIPS), the 3-step counterfactual
    (Thm 7.1.7): (1) ABDUCTION -- infer the latent posterior from the factual observation; (2) ACTION --
    apply the intervention to that latent (the do-operator / clamp); (3) PREDICTION -- generate downward
    to the counterfactual observation. Here: z = infer(x_factual); z' = do(z, intervention);
    x_cf = generate(z') -- the counterfactual percept under the SAME abducted latent.

Honest ceiling (named, never overclaimed): this is BOUNDED candidate-set abduction over a LOW-DIM spiking
generative model. It does NOT discover novel causal structure from raw sensory data and it does NOT learn
new causes; it selects, among a supplied/stored candidate set, the maximum-posterior cause under the
model, and it returns an HONEST GAP (known=False) when the top-2 posterior margin is below threshold
(ambiguous) so it never overclaims -- and NEVER fabricates a cause. The AbductionExpert's effect->cause
grounding is a deterministic encoding of the brain's stored causal triples (the effect observation is the
generative signature of its taught cause), so the expert demonstrates the abduction MECHANISM (Bayesian
model selection over the spiking generative model), not sensory causal discovery.

Works with any model exposing generate(top_cause)->obs and infer(obs)->top_cause plus in_dim/top_dim
(e.g. PcCortexServe). Deterministic (per-instance random.Random(seed); zlib.crc32, never builtin hash;
no wall-clock, no global RNG). ASCII-only prints. Every public call is guarded: a failure returns an
honest gap / None, never raises.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reasoning.pc_abduction
"""

import sys

# Captured BEFORE EVERY OTHER IMPORT IN THIS MODULE -- the position is load-bearing, not stylistic.
# It records whether a CALLER already had torch loaded, which makes criterion (7) in main() exact
# and structural: "torch is loaded now AND was not loaded before us" means THIS module's closure
# dragged it in, which is the only thing that criterion claims.
#
# Two earlier drafts of this were wrong and both were caught by running the constructed red rather
# than by reading: (a) gating the assert on `len(sys.modules) < 400`, a magic constant arbitrating
# whether a check runs at all, which is what P1 forbids; (b) placing this capture BELOW the stdlib
# import block, where an `import torch` added among them would execute first and set the flag True,
# so the criterion would report "preloaded by caller" about an import in its own file and pass.
# Only `sys` may precede this line, and stdlib `sys` cannot pull torch.
_TORCH_PRELOADED = "torch" in sys.modules

import math  # noqa: E402
import random  # noqa: E402
import zlib  # noqa: E402

# Relations that license a cause->effect reading for the AbductionExpert. (Kept local + explicit so the
# module stays self-contained; a triple whose relation is not genuinely causal is not turned into an
# explanation.)
_CAUSAL_RELS = frozenset(
    {
        "causes",
        "cause",
        "caused_by",
        "leads_to",
        "produces",
        "results_in",
        "makes",
        "triggers",
        "because_of",
    }
)

# Honest-gap threshold: if the top-2 posterior margin is below this, abduction is AMBIGUOUS -> known=False.
MIN_MARGIN = 0.1


# ------------------------------------------------------------- free-energy scorer --------------------
def _free_energy(pc, x_obs, z):
    """Settled FREE ENERGY of observation x_obs when the generative model's top cause is CLAMPED to z:
    F(z) = ||x_obs - g(z)||^2, where g = pc.generate is the spiking model's top-down generative pass. At
    the clamped fixed point the inter-level error populations are silent and only the input-level residual
    remains, so this squared residual IS the settled free energy (the quantity the input error population
    spike-encodes). Not a rate-ANN score -- the weights are the repo's spiking predictive coder's."""
    x_hat = pc.generate([float(v) for v in z])
    return sum((float(x_obs[i]) - x_hat[i]) ** 2 for i in range(len(x_hat)))


def _norm_candidates(candidate_causes):
    """Normalise candidate causes to an ordered [(name, latent), ...] list. Accepts a dict {name: latent}
    (insertion order preserved) or a plain sequence of latents (names default to their integer index)."""
    if isinstance(candidate_causes, dict):
        return [(k, [float(x) for x in v]) for k, v in candidate_causes.items()]
    return [(i, [float(x) for x in v]) for i, v in enumerate(candidate_causes)]


# ------------------------------------------------------------- abduction ------------------------------
def abduce(pc, x_obs, candidate_causes, beta=1.0, min_margin=MIN_MARGIN):
    """ABDUCTION as Bayesian model selection over a bounded candidate set (Koerding 2007). For each
    candidate hidden cause z_k score its settled free energy F_k = ||x_obs - generate(z_k)||^2 under the
    spiking generative model, form the posterior p_k = softmax(-beta * F_k) (the likelihood of the
    evidence under each cause), and return the argmin-F (max-posterior) cause.

    Returns {"best": <cause>, "posterior": {cause: p}, "margin": float, "known": bool}. `known` is False
    (an HONEST GAP) when the top-2 posterior margin < min_margin -- ambiguous, so the cause is NOT
    overclaimed. Any failure returns a guarded honest gap (known=False), never raises, never fabricates."""
    try:
        items = _norm_candidates(candidate_causes)
        if not items:
            return {"best": None, "posterior": {}, "margin": 0.0, "known": False}
        x = [float(v) for v in x_obs]
        energies = [(name, _free_energy(pc, x, z)) for name, z in items]
        b = float(beta)
        f_min = min(f for _n, f in energies)
        weights = [
            (name, math.exp(-b * (f - f_min))) for name, f in energies
        ]  # subtract min -> stable
        total = sum(w for _n, w in weights) or 1.0
        posterior = {name: w / total for name, w in weights}
        order = sorted(posterior, key=lambda n: (-posterior[n], str(n)))  # deterministic tie-break
        best = order[0]
        margin = posterior[best] - (posterior[order[1]] if len(order) > 1 else 0.0)
        known = margin >= float(min_margin)
        return {"best": best, "posterior": posterior, "margin": margin, "known": bool(known)}
    except Exception:
        return {"best": None, "posterior": {}, "margin": 0.0, "known": False}


# ------------------------------------------------------------- counterfactual -------------------------
def _apply_intervention(z, intervention):
    """ACTION step of the Pearl 3-step: apply an intervention to the abducted latent z. Accepts a dict
    {index: value} (the do-operator do(z_i = value) on latent dims), a full replacement sequence (same
    length as z), or a callable z -> z'. Returns the intervened latent (z is not mutated)."""
    base = [float(v) for v in z]
    if callable(intervention):
        return [float(v) for v in intervention(list(base))]
    if isinstance(intervention, dict):
        out = list(base)
        for k, v in intervention.items():
            out[int(k)] = float(v)
        return out
    seq = [float(v) for v in intervention]
    if len(seq) != len(base):
        raise ValueError(f"intervention length {len(seq)} != latent length {len(base)}")
    return seq


def counterfactual(pc, x_factual, intervention, tol=1e-9):
    """PEARL 3-step counterfactual (Xia & Bareinboim 2022, Thm 7.1.7) over the spiking generative model:
      (1) ABDUCTION  -- z = infer(x_factual): settle the spiking net to the latent that explains the fact.
      (2) ACTION     -- z' = do(z, intervention): apply the intervention to that latent (clamp/do-op).
      (3) PREDICTION -- x_cf = generate(z'): generate the counterfactual percept downward.

    Returns {"x_cf": [...], "changed": bool}. `changed` compares the counterfactual to the FACTUAL outcome
    rendered under the SAME abducted latent (generate(z)) -- so an intervention equal to the factual latent
    is (near-)identical (changed=False) and a differing intervention changes the outcome (changed=True).
    Any failure returns {"x_cf": None, "changed": False} -- guarded, never raises."""
    try:
        z_fact = [float(v) for v in pc.infer([float(v) for v in x_factual])]  # (1) abduction
        z_cf = _apply_intervention(z_fact, intervention)  # (2) action / do()
        x_cf = pc.generate(z_cf)  # (3) prediction
        x_fact_recon = pc.generate(z_fact)  # factual outcome, same z
        diff = sum((x_cf[i] - x_fact_recon[i]) ** 2 for i in range(len(x_cf)))
        return {"x_cf": x_cf, "changed": diff > float(tol)}
    except Exception:
        return {"x_cf": None, "changed": False}


# ------------------------------------------------------------- deterministic latent encoding ----------
def _latent_for(name, dim, seed):
    """Deterministic unit-ish latent code for a named cause: crc32(name) (never builtin hash) seeds a
    fresh random.Random -> a length-dim vector in [-1, 1], scaled to unit-ish norm. Distinct names give
    distinct, reproducible latents; the same name always yields the same code."""
    rng = random.Random(zlib.crc32(f"PCA|{seed}|{name}".encode("ascii")))
    v = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    nrm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / nrm for x in v]


# ------------------------------------------------------------- the abduction expert -------------------
class AbductionExpert:
    """Thin reasoning-cortex expert over abduce(): given a query naming a stored EFFECT, abduce the
    maximum-posterior CAUSE among the brain's stored causal triples, or return an honest gap. Follows the
    reasoning_cortex expert protocol -- class attrs (name/module/source/uses) + score(q)/answer(q) -- so
    build_default_cortex can register it; also exposes route(text) returning the router's {known, reply,
    source, confidence, expert} dict for standalone use.

    Grounding (honest): each distinct cause symbol gets a deterministic latent (crc32-seeded); the observed
    effect is the generative signature generate(z_cause) of the cause(s) taught for that effect. Abduction
    then selects, among ALL known cause latents, the one whose settled free energy best explains that
    observation. This makes the spiking abduction MECHANISM live over the brain's own triples; it does not
    claim sensory causal discovery. Inert (score 0) on ordinary text and when constructed without causal
    triples -- abduction needs an effect + candidate causes, which ordinary chat does not supply."""

    name, module, source, uses = "pc-abduction", "pc_abduction", "abduce", "stdlib"
    _TRIGGERS = (
        "abduce",
        "explain",
        "best explanation",
        "what explains",
        "why ",
        "what causes",
        "cause of",
        "reason for",
        "because of",
    )

    def __init__(self, seed=0, triples=None, pc=None, beta=1.0):
        self.seed = int(seed)
        self.beta = float(beta)
        self._pc = pc  # optional injected generative model
        self._built = None  # lazy (pc, {cause: latent}) on first answer
        self._causes_by_effect = {}  # normalized effect -> [cause, ...] (order kept)
        for tri in triples or []:
            if not tri or len(tri) < 3:
                continue
            cause, rel, effect = str(tri[0]).lower(), str(tri[1]).lower(), str(tri[2]).lower()
            if rel.replace(" ", "_") not in _CAUSAL_RELS:
                continue
            eff = " ".join(_toks(effect))
            cau = " ".join(_toks(cause))
            if not eff or not cau:
                continue
            bucket = self._causes_by_effect.setdefault(eff, [])
            if cau not in bucket:
                bucket.append(cau)
        self._effects = sorted(self._causes_by_effect)
        self._causes = sorted({c for cs in self._causes_by_effect.values() for c in cs})

    # ---- lazy generative model + candidate latents (built to SEPARATE the stored causes) ------------
    def _ensure(self):
        """Build (and cache) the (pc, {cause: latent}) pair. Default (no injected model): a spiking PC
        coder sized to the causes with ORTHOGONAL per-cause signatures + one-hot cause latents, so every
        distinct cause is separable (no spurious collisions). An injected model uses crc32-seeded latents
        in its own top_dim (best effort; genuine collisions then surface as an honest gap)."""
        if self._built is None:
            if self._pc is not None:
                pc = self._pc
                latents = {c: _latent_for(c, pc.top_dim, self.seed) for c in self._causes}
            else:
                pc, basis = _build_ortho_pc(max(2, len(self._causes)))
                latents = {c: basis[i] for i, c in enumerate(self._causes)}
                self._pc = pc
            self._built = (pc, latents)
        return self._built

    def _find_effect(self, q):
        """The stored effect whose tokens all appear in the query (longest match wins)."""
        low = getattr(q, "low", str(q).lower())
        toks = set(_toks(low))
        hits = [e for e in self._effects if set(_toks(e)) <= toks]
        return max(hits, key=lambda e: len(_toks(e))) if hits else None

    def score(self, q):
        if not self._causes_by_effect:
            return 0.0
        low = getattr(q, "low", str(q).lower())
        if not any(t in low for t in self._TRIGGERS):
            return 0.0
        return 0.9 if self._find_effect(q) else 0.0

    def answer(self, q):
        effect = self._find_effect(q)
        if effect is None:
            return (
                "[honest gap] name a stored effect to abduce a cause for (e.g. 'why wet_ground').",
                False,
            )
        candidates = self._causes_by_effect.get(effect, [])
        if not candidates or len(self._causes) < 2:
            return (
                "[honest gap] too few candidate causes in the stored triples to abduce over.",
                False,
            )
        pc, latents = self._ensure()
        # observed effect = generative signature of its taught cause(s) (mean over the stored candidates)
        x_obs = _mean_obs(pc, [latents[c] for c in candidates])
        res = abduce(pc, x_obs, latents, beta=self.beta)
        post = res["posterior"]
        best = res["best"]
        if not res["known"] or best is None:
            m = res["margin"]
            return (
                f"[honest gap] the spiking generative model does not single out a cause for '{effect}' "
                f"(top-2 posterior margin {m:.2f} < {MIN_MARGIN:.2f}) -- ambiguous, no overclaim.",
                False,
            )
        return (
            f"[abduce] best explanation for '{effect}' -> cause '{best}' (posterior "
            f"{post.get(best, 0.0):.2f}, margin {res['margin']:.2f} over {len(self._causes)} candidate "
            f"causes; Bayesian model selection = argmin settled free energy under the spiking "
            f"predictive-coding model).",
            True,
        )

    def route(self, text):
        """Standalone {known, reply, source, confidence, expert} for a raw query (router-compatible)."""
        q = _Q(text)
        s = self.score(q)
        if s <= 0.0:
            return {
                "known": False,
                "source": None,
                "confidence": 0.0,
                "expert": self.name,
                "reply": "[honest gap] not an abduction query (name a stored effect to explain).",
            }
        try:
            reply, known = self.answer(q)
        except Exception as exc:  # never let a broken expert crash the turn
            return {
                "known": False,
                "source": None,
                "confidence": round(s, 2),
                "expert": self.name,
                "reply": f"[honest gap] the abduction expert could not answer ({type(exc).__name__}).",
            }
        return {
            "known": bool(known),
            "source": self.source if known else None,
            "confidence": round(s, 2),
            "expert": self.name,
            "reply": reply,
        }


# ------------------------------------------------------------- small helpers --------------------------
def _toks(text):
    """Lowercase alphanumeric+underscore tokens (matches the lake's triple tokenisation, keeps 'wet_ground')."""
    import re

    return re.findall(r"[a-z0-9_]+", str(text).lower())


class _Q:
    """Minimal query shim for route(text): exposes .text/.low/.has like reasoning_cortex.Query."""

    def __init__(self, text):
        self.text = text or ""
        self.low = self.text.lower()

    def has(self, *kw):
        return any(k in self.low for k in kw)


def _mean_obs(pc, latents):
    """Mean generative observation over a set of cause latents (the observed effect grounding)."""
    obs = [pc.generate(list(z)) for z in latents]
    n = len(obs)
    d = len(obs[0])
    return [sum(o[i] for o in obs) / n for i in range(d)]


def _build_ortho_pc(n):
    """A tiny deterministic spiking predictive-coding generative model (PcCortexServe) with n causes whose
    ORTHOGONAL signatures separate them cleanly: dims [2n, n, n], W_2 = I(n), W_1 columns orthogonal
    (cause k drives its own observation pair). Returns (pc, basis) where basis[k] is the one-hot top-cause
    latent for cause k, so generate(basis[k]) are pairwise orthogonal -> clean free-energy separation.
    Constructed via the public keyword __init__ (already-validated components) -- no JSON round-trip."""
    from snn_data_lake.sensorimotor.pc_cortex_serve import PcCortexServe

    dims = [2 * n, n, n]
    w1 = [[0.0] * n for _ in range(2 * n)]  # 2n x n, orthogonal columns
    for k in range(n):
        w1[2 * k][k] = 1.0
        w1[2 * k + 1][k] = 1.0
    w2 = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # n x n identity
    pc = PcCortexServe(
        dims=dims, gen_weights=[w1, w2], theta=0.5, leak=0.02, prec=[1.0, 1.0], steps=200, lr=0.05
    )
    basis = [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    return pc, basis


def _build_demo_pc(seed=0):
    """The core self-check's generative model: three orthogonal causes (dims [6, 3, 3]); a thin wrapper
    over _build_ortho_pc(3). generate([1,0,0]) = [1,1,0,0,0,0], etc. Deterministic; seed unused (hand-set)."""
    _ = seed
    return _build_ortho_pc(3)[0]


# ------------------------------------------------------------- headless self-check --------------------
def main():
    import sys

    pc = _build_demo_pc(seed=0)
    top = pc.top_dim

    # THREE distinct causes; each generates an orthogonal observation via the spiking generative pass.
    causes = {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "C": [0.0, 0.0, 1.0]}
    obs = {k: pc.generate(v) for k, v in causes.items()}

    print("=" * 92)
    print("PC ABDUCTION -- spiking abduction (Bayesian model selection) + Pearl counterfactual")
    print("=" * 92)
    print(
        f"model       : PcCortexServe dims={pc.dims} top={top} (W_2=I, orthogonal W_1 columns) -- spiking PC coder"
    )
    for k in ("A", "B", "C"):
        print(f"  cause {k}   : z={causes[k]} -> generate(z)={[round(x, 2) for x in obs[k]]}")
    print("-" * 92)

    # ---- (1) abduce recovers the correct cause for an observation that cause generated (argmin free energy)
    x_from_A = obs["A"]
    ab = abduce(pc, x_from_A, causes, beta=1.0)
    post = ab["posterior"]
    print(
        f"(1) abduce   : x = generate(cause A); posterior = { ({k: round(post[k], 3) for k in post}) }"
    )
    print(f"               -> best='{ab['best']}' margin={ab['margin']:.3f} known={ab['known']}")
    assert ab["best"] == "A" and ab["known"], (
        "abduce must recover cause A (argmin free energy) with confidence"
    )
    assert post["A"] > post["B"] and post["A"] > post["C"], (
        "cause A must carry the highest posterior"
    )

    # ---- (2) an AMBIGUOUS observation (equidistant between A and B) returns an HONEST GAP (no overclaim)
    x_ambig = [0.5 * (obs["A"][i] + obs["B"][i]) for i in range(pc.in_dim)]
    ag = abduce(pc, x_ambig, causes, beta=1.0)
    print(
        f"(2) honest gap: x = midpoint(A,B); posterior = "
        f"{ ({k: round(ag['posterior'][k], 3) for k in ag['posterior']}) }"
    )
    print(
        f"               -> best='{ag['best']}' margin={ag['margin']:.3f} known={ag['known']} "
        f"(ambiguous -> no cause overclaimed)"
    )
    assert not ag["known"], (
        "an ambiguous (equidistant) observation must be an honest gap (known=False)"
    )
    assert abs(ag["posterior"]["A"] - ag["posterior"]["B"]) < 1e-9, (
        "A and B must tie on the midpoint"
    )

    # ---- (3) Pearl 3-step counterfactual: changes the outcome iff the intervention differs from the fact
    x_fact = obs["A"]  # factual: an A-generated percept
    cf_diff = counterfactual(
        pc, x_fact, [0.0, 1.0, 0.0]
    )  # do(z := cause B latent) -> should change
    cf_do = counterfactual(pc, x_fact, {0: 3.0})  # do(z_0 := 3.0) (dict do-op) -> should change
    cf_same = counterfactual(pc, x_fact, {})  # empty do -> no change (near-identical)
    print("(3) counterfactual (Pearl abduction->action->prediction):")
    print(
        f"     do(z:=B)   changed={cf_diff['changed']}  x_cf={[round(x, 2) for x in cf_diff['x_cf']]}"
    )
    print(
        f"     do(z0:=3)  changed={cf_do['changed']}  x_cf={[round(x, 2) for x in cf_do['x_cf']]}"
    )
    print(
        f"     do(non-op) changed={cf_same['changed']}  (intervention == factual latent -> outcome unchanged)"
    )
    assert cf_diff["changed"], "a differing intervention must change the counterfactual outcome"
    assert cf_do["changed"], (
        "a do(index) intervention that moves the latent must change the outcome"
    )
    assert not cf_same["changed"], (
        "an empty intervention (== factual latent) must leave the outcome unchanged"
    )

    # ---- (4) determinism: identical inputs -> identical results (no wall-clock, no global RNG)
    assert abduce(pc, x_from_A, causes, beta=1.0) == ab, "abduce must be deterministic"
    assert counterfactual(pc, x_fact, [0.0, 1.0, 0.0]) == cf_diff, (
        "counterfactual must be deterministic"
    )

    # ---- (5) guarded: a malformed call returns an honest gap, never raises
    bad = abduce(pc, [0.0], causes, beta=1.0)  # wrong-length observation
    assert bad["known"] is False and bad["best"] is None, (
        "a malformed abduce must be a guarded honest gap"
    )
    bad_cf = counterfactual(pc, [0.0], [0.0, 1.0, 0.0])  # wrong-length factual
    assert bad_cf["x_cf"] is None and bad_cf["changed"] is False, (
        "a malformed counterfactual must be guarded"
    )

    # ---- (6) the AbductionExpert: abduce a cause for a stored EFFECT over the brain's causal triples
    triples = [
        ("rain", "causes", "wet_ground"),
        ("sprinkler", "causes", "wet_lawn"),
        ("fire", "causes", "smoke"),
        ("virus", "causes", "fever"),
    ]
    expert = AbductionExpert(seed=0, triples=triples)
    r_ok = expert.route("why wet_ground")
    r_ok2 = expert.route("what explains smoke")
    r_gap = expert.route("why sunshine")  # unknown effect -> router-level gap
    r_none = expert.route("tell me a joke")  # not an abduction query -> inert
    print("-" * 92)
    print("(6) AbductionExpert over stored causal triples:")
    print(
        f"     Q: why wet_ground     -> [known={r_ok['known']} src={r_ok['source']}] {r_ok['reply']}"
    )
    print(
        f"     Q: what explains smoke-> [known={r_ok2['known']} src={r_ok2['source']}] {r_ok2['reply']}"
    )
    print(f"     Q: why sunshine       -> [known={r_gap['known']}] {r_gap['reply']}")
    print(f"     Q: tell me a joke     -> [known={r_none['known']}] {r_none['reply']}")
    assert r_ok["known"] and r_ok["source"] == "abduce" and "rain" in r_ok["reply"], r_ok
    assert r_ok2["known"] and "fire" in r_ok2["reply"], r_ok2
    assert not r_gap["known"], "an unknown effect must be an honest gap"
    assert not r_none["known"] and r_none["expert"] == "pc-abduction", (
        "non-abduction text must be inert"
    )
    # protocol: registerable in reasoning_cortex (name/module/source/uses + score/answer)
    assert (expert.name, expert.module, expert.source, expert.uses) == (
        "pc-abduction",
        "pc_abduction",
        "abduce",
        "stdlib",
    ), "expert must expose the router protocol attrs"

    # ---- (7) the abduction path stays off the TRAINING tier.
    #
    # This was `pure = "torch" not in sys.modules` followed by a print of the result, and NOTHING
    # ELSE -- no assert. P8: a criterion that cannot go red is not a criterion, and this one could
    # not even go red in principle, because both outcomes printed and both passed. Worse, the
    # variable was named `pure`, which is the vocabulary of the zero-dependency rule abolished on
    # 2026-07-20; P7 lists surviving `"torch" not in sys.modules` probes as live residue by name.
    #
    # Kept and PROMOTED TO A CRITERION rather than deleted, because unlike a purity boast the
    # underlying claim is structural and P7-legitimate: torch here would be the TRAINING tier, and
    # this module sits on the always-on abduction path where models are SERVED FROZEN. Importing a
    # trainer to answer "why is the ground wet" costs VRAM and import time for arithmetic it would
    # not accelerate. That is a statement about THIS path's workload, which is the form P7 requires
    # of a rejection -- not "one fewer dependency".
    #
    # SCOPE, stated because it is what makes the criterion honest rather than a purity score: this
    # says NOTHING about `run_selfchecks`, which imports torch through other modules in the same
    # process and SHOULD -- P7 abolished the purity rule. The claim is only about what THIS
    # module's own import closure drags in, so it is measured exactly that way, against
    # `_TORCH_PRELOADED` captured at the top of this file before any local import. No threshold, no
    # module count, nothing to tune.
    #
    # REACHABLE RED, constructed and run 2026-08-04: adding `import torch` to this module's import
    # block makes `dragged_in` True and this assert exits 1.
    torch_loaded = "torch" in sys.modules
    dragged_in = torch_loaded and not _TORCH_PRELOADED
    print("-" * 92)
    print(
        f"(7) imports   : torch (training tier) loaded={torch_loaded} "
        f"preloaded_by_caller={_TORCH_PRELOADED} -> dragged in by this module = {dragged_in}"
    )
    assert not dragged_in, (
        "the always-on abduction path must not import the TRAINING tier: torch was NOT in "
        "sys.modules before this module's imports and IS now, so this module's closure pulled it in"
    )

    print("=" * 92)
    print(
        "self-check OK: spiking ABDUCTION recovers the argmin-free-energy cause (Bayesian model selection "
        "under the repo's spiking predictive-coding generative model) and returns an HONEST GAP on an "
        "ambiguous observation (no overclaim); the PEARL 3-step counterfactual (abduction -> action/do -> "
        "prediction) changes the outcome iff the intervention differs from the factual latent; the "
        "AbductionExpert abduces a cause for a stored effect over the brain's causal triples or gaps "
        "honestly. Bounded candidate-set abduction over a low-dim model; deterministic, no "
        "actuator route."
    )
    return 0


if __name__ == "__main__":
    main()
