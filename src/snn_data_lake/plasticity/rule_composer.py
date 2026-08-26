"""
Rule composer KERNEL -- specs/022 T051a (P8 invention #3's core): N rules on one
substrate value with TYPED READ/WRITE SETS and per-rule neuter attribution --
the INSTRUMENT'S KERNEL (standing at T051b, when the charge probe becomes its
first live reader; today's readers are zero, said plainly), authored through
T050's eligibility algebra.

WHAT IT IS. The arbitration CONTRACT over declared rules -- not a clever policy:
  RuleSpec(name, rule: AuthoredRule, reads, writes, claim_key=None) -- the tree's
  first structured read/write-set form at the per-rule grain. `reads` must EQUAL
  the AuthoredRule's used factors, cross-validated both directions (a dead read
  declaration is refused; an undeclared use is impossible by construction).
  `writes` is LOAD-BEARING: the composer routes its substrate by it (below) --
  the first landing validated writes and then ignored it, and the panel measured
  a theta-targeted update deposited into the weight with no refusal; a typed set
  nothing consumes is decoration, the same bar this module's own docstring sets
  for reads. `claim_key` is the OPTIONAL binding to the rule's GRADIENT_CLAIM
  declaration -- T050 promised "the composer will hold BOTH declarations about
  one rule to each other"; the SLOT exists now, the cross-check mechanism is
  DATED to T051c (whose row names GRADIENT_CLAIM per rule), disclosed.

  RuleComposer(policy, target) -- `target` is REQUIRED (no default): it is BOTH
  the uniform write target every added spec must declare AND the substrate
  factor name the policies rebind. The first landing hardcoded the literal "w"
  at four production sites, and the panel measured the sequential policy
  SILENTLY DEGRADING TO SUM for any rule whose substrate factor was named
  anything else (the rebind was a string match); deriving the name from the
  declared write set is what makes the write set typed rather than decorative.
  Policies v1: "sum" (simultaneous -- every rule sees the same pre-step
  substrate) and "sequential" (ordered -- each rule sees the substrate the
  previous one left); unknown names refused; richer policies are later tasks'
  work under this contract.

THE NAIVE-SUM RED, stated at its MEASURED boundary (the panel's sweep): for the
decay family dw = c*w, a single rule converges iff -2 < c < 0; the naive sum of
two converges iff |1 + 2c| < 1. On (-1, 0) the sum is FINE -- it even converges
faster than either rule alone (c=-0.5: 1 step vs 30) -- so "naive summation of
individually-stable rules is wrong" holds only PAST the summed-magnitude
boundary. The harness drives BOTH sides: c=-1.0 (the critical point: a permanent
period-2 orbit, ENTAILED by construction and driven 100 steps to show it does
not decay) and c=-1.5 (the open-interval red: each rule alone converges, the
sum genuinely DIVERGES, |w| strictly increasing) with the sequential control
converging in both. The REAL instance already measured in this tree is T048's
structural-write x consolidation-cascade conflict, MODELED at T051b-ii at the
COMPONENT grain (harness section 1b: incumbent diffuse-step x incumbent prune,
composed; the wrapper, relax() -- the actual T048 erasure -- and any live path
stay undriven; the tissue wire is T054's).

ATTRIBUTION. last_attribution() returns the latest step's per-rule dw
decomposition, cumulative |dw| totals (cumulative_abs_dw -- TOTALS, not shares),
a "stepped" flag (never-stepped is not the same fact as measured-zero), and the
policy's GUARANTEE: "causal-decomposition" under sum (with per-step frozen
bindings the ordered-other-parts predictor is exact by construction -- the
harness proves it to the bit at 3 rules x 3 ablations, and proves the
total-minus-part form BIT-FAILS on a constructed 5-rule point, so reverting the
predictor reds) or "order-coupled" under sequential (leave-one-out is MEASURED
by re-running, never predicted -- the first landing labelled this key
"causality" and derived it from the policy NAME; a sequential composition of
substrate-independent rules is exactly causal and was labelled otherwise, so
the key now names what it actually is: the policy's guarantee, not a per-step
measurement).

P2 ADJUDICATION (BOTH incumbents, named): the tree's live N-rules-one-synapse
compositions are plasticity_rule_stack's PlasticityRuleStack AND its base
HomeostaticPlasticityStack (a second live composition in its own module: the
short-term-gain -> R-STDP -> homeostatic-scaling + intrinsic sandwich; the
subclass extends it with the BCM metaplastic gate -- which runs AFTER the base
short-term gain, not first -- the E/I trim, and Benna-Fusi consolidation, with
the theta slide running after consolidation in the POST phase; per-arm
diagnostics ride an untyped ctx dict, so "no typed sets" holds while "exposes
nothing" would not). This kernel is the DECLARED generalization; all three
survive today at different grains, and the MERGE PATH -- expressing BOTH
stacks' sandwiches as RuleSpecs held to equivalence the way T050 held r_stdp --
is dated to T051b, whose row names it. wholistic_loop's "arbitration" is
FACULTY-grain tick arbitration, distinct.

NO TISSUE, NO WIRE: dated consumers -- T051b (incumbent discharges + the charge
probe as the instrument's first live reader), T053 (the first tick-charged
composed rule -- its row names the composition). Charged surface until then:
this gate-run self-check.

  PYTHONPATH=src python -m snn_data_lake.plasticity.rule_composer
"""

import math
import threading

from snn_data_lake.plasticity.eligibility_algebra import AuthoredRule

# distinct-from: plasticity/plasticity_rule_stack.py -- live incumbent composition
#   #1 (the extended sandwich); adjudicated in the module docstring, merge dated
#   T051b. Site read.
# distinct-from: plasticity/homeostatic_plasticity_stack.py -- live incumbent
#   composition #2 (the BASE sandwich the first landing's census missed -- the
#   T046a all-incumbents lesson); same adjudication, same dated merge. Site read.
# distinct-from: cortex/wholistic_loop.py -- FACULTY-grain tick arbitration (a
#   metacognitive spine over tasks); this composes RULES on one substrate. Site read.
# distinct-from: plasticity/eligibility_algebra.py -- the AUTHORING contract this
#   kernel consumes; rules arrive as AuthoredRule, never re-parsed. Site read.
# distinct-from: substrate/wiring_manifest.py -- ports/edges between faculties;
#   RuleSpec's reads/writes are per-RULE factor/target sets. Producer references
#   composed with the manifest land at T051b. Site read.
# distinct-from: substrate/gradient_claim_gate.py -- the per-SYMBOL claim system;
#   RuleSpec.claim_key is the SLOT binding a rule to its claim, and the
#   cross-check holding both declarations to each other is dated T051c. Site read.
# distinct-from: substrate/basic_components.py -- compose_step is a STAGE pipeline
#   (encoder -> neuron -> connection -> one rule), not N rules on one substrate;
#   named here because the sibling module's convention names it. Site read.

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "RuleComposer.step": (
        "NOT_A_RULE",
        "applies N declared AuthoredRules under a named composition policy and "
        "records the per-rule decomposition; owns no learning rate, no traces, no "
        "claim of its own -- every rule it composes carries its own GRADIENT_CLAIM",
    ),
}

POLICIES = ("sum", "sequential")


class RuleSpec:
    """One rule's typed declaration. Frozen. `claim_key`: the optional
    GRADIENT_CLAIM key of the rule's owning symbol (the T051c cross-check's
    slot; carrying it is honest even before the check exists)."""

    __slots__ = ("name", "rule", "reads", "writes", "claim_key")

    def __init__(self, name, rule, reads, writes, claim_key=None):
        if not (isinstance(name, str) and name.strip()):
            raise ValueError("RuleSpec name must be a non-empty string")
        if not isinstance(rule, AuthoredRule):
            raise ValueError(
                f"spec {name!r}: rule must be an AuthoredRule (authored through the "
                f"eligibility algebra), got {type(rule).__name__}"
            )
        # a bare string iterates into per-character targets ('bias' -> i,s,a,b --
        # all valid identifiers, silently wrong); the sibling module refuses the
        # same shape for factors, and the guard was dropped in the first landing
        for label, val in (("reads", reads), ("writes", writes)):
            if isinstance(val, (str, bytes)):
                raise ValueError(f"spec {name!r}: {label} must be a set of names, not a string")
        if claim_key is not None and not (isinstance(claim_key, str) and claim_key.strip()):
            raise ValueError(f"spec {name!r}: claim_key must be None or a non-empty string")
        reads = frozenset(reads)
        writes = frozenset(writes)
        used = {f for _c, fs in rule.terms for f in fs}
        dead = sorted(reads - used)
        if dead:
            raise ValueError(
                f"spec {name!r}: declared reads {dead} are never used by the rule -- "
                "a dead declaration reads as coverage (refused)"
            )
        missing = sorted(used - reads)
        if missing:
            raise ValueError(
                f"spec {name!r}: the rule uses factors {missing} the read set does not declare"
            )
        if not writes:
            raise ValueError(f"spec {name!r}: an empty write set composes nothing")
        for w in writes:
            if not (isinstance(w, str) and w.isidentifier()):
                raise ValueError(f"spec {name!r}: write target {w!r} must be an identifier")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "claim_key", claim_key)

    def __setattr__(self, key, value):
        raise AttributeError(f"RuleSpec is frozen ({key!r} is not assignable)")


class RuleComposer:
    """N declared rules -> one substrate value, under a NAMED policy, onto a
    DECLARED target. Sealed by use: add() refuses after the first step (mixing
    attribution epochs silently was the panel's construct). Free-threaded: one
    lock covers add/step/last_attribution (the unlocked first landing measured
    1.26%% lost cumulative updates and 8.42%% torn share reads under GIL=False)."""

    def __init__(self, policy, target, _loo_of=None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r} (contract: {POLICIES})")
        if not (isinstance(target, str) and target.isidentifier()):
            raise ValueError(
                f"target must be an identifier naming the substrate factor/write target, got {target!r}"
            )
        self._policy = policy
        self._target = target
        self._specs = []
        self._names = set()
        self._last = None  # (total_dw, {name: dw}) of the latest step
        self._cum = {}  # name -> cumulative |dw| (bounded: one float per rule)
        self._loo_of = _loo_of  # set ONLY by neutered(): permits the empty LOO arm
        self._lock = threading.Lock()

    def add(self, spec):
        if not isinstance(spec, RuleSpec):
            raise ValueError(f"add() takes a RuleSpec, got {type(spec).__name__}")
        if spec.writes != {self._target}:
            raise ValueError(
                f"spec {spec.name!r} writes {sorted(spec.writes)}; this composer composes "
                f"onto {self._target!r} -- a single-substrate step cannot route a different "
                "target, so it refuses rather than silently coercing (the first landing "
                "deposited a theta-targeted update into the weight)"
            )
        with self._lock:
            if self._last is not None:
                raise ValueError(
                    "composer is sealed by use: adding rules after step() would mix "
                    "attribution epochs silently (cumulative totals would span different "
                    "compositions with no marker)"
                )
            if spec.name in self._names:
                raise ValueError(f"duplicate rule name {spec.name!r}")
            self._specs.append(spec)
            self._names.add(spec.name)

    def step(self, w, bindings):
        """One composed update on substrate value `w`. Returns (w', parts).

        Bindings are frozen to float ONCE per step (the first landing
        re-snapshotted per rule, so a stateful __float__ gave different rules
        different values while the attribution still claimed exactness). A
        caller binding the target to a DIFFERENT value than the positional `w`
        is refused -- the first landing silently resolved the disagreement."""
        if not isinstance(bindings, dict):
            raise ValueError(f"bindings must be a dict, got {type(bindings).__name__}")
        wf = float(w)
        if not math.isfinite(wf):
            raise ValueError(f"non-finite substrate value {w!r}")
        base = {}
        for k, v in bindings.items():
            try:
                base[k] = float(v)
            except (TypeError, ValueError) as ex:
                raise ValueError(f"binding {k}={v!r} is not a real scalar ({ex})") from None
        if self._target in base and base[self._target] != wf:
            raise ValueError(
                f"bindings[{self._target!r}]={base[self._target]!r} disagrees with the "
                f"positional substrate {wf!r} -- refusing to silently pick one"
            )
        with self._lock:
            if not self._specs:
                if self._loo_of is not None:
                    # the leave-one-out arm of a single-rule composer: the identity
                    # trajectory IS the well-defined LOO base case (the first landing
                    # refused it, forcing every caller to special-case the identity)
                    self._last = (0.0, {})
                    return wf, {}
                raise ValueError("a composer with no rules composes nothing")
            if self._policy == "sequential" and not any(
                self._target in s.reads for s in self._specs
            ):
                raise ValueError(
                    f"sequential composition where NO rule reads the target {self._target!r} "
                    "is policy-indistinguishable from sum -- refusing to label it sequential "
                    "(the first landing silently degraded, measured bit-identical to sum)"
                )
            parts = {}
            if self._policy == "sum":
                total = None
                for spec in self._specs:
                    b = dict(base)
                    if self._target in spec.reads:
                        b[self._target] = wf  # every rule sees the SAME pre-step substrate
                    dw = spec.rule.evaluate(**{k: v for k, v in b.items() if k in spec.reads})
                    parts[spec.name] = dw
                    total = dw if total is None else total + dw
                new_w = wf + total
            else:  # sequential
                running = wf
                for spec in self._specs:
                    b = dict(base)
                    if self._target in spec.reads:
                        b[self._target] = running  # the substrate the previous rule left
                    dw = spec.rule.evaluate(**{k: v for k, v in b.items() if k in spec.reads})
                    parts[spec.name] = dw
                    running = running + dw
                new_w = running
                total = new_w - wf
            if not math.isfinite(new_w):
                raise ValueError(
                    f"composed update produced a non-finite RESULT {new_w!r} from finite "
                    f"per-rule dw {parts} -- the per-rule output guard does not cover the "
                    "composition (measured: two 1.5e308 rules under sum returned inf "
                    "silently and permanently poisoned the cumulative counters)"
                )
            self._last = (total, dict(parts))
            for name, dw in parts.items():
                self._cum[name] = self._cum.get(name, 0.0) + abs(dw)
        return new_w, dict(parts)

    def last_attribution(self):
        """The attribution kernel's read surface (per-call today; its first LIVE
        reader is T051b's charge-probe integration, dated): the latest step's
        per-rule decomposition, cumulative |dw| TOTALS (cumulative_abs_dw -- not
        shares; divide by their sum if a share is wanted), whether a step has
        happened at all (never-stepped != measured-zero), and the policy's
        GUARANTEE -- a property of the policy, deliberately not called
        'causality' (the first landing derived that label from the policy name;
        a sequential composition of substrate-independent rules is exactly
        causal and was labelled otherwise)."""
        with self._lock:
            stepped = self._last is not None
            total, parts = self._last if stepped else (None, {})
            return {
                "policy": self._policy,
                "target": self._target,
                "guarantee": "causal-decomposition" if self._policy == "sum" else "order-coupled",
                "stepped": stepped,
                "last_total": total,
                "last_parts": dict(parts),
                "cumulative_abs_dw": dict(self._cum),
            }

    def neutered(self, name):
        """A NEW composer identical to this one with rule `name` removed -- the
        leave-one-out arm. A single-rule composer's LOO is the IDENTITY
        trajectory (w' = w, parts = {}), returned as such; its cumulative
        counters start fresh ON PURPOSE (a LOO arm's history is its own).
        Refuses unknown names; a hand-built empty composer still refuses to
        step (provenance separates the valid LOO base case from a caller
        error)."""
        with self._lock:
            if name not in self._names:
                raise ValueError(f"cannot neuter unknown rule {name!r}")
            out = RuleComposer(self._policy, self._target, _loo_of=name)
            for spec in self._specs:
                if spec.name != name:
                    out.add(spec)
            return out


# ---- self-check ----------------------------------------------------------------------------
def main():
    import struct

    from snn_data_lake.plasticity.eligibility_algebra import EligibilityAlgebra

    print("=" * 96)
    print("rule_composer -- specs/022 T051a: the composition kernel (typed sets + attribution)")
    print("=" * 96)

    def bits(x):
        return struct.pack(">d", x)

    def decay_rule(alg, name, c):
        return RuleSpec(name, alg.author(name, [(c, ("w",))]), reads={"w"}, writes={"w"})

    alg = EligibilityAlgebra(term_budget=4)
    alg.declare_factor("w", "the substrate value itself (the decay family's one factor)")

    # 1) THE NAIVE-SUM RED, both sides of the measured boundary (|1 + sum c| >= 1).
    #    c=-1.0: the critical point -- a permanent period-2 orbit (ENTAILED by
    #    construction on exact floats; driven 100 steps to show it does not decay).
    #    c=-1.5: the OPEN-INTERVAL red -- each rule alone converges, the naive sum
    #    genuinely DIVERGES with |w| strictly increasing. Sequential converges in
    #    both. And the disclosure the first landing lacked: on (-1, 0) the naive
    #    sum is FINE (c=-0.5 converges in 1 step vs a single rule's 30).
    for c, want in ((-1.0, "orbit"), (-1.5, "diverge"), (-0.5, "converge-faster")):
        single = RuleComposer("sum", target="w")
        single.add(decay_rule(alg, "d0", c))
        w = 1.0
        for _ in range(100):
            w, _p = single.step(w, {"w": w})
        assert abs(w) < 1e-6 or c == -1.0 and w == 0.0 or abs(w) < 1.0, (
            f"single rule at c={c} must converge, got {w!r}"
        )
        pair = RuleComposer("sum", target="w")
        pair.add(decay_rule(alg, "d0", c))
        pair.add(decay_rule(alg, "d1", c))
        w = 1.0
        traj = []
        for _ in range(100):
            try:
                w, _p = pair.step(w, {"w": w})
            except ValueError:
                traj.append(float("inf"))  # the composed-total guard fired mid-divergence
                break
            traj.append(w)
        if want == "orbit":
            assert all(abs(v) == 1.0 for v in traj), f"c=-1.0 must orbit at |w|=1: {traj[:4]}"
        elif want == "diverge":
            mags = [abs(v) for v in traj if math.isfinite(v)]
            assert all(b > a for a, b in zip(mags, mags[1:])) and (
                traj[-1] == float("inf") or mags[-1] > 1e6
            ), f"c=-1.5 sum must strictly diverge: {traj[:5]}..."
        else:
            assert traj[0] == 0.0, (
                f"c=-0.5 sum must converge in ONE step (faster than alone): {traj[0]!r}"
            )
        seq = RuleComposer("sequential", target="w")
        seq.add(decay_rule(alg, "d0", c))
        seq.add(decay_rule(alg, "d1", c))
        w = 1.0
        for _ in range(100):
            w, _p = seq.step(w, {"w": w})
        assert abs(w) < 1e-6, f"sequential control at c={c} must converge, got {w!r}"
    print(
        "  RED    naive sum at the MEASURED boundary: c=-1.0 orbits (entailed, driven 100 steps), "
        "c=-1.5 DIVERGES on the open interval while each rule alone and the sequential control "
        "converge; c=-0.5 disclosed convergent-and-faster (the sum is only wrong PAST |1+sum c|>=1)"
    )

    # 1b) T051b-ii: T048's two-writer conflict pair MODELED AT THE COMPONENT GRAIN
    #     (structural_rewiring x the Benna-Fusi cascade), expressed through the kernel.
    #     Both component rules are AUTHORED from their incumbents and HELD to the
    #     incumbent COMPONENT DYNAMICS (diffuse()'s fast-beaker step; rewire()'s
    #     prune), then composed. Said plainly, what this arm does NOT drive (the
    #     panel's finding -- "the REAL instance" was the first draft's overclaim):
    #     the _BennaFusiConsolidation wrapper (equilibrium init where the drag term
    #     is exactly 0, w_max=4.0, delta-vs-prev deposit); relax() -- T048's ACTUAL
    #     erasure mechanism, which REBINDS rows from the deep store wholesale and
    #     produces NO sign crossing (measured -0.34 then -0.289); the regen-washed-
    #     back-in-2-steps case (regens are rng-steered, disclosed out); and any live
    #     path at all -- StructuralRewiring has ZERO live callers (T048 reverted the
    #     wire whole; the tissue wire is T054's). What the composition shows at this
    #     grain: the naive SUM mispredicts the warm bare-cascade post-state by
    #     exactly k*w and at a pruned excitatory slot drives the weight NEGATIVE
    #     (the birth-sign violation the structural module exists to prevent), while
    #     SEQUENTIAL reproduces the bare-cascade component step to the bit. On the
    #     T048 fix ("a structural write must also reach the cascade"): that write
    #     touches a SECOND substrate (the cascade state), and kernel v1's single
    #     write target cannot hold both in one spec -- the existing target guard
    #     (exercised by the refusal battery below at the theta construction)
    #     refuses writes != {target}, so the fix is representable only as
    #     per-substrate composition. A scope statement; the first draft dressed a
    #     re-exercise of that same guard as "the adjudication record" and the
    #     panel measured it indistinguishable from the shipped construction --
    #     deleted, not weakened. No live stack behavior changes here.
    from snn_data_lake.plasticity.benna_fusi import BennaFusiSynapse
    from snn_data_lake.plasticity.structural_rewiring import StructuralRewiring

    bf = BennaFusiSynapse()
    k_drag = bf.dt * bf.g[0] / bf.c[0]  # instance-read, never retyped (T050 discipline)
    assert 0.0 < k_drag <= 1.0, (
        f"the drag coefficient must make diffuse a convex step (k={k_drag!r}) -- past 1.0 "
        "the clip becomes reachable on the drag path and the in-box declaration narrows"
    )
    # the bit-equality arm below needs dt EXACTLY a power of two: the reassociation
    # fl(dt*fl(g0*u1)) vs fl(fl(dt*g0)*u1) is exact only under exact scaling (panel-
    # measured bit-equal at dt in {0.25, 0.5, 1, 2}, NOT at {0.3, 0.7, 1.3}); c[0]
    # is STRUCTURALLY 1.0 (1.0 * c_ratio**0) and is not a condition at all -- the
    # first draft credited "dt=c0=1", half wrong and half vacuous.
    assert math.frexp(bf.dt)[0] == 0.5, (
        f"dt={bf.dt!r} is not a power of two -- the sequential==cascade assert below "
        "is bit-exact only under exact dt scaling; a non-dyadic dt needs a tol arm"
    )
    alg_ii = EligibilityAlgebra(term_budget=2)
    alg_ii.declare_factor("w", "the fast beaker u[0] / the weight row slot (the shared substrate)")
    alg_ii.declare_factor("u1", "the second beaker's level (the deep anchor's near end)")
    alg_ii.declare_factor(
        "s_target", "the structural target: 0.0 for a prune, +-seed-mag for a regen"
    )
    drag_rule = alg_ii.author("bf.anchor_drag", [(k_drag, ("u1",)), (-k_drag, ("w",))])
    struct_rule = alg_ii.author("sr.structural_write", [(1.0, ("s_target",)), (-1.0, ("w",))])

    # anchor-drag equivalence vs the incumbent diffuse(), IN-BOX states: the incumbent
    # computes u0 + dt*(g0*(u1-u0))/c0-shaped flow from a PRE-STATE snapshot (so u[0]'s
    # one-step update depends on (u0, u1) only -- panel-verified bit-identical across
    # u[2] settings); the declaration computes k*u1 - k*w. Different association ->
    # ulp-scale tol. Clip coverage needs BOTH halves (the first draft asserted only
    # one): k <= 1 (asserted above) AND endpoints inside [-w_max, w_max] (asserted
    # per point below) make the update a convex combination, so the clip cannot fire
    # on the drag path IN-BOX -- and diffuse() does clip (benna_fusi clips inside the
    # loop; a constructed out-of-box u1=40 drives declared 3.9 vs observed 0.0), but
    # out-of-box u is unreachable through the module's own API, disclosed. The grid's
    # u0=1.0 column sits ON the box boundary, not in its interior.
    import sys as _sys2

    _EPS2 = _sys2.float_info.epsilon
    worst_drag = 0.0
    n_drag = 0
    _drag_obs = []
    for u0 in (-0.9, -0.3, 0.0, 0.17, 0.5, 1.0):
        for u1 in (-0.8, -0.1, 0.0, 0.25, 0.9):
            assert abs(u0) <= bf.w_max and abs(u1) <= bf.w_max, "sweep left the box"
            bf2 = BennaFusiSynapse()
            bf2.u[0] = u0
            bf2.u[1] = u1
            declared = drag_rule.evaluate(w=u0, u1=u1)
            bf2.diffuse(1)
            observed = bf2.u[0] - u0
            # 8 ulps of the term-magnitude scale: the reassociation depth is 3
            # roundings; the band's red is measured REACHABLE (a 13-16 ulp
            # coefficient error reds; the k+16-ulp control below drives it).
            tol2 = 8.0 * _EPS2 * (abs(k_drag * u1) + abs(k_drag * u0) + abs(u0))
            d2 = abs(declared - observed)
            assert d2 <= tol2, (
                f"anchor-drag declaration diverges from diffuse() at (u0={u0}, u1={u1}): "
                f"|d|={d2!r} > {tol2:.1e}"
            )
            worst_drag = max(worst_drag, d2)
            n_drag += 1
            _drag_obs.append((u0, u1, observed, tol2))
    # in-command POSITIVE CONTROL (the sweep's reachable red): a coefficient
    # mis-declared by 32 ulps must violate the same band on the same grid.
    _alg_bad2 = EligibilityAlgebra(term_budget=2)
    _alg_bad2.declare_factor("w", "control copy (mis-declared drag)")
    _alg_bad2.declare_factor("u1", "control copy (mis-declared drag)")
    _k_bad = k_drag * (1.0 + 32.0 * _EPS2)
    _drag_bad = _alg_bad2.author(
        "bf.anchor_drag.misdeclared", [(_k_bad, ("u1",)), (-_k_bad, ("w",))]
    )
    _ctrl_hit = False
    for u0, u1, observed, tol2 in _drag_obs:
        if abs(_drag_bad.evaluate(w=u0, u1=u1) - observed) > tol2:
            _ctrl_hit = True
            break
    assert _ctrl_hit, "the 32-ulp mis-declared drag must violate the band (dead sweep otherwise)"

    # structural-write equivalence vs the incumbent rewire(), REAL prune sites made
    # observable BY CONSTRUCTION (the panel's fix): alongside each sign flip, a
    # previously-vacant slot is externally populated -- the module's own documented
    # ZERO->LIVE re-adoption path -- so live == budget after the prune loop,
    # deficit == 0, regenerated == 0, and the seeded regen RNG is NEVER CONSULTED.
    # The first draft diluted the vacancy pool and disclosed "the floor still
    # binds"; the panel measured that FALSE (7-9% of neighboring seed pairs fell
    # below the floor, and +14 behavior-neutral RNG draws inside rewire() reddened
    # it) -- the floor is now structural, not a seed draw. The first-sight prune
    # count is NOT asserted: first sight derives the signs from these same rows,
    # so a first-sight prune is structurally impossible (an assert there had no
    # reachable red and was deleted, not weakened). Regens stay disclosed out of
    # the arm entirely (regenerated == 0 is asserted).
    import random as _random2

    _rng2 = _random2.Random(7)
    rows2 = []
    for _r in range(4):
        row_v = []
        for _c in range(12):
            if _rng2.random() < 0.5:
                row_v.append(0.0)  # vacancies: the re-adoption slots below
            else:
                row_v.append(_rng2.uniform(0.2, 1.0) * (1 if _rng2.random() < 0.5 else -1))
        rows2.append(row_v)
    sr2 = StructuralRewiring(seed=3)
    sr2.rewire(rows2)  # first sight: birth signs recorded (cannot prune, see above)
    flipped = []
    vacant0 = [(r3, c3) for r3, row3 in enumerate(rows2) for c3, v3 in enumerate(row3) if v3 == 0.0]
    for r2 in range(4):
        for c2 in range(12):
            w_old2 = rows2[r2][c2]
            if w_old2 != 0.0:
                rows2[r2][c2] = -w_old2  # drift across zero against the birth sign
                flipped.append((r2, c2, -w_old2))
                break
    assert len(flipped) == 4, (
        f"expected one flip per row, got {len(flipped)} -- an all-zero row changed the substrate"
    )
    assert len(vacant0) >= len(flipped), "not enough vacant slots for the re-adoptions"
    for _i3, (r3, c3) in enumerate(vacant0[: len(flipped)]):
        rows2[r3][c3] = 0.3 + 0.01 * _i3  # external ZERO->LIVE: re-adopted, fills the budget
    rep2 = sr2.rewire(rows2)
    assert (
        rep2["pruned"] == len(flipped)
        and rep2["regenerated"] == 0
        and rep2["readopted"] == len(flipped)
    ), (
        f"the deficit-zero construction must hold exactly (pruned={rep2['pruned']}, "
        f"regenerated={rep2['regenerated']}, readopted={rep2['readopted']}, flips={len(flipped)})"
    )
    n_prune = 0
    for r2, c2, w_flip in flipped:
        post = rows2[r2][c2]
        dw2 = struct_rule.evaluate(w=w_flip, s_target=0.0)
        # the EQUIVALENCE (red reachable: an incumbent prune-target change fails
        # HERE, correctly named -- the first draft compared w+dw to a constant 0.0,
        # an IEEE identity true for any w, with the incumbent only filtering sites)
        assert (w_flip + dw2) == post, (
            f"declared prune diverges from the incumbent's post-state: {w_flip + dw2!r} vs {post!r}"
        )
        assert post == 0.0, f"the incumbent's prune target moved: {post!r}"
        n_prune += 1

    # THE COMPOSED MEASUREMENT: prune an inhibitory slot (w=-0.4) anchored at a
    # positive near-deep level (u1=+0.2). The control is the warm BARE cascade
    # driven through its own API: deposit(-w) vacates, diffuse(1) pulls back
    # toward the anchor -- the component-grain ANALOG of T048's regen wash-back
    # (which this arm does not drive; see the section header).
    w0_ii, u1_ii = -0.4, 0.2
    bf3 = BennaFusiSynapse()
    bf3.u[0] = w0_ii
    bf3.u[1] = u1_ii
    bf3.deposit(0.0 - w0_ii)  # the structural prune, expressed through the incumbent API
    _vacated = bf3.u[0]
    bf3.diffuse(1)
    live_post = bf3.u[0]

    comp_seq = RuleComposer("sequential", target="w")
    comp_seq.add(RuleSpec("structural", struct_rule, reads={"s_target", "w"}, writes={"w"}))
    comp_seq.add(RuleSpec("drag", drag_rule, reads={"u1", "w"}, writes={"w"}))
    w_seq, _p_seq = comp_seq.step(w0_ii, {"w": w0_ii, "s_target": 0.0, "u1": u1_ii})

    comp_sum = RuleComposer("sum", target="w")
    comp_sum.add(RuleSpec("structural", struct_rule, reads={"s_target", "w"}, writes={"w"}))
    comp_sum.add(RuleSpec("drag", drag_rule, reads={"u1", "w"}, writes={"w"}))
    w_sum, _p_sum = comp_sum.step(w0_ii, {"w": w0_ii, "s_target": 0.0, "u1": u1_ii})

    assert _vacated == 0.0, "the bare-cascade prune must vacate the fast beaker exactly"
    assert bits(w_seq) == bits(live_post), (
        f"SEQUENTIAL must reproduce the bare-cascade component step to the bit "
        f"(dt a power of two, asserted above): {w_seq!r} vs cascade {live_post!r}"
    )
    assert w_seq > 0.0 and w0_ii < 0.0, (
        "the component-grain wash-back crossing must reproduce: an inhibitory prune "
        "anchored at a positive near-deep level re-crosses positive (the ANALOG of "
        "T048's regen wash-back, which this arm does not drive)"
    )
    _sum_err = abs(w_sum - live_post)
    _pred_err = abs(k_drag * w0_ii)  # the sum's mispredict is exactly k*w
    assert _sum_err > 0.0 and abs(_sum_err - _pred_err) <= 8.0 * _EPS2 * _pred_err + 1e-300, (
        f"the naive sum must mispredict the bare-cascade post-state by exactly |k*w|: "
        f"err={_sum_err!r} vs k*|w|={_pred_err!r}"
    )

    # the EXCITATORY instance carries the sign violation (the first draft of this
    # assert put it on the inhibitory case and was WRONG at its own point -- there
    # the sum overshoots on the same side; asserted above as the exact k*w
    # mispredict): pruning w=+1 anchored at u1=+0.25, the naive sum lands at
    # k*(u1 - w) < 0 -- it drives a pruned excitatory slot NEGATIVE, the violation
    # the structural module's birth-sign identity exists to prevent, while the
    # bare cascade (and sequential) stay positive.
    w0_ex, u1_ex = 1.0, 0.25
    bf4 = BennaFusiSynapse()
    bf4.u[0] = w0_ex
    bf4.u[1] = u1_ex
    bf4.deposit(0.0 - w0_ex)
    bf4.diffuse(1)
    live_ex = bf4.u[0]
    comp_sum2 = RuleComposer("sum", target="w")
    comp_sum2.add(RuleSpec("structural", struct_rule, reads={"s_target", "w"}, writes={"w"}))
    comp_sum2.add(RuleSpec("drag", drag_rule, reads={"u1", "w"}, writes={"w"}))
    w_sum_ex, _p2 = comp_sum2.step(w0_ex, {"w": w0_ex, "s_target": 0.0, "u1": u1_ex})
    assert w_sum_ex < 0.0 < live_ex, (
        f"at a pruned EXCITATORY slot the naive sum must cross negative while the bare "
        f"cascade stays positive (sum {w_sum_ex!r} vs live {live_ex!r})"
    )
    # NO dual-write refusal arm here, deliberately: the first draft dressed the
    # kernel's generic writes-!=-target guard (already constructed by the refusal
    # battery below at the theta case) as "the T048 adjudication record", with a
    # bare "write" substring the panel measured satisfiable by FOUR non-dual
    # constructions including a single-wrong-target spec -- deleted as duplicative
    # (one mechanism per claim). The scope FACT stands in the section header: the
    # T048 fix writes the cascade state too, and kernel v1 holds one write target
    # per spec, so that fix is representable only as per-substrate composition.
    print(
        f"  RED    the T048 pair at COMPONENT grain (T051b-ii): structural x anchor-drag, both "
        f"authored from their incumbents (drag vs diffuse() worst |d|={worst_drag:.1e} on "
        f"{n_drag} in-box states, 32-ulp mis-declaration control REDS; structural vs rewire() "
        f"post-state-equal on {n_prune} REAL prune sites, deficit-zero construction: regen 0, "
        f"readopted {len(flipped)}) -- SEQUENTIAL reproduces the bare-cascade component step to "
        f"the bit ({w_seq:.6f}, the inhibitory wash-back ANALOG); the naive SUM mispredicts by "
        f"exactly |k*w| ({w_sum:.6f} vs cascade {live_post:.6f}) and at the EXCITATORY instance "
        f"crosses NEGATIVE ({w_sum_ex:.6f} vs {live_ex:.6f}); wrapper/relax()/regen-case/live-"
        f"path NOT driven (relax rebinds wholesale -- no crossing; zero live callers; T054)"
    )

    # 2) Causal decomposition under sum, and the UNSOUND predictor's reachable red.
    #    The ordered-other-parts predictor is bit-exact (3 rules x 3 ablations,
    #    frozen bindings); the total-minus-part form is float-unsound -- proven by
    #    CONSTRUCTION on a 5-rule point where it BIT-FAILS (the first landing's
    #    comment claimed it 'reds at 3 rules', measured false at its own point;
    #    reverting the predictor now reds HERE).
    alg2 = EligibilityAlgebra(term_budget=4)
    alg2.declare_factor("w", "substrate")
    alg2.declare_factor("g", "a gain factor")
    a = RuleSpec("A", alg2.author("A", [(-0.3, ("w",))]), reads={"w"}, writes={"w"})
    bsp = RuleSpec("B", alg2.author("B", [(0.7, ("g", "w"))]), reads={"g", "w"}, writes={"w"})
    csp = RuleSpec("C", alg2.author("C", [(0.11, ("w",))]), reads={"w"}, writes={"w"})
    comp = RuleComposer("sum", target="w")
    for s in (a, bsp, csp):
        comp.add(s)
    w0 = 0.8
    _w1, parts = comp.step(w0, {"w": w0, "g": 0.5})
    order = ["A", "B", "C"]
    for name in order:
        loo_w, _ = comp.neutered(name).step(w0, {"w": w0, "g": 0.5})
        pred_total = None
        for other in order:
            if other == name:
                continue
            pred_total = parts[other] if pred_total is None else pred_total + parts[other]
        assert bits(loo_w) == bits(w0 + pred_total), f"ordered-parts predictor failed at {name}"
    # the unsound form's constructed red: five rules whose parts defeat re-association
    alg5 = EligibilityAlgebra(term_budget=8)
    alg5.declare_factor("w", "substrate")
    coeffs = [1.0, 1e16, -1e16, 3.0, 7.0]
    comp5 = RuleComposer("sum", target="w")
    for i, c in enumerate(coeffs):
        comp5.add(RuleSpec(f"r{i}", alg5.author(f"r{i}", [(c, ("w",))]), reads={"w"}, writes={"w"}))
    _w5, parts5 = comp5.step(1.0, {"w": 1.0})
    att5_total = sum(parts5[f"r{i}"] for i in range(5))
    hit = False
    for name in parts5:
        loo_w, _ = comp5.neutered(name).step(1.0, {"w": 1.0})
        unsound = 1.0 + (att5_total - parts5[name])
        if bits(loo_w) != bits(unsound):
            hit = True
            break
    assert hit, (
        "no 5-rule ablation bit-fails the unsound total-minus-part form -- the predictor red is unreachable"
    )
    print(
        "  green  causal decomposition (sum): ordered-other-parts predictor bit-exact 3x3; the "
        "total-minus-part form BIT-FAILS on the constructed 5-rule point (reverting the predictor reds)"
    )

    # 3) The sequential guarantee, swept over NON-DEGENERATE points (the first
    #    landing asserted order-sensitivity at ONE point; leave-one-out coincides
    #    with the naive prediction exactly when g*w0 == 0 -- the zero-gain and
    #    BIRTH-STATE cases -- documented, not asserted against).
    sensitive = 0
    for w0s, gs in ((0.8, 0.5), (0.3, 1.2), (-0.6, 0.9), (1.1, -0.4)):
        comp = RuleComposer("sequential", target="w")
        comp.add(a)
        comp.add(bsp)
        _w1, parts = comp.step(w0s, {"w": w0s, "g": gs})
        att = comp.last_attribution()
        assert att["guarantee"] == "order-coupled"
        loo_w, _ = comp.neutered("A").step(w0s, {"w": w0s, "g": gs})
        naive = w0s + (att["last_total"] - parts["A"])
        if loo_w != naive:
            sensitive += 1
    assert sensitive == 4, f"order sensitivity held on only {sensitive}/4 non-degenerate points"
    comp0 = RuleComposer("sequential", target="w")
    comp0.add(a)
    comp0.add(bsp)
    comp0.step(0.8, {"w": 0.8, "g": 0.0})  # g*w0 == 0: coincidence EXPECTED there
    print(
        "  green  sequential guarantee: order-coupled LOO differs from the naive prediction on 4/4 "
        "non-degenerate points (coincidence at g*w0==0 documented as expected, incl. the birth state)"
    )

    # 4) The attribution surface CONSUMED (write-only state was the T048 deletion
    #    precedent): after 100 sum-steps of two equal decay rules, each rule's
    #    cumulative_abs_dw is EXACTLY the trajectory's |dw| sum and the two are
    #    equal; last_parts mirrors step()'s return; 'stepped' separates
    #    never-polled from measured-zero.
    comp = RuleComposer("sum", target="w")
    comp.add(decay_rule(alg, "d0", -1.0))
    comp.add(decay_rule(alg, "d1", -1.0))
    pre = comp.last_attribution()
    assert pre["stepped"] is False and pre["last_total"] is None, (
        "never-stepped must not read as zero"
    )
    w = 1.0
    expect_abs = 0.0
    for _ in range(100):
        expect_abs += abs(w)  # each rule's |dw| this step is |w|
        w, parts = comp.step(w, {"w": w})
    att = comp.last_attribution()
    assert att["stepped"] is True and att["last_parts"] == parts
    ca = att["cumulative_abs_dw"]
    assert ca["d0"] == ca["d1"] == expect_abs, f"cumulative totals wrong: {ca} vs {expect_abs}"
    print(
        f"  green  attribution consumed: cumulative_abs_dw exact ({expect_abs:.1f} each, equal), last_parts mirrors step(), stepped-flag separates never-polled from zero"
    )

    # 5) REFUSALS by message -- incl. every panel-constructed silent failure.
    for build, must in (
        (
            lambda: RuleSpec(
                "x", alg2.author("x", [(1.0, ("w",))]), reads={"w", "g"}, writes={"w"}
            ),
            "never used by the rule",
        ),
        (
            lambda: RuleSpec("x", alg2.author("x", [(1.0, ("g", "w"))]), reads={"w"}, writes={"w"}),
            "does not declare",
        ),
        (
            lambda: RuleSpec("x", alg2.author("x", [(1.0, ("w",))]), reads={"w"}, writes=set()),
            "empty write set",
        ),
        (
            lambda: RuleSpec("x", alg2.author("x", [(1.0, ("w",))]), reads="w", writes={"w"}),
            "not a string",
        ),
        (
            lambda: RuleSpec("x", alg2.author("x", [(1.0, ("w",))]), reads={"w"}, writes="bias"),
            "not a string",
        ),
        (lambda: RuleSpec("x", "not-a-rule", reads={"w"}, writes={"w"}), "must be an AuthoredRule"),
        (
            lambda: RuleSpec(
                "x", alg2.author("x", [(1.0, ("w",))]), reads={"w"}, writes={"w"}, claim_key=""
            ),
            "claim_key",
        ),
        (lambda: RuleComposer("vote", target="w"), "unknown policy"),
        (lambda: RuleComposer("sum", target="not an id"), "must be an identifier"),
        (lambda: RuleComposer("sum", target="w").step(1.0, {}), "composes nothing"),
        (lambda: comp.neutered("ghost"), "unknown rule"),
        (lambda: RuleComposer("sum", target="theta").add(a), "composes onto 'theta'"),
        (lambda: RuleComposer("sum", target="w").step(1.0, [("w", 1.0)]), "must be a dict"),
    ):
        try:
            build()
            raise AssertionError(f"accepted an invalid construction (wanted {must!r})")
        except ValueError as ex:
            assert must in str(ex), f"refused for the WRONG reason: {ex}"
    # sealed-after-step; duplicate; frozen; target-binding disagreement; the
    # sequential-degrades red; the composed-total overflow through step()
    sealed = RuleComposer("sum", target="w")
    sealed.add(decay_rule(alg, "d0", -0.5))
    sealed.step(1.0, {"w": 1.0})
    for call, must, exc in (
        (lambda: sealed.add(decay_rule(alg, "d1", -0.5)), "sealed by use", ValueError),
        (lambda: sealed.step(1.0, {"w": 0.3}), "disagrees with the positional", ValueError),
    ):
        try:
            call()
            raise AssertionError(f"accepted (wanted {must!r})")
        except exc as ex:
            assert must in str(ex)
    dup = RuleComposer("sum", target="w")
    dup.add(a)
    try:
        dup.add(a)
        raise AssertionError("accepted a duplicate rule name")
    except ValueError as ex:
        assert "duplicate" in str(ex)
    try:
        a.reads = frozenset()
        raise AssertionError("RuleSpec accepted attribute reassignment")
    except AttributeError:
        pass
    alg3 = EligibilityAlgebra(term_budget=2)
    alg3.declare_factor("weight", "a substrate factor NOT named w")
    misnamed = RuleComposer("sequential", target="w")
    misnamed.add(
        RuleSpec("m", alg3.author("m", [(-0.5, ("weight",))]), reads={"weight"}, writes={"w"})
    )
    try:
        misnamed.step(1.0, {"weight": 1.0})
        raise AssertionError(
            "sequential with no target-reading rule stepped (the degrade-to-sum trap)"
        )
    except ValueError as ex:
        assert "policy-indistinguishable" in str(ex)
    algh = EligibilityAlgebra(term_budget=2)
    algh.declare_factor("h", "a huge factor")
    ovf = RuleComposer("sum", target="w")
    for i in range(2):
        ovf.add(
            RuleSpec(f"h{i}", algh.author(f"h{i}", [(1.5e308, ("h",))]), reads={"h"}, writes={"w"})
        )
    try:
        ovf.step(0.0, {"h": 1.0})
        raise AssertionError("a non-finite composed total reached the substrate")
    except ValueError as ex:
        assert "does not cover the composition" in str(ex)
    # solo-LOO identity (the base case, not an error) vs a hand-built empty composer
    solo = RuleComposer("sum", target="w")
    solo.add(decay_rule(alg, "only", -0.5))
    loo_w, loo_parts = solo.neutered("only").step(2.0, {"w": 2.0})
    assert loo_w == 2.0 and loo_parts == {}, "solo LOO must be the identity trajectory"
    print(
        "  RED    refusals BY MESSAGE (13 constructions + 6 behavioural): incl. disjoint-writes routing, "
        "the sequential-degrades-to-sum trap, target-binding disagreement, composed-total overflow, "
        "sealed-after-step; solo-LOO returns the identity (the base case, not an error)"
    )

    print()
    print(
        "DISCLOSURES: attribution is the instrument's KERNEL -- live readers today: "
        "ZERO (no report(), no state() row; the neutered information is recoverable "
        "from step()'s returned parts, so the instrument is not the sole carrier); "
        "the first live reader is T051b's charge-probe integration, dated. "
        "Arbitration ORDER is bare add() insertion order -- not learned, not "
        "recorded in the payload; the register's order disclosure is paid there. "
        "claim_key is a SLOT: the GRADIENT_CLAIM cross-check lands at T051c. "
        "Harness constants (coefficients, budgets, the 100-step horizon, w0/g) are "
        "exercise values under P1 exception (c) -- this module is a wire exemption "
        "with zero live callers; the exception expires with T053's tick-charged "
        "composed rule, the SAME date the sibling algebra's exception now carries "
        "(one expiry, one date). The naive-sum claim is SCOPED: wrong only past "
        "|1 + sum c| >= 1 for the decay family; on (-1,0) the sum converges faster "
        "than either rule alone -- AND (T051b-ii) measured on the T048 conflict pair "
        "at the COMPONENT grain: incumbent diffuse-step x incumbent prune, where the "
        "sum mispredicts the bare-cascade post-state by exactly |k*w| and crosses "
        "sign at an excitatory prune while sequential reproduces the component step "
        "to the bit (exact because dt is a POWER OF TWO, asserted; c[0] is "
        "structurally 1.0 and no condition). NOT driven, said plainly: the "
        "_BennaFusiConsolidation wrapper, relax() -- T048's actual erasure, a "
        "wholesale rebind with no crossing -- the regen case, and any live path "
        "(StructuralRewiring has zero live callers; the wire is T054's). The T048 "
        "fix writes the cascade state too -- a second substrate; kernel v1 holds "
        "one write target per spec, so that fix is representable only as "
        "per-substrate composition (a scope statement; the first draft's "
        "'adjudication record' refusal arm was the generic target guard and was "
        "deleted as duplicative). The live stack is UNCHANGED by ii. Harness "
        "exercise values under P1 exception (c) now also include the prune arm's "
        "seeds/geometry (7, 3; 4x12 at density 0.5), the probe points (w0/u1 "
        "pairs), and the 8/32-ulp band factors. The stack MERGE (both incumbent "
        "sandwiches as RuleSpecs, held to equivalence) is T051b's, named in its row."
    )
    print(
        "self-check OK: boundary-scoped naive-sum reds (orbit + open-interval divergence + "
        "faster-convergence disclosure) + the T048 pair at component grain (authored-from-"
        "incumbent structural x drag; sequential==component step to the bit, sum "
        "sign-violates at the excitatory prune; wrapper/relax/live-path undriven, said) + "
        "causal decomposition with the unsound predictor's constructed bit-red + swept "
        "sequential guarantee + consumed attribution surface + 19 refusals"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
