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
structural-write x consolidation-cascade conflict, arbitrated at T051b (dated).

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
        "than either rule alone. The stack MERGE (both incumbent sandwiches as "
        "RuleSpecs, held to equivalence) is T051b's, named in its row."
    )
    print(
        "self-check OK: boundary-scoped naive-sum reds (orbit + open-interval divergence + "
        "faster-convergence disclosure) + causal decomposition with the unsound predictor's "
        "constructed bit-red + swept sequential guarantee + consumed attribution surface + 19 refusals"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
