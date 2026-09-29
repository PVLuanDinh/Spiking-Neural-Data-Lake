"""
Declared + budgeted eligibility algebra -- specs/022 T050 (research.md D-row): the
sum-of-products authoring contract for three-factor rules.

WHAT IT IS. A CONTRACT, not a learning rule: a three-factor rule is AUTHORED as
    dw = sum_k  c_k * prod(factors_k)
over a vocabulary of DECLARED trace factors (each factor named, role stated at
declaration), under a TERM BUDGET the authoring target supplies -- NO default, the
`lake_engine.q_step` P1 discipline (the executing target's own limit; a first draft
cited a "tau_ticks discipline" here, a precedent that existed nowhere in the tree or
its history -- the panel measured the citation fabricated, and the symbol-precedent
rule failed worse than a rotted line ever has). "Loihi-style", stated honestly: the
sum-of-products FORM Loihi 2's synaptic microcode uses, with a term-count budget --
NO trace model, fixed-point arithmetic, per-product operand limit, saturation
semantics or hardware constant is implemented here, and the budget counts TERMS, not
operands (a one-term product of fifty factors passes budget 1; the operand limit is
the missing half, disclosed).

THE CHARGE: the incumbent `plasticity/r_stdp.py::RSTDPNeuron.reward` (dw_i =
lr * d * e_i) is authored as ONE term and proven BIT-equivalent -- struct-compared,
sign of zero included -- against the incumbent's own production reward() on cases
sampled from the MEASURED live operating ranges (the panel spied 37,995 live reward
calls: lr takes six values {0.02,0.06,0.08,0.1,0.15,0.2}, d spans [-1,+1.95], e
spans [-17.9,+0.95] with 26%% exact zeros -- the first draft's uniform[-2,2] band
never drew a zero and pinned one lr). The equivalence HOLDS STRUCTURALLY (the
coefficient is exactly 1.0 and evaluation keeps authored multiplication order), so
the case sweep is a REGRESSION guard, not the proof -- said plainly, because a case
count implying statistical coverage was the first draft's rhetoric. The
MIS-DECLARATION RED: an lr off by one part in 1e6 reds `equivalence_check` -- run
against the REAL incumbent, through the exported function.

EXPRESSIBILITY CENSUS (the panel's, kept whole -- "for ALL three-factor rules" is
the task's phrase, and the honest census is narrower): expressible today --
r_stdp (PROVEN), learned_evc value/cost, drive_reduction, brain_native_selfbuild
(with the population-mean bound as a factor). NEEDS PRE-COLLAPSE (division by a
population normalizer, sign-conditional dispatch, piecewise nonlinearity) --
valence_stdp, fear_conditioning/affective_appraisal, naca, AND (moved here at
T051b-i, which FALSIFIED this census's own NOT-EXPRESSIBLE row by expressing it)
spiking_jepa's LPL modulator: the cov reduction, the state division, and the
anti_collapse-conditional coefficients all discharge through five caller-computed
pre-collapsed factors (SpikingJEPA.lpl_synapse_rule, held to the production path
by this module's equivalence_check); the pre-collapse COSTS five state quantities
the caller owns, and the layer-conditional lambda_jepa top term plus
transport-gated (gate!=1) sites stay outside the declaration (the espp branch
joined the declared side at T052-i: plasticity/espp_rule.espp_synapse_rule
authors it through this same contract, three pre-collapsed deviation factors,
gate==1 bottom sites only -- the same scoping as the lpl declaration).
NOT EXPRESSIBLE -- the eprop tensor
family (evaluate is SCALAR-only: float(tensor) raises). Rules become
"expressible" by binding arbitrary structure as one factor -- the escape hatch
is disclosed, not denied; the LPL pre-collapse is that hatch used honestly, each
factor a NAMED state quantity with a declared role, never "the rest of the
computation".

NO TISSUE, NO WIRE: dated consumer = T051's composer, whose accumulated scope this
task's row enumerates. Charged surface until then: this gate-run self-check.

  PYTHONPATH=src python -m snn_data_lake.plasticity.eligibility_algebra
"""

import keyword
import math
import threading
import zlib

# distinct-from: plasticity/r_stdp.py -- the INCUMBENT rule this contract
#   EXPRESSES and checks against; the algebra evaluates a declaration, it does not
#   re-implement the neuron. Site read.
# distinct-from: cerebellum/spiking_jepa.py -- the LPL modulator (T049's adjudicated
#   spine); FOUR of its five terms are authored at T051b-i (lpl_synapse_rule, held to
#   the production path by equivalence; the fifth is the layer-conditional lambda_jepa
#   addition, not a modulator term) -- the census row moved NOT-EXPRESSIBLE ->
#   NEEDS-PRE-COLLAPSE (docstring). Site read.
# distinct-from: plasticity/eprop_kernel.py -- an eligibility-propagation KERNEL
#   (executes traces on tissue); this is the AUTHORING contract layer, and its
#   scalar evaluate cannot express the tensor family (disclosed). Site read.
# distinct-from: engine/wiring_manifest.py -- declares PORTS/EDGES between
#   faculties; this declares TERMS inside one rule. A factor here carries NO
#   manifest-resolvable producer reference yet -- FR-019's "declared producers"
#   lands at T051 by composing WITH the manifest, not by a third silo growing its
#   own port model. Site read.
# distinct-from: gates/gradient_claim_gate.py -- the tree's OTHER declaration
#   system (per-SYMBOL gradient claims, prose): different grain (per-TERM structure
#   inside one rule vs one claim per symbol); this module carries its own
#   GRADIENT_CLAIM below like every rule module, and the composer will hold BOTH
#   declarations about one rule to each other. Site read.
# distinct-from: reasoning/triplet_stdp.py -- a RULE (Pfister-Gerstner traces), not
#   a contract; expressible-or-not is the census's row, not a duplication. Site read.
# distinct-from: engine/signal_ring.py -- the silent-sum JOIN defect at the port
#   grain; the matcher fired on shared declared/refuse words. Site read.

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
#: VOLUNTARY here (the gate's verb heuristics do not reach `evaluate`); keyed to the
#: class that actually owns the method -- the first draft keyed a class with no such
#: method and the gate censused the phantom green (filed as its own gate chip).
GRADIENT_CLAIM = {
    "AuthoredRule.evaluate": (
        "NOT_A_RULE",
        "pure arithmetic over caller-supplied factor bindings (sum of declared "
        "products); holds no state, learns nothing -- the CONTRACT layer, not a "
        "learning rule; the rules it expresses carry their own claims",
    ),
}


class EligibilityAlgebra:
    """One authoring target: a declared factor vocabulary + a term budget.

    Factors are declared ONCE with a stated role; rules are authored against the
    vocabulary and validated at authoring time. Rules also RE-validate their factors
    at evaluation/render time against the vocabulary snapshot they were authored
    under, so a hand-built AuthoredRule bypassing author() still cannot evaluate
    undeclared ghosts (the panel constructed exactly that bypass)."""

    def __init__(self, term_budget):
        if not (
            isinstance(term_budget, int) and not isinstance(term_budget, bool) and term_budget >= 1
        ):
            raise ValueError(
                f"term_budget must be an int >= 1 (the executing target's own limit), got {term_budget!r}"
            )
        self._budget = term_budget
        self._factors = {}  # name -> role string
        self._lock = threading.Lock()  # free-threaded runtime: check-then-set must be atomic

    def declare_factor(self, name, role):
        """Add one trace factor. Refuses shadowing (two meanings, one symbol),
        non-identifiers, Python keywords, and 'self' (unbindable through **kwargs --
        the panel authored a permanently dead rule that way)."""
        if not (isinstance(name, str) and name.isidentifier()):
            raise ValueError(f"factor name must be an identifier, got {name!r}")
        if keyword.iskeyword(name) or name == "self":
            raise ValueError(f"factor name {name!r} cannot be bound through **kwargs")
        if not (isinstance(role, str) and role.strip()):
            raise ValueError(f"factor {name!r} needs a non-empty role statement")
        with self._lock:
            if name in self._factors:
                raise ValueError(f"factor {name!r} already declared ({self._factors[name]!r})")
            self._factors[name] = role.strip()

    def author(self, rule_name, terms):
        """Author one rule: terms = [(coefficient, (factor, ...)), ...], an ORDERED
        sequence of PAIRS -- shapes are validated explicitly, because iteration-order
        traps produce silently WRONG rules rather than errors: a set of terms made dw
        differ across processes by hash order, and a string as `factors` misparsed
        into a per-character product ('abc' -> a*b*c) whenever the letters were
        declared factors -- both constructed by the panel."""
        if not (isinstance(rule_name, str) and rule_name.strip()):
            raise ValueError("rule_name must be a non-empty string")
        if not isinstance(terms, (list, tuple)):
            raise ValueError(f"terms must be an ordered list/tuple, got {type(terms).__name__}")
        if not terms:
            raise ValueError(f"rule {rule_name!r}: a rule with no terms updates nothing")
        if len(terms) > self._budget:
            raise ValueError(
                f"rule {rule_name!r}: {len(terms)} terms exceeds the target's budget {self._budget}"
            )
        canon = []
        for k, term in enumerate(terms):
            if not (isinstance(term, (list, tuple)) and len(term) == 2):
                raise ValueError(
                    f"rule {rule_name!r} term {k}: must be a (coefficient, factors) pair, got {term!r}"
                )
            coeff, factors = term
            if isinstance(coeff, bool) or not isinstance(coeff, (int, float)):
                raise ValueError(
                    f"rule {rule_name!r} term {k}: coefficient must be a real number, got {coeff!r}"
                )
            c = float(coeff)
            if not math.isfinite(c):
                raise ValueError(f"rule {rule_name!r} term {k}: non-finite coefficient {coeff!r}")
            if isinstance(factors, (str, bytes)) or not isinstance(factors, (list, tuple)):
                raise ValueError(
                    f"rule {rule_name!r} term {k}: factors must be a list/tuple of names, got {factors!r}"
                )
            if not factors:
                raise ValueError(f"rule {rule_name!r} term {k}: empty product")
            for f in factors:
                if not isinstance(f, str) or f not in self._factors:
                    raise ValueError(
                        f"rule {rule_name!r} term {k}: undeclared factor {f!r} "
                        f"(vocabulary: {sorted(self._factors)})"
                    )
            # AUTHORED order is EXECUTION order: float multiplication is not
            # associative, and incumbent bit-equivalence follows the incumbent's own
            # order (a draft that sorted here diverged from r_stdp by 1 ulp).
            canon.append((c, tuple(factors)))
        return AuthoredRule(rule_name, tuple(canon), dict(self._factors))

    def factors(self):
        """The declared vocabulary {name: role} -- the contract's own documentation."""
        with self._lock:
            return dict(self._factors)


# distinct-from: growth/basic_components.py -- its `Rule` is the neuron-protocol
#   plasticity base class; this class is named AuthoredRule to keep the two one
#   import apart without a name collision (the panel's find). Site read.
class AuthoredRule:
    """An authored sum-of-products rule. FROZEN after construction (attribute
    reassignment raises -- the docstring said 'immutable' while every attribute was
    rebindable; the panel rebound terms to a ghost product). Direct construction
    bypasses author()'s validation, so evaluate/canonical RE-validate factors
    against the vocabulary snapshot -- a bypass construction with undeclared
    factors still refuses at use."""

    __slots__ = ("name", "terms", "_vocab")

    def __init__(self, name, terms, vocabulary):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "_vocab", dict(vocabulary))

    def __setattr__(self, key, value):
        raise AttributeError(f"AuthoredRule is frozen ({key!r} is not assignable)")

    # T053-ii: authored rules now live inside production composers (InhibitorySTDP), so the
    # controller family must stay copy/pickle-able; the default reconstruction path goes
    # through setattr, which the freeze correctly refuses. These restore slots on a FRESH
    # object during copy/unpickle only -- post-construction mutation stays frozen.
    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    def _needed(self):
        used = {f for _c, fs in self.terms for f in fs}
        ghosts = sorted(f for f in used if f not in self._vocab)
        if ghosts:
            raise ValueError(
                f"rule {self.name!r}: factors {ghosts} not in the authoring vocabulary"
            )
        return sorted(used)

    def evaluate(self, **bindings):
        """dw for one site. Pure; refuses (ValueError) unbound factors, bindings
        outside the vocabulary, non-finite/non-scalar values, AND a non-finite
        RESULT (output overflow -- finite inputs can still overflow the product;
        measured at coeff 1e200 x binding 1e200 by T051a's composer harness). Each
        binding is converted to float ONCE and that value is used -- the first
        draft converted twice, so an object whose __float__ drifted passed
        validation as 1.0 and multiplied as inf (the panel's Drifting probe)."""
        needed = self._needed()
        extra = sorted(k for k in bindings if k not in self._vocab)
        if extra:
            raise ValueError(f"rule {self.name!r}: bindings {extra} not in the vocabulary")
        vals = {}
        for f in needed:
            if f not in bindings:
                raise ValueError(f"rule {self.name!r}: unbound factor {f!r}")
            try:
                v = float(bindings[f])
            except (TypeError, ValueError) as ex:
                raise ValueError(
                    f"rule {self.name!r}: binding {f}={bindings[f]!r} is not a real scalar ({ex})"
                ) from None
            if not math.isfinite(v):
                raise ValueError(f"rule {self.name!r}: non-finite binding {f}={bindings[f]!r}")
            vals[f] = v
        # seed from the FIRST term, never 0.0: `0.0 + (-0.0)` re-signs the incumbent's
        # own first-reward output (every RSTDPNeuron is born with e == [0.0]*n, so a
        # negative-d first reward returns -0.0 on every synapse -- the panel's B1).
        dw = None
        for c, fs in self.terms:
            p = c
            for f in fs:
                p *= vals[f]
            dw = p if dw is None else dw + p
        if not math.isfinite(dw):
            raise ValueError(
                f"rule {self.name!r}: non-finite RESULT {dw!r} from finite bindings "
                "(overflow or cancellation) -- the refusal contract covers outputs "
                "too; the LANDED T050 module (battery #92 green) guarded inputs only "
                "and returned inf from coeff 1e200 x binding 1e200 -- exposed by "
                "T051a's composer harness, one commit later"
            )
        return dw

    def canonical(self):
        """The rule as one deterministic line: LIKE TERMS FOLDED (coefficients summed
        over identical sorted factor multisets, exact zeros dropped, -0.0 normalized)
        plus a vocabulary digest -- without the fold, `3*x` and `1*x + 2*x` rendered
        as different rules; without the digest, two vocabularies whose factor names
        collide but whose ROLES differ rendered identically (both panel finds). This
        is like-term-folded multiset identity WITHIN a named vocabulary -- not full
        algebraic equivalence, and a rule differing only in authored ORDER shares
        this text while its evaluation may differ in final ulps (disclosed)."""
        self._needed()  # ghost validation for bypass constructions
        folded = {}
        for c, fs in self.terms:
            key = tuple(sorted(fs))
            folded[key] = folded.get(key, 0.0) + c
        parts = []
        for key in sorted(folded):
            c = folded[key] + 0.0  # normalize -0.0
            if c == 0.0:
                continue
            parts.append(f"{c:+.17g}*{'*'.join(key)}")
        body = " ".join(parts) if parts else "+0"
        digest = zlib.crc32(repr(sorted(self._vocab.items())).encode()) & 0xFFFFFFFF
        return f"{self.name} [vocab:{digest:08x}] : dw = {body}"


def equivalence_check(rule, incumbent_fn, cases, tol=0.0):
    """THE CONTRACT'S TEETH: evaluate `rule` against the incumbent implementation on
    `cases` (each a bindings dict; incumbent_fn(case) returns the incumbent's dw).

    HONEST SCOPE, stated where callers read it: this is EXAMPLE-BASED -- `cases` IS
    the caller's coverage claim, and a rule wrong only outside the cases passes (the
    panel constructed an incumbent abs() invisible to 300 all-positive cases; live
    r_stdp paths measured 0 negative eligibilities over 925k updates, so all-positive
    is the NATURAL under-coverage there). tol=0.0 by default -- a declaration IS its
    implementation; a nonzero tol must be the caller's own measured numeric argument.
    Refuses an empty/exhausted case set (a vacuous pass returned the best possible
    score -- empty-result == dead-detector) and a non-finite tol; a NaN from either
    side REDS (the comparison is `not (d <= tol)`, so NaN cannot slip through --
    `max(0.0, nan)` had made a broken incumbent indistinguishable from a perfect
    one). Returns the max absolute divergence; AssertionError past tol."""
    t = float(tol)
    if not (math.isfinite(t) and t >= 0.0):
        raise ValueError(f"tol must be finite and >= 0, got {tol!r}")
    cases = list(cases)
    if not cases:
        raise ValueError("empty case set: zero cases prove nothing (the coverage claim is yours)")
    worst = 0.0
    for i, case in enumerate(cases):
        a = rule.evaluate(**case)
        b = float(incumbent_fn(case))
        d = abs(a - b)
        if not (d <= t):
            raise AssertionError(
                f"rule {rule.name!r} diverges from its implementation at case {i}: "
                f"declared {a!r} vs incumbent {b!r} (|d|={d!r} > tol {t:.1e})"
            )
        worst = max(worst, d)
    return worst


# ---- self-check ----------------------------------------------------------------------------
def main():
    import random as _random
    import struct

    print("=" * 96)
    print("eligibility_algebra -- specs/022 T050: the sum-of-products authoring contract")
    print("=" * 96)

    def bits(x):
        return struct.pack(">d", x)

    # 1) INCUMBENT BIT-EQUIVALENCE, through the exported check, against the REAL
    #    production reward(), on the MEASURED live ranges (six live lr values; d and
    #    e beyond them; 26% exact-zero e live -> zero and negative e cases included;
    #    the fresh-neuron birth state e=[0]*n with d<0 is the -0.0 case the first
    #    draft's 0.0 accumulator re-signed). The equivalence holds STRUCTURALLY
    #    (coefficient exactly 1.0, authored multiplication order == the incumbent's);
    #    the sweep is a REGRESSION guard, not statistical proof.
    from snn_data_lake.plasticity.r_stdp import RSTDPNeuron

    alg = EligibilityAlgebra(term_budget=4)
    alg.declare_factor("lr", "the neuron's own learning rate (a constant factor)")
    alg.declare_factor("d", "the third factor: reward / dopamine burst")
    alg.declare_factor("e", "the synapse's stored STDP eligibility trace")
    r = alg.author("r_stdp.reward", [(1.0, ("lr", "d", "e"))])
    rng = _random.Random(0)
    live_lr = (0.02, 0.06, 0.08, 0.1, 0.15, 0.2)
    cases = []
    for lr in live_lr:
        for _ in range(40):
            e = rng.choice((0.0, rng.uniform(-17.9, 0.95), rng.uniform(-1e-6, 1e-6)))
            cases.append({"lr": lr, "d": rng.uniform(-1.0, 1.95), "e": e})

    def incumbent(case):
        n = RSTDPNeuron(1, seed=1)
        n.lr = case["lr"]
        n.e = [case["e"]]
        return n.reward(case["d"])[0]

    worst = equivalence_check(r, incumbent, cases, tol=0.0)
    # the BIT half incl. sign of zero: the birth-state case, struct-compared
    fresh = RSTDPNeuron(4, seed=3)  # e == [0.0]*4
    got = fresh.reward(-0.4)
    for i in range(4):
        a = r.evaluate(lr=fresh.lr, d=-0.4, e=0.0)
        assert bits(a) == bits(got[i]), f"sign-of-zero divergence: {a!r} vs {got[i]!r}"
    print(
        f"  green  incumbent equivalence via equivalence_check: {len(cases)} live-range cases, worst |d| = {worst} "
        "(structural: coeff 1.0 + authored order); birth-state -0.0 BIT-equal (struct-compared)"
    )

    # 2) THE MIS-DECLARATION RED through the same exported check + real incumbent.
    bad = alg.author("r_stdp.reward.MUTANT", [(1.0 + 1e-6, ("lr", "d", "e"))])
    try:
        equivalence_check(bad, incumbent, cases[:20], tol=0.0)
        raise AssertionError("a mis-declared coefficient passed the equivalence check")
    except AssertionError as ex:
        if "diverges" not in str(ex):
            raise
    print(
        "  RED    mis-declaration: lr off by 1e-6 reds equivalence_check vs the REAL reward() (arm 1 is the in-command control)"
    )

    # 3) Canonical-identity PROPERTY FIXTURE, synthetic constants throughout. Its
    #    former coefficient-pin half (lambda_var read from a live SpikingJEPA) moved
    #    to jepa's own self-check at T051b-i -- the T051b-i panel measured the pin
    #    DUPLICATED VERBATIM across the two modules (P2: one mechanism per claim),
    #    and the declaration of record is SpikingJEPA.lpl_synapse_rule, held to the
    #    production train_step path by this module's exported equivalence_check.
    #    What stays here is what needs no incumbent: order-independence, like-term
    #    folding, and the role digest.
    alg2 = EligibilityAlgebra(term_budget=8)
    alg2.declare_factor("pred_dev", "synthetic slowness deviation (property fixture)")
    alg2.declare_factor("var_dev", "synthetic variance deviation (property fixture)")
    alg2.declare_factor("sg", "post sensitivity / surrogate gradient factor")
    alg2.declare_factor("e", "presynaptic eligibility trace")
    terms = [
        (-2.0, ("pred_dev", "sg", "e")),
        (7.25, ("var_dev", "sg", "e")),
    ]
    lpl = alg2.author("lpl.modulator.core2", terms)
    lpl_b = alg2.author("lpl.modulator.core2", [terms[1], terms[0]])
    assert lpl.canonical() == lpl_b.canonical(), "canonical form depends on authoring order"
    # like-term folding: 3*x == 1*x + 2*x in canonical text (the T051 diff substrate)
    alg3 = EligibilityAlgebra(term_budget=4)
    alg3.declare_factor("x", "probe factor")
    one = alg3.author("f", [(3.0, ("x",))])
    two = alg3.author("f", [(1.0, ("x",)), (2.0, ("x",))])
    assert one.canonical() == two.canonical(), "like terms not folded"
    # role digest: same names, different roles -> different canonical text
    alg4 = EligibilityAlgebra(term_budget=4)
    alg4.declare_factor("x", "a DIFFERENT role entirely")
    assert alg4.author("f", [(3.0, ("x",))]).canonical() != one.canonical(), "role collision"
    print(
        "  green  order-independent + like-term-folded + role-digested canonical identity "
        "(synthetic fixture; the incumbent-coefficient pin lives in jepa's own self-check, T051b-i)"
    )

    # 4) REFUSALS, each asserted BY MESSAGE (the panel: a bare `except ValueError:
    #    pass` loop stays green when an arm reds for the WRONG reason -- two such
    #    misfires were constructed). Includes the panel's silent-misparse traps.
    for build, must_contain in (
        (lambda: alg.author("x", [(1.0, ("ghost",))]), "undeclared factor"),
        (
            lambda: EligibilityAlgebra(term_budget=1).author("x", [(1.0, ("a",)), (1.0, ("a",))]),
            "exceeds the target's budget",
        ),
        (lambda: alg.author("x", [(1.0, ())]), "empty product"),
        (lambda: alg.author("x", [(float("nan"), ("d",))]), "non-finite coefficient"),
        (lambda: alg.author("x", [(1.0, "de")]), "factors must be a list/tuple"),
        (lambda: alg.author("x", [(1.0, {"d": 1})]), "factors must be a list/tuple"),
        (lambda: alg.author("x", {(1.0, ("d",))}), "ordered list/tuple"),
        (lambda: alg.author("x", [("1.0", ("d",))]), "coefficient must be a real number"),
        (lambda: alg.author("x", [(1.0, ("d",), "extra")]), "(coefficient, factors) pair"),
        (lambda: alg.declare_factor("d", "shadow"), "already declared"),
        (lambda: alg.declare_factor("self", "unbindable"), "cannot be bound"),
        (lambda: alg.declare_factor("lambda", "keyword"), "cannot be bound"),
        (lambda: EligibilityAlgebra(term_budget=0), "int >= 1"),
        (lambda: EligibilityAlgebra(term_budget=True), "int >= 1"),
        (lambda: alg.author("", [(1.0, ("d",))]), "non-empty string"),
        (lambda: equivalence_check(r, lambda c: 0.0, []), "empty case set"),
        (
            lambda: equivalence_check(
                r, lambda c: 0.0, [{"lr": 1, "d": 1, "e": 1}], tol=float("nan")
            ),
            "tol must be finite",
        ),
    ):
        try:
            build()
            raise AssertionError(f"accepted an invalid construction (wanted {must_contain!r})")
        except ValueError as ex:
            assert must_contain in str(ex), f"refused for the WRONG reason: {ex}"
    alg_ovf = EligibilityAlgebra(term_budget=2)
    alg_ovf.declare_factor("x", "overflow probe factor")
    r_ovf = alg_ovf.author("overflow", [(1e200, ("x",))])
    for call, must_contain in (
        (lambda: r.evaluate(lr=0.1, d=float("inf"), e=1.0), "non-finite binding"),
        (lambda: r.evaluate(lr=0.1, d=0.5), "unbound factor"),
        (lambda: r.evaluate(lr=0.1, d=0.5, e=1.0, GHOST=1.0), "not in the vocabulary"),
        (lambda: r.evaluate(lr=0.1, d=1 + 0j, e=1.0), "not a real scalar"),
        (lambda: r_ovf.evaluate(x=1e200), "non-finite RESULT"),
    ):
        try:
            call()
            raise AssertionError(f"accepted an invalid evaluation (wanted {must_contain!r})")
        except ValueError as ex:
            assert must_contain in str(ex), f"refused for the WRONG reason: {ex}"
    # frozen: post-construction reassignment refuses; NaN cross-side reds the check;
    # a bypass construction with a ghost factor refuses at use
    try:
        r.terms = ((1e9, ("ghost",)),)
        raise AssertionError("AuthoredRule accepted attribute reassignment")
    except AttributeError:
        pass
    try:
        equivalence_check(r, lambda c: float("nan"), cases[:1])
        raise AssertionError("a NaN incumbent passed as a confident green")
    except AssertionError as ex:
        assert "diverges" in str(ex)
    ghost_rule = AuthoredRule("bypass", ((1.0, ("ghost",)),), {})
    for use in (lambda: ghost_rule.evaluate(ghost=1.0), ghost_rule.canonical):
        try:
            use()
            raise AssertionError("a bypass construction evaluated an undeclared ghost")
        except ValueError as ex:
            assert "authoring vocabulary" in str(ex)
    print(
        "  RED    refusals BY MESSAGE: 17 authoring + 5 evaluation + frozen-rule + NaN-incumbent + bypass-ghost reds"
    )

    print()
    print(
        "DISCLOSURES: example-based teeth -- `cases` IS the caller's coverage claim "
        "(an incumbent abs() invisible to all-positive cases is the constructed "
        "illustration; live r_stdp paths measured 0 negative eligibilities over 925k "
        "updates, so natural sampling under-covers there). SCALAR-only: the eprop "
        "tensor family is NOT expressible (float(tensor) raises; the e-prop rules "
        "keep their own kernels). The expressibility census over the tree's "
        "three-factor rules is in the module docstring -- 'for ALL three-factor "
        "rules' is the task's phrase, the census is what is true. term_budget=4/8 in "
        "THIS harness are TEST FIXTURES: the P1 exception that covered them EXPIRED "
        "at T053-ii (2026-08-31) when InhibitorySTDP's production step() began applying "
        "its authored declaration through the composer kernel -- the tick-charged "
        "composed rule this exception was dated to (one expiry, one date, both "
        "siblings; the composer's DISCLOSURES carry the same landing). Fixtures "
        "parameterize these arms and arbitrate no production behaviour. Factor "
        "producers are NOT declared here (name + role prose only) -- FR-019's "
        "'declared producers' lands at T051b by composing with wiring_manifest, not "
        "a third port model. The GRADIENT_CLAIM above is VOLUNTARY (the gate's verb "
        "list does not reach `evaluate`) and keyed to the class that owns the "
        "method -- the first draft's phantom key censused green, filed as a gate "
        "chip."
    )
    print(
        "self-check OK: live-range incumbent equivalence through the exported check (+ birth-state "
        "-0.0 bit case) + mis-declaration red vs the real reward() + synthetic canonical fixture "
        "(the LPL coefficient pin moved to jepa's self-check, T051b-i) + 25 refusals asserted by "
        "message (incl. output overflow)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
