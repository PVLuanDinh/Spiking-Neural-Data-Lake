"""
Rule composer KERNEL -- specs/022 T051a (P8 invention #3's core): N rules on one
substrate value with TYPED READ/WRITE SETS and per-rule neuter attribution --
the INSTRUMENT'S KERNEL. Its first reader landed at T051b-iii: rule_charge, the
per-rule charge A/B consuming last_attribution(), GATE-RUN grade (zero live
compositions exist in the tree; the first TICK-live reading arrives with T053's
charged composition, said plainly). Authored through T050's eligibility algebra.

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
  one rule to each other"; the cross-check LANDED at T051c (check_claim, holding
  the key to the gate's own declared surface -- a phantom key REDS), alongside
  the per-rule disposition census (tools/rule_census.json + rule_census_check,
  the ratcheted WAVE structure).

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
survive today at different grains, and the MERGE -- expressing BOTH stacks'
sandwiches as RuleSpecs held to equivalence the way T050 held r_stdp -- LANDED
at T051b-iv (harness section 1d: gain as the sequential two-writer composition,
base weights as deposit-then-trim, each vs spied production calls; the subclass
weights phases and rate/theta stay the disclosed remainder, and both stacks
REMAIN the production path -- the merge is the declaration-of-record layer).
wholistic_loop's "arbitration" is FACULTY-grain tick arbitration, distinct.

NO TISSUE, NO WIRE: the reader landed at T051b-iii (rule_charge, gate-run
grade) and the stack-sandwich MERGE landed at T051b-iv (section 1d) -- P2
merge-must-reduce NOT discharged by it, said plainly: both incumbents remain
production and the expression is a declaration layer. The first TICK-charged
composed rule is T053's (its row names the composition). Charged surface until
then: this gate-run self-check.

  PYTHONPATH=src python -m snn_data_lake.plasticity.rule_composer
"""

import math
import threading
import types

from snn_data_lake.plasticity.eligibility_algebra import AuthoredRule

# distinct-from: plasticity/plasticity_rule_stack.py -- live incumbent composition
#   #1 (the extended sandwich); adjudicated in the module docstring, merge LANDED
#   at T051b-iv as the declaration layer. Site read.
# distinct-from: plasticity/homeostatic_plasticity_stack.py -- live incumbent
#   composition #2 (the BASE sandwich the first landing's census missed -- the
#   T046a all-incumbents lesson); same adjudication, same landed merge. Site read.
# distinct-from: cortex/wholistic_loop.py -- FACULTY-grain tick arbitration (a
#   metacognitive spine over tasks); this composes RULES on one substrate. Site read.
# distinct-from: plasticity/eligibility_algebra.py -- the AUTHORING contract this
#   kernel consumes; rules arrive as AuthoredRule, never re-parsed. Site read.
# distinct-from: substrate/wiring_manifest.py -- ports/edges between faculties;
#   RuleSpec's reads/writes are per-RULE factor/target sets. Producer references
#   composed with the manifest landed at T051b-iii (check_producers). Site read.
# distinct-from: substrate/gradient_claim_gate.py -- the per-SYMBOL claim system;
#   RuleSpec.claim_key is the SLOT binding a rule to its claim, and the
#   cross-check holding both declarations to each other LANDED at T051c
#   (check_claim + the rule_census ratchet). Site read.
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


# distinct-from: substrate/wiring_manifest.py -- Port declares a FACULTY signal surface;
#   RuleSpec declares one RULE's factor/target sets, and its `producers` entries only
#   REFERENCE manifest ports (resolved by check_producers), never re-model them. Site read.
class RuleSpec:
    """One rule's typed declaration. Frozen. `claim_key`: the optional
    GRADIENT_CLAIM key of the rule's owning symbol (held by check_claim,
    landed at T051c; a phantom key reds against the declared surface). `producers`
    (T051b-iii, FR-019): the optional factor->producer declaration --
    "self" for a caller-computed internal state quantity, or a
    (faculty, port_name, expected_unit) triple naming the manifest OUT port
    the factor is fed from; resolution against the manifest is
    check_producers' job, never construction's (a spec must be
    constructible without importing the substrate tier)."""

    __slots__ = ("name", "rule", "reads", "writes", "claim_key", "producers")

    def __init__(self, name, rule, reads, writes, claim_key=None, producers=None):
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
        if producers is not None:
            if not isinstance(producers, dict):
                raise ValueError(
                    f"spec {name!r}: producers must be a dict, got {type(producers).__name__}"
                )
            orphan = sorted(set(producers) - reads)
            if orphan:
                raise ValueError(
                    f"spec {name!r}: producers declared for factors {orphan} the read set "
                    "does not carry -- a producer for an unread factor is a dead declaration"
                )
            for f, prod in producers.items():
                ok = prod == "self" or (
                    isinstance(prod, tuple)
                    and len(prod) == 3
                    and all(isinstance(x, str) and x.strip() for x in prod)
                )
                if not ok:
                    raise ValueError(
                        f"spec {name!r}: producer for {f!r} must be 'self' or a "
                        f"(faculty, port, expected_unit) triple, got {prod!r}"
                    )
            # a read-only view: reads/writes are frozensets, and a bare dict here was
            # panel-measured mutable IN PLACE post-construction -- re-opening the
            # unread-key invariant this constructor just refused
            producers = types.MappingProxyType(dict(producers))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "claim_key", claim_key)
        object.__setattr__(self, "producers", producers)

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
        """The attribution kernel's read surface (first reader: rule_charge,
        T051b-iii -- gate-run grade; tick-live reading is T053's): the latest step's
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
            # deliberately RuleComposer, not type(self): the LOO arm measures the
            # KERNEL's composition semantics; a subclass wanting subclass-semantics
            # LOO overrides neutered() itself (panel-noted, stated here)
            out = RuleComposer(self._policy, self._target, _loo_of=name)
            for spec in self._specs:
                if spec.name != name:
                    out.add(spec)
            return out


# distinct-from: substrate/signal_ring.py -- JoinConsumer ENFORCES join/delivery policies on
#   live traffic between faculties; check_producers only RESOLVES a rule's declared factor
#   references against the manifest (a static census check, no traffic touched). Site read.
# distinct-from: substrate/manifest_gate.py -- that gate checks the MANIFEST's own coherence;
#   this checks one RuleSpec AGAINST the manifest it produces. Consumes, never re-models. Site read.
def check_producers(spec, manifest, consumer=None):
    """FR-019 / T051b-iii: resolve one RuleSpec's declared factor producers against
    the wiring manifest. Returns a FINDINGS list (empty == green -- the manifest
    checks' idiom, composable with gates); refuses (ValueError) a spec with no
    producers declaration at all, because 'nothing to check' silently reading as
    green is the vacuous-pass shape the tree keeps deleting.

    Semantics per entry: "self" = a caller-computed internal state quantity
    (declared, deliberately NOT resolved -- inv_var or cov_pull has no faculty
    port and pretending otherwise would be a third port model, which FR-019's
    own text forbids); a (faculty, port, expected_unit) triple must resolve to a
    DECLARED manifest port that is kind OUT -- only an OUT port produces (an IN
    port consumes; REF, the audited escape hatch, is deliberately NOT a legal
    producer) -- whose declared unit equals the spec's expectation (the unit-lie
    red: a consumer expecting spikes_hz wired to a level01 port would scale
    silently). WITHOUT `consumer`, resolution is NAME/KIND/UNIT ONLY and
    ROUTE-BLIND, said plainly: the panel measured 9 of 10 unrouted level01 OUT
    ports passing green beside the one routed edge. WITH `consumer` (a faculty
    name), each resolved triple must ALSO source a declared edge into that
    faculty -- the no-declared-route red, presence-is-not-participation at the
    port grain. The manifest is the PRODUCTION assembly
    (manifest_gate.rebuild_from_registrants), never a hand-built one."""
    if spec.producers is None:
        raise ValueError(
            f"spec {spec.name!r} declares no producers -- check_producers has nothing "
            "to check, and a vacuous green is refused rather than returned"
        )
    findings = []
    for f in sorted(spec.reads):
        if f not in spec.producers:
            findings.append(f"factor {f!r}: read but no producer declared")
    for f in sorted(spec.producers):
        prod = spec.producers[f]
        if prod == "self":
            continue
        fac, port_name, expected_unit = prod
        port = manifest.ports.get((fac, port_name))
        if port is None:
            findings.append(
                f"factor {f!r}: producer ({fac!r}, {port_name!r}) is not a declared port"
            )
            continue
        if port.kind != "OUT":
            findings.append(
                f"factor {f!r}: producer ({fac!r}, {port_name!r}) is kind {port.kind!r} -- "
                "only an OUT port produces"
            )
        if port.unit != expected_unit:
            findings.append(
                f"factor {f!r}: unit lie -- the spec expects {expected_unit!r}, the port "
                f"declares {port.unit!r}"
            )
        if consumer is not None and not any(
            e[0] == fac and e[1] == port_name and e[2] == consumer for e in manifest.edges
        ):
            findings.append(
                f"factor {f!r}: declared producer ({fac!r}, {port_name!r}) has NO declared "
                f"route into {consumer!r} -- a resolvable name is not a wire"
            )
    return findings


# distinct-from: tools/charge_probe.py -- the SAME neuter-A/B pattern at a DIFFERENT grain:
#   the probe neuters drive sources on a live brain; this neuters RULES on a caller-built
#   composed trajectory. The probe stays the brain-grain surface (cross-referenced there);
#   one pattern, two grains, neither re-implements the other. Site read.
# distinct-from: thalamus/ignition_instrument.py -- that sweep drives a THALAMIC actuator
#   ensemble; this drives a caller-built RULE composition's A/B arms. Site read.
def rule_charge(composer, w0, bindings_seq):
    """The attribution instrument's READER (T051b-iii) -- per-rule charge A/B at
    the composed-trajectory grain. Drives the FULL composition over bindings_seq
    from w0, then each rule's leave-one-out arm (neutered()) over the SAME
    bindings. Per-rule verdicts, THREE-WAY (the panel's convergent finding: a
    two-way endpoint test read INERT on a rule that wrote every step):
      CHARGED          -- the endpoint moved.
      ENDPOINT-NEUTRAL -- endpoint delta exactly 0.0 while the rule's own
                          cumulative |dw| is nonzero (sign cancellation, or the
                          delta absorbed below the substrate's resolution).
                          EXPLICITLY NOT a P3 retirement trigger: the rule
                          participated; only its net endpoint effect is zero.
      INERT            -- endpoint delta 0.0 AND cumulative |dw| exactly 0.0:
                          present, composed, never moved anything (P3's shape).
      UNCHARGEABLE     -- sequential policy only: removing this rule leaves no
                          target reader, so its LOO arm is not a sequential
                          composition (the kernel would refuse the label); the
                          reader says so instead of crashing mid-A/B.
    CONSUMES last_attribution(): the verdicts key on cumulative_abs_dw, and
    under "sum" the first step's LOO endpoint predicted from the parts is held
    to the measured arm -- a CROSS-CHECK, not a red (0/2000 random pure-rule
    compositions can fire it; bindings are float-frozen ONCE at entry so a
    stateful __float__ cannot reach it either). HONEST SCOPE, stated where a
    reader looks: this runs on whatever composition the CALLER built; with zero
    live compositions in the tree today it is a gate-run instrument, and the
    first TICK-live reading arrives with T053's charged composition, not here.

    Refuses: an already-stepped composer (its attribution epoch would mix with
    this run's), a non-finite w0, an empty bindings_seq, a seq whose bindings
    carry the target key. A MID-RUN kernel refusal (overflow, non-finite
    binding, LOO divergence) is re-raised with the arm and step named -- and it
    leaves the caller's composer STEPPED, so this reader is one-shot per
    composer, said plainly."""
    if composer.last_attribution()["stepped"]:
        raise ValueError(
            "rule_charge needs a FRESH composer: an already-stepped one would mix "
            "attribution epochs with this run's trajectory"
        )
    try:
        w0f = float(w0)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"rule_charge: w0={w0!r} is not a real scalar ({ex})") from None
    if not math.isfinite(w0f):
        raise ValueError(f"rule_charge: non-finite w0 {w0!r}")
    raw_seq = list(bindings_seq)
    if not raw_seq:
        raise ValueError("empty bindings_seq: zero steps prove nothing")
    target = composer.last_attribution()["target"]
    seq = []
    for i, bnd in enumerate(raw_seq):
        if target in bnd:
            raise ValueError(
                f"bindings_seq[{i}] carries the target {target!r} -- the trajectory owns "
                "the substrate value; a caller-pinned target would fork the A/B arms"
            )
        frozen = {}
        for k, v in dict(bnd).items():
            try:
                fv = float(v)  # frozen ONCE: both arms and the cross-check
            except (TypeError, ValueError) as ex:  # see the SAME value forever
                raise ValueError(
                    f"rule_charge: bindings_seq[{i}][{k}]={v!r} is not a real scalar ({ex})"
                ) from None
            if not math.isfinite(fv):
                raise ValueError(  # refused at ENTRY: a mid-run kernel refusal
                    f"rule_charge: bindings_seq[{i}][{k}]={v!r} is non-finite"
                )  # would burn the caller's composer before any A/B ran
            frozen[k] = fv
        seq.append(frozen)

    def _drive(comp, arm):
        w = w0f
        first_parts = None
        for i, bnd in enumerate(seq):
            b = dict(bnd)
            b[target] = w
            try:
                w, parts = comp.step(w, b)
            except ValueError as ex:
                raise ValueError(
                    f"rule_charge: the {arm} arm refused at step {i + 1}/{len(seq)}: {ex}"
                ) from ex
            if first_parts is None:
                first_parts = parts
        return w, first_parts

    w_full, first_parts = _drive(composer, "full")
    att = composer.last_attribution()  # THE read: the instrument consuming its kernel
    seq_policy = att["policy"] == "sequential"
    per_rule = {}
    for name in sorted(att["cumulative_abs_dw"]):
        cum = att["cumulative_abs_dw"][name]
        if seq_policy:
            remaining = [s for s in composer._specs if s.name != name]
            if remaining and not any(target in s.reads for s in remaining):
                per_rule[name] = {
                    "verdict": "UNCHARGEABLE",
                    "reason": (
                        "removing this rule leaves no target reader -- the LOO arm "
                        "is not a sequential composition"
                    ),
                    "cumulative_abs_dw": cum,
                    "first_step_part": first_parts.get(name),
                    "first_step_cross_check": "skipped (policy != 'sum')",
                }
                continue
        loo = composer.neutered(name)
        w_loo, _ = _drive(loo, f"LOO({name})")
        measured = w_full - w_loo
        if measured != 0.0:
            verdict = "CHARGED"
        elif cum != 0.0:
            verdict = "ENDPOINT-NEUTRAL"
        else:
            verdict = "INERT"
        entry = {
            "measured_endpoint_delta": measured,
            "verdict": verdict,
            "cumulative_abs_dw": cum,
            "first_step_part": first_parts.get(name),
        }
        if not seq_policy:
            # the first step's LOO endpoint, predicted from the parts by the
            # ordered-parts decomposition -- a CROSS-CHECK consuming the
            # attribution (bit-identical by construction on pure rules; kept
            # because it binds the reader to the kernel's proven property).
            pred = None
            for other, dw in first_parts.items():
                if other != name:
                    pred = dw if pred is None else pred + dw
            pred_w1 = w0f + pred if pred is not None else w0f
            loo1 = composer.neutered(name)
            w1_loo, _ = loo1.step(w0f, {**seq[0], target: w0f})
            if w1_loo != pred_w1:
                raise ValueError(
                    f"attribution inconsistency at rule {name!r}: the parts predict a "
                    f"first-step LOO endpoint {pred_w1!r}, the measured arm gives {w1_loo!r}"
                )
            entry["first_step_cross_check"] = True
        else:
            entry["first_step_cross_check"] = "skipped (policy != 'sum')"
        per_rule[name] = entry
    return {
        "policy": att["policy"],
        "target": target,
        "steps": len(seq),
        "endpoint_full": w_full,
        "per_rule": per_rule,
        "attribution": att,
    }


# distinct-from: substrate/gradient_claim_gate.py -- the gate CENSUSES declarations
#   tree-wide and asserts ratchet bounds; check_claim holds ONE RuleSpec's claim_key
#   to that census (a consumer of the gate's own walker, never a second census). Site read.
def check_claim(spec, declared_keys):
    """T051c: hold a RuleSpec's claim_key to the tree's DECLARED GRADIENT_CLAIM keys
    (the T050 panel's phantom-qualname class, closed for composed rules: a claim_key
    naming a declaration that does not exist reads green to the gate -- enumeration
    is seeded by declarations -- so the composer holds the reference itself).
    `declared_keys` = the `relpath::key` strings from the gate's OWN census (one
    census instrument); the harness shows the build. HONEST SCOPE, said where a
    reader looks: EXISTENCE only -- the claim's KIND is not held to the spec, and
    a key declared in an UNGOVERNED module resolves green (the walker's whole
    surface is the reference); gate-run grade, zero live callers today. Incoming
    keys are separator/whitespace-normalized (the panel measured a TRUE claim in
    Windows rel-path form misread as phantom -- a false red asserting the opposite
    of the truth). Refuses a keyless spec AND an empty declared surface (every key
    would phantom -- the dead-detector shape)."""
    if spec.claim_key is None:
        raise ValueError(
            f"spec {spec.name!r} carries no claim_key -- check_claim has nothing to hold, "
            "and a vacuous green is refused rather than returned"
        )
    if not declared_keys:
        raise ValueError(
            "empty declared_keys: every key would read phantom -- a dead detector, refused"
        )
    key = spec.claim_key.strip().replace("\\", "/")
    if key in declared_keys:
        return []
    return [
        f"spec {spec.name!r}: claim_key {spec.claim_key!r} names NO declared GRADIENT_CLAIM "
        "-- the phantom-qualname class (a declaration the gate would census green because "
        "nothing anchors it)"
    ]


def _is_undispositioned(disposition):
    """Case/whitespace-normalized, and an EMPTY disposition counts as undispositioned
    (the panel measured '' / 'TODO' / lowercase silently counting as dispositioned --
    the ratchet's denominator matched one exact token)."""
    if not isinstance(disposition, str) or not disposition.strip():
        return True
    return disposition.strip().upper().startswith("UNDISPOSITIONED")


def rule_census_check(artifact, rules):
    """T051c: the per-rule disposition artifact (tools/rule_census.json) held to the
    LIVE census. `rules` = gradient_claim_gate.census()[0]. Findings when: a live
    rule-bearing row has no artifact row (the artifact lags the tree); an artifact
    row names no live rule (stale); the UNDISPOSITIONED count RISES above the
    artifact's recorded baseline, or total rows FALL below it. THE DIVISION OF
    LABOR, said plainly (the panel measured the first draft's baseline recomputed
    by its own edit path -- a pin rewriting the incumbent): THESE are the cheap
    membership/count bounds; the MECHANIZED per-row guard lives in the sanctioned
    edit path itself (regen_rule_census refuses disposition retractions and dead
    ADJUDICATED entries -- the gradient-claim baseline's refuses-worse pattern);
    the ADVANCE direction (whether a disposition's text is honest) is review-grade.
    UNDISPOSITIONED day-one rows are the honest state, counted and bounded, never
    hidden (the T046 day-one precedent). Shape errors name the artifact, never a
    bare KeyError."""
    try:
        art_rules = artifact["rules"]
        base = artifact["_meta"]["ratchet_baseline"]
        base_undisp = base["undispositioned"]
        base_rows = base["rows"]
    except (KeyError, TypeError) as ex:
        raise ValueError(
            f"rule_census artifact is malformed (missing {ex!r}) -- expected "
            "{'_meta': {'ratchet_baseline': {'rows', 'undispositioned'}}, 'rules': {...}}"
        ) from None
    bearing = {"LOCAL", "APPROXIMATES_GRADIENT", "EXACT_GRADIENT", "DIRECT_SOLVE"}
    live = {
        r["qualname"].replace("\\", "/")
        for r in rules
        if (r["declared"] or r["inferred"]) in bearing
    }
    art = set(art_rules)
    findings = []
    for q in sorted(live - art):
        findings.append(
            f"live rule-bearing claim {q!r} has NO census row -- the artifact lags the tree"
        )
    for q in sorted(art - live):
        findings.append(f"census row {q!r} names no live rule-bearing claim -- stale row")
    n_undisp = 0
    for q, v in art_rules.items():
        if not isinstance(v, dict) or "disposition" not in v:
            raise ValueError(f"rule_census artifact row {q!r} is malformed (no disposition)")
        if _is_undispositioned(v["disposition"]):
            n_undisp += 1
    if n_undisp > base_undisp:
        findings.append(
            f"UNDISPOSITIONED rose ({n_undisp} > baseline {base_undisp}) -- "
            "dispositions only ever advance"
        )
    if len(art) < base_rows:
        findings.append(f"census rows fell ({len(art)} < baseline {base_rows})")
    return findings


#: The adjudicated disposition seedings (T050's expressibility census + the T051b
#: landings) -- checked FIRST, before the kind fallbacks (the panel measured the
#: first draft dispatching kind first, which made the jepa correction DEAD CODE and
#: re-published the NOT-EXPRESSIBLE row T051b-i had falsified); every entry must
#: match >=1 live row or regeneration REFUSES (the first draft's wrong naca path
#: silently cost a disposition and loosened the ratchet).
_CENSUS_ADJUDICATED = (
    (
        "basal_ganglia/r_stdp.py",
        "reward",
        "expressible -- PROVEN: T050 equivalence via the exported check; T051b-iv deposit through the kernel",
    ),
    ("cortex/learned_evc.py", None, "expressible -- T050 census (value/cost forms)"),
    ("hypothalamus/homeostatic_drives.py", None, "expressible -- T050 census (drive_reduction)"),
    (
        "basal_ganglia/valence_stdp.py",
        None,
        "needs-pre-collapse -- T050 census (sign-conditional dispatch)",
    ),
    ("amygdala/fear_conditioning.py", None, "needs-pre-collapse -- T050 census"),
    ("amygdala/affective_appraisal.py", None, "needs-pre-collapse -- T050 census"),
    ("neuromodulation/naca.py", None, "needs-pre-collapse -- T050 census"),
    (
        "cerebellum/spiking_jepa.py",
        None,
        "mixed -- the LOCAL three-factor block is EXPRESSED at T051b-i (lpl_synapse_rule, "
        "five named pre-collapsed factors, held to the production path) and its "
        "rule='espp' branch at T052-i (delegated to plasticity/espp_rule); the "
        "readout/predictor gradient half is not-expressible (tensor family, T050 census)",
    ),
    (
        "plasticity/espp_rule.py",
        "espp_modulator_core",
        "expressed -- T052-i: the extracted incumbent math (jepa's rule='espp' branch "
        "delegates here); held by the frozen-transcription regression arm (tol=0) and "
        "bound to the applied weight through the DELTA arm's declaration",
    ),
    (
        "plasticity/espp_rule.py",
        "espp_synapse_rule",
        "expressed -- T052-i: authors the per-synapse declaration through the T050 "
        "algebra; held to the applied weight by its BIT/FULL/DELTA arms",
    ),
    (
        "plasticity/homeostatic_plasticity_stack.py",
        None,
        "expressed -- T051b-iv: deposit-then-trim through the kernel vs spied production",
    ),
    (
        "plasticity/plasticity_rule_stack.py",
        "learning_step",
        "expressed (gain pipeline) -- T051b-iv: two-writer sequential vs spied pre_gain; "
        "weights subclass phases = disclosed remainder",
    ),
    (
        "plasticity/plasticity_rule_stack.py",
        "consolidate",
        "remainder -- DISCLOSED at T051b-iv (E/I + Benna-Fusi + rate + theta undriven; "
        "ii's component grain models the bare cascade; the tissue wire is T054's)",
    ),
    (
        "plasticity/benna_fusi.py",
        None,
        "expressed (component grain) -- T051b-ii: the anchor-drag form vs diffuse()",
    ),
    (
        "plasticity/structural_rewiring.py",
        None,
        "expressed (component grain) -- T051b-ii: the prune form vs rewire()",
    ),
)

#: T050's expressible list has a FOURTH entry, brain_native_selfbuild, which declares
#: no GRADIENT_CLAIM and is therefore INVISIBLE to the declared surface this census
#: binds -- recorded so "3 expressible" is never read as the whole T050 list.
_CENSUS_INVISIBLE = (
    "basal_ganglia/brain_native_selfbuild.py (T050: expressible; no GRADIENT_CLAIM declared)",
)


def regen_rule_census():
    """THE ONLY EDIT PATH for tools/rule_census.json (T051c; run via
    `python -m snn_data_lake.plasticity.rule_composer --regen-census`). Lives HERE,
    in the governed registered module beside the check that polices the artifact --
    the sibling pattern (gradient_claim_gate owns --update-baseline), and the panel
    measured a standalone tools/*.py breaking the zero-headroom p4c tools zone.
    THE MECHANIZED RATCHET: reads the committed artifact first and REFUSES (raises)
    any qualname moving FROM a disposition BACK to UNDISPOSITIONED -- per-row
    monotonicity through the sanctioned door (the first draft recomputed the
    baseline from the rows it had just written: green through its own edit path).
    Also refuses ADJUDICATED entries matching zero live rows. Path resolution uses
    the gate's own marker-anchored _repo_root (the panel measured the first draft's
    counted parents[] climb resolving OUTSIDE the repo, with a non-terminating
    fallback on marker-less layouts)."""
    import json

    from snn_data_lake.substrate.gradient_claim_gate import _repo_root, census

    rules, failures, _n = census()
    if failures:
        raise ValueError(f"census failures: {failures}")
    bearing = {"LOCAL", "APPROXIMATES_GRADIENT", "EXACT_GRADIENT", "DIRECT_SOLVE"}
    hits = [0] * len(_CENSUS_ADJUDICATED)

    def _disp(qualname, kind):
        for i, (mod_sub, qual_sub, disp) in enumerate(_CENSUS_ADJUDICATED):
            if mod_sub in qualname and (qual_sub is None or qual_sub in qualname):
                hits[i] += 1
                return disp
        if kind == "DIRECT_SOLVE":
            return "direct-solve -- no dw stream to compose (T050 census category)"
        if kind in ("APPROXIMATES_GRADIENT", "EXACT_GRADIENT"):
            return (
                "not-expressible -- the gradient/eprop tensor family "
                "(scalar-only evaluate; T050 census)"
            )
        return "UNDISPOSITIONED"

    rows = {}
    for r in rules:
        kind = r["declared"] or r["inferred"]
        if kind not in bearing:
            continue
        q = r["qualname"].replace("\\", "/")
        rows[q] = {"kind": kind, "governed": r["governed"], "disposition": _disp(q, kind)}
    dead = [_CENSUS_ADJUDICATED[i][:2] for i in range(len(_CENSUS_ADJUDICATED)) if hits[i] == 0]
    if dead:
        raise ValueError(
            f"REFUSED: ADJUDICATED entries matching zero live rows {dead} -- a dead entry "
            "is a disposition silently lost"
        )
    import pathlib

    out = pathlib.Path(_repo_root()) / "tools" / "rule_census.json"
    if out.exists():
        prior = json.loads(out.read_text(encoding="utf-8"))["rules"]
        regressions = sorted(
            q
            for q, v in rows.items()
            if q in prior
            and not _is_undispositioned(prior[q]["disposition"])
            and _is_undispositioned(v["disposition"])
        )
        if regressions:
            raise ValueError(
                f"REFUSED: {len(regressions)} row(s) would move from a disposition BACK to "
                f"UNDISPOSITIONED {regressions[:5]} -- dispositions only ever advance; "
                "re-adjudicate the text in _CENSUS_ADJUDICATED, never retract by regeneration"
            )
    n_undisp = sum(1 for v in rows.values() if _is_undispositioned(v["disposition"]))
    artifact = {
        "_meta": {
            "task": "specs/022 T051c",
            "generated_by": "PYTHONPATH=src python -m snn_data_lake.plasticity.rule_composer --regen-census",
            "source": "gradient_claim_gate.census() via rule_composer.regen_rule_census -- regeneration is the only edit path for rows; new adjudications join _CENSUS_ADJUDICATED there; the regen refuses disposition retractions and dead entries",
            "ratchet_baseline": {"rows": len(rows), "undispositioned": n_undisp},
            "invisible_to_surface": list(_CENSUS_INVISIBLE),
            "note": "dispositions per T050's expressibility categories; UNDISPOSITIONED is the honest day-one state for unadjudicated LOCAL rules -- the WAVE (T051c-wave) fills rows; regen_rule_census enforces per-row monotonicity through the sanctioned edit path, rule_composer 1e polices membership vs the live census, and the ADVANCE direction (whether a disposition's text is honest) is review-grade, said plainly",
        },
        "rules": rows,
    }
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(artifact, f, indent=1, sort_keys=True)
    return out, len(rows), n_undisp


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

    # 1c) T051b-iii: PRODUCERS x THE MANIFEST + THE ATTRIBUTION'S READER.
    #     Producer resolution runs against the PRODUCTION manifest assembly
    #     (rebuild_from_registrants -- never hand-built). THE CENSUS, said the
    #     way it was actually done (the panel measured the first draft's
    #     "measured" claim hand-typed): jepa's declaration is ENUMERATED below
    #     (SpikingJEPA.lpl_synapse_rule imported, its factors iterated -- a new
    #     factor there reds this arm); r_stdp's form is HAND-ADJUDICATED against
    #     eligibility_algebra's arm (which holds it to the incumbent THERE;
    #     canonical dw = +1*d*e*lr); NO tree-wide enumeration gate exists --
    #     the producers enumeration is T051c-wave's (open, undated), cited. rule_charge is the attribution's FIRST
    #     READER -- gate-run grade, said plainly (zero live compositions exist;
    #     T053's is the first tick-live one).
    from snn_data_lake.cerebellum.spiking_jepa import SpikingJEPA
    from snn_data_lake.substrate.manifest_gate import rebuild_from_registrants

    _mani = rebuild_from_registrants()
    # the ENUMERATED half: jepa's production declaration, factors iterated;
    # every one must be in the adjudicated self-set (all caller-computed
    # internal state) -- a cross-faculty factor appearing there reds THIS arm.
    _jepa_rule = SpikingJEPA(("noop", "left"), seed=0).lpl_synapse_rule()
    _jepa_factors = {f for _c9, _fs9 in _jepa_rule.terms for f in _fs9}
    _jepa_self_set = {"d_pred", "d_mu", "inv_var", "cov_pull", "d_homeo", "lr", "sg", "e"}
    assert _jepa_factors <= _jepa_self_set, (
        f"jepa's declaration grew factors outside the adjudicated self-set: "
        f"{sorted(_jepa_factors - _jepa_self_set)} -- re-run the cross-faculty census"
    )
    _alg_p = EligibilityAlgebra(term_budget=1)
    _alg_p.declare_factor("d", "phasic dopamine (RPE) -- the cross-faculty third factor")
    _alg_p.declare_factor("e", "the neuron's own eligibility trace")
    _alg_p.declare_factor("lr", "the neuron's own learning rate")
    _rrule = _alg_p.author("rstdp.form", [(1.0, ("lr", "d", "e"))])
    _rspec = RuleSpec(
        "rstdp.form",
        _rrule,
        reads={"lr", "d", "e"},
        writes={"w"},
        producers={
            "d": ("dopamine_nuclei", "da_out", "level01"),
            "e": "self",
            "lr": "self",
        },
    )
    _pf = check_producers(_rspec, _mani)
    assert _pf == [], f"the r_stdp producer census must resolve green: {_pf}"
    # ROUTED, not just resolvable (the panel measured 9 of 10 unrouted level01
    # OUT ports passing the name-only check): with the consumer named, da_out
    # must source a declared edge into r_stdp -- and an unrouted port reds.
    _pf_routed = check_producers(_rspec, _mani, consumer="r_stdp")
    assert _pf_routed == [], f"da_out must be ROUTED into r_stdp: {_pf_routed}"
    _unrouted = RuleSpec(
        "rstdp.form",
        _rrule,
        reads={"lr", "d", "e"},
        writes={"w"},
        producers={"d": ("learning_progress", "lp_out", "level01"), "e": "self", "lr": "self"},
    )
    _pf_norot = check_producers(_unrouted, _mani, consumer="r_stdp")
    assert any("NO declared route" in x for x in _pf_norot), (
        f"an unrouted producer must red the route check: {_pf_norot}"
    )
    # reds, each by its finding text: unresolvable; IN-port-as-producer (a REAL
    # IN port picked from the production manifest, never hardcoded); unit lie;
    # missing entry; and the construction refusal for an unread producer key.
    _in_ports = sorted(k for k, p in _mani.ports.items() if p.kind == "IN")
    assert _in_ports, "the production manifest carries no IN port -- the red needs one"
    _in_port = _in_ports[0]
    for _prod, _needle in (
        (("no_such_faculty", "nope", "level01"), "not a declared port"),
        ((_in_port[0], _in_port[1], "level01"), "only an OUT port produces"),
        (("dopamine_nuclei", "da_out", "spikes_hz"), "unit lie"),
    ):
        _bad_spec = RuleSpec(
            "rstdp.form",
            _rrule,
            reads={"lr", "d", "e"},
            writes={"w"},
            producers={"d": _prod, "e": "self", "lr": "self"},
        )
        _bf2 = check_producers(_bad_spec, _mani)
        assert any(_needle in x for x in _bf2), f"expected {_needle!r} in {_bf2}"
    _missing = check_producers(
        RuleSpec(
            "rstdp.form",
            _rrule,
            reads={"lr", "d", "e"},
            writes={"w"},
            producers={"e": "self", "lr": "self"},
        ),
        _mani,
    )
    assert any("no producer declared" in x for x in _missing), (
        f"missing-entry red absent: {_missing}"
    )
    try:
        RuleSpec(
            "rstdp.form",
            _rrule,
            reads={"lr", "d", "e"},
            writes={"w"},
            producers={"zzz": "self", "d": "self", "e": "self", "lr": "self"},
        )
        raise AssertionError("a producer for an unread factor must refuse at construction")
    except ValueError as _ex_p:
        assert "dead declaration" in str(_ex_p), f"wrong refusal: {_ex_p}"
    try:
        check_producers(RuleSpec("bare", _rrule, reads={"lr", "d", "e"}, writes={"w"}), _mani)
        raise AssertionError("a spec with no producers must refuse, not read green")
    except ValueError as _ex_p2:
        assert "vacuous" in str(_ex_p2), f"wrong refusal: {_ex_p2}"

    # the READER: rule_charge on a constructed 3-rule sum composition -- two
    # charged decays + one rule whose only factor is bound 0.0 throughout (the
    # TRUE INERT case: cumulative |dw| exactly 0.0 -- present, composed, moving
    # nothing).
    _alg_c = EligibilityAlgebra(term_budget=1)
    _alg_c.declare_factor("w", "substrate")
    _alg_c.declare_factor("z", "a factor bound to zero throughout (the inert arm)")
    _alg_c.declare_factor("k", "a constant push factor (the cancellation arm)")
    _cc = RuleComposer("sum", target="w")
    _cc.add(RuleSpec("dA", _alg_c.author("dA", [(-0.3, ("w",))]), reads={"w"}, writes={"w"}))
    _cc.add(RuleSpec("dB", _alg_c.author("dB", [(-0.1, ("w",))]), reads={"w"}, writes={"w"}))
    _cc.add(RuleSpec("zz", _alg_c.author("zz", [(5.0, ("z",))]), reads={"z"}, writes={"w"}))
    _verdict = rule_charge(_cc, 1.0, [{"z": 0.0, "k": 1.0}] * 6)
    assert _verdict["per_rule"]["dA"]["verdict"] == "CHARGED"
    assert _verdict["per_rule"]["dB"]["verdict"] == "CHARGED"
    assert _verdict["per_rule"]["zz"]["verdict"] == "INERT", (
        f"the zero-bound rule must read INERT: {_verdict['per_rule']['zz']}"
    )
    assert _verdict["per_rule"]["zz"]["cumulative_abs_dw"] == 0.0
    assert all(v["first_step_cross_check"] is True for v in _verdict["per_rule"].values()), (
        "the sum-policy first-step cross-check must hold on every rule"
    )
    # the DISCRIMINATING red the panel demanded (all three lenses): the c=-2.0
    # orbit cancels a constant push at the endpoint -- the pushing rule wrote
    # every step (cumulative 6*0.5) and must read ENDPOINT-NEUTRAL, never INERT
    # (a two-way endpoint test retired it; the register binds INERT to P3).
    _co = RuleComposer("sum", target="w")
    _co.add(RuleSpec("orbit", _alg_c.author("orbit", [(-2.0, ("w",))]), reads={"w"}, writes={"w"}))
    _co.add(RuleSpec("push", _alg_c.author("push", [(0.5, ("k",))]), reads={"k"}, writes={"w"}))
    _vo = rule_charge(_co, 1.0, [{"k": 1.0, "z": 0.0}] * 2)
    assert _vo["per_rule"]["push"]["verdict"] == "ENDPOINT-NEUTRAL", (
        f"a cancelled-but-contributing rule must read ENDPOINT-NEUTRAL: {_vo['per_rule']['push']}"
    )
    assert _vo["per_rule"]["push"]["cumulative_abs_dw"] == 1.0
    # SEQUENTIAL, both shapes (the adversarial lens's BLOCKER: the first draft
    # crashed mid-A/B neutering the only target reader and burned the composer):
    _sq = RuleComposer("sequential", target="w")
    _sq.add(RuleSpec("decay", _alg_c.author("decay", [(-0.3, ("w",))]), reads={"w"}, writes={"w"}))
    _sq.add(RuleSpec("push", _alg_c.author("push", [(0.2, ("k",))]), reads={"k"}, writes={"w"}))
    _vs = rule_charge(_sq, 1.0, [{"k": 1.0, "z": 0.0}] * 2)
    assert _vs["per_rule"]["decay"]["verdict"] == "UNCHARGEABLE", (
        f"neutering the only target reader must read UNCHARGEABLE, not crash: "
        f"{_vs['per_rule']['decay']}"
    )
    assert _vs["per_rule"]["push"]["verdict"] == "CHARGED"
    assert _vs["per_rule"]["push"]["first_step_cross_check"] == "skipped (policy != 'sum')"
    _sq2 = RuleComposer("sequential", target="w")
    _sq2.add(RuleSpec("dA", _alg_c.author("dA", [(-0.3, ("w",))]), reads={"w"}, writes={"w"}))
    _sq2.add(RuleSpec("dB", _alg_c.author("dB", [(-0.1, ("w",))]), reads={"w"}, writes={"w"}))
    _vs2 = rule_charge(_sq2, 1.0, [{"k": 1.0, "z": 0.0}] * 2)
    assert all(v["verdict"] == "CHARGED" for v in _vs2["per_rule"].values()), (
        f"the two-reader sequential control must charge both: {_vs2['per_rule']}"
    )
    # reader refusals: a stepped composer; an empty seq; a target-carrying seq;
    # and a mid-run kernel refusal re-raised WITH the arm and step named
    try:
        rule_charge(_cc, 1.0, [{"z": 0.0}])
        raise AssertionError("a stepped composer must refuse")
    except ValueError as _ex_c:
        assert "attribution epochs" in str(_ex_c), f"wrong refusal: {_ex_c}"
    _cc2 = RuleComposer("sum", target="w")
    _cc2.add(RuleSpec("dA", _alg_c.author("dA", [(-0.3, ("w",))]), reads={"w"}, writes={"w"}))
    try:
        rule_charge(_cc2, 1.0, [])
        raise AssertionError("an empty seq must refuse")
    except ValueError as _ex_c2:
        assert "zero steps" in str(_ex_c2), f"wrong refusal: {_ex_c2}"
    try:
        rule_charge(_cc2, 1.0, [{"w": 9.0}])
        raise AssertionError("a target-carrying seq must refuse")
    except ValueError as _ex_c3:
        assert "owns the substrate" in str(_ex_c3), f"wrong refusal: {_ex_c3}"
    try:
        rule_charge(_cc2, "abc", [{"z": 0.0}])
        raise AssertionError("a non-scalar w0 must refuse with context")
    except ValueError as _ex_c4:
        assert "w0" in str(_ex_c4), f"wrong refusal: {_ex_c4}"
    _cg = RuleComposer("sum", target="w")
    _cg.add(RuleSpec("grow", _alg_c.author("grow", [(0.5, ("w",))]), reads={"w"}, writes={"w"}))
    _cg.add(
        RuleSpec("nangate", _alg_c.author("nangate", [(1.0, ("z",))]), reads={"z"}, writes={"w"})
    )
    try:
        rule_charge(_cg, 1.0, [{"z": 0.0}, {"z": float("nan")}])
        raise AssertionError("a non-finite binding must refuse at ENTRY")
    except ValueError as _ex_c5:
        assert "bindings_seq[1]" in str(_ex_c5), f"the index must be named: {_ex_c5}"
    assert not _cg.last_attribution()["stepped"], (
        "the entry refusal must leave the composer FRESH (no A/B ran)"
    )
    # the mid-run wrapper: finite bindings whose FULL trajectory overflows --
    # re-raised with the arm and step named (and the composer IS burned, said).
    _vd = rule_charge(_cg, 1.0, [{"z": 0.0}] * 2)  # _cg still fresh: use it once
    assert _vd["per_rule"]["grow"]["verdict"] == "CHARGED"
    _cbig = RuleComposer("sum", target="w")
    _cbig.add(RuleSpec("big", _alg_c.author("big", [(1e155, ("k",))]), reads={"k"}, writes={"w"}))
    _cbig.add(RuleSpec("dA", _alg_c.author("dA", [(-0.3, ("w",))]), reads={"w"}, writes={"w"}))
    try:
        rule_charge(_cbig, 1e155, [{"k": 1e155}, {"k": 1e155}, {"k": 1e155}])
        raise AssertionError("a mid-run overflow must re-raise with the arm and step named")
    except ValueError as _ex_c6:
        assert "arm refused at step" in str(_ex_c6), f"arm/step missing: {_ex_c6}"
    print(
        f"  green  producers x manifest (T051b-iii): jepa's declaration ENUMERATED "
        f"({len(_jepa_factors)} factors, all in the self-set); r_stdp's 'd' resolves AND routes "
        f"to dopamine_nuclei.da_out -> r_stdp on the PRODUCTION assembly ({len(_mani.ports)} "
        f"ports, {len(_mani.edges)} edges); five reds by finding text (unresolvable, "
        f"IN-port-as-producer, unit lie, missing entry, no-declared-route) + two construction "
        f"refusals (unread key, vacuous green). rule_charge (the attribution's first reader, "
        f"gate-run grade): CHARGED/CHARGED/INERT on the trio; ENDPOINT-NEUTRAL on the orbit-"
        f"cancelled push (the three-lens red); sequential UNCHARGEABLE-not-crash + two-reader "
        f"control; 6 refusals by message incl. non-finite-at-ENTRY (composer left fresh) and "
        f"the arm-and-step-named mid-run overflow"
    )

    # 1d) T051b-iv: THE STACK-SANDWICH MERGE -- both live incumbent compositions
    #     (HomeostaticPlasticityStack AND PlasticityRuleStack) expressed as
    #     RuleSpecs held to their PRODUCTION paths.
    # distinct-from: cortex/learned_evc.py (and the other gated-update sites) -- those ARE
    #   learning rules; this block EXPRESSES the two stack sandwiches' pipelines through the
    #   kernel and holds the expression to spied production calls. No other site does that job.
    #     NOT adopt-and-delete, said plainly: both stacks REMAIN the production
    #     path; the merge is the DECLARATION-OF-RECORD layer -- the two
    #     hard-wired sandwiches' write pipelines under the one kernel contract,
    #     coefficients and order machine-checked (the T050 r_stdp form of "held
    #     to equivalence", which the row's own text sanctions). EXPRESSED HERE:
    #     the GAIN pipeline (the genuine two-multiplicative-writer composition:
    #     short-term gain then the BCM brake; the base stack is the
    #     single-writer control) and the base WEIGHTS pipeline (the R-STDP
    #     deposit then the conditional row-uniform scaling trim), each vs SPIED
    #     production calls. DISCLOSED REMAINDER: the subclass weights phases
    #     (E/I trim + Benna-Fusi -- ii's component grain; the tissue wire is
    #     T054's), rate (Triesch cell-state, single writer -- no composition
    #     question), theta (BCM state). The multiplicative-to-additive bridge:
    #     g' = g*f becomes dw = g*(f-1) with (f-1) pre-collapsed from the
    #     recorded production value.
    from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron
    from snn_data_lake.plasticity.homeostatic_plasticity_stack import (
        HomeostaticPlasticityStack,
        _drive_code,
        _rstdp_update_fn,
    )
    from snn_data_lake.plasticity.plasticity_rule_stack import PlasticityRuleStack

    _alg_g = EligibilityAlgebra(term_budget=1)
    _alg_g.declare_factor("gain", "the composed gain substrate (seeds 1.0 per pre_gain call)")
    _alg_g.declare_factor("f_short_m1", "short-term Tsodyks-Markram gain minus 1 (recorded)")
    _alg_g.declare_factor("f_bcm_m1", "BCM metaplastic brake minus 1 (recorded)")
    _r_short = _alg_g.author("stack.short_term", [(1.0, ("gain", "f_short_m1"))])
    _r_bcm = _alg_g.author("stack.bcm_brake", [(1.0, ("gain", "f_bcm_m1"))])

    # spy the SUBCLASS's production pre_gain: instance-level wrappers RECORD the
    # two phase outputs without altering them (f_short is exact -- ctx gain
    # seeds 1.0 and the base has ONE PRE adapter; f_bcm is _meta.gate's return).
    _sub = PlasticityRuleStack(1, seed=0)
    _sg_box, _mg_box = [], []
    _orig_sh = _sub._short.stabilize
    _orig_gate = _sub._meta.gate

    def _rec_sh(ctx):
        out = _orig_sh(ctx)
        _sg_box.append(out["gain"])
        return out

    def _rec_gate(i):
        v = _orig_gate(i)
        _mg_box.append(v)
        return v

    _sub._short.stabilize = _rec_sh
    _sub._meta.gate = _rec_gate

    def _k_ulp(rule):  # K counted from the authored structure (the i method)
        return float(sum(len(_fs9) for _c9, _fs9 in rule.terms) + len(rule.terms))

    _code_d, _act_d, _tp_d = _drive_code(10)
    _nrn_w = RSTDPNeuron(
        10, lr=0.1, a_plus=1.0, a_minus=1.0, tau_plus=8.0, tau_minus=8.0, w0=0.3, seed=0
    )
    _updw = _rstdp_update_fn(_nrn_w, _code_d, _tp_d)
    # warm UNTIL the BCM brake actually ENGAGES (gate < 1.0), derived from the
    # substrate itself, never a step count: the panel measured a 12-step warm
    # leaving theta (0.0375) BELOW theta0 (0.05) -- the gate PINNED at its clip
    # ceiling 1.0 on every spied call, so the "two-writer" arm measured ONE
    # writer and DELETING the brake (coefficient 0.0) passed 8/8. A pinned
    # gauge can be a live signal -- the recorded lesson, relearned here.
    # First crossing on this drive: step ~1251 (panel-measured).
    _warm_n = 0
    while _orig_gate(0) >= 1.0:
        _sub.learning_step(0, list(_nrn_w.w), _act_d, _updw)
        _warm_n += 1
        assert _warm_n <= 3000, "the BCM gate never engaged -- the warm drive is wrong"
    _sg_box.clear()
    _mg_box.clear()
    _k21 = _k_ulp(_r_short) + _k_ulp(_r_bcm)
    _gain_cases = 0
    worst_gain = 0.0
    _gain_binds = []
    _fs_seen, _fb_seen = set(), set()
    for _gstep in range(8):
        _sg_box.clear()
        _mg_box.clear()
        _g_out = _sub.pre_gain(0)  # THE production call (SpikingInteractionLearner's API)
        _fs, _fb = _sg_box[0], _mg_box[0]
        assert _fb != 1.0 and _fs != 1.0, (
            f"a phase output pinned at identity on the measured call (fs={_fs!r}, fb={_fb!r}) "
            "-- the pinned-gauge degeneracy; re-derive the warm"
        )
        _fs_seen.add(_fs)
        _fb_seen.add(_fb)
        _gain_binds.append({"f_short_m1": _fs - 1.0, "f_bcm_m1": _fb - 1.0})
        _cg2 = RuleComposer("sequential", target="gain")
        _cg2.add(RuleSpec("short_term", _r_short, reads={"gain", "f_short_m1"}, writes={"gain"}))
        _cg2.add(RuleSpec("bcm_brake", _r_bcm, reads={"gain", "f_bcm_m1"}, writes={"gain"}))
        _g_comp, _ = _cg2.step(1.0, {"gain": 1.0, "f_short_m1": _fs - 1.0, "f_bcm_m1": _fb - 1.0})
        _tolg = _k21 * _sys2.float_info.epsilon * (abs(_fs) + abs(_fb) + 1.0)
        _dg = abs(_g_comp - _g_out)
        assert _dg <= _tolg, (
            f"the gain composition diverges from the production pre_gain at step {_gstep}: "
            f"|d|={_dg!r} > {_tolg:.1e}"
        )
        worst_gain = max(worst_gain, _dg)
        _gain_cases += 1
        _sub.learning_step(0, list(_nrn_w.w), _act_d, _updw)  # move the state between calls
    # in-command mis-declaration control on the SECOND writer (the panel measured
    # the unfixed arm passing a brake coefficient of 0.0 -- the declaration was
    # unfalsifiable by its own arm): near-deleting the brake must red every
    # engaged case against the incumbent product.
    _alg_bad9 = EligibilityAlgebra(term_budget=1)
    _alg_bad9.declare_factor("gain", "control copy (near-deleted brake)")
    _alg_bad9.declare_factor("f_bcm_m1", "control copy (near-deleted brake)")
    _r_bcm_del = _alg_bad9.author("stack.bcm_brake.deleted", [(1e-30, ("gain", "f_bcm_m1"))])
    _ctrl_reds = 0
    for _bind9 in _gain_binds:
        _cgx = RuleComposer("sequential", target="gain")
        _cgx.add(RuleSpec("short_term", _r_short, reads={"gain", "f_short_m1"}, writes={"gain"}))
        _cgx.add(RuleSpec("bcm_del", _r_bcm_del, reads={"gain", "f_bcm_m1"}, writes={"gain"}))
        _gx, _ = _cgx.step(1.0, {"gain": 1.0, **_bind9})
        _fsx = _bind9["f_short_m1"] + 1.0
        _fbx = _bind9["f_bcm_m1"] + 1.0
        _tolx = _k21 * _sys2.float_info.epsilon * (abs(_fsx) + abs(_fbx) + 1.0)
        if abs(_gx - _fsx * _fbx) > _tolx:
            _ctrl_reds += 1
    assert _ctrl_reds == len(_gain_binds), (
        f"a near-deleted brake must red EVERY engaged case ({_ctrl_reds}/{len(_gain_binds)}) "
        "-- any pass means the brake's declaration is unfalsifiable on that case"
    )
    # the single-writer CONTROL: the base stack's pre_gain is f_short alone
    _base_g = HomeostaticPlasticityStack(1, seed=0)
    _bg_box = []
    _orig_bsh = _base_g._short.stabilize

    def _rec_bsh(ctx):
        out = _orig_bsh(ctx)
        _bg_box.append(out["gain"])
        return out

    _base_g._short.stabilize = _rec_bsh
    for _bstep in range(4):
        _bg_box.clear()
        _g_out_b = _base_g.pre_gain(0)
        _cg1 = RuleComposer("sequential", target="gain")
        _cg1.add(RuleSpec("short_term", _r_short, reads={"gain", "f_short_m1"}, writes={"gain"}))
        _g_comp_b, _ = _cg1.step(1.0, {"gain": 1.0, "f_short_m1": _bg_box[0] - 1.0})
        assert abs(_g_comp_b - _g_out_b) <= _k_ulp(_r_short) * _sys2.float_info.epsilon * 2.0, (
            f"the single-writer control diverged: {_g_comp_b!r} vs {_g_out_b!r}"
        )
    # the reader on the expressed pair, fed the RECORDED production bindings
    # (the panel measured the first draft's hand-typed bindings producing
    # CHARGED/CHARGED while the recorded ones -- brake pinned -- read INERT;
    # with the engaged warm the recorded bindings are the honest input):
    _cgr = RuleComposer("sequential", target="gain")
    _cgr.add(RuleSpec("short_term", _r_short, reads={"gain", "f_short_m1"}, writes={"gain"}))
    _cgr.add(RuleSpec("bcm_brake", _r_bcm, reads={"gain", "f_bcm_m1"}, writes={"gain"}))
    _vg = rule_charge(_cgr, 1.0, _gain_binds)
    assert all(v["verdict"] == "CHARGED" for v in _vg["per_rule"].values()), (
        f"both expressed gain writers must charge on the RECORDED bindings: {_vg['per_rule']}"
    )
    # ORDER is NOT claimed for this pair, said plainly: two pure multiplicative
    # writers commute (the panel measured the order-swap mutant passing even in
    # the engaged regime); the order/policy claim lives on the weights pair
    # below, where sequential and sum genuinely differ.

    # base WEIGHTS pipeline THROUGH THE KERNEL (the panel measured the first
    # draft hand-composing evaluate() calls -- no RuleSpec, no composer, no
    # reader -- while claiming kernel expression): the R-STDP deposit (the
    # T050 canonical dw=lr*d*e; d = pre_gain output x reward) THEN the
    # conditional trim, composed SEQUENTIALLY on target "w" per synapse.
    # distinct-from: plasticity/meta_plasticity.py -- that records rates to lag theta (a
    #   rule's own state); this records the incumbent's applied weights to hold a
    #   declaration to them (a harness spy). Site read.
    # learning_step takes the deposit closure FROM the caller, so the harness
    # records the applied weights natively -- no monkeypatch on the write path.
    _alg_w = EligibilityAlgebra(term_budget=1)
    _alg_w.declare_factor("lr", "the neuron's own learning rate")
    _alg_w.declare_factor("d", "the effective reward: pre_gain output x reward (+1 here)")
    _alg_w.declare_factor("e", "the synapse's eligibility trace at reward time")
    _alg_w.declare_factor("w", "the weight (the trim's substrate)")
    _alg_w.declare_factor("trim_m1", "row-uniform scaling ratio minus 1; 0.0 when untrimmed")
    _r_dep = _alg_w.author("stack.rstdp_deposit", [(1.0, ("lr", "d", "e"))])
    _r_trim = _alg_w.author("stack.scaling_trim", [(1.0, ("w", "trim_m1"))])
    _base_w = HomeostaticPlasticityStack(1, seed=1)
    _nrn2 = RSTDPNeuron(
        10, lr=0.1, a_plus=1.0, a_minus=1.0, tau_plus=8.0, tau_minus=8.0, w0=0.3, seed=1
    )
    _upd2 = _rstdp_update_fn(_nrn2, _code_d, _tp_d)
    _rec = []

    def _rec_upd(gain, weights):
        before = list(weights)
        out = _upd2(gain, weights)
        _rec.append((gain, before, list(out)))
        return out

    # the eligibility at reward time, recomputed EXACTLY as the closure does
    # (fresh neuron, same params/seed, replay the code): deterministic bit-same.
    _nrn_e = RSTDPNeuron(
        10, lr=0.1, a_plus=1.0, a_minus=1.0, tau_plus=8.0, tau_minus=8.0, w0=0.3, seed=1
    )
    _nrn_e.e = [0.0] * _nrn_e.n
    _nrn_e.t_pre = [None] * _nrn_e.n
    _nrn_e.t_post = None
    for _i9, _t9 in enumerate(_code_d):
        _nrn_e.pre_spike(_i9, float(_t9))
    _nrn_e.post_spike(_tp_d)
    _e_at_reward = list(_nrn_e.e)
    _w_cur = list(_nrn2.w)
    n_trim = 0
    n_untrim = 0
    worst_dep = 0.0
    _gains_seen = set()
    _kw = _k_ulp(_r_dep) + _k_ulp(_r_trim)  # K counted, never fitted (panel: the
    _sum_policy_differs = False  # first draft's 8/64 bands were picked; 1-ulp passes)
    for _wstep2 in range(60):
        _rec.clear()
        _w_final = _base_w.learning_step(0, _w_cur, _act_d, _rec_upd)[0]
        _gain_in, _w_before, _w_mid = _rec[0]
        _gains_seen.add(_gain_in)
        for _j9 in range(10):  # the deposit vs the APPLIED delta (the i lesson)
            _dw_dec = _r_dep.evaluate(lr=_nrn2.lr, d=_gain_in * 1.0, e=_e_at_reward[_j9])
            _dw_app = _w_mid[_j9] - _w_before[_j9]
            _dd = abs(_dw_dec - _dw_app)
            _tolw = _k_ulp(_r_dep) * _sys2.float_info.epsilon * (abs(_dw_dec) + abs(_w_before[_j9]))
            assert _dd <= _tolw, (
                f"the deposit declaration diverges from the applied delta at synapse {_j9}, "
                f"step {_wstep2}: |d|={_dd!r} > {_tolw:.1e}"
            )
            worst_dep = max(worst_dep, _dd)
        if _w_final == _w_mid:
            n_untrim += 1
        else:
            n_trim += 1
            _ratios = [_w_final[_j9] / _w_mid[_j9] for _j9 in range(10) if _w_mid[_j9] != 0.0]
            assert _ratios, "an all-zero trimmed row leaves no ratio to bind (panel overflow)"
            _s_row = _ratios[0]
            # ROW-UNIFORMITY is ENTAILED, not measured -- HomeostaticNeuron's
            # stabilize applies ONE scalar f to the whole row, so this is a
            # structural cross-check on that two-line body (the 1b c0-wording;
            # the first draft advertised it as "a claim made checkable").
            assert all(
                abs(_r9 - _s_row) <= 8.0 * _sys2.float_info.epsilon * abs(_s_row) for _r9 in _ratios
            ), "the entailed row-uniformity cross-check failed -- the incumbent's body changed"
            # the trim MAGNITUDE bound to Turrigiano's OWN rule (the panel
            # measured an arbitrary uniform 0.5 scale passing the first draft:
            # the ratio was derived from the value it predicted). Expected
            # f = (target / drive)**scale_rate from the incumbent's cell.
            _cell9 = _base_w._homeo._cell[0]
            _y9 = sum(_w_mid[_j8] * _act_d[_j8] for _j8 in range(10))
            _f_exp = (_cell9.target / _y9) ** _cell9.scale_rate
            assert abs(_s_row - _f_exp) <= 64.0 * _sys2.float_info.epsilon * abs(_f_exp), (
                f"the trim ratio {_s_row!r} is not Turrigiano's own "
                f"(target/drive)**scale_rate = {_f_exp!r} -- an arbitrary uniform scale "
                "would satisfy uniformity alone (panel-constructed)"
            )
            for _j9 in range(10):
                # deposit THEN trim, THROUGH the kernel: the sequential composer
                # per synapse must land on the incumbent's w_final...
                _cw = RuleComposer("sequential", target="w")
                _cw.add(RuleSpec("deposit", _r_dep, reads={"lr", "d", "e"}, writes={"w"}))
                _cw.add(RuleSpec("trim", _r_trim, reads={"w", "trim_m1"}, writes={"w"}))
                _wk, _ = _cw.step(
                    _w_before[_j9],
                    {
                        "lr": _nrn2.lr,
                        "d": _gain_in * 1.0,
                        "e": _e_at_reward[_j9],
                        "trim_m1": _s_row - 1.0,
                    },
                )
                assert abs(_wk - _w_final[_j9]) <= _kw * _sys2.float_info.epsilon * (
                    abs(_w_final[_j9]) + 1.0
                ), f"the kernel-composed deposit-then-trim missed w_final at synapse {_j9}"
            # ...and the SUM policy must MISS it (the order/policy claim lives
            # HERE -- deposit does not read w, trim does, so sum applies the
            # trim to the PRE-deposit weight; the gain pair commutes and
            # carries no order claim): one synapse suffices per trimmed step.
            _cs = RuleComposer("sum", target="w")
            _cs.add(RuleSpec("deposit", _r_dep, reads={"lr", "d", "e"}, writes={"w"}))
            _cs.add(RuleSpec("trim", _r_trim, reads={"w", "trim_m1"}, writes={"w"}))
            _ws, _ = _cs.step(
                _w_before[0],
                {
                    "lr": _nrn2.lr,
                    "d": _gain_in * 1.0,
                    "e": _e_at_reward[0],
                    "trim_m1": _s_row - 1.0,
                },
            )
            if abs(_ws - _w_final[0]) > _kw * _sys2.float_info.epsilon * (abs(_w_final[0]) + 1.0):
                _sum_policy_differs = True
        _w_cur = _w_final
    assert n_trim >= 1 and n_untrim >= 1, (
        f"both trim branches must be spied (trimmed {n_trim}, untrimmed {n_untrim}) -- "
        "a one-branch sweep is the vacuous-coverage shape"
    )
    assert _sum_policy_differs, (
        "the SUM policy must measurably miss the incumbent on at least one trimmed step -- "
        "otherwise the weights pair carries no order/policy content either"
    )
    # deposit mis-declaration control (the panel measured 1d shipping ZERO
    # in-command reds): a 1e-9-relative coefficient error must red every site.
    _alg_badw = EligibilityAlgebra(term_budget=1)
    for _fname in ("lr", "d", "e"):
        _alg_badw.declare_factor(_fname, "control copy (mis-declared deposit)")
    _r_dep_bad = _alg_badw.author("stack.rstdp_deposit.bad", [(1.0 + 1e-9, ("lr", "d", "e"))])
    _dep_reds = 0
    for _j9 in range(10):
        _dw_bad = _r_dep_bad.evaluate(lr=_nrn2.lr, d=_gain_in * 1.0, e=_e_at_reward[_j9])
        _dw_app = _w_mid[_j9] - _w_before[_j9]
        if abs(_dw_bad - _dw_app) > _k_ulp(_r_dep) * _sys2.float_info.epsilon * (
            abs(_dw_bad) + abs(_w_before[_j9])
        ):
            _dep_reds += 1
    assert _dep_reds == 10, f"a 1e-9 mis-declared deposit must red all 10 synapses ({_dep_reds}/10)"
    # rule_charge on the weights pair -- the honest verdicts: deposit CHARGED;
    # trim UNCHARGEABLE (neutering it leaves no target reader -- the iii
    # verdict demonstrated on an incumbent-derived pair, not a construction).
    _cwr = RuleComposer("sequential", target="w")
    _cwr.add(RuleSpec("deposit", _r_dep, reads={"lr", "d", "e"}, writes={"w"}))
    _cwr.add(RuleSpec("trim", _r_trim, reads={"w", "trim_m1"}, writes={"w"}))
    _vw = rule_charge(
        _cwr,
        0.3,
        [{"lr": _nrn2.lr, "d": 0.6, "e": _e_at_reward[0], "trim_m1": -0.02}] * 2,
    )
    assert _vw["per_rule"]["deposit"]["verdict"] == "CHARGED"
    assert _vw["per_rule"]["trim"]["verdict"] == "UNCHARGEABLE"
    print(
        f"  green  the stack-sandwich MERGE (T051b-iv): GAIN expressed as the sequential "
        f"two-writer composition vs the SPIED production pre_gain with the BCM brake ENGAGED "
        f"(warm derived: {_warm_n} steps to gate<1; {len(_fs_seen)}/{len(_fb_seen)} distinct "
        f"phase values on {_gain_cases} calls; worst |d|={worst_gain:.1e}; near-deleted-brake "
        f"control REDS {_ctrl_reds}/{_gain_cases}; order NOT claimed -- multiplicative pair "
        f"commutes); rule_charge CHARGED/CHARGED on the RECORDED bindings; base WEIGHTS "
        f"THROUGH THE KERNEL (deposit-then-trim sequential per synapse lands on w_final, "
        f"worst deposit |d|={worst_dep:.1e} over 60 steps [{len(_gains_seen)} distinct gains] "
        f"x 10 synapses; SUM policy measurably misses -- the order/policy content; trim "
        f"magnitude bound to Turrigiano's own (target/drive)**scale_rate, row-uniformity an "
        f"ENTAILED cross-check; {n_trim} trimmed / {n_untrim} untrimmed; 1e-9 mis-declared "
        f"deposit REDS 10/10); rule_charge deposit=CHARGED trim=UNCHARGEABLE (the iii verdict "
        f"on an incumbent-derived pair); E/I + Benna-Fusi + rate + theta = the disclosed "
        f"remainder (ii's component grain; T054); both stacks REMAIN production"
    )

    # 1e) T051c: THE CLAIM CROSS-CHECK + THE CENSUS RATCHET. check_claim holds a
    #     RuleSpec.claim_key to the tree's DECLARED GRADIENT_CLAIM keys, derived
    #     from ONE census() call (the panel measured the first draft running TWO
    #     full AST walks -- walk() + census()-which-walks-again -- for a key set
    #     that is a subset of the census rows; 8.3x self-check slowdown, halved
    #     here); rule_census_check holds tools/rule_census.json to the live
    #     census; the artifact path is the gate's own marker-anchored _repo_root
    #     (the first draft's counted parents[] climb resolved OUTSIDE the repo
    #     with a non-terminating fallback -- panel-constructed).
    import json as _json

    from snn_data_lake.substrate.gradient_claim_gate import (
        _repo_root as _gc_root,
    )
    from snn_data_lake.substrate.gradient_claim_gate import (
        census as _gc_census,
    )

    _gc_rules, _gc_fails, _ = _gc_census()
    assert not _gc_fails, f"census failures: {_gc_fails}"
    _declared_keys = {
        r["qualname"].replace("\\", "/") for r in _gc_rules if r["declared"] is not None
    }
    _spec_ok = RuleSpec(
        "claimed",
        _alg_c.author("claimed", [(-0.1, ("w",))]),
        reads={"w"},
        writes={"w"},
        claim_key="basal_ganglia/r_stdp.py::RSTDPNeuron.reward",
    )
    assert check_claim(_spec_ok, _declared_keys) == [], "a real declared claim must resolve green"
    _spec_ph = RuleSpec(
        "phantom",
        _alg_c.author("phantom", [(-0.1, ("w",))]),
        reads={"w"},
        writes={"w"},
        claim_key="basal_ganglia/r_stdp.py::EligibilityAlgebra.evaluate",
    )
    _phf = check_claim(_spec_ph, _declared_keys)
    assert any("phantom-qualname" in x for x in _phf), (
        f"the T050 phantom class (a claim key naming no declaration) must red: {_phf}"
    )
    try:
        check_claim(
            RuleSpec("bare2", _alg_c.author("bare2", [(-0.1, ("w",))]), reads={"w"}, writes={"w"}),
            _declared_keys,
        )
        raise AssertionError("a spec with no claim_key must refuse, not read green")
    except ValueError as _ex_k:
        assert "vacuous" in str(_ex_k), f"wrong refusal: {_ex_k}"

    import pathlib as _pathlib2

    _art_path = _pathlib2.Path(_gc_root()) / "tools" / "rule_census.json"
    _artifact = _json.loads(_art_path.read_text(encoding="utf-8"))
    assert len(_declared_keys) >= len(_artifact["rules"]), (
        f"the declared surface ({len(_declared_keys)}) fell below the census rows "
        f"({len(_artifact['rules'])}) -- walker defect (floor DERIVED, the panel measured "
        "the first draft's round-100 floor passing a 40%-loss walker)"
    )
    _cf = rule_census_check(_artifact, _gc_rules)
    assert _cf == [], f"the rule census must hold to the live tree: {_cf}"
    # the ratchet's reds, constructed on COPIES (the artifact itself is tracked):
    _art_lag = {"_meta": _artifact["_meta"], "rules": dict(_artifact["rules"])}
    _art_lag["rules"].pop(sorted(_art_lag["rules"])[0])
    _lf = rule_census_check(_art_lag, _gc_rules)
    assert any("lags the tree" in x for x in _lf), "a live claim missing must red"
    assert any("census rows fell" in x for x in _lf), (
        "the rows-fell arm must also fire on the same construction (its positive control)"
    )
    _art_stale = {"_meta": _artifact["_meta"], "rules": dict(_artifact["rules"])}
    _art_stale["rules"]["ghost/mod.py::Ghost.update"] = {
        "kind": "LOCAL",
        "governed": True,
        "disposition": "UNDISPOSITIONED",
    }
    _sf = rule_census_check(_art_stale, _gc_rules)
    assert any("stale row" in x for x in _sf) and any("UNDISPOSITIONED rose" in x for x in _sf), (
        f"a ghost row must red as stale AND as an undispositioned rise: {_sf}"
    )
    # the lowercase/empty-disposition red (the panel measured '' and 'undispositioned'
    # silently counting as DISPOSITIONED under the one-token compare):
    _art_case = {"_meta": _artifact["_meta"], "rules": dict(_artifact["rules"])}
    _first_disp = next(
        q
        for q, v in sorted(_art_case["rules"].items())
        if not _is_undispositioned(v["disposition"])
    )
    _art_case["rules"][_first_disp] = {**_art_case["rules"][_first_disp], "disposition": ""}
    assert any("UNDISPOSITIONED rose" in x for x in rule_census_check(_art_case, _gc_rules)), (
        "an EMPTIED disposition must count as undispositioned and red the rise arm"
    )
    # the regen path's own red: a retraction through the SANCTIONED door must refuse
    # (the panel measured the first draft's baseline recomputed by its edit path --
    # green through the very door it governs). Constructed via a temp prior file?
    # No -- regen_rule_census reads the REAL artifact; the retraction refusal is
    # exercised by regenerating against a HAND-ADVANCED copy... the honest cheap
    # form: call the internal comparison the way regen does, on copies.
    _prior_rows = dict(_artifact["rules"])
    _new_rows = dict(_artifact["rules"])
    _new_rows[_first_disp] = {**_new_rows[_first_disp], "disposition": "UNDISPOSITIONED"}
    _regr = sorted(
        q
        for q, v in _new_rows.items()
        if q in _prior_rows
        and not _is_undispositioned(_prior_rows[q]["disposition"])
        and _is_undispositioned(v["disposition"])
    )
    assert _regr == [_first_disp], (
        f"the per-row monotonicity comparison must catch the constructed retraction: {_regr}"
    )
    _n_undisp_now = sum(
        1 for v in _artifact["rules"].values() if _is_undispositioned(v["disposition"])
    )
    print(
        f"  green  claim cross-check + census ratchet (T051c): claim_key held to "
        f"{len(_declared_keys)} declared GRADIENT_CLAIM keys derived from ONE census call "
        f"(real key green, the T050 phantom key REDS, no-key + empty-surface refuse, "
        f"incoming keys normalized); tools/rule_census.json holds to the live census -- "
        f"{len(_artifact['rules'])} rule-bearing rows, {_n_undisp_now} UNDISPOSITIONED (LIVE "
        f"counts from the committed artifact, not a frozen figure -- a 'day-one' label on "
        f"this line once carried a drifting number, the T052-i panel's overflow class; "
        f"regen_rule_census is the sanctioned edit path and REFUSES "
        f"retractions + dead entries -- the mechanized ratchet); lag/rows-fell/stale/rise/"
        f"emptied-disposition reds each constructed, the retraction comparison verified"
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
        "  RED    refusals BY MESSAGE (13 constructions + 6 behavioural -- THIS battery's census; "
        "the 1b/1c arms carry their own): incl. disjoint-writes routing, "
        "the sequential-degrades-to-sum trap, target-binding disagreement, composed-total overflow, "
        "sealed-after-step; solo-LOO returns the identity (the base case, not an error)"
    )

    print()
    print(
        "DISCLOSURES: attribution is the instrument's KERNEL -- its first reader is "
        "rule_charge (T051b-iii): the per-rule charge A/B consuming last_attribution() "
        "and holding the sum-policy first-step prediction to the measured LOO arm. "
        "GATE-RUN grade, said plainly: zero live compositions exist, so TICK-live "
        "reading arrives with T053's charged composition, not before (the neutered "
        "information stays recoverable from step()'s returned parts -- the instrument "
        "is not the sole carrier). check_producers (FR-019) resolves against the "
        "PRODUCTION manifest assembly -- name/kind/unit WITHOUT `consumer`, "
        "ROUTE-checked with it (a resolvable name is not a wire: 9 of 10 unrouted "
        "level01 ports passed the name-only form, panel-measured); 'self' entries are "
        "declared, not resolved (no third port model). THE CENSUS, said as done: "
        "jepa's declaration is ENUMERATED (its factors held to the self-set "
        "in-harness); r_stdp's form is HAND-ADJUDICATED against the algebra's "
        "incumbent-held arm; the tree-wide producers enumeration is T051c-wave's "
        "(open, undated). The claim cross-check + the ratcheted disposition "
        "census (99 rows at its T051c landing; the live count is the artifact's own "
        "ratchet_baseline, never this string) LANDED at T051c: check_claim (phantom keys red against the declared "
        "surface; existence-only, ungoverned-resolves-green, keys normalized) + "
        "tools/rule_census.json held to the live census by rule_census_check, with "
        "--regen-census on THIS module as the sanctioned edit path refusing "
        "retractions and dead entries. TWO ratcheted artifacts ride ONE census "
        "walker, different quantities (the P2 sentence the panel demanded): "
        "gradient_claim_baseline.json ratchets CLASSIFICATION counts "
        "(machine-written, refuses worse); rule_census.json ratchets DISPOSITIONS "
        "(the regen refuses retraction). The 'level01'/'spikes_hz' strings are CROSS-TIER PINS on "
        "dopamine_nuclei.da_out's declared unit, not exercise values -- a red there "
        "attributes to the PORT, not this module. check_producers and rule_charge are "
        "MODULE-LEVEL functions: outside method_dormancy_gate's ClassDef scan, no "
        "LEARN_VERB, host outside reachable() -- invisible to both method-grade "
        "gates, and this gate-run self-check is their only charged surface, said "
        "plainly. Arbitration ORDER is bare add() insertion order -- not learned, not "
        "recorded in the payload; the register's order disclosure is paid there. "
        "The claim_key cross-check LANDED at T051c (check_claim: a phantom key reds "
        "against the gate's own declared surface), with the per-rule disposition "
        "census (tools/rule_census.json; 99 rows / 56 UNDISPOSITIONED / 43 dispositioned "
        "were the DAY-ONE figures -- the committed artifact's own ratchet_baseline; a '57' "
        "that stood here through T051c was a pre-landing draft figure the artifact never "
        "held, caught by the T052-i panel) held to the live tree by rule_census_check -- the "
        "WAVE advances by dispositioning rows, the ratchet forbids lag/stale/rise. "
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
        "pairs), the 8/32-ulp band factors, and the iv sandwich arm's tranche "
        "(stack seeds 0/1, the RSTDPNeuron params lr=0.1/tau=8/w0=0.3/n=10, the "
        "8-call/60-step horizons, the 3000-step warm bound, the 64-ulp trim-"
        "magnitude band -- the 'level01' pins note applies to none of these; "
        "they are exercise values). The stack MERGE (both incumbent "
        "sandwiches as RuleSpecs, held to equivalence) LANDED at T051b-iv (harness 1d): "
        "the GAIN pipeline as the sequential two-writer composition vs the spied "
        "production pre_gain, the base WEIGHTS pipeline (T050-canonical deposit vs the "
        "APPLIED deltas + the conditional row-uniform trim), rule_charge on the "
        "expressed pair; NOT adopt-and-delete -- both stacks REMAIN the production "
        "path, the merge is the declaration-of-record layer; E/I + Benna-Fusi + rate "
        "+ theta are the disclosed remainder (component grain at ii; tissue at T054)."
    )
    print(
        "self-check OK: boundary-scoped naive-sum reds (orbit + open-interval divergence + "
        "faster-convergence disclosure) + the T048 pair at component grain (authored-from-"
        "incumbent structural x drag; sequential==component step to the bit, sum "
        "sign-violates at the excitatory prune; wrapper/relax/live-path undriven, said) + "
        "producers x the production manifest (five reds by finding text incl. the route red, "
        "two construction refusals, jepa's declaration enumerated) + rule_charge (the "
        "attribution's first reader: three-way verdicts incl. ENDPOINT-NEUTRAL on the "
        "orbit-cancelled push and sequential UNCHARGEABLE-not-crash; 6 refusals) + the "
        "stack-sandwich MERGE (gain two-writer + base weights deposit-then-trim, each vs "
        "spied production calls; remainder disclosed) + the claim cross-check and census "
        "ratchet (T051c: a phantom claim_key reds against the declared surface; the "
        "disposition artifact held to the live census, lag/stale/rise reds constructed) + "
        "causal decomposition with the unsound predictor's constructed bit-red + swept "
        "sequential guarantee + consumed attribution surface + 19 battery refusals "
        "(section 5's census, battery-local)"
    )
    return 0


if __name__ == "__main__":
    import sys as _sys_main

    if "--regen-census" in _sys_main.argv[1:]:
        _out, _nrows, _nund = regen_rule_census()
        print(f"wrote {_out}: rows={_nrows} undispositioned={_nund}")
        raise SystemExit(0)
    raise SystemExit(main())
