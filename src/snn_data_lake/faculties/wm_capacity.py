"""
WM capacity DERIVED, never picked -- K = T_max/t_s (Mi-Katkov-Tsodyks) from the surviving
store's own synaptic constants, with the five guards FR-020 owes (specs/022 T056).

WHAT SETTLED HERE, so the next reader does not re-derive it. T055 built a standalone
synaptic WM (`plasticity/wm_attractor.py`), measured it IDENTICAL to the incumbent
`faculties/prefrontal_executive` on the settling benchmark (tuned, both (4 hits, 0 FA); with
distractors both (4, 2), neither suppressing), and its dormancy row read RETIRE dated T056.
T056 is the port-then-retire P2's honesty guard requires: the unique capability -- the
DECOUPLED retention/release pair (facilitation u on tau_f vs resources R on tau_d) and a
capacity that FALLS OUT of those time constants -- was ported into the surviving store
(each goal frame now carries a Tsodyks-Markram synapse; a single resource-gated rehearsal
channel refreshes background frames; `max_goals` defaults to the DERIVED K), and
`wm_attractor.py` was DELETED. Exactly one mechanism survives.

THE P2/P8(5) VERDICT, measured not asserted: the port is BETTER than the declared constant
on the measurement that exists -- the old `max_goals=4` was unfalsifiable (frozen lower
frames survived any delay for free, so ANY declared number "held"; T055 measured the
freeze: the incumbent's hit count ROSE with distractor count), while the derived K carries
a reachable red in each direction: it must match the behaviorally measured capacity within
+/-1 (guard 1, and it is dynamics that kill over-capacity frames, not the cap check), it
must MOVE under a tau_f sweep (guard 2), and it must REFUSE when the mechanism cannot
support a derivation at all (guard 3). Had the port measured no better, P8(5) said delete
it too; it did not come to that, and this paragraph is the record of why.

THE DERIVATION, all quantities MEASURED on the mechanism itself (never computed from a
formula the mechanism does not implement -- the T_max probe reads decay through the
store's own elapsed-time read, so a frozen synapse is SEEN as frozen rather than assumed
to decay):
  u_ign  : minimal facilitation that still ignites the rehearsal pool at full resources
           (binary search over the pool itself) -- the retention floor, derived not picked.
  D, u_hi: serve a frame from its own ignition floor until RESOURCES end the burst --
           D fired ticks (the dwell), u_hi the facilitation the refresh restores.
  t_s    : D + 1 -- the serial channel occupancy per frame per cycle (dwell + the
           handoff tick on which depleted resources yield the channel).
  T_max  : largest silent gap after which u_hi still ignites at recovered resources,
           binary-searched THROUGH the frame's own decayed() read.
  K      : floor((T_max + D - 1) / t_s) -- survive iff the unrefreshed gap
           k*t_s - D + 1 <= T_max. The store adds +1 for the active frame, which rides
           its own per-tick refresh outside the rotation.
REGIME PRECONDITION, refused outside it: T_max assumes full resource recovery between a
frame's services (cycle K*t_s >= 5*tau_d). Violated, the formula is measurably ~7% high
(derived 97 vs held 90 at cycle ~2*tau_d); inside, the match is EXACT (97==97, 258==258).
The store's default constants sit inside the regime; derive_capacity REFUSES outside it.

CHOSEN, not derived, each disclosed with what is known of its cliff:
  - the rehearsal gain (`_REFRESH_GAIN`, 1.1): swept at the default constants through the
    store's own path (seed-u viability included), gain 0.7 is REFUSED outright -- its
    ignition floor (0.2468) sits above the seed U=0.2, so no store exists there (the fold
    panel caught this endpoint published as "K=370", a pre-viability-guard artifact
    describing a store whose behavioral capacity is zero) -- and the admitted band moves
    K 461 -> 482 -> 536 -> 513 over 0.9/1.1/1.3/1.5 (t_s shifts 3 -> 5, so the response
    is non-monotone at the top). The derivation is exact GIVEN the operating point; the
    operating point is a choice, said plainly.
  - the regime factor 5 in `K*t_s >= 5*tau_d`: a picked safety margin BRACKETED by two
    measurements, not itself swept -- the formula fails at cycle ~2*tau_d (derived 97 vs
    held 90) and is exact at ~9.6*tau_d (97==97, 258==258); 5 sits between them on the
    safe side. A tighter factor would need the boundary swept, which has not been done.
  - the derivation's validated MEMBRANE is the store's default (leak=0.2, md=0.8):
    measured drift derived-vs-held is 0 at md=0.8, ~2 at md=0.9, ~6 at md=0.95, so the
    derived-default path REFUSES a non-default membrane rather than shipping a drifting
    K (an explicit max_goals remains the caller's own).
Also disclosed: within one integer-dwell plateau tau_d is quantized away (D never moves,
K is flat across tau_d 20..385 at the defaults); ACROSS plateaus tau_d does move K, and
its primary charge is the dwell/handoff gating itself plus the regime bound above.

THE FIVE GUARDS (FR-020):
  1. derived == held +/-1  -- behavioral capacity measured on the REAL executive with the
     cap lifted and commitment decay disabled (recurrence=1.0 isolates the memory from the
     commitment policy); over-capacity frames die by refresh starvation, not by the cap.
  2. monotone tau_f sweep -- K must rise with tau_f; the sweep is its own positive
     control (an instrument that cannot see K move cannot certify K).
  3. frozen-STP REFUSES -- a synapse with frozen dynamics reaches a refusal on BOTH
     paths (unbounded dwell / no finite retention horizon), never a number; the live
     mechanism NOT refusing is the same command's control.
  4. regime gate -- static: any Tsodyks-Markram parameterization in src/ with
     tau_f <= tau_d is a finding unless exempt (a depression-dominant pair cannot hold a
     memory; the constructor refusal is the runtime half). Exemptions below, each
     REQUIRED to bless a real site or the gate fails loudly.
  5. no theta/gamma as K -- static: no expression in src/ derives a capacity by dividing
     a theta-named quantity by a gamma-named one (either orientation).
     `faculties/theta_gamma_code` computes the RATIO FROM K (steps_per_theta =
     capacity * steps_per_gamma) -- that direction is lawful and is this gate's liveness
     control; the reverse is circular here and is the constructed red.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.faculties.wm_capacity  # (run by run_selfchecks.py)
"""

import ast
import math
import os

# distinct-from: faculties/prefrontal_executive.py -- the STORE (frames, rehearsal channel,
#   goal stack). This module is the store's METROLOGY: it measures the store's own
#   mechanism and derives what the store may claim as capacity. The store imports this
#   module (top-level, so the dormancy walk sees the edge); this module imports the store
#   only lazily inside its own self-check, so there is no cycle. Site read.
# distinct-from: faculties/theta_gamma_code.py -- ORDERED WM by oscillatory phase whose
#   theta/gamma ratio is derived FROM a capacity argument; guard 5 exists precisely to
#   keep that arrow one-way. Site read.
# distinct-from: plasticity/short_term_plasticity.py -- owns the Tsodyks-Markram synapse
#   arithmetic this derivation exercises; imported by the store, measured through the
#   store's frames, never re-implemented here. Site read.


class CapacityRefusal(ValueError):
    """The mechanism cannot support a capacity derivation; no K is emitted (guard 3)."""


# distinct-from: instruments/ignition_instrument.py -- GWT ignition (workspace broadcast
#   threshold sweep over ensembles); this is the WM rehearsal pool's facilitation floor,
#   measured through _RehearsalFrame.ignites. Different substrate, same word. Site read.
# distinct-from: faculties/access_consciousness.py -- workspace ignition competition, not a
#   synaptic retention floor. Site read.
def ignition_floor(frame_factory):
    """The minimal facilitation that ignites a FRESH pool at full resources -- the derived
    retention floor, binary-searched over the mechanism's own ignites()."""
    probe = frame_factory()
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if probe.ignites(mid):
            hi = mid
        else:
            lo = mid
    return hi


def seed_viability(frame_factory, seed_u):
    """(u_ign, viable): can a frame seeded at `seed_u` survive to its first maintain tick?

    The store seeds every new frame's facilitation at U; the retention cull removes any
    frame whose decayed u cannot ignite. A seed at or below the floor means EVERY committed
    goal dies on its first tick with the LIF pool still firing normally -- the panel's
    blocker: derive_capacity emitted a confident K for a store whose behavioral capacity
    was ZERO (U=0.1: derived 68, held 0; leak=0.3 at default U: derived 80, held 0).
    Viability is checked at one tick of decay, the earliest cull point."""
    probe = frame_factory()
    u_ign = ignition_floor(frame_factory)
    probe.syn.u = float(seed_u)
    probe.last = 0.0
    u_at_first_tick, _r = probe.decayed(1.0)
    return u_ign, u_at_first_tick > u_ign


def derive_capacity(frame_factory, tau_f, dwell_cap=5000, seed_u=None):
    """K = T_max/t_s measured on the rehearsal-frame mechanism itself.

    `frame_factory()` must return a fresh frame exposing the store's own arithmetic:
      .syn                -- the Tsodyks-Markram synapse (u, R, U, tau_f, tau_d)
      .decayed(now)       -- (u, R) decayed to `now` through the store's elapsed-time read
      .serve(now)         -- one rehearsal tick: 'fired' | 'depleted' | 'dead'
      .ignites(u)         -- does a FRESH pool ignite at facilitation u, full resources
      .last               -- last-event timestamp (writable)

    REFUSES (CapacityRefusal) rather than emitting a number when: tau_f is non-finite;
    the pool cannot ignite at its own ignition floor; the dwell never ends (resources do
    not gate the channel -- e.g. frozen depression); or retention shows no finite horizon
    (e.g. frozen facilitation). A frozen-STP twin MUST land in one of these refusals --
    that reachability is guard 3's assert, not a hope."""
    if not (isinstance(tau_f, (int, float)) and math.isfinite(tau_f) and tau_f > 0):
        raise CapacityRefusal(f"non-finite or non-positive retention constant tau_f={tau_f!r}")
    probe = frame_factory()
    # u_ign: minimal facilitation igniting a fresh pool at full resources (derived floor).
    u_ign = ignition_floor(frame_factory)
    if not probe.ignites(u_ign):
        raise CapacityRefusal("pool cannot ignite at its own ignition floor (no readable memory)")
    # THE PANEL'S BLOCKER, closed here: a K for a store that seeds its frames BELOW the
    # floor describes a store that cannot hold even one item -- refuse, never number.
    if seed_u is not None:
        floor, viable = seed_viability(frame_factory, seed_u)
        if not viable:
            raise CapacityRefusal(
                f"the store seeds frames at u={float(seed_u):.4f}, at or below its own "
                f"ignition floor {floor:.4f} -- every committed goal dies on its first "
                f"tick and the behavioral capacity is ZERO; no K is emitted (raise U, or "
                f"lower the floor via leak/gain)"
            )
    # steady refresh from the floor: dwell D and the restored facilitation u_hi. The first
    # serve is at gap 0 (the frame sits exactly at its floor), later events at gap 1.
    f = frame_factory()
    f.syn.u = u_ign
    now = 1.0
    f.last = now
    D = 0
    while True:
        if f.serve(now) != "fired":
            break
        D += 1
        now += 1.0
        if D > dwell_cap:
            raise CapacityRefusal(
                f"dwell unbounded after {dwell_cap} fired ticks -- resources never gate the "
                f"rehearsal channel (frozen depression, or a neutered R route)"
            )
    if D == 0:
        raise CapacityRefusal("channel cannot ignite at its own ignition point (gain below cliff)")
    u_hi = f.syn.u
    t_s = D + 1
    # T_max: largest silent gap after which u_hi still ignites -- read THROUGH the frame's
    # decayed(). PRECISION, per the panel: the retention both the probe AND the live store
    # read lives in the FRAME's elapsed-time bookkeeping (closed-form decay from
    # syn.tau_f), not inside DynamicSynapse.spike -- guard 3's twins freeze exactly that
    # shared read. A freeze inside spike() alone (deleting its in-event u decay) moves
    # u_hi by ~0.4% and K not at all, IN THE STORE AS WELL AS HERE, so emitting the same
    # K for it is the probe agreeing with the mechanism, not missing a freeze.
    cap = 50.0 * float(tau_f)

    def _fires_after(gap):
        p = frame_factory()
        p.syn.u = u_hi
        p.last = 0.0
        u, _r = p.decayed(gap)
        return p.ignites(u)

    if _fires_after(cap):
        raise CapacityRefusal(
            f"retention shows no finite horizon (still ignites after a {cap:.0f}-tick silence) "
            f"-- frozen facilitation cannot yield a capacity"
        )
    lo, hi = 0.0, cap
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _fires_after(mid):
            lo = mid
        else:
            hi = mid
    T_max = lo
    K = int((T_max + D - 1) // t_s)
    # THE DERIVATION'S OWN REGIME, refused outside it rather than silently overestimated.
    # T_max is measured at FULL resources, which assumes a frame's resources recover within
    # one refresh cycle (K*t_s >> tau_d). MEASURED violation: at tau_f=300, tau_d=200 the
    # cycle is ~2*tau_d, revisit-R is ~0.87, and the formula said K=97 while the store
    # behaviorally held 90 -- a +7% overestimate. Inside the regime (cycle >= 5*tau_d) the
    # match is exact (derived 97 == held 97 at tau_d=40; 258 == 258 at tau_f=800). The
    # store's default constants (tau_f=1500, tau_d=200: cycle ~9.6*tau_d) are inside it.
    tau_d = float(probe.syn.tau_d)
    if K * t_s < 5.0 * tau_d:
        raise CapacityRefusal(
            f"derivation regime violated: the refresh cycle K*t_s = {K * t_s} is under "
            f"5*tau_d = {5.0 * tau_d:.0f}, so resources do NOT recover between a frame's "
            f"services and the full-recovery T_max overestimates K (measured +7% at "
            f"cycle ~2*tau_d). No K is emitted outside the measured-valid regime."
        )
    return {
        "u_ign": u_ign,
        "u_hi": u_hi,
        "D": D,
        "t_s": t_s,
        "T_max": T_max,
        "K": K,
    }


# --------------------------------------------------------------------------------------------------
# Guard 4 -- the tau_f > tau_d regime gate (static). A Tsodyks-Markram pair with
# tau_f <= tau_d is depression-dominant: facilitation dies before resources recover and
# nothing holds a memory. The runtime half is the store constructor's refusal; this half
# sweeps src/ so a NEW depression-dominant pair cannot land silently. Exemptions name the
# two deliberate onset DETECTORS -- depression-dominance is their mechanism, not a defect
# -- each quoting the site's own docstring verbatim, and each REQUIRED to bless a real
# finding (an exemption that blesses nothing fails loudly below).
# --------------------------------------------------------------------------------------------------
REGIME_EXEMPT = {
    ("short_term_plasticity.py", "novelty_filter"): (
        "onset/novelty filter, depression-dominant BY DESIGN: 'a DEPRESSING synapse turns one "
        "into an onset detector for free -- each spike transiently depletes release resources, "
        "so a burst of closely-spaced spikes fades while a spike arriving after a gap (an "
        "onset) rings out at full strength' (the module's own docstring, verbatim). Not a WM "
        "store; restored per T056's row."
    ),
    ("homeostatic_plasticity_stack.py", "_ShortTermGain.__init__"): (
        "PRE-phase dynamic gain, the same deliberate depression-dominant family: 'one "
        "depressing synapse per neuron turns the stream of deposits to that neuron into an "
        "onset detector' (the class docstring, verbatim). Found by running this gate against "
        "the tree BEFORE the rule landed (the gate-before-rule invariant); not a WM store."
    ),
}


def _src_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _iter_sources(root=None, source_by_name=None):
    """Yield (basename, source) for every governed src/ module, or for injected sources
    (the constructed-red path)."""
    if source_by_name is not None:
        yield from source_by_name.items()
        return
    root = root or _src_root()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("reasoning", "__pycache__")]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                    yield fn, fh.read()


def _tau_pairs(tree):
    """(symbol, tau_f, tau_d) for every def whose signature defaults BOTH taus to numeric
    literals, and every call passing both as numeric keyword literals."""
    out = []

    def _num(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = node.args
            named = args.args[-len(args.defaults) :] if args.defaults else []
            got = {}
            for a, d in zip(named, args.defaults):
                if a.arg in ("tau_f", "tau_d"):
                    v = _num(d)
                    if v is not None:
                        got[a.arg] = v
            if len(got) == 2:
                owner = parents.get(node)
                sym = f"{owner.name}.{node.name}" if isinstance(owner, ast.ClassDef) else node.name
                out.append((sym, got["tau_f"], got["tau_d"]))
        elif isinstance(node, ast.Call):
            got = {}
            for kw in node.keywords:
                if kw.arg in ("tau_f", "tau_d"):
                    v = _num(kw.value)
                    if v is not None:
                        got[kw.arg] = v
            # positional literals: DynamicSynapse(U, tau_f, tau_d) is the tree's ONE
            # positional T-M idiom (the panel measured the keyword-only arm blind to it).
            # MERGED with the keyword arm, not gated behind it -- the fold's first form
            # required `not got` and exactly 3 positionals, so the MIXED spelling
            # DynamicSynapse(0.5, 100.0, tau_d=200.0) satisfied neither arm (fold panel).
            if getattr(node.func, "id", getattr(node.func, "attr", "")) == "DynamicSynapse":
                pos = [_num(a) for a in node.args]
                if len(pos) >= 2 and "tau_f" not in got and pos[1] is not None:
                    got["tau_f"] = pos[1]
                if len(pos) >= 3 and "tau_d" not in got and pos[2] is not None:
                    got["tau_d"] = pos[2]
            if len(got) == 2:
                # Calls inside a self-check main() are out of scope: that is where the
                # constructor-refusal PROBES live (a tau_f<=tau_d call built to be REFUSED
                # at runtime is the regime rule's other half being tested, not a store).
                # Signature DEFAULTS in main() stay in scope.
                anc = parents.get(node)
                in_main = False
                while anc is not None:
                    if isinstance(anc, ast.FunctionDef) and anc.name == "main":
                        in_main = True
                        break
                    anc = parents.get(anc)
                if not in_main:
                    name = getattr(node.func, "id", getattr(node.func, "attr", "<call>"))
                    out.append((f"call:{name}", got["tau_f"], got["tau_d"]))
    return out


def regime_findings(root=None, source_by_name=None):
    """(file, symbol, tau_f, tau_d) for every statically resolvable T-M pair with
    tau_f <= tau_d that no exemption blesses. Non-literal parameterizations are OUTSIDE
    this gate's sight and are reported as such by main(), never counted as clean."""
    findings, blessed = [], set()
    for fn, src in _iter_sources(root, source_by_name):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for sym, tf, td in _tau_pairs(tree):
            if tf <= td:
                key = (fn, sym)
                if key in REGIME_EXEMPT:
                    blessed.add(key)
                else:
                    findings.append((fn, sym, tf, td))
    return findings, blessed


# --------------------------------------------------------------------------------------------------
# Guard 5 -- no theta/gamma ratio as a K source (static, permanent). theta_gamma_code
# derives its ratio FROM capacity; the reverse read is circular in this tree and FR-020
# forbids it outright.
# --------------------------------------------------------------------------------------------------
def _name_of(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def theta_gamma_findings(root=None, source_by_name=None):
    """Every division (either orientation) between a theta-named and a gamma-named
    quantity in src/. Census is asserted 0; the lawful direction (multiplication in
    theta_gamma_code) doubles as the scanner's liveness control in main()."""
    findings = []
    mult_control = 0
    for fn, src in _iter_sources(root, source_by_name):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                ln = _name_of(node.left).lower()
                rn = _name_of(node.right).lower()
                crossed = ("theta" in ln and "gamma" in rn) or ("gamma" in ln and "theta" in rn)
                if isinstance(node.op, (ast.Div, ast.FloorDiv)) and crossed:
                    findings.append(
                        (fn, node.lineno, f"{_name_of(node.left)} / {_name_of(node.right)}")
                    )
                if isinstance(node.op, ast.Mult) and fn == "theta_gamma_code.py":
                    if "gamma" in ln or "gamma" in rn or "theta" in ln or "theta" in rn:
                        mult_control += 1
    return findings, mult_control


def main():
    from snn_data_lake.faculties.prefrontal_executive import PrefrontalExecutive

    line = "=" * 92
    print(line)
    print("faculties/wm_capacity -- T056: K = T_max/t_s derived from the store's own constants")
    print(line)

    TAU_F, TAU_D = (
        300.0,
        40.0,
    )  # fast constants inside BOTH regimes (tau_f > tau_d; cycle >= 5*tau_d)

    def _factory(tau_f=TAU_F, tau_d=TAU_D):
        return PrefrontalExecutive(
            n_options=4, commit_races=1, seed=0, tau_f=tau_f, tau_d=tau_d
        )._probe_frame()

    # ---- guard 2: monotone tau_f sweep, the positive control first -- if K cannot be seen
    # to move, no derived number below means anything.
    sweep = {}
    for tf in (300.0, 800.0, 1500.0):
        sweep[tf] = derive_capacity(lambda tf=tf: _factory(tau_f=tf, tau_d=TAU_D), tf)
    ks = [sweep[tf]["K"] for tf in (300.0, 800.0, 1500.0)]
    assert ks[0] < ks[1] < ks[2], (
        f"K must rise with tau_f -- the capacity is DERIVED from the retention constant, and a "
        f"sweep that cannot move it says the derivation reads something else ({ks})"
    )

    # ---- guard 1: derived == held +/-1, on the REAL executive. recurrence=1.0 disables the
    # commitment-decay policy so the measurement isolates the MEMORY (otherwise the stack
    # drains from the top at the recurrence clock regardless of retention); the cap is
    # lifted so the DYNAMICS do the limiting, not the number under test.
    terms = sweep[TAU_F]
    derived_store = terms["K"] + 1  # the active frame rides its own per-tick refresh
    ex = PrefrontalExecutive(
        n_options=4,
        commit_races=1,
        seed=0,
        tau_f=TAU_F,
        tau_d=TAU_D,
        recurrence=1.0,
        max_goals=10**9,
    )
    n_push = derived_store + 2
    ev = [[1.0 if j == (i % 4) else 0.0 for j in range(4)] for i in range(4)]
    for i in range(n_push):
        ex.push_subgoal(ev[i % 4])
        for _ in range(terms["D"] + 1):  # attend the new goal to facilitation equilibrium
            ex.maintain()
    horizon = int(4 * n_push * terms["t_s"] + 6 * TAU_F)  # past convergence (measured ~1600 ticks)
    for _ in range(horizon):
        ex.maintain()
    held = len(ex.goals)
    assert abs(derived_store - held) <= 1, (
        f"THE DERIVATION'S OWN RED: derived K ({derived_store}) must match the behaviorally "
        f"held capacity ({held}) within +/-1 -- pushed {n_push}, over-capacity frames must "
        f"die by refresh starvation at exactly the derived boundary"
    )
    assert held < n_push, (
        f"the over-capacity pushes must actually die ({held} of {n_push} held) -- if all "
        f"survive, the boundary was never reached and the +/-1 match above was vacuous"
    )

    # ---- guard 3: frozen STP refuses, on BOTH paths, with the live mechanism as the
    # same-command control (it derived above without refusing).
    class _FrozenRetention:
        """Facilitation frozen: decayed() returns the written u forever."""

        def __init__(self, inner):
            self._f = inner
            self.syn = inner.syn
            self.last = 0.0

        def decayed(self, now):
            return self.syn.u, 1.0

        def ignites(self, u):
            return self._f.ignites(u)

        def serve(self, now):
            # resources still gate (R depletes), so this twin passes the dwell and must be
            # caught by the retention-horizon refusal.
            u, r = self.syn.u, self.syn.R
            if not self._f.ignites(u * r):
                return "depleted" if self._f.ignites(u) else "dead"
            self.syn.spike(1.0)
            self.syn.u += self.syn.U * (1.0 - self.syn.u)  # refresh writes; decayed() never decays
            return "fired"

    # distinct-from: plasticity/plasticity_rule_stack.py and reasoning/*_lm_serve.py -- those
    #   "frozen" symbols are trained-artifact writers/sandwich checks; this is guard 3's
    #   in-command STP-frozen TWIN (a control, not a mechanism). Sites read.
    class _FrozenDepression:
        """Resources frozen: serve always finds full R -- the dwell never ends. `last` is
        DELEGATED to the wrapped frame (the first build kept its own copy, the harness
        wrote the wrapper's while serve read the inner's, and the phantom 1-tick gap
        landed the twin in the wrong refusal)."""

        def __init__(self, inner):
            self._f = inner
            self.syn = inner.syn

        @property
        def last(self):
            return self._f.last

        @last.setter
        def last(self, value):
            self._f.last = value

        def decayed(self, now):
            return self._f.decayed(now)

        def ignites(self, u):
            return self._f.ignites(u)

        def serve(self, now):
            self.syn.R = 1.0
            return self._f.serve(now)

    refusals = []
    for twin, expect in (
        (_FrozenRetention, "no finite horizon"),
        (_FrozenDepression, "dwell unbounded"),
    ):
        try:
            derive_capacity(lambda t=twin: t(_factory()), TAU_F)
            raise AssertionError(f"frozen STP ({twin.__name__}) must REFUSE, not emit a K")
        except CapacityRefusal as e:
            assert expect in str(e), f"{twin.__name__}: wrong refusal ({e})"
            refusals.append(twin.__name__)
    try:
        derive_capacity(_factory, float("inf"))
        raise AssertionError("a non-finite tau_f must refuse")
    except CapacityRefusal as e:
        assert "non-finite" in str(e)

    # ---- guard 4: the regime gate, constructed red first, then the tree census.
    red_src = {
        "candidate_wm.py": '"""A working-memory candidate."""\n'
        "def build(tau_f=100.0, tau_d=200.0):\n    return (tau_f, tau_d)\n"
    }
    red, _ = regime_findings(source_by_name=red_src)
    assert len(red) == 1 and red[0][2] == 100.0, (
        f"the regime gate must RED a constructed tau_f<=tau_d WM candidate ({red})"
    )
    pos_red, _ = regime_findings(
        source_by_name={"pos.py": "def build():\n    return DynamicSynapse(0.5, 100.0, 200.0)\n"}
    )
    assert len(pos_red) == 1, (
        f"the gate must RED a POSITIONAL DynamicSynapse literal -- the tree's one positional "
        f"T-M idiom was invisible to the keyword-only arm ({pos_red})"
    )
    mixed_red, _ = regime_findings(
        source_by_name={"mix.py": "x = DynamicSynapse(0.5, 100.0, tau_d=200.0)\n"}
    )
    assert len(mixed_red) == 1, (
        f"the MIXED positional/keyword spelling must red too -- the fold's first arm let it "
        f"satisfy neither test (fold panel) ({mixed_red})"
    )
    fixed, _ = regime_findings(
        source_by_name={"candidate_wm.py": red_src["candidate_wm.py"].replace("100.0", "400.0")}
    )
    assert fixed == [], f"the fixed candidate must clear the gate ({fixed})"
    tree_findings, blessed = regime_findings()
    assert tree_findings == [], (
        f"NEW depression-dominant T-M pairs in src/ with no exemption: {tree_findings} -- a "
        f"tau_f <= tau_d pair cannot hold a memory; exempt it with its reason or fix the regime"
    )
    missing = set(REGIME_EXEMPT) - blessed
    assert not missing, (
        f"exemptions blessing NOTHING: {sorted(missing)} -- the site moved or was deleted; an "
        f"exemption that blesses nothing hides the next real finding"
    )

    # ---- guard 5: theta/gamma-as-K, constructed red + census + liveness control.
    tg_red, _ = theta_gamma_findings(
        source_by_name={"bad.py": "K = steps_per_theta / steps_per_gamma\n"}
    )
    assert len(tg_red) == 1, f"the theta/gamma gate must RED a constructed ratio-read ({tg_red})"
    tg_floor, _ = theta_gamma_findings(
        source_by_name={"bad3.py": "K = steps_per_theta // steps_per_gamma\n"}
    )
    assert len(tg_floor) == 1, (
        f"floor division is a capacity read too -- an integer K is the LIKELIER spelling "
        f"({tg_floor})"
    )
    tg, mult_control = theta_gamma_findings()
    assert tg == [], (
        f"src/ reads a theta/gamma ratio where a capacity belongs: {tg} -- circular here "
        f"(theta_gamma_code derives the ratio FROM K), forbidden by FR-020"
    )
    assert mult_control >= 1, (
        "liveness control failed: the scanner did not see theta_gamma_code's lawful "
        "capacity * steps_per_gamma multiplication -- an empty census from a scanner that "
        "never reached the one relevant file is a dead detector, not a clean tree"
    )

    print(
        f"self-check OK: K DERIVED from the store's own constants and MATCHED behaviorally -- "
        f"tau_f={TAU_F}: t_s={terms['t_s']} (dwell {terms['D']} resource-gated ticks + handoff), "
        f"T_max={terms['T_max']:.0f}, K={terms['K']}+1 active = {derived_store} derived vs "
        f"{held} held of {n_push} pushed (+/-1; over-capacity frames died by starvation); "
        f"monotone sweep K={ks} over tau_f=(300, 800, 1500) is the positive control; frozen "
        f"STP REFUSES both ways ({', '.join(refusals)}) while the live mechanism derives; "
        f"regime gate: constructed red fired, tree census 0 with {len(REGIME_EXEMPT)} "
        f"exemptions each blessing a real site; theta/gamma-as-K: constructed red fired, "
        f"census 0, lawful-direction control seen ({mult_control} mult). STATIC-GATE SCOPE, "
        f"said rather than counted as clean: non-literal T-M parameterizations (variables "
        f"into DynamicSynapse) are outside both gates' sight; the walk excludes reasoning/ "
        f"(ungoverned tier, ~90 modules unscanned); call-keyword pairs under ANY function "
        f"named main() are carved out (where constructor-refusal probes live -- a REAL "
        f"store built inside a main() would ride that carve-out); exemption keys are "
        f"(basename, symbol), so a future same-named file could inherit one; and the "
        f"theta/gamma match is a name-substring heuristic (a ratio through an intermediate "
        f"variable is invisible, an unrelated theta-by-gamma division would false-red)."
    )


if __name__ == "__main__":
    main()
