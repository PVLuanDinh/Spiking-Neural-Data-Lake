"""ESPP -- the contrastive temporal-stream element of the plasticity palette
(specs/022 T052-i; FR-019's "ESPP for temporal streams (same/different oracle named)").

WHAT THIS IS, honestly: an EXTRACTION, not an invention and not a re-implementation.
The ESPP mechanism (contrastive local predictive plasticity -- Illing, Ventura,
Bellec & Gerstner, "Local plasticity rules can learn deep representations using
self-supervised contrastive predictions", 2021) ALREADY LIVED in this tree as
spiking_jepa's selectable `rule="espp"` branch (the contrastive variant of its
`_modulator`). P2 ADOPT: `espp_modulator_core` below IS that math, moved here, and
the jepa branch now DELEGATES to it -- the inline expression left
cerebellum/spiking_jepa.py in the same commit, so exactly ONE mechanism survives.
A fresh implementation beside the living branch was REFUSED on the T049 precedent:
the standalone lpl_rule.py built beside jepa's living lpl branch was panel-measured
DEGENERATE and deleted per P8(5), and the palette extraction was dated to the T051
composer interface -- which has since landed (rule_composer + eligibility_algebra),
so this extraction rides that interface rather than preceding it.

THE MODULATOR (per-post m_j; jepa applies dw_ij = lr*m_j*sg_j*gate*e_i):

    m_j = -( lambda_pred*(2*(s_bar_j - echo_j) - 2*alpha*(s_bar_j - neg_j))
           + lambda_homeo*(mu_j - rho_star) )

pull-to-positive-echo / push-from-negative, plus firing-rate homeostasis.
`alpha` and `lambda_homeo` arrive PRE-GATED by the caller (jepa zeroes both under
anti_collapse=False -- its ablation control); the gating is instance policy, not
rule math, so it stays with the instance.

THE SAME/DIFFERENT ORACLE, NAMED (FR-019's clause, discharged here): for temporal
streams the positive/negative assignment is supervised by
`substrate/episode_driver.EpisodeDriver` -- each trial presents a SAMPLE then a
PROBE and `trial['oracle']` is 'same'/'different', ground truth BY CONSTRUCTION
(the T035 review proved the oracle cannot lie: the 'different' draw redraws until
distinct). 'same' trials supply the pull target (echo), 'different' trials supply
the push negative (neg). Inside jepa's own loop the negative comes instead from
its `_neg` ring buffer (temporally-adjacent activity) -- the UNSUPERVISED negative
source; the oracle is the supervisory source the temporal-stream adoption
(T052-ii, the SHD/N-MNIST benchmark vs supervised e-prop) drives through this same
core. Two negative SOURCES, one RULE -- the source is a state quantity the caller
owns, exactly like every other factor here.

DECLARATION SCOPE (mirrors SpikingJEPA.lpl_synapse_rule, the T051b-i pattern):
`espp_synapse_rule(inst)` authors the per-synapse sum-of-products declaration for
an ESPP instance's UNCONDITIONAL bottom sites -- gate==1 (transport refused),
no lambda_jepa top-layer addition (top sites remain the disclosed remainder,
exactly as they are for the lpl declaration). Coefficients are READ from the
instance, never retyped (the T050 discipline; a retyped lambda once shipped a
100% error). Distributed to the synapse grain:

    dw = (-2*lambda_pred)       * d_pred  * lr * sg * e    d_pred  = s_bar_j - echo_j
       + (+2*lambda_pred*alpha) * d_neg   * lr * sg * e    d_neg   = s_bar_j - neg_j
       + (-lambda_homeo)        * d_homeo * lr * sg * e    d_homeo = mu_j - rho_star

Zero-coefficient terms are DROPPED at authoring (the ablation instance authors the
1-term form), mirroring the caller's own zeroing. The d_neg coefficient is a
PRODUCT of two instance coefficients (2*lambda_pred*alpha) -- pinned exactly by
the self-check, both factors.

HONEST RECORD, carried from the incumbent rather than smoothed away: at jepa's toy
scale (3 temporally-persistent shape identities) espp does NOT improve the
representation -- contrastive negatives are dominated by the SAME identity at a
different position -- which is WHY it is not jepa's default. That measurement is
about jepa's world, not about temporal streams; what adoption buys on the real
event packs (SHD/N-MNIST) is T052-ii's measurement, against the in-tree supervised
e-prop encoders, with the untrained-twin and random-encoder controls jepa's own
honest negatives demand. Nothing here claims espp beats anything, anywhere, yet.

DORMANCY, said where a reader looks: this module has ZERO production callers
(panel-measured: every `rule="espp"` construction tree-wide sits inside a
`__main__` -- jepa's branch is the delegation ROUTE, not a caller; consumers =
this module's self-check + jepa's self-check exercising the branch). jepa itself
is dormancy-EXEMPT `permanent` and espp is its non-default path, so even the
route is presence, not participation. The EXEMPT row carries T052-ii as the
dated consumer (the temporal-stream benchmark drives the core directly). Both
method-grade gates structurally miss module-level functions (no ClassDef scan
hit, no LEARN_VERB); this registered self-check is the charged surface.

WHY EXTRACTION HERE WHEN T051b-i REFUSED IT FOR LPL (the asymmetry, adjudicated):
i's record refused making jepa ADOPT the authored lpl declaration -- that would
have WIRED a new consumer into a permanently-off-path host and claimed a charge
that does not exist. This extraction claims NO charge and wires NO consumer: it
MOVES a function the production branch already computes and delegates the branch
to it -- byte-equal behavior, zero new reach, the P2 one-mechanism move only.
The charge claim stays exactly where it was: nowhere, until T052-ii measures.

# distinct-from: cerebellum/spiking_jepa.py -- the HOST this extracts FROM: its
#   `rule="espp"` branch now delegates to espp_modulator_core (one mechanism); its
#   lpl_synapse_rule is the SIBLING declaration for the rule="lpl" branch and
#   refuses espp instances (each branch declares itself, neither declares the
#   other). Read before writing here.
# distinct-from: plasticity/eligibility_algebra.py -- the authoring CONTRACT this
#   module composes THROUGH (espp_synapse_rule authors against it, the consumer
#   its T052 row promised); nothing here re-implements term algebra.
# distinct-from: cerebellum/jepa_world_model.py -- population-normalized VICReg
#   loss gradient at encoder-loss grain; coefficient-NAME overlap only (the same
#   adjacency lpl_synapse_rule records).
# distinct-from: plasticity/competitive_learning.py -- competitive/WTA locality,
#   no contrastive negative, no echo target; different rule family.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.plasticity.espp_rule
"""

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "espp_modulator_core": (
        "LOCAL",
        "per-post contrastive modulator from layer-local quantities only (s_bar, echo, "
        "neg, mu) -- no error transport, no gradient chain; the extracted incumbent of "
        "spiking_jepa's rule='espp' branch, which delegates here",
    ),
    "espp_synapse_rule": (
        "LOCAL",
        "authors the per-synapse sum-of-products declaration (three pre-collapsed "
        "deviation factors x lr*sg*e) for an ESPP instance through the T050 algebra; "
        "declares, never applies",
    ),
    "main": (
        "NOT_A_RULE",
        "the self-check harness; its T052-ii benchmark arm APPLIES the declared LOCAL "
        "espp update (dw = lr*m*sg*e with m from espp_modulator_core above) to measure "
        "adoption -- the rule's claim is the core's, the harness only drives it",
    ),
}


def espp_modulator_core(n, sb, echo, neg, mu, lambda_pred, alpha, lambda_homeo, rho_star):
    """THE espp modulator -- the ONE mechanism (P2; spiking_jepa's rule='espp' branch
    delegates here). Expression order is the incumbent's, verbatim. WHAT WARRANTS
    THAT, said precisely (panel F4): faithfulness to the pre-extraction source was
    verified against `git show HEAD:.../spiking_jepa.py` at landing (a record, not a
    runtime assert); the self-check's frozen-transcription arm is a FORWARD
    REGRESSION GUARD from that landing on (a mis-transcription made once would sit
    in both copies -- the arm cannot see it); and the DELTA arm independently binds
    this core to the weight production ACTUALLY applies, through the separately
    authored declaration. `alpha`/`lambda_homeo` arrive pre-gated by the caller."""
    m = [0.0] * n
    for j in range(n):
        contr = lambda_pred * (2.0 * (sb[j] - echo[j]) - 2.0 * alpha * (sb[j] - neg[j]))
        homeo = lambda_homeo * (mu[j] - rho_star)
        m[j] = -(contr + homeo)
    return m


def espp_synapse_rule(inst):
    """The AUTHORED per-synapse ESPP rule for `inst` (a SpikingJEPA-shaped instance,
    duck-typed -- no cerebellum import): coefficients READ from the instance at
    authoring time. Refuses non-espp instances and the transport/freeze_top controls
    (their production dw is gate-zeroed or top-skipped; an unconditional declaration
    would not describe them) -- the exact scoping lpl_synapse_rule applies to its own
    branch. Factor semantics (state quantities the caller owns per site): d_pred,
    d_neg, d_homeo are pre-collapsed deviations; lr/sg/e the standard triple. The
    TRAILING (sg, e) factor order matches the incumbent chain ((lr*m)*sg)*gate*e
    (the T051b-i factor-order measurement). Returns the algebra-authored rule
    ('jepa.espp.synapse'); always a FRESH algebra (the caller-supplied-algebra
    parameter was deleted at T051b-i for corrupting on refusal paths)."""
    rule = getattr(inst, "rule", None)
    if rule != "espp":
        raise ValueError(f"espp_synapse_rule declares the rule='espp' branch only (rule={rule!r})")
    if getattr(inst, "transport", False) or getattr(inst, "freeze_top", False):
        raise ValueError(
            "espp_synapse_rule refuses transport/freeze_top control instances: their "
            "production dw is gate-zeroed or top-skipped, so the unconditional "
            "declaration would not describe them"
        )
    from snn_data_lake.plasticity.eligibility_algebra import EligibilityAlgebra

    lp = float(inst.lambda_pred)
    alpha = float(inst.espp_alpha) if inst.anti_collapse else 0.0
    lh = float(inst.lambda_homeo) if inst.anti_collapse else 0.0
    terms = [(-(lp * 2.0), ("d_pred", "lr", "sg", "e"))]
    if alpha != 0.0:
        terms.append((lp * 2.0 * alpha, ("d_neg", "lr", "sg", "e")))
    if lh != 0.0:
        terms.append((-lh, ("d_homeo", "lr", "sg", "e")))
    roles = {
        "d_pred": "positive-pair pull deviation s_bar_j - echo_j (pre-collapsed)",
        "d_neg": (
            "contrastive push deviation s_bar_j - neg_j (pre-collapsed); neg's source is "
            "the caller's -- jepa's _neg ring buffer unsupervised, or EpisodeDriver's "
            "same/different oracle supervised (the FR-019 naming; trial['oracle'] is "
            "truth by construction)"
        ),
        "d_homeo": "homeostatic deviation mu_j - rho_star (pre-collapsed)",
        "lr": "learning rate, read from the instance per step",
        "sg": "post surrogate-gradient sensitivity sg_j",
        "e": "presynaptic eligibility trace (the STDP window as state)",
    }
    alg = EligibilityAlgebra(term_budget=len(terms))
    used = []
    for _c, fs in terms:  # declare exactly what the built terms use, in first-use order
        for f in fs:
            if f not in used:
                used.append(f)
    for f in used:
        alg.declare_factor(f, roles[f])
    return alg.author("jepa.espp.synapse", terms)


# ---- self-check ----------------------------------------------------------------------------
def main():
    import sys as _sys

    from snn_data_lake.cerebellum.spiking_jepa import SpikingJEPA
    from snn_data_lake.plasticity.eligibility_algebra import EligibilityAlgebra, equivalence_check

    line = "=" * 96
    print(line)
    print("espp_rule -- specs/022 T052-i: the extracted ESPP palette element (P2: jepa delegates)")
    print(line)

    _ACTIONS = ("L", "R", "H")
    _R = 12
    import random as _random

    _srng = _random.Random(20260826)

    def _stream(n):
        # SYNTHETIC in-box stream (the T050 fixture lesson: synthetic on purpose --
        # this harness measures rule/declaration identities, not representation quality).
        out = []
        prev = [_srng.random() for _ in range(_R)]
        for _i in range(n):
            cur = [min(1.0, max(0.0, v + _srng.uniform(-0.35, 0.35))) for v in prev]
            out.append((prev, _ACTIONS[_srng.randrange(3)], cur))
            prev = cur
        return out

    # (1) DELEGATION IS LIVE, not a copy (presence-is-not-participation): rebinding the
    # core through the module namespace moves production output -- proving jepa's espp
    # branch actually routes through THIS module -- then the identity guard below proves
    # the routed math is the incumbent's.
    import snn_data_lake.cerebellum.spiking_jepa as _jepa_mod
    import snn_data_lake.plasticity.espp_rule as _self_mod

    _probe = SpikingJEPA(_ACTIONS, retina=_R, seed=0, rule="espp")
    for x, a, y in _stream(6):
        _probe.train_step(x, a, y)
    _sb = [0.3] * _probe.sizes[0]
    _echo = [0.1] * _probe.sizes[0]
    _m_live = _probe._modulator(0, _sb, _echo)
    # THE DECISIVE CHECK (panel F1): the rebind arm alone is satisfiable by a PRIVATE
    # COPY in jepa (a local `def espp_modulator_core` occupies the same global the
    # rebind writes) -- identity of the bound object is what excludes that.
    assert _jepa_mod.espp_modulator_core is _self_mod.espp_modulator_core, (
        "jepa's espp_modulator_core global must BE this module's function object -- "
        "a same-named private copy would satisfy the rebind arm below"
    )
    _orig_core = _self_mod.espp_modulator_core
    try:
        _self_mod.espp_modulator_core = lambda n, *a2, **k2: [7.0] * n
        _jepa_mod.espp_modulator_core = _self_mod.espp_modulator_core
        _m_neutered = _probe._modulator(0, _sb, _echo)
    finally:
        _self_mod.espp_modulator_core = _orig_core
        _jepa_mod.espp_modulator_core = _orig_core
    delegation_live = _m_neutered == [7.0] * _probe.sizes[0] and _m_live != _m_neutered
    # RESTORATION is asserted by IDENTITY, deterministically. The first draft compared
    # a post-restore _modulator() output to _m_live -- but the espp branch CONSUMES AN
    # RNG DRAW per call (_sample_neg), so that equality was a 7-in-40-seed coin flip
    # (adversarial panel, seed-swept): a latent false RED in a CORE self-check, green
    # at the shipped seed only by luck. The finally already restores; identity is the
    # functional guarantee.
    _restored_ok = (
        _jepa_mod.espp_modulator_core is _orig_core and _self_mod.espp_modulator_core is _orig_core
    )

    # (2) THE FROZEN PRE-EXTRACTION REFERENCE (the removed inline block, transcribed
    # verbatim -- comparing the core to itself would be an identity in disguise), BIT
    # tol=0 across in-box states, with the corrupted-production control IN THE SAME
    # COMMAND (a perturbed lambda must red the same comparison, proving the arm can
    # fail at all).
    def _pre_extraction_reference(n, sb, echo, neg, mu, lp, alpha, lh, rho_star):
        m = [0.0] * n
        for j in range(n):
            contr = lp * (2.0 * (sb[j] - echo[j]) - 2.0 * alpha * (sb[j] - neg[j]))
            homeo = lh * (mu[j] - rho_star)
            m[j] = -(contr + homeo)
        return m

    # THE ANCHOR (policy panel F-3): the transcription above is frozen against the
    # pre-extraction source at commit 954af8a1 (spiking_jepa's removed inline block,
    # verified by git show at landing). It is a co-edited twin by construction -- a
    # lockstep edit to both copies would silently vacate the tol=0 arm -- so the
    # PRODUCTION core's source is crc-pinned here: any edit to espp_modulator_core
    # reds THIS assert first, making the re-freeze a deliberate act (re-verify the
    # transcription against the anchored source, then update pin + anchor together).
    import inspect as _inspect
    import zlib as _zlib

    _core_src_crc = _zlib.crc32(_inspect.getsource(espp_modulator_core).encode())
    assert _core_src_crc == 0x4307C9FB, (
        f"espp_modulator_core's source changed (crc 0x{_core_src_crc:08x}): re-verify the "
        "frozen transcription against the anchored pre-extraction source, then re-freeze "
        "this pin deliberately (see the ANCHOR comment)"
    )

    _brng = _random.Random(7)
    bit_states = 0
    core_bit_ok = True
    control_red_states = 0  # per-state, matching the pass condition's grain (panel
    for _ in range(200):  # overflow: an OR-accumulated control against an AND pass
        n = _brng.randrange(2, 9)  # certifies only "can fail somewhere")
        args = [[_brng.uniform(-1.5, 1.5) for _ in range(n)] for _ in range(4)]
        lp2, a2v, lh2, rs2 = (
            _brng.uniform(0.1, 3.0),
            _brng.uniform(0.0, 2.0),
            _brng.uniform(0.0, 1.0),
            _brng.uniform(0.0, 0.5),
        )
        got = espp_modulator_core(n, *args, lp2, a2v, lh2, rs2)
        ref = _pre_extraction_reference(n, *args, lp2, a2v, lh2, rs2)
        if got != ref:
            core_bit_ok = False
        if espp_modulator_core(n, *args, lp2 + 1e-6, a2v, lh2, rs2) != ref:
            control_red_states += 1
        bit_states += 1
    control_red = control_red_states == bit_states  # continuous draws: lp shift moves every state

    # (3) EQUIVALENCE ARMS vs the APPLIED weight (the T051b-i three-arm pattern, scoped
    # to gate==1 bottom sites): a spy records the modulator inputs (including the neg
    # DRAW -- a state quantity, captured not re-derived) and the weights the production
    # loop actually wrote; the declaration must predict the applied dw.
    class _SpyESPP(SpikingJEPA):
        def __init__(self, *a2, **k2):
            super().__init__(*a2, **k2)
            self._rec_on = False
            self._cur = None
            self._spied = []
            self._last_neg = {}

        def _sample_neg(self, l):
            v = super()._sample_neg(l)
            self._last_neg[l] = v[:]
            return v

        # distinct-from: cerebellum/spiking_jepa.py -- this override is the harness SPY
        #   wrapper around the base method (super() call; records weights before/after),
        #   not a second update mechanism; the lpl harness's _SpyJEPA is its precedent.
        def train_step(self, x, a2, y):
            if self._rec_on:
                self._cur = {"recs": [], "mod": {}}
            out = super().train_step(x, a2, y)
            if self._cur is not None:
                self._cur["wff_after"] = [row[:] for row in self.Wff[0]]
                self._cur["wrec_after"] = [row[:] for row in self.Wrec[0]]
                self._spied.append(self._cur)
                self._cur = None
            return out

        def _forward(self, pattern, graded=None):  # graded=None: DELEGATE to self.graded
            rec = super()._forward(pattern, graded=graded)
            if self._cur is not None and len(self._cur["recs"]) < 2:
                self._cur["recs"].append(rec)
            return rec

        def _modulator(self, l, sb, echo):
            m = super()._modulator(l, sb, echo)
            if self._cur is not None and l == 0:
                self._cur["wff_before"] = [row[:] for row in self.Wff[0]]
                self._cur["wrec_before"] = [row[:] for row in self.Wrec[0]]
                self._cur["mod"][0] = {
                    "sb": sb[:],
                    "echo": echo[:],
                    "mu": self.mu[0][:],
                    "neg": self._last_neg[0][:],
                    "m": m[:],
                }
            return m

    def _espp_sites(spy):
        """(cases, chain-incumbents, observed applied deltas) from spied BOTTOM ff
        sites -- gate==1 exactly, no lambda_jepa addition (top sites are the disclosed
        remainder). Chain incumbent = the production association lr*m*sg*1.0*e from
        the RECORDED modulator (the BIT/FULL arms' target -- the applied delta carries
        the `w +=` rounding and belongs to the DELTA arm's tolerance, the exact
        conflation this harness's own first draft shipped). Factors go through the
        BASE-CLASS _elig/_surrogate explicitly, the declared semantics."""
        assert spy.transport is False
        cases, incs, deltas = [], [], []
        off_sampled = 0
        for st in spy._spied:
            rec_x = st["recs"][0]
            mod = st["mod"].get(0)
            if mod is None:
                continue
            n = spy.sizes[0]
            sg = SpikingJEPA._surrogate(spy, rec_x["vpre"][0], n)
            e_ff = SpikingJEPA._elig(spy, rec_x["ev"], 2 * spy.R)
            e_rec = SpikingJEPA._elig(spy, rec_x["spk"][0], n)
            sb, echo, mu, neg = mod["sb"], mod["echo"], mod["mu"], mod["neg"]
            for j in range(n):
                m_j = mod["m"][j]
                # BOTH halves of the bottom write loop (adversarial panel: the first
                # draft read Wff only -- a 3x recurrent-write defect passed all arms)
                # and BOTH event polarities (sorted[:3] had sampled the ON half almost
                # exclusively: 71/1 ON/OFF; sorted[-1:] reaches the OFF lines >= R).
                _ffs = sorted(spy.conn_ff[0][j])
                sites = [("ff", i) for i in _ffs[:2]] + [("ff", i) for i in _ffs[-1:]]
                sites += [("rec", k) for k in sorted(spy.conn_rec[0][j])[:1]]
                for kind, i in sites:
                    e_val = e_ff[i] if kind == "ff" else e_rec[i]
                    if kind == "ff" and i >= spy.R:
                        off_sampled += 1
                    case = {"d_pred": sb[j] - echo[j], "lr": spy.lr, "sg": sg[j], "e": e_val}
                    if spy.anti_collapse:
                        case["d_neg"] = sb[j] - neg[j]
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
        return cases, incs, deltas, off_sampled

    def _spy_run(cls=_SpyESPP, steps=60, record=4, **kw):
        spy = cls(_ACTIONS, retina=_R, seed=0, rule="espp", **kw)
        data = _stream(steps)
        for x, a2, y in data[:-record]:
            spy.train_step(x, a2, y)
        spy._rec_on = True
        for x, a2, y in data[-record:]:
            spy.train_step(x, a2, y)
        return spy

    def _term_mag(rule, case):
        s = 0.0
        for _c2, _fs2 in rule.terms:
            p = abs(_c2)
            for _f2 in _fs2:
                p *= abs(float(case[_f2]))
            s += p
        return s

    def _k_ulp(rule):
        # one ulp per rounding in the longest re-association chain: every factor
        # multiply + every term add + the incumbent's own factored-contr composition
        # (the espp incumbent computes lp*(2*dp - 2*a*dn) as ONE factored expression,
        # two extra roundings vs the distributed two-term form).
        return float(sum(len(_fs2) for _c2, _fs2 in rule.terms) + len(rule.terms) + 2)

    # BIT arm: the ablation instance (anti_collapse=False -> single pred term; lp*2.0
    # and 2.0*dp are exact so the factored and distributed forms coincide bit-for-bit),
    # against the CHAIN incumbent, tol=0.
    spy_bit = _spy_run(anti_collapse=False)
    rule_bit = espp_synapse_rule(spy_bit)
    cases_b, incs_b, _deltas_b, _off_b = _espp_sites(spy_bit)
    live_b = sum(1 for v in incs_b if v != 0.0)
    _inc_map_b = {id(c): v for c, v in zip(cases_b, incs_b)}  # id-keyed; cases_b keeps refs alive
    worst_bit = equivalence_check(rule_bit, lambda c: _inc_map_b[id(c)], cases_b, tol=0.0)
    bit_ok = worst_bit == 0.0

    # FULL arm (3 terms) against the CHAIN incumbent: tol DERIVED from the authored
    # structure (K ulps of the max term magnitude), never chosen.
    spy_full = _spy_run()
    rule_full = espp_synapse_rule(spy_full)
    cases_f, incs_f, deltas_f, off_f = _espp_sites(spy_full)
    live_f = sum(1 for v in incs_f if v != 0.0)
    _inc_map_f = {id(c): v for c, v in zip(cases_f, incs_f)}
    _mag = max(_term_mag(rule_full, c) for c in cases_f)
    tol_full = _k_ulp(rule_full) * _sys.float_info.epsilon * _mag
    worst_full = equivalence_check(rule_full, lambda c: _inc_map_f[id(c)], cases_f, tol=tol_full)
    full_ok = worst_full <= tol_full

    # DELTA arm: the declaration vs the weight the production loop ACTUALLY wrote
    # (the chain arms are blind to an application defect); tol adds the += rounding
    # scale (|w| + |terms|).
    _magd = max(abs(w0) + _term_mag(rule_full, c) for c, (_d, w0) in zip(cases_f, deltas_f))
    tol_delta = _k_ulp(rule_full) * _sys.float_info.epsilon * _magd
    worst_delta = 0.0
    for case, (d, _w0) in zip(cases_f, deltas_f):
        dv = abs(rule_full.evaluate(**case) - d)
        if not (dv <= tol_delta):
            raise AssertionError(
                f"declaration diverges from the APPLIED weight delta: |d|={dv!r} > {tol_delta:.3e}"
            )
        worst_delta = max(worst_delta, dv)

    # THE NEGATIVE-HOMEO REGIME (adversarial panel MEDIUM: mu never fell below
    # rho_star=0.25 in-stream, so d_homeo was one-signed on every site -- the exact
    # one-sign under-coverage the algebra's own equivalence_check docstring warns
    # about). A spy with rho_star OVERRIDDEN above the running rates flips the sign
    # at every site; production (_modulator reads self.rho_star) and the authored
    # declaration (case d_homeo = mu - spy.rho_star) both read the instance, so the
    # arms stay coherent and now exercise d_homeo < 0.
    spy_nh = _SpyESPP(_ACTIONS, retina=_R, seed=0, rule="espp")
    _data_nh = _stream(36)
    for x2, a2, y2 in _data_nh[:30]:  # warm under the default; NOT recorded
        spy_nh.train_step(x2, a2, y2)
    spy_nh.rho_star = 0.9  # override BEFORE recording -- every spied step and every
    spy_nh._rec_on = True  # authored binding read the same value
    for x2, a2, y2 in _data_nh[30:]:
        spy_nh.train_step(x2, a2, y2)
    rule_nh = espp_synapse_rule(spy_nh)
    cases_nh, incs_nh, deltas_nh, _off_nh = _espp_sites(spy_nh)
    neg_homeo_sites = sum(1 for c in cases_nh if c.get("d_homeo", 0.0) < 0.0)
    _magd_nh = max(abs(w0) + _term_mag(rule_nh, c) for c, (_d, w0) in zip(cases_nh, deltas_nh))
    tol_nh = _k_ulp(rule_nh) * _sys.float_info.epsilon * _magd_nh
    worst_nh = 0.0
    for case, (d, _w0) in zip(cases_nh, deltas_nh):
        dv = abs(rule_nh.evaluate(**case) - d)
        worst_nh = max(worst_nh, dv)
    nh_ok = worst_nh <= tol_nh

    # the delta arm's reachable red, in-command: production _elig corrupted x1.25
    # while the declared factor semantics (base-class _elig) stay clean.
    class _CorruptElig(_SpyESPP):
        def _elig(self, train_sig, n):
            return [v * 1.25 for v in SpikingJEPA._elig(self, train_sig, n)]

    spy_c = _spy_run(cls=_CorruptElig)
    rule_c = espp_synapse_rule(spy_c)
    cases_c, _incs_c, deltas_c, _off_c = _espp_sites(spy_c)
    delta_control_red = False
    for case, (d, _w0) in zip(cases_c, deltas_c):
        if d != 0.0 and abs(rule_c.evaluate(**case) - d) > tol_delta:
            delta_control_red = True
            break

    # (4) COEFFICIENT PINS -- exact terms, all three lambdas INCLUDING the composed
    # d_neg product (2*lambda_pred*alpha, both factors), plus the mutated-instance
    # retype red (a retyped literal cannot track a mutation).
    _cmap = {fs[0]: cf for cf, fs in rule_full.terms}
    coeffs_pin = _cmap == {
        "d_pred": -(float(spy_full.lambda_pred) * 2.0),
        "d_neg": float(spy_full.lambda_pred) * 2.0 * float(spy_full.espp_alpha),
        "d_homeo": -float(spy_full.lambda_homeo),
    }
    _mut = SpikingJEPA(_ACTIONS, retina=_R, seed=0, rule="espp")
    _mut.lambda_pred = 1.5
    _mut.espp_alpha = 0.25
    _mut.lambda_homeo = 0.9
    _rmut = espp_synapse_rule(_mut)
    _mmap = {fs[0]: cf for cf, fs in _rmut.terms}
    mut_tracks = _mmap == {"d_pred": -3.0, "d_neg": 0.75, "d_homeo": -0.9}

    # (5) REFUSALS + the ablation authoring form.
    refuse_lpl = False
    try:
        espp_synapse_rule(SpikingJEPA(_ACTIONS, retina=_R, seed=0))  # rule='lpl' default
    except ValueError:
        refuse_lpl = True
    ctrl_refuses = 0
    for _kw in ({"transport": True}, {"freeze_top": True}):
        _inst = SpikingJEPA(_ACTIONS, retina=_R, seed=0, rule="espp")
        for _k2, _v2 in _kw.items():
            setattr(_inst, _k2, _v2)
        try:
            espp_synapse_rule(_inst)
        except ValueError:
            ctrl_refuses += 1
    ablation_form = [fs[0] for _c2, fs in rule_bit.terms] == ["d_pred"]

    # (6) COMPOSER SMOKE, value-bound (the third-landing lesson: a no-red smoke with
    # e == 0.0 contributes nothing to itself).
    from snn_data_lake.plasticity.rule_composer import RuleComposer, RuleSpec

    _reads = set()
    for _c2, _fs2 in rule_full.terms:
        _reads.update(_fs2)
    _comp = RuleComposer("sum", "w")
    _comp.add(RuleSpec("jepa_espp", rule_full, reads=_reads, writes={"w"}))
    _alg_d = EligibilityAlgebra(term_budget=1)
    _alg_d.declare_factor("w", "the composed substrate (smoke)")
    _comp.add(
        RuleSpec("decay", _alg_d.author("decay", [(-0.5, ("w",))]), reads={"w"}, writes={"w"})
    )
    _best = max(cases_f, key=lambda c2: abs(rule_full.evaluate(**c2)))
    _contrib = rule_full.evaluate(**_best)
    _bind = dict(_best)
    _bind["w"] = 1.0
    _w1, _parts = _comp.step(1.0, _bind)
    composes = (
        _contrib != 0.0
        and _parts["jepa_espp"] == _contrib
        and _parts["decay"] == -0.5
        and _w1 == 1.0 + (_parts["jepa_espp"] + _parts["decay"])
    )

    # (7) the mis-declaration control through the exported check (the algebra's own red).
    _alg_bad = EligibilityAlgebra(term_budget=len(rule_full.terms))
    _seen_bad = []
    for _cf, _fs2 in rule_full.terms:
        for _f2 in _fs2:
            if _f2 not in _seen_bad:
                _seen_bad.append(_f2)
                _alg_bad.declare_factor(
                    _f2, "control copy of the declared factor (mis-declaration arm)"
                )
    _bad = _alg_bad.author(
        "jepa.espp.synapse.misdeclared",
        [((cf + 1e-6) if fs[0] == "d_neg" else cf, fs) for cf, fs in rule_full.terms],
    )
    misdecl_red = False
    try:
        equivalence_check(_bad, lambda c: _inc_map_f[id(c)], cases_f, tol=tol_full)
    except AssertionError:
        misdecl_red = True

    print(
        f"delegation : LIVE={delegation_live} (identity asserted; rebound core moved "
        f"production; restored by identity={_restored_ok})"
    )
    print(
        f"core       : BIT vs frozen pre-extraction reference on {bit_states} in-box states, "
        f"exact={core_bit_ok}; corrupted-lambda control REDS on "
        f"{control_red_states}/{bit_states} states (per-state, the pass condition's grain)"
    )
    print(
        f"declaration: BIT arm (ablation, tol=0.0) worst |d|={worst_bit:.1e} on {len(cases_b)} "
        f"sites of which {live_b} LIVE; FULL arm worst |d|={worst_full:.3e} <= derived tol "
        f"{tol_full:.3e} (K={_k_ulp(rule_full):.0f} ulps) on {len(cases_f)} sites of which "
        f"{live_f} LIVE ({off_f} OFF-polarity ff + rec sites from BOTH write-loop halves); "
        f"DELTA arm (the APPLIED weight, ff AND rec) worst |d|={worst_delta:.3e} <= "
        f"{tol_delta:.3e}, corrupted-production control REDS={delta_control_red}; 1e-6 "
        f"mis-declaration REDS={misdecl_red}; negative-homeo regime (rho_star=0.9 override) "
        f"worst |d|={worst_nh:.3e} <= {tol_nh:.3e} with {neg_homeo_sites}/{len(cases_nh)} "
        f"sites at d_homeo < 0"
    )
    print(
        f"pins       : coefficient terms pin={coeffs_pin} (defaults are DEGENERATE at "
        f"lp=alpha=1 -- 2*lp*alpha == 2*lp == 2.0; the composed product's discrimination is "
        f"the mutated-instance arm={mut_tracks}); lpl+transport+freeze_top refuse="
        f"{refuse_lpl}/{ctrl_refuses}; ablation authors 1-term form={ablation_form}"
    )
    print(
        f"palette    : composes through the T051a kernel={composes} (measured halves: "
        f"contribution nonzero + w1 arithmetic consumed; parts==evaluate is the same "
        f"evaluation both sides, consistency not evidence)"
    )
    print(line)

    assert delegation_live and _restored_ok, (
        "jepa's espp branch must route through THIS module's core (and both globals "
        "must be restored to the original function object)"
    )
    assert core_bit_ok, "the core must be bit-identical to the pre-extraction incumbent math"
    assert control_red, (
        f"the corrupted-lambda control must red the BIT comparison on EVERY state "
        f"(continuous draws; got {control_red_states}/{bit_states})"
    )
    assert bit_ok, f"BIT arm (ablation) must be exact, worst |d|={worst_bit!r}"
    assert live_b > 0 and live_f > 0, "equivalence needs LIVE sites (nonzero incumbents)"
    assert off_f > 0, (
        "OFF-polarity ff sites must be sampled (the first draft's sorted[:3] sampled "
        "71/1 ON/OFF -- half the event vector unexercised, adversarial panel)"
    )
    assert full_ok, f"FULL arm |d|={worst_full!r} > derived tol {tol_full!r}"
    assert nh_ok and neg_homeo_sites > 0, (
        f"the negative-homeo regime must hold ({worst_nh!r} <= {tol_nh!r}) with d_homeo < 0 "
        f"actually exercised ({neg_homeo_sites} sites) -- one-signed coverage was the panel's "
        "MEDIUM"
    )
    assert delta_control_red, "the corrupted-production control must red the delta arm"
    assert misdecl_red, "the 1e-6 mis-declaration must red the exported equivalence check"
    assert coeffs_pin and mut_tracks, "coefficients must pin exactly and track mutation"
    assert refuse_lpl and ctrl_refuses == 2, (
        f"must refuse lpl AND transport AND freeze_top (lpl={refuse_lpl}, controls={ctrl_refuses}/2)"
    )
    assert ablation_form, "the ablation instance must author the 1-term form"
    assert composes, "the authored rule must compose through the kernel with a nonzero contribution"

    # ---- (8) T052-ii: THE TEMPORAL-STREAM ADOPTION BENCHMARK (P2 measurement), and it
    # measured a MIXED NULL -- recorded with its own review history, not smoothed.
    # Protocol: the core driven on the REAL packs with the EpisodeDriver trial SHAPE
    # (sample-phase half-window = sb, probe-phase = echo for 'same'; a DIFFERENT-LABEL
    # sample's probe half = neg -- oracle truth by construction, FR-019's named
    # supervisory source); encoder = input-D -> hidden-64 LIF (tau 2.0), auto-gained
    # into the 0.15-0.3 rate band on 20 samples (achieved 0.27-0.29 -- an early draft
    # said "~0.2" and "no hand constant"; the BAND is a hand-chosen harness constant,
    # the GAIN inside it is the network's own); e = the per-input WINDOW INTEGRAL
    # (mean over the window -- the first draft returned a final-EMA SAMPLE carrying
    # ~3 frames, which under-integrated SHD 689x and made its arm a FAILED MANIPULATION
    # its record mislabeled "sparse audio": SHD is DENSER than N-MNIST, the panel
    # measured; corrected before landing); dw = lr*outer(m*sg, e); ridge probe
    # IDENTICAL across arms. RECORDED, corrected protocol (3 seeds x 2 packs, 200/56
    # split, hidden 64 == the eprop encoders' width, LR .05, 5 epochs, 2026-08-27):
    #   N-MNIST (chance .10): twin .589/.446/.607 -> espp .464/.500/.464 (|dW|/scale
    #     ~2.5) -- 2/3 seeds degrade, the improving seed is the WEAKEST twin draw;
    #     mean .547 -> .476.
    #   SHD (chance .05): twin .125/.304/.304 -> espp .232/.250/.179 (|dW|/scale
    #     .18-.56, now genuinely coupled) -- 2/3 degrade, same weak-draw pattern;
    #     mean .244 -> .220.
    #   RANDOM-ENCODER control (identical construction, independent init seeds 7/8/9,
    #     NEVER trained -- differs from the twin ONLY in the draw): N-MNIST
    #     .429/.554/.536, SHD .143/.286/.161. espp lands INSIDE the random-encoder
    #     band on both packs: the training does not add class information the
    #     architecture doesn't already carry (the attribution).
    #   Supervised incumbent (models/eprop_nmnist.pt, SAME width): latent-probe .875
    #     through the identical ridge probe, own head .964 -- dominates every
    #     self-supervised arm by 27+ points.
    # ROBUSTNESS (the adversarial panel's six probes, all survived by the null):
    # surrogate swaps incl. the production membrane surrogate at MATCHED drive
    # (degrades on both probes; the harness sg = exp(-|sb-.5|) is near-constant, mean
    # .74 -- a harness choice, disclosed), eligibility realignment, per-term neuters
    # (the contrastive term carries 93%, and at alpha=1 -- production's own default --
    # contr reduces exactly to 2*(neg-echo), sb cancelling), an independent
    # nearest-centroid probe (gap WIDER), a ridge-lambda sweep (gap at every lambda),
    # 8 train seeds + 5 encoder seeds (degrades or ties, NEVER helps; one encoder
    # seed ties exactly -- the effect is seed-graded, the direction is not).
    # ADJUDICATION: NO PRODUCTION WIRE lands on this measurement. Precisely argued
    # (the policy panel refused the first draft's "P8#5 applied to the wire" gloss):
    # P8 #5 proper (method-removal) does not bind -- espp is adopted literature, not
    # an invention, and its in-tree expression is the T052-i declaration, whose OWN
    # declared scope measures green (the BIT/FULL/DELTA arms above); FR-018's
    # retirement trigger ("contributing nothing measurable in its declared scope")
    # therefore does not fire. What the null kills is any WIRE and any capability
    # claim. FR-019 is satisfied declaratively (the palette clause requires inclusion
    # with the oracle NAMED -- its siblings say "ON BUILT TISSUE" where a live
    # substrate is meant, and say nothing of the sort for ESPP). e-prop REMAINS the
    # one perception path; the host branch stays presence-not-participation,
    # disclosed. THE ORACLE, honestly (the honesty panel's control, which refuted the
    # draft sentence "the null is the RULE's, not the oracle's"): the same/different
    # supervision was PRESENT in every arm (the different-LABEL draw uses labels), and
    # a LABEL-BLIND control measured IDENTICAL means -- because on 10-class balanced
    # data a blind draw is ~90% different-label BY CHANCE, the oracle's guarantee adds
    # almost nothing at this class count. Presence, not participation: the measurement
    # is a null for the espp contrast mechanism at this scale, supervised or not, and
    # it CANNOT separate "the rule fails" from "the supervision never reaches the
    # weights" -- both are recorded, neither is claimed. The in-gate arm below
    # re-derives seed 1's N-MNIST sign on the RECORD's exact rng path (espp <= twin:
    # the NULL is the assertable fact on the pinned path; the effect is SEED-GRADED --
    # one weight seed improves, one ties -- and the mixed table above is the record;
    # no accuracy bar, the eprop_shd separability lesson) and RAISES LOUDLY when the
    # pack is absent.
    # P1 EXCEPTION (c), claimed where the siblings claim it (rule_composer /
    # eligibility_algebra carry the same block): every protocol constant in this
    # section (the reshape, the 200/56 split, 12 gain iterations, the 0.15-0.3 band,
    # the elig/tau time-constants, mu seed .25, 5 epochs, EMA .98/.02, LR .05, clip
    # 4.0, ridge 1e-2) is a benchmark EXERCISE VALUE recorded with its measurement --
    # this module is a wire exemption with zero live callers, and the exception
    # persists exactly as long as the exemption does (the re-adjudication trigger in
    # dormancy_gate's row).
    # distinct-from: sensorimotor/eprop_nmnist.py -- the SUPERVISED incumbent this benchmark
    #   compares AGAINST (torch, e-prop, classifier); this arm is the numpy espp probe
    #   harness, benchmark-local. Site read.
    # distinct-from: cerebellum/spiking_pid.py, basal_ganglia/action_gating.py,
    #   sensorimotor/signal_loop.py -- generic run/window vocabulary only (PID drives, race
    #   circuits, min-max normalizers); nothing shared with this probe. Sites read.
    # PACK GATING, the percept_surprise precedent (its :320-348 adjudicated this exact
    # artifact: an unverifiable check must be LOUD, not quiet -- a bare-return SKIP exits
    # 0 and is scored PASS; and the raise is kept "for the case that actually matters --
    # the artifact is present but unreadable"). The pack is TRACKED, so absence means a
    # broken checkout, and a corrupt pack or numpy fault RAISES rather than reporting a
    # quiet CANNOT-VERIFY green (the first draft's bare `except Exception` swallowed
    # exactly that case -- the policy panel's finding 4).
    import pathlib as _pl8

    import numpy as _np

    from snn_data_lake.gates.gradient_claim_gate import _repo_root as _rr8

    _pack_path = _pl8.Path(_rr8()) / "models" / "nmnist_pack.npz"
    assert _pack_path.exists(), (
        f"T052-ii benchmark CANNOT VERIFY: {_pack_path} is absent -- the pack is a TRACKED "
        "artifact (git ls-files models/nmnist_pack.npz), so a missing file is a broken "
        "checkout, not an optional extra. Regenerate with `python -m training.make_packs`."
    )
    _pack = _np.load(_pack_path)
    import math as _math8

    _X = _pack["frames"].astype("float32").reshape(256, 15, -1)
    _y = _pack["labels"].astype("int64")
    _tr, _te = _np.arange(200), _np.arange(200, 256)
    _tau8 = 2.0  # benchmark encoder tau -- harness exercise value (the recorded protocol's)
    _decay = float(_math8.exp(-1.0 / _tau8))

    def _run_win(w, xs):
        v = _np.zeros((xs.shape[0], 64), "float32")
        c = _np.zeros((xs.shape[0], 64), "float32")
        for t in range(xs.shape[1]):
            v = v * _decay + xs[:, t] @ w.T
            sp = v >= 1.0
            c += sp
            v = _np.where(sp, 0.0, v)
        return c / xs.shape[1]

    _wr = _np.random.default_rng(1)
    _w0 = _wr.normal(0.0, 1.0, (64, _X.shape[2])).astype("float32")
    for _ in range(12):
        _rt = float(_run_win(_w0, _X[:20]).mean())
        if 0.15 < _rt < 0.3:
            break
        _w0 *= 0.2 / max(_rt, 1e-4)

    def _elig8(xs):
        # the per-input WINDOW INTEGRAL. The first draft returned the low-pass EMA's
        # value AT the last step / T -- a final SAMPLE carrying ~3 frames, which
        # under-integrated SHD 689x (the adversarial panel's blocker: its arm could
        # not move W by construction and supported no claim). Corrected pre-landing.
        return xs.mean(axis=1)

    _w = _w0.copy()
    _r8 = _np.random.default_rng(1)  # the RECORD's rng path (default_rng(seed), seed 1;
    _mu8 = _np.full(64, 0.25, "float32")  # a first draft used rng 0 + a per-epoch-rebuilt
    _order = _np.arange(200)  # order and printed a value the record could not explain)
    for _ep in range(5):
        _r8.shuffle(_order)  # cumulative shuffle, exactly as the standalone protocol
        for _k in _order:
            _xk = _X[_k : _k + 1]
            _sb = _run_win(_w, _xk[:, :7])[0]
            _echo = _run_win(_w, _xk[:, 7:])[0]
            _j = int(_r8.choice(_np.flatnonzero(_y[:200] != _y[_k])))
            _neg = _run_win(_w, _X[_j : _j + 1, 7:])[0]
            _mu8 = 0.98 * _mu8 + 0.02 * _sb
            _m8 = _np.array(
                espp_modulator_core(
                    64,
                    _sb.tolist(),
                    _echo.tolist(),
                    _neg.tolist(),
                    _mu8.tolist(),
                    1.0,
                    1.0,
                    0.5,
                    0.25,
                ),
                "float32",
            )
            _sg8 = _np.exp(-_np.abs(_sb - 0.5))
            _w += 0.05 * _np.outer(_m8 * _sg8, _elig8(_xk)[0])
            _np.clip(_w, -4.0, 4.0, out=_w)

    def _probe8(w):
        zt = _np.hstack([_run_win(w, _X[_tr]), _np.ones((200, 1), "float32")])
        zs = _np.hstack([_run_win(w, _X[_te]), _np.ones((56, 1), "float32")])
        Y8 = _np.eye(10, dtype="float32")[_y[_tr]]
        A8 = zt.T @ zt + 1e-2 * _np.eye(zt.shape[1], dtype="float32")
        return float((_np.argmax(zs @ _np.linalg.solve(A8, zt.T @ Y8), 1) == _y[_te]).mean())

    _acc_twin, _acc_espp = _probe8(_w0), _probe8(_w)
    print(
        f"(8) T052-ii benchmark (N-MNIST, seed 1, live): twin {_acc_twin:.4f} vs "
        f"espp {_acc_espp:.4f} -- the recorded NULL reproduces (espp <= twin; 3-seed "
        f"record + SHD + the dominant supervised incumbent in the comment above)"
    )
    assert _acc_espp <= _acc_twin, (
        f"the adoption benchmark's RECORDED null must reproduce: espp {_acc_espp:.4f} "
        f"vs twin {_acc_twin:.4f} -- this arm is FULLY seeded and deterministic, so any "
        "espp > twin here is a real change (a numpy/BLAS shift or a rule edit), and the "
        "T052-ii adjudication (no wire) must be RE-OPENED, not silently enjoyed. (A "
        "first draft carried +0.02 slack; the panel measured the arm deterministic, so "
        "the slack absorbed nothing but the advertised trigger.)"
    )
    print("espp_rule self-check OK")


if __name__ == "__main__":
    main()
