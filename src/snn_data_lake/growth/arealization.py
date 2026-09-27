r"""Arealization -- organizer fields, Boolean + graded GRN machinery, the prosomeric grid,
and two measured negative results (specs/022 T040 + T041, L5).

P8 INVENTION #7's first half (docs/INVENTION_REGISTER.md row 7, cross-linked per FR-031):
ADOPTED parts here -- Wolpert positional-information organizer fields (decaying scalar
fields from organizer nodes) and a Boolean repression-dominant GRN settled to stable
states per node (the Giacomantonio-Goodhill readout family). INVENTED composition (lands
across T040-T042): stage-1 region emission at birth with cell-type emission DEFERRED
(T041), sharp-vs-graded two-regime discretization (T041), one-shared-program locality
(T042). The Boolean half is the SHARP-regime engine; T041 added the prosomeric
(neuromere, zone) grid, the two-regime discretization with both category-error reds, the
deferred cell-type stage, and the GRADED arm carrying D12's re-scoped measurement --
where repression is ENRICHED among admissible networks, NEVER NECESSARY (pooled over
three seeds: repression fraction 0.6249 vs family 0.5010, admission rate 1.66e-4 vs the
EXHAUSTIVELY enumerated zero-repression slice's 5.73e-5 -- the T041 panel's M0 killed a
drafted necessity conjunct with 19 zero-repression admissibles of 331,776, and the
deterministic comparator is that kill kept as the instrument; direction stable at g/2
and 2g, both run at full sample size, loose-bar red reachable) and target-discrimination
is shown STRUCTURALLY out of reach for any permutation-closed sign family (a uniform
census cannot distinguish gene-relabeled targets; symmetry-breaking interaction priors
-- G-G's curated edge list -- are what discrimination costs; adopting one is future
work, not this build's claim).

WHAT T040 DELIVERS, honestly sized -- the machinery, a structure theorem with its own
sensitivity control, a measured NON-result, and a re-scope:

1. MACHINERY: derived organizer fields (lambda = L/(2 ln 2), the unique length putting
   the 0.5 crossing at the midline -- no free decay constant); deterministic Boolean
   settle over exhaustively enumerated wirings (3^12 off-diagonal sign matrices x 2^4
   basal vectors = 8,503,056 candidates -- nothing curated, nothing mis-copyable from a
   paper's edge table); the census-derived consensus readout behind emit_regions()
   (a derivation PIN policed by the self-check, so the T043 consumer never pays the census).

2. THE STRUCTURE THEOREM, asserted with its control in one command: under the
   repression-dominant update rule the activation-only family is MONOTONE, and the
   single-organizer drive is NESTED (anterior drive is a superset of posterior drive),
   so settled states are nested and NO non-nested map -- the countergradient included --
   is reachable without repression. Verified empirically on every self-check: 0 nesting
   violations across all 65,536 monotone candidates, against >0 violations in the full
   signed family (the sensitivity arm; a harness that could not find violations at all
   would prove nothing by finding none in the monotone family).

3. THE MEASURED NON-RESULT, which is the load-bearing honesty of this task: the first
   draft asserted a "necessity dissociation" (activation-only admits 0 under one
   organizer; admits >0 when a posterior organizer hands the answer over) as a
   reproduction of G-G's repression-dominance finding. THE REVIEW PANEL'S BLOCKER
   KILLED THAT CLAIM, and the kill is kept: the zero is the monotonicity theorem
   restated (repression is the rule's ONLY non-monotone primitive, so "every admissible
   network carries repression" is true of ANY non-nested target); a gene-label
   permutation control showed the dissociation is INVARIANT under which gene the
   organizer drives -- it carries zero bits about the {fgf8,pax6}/{emx2,coup_tf1} map.
   And the census is NOT target-discriminative: a biologically meaningless permuted
   target ([1,0,1,0]/[0,1,0,1]) admits 5,150 candidates at repression fraction 0.5188,
   ABOVE the cortical target's 5,292 at 0.5088 -- measured in the same pass, printed on
   every run. Per P8, a criterion whose red is unreachable within its family is
   DELETED, not weakened: the "necessity dissociation" is gone as a claim about the
   map, and survives only as what it is -- a regression guard on the settle rule.

4. THE RE-SCOPE, recorded rather than smoothed: research.md D12's owed EMPIRICAL
   admissibility measurement is NOT dischargeable on a Boolean substrate with nested
   binary drive -- Boolean logic makes repression-necessity analytic wherever the
   target is non-nested. The empirical form needs GRADED fields, and the invention's
   own two-regime split puts graded cortex at T041: the D12 measurement moves there,
   named in the tasks row. Two earlier development reds are also kept (a two-organizer
   draft nulled the graded dominance statistic at 0.4858 vs 0.5000; the single-organizer
   graded statistic passed only by a whisker at 0.5088 and its hand-the-answer arm
   inverted once basal bits entered) -- three constructed failures shaped this file,
   and recording them is what P8's disclosure clause is for.

THE UPDATE RULE AND ITS TWO DISCLOSED ORDER SENSITIVITIES. Genes settle by deterministic
ROUND-ROBIN asynchronous sweeps in the declared order GENES = (fgf8, pax6, emx2,
coup_tf1):

    next_i = (basal_i OR any-active-activator OR organizer-input_i) AND NOT any-active-repressor

(the veto outranks basal and extrinsic drive -- repression-DOMINANT semantics). Two
order effects, both disclosed with measurements rather than asserted away: (a) a node
with ambiguous drive resolves by gene update order (the name-decided-rank-key defect
family); (b) THE SIGNED CENSUS'S ADMISSIBLE SET DEPENDS ON THE GENE ORDER -- the panel
measured different admissible counts under permuted visit orders, so the census is
exact over its family GIVEN the declared order semantics, never order-free; the
self-check prints the identity-vs-reversed admissible delta on every run so the
sensitivity is a number, not a rumor. The MONOTONE-family theorem in (2) is the one
order-free quantity here (a monotone sweep from the drive-up initial converges to the
least fixed point regardless of visit order), which is exactly why the theorem, and
not the signed census, carries the assert.

P1: lambda derived; thresholds are fixed points on normalized outputs (0.5 on level01
fields, P1 case (a)); the census family is exhaustive; _SWEEP_CAP = 2 * 2**K is derived
from the state-space bound (2**K distinct states, doubled for the sweep-operator
transient) and recomputed from K, not written as a literal. The retired first-cut noise
control carried an underived sigma and compared two arms a panel refuter proved
ALGEBRAICALLY IDENTICAL on the consensus network (the settled label is a bijection of
the thresholded fgf8 bit) -- deleted per P8, not tuned; boundary statistics belong to
T046's gate with T041's graded fields, where the arms can actually differ.

Deferred, named so nobody builds the wrong thing next: locality masks and scaffold
refusal are T042; the spec-backed external consumer is T043's placement priors (the
dormancy exemption names exactly that, corrected by the panel's M7); cell-type emission
is invention #7's stage 2 (CELL_TYPE_STAGE says so at the site); and a
symmetry-breaking interaction-prior census, if ever wanted, is new work owing its own
P8 debts.

# distinct-from: connectome.py -- that file DECLARES static mesoscale anatomy; this one
#   GROWS region labels from fields. Read.
# distinct-from: growth/cortical_prior.py -- H01 measured reference data (an external
#   prior); this is the generative mechanism the prior will seed at T043. Read.
# distinct-from: growth/placement.py -- placement CONSUMES region labels to order grown
#   units; this module EMITS the labels. Read.
# distinct-from: neuromodulation/locus_coeruleus.py -- its _population_spikes integrates
#   a LIF pool's spike count under a scalar drive (a neuromodulator readout); this file
#   settles a Boolean gene network to fixed points (no membranes, no spikes). Read.
# distinct-from: reasoning/random_boolean_network.py -- Kauffman NK: RANDOM truth tables
#   per node, synchronous update, order-chaos statistics over seeded ensembles; this
#   file enumerates SIGNED-EDGE threshold networks against a TARGET map under
#   repression-dominant asynchronous semantics. Same broad family (Boolean nets),
#   different mechanism, different question. Read.
# distinct-from: reasoning/test_prototype.py + cerebellum/spiking_pid.py +
#   growth/autapse_neuron.py + plasticity/intrinsic_plasticity.py +
#   gates/gradient_claim_gate.py + substrate/lake_bridge.py +
#   gates/perf_baseline.py -- recall controls, LIF populations, delayed-match
#   trials, drive shaping, gradient nulls, a storage iterator, and the
#   reported-not-asserted idiom's precedent; none is a gene-network mechanism. Read.
"""

import math

import numpy as np

GENES = ("fgf8", "pax6", "emx2", "coup_tf1")
K = len(GENES)
# The countergradient target (textbook, not paper-transcribed): the anterior compartment
# expresses {fgf8, pax6}, the posterior expresses {emx2, coup_tf1}.
_ANTERIOR = np.array([1, 1, 0, 0], dtype=np.uint8)
_POSTERIOR = np.array([0, 0, 1, 1], dtype=np.uint8)
# The permutation-null target for the non-discrimination measurement: biologically
# meaningless gene grouping, same non-nested two-compartment shape.
_NULL_ANTERIOR = np.array([1, 0, 1, 0], dtype=np.uint8)
_NULL_POSTERIOR = np.array([0, 1, 0, 1], dtype=np.uint8)
_FGF8, _EMX2 = 0, 2  # organizer routing indices into GENES
# Derived, not picked: 2**K distinct Boolean states bound any deterministic orbit; the
# factor 2 covers the sweep operator's transient before the orbit is entered.
_SWEEP_CAP = 2 * (2**K)


class OrganizerField:
    """A decaying scalar field from one organizer node on a 1-D axis, normalized [0, 1].

    P1: the decay length is DERIVED -- lambda = L / (2 ln 2) puts the 0.5 crossing exactly
    at the midline, which is what the half-field target means geometrically. No free decay
    constant exists to tune."""

    def __init__(self, span, origin):
        self.span = float(span)
        self.origin = float(origin)
        self.lam = self.span / (2.0 * math.log(2.0))

    def sample(self, xs):
        d = np.abs(np.asarray(xs, dtype=np.float64) - self.origin)
        return np.exp(-d / self.lam)


def organizer_inputs(n_nodes, posterior_organizer=False):
    """(n_nodes, K) uint8 organizer drive. Production formulation: ONE anterior organizer
    driving fgf8, thresholded at 0.5 (P1 case (a)). `posterior_organizer=True` adds a hem
    source driving emx2 -- the hand-the-answer arm of the nesting measurement (a second
    organizer un-nests the drive). EVEN n_nodes only: an odd grid puts a node exactly on
    the midline where the field equals the threshold and the strict `>` decides it -- a
    knife-edge this module refuses rather than tie-breaks."""
    if n_nodes % 2:
        raise ValueError(f"n_nodes must be even (a node would sit on the midline): {n_nodes}")
    xs = (np.arange(n_nodes, dtype=np.float64) + 0.5) / n_nodes  # cell centers
    a = OrganizerField(1.0, 0.0).sample(xs)
    p = OrganizerField(1.0, 1.0).sample(xs)
    inputs = np.zeros((n_nodes, K), dtype=np.uint8)
    inputs[:, _FGF8] = a > 0.5
    if posterior_organizer:
        inputs[:, _EMX2] = p > 0.5
    return inputs


def _all_wirings():
    """(3**12, K, K) int8: every off-diagonal sign assignment in {-1, 0, +1}; self-edges 0
    (DISCLOSED: autoregulation is outside this census's family)."""
    off = [(i, j) for i in range(K) for j in range(K) if i != j]
    n = 3 ** len(off)
    idx = np.arange(n, dtype=np.int64)
    w = np.zeros((n, K, K), dtype=np.int8)
    for e, (i, j) in enumerate(off):
        w[:, i, j] = ((idx // (3**e)) % 3).astype(np.int8) - 1  # regulator j -> target i
    return w


def settle(wirings, drive, gene_order=None):
    """Deterministic round-robin Boolean settle for every wiring over every node.

    wirings: (M, K, K) int8, entry [m, i, j] = sign of regulator j onto target i.
    drive  : (M, N, K) uint8 standing drive (basal OR organizer input; repressible).
    gene_order: visit order for the round-robin sweep (default: the declared GENES
    order). The order is a SEMANTIC parameter of the signed census -- see the module
    docstring's disclosure -- and the self-check measures its effect rather than
    assuming it away.
    Returns (states, converged): states (M, N, K) uint8, converged (M, N) bool -- False
    where a full sweep still changed the state at the cap (a deterministic limit cycle;
    CYCLIC nodes are inadmissible, never scored)."""
    order = tuple(range(K)) if gene_order is None else tuple(gene_order)
    m, n = wirings.shape[0], drive.shape[1]
    act = (wirings > 0).astype(np.uint8)
    rep = (wirings < 0).astype(np.uint8)
    state = drive.copy()  # init = driven genes on
    converged = np.zeros((m, n), dtype=bool)
    live = np.ones(m, dtype=bool)
    for _ in range(_SWEEP_CAP):
        before = state[live].copy()
        s = state[live]
        for g in order:
            a_on = np.einsum("mk,mnk->mn", act[live, g, :], s) > 0
            r_on = np.einsum("mk,mnk->mn", rep[live, g, :], s) > 0
            s[:, :, g] = (((drive[live][:, :, g] > 0) | a_on) & ~r_on).astype(np.uint8)
        state[live] = s
        conv = (state[live] == before).all(axis=2)
        converged[live] = conv
        live_next = ~converged.all(axis=1)
        if not live_next.any():
            break
        live = live_next
    return state, converged


def census(family="all", posterior_organizer=False, gene_order=None):
    """One exhaustive pass over the family (every off-diagonal wiring x every basal
    vector) on the two organizer-input classes (every node in a compartment sees an
    identical thresholded input, so two representative nodes ARE the clean grid).
    Scores BOTH the countergradient target and the permutation-null target in the same
    pass, and counts NESTING VIOLATIONS (anterior settled state not a superset of the
    posterior one) among fully-converged candidates -- the structure theorem's quantity
    and its sensitivity control come from one command. Returns COUNTED output plus the
    consensus (wiring, basal) for the countergradient target (parsimony-then-repression
    pick; lexicographic last resort DISCLOSED, tie count returned).

    family="nonneg" restricts signs to {0, +1} (the monotone family). The census is
    exact over its family UNDER THE DECLARED GENE ORDER; pass gene_order to measure the
    order sensitivity (the self-check does)."""
    inputs = organizer_inputs(2, posterior_organizer=posterior_organizer)
    tgt = np.stack([_ANTERIOR, _POSTERIOR]).astype(np.uint8)
    null_tgt = np.stack([_NULL_ANTERIOR, _NULL_POSTERIOR]).astype(np.uint8)
    w_all = _all_wirings()
    if family == "nonneg":
        w_all = w_all[(w_all >= 0).all(axis=(1, 2))]
    n_adm = n_adm_null = n_fixed = n_nest_viol = n_total = 0
    sum_rep_adm = sum_rep_all = 0.0
    best = None
    best_ties = 0
    nz_all = np.abs(w_all).sum(axis=(1, 2)).astype(np.float64)
    neg_all = (w_all < 0).sum(axis=(1, 2)).astype(np.float64)
    with np.errstate(invalid="ignore"):
        repfrac_all = np.where(nz_all > 0, neg_all / np.maximum(nz_all, 1), 0.0)
    for b_idx in range(2**K):
        basal = np.array([(b_idx >> g) & 1 for g in range(K)], dtype=np.uint8)
        drive = np.maximum(basal[None, None, :], inputs[None, :, :])
        drive = np.broadcast_to(drive, (w_all.shape[0], 2, K)).copy()
        states, conv = settle(w_all, drive, gene_order=gene_order)
        all_conv = conv.all(axis=1)
        ok = all_conv & (states == tgt[None, :, :]).all(axis=(1, 2))
        ok_null = all_conv & (states == null_tgt[None, :, :]).all(axis=(1, 2))
        nest_viol = all_conv & ~(states[:, 0, :] >= states[:, 1, :]).all(axis=1)
        n_total += w_all.shape[0]
        n_fixed += int(all_conv.sum())
        n_adm += int(ok.sum())
        n_adm_null += int(ok_null.sum())
        n_nest_viol += int(nest_viol.sum())
        sum_rep_all += float(repfrac_all.sum())
        sum_rep_adm += float(repfrac_all[ok].sum())
        if ok.any():
            idx = np.flatnonzero(ok)
            cost = nz_all[idx] + basal.sum()  # parsimony: edges + basal bits
            order_ix = np.lexsort((idx, -neg_all[idx], cost))
            pick = idx[order_ix[0]]
            key = (float(cost[order_ix[0]]), float(-neg_all[pick]))
            ties = int(((cost == cost[order_ix[0]]) & (neg_all[idx] == neg_all[pick])).sum())
            if best is None or key < best[0]:
                best = (key, w_all[pick].copy(), basal.copy())
                best_ties = ties
            elif key == best[0]:
                best_ties += ties
    out = {
        "n_family": n_total,
        "n_admissible": n_adm,
        "n_admissible_null_target": n_adm_null,
        "n_all_fixed": n_fixed,
        "n_nesting_violations": n_nest_viol,
        "rep_frac_admissible": (sum_rep_adm / n_adm) if n_adm else float("nan"),
        "rep_frac_all": sum_rep_all / n_total,
    }
    if best is not None:
        out["consensus_wiring"] = best[1]
        out["consensus_basal"] = best[2]
        out["consensus_ties_at_pick"] = best_ties
    return out


# The census-derived consensus pair, PINNED BY DERIVATION: census()'s parsimony-then-
# representativeness pick at the declared gene order, copied here verbatim because the
# derivation costs ~28 s and T043 put emit_regions on placement's import path and per-region
# prior surface -- a first-call census would stall any process that reaches the GRN arm.
# (Honest scope, per the T043 panel: today those processes are placement's own main() and
# self-check, i.e. every gate run; the sleep-tick grower takes the cortex arm and does NOT
# reach this yet.) The pin is not a cached measurement behind a patched constant (the
# lru-cache failure this repo has recorded): the self-check re-runs census() FRESH and asserts
# this pin equals the derivation, so the pin cannot drift silently -- and the arrays are
# write-protected against ACCIDENTAL mutation (setflags can reverse it; the police is the
# self-check, not the flag).
CONSENSUS_WIRING = np.array(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]], dtype=np.int8
)
CONSENSUS_WIRING.setflags(write=False)
CONSENSUS_BASAL = np.array([0, 0, 1, 0], dtype=np.uint8)
CONSENSUS_BASAL.setflags(write=False)


def _consensus():
    """The census-derived consensus pair -- returns the DERIVATION PIN above, so a consumer
    never pays the census. The self-check re-derives the census fresh and asserts the pin
    matches, so the pin cannot drift."""
    return (CONSENSUS_WIRING, CONSENSUS_BASAL)


def emit_regions(n_nodes=16, wiring=None, basal=None):
    """Stage-1 region emission (T043's consumer surface): per-node labels from the
    consensus network's stable states on the presentation grid (even n_nodes; the
    consensus is a derivation PIN policed by the self-check, so a consumer never pays
    the census). Labels in
    {"anterior", "posterior", "other", "cyclic"} -- on the current consensus network the
    settled state is a bijection of the fgf8 input bit, so only the first two occur, and
    T041's graded arm did NOT change that (the panel's minor is right: the graded path
    has its own postmitotic readout and never routes through here; the wider contract is
    kept for a non-bijective future consensus, measured unreachable today). Cell-type
    emission is invention #7's deferred stage 2 (CELL_TYPE_STAGE)."""
    if wiring is None or basal is None:
        wiring, basal = _consensus()
    inputs = organizer_inputs(n_nodes)
    drive = np.maximum(basal[None, None, :], inputs[None, :, :]).astype(np.uint8)
    states, conv = settle(wiring[None, :, :], drive)
    labels = []
    for i in range(n_nodes):
        s = states[0, i]
        if not conv[0, i]:
            labels.append("cyclic")
        elif (s == _ANTERIOR).all():
            labels.append("anterior")
        elif (s == _POSTERIOR).all():
            labels.append("posterior")
        else:
            labels.append("other")
    return labels, states[0]


# ==================================================================================================
# T041 -- the prosomeric (neuromere, zone) label grid, the two-regime discretization, the
# deferred cell-type stage, and the D12 graded-admissibility instrument (re-scoped here from
# T040, whose Boolean arm measured non-discrimination). P8 invention #7, register row 7.
# distinct-from: cortex/second_order_access.py -- its state() is a per-tick functional
#   snapshot; nothing here snapshots brain state. Read.
# distinct-from: growth/function_preserving_growth.py -- its run() steps a spiking net
#   over an input window; the settle below iterates a gene-network fixed point. Read.
# ==================================================================================================

# The prosomeric AP axis (Puelles & Rubenstein), COARSE-GRAINED and disclosed as such: the
# secondary prosencephalon is one 'sp' entry (hp1/hp2 folded), the diencephalon keeps its three
# prosomeres, the isthmus and r0/r1 are the cerebellar territory, and r4-r7 stand in for the
# caudal rhombomeres (the full r8-r11 pseudo-rhombomere detail is below this build's grain).
NEUROMERES = ("sp", "p3", "p2", "p1", "m1", "is", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "sc")
ZONES = ("roof", "alar", "basal", "floor")  # dorsoventral plates

# SNDL anatomical region -> (neuromere, zone), the declared prosomeric coordinate of each region
# this project grows or scaffolds. Anatomy DECLARATION at textbook grain (the atlas-node gate's
# analogue): cortex/hippocampus/insula are secondary-prosencephalic alar (pallium);
# basal_ganglia is SUBPALLIAL -- which is ALAR, like the pallium (the T041 panel's M1 caught the
# first draft calling it basal-plate: in the prosomeric model the entire telencephalon, pallium
# AND subpallium, is alar sp; the sp BASAL plate is the tuberal/mammillary hypothalamus);
# thalamus is p2 alar with the epithalamus its roof; hypothalamus is sp basal (prosomeric, NOT
# diencephalic -- the model's own headline revision); cerebellum is isthmo-cerebellar (is/r1
# alar); pons r2-r4 basal; medulla caudal-rhombomeric. The pallium/subpallium distinction is
# BELOW this grain's zone axis (it would need a finer dorsoventral set, not the plate axis).
REGION_COORDS = {
    "cortex": ("sp", "alar"),
    "hippocampus": ("sp", "alar"),
    "insula": ("sp", "alar"),
    "claustrum": ("sp", "alar"),
    "amygdala": ("sp", "alar"),
    "basal_ganglia": ("sp", "alar"),
    "hypothalamus": ("sp", "basal"),
    "epithalamus": ("p2", "roof"),
    "thalamus": ("p2", "alar"),
    "midbrain": ("m1", "alar"),
    "cerebellum": ("r1", "alar"),
    "pons": ("r3", "basal"),
    "medulla": ("r5", "basal"),
    "brainstem": ("r5", "basal"),
    "neuromodulation": ("r1", "basal"),  # LC/raphe: rostral hindbrain tegmentum, coarse
    "olfactory": ("sp", "alar"),
}

# THE TWO REGIMES (the invention's split; importing one into the other is the category error
# the register row names): hindbrain territory discretizes SHARP (rhombomeres are
# lineage-restriction compartments -- boundaries exist from the start, width zero); the
# pallium discretizes GRADED-THEN-POSTMITOTIC (areal identity is a graded TF field that
# becomes discrete only at the postmitotic step). Midbrain/diencephalic prosomeres are
# compartment-like and take the sharp regime at this grain (disclosed simplification).
_GRADED_NEUROMERES = frozenset({"sp"})

#: Neuromere -> macro segment (prosomeric model, coarse): sp + the three diencephalic
#: prosomeres are forebrain; m1 is midbrain; isthmus + rhombomeres are hindbrain; sc is
#: spinal cord (no REGION_COORDS entry maps there). Declared here BESIDE the axis it grains
#: (the T046 panel: a second neuromere table living in a gate drifts from the first);
#: consumed by gates/growth_coverage_gate's Allen external control.
MACRO_OF_NEUROMERE = {
    "sp": "forebrain",
    "p3": "forebrain",
    "p2": "forebrain",
    "p1": "forebrain",
    "m1": "midbrain",
    "is": "hindbrain",
    "r1": "hindbrain",
    "r2": "hindbrain",
    "r3": "hindbrain",
    "r4": "hindbrain",
    "r5": "hindbrain",
    "r6": "hindbrain",
    "r7": "hindbrain",
    "sc": "spinal",
}


def regime_of(neuromere):
    """'graded' for the pallial territory, 'sharp' for compartment neuromeres."""
    if neuromere not in NEUROMERES:
        raise ValueError(f"unknown neuromere {neuromere!r}")
    return "graded" if neuromere in _GRADED_NEUROMERES else "sharp"


# Stage-2 (cell-type) emission is DEFERRED BY DESIGN -- invention #7's two-stage split: region
# at birth, cell type later. A constant, not a stub function: a callable that always refuses
# would be a dead method wearing an API's clothes (P3 shape 1).
CELL_TYPE_STAGE = "DEFERRED (invention #7 stage 2: region-at-birth only at this build)"

# The gradient-claim census flags the graded family's verbs (T041 panel B2: the diff turned
# the CI ratchet RED with these three as the tree's only governed findings). None is a
# learning rule: nothing updates a weight, nothing propagates an error, every output is a
# report field of a measurement instrument -- the ignition_instrument _logistic_fit precedent.
GRADIENT_CLAIM = {
    "graded_settle": (
        "NOT_A_RULE",
        "fixed-point iteration of a STATIC gene network to a settled expression state; the "
        "wiring W is an enumerated CANDIDATE under census, never a learned parameter -- "
        "nothing is updated, nothing owes a cosine",
    ),
    "graded_census": (
        "NOT_A_RULE",
        "a measurement instrument: samples candidate wirings, settles each, scores RMSE "
        "against a target profile, and reports counts; updates no weights, touches no "
        "substrate",
    ),
    "graded_profiles": (
        "NOT_A_RULE",
        "pure target-profile constructor (the countergradient shapes the census scores "
        "against); no dynamics, no updates",
    ),
}


def discretize_sharp(ap_positions, n_segments):
    """SHARP-regime discretization: a lineage-restriction step function -- equal partition of
    [0, 1) into n_segments compartments, boundary width ZERO by construction. No field, no
    threshold, no noise sensitivity: a cell's compartment is its lineage, decided at birth."""
    ap = np.asarray(ap_positions, dtype=np.float64)
    return np.minimum((ap * n_segments).astype(np.int64), n_segments - 1)


def graded_profiles(n_nodes):
    """GRADED-regime state: the continuous countergradient target profiles on the AP axis --
    anterior-high genes follow the anterior organizer field a(x), posterior-high genes its
    complement 1 - a(x). These are the PRE-discretization areal identities; labels appear only
    at the postmitotic step. EVEN n_nodes only, the organizer_inputs refusal (a midline node
    is a knife-edge the postmitotic block-mean would tie-break silently)."""
    if n_nodes % 2:
        raise ValueError(f"n_nodes must be even (a node would sit on the midline): {n_nodes}")
    xs = (np.arange(n_nodes, dtype=np.float64) + 0.5) / n_nodes
    a = OrganizerField(1.0, 0.0).sample(xs)
    tgt = np.empty((n_nodes, K), dtype=np.float64)
    tgt[:, 0] = a  # fgf8
    tgt[:, 1] = a  # pax6
    tgt[:, 2] = 1.0 - a  # emx2
    tgt[:, 3] = 1.0 - a  # coup_tf1
    return tgt


def postmitotic_discretize(profiles):
    """The graded regime's LATE threshold: per-node argmax-block labels from the graded
    profiles at the postmitotic step (anterior if the anterior-block mean exceeds the
    posterior-block mean). Before this step the identity IS the graded vector."""
    p = np.asarray(profiles, dtype=np.float64)
    ant = p[:, :2].mean(axis=1)
    post = p[:, 2:].mean(axis=1)
    return np.where(ant > post, "anterior", "posterior")


def graded_settle(W, b, inputs, gain, tol=None, iter_cap=200):
    """Continuous fixed-point settle for the graded GRN family: x <- sigma_g(W x + b + u) with
    sigma_g(z) = 1 / (1 + exp(-gain * (z - 1/2))), iterated to a fixed point (max-delta below
    tol, derived from the admissibility bar -- see graded_census) or declared NON-CONVERGED at
    iter_cap (inadmissible, never scored). Vectorized over candidates.
    W: (M, K, K) float; b: (M, K); inputs: (N, K); returns (x, converged)."""
    m, n = W.shape[0], inputs.shape[0]
    tol = 1e-6 if tol is None else float(tol)
    x = np.broadcast_to(inputs[None, :, :], (m, n, K)).astype(np.float64).copy()
    converged = np.zeros(m, dtype=bool)
    live = np.ones(m, dtype=bool)
    for _ in range(iter_cap):
        z = np.einsum("mij,mnj->mni", W[live], x[live]) + b[live][:, None, :] + inputs[None, :, :]
        nxt = 1.0 / (1.0 + np.exp(-gain * (z - 0.5)))
        delta = np.abs(nxt - x[live]).max(axis=(1, 2))
        x[live] = nxt
        conv = delta < tol
        converged[live] = conv
        live_next = ~converged
        if not live_next.any():
            break
        live = live_next
    return x, converged


def graded_census(n_samples=40000, seed=13, n_nodes=16, gain=None, bar_mult=1.0):
    """The D12 instrument, GRADED arm (re-scoped from T040): quantitative profile-fit
    admissibility over a SEEDED SAMPLE of the signed family (sampling disclosed -- the
    continuous settle prices exhaustion out; the sampler is uniform over the same 3^12 sign
    family with b sampled per candidate from {0, 1/2, 1}^K), under the SINGLE anterior
    organizer field (graded, unthresholded). A candidate is admissible when its settled
    profile's RMSE against the countergradient target is under the bar; the bar is a fixed
    point on the normalized profile range (range/4, P1 case (a)). The PERMUTED target
    ({fgf8, emx2} anterior-high / {pax6, coup_tf1} posterior-high -- biologically meaningless)
    is scored in the SAME pass: the discrimination question T040's Boolean arm answered NO is
    re-asked here quantitatively, and the answer is REPORTED either way.

    gain: sigmoid steepness; DERIVED default g = 2*ln((1-eps)/eps) with eps = 1/16 (the bar
    fraction over the 4-gene block mean), so sigma spans [eps, 1-eps] over the unit input
    range -- and the self-check re-runs the verdict at g/2 AND 2g, both at full sample
    size with their admissible-n printed (the gain-sensitivity control: a verdict that
    flips with the gain is a gain artifact, not a finding; the T041 panel's B0 caught
    the first draft claiming the 2g arm while never running it -- the arm that widens
    the admissible set had shipped alone, and prose exceeded its measurement)."""
    eps = 1.0 / 16.0
    g = (2.0 * math.log((1.0 - eps) / eps)) if gain is None else float(gain)
    import random as _random

    rng = _random.Random(seed)
    W = np.empty((n_samples, K, K), dtype=np.float64)
    b = np.empty((n_samples, K), dtype=np.float64)
    for m in range(n_samples):
        for i in range(K):
            for j in range(K):
                W[m, i, j] = 0.0 if i == j else float(rng.choice((-1, 0, 1)))
            b[m, i] = rng.choice((0.0, 0.5, 1.0))
    xs = (np.arange(n_nodes, dtype=np.float64) + 0.5) / n_nodes
    a = OrganizerField(1.0, 0.0).sample(xs)
    inputs = np.zeros((n_nodes, K), dtype=np.float64)
    inputs[:, _FGF8] = a  # graded, unthresholded -- the arm the Boolean census lacked
    tgt = graded_profiles(n_nodes)
    perm = tgt[:, [0, 2, 1, 3]]  # {fgf8, emx2} anterior-high: the meaningless grouping
    # bar_mult != 1 is the LOOSE-BAR red's arm only: at bar_mult=4 the bar swallows most of
    # the family and the dominance separation must VANISH (the criterion can fail).
    bar = float(bar_mult) * (tgt.max() - tgt.min()) / 4.0
    x, conv = graded_settle(W, b, inputs, gain=g, tol=(tgt.max() - tgt.min()) / 256.0)
    rmse = np.sqrt(((x - tgt[None, :, :]) ** 2).mean(axis=(1, 2)))
    rmse_perm = np.sqrt(((x - perm[None, :, :]) ** 2).mean(axis=(1, 2)))
    ok = conv & (rmse <= bar)
    ok_perm = conv & (rmse_perm <= bar)
    neg = (W < 0).sum(axis=(1, 2)).astype(np.float64)
    nz = (np.abs(W) > 0).sum(axis=(1, 2)).astype(np.float64)
    with np.errstate(invalid="ignore"):
        repfrac = np.where(nz > 0, neg / np.maximum(nz, 1), 0.0)
    out = {
        "n_samples": int(n_samples),
        "n_converged": int(conv.sum()),
        "n_admissible": int(ok.sum()),
        "n_admissible_perm": int(ok_perm.sum()),
        "rep_frac_admissible": float(repfrac[ok].mean()) if ok.any() else float("nan"),
        "rep_frac_family": float(repfrac.mean()),
        # admission RATE per CONVERGED candidate (panel minor: non-converged candidates are
        # ineligible, so a per-sample rate would deflate the family side).
        "adm_rate": float(ok.sum() / max(int(conv.sum()), 1)),
        "best_rmse": float(rmse.min()),
        "bar": float(bar),
        "gain": float(g),
        "rep_fracs_admissible": [float(v) for v in repfrac[ok]],
    }
    return out


def zero_repression_census(n_nodes=16, gain=None):
    """The DETERMINISTIC comparator (T041 panel M0's replacement for a false necessity
    claim): the zero-repression slice of the graded family is small enough to enumerate
    EXHAUSTIVELY -- off-diagonal signs in {0, +1} (2^12 wirings) x basal in {0, 1/2, 1}^K
    (3^4) = 331,776 candidates, no sampling, no seed. The panel's exhaustive run found 19
    admissible here, refuting the sampled census's apparent 'min 2 repressive edges'
    (a sampling artifact): on the graded arm repression is ENRICHED among admissible
    networks, never NECESSARY -- and this arm is what makes that an exact statement."""
    eps = 1.0 / 16.0
    g = (2.0 * math.log((1.0 - eps) / eps)) if gain is None else float(gain)
    off = [(i, j) for i in range(K) for j in range(K) if i != j]
    n_w = 2 ** len(off)
    idx = np.arange(n_w, dtype=np.int64)
    w_bin = np.zeros((n_w, K, K), dtype=np.float64)
    for e, (i, j) in enumerate(off):
        w_bin[:, i, j] = (idx >> e) & 1
    basals = (
        np.array([[(b // (3**k)) % 3 for k in range(K)] for b in range(3**K)], dtype=np.float64)
        / 2.0
    )
    xs = (np.arange(n_nodes, dtype=np.float64) + 0.5) / n_nodes
    a = OrganizerField(1.0, 0.0).sample(xs)
    inputs = np.zeros((n_nodes, K), dtype=np.float64)
    inputs[:, _FGF8] = a
    tgt = graded_profiles(n_nodes)
    bar = (tgt.max() - tgt.min()) / 4.0
    n_adm = n_conv = 0
    for b_row in basals:
        b = np.broadcast_to(b_row[None, :], (n_w, K)).copy()
        x, conv = graded_settle(w_bin, b, inputs, gain=g, tol=(tgt.max() - tgt.min()) / 256.0)
        rmse = np.sqrt(((x - tgt[None, :, :]) ** 2).mean(axis=(1, 2)))
        n_adm += int((conv & (rmse <= bar)).sum())
        n_conv += int(conv.sum())
    return {
        "n_family": int(n_w * len(basals)),
        "n_converged": n_conv,
        "n_admissible": n_adm,
        "adm_rate": float(n_adm / max(n_conv, 1)),
    }


def main() -> int:
    # ---- ONE exhaustive pass: census + null target + nesting quantities ----------------
    c = census()
    print(
        f"arealization census (declared gene order): {c['n_admissible']} admissible of "
        f"{c['n_family']} candidates; permutation-null target admits "
        f"{c['n_admissible_null_target']}; nesting violations among converged: "
        f"{c['n_nesting_violations']}; repression fraction admissible "
        f"{c['rep_frac_admissible']:.4f} vs family {c['rep_frac_all']:.4f}"
    )
    if c["n_admissible"] == 0:
        print("arealization self-check FAILED: zero admissible candidates for the target")
        return 1
    # ---- THE STRUCTURE THEOREM, with its sensitivity control in the same command ------
    c_mono = census(family="nonneg")
    if c_mono["n_nesting_violations"] != 0 or c_mono["n_admissible"] != 0:
        print(
            f"arealization self-check FAILED: the monotone family produced "
            f"{c_mono['n_nesting_violations']} nesting violation(s) and "
            f"{c_mono['n_admissible']} admissible countergradient(s) -- the nesting "
            f"theorem is broken, so settle() no longer implements a monotone rule "
            f"without repressors"
        )
        return 1
    if c["n_nesting_violations"] == 0:
        print(
            "arealization self-check FAILED: the SIGNED family shows zero nesting "
            "violations -- the violation counter is a dead detector and the monotone "
            "family's 0 proves nothing"
        )
        return 1
    # ---- the un-nesting arm: a second organizer un-nests the drive itself -------------
    c_mono_two = census(family="nonneg", posterior_organizer=True)
    if c_mono_two["n_admissible"] == 0:
        print(
            "arealization self-check FAILED: with the posterior organizer un-nesting "
            "the drive, the monotone family still admits nothing -- the theorem's "
            "nested-drive premise is not what the zero rests on, so something else "
            "broke"
        )
        return 1
    # ---- NON-DISCRIMINATION, printed as the measured negative result ------------------
    print(
        f"arealization NON-DISCRIMINATION (the measured negative result, recorded per "
        f"P8): the permutation-null target admits {c['n_admissible_null_target']} vs "
        f"the cortical target's {c['n_admissible']} -- this Boolean census does NOT "
        f"distinguish the biological map from a meaningless one; D12's empirical "
        f"admissibility measurement moves to the graded regime (T041), as the tasks "
        f"row records"
    )
    # ---- order sensitivity, measured and printed every run ----------------------------
    c_rev = census(gene_order=tuple(reversed(range(K))))
    print(
        f"arealization order sensitivity (disclosed): admissible under declared order "
        f"{c['n_admissible']} vs reversed order {c_rev['n_admissible']} (delta "
        f"{abs(c['n_admissible'] - c_rev['n_admissible'])}) -- the signed census is "
        f"exact over its family GIVEN the declared order semantics, never order-free"
    )
    # ---- the consensus PIN cannot drift: fresh census vs the derivation pin ------------
    w_cached, b_cached = _consensus()
    if not (
        np.array_equal(w_cached, c["consensus_wiring"])
        and np.array_equal(b_cached, c["consensus_basal"])
    ):
        print("arealization self-check FAILED: the consensus PIN differs from a fresh census")
        return 1
    # ---- the accidental-write protection is LIVE (its red, attempted not asserted) -----
    try:
        CONSENSUS_WIRING[0, 0] = 1
        print("arealization self-check FAILED: the consensus pin accepted a write")
        return 1
    except ValueError:
        pass  # write-protected; a deliberate setflags(write=True) remains possible, disclosed
    # ---- clean emission is the exact half-field under the declared order ---------------
    labels, _ = emit_regions(16, wiring=w_cached, basal=b_cached)
    n_ant, n_post = labels.count("anterior"), labels.count("posterior")
    if not (n_ant == 8 and n_post == 8):
        print(
            f"arealization self-check FAILED: clean-field emission is not the exact "
            f"half-field map (anterior {n_ant}, posterior {n_post}, labels {labels})"
        )
        return 1
    # ================================ T041 checks =====================================
    # ---- the prosomeric grid is internally consistent, AND the regime selector is the
    # thing that routes it (T041 panel B1/M6/M8: regime_of had zero readers -- invention
    # #7's own split was dead code; now the partition is asserted THROUGH the selector,
    # and its constructed red is one membership flip in _GRADED_NEUROMERES, which breaks
    # the routing asserts below).
    regimes = {}
    for region, (nm, zone) in REGION_COORDS.items():
        if nm not in NEUROMERES or zone not in ZONES:
            print(f"arealization self-check FAILED: {region} has off-grid coords ({nm},{zone})")
            return 1
        regimes[region] = regime_of(nm)
    n_graded = sum(1 for r in regimes.values() if r == "graded")
    n_sharp = sum(1 for r in regimes.values() if r == "sharp")
    if regimes["cortex"] != "graded" or regimes["cerebellum"] != "sharp":
        print(
            f"arealization self-check FAILED: the regime selector routes cortex to "
            f"{regimes['cortex']} and cerebellum to {regimes['cerebellum']} -- the "
            f"invention's own split is mis-assigned"
        )
        return 1
    # ---- SHARP regime, reached THROUGH the selector: compartments are a step of width
    # zero by construction.
    if regime_of(REGION_COORDS["cerebellum"][0]) != "sharp":
        print("arealization self-check FAILED: sharp arm not selector-routed")
        return 1
    xs = (np.arange(1000, dtype=np.float64) + 0.5) / 1000.0
    seg = discretize_sharp(xs, 7)
    if not (np.diff(seg) >= 0).all() or len(set(seg.tolist())) != 7:
        print("arealization self-check FAILED: sharp discretization is not a 7-compartment step")
        return 1
    # ---- CATEGORY-ERROR RED 1: importing SHARP into the graded regime loses identity --
    # Reconstruct the graded areal profile from its own postmitotic labels (the sharp,
    # label-only reading of cortex, selector-routed): the reconstruction must MISS the
    # profile bar -- region-at-birth labels cannot carry graded areal identity, which is
    # the whole reason stage-1/stage-2 are split.
    if regime_of(REGION_COORDS["cortex"][0]) != "graded":
        print("arealization self-check FAILED: graded arm not selector-routed")
        return 1
    tgt = graded_profiles(16)
    labels_pm = postmitotic_discretize(tgt)
    recon = np.where(
        (labels_pm == "anterior")[:, None],
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )
    bar = (tgt.max() - tgt.min()) / 4.0
    rmse_sharp_on_graded = float(np.sqrt(((recon - tgt) ** 2).mean()))
    if rmse_sharp_on_graded <= bar:
        print(
            f"arealization self-check FAILED: the sharp (label-only) reading reproduces the "
            f"graded profile within the bar ({rmse_sharp_on_graded:.4f} <= {bar:.4f}) -- the "
            f"two regimes are not measurably distinct and the split is decoration"
        )
        return 1
    # ---- CATEGORY-ERROR RED 2: importing GRADED into the sharp regime misplaces
    # compartment boundaries. A graded-threshold reading of the hindbrain axis puts each
    # boundary where the exponential field crosses a level; lineage puts them at the
    # equal-partition positions. The max misplacement must be nonzero and measured.
    n_seg = 7
    f = OrganizerField(1.0, 0.0)
    levels = 1.0 - (np.arange(1, n_seg, dtype=np.float64) / n_seg)  # descending field levels
    # T041 panel M5: levels BELOW the field's on-axis minimum (f.sample([1.0]) = 0.25) put
    # the graded boundary OFF the [0, 1] axis -- the first draft's headline 0.5465 came
    # from a boundary at x = 1.4037. Only attainable levels are compared; the honest
    # on-axis maximum is republished below.
    f_min = float(f.sample([1.0])[0])
    attainable = levels >= f_min
    graded_bounds = -f.lam * np.log(levels[attainable])  # where exp(-x/lam) == level
    lineage_bounds = (np.arange(1, n_seg, dtype=np.float64) / n_seg)[attainable]
    misplacement = float(np.abs(graded_bounds - lineage_bounds).max())
    if not misplacement > 0.01:
        print(
            f"arealization self-check FAILED: the graded reading places hindbrain boundaries "
            f"within 1% of lineage positions (max {misplacement:.4f} over "
            f"{int(attainable.sum())} attainable levels) -- the category-error red is "
            f"unreachable and the regime split proves nothing"
        )
        return 1
    # ---- D12 GRADED ARM (re-scoped from T040): repression ENRICHMENT, never necessity
    # (T041 panel M0: the zero-repression slice, enumerated EXHAUSTIVELY, admits 19 of
    # 331,776 -- the sampled 'min 2 repressive edges' was a sampling artifact, so the
    # necessity conjunct is DELETED, not weakened). The claim pools three seeds (panel
    # M2: a single seed's n=15 point cleared by ~1.3 SE; pooling triples n), with THREE
    # controls in the same command: gain arms at g/2 AND 2g at FULL sample size (panel
    # B0: the 2g arm was documented and never run -- it runs now, its n printed), the
    # loose-bar red, and the deterministic zero-repression comparator.
    pooled_fracs = []
    pooled_adm = pooled_conv = pooled_perm = 0
    per_seed = []
    gc = None
    for sd in (13, 29, 47):
        c_s = graded_census(n_samples=100000, seed=sd)
        if gc is None:
            gc = c_s  # seed 13 stays the reference draw for the gain arms below
        pooled_fracs.extend(c_s["rep_fracs_admissible"])
        pooled_adm += c_s["n_admissible"]
        pooled_conv += c_s["n_converged"]
        pooled_perm += c_s["n_admissible_perm"]
        per_seed.append((sd, c_s["n_admissible"], round(c_s["rep_frac_admissible"], 4)))
    zr = zero_repression_census()
    pooled_rep = float(np.mean(pooled_fracs)) if pooled_fracs else float("nan")
    pooled_rate = pooled_adm / max(pooled_conv, 1)
    print(
        f"arealization graded census (D12 arm, pooled seeds 13/29/47): {pooled_adm} "
        f"admissible of {pooled_conv} converged (per-seed {per_seed}); pooled repression "
        f"fraction {pooled_rep:.4f} vs family {gc['rep_frac_family']:.4f}; permuted "
        f"target admits {pooled_perm} pooled (permutation-closure: a uniform sign family "
        f"CANNOT distinguish gene-relabeled targets -- discrimination needs "
        f"symmetry-breaking interaction priors, which is exactly what G-G's curated edge "
        f"list supplied); zero-repression slice EXHAUSTIVE: {zr['n_admissible']} of "
        f"{zr['n_family']} admissible (rate {zr['adm_rate']:.2e} vs pooled family rate "
        f"{pooled_rate:.2e}) -- repression is ENRICHED among admissible networks, never "
        f"necessary"
    )
    if pooled_adm == 0:
        print("arealization self-check FAILED: the graded census admits nothing at the bar")
        return 1
    if not (pooled_rep > gc["rep_frac_family"] and pooled_rate > zr["adm_rate"] * 1.5):
        print(
            f"arealization self-check FAILED: graded repression-enrichment does not "
            f"separate (pooled {pooled_rep:.4f} vs family {gc['rep_frac_family']:.4f}; "
            f"pooled rate {pooled_rate:.2e} vs zero-repression exhaustive "
            f"{zr['adm_rate']:.2e} x1.5) -- the D12 graded result does not reproduce"
        )
        return 1
    g_half = graded_census(n_samples=100000, gain=gc["gain"] / 2.0)
    g_two = graded_census(n_samples=100000, gain=gc["gain"] * 2.0)
    for arm_name, arm in (("gain/2", g_half), ("2x gain", g_two)):
        if not (arm["n_admissible"] > 0 and arm["rep_frac_admissible"] > arm["rep_frac_family"]):
            print(
                f"arealization self-check FAILED: the enrichment direction flips at "
                f"{arm_name} ({arm['n_admissible']} admissible, "
                f"{arm['rep_frac_admissible']:.4f} vs {arm['rep_frac_family']:.4f}) -- a "
                f"gain artifact, not a finding"
            )
            return 1
    g_loose = graded_census(n_samples=10000, bar_mult=4.0)
    if not (
        g_loose["n_admissible"] == g_loose["n_converged"]
        and abs(g_loose["rep_frac_admissible"] - g_loose["rep_frac_family"]) < 0.02
    ):
        print(
            f"arealization self-check FAILED: the loose-bar red did not fire "
            f"({g_loose['n_admissible']} admissible of {g_loose['n_converged']} converged, "
            f"{g_loose['rep_frac_admissible']:.4f} vs {g_loose['rep_frac_family']:.4f}) -- "
            f"the bar was meant to swallow every converged candidate and erase the "
            f"separation; it did not, so the red is broken"
        )
        return 1
    print(
        f"arealization self-check OK: structure theorem holds (monotone family: 0 "
        f"nesting violations, 0 admissible countergradients across {c_mono['n_family']} "
        f"candidates; signed-family control: {c['n_nesting_violations']} violations; "
        f"un-nested drive re-admits {c_mono_two['n_admissible']} monotone candidates); "
        f"non-discrimination recorded ({c['n_admissible_null_target']} null vs "
        f"{c['n_admissible']} cortical); order delta "
        f"{abs(c['n_admissible'] - c_rev['n_admissible'])} disclosed; consensus cache "
        f"verified; clean emission 8/8; consensus ties at pick: "
        f"{c['consensus_ties_at_pick']}. T041: prosomeric grid consistent "
        f"({len(REGION_COORDS)} regions on {len(NEUROMERES)}x{len(ZONES)}, selector-routed: "
        f"{n_graded} graded / {n_sharp} sharp); sharp regime 7-compartment step; "
        f"category-error reds both fired (sharp-on-graded RMSE {rmse_sharp_on_graded:.4f} "
        f"> bar {bar:.4f}; graded-on-sharp max ON-AXIS boundary misplacement "
        f"{misplacement:.4f}); D12 graded ENRICHMENT separates (pooled "
        f"{pooled_rep:.4f} > family {gc['rep_frac_family']:.4f}; pooled rate "
        f"{pooled_rate:.2e} > zero-repression exhaustive {zr['adm_rate']:.2e} x1.5; "
        f"stable at gain/2 n={g_half['n_admissible']} and 2x gain n={g_two['n_admissible']}; "
        f"loose-bar red fires); cell-type stage: {CELL_TYPE_STAGE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
