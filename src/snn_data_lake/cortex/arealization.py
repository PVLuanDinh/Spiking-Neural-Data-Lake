r"""Arealization -- organizer fields + Boolean GRN machinery, and a measured NEGATIVE result
(specs/022 T040, L5).

P8 INVENTION #7's first half (docs/INVENTION_REGISTER.md row 7, cross-linked per FR-031):
ADOPTED parts here -- Wolpert positional-information organizer fields (decaying scalar
fields from organizer nodes) and a Boolean repression-dominant GRN settled to stable
states per node (the Giacomantonio-Goodhill readout family). INVENTED composition (lands
across T040-T042): stage-1 region emission at birth with cell-type emission DEFERRED
(T041), sharp-vs-graded two-regime discretization (T041), one-shared-program locality
(T042). This module is the SHARP-regime engine and the field mechanics; the graded
cortex regime is T041's.

WHAT T040 DELIVERS, honestly sized -- the machinery, a structure theorem with its own
sensitivity control, a measured NON-result, and a re-scope:

1. MACHINERY: derived organizer fields (lambda = L/(2 ln 2), the unique length putting
   the 0.5 crossing at the midline -- no free decay constant); deterministic Boolean
   settle over exhaustively enumerated wirings (3^12 off-diagonal sign matrices x 2^4
   basal vectors = 8,503,056 candidates -- nothing curated, nothing mis-copyable from a
   paper's edge table); the census-derived consensus readout behind emit_regions()
   (module-cached, so the T043 consumer never pays the census).

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

Deferred, named so nobody builds the wrong thing next: prosomeric (neuromere, zone)
grid, two-regime discretization, graded cortex fields and the D12 empirical
measurement are T041; locality masks and scaffold refusal are T042; the spec-backed
external consumer is T043's placement priors (the dormancy exemption names exactly
that, corrected by the panel's M7).

# distinct-from: connectome.py -- that file DECLARES static mesoscale anatomy; this one
#   GROWS region labels from fields. Read.
# distinct-from: cortex/cortical_prior.py -- H01 measured reference data (an external
#   prior); this is the generative mechanism the prior will seed at T043. Read.
# distinct-from: cortex/placement.py -- placement CONSUMES region labels to order grown
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
#   cortex/autapse_neuron.py + plasticity/intrinsic_plasticity.py +
#   substrate/gradient_claim_gate.py + substrate/lake_bridge.py +
#   substrate/perf_baseline.py -- recall controls, LIF populations, delayed-match
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


_CONSENSUS = None  # (wiring, basal), derived once -- see _consensus()


def _consensus():
    """The census-derived consensus network, computed once per process and cached as the
    DERIVED PAIR (never a cached census -- a cached measurement behind a patched
    constant is the lru-cache failure this repo has already recorded). The self-check
    re-derives it fresh and asserts the cache matches, so the cache cannot drift."""
    global _CONSENSUS
    if _CONSENSUS is None:
        c = census()
        _CONSENSUS = (c["consensus_wiring"], c["consensus_basal"])
    return _CONSENSUS


def emit_regions(n_nodes=16, wiring=None, basal=None):
    """Stage-1 region emission (T043's consumer surface): per-node labels from the
    consensus network's stable states on the presentation grid (even n_nodes; the
    consensus is module-cached, so a consumer never pays the census). Labels in
    {"anterior", "posterior", "other", "cyclic"} -- on the current consensus network the
    settled state is a bijection of the fgf8 input bit, so only the first two occur
    (measured; the wider contract is kept for T041's graded regime, where the others
    become reachable). Cell-type emission is DEFERRED to T041 by design."""
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
    # ---- the consensus cache cannot drift ----------------------------------------------
    w_cached, b_cached = _consensus()
    if not (
        np.array_equal(w_cached, c["consensus_wiring"])
        and np.array_equal(b_cached, c["consensus_basal"])
    ):
        print("arealization self-check FAILED: the cached consensus differs from a fresh census")
        return 1
    # ---- clean emission is the exact half-field under the declared order ---------------
    labels, _ = emit_regions(16, wiring=w_cached, basal=b_cached)
    n_ant, n_post = labels.count("anterior"), labels.count("posterior")
    if not (n_ant == 8 and n_post == 8):
        print(
            f"arealization self-check FAILED: clean-field emission is not the exact "
            f"half-field map (anterior {n_ant}, posterior {n_post}, labels {labels})"
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
        f"{c['consensus_ties_at_pick']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
