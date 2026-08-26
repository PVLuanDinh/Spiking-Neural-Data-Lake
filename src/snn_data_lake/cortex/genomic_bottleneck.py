"""
Genomic bottleneck -- the g-network: tag vectors + ONE small map (pre_tag, post_tag) -> weight.

WHAT THIS IS (specs/022 T044, FR-033's mechanism). Today a grown unit's incoming weights are a
RANDOM DRAW: the candidate pool inside `hippocampus/ng_snn.py::NGSNN.fit` is `(rand*2-1)*lambda`,
selected by residual reduction (reached on the live path through `neurogenesis_engine`'s refit)
-- an initializer with no structure carried from anywhere. FR-033 says grown units MUST receive
initial weights from a genomic-bottleneck map: a unit carries a small TAG (its declared
developmental identity), a feature carries a small tag, and initial weight w(j, i) is read from
ONE shared bilinear map w = f_tag_j^T M u_tag_i. The "genome" is M -- constant-size, shared by
every unit ever grown -- so the wiring RULE is compressed through the tags instead of every
weight being free at birth. This is the classic genomic-bottleneck framing (a compact rule
generating connectivity), built on the identity machinery this arc just landed: a unit's tag IS
its T043 placement (H01 laminar identity for cortex; the arealization GRN state elsewhere).

THE TAG LAYOUT IS DECLARED, DERIVED, AND GUARDED. UNIT tag (1 + len(LAYER_NAMES) + 1 + 4 slots,
and D_UNIT_TAG is DERIVED from that identity so a grown layer list cannot silently overwrite the
sign slot): [0] fractional depth/position; [1:7] the H01 layer one-hot (zeros for a GRN-arm unit
-- laminar identity is a cortical measurement); [_SIGN_SLOT] the Dale sign, where an UNMEASURED
sign (None, the T043 refusal) is encoded 0, never defaulted -- and a sign outside {+1, -1, None}
is REFUSED, not stored; [8:12] the settled GRN consensus state at the unit's region coordinate
(zeros for cortex). The GRN arm CROSS-CHECKS the placement label against the claimed region and
refuses a mismatch -- a hippocampal placement sold as medulla would otherwise carry another
region's identity silently. FEATURE tag (D_FEAT_TAG slots): [0] a constant BIAS 1.0; [1:] the
(D_FEAT_TAG-1)-bit binary code of j. The bias slot exists because the first draft used j/d there
and the T044 panel measured two defects in one slot: feature 0's tag was the ALL-ZERO vector, so
its initial weight was identically 0 under every genome (a hole the tag-structured ground truth
inherited, leaving no reachable red), and j/d is an EXACT linear combination of the bit slots at
every d, so the slot bought rank-deficiency and nothing else. d > MAX_FEATURES (= 2^(D_FEAT_TAG-1),
derived not typed) is REFUSED rather than allowed to collide silently. The bit code is a DECLARED
positional choice with no biology claim -- and it imposes a similarity geometry (bit-neighbors
are map-neighbors) that the eventual real-substrate A/B must treat as a variable, not a given.

WHAT THE FIT IS, AND WHAT IT IS NOT. `GenomicMap.fit` solves min_M ||W^T - F M U^T||_F by two
Moore-Penrose pseudoinverses (M = F+ W^T (U^T)+) -- ANALYTICAL, deterministic, no iteration, no
gradient anywhere (GRADIENT_CLAIM: DIRECT_SOLVE, the `NGSNN.fit` precedent). M is identifiable
only on the SPAN of the observed tags, and the fit RECORDS that span: `rank_u`, `rank_f` and the
identifiable-parameter count ride on the map, and `init_fanin` REFUSES a tag outside the fitted
span instead of silently returning its projection -- the panel measured the silent version
handing a cortical unit fan-in with correlation -0.05 against truth after a hippocampus-only fit.

WHAT IS MEASURED HERE AND WHAT IS DELIBERATELY NOT. main() proves the ALGEBRA on tag-structured
ground truth: exact reconstruction; a MIXED held-out set (cortical + GRN units) predicted from
tags alone under an asserted span precondition -- noting honestly that on ground truth built
inside the model class, held-out exactness is reconstruction + span membership, not independent
generalization; BOTH permutation nulls (unit tags and feature tags -- per-family attribution);
the out-of-span refusal red; a refusal battery discriminated BY MESSAGE (the T043 lesson: a bare
except proves nothing); and the compression arithmetic with its honest comparator: the genome is
96 numbers vs n*d free weights, but the INCUMBENT random pool stores ZERO numbers, so the ratio
is a claim about carrying structure, never a storage win over the incumbent. Tags are derived,
not stored, ONLY while a consumer keeps (region, seed, index) to re-derive placements.
NO CAPABILITY CLAIM IS MADE ABOUT THE REAL SUBSTRATE: whether the EARNED ng_snn weights are
tag-compressible is a measurement this module has not run. The wiring that would run it (swap
the pool inside `hippocampus/ng_snn.py::NGSNN.fit`, reached through `neurogenesis_engine`'s
refit) has NO task id in specs/022 tasks.md yet -- stated plainly rather than dressed as a
dated consumer. NULL DISPOSITION (P8 clause 5): if that A/B shows map-drawn fan-in does not
beat the random pool on the real earned substrate, this module is DELETED, not parked.

  PYTHONPATH=src python -m snn_data_lake.cortex.genomic_bottleneck
"""

import math

import numpy as np

from snn_data_lake.cortex.placement import LAYER_NAMES, place, region_prior

#: Unit-tag layout, DERIVED so the slot boundaries cannot drift: [depth][layer one-hot][sign][GRN state].
_N_GRN_STATE = 4
_SIGN_SLOT = 1 + len(LAYER_NAMES)
D_UNIT_TAG = 1 + len(LAYER_NAMES) + 1 + _N_GRN_STATE
#: Feature-tag width: [bias 1.0, (D_FEAT_TAG-1)-bit binary code of j]. Declared positional code.
D_FEAT_TAG = 8
#: DERIVED from the code width -- more features than the bits can name would collide SILENTLY,
#: so the constructor refuses them (a collision is a wrong answer wearing a right answer's shape).
MAX_FEATURES = 2 ** (D_FEAT_TAG - 1)

_LAYER_INDEX = {name: i for i, name in enumerate(LAYER_NAMES)}


def unit_tag(placed_unit, region, _prior=None):
    """The D_UNIT_TAG-slot tag for one placed unit -- its T043 developmental identity, re-encoded.

    `placed_unit` is one (depth_or_pos, label, sign) tuple from `placement.place(...)`, and
    `region` names the arm that produced it. Cortex fills the laminar slots (measured identity)
    and REFUSES a sign outside {+1, -1} -- None ("no measured sign") is a GRN property that
    must not appear on the cortical arm, and any other value is not a Dale sign. A GRN-arm
    unit fills the consensus-state slots (model identity), carries 0 in the sign slot BECAUSE
    its sign is None -- the refusal survives the encoding; 0 means UNMEASURED, never
    "excitatory-ish" -- and its placement LABEL is cross-checked against the claimed region's
    prior, so a placement from one GRN region sold under another's name is refused instead of
    silently wearing the wrong identity (T044 panel). `_prior` lets `unit_tags` hoist the
    region_prior lookup (it is per-region, not per-unit); callers pass the region name only.
    """
    pos, label, sign = placed_unit
    if not (isinstance(pos, (int, float)) and math.isfinite(pos)):
        raise ValueError(f"non-finite position {pos!r} -- a NaN depth would poison the genome")
    tag = np.zeros(D_UNIT_TAG, dtype=np.float64)
    tag[0] = float(pos)
    if region == "cortex":
        idx = _LAYER_INDEX.get(label)
        if idx is None:
            raise ValueError(
                f"{label!r} is not a cortical layer -- the cortex arm tags H01 laminar "
                f"identity only ({', '.join(LAYER_NAMES)}); White matter is not a birthplace"
            )
        if sign not in (1, -1):
            raise ValueError(
                f"cortical sign {sign!r} is not a Dale sign -- place() assigns +1/-1 at "
                "birth, and an unmeasured None belongs to the GRN arm, never to cortex"
            )
        tag[1 + idx] = 1.0
        tag[_SIGN_SLOT] = float(sign)
    else:
        if sign is not None:
            raise ValueError(
                f"a GRN-arm unit of {region!r} arrived with sign {sign!r} -- T043 refuses "
                "signs outside measured H01 cortex, so this input did not come from place()"
            )
        prior = _prior if _prior is not None else region_prior(region)
        if label != prior["label"]:
            raise ValueError(
                f"placement label {label!r} does not match region {region!r}'s prior label "
                f"{prior['label']!r} -- a unit sold under the wrong region would carry "
                "another region's identity silently"
            )
        tag[_SIGN_SLOT + 1 :] = np.asarray(prior["consensus_state"], dtype=np.float64)
    return tag


def feature_tag(j, d):
    """The D_FEAT_TAG-slot tag for input feature j of d: [bias 1.0, binary code of j]. A
    DECLARED CODE -- it claims position, nothing else. The bias slot means NO feature has the
    all-zero tag (the T044 panel measured feature 0's weight pinned at exactly 0 under every
    genome when this slot held j/d). Refuses d > MAX_FEATURES: past the code's capacity two
    features would share one identity silently."""
    if not 0 <= j < d:
        raise ValueError(f"feature index {j} outside [0, {d})")
    if d > MAX_FEATURES:
        raise ValueError(
            f"d={d} exceeds MAX_FEATURES={MAX_FEATURES}: the {D_FEAT_TAG - 1}-bit positional "
            "code would collide silently -- widen the code deliberately, never implicitly"
        )
    tag = np.zeros(D_FEAT_TAG, dtype=np.float64)
    tag[0] = 1.0
    for b in range(D_FEAT_TAG - 1):
        tag[1 + b] = (j >> b) & 1
    return tag


def feature_tags(d):
    """(d, D_FEAT_TAG) tag matrix for all d features."""
    return np.stack([feature_tag(j, d) for j in range(d)])


def unit_tags(placed, region):
    """(n, D_UNIT_TAG) tag matrix for a placement list from ONE region. Hoists the per-REGION
    prior lookup out of the per-unit loop (the panel measured 176 us/unit of pure GRN
    re-derivation without the hoist -- ~7 s per tagging pass at the repo's ~40k ceiling)."""
    prior = None if region == "cortex" else region_prior(region)
    return np.stack([unit_tag(p, region, _prior=prior) for p in placed])


class GenomicMap:
    """The one shared map: w(feature j, unit i) = f_tag_j^T M u_tag_i. M is the genome --
    D_FEAT_TAG x D_UNIT_TAG = 96 numbers, constant in the number of units ever grown. A map
    KNOWS WHAT IT WAS FITTED ON: `fit` records the tag-span basis and ranks, and `init_fanin`
    refuses a tag outside that span rather than returning its silent projection."""

    def __init__(self):
        self.M = np.zeros((D_FEAT_TAG, D_UNIT_TAG), dtype=np.float64)
        self.rank_u = None  # unit-tag span rank, recorded at fit
        self.rank_f = None  # feature-tag span rank, recorded at fit
        self._span = None  # orthonormal basis (rows) of the fitted unit-tag span

    @property
    def n_params(self):
        return int(self.M.size)

    @property
    def n_identifiable(self):
        """rank_f * rank_u -- how many of the 96 genome parameters the fitted tags actually
        pinned. The rest are zero by the minimum-norm property and carry NO claim."""
        if self.rank_u is None:
            return None
        return int(self.rank_f * self.rank_u)

    def weight(self, ftag, utag):
        """One weight, read raw from the genome -- the low-level read; `init_fanin` is the
        guarded FR-033 path and the one a consumer should call."""
        return float(np.asarray(ftag) @ self.M @ np.asarray(utag))

    def init_fanin(self, ftags, utag):
        """(d,) initial incoming weights for ONE unit from its tag -- the FR-033 read path.

        REFUSES an unfitted genome and a tag outside the fitted span: the T044 panel measured
        the silent alternative (fit on hippocampus-only tags, read a cortical unit) returning
        fan-in with correlation -0.05 against truth -- noise wearing a measurement's shape.
        INCOMING only, on purpose: function-preserving growth
        (`hippocampus/function_preserving_growth.py::SpikingGrowthLayer.add_unit_gradmax`, and
        the zero-init contract in `substrate/growable.py`'s primitives) requires a new unit's
        OUTGOING weights to be exactly zero at insertion, and this map does not touch that half.
        """
        if self._span is None:
            raise ValueError("unfitted genome -- fit() before reading initial weights")
        u = np.asarray(utag, dtype=np.float64)
        resid = u - self._span.T @ (self._span @ u)
        if np.linalg.norm(resid) > 1e-9 * (1.0 + np.linalg.norm(u)):
            raise ValueError(
                "tag outside the fitted span -- this genome never saw a unit like this one, "
                "and a silent projection would be noise wearing a measurement's shape; refit "
                "with the new tag family before reading it"
            )
        return np.asarray(ftags) @ self.M @ u

    def fit(self, W, ftags, utags):
        """min_M ||W^T - F M U^T||_F by two pseudoinverses: M = F+ W^T (U^T)+.

        W is (n_units, d) fan-in rows, F is (d, D_FEAT_TAG), U is (n, D_UNIT_TAG). ANALYTICAL
        and deterministic -- no iteration, no learning rate, no gradient (GRADIENT_CLAIM:
        DIRECT_SOLVE). Refuses empty and mis-shaped input with its OWN message (numpy's matmul
        ValueError satisfied the first draft's bare-except refusal check -- the T043 lesson,
        re-learned). Records rank_u / rank_f / the span basis, because the returned
        reconstruction RMSE is MAXIMALLY OPTIMISTIC exactly when the fit is maximally
        under-determined: a rank-2 tag family fits to 1e-15 while 9 of 12 tag columns of M
        stay zero. Returns the reconstruction RMSE over all n*d entries.
        """
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] == 0:
            raise ValueError(
                f"shape mismatch: W must be a non-empty (n_units, d) matrix, got {W.shape} -- "
                "an empty fit would return a NaN RMSE that reads as success"
            )
        F = np.asarray(ftags, dtype=np.float64)
        U = np.asarray(utags, dtype=np.float64)
        n, d = W.shape
        if F.shape != (d, D_FEAT_TAG) or U.shape != (n, D_UNIT_TAG):
            raise ValueError(f"shape mismatch: W {W.shape}, F {F.shape}, U {U.shape}")
        self.M = np.linalg.pinv(F) @ W.T @ np.linalg.pinv(U.T)
        _u_svd, s, vt = np.linalg.svd(U, full_matrices=False)
        tol = s.max() * max(U.shape) * np.finfo(np.float64).eps if s.size else 0.0
        self.rank_u = int((s > tol).sum())
        self._span = vt[: self.rank_u]
        self.rank_f = int(np.linalg.matrix_rank(F))
        R = U @ self.M.T @ F.T - W
        return float(np.sqrt((R * R).mean()))


# DIRECT_SOLVE, not NOT_A_RULE (T044 panel): NOT_A_RULE is the census's enumeration-imprecision
# meter, and this is a REAL closed-form solve, the `NGSNN.fit` precedent's family -- with the
# distinction that it solves for the GENERATOR M (initial-weight synthesis for units that have
# not fired), never for the network's own weights. No iteration, no error propagation, no
# transport; the e-prop invariant governs gradient flow through spiking dynamics, absent here.
GRADIENT_CLAIM = {
    "GenomicMap.fit": (
        "DIRECT_SOLVE",
        "closed-form bilinear least squares (two Moore-Penrose pseudoinverses) solving for "
        "the tag-to-weight GENERATOR M, not the network's weights; no iteration, no error "
        "propagation, no plasticity -- initial-weight synthesis, the NGSNN.fit direct-solve "
        "family",
    ),
}


def main():
    print("=" * 96)
    print("GENOMIC BOTTLENECK -- tag vectors + one small map (pre_tag, post_tag) -> weight")
    print("=" * 96)
    rng = np.random.default_rng(7)

    # REAL TAGS from the T043 surface: cortical units (measured laminar identity) plus two
    # GRN-arm regions (model identity) -- the tag population a real mixed-region growth would
    # produce, not a synthetic convenience.
    d = 32
    F = feature_tags(d)
    cort = place(48, seed=11)
    hipp = place(24, seed=12, region="hippocampus")
    medu = place(24, seed=13, region="medulla")
    U = np.vstack(
        [unit_tags(cort, "cortex"), unit_tags(hipp, "hippocampus"), unit_tags(medu, "medulla")]
    )
    n = U.shape[0]

    # Sign-slot honesty, and the zero-hole red: NO feature carries the all-zero tag (the bias
    # slot is what guarantees it -- under the first draft's j/d slot, feature 0's initial
    # weight was EXACTLY 0 for every genome and no check could red).
    assert set(np.unique(U[:48, _SIGN_SLOT])) <= {-1.0, 1.0}, "a cortical tag lost its sign"
    assert (U[48:, _SIGN_SLOT] == 0.0).all(), "a GRN tag fabricated a sign"
    assert all(feature_tags(k)[0].any() for k in (1, 2, 32, 128)), (
        "feature 0's tag is all-zero -- its initial weight would be pinned at 0 under every "
        "genome, silently"
    )
    print(f"  green  tags: {n} units x {D_UNIT_TAG} slots from the T043 surface; sign honesty")
    print("  RED    no all-zero feature tag at any d (feature 0's weight is NOT pinned to 0)")

    # 1. ALGEBRA -- exact reconstruction on tag-structured ground truth, with BOTH rank
    # disclosures: the identifiable-parameter count is what the RMSE alone hides.
    M_star = rng.standard_normal((D_FEAT_TAG, D_UNIT_TAG))
    W_star = U @ M_star.T @ F.T  # (n, d) tag-structured fan-in
    g = GenomicMap()
    rmse = g.fit(W_star, F, U)
    assert rmse < 1e-9, f"exact tag-structured reconstruction failed (RMSE {rmse:.2e})"
    scale = float(np.sqrt((W_star * W_star).mean()))
    assert np.abs(W_star[:, 0]).max() > 0, "feature 0's column is dead in the ground truth"
    dM = float(np.abs(g.M - M_star).max())
    print(
        f"  green  exact reconstruction: RMSE {rmse:.2e} (scale {scale:.3f}); tag ranks "
        f"U {g.rank_u} of {D_UNIT_TAG}, F {g.rank_f} of {D_FEAT_TAG} -> {g.n_identifiable} of "
        f"{g.n_params} genome params identifiable; |M-M*|_max {dM:.3f} -- null space unclaimed"
    )

    # 2. HELD-OUT UNITS, MIXED on purpose (the panel measured the first draft holding out an
    # all-medulla block whose tags differ in ONE uninformative scalar): 6 cortical + 6 medulla
    # units predicted from tags alone. HONESTY NOTE: on ground truth built inside the model
    # class, held-out exactness IS reconstruction + span membership -- the span assert is the
    # load-bearing precondition, not a separate generalization result.
    out_idx = list(range(42, 48)) + list(range(90, 96))
    in_idx = [i for i in range(n) if i not in out_idx]
    g2 = GenomicMap()
    g2.fit(W_star[in_idx], F, U[in_idx])
    span_ok = np.linalg.matrix_rank(U) == np.linalg.matrix_rank(U[in_idx])
    assert span_ok, "held-out tags left the training span -- the prediction would be vacuous"
    pred = np.stack([g2.init_fanin(F, U[i]) for i in out_idx])
    err = float(np.sqrt(((pred - W_star[out_idx]) ** 2).mean()))
    assert err < 1e-9, f"held-out unit fan-in prediction failed (RMSE {err:.2e})"
    print(
        f"  green  held-out units (6 cortex + 6 medulla, layer/sign slots exercised): "
        f"predicted from tags alone, RMSE {err:.2e}"
    )

    # 3. OUT-OF-SPAN REFUSAL -- the deployment-shaped red: fit on ONE GRN region's tags
    # (rank-2 family), then ask for a cortical unit. The first draft returned fan-in with
    # correlation -0.05 against truth here, silently; now it refuses.
    g_h = GenomicMap()
    g_h.fit(W_star[48:72], F, U[48:72])  # hippocampus block only
    try:
        g_h.init_fanin(F, U[0])  # a cortical tag
        raise AssertionError("an out-of-span tag was read instead of refused")
    except ValueError as e:
        assert "outside the fitted span" in str(e), "refused by the wrong branch"
    print("  RED    out-of-span tag REFUSED (hippocampus-only genome asked for a cortical unit)")

    # 4. PERMUTATION NULLS, one per tag family (P8 per-item attribution): permuting WHICH unit
    # carries WHICH tag, and separately WHICH feature carries WHICH tag, each collapses the
    # fit -- so BOTH families carry structure, and neither control touches the solver.
    perm = rng.permutation(len(in_idx))
    g3 = GenomicMap()
    g3.fit(W_star[in_idx], F, np.asarray(U[in_idx])[perm])
    pred_null = np.stack([g3.init_fanin(F, U[i]) for i in out_idx])
    err_null = float(np.sqrt(((pred_null - W_star[out_idx]) ** 2).mean()))
    # The bar is 0.1*scale -- ONE clause with teeth (the first draft's 100x-ratio clause
    # evaluated to err_null > 1e-10, vacuous beside this one).
    assert err_null > 0.1 * scale, (
        f"the permuted-UNIT-tag null did not degrade prediction ({err_null:.3f} vs {err:.2e})"
    )
    fperm = rng.permutation(d)
    g3f = GenomicMap()
    rmse_fnull = g3f.fit(W_star, F[fperm], U)
    assert rmse_fnull > 0.1 * scale, (
        f"the permuted-FEATURE-tag null did not degrade reconstruction ({rmse_fnull:.3f})"
    )
    print(
        f"  RED    permutation nulls: unit tags {err_null:.3f}, feature tags {rmse_fnull:.3f} "
        f"(structured: {err:.2e}) -- both families carry the map"
    )

    # 5. Degeneracy control (NOT a method red -- it polices W*, not the fit): a random genome
    # of the right shape reconstructs nothing, so the exactness above is not W* being trivial.
    g4 = GenomicMap()
    g4.M = rng.standard_normal(g4.M.shape)
    R4 = U @ g4.M.T @ F.T - W_star
    err_rand = float(np.sqrt((R4 * R4).mean()))
    assert err_rand > 0.1 * scale, "a RANDOM genome reconstructed the weights -- W* degenerate"
    print(f"  green  degeneracy control: random genome RMSE {err_rand:.3f} (fit {rmse:.2e})")

    # 6. COMPRESSION, with the honest comparator: 96 numbers vs n*d free weights -- but the
    # INCUMBENT (ng_snn's random pool) stores ZERO numbers, so this is a claim about carrying
    # structure at constant size, never a storage win over the incumbent. Tags are derived,
    # not stored, ONLY while the consumer keeps (region, seed, index) to re-derive placements.
    full = n * d
    print(
        f"  green  compression: genome {g.n_params} params vs {full} free weights "
        f"({full / g.n_params:.1f}x at n={n}, d={d}; grows with n) -- vs storing every "
        "initial weight; the incumbent random pool stores none, so this is structure, not "
        "a storage win over it"
    )

    # 7. REFUSAL BATTERY, discriminated BY MESSAGE (the T043 lesson, which the first draft of
    # THIS check re-broke: numpy's own matmul ValueError satisfied a bare except with the
    # module's shape guard deleted).
    refusals = [
        (lambda: feature_tag(0, MAX_FEATURES + 1), "MAX_FEATURES"),
        (lambda: feature_tag(40, 32), "outside [0"),
        (lambda: g.fit(W_star, F[:5], U), "shape mismatch"),
        (lambda: g.fit(np.zeros((0, d)), F, np.zeros((0, D_UNIT_TAG))), "shape mismatch"),
        (lambda: GenomicMap().init_fanin(F, U[0]), "unfitted genome"),
        (lambda: unit_tag((0.5, "Layer 4", None), "cortex"), "not a Dale sign"),
        (lambda: unit_tag((0.5, "Layer 4", 7), "cortex"), "not a Dale sign"),
        (lambda: unit_tag((0.5, "White matter", 1), "cortex"), "not a cortical layer"),
        (lambda: unit_tag((float("nan"), "Layer 4", 1), "cortex"), "non-finite"),
        (lambda: unit_tag((0.5, "anterior", 1.0), "medulla"), "arrived with sign"),
        (lambda: unit_tag((0.5, "anterior", None), "medulla"), "does not match region"),
    ]
    for bad, needle in refusals:
        try:
            bad()
            raise AssertionError(f"a refusal path accepted bad input (wanted {needle!r})")
        except ValueError as e:
            assert needle in str(e), f"refused by the WRONG branch: {needle!r} not in {e}"
    print(f"  RED    {len(refusals)} refusals, each discriminated by its own message")

    # 8. Determinism -- BY CONSTRUCTION (pure numpy algebra, no rng in fit); the re-run below
    # is a smoke check of that construction, not an independent measurement.
    g5 = GenomicMap()
    g5.fit(W_star, F, U)
    assert (g5.M == g.M).all(), "the analytical fit is not deterministic"
    print("  green  determinism (by construction; smoke re-run bit-identical)")

    print()
    print(
        "DISCLOSURE: everything above is ALGEBRA on tag-structured ground truth. Whether the "
        "REAL earned substrate (ng_snn's W) is tag-compressible is UNMEASURED here. The wiring "
        "that would measure it -- swapping the candidate pool inside "
        "hippocampus/ng_snn.py::NGSNN.fit, reached through neurogenesis_engine's refit -- has "
        "NO task id in specs/022 tasks.md yet; until it lands, the random-uniform pool remains "
        "the live initializer, undisplaced. The A/B must also treat the 7-bit positional code "
        "as a variable (it imposes a similarity geometry this module chose, not measured). "
        "NULL DISPOSITION: if map-drawn fan-in does not beat the random pool on the real "
        "earned substrate, this module is DELETED, not parked."
    )
    print(
        "self-check OK: g-network algebra proven (exact + mixed held-out), out-of-span "
        "refusal, two permutation nulls, message-discriminated refusals, determinism; "
        f"genome {GenomicMap().n_params} params, {len(refusals)} refusal reds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
