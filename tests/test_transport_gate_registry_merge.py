"""FR-019: `eprop_transport_gate.EXEMPT` is retired INTO the `GRADIENT_CLAIM` registry (P2).

Exactly ONE hand-maintained record of a symbol's gradient disposition may survive. Two records is
the additive shape P2 names, and it is the shape that lets an allowlist drift out of agreement with
the declaration beside the code.

The merge is NOT a rename. The two records are keyed DIFFERENTLY, and that difference bites:

    EXEMPT       keyed `reasoning.gf.train`  -- module path + the ENCLOSING FUNCTION name
    GRADIENT_CLAIM keyed `MLP.train`         -- the QUALIFIED symbol

so a naive lookup silently misses. Worse, resolving by "last dotted component" is AMBIGUOUS on the
real tree: measured, 4 modules carry the same bare method name on two or more classes, and 2 of
those disagree about the disposition -- `latent_q` (LOCAL vs CONTROL) and `reasoning_curriculum`
(CONTROL vs LOCAL). A resolver that picks one silently would be wrong in both.
"""

import importlib

etg = importlib.import_module("snn_data_lake.gates.eprop_transport_gate")
gate = importlib.import_module("snn_data_lake.gates.gradient_claim_gate")


def test_the_owner_adjudication_resolves_through_the_registry():
    """FR-019a. The one live exemption must still be found, keyed the way the registry keys it."""
    disp, reason = etg.resolve_disposition("reasoning.gf", "train")
    assert disp == gate.CONTROL, disp
    assert "Gc arm" in reason or "control" in reason.lower(), reason

    # NEGATIVE CONTROL: a symbol with no declaration must resolve to None, or "found" means nothing.
    assert etg.resolve_disposition("reasoning.gf", "zz_no_such_function") is None
    assert etg.resolve_disposition("zz.no.such.module", "train") is None


def test_an_ambiguous_bare_name_is_refused_rather_than_guessed():
    """T068. The keying hazard, driven on the REAL modules where it bites.

    `latent_q` declares `LatentQ.update` LOCAL and `_TabularQ.update` CONTROL. A finding has only
    the bare `update`, so the correct answer is REFUSE -- not "pick the first" and not "pick LOCAL".
    A resolver that guesses here would hand the gate a disposition the tree does not support.
    """
    for rel, fn in (
        ("basal_ganglia.latent_q", "update"),
        ("cortex.reasoning_curriculum", "train_premise"),
    ):
        try:
            got = etg.resolve_disposition(rel, fn)
        except etg.AmbiguousDisposition as exc:
            assert fn in str(exc), exc
            continue
        raise AssertionError(f"{rel}.{fn} resolved to {got!r} instead of refusing an ambiguous key")

    # CONTROL: a bare name colliding on several symbols that AGREE is NOT ambiguous -- refusing
    # there would make the resolver useless on the majority case.
    disp, _ = etg.resolve_disposition("readout.growable_readout", "train_step")
    assert disp == gate.APPROXIMATES_GRADIENT, disp


def test_an_exemption_blessing_nothing_still_fails_loudly():
    """FR-019a / T064. The honesty property is PORTED, not assumed to survive the port.

    `stale_exempt` caught an allowlist row matching no real finding. Moving the record into
    the registry must carry that property with it -- and the property needs its own red, because an
    implementation that returns `[]` unconditionally passes any "nothing is stale" assertion.
    """
    violations = etg.scan()[0] if isinstance(etg.scan(), tuple) else etg.scan()
    assert not etg.stale_exemptions(violations), (
        f"a declared exemption matches no finding: {etg.stale_exemptions(violations)}"
    )

    # THE REACHABLE RED: a declaration claiming an exemption for a symbol that produces no finding.
    planted = etg.stale_exemptions(violations, extra={"zz.nowhere.function": gate.CONTROL})
    assert planted, "the staleness check did not fire on a planted key -- it blesses anything"
    assert any("zz.nowhere" in p for p in planted), planted


def test_the_transport_gate_census_did_not_move_when_it_adopted_the_registry():
    """FR-019b / SC-015 / T063. The merge must be a swap of RECORD, never a change of VERDICT.

    A test cannot run the pre-merge code, so the incumbent's census is PINNED here from the run
    captured before the swap:

        GOVERNED: 0   UNGOVERNED: 0   EXEMPT: 1   SWAP gov/ung: 0/1

    The `EXEMPT: 1` is the load-bearing number. T068's keying hazard, had it bitten, would have
    driven it to 0 and GOVERNED to 1 -- the owner's grant would have silently stopped applying and
    the finding would have re-appeared as an unexcused governed violation. Pinning the count is what
    makes that failure loud instead of a quiet re-classification.
    """
    violations = etg.scan()
    if isinstance(violations, tuple):
        violations = violations[0]
    exempt = [v for v in violations if etg.claims_transport_exemption(v[0], v[1])]
    gov = [v for v in violations if v[5] and not etg.claims_transport_exemption(v[0], v[1])]
    ungov = [v for v in violations if not v[5] and not etg.claims_transport_exemption(v[0], v[1])]

    assert len(gov) == 0, f"a GOVERNED transport finding appeared: {gov}"
    assert len(ungov) == 0, f"an UNGOVERNED transport finding appeared: {ungov}"
    assert len(exempt) == 1, (
        f"expected exactly 1 resolved exemption, got {len(exempt)}: {exempt}. Zero means the "
        "registry lookup silently stopped matching the owner's grant (the T068 keying hazard)."
    )
    # And it must be the SAME grant, not a different symbol that happens to resolve.
    rel, fn = exempt[0][0], exempt[0][1]
    assert rel.endswith("gf") and fn == "train", (rel, fn)
    assert etg.resolve_disposition(rel, fn)[0] == gate.CONTROL
