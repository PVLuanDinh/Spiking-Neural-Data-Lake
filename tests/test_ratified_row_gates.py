"""The three instruments the owner's 2026-09-06 rows owe (specs/022 X1 / L5 / C5, tasks T088-T090;
evidence/gates-for-the-ratified-rows.md). Brain-free: each gate's constructed reds must fire, each
positive control must pass, and the count each produces on today's tree is asserted, so a silent
change is caught."""

from snn_data_lake.substrate import capability_bar, conclusion_protocol, immortal_retention


def test_the_capability_bar_catches_a_claim_without_its_measurement():
    lp = "a protocol long enough to be a protocol"
    bp = "evidence/x.md, commit abc1234"
    mh = "measured here: on this host, protocol recorded 2026-09-06"
    assert capability_bar.findings({}, {}) == []
    assert capability_bar.findings({"t": ("short", 1.0, mh, 0.5, bp, "2026-09-06", "below")}, {})
    assert capability_bar.findings(
        {"t": (lp, 1.0, "cited: a paper", 0.5, bp, "2026-09-06", "below")}, {}
    )
    assert capability_bar.findings({"t": (lp, None, "", 0.5, bp, "2026-09-06", "at parity")}, {})
    refused = {
        "widget": (
            lp,
            None,
            "",
            0.5,
            bp,
            "2026-09-06",
            "REFUSED: no human number under a matching protocol",
        )
    }
    assert capability_bar.findings(refused, {}) == []
    assert capability_bar.findings(
        refused, {"docs/PITCH.md": "the brain is superhuman on the widget task"}
    )


def test_the_capability_registry_is_empty_and_clean_today():
    assert capability_bar._registry_from_source() == {}
    assert capability_bar.findings() == []


def test_the_retention_label_is_consistent_and_nothing_reclaims_without_it():
    assert immortal_retention.label_findings() == []
    assert immortal_retention.round_trip() == []
    bad, scratch = immortal_retention.reclaimers(with_scratch=True)
    assert bad == [], bad
    assert len(scratch) >= 1  # the scratch-home deletions are EXCLUDED and still listed
    imm = immortal_retention.immortal_tags()
    assert (
        len(imm) == 9 and 0x03 not in imm and 0x08 not in imm
    )  # the grown substrate is COMPACTABLE


def test_the_retention_reds_fire():
    disp = {0x01: ("x", "y")}
    good = {0x01: ("IMMORTAL", False, "a reason long enough to be a reason")}
    assert immortal_retention.label_findings(good, disp) == []
    assert immortal_retention.label_findings(
        {0x01: ("IMMORTAL", True, "a reason long enough to be a reason")}, disp
    )
    assert immortal_retention.label_findings({0x01: ("IMMORTAL", False, "short")}, disp)
    assert immortal_retention.label_findings(
        {0x01: ("KEEPISH", False, "a reason long enough to be a reason")}, disp
    )
    assert immortal_retention.label_findings(good, {**disp, 0x77: ("x", "y")})


def test_every_consciousness_instrument_owns_its_licensed_and_refused_sentences():
    reg = conclusion_protocol._registry_from_source()
    assert set(reg) == set(conclusion_protocol.INSTRUMENTS)
    for name, row in reg.items():
        licensed, refused, dissociation, reading, _date, symbols = row
        assert len(licensed) >= conclusion_protocol._MIN_SENTENCE, name
        assert len(refused) >= conclusion_protocol._MIN_SENTENCE, name
        assert len(dissociation) >= conclusion_protocol._MIN_SENTENCE, name
        assert reading is None, (
            name
        )  # C4 is unbuilt; a reading without its instrument would be a fabrication
        assert isinstance(symbols, tuple)
    assert conclusion_protocol.asserted_findings() == []


def test_the_conclusion_refusals_are_enforced_not_merely_written(tmp_path):
    ok = dict.fromkeys(
        conclusion_protocol.INSTRUMENTS,
        (
            "the reported quantity tracks the named internal state rather than the prompt asking",
            "this does not license any claim that the system has experience of that state",
            "the scrambled-intermediate input, on which the instrument must read at or below null",
            None,
            "2026-09-06",
            ("widget_reading",),
        ),
    )
    assert conclusion_protocol.row_findings(ok) == []
    assert conclusion_protocol.surface_findings(ok, {"docs/PITCH.md": "the brain is conscious"})
    assert conclusion_protocol.surface_findings(ok, {"docs/PITCH.md": "widget_reading is 0.42"})
    assert (
        conclusion_protocol.surface_findings(
            ok, {"docs/PITCH.md": "an integration point for a sensor"}
        )
        == []
    )
    (tmp_path / "aggregator.py").write_text(
        "def s(integration, agency):\n    return (integration + agency) / 2\n"
    )
    assert conclusion_protocol.aggregate_findings(tmp_path)


def test_the_surface_debt_is_at_or_under_its_measured_baseline():
    surf = conclusion_protocol.surface_findings()
    assert len(surf) <= conclusion_protocol.SURFACE_BASELINE, surf
