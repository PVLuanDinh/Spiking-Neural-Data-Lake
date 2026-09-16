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


def test_the_retention_label_is_consistent_and_every_reclaimer_is_accounted_for():
    assert immortal_retention.label_findings() == []
    bad, scratch = immortal_retention.reclaimers(with_scratch=True)
    # ZERO again, and this time it is a zero that has been earned rather than assumed: the census
    # DID find substrate/home.py::consolidate once the rewrite arm landed, row L6 closed the gap at
    # both ends, and stale_known() is what forced the debt entry to be paid out. The detector is
    # proven live by test_a_rewrite_shaped_reclaimer_is_visible_to_the_census below.
    assert bad == [], bad
    assert immortal_retention.KNOWN_UNLABELED == {}, immortal_retention.KNOWN_UNLABELED
    assert immortal_retention.unlisted_reclaimers() == [], immortal_retention.unlisted_reclaimers()
    assert immortal_retention.stale_known() == [], immortal_retention.stale_known()
    assert len(scratch) >= 1  # the scratch-home deletions are EXCLUDED and still listed
    imm = immortal_retention.immortal_tags()
    assert (
        len(imm) == 9 and 0x0C not in imm and 0x03 not in imm and 0x08 not in imm
    )  # the grown substrate is COMPACTABLE. 0x0C joined the IMMORTAL class on 2026-09-08
    # (specs/022 T066) and LEFT IT on 2026-09-09 with the paging layer it was minted for, which
    # T068 measured at a 35.2x tick cost against a 3.0x bound and deleted on its null
    # (evidence/s5-paging-null.md 3.4/3.5); 10 -> 9. Removing a tag from the IMMORTAL class is
    # normally the dangerous direction, so the measurement that licenses it is named here rather
    # than left in the commit: no lake has ever held a 0x0C or 0x0D record -- the live home has no
    # `segments/` and no `lake_index.db` -- so no durable record changes class, and "never
    # forgets" is untouched.
    # The count is PINNED, never derived from immortal_tags() -- immortal_tags() reads RETENTION,
    # so an expected count derived from it would be a tautology and this detector would be zero.


def test_the_round_trip_crosses_a_process_boundary_and_its_red_is_reachable():
    # The red FIRST: a green from an arm that cannot fail proves nothing -- this arm
    # shipped that way once, and the panel of 2026-09-06 is what caught it.
    tag = immortal_retention.immortal_tags()[0]
    assert immortal_retention.round_trip(withhold=tag), "the withheld-record red did not fire"
    assert immortal_retention.round_trip() == []


def test_a_rewrite_shaped_reclaimer_is_visible_to_the_census(tmp_path):
    # The shape this repo actually uses, and the one the first shipped scan could not see.
    (tmp_path / "m.py").write_text(
        "from snn_data_lake.substrate.home import _T_FACT\n"
        "def consolidate(log, keep):\n"
        "    log.rewrite([_T_FACT + r for r in keep])\n",
        encoding="utf-8",
    )
    assert ("m.py", "consolidate") in immortal_retention.reclaimers(tmp_path)


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
        # A reading exists ONLY where its C4 instrument task is marked [X] and its pin exists on
        # the tree -- a reading without its instrument would be a fabrication. Until 2026-09-16
        # every row read None ("C4 is unbuilt"); T080 charged temporal_binding from
        # tools/t080_temporal_binding.json, so the premise is now checked per row, not assumed.
        if reading is not None:
            task = conclusion_protocol.C4_TASK.get(name)
            assert conclusion_protocol._task_done(task), (name, task)
            assert symbols, name
            pins = list((conclusion_protocol.REPO / "tools").glob(f"{task.lower()}_*.json"))
            assert pins, (name, task)
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


def test_refusal_three_arms_itself_when_a_c4_instrument_task_lands():
    # ONE of seven rows (attention_schema, T078 still [ ]) declares no surface symbol today -- this
    # header once read "six of seven", inverted on the head it was written for (the T082 panel,
    # 2026-09-16); the arm is CONSTRUCTED so the refusal's red stays reachable whatever the census.
    assert conclusion_protocol._task_done("T034") is True  # a task marked [X] in tasks.md
    # T078 is the C4 task still unmarked; T077 read [X] at f27f436 (2026-09-15) and this line was
    # RED on that head for a day -- the panel found it (evidence/t077-metacognition.md 3.5).
    assert conclusion_protocol._task_done("T078") is False  # a C4 task still unbuilt
    assert conclusion_protocol._task_done(None) is False
    # The fixture row must be one that declares NO symbol. `agency` served until T079 read NOT MET
    # (2026-09-15, evidence/t079-agency.md 3.4) and named `agency_controllability` printed
    # nowhere -- T077's shape; `temporal_binding` (T080, unbuilt) is the next symbol-less row.
    assert conclusion_protocol._task_done("T079") is True  # marked [X] NOT MET, its symbol named
    # T080 read PASS (2026-09-16, evidence/t080-temporal-binding.md 3.4): its row carries a reading
    # AND a symbol, so `report_order` (T081, unbuilt) became the next symbol-less row.
    assert conclusion_protocol._task_done("T080") is True
    # T081 read PASS too (2026-09-16, evidence/t081-report-order.md 2.6; pin
    # tools/t081_report_order.json): its row now carries reading 1.0 AND `report_order_margin`.
    assert conclusion_protocol._task_done("T081") is True
    # THE ARM IS A CONSTRUCTED ROW SINCE T082 (2026-09-16, evidence/t082-social-cognition.md 2.6:
    # "in every branch the arming arm is rewritten to a constructed row"). It used to point a REAL
    # symbol-less row at a complete task, so it had to be RE-POINTED on every charge -- `agency` ->
    # `temporal_binding` -> `report_order` -> `social_cognition`, four hand-offs in two days, and
    # the T079 panel found the twin self-check RED for a day on one of them. `social_cognition` is
    # CHARGED in this commit (reading 0.4437, symbol `recursive_tom_false_beliefs`), so the red is
    # BUILT here instead: its own one-name instrument tuple and its own symbol-less row, with no
    # live row's state in it and no hand-off left to forget.
    #
    # T082's own flip rides in THIS commit and not in the dated-notes commit that follows, because
    # `test_every_consciousness_instrument_owns_its_licensed_and_refused_sentences` above refuses a
    # reading whose C4 task is still [ ] -- the charge and the flip are bound by that gate, and
    # splitting them would commit a red in whichever order they were split.
    assert conclusion_protocol._task_done("T082") is True
    name = "widget"  # deliberately NOT an INSTRUMENTS name -- this row lives only inside this test
    row = (
        "the reported quantity tracks the named internal state rather than the prompt asking",
        "this does not license any claim that the system has experience of that state",
        "the scrambled-intermediate input, on which the instrument must read at or below null",
        None,
        "2026-09-06",
        (),
    )
    one = (name,)
    fired = conclusion_protocol.row_findings({name: row}, instruments=one, tasks={name: "T034"})
    assert any("T034" in x and name in x for x in fired), fired
    # TWO controls on the SAME constructed row, so the red is attributable to the missing symbol
    # and not to the arming: an unarmed id fires nothing, and the row WITH a symbol fires nothing
    # even while its task is complete.
    assert conclusion_protocol.row_findings({name: row}, instruments=one, tasks={name: None}) == []
    with_symbol = row[:5] + (("widget_reading",),)
    assert (
        conclusion_protocol.row_findings({name: with_symbol}, instruments=one, tasks={name: "T034"})
        == []
    )
    # ...and the LIVE registry stays clean when no task is armed at all.
    unarmed = {k: None for k in conclusion_protocol.INSTRUMENTS}
    assert conclusion_protocol.row_findings(tasks=unarmed) == []


def _mod(tmp_path, name, body):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_a_label_read_is_credited_per_tag_not_per_function(tmp_path):
    """The shape that slipped through L6's first landing: the label read for ONE tag
    while a helper one hop away drops ANOTHER -- home.consolidate passed the census that
    way while _prune_state_history dropped 0x07."""
    head = (
        "from snn_data_lake.substrate.home import _T_FACT, _T_STATE\n"
        "from snn_data_lake.substrate.lake_engine import retention_of\n"
    )
    tail = (
        "def consolidate(log, keep):\n"
        "    if not retention_of(_T_FACT[0])[1]:\n"
        "        pass\n"
        "    log.rewrite(_prune(log.replay()) + [_T_FACT + k for k in keep])\n"
    )
    bad = _mod(
        tmp_path / "a",
        "m.py",
        head + "def _prune(p):\n    return [x for x in p if x[:1] != _T_STATE]\n" + tail,
    )
    good = _mod(
        tmp_path / "b",
        "m.py",
        head + "def _prune(p):\n"
        "    if not retention_of(_T_STATE[0])[1]:\n"
        "        return list(p)\n"
        "    return [x for x in p if x[:1] != _T_STATE]\n" + tail,
    )
    assert ("m.py", "consolidate") in immortal_retention.reclaimers(bad)
    assert ("m.py", "consolidate") not in immortal_retention.reclaimers(good)


def test_a_runtime_tag_is_credited_for_reading_the_label_at_all(tmp_path):
    """A function whose tag is a PARAMETER touches no _T_* symbol, so the per-tag subset
    test alone would read its empty set as non-compliant. The disclosed hole, pinned here
    so it cannot widen silently."""
    (tmp_path / "m.py").write_text(
        "from snn_data_lake.substrate.lake_engine import retention_of\n"
        "def compact(path, tag):\n"
        "    if retention_of(tag)[0] == 'IMMORTAL':\n"
        "        return 0\n"
        "    (path / 'lake.bin').unlink()\n",
        encoding="utf-8",
    )
    assert immortal_retention.reclaimers(tmp_path) == []


def test_the_state_class_carries_its_stated_exemption_and_the_prune_reads_it():
    from snn_data_lake.substrate import home as home_mod
    from snn_data_lake.substrate import lake_engine as le

    label, reclaimable, reason = le.retention_of(0x07)
    assert (label, reclaimable) == ("IMMORTAL", False)
    assert "SUPERSEDED" in reason  # what licenses the prune, stated in the registry

    def frame(name, seq, blob):
        return (
            home_mod._T_STATE
            + bytes([len(name)])
            + name.encode("ascii")
            + int(seq).to_bytes(4, "little")
            + blob
        )

    probe = [frame("n", 1, b"v1"), frame("n", 2, b"v2"), b"\x04foreign"]
    assert len(home_mod._prune_state_history(probe)) == 2  # superseded frame dropped, foreign kept
    was = le.RETENTION[0x07]
    le.RETENTION[0x07] = (was[0], False, "no exemption stated here")
    try:
        assert len(home_mod._prune_state_history(probe)) == 3  # the label stops it dead
    finally:
        le.RETENTION[0x07] = was


def test_the_refusal_has_its_own_type_so_the_counter_means_what_it_says():
    from snn_data_lake.substrate.home import ImmortalRefusal

    # A ValueError subclass, so existing catchers still catch it -- and a distinct type,
    # so the sleep tick counts the guard, not a corrupt lake or a malformed sidecar.
    assert issubclass(ImmortalRefusal, ValueError)
    assert not isinstance(ValueError("x"), ImmortalRefusal)
