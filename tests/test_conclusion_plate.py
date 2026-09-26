"""T090 (C5) -- the conclusion plate (specs/022 evidence/t090-conclusion-plate.md 2.4).

The reds R2-R5 of the pre-registration, each on its constructed input. R1 is the orchestrator's
RENDERED measurement in a browser and is not a test; everything here is brain-free except the one
case that needs a snapshot, and that one runs on a SCRATCH brain under a throwaway home.
"""

import re

import pytest

from snn_data_lake.gates import conclusion_protocol
from snn_data_lake.substrate import brain_telemetry

# A row shaped like a live one: two sentences, a dissociation input, a reading, a surface symbol.
CHARGED = (
    "the reported quantity tracks the named internal state rather than the prompt asking for it",
    "this does not license any claim that the system has experience of that state",
    "the scrambled-intermediate input, on which the instrument must read at or below its own null",
    0.42,
    "2026-09-16",
    ("widget_reading",),
)
NOT_MET = CHARGED[:3] + (None, "2026-09-16", ())


def test_a_charged_row_without_its_sentence_refuses_rather_than_serving_it():
    """R2. Mutation `serve_anyway` (the raise deleted from conclusion_rows): red.

    The registry's whole point is that a reading and the sentence saying what it licenses travel
    together. A blank sentence therefore cannot be served degraded -- the page would print the
    number alone, which is the exact thing refusal 3 forbids.
    """
    assert brain_telemetry.conclusion_rows({}, {"widget": CHARGED})  # the positive control first
    blank_refused = CHARGED[:1] + ("",) + CHARGED[2:]
    blank_licensed = ("",) + CHARGED[1:]
    for reg in ({"widget": blank_refused}, {"widget": blank_licensed}):
        with pytest.raises(ValueError) as e:
            brain_telemetry.conclusion_rows({}, reg)
        assert "widget" in str(e.value) and "sentence" in str(e.value)
    # whitespace is not a sentence either -- " " passed the first version of this check
    with pytest.raises(ValueError):
        brain_telemetry.conclusion_rows({}, {"widget": CHARGED[:1] + ("   ",) + CHARGED[2:]})
    # ...and a row that is not a 6-tuple at all is named rather than raising a bare unpack error
    with pytest.raises(ValueError):
        brain_telemetry.conclusion_rows({}, {"widget": ("too", "short")})


def test_a_not_met_row_never_reaches_the_page():
    """R3. Mutation `serve_all_rows` (the `reading is None` skip deleted): red.

    A NOT MET instrument prints NOTHING -- the standing refusal, rendered as an absence rather
    than as a row saying 'pending', which a reader would score as a result.
    """
    rows = brain_telemetry.conclusion_rows({}, {"quiet": NOT_MET, "widget": CHARGED})
    assert [r["instrument"] for r in rows] == ["widget"]
    assert brain_telemetry.conclusion_rows({}, {"quiet": NOT_MET}) == []
    # ORDER IS THE REGISTRY'S, not sorted: a page that re-sorted would put the newest row first
    # and read as a ranking. 'a' sorts before 'z'; the registry says otherwise.
    two = {"z_first": CHARGED, "a_second": CHARGED}
    assert [r["instrument"] for r in brain_telemetry.conclusion_rows({}, two)] == [
        "z_first",
        "a_second",
    ]


def test_the_live_value_travels_beside_the_reading_and_is_not_it():
    """The two quantities a surface must never confuse: the CHARGED reading is a campaign
    measurement on scratch brains, `live` is this daemon's own state key right now."""
    rows = brain_telemetry.conclusion_rows({"widget_reading": 7.0}, {"widget": CHARGED})
    assert rows[0]["reading"] == 0.42
    assert rows[0]["live"] == {"widget_reading": 7.0}
    # a symbol the brain does not publish reads None, never 0 -- an em-dash on the page
    assert brain_telemetry.conclusion_rows({}, {"widget": CHARGED})[0]["live"] == {
        "widget_reading": None
    }


def test_the_served_section_is_a_contract_key_whose_absence_is_a_finding():
    """The no-orphan-panel gate covers the new section: without it the CONSCIOUSNESS plate would
    draw an empty column indistinguishable from 'no row is charged yet'."""
    assert "conclusion" in brain_telemetry.verify_contract({})
    assert "conclusion" not in brain_telemetry.verify_contract({"conclusion": []})


def test_a_scratch_brains_snapshot_serves_todays_three_charged_rows():
    """The section as a browser receives it. Three rows are charged on this tree
    (T080/T081/T082); the other four read None and are absent."""
    from snn_data_lake.cortex.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    with throwaway_home():  # never the live lake
        brain = make_brain(seed=0, persist=False)
        for _ in range(12):
            brain.tick()
        snap = brain_telemetry.snapshot(brain)

    served = snap["conclusion"]
    assert [(e["instrument"], e["task"], e["reading"]) for e in served] == [
        ("temporal_binding", "T080", 0.2309),
        ("report_order", "T081", 1.0),
        ("social_cognition", "T082", 0.4437),
    ]
    for e in served:
        assert e["licensed"].strip() and e["refused"].strip() and e["dissociation"].strip()
        assert e["symbols"] and set(e["live"]) == set(e["symbols"])
    assert "conclusion" not in brain_telemetry.verify_contract(snap)


def test_the_gates_mechanism_clause_fires_where_the_text_clause_cannot_see():
    """R4. Mutation `text_only` (the `mechanism_findings` call removed from surface_findings): red.

    The text clause credits a licensed sentence sitting within three lines of a symbol in a tracked
    surface's SOURCE -- satisfiable by a comment, and blind to what is served. This clause asks the
    served path itself. Counted before the rule landed: 0 on the live registry, 1 on the blank.
    """
    assert conclusion_protocol.mechanism_findings() == []  # the live registry, clean today
    assert conclusion_protocol.mechanism_findings({"widget": CHARGED}) == []
    blanked = {"widget": ("",) + CHARGED[1:]}
    fired = conclusion_protocol.mechanism_findings(blanked)
    assert len(fired) == 1 and "widget" in fired[0], fired
    # IT REACHES THE GATE. Without this the clause could be green in its own function while
    # --assert never called it -- which is what the `text_only` mutation removes.
    assert any("widget" in x for x in conclusion_protocol.surface_findings(blanked, {}))
    assert conclusion_protocol.surface_findings({"widget": CHARGED}, {}) == []
    # ...and the blank pushes the real debt ABOVE the baseline, which is how --assert fails on it
    live = conclusion_protocol._registry_from_source()
    charged = next(n for n, r in live.items() if r[3] is not None)
    live_blank = {**live, charged: ("",) + live[charged][1:]}
    assert (
        len(conclusion_protocol.surface_findings(live_blank)) > conclusion_protocol.SURFACE_BASELINE
    )
    assert len(conclusion_protocol.surface_findings()) <= conclusion_protocol.SURFACE_BASELINE


def test_the_consciousness_plate_ships_outside_the_fixed_height_strip_and_wraps():
    """R5, DECLARED A RATCHET OF SOURCE TEXT and never the rendered red (obs #2195).

    It proves the page SHIPS a plate outside `#bottom` carrying a wrapping rule, and that the
    three C5 readings left the ACCESS fill. Whether every character of every sentence lands inside
    the plate's visible box is the RENDERED measurement at two widths (2.2,
    tests/instruments/t090_measure_plate.js), which no test in this suite can see: a CSS rule in a
    source string and a laid-out character are different claims, and this file can only make the
    first one. Mutation `nowrap` (`.c5{white-space:nowrap`): red.
    """
    from snn_data_lake.substrate import cockpit

    html = cockpit.COCKPIT_HTML
    assert ".c5{white-space:normal" in html
    head, sep, bottom = html.partition('<div id="bottom">')
    assert sep, "the page lost its #bottom row"
    assert 'id="wC5"' in head, "the plate must not live in the fixed-height #bottom grid row"
    assert 'id="wC5"' not in bottom
    # P2: one place per reading. The three C5 symbols no longer render in the ACCESS fill.
    for sym in ("crossmodal_bound_rate", "report_order_margin", "recursive_tom_false_beliefs"):
        assert f"'{sym}'" not in html, f"{sym} is still passed to a row() call"
    # and the plate reads the SERVED section rather than re-deriving anything from cognition
    assert "d.conclusion" in html


def test_the_widgets_column_scrolls_and_every_plate_is_its_own_content_height():
    """A SECOND RATCHET OF SOURCE TEXT beside R5, and never the rendered red (obs #2195).

    R5 proves the plate ships outside `#bottom` and wraps. This proves the COLUMN it ships in can
    be scrolled, that no plate in it can be squeezed below its own content, and that the plate
    carrying the in-view criterion is the column's FIRST child -- the three source facts the second
    reading's geometry rests on. Whether a character lands inside the visible box is still the
    rendered measurement at two widths (tests/instruments/t090_measure_plate.js). Mutations, each
    red: `no_scroll`, `no_gutter`, `shrinkable`, `reorder`.
    """
    from snn_data_lake.substrate import cockpit

    html = cockpit.COCKPIT_HTML
    rule = html.partition("#widgets{")[2].partition("}")[0]
    assert "overflow-y:auto" in rule and "overflow-x:hidden" in rule, rule
    assert "scrollbar-gutter:stable" in rule, rule
    block = html.partition('<div id="widgets">')[2].partition('<div id="chat"')[0]
    ids = re.findall(r'id="(w[A-Za-z0-9]+)"', block)
    assert len(ids) == 7 and ids[0] == "wC5", ids
    flexes = [f.strip() for f in re.findall(r'style="flex:([^"]*)"', block)]
    assert len(flexes) == 7 and set(flexes) == {"0 0 auto"}, flexes
