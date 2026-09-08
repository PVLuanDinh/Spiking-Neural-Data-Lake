"""P0's honesty guard as a gate (owner rule 2026-09-02, specs/022 PLAN-FINAL Part C / O4).

A unit COUNT is reported to the owner as units ADMITTED, never as growth. The rule was
time-bound to SC-012a (accuracy rises with unit count under a capped-admission grid); SC-012a
FIRED 2026-09-07 on a QUALIFIED reading (accuracy rises with admitted units; forced width does
not beat the uncapped incumbent; nothing reaches the persistence floor), and on 2026-09-08 the
rule was RE-BASED rather than lifted (PLAN-FINAL Part C, the dated decision): a count is labelled
growth only when that count is shown to change an outcome the brain is measured on -- P0's
"growth is not improvement". The row that shows the improvement names this gate as the one it
retires. Two owner-facing surfaces carry the count: the cockpit row and the PROVENANCE row
rendered as its tag.

Red counted on the pre-change tree (gate-before-rule): the cockpit row read 'grown units'
and the PROVENANCE row said nothing about admission -> both tests FAILED before the wording
landed. The review panel then showed the first draft was a spelling tripwire (a second row
labelled 'growth' beside the value passed it), so the label is bound to the VALUE here: the
one row that renders `s.grown_units` must carry the admitted label and its PROVENANCE key,
and no label on the page may say grow/growth at all.
"""

import re

from snn_data_lake.substrate import brain_telemetry, cockpit

# row('<label>', s.grown_units[, <decimals>[, '<provenance key>']])
ROW_CALL = re.compile(r"row\('([^']*)',\s*s\.grown_units(?:,\s*([^,)]+))?(?:,\s*'([^']*)')?\)")
LABEL_CALL = re.compile(r"(?:row|sub|big)\('([^']*)'")


def test_the_count_row_is_labelled_admitted_and_tagged_with_its_provenance():
    calls = ROW_CALL.findall(cockpit.COCKPIT_HTML)
    assert len(calls) == 1, f"exactly one row renders s.grown_units, found {calls}"
    label, _decimals, key = calls[0]
    assert label == "units admitted", label
    assert key == "grown_units", "the row must carry its PROVENANCE key so the tag renders"


def test_no_cockpit_label_says_growth():
    labels = LABEL_CALL.findall(cockpit.COCKPIT_HTML)
    assert len(labels) > 20, "the label regex found too few labels to be trusted"
    offenders = [x for x in labels if "grow" in x.lower()]
    assert not offenders, f"a count labelled growth is P0's theatre shape: {offenders}"
    assert "grown units" not in cockpit.COCKPIT_HTML.lower()


def test_provenance_row_says_admitted_not_growth():
    row = brain_telemetry.PROVENANCE["grown_units"].lower()
    assert "admitted" in row, row
    assert "not growth" in row, "the disclaimer is part of the row, not a synonym: " + row
