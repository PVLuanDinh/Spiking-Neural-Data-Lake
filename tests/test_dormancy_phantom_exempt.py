"""G8 S0b: the dormancy gate reds a phantom EXEMPT row.

specs/022-unified-brain-arc/evidence/g8-reorganization.md, 2.4 (S0b) and 2.6 (its declared delta). A
`dormancy_gate.EXEMPT` key that names no brain module (absent from `_CANON`) exempts nothing, and it
hides the rename or retirement that orphaned it, because `coverage()` only ever read the keys that
DO resolve. `coverage()` now reports `phantom_exempt`, `main()` fails on a non-empty list with a
message naming the rows, and the gate's own self-check plants such a row in a COPY of `EXEMPT` and
asserts that the gate reds. Counted first: 0 phantom rows at the base head, so it lands green.
"""

import pytest

from snn_data_lake.substrate import dormancy_gate as dg

_PLANT = "g8_s0b_no_such_module"


def test_the_report_carries_phantom_exempt_and_it_is_empty():
    assert dg.coverage()["phantom_exempt"] == []


def test_main_reds_on_a_phantom_row_and_names_it(monkeypatch):
    assert _PLANT not in dg._CANON
    planted = {**dg.EXEMPT, _PLANT: ("permanent", "planted by this test; names no module")}
    monkeypatch.setattr(dg, "EXEMPT", planted)
    with pytest.raises(AssertionError, match=_PLANT):
        dg.main()


def test_the_self_check_plants_in_a_copy_only():
    before = dict(dg.EXEMPT)
    dg._phantom_selfcheck()
    assert dg.EXEMPT == before


def test_the_self_check_reds_when_the_detector_is_blind(monkeypatch):
    monkeypatch.setattr(dg, "_phantom_rows", lambda exempt: [])
    with pytest.raises(AssertionError, match="did not red"):
        dg._phantom_selfcheck()
