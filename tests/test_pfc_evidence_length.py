"""Codex B2 pfc-p2 MINOR 2: evidence LONGER than the executive's option count.
Contract: no evidence (no entry > 0: zeros, negatives, NaN) of any length is `(0, 1)` with no
draw, as the deleted Python body returned before it indexed `_dec_bias`; positive evidence longer
than the option count is an explicit RuntimeError naming the mismatch, with no draw (the body
raised IndexError after consuming draws). Shorter positive evidence is a narrower race, not an
error: oracle chain B (3 options, evidence of length 1..3) pins it in tests/test_native_pfc.py.
"""

import pytest
from test_native_pfc import FIX, _executive


def _chain_a():  # 13 options
    (ch,) = [c for c in FIX["chains"] if c["name"] == "A"]
    return _executive(ch)


@pytest.mark.parametrize(
    "evidence", [[0.0] * n for n in (0, 1, 13, 14, 40)] + [[-1.0, float("nan")] * 7]
)
def test_no_evidence_of_any_length_is_no_race_and_no_draw(evidence):
    p = _chain_a()
    pre = p.rng.getstate()
    assert list(p._decide_spiking(evidence)) == [0, 1]
    assert p.rng.getstate() == pre


@pytest.mark.parametrize("n", [14, 40])
def test_positive_evidence_longer_than_the_option_count_is_an_explicit_error_and_no_draw(n):
    p = _chain_a()
    pre = p.rng.getstate()
    with pytest.raises(
        RuntimeError, match=r"dec_bias has \d+ entries, the race needs n_dec \* n_options"
    ):
        p._decide_spiking([1.0] + [0.0] * (n - 1))
    assert p.rng.getstate() == pre
