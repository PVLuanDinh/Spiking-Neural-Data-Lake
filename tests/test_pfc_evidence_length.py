"""Codex B2 pfc-p2 MINOR 2: evidence LONGER than the executive's option count.
The deleted Python body returned `(0, 1)` for all-zero evidence of ANY length before it indexed
`_dec_bias`, and raised IndexError (after consuming draws) for positive evidence longer than the
option count. The B2 binding raised RuntimeError for both from its `dec_bias` length check.
Contract: zero evidence of any length is `(0, 1)` with no draw; positive evidence longer than the
option count is an explicit RuntimeError naming the mismatch, with no draw. Shorter positive
evidence is a narrower race, not an error: oracle chain B (3 options, evidence of length 1..3)
pins it in tests/test_native_pfc.py.
"""

import pytest
from test_native_pfc import FIX, _executive


def _chain_a():  # 13 options
    (ch,) = [c for c in FIX["chains"] if c["name"] == "A"]
    return _executive(ch)


@pytest.mark.parametrize("n", [0, 1, 13, 14, 40])
def test_zero_evidence_of_any_length_is_no_race_and_no_draw(n):
    p = _chain_a()
    pre = p.rng.getstate()
    assert list(p._decide_spiking([0.0] * n)) == [0, 1]
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
