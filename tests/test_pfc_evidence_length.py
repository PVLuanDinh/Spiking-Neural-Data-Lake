"""Codex B2 pfc-p2 MINOR 2: evidence whose length is not the executive's option count.
The deleted Python body returned `(0, 1)` for all-zero evidence of ANY length before it indexed
`_dec_bias`, and raised IndexError (after consuming draws) for positive evidence of the wrong
length. The B2 binding raised RuntimeError for both from its `dec_bias` length check. Contract:
zero evidence of any length is `(0, 1)` with no draw; positive evidence of the wrong length is an
explicit RuntimeError naming the mismatch, with no draw.
"""

import json
import pathlib

import pytest

from snn_data_lake.faculties.prefrontal_executive import PrefrontalExecutive

FIX = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "native"
        / "sndl-core"
        / "tests"
        / "fixtures_pfc.json"
    ).read_text()
)


def _chain_a_executive():
    (ch,) = [c for c in FIX["chains"] if c["name"] == "A"]
    p = PrefrontalExecutive(**ch["ctor"])  # 13 options
    p.rng.setstate((3, tuple(ch["state0"]), None))
    return p


@pytest.mark.parametrize("n", [0, 1, 13, 14, 40])
def test_zero_evidence_of_any_length_is_no_race_and_no_draw(n):
    p = _chain_a_executive()
    pre = p.rng.getstate()
    assert list(p._decide_spiking([0.0] * n)) == [0, 1]
    assert p.rng.getstate() == pre


@pytest.mark.parametrize("n", [14, 40])
def test_positive_evidence_of_the_wrong_length_is_an_explicit_error_and_no_draw(n):
    p = _chain_a_executive()
    pre = p.rng.getstate()
    with pytest.raises(
        RuntimeError, match=r"dec_bias has \d+ entries, the race needs n_dec \* n_options"
    ):
        p._decide_spiking([1.0] + [0.0] * (n - 1))
    assert p.rng.getstate() == pre
