"""
RSA drift: the correlation of two response matrices' pairwise-dissimilarity (geometry) matrices
(G9 criterion 1, the one piece the F3 shared drift reader did not cover;
g9-campaign-design-input-DRAFT.md).

Planted pair, known geometry: 5 probes x 6 cells, two probe clusters ({0,1}, {2,3}) and a loner (4).
  - rotated cells (same geometry, cells permuted)  -> RSA drift reads 1.0 (geometry unchanged)
  - shuffled probes (probe i answered with another probe's response) -> RSA reads low
  - the red: a mutant that correlates the RAW responses (not their geometry) must FAIL the
    rotated case, so the test only passes for a reader that compares geometry.

Run: pytest tests/test_rsa_drift.py
"""

import pytest

from snn_data_lake.infra._mathutils import pearson, rsa_drift

PRE = [
    [9, 1, 0, 0, 2, 1],
    [8, 2, 1, 0, 1, 0],
    [0, 1, 9, 8, 0, 1],
    [1, 0, 8, 9, 1, 0],
    [0, 9, 1, 0, 9, 8],
]
ROTATED = [row[3:] + row[:3] for row in PRE]
SHUFFLED = [PRE[i] for i in (2, 4, 0, 3, 1)]


def _raw_mutant(pre, post):
    """The mutant the red guards against: correlates the raw responses, ignoring geometry."""
    return pearson([v for r in pre for v in r], [v for r in post for v in r])


def test_rotated_cells_same_geometry_reads_one():
    assert rsa_drift(PRE, ROTATED) == pytest.approx(1.0, abs=1e-9)


def test_identical_reads_one():
    assert rsa_drift(PRE, PRE) == pytest.approx(1.0, abs=1e-9)


def test_shuffled_probes_reads_low():
    assert rsa_drift(PRE, SHUFFLED) < 0.5


def test_raw_response_mutant_fails_the_rotated_case():
    # the red: comparing raw responses cannot pass the rotated case that the real reader passes
    assert _raw_mutant(PRE, ROTATED) < 0.5
    assert rsa_drift(PRE, ROTATED) - _raw_mutant(PRE, ROTATED) > 0.5


def test_flat_probe_row_refuses():
    flat = [list(r) for r in PRE]
    flat[2] = [3, 3, 3, 3, 3, 3]
    with pytest.raises(ValueError):
        rsa_drift(PRE, flat)


def test_shape_and_size_refuse():
    with pytest.raises(ValueError):
        rsa_drift(PRE, PRE[:4])
    with pytest.raises(ValueError):
        rsa_drift(PRE[:2], PRE[:2])
    with pytest.raises(ValueError):
        rsa_drift(PRE, [r[:5] for r in PRE])
