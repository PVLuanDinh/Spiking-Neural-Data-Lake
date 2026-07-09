"""
Edge tests for signal_loop's pure wire-contract + matcher helpers: normalize/resize on
empty/short/long/constant windows, parse_line on comments/malformed/CSV-or-space input, and
the Lowe ratio test (a signal identical to a reference matches; an ambiguous/equidistant
window is rejected so the loop stays silent rather than firing a wrong command).
Determinism: _proto uses crc32 (not hash()), so references are stable across processes.

Run: pytest tests/test_signal_loop.py
"""

from snn_data_lake import signal_loop as sl
from snn_data_lake.cortex.spike_preprocessing import N


def test_normalize_empty_and_constant_windows():
    assert sl.normalize([]) == [0.0] * N  # empty -> zero vector of length N
    assert sl.normalize([0.0, 0.0, 0.0]) == [0.0] * N  # all-zero window -> zeros (zero range)
    assert sl.normalize([5.0] * N) == [0.0] * N  # full-length constant -> zero range -> zeros
    # NOTE: a short *non-zero* constant like [5,5,5] does NOT collapse — it is padded to N with
    # zeros first, so the min-max range is 0..5 and it scales to [1,1,1,0,...]. Verify that:
    short_const = sl.normalize([5.0, 5.0, 5.0])
    assert short_const[:3] == [1.0, 1.0, 1.0] and short_const[3:] == [0.0] * (N - 3)
    assert len(sl.normalize([1.0])) == N  # short window is padded to N


def test_normalize_scales_to_unit_interval():
    out = sl.normalize([0.0, 5.0, 10.0])
    assert len(out) == N
    assert min(out) == 0.0 and max(out) == 1.0  # min-max hits both endpoints
    assert all(0.0 <= x <= 1.0 for x in out)


def test_normalize_truncates_long_windows():
    assert len(sl.normalize(list(range(N + 50)))) == N


def test_resize_to_n_pads_and_truncates():
    assert len(sl.resize_to_n([1.0])) == N
    assert len(sl.resize_to_n([1.0] * (N + 99))) == N
    assert sl.resize_to_n([1.0, 2.0], n=4) == [1.0, 2.0, 0.0, 0.0]


def test_parse_line_contract():
    assert sl.parse_line("# comment") is None
    assert sl.parse_line("   ") is None
    assert sl.parse_line("") is None
    assert sl.parse_line("0.1, 0.2 0.3") == [0.1, 0.2, 0.3]  # comma OR space separated
    assert sl.parse_line("nope,7") is None  # unparseable -> skipped, no crash


def test_line_windows_direct_and_windowed():
    rows = [[float(i)] for i in range(3 * N)]
    wins = list(sl.line_windows(rows, window=N))
    assert len(wins) == 3 and all(len(w[1]) == N for w in wins)  # W rows -> 1 window, N wide
    direct = list(sl.line_windows([[1.0, 2.0]], window=1))
    assert len(direct) == 1 and len(direct[0][1]) == N  # direct mode: 1 row -> 1 window


def test_matcher_matches_exact_reference():
    lib = sl.build_default_library()
    match = sl.build_matcher(lib)
    label = next(iter(lib))
    matched, best_d, second_d, _ = match(lib[label])
    assert matched == label  # a signal identical to a ref matches it
    assert best_d < second_d  # and clearly beats the runner-up


def test_matcher_rejects_ambiguous_window():
    # two identical references -> every query is equidistant -> ratio test cannot separate -> reject
    lib = {"A": sl._proto("A", None), "B": sl._proto("A", None)}
    match = sl.build_matcher(lib)
    matched, best_d, second_d, _ = match(sl._proto("A", None))
    assert matched is None  # best_d == second_d -> not < RATIO*second_d
