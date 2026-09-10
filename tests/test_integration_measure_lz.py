"""Amendment 2.11's RED (specs/022 T070, `evidence/t070-integration-kernel.md` 2.11): the LZ76
scorer's COST, and the EQUALITY that licenses any faster algorithm to replace its body.

`integration_measure.lz76_complexity` is the Kaspar & Schuster (1987) form of the LZ76 exhaustive-
history parse; its inner loop extends a candidate match one character at a time from EVERY earlier
position, so on a sparse binary string with long zero runs the cost grows quadratically with the
length. The 64-unit fixture (5,120 characters) parses in 0.24 s; the campaign's footprint is
14,215 x 80 = 1,137,200 characters, and the first campaign spent 9 h 25 min inside ONE seed's arms
without producing a number. The fix 2.11 pre-registers keeps the STATISTIC -- c(L), the phrase
count of the exhaustive history; normalized_lz = c(L) log2(L) / L -- and changes only the
ALGORITHM, under this test: the incumbent's body, copied VERBATIM below as `_reference_lz76`, is
the oracle, and the new body must return the SAME count on every string here, or the fix is
REFUSED (never the oracle "fixed", never the test weakened).

  (i)   every string the incumbent's own self-check scores (its factories at their fixture sizes,
        captured by wrapping the function the self-check calls);
  (ii)  500 seeded random binary strings, lengths 2-4,096, densities 0.5 / 0.1 / 0.02 / 0.005;
  (iii) all-zero, all-one, periodic ("01" * k, "0001" * k), one 1 at every position, length 0/1/2;
  COST: a 200,000-character density-0.02 string parses in under 5 s -- the RED today.

  PYTHONUTF8=1 PYTHONPATH=src pytest tests/test_integration_measure_lz.py -q --no-cov
"""

import random
import threading
import time

from snn_data_lake.cortex import integration_measure as IM

COST_CHARS = 200_000
COST_DENSITY = 0.02
COST_BUDGET_S = 5.0
RANDOM_STRINGS = 500
RANDOM_DENSITIES = (0.5, 0.1, 0.02, 0.005)
RANDOM_MAX_LEN = 4_096


def _reference_lz76(s):
    """THE ORACLE: the incumbent's body at 99fe3ef, verbatim (Kaspar & Schuster 1987). Lives ONLY
    here (2.11, P2: one implementation survives in `src`; the old body is a test oracle, never a
    code path). The `noqa` marks keep the body verbatim under ruff's E741."""
    n = len(s)
    if n < 2:
        return 1
    i, k, l = 0, 1, 1  # noqa: E741
    k_max = 1
    c = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max  # noqa: E741
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def _sparse(rng, n, density):
    return "".join("1" if rng.random() < density else "0" for _ in range(n))


def _assert_same(s, label):
    got, want = IM.lz76_complexity(s), _reference_lz76(s)
    assert got == want, (
        f"lz76_complexity DIFFERS from the incumbent's count on {label} (len {len(s)}, ones "
        f"{s.count('1')}): got {got}, the oracle says {want}; the fix is REFUSED on this string: "
        f"{s!r}"
    )


# =============================================================================================
# (i) every string the incumbent's self-check scores
# =============================================================================================
def test_equal_to_the_oracle_on_every_string_the_self_check_scores(monkeypatch):
    seen = []
    scorer = IM.lz76_complexity

    def recording(s):
        seen.append(s)
        return scorer(s)

    monkeypatch.setattr(IM, "lz76_complexity", recording)
    IM.main()  # the incumbent's own self-check, tripwires and all
    monkeypatch.undo()
    strings = sorted(set(seen), key=lambda s: (len(s), s))
    assert len(strings) >= 100, f"the self-check scored only {len(strings)} distinct strings"
    for s in strings:
        _assert_same(s, "a self-check footprint")


# =============================================================================================
# (ii) 500 seeded random binary strings at four densities
# =============================================================================================
def test_equal_to_the_oracle_on_random_strings_at_four_densities():
    rng = random.Random(2026_09_10)
    for k in range(RANDOM_STRINGS):
        density = RANDOM_DENSITIES[k % len(RANDOM_DENSITIES)]
        n = rng.randint(2, RANDOM_MAX_LEN)
        _assert_same(_sparse(rng, n, density), f"random string #{k} (density {density})")


# =============================================================================================
# (iii) the structured edge cases
# =============================================================================================
def test_equal_to_the_oracle_on_structured_and_edge_strings():
    cases = {"": "empty", "0": "single 0", "1": "single 1"}
    for s in ("00", "01", "10", "11"):
        cases[s] = f"length-2 {s}"
    for n in (3, 7, 64, 1_000, 4_096):
        cases["0" * n] = f"all-zero x{n}"
        cases["1" * n] = f"all-one x{n}"
    for k in (1, 2, 3, 10, 100, 1_024):
        cases["01" * k] = f"'01' x{k}"
        cases["0001" * k] = f"'0001' x{k}"
    for n in (*range(1, 17), 64, 257):
        for p in range(n):
            s = "0" * p + "1" + "0" * (n - p - 1)
            cases[s] = f"one 1 at position {p} of {n}"
    for s, label in cases.items():
        _assert_same(s, label)


# =============================================================================================
# COST: the campaign-shaped string, bounded by a thread so the RED is 5 s and not an afternoon
# =============================================================================================
def test_a_200k_character_sparse_string_parses_in_under_five_seconds():
    s = _sparse(random.Random(7), COST_CHARS, COST_DENSITY)
    out = {}

    def parse():
        t0 = time.perf_counter()
        out["c"] = IM.lz76_complexity(s)
        out["s"] = time.perf_counter() - t0

    worker = threading.Thread(target=parse, daemon=True)
    worker.start()
    worker.join(COST_BUDGET_S)
    assert not worker.is_alive(), (
        f"lz76_complexity on {COST_CHARS:,} characters at density {COST_DENSITY}: still parsing "
        f"after {COST_BUDGET_S} s (2.11's cost bar)"
    )
    assert out["s"] < COST_BUDGET_S, (
        f"lz76_complexity on {COST_CHARS:,} characters at density {COST_DENSITY}: {out['s']:.3f} s "
        f"(bar {COST_BUDGET_S} s; c = {out['c']})"
    )
