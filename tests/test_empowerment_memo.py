"""Empowerment prefix memo (perf, 2026-10-19): same results, unhashable actions still accepted."""

from snn_data_lake.faculties.empowerment import Empowerment


class _Pure:
    """A pure world model: next state depends only on (state, action)."""

    def predict(self, z, a):
        return [z[0] + float(a[0]), z[1] - 0.5 * float(a[0])]


def _no_memo(emp, z0):
    import math

    cells = {emp._cell(emp._roll(z0, s)) for s in emp._seqs()}
    return math.log2(len(cells)) if cells else 0.0


def test_unhashable_actions_on_the_exhaustive_path_still_work():
    # Codex re-review repro: list actions, H=2, total=4 <= max_seqs -> itertools.product path
    emp = Empowerment(_Pure(), [[0], [1]], horizon=2)
    assert emp.empowerment([0.0, 0.0]) == _no_memo(
        emp, [0.0, 0.0]
    )  # red: TypeError if keyed on actions


def test_memo_is_bit_identical_to_no_memo_on_the_sampling_path():
    # every terminal VECTOR, not just the cell count (Codex re-review MINOR 4: a memo that shifted
    # every terminal by a constant kept the count and passed the old check)
    emp = Empowerment(_Pure(), [(i,) for i in range(5)], horizon=3, max_seqs=20, seed=3)
    z0 = [0.3, -0.2]
    seqs = emp._seqs()
    memo = {}
    assert [emp._roll(z0, s, memo) for s in seqs] == [emp._roll(z0, s) for s in seqs]
    assert len(memo) > 0
