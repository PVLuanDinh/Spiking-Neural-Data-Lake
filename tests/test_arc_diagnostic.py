"""
Tests for reasoning/arc_diagnostic.py -- the instrument that decides WHY the ARC solver fails.

The distinction it draws is load-bearing: a BUDGET-exhausted failure could be fixed by searching
harder (so neural-guided search would pay off), while a COVERAGE-limited failure has no DSL-
expressible program at all (so no amount of search helps). Its measured verdict -- 11/11 failures
coverage-limited, guided-search ceiling +0 tasks -- is what justifies NOT building neural-guided
search, so the classifier itself must be trustworthy.

These tests exercise the synthetic controls (fast + deterministic) rather than the real ARC split,
which is a ~20 min offline run.

Run: pytest tests/test_arc_diagnostic.py
"""

from snn_data_lake.reasoning.arc_diagnostic import diagnose_task, node_bound, synthetic_tasks


def test_synthetic_controls_classify_correctly():
    """Each synthetic control carries the classification it MUST receive -- ground truth for
    the classifier itself."""
    tasks = synthetic_tasks()
    assert tasks, "synthetic control set must not be empty"
    for name, task, want in tasks:
        # syn_budget only fails-on-budget under the reduced baseline the module's own self-check
        # uses; at the shipped budget it simply solves (which is itself correct behaviour).
        kw = {"baseline_nodes": 20} if name == "syn_budget" else {}
        got = diagnose_task(task, **kw)["status"]
        assert got == want, f"control {name}: expected {want}, classified {got}"


def test_controls_cover_both_sides_of_the_verdict():
    """The set must contain BOTH a budget-limited and a coverage-limited case, or the classifier
    could pass every control while always answering one way."""
    kinds = {want for _name, _task, want in synthetic_tasks()}
    assert "budget_proven" in kinds, (
        "no budget-limited control -- that side of the verdict is untested"
    )
    assert "coverage_proven" in kinds, "no coverage-limited control -- ditto"


def test_node_bound_is_a_real_ceiling():
    """node_bound must be a finite, positive ceiling on the depth-limited program space."""
    _name, task, _want = synthetic_tasks()[0]
    train = [(p["input"], p["output"]) for p in task["train"]]
    n = node_bound(train)
    assert isinstance(n, int) and n > 0
    assert n < 10**9, "the bound must be a usable search ceiling, not effectively infinite"


def test_classification_is_deterministic():
    tasks = synthetic_tasks()
    first = [diagnose_task(t)["status"] for _n, t, _w in tasks]
    second = [diagnose_task(t)["status"] for _n, t, _w in tasks]
    assert first == second
