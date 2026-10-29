"""D-33's red: every test that reaches a process pool carries the `pool` marker.

The arc-4 battery (battery-20261028-191859, int3 fceab25e) was stopped VOID at 85.7% commit: two
xdist workers each ran a 4-process spawn pool at once, about 23 GB of commit in 3 min. Process-pool
tests therefore run in their own serial stage (`-m pool`, pyproject markers), never beside xdist
workers.

This census reads every tests/test_*.py by AST. A function that names `ProcessPoolExecutor` or
`multiprocessing` (or an alias imported from them) starts a pool, and so does any module function
that calls one of those, transitively, by name. Every `test_*` among them must carry
`@pytest.mark.pool`, or the module's `pytestmark` must name it. Disclosed limit: it is static and
per module, so it cannot see a pool started from src/ (g5_readout_cells.run_cells's default pool is
the known case; its tests inject ThreadPoolExecutor). test_real_percept_stream's `_run_arms` also
refuses to run under an xdist worker at runtime.
"""

import ast
import pathlib

TESTS = pathlib.Path(__file__).parent
POOL_ROOTS = {"ProcessPoolExecutor", "multiprocessing"}


def _pool_names(tree):
    """The pool roots plus every alias this module imports for them."""
    names = set(POOL_ROOTS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "multiprocessing":
                    names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if node.module.split(".")[0] == "multiprocessing" or (
                    node.module == "concurrent.futures" and a.name == "ProcessPoolExecutor"
                ):
                    names.add(a.asname or a.name)
    return names


def _refs(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def _is_pool_mark(dec):
    d = dec.func if isinstance(dec, ast.Call) else dec
    return (
        isinstance(d, ast.Attribute)
        and d.attr == "pool"
        and isinstance(d.value, ast.Attribute)
        and d.value.attr == "mark"
    )


def pool_census(source):
    """(tests that reach a process pool, those among them NOT marked `pool`) for one module."""
    tree = ast.parse(source)
    pools = _pool_names(tree)
    funcs = {
        n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = {
        name: {
            c.func.id
            for c in ast.walk(f)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        for name, f in funcs.items()
    }
    reach = {name for name, f in funcs.items() if _refs(f) & pools}
    while True:
        more = {name for name, called in calls.items() if called & reach} - reach
        if not more:
            break
        reach |= more
    tests = sorted(name for name in reach if name.startswith("test_"))
    module_marked = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets)
        and "pool" in _refs(n.value)
        for n in tree.body
    )
    if module_marked:
        return tests, []
    return tests, [t for t in tests if not any(_is_pool_mark(d) for d in funcs[t].decorator_list)]


_DIRECT = """
import concurrent.futures as _cf
def test_direct():
    with _cf.ProcessPoolExecutor(2) as ex:
        pass
"""
_VIA_HELPER = """
import multiprocessing as mp
def _helper():
    return mp.get_context("spawn").Pool(2)
def _middle():
    return _helper()
def test_via_helper():
    _middle()
def test_threads_only():
    pass
"""


def test_the_census_reds_an_unmarked_pool_test_and_passes_a_marked_one():
    """Reachable reds on constructed modules: a direct pool, a pool two helpers down through an
    aliased `multiprocessing`, and the same tests passing once marked (per test, or per module)."""
    assert pool_census(_DIRECT) == (["test_direct"], ["test_direct"])
    assert pool_census(_VIA_HELPER) == (["test_via_helper"], ["test_via_helper"])
    marked = _VIA_HELPER.replace("def test_via_helper", "@pytest.mark.pool\ndef test_via_helper")
    assert pool_census(marked) == (["test_via_helper"], [])
    module = "import pytest\npytestmark = [pytest.mark.slow, pytest.mark.pool]\n" + _DIRECT
    assert pool_census(module) == (["test_direct"], [])


def test_no_test_that_reaches_a_process_pool_lacks_the_pool_marker():
    """The census over the real tests/: no unmarked pool test, and it SEES the known pool tests
    (an instrument that found none would pass vacuously)."""
    found, unmarked = {}, {}
    for path in sorted(TESTS.glob("test_*.py")):
        tests, bad = pool_census(path.read_text(encoding="utf-8"))
        if tests:
            found[path.name] = tests
        if bad:
            unmarked[path.name] = bad
    assert len(found.get("test_real_percept_stream.py", [])) == 4, found
    assert not unmarked, f"tests reach a process pool without @pytest.mark.pool: {unmarked}"


def test_removing_one_pool_marker_from_the_real_module_is_caught():
    """The mutant the launcher asked for (D-33): test_real_percept_stream with its first `pool`
    marker removed must red the census on exactly that test."""
    src = (TESTS / "test_real_percept_stream.py").read_text(encoding="utf-8")
    first = src.index("@pytest.mark.pool")
    mutant = src[:first] + src[first:].replace("@pytest.mark.pool", "", 1)
    tests, bad = pool_census(mutant)
    assert len(tests) == 4
    assert bad == ["test_the_persistence_oracle_clears_its_own_floor_under_the_admission_window"]
