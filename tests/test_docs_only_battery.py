"""docs_only_battery: the refusals, the selection rules and the audit hook (owner 2026-10-02).

`run()` is deliberately never called here: on a Markdown-only diff with a current map it would start
a real battery from inside pytest.
"""

import json

import pytest

from snn_data_lake.gates import docs_only_battery as dob


def test_rules_and_reds():
    dob._selfcheck_rules()


def test_instrument_controls_in_child_processes():
    dob._selfcheck_instrument()


@pytest.mark.parametrize(
    "changes",
    [[("M", "README.md"), ("M", "tools/policy_certification.json")], [("A", "src/x.py")]],
)
def test_r1_refuses_any_non_markdown_path(changes):
    with pytest.raises(dob.Refused, match="R1"):
        dob.check_changed(changes)


def test_r2_refuses_the_committed_map_against_edited_code():
    """The live map (once one is committed) describes this tree's code; one changed byte of code
    makes it a map for other code."""
    ls = dob._git("ls-files", "-s")
    chash = dob.code_hash(ls)
    m = {"version": dob.MAP_VERSION, "code_hash": chash, "head": "h"}
    dob.check_map(m, chash)
    edited = ls.replace(
        "\trun_selfchecks.py",
        "X\trun_selfchecks.py",
        1,
    )
    assert edited != ls
    with pytest.raises(dob.Refused, match="R2"):
        dob.check_map(m, dob.code_hash(edited))


def test_pth_line_is_one_inert_line():
    line = dob.pth_line()
    assert line.count("\n") == 1 and line.startswith("import os;")
    assert 'os.environ.get("SNDL_DOC_AUDIT")' in line


def test_committed_map_if_present_is_well_formed():
    if not dob.MAP_PATH.exists():
        pytest.skip("no map committed yet (the first instrumented battery writes it)")
    m = json.loads(dob.MAP_PATH.read_text(encoding="utf-8"))
    assert m["version"] == dob.MAP_VERSION
    assert set(m["pytest_all"]) == {"read", "list", "listr", "git_scan", "blind"}
    for t, e in m["tests"].items():
        assert t.startswith("tests/") and set(e) == {"read", "list", "listr", "git_scan", "blind"}


def _xdist_records():
    """An xdist-shaped run. The pytest master spawns a worker OUTSIDE any test; the worker's argv
    is the execnet bootstrap (no "pytest" in it) and it reads a doc at collection time (no test
    id). A gate process (not pytest) also spawns a python child that reads a doc: unmapped."""
    return [
        {"k": "start", "_pid": "m", "argv": ["python", "-m", "pytest", "-n", "8"], "tok": "t0"},
        {
            "k": "spawn",
            "_pid": "m",
            "t": "",
            "a": ["python", "-u", "-c", "import sys;exec(...)"],
            "tok": "w1",
        },
        {
            "k": "start",
            "_pid": "w",
            "argv": ["python", "-u", "-c", "import sys;exec(...)"],
            "tok": "w1",
        },
        {"k": "read", "_pid": "w", "t": "", "p": "docs/collected.md", "c": "conftest"},
        {"k": "start", "_pid": "g", "argv": ["python", "tools/some_gate.py"], "tok": "g0"},
        {"k": "spawn", "_pid": "g", "t": "", "a": ["python", "-c", "pass"], "tok": "g1"},
        {"k": "start", "_pid": "gc", "argv": ["python", "-c", "pass"], "tok": "g1"},
        {"k": "read", "_pid": "gc", "t": "", "p": "docs/gate_only.md", "c": "x"},
    ]


def test_xdist_worker_collection_reads_are_mapped_not_dropped():
    m = dob.build_map(_xdist_records(), "head", "chash")
    assert "docs/collected.md" in m["pytest_all"]["read"]  # red without the worker mapping
    assert (
        "docs/gate_only.md" not in m["pytest_all"]["read"]
    )  # a non-pytest parent's child is not pytest


def test_xdist_worker_child_listed_before_its_worker_is_still_mapped():
    # pid "10234" (the worker's child) sorts before "9876" (the worker) as a string
    recs = [
        {"k": "start", "_pid": "10234", "argv": ["python", "-c", "pass"], "tok": "c1"},
        {"k": "read", "_pid": "10234", "t": "", "p": "docs/grandchild.md", "c": "x"},
        {"k": "start", "_pid": "1", "argv": ["python", "-m", "pytest", "-n", "2"], "tok": "t0"},
        {"k": "spawn", "_pid": "1", "t": "", "a": ["python", "-u", "-c", "boot"], "tok": "w1"},
        {"k": "start", "_pid": "9876", "argv": ["python", "-u", "-c", "boot"], "tok": "w1"},
        {"k": "spawn", "_pid": "9876", "t": "", "a": ["python", "-c", "pass"], "tok": "c1"},
    ]
    m = dob.build_map(recs, "head", "chash")
    assert "docs/grandchild.md" in m["pytest_all"]["read"]  # red under a single pass
