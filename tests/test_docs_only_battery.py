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
