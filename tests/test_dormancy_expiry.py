"""Launcher D-90 / enforcer #7333: a dormancy_gate.EXEMPT row may carry a tasks.md expiry, at
MODULE grain. Ticked with the module still unreachable it REDS; the key fails CLOSED on every
audit. feed_readout's row is keyed to T117a (launcher D-92)."""

import pytest

import snn_data_lake.gates.conclusion_protocol as cp
from snn_data_lake.gates import dormancy_gate as dg

FEED_EXPIRED = "feed_readout (expiry T117a ticked; still open)"
T117A_OPEN = "- [ ] T117a [G3] bind the reservoir join into the live grower\n"
T117A_DONE = "- [X] T117a [G3] bind the reservoir join into the live grower\n"


def _tasks(monkeypatch, tmp_path, text):
    """A tasks.md holding exactly `text`, read through conclusion_protocol.TASKS_MD."""
    tasks = tmp_path / "tasks.md"
    tasks.write_text(text, encoding="utf-8")
    monkeypatch.setattr(cp, "TASKS_MD", tasks)


def test_the_feed_readout_row_carries_the_t117a_expiry():
    assert dg.EXEMPT["feed_readout"][0] == "wire"
    assert dg.EXEMPT["feed_readout"][2] == "T117a"


def test_the_real_tree_has_no_expired_row():
    """Fail closed on the real key: T117a must be a tasks.md row on the tree this lands on."""
    assert dg.coverage()["expired"] == []


def test_feed_readout_expires_unreachable_when_t117a_ticks(monkeypatch, tmp_path):
    """The red: T117a ticked, feed_readout still unreached by any live root."""
    _tasks(monkeypatch, tmp_path, T117A_DONE)
    assert FEED_EXPIRED in dg.coverage()["expired"]
    with pytest.raises(AssertionError, match="past or unable to check their expiry"):
        dg.main()


def test_no_expiry_while_its_letter_suffixed_row_is_open(monkeypatch, tmp_path):
    _tasks(monkeypatch, tmp_path, T117A_OPEN)
    assert dg.coverage()["expired"] == []


def test_a_ticked_row_reds_only_while_still_open(monkeypatch, tmp_path):
    """Control: the same ticked key, with the predicate saying the module is now reached."""
    _tasks(monkeypatch, tmp_path, T117A_DONE)
    assert dg.expired_rows({"x": ("wire", "ctl", "T117a")}, lambda m: False) == []
    assert dg.expired_rows({"x": ("wire", "ctl", "T117a")}, lambda m: True) == [
        "x (expiry T117a ticked; still open)"
    ]


def test_feed_readout_passes_its_tick_once_reachable(monkeypatch, tmp_path):
    """The retire path through coverage()'s own walk: T117a ticked, feed_readout reached."""
    _tasks(monkeypatch, tmp_path, T117A_DONE)
    walk = dg.reachable
    monkeypatch.setattr(dg, "reachable", lambda: walk() | {dg._CANON["feed_readout"]})
    assert dg.coverage()["expired"] == []


def test_an_indented_ticked_row_resolves_and_ticks(monkeypatch, tmp_path):
    _tasks(monkeypatch, tmp_path, "  " + T117A_DONE)
    assert FEED_EXPIRED in dg.coverage()["expired"]


def test_a_lowercase_tick_is_a_tick(monkeypatch, tmp_path):
    _tasks(monkeypatch, tmp_path, T117A_DONE.replace("[X]", "[x]"))
    assert FEED_EXPIRED in dg.coverage()["expired"]


def test_a_typo_key_reds_before_any_tick(monkeypatch, tmp_path):
    _tasks(monkeypatch, tmp_path, T117A_OPEN)
    monkeypatch.setitem(dg.EXEMPT, "feed_readout", (*dg.EXEMPT["feed_readout"][:2], "T1111"))
    assert dg.coverage()["expired"] == ["feed_readout (expiry T1111 names no tasks.md row)"]


def test_a_prefix_row_does_not_resolve_a_suffixed_key(monkeypatch, tmp_path):
    """T117 present, T117a absent: the key names no row (no prefix match)."""
    _tasks(monkeypatch, tmp_path, "- [X] T117 [G9] the engine end to end\n")
    assert dg.coverage()["expired"] == ["feed_readout (expiry T117a names no tasks.md row)"]


def test_a_suffixed_row_does_not_resolve_a_prefix_key(monkeypatch, tmp_path):
    """T117a present, key T117: the key names no row (no suffix match)."""
    _tasks(monkeypatch, tmp_path, T117A_DONE)
    monkeypatch.setitem(dg.EXEMPT, "feed_readout", (*dg.EXEMPT["feed_readout"][:2], "T117"))
    assert dg.coverage()["expired"] == ["feed_readout (expiry T117 names no tasks.md row)"]


def test_a_tasks_md_that_is_not_utf8_fails_closed(monkeypatch, tmp_path):
    tasks = tmp_path / "tasks.md"
    tasks.write_bytes(b"- [X] T117a \xff\xfe\n")
    monkeypatch.setattr(cp, "TASKS_MD", tasks)
    (row,) = dg.coverage()["expired"]
    assert row.startswith("feed_readout (expiry T117a: tasks.md unreadable:")


def test_an_unreadable_tasks_md_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "TASKS_MD", tmp_path / "absent" / "tasks.md")
    (row,) = dg.coverage()["expired"]
    assert row.startswith("feed_readout (expiry T117a: tasks.md unreadable:")


def test_the_self_check_plants_in_its_own_table_only():
    before = dict(dg.EXEMPT)
    dg._expiry_selfcheck()
    assert dg.EXEMPT == before


def test_the_self_check_reds_when_the_detector_is_blind(monkeypatch):
    monkeypatch.setattr(dg, "expired_rows", lambda exempt, still_open, tasks_md=None: [])
    with pytest.raises(AssertionError, match="did not red"):
        dg._expiry_selfcheck()


def test_the_self_check_reds_when_the_tick_arm_is_dropped(monkeypatch):
    """Enforcer #7373 NIT-1: dropping the 'ticked; still open' branch fails main()'s self-check."""
    real = dg.expired_rows
    monkeypatch.setattr(
        dg,
        "expired_rows",
        lambda exempt, still_open, tasks_md=None: [
            r for r in real(exempt, still_open, tasks_md) if "ticked" not in r
        ],
    )
    with pytest.raises(AssertionError, match="did not red"):
        dg._expiry_selfcheck()


QUOTED = "> - [X] T117a note\n"


def test_one_row_parser_reads_a_quoted_row_as_no_row():
    """Enforcer #7373 MINOR-1: the quoted row reads the same in both callers -- no row."""
    assert cp.task_state("T117a", QUOTED) is None
    assert cp._task_done("T117a", QUOTED) is False
    assert dg.expired_rows({"x": ("wire", "q", "T117a")}, lambda m: True, QUOTED) == [
        "x (expiry T117a names no tasks.md row)"
    ]
    beside = QUOTED + T117A_OPEN  # the quote must not shadow the real open row
    assert cp.task_state("T117a", beside) == " "
    assert cp._task_done("T117a", beside) is False


def test_task_state_reads_absent_open_and_ticked():
    assert cp.task_state("T117a", T117A_OPEN) == " "
    assert cp.task_state("T117a", T117A_DONE) == "x"
    assert cp.task_state("T117a", T117A_DONE.replace("[X]", "[x]")) == "x"
    assert cp.task_state("T117", T117A_DONE) is None


def test_task_state_raises_on_an_unreadable_file_and_task_done_fails_open(monkeypatch, tmp_path):
    monkeypatch.setattr(cp, "TASKS_MD", tmp_path / "absent" / "tasks.md")
    with pytest.raises(OSError):
        cp.task_state("T117a")
    assert cp._task_done("T117a") is False


def test_task_state_raises_on_non_utf8_and_task_done_fails_open(monkeypatch, tmp_path):
    tasks = tmp_path / "tasks.md"
    tasks.write_bytes(b"- [X] T117a \xff\xfe\n")
    monkeypatch.setattr(cp, "TASKS_MD", tasks)
    with pytest.raises(UnicodeDecodeError):
        cp.task_state("T117a")
    assert cp._task_done("T117a") is False


def test_explicit_tasks_text_wins_over_the_file(monkeypatch, tmp_path):
    """Passed text decides, even empty: the self-check's planted text never falls back to disk."""
    _tasks(monkeypatch, tmp_path, T117A_OPEN)
    row = {"x": ("wire", "ctl", "T117a")}
    assert dg.expired_rows(row, lambda m: True, T117A_DONE) == [
        "x (expiry T117a ticked; still open)"
    ]
    assert dg.expired_rows(row, lambda m: True, "") == ["x (expiry T117a names no tasks.md row)"]


def test_a_third_element_does_not_break_the_scale_retest_unpack():
    dg._assert_scale_retest({"feed_readout": dg.EXEMPT["feed_readout"]})
    with pytest.raises(AssertionError, match="g3_planted"):
        dg._assert_scale_retest(
            {"g3_planted": ("scale-retest", "no record, no sha", "T117a")},
            history=lambda sha: None,
        )
