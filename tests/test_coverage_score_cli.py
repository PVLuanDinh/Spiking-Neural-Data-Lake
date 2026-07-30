"""`tools/coverage_score.py` must refuse an argument it does not understand.

The failure this file exists to prevent is not a crash, it is a SUCCESS: `rerun drive_60` written
positionally fell through to `report()`, printed a full coverage table and exited 0. The forgery
check could be asked for, silently not run, and still leave the operator holding a confident
number -- exactly how a sweep came to report artifacts as reproducing while rerunning none.

The bare-`--rerun` guard inside main() already fixed one instance of this shape. These tests hold
the outer one, and equally that the guard did not become over-eager: refusing a VALID invocation
would be the same class of damage pointed the other way.
"""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "coverage_score_cli", os.path.join(_ROOT, "tools", "coverage_score.py")
)
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)


@pytest.mark.parametrize(
    "argv,want",
    [
        (["rerun", "drive_60"], "rerun"),  # THE bug: positional verb, used to score and exit 0
        (["--reurn", "drive_60"], "--reurn"),  # a typo must not become a full report
        (["score", "--chekc"], "--chekc"),  # nor must a typo'd flag
        (["drive_60"], "drive_60"),  # a bare artifact name is not an invocation
    ],
)
def test_unknown_tokens_are_named(argv, want):
    assert cs._unknown_arg(argv) == want


@pytest.mark.parametrize(
    "argv",
    [
        [],  # the documented default: a plain report
        ["score"],
        ["report", "--check"],
        ["--check"],
        ["--rerun", "drive_60"],
        ["--root", "/tmp/x", "--evidence", "/tmp/y", "--check"],
        ["record", "--name", "n", "--kind", "ab", "--", "python", "probe.py", "--check"],
    ],
)
def test_valid_invocations_are_not_refused(argv):
    assert cs._unknown_arg(argv) is None


def test_a_flag_value_is_never_mistaken_for_a_token():
    """`--rerun score` names an artifact spelling a verb; the value must be consumed."""
    assert cs._unknown_arg(["--rerun", "score"]) is None
    assert cs._unknown_arg(["--name", "--check"]) is None


def test_the_recorded_command_after_dashdash_is_not_validated():
    """Everything past `--` is the probe's own argv and may hold flags this CLI never heard of."""
    argv = ["record", "--name", "n", "--", "python", "p.py", "--weird-probe-flag", "rerun"]
    assert cs._unknown_arg(argv) is None


def test_main_refuses_with_exit_2_and_runs_nothing(capsys, monkeypatch):
    """The whole point: a bad invocation must not reach report() and print a coverage table."""
    monkeypatch.setattr(cs, "report", lambda *a, **k: pytest.fail("report() must not run"))
    monkeypatch.setattr(cs, "rerun", lambda *a, **k: pytest.fail("rerun() must not run"))
    assert cs.main(["rerun", "drive_60"]) == 2
    out = capsys.readouterr().out
    assert "unrecognized argument" in out and "usage:" in out
