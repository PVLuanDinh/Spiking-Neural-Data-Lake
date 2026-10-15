"""Tests for tools/lang_coverage.py -- the instrument that freezes the language-organ pair set.

specs/005-language-organ, contract A. These run BEFORE any organ exists, because the whole point of
contract A is that the restriction rule is fixed before either arm is scored (FR-023).
"""

import importlib.util
import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_TOOL = os.path.join(_ROOT, "tools", "lang_coverage.py")


def _load_tool():
    """Import tools/lang_coverage.py by path -- tools/ is not a package."""
    spec = importlib.util.spec_from_file_location("lang_coverage_under_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    # The leaf tool isolates its imports; collection must preserve the pytest caller's env.
    previous_env = {key: os.environ.get(key) for key in ("NORD_HOME", "NORD_PERSIST")}
    try:
        spec.loader.exec_module(mod)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return mod


LC = _load_tool()


def _fake_blimp(tmp_path, paradigms):
    """Write a miniature BLiMP tree. `paradigms` is {name: [(good, bad), ...]}."""
    d = tmp_path / "blimp"
    d.mkdir()
    for name, pairs in paradigms.items():
        with open(d / (name + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for good, bad in pairs:
                fh.write(json.dumps({"sentence_good": good, "sentence_bad": bad}) + "\n")
    return str(d)


# ------------------------------------------------------------------ T004: rule is score-independent


def test_rule_is_score_independent(tmp_path):
    """A2/FR-023: membership is a function of the lexicon and the rule only.

    Two things are asserted, because the weaker one alone is not enough. First, `restrict` accepts
    no scorer, arm, score or threshold parameter -- a rule that cannot see a score cannot be tuned
    to one. Second, perturbing every arm's score leaves the returned ids byte-identical, which is
    the observable form of the same claim."""
    import inspect

    params = set(inspect.signature(LC.restrict).parameters)
    assert params == {"lexicon", "pairs"}, (
        f"restrict must take only the lexicon and the pairs; found {sorted(params)}. "
        "A score-shaped parameter here is how a fixed rule becomes a tuned one."
    )

    lexicon = {"the", "cat", "sat", "sats", "dog"}
    pairs = [
        ("p:0", "the cat sat", "the cat sats"),
        ("p:1", "the dog sat", "the dog sats"),
        ("p:2", "the fox sat", "the fox sats"),  # 'fox' out of lexicon -> excluded
    ]
    before = LC.restrict(lexicon, pairs)
    assert before == ["p:0", "p:1"]

    # Perturb every arm's score. There is no arm to perturb *through* the rule, which is the point:
    # the strongest available form of the assertion is that scoring the same pairs with wildly
    # different scorers cannot change the membership, because membership never consulted one.
    for scorer in (
        lambda s: 0.0,
        lambda s: -len(s),
        lambda s: float(len(s)) * 1e6,
        lambda s: 1.0 if "cat" in s else -1.0,
    ):
        LC.score_pairs(scorer, pairs)
        after = LC.restrict(lexicon, pairs)
        assert after == before, "an arm's score changed the frozen membership"

    # And byte-identical when serialised, which is what coverage.md actually commits.
    assert "\n".join(before).encode("ascii") == "\n".join(LC.restrict(lexicon, pairs)).encode(
        "ascii"
    )


def test_score_pairs_agrees_with_the_instrument(tmp_path):
    """The local comparison loop must not drift from blimp_bench.score.

    tools/lang_coverage.py repeats the comparison loop because score() takes whole paradigms and the
    restricted set is a subset of pairs inside them. A repeated loop that quietly disagreed with the
    instrument would be an instrument edited by the party being measured (C5.5), so the agreement is
    asserted rather than assumed."""
    from snn_data_lake.reasoning import blimp_bench as BB

    root = _fake_blimp(
        tmp_path,
        {
            "toy": [
                ("the cat sat down", "the cat sat"),  # length_scorer prefers the shorter -> wrong
                ("a", "a longer sentence"),  # length_scorer prefers the shorter -> right
                ("same", "same"),  # a tie -> must count as WRONG
            ]
        },
    )
    _names, pairs = LC.load_universe(root=root)
    local_correct, local_total = LC.score_pairs(BB.length_scorer, pairs)
    ref = BB.score(BB.length_scorer, root=root)
    assert (local_correct, local_total) == (ref["correct"], ref["total"])
    assert local_correct == 1, "a tie must count as wrong; credit is never awarded for no opinion"


# ------------------------------------------------------------------------------ T005: floors exit 2


def test_floors_exit_two(tmp_path, monkeypatch, capsys):
    """A4: an artificially tiny lexicon must exit 2 with NOT-P2-ELIGIBLE, not emit a set.

    The floors are >= 8 of 67 paradigms and >= 200 pairs. A lexicon of two words clears neither, and
    the required behaviour is a refusal -- not a smaller set quietly labelled as frozen."""
    root = _fake_blimp(
        tmp_path,
        {"toy": [("the cat sat", "the cat sats"), ("the dog ran", "the dog rans")]},
    )
    monkeypatch.setattr(LC, "exposure_lexicon", lambda triples: {"the", "cat"})
    out_path = tmp_path / "coverage.md"
    rc = LC.main(["--blimp", root, "--out", str(out_path)])
    text = capsys.readouterr().out
    assert rc == 2, "a missed floor must exit 2"
    assert "NOT-P2-ELIGIBLE" in text
    assert "pair_ids: NONE" in text, "a refusal must not emit a pair set"
    written = out_path.read_text(encoding="ascii")
    assert "NOT-P2-ELIGIBLE" in written and "pair_ids: NONE" in written


def test_floors_pass_emits_a_set(tmp_path, monkeypatch, capsys):
    """The other side of T005: when the floors ARE met the tool exits 0 and emits the ids.

    Without this, `test_floors_exit_two` passes for a tool that can only ever exit 2 -- a gate that
    cannot go green is as uninformative as one that cannot go red."""
    paradigms = {
        f"toy{p}": [(f"the cat sat {i}", f"the cat sats {i}") for i in range(30)] for p in range(8)
    }
    root = _fake_blimp(tmp_path, paradigms)
    digits = {str(i) for i in range(30)}
    monkeypatch.setattr(
        LC, "exposure_lexicon", lambda triples: {"the", "cat", "sat", "sats"} | digits
    )
    out_path = tmp_path / "coverage.md"
    rc = LC.main(["--blimp", root, "--out", str(out_path)])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "outcome: FROZEN" in text
    assert "paradigms with at least one covered pair: 8 of 67" in text
    assert "covered pairs: 240 of 240" in text
    assert "toy0:0" in out_path.read_text(encoding="ascii")


def test_counts_print_as_n_of_m_and_are_ascii(tmp_path, monkeypatch, capsys):
    """A3/FR-019: `N of M`, never `N/M`, and ASCII only.

    The artifact fingerprint harvests bare ratios positionally, so one stray `N/M` changes every
    artifact's fingerprint."""
    root = _fake_blimp(tmp_path, {"toy": [("the cat sat", "the cat sats")]})
    monkeypatch.setattr(LC, "exposure_lexicon", lambda triples: {"the", "cat"})
    LC.main(["--blimp", root])
    text = capsys.readouterr().out
    text.encode("ascii")  # raises on any non-ASCII
    assert " of " in text
    for line in text.splitlines():
        if line.strip().startswith(("paradigms with", "covered pairs")):
            assert "/" not in line, f"bare ratio in a count line: {line!r}"


# ------------------------------------------------------------- T006: absent data SKIPs, not passes


def test_absent_data_skips_not_passes(tmp_path, capsys):
    """A8/US1 scenario 3: an empty BLiMP directory exits 0 with a SKIP, and the record says so.

    The failure this guards against is a downstream reader treating exit 0 as a passed A/B. The
    written record therefore carries `status: SKIP`, `pair_ids: NONE`, and the sentence that no arm
    may be scored against it."""
    empty = tmp_path / "empty"
    empty.mkdir()
    out_path = tmp_path / "coverage.md"
    rc = LC.main(["--blimp", str(empty), "--out", str(out_path)])
    text = capsys.readouterr().out
    assert rc == 0, "absent data is not a tool failure"
    assert text.startswith("SKIP lang_coverage:")
    written = out_path.read_text(encoding="ascii")
    assert "status: SKIP" in written
    assert "pair_ids: NONE" in written
    assert "NOT A FROZEN PAIR SET" in written
    assert "outcome: FROZEN" not in written
    assert "ADOPT" not in written


# ---------------------------------------------------------------------------------- misc invariants


def test_lexicon_is_from_the_exposure_plan_not_an_organ():
    """A1: the lexicon comes from the KB, and the module imports no organ to read it off."""
    src = open(_TOOL, encoding="utf-8").read()
    assert "language_organ" not in src, (
        "the coverage tool must not import the organ; a lexicon read off a trained organ is a "
        "lexicon that could have been tuned"
    )
    assert "world_kb_stream" in src
    triples = [("water", "phase", "liquid"), ("iron", "symbol", "fe")]
    assert LC.exposure_lexicon(triples) == {"water", "phase", "liquid", "iron", "symbol", "fe"}


def test_reproducible_membership():
    """A9: the same inputs give the same ids, byte for byte."""
    lexicon = {"a", "b", "c"}
    pairs = [("x:0", "a b", "b a"), ("x:1", "a z", "z a")]
    assert LC.restrict(lexicon, pairs) == LC.restrict(lexicon, pairs) == ["x:0"]


def test_nord_home_is_isolated():
    """A10: the always-on daemon writes ~/.snn_data_lake; the probe must not read it.

    The live env var cannot be asserted here -- the test suite's own conftest points NORD_HOME at a
    per-test directory, which would make this pass for the wrong reason. What is asserted instead is
    the property that matters: the tool allocates its own home, and the assignment happens BEFORE
    the first snn_data_lake import, since an import that reads the env var first is not isolated."""
    assert "lang_coverage_home_" in LC._ISOLATED_HOME
    assert os.path.isdir(LC._ISOLATED_HOME)
    src = open(_TOOL, encoding="utf-8").read()
    assert src.index('os.environ["NORD_HOME"]') < src.index("from snn_data_lake"), (
        "NORD_HOME must be redirected before the first snn_data_lake import"
    )


@pytest.mark.parametrize(
    "sentence,expected",
    [("The cat's hat.", ["the", "cat's", "hat"]), ("30 days", ["30", "days"])],
)
def test_one_tokeniser_for_both_sides(sentence, expected):
    """A lexicon built with a looser tokeniser than the membership test would overstate coverage."""
    assert LC.sentence_words(sentence) == expected
