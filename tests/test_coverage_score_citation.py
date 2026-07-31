"""A citation must keep naming the code it was written about.

`_cite_ok` checked only that the file existed and the line number was inside it, so a cited file
that gained lines ABOVE the cited point left the number in range and pointing at something else,
while `score --check` stayed green. Real instance: `SpatialTemporal.md` cited `brain.py:1325` for
the place binding; a later commit added 14 lines above it and 1325 became a comment.

Two mechanisms are tested here. `#anchor` pins a citation to the def/class it sits in, so F5 can
refuse drift outright. `_cite_kind` is the census used to size the damage before anything leans on
the rule -- a citation naming a comment or a blank line has probably rotted.
"""

import importlib.util
import os
import textwrap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "coverage_score_cite", os.path.join(_ROOT, "tools", "coverage_score.py")
)
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)

_SRC = textwrap.dedent(
    '''\
    """mod docstring"""
    import os


    def alpha(x):
        # a comment inside alpha
        return x + 1


    class Holder:
        def beta(self):

            return 2
    '''
)


def _write(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text(_SRC, encoding="utf-8")
    return str(tmp_path), "sample.py"


def test_enclosing_symbol_finds_the_def_a_line_sits_in(tmp_path):
    root, rel = _write(tmp_path)
    lines = _SRC.splitlines()
    assert cs._enclosing_symbol(lines, 7) == "alpha"  # `return x + 1`
    assert cs._enclosing_symbol(lines, 13) == "beta"  # inside the method
    assert cs._enclosing_symbol(lines, 2) == "module"  # import, top level
    assert cs._cite_ok(root, f"{rel}:7#alpha")[0] is True


def test_a_drifted_anchor_is_refused_with_the_symbol_it_landed_in(tmp_path):
    root, rel = _write(tmp_path)
    ok, why = cs._cite_ok(root, f"{rel}:7#beta")
    assert ok is False
    assert "anchor drift" in why and "alpha" in why, why


def test_a_bare_citation_still_resolves_so_the_ledger_does_not_break(tmp_path):
    """Anchors are OPTIONAL. Every existing row cites `path:line`; adding the anchor form must not
    invalidate them, or the instrument would refuse the whole ledger the day it shipped."""
    root, rel = _write(tmp_path)
    assert cs._cite_ok(root, f"{rel}:7")[0] is True


def test_cite_kind_separates_code_from_the_symptoms_of_rot(tmp_path):
    root, rel = _write(tmp_path)
    assert cs._cite_kind(root, f"{rel}:7") == "code"
    assert cs._cite_kind(root, f"{rel}:6") == "comment"
    assert cs._cite_kind(root, f"{rel}:12") == "blank"


def test_kind_reads_through_an_anchor_and_never_raises_on_junk(tmp_path):
    root, rel = _write(tmp_path)
    assert cs._cite_kind(root, f"{rel}:7#alpha") == "code"
    assert cs._cite_kind(root, f"{rel}:9999") == ""
    assert cs._cite_kind(root, "no_such_file.py:1") == ""
    assert cs._cite_kind(root, "not-a-citation") == ""


def test_past_eof_is_still_a_hard_failure(tmp_path):
    root, rel = _write(tmp_path)
    ok, why = cs._cite_ok(root, f"{rel}:9999")
    assert ok is False and "past EOF" in why
