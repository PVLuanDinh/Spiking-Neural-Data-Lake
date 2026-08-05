"""Exemption keys must be able to name ONE rule, and the def table must not collapse two.

Two defects motivated this, both measured on 2026-07-31:

1. `defs` was keyed (module, name), so `basal_ganglia/latent_q.py` -- which defines `update` on BOTH
   `LatentQ` (:102, genuinely dormant open work) and `_TabularQ` (:165, the self-check's control,
   correctly dead) -- collapsed to ONE entry whose surviving lineno named whichever class came last.
   The finding was therefore reported against the wrong class.

2. `EXEMPT_METHOD` was keyed by bare method name, so the only way to exempt the control was the key
   `"update"`, which would have exempted that verb across all 194 modules -- hiding the very defect
   class this gate exists to catch. The gate's own motivating case was a method named `learn`.

These tests pin the fix at the level that matters: an exemption may suppress exactly the rule it
names, and no other.
"""

import ast
import importlib

import pytest

gate = importlib.import_module("snn_data_lake.substrate.method_dormancy_gate")


def test_two_same_named_methods_in_one_module_stay_distinct():
    """The regression that reported a finding against the wrong class."""
    r = gate.audit()
    latent = [u for u in r["uncertain"] + r["dormant"] if "basal_ganglia.latent_q" in u]
    assert len(latent) == 2, f"expected LatentQ.update AND _TabularQ.update, got {latent}"
    assert any("LatentQ.update()" in u and ":102" in u for u in latent), latent
    assert any("_TabularQ.update()" in u and ":165" in u for u in latent), latent


def test_entries_name_their_class():
    """A finding that does not name its class cannot be dispositioned without ambiguity."""
    r = gate.audit()
    for entry in r["uncertain"] + r["dormant"]:
        head = entry.split("  [")[0]
        assert "." in head.split(" ", 1)[1], f"entry does not name a class: {entry}"


# The two dormant findings the gate reports when EXEMPT_METHOD is emptied. MEASURED 2026-08-04 by
# clearing the table and calling audit(); they are what makes the exemption branch REACHABLE, which
# is what the tests below need.
#
# MATCHED BY `Class.method()`, NEVER BY THE WHOLE ENTRY. The first version of these constants was
# the full entry string, line number included -- and it went red within the hour, because
# `hippocampus/affective_memory.py` gained two lines and `:173` became `:175`. A test that pins a
# line number is the stale-citation defect wearing a test's clothes: it fails on an edit that
# changed nothing it claims to be about. The class-qualified name is the identity the gate keys on,
# so it is the identity to assert; a genuine RENAME still fails here, loudly, which is what these
# constants are for.
_DORMANT_A = "AffectiveEpisodicStore.consolidate()"
_DORMANT_B = "DriveReductionLearner.learned_map()"


def _has(entries, needle):
    return any(needle in e for e in entries)


def _dormant_under(monkeypatch, table):
    """The gate's OWN dormant list with EXEMPT_METHOD replaced by `table`.

    Drives `audit()` rather than re-evaluating the lookup expression: an earlier version of these
    tests re-typed `f"{dm}.{dcls}.{name}" in EXEMPT_METHOD or ...` inline and asserted on THAT, so
    it was a test of Python's `in` operator against a dict the test itself had just built. It could
    not go red for any change to method_dormancy_gate.py -- deleting the exemption branch outright
    left it green. This routes every case through the shipped code path instead.
    """
    monkeypatch.setattr(gate, "EXEMPT_METHOD", dict(table))
    return gate.audit()["dormant"]


def test_the_exemption_branch_is_reachable_at_all(monkeypatch):
    """The control for the three tests below: with NO exemptions the gate must report findings.

    Without this, every assertion about suppression would pass vacuously on a gate that reports
    nothing whatever the table says -- the green-gate-that-cannot-see failure this suite exists for.
    """
    dormant = _dormant_under(monkeypatch, {})
    assert _has(dormant, _DORMANT_A), dormant
    assert _has(dormant, _DORMANT_B), dormant


@pytest.mark.parametrize(
    "key,suppresses",
    [
        # class-qualified -- the form new dispositions should use
        ("hippocampus.affective_memory.AffectiveEpisodicStore.consolidate", True),
        ("hippocampus.affective_memory.consolidate", True),  # module-qualified, legacy
        ("consolidate", True),  # bare, legacy
        ("hippocampus.affective_memory.SomeOtherClass.consolidate", False),  # a DIFFERENT class
        ("amygdala.affective_memory.AffectiveEpisodicStore.consolidate", False),  # another module
    ],
)
def test_exemption_lookup_precedence(key, suppresses, monkeypatch):
    """The lookup must honour the three key forms and reject keys that name something else.

    Measured through the gate: the finding either leaves `dormant` or it does not.
    """
    dormant = _dormant_under(monkeypatch, {key: ("permanent", "test")})
    assert (not _has(dormant, _DORMANT_A)) is suppresses, f"key {key!r} -> dormant {dormant}"


def test_a_class_qualified_exemption_cannot_suppress_a_sibling_finding(monkeypatch):
    """The whole point: exempting one rule must leave every other finding visible.

    A bare key (`"consolidate"`) would exempt that verb across all 194 modules, which is the
    defect the class-granular keying was introduced to end. Asserted as the CONTRAST between two
    real findings on the real tree -- the second must survive an exemption aimed at the first.
    """
    dormant = _dormant_under(
        monkeypatch,
        {"hippocampus.affective_memory.AffectiveEpisodicStore.consolidate": ("permanent", "x")},
    )
    assert not _has(dormant, _DORMANT_A), "the named rule was not suppressed"
    assert _has(dormant, _DORMANT_B), "an exemption for one rule silenced an unrelated one"


def test_parse_all_still_returns_modules():
    """Guard the AST layer the keying depends on."""
    trees = gate._parse_all()
    assert trees, "no modules parsed"
    assert all(isinstance(t, ast.Module) for t in trees.values())
    assert "substrate.method_dormancy_gate" in trees


def test_uncertain_is_reported_not_asserted():
    """`uncertain` is a list of open questions. Turning it into a failure would create pressure to
    silence them, which is how a gate stops meaning anything.

    The second assert read `r["dormant"] == [] or isinstance(r["dormant"], list)` until 2026-08-05.
    The right disjunct is true whenever the left one is, so the whole expression was `isinstance`
    with a decoration -- and the decoration read as if an EMPTY dormant list were the asserted
    property, which is the opposite of what this gate is for. Reduced to the shape check it always
    was; the behaviour of `dormant` is asserted by the exemption tests above, which drive it both
    ways.
    """
    r = gate.audit()
    assert isinstance(r["uncertain"], list)
    assert isinstance(r["dormant"], list)
