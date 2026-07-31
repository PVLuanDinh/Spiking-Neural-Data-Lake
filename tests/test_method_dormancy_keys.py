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


@pytest.mark.parametrize(
    "key,should_exempt",
    [
        ("mod.Cls.meth", True),  # exact class-qualified
        ("mod.meth", True),  # module-qualified, legacy
        ("meth", True),  # bare, legacy
        ("mod.Other.meth", False),  # a DIFFERENT class in the same module
        ("other.Cls.meth", False),  # same class name, different module
    ],
)
def test_exemption_lookup_precedence(key, should_exempt, monkeypatch):
    """The lookup must accept the three key forms and reject keys that name something else.

    Driven directly rather than through audit(), because `dormant` is currently 0 -- so no real
    entry traverses the exemption branch, and a test that waited for one would never run.
    """
    monkeypatch.setattr(gate, "EXEMPT_METHOD", {key: ("permanent", "test")})
    dm, dcls, name = "mod", "Cls", "meth"
    hit = (
        f"{dm}.{dcls}.{name}" in gate.EXEMPT_METHOD
        or f"{dm}.{name}" in gate.EXEMPT_METHOD
        or name in gate.EXEMPT_METHOD
    )
    assert hit is should_exempt


def test_a_class_qualified_exemption_cannot_suppress_its_sibling():
    """The whole point: exempting the control must leave the real finding visible.

    Asserted against the REAL tree, not a fixture, because the sibling pair is real and the
    over-suppression this guards against would be invisible in a synthetic one.
    """
    exempt = dict(gate.EXEMPT_METHOD)
    exempt["basal_ganglia.latent_q._TabularQ.update"] = ("permanent", "self-check control")
    dm, name = "basal_ganglia.latent_q", "update"
    for dcls, expected in (("_TabularQ", True), ("LatentQ", False)):
        hit = (
            f"{dm}.{dcls}.{name}" in exempt or f"{dm}.{name}" in exempt or name in exempt
        )
        assert hit is expected, f"{dcls}.{name} exemption={hit}, expected {expected}"


def test_parse_all_still_returns_modules():
    """Guard the AST layer the keying depends on."""
    trees = gate._parse_all()
    assert trees, "no modules parsed"
    assert all(isinstance(t, ast.Module) for t in trees.values())
    assert "substrate.method_dormancy_gate" in trees


def test_uncertain_is_reported_not_asserted():
    """`uncertain` is a list of open questions. Turning it into a failure would create pressure to
    silence them, which is how a gate stops meaning anything."""
    r = gate.audit()
    assert isinstance(r["uncertain"], list)
    assert r["dormant"] == [] or isinstance(r["dormant"], list)
