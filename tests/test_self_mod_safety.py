"""The self-modification safety kernel is a list of FILE PATHS, and paths go stale silently.

Several modules name the files the brain may not rewrite. Since 2026-08-18 they all derive from ONE
definition, `substrate/safety_envelope.ENVELOPE`, and each adds only what is genuinely its own:

  * `safety_envelope.ENVELOPE`     -- THE five brainstem files, package-relative. One definition.
  * `godel_machine._SAFETY_KERNEL` -- + run_selfchecks.py + itself, repo-relative (the verifier).
  * `self_improve.PROTECTED`       -- + its own proposer, package-relative.
  * `gradient_claim_gate`          -- the envelope alone, aliased.

Neither list is checked against the filesystem anywhere else. A rename or deletion that misses an
entry does not fail loudly -- the path simply stops matching any file, and the protection evaporates
while `NORD_SELFMOD=1` keeps autonomous self-build running every 8 nights. That is a silent widening
of what the brain is allowed to rewrite, and it includes the arm's E-STOP.

These tests exist so that cannot happen quietly. They already caught one: `PROTECTED` named
"research/self_improve.py", which resolved nowhere, so the self-improver was not protecting its own
proposer.

WHAT CHANGED WITH THE CONSOLIDATION, stated because a deleted test is worth more scrutiny than an
added one. `_protected_paths()` used to PARSE `PROTECTED` out of `self_improve.py` with a regex,
because the set was a local inside a function and could not be imported -- the docstring called that
"the price of checking it at all". `PROTECTED` is module scope now, so the parser is DELETED rather
than kept beside the import (P2: a merge must reduce).
`test_arm_safety_files_are_named_by_both_lists` compared the two lists by basename; that comparison
is now true by construction, so it is replaced by
`test_every_guard_derives_from_the_one_envelope`, which asserts the property that actually carries
the guarantee -- each consumer still CONTAINS all five, whatever spelling it uses. The
does-it-exist-on-disk half is unchanged and is what neither construction nor import can prove.
"""

import pathlib

from snn_data_lake.reasoning.godel_machine import _SAFETY_KERNEL
from snn_data_lake.reasoning.self_improve import PROTECTED
from snn_data_lake.substrate import safety_envelope
from snn_data_lake.substrate.gradient_claim_gate import _SAFETY_ENVELOPE

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "snn_data_lake"


def test_godel_safety_kernel_paths_all_exist():
    """Every immutable-axiom path must name a real file."""
    missing = [p for p in _SAFETY_KERNEL if not (REPO_ROOT / p).exists()]
    assert not missing, (
        f"_SAFETY_KERNEL names {len(missing)} path(s) that do not exist: {missing}. "
        "A protected path that matches no file protects nothing -- the brain may now rewrite it."
    )


def test_self_improve_protected_paths_all_exist():
    """Every PROTECTED path must resolve, package-relative or repo-relative."""
    missing = [p for p in PROTECTED if not (PKG_ROOT / p).exists() and not (REPO_ROOT / p).exists()]
    assert not missing, (
        f"PROTECTED names {len(missing)} path(s) that resolve nowhere: {missing}. "
        "Package-relative is the convention here; a stale entry silently protects nothing."
    )


def test_the_envelope_itself_exists_on_disk():
    """The one definition must name five real files. Everything below inherits from this."""
    missing = [p for p in safety_envelope.ENVELOPE if not (PKG_ROOT / p).exists()]
    assert not missing, (
        f"safety_envelope.ENVELOPE names {len(missing)} path(s) that do not exist: {missing}. "
        "This is the ONE definition -- a stale entry here unprotects the file in every consumer."
    )


def test_every_guard_derives_from_the_one_envelope():
    """Each consumer must still contain all five, in whatever spelling it uses.

    Normalized rather than compared by basename. A basename comparison would accept
    `reasoning/brainstem/reflex.py` as if it were the envelope's `reflex.py`, which is the exact
    suffix-match hazard `safety_envelope.normalize` exists to refuse.
    """
    envelope = set(safety_envelope.ENVELOPE)
    for name, listing in (
        ("godel_machine._SAFETY_KERNEL", _SAFETY_KERNEL),
        ("self_improve.PROTECTED", PROTECTED),
        ("gradient_claim_gate._SAFETY_ENVELOPE", _SAFETY_ENVELOPE),
    ):
        got = {safety_envelope.normalize(p) for p in listing}
        assert envelope <= got, (
            f"{name} is missing {sorted(envelope - got)}. Every guard must protect all five; "
            "one that does not has silently stopped guarding what stops the brain."
        )


def test_the_document_and_the_code_agree_about_the_envelope():
    """SECURITY.md's table is the authority; the code must match it.

    The reds for this comparison are proven by
    `python -m snn_data_lake.substrate.safety_envelope --self-check`, which plants six mismatches
    against the live document. This test asserts the live tree is on the green side of it.
    """
    findings, path = safety_envelope.check_document(root=str(REPO_ROOT))
    assert findings == [], f"{path} and the code disagree about the safety envelope: {findings}"


def test_the_document_check_can_go_red():
    """A reachable red, here and not only in the module self-check.

    Without this, the test above is a check nobody has shown can fail, which is the shape this
    project deletes rather than trusts.
    """
    text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    planted = text.replace(
        "src/snn_data_lake/" + safety_envelope.ENVELOPE[0],
        "src/snn_data_lake/brainstem/zz_not_the_envelope.py",
    )
    assert planted != text, "the red could not be planted -- SECURITY.md no longer names that path"
    assert safety_envelope.document_findings(planted), (
        "a renamed envelope entry in SECURITY.md produced NO finding -- the check is vacuous"
    )
