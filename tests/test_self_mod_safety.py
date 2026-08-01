"""The self-modification safety kernel is a list of FILE PATHS, and paths go stale silently.

Two independent lists name the files the brain may not rewrite:

  * `godel_machine._SAFETY_KERNEL` -- repo-relative paths, the immutable axioms of the Godel
    machine's real-mode rewrite protocol (`attempt_on_repo(..., protected=_SAFETY_KERNEL)`).
  * `self_improve`'s `PROTECTED` -- package-relative paths, the same idea for the simpler
    self-improver.

Neither is checked against the filesystem anywhere. A rename or deletion that misses one entry
does not fail loudly -- the path simply stops matching any file, and the protection evaporates
while `NORD_SELFMOD=1` keeps autonomous self-build running every 8 nights. That is a silent
widening of what the brain is allowed to rewrite, and it includes the arm's E-STOP.

These tests exist so that cannot happen quietly. They already caught one: `PROTECTED` named
"research/self_improve.py", which resolved nowhere, so the self-improver was not protecting its
own proposer.

  pytest tests/test_self_mod_safety.py
"""

import pathlib
import re

from snn_data_lake.reasoning.godel_machine import _SAFETY_KERNEL

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "src" / "snn_data_lake"


def _protected_paths():
    """Read `PROTECTED` out of self_improve.py.

    It is a local inside a function, so it cannot be imported. Parsing the source is the price
    of checking it at all, and checking it is worth more than the fragility costs -- an unparsed
    list is exactly how the stale entry survived.
    """
    src = (PKG_ROOT / "reasoning" / "self_improve.py").read_text(encoding="utf-8")
    block = re.search(r"PROTECTED = \{(.*?)\}", src, re.S)
    assert block, "PROTECTED set not found in reasoning/self_improve.py -- did it move?"
    # Strip comments first. A quoted path inside an explanatory comment is not an entry, and
    # reading one as an entry is a false positive that looks exactly like a real stale path.
    body = "\n".join(line.split("#", 1)[0] for line in block.group(1).splitlines())
    paths = re.findall(r'"([^"]+)"', body)
    assert paths, "PROTECTED parsed as empty -- the regex no longer matches its shape"
    return paths


def test_godel_safety_kernel_paths_all_exist():
    """Every immutable-axiom path must name a real file."""
    missing = [p for p in _SAFETY_KERNEL if not (REPO_ROOT / p).exists()]
    assert not missing, (
        f"_SAFETY_KERNEL names {len(missing)} path(s) that do not exist: {missing}. "
        "A protected path that matches no file protects nothing -- the brain may now rewrite it."
    )


def test_self_improve_protected_paths_all_exist():
    """Every PROTECTED path must resolve, package-relative or repo-relative."""
    missing = [
        p
        for p in _protected_paths()
        if not (PKG_ROOT / p).exists() and not (REPO_ROOT / p).exists()
    ]
    assert not missing, (
        f"PROTECTED names {len(missing)} path(s) that resolve nowhere: {missing}. "
        "Package-relative is the convention here; a stale entry silently protects nothing."
    )


def test_arm_safety_files_are_named_by_both_lists():
    """The arm-safety files must appear in BOTH lists, so a rename cannot half-land.

    Compared by basename because the two lists use different path roots. Derived from
    `_SAFETY_KERNEL` rather than hardcoded, so this keeps working when the arm module is
    renamed -- which is precisely the change these tests are here to police.
    """
    kernel_brainstem = {pathlib.PurePosixPath(p).name for p in _SAFETY_KERNEL if "/brainstem/" in p}
    protected = {pathlib.PurePosixPath(p).name for p in _protected_paths()}
    assert kernel_brainstem, "no brainstem files in _SAFETY_KERNEL -- the arm left the kernel"
    gap = kernel_brainstem - protected
    assert not gap, (
        f"{sorted(gap)} are protected by the Godel kernel but NOT by self_improve's PROTECTED. "
        "Both lists must be updated together or one path of self-modification loses the guard."
    )
