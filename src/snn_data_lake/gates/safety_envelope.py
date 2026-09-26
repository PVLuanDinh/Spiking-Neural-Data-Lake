r"""THE SAFETY ENVELOPE -- one definition of the five approval-gated `brainstem/` files.

# consolidates: src/snn_data_lake/reasoning/godel_machine.py -- `_SAFETY_KERNEL`'s five brainstem
#   entries (repo-relative spelling), retyped there; it now imports them and adds only the two
#   verifier files that are its own.
# consolidates: src/snn_data_lake/reasoning/self_improve.py -- `PROTECTED`'s five entries
#   (package-relative, a function local), retyped there; it now imports them and adds only its own
#   proposer.
# consolidates: src/snn_data_lake/gates/gradient_claim_gate.py -- `_SAFETY_ENVELOPE`, the third
#   copy; the name survives as an alias of this constant, so its two use sites are unchanged.

`SECURITY.md` is the human-facing authority on what the envelope IS; this module is the ONE place
`src/` says it in code. The three modules above used to say it separately, in two different path
spellings, and nothing kept them in step. They AGREED when measured on 2026-08-18, so this closes a
LATENT defect rather than an active one -- which is exactly when it is cheap to close.

WHY A LATENT ONE IS WORTH THE MODULE. The guards these lists feed are the guards that stop the brain
modifying what stops it (P0: *"the brain may rewrite what it thinks, never what stops it"*). A list
that misses a file does not fail -- it silently stops protecting that file while every gate stays
green, which is the same silent-widening shape `tests/test_self_mod_safety.py` was written for after
`PROTECTED` was found naming `research/self_improve.py`, a path that resolved nowhere. Under P2 the
resolution is not a fourth checker over three lists: exactly one definition survives and the others
IMPORT it. There is nothing left to keep in sync.

WHAT THIS MODULE IS NOT. It is not a fourth guard and it enforces no policy about who may edit what
-- that stays with each consumer, because each protects a DIFFERENT set. Every superset is a real
capability and is PORTED, not dropped (P2's honesty guard: port the unique capability FIRST):

    godel_machine._SAFETY_KERNEL  = envelope + run_selfchecks.py + godel_machine.py  (the verifier)
    self_improve.PROTECTED        = envelope + reasoning/self_improve.py             (its proposer)
    gradient_claim_gate           = envelope alone (a rule in it is never asked to change)
    self_mod_harness._PROTECTED   = _SAFETY_KERNEL + its own two files   (already derived -- and so
    self_build_loop._FIREWALLED   = _PROTECTED + its own accept path      is this. Both unchanged.)

Only the shared five stop being retyped. The derivation tree already rooted at `_SAFETY_KERNEL` is
re-rooted here; no link in it is added or removed.

THE SELF-CHECK IS THE HONESTY HALF. Code agreeing with itself is free once there is one definition;
code agreeing with the DOCUMENT is not, and prose is where this drifts -- `SECURITY.md`'s own
envelope paragraph has been corrected twice, for naming a class that does not exist and an arm with
the wrong number of joints. `document_findings()` compares this constant against the table
`SECURITY.md` enumerates, and `--self-check` plants six mismatches to prove the comparison can go
red (P8: a criterion with no reachable red is deleted, not trusted).

Deps: stdlib only, and deliberately NO intra-package import -- `godel_machine` is imported BY
`substrate/self_mod_harness`, so any import back into the package from here would close a cycle.

  PYTHONPATH=src python -m snn_data_lake.gates.safety_envelope               # assert
  PYTHONPATH=src python -m snn_data_lake.gates.safety_envelope --self-check  # prove the reds
"""

import os
import re
import sys

#: The package-relative spelling is CANONICAL. Both spellings exist in the tree because the two
#: consumers resolve against different roots (`godel_machine` matches `git status` output at the
#: repo root; `self_improve` names files inside the package), so the constant is stored in ONE form
#: and `repo_relative()` derives the other. That is the form half of this consolidation: a match can
#: no longer fail because two lists chose different spellings of the same file.
_PKG_PREFIX = "src/snn_data_lake/"

#: THE five files. Modified only with explicit approval (POLICIES.md cross-cutting invariants;
#: SECURITY.md "Changing the envelope"). `so101_arm.py` is here from BEFORE it became the live gate,
#: deliberately: the window between "so101_arm is the live gate" and "so101_arm is protected" is a
#: window in which the self-rewrite harness may rewrite the arm's E-STOP, and there is no acceptable
#: length for that window.
ENVELOPE = (
    "brainstem/reflex.py",
    "brainstem/serial_arm.py",
    "brainstem/so101_arm.py",
    "brainstem/arm_config.py",
    "brainstem/interpreter.py",
)

#: The document that is the authority. Compared against `ENVELOPE` by `main()` on every gate run.
SECURITY_DOC = "SECURITY.md"

#: A markdown table row whose first cell is a single backticked token: ``| `path` | role |``.
_DOC_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")

_SECTION_HEADING = "## safety envelope"


def repo_relative():
    """`ENVELOPE` spelled from the repository root, for callers matching `git status` output."""
    return tuple(_PKG_PREFIX + p for p in ENVELOPE)


def normalize(path):
    """One spelling for a path that may arrive prefixed, bare, backslashed or `./`-led.

    ANCHORED, never a suffix match. `str.endswith("brainstem/reflex.py")` also accepts
    `src/snn_data_lake/reasoning/brainstem/reflex.py` -- a hazard `self_mod_harness` records having
    traded away once -- so the prefix is STRIPPED and the remainder compared whole. A path outside
    the package normalizes to itself and simply fails to match, rather than matching by accident.

    # distinct-from: src/snn_data_lake/substrate/self_mod_harness.py::_rel_to_repo -- that resolves
    #   a REAL filesystem path against the repo root via os.path.relpath, so it needs the file to be
    #   locatable on disk and cannot be reused here: this one normalizes a STRING SPELLING, and the
    #   strings it is handed come out of a markdown document and routinely name a file that has been
    #   renamed away. It is also unimportable from here without closing a cycle
    #   (self_mod_harness -> godel_machine -> this module).
    # distinct-from: src/snn_data_lake/reasoning/self_improve.py::_self_pkg_relpath -- that derives
    #   ONE path, its own module's, from `__file__` by walking up past `__init__.py`. It takes no
    #   argument and answers "where am I", not "are these two spellings the same file".
    """
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if p.startswith(_PKG_PREFIX):
        p = p[len(_PKG_PREFIX) :]
    return p


def is_envelope(path):
    """True when `path`, in either spelling, names one of the five. Exact after normalization."""
    return normalize(path) in ENVELOPE


# ---------------------------------------------------------------- the document agreement check ----


def _envelope_section(text):
    """The lines under `SECURITY.md`'s `## Safety envelope` heading, or None when it is GONE.

    None rather than an empty list, and the distinction is the whole point: an empty match reads as
    "the document lists nothing that disagrees", which is a clean bill of health handed out by a
    parser that found nothing to read. The caller turns None into a FINDING.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == _SECTION_HEADING:
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return lines[start:end]


def _enumerated(section):
    """The normalized paths the table rows in `section` name."""
    out = set()
    for line in section:
        m = _DOC_ROW.match(line.strip())
        if m:
            out.add(normalize(m.group(1)))
    return out


def document_findings(text):
    """Disagreements between `ENVELOPE` and the table `SECURITY.md` enumerates. Returns a list.

    PURE -- it is handed the document TEXT and reads no file, so `--self-check` can plant a mismatch
    without editing the safety document to prove the check fires. A check never shown to produce a
    finding cannot evidence the absence of one.

    BOTH directions are reported. A rename in the document surfaces as one finding of each kind
    rather than vanishing, and reporting only "missing from the doc" would let a sixth entry be
    added to `SECURITY.md` and read as agreement.

    LIMIT, stated because a green here is easy to over-read: it compares the TABLE, which is the
    enumeration. Prose elsewhere in the section that names an envelope file is not checked -- and
    the two corrections `SECURITY.md` already carries were both in prose. The table is what a reader
    treats as the list; the prose is what a reader treats as description.
    """
    section = _envelope_section(text)
    if section is None:
        return [
            f"{SECURITY_DOC} has no '## Safety envelope' section -- the enumeration this check "
            f"compares against is GONE, so nothing was compared"
        ]
    doc = _enumerated(section)
    if not doc:
        return [
            f"{SECURITY_DOC}'s '## Safety envelope' section enumerates NO file -- the comparison "
            f"would be vacuous, which is not the same as agreement"
        ]
    code = set(ENVELOPE)
    out = [f"IN CODE, NOT IN {SECURITY_DOC}: {p}" for p in sorted(code - doc)]
    out += [f"IN {SECURITY_DOC}, NOT IN CODE: {p}" for p in sorted(doc - code)]
    return out


def _repo_root():
    """The repo root, found by walking up for `pyproject.toml`. None in an installed wheel.

    Keyed on `pyproject.toml` rather than on `SECURITY.md` DELIBERATELY: keying on the document
    would make a DELETED document indistinguishable from "not running in a repo", and the check
    would skip silently in exactly the state it exists to catch.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(here, "pyproject.toml")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def check_document(root=None):
    """(findings, doc_path) against the real `SECURITY.md`; (None, None) when not in a repo."""
    root = _repo_root() if root is None else root
    if root is None:
        return None, None
    path = os.path.join(root, SECURITY_DOC)
    if not os.path.exists(path):
        return [f"{SECURITY_DOC} is MISSING at {path} -- the envelope's authority is gone"], path
    with open(path, encoding="utf-8") as fh:
        return document_findings(fh.read()), path


# ---------------------------------------------------------------- the self-check ------------------

_ROW_TEMPLATE = "| `{}` | role |"


def _row_line(text, pkg_rel):
    """The exact line in `text` whose table row names `pkg_rel`. Raises if absent (never silent)."""
    for line in text.splitlines():
        m = _DOC_ROW.match(line.strip())
        if m and normalize(m.group(1)) == pkg_rel:
            return line
    raise AssertionError(f"no table row for {pkg_rel} -- the red could not be constructed")


def _selfcheck():
    """Construct each RED against the REAL `SECURITY.md`. Verify the verifier.

    A self-check that only runs the happy path verifies nothing. Every red below is CONSTRUCTED from
    the live document and then confirmed to fire, and two GREEN controls prove the check is not
    simply always-red -- RED 5 specifically proves `normalize()` is not matching by suffix, which is
    the one defect that would let every other red pass while the check protected nothing.
    """
    ok = True
    findings, path = check_document()
    if findings is None:
        print("FAIL   no repo root found -- every red below would be vacuous")
        return False
    with open(path, encoding="utf-8") as fh:
        real = fh.read()

    # CONTROL -- the live document agrees, and agrees about FIVE files. A control that passed on an
    # empty parse would make every red below meaningless.
    section = _envelope_section(real)
    parsed = _enumerated(section) if section is not None else set()
    print(f"[control ] live {SECURITY_DOC} -> {len(parsed)} enumerated, {len(findings)} finding(s)")
    if findings or len(parsed) != len(ENVELOPE):
        print(f"  FAIL   the control is not clean and complete: {findings}")
        return False

    def red(label, text, expect):
        nonlocal ok
        got = document_findings(text)
        fired = any(expect in f for f in got)
        print(f"  {'RED  ' if fired else 'FAIL '}  {label} -> {got if got else 'NO FINDING'}")
        ok = ok and fired

    def green(label, text):
        nonlocal ok
        got = document_findings(text)
        print(f"  {'GREEN' if not got else 'FAIL '}  {label} -> {got if got else 'no finding'}")
        ok = ok and not got

    one = ENVELOPE[0]
    row = _row_line(real, one)

    # RED 1 -- a file the code protects vanishes from the document's table.
    red(
        f"{one} dropped from the table",
        real.replace(row, ""),
        f"IN CODE, NOT IN {SECURITY_DOC}: {one}",
    )

    # RED 2 -- the document grows a sixth entry the code does not protect.
    sixth = _ROW_TEMPLATE.format(_PKG_PREFIX + "brainstem/zz_new.py")
    red(
        "a sixth file added to the table",
        real.replace(row, row + "\n" + sixth),
        f"IN {SECURITY_DOC}, NOT IN CODE: brainstem/zz_new.py",
    )

    # RED 3 -- a rename. Fires in BOTH directions; the second is what a one-way check would miss.
    renamed = real.replace(_PKG_PREFIX + one, _PKG_PREFIX + "brainstem/reflex_v2.py")
    red("a renamed entry (code side)", renamed, f"IN CODE, NOT IN {SECURITY_DOC}: {one}")
    red(
        "a renamed entry (doc side)",
        renamed,
        f"IN {SECURITY_DOC}, NOT IN CODE: brainstem/reflex_v2.py",
    )

    # RED 4 -- the section itself is deleted. Proves an empty parse cannot read as agreement.
    red(
        "the '## Safety envelope' section deleted",
        real.replace("## Safety envelope", "## Something Else"),
        "has no '## Safety envelope' section",
    )

    # RED 5 -- the CONTROL ON `normalize()`, and the reason it is worth its own case. A path that
    # merely ENDS with an envelope name is a DIFFERENT file. Without this, a normalizer that matched
    # on suffix -- or one that collapsed every input to a constant -- would pass RED 1-4 and both
    # greens while the check protected nothing.
    red(
        "a same-named file at another path (reasoning/brainstem/reflex.py)",
        real.replace(_PKG_PREFIX + one, _PKG_PREFIX + "reasoning/" + one),
        f"IN CODE, NOT IN {SECURITY_DOC}: {one}",
    )

    # GREEN 1 -- the whole table respelled BARE. The two spellings must not be able to disagree.
    green("every row respelled package-relative (bare)", real.replace("`" + _PKG_PREFIX, "`"))

    # GREEN 2 -- the untouched document, re-derived AFTER the mutations, so a red that accidentally
    # mutated shared state shows up here rather than passing quietly.
    green("the live document, unmutated", real)

    print("self-check OK: 6/6 reds fired, 2/2 greens clean" if ok else "self-check FAILED")
    return ok


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--self-check" in argv:
        return 0 if _selfcheck() else 1

    print(
        f"SAFETY ENVELOPE -- {len(ENVELOPE)} files, ONE definition, checked against {SECURITY_DOC}"
    )
    for p in ENVELOPE:
        print(f"  {p}")
    findings, path = check_document()
    if findings is None:
        print("NOT ELIGIBLE: no repo root (installed wheel) -- nothing to compare against.")
        return 0
    if findings:
        print(f"gate FAIL: code and {SECURITY_DOC} disagree about the envelope:")
        for f in findings:
            print(f"  {f}")
        print("Resolve by AGREEING them deliberately -- the envelope changes only with approval.")
        return 1
    print(f"gate OK: {os.path.basename(path)} enumerates exactly these five")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
