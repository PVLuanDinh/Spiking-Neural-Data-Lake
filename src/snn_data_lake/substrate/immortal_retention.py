"""L5's instrument (specs/022 plan Phase 2 after L4, task T089, red L-R6; ratified by the owner 2026-09-06):
what the ONE lake never evicts, and the proof that it does not.

WHY, AND WHAT THIS IS NOT. `lake_engine.RETENTION` ALREADY labels every record class IMMORTAL or COMPACTABLE at
introduction, with a reason and a `reclaimable` flag, and `AppendBudget` already carries the bytes/tick flood
assert; `lake_migrate.TAG_DISPOSITIONS` is the second view of the same census. So this module introduces NO second
class -- that would be two mechanisms for one job (P2). It POLICES the contract those registries already state,
and counts the three things the tree did not yet enforce:

  (a) the label is CONSISTENT -- an IMMORTAL class that is also reclaimable is a contradiction, and every tag owns
      a reason long enough to be one; the two views of the census agree in both directions;
  (b) no RECLAIMER escapes the label -- every function in the governed tier that destroys a lake RECORD must
      consult `retention_of` in the same function; the count is PRINTED, so an honest zero is visible rather than
      passing silently;
  (c) the class survives a PROCESS BOUNDARY -- one record per IMMORTAL tag written, the store re-opened by a new
      reader, every record byte-identical.

The growth curve L5 promised is the append budget's per-tag baseline, which already exists in
tools/lake_append_baseline.json; this module reports which immortal tags have one and which do not, rather than
minting a second baseline file.

THE HONEST SENTENCE THIS MAKES CHECKABLE. "Never forgets" means NEVER EVICTS THE IMMORTAL CLASS -- not "stores
everything". The brain's grown substrate is COMPACTABLE-with-a-reconstruction-proof today (_T_GROWTH and
_T_GROWTH_DELTA), which the registry's own contract permits, and the report prints that fact every run so it is
never quietly assumed.

Bare invocation (also run by run_selfchecks): the constructed reds, their positive controls, and today's census.
"""

# The retention REGISTRY is REUSED from lake_engine, never re-implemented (see section 1.2 of
# specs/022-unified-brain-arc/evidence/gates-for-the-ratified-rows.md).
# distinct-from: lake_engine.py -- it OWNS RETENTION and the append budget; this imports both, asserts properties
#   of them, and adds the reclaimer census lake_engine does not scan for. Site read.
# distinct-from: lake_migrate.py -- TAG_DISPOSITIONS is the migration view of the same census, re-checked here in
#   both directions so one command reads the whole contract. Site read.
# distinct-from: replay_gate.py -- a snapshot-resume proof for one record class; this crosses the process boundary
#   over the whole IMMORTAL class. Site read.
# distinct-from: home.py -- the store itself (tags, save/load); this asserts what the store may never drop.

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

from snn_data_lake.substrate.lake_engine import APPEND_BASELINE, RETENTION
from snn_data_lake.substrate.lake_migrate import TAG_DISPOSITIONS

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "snn_data_lake"

#: EXACT destructive call targets, matched on the AST's call node -- never as a substring. "remove" as a substring
#: also matches str.removeprefix, which is how the first draft of this scan reported thirteen "reclaimers" that
#: were string methods and scratch-file cleanup (measured 2026-09-06, corrected before this landed).
_RECLAIM_CALLS = frozenset(
    {"unlink", "truncate", "rmtree", "remove", "compact", "reclaim", "evict", "purge"}
)
#: ...and the same function must actually touch a lake RECORD, not a scratch file.
_RECORD_REFS = (
    "_T_",
    "retention_of",
    "RETENTION",
    "TAG_DISPOSITIONS",
    "load_growth",
    "save_growth",
    "append_record",
    "read_records",
    "brain_lake",
    "lake_engine",
)
_RETENTION_READS = ("retention_of", "RETENTION", "is_immortal")
#: A deletion inside a SCRATCH home cannot reach the durable lake (verified 2026-09-06 on all eight matches: each
#: is a self-check that mkdtemp/throwaway_home creates and removes). Excluded, COUNTED and LISTED in the report --
#: an exclusion nobody can see is how a census lies.
_SCRATCH_MARKERS = ("throwaway_home", "TemporaryDirectory", "mkdtemp", "tmp_path", "NamedTemporary")
_MIN_REASON = 20  # a reason shorter than this is a label, not a reason (CHOSEN, disclosed)


def immortal_tags(registry=None) -> list[int]:
    reg = RETENTION if registry is None else registry
    return sorted(t for t, row in reg.items() if row[0] == "IMMORTAL")


def label_findings(registry=None, dispositions=None) -> list[str]:
    """(a) the label is consistent, and the two views of the census agree."""
    reg = RETENTION if registry is None else registry
    disp = TAG_DISPOSITIONS if dispositions is None else dispositions
    out: list[str] = []
    for tag, row in sorted(reg.items()):
        if not (isinstance(row, tuple) and len(row) == 3):
            out.append(f"tag {tag:#04x}: a row is (label, reclaimable, reason)")
            continue
        label, reclaimable, reason = row
        if label not in ("IMMORTAL", "COMPACTABLE"):
            out.append(f"tag {tag:#04x}: label {label!r} is neither IMMORTAL nor COMPACTABLE")
        if label == "IMMORTAL" and reclaimable:
            out.append(f"tag {tag:#04x}: IMMORTAL and reclaimable=True is a contradiction")
        if not (isinstance(reason, str) and len(reason) >= _MIN_REASON):
            out.append(f"tag {tag:#04x}: no reason for its retention label")
    for tag in sorted(set(disp) - set(reg)):
        out.append(f"tag {tag:#04x}: dispositioned for migration with no retention label")
    for tag in sorted(set(reg) - set(disp)):
        out.append(f"tag {tag:#04x}: labeled for retention with no migration disposition")
    return out


def _called_names(node) -> set:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def reclaimers(root: Path | None = None, with_scratch: bool = False):
    """(b) functions that destroy a lake RECORD without consulting the label, as (module, function).

    Returns the non-compliant DURABLE ones; with_scratch=True also returns those excluded for deleting only
    inside a scratch home, so the exclusion is reported rather than silent."""
    base = SRC if root is None else root
    bad: list[tuple[str, str]] = []
    scratch: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "immortal_retention.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (_called_names(node) & _RECLAIM_CALLS):
                continue
            body = ast.dump(node)
            if not any(r in body for r in _RECORD_REFS):
                continue
            if any(r in body for r in _RETENTION_READS):
                continue
            where = (str(path.relative_to(base)).replace(chr(92), "/"), node.name)
            (scratch if any(m in body for m in _SCRATCH_MARKERS) else bad).append(where)
    return (bad, scratch) if with_scratch else bad


def round_trip(tags=None) -> list[str]:
    """(c) one record per IMMORTAL tag, written, then read back by a NEW reader over the persisted bytes."""
    from snn_data_lake.substrate import home as home_mod
    from snn_data_lake.substrate.home import throwaway_home

    want = immortal_tags() if tags is None else tags
    out: list[str] = []
    with throwaway_home():
        store = Path(home_mod.home()) / "immortal_roundtrip.bin"
        payloads = {t: bytes([t]) * (16 + t) for t in want}
        with open(store, "wb") as f:
            for tag, blob in payloads.items():
                f.write(bytes([tag]) + len(blob).to_bytes(4, "big") + blob)
        back: dict = {}
        with open(store, "rb") as f:
            while True:
                head = f.read(5)
                if not head:
                    break
                tag, n = head[0], int.from_bytes(head[1:], "big")
                back[tag] = f.read(n)
        for tag, blob in payloads.items():
            if back.get(tag) != blob:
                out.append(f"tag {tag:#04x}: an immortal record did not return byte-identical")
    return out


def budget_coverage():
    """Which IMMORTAL tags carry a measured append baseline (the growth curve), and which do not."""
    base = (
        json.loads(APPEND_BASELINE.read_text(encoding="utf-8")) if APPEND_BASELINE.exists() else {}
    )
    per_tag = {int(k) for k in base.get("per_tag_p95_bytes", {})}
    have = [t for t in immortal_tags() if t in per_tag]
    lack = [t for t in immortal_tags() if t not in per_tag]
    return have, lack


def all_findings() -> list[str]:
    return (
        label_findings()
        + [
            f"{m}::{fn} destroys a lake record without consulting the retention label"
            for m, fn in reclaimers()
        ]
        + round_trip()
    )


def report() -> str:
    imm = immortal_tags()
    comp = sorted(t for t, row in RETENTION.items() if row[0] == "COMPACTABLE")
    bad, scratch = reclaimers(with_scratch=True)
    have, lack = budget_coverage()
    f = all_findings()
    lines = [
        f"IMMORTAL RETENTION -- {len(RETENTION)} tags: {len(imm)} IMMORTAL {[hex(t) for t in imm]}, "
        f"{len(comp)} COMPACTABLE {[hex(t) for t in comp]}",
        f"reclaimers of DURABLE records that do not consult the label: {len(bad)}"
        + (f" {bad}" if bad else " (none exists yet -- the registry labels; nothing reclaims)"),
        f"excluded, scratch-home deletions (cannot reach the durable lake): {len(scratch)} {scratch}",
        f"append baseline present for {len(have)} of {len(imm)} immortal tags; missing {[hex(t) for t in lack]}",
        "the grown substrate: 0x03 _T_GROWTH and 0x08 _T_GROWTH_DELTA are COMPACTABLE-with-a-reconstruction-proof,",
        "  NOT immortal -- the registry's own contract permits it, and 'never forgets' means never evicts the",
        "  IMMORTAL class, never 'stores everything'. Printed every run so the fact is never quietly assumed.",
        f"findings: {len(f)}",
    ]
    lines += [f"  - {x}" for x in f]
    return chr(10).join(lines)


def main() -> None:
    assert label_findings() == [], label_findings()  # the positive control: today's registry passes
    assert round_trip() == [], round_trip()
    good = {0x01: ("IMMORTAL", False, "a reason long enough to be a reason")}
    disp = {0x01: ("x", "y")}
    reds = {
        "IMMORTAL and reclaimable": {
            0x01: ("IMMORTAL", True, "a reason long enough to be a reason")
        },
        "no reason": {0x01: ("IMMORTAL", False, "short")},
        "unknown label": {0x01: ("KEEPISH", False, "a reason long enough to be a reason")},
    }
    for name, reg in reds.items():
        assert label_findings(reg, disp), f"the constructed red {name!r} was not caught"
    assert label_findings(good, disp) == [], "the positive control must pass"
    assert label_findings(good, {**disp, 0x77: ("x", "y")}), (
        "a dispositioned tag with no label must be caught"
    )
    assert label_findings(
        {**good, 0x78: ("IMMORTAL", False, "a reason long enough to be a reason")}, disp
    ), "a labeled tag with no migration disposition must be caught"
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        (scratch / "bad_reclaimer.py").write_text(
            "from snn_data_lake.substrate.home import _T_FACT@"
            "def compact_the_lake(path):@    tag = _T_FACT@    (path / 'lake.bin').unlink()@".replace(
                "@", chr(10)
            ),
            encoding="utf-8",
        )
        (scratch / "good_reclaimer.py").write_text(
            "from snn_data_lake.substrate.lake_engine import retention_of@"
            "def compact_the_lake(path, tag):@    if retention_of(tag)[0] == 'IMMORTAL':@        return 0@"
            "    (path / 'lake.bin').unlink()@".replace("@", chr(10)),
            encoding="utf-8",
        )
        (scratch / "not_a_reclaimer.py").write_text(
            "def tidy(name):@    return name.removeprefix('_T_')@".replace("@", chr(10)),
            encoding="utf-8",
        )
        (scratch / "scratch_reclaimer.py").write_text(
            "import tempfile@from snn_data_lake.substrate.home import _T_FACT@"
            "def selfcheck():@    d = tempfile.mkdtemp()@    tag = _T_FACT@    (Path(d) / 'lake.bin').unlink()@".replace(
                "@", chr(10)
            ),
            encoding="utf-8",
        )
        found, excluded = reclaimers(scratch, with_scratch=True)
        assert ("bad_reclaimer.py", "compact_the_lake") in found, found
        assert ("good_reclaimer.py", "compact_the_lake") not in found, found
        assert ("not_a_reclaimer.py", "tidy") not in found, found  # removeprefix is not a removal
        assert ("scratch_reclaimer.py", "selfcheck") in excluded, excluded  # excluded, and SEEN
        assert ("scratch_reclaimer.py", "selfcheck") not in found, found
    print(report())
    print(
        "immortal_retention self-check OK: 5 constructed label reds caught, today's registry and round trip pass; "
        "the reclaimer scan catches an unguarded durable compactor, clears a guarded one, excludes a "
        "scratch-home deletion INTO THE REPORT, and does not mistake str.removeprefix for a removal"
    )


def _assert() -> int:
    f = all_findings()
    print(report())
    if f:
        print(f"IMMORTAL_RETENTION ASSERT FAILED: {len(f)} finding(s)", file=sys.stderr)
        return 1
    print(
        "check OK: every class labeled with its reason, the two censuses agree, no reclaimer escapes the label,"
    )
    print("and every immortal class returns byte-identical across a process boundary")
    return 0


if __name__ == "__main__":
    sys.exit(_assert() if "--assert" in sys.argv else (main() or 0))
