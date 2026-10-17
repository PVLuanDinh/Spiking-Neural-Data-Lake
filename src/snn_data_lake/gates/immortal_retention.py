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
#: `rewrite` is here because THIS REPO RECLAIMS BY REWRITING. The first shipped form of this set omitted it
#: and printed a zero; the adversarial panel of 2026-09-06 measured that KnowledgeLog.consolidate destroys
#: records by rebuilding the payload list and calling BinLog.rewrite, using none of the other seven names --
#: so the zero was a claim about the instrument. The pre-registration had promised "delete, truncate OR
#: rewrite"; the arm was dropped in the build and the record then reported the result as measured.
_RECLAIM_CALLS = frozenset(
    {"unlink", "truncate", "rmtree", "remove", "compact", "reclaim", "evict", "purge", "rewrite"}
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

#: DURABLE reclaimers that do NOT consult the label, each with the row that owes the fix and what retires the
#: entry. This table is the opposite of an exclusion: an entry is a RED that is being carried in the open
#: rather than a site the scan pretends not to see, it is PRINTED as debt on every run, and it may only
#: SHRINK -- `findings()` fires on any reclaimer not listed here, and `stale_known()` fires on any listed
#: entry that no longer reclaims. It is not a licence: the sentence "the lake never evicts the IMMORTAL
#: class" is FALSE while this table is non-empty, and the report says so in those words.
KNOWN_UNLABELED: dict[tuple[str, str], str] = {}
#: PAID OUT 2026-09-06, the same day it was opened. It carried exactly one entry,
#: ("substrate/home.py", "consolidate"): the sleep tick ran LakeConsolidation ("use it or lose it", learned
#: capped at 250,000) and then rewrote the one lake to the survivor set, while _T_FACT (0x01) was labelled
#: IMMORTAL with reclaimable=False. Row L6 (T091) measured the live lake read-only -- 26,037 facts, ALL
#: learned, ZERO re-derivable -- found the cap had never bound, and closed the gap at both ends: the
#: registry's 0x01 reason now states its per-record exemption, LakeConsolidation refuses to cap a
#: non-reclaimable class, and KnowledgeLog.consolidate refuses a rewrite that would lose a learned fact.
#: `stale_known()` is what forced the payout: it red the moment the reclaimer stopped reclaiming.


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


#: How a function names the record classes it touches. A `_T_*` symbol in the body is the tag; the check
#: below requires each one to reach `retention_of`, so reading the label for ONE class can no longer exempt a
#: function that drops another. MEASURED 2026-09-06: before this, `home.consolidate` passed the census with a
#: single `retention_of(_T_FACT[0])` while `_prune_state_history` dropped tag 0x07 twelve lines below it.
def _tag_symbols(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id.startswith("_T_")}


def _tags_given_to_retention(node) -> set:
    """The `_T_*` symbols passed to a retention read inside `node` -- a label read is credited to the TAG it
    names, never to the function that contains it."""
    out: set = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in _RETENTION_READS:
            continue
        for arg in sub.args:
            out |= _tag_symbols(arg)
    return out


def _local_calls(node) -> set:
    """Bare names called inside `node` -- used for ONE hop into same-module helpers. One hop, not a full call
    graph, and that is a disclosed limit: `home.consolidate` destroys through `_prune_state_history`, so the
    tag reference and the destruction sit in different functions exactly one call apart. A two-hop laundering
    of the same shape would still pass, and there is no red for it here."""
    return {
        s.func.id
        for s in ast.walk(node)
        if isinstance(s, ast.Call) and isinstance(s.func, ast.Name)
    }


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

    Returns EVERY non-compliant DURABLE one, listed or not -- KNOWN_UNLABELED is applied by `findings()`, not
    here, so the census and the verdict can never quietly diverge. with_scratch=True also returns those
    excluded for deleting only inside a scratch home, so the exclusion is reported rather than silent."""
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
            # A label read is credited PER TAG, and one hop of same-module helpers is included, because a
            # function can destroy through a helper that names the tag it never mentions itself.
            helpers = {
                f
                for f in ast.walk(tree)
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                and f.name in _local_calls(node)
            }
            touched = _tag_symbols(node) | {t for h in helpers for t in _tag_symbols(h)}
            labeled = _tags_given_to_retention(node) | {
                t for h in helpers for t in _tags_given_to_retention(h)
            }
            # Compliant = it reads the label AND every STATIC tag it touches is named to a read. The
            # `reads_label` term is not redundant: a function whose tag is a runtime PARAMETER touches no
            # `_T_*` symbol, so the subset test alone would read its empty set as non-compliant and flag the
            # tree's own guarded fixture -- measured 2026-09-06. A dynamic tag cannot be attributed by a
            # static scan; such a function is credited for reading the label, and that is a DISCLOSED hole:
            # a reclaimer that computes its tag at runtime can drop an immortal class and pass here.
            if any(r in body for r in _RETENTION_READS) and touched <= labeled:
                continue
            where = (str(path.relative_to(base)).replace(chr(92), "/"), node.name)
            (scratch if any(m in body for m in _SCRATCH_MARKERS) else bad).append(where)
    return (bad, scratch) if with_scratch else bad


def round_trip(tags=None, withhold=None) -> list[str]:
    """(c) one record per IMMORTAL tag written BY A SEPARATE PROCESS through the lake's own writer, then read
    back here by a fresh reader over the persisted file.

    The first form of this arm wrote its own framing with plain `open()` in this same interpreter, so no lake
    code was under test and nothing could ever make it fail -- a criterion with no reachable red, which P8
    forbids (adversarial panel, 2026-09-06). It now uses the tree's own cross-process idiom (binlog.py's
    PROPERTY 3): the child appends through `BinLog`, this process re-opens the SAME file with a new `BinLog`,
    and byte identity is checked per tag.

    `withhold` is the red: the tag the child is told to skip. The record for it cannot come back, so the arm
    MUST report it -- that is what makes the empty return believable.
    """
    import subprocess

    from snn_data_lake.substrate import home as home_mod
    from snn_data_lake.substrate.binlog import BinLog
    from snn_data_lake.substrate.home import throwaway_home

    want = list(immortal_tags() if tags is None else tags)
    out: list[str] = []
    with throwaway_home():
        store = str(home_mod.knowledge_path("immortal_roundtrip.bin"))
        payloads = {t: bytes([t]) + (b"immortal" * 4) + bytes([t]) * (8 + t) for t in want}
        send = [t for t in want if t != withhold]
        blobs = [payloads[t] for t in send]
        child = (
            f"import sys; sys.path.insert(0, {str(SRC.parent)!r})\n"
            "from snn_data_lake.substrate.binlog import BinLog\n"
            f"log = BinLog({store!r})\n"
            f"for blob in {blobs!r}:\n"
            "    log.append(blob)\n"
        )
        rc = subprocess.run(  # noqa: S603 - our own interpreter, our own literal source
            [sys.executable, "-c", child], capture_output=True, text=True
        )
        if rc.returncode != 0:
            return [
                f"the writing child process failed (rc={rc.returncode}): {rc.stderr.strip()[:200]}"
            ]
        back = {
            p[:1][0]: p for p in BinLog(store).replay()
        }  # a FRESH reader over the persisted file
        for tag, blob in payloads.items():
            if back.get(tag) != blob:
                out.append(
                    f"tag {tag:#04x}: an immortal record did not survive the process boundary byte-identical"
                )
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


def unlisted_reclaimers() -> list[tuple[str, str]]:
    """Durable reclaimers that are NOT carried in KNOWN_UNLABELED -- the arm that asserts."""
    return [w for w in reclaimers() if w not in KNOWN_UNLABELED]


def stale_known() -> list[str]:
    """Entries in KNOWN_UNLABELED that no longer reclaim. The table may only SHRINK, so a stale row is a red:
    it means the debt was paid and the entry outlived it, or the scan lost sight of the site."""
    live = set(reclaimers())
    return [
        f"KNOWN_UNLABELED carries {m}::{fn}, which this scan no longer finds -- pay the entry out or explain"
        for (m, fn) in KNOWN_UNLABELED
        if (m, fn) not in live
    ]


# ---- the Rust half (R0.2 arc 2, DISPOSITION row 9): the same census over native/ -----------------
#: The constructed `native/sndl-core/src/lake.rs` the self-check and tests share (injected, never forked).
#: RED plants the four reclaim spellings of the pre-registration, each naming the lake and reading no label;
#: the controls in the same file are a guarded twin, a scratch-file removal, a non-truncating open, a
#: `#[cfg(test)]` purge and a plain writer. The GREEN twin gives each red fn a `retention_of` read.
_RUST_LAKE_RED = """\
use std::fs;
use std::fs::{File, OpenOptions};
use std::path::Path;
use crate::retention::retention_of;

pub fn compact_lake(tmp: &Path, p: &Path, recs: &[Vec<u8>], tag: u8) -> std::io::Result<()> {
    write_records(tmp, recs)?;
    let _label = ();
    fs::rename(tmp, p)
}
pub fn evict(lake: &Path, tag: u8) -> std::io::Result<()> {
    let _label = ();
    fs::remove_file(lake)
}
pub fn truncate_log(lake: &File, tag: u8) -> std::io::Result<()> {
    let _label = ();
    lake.set_len(0)
}
pub fn reopen_empty(lake: &Path, tag: u8) -> std::io::Result<File> {
    let _label = ();
    OpenOptions::new().write(true).truncate(true).open(lake)
}
pub fn compact_guarded(tmp: &Path, p: &Path, recs: &[Vec<u8>], tag: u8) -> std::io::Result<()> {
    if retention_of(tag).immortal {
        return Ok(());
    }
    write_records(tmp, recs)?;
    fs::rename(tmp, p)
}
pub fn tidy_scratch(dir: &Path) -> std::io::Result<()> {
    fs::remove_file(dir.join("scratch.tmp"))
}
pub fn open_append(lake: &Path) -> std::io::Result<File> {
    OpenOptions::new().append(true).truncate(false).open(lake)
}
#[cfg(test)]
fn purge_all(lake: &Path) -> std::io::Result<()> {
    fs::remove_file(lake)
}
fn write_records(p: &Path, recs: &[Vec<u8>]) -> std::io::Result<()> {
    Ok(())
}
"""
_RUST_LAKE_GREEN = _RUST_LAKE_RED.replace("let _label = ();", "let _label = retention_of(tag);")


def _rust_fixture(red: bool) -> dict[str, str]:
    """The fixture `native/` tree (paths relative to native/), in plan section 11's crate layout."""
    return {
        "sndl-core/src/lib.rs": "pub mod lake;\npub mod retention;\n",
        "sndl-core/src/lake.rs": _RUST_LAKE_RED if red else _RUST_LAKE_GREEN,
        "sndl-core/src/retention.rs": (
            "pub struct Label {\n    pub immortal: bool,\n}\n"
            "pub fn retention_of(tag: u8) -> Label {\n    Label { immortal: tag == 1 }\n}\n"
        ),
        "sndl-python/src/lib.rs": "#[pyfunction]\nfn add(a: f64, b: f64) -> f64 {\n    a + b\n}\n",
    }


def all_findings() -> list[str]:
    return (
        label_findings()
        + [
            f"{m}::{fn} destroys a lake record without consulting the retention label"
            for m, fn in unlisted_reclaimers()
        ]
        + stale_known()
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
        + (f" {bad}" if bad else " (none -- and the red for this arm is constructed in main())"),
        f"  of those, CARRIED as known debt with an owning row: {len(KNOWN_UNLABELED)}; unlisted (asserts): "
        f"{len(unlisted_reclaimers())}",
        f"excluded, scratch-home deletions (cannot reach the durable lake): {len(scratch)} {scratch}",
        f"append baseline present for {len(have)} of {len(imm)} immortal tags; missing {[hex(t) for t in lack]}",
        "the grown substrate: 0x03 _T_GROWTH and 0x08 _T_GROWTH_DELTA are COMPACTABLE-with-a-reconstruction-proof.",
        "  0x0D _T_BLOCKDELTA joined them on 2026-09-08 and was DELETED WITH THE PAGING LAYER on 2026-09-09 (T068",
        "  LOSS, specs/022 evidence/s5-paging-null.md 3.4/3.5), together with 0x0C _T_UNITBLOCK -- no lake ever held",
        "  a record under either, so this census loses no observed bytes. NEITHER of the remaining two is immortal",
        "  -- the registry's own contract permits it, and 'never forgets' means never evicts the IMMORTAL class,",
        "  never 'stores everything'. Nothing in the segment family reclaims anything today, so the label is a",
        "  licence nobody has exercised. Printed every run so the fact is never quietly assumed.",
    ]
    if KNOWN_UNLABELED:
        lines += [
            "THE SENTENCE 'THE LAKE NEVER EVICTS THE IMMORTAL CLASS' IS FALSE ON THIS TREE. Carried debt:",
        ]
        lines += [f"  ! {m}::{fn} -- {why}" for (m, fn), why in sorted(KNOWN_UNLABELED.items())]
    lines += [
        f"findings: {len(f)}",
    ]
    lines += [f"  - {x}" for x in f]
    return chr(10).join(lines)


def main() -> None:
    assert label_findings() == [], label_findings()  # the positive control: today's registry passes
    # (c) is believable only because its red is: a record the child never wrote cannot come back. Run the red
    # FIRST, so a round trip that has quietly stopped writing anything cannot pass as a green.
    withheld = immortal_tags()[0]
    assert round_trip(withhold=withheld), (
        "the withheld-record red did not fire -- (c) proves nothing"
    )
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
        # THE SHAPE THIS REPO ACTUALLY USES, and the one the first shipped scan could not see: reclamation by
        # rebuilding the survivor list and rewriting the log. Without this red, adding "rewrite" to
        # _RECLAIM_CALLS would be an unverified edit -- which is how the arm went missing in the first place.
        (scratch / "rewrite_reclaimer.py").write_text(
            "from snn_data_lake.substrate.home import _T_FACT@"
            "def consolidate(log, keep):@    survivors = [_T_FACT + r for r in keep]@"
            "    log.rewrite(survivors)@".replace("@", chr(10)),
            encoding="utf-8",
        )
        # THE SHAPE THAT SLIPPED THROUGH THE FIRST LANDING (adversarial panel, 2026-09-06): a label read
        # for ONE tag while a helper ONE HOP away drops ANOTHER. `home.consolidate` passed the census with a
        # single retention_of(_T_FACT) while `_prune_state_history` dropped tag 0x07 twelve lines below it.
        (scratch / "one_tag_only.py").write_text(
            "from snn_data_lake.substrate.home import _T_FACT, _T_STATE@"
            "from snn_data_lake.substrate.lake_engine import retention_of@"
            "def _prune(payloads):@    return [p for p in payloads if p[:1] != _T_STATE]@"
            "def consolidate(log, keep):@    if not retention_of(_T_FACT[0])[1]:@        pass@"
            "    log.rewrite(_prune(log.replay()) + [_T_FACT + k for k in keep])@".replace(
                "@", chr(10)
            ),
            encoding="utf-8",
        )
        # ...and its COMPLIANT TWIN, identical but for the second label read. Without this the red above
        # would also pass a rule that simply refused every multi-tag function.
        (scratch / "both_tags.py").write_text(
            "from snn_data_lake.substrate.home import _T_FACT, _T_STATE@"
            "from snn_data_lake.substrate.lake_engine import retention_of@"
            "def _prune(payloads):@    if not retention_of(_T_STATE[0])[1]:@        return list(payloads)@"
            "    return [p for p in payloads if p[:1] != _T_STATE]@"
            "def consolidate(log, keep):@    if not retention_of(_T_FACT[0])[1]:@        pass@"
            "    log.rewrite(_prune(log.replay()) + [_T_FACT + k for k in keep])@".replace(
                "@", chr(10)
            ),
            encoding="utf-8",
        )
        found, excluded = reclaimers(scratch, with_scratch=True)
        assert ("one_tag_only.py", "consolidate") in found, found
        assert ("both_tags.py", "consolidate") not in found, found
        assert ("rewrite_reclaimer.py", "consolidate") in found, found
        assert ("bad_reclaimer.py", "compact_the_lake") in found, found
        assert ("good_reclaimer.py", "compact_the_lake") not in found, found
        assert ("not_a_reclaimer.py", "tidy") not in found, found  # removeprefix is not a removal
        assert ("scratch_reclaimer.py", "selfcheck") in excluded, excluded  # excluded, and SEEN
        assert ("scratch_reclaimer.py", "selfcheck") not in found, found
    print(report())
    print(
        "immortal_retention self-check OK: 5 constructed label reds caught, today's registry passes; the "
        f"cross-process round trip reds on a withheld immortal record (tag {withheld:#04x}) and passes without "
        "it; the reclaimer scan catches BOTH an unlink-shaped and a rewrite-shaped durable compactor, clears a "
        "guarded one, excludes a scratch-home deletion INTO THE REPORT, and does not mistake str.removeprefix "
        "for a removal; a label read is credited PER TAG, so a function that labels one class while a "
        "one-hop helper drops another is caught while its compliant twin clears. Carried debt: "
        f"{len(KNOWN_UNLABELED)} known reclaimer(s), {len(unlisted_reclaimers())} unlisted"
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
