"""
Unit-label gate -- P4d's provenance check, red G-R7 (G7 / T115).

Pre-registration: specs/022-unified-brain-arc/evidence/g7-unit-label-gate.md (frozen 854de0e7, amended
da4bcfb7). Every grown unit carries a label -- position, layer, Dale sign, region, the problem that licensed
it -- and an unlabeled unit is REFUSED. One record per unit (growth/placement.place_labeled produces it);
this module only checks it, it defines no second vocabulary: REGION_COORDS, NEUROMERES, ZONES, GROWABLE,
LAYER_NAMES and DEPTH_BANDS are imported, never copied.

WHAT IT GIVES, each with a constructed red in tests/test_unit_label_gate.py:
  check_admission / require_admissible  -- the FIRST failing field in FIELD_ORDER; duplicate unit ids
      (within a batch, or already in the journal) refused before any other field; an AdmissionTicket naming
      exactly the admitted ids, consumed once -- G3's single row-append point takes it, so the gate PRECEDES
      admission structurally (Codex b3 #8).
  classify_restored -- origin from JOURNAL EVIDENCE: born (journal entry, identical label), derived (full
      label, no entry), legacy (no full label); a journaled id with a different label is REFUSED as a
      tampered born unit (D21). The persisted `origin` array is a cache, never evidence.
  check_saved -- KEPT / PRUNED (a committed verdict), PENDING (only uncommitted verdicts), REFUSED (a label
      or Dale defect, or a missing sleep field) -- G6's A12 commit protocol (D20).
  require_labeled_targets -- G5's return path projects only into admitted born units (D5).

THE MAP (one rule, 2.2): ap, dv in [0, 1) select the cell (NEUROMERES[floor(14 ap)], ZONES[floor(4 dv)]);
an empty cell is unknown territory (refused); a one-label cell must match the region (region_read "map");
a several-label cell admits a GROWABLE member by NAME (region_read "name") -- the O2 fallback; the owner's
strict sentence would refuse it (STRICT_TERRITORY). Only cortex carries a measured Dale sign, so only
cortex units pass today (O3).
"""

from __future__ import annotations

import ast
import functools
import math
from collections import namedtuple
from collections.abc import Mapping

from snn_data_lake.growth.arealization import NEUROMERES, REGION_COORDS, ZONES
from snn_data_lake.growth.growth_program import GROWABLE
from snn_data_lake.growth.placement import DEPTH_BANDS, LAYER_NAMES, label_of

#: The fixed check order; a verdict names the FIRST failing field (per-field attribution, R2).
FIELD_ORDER = ("unit_id", "batch_id", "position", "layer", "sign", "region", "problem_id", "origin")

#: The label fields a restored row must carry to count as labeled (origin is a cache, not a label).
LABEL_FIELDS = FIELD_ORDER[:-1]

#: O2 (owner, 2026-10-04): False = the name fallback in a several-label cell, as ruled.
STRICT_TERRITORY = False

#: O3 (owner, 2026-10-04): cortex only -- the one region with a measured Dale sign (H01).
ADMISSIBLE_REGIONS = frozenset({"cortex"})

Verdict = namedtuple("Verdict", "admitted field reason region_read")
SavedVerdict = namedtuple("SavedVerdict", "outcome field reason")


class UnitLabelRefused(ValueError):
    def __init__(self, unit_id, field, reason):
        super().__init__(f"unit {unit_id!r} refused: {field}: {reason}")
        self.unit_id, self.field, self.reason = unit_id, field, reason


def layer_from_depth(depth):
    """The layer whose DEPTH_BANDS interval [lo, hi) holds `depth` (the last band closed at 1.0)."""
    for layer in LAYER_NAMES:
        lo, hi = DEPTH_BANDS[layer]
        if lo <= depth < hi:
            return layer
    return LAYER_NAMES[-1] if depth == 1.0 else None


def cell_labels(ap, dv):
    """The REGION_COORDS names declared at the map cell (ap, dv) falls in; ap, dv must be in [0, 1)."""
    cell = (NEUROMERES[math.floor(len(NEUROMERES) * ap)], ZONES[math.floor(len(ZONES) * dv)])
    return sorted(r for r, c in REGION_COORDS.items() if c == cell)


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _text(v):
    return isinstance(v, str) and v != ""


def _ident(v):
    """unit_id / batch_id: a non-bool int >= 0 -- G3's journal ids; one representation (lead, 2026-10-06)."""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _check_unit_id(rec, ctx):
    return None if _ident(rec.get("unit_id")) else "missing or not a non-negative int unit_id"


def _check_batch_id(rec, ctx):
    return None if _ident(rec.get("batch_id")) else "missing or not a non-negative int batch_id"


def _check_position(rec, ctx):
    pos = rec.get("position")
    if pos is None:
        return "missing position"
    if not (isinstance(pos, (tuple, list)) and len(pos) == 5):
        return "position must be (ap, dv, x_um, y_um, depth)"
    if not all(_number(v) for v in pos):
        return "position has a non-finite or non-numeric coordinate"
    ap, dv, _x, _y, depth = pos
    if not (0.0 <= ap < 1.0 and 0.0 <= dv < 1.0):
        return "ap/dv outside [0, 1) (no clamping)"
    if not 0.0 <= depth <= 1.0:
        return "depth outside [0, 1]"
    return None


def _depth(rec):
    pos = rec.get("position")
    ok = isinstance(pos, (tuple, list)) and len(pos) == 5 and _number(pos[4])
    return pos[4] if ok else None


def _check_layer(rec, ctx):
    layer = rec.get("layer")
    if layer is None:
        return "missing layer"
    if layer not in LAYER_NAMES:
        return f"{layer!r} is not a cortical layer"
    depth = _depth(rec)
    if depth is not None and layer_from_depth(depth) != layer:
        return f"layer {layer!r} disagrees with depth {depth!r} ({layer_from_depth(depth)!r} band)"
    return None


def _check_sign(rec, ctx):
    sign = rec.get("sign")
    if "sign" not in rec:
        return "missing sign"
    if isinstance(sign, bool) or sign not in (1, -1):
        return f"sign must be +1 or -1, got {sign!r}"
    if sign == 1 and rec.get("layer") == "Layer 1":
        return "Layer 1 is never excitatory (assign_sign's structural rule)"
    return None


def _check_region(rec, ctx):
    region = rec.get("region")
    if region is None:
        return "missing region"
    if region not in REGION_COORDS:
        return f"unknown region name {region!r}"
    if region not in GROWABLE:
        return f"{region!r} is not growable (scaffold or charged-never-grown)"
    if region not in ADMISSIBLE_REGIONS:
        return (f"{region!r} refused: cortex only -- no other region has a measured Dale sign "
                "(owner, O3, 2026-10-04)")
    pos = rec.get("position")
    if not (isinstance(pos, (tuple, list)) and len(pos) == 5 and _number(pos[0]) and _number(pos[1])
            and 0.0 <= pos[0] < 1.0 and 0.0 <= pos[1] < 1.0):
        return "position unreadable on the map"
    labels = cell_labels(pos[0], pos[1])
    if not labels:
        return "unknown territory (no REGION_COORDS label at this cell)"
    if region not in labels:
        return f"position reads {labels}, not {region!r}"
    if len(labels) > 1 and STRICT_TERRITORY:
        return f"ambiguous territory {labels} (the strict sentence, O2)"
    ctx["region_read"] = "map" if len(labels) == 1 else "name"
    return None


def _check_problem_id(rec, ctx):
    pid = rec.get("problem_id")
    if not _text(pid):
        return "missing or empty problem_id"
    return None if ctx["resolve_problem"](pid) else f"problem {pid!r} does not resolve"


def _check_origin(rec, ctx):
    origin = rec.get("origin")
    return None if origin == "born" else f"origin {origin!r} is not admissible (only 'born')"


#: field -> check(rec, ctx) -> reason or None. A dict, so R10 can neuter one check at a time.
_CHECKS = {
    "unit_id": _check_unit_id,
    "batch_id": _check_batch_id,
    "position": _check_position,
    "layer": _check_layer,
    "sign": _check_sign,
    "region": _check_region,
    "problem_id": _check_problem_id,
    "origin": _check_origin,
}


def check_admission(unit, *, resolve_problem):
    """Verdict(admitted, field, reason, region_read) for one grown-unit record."""
    if not isinstance(unit, Mapping):
        return Verdict(False, "unit_id", "not a label record", None)
    ctx = {"resolve_problem": resolve_problem, "region_read": None}
    for field in FIELD_ORDER:
        reason = _CHECKS[field](unit, ctx)
        if reason is not None:
            return Verdict(False, field, reason, None)
    return Verdict(True, None, None, ctx["region_read"])


# distinct-from: growth/h01_ingest.py -- a binary site-field cursor; unrelated. Site read.
# distinct-from: growth/similarity_sized_growth.py -- a weights-moved predicate; unrelated. Site read.
# distinct-from: substrate/lake_segments.py -- a durable WAL record appender; this checks an append's
#   rows against validated records, it writes nothing. Site read.
class AdmissionTicket:
    """The VALIDATED admission records (frozen copies), consumed ONCE by the segment's row-append point
    (G3's admit_rows): the appended rows must be exactly those records -- no duplicate, no stranger, none
    missing, none changed since validation (Codex cp3 #1)."""

    def __init__(self, records):
        self._records = {r["unit_id"]: {k: r[k] for k in FIELD_ORDER} for r in records}
        self.ids = frozenset(self._records)
        self._consumed = False

    def consume(self, rows):
        rows = list(rows)
        if self._consumed:
            raise UnitLabelRefused(None, "ticket", "ticket already consumed")
        seen = set()
        for row in rows:
            uid = row.get("unit_id") if isinstance(row, Mapping) else None
            if uid in seen:
                raise UnitLabelRefused(uid, "ticket", "duplicate row under one ticket")
            seen.add(uid)
            frozen = self._records.get(uid)
            if frozen is None:
                raise UnitLabelRefused(uid, "ticket", "row not covered by the admission ticket")
            diff = [k for k in FIELD_ORDER if row.get(k) != frozen[k]]
            if diff:
                raise UnitLabelRefused(uid, "ticket", f"row differs from the validated record in {diff}")
        missing = sorted(self.ids - seen)
        if missing:
            raise UnitLabelRefused(missing[0], "ticket", f"validated records missing from the append: {missing}")
        self._consumed = True


def require_admissible(units, *, resolve_problem, known_unit):
    """AdmissionTicket for `units`, or UnitLabelRefused. Duplicates first (within the batch, then
    against the journal via `known_unit`), then every record's label (Codex b3 #9)."""
    units = list(units)
    seen = set()
    for u in units:
        uid = u.get("unit_id") if isinstance(u, Mapping) else None
        if _ident(uid):
            if uid in seen:
                raise UnitLabelRefused(uid, "unit_id", "duplicate within batch")
            seen.add(uid)
            if known_unit(uid):
                raise UnitLabelRefused(uid, "unit_id", "duplicate across batches")
    for u in units:
        v = check_admission(u, resolve_problem=resolve_problem)
        if not v.admitted:
            uid = u.get("unit_id") if isinstance(u, Mapping) else None
            raise UnitLabelRefused(uid, v.field, v.reason)
    return AdmissionTicket(units)


def classify_restored(rows, journal):
    """[origin] aligned with `rows`, from journal EVIDENCE (`journal`: unit_id -> birth record). A
    journaled id whose restored label differs is a tampered born unit: UnitLabelRefused (D21)."""
    out = []
    for row in rows:
        if not all(k in row for k in LABEL_FIELDS):
            out.append("legacy")
            continue
        uid = row["unit_id"]
        entry = journal.get(uid)
        if entry is None:
            out.append("derived")
            continue
        diff = [k for k in LABEL_FIELDS if row[k] != entry.get(k)]
        if diff:
            raise UnitLabelRefused(uid, "label", f"tampered born unit: fields differ {diff}")
        out.append("born")
    return out


def check_saved(record, *, resolve_problem, resolve_committed, first_sign):
    """SavedVerdict for a saved unit record (D20): the label first, Dale against the JOURNAL's birth
    sign, then the sleep clause on the EXACT committed (sleep_id, mask_digest) pair."""
    v = check_admission(record, resolve_problem=resolve_problem)
    if not v.admitted:
        return SavedVerdict("REFUSED", v.field, v.reason)
    uid = record["unit_id"]
    if uid not in first_sign:
        return SavedVerdict("REFUSED", "sign", "no birth sign in the journal")
    if record["sign"] != first_sign[uid]:
        return SavedVerdict("REFUSED", "sign", "Dale: sign differs from the birth sign")
    for field in ("sleep_id", "mask_digest", "verdict"):
        if not _text(record.get(field)):
            return SavedVerdict("REFUSED", field, f"missing or empty {field}")
    if record["verdict"] not in ("kept", "pruned"):
        return SavedVerdict("REFUSED", "verdict", f"verdict {record['verdict']!r} is not kept/pruned")
    if not resolve_committed(record["sleep_id"], record["mask_digest"]):
        return SavedVerdict("PENDING", None, "no committed sleep has kept or pruned this unit yet")
    return SavedVerdict(record["verdict"].upper(), None, None)


def require_labeled_targets(target_ids, records, *, resolve_problem):
    """Refuse any return-path target that is not an admitted born unit (D5; at projection build only)."""
    for tid in target_ids:
        rec = label_of(records, tid)
        if rec is None:
            raise UnitLabelRefused(tid, "unit_id", "unknown target unit")
        v = check_admission(rec, resolve_problem=resolve_problem)
        if not v.admitted:
            raise UnitLabelRefused(tid, v.field, v.reason)


# ---- coverage, anchored on ROW ADMISSION (Codex b3 #8): two scans, a census, two debt tables ----------
#
# (a) the label-creation scan: functions that create labels (place / place_grown, resolved by IMPORT
#     through growth_coverage_gate.live_growth_census -- the one resolver, P2) must call the gate;
# (b) the row-growth scan, with NO reference to placement: an assignment that regrows a name from itself
#     along axis 0 (`W = torch.cat([W, new], 0)`, `self.W = np.vstack([self.W, new])`) must pass a ticket
#     (`.consume(` / `admit_rows(`); the live grower's own rows grow in exactly this shape (ng_snn.py);
# (c) the census: per GROWN_SEGMENTS row, any derived/legacy row reds unless that segment has open debt.
# BLIND SPOTS, disclosed: row growth by slice-assignment past the end, np.resize, list-of-rows appends,
# or a name concatenated under a different name; place_labeled callers are not in (a) yet (no live caller
# exists; it lands with G3's admitting commit) -- (b) and the ticket cover them.

#: Grown-row segments: name -> (module, what grows, owning row, the debt keys that keep it open, the
#: declared ADMIT POINT -- the one "rel::qualname" whose ticket-first append is admissible; None until the
#: owner builds it). Row growth anywhere else is a finding unless debt-listed (Codex cp3 #5).
GROWN_SEGMENTS = {
    "neurogenesis_engine": (
        "snn_data_lake/growth/ng_snn.py",
        "NGSNN.W rows, grown by NeurogenesisSelfImprove._refit (the one live percept grower)",
        "G3/T111",
        ("snn_data_lake/growth/ng_snn.py::NGSNN.fit",
         "snn_data_lake/growth/neurogenesis_engine.py::NeurogenesisSelfImprove._refit"),
        None,  # G3's admit_rows lands with its admitting commit
    ),
}

#: Shrink-only debt: label-creating functions that do not call the gate -> (owner, what retires it).
KNOWN_UNGATED_CREATION_SITES = {
    "snn_data_lake/growth/neurogenesis_engine.py::NeurogenesisSelfImprove._refit": (
        "G3/T111", "the clock-driven refit admits units with no problem, region or map position; "
        "retired when the one grower admits through G3's problem-keyed loop and a ticket"),
    "snn_data_lake/growth/neurogenesis_engine.py::NeurogenesisSelfImprove.restore_growth_state": (
        "G3/T111", "re-derives placement on restore; retired when its rows are classified by "
        "classify_restored from G3's journal (G7's stacked neurogenesis_engine commit)"),
}

#: Shrink-only debt: functions that grow weight rows with no ticket -> (owner, what retires it).
KNOWN_UNGATED_ROW_GROWTH = {
    "snn_data_lake/growth/ng_snn.py::NGSNN.fit": (
        "G3/T111", "the live grower's rows (W, b); retired when its append takes an AdmissionTicket"),
    "snn_data_lake/growth/ng_snn.py::_novelty_control": (
        "G3/T111", "the self-check's novelty control grows rows the same way; retired with NGSNN.fit"),
}

#: THE SCOPE RULE (lead ruling 23, 2026-10-04): P4d's "grown unit" is a unit grown on the ENGINE'S
#: GROWTH PATH. Row growth elsewhere (readout layers, reasoning learners) is out of this gate's scope --
#: printed on every run and listed for the owner, never silently dropped and never counted as debt; a
#: row here that stops growing rows reds as stale like any debt row.
OUT_OF_SCOPE_ROW_GROWTH = {
    "snn_data_lake/readout/growable_readout.py::GrowableReadout._grow_one":
        "a growable readout's rows (readout layer, not the engine's growth path)",
    "snn_data_lake/readout/laminar_local_readout.py::LaminarLocalReadout.grow":
        "a laminar readout's hidden rows (readout layer, not the engine's growth path)",
    "snn_data_lake/reasoning/developmental_loop.py::GrowableSTDP.maybe_recruit":
        "a reasoning learner's recruited rows (not the engine's growth path)",
}

#: Reviewed FALSE POSITIVES of the row-growth scan (Codex cp3 #4 widened its forms; these hits grow no
#: rows of units). Printed every run; a row that stops matching reds as stale, like debt.
SCAN_FALSE_POSITIVES = {
    "snn_data_lake/faculties/learned_attention.py::LearnedAttention._goal_gain":
        "np.resize of a 1-D gain vector to the attention width -- an input reshape, no unit rows",
    "snn_data_lake/faculties/learned_attention.py::LearnedAttention.attend":
        "np.resize of a 1-D input vector to the attention width -- an input reshape, no unit rows",
    "snn_data_lake/reasoning/ng_snn/softhebb_mnist.py::HebbConv2d.forward":
        "F.pad of a convolution INPUT tensor -- spatial padding of activations, no unit rows",
}

_GATE_CALLS = frozenset({"require_admissible", "check_admission"})



def _repo_src():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


def _qualnames(tree):
    # REUSE (P2): gradient_claim_gate's one (qualname, node) walker -- module defs and class methods; a
    # nested function's lines attribute to the def that holds it, which is the right unit for a debt row.
    from snn_data_lake.gates.gradient_claim_gate import _qualnames as walker

    return walker(tree)


def _enclosing(funcs, lineno):
    best = None
    for name, node in funcs:
        if node.lineno <= lineno <= node.end_lineno and (best is None or node.lineno >= best[1].lineno):
            best = (name, node)
    return best


def _calls(node, names):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            n = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if n in names:
                return True
    return False


@functools.lru_cache(maxsize=8)  # one parse per root per process: report + assert re-read the same tree
def _creation_functions(root):
    """{"rel::qualname": gated?} for every function holding a resolved label-creation call."""
    from snn_data_lake.gates.growth_coverage_gate import live_growth_census

    by_file = {}
    for sites in live_growth_census(src_root=root).values():
        for site in sites:
            rel, line = site.rsplit(":", 1)
            by_file.setdefault(rel, set()).add(int(line))
    out = {}
    for rel, lines in by_file.items():
        funcs = _qualnames(ast.parse((root / rel).read_text(encoding="utf-8")))
        for line in lines:
            hit = _enclosing(funcs, line)
            key = f"{rel}::{hit[0] if hit else '<module>'}"
            out[key] = bool(hit) and _calls(hit[1], _GATE_CALLS)
    return out


# distinct-from: gates/eprop_transport_gate.py -- its _transposed names a TRANSPOSED operand; this
#   reads a concatenation/resize/pad AXIS. Different AST question. Site read.
# distinct-from: gates/move_gate.py -- its _assign returns a module-level assignment's value; unrelated.
# distinct-from: faculties/access_consciousness.py, growth/neurogenesis_engine.py, infra/device.py --
#   a constructor argument, a persistence predicate, a CUDA crossover; the matcher fired on "None". Read.
def _axis(call, positional):
    """The call's axis/dim: its constant, None when absent, "?" when not a constant."""
    for kw in call.keywords:
        if kw.arg in ("axis", "dim"):
            return kw.value.value if isinstance(kw.value, ast.Constant) else "?"
    if len(call.args) > positional:
        a = call.args[positional]
        return a.value if isinstance(a, ast.Constant) else "?"
    return None


def _same(node, name):
    return isinstance(node, (ast.Name, ast.Attribute)) and ast.unparse(node) == name


def _pads_axis0(call):
    """False only when the pad width provably leaves axis 0 alone ((0, 0) first); else True."""
    width = call.args[1] if len(call.args) > 1 else next(
        (kw.value for kw in call.keywords if kw.arg == "pad_width"), None)
    if isinstance(width, (ast.Tuple, ast.List)) and width.elts and isinstance(width.elts[0], (ast.Tuple, ast.List)):
        return not all(isinstance(e, ast.Constant) and e.value == 0 for e in width.elts[0].elts)
    return True


def _self_growth(node):
    """True if `node` grows a name's LEADING dimension from itself -- the registered forms (2.3(b), Codex
    cp3 #4): vstack / row_stack / concatenate / cat of a sequence holding the name on axis 0;
    np.append(name, x, axis=0); np.resize(name, ...); np.pad(name, ...) touching axis 0; and slice
    extension past the end, name[len(name):] / name[name.shape[0]:] = ..."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return False
    target, value = node.targets[0], node.value
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Slice) and target.slice.lower:
        me, lower = ast.unparse(target.value), ast.unparse(target.slice.lower)
        return f"len({me})" in lower or f"{me}.shape" in lower
    if not (isinstance(target, (ast.Name, ast.Attribute)) and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute) and value.args):
        return False
    me, fn, first = ast.unparse(target), value.func.attr, value.args[0]
    if fn in ("vstack", "row_stack", "concatenate", "cat"):
        holds = isinstance(first, (ast.List, ast.Tuple)) and any(_same(e, me) for e in first.elts)
        return holds and (fn in ("vstack", "row_stack") or _axis(value, 1) in (None, 0))
    if fn == "append":
        return _same(first, me) and _axis(value, 2) == 0  # without an axis np.append flattens
    if fn == "resize":
        return _same(first, me)
    if fn == "pad":
        return _same(first, me) and _pads_axis0(value)
    return False


@functools.lru_cache(maxsize=8)
# distinct-from: engine/brain.py -- the growth-gate verdict callable; no source scan. Site read.
# distinct-from: faculties/self_improvement_suite.py -- an eps-invariance rung; no scan. Site read.
# distinct-from: faculties/self_model.py -- a bucketing helper; no scan. Site read.
# distinct-from: faculties/time_cells.py -- a would-earn estimate; no scan. Site read.
# distinct-from: gates/doc_budget_gate.py -- prose word ceilings; the matcher fired on "growth". Read.
def _row_growth_functions(root):
    """{"rel::qualname": ticketed?} for every function that regrows rows (module harness fns skipped)."""
    from snn_data_lake.gates.growth_coverage_gate import _HARNESS_FNS

    out, first_growth, nodes = {}, {}, {}
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root).as_posix()
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        funcs = None
        for node in ast.walk(tree):
            if not _self_growth(node):
                continue
            funcs = funcs if funcs is not None else _qualnames(tree)
            hit = _enclosing(funcs, node.lineno)
            if hit and hit[0] in _HARNESS_FNS:
                continue
            key = f"{rel}::{hit[0] if hit else '<module>'}"
            first_growth[key] = min(first_growth.get(key, node.lineno), node.lineno)
            nodes[key] = hit[1] if hit else None
    for key, fn in nodes.items():
        out[key] = fn is not None and _consumes_first(fn, first_growth[key])
    return out


# distinct-from: reasoning/nengo_gui_brain.py -- its _module_scope_is_inert checks THAT file's
#   module-level statements for import side effects; this checks a function body's statement order
#   for a ticket spent before an append. Site read.
def _consumes_first(fn, growth_line):
    """True iff `fn`'s body holds, as a TOP-LEVEL statement before `growth_line`, a call
    `<parameter>.consume(...)` -- a ticket passed IN and spent before any row grows (Codex cp3 #5: a call
    nested under a branch, on an unrelated object, or after the append does not count)."""
    a = fn.args
    params = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    for stmt in fn.body:
        if stmt.lineno >= growth_line:
            break
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "consume" and isinstance(call.func.value, ast.Name)
                and call.func.value.id in params):
            return True
    return False


def _root(src_root):
    from pathlib import Path

    return Path(src_root) if src_root is not None else _repo_src()


def label_creation_findings(src_root=None, known=None):
    known = KNOWN_UNGATED_CREATION_SITES if known is None else known
    return sorted(k for k, gated in _creation_functions(_root(src_root)).items()
                  if not gated and k not in known)


# distinct-from: gates/conclusion_protocol.py -- its row_findings checks instrument REGISTRY rows; this
#   reports unticketed weight-row growth. Site read.
# distinct-from: engine/brain.py, faculties/self_improvement_suite.py, faculties/self_model.py,
#   faculties/time_cells.py -- see _row_growth_functions above; all sites read.
# distinct-from: growth/neurogenesis_engine.py -- a crc32 growth fingerprint; no scan. Site read.
# distinct-from: growth/arealization.py -- a GRN fixed-point settle; unrelated. Site read.
# distinct-from: growth/growth_stop.py -- a port declaration; unrelated. Site read.
# distinct-from: growth/h01_ingest.py -- a join-count refusal; unrelated. Site read.
# distinct-from: growth/ng_snn.py -- prune_unearned drops units; it is a SCANNED site, not a scanner. Read.
def row_growth_findings(src_root=None, known=None, out_of_scope=None, segments=None):
    """Row growth that is neither ADMITTED -- the declared admit point of a GROWN_SEGMENTS row, spending
    its ticket first (_consumes_first) -- nor debt-listed, out of scope, or a reviewed false positive."""
    known = KNOWN_UNGATED_ROW_GROWTH if known is None else known
    oos = {**OUT_OF_SCOPE_ROW_GROWTH, **SCAN_FALSE_POSITIVES} if out_of_scope is None else out_of_scope
    segments = GROWN_SEGMENTS if segments is None else segments
    admit_points = {seg[4] for seg in segments.values() if seg[4]}
    return sorted(k for k, ticketed in _row_growth_functions(_root(src_root)).items()
                  if not (ticketed and k in admit_points) and k not in known and k not in oos)


# distinct-from: gates/immortal_retention.py -- its stale_known is bound to KNOWN_UNLABELED and its
#   reclaimer scan (no arguments, nothing to import); same shrink-only PATTERN, different table. Read.
# distinct-from: growth/ng_snn.py, faculties/ring_attractor.py, growth/function_preserving_growth.py,
#   growth/growing_learner.py -- a prune, a mismatch estimate, a neuron clone, a classifier; the
#   matcher fired on "no longer" / "grows". All sites read.
def stale_known(src_root=None, known_creation=None, known_rows=None, out_of_scope=None):
    """Debt and scope rows naming a function that no longer creates labels / grows rows (a table may
    only shrink by retiring a row, never by quietly outliving its site)."""
    root = _root(src_root)
    kc = KNOWN_UNGATED_CREATION_SITES if known_creation is None else known_creation
    kr = KNOWN_UNGATED_ROW_GROWTH if known_rows is None else known_rows
    oos = {**OUT_OF_SCOPE_ROW_GROWTH, **SCAN_FALSE_POSITIVES} if out_of_scope is None else out_of_scope
    stale = [k for k in kc if k not in _creation_functions(root)]
    stale += [k for k in list(kr) + list(oos) if k not in _row_growth_functions(root)]
    return sorted(stale)


def open_debt_segments():
    debt = set(KNOWN_UNGATED_CREATION_SITES) | set(KNOWN_UNGATED_ROW_GROWTH)
    return {name for name, seg in GROWN_SEGMENTS.items() if debt & set(seg[3])}


def census(segments, journal, open_debt=None):
    """{segment: {born, derived, legacy, duplicate, rows, red}}; red = a derived/legacy row in a segment
    with no open debt (the partition always sums to rows, so the sum is never the red), or ANY repeated
    unit_id -- open debt never excuses a duplicate (Codex cp3 #1)."""
    open_debt = open_debt_segments() if open_debt is None else open_debt
    out = {}
    for name, rows in segments.items():
        origins = classify_restored(rows, journal)
        counts = {o: origins.count(o) for o in ("born", "derived", "legacy")}
        ids = [r["unit_id"] for r in rows if isinstance(r, Mapping) and _ident(r.get("unit_id"))]
        duplicate = len(ids) - len(set(ids))
        unticketed = counts["derived"] + counts["legacy"]
        red = bool(duplicate) or (bool(unticketed) and name not in open_debt)
        out[name] = {**counts, "duplicate": duplicate, "rows": len(rows), "red": red}
    return out


def engine_rows(engine):
    """The live percept grower's rows as census rows: today's (depth, layer, sign) tuples carry no
    unit_id, so every row is legacy; rows beyond the placement (a drifted list) are bare and legacy."""
    model = getattr(engine, "_model", None)
    n = int(model.W.shape[0]) if model is not None and model.W is not None else 0
    placed = list(engine.placement)
    rows = [{"place_depth": d, "place_layer": lay, "place_sign": s} for d, lay, s in placed[:n]]
    return rows + [{} for _ in range(n - len(rows))]


def report():
    lines = ["UNIT-LABEL GATE (G-R7, G7/T115) -- coverage anchored on row admission"]
    creation, rows, stale = label_creation_findings(), row_growth_findings(), stale_known()
    for title, table in (("label-creation debt", KNOWN_UNGATED_CREATION_SITES),
                         ("row-growth debt", KNOWN_UNGATED_ROW_GROWTH)):
        lines.append(f"  {title}: {len(table)} row(s)")
        lines += [f"    {k}  [{owner}] {why}" for k, (owner, why) in sorted(table.items())]
    lines.append(f"  out of scope (ruling 23: P4d's grown unit = the engine's growth path; listed for the "
                 f"owner): {len(OUT_OF_SCOPE_ROW_GROWTH)} row(s)")
    lines += [f"    {k}  {why}" for k, why in sorted(OUT_OF_SCOPE_ROW_GROWTH.items())]
    lines.append(f"  reviewed scan false positives (no unit rows grown): {len(SCAN_FALSE_POSITIVES)} row(s)")
    lines += [f"    {k}  {why}" for k, why in sorted(SCAN_FALSE_POSITIVES.items())]
    lines.append(f"  findings: creation {creation}  rows {rows}  stale {stale}")
    if KNOWN_UNGATED_CREATION_SITES or KNOWN_UNGATED_ROW_GROWTH:
        lines.append('  the sentence "no unlabeled grown unit is admitted" is FALSE while either debt '
                     "table is non-empty; T115 stays OPEN")
    return "\n".join(lines)


# distinct-from: gates/selfcheck_cache.py, gates/move_gate.py, gates/gradient_claim_gate.py,
#   instruments/attention_schema.py, instruments/everyday_battery.py, faculties/agency_monitor.py --
#   each is THAT module's own constructed reds (a cache's key invariance, a move diff, a baseline
#   refusal, a schema's scorers, a battery's questions, an agency monitor's six reds); this fires the
#   label gate's. Per-module self-checks by convention, no shared mechanism.
# distinct-from: faculties/competence_executive.py -- a PYTHONPATH root finder; unrelated. All sites read.
def _self_check():
    """Positive control R6 first, then ten constructed reds that must each fire; a FIXTURE label
    density (G4's Part B records the real one); no Brain is built."""
    import tempfile
    from pathlib import Path

    from snn_data_lake.growth.placement import place_labeled

    def resolve(pid):
        return pid == "prob-1"

    recs = place_labeled(9, unit_ids=list(range(9)), region="cortex",
                         site=(0.5 / len(NEUROMERES), 1.5 / len(ZONES)), problem_id="prob-1",
                         batch_id=0, seed=13015, centre_um=(0.0, 0.0), density_per_mm2=2.7e4,
                         roles=("E",) * 8 + ("I",))
    good = recs[0]
    assert check_admission(good, resolve_problem=resolve).admitted, "R6: a full label must be admitted"

    def refused(rec):
        return not check_admission(rec, resolve_problem=resolve).admitted

    def raises(fn):
        try:
            fn()
        except UnitLabelRefused:
            return True
        return False

    pons = (NEUROMERES.index("r3") + 0.5) / len(NEUROMERES), (ZONES.index("basal") + 0.5) / len(ZONES)
    saved = dict(good, sleep_id="s1", mask_digest="m1", verdict="kept")
    first = {good["unit_id"]: good["sign"]}
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "fx"
        pkg.mkdir()
        (pkg / "m.py").write_text("import numpy as np\nclass G:\n    def grow(self, n):\n"
                                  "        self.W = np.vstack([self.W, n])\n", encoding="utf-8")
        r14 = row_growth_findings(src_root=tmp, known={}) == ["fx/m.py::G.grow"]
    reds = {
        "R1 no label": refused({}),
        "R3 insula": refused(dict(good, region="insula")),
        "R4 cortex in pons": refused(dict(good, position=pons + tuple(good["position"][2:]))),
        "R5 neuromodulation": refused(dict(good, region="neuromodulation")),
        "R7 sign 0": refused(dict(good, sign=0)),
        "R16 duplicate": raises(lambda: require_admissible(
            [good, dict(recs[1], unit_id=good["unit_id"])], resolve_problem=resolve,
            known_unit=lambda u: False)),
        "R18 tampered": raises(lambda: classify_restored(
            [dict(good, sign=-good["sign"])], {good["unit_id"]: good})),
        "R9 no sleep_id": check_saved(
            {k: v for k, v in saved.items() if k != "sleep_id"}, resolve_problem=resolve,
            resolve_committed=lambda s, m: True, first_sign=first).outcome == "REFUSED",
        "R9b uncommitted": check_saved(
            saved, resolve_problem=resolve, resolve_committed=lambda s, m: False,
            first_sign=first).outcome == "PENDING",
        "R14 row growth": r14,
    }
    silent = [name for name, fired in reds.items() if not fired]
    print(f"constructed reds fired: {len(reds) - len(silent)} of {len(reds)}")
    assert not silent, f"reds that did not fire: {silent}"


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if "--assert" not in argv:
        _self_check()
    print(report())
    bad = label_creation_findings() + row_growth_findings() + stale_known()
    if "--assert" in argv and bad:
        print(f"UNIT_LABEL_GATE ASSERT FAILED: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
