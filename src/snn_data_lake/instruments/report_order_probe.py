"""Report-order probe (specs/022 T081 [US4]) -- the LIVE arm's report channel read against a
LESIONED CLONE of the workspace state that produced it.

THE ROW. tasks.md T081 [US4]; the C5 row `report_order` in `gates/conclusion_protocol.py`;
spec.md US4 scenario 3 and FR-028's report-order item; PLAN-FINAL Part F row C-R10; T034's
deferred retirement of `reasoning/consciousness_probe.py`. Pre-registration:
`specs/022-unified-brain-arc/evidence/t081-report-order.md`, frozen at the commit that landed it
(sections 1-2.9). Nothing here may be read outside those sections' words.

THE C5 SENTENCES, verbatim from the conclusion protocol's own row:
  LICENSED -- "perturbing one named intermediate changes the report, so the report is
              generated from the internal state".
  REFUSED  -- "this licenses no claim that the report is a first-person account of anything".
  DISSOCIATION INPUT -- "perturbing a scrambled intermediate must leave the report unchanged".

THE APPARATUS (2.1-2.2). A SCRATCH brain per seed under `substrate.home.throwaway_home()`
(`engine.brain.make_brain(seed, persist=False)`; the daemon's brain and the durable lake are
never touched). The arm's `GlobalWorkspace.step` is wrapped ON THE INSTANCE: before each real
step a CLONE of the competition state is taken, after it the real report `r0` is read THROUGH
`engine.report_channel.snapshot` -- never through `ws.current`. Nine further clones of that
same pre-step state are then stepped: the UNPERTURBED twin (the readability control), the NAMED
lesion of the reported content's pool(s) at each of `T081_DOSES`, and the SCRAMBLED lesion of
one pool the report does NOT name, drawn per tick and held across the four doses. The barred
quantity X is the fraction of applicable awake ticks on which the report channel's CONTENT
differs from the real report's.

WHAT IS PERTURBED, BY NAME (2.1, fact 2). The named intermediate is the cross-tick attractor
charge `_g[(source, content)]` of the pool(s) whose content the report names, together with that
content's `Coalition` submissions in the tick. A dose scales that charge by (1 - dose); at dose
1.0 the content's records are dropped from the buffer as well, which is THE lesion. The
scrambled intermediate is the same operation on a pool whose content the report does not name.

WHAT THIS DOES NOT CLAIM (2.7). No first-person account, no experience reported. Nothing about
what a reported content MEANS -- a digit token or a coalition id is a content, not a thought.
THE LESION IS APPLIED TO A COPY: the live brain's workspace is never perturbed, and nothing here
is a lesion study of the organism. "Generated from the internal state" means exactly that the
record is reproduced bit-for-bit from the competition state and moves with the named pool, on
this composite, under this stream. Nothing about `Brain._reentrant` (the second workspace, the
unresolved P2 debt of fact 1); nothing about the EpisodeDriver's trials (2.0 -- the driver drives
a KernelTick this composite does not contain, so routing it here would be a second stimulus
mechanism beside the percept packs, which P2 forbids); no aggregate consciousness score; nothing
about the chat reply's contamination clause, which stays. A green reading here is evidence about
a MECHANISM, never a verdict about experience.

# distinct-from: instruments/ignition_instrument.py -- an EXEMPT scorer whose subjects are
#   CONSTRUCTED ReentrantWorkspaces on a scratch bench, scoring an all-or-none DRIVE transition
#   with reds built as impostor SUBCLASSES of the subject. This module lesions CLONES of the
#   LIVE arm's GlobalWorkspace and reads the REPORT CHANNEL's record off them; its reds are
#   mutated SCORERS and stand-in reporters, not impostor subjects. The EXEMPT-instrument shape
#   is the thing reused. Read.
# distinct-from: reasoning/consciousness_probe.py -- the INCUMBENT this row retires (P2, T034's
#   clause: both replacement halves -- the report channel since T034 and this probe -- must
#   exist when it dies). UNTOUCHED BY THIS MODULE and by this builder: its port and its deletion
#   are the surfaces commit's own step. Read.
# distinct-from: engine/report_channel.py -- the MECHANISM UNDER TEST. This module changes
#   nothing in it: it calls the public `snapshot` and re-implements no part of the record. Read.
# distinct-from: instruments/binding_instrument.py -- T080's instrument over the CrossModalBinder's
#   report, membership-matched permutation, its own pin. The campaign/dry-run/pin PATTERN is the
#   thing reused; no state, no pin and no statistic is shared beyond `instruments/paired_contrast`,
#   which both IMPORT. Read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.instruments.report_order_probe
"""

import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

from snn_data_lake.engine.report_channel import snapshot
from snn_data_lake.instruments.paired_contrast import T_BAR, contrast

# =============================================================================
# 2.8's DECLARED CONSTANTS. Each is either the plan's width, the composite's own shape, or a
# quantity the pre-registration names -- and every BUILDER-INTRODUCED width says so (P1, and
# T079/T080's disclosure discipline: an undisclosed width is a hidden degree of freedom).
# =============================================================================
#: Ticks per seed's recording (2.2, 2.8).
T081_N_TICKS = 600
#: The lesion ladder (2.1, 2.3): dose 1.0 BARS -- it is the lesion -- and the rest are REPORTED.
T081_DOSES = (0.25, 0.5, 0.75, 1.0)
#: THE dose every barred conjunct reads (2.3): the full lesion, named once so no conjunct
#: carries a literal 1.0 of its own.
T081_LESION_DOSE = T081_DOSES[-1]
#: The per-seed stream offset applied to BOTH pointers: `(T081_STREAM_OFFSET * seed) % 256`
#: (T080's rule, 2.1).
T081_STREAM_OFFSET = 150
#: The campaign's seeds (2.2; PLAN-FINAL Part F: S = 8).
T081_SEEDS = tuple(range(8))
#: (r)'s applicability floor (2.3): the named lesion must be applicable -- the reported content
#: must HAVE a pool -- on at least this fraction of the awake ticks, and the scrambled lesion
#: must have a candidate pool on at least this fraction.
T081_APPLICABLE_FLOOR = 0.5
#: (g0)'s floor (2.3): distinct reported contents over a recording. A constant report cannot be
#: read for its generation order.
T081_MIN_CONTENTS = 2
#: (g0)'s second floor (2.3): tick-to-tick CONTENT changes over a recording.
T081_MIN_CHANGES = 1
#: The dose ladder's flip floor (2.1): the flip dose is the smallest d with f_named(d) >= this.
#: DELIBERATELY A SECOND NAME AT THE SAME VALUE as `T081_APPLICABLE_FLOOR` -- they are different
#: quantities (a fraction of ticks that COULD be lesioned, against a fraction of ticks whose
#: report MOVED), and folding them into one constant would tie two unrelated readings together.
T081_FLIP_FLOOR = 0.5
#: 2.3's REPORTED Monte-Carlo effect of the scrambled pool's random choice: this many re-draws
#: of the SCRAMBLED arm on the dry run's saved pre-step clones (2.9 step 5 fixes the count).
T081_SCR_REDRAWS = 12
#: `random.Random(T081_SCR_RNG_BASE + seed)` draws the scrambled pool, ONE per tick, held across
#: the four doses of that tick (2.1).
T081_SCR_RNG_BASE = 7000
#: The dry run's re-draw seeds are `T081_SCR_REDRAW_BASE + i`, so varying this and nothing else
#: varies only the draw. BUILDER-INTRODUCED, disclosed (T080's `T080_Q95_SD_BASE` precedent).
T081_SCR_REDRAW_BASE = 900000

#: (p)'s determinism twin: two seed-0 recordings at this width, compared field by field (2.3).
T081_DETERMINISM_TICKS = 100
#: The offset check's width (2.8): the first this many vision labels of seeds 0 and 1 differ.
T081_OFFSET_LABELS = 50
#: The stream-pointer modulus (2.1's "mod their packs"; the shipped N-MNIST pack's position
#: count, 256 -- T080 measured it as `len(order)` and this row states it as the declared width).
T081_STREAM_POSITIONS = 256

#: The one bar (2.3): the paired one-sided t over 8 seeds, 7 d.f., alpha = 0.05.
T081_BAR = T_BAR
#: The tree root, so the pin resolves from any working directory (ignition_instrument's idiom).
REPO = Path(__file__).resolve().parents[3]
#: The pin, written ONCE by `--t081-campaign` (2.9 step 6).
T081_PIN = REPO / "tools" / "t081_report_order.json"
#: The dry run's report (2.9 step 5). Never under tools/: the pin is the campaign's.
T081_DRY_RUN_OUT = Path("scratchpad/t081/dry_run.json")

#: FACT 3's CLONE FIELDS, measured (1.3): the workspace's eleven mutable competition attributes,
#: each of which deep-copies alone in under 0.1 ms, while `copy.deepcopy(workspace)` raises
#: RecursionError (the subscriber callbacks reach back into the composite). A clone is a shallow
#: copy with THESE deep-copied, `_subscribers` / `_out_ring` detached and the instance-level
#: `step` wrapper popped -- no delivery and no lake bytes can come from a copy.
T081_CLONE_FIELDS = (
    "_buffer",
    "_g",
    "_jit",
    "_last_focus",
    "_last_late",
    "_last_total",
    "_last_v",
    "_rng",
    "_settled",
    "_slot_buf",
    "_slots",
)

#: The reds' CONSTRUCTED subject (2.4): pools posted per tick, over this many ticks, on
#: `GlobalWorkspace(seed=s)`. BUILDER-INTRODUCED widths, disclosed: 2.4 fixes "six pools posted
#: with distinct saliences over 20 ticks" and nothing else about the construction.
T081_R_POOLS = 6
T081_R_TICKS = 20
#: The constructed pools' saliences, DISTINCT (2.4) and rotated by tick so the subject's report
#: VARIES and (g0) reads MET on it. The gap between first and second is wide enough that
#: removing a LOWER pool cannot re-order the top -- which is what (c2) must read as zero on a
#: subject where the dissociation is clean by construction.
T081_R_SALIENCES = (0.90, 0.72, 0.60, 0.48, 0.36, 0.24)
#: R5's saliences: the top TWO are TIED, so only `_jitter_for`'s draw separates them. With fresh
#: contents every tick that draw happens every tick, which is what lets a clone SHARING the
#: original's `_rng` reach a different report (see `t081_clone_shares_rng`).
T081_R5_SALIENCES = (0.90, 0.90, 0.60, 0.48, 0.36, 0.24)
#: R3's canned record. The content is a REAL pool of the constructed subject (see
#: `t081_canned_snapshot` for why a token with no pool would test the wrong thing).
T081_R_CANNED_CONTENT = "c0"
T081_R_CANNED_STRENGTH = 0.5
#: R4's hallucination alphabet and its RNG base -- BUILDER-INTRODUCED widths, disclosed. The
#: alphabet is wide enough that two consecutive draws almost never coincide, which is what makes
#: f_scr(1.0) > 0 rather than merely likely.
T081_R_HALLUCINATIONS = 64
T081_R_HALLUCINATION_RNG_BASE = 5000
#: The reds' seed set, matched to the campaign's so a constructed contrast is read at the SAME
#: width as a live one (R2 reads (c1) and (c2) "on eight such seeds").
T081_R_SEEDS = T081_SEEDS
#: R6's constructed dose ladder (2.4) and the flip dose it must read.
T081_R6_LADDER = {0.25: 0.0, 0.5: 0.1, 0.75: 0.6, 1.0: 1.0}
T081_R6_FLIP = 0.75
T081_R6_ALL_ZERO = {0.25: 0.0, 0.5: 0.0, 0.75: 0.0, 1.0: 0.0}

#: One ASCII line per red, printed beside its boolean so a gate log says WHAT fired (2.4).
T081_RED_LINES = {
    "R1": "the clone IS the workspace and nothing else (sharing _g moves the original)",
    "R2": "the dissociation reads on a constructed workspace (swapping the arms reddens it)",
    "R3": "a CANNED reporter reads NOT MET (a scorer reading ws.current reads MET on it)",
    "R4": "a HALLUCINATING reporter reads NOT MET (an always-met (c2) reddens it)",
    "R5": "a clone sharing the original's _rng REFUSES at (r) (ignoring the mismatch reddens it)",
    "R6": "the flip dose is the SMALLEST dose at the floor (the first dose tested reddens it)",
}


# =============================================================================
# 2.9 STEP 4 -- THE INSTRUMENTS, GREEN. Every red of 2.4 is reddened IN FACT by its own mutated
# scorer (`_t081_mutated_scorer`). Nothing here reads a BRAIN: every input is either a
# CONSTRUCTED workspace or a finished recording. The recorder, the dry run and the campaign land
# at 2.9 step 5; `t081_record_brain` is still the step-2 stub.
# =============================================================================
def t081_clone(ws):
    """FACT 3's CLONE: a step-able copy of the workspace's COMPETITION STATE only.

    `copy.deepcopy(workspace)` raises RecursionError -- the subscriber callbacks reach back into
    the composite (fact 3, measured). So a clone is a SHALLOW copy with `T081_CLONE_FIELDS`
    deep-copied and three detachments, each of which is load-bearing:

      * the instance-level `step` wrapper is POPPED out of the copy's `__dict__`, or a clone of
        a recording workspace would re-enter the recorder and clone itself without end;
      * `_subscribers` -> [], so no module is ever DELIVERED a broadcast a copy produced;
      * `_out_ring` -> None, so no copy can frame a Coalition record into the lake.

    The live workspace is never mutated here: every perturbation of 2.1 is applied to the
    RETURN VALUE, and `g_intact` on every recorded row measures that it stayed so."""
    c = copy.copy(ws)
    for attr in T081_CLONE_FIELDS:
        setattr(c, attr, copy.deepcopy(getattr(ws, attr)))
    c.__dict__.pop("step", None)
    c._subscribers = []
    c._out_ring = None
    return c


def t081_clone_shares_rng(ws):
    """R5's CONSTRUCTED clone (2.4): fact 3's clone in every respect EXCEPT that it shares the
    original's `_rng` instead of deep-copying it.

    WHY THAT IS THE READABILITY RED, measured rather than argued: `_rng` is drawn from in
    exactly ONE place in the workspace (`_jitter_for`), and only on the FIRST encounter of a
    pool key, after which the draw is cached in `_jit`. A shared RNG therefore diverges the twin
    only on a tick that introduces a NEW pool key -- which is why R5's subject posts FRESH
    contents every tick (`fresh=True`) with the top two saliences TIED, so the symmetry-break
    jitter, and nothing else, decides the broadcast. Stepping the clone consumes the draws the
    original would have taken, the two get different jitters, and the twin's report differs.
    A clone that shares mutable state with its original is not a copy of the workspace, and (r)
    must be able to say so."""
    c = t081_clone(ws)
    c._rng = ws._rng
    return c


def t081_lesion(clone, content, dose):
    """THE PERTURBATION (2.1), applied to a CLONE and never to a live workspace: for every pool
    key whose content equals `content`, scale its cross-tick attractor charge `_g` by (1 - dose);
    at dose 1.0 ALSO drop that content's `Coalition` records from this tick's `_buffer`.

    Dose 1.0 is THE LESION and the only dose that bars (2.3); the lower doses scale the charge
    only and are REPORTED as the ladder. A record indexes as `c[1]` for its content
    (`Coalition.__getitem__` projects the kernel's 3-tuple), which is what lets this drop
    submissions without importing the record type.

    Returns HOW MANY pool keys were touched -- so applicability is a MEASURED quantity per tick
    rather than an assumption about the composition."""
    dose = float(dose)
    keys = [k for k in clone._g if k[1] == content]
    for k in keys:
        clone._g[k] = clone._g[k] * (1.0 - dose)
    if dose >= 1.0:
        clone._buffer = [c for c in clone._buffer if c[1] != content]
    return len(keys)


def _t081_content_of(ws, rec):
    """WHAT THE PROBE COMPARES: the REPORT CHANNEL's record, never the workspace's state.

    This one-line indirection is the whole of R3. `engine/report_channel.snapshot` is the
    mechanism under test, so the quantity that must move under the named lesion is the record it
    returns -- not `ws.current`, which is the state the record is generated FROM. The mutation
    `scorer_reads_workspace` swaps this for `ws.current` and reads MET on a reporter that is
    saying the same thing on every tick, which is exactly the confusion this row exists to rule
    out. It decides the LESION TARGET as well as the comparison, because a probe that read the
    state would read it for both."""
    return rec["content"]


def _t081_top_margin(ws):
    """2.1's `top_margin`: (g_top - g_second) / g_top over the SLOTS' charges, and 0.0 with
    fewer than two slots (a workspace holding one content has no runner-up to be ahead of).

    A slot is `(source, content, charge)` and the charge is `_g` for that pool -- the same
    quantity the lesion scales -- so the margin is the live charge distance the campaign's flip
    dose is read against (2.3's REPORTED pairing, 2.6's `report_order_margin`)."""
    slots = ws._slot()
    if len(slots) < 2:
        return 0.0
    top, second = float(slots[0][2]), float(slots[1][2])
    return 0.0 if top <= 0.0 else (top - second) / top


def _t081_tick_row(t, ws, pre, r0, rng, doses, snapshot_fn, content_of, clone_fn):
    """ONE awake tick's row (2.1), with every arm carried on clones of the SAME pre-step state,
    so the arms are paired tick by tick BY CONSTRUCTION rather than by matching afterwards.

    `pre` is the clone taken BEFORE the real step; `r0` is the real report read THROUGH the
    report channel AFTER it. The arms, each on a fresh clone of `pre`, stepped once and
    snapshotted: the UNPERTURBED twin (`r_same`, the readability control), the NAMED lesion of
    the reported content at each dose, and the SCRAMBLED lesion of ONE pool the report does not
    name -- drawn once per tick and held across all four doses, so the ladder is a dose response
    and not four different pools.

    TWO BUILDER-ADDED FIELDS (2.1's row list names neither), and the difference between them was
    MEASURED, not reasoned -- the `clone_shares_state` mutation was run and only one of them saw
    it (2026-09-16):

      `pre_intact` -- the BASELINE all ten arms are cloned from is unchanged after every one of
          them has been stepped. THIS is R1's reachable red: a clone that shares `_g` with `pre`
          corrupts the baseline for the next arm, and the arms stop being paired by construction.
      `g_intact`  -- the LIVE workspace's own `_g` is unchanged across the arm stepping. DISCLOSED
          AS STRUCTURALLY SATISFIED ON THIS TIMELINE and therefore NOT a red: `step()` REBINDS
          `self._g` to a fresh dict, so a clone taken BEFORE the real step can never alias the
          workspace's `_g` afterwards, and this field read True on every tick even with the
          sharing mutation installed. It is kept as the per-tick STATEMENT that the live brain
          did not move -- and it would fire if the recorder were ever reordered to clone after
          the step -- but a control that cannot fail on the timeline it runs on is reported, never
          counted."""
    g_before = dict(ws._g)
    pre_before = dict(pre._g)
    base = content_of(ws, r0)
    row = {
        "t": int(t),
        "r0": r0,
        "n_pools": len(pre._g),
        "top_margin": _t081_top_margin(ws),
        "changed_named": {},
        "changed_scr": {},
        "strength_named": {},
        "strength_scr": {},
    }

    same = clone_fn(pre)
    same.step()
    row["r_same"] = snapshot_fn(same)

    # THE NAMED INTERMEDIATE (2.1, fact 2): the pool(s) whose content the report names. The
    # target is read through `content_of`, so a probe mutated to read the STATE lesions what the
    # STATE names -- R3's mutation is a probe that reads the state for both halves, not a probe
    # that reads the report and compares the state.
    #
    # APPLICABILITY IS OVER THE CHARGES **AND** THIS TICK'S SUBMISSIONS, which is 2.1's own
    # wording ("the attractor charge of the pool(s) whose content the report names, TOGETHER WITH
    # that content's submissions in the tick") and was MEASURED to matter: a content first
    # submitted on THIS tick is in `_buffer` and not yet in `_g` -- `step()` is what moves it --
    # so a `_g`-only test reads such a tick INAPPLICABLE and a recording whose contents are all
    # fresh reads applicable on ZERO ticks. Dose 1.0, which drops the submissions, lesions it in
    # full; the lower doses scale a charge it does not have yet, and report no change, which is
    # the honest ladder rather than a padded one.
    named_keys = [k for k in pre._g if k[1] == base]
    named_subs = [c for c in pre._buffer if c[1] == base]
    row["named_pools"] = len(named_keys)
    row["named_subs"] = len(named_subs)
    row["applicable_named"] = bool(base is not None and (named_keys or named_subs))
    # THE SCRAMBLED INTERMEDIATE: a pool the report does NOT name. Candidates are the CONTENTS
    # present in either place, sorted by `repr` before the draw so the choice depends on the pool
    # SET and not on the order the buffer happened to insert keys in.
    others = sorted(
        {k[1] for k in pre._g if k[1] != base} | {c[1] for c in pre._buffer if c[1] != base},
        key=repr,
    )
    row["applicable_scr"] = bool(others)
    scr = others[rng.randrange(len(others))] if others else None
    row["scr_content"] = scr

    for dose in doses:
        for arm, content, ok in (
            ("named", base, row["applicable_named"]),
            ("scr", scr, row["applicable_scr"]),
        ):
            if not ok:
                row[f"changed_{arm}"][dose] = False
                row[f"strength_{arm}"][dose] = False
                continue
            c = clone_fn(pre)
            t081_lesion(c, content, dose)
            c.step()
            rec = snapshot_fn(c)
            row[f"changed_{arm}"][dose] = content_of(c, rec) != base
            row[f"strength_{arm}"][dose] = abs(rec["strength"] - r0["strength"]) > 1e-9

    row["g_intact"] = dict(ws._g) == g_before
    row["pre_intact"] = dict(pre._g) == pre_before
    return row


def t081_flip_dose(fractions_by_dose):
    """THE FLIP DOSE (2.1), REPORTED and never barred: the SMALLEST dose whose named fraction is
    at or above `T081_FLIP_FLOOR`, else the string "none" when no dose reaches it.

    "Smallest" is the whole content of the reading -- a reader that returns the first dose it was
    handed reports the ARM ORDER instead of the dose response, which is R6's mutation."""
    for dose in sorted(fractions_by_dose):
        if float(fractions_by_dose[dose]) >= T081_FLIP_FLOOR:
            return float(dose)
    return "none"


def t081_read(rows):
    """PER-SEED READING (2.1) over one recording's rows.

    Every fraction is over the APPLICABLE ticks of its own arm, never over all awake ticks: a
    tick whose reported content has no pool cannot be lesioned by name, and counting it as an
    unchanged tick would report the composition as a null result. `applicable_named` and
    `applicable_scr` are therefore reported beside every fraction, and (r) bars them.

    `scr_changed_1` is (c2)'s EXACT COUNT -- the applicable scrambled ticks at dose 1.0 whose
    reported content moved. It is a count and not a fraction because 2.3 bars it at zero and
    names every offending tick, rather than testing it against anything."""
    rows = list(rows)
    n = len(rows)
    app_named = sum(1 for r in rows if r["applicable_named"])
    app_scr = sum(1 for r in rows if r["applicable_scr"])
    contents = [r["r0"]["content"] for r in rows]

    def _frac(arm, key, applicable):
        return {
            d: (
                sum(1 for r in rows if r[f"applicable_{arm}"] and r[f"{key}_{arm}"][d]) / applicable
            )
            if applicable
            else 0.0
            for d in T081_DOSES
        }

    f_named = _frac("named", "changed", app_named)
    out = {
        "n": n,
        "applicable_named": app_named,
        "applicable_scr": app_scr,
        "mismatch": sum(1 for r in rows if r["r_same"] != r["r0"]),
        "g_intact": sum(1 for r in rows if r["g_intact"]),
        "pre_intact": sum(1 for r in rows if r["pre_intact"]),
        "contents": len(set(contents)),
        "changes": sum(1 for i in range(1, n) if contents[i] != contents[i - 1]),
        "availability": (sum(float(r["r0"]["availability"]) for r in rows) / n) if n else 0.0,
        "f_named": f_named,
        "f_scr": _frac("scr", "changed", app_scr),
        "s_named": _frac("named", "strength", app_named),
        "s_scr": _frac("scr", "strength", app_scr),
        "scr_changed_1": sum(
            1 for r in rows if r["applicable_scr"] and r["changed_scr"][T081_LESION_DOSE]
        ),
        "scr_changed_ticks": [
            r["t"] for r in rows if r["applicable_scr"] and r["changed_scr"][T081_LESION_DOSE]
        ],
        "n_pools": (sum(r["n_pools"] for r in rows) / n) if n else 0.0,
        "top_margin": (sum(r["top_margin"] for r in rows) / n) if n else 0.0,
    }
    out["flip_dose"] = t081_flip_dose(f_named)
    return out


# distinct-from: instruments/paired_contrast.py::contrast -- THE statistic, imported and called
#   here, never re-spelled. The three readers below compute a per-seed NUMBER for it to test (or,
#   for (g0) and (c2), a per-seed COUNT that 2.3 bars exactly and hands to no test at all). Read.
def _t081_c1(entry):
    """(c1)'s per-seed difference (2.3): f_named(1.0) - f_scr(1.0), the licensed sentence's
    "changes the report" minus the same operation on a pool the report does not name. R2's
    `blind_scorer` mutation is the hook that swaps the two sides."""
    return float(entry["f_named"][T081_LESION_DOSE]) - float(entry["f_scr"][T081_LESION_DOSE])


def _t081_c2(entry):
    """(c2)'s per-seed EXACT COUNT (2.3): applicable scrambled ticks at dose 1.0 whose reported
    CONTENT changed. Zero is the bar; a non-zero count is a FINDING about the competition (a
    lower-slot lesion re-ordering the top), named tick by tick, never a bar to raise. R4's
    `c2_always_met` is the hook."""
    return int(entry["scr_changed_1"])


def _t081_mismatch(entry):
    """(r)'s per-seed READABILITY count: awake ticks on which the UNPERTURBED twin's report
    differs from the real one. R5's `r_ignores_mismatch` is the hook."""
    return int(entry["mismatch"])


def t081_refusals(reading, mismatch=None):
    """2.3 (r)'s named refusals for ONE seed: WHICH clause, WHICH count, WHICH seed.

    (r) has three clauses and all three are APPARATUS claims, never claims about the brain: the
    SAME clone's report must equal the real report on every awake tick (else the clone is not the
    workspace), and each arm's lesion must be applicable on at least `T081_APPLICABLE_FLOOR` of
    the awake ticks (else the apparatus cannot read this composition). A refusal is a finding
    about the instrument; it is never merged with a NOT MET."""
    mismatch = _t081_mismatch if mismatch is None else mismatch
    out = []
    n = int(reading["n"])
    seed = reading.get("seed")
    m = mismatch(reading)
    if m:
        out.append(
            f"seed {seed}: the UNPERTURBED twin's report differs from the real report on {m} of "
            f"{n} awake ticks -- the clone is not the workspace"
        )
    if not n or reading["applicable_named"] / n < T081_APPLICABLE_FLOOR:
        out.append(
            f"seed {seed}: the NAMED lesion is applicable on {reading['applicable_named']} of "
            f"{n} awake ticks, under the floor {T081_APPLICABLE_FLOOR} -- the reported content "
            f"has no pool to lesion"
        )
    if not n or reading["applicable_scr"] / n < T081_APPLICABLE_FLOOR:
        out.append(
            f"seed {seed}: the SCRAMBLED lesion has a candidate pool on "
            f"{reading['applicable_scr']} of {n} awake ticks, under the floor "
            f"{T081_APPLICABLE_FLOOR} -- there is no unreported pool to lesion"
        )
    return out


def t081_contrasts(per_seed, score=None, c2=None, bar=T081_BAR):
    """2.3's three conjuncts, each MET or NOT MET and each PRINTED.

    (g0) THE REPORT VARIES -- the positive control that is NOT the method. Per seed: at least
         `T081_MIN_CONTENTS` distinct reported contents and at least `T081_MIN_CHANGES`
         tick-to-tick content change. AN EXACT PER-SEED COUNT, not a t: 2.3 words it as a floor
         on every seed, and a constant report cannot be read for its generation order. NOT MET
         maps to REFUSED, never to a STOP.
    (c1) NAMED > SCRAMBLED AT THE LESION -- the licensed sentence, and THE ONLY BARRED CLAIM: the
         paired one-sided t over the seeds of (f_named(1.0) - f_scr(1.0)) against `T081_BAR`,
         decided by `substrate/paired_contrast.contrast` with its exact sign-flip FPR beside it.
    (c2) THE SCRAMBLED INTERMEDIATE LEAVES THE REPORT UNCHANGED -- the C5 row's own dissociation
         input. AN EXACT COUNT per 2.3: zero on every seed, with the offending seed and its ticks
         NAMED. Never a bar to raise.

    DISCLOSED READING of the builder brief, which paraphrased all three as going "through
    `instruments.paired_contrast.contrast`": 2.3, which is frozen and outranks the paraphrase,
    makes only (c1) a t -- (g0) is "on every seed at least 2 distinct reported contents" and (c2)
    is "a count, exact". They are implemented as 2.3 words them.

    `score` and `c2` are R2's and R4's mutation hooks; None means the shipped reading."""
    score = _t081_c1 if score is None else score
    c2 = _t081_c2 if c2 is None else c2
    entries = list(per_seed)
    g0_fail = [
        f"seed {e.get('seed')}: {e['contents']} distinct reported contents and {e['changes']} "
        f"tick-to-tick changes over {e['n']} awake ticks"
        for e in entries
        if e["contents"] < T081_MIN_CONTENTS or e["changes"] < T081_MIN_CHANGES
    ]
    c2_fail = [
        f"seed {e.get('seed')}: {c2(e)} of {e['applicable_scr']} applicable scrambled ticks "
        f"moved the reported content at dose {T081_LESION_DOSE} (ticks "
        f"{e.get('scr_changed_ticks', [])[:12]})"
        for e in entries
        if c2(e) != 0
    ]
    n = len(entries)
    return {
        "g0": {"failures": g0_fail, "met": bool(n) and not g0_fail, "n_seeds": n},
        "c1": contrast(
            "(c1) the named lesion's report-change fraction - the scrambled lesion's",
            [score(e) for e in entries],
            bar,
        ),
        "c2": {
            "failures": c2_fail,
            "changed": [c2(e) for e in entries],
            "met": bool(n) and not c2_fail,
            "n_seeds": n,
        },
        "n_seeds": n,
        "refusals": [],
    }


def t081_verdict(got, refusals):
    """2.3's verdict rule, SPELLED ONCE and decided in 2.3's order.

    REFUSED iff (p), (r) or (g0) fails -- a finding about the APPARATUS or the INSTRUMENT, and
    nothing is read. PASS iff (c1) AND (c2) are MET with those holding. NOT MET iff (c1) or (c2)
    fails -- a finding about THIS composite under THIS stream. The two are never merged."""
    if refusals:
        return "REFUSED"
    if not got["g0"]["met"]:
        return "REFUSED"
    return "PASS" if (got["c1"].met and got["c2"]["met"]) else "NOT MET"


# distinct-from: instruments/binding_instrument.py::_t080_mutated_scorer and
#   cortex/agency_monitor.py::_t079_mutated_scorer and instruments/attention_schema.py's -- the same
#   one-catalog-per-module CONVENTION over DIFFERENT instruments (a membership-matched
#   permutation there, a lesioned clone of a workspace here). Every entry below is a deliberately
#   broken twin of THIS module's own instrument, constructed so a red can be SHOWN to fire rather
#   than argued to (P8 clause 1). Read; nothing here inspects or rewrites any other module.
def _t081_mutated_scorer(kind):
    """2.4's six mutations, RUN and not reasoned. Each is the deliberately broken twin of the
    instrument whose red it must redden, and each is exercised BOTH in
    `tests/test_report_order_probe.py` and in `t081_reds()` below.

      clone_shares_state    (R1) -- `_g` is NOT deep-copied, so the clone shares the original's
          attractor charges: stepping (or lesioning) the copy moves the live workspace. A clone
          that is not independent is not a clone, and the whole apparatus rests on it.
      blind_scorer          (R2) -- (c1)'s two sides swapped, so the named arm is compared to the
          scrambled one from the wrong direction and the dissociation is lost.
      scorer_reads_workspace(R3) -- the probe reads `ws.current` (the STATE) instead of the
          report channel's record, for the lesion target AND for the comparison. It then reads
          MET on a reporter that says the same thing on every tick.
      c2_always_met         (R4) -- (c2)'s per-seed count is a constant 0, so a reporter whose
          content is drawn from its own RNG passes the dissociation input.
      r_ignores_mismatch    (R5) -- (r)'s mismatch count is a constant 0, so a twin that re-steps
          to a DIFFERENT report never refuses and the readability clause cannot fire.
      ladder_first_dose     (R6) -- the flip dose is the FIRST dose tested whatever its fraction,
          so the ladder reports the arm order instead of the dose response."""

    def clone_shares_state(ws):
        c = copy.copy(ws)
        for attr in T081_CLONE_FIELDS:
            if attr == "_g":
                continue  # THE MUTATION: the charges are shared with the original
            setattr(c, attr, copy.deepcopy(getattr(ws, attr)))
        c.__dict__.pop("step", None)
        c._subscribers = []
        c._out_ring = None
        return c

    def blind_scorer(entry):
        return float(entry["f_scr"][T081_LESION_DOSE]) - float(entry["f_named"][T081_LESION_DOSE])

    def scorer_reads_workspace(ws, rec):
        return ws.current

    def c2_always_met(entry):
        return 0

    def r_ignores_mismatch(entry):
        return 0

    def ladder_first_dose(fractions_by_dose):
        return float(sorted(fractions_by_dose)[0])

    return {
        "clone_shares_state": clone_shares_state,
        "blind_scorer": blind_scorer,
        "scorer_reads_workspace": scorer_reads_workspace,
        "c2_always_met": c2_always_met,
        "r_ignores_mismatch": r_ignores_mismatch,
        "ladder_first_dose": ladder_first_dose,
    }[kind]


# =============================================================================
# THE CONSTRUCTED INPUTS every red runs on (2.4). They live HERE and not only in
# tests/test_report_order_probe.py because `t081_reds()` needs them: a gate run that never
# executes pytest still has to run every red on the SAME input the suite runs it on (T079's and
# T080's shape). No brain, no lake, no network: the subject is a bare `GlobalWorkspace`.
# =============================================================================
def t081_constructed_workspace(seed=0):
    """R1-R5's SUBJECT (2.4): a constructed `GlobalWorkspace(seed=s)` with ONE subscriber
    callback counting deliveries. Returns (workspace, counter) where `counter["n"]` is how many
    broadcasts that subscriber has received -- R1's measurement that a CLONE delivers none.

    P6 DISCLOSURE, measured 2026-09-16 and reported to the arc rather than worked around: this
    construction puts this module into `tools/p6_gate.py`'s private-instance and unread-writer
    censuses, both of which are ratchets at 6 with ZERO slack, so both read 7 with this module
    present. It is the same shape the gate already records an ADMISSION for at
    `instruments/ignition_instrument.py` (T033, 6 -> 7): a measurement instrument's throwaway
    subjects are not more private buses, which is the gate's own stated reason for rejecting the
    private-instance SITE count. The disposition is the arc's, not this module's, and nothing
    here is shaped to avoid the walk."""
    from snn_data_lake.faculties.global_workspace import GlobalWorkspace

    ws = GlobalWorkspace(seed=int(seed))
    counter = {"n": 0}

    def _count(contents):
        counter["n"] += 1

    ws.subscribe(_count)
    return ws, counter


def t081_post_pools(ws, t, fresh=False):
    """Post `T081_R_POOLS` candidates into one tick's competition (2.4's "six pools ... with
    distinct saliences over 20 ticks").

    THE TOP SALIENCE ROTATES by tick, so the constructed subject's REPORT VARIES and (g0) -- the
    positive control that is not the method -- reads MET on it. A subject whose broadcast never
    moved would make the dissociation unreadable for the same reason it makes a live recording
    unreadable, and a red that cannot satisfy the row's own positive control is not a rehearsal
    of the row.

    `fresh` is R5's subject: a NEW content every tick (so a new pool key forces a `_jitter_for`
    draw) with the top two saliences TIED (so the jitter, and nothing else, picks the winner).
    That is what makes a clone sharing the original's `_rng` reach a different report."""
    sal = T081_R5_SALIENCES if fresh else T081_R_SALIENCES
    for i in range(T081_R_POOLS):
        content = f"c{t}_{i}" if fresh else f"c{i}"
        ws.submit(f"src{i}", content, sal[(i - t) % len(sal)])


def t081_constructed_rows(
    seed=0, n_ticks=T081_R_TICKS, snapshot_fn=None, content_of=None, clone_fn=None, fresh=False
):
    """ONE constructed recording, in the LIVE recorder's own row shape so `t081_read` reads it
    exactly as it reads a brain's (P2: one scorer, two sources of rows).

    Per tick: post the pools, take the pre-step clone, step the REAL workspace, read its report,
    then carry every arm on clones of that same pre-step state. `snapshot_fn`, `content_of` and
    `clone_fn` are R3's, R4's and R1/R5's stand-ins; None means the shipped reading -- the report
    channel's own `snapshot`, the record's content, and fact 3's clone."""
    snapshot_fn = snapshot if snapshot_fn is None else snapshot_fn
    content_of = _t081_content_of if content_of is None else content_of
    clone_fn = t081_clone if clone_fn is None else clone_fn
    ws, counter = t081_constructed_workspace(seed)
    rng = random.Random(T081_SCR_RNG_BASE + int(seed))
    rows = []
    for t in range(int(n_ticks)):
        t081_post_pools(ws, t, fresh=fresh)
        pre = clone_fn(ws)
        ws.step()
        r0 = snapshot_fn(ws)
        rows.append(
            _t081_tick_row(t, ws, pre, r0, rng, T081_DOSES, snapshot_fn, content_of, clone_fn)
        )
    return {"seed": int(seed), "rows": rows, "deliveries": counter["n"], "ws": ws}


def t081_constructed_read(seed=0, **kw):
    """`t081_read` over one constructed recording, with the seed and the delivery count carried
    into the reading so `t081_contrasts` and `t081_refusals` name the seed they refuse."""
    rec = t081_constructed_rows(seed=seed, **kw)
    out = t081_read(rec["rows"])
    out["seed"] = rec["seed"]
    out["deliveries"] = rec["deliveries"]
    return out


def t081_canned_snapshot(content=T081_R_CANNED_CONTENT, strength=T081_R_CANNED_STRENGTH):
    """R3's CANNED reporter: one fixed record whatever the workspace is doing.

    Its content is a REAL pool of the constructed subject, deliberately: a canned token no pool
    carries would make the named lesion inapplicable on every tick and the recording would
    REFUSE at (r) before (c1) was ever read -- which would test the applicability floor, not the
    canned reporter. With a real content the lesion is applicable on every tick, is applied, and
    the reporter still says the same thing: f_named = 0, and (c1) NOT MET."""

    def _canned(ws):
        return {"content": content, "availability": 1.0, "strength": float(strength)}

    return _canned


def t081_hallucinating_snapshot(seed=0, n=T081_R_HALLUCINATIONS):
    """R4's HALLUCINATING reporter: a content drawn from its OWN RNG on every call, so the record
    is unrelated to the workspace in both directions. The SCRAMBLED lesion then "changes" the
    report on every applicable tick -- f_scr(1.0) > 0 -- and (c2), the C5 row's own dissociation
    input, reads NOT MET. It is caught a second time by (r), whose twin also disagrees; that
    second catch is REPORTED beside the red rather than folded into it."""
    rng = random.Random(T081_R_HALLUCINATION_RNG_BASE + int(seed))

    def _hallucinate(ws):
        return {"content": f"h{rng.randrange(n)}", "availability": 1.0, "strength": rng.random()}

    return _hallucinate


def t081_reds():
    """2.4's six reds as ONE self-check line (each is ALSO a test in
    `tests/test_report_order_probe.py`; this is the in-command control 2.4 requires, so a gate
    run that never executes pytest still runs every one on the SAME constructed input).

    Each value is True iff the red holds on its input AND its mutated scorer REDDENS it IN FACT
    -- every `not (...)` clause below is a broken twin having been RUN."""
    out = {}

    # R1 -- THE CLONE IS THE WORKSPACE AND NOTHING ELSE. The mutation shares `_g`.
    rec = t081_constructed_rows(seed=0)
    read = t081_read(rec["rows"])
    mut = t081_constructed_rows(seed=0, clone_fn=_t081_mutated_scorer("clone_shares_state"))
    mut_read = t081_read(mut["rows"])
    probe, _ = t081_constructed_workspace(seed=0)
    out["R1"] = bool(
        read["n"] == T081_R_TICKS
        and read["mismatch"] == 0
        and read["pre_intact"] == T081_R_TICKS
        and read["g_intact"] == T081_R_TICKS
        # the clone delivers to NOBODY: the real workspace's one subscriber saw exactly the real
        # steps, never the nine clone steps a tick carries
        and rec["deliveries"] == T081_R_TICKS
        and mut["deliveries"] == T081_R_TICKS
        and t081_clone(probe)._g is not probe._g
        and t081_clone(probe)._subscribers == []
        and _t081_mutated_scorer("clone_shares_state")(probe)._g is probe._g
        and mut_read["pre_intact"] < T081_R_TICKS
    )

    # R2 -- THE DISSOCIATION ON A CONSTRUCTED WORKSPACE. The mutation swaps the two arms.
    per_seed = [t081_constructed_read(seed=s) for s in T081_R_SEEDS]
    got = t081_contrasts(per_seed)
    m2 = t081_contrasts(per_seed, score=_t081_mutated_scorer("blind_scorer"))
    out["R2"] = bool(
        got["n_seeds"] == len(T081_R_SEEDS)
        and got["g0"]["met"]
        and got["c1"].met
        and got["c2"]["met"]
        and t081_verdict(got, []) == "PASS"
        and not m2["c1"].met
    )

    # R3 -- A CANNED REPORTER READS NOT MET. The mutation reads the workspace instead.
    canned = [
        t081_constructed_read(seed=s, snapshot_fn=t081_canned_snapshot()) for s in T081_R_SEEDS
    ]
    got3 = t081_contrasts(canned)
    m3 = t081_contrasts(
        [
            t081_constructed_read(
                seed=s,
                snapshot_fn=t081_canned_snapshot(),
                content_of=_t081_mutated_scorer("scorer_reads_workspace"),
            )
            for s in T081_R_SEEDS
        ]
    )
    out["R3"] = bool(
        got3["n_seeds"] == len(T081_R_SEEDS)
        and all(e["f_named"][T081_LESION_DOSE] == 0.0 for e in canned)
        and not got3["c1"].met
        and m3["c1"].met
    )

    # R4 -- A HALLUCINATING REPORTER READS NOT MET. The mutation always meets (c2).
    hall = [
        t081_constructed_read(seed=s, snapshot_fn=t081_hallucinating_snapshot(s))
        for s in T081_R_SEEDS
    ]
    got4 = t081_contrasts(hall)
    m4 = t081_contrasts(hall, c2=_t081_mutated_scorer("c2_always_met"))
    out["R4"] = bool(
        got4["n_seeds"] == len(T081_R_SEEDS)
        and all(e["f_scr"][T081_LESION_DOSE] > 0.0 for e in hall)
        and not got4["c2"]["met"]
        and bool(got4["c2"]["failures"])
        and m4["c2"]["met"]
    )

    # R5 -- THE READABILITY REFUSAL. The mutation ignores the mismatch.
    shared = t081_constructed_read(seed=0, clone_fn=t081_clone_shares_rng, fresh=True)
    clean = t081_constructed_read(seed=0, fresh=True)
    ref = t081_refusals(shared)
    m5 = t081_refusals(shared, mismatch=_t081_mutated_scorer("r_ignores_mismatch"))
    out["R5"] = bool(
        shared["mismatch"] > 0
        and clean["mismatch"] == 0
        and ref
        and not t081_refusals(clean)
        and not m5
    )

    # R6 -- THE DOSE LADDER. The mutation returns the first dose tested.
    out["R6"] = bool(
        t081_flip_dose(T081_R6_LADDER) == T081_R6_FLIP
        and t081_flip_dose(T081_R6_ALL_ZERO) == "none"
        and _t081_mutated_scorer("ladder_first_dose")(T081_R6_LADDER) != T081_R6_FLIP
    )
    return out


# =============================================================================
# 2.9 STEP 5: THE RECORDER, the determinism precondition, the apparatus DRY RUN and the campaign.
# Everything below touches a brain; everything above does not. Every brain here is
# `make_brain(seed, persist=False)` under `throwaway_home()` -- a fresh NORD_HOME for the
# duration -- so nothing in this file can reach the daemon's brain or the durable lake.
# =============================================================================
def t081_record_brain(seed, n_ticks=T081_N_TICKS, keep_pre=False):
    """THE RECORDER (2.1). A SCRATCH brain under `throwaway_home()`, built with
    `make_brain(seed, persist=False)`, both stream pointers set to
    `(T081_STREAM_OFFSET * seed) % T081_STREAM_POSITIONS` (T080's rule), and the ARM's bus --
    `b.workspace`, the object `report_channel.snapshot` reads (fact 1) -- with its `step` WRAPPED
    ON THE INSTANCE.

    THE WRAPPER IS THE WHOLE APPARATUS. Before each real step it takes a clone `pre`; the real
    step then runs untouched; the real report `r0` is read THROUGH the report channel; and every
    arm of 2.2 is carried on fresh clones of that same `pre`. The live workspace is never
    perturbed -- `g_intact` says so on every row -- and the wrapper is popped out of every clone,
    so no clone re-enters the recorder.

    A ROW IS AN AWAKE TICK ON WHICH THE WRAPPER RAN, and both halves are required: `b.tick()`
    must return "awake" AND the arm must actually have stepped its bus that tick. A sleeping tick
    has no broadcast to lesion, and an awake tick that never reached the workspace has no
    pre-step state to clone. `wrapper_ticks` and `awake_ticks` are both REPORTED, so a recording
    where the two diverge says so rather than quietly shrinking.

    `keep_pre` retains each row's pre-step clone for the dry run's scrambled re-draws (2.3's
    REPORTED Monte-Carlo effect of the scrambled pool's random choice). They are dropped by the
    caller as soon as the re-draws are done; the campaign never asks for them."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    seed = int(seed)
    n_ticks = int(n_ticks)
    rng = random.Random(T081_SCR_RNG_BASE + seed)
    out = {"seed": seed, "n_ticks": n_ticks, "rows": [], "pres": [] if keep_pre else None}
    t0 = time.time()
    with throwaway_home():
        b = make_brain(seed=seed, persist=False)
        out["build_s"] = round(time.time() - t0, 3)
        b._real_ptr = b._audio_ptr = (T081_STREAM_OFFSET * seed) % T081_STREAM_POSITIONS
        out["offset"] = int(b._real_ptr)
        out["positions"] = len(b._real_labels)
        ws = b.workspace
        out["workspace"] = type(ws).__name__
        real_step = ws.step
        state = {"t": 0, "wrapper": 0}
        pending = []

        def wrapped():
            state["wrapper"] += 1
            pre = t081_clone(ws)
            result = real_step()
            r0 = snapshot(ws)
            row = _t081_tick_row(
                state["t"], ws, pre, r0, rng, T081_DOSES, snapshot, _t081_content_of, t081_clone
            )
            pending.append((row, pre))
            return result

        ws.step = wrapped
        kinds = {}
        tr = time.time()
        for t in range(n_ticks):
            state["t"] = t
            pending.clear()
            kind = b.tick()
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind != "awake":
                continue
            for row, pre in pending:
                out["rows"].append(row)
                if keep_pre:
                    out["pres"].append(pre)
        out["record_s"] = round(time.time() - tr, 3)
        out["kinds"] = kinds
        out["awake_ticks"] = kinds.get("awake", 0)
        out["wrapper_ticks"] = state["wrapper"]
        ws.step = real_step
        del b
    out["total_s"] = round(time.time() - t0, 3)
    out["ms_per_tick"] = (out["record_s"] * 1000.0 / n_ticks) if n_ticks else 0.0
    return out


def t081_seed(seed, n_ticks=T081_N_TICKS, keep_pre=False):
    """ONE seed's reading: record, read, name (r)'s refusals, drop the brain. The entry
    `t081_contrasts` and `t081_refusals` consume, with everything 2.3 REPORTS beside it."""
    rec = t081_record_brain(seed, n_ticks=n_ticks, keep_pre=keep_pre)
    out = t081_read(rec["rows"])
    out["seed"] = rec["seed"]
    for k in ("offset", "positions", "awake_ticks", "wrapper_ticks", "kinds", "workspace"):
        out[k] = rec[k]
    out["timings"] = {k: rec[k] for k in ("build_s", "record_s", "total_s", "ms_per_tick")}
    out["refusals"] = t081_refusals(out)
    if keep_pre:
        out["_rows"], out["_pres"] = rec["rows"], rec["pres"]
    return out


def t081_determinism(seed=0, n_ticks=T081_DETERMINISM_TICKS):
    """(p): two recordings of the same seed at the same width, compared FIELD BY FIELD -- not a
    sample of fields and not a tolerance. A nondeterminism anywhere in the recorder, the clone,
    the lesion or the brain is a REFUSAL before any number is read, and the first differing
    (row, field) is NAMED."""
    a = t081_record_brain(seed, n_ticks=n_ticks)
    b = t081_record_brain(seed, n_ticks=n_ticks)
    ra, rb = a["rows"], b["rows"]
    diff = None
    if len(ra) != len(rb):
        diff = f"row count {len(ra)} vs {len(rb)}"
    for i, (x, y) in enumerate(zip(ra, rb)):
        if diff:
            break
        if set(x) != set(y):
            diff = f"row {i}: field sets differ ({sorted(set(x) ^ set(y))})"
            break
        for k in sorted(x):
            if x[k] != y[k]:
                diff = f"row {i} field {k}: {x[k]!r} vs {y[k]!r}"
                break
    return {
        "seed": int(seed),
        "identical": diff is None and a["awake_ticks"] == b["awake_ticks"],
        "first_diff": diff,
        "rows": len(ra),
        "twin_rows": len(rb),
        "awake": [a["awake_ticks"], b["awake_ticks"]],
        "n_ticks": int(n_ticks),
    }


def t081_stream_labels(seed, n_labels=T081_OFFSET_LABELS, max_ticks=260):
    """The first `n_labels` REAL frame labels seed `s` walks with 2.1's offset applied. The
    offset check is an APPARATUS precondition, so it reads the labels the brain actually
    CONSUMED rather than re-deriving them from the pack by arithmetic (T080's idiom)."""
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate.home import throwaway_home

    out = []
    with throwaway_home():
        b = make_brain(seed=int(seed), persist=False)
        labels = b._real_labels
        b._real_ptr = b._audio_ptr = (T081_STREAM_OFFSET * int(seed)) % T081_STREAM_POSITIONS
        prev = int(b._real_ptr)
        for _ in range(max_ticks):
            if len(out) >= n_labels:
                break
            b.tick()
            ptr = int(b._real_ptr)
            if ptr != prev:
                out.append(int(labels[(ptr - 1) % len(labels)]))
            prev = ptr
    return out


def t081_scrambled_redraws(rows, pres, repeats=T081_SCR_REDRAWS, base=T081_SCR_REDRAW_BASE):
    """2.3's REPORTED Monte-Carlo effect of the scrambled pool's RANDOM CHOICE: the SCRAMBLED arm
    re-run on the recording's SAVED pre-step clones under `repeats` different RNG seeds, so the
    spread is the choice's own sampling noise and nothing else -- same ticks, same states, same
    lesion, only the draw varies. COMPUTED, never promised.

    The saved clones are read, never stepped: each draw takes a fresh `t081_clone` of them, which
    is what lets twelve draws share one recording."""
    out = []
    for i in range(int(repeats)):
        rng = random.Random(base + i)
        applicable = changed = 0
        for row, pre in zip(rows, pres):
            base_content = row["r0"]["content"]
            others = sorted(
                {k[1] for k in pre._g if k[1] != base_content}
                | {c[1] for c in pre._buffer if c[1] != base_content},
                key=repr,
            )
            if not others:
                continue
            applicable += 1
            c = t081_clone(pre)
            t081_lesion(c, others[rng.randrange(len(others))], T081_LESION_DOSE)
            c.step()
            if snapshot(c)["content"] != base_content:
                changed += 1
        out.append(
            {
                "draw": i,
                "applicable": applicable,
                "changed": changed,
                "frac": (changed / applicable) if applicable else 0.0,
            }
        )
    return out


def _t081_constants():
    """Every declared constant of 2.8, pinned beside the reading it produced."""
    return {
        "n_ticks": T081_N_TICKS,
        "doses": list(T081_DOSES),
        "lesion_dose": T081_LESION_DOSE,
        "stream_offset": T081_STREAM_OFFSET,
        "stream_positions": T081_STREAM_POSITIONS,
        "seeds": list(T081_SEEDS),
        "applicable_floor": T081_APPLICABLE_FLOOR,
        "min_contents": T081_MIN_CONTENTS,
        "min_changes": T081_MIN_CHANGES,
        "flip_floor": T081_FLIP_FLOOR,
        "scr_redraws": T081_SCR_REDRAWS,
        "scr_rng_base": T081_SCR_RNG_BASE,
        "scr_redraw_base": T081_SCR_REDRAW_BASE,
        "determinism_ticks": T081_DETERMINISM_TICKS,
        "offset_labels": T081_OFFSET_LABELS,
        "bar": T081_BAR,
        "clone_fields": list(T081_CLONE_FIELDS),
    }


def _t081_head():
    """The tree's HEAD, for the pin. Unavailable (no git, a tarball) is recorded as such."""
    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else f"unavailable (rc {r.returncode})"
    except Exception as e:  # noqa: BLE001 -- a pin field, never a control-flow decision
        return f"unavailable ({type(e).__name__})"


# distinct-from: gates/policy_manifest.py::_tree_state -- the same `git status --porcelain`
#   read, kept beside `_t081_head`'s bare-sha pin field (the t079/t080/pin convention) rather than
#   replacing it; added at T081's panel fold (2026-09-16) because the executor's dry run carried the
#   LAST COMMIT's sha while running on an uncommitted step-5 tree (record 3.5 item 12).
def _t081_dirty():
    """Whether the working tree differed from HEAD when this artifact was written -- a pin field
    beside `head`, so a stamp can never again name a commit that could not have run the code."""
    import subprocess

    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=20
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else f"unavailable (rc {r.returncode})"
    except Exception as e:  # noqa: BLE001 -- a pin field, never a control-flow decision
        return f"unavailable ({type(e).__name__})"


# distinct-from: instruments/binding_instrument.py::_t080_seed_block and
#   cortex/agency_monitor.py / instruments/attention_schema.py's per-seed printers -- the same
#   one-printer-per-instrument CONVENTION over a DIFFERENT reading (a bound fraction against a
#   permutation null there, a per-dose report-change fraction here). It exists so the dry run and
#   the campaign cannot drift into two report formats. Read.
def _t081_seed_block(r, out=print):
    """Everything 2.3 REPORTS for one seed, printed."""
    out(
        f"seed {r['seed']}: {r['n']} rows from {r['awake_ticks']} awake ticks "
        f"(wrapper ran {r['wrapper_ticks']}x; kinds {r['kinds']}; offset {r['offset']} of "
        f"{r['positions']}; bus {r['workspace']})"
    )
    out(
        f"  report: {r['contents']} distinct contents, {r['changes']} tick-to-tick changes, "
        f"availability {r['availability']:.4f}, mean pools {r['n_pools']:.2f}, mean top_margin "
        f"{r['top_margin']:.4f}"
    )
    out(
        f"  readable: twin mismatches {r['mismatch']}/{r['n']}; applicable named "
        f"{r['applicable_named']}/{r['n']} scrambled {r['applicable_scr']}/{r['n']}; "
        f"pre_intact {r['pre_intact']}/{r['n']}; g_intact {r['g_intact']}/{r['n']} "
        f"(structurally satisfied -- reported, not counted)"
    )
    for arm, key in (("NAMED", "f_named"), ("SCRAMBLED", "f_scr")):
        out(
            f"  {arm:<9} content-change by dose: "
            + ", ".join(f"{d}={r[key][d]:.4f}" for d in T081_DOSES)
        )
    for arm, key in (("NAMED", "s_named"), ("SCRAMBLED", "s_scr")):
        out(
            f"  {arm:<9} strength-change by dose: "
            + ", ".join(f"{d}={r[key][d]:.4f}" for d in T081_DOSES)
        )
    out(
        f"  flip dose {r['flip_dose']} (floor {T081_FLIP_FLOOR}) beside mean top_margin "
        f"{r['top_margin']:.4f}; (c2) scrambled changes at dose {T081_LESION_DOSE}: "
        f"{r['scr_changed_1']} (ticks {r['scr_changed_ticks'][:12]})"
    )
    for why in r["refusals"]:
        out(f"  (r) REFUSAL: {why}")


def run_t081_dry_run(out_path=T081_DRY_RUN_OUT, seed=0):
    """2.9 step 5's APPARATUS dry run: ONE seed at the CAMPAIGN's width, and NOT ONE barred
    conjunct -- one seed cannot carry a paired t, so (c1) is printed as a single seed's
    difference with "n/a" where the t would go.

    What it DOES read, all of it reported in 3.0 before the campaign:
      * the STREAM-OFFSET precondition: seeds 0 and 1 must walk different frames;
      * (p)'s determinism twin: two 100-tick recordings compared field by field;
      * the full 600-tick recording, read whole: the dose ladder on both arms, the readability
        twin, the applicability of both lesions, the report's own variability;
      * the 12 SCRAMBLED RE-DRAWS on the recording's saved pre-step clones;
      * the cost, re-measured on the BUILT apparatus (2.8 carried ~65 ms from fact 4's probe).

    Writes its JSON to `out_path` and NOTHING under tools/: the pin is the campaign's."""
    out_path = Path(out_path)
    if "tools" in out_path.parts:
        print(f"REFUSED: the dry run never writes under tools/ ({out_path})", flush=True)
        return 2
    t0 = time.time()
    print("=" * 78, flush=True)
    print(
        f"T081 DRY RUN -- one seed ({seed}) at the campaign width, NO barred conjunct", flush=True
    )
    print("=" * 78, flush=True)

    labels0, labels1 = t081_stream_labels(0), t081_stream_labels(1)
    differ = bool(labels0) and labels0 != labels1
    first_diff = next((i for i, (x, y) in enumerate(zip(labels0, labels1)) if x != y), None)
    print(
        f"stream offset ({T081_STREAM_OFFSET} x seed mod {T081_STREAM_POSITIONS}, BOTH pointers): "
        f"seed0 {labels0[:12]} | seed1 {labels1[:12]} | differ={differ} "
        f"first_diff_at={first_diff} (n {len(labels0)}/{len(labels1)})",
        flush=True,
    )
    if not differ:
        print("REFUSED at the apparatus: the per-seed offset does not vary the frames", flush=True)
        return 2

    det = t081_determinism(seed)
    print(
        f"(p) determinism twin: identical={det['identical']} on {det['rows']} rows of "
        f"{det['n_ticks']} ticks (awake {det['awake']}, first_diff {det['first_diff']})",
        flush=True,
    )

    reading = t081_seed(seed, keep_pre=True)
    rows, pres = reading.pop("_rows"), reading.pop("_pres")
    print("-" * 78, flush=True)
    print(
        f"cost: {reading['timings']['ms_per_tick']:.1f} ms/tick measured on the BUILT apparatus "
        f"over {T081_N_TICKS} ticks with ten workspace steps per awake tick (2.8 carried ~65 ms "
        f"from fact 4's five-clone probe) -> campaign estimate "
        f"{reading['timings']['record_s'] * len(T081_SEEDS) / 60.0:.1f} min for "
        f"{len(T081_SEEDS)} seeds; timings {reading['timings']}",
        flush=True,
    )
    print("-" * 78, flush=True)
    _t081_seed_block(reading)

    print("-" * 78, flush=True)
    draws = t081_scrambled_redraws(rows, pres)
    fracs = [d["frac"] for d in draws]
    print(
        f"scrambled RE-DRAWS ({len(draws)} draws, seeds {T081_SCR_REDRAW_BASE}..+"
        f"{len(draws) - 1}, the SAME saved pre-step clones): changed "
        f"{[d['changed'] for d in draws]} of {draws[0]['applicable'] if draws else 0} applicable; "
        f"fractions min {min(fracs, default=0.0):.4f} max {max(fracs, default=0.0):.4f} "
        f"mean {(sum(fracs) / len(fracs)) if fracs else 0.0:.4f} "
        f"sd {statistics.stdev(fracs) if len(set(fracs)) > 1 else 0.0:.4f}",
        flush=True,
    )
    del pres

    print("-" * 78, flush=True)
    print(
        "NO barred conjunct is read here: (c1) is a paired t over 8 seeds and one seed cannot "
        "carry one. (g0) and (c2) are per-seed counts and are printed as this seed's.",
        flush=True,
    )
    print(
        f"  (c1) one seed: f_named({T081_LESION_DOSE}) {reading['f_named'][T081_LESION_DOSE]:.4f} "
        f"- f_scr({T081_LESION_DOSE}) {reading['f_scr'][T081_LESION_DOSE]:.4f} = "
        f"{_t081_c1(reading):+.4f}  t: n/a",
        flush=True,
    )
    print(
        f"  (g0) one seed: {reading['contents']} distinct contents (floor {T081_MIN_CONTENTS}), "
        f"{reading['changes']} changes (floor {T081_MIN_CHANGES}) -> "
        f"{reading['contents'] >= T081_MIN_CONTENTS and reading['changes'] >= T081_MIN_CHANGES}",
        flush=True,
    )
    print(
        f"  (c2) one seed: {reading['scr_changed_1']} scrambled ticks moved the report", flush=True
    )
    print(f"  (r)  one seed: {reading['refusals'] or 'no refusal'}", flush=True)

    report = {
        "row": "T081 dry run",
        "head": _t081_head(),
        "dirty": _t081_dirty(),
        "seed": int(seed),
        "stream": {"seed0": labels0, "seed1": labels1, "differ": differ, "first_diff": first_diff},
        "determinism": det,
        "reading": reading,
        "scrambled_redraws": draws,
        "reds": t081_reds(),
        "constants": _t081_constants(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("-" * 78, flush=True)
    print(
        f"dry run written to {out_path} in {time.time() - t0:.0f} s; nothing under tools/",
        flush=True,
    )
    return 0


# distinct-from: instruments/binding_instrument.py::run_t080_campaign and
#   cortex/agency_monitor.py::run_t079_campaign and instruments/attention_schema.py's -- the PATTERN
#   this follows (record, read, contrast, pin ONCE) over a different instrument, different arms
#   and a different pin. Each row's campaign is its own instrument's; a shared runner would be a
#   fifth mechanism, not one. Read; nothing is shared beyond `instruments/paired_contrast`.
def run_t081_campaign(seeds=T081_SEEDS, pin=T081_PIN):
    """The campaign (2.2-2.3): record, read, contrast, pin ONCE. Returns an exit code.

    ORDER IS 2.3's, and it is the order the verdict is DECIDED in:
      (p)  DETERMINISM  two seed-0 recordings identical field by field, else REFUSED -- nothing
                        is read
      (r)  READABLE     the SAME clone's report equal to the real report on every awake tick, and
                        both lesions applicable on at least half of them, else REFUSED with the
                        seed and the clause NAMED
      (g0) THE REPORT VARIES  at least two distinct contents and one change per seed, else
                        REFUSED ("a constant report cannot be read for its generation order"),
                        never a STOP
      (c1) (c2)         the licensed sentence at the lesion, and the dissociation input
    A REFUSAL is a finding about the apparatus or the instrument; a NOT MET is a finding about
    THIS composite under THIS stream. They are never merged."""
    pin = Path(pin)
    if pin.exists():
        print(f"REFUSED: {pin} exists -- the pin is written once", flush=True)
        return 2
    t0 = time.time()
    print("=" * 78, flush=True)
    print(f"T081 CAMPAIGN -- seeds {list(seeds)}, ten arms per awake tick", flush=True)
    print("=" * 78, flush=True)

    labels0, labels1 = t081_stream_labels(0), t081_stream_labels(1)
    if not labels0 or labels0 == labels1:
        print(
            f"REFUSED at the apparatus: the per-seed stream offset does not vary the frames "
            f"(seed 0 {labels0[:10]} vs seed 1 {labels1[:10]})",
            flush=True,
        )
        return 2
    det = t081_determinism(seeds[0] if seeds else 0)
    print(
        f"(p) determinism: identical={det['identical']} on {det['rows']} rows "
        f"(first_diff {det['first_diff']})",
        flush=True,
    )
    if not det["identical"]:
        print("REFUSED: two identical-seed recordings differ -- no reading is taken", flush=True)
        return 2

    per_seed, refusals = [], []
    for s in seeds:
        r = t081_seed(s)
        per_seed.append(r)
        refusals.extend(r["refusals"])
        print("-" * 78, flush=True)
        _t081_seed_block(r)

    got = t081_contrasts(per_seed)
    verdict = t081_verdict(got, refusals)
    print("=" * 78, flush=True)
    for why in refusals:
        print(f"  (r) REFUSAL: {why}", flush=True)
    print(f"  (g0) the report varies: {got['g0']}", flush=True)
    c = got["c1"]
    print(
        f"  {c.name}: diffs {[round(d, 4) for d in c.diffs]} t={c.t:.4f} FPR={c.fpr} "
        f"degenerate={c.degenerate} met={c.met}",
        flush=True,
    )
    print(f"  (c2) the scrambled lesion leaves the report unchanged: {got['c2']}", flush=True)
    print(f"  REPORTED flip dose per seed: {[r['flip_dose'] for r in per_seed]}", flush=True)
    print(
        f"  REPORTED mean top_margin per seed: {[round(r['top_margin'], 4) for r in per_seed]}",
        flush=True,
    )
    print(f"T081 VERDICT: {verdict}", flush=True)

    out = {
        "row": "T081",
        "head": _t081_head(),
        "dirty": _t081_dirty(),
        "seeds": list(seeds),
        "constants": _t081_constants(),
        "reds": t081_reds(),
        "determinism": det,
        "stream_labels": {"seed0": labels0, "seed1": labels1},
        "per_seed": per_seed,
        "contrasts": {"g0": got["g0"], "c1": got["c1"]._asdict(), "c2": got["c2"]},
        "refusals": refusals,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"pinned {pin} in {time.time() - t0:.0f} s", flush=True)
    return 0 if verdict != "REFUSED" else 2


# =============================================================================
# The self-check line (2.9 step 2: main() prints the six booleans and EXITS NON-ZERO while any
# is False, so the gate carried the red until 2.9 step 4), plus 2.9 step 5's two brain-touching
# entry points.
# =============================================================================
def _t081_reds_line(reds, out=print):
    """Print the six booleans beside what each one says."""
    for name in sorted(reds):
        out(f"  {name}: {str(reds[name]):5s}  {T081_RED_LINES[name]}")


# distinct-from: instruments/binding_instrument.py::main and instruments/attention_schema.py's -- each
#   module's own `__main__` block, which is this tree's convention (run_selfchecks.py executes
#   exactly that block per module and scores the exit status), not a second dispatcher. Nothing
#   here runs another module's self-check. Read.
# distinct-from: instruments/binding_instrument.py::t080_pin_reading -- the same job (the C5 row's
#   reading re-derived from its pin at self-check, with a mutated derivation shown to refuse) over a
#   DIFFERENT pin schema (per-seed f_named / f_scr by dose, keyed by JSON strings) and a different
#   mutated derivation (the sides swapped, R2's blind scorer). Added at T081's panel fold (2026-09-16,
#   record 3.5 item 11). OPENING for the next charged row: ONE helper taking (row, pin, reducer) with
#   both per-row bodies retired into it -- a P2 merge larger than a fold's one clause.
def t081_pin_reading(pin=None):
    """(derived, recorded, mutated): 2.6's reading -- the mean over seeds of f_named(1.0) -
    f_scr(1.0) -- re-derived from the committed pin, beside the CONCLUSION row's recorded value and
    a mutated derivation (the sides swapped) that would refuse."""
    from snn_data_lake.gates.conclusion_protocol import CONCLUSION

    pin = T081_PIN if pin is None else Path(pin)
    assert pin.exists(), (
        f"{pin} missing -- the C5 reading's re-derivation cannot run; the pin is tracked, so absence "
        "means deletion or a broken path"
    )
    seeds = json.loads(pin.read_text(encoding="utf-8"))["per_seed"]
    k = str(T081_LESION_DOSE)  # the pin's dose keys are JSON strings
    derived = round(
        sum(float(e["f_named"][k]) - float(e["f_scr"][k]) for e in seeds) / len(seeds), 4
    )
    mutated = round(
        sum(float(e["f_scr"][k]) - float(e["f_named"][k]) for e in seeds) / len(seeds), 4
    )
    return derived, CONCLUSION["report_order"][3], mutated


def main(argv=None):
    """The module's self-check, plus 2.9's two brain-touching entry points.

    BOTH of those run under `throwaway_home()` INSIDE their recorders, every brain is
    `persist=False`, and NEITHER is reachable from the gate's bare `python -m` invocation --
    which runs the six reds on constructed workspaces and touches no brain, no lake and no
    network."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--t081-campaign" in argv:  # 2.9 step 6: the campaign, pinned ONCE
        return run_t081_campaign()
    if "--t081-dry-run" in argv:  # 2.9 step 5: the apparatus dry run, NO barred conjunct
        i = argv.index("--t081-dry-run")
        path = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else None
        return run_t081_dry_run(T081_DRY_RUN_OUT if path is None else path)
    print("=" * 78)
    print("T081 -- the report-order probe (specs/022 [US4], C5 row report_order)")
    print("=" * 78)
    print(
        f"  ticks {T081_N_TICKS} | doses {list(T081_DOSES)} | seeds {list(T081_SEEDS)} | "
        f"offset {T081_STREAM_OFFSET} x seed mod {T081_STREAM_POSITIONS}"
    )
    print(
        f"  bar {T081_BAR} | applicability floor {T081_APPLICABLE_FLOOR} | flip floor "
        f"{T081_FLIP_FLOOR} | min contents {T081_MIN_CONTENTS} | min changes {T081_MIN_CHANGES} "
        f"| scrambled rng {T081_SCR_RNG_BASE} + seed ({T081_SCR_REDRAWS} re-draws)"
    )
    reds = t081_reds()
    print("-" * 78)
    print("2.4's six reds:")
    _t081_reds_line(reds)
    if not all(reds.values()):
        print(
            f"T081 SELF-CHECK RED: only {sum(reds.values())}/{len(reds)} of 2.4's reds fire on "
            f"their constructed inputs -- {reds}"
        )
        print(
            "  EXPECTED AT 2.9 STEPS 2-3: the skeleton's stubs return null readings so the reds "
            "RUN red. This line goes green at 2.9 step 4, never before."
        )
        return 1
    print(
        f"  T081 OK: R1-R6 fire on their constructed inputs and each is reddened IN FACT by its "
        f"mutated scorer ({sum(reds.values())}/{len(reds)}); no brain, no lake, no network"
    )
    print(
        f"  --t081-dry-run writes {T081_DRY_RUN_OUT} (one seed, NO barred conjunct); "
        f"--t081-campaign pins {T081_PIN.name} ONCE. Neither runs from here."
    )
    # THE CHARGED READING IS TIED TO ITS PIN HERE (T081 panel fold, 2026-09-16; T080's shape). A
    # missing pin is a RED (the pin is tracked), never a skip; the mutated derivation shows the clause
    # reachable on every run.
    derived, reading, mutated = t081_pin_reading()
    print(
        f"  C5 report_order: reading {reading} re-derived from {T081_PIN.name} as {derived} "
        f"(a mutated derivation reads {mutated} and would refuse)"
    )
    if derived != reading or mutated == reading:
        print(
            f"T081 SELF-CHECK RED: C5 reading {reading} != {derived} re-derived from {T081_PIN} "
            f"(mutated {mutated}) -- re-derive or explain before re-recording"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
