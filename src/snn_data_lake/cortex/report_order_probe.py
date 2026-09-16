"""Report-order probe (specs/022 T081 [US4]) -- the LIVE arm's report channel read against a
LESIONED CLONE of the workspace state that produced it.

THE ROW. tasks.md T081 [US4]; the C5 row `report_order` in `substrate/conclusion_protocol.py`;
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
(`cortex.brain.make_brain(seed, persist=False)`; the daemon's brain and the durable lake are
never touched). The arm's `GlobalWorkspace.step` is wrapped ON THE INSTANCE: before each real
step a CLONE of the competition state is taken, after it the real report `r0` is read THROUGH
`substrate.report_channel.snapshot` -- never through `ws.current`. Nine further clones of that
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

# distinct-from: thalamus/ignition_instrument.py -- an EXEMPT scorer whose subjects are
#   CONSTRUCTED ReentrantWorkspaces on a scratch bench, scoring an all-or-none DRIVE transition
#   with reds built as impostor SUBCLASSES of the subject. This module lesions CLONES of the
#   LIVE arm's GlobalWorkspace and reads the REPORT CHANNEL's record off them; its reds are
#   mutated SCORERS and stand-in reporters, not impostor subjects. The EXEMPT-instrument shape
#   is the thing reused. Read.
# distinct-from: reasoning/consciousness_probe.py -- the INCUMBENT this row retires (P2, T034's
#   clause: both replacement halves -- the report channel since T034 and this probe -- must
#   exist when it dies). UNTOUCHED BY THIS MODULE and by this builder: its port and its deletion
#   are the surfaces commit's own step. Read.
# distinct-from: substrate/report_channel.py -- the MECHANISM UNDER TEST. This module changes
#   nothing in it: it calls the public `snapshot` and re-implements no part of the record. Read.
# distinct-from: thalamus/binding_instrument.py -- T080's instrument over the CrossModalBinder's
#   report, membership-matched permutation, its own pin. The campaign/dry-run/pin PATTERN is the
#   thing reused; no state, no pin and no statistic is shared beyond `substrate/paired_contrast`,
#   which both IMPORT. Read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.cortex.report_order_probe
"""

import sys
from pathlib import Path

from snn_data_lake.substrate.paired_contrast import T_BAR

# =============================================================================
# 2.8's DECLARED CONSTANTS. Each is either the plan's width, the composite's own shape, or a
# quantity the pre-registration names -- and every BUILDER-INTRODUCED width says so (P1, and
# T079/T080's disclosure discipline: an undisclosed width is a hidden degree of freedom).
# =============================================================================
#: Ticks per seed's recording (2.2, 2.8).
T081_N_TICKS = 600
#: The lesion ladder (2.1, 2.3): dose 1.0 BARS -- it is the lesion -- and the rest are REPORTED.
T081_DOSES = (0.25, 0.5, 0.75, 1.0)
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
# 2.9 STEP 2 -- THE STUBS. Every one returns a NULL reading so that 2.9 step 3's reds run on
# their constructed inputs and FAIL AN ASSERTION (never a collection error, never a TypeError):
# the suite must say what the instrument OWES, not that it is absent. They are replaced, one
# function at a time, at 2.9 step 4 (the scorers) and 2.9 step 5 (the recorder).
# =============================================================================
def t081_clone(ws):
    """FACT 3's CLONE: a step-able copy of the workspace's COMPETITION STATE only.

    STUB (2.9 step 2) -- returns None, the null reading. The built form deep-copies
    `T081_CLONE_FIELDS`, pops the instance-level `step` wrapper out of the copy's `__dict__`,
    and detaches `_subscribers` (-> []) and `_out_ring` (-> None)."""
    return None


def t081_lesion(clone, content, dose):
    """THE PERTURBATION (2.1): scale `_g` by (1 - dose) for every pool key whose content equals
    `content`, and at dose 1.0 drop that content's `Coalition` records from `_buffer`.

    STUB (2.9 step 2) -- returns 0, the null reading (no pool lesioned). The built form returns
    how many pool keys it touched, which is what makes APPLICABILITY a measured quantity rather
    than an assumed one."""
    return 0


def t081_record_brain(seed, n_ticks=T081_N_TICKS):
    """THE RECORDER (2.1): a scratch brain under `throwaway_home()`, its arm's `workspace.step`
    wrapped on the INSTANCE, one row per awake tick carrying every arm.

    STUB (2.9 step 2) -- returns the empty recording. Built at 2.9 step 5; nothing above that
    line touches a brain, a lake or a network."""
    return {"seed": int(seed), "rows": [], "n_ticks": int(n_ticks), "timings": {}}


def t081_read(rows):
    """PER-SEED READING (2.1): the fractions f_named(d) and f_scr(d) over the APPLICABLE ticks
    at each dose, the mismatch count (`r_same != r0`), the distinct contents, the change count,
    the flip dose, and the mean `top_margin`.

    STUB (2.9 step 2) -- returns the null reading: no tick, no fraction, no flip dose."""
    return {
        "n": 0,
        "awake": 0,
        "applicable_named": 0,
        "applicable_scr": 0,
        "mismatch": 0,
        "contents": 0,
        "changes": 0,
        "f_named": dict.fromkeys(T081_DOSES, 0.0),
        "f_scr": dict.fromkeys(T081_DOSES, 0.0),
        "s_named": dict.fromkeys(T081_DOSES, 0.0),
        "s_scr": dict.fromkeys(T081_DOSES, 0.0),
        "scr_changed_1": 0,
        "flip_dose": None,
        "top_margin": 0.0,
        "rows": list(rows),
    }


def t081_flip_dose(fractions_by_dose):
    """THE FLIP DOSE (2.1, REPORTED never barred): the smallest dose whose named fraction is at
    or above `T081_FLIP_FLOOR`, else the string "none".

    STUB (2.9 step 2) -- returns None, which is neither a dose nor "none"."""
    return None


def t081_contrasts(per_seed):
    """2.3's conjuncts (g0), (c1) and (c2), each decided through
    `substrate/paired_contrast.contrast` -- the ONE statistic, re-spelled nowhere.

    STUB (2.9 step 2) -- returns the null contrasts: nothing met, nothing named."""
    return {"g0": None, "c1": None, "c2": None, "n_seeds": 0, "refusals": []}


def t081_reds():
    """2.4's six reds as ONE self-check line (each is ALSO a test in
    `tests/test_report_order_probe.py`; this is the in-command control 2.4 requires, so a gate
    run that never executes pytest still runs every one on the SAME constructed input).

    STUB (2.9 step 2) -- every red reads False, so `main()` exits non-zero and the gate CARRIES
    the red until 2.9 step 4. That is 2.9's order: the stubs before the reds, so the reds RUN
    red rather than being asserted to have been red."""
    return dict.fromkeys(sorted(T081_RED_LINES), False)


# =============================================================================
# The self-check line (2.9 step 2: main() prints the six booleans and EXITS NON-ZERO while any
# is False, so the gate carries the red until 2.9 step 4).
# =============================================================================
def _t081_reds_line(reds, out=print):
    """Print the six booleans beside what each one says."""
    for name in sorted(reds):
        out(f"  {name}: {str(reds[name]):5s}  {T081_RED_LINES[name]}")


def main(argv=None):
    """The module's self-check. It touches NO brain, NO lake and NO network: the six reds run on
    constructed inputs only. The brain-touching entry points (`--t081-dry-run`,
    `--t081-campaign`) land at 2.9 step 5 and are not reachable from the gate's bare
    `python -m` invocation."""
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
