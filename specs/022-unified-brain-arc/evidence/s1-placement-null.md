# Growth → placement wire: measured null, reverted under P3 (audit §8 rank 1, 2026-09-02)

Measured on `main` at `8424744a` (the trajectory audit's own commit), pinned host, `.venv-lake`,
every arm under `substrate/home.throwaway_home()` with `persist=False` — the live lake and the
daemon were never touched (the panel's open-probe over the whole run: 0 reads or writes of the real
home, 0 repo writes). Owner directive: *"Start with audit §8 rank 1, wire compose_growers."*

## What was built

The wire the audit asked for, exactly: `cortex/placement.GROWER_REGIONS` gained the live
experience-driven grower (`"growing_learner": "hippocampus"` — `hippocampus/growing_learner.py`,
the ART-style recruiter behind the perception units); `Brain.__init__` composed the growers through
the 007 interface (`self._grow_coord, self._grow_ledger = compose_growers(base_seed=seed)`); and
`Brain._sleep_tick` reported the seam's width after the engine's tick
(`self._grow_coord.grow("growing_learner", self._real_units)`), so each night the day's recruits
rippled into `place_grown("growing_learner", delta)` placements. It worked, deterministically: the
archived arms recruit 24 perception prototypes in 600 ticks and the ledger holds 24 hippocampal
placements (an in-session 120-tick smoke, not archived, showed 20 → 20). The diff is 65 lines,
archived outside the repo at `~/.claude/projects/D--AI-project/memory/archive/audit-2026-09-01/`
(`wire_compose_growers.diff`, beside the five-arm harness `ab_placement.py`).

## What was measured

Five arms, one subprocess each, seed 0, 600 ticks (≥ one engine refit; the live arms grew 7 cortex
units). Differencer: every key of `Brain.state()` and `brain_telemetry.snapshot()` flattened (617
keys; the snapshot embeds `state()`, so a moved quantity counts twice); the nondeterminism baseline
is the key set that differs between two identical live runs, measured in the same command. Value
equality, not bytes.

| arm | what it neuters | reach control | keys differing from live beyond the baseline |
|---|---|---|---|
| live_b (baseline) | nothing — a second identical run | — | **1** (`snap.t`, the wall clock) |
| neuter_wire | `brain._grow_coord = None` — no hippocampal placement ever happens | ledger `growing_learner` 24 → 0 | **0** |
| neuter_place | `placement.place` → constant `(0.5, "Layer 2", 1)` — BOTH regions' placement carries no information | engine `layer_census` 5 layers → `{Layer 2: 7}` | **0** |
| growth_off (positive control) | `neuro.tick` disabled | `grown_units` 7 → 0 | **18** (9 unique: `grown_units`, `predict_accuracy` 0.717→0.200, `growth_gate.cortex`, `reliability`, `surprise`, `self_state.*`) |

The review panel's instrument lens reproduced the null under a stronger differencer of its own
(eight arms, 300 ticks): a deep walk of the whole `Brain` object graph (263,004 paths, arrays
hashed) differs from live only under `neuro._placement[*]`; the growth blob with its `place_*`
arrays removed hashes identically across live / neuter / placement-seed-shift; the cockpit's `/`,
`/api/state` and `/api/all` renders are identical but for `t`; and a **same-stage control** — the
engine's seed shifted inside `_refit`, where placement is drawn — moves `predict_accuracy`
(0.630 → 0.457) while a placement-seed-only shift moves nothing outside the record. That control is
now part of the committed test.

Reproduce: `tests/test_placement_charge_null.py` (live ×2, growth-off, engine-seed-shift,
placement-neutered; 300 ticks each; ~2 minutes) or `ab_placement.py` in the archive directory
above (600 ticks, subprocess arms).

## Verdict

**Placement is write-only on the live path** — at HEAD for cortex, and on the wired tree for
hippocampus. Its only effect on the running brain is its own record: the engine's
`layer_census`/`placement()` accessors (read by nothing outside the engine's self-check) and the
bytes persisted in the growth blob (read back only to rebuild the record). No decision, value or
learning rule on the tick reads a unit's position, layer or sign; the reply path was not exercised
by these arms and is not claimed. Outside cortex the record additionally carries no per-unit
information by construction — `_place_grn` returns a uniform random position, one label per region
(`'anterior'` for every hippocampal unit at this grain) and sign `None` (refused: no measured
E-fraction exists outside H01 cortex).

**One reader of the record WAS built, and it was not on a live path** (as of `c4a64613`; wired at
`847c5f6a`, measured a NULL and DELETED in the commit after it — see the rank 1b paragraph below):
`cortex/genomic_bottleneck` (FR-033, T044 — `unit_tag` read position, layer and sign; `init_fanin`
mapped tags to a grown unit's initial fan-in). It was dormancy-EXEMPT `wire`, its docstring named the wiring it awaited
(swap the random candidate pool inside `hippocampus/ng_snn.py::NGSNN.fit`, reached through the
engine's refit), states that this has no task id, and pre-registers its own delete-on-null. The
static "nothing reads placement" this note first carried was therefore an over-claim of exactly the
kind P3 warns about from the other side; the runtime null stands, the sentence did not.

So the wire "does not charge" in P3's exact sense, and P3's honesty guard decides the disposition:
*"a wire that does not charge is reverted and retired/exempted, never faked with a dead consumer."*
The wire is **reverted**. The composition (`compose_growers`, the coordinator, the routing table)
stays as it was — built and self-checked — and its docstrings now give the measured reason for
having no live caller in place of the expired scheduling one ("T049+ is the dated consumer": T049
closed 2026-08-26 without consuming it).

## What this changes about the coverage red

`growth_coverage_gate` reports *"1 of 10 growable regions receive grown units from a LIVE path"*. That
census resolves `place`/`place_grown` call sites; it is a census of who WRITES a placement record.
Had the wire landed with the pin moved to `{cortex, hippocampus}`, the red would have read 2 of 10
while the second region's record carried zero information and no reader — coverage of a write-only
record, which P0's guard names directly (*"no coverage band or P0 avenue may claim the capability
until it is measured"*). The gate's report now says so in one line, printed on every run beside the
red and pinned by its self-check. The 1 is the same kind of 1: cortex placement is equally unread on
the live path. The honest reading of SC-004's first clause today is therefore not "one region grows,
nine do not" but "the growth→placement channel exists and no live path consumes it in any region."

## What would make placement load-bearing — rank 1b is a WIRE, not a build

1. **Run FR-033's own A/B first** (`cortex/genomic_bottleneck`, already built): map-drawn initial
   fan-in vs the incumbent random pool inside `NGSNN.fit`'s candidate loop, on the earned substrate.
   Its red is this note's pinned test going RED (a live reader now exists and moves a downstream
   key); its null is the module's own pre-registered deletion (P8 clause 5). The project's standing
   rule applies: zero refs → WIRE, not build.
2. Only if that A/B is a null do new readers come into question — Dale's sign on the NG-SNN readout
   (`beta`, a sign-constrained direct solve) or layer-keyed lateral connectivity (the
   `cortical_microcircuit` per-layer probabilities). Nearest recorded negative, to be read first:
   v3.59 "neocortical laminar readout + neurogenesis" (63.7%, `project-snn-refuted-and-negatives`),
   an STDP label-vote regime; NG-SNN's direct-solve readout is a different regime and the claim
   here would be charge, not accuracy — but the citation is owed.
3. Outside cortex nothing can be consumed until a per-unit prior exists (a dorsoventral field or a
   measured occupancy for a second region) — the gap `placement.py`'s own docstring names.

The pinned test is what will say so: `test_placement_charge_null.py` turns RED the day any key of
`state()`/telemetry depends on placement, with the reopen and retire-the-disclosure instructions
in its message.

**Rank 1b executed the same day: NULL.** The `genomic_bottleneck` wire was built and measured
(`evidence/s1-genome-fanin-ab.md`): at 24 paired seeds map-drawn fan-in is +0.002 ± 0.005 over the
random pool and indistinguishable from a tag-shuffled control (and, on the panel's carried
continuation, a measured LOSS), so the module's pre-registered deletion applied: wire + measurement
at `847c5f6a`, the module deleted in the commit after it. No reader of the placement record exists
now. Item 1 above is closed as a null; items 2 and 3 stand.

## Correction to the audit

The audit's §8 wrote that each of its ten items is *"a wire or a measurement over machinery that
exists"* and listed `compose_growers` / `place_grown` as *"built with zero callers outside their
module"*; the lens report it drew on (L14) called this rank *"a wire and an attribution, not a
build"*. The wire half was right and took an hour; the attribution half is what the measurement
refused, and the audit did not know about the built-but-unwired reader in `genomic_bottleneck`.
Ranks 2–4 (real percepts into the grower, `GrowingLearner` persistence, unit blocks in the lake) do
not depend on this result and stand as ordered; rank 1b above precedes any second attempt at rank 1.
