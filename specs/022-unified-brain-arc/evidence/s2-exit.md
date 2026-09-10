# S2 (US3) exit certification -- specs/022 T071

Certified on the stacked head that lands this file, on the pinned measurement host
(`LUAN_AND_ANNIE`, `.venv-lake`). **This file certifies what was measured, including what did not
pass** -- the `s0-exit.md` / `s1-exit.md` precedent, kept: an exit certification that certifies only
its greens certifies nothing. Every number in this file is either (a) printed by a command run FOR
THIS FILE and quoted beside that command, or (b) quoted from a named pin or record with its path and
section. **Nothing here is derived by this file.**

The certification is the GATE for T072-T075 and T083 (PLAN-FINAL O5(d) `:289`, "T071 stays the gate
for T072-T075 and T083 -- that clause is NOT rescinded"; Part E `:812`, "Phase 5 waits for BOTH T071
and Phase 4"). The gate is discharged by this certification's EXISTENCE with every named item READ,
not by a pass: US2 exited with its headline criterion UNMET (`s1-exit.md` §1, "1 of 10 growable
regions") and T068, gated on it, ran.

## 0. Pre-registration -- the rules, written and COMMITTED BEFORE any command below was run

This section is frozen at the commit that lands it (`docs(022): T071 -- the S2 exit certification,
pre-registered`). Sections 1-4 and the Verdict were added by the later measured commit and **no line
of this section was edited by it.** If a command's reading differs from an expectation written here,
the difference is recorded in that command's own section as a FINDING; §0 is not corrected to match.

### 0.1 The three named items (PLAN-FINAL `:711`, tasks.md T071, FR-015) and the rule each is read by

**Item 1 -- prior separation.**
*Definition of record*: T063's pre-registration 2.6 -- conjuncts (b), (d) and (e) TOGETHER (the
harness separates the true prior from its marginal-matched null AND from its corruption on one
task), licensed by (c).
*Reading of record*: `tools/t063_campaign.json` at head `f37fe90` (K = 8, D = 4, W = 1.0, 8 seeds):
(c) MET t +9.2601; (b) NOT MET t -0.2277; (d) NOT MET t(random - wrong) -2.5459 with WRONG VALID on
8 of 8 seeds; (e) NOT MET t -5.2271.
*Exit reading*: **NOT MET, measured** -- INHERITED from T063 (`evidence/t063-prior-drift-harness.md`
§3.3 and its §3.7 fold), **NOT re-run**: the pin is the record of the learner's run and `--campaign`
rewrites it.
*Rule*: the item is read from the pin's five contrasts, QUOTED. The exit may **NOT** widen it to
"H01 contributes nothing" -- the null retains H01's degrees, signs and weight magnitudes (T063 2.2);
one task; one width.
*Carried beside it, not smoothed*: the learner was DELETED on its own criterion (t(d_f) -13.0154);
the SURVIVING no-learner harness is BLIND on H01 at D = 4 (PLANTED - RANDOM t +0.8143 -> NO VERDICT,
`t063` §3.4 appended paragraph); (b) was read between two arms 0.50 and 0.62 held-out trials above
chance (`t063` §3.7 opening table); the WRONG arm is not degree-matched to RANDOM (`t063` §3.5 item
12). **No number in item 1's section is produced by a command run for this certification.**

**Item 2 -- the paging verdict.**
*Definition of record*: SC-007 (spec.md `:178`) with its 2026-09-08 amendment; T068's
pre-registration 2.3 / 2.6 / 2.10.
*Reading of record*: `evidence/s5-paging-null.md` §3.4 -- the memory claim MET on every conjunct
(paged/twin footprint fractions 0.602 and 0.399, t 203.8 / 229.8, p 0.0039; paged < neutered 0.709,
t 107.8), the bounded secondary NOT MET by an order of magnitude (median paged/twin tick ratio RAW
35.2x, reference-corrected 26.0x, against a 3.0x bound fixed before the run); arm F does NOT
discriminate on this host; **LAYER VERDICT: LOSS** under 2.3. The panel's verdict on the null (§4:
31 of 31 standing findings, the LOSS stands; Codex's two refutations answered) and the deletion
applied as the fold.
*Exit reading*: **SC-007 SATISFIED by its second branch** -- "the layer is deleted and the deletion
recorded" -- and the paging CAPABILITY is a measured **LOSS**.
*Rule*: quote §3.4's conjunct table and the deletion's commit list; **nothing is re-run** (the
harness is archived under the tag `t068-last-with-pager`). The exit may **NOT** call the LOSS
"paging cannot work": §3.4 records the cost as THIS implementation's.
*Carried beside it*: what survived the deletion and what it leaves owed -- T064's 0x09 synapse-delta
path (its block framing ported to `lake_engine.SynapseDelta.group_by_block`; the row void until a
successor paging row) and T065's binding-width derivation, whose slope was found to be the STREAM'S
LABEL CARDINALITY (271 + 8*m), so the derived width is a projection with named debts (persisted
bytes, m held fixed, more than one seed; the 5.0 MB projection 16x high).

**Item 3 -- the region charge gate at floor(all-but-exempt).**
*Definition of record*: FR-015 (spec.md `:124`) -- the gate MUST source from live reachability PLUS
port-edge charge attribution; exemptions carry date+owner+expiry with a ratcheting COUNT ceiling;
the S0 exit is an integer FLOOR of regions charged whose IDENTITY is pinned
(`tools/region_charge_floor.json`: floor 8, raised 7 -> 8 at T039f, six named regions,
`named_and_NOT_charged` empty, identity 6 of 6). The gate's denominator is
REGIONS - {substrate, plasticity} + insula = 16.

**THE INTERPRETIVE STEP, stated before the number.** "at floor(all-but-exempt)" has two readings.

  (A) the gate holds at its recorded floor, the floor's denominator being every region that is not
      EXEMPT (exempt rows never count against the floor; today EXEMPT is empty and its ceiling is 0,
      so all-but-exempt = all 16) -- i.e. `charged >= floor` with `exempt <= ceiling`, identity
      intact.
  (B) every non-exempt region is CHARGED (16 of 16 with 0 exemptions).

**This certification READS THE ITEM BY (A)** -- FR-015 defines the exit as an integer floor with a
pinned identity, never as full coverage, and the "(all-but-exempt)" qualifier names the floor's
DENOMINATOR -- **and REPORTS (B) beside it** as the measured coverage with every uncharged region
named. If (A) reads MET and (B) reads NOT MET, the Verdict table carries **BOTH** rows, and the
panel's pre-registration-honesty lens is asked FIRST whether choosing (A) is a relabelling; if it
rules so, the item is recorded NOT MET under (B) and this paragraph stays as the record of the
choice.

*Command* (run fresh at the head for this certification; read-only, no brain, no lake-home writer):

    PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.region_charge_gate --assert

*Expected from the last green battery (T062 head)*: 8 of 16 CHARGED, 0 exempt (ceiling 0), floor
holds 8 >= 8; REACHABLE-UNCHARGED amygdala, brainstem, cerebellum, claustrum, epithalamus,
hippocampus; UNREACHABLE olfactory, pons. **The fresh reading is what this file quotes; the
expectation is written here so a difference is a FINDING, not a surprise.**

### 0.2 The US3 row census -- the exit certifies the SET, not one row

Every row T062-T071 with its recorded state, as of this pre-registration:

| row | recorded state |
|---|---|
| T062 | DONE (artifact pinned: 14,215 signed units / 70,001 edges; record `1d4602c`) |
| T063 | DONE, **NEGATIVE** (item 1 above) |
| T064 | **VOID with the pager** (the 0x09 synapse-delta path survives; a successor paging row is owed) |
| T065 | **PARTIAL** (slope = label cardinality; the debts named) |
| T066 | DONE, then **DELETED with T068** (its reds and the byte-identical calm paged run are the record) |
| T067 | DONE (`lake_compactor`); its pager-dependent halves deleted at T068, the reorder over the store with them |
| T068 | DONE, **LOSS** |
| T069 | **REFUSED ON MEASUREMENT** (removed; Phase 2 per O5(a)) |
| T070 | **OPEN** and NOT a gate for this exit (O5(b): Phase 4, parallel) |
| T071 | this file |

Also carried, and NOT one of the three items: **S6** -- the kernel wire DECLINED at the deployed
width by Gate Zero (`evidence/s6-kernel-wire.md`).

### 0.3 The commands this certification runs (all read-only, all at the head)

1. `region_charge_gate --assert` (item 3; **the only fresh measurement of a criterion**).
2. `dormancy_gate --assert` -- the US3 modules' rows quoted (`h01_ingest`'s FIXTURE role;
   `prior_drift_harness` as INSTRUMENT; `lake_compactor`) so the exit names what is live and what is
   a fixture.
3. `python -c` readers over `tools/t063_campaign.json` (the five contrasts and the per-arm means
   against chance) and over `tools/region_charge_floor.json` (floor, identity) -- QUOTATIONS with
   their pins, never derivations.
4. `git log --format=%h` over the cited commits (an EXISTENCE check only).

**NOT run, by rule:** no `--campaign`, no `--pilot` (both rewrite the pins), no brain construction,
no lake-home writer, and never the live lake.

### 0.4 What this certification does NOT cover -- drafted BEFORE the runs; the build may only ADD

1. **Prior separation** -- NOT MET, and not re-readable with the surviving instrument at D = 4
   (BLIND) or with any learner (none exists); a successor owes a NEW pre-registration of the
   capacity guard AND a learning signal a local rule can use.
2. **Paging** -- a LOSS; residency beyond RAM is unowned; T064's and T065's debts as named above.
3. **Region charge** -- 8 of 16 charged; six reachable-uncharged, two unreachable; the floor is a
   RATCHET, not coverage.
4. **S6** -- the kernel wire declined; the exemptions re-arm at the first tick population at or
   above N*.
5. Nothing here is a claim about the LIVE brain's capability (`capability_bar`: no row earned).
