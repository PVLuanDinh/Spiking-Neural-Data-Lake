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

---

*Everything below this line was written and committed AFTER the commands ran
(`docs(022): T071 -- the S2 exit certification, measured`). Section 0 above is untouched by it.*

## Verdict

| # | criterion | measured | status |
|---|---|---|---|
| 1 | **Prior separation** -- T063 2.6's conjuncts (b), (d), (e) together, licensed by (c) | (c) +9.2601 MET (the instrument is not blind); (b) -0.2277, (d) -2.5459, (e) -5.2271, all NOT MET -- `tools/t063_campaign.json` at `f37fe90` | **NOT MET, measured** -- see §1 |
| 2 | **The paging verdict** -- SC-007 at the derived comparison width | memory claim MET on every conjunct (0.602 / 0.399, t 203.8 / 229.8, p 0.0039; paged < neutered 0.709); bounded secondary NOT MET by an order of magnitude (35.2x raw / 26.0x corrected against a 3.0x bound); **LAYER VERDICT: LOSS**; the layer deleted and the deletion recorded | **SC-007 SATISFIED by its second branch; the paging CAPABILITY is a measured LOSS** -- see §2 |
| 3 | **Region charge at floor(all-but-exempt), reading (A)** -- `charged >= floor`, `exempt <= ceiling`, identity intact | 8 of 16 CHARGED, 0 exempt (ceiling 0), `floor holds: 8 >= 8`, identity 6 of 6 (`named_and_NOT_charged` empty) | **MET** -- see §3 |
| 3b | **The same item read by (B)** -- every non-exempt region CHARGED | 8 of 16; six REACHABLE-UNCHARGED (amygdala, brainstem, cerebellum, claustrum, epithalamus, hippocampus), two UNREACHABLE (olfactory, pons) | **NOT MET** -- see §3 |

**This certification is a GATE BY EXISTENCE, not a pass:** T072-T075 and T083 are unblocked because
every one of T071's three named items has been READ and recorded here with its measurement, and two
of the three read NEGATIVE -- exactly as US2 exited with its headline criterion UNMET and T068, gated
on it, ran.

## 1. Item 1 -- prior separation: NOT MET, inherited from T063 and not re-run

**Nothing in this section was measured for this file.** Every number is quoted from
`tools/t063_campaign.json` (the pin) or from `evidence/t063-prior-drift-harness.md` (the record).
The pin was READ, never regenerated: `--campaign` writes `tools/t063_campaign.json`
unconditionally, and the pin is the record of the learner's run.

    PYTHONUTF8=1 .venv-lake/Scripts/python.exe -c "import json; d=json.load(open('tools/t063_campaign.json')); ..."

Printed, verbatim:

    git_head f37fe90234ca7e76fa3dbc2a0800ea496843100a | K 8 | D 4 | W_frac 1.0 | units 14215 | edges 70001
    sc006 NOT MET | wrong_valid True
    contrast planted - random {'fpr': 0.05078125, 't': 9.260129588726068}
    contrast random - wrong {'fpr': 0.0390625, 't': -2.545875386086578}
    contrast seeded - frozen {'fpr': 0.04296875, 't': -13.015375522819156}
    contrast seeded - random {'fpr': 0.078125, 't': -0.22771001702132443}
    contrast seeded - wrong {'fpr': 0.03125, 't': -5.227100596426406}
    learner {'disposition': 'DELETED', 'fpr': 0.04296875, 't': -13.015375522819156}

and the conjunct states the same reader printed:

    a -> {'state': 'MET'}    (SEEDED effective change > 0 and both distances > 0 on every seed;
                              red 1 fired on the FROZEN pin)
    b -> {'state': 'NOT MET', 't': -0.22771001702132443, 'fpr': 0.078125}   'seeded beats random'
    c -> {'state': 'MET',     't':  9.260129588726068,  'fpr': 0.05078125}  'the harness can see an advantage'
    d -> {'state': 'NOT MET', 't': -2.545875386086578,  'fpr': 0.0390625}
         'underperformance NOT ESTABLISHED -- red 2 is NOT FIRED and nothing is attributed to the prior'
    e -> {'state': 'NOT MET', 't': -5.227100596426406,  'fpr': 0.03125}     'seeded beats wrong'

**The exit reading, by T063 2.6's rule:** (b), (d) and (e) are the conjuncts that define prior
separation and all three are NOT MET; (c) at +9.2601 is what licenses reading them at all. **PRIOR
SEPARATION IS NOT MET, MEASURED** (`t063` §3.3: *"PRIOR SEPARATION (T071's item, as 2.6 defined it)
is NOT MET"*).

**What this exit may NOT conclude from it, restated because the table above would otherwise be read
as more than it is.** Not "H01 contributes nothing": the null retains H01's degrees, signs and
weight magnitudes (T063 2.2), so the reading is restricted to wiring information BEYOND those
marginals, on ONE task (K = 8 drive classes decoded from a 4-dimensional projection), at ONE width
(14,215 units / 70,001 edges), at the one capacity the guard selected.

**Carried beside it, from the record, because the item's warrant has named openings:**

* the LEARNER was DELETED on its own pre-registered criterion, t(d_f) = -13.0154, the frozen twin
  ahead on 8 of 8 seeds (`t063` §3.3, and §3.5 item 16 on the P8 clause cited for it);
* the SURVIVING no-learner harness is **BLIND on this population at this capacity**: `t063` §3.4's
  appended paragraph reads (c) NOT MET at t = +0.8143 on `data/h01/h01_c3_graph.npz` at K = 8,
  D = 4, W = 1.0 over seeds 0-7, so `verdict()` prints SC-006 = **NO VERDICT** -- *"no verdict is
  read: the harness is BLIND"*. **(b)(c)(d)(e) CANNOT be re-read on H01 with this instrument as-is;
  re-reading them needs a NEW pre-registration of the capacity guard.**
* (b) was read between two arms **0.50 and 0.62 held-out trials above chance** on a mean gap of one
  eighth of one trial (`t063` §3.7's per-arm table: SEEDED 0.1875, RANDOM 0.2031, WRONG 0.3594,
  PLANTED 0.6406, FROZEN 0.8750 against chance 0.1250) -- and FROZEN **is the SEEDED table at
  lr = 0**, so the wiring the verdict scored at 0.1875 decodes at 0.8750 in the same pin;
* the WRONG arm is **not degree-matched** to RANDOM (`t063` §3.5 item 12): RANDOM preserves every
  in-degree (14,215/14,215), the band-permuted arm leaves 3,701/14,215 unchanged. A disclosure gap
  and an interpretive confound, and the record states SC-006's verdict is unaffected because it
  rests on SEEDED-vs-RANDOM, where both arms preserve in-degree exactly.

**A reading the pre-registration did not anticipate, recorded as a FINDING (§0 not edited).** The
pin holds **no `arms_vs_chance` block** -- the reader above printed `arms_vs_chance present: False`.
That field was added to `verdict()` by T063's own fold (`82eafb1`) AFTER the campaign was pinned,
and the pin is deliberately byte-untouched. So every per-arm-versus-chance number this section
quotes comes from `t063` §3.7's table (which states it was applied to the pin's per-seed accuracies
offline, read and not re-run), never from the pin's own fields.

## 2. Item 2 -- the paging verdict: SC-007 satisfied by its second branch, the capability a LOSS

**Nothing in this section was measured for this file.** The harness is archived under the tag
`t068-last-with-pager` and the code is deleted; the numbers are `evidence/s5-paging-null.md` §3.4's,
quoted with their section.

§3.4's conjunct table, quoted:

| R2 conjunct | required | measured | verdict |
|---|---|---|---|
| paged beats twin, both instruments | t > 1.895, p <= 0.05, both | 0.602 (t 203.8, p 0.0039) and 0.399 (t 229.8, p 0.0039) | **MET** |
| paged beats neutered (2.10) | same test, primary | 0.709 (t 107.8, p 0.0039); traced 0.488 (t 159.6, p 0.0039) | **MET** |
| eviction conjunct reachable (2.11) | the derived w realizes the regime | w = 0.2; 61 evictions per cell; the neuter discriminates | **MET** |
| secondary cost <= 3.0x | median paged / twin tick ratio | RAW 35.2x (34.3-37.6); reference-corrected 26.0x -- the most pager-favourable reading is 8.7x over the bound | **NOT MET** |
| arm F discriminates | twin fails at 2*W*, paged runs at both | the twin CONSTRUCTED and ticked at 2*W* (17.7 GB working set) -- F does not discriminate on this host | **NOT MET** (a WIN conjunct only) |

and its ruling: *"R2 is NOT a WIN ... Under 2.3, 'LOSS iff R2 is not a WIN': **the layer's verdict is
LOSS.**"* The panel's verdict on the null (§4) is that the LOSS **stands**: 37 findings filed, **31
stood** (1 BLOCKER, 12 MAJOR, 18 MINOR), 6 refuted, and the pre-registration-honesty lens found no
relabelling; Codex's two independent refutations were answered (the first refuted at runtime, the
second folded into §3.7 item 1 and the tag).

**SC-007's own words settle the item:** *"or the layer is deleted and the deletion recorded (both
outcomes satisfy this criterion; silence satisfies nothing)"* (spec.md `:178`). The deletion was
executed as the fold's stacked commits, one logical change each, re-homing before removal.
Existence-checked for this file:

    git log --format=%h -1 <rev>

    f37fe90                  f37fe90        (the T063 campaign head, item 1)
    1d4602c                  1d4602c        (T062's record)
    343cd48                  343cd48        (lake_compactor + its 34 tests deleted)
    229cdf5                  229cdf5        (neuron_pager + its 27 tests deleted)
    327c184                  327c184        (unit_blocks retired, its one capability ported FIRST)
    5b3fa46                  5b3fa46        (KernelTick's residency-mask branch + the paged refusal)
    0f48588                  0f48588        (the block-tier schema; immortal tags 10 -> 9)
    22db40a                  22db40a        (the documentation census, 395 hits in 37 files)
    fd1359e                  fd1359e        (the T068 record commit)
    6268a39                  6268a39        (the T068 closing record commit)
    t068-last-with-pager     36b2b6a        (the last head carrying the layer)

**What the LOSS is and is NOT.** §3.4: it IS the pre-registered finding that THIS pager, in the
regime it exists for, buys a 40% / 60% footprint saving at a 35x tick cost. It is **NOT** a finding
that paging cannot work -- with page-ins and evictions ablated to zero the residency-mask branch's
own tick is still 2.5-3.4x the twin's, so the cost is this implementation's, and what a successor
inherits is the decomposition and the 3.38x OS-paged comparison of §3.3.

**Carried beside it -- what survived the deletion and what it leaves owed:**

* **T064** is VOID with the pager. What survives is the 0x09 synapse-delta path in
  `cortex/interaction_learning.py`, whose block framing was PORTED to
  `lake_engine.SynapseDelta.group_by_block` BEFORE `unit_blocks` was deleted (P2's order: port the
  surviving capability first; `327c184`). The row stays open until a successor paging row.
* **T065** is PARTIAL. Its slope was found by its own panel to measure the STREAM'S LABEL
  CARDINALITY, not the substrate: `evidence/s2-state-per-unit.md` §6.1, *"slope = 271 + 8*m, where m
  is the number of distinct labels the stream has shown"*, with m moving 3 -> 6 across the
  conditions the row averaged over. So the derived binding width is a projection conditional on a
  stated label cardinality, with the debts §6.7 names still open: the same regression over PERSISTED
  bytes (§6.3: the measurement is an in-memory `growth_state_bytes()` snapshot), m held fixed or
  entered as a term, more than one seed, and a width range that does not extrapolate five orders of
  magnitude past an 11-unit top end (§6.4). T062's record measured the related "~5.0 MB at 16k"
  projection **16x high** for the state array (301,027 B at 14,215 units).

## 3. Item 3 -- the region charge gate: (A) MET, (B) NOT MET, both rows carried

**The one fresh measurement of a criterion in this certification.**

    PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.region_charge_gate --assert
    rc=0

Output, verbatim:

    region charge: 8 of 16 CHARGED (10 evidence records; 0 exempt, ceiling 0)
      amygdala         REACHABLE-UNCHARGED
      basal_ganglia    CHARGED
      brainstem        REACHABLE-UNCHARGED
      cerebellum       REACHABLE-UNCHARGED
      claustrum        REACHABLE-UNCHARGED
      cortex           CHARGED
      epithalamus      REACHABLE-UNCHARGED
      hippocampus      REACHABLE-UNCHARGED
      hypothalamus     CHARGED
      insula           CHARGED
      medulla          CHARGED
      midbrain         CHARGED
      neuromodulation  CHARGED
      olfactory        UNREACHABLE
      pons             UNREACHABLE
      thalamus         CHARGED
      floor holds: 8 >= 8

**This reproduces §0.1's written expectation exactly** -- 8 of 16 CHARGED, 0 exempt with ceiling 0,
`floor holds: 8 >= 8`, the same six REACHABLE-UNCHARGED regions and the same two UNREACHABLE. There
is no difference to record as a finding.

The floor pin, read for this file:

    PYTHONUTF8=1 .venv-lake/Scripts/python.exe -c "import json; d=json.load(open('tools/region_charge_floor.json')); ..."

    floor 8
    charged_at_recording ['basal_ganglia', 'cortex', 'hypothalamus', 'insula', 'medulla', 'midbrain', 'neuromodulation', 'thalamus']
    fr_015_named_minimum ['neuromodulation', 'medulla', 'hypothalamus', 'midbrain', 'thalamus', 'insula']
    named_and_charged ['neuromodulation', 'medulla', 'hypothalamus', 'midbrain', 'thalamus', 'insula']
    named_and_NOT_charged []
    charged_but_not_named ['basal_ganglia', 'cortex']

**Reading (A) -- MET.** `charged >= floor` (8 >= 8, the gate's own assert, rc 0); `exempt <= ceiling`
(0 <= 0); and the IDENTITY FR-015 pins is intact -- all six named regions charged,
`named_and_NOT_charged` empty, identity 6 of 6. Two regions are charged and not named
(basal_ganglia, cortex), which the pin records and which raises the integer above the named six
rather than substituting for them; that substitution is the exact failure the `identity_gap` block
exists to make visible, and at this head it is not occurring.

**Reading (B) -- NOT MET, and the uncharged regions are named rather than counted.** Eight of the
sixteen non-exempt regions are NOT charged: six REACHABLE-UNCHARGED (amygdala, brainstem,
cerebellum, claustrum, epithalamus, hippocampus) and two UNREACHABLE (olfactory, pons).
**Reachability is not charge** (P3), so the six are not nearer to charged than the two in any sense
this gate measures -- they differ only in whether a live path imports them.

**The interpretive choice is §0.1's and is not re-argued here.** This file reads the item by (A) on
FR-015's own words -- *"The S0 exit is an integer floor of regions charged whose IDENTITY is pinned
now"* -- and reports (B) beside it. The panel's pre-registration-honesty lens is asked first whether
choosing (A) is a relabelling; if it rules so, the item is recorded NOT MET under (B) and §0.1
stands as the record of the choice. **Neither reading is argued for in this section, by rule.**

## 4. What the tree says is live and what is a fixture (the dormancy census)

    PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.dormancy_gate --assert
    rc=0

    brain modules 288  reachable 234  exempt 54  dormant 0
    STANDING TODO -- 6 non-permanent exemption(s) still open (module-gate arm, US1 audit B3; report-only):
        h01_ingest (wire)
        kernel_state (wire)
        kernel_tick (wire)
        lake_index (wire)
        microcircuit (wire)
        mossy_relay (wire)
    dormancy gate OK: every brain module is reachable from a live path or explicitly exempt

The US3 modules' own rows, read out of `dormancy_gate.EXEMPT` by a `python -c` reader run for this
file:

    h01_ingest           -> wire
    prior_drift_harness  -> permanent
    lake_compactor       -> NOT IN EXEMPT
    neuron_pager         -> NOT IN EXEMPT
    unit_blocks          -> NOT IN EXEMPT

`h01_ingest`'s row, quoted from its own appended succession (`371d607`, T063's disposition):
*"THE CONSUMER IS UNCHANGED AND THE ROLE IS NOT: `cortex/prior_drift_harness` still reaches this
module, now as the harness's FIXTURE -- the population an instrument constructs and scores -- and no
longer as a prior anything is seeded from."* `prior_drift_harness`'s row states its own role: *"A
CI/run_selfchecks MEASUREMENT INSTRUMENT on a SCRATCH population ... never constructed BY a brain
... Reachable from run_selfchecks ..., never from a tick."*

**So the exit names it plainly: NEITHER US3 module is on a live tick.** The H01 population is a
fixture an instrument constructs and scores; the harness is an instrument; and the
`kernel_tick` / `kernel_state` `wire` rows above are S6's, re-dated to the first tick population at
or above N* -- which today only growth can produce (tasks.md T071's panel-fold note, C11).

**Two readings the pre-registration did not anticipate, recorded as FINDINGS (§0 not edited).**

1. §0.3 listed `lake_compactor` among the rows to quote. **There is no row: the module is gone.** It
   was deleted with the pager at `343cd48` together with its 34 tests and its REGIONS/CORE/EXEMPT
   rows, and the reader above returns `NOT IN EXEMPT` for it exactly as it does for `neuron_pager`
   and `unit_blocks`. A deleted module has no dormancy row, and that absence is the correct state,
   not a gap.
2. §0.3 described this command as naming "what is live and what is a fixture". The measured answer
   is stronger and is written as measured: **both** US3 modules are EXEMPT (one `wire`, one
   `permanent`) and NEITHER is reachable from a live path. `lake_index` survives the deletion with
   its own `wire` row (the segment tier was re-homed, not deleted).

## What this certification does NOT cover (the misses, named)

Drafted at §0.4 BEFORE the commands and reproduced here with the commands' additions; nothing
drafted has been removed.

1. **Prior separation** (§1) -- NOT MET, and not re-readable with the surviving instrument at D = 4
   (BLIND, t +0.8143 against a 1.895 bar) or with any learner (none exists in the tree). A successor
   owes a NEW pre-registration of the capacity guard AND a learning signal a local rule can use.
2. **Paging** (§2) -- a LOSS; residency beyond RAM is UNOWNED at this head; T064's row is void until
   a successor paging row, and T065's derived width carries the debts §6.7 names.
3. **Region charge** (§3) -- 8 of 16 charged; six reachable-uncharged, two unreachable. The floor is
   a RATCHET and a ratchet asserts the structure did not get worse, never that the policy is
   satisfied.
4. **S6** -- the kernel wire DECLINED at the deployed width by Gate Zero; the `kernel_tick` /
   `kernel_state` exemptions re-arm at the first tick population at or above N*.
5. Nothing here is a claim about the LIVE brain's capability (`capability_bar`: no row earned).
6. **ADDED by the runs.** No number in §1 or §2 was produced at this head: both items are inherited
   readings from pins and records whose apparatus is archived (`t068-last-with-pager`) or must not
   be re-run (`--campaign` overwrites its own pin). Only item 3 was measured fresh here, and its
   verdict is a floor's, not a coverage's.
7. **ADDED by the runs.** The per-arm-versus-chance figures behind §1's warrant are **not in the
   pin** (`arms_vs_chance present: False`); they are `t063` §3.7's offline reading of the pin's
   per-seed accuracies. Anyone re-deriving them must read the record, not the artifact.
8. **ADDED by the runs.** T070 is OPEN and is deliberately not a gate for this exit (O5(b)); the
   live-integration instrument it owns has therefore never been read against these three items.
