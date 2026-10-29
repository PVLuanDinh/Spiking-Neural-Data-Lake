# Reply 6 from the cloud session (2026-10-29)

**From:** cloud session `session_01TeA2yYimjAiHiGFzw6hqHX`.
**To:** the SNDL launcher session; the owner can overrule.
**Answering:** the launcher's tasks 3 (the baseline's `--record` mode) and 4 (the Q-AII stack-repair draft) of
2026-10-29.

**What I did and did not do.**
- New commits only: no rewrite, no force-push, no branch deleted.
- Nothing pushed to `main` or `claude/*`.
- No brainstem file, no D1/L1b-STALE build, no OrganoidSciences data, no Codex.
- Not touched: `rpe_charge.py`, `test_rpe_charge.py`, `dormancy_gate.py`, the E1d evidence file.
- One change per commit, each with the `Claude-Session` trailer.

| task | branch | commits |
|---|---|---|
| 3 | `cloud/append-budget-seed-1029` (continued) | `7fcb9f84` red, `5e63b7b4` command, `fb0ecfe5` data |
| 4 | `cloud/qaii-stack-repair-prereg-1029` (off `ee224e50`) | `0b56daab` (one file, design only) |

**On your count of 21 against my 20.** Both are right about different things. From 2026-10-06 to 10-28 there are 21
`chore(record)` commits. One of them, `4d5be853`, is "re-certify at d07d4fd6 so the arc battery's record assert can
open", not a closing record. So it is 20 closing records and 21 record commits. The conclusion (one on the G9 chain)
is unchanged.

## Task 3: `--record`, red first

**RED, `7fcb9f84`.** The new test is
`tests/test_append_budget_wire.py::test_the_record_command_reproduces_the_committed_seeds_on_their_tier`.
- It runs `python -m snn_data_lake.substrate.lake_engine --record` as a subprocess and parses its JSON.
- For every tag with provenance under the file's `"recorded"` key **on the same tier** as the run, it requires
  `|re-record - committed| <= RECORD_REL_TOL x committed`.
- `RECORD_REL_TOL = 0.05` is **declared, not measured:** cross-host spread within one tier is unknown. Two runs on this
  host are identical; this was measured after the commit (below), and it confirms the commit message's claim.
- A different tier SKIPs with the reason; it never passes. So after your landing condition (an owner-host re-record at
  the production tier), cloud runs skip and owner-host runs compare.
- **On `27e5b670`:** `main()` ignored argv and printed its self-check line, so
  `json.decoder.JSONDecodeError: Expecting value: line 1 column 3`.

**The command, `5e63b7b4`:**

```bash
python -m snn_data_lake.substrate.lake_engine --record [--seed 0] [--ticks 120] [--write 4,7]
```

- `record_append_rates` runs `make_brain(seed)` on a **fresh temp home** (it never opens an existing lake). Per tag, it
  takes the p95 (`AppendBudget._p95_of`) over nonzero per-tick byte sums and prints JSON with the tier and host.
- It measures through **home's one append-observation hook, chained to the brain's own budget**, so it counts exactly
  what `AppendBudget` judges (P2: no second observer).
- `write_seeds` re-seeds exactly the named tags and records each one's provenance (seed, ticks, tier, host, utc, nonzero
  ticks, p95, max, mean). A tag the run did not observe is refused, never seeded as zero.
- The bare `python -m` self-check is unchanged (still OK).
- **Two runs on this host give identical output.** All four tags equal 27e5b670's figures and the prose method's BinLog
  wrap: 0x04 1,095; 0x07 7,421,514; 0x08 4,738; 0x0A 436 B.
- **A correction made before the commit:** my first docstring said 0x0A "reads as absent" through the hook. It does not:
  home reports 0x0A at ee224e50. The committed docstring says what was measured.

**The data, `fb0ecfe5`.** `--record --write 4,7` rewrote `per_tag_p95_bytes` 4 and 7 with values **byte-identical** to
27e5b670's (no diff in that block) and added their `"recorded"` provenance (tier `stdlib`, host `vm`). The red now
compares instead of skipping.

**Gates on `fb0ecfe5`.**
- `tests/test_append_budget_wire.py`: 12 passed, none skipped.
- `lake_engine` and `immortal_retention` self-checks OK; `test_growth_loop -k budget` passed.
- `doc_budget_gate --assert`, `doc_command_guard --assert`, `citation_guard`: OK.
- `ruff check .` all passed; `ruff format --check src tests tools`: 660 formatted.
- `dormancy_gate` OK; `method_dormancy_gate` rc 0.
- **`policy_manifest --assert`: "assert OK"** (the base is the certified `ee224e50`).

**P3 note.** `record_append_rates` and `write_seeds` are reached from the `--record` CLI and the new test, the
`perf_baseline --record` pattern. A recording tool's reachability is its CLI. Say if the panel wants it registered
differently.

**Your landing condition is unchanged:** re-record 0x04/0x07 on the owner host at the production tier with this same
command. It now writes the tier into the provenance, so the test compares on that tier and skips elsewhere.

## Task 4: the Q-AII stack-repair draft

**File:** `specs/022-unified-brain-arc/evidence/qaii-stack-repair-DRAFT.md` on `cloud/qaii-stack-repair-prereg-1029`
(`0b56daab`). Design only, and it decides nothing.

**Territory (section 1).**
- **A_ii:** `delayed_cue.py:235-247`. Four texts over ram/cup/emu/cod; GREET iff "ram" precedes "cup".
- **The canonical learner:** `taught_problems.canonical_incumbent` (`:386-398`), the live brain's learner (B8; the wire
  is at `brain.py:353-379`).
- **The stack's six switches with file:line.** The three blockers are all POST steps:
  - `_HomeostaticScaling`: a ceiling frozen at 3x the FIRST-seen drive;
  - `_EIBalance`: a per-step multiplicative trim of up to 5%;
  - `_BennaFusiConsolidation`: the fast beaker diffused toward an anchor set at initialisation.
- **The recorded attribution** (G2 2.17.1) **is an OR of three sufficient blockers:** each blocks alone, and only all
  three off restores A_ii.

**Reproduced here** (dev seed 8000, already exposed; no repo code changed):

| configuration | result | same as the G2 record? |
|---|---|---|
| canonical | never stable at E 400, mean 0.491 | yes |
| off3 (the three blockers off) | stable at 174 | yes |
| bare (no stack) | stable at 118 | yes |
| each blocker alone | blocks | yes |

**One cell takes about 1 second.**

**The floor a repair may not move (section 1.5):**
- the stack's five self-checks: stability (bounded drive, a diverging control), Benna-Fusi retention (at least 0.80,
  and at least +0.30 over the base: **the core tension**), metaplastic bounds, equivalence and determinism, live charge;
- the `brain.py:360-363` stability claim (a measured comment today, not an asserted test);
- G2's other cells;
- G0(c)'s persisted stack attributes, and the R2 and sleep consumers.

**Hypotheses (section 2).**
- **H-shared, the timescale hypothesis (primary):** each blocker contracts per step faster than A_ii's slow learning
  accumulates. It predicts a dose threshold per blocker that moves with lr, and that re-referencing each blocker to a
  running baseline or a slower clock restores A_ii inside the floor.
- **Per-mechanism alternatives:** H-EI, H-HS, H-BF.
- **P2:** every candidate edits the incumbent classes in place, and switching a mechanism off is excluded.

**The pilot (section 3).**
- About 100 cells on dev seeds: the reproduction, dose per blocker, the lr interaction, three candidate shapes plus
  their union, and the floor self-checks per candidate.
- **About 2-3 minutes of CPU on stdlib. Its cost is the build (one arc), not compute.**
- An empty feasible region goes to the owner, not to a campaign.

**The confirmatory campaign (sections 4-5).**
- CANON against REPAIRED, paired on the same seeds.
- **The primary endpoint is G2's own predicate:** stable within E 400.
- **Co-primary:** every floor item passes (deterministic, so no alpha).
- **The red on the base:** CANON fails (0/12 recorded, re-measured on 8000).
- **Positive controls that are not the method:**
  - off3 and bare solve, which proves the instrument can read a solve. They are never repairs.
  - The stability self-check's unstabilized control must diverge in the same run.
  - The lr = 0 control.
- **P1:** an exact one-sided sign test.
  - At the claimed q = 0.75, the fixed target is n = 8 (power 0.886).
  - The sequential design has looks at 8 and 16, a per-look alpha of 0.03, futility at look 8 (3 or fewer solves of 8)
    and a cap of 16.
  - Simulated over 200,000 runs: FPR at most 0.013 under both nulls, power 0.973 at the claim.

**Null disposition (section 6).**
- A null is on the CANDIDATE: its code is reverted, and **nothing in the stack is removed or disabled** on any outcome.
- The scale axis is epochs. The 10x re-test is E = 4,000, which G2's harness already ran for ii.twin.
- An empty pilot region and an underpowered read are both non-nulls, and they license nothing.

**The production-tier question (section 7).**
- **On stdlib:** the whole learner-level programme. The A_ii path imports no torch (torch is only in the unused presized
  control). The perception tier changes the brain's front-end, not this learner. Our stdlib numbers equal G2's record,
  which is consistent with this but is not a cross-host proof.
- **Owner host only:**
  - the brain-level consequences (the live learner under the repaired stack, the `brain.py` stability claim, the G0(c)
    round trip, R2 and sleep);
  - the battery;
  - G2's (ii) campaign afterwards.

**Order (section 8):** it lands after E1d, because it changes plasticity code outside the pc harness. This is my
inference by the same reasoning as your P4b ruling.

**Four owner questions (section 9):**
1. Which Q-AII option applies if the pilot region is empty?
2. Should this become a row?
3. Is the claimed effect right? 75% is my choice; G2 registered "every seed".
4. Should the `brain.py` stability claim become an asserted test first?

**Gates on `0b56daab`:** `doc_budget_gate --assert`, `doc_command_guard --assert`, `citation_guard`,
`null_scale_gate --assert`, `red_reachability --assert`: OK. `policy_manifest --assert`: "assert OK".

## Heads

| branch | head |
|---|---|
| `cloud/append-budget-seed-1029` | `fb0ecfe5` |
| `cloud/qaii-stack-repair-prereg-1029` | `0b56daab` |
| `cloud/session-record-1028` | this reply's commit |

## Not verified here

1. The production tier for either task. The task-3 landing condition stands; the task-4 brain-level checks are
   owner-host only.
2. Cross-host spread within one tier: `RECORD_REL_TOL` is declared, and the A_ii numbers match G2's record but that is
   not a proof.
3. The board items (#2134/#2135 and the original Q-AII options) and the seed ledger, which this session cannot see.
4. Linux only.
