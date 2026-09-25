<!--
Sync Impact Report
==================
Version change: 1.1.0 -> 1.2.0 (2026-09-25)
Rationale: MINOR. Two principles ADDED (P4c, missing since 1.1.0; P4d, the owner's 2026-09-25 engine
directive), P0 AMENDED (growth is problem-driven under P4d), P6's restated gate status turned into
a pointer, and the Governance cap on provisional principles AMENDED. Nothing was removed or
redefined, so this is not a MAJOR. As before, no principle is invented here: each addition
restates docs/POLICIES.md, which remains the authority.

Principles added:
  + P4c Every build is part of the brain   (owner directive, 2026-08-04)
  + P4d The engine                         (owner directive, 2026-09-25; PROVISIONAL, recorded as a
                                            disclosed session call; source: specs/022-unified-brain-arc/
                                            evidence/engine-structure-2026-09-25.md)

Principles amended:
  ~ P0  growth is problem-driven (P4d)
  ~ P6  the restated gate status ("manual, NON-asserting and not in CI", with 2026-08-04 counts) was
        stale against POLICIES.md; it is now a pointer, as P7 and P8 already were

Sections amended:
  ~ Governance   the P4-family and P6 enforcement rows become pointers ("P4b is NOT mechanized" and
                 "P6 manual, NON-asserting, not in CI" had both gone stale); the provisional cap now
                 counts owner directives adopted with a specified gate separately (the audit offered
                 this or dropping P7/P8's PROVISIONAL marker; a disclosed session call); the Constitution
                 Check names the P4 sub-clauses

Synced in the same arc: docs/POLICIES.md (P4d and its pointers), docs/AGENT_COLD_START.md (its design
paragraph and binding sections), AGENTS.md (the entry count, recounted: thirteen).

Templates: .specify/templates/plan-template.md OK (its Constitution Check points to docs/POLICIES.md
and the manifest and restates no principle); spec-template.md, tasks-template.md and the speckit-*
skills OK (no principle enumeration; checked by grep, whose positive control found "constitution" in
ten skills).

Deferred TODOs: none. RATIFICATION_DATE unchanged (2026-07-24); LAST_AMENDED_DATE 2026-09-25.

Previous report (1.0.0 -> 1.1.0, 2026-08-04), kept for the record
==================================================================
Version change: 1.0.0 -> 1.1.0
Rationale: MINOR. Three principles ADDED (P4b, P7, P8), three AMENDED (P0, P1, P5), and the
Governance compliance clause re-scoped from P0-P6 to P0-P8. Nothing was removed or redefined,
so this is not a MAJOR. As in 1.0.0, no principle is invented here: every addition restates an
owner directive already carried in docs/POLICIES.md, which remains the authority.

Principles added:
  + P4b The durable lake is part of the brain     (owner directive, 2026-08-04)
  + P7  Any dependency that helps is accepted     (owner policy, 2026-08-04; PROVISIONAL)
  + P8  Invent the method                         (owner policy, 2026-08-04; PROVISIONAL)

Principles amended:
  ~ P0  avenue list is a BUILD LIST, not a status ledger; honesty guards gain
        "a measurement that cannot fail is not a measurement"
  ~ P1  three documented exceptions -> FOUR (the fourth is the safety envelope)
  ~ P5  adds the 2026-08-03 target-scale re-measurement, which INVERTS the practical
        consequence at the design population; hard constraint restated as GPU-INDEPENDENCE
        of the suite rather than as one authoritative environment

Sections amended:
  ~ Governance   Constitution Check re-scoped P0-P6 -> P0-P8; enforcement surface now records
                 that P3 has THREE gates and that P6/P7/P8 are all provisional

Removed sections: none.

Templates requiring updates (the three 1.0.0 TODOs are hereby RESOLVED):
  - .specify/templates/plan-template.md      OK - no change needed. Line 43 reads
                                             "[Gates determined based on constitution file]",
                                             i.e. it cites this file generically rather than
                                             enumerating P0-P6, so it did not go stale.
  - .specify/templates/spec-template.md      OK - no change needed (no principle enumeration).
  - .specify/templates/tasks-template.md     OK - no change needed (no principle enumeration).
  - docs/POLICIES.md                         unchanged and REMAINS AUTHORITATIVE (see Governance)
  - CLAUDE.md                                already updated to P0-P8 on 2026-08-04

Deferred TODOs: none. RATIFICATION_DATE unchanged (2026-07-24, first commit of docs/POLICIES.md);
LAST_AMENDED_DATE 2026-08-04.

KNOWN DEBT recorded here rather than hidden: P6, P7 and P8 are all provisional with their gates
unbuilt. docs/POLICIES.md states that three is the most this file can carry before "provisional"
stops meaning anything, and that P8's gate is the cheapest to build and the first that should
stop being provisional.
-->

# Spiking Neural Data Lake Constitution

This constitution restates the policies in `docs/POLICIES.md`. That file remains the
authoritative, longer-form text with worked cases and measurements. Where the two ever
disagree, `docs/POLICIES.md` wins and this file is the defect.

## Core Principles

### P0. Autonomous self-improvement, by every available avenue (PURPOSE)

The brain MUST improve itself autonomously: self-writing code, knowledge acquisition, and neural
growth. Safety is what makes this pursuable, not what limits it — the self-build is
sandbox-verified and byte-exact rolled back, and the brain may rewrite what it thinks, never what
stops it.

**The avenue list is a BUILD LIST, not a status ledger** (owner directive, 2026-08-04). Every
avenue named in `docs/POLICIES.md`, current AND future, WILL be built. An avenue that does not
fire yet is open P0 work; it is never a settled state, never a boundary to respect, and never a
reason to narrow this principle to whatever already works. The list is expected to GROW: a newly
identified channel of self-improvement joins it and is built.

The honesty guards are the operative half and are NON-NEGOTIABLE:
- Growth is NOT improvement. A grown unit that changes no outcome is dormant.
- A grown unit that is not KEPT is not even growth.
- A self-written duplicate is bloat.
- **A capability reported without measurement is fake.**
- **A measurement that cannot fail is not a measurement.** (Added 2026-08-04 with P8. This is
  the harder half: the guard above is SATISFIED by a probe whose success criterion has no
  reachable red, while its entire purpose is defeated.)
- The census of what the brain has is NOT a measure of what it can do.
- Scaffolding a not-yet-measured capability is PERMITTED, on three conditions: label it as
  infrastructure, claim no band or avenue until measured, and name the gap at the site.

**Growth is problem-driven (P4d, owner directive 2026-09-25).** The brain grows while a problem is
certified unsolved, pauses at a growth budget, and prunes and saves after a solve; the honesty
guards above bind every part of it.

### P1. Brain-native / no-hardcode

Behaviour-arbitrating values MUST be learned weights or running-baseline-relative forms. A magic
constant MUST NOT be kept fixed where a computed value belongs. A constant sitting where a
computed value belongs is this project's signature defect; a value that MOVES but arbitrates
nothing is its mirror image, and movement MUST NOT be read as life.

**FOUR** documented exceptions live in `docs/POLICIES.md`: (a) a fixed point on a NORMALIZED
output; (b) a running-baseline replacement not reproducible from the persisted policy; (c) an
INERT constant (default-off, zero live callers); and (d) **raw physical limits inside the safety
envelope, which stay fixed BECAUSE they are safety constants** — a reflex threshold that drifts
with a running baseline is a reflex that can be trained away.

### P2. Adopt-better or merge, then retire the inferior (NON-ADDITIVE)

When two mechanisms do the same job, exactly ONE survives. Either ADOPT (the better becomes the
path, delete the other) or MERGE (build the one mechanism superseding both, retire both). A
"merge" that keeps both paths behind a flag is additive with extra steps. A/B before deleting,
and port a retired mechanism's unique capability BEFORE deleting it. "Better" requires a
benchmark that exists.

P7 removes exactly one argument from this evaluation — "it would be another dependency" — and
removes nothing else. A framework proposed as a replacement for an incumbent is still a P2 event
held to P2's evidence bar.

### P3. No dormancy

Every module MUST be wired-and-charged, retired, or EXEMPT with a reason. Dormancy is per-METHOD,
not per-module: if a live module computes a learning signal, the rule consuming it MUST also be
called from a live path.

**Mechanized by THREE gates**: `substrate/dormancy_gate.py` (per-MODULE, CORE),
`substrate/method_dormancy_gate.py` (per-METHOD, CORE), and `substrate/import_manifest.py`
(cross-tier — the governed tiers reach into ruff-excluded `reasoning/` and `training/` through
bare `except Exception:` swallows that hide a deletion).

Honesty guard: **reachability is NOT charge.** Import proves import. Only a neuter-A/B proves a
wire carries behaviour, and a consumer MUST NOT be faked to satisfy a gate. Equally: do NOT
force-wire. A wire that does not charge is reverted and retired or exempted, never faked.
`docs/POLICIES.md` catalogues SEVEN observed shapes of live-but-inert; the transferable rule is
**presence is not participation**.

### P4. One brain / no replication

Use the canonical `Brain -> LiveBrain -> AutonomousArm` composition; never re-implement it.
Construct only via `cortex.brain.make_brain()`. Mechanized by `substrate/replication_guard.py`.

### P4b. The durable lake is part of the brain

Owner directive, 2026-08-04. "One brain" and "one lake" are the same rule seen from two sides. A
build that keeps its own memory somewhere else has replicated the brain's memory as surely as a
second `Brain` class would replicate its cognition — and it is harder to see, because a private
store still works right up until the process exits.

- What the brain learns goes **IN** the lake — `brain_lake.bin`, under its own record tag. Not a
  sidecar file, not a JSON blob, not an in-memory structure that survives only as long as the
  process.
- What the brain recalls comes **FROM** the lake. An in-memory index is a CACHE, never the source
  of truth; a miss in working memory means "not resident", not "never happened".
- **A restart is not amnesia.** A brain constructed over an existing lake starts with what that
  lake holds (restore gated on `persist` so the deterministic gates stay reproducible).
- Foreign records are preserved, never dropped: each tenant reads only its own tag and passes the
  rest through.

Honesty guard: **"it persists" is a claim about a NEW process**, not about the object in front of
you. Prove it by reading the lake back from disk after the writer is gone. The motivating
measurement: `hippocampus/episode_consolidation` stored 59 episodes over 600 ticks with
`persist=True` and added **0** records to the lake, while `retained()` reported **1.000**
throughout — perfectly recalling episodes it was about to lose entirely.

### P4c. Every build is part of the brain

Owner directive, 2026-08-04. Every current and future build in this tree MUST be part of the brain
or retired: there is no "outside the brain", the directory a build sits in is a fact about the
filesystem, the daemon is the brain's drive and the cockpit its sensorimotor surface. The rule, its
retention rule, its amendment (a build that has not reached `main` is not a build the project can
claim) and its one named exception (`CLAUDE.md`) live in `docs/POLICIES.md` P4c.

### P4d. The engine (PROVISIONAL)

Owner directive, 2026-09-25. The brain is one of TWO networks in one organism: it grows new neurons
as column-sized batches on an anatomical map to solve a problem, feeds them into a separate liquid
reservoir into which it may transport weights, tunes readout neurons that return to it, pauses at a
growth budget, and prunes and saves after a solve, with every network's state in the one lake. Its
conditions (the e-prop invariant unchanged, P2 inside each job, "the engine can work" as a P8
criterion) and its gate specification live in `docs/POLICIES.md` P4d, and it stays PROVISIONAL
until that gate is built and counted.

### P5. Use the GPU for everything it actually speeds up

"Actually" is decided by MEASUREMENT, not expectation, and the measurement depends on WIDTH.

At today's live widths the answer is unchanged: the measured eager-CUDA crossover is
N≈24,576–32,768 (≈4,096 with CUDA graphs) while live populations are 24–240, so the `train`/`gpu`
tiers belong on the card and the live tick does not.

**Re-measured at TARGET scale 2026-08-03, and the practical consequence INVERTS.** The design
population is far larger than the current one. At N=1,000,000 CUDA is 3.0x torch-CPU and **54x
numpy**; at N=4,000,000 it is 12.2x. And sparse connectivity is not an optimisation but the only
representation that fits — a dense 1M x 1M weight matrix is **4000 GB against 12.8 GB of VRAM**.
Re-measure as populations grow; the crossover is a number, not a doctrine.

**HARD CONSTRAINT — the suite MUST remain runnable WITHOUT a GPU**, because CI has no card.
Determinism comes from seeding, not from the absence of hardware. A GPU path MUST NOT become the
only way to verify the project.

**AMENDED 2026-08-05: there is now ONE environment, not two.** `.venv-cu128` is RETIRED and
`.venv-lake` carries CUDA itself — `python 3.14.6 free-threading` / `GIL False` /
`torch 2.11.0+cu128` / `cuda True` / RTX 5070. The previous sentence here said the two envs "are
verified equivalent, so either may be used locally"; that is now FALSE in a way worth stating,
because it was false for a period before it was noticed. Free-threading became REQUIRED
(`requires-python = ">=3.14"` plus a runtime guard refusing GIL builds), which made the 3.11-based
`.venv-cu128` both uninstallable and unimportable — two envs that had been equivalent became one
working env and one dead one, with the GPU briefly unreachable.

The hard constraint above is UNCHANGED by that merge and is the reason this is safe: CI installs
from the default index, which serves CPU-only torch on Windows, so CI verifies on a GPU-free
free-threaded interpreter exactly as before. What the merge DOES introduce, stated rather than
discovered later: CI and the local gate now run different torch BUILDS (cpu vs cu128), so a
build-sensitive numeric assertion can be green in one and red in the other.

### P6. Every build must reach every other, and actually be used

A build satisfies P6 when it is reachable from at least one other FACULTY and at least one other
faculty consumes it in a way that changes THAT CONSUMER's output. One-way is allowed; zero-way is
not.

P6 is NOT P3. **P3 asks whether a faculty changes BEHAVIOUR; P6 asks whether it changes ANOTHER
FACULTY.** Wiring a faculty into the single action gate is the DEFECT P6 names, not a way to
satisfy it. Honesty guard: a route is not a conversation — neuter faculty A and show faculty B's
own output changes; adding a reader that stores a value is not consumption.

PROVISIONAL; mechanization status lives in `docs/POLICIES.md`'s enforcement rows (authoritative;
re-derive with `python -m snn_data_lake.substrate.policy_manifest --manifest`). The governing
finding stays: each writer holds a PRIVATE workspace instance, so adding readers is not a remedy
(POLICIES.md P6). *(Converted to a pointer 2026-09-25, 1.2.0: this paragraph restated the gate as
"manual, NON-asserting and not in CI" with 2026-08-04 counts, and POLICIES.md records it BUILT AND
ASSERTING.)*

### P7. Any dependency that helps this become a phenomenal brain is accepted

Owner policy, 2026-08-04. Any dependency or framework that helps this become a phenomenal brain
is ACCEPTED, including any that helps the brain become sentient. There is no dependency budget and
no dependency count to defend.

**"It would be another dependency" is NOT a valid reason to reject anything.** A rejection MUST
cite a measurement on the real workload or a structural argument about this architecture. Adoption
MUST ship with the measurement of what it bought — "it is a serious framework" is not evidence. A
declared-but-never-imported dependency is not a capability.

Does NOT relax (each a pre-existing invariant, restated so P7 is not read as a trapdoor): the
safety envelope stays in-tree and approval-gated; **P4b — a dependency MUST NOT become a second
durable store**; **e-prop — adopt the library, not its BPTT trainer**; P2 — adopting a library
means RETIRING what it supersedes, not running both; P3 gives no dormancy pass for being
third-party; and the suite stays runnable without a GPU.

PROVISIONAL; mechanization status lives in `docs/POLICIES.md`'s enforcement rows (authoritative -- this file restated it, was false for 13 days, and now points instead; re-derive with `python -m snn_data_lake.substrate.policy_manifest --manifest`).

### P8. Invent the method; the literature is a starting point, not a boundary

Owner policy, 2026-08-04. Claude is AUTHORIZED to synthesize across any information, data,
framework or dependency, and to **INVENT ANY METHOD** required to bring this brain to sentience.
Not restricted to published architectures, to what a framework already does, or to what
neuroscience currently describes. Where no existing method reaches the target, inventing one is
the expected move, not an overreach.

P8 is a HOW principle, not a research direction: it names no architecture and no theory. It grants
latitude and fixes a burden. **An invented method OWES:**

1. A success criterion with a **REACHABLE RED** — construct the input that makes it fail, and
   record it. If no such input exists, the criterion is **DELETED, not weakened**.
2. A control that is not the method itself: an untrained twin, a permutation or label-shuffle
   null, or a neuter-A/B.
3. Attribution by per-item neuter, not by the total moving.
4. Unprompted disclosure of contamination, overlap, or a precondition that makes a check vacuous.
5. **Deletion on a null.** That is the licence working, not failing, and it is the expected
   outcome most of the time.

**Transferable rule: an invented method's debt is proportional to how LOAD-BEARING its output is,
not to how clever it is.**

On sentience: the goal is not hedged, and it is a build target. But **there is no measurement of
sentience in this tree today** — inventing one is inside the mandate, and it is held to the burden
above: a reachable red, a control that is not itself, and a **dissociation rather than a marker**.
A passing criterion is evidence about a MECHANISM, never a verdict about EXPERIENCE; nothing here
authorizes the system to assert its own sentience. **The brain may report what it measures; it
MUST NOT report what it cannot measure.** The safety envelope is NEVER a target of invention.

PROVISIONAL; mechanization status lives in `docs/POLICIES.md`'s enforcement rows (authoritative -- this file restated it, was false for 13 days, and now points instead; re-derive with `python -m snn_data_lake.substrate.policy_manifest --manifest`).

## Standing Invariants (never relaxed by any policy)

1. **Count a gate's output before any rule leans on it.** Implement the check, run it, publish
   what it finds, and only then let a policy depend on it. The owner MAY adopt a policy with its
   gate unbuilt; such a policy lands marked PROVISIONAL and MUST still carry its quoted source
   line, its named real incidents each re-verified against the tree, and its gate specification.
2. **Never silence a check to make it pass.** No exemption that hides the class of defect the
   check exists to catch; no re-recording of a moving number; no narrowing a heuristic to dodge
   a true positive. An exemption MUST carry a disposition and a reason.
3. **The negative verdict is the dangerous one**, because it is the one that retires a working
   faculty. Where a probe cannot distinguish "did not fire" from "fired and did not matter", it
   MUST report UNKNOWN rather than a confident negative.
4. **Absence of a warning is not evidence of absence.** A check reports a floor, not a census,
   unless proven otherwise.
5. **Verify the verifier.** A green gate that structurally cannot see the thing it checks is
   worse than no gate, because it manufactures confidence.
6. **Runtime evidence outranks static reading.** Before optimising a hotspot, neuter it and
   measure the delta; before believing a signal is live, neuter it and watch what changes. A
   static "never imported" is a hypothesis, not a finding.
7. **The safety envelope is untouched by every principle above.** The five `brainstem/` files
   (`reflex.py`, `serial_arm.py`, `so101_arm.py`, `arm_config.py`, `interpreter.py`) are modified
   only with explicit approval; the reflex E-STOP is never relaxed; and neither P7 nor P8
   authorizes a dependency or an invented method inside it.
8. **E-prop, not backprop.** Any backprop consideration pivots to e-prop, verified cosine > 0.5
   against an independent BPTT gradient. Absence of `.backward()` is NOT evidence of locality.
   **Scope, verdict semantics, reachability (including its one-way ratchet) and every count live in
   `docs/POLICIES.md`'s SCOPE paragraph, which this invariant deliberately restates NONE of** — a
   restated adjudication here drifted from the authority twice in one day (2026-08-18, commit a6eb7a5f),
   because every fix applied to one surface manufactures a mismatch on the other. The bar's value is
   unchanged; that is the only figure this file asserts.

## Development Workflow and Quality Gates

Three gates MUST stay green, and a change is not done until they are:

```
python run_selfchecks.py          # every core module's __main__ self-check
pytest                            # >=70% line coverage
python -m ruff check .            # E, F, I, UP, B
```

`tools/citation_guard.py` additionally asserts in the CI lint job (repo hygiene, not a policy):
no tracked file may cite an untracked `.py` as a runnable reproduction path.

Evidence discipline for the coverage ledger (`docs/coverage/`):
- Every score cites `file:line`; a score without a citation is not a score.
- Nothing above band 40 without a runtime A/B whose recorded exit code passed.
- Nothing above band 80 without held-out evidence.
- A faculty that cannot be measured stays in the ledger as UNSCORED, counting 0. A row MUST NOT
  be deleted to raise a score.

Change discipline: one change per commit, full gate green, commit and tag locally, so a
regression is isolated to its own change.

## Governance

`docs/POLICIES.md` is the authoritative policy text; this constitution is its governance
surface and MUST NOT contradict it. Amendments to a principle here REQUIRE the corresponding
amendment in `docs/POLICIES.md`, with the measurement or worked case that motivated it.

Enforcement surface, stated plainly because it is easy to assume a gate exists:

<!-- Every row below is a SUMMARY that can lag; docs/POLICIES.md is authoritative for each
     (this file's own header: where the two disagree, this file is the defect). The e-prop, P7
     and P8 rows were pointers already; the P4-family and P6 rows became pointers on 2026-09-25
     (1.2.0), after "P4b is NOT mechanized" and "P6 HALF: manual, NON-asserting, not in CI" had
     both lagged POLICIES.md. -->
| principle | mechanized in CI | review-enforced |
|---|---|---|
| P0, P1, P2, P5 | — | yes |
| P3 | `dormancy_gate.py`, `method_dormancy_gate.py`, `import_manifest.py` | the charge tier |
| P4, P4b, P4c, P4d | see `docs/POLICIES.md`'s enforcement rows (authoritative; this table deliberately does not restate them, including the CI column) | see POLICIES (P4d is PROVISIONAL and review-enforced) |
| P6 | see `docs/POLICIES.md`'s enforcement row (authoritative; not restated here) | see POLICIES (PROVISIONAL: a ratchet asserts "no worse", never "satisfied") |
| P7, P8 | see `docs/POLICIES.md`'s enforcement rows (authoritative; this table deliberately does not restate them, including the CI column) | see POLICIES |
| e-prop not backprop | see `docs/POLICIES.md`'s enforcement row (authoritative; this table deliberately does not restate it, including the CI column) | see POLICIES |

Three provisional principles is the MAXIMUM this file should carry before "provisional" stops
meaning anything. Mechanization status and dates live in `docs/POLICIES.md`'s enforcement rows
(this file restated them once and got the DATE wrong within a day); PROVISIONAL is kept because
a ratchet asserts "no worse", never "satisfied" — superseding the stale sentence that stood
here; P8's was the cheapest and the one
with a named expected output, so it is the first that SHOULD stop being provisional.
**Amended 2026-09-25 (1.2.0; a disclosed session call, the second of the two options the audit
offered): owner directives adopted with a specified gate are counted separately.** P4d is such a
directive; the three counted under the cap stay P6, P7 and P8.

Versioning: MAJOR for a backward-incompatible principle removal or redefinition, MINOR for a new
or materially expanded principle, PATCH for clarification.

Compliance: every plan produced under Spec Kit MUST pass a Constitution Check against **P0–P8**
(P4's sub-clauses P4b, P4c and P4d included) and the Standing Invariants before implementation. Complexity MUST be justified against P2 and
the rule that the best code is the code never written. Note for plan authors: a Constitution
Check certifying "no new dependency" is **no longer evidence of anything** under P7 — a plan
should state what a dependency would BUY, or say nothing. `CLAUDE.md` and `docs/POLICIES.md` are
the runtime guidance files.

**Version**: 1.2.0 | **Ratified**: 2026-07-24 | **Last Amended**: 2026-09-25
