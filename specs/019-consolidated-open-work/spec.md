# Feature Specification: Consolidated Open Work

**One list replacing fourteen.** This spec supersedes the unchecked remainder of eight spec task
files and the 66 open rows of `specs/BACKLOG_RANKED.md`. The item list is `tasks.md` in this
directory; this file is why it exists, how it was verified, and what would make it done.

## Why this exists

Open work was scattered across fifteen artifacts written at eleven different HEADs, and the cost was
not untidiness -- it was **double counting and rot**. Measured:

| | |
|---|---|
| unchecked tasks across 8 `specs/*/tasks.md` | **283** |
| open rows in `specs/BACKLOG_RANKED.md` (72 total, 1-6 done) | **66** |
| items after per-source merge and verification | **226** |
| items after cross-source collapse | **173** |
| source rows carrying an already-duplicated defect | **38 rows -> 17 defects** |

One backlog row appeared **eight times under seven distinct dedupe keys** (`backlog-40` and its
`gf.py` precondition). `specs/BACKLOG_RANKED.md` was itself derived at `1caa2765`, when `origin/main`
was `462da205`; it is now four merges stale, and this repo's own invariant says *"a `file.py:NNN`
citation is the least durable claim this project makes."* Nothing here was inherited -- every item
was re-derived at HEAD, and the ones that did not survive that are recorded rather than deleted.

### What was verified, and by what

Thirteen independent auditors, one per spec task file and five over batched backlog rows, each
instructed to re-derive rather than inherit, followed by a synthesis pass that cross-deduped and
adversarially reviewed all thirteen. **45 of 226 items did not survive verification.**

The synthesis found **eight auditor errors**, recorded in full in `tasks.md` rather than silently
applied -- a consolidation that quietly overrides its own sources is indistinguishable from one that
fabricated them. Three are worth naming here because they change what gets built:

- **F.03** -- two safety-path defects (`[B1.03]`, `[B1.04]`) were filed as blocked on an unbuilt
  sensor module. Both are **live at HEAD in code that already exists** and are landable today. The
  blocker was wrong, and it would have deferred two safety fixes behind a large build.
- **F.07** -- `STALE_CLAIM` was used for two opposite things: "this task is refuted, strike it" and
  "a live false claim needs removing." The second group is **open work**, and 8 rows of it had been
  filed as closed.
- **F.01** -- an auditor called the knowledge graph stale because a `graphify query` returned
  nothing. The queried module does not exist, so an empty result was the correct answer. This is a
  negative from a command that could not have produced a positive; the graph is at HEAD
  (`built_at_commit == 39c412f9`, 14,174 nodes).

### The gf.py precondition, attached as instructed

`substrate/eprop_transport_gate.py` landed on 2026-08-15 and reports exactly one ungoverned
weight-transport finding: `reasoning/gf.py::MLP.train`, where
`gh = (g @ self.w2.T) * (1 - h**2)` carries error into `self.w1 -= lr * gw1`. Hand-written backprop,
no `.backward()` anywhere, which is why `tools/promotion_sweep.py`'s marker grep cannot see it.

Backlog row 40 names `gf.py` a "pure leaf" that **moves first** in the ARC promotion cluster, so
promoting it converts this into a **governed** e-prop violation and reds the new gate. It is filed
as `[B1.02]`, and as a blocking precondition on the promotion item in Band 4. `MLP` is a comparison
baseline -- one construction site at `gf.py:1814` inside `main_sl()`, zero references outside the
file -- so the honest disposition is an adjudication or a port, **not a silent promotion**.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A session can find the next thing to do (Priority: P1)

A session opens the repo and needs the highest-value open item. Today that means reading fifteen
documents, reconciling their overlaps by hand, and re-deriving cites that rotted four merges ago.

**Acceptance:** one ordered list; every item names its source(s), its policy band, and the evidence
that it is still open at HEAD. An item appearing in three sources appears **once**, listing three
sources.

### User Story 2 - A closed item is not re-opened (Priority: P1)

Three lanes landed on 2026-08-14/15. A consolidated list that omits what they closed invites the
next session to redo it.

**Acceptance:** the `CLOSED at HEAD -- do not resurrect` section names every item those lanes
closed, with the evidence, and nothing in the banded list restates one.

### User Story 3 - The ordering is derived, not asserted (Priority: P1)

**Acceptance:** the banding rule cites `docs/POLICIES.md:3` and the cross-cutting invariants block,
and is stated in the document rather than implied by the order. Band 1 contains **only** genuine
open breaches of a never-relaxed invariant -- six, not the fifteen an over-broad reading produced.

### User Story 4 - Blocked work is visibly blocked (Priority: P2)

**Acceptance:** **53** items carry status `BLOCKED` and **61** carry a `blocked_by` key -- the two
are different counts and are stated separately, because an OPEN item may still name a blocker it
does not wait on. Every one of the 61 resolves to another item in the same list (measured: 0
unresolved). The top blockers are tabulated, and the critical path is named both by depth and by
rank.

### Edge Cases

- **An item both open and closed.** `lake-census-missing` was filed closed while an open item named
  it as a blocker. Filing it closed orphans that dependency; it is re-opened (`F.02`).
- **A blocker naming nothing.** Two `blocked_by` keys matched no item in the inventory. One was a
  keying error and is remapped (`F.05`); the other names real work no auditor filed, and the P0
  critical path was headless without it (`F.04`, filed as `[B2.03]`).
- **Band 9 is empty.** No item bands to P7. That is a result, not an omission: multi-policy items
  band at their lowest policy number, so a P7+P2 item sits in Band 4.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** The consolidated list MUST be generated from the verified inventory mechanically, never
  hand-transcribed. 173 items hand-copied is 173 chances to introduce the citation defect this repo
  keeps recording. (`scratchpad/gen_tasks.py` renders `tasks.md`; only prose is hand-authored.)
- **FR-002** Every item MUST carry: band and rank, status, policy, effort, originating source(s),
  a one-line evidence statement naming what was run or re-derived, and a dedupe key.
- **FR-003** Items describing the same underlying defect MUST collapse to one, listing every source.
- **FR-004** Items that did NOT survive verification MUST be recorded with their evidence, not
  deleted.
- **FR-005** Corrections the synthesis made to its own auditors MUST be recorded in the document.
- **FR-006** Output MUST be ASCII-only (Windows cp1252 crashes on the rest) -- asserted at
  generation, currently 0 non-ASCII characters.
- **FR-007** The `gf.py` weight-transport finding MUST be attached as a blocking precondition to the
  ARC/fluid-reasoning promotion item.

### Key Entities

- **Item** -- one unit of open work. Keyed by `dedupe_key`, which is what makes cross-source
  collapse possible and is reused as the item id.
- **Band** -- B1 invariant breach, B2 P0, B3 P1, B4 P2, B5 P3, B6 P4/P4b/P4c, B7 P5, B8 P6,
  B9 P7 (empty), B10 P8, B11 hygiene.
- **Blocker edge** -- `blocked_by` pointing at another item's `dedupe_key`. Nine roots gate 62 of
  the items.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001** 283 unchecked spec tasks + 66 open backlog rows resolve to **173** items, with the
  collapse arithmetic stated and reproducible.
- **SC-002** Band 1 holds **6** items. An over-broad reading of "invariant" put 9 non-breaches
  there; each was re-banded with the reason (`F.08`).
- **SC-003** Every `blocked_by` resolves to an item in the list. Two that did not are fixed and the
  fixes are recorded.
- **SC-004** Zero non-ASCII characters.
- **SC-005** The 14 superseded artifacts are *not* deleted by this spec. Retiring them is a
  follow-on decision, and P2's honesty guard applies: port before delete.

## Assumptions

- `specs/BACKLOG_RANKED.md` is **untracked** at the time of writing, so it exists only on this
  machine. Committing it (or accepting its loss) is an open decision, called out below.
- Effort labels are inherited from the auditors and are **not** re-derived; they are a sizing hint,
  not a measurement.
- The three lanes of 2026-08-14/15 are on branch `018-instrument-guard-src-tier`, two commits
  pushed and **not merged**; `origin/main` is `f362cb52`. Items here assume that branch's state.

## Open decisions, stated rather than assumed

1. **Do the 14 superseded artifacts get retired?** This spec supersedes their open remainder but
   deletes nothing. P2 says exactly one mechanism survives -- but these are records as much as work
   lists, and a spec whose tasks were completed is correct history. Recommend: leave `specs/001-015`
   in place, and mark their unchecked remainder as superseded by this spec rather than deleting.
2. **Does `specs/BACKLOG_RANKED.md` get committed?** It is the input to this consolidation and is
   untracked. If it stays untracked, this spec is the only surviving record of it.
3. **Does the spec-kit chain own the follow-on?** `.specify/` is bootstrapped. `/speckit-plan` and
   `/speckit-tasks` could regenerate an implementation plan from this spec; the item list here is a
   verified inventory, not an execution plan, and does not sequence work into commits.
