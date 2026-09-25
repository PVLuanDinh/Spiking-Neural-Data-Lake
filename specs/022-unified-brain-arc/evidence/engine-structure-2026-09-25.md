# The SNDL engine -- the owner's structure and decisions (2026-09-25)

Record: the owner's architecture of record from 2026-09-25, carried into the tree verbatim so that every amendment
following from it cites a tracked source. The adding-a-policy bar in `docs/POLICIES.md` asks for the source line;
the directive this record feeds is POLICIES P4d. Status: RECORD. No run, no pre-registration and no result. The
owner's words are quoted exactly; the session's own calls are marked as session calls.

## 1. The diagram

The owner shared `SNDL-Engine.drawio` (Google Drive, created 2026-09-25 03:52 UTC). Every label below is the owner's
text, verbatim, with the capitalization as drawn. The geometry is summarized.

Components:
- **Brain**: "Brain (split either by cortical columns or brain regions)". It sits INSIDE an ellipse labeled "DURABLE
  DATA LAKE THAT THE BRAIN LIVES INSIDE".
- **New neuron(s)**: "New neuron(s) to solve a problem", grown by the brain.
- **Liquid Reservoir**: "Liquid Reservoir (Liquid State Machine)", a SEPARATE structure drawn above the brain and
  outside the lake ellipse.
- **Readout neurons**: "Readout neurons (Contain relational properties to connect to the brain, tuned by the brain)".

Connections:
1. Brain -> New neuron(s). The brain generates new neurons to solve a problem.
2. New neuron(s) -> Liquid Reservoir: "Neurons is fed into the Reservoir".
3. Brain -> Liquid Reservoir: "WEIGHTS TRANSPORTED BY THE BRAIN".
4. Liquid Reservoir -> Readout neurons.
5. Brain -> Readout neurons: "THE BRAIN TUNES NEURONS VIA A LEARNING RULE (e.g.: STDP, Topographical Clustering,
   Predictive coding)".
6. Readout neurons -> Brain, a return path: "THE NEURONS GAIN RELATIONAL PROPERTIES AND INFORMATION TO CONNECT TO THE
   BRAIN".

Callouts, the owner's rules of operation:
- "AFTER GAINING NEW NEURONS, THE BRAIN STOP GENERATING NEW NEURONS AND ATTEMPTS TO SOLVE THE PROBLEM. IF THE PROBLEM
  DOES NOT GET SOLVED, THE BRAIN REPEATS TO GENERATE MORE NEURONS UNTIL PROBLEM IS FULLY RESOLVED."
- "AFTER PROBLEM IS SOLVED, THE BRAIN RUNS A SLEEP CYCLE TO PRUNE AND SAVE THESE NEW NEURONS AND DATA OF THE PROBLEM.
  THE BRAIN DO SO USING A METHOD (Either Simulated Annealing or something different)"
- "THE BRAIN CAN RECALL ANY INFORMATION WITHIN THE SAME DURABLE DATA LAKE ENVIRONMENT BECAUSE IT LIVES INSIDE THE SAME
  ENVIRONMENT"

## 2. The owner's words, verbatim (2026-09-25)

- The instruction: "Audit any part of the project including POLICIES.md and CLAUDE.md to follow this new structure"
- The answers: "Weights transport is allowed as long as the engine can work, the repo should be reorganized to fit
  the new structure as well. 2 networks are allowed as long as it fits the new engine structure"
- The clarification: "The brain regions should be neurons and not coded like the current work tree structure on
  github, the engine grows the neurons and attaching them over time, it could be either cortical columns or
  anatomically accurate brain, I don't know which structure to consider since we need to do more research on the
  brain design philosophy through web grab"

What the answers settle, as the session reads them (the words above are the authority, never this reading):
- Weight transport from the brain into the reservoir is PERMITTED, as a copy of weight values or as the brain's
  activity driving it, on condition that the engine works. That condition is a measurement (P8).
- Two networks, the brain and the liquid reservoir, are PERMITTED within the engine structure.
- The repository is reorganized to fit the structure: code follows the engine's components, and the brain's regions
  or columns are attributes of grown neurons (data), never a package per region.
- The five `brainstem/` safety-envelope files still move or change only with the owner's explicit approval. Nothing
  here reaches them.

What it overturns: the 2026-09-22/23 plan reading of the owner's earlier proposal, "growth INTO one recurrent
substrate" with "No weights are copied into a second structure" (P2, P4b), and readout self-feedback as a side arm
(plan Part D, G workstream, Disposition). This arc's amendments supersede that reading with dated notes; the old
text stays as the record of what was decided then.

## 3. The research and the owner's four decisions

The research came first, as the clarification asked. Four Claude Opus research agents read primary papers through
webgrab (columns, anatomy, growing networks, reservoirs with feedback and sleep). Their decision brief, "Columns or
anatomy?", is published at https://claude.ai/artifact/4S9PP8fP7PhgBU1c72Tkyn and recommended option C. Its 18
load-bearing quotes were all re-found in the saved pages, and a planted nonsense phrase was not. Its stated limits:
one brief and one model behind the four agents, several full texts read as abstracts only, and no study comparing
grown columns against grown single neurons in a spiking network at an equal budget.

The owner then answered four questions (2026-09-25). Each answer was the brief's recommended option:
1. **Organizing unit: C, "grown columns on an anatomical map".** Growth adds small column-sized batches where error
   builds up: one radial unit, about 8-10 excitatory neurons with inhibitory neurons from a shared pool (the batch
   size is an arm). Every grown neuron carries a position (x, y, depth), a layer, a Dale sign, a region label and the
   problem that grew it, in the lake. Wiring follows rules: layer target x distance kernel (a Gaussian of about
   85-114 µm locally, falling off about 4.6 per mm, to be refitted on H01) x shared-partner bonus. Regions are
   LABELS read off the map and measured, never code packages. H01 is a fitted prior that must beat a shuffled copy
   of itself. Four deciding tests, each with its red, as the brief fixed them:
   - T1, the growth-unit race: a column batch vs single neurons vs a presized network vs random attachment, on the
     same problems at the same neuron budget. Red: columns need 20% or more extra neurons at equal accuracy, or
     forget more. A presized network matching both means growth itself earns nothing.
   - T2, H01 vs shuffled H01 (and a generic template). Red: H01 does not beat shuffled H01, and the map part of C is
     then deleted.
   - T3, the H01 kernel fit: Gaussian, exponential, both or flat, scored on held-out neurons, after a planted
     Gaussian (sigma = 100 µm) is recovered within 10%. Red: the Gaussian loses to the exponential or the flat model.
   - T4, emergent vs imposed organization: rule-grown wiring vs imposed columns vs random attachment, scored on
     modularity and task score. Red: modularity no higher than random attachment, or a task score below imposed
     columns.
2. **G1(b): "Spiking only can win".** All four candidates run as built; only g1 (LSM) and g2 (ESN-LSM) can be
   selected, and SORN (g3) and RBN (g4) are reference points. It landed before the campaign's first step as a dated
   pre-data amendment: `evidence/g1b-pool-selection.md` section 2.1 and `cortex/pool_selection.py::SELECTABLE`
   (commit `4b237c6`).
3. **Unsolved problems: "Pause at a budget".** Each problem gets a growth budget relative to a persisted running
   baseline (P1). At the limit the brain reports UNSOLVED, stops growing, and keeps the problem OPEN to retry after
   sleep or new teaching.
4. **Reservoir state: "In the lake, rebuilt at start".** There is no separate store: the reservoir's seed, its
   learned deltas, the transport record and the neurons fed into it are lake records, and a restart rebuilds it
   exactly. The drawn ellipse is the NETWORK boundary, not the durability boundary.

## 4. Session calls, disclosed (no owner objection; the owner's 2026-09-05 authorization)

- The engine is recorded in `docs/POLICIES.md` as a dated PROVISIONAL directive, P4d, a sub-clause of P4, with a
  dated owner exception appended to P2's corollary ("a research direction is not a policy").
- The e-prop invariant is UNCHANGED. The engine's transport feeds a reservoir's activations, never a weight update;
  a transposed forward weight feeding a weight update stays forbidden, including through a transported copy.
- Doc-budget ceilings are raised by the measured word delta of each amendment, each raise commented with its date
  and reason. The owner lowered the POLICIES.md ceiling on 2026-08-30.
- The constitution's cap of three provisional principles counts owner directives adopted with a specified gate
  separately. The audit offered this or dropping the constitution's PROVISIONAL marker on P7 and P8.
- *(Added in the 2026-09-25 panel fold.)* The null rule, one sentence written verbatim into POLICIES P4d and its
  e-prop pointer, FR-041's and SC-019's notes, the plan's glossary and G9, and T117: an owner-drawn or owner-chosen
  part goes to the owner on a measured null, and only transport lapses on its own. For decision C's parts it departs
  from the decision brief's "The four tests below then delete any part of it that does not pay" (section 5).

## 5. Where this lands, and what stays open

The amendments of this arc: POLICIES P4d with its pointers (the P2 corollary, the e-prop invariant, the P3 region
rules, the P0 avenue table, the enforcement table) and its incident block in `docs/POLICIES_EVIDENCE.md`; spec
FR-041, SC-019 and their dated notes; plan rows G0-G9 with reds G-R7 to G-R9 and the G7 decision record
(`evidence/g7-organizing-unit.md`); tasks T115-T117; the cold-start protocol doc; constitution 1.2.0.

Open for the owner, carried from the audit and not decided by any session call:
- whether the transport permission may ever reach credit assignment (P4d says it does not);
- whether P1 admits a value fitted once from measured external data (an H01 kernel width), or G4 keeps the fitted
  width running-baseline-relative;
- a growth target INSIDE the reservoir (a fed neuron becoming a reservoir unit, one of G3's arms);
- the reorganization's package layout, each path-keyed gate's disposition, and any envelope-file move (G8);
- whether a G-R9 null on an owner-drawn part changes the directive (POLICIES P4d's null rule sends such a null to the
  owner; the question is asked when one lands);
- whether the decision brief's "delete any part of it that does not pay" stands for decision C's four tests, which
  P4d's null rule (a session call of the 2026-09-25 panel fold) routes to the owner instead.

## 6. Sources

- The diagram labels as transcribed, the four research reports, the three audit lenses (governing text; plan, spec
  and docs; code gaps), the published brief, and the raw captures with the quote checker: the memory store's
  `archive/engine-research-audit-2026-09-25/` (MANIFEST inside).
- The decision brief: https://claude.ai/artifact/4S9PP8fP7PhgBU1c72Tkyn.
