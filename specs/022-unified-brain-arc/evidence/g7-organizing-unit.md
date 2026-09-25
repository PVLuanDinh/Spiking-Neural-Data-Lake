# G7 / T115 -- the organizing unit (owner decision 2026-09-25: C)

Row: plan Part D "G workstream", G7 / T115; red G-R7 (Part F); FR-041's dated note and FR-021's; POLICIES P4d
("Organizing unit"). The owner's diagram draws the brain "(split either by cortical columns or brain regions)", and
the owner's clarification left the unit open: "it could be either cortical columns or anatomically accurate brain, I
don't know which structure to consider since we need to do more research on the brain design philosophy through web
grab". After that research the owner chose. This file is the row's decision record. Status: DECIDED (the unit). The
row stays OPEN until its label gate lands with its constructed red (G-R7); the four deciding tests are owed by the
rows named in section 6. No code, no run.

## 1. The decision, as recorded

The owner's answer (2026-09-25, the recommended option of the decision brief
https://claude.ai/artifact/4S9PP8fP7PhgBU1c72Tkyn): **C, "grown columns on an anatomical map".** The option as
recorded in the session memory the same day: growth adds small column-sized batches (one radial unit, about 8-10
excitatory neurons plus inhibitory neurons from a shared pool; the size is an arm) where error builds up; every grown
neuron carries a position (x, y, depth), a layer, a Dale sign, a region label and the problem that grew it, in the
lake; wiring by rules (layer target x distance kernel x shared-partner bonus); regions are LABELS read off the map and
measured, never code packages; H01 is a fitted prior that must beat a shuffled copy. The owner's verbatim words and
the other three decisions are in `evidence/engine-structure-2026-09-25.md`.

## 2. The three options and their evidence, in brief

The brief's four research passes read primary papers through webgrab. Its 18 load-bearing quotes were re-found in the
saved pages; its stated limits were one brief and one model behind four agents, several abstract-only texts, and no
study comparing grown columns against grown single neurons in a spiking network at an equal budget (the gap test T1
exists to fill).

- **A, cortical columns** (one canonical circuit, repeated; a region is a group of columns). For: cortex grows by
  adding radial units in development (Rakic 1995; about 8-9 neurons per progenitor, Gao 2014); the liquid state
  machine's founding paper grew its reservoir by adding columns (Maass 2002); repeated units share parameters
  (Lappalainen 2024); one template is the cheapest build. Against: the column is contested as a functional unit
  (Horton & Adams 2005; rat V1 has orientation tuning without columns, Ohki 2005); more columns buy speed, not
  capacity, in Hawkins's models (Hawkins 2017); a layered column's edge over random wiring comes mostly from simple
  statistics (Schulte to Brinke 2022); areas differ systematically (Tasic 2018, Elston 2003, Chaudhuri 2015).
- **B, an anatomically accurate brain** (real regions and pathways; a connectome is the blueprint). For: a whole
  fly-brain wiring diagram with one free parameter predicted feeding behaviour (Shiu 2024); layered,
  distance-dependent wiring learned faster than random wiring of the same size (Chen 2022); a human connectome
  reservoir beat count-matched rewires near the edge of chaos (Suarez 2024); a region-level model ran eight tasks
  (Eliasmith 2012). Against: no mammalian whole-brain wiring diagram exists (the best human data is one 1 mm^3
  fragment, Shapson-Coe 2024); wiring does not fix the dynamics (Billeh 2020, Beiran 2025); connectome reservoirs did
  no better than random ones once weights had some randomness (Damicelli 2022); a connectome is a finished brain, not
  a recipe for growing one (Zador 2019).
- **C, grown columns on an anatomical map** (grow small units where error builds up; wire by rules; read regions off
  the map). For: every model that grows a cortex grows it from rules (Zubler 2013, Bauer 2014, Akarca 2021);
  brain-like modules emerge from a wiring cost alone (Clune 2013, Achterberg 2023); it keeps A's simple growth step
  and B's spatial statistics; it matches "regions should be neurons". Against: the most to build (positions, a
  distance kernel, layer targets, a growth-site rule); the structure gains were measured mostly in backprop-trained
  networks (Chen 2022); the grower holds 2-4 units today, so a map means little until hundreds exist; H01's 70,001
  edges may be too few to tell a Gaussian from an exponential falloff at short range.

True whichever was chosen, and binding on the rows: adult neocortex adds no neurons (Bhardwaj 2006), so the closest
biology to growing-to-solve is development and the adult olfactory bulb, where about half the new neurons survive in
step with sleep (Yokoyama 2011); grow-until-solved is a proven algorithm (cascade-correlation, Fahlman 1990); growth
mostly buys speed, not a better end result, so every growth claim needs a presized twin (Evci 2022); long-running
growth needs turnover (Dohare 2024).

## 3. The attachment rule

A grown neuron is placed where error builds up, in a column-sized batch, and gets a position (x, y, depth), a layer
and a Dale sign; its region label is read off the map from its position, never assigned by a code package. The
connection probability to a candidate partner is layer target x distance kernel x shared-partner bonus. The starting
kernel is a Gaussian of about 85-114 µm locally, falling off about 4.6 per mm through grey matter (the brief's
literature values), refitted on H01 by T3 before any arm uses a distance kernel (the plan's order puts T3 before
G3's attachment arms; a hand-picked width is refused).

The label vocabulary is all sixteen `cortex/growth_program.py::REGION_COORDS` names, a session call under the owner's
2026-09-05 authorization (dated 2026-09-25, the panel fold; open for the owner together with re-ratifying the
2026-08-19 pin of the ten growable names): only the ten `GROWABLE` names are admitted, and the five `SCAFFOLD` names
(brainstem, insula, medulla, olfactory, pons), neuromodulation and any position the map cannot tell apart from their
territory are refused. The ten names alone could not make the scaffold refusal fire: a label drawn only from
`GROWABLE` is never in `SCAFFOLD`. At today's grain only the neuromere is load-bearing
(`cortex/placement.py::region_prior`), so the 'sp' neuromere (six growable names with insula and olfactory) and 'r1'
(cerebellum with neuromodulation) cannot yet be told apart from refused territory; until the map resolves positions
finer, `cortex/placement.py::place_grown`'s name-based SCAFFOLD refusal stays the working guard.

Whether a width fitted once from H01 may then stay fixed is a P1 question the owner has not ruled on
(`engine-structure-2026-09-25.md` section 5). Until then G4 keeps the fitted width running-baseline-relative (a refit
per campaign on the pinned H01 data would return the fitted-once value).

## 4. What C means for the engine's other connections

- **The feed (connection 2).** A grown neuron keeps its identity, position, label and problem lineage when it projects
  into the liquid reservoir through declared input synapses (G3's feed arm). The reservoir needs no position; the
  label travels with the neuron, so a fed population is attributable per column batch and per label.
- **The return path (connection 6).** Readout neurons project back into the brain by LABEL: into the labeled
  grown-neuron population that posed the problem, never into the action gate (P6). G5's "anatomical" relational
  reading becomes which layers and labels the projection reaches, ablated separately from the categorical reading.
- **Recall and sleep.** A saved problem's neurons are saved with their labels (G6(d)); recall across a restart reads
  them from the lake like any other record (P4b).

## 5. The incumbents it adopts (P2: extend, never a second store)

- `cortex/placement.py` (`place`, `place_grown`, `region_prior`): already gives each grown unit a fractional depth, a
  layer and an immutable Dale sign, persisted with the growth state and read by nothing. Its record is EXTENDED with
  the position, the region label and the problem id. There is no second label store.
- `cortex/arealization.py::emit_regions`: FR-021's region program, the source of the label from the organizer field.
- `cortex/cortical_microcircuit.py::CONN_PROB`: a candidate source of the column batch's layer targets (driverless
  today; its dormancy row says so).
- `cortex/h01_ingest.py` and the gitignored `data/h01/h01_c3_graph.npz` (`xyz_nm` for 14,215 signed somata, 70,001
  aggregated pairs; pinned by `tools/h01_c3_graph_pin.json`): the statistics T2 and T3 fit.
- The one grower: `cortex/neurogenesis_engine.py::NeurogenesisSelfImprove` with `hippocampus/ng_snn.py::NGSNN.fit`
  (FR-041: no second grower).

Disclosed prior negative, binding on T2: T063 (2026-09-09, plan Part F H-R1; `evidence/t063-prior-drift-harness.md`)
recorded "a beatable prior" NOT MET for H01's wiring beyond its marginals on T063's task: the seeded arm beat neither
the random nor the wrong prior, and the learner was deleted on its own criterion. T063's null kept H01's degrees,
signs and magnitudes, so it did not test position; T2's shuffled-positions null asks a different, spatial question.
T2 may not cite T063's learner-ON pin as its evidence, and H-R1 already says an evidential re-reading on H01 needs a
new pre-registration.

## 6. The four deciding tests, mapped to the rows that run them

Each is pre-registered in its row's evidence file before its first run, with its bar, under Part F's paired-seed rule.

| test | what it compares | red | runs in |
|---|---|---|---|
| T1, the growth-unit race | a column-sized batch vs single neurons grown next to the worst readouts vs a presized network of the final size vs random attachment, on the same problems at the same neuron budget | columns need 20% or more extra neurons than single-neuron growth at equal accuracy, or forget more; a presized network matching both means growth itself earns nothing | G3's growth-unit arms; G9 at engine level |
| T2, anatomy vs shuffled anatomy | growth with the H01 prior (position, layer, sign, distance rule) vs the same prior with positions shuffled vs a generic template | H01 does not beat shuffled H01; the H01 prior is then reverted, and the map part of C goes to the owner | G4 |
| T3, does H01 favour a Gaussian | connection probability against distance on H01: Gaussian, exponential, both, or flat, scored on held-out neurons | the Gaussian loses to the exponential or the flat model; REFUSED (a blind fit) if a planted Gaussian (sigma = 100 µm) is not first recovered within 10% | G4 |
| T4, emergent vs imposed organization | rule-grown wiring (distance x shared partners, with a wiring cost) vs imposed columns vs random attachment, scored on modularity and task score | modularity no higher than random attachment, or a task score below imposed columns | G4 (wiring); G9 (task score in the engine) |

*Dispositions (dated 2026-09-25, the panel fold; a session call under the owner's 2026-09-05 authorization):* a red
on any test reverts the implementation it tests, and the part of C it tests is not claimed and goes to the owner under
POLICIES P4d's null rule. The decision brief the owner chose C from said "The four tests below then delete any part
of it that does not pay" (T2: "the map part of C is deleted"); whether that deletion stands without a further ruling
is open for the owner (`engine-structure-2026-09-25.md` section 5).

## 7. The red (G-R7, plan Part F)

G-R7 fires if a grown neuron can be admitted without its unit label (position, layer, Dale sign, region label, the
problem that grew it) and no gate refuses it, if a neuron whose position reads a refused label is admitted (section
3), or if the tests above name no falsifiable comparison. The controls are a constructed grown neuron with no label and
a neuron placed in refused territory (insula), each of which the gate must refuse; five constructed grown neurons,
each missing ONE field and each refused (per-field attribution); and one fully labeled neuron, which the gate must
admit, so a gate that refuses everything reds (specificity, added 2026-09-25 in the docs panel's fold). The sleep
that kept or pruned a unit is a field of its SAVED record (G6(d), G-R6), not of its admission; the gate lands with the first code that admits
labeled grown neurons (G3's attachment arms at the latest), and T115 closes only then.

## 8. Sources

The decision brief (https://claude.ai/artifact/4S9PP8fP7PhgBU1c72Tkyn) lists all 36 sources with DOIs; its four
research reports and the raw captures are in the memory store's `archive/engine-research-audit-2026-09-25/`.
References, as the brief gives them: Mountcastle 1997 (10.1093/brain/120.4.701); Rakic 1995
(10.1016/0166-2236(95)93934-P); Gao 2014 (10.1016/j.cell.2014.10.027); Maass, Natschlaeger & Markram 2002
(10.1162/089976602760407955); Lappalainen 2024 (10.1038/s41586-024-07939-3); Horton & Adams 2005
(10.1098/rstb.2005.1623); Ohki 2005 (10.1038/nature03274); Hawkins, Ahmad & Cui 2017 (10.3389/fncir.2017.00081);
Schulte to Brinke 2022 (10.3389/fnint.2022.923468); Tasic 2018 (10.1038/s41586-018-0654-5); Elston 2003
(10.1093/cercor/bhg093); Chaudhuri 2015 (10.1016/j.neuron.2015.09.008); Shiu 2024 (10.1038/s41586-024-07763-9);
Chen, Scherr & Maass 2022 (10.1126/sciadv.abq7592); Suarez 2024 (10.1038/s41467-024-44900-4); Eliasmith 2012
(10.1126/science.1225266); Shapson-Coe 2024 (10.1126/science.adk4858); Billeh 2020 (10.1016/j.neuron.2020.01.040);
Beiran & Litwin-Kumar 2025 (10.1038/s41593-025-02080-4); Damicelli 2022 (10.1371/journal.pcbi.1010639); Zador 2019
(10.1038/s41467-019-11786-6); Zubler 2013 (10.1371/journal.pcbi.1003173); Bauer 2014 (10.1371/journal.pcbi.1003994);
Akarca 2021 (10.1038/s41467-021-24430-z); Clune, Mouret & Lipson 2013 (arXiv:1207.2743); Achterberg 2023
(10.1038/s42256-023-00748-9); Levy & Reyes 2012 (10.1523/JNEUROSCI.5158-11.2012); Horvat 2016
(10.1371/journal.pbio.1002512); Bhardwaj 2006 (10.1073/pnas.0605177103); Yokoyama 2011 (PubMed 21903081); Fahlman &
Lebiere 1990 (NeurIPS 2); Evci 2022 (arXiv:2201.05125); Dohare 2024 (10.1038/s41586-024-07711-7).
