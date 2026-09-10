# T062 — the H01 c3 synapse graph as a beatable prior: ingest, gates, population

Pre-registered 2026-09-09 before any ingest code exists. Rules are written before numbers; every
number that appears in §1 was MEASURED on this date from the release itself and is cited to its
source. The plan of record orders this row next (`PLAN-FINAL-2026-09-02.md`, critical path, dated
note 2026-09-09: "T062 is next"); its content is fixed there — *the c3 SYNAPSE GRAPH only, composed
onto `cortex/cortical_prior.py` … the join gate, unit gate, layer computation, CC-BY provenance* —
and by `tasks.md::T062` and `spec.md::FR-022`. The audit's item (5) (a second ingest mechanism with no
P2 adjudication against `cortical_prior`) binds: everything `cortical_prior` already holds is REUSED,
never re-derived here.

## 1. Territory (measured 2026-09-09; the orientation is `scratchpad/t062/design.md`, 82 facts)

**1.1 The export.** `gs://h01-release/data/20210729/c3/synapses/exported/` holds **166 Avro shards**
(`export000000000000`…`export000000000165`, 197.3–198.0 MB each, **32,859,876,087 bytes** total) and
166 JSON twins (126.1 GB). Listed through the GCS JSON API with each object's `size`, `md5Hash`,
`crc32c` and `generation`; the listing is pinned verbatim as the shard manifest (§2.1). The Avro
container's codec is **`null`** (uncompressed) and its header schema (record `Root`, 7 fields:
`pre_synaptic_site`, `post_synaptic_partner`, `location`, `type`, `contact_area`, `confidence`,
`bounding_box`; each site carries `neuron_id: long`, `base_neuron_id: string`, `class_label: string`,
`centroid`, `type`, `mask: bytes`) was read from shard 0's first 64 KB. The release's data page
(`https://storage.googleapis.com/h01-release/data.html`) names it *"Synaptic connections database:
Export of synaptic connections in Apache Avro format"* and states the licence: ***"All released
datasets are licensed under a Creative Commons Attribution 4.0 License."*** `ingest_guard.license_ok
("cc-by-4.0")` is True.

**1.2 The coordinate frame.** The synapse points' own `info`
(`…/c3/synapses/precomputed/info`) declares `dimensions x: 8e-9 m, y: 8e-9 m, z: 3.3e-8 m`, bounds
x 64,667–483,997, y 39,426–296,180, z 63–5,231. The pinned cell file (`agglo_20200916c3_cell_data.json`)
carries BOTH `true_x/y/z` (nanometres) and `x_vx/y_vx/z_vx`, with `true_x / x_vx = 4.0` exactly on
every sampled row. So the export is in **8 nm voxels** and the somas in **nm**; `placement.layer_of`
takes **microns**. Three unit systems in one join — the trap the task row names.

**1.3 The join keys.** A site's `neuron_id` is the c3 AGGLOMERATED id; `base_neuron_id` is the base
segment. The cell file's keys are `agglo_seg` and `base_seg`. The upstream helper `get_base2agglo`
(`common_functions_h01.py`) exists precisely because the two id spaces differ. "base↔agglo instead of
neuron_id↔agglo_seg" (the task row's forbidden join) is therefore `base_neuron_id ↔ agglo_seg`, and
the silently-empty graph it produces is a measurable count.

**1.4 The E/I code.** Upstream counts a synapse `type` of **2 as excitatory and 1 as inhibitory**
(`get_neuron_e_to_i_ratios.py` lines 237–238: `e_count = syn.count(2)`, `i_count = syn.count(1)`).
The site-level `type` is 1 = pre, 2 = post (the sample record). The cell-type → sign map is
`cortical_prior.SIGNED_TYPES` (already in-tree; `edge_list_to_graph.py` upstream carries the same
map).

**1.5 The pinned sources resolve.** The GitHub tree at `cortical_prior.H01_COMMIT`
(`db223b8d…`) lists all five `SOURCES` files with byte-exact sizes (11,186,494 / 1,998,308 / 7,994 /
9,230 / 496) and `classify_layers_all_segments.py` (the "port classify_layers" of the task row; its
rule is `fix_layer_mem`, which `placement.layer_of` already implements at a pinned agreement of
45,404 / 49,377 = 0.9195 against the CSV's `cortical layer` column).

**1.6 The decoder.** No Avro reader is installed in `.venv-lake` and PyPI has no `cp314t` Windows
wheel for `fastavro` (cp314t wheels exist for Linux/macOS only). A schema-specific stdlib decoder
(`scratchpad/t062/avro_probe.py`, 120 lines) decoded shard 0's head at **15,491 records/s** (253
blocks, 21,112 records, 4.18 MB) and its first record matches the JSON twin's first record on
`pre.neuron_id`, `post.neuron_id` and `type`. Projection: ~1.0 M records/shard, ~70 s/shard/thread;
the interpreter is free-threaded, so the shards parallelise on threads.

**1.7 The substrate it seeds.** `KernelState([(faculty, n)])` (SoA: v, thr, refrac, last_spike, sign)
and `Synapses(pre, post, weight, delay, n_units, p_parts, frac_bits)` exist; no constructor builds
either from external data; `Synapses.empty` has zero callers. The kernel has no plasticity (S6).
`tests/test_placement_charge_null.py` pins that NOTHING on a live path reads placement or the prior.
`web_ingest` carries the network firewall (`validate_url`, the re-validating redirect handler) with a
5 MB text cap and no binary fetch. `data/` is gitignored; `models/` carries tracked-pack exceptions.

**1.8 The host.** D: 880 GB free (the repo and `data/` live on D:); C: holds the live lake and is not
touched. 34.14 GB RAM, ~11.8 GB available with the daemon up.

## 2. Pre-registration

**2.1 Deliverables.**
- `cortex/h01_ingest.py` — the ONE ingest mechanism: manifest-verified streaming fetch of Avro shards
  (never stored), the stdlib decoder with the header schema PINNED by sha256, the four gates (2.3), the
  soma-to-soma aggregation, the artifact writer, and `population()` (2.2). Registered in REGIONS,
  `run_selfchecks` CORE, and `dormancy_gate.EXEMPT` as a dated `wire` row whose consumer is T063 (the
  three-arm harness); reachable from `run_selfchecks` only, never from a tick.
- `tools/h01_c3_shards.json` — the 166-shard manifest (name, size, md5, crc32c, generation, listed-at),
  plus the export's declared voxel size, codec and licence sentence with its URL.
- `tools/h01_c3_graph_pin.json` — the derived artifact's sha256, the shards it was built from, every
  gate's reading, the harness sha256s (`h01_ingest.py`, this file's decoder) and the git head — the
  T068 debt (§3.6 item 1 there) discharged: no cell without its harness hash.
- `data/h01/` (gitignored, on D:) — the five pinned SOURCES files fetched from the pinned commit's raw
  URLs and refused unless their sha256 equals `cortical_prior.SOURCES`; and `h01_c3_graph.npz`.
- `tests/test_h01_ingest.py` — every red in 2.3 on synthetic fixtures (an in-test null-codec Avro
  WRITER produces the shard fixtures; no network in tests), plus the P4b and determinism reds.
- A PROVENANCE line naming H01, the licence and FR-007 ("not a measurement of this brain") wherever
  a cockpit-reported quantity derives from the prior; if none does, the line lives on the artifact's
  meta and the record says so.
- `specs/011-h01-neurogenesis-prior/data-model.md`'s read-once rule holds: the artifact is read at
  construction of a population, never on a tick.

**2.2 The population (what "ingested population" means, so T063 and the S6 re-dating have a
referent).** Units = the SIGNED neurons of the cell file (pyramidal, spiny stellate, atypical spiny →
+1; interneuron → −1; every other type EXCLUDED, never signed by default — `cortical_prior`'s rule).
Faculties = the seven laminar bands (`placement.LAYER_NAMES` + `WHITE_MATTER`), units ordered by
(band, agglo id) so each band is an O(1) slice. Edges = soma-to-soma synapses aggregated by (pre,
post) with counts `n_syn`, `n_exc`, `n_inh`. Weight = `sign(pre) · n_syn(pre, post) / Σ_pre'
n_syn(pre', post)` — the fan-in share, dimensionless, so a unit's summed |input| is 1 by construction
(a normalisation, DISCLOSED as not a measurement). Delay = 1 tick for every edge — H01 carries no
conduction delays and this project's connectome delay FIELD has no VALUES (memory); DISCLOSED, never
derived from a velocity constant. Threshold and decay = the live LIF equation owner's defaults by
import (P1: cited, not invented); `p_parts` and `frac_bits` from the lineage's pinned baselines
(`tools/kernel_partition_baseline.json`, `tools/determinism_baseline.json`). The constructor is
deterministic (2.3 f). It is NOT wired to the live tick: the wire is the kernel wire, re-armed at N*
by S6, and its first consumer is T063.

**2.3 Gates — each a refusal with a constructed RED run RED before the mechanism, and a control that
is not the method, in the same command.** No gate carries an invented threshold; each compares the
declared reading against its alternatives and requires the declared one to be the extremum.
- **(a) Bytes.** A shard's streamed md5 must equal the manifest's; a byte overrun past the manifest
  size is a refusal. RED: a manifest entry with one hex digit flipped → refused, naming the shard.
  CONTROL: the same bytes under the true entry pass. The header's `avro.schema` sha256 must equal the
  pinned value; RED: a fixture whose schema reorders two fields → refused before any record is read.
- **(b) Join.** The declared key pairing is `neuron_id ↔ agglo_seg`. In the SAME pass the ingest counts
  both-ends-hit synapses under all four pairings ({`neuron_id`, `base_neuron_id`} × {`agglo_seg`,
  `base_seg`}) and prints all four. REFUSE unless the declared pairing's count is the strict maximum.
  RED: the caller declares `base_neuron_id ↔ agglo_seg` → refused, the four counts printed. CONTROL:
  the declared pairing on the same shard passes. (Direction: this refusal protects "the graph is the
  c3 graph" — a refusal yields NO artifact; there is no verdict for it to hide.)
- **(c) Units.** The declared scale is (8, 8, 33) nm/voxel. The ingest converts the joined synapses'
  locations under four candidate scalings — declared (×8, ×8, ×33), the cell file's voxel (×4, ×4,
  ×30), identity (×1), and the inversion (÷8, ÷8, ÷33) — and for each prints the ratio of the synapse
  cloud's nm extent to the soma cloud's nm extent per axis. REFUSE unless the declared scaling's
  extent ratios are the closest to 1 on every axis. RED: declare the inversion → refused. CONTROL: the
  declared scaling passes. (Direction: protects "locations are in nm".)
- **(d) Layer.** Computed, not read (FR-022): `placement.layer_of(true_x / 1000, true_y / 1000)` per
  soma record. VALIDATOR: agreement with the CSV's `cortical layer` over all soma records must
  REPRODUCE `placement.LAYER_RULE_AGREEMENT`'s pinned fraction 45,404 / 49,377 EXACTLY — the P2
  evidence that this is the same rule and the same bytes. RED: feed nm instead of microns → every
  soma falls through to Layer 1 and the agreement collapses → refused (specs/011 Scenario 3's red).
  CONTROL: microns pass.
- **(e) Sign, two instruments.** Instrument 1: the PRE cell's type → sign (Dale, `SIGNED_TYPES`).
  Instrument 2: the export's per-synapse `type` (2 = E, 1 = I). Their agreement rate over joined
  synapses is printed; the population uses instrument 1 (sign is a property of the cell, immutable
  at birth — `placement.assign_sign`'s rule). REFUSE unless agreement > 0.5 — under the swapped map
  (1 ↔ 2) the rate is exactly 1 − r, so the rule has its red built in. RED: swap the map → refused.
  CONTROL: the upstream map passes. The disagreement rate is a disclosed number, never rounded to zero.
- **(f) Determinism and order-independence.** The artifact built from shards [0..k) equals, byte for
  byte (sha256), the artifact built from the same shards in REVERSE order; two constructions of the
  population from one artifact yield byte-equal `KernelState` arrays and a byte-equal 30-tick spike
  train under a fixed drive. RED: a fixture ingest that appends edges in encounter order without
  sorting → the two orders differ → refused. CONTROL: the sorted path passes.
- **(g) P4b — the prior never enters the lake.** A full synthetic ingest + population construction
  under `throwaway_home()` leaves the lake's record count unchanged (counted through the lake's own
  parser). RED: a deliberate `save_named_state` of the artifact → the count rises → the test names it.
- **(h) Liveness of the population.** Driving 1 % of units for one tick propagates to at least one
  further spike within 30 ticks on the real artifact (a connected graph). RED: zero all weights →
  no propagation → the self-check names "no propagation". This is a liveness check on the seed, not a
  claim about learning.

**2.4 Stages.** Stage A: the first slice = the lexicographic PREFIX of 35 shards (`export…000` …
`…034`, 6,913 MB, measured under 7 GB — the plan's "cheapest first slice"); its artifact and pin are
recorded. Stage B: all 166 shards in the same arc if stage A's measured rate projects the full run
under 90 minutes on this host; the stage-B artifact REPLACES stage A's (the pin records both). If
stage B does not run, the record says so and names the projection.

**2.5 What is NOT claimed.** No learning, no drift, no "beats random" — those are T063's (SC-006, both
reds). No timing claim beyond the decode rate, reported with the host condition. No claim that the
graph is the full H01 connectome: it is the soma-to-soma subgraph of the c3 export, measured. The
"~16k signed neurons" of the spec is a rounding; the population's unit count is measured and printed
beside `cortical_prior`'s 14,331.

**2.6 Host condition (obs #2147; feedback: a timing reading after a commit measures the daemon's
boot).** The real ingest runs as its own process, NORD_HOME on a scratch dir, with the daemon UP and
its cockpit LISTENING (port 8137) — and NOT during any battery or panel. The decode rate is read
with the daemon's PID and working set recorded beside it.

**2.7 Null clause.** The ingest has no WIN/LOSS; it has REFUSALS. If any gate in 2.3 refuses on the
real data, no artifact is written, the readings are recorded, and the row stays OPEN with the
refusal named — nothing is weakened to pass. If the population cannot tick (the kernel refuses its
parameters), the same. Deletion applies to the mechanism only if T063 later measures that the
seeded arm cannot be distinguished from random on any statistic — that is T063's null, recorded
there.

**2.8 Order.** Reds committed RED (a collection error or failing tests naming each gate) → the
mechanism → the synthetic self-check green with every red in the same command → stage A → stage B
→ the record (§3) → the panel (three lenses; the fold is stacked commits) → ONE battery on the
stacked head → the closing record → push.

### 2.9 Amendment BEFORE the re-run (2026-09-09; the readings already taken are quoted, and none is discarded)

**What happened.** The ingest ran as pre-registered on stage A (35 shards, 34,994,654 records) and, its
projection being 16.3 min, on stage B (all 166 shards, 166,216,068 records, 845.78 s wall). Gates (a)–(d)
PASSED on both: 166/166 md5 equal to the manifest and the pinned schema hash on every shard; the declared
join `neuron_id ↔ agglo_seg` the strict maximum (516,214 both-ends hits against 56,707 / 1,422 / 305 for
the other pairings — the forbidden `base_neuron_id ↔ agglo_seg` is 363× smaller); the declared (8, 8, 33)
scaling closest to 1 on every axis (0.8601, 0.8519, 1.0166 against 0.43 / 0.11 / 0.013 for the cell-voxel,
identity and inverted scalings); the computed layer reproducing 45,404 / 49,377 exactly. Gate (e) REFUSED
on both stages: 12,802 / 42,610 = 0.3004 and 60,949 / 202,232 = 0.3014 — a reading that moves 0.001 across
a 4.7× larger sample is a reading, not noise, and the swapped map (0.6986) is not near 1 either. Per 2.7
NO ARTIFACT WAS WRITTEN and the pin records the refusal with every gate's reading.

**Why gate (e) was the instrument's defect, not the data's (measured on shard 0, 998,235 records).**
2.3(e) compared the PRE cell's Dale sign with the export's per-synapse `type`. Three readings show the
two are not readings of one quantity: (1) by PRE cell type the export's excitatory fraction does not
separate excitatory from inhibitory classes (pyramidal 0.370, spiny stellate 0.344, atypical spiny 0.338,
interneuron 0.308) and 54.4 % of the pre-end soma hits are glia or vasculature, which cannot be
presynaptic; (2) by POST cell type it separates cleanly (pyramidal 0.639, spiny stellate 0.672, atypical
spiny 0.630, interneuron 0.538, oligodendrocyte 0.448, astrocyte 0.337); (3) a third instrument that
fails differently — upstream's own per-segment `excitatory synapses` / `inhibitory synapses` counts in the
e/i CSV — agrees on the POST side (over 9,716 post segments, Pearson r 0.494 between the export-derived and
the CSV excitatory input fractions). The join is 19× asymmetric by end (174,473 post-end hits per shard
against 9,170 pre-end): the pre end of a synapse is an axon, and axons are the fragmented, mis-merged
side of this reconstruction. 1.4's reading of `get_neuron_e_to_i_ratios.py` (2 = excitatory,
1 = inhibitory) is correct, but that script counts synapses RECEIVED by a neuron — a POST-anchored
quantity — and 2.3(e) compared it to a PRE-anchored one.

**The replacement gate (e′), pre-registered here before the re-run.** Instrument 1: the export's `type`
aggregated per POST cell into that cell's excitatory INPUT fraction. Instrument 2: the e/i CSV's own
counts for the same segment id. Same quantity, failing differently. RULE (no invented threshold): the
Pearson correlation between the two over the joined post cells must exceed the 95th percentile of the
same correlation under 1,000 permutations of the CSV's segment labels (the constructed red: shuffled
labels → the correlation collapses to the permutation band → refused); both the real and the permuted
readings are printed. CONTROL: the real labels pass. The export's `type` is used for NOTHING else.

**What is kept unchanged.** The population's sign is the CELL's (Dale, immutable at birth) — it never
depended on gate (e). Every other gate, red and control of 2.3 stands as written and is re-run.

**Disclosure carried into T063 (the prior's quality is now a measured fact).** Of the 516,214
soma-to-soma synapses, 202,232 carry a Dale-signed pre; the export's E/I type disagrees with the pre
cell's class on roughly two thirds of those, and the pre-end soma hits are majority glia/vascular. The
population therefore carries the export's PRE identities as they are (edges from signed pre cells only;
no edge is filtered by the export's type — filtering edges to match the prior would be selecting the data
to fit the seed), and the record states the fraction of pre-end hits that are non-neuronal. Whether that
wiring beats its marginal-matched null is exactly T063's question.

**Three defects the gates found in the builder's own apparatus, recorded.** The first synthetic fixture
let every base id resolve, so the join gate tied 6–6 and refused (a fixture that cannot tell which key it
got); the artifact meta carried shards in READ order, so the determinism gate compared two hashes that
differed only in the list it was meant to be independent of; `np.savez` embeds the wall clock in its
zip, which would have made the artifact's sha256 a timestamp. Each was fixed before any real reading.

**Authorized by this amendment:** a second stage-B run under gate (e′), producing the artifact and the pin
iff every gate passes; the stage-B readings above are kept in the record beside the re-run's.

## 3. Results (written after the runs; nothing above this line was edited after §2.9)

### 3.1 The reds, RUN RED, and the instrument as built

The reds were committed before the mechanism existed (`bd886a2`: a collection error, `ImportError: cannot
import name 'h01_ingest'`) and again before each later mechanism (`6e6917f` a refusing ingest must carry
every gate reading — `AttributeError: 'IngestRefused' object has no attribute 'readings'`; `194223d` the pin
must never erase an earlier stage; `a8f3541` gate (e′) — 17 tests failing, each naming the replacement gate
or its fixture; `0116c09` a same-stage re-run must not erase the refusal it stands beside). The mechanism
landed at `5b51179` (`cortex/h01_ingest.py`, REGIONS, `run_selfchecks` CORE, a dated `wire` row in
`dormancy_gate` naming T063 as the consumer) after `98b88f7` gave `web_ingest` ONE firewalled binary
download (`fetch_bytes`; `fetch_text` now decodes it — P2, one mechanism; all eight prior reds plus a ninth
mutation-verified reachable). Gate (e′) replaced gate (e) at `7004148` (the retired pre-anchored refusal is
DELETED, not flagged); `169348b` makes a re-run of the same stage a NEW pin reading (`#2` suffix; the first
reading keeps the plain key). The pinned inputs are `tools/h01_c3_shards.json` (166 shards, 32,859,876,087
bytes, the GCS listing verbatim) and `tests/fixtures/h01_c3_shard0_head.avro` (185,659 bytes, 11 whole
blocks, 928 records of real bytes) at `d9e2b71`.

Three defects the gates found in the builder's OWN apparatus before any real reading, each pointing at
"nothing is wrong": the first synthetic fixture let every base id resolve, so the join gate tied 6–6 and
refused (a fixture that cannot tell which key it got); the artifact meta carried shards in READ order, so the
determinism gate compared two hashes that differed only in the list it was meant to be independent of;
`np.savez` embeds the wall clock in its zip, which would have made the artifact's sha256 a timestamp.

How gate (e′) was realised within 2.9's words (the builder's choices, disclosed): "the joined post cells"
means every synapse whose POST end resolves to a CSV segment (35,374 of 35,376 usable segments), not only
both-ends-joined synapses; the CSV's 10,618 rows with 0/0 counts are dropped before the null is drawn (0/0 is
not "half excitatory"), so the permutation draws from exactly the population the real reading joins; the
1,000 permutations are a uniform draw without replacement of |ids| labels from the CSV, distributionally
identical to permuting the file's labels and re-joining.

*Appended 2026-09-09 (panel fold, C16) — a FOURTH realisation choice, not named in the paragraph above:* the
permutation RNG is `EI_SEED = 0` (`cortex/h01_ingest.py`, beside `EI_PERMUTATIONS` and `EI_PERCENTILE`). 2.9's
rule names the percentile and the trial count and says nothing about a seed, and the seed genuinely determines
the bar `check_ei` compares against, so it belongs in this paragraph and was omitted. What makes it inert is
the part worth recording: the bar is set by the joined-cell count, not by the seed — the 95th percentile moves
0.0009 across seeds 0/1/2/3/7 at this shape, and the analytic 1.645 / sqrt(35,374) = 0.00875 reproduces the
measured 0.00888 — against r = 0.4527, a 49x margin no seed reaches (panel C16's measurement). 2.9 is NOT
edited: the seed is disclosed here, in post-run text, and a pre-registration is never retro-fitted to what the
code did.

*Appended 2026-09-09 (panel fold, C13) — how gate (h) was realised within 2.3's words (the builder's choices,
disclosed).* 2.3(h) says "Driving 1 % of units for one tick" and fixes neither WHICH units nor at what
amplitude. As built the drive set is `drive_indices` — the 142 highest-OUT-DEGREE units, ties by index — and
the amplitude is 1.2 x threshold in fixed point (4,915 at `frac_bits` 12). Neither was pre-registered, and
3.2's four-word parenthetical states the choice without saying that 2.3 never fixed it. The record already had
a convention for exactly this — the (e′) paragraph above — and it was applied to (e′) and not to (h).
POST-HOC CHECK, measured in this fold on 2026-09-09 on the pinned artifact and NOT one of the run's readings
(`h01_ingest.population()` + `drive_indices`, then five INDEPENDENT `random.Random(s)` samples of 142 of
14,215): the gate and its constructed red BOTH hold under an unbiased drive — seeds 0..4 give totals 159 /
161 / 155 / 153 / 158, range 153–161, and all five propagate — and the zero-weight red stays RED under a
random drive as well as under the degree-selected one ([142, 0, 0, 0], propagates False). The amplitude is
inert across an order of magnitude: 1.0x, 1.2x, 2.0x and 10.0x threshold all give the identical train
[142, 71, 6, 1, 0], total 220, because a driven unit spikes and resets and discards the excess. So the
selection buys the gate nothing and changes only the published cascade counts (220 against 153–161), not any
verdict; 10,606 of the 14,215 units carry at least one out-edge. The seed convention is stated exactly so the
range reproduces: FIVE INDEPENDENT SEEDS, `Random(0)`..`Random(4)`. Five successive draws from ONE
`Random(0)` stream give 151–175 — a different quantity, which the stated method does not produce, and it is
not carried into this record. `tools/h01_c3_graph_pin.json` stores only `propagates_within_30: true`, so the
selection reaches no pinned value; and it feeds gate (f)'s byte-equal 30-tick comparison, where a richer
220-spike train makes that comparison strictly more sensitive than a ~155-spike one.

*Appended 2026-09-09 (panel fold, C6) — what "all eight prior reds plus a ninth mutation-verified reachable"
does and does NOT cover.* The ninth red is STRUCTURAL (`"fetch_bytes" in fetch_text.__code__.co_names`), and
`web_ingest.main()` never invokes `fetch_bytes` or `fetch_text` at all — so NO committed red exercises any
behaviour inside the new function's body. Measured by the panel on 2026-09-09, six mutations of a scratch copy
with the repo restored byte-identical afterwards, ALL GREEN (panel C6): the streamed byte cap; `extra_check`
on the ORIGINAL url; `validate_url` on the ORIGINAL url; `extra_check` on the FINAL url; `validate_url` on the
FINAL url; and `_ValidatingRedirectHandler` NOT INSTALLED in the opener at all. The two reds that DO survive
(firewall, redirect) drive `validate_url` and `redirect_request` DIRECTLY, so they prove those components
work — never that the download path uses them. Presence is not participation, and here it sits on a security
boundary. The module's own printed disclosure names three of the six and therefore understates by three; the
sharpest of the missing three is the uninstalled handler, under which the whole self-check stays green while
the download path silently follows unvalidated redirects. The cap itself was found behaviourally CORRECT by a
REVIEW-TIME, out-of-tree fake-opener probe — body = cap returns; body = cap+1 refuses on both the buffered and
the `sink` paths; a 5 MB body refuses with 1,001 bytes handed to the sink, i.e. never buffered whole — and
that probe is labelled what it is: a review-time measurement, not an arc gate, not committed, not registered
and not re-runnable from the repo. Its one substantive reading is that with `sink=` the sink receives cap+1
bytes before the refusal is raised. The owed red is 3.5(13), named as future work rather than as an option.

### 3.2 The readings (every gate, every run)

| gate | stage A (35 shards, 34,994,654 rec) | stage B (166 shards, 166,216,068 rec) | stage B re-run under (e′) |
|---|---|---|---|
| (a) bytes | 35/35 md5 = manifest | 166/166 md5 = manifest; schema sha256 pinned on every shard | 166/166; identical |
| (b) join, declared `neuron_id↔agglo_seg` | **108,958** (strict max) | **516,214** (strict max) | 516,214 |
| `base_neuron_id↔base_seg` | 12,164 | 56,707 | 56,707 |
| `base_neuron_id↔agglo_seg` (forbidden) | 309 | 1,422 (363× smaller) | 1,422 |
| `neuron_id↔base_seg` | 65 | 305 | 305 |
| (c) declared (8,8,33) extent ratios x/y/z | 0.8585 / 0.8467 / 1.0166 | 0.8601 / 0.8519 / 1.0166 (closest to 1 on every axis) | identical |
| cell-voxel (4,4,30) | 0.4292 / 0.4234 / 0.9241 | 0.4301 / 0.4259 / 0.9241 | identical |
| identity | 0.1073 / 0.1058 / 0.0308 | 0.1075 / 0.1065 / 0.0308 | identical |
| inversion (÷8,÷8,÷33) | 0.0134 / 0.0132 / 0.0009 | 0.0134 / 0.0133 / 0.0009 | identical |
| (d) layer agreement | 45,404 / 49,377 exact | 45,404 / 49,377 exact | identical |
| (e) pre-anchored sign (RETIRED by 2.9) | 12,802/42,610 = 0.3004 → REFUSED | 60,949/202,232 = 0.3014 → REFUSED | — |
| (e′) post-anchored r vs permutation bar | — | — | **r = 0.4527** over 35,374 cells; bar 0.0089 (null −0.0180..0.0200, mean 0.0002) → PASS |
| artifact | none (refused) | none (refused) | written |

Offline (fixture) arms, run in the self-check on every invocation: (f) forward and reverse shard order
give one artifact sha256 and the unsorted control diverges; two constructions give byte-equal
`KernelState` arrays and a byte-equal 30-tick train; (g) a full synthetic ingest + population under
`throwaway_home()` leaves the lake at 0 → 0 records while the deliberate `save_named_state` moves the same
counter to 1 (the arm is sensitive); (h) driving 1 of 4 units propagates within 30 ticks and the zeroed-weight
control does not. On the real artifact (`scratchpad/t062/item5.py`): order-independence measured on the FIRST
8 SHARDS ONLY (forward = reverse = `227f1527…307f`, 7,992,186 records, two re-fetched passes 123.2 s; the
unsorted control diverges); two constructions byte-equal (state and a 1,760-byte 30-tick train); liveness:
142 driven units (1 %, highest out-degree) → per-tick spikes 142, 71, 6, 1, 0 … (220 total, propagates);
the zero-weight control 142 (drive only), does not propagate.

Disclosures measured on the full export and filtered on by nothing: pre-end soma hits 1,494,143, of which
831,649 (0.5566) are non-neuronal; soma-to-soma synapses 516,214; with a Dale-signed pre 202,232; with BOTH
ends signed 104,959; among the signed-pre synapses the export's type disagrees with the pre cell's class on
141,283 (0.6986).

Timing (reported, never asserted): the final stage-B run 840.66 s wall with 8 threads, 197,720 records/s
aggregate (31,687 per thread; per-shard decode 27.5–34.0 s); threads beat one thread, measured on 4 shards
(1 → 46,969 rec/s; 4 → 146,089, 3.11×; 8 → 149,834). The earlier stage A 206.06 s projected stage B at 977 s
(over-estimating by 15.6 %). Host condition: the daemon UP (PID 2172 during the final run, started 16:32:08,
working set 5,871–8,420 MB) and its cockpit LISTENING in all 27 samples at 30 s; free RAM 8.4–11.1 GB;
NORD_HOME on a scratch dir; no battery or panel running.

*Appended 2026-09-09 (panel fold, C2) — gate (c)'s denominator included 12 null-coordinate rows.* The as-run
triples in the table above STAND as emitted and are not restated. The denominator was `soma_bounds` over the
FULL 49,377-row soma cloud, and 12 of those rows are null placeholders — `base_seg` '0', `agglo_seg` '0', type
'unknown cell', `x_vx`/`y_vx`/`z_vx`/`soma_cubic_um` all 0 beside `true_x`/`true_y`/`true_z` all 0. They are
the ONLY rows at x == 0 or y == 0, so they alone put the ORIGIN at the cloud's corner and inflated the x and y
spans by 1.163x and 1.164x; a min() taken over a row with no coordinates is not an extent of anything.
CORRECTED by EXACT ARITHMETIC on the sha256-pinned cell file, NOT by a re-run — measured in this fold, bounds
as-built [(0, 3,864,336), (0, 2,361,656), (0, 167,670)] → corrected [(541,376, 3,864,336), (331,968,
2,361,656), (0, 167,670)]; z does not move because 715 real somata sit at z == 0:

| scaling | as run (stands) | corrected denominator |
|---|---|---|
| declared (8,8,33) | 0.8601 / 0.8519 / 1.0166 | **1.0002 / 0.9912 / 1.0166** |
| cell-voxel (4,4,30) | 0.4301 / 0.4259 / 0.9241 | 0.5001 / 0.4956 / 0.9241 |
| identity | 0.1075 / 0.1065 / 0.0308 | 0.1250 / 0.1239 / 0.0308 |
| inversion (÷8,÷8,÷33) | 0.0134 / 0.0133 / 0.0009 | 0.0156 / 0.0155 / 0.0009 |

NO gate verdict moves on any axis — the declared scaling stays the closest to 1 everywhere — so the reading
was 14.0 % low on x and y rather than wrong in kind, and the error ran in the PERMISSIVE direction: the
as-built gate accepted an x/y-only over-scale up to 1.550x where the corrected one accepts 1.332x (panel C2).
A UNIFORM over-scale binds on z and is refused at 1.03x under BOTH denominators, so the null rows never moved
that boundary at all. `soma_bounds` now skips rows whose three coordinates are all zero, on the ROW's own
coordinates and NEVER on `agglo_seg == '0'` — that id also carries 1,083 somata with real locations, and the
id rule would discard measurements to drop a placeholder. It is not pushed up into `load_cells`: gate (d)'s
pin counts all 49,377 rows. ALSO DISCLOSED, and not worked around: `artifact_meta` carries
`unit_extent_ratios`, so this is not confined to a gate reading — the NEXT legitimate re-ingest will write the
corrected triples into the artifact meta and therefore produce a DIFFERENT artifact sha256 than the pinned
`a1d0491a…`. Nothing is re-run in this fold and no artifact byte changes: the offline self-check's gate (f)
synthetic artifact sha256 is byte-identical before and after (`f0ec156d02bb3303…`), because the 31-row
self-check fixture carries no null-coordinate row. SEPARATELY DISCLOSED and deliberately NOT changed: gate
(c)'s numerator is the JOINED-SYNAPSE cloud while its denominator is the FULL soma cloud, two different
populations. Swapping in the joined-soma denominator now would redefine a pre-registered gate after seeing its
number, and it changes nothing anyway — that variant reads 1.0478 / 1.0318 / 1.0168, still closest to 1 on
every axis (panel C2).

*Appended 2026-09-09 (panel fold, C10) — "over-estimating by 15.6 %" above has an unstated referent.* The
paragraph names the FINAL stage-B run's 840.66 s and never names the FIRST stage-B run's 845.78 s, which
appears only in the frozen 2.9 — so the 15.6 % cannot be reconstructed from the numbers the paragraph itself
puts in front of the reader. In full: stage A's 206.06 s projected stage B at 977.3 s, over-estimating the
FIRST stage-B run's 845.78 s by 15.6 % and THIS re-run's 840.66 s by 16.3 %. No basis reproduces 15.6 %
against 840.66 s (shards 15.55 / 16.26, records 15.72 / 16.42, bytes 15.72 / 16.42, the under-estimate framing
13.98 — panel C10), so the figure uniquely identifies the first, refused stage-B run. 2.9's "16.3 min" is a
MINUTES figure, correct and frozen, and must not acquire a percent sibling. This is the ONE finding of the 25
whose lens-proposed fix survived refutation; it is appended rather than substituted into the sentence, per the
fold's rule that only 3.5(3) is rewritten in place. The fold of this correction is itself unreviewed text — it
was battery-verified only.

### 3.3 The verdict, by the rule of record

Every gate of 2.3 as amended by 2.9 PASSED on the full 166-shard export; the artifact
`data/h01/h01_c3_graph.npz` (421,421 bytes, sha256 `a1d0491ac486bf194a479a8c6aab419df90c7f05d2ba643d6b16ed3c3d4a0549`)
is written and pinned in `tools/h01_c3_graph_pin.json` beside both refused readings, the harness sha256s and
the git head. There is no WIN/LOSS here (2.7): the row's deliverable is an artifact whose gates all pass, and
they do. The population: 14,215 signed units (9,793 excitatory, 4,422 inhibitory; 362 agglo ids excluded —
see 3.5) in seven bands — Layer 1 527, Layer 2 4,709, Layer 3 2,946, Layer 4 2,509, Layer 5 2,015, Layer 6
1,040, White matter 469 (against `cortical_prior`'s published column 540 / 4,469 / 2,851 / 2,998 / 2,092 /
883 / 498; the L4 deficit and L6 surplus are the pinned adjacent-layer confusions of `layer_of`) — 70,001 edges
aggregating 104,959 both-ends-signed synapses; `p_parts` 1, `frac_bits` 12, delay 1. Resident bytes:
`KernelState.to_blob()` 301,027; the `Synapses` bundles 1,120,016 bytes of post/qweight arrays across 10,606
bundles (2,857,424 with the Python dict) — about 3.16 MB for the whole population.

*Appended 2026-09-09 (panel fold, C7) — the resident-bytes sentence above mixes two bases, and its "about
3.16 MB for the whole population" is WITHDRAWN.* It counted the ndarray payloads at `nbytes` but the Python
dict and its tuples at `sys.getsizeof`, and then dropped the ndarray objects' OWN headers — which on this
population are the largest of the three terms. The label is "Resident bytes:", so this is not a statable
alternative basis. Re-measured in this fold on the pinned artifact: the bundles hold **1,120,016 B** of
post/qweight data (`nbytes`); the Python containers around them cost **1,737,408 B** (dict table 294,992 +
10,606 3-tuple keys 763,632 + 10,606 2-tuple values 678,784); and the 21,212 one-dimensional owning arrays
cost a further **2,715,136 B** of object headers (128 B each, and mean payload is 52.8 B per array, so at this
bundle count the header outweighs the data it wraps by 2.4x). On named bases: the bundles are **1.12 MB as
data** and **5.57 MB as live objects**; the whole population is **~1.42 MB serialized** (`KernelState.to_blob()`
301,027 + the 1,120,016 payload) or **~5.87 MB resident-as-objects** — and that second figure is a FLOOR,
since per-allocation overhead across 21,212 tiny buffers is uncounted. 3.5(9) is NOT re-stated against a
live-heap total: its claim is "16x high FOR THE STATE ARRAY", which is blob-to-blob (T065's regression of BLOB
bytes on units, 302.01 B/unit + 5,195.57, gives 4,953,327 B at 16,384 units against a measured 301,027 at
14,215 = 16.45x) and the bundle accounting does not reach it; on T065's own serialized basis this population
is ~1.42 MB, far BELOW its 5.0 MB, so that item's direction is unchanged and strengthened. One line worth
carrying to any later paging row: at 10,606 bundles the per-object overhead is 4.4x the data, so a
bundle-count-driven memory projection must be made on the object basis, never on `nbytes`.

*Appended 2026-09-09 (panel fold, C14) — the per-band parenthetical above names the two deltas it can explain,
omits the largest of the ones it drops, and compares against a column built over a different population.*
`cortical_prior`'s published column 540 / 4,469 / 2,851 / 2,998 / 2,092 / 883 / 498 is KEPT — frozen 2.5
pre-registers the comparison beside its 14,331 and the column is that module's own pinned provenance — but it
is over the e/i CSV's 14,331 signed neurons, a STRICT SUPERSET of this population by 116 ids (re-measured in
this fold: population 14,215, published ids 14,331, population − published = 0, published − population = 116),
so it contributes at most 28 to any one band. ON THE SAME 14,215 ids the CSV's own labels give 532 / 4,441 /
2,835 / 2,975 / 2,074 / 872 / 486, and the full delta vector is L1 −5, **L2 +268**, L3 +111, L4 −466, L5 −59,
L6 +168, WM −17: the omitted Layer 2 surplus is LARGER than the Layer 6 surplus the sentence does name, and
`placement.py`'s own pinned top confusions (L4→L3 971, L5→L6 537, L3→L2 501, over all 49,377) list exactly
three of which the sentence explains two and drops the third — which is that Layer 2 surplus. All of it is
adjacent-layer confusion in `layer_of` and it closes EXACTLY, with nothing left over: the confusion matrix on
this population has only SIX off-diagonal cells — L4→L3 374, L3→L2 263, L5→L6 151, L4→L5 92, WM→L6 17,
L1→L2 5 — so L4's deficit is exactly 374 + 92, L2's surplus exactly 263 + 5 (not 263 alone), L6's exactly
151 + 17, L3's +111 exactly 374 − 263. Agreement on this population is 13,313 / 14,215 = 0.9365; the pinned
0.9195 is over all 49,377 somata, which gate (d) reproduces exactly at 45,404 / 49,377. The record's
attribution above is directionally true but never exact against the published column — the L4 deficit there is
−489 while the actual outflow is 466, the 23 gap being the population difference.

### 3.4 The disposition applied

The T062 row's four named items: the join gate (b), the nm-unit gate (c), the layer computed (d, via
`placement.layer_of`, validated against the CSV column at the pinned 0.9195), c3 pinned (the manifest, the
schema hash, the five SOURCES sha256-verified), CC-BY on PROVENANCE — `brain_telemetry.PROVENANCE` has no row
that derives from the prior or placement (measured: the only "placement" matches are `displacement`, the arm
end-effector), so the licence sentence and FR-007 live in the module's `ATTRIBUTION` constant, in the
artifact meta and in the manifest, as 2.1 directs when no telemetry row exists. `tests/test_placement_charge_null.py`
stays green: no live reader was added. The kernel's two `wire` exemptions are untouched: this population is
off the live tick and its consumer is T063.

*Appended 2026-09-09 (panel fold, C20) — the location list above is wrong on one of its three conjuncts, and
it attributes a three-place list to one directive that is really two.* Re-measured in this fold:
`tools/h01_c3_shards.json` carries the licence sentence with its URL — which is what 2.1's DELIVERABLES bullet
asks of the manifest — and carries FR-007 in NO form at all ('fr-007' 0, 'fr_007' 0, 'not a measurement' 0,
'this brain' 0 over the whole file; its top-level keys hold no attribution key). The licence sentence AND
FR-007 live in `cortex/h01_ingest.ATTRIBUTION` and in the artifact meta — the full FR-007 sentence in
`meta["attribution"]`, tagged again by `meta["not_a_measurement_of_this_brain"]` = 'FR-007' — which is what
2.1's PROVENANCE bullet directs when no telemetry row derives from the prior (re-measured:
`brain_telemetry.PROVENANCE` 36 rows, prior-derived rows none). `tools/h01_c3_shards.json` is left
BYTE-IDENTICAL rather than gaining an FR-007 key: its sha256 `8142b747…` is the pinned `harness_sha256` in ALL
THREE stage entries of `tools/h01_c3_graph_pin.json`, including the two REFUSED stages, so editing it would
silently falsify the provenance of readings already in the record of a red — and it is a file whose
`listing_method` declares its fields to be the API's own, verbatim. `tools/h01_c3_graph_pin.json` carries no
licence or attribution field at all (top-level keys `latest` / `stages` / `task`) and the claim is not
extended to it. The over-claim was caught by review, in the fold, with no code and no artifact change. The two
acceptance sites now carry dated notes as well (panel C22, `spec.md` FR-022 and `quickstart.md` S2 step 1):
the FR-007 they mean is `specs/011-h01-neurogenesis-prior/spec.md`'s, NOT specs/022's own FR-007, which is the
RNG-stream-position requirement and a different rule entirely.

### 3.5 Deviations from the pre-registration, each with its reason (dated 2026-09-09)

1. Gate (e) as pre-registered was the instrument's defect (2.9): retired, replaced by (e′) before the re-run;
   its refused readings stand in 3.2 and in the pin.
2. The population admits an edge only when BOTH ends are signed: 104,959 synapses, not the 202,232 that 2.9's
   disclosure named as "what the population carries" — 411,255 of the 516,214 soma-to-soma synapses have a
   post that is not a signed neuron. A structural consequence of "units are the signed neurons", not a filter
   on the export's type; the number T063's null must be matched against.
3. `dale_signs` excludes 362 agglo ids; only 25 are excitatory-vs-inhibitory conflicts. 337 are a signed row
   co-existing with an UNSIGNED row for the same agglo id (14,792 signed rows over 14,577 distinct ids; 378 of
   those 14,577 carry more than one row and 44 more than one SIGNED row — and those 378 are exactly the 362
   excluded ids plus 16 population ids whose extra rows are all signed with one sign; the extreme is
   `agglo_seg` 0, one of the 25 conflicts, a single-`base_seg` bucket of 1,095 rows across nine cell types of
   which 133 are signed. Over the whole cell file — 49,377 rows, 45,994 distinct ids — 1,336 ids are
   multi-row). The docstring's "a node cannot be both" describes the 25;
   the 337 are a different claim (a segment recorded as a neuron and as something else) and get their own
   disposition in T063's brief: excluded, reported, never signed by default.
   *Parenthetical REWRITTEN IN PLACE 2026-09-09 (panel fold, C5, coordinated with C4). It is the only sentence
   in this record the fold rewrites rather than annotates, because a reference-class error inside a
   parenthetical cannot be repaired by a sentence placed after it. As published it read "one id carries 1,095
   rows; 1,336 carry more than one", which switched population mid-parenthetical: 1,336 is over the whole
   file's 45,994 ids, not over the 14,577 signed ids the sentence is describing, so a reader carrying the
   signed set forward got 9.2 % multi-row where the truth is 2.6 % (378/14,577) and 0.3 % multi-SIGNED-row
   (44/14,577) — a 3.5x overstatement of the described set. It also placed `agglo_seg` 0 beside the 337 it was
   illustrating, when 0 is one of the 25 CONFLICTS (133 of its 1,095 rows are signed and both signs are
   present). The 378 are now tied to the exclusion they explain, which is what makes the item cohere: multi-row
   structure IS the exclusion mechanism, and only 16 multi-row signed ids survive into the 14,215. Every other
   number in the item reproduced exactly on two instruments that fail differently — the panel's set-based
   classifier at the consumer and a Counter-based probe that never builds the id→rows grouping (panel C5).*
   *Appended 2026-09-09 (panel fold, C19): `dale_signs`' docstring said the disagreeing ids are "excluded too
   and counted" and glossed all 362 as "a node cannot be both". NOTHING in the module counts them — neither
   call site tallies exclusions, `_disclose` returns five numbers and no exclusion count, and 362 appears
   nowhere in the module or in `tools/h01_c3_graph_pin.json`; the 362 in this record came from an out-of-band
   probe, not from the instrument the docstring described. The docstring now states the CODE's rule — an id is
   excluded whenever its records do not agree on a single sign, either shape — with no counts and no forward
   reference. The split stays here, where it is dated and re-derivable, rather than being pinned unguarded in a
   general function's docstring where a re-fetch of the cell file would make it stale in silence.*
4. 2.3(d)'s red was written as "the agreement collapses"; measured, a soma genuinely in Layer 1 agrees with
   the nm collapse by accident, so the claim is "every soma lands in Layer 1", and the self-check says so.
5. 2.1 asked the pin to hold "every gate reading"; on a refusal the per-shard md5s are recorded as verified
   against the manifest rather than restated (they are the manifest's own values).
6. The order-independence arm on the real data ran on the first 8 shards, not all 166 (two full re-fetches
   would be 28 minutes for a hash the fixture arm already checks on every invocation); said so in 3.2.
7. `p_parts = 1` from `tools/kernel_partition_baseline.json`: the seeded population is single-partition, so a
   green tick here says nothing about the kernel's partitioned delivery path.
8. Gate (h) licenses "connected", not "self-sustaining": the seeded cascade dies by tick 4 (142, 71, 6, 1, 0).
   T063's arms use the identical drive and report the same five numbers.
   *Appended 2026-09-09 (panel fold, C3): "connected" is used loosely here and is never quantified anywhere in
   this record. Measured on the pinned artifact in this fold — 11,227 of the 14,215 units have at least one
   in-edge, 2,988 have NONE, 3,609 have no out-edge, 1,702 have neither, and there are no self-edges. What
   gate (h) licenses is that A SIGNAL PROPAGATED, never that the graph is connected: 2,988 units can never be
   recruited by the seed and 1,702 can never participate at all. Carried into T063 as a fact about the seed.*
   *Appended 2026-09-09 (panel fold, C13): the handoff sentence above hands T063 the run's five numbers, and
   those are the best-case draw of an UNREGISTERED selection (3.1 now discloses it). What T063 owes is a
   PRE-REGISTERED drive rule, applied per arm on that arm's OWN graph, with the cascade reported under both a
   degree-selected and an unbiased drive. This record deliberately does NOT name a random drive as T063's
   drive: a fixed random index set is not neutral under rewiring either, a degree rule recomputed per arm is
   degree-matched by construction, and a null's design is not settled in the row that supplies its seed.*
9. T065's "~5.0 MB at 16k" projection is 16× high for the state array (301,027 bytes at 14,215 units); the
   mass it lacked is the synapse table. The honest resident figure for this population is ~3.16 MB; the
   projection is corrected by this reading, not re-derived.
10. P2 debt named, not silently left: `reasoning/web_ingest_gate.py::fetch_binary` already streams a binary
    through the same firewall; `fetch_bytes`'s `sink=` is the capability ported first so that module can
    retire onto it, not retired here because a CORE self-check must not depend on the unregistered
    `reasoning/` tier and that module's own check performs a ~15 MB network fetch.
11. Stage A's artifact does not exist (stage A refused); the plan's "cheapest slice first" was honoured as a
    run, and the final artifact is stage B's re-run — the pin names both refused stages and the re-run by key.

*Items 12–21 were added by the panel fold on 2026-09-09, after the record commit `c34a328`. Items 1–11 above
are as pushed and are corrected only by the dated lines appended under them.*

12. **(2026-09-09, panel C1 — a CORE self-check was RED at the record commit and this section did not say so.)**
    The row's new module put a symbol named `drive_train` into `substrate/gradient_claim_gate.py`'s GOVERNED
    census, where the verb `train` reads as a learning rule. It is gate (f)'s byte-form determinism helper: it
    has exactly one call site, inside gate (f)'s determinism arm, it returns the concatenated byte-form of the
    tick outputs, and gate (h)'s liveness driver is a different function (`propagates`), which the census does
    not enumerate. The gate, which `run_selfchecks` invokes in its asserting BARE form, read `UNDETERMINABLE=1
    governed against a ratchet baseline of 0` and exited rc=1 at `c34a328`; a parent-tree control at `3f29d02`
    reads 0 governed / 48 all against the head's 1 / 49, so the row is ARC-INTRODUCED, re-derived rather than
    quoted. Declared `NOT_A_RULE` at module scope in the in-tree convention, placed immediately after the last
    top-level import (the same declaration inserted after the module docstring fires ruff E402 fourteen times
    — the placement is load-bearing, and the AST-only overlay the finding used could not have seen it).
    `tools/gradient_claim_baseline.json` is NOT ratcheted: the contract is "counts may only fall", and
    `--update-baseline` here would launder an arc-introduced red into the floor. After the declaration:
    `UNDETERMINABLE 0 governed / 48 all (SATISFIED)`, rc=0, with `rules seen` unchanged at 169 governed / 251
    all and MISSING / PHANTOM / unclassified / parse failures all 0 — a moved `rules_seen` would have meant the
    edit did something other than declare. 3.3's "every gate PASSED" was scoped to 2.3's ingest gates (a)–(h)
    and never covered the repo's registered self-checks; this is that gap, named.

13. **(2026-09-09, panel C6 — an OWED RED, named as future work rather than left as an option.)** No committed
    red exercises any behaviour inside `substrate/web_ingest.fetch_bytes`' body (3.1 now carries the six
    mutation arms and the measured evidence). What is owed is a fake-opener harness inside `web_ingest.main()`
    that goes RED under all six mutations and GREEN on the real code, landed as its own change under the full
    per-arc gate with its own mutation verification. It is NOT landed in this fold: a harness bolted into a
    CORE security module's self-check during a review is unreviewed code folded in from a review, and it owes
    its own control. Nothing about the download path's behaviour is claimed by this row.

14. **(2026-09-09, panel C3 — 2.2's "a unit's summed |input| is 1 by construction" is universally quantified
    over units and false for 21 % of them.)** It holds for the 11,227 units with at least one in-edge (max
    |1 − Σ|w|| = 1.110e-15) and is exactly 0 for the 2,988 with none. The wording is corrected HERE and not in
    the frozen 2.2, and the identical sentence in `artifact_meta["weight"]` is corrected at the next legitimate
    re-ingest, where the artifact sha256 and the pin change together in one commit — editing it now would
    change the .npz bytes and desynchronise a pin whose `harness_sha256` for this module matched the tree at
    the record commit. The self-check's fan-in block and the matching test were BOTH keyed on `post_idx`, so a
    unit with no in-edge was structurally invisible to them. The TEST now PARTITIONS the whole unit index —
    |Σ|w| − 1| ≤ 1e-9 for every unit with an in-edge, Σ|w| == 0.0 for every unit without, both counts reported
    — and never DISJOINS: "1.0 or 0.0" would be strictly WEAKER than the assertion it replaced, admitting a
    unit that has in-edges and sums to 0.0, which is the case `fan_in_shares`' own `where=denom > 0` guard
    exists to produce. Its red is CONSTRUCTED (the synthetic fixture has no in-edge-less unit) and
    mutation-proved: with all weights zeroed the partition goes red where the disjunction would have passed.
    DELIBERATELY NOT CHANGED in this fold: the module self-check's own fan-in block shares the blind spot; the
    fold's adjudication scoped the code touch to the test, so that site is owed. And the arm is FIXTURE-only
    either way — the real artifact's 2,988 has no in-command red until a re-ingest records the four counts in
    `tools/h01_c3_graph_pin.json`.

15. **(2026-09-09, panel C4 — gate (d)'s 45,404 / 49,377 is an IDENTITY check and stays exactly that; as an
    ACCURACY reading it is depressed, and nothing is excluded from the gate.)** 2.3(d) frames it as
    reproduction — the P2 evidence that this is the same rule on the same bytes — and it reproduces exactly.
    But 3.4 calls it "validated against the CSV column at the pinned 0.9195" and `placement.py` calls the same
    number "the accuracy bound for anything derived from placement", and as an accuracy reading it is scored
    per-SOMA against a per-SEGMENT label: 1,819 of the 49,377 somata carry one of EIGHT AGGREGATE segment ids —
    '0' with 1,095 rows and the seven round ids 100000000000–106000000000 with 724 between them (52 / 100 / 96
    / 121 / 159 / 173 / 23) — every one of the eight carrying 0 excitatory and 0 inhibitory synapses in the CSV
    and a single arbitrary layer label, and all 1,819 scored against one CSV row each. Re-measured in this
    fold: those 1,819 rows contribute 1,544 of the 3,973 disagreements (38.9 %), and over the remaining 47,558
    somata the rule agrees on 45,129 = **0.9489**. Both readings are recorded; the gate excludes nothing and
    keeps its exact refusal. The intermediate 45,266 / 48,282 = 0.9375 (dropping only id '0') is NOT adopted —
    it still contains 724 per-segment joins and under-reports by a further 1.1 points, which would be the same
    defect committed a second time. The exclusion is principled rather than prior-flattering: ordinary
    assigned multi-soma ids (1,328 ids, 2,900 rows) agree at 2,741 / 2,900 = 0.945, ABOVE the overall rate, so
    genuine multi-soma segments are not the problem. OPEN FINDING, filed rather than silently left:
    `placement.py`'s neighbouring sentence "Every disagreement is an ADJACENT layer" is false as measured —
    1,277 of the 3,973 disagreements are non-adjacent (519 of 3,016 after dropping the buckets), the largest
    being Layer 3 → White matter 315 and, from id '0' alone, Layer 1 → White matter 246. The fold's
    adjudication made C4 record-only, so neither `placement.py` nor `cortical_prior.py` is edited here and
    `LAYER_RULE_AGREEMENT = 0.9195` stands (two self-checks and a test assert on it); correcting those two
    comment blocks is owed work with its own gate.

    2026-09-10: the parenthetical '519 of 3,016 after dropping the buckets' uses 'buckets' for the id-'0'-only
    cut -- by this item's own figures 48,282 - 45,266 = 3,016, whereas dropping all eight aggregate ids leaves
    47,558 - 45,129 = 3,973 - 1,544 = 2,429 disagreements (the seven-round-ids-only cut gives 3,386). 519 is
    therefore the non-adjacent count on the id-'0'-only cut, by arithmetic, not re-measured; the eight-id
    non-adjacent count is unmeasured. No number changes. (Appended by the T071 panel fold; the item above is
    untouched.)

16. **(2026-09-09, panel C12 — gate (e′) is pre-registered relative to the re-run but SELECTED ON A POSITIVE
    PILOT, and 3.3's undifferentiated "every gate PASSED" should be read with that asymmetry.)** (e′)'s
    statistic was first read as a diagnostic on shard 0 — r 0.494 over 9,716 post segments, quoted in 2.9 item
    (3) — before 2.9, in the SAME commit, pre-registered the rule that governs it (`cd29a1b` 16:05:53, after
    both refused stages at 15:38:52 and 15:59:18). The fact is disclosed twice in 2.9; the CONSEQUENCE was
    never drawn, and this section exists to draw it. Its real-data reading was known to sit far above any
    plausible permutation band before the rule existed, so its PASS carries less evidential weight than gates
    (a)–(d), which were fixed at `5ebdf7c` before any real reading. What the re-run ADDS that the pilot did not
    have: the 1,000-permutation band and its bar (0.0089), the full-export scale (35,374 joined post cells
    against 9,716), and the stability of r under a 3.6x larger join (0.494 → 0.4527). The gate is UNCHANGED
    and is not demoted: `check_ei` still raises `IngestRefused`, its shuffled-label red is reachable and green,
    and T063 re-runs it as written on any new artifact.

17. **(2026-09-09, panel C15 — 2.9's "The export's `type` is used for NOTHING else" and the artifact meta's
    "used by gate (e′) and by nothing else" should both read "DECIDES nothing outside gate (e′)".)** That is
    the wording the same artifact's own disclosures block already uses. The export's type has two further,
    non-deciding uses, both pre-registered: the pre-side disclosure (`signed_pre_export_type_*`, 60,949 /
    202,232), and the per-edge tallies `n_exc` / `n_inh` of 2.2, which are CARRIED IN THE ARTIFACT (31,934 and
    73,025 against `n_syn` 104,959, so the type accounts for every synapse in the population) and are read by
    no weight, sign or edge-admission path — their only consumer in the tree is one inequality assertion in
    `tests/test_h01_ingest.py`. 2.2 pre-registers those counts verbatim, so two paragraphs of the same frozen
    block disagreed. Both frozen sentences are LEFT AS WRITTEN: 2.9 is pre-registration, and editing the meta
    string would change the .npz bytes and invalidate the pinned artifact sha256 `a1d0491a…` and the pin's
    `harness_sha256` — with no test anywhere asserting either, so the break would be silent. The meta string
    rides the next legitimate re-run and re-pin, never a standalone edit.

18. **(2026-09-09, panel C17 — 2.9's central diagnostic, the 19x pre/post join asymmetry, reaches neither this
    section nor the pin.)** "174,473 post-end hits per shard against 9,170 pre-end" was measured on SHARD 0
    during the untracked scratchpad orientation. It is NOT among the pinned full-export disclosures — the pin
    carries `pre_end_soma_hits`, `pre_end_soma_hits_non_neuronal` and `pre_end_non_neuronal_fraction` with no
    post-end analogue — and 2.9 pre-registered exactly one pre-end disclosure, which IS in 3.2, in 3.6 and in
    the pin, so no pre-registered number is missing. The shard-0 pre-end reading is consistent with what is
    pinned: 1,494,143 over 166 shards is 9,001 per shard against shard 0's 9,170, and 174,473 / 9,170 = 19.03.
    The same sentence is carried in the shipped rationale at `cortex/h01_ingest.py::post_excitatory_fractions`,
    which is the one site where the diagnostic actually decides behaviour (the anchoring of instrument 1). The
    pin is MACHINE-WRITTEN and is not hand-edited: it gains a post-end count only at a legitimate re-ingest,
    and when it does, that counter must be a NEW `post_hits` gated only on `post_n in agglo`, mirroring
    `pre_hits` exactly — deriving it from `post_type` would be type-filtered where the pre side is not, i.e.
    two readings that are not readings of one quantity, which is the defect 2.9 exists to record. The
    full-export ratio a re-run produces is a different scale from 2.9's shard-0 19x and must be labelled so.

19. **(2026-09-09, panel C9 — two measured territory numbers in the frozen sections are wrong, corrected here
    because §1 and §2 are never edited.)** 1.1's "197.3–198.0 MB each" is false at the top end: the pinned
    manifest gives per-shard sizes of 197,344,616–198,456,495 bytes (197.34–198.46 MB) and 68 of the 166
    shards exceed 198.0 MB, so the stated ceiling sits just under the actual mean of 197.95 MB. It is not an
    MB/MiB slip — in MiB the range is 188.20–189.26. 2.4's "6,913 MB" is false: the lexicographic 35-shard
    prefix `export…000`…`…034` sums to 6,918,228,982 bytes = 6,918.2 MB (the manifest is already in
    lexicographic order, verified, so prefix-by-slice and prefix-by-sort agree byte for byte); 6,913 is
    35 x 197.5, a per-shard envelope multiplied out instead of the manifest summed — the same root cause as
    1.1's. Re-measured in this fold from `tools/h01_c3_shards.json`, which is the authority for both. 1.1's
    total, 32,859,876,087 bytes, is EXACT and unchanged, and 2.4's operative condition — "measured under
    7 GB" — holds on the corrected figure (6.9182 GB, 6.4431 GiB), so no gate, red, stage decision or verdict
    moves.

20. **(2026-09-09, panel C18 + C23 — the pin's `verdict` was written only on the refusal path.)** The two
    refused stages carry `verdict: "REFUSED"` beside `refusals` and `message`; the artifact-bearing re-run that
    `latest` names carries no `verdict` key at all, so the machine-readable record held a refusal marker and no
    counterpart. 2.1's enumeration of the pin's required contents never named a verdict field and 2.7 gives
    this row REFUSALS and no WIN/LOSS, so the asymmetry violated nothing; but the passing stage was inferable
    only from a non-null `artifact`/`artifact_sha256` plus the absence of `refusals`, and the pin's two test
    fixtures used "OK", a word the writer never emitted. The writer now emits `verdict: "PASSED"` on the
    success path, asserted by a test over the module's own dict literals (the success path is reachable only
    through a 166-shard network fetch), and the fixtures use the word the writer produces. THE EXISTING PIN
    FILE IS NOT HAND-EDITED: the `…#2` entry stays verdict-less, and only a later legitimate ingest writes one
    of its own — a field back-filled into a stage record is a disclosure no run produced. DISCLOSED as a
    deviation from the panel: BOTH refuters argued record-only and against the token itself, on the grounds
    that 2.7 disclaims a win vocabulary and that no sibling pin in `tools/` spells it "PASSED"; the fold's
    adjudication overrode them under three conditions, each met (no artifact byte changes, it is tested, the
    pin is not hand-edited). The word says only that no gate raised. The readings that justify it are the
    pin's `gates` block, and a later reader who quotes the token instead of the numbers is reading it wrong.

21. **(2026-09-09, panel C8 — `fetch_bytes`' `sink=` is ported but UNCHARGED, and this row's own ingest does
    not take that path.)** Item 10 above names the P2 debt and says the duplicate was not retired; what it does
    not say is that `sink=` has ZERO call sites in the tree and no test — `fetch_text`, `fetch_sources` and
    `_net_fetch` all call `fetch_bytes` without it, and `web_ingest.main()` asserts only the structural
    `fetch_text`-routes-through-`fetch_bytes` pin. So stage B's fetch path buffered each ~197 MB shard WHOLE
    across 8 threads and copied it once more on return (`bytes(buf)`), and it is `decode_shard` and
    `verify_md5` over that whole buffer that hash and decode it — the docstring's "that is how a 197 MB shard
    is hashed and decoded without a second copy" describes a path nothing takes. NOT rewired here, and
    deliberately: `one()` needs the whole `data` for `decode_shard`, `len(data)` and the progress reading, and
    `verify_md5` hashes the whole buffer, so streaming means rewriting the md5 verifier and the block decoder —
    an unverified refactor of the CORE ingest's network path, in an arc that cannot re-run it, for the same
    reason item 10 gives for not retiring `fetch_binary`. And the cheap version of that wiring,
    `sink=buf.extend`, buffers identically with one extra indirection: charge in name only, which is exactly
    what P3's "reachability is not charge" forbids. The honest discharge is the one already named — retire
    `reasoning/web_ingest_gate.fetch_binary` onto `sink=` in a row that can run its ~15 MB fetch. The fold's
    adjudication made this record-only, so the docstring at `substrate/web_ingest.py::fetch_bytes` and the
    self-check's printed line still describe the capability without saying no caller takes it; both are owed.

### 3.6 What this row does NOT claim

No learning, no drift, no "beats random" — T063's. Not the connectome: the c3 export's soma-to-soma
subgraph, whose pre-end identity is majority non-neuronal (0.5566) and whose export type disagrees with the
pre cell's class on 0.6986 of signed-pre synapses — the prior's measured quality, carried into T063 as a fact
about the seed, not corrected by filtering. Not a live reader: the population is off the tick, the kernel
exemptions stand, the charge-null test stays green. Not "16k": 14,215 signed units, measured. Not a claim
about timing: the decode rate is reported with its host condition and asserted on by nothing.

*Appended 2026-09-09 (panel fold, C11): gate (e′) is a NULL-COMPARISON, not an absolute bar — 2.3(e) refused
below a literal 0.5; (e′) refuses only below the permutation band's 95th percentile, measured here at 0.008881
over 35,374 cells, so at this n it would have admitted r = 0.01 with the same PASS (fed the real pinned band
with r replaced, 0.30 / 0.10 / 0.05 / 0.02 / 0.01 / 0.0089 ALL pass — panel C11). Beating that null requires
BOTH that the CSV's segment labels join correctly AND that the two E/I readings co-vary, and it separates
neither. The seed's E/I quality is therefore read from the measured r = 0.4527 against a 0.008881 bar, never
from "(e′) passed"; T063 quotes the pair, not the verdict. This is a change of KIND that 2.9 argues only one
side of — it retires (e) on the ANCHOR defect, demonstrated by three independent readings, and the move from a
binary agreement rate (chance 0.5) to a Pearson r (chance ~0.0089 at n = 35k) is a side effect, not a weakening
to pass. Nothing in this record is falsified by it and no verdict changes.*

*Appended 2026-09-09 (panel fold, C6): this row does NOT claim that the new download path's behaviour is
policed by any red. See 3.1 and 3.5(13).*

*Appended 2026-09-09 (panel fold, C3): this row does NOT claim the seeded graph is connected. Measured on the
pinned artifact — 11,227 of 14,215 units have an in-edge, 2,988 have none, 3,609 have no out-edge, 1,702 have
neither. Gate (h) licenses that a signal propagated. Carried into T063 as a fact about the seed: 2,988 units
can never be recruited by it and 1,702 can never participate at all.*

### 3.7 Panel fold (2026-09-09)

Three lenses (instrument, pre-registration-honesty, disposition) over the record head `c34a328`. **25 findings
stood — 1 BLOCKER, 5 MAJOR, 19 MINOR — and 24 of the 25 lens-proposed FIXES were themselves refuted;** only
C10's was adopted as filed. Each entry below names what the panel found, the refuter's verdict on the CLAIM,
what changed and where, and what was deliberately NOT changed. Every fix applied is the REFUTER's minimal fix,
not the lens's, with two disclosed exceptions (C18/C23 and C19) where the fold's adjudication overrode a
refuter and the entry says so. **This whole section is itself unreviewed text: it was battery-verified only.**

Numbers below are either re-measured in this fold (offline, on the sha256-pinned inputs, with no stage-A or
stage-B re-run and no artifact byte changed) or copied from the panel material with the finding cited.

1. **C1 (BLOCKER) — `gradient_claim_gate` was RED at the head on a symbol this arc introduced, and §3 never
   mentioned it.** REFUTER: confirmed, and strengthened with a parent-tree control (`3f29d02` reads 0 governed
   UNDETERMINABLE / 48 all against the head's 1 / 49) — but the lens's fix named the WRONG GATE and the WRONG
   RETURN VALUE, and its "module-scope" placement fires ruff E402 fourteen times. CHANGED:
   `cortex/h01_ingest.py::GRADIENT_CLAIM` — `drive_train` declared `NOT_A_RULE` with the refuter's corrected
   reason, immediately after the last top-level import; recorded as 3.5(12). NOT CHANGED:
   `tools/gradient_claim_baseline.json` — ratcheting would launder an arc-introduced red into the floor; and
   `drive_train` is not renamed and `train` is not dropped from the gate's verb list, either of which would be
   the real weakening.
2. **C2 (MAJOR) — gate (c)'s extent ratios are 14 % low on x and y: 12 null-coordinate rows sat in the soma
   denominator.** REFUTER: confirmed on two readings, with the finding's framing narrowed — the widening is
   x/y-only, since a uniform over-scale binds on z at 1.03x under both denominators. CHANGED:
   `cortex/h01_ingest.py::soma_bounds` skips rows whose three coordinates are all zero, on the ROW's
   coordinates; the corrected triples and the permissive direction are appended under 3.2. NOT CHANGED: the
   filter is NOT `agglo_seg == '0'` (that id carries 1,083 real somata) and is NOT pushed into `load_cells`
   (gate (d)'s pin counts all 49,377 rows); `tools/h01_c3_graph_pin.json` is not retyped and 3.2's as-run
   triples stand; the joined-soma denominator is disclosed, not swapped in. DISCLOSED AGAINST THE FOLD'S OWN
   PREMISE: `artifact_meta` carries `unit_extent_ratios`, so a future re-ingest writes a different artifact
   sha256; verified in this fold that the offline self-check's gate (f) synthetic sha256 is unchanged.
3. **C3 (MAJOR) — "a unit's summed |input| is 1 by construction" is false for 2,988 of 14,215 units, and the
   fan-in assertion cannot see them.** REFUTER: confirmed and STRENGTHENED (the self-check's block runs on the
   synthetic self-shard, so the claim has no red on the real population at all), but all three parts of the
   lens's fix were defective — the worst being that "assert 1.0 or 0.0" is strictly WEAKER than the assertion
   it replaces. CHANGED: `tests/test_h01_ingest.py::test_population_fan_in_partitions_the_whole_unit_index`
   PARTITIONS the whole index and reports both counts, with a constructed red and a mutation proof; recorded as
   3.5(14) with the four counts, appended to 3.5(8) and into 3.6. NOT CHANGED: frozen 2.2 and
   `artifact_meta["weight"]` — the meta string is corrected at the next legitimate re-ingest, where the
   artifact sha256 and the pin move together; and the self-check's own fan-in block, which shares the blind
   spot and is owed (the adjudication scoped the code touch to the test).
4. **C4 (MAJOR) — gate (d)'s 45,404 / 49,377 is presented as the layer rule's accuracy, but 1,095 denominator
   rows join on the null segment id '0'.** REFUTER: confirmed, and the lens's replacement figure (0.9375)
   commits the same defect a second time — SEVEN further aggregate ids carry 724 more rows with the same
   signature. CHANGED: 3.5(15) records both readings — the gate stays exactly 45,404 / 49,377 as an identity
   check, and over the 47,558 non-bucket somata the rule agrees on 45,129 = 0.9489. NOT CHANGED: nothing is
   excluded from the gate; 0.9375 is not adopted; `placement.LAYER_RULE_AGREEMENT` and `cortical_prior`'s
   `accuracy_bound` are untouched (the fold made this record-only), and their "every disagreement is an
   ADJACENT layer" claim is FILED as an open finding in 3.5(15) — measured, 1,277 of 3,973 are non-adjacent.
5. **C5 (MAJOR) — §3.5(3)'s "1,336 carry more than one" counts over 45,994 ids, not the 14,577 the sentence
   describes.** REFUTER: confirmed on two instruments that fail differently, and the lens's fix was a HALF
   correction that would have left a second misattribution in the same clause harder to catch. CHANGED: 3.5(3)'s
   parenthetical REWRITTEN IN PLACE, exactly as the refuter wrote it — the only in-place rewrite in this fold —
   with each number naming its reference class, `agglo_seg` 0 placed with the 25 conflicts it belongs to, and
   the 378 tied to the exclusion they explain. NOT CHANGED: 1,336 is kept and labelled rather than deleted; it
   was published, and labelling it is the honest repair.
6. **C6 (MAJOR) — the record is silent that `fetch_bytes`' cap and its `extra_check` calls are covered by no
   red.** REFUTER: confirmed and the NUMBER IS WRONG — TOO LOW. Six invariants are unpoliced, not three or
   four, and the sharpest arm is the one the lens did not name: nothing proves the validating redirect handler
   is INSTALLED in the opener. CHANGED: 3.1 carries all six arms, the structural ninth red,
   `web_ingest.main()`'s never invoking `fetch_bytes`, the discriminating "presence is not participation"
   sentence, and the fake-opener probe LABELLED a review-time measurement; 3.5(13) names the owed red as future
   work; one sentence mirrored into 3.6. NOT CHANGED: no code. A fake-opener harness landed inside a CORE
   security module's self-check from a review is unreviewed code that owes its own control.
7. **C7 (MINOR, filed MAJOR) — "about 3.16 MB" drops 2.72 MB of ndarray headers while counting the tuples'
   and the dict's.** REFUTER: the arithmetic is confirmed (delta +0 against the record's 2,857,424) but the
   finding's own severity case is REFUTED — it does not reverse 3.5(9), which is a blob-to-blob claim about the
   STATE ARRAY that the bundle accounting cannot touch. CHANGED: 3.3 carries every term on a named basis,
   re-measured in this fold, and the bare 3.16 MB is WITHDRAWN. NOT CHANGED: 3.5(9)'s "16x high for the state
   array" stands and is NOT re-stated against a live-heap total — doing so would set a live-object total
   against T065's serialized projection, which is the same mixed-basis move being repaired.
8. **C8 (MINOR) — `fetch_bytes(sink=...)` has zero callers and its docstring describes a path this ingest does
   not take.** REFUTER: the P3 dormancy IS disclosed twice (at the site and in 3.5(10)); what survives, and is
   undisclosed anywhere, is that the docstring's present-tense factual sentence is false of the shipped tree.
   CHANGED: 3.5(21) records the zero call sites, that stage B buffered each shard whole across 8 threads and
   copied it again on return, and why it is not rewired. NOT CHANGED: no code — the fold made this
   record-only, so `web_ingest.py`'s docstring and its self-check print are owed; and `_net_fetch` is NOT
   rewired, since `sink=buf.extend` would be charge in name only.
9. **C9 (MINOR) — two territory numbers are wrong: §1.1's shard-size range and §2.4's stage-A byte total.**
   REFUTER: confirmed on two code paths, and the lens's fix could only be executed by EDITING FROZEN §1 and
   §2 — which would falsify §3's own "nothing above this line was edited" assertion. CHANGED: 3.5(19) carries
   both corrections, re-measured in this fold from the pinned manifest. NOT CHANGED: §1.1 and §2.4 are
   byte-for-byte as pre-registered; §1.1's total is exact and §2.4's operative "under 7 GB" holds.
10. **C10 (MINOR) — "over-estimating by 15.6 %" is computed against the first stage-B wall, not the 840.66 s
    named in the same paragraph.** REFUTER: confirmed, and this is the ONE finding of 25 whose proposed fix
    introduces no defect — with two refinements: the wall must be QUOTED, not just the run named, and the edit
    must not touch 2.9. CHANGED: an appended dated line under 3.2 carries the referent-bearing form with both
    figures. NOT CHANGED: the sentence itself, per the fold's rule that only 3.5(3) is rewritten in place; and
    2.9's "16.3 min", a correct minutes figure that must not acquire a percent sibling.
11. **C11 (MINOR, filed MAJOR) — gate (e′) has no absolute bar; neither 2.9 nor §3 says the replacement
    narrowed the claim.** REFUTER: confirmed — at n = 35,374 the bar is ~1.645/sqrt(n) = 0.0089, and r = 0.01
    would have passed — but the lens's fix would have broken the artifact's provenance (a docstring edit
    changes the harness hash the pin names) and its proposed wording was wrong in the other direction, denying
    the E/I co-variation half. CHANGED: the refuter's dated line appended to 3.6, the section for warrant
    limits. NOT CHANGED: `check_ei`'s docstring — a prose sentence must not cost a stage-B re-run; and no
    3.5 row, because a change of kind inside pre-registered 2.9 is not a deviation of the RUN from it.
12. **C12 (MINOR, filed MAJOR) — the replacement gate was chosen after a pilot of its own statistic read
    0.494.** REFUTER: confirmed on the commit timeline, but the lens's closing directive was FALSE about the
    gate and would have weakened a live red — `check_ei` raises and its shuffled-label red is reachable and
    green (5 passed). CHANGED: 3.5(16), the refuter's version — the ordering, what the re-run adds that the
    pilot did not have, and the gate explicitly unchanged. NOT CHANGED: nothing that demotes (e′) to a
    "described property" for T063, which would post-hoc relabel a pre-registered refusal after seeing its
    number — the sin the finding itself names, run in reverse.
13. **C13 (MINOR, filed MAJOR) — gate (h)'s drive set is the highest-out-degree 1 %, never pre-registered.**
    REFUTER: the unregistered-and-undisclosed half stands; the "code's reason is REFUTED by measurement" half
    OVERSTATES (3,609 units really do have zero out-degree, so the docstring is literally accurate — it just
    does not BITE), and the lens's proposed record sentence publishes a range its own stated method does not
    reproduce. CHANGED: 3.1 carries the realisation paragraph and the post-hoc check with its date and method,
    re-measured in this fold — FIVE INDEPENDENT SEEDS give 153–161, not the one-stream 151–175; the zero-weight
    red stays red under a random drive; the amplitude is inert 1.0x–10.0x. 3.5(8) gains what T063 must
    PRE-REGISTER. NOT CHANGED: no 3.5 deviation row — the run did not deviate from 2.3(h), which never fixed
    the selection, and filing a SPECIFICATION gap as an EXECUTION deviation is itself a pre-registration
    honesty error. `drive_indices`' docstring is corrected in a separate commit of this fold.
14. **C14 (MINOR, filed MAJOR) — the per-band attribution omits the largest omitted delta (Layer 2) and
    compares against a column over a different population.** REFUTER: both substantive halves confirmed, but
    the lens's own headline is wrong (L4 −489 is the largest deviation and IS named; L2 is the largest
    OMITTED one) and its fix commits the same sin — "the L2 surplus is L3→L2 263" is short by the L1→L2 5, and
    it silently de-attributes L6. CHANGED: an appended dated line under 3.3 with the same-ids column, the full
    delta vector and the COMPLETE six-cell confusion accounting, re-measured in this fold. NOT CHANGED: the
    published column is KEPT — frozen 2.5 pre-registers the comparison beside its 14,331 and the column is
    `cortical_prior`'s own pinned provenance.
15. **C15 (MINOR) — "The export's `type` is used for NOTHING else" is contradicted by the artifact's per-edge
    `n_exc`/`n_inh`.** REFUTER: confirmed and worse than filed — a THIRD use (the pre-side disclosure) is
    named in the module's own constant docstring, and 2.2 and 2.9, two paragraphs of the same frozen block,
    contradict each other. CHANGED: 3.5(17), recording that both sentences should read "DECIDES nothing
    outside gate (e′)", with the counts and the sole consumer. NOT CHANGED: frozen 2.9 and the artifact meta
    string — editing the meta changes the .npz bytes and invalidates the pinned artifact sha256 and the pin's
    harness hash, with no test asserting either, so the break would be silent.
16. **C16 (MINOR) — `EI_SEED = 0` fixes gate (e′)'s permutation band and was not named in 2.9.** REFUTER:
    confirmed, and the lens's PRIMARY branch would have committed the very defect the finding is about —
    adding "seed 0" to 2.9's rule sentence edits frozen pre-registered text after its number is known.
    CHANGED: an appended dated line under 3.1's builder's-choices paragraph, naming the seed and what makes it
    inert. NOT CHANGED: 2.9's rule sentence; and the code comment's "pre-registered in 2.9" is NOT dropped —
    that clause attaches to the permutation null and to the two constants 2.9 does name, and deleting it would
    de-attribute them to fix an over-reach about the third.
17. **C17 (MINOR) — the amendment's central diagnostic, the 19x pre/post asymmetry, reaches neither §3 nor the
    pin.** REFUTER: stands at the sites named, with one conjunct REFUTED — "carried nowhere outside 2.9's
    prose" is false, since the shipped rationale at `h01_ingest.py::post_excitatory_fractions` carries the same
    sentence at the site where the diagnostic decides behaviour. CHANGED: 3.5(18), naming it a disclosed
    limitation with the consistency check against what IS pinned. NOT CHANGED: no code and no re-run; the pin
    is machine-written and gains a post-end count only at a legitimate re-ingest — and then from a NEW
    `post_hits` counter, never from the type-filtered `post_type`.
18. **C18 (MINOR) — the passing pin entry carries no `verdict` field.** REFUTER: the fact reproduces, but 2.1
    never listed `verdict` among the pin's required contents and 2.7 says the opposite of what the finding
    attributes to it, and the lens's fix would break the one property 2.1 designates this file as discharging.
    CHANGED: the pin writer now emits `verdict: "PASSED"` on success, with a test; recorded as 3.5(20).
    DEVIATION FROM THE REFUTER, DISCLOSED: the refuter said record-only and "never the word PASSED"; the
    fold's adjudication overrode under three conditions, each met. NOT CHANGED: `tools/h01_c3_graph_pin.json`
    is not hand-edited — the `…#2` entry stays verdict-less until a run writes its own.
19. **C19 (MINOR) — `dale_signs`' docstring still says "a node cannot be both" for all 362 exclusions.**
    REFUTER: confirmed on two instruments and BROADER than filed — the same sentence claims the exclusions are
    "counted" and nothing counts them; and the lens's fix bakes dataset-specific counts into a general
    function's docstring and cites a T063 brief that does not exist. CHANGED: the docstring now states the
    CODE's rule, with no counts and no forward reference, and "and counted" is dropped; recorded under 3.5(3).
    DEVIATION FROM THE FOLD'S ADJUDICATION, DISCLOSED: its parenthetical expected the 25/337 split IN the
    docstring; the refuter refused that and the refuter's minimal fix was applied. NOT CHANGED: `dale_signs`
    does not gain an exclusion tally — that is a code change with its own gate, not a docstring change.
20. **C20 (MINOR, filed MAJOR) — §3.4 says FR-007 lives in the manifest; the manifest carries no FR-007 in any
    form.** REFUTER: confirmed and re-verified, and the lens's first option (add an `fr_007` key) is
    DESTRUCTIVE — the manifest's sha256 is the pinned harness hash in all three stage entries, including the
    two refused stages. CHANGED: an appended dated line under 3.4 attributing each part to the directive that
    actually asked for it, re-measured in this fold. NOT CHANGED: `tools/h01_c3_shards.json` is byte-identical;
    the claim is not extended to the pin, which carries no attribution field at all.
21. **C21 (MINOR, filed MAJOR) — `tasks.md` still carried T062 as an unchecked row.** REFUTER: confirmed
    against the TREE rather than the record, and the row flip belongs to the RECORDS step by 2.8 and by both
    immediate precedents — but the lens's wording would have stripped the ordering fact that legitimises the
    gate retirement, and would have filed a MET pre-registered condition (CC-BY via 2.1's own branch) as a
    deviation. CHANGED: `specs/022-unified-brain-arc/tasks.md` T062 flipped to `[X]` with a dated DONE block in
    the T066/T067/T068 shape, carrying the ordering fact explicitly and the P2 debt pointer. NOT CHANGED: the
    original row text; and "3.16 MB" is not repeated there (see C7).
22. **C22 (MINOR, filed MAJOR) — two acceptance sites still state "attribution row on PROVENANCE" with no
    dated note.** REFUTER: confirmed, including that the paging half of the same quickstart section already
    carries that shape — but the lens's text would plant a bare "FR-007" inside specs/022, whose OWN FR-007 is
    the RNG-stream-position requirement. CHANGED: dated notes appended at `spec.md` FR-022 and `quickstart.md`
    S2 step 1, each qualifying FR-007 to specs/011 and stating the RUNNABLE check. NOT CHANGED: neither
    original sentence is edited; the claim is not extended to the pin.
23. **C23 (MINOR) — the pin never states that the passing stage passed, and the test's fixture uses a shape
    the writer never produces.** REFUTER: confirmed, with two overstatements removed (the KeyError has no
    reachable caller, and the pass row carries the full `gates` block) — and the lens's fix is INERT on the
    artifact it names and invites hand-editing a recorded reading. CHANGED: covered by C18's writer change;
    the two fixtures now use "PASSED"; a new test asserts the writer's vocabulary is exactly
    {REFUSED, PASSED} by AST, red-proved two ways. NOT CHANGED: the pin file; and the bracket reads at the
    fixtures are left strict rather than softened to `.get`.
24. **C24 (MINOR) — the REGIONS comment says "four gates"; the module ships eight.** REFUTER: confirmed on
    three instruments, and decisively — the comment landed in a commit whose own SUBJECT says "8 gates". The
    collision is what makes it more than a typo: "the four gates" is this repo's term of art for the CI
    quartet. CHANGED: `snn_data_lake/__init__.py`'s REGIONS row now reads "8 gates", the digit, so one string
    greps the count across every registration site. NOT CHANGED: no 3.5 row — the run did not deviate, all
    eight gates ran and 3.2 tables every one. 2.1's own "the four gates (2.3)" is the same wording slip and
    resolves on the page via its "(2.3)" citation; it is frozen text and is noted here only.
25. **C25 (MINOR) — the P2 debt on `web_ingest_gate.fetch_binary` is recorded only at the site that ported the
    capability.** REFUTER: the back-reference gap stands (the project annotates BOTH duplicates 3 of 3 times
    elsewhere), but two of the lens's three grounds are FALSE — the debt DOES carry a dated task id, and
    `docs/AUDIT_BACKLOG.md` is a closed 2026-07-28 register, not a live one; and `fetch_binary` has ZERO call
    sites, which the finding omits and which changes what retirement means. CHANGED: two lines in
    `reasoning/web_ingest_gate.py::WebIngestGate.fetch_binary`'s docstring, by SYMBOL and never by line
    number, naming the retirement target and the zero call sites; plus the pointer in T062's DONE note. NOT
    CHANGED: no new numbered task row (a reviewer does not expand a ratified plan, and `tasks.md`'s own "92
    tasks total" arithmetic would contradict itself), no `docs/AUDIT_BACKLOG` entry, and the debt's rationale
    is not restated at the gate site — `web_ingest.py` already carries it in full.
