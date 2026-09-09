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

### 3.4 The disposition applied

The T062 row's four named items: the join gate (b), the nm-unit gate (c), the layer computed (d, via
`placement.layer_of`, validated against the CSV column at the pinned 0.9195), c3 pinned (the manifest, the
schema hash, the five SOURCES sha256-verified), CC-BY on PROVENANCE — `brain_telemetry.PROVENANCE` has no row
that derives from the prior or placement (measured: the only "placement" matches are `displacement`, the arm
end-effector), so the licence sentence and FR-007 live in the module's `ATTRIBUTION` constant, in the
artifact meta and in the manifest, as 2.1 directs when no telemetry row exists. `tests/test_placement_charge_null.py`
stays green: no live reader was added. The kernel's two `wire` exemptions are untouched: this population is
off the live tick and its consumer is T063.

### 3.5 Deviations from the pre-registration, each with its reason (dated 2026-09-09)

1. Gate (e) as pre-registered was the instrument's defect (2.9): retired, replaced by (e′) before the re-run;
   its refused readings stand in 3.2 and in the pin.
2. The population admits an edge only when BOTH ends are signed: 104,959 synapses, not the 202,232 that 2.9's
   disclosure named as "what the population carries" — 411,255 of the 516,214 soma-to-soma synapses have a
   post that is not a signed neuron. A structural consequence of "units are the signed neurons", not a filter
   on the export's type; the number T063's null must be matched against.
3. `dale_signs` excludes 362 agglo ids; only 25 are excitatory-vs-inhibitory conflicts. 337 are a signed row
   co-existing with an UNSIGNED row for the same agglo id (14,792 signed rows over 14,577 distinct ids; one id
   carries 1,095 rows; 1,336 carry more than one). The docstring's "a node cannot be both" describes the 25;
   the 337 are a different claim (a segment recorded as a neuron and as something else) and get their own
   disposition in T063's brief: excluded, reported, never signed by default.
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
9. T065's "~5.0 MB at 16k" projection is 16× high for the state array (301,027 bytes at 14,215 units); the
   mass it lacked is the synapse table. The honest resident figure for this population is ~3.16 MB; the
   projection is corrected by this reading, not re-derived.
10. P2 debt named, not silently left: `reasoning/web_ingest_gate.py::fetch_binary` already streams a binary
    through the same firewall; `fetch_bytes`'s `sink=` is the capability ported first so that module can
    retire onto it, not retired here because a CORE self-check must not depend on the unregistered
    `reasoning/` tier and that module's own check performs a ~15 MB network fetch.
11. Stage A's artifact does not exist (stage A refused); the plan's "cheapest slice first" was honoured as a
    run, and the final artifact is stage B's re-run — the pin names both refused stages and the re-run by key.

### 3.6 What this row does NOT claim

No learning, no drift, no "beats random" — T063's. Not the connectome: the c3 export's soma-to-soma
subgraph, whose pre-end identity is majority non-neuronal (0.5566) and whose export type disagrees with the
pre cell's class on 0.6986 of signed-pre synapses — the prior's measured quality, carried into T063 as a fact
about the seed, not corrected by filtering. Not a live reader: the population is off the tick, the kernel
exemptions stand, the charge-null test stays green. Not "16k": 14,215 signed units, measured. Not a claim
about timing: the decode rate is reported with its host condition and asserted on by nothing.
