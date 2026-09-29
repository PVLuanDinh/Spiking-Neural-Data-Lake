"""H01 c3 synapse graph -> a beatable prior (specs/022 T062, FR-022). ONE ingest mechanism.

WHAT THIS ADDS AND WHAT IT REFUSES TO RE-DERIVE. `growth/cortical_prior.py` already holds the
soma / layer / sign constants with pinned byte counts and SHA-256s, and `growth/placement.py`
already implements H01's own `fix_layer_mem` layer rule. The audit's item (5) (a second H01
ingest mechanism with no P2 adjudication against `cortical_prior`) binds: everything either of
them holds is IMPORTED here, never recomputed. The genuinely new content is the c3 SYNAPSE
GRAPH -- the Avro export under `gs://h01-release/data/20210729/c3/synapses/exported/` -- which
neither module has ever seen.

THE ARTIFACT IS NOT LEARNED STATE (P4b). Nothing here writes to the lake. `cortical_prior`'s
docstring states the rule this module obeys: "The lake is what the brain LEARNED. This is what a
microscope saw." The graph lands in `data/h01/h01_c3_graph.npz` (gitignored, on D:) and is read
ONCE at construction of a population, never on a tick (specs/011 data-model.md 3).

FOUR GATES, EACH A REFUSAL WITH A CONSTRUCTED RED, and none carrying an invented threshold --
each compares the declared reading against its alternatives and requires the declared one to be
the extremum, so the number that decides is measured in the same pass as the number decided:
  (a) BYTES     -- streamed md5 == the manifest's, byte count == the manifest's size, and the
                   header's `avro.schema` sha256 == AVRO_SCHEMA_SHA256, checked BEFORE a record.
  (b) JOIN      -- both-ends-hit synapses counted under ALL FOUR pairings of {neuron_id,
                   base_neuron_id} x {agglo_seg, base_seg}; the declared `neuron_id <->
                   agglo_seg` must be the STRICT maximum. This is US3 Scenario 1's
                   "silently-empty-graph trap made loud": a refusal yields NO artifact, so there
                   is no verdict for it to hide.
  (c) UNITS     -- the joined synapses' locations converted under four candidate scalings; the
                   declared (8, 8, 33) nm/voxel must give extent ratios closest to 1 on EVERY
                   axis. The trap is real and 1000x: the export is in 8 nm voxels, the cell file
                   in nanometres, and `placement.layer_of` takes MICRONS.
  (d) LAYER     -- computed, not read (FR-022), and validated by REPRODUCING
                   `placement.LAYER_RULE_AGREEMENT`'s pinned 45,404 / 49,377 exactly. Exactness
                   is the point: it is the P2 evidence that this is the same rule on the same
                   bytes, which a "close enough" bar would not be.
  (e') E/I      -- two POST-ANCHORED instruments that fail differently: the export's own
                   per-synapse `type` aggregated per POST cell into that cell's excitatory INPUT
                   fraction, and upstream's OWN curated per-segment `excitatory synapses` /
                   `inhibitory synapses` counts in the e/i CSV. Their Pearson r over the joined
                   post cells must EXCEED the 95th percentile of the same r under 1,000
                   permutations of the CSV's segment labels; the constructed red is the shuffled
                   labels themselves. The bar is re-measured from the same data on every run, so
                   there is nothing in it to tune.

THIS MODULE ONCE CARRIED A DIFFERENT GATE (e) AND IT WAS THE INSTRUMENT'S DEFECT, not the data's.
It compared the PRE cell's Dale sign with the export's per-synapse `type` and refused the real
export at 0.3004 (35 shards) and 0.3014 (166 shards) -- a reading that moves 0.001 across a 4.7x
larger sample is a reading. Measured on shard 0 (998,235 records): by PRE cell type the export's
excitatory fraction does not separate excitatory from inhibitory classes at all (pyramidal 0.370,
spiny stellate 0.344, atypical spiny 0.338, interneuron 0.308) and 54.4 % of pre-end soma hits are
glia or vasculature, which cannot be presynaptic; by POST cell type it separates cleanly (0.639 /
0.672 / 0.630 / 0.538). `get_neuron_e_to_i_ratios.py` reads 2 = excitatory and 1 = inhibitory
correctly -- but it counts synapses RECEIVED by a neuron, a POST-anchored quantity, and the old
gate compared it to a PRE-anchored one. The replacement is pre-registered in the evidence file's
dated 2.9. P2: the old refusal is DELETED, not flagged off. What the pre side still owes is
DISCLOSURE, and `ingest` measures and prints five numbers it never filters on -- see `_disclose`.

THE DECODER IS STDLIB AND SCHEMA-SPECIFIC. No Avro reader is installed in `.venv-lake` and PyPI
serves no `cp314t` Windows wheel for `fastavro`. The container's codec is `null` (uncompressed),
so a block is a varint pair followed by records and a sync marker -- ~120 lines. The field ORDER
below is not a guess: it is the order in the header's own `avro.schema`, whose sha256 is pinned,
so a release that reorders two fields is refused rather than silently misread.

THE WRITER LIVES HERE TOO, and deliberately: `tests/test_h01_ingest.py` builds its shard fixtures
with `synthetic_shard` rather than carrying a second encoder that could drift from this decoder
(P2). The honest consequence is that a writer/decoder round trip cannot catch a bug they SHARE --
which is why the independent control is `tests/fixtures/h01_c3_shard0_head.avro`, real release
bytes whose first record is checked against values read by a different reader entirely (the JSON
twin: pre 5472528459, post 5414315245, type 2).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.h01_ingest        # offline gate
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.h01_ingest --ingest --shards 0:35

APPENDED 2026-09-10 (T063 panel of 2026-09-09), the text above kept VERBATIM (including line 1's
"a beatable prior" and `_disclose`'s "is the question", which is exactly the question T063
answered -- a module that rewrites its own premise destroys the record of what it used to claim):
T063 LANDED AND ITS ANSWER WAS NO. The campaign read the PLANTED control MET at t = +9.2601, so
the instrument is not blind and the verdict is read; conjunct (b) then read NOT MET at
t = -0.2277. Under 2.9's clause for that combination, `population()` LOSES its "prior" role and
is retained ONLY as `instruments/prior_drift_harness`'s FIXTURE -- the population an instrument
constructs and scores -- while the eight gates, the artifact and `cortical_prior` remain
REFERENCE DATA and are untouched. The reading is BOUNDED and this bound is 2.2's own: ONE task
(K = 8 drive classes decoded from a 4-dimensional projection) at the ONE capacity the guard
selected, and the null RETAINS H01's degrees, signs and weight magnitudes, so it cannot separate
"H01 contributes nothing" from "H01's useful information is exactly what both arms share".
Registration of record: `gates/dormancy_gate`'s h01_ingest row (371d607); evidence 3.4(4).
Not touched by any of it: `kernel_state`'s and `kernel_tick`'s wire rows (T063 2.8).

AND THE FIXTURE READING'S WARRANT IS AN OPENING, not a settled result: (c), the control that
licenses the whole verdict, was read against a RANDOM arm 0.62 held-out trials above chance -- a
shape `prior_drift_harness.synthetic_artifact`'s own floor rule rejects in the fixture ("a
positive control that beats a floor is a formality, not a reading") -- so the row is re-read at a
capacity where the base arms are above the floor before any of the three retirements (this
population's prior role, the learner, T071's "prior separation") is treated as settled.
"""

# distinct-from: sensorimotor/web_ingest.py -- the network boundary (SSRF firewall, redirect
#   re-validation, byte cap); this module CALLS it and adds no opener of its own.
# distinct-from: sensorimotor/ingest_guard.py -- the CONTENT boundary (licence deny-by-default,
#   injection flagging) over TEXT corpora; called here for the licence, not re-implemented.
# distinct-from: growth/cortical_prior.py -- the SOMA/layer/sign reference constants; imported,
#   never re-derived. distinct-from: growth/placement.py -- owns the layer rule and the Dale
#   assignment for GROWN units; its layer_of is imported and its agreement figure is the
#   validator here. distinct-from: substrate/data_quality.py::gate -- a Bronze->Gold row-level
#   quality gate over lake records; nothing to do with an external release's bytes.
#   distinct-from: gates/manifest_gate.py -- asserts the wiring manifest's port declarations;
#   "manifest" there is faculty ports, here it is a bucket listing. All sites read 2026-09-09.

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import struct
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

from snn_data_lake.growth.cortical_prior import H01_COMMIT, SIGNED_TYPES, SOURCES, UNSIGNED_TYPES
from snn_data_lake.growth.placement import (
    EXCITATORY,
    INHIBITORY,
    LAYER_NAMES,
    WHITE_MATTER,
    layer_of,
)

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "drive_train": (
        "NOT_A_RULE",
        "gate (f)'s determinism helper: drives 1 % of units for one tick and returns the "
        "concatenated BYTE-FORM of `ticks` tick outputs, which two constructions are compared "
        "for byte-equality. No weight is read or written and no error signal exists -- 'train' "
        "here is a SPIKE TRAIN, not training. (Gate (h)'s liveness driver is `propagates`, a "
        "separate function the census does not enumerate.)",
    )
}

REPO = Path(__file__).resolve().parents[3]
SHARD_MANIFEST = REPO / "tools" / "h01_c3_shards.json"
GRAPH_PIN = REPO / "tools" / "h01_c3_graph_pin.json"
DATA_DIR = REPO / "data" / "h01"
ARTIFACT = DATA_DIR / "h01_c3_graph.npz"
REAL_HEAD_FIXTURE = REPO / "tests" / "fixtures" / "h01_c3_shard0_head.avro"

#: raw.githubusercontent.com template for the five files `cortical_prior.SOURCES` pins.
SOURCES_URL = "https://raw.githubusercontent.com/ashapsoncoe/h01/{commit}/{name}"

#: sha256 of shard 0's header `avro.schema` bytes, measured 2026-09-09 from the release itself.
AVRO_SCHEMA_SHA256 = "4f09c112b46f1efd8608934f19e01a9aa716315b8646d3c424e9778507e9c5e6"

#: The licence identifier `substrate/ingest_guard.license_ok` is asked about before any use.
LICENSE_ID = "cc-by-4.0"
ATTRIBUTION = (
    "H01 (Shapson-Coe et al.), c3 agglomeration, synapse export "
    "gs://h01-release/data/20210729/c3/synapses/exported/ and the five files pinned by "
    "cortex/cortical_prior.SOURCES at commit " + H01_COMMIT + ". "
    "'All released datasets are licensed under a Creative Commons Attribution 4.0 License' "
    "(https://storage.googleapis.com/h01-release/data.html). CC-BY-4.0. "
    "FR-007: this is a measurement of a HUMAN brain that a microscope saw, NOT a measurement of "
    "this brain -- any quantity derived from it is a prior, never an observation of the running "
    "organism."
)

#: (b) the four pairings, and the declared one.
JOIN_PAIRINGS = (
    ("neuron_id", "agglo_seg"),
    ("neuron_id", "base_seg"),
    ("base_neuron_id", "agglo_seg"),
    ("base_neuron_id", "base_seg"),
)
DECLARED_JOIN = ("neuron_id", "agglo_seg")

#: (c) the declared voxel size and the three alternatives it must beat on every axis.
DECLARED_VOXEL_NM = (8, 8, 33)
DECLARED_SCALE = "declared_8_8_33"
CANDIDATE_SCALES = {
    DECLARED_SCALE: (8.0, 8.0, 33.0),
    "cell_voxel_4_4_30": (4.0, 4.0, 30.0),
    "identity": (1.0, 1.0, 1.0),
    "inversion": (1.0 / 8.0, 1.0 / 8.0, 1.0 / 33.0),
}

#: (d) `placement.LAYER_RULE_AGREEMENT` as the pair it was rounded from -- 45,404 of 49,377 soma
#: records agree with H01's own published `cortical layer`. Reproducing the PAIR, not the rounded
#: fraction, is what proves the same rule ran over the same bytes.
LAYER_AGREEMENT_PIN = (45404, 49377)

#: The export's own E/I code. Upstream counts a synapse `type` of 2 as excitatory and 1 as
#: inhibitory (`get_neuron_e_to_i_ratios.py`: `e_count = syn.count(2)`, `i_count = syn.count(1)`).
#: Used by (e') instrument 1 and by the pre-side DISCLOSURE, and by nothing else: no edge, node,
#: weight or sign in the artifact is decided by it.
SYN_TYPE_SIGN = {2: EXCITATORY, 1: INHIBITORY}
EXPORT_TYPE_E = 2
EXPORT_TYPE_I = 1

#: (e') the permutation null, pre-registered in 2.9. A fixed seed so the band is a reading and not
#: a draw, and a percentile rather than a number somebody chose.
EI_PERMUTATIONS = 1000
EI_PERCENTILE = 95.0
EI_SEED = 0

#: The seven laminar bands, in the order a population's faculties are laid out.
BANDS = (*LAYER_NAMES, WHITE_MATTER)

#: The 11 H01 cell-type literals, ordered so a type's int8 code is stable across rebuilds.
CELL_TYPES = tuple(
    sorted(
        {
            *SIGNED_TYPES["value"]["excitatory"],
            *SIGNED_TYPES["value"]["inhibitory"],
            *UNSIGNED_TYPES["value"],
        }
    )
)
_EXCITATORY_TYPES = frozenset(SIGNED_TYPES["value"]["excitatory"])
_INHIBITORY_TYPES = frozenset(SIGNED_TYPES["value"]["inhibitory"])

#: A NEURON of any sign, including the 1,275 "unclassified neuron" somata that carry no sign and
#: are therefore excluded from the population. The complement -- oligodendrocyte, microglia/opc,
#: astrocyte, blood vessel cell, c-shaped cell, unknown cell -- CANNOT be presynaptic, so the
#: fraction of pre-end soma hits landing there is a measure of the reconstruction's axon side and
#: is DISCLOSED. It is never a filter: 2.9 is explicit that selecting edges to match the prior
#: would be selecting the data to fit the seed.
NEURONAL_TYPES = frozenset({*_EXCITATORY_TYPES, *_INHIBITORY_TYPES, "unclassified neuron"})


def _lif_defaults():
    """Threshold, decay, reset and refractory BY IMPORT from the live equation owner (P1: cited,
    not invented). `cortex/lif_neuron.LIFNeuron` is the module `reservoir/kernel_tick` names as
    the owner of the equation its vectorized step implements."""
    from snn_data_lake.growth.lif_neuron import LIFNeuron

    lif = LIFNeuron()
    return {
        "thr": lif.v_threshold,
        "decay": lif.decay,
        "v_reset": lif.v_reset,
        "refrac_ticks": int(lif.refractory),
        "owner": "growth/lif_neuron.py::LIFNeuron (tau_m=10.0 -> decay=exp(-1/tau_m))",
    }


LIF_DEFAULTS = _lif_defaults()


# distinct-from: gates/gradient_claim_gate.py::_further_nulls -- publishes three nulls nothing
#   asserts on, inside its own verdict; it is a REPORTING FUNCTION of one gate, not a refusal type
#   carried across gates. distinct-from: gates/amdahl_gate.py::red_repeatability -- a
#   band-overlap red for a timing rig. distinct-from:
#   infra/brain_native_selfbuild.py::ThresholdRecalibrationSource -- a self-build source
#   that emits a source patch. distinct-from: reasoning/ar_spiking_lm_serve.py::
#   build_random_artifact and reasoning/eprop_lm_serve.py::build_random_artifact -- random-init
#   demo artifacts labelled trained=False. None of the five is an exception type, and none carries
#   a gate's readings past a refusal. All sites read 2026-09-09; the guard matched on the words
#   "REPORTED" and "nulls".
class IngestRefused(ValueError):
    """A gate refused. No artifact is written and the readings are reported -- 2.7's null clause:
    nothing is weakened to pass.

    `.readings` carries EVERY gate's measurement: the ones that passed, the one that refused, and
    the ones measured after it. A refusal IS the row's result when it happens, and a result that
    arrives without its evidence cannot be audited, argued with, or re-derived -- 2.7 says "the
    readings are recorded" in the same breath as "no artifact is written", and a
    first-refusal-wins raise honours only the second half."""

    def __init__(self, message, readings=None):
        super().__init__(message)
        self.readings = dict(readings or {})


# =============================================================================================
# The decoder (ported from the T062 probe, field order pinned by the header's avro.schema)
# =============================================================================================
class Reader:
    __slots__ = ("b", "p")

    def __init__(self, b, p=0):
        self.b = b
        self.p = p

    def long(self):
        b = self.b
        p = self.p
        shift = 0
        acc = 0
        while True:
            c = b[p]
            p += 1
            acc |= (c & 0x7F) << shift
            if c < 0x80:
                break
            shift += 7
        self.p = p
        return (acc >> 1) ^ -(acc & 1)

    def double(self):
        v = struct.unpack_from("<d", self.b, self.p)[0]
        self.p += 8
        return v

    def bytes_(self):
        n = self.long()
        v = self.b[self.p : self.p + n]
        self.p += n
        return v

    def string(self):
        return self.bytes_().decode("utf-8")


def parse_header(r):
    """(meta, sync) from an Avro OCF header; leaves `r.p` on the first block."""
    if r.b[:4] != b"Obj\x01":
        raise IngestRefused("not an Avro object container file (magic mismatch)")
    r.p = 4
    meta = {}
    while True:
        n = r.long()
        if n == 0:
            break
        if n < 0:
            n = -n
            r.long()  # block byte size, unused
        for _ in range(n):
            k = r.string()
            meta[k] = r.bytes_()
    sync = r.b[r.p : r.p + 16]
    r.p += 16
    return meta, sync


def verify_schema(meta):
    """(a) The header's `avro.schema` sha256, refused unless it equals the pinned value."""
    codec = meta.get("avro.codec", b"null")
    if codec not in (b"null", b""):
        raise IngestRefused(f"unsupported Avro codec {codec!r}: this decoder reads codec null only")
    got = hashlib.sha256(meta.get("avro.schema", b"")).hexdigest()
    if got != AVRO_SCHEMA_SHA256:
        raise IngestRefused(
            f"avro.schema sha256 {got} != pinned {AVRO_SCHEMA_SHA256} -- the release's record "
            "layout changed, so every field offset below is a guess. Refused before reading a "
            "single record."
        )
    return got


def _opt(r, fn):
    """A ["null", X] union: index 0 -> None, index 1 -> X."""
    idx = r.long()
    if idx == 0:
        return None
    if idx != 1:
        raise IngestRefused(f"union index {idx} outside [null, X]")
    return fn(r)


def _xyz(r):
    return (_opt(r, Reader.long), _opt(r, Reader.long), _opt(r, Reader.long))


def _bbox(r):
    _opt(r, _xyz)  # start
    _opt(r, _xyz)  # size
    _opt(r, Reader.string)  # description
    _opt(r, Reader.string)  # object_label (string INSIDE a site)
    return None


def _bbox_top(r):
    _opt(r, _xyz)
    _opt(r, _xyz)
    _opt(r, Reader.string)
    _opt(r, Reader.long)  # object_label is a long at top level, a string inside sites
    return None


def _site(r):
    """The 11 site fields in the pinned order. Every field is CONSUMED (the cursor is the format);
    only the three the join and the sign need are returned."""
    neuron_id = _opt(r, Reader.long)
    _opt(r, Reader.long)  # id
    _opt(r, Reader.long)  # num_voxels
    _opt(r, _xyz)  # centroid
    _opt(r, Reader.long)  # type (1 = pre-site, 2 = post-site; NOT the synapse type)
    _opt(r, Reader.long)  # subtype
    _opt(r, Reader.double)  # confidence
    _opt(r, _bbox)  # bounding_box
    _opt(r, Reader.bytes_)  # mask
    _opt(r, Reader.string)  # class_label
    base = _opt(r, Reader.string)  # base_neuron_id
    return neuron_id, base


class Record:
    """The six fields of a synapse this ingest reads. A tuple-like with names, so a field cannot
    be swapped by position in a caller."""

    __slots__ = ("pre_id", "pre_base", "post_id", "post_base", "location", "syn_type")

    def __init__(self, pre_id, pre_base, post_id, post_base, location, syn_type):
        self.pre_id = pre_id
        self.pre_base = pre_base
        self.post_id = post_id
        self.post_base = post_base
        self.location = location
        self.syn_type = syn_type

    def __repr__(self):
        return (
            f"Record(pre={self.pre_id}/{self.pre_base}, post={self.post_id}/{self.post_base}, "
            f"loc={self.location}, type={self.syn_type})"
        )


def read_record(r):
    pre_id, pre_base = _opt(r, _site) or (None, None)
    post_id, post_base = _opt(r, _site) or (None, None)
    loc = _opt(r, _xyz)
    syn_type = _opt(r, Reader.long)
    _opt(r, Reader.double)  # contact_area
    _opt(r, Reader.double)  # confidence
    _opt(r, _bbox_top)  # bounding_box
    return Record(pre_id, pre_base, post_id, post_base, loc, syn_type)


def iter_records(data):
    """Every record of a complete null-codec OCF. The schema is verified BEFORE the first one."""
    r = Reader(data)
    meta, sync = parse_header(r)
    verify_schema(meta)
    n = len(data)
    while r.p < n:
        count = r.long()
        size = r.long()
        end = r.p + size
        if end + 16 > n:
            raise IngestRefused(
                f"truncated block at byte {r.p}: {size} declared, {n - r.p} left. The manifest "
                "size is the byte cap, so a short object is a failed transfer, not a short file."
            )
        for _ in range(count):
            yield read_record(r)
        if r.p != end:
            raise IngestRefused(f"block cursor {r.p} != declared end {end}: schema/decoder drift")
        if data[end : end + 16] != sync:
            raise IngestRefused(f"sync marker mismatch after block ending at {end}")
        r.p = end + 16


# =============================================================================================
# The writer (ONE encoder, shared with tests/test_h01_ingest.py -- see the module docstring)
# =============================================================================================
def _e_long(v):
    z = (v << 1) ^ (v >> 63)
    out = bytearray()
    while True:
        b = z & 0x7F
        z >>= 7
        if z:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _e_bytes(b):
    return _e_long(len(b)) + b


def _e_str(s):
    return _e_bytes(s.encode("utf-8"))


def _e_null():
    return _e_long(0)


def _e_some(payload):
    return _e_long(1) + payload


def _e_xyz(t):
    if t is None:
        return _e_null()
    return _e_some(b"".join(_e_some(_e_long(int(v))) for v in t))


def _e_site(neuron_id, base_neuron_id, class_label="AXON"):
    body = b"".join(
        [
            _e_some(_e_long(int(neuron_id))),  # neuron_id
            _e_null(),  # id
            _e_null(),  # num_voxels
            _e_null(),  # centroid
            _e_null(),  # type
            _e_null(),  # subtype
            _e_null(),  # confidence
            _e_null(),  # bounding_box
            _e_null(),  # mask
            _e_some(_e_str(class_label)),  # class_label
            _e_some(_e_str(str(base_neuron_id))),  # base_neuron_id
        ]
    )
    return _e_some(body)


def encode_record(pre, pre_base, post, post_base, loc, syn_type):
    """One synapse in the release's own field order (the fields this ingest does not read are
    encoded as nulls -- a null is a legal value of every one of them)."""
    return b"".join(
        [
            _e_site(pre, pre_base),
            _e_site(post, post_base),
            _e_xyz(loc),
            _e_some(_e_long(int(syn_type))),
            _e_null(),  # contact_area
            _e_null(),  # confidence
            _e_null(),  # bounding_box
        ]
    )


def ocf_header(schema_bytes, sync=None):
    """An OCF header carrying `schema_bytes` -- how the reordered-schema red is constructed."""
    sync = bytes(range(16)) if sync is None else sync
    body = b"Obj\x01" + _e_long(2)
    body += _e_str("avro.schema") + _e_bytes(schema_bytes)
    body += _e_str("avro.codec") + _e_bytes(b"null")
    body += _e_long(0) + sync
    return body, sync


def real_header():
    """(header_bytes, sync, schema_bytes) from the tracked real-release fixture."""
    if not REAL_HEAD_FIXTURE.exists():
        raise IngestRefused(
            f"the tracked real-bytes fixture {REAL_HEAD_FIXTURE} is missing -- without it the "
            "pinned avro.schema sha256 can only ever be checked against bytes this module wrote"
        )
    raw = REAL_HEAD_FIXTURE.read_bytes()
    r = Reader(raw)
    meta, sync = parse_header(r)
    return raw[: r.p], sync, meta["avro.schema"]


def synthetic_shard(records, header=None, sync=None, per_block=4):
    """A complete null-codec OCF over already-encoded `records`."""
    if header is None:
        header, sync, _ = real_header()
    out = bytearray(header)
    for i in range(0, len(records), per_block):
        chunk = records[i : i + per_block]
        body = b"".join(chunk)
        out += _e_long(len(chunk)) + _e_long(len(body)) + body + sync
    return bytes(out)


# =============================================================================================
# The pinned inputs
# =============================================================================================
def load_manifest(path=None):
    return json.loads(Path(path or SHARD_MANIFEST).read_text(encoding="utf-8"))


def shard_url(name, manifest=None):
    m = manifest or load_manifest()
    return m["base_url"] + name


def verify_md5(name, data, entry):
    """(a) The streamed bytes against the manifest -- size first, then md5."""
    if len(data) != int(entry["size"]):
        raise IngestRefused(
            f"{name}: {len(data)} bytes, manifest says {entry['size']} -- a byte overrun or a "
            "short transfer; the manifest size IS the cap"
        )
    got = base64.b64encode(hashlib.md5(data).digest()).decode()
    if got != entry["md5_b64"]:
        raise IngestRefused(f"{name}: md5 {got} != manifest {entry['md5_b64']}")
    return got


def _verified_source(name):
    """One of the five pinned SOURCES files, refused unless its sha256 matches cortical_prior."""
    n_bytes, sha = SOURCES[name]
    path = DATA_DIR / name
    if not path.exists():
        raise IngestRefused(
            f"{path} is absent. Fetch it with --fetch-sources; it is refused unless its sha256 "
            f"equals cortical_prior.SOURCES[{name!r}]."
        )
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if len(raw) != n_bytes or got != sha:
        raise IngestRefused(f"{name}: {len(raw)} B sha256 {got} != pinned {n_bytes} B {sha}")
    return raw


def load_cells():
    """The 49,377 soma records, sha256-verified against `cortical_prior.SOURCES`."""
    return json.loads(_verified_source("agglo_20200916c3_cell_data.json"))


# distinct-from: substrate/lake_engine.py::_iter_segment and substrate/lake_bridge.py's sealed
#   segment reader -- both read the LAKE's own CRC-framed binary segments; this parses one
#   external CSV release file. distinct-from: sensorimotor/web_ingest.py::fetch_bytes -- the network
#   boundary, which this module CALLS to obtain the file and does not re-implement.
#   distinct-from: growth/cortical_prior.py::e_fraction -- a per-LAYER excitatory fraction from a
#   pinned aggregate; this is a per-SEGMENT synapse count read row by row out of the release.
#   distinct-from: growth/placement.py::layer_of -- computes a band from geometry, where
#   load_layer_labels below READS the band upstream published, and the whole point of gate (d) is
#   that the two are different instruments. All five sites read 2026-09-09.
def _ei_rows():
    """The e/i CSV's rows, sha256-verified against `cortical_prior.SOURCES`. ONE parse of one
    file feeds both the layer validator (d) and instrument 2 of (e'): they are two readings of the
    same bytes, and a second parser here could drift from the first."""
    import csv

    raw = _verified_source("goog14r0s5c3_e_i_ratios_feb_2023.csv").decode("utf-8")
    return list(csv.DictReader(raw.splitlines()))


def load_layer_labels(rows=None):
    """agglo id -> H01's OWN published `cortical layer`, from the e/i CSV's `segment id`."""
    return {r["segment id"]: r["cortical layer"] for r in (_ei_rows() if rows is None else rows)}


def load_ei_counts(rows=None):
    """(e') instrument 2: agglo id -> (excitatory synapses, inhibitory synapses), UPSTREAM's OWN
    curated per-segment counts.

    A row with 0 and 0 is DROPPED rather than read as a fraction: 10,618 of the 45,994 rows carry
    no synapses at all, and 0/0 is not "half excitatory", it is no measurement. Dropping it here
    rather than at the correlation keeps the permutation null drawing from exactly the population
    the real reading joins against."""
    out = {}
    for r in _ei_rows() if rows is None else rows:
        e = int(r["excitatory synapses"])
        i = int(r["inhibitory synapses"])
        if e + i > 0:
            out[r["segment id"]] = (e, i)
    return out


def fetch_sources(fetch=None):
    """Fetch the five pinned files into DATA_DIR, refusing any whose sha256 does not match."""
    from snn_data_lake.sensorimotor.web_ingest import fetch_bytes

    fetch = fetch or fetch_bytes
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, (n_bytes, sha) in SOURCES.items():
        path = DATA_DIR / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            out.append((name, "have"))
            continue
        data = fetch(SOURCES_URL.format(commit=H01_COMMIT, name=name), max_bytes=n_bytes)
        got = hashlib.sha256(data).hexdigest()
        if len(data) != n_bytes or got != sha:
            raise IngestRefused(
                f"{name}: fetched {len(data)} B sha256 {got}, pinned {n_bytes} {sha}"
            )
        path.write_bytes(data)
        out.append((name, "fetched"))
    return out


# =============================================================================================
# The gates
# =============================================================================================
class JoinKeys:
    """The cell file's two id spaces, as sets of strings (the export carries `neuron_id` as a
    long and `base_neuron_id` as a string; both are compared as decimal strings so the pairing
    is the only thing that differs between the four counts)."""

    __slots__ = ("agglo", "base", "neuronal")

    def __init__(self, agglo, base, neuronal=frozenset()):
        self.agglo = agglo
        self.base = base
        self.neuronal = neuronal


def join_keys(cells):
    return JoinKeys(
        frozenset(str(c["agglo_seg"]) for c in cells),
        frozenset(str(c["base_seg"]) for c in cells),
        frozenset(str(c["agglo_seg"]) for c in cells if c.get("type") in NEURONAL_TYPES),
    )


def check_join(counts, declared=DECLARED_JOIN):
    """(b) REFUSE unless `declared` is the STRICT maximum of the four measured counts."""
    lines = ", ".join(f"{a}<->{b}={counts.get((a, b), 0)}" for a, b in JOIN_PAIRINGS)
    mine = counts.get(tuple(declared), 0)
    others = [v for k, v in counts.items() if tuple(k) != tuple(declared)]
    if others and mine <= max(others):
        raise IngestRefused(
            f"join gate: the declared pairing {declared[0]}<->{declared[1]} is not the strict "
            f"maximum. Four both-ends counts: {lines}. US3 Scenario 1: the silently-empty graph "
            "is made loud here, and a refusal yields NO artifact."
        )
    return lines


# distinct-from: basal_ganglia/r_stdp.py::_da_consumer_factory -- builds a ring and a JoinConsumer
#   over a declared dopamine port; no coordinates and no extent. distinct-from:
#   growth/arealization.py::postmitotic_discretize -- an argmax over graded GRN profiles per node.
#   distinct-from: cortex/eprop_kernel.py::bptt_grads -- reverse-mode BPTT over an unrolled net.
#   distinct-from: cortex/theory_of_mind.py::TheoryOfMind -- tracks per-agent beliefs about a world
#   state. distinct-from: reasoning/reservoir_eprop.py::run -- a full two-arm SHD experiment.
#   None of the five computes a per-axis bounding box, and nothing in the tree computes the extent
#   of THIS soma cloud. All five sites read 2026-09-09 (the duplicate guard reported seven matches
#   and named these five); the guard matched on the docstring's first sentence, which is unchanged.
def soma_bounds(cells):
    """Per-axis (min, max) of the soma cloud, in NANOMETRES (`true_x/y/z` are nm)."""
    # A ROW THAT CARRIES NO COORDINATES IS NOT A POINT IN THE CLOUD, and a min() over one is not
    # a physical extent. The pinned cell file holds 12 null placeholder rows -- base_seg '0',
    # agglo_seg '0', type 'unknown cell', and x_vx/y_vx/z_vx/soma_cubic_um all 0 beside true_x/
    # true_y/true_z all 0 -- and they are the ONLY rows at x == 0 or y == 0, so counting them put
    # the ORIGIN at the cloud's corner and inflated gate (c)'s x and y denominators by 1.16x each.
    # The test is on the ROW's own coordinates and NEVER on agglo_seg == '0': that id also carries
    # 1,083 somata with real locations, so the id rule would discard measurements to drop a
    # placeholder. Not pushed up into load_cells: gate (d)'s pin counts all 49,377 rows.
    located = [c for c in cells if (c["true_x"], c["true_y"], c["true_z"]) != (0, 0, 0)]
    out = []
    for key in ("true_x", "true_y", "true_z"):
        vs = [c[key] for c in located]
        out.append((min(vs), max(vs)))
    return out


def extent_ratios(loc_bounds, bounds_nm):
    """(c) Per candidate scaling, the ratio of the synapse cloud's nm extent to the soma cloud's,
    per axis. A ratio of 1 means the two clouds span the same physical volume, which they must:
    they are the same tissue."""
    ratios = {}
    for name, scale in CANDIDATE_SCALES.items():
        per_axis = []
        for ax in range(3):
            lo, hi = loc_bounds[ax]
            soma = bounds_nm[ax][1] - bounds_nm[ax][0]
            span = (hi - lo) * scale[ax]
            per_axis.append((span / soma) if soma else float("inf"))
        ratios[name] = tuple(per_axis)
    return ratios


def check_units(ratios, declared=DECLARED_SCALE):
    """(c) REFUSE unless `declared`'s ratios are the closest to 1 on EVERY axis."""
    mine = ratios[declared]
    for name, other in ratios.items():
        if name == declared:
            continue
        for ax in range(3):
            if abs(other[ax] - 1.0) < abs(mine[ax] - 1.0):
                raise IngestRefused(
                    f"unit gate: the declared scaling {declared} is not closest to 1 on axis "
                    f"{'xyz'[ax]} -- {name} gives {other[ax]:.4f} against {mine[ax]:.4f}. "
                    f"All candidates: {unit_report(ratios)}"
                )
    return unit_report(ratios)


def unit_report(ratios):
    return "; ".join(f"{n}=({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})" for n, r in sorted(ratios.items()))


def layer_of_cell(cell, divisor=1000.0):
    """(d) COMPUTED, not read: H01's own circle rule via `placement.layer_of`, in MICRONS. The
    divisor is a parameter ONLY so the nm-instead-of-microns red can be constructed."""
    return layer_of(cell["true_x"] / divisor, cell["true_y"] / divisor)


def layer_agreement(cells, labels, divisor=1000.0):
    """(d) (agree, total) of the computed band against H01's own published `cortical layer`."""
    agree = total = 0
    for c in cells:
        published = labels.get(str(c["agglo_seg"]))
        if published is None:
            continue
        total += 1
        agree += layer_of_cell(c, divisor=divisor) == published
    return agree, total


def check_layer(agree, total, pin=LAYER_AGREEMENT_PIN):
    """(d) REFUSE unless the reading reproduces the pin EXACTLY. Not a tolerance: the pin is the
    P2 evidence that this is placement's rule on cortical_prior's bytes, and a rule that lands
    'close' is a different rule or different bytes, either of which invalidates the reuse."""
    if (agree, total) != tuple(pin):
        raise IngestRefused(
            f"layer gate: computed agreement {agree}/{total} != pinned {pin[0]}/{pin[1]} "
            "(placement.LAYER_RULE_AGREEMENT). Either the layer rule changed or the cell bytes "
            "did; both make every layer in this artifact unattributable."
        )
    return agree / total


def dale_signs(cells):
    """(e) instrument 1: agglo id -> Dale sign, from the cell's TYPE (`cortical_prior`'s rule).

    Every type outside SIGNED_TYPES is EXCLUDED, never signed by default. An agglo id is
    excluded whenever its records do not agree on a SINGLE sign -- whether two signed rows
    disagree (excitatory against inhibitory) or a signed row co-exists with an unsigned one --
    because this function cannot pick a sign for a node its own source describes two ways. The
    per-export split of those two cases is a MEASURED fact, recorded in T062 3.5(3), not a
    property of this function; nothing here counts the exclusions."""
    per_id = {}
    for c in cells:
        aid = str(c["agglo_seg"])
        t = c.get("type")
        if t in _EXCITATORY_TYPES:
            s = EXCITATORY
        elif t in _INHIBITORY_TYPES:
            s = INHIBITORY
        else:
            s = 0
        prev = per_id.get(aid)
        per_id[aid] = s if prev is None else (s if s == prev else 0)
    return {k: v for k, v in per_id.items() if v}


# distinct-from: plasticity/benna_fusi.py::MetaplasticNet.accuracy -- scores a LEARNED weight
#   vector's sign against a target pattern the caller chose, on a 0.5 chance line.
# distinct-from: plasticity/plasticity_rule_stack.py::_sign_agree -- the same learning readout in
#   a self-check. Here two EXTERNAL labels of one reconstructed synapse are compared (a cell type
#   from one release file, a synapse type from another) and NOTHING is decided by the result:
#   there is no weight, no target and no learning. Both sites read 2026-09-09.
def sign_agreement(edges, sign_of, type_sign=None):
    """DISCLOSURE (not a gate since 2.9): (agree, total) over SYNAPSES (not edges) whose PRE
    carries a Dale sign, comparing that sign with the export's own per-synapse type.

    This is the quantity the retired gate (e) refused on. It is still MEASURED and still printed,
    because 2.9's disclosure clause makes the prior's quality a measured fact carried into T063 --
    what it no longer does is decide anything. Nothing filters on it."""
    type_sign = SYN_TYPE_SIGN if type_sign is None else type_sign
    agree = total = 0
    for (pre, _post, syn_type), n in edges.items():
        dale = sign_of.get(pre)
        export = type_sign.get(syn_type)
        if dale is None or export is None:
            continue
        total += n
        if dale == export:
            agree += n
    return agree, total


def sign_report(agree, total):
    rate = (agree / total) if total else float("nan")
    return f"sign: {agree}/{total} agree (rate {rate:.4f}), {total - agree} disagree"


# distinct-from: gates/amdahl_gate.py::red_repeatability -- judges two neutered readings of a
#   TIMING rig by overlapping bootstrap bands over its own resamples; the band below is a
#   PERMUTATION null over an external file's segment labels, and the two rigs share no input.
# distinct-from: brainstem/autonomous_loop.py::coverage_scale_cells -- a lattice-point count that
#   derives an arm's workspace scale; no correlation, no null, no external file.
# distinct-from: basal_ganglia/r_stdp.py::pre_spike -- pairs a PRE spike against the most recent
#   POST spike inside a live plasticity rule; "post" there is a spike time in this brain, here it
#   is the postsynaptic SOMA of a reconstructed human synapse.
# distinct-from: sensorimotor/signal_loop.py::stdin_stream and cortex/brain.py::_flush_coverage --
#   a wire-format reader and a lake write; neither computes a fraction or a null.
#   All five sites read 2026-09-09; the guard matched on "POST"/"fraction"/"same".
def post_excitatory_fractions(post_type):
    """(e') instrument 1: agglo id -> that POST cell's excitatory INPUT fraction, from the
    export's OWN per-synapse `type` over every synapse whose POST end resolves to a soma.

    POST-ANCHORED ON PURPOSE. The join is 19x asymmetric by end (174,473 post-end hits per shard
    against 9,170 pre-end): the pre end of a synapse is an axon, and axons are the fragmented,
    mis-merged side of this reconstruction. Instrument 2 counts synapses RECEIVED by a segment, so
    instrument 1 must count the same way or the two are not readings of one quantity -- which is
    exactly the defect 2.9 records."""
    exc, tot = {}, {}
    for (post, syn_type), n in post_type.items():
        if syn_type == EXPORT_TYPE_E:
            exc[post] = exc.get(post, 0) + n
        elif syn_type != EXPORT_TYPE_I:
            continue
        tot[post] = tot.get(post, 0) + n
    return {k: exc.get(k, 0) / v for k, v in tot.items() if v > 0}


def csv_fractions(ei_counts):
    """(e') instrument 2 as the same fraction: excitatory / (excitatory + inhibitory)."""
    return {k: e / (e + i) for k, (e, i) in ei_counts.items() if (e + i) > 0}


def _pearson(x, y):
    """Pearson r over two aligned float64 arrays, written out rather than delegated: it is called
    1,001 times per gate reading and `np.corrcoef` builds a 2x2 matrix and warns on a constant
    vector, where a constant vector here is a legitimate reading that must come back NaN."""
    xc = x - x.mean()
    yc = y - y.mean()
    den = float(((xc * xc).sum() * (yc * yc).sum()) ** 0.5)
    return float((xc * yc).sum() / den) if den > 0 else float("nan")


# distinct-from: gates/gradient_claim_gate.py -- a permutation null over a LEARNED gradient's
#   alignment; that one shuffles a signal this brain produced, this one shuffles the segment
#   labels of an external release file and decides nothing about learning. Read 2026-09-09.
def ei_correlation(post_frac, csv_frac, trials=EI_PERMUTATIONS, seed=EI_SEED, pct=EI_PERCENTILE):
    """(e') The two instruments' Pearson r over the joined post cells, WITH the permutation band
    the gate compares it against. Both readings are returned and both are printed (2.9).

    THE NULL IS 2.9's, COMPUTED THE CHEAP WAY, AND IT IS THE SAME NULL. Permuting the CSV's
    segment labels across the whole file and re-joining leaves the joined ID SET untouched (the
    labels are the same labels) and hands each joined cell the value of a uniformly random other
    row -- i.e. a uniform draw WITHOUT replacement of the same size from the CSV's own values.
    That is what is drawn here, so the arithmetic is identical to a rebuilt join and costs one
    draw instead of a 35,376-element permutation per trial.
    """
    import numpy as np

    ids = sorted(set(post_frac) & set(csv_frac))
    csv_ids = sorted(csv_frac)
    csv_vals = np.array([csv_frac[k] for k in csv_ids], dtype=np.float64)
    x = np.array([post_frac[k] for k in ids], dtype=np.float64)
    y = np.array([csv_frac[k] for k in ids], dtype=np.float64)
    r = _pearson(x, y) if len(ids) >= 2 else float("nan")

    rng = np.random.default_rng(seed)
    n_csv = int(csv_vals.size)
    null = np.empty(int(trials), dtype=np.float64)
    for t in range(int(trials)):
        drawn = csv_vals[rng.choice(n_csv, size=max(len(ids), 1), replace=False)]
        null[t] = _pearson(x, drawn) if len(ids) >= 2 else float("nan")
    finite = null[np.isfinite(null)]
    return {
        "n_cells": len(ids),
        "n_csv_segments": n_csv,
        "r": r,
        "trials": int(trials),
        "percentile": float(pct),
        "seed": int(seed),
        "bar": float(np.percentile(finite, pct)) if finite.size else float("nan"),
        "null_min": float(finite.min()) if finite.size else float("nan"),
        "null_max": float(finite.max()) if finite.size else float("nan"),
        "null_mean": float(finite.mean()) if finite.size else float("nan"),
    }


def ei_report(band):
    return (
        f"e/i: r={band['r']:.4f} over {band['n_cells']} joined post cells against a "
        f"{band['percentile']:.0f}th-percentile bar of {band['bar']:.4f} from {band['trials']} "
        f"label permutations (null {band['null_min']:.4f}..{band['null_max']:.4f}, mean "
        f"{band['null_mean']:.4f}, seed {band['seed']}, {band['n_csv_segments']} CSV segments)"
    )


def check_ei(band):
    """(e') REFUSE unless the two instruments' correlation EXCEEDS the permutation band's 95th
    percentile.

    There is no threshold to tune: the bar is re-measured from the same data on every run under
    the null that a segment's label carries no information about its E/I counts, and the
    constructed red -- the shuffled labels themselves -- lands inside the band by construction.
    A reading that cannot beat its own shuffled labels is not evidence that these two files are
    describing the same tissue."""
    if band["n_cells"] < 2 or band["r"] != band["r"] or band["bar"] != band["bar"]:
        raise IngestRefused(
            f"e/i gate: vacuous -- {band['n_cells']} joined post cells is not a correlation. "
            f"{ei_report(band)}"
        )
    if band["r"] <= band["bar"]:
        raise IngestRefused(
            f"e/i gate: the export's per-POST excitatory input fraction does NOT beat its own "
            f"permutation band. {ei_report(band)}"
        )
    return band["r"]


def _disclose(edges, sign_of, soma_to_soma, pre_hits, pre_non_neuronal):
    """The 2.9 disclosures: five numbers MEASURED and PRINTED, and NEVER filtered on.

    2.9 is explicit that no edge may be dropped for disagreeing with the prior -- "filtering edges
    to match the prior would be selecting the data to fit the seed" -- so these are reported and
    carried into T063, where whether this wiring beats its marginal-matched null is the question.
    The pre end is the honest weak point: it is an axon, it is the fragmented side of the
    reconstruction, and the majority of pre-end soma hits land on cells that cannot be
    presynaptic at all. Saying so is the disclosure; acting on it would be the defect."""
    agree, total = sign_agreement(edges, sign_of)
    both = sum(n for (pre, post, _t), n in edges.items() if pre in sign_of and post in sign_of)
    return {
        "pre_end_soma_hits": pre_hits,
        "pre_end_soma_hits_non_neuronal": pre_non_neuronal,
        "pre_end_non_neuronal_fraction": (
            (pre_non_neuronal / pre_hits) if pre_hits else float("nan")
        ),
        "soma_to_soma_synapses": soma_to_soma,
        "synapses_with_dale_signed_pre": total,
        "synapses_with_both_ends_signed": both,
        "signed_pre_export_type_agreement_counts": [agree, total],
        "signed_pre_export_type_disagreement": (
            ((total - agree) / total) if total else float("nan")
        ),
        "signed_pre_export_type_report": sign_report(agree, total),
        "never_filtered_on": (
            "MEASURED AND PRINTED ONLY. No edge is dropped, reweighted or re-signed by any of "
            "these; the population's sign is the CELL's (Dale, immutable at birth) and the "
            "export's type decides nothing outside gate (e')."
        ),
    }


# =============================================================================================
# The ingest
# =============================================================================================
# distinct-from: instruments/neuro_chip_stream.py::NeuroChipStream.add_core -- its "shard" is a
#   neuromorphic CORE's LIF state written to disk by this brain; this one is a read-only tally of
#   one Avro object from an external release, and nothing here is written or persisted. Site read
#   2026-09-09; the guard matched on the word "shard".
class ShardTally:
    __slots__ = (
        "name",
        "n_records",
        "join_counts",
        "edges",
        "loc_bounds",
        "md5",
        "post_type",
        "pre_hits",
        "pre_hits_non_neuronal",
    )

    def __init__(
        self,
        name,
        n_records,
        join_counts,
        edges,
        loc_bounds,
        md5,
        post_type=None,
        pre_hits=0,
        pre_hits_non_neuronal=0,
    ):
        self.name = name
        self.n_records = n_records
        self.join_counts = join_counts
        self.edges = edges
        self.loc_bounds = loc_bounds
        self.md5 = md5
        #: (e') instrument 1's raw material: (post agglo id, export type) -> count, over every
        #: synapse whose POST end resolves, whether or not its PRE does.
        self.post_type = Counter() if post_type is None else post_type
        #: The 2.9 disclosure: how many synapses resolve at the PRE end, and how many of those
        #: land on a cell that cannot be presynaptic. Counted, never filtered on.
        self.pre_hits = pre_hits
        self.pre_hits_non_neuronal = pre_hits_non_neuronal


def decode_shard(data, keys, name="<memory>", entry=None, verify_bytes=True):
    """ONE pass: the four join counts, the declared pairing's aggregated edges, and the joined
    synapses' location bounds. The four counts are measured in the SAME pass as the graph, so the
    number that justifies the join cannot come from a different read of different bytes."""
    md5 = verify_md5(name, data, entry) if (entry is not None and verify_bytes) else None
    agglo, base, neuronal = keys.agglo, keys.base, keys.neuronal
    counts = dict.fromkeys(JOIN_PAIRINGS, 0)
    edges = Counter()
    post_type = Counter()
    pre_hits = 0
    pre_non_neuronal = 0
    lo = [None, None, None]
    hi = [None, None, None]
    n = 0
    dj = DECLARED_JOIN
    for rec in iter_records(data):
        n += 1
        pre_n = None if rec.pre_id is None else str(rec.pre_id)
        post_n = None if rec.post_id is None else str(rec.post_id)
        pre_b, post_b = rec.pre_base, rec.post_base
        # (e') and the 2.9 disclosures, measured in the SAME pass over the same bytes as the
        # join. Both are anchored at ONE end on purpose: instrument 1 counts what a post cell
        # RECEIVES whether or not the axon that delivered it resolved, and the pre-end tally
        # counts what claims to be presynaptic whether or not it can be.
        if post_n in agglo and rec.syn_type in SYN_TYPE_SIGN:
            post_type[(post_n, rec.syn_type)] += 1
        if pre_n in agglo:
            pre_hits += 1
            if pre_n not in neuronal:
                pre_non_neuronal += 1
        for pair in JOIN_PAIRINGS:
            pre_k, post_k = (pre_n, post_n) if pair[0] == "neuron_id" else (pre_b, post_b)
            target = agglo if pair[1] == "agglo_seg" else base
            if pre_k in target and post_k in target:
                counts[pair] += 1
                if pair == dj:
                    edges[(pre_k, post_k, rec.syn_type)] += 1
                    loc = rec.location
                    if loc and None not in loc:
                        for ax in range(3):
                            v = loc[ax]
                            lo[ax] = v if lo[ax] is None or v < lo[ax] else lo[ax]
                            hi[ax] = v if hi[ax] is None or v > hi[ax] else hi[ax]
    bounds = [(lo[ax], hi[ax]) for ax in range(3)] if lo[0] is not None else None
    return ShardTally(name, n, counts, edges, bounds, md5, post_type, pre_hits, pre_non_neuronal)


class Graph:
    """The summed tallies plus every gate's reading. Held in memory; the artifact is written from
    it, and a refusal means it is never written at all (2.7)."""

    __slots__ = (
        "shards",
        "n_records",
        "join_counts",
        "edges",
        "loc_bounds",
        "gates",
        "cells",
        "md5s",
    )

    def __init__(self, shards, n_records, join_counts, edges, loc_bounds, gates, cells, md5s):
        self.shards = shards
        self.n_records = n_records
        self.join_counts = join_counts
        self.edges = edges
        self.loc_bounds = loc_bounds
        self.gates = gates
        self.cells = cells
        self.md5s = md5s


def _net_fetch(name, entry, manifest):
    from snn_data_lake.sensorimotor.web_ingest import fetch_bytes

    return fetch_bytes(shard_url(name, manifest), max_bytes=int(entry["size"]), timeout=600)


def ingest(
    names,
    cells=None,
    ei_counts=None,
    fetch=None,
    threads=1,
    verify_bytes=True,
    manifest=None,
    progress=None,
    declared_join=DECLARED_JOIN,
):
    """Decode `names` (shard names, or any key `fetch` understands) and run gates (b), (c), (e').

    THREADS, NOT PROCESSES: the interpreter is free-threaded, the fetch is I/O and the decode is
    pure Python, so N workers is real parallelism here and a process pool would pay to ship a
    197 MB shard across a pipe. Determinism survives because each worker owns its own Counter and
    the sum of integers does not care about order -- which is exactly what gate (f) measures.
    """
    cells = load_cells() if cells is None else cells
    ei_counts = load_ei_counts() if ei_counts is None else ei_counts
    manifest = manifest if manifest is not None else (load_manifest() if fetch is None else None)
    entries = {s["name"]: s for s in manifest["shards"]} if manifest else {}
    if fetch is None:

        def fetch(n):
            return _net_fetch(n, entries[n], manifest)

    keys = join_keys(cells)

    def one(name):
        t0 = time.perf_counter()
        data = fetch(name)
        t1 = time.perf_counter()
        tally = decode_shard(
            data, keys, name=name, entry=entries.get(name), verify_bytes=verify_bytes
        )
        if progress is not None:
            progress(name, len(data), tally, t1 - t0, time.perf_counter() - t1)
        return tally

    # FOLDED AS THEY ARRIVE, not collected. Holding all 166 tallies was affordable while a tally
    # carried only the edge Counter (516,214 soma-to-soma synapses across the whole export); the
    # per-POST type tally is ~19,000 entries PER SHARD, and keeping 166 of those alive at once is
    # gigabytes of dict for a sum of integers. Order-independence is untouched: integer addition
    # is what gate (f) asserts about, and arrival order changes no sum.
    counts = dict.fromkeys(JOIN_PAIRINGS, 0)
    edges = Counter()
    post_type = Counter()
    md5s = {}
    lo = [None, None, None]
    hi = [None, None, None]
    n_records = 0
    pre_hits = 0
    pre_non_neuronal = 0

    def fold(tallies):
        nonlocal n_records, pre_hits, pre_non_neuronal, bounds_seen
        for t in tallies:
            n_records += t.n_records
            pre_hits += t.pre_hits
            pre_non_neuronal += t.pre_hits_non_neuronal
            md5s[t.name] = t.md5
            for k, v in t.join_counts.items():
                counts[k] += v
            edges.update(t.edges)
            post_type.update(t.post_type)
            if t.loc_bounds:
                bounds_seen = True
                for ax in range(3):
                    a, b = t.loc_bounds[ax]
                    lo[ax] = a if lo[ax] is None or a < lo[ax] else lo[ax]
                    hi[ax] = b if hi[ax] is None or b > hi[ax] else hi[ax]

    bounds_seen = False
    if threads and threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(threads)) as ex:
            fold(ex.map(one, names))
    else:
        fold(one(n) for n in names)
    bounds = [(lo[ax], hi[ax]) for ax in range(3)] if bounds_seen else None

    # EVERY GATE IS MEASURED BEFORE ANY REFUSAL IS RAISED. 2.7 requires BOTH halves -- "no
    # artifact is written" AND "the readings are recorded" -- and a raise on the first refusing
    # gate honours only the second. It would also throw away readings taken in the same pass over
    # bytes nobody wants to stream twice. Refusals accumulate and are raised together, each named.
    gates = {}
    refusals = []
    gates["join_counts"] = {f"{a}<->{b}": counts[(a, b)] for a, b in JOIN_PAIRINGS}
    try:
        gates["join"] = check_join(counts, declared=declared_join)
    except IngestRefused as e:
        refusals.append(f"join: {e}")
    if bounds is None:
        refusals.append("units: no joined synapse carried a location -- the gate is vacuous")
    else:
        ratios = extent_ratios(bounds, soma_bounds(cells))
        gates["unit_ratios"] = {k: list(v) for k, v in ratios.items()}
        try:
            gates["units"] = check_units(ratios)
        except IngestRefused as e:
            refusals.append(f"units: {e}")
    band = ei_correlation(post_excitatory_fractions(post_type), csv_fractions(ei_counts))
    gates["ei"] = band
    gates["ei_report"] = ei_report(band)
    try:
        gates["ei_r"] = check_ei(band)
    except IngestRefused as e:
        refusals.append(f"ei: {e}")
    signs = dale_signs(cells)
    gates["disclosures"] = _disclose(
        edges, signs, counts[tuple(declared_join)], pre_hits, pre_non_neuronal
    )
    gates["sign"] = gates["disclosures"]["signed_pre_export_type_report"]
    gates["records_decoded"] = n_records
    gates["refusals"] = refusals
    if refusals:
        raise IngestRefused(
            f"{len(refusals)} gate(s) REFUSED over {len(gates['join_counts'])} pairings on "
            f"{n_records:,} records; NO artifact is written and every reading is attached: "
            + " | ".join(refusals),
            readings=gates,
        )
    return Graph(list(names), n_records, counts, edges, bounds, gates, cells, md5s)


# =============================================================================================
# The artifact
# =============================================================================================
def _savez_deterministic(path, arrays):
    """np.savez writes a zip, and a zip carries the wall clock -- which would make the artifact's
    sha256 a timestamp rather than a function of the data, and gate (f) is exactly a sha256
    comparison. Fixed member order, fixed member timestamps, fixed permissions."""
    import numpy as np

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())


def build_arrays(graph, sort_edges=True):
    """Node and edge arrays. Units are the SIGNED neurons (2.2); edges are the aggregated
    soma-to-soma synapses with BOTH ends signed."""
    import numpy as np

    signs = dale_signs(graph.cells)
    by_id = {}
    for c in graph.cells:
        by_id.setdefault(str(c["agglo_seg"]), c)
    node_ids = sorted((i for i in signs if i in by_id), key=int)
    index = {aid: i for i, aid in enumerate(node_ids)}

    agglo = np.array([int(a) for a in node_ids], dtype=np.int64)
    sign = np.array([signs[a] for a in node_ids], dtype=np.int8)
    layer = np.array([BANDS.index(layer_of_cell(by_id[a])) for a in node_ids], dtype=np.int8)
    ctype = np.array([CELL_TYPES.index(by_id[a]["type"]) for a in node_ids], dtype=np.int8)
    xyz = np.array(
        [[by_id[a]["true_x"], by_id[a]["true_y"], by_id[a]["true_z"]] for a in node_ids],
        dtype=np.int64,
    )

    agg = {}
    for (pre, post, syn_type), n in graph.edges.items():
        if pre not in index or post not in index:
            continue
        cell = agg.setdefault((index[pre], index[post]), [0, 0, 0])
        cell[0] += n
        if syn_type == 2:
            cell[1] += n
        elif syn_type == 1:
            cell[2] += n
    keys = sorted(agg) if sort_edges else list(agg)
    pre_idx = np.array([k[0] for k in keys], dtype=np.int32)
    post_idx = np.array([k[1] for k in keys], dtype=np.int32)
    n_syn = np.array([agg[k][0] for k in keys], dtype=np.int32)
    n_exc = np.array([agg[k][1] for k in keys], dtype=np.int32)
    n_inh = np.array([agg[k][2] for k in keys], dtype=np.int32)
    return {
        "agglo_id": agglo,
        "sign": sign,
        "layer": layer,
        "type": ctype,
        "xyz_nm": xyz,
        "pre_idx": pre_idx,
        "post_idx": post_idx,
        "n_syn": n_syn,
        "n_exc": n_exc,
        "n_inh": n_inh,
    }


def artifact_meta(graph, arrays):
    import numpy as np

    return {
        "task": "specs/022 T062",
        "source": "gs://h01-release/data/20210729/c3/synapses/exported/",
        "h01_commit": H01_COMMIT,
        "license": LICENSE_ID,
        "attribution": ATTRIBUTION,
        # SORTED, not as-read: the artifact is a function of the shard SET, and gate (f) is the
        # assertion that reading them in the other order gives the same bytes. Leaving the read
        # order in here would have made that gate measure its own input list -- caught by the
        # gate on its first run (two sha256s differing only in this field).
        "shards": sorted(graph.shards),
        "shard_md5_b64": graph.md5s,
        "avro_schema_sha256": AVRO_SCHEMA_SHA256,
        "records_decoded": graph.n_records,
        "declared_join": list(DECLARED_JOIN),
        "join_counts": graph.gates["join_counts"],
        "declared_scale_nm_per_voxel": list(DECLARED_VOXEL_NM),
        "unit_extent_ratios": graph.gates["unit_ratios"],
        "layer_rule": "COMPUTED via cortex/placement.layer_of (H01 fix_layer_mem circles), microns",
        "layer_agreement": list(LAYER_AGREEMENT_PIN),
        "ei_instrument_1": (
            "the export's own synapse type (2 = E, 1 = I) aggregated per POST cell into that "
            "cell's excitatory INPUT fraction"
        ),
        "ei_instrument_2": (
            "goog14r0s5c3_e_i_ratios_feb_2023.csv's own per-segment `excitatory synapses` / "
            "`inhibitory synapses` counts"
        ),
        "ei_gate": graph.gates["ei"],
        "ei_report": graph.gates["ei_report"],
        "disclosures": graph.gates["disclosures"],
        "sign_of_a_unit": (
            "the CELL's type under Dale (cortical_prior.SIGNED_TYPES), immutable at birth. The "
            "export's per-synapse type is used by gate (e') and by nothing else."
        ),
        "bands": list(BANDS),
        "cell_types": list(CELL_TYPES),
        "n_nodes": int(arrays["agglo_id"].size),
        "n_edges": int(arrays["pre_idx"].size),
        "n_synapses": int(np.asarray(arrays["n_syn"]).sum()),
        "weight": (
            "sign(pre) * n_syn(pre, post) / sum_pre' n_syn(pre', post) -- the fan-in share, "
            "dimensionless. DISCLOSED as a NORMALISATION, not a measurement: a unit's summed "
            "|input| is 1 by construction."
        ),
        "delay": (
            "1 tick for every edge. DISCLOSED: H01 carries no conduction delays and this "
            "project's connectome delay FIELD has no VALUES; never derived from a velocity "
            "constant."
        ),
        "lif": LIF_DEFAULTS,
        "not_a_measurement_of_this_brain": "FR-007",
    }


def write_artifact(graph, path=None, sort_edges=True):
    """Write the .npz (allow_pickle=False) and return its sha256. Deterministic by construction:
    sorted nodes, sorted edges, fixed zip member order and timestamps."""
    import numpy as np

    path = Path(path or ARTIFACT)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = build_arrays(graph, sort_edges=sort_edges)
    payload = dict(arrays)
    payload["meta"] = np.array(json.dumps(artifact_meta(graph, arrays), sort_keys=True))
    _savez_deterministic(path, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifact(path=None):
    """Read the artifact back. `allow_pickle=False` -- an artifact that needs pickle to load is a
    code-execution surface, not data."""
    import numpy as np

    path = Path(path or ARTIFACT)
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def fan_in_shares(art):
    """sign(pre) * n_syn / (the post's total fan-in) -- aligned with `pre_idx`/`post_idx`."""
    import numpy as np

    n_syn = art["n_syn"].astype(np.float64)
    post = art["post_idx"]
    totals = np.zeros(art["agglo_id"].size, dtype=np.float64)
    np.add.at(totals, post, n_syn)
    denom = totals[post]
    share = np.divide(n_syn, denom, out=np.zeros_like(n_syn), where=denom > 0)
    return share * art["sign"][art["pre_idx"]].astype(np.float64)


# =============================================================================================
# The population (2.2)
# =============================================================================================
def population(artifact=None, p_parts=None, frac_bits=None, zero_weights=False, threads=1):
    """(KernelState, Synapses, KernelTick) from the artifact -- read ONCE, here, never on a tick.

    Units are the SIGNED neurons; faculties are the seven laminar bands with units ordered by
    (band, agglo id) so each band is an O(1) slice. `p_parts` and `frac_bits` come from the
    lineage's pinned baselines, not from a choice made here.
    """
    import numpy as np

    from snn_data_lake.gates.determinism_gate import binding_frac_bits
    from snn_data_lake.reservoir.kernel_state import KernelState
    from snn_data_lake.reservoir.kernel_tick import KernelTick, Synapses

    art = artifact if isinstance(artifact, dict) else load_artifact(artifact)
    if p_parts is None:
        p_parts = int(
            json.loads(Path(REPO / "tools" / "kernel_partition_baseline.json").read_text())["P"]
        )
    if frac_bits is None:
        frac_bits = int(binding_frac_bits(default=16))

    layer = art["layer"]
    order = np.lexsort((art["agglo_id"], layer))  # (band, agglo id)
    faculties = []
    for code, band in enumerate(BANDS):
        n = int((layer == code).sum())
        if n:
            faculties.append((band, n))
    state = KernelState(faculties)
    n_units = state.n_units
    pos = np.empty(n_units, dtype=np.int64)
    pos[order] = np.arange(n_units, dtype=np.int64)  # artifact index -> population index

    state.arrays["thr"][:] = LIF_DEFAULTS["thr"]
    state.arrays["sign"][:] = art["sign"][order]

    weight = np.zeros(art["pre_idx"].size) if zero_weights else fan_in_shares(art)
    syn = Synapses(
        pre=pos[art["pre_idx"].astype(np.int64)],
        post=pos[art["post_idx"].astype(np.int64)],
        weight=weight,
        delay=np.ones(art["pre_idx"].size, dtype=np.int64),
        n_units=n_units,
        p_parts=p_parts,
        frac_bits=frac_bits,
        max_delay=1,
    )
    kt = KernelTick(
        state,
        syn,
        decay=LIF_DEFAULTS["decay"],
        v_reset=LIF_DEFAULTS["v_reset"],
        refrac_ticks=LIF_DEFAULTS["refrac_ticks"],
        threads=threads,
    )
    return state, syn, kt


def drive_indices(kt, frac=0.01):
    """The 1 % of units driven by the liveness check: the highest OUT-DEGREE units, ties by
    index. Deterministic and DISCLOSED, and 2.3(h) fixed NEITHER the selection nor the amplitude.

    The stated reason used to be that a random draw "would measure how many isolated somata the
    release has". That is true and it does not BITE. Measured 2026-09-09 on the pinned artifact:
    a random 1 % propagates too (five INDEPENDENT seeds, totals 153-161, 5/5) and the zero-weight
    red stays RED under a random drive as well ([142, 0, 0, 0]) -- so the selection buys the gate
    nothing. What it does buy is determinism, and a richer 220-spike train for gate (f)'s
    byte-equality comparison than a ~155-spike one. See T062 3.1 and 3.5(8)."""
    deg = {}
    for (pre, _part, _delay), (posts, _q) in kt.syn.bundles.items():
        deg[pre] = deg.get(pre, 0) + len(posts)
    k = max(1, int(kt.state.n_units * frac))
    return sorted(deg, key=lambda u: (-deg[u], u))[:k]


def drive_train(kt, ticks=30, frac=0.01):
    """The canonical byte-form of `ticks` ticks under a one-tick drive of 1 % of units."""
    import numpy as np

    ext = np.zeros(kt.state.n_units, dtype=np.int64)
    ix = drive_indices(kt, frac)
    ext[ix] = int(1.2 * LIF_DEFAULTS["thr"] * (1 << kt.syn.frac_bits))
    out = [kt.tick(ext).tobytes()]
    for _ in range(max(0, ticks - 1)):
        out.append(kt.tick().tobytes())
    return b"".join(out)


def propagates(kt, ticks=30, frac=0.01):
    """(h) True if driving 1 % of units for ONE tick produces a further spike within `ticks`."""
    import numpy as np

    ext = np.zeros(kt.state.n_units, dtype=np.int64)
    ix = drive_indices(kt, frac)
    ext[ix] = int(1.2 * LIF_DEFAULTS["thr"] * (1 << kt.syn.frac_bits))
    kt.tick(ext)
    for _ in range(max(0, ticks - 1)):
        if kt.tick().size:
            return True
    return False


# =============================================================================================
# The offline self-check (run_selfchecks CORE) -- synthetic fixtures, no network, no lake
# =============================================================================================
_SELF_CELLS_CORE = [
    ("1001", "9001", "pyramidal neuron", 462268, 2805973, 40000),
    ("1002", "9002", "interneuron", 1962268, 2805973, 41000),
    ("1003", "9003", "spiny stellate neuron", 2349931, 2458418, 42000),
    ("1004", "9004", "interneuron", 5000000, 5000000, 43000),
    ("1005", "9005", "oligodendrocyte", 1000000, 2000000, 44000),
]
#: (pre_neuron_id, pre_base_neuron_id, post_neuron_id, post_base_neuron_id, location, syn_type).
#: The base ids mostly do NOT resolve against the cell file's `base_seg`, which is the release's
#: own shape (measured on shard 0: neuron_id<->agglo_seg 75 both-ends hits against 8 for
#: base_neuron_id<->base_seg) -- a fixture where every base id resolved would make gate (b) TIE
#: and refuse, which is exactly what the first draft of this list did.
_SELF_SYNAPSES_CORE = [
    (1001, 9001, 1003, 4003, (100_000, 300_000, 1_000), 2),
    (1003, 9003, 1001, 9001, (667_216, 617_698, 1_121), 2),
    (1002, 9002, 1004, 4004, (200_000, 350_000, 1_050), 1),
    (1001, 9001, 1002, 4002, (300_000, 400_000, 1_060), 2),
    (1001, 9001, 1002, 4002, (400_000, 450_000, 1_070), 2),
    (1004, 9004, 1002, 9002, (500_000, 500_000, 1_080), 2),
    (7777, 8888, 1002, 9002, (900_000, 900_000, 9_000), 2),  # pre is not a soma
    (1001, 9001, 6666, 5555, (900_000, 900_000, 9_000), 1),  # post is not a soma
]


# distinct-from: gates/doc_budget_gate.py::_selfcheck -- builds constructed manifests to make
#   a DOC gate's statuses reachable; nothing about synapses, cells or a correlation.
# distinct-from: gates/amdahl_gate.py::red_repeatability -- constructs a red for a TIMING
#   rig's own resamples. distinct-from: cortex/theta_gamma_code.py::_decode_cycle -- decodes item
#   order from live SPIKE PHASE in this brain. distinct-from:
#   cerebellum/forward_model.py::predict -- an efference-copy sensory prediction.
# distinct-from: cortex/access_consciousness.py::_post -- the ONE workspace submit site a
#   self-check uses to stand in for other faculties on a live bus. None of the five builds an
#   EXTERNAL-release fixture, and none of them is a fixture this could import instead: the shape
#   below is (Avro record, soma row, e/i CSV row) and exists nowhere else in the tree. All five
#   sites read 2026-09-09; the guard matched on "self-check"/"red"/"POST".
def _ei_fixture(n_glia=26, n_decoys=40, seed=0):
    """POST-ONLY traffic, glial somata and an e/i CSV, so gate (e') has a sample to correlate over.

    FOUR POST CELLS CANNOT RUN A PERMUTATION TEST: the null over 4! = 24 pairings puts ~4 % of its
    mass on the identity, so a fixture that correlates perfectly sits a hair above its own 95th
    percentile and the arm would be measuring luck. This tail lifts the joined post cells to 30.

    IT IS SURGICAL BY CONSTRUCTION, and that is what makes it safe to bolt onto the world gates
    (a)-(d) and (f)-(h) already measure. Every synapse here has an UNRESOLVABLE pre id, so it adds
    no both-ends join hit under any of the four pairings, no edge, no location bound and no node;
    every cell here is GLIAL, so it carries no Dale sign and never becomes a unit; and every
    coordinate is strictly inside the core soma cloud, so the extent ratios do not move. The nine
    records at the end are the mirror image -- a resolvable NON-NEURONAL pre against an
    unresolvable post -- and they exist so the pre-end disclosure has both classes in it rather
    than a 0.0 that no red could distinguish from a broken counter.
    """
    rng = random.Random(seed)
    cells, syn, ei = [], [], {}
    for k in range(n_glia):
        aid = 2000 + k
        cells.append(
            (
                str(aid),
                str(8000 + k),
                "astrocyte" if k % 2 else "oligodendrocyte",
                1_000_000 + 70_000 * k,
                3_000_000 + 30_000 * k,
                42_000,
            )
        )
        n_e = 1 + rng.randrange(9)
        n_i = 1 + rng.randrange(9)
        for j in range(n_e + n_i):
            syn.append(
                (
                    900_000 + 1_000 * k + j,  # pre: resolves nowhere, under any pairing
                    990_000 + 1_000 * k + j,
                    aid,
                    7_000 + k,
                    None,
                    EXPORT_TYPE_E if j < n_e else EXPORT_TYPE_I,
                )
            )
        # instrument 2 agrees with instrument 1 up to a curation difference, never exactly
        ei[str(aid)] = (n_e * 10 + rng.randrange(6), n_i * 10 + rng.randrange(6))
    for k in range(9):
        syn.append((2000 + k, 8000 + k, 6666, 5555, None, EXPORT_TYPE_E))
    ei.update({"1001": (10, 0), "1002": (30, 1), "1003": (8, 0), "1004": (0, 12), "1005": (5, 5)})
    for d in range(n_decoys):
        ei[str(3000 + d)] = (rng.randrange(40), rng.randrange(40))
    return cells, syn, ei


_EI_CELLS, _EI_SYNAPSES, _SELF_EI = _ei_fixture()
_SELF_CELLS = [*_SELF_CELLS_CORE, *_EI_CELLS]
_SELF_SYNAPSES = [*_SELF_SYNAPSES_CORE, *_EI_SYNAPSES]


def self_cells():
    return [
        {"agglo_seg": a, "base_seg": b, "type": t, "true_x": x, "true_y": y, "true_z": z}
        for a, b, t, x, y, z in _SELF_CELLS
    ]


def self_ei(shuffle=False, seed=1):
    """(e') instrument 2 for the offline gate. `shuffle=True` is the CONSTRUCTED RED of 2.9 --
    the same values under permuted segment labels, which is one draw from exactly the null the
    gate's bar is measured from. The seed is fixed so the red is a reading, not a coin flip."""
    ei = dict(_SELF_EI)
    if shuffle:
        keys = sorted(ei)
        vals = [ei[k] for k in keys]
        random.Random(seed).shuffle(vals)
        ei = dict(zip(keys, vals, strict=True))
    return ei


def self_shard(syn=None):
    rows = _SELF_SYNAPSES if syn is None else syn
    return synthetic_shard([encode_record(*s) for s in rows])


def self_shard_pair():
    """THE two-shard split (P2: one splitter, used by the offline gate AND by the suite).

    The CORE synapses are cut in half so BOTH shards carry soma-to-soma edges -- gate (f)'s
    unsorted-order red needs two Counters that actually differ, and a naive halfway cut of the
    whole synapse list now puts every edge in the first shard, which would defuse that red
    silently. The post-only tail is dealt alternately."""
    a = [*_SELF_SYNAPSES_CORE[:4], *_EI_SYNAPSES[0::2]]
    b = [*_SELF_SYNAPSES_CORE[4:], *_EI_SYNAPSES[1::2]]
    return self_shard(a), self_shard(b)


def _selfcheck():
    """Every gate of 2.3 with its red and its control, on synthetic fixtures. ASCII only."""
    import tempfile

    import numpy as np

    from snn_data_lake.substrate.home import throwaway_home

    cells = self_cells()
    keys = join_keys(cells)
    data = self_shard()
    ok = []

    # ---- (a) bytes -------------------------------------------------------------------------
    entry = {
        "name": "exportSELF",
        "size": len(data),
        "md5_b64": base64.b64encode(hashlib.md5(data).digest()).decode(),
    }
    verify_md5("exportSELF", data, entry)
    flipped = bytearray(hashlib.md5(data).digest())
    flipped[0] ^= 0x10
    for bad in (
        dict(entry, md5_b64=base64.b64encode(bytes(flipped)).decode()),
        dict(entry, size=len(data) - 1),
    ):
        try:
            verify_md5("exportSELF", data, bad)
            print("h01_ingest self-check FAILED: (a) accepted a wrong manifest entry")
            return 1
        except IngestRefused:
            pass
    header, _sync, schema = real_header()
    r = Reader(header)
    meta, _ = parse_header(r)
    if verify_schema(meta) != AVRO_SCHEMA_SHA256:
        print("h01_ingest self-check FAILED: (a) the real release header missed the pin")
        return 1
    doc = json.loads(schema)
    doc["fields"][0], doc["fields"][1] = doc["fields"][1], doc["fields"][0]
    bad_header, bad_sync = ocf_header(json.dumps(doc).encode())
    bad_shard = synthetic_shard(
        [encode_record(*_SELF_SYNAPSES[0])], header=bad_header, sync=bad_sync
    )
    try:
        list(iter_records(bad_shard))
        print("h01_ingest self-check FAILED: (a) a reordered schema was accepted")
        return 1
    except IngestRefused:
        pass
    ok.append(
        "(a) bytes: md5 + size + the REAL header's schema sha256 pinned; a flipped digest, "
        "a short object and a two-field schema reorder all REFUSED"
    )

    # ---- (b) join --------------------------------------------------------------------------
    tally = decode_shard(data, keys)
    if tally.join_counts[DECLARED_JOIN] != 6:
        print(
            f"h01_ingest self-check FAILED: (b) expected 6 joined synapses, got {tally.join_counts}"
        )
        return 1
    joined = check_join(tally.join_counts)
    try:
        check_join(tally.join_counts, declared=("base_neuron_id", "agglo_seg"))
        print("h01_ingest self-check FAILED: (b) the base<->agglo pairing was accepted")
        return 1
    except IngestRefused:
        pass
    try:
        check_join(dict.fromkeys(JOIN_PAIRINGS, 7))
        print("h01_ingest self-check FAILED: (b) a TIE was accepted as a strict maximum")
        return 1
    except IngestRefused:
        pass
    ok.append(
        f"(b) join: {joined}; the declared pairing is the STRICT maximum, base<->agglo and "
        "a four-way tie both REFUSED"
    )

    # ---- (c) units -------------------------------------------------------------------------
    ratios = extent_ratios(tally.loc_bounds, soma_bounds(cells))
    report = check_units(ratios)
    try:
        check_units(ratios, declared="inversion")
        print("h01_ingest self-check FAILED: (c) the nm inversion was accepted")
        return 1
    except IngestRefused:
        pass
    ok.append(
        f"(c) units: {report}; the declared (8,8,33) is closest to 1 on every axis, the "
        "inversion REFUSED"
    )

    # ---- (d) layer -------------------------------------------------------------------------
    labels = {c["agglo_seg"]: layer_of_cell(c) for c in cells}
    good = layer_agreement(cells, labels)
    nm = layer_agreement(cells, labels, divisor=1.0)
    bands = {layer_of_cell(c) for c in cells}
    # THE RED IS THE COLLAPSE, NOT A ZERO, and the difference is load-bearing. Feeding nanometres
    # puts EVERY soma outside every circle, so every one is classified "Layer 1" -- but a soma
    # that genuinely IS in Layer 1 then agrees by accident. This fixture has one (1004), so the
    # nm agreement is 1/5 rather than 0/5. Asserting a zero here would be asserting that the
    # fixture has no Layer 1 cell, which is a fact about the fixture, not about the rule.
    if good != (len(cells), len(cells)):
        print(
            f"h01_ingest self-check FAILED: (d) the micron control did not agree with itself {good}"
        )
        return 1
    if {layer_of_cell(c, divisor=1.0) for c in cells} != {"Layer 1"}:
        print("h01_ingest self-check FAILED: (d) the nm red did not collapse into Layer 1")
        return 1
    if nm[0] >= good[0]:
        print(f"h01_ingest self-check FAILED: (d) nm agreement {nm} did not fall below {good}")
        return 1
    try:
        check_layer(LAYER_AGREEMENT_PIN[0] - 1, LAYER_AGREEMENT_PIN[1])
        print("h01_ingest self-check FAILED: (d) an off-by-one agreement was accepted")
        return 1
    except IngestRefused:
        pass
    check_layer(*LAYER_AGREEMENT_PIN)
    ok.append(
        f"(d) layer: COMPUTED via placement.layer_of over {len(bands)} bands {sorted(bands)}; "
        f"nanometres instead of microns puts all {len(cells)} somata in Layer 1 and drops "
        f"agreement {good[0]}/{good[1]} -> {nm[0]}/{nm[1]} (NOT to zero: this fixture has one "
        f"cell genuinely IN Layer 1, which the red cannot distinguish -- disclosed); the pin "
        f"{LAYER_AGREEMENT_PIN[0]}/{LAYER_AGREEMENT_PIN[1]} admits no off-by-one"
    )

    # ---- (e') e/i, POST-anchored -------------------------------------------------------------
    ei = self_ei()
    post_frac = post_excitatory_fractions(tally.post_type)
    band = ei_correlation(post_frac, csv_fractions(ei))
    r = check_ei(band)
    red = ei_correlation(post_frac, csv_fractions(self_ei(shuffle=True)))
    if red["n_cells"] != band["n_cells"]:
        print(
            f"h01_ingest self-check FAILED: (e') the shuffled red changed the joined cell set "
            f"{red['n_cells']} != {band['n_cells']}, so it measures two things at once"
        )
        return 1
    try:
        check_ei(red)
        print("h01_ingest self-check FAILED: (e') shuffled segment labels beat the band")
        return 1
    except IngestRefused:
        pass
    if not (band["null_min"] <= band["bar"] <= band["null_max"]):
        print(f"h01_ingest self-check FAILED: (e') the bar sits outside its own null {band}")
        return 1
    try:
        check_ei(ei_correlation({"1001": 1.0}, {"1001": 1.0}, trials=8))
        print("h01_ingest self-check FAILED: (e') a one-cell correlation was accepted")
        return 1
    except IngestRefused:
        pass
    # A REFUSAL MUST CARRY ITS READINGS (2.7). Run the whole ingest under the shuffled labels and
    # assert the raise brings the join counts, the extent ratios and the disclosures out with it
    # -- a refusal IS the result when it happens, and one that arrives without its evidence
    # cannot be audited, argued with or re-derived.
    carried = None
    try:
        ingest(
            ["S"],
            cells=cells,
            ei_counts=self_ei(shuffle=True),
            fetch={"S": data}.get,
            verify_bytes=False,
        )
        print("h01_ingest self-check FAILED: (e') a shuffled-label world did not refuse")
        return 1
    except IngestRefused as refusal:
        missing = [
            k
            for k in ("join_counts", "unit_ratios", "ei", "disclosures", "refusals")
            if k not in refusal.readings
        ]
        if missing:
            print(f"h01_ingest self-check FAILED: (e') the refusal dropped {missing}")
            return 1
        carried = refusal.readings
    ok.append(
        f"(e') {ei_report(band)} -- r={r:.4f} BEATS its own permutation bar; the same values "
        f"under SHUFFLED segment labels score r={red['r']:.4f} against bar {red['bar']:.4f} over "
        f"the identical {red['n_cells']} cells and are REFUSED, a one-cell reading is refused as "
        f"vacuous, and a shuffled-label ingest carries every reading out with it "
        f"({len(carried)} keys, {carried['refusals']})"
    )

    # ---- (f) determinism + (g) P4b + (h) liveness, in one throwaway home --------------------
    with throwaway_home(), tempfile.TemporaryDirectory(prefix="h01_selfcheck_") as tmp:
        from snn_data_lake.substrate.home import BinLog, knowledge_path

        def n_records():
            p = Path(knowledge_path())
            return sum(1 for _ in BinLog(str(p)).replay()) if p.exists() else 0

        before = n_records()
        shard_a, shard_b = self_shard_pair()
        blobs = {"A": shard_a, "B": shard_b}
        fwd = ingest(["A", "B"], cells=cells, ei_counts=ei, fetch=blobs.get, verify_bytes=False)
        rev = ingest(["B", "A"], cells=cells, ei_counts=ei, fetch=blobs.get, verify_bytes=False)
        tmp = Path(tmp)
        a = write_artifact(fwd, tmp / "fwd.npz")
        b = write_artifact(rev, tmp / "rev.npz")
        if a != b:
            print(f"h01_ingest self-check FAILED: (f) shard order changed the artifact {a} {b}")
            return 1
        ua = write_artifact(fwd, tmp / "ua.npz", sort_edges=False)
        ub = write_artifact(rev, tmp / "ub.npz", sort_edges=False)
        if ua == ub:
            print(
                "h01_ingest self-check FAILED: (f) the unsorted control did not diverge, so the "
                "sorted arm proves nothing"
            )
            return 1
        blob_pair, train_pair = [], []
        for _ in range(2):
            st, _sy, kt = population(tmp / "fwd.npz")
            blob_pair.append(st.to_blob())
            train_pair.append(drive_train(kt, ticks=30))
        if blob_pair[0] != blob_pair[1] or train_pair[0] != train_pair[1]:
            print("h01_ingest self-check FAILED: (f) two constructions diverged")
            return 1
        art = load_artifact(tmp / "fwd.npz")
        shares = fan_in_shares(art)
        totals = {}
        for post, w in zip(art["post_idx"].tolist(), shares.tolist(), strict=True):
            totals[post] = totals.get(post, 0.0) + abs(w)
        if any(abs(v - 1.0) > 1e-9 for v in totals.values()):
            print(f"h01_ingest self-check FAILED: fan-in shares do not sum to 1 ({totals})")
            return 1
        ok.append(
            f"(f) determinism: forward and reverse shard order give the SAME artifact "
            f"sha256 {a[:16]}..., the unsorted control DIVERGES ({ua[:8]} vs {ub[:8]}), and "
            "two populations give byte-equal KernelState and byte-equal 30-tick trains"
        )

        # The 2.9 disclosures, printed beside the gates and asserted to reach the artifact meta.
        # Nothing refuses on them; what is checked is that no edge was dropped between the join
        # and the artifact, because that -- not the numbers -- is what "never filtered on" means.
        disc = fwd.gates["disclosures"]
        if int(art["n_syn"].sum()) != disc["synapses_with_both_ends_signed"]:
            print(
                f"h01_ingest self-check FAILED: {int(art['n_syn'].sum())} synapses reached the "
                f"artifact but {disc['synapses_with_both_ends_signed']} were joined with both "
                "ends signed -- an edge was filtered between the join and the artifact"
            )
            return 1
        if art["meta"]["disclosures"] != disc:
            print("h01_ingest self-check FAILED: the artifact meta lost the disclosures")
            return 1
        ok.append(
            "disclosures (2.9, measured and printed, NEVER filtered on): "
            f"{disc['pre_end_soma_hits_non_neuronal']}/{disc['pre_end_soma_hits']} = "
            f"{disc['pre_end_non_neuronal_fraction']:.4f} of pre-end soma hits are non-neuronal; "
            f"{disc['soma_to_soma_synapses']} soma-to-soma synapses, "
            f"{disc['synapses_with_dale_signed_pre']} with a Dale-signed pre, "
            f"{disc['synapses_with_both_ends_signed']} with BOTH ends signed; the export type "
            f"disagrees with the pre cell's class on "
            f"{disc['signed_pre_export_type_disagreement']:.4f} of the signed-pre synapses, and "
            "every one of them is in the artifact"
        )

        after = n_records()
        if after != before:
            print(f"h01_ingest self-check FAILED: (g) the prior entered the lake {before}->{after}")
            return 1
        from snn_data_lake.substrate.home import save_named_state

        if not save_named_state("t062_p4b_red", (tmp / "fwd.npz").read_bytes()):
            print("h01_ingest self-check FAILED: (g) the deliberate write was refused")
            return 1
        if n_records() <= before:
            print(
                "h01_ingest self-check FAILED: (g) a deliberate save did not move the counter, "
                "so the P4b arm is blind, not green"
            )
            return 1
        ok.append(
            f"(g) P4b: a full ingest + population added 0 lake records ({before} before and "
            f"after), and a DELIBERATE save_named_state moved the same counter to "
            f"{n_records()} -- the arm is sensitive, not blind"
        )

        _s, _y, kt = population(tmp / "fwd.npz")
        if not propagates(kt, ticks=30):
            print("h01_ingest self-check FAILED: (h) the seeded population is inert")
            return 1
        _s0, _y0, kt0 = population(tmp / "fwd.npz", zero_weights=True)
        if propagates(kt0, ticks=30):
            print("h01_ingest self-check FAILED: (h) the zeroed-weight control still propagated")
            return 1
        st, sy, _kt = population(tmp / "fwd.npz")
        ok.append(
            f"(h) liveness: driving {len(drive_indices(kt))} of {st.n_units} units for ONE "
            f"tick propagates within 30; the zeroed-weight control does NOT. Population: "
            f"{st.n_units} signed units in {len(st.ranges)} bands, {sy.bundles and len(art['pre_idx'])} "
            f"edges, KernelState {len(st.to_blob())} bytes"
        )

    if not np.isfinite(LIF_DEFAULTS["decay"]):
        print("h01_ingest self-check FAILED: the imported LIF decay is not finite")
        return 1

    print("=" * 96)
    print("H01 c3 SYNAPSE GRAPH INGEST -- offline gate (synthetic fixtures + the real header)")
    print("=" * 96)
    for line in ok:
        print("  " + line)
    print(f"  licence {LICENSE_ID}; thr/decay imported from {LIF_DEFAULTS['owner']}")
    print(
        "self-check OK: 8 gates (2.3 (a)-(d), (f)-(h) plus 2.9's replacement (e')), every one "
        "with its constructed RED in this same command; no network, no lake write, the artifact "
        "never leaves a temp dir"
    )
    return 0


# =============================================================================================
# CLI
# =============================================================================================
def _parse_slice(spec, n):
    if ":" in spec:
        a, _, b = spec.partition(":")
        return int(a or 0), int(b or n)
    return 0, int(spec)


def _run_ingest(argv):
    import os

    manifest = load_manifest()
    names = [s["name"] for s in manifest["shards"]]
    spec = "0:35"
    threads = 8
    out = ARTIFACT
    for i, a in enumerate(argv):
        if a == "--shards":
            spec = argv[i + 1]
        elif a == "--threads":
            threads = int(argv[i + 1])
        elif a == "--out":
            out = Path(argv[i + 1])
    lo, hi = _parse_slice(spec, len(names))
    chosen = names[lo:hi]
    print(f"H01 c3 ingest: shards [{lo}:{hi}) = {len(chosen)}, threads {threads}")
    print(f"NORD_HOME={os.environ.get('NORD_HOME')}  out={out}")
    cells = load_cells()
    rows = _ei_rows()  # ONE parse; (d)'s labels and (e')'s counts are two readings of it
    labels = load_layer_labels(rows)
    ei_counts = load_ei_counts(rows)
    agree, total = layer_agreement(cells, labels)
    print(f"(d) layer agreement {agree}/{total} = {agree / total:.6f}")
    check_layer(agree, total)
    print(f"(e') instrument 2: {len(ei_counts)} of {len(rows)} CSV segments carry a synapse count")
    t0 = time.perf_counter()
    seen = {"bytes": 0, "records": 0, "fetch_s": 0.0, "decode_s": 0.0}

    def progress(name, nbytes, tally, fetch_s, decode_s):
        seen["bytes"] += nbytes
        seen["records"] += tally.n_records
        seen["fetch_s"] += fetch_s
        seen["decode_s"] += decode_s
        print(
            f"  {name} {nbytes / 1e6:.1f} MB {tally.n_records} rec fetch {fetch_s:.1f}s "
            f"decode {decode_s:.1f}s ({tally.n_records / decode_s:,.0f} rec/s) "
            f"joined {tally.join_counts[DECLARED_JOIN]}",
            flush=True,
        )

    try:
        graph = ingest(
            chosen,
            cells=cells,
            ei_counts=ei_counts,
            threads=threads,
            manifest=manifest,
            progress=progress,
        )
    except IngestRefused as refusal:
        # 2.7's null clause, executed: NO artifact, the readings recorded, the row left OPEN with
        # the refusal named. The pin is written as a REFUSAL record rather than not written at
        # all -- an unrecorded refusal is indistinguishable from a run that never happened.
        wall = time.perf_counter() - t0
        rec = {
            "verdict": "REFUSED",
            "refusals": refusal.readings.get("refusals", []),
            "message": str(refusal),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.environ.get("COMPUTERNAME", "?"),
            "git_head": _git_head(),
            "shards": chosen,
            "threads": threads,
            "wall_s": round(wall, 2),
            "bytes_streamed": seen["bytes"],
            "fetch_s_total": round(seen["fetch_s"], 2),
            "decode_s_total": round(seen["decode_s"], 2),
            "harness_sha256": {
                "growth/h01_ingest.py": _sha256_of(Path(__file__)),
                "tools/h01_c3_shards.json": _sha256_of(SHARD_MANIFEST),
                "tests/fixtures/h01_c3_shard0_head.avro": _sha256_of(REAL_HEAD_FIXTURE),
            },
            "gates": {
                "a_shard_md5": "all verified against tools/h01_c3_shards.json",
                "a_schema_sha256": AVRO_SCHEMA_SHA256,
                "d_layer_agreement": [agree, total],
                **refusal.readings,
            },
            "artifact": None,
            "artifact_sha256": None,
        }
        write_pin(rec)
        print(json.dumps(rec, indent=2, sort_keys=True))
        print(f"\nREFUSED: {refusal}")
        return 2
    wall = time.perf_counter() - t0
    sha = write_artifact(graph, out)
    art = load_artifact(out)
    state, syn, kt = population(out)
    pin = {
        # SYMMETRY, not a WIN. 2.7 gives this row refusals and no WIN/LOSS, and 2.1 never named a
        # verdict field -- but the refusal path above writes one and this path wrote nothing, so
        # the machine-readable record carried a refusal marker and no counterpart. The token says
        # only that no gate raised; the readings that justify it are in `gates` below, and a
        # reader who trusts this string instead of those numbers is reading it wrong. Written by
        # the RUN and never back-filled: the "...#2" entry already in the pin has no verdict, is
        # not hand-edited, and gains one only when a later legitimate ingest writes its own.
        "verdict": "PASSED",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.environ.get("COMPUTERNAME", "?"),
        "git_head": _git_head(),
        "artifact": str(out),
        "artifact_sha256": sha,
        "artifact_bytes": Path(out).stat().st_size,
        "shards": graph.shards,
        "shard_md5_b64": graph.md5s,
        "harness_sha256": {
            "growth/h01_ingest.py": _sha256_of(Path(__file__)),
            "tools/h01_c3_shards.json": _sha256_of(SHARD_MANIFEST),
            "tests/fixtures/h01_c3_shard0_head.avro": _sha256_of(REAL_HEAD_FIXTURE),
        },
        "gates": {
            "a_shard_md5": "all verified against tools/h01_c3_shards.json",
            "a_schema_sha256": AVRO_SCHEMA_SHA256,
            "b_join_counts": graph.gates["join_counts"],
            "b_declared": list(DECLARED_JOIN),
            "c_unit_ratios": graph.gates["unit_ratios"],
            "c_declared": DECLARED_SCALE,
            "d_layer_agreement": [agree, total],
            "e_prime_ei": graph.gates["ei"],
            "e_prime_report": graph.gates["ei_report"],
            "e_prime_r": graph.gates["ei_r"],
        },
        "disclosures": graph.gates["disclosures"],
        "records_decoded": graph.n_records,
        "bytes_streamed": seen["bytes"],
        "wall_s": round(wall, 2),
        "fetch_s_total": round(seen["fetch_s"], 2),
        "decode_s_total": round(seen["decode_s"], 2),
        "threads": threads,
        "n_nodes": int(art["agglo_id"].size),
        "n_edges": int(art["pre_idx"].size),
        "n_synapses": int(art["n_syn"].sum()),
        "kernel_state_bytes": len(state.to_blob()),
        "bundle_count": len(syn.bundles),
        "units_per_band": {b: (r[1] - r[0]) for b, r in state.ranges.items()},
        "propagates_within_30": bool(propagates(kt, ticks=30)),
    }
    write_pin(pin)
    print(json.dumps(pin, indent=2, sort_keys=True))
    return 0


# distinct-from: instruments/port_charge_probe.py::record_attribution -- THE one writer for
#   tools/region_charge_evidence.json, and the CLOSEST relative here: same read-modify-write
#   discipline, for the same reason ("one registrant's regeneration can never delete another's").
#   Not reusable: it is hard-wired to that path and to an (edge, diverged, measured, provenance)
#   row, while a stage pin is keyed by shard range and carries gate readings. What is reused is
#   the LESSON, cited so the next reader sees it was not reinvented.
# distinct-from: cortex/brain.py::_flush_growth -- writes GROWN SUBSTRATE into the durable lake
#   (P4b state, the thing this module must never write). distinct-from:
#   neuromodulation/intrinsic_motivation.py::observe and neuromodulation/spiking_curiosity.py::
#   observe -- fold a prediction error into running EMAs. distinct-from:
#   reasoning/developmental_loop_wm.py::_wm_persistence_probe -- a working-memory hold probe.
#   distinct-from: substrate/region_coherence.py::record_charge -- measures an A/B and delegates
#   to record_attribution above. All sites read 2026-09-09; the guard matched on "Fold"/"record".
def stage_key(shards):
    """The pin's key for one stage: how many shards, and which range."""
    s = list(shards)
    return f"{len(s)}-shards:{s[0]}..{s[-1]}" if s else "0-shards"


def merge_pin(existing, record):
    """Fold `record` into the pin under its own stage key, NEVER erasing an earlier stage.

    2.4 says the stage-B artifact replaces stage A's and "the pin records BOTH". With a refusal
    there is no artifact at all, so the readings ARE the record and overwriting them would throw
    away the row's only result. A LEGACY flat record (the shape the first stage-A run wrote) is
    MIGRATED into the stage map rather than dropped -- a fix that erased the readings it exists to
    protect would be worse than the defect it fixes."""
    out = {"task": "specs/022 T062", "stages": {}}
    if existing:
        if isinstance(existing.get("stages"), dict):
            out["stages"].update(existing["stages"])
        elif existing.get("shards"):
            out["stages"][stage_key(existing["shards"])] = existing
    key = stage_key(record.get("shards", []))
    # A RE-RUN OF THE SAME SHARDS IS A NEW READING, NOT A CORRECTION OF THE OLD ONE. 2.9
    # authorises a second run over the same 166 shards under a different gate set, and its stage
    # key is identical to the run that REFUSED -- so keying by stage alone would overwrite the
    # refusal it is meant to stand beside, and with no artifact written a refusal's readings are
    # the row's only result. The FIRST reading keeps the plain key; a re-run is suffixed, so the
    # unqualified key never becomes ambiguous to whoever reads the pin later.
    if key in out["stages"]:
        n = 2
        while f"{key}#{n}" in out["stages"]:
            n += 1
        key = f"{key}#{n}"
    out["stages"][key] = record
    out["latest"] = key
    return out


def write_pin(record, path=None):
    """Read-modify-write the pin so a later stage cannot erase an earlier one."""
    path = Path(path or GRAPH_PIN)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    merged = merge_pin(existing, record)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged


def _sha256_of(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _git_head():
    head = REPO / ".git" / "HEAD"
    if not head.exists():
        return None
    line = head.read_text().strip()
    if line.startswith("ref: "):
        ref = REPO / ".git" / line[5:]
        return ref.read_text().strip() if ref.exists() else None
    return line


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--fetch-sources" in argv:
        for name, how in fetch_sources():
            print(f"  {how} {name}")
        return 0
    if "--ingest" in argv:
        return _run_ingest(argv)
    return _selfcheck()


if __name__ == "__main__":
    sys.exit(main())
