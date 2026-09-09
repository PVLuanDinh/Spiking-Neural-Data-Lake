"""specs/022 T062 -- every gate of the H01 c3 synapse-graph ingest, each with its RED.

Pre-registration: `specs/022-unified-brain-arc/evidence/t062-h01-ingest.md` 2.3 (a)..(h). Each
gate below is a REFUSAL with a constructed red run in the same command as its control, and no
gate carries an invented threshold: (b) requires the declared join to be the strict maximum of
four measured counts, (c) requires the declared scaling's extent ratios to be the closest to 1
on every axis of four candidates, (d) requires the layer rule to REPRODUCE
`placement.LAYER_RULE_AGREEMENT`'s pinned 45,404 / 49,377 exactly, (e) requires sign agreement
> 0.5 because the swapped map scores exactly 1 - r.

NO NETWORK. The shard fixtures are written by the in-test Avro writer below, which reuses the
REAL release header from `tests/fixtures/h01_c3_shard0_head.avro` -- so the pinned `avro.schema`
sha256 is checked against bytes Google published, not against bytes this file produced. The
writer's own round trip through the module's decoder is asserted first: a fixture nobody has
proven decodes is not a fixture, it is a second bug waiting to be blamed on the mechanism.

The two data-backed arms (the real 45,404 / 49,377 reading and the real-header schema pin) skip
when `data/h01/` is absent -- `data/` is gitignored, so CI has neither. They are NOT the only
cover for gate (d): the nm-instead-of-microns red and the synthetic-agreement arms below run
unconditionally, and the pinned pair is asserted against `placement.LAYER_RULE_AGREEMENT` in
every environment.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import struct
from pathlib import Path

import pytest

from snn_data_lake.cortex import h01_ingest as H
from snn_data_lake.cortex.placement import LAYER_NAMES, LAYER_RULE_AGREEMENT, WHITE_MATTER
from snn_data_lake.substrate.home import throwaway_home

FIXTURE = Path(__file__).parent / "fixtures" / "h01_c3_shard0_head.avro"


# ---------------------------------------------------------------------------------------------
# The in-test Avro writer (null codec, the release's own field order). Mirrors the decoder the
# module ports; every optional field is a ["null", X] union, index 0 = null, 1 = value.
# ---------------------------------------------------------------------------------------------
def _long(v):
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


def _bytes(b):
    return _long(len(b)) + b


def _str(s):
    return _bytes(s.encode("utf-8"))


def _null():
    return _long(0)


def _some(payload):
    return _long(1) + payload


def _xyz(t):
    if t is None:
        return _null()
    return _some(b"".join(_some(_long(int(v))) for v in t))


def _site(neuron_id, base_neuron_id, class_label="AXON"):
    """The 11 site fields in the pinned order; only the three the join needs are non-null."""
    body = b"".join(
        [
            _some(_long(int(neuron_id))),  # neuron_id
            _null(),  # id
            _null(),  # num_voxels
            _null(),  # centroid
            _null(),  # type
            _null(),  # subtype
            _null(),  # confidence
            _null(),  # bounding_box
            _null(),  # mask
            _some(_str(class_label)),  # class_label
            _some(_str(str(base_neuron_id))),  # base_neuron_id
        ]
    )
    return _some(body)


def _record(pre, pre_base, post, post_base, loc, syn_type):
    return b"".join(
        [
            _site(pre, pre_base),
            _site(post, post_base),
            _xyz(loc),
            _some(_long(int(syn_type))),
            _null(),  # contact_area
            _null(),  # confidence
            _null(),  # bounding_box
        ]
    )


def real_header():
    """(header_bytes, sync, schema_bytes) from the REAL release fixture."""
    raw = FIXTURE.read_bytes()
    r = H.Reader(raw)
    meta, sync = H.parse_header(r)
    return raw[: r.p], sync, meta["avro.schema"]


def _header_for(schema_bytes):
    """An OCF header carrying `schema_bytes` (used to build the reordered-schema red)."""
    sync = bytes(range(16))
    body = b"Obj\x01" + _long(2)
    body += _str("avro.schema") + _bytes(schema_bytes)
    body += _str("avro.codec") + _bytes(b"null")
    body += _long(0) + sync
    return body, sync


def make_shard(records, header=None, sync=None, per_block=4):
    """A complete null-codec OCF over `records` (already-encoded record bytes)."""
    if header is None:
        header, sync, _ = real_header()
    out = bytearray(header)
    for i in range(0, len(records), per_block):
        chunk = records[i : i + per_block]
        body = b"".join(chunk)
        out += _long(len(chunk)) + _long(len(body)) + body + sync
    return bytes(out)


# ---------------------------------------------------------------------------------------------
# A tiny synthetic world: 4 signed somata + 1 unsigned, their layers, and a handful of synapses.
# ---------------------------------------------------------------------------------------------
CELLS = [
    # (agglo_seg, base_seg, type, true_x nm, true_y nm, true_z nm)
    ("1001", "9001", "pyramidal neuron", 1_200_000, 2_500_000, 40_000),
    ("1002", "9002", "interneuron", 1_150_000, 2_450_000, 41_000),
    ("1003", "9003", "spiny stellate neuron", 1_300_000, 2_600_000, 42_000),
    ("1004", "9004", "pyramidal neuron", 400_000, 2_800_000, 43_000),
    ("1005", "9005", "oligodendrocyte", 1_000_000, 2_000_000, 44_000),
]


def cells():
    return [
        {
            "agglo_seg": a,
            "base_seg": b,
            "type": t,
            "true_x": x,
            "true_y": y,
            "true_z": z,
        }
        for a, b, t, x, y, z in CELLS
    ]


def synapses():
    """(pre, pre_base, post, post_base, location_in_8nm_voxels, syn_type).

    Locations are chosen so that under the DECLARED (8, 8, 33) nm scaling the synapse cloud's
    nm extent lands nearest the soma cloud's, which is what gate (c) requires.
    """
    return [
        (1001, 9001, 1002, 9002, (150_000, 312_500, 1_212), 2),
        (1002, 9002, 1003, 9003, (162_500, 325_000, 1_242), 1),
        (1001, 9001, 1003, 9003, (50_000, 350_000, 1_303), 2),
        (1003, 9003, 1001, 9001, (162_500, 306_250, 1_272), 2),
        (1001, 9001, 1002, 9002, (156_250, 318_750, 1_227), 2),
        (1004, 9004, 1002, 9002, (55_000, 348_000, 1_290), 2),
        (7777, 8888, 1002, 9002, (100_000, 320_000, 1_250), 2),  # pre not a soma
        (1001, 9001, 6666, 5555, (110_000, 330_000, 1_260), 1),  # post not a soma
    ]


def shard_bytes(syn=None):
    syn = synapses() if syn is None else syn
    return make_shard([_record(*s) for s in syn])


# ---------------------------------------------------------------------------------------------
# 0. the writer's own round trip -- validate the validator before it validates anything
# ---------------------------------------------------------------------------------------------
def test_the_in_test_writer_round_trips_through_the_module_decoder():
    syn = synapses()
    got = list(H.iter_records(shard_bytes(syn)))
    assert len(got) == len(syn), f"writer/decoder disagree on record count: {len(got)}"
    for want, have in zip(syn, got, strict=True):
        assert have.pre_id == want[0] and have.pre_base == str(want[1])
        assert have.post_id == want[2] and have.post_base == str(want[3])
        assert have.location == want[4] and have.syn_type == want[5]


def test_the_decoder_reads_the_real_release_bytes():
    got = list(H.iter_records(FIXTURE.read_bytes()))
    assert len(got) == 928, f"the pinned fixture holds 928 records, decoded {len(got)}"
    first = got[0]
    assert (first.pre_id, first.post_id, first.syn_type) == (5472528459, 5414315245, 2), first


# ---------------------------------------------------------------------------------------------
# (a) BYTES -- the shard md5 and the pinned header schema
# ---------------------------------------------------------------------------------------------
def test_gate_a_shard_md5_refuses_a_flipped_manifest_digest():
    data = shard_bytes()
    true_md5 = hashlib.md5(data).digest()
    entry = {
        "name": "exportTEST",
        "size": len(data),
        "md5_b64": base64.b64encode(true_md5).decode(),
    }
    H.verify_md5("exportTEST", data, entry)  # CONTROL: the true entry passes

    flipped = bytearray(true_md5)
    flipped[0] ^= 0x10  # one hex digit of the digest
    bad = dict(entry, md5_b64=base64.b64encode(bytes(flipped)).decode())
    with pytest.raises(H.IngestRefused) as e:
        H.verify_md5("exportTEST", data, bad)
    assert "exportTEST" in str(e.value), f"the refusal must name the shard: {e.value}"


def test_gate_a_byte_overrun_past_the_manifest_size_refuses():
    data = shard_bytes()
    entry = {
        "name": "exportTEST",
        "size": len(data) - 1,
        "md5_b64": base64.b64encode(hashlib.md5(data).digest()).decode(),
    }
    with pytest.raises(H.IngestRefused):
        H.verify_md5("exportTEST", data, entry)


def test_gate_a_schema_sha256_refuses_a_reordered_schema_before_any_record_is_read():
    header, sync, schema = real_header()
    r = H.Reader(header)
    meta, _ = H.parse_header(r)
    assert H.verify_schema(meta) == H.AVRO_SCHEMA_SHA256  # CONTROL: the real header passes

    doc = json.loads(schema)
    doc["fields"][0], doc["fields"][1] = doc["fields"][1], doc["fields"][0]
    bad_header, bad_sync = _header_for(json.dumps(doc).encode())
    bad = make_shard([_record(*s) for s in synapses()], header=bad_header, sync=bad_sync)
    with pytest.raises(H.IngestRefused) as e:
        list(H.iter_records(bad))
    assert "schema" in str(e.value).lower(), e.value


# ---------------------------------------------------------------------------------------------
# (b) JOIN -- neuron_id <-> agglo_seg must be the STRICT MAXIMUM of the four pairings
# ---------------------------------------------------------------------------------------------
def test_gate_b_join_counts_all_four_pairings_and_the_declared_one_is_the_maximum():
    tally = H.decode_shard(shard_bytes(), H.join_keys(cells()))
    counts = tally.join_counts
    assert set(counts) == set(H.JOIN_PAIRINGS), counts
    assert counts[H.DECLARED_JOIN] == 6, f"declared pairing both-ends count: {counts}"
    assert counts[("base_neuron_id", "agglo_seg")] == 0, counts
    H.check_join(counts)  # CONTROL: the declared pairing passes


def test_gate_b_refuses_the_base_agglo_pairing_and_prints_the_four_counts(capsys):
    tally = H.decode_shard(shard_bytes(), H.join_keys(cells()))
    with pytest.raises(H.IngestRefused) as e:
        H.check_join(tally.join_counts, declared=("base_neuron_id", "agglo_seg"))
    msg = str(e.value)
    for pre_key, cell_key in H.JOIN_PAIRINGS:
        assert f"{pre_key}<->{cell_key}" in msg, (
            f"count for {pre_key}<->{cell_key} not printed: {msg}"
        )


def test_gate_b_refuses_a_tie_too():
    """The rule is STRICT maximum: a pairing that only ties is not evidence."""
    counts = dict.fromkeys(H.JOIN_PAIRINGS, 7)
    with pytest.raises(H.IngestRefused):
        H.check_join(counts)


# ---------------------------------------------------------------------------------------------
# (c) UNITS -- the declared (8, 8, 33) nm/voxel scaling must be closest to 1 on EVERY axis
# ---------------------------------------------------------------------------------------------
def _ratios():
    tally = H.decode_shard(shard_bytes(), H.join_keys(cells()))
    return H.extent_ratios(tally.loc_bounds, H.soma_bounds(cells()))


def test_gate_c_declared_scaling_is_closest_to_one_on_every_axis():
    ratios = _ratios()
    assert set(ratios) == set(H.CANDIDATE_SCALES), ratios
    H.check_units(ratios)  # CONTROL: the declared scaling passes
    dec = ratios[H.DECLARED_SCALE]
    for name, other in ratios.items():
        if name == H.DECLARED_SCALE:
            continue
        for ax in range(3):
            assert abs(dec[ax] - 1.0) <= abs(other[ax] - 1.0), (name, ax, dec, other)


def test_gate_c_refuses_the_nm_inversion():
    with pytest.raises(H.IngestRefused) as e:
        H.check_units(_ratios(), declared="inversion")
    assert "inversion" in str(e.value), e.value


# ---------------------------------------------------------------------------------------------
# (d) LAYER -- computed, and it must REPRODUCE the pinned 45,404 / 49,377
# ---------------------------------------------------------------------------------------------
def test_gate_d_the_pin_matches_placements_published_agreement():
    agree, total = H.LAYER_AGREEMENT_PIN
    assert (agree, total) == (45404, 49377)
    assert round(agree / total, 4) == LAYER_RULE_AGREEMENT, (
        f"{agree}/{total} = {agree / total} does not round to placement's "
        f"LAYER_RULE_AGREEMENT {LAYER_RULE_AGREEMENT}"
    )


def test_gate_d_nanometres_instead_of_microns_collapses_every_soma_into_layer_1():
    """specs/011 quickstart Scenario 3's red, made numeric."""
    labels = {c["agglo_seg"]: "Layer 2" for c in cells()}
    good_agree, good_total = H.layer_agreement(cells(), labels)
    assert good_total == len(CELLS)
    nm_agree, nm_total = H.layer_agreement(cells(), labels, divisor=1.0)
    assert nm_total == len(CELLS)
    assert nm_agree < good_agree, (
        f"feeding nanometres must collapse the agreement: {nm_agree} vs {good_agree}"
    )
    computed = [H.layer_of_cell(c, divisor=1.0) for c in cells()]
    assert set(computed) == {"Layer 1"}, f"the nm red must land every soma in Layer 1: {computed}"


def test_gate_d_check_layer_refuses_anything_but_the_pin():
    H.check_layer(*H.LAYER_AGREEMENT_PIN)  # CONTROL
    with pytest.raises(H.IngestRefused):
        H.check_layer(45403, 49377)
    with pytest.raises(H.IngestRefused):
        H.check_layer(45404, 49376)


@pytest.mark.skipif(
    not (H.DATA_DIR / "agglo_20200916c3_cell_data.json").exists(), reason="data/h01 absent"
)
def test_gate_d_reproduces_45404_of_49377_on_the_real_cell_file():
    real = H.load_cells()
    labels = H.load_layer_labels()
    assert H.layer_agreement(real, labels) == H.LAYER_AGREEMENT_PIN


# ---------------------------------------------------------------------------------------------
# (e) SIGN -- two instruments; the swapped map scores exactly 1 - r, so > 0.5 has its red built in
# ---------------------------------------------------------------------------------------------
def _edges():
    return H.decode_shard(shard_bytes(), H.join_keys(cells())).edges


def test_gate_e_two_instruments_agree_above_half_and_the_swapped_map_is_its_complement():
    sign_of = H.dale_signs(cells())
    agree, total = H.sign_agreement(_edges(), sign_of)
    assert total > 0, "no joined synapse carried a signed pre -- the arm would be vacuous"
    H.check_sign(agree, total)  # CONTROL: the upstream map passes
    swapped = {2: -1, 1: +1}
    s_agree, s_total = H.sign_agreement(_edges(), sign_of, type_sign=swapped)
    assert s_total == total and s_agree == total - agree, (
        f"the swapped map must score exactly 1 - r ({s_agree}/{s_total} vs {agree}/{total})"
    )
    with pytest.raises(H.IngestRefused):
        H.check_sign(s_agree, s_total)


def test_gate_e_disagreement_is_reported_not_rounded_to_zero():
    sign_of = H.dale_signs(cells())
    agree, total = H.sign_agreement(_edges(), sign_of)
    assert 0 <= total - agree, (agree, total)
    assert H.sign_report(agree, total).count("disagree") == 1, H.sign_report(agree, total)


# ---------------------------------------------------------------------------------------------
# (f) DETERMINISM and ORDER-INDEPENDENCE
# ---------------------------------------------------------------------------------------------
def _two_shards():
    half = len(synapses()) // 2
    return {
        "shardA": shard_bytes(synapses()[:half]),
        "shardB": shard_bytes(synapses()[half:]),
    }


def test_gate_f_artifact_is_byte_identical_under_reversed_shard_order(tmp_path):
    blobs = _two_shards()
    fwd = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    rev = H.ingest(["shardB", "shardA"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    a = H.write_artifact(fwd, tmp_path / "fwd.npz")
    b = H.write_artifact(rev, tmp_path / "rev.npz")
    assert a == b, f"forward and reverse shard order produced different artifacts: {a} vs {b}"


def test_gate_f_unsorted_encounter_order_is_the_red(tmp_path):
    """RED: append edges in encounter order without sorting -> the two orders differ."""
    blobs = _two_shards()
    fwd = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    rev = H.ingest(["shardB", "shardA"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    a = H.write_artifact(fwd, tmp_path / "u_fwd.npz", sort_edges=False)
    b = H.write_artifact(rev, tmp_path / "u_rev.npz", sort_edges=False)
    assert a != b, (
        "the unsorted control did not diverge, so the sorted arm above proves nothing about "
        "order-independence -- it would be measuring a Counter that happens to iterate the same way"
    )


def test_gate_f_population_construction_is_deterministic(tmp_path):
    blobs = _two_shards()
    graph = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    path = tmp_path / "g.npz"
    H.write_artifact(graph, path)
    trains, blobs_out = [], []
    for _ in range(2):
        state, syn, kt = H.population(path)
        blobs_out.append(state.to_blob())
        trains.append(H.drive_train(kt, ticks=30))
    assert blobs_out[0] == blobs_out[1], "two constructions gave different KernelState bytes"
    assert trains[0] == trains[1], "two constructions gave different 30-tick spike trains"


# ---------------------------------------------------------------------------------------------
# (g) P4b -- the prior never enters the lake
# ---------------------------------------------------------------------------------------------
def _lake_record_count():
    from snn_data_lake.substrate.home import BinLog, knowledge_path

    p = Path(knowledge_path())
    if not p.exists():
        return 0
    return sum(1 for _ in BinLog(str(p)).replay())


def test_gate_g_a_full_ingest_and_population_add_no_lake_record(tmp_path):
    blobs = _two_shards()
    with throwaway_home():
        before = _lake_record_count()
        graph = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
        path = tmp_path / "p4b.npz"
        H.write_artifact(graph, path)
        state, syn, kt = H.population(path)
        H.drive_train(kt, ticks=10)
        after = _lake_record_count()
        assert after == before, (
            f"the H01 prior entered the lake: {before} -> {after} records. P4b: the lake is what "
            "the brain LEARNED; this is what a microscope saw."
        )

        # RED: the same artifact deliberately persisted MUST move the counter, or the arm above
        # is measuring a lake that cannot be written at all.
        from snn_data_lake.substrate.home import save_named_state

        assert save_named_state("t062_p4b_red", path.read_bytes())
        assert _lake_record_count() > before, (
            "a deliberate save_named_state did not raise the record count -- the P4b arm above "
            "is blind, not green"
        )


# ---------------------------------------------------------------------------------------------
# (h) LIVENESS of the population
# ---------------------------------------------------------------------------------------------
def test_gate_h_driving_one_unit_propagates_within_30_ticks(tmp_path):
    blobs = _two_shards()
    graph = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    path = tmp_path / "live.npz"
    H.write_artifact(graph, path)
    _state, _syn, kt = H.population(path)
    assert H.propagates(kt, ticks=30), "the seeded population is inert: no spike beyond the drive"

    _s0, _y0, kt0 = H.population(path, zero_weights=True)
    assert not H.propagates(kt0, ticks=30), (
        "the zeroed-weight control still propagated -- the liveness arm is not measuring the graph"
    )


# ---------------------------------------------------------------------------------------------
# The population's shape (2.2): signed neurons only, seven laminar bands, fan-in-share weights
# ---------------------------------------------------------------------------------------------
def test_population_units_are_the_signed_neurons_in_seven_bands(tmp_path):
    blobs = _two_shards()
    graph = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    path = tmp_path / "shape.npz"
    H.write_artifact(graph, path)
    art = H.load_artifact(path)
    assert "oligodendrocyte" not in json.dumps(art["meta"])
    assert len(art["agglo_id"]) == 4, "only the four SIGNED somata may become units"
    assert list(art["agglo_id"]) == sorted(art["agglo_id"]), "node ids must be sorted"
    assert set(art["sign"].tolist()) <= {1, -1}, art["sign"]
    state, _syn, _kt = H.population(path)
    assert set(state.ranges) <= set(LAYER_NAMES + (WHITE_MATTER,)), state.ranges


def test_population_fan_in_shares_sum_to_one_per_post(tmp_path):
    blobs = _two_shards()
    graph = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    path = tmp_path / "w.npz"
    H.write_artifact(graph, path)
    art = H.load_artifact(path)
    totals = {}
    for post, w in zip(art["post_idx"].tolist(), H.fan_in_shares(art).tolist(), strict=True):
        totals[post] = totals.get(post, 0.0) + abs(w)
    for post, s in totals.items():
        assert math.isclose(s, 1.0, rel_tol=1e-9), f"post {post} fan-in |weights| sum to {s}, not 1"


def test_population_delay_is_one_tick_everywhere_and_disclosed(tmp_path):
    blobs = _two_shards()
    graph = H.ingest(["shardA", "shardB"], cells=cells(), fetch=blobs.get, verify_bytes=False)
    path = tmp_path / "d.npz"
    H.write_artifact(graph, path)
    _state, syn, _kt = H.population(path)
    assert syn.max_delay == 1, syn.max_delay
    meta = H.load_artifact(path)["meta"]
    assert "DISCLOSED" in meta["delay"], meta["delay"]


def test_thresholds_and_decay_are_imported_from_the_live_lif_owner():
    from snn_data_lake.cortex.lif_neuron import LIFNeuron

    lif = LIFNeuron()
    assert H.LIF_DEFAULTS["thr"] == lif.v_threshold
    assert H.LIF_DEFAULTS["decay"] == lif.decay
    assert H.LIF_DEFAULTS["v_reset"] == lif.v_reset


# ---------------------------------------------------------------------------------------------
# Provenance and licence
# ---------------------------------------------------------------------------------------------
def test_the_licence_passes_the_ingest_guard_and_the_attribution_names_fr_007():
    from snn_data_lake.substrate.ingest_guard import license_ok

    assert license_ok(H.LICENSE_ID)
    assert "CC-BY" in H.ATTRIBUTION.upper() or "Creative Commons" in H.ATTRIBUTION
    assert "FR-007" in H.ATTRIBUTION


def test_the_manifest_pins_166_shards_and_the_schema_sha():
    m = H.load_manifest()
    assert m["shard_count"] == 166 and len(m["shards"]) == 166
    assert sum(s["size"] for s in m["shards"]) == m["total_bytes"] == 32859876087
    assert m["avro_schema_sha256"] == H.AVRO_SCHEMA_SHA256
    assert m["codec"] == "null" and tuple(m["voxel_nm"]) == H.DECLARED_VOXEL_NM


def test_every_shard_url_passes_the_web_ingest_firewall():
    from urllib.parse import urlparse

    m = H.load_manifest()
    for name in (m["shards"][0]["name"], m["shards"][-1]["name"]):
        u = H.shard_url(name)
        assert urlparse(u).scheme == "https" and u.startswith(m["base_url"])


# ---------------------------------------------------------------------------------------------
# The module's bare self-check is the offline gate run_selfchecks drives
# ---------------------------------------------------------------------------------------------
def test_the_bare_self_check_is_offline_and_green():
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), throwaway_home():
        rc = H.main([])
    out = buf.getvalue()
    assert rc == 0, out
    assert out.isascii(), "self-check output must be ASCII (Windows cp1252)"
    for gate in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"):
        assert gate in out, f"gate {gate} is not reported by the self-check:\n{out}"


def test_struct_module_is_used_for_doubles_not_a_hand_rolled_ieee():
    """The decoder's double reader is stdlib struct -- pinned so a 'simplification' cannot
    replace it with an arithmetic reconstruction that quietly loses subnormals."""
    r = H.Reader(struct.pack("<d", -1.5))
    assert r.double() == -1.5
