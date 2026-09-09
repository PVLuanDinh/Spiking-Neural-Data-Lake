"""specs/022 T062 -- every gate of the H01 c3 synapse-graph ingest, each with its RED.

Pre-registration: `specs/022-unified-brain-arc/evidence/t062-h01-ingest.md` 2.3 (a)..(h) as
AMENDED by its dated 2.9. Each gate below is a REFUSAL with a constructed red run in the same
command as its control, and no gate carries an invented threshold: (b) requires the declared join
to be the strict maximum of four measured counts, (c) requires the declared scaling's extent
ratios to be the closest to 1 on every axis of four candidates, (d) requires the layer rule to
REPRODUCE `placement.LAYER_RULE_AGREEMENT`'s pinned 45,404 / 49,377 exactly, and (e') requires
the two POST-anchored E/I instruments to correlate ABOVE the 95th percentile of the same
correlation under 1,000 permutations of the CSV's segment labels.

2.9 RETIRED GATE (e), and P2 means the old mechanism is DELETED rather than flagged off: the
pre-anchored comparison (the PRE cell's Dale sign against the export's per-synapse `type`) was
comparing two different quantities, refused at 0.3004 / 0.3014 on the real export, and is gone.
`test_the_pre_anchored_sign_gate_is_removed_not_flagged_off` is the assertion that it stayed
gone. What the pre side still owes is DISCLOSURE, not a refusal -- five measured numbers that
nothing filters on, pinned by `test_the_disclosures_are_measured_and_never_filtered_on`.

NO NETWORK. The shard fixtures come from the module's OWN null-codec Avro writer (P2: one
encoder in the tree, not one here and one there that drift), and it builds them on the REAL
release header from `tests/fixtures/h01_c3_shard0_head.avro` -- so the pinned `avro.schema`
sha256 is checked against bytes Google published rather than bytes this suite produced.

A writer and a decoder that share an author cannot catch a bug they SHARE, so the independent
control is `test_the_decoder_reads_the_real_release_bytes`: real release bytes checked against
values a different reader produced (the export's JSON twin -- pre 5472528459, post 5414315245,
type 2). Read the round-trip test as "the fixture is well-formed", never as "the decoder is
right"; that second claim is the real-bytes arm's.

One arm (the real 45,404 / 49,377 reading) skips when `data/h01/` is absent -- `data/` is
gitignored, so CI has neither the 11 MB cell file nor the CSV. It is NOT the only cover for gate
(d): the nm-instead-of-microns red, the off-by-one refusals and the assertion that the pinned
pair rounds to `placement.LAYER_RULE_AGREEMENT` all run unconditionally.
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
# ONE Avro writer, and it lives in the MODULE (P2). A second encoder here could drift from the
# decoder it is supposed to exercise, and the drift would look like a mechanism bug. The honest
# consequence -- a writer/decoder pair cannot catch a bug they SHARE -- is covered by
# `test_the_decoder_reads_the_real_release_bytes` below, which checks real release bytes against
# values a different reader produced (the export's JSON twin).
# ---------------------------------------------------------------------------------------------
_record = H.encode_record
make_shard = H.synthetic_shard
real_header = H.real_header
_header_for = H.ocf_header

CELLS = H._SELF_CELLS


def cells():
    return H.self_cells()


def ei():
    """(e') instrument 2's fixture: the e/i CSV's own per-segment counts for this world."""
    return H.self_ei()


def _ingest(names, **kw):
    """Every offline ingest in this file, with the fixture world's two instruments attached.

    ONE call shape, so a new gate that needs a new input shows up as one edit here rather than as
    a dozen call sites that drift apart -- which is how the fixture and the self-check's fixture
    stopped being the same world once already.
    """
    kw.setdefault("cells", cells())
    kw.setdefault("ei_counts", ei())
    kw.setdefault("verify_bytes", False)
    return H.ingest(names, **kw)


def synapses():
    """(pre, pre_base, post, post_base, location_in_8nm_voxels, syn_type) -- the module's own
    self-check world, so the offline gate and this suite measure the SAME fixture."""
    return list(H._SELF_SYNAPSES)


def shard_bytes(syn=None):
    return H.self_shard(syn)


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
    labels = {c["agglo_seg"]: H.layer_of_cell(c) for c in cells()}
    good_agree, good_total = H.layer_agreement(cells(), labels)
    assert good_total == len(CELLS) and good_agree == len(CELLS)
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
# (e') E/I, POST-ANCHORED -- the export's per-POST excitatory input fraction against the CSV's
# own per-segment counts, against a permutation band. 2.9's replacement for the retired gate (e).
# ---------------------------------------------------------------------------------------------
def _edges():
    return H.decode_shard(shard_bytes(), H.join_keys(cells())).edges


def _post_fractions():
    """Instrument 1: the export's own `type` aggregated per POST cell into that cell's excitatory
    INPUT fraction. POST-anchored, which is the whole point of 2.9 -- the retired gate compared a
    post-anchored quantity to a pre-anchored one and measured their difference, not their
    agreement."""
    return H.post_excitatory_fractions(
        H.decode_shard(shard_bytes(), H.join_keys(cells())).post_type
    )


def test_gate_e_prime_the_two_post_anchored_instruments_beat_the_permutation_band():
    band = H.ei_correlation(_post_fractions(), H.csv_fractions(ei()))
    assert band["n_cells"] >= 20, f"too few joined post cells for a permutation test: {band}"
    assert band["r"] > band["bar"], band
    H.check_ei(band)  # CONTROL: the real labels pass


def test_gate_e_prime_shuffled_segment_labels_fall_inside_the_band_and_refuse():
    """2.9's constructed RED, and it is the shuffled labels themselves -- the same values, the
    same joined cells, only the pairing destroyed."""
    real = H.ei_correlation(_post_fractions(), H.csv_fractions(ei()))
    shuffled = H.ei_correlation(_post_fractions(), H.csv_fractions(H.self_ei(shuffle=True)))
    assert shuffled["n_cells"] == real["n_cells"], (
        "the red must change the PAIRING and nothing else; a red that also changes which cells "
        "are joined is measuring two things at once"
    )
    assert shuffled["r"] <= shuffled["bar"], shuffled
    with pytest.raises(H.IngestRefused) as e:
        H.check_ei(shuffled)
    assert "permutation" in str(e.value).lower(), e.value


def test_gate_e_prime_the_bar_is_measured_from_the_same_data_not_invented():
    """No threshold literal anywhere in this gate: the bar is the 95th percentile of the SAME
    statistic under the null that segment labels carry no information, so it is re-measured on
    every run and cannot be tuned to admit a reading."""
    band = H.ei_correlation(_post_fractions(), H.csv_fractions(ei()))
    assert band["trials"] == H.EI_PERMUTATIONS and band["percentile"] == H.EI_PERCENTILE
    assert band["null_min"] <= band["bar"] <= band["null_max"], band
    assert abs(band["null_mean"]) < 0.25, f"a null not centred near 0 is not a null: {band}"


def test_gate_e_prime_is_deterministic_in_its_seed():
    a = H.ei_correlation(_post_fractions(), H.csv_fractions(ei()))
    b = H.ei_correlation(_post_fractions(), H.csv_fractions(ei()))
    assert a == b, "the permutation band moved between two runs of the same seed"


def test_gate_e_prime_refuses_a_vacuous_reading():
    with pytest.raises(H.IngestRefused):
        H.check_ei(H.ei_correlation({"1001": 1.0}, {"1001": 1.0}, trials=8))


def test_the_pre_anchored_sign_gate_is_removed_not_flagged_off():
    """P2: exactly one mechanism survives. 2.9 retired gate (e) as the instrument's defect, so
    the refusal is DELETED -- not kept behind a flag, an env var or a default argument."""
    assert not hasattr(H, "check_sign"), "the retired pre-anchored sign gate is still callable"
    src = Path(H.__file__).read_text(encoding="utf-8")
    assert "def check_sign" not in src, "check_sign survives in the module source"
    blobs = _two_shards()
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
    assert "sign_rate" not in graph.gates, graph.gates
    assert "ei" in graph.gates, graph.gates


# ---------------------------------------------------------------------------------------------
# The disclosures 2.9 requires: measured, printed, and NEVER filtered on
# ---------------------------------------------------------------------------------------------
def test_the_disclosures_are_measured_and_never_filtered_on(tmp_path):
    blobs = _two_shards()
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
    d = graph.gates["disclosures"]
    for key in (
        "pre_end_soma_hits",
        "pre_end_soma_hits_non_neuronal",
        "pre_end_non_neuronal_fraction",
        "soma_to_soma_synapses",
        "synapses_with_dale_signed_pre",
        "synapses_with_both_ends_signed",
        "signed_pre_export_type_disagreement",
    ):
        assert key in d, f"{key} is not disclosed: {d}"
    assert d["soma_to_soma_synapses"] == 6, d
    assert 0.0 < d["pre_end_non_neuronal_fraction"] < 1.0, (
        f"the fixture must carry BOTH neuronal and non-neuronal pre-end hits, or the disclosure "
        f"is untested: {d}"
    )
    assert (
        d["pre_end_soma_hits_non_neuronal"] / d["pre_end_soma_hits"]
        == d["pre_end_non_neuronal_fraction"]
    )
    # NEVER FILTERED ON: every both-ends-signed synapse reaches the artifact whatever its export
    # type says. Filtering edges to agree with the prior would be selecting the data to fit the
    # seed -- 2.9 names that explicitly.
    path = tmp_path / "disc.npz"
    H.write_artifact(graph, path)
    art = H.load_artifact(path)
    assert int(art["n_syn"].sum()) == d["synapses_with_both_ends_signed"], (
        "an edge was dropped between the join and the artifact"
    )
    assert int(art["n_exc"].sum() + art["n_inh"].sum()) <= int(art["n_syn"].sum())
    assert art["meta"]["disclosures"] == d, "the artifact meta must carry the same numbers"


def test_a_refusing_ingest_still_carries_every_gate_reading():
    """2.7: 'no artifact is written, the readings are recorded'. A refusal that throws away the
    join counts and the extent ratios makes the row unauditable -- the refusal IS the result, so
    it has to carry the evidence for it."""
    blobs = _two_shards()
    with pytest.raises(H.IngestRefused) as e:
        _ingest(["shardA", "shardB"], fetch=blobs.get, ei_counts=H.self_ei(shuffle=True))
    r = e.value.readings
    assert r["join_counts"][f"{H.DECLARED_JOIN[0]}<->{H.DECLARED_JOIN[1]}"] == 6, r
    assert set(r["unit_ratios"]) == set(H.CANDIDATE_SCALES), r
    assert "disclosures" in r, r
    assert r["refusals"] and any("ei" in x for x in r["refusals"]), r


def test_a_refusal_names_every_gate_that_refused_not_just_the_first():
    blobs = _two_shards()
    with pytest.raises(H.IngestRefused) as e:
        _ingest(
            ["shardA", "shardB"],
            fetch=blobs.get,
            ei_counts=H.self_ei(shuffle=True),
            declared_join=("base_neuron_id", "agglo_seg"),
        )
    named = {x.split(":")[0] for x in e.value.readings["refusals"]}
    assert named == {"join", "ei"}, e.value.readings["refusals"]


def test_the_pre_side_is_disclosed_as_a_rate_not_rounded_to_zero():
    sign_of = H.dale_signs(cells())
    agree, total = H.sign_agreement(_edges(), sign_of)
    assert 0 <= total - agree, (agree, total)
    assert H.sign_report(agree, total).count("disagree") == 1, H.sign_report(agree, total)


# ---------------------------------------------------------------------------------------------
# (f) DETERMINISM and ORDER-INDEPENDENCE
# ---------------------------------------------------------------------------------------------
def _two_shards():
    """The MODULE's own two-shard split (P2: one splitter, so the offline gate and this suite
    measure the same two shards). Both shards carry soma-to-soma edges on purpose -- the
    unsorted-order red below needs two Counters that actually differ, and a naive halfway cut of
    the synapse list puts every edge in the first shard once the post-only e/i traffic is
    appended."""
    a, b = H.self_shard_pair()
    return {"shardA": a, "shardB": b}


def test_gate_f_artifact_is_byte_identical_under_reversed_shard_order(tmp_path):
    blobs = _two_shards()
    fwd = _ingest(["shardA", "shardB"], fetch=blobs.get)
    rev = _ingest(["shardB", "shardA"], fetch=blobs.get)
    a = H.write_artifact(fwd, tmp_path / "fwd.npz")
    b = H.write_artifact(rev, tmp_path / "rev.npz")
    assert a == b, f"forward and reverse shard order produced different artifacts: {a} vs {b}"


def test_gate_f_unsorted_encounter_order_is_the_red(tmp_path):
    """RED: append edges in encounter order without sorting -> the two orders differ."""
    blobs = _two_shards()
    fwd = _ingest(["shardA", "shardB"], fetch=blobs.get)
    rev = _ingest(["shardB", "shardA"], fetch=blobs.get)
    a = H.write_artifact(fwd, tmp_path / "u_fwd.npz", sort_edges=False)
    b = H.write_artifact(rev, tmp_path / "u_rev.npz", sort_edges=False)
    assert a != b, (
        "the unsorted control did not diverge, so the sorted arm above proves nothing about "
        "order-independence -- it would be measuring a Counter that happens to iterate the same way"
    )


def test_gate_f_population_construction_is_deterministic(tmp_path):
    blobs = _two_shards()
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
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
        graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
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
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
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
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
    path = tmp_path / "shape.npz"
    H.write_artifact(graph, path)
    art = H.load_artifact(path)
    assert 1005 not in art["agglo_id"].tolist(), "an unsigned (glial) soma became a unit"
    assert len(art["agglo_id"]) == 4, "only the four SIGNED somata may become units"
    assert list(art["agglo_id"]) == sorted(art["agglo_id"]), "node ids must be sorted"
    assert set(art["sign"].tolist()) <= {1, -1}, art["sign"]
    state, _syn, _kt = H.population(path)
    assert set(state.ranges) <= set(LAYER_NAMES + (WHITE_MATTER,)), state.ranges


def test_population_fan_in_shares_sum_to_one_per_post(tmp_path):
    blobs = _two_shards()
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
    path = tmp_path / "w.npz"
    H.write_artifact(graph, path)
    art = H.load_artifact(path)
    totals = {}
    for post, w in zip(art["post_idx"].tolist(), H.fan_in_shares(art).tolist(), strict=True):
        totals[post] = totals.get(post, 0.0) + abs(w)
    for post, s in totals.items():
        assert math.isclose(s, 1.0, rel_tol=1e-9), f"post {post} fan-in |weights| sum to {s}, not 1"


def test_population_fan_in_partitions_the_whole_unit_index(tmp_path):
    """The assertion above is keyed on `post_idx`, so a unit with NO in-edge is structurally
    invisible to it -- it never appears in `totals` and nothing ever checks it.

    That blind spot hides a real, measured falsehood. On the pinned artifact the T062 panel
    measured 11,227 of 14,215 units with at least one in-edge (max |1 - sum| = 1.1e-15), 2,988
    with none (sum exactly 0.0), 3,609 with no out-edge and 1,702 with neither -- so 2.2's and
    the artifact meta's "a unit's summed |input| is 1 by construction" is universally quantified
    over units and false for 21 % of them.

    PARTITION, never disjoin. Asserting "every entry is 1.0 OR 0.0" is strictly WEAKER than the
    assertion above: it admits a unit that HAS in-edges and sums to 0.0, which is precisely the
    case `fan_in_shares`' own `where=denom > 0` guard exists to produce. Every unit WITH an
    in-edge must sum to 1, every unit WITHOUT must sum to exactly 0, and both counts are
    reported rather than inferred.

    This arm runs on the SYNTHETIC artifact, and on it the zero side is empty -- so the red is
    CONSTRUCTED below rather than found, and the real artifact's 2,988 has no in-command red
    until the next legitimate re-ingest records the four counts in the pin. Said plainly rather
    than left for the next reader."""
    import numpy as np

    def partitioned_counts(art):
        n = int(art["agglo_id"].size)
        has_in = np.zeros(n, dtype=bool)
        has_in[art["post_idx"]] = True
        totals = np.zeros(n, dtype=np.float64)
        np.add.at(totals, art["post_idx"], np.abs(H.fan_in_shares(art)))
        for i in range(n):
            if has_in[i]:
                assert abs(totals[i] - 1.0) <= 1e-9, (
                    f"unit {i} has an in-edge but its fan-in |weights| sum to {totals[i]}, not 1"
                )
            else:
                assert totals[i] == 0.0, (
                    f"unit {i} has NO in-edge but its fan-in |weights| sum to {totals[i]}, not 0"
                )
        with_in = int(has_in.sum())
        assert with_in + int((~has_in).sum()) == n, "the partition lost a unit"
        return with_in, n - with_in

    blobs = _two_shards()
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
    path = tmp_path / "fanin.npz"
    H.write_artifact(graph, path)
    art = H.load_artifact(path)
    with_in, without_in = partitioned_counts(art)
    assert with_in + without_in == int(art["agglo_id"].size)
    assert without_in == 0, (
        f"the synthetic artifact grew an in-edge-less unit ({without_in}); the docstring's "
        "'the zero side is empty here' no longer describes this fixture"
    )

    # THE RED, CONSTRUCTED -- the synthetic artifact has no in-edge-less unit, so without this
    # the zero half of the partition never fires and the arm is blind, not green. One edge over
    # three units leaves units 0 and 2 with no in-edge at all, which is the shape the
    # post_idx-keyed form cannot see.
    iso = {
        "agglo_id": np.array([10, 20, 30], dtype=np.int64),
        "sign": np.array([1, -1, 1], dtype=np.int8),
        "pre_idx": np.array([0], dtype=np.int32),
        "post_idx": np.array([1], dtype=np.int32),
        "n_syn": np.array([4], dtype=np.int32),
    }
    assert partitioned_counts(iso) == (1, 2), partitioned_counts(iso)
    assert {int(p) for p in iso["post_idx"].tolist()} == {1}, (
        "the post_idx-keyed form must be blind to units 0 and 2 -- if it is not, this red does "
        "not reconstruct the defect the partition exists to close"
    )


def test_population_delay_is_one_tick_everywhere_and_disclosed(tmp_path):
    blobs = _two_shards()
    graph = _ingest(["shardA", "shardB"], fetch=blobs.get)
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


def test_the_pin_records_every_stage_and_never_erases_an_earlier_one():
    """2.4: 'the stage-B artifact REPLACES stage A's (the pin records BOTH)'. A pin writer that
    overwrites is a pin that quietly loses the cheaper stage's readings the moment the expensive
    one runs -- and with a refusal there is no artifact, so the readings ARE the record."""
    a = {"shards": ["export000000000000", "export000000000034"], "verdict": "REFUSED", "n": 1}
    b = {"shards": ["export000000000000", "export000000000165"], "verdict": "REFUSED", "n": 2}
    first = H.merge_pin(None, a)
    assert list(first["stages"]) == [H.stage_key(a["shards"])], first
    both = H.merge_pin(first, b)
    assert set(both["stages"]) == {H.stage_key(a["shards"]), H.stage_key(b["shards"])}, both
    assert both["stages"][H.stage_key(a["shards"])]["n"] == 1, "stage A's readings were erased"
    assert both["latest"] == H.stage_key(b["shards"])

    # a LEGACY top-level record (what the first stage-A run wrote) must migrate, not vanish
    legacy = dict(a, utc="2026-09-09T20:38:52Z")
    migrated = H.merge_pin(legacy, b)
    assert migrated["stages"][H.stage_key(a["shards"])]["utc"] == "2026-09-09T20:38:52Z", migrated


def test_a_re_run_of_the_same_stage_lands_beside_the_reading_it_replaces():
    """2.9 authorises a SECOND run over the SAME 166 shards, under a different gate set. Its stage
    key is identical to the refused run's, so a pin keyed by stage alone silently overwrites the
    refusal it is meant to stand beside -- and with no artifact written, that refusal IS the row's
    only result. The same rule the legacy-migration arm above exists for: a fix that erases the
    readings it exists to protect is worse than the defect."""
    shards = ["export000000000000", "export000000000165"]
    first = H.merge_pin(None, {"shards": shards, "verdict": "REFUSED", "gates": "e"})
    second = H.merge_pin(first, {"shards": shards, "verdict": "PASSED", "gates": "e-prime"})
    assert len(second["stages"]) == 2, f"the refused run was erased by its re-run: {second}"
    kept = [v for v in second["stages"].values() if v["verdict"] == "REFUSED"]
    assert kept and kept[0]["gates"] == "e", second
    assert second["stages"][second["latest"]]["verdict"] == "PASSED", second
    assert H.stage_key(shards) in second["stages"], (
        "the FIRST reading keeps the plain stage key; a re-run may not take a name that is "
        "already spoken for"
    )
    third = H.merge_pin(second, {"shards": shards, "verdict": "PASSED", "gates": "e-prime-again"})
    assert len(third["stages"]) == 3, third


def test_the_pin_writer_marks_both_outcomes_and_spells_each_one_way():
    """The refusal path wrote `verdict: "REFUSED"` while the success path wrote no verdict at
    all, so the machine-readable record carried a refusal marker and no counterpart: a passing
    stage was inferable only from a non-null `artifact_sha256` plus the absence of `refusals`.
    The fixtures above used "OK", a word the writer never emitted, so the divergence had no arm.

    Asserted on the module's own dict LITERALS rather than by running an ingest, because the
    success path is reachable only through a 166-shard network fetch. The red is reachable:
    delete either key, or spell either token a second way, and this goes red.

    2.7 still governs what the word may mean -- the row has refusals and no WIN/LOSS. "PASSED"
    says only that no gate raised; the readings that justify it are the pin's `gates` block, and
    a later reader who quotes the token instead of the numbers is reading it wrong."""
    import ast

    verdicts = set()
    for node in ast.walk(ast.parse(Path(H.__file__).read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "verdict":
                assert isinstance(value, ast.Constant), ast.dump(value)
                verdicts.add(value.value)
    assert verdicts == {"REFUSED", "PASSED"}, (
        f"the pin writer's verdict vocabulary is {sorted(verdicts)}; both outcomes must be "
        "written by the run that produced them, and neither may be spelled two ways"
    )


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
    for gate in ("(a)", "(b)", "(c)", "(d)", "(e')", "(f)", "(g)", "(h)"):
        assert gate in out, f"gate {gate} is not reported by the self-check:\n{out}"
    assert "disclosures" in out, f"the 2.9 disclosures are not printed:\n{out}"
    assert "permutation" in out, f"gate (e')'s band is not printed:\n{out}"


def test_struct_module_is_used_for_doubles_not_a_hand_rolled_ieee():
    """The decoder's double reader is stdlib struct -- pinned so a 'simplification' cannot
    replace it with an arithmetic reconstruction that quietly loses subnormals."""
    r = H.Reader(struct.pack("<d", -1.5))
    assert r.double() == -1.5
