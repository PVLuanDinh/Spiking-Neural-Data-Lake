"""Tests for VSA structure-mapping analogy (relational-system correspondence over spike codes)."""

from snn_data_lake.reasoning import vsa_structure_mapping as sm

SOLAR = [
    ("attracts", "sun", "planet"),
    ("revolves_around", "planet", "sun"),
    ("more_massive", "sun", "planet"),
]
ATOM = [
    ("attracts", "nucleus", "electron"),
    ("revolves_around", "electron", "nucleus"),
    ("more_massive", "nucleus", "electron"),
]
CHAIN_BASE = [("cause", "A", "B"), ("cause", "B", "C"), ("bigger", "A", "C")]
CHAIN_TGT = [("cause", "X", "Y"), ("cause", "Y", "Z"), ("bigger", "X", "Z")]


def test_encode_returns_spike_code_of_right_dim():
    mapper = sm.StructureMapper(dim=1024, seed=0)
    v = mapper.encode(SOLAR)
    assert len(v) == 1024
    assert all(x in (-1, 1) for x in v)


def test_recovers_solar_system_to_atom():
    mapper = sm.StructureMapper(dim=1024, seed=0)
    mapping = mapper.map(SOLAR, ATOM)
    assert mapping == {"sun": "nucleus", "planet": "electron"}
    assert mapper.score(mapping, SOLAR, ATOM) == 1.0


def test_recovers_three_object_causal_chain():
    mapper = sm.StructureMapper(dim=1024, seed=0)
    mapping = mapper.map(CHAIN_BASE, CHAIN_TGT)
    assert mapping == {"A": "X", "B": "Y", "C": "Z"}
    assert mapper.score(mapping, CHAIN_BASE, CHAIN_TGT) == 1.0


def test_non_analogous_target_has_low_systematicity():
    mapper = sm.StructureMapper(dim=1024, seed=0)
    junk = [("likes", "cat", "dog"), ("owns", "dog", "ball"), ("near", "ball", "cat")]
    mapping = mapper.map(SOLAR, junk)
    score = mapper.score(mapping, SOLAR, junk)
    assert score < 0.34
    assert score < mapper.score(mapper.map(SOLAR, ATOM), SOLAR, ATOM)


def test_score_penalises_a_wrong_mapping():
    mapper = sm.StructureMapper(dim=1024, seed=0)
    good = mapper.score({"sun": "nucleus", "planet": "electron"}, SOLAR, ATOM)
    bad = mapper.score({"sun": "electron", "planet": "nucleus"}, SOLAR, ATOM)
    assert good == 1.0
    assert bad < good


def test_deterministic_across_instances():
    m_a = sm.StructureMapper(dim=1024, seed=0).map(SOLAR, ATOM)
    m_b = sm.StructureMapper(dim=1024, seed=0).map(SOLAR, ATOM)
    assert m_a == m_b


def test_works_at_native_resonator_dim_512():
    # dim == vsa.D exercises the direct resonator_network.cleanup_project reuse path.
    mapper = sm.StructureMapper(dim=512, seed=0)
    assert mapper.map(CHAIN_BASE, CHAIN_TGT) == {"A": "X", "B": "Y", "C": "Z"}


def test_mapping_is_one_to_one():
    mapper = sm.StructureMapper(dim=1024, seed=0)
    mapping = mapper.map(CHAIN_BASE, CHAIN_TGT)
    assert len(set(mapping.values())) == len(mapping)
