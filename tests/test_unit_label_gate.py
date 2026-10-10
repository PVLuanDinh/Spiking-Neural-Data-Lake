"""G7 / T115 -- the unit-label producer (placement.place_labeled) and the unit-label gate (G-R7).

Pre-registration: specs/022-unified-brain-arc/evidence/g7-unit-label-gate.md (frozen 854de0e7,
amended da4bcfb7). Every red R1-R19 there is constructed here and must fire; R10 neuters one check
at a time and proves the matching red stops firing. Pure: no Brain is built in this file except
where a test says so,
and those run under substrate.home.throwaway_home().
"""

from __future__ import annotations

import math

import pytest

from snn_data_lake.growth import placement as P

# per mm^2 -- a FIXTURE value (G4's projection, 2.7); G4's Part B supplies the real one
FIXTURE_DENSITY = 2.7e4
SITE = (0.5 / 14, 1.5 / 4)  # ('sp', 'alar'): the cortex cell


def _labeled(n=9, roles=None, seed=13010, **kw):
    args = dict(
        unit_ids=[seed * 100 + i for i in range(n)],
        region="cortex",
        site=SITE,
        problem_id="prob-1",
        batch_id=seed,
        seed=seed,
        roles=roles,
        centre_um=(0.0, 0.0),
        density_per_mm2=FIXTURE_DENSITY,
    )
    args.update(kw)
    return P.place_labeled(n, **args)


# ---- P2: the producer ----------------------------------------------------------------------------


def test_default_draw_is_byte_identical_to_place():
    recs = _labeled(n=50, seed=13011)
    ref = P.place(50, seed=13011)
    assert [(r["position"][4], r["layer"], r["sign"]) for r in recs] == ref


def test_record_carries_exactly_the_contract_fields():
    rec = _labeled(n=1)[0]
    assert set(rec) == {
        "unit_id",
        "batch_id",
        "position",
        "layer",
        "sign",
        "region",
        "problem_id",
        "origin",
    }
    assert len(rec["position"]) == 5 and rec["origin"] == "born"


def test_roles_fix_the_sign_and_keep_depth_in_the_drawn_band():
    roles = ("E",) * 8 + ("I",)
    for seed in range(13012, 13016):
        for rec, role in zip(_labeled(roles=roles, seed=seed), roles, strict=True):
            assert rec["sign"] == (P.EXCITATORY if role == "E" else P.INHIBITORY)
            lo, hi = P.DEPTH_BANDS[rec["layer"]]
            assert lo <= rec["position"][4] < hi


def test_an_excitatory_role_is_never_layer_1():
    recs = _labeled(n=2000, roles=("E",) * 2000, seed=13016)
    assert all(r["layer"] != "Layer 1" for r in recs)


def test_role_conditioned_layer_shares_follow_h01_joint():
    from snn_data_lake.growth.cortical_prior import LAYER_E_FRACTION, LAYER_OCCUPANCY

    occ, ef = LAYER_OCCUPANCY["value"], LAYER_E_FRACTION["value"]
    n = 20000
    recs = _labeled(n=n, roles=("I",) * n, seed=13017)
    w = {L: occ[L] * (1.0 - ef[L]) for L in P.LAYER_NAMES}
    tot = math.fsum(w.values())
    for L in P.LAYER_NAMES:
        got = sum(r["layer"] == L for r in recs) / n
        assert abs(got - w[L] / tot) < 0.02, (L, got, w[L] / tot)


def test_roles_outside_cortex_are_refused():
    with pytest.raises(ValueError, match="measured"):
        _labeled(region="hippocampus", roles=("E",) * 9)


def test_density_has_no_default():
    with pytest.raises(TypeError):
        P.place_labeled(
            1,
            unit_ids=[1],
            region="cortex",
            site=SITE,
            problem_id="p",
            batch_id=2,
            seed=1,
            centre_um=(0.0, 0.0),
        )


def test_offsets_are_deterministic_and_inside_the_batch_disc():
    a, b = _labeled(seed=13018), _labeled(seed=13018)
    assert a == b
    r_um = math.sqrt(9 / (FIXTURE_DENSITY * math.pi)) * 1000.0
    for rec in a:
        x, y = rec["position"][2], rec["position"][3]
        assert math.hypot(x, y) <= r_um + 1e-9


def test_unit_ids_must_match_n():
    with pytest.raises(ValueError):
        _labeled(n=3, unit_ids=[1, 2])


def test_label_of_and_labels_by():
    recs = _labeled(roles=("E",) * 8 + ("I",), seed=13019)
    assert P.label_of(recs, recs[3]["unit_id"]) is recs[3]
    assert P.label_of(recs, 987654) is None
    by_batch = P.labels_by(recs, "batch_id")
    assert by_batch == {13019: [r["unit_id"] for r in recs]}
    by_layer = P.labels_by(recs, "layer")
    assert sum(len(v) for v in by_layer.values()) == 9
    with pytest.raises(ValueError):
        P.labels_by(recs, "colour")


# ---- P1: the gate --------------------------------------------------------------------------------

from snn_data_lake.gates import unit_label_gate as G  # noqa: E402
from snn_data_lake.growth.arealization import NEUROMERES, ZONES  # noqa: E402

PROBLEMS = {"prob-1"}


def _resolve(pid):
    return pid in PROBLEMS


def _cell(nm, zone):
    return ((NEUROMERES.index(nm) + 0.5) / len(NEUROMERES), (ZONES.index(zone) + 0.5) / len(ZONES))


def _good(seed=13010, i=0):
    return dict(_labeled(roles=("E",) * 8 + ("I",), seed=seed)[i])


def _with(rec, **kw):
    out = dict(rec)
    out.update(kw)
    return out


def _at(rec, nm, zone):
    ap, dv = _cell(nm, zone)
    _, _, x, y, depth = rec["position"]
    return _with(rec, position=(ap, dv, x, y, depth))


def _refused_field(rec):
    v = G.check_admission(rec, resolve_problem=_resolve)
    assert not v.admitted, rec
    return v.field


def test_r6_a_full_cortex_label_is_admitted_by_name():
    v = G.check_admission(_good(), resolve_problem=_resolve)
    assert v.admitted and v.region_read == "name"


def test_r1_a_record_without_labels_is_refused():
    assert _refused_field({}) == "unit_id"


@pytest.mark.parametrize("field", ["position", "layer", "sign", "region", "problem_id"])
def test_r2_removing_one_gr7_field_names_that_field(field):
    rec = _good()
    del rec[field]
    assert _refused_field(rec) == field


@pytest.mark.parametrize("field", ["unit_id", "batch_id"])
def test_r2b_removing_an_identity_field_names_it(field):
    rec = _good()
    del rec[field]
    assert _refused_field(rec) == field


def test_r3_scaffold_region_refused_by_name():
    assert _refused_field(_with(_good(), region="insula")) == "region"


@pytest.mark.parametrize("cell", [("r3", "basal"), ("r5", "basal")])
def test_r4_cortex_name_in_refused_territory_refused_by_position(cell):
    assert _refused_field(_at(_good(), *cell)) == "region"


def test_cp3_6_a_non_cortex_region_with_valid_cortical_fields_is_refused():
    # Codex cp3 #6: hippocampus shares ('sp', 'alar'); its cortical layer and sign would pass every
    # other check, so only the owner's cortex-only ruling (O3, 2026-10-04) can refuse it
    v = G.check_admission(_with(_good(), region="hippocampus"), resolve_problem=_resolve)
    assert v.field == "region" and "cortex only" in v.reason


def test_r5_neuromodulation_refused():
    assert _refused_field(_at(_with(_good(), region="neuromodulation"), "r1", "basal")) == "region"


def test_r7_value_rules():
    good = _good()
    lo5, hi5 = P.DEPTH_BANDS["Layer 5"]
    ap, dv, x, y, _ = good["position"]
    cases = {
        "layer": [
            _with(good, layer="White matter"),
            _with(good, layer="Layer 2", position=(ap, dv, x, y, (lo5 + hi5) / 2)),
        ],
        "position": [
            _with(good, position=(ap, dv, x, y, 1.5)),
            _with(good, position=(ap, dv, float("nan"), y, good["position"][4])),
        ],
        "sign": [
            _with(good, sign=0),
            _with(good, sign=None),
            _with(good, sign=True),
            _with(good, sign=1.0),  # a float is not a Dale sign: `1.0 in (1, -1)` was True
            _with(good, sign=-1.0),
            _with(
                good, layer="Layer 1", sign=1, position=(ap, dv, x, y, P.DEPTH_BANDS["Layer 1"][0])
            ),
        ],
        "region": [_with(good, region="atlantis")],
        "problem_id": [_with(good, problem_id="unknown-problem"), _with(good, problem_id="")],
        "origin": [_with(good, origin="derived"), _with(good, origin="legacy")],
    }
    for field, recs in cases.items():
        for rec in recs:
            assert _refused_field(rec) == field, (field, rec)


def test_r19_position_contract():
    good = _good()
    _, dv, x, y, d = good["position"]
    assert _refused_field(_with(good, position=(1.0, dv, x, y, d))) == "position"
    assert _refused_field(_with(good, position=(0.01, float("nan"), x, y, d))) == "position"
    rec = _at(good, "p1", "floor")  # a cell no REGION_COORDS label occupies
    v = G.check_admission(rec, resolve_problem=_resolve)
    assert v.field == "region" and "unknown territory" in v.reason


def test_strict_territory_refuses_the_ambiguous_cell(monkeypatch):
    monkeypatch.setattr(G, "STRICT_TERRITORY", True)
    v = G.check_admission(_good(), resolve_problem=_resolve)
    assert v.field == "region" and "ambiguous territory" in v.reason


def test_born_depths_are_never_inconsistent_with_their_layer():
    # the R7 consistency rule is safe only if lo + r*(hi-lo) never leaves [lo, hi) (advisor)
    bad = [d for d, layer, _ in P.place(100_000, seed=13013) if G.layer_from_depth(d) != layer]
    assert bad == []


def test_r16_r17_duplicate_unit_ids_refused():
    a, b = _good(i=0), _good(i=1)
    with pytest.raises(G.UnitLabelRefused, match="duplicate within batch"):
        G.require_admissible(
            [a, _with(b, unit_id=a["unit_id"])],
            resolve_problem=_resolve,
            known_unit=lambda u: False,
        )
    with pytest.raises(G.UnitLabelRefused, match="duplicate across batches"):
        G.require_admissible([a], resolve_problem=_resolve, known_unit=lambda u: u == a["unit_id"])


def test_ticket_covers_exactly_the_admitted_records_and_is_consumed_once():
    recs = [_good(i=i) for i in range(3)]
    ticket = G.require_admissible(recs, resolve_problem=_resolve, known_unit=lambda u: False)
    with pytest.raises(G.UnitLabelRefused):
        ticket.consume(recs + [_good(i=5)])  # a stranger
    with pytest.raises(G.UnitLabelRefused, match="missing"):
        ticket.consume(recs[:2])  # the ticket binds the whole validated batch
    ticket.consume(recs)
    with pytest.raises(G.UnitLabelRefused, match="consumed"):
        ticket.consume(recs)


def test_cp3_blocker_duplicate_append_under_one_ticket_is_refused():
    # Codex cp3 #1: a ticket for u, then append [u, u] -- consumption must refuse, and a census that
    # sees the duplicate must red even though both rows match the journal
    u = _good(i=0)
    ticket = G.require_admissible([u], resolve_problem=_resolve, known_unit=lambda x: False)
    with pytest.raises(G.UnitLabelRefused, match="duplicate"):
        ticket.consume([u, u])
    journal = {u["unit_id"]: dict(u)}
    out = G.census({"fx": [u, u]}, journal, open_debt={"fx"})["fx"]
    assert out["duplicate"] == 1 and out["red"] is True  # open debt never excuses a duplicate


def test_ticket_is_bound_to_the_validated_records_not_to_the_ids():
    recs = [_good(i=i) for i in range(2)]
    ticket = G.require_admissible(recs, resolve_problem=_resolve, known_unit=lambda u: False)
    recs[0]["sign"] = -recs[0]["sign"]  # mutated AFTER validation, same id
    with pytest.raises(G.UnitLabelRefused, match="differs from the validated"):
        ticket.consume(recs)


def test_r18_restore_is_classified_from_journal_evidence():
    born, other = _good(i=0), _good(i=1)
    journal = {born["unit_id"]: dict(born)}
    legacy_row = {"place_depth": 0.3, "place_layer": "Layer 3", "place_sign": 1}
    assert G.classify_restored([born, other, legacy_row], journal) == ["born", "derived", "legacy"]
    tampered = _with(born, sign=-born["sign"])
    with pytest.raises(G.UnitLabelRefused, match="tampered"):
        G.classify_restored([tampered], journal)


def _saved(rec, **kw):
    out = _with(rec, sleep_id="s1-n1", mask_digest="m1", verdict="kept")
    out.update(kw)
    return out


def test_r9_saved_record_defects_refused():
    rec = _good()
    first = {rec["unit_id"]: rec["sign"]}

    def committed(s, m):
        return (s, m) == ("s1-n1", "m1")

    for bad in (
        {k: v for k, v in _saved(rec).items() if k != "sleep_id"},
        {k: v for k, v in _saved(rec).items() if k != "mask_digest"},
        _saved(rec, verdict="evicted"),
        _saved(rec, sign=-rec["sign"]),
    ):
        v = G.check_saved(
            bad, resolve_problem=_resolve, resolve_committed=committed, first_sign=first
        )
        assert v.outcome == "REFUSED", bad


def test_r9b_only_committed_verdicts_count():
    rec = _good()
    first = {rec["unit_id"]: rec["sign"]}
    commits = {("s1-n1", "m1")}

    def committed(s, m):
        return (s, m) in commits

    def outcome(r):
        return G.check_saved(
            r, resolve_problem=_resolve, resolve_committed=committed, first_sign=first
        ).outcome

    assert outcome(_saved(rec)) == "KEPT"
    assert outcome(_saved(rec, verdict="pruned")) == "PRUNED"
    assert outcome(_saved(rec, sleep_id="s2-n1", mask_digest="m2")) == "PENDING"  # (a)/(c)
    assert outcome(_saved(rec, mask_digest="m-other")) == "PENDING"  # (b) other digest
    commits.add(("s2-n1", "m2"))  # (d) restore wrote the missing commit row
    assert outcome(_saved(rec, sleep_id="s2-n1", mask_digest="m2")) == "KEPT"
    broken = _with(_saved(rec, sleep_id="s9", mask_digest="m9"), layer="Layer 9")
    assert outcome(broken) == "REFUSED"  # label checks run before the sleep clause


def test_r13_return_targets_must_be_admitted_born_units():
    born = _good(i=0)
    derived = _with(_good(i=1), origin="derived")
    no_region = {k: v for k, v in _good(i=2).items() if k != "region"}
    records = [born, derived, no_region]
    G.require_labeled_targets([born["unit_id"]], records, resolve_problem=_resolve)
    for bad in (derived["unit_id"], no_region["unit_id"], 987654):
        with pytest.raises(G.UnitLabelRefused):
            G.require_labeled_targets([bad], records, resolve_problem=_resolve)


def test_r13_target_ids_are_non_bool_integers_never_aliases():
    import numpy as np

    recs = [_with(_good(i=0), unit_id=1), _with(_good(i=1), unit_id=0)]
    for ok in (1, 0, np.int64(1), np.int32(0)):  # numpy ints from a synapse array stay accepted
        G.require_labeled_targets([ok], recs, resolve_problem=_resolve)
    for alias in (
        True,
        False,
        1.0,
        0.0,
        np.float64(1.0),
        "1",
        None,
        -1,
    ):  # == would alias these to unit 1 / 0
        with pytest.raises(G.UnitLabelRefused):
            G.require_labeled_targets([alias], recs, resolve_problem=_resolve)


def test_r13_negative_target_id_is_refused_by_the_type_branch_not_as_unknown():
    # without `tid < 0`, -1 still refuses but as "unknown target unit": the reason pins WHICH branch fired
    recs = [_with(_good(i=0), unit_id=0)]
    with pytest.raises(G.UnitLabelRefused) as exc:
        G.require_labeled_targets([-1], recs, resolve_problem=_resolve)
    assert exc.value.reason == "target id is not a non-negative integer"


# R10: each check is load-bearing -- neuter ONE at a time; only its own red stops firing.
_DEFECTS = {
    "unit_id": lambda g: {k: v for k, v in g.items() if k != "unit_id"},
    "batch_id": lambda g: _with(g, batch_id=-1),
    # x_um NaN: a defect ONLY the position check can see (an out-of-range ap is ALSO refused by the
    # region check's map read -- two checks on one defect, so it cannot isolate either one)
    "position": lambda g: _with(
        g, position=g["position"][:2] + (float("nan"),) + g["position"][3:]
    ),
    "layer": lambda g: _with(g, layer="White matter"),
    "sign": lambda g: _with(g, sign=0),
    "region": lambda g: _with(g, region="insula"),
    "problem_id": lambda g: _with(g, problem_id="unknown-problem"),
    "origin": lambda g: _with(g, origin="derived"),
}


def test_r10_every_check_is_load_bearing(monkeypatch):
    assert set(_DEFECTS) == set(G.FIELD_ORDER) == set(G._CHECKS)
    good = _good()
    for neutered in G.FIELD_ORDER:
        with monkeypatch.context() as m:
            m.setitem(G._CHECKS, neutered, lambda rec, ctx: None)
            for field, make in _DEFECTS.items():
                v = G.check_admission(make(good), resolve_problem=_resolve)
                if field == neutered:
                    assert v.admitted, f"neutering {neutered} must silence its own red"
                else:
                    assert not v.admitted and v.field == field, (neutered, field, v)


# ---- coverage: the two scans, the census, the debt tables (Codex b3 #8) --------------------------


def _tree(tmp_path, body):
    pkg = tmp_path / "fakepkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "mod.py").write_text(body, encoding="utf-8")
    return tmp_path


def test_r11_creation_scan_reds_an_ungated_site_and_a_stale_debt_row(tmp_path):
    root = _tree(
        tmp_path,
        (
            "from snn_data_lake.growth.placement import place\n"
            "def grow():\n    return place(3)\n"
            "def grow_gated(rec):\n    check_admission(rec, resolve_problem=None)\n"
            "    return place(3)\n"
        ),
    )
    found = G.label_creation_findings(src_root=root, known={})
    assert found == ["fakepkg/mod.py::grow"]
    stale = G.stale_known(
        src_root=root,
        known_creation={"fakepkg/mod.py::gone": ("G0", "x")},
        known_rows={},
        out_of_scope={},
    )
    assert stale == ["fakepkg/mod.py::gone"]


def test_r14b_a_same_root_rescan_sees_growth_added_since_the_last_scan(tmp_path):
    # Codex REQUEST 1 finding 1 (MAJOR): the census memoized solely by root path, so a same-root
    # re-scan after a source edit returned the FIRST scan's findings -- a warm census in one process
    # hid newly added ungated growth. Both caches are now re-read at every public boundary.
    root = _tree(
        tmp_path,
        (
            "import numpy as np\n"
            "class Grower:\n"
            "    def grow(self, new):\n        self.W = np.vstack([self.W, new])\n"
        ),
    )
    admit = "fakepkg/mod.py::Grower.grow"
    assert G.row_growth_findings(src_root=root, known={}, out_of_scope={}, segments={}) == [admit]
    (root / "fakepkg" / "second.py").write_text(
        "import numpy as np\n"
        "class Later:\n"
        "    def grow(self, new):\n        self.W = np.vstack([self.W, new])\n",
        encoding="utf-8",
    )
    (root / "fakepkg" / "site.py").write_text(
        "from snn_data_lake.growth.placement import place\ndef grow():\n    return place(3)\n",
        encoding="utf-8",
    )
    rows = G.row_growth_findings(src_root=root, known={}, out_of_scope={}, segments={})
    assert admit in rows and "fakepkg/second.py::Later.grow" in rows, rows
    assert "fakepkg/site.py::grow" in G.label_creation_findings(src_root=root, known={})


def test_r14_row_growth_scan_sees_what_the_creation_scan_cannot(tmp_path):
    root = _tree(
        tmp_path,
        (
            "import numpy as np\n"
            "class Grower:\n"
            "    def grow(self, new):\n        self.W = np.vstack([self.W, new])\n"
            "    def grow_ticketed(self, new, ticket):\n"
            "        ticket.consume([1])\n        self.W = np.vstack([self.W, new])\n"
        ),
    )
    admit = "fakepkg/mod.py::Grower.grow_ticketed"
    inventory = {"fx": ("fakepkg/mod.py", "fixture rows", "G7", (), admit)}
    assert G.row_growth_findings(src_root=root, known={}, out_of_scope={}, segments=inventory) == [
        "fakepkg/mod.py::Grower.grow"
    ]
    assert G.label_creation_findings(src_root=root, known={}) == []


_ADMIT = "fakepkg/mod.py::Seg.admit_rows"


@pytest.mark.parametrize(
    "body, inventory, admitted",
    [
        ("        ticket.consume(rows)\n        self.W = np.vstack([self.W, rows])\n", True, True),
        # uncensused
        (
            "        ticket.consume(rows)\n        self.W = np.vstack([self.W, rows])\n",
            False,
            False,
        ),
        # unreachable
        (
            "        if False:\n            ticket.consume(rows)\n"
            "        self.W = np.vstack([self.W, rows])\n",
            True,
            False,
        ),
        # unrelated
        (
            "        self.cache.consume(rows)\n        self.W = np.vstack([self.W, rows])\n",
            True,
            False,
        ),
        # after append
        ("        self.W = np.vstack([self.W, rows])\n        ticket.consume(rows)\n", True, False),
    ],
)
def test_cp3_5_a_ticket_counts_only_structurally_at_a_declared_admit_point(
    tmp_path, body, inventory, admitted
):
    head = "import numpy as np\nclass Seg:\n    def admit_rows(self, rows, ticket):\n"
    root = _tree(tmp_path, head + body)
    segs = {"fx": ("fakepkg/mod.py", "fixture rows", "G7", (), _ADMIT)} if inventory else {}
    found = G.row_growth_findings(src_root=root, known={}, out_of_scope={}, segments=segs)
    # an unspent ticket now also reds the declared point itself (a '[admit point ...]' suffix)
    assert {f.split(" [")[0] for f in found} == (set() if admitted else {_ADMIT})


@pytest.mark.parametrize(
    "stmt",
    [
        "self.W = np.append(self.W, new, axis=0)",
        "self.W = np.resize(self.W, (self.W.shape[0] + 1, self.W.shape[1]))",
        "self.W = np.pad(self.W, ((0, 1), (0, 0)))",
        "self.W[len(self.W):] = new",
        "self.W[self.W.shape[0]:] = new",
    ],
)
def test_cp3_4_every_registered_row_growth_form_is_seen(tmp_path, stmt):
    head = "import numpy as np\nclass Grower:\n    def grow(self, new):\n"
    root = _tree(tmp_path, head + f"        {stmt}\n")
    found = G.row_growth_findings(src_root=root, known={}, out_of_scope={})
    assert found == ["fakepkg/mod.py::Grower.grow"]


def test_cp3_4_flattening_append_and_column_concat_are_not_row_growth(tmp_path):
    root = _tree(
        tmp_path,
        (
            "import numpy as np\nclass Grower:\n    def f(self, new):\n"
            "        self.W = np.append(self.W, new)\n"
            "        self.S = np.concatenate([self.S, new], axis=1)\n"
        ),
    )
    assert G.row_growth_findings(src_root=root, known={}, out_of_scope={}) == []


def test_cp3_4_place_labeled_callers_are_label_creation_sites(tmp_path):
    root = _tree(
        tmp_path,
        (
            "from snn_data_lake.growth.placement import place_labeled\n"
            "def grow(ids):\n"
            "    return place_labeled(1, unit_ids=ids, region='cortex', site=(0, 0),\n"
            "        problem_id='p', batch_id=2, seed=1, centre_um=(0, 0), density_per_mm2=1.0)\n"
        ),
    )
    assert G.label_creation_findings(src_root=root, known={}) == ["fakepkg/mod.py::grow"]


def test_real_tree_scans_are_clean_against_their_debt_tables():
    assert G.label_creation_findings() == []
    assert G.row_growth_findings() == []
    assert G.stale_known() == []
    # the debt tables are not empty today, so the sentence is printed FALSE
    assert G.KNOWN_UNGATED_CREATION_SITES and G.KNOWN_UNGATED_ROW_GROWTH
    assert "is FALSE" in G.report()


def test_out_of_scope_row_growth_is_a_printed_scope_rule_not_debt():
    # ruling 23: P4d's grown unit = the engine's growth path; readout/reasoning row growth is OUT of
    # scope by a PRINTED rule, listed for the owner -- never silently dropped, never counted as debt
    oos = G.OUT_OF_SCOPE_ROW_GROWTH
    assert set(oos) == {
        "snn_data_lake/readout/growable_readout.py::GrowableReadout._grow_one",
        "snn_data_lake/readout/laminar_local_readout.py::LaminarLocalReadout.grow",
        "snn_data_lake/reasoning/developmental_loop.py::GrowableSTDP.maybe_recruit",
    }
    assert not set(oos) & set(G.KNOWN_UNGATED_ROW_GROWTH)
    text = G.report()
    assert "out of scope" in text and all(k in text for k in oos)
    # a stale scope row reds like a stale debt row
    stale = G.stale_known(known_creation={}, known_rows={}, out_of_scope={"x.py::gone": "r"})
    assert stale == ["x.py::gone"]


class _FixtureSegment:
    """A grown segment with ONE ticket-checked append point (the shape G3's admit_rows takes)."""

    def __init__(self):
        self.rows = []

    def admit_rows(self, rows, ticket):
        ticket.consume(rows)
        self.rows.extend(rows)


def test_r15_append_needs_a_ticket_and_the_census_reds_a_bypass():
    recs = [_good(i=i) for i in range(3)]
    seg = _FixtureSegment()
    ticket = G.require_admissible(recs, resolve_problem=_resolve, known_unit=lambda u: False)
    with pytest.raises(G.UnitLabelRefused):
        seg.admit_rows(recs + [_good(i=4)], ticket)  # a row the ticket does not cover
    seg.admit_rows(recs, ticket)
    with pytest.raises(G.UnitLabelRefused, match="consumed"):
        seg.admit_rows(recs, ticket)
    journal = {r["unit_id"]: dict(r) for r in recs}
    assert G.census({"fx": seg.rows}, journal, open_debt=set())["fx"]["red"] is False
    seg.rows.append({"place_depth": 0.4, "place_layer": "Layer 4", "place_sign": 1})  # the bypass
    out = G.census({"fx": seg.rows}, journal, open_debt=set())["fx"]
    assert out["legacy"] == 1 and out["red"] is True
    assert G.census({"fx": seg.rows}, journal, open_debt={"fx"})["fx"]["red"] is False


def test_bare_main_fires_its_constructed_reds_and_assert_is_clean(capsys):
    assert G.main([]) == 0
    out = capsys.readouterr().out
    assert "constructed reds fired: 10 of 10" in out and "is FALSE" in out
    assert G.main(["--assert"]) == 0


def test_r12_census_counts_todays_refit_as_legacy_and_reds_without_its_debt_row():
    from pathlib import Path

    from snn_data_lake.growth.neurogenesis_engine import (
        NeurogenesisSelfImprove,
        SyntheticPerceptStream,
    )
    from snn_data_lake.substrate.home import throwaway_home

    live = Path.home() / ".snn_data_lake"
    before = live.stat().st_mtime if live.exists() else None
    with throwaway_home():
        eng = NeurogenesisSelfImprove(device="cpu", seed=13014)
        SyntheticPerceptStream(seed=13014).drive(eng, 200)
        rows = G.engine_rows(eng)
    n = len(rows)
    assert n > 0, "the positive control grew nothing -- the census claim would be about nothing"
    out = G.census({"neurogenesis_engine": rows}, journal={}, open_debt={"neurogenesis_engine"})
    assert out["neurogenesis_engine"]["legacy"] == n and out["neurogenesis_engine"]["born"] == 0
    assert out["neurogenesis_engine"]["red"] is False
    census = G.census({"neurogenesis_engine": rows}, journal={}, open_debt=set())
    assert census["neurogenesis_engine"]["red"]
    assert (live.stat().st_mtime if live.exists() else None) == before


@pytest.mark.parametrize("field", ["unit_id", "batch_id"])
@pytest.mark.parametrize("bad", ["7", True, -1, 1.0, None])
def test_ids_are_non_negative_ints_only(field, bad):
    assert _refused_field(_with(_good(), **{field: bad})) == field


def test_a_json_replayed_list_position_is_not_a_tampered_label():
    born = _good(i=0)
    replayed = _with(born, position=list(born["position"]))  # JSON turns the tuple into a list
    assert G.classify_restored([replayed], {born["unit_id"]: dict(born)}) == ["born"]
    ticket = G.require_admissible([born], resolve_problem=_resolve, known_unit=lambda x: False)
    ticket.consume([replayed])  # a list-position row IS the validated record
    moved = _with(born, position=[x + 1.0 for x in born["position"]])  # a real change still reds
    with pytest.raises(G.UnitLabelRefused, match="tampered"):
        G.classify_restored([moved], {born["unit_id"]: dict(born)})


def test_ticket_consumed_is_read_only_and_flips_on_consume():
    u = _good(i=0)
    ticket = G.require_admissible([u], resolve_problem=_resolve, known_unit=lambda x: False)
    assert ticket.consumed is False
    ticket.consume([u])
    assert ticket.consumed is True
    with pytest.raises(AttributeError):
        ticket.consumed = False


def test_a_ticket_freezes_list_positions_at_mint():
    # Codex G3 panel: a list-position record mutated AFTER validation must not be admitted
    rec = _with(_good(), position=list(_good()["position"]))
    ticket = G.require_admissible([rec], resolve_problem=_resolve, known_unit=lambda u: False)
    rec["position"][2] = float(
        "inf"
    )  # the live label is now invalid; the frozen evidence must not follow
    with pytest.raises(G.UnitLabelRefused, match="differs from the validated"):
        ticket.consume([rec])
    assert ticket.consumed is False


@pytest.mark.parametrize(
    "change", [{"unit_id": True}, {"batch_id": 0.0}, {"unit_id": 1.0}, {"sign": True}]
)
def test_a_ticket_refuses_an_equal_but_differently_typed_identity(change):
    # Codex G3 panel: 1 == True and 0 == 0.0 in Python, so == let a forbidden id type through
    rec = _with(_good(), unit_id=1, batch_id=0)
    ticket = G.require_admissible([rec], resolve_problem=_resolve, known_unit=lambda u: False)
    with pytest.raises(G.UnitLabelRefused):
        ticket.consume([_with(rec, **change)])
    assert ticket.consumed is False


@pytest.mark.parametrize("field", ["layer", "region", "position", "problem_id"])
def test_a_journaled_born_unit_missing_a_label_field_is_refused_not_legacy(field):
    # Codex G3 panel: deleting a label field from a journaled id used to read 'legacy', and an open
    # debt kept the census green -- tampering laundered through the debt table
    born = _good()
    journal = {born["unit_id"]: dict(born)}
    stripped = {k: v for k, v in born.items() if k != field}
    with pytest.raises(G.UnitLabelRefused, match="tampered"):
        G.classify_restored([stripped], journal)
    with pytest.raises(G.UnitLabelRefused, match="tampered"):
        G.census({"fx": [stripped]}, journal, open_debt={"fx"})


def test_a_declared_admit_point_must_exist_and_spend_the_ticket(tmp_path):
    # Codex G3 panel: a declared admit point was never checked -- a missing function, or an
    # admit_rows with its ticket.consume deleted, produced no finding
    head = "class Seg:\n    def admit_rows(self, rows, ticket):\n"
    good = head + "        ticket.consume(rows)\n        return rows\n"
    bad = head + "        return rows\n"
    seg = {"fx": ("fakepkg/mod.py", "fixture rows", "G7", (), _ADMIT)}
    ghost = {
        "fx": ("fakepkg/mod.py", "fixture rows", "G7", (), "fakepkg/missing.py::no_such_function")
    }

    def found(body, segments):
        root = _tree(tmp_path, body)
        return G.row_growth_findings(src_root=root, known={}, out_of_scope={}, segments=segments)

    assert found(good, seg) == []  # control: a real point that spends its ticket
    assert found(bad, seg) == [f"{_ADMIT} [admit point does not spend its ticket]"]
    assert found(good, ghost) == ["fakepkg/missing.py::no_such_function [admit point not found]"]


def test_a_ticket_refuses_a_bool_inside_a_position():
    # G3 refutation: 1.0 == True inside a tuple, so a top-level-only type check let it through
    rec = _good()
    pos = list(rec["position"])
    pos[2] = 1.0
    rec = _with(rec, position=tuple(pos))
    ticket = G.require_admissible([rec], resolve_problem=_resolve, known_unit=lambda u: False)
    pos[2] = True
    bad = _with(rec, position=tuple(pos))
    assert not G.check_admission(bad, resolve_problem=_resolve).admitted  # the gate refuses it
    with pytest.raises(G.UnitLabelRefused, match="differs from the validated"):
        ticket.consume([bad])
    ticket.consume(
        [_with(rec, position=list(rec["position"]))]
    )  # a JSON list of the same numbers is fine


@pytest.mark.parametrize("bad_id", [True, 1.0, "1", -1])
def test_a_restored_row_with_a_non_int_unit_id_is_refused_never_derived(bad_id):
    # G3 refutation: unit_id True == journaled id 1 but is not an _ident, so it skipped the journal
    # lookup and read 'derived' (an open debt row then keeps the census green)
    born = _good()
    journal = {1: _with(born, unit_id=1)}
    row = _with(born, unit_id=bad_id)
    with pytest.raises(G.UnitLabelRefused, match="unit_id"):
        G.classify_restored([row], journal)
    with pytest.raises(G.UnitLabelRefused, match="unit_id"):
        G.census({"fx": [row]}, journal, open_debt={"fx"})


_AFTER = "        ticket.consume(rows)\n        return rows\n"


@pytest.mark.parametrize(
    "body",
    [
        "        seg = list(rows)\n        return seg\n" + _AFTER,
        "        if rows:\n            return list(rows)\n" + _AFTER,
        "        for r in rows:\n            return [r]\n" + _AFTER,
        "        try:\n            return list(rows)\n        finally:\n            pass\n"
        + _AFTER,
    ],
)
def test_an_admit_point_with_a_return_before_the_ticket_is_spent_is_refused(tmp_path, body):
    # Codex fold re-panel: the segment was returned (units admitted) with ticket.consume unreachable
    # or skipped on a conditional early return, and the gate read it as spending the ticket
    head = "class Seg:\n    def admit_rows(self, rows, ticket):\n"
    seg = {"fx": ("fakepkg/mod.py", "fixture rows", "G7", (), _ADMIT)}
    root = _tree(tmp_path, head + body)
    found = G.row_growth_findings(src_root=root, known={}, out_of_scope={}, segments=seg)
    assert found == [f"{_ADMIT} [admit point does not spend its ticket]"]


def test_an_admit_point_may_raise_before_the_ticket_is_spent(tmp_path):
    # control: a refusal path is not a successful return
    head = "class Seg:\n    def admit_rows(self, rows, ticket):\n"
    body = "        if not rows:\n            raise ValueError('empty')\n" + _AFTER
    seg = {"fx": ("fakepkg/mod.py", "fixture rows", "G7", (), _ADMIT)}
    root = _tree(tmp_path, head + body)
    found = G.row_growth_findings(src_root=root, known={}, out_of_scope={}, segments=seg)
    assert found == []
