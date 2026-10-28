"""R3 / T104 P1 -- the sleep-plasticity arms' instrument (instruments/sleep_plasticity_ab.py).

Fast tests are pure: the statistic wrapper (rulings 9 / 9a, the enumerated sign-flip rate), K11's
table reachability, K12's seed rule, K14's decoy red, the census, and one constructed red per
record-level check on a synthetic record that is green before the plant. One integration test
drives two real arm-seeds (ON, OFF; 2 sleep ticks; a red-block seed) through the two-process path
in a scratch home.
"""

import copy
import os
import sys
from fractions import Fraction

import pytest

from snn_data_lake.instruments import sleep_plasticity_ab as m

DEV = m.SEED_SETS["dev"]


def test_offline_selfcheck():
    m.main()


def test_verdict_superiority_degenerate_is_not_met_and_fpr_is_enumerated():
    v = m.verdict("c", [Fraction(1, 20)] * 8, DEV)
    assert v.degenerate and not v.met
    assert v.fpr == 8 / 256  # the eight 7+/1- flips give t = 3.0 under the actual rule (Codex 21)
    assert v.ref_fpr == 0.0  # the incumbent's shortcut, labelled reference


def test_verdict_ni_degenerate_reads_by_common_value():
    assert m.verdict("ni", [m.EPS_A] * 8, DEV, kind="ni").met  # a lossless arm is WITHIN BOUND (9a)
    assert not m.verdict("ni", [0] * 8, DEV, kind="ni").met  # one A item lost on every seed -> RED
    assert not m.verdict("ni", [-m.EPS_A] * 8, DEV, kind="ni").met


def test_verdict_nondegenerate_matches_paired_contrast():
    d = [0.2, 0.3, 0.25, 0.4, 0.15, 0.35, 0.3, 0.2]
    v = m.verdict("c", d, DEV)
    ref = m.contrast("c", d)
    assert (v.met, v.t, v.fpr) == (ref.met, ref.t, ref.fpr)
    assert not m.verdict("noise", [0.1, -0.1, 0.12, -0.11, 0.09, -0.1, 0.1, -0.12], DEV).met


@pytest.mark.parametrize("diffs,seeds", [([0.1] * 7, DEV[:7]), ([0.1] * 8, (300,) + DEV[1:])])
def test_k12_refuses_partial_lists_and_foreign_seeds(diffs, seeds):
    with pytest.raises(ValueError):
        m.verdict("red", diffs, seeds)


def test_cp2_5_repeated_seeds_refused():
    # seven copies of one seed + one other: eight entries, two independent seeds -- refused
    with pytest.raises(ValueError):
        m.verdict("red", [0.1] * 7 + [-0.1], (16000,) * 7 + (16001,))


def test_cp2_5_a_family_reads_only_its_own_set_or_dev():
    assert m.check_set("F1", "F1") == m.SEED_SETS["F1"]
    assert m.check_set("F1", "dev") == m.SEED_SETS["dev"]
    with pytest.raises(ValueError):
        m.check_set("F1", "F2")


def test_cp2_7_a_missing_twin_halts_before_any_contrast():
    seeds = m.SEED_SETS["F1"]
    arms = m.registered_arms("F1", "F1")
    assert "ON-TWIN" in arms and "A-ONLY-TWIN" in arms
    full = [_record(a, s) for a in arms for s in seeds]
    assert m.completeness("F1", full, seeds, "F1") == []
    partial = [r for r in full if r["arm"] != "ON-TWIN"]
    res = m.decide_family("F1", partial, seeds, set_name="F1")
    assert res["row"].startswith("HALT") and res["contrasts"] == {}
    assert ("missing", ("ON-TWIN", seeds[0])) in res["checks"]["COMPLETE"]
    assert ("duplicate", ("ON", seeds[0])) in m.completeness("F1", full + full[:1], seeds, "F1")


def test_cp2_11_load_reads_only_json_records_and_refuses_malformed(tmp_path):
    import json

    d = tmp_path / "records"
    d.mkdir()
    rec = dict(_record("ON", 16000), stores=None, red=None, wall={}, eval_s=0.0, budget=40)
    (d / "F1_ON_16000.json").write_text(json.dumps(rec), encoding="ascii")
    (d / "F1_ON_16000.stderr.txt").write_text("--- bout\nwarning: something\n", encoding="utf-8")
    recs = m._load(str(tmp_path), "F1", m.SEED_SETS["dev"])
    assert [r["arm"] for r in recs] == ["ON"]  # the stderr log is never parsed as a record
    (d / "F1_OFF_16000.json").write_text(json.dumps({"family": "F1"}), encoding="ascii")
    with pytest.raises(ValueError):
        m._load(str(tmp_path), "F1", m.SEED_SETS["dev"])


def test_k11_every_row_reached_and_the_inventory_is_complete():
    ok, misses = m.table_reach_ok()
    assert ok, misses


def test_k11_red_a_row_without_input(monkeypatch):
    short = dict(m.TABLE_REACH)
    short["F3"] = short["F3"][:-1]
    monkeypatch.setattr(m, "TABLE_REACH", short)
    assert not m.table_reach_ok()[0]


def test_k14_decoy_red_fires(tmp_path):
    f = tmp_path / "brain_lake.bin"
    f.write_bytes(b"x")
    before = m.lake_snapshot(str(tmp_path))
    assert m.live_lake_untouched(str(tmp_path), before)[0]
    f.write_bytes(b"xy")
    assert not m.live_lake_untouched(str(tmp_path), before)[0]


def _decoy(tmp_path, monkeypatch):
    """A DECOY live lake holding one file; the module's LIVE_LAKE points at it."""
    live = tmp_path / "decoy_live"
    live.mkdir()
    (live / "brain_lake.bin").write_bytes(b"x")
    monkeypatch.setattr(m, "LIVE_LAKE", str(live))
    return live


def test_scratch_home_refusals(tmp_path, monkeypatch):
    from snn_data_lake.substrate.home import LiveHomeRefused

    live = _decoy(tmp_path, monkeypatch)
    monkeypatch.setenv("NORD_PERSIST", "1")
    (live / "inside").mkdir()
    for bad in (str(live), str(live / "inside"), str(tmp_path)):  # equal, contained, contains
        monkeypatch.setenv("NORD_HOME", bad)
        with pytest.raises(LiveHomeRefused):
            m.require_scratch_home()
    monkeypatch.delenv("NORD_HOME")
    with pytest.raises(LiveHomeRefused):
        m.require_scratch_home()


@pytest.mark.parametrize("sub", ["", "records", "new/deeper"])
def test_run_arm_refuses_the_live_lake_before_any_mutation(tmp_path, monkeypatch, sub):
    """Codex cp2 #1's red: --out at, or anywhere under, the (decoy) live lake is refused before any
    mkdir, write or delete -- the decoy reads byte-for-byte unchanged afterwards."""
    from snn_data_lake.substrate.home import LiveHomeRefused

    live = _decoy(tmp_path, monkeypatch)
    before = m.lake_snapshot(str(live))
    dirs_before = sorted(p for p, _d, _f in os.walk(live))
    with pytest.raises(LiveHomeRefused):
        m.run_arm(str(live / sub) if sub else str(live), "F5", "STAGED", m.RED_SEEDS[6])
    assert m.lake_snapshot(str(live)) == before
    assert sorted(p for p, _d, _f in os.walk(live)) == dirs_before


def test_guard_out_refuses_a_junction_alias(tmp_path, monkeypatch):
    """An alias (a directory junction pointing INTO the decoy live lake) resolves and is refused."""
    import subprocess

    from snn_data_lake.substrate.home import LiveHomeRefused

    live = _decoy(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    if sys.platform == "win32":  # a directory junction, the alias the operator's host makes
        made = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(live)], capture_output=True, text=True
        )
        if made.returncode != 0:
            pytest.skip(f"no junction on this host: {made.stderr.strip()}")
    else:  # no `cmd` off Windows (FileNotFoundError, cloud audit 2026-10-27): a directory symlink
        os.symlink(live, alias, target_is_directory=True)
    with pytest.raises(LiveHomeRefused):
        m.guard_out(str(alias / "out"))


def test_seed_sets_inside_the_block_and_disjoint():
    seen = set()
    for name, seeds in m.SEED_SETS.items():
        assert len(seeds) == 8 and all(m.SEED_BLOCK[0] <= s <= m.SEED_BLOCK[1] for s in seeds), name
        assert not seen & set(seeds), name
        seen |= set(seeds)
    assert not seen & set(m.RED_SEEDS)


def test_census_counts_the_tests_read():
    read = [
        m.verdict("C1", [0.1, 0.2] * 4, DEV),
        m.verdict("C1-NI", [m.EPS_A] * 8, DEV, kind="ni"),
        m.verdict("C-SENS", [0.1, 0.2] * 4, DEV, kind="ctl"),
    ]
    lines = m.census(read)
    assert lines[-1].startswith("tests read in all: 3")
    assert "C1-NI=MET (degenerate)" in lines[2]


def _split(seed):
    """The seed's frozen A / B split (fold lens M1: a record's a / b must be THIS one)."""
    a, b, *_ = m.split_and_order(seed)
    return a, b


A_INTENTS, B_INTENTS = _split(16000)
HELD_KEYS = m.expected_heldout_keys(A_INTENTS, B_INTENTS)


def _record(arm="ON", seed=16000, family="F1"):
    """A synthetic record that every record-level check reads green."""
    a_intents, b_intents = _split(seed)
    return {
        "family": family, "arm": arm, "seed": seed, "budget": 2, "d0": "d0", "d1": "d1",
        "weights_d1": "w", "weights_post": "w", "taught": ["t1"], "ring_keys": ["t1"],
        "a": a_intents, "b": b_intents,
        "heldout_keys": m.expected_heldout_keys(a_intents, b_intents),
        "drawn": ["t1"],
        "tick": {"presentations": [m.K, m.K], "turnover": [None, None], "cacc": [None, None]},
        "att_calls": [[1, 3], [2, 3]], "k8a": [], "stored_any": False,
        "chrono_before": {"nights": 0, "wake_ticks": 0, "arm_ticks": 0, "step": 0},
        "chrono_after": {"nights": 1, "wake_ticks": 0, "arm_ticks": 0, "step": 0},
        "inproc": [1] * 10 + [0] * 10, "restored": [1] * 10 + [0] * 10, "n_a": 8,
        "eval_before": "e", "eval_after": "e",
        "margin": 0.5,
        "live_lake_untouched": True,
    }  # fmt: skip


PLANTS = {
    "K1": lambda r: r.update(d0="other"),
    "K2": lambda r: r["tick"].update(presentations=[m.K - 1, m.K]),
    "K3": lambda r: r.update(drawn=[r["heldout_keys"][0]]),
    "K4": lambda r: r.update(eval_after="moved"),
    "K5": lambda r: r.update(restored=[0] * 20),
    "K6": lambda r: r.update(weights_post="moved"),
    "K8a": lambda r: r.update(k8a=[{"before": {}, "after": {}}]),
    "K8b": lambda r: r["chrono_after"].update(nights=2),
    "K8b-step": lambda r: r["chrono_after"].update(step=1),  # cp2 #10: _step asserted, not printed
    "K14": lambda r: r.update(live_lake_untouched=False),
}


@pytest.mark.parametrize("check", sorted(PLANTS))
def test_record_check_reds(check):
    arm = "OFF" if check == "K6" else "ON"
    base = [_record(arm), _record("ON-TWIN") if check != "K6" else _record("ON")]
    assert not any(m.check_records("F1", base).values()), m.check_records("F1", base)
    planted = copy.deepcopy(base)
    PLANTS[check](planted[0])
    assert m.check_records("F1", planted)[check.split("-")[0]], check


def test_k13_red_unequal_attractor_draws():
    a, b = _record("AT-ON", family="F3"), _record("AT-OFF", family="F3")
    assert not m.check_records("F3", [a, b])["K13"]
    b["att_calls"] = [[1, 2], [2, 2]]
    assert m.check_records("F3", [a, b])["K13"]


def test_f3_ladder_rule_and_the_unfrozen_refusal():
    below, at = [34] * 40, [35] * 40  # per-tick correct counts of 36
    assert m.ladder_qualifies([below] * 8)
    assert not m.ladder_qualifies([below] * 7 + [at])  # one seed at 35/36 fails the rung
    assert not m.ladder_qualifies([below] * 7)  # a partial seed list never qualifies
    assert m.F3_W_LADDER == (1, 2, 3, 6, 12)
    assert m.F3_STORES == 1  # the first rung that qualified on dev (2026-10-06 amendment)


def test_f1_arm_seeds_refuse_retired_production_before_writes(tmp_path):
    """Owner1039, 2026-10-13: F1 is archival; production refuses before creating output."""
    for arm in ("ON", "OFF"):
        with pytest.raises(RuntimeError, match="retired live sleep family"):
            m.run_arm(str(tmp_path), "F1", arm, m.RED_SEEDS[5], budget=2)
    assert list(tmp_path.iterdir()) == [tmp_path / "nord_home"]


def test_cp2_13_k5_does_not_apply_to_f3():
    a, b = _record("AT-ON", family="F3"), _record("AT-OFF", family="F3")
    a["restored"] = [0, 0]  # an unrelated restore discrepancy: P-5 says K5 is not an F3 check
    assert not m.check_records("F3", [a, b])["K5"]


def test_cp2_12_margin_is_correct_minus_best_other_and_claims_read_primary():
    class Ixn:
        n_in, responses, neurons = 8, ["x", "y", "z"], [3.0, 1.0, 2.0]

        def encode(self, s):
            return s

        def _drive(self, n, code):
            return n

    items = [("hello world", "x"), ("hello world", "z")]  # margins 3-2 and 2-3
    assert m.margin(Ixn(), items) == (1.0 + -1.0) / 2
    recs = [_record("ON", s) for s in (16000, 16001)]
    c = type("C", (), {"met": True})()
    lines = m.claims_report("F1", {"C1": c, "C1-NI": c, "C1-NI-B": c}, recs, "MET")
    assert "met: 3 of 3" in lines[0] and "STANDS" in lines[0] and "1/2" in lines[1]
    n = type("N", (), {"met": False})()  # F1 dev reading: C1 NOT MET, C1-NI MET -> no claim
    assert "NOT MADE" in m.claims_report("F1", {"C1": n, "C1-NI": c}, recs, "NULL")[0]
    m.PRIMARY["F1"] = ("C1", "C9")  # the declaration is CONSUMED: changing it changes the report
    try:
        assert "met: 1 of 2" in m.claims_report("F1", {"C1": c}, recs, "MET")[0]
    finally:
        m.PRIMARY["F1"] = ("C1", "C1-NI", "C1-NI-B")


# ---- Codex lens-r3-amend: endpoint shapes are checked (KE)
def _met_all():
    return type("C", (), {"met": True})()


@pytest.mark.parametrize(
    "row, stands",
    [
        ("REFUSED (blind harness: C-SENS NOT MET)", False),  # C1 and C1-NI MET, C-SENS NOT MET
        ("NULL: sleep synaptic plasticity not shown", False),
        ("INTERFERENCE RED (C1 MET, C1-NI NOT MET)", False),  # also the row of any added NI guard
        ("HALT (instrument defect, no verdict)", False),
        ("MET: old-experience replay carries the gain (C2 MET)", True),
        ("MET; no benefit over recency-only replay demonstrated (C2 NOT MET)", True),
    ],
)
def test_claims_follow_the_final_guarded_row(row, stands):
    recs = [_record("ON", s) for s in (16000, 16001)]
    c = _met_all()
    line = m.claims_report("F1", {"C1": c, "C1-NI": c}, recs, row)[0]
    assert ("STANDS" in line) is stands and ("NOT MADE" in line) is not stands, line


def test_claims_f4_partial_credit_stands_and_red_does_not():
    recs = [_record("MED", s, "F4") for s in (16130, 16131)]
    c = _met_all()
    ok = "EXT credited; no sleep-duration benefit over massed wake demonstrated (C9 NOT MET)"
    assert "STANDS" in m.claims_report("F4", {}, recs, ok)[0]
    red = "INTERFERENCE RED at MED (C7 MET, its NI NOT MET)"
    assert "NOT MADE" in m.claims_report("F4", {"C7": c}, recs, red)[0]


def test_a_dev_set_never_claims_even_on_a_met_row():
    """Fold lens M2: the dev seeds are EXPOSED; a MET row there is a reading, never a claim."""
    recs = [_record("ON", s) for s in (16000, 16001)]
    c = _met_all()
    met = "MET: old-experience replay carries the gain (C2 MET)"
    args = ("F1", {"C1": c, "C1-NI": c, "C1-NI-B": c}, recs, met)
    assert "STANDS" in m.claims_report(*args)[0]
    line = m.claims_report(*args, set_name="dev")[0]
    assert "NOT MADE" in line and "STANDS" not in line and "dev" in line


def test_a_halted_incomplete_set_reports_without_crashing():
    """Found reproducing the fold lens: `decide` on an incomplete set (HALT, no contrasts, and
    possibly no records of the family at all) died in claims_report on recs[0]."""
    halt = "HALT (instrument defect, no verdict): the record set is incomplete"
    for recs in ([], [_record("MED", 16130, "F4")]):
        assert "NOT MADE" in m.claims_report("F4", {}, recs, halt)[0]


KE_PLANTS = {
    "short": lambda rs: rs[0].update(restored=[1] * 6 + [0] * 6, inproc=[1] * 12, n_a=6),
    "empty_b": lambda rs: rs[0].update(restored=[1] * 8, inproc=[1] * 8, n_a=8),
    "empty": lambda rs: rs[0].update(restored=[], inproc=[], n_a=0),
    "non_binary": lambda rs: rs[0].update(restored=[2] * 20),
    "bool": lambda rs: rs[0].update(restored=[True] * 20),
    "inproc_len": lambda rs: rs[0].update(inproc=[1] * 19),
    "identity": lambda rs: rs[1].update(heldout_keys=["h2"]),
    # Codex round 2: identities must BE the 20 frozen A/B items, not merely agree across arms
    "empty_ids": lambda rs: [r.update(heldout_keys=[]) for r in rs],
    "same_ids": lambda rs: [r.update(heldout_keys=["h"] * 20) for r in rs],
    "wrong_count": lambda rs: [r.update(heldout_keys=HELD_KEYS[:19]) for r in rs],
    "duplicate": lambda rs: [r.update(heldout_keys=HELD_KEYS[:19] + HELD_KEYS[:1]) for r in rs],
    "reordered": lambda rs: [r.update(heldout_keys=HELD_KEYS[::-1]) for r in rs],
    "unknown_intent": lambda rs: [r.update(a=["NOPE", "GOAL"]) for r in rs],
    # fold lens M1: a / b / keys can agree with each other and still not be the seed's split
    "empty_split": lambda rs: [r.update(a=[], b=[], heldout_keys=[]) for r in rs],
    "repeated_definition": lambda rs: [
        r.update(
            a=["GREET", "GREET"],
            b=["GREET"] * 3,
            heldout_keys=m.expected_heldout_keys(["GREET", "GREET"], ["GREET"] * 3),
        )
        for r in rs
    ],
    "foreign_split": lambda rs: [
        r.update(
            a=_split(16001)[0],
            b=_split(16001)[1],
            heldout_keys=m.expected_heldout_keys(*_split(16001)),
        )
        for r in rs
    ],
    # fold lens m1: malformed identities HALT under KE, they never raise
    "none_ids": lambda rs: [r.update(heldout_keys=None) for r in rs],
    "unhashable_ids": lambda rs: [r.update(heldout_keys=[{}] * 20) for r in rs],
    "float_n_a": lambda rs: rs[0].update(n_a=8.0),  # Codex round 2 MINOR: crashed, now HALTs
}


@pytest.mark.parametrize("plant", sorted(KE_PLANTS))
def test_ke_endpoint_shape_reds(plant):
    base = [_record("ON"), _record("OFF")]
    assert not m.check_records("F1", base)["KE"]
    planted = copy.deepcopy(base)
    KE_PLANTS[plant](planted)
    assert m.check_records("F1", planted)["KE"], plant


def test_malformed_endpoint_halts_before_any_contrast_and_never_crashes():
    seeds = m.SEED_SETS["F1"]
    full = [_record(a, s) for a in m.registered_arms("F1", "F1") for s in seeds]
    bad = copy.deepcopy(full)
    on = next(r for r in bad if r["arm"] == "ON" and r["seed"] == seeds[0])
    on.update(restored=[1] * 8, inproc=[1] * 8, n_a=8)  # an EMPTY B endpoint
    res = m.decide_family("F1", bad, seeds, set_name="F1")
    assert res["row"].startswith("HALT") and res["contrasts"] == {}
    assert res["checks"]["KE"]
    # Codex's false-B-NI shape: ON with 20 items against an OFF of 6 must not reach a contrast
    short = copy.deepcopy(full)
    off = next(r for r in short if r["arm"] == "OFF" and r["seed"] == seeds[0])
    off.update(restored=[1] * 6, inproc=[1] * 6, n_a=6)
    res = m.decide_family("F1", short, seeds, set_name="F1")
    assert res["row"].startswith("HALT") and res["contrasts"] == {}


# ---- the E_B non-inferiority guard (dated amendment; Codex round 2 #1): a B loss blocks the claim
def _f1_set(overrides):
    seeds = m.SEED_SETS["F1"]
    recs = []
    for a in m.registered_arms("F1", "F1"):
        for i, sd in enumerate(seeds):
            r = _record(a, sd)
            r["restored"] = r["inproc"] = list(
                overrides.get((a.replace("-TWIN", ""), i), r["restored"])
            )
            recs.append(r)
    return seeds, recs


def _vec(a_ones, b_ones):
    return [1] * a_ones + [0] * (8 - a_ones) + [1] * b_ones + [0] * (12 - b_ones)


def test_eb_guard_blocks_an_f1_claim_when_session_b_loses_one_item():
    ov = {}
    for i in range(8):
        ov["OFF", i] = _vec(4, 10)
        ov["ON", i] = _vec(7 + i % 2, 9)  # A +3/+4 items, B -1 item on every seed: d'_B = 0
        ov["ORACLE", i] = _vec(7 + i % 2, 10)  # C-SENS MET
    seeds, recs = _f1_set(ov)
    res = m.decide_family("F1", recs, seeds, set_name="F1")
    c = res["contrasts"]
    assert c["C1"].met and c["C1-NI"].met and c["C-SENS"].met
    assert "C1-NI-B" in c and not c["C1-NI-B"].met and c["C1-NI-B"].degenerate
    assert res["row"].startswith("INTERFERENCE RED") and "session B" in res["row"], res["row"]
    assert "NOT MADE" in m.claims_report("F1", c, recs, res["row"])[0]


def test_eb_guard_is_a_conjunct_not_a_substitute_and_a_lossless_arm_passes():
    ov = {}
    for i in range(8):
        ov["OFF", i] = _vec(4, 10)
        ov["ON", i] = _vec(7 + i % 2, 10)  # B unchanged: d'_B = +1/12 constant -> MET (ruling 9a)
        ov["ORACLE", i] = _vec(7 + i % 2, 10)
    seeds, recs = _f1_set(ov)
    res = m.decide_family("F1", recs, seeds, set_name="F1")
    assert res["contrasts"]["C1-NI-B"].met and res["row"].startswith("MET"), res["row"]


def test_eb_rows_precede_the_c_if_row_and_f4_keeps_med_but_blocks_ext():
    # F1: C-IF NOT MET must not hide a B-bar failure (the B row is read first)
    assert m.table_f1(True, 1, 0, 1, 1, 1, 0).startswith("INTERFERENCE RED")
    assert m.table_f1(True, 1, 1, 1, 0, 1, 1).startswith("INTERFERENCE RED")  # the A bar still reds
    # F4: a failed C7-NI-B blocks MED credit; a failed C8-NI-B keeps MED, blocks EXT
    assert m.table_f4(True, 1, 1, 1, 1, 1, 0, 0).startswith("INTERFERENCE RED at MED")
    kept = m.table_f4(True, 1, 1, 1, 1, 1, 1, 0)
    assert kept.startswith("MED credited") and "EXT" in kept and not kept.startswith("EXT credited")
    assert m.claim_stands(kept)
    assert m.table_f4(True, 1, 1, 1, 1, 1, 1, 1).startswith("EXT credited: a")
    # F2: the B bar reds a structural claim too
    assert m.table_f2(True, True, True, 1, 1, True, 0).startswith("INTERFERENCE RED")


def test_no_instrument_function_reclaims_a_record_outside_a_scratch_home():
    """Batch-2 precheck: the retention gate flagged run_evaluate (K5 deleted the lake in a
    child that only knows NORD_HOME). The delete is now in run_arm, which makes the throwaway
    home it deletes in; the gate lists it as a scratch deletion; run_evaluate no longer reclaims."""
    from snn_data_lake.gates import immortal_retention as gate

    bad, scratch = gate.reclaimers(with_scratch=True)
    mine = "instruments/sleep_plasticity_ab.py"
    assert (mine, "run_evaluate") not in bad and (mine, "run_evaluate") not in scratch
    assert (mine, "run_arm") not in bad
    assert (mine, "run_arm") in scratch  # K5's delete: counted and listed, never silent


# ------------------------------------------------------------------ F5 (the amendment 2026-10-08)
def _f5_block(arm):
    stages = m.FAMILIES["F5"][arm].stages
    st = [
        {"name": nm, "budget": n, "presentations": n, "starved": 0, "refused": 0, "deposits": 0}
        for nm, n in stages
    ]
    drives = sum(n for nm, n in stages if nm in ("nrem", "flat"))
    return {
        "stages": st, "pause": "budget", "runner_total": m.F5_P, "log_total": m.F5_P,
        "n_interact": drives, "interact_in_ring": True, "ring_pre": "r", "ring_post": "r",
        "lake_ok": True, "lake_changed": [],
    }  # fmt: skip


def _f5_record(arm, seed=16140, a_ok=4, b_ok=6):
    r = _record(arm, seed, family="F5")
    r["budget"] = 0
    r["tick"] = {"presentations": [], "turnover": [], "cacc": []}
    r["inproc"] = r["restored"] = [1] * a_ok + [0] * (8 - a_ok) + [1] * b_ok + [0] * (12 - b_ok)
    r["f5"] = _f5_block(arm) if m.FAMILIES["F5"][arm].stages else None
    return r


def test_f5_design_constants_and_adapter():
    from snn_data_lake.instruments import staged_sleep_bout as g6

    assert m.g6_run_bout is g6.run_bout  # the one runner, used as landed (P2)
    assert all(sum(n for _, n in v) == m.F5_P == 320 for v in m.F5_STAGES.values())
    assert [nm for nm, _ in m.F5_STAGES["SHUFFLED"]] == ["nrem", "nrem", "rem"]  # rem last
    assert m.F5_REM_FIRST[0][0] == "rem"  # the K2-F5 red order
    assert set(m.FAMILIES["F5"]) == {"STAGED", "FLAT", "SHUFFLED", "NONE", "ORACLE"}
    assert m.SEED_SETS["F5"] == tuple(range(16140, 16148))
    assert m.BUDGETS["F5"] == 0  # no incumbent sleep tick in an F5 bout


F5_PLANTS = {
    "K1": lambda r: r.update(d0="other"),
    "K3": lambda r: r.update(drawn=[r["heldout_keys"][0]]),
    "K8a": lambda r: r.update(k8a=[{"before": {}, "after": {}}]),
    "K2F5-starved": lambda r: r["f5"]["stages"][0].update(starved=1),
    "K2F5-total": lambda r: r["f5"].update(runner_total=256),
    "K2F5-count": lambda r: r["f5"].update(log_total=319),
    "K2F5-pause": lambda r: r["f5"].update(pause="plateau"),
    "K2F5-refused": lambda r: r["f5"]["stages"][0].update(refused=1),
    "K10a-member": lambda r: r["f5"].update(interact_in_ring=False),
    "K10a-count": lambda r: r["f5"].update(n_interact=r["f5"]["n_interact"] + 1),
    "K10b": lambda r: r["f5"].update(ring_post="grown"),
    "K10c": lambda r: r["f5"].update(lake_ok=False),
    "K10d": lambda r: r.update(drawn=[r["heldout_keys"][3]]),
}


@pytest.mark.parametrize("plant", sorted(F5_PLANTS))
def test_f5_record_check_reds(plant):
    base = [_f5_record(a) for a in ("STAGED", "FLAT")]
    if plant == "K1":
        base[1]["d0"] = base[0]["d0"]
    assert not any(m.check_records("F5", base).values()), m.check_records("F5", base)
    planted = copy.deepcopy(base)
    F5_PLANTS[plant](planted[0])
    check = plant.split("-")[0]
    got = m.check_records("F5", planted)
    assert got[check], (plant, got)


def test_f5_missing_block_is_k2f5_red_and_no_stage_arm_needs_none():
    r = _f5_record("STAGED")
    r["f5"] = None
    assert m.check_records("F5", [r])["K2F5"]
    assert not any(m.check_records("F5", [_f5_record("NONE"), _f5_record("ORACLE")]).values())


def _f5_set(seeds, per_arm):
    """Records for every F5 arm x seed; per_arm[arm](i) -> (a_ok, b_ok) for the i-th seed."""
    return [
        _f5_record(arm, s, *per_arm[arm](i))
        for i, s in enumerate(seeds)
        for arm in m.FAMILIES["F5"]
    ]


F5_SEEDS = m.SEED_SETS["F5"]
_GOOD = {  # varied (non-degenerate) per-seed endpoints
    "NONE": lambda i: (3, 5),
    "ORACLE": lambda i: (3 + 1 + i % 3, 5),
    "FLAT": lambda i: (4, 6),
    "SHUFFLED": lambda i: (4 - i % 2, 6),
    "STAGED": lambda i: (4 + i % 2, 6 + i % 3),
}


def _f5_row(per_arm, mutate=None):
    recs = _f5_set(F5_SEEDS, per_arm)
    if mutate:
        mutate(recs)
    return m.decide_family("F5", recs, F5_SEEDS, set_name="F5")


def test_f5_decision_flow_rows():
    res = _f5_row(_GOOD)
    assert res["row"].startswith(m.F5_MET), res["row"]
    assert {"C-SENS5", "C10", "C10-NI", "C10-NI-B", "C11", "FLAT-NONE"} <= set(res["contrasts"])
    assert res["contrasts"]["FLAT-NONE"].kind == "rep"  # F5-3a: reported beside C10, never a claim
    lines = m.census(res["contrasts"].values())
    assert any(x.startswith("reported only read: 1") for x in lines)
    # NULL: STAGED equal to FLAT everywhere -> C10 degenerate zero -> NOT MET
    assert _f5_row(dict(_GOOD, STAGED=_GOOD["FLAT"]))["row"].startswith("NULL")
    # REFUSED: ORACLE == NONE -> C-SENS5 degenerate zero
    assert _f5_row(dict(_GOOD, ORACLE=_GOOD["NONE"]))["row"].startswith("REFUSED")
    # INTERFERENCE RED on A: STAGED gains on B but loses A items against FLAT on every seed
    row = _f5_row(dict(_GOOD, STAGED=lambda i: (1, 11 + i % 2)))["row"]
    assert row.startswith("INTERFERENCE RED (C10"), row
    # INTERFERENCE RED on B: STAGED gains on A but loses B items
    row = _f5_row(dict(_GOOD, STAGED=lambda i: (8, 3 + i % 2)))["row"]
    assert row.startswith("INTERFERENCE RED on session B"), row

    def break_restore(recs):
        recs[0]["restored"] = [1 - x for x in recs[0]["restored"]]

    res = _f5_row(_GOOD, break_restore)  # a restore red is NOT CLAIMED, not a halt (table row 3)
    assert res["row"].startswith("NOT CLAIMED") and res["checks"]["K5"]

    def break_digest(recs):
        recs[0]["d0"] = "other"

    assert _f5_row(_GOOD, break_digest)["row"].startswith("HALT")  # any OTHER red check halts


def test_f5_dev_gate_blind_cases():
    dev = m.SEED_SETS["dev"]

    def recs(flat, none):
        out = []
        for s in dev:
            out.append(_f5_record("FLAT", s, *flat))
            out.append(_f5_record("NONE", s, *none))
        return out

    ceiling = m.f5_dev_gate(recs((8, 12), (3, 5)), dev)
    assert len(ceiling) == 1 and "ceiling" in ceiling[0]
    assert any("does not move" in x for x in m.f5_dev_gate(recs((4, 6), (4, 6)), dev))
    assert any("does not move" in x for x in m.f5_dev_gate(recs((3, 5), (4, 6)), dev))
    assert m.f5_dev_gate(recs((4, 6), (3, 5)), dev) == []  # the control: a moving, sub-ceiling FLAT


def test_f5_real_staged_and_flat_arm_seeds(tmp_path):
    """STAGED and FLAT on a red-block seed through the two-process path: every record check green
    (K2-F5, K10a-d, K1, K3, K5, K8), 320 presentations on both counts, 256 vs 320 deposit-bearing
    drives (F5-3), live lake untouched."""
    out = str(tmp_path)
    seed = m.RED_SEEDS[7]
    recs = [m.run_arm(out, "F5", arm, seed) for arm in ("STAGED", "FLAT")]
    fails = m.check_records("F5", recs)
    assert not any(fails.values()), fails
    st, fl = recs
    assert st["f5"]["runner_total"] == fl["f5"]["runner_total"] == m.F5_P
    assert st["f5"]["log_total"] == fl["f5"]["log_total"] == m.F5_P
    # deposit-bearing drives are 256 vs 320 (F5-3); the runner counts every `_deposit` CALL and an
    # error-driven interact can deposit more than once, so the call count is reported, not matched
    assert (st["f5"]["n_interact"], fl["f5"]["n_interact"]) == (256, 320)
    assert sum(x["deposits"] for x in st["f5"]["stages"]) >= 256
    assert sum(x["deposits"] for x in fl["f5"]["stages"]) >= 320
    print("F5 deposit calls", [sum(x["deposits"] for x in r["f5"]["stages"]) for r in recs])
    assert st["d0"] == fl["d0"] and st["d1"] == fl["d1"]
    assert all(r["live_lake_untouched"] for r in recs)
    assert os.listdir(os.path.join(out, "homes")) == []


@pytest.mark.parametrize(
    "name,arm",
    [
        ("K10a", "STAGED"),
        ("K10b", "STAGED"),
        ("K10c", "STAGED"),
        ("K10d", "STAGED"),
        ("K2F5", "SHUFFLED"),
        ("K8a-f5", "STAGED"),
    ],
)
def test_f5_brain_level_reds_fire(tmp_path, name, arm):
    """Each constructed red, built through the real child, is caught by its check on a record whose
    unplanted twin is green (the unplanted run is the control)."""
    seed = m.RED_SEEDS[8]
    out = str(tmp_path)
    base = m.run_arm(out, "F5", arm, seed)
    assert not any(m.check_records("F5", [base]).values())
    # 2026-10-13 owner1037-1039/2106: forbidden synthetic teaching/ring injection
    # can refuse at the common R2 boundary before a record exists. K10a/b/d are unchanged.
    refusal = {
        "K10a": "persistent learner effect requires the common R2 boundary",
        "K10b": "R2 sleep_bout ring/authority mutation refused",
        "K10d": "malformed R2 ring identities",
    }
    if name in refusal:
        with pytest.raises(RuntimeError, match=refusal[name]):
            m.run_arm(out, "F5", arm, seed, red=name)
        return
    red = m.run_arm(out, "F5", arm, seed, red=name)
    check = name.split("-")[0]
    assert m.check_records("F5", [red])[check], (name, red["f5"])


def test_k18_planted_c10_gaps_are_read_through_the_whole_decision_path():
    """F5-8: C10 reads MET and NULL on the real path (records -> _d -> verdict -> table)."""
    flat = _GOOD["FLAT"]
    plants = {  # name -> (STAGED endpoints per seed, expected table-row prefix)
        "gap +2": (lambda i: (4 + 1, 6 + 1 + i % 2), m.F5_MET),
        "gap 0": (lambda i: (4 + (1 if i % 2 else -1), 6), "NULL"),
        "gap -2": (lambda i: (4 - 1, 6 - 1 - i % 2), "NULL"),
        "gap +2, costs 3 on A": (lambda i: (1, 6 + 5 + i % 2), "INTERFERENCE RED (C10"),
        "gap +1.5, costs 3 on B": (lambda i: (8, 3 + i % 2), "INTERFERENCE RED on session B"),
    }
    for name, (staged, prefix) in plants.items():
        row = _f5_row(dict(_GOOD, FLAT=flat, STAGED=staged))["row"]
        assert row.startswith(prefix), (name, row)


def test_c10_detectable_gap_is_the_smallest_planted_gap_that_reads_met():
    noise = [Fraction(x, 20) for x in (1, -1, 2, -2, 0, 1, -1, 0)]  # mean 0, sd ~1.2 items
    g = m.c10_detectable_gap(noise)
    assert g is not None and g >= 1
    shifted = [x + Fraction(g, 20) for x in noise]
    assert m._met(tuple(float(x) for x in shifted), "sup")
    below = [x + Fraction(g - 1, 20) for x in noise]
    assert g == 1 or not m._met(tuple(float(x) for x in below), "sup")
    # a mean offset in the dev diffs is removed first: the control reads NOISE, not the dev signal
    assert m.c10_detectable_gap([x + Fraction(1, 2) for x in noise]) == g
    # zero-variance noise: every constant reads NOT MET (ruling 9), so nothing is readable
    assert m.c10_detectable_gap([Fraction(1, 20)] * 8) is None
