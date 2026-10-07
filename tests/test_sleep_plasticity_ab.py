"""R3 / T104 P1 -- the sleep-plasticity arms' instrument (instruments/sleep_plasticity_ab.py).

Fast tests are pure: the statistic wrapper (rulings 9 / 9a, the enumerated sign-flip rate), K11's
table reachability, K12's seed rule, K14's decoy red, the census, and one constructed red per
record-level check on a synthetic record that is green before the plant. One integration test
drives two real arm-seeds (ON, OFF; 2 sleep ticks; a red-block seed) through the two-process path
in a scratch home.
"""

import copy
import os
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
        m.run_arm(str(live / sub) if sub else str(live), "F1", "ON", m.RED_SEEDS[6], budget=1)
    assert m.lake_snapshot(str(live)) == before
    assert sorted(p for p, _d, _f in os.walk(live)) == dirs_before


def test_guard_out_refuses_a_junction_alias(tmp_path, monkeypatch):
    """An alias (a directory junction pointing INTO the decoy live lake) resolves and is refused."""
    import subprocess

    from snn_data_lake.substrate.home import LiveHomeRefused

    live = _decoy(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(live)], capture_output=True, text=True
    )
    if made.returncode != 0:
        pytest.skip(f"no junction on this host: {made.stderr.strip()}")
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


A_INTENTS, B_INTENTS = ["GREET", "GOAL"], ["MEMORY", "CONSCIOUS", "ARM"]
HELD_KEYS = m.expected_heldout_keys(A_INTENTS, B_INTENTS)


def _record(arm="ON", seed=16000, family="F1"):
    """A synthetic record that every record-level check reads green."""
    return {
        "family": family, "arm": arm, "seed": seed, "budget": 2, "d0": "d0", "d1": "d1",
        "weights_d1": "w", "weights_post": "w", "taught": ["t1"], "ring_keys": ["t1"],
        "a": A_INTENTS, "b": B_INTENTS, "heldout_keys": list(HELD_KEYS), "drawn": ["t1"],
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


def test_two_real_arm_seeds_restore_exactly(tmp_path):
    """ON and OFF on a red-block seed, 2 sleep ticks: every record check green, restore exact, the
    ledger k per tick, OFF's weights untouched by its bout, and the live lake untouched."""
    out = str(tmp_path)
    seed = m.RED_SEEDS[5]
    recs = [m.run_arm(out, "F1", arm, seed, budget=2) for arm in ("ON", "OFF")]
    fails = m.check_records("F1", recs)
    assert not any(fails.values()), fails
    on, off = recs
    assert on["inproc"] == on["restored"] and off["inproc"] == off["restored"]
    assert on["tick"]["presentations"] == off["tick"]["presentations"] == [m.K, m.K]
    assert sum(on["tick"]["deposits"]) > 0 and sum(off["tick"]["deposits"]) == 0
    assert on["d0"] == off["d0"] and on["d1"] == off["d1"]
    assert off["weights_d1"] == off["weights_post"] and on["weights_d1"] != on["weights_post"]
    assert all(r["live_lake_untouched"] for r in recs)
    assert os.listdir(os.path.join(out, "homes")) == []  # every throwaway home removed


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


def test_an_unreadable_file_fails_the_live_lake_check_closed(tmp_path, monkeypatch):
    """A file whose stat fails used to be skipped, so a change under it was invisible. It is now
    recorded as None, and the check cannot read untouched while any file is None."""
    decoy = tmp_path / "decoy_live"
    decoy.mkdir()
    (decoy / "brain_lake.bin").write_bytes(b"abc")
    (decoy / "latch.json").write_text("x")
    before = m.lake_snapshot(str(decoy))
    assert m.live_lake_untouched(str(decoy), before) == (True, [])
    real = os.stat

    def flaky(path, *a, **k):
        if str(path).endswith("latch.json"):
            raise PermissionError(path)
        return real(path, *a, **k)

    monkeypatch.setattr(os, "stat", flaky)
    unreadable = m.lake_snapshot(str(decoy))
    assert unreadable["latch.json"] is None  # recorded, never skipped
    ok, changed = m.live_lake_untouched(str(decoy), unreadable)  # None on both reads
    assert not ok and "latch.json" in changed
