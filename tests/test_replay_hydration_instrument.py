"""R2 campaign instrument (instruments/replay_hydration.py): every registered red as a constructed
failing input the instrument must flag, the scorer's controls, the licence refusal, and one real
dev-seed smoke. Dev seeds 15000-15007 only; no confirmatory or reserve seed is ever run here."""

import copy
import json
import math
import sys
from pathlib import Path

import pytest

from snn_data_lake.instruments import replay_hydration as rh
from snn_data_lake.substrate import home
from snn_data_lake.substrate.learner_events import KIND_SPECS, scan_learner_events

REDS = (
    "O_wm",
    "O_hyd",
    "O_rng",
    "O_w",
    "O_stack",
    "O_rw",
    "O_sleepskip",
    "O_cursor",
    "O_bout",
    "CHRONO",
    "MOTOR",
)


@pytest.mark.parametrize("returncode", [37, 0])
@pytest.mark.parametrize("reused", [True, False])
def test_r4_child_without_output_never_inherits_a_result(tmp_path, monkeypatch, returncode, reused):
    out = tmp_path / "r2-15000-twin_K1.json"
    if reused:
        out.write_text(json.dumps({"role": "twin_K1", "old_result": True}), encoding="utf-8")
    existed_at_launch = []

    def child(args, **kwargs):
        existed_at_launch.append(Path(args[6]).exists())
        return rh.subprocess.CompletedProcess(args, returncode, stdout="", stderr="")

    monkeypatch.setattr(rh.subprocess, "run", child)
    arms = rh.run_seed(15000, tmp_path, roles=("twin_K1",))
    assert arms["twin_K1"] == {
        "role": "twin_K1",
        "error": f"exit {returncode}" if returncode else "no output",
    }
    assert existed_at_launch == [False]
    assert not out.exists()


@pytest.mark.parametrize("outcome", ["success", "nonzero", "unparsable"])
def test_r4_child_result_requires_success_and_fresh_parseable_output(
    tmp_path, monkeypatch, outcome
):
    out = tmp_path / "r2-15000-twin_K1.json"
    out.write_text(json.dumps({"role": "twin_K1", "old_result": True}), encoding="utf-8")
    fresh = {"role": "twin_K1", "current_result": True}
    existed_at_launch = []

    def child(args, **kwargs):
        target = Path(args[6])
        existed_at_launch.append(target.exists())
        target.write_text("{" if outcome == "unparsable" else json.dumps(fresh), encoding="utf-8")
        return rh.subprocess.CompletedProcess(
            args, 37 if outcome == "nonzero" else 0, stdout="", stderr="child evidence"
        )

    monkeypatch.setattr(rh.subprocess, "run", child)
    arms = rh.run_seed(15000, tmp_path, roles=("twin_K1",))
    if outcome == "success":
        assert arms["twin_K1"] == fresh
    else:
        assert arms["twin_K1"] == {
            "role": "twin_K1",
            "error": "exit 37" if outcome == "nonzero" else "unparsable output",
            "stderr": "child evidence",
        }
    assert existed_at_launch == [False]
    assert (tmp_path / "r2-15000-twin_K1.stderr.txt").read_text(encoding="utf-8") == (
        "child evidence"
    )


@pytest.mark.parametrize("kind", ["BOUTDIGEST", "BOUTPROGRESS"])
def test_bout_plant_requires_its_bound_private_home(tmp_path, monkeypatch, kind):
    private, token = rh._create_private_home(tmp_path)
    monkeypatch.setattr(rh, "_throwaway_token", token)
    monkeypatch.setenv("NORD_HOME", str(private))
    assert home.save_episode(
        {
            "kind": "sleep_bout",
            "event_id": "1-1",
            "weights_digest": "a" * 64,
            "presentations_spent": "2",
        }
    )
    path = Path(home.knowledge_path())
    original = path.read_bytes()
    monkeypatch.setattr(rh, "_throwaway_home", None, raising=False)
    with pytest.raises(ValueError, match="private home"):
        rh._plant_bout(kind)
    assert path.read_bytes() == original
    monkeypatch.setattr(rh, "_throwaway_home", private.resolve())
    # A changed NORD_HOME must not grant access to another lake, even another scratch lake.
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("NORD_HOME", str(outside))
    outside_path = Path(home.knowledge_path())
    outside_path.write_bytes(original)
    with pytest.raises(ValueError, match="private home"):
        rh._plant_bout(kind)
    assert outside_path.read_bytes() == original and path.read_bytes() == original
    monkeypatch.setenv("NORD_HOME", str(private))
    assert rh._plant_bout(kind) == "1-1"
    assert path.read_bytes() != original


def _walk():
    return {"class": "L", "fields": {"w": [[0.5, 0.25], [0.125, 0.0625]], "n": 3}}


def _record(**over):
    rec = {
        "role": "x",
        "stack_present": True,
        "seed": 15000,
        "fault": None,
        "construct_motor": 0,
        "construct_chrono": {"step": 0, "nights": 0, "observed": None, "live_n": None},
        "before": {
            "walk": _walk(),
            "ring": [["GREET", "hello"], ["GOAL", "set a goal"], ["MEMORY", "recall"]],
            "ring_ids": ["1-1", "1-2", "1-3"],
            "rng": "state0",
            "cursor": [1, 3],
            "replay_counts": {},
            "rewirer": {"cycles": 1, "rng": "r0"},
        },
        "rows": [
            {
                "weights": [[0.5 + i, 0.25], [0.125, 0.0625]],
                "drawn": [],
                "drawn_ids": [],
                "rng": f"s{i}",
                "replay_counts": {},
                "rewirer": {"cycles": 2 + i, "rng": f"r{i}"},
            }
            for i in range(rh.S_SLEEP)
        ],
        "replay_windows": [{"before": {"step": 3}, "after": {"step": 3}, "motor": 0}],
    }
    rec.update(over)
    return rec


def _arms():
    base = _record()
    arms = {
        role: copy.deepcopy(base)
        for role, _ in rh.R2_ROLES
        if role.startswith(("res_", "twin_", "O_", "CHRONO", "MOTOR"))
    }
    for role, _ in rh.R2_ROLES:
        if role.startswith("pre_"):
            arms[role] = {"role": role}  # a pre process writes only its construction facts
    off = copy.deepcopy(base)
    off["rows"][-1]["weights"][0][0] += 1.0  # structural-off moves the observable
    arms["twin_off"] = off
    for kind in rh.PLANTED:
        for side in ("on", "off"):
            arms[f"{kind}_{side}"] = copy.deepcopy(base)
    arms["O_cursor"]["cursor_probe"] = {"binding_valid": True, "redo_steps": 1}
    # the cut night's onset is step 8 (the cut sits J_NIGHT - 1 = 2 ticks later, at step 10); the
    # circadian flush runs first in `_enter_sleep`, so the parent pre_K3 flushed step 8 and the
    # restore adopted that record and constructed its clock at it
    arms["twin_K3"]["onset_step"] = 8
    arms["pre_K3"]["flushed_step"] = 8
    arms["res_K3"]["construct_circadian_restored"] = {"step": 8}
    arms["res_K3"]["construct_chrono"]["step"] = 8
    return arms


def _base_arms():
    base = {role: _record() for role, _ in rh.BASE_ROLES if not role.startswith("pre_")}
    base["res_K2"]["rows"][0]["weights"][0][0] += 1.0  # the incumbent's I_K2 must differ
    return base


def _differ(rec):
    rec["rows"][0]["weights"][0][0] = math.nextafter(rec["rows"][0]["weights"][0][0], math.inf)


def _flag(c, red):
    return c["reds"][red]


@pytest.mark.parametrize("red", REDS)
def test_each_red_is_flagged_when_it_differs_and_not_when_vacuously_equal(red):
    arms = _arms()
    assert not _flag(rh.seed_comparisons(arms), red), "an equal red arm must read NOT MET"
    if red == "CHRONO":
        arms["CHRONO"]["construct_chrono"]["step"] = 1
    elif red == "MOTOR":
        arms["MOTOR"]["construct_motor"] = 1
    else:
        _differ(arms[red])
    assert _flag(rh.seed_comparisons(arms), red)


def test_comparator_flags_one_ulp_and_one_swapped_ring_entry_and_missing_record():
    rec = _record()
    assert rh.records_equal(rec, copy.deepcopy(rec))
    assert rh.comparator_controls(rec)
    ulp = copy.deepcopy(rec)
    _differ(ulp)
    assert not rh.records_equal(rec, ulp)
    swapped = copy.deepcopy(rec)
    r = swapped["before"]["ring"]
    r[0], r[1] = r[1], r[0]
    assert not rh.records_equal(rec, swapped)
    assert not rh.records_equal(None, rec) and not rh.records_equal(rec, {})
    latched = copy.deepcopy(rec)
    latched["fault"] = {"phase": "reconcile", "cause": "x"}
    assert not rh.records_equal(rec, latched), "a latched restore is never equal to its twin"


def test_chrono_clean_flags_a_tick_or_a_motor_call():
    assert rh.chrono_clean(_record())
    dirty = _record()
    dirty["replay_windows"][0]["after"]["step"] = 4
    assert not rh.chrono_clean(dirty)
    assert not rh.chrono_clean(_record(construct_motor=1))


def _per_seed(**over):
    c = {
        "c1": {"K1": True},
        "c6": {"K3": True},
        "chrono": {"r": True},
        "reds": dict.fromkeys(REDS, True),
        "planted": {
            k: {"guard_on_ok": True, "guard_on_refused": True, "guard_off_flagged": True}
            for k in rh.PLANTED
        },
        "c4_structural_moves": True,
        "c4_comparator": True,
        "c2": {"I_K1_equal": True, "I_K2_differs": True},
        "arms_complete": True,
        "incomplete": [],
        "failwrite": {"ok": True, "reds": {}},
        "c8": {"a": True, "b": True, "c": True, "d": True},
    }
    c.update(over)
    return c


def test_verdict_invalid_seed_replacement_halt_and_must_differ_bars():
    s1, s2, s3, s4 = rh.DEV_SEEDS[:4]
    per = {s1: _per_seed(), s2: _per_seed()}
    per[s2]["c2"]["I_K2_differs"] = False
    v = rh.verdict(per, [s1, s2])
    assert v["invalid_seeds"] == [s2] and not v["halted"]
    assert v["R2-c3"] == "NOT MET" and v["verdict"] != "PASS", "an INVALID seed is never dropped"
    per = {s: _per_seed() for s in (s1, s2, s3, s4)}
    for s in (s2, s3, s4):
        per[s]["c2"]["I_K2_differs"] = False
    assert rh.verdict(per, [s1, s2, s3, s4])["halted"], "more than 2 INVALID seeds halts the arc"
    one = {s1: _per_seed()}
    one[s1]["reds"]["O_w"] = False
    assert rh.verdict(one, [s1])["R2-c3"] == "RED", "a red that fires on no seed is RED"
    one = {s1: _per_seed(c4_structural_moves=False)}
    assert rh.verdict(one, [s1])["R2-c4"] == "RED", "a vacuous must-differ control is NOT MET"
    one = {s1: _per_seed(c6={"K3": False})}
    assert rh.verdict(one, [s1])["R2-c6"] == "RED"


def test_codex14_memory_unit_gate_is_reported_unverified_never_pass():
    status = rh.codex14_status()
    assert status["memory_unit_unreconstructible"] == "UNVERIFIED"
    assert "UNVERIFIED" in json.dumps(rh.verdict({15000: _per_seed()}, [15000]))


def test_no_seed_outside_the_dev_block_runs_without_the_licence(monkeypatch, tmp_path):
    monkeypatch.delenv("R2_CONFIRMATORY_GO", raising=False)
    assert rh.DEV_SEEDS == tuple(range(15000, 15008))
    for seed in (15010, 15020, 15030, 15050, 15999):
        with pytest.raises(ValueError, match="licence"):
            rh.check_seeds([seed])
    rh.check_seeds(rh.DEV_SEEDS)
    with pytest.raises(ValueError, match="licence"):
        rh.run_seed(15010, tmp_path)


def test_base_head_roles_are_the_four_incumbent_arms_and_their_twins_only():
    names = {r for r, _ in rh.BASE_ROLES}
    assert names == {"pre_K1", "pre_K2", "twin_K1", "twin_K2", "res_K1", "res_K2"}


# ---- planted faults on a real lake: the reader refuses each; the neutered guard admits ----


def _ev(kind, seq, **kw):
    e = {
        "kind": kind,
        "event_id": f"1-{seq}",
        "stream": "chat",
        "session": "1",
        "seq": str(seq),
        "source": f"line {seq}",
        "target": "GREET",
        "provenance": "self",
        "encoder": "text_features n_in=52",
        "schema": "r2.v1",
        "valid": "1",
        "split": "train",
    }
    e.update(kw)
    return e


def _sleep_ev(seq, tick):
    e = _ev("sleep", seq, tick=str(tick))
    for name in ("source", "target", "provenance"):
        del e[name]
    e.update(
        sleep_id="1-n1",
        night="1",
        stage_index="0",
        stage_name="night",
        presentations_spent="0",
        drawn="none",
        pending="none",
        applied_mask_digest="none",
        rng_digest="a" * 64,
        weights_digest="b" * 64,
        rewirer_digest="c" * 64,
    )
    return e


def _scan():
    return scan_learner_events(tuple(KIND_SPECS), strict=True, return_receipt=True)


@pytest.fixture
def lake(monkeypatch):
    import snn_data_lake.substrate.learner_events as le

    saved = (le._refusal, le._identity_refusal, le.SLEEP_DIGESTS, le.SLEEP_PROGRESS)
    with home.throwaway_home():
        home.require_isolated_home()
        for i in range(1, 6):
            assert home.save_episode(_ev("teach" if i % 2 else "turn", i))
        assert home.save_episode(_sleep_ev(6, 1))
        yield le
    le._refusal, le._identity_refusal, le.SLEEP_DIGESTS, le.SLEEP_PROGRESS = saved


@pytest.mark.parametrize("kind", ["DUP", "REORDER", "HELDOUT", "MISSING"])
def test_planted_reader_faults_are_refused_and_the_neutered_guard_admits(lake, kind):
    _, clean, _, _ = _scan()
    assert not clean or True
    planted = rh._plant(kind)
    events, refusals, _, receipt = _scan()
    assert refusals.get(rh.PLANTED[kind], 0) >= 1, (kind, refusals)
    assert planted not in [e["event_id"] for e in events] or kind == "DUP"
    proof = json.loads(receipt)
    assert proof["complete"] is (kind == "HELDOUT"), "only a non-critical refusal keeps it complete"
    rh._neuter(kind)
    events, refusals, _, _ = _scan()
    assert refusals.get(rh.PLANTED[kind], 0) == 0, "the neutered guard must stop refusing"
    if kind in ("HELDOUT", "MISSING"):
        assert planted in [e["event_id"] for e in events], "the neutered arm admits it"


def test_planted_ctxread_and_sleep_digest_rows_are_eligible_so_the_brain_must_refuse(lake):
    ctx = rh._plant("CTXREAD")
    events, refusals, _, _ = _scan()
    assert ctx in [e["event_id"] for e in events] and not refusals
    assert any(e["kind"] == "ctxread" for e in events)


def test_ctxread_plant_follows_the_shared_sequence_including_p3_percepts(lake):
    import base64
    import struct

    percept = {
        "kind": "percept",
        "event_id": "1-7",
        "stream": "arm",
        "session": "1",
        "seq": "7",
        "schema": "r2.v1",
        "valid": "1",
        "split": "train",
        "x": base64.b64encode(struct.pack("<64d", *([0.25] * 64))).decode("ascii"),
        "category": "0",
        "n_protos": "1",
        "proto_boot": "1",
        "proto_flushed": "1",
        "pseq": "1",
        "provenance": "live",
        "encoder": "arm.percept gained width=64",
    }
    assert home.save_episode(percept)
    planted = rh._plant("CTXREAD")
    events, refusals, _, receipt = _scan()
    assert not refusals, refusals
    assert json.loads(receipt)["complete"]
    assert planted == "1-8"
    assert events[-1]["kind"] == "ctxread"


def test_real_k4_restart_equals_its_twin_at_the_precheckpoint_boundary(tmp_path):
    arms = rh.run_seed(
        15000,
        tmp_path,
        src=Path(rh.__file__).resolve().parents[2],
        roles=("pre_K4", "twin_K1", "twin_K4", "res_K4"),
    )
    assert all("error" not in a for a in arms.values()), arms
    assert rh.seed_comparisons(arms)["c6"]["K4"], {
        role: a.get("before", {}).get("cursor") for role, a in arms.items()
    }
    twin = arms["twin_K4"]
    assert twin["k4_checkpoint_ok"] is True, "the never-killed twin delegates the real flush"
    assert twin["before"]["cursor"] == [1, 25]
    assert twin["post_boundary_cursor"] == [1, 26], "the extra percept is after K4's cut"
    assert not rh.records_equal(arms["res_K4"], arms["twin_K1"])
    wrong = copy.deepcopy(twin)
    wrong["before"]["cursor"][1] += 1
    assert not rh.records_equal(arms["res_K4"], wrong), "cursor equality remains exact"


def test_k4_comparison_uses_its_own_twin_and_rejects_a_changed_record():
    arms = _arms()
    _differ(arms["twin_K1"])
    assert rh.seed_comparisons(arms)["c6"]["K4"]
    _differ(arms["twin_K4"])
    assert not rh.seed_comparisons(arms)["c6"]["K4"]


def test_planted_sleep_digest_row_is_eligible_with_a_wrong_digest(lake):
    row = rh._plant("SLEEPDIGEST")
    events, refusals, _, _ = _scan()
    assert row in [e["event_id"] for e in events] and not refusals
    assert [e for e in events if e["event_id"] == row][0]["weights_digest"] == "0" * 64


# ---- one real process set on a dev seed: K1 restart equals its twin, an omission red moves ------


def test_real_k1_restart_equals_twin_and_an_omitted_weight_is_flagged(tmp_path):
    src = str(Path(rh.__file__).resolve().parents[2])
    arms = rh.run_seed(
        15000,
        tmp_path,
        src=src,
        roles=("pre_K2", "twin_K2", "res_K2", "O_w"),
    )
    assert all("error" not in a for a in arms.values()), {
        r: a.get("stderr", "")[-800:] for r, a in arms.items() if "error" in a
    }
    assert rh.records_equal(arms["res_K2"], arms["twin_K2"])
    assert not rh.records_equal(arms["O_w"], arms["twin_K2"])
    assert rh.chrono_clean(arms["res_K2"])
    assert sys.version_info >= (3, 12)


def test_real_cursor_red_reaches_sleep_redo_with_a_valid_prefix_binding(tmp_path):
    arms = _run(tmp_path, "cursor", ("pre_K1", "twin_K1", "res_K1", "O_cursor"))
    assert rh.records_equal(arms["res_K1"], arms["twin_K1"]), "clean cursor control"
    red = arms["O_cursor"]
    cause = (red["fault"] or {}).get("cause", "")
    assert "prefix" not in cause, cause
    assert red["cursor_probe"]["binding_valid"]
    assert red["cursor_probe"]["redo_steps"] > 0, "must execute the sleep effect/comparison"
    assert red["cursor_probe"]["saved_cursor"] != red["cursor_probe"]["mutated_cursor"]
    assert not rh.records_equal(red, arms["twin_K1"])
    assert not cause or cause.endswith("R2 sleep redo digest/progress mismatch"), cause


def test_cursor_prefix_binding_refusal_never_counts_as_a_successful_red():
    arms = _arms()
    red = arms["O_cursor"]
    red["fault"] = {"cause": "R2 checkpoint canonical prefix binding missing"}
    assert not rh.seed_comparisons(arms)["reds"]["O_cursor"]
    red["fault"] = None
    red["before"]["cursor"] = [1, 1]
    red["cursor_probe"]["redo_steps"] = 0
    assert not rh.seed_comparisons(arms)["reds"]["O_cursor"]
    red["cursor_probe"]["redo_steps"] = 1
    assert rh.seed_comparisons(arms)["reds"]["O_cursor"]


@pytest.mark.parametrize("foreign", [OSError, ValueError])
def test_cursor_foreign_step_exception_is_invalid_not_a_red(monkeypatch, foreign):
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner as Learner
    from snn_data_lake.substrate.learner_events import prefix_digest

    events = [_ev("teach", 1), _sleep_ev(2, 1)]

    def broken(*args):
        # Even the intended message is foreign when raised INSIDE the step.
        raise foreign("R2 sleep redo digest/progress mismatch")

    def learner():
        obj = Learner(["GREET", "GOAL", "MEMORY"], 52, seed=15000)
        obj._cursor = (1, 2)
        return obj

    kw = {
        "receipt": json.dumps({"complete": True, "eligible_ids": ["1-1", "1-2"]}).encode(),
        "sleep_step": {"rewirer": None, "step": broken},
        "checkpoint_prefix_digest": prefix_digest(events, (1, 2)),
    }
    learner().hydrate(events, **kw)  # Honest cursor covers the broken step: no execution.
    monkeypatch.setattr(Learner, "hydrate", Learner.hydrate)
    probe = rh._patch_restore("O_cursor")
    with pytest.raises(foreign) as exc:
        learner().hydrate(events, **kw)
    arms = _arms()
    arms["O_cursor"].update(fault={"cause": f"{foreign.__name__}: {exc.value}"}, cursor_probe=probe)
    comparison = rh.seed_comparisons(arms, _base_arms())
    assert not comparison["reds"]["O_cursor"], "foreign execution failure is never a red"
    assert "O_cursor" in comparison["incomplete"] and not comparison["arms_complete"]
    assert probe["redo_attempts"] == 1 and probe["redo_steps"] == 0
    assert probe["foreign_exception"]


def test_cursor_completed_step_intended_mismatch_still_fires(monkeypatch):
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner as Learner
    from snn_data_lake.substrate.learner_events import prefix_digest

    events = [_ev("teach", 1), _sleep_ev(2, 1)]
    learner = Learner(["GREET", "GOAL", "MEMORY"], 52, seed=15000)
    learner._cursor = (1, 2)
    monkeypatch.setattr(Learner, "hydrate", Learner.hydrate)
    probe = rh._patch_restore("O_cursor")
    with pytest.raises(ValueError, match="^R2 sleep redo digest/progress mismatch$") as exc:
        learner.hydrate(
            events,
            receipt=json.dumps({"complete": True, "eligible_ids": ["1-1", "1-2"]}).encode(),
            sleep_step={"rewirer": None, "step": lambda *a: (None, {"_structural_report": {}})},
            checkpoint_prefix_digest=prefix_digest(events, (1, 2)),
        )
    arms = _arms()
    arms["O_cursor"].update(fault={"cause": f"ValueError: {exc.value}"}, cursor_probe=probe)
    assert rh.seed_comparisons(arms)["reds"]["O_cursor"]
    assert probe["redo_attempts"] == probe["redo_steps"] == 1
    assert probe["mismatch_verified"] and probe["foreign_exception"] is None


@pytest.mark.parametrize("populated", [False, True])
def test_process_refuses_caller_directory_before_instrument_io(tmp_path, monkeypatch, populated):
    private = tmp_path / "caller-owned"
    private.mkdir()
    sentinel = private / "brain_lake.bin"
    if populated:
        sentinel.write_bytes(b"durable caller lake: do not truncate")
    before = {p.name: p.read_bytes() for p in private.iterdir()}
    old_home = str(tmp_path / "previous-home")
    monkeypatch.setenv("NORD_HOME", old_home)

    def forbidden(*a, **kw):
        pytest.fail("unowned directory reached instrument IO")

    monkeypatch.setattr(rh, "_MotorCount", forbidden)
    monkeypatch.setattr(rh, "_raw", forbidden)
    monkeypatch.setattr(rh, "_patch_restore", forbidden)
    with pytest.raises(ValueError, match="private home"):
        rh.run_process("BOUTDIGEST_on", 15000, str(private), str(tmp_path / "unused.json"))
    assert {p.name: p.read_bytes() for p in private.iterdir()} == before
    assert rh.os.environ["NORD_HOME"] == old_home


@pytest.mark.parametrize("alias", ["hardlink", "symlink", "outside"])
def test_bout_plant_refuses_lake_aliases(tmp_path, monkeypatch, alias):
    private, token = rh._create_private_home(tmp_path)
    monkeypatch.setattr(rh, "_throwaway_home", private.resolve())
    monkeypatch.setattr(rh, "_throwaway_token", token)
    monkeypatch.setenv("NORD_HOME", str(private))
    outside = tmp_path / "durable.bin"
    outside.write_bytes(b"durable lake")
    lake_path = private / "brain_lake.bin"
    if alias == "outside":
        monkeypatch.setattr(home, "knowledge_path", lambda: str(outside))
    else:
        try:
            if alias == "hardlink":
                rh.os.link(outside, lake_path)
            else:
                lake_path.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"filesystem cannot construct {alias}: {exc}")
    with pytest.raises(ValueError, match="private home"):
        rh._plant_bout("BOUTDIGEST")
    assert outside.read_bytes() == b"durable lake"


def test_bout_plant_rechecks_ownership_before_truncation(tmp_path, monkeypatch):
    private, token = rh._create_private_home(tmp_path)
    monkeypatch.setattr(rh, "_throwaway_home", private.resolve())
    monkeypatch.setattr(rh, "_throwaway_token", token)
    monkeypatch.setenv("NORD_HOME", str(private))
    assert home.save_episode(
        {
            "kind": "sleep_bout",
            "event_id": "1-1",
            "weights_digest": "a" * 64,
            "presentations_spent": "2",
        }
    )
    path = Path(home.knowledge_path())
    original = path.read_bytes()

    def revoke():
        (private / rh._HOME_MARK).unlink()
        return original

    monkeypatch.setattr(rh, "_raw", revoke)
    with pytest.raises(ValueError, match="private home"):
        rh._plant_bout("BOUTDIGEST")
    assert path.read_bytes() == original


# ---- round 2: complete arms only, every seed, weight-only c4 (#2769, #2773, #2779, #2797) --------


@pytest.mark.parametrize("role", ["O_cursor", "O_wm", "O_stack"])
@pytest.mark.parametrize("failure", ["actor", "step", "mismatch"])
def test_r3_omission_fault_provenance(monkeypatch, role, failure):
    from snn_data_lake.readout.interaction_learning import SpikingInteractionLearner as Learner
    from snn_data_lake.substrate.learner_events import prefix_digest, sleep_digests

    events = [_sleep_ev(1, 1)]
    learner = Learner(["GREET", "GOAL", "MEMORY"], 52, seed=15000)
    if failure != "mismatch":
        events[0].update(sleep_digests(learner, None), drawn="none", presentations_spent="0")
    if role in ("O_cursor", "O_wm"):
        learner._cursor = (1, 1)
    kw = {
        "receipt": json.dumps({"complete": True, "eligible_ids": ["1-1"]}).encode(),
        "checkpoint_prefix_digest": prefix_digest(events, learner._cursor),
    }

    def step(*args):
        if failure == "step":
            raise OSError("unrelated structural worker failure")
        return None, {"_structural_report": {}}

    def actor(obj, event, covered):
        if not covered and failure == "actor":
            raise ValueError("R2 sleep redo digest/progress mismatch")

    monkeypatch.setattr(Learner, "hydrate", Learner.hydrate)
    monkeypatch.setattr(Learner, "load_state_dict", Learner.load_state_dict)
    probe = rh._patch_restore(role)
    with pytest.raises((OSError, ValueError)) as caught:
        learner.hydrate(
            events, sleep_step={"rewirer": None, "step": step}, effect_actor=actor, **kw
        )
    arms = _arms()
    arms[role].update(
        fault={"cause": f"{type(caught.value).__name__}: {caught.value}"},
        omission_probe=probe,
    )
    if role == "O_cursor":
        arms[role]["cursor_probe"] = probe
    comparison = rh.seed_comparisons(arms)
    if failure == "mismatch":
        assert isinstance(caught.value, rh.RedoMismatch)
        assert comparison["reds"][role]
        assert role not in comparison["incomplete"]
        assert probe["redo_steps"] == 1 and probe["mismatch_verified"]
    else:
        assert not comparison["reds"][role], "foreign faults must never fire an omission red"
        assert role in comparison["incomplete"] and not comparison["arms_complete"]
        assert probe["redo_attempts"] == 1
        assert probe["redo_steps"] == (0 if failure == "step" else 1)
        assert probe["foreign_exception"] and not probe["mismatch_verified"]


@pytest.mark.parametrize("role", list(rh._RED_TWIN))
def test_r3_every_omission_arm_rejects_unattributed_fault(role):
    arms = _arms()
    arms[role]["fault"] = {"cause": "ValueError: R2 sleep redo digest/progress mismatch"}
    comparison = rh.seed_comparisons(arms)
    assert not comparison["reds"][role]
    assert role in comparison["incomplete"] and not comparison["arms_complete"]


@pytest.mark.parametrize("mark", ["missing", "stale", "wrong-root"])
def test_r3_process_launch_nonce_refuses_misdirected_home(tmp_path, monkeypatch, mark):
    private, old_nonce = rh._create_private_home(tmp_path, prefix="earlier-launch-")
    sentinel = private / "brain_lake.bin"
    sentinel.write_bytes(b"durable caller lake")
    if mark == "missing":
        (private / rh._HOME_MARK).unlink()
    elif mark == "wrong-root":
        (private / rh._HOME_MARK).write_text(
            json.dumps({"root": str(tmp_path), "token": old_nonce}), encoding="ascii"
        )
    # The previous environment-token scheme must confer no authority on this launch.
    monkeypatch.setenv("R2_PRIVATE_HOME_TOKEN", old_nonce)
    previous_home = rh.os.environ.get("NORD_HOME")
    with pytest.raises(ValueError, match="private home"):
        rh.run_process(
            "BOUTDIGEST_on", 15000, private, tmp_path / "unused.json", nonce="new-launch"
        )
    assert sentinel.read_bytes() == b"durable caller lake"
    assert rh.os.environ.get("NORD_HOME") == previous_home


def test_r3_omission_control_foreign_fault_is_invalid():
    arms = _arms()
    arms["O_rngv"]["fault"] = {"cause": "OSError: unrelated structural worker failure"}
    comparison = rh.seed_comparisons(arms)
    assert not comparison["c3_controls"]["O_rng_lone_replay_equal"]
    assert "O_rngv" in comparison["incomplete"] and not comparison["arms_complete"]


def test_crashed_or_missing_arm_is_never_a_fire_and_the_seed_is_not_met():
    arms, base = _arms(), _base_arms()
    clean = rh.seed_comparisons(arms, base)
    assert (
        clean["arms_complete"] and clean["c2"]["I_K2_differs"] and not any(clean["reds"].values())
    )
    crashed = copy.deepcopy(arms)
    crashed["O_rng"] = {"role": "O_rng", "error": "exit 1"}
    c = rh.seed_comparisons(crashed, base)
    assert not c["reds"]["O_rng"], "a crashed red arm must not read as FIRED"
    assert not c["arms_complete"] and "O_rng" in c["incomplete"]
    missing = copy.deepcopy(arms)
    del missing["O_rng"]
    assert not rh.seed_comparisons(missing, base)["reds"]["O_rng"]
    for role in ("CHRONO", "MOTOR"):
        broken = copy.deepcopy(arms)
        broken[role] = {"role": role, "error": "exit 1"}
        assert not rh.seed_comparisons(broken, base)["reds"][role]
    gone = copy.deepcopy(base)
    del gone["res_K2"]
    c = rh.seed_comparisons(arms, gone)
    assert not c["c2"]["I_K2_differs"] and not c["arms_complete"]
    per = {s: _per_seed() for s in rh.DEV_SEEDS}
    per[15003] = rh.seed_comparisons(crashed, base)
    v = rh.verdict(per, list(rh.DEV_SEEDS))
    assert 15003 in v["invalid_seeds"] and 15003 in v["incomplete_seeds"]
    assert v["R2-c3"] == "NOT MET" and v["verdict"] != "PASS"


def test_a_red_that_fires_on_one_of_eight_seeds_does_not_pass_r2_c3():
    seeds = list(rh.DEV_SEEDS)
    per = {s: _per_seed() for s in seeds}
    for s in seeds[1:]:
        per[s]["reds"]["O_rng"] = False
    v = rh.verdict(per, seeds)
    assert v["R2-c3"] != "PASS" and v["verdict"] != "PASS"
    assert v["reds"]["O_rng"] is False
    per = {s: _per_seed() for s in seeds}
    assert rh.verdict(per, seeds)["R2-c3"] == "PASS"
    for s in seeds:
        per[s]["reds"]["O_rng"] = False
    assert rh.verdict(per, seeds)["halted"], "a red that fires on no seed halts the arc"


def test_c4_needs_differing_post_night_weight_rows_and_a_stack():
    arms = _arms()
    c = rh.seed_comparisons(arms)
    assert c["c4_structural_moves"], "twin_off's weights differ from twin_rw's in _arms()"
    rewirer_only = copy.deepcopy(arms)
    rewirer_only["twin_off"] = copy.deepcopy(rewirer_only["twin_rw"])
    rewirer_only["twin_off"]["rows"][0]["rewirer"] = {"cycles": 99, "rng": "other"}
    assert not rh.seed_comparisons(rewirer_only)["c4_structural_moves"]
    none_vs_object = copy.deepcopy(rewirer_only)
    for row in none_vs_object["twin_off"]["rows"]:
        row["rewirer"] = None
    assert not rh.seed_comparisons(none_vs_object)["c4_structural_moves"]
    no_stack = copy.deepcopy(arms)
    no_stack["twin_rw"]["stack_present"] = False
    assert not rh.seed_comparisons(no_stack)["c4_structural_moves"], "no stack: NOT MET"


def test_o_rng_real_construction_fires_and_the_lone_after_restore_replay_does_not(tmp_path):
    src = str(Path(rh.__file__).resolve().parents[2])
    fired = rh.run_seed(
        15000, tmp_path / "f", src=src, roles=("pre_rw", "twin_rw", "res_rw", "O_rng")
    )
    vac = rh.run_seed(15000, tmp_path / "v", src=src, roles=("pre_K1", "twin_rwv", "O_rngv"))
    for arms in (fired, vac):
        assert all("error" not in a for a in arms.values()), {
            r: a.get("stderr", "")[-600:] for r, a in arms.items() if "error" in a
        }
    assert rh.records_equal(fired["res_rw"], fired["twin_rw"])
    o = fired["O_rng"]
    assert o["fault"] is None and not rh.records_equal(o, fired["twin_rw"])
    assert o["before"]["rng"] != fired["twin_rw"]["before"]["rng"], "the RNG carrier differs"
    assert rh.records_equal(vac["O_rngv"], vac["twin_rwv"]), (
        "a lone after-restore replay leaves the omitted RNG equal to the fresh one: no fire"
    )


# ---- FAILWRITE (A3) bars and reds, on constructed records --------------------------------------

_RAW_B = {"len": 100, "sha": "b" * 8}
_CALLS = ("i", "ii", "iii", "iv", "v", "vi", "vii")


def _fw_rec(kind="teach", n_i=1, **over):
    """A clean survivor record: every bar of A3 met (totals b + n_i + 2, then b + 2(n_i + 2))."""
    b = 1 if kind == "teach" else 0
    run, calls = b, []
    for p in (1, 2):
        for c in _CALLS:
            run += {"i": n_i, "ii": 1, "iii": 1}.get(c, 0)
            call = {
                "pass": p,
                "call": c,
                "ret": {"iv": "False", "vi": "False"}.get(c),
                "exc": {
                    "v": "RuntimeError: R2 sleep_bout refused by latch until restart",
                    "vii": "PosedRefused: common all-learning latch is closed",
                }.get(c),
                "err": ["R2 checkpoint refused by latch: RuntimeError"] if c == "vi" else [],
                "seq": 2,
                "raw": dict(_RAW_B),
                "cp": "cp_a",
                "sem_equal": True,
                "refused": run,
                "redo": 0,
                "phase": "append" if kind == "teach" else "bout_append_or_preview",
            }
            calls.append(call)
    rec = {
        "role": "fw_t_absent",
        "kind": kind,
        "arm": "absent",
        "a": {"seq": 1, "hash": "h", "redo": 0, "refused": 0, "raw": dict(_RAW_B), "cp": "cp_a"},
        "b": {
            "witnessed": 1,
            "phase": "append" if kind == "teach" else "bout_append_or_preview",
            "seq": 2,
            "sem_equal": True,
            "refused": b,
            "raw": dict(_RAW_B),
            "exc": None,
        },
        "calls": calls,
    }
    rec.update(over)
    return rec


def _at(rec, pass_, call):
    return next(c for c in rec["calls"] if c["pass"] == pass_ and c["call"] == call)


@pytest.mark.parametrize("kind", ["teach", "bout"])
def test_failwrite_clean_record_meets_every_bar_with_conditional_totals(kind):
    for n_i in (0, 1):
        assert rh.fw_survivor_failures(_fw_rec(kind, n_i), n_i) == []
    assert _at(_fw_rec("teach", 1), 1, "vii")["refused"] == 4  # b + n_i + 2
    assert _at(_fw_rec("teach", 1), 2, "vii")["refused"] == 7  # b + 2(n_i + 2)
    assert _at(_fw_rec("bout", 1), 2, "vii")["refused"] == 6
    assert _at(_fw_rec("bout", 0), 2, "vii")["refused"] == 4


def test_failwrite_reds_each_fail_the_named_bar():
    rec = _fw_rec()
    # RED-KIND: the next kind gets through: bytes and seq move at C(iii)
    bad = copy.deepcopy(rec)
    for c in bad["calls"]:
        if c["pass"] == 1 and c["call"] in ("iii", "iv", "v", "vi", "vii"):
            c["seq"], c["raw"] = 3, {"len": 150, "sha": "c" * 8}
    assert rh.fw_red_fired("KIND", bad, 1)
    assert not rh.fw_red_fired("KIND", rec, 1), "the correct build must not fire a red"
    # RED-RETRY: only the repeats append
    bad = copy.deepcopy(rec)
    for c in bad["calls"]:
        if c["pass"] == 2:
            c["seq"], c["raw"] = 3, {"len": 150, "sha": "c" * 8}
    assert rh.fw_red_fired("RETRY", bad, 1) and not rh.fw_red_fired("RETRY", rec, 1)
    # RED-COUNT: refused calls are not counted
    bad = copy.deepcopy(rec)
    for c in bad["calls"]:
        c["refused"] = 1
    assert rh.fw_red_fired("COUNT", bad, 1) and not rh.fw_red_fired("COUNT", rec, 1)
    # RED-CKPT: fails on the True return and the missing line (bytes alone do not satisfy it)
    bad = copy.deepcopy(rec)
    for c in bad["calls"]:
        if c["call"] == "vi":
            c["ret"], c["err"] = "True", []
    assert rh.fw_red_fired("CKPT", bad, 1) and not rh.fw_red_fired("CKPT", rec, 1)
    bytes_only = copy.deepcopy(rec)
    _at(bytes_only, 1, "vi")["raw"] = {"len": 1, "sha": "d" * 8}
    assert not rh.fw_red_fired("CKPT", bytes_only, 1), "bytes alone are not the CKPT bar"
    # RED-POSED: a posed frame is appended
    bad = copy.deepcopy(rec)
    c7 = _at(bad, 1, "vii")
    c7["seq"], c7["raw"], c7["exc"] = 3, {"len": 150, "sha": "c" * 8}, None
    assert rh.fw_red_fired("POSED", bad, 1) and not rh.fw_red_fired("POSED", rec, 1)


def _sem(cursor=(1, 1), ring=("a",), ids=("1-1",), authority=None):
    return {
        "learner": {
            "state": {"tree": {"attrs": {"_cursor": list(cursor), "w": [0.5]}}},
            "cache": {"ring": list(ring), "ids": list(ids), "authority": authority or {}},
        }
    }


def test_failwrite_oracle_red_names_cursor_ring_or_authority():
    restart = _sem(cursor=(1, 2), ring=("a", "b"), ids=("1-1", "1-2"))
    a_only = _sem()
    precursor = _sem()
    assert rh.fw_oracle_fields(restart, a_only) & {"_cursor", "cache.ring", "cache.ids"}
    assert rh.fw_oracle_fields(restart, precursor)
    assert not rh.fw_oracle_fields(restart, copy.deepcopy(restart))


def test_failwrite_gate_red_rearms_where_the_control_latches_reconcile():
    ctl = {"fault": {"phase": "reconcile", "cause": "x"}, "probe_grew": False}
    red = {"fault": None, "probe_grew": True}
    assert rh.fw_gate_fired(ctl, red)
    assert not rh.fw_gate_fired(ctl, {"fault": None, "probe_grew": False})
    assert not rh.fw_gate_fired({"fault": None, "probe_grew": True}, red)
    assert not rh.fw_gate_fired(ctl, {"role": "x", "error": "exit 1"})


def test_failwrite_restart_bars_equality_with_the_right_reference_and_rearm():
    ref_a, ref_ab = {"sem": _sem()}, {"sem": _sem(cursor=(1, 2))}
    good = {
        "fault": None,
        "hydrations": 1,
        "sem": _sem(),
        "raw_before": {"len": 10, "sha": "s", "prefix_len": 10, "prefix_sha": "s", "tail": 0},
        "raw_after": {"len": 10, "sha": "s"},
        "rearm": {"latched": True, "phase": "append"},
    }
    assert rh.fw_restart_failures(good, ref_a, ref_ab, torn=False) == []
    latched = dict(good, fault={"phase": "reconcile", "cause": "x"})
    assert rh.fw_restart_failures(latched, ref_a, ref_ab, torn=False)
    wrong = dict(good, sem=_sem(cursor=(1, 2)))
    assert rh.fw_restart_failures(wrong, ref_a, ref_ab, torn=False)
    same_sides = rh.fw_restart_failures(good, ref_a, ref_a, torn=False)
    assert same_sides, "identical oracle sides read NOT MET (ruling 9)"
    norearm = dict(good, rearm={"latched": False, "phase": None})
    assert rh.fw_restart_failures(norearm, ref_a, ref_ab, torn=False)
    torn = dict(
        good,
        raw_before={"len": 14, "sha": "t", "prefix_len": 10, "prefix_sha": "s", "tail": 4},
    )
    assert rh.fw_restart_failures(torn, ref_a, ref_ab, torn=True) == []
    unrepaired = dict(torn, raw_after={"len": 14, "sha": "t"})
    assert rh.fw_restart_failures(unrepaired, ref_a, ref_ab, torn=True)


def test_c8_needs_all_three_crash_points_one_event_and_the_planted_bout_slots():
    assert set(rh.BOUT_CRASH_POINTS) == {"Bpre", "B", "Bpost"}
    arms = _arms()
    ok = {"frames_added": 1, "kinds_added": ["sleep_bout"], "prefix_preserved": True}
    ok.update(cache_equal=True, replay_counts_equal=True, effect_applied=True)
    for r in ("twin_B", "pre_B", "pre_Bpost"):
        arms[r]["bout_check"] = dict(ok)
    arms["pre_Bpost"]["checkpoint_ok"] = True
    arms["pre_Bpre"] = {"role": "pre_Bpre", "bout_check": dict(ok, effect_applied=False)}
    c = rh.c8_seed(arms)
    assert c["a"] and c["c"], c
    killed = copy.deepcopy(arms)
    killed["res_Bpre"]["rows"][0]["weights"][0][0] += 1.0
    assert not rh.c8_seed(killed)["a"], "an unequal restart at the first crash point fails (a)"
    two = copy.deepcopy(arms)
    two["twin_B"]["bout_check"]["frames_added"] = 2
    assert not rh.c8_seed(two)["c"], "two lake records for one bout fail (c)"
    leak = copy.deepcopy(arms)
    leak["twin_B"]["bout_check"]["prefix_preserved"] = False
    assert not rh.c8_seed(leak)["c"]
    ring = copy.deepcopy(arms)
    ring["twin_B"]["bout_check"]["cache_equal"] = False
    assert not rh.c8_seed(ring)["c"], "ring/authority must be unchanged by a bout"
    assert set(rh.PLANTED) >= {"BOUTDIGEST", "BOUTPROGRESS"}


# ---- real processes, dev seed: FAILWRITE absent arm + RED-CKPT, R2-c8 first crash point ----


def _run(tmp_path, name, roles):
    src = str(Path(rh.__file__).resolve().parents[2])
    arms = rh.run_seed(15000, tmp_path / name, src=src, roles=roles)
    assert all("error" not in a for a in arms.values()), {
        r: a.get("error", "")[-800:] for r, a in arms.items() if "error" in a
    }
    return arms


def test_real_failwrite_absent_arm_meets_every_bar_and_red_ckpt_fires(tmp_path):
    arms = _run(
        tmp_path,
        "fw",
        ("fw_ref_A", "fw_ref_AB", "fw_t_absent", "fw_rs_t_absent", "fw_red_ckpt"),
    )
    n_i = arms["fw_ref_AB"]["n_i"]
    assert n_i in (0, 1), "n_i is read from the never-faulted reference, not assumed"
    assert rh.fw_survivor_failures(arms["fw_t_absent"], n_i) == []
    assert rh.fw_restart_failures(arms["fw_rs_t_absent"], arms["fw_ref_A"], arms["fw_ref_AB"]) == []
    assert rh.fw_red_fired("CKPT", arms["fw_red_ckpt"], n_i)
    assert not rh.fw_red_fired("KIND", arms["fw_t_absent"], n_i)


def test_real_bout_killed_between_event_and_effect_restarts_equal_to_its_twin(tmp_path):
    arms = _run(
        tmp_path,
        "c8",
        ("twin_B", "pre_B", "pre_Bpre", "res_Bpre", "BOUTDIGEST_on", "BOUTDIGEST_off"),
    )
    chk = arms["pre_Bpre"]["bout_check"]
    assert chk["frames_added"] == 1 and chk["effect_applied"] is False
    assert arms["twin_B"]["bout_check"]["kinds_added"] == ["sleep_bout"]
    assert rh.records_equal(arms["res_Bpre"], arms["twin_B"])
    assert "sleep_bout redo digest/progress mismatch" in arms["BOUTDIGEST_on"]["fault"]["cause"]
    assert arms["BOUTDIGEST_off"]["fault"] is None


# ---- round 3 (Codex #2885): the observer is non-mutating; FW-PARTIAL --------------------------


def test_observer_decodes_the_checkpoint_from_raw_bytes_and_never_repairs_a_torn_tail(lake):
    import hashlib
    import os

    assert home.save_named_state("interaction_learner.json", b"cp-one")
    assert home.save_named_state("interaction_learner.json", b"cp-two")
    path = home.knowledge_path()
    with open(path, "ab") as f:  # a torn tail: a length header and part of its payload
        f.write(b"\x40\x00\x00\x00" + b"\x01" * 9)
    before = Path(path).read_bytes()
    assert rh._frames(before)[1] < len(before)
    assert rh._raw_state(before, "interaction_learner.json") == b"cp-two"
    assert rh._raw_state(before, "absent.json") is None
    after = Path(path).read_bytes()
    assert (len(after), hashlib.sha256(after).hexdigest()) == (
        len(before),
        hashlib.sha256(before).hexdigest(),
    )
    assert os.path.getsize(path) == len(before)


def test_fw_partial_is_caught_by_the_strict_bar_but_not_by_the_prefix_comparator():
    rec = _fw_rec()
    assert not rh.fw_red_fired("PARTIAL", rec, 1), "the correct build must not fire a red"
    bad = copy.deepcopy(rec)
    for c in bad["calls"]:
        if c["pass"] == 2 or c["call"] in ("iii", "iv", "v", "vi", "vii"):
            c["raw"] = {"len": _RAW_B["len"] + 13, "sha": "partial"}
        c["prefix"] = dict(_RAW_B)
    bad["b"]["prefix"] = dict(_RAW_B)
    assert rh.fw_red_fired("PARTIAL", bad, 1)
    assert rh.fw_survivor_failures(bad, 1, raw_key="prefix") == []
    early = copy.deepcopy(bad)  # installed from C(i), not C(iii): not the registered construction
    _at(early, 1, "i")["raw"] = {"len": _RAW_B["len"] + 13, "sha": "partial"}
    assert not rh.fw_red_fired("PARTIAL", early, 1)


def test_real_partial_fires_on_all_six_arms_and_the_strict_bar_holds_without_it(tmp_path):
    roles = ["fw_ref_A", "fw_ref_AB", "fw_ref_ABb"]
    roles += [f"fw_{k}_{a}" for k in "tb" for a in ("absent", "landed", "torn")]
    roles += [f"fw_p_{k}_{a}" for k in "tb" for a in ("absent", "landed", "torn")]
    arms = _run(tmp_path, "six", tuple(roles))
    moved = {}
    for kind, ref in (("t", "fw_ref_AB"), ("b", "fw_ref_ABb")):
        n_i = arms[ref]["n_i"]
        for arm in ("absent", "landed", "torn"):
            clean, red = arms[f"fw_{kind}_{arm}"], arms[f"fw_p_{kind}_{arm}"]
            assert rh.fw_survivor_failures(clean, n_i) == [], (kind, arm)
            assert rh.fw_red_fired("PARTIAL", red, n_i), (kind, arm)
            assert not rh.fw_red_fired("PARTIAL", clean, n_i), (kind, arm)
            moved[kind, arm] = bool(rh.fw_survivor_failures(red, n_i, raw_key="prefix"))
    # on a torn arm the partial append may complete the torn frame: the prefix moving there is fine
    assert not moved["t", "absent"], "teach/absent: the prefix comparator alone stays blind"


def test_survivor_path_never_truncates_a_torn_lake_between_c_calls(tmp_path):
    arms = _run(tmp_path, "surv", ("fw_ref_A", "fw_ref_AB", "fw_t_torn"))
    rec = arms["fw_t_torn"]
    assert rec["b"]["tail"] > 0
    assert {c["raw"]["len"] for c in rec["calls"]} == {rec["b"]["raw"]["len"]}
