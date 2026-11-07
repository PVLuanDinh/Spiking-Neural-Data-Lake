"""Round 3 software controls; constructed parameters, no deciding campaign."""

import copy
import math
from dataclasses import replace

import pytest
from test_r2_replay_hydration import _close_for_restart, _private_brain, _restart, _semantic


def _bind(brain, *, n=2, origin="born", arm="RSTDP_BRAIN", population=None, routing=None):
    from snn_data_lake.growth.grown_code import GrownUnit
    from snn_data_lake.growth.growth_loop import BudgetValues, Ledger, make_admission
    from snn_data_lake.instruments.taught_problems import PROBLEMS
    from snn_data_lake.readout.posed_problem import unit_rows
    from snn_data_lake.substrate import home

    problem = PROBLEMS["ii.xor"]
    labels = tuple(brain.brain._ixn.responses)
    units = (
        tuple(
            GrownUnit(i, (0.1,) * 52, 0.5, 12, -1 if i == n - 1 else 1, (0.0,) * len(labels))
            for i in range(n)
        )
        if population is None
        else population
    )
    ledger = Ledger(home.knowledge_path(), BudgetValues(52, None))
    ledger.open_problem(problem.id, problem.n_teach_codes(0))
    births = []
    if origin == "born":
        admission = make_admission(
            site=(0.5 / 14, 1.5 / 4), centre_um=(0.0, 0.0), density_per_mm2=2.7e4, seed=9000
        )
        rows, ticket = admission(
            units,
            problem.id,
            0,
            resolve_problem=lambda p: p == problem.id,
            known_unit=ledger.known_unit,
        )
        ledger.record_units(units, rows, problem.id, 0)
        ledger.record_spend(problem.id, n, 3)
        # Bind is a first installation of already journaled births; the receipt
        # comes from this same lake, and the real G7 admission ticket is consumed.
        ticket.consume(rows)
        births = list(ledger.birth_records().values())
    factory = {
        "problem_id": problem.id,
        "labels": list(labels),
        "seed": 0,
        "arm": arm,
        "population_origin": origin,
        "units": unit_rows(units),
        "birth_rows": births,
        "w_feed": [[0.0] * n for _ in range(256)],
        "t_post_g": float(max(u.window for u in units) + 3),
        "source_signs": {label: 1 for label in labels},
        "attachments": [
            {
                "source": label,
                "target": unit.unit_id,
                "sign": 1,
                "weight": 0.2,
                "delay": 0,
                "receipt": "1" * 64,
            }
            for label in labels
            for unit in units
        ],
        "out_eligible": [[True] * len(labels) for _ in units],
    }
    if routing is not None:
        factory["attachments"] = routing
    brain.bind_part_d_reader(factory)
    assert brain._r2_fault is None
    return ledger, factory, brain._g9_actors[problem.id]


def _episode():
    from snn_data_lake.instruments.taught_problems import PROBLEMS

    return next(iter(PROBLEMS["ii.xor"].teach_epochs(0, 1)))[0]


def _admit(brain, ledger, actor):
    from snn_data_lake.growth.growth_loop import make_admission
    from snn_data_lake.readout.posed_problem import unit_rows

    new = replace(actor.units[0], unit_id=max(u.unit_id for u in actor.units) + 1)
    admission = make_admission(
        site=(0.5 / 14, 1.5 / 4), centre_um=(0.0, 0.0), density_per_mm2=2.7e4, seed=9000
    )
    pid, batch = actor.factory["problem_id"], ledger.next_batch_id()
    labels, ticket = admission(
        (new,), pid, batch, resolve_problem=lambda p: p == pid, known_unit=ledger.known_unit
    )
    ledger.record_units((new,), labels, pid, batch)
    ledger.record_spend(pid, 1, 1)
    attachments = copy.deepcopy(actor.factory["attachments"])
    attachments += [
        {
            "source": label,
            "target": new.unit_id,
            "sign": 1,
            "weight": 0.2,
            "delay": 0,
            "receipt": "2" * 64,
        }
        for label in actor.reader.labels
    ]
    brain._g9_apply_source(
        {
            "op": "admit",
            "problem_id": pid,
            "units": unit_rows((*actor.units, new)),
            "birth_rows": list(ledger.birth_records().values()),
            "w_feed": [row + [0.0] for row in actor.feed],
            "attachments": attachments,
        },
        ticket=ticket,
    )
    assert ticket.consumed
    return brain._g9_actors[pid]


def _teach(brain, actor, episode=None):
    return brain.learn_part_d_episode(
        actor.factory["problem_id"],
        _episode() if episode is None else episode,
        neuron_mask=[True] * len(actor.reader.learner.neurons),
        return_mask=[True] * len(actor.port.rows),
        return_neutered=False,
    )


@pytest.mark.pool
def test_posed_checkpoint_and_pending_event_survive_real_restart():
    with _private_brain() as (brain, private):
        _, _, actor = _bind(brain)
        _teach(brain, actor)
        assert brain._flush_interaction_learner()
        _teach(brain, actor)
        expected = _semantic(brain)
        brain._r2_close("constructed_crash", RuntimeError("before checkpoint"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected
        assert restarted["redo"] == 1


def test_learning_off_read_preserves_weights_and_traces():
    from snn_data_lake.instruments.taught_problems import Turn

    with _private_brain() as (brain, _):
        _, _, actor = _bind(brain)
        _teach(brain, actor)
        before = actor.reader.learner.state_dict()
        weights = list(actor.port.weights)
        episode = tuple(Turn(t.text, t.role, None) for t in _episode())
        brain.read_part_d_episode(
            actor.factory["problem_id"],
            episode,
            neuron_mask=[False] * len(actor.reader.learner.neurons),
            return_mask=[False] * len(actor.port.rows),
            return_neutered=False,
        )
        assert actor.reader.learner.state_dict() == before
        assert actor.port.weights == weights


def test_complete_actor_carrier_keeps_real_replay_ring_and_clock_join():
    from snn_data_lake.readout.posed_problem import PosedActor, PosedRefused

    with _private_brain() as (brain, _):
        _, _, actor = _bind(brain)
        _teach(brain, actor)
        expected = actor.reader.learner._snapshot()
        assert actor.reader.learner._replay_ids == [actor.last_event_id]
        saved = actor.state_dict()
        restored = PosedActor.restore(saved, brain._g9_birth_records())
        assert restored.reader.learner._snapshot() == expected
        broken = copy.deepcopy(saved)
        broken["clock"] += 1.0
        with pytest.raises(PosedRefused, match="G9_RETURN_CLOCK_JOIN"):
            PosedActor.restore(broken, brain._g9_birth_records())


def test_real_resource_unit_and_silencing_reds():
    from snn_data_lake.readout.posed_problem import (
        PosedActor,
        PosedRefused,
        resource_receipt,
        unit_rows,
    )

    with _private_brain() as (brain, _):
        ledger, factory, engine = _bind(brain)
        constructed = copy.deepcopy(factory)
        constructed.update(population_origin="constructed", birth_rows=[])
        control = PosedActor(constructed, {})
        receipt = resource_receipt({"engine": engine, "presized": control}, "engine", ledger)
        assert receipt["arms"]["engine"]["U"] == 2
        assert receipt["arms"]["engine"]["I"] == 1
        assert receipt["arms"]["engine"]["candidate_evaluations"] == 3
        assert receipt["arms"]["presized"]["candidate_evaluations"] == 0
        enlarged = _admit(brain, ledger, engine)
        with pytest.raises(PosedRefused, match="G9_RQ1_UNIT_COUNT_MISMATCH"):
            resource_receipt({"engine": enlarged, "presized": control}, "engine", ledger)
        # A separately constructed real comparator matches the now-admitted size.
        constructed.update(
            units=unit_rows(enlarged.units),
            w_feed=copy.deepcopy(enlarged.feed),
            attachments=copy.deepcopy(enlarged.factory["attachments"]),
            out_eligible=copy.deepcopy(enlarged.factory["out_eligible"]),
        )
        control = PosedActor(constructed, {})
        engine = enlarged
        engine.units = tuple(u.silenced() if u.unit_id == 0 else u for u in engine.units)
        control.units = tuple(u.silenced() for u in control.units)
        engine.factory["units"] = unit_rows(engine.units)
        control.factory["units"] = unit_rows(control.units)
        with pytest.raises(PosedRefused, match="G9_RQ2_SILENCING_COUNT_MISMATCH"):
            resource_receipt(
                {"engine": engine, "presized": control},
                "engine",
                ledger,
                silenced={"engine": (0,), "presized": (0, 1, 2)},
            )
        # Challenger #2186: actual inactive counts are 1 versus 3, but the
        # caller lists only one in each arm. The complete intervention must refuse.
        assert sum(not u.active for u in engine.units) == 1
        assert sum(not u.active for u in control.units) == 3
        with pytest.raises(PosedRefused, match="G9_RQ2_UNLISTED_INACTIVE_UNITS: presized"):
            resource_receipt(
                {"engine": engine, "presized": control},
                "engine",
                ledger,
                silenced={"engine": (0,), "presized": (0,)},
            )
        # Actual matched green: restore the comparator and silence one other ID.
        control.units = tuple(replace(u, active=u.unit_id != 1) for u in control.units)
        control.factory["units"] = unit_rows(control.units)
        receipt = resource_receipt(
            {"engine": engine, "presized": control},
            "engine",
            ledger,
            silenced={"engine": (0,), "presized": (1,)},
        )
        assert receipt["equalized"] == ["U", "silenced_N"]
        assert receipt["arms"]["engine"]["silenced_N"] == 1
        assert receipt["arms"]["presized"]["silenced_N"] == 1


@pytest.mark.parametrize("omitted_arm", ["engine", "presized"])
def test_resource_silencing_refuses_unlisted_inactive_even_at_equal_actual_count(omitted_arm):
    from snn_data_lake.readout.posed_problem import PosedActor, PosedRefused, resource_receipt

    with _private_brain() as (brain, _):
        ledger, factory, engine = _bind(brain)
        constructed = copy.deepcopy(factory)
        constructed.update(population_origin="constructed", birth_rows=[])
        control = PosedActor(constructed, {})
        # Both actual interventions silence two units; a subset declaration
        # still cannot certify completeness, including an empty subset.
        engine.units = tuple(u.silenced() for u in engine.units)
        control.units = tuple(u.silenced() for u in control.units)
        arms = {"engine": engine, "presized": control}
        for omitted in ((0,), ()):
            declared = {"engine": (0, 1), "presized": (0, 1)}
            declared[omitted_arm] = omitted
            with pytest.raises(
                PosedRefused, match=f"G9_RQ2_UNLISTED_INACTIVE_UNITS: {omitted_arm}"
            ):
                resource_receipt(arms, "engine", ledger, silenced=declared)
        receipt = resource_receipt(
            arms, "engine", ledger, silenced={"engine": (0, 1), "presized": (0, 1)}
        )
        assert receipt["arms"]["engine"]["silenced_N"] == 2
        assert receipt["arms"]["presized"]["silenced_N"] == 2


@pytest.mark.pool
def test_new_admission_is_journaled_before_install_and_survives_restart():
    with _private_brain() as (brain, private):
        ledger, _, actor = _bind(brain)
        _teach(brain, actor)
        assert brain._flush_interaction_learner()
        actor = _admit(brain, ledger, actor)
        assert len(actor.units) == 3
        expected = _semantic(brain)
        brain._r2_close("constructed_crash", RuntimeError("after admitted birth"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected


@pytest.mark.pool
def test_no_checkpoint_complete_prefix_rebuilds_first_actor_and_births():
    with _private_brain() as (brain, private):
        ledger, _, actor = _bind(brain)
        actor = _admit(brain, ledger, actor)
        _teach(brain, actor)
        expected = _semantic(brain)
        brain._r2_close("constructed_crash", RuntimeError("no first checkpoint"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected
        assert restarted["redo"] == 3  # bind + actual admission + teach


def test_readout_and_return_masks_freeze_actual_weights():
    with _private_brain() as (brain, _):
        _, _, actor = _bind(brain)
        before = [list(n.w) for n in actor.reader.learner.neurons]
        return_before = list(actor.port.weights)
        brain.learn_part_d_episode(
            actor.factory["problem_id"],
            _episode(),
            neuron_mask=[False] * len(actor.reader.learner.neurons),
            return_mask=[False] * len(actor.port.rows),
            return_neutered=False,
        )
        assert [n.w for n in actor.reader.learner.neurons] == before
        assert actor.port.weights == return_before


@pytest.mark.pool
def test_posed_payload_tamper_preserves_full_prefix_refusal():
    import json

    from snn_data_lake.substrate import home
    from snn_data_lake.substrate.binlog import pack_episode, unpack_episode

    with _private_brain() as (brain, private):
        _, _, actor = _bind(brain)
        _teach(brain, actor)
        assert brain._flush_interaction_learner()
        _close_for_restart(brain)
        records = list(home._log().replay())
        changed = False
        for i, row in enumerate(records):
            if row[:1] != home._T_EPISODE:
                continue
            event = unpack_episode(row[1:])
            if event["kind"] == "posed_problem":
                value = json.loads(event["source"])
                if value["op"] != "teach":
                    continue
                value["turns"][-1][0] += " payload red"
                event["source"] = json.dumps(value, sort_keys=True, separators=(",", ":"))
                records[i] = home._T_EPISODE + pack_episode(event)
                changed = True
                break
        assert changed
        home._log().rewrite(records)
        restarted = _restart(private)
        assert restarted["fault"] is not None
        assert "canonical prefix digest mismatch" in restarted["fault"]["cause"]
        assert restarted["cursor"] is None and restarted["hydrations"] == 0


def test_named_width_refusal_leaves_writer_untouched():
    from snn_data_lake.readout.posed_problem import PosedRefused

    with _private_brain() as (brain, _):
        _, factory, _ = _bind(brain)
        before = brain._r2_seq
        factory["units"][0]["w_in"] = [0.0] * 64
        with pytest.raises(PosedRefused, match="G9_CHAT52_WIDTH"):
            brain.bind_part_d_reader(factory)
        assert brain._r2_seq == before


def test_pool_width_refusal_is_named_before_writer_effect():
    from snn_data_lake.readout.posed_problem import PosedRefused

    with _private_brain() as (brain, _):
        _, factory, _ = _bind(brain)
        before = brain._r2_seq
        factory["w_feed"] = factory["w_feed"][:64]
        with pytest.raises(PosedRefused, match="G9_POOL256_WIDTH"):
            brain.bind_part_d_reader(factory)
        assert brain._r2_seq == before


def test_actual_snapshot_failure_closes_all_learning_before_append(monkeypatch):
    with _private_brain() as (brain, _):
        _, _, actor = _bind(brain)
        sequence = brain._r2_seq

        def broken():
            raise ValueError("constructed actual snapshot refusal")

        monkeypatch.setattr(actor, "snapshot", broken)
        with pytest.raises(ValueError, match="snapshot refusal"):
            _teach(brain, actor)
        assert brain._r2_seq == sequence
        assert brain._r2_fault["phase"] == "g9_snapshot"
        assert not brain._flush_interaction_learner()


def test_posed_entry_uses_existing_common_chat_carrier_guard():
    with _private_brain() as (brain, _):
        _, _, actor = _bind(brain)
        sequence = brain._r2_seq
        brain.brain._ixn.n_in = 64  # planted unsupported owner profile after boot
        with pytest.raises(ValueError, match="52-input chat carrier"):
            _teach(brain, actor)
        assert brain._r2_seq == sequence
        assert brain._r2_fault["phase"] == "g9_snapshot"
        brain.brain._ixn.n_in = 52


def test_landed_append_failure_rolls_back_and_closes_common_latch(monkeypatch):
    from snn_data_lake.substrate import home

    with _private_brain() as (brain, _):
        _, _, actor = _bind(brain)
        before = _semantic(brain)
        original = home.save_episode

        def landed(row):
            assert original(row)
            raise OSError("constructed landed append failure")

        monkeypatch.setattr(home, "save_episode", landed)
        with pytest.raises(OSError, match="landed append failure"):
            _teach(brain, actor)
        assert brain._r2_fault is not None
        assert _semantic(brain) == before
        assert not brain._flush_interaction_learner()


#: Fixed, closed candidate texts for the D10 probe and isolation queries (no search over a corpus).
_D10_QUERIES = (
    "hello unseen return receiver",
    "what time is it",
    "please help me",
    "tell me a story",
    "goodbye for now",
    "thank you very much",
    "open the door",
    "how are you today",
)
#: Probe basal peak as a fraction of theta (A2.T condition 4 requires [0.5, 1)).
_D10_PROBE_RATIO = 0.9


def _d10_fixture(teach_ranks, tau):
    """Two rank coordinates (i, j) and a shared window W: the teach episode has BOTH inside W
    (supra-threshold basal, the unit fires and its return can learn), the probe query only i
    (sub-threshold basal), the isolation query neither (zero basal)."""
    from snn_data_lake.instruments import inseparable_mapping as im

    codes = {q: im.ranks(q) for q in _D10_QUERIES}
    span = tau * math.log(1.0 / (1.0 / _D10_PROBE_RATIO - 1.0))
    for i in range(52):
        for j in range(52):
            ri, rj = teach_ranks[i], teach_ranks[j]
            if i == j or ri > rj:
                continue
            window = rj + 1
            if window - 1 - ri > span:
                continue
            for probe, qp in codes.items():
                if not qp[i] < window <= qp[j]:
                    continue
                for iso, qz in codes.items():
                    if iso != probe and qz[i] >= window and qz[j] >= window:
                        return i, j, window, probe, iso
    raise AssertionError("no D10 fixture in the closed candidate list")


def test_apical_return_changes_brains_own_next_answer_untrained_and_random_do_not(monkeypatch):
    """D10 replacement of the additive-era positive (PREREG 4fc3ff89c2e9, A1.T + A2.T; D-362).

    CONSTRUCTED fixture: it proves the return port CAN charge the brain's own next answer under
    apical entry; it is NOT the engine charge (that is E2's floor on the brain's own posed problems,
    PREREG A2.B3). theta is ~1e-17 (below Part D's tiny first anti-causal update), so A/theta is
    large and tanh saturates: the gate acts as a fixed ~2x step here, not a graded gain.

    The teach episode drives two coordinates inside the window (supra-threshold basal, the unit
    fires, the zero-born return learns through the genuine _teach scalar). The probe query drives
    one (basal peak in [theta/2, theta): silent alone, crosses only with the learned return). The
    isolation query drives none: the same learned return alone must NOT flip it (the additive
    entry's positive, archived as R4 on the 2f5a9c22 build). Kept from the additive-era test:
    pre-teach == baseline, zero-born weights, no assigned reward, `assert emitted`, the untrained
    and the random-routing controls. Added: the isolation probe and the gain=1 plant red.
    """
    from snn_data_lake.growth import grown_code as gc
    from snn_data_lake.growth.grown_code import GrownUnit
    from snn_data_lake.instruments import inseparable_mapping as im
    from snn_data_lake.readout.posed_problem import actor_context

    with _private_brain() as (brain, _):
        ixn = brain.brain._ixn
        episode = _episode()
        ranks = im.ranks(episode[-1].text)
        index, other, window, query, iso_query = _d10_fixture(ranks, ixn.tau)
        baseline = ixn.route_text(query)
        iso_baseline = ixn.route_text(iso_query)
        target = next(label for label in ixn.responses if label != baseline)
        future = im.ranks(query)
        w = tuple(1e-17 if k in (index, other) else 0.0 for k in range(52))
        theta = 1e-17 * math.exp(-1.0 / ixn.tau) / _D10_PROBE_RATIO
        output = tuple(10000.0 if label == target else 0.0 for label in ixn.responses)
        units = (
            GrownUnit(0, w, theta, window, 1, output),
            GrownUnit(1, w, theta, window, -1, (0.0,) * len(output)),
        )
        labels = ixn.responses
        routing = [
            {
                "source": label,
                "target": uid,
                "sign": 1,
                "weight": 0.0,
                "delay": 0,
                "receipt": "3" * 64,
            }
            for label in labels
            for uid in (0, 1)
        ]
        _, _, actor = _bind(brain, population=units, routing=routing)
        teach_tau = actor.reader.learner.tau
        # Fixture preconditions, measured with the no-return (basal) drive.
        teach_peak = max(gc._drive(units[0], ranks, teach_tau, t) for t in range(1, window + 1))
        probe_peak = max(gc._drive(units[0], future, ixn.tau, t) for t in range(1, window + 1))
        iso_peak = max(
            gc._drive(units[0], im.ranks(iso_query), ixn.tau, t) for t in range(1, window + 1)
        )
        b_ratio = probe_peak / theta
        print(f"D10 fixture: i={index} j={other} W={window} B/theta={b_ratio!r}")
        assert teach_peak >= theta, "teach episode must fire the unit on its own basal drive"
        assert 0.5 <= b_ratio < 1.0, b_ratio
        assert iso_peak == 0.0
        assert ixn.route_text(query) == baseline
        assert ixn.route_text(iso_query) == iso_baseline
        # Use actual frozen teaching episodes until a genuine wrong-answer
        # scalar reaches the real return receiver; do not assign a reward.
        _teach(brain, actor, episode)
        assert max(actor.port.weights) > 0.0, "the real learning positive must move return weights"
        intact = ixn.route_text(query)
        aff = dict(actor.port.context(actor.units, actor.clock))
        a_ratio = (
            max(
                sum(wt * gc._psp(t - arrival, ixn.tau) for arrival, wt in aff[0])
                for t in range(1, window + 1)
            )
            / theta
        )
        print(f"D10 probe: A/theta={a_ratio!r} intact={intact} baseline={baseline}")
        actor.return_neutered = True
        brain._g9_publish_context(actor)
        neutered = ixn.route_text(query)
        assert intact == target and neutered == baseline, {
            "intact": intact,
            "target": target,
            "neutered": neutered,
            "baseline": baseline,
            "weights": actor.port.weights,
            "clock": actor.clock,
            "window": window,
            "B/theta": b_ratio,
            "A/theta": a_ratio,
            "context": aff,
        }
        # Isolation probe: the same learned return alone (zero basal) must not flip the answer.
        actor.return_neutered = False
        brain._g9_publish_context(actor)
        assert ixn.route_text(iso_query) == iso_baseline, "the return alone flipped the answer"
        # gain=1 plant (R3 inside the live brain): A computed, never applied -> no charge.
        with monkeypatch.context() as m:
            m.setattr(gc, "_apical_gain", lambda apical, th: 1.0, raising=False)
            assert ixn.route_text(query) == baseline, "gain=1 plant still charged the answer"
        emitted = {
            row["source"]
            for row, weight in zip(actor.port.rows, actor.port.weights, strict=True)
            if row["target"] == 0 and weight > 0.0
        }
        assert emitted
        # Untrained carrier has identical born parameters, but no learned
        # positive return: replay the actual arrivals with zero-born weights.
        zero = copy.deepcopy(actor_context(actor))
        zero["afferents"] = [
            [uid, [[t, 0.0] for t, _ in values]] for uid, values in zero["afferents"]
        ]
        ixn._g9_context = zero
        assert ixn.route_text(query) == baseline
        # Equal-count wrong-target control delivers the same actual learned
        # arrivals to the pre-existing inhibitory unit with zero output.
        random_context = copy.deepcopy(zero)
        actual = actor.port.context(actor.units, actor.clock)
        import random

        permutation = [0, 1]
        random.Random(1).shuffle(permutation)  # constructed, reproducible random routing control
        assert permutation == [1, 0]
        random_context["afferents"] = [
            [uid, [list(pair) for pair in actual[permutation[uid]]] if uid == 1 else []]
            for uid in (0, 1)
        ]
        ixn._g9_context = random_context
        assert ixn.route_text(query) == baseline
        actor.return_neutered = False
        brain._g9_publish_context(actor)


def _malformed_teaches(actor):
    """Deterministic input errors that the actor itself refuses (finding A1)."""
    from snn_data_lake.instruments.taught_problems import Turn

    width, rows = len(actor.reader.learner.neurons), len(actor.port.rows)
    episode = _episode()
    bad_target = [*episode[:-1], Turn(episode[-1].text, episode[-1].role, "NO_SUCH_LABEL")]
    return {
        "empty neuron mask": dict(neuron_mask=[], return_mask=[True] * rows),
        "short return mask": dict(neuron_mask=[True] * width, return_mask=[True] * (rows - 1)),
        "long neuron mask": dict(neuron_mask=[True] * (width + 1), return_mask=[True] * rows),
        "unregistered target": dict(
            neuron_mask=[True] * width, return_mask=[True] * rows, episode=bad_target
        ),
        "unregistered shuffle": dict(
            neuron_mask=[True] * width, return_mask=[True] * rows, shifted_label="NO_SUCH_LABEL"
        ),
    }


@pytest.mark.pool
def test_malformed_effect_is_refused_before_append_and_restart_recovers():
    from snn_data_lake.readout.posed_problem import PosedRefused

    with _private_brain() as (brain, private):
        _, _, actor = _bind(brain)
        _teach(brain, actor)
        assert brain._flush_interaction_learner()
        for name, kw in _malformed_teaches(actor).items():
            sequence, teaches = brain._r2_seq, actor.teaches
            episode = kw.pop("episode", None) or _episode()
            with pytest.raises(PosedRefused):
                brain.learn_part_d_episode(
                    actor.factory["problem_id"], episode, return_neutered=False, **kw
                )
            assert brain._r2_seq == sequence, name
            assert actor.teaches == teaches, name
            assert brain._r2_fault is None, name
        _teach(brain, actor)
        expected = _semantic(brain)
        brain._r2_close("constructed_crash", RuntimeError("before checkpoint"))
        _close_for_restart(brain)
        restarted = _restart(private)
        assert restarted["fault"] is None
        assert restarted["semantic"] == expected
