"""G2(ii)'s generator and G-R2's construction check (instruments/inseparable_mapping.py): the four
words' buckets re-derived with zlib.crc32 here (a second instrument beside the module's own, which
reads the learner's featurizer), a source-text pin of live_brain's learner construction, every check
A1-A5 and every constructed red, the development measurement's recorded numbers, and the
pool-reading stream's schema against rb6_stream's. Only learner seeds 7100-7103 run here; the check
seeds 7000-7007 are the campaign's, reachable only through `construction_check`'s default, which no
test calls."""

import ast
import copy
import inspect
import json
import math
import random
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import pytest

from snn_data_lake.engine import live_brain
from snn_data_lake.instruments import carried_state_campaign as ccc
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments import memory_diagnostic as md
from snn_data_lake.readout import interaction_learning as il

DEV = ccc.G2II_DEV_SEEDS
DEV_EPOCHS_TO_STABLE = {7100: 122, 7101: 170, 7102: 112, 7103: 124}  # the recorded measurement
FORBIDDEN = (
    "learn",
    "update",
    "train",
    "adapt",
    "consolidate",
    "reinforce",
    "fit",
    "plast",
    "potentiate",
    "depress",
    "calibrate",
)
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "specs/022-unified-brain-arc/evidence/g1c-carried-state.md"
)


def independent_ranks(text):
    """The rank-order code of a text, derived here from crc32 and the featurizer's documented
    weights (1 + 0.1 * position + 0.01 * length), ties to the lowest index."""
    vec = [0.0] * 52
    for pos, word in enumerate(text.split(), start=1):
        vec[zlib.crc32(word.encode("ascii")) % 52] += 1.0 + 0.1 * pos + 0.01 * len(word)
    order = sorted(range(52), key=lambda i: (-vec[i], i))
    out = [0] * 52
    for rank, i in enumerate(order):
        out[i] = rank
    return out


def independent_code(text):
    return [math.exp(-(6 - r) / 8.0) if 6 - r > 0 else 0.0 for r in independent_ranks(text)]


# ---- the generator ----------------------------------------------------------------------------
def test_the_four_words_hash_to_the_plans_buckets_by_an_independent_derivation():
    derived = {w: zlib.crc32(w.encode("ascii")) % 52 for w in ("cat", "dog", "sun", "sky")}
    assert derived == {"cat": 28, "dog": 41, "sun": 31, "sky": 51} == im.BUCKETS_52
    assert len(set(derived.values())) == 4 and all(b > 1 for b in derived.values())
    # the module's own derivation (the learner's featurizer) agrees; its constant builds nothing
    assert {w: im._bucket(w) for w in im.WORDS} == derived
    assert im.check_a1()["buckets"] == derived


def test_live_brain_source_pins_the_learners_construction_and_the_two_intents():
    source = Path(inspect.getsourcefile(live_brain)).read_text(encoding="utf-8")
    assert re.search(
        r"SpikingInteractionLearner\(\s*list\(_IXN_INTENTS\),\s*n_in=52,\s*seed=seed,\s*t_post=6\s*\)",
        source,
    )
    block = re.search(r"_IXN_INTENTS = \((.*?)\)", source, re.S).group(1)
    names = re.findall(r'"([A-Z]+)"', block)
    assert names == list(live_brain._IXN_INTENTS) and len(names) == 9
    assert (im.N_IN, im.T_POST) == (52, 6)
    assert im.GREET in names and im.GOAL in names and im.GREET != im.GOAL
    unit = im._incumbent(0)
    assert unit.responses == names and (unit.n_in, unit.t_post) == (52, 6.0)


def test_the_generator_is_the_evidence_texts_and_the_plans_labels():
    assert im.TEXTS == {
        (0, 0): "cat dog sun sky",
        (0, 1): "cat dog sky sun",
        (1, 0): "dog cat sun sky",
        (1, 1): "dog cat sky sun",
    }
    # u = "dog precedes cat", v = "sky precedes sun", read off the texts themselves
    for (u, v), text in im.TEXTS.items():
        words = text.split()
        assert u == (words.index("dog") < words.index("cat"))
        assert v == (words.index("sky") < words.index("sun"))
    # the plan's A/B: 00 -> A, 01 -> B, 10 -> B, 11 -> A, as GREET / GOAL
    assert [im.label_xor(*uv) for uv in im.UV] == ["GREET", "GOAL", "GOAL", "GREET"]
    # the twin: GREET iff "cat" precedes "dog"
    assert [im.label_twin(*uv) for uv in im.UV] == ["GREET", "GREET", "GOAL", "GOAL"]
    for (u, v), text in im.TEXTS.items():
        words = text.split()
        assert im.label_twin(u, v) == (
            "GREET" if words.index("cat") < words.index("dog") else "GOAL"
        )
    prose = EVIDENCE.read_text(encoding="utf-8")
    assert '"cat dog sun sky", "cat dog sky sun", "dog cat sun sky", "dog cat sky sun"' in prose
    assert (im.STABLE, im.RANDOM_SETTINGS, im.RANDOM_SETTINGS_SEED) == (10, 2000, 20261003)


def test_code_and_ranks_match_an_independent_derivation():
    for text in im.TEXTS.values():
        assert im.ranks(text) == independent_ranks(text)
        assert im.code(text) == independent_code(text)
        # the four words hold ranks 0-3; ranks 4 and 5 go to the empty buckets 0 and 1
        ranks = im.ranks(text)
        assert sorted(ranks[b] for b in im.BUCKETS_52.values()) == [0, 1, 2, 3]
        assert (ranks[0], ranks[1]) == (4, 5)


def test_code_and_ranks_go_through_the_learners_own_methods(monkeypatch):
    text = im.TEXTS[0, 0]
    unit = im._probe()
    monkeypatch.setattr(type(unit), "_psp", lambda self, dt: 7.0)
    assert im.code(text) == [7.0] * 52
    monkeypatch.setattr(il, "rank_code", lambda x: [0] * len(x))
    assert im.ranks(text) == [0] * 52


# ---- A1 .. A5 and their reds -------------------------------------------------------------------
def test_a1_holds_and_each_of_its_reds_fails_its_own_clause():
    a1 = im.check_a1()
    assert a1["ok"] and all(a1["conditions"].values())
    dup = im.check_a1(texts={**im.TEXTS, (1, 1): "cat dog sun sun"})
    assert not dup["ok"] and not dup["conditions"]["texts_are_the_words_once"]
    narrow = im.check_a1(n_in=3)
    assert not narrow["ok"] and not narrow["conditions"]["plans_buckets"]
    assert not narrow["conditions"]["distinct_buckets"]  # four words cannot fit three buckets
    swapped = im.check_a1(buckets={**im.BUCKETS_52, "cat": 41, "dog": 28})
    assert not swapped["ok"] and not swapped["conditions"]["plans_buckets"]
    assert swapped["conditions"]["distinct_buckets"]  # only the asserted numbers are wrong


def test_a2_the_codes_are_additive_exactly_on_ranks_and_within_1e_12_in_float():
    a2 = im.check_a2()
    assert a2["ok"] and a2["max_rank_deviation"] == 0
    assert a2["max_code_deviation"] <= 1e-12  # observed 1.11e-16, the plan's 1.1e-16
    # independently: the sums on the integer ranks, and the sums of the code compared as sums
    r = {uv: independent_ranks(t) for uv, t in im.TEXTS.items()}
    p = {uv: independent_code(t) for uv, t in im.TEXTS.items()}
    assert [a + b for a, b in zip(r[0, 0], r[1, 1], strict=True)] == [
        a + b for a, b in zip(r[0, 1], r[1, 0], strict=True)
    ]
    assert [a + b for a, b in zip(p[0, 0], p[1, 1], strict=True)] == [
        a + b for a, b in zip(p[0, 1], p[1, 0], strict=True)
    ]
    red = im.check_a2(texts={**im.TEXTS, (1, 1): "dog sky cat sun"})
    assert (
        not red["ok"] and not red["conditions"]["ranks_additive"] and red["max_rank_deviation"] > 0
    )
    # identical codes would be additive vacuously; the check refuses them
    same = im.check_a2(texts=dict.fromkeys(im.UV, im.TEXTS[0, 0]))
    assert not same["ok"] and same["conditions"]["ranks_additive"]
    assert not same["conditions"]["codes_distinct"]


def test_a3_the_incumbent_never_reads_xor_four_of_four_and_the_counter_can_count():
    a3 = im.check_a3(DEV, im.TEACH_EPOCHS)
    assert a3["ok"] and a3["max_correct"] <= 3 and a3["random_4_of_4"] == 0, a3
    assert a3["random_settings"] == 2000 and a3["presentations"] == len(DEV) * im.TEACH_EPOCHS * 4
    # the counter's positive controls: the twin through the XOR slot reaches 4 of 4 on this loop ...
    red = im.check_a3(DEV[:1], im.TEACH_EPOCHS, labelling=im.label_twin, settings=0)
    assert not red["ok"] and red["max_correct"] == 4 and not red["conditions"]["never_4_of_4"]
    # ... and the random-settings counter counts a 4 of 4: the first setting's reads as a labelling
    rng = random.Random(im.RANDOM_SETTINGS_SEED)
    unit = im._incumbent(0)
    for neuron in unit.neurons:
        neuron.w = [rng.uniform(-1.0, 1.0) for _ in range(52)]
    reads = [unit.route_text(im.TEXTS[uv]) for uv in im.UV]
    assert im._random_hits(1, lambda u, v: reads[im.UV.index((u, v))]) == 1


def test_a4_the_incumbent_learns_the_twin_and_the_witness_reads_it():
    a4 = im.check_a4(DEV, im.TEACH_EPOCHS)
    assert a4["ok"], a4
    for row in a4["by_seed"].values():
        assert row["tail"] == [4] * im.STABLE and row["bystanders"] == 0
    assert (a4["witness_on_slot"], a4["witness_on_xor"]) == (4, 2)
    assert {s: r["first_step_4of4"] for s, r in a4["by_seed"].items()} == {
        7100: 129,
        7101: 106,
        7102: 112,
        7103: 84,
    }
    with pytest.raises(ValueError, match="at least 10"):
        im.check_a4(DEV, im.STABLE - 1)


def test_a4_reds_each_fail_the_stable_clause():
    for options in ({"lr": 0.0}, {"shuffled": True}, {"labelling": im.label_xor}):
        red = im.check_a4(DEV[:2], im.TEACH_EPOCHS, **options)
        assert not red["ok"] and not red["conditions"]["stable_on_every_seed"], options
    xor = im.check_a4(DEV[:1], im.TEACH_EPOCHS, labelling=im.label_xor)
    assert not xor["conditions"]["witness_4_of_4"] and xor["witness_on_slot"] == 2


def test_the_shuffled_red_never_teaches_the_true_mapping():
    taught = im._shuffled(7100, im.label_twin)
    for order in im._schedule(7100, 300):
        true = [im.label_twin(*uv) for uv in order]
        labels = taught(order)
        assert labels != true and sorted(labels) == sorted(true)


def test_the_witness_weights_through_the_real_route_text():
    unit = im._witness_unit()
    cat, dog = zlib.crc32(b"cat") % 52, zlib.crc32(b"dog") % 52
    for name, neuron in zip(unit.responses, unit.neurons, strict=True):
        expected = [0.0] * 52
        if name == "GREET":
            expected[cat], expected[dog] = 1.0, -1.0
        if name == "GOAL":
            expected[cat], expected[dog] = -1.0, 1.0
        assert neuron.w == expected
    twin = [unit.route_text(im.TEXTS[uv]) for uv in im.UV]
    assert twin == [im.label_twin(*uv) for uv in im.UV]
    assert sum(r == im.label_xor(*uv) for r, uv in zip(twin, im.UV, strict=True)) == 2
    assert im.witness_reads() == (4, 2)


def test_a5_two_threshold_units_separate_xor_and_one_cannot():
    a5 = im.check_a5()
    assert a5["ok"] and a5["right"] == 4 and a5["min_margin"] > 0.03
    w = [0.0] * 52
    for word, sign in (("dog", 1.0), ("sky", 1.0), ("cat", -1.0), ("sun", -1.0)):
        w[zlib.crc32(word.encode("ascii")) % 52] = sign
    s = {
        uv: sum(a * b for a, b in zip(w, independent_code(t), strict=True))
        for uv, t in im.TEXTS.items()
    }
    ordered = sorted(s.values())
    midpoints = ((a + b) / 2 for a, b in zip(ordered, ordered[1:], strict=False))
    thresholds = [ordered[0] - 1.0, *midpoints, ordered[-1] + 1.0]
    for t in thresholds:  # no single unit, in either sense, answers XOR
        for sign in (1.0, -1.0):
            single = {uv: "GREET" if sign * (s[uv] - t) > 0 else "GOAL" for uv in im.UV}
            assert sum(single[uv] == im.label_xor(*uv) for uv in im.UV) < 4


# ---- the construction check, the reds, the development measurement ---------------------------
def test_the_construction_check_passes_on_the_development_seeds():
    result = im.construction_check(check_seeds=DEV, epochs=im.TEACH_EPOCHS)
    assert result["verdict"] is True and result["seeds"] == DEV and result["epochs"] == 400
    assert all(result[key]["ok"] for key in ("a1", "a2", "a3", "a4", "a5"))
    assert result["a3"]["max_correct"] <= 3 and result["a3"]["random_4_of_4"] == 0
    # the verdict is the conjunction of A1-A4 and not of A5 (witness only)
    broken = {**result, "a4": {**result["a4"], "ok": False}}
    assert not all(broken[key]["ok"] for key in ("a1", "a2", "a3", "a4"))


def test_every_construction_red_fires():
    reds = im.construction_reds()
    assert set(reds["fired"]) == {
        "a1_duplicate_word_text",
        "a1_n_in_3",
        "a1_swapped_buckets",
        "a2_text_11_replaced",
        "a3_twin_through_the_xor_counter",
        "a4_lr_zero",
        "a4_shuffled_labels",
        "a4_xor_in_the_twin_slot",
    }
    assert reds["all_fired"] and all(reds["fired"].values()) and reds["a3_red_max_correct"] == 4
    assert (
        im._fires({"ok": True, "conditions": {"c": True}}, "c") is False
    )  # a green check is no red
    assert (
        im._fires({"ok": False, "conditions": {"c": True}}, "c") is False
    )  # failing, wrong clause


def test_the_development_measurement_is_recorded_and_sets_teach_epochs():
    measured = {s: im.epochs_to_stable(s, im.label_twin, max_epochs=4000) for s in DEV}
    assert measured == DEV_EPOCHS_TO_STABLE
    # g1c 2.8 amendment (pre-data): TEACH_EPOCHS reads the stronger stability, the last epoch
    # end that is not 4 of 4 within 4000 epochs -- 179, 170, 162, 161 -- so 2 * 179 = 358 -> 400
    last_unstable = {}
    for s in DEV:
        ends = im._run(s, im.label_twin, 4000, every_step=False)["epoch_ends"]
        last_unstable[s] = max((i + 1 for i, c in enumerate(ends) if c != 4), default=0)
    assert last_unstable == {7100: 179, 7101: 170, 7102: 162, 7103: 161}
    assert im.TEACH_EPOCHS == -(-2 * max(last_unstable.values()) // 50) * 50 == 400
    # the stable run is the 10 epoch ends e + 1 .. e + 10: one epoch short of it, nothing is found
    e = measured[7100]
    assert im.epochs_to_stable(7100, im.label_twin, e + im.STABLE - 1) is None
    assert im.epochs_to_stable(7100, im.label_twin, e + im.STABLE) == e
    # XOR never stabilizes (the theorem), so the answer is None, not a number
    assert im.epochs_to_stable(7100, im.label_xor, 120) is None
    # the full trace never left 4 of 4 after the recorded last failing epoch end (179 on seed 7100)
    run = im._run(7100, im.label_twin, 400, every_step=False)
    assert all(c == 4 for c in run["epoch_ends"][179:]) and run["epoch_ends"][178] != 4


def test_teaching_goes_through_learn_turn_on_one_shared_schedule(monkeypatch):
    rng = random.Random(7100)
    assert list(im._schedule(7100, 3)) == [rng.sample(im.UV, 4) for _ in range(3)]
    calls = []
    original = il.SpikingInteractionLearner.learn_turn

    def spy(self, utterance, firewall_intent, feedback=None):
        calls.append((utterance, firewall_intent, feedback))
        return original(self, utterance, firewall_intent, feedback)

    monkeypatch.setattr(il.SpikingInteractionLearner, "learn_turn", spy)
    im._run(7100, im.label_twin, 2)
    order = list(im._schedule(7100, 2))
    assert calls == [(im.TEXTS[uv], None, im.label_twin(*uv)) for epoch in order for uv in epoch]
    # A3's and A4's runs consume the same schedule: the same presentations, whatever the labelling
    calls.clear()
    im._run(7100, im.label_xor, 2)
    assert [c[0] for c in calls] == [im.TEXTS[uv] for epoch in order for uv in epoch]


def test_the_defaults_are_the_check_seeds_and_the_reds_use_a_development_seed(monkeypatch):
    seeds, epochs = im._resolve(None, None)  # resolves the campaign's defaults; runs nothing
    assert seeds == ccc.G2II_CHECK_SEEDS == tuple(range(7000, 7008)) and epochs == 400
    assert not set(seeds) & set(DEV)
    assert im._resolve(DEV, 50) == (DEV, 50)
    assert im._dev_seed() == 7100 and im._dev_seed() in ccc.G2II_DEV_SEEDS
    with pytest.raises(ValueError, match="at least 10"):
        im._resolve(DEV, 9)
    monkeypatch.setattr(im, "TEACH_EPOCHS", None)  # an unset schedule is refused, never defaulted
    with pytest.raises(ValueError, match="TEACH_EPOCHS is unset"):
        im._epochs(None)
    assert inspect.signature(im.construction_check).parameters["check_seeds"].default is None


def test_no_function_name_carries_a_learning_verb():
    tree = ast.parse(Path(inspect.getsourcefile(im)).read_text(encoding="utf-8"))
    names = [
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    assert names and not [n for n in names if any(v in n.lower() for v in FORBIDDEN)]


def test_importing_the_module_does_not_import_the_campaign_module():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import snn_data_lake.instruments.inseparable_mapping; "
            "print('snn_data_lake.instruments.carried_state_campaign' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# ---- the stream -------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def streams():
    return im.stream(7100, im.label_xor), im.stream(7100, im.label_twin)


def test_the_stream_schema_is_rb6s(streams):
    xor, twin = streams
    base = md.rb6_stream(4242)
    for stream, name in ((xor, "xor"), (twin, "twin")):
        assert list(stream) == list(base)
        assert stream["seed"] == 7100
        assert stream["synthetic"] == {
            "family": "g2ii",
            "labelling": name,
            "intents": ["GREET", "GOAL"],
            "feed": "psp52_pad64_norm1.73",
        }
        assert stream["config"] == {**base["config"], "seed": 7100}
        assert set(stream["policy"]) == set(base["policy"]) and stream["policy"]["threads"] == 1
        assert set(stream["final"]) == {"head", "dirty"}
        events, sleep = stream["events"], stream["sleep"]
        assert len(events) == len(sleep) == 1200 == md.RB6_EVENTS
        assert (im.STREAM_WARM_UP, im.STREAM_NORM, im.STREAM_EVENTS) == (
            md.RB6_WARM_UP,
            md.RB6_NORM,
            1200,
        )
        assert all(set(e) == set(base["events"][0]) for e in events)  # no "text" key rides along
        assert all(set(s) == set(base["sleep"][0]) for s in sleep)
        for i, (event, tick) in enumerate(zip(events, sleep, strict=True)):
            assert (event["t"], event["tick"]) == (i + 1, 2 * i + 1)
            assert (tick["tick"], tick["brain_tick"], tick["event"]) == (i + 1, 2 * i + 2, i + 1)
            assert event["causal_lp"] is None and tick["live_lp"] is None
            assert event["segment"] == ("warm-up" if i < 256 else "unseen-order")
            assert len(event["x"]) == 64 and event["c"] in (0, 1)
        json.dumps(stream)  # serializable, and the replay's source digest reads it
        assert isinstance(md._source_digest(stream), int)


def test_the_stream_is_rederived_here_from_the_rng_and_an_independent_code(streams):
    xor, twin = streams
    rng = random.Random(7100)
    order = [rng.randrange(4) for _ in range(1201)]  # order[0] is the virtual text_0
    refs = []
    for uv in im.UV:
        c = independent_code(im.TEXTS[uv])
        scale = 1.73 / math.sqrt(sum(v * v for v in c))
        refs.append([v * scale for v in c] + [0.0] * 12)
    for stream, label in ((xor, im.label_xor), (twin, im.label_twin)):
        for i, event in enumerate(stream["events"]):
            assert (
                max(abs(a - b) for a, b in zip(event["x"], refs[order[i + 1]], strict=True)) < 1e-12
            )
            assert event["c"] == ["GREET", "GOAL"].index(label(*im.UV[order[i]]))
    # the XOR and twin streams share every x byte but not every c
    assert json.dumps([e["x"] for e in xor["events"]]) == json.dumps(
        [e["x"] for e in twin["events"]]
    )
    assert [e["c"] for e in xor["events"]] != [e["c"] for e in twin["events"]]
    assert im.stream(7100, im.label_xor)["events"] == xor["events"]  # deterministic
    assert im.stream(7101, im.label_xor)["events"] != xor["events"]  # and seed-dependent


def test_stream_checks_hold_and_each_mutant_fails_its_clause(streams):
    xor, twin = streams
    green = im.stream_checks(7100, xor=xor, twin=twin)
    assert green["ok"] and all(green["checks"].values()) and green["events"] == 1200
    assert 0.0 < green["repeat_rate"] < 1.0 and sum(green["text_counts"].values()) == 1200

    def red(clause, mutate):
        bad = copy.deepcopy(xor)
        mutate(bad)
        result = im.stream_checks(7100, xor=bad, twin=twin)
        assert not result["ok"] and not result["checks"][clause], clause

    red("x_norm_1_73", lambda s: s["events"][5].update(x=[v * 1.0001 for v in s["events"][5]["x"]]))
    red("x_zero_outside_dims_0_51", lambda s: s["events"][9]["x"].__setitem__(60, 0.1))
    red("x_is_a_text_code", lambda s: s["events"][3].update(x=[0.0] * 64))
    red(
        "c_is_the_previous_texts_xor_label",
        lambda s: s["events"][8].update(c=1 - s["events"][8]["c"]),
    )
    red("exactly_1200_events", lambda s: s["events"].pop())
    red(
        "xor_twin_x_identical",
        lambda s: s["events"][2].update(x=list(reversed(s["events"][2]["x"]))),
    )
    # one text starved: the frequency clause (every x replaced by one text's)
    red("texts_within_4_sd", lambda s: [e.update(x=list(s["events"][0]["x"])) for e in s["events"]])


def test_the_stream_reds_all_fire():
    reds = im.stream_reds(7100)
    assert set(reds["fired"]) == {
        "lag_0",
        "events_256",
        "permutation_blocks_of_4",
        "twin_from_another_rng_state",
    }
    assert reds["all_fired"]
    # the permutation-block red is caught by the repeat rate ONLY: its text counts are exactly 25 %
    frame = im._frame(7100)
    block = im._build_stream(7100, im.label_xor, frame=frame, blocks=True)
    result = im.stream_checks(
        7100, xor=block, twin=im._build_stream(7100, im.label_twin, frame=frame, blocks=True)
    )
    assert result["checks"]["texts_within_4_sd"] and not result["checks"]["repeat_rate_within_4_sd"]
    assert result["repeat_rate"] < 0.1


def test_the_stream_replays_and_reads_the_twin_but_not_xor_through_arm_a(streams):
    xor, twin = streams
    accuracy = {}
    for name, stream in (("xor", xor), ("twin", twin)):
        run = md.replay(stream, "a")
        count = len(stream["events"])
        scored = md.score(stream, run["readout"]["predictions"], [False] * count, floors=False)
        accuracy[name] = scored["unseen-order"]["accuracy"]
    # g1c 2.8: arm (a) on XOR stays below the affine ceiling 0.75 + 4 binomial sd; it reads the twin
    assert accuracy["xor"] < 0.75 + 4 * math.sqrt(0.25 * 0.75 / 944)
    assert accuracy["twin"] > 0.95


# ---- the self-check ---------------------------------------------------------------------------
def test_main_is_fast_and_cwd_independent(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    started = time.perf_counter()
    im.main()
    assert time.perf_counter() - started < 20.0
    printed = capsys.readouterr().out
    assert printed.startswith("self-check OK (smoke on development seed 7100") and printed.isascii()
    assert "8 of 8 construction reds fired and 4 of 4 stream reds fired" in printed
