"""Tests that reasoning/blimp_bench.py really CONSUMES ar_spiking_lm_serve.SpikingLMServe.

The wiring exists (blimp_bench.SERVES -> frozen_serves() -> cached_logprob -> score). What is
easy to lose and hard to notice is the CHARGE: an importlib table can keep resolving, a serve can
keep being constructed, and the printed number can stop depending on anything the model computed.
Policy 3's honesty guard is that reachability is not charge, so these tests do not assert that the
module is imported -- they assert that swapping what the served model returns SWAPS THE REPORTED
NUMBER, and that with no checkpoint the bench prints a non-claim instead of a fabricated score.

Deliberately NOT asserted: any particular BLiMP accuracy for the real artifact. The length-alone
control sits at 29.4% on the full set and the honest char-level bar is 53.0%; pinning a number here
would turn a measurement into a promise. The real-artifact test asserts only that the model's output
moves the score, never which way.

Run: pytest tests/test_blimp_bench_serve.py
"""

import json
import os

import pytest

from snn_data_lake.reasoning import blimp_bench as bb

# Six synthetic paradigms of EQUAL-LENGTH minimal pairs. Equal length is the point: the length
# control ties on every pair, so any accuracy a stub model earns here came from the stub, not from
# the surface leak the real bench has to keep reporting.
_PAIRS = [
    ("the cat eats fish", "the cat eatz fish"),
    ("a dog runs fast!", "a dog runz fast!"),
    ("she sells shells", "she zells shells"),
    ("they read a book", "they raed a book"),
    ("we sang the song", "we snag the song"),
    ("he drinks water!", "he drinkz water!"),
]


@pytest.fixture
def data_root(tmp_path):
    """A miniature BLiMP tree on disk, in the real jsonl schema load_pairs() parses."""
    for i in range(6):
        p = tmp_path / f"synthetic_paradigm_{i}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for good, bad in _PAIRS:
                f.write(json.dumps({"sentence_good": good, "sentence_bad": bad}) + "\n")
    return str(tmp_path)


class _StubServe:
    """A serve with SpikingLMServe's contract: available() + logprob() -> float or None.

    Records every sentence it is asked about, so a test can prove it was reached rather than
    assuming it from an import edge."""

    def __init__(self, prefer_good=True, ok=True):
        self.prefer_good = bool(prefer_good)
        self.ok = bool(ok)
        self.seen = []
        self._goods = {g for g, _b in _PAIRS}

    def available(self):
        return self.ok

    def logprob(self, sentence):
        self.seen.append(sentence)
        if not self.ok:
            return None
        hit = sentence in self._goods
        if not self.prefer_good:
            hit = not hit
        # Per-character magnitudes so the per-char scoring is a real rescoring, not a copy.
        return -1.0 * len(sentence) if hit else -3.0 * len(sentence)


# --------------------------------------------------------------------------------------------
# the wire exists
# --------------------------------------------------------------------------------------------
def test_the_ar_spiking_lm_serve_row_is_in_the_scorer_table():
    """SERVES must still name the module+class. The table is resolved by importlib, so a typo here
    degrades to a silent 'n/a' row rather than an ImportError -- which is exactly the failure mode
    that once left this model never wired in at all."""
    entries = {mod: cls for _label, mod, cls in bb.SERVES}
    assert entries.get("ar_spiking_lm_serve") == "SpikingLMServe", (
        "blimp_bench.SERVES no longer names ar_spiking_lm_serve.SpikingLMServe"
    )


def test_frozen_serves_resolves_the_ar_row_to_a_real_object():
    """The dynamic import must actually resolve. Unavailable is fine (models/ is gitignored);
    None is not -- None means the module or class could not be imported at all."""
    rows = dict(bb.frozen_serves())
    label = bb.SERVES[0][0]
    assert label in rows, f"frozen_serves() dropped the {label} row"
    serve = rows[label]
    assert serve is not None, (
        "frozen_serves() could not import ar_spiking_lm_serve.SpikingLMServe -- the bench would "
        "print n/a for a model that is present"
    )
    assert hasattr(serve, "logprob") and hasattr(serve, "available")


# --------------------------------------------------------------------------------------------
# the wire is CHARGED: what the serve returns decides the reported number
# --------------------------------------------------------------------------------------------
def test_the_serve_is_actually_invoked_for_every_sentence(data_root):
    stub = _StubServe()
    bb.score(bb.cached_logprob(stub), root=data_root)
    assert stub.seen, "score() never called the serve"
    asked = set(stub.seen)
    for good, bad in _PAIRS:
        assert good in asked and bad in asked, "a minimal pair was scored without asking the model"


def test_the_serve_result_decides_the_score_under_both_scorings(data_root):
    """The charge proof. Same harness, same pairs, same code path -- only the model's opinion
    differs, and both reported numbers move with it."""
    good_model = _StubServe(prefer_good=True)
    bad_model = _StubServe(prefer_good=False)

    lp_good, lp_bad = bb.cached_logprob(good_model), bb.cached_logprob(bad_model)
    sum_good = bb.score(lp_good, root=data_root)["accuracy"]
    sum_bad = bb.score(lp_bad, root=data_root)["accuracy"]
    pc_good = bb.score(bb.per_char(lp_good), root=data_root)["accuracy"]
    pc_bad = bb.score(bb.per_char(lp_bad), root=data_root)["accuracy"]

    assert sum_good == 1.0 and sum_bad == 0.0, (
        f"sum-logprob did not follow the model (got {sum_good}, {sum_bad})"
    )
    assert pc_good == 1.0 and pc_bad == 0.0, (
        f"per-char did not follow the model (got {pc_good}, {pc_bad})"
    )


def test_main_prints_the_serve_backed_row_and_the_length_control(data_root, monkeypatch, capsys):
    """End to end through main(): the stub's number reaches the printed table, and the length-alone
    control is printed in the same table so the length prior cannot be mistaken for language."""
    stub = _StubServe()
    monkeypatch.setattr(bb, "DATA", data_root)
    monkeypatch.setattr(bb, "frozen_serves", lambda: [("AR spik-RWKV", stub)])
    bb.main()
    out = capsys.readouterr().out

    assert stub.seen, "main() never reached the served model"
    assert "control length" in out, "main() stopped printing the length-alone control"
    assert "length only" in out, "the results table lost its length-only row"
    assert "AR spik-RWKV" in out and "100.0%" in out, (
        "the served model's score did not reach the printed table"
    )


def test_a_wrong_model_prints_a_different_table(data_root, monkeypatch, capsys):
    """Two models that disagree must produce two different printed tables. If the output is
    byte-identical, the serve is decorative no matter how thoroughly it was imported."""
    tables = []
    for prefer in (True, False):
        monkeypatch.setattr(bb, "DATA", data_root)
        monkeypatch.setattr(bb, "frozen_serves", lambda p=prefer: [("AR spik-RWKV", _StubServe(p))])
        bb.main()
        tables.append([ln for ln in capsys.readouterr().out.splitlines() if "AR spik-RWKV" in ln])
    assert tables[0] and tables[1]
    assert tables[0] != tables[1], "the printed table did not change when the model's opinion did"


# --------------------------------------------------------------------------------------------
# honest degradation: no checkpoint -> a non-claim, never a fabricated number
# --------------------------------------------------------------------------------------------
def test_no_checkpoint_prints_na_and_scores_nothing(data_root, monkeypatch, capsys):
    stub = _StubServe(ok=False)
    monkeypatch.setattr(bb, "DATA", data_root)
    monkeypatch.setattr(bb, "frozen_serves", lambda: [("AR spik-RWKV", stub)])
    bb.main()
    out = capsys.readouterr().out
    assert "n/a" in out and "no frozen artifact" in out, (
        "an unavailable model must print an explicit non-claim"
    )
    assert not stub.seen, "an unavailable model was scored anyway -- that number is fabricated"


def test_an_unscorable_sentence_never_wins_on_length(data_root):
    """The sentinel must survive per-char normalization UNNORMALIZED.

    UNSCORABLE/len(s) is larger for a longer sentence, so a pair the model failed on twice would be
    decided by length -- credit for having no opinion, which score() forbids. Both scorings must
    report 0% here, matching constant_scorer."""
    dead = _StubServe(ok=False)
    lp = bb.cached_logprob(dead)
    assert bb.score(lp, root=data_root)["accuracy"] == 0.0
    assert bb.score(bb.per_char(lp), root=data_root)["accuracy"] == 0.0, (
        "per-char turned an unscorable model into a length heuristic"
    )


def test_length_alone_is_reported_as_a_control_not_a_model():
    """The number a reader has to hold next to any model row. On the real 67x25 set the length-only
    scorer lands at 29.4%; the assertion here is only that the control still exists and still
    expresses pure surface length, since pinning 29.4% would pin the dataset too."""
    assert bb.length_scorer("abc") > bb.length_scorer("abcd")
    # normalized, the length heuristic collapses to a constant -- the 'const' the table prints
    norm = bb.per_char(bb.length_scorer)
    assert norm("abc") == norm("a much longer sentence here")


# --------------------------------------------------------------------------------------------
# the real artifact, when one is on disk
# --------------------------------------------------------------------------------------------
def test_real_spiking_lm_serve_charges_the_score_when_present():
    """Neuter-A/B on the real checkpoint: same slice, same code path, model output removed.

    Skipped (not failed) with no artifact -- models/ is gitignored, so a fresh clone legitimately
    has nothing to score. Asserts only that the number MOVES; which way is a measurement, and this
    file makes no claim about it."""
    from snn_data_lake.reasoning.ar_spiking_lm_serve import SpikingLMServe

    serve = SpikingLMServe()
    if not serve.available():
        pytest.skip("no frozen AR-LM artifact at " + os.path.basename(serve.path))

    names = bb.paradigms()
    if not names:
        pytest.skip("BLiMP data not downloaded")
    slice_ = names[:3]

    real = bb.score(bb.cached_logprob(serve), per_paradigm=4, only=slice_)["accuracy"]

    original = SpikingLMServe.logprob
    try:
        SpikingLMServe.logprob = lambda self, sentence: None
        neutered = bb.score(bb.cached_logprob(serve), per_paradigm=4, only=slice_)["accuracy"]
    finally:
        SpikingLMServe.logprob = original

    assert real != neutered, (
        "removing the served model's output left the BLiMP score unchanged -- the serve is not "
        "charging this bench"
    )
