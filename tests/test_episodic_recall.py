"""Tests for hippocampus.episodic_recall.EpisodicRecall.

Covers the pre-existing public API (encode/complete/gist/narrate) and the v3.79 GENERALIZATION
addition (apply_schema): schema-driven transfer to situations that were NEVER stored, plus the
honest-failure paths that must refuse to guess -- an ungrounded situation, a family too small,
a family too variable to abstract anything, and an ambiguous cue spanning two conflicting
families. The last one is the case where the prototype-margin test alone was measured to
fabricate a winner, so it is asserted explicitly.
"""

import zlib

from snn_data_lake.hippocampus.episodic_recall import (
    DEFAULT_SLOTS,
    EpisodicRecall,
    _generalization_check,
    main,
)

INVARIANTS = {"who": "self", "where": "kitchen", "when": "morning"}
FOODS = ("toast", "coffee", "eggs", "cereal", "fruit", "bagel", "porridge", "waffle")
FEELS = ("calm", "tired", "happy", "rushed", "sleepy", "content", "warm", "bright")


def _family_store(n=8, seed=5, distractors=12):
    """A coherent family (who/where/when invariant, what/affect varying) among distractors."""
    er = EpisodicRecall(seed=seed)
    for i in range(n):
        er.encode({**INVARIANTS, "what": FOODS[i], "affect": FEELS[i]})
    for i in range(distractors):
        er.encode(
            {"who": f"w{i}", "what": f"x{i}", "where": f"p{i}", "when": f"q{i}", "affect": f"a{i}"}
        )
    return er


def _episode_key(episode):
    """Reproduce EpisodicRecall.encode's crc32 key so a test can prove an episode is UNSTORED."""
    raw = "|".join(f"{s}={episode[s]}" for s in sorted(episode)).encode("ascii")
    return zlib.crc32(raw)


# --------------------------------------------------------------------- existing API ----------
def test_encode_returns_stable_key_and_grows_the_store():
    er = EpisodicRecall(seed=0)
    k1 = er.encode({"who": "self", "what": "toast", "where": "kitchen"})
    assert len(er) == 1
    er2 = EpisodicRecall(seed=0)
    k2 = er2.encode({"where": "kitchen", "what": "toast", "who": "self"})  # key order irrelevant
    assert k1 == k2


def test_complete_reconstructs_held_out_slots_from_a_partial_cue():
    er = EpisodicRecall(seed=1)
    episodes = [
        {
            "who": f"agent{i}",
            "what": f"grasp{i}",
            "where": f"room{i}",
            "when": f"t{i}",
            "affect": f"mood{i}",
        }
        for i in range(12)
    ]
    for e in episodes:
        er.encode(e)
    for e in episodes:
        recon = er.complete({"where": e["where"], "when": e["when"]})["episode"]
        for held in ("who", "what", "affect"):
            assert recon[held] == e[held]


def test_complete_on_an_empty_store_is_honest():
    er = EpisodicRecall(seed=0)
    out = er.complete({"where": "kitchen"})
    assert out == {"episode": {}, "confidence": 0.0, "key": None, "slot_conf": {}}


def test_gist_keeps_invariant_slots_and_drops_varying_ones():
    er = EpisodicRecall(seed=2)
    for i, f in enumerate(FOODS[:6]):
        er.encode(
            {"who": "self", "what": f, "where": "kitchen", "when": f"day{i}", "affect": FEELS[i]}
        )
    g = er.gist({"where": "kitchen"}, m=6)
    assert set(g["schema"]) == {"who", "where"}
    assert g["n"] == 6
    assert min(g["margins"][s] for s in ("who", "where")) > max(
        g["margins"][s] for s in ("what", "when", "affect")
    )


def test_gist_of_a_single_episode_is_empty():
    er = EpisodicRecall(seed=4)
    er.encode({"who": "self", "what": "toast", "where": "kitchen"})
    g = er.gist({"where": "kitchen"})
    assert g["schema"] == {} and g["n"] == 1


def test_narrate_is_ascii_and_fills_the_frame():
    s = EpisodicRecall.narrate(
        {"who": "self", "what": "grasped a cup", "where": "kitchen", "affect": "calm"}
    )
    assert s.isascii() and s.endswith(".") and "kitchen" in s and "calm" in s


# ------------------------------------------------------------- apply_schema: transfer ---------
def test_apply_schema_predicts_an_unstored_situation():
    er = _family_store()
    target = {**INVARIANTS, "what": "pancakes"}
    stored = {ep["key"] for ep in er._episodes}
    assert _episode_key(target) not in stored, "the probed episode must never have been stored"

    out = er.apply_schema({"where": "kitchen", "what": "pancakes"})
    assert out["prediction"] == {"who": "self", "when": "morning"}
    assert out["n_family"] == 8
    assert out["confidence"] > 0.0
    assert "affect" not in out["prediction"], "a VARYING slot must never be predicted"


def test_apply_schema_result_shape():
    out = _family_store().apply_schema({"where": "kitchen", "what": "pancakes"})
    assert set(out) == {"prediction", "schema", "confidence", "n_family"}
    assert isinstance(out["confidence"], float) and isinstance(out["n_family"], int)


def test_apply_schema_generalizes_across_many_unseen_probes():
    er = _family_store()
    novel = ("pancakes", "omelette", "granola", "muffin", "scone", "crepe", "grits", "congee")
    hits = 0
    for w in novel:
        assert _episode_key({**INVARIANTS, "what": w}) not in {ep["key"] for ep in er._episodes}
        hits += er.apply_schema({"where": "kitchen", "what": w})["prediction"] == {
            "who": "self",
            "when": "morning",
        }
    assert hits == len(novel)


def test_apply_schema_works_from_a_different_grounded_slot():
    """Transfer must not depend on which invariant slot the situation happens to supply."""
    er = _family_store()
    out = er.apply_schema({"who": "self", "what": "pancakes"})
    assert out["prediction"] == {"where": "kitchen", "when": "morning"}


# ------------------------------------------------------ apply_schema: honest failures ---------
def test_apply_schema_refuses_an_ungrounded_situation():
    """No cue value was ever stored -> no family at all. Must NOT fall back on a nearest match."""
    out = _family_store().apply_schema({"where": "nowhere_at_all", "what": "unseen_thing"})
    assert out == {"prediction": {}, "schema": {}, "confidence": 0.0, "n_family": 0}


def test_apply_schema_refuses_an_empty_store():
    er = EpisodicRecall(seed=0)
    out = er.apply_schema({"where": "kitchen"})
    assert out["prediction"] == {} and out["confidence"] == 0.0 and out["n_family"] == 0


def test_apply_schema_refuses_an_empty_situation():
    out = _family_store().apply_schema({})
    assert out["prediction"] == {} and out["confidence"] == 0.0 and out["n_family"] == 0


def test_apply_schema_refuses_a_family_of_one():
    """A single episode cannot separate typical from idiosyncratic -> no schema, no prediction."""
    er = EpisodicRecall(seed=8)
    er.encode({**INVARIANTS, "what": "toast", "affect": "calm"})
    out = er.apply_schema({"where": "kitchen", "what": "pancakes"})
    assert out["prediction"] == {} and out["confidence"] == 0.0 and out["n_family"] == 1


def test_apply_schema_min_family_cannot_be_lowered_below_two():
    er = EpisodicRecall(seed=8)
    er.encode({**INVARIANTS, "what": "toast", "affect": "calm"})
    assert er.apply_schema({"where": "kitchen"}, min_family=1)["confidence"] == 0.0


def test_apply_schema_makes_no_false_abstraction_from_a_variable_family():
    """8 episodes sharing ONLY the cue slot: the family is found, but there is nothing to
    abstract beyond what the cue already supplied -> empty prediction at 0.0."""
    er = EpisodicRecall(seed=6)
    for i in range(8):
        er.encode(
            {"who": f"w{i}", "what": f"x{i}", "where": "lab", "when": f"q{i}", "affect": f"a{i}"}
        )
    out = er.apply_schema({"where": "lab", "what": "brandnew"})
    assert out["n_family"] == 8, "the family must still be FOUND"
    assert out["schema"] == {"where": "lab"}, "only the genuinely invariant slot is abstracted"
    assert out["prediction"] == {} and out["confidence"] == 0.0


def test_apply_schema_does_not_pick_a_winner_for_an_ambiguous_cue():
    """A cue spanning two families that split 6/6 on `who`. The prototype-margin test ALONE was
    measured to fabricate `who='bob'` at margin 0.367; the per-member consensus gate must reject
    it while still transferring the slot the two families genuinely share."""
    er = EpisodicRecall(seed=7)
    for i in range(6):
        er.encode(
            {"who": "alice", "what": f"m{i}", "where": "cafe", "when": "noon", "affect": f"f{i}"}
        )
    for i in range(6):
        er.encode(
            {"who": "bob", "what": f"n{i}", "where": "cafe", "when": "noon", "affect": f"g{i}"}
        )
    out = er.apply_schema({"where": "cafe", "what": "novel_dish"})
    assert out["n_family"] == 12
    assert "who" not in out["prediction"] and "who" not in out["schema"]
    assert out["prediction"] == {"when": "noon"}


def test_consensus_gate_is_what_rejects_the_ambiguous_slot():
    """Relaxing schema_consensus to a bare majority re-admits the fabricated slot -- proof the
    gate (not luck, and not the margin threshold) is doing the rejecting."""
    strict = EpisodicRecall(seed=7)
    loose = EpisodicRecall(seed=7, schema_consensus=0.4)
    for er in (strict, loose):
        for i in range(6):
            er.encode(
                {
                    "who": "alice",
                    "what": f"m{i}",
                    "where": "cafe",
                    "when": "noon",
                    "affect": f"f{i}",
                }
            )
        for i in range(6):
            er.encode(
                {"who": "bob", "what": f"n{i}", "where": "cafe", "when": "noon", "affect": f"g{i}"}
            )
    cue = {"where": "cafe", "what": "novel_dish"}
    assert "who" not in strict.apply_schema(cue)["prediction"]
    assert "who" in loose.apply_schema(cue)["prediction"]


# ------------------------------------------------------------------- determinism --------------
def test_apply_schema_is_deterministic_across_instances():
    a, b = _family_store(), _family_store()
    cue = {"where": "kitchen", "what": "pancakes"}
    assert a.apply_schema(cue) == b.apply_schema(cue)


def test_generalization_check_is_deterministic():
    assert _generalization_check()[0] == _generalization_check()[0]


def test_gist_is_unchanged_by_the_consensus_gate():
    """The additive consensus gate must not alter the pre-existing gist contract."""
    er = _family_store()
    g = er.gist({"where": "kitchen"}, m=8)
    assert set(g["schema"]) == {"who", "where", "when"} and g["n"] == 8


def test_default_slots_and_unknown_slots_are_accepted():
    er = EpisodicRecall(seed=0)
    assert set(DEFAULT_SLOTS) <= set(er._role)
    er.encode({"who": "self", "smell": "coffee"})
    assert "smell" in er._role


def test_selfcheck_main_passes():
    assert main() == 0
