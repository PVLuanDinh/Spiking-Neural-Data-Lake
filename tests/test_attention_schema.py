"""The reds for T078 Part I -- specs/022 Phase 4 / C4, the attention-schema gates
(evidence/t078-attention-schema.md 2.4).

WRITTEN AND COMMITTED **RED** at 4971a99 against the 2.9 step-2 stubs (7F/7, every failure an
ASSERTION and not a collection ImportError -- the shape T077's panel named, evidence 3.5 item 10,
finding C1). This is the step-4 revision: the instruments are green, the five constructed-input
helpers and the duck-typed stub brain have MOVED INTO the module (where `t078_reds()` needs them,
so that a gate run which never executes pytest still runs every red), and each red now also RUNS
the mutated scorer that reddens it.

  R1 THE FLOOR ABSORBS THE PLUG-IN BIAS -- an i.i.d. pair of label streams (|X| = 4, |Y| = 12,
     n = 600) reads `mi` at or below `perm_q95`; the control y := f(x) reads above it by more
     than 1 bit.  MUTATION: a floor drawn on UN-permuted y.
  R2 THE POSITIVE CONTROL IS NOT AN IDENTITY -- MI(a_t ; a_t+1) reads AT its floor on an i.i.d.
     action stream and above it on a 0.7-self-transition Markov stream, so G3's precondition can
     FAIL and its passing means something (the withdrawal of T077's oracle, C9).  MUTATION: an
     estimator that scores y against itself.
  R3 THE REFUSAL -- 145 pairs over 39 observed joint cells (the 300-tick probe's shape) REFUSE
     under the pre-registered per-CELL rule; the SAME 145 pairs are readable under 2.10's
     per-CLASS rule (the (g3c) control's twin, added to R3 at 455236d); 600 pairs over the same
     39 cells are readable; a one-class axis refuses.  MUTATIONS, one per half: N_MIN_PER_CELL
     := 0 reddens the per-CELL refusal; a `min_rule`-BLIND scorer, which discards the argument
     and reads everything per-CELL, reddens the per-CLASS twin (added in the 2026-09-15 panel
     fold -- the twin landed at 455236d with the implementation, so it was never committed red
     and no catalog mutation reached it).
  R4 THE DECODER'S BAND -- a 15-d input with the label planted in one coordinate decodes above
     its 200-shuffle band; the same input with the labels drawn independently decodes inside it;
     the band's q95 on n_test = 300 sits below 0.2 against a chance of 0.1 over 10 classes.
     MUTATION: a decoder scored on the half it was FITTED on.
  R5 CONTENT-BLIND BY TYPE -- two `control_state` calls whose control fields are equal and whose
     latents differ are bit-identical, and the 15 field names carry none of the content tokens.
     MUTATION: chan[0] appended to the vector.
  R6 THE VOID RULE CAN FIRE -- on a constructed recording where `u_t` IS a one-hot of the label,
     (g4a) is MET and (g4b) is NOT MET -> VOID; where `u_t` is noise, (g4a) is NOT MET -> PASS.
     MUTATION: (g4b)'s operands swapped.

R5 CARRIES A VACUITY CLAUSE 2.4 DOES NOT WORD, and it is here on purpose: a `control_state` that
returns a constant is content-blind by type TRIVIALLY, which is exactly the defect the T077 panel
found in R2/R5 (a constant stub passed an f(x)==f(x) check). So R5 also asserts that two brains
whose CONTROL fields differ read DIFFERENT vectors.

R6 reads `g4_rule` -- the SAME implementation of 2.3's branch map that `run_t078_campaign` will
read (P2: one rule, not one per caller). At the RED commit it was computed inline here because
`g4_rule` did not exist yet.

Every red runs on a CONSTRUCTED stream; none touches a brain, a lake or the network. The live
reading (a scratch brain, seeds 0-7, 1,200 ticks) is NOT here -- the pin is written once by
`--t078-campaign`.

The permutation count here is the TEST's, not the campaign's: 2.1 fixes n_perm = 2,000 for the
campaign's four MI readings and these reds use 200 so the suite stays cheap (the module's
`REDS_N_PERM`, disclosed there). 2.4's n_shuffle = 200 and n_test = 300 are used verbatim because
R4 asserts on them.
"""

import random

from snn_data_lake.cortex.attention_schema import (
    ESTOP_LABEL,
    LATENT_DIM,
    N_MIN_PER_CELL,
    REDS_N_PERM,
    U_DIM,
    U_FIELDS,
    U_FORBIDDEN,
    _reported_mi,
    _t078_mutated_scorer,
    control_state,
    decode_heldout,
    g4_rule,
    mi_bits,
    mi_read,
    t078_cell_stream,
    t078_g3,
    t078_g4,
    t078_g4_recording,
    t078_label_stream,
    t078_markov_stream,
    t078_pack_latents,
    t078_pairs,
    t078_planted_rows,
    t078_projection,
    t078_reds,
    t078_stub_brain,
)
from snn_data_lake.substrate.paired_contrast import contrast

TEST_SEEDS = tuple(range(8))
ACTION_N = 2000
ACTION_K = 12


def test_r1_the_permutation_floor_absorbs_the_plug_in_bias():
    """An i.i.d. (x, y) pair carries ZERO mutual information, but the plug-in estimate of a finite
    sample is positive by construction. The floor is what makes that bias harmless: the reading
    must sit AT or BELOW the 95th percentile of the same estimate on permuted y. The control is a
    deterministic map y := f(x), whose margin over the floor exceeds one bit -- so the floor is not
    simply swallowing everything. REDDENED IN FACT below by a floor drawn on UN-permuted y, whose
    every draw equals the reading, so the control's margin vanishes."""
    x = t078_label_stream(600, 4, seed=1)
    y = t078_label_stream(600, 12, seed=2)
    r = mi_read(x, y, n_perm=REDS_N_PERM, seed=0)
    assert r["readable"] is True, r
    assert r["mi"] == mi_bits(x, y), "the read must report the plug-in MI it scores"
    assert r["mi"] <= r["perm_q95"], r
    mapped = [(v * 3) % 12 for v in x]
    c = mi_read(x, mapped, n_perm=REDS_N_PERM, seed=0)
    assert c["readable"] is True, c
    assert c["mi"] - c["perm_q95"] > 1.0, c
    mutant = _t078_mutated_scorer("floor_unpermuted")(x, mapped, n_perm=REDS_N_PERM, seed=0)
    assert mutant["mi"] == mutant["perm_q95"], mutant
    assert not (mutant["mi"] - mutant["perm_q95"] > 1.0), "the un-permuted floor must redden R1"


def test_r2_the_positive_control_is_not_an_identity_of_the_estimator():
    """G3's precondition is MI(action_t ; action_t+1). It is a CONTROL only if it can fail: on an
    i.i.d. action stream it must read at its floor, and on a 0.7-self-transition Markov stream
    above it. T077's oracle was withdrawn for exactly this reason -- a statistic that reads 1.0 on
    any two-class stream is an identity, not a control (P8 clause 1). REDDENED IN FACT below by an
    estimator that scores y against itself, which reads the stream's own entropy."""
    iid = t078_label_stream(ACTION_N, ACTION_K, seed=3)
    r = mi_read(iid[:-1], iid[1:], n_perm=REDS_N_PERM, seed=0)
    assert r["readable"] is True, r
    assert r["mi"] <= r["perm_q95"], r
    mk = t078_markov_stream(ACTION_N, ACTION_K, 0.7, seed=4)
    c = mi_read(mk[:-1], mk[1:], n_perm=REDS_N_PERM, seed=0)
    assert c["readable"] is True, c
    assert c["mi"] > c["perm_q95"], c
    mutant = _t078_mutated_scorer("self_score")(iid[:-1], iid[1:], n_perm=REDS_N_PERM, seed=0)
    assert mutant["readable"] is True, mutant
    assert mutant["mi"] > mutant["perm_q95"], "scoring y against itself must redden R2"


def test_r3_the_refusal_fires_on_the_probes_own_shape():
    """N_MIN_PER_CELL = 10 pairs per OBSERVED joint cell. The 300-tick apparatus probe measured
    145 pairs over 39 cells, so that shape REFUSES and emits no `mi` -- which is why no dry run
    below about 400 pairs can preview a G3 number (2.8). The same 39 cells at 600 pairs are
    readable, and a one-class axis refuses whatever the count. REDDENED IN FACT below by
    N_MIN_PER_CELL := 0, under which the probe's own shape stops refusing."""
    x, y = t078_cell_stream(145, 39)
    r = mi_read(x, y, n_perm=REDS_N_PERM, seed=0)
    assert r["readable"] is False, r
    assert r["mi"] is None, "a refused reading must emit NO mi"
    assert r["refusal"], "a refusal must name why"
    assert 145 < N_MIN_PER_CELL * 39, "the constructed shape must be below the declared floor"
    x2, y2 = t078_cell_stream(600, 39)
    ok = mi_read(x2, y2, n_perm=REDS_N_PERM, seed=0)
    assert ok["readable"] is True, ok
    assert ok["cells"] == 39, ok
    one = mi_read([0] * 600, y2, n_perm=REDS_N_PERM, seed=0)
    assert one["readable"] is False and one["mi"] is None, one
    mutant = _t078_mutated_scorer("no_min")(x, y, n_perm=REDS_N_PERM, seed=0)
    assert mutant["readable"] is True, "N_MIN_PER_CELL := 0 must redden R3"
    assert mutant["mi"] is not None, mutant
    # 2.10: the (g3c) CONTROL reads under a per-CLASS minimum -- 145 pairs over 12 next-tick classes
    # (120 required) are readable while the same 145 pairs over 39 cells still refuse as the PRIMARY
    ctrl = mi_read(x, y, n_perm=REDS_N_PERM, seed=0, min_rule="per_class")
    assert ctrl["readable"] is True, ctrl
    assert mi_read(x, y, n_perm=REDS_N_PERM, seed=0, min_rule="per_cell")["readable"] is False
    # ... and the twin's OWN mutation (the panel fold): a scorer blind to `min_rule` reads the same
    # 145 pairs under the per-CELL floor and REFUSES, so R3's `ctrl_ok` conjunct goes red in fact.
    # None of the six 2.4 mutations reached it -- `no_min` leaves the twin readable.
    blind = _t078_mutated_scorer("blind_min_rule")(
        x, y, n_perm=REDS_N_PERM, seed=0, min_rule="per_class"
    )
    assert blind["readable"] is False, "a min_rule-blind scorer must redden R3's per-class twin"
    assert "per cell" in blind["refusal"], blind
    assert (
        _t078_mutated_scorer("no_min")(x, y, n_perm=REDS_N_PERM, seed=0, min_rule="per_class")[
            "readable"
        ]
        is True
    ), "R3's own N_MIN_PER_CELL := 0 mutation leaves the twin readable -- it is not its red"


def test_r4_the_decoder_reads_above_its_own_label_shuffle_band():
    """A 15-d input with the label planted in one coordinate must decode above a band of 200
    decoders trained on label-permuted train halves and scored on the TRUE test labels; the same
    construction with labels drawn independently of the rows must decode INSIDE that band; and the
    band's q95 on n_test = 300 must sit below 0.2 against a chance of 0.1 over 10 classes -- a band
    that has drifted up to the reading is not a band. REDDENED IN FACT below by a decoder scored on
    the half it was fitted on, which collapses the band."""
    rows, y = t078_planted_rows(600, U_DIM, 10, seed=5, planted=True)
    r = decode_heldout(rows, y, seed=0)
    assert r["readable"] is True, r
    assert r["n_test"] == 300, r
    assert r["shuffle_q95"] < 0.2, r
    assert r["acc"] > r["shuffle_q95"], r
    flat_rows, flat_y = t078_planted_rows(600, U_DIM, 10, seed=6, planted=False)
    n = decode_heldout(flat_rows, flat_y, seed=0)
    assert n["readable"] is True, n
    assert n["acc"] <= n["shuffle_q95"], n
    mutant = _t078_mutated_scorer("train_scored")(rows, y, 0)
    assert mutant["shuffle_q95"] >= 0.2, "a train-scored band must redden R4"


def test_r5_the_control_state_is_content_blind_by_type_and_not_constant():
    """`u_t` is content-blind BY TYPE: it is a pure function of the control fields, so two brains
    whose control fields are equal and whose latents and sensory channels differ read a
    bit-identical vector, and none of the 15 field names is a content token. The VACUITY CLAUSE
    (not worded in 2.4; see this file's docstring): two brains whose CONTROL fields DIFFER must
    read different vectors, or content-blindness is satisfied by a constant. REDDENED IN FACT
    below by appending chan[0], which breaks both the bit-identity and the length guard."""
    assert len(U_FIELDS) == U_DIM, U_FIELDS
    assert not (set(U_FIELDS) & set(U_FORBIDDEN)), sorted(set(U_FIELDS) & set(U_FORBIDDEN))
    a = t078_stub_brain(gate=(0.11, 0.12, 0.13, 0.14), latent_seed=1)
    b = t078_stub_brain(gate=(0.11, 0.12, 0.13, 0.14), latent_seed=2)
    ua, ub = control_state(a), control_state(b)
    assert len(ua) == U_DIM, ua
    assert ua == ub, (ua, ub)
    c = t078_stub_brain(gate=(0.90, 0.02, 0.03, 0.04), latent_seed=1)
    assert control_state(c) != ua, "a constant vector is content-blind trivially"
    leak = _t078_mutated_scorer("content_leak")
    assert len(leak(a)) != U_DIM, "the appended chan[0] must trip the length guard"
    assert leak(a) != leak(b), "a leaking vector must redden R5 on two brains with equal controls"


def test_r6_the_void_rule_can_fire_and_the_pass_branch_can_too():
    """2.3's G4 branches, read through `g4_rule` -- the same implementation the campaign reads.
    VOID: `u_t` IS a one-hot of the label, so (g4a) t(u_acc - u_shuffle_q95) clears the bar while
    (g4b) t(c15_acc - u_acc) does not; content-blindness would then be a naming convention. PASS:
    `u_t` is noise, so (g4a) does not clear the bar. REDDENED IN FACT below by swapping (g4b)'s
    operands, under which the VOID construction reads PASS-WITH-LEAK."""
    void_u, void_c, pass_u, pass_c = [], [], [], []
    for s in TEST_SEEDS:
        u_rows, c_rows, y = t078_g4_recording(s, "void")
        void_u.append(decode_heldout(u_rows, y, seed=s))
        void_c.append(decode_heldout(c_rows, y, seed=s))
        u_rows, c_rows, y = t078_g4_recording(s, "pass")
        pass_u.append(decode_heldout(u_rows, y, seed=s))
        pass_c.append(decode_heldout(c_rows, y, seed=s))
    assert all(r["readable"] for r in void_u + void_c + pass_u + pass_c)
    void = g4_rule(void_u, void_c)
    assert void["g4a"].met is True, void["g4a"]
    assert void["g4b"].met is False, void["g4b"]
    assert void["reading"] == "VOID", void["reading"]
    noise = g4_rule(pass_u, pass_c)
    assert noise["g4a"].met is False, noise["g4a"]
    assert noise["reading"] == "PASS", noise["reading"]
    swapped = _t078_mutated_scorer("g4_swap")(void_u, void_c)
    assert swapped["reading"] == "PASS-WITH-LEAK", swapped["reading"]
    assert swapped["reading"] != "VOID", "swapped (g4b) operands must redden R6"


def _row(t, awake, action="ENGAGE", band=2):
    """One recorder row, carrying only the fields `t078_pairs` reads. Constructed, never recorded:
    no brain is ticked anywhere in this file."""
    return {"t": t, "awake": awake, "action": action, "focus_band": band, "gate": [0.1] * 4}


def test_the_pairing_drops_sleep_straddling_and_estop_pairs_and_counts_both():
    """2.2's G3 pairing on a constructed recording. Ticks 0-3 are awake, 4-5 asleep, 6-9 awake,
    and tick 8 commits EMERGENCY_STOP. The awake-to-awake links are 0-1, 1-2, 2-3, 3->6 (a sleep
    bout between), 6-7, 7-8, 8-9, so: three clean pairs at the start, ONE dropped for straddling
    the bout, one clean pair (6-7), and TWO dropped for touching the E-STOP tick (7-8 and 8-9).
    Both drops are COUNTED, never silently skipped -- T077 3.2's no-stale-pairing amendment, and
    2.2's note that a nonzero E-STOP count is itself a finding because it never occurs under
    `_safe_ctx`."""
    rows = [_row(t, True) for t in range(4)]
    rows += [_row(t, False) for t in (4, 5)]
    rows += [_row(t, True) for t in (6, 7)]
    rows += [_row(8, True, action=ESTOP_LABEL), _row(9, True)]
    p = t078_pairs(rows)
    assert p["n_awake"] == 8, p
    assert p["dropped_sleep"] == 1, p
    assert p["dropped_estop"] == 2, p
    assert [(a["t"], b["t"]) for a, b in p["pairs"]] == [(0, 1), (1, 2), (2, 3), (6, 7)], p
    all_awake = [_row(t, True) for t in range(5)]
    q = t078_pairs(all_awake)
    assert q["dropped_sleep"] == 0 and q["dropped_estop"] == 0, q
    assert len(q["pairs"]) == 4, q
    assert t078_pairs([])["pairs"] == [], "an empty recording yields no pairs, not an error"


def test_the_readers_run_on_a_constructed_recording_and_the_control_axis_can_refuse():
    """The row SCHEMA contract between `t078_record_brain` and the two readers, pinned on a
    constructed recording so a KeyError cannot first appear fifteen minutes into the campaign.

    It also pins an ARITHMETIC CONSEQUENCE that 2.8 computes for the PRIMARY reading and not for
    the positive control, and that is worth seeing before the campaign runs rather than after.
    2.8 bounds the primary at "~580 pairs over <= 48 cells" -- focus(4) x actions(12). The (g3c)
    control's axes are actions x actions, up to 144 observed cells, and N_MIN_PER_CELL = 10 then
    demands 1,440 pairs. On a UNIFORM 12-class action stream the control therefore REFUSES at any
    pair count the pre-registered N = 1,200 ticks can produce, as it does here. Whether it refuses
    on the LIVE loop depends on how concentrated the real action marginal is (the apparatus probe
    read ENGAGE on 48 of 146 awake ticks, so the observed joint will be far below 144) -- the
    campaign's own per-seed `cells` line is what answers it. This test asserts the mechanism, not
    a prediction about the brain."""
    rng = random.Random(0)
    rows = []
    for t in range(1000):  # 750 awake -> ~700 pairs, above the primary's own 10 x 48 = 480 floor
        awake = (t % 20) < 15
        rows.append(
            {
                "t": t,
                "kind": "awake" if awake else "asleep",
                "awake": awake,
                "attend_calls": 1,
                "u": [rng.random() for _ in range(U_DIM)],
                "gate": [rng.random() for _ in range(4)],
                "focus_band": rng.randrange(4),
                "focus_strength": 0.03,
                "ignited": False,
                "sustain": 0.0,
                "drive_margin": 1.0,
                "pc_surprise": 0.1,
                "action": f"A{rng.randrange(12)}",
                "digit": rng.randrange(10),
                "label": rng.randrange(10) if awake else None,
                "chan": [0.1] * 4,
                "chan_from_gated": [0.1] * 4,
                "gated": [rng.random() for _ in range(LATENT_DIM)] if awake else [],
            }
        )
    g3 = t078_g3(rows, 0, n_perm=50)
    assert g3["n_pairs"] > 0 and g3["dropped_sleep"] > 0, g3
    assert g3["primary"]["readable"] is True, g3["primary"]
    # 2.10: the control reads under the per-CLASS rule (12 classes -> 120 pairs) where the
    # per-cell rule (up to 144 cells -> 1,440 pairs) refused it; the primary keeps per-cell
    assert g3["control"]["readable"] is True, g3["control"]
    per_cell = mi_read(
        [a["action"] for a, _ in t078_pairs(rows)["pairs"]],
        [b["action"] for _, b in t078_pairs(rows)["pairs"]],
        n_perm=50,
        seed=0,
        min_rule="per_cell",
    )
    assert per_cell["readable"] is False and "per cell" in per_cell["refusal"], per_cell
    g4 = t078_g4(rows, 0, n_shuffle=20)
    assert g4["n_rows"] > 0 and g4["n_classes"] == 10, g4
    for key in ("u", "c15", "raw64", "chan", "u_digit"):
        assert g4[key]["readable"] is True, (key, g4[key])
    assert len(g4["c15"]) and g4["c15"]["n_train"] + g4["c15"]["n_test"] == g4["n_rows"], g4["c15"]


def test_the_latent_packer_skips_ticks_without_a_latent_and_keeps_their_indices():
    """The red the first dry run fired (2026-09-15 14:48): asleep ticks record an empty `gated`,
    and a ragged np.asarray over every row crashed the campaign 16 s in. The packer keeps only
    the rows that carry a latent, as a (m, 64) float32 array, with their tick indices beside it."""
    rows = []
    for t in range(40):
        awake = (t % 4) != 3
        rows.append({"t": t, "awake": awake, "gated": [0.5] * LATENT_DIM if awake else []})
    arr, ticks = t078_pack_latents(rows)
    assert arr.shape == (30, LATENT_DIM) and arr.dtype.name == "float32", arr.shape
    assert ticks.tolist() == [t for t in range(40) if (t % 4) != 3], ticks.tolist()
    empty, no_ticks = t078_pack_latents([{"t": 0, "awake": False, "gated": []}])
    assert empty.shape == (0, LATENT_DIM) and no_ticks.shape == (0,)


def test_the_content_projection_is_fixed_by_its_crc32_seed():
    """2.1's matched arm: the 64-d gated latent is projected to the SAME 15 inputs `u_t` carries,
    by a Gaussian matrix seeded `crc32(b"t078-proj") + seed`. Two calls at one seed must be
    identical (a projection that moved between the two decodes would make (g4b) a comparison of
    two different arms), two seeds must differ, and the shape must be the matched one."""
    a, b = t078_projection(3), t078_projection(3)
    assert a.shape == (LATENT_DIM, U_DIM), a.shape
    assert bool((a == b).all()), "the projection must be a pure function of the seed"
    assert not bool((a == t078_projection(4)).all()), "a different seed must give a different arm"


def _g3_reading(readable, mi=0.30, q95=0.10, refusal=None):
    """One seed's slot of a `t078_g3` result, carrying only what `_reported_mi` reads."""
    if readable:
        return {"readable": True, "mi": mi, "perm_q95": q95, "refusal": None}
    return {"readable": False, "mi": None, "perm_q95": None, "refusal": refusal or "a refusal"}


def test_a_refused_reported_reading_carries_REFUSED_and_never_a_number():
    """The campaign's reported block must carry a REFUSAL for a refused reading, never a number.

    `reported.gate16` and `reported.same_tick` were built over `(mi or 0.0) - (perm_q95 or 0.0)`,
    which coerces a refused None into a 0.0 DIFF. On the landed campaign gate16 refused on all
    eight seeds, so the pin carries a contrast of eight fabricated zeros (t 0.0000, FPR 0.0,
    degenerate true) and the printed line dropped even the DEGENERATE marker. The sharper case is
    PARTIAL refusal: one fabricated zero among seven real diffs is NOT degenerate, so it would
    carry no flag at all. `_reported_mi` refuses on ANY refusing seed -- 2.9's (r) rule -- and
    emits no `t`, no `fpr` and no `met`, so nothing downstream can lift a number that never was."""
    seeds = tuple(range(8))
    # the readable diffs VARY by seed, or `contrast` is degenerate and its t is 0.0 by construction
    readable = [{"g3": {"k": _g3_reading(True, mi=0.30 + 0.01 * s)}} for s in seeds]
    c = _reported_mi("k (reported)", "k", seeds, readable)
    assert "refused" not in c, c
    assert c["degenerate"] is False and c["t"] > 0.0 and c["met"] is True, c

    all_refused = [{"g3": {"k": _g3_reading(False, refusal=f"seed {s} floor")}} for s in seeds]
    r = _reported_mi("k (reported)", "k", seeds, all_refused)
    assert r["refused"] is True and r["n_refused"] == 8, r
    assert not ({"t", "fpr", "met"} & set(r)), "a refused reading must carry NO number"
    assert r["refusals"]["3"] == "seed 3 floor", "a refusal must name why, per seed"

    partial = [{"g3": {"k": _g3_reading(True, mi=0.30 + 0.01 * s)}} for s in seeds]
    partial[3] = {"g3": {"k": _g3_reading(False, refusal="seed 3 floor")}}
    p = _reported_mi("k (reported)", "k", seeds, partial)
    assert p["refused"] is True and p["n_refused"] == 1, p
    assert not ({"t", "fpr", "met"} & set(p)), "one refusal must not ride inside seven real diffs"
    # THE RED: the incumbent `or 0.0` coercion fabricated a non-degenerate number on that input
    fabricated = contrast(
        "k (reported)",
        [(x["g3"]["k"]["mi"] or 0.0) - (x["g3"]["k"]["perm_q95"] or 0.0) for x in partial],
    )
    assert fabricated.degenerate is False and fabricated.t != 0.0, fabricated


def test_t078_reds_all_fire_in_the_modules_own_self_check_line():
    """The in-command control: the six reds above are ALSO one line of the module's self-check,
    so a gate run that never executes pytest still runs them -- including every mutated scorer.
    This is the line main() asserts."""
    reds = t078_reds()
    assert sorted(reds) == ["R1", "R2", "R3", "R4", "R5", "R6"], reds
    assert all(reds.values()), reds
