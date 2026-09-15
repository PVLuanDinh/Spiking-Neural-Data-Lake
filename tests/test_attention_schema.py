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
  R3 THE REFUSAL -- 145 pairs over 39 observed joint cells (the 300-tick probe's shape) REFUSE;
     600 pairs over the same 39 cells are readable; a one-class axis refuses.  MUTATION:
     N_MIN_PER_CELL := 0.
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

from snn_data_lake.cortex.attention_schema import (
    N_MIN_PER_CELL,
    REDS_N_PERM,
    U_DIM,
    U_FIELDS,
    U_FORBIDDEN,
    _t078_mutated_scorer,
    control_state,
    decode_heldout,
    g4_rule,
    mi_bits,
    mi_read,
    t078_cell_stream,
    t078_g4_recording,
    t078_label_stream,
    t078_markov_stream,
    t078_planted_rows,
    t078_reds,
    t078_stub_brain,
)

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


def test_t078_reds_all_fire_in_the_modules_own_self_check_line():
    """The in-command control: the six reds above are ALSO one line of the module's self-check,
    so a gate run that never executes pytest still runs them -- including every mutated scorer.
    This is the line main() asserts."""
    reds = t078_reds()
    assert sorted(reds) == ["R1", "R2", "R3", "R4", "R5", "R6"], reds
    assert all(reds.values()), reds
