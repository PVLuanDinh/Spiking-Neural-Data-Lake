"""The reds for T078 Part I -- specs/022 Phase 4 / C4, the attention-schema gates
(evidence/t078-attention-schema.md 2.4).

WRITTEN AND COMMITTED **RED**, against the STUBS that landed in 2.9 step 2. Every test below
CONSTRUCTS its input and RUNS on the real symbol imported from `cortex/attention_schema.py`, so
each one fails on an ASSERTION -- never on a collection ImportError, which is the shape T077's
panel named (evidence/t077-metacognition.md 3.5 item 10, finding C1). The stubs return null
readings (`readable False`, `mi`/`acc` None, the 15 zeros), so the assertions that fail are the
ones that say "the instrument read something".

  R1 THE FLOOR ABSORBS THE PLUG-IN BIAS -- an i.i.d. pair of label streams (|X| = 4, |Y| = 12,
     n = 600) reads `mi` at or below `perm_q95`; the control y := f(x) reads above it by more
     than 1 bit.
  R2 THE POSITIVE CONTROL IS NOT AN IDENTITY -- MI(a_t ; a_t+1) reads AT its floor on an i.i.d.
     action stream and above it on a 0.7-self-transition Markov stream, so G3's precondition can
     FAIL and its passing means something (the withdrawal of T077's oracle, C9).
  R3 THE REFUSAL -- 145 pairs over 39 observed joint cells (the 300-tick probe's shape) REFUSE;
     600 pairs over the same 39 cells are readable; a one-class axis refuses.
  R4 THE DECODER'S BAND -- a 15-d input with the label planted in one coordinate decodes above
     its 200-shuffle band; the same input with the labels permuted decodes inside it; the band's
     q95 on n_test = 300 sits below 0.2 against a chance of 0.1 over 10 classes.
  R5 CONTENT-BLIND BY TYPE -- two `control_state` calls whose control fields are equal and whose
     latents differ are bit-identical, and the 15 field names carry none of the content tokens.
  R6 THE VOID RULE CAN FIRE -- on a constructed recording where `u_t` IS a one-hot of the label,
     (g4a) is MET and (g4b) is NOT MET -> VOID; where `u_t` is noise, (g4a) is NOT MET -> PASS.

R5 CARRIES A VACUITY CLAUSE 2.4 DOES NOT WORD, and it is here on purpose: a `control_state` that
returns a constant is content-blind by type TRIVIALLY, which is exactly the defect the T077 panel
found in R2/R5 (a constant stub passed an f(x)==f(x) check). So R5 also asserts that two brains
whose CONTROL fields differ read DIFFERENT vectors. 2.9 step 3 requires every red to fail against
a null-returning stub, and without this clause R5 would be green at this commit.

The mutated scorers that redden each of these IN FACT (2.4's six mutations) land with the
instruments in 2.9 step 4, together with `t078_reds()`'s six Trues -- they cannot be imported here
because they do not exist yet, and importing them would turn every test in this file into a
collection error instead of a red.

Every red runs on a CONSTRUCTED stream; none touches a brain, a lake or the network. The live
reading (a scratch brain, seeds 0-7, 1,200 ticks) is NOT here -- the pin is written once by
`--t078-campaign`.

The permutation counts here are the TEST's, not the campaign's: 2.1 fixes n_perm = 2,000 for the
campaign's four MI readings, and these reds use 200 draws so the suite stays cheap. 2.4 fixes
n_shuffle = 200 for the decoder band and that number is used verbatim.
"""

import random

from snn_data_lake.cortex.attention_schema import (
    N_MIN_PER_CELL,
    U_DIM,
    U_FIELDS,
    U_FORBIDDEN,
    control_state,
    decode_heldout,
    mi_bits,
    mi_read,
    t078_reds,
)
from snn_data_lake.substrate.paired_contrast import contrast

TEST_N_PERM = 200
TEST_SEEDS = tuple(range(8))


# ---- constructed inputs (moved into the module in 2.9 step 4, where `t078_reds()` needs them) ----
def _label_stream(n, k, seed):
    """n i.i.d. labels drawn uniformly from range(k) by a seeded random.Random."""
    rng = random.Random(seed)
    return [rng.randrange(k) for _ in range(n)]


def _markov_stream(n, k, p_self, seed):
    """n labels from a k-state chain that repeats the previous label with probability p_self and
    otherwise jumps uniformly to one of the other k-1 states -- autocorrelated by construction."""
    rng = random.Random(seed)
    out = [rng.randrange(k)]
    for _ in range(n - 1):
        if rng.random() < p_self:
            out.append(out[-1])
        else:
            out.append((out[-1] + 1 + rng.randrange(k - 1)) % k)
    return out


def _cell_stream(n_pairs, n_cells, n_x=4, n_y=12):
    """A (x, y) pair stream that occupies EXACTLY `n_cells` observed joint cells, by cycling
    through the first `n_cells` of the n_x by n_y grid. The shape R3 needs: the count of observed
    cells is fixed and the pair count is the only thing that moves."""
    grid = [(i, j) for i in range(n_x) for j in range(n_y)][:n_cells]
    pairs = [grid[k % len(grid)] for k in range(n_pairs)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _planted_rows(n, d, k, seed, planted=True):
    """n rows of d uniform-noise features with the label planted in coordinate 0 as y / (k - 1)
    (an ORDERED 1-d separation a linear multinomial decoder can reach). With planted=False the
    labels are permuted against the rows, so nothing predicts them."""
    rng = random.Random(seed)
    y = [rng.randrange(k) for _ in range(n)]
    rows = [[rng.random() for _ in range(d)] for _ in range(n)]
    if planted:
        for i in range(n):
            rows[i][0] = y[i] / float(k - 1)
    return rows, y


def _g4_recording(seed, kind, n=200, k=10):
    """A constructed (u_rows, content_rows, labels) recording for R6.

    kind "void": `u_t` IS a one-hot of the label (a perfect decode) and the content is a LOSSY
    one-hot of label // 2 (five distinguishable groups) -- so (g4a) is MET and (g4b), which reads
    content minus u_t, is NOT MET: the VOID branch. Deliberately NOT a tie, so that swapping
    (g4b)'s operands (2.4's mutation) flips the reading instead of leaving it degenerate.
    kind "pass": `u_t` is pure noise -- (g4a) NOT MET, the PASS branch."""
    rng = random.Random(seed)
    y = [rng.randrange(k) for _ in range(n)]
    u_rows, c_rows = [], []
    for i in range(n):
        u = [rng.random() * 1e-3 for _ in range(U_DIM)]
        if kind == "void":
            u[y[i] % U_DIM] += 1.0
        else:
            u = [rng.random() for _ in range(U_DIM)]
        c = [rng.random() * 1e-3 for _ in range(U_DIM)]
        c[(y[i] // 2) % U_DIM] += 1.0
        u_rows.append(u)
        c_rows.append(c)
    return u_rows, c_rows, y


def test_r1_the_permutation_floor_absorbs_the_plug_in_bias():
    """An i.i.d. (x, y) pair carries ZERO mutual information, but the plug-in estimate of a finite
    sample is positive by construction. The floor is what makes that bias harmless: the reading
    must sit AT or BELOW the 95th percentile of the same estimate on permuted y. The control is a
    deterministic map y := f(x), whose margin over the floor exceeds one bit -- so the floor is not
    simply swallowing everything. Reddened in 2.9 step 4 by a floor drawn on UN-permuted y."""
    x = _label_stream(600, 4, seed=1)
    y = _label_stream(600, 12, seed=2)
    r = mi_read(x, y, n_perm=TEST_N_PERM, seed=0)
    assert r["readable"] is True, r
    assert r["mi"] == mi_bits(x, y), "the read must report the plug-in MI it scores"
    assert r["mi"] <= r["perm_q95"], r
    mapped = [(v * 3) % 12 for v in x]
    c = mi_read(x, mapped, n_perm=TEST_N_PERM, seed=0)
    assert c["readable"] is True, c
    assert c["mi"] - c["perm_q95"] > 1.0, c


def test_r2_the_positive_control_is_not_an_identity_of_the_estimator():
    """G3's precondition is MI(action_t ; action_t+1). It is a CONTROL only if it can fail: on an
    i.i.d. action stream it must read at its floor, and on a 0.7-self-transition Markov stream
    above it. T077's oracle was withdrawn for exactly this reason -- a statistic that reads 1.0 on
    any two-class stream is an identity, not a control (P8 clause 1). Reddened in 2.9 step 4 by an
    estimator that scores y against itself."""
    iid = _label_stream(600, 12, seed=3)
    r = mi_read(iid[:-1], iid[1:], n_perm=TEST_N_PERM, seed=0)
    assert r["readable"] is True, r
    assert r["mi"] <= r["perm_q95"], r
    mk = _markov_stream(600, 12, 0.7, seed=4)
    c = mi_read(mk[:-1], mk[1:], n_perm=TEST_N_PERM, seed=0)
    assert c["readable"] is True, c
    assert c["mi"] > c["perm_q95"], c


def test_r3_the_refusal_fires_on_the_probes_own_shape():
    """N_MIN_PER_CELL = 10 pairs per OBSERVED joint cell. The 300-tick apparatus probe measured
    145 pairs over 39 cells, so that shape REFUSES and emits no `mi` -- which is why no dry run
    below about 400 pairs can preview a G3 number (2.8). The same 39 cells at 600 pairs are
    readable, and a one-class axis refuses whatever the count. Reddened in 2.9 step 4 by
    N_MIN_PER_CELL := 0."""
    x, y = _cell_stream(145, 39)
    r = mi_read(x, y, n_perm=TEST_N_PERM, seed=0)
    assert r["readable"] is False, r
    assert r["mi"] is None, "a refused reading must emit NO mi"
    assert r["refusal"], "a refusal must name why"
    assert 145 < N_MIN_PER_CELL * 39, "the constructed shape must be below the declared floor"
    x2, y2 = _cell_stream(600, 39)
    ok = mi_read(x2, y2, n_perm=TEST_N_PERM, seed=0)
    assert ok["readable"] is True, ok
    assert ok["cells"] == 39, ok
    one = mi_read([0] * 600, y2, n_perm=TEST_N_PERM, seed=0)
    assert one["readable"] is False and one["mi"] is None, one


def test_r4_the_decoder_reads_above_its_own_label_shuffle_band():
    """A 15-d input with the label planted in one coordinate must decode above a band of 200
    decoders trained on label-permuted train halves and scored on the TRUE test labels; the same
    input with the labels permuted must decode INSIDE that band; and the band's q95 on
    n_test = 300 must sit below 0.2 against a chance of 0.1 over 10 classes -- a band that has
    drifted up to the reading is not a band. Reddened in 2.9 step 4 by a decoder scored on its
    own TRAIN half."""
    rows, y = _planted_rows(600, U_DIM, 10, seed=5, planted=True)
    r = decode_heldout(rows, y, seed=0)
    assert r["readable"] is True, r
    assert r["n_test"] == 300, r
    assert r["shuffle_q95"] < 0.2, r
    assert r["acc"] > r["shuffle_q95"], r
    noise_rows, noise_y = _planted_rows(600, U_DIM, 10, seed=6, planted=False)
    n = decode_heldout(noise_rows, noise_y, seed=0)
    assert n["readable"] is True, n
    assert n["acc"] <= n["shuffle_q95"], n


def test_r5_the_control_state_is_content_blind_by_type_and_not_constant():
    """`u_t` is content-blind BY TYPE: it is a pure function of the control fields, so two brains
    whose control fields are equal and whose latents and sensory channels differ read a
    bit-identical vector, and none of the 15 field names is a content token. The VACUITY CLAUSE
    (not worded in 2.4; see this file's docstring): two brains whose CONTROL fields DIFFER must
    read different vectors, or content-blindness is satisfied by a constant. Reddened in 2.9
    step 4 by appending chan[0] to the vector."""
    assert len(U_FIELDS) == U_DIM, U_FIELDS
    assert not (set(U_FIELDS) & set(U_FORBIDDEN)), sorted(set(U_FIELDS) & set(U_FORBIDDEN))
    a = _stub_brain(gate=(0.11, 0.12, 0.13, 0.14), latent_seed=1)
    b = _stub_brain(gate=(0.11, 0.12, 0.13, 0.14), latent_seed=2)
    ua, ub = control_state(a), control_state(b)
    assert len(ua) == U_DIM, ua
    assert ua == ub, (ua, ub)
    c = _stub_brain(gate=(0.90, 0.02, 0.03, 0.04), latent_seed=1)
    assert control_state(c) != ua, "a constant vector is content-blind trivially"


def test_r6_the_void_rule_can_fire_and_the_pass_branch_can_too():
    """2.3's G4 branches, on constructed recordings. VOID: `u_t` IS a one-hot of the label, so
    (g4a) t(u_acc - u_shuffle_q95) clears the bar while (g4b) t(c15_acc - u_acc) does not --
    content-blindness would then be a naming convention. PASS: `u_t` is noise, so (g4a) does not
    clear the bar and the vector decodes nothing above chance. Reddened in 2.9 step 4 by swapping
    (g4b)'s operands."""
    u_reads, c_reads = [], []
    for s in TEST_SEEDS:
        u_rows, c_rows, y = _g4_recording(s, "void")
        u = decode_heldout(u_rows, y, seed=s)
        c = decode_heldout(c_rows, y, seed=s)
        assert u["readable"] is True and c["readable"] is True, (u, c)
        u_reads.append(u)
        c_reads.append(c)
    g4a = contrast("g4a u_t decodes content", [u["acc"] - u["shuffle_q95"] for u in u_reads])
    g4b = contrast(
        "g4b content decodes better",
        [c["acc"] - u["acc"] for c, u in zip(c_reads, u_reads, strict=True)],
    )
    assert g4a.met is True, g4a
    assert g4b.met is False, g4b
    p_reads = []
    for s in TEST_SEEDS:
        u_rows, _c_rows, y = _g4_recording(s, "pass")
        u = decode_heldout(u_rows, y, seed=s)
        assert u["readable"] is True, u
        p_reads.append(u)
    p_g4a = contrast("g4a on noise", [u["acc"] - u["shuffle_q95"] for u in p_reads])
    assert p_g4a.met is False, p_g4a


def test_t078_reds_all_fire_in_the_modules_own_self_check_line():
    """The in-command control: the six reds above are ALSO one line of the module's self-check,
    so a gate run that never executes pytest still runs them. This is the line main() asserts
    from 2.9 step 4 onward."""
    reds = t078_reds()
    assert sorted(reds) == ["R1", "R2", "R3", "R4", "R5", "R6"], reds
    assert all(reds.values()), reds


def _stub_brain(gate, latent_seed, pc=0.4, ema=0.2, focus=(2, 0.05), ignited=False, margin=1.0):
    """The attribute shape `control_state` reads, and nothing else: a duck-typed stand-in so R5
    runs without constructing a brain. Its `_real_percept` / `_attn_real` carry CONTENT that
    varies with `latent_seed` and that `control_state` must ignore."""
    import types

    rng = random.Random(latent_seed)
    latent = [rng.random() for _ in range(64)]
    ws = types.SimpleNamespace(broadcast=lambda: "digit-3", broadcast_hold=0.7)
    return types.SimpleNamespace(
        _learned_attn=types.SimpleNamespace(gate=list(gate), focus=focus),
        _attn_focus=focus,
        _pc_surprise=pc,
        _pc_surprise_ema=ema,
        _access_state={"ignited": ignited},
        _real_percept=latent,
        _attn_real={"channel": [sum(latent[:16]) / 16.0] * 4, "band": focus[0], "focus": focus[1]},
        workspace=ws,
        brain=types.SimpleNamespace(arm=types.SimpleNamespace(_decision_margin=margin)),
    )
