"""The REPAIRED alignment criterion for specs/015-resolve-weight-transport, and its qualification.

SCRATCH PROBE. Lives under specs/, is never promoted to src/, and is registered in no gate. It is
an instrument, not a build (plan.md, Project Structure).

WHAT THE CRITERION COMPARES (contracts/criterion.md):

    cos( dh_rule , dh_transport )

both computed on the SAME input and the SAME pre-update weights, so they differ in the feedback
vector alone. `dh` is the hidden-layer update direction -- the thing the e-prop invariant is
actually about, and the thing MlpQ.update actually applies.

WHAT THE OLD CRITERION COMPARED, and why it could not discriminate: mlp_q's own docstring records
it as "essentially cos(W2[a], B[a]) under the relu mask (`err` cancels)". For a FIXED RANDOM B that
is near zero by construction -- it asks whether the feedback is PARALLEL to the forward weights,
which is the weight symmetry feedback alignment exists to deny.

DISCLOSED DEGENERACY, stated rather than smoothed: under this module's arithmetic the new quantity
remains a MONOTONE FUNCTION of the old one. `err` is a shared scalar and the relu mask zeroes the
same coordinates on both sides, so the algebra is not what changed. THE REPAIR IS:
    (a) reading it PER EPOCH, so alignment DEVELOPMENT is visible rather than an endpoint, and
    (b) pairing it with a feedback vector that CAN MOVE.
A reader should predict that a fixed-random draw still scores near zero here. That is CORRECT
BEHAVIOUR, not a residual bug -- the old criterion's defect was that its subject was frozen by
construction, so no alignment rule could ever pass. A rule whose feedback converges can.

QUALIFICATION -- this criterion may not judge any candidate until all three pass:
    (1) identity        transport against itself scores exactly 1.0
    (2) control         the permutation null scores LOW
    (3) reachable red   an input exists that makes it fail, and the output is recorded
Failing (1) means the harness is wrong. Failing (2) means it measures MAGNITUDE, not direction, and
per FR-002 it is DELETED rather than weakened.

Run:  D:\\AI-project\\spiking-neural-data-lake\\.venv-lake\\Scripts\\python.exe
      specs/015-resolve-weight-transport/probe/criterion.py
"""

import math
import random

# The bar this project's cross-cutting e-prop invariant already sets. NOT amended by this lane.
COSINE_BAR = 0.5


def dh(err, feedback, pre):
    """The hidden-layer update direction MlpQ.update applies, for an arbitrary feedback vector.

    Mirrors the shipped `dh` step exactly: the relu mask gates dead units to zero, and `err` is a
    shared scalar. Passing `feedback=W2[a]` reproduces the TRANSPORT gradient; passing a fixed
    random draw reproduces the feedback-alignment twin; passing a converging estimate is the
    Kolen-Pollack candidate.
    """
    return [err * feedback[j] if pre[j] > 0.0 else 0.0 for j in range(len(pre))]


def cosine(a, b):
    """Cosine between two equal-length vectors. Returns 0.0 if either has no magnitude."""
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


def permutation_null(vec, rng):
    """THE CONTROL, and it is not the method itself (P8 guard 2).

    Same magnitude, same element multiset, different direction. A criterion that scores this near
    the real gradient is measuring MAGNITUDE and is deleted, not weakened.

    `rng` is a per-instance random.Random -- never module-level seeding.
    """
    out = list(vec)
    rng.shuffle(out)
    return out


def _fixture(seed=0, hidden=32):
    """One deterministic (err, W2 row, pre-activation) triple. Per-instance RNG."""
    rng = random.Random(seed)
    err = rng.uniform(-1.0, 1.0)
    w2 = [rng.uniform(-1.0, 1.0) / math.sqrt(hidden) for _ in range(hidden)]
    # Roughly half the units dead, which is what makes the relu mask load-bearing.
    pre = [rng.uniform(-1.0, 1.0) for _ in range(hidden)]
    return err, w2, pre


def _selfcheck():
    print("=" * 92)
    print("criterion qualification -- specs/015-resolve-weight-transport")
    print("=" * 92)

    err, w2, pre = _fixture()
    live = sum(1 for p in pre if p > 0.0)
    print(f"  fixture: hidden={len(pre)}, live relu units={live}, err={err:+.6f}")
    assert live > 0, "DENOMINATOR: every unit is dead, so every dh is the zero vector and every"
    " comparison below would be vacuous"

    d_transport = dh(err, w2, pre)
    nz = sum(1 for x in d_transport if x != 0.0)
    print(f"  dh_transport: {len(d_transport)} coords, {nz} non-zero  (must be > 0)")
    assert nz > 0, "DENOMINATOR: dh_transport is all zeros; nothing below measures anything"

    # (1) IDENTITY -- if this is not 1.0 the harness is wrong and nothing downstream means anything.
    ident = cosine(d_transport, d_transport)
    print(f"\n  (1) IDENTITY      cos(dh_transport, dh_transport) = {ident:+.12f}   expect 1.0")
    assert abs(ident - 1.0) < 1e-12, f"identity is {ident!r}, so the harness is wrong"

    # (2) CONTROL -- the permutation null. Same magnitude, same multiset, shuffled direction.
    #
    # ASSERTED ON A PERCENTILE, NOT A MAX, and the reason is a measurement not a preference: at
    # width 32 the null's tail CROSSES the bar (see NOISE FLOOR below). An earlier revision of this
    # check asserted `max < BAR` over 64 shuffles and passed at 0.4765 -- by luck. Over 2000 it
    # exceeds the bar 3.8% of the time, so that assert was under-powered and would flake. A control
    # whose pass depends on how many samples you happened to draw is not a control.
    rng = random.Random(12345)
    nulls = sorted(
        abs(cosine(dh(err, w2, pre), permutation_null(d_transport, rng))) for _ in range(2000)
    )
    n_p95 = nulls[int(0.95 * len(nulls))]
    n_med = nulls[len(nulls) // 2]
    mag_t = math.sqrt(sum(x * x for x in d_transport))
    mag_n = math.sqrt(sum(x * x for x in permutation_null(d_transport, random.Random(1))))
    print(
        f"  (2) CONTROL       permutation null, 2000 shuffles: median={n_med:.4f} "
        f"p95={n_p95:.4f} max={nulls[-1]:.4f}"
    )
    print(
        f"                    magnitudes identical? {abs(mag_t - mag_n) < 1e-12} "
        f"({mag_t:.6f} vs {mag_n:.6f})"
    )
    assert abs(mag_t - mag_n) < 1e-12, "the null must preserve magnitude or it is not this control"
    assert n_med < COSINE_BAR, (
        f"the permutation null MEDIAN is {n_med:.4f} >= {COSINE_BAR}: this criterion measures "
        f"MAGNITUDE, not direction. Per FR-002 it is DELETED, not weakened."
    )

    # (3) REACHABLE RED -- construct the input that makes the criterion fail, and record it.
    anti = [-w for w in w2]
    red = cosine(dh(err, anti, pre), d_transport)
    print(f"  (3) REACHABLE RED anti-aligned (-W2): cos = {red:+.6f} -> fails > {COSINE_BAR}")
    assert red < COSINE_BAR, "the criterion has no reachable red and must be DELETED (P8 guard 1)"

    # NOISE FLOOR -- is the 0.5 bar even above it at this width? If a shuffled or random vector
    # routinely clears 0.5, no criterion at hidden=32 can discriminate and this lane stops.
    floor_rng = random.Random(999)
    floor = [
        abs(cosine(dh(err, permutation_null(w2, floor_rng), pre), d_transport)) for _ in range(2000)
    ]
    over = sum(1 for c in floor if c > COSINE_BAR)
    floor.sort()
    p95, p99 = floor[int(0.95 * len(floor))], floor[int(0.99 * len(floor))]
    print(f"\n  NOISE FLOOR over 2000 shuffles: p95={p95:.4f} p99={p99:.4f} max={max(floor):.4f}")
    print(f"             |cos| > {COSINE_BAR} in {over} of 2000 ({100.0 * over / len(floor):.2f}%)")

    # Disclosed: a fixed random draw scores near zero HERE TOO. That is correct, not a bug.
    # The RNG is constructed ONCE PER DRAW -- building it inside the comprehension would give
    # every coordinate the same first value, i.e. a CONSTANT vector, not a random one.
    fr = []
    for s in range(1, 9):
        r = random.Random(s)
        b = [r.uniform(-1.0, 1.0) / math.sqrt(len(pre)) for _ in range(len(pre))]
        fr.append(cosine(dh(err, b, pre), d_transport))
    print(
        f"\n  DISCLOSED: fixed-random feedback over 8 draws: "
        f"[{min(fr):+.4f}, {max(fr):+.4f}], |cos| > {COSINE_BAR} in "
        f"{sum(1 for c in fr if abs(c) > COSINE_BAR)} of 8"
    )
    print("             This is EXPECTED. A frozen draw cannot align; the repair is that the")
    print("             candidate's feedback can MOVE, and that alignment is read per-epoch.")

    # THE CONSEQUENCE FOR THE DECISION BAR, and it is the reason this section exists.
    # A SINGLE draw clearing 0.5 is weak evidence -- noise does it 3.8% of the time here. The
    # spec's bar is 6 of 8 draws, and that IS robust: P(>=6 of 8 | p=0.038) is on the order of
    # 1e-7. So the bar survives this finding; a per-draw reading does not.
    p_single = over / len(floor)
    p6of8 = sum(math.comb(8, k) * p_single**k * (1 - p_single) ** (8 - k) for k in (6, 7, 8))
    print(
        f"  CONSEQUENCE: single draw over the bar by chance = {p_single:.4f}; "
        f"6-of-8 by chance = {p6of8:.3e}"
    )
    print("             The spec's 6-of-8 bar is robust to this floor. A SINGLE draw is not,")
    print("             and must not be read as evidence on its own.")

    print("\n  qualification OK: identity exact, control separates on median, red reachable")
    return {
        "identity": ident,
        "null_median": n_med,
        "null_p95": n_p95,
        "red": red,
        "fixed_random": fr,
        "noise_floor_p99": p99,
        "p_single": p_single,
        "p6of8": p6of8,
    }


if __name__ == "__main__":
    _selfcheck()
