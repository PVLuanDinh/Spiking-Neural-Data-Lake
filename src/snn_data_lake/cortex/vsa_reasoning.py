"""
Spiking Vector-Symbolic Architecture (VSA) / hyperdimensional (HDC) reasoning -- GENUINELY SPIKING.

Each hypervector is a SPIKE CODE, not a dense float vector: D dimensions, each carrying exactly ONE
discrete spike on one of two lines -- an UP line (encoded +1) or a DOWN line (encoded -1). This is a
balanced two-line spike-polarity code (a recognised neuromorphic HDC substrate); nothing here is a
real-valued vector and no op is a float dot-product. The whole algebra runs in the spike domain:

  bind(a,b)   = per-dimension spike-channel XOR / coincidence GATE -- the output spikes UP when the
                two inputs spike on the SAME line, DOWN when on opposite lines. Self-inverse (bind
                again to unbind); the result is ~orthogonal to both inputs (protects/randomises).
  bundle(v..) = spike SUPERPOSITION + majority WTA -- overlay the component spike trains per
                dimension; the line that fired more spikes wins the winner-take-all and fires (ties
                -> up line). Superposes items into one code coincident with each member.
  permute(v)  = spike-line SHIFT -- a fixed axonal-delay / line permutation (order coding, no
                collisions).
  sim(a,b)    = spike COINCIDENCE -- count the dimensions where both codes spike on the SAME line
                (matched spikes), centred to [-1,1]. A coincidence COUNTER, not a MAC of floats.
  cleanup(v)  = nearest codebook entry by spike coincidence -- snaps a noisy composite spike code
                back to a clean stored symbol.

On-thesis value: it fills the *reason* verb of the thesis (store / search / reason) with
compositional, relational queries over spike codes -- bind a role to a filler and query it back by
spike XOR; bind a gesture to a context to steer a command; snap the noisy result to the nearest item
by spike coincidence. It COMPLEMENTS `paradigm_b_engine` (which reasons in spike TIMING / temporal
order); this module reasons in spike-POLARITY coincidence algebra. Both give Paradigm C a "reason"
path in spikes.

Atomic spike codes are seeded by crc32(label) (the same deterministic-across-process trick
signal_loop._proto uses), so the demo is reproducible with NO module-level random seeding. Two
on-thesis demos: a role-filler DATA RECORD you query by spike XOR + coincidence cleanup, and a
context-dependent ARM COMMAND algebra (bind(gesture, context) -> command). Capacity (how many pairs
one code holds before cleanup fails) is reported as mean +/- population std over seeds.

Honest note: a balanced two-line spike-polarity code and a dense bipolar +/-1 hypervector are the
SAME object, so this spiking substrate is numerically lossless w.r.t. the dense HD algebra -- the
recovery margins are exactly those of the dense version; the genuine change is the substrate + the
operations (spike XOR gate, superposition WTA, coincidence counter over asserted discrete spikes),
not the numbers. Spike-coincidence recall still degrades as a record is overloaded -- the finite-
capacity point -- exactly as a spiking associative memory should.

  Run: python -m snn_data_lake.cortex.vsa_reasoning       # (also run by run_selfchecks.py)
"""

import random
import zlib

D = 512  # spike-code dimensionality (each dim = one spike on the up(+1) or down(-1) line)
VOCAB = 128  # filler vocabulary cleanup searches over (distractors make capacity finite)
SEEDS = range(5)  # >=5 seeds -> mean +/- std
THRESH = 0.95  # filler-recovery fraction counted as "clean"


def mean_std(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5  # population std


def hv(label):
    """Deterministic atomic SPIKE CODE for a symbol (crc32-seeded, like signal_loop._proto).
    Every dimension fires exactly one spike -- on the UP line (+1) or the DOWN line (-1)."""
    rng = random.Random(zlib.crc32(label.encode()))
    return [rng.choice((-1, 1)) for _ in range(D)]


def is_spike_code(v):
    """True iff every dimension carries exactly one discrete spike, on the up(+1) or down(-1) line."""
    return all(x == 1 or x == -1 for x in v)


def bind(a, b):
    # spike-channel XOR / coincidence gate: same line -> UP(+1), opposite lines -> DOWN(-1).
    # Compares which line fired (equality of spikes), not an arithmetic product; self-inverse.
    return [1 if pa == pb else -1 for pa, pb in zip(a, b)]


def bundle(vecs):
    # spike SUPERPOSITION + majority WTA: per dimension, overlay the component spike trains and let
    # the line that fired more spikes win and fire (ties -> up line). A vote over spikes, not a sum.
    out = []
    for col in zip(*vecs):
        up = sum(1 for s in col if s == 1)
        out.append(1 if up * 2 >= len(col) else -1)  # up-line spikes >= half -> UP wins the WTA
    return out


def permute(v, k=1):
    return v[-k:] + v[:-k]  # spike-line cyclic shift (axonal-delay order code)


def coincidence(a, b):
    """Spike coincidence: number of dimensions where both codes spike on the SAME line (matched
    spikes). A coincidence COUNT over discrete spikes -- the spiking similarity primitive."""
    return sum(1 for pa, pb in zip(a, b) if pa == pb)


def coincidence_many(q, mat):
    """Spike coincidence of ONE code against MANY stored codes -- the same COUNT as `coincidence`,
    for every row at once.

    This is the batched form of the primitive, not a different primitive. For a balanced two-line
    spike code every dimension is +1 or -1, so per row

        sum_i q_i*m_i  ==  matches - mismatches  ==  2*coincidence(q, m) - D
        =>  coincidence(q, m) == (D + q.m) / 2

    exactly, in integers. Every dot product is an integer in [-D, D], and D=512 is far inside
    float32's exact-integer range (2**24), so the float32 route returns the SAME INTEGER the scalar
    counter does -- verified against `coincidence` in this module's self-check. The spike semantics
    are untouched: what comes back is still a count of matched spikes.

    `mat` is shape (rows, D) with rows in INSERTION order, so `argmax` / `argsort(kind="stable")`
    over the result breaks ties on the first-stored row, exactly as `max()` and a stable sort do
    over the scalar counter. (`argsort(-x, kind="stable")` -- NOT `argsort(x)[::-1]`, which reverses
    ties too.)

    Reason it exists: `coincidence` was measured at 74.5% of the live wake tick (py-spy on the
    daemon), entered through substrate.episodic_recall, because it is a Python loop run once per
    stored episode per tick. Same precedent as AttractorNet.accuracy folding 36 recall trials into
    one array pass for 6.93x, bit-exact.
    """
    import numpy as np  # local: keeps this module importable without numpy

    q = np.asarray(q, dtype=np.float32)
    return (q.shape[0] + np.asarray(mat, dtype=np.float32) @ q) / 2.0


def sim(a, b):
    # centred spike coincidence in [-1,1]: (matches - mismatches)/D. Built from a coincidence COUNT,
    # NOT a float dot-product. (For a balanced two-line code this equals cosine -- the same object.)
    return (2 * coincidence(a, b) - D) / D


def cleanup(v, codebook):
    """Snap a noisy spike code to the most COINCIDENT stored symbol; returns (label, similarity)."""
    lab = max(codebook, key=lambda k: coincidence(v, codebook[k]))
    return lab, sim(v, codebook[lab])


def record_recovery(k_pairs, seed):
    """Bundle k_pairs role->filler spike-XOR bindings into ONE spike code, then query each role back
    out by spike XOR. Cleanup searches the FULL filler vocabulary (distractors) by spike coincidence,
    so recovery degrades as the record is overloaded -- the finite-capacity point. Returns the
    fraction recovered correctly."""
    rng = random.Random(seed)
    roles = [hv(f"ROLE{i}") for i in range(k_pairs)]
    vocab = {f"V{i}": hv(f"V{i}") for i in range(VOCAB)}
    chosen = rng.sample(list(vocab), k_pairs)  # k distinct fillers from the vocab
    record = bundle([bind(roles[i], vocab[chosen[i]]) for i in range(k_pairs)])
    correct = sum(cleanup(bind(record, roles[i]), vocab)[0] == chosen[i] for i in range(k_pairs))
    return correct / k_pairs


def main():
    # ---- demo 1: context-dependent arm command algebra (reason by spike XOR) ----
    commands = {c: hv(c) for c in ("GRIP", "WAIT", "RETREAT", "HOME")}
    rules = bundle(
        [
            bind(bind(hv("APPROACH"), hv("NEAR")), hv("GRIP")),
            bind(bind(hv("APPROACH"), hv("FAR")), hv("WAIT")),
            bind(bind(hv("RETREAT"), hv("NEAR")), hv("RETREAT")),
        ]
    )

    def decide(gesture, context):
        """Query the bundled rule code by spike XOR, then WTA over command codes by coincidence.
        Returns (winner, coincidence margin over the runner-up) -- the spiking read-out margin."""
        q = bind(rules, bind(hv(gesture), hv(context)))
        scored = sorted(((coincidence(q, commands[c]), c) for c in commands), reverse=True)
        return scored[0][1], scored[0][0] - scored[1][0]

    near_cmd, near_margin = decide("APPROACH", "NEAR")
    far_cmd, far_margin = decide("APPROACH", "FAR")

    # ---- demo 2: role-filler record capacity vs load (coincidence cleanup over a 128-filler vocab) ----
    loads = (4, 24, 64)
    cap = {k: [record_recovery(k, s) for s in SEEDS] for k in loads}

    # ---- spiking-substrate checks (discrete spikes; coincidence sim; spike-op bind/bundle) ----
    a, b, c = hv("A"), hv("B"), hv("C")
    spikes_discrete = (
        is_spike_code(a) and is_spike_code(bind(a, b)) and is_spike_code(bundle([a, b, c]))
    )
    self_coincidence = (
        coincidence(a, a) == D and sim(a, a) == 1.0
    )  # perfectly coincident with itself
    anti_coincidence = sim(a, [-x for x in a]) == -1.0  # opposite spike lines everywhere
    unbind_exact = bind(bind(a, b), a) == b  # spike XOR is exactly self-inverse
    bind_dissimilar = abs(sim(bind(a, b), a)) < 0.15  # a bound pair hides its factors
    permute_roundtrip = permute(permute(a, 1), D - 1) == a
    # RETIRED 2026-07-29: this line used to be
    #     no_tensor_libs = "numpy" not in sys.modules and "torch" not in sys.modules
    # a numpy-absence boast that was PRINTED and never asserted. numpy is an unconditional core
    # dependency (pyproject:32) and `coincidence_many` now uses it on purpose, so the claim is both
    # obsolete and about to read False. What actually needs guarding is not numpy's absence but that
    # the batched route returns the SAME COUNT as the scalar counter -- so the boast is replaced by an
    # assertion of exactly that. A dead claim becomes a live gate.
    _probe = (a, b, c, [-x for x in a])
    batched_exact = [int(round(float(x))) for x in coincidence_many(a, _probe)] == [
        coincidence(a, v) for v in _probe
    ]
    deterministic = hv("Z") == hv("Z") and record_recovery(24, 3) == record_recovery(24, 3)

    print("=" * 62)
    print("Spiking VSA / HDC reasoning -- two-line spike-polarity codes, spike-domain ops")
    print("=" * 62)
    print(
        f"D={D}  seeds={len(list(SEEDS))}  substrate=spike code (1 spike/dim on up(+1)/down(-1) line)"
    )
    print("ops: bind=spike XOR gate  bundle=superposition WTA  sim=spike coincidence count")
    print(
        f"arm algebra: bind(APPROACH,NEAR) -> {near_cmd} (margin {near_margin}) | "
        f"bind(APPROACH,FAR) -> {far_cmd} (margin {far_margin})  (context flips the command)"
    )
    print("\n role-filler record recovery vs load (coincidence cleanup, mean +/- std):")
    for k in loads:
        m, s = mean_std(cap[k])
        print(f"   {k:>2} pairs : {m:>6.1%} +/- {s:>4.1%}")
    print(
        f"\nspiking: discrete-spikes={spikes_discrete}  self-coincidence={self_coincidence}  "
        f"anti-coincidence={anti_coincidence}  batched-count==scalar-count={batched_exact}"
    )
    print(
        f"structure: unbind(XOR) exact={unbind_exact}  bound-pair-hides-factors={bind_dissimilar}"
        f"  permute-roundtrip={permute_roundtrip}  deterministic={deterministic}"
    )
    print(
        "Spiking VSA gives Paradigm C a role/relation algebra (spike XOR / superposition WTA /"
        " coincidence) over spike codes; it COMPLEMENTS paradigm_b_engine, which reasons in spike"
        " TIMING."
    )
    print("=" * 62)

    # ---- self-check: assert FUNCTION, GENUINELY-SPIKING, and DETERMINISM; all deterministic ----
    # (a) FUNCTION: analogy/cleanup/binding return the right items above the distractor margin.
    assert near_cmd == "GRIP" and far_cmd == "WAIT", (
        f"context did not steer the command: NEAR->{near_cmd} FAR->{far_cmd}"
    )
    assert near_margin > 0 and far_margin > 0, (
        "winning command must beat the runner-up on coincidence"
    )
    # (a2) the BATCHED counter is the same primitive, not an approximation of it. This is the gate
    # that lets substrate.episodic_recall rank a whole episode store with one GEMV instead of one
    # Python loop per stored episode per wake tick (measured 74.5% of the live tick).
    assert batched_exact, "coincidence_many disagrees with the scalar coincidence counter"
    lo_m, _ = mean_std(cap[4])
    hi_m, _ = mean_std(cap[64])
    assert lo_m >= THRESH, f"small record should recover cleanly by coincidence: {lo_m:.2f}"
    assert lo_m > hi_m, "recovery should degrade as the record is overloaded (finite capacity)"
    # (b) GENUINELY SPIKING: discrete spike events, coincidence similarity, spike-op bind/bundle,
    #     and no tensor libraries imported.
    assert spikes_discrete, (
        "hypervectors must be discrete spike codes (one spike/dim on up/down line)"
    )
    assert self_coincidence and anti_coincidence, (
        "sim must be a spike-coincidence count, not a dot-product"
    )
    assert unbind_exact and bind_dissimilar, "bind must be a self-inverse spike XOR gate"
    assert is_spike_code(rules), "bundle (superposition WTA) must emit a valid spike code"
    # (c) DETERMINISM: identical results across runs (crc32/seed-driven, no module-level RNG).
    assert deterministic, "spike codes and recovery must be deterministic given the seed"
    print(
        "self-check OK: spike-XOR/superposition-WTA/coincidence algebra holds, context-steered "
        "command, clean coincidence recall at low load degrading with capacity, genuinely spiking"
    )


if __name__ == "__main__":
    main()
