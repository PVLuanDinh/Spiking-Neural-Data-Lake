"""
Vector-Symbolic Architecture (VSA) / hyperdimensional (HDC) reasoning — neuromorphic-ADJACENT,
NOT spike-timing.

This is dense bipolar-hypervector algebra: symbols, records, and relations are represented as
high-dimensional +/-1 vectors (a balanced code) and COMPOSED with a few O(D) algebraic ops. The
substrate is a DENSE HD vector, not a spike train — nothing here computes in spike timing. HDC is a
recognised neuromorphic-computing primitive (and a bipolar hypervector maps onto a balanced spike-
polarity / phasor code), so the paradigm is neuromorphic-ADJACENT — but the honest label for THIS
module is HD/VSA reasoning, not "spiking".

On-thesis value: it fills the *reason* verb of the thesis (store / search / reason) with
compositional, relational queries — bind a role to a filler and query it back by unbinding; bind a
gesture to a context to steer a command. It COMPLEMENTS `paradigm_b_engine`, which reasons in genuine
spike timing (temporal-order queries); this module reasons in HD-vector algebra instead. Both give
Paradigm C a "reason" path; only paradigm_b_engine computes in spikes.

The algebra (all pure-stdlib, all O(D)):
  bind(a,b)   = elementwise product  — associates a role with a filler; SELF-INVERSE (bind again to
                unbind), and the result is dissimilar to both inputs (protects/randomises).
  bundle(v..) = elementwise majority sign — superposes items into one vector similar to each member.
  permute(v)  = cyclic shift — encodes order/position without collisions.
  cleanup(v)  = nearest stored symbol — snaps a noisy composite back to a clean codebook entry.

Atomic hypervectors are seeded by crc32(label) (the same deterministic-across-process trick
signal_loop._proto uses), so the demo is reproducible with NO module-level random seeding. Two
on-thesis demos: a role-filler DATA RECORD you query by unbinding, and a context-dependent ARM
COMMAND algebra (bind(gesture, context) -> command). Capacity (how many pairs one vector holds before
cleanup fails) is reported as mean +/- population std over seeds via a tiny local mean_std.

  Run: python -m snn_data_lake.vsa_reasoning       # (also run by run_selfchecks.py)
"""
import random
import zlib

D = 512           # hypervector dimensionality (bipolar +/-1 balanced code)
VOCAB = 128       # filler vocabulary cleanup searches over (distractors make capacity finite)
SEEDS = range(5)  # >=5 seeds -> mean +/- std
THRESH = 0.95     # filler-recovery fraction counted as "clean"


def mean_std(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5   # population std


def hv(label):
    """Deterministic atomic hypervector for a symbol (crc32-seeded, like signal_loop._proto)."""
    rng = random.Random(zlib.crc32(label.encode()))
    return [rng.choice((-1, 1)) for _ in range(D)]


def bind(a, b):
    return [x * y for x, y in zip(a, b)]          # elementwise product (self-inverse)


def bundle(vecs):
    return [1 if s >= 0 else -1 for s in (sum(col) for col in zip(*vecs))]  # majority sign


def permute(v, k=1):
    return v[-k:] + v[:-k]                          # cyclic roll (order coding)


def sim(a, b):
    return sum(x * y for x, y in zip(a, b)) / D     # cosine for bipolar, in [-1,1]


def cleanup(v, codebook):
    """Snap a noisy vector to the most similar stored symbol; returns (label, similarity)."""
    lab = max(codebook, key=lambda k: sim(v, codebook[k]))
    return lab, sim(v, codebook[lab])


def record_recovery(k_pairs, seed):
    """Bundle k_pairs role->filler bindings into ONE vector, then query each role back out.
    Cleanup searches the FULL filler vocabulary (distractors), so recovery degrades as the record
    is overloaded — the finite-capacity point. Returns the fraction recovered correctly."""
    rng = random.Random(seed)
    roles = [hv(f"ROLE{i}") for i in range(k_pairs)]
    vocab = {f"V{i}": hv(f"V{i}") for i in range(VOCAB)}
    chosen = rng.sample(list(vocab), k_pairs)                     # k distinct fillers from the vocab
    record = bundle([bind(roles[i], vocab[chosen[i]]) for i in range(k_pairs)])
    correct = sum(cleanup(bind(record, roles[i]), vocab)[0] == chosen[i] for i in range(k_pairs))
    return correct / k_pairs


def main():
    # ---- demo 1: context-dependent arm command algebra (reason by binding) ----
    commands = {c: hv(c) for c in ("GRIP", "WAIT", "RETREAT", "HOME")}
    rules = bundle([
        bind(bind(hv("APPROACH"), hv("NEAR")), hv("GRIP")),
        bind(bind(hv("APPROACH"), hv("FAR")), hv("WAIT")),
        bind(bind(hv("RETREAT"), hv("NEAR")), hv("RETREAT")),
    ])

    def decide(gesture, context):
        return cleanup(bind(rules, bind(hv(gesture), hv(context))), commands)[0]

    near_cmd = decide("APPROACH", "NEAR")
    far_cmd = decide("APPROACH", "FAR")

    # ---- demo 2: role-filler record capacity vs load (cleanup over a 128-filler vocab) ----
    loads = (4, 24, 64)
    cap = {k: [record_recovery(k, s) for s in SEEDS] for k in loads}

    # ---- structural checks ----
    a, b = hv("A"), hv("B")
    unbind_exact = bind(bind(a, b), a) == b          # bipolar product is exactly self-inverse
    bind_dissimilar = abs(sim(bind(a, b), a)) < 0.15  # a bound pair hides its factors
    permute_roundtrip = permute(permute(a, 1), D - 1) == a

    print("=" * 62)
    print("VSA / HDC reasoning (dense HD vectors) -- neuromorphic-adjacent, NOT spike-timing")
    print("=" * 62)
    print(f"D={D}  seeds={len(list(SEEDS))}  substrate=dense bipolar +/-1 hypervectors")
    print(f"arm algebra: bind(APPROACH,NEAR) -> {near_cmd} | bind(APPROACH,FAR) -> {far_cmd}"
          "  (same gesture, context flips the command)")
    print("\n role-filler record recovery vs load (mean +/- std):")
    for k in loads:
        m, s = mean_std(cap[k])
        print(f"   {k:>2} pairs : {m:>6.1%} +/- {s:>4.1%}")
    print(f"\nstructure: unbind exact={unbind_exact}  bound-pair-hides-factors={bind_dissimilar}"
          f"  permute-roundtrip={permute_roundtrip}")
    print("VSA gives Paradigm C a role/relation algebra (bind/bundle/permute) over dense HD vectors;"
          " the arm interpreter's label->command map becomes compositional. It COMPLEMENTS"
          " paradigm_b_engine, which reasons in genuine spike timing.")
    print("=" * 62)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert unbind_exact, "bind must be self-inverse for bipolar hypervectors"
    assert bind_dissimilar, "a bound pair must be ~orthogonal to its factors"
    assert permute_roundtrip, "permute/unpermute must round-trip"
    assert near_cmd == "GRIP" and far_cmd == "WAIT", \
        f"context did not steer the command: NEAR->{near_cmd} FAR->{far_cmd}"
    lo_m, _ = mean_std(cap[4])
    hi_m, _ = mean_std(cap[64])
    assert lo_m >= THRESH, f"small record should recover cleanly: {lo_m:.2f}"
    assert lo_m > hi_m, "recovery should degrade as the record is overloaded (finite capacity)"
    print("self-check OK: bind/bundle/permute algebra holds, context-steered command, "
          "clean recall at low load degrading with capacity")


if __name__ == "__main__":
    main()
