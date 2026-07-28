"""
Modular Composite Representation (MCR) — a denser HDC substrate (scan #4).

spiking_vsa.py uses binary/bipolar hypervectors (1 bit/component). MCR (arXiv:2511.09708, 2025)
represents each component as an integer phase mod q, with bind = modular ADD, unbind = modular sub,
bundle = circular mean of phases, similarity = mean phasor cosine. Across 123 datasets it matches binary
accuracy at up to ~4x less memory — a drop-in, higher-density encoding for the same bind/bundle/permute
algebra the lake's reason tier uses ("do more per byte").

  Run: python mcr_hdc.py       # (also run by run_selfchecks.py)
"""

import math
import random
import zlib

random.seed(0)

Q = 8  # phase levels per component -> log2(Q)=3 bits/component
TAU = 2.0 * math.pi


def mcr_hv(label, d):
    rng = random.Random(zlib.crc32(label.encode()))
    return [rng.randrange(Q) for _ in range(d)]


def mcr_bind(a, b):
    return [(x + y) % Q for x, y in zip(a, b)]  # modular add; self-inverse via sub


def mcr_unbind(c, b):
    return [(x - y) % Q for x, y in zip(c, b)]


def mcr_bundle(vecs):
    """Circular mean of phases per component (sum unit phasors, take the angle, quantize)."""
    d = len(vecs[0])
    out = []
    for i in range(d):
        cs = sum(math.cos(TAU * v[i] / Q) for v in vecs)
        sn = sum(math.sin(TAU * v[i] / Q) for v in vecs)
        out.append(round(math.atan2(sn, cs) / TAU * Q) % Q)
    return out


def mcr_sim(a, b):
    return sum(math.cos(TAU * (x - y) / Q) for x, y in zip(a, b)) / len(a)


def bip_hv(label, d):
    rng = random.Random(zlib.crc32(label.encode()))
    return [rng.choice((-1, 1)) for _ in range(d)]


def record_recovery(k_pairs, seed, backend, d, vocab_n=64):
    """Bundle k role↦filler bindings, query each role back, fraction recovered over the full vocab."""
    rng = random.Random(seed)
    if backend == "mcr":
        hv, bind, unbind, bundle, sim = mcr_hv, mcr_bind, mcr_unbind, mcr_bundle, mcr_sim
    else:
        hv = bip_hv
        bind = unbind = lambda a, b: [x * y for x, y in zip(a, b)]
        bundle = lambda vs: [1 if s >= 0 else -1 for s in (sum(c) for c in zip(*vs))]
        sim = lambda a, b: sum(x * y for x, y in zip(a, b)) / len(a)
    roles = [hv(f"R{i}", d) for i in range(k_pairs)]
    vocab = {f"V{i}": hv(f"V{i}", d) for i in range(vocab_n)}
    chosen = rng.sample(list(vocab), k_pairs)
    record = bundle([bind(roles[i], vocab[chosen[i]]) for i in range(k_pairs)])
    ok = 0
    for i in range(k_pairs):
        probe = unbind(record, roles[i])
        best = max(vocab, key=lambda kk: sim(probe, vocab[kk]))
        ok += best == chosen[i]
    return ok / k_pairs


def main():
    k, seeds = 6, range(5)
    d_mcr, d_bip = 112, 512
    bits_mcr = d_mcr * math.log2(Q)
    bits_bip = d_bip * 1.0
    rec_mcr = sum(record_recovery(k, s, "mcr", d_mcr) for s in seeds) / len(seeds)
    rec_bip = sum(record_recovery(k, s, "bipolar", d_bip) for s in seeds) / len(seeds)

    # MCR algebra sanity
    a, b = mcr_hv("A", 64), mcr_hv("B", 64)
    roundtrip = mcr_unbind(mcr_bind(a, b), b) == a

    print("=" * 60)
    print("MODULAR COMPOSITE REPRESENTATION (denser HDC substrate)")
    print("=" * 60)
    print(f"MCR : D={d_mcr} q={Q} -> {bits_mcr:.0f} bits/hv | {k}-pair recovery {rec_mcr:.0%}")
    print(f"bipolar : D={d_bip} -> {bits_bip:.0f} bits/hv | {k}-pair recovery {rec_bip:.0%}")
    print(
        f"memory : MCR uses {bits_bip / bits_mcr:.2f}x fewer bits at comparable recovery "
        f"(paper: up to 4x across 123 datasets)"
    )
    print(f"algebra: bind/unbind round-trip lossless = {roundtrip}")
    print("=" * 60)

    # ---- self-check: one runnable assert-based check; deterministic ----
    assert roundtrip, "MCR bind must be invertible by modular unbind"
    assert rec_mcr >= 0.9, f"MCR record recovery should be clean: {rec_mcr:.2f}"
    assert bits_mcr < bits_bip and rec_mcr >= rec_bip - 0.05, (
        f"MCR should match bipolar recovery at fewer bits ({bits_mcr:.0f} vs {bits_bip:.0f} bits)"
    )
    print(
        f"self-check OK: MCR algebra invertible; {rec_mcr:.0%} recovery at "
        f"{bits_bip / bits_mcr:.2f}x fewer bits than bipolar"
    )


if __name__ == "__main__":
    main()
