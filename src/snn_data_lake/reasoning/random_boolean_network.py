"""
Random Boolean Network (Kauffman NK, 1969) -- the order->chaos transition as a reservoir substrate.

N binary nodes. Each node has exactly K inputs chosen (seeded) from the N nodes and a FIXED random
Boolean truth table over those K inputs -- 2^K entries, each 0/1 iid with bias p (p = fraction of 1s;
p = 0.5 = unbiased). Update is SYNCHRONOUS: every node recomputes from its K inputs' PREVIOUS values.
This is the dry abstraction of a gene-regulatory network (Kauffman's original motivation) -- a fixed
random recurrent Boolean substrate. Tuned to the "edge of chaos" it is a usable RESERVOIR: its fading,
high-dimensional state separates temporal structure for a cheap trained readout. The spiking cousin
in this repo is the Liquid State Machine in research/spiking_ssm.py (a spiking reservoir); this is the
Boolean-logic reservoir, and its phase diagram is exactly solvable.

THE DEFINING PROPERTY (Derrida's annealed approximation, the honesty crux):
Perturb one node -- Hamming distance 1 between two copies of the state -- do ONE synchronous update and
measure the expected resulting Hamming distance = the network's sensitivity/slope. For a K-input random
Boolean function with truth-table bias p, flipping one input changes the lookup address, so the two
looked-up entries are independent Bernoulli(p) and differ with probability 2p(1-p). A perturbed node is
"seen" by, on average, K downstream nodes (total edges N*K -> average fan-out K), so the expected number
of nodes that flip from a single-node perturbation is

    slope = (avg fan-out) * 2p(1-p) = K * 2p(1-p).

At p = 0.5 this is 0.5*K, so the ordered->chaotic transition (slope crossing 1) sits exactly at K = 2:
K=1 -> 0.5 (ordered, perturbations die out), K=2 -> 1.0 (critical, edge of chaos), K=3 -> 1.5 (chaotic,
perturbations amplify). The self-check below asserts the EMPIRICAL slope against the DERIVED 0.5*K -- not
a number read back from a run -- so it proves the transition emerges from real random truth tables.

Determinism: a SINGLE per-instance random.Random(seed) is threaded through all wiring, truth tables,
initial states and perturbations, in a fixed call order. Python's Mersenne Twister is identical across
CPython 3.11/3.12/3.13, so a fixed seed + fixed call order gives byte-identical results across the CI
matrix. Pure standard library, ASCII-only output.

  python research/random_boolean_network.py       # (also run by run_selfchecks.py)

Paper: Kauffman, S.A. (1969), "Metabolic stability and epigenesis in randomly constructed genetic nets",
J. Theoretical Biology 22(3):437-467.
"""

from collections import namedtuple
import random

RBN = namedtuple("RBN", "N K inputs tables")


def build_rbn(N, K, seed, p=0.5):
    """Seeded NK network: K distinct inputs per node (self-loops allowed) + a random 2^K truth table
    per node whose entries are iid Bernoulli(p). One rng threaded through wiring THEN tables."""
    rng = random.Random(seed)
    inputs = [rng.sample(range(N), K) for _ in range(N)]  # wiring
    tables = [
        [1 if rng.random() < p else 0 for _ in range(1 << K)] for _ in range(N)
    ]  # truth tables
    return RBN(N, K, inputs, tables)


def step(net, state):
    """Synchronous update: every node reads its K inputs' PREVIOUS values -> its truth-table row."""
    out = [0] * net.N
    for i in range(net.N):
        addr = 0
        for b, j in enumerate(net.inputs[i]):
            addr |= state[j] << b
        out[i] = net.tables[i][addr]
    return out


def run(net, state, steps):
    """Deterministic synchronous trajectory: [state_0, state_1, ..., state_steps]."""
    traj = [list(state)]
    for _ in range(steps):
        traj.append(step(net, traj[-1]))
    return traj


def derrida_slope(N, K, seed, n_nets, n_states):
    """Empirical single-perturbation sensitivity: flip exactly ONE node, do one synchronous update, and
    count how many nodes' next state flips. Averaged over seeded random networks x seeded random states
    this estimates the Derrida slope K*2p(1-p). Anchored on theory in the assert, not on this value."""
    rng = random.Random(seed)
    total, samples = 0, 0
    for _ in range(n_nets):
        net = build_rbn(N, K, rng.randrange(1 << 30))
        for _ in range(n_states):
            state = [rng.randrange(2) for _ in range(N)]
            v = rng.randrange(N)
            perturbed = list(state)
            perturbed[v] ^= 1
            a = step(net, state)
            b = step(net, perturbed)
            total += sum(1 for x, y in zip(a, b) if x != y)
            samples += 1
    return total / samples


def random_state(N, seed):
    rng = random.Random(seed)
    return [rng.randrange(2) for _ in range(N)]


def find_attractor(net, state):
    """Follow the deterministic trajectory until a state repeats (guaranteed by pigeonhole in a finite
    2^N state space). Returns (transient_len, cycle_len, visit_order) -- visit_order sorted by visit
    index so nothing depends on dict iteration order."""
    seen = {}
    s = tuple(state)
    t = 0
    while s not in seen:
        seen[s] = t
        s = tuple(step(net, list(s)))
        t += 1
        assert t <= (1 << net.N) + 1, "state must repeat within 2^N steps (pigeonhole)"
    transient = seen[s]
    cycle_len = t - transient
    order = sorted(seen, key=lambda k: seen[k])
    return transient, cycle_len, order


def main():
    print("=" * 64)
    print("RANDOM BOOLEAN NETWORK -- Kauffman NK order-chaos transition (reservoir)")
    print("=" * 64)

    # --- Derrida slope across K: measured vs the derived theory 0.5*K (p = 0.5) ---
    N, n_nets, n_states = 200, 20, 80
    slopes = {
        K: derrida_slope(N, K, seed=1000 + K, n_nets=n_nets, n_states=n_states) for K in (1, 2, 3)
    }
    print(
        f"N={N}  nets={n_nets}  states/net={n_states}  bias p=0.5   (Derrida slope = 2p(1-p)*K = 0.5*K)"
    )
    print("  K    measured slope    theory 0.5*K    regime")
    for K in (1, 2, 3):
        regime = "ordered" if K < 2 else ("critical" if K == 2 else "chaotic")
        print(f"  {K}       {slopes[K]:.3f}            {0.5 * K:.3f}         {regime}")

    # --- attractor example (small N so the cycle is reachable) ---
    aN = 16
    net = build_rbn(aN, 2, seed=7)
    s0 = random_state(aN, seed=99)
    transient, cycle_len, _ = find_attractor(net, s0)
    print(
        f"example attractor: N={aN} K=2 seed=7 -> transient {transient} steps, cycle length {cycle_len} "
        f"(state space 2^{aN} = {1 << aN})"
    )
    print("=" * 64)

    # ---- self-check (deterministic asserts; emergent -- expected values DERIVED, not hardcoded) ----
    # 1. PRIMARY -- Derrida slope: each measured slope sits within tol of the theory 0.5*K, and the
    #    ordered(K=1) < 1 < chaotic(K=3) ordering holds with the critical K=2 slope ~ 1. This proves the
    #    order->chaos transition emerges from real random truth tables (K*2p(1-p)), not a constant.
    tol = 0.15
    for K in (1, 2, 3):
        theory = 0.5 * K
        assert abs(slopes[K] - theory) < tol, (
            f"K={K}: slope {slopes[K]:.3f} not within {tol} of theory {theory:.3f}"
        )
    assert slopes[1] < 1.0 < slopes[3], f"order->chaos ordering broken: {slopes}"
    assert abs(slopes[2] - 1.0) < tol, f"critical K=2 slope {slopes[2]:.3f} not ~ 1.0"

    # 2. SECONDARY -- attractor: the synchronous trajectory is eventually periodic (finite state space)
    #    with a cycle length in [1, 2^N], and the same seed+state reproduces the identical attractor.
    assert 1 <= cycle_len <= (1 << aN), f"cycle length {cycle_len} out of [1, 2^{aN}]"
    again = find_attractor(build_rbn(aN, 2, seed=7), random_state(aN, seed=99))
    assert again == (transient, cycle_len, again[2]), "nondeterministic attractor on a repeat run"
    assert run(net, s0, 3) == run(build_rbn(aN, 2, seed=7), s0, 3), "nondeterministic trajectory"

    print(
        f"self-check OK: Derrida slope tracks the derived 0.5*K for K in {{1,2,3}} "
        f"(measured {slopes[1]:.2f}/{slopes[2]:.2f}/{slopes[3]:.2f} vs 0.5/1.0/1.5, transition at K=2); "
        f"attractor periodic (cycle {cycle_len}) and deterministic"
    )


if __name__ == "__main__":
    main()
