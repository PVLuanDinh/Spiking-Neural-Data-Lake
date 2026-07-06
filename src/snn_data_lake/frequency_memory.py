"""
Frequency-gated cell-assembly associative memory (FIP/FID) -- stdlib-only, zero deps.

Implements the DESCRIBED mechanism of Lei, Li, Li, Chen, Zhang, Jin & Yang, "How to
form brain-like memory in spiking neural networks with the help of frequency-induced
mechanism", Neurocomputing 623:129361 (2025), DOI 10.1016/j.neucom.2025.129361. The
paper's exact equations are paywalled (ScienceDirect); this follows the ABSTRACT's
account of frequency-induced potentiation (FIP) and depression (FID) plus structural /
weight plasticity that forms recurrent CELL ASSEMBLIES -- it is not a verbatim port.

Mechanism: N recurrent rate-neurons with recurrent weights w[i][j] (start 0, sparse).
Each neuron tracks a firing-frequency EMA of its 0/1 activity. Presenting a PATTERN
drives its neurons active; their frequency rises. FREQUENCY GATES PLASTICITY: a neuron
whose frequency crosses a HIGH threshold enters FIP (excitable + PLASTIC); a low-freq
neuron enters FID (inactive, plasticity FROZEN). Hebbian + structural potentiation
grows w[i][j] ONLY between two co-active FIP neurons (creating the edge from 0) -> the
mutually-potentiated FIP set IS the pattern's cell assembly. FID neurons are frozen,
which protects an already-formed assembly while a LATER pattern is learned (this is the
interference immunity). RECALL seeds a partial cue and lets the potentiated intra-
assembly weights complete the rest of the pattern while non-assembly neurons stay quiet.

Distinct from its core siblings -- headline = frequency GATES plasticity on/off ->
recurrent cell assemblies -> interference-immune recall:
  * bcm_plasticity: BCM scales LTP/LTD MAGNITUDE by post-rate vs a sliding threshold on
    ONE feed-forward unit; it never switches plasticity off and forms no recurrent
    assembly. Here frequency GATES plasticity fully on (FIP) / off (FID).
  * growing_learner: recruits feed-forward PROTOTYPE units (ART neurogenesis); no
    recurrent co-active assembly and no frozen-neuron interference immunity.
  * hippocampus: FIXED random DG expansion + CA3 max-overlap attractor; the assemblies
    here are FORMED by frequency-gated plasticity, not pre-wired.

ponytail: a small rate-based caricature of the paper's FIP/FID + structural plasticity
-- exact paper equations are paywalled, patterns are synthetic, and this is no brain-
scale sparse memory or cognition/AGI. Additive / opt-in: a new member of the memory /
plasticity family; it changes no existing behaviour. Credit: Lei et al. 2025 (DOI
10.1016/j.neucom.2025.129361). Deterministic (seeded; no module-level random.seed; no
builtin hash()).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.frequency_memory
"""
import random


class FrequencyMemory:
    """Recurrent rate-net whose per-neuron firing FREQUENCY gates plasticity (FIP on /
    FID off) to grow interference-immune cell assemblies.

    n:         number of neurons.
    ema:       frequency EMA rate (higher -> frequency tracks activity faster).
    fip_high:  frequency to enter FIP (plastic). fid_low: to enter FID (frozen).
    lr:        Hebbian potentiation step. wmax: recurrent weight ceiling.
    recall_th: recurrent input a silent neuron needs to fire during recall."""

    def __init__(self, n=60, ema=0.35, fip_high=0.5, fid_low=0.25,
                 lr=0.25, wmax=1.0, recall_th=3.0, seed=0):
        self.n = n
        self.ema, self.fip_high, self.fid_low = ema, fip_high, fid_low
        self.lr, self.wmax, self.recall_th = lr, wmax, recall_th
        self.rng = random.Random(seed)                 # all RNG seeded -> deterministic
        self.w = [[0.0] * n for _ in range(n)]         # recurrent weights, start empty
        self.freq = [0.0] * n                          # per-neuron firing-rate EMA

    def pattern(self, size):
        """A random pattern = a sorted list of `size` distinct neuron ids."""
        return sorted(self.rng.sample(range(self.n), size))

    def _drive(self, pat):
        """One presentation: clamp `pat` active, update every neuron's frequency EMA
        from its 0/1 activity, and return the current FIP mask (freq >= fip_high)."""
        on = set(pat)
        for i in range(self.n):
            a = 1.0 if i in on else 0.0
            self.freq[i] += self.ema * (a - self.freq[i])          # EMA toward activity
        return [self.freq[i] >= self.fip_high for i in range(self.n)]

    def states(self):
        """Per-neuron state: 'FIP' (plastic), 'FID' (frozen), '-' (neutral band)."""
        return ["FIP" if f >= self.fip_high else "FID" if f <= self.fid_low else "-"
                for f in self.freq]

    def learn(self, pat, epochs=25):
        """Present `pat` repeatedly. Its neurons' frequency rises into FIP; Hebbian +
        structural potentiation grows w[i][j] between co-active FIP pairs (from 0). FID
        neurons are frozen -> untouched (no depression), which is what preserves earlier
        assemblies. Returns the cell assembly = the FIP neurons that potentiated."""
        assembly = set()
        for _ in range(epochs):
            fip = self._drive(pat)
            active_fip = [i for i in pat if fip[i]]               # co-active AND FIP
            for i in active_fip:
                for j in active_fip:
                    if i != j:
                        self.w[i][j] = min(self.wmax, self.w[i][j] + self.lr)
            assembly.update(active_fip)
        return frozenset(assembly)

    def recall(self, cue, steps=4):
        """Pattern completion: clamp partial `cue` active and settle. A silent neuron
        fires when its recurrent input from currently-active neurons >= recall_th, so
        potentiated intra-assembly weights pull in the rest of the pattern while neurons
        with no assembly weights stay quiet. Returns the settled active id set."""
        clamp = set(cue)
        act = [1 if i in clamp else 0 for i in range(self.n)]
        for _ in range(steps):
            nxt = [0] * self.n
            for i in range(self.n):
                if i in clamp:
                    nxt[i] = 1
                    continue
                inp = sum(self.w[j][i] for j in range(self.n) if act[j])
                if inp >= self.recall_th:
                    nxt[i] = 1
            if nxt == act:
                break
            act = nxt
        return {i for i in range(self.n) if act[i]}

    def mean_weight(self, ids):
        """Mean recurrent weight over ordered off-diagonal pairs within `ids`."""
        ids = sorted(ids)
        pairs = [(i, j) for i in ids for j in ids if i != j]
        return sum(self.w[i][j] for i, j in pairs) / len(pairs) if pairs else 0.0


def main():
    print("=" * 64)
    print("FREQUENCY-GATED CELL-ASSEMBLY MEMORY (FIP/FID) -- Lei et al. 2025")
    print("=" * 64)

    N, M, EP = 60, 12, 25
    k = round(0.4 * M)                                  # partial-cue size (40% of A)

    # 1 FORMATION: present A repeatedly -> a cell assembly forms; the co-active FIP
    #   neurons' mutual recurrent weights strengthen far above baseline.
    m1 = FrequencyMemory(n=N, seed=0)
    A = m1.pattern(M)
    asmA = m1.learn(A, epochs=EP)
    w_in = m1.mean_weight(asmA)
    w_base = m1.mean_weight([i for i in range(N) if i not in set(A)])
    n_fip = sum(1 for s in m1.states() if s == "FIP")
    assert asmA == frozenset(A), "assembly must be exactly the driven FIP neurons"
    assert w_in > 0.8, f"assembly weight must potentiate high (got {w_in:.3f})"
    assert w_in > 20 * max(w_base, 1e-9), \
        f"assembly weight must dwarf baseline ({w_in:.3f} vs {w_base:.3f})"

    # 1b FREQUENCY-GATING (the core FIP claim): an INFREQUENT one-shot pattern stays below
    #    the FIP threshold -> NO potentiation; only a FREQUENT (repeated) pattern crosses
    #    into FIP and forms an assembly. Frequency is the gate, not mere co-activation.
    m1b = FrequencyMemory(n=N, seed=0)
    C = m1b.pattern(M)
    once = m1b.learn(C, epochs=1)                        # one shot: frequency stays sub-FIP
    w_once = m1b.mean_weight(C)
    many = m1b.learn(C, epochs=EP)                       # repeated: crosses FIP -> forms
    w_many = m1b.mean_weight(C)
    assert once == frozenset() and w_once == 0.0, \
        "an infrequent one-shot pattern must NOT potentiate (FIP gate closed)"
    assert many == frozenset(C) and w_many > 0.8, \
        "a frequent repeated pattern must form the assembly (FIP gate opens)"

    # 2 RECOLLECTION: a 40% partial cue of A completes the FULL assembly while non-A
    #   neurons stay quiet (pattern completion / convergence).
    cue = A[:k]
    rec = m1.recall(cue)
    acc = len(rec & set(A)) / M
    spurious = len(rec - set(A))
    assert acc >= 0.8, f"partial cue must recall >=80% of A (got {acc:.2f})"
    assert spurious == 0, f"non-A neurons must stay quiet (got {spurious})"

    # 3 SEPARATION: two DISTINCT patterns -> assemblies with LOW overlap.
    m3 = FrequencyMemory(n=N, seed=0)
    a_pat, b_pat = m3.pattern(M), m3.pattern(M)
    asm_a, asm_b = m3.learn(a_pat, epochs=EP), m3.learn(b_pat, epochs=EP)
    inter, union = len(asm_a & asm_b), len(asm_a | asm_b)
    jacc = inter / union
    assert jacc < 0.4, f"distinct patterns must give distinct assemblies (J={jacc:.2f})"

    # 4 INTERFERENCE IMMUNITY (headline): learn A, then learn B; recall of A STILL works
    #   because FID froze A's (now low-frequency) neurons while B was learned.
    m4 = FrequencyMemory(n=N, seed=0)
    ai, bi = m4.pattern(M), m4.pattern(M)
    m4.learn(ai, epochs=EP)
    cue_a = ai[:k]
    acc_before = len(m4.recall(cue_a) & set(ai)) / M
    a_only = [i for i in ai if i not in set(bi)]        # A neurons frozen while B learns
    w_only_before = m4.mean_weight(a_only)
    m4.learn(bi, epochs=EP)
    acc_after = len(m4.recall(cue_a) & set(ai)) / M
    w_only_after = m4.mean_weight(a_only)
    st = m4.states()
    a_only_fid = sum(1 for i in a_only if st[i] == "FID")
    assert acc_after >= acc_before, \
        f"learning B must NOT degrade recall of A ({acc_before:.2f} -> {acc_after:.2f})"
    assert acc_after >= 0.8, f"A must still recall after learning B (got {acc_after:.2f})"
    assert w_only_after == w_only_before, "A's own assembly weights must be frozen during B"
    assert a_only_fid == len(a_only), "A-only neurons must be FID (frozen) after learning B"

    # 5 DETERMINISM: same seed -> identical patterns, assemblies, weights, and recall.
    d1 = FrequencyMemory(n=N, seed=0)
    ad, bd = d1.pattern(M), d1.pattern(M)
    d2 = FrequencyMemory(n=N, seed=0)
    ad2, bd2 = d2.pattern(M), d2.pattern(M)
    assert ad2 == ad and bd2 == bd, "same seed -> identical sampled patterns"
    fa1, fb1 = d1.learn(ad, epochs=EP), d1.learn(bd, epochs=EP)
    fa2, fb2 = d2.learn(ad2, epochs=EP), d2.learn(bd2, epochs=EP)
    assert fa1 == fa2 and fb1 == fb2, "same seed -> identical assemblies"
    assert d1.w == d2.w, "same seed -> identical recurrent weights"
    assert d1.recall(ad[:k]) == d2.recall(ad2[:k]), "same seed -> identical recall"

    print(f"units n={N}  pattern={M}  epochs={EP}  FIP={n_fip} after A")
    print(f"1 FORMATION    : assembly w={w_in:.3f} vs baseline w={w_base:.3f}")
    print(f"1b FREQ-GATE   : one-shot -> no assembly (w={w_once:.3f}); repeated -> forms (w={w_many:.3f})")
    print(f"2 RECOLLECTION : 40% cue ({k}/{M}) -> recall acc={acc:.2f}, spurious={spurious}")
    print(f"3 SEPARATION   : |A&B|={inter} |AUB|={union} Jaccard={jacc:.2f}")
    print(f"4 INTERFERENCE : recall-A before={acc_before:.2f} after B={acc_after:.2f} "
          f"(A-only FID={a_only_fid}/{len(a_only)})")
    print("5 DETERMINISM  : identical patterns, assemblies, weights, recall")
    print("=" * 64)
    print(f"self-check OK: assembly w={w_in:.3f}>>base {w_base:.3f}, recollection {acc:.2f}, "
          f"separation J={jacc:.2f}, recall-A {acc_before:.2f}->{acc_after:.2f} after B "
          f"(frozen); deterministic")


if __name__ == "__main__":
    main()
