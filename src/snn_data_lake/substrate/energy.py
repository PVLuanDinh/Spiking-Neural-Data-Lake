"""
Energy estimate: turn measured SynOps into Joules/inference vs a dense-ANN baseline.

The prototypes across this repo already MEASURE synaptic operations (SynOps) and report
op-count ratios ("3.6-83x fewer ops"). This converts those into an estimated ENERGY figure —
the number reviewers and an INRC/neuromorphic proposal actually ask for. It is analytical, not
measured silicon: multiply an op count by a cited per-op energy. Idea borrowed from
SpikingJelly's compute_energy/synop counters, kept pure-stdlib here (zero deps). The NeuroBench
(arXiv:2304.04640, Nat.Comms 2025) metric definitions — effective ACs, activation sparsity, model
footprint — are exposed too, so the efficiency claims are externally comparable (leaderboard-legible).

Model (the standard SNN-vs-ANN energy argument):
  - a dense ANN does a multiply-accumulate (MAC) per connection;
  - a spiking net does only an accumulate (AC) per SynOp — no multiply — and only when a
    spike arrives (event-driven sparsity is already in the SynOp count).
So E_ann = MACs x E_MAC  and  E_snn = SynOps x E_AC, and the energy win is the op-count win
TIMES the MAC/AC energy asymmetry.

Constants (45 nm CMOS, Horowitz, "Computing's Energy Problem", ISSCC 2014):
  32-bit FP MAC = 3.7 pJ (mult) + 0.9 pJ (add) = 4.6 pJ ; 32-bit FP add (AC) = 0.9 pJ.
A neuromorphic-silicon reference point (per-SynOp on Loihi, incl. routing overhead) is offered
as a second lens: ~23.6 pJ (An Analytical Estimation of SNN Energy Efficiency, arXiv:2210.13107).
At system scale, Intel's Hala Point (1,152 Loihi 2 chips) reports ~15 TOPS/W INT8.
NOTE: the Loihi per-SynOp figure bundles fixed routing/overhead, so at TINY models it can exceed
the idealised CMOS AC -- its win amortises with scale and sparsity. Reported honestly, not hidden.

HONESTY -- what a compute-only "SynOps x E_AC" number leaves out (each shrinks or reverses the win):
  1. analytical, not measured silicon (already flagged above);
  2. the Loihi per-SynOp overhead amortises only at scale (already flagged above);
  3. MEMORY ACCESS: fetching each weight/membrane value costs ~5 pJ on-chip SRAM (up to 640 pJ off-chip
     DRAM) -- often MORE than the accumulate itself. snn_energy_full() adds it; the SynOps count omits it.
  4. INT8 BASELINE: the fair ANN is a quantized INT8 net (0.23 pJ/MAC), not FP32 (4.6 pJ/MAC). Comparing
     spiking AC against an FP32 MAC overstates the asymmetry; ann_energy_int8() is the head-to-head baseline.
  5. PER-TIMESTEP NEURON UPDATE: every neuron reads+writes membrane state each of T timesteps even when
     it does not spike -- absent from a pure SynOps count and growing with T.
  6. OP-COUNT BASELINE: the "3.6x fewer ops" charges the dense ANN N*C MACs on EVERY of STEPS timesteps
     (snn_classifier: dense_macs = n*STEPS*N*C). A static image needs ONE dense matvec, so a fair
     single-matvec baseline is STEPS-x smaller -- against which the SNN does MORE ops, not fewer.
So the SNN wins only once spiking sparsity is below crossover_sparsity() -- see main().

  Run: python energy.py       # (also run by run_selfchecks.py)
"""

# ---- cited per-operation energies (Joules) -------------------------------------
E_MAC_FP32_45NM = 4.6e-12   # dense ANN multiply-accumulate (Horowitz 2014)
E_AC_FP32_45NM = 0.9e-12    # spiking accumulate-only op        (Horowitz 2014)
E_SYNOP_LOIHI = 23.6e-12    # per-SynOp on Loihi, incl. overhead (arXiv:2210.13107)

# ---- memory-access + quantized-baseline energies (what a compute-only SynOps count omits) -------
# Horowitz, "Computing's Energy Problem", ISSCC 2014, 45 nm CMOS:
E_MAC_INT8_45NM = 0.23e-12   # 8-bit integer MAC: 0.2 pJ (mult) + 0.03 pJ (add)
E_MEM_SRAM32_45NM = 5.0e-12  # 32-bit on-chip SRAM read; grows with array size, so this is a floor
E_MEM_DRAM32_45NM = 640e-12  # 32-bit off-chip DRAM read (dominates whenever weights spill off-chip)


def ann_energy(macs, e_mac=E_MAC_FP32_45NM):
    """Dense-ANN inference energy (J) = MAC count x per-MAC energy."""
    return macs * e_mac


def snn_energy(synops, e_op=E_AC_FP32_45NM):
    """Spiking inference energy (J) = SynOps x per-op energy (accumulate-only by default)."""
    return synops * e_op


def compare(synops, dense_macs, e_op=E_AC_FP32_45NM, e_mac=E_MAC_FP32_45NM):
    """Estimate spiking vs dense-ANN energy for one inference. Returns a report dict:
    ops_ratio = MACs/SynOps (what the repo already prints); energy_ratio folds in the
    MAC/AC asymmetry, so energy_ratio = ops_ratio x (e_mac/e_op)."""
    e_snn = snn_energy(synops, e_op)
    e_ann = ann_energy(dense_macs, e_mac)
    return {
        "synops": synops,
        "dense_macs": dense_macs,
        "snn_nj": e_snn * 1e9,
        "ann_nj": e_ann * 1e9,
        "ops_ratio": dense_macs / synops,
        "energy_ratio": e_ann / e_snn,
    }


# ---- honest energy: add memory access + a quantized INT8 ANN baseline + the sparsity crossover ----
def snn_energy_full(synops, neuron_updates, e_op=E_AC_FP32_45NM, e_mem=E_MEM_SRAM32_45NM):
    """Spiking energy WITH the terms snn_energy()'s compute-only SynOps count omits.
        total = SynOps*e_op                       (spike-driven AC compute)
              + SynOps*e_mem                      (one weight fetch per SynOp)
              + neuron_updates*(2*e_mem + e_op)   (per neuron-timestep: read+write membrane state + 1 AC)
    with neuron_updates = n_neurons * T. On real hardware the memory-access term dominates, so this can
    far exceed the SynOps*e_op figure -- omitting it is what makes a compute-only SNN win look large."""
    return synops * e_op + synops * e_mem + neuron_updates * (2 * e_mem + e_op)


def ann_energy_int8(macs, e_mem=E_MEM_SRAM32_45NM):
    """Fair quantized (INT8) ANN baseline: macs*E_MAC_INT8_45NM + macs*e_mem (one weight fetch per MAC).
    Papers that report SNN-vs-ANN energy honestly compare against this INT8 ANN, not an FP32 one, so the
    MAC/AC asymmetry (4.6 vs 0.9 pJ) does not do all the work for the SNN."""
    return macs * E_MAC_INT8_45NM + macs * e_mem


def crossover_sparsity(connections, n_neurons, T, e_mem=E_MEM_SRAM32_45NM):
    """Average spike-probability s (fraction of neuron-slots that spike per timestep) BELOW which the
    SNN's FULL energy beats the INT8 ANN's. Solve, for s,
        snn_energy_full(SynOps=connections*T*s, neuron_updates=n_neurons*T) == ann_energy_int8(connections)
    and clamp to [0, 1]. This is the analytically-derived form of the literature's ~6.4% sparsity
    threshold (arXiv:2409.08290, "Reconsidering the Energy Efficiency of SNN Inference"): above it the
    spiking activity costs more (compute + memory) than doing the dense INT8 matvec once."""
    e_op = E_AC_FP32_45NM
    ann = connections * E_MAC_INT8_45NM + connections * e_mem
    fixed = n_neurons * T * (2 * e_mem + e_op)     # neuron-update floor, independent of s
    per_s = connections * T * (e_op + e_mem)       # SynOps energy per unit spike-probability
    if per_s <= 0:
        return 0.0
    s = (ann - fixed) / per_s
    return min(1.0, max(0.0, s))


# ---- NeuroBench-style metrics (arXiv:2304.04640) — definitions only, no torch harness ----
def activation_sparsity(active_slots, total_slots):
    """Fraction of neuron-activation slots that did NOT spike (higher = sparser = cheaper)."""
    return 1.0 - active_slots / total_slots


def footprint_bytes(n_params, bytes_per_param=4):
    """Model memory footprint = parameter count x bytes/param (NeuroBench 'footprint')."""
    return n_params * bytes_per_param


def neurobench(synops, dense_macs, n_params):
    """Leaderboard-legible summary: effective ACs (= the measured SynOps), the dense Eff-MAC
    baseline, and footprint. Adopts the NeuroBench definitions; leaves out its torch harness."""
    return {
        "eff_acs": synops,                # spiking effective accumulate ops = measured SynOps
        "eff_macs_dense": dense_macs,      # dense-ANN multiply-accumulate baseline
        "footprint_bytes": footprint_bytes(n_params),
    }


def main():
    # real measured numbers from this repo's supervised spiking classifier (snn_classifier.py):
    # 113,676 SynOps vs a 409,600-MAC dense ANN of the same shape.
    synops, dense_macs = 113_676, 409_600
    r = compare(synops, dense_macs)
    loihi_nj = snn_energy(synops, E_SYNOP_LOIHI) * 1e9

    print("=" * 60)
    print("ENERGY ESTIMATE  (SynOps -> Joules/inference vs dense ANN)")
    print("=" * 60)
    print(f"supervised classifier: {synops:,} SynOps  vs  {dense_macs:,} dense MACs")
    print(f"  op-count reduction         : {r['ops_ratio']:.1f}x  (what the repo already reports)")
    print("  45 nm CMOS (Horowitz 2014):")
    print(f"    dense ANN  (MAC 4.6 pJ)  : {r['ann_nj']:.1f} nJ/inference")
    print(f"    spiking    (AC 0.9 pJ)   : {r['snn_nj']:.1f} nJ/inference")
    print(f"    energy reduction         : {r['energy_ratio']:.1f}x  "
          f"(= {r['ops_ratio']:.1f}x ops x {E_MAC_FP32_45NM / E_AC_FP32_45NM:.1f}x MAC/AC)")
    print(f"  Loihi lens (SynOp 23.6 pJ) : {loihi_nj:.1f} nJ  (incl. routing overhead;")
    print("                               its win amortises at scale + sparsity, not this toy size)")
    nb = neurobench(synops, dense_macs, n_params=4 * 64)          # classifier: 4 classes x 64 features
    print(f"  NeuroBench   : Eff_ACs {nb['eff_acs']:,} vs dense Eff_MACs {nb['eff_macs_dense']:,};"
          f" footprint {nb['footprint_bytes'] / 1024:.1f} KB  (defs: arXiv:2304.04640)")

    # ---- honest picture: memory access + a FAIR INT8 ANN baseline + the sparsity crossover ----
    # single-inference shape of this classifier: connections = N*C weights, C class neurons hold
    # membrane state, T = STEPS timesteps (N=64 inputs, C=4 classes, STEPS=20; 80-sample test set).
    samples, n_class_neurons, T = 80, 4, 20
    connections = samples * 64 * 4                    # N*C weights, one dense INT8 matvec per sample
    neuron_updates = samples * n_class_neurons * T    # membrane read/write per class neuron per timestep
    e_snn_full = snn_energy_full(synops, neuron_updates) * 1e9
    e_ann_int8 = ann_energy_int8(connections) * 1e9
    s_cross = crossover_sparsity(connections, samples * n_class_neurons, T)
    s_obs = synops / dense_macs                       # observed avg synapse-timestep spike fraction
    print("  honest picture (memory access + fair INT8 baseline included):")
    print(f"    op count is baseline-dependent: dense_macs={dense_macs:,} charges the ANN N*C MACs on EVERY")
    print(f"      of STEPS={dense_macs // connections} timesteps; a static image needs ONE matvec, so the fair dense")
    print(f"      count is {connections:,} -- against it the {synops:,} SynOps are {synops / connections:.1f}x MORE ops, not fewer.")
    print(f"    spiking WITH memory      : {e_snn_full:.1f} nJ  "
          f"(compute-only was {r['snn_nj']:.1f} nJ; weight+state fetches dominate)")
    print(f"    INT8 ANN  (fair baseline): {e_ann_int8:.1f} nJ  (0.23 pJ MAC + 5 pJ SRAM fetch)")
    print(f"    crossover sparsity       : {s_cross:.1%}  (SNN wins only BELOW it; observed {s_obs:.1%})")
    print("    => with memory + an INT8 baseline counted, the compute-only win shrinks or reverses")
    print("       unless spiking sparsity is below the crossover (here observed activity is well above).")
    print("=" * 60)

    # ---- self-check: one runnable assert-based check ----
    assert r["snn_nj"] < r["ann_nj"], "spiking energy should beat dense ANN at 45 nm"
    assert nb["eff_acs"] == synops and nb["footprint_bytes"] > 0, "NeuroBench metrics wrong"
    assert activation_sparsity(1, 4) == 0.75, "activation-sparsity definition wrong"
    exp = r["ops_ratio"] * (E_MAC_FP32_45NM / E_AC_FP32_45NM)
    assert abs(r["energy_ratio"] - exp) < 1e-6, "energy_ratio must be ops_ratio x MAC/AC asymmetry"
    assert r["energy_ratio"] > r["ops_ratio"], "energy win must exceed the op-count win (MAC > AC)"
    assert ann_energy(0) == 0.0 and snn_energy(0) == 0.0, "zero ops -> zero energy"
    assert loihi_nj > 0, "Loihi estimate must be positive"
    # honest additions: memory access makes the SNN cost MORE than the compute-only figure...
    assert e_snn_full > r["snn_nj"], "with-memory SNN energy must exceed the compute-only figure"
    # ...the crossover is a real interior threshold...
    assert 0.0 < s_cross < 1.0, f"crossover sparsity must lie in (0,1), got {s_cross}"
    # ...and it actually separates the regimes: below it the SNN full energy beats the INT8 ANN, above not.
    below, above = s_cross * 0.5, min(1.0, s_cross * 2.0)
    e_below = snn_energy_full(connections * T * below, neuron_updates)
    e_above = snn_energy_full(connections * T * above, neuron_updates)
    e_ann_ref = ann_energy_int8(connections)
    assert e_below < e_ann_ref, "below crossover: SNN full energy must beat the INT8 ANN"
    assert e_above >= e_ann_ref, "above crossover: SNN full energy must NOT beat the INT8 ANN"
    print("self-check OK: SynOps -> nJ/inference; spiking beats FP32 ANN on compute; with memory + INT8"
          f" baseline the win reverses above the {s_cross:.1%} sparsity crossover")


if __name__ == "__main__":
    main()
