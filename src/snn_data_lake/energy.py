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
the idealised CMOS AC — its win amortises with scale and sparsity. Reported honestly, not hidden.

  Run: python energy.py       # (also run by run_selfchecks.py)
"""

# ---- cited per-operation energies (Joules) -------------------------------------
E_MAC_FP32_45NM = 4.6e-12   # dense ANN multiply-accumulate (Horowitz 2014)
E_AC_FP32_45NM = 0.9e-12    # spiking accumulate-only op        (Horowitz 2014)
E_SYNOP_LOIHI = 23.6e-12    # per-SynOp on Loihi, incl. overhead (arXiv:2210.13107)


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
    print("self-check OK: SynOps -> nJ/inference; spiking beats dense ANN; ratio = ops x MAC/AC")


if __name__ == "__main__":
    main()
