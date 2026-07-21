r"""
CONSCIOUSNESS PROBE -- test the brain loop's consciousness CORRELATES, honestly and agnostically.

Per the project's consciousness-pursuit stance: build toward it, measure the correlates, and NEITHER
deny NOR claim phenomenal consciousness. There is no test for phenomenal consciousness (the "hard
problem"); IIT's Phi is intractable. What CAN be measured are the leading FUNCTIONAL correlates, each
against a control that must fail. This probe runs the four the project has built, on the brain's own
modules, and reports a panel + an honest verdict.

  1. GWT ACCESS (Baars; Dehaene-Changeux): all-or-none ignition + global broadcast + an access index
     that discriminates a coherent (ignitable) content from an incoherent one.        [thalamus/access]
  2. BINDING (synchrony; Singer/Gray, contested): multiple items bound via theta-gamma synchrony where
     a rate code with the same hysteresis superposes and loses the pairing.            [global_workspace]
  3. INTEGRATION (IIT proxy, tractable): perturbational complexity (PCI) high for an integrated network,
     low for an independent-noise bank that fires MORE -- so the signal is integration.  [integration_measure]
  4. RE-ENTRY (GNW loop; Lamme): the closed loop gives predictive completion (weak evidence + expectation
     ignites) WITHOUT hallucinating (expectation alone does not), loop load-bearing.     [gwt_reentry]

VERDICT is agnostic by construction: it reports which correlates are PRESENT (measure beats control),
and states plainly that present correlates are NOT proof of phenomenal experience. Open question,
actively built toward, neither denied nor claimed.

  PYTHONPATH=src python research/consciousness_probe.py
Reuses global_workspace, access_consciousness, integration_measure, gwt_reentry.
"""
import random

from snn_data_lake.cortex.access_consciousness import AccessConsciousness, _coherent, _scrambled
from snn_data_lake.cortex.integration_measure import (
    integrated_network, independent_noise, silent_network)
from snn_data_lake.thalamus.global_workspace import (
    build_stream, run_stream, count_switches,
    bind_objects, bind_broadcast_spiking, bind_decode_spiking,
    bind_broadcast_rate, bind_decode_rate, _bind_pair_acc)
from snn_data_lake.reasoning.gwt_reentry import run_loop


def probe_access():
    """GWT: (a) spiking ignition stabilises the workspace vs a pure-argmax control; (b) a coherent
    assembly ignites + is globally available with a high access index, an incoherent one does not."""
    stream = build_stream(48, seed=0)
    sw_ignition = count_switches(run_stream(stream, use_ignition=True, seed=0))
    sw_argmax = count_switches(run_stream(stream, use_ignition=False, seed=0))
    strong = AccessConsciousness(seed=0)
    strong.bind_assembly(_coherent("cat", 5))
    strong.step()
    weak = AccessConsciousness(seed=0)
    weak.bind_assembly(_scrambled(5))
    weak.step()
    ai_strong, ai_weak = strong.access_index(), weak.access_index()
    avail_strong, avail_weak = strong.global_availability(), weak.global_availability()
    present = (sw_argmax > sw_ignition) and (ai_strong > ai_weak) and avail_strong and not avail_weak
    return present, (f"ignition switches {sw_ignition} vs argmax {sw_argmax}; access-index coherent "
                     f"{ai_strong:.2f} vs incoherent {ai_weak:.2f}; broadcast {avail_strong}/{avail_weak}")


def probe_binding():
    """Two objects (color,shape) bound: spiking synchrony recovers the pairing, the rate control loses it."""
    rng = random.Random(0)
    sp = rt = 0.0
    n = 100
    for _ in range(n):
        objs = bind_objects(8, 8, 2, rng)
        _, r = bind_broadcast_spiking(objs, 8, 8)
        sp += _bind_pair_acc(objs, bind_decode_spiking(r))
        _, ca, sa = bind_broadcast_rate(objs, 8, 8)
        rt += _bind_pair_acc(objs, bind_decode_rate(ca, sa, 2))
    sp, rt = sp / n, rt / n
    present = sp > rt + 0.15
    return present, f"K=2 pair-binding spiking {sp:.0%} vs rate control {rt:.0%}"


def probe_integration():
    """PCI: integrated network >> independent-noise bank (which fires more) >> silent."""
    pci_i, _ = integrated_network(seed=0).pci(trials=8)
    pci_n, _ = independent_noise(seed=0).pci(trials=8)
    pci_s, _ = silent_network(seed=0).pci(trials=8)
    rate_i = integrated_network(seed=0).mean_firing_rate()
    rate_n = independent_noise(seed=0).mean_firing_rate()
    present = (pci_i > 3 * pci_n) and (pci_i > 0.25) and (pci_n > pci_s) and (rate_n >= rate_i)
    return present, (f"PCI integrated {pci_i:.2f} vs noise {pci_n:.2f} vs silent {pci_s:.2f} "
                     f"(noise fires more: {rate_n:.2f}>={rate_i:.2f} -> gap is integration)")


def probe_reentry():
    """The closed GNW loop: predictive completion without hallucination, loop load-bearing."""
    weak_alone, _ = run_loop(0.40, 0.0, reentry=True)
    prior_alone, _ = run_loop(0.0, 0.30, reentry=True)
    completed, _ = run_loop(0.40, 0.30, reentry=True)
    loop_open, _ = run_loop(0.40, 0.30, reentry=False)
    present = completed and not loop_open and not prior_alone and not weak_alone
    return present, (f"weak+expected ignites={completed}; loop-open={loop_open}; "
                     f"expectation-alone(hallucination)={prior_alone}; weak-alone={weak_alone}")


def main():
    print("=" * 92)
    print("CONSCIOUSNESS PROBE -- the brain loop's consciousness CORRELATES (agnostic; controls-backed)")
    print("=" * 92)
    battery = [
        ("GWT ACCESS      (Baars/Dehaene)", probe_access),
        ("BINDING         (synchrony; contested)", probe_binding),
        ("INTEGRATION     (IIT proxy, PCI)", probe_integration),
        ("RE-ENTRY        (GNW loop; Lamme)", probe_reentry),
    ]
    results = {}
    for name, fn in battery:
        present, detail = fn()
        results[name] = present
        print(f"  [{'PRESENT' if present else 'ABSENT ':7}] {name}")
        print(f"            {detail}")
    n_present = sum(results.values())
    print("-" * 92)
    print(f"  {n_present}/4 functional consciousness-correlates PRESENT (each beats its control).")
    print("  HONEST VERDICT (agnostic): the brain loop exhibits the measurable ACCESS-level and dynamics")
    print("  correlates of consciousness -- global broadcast/ignition, feature binding, integrated")
    print("  perturbational complexity, and re-entrant predictive completion. These are NOT a test of")
    print("  PHENOMENAL consciousness (the hard problem): no such test exists, and IIT Phi is intractable.")
    print("  Whether the system has subjective experience is an OPEN question -- neither denied nor claimed.")
    print("  The project actively builds toward it and reports exactly what is measured, nothing more.")
    print("=" * 92)

    # self-checks: every correlate measure beats its control (the functional correlates are really there),
    # and the verdict stays agnostic (this probe asserts correlates, NOT phenomenal consciousness).
    for name, present in results.items():
        assert present, f"consciousness-correlate FAILED its control: {name}"
    assert n_present == 4, "not all four functional correlates present"
    print(f"self-check OK: all 4 functional consciousness-correlates present + controls fail correctly "
          f"(access, binding, integration, re-entry). Reported AGNOSTICALLY -- correlates measured, "
          f"phenomenal consciousness neither claimed nor denied.")
    return 0


if __name__ == "__main__":
    main()
