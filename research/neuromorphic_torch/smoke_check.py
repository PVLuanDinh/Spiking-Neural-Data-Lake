"""
CPU-only environment smoke check for the neuromorphic_torch subproject.

Verifies the torch/SpikingJelly/cupy environment and prints the phase map. Skips cleanly if a
dependency is absent (prints NOT available), so it can never fail hard.

This subproject is not run by run_selfchecks.py and not part of the CI matrix. Under P4c that is
DEBT to be closed, not a property to state approvingly -- "a build the gates cannot see is not a
build the project can claim". The blocker is real and worth naming so it is actionable: the four
phase smokes need SpikingJelly, and CI installs `-e .` plus `[dev]` only, so registering them today
would turn the gate RED rather than SKIP. Run it manually inside .venv-neuro until that is fixed.

  python smoke_check.py
"""


def _probe(name):
    try:
        mod = __import__(name)
        return mod
    except Exception as exc:  # noqa: BLE001 - report any import failure, don't crash the smoke check
        print(f"  {name:<14}: NOT available ({type(exc).__name__})")
        return None


def main():
    print("=" * 66)
    print("neuromorphic_torch - environment smoke check")
    print("=" * 66)
    torch = _probe("torch")
    sj = _probe("spikingjelly")
    cupy = _probe("cupy")
    if torch is not None:
        cuda = torch.cuda.is_available()
        print(f"  torch         : {torch.__version__}  (cuda_available={cuda})")
    if sj is not None:
        print(f"  spikingjelly  : {getattr(sj, '__version__', 'installed')}")
    if cupy is not None:
        print("  cupy          : installed (SpikingJelly cupy backend available)")
    else:
        print("  cupy          : absent -> CPU backend fallback (fine for smoke tests)")

    print("-" * 66)
    print("phase map:")
    print("  P1 (in the core): snn_data_lake.spike_preprocessing (van_rossum, encode_latency,")
    print("                    victor_purpura) + snn_data_lake.proximity_index")
    print("  P2: encoding/spikeslicer.py       (SpikeSlicer + SPA-Loss)")
    print("  P3: embedding/is_snn.py           (IS-SNN, surrogate-grad STBP, cupy backend)")
    print("  P4: memory/lt_gate.py, memory/maxformer_pool.py")
    # "reuses stdlib core" until 2026-08-04 -- the word "stdlib" was retired-rule residue (P7).
    # The reuse itself is the point and is real (P2: one definition).
    print("reuses the core: van_rossum/encode_latency (spike_preprocessing), population_encoding,")
    print("                 energy constants (instruments.energy -> neuro_chip_gpu*)")
    print("=" * 66)
    if torch is None or sj is None:
        print("NOTE: install requirements.txt into .venv-neuro to run the phase modules.")
        return

    print("per-phase smokes:")
    from embedding.is_snn import smoke as p3
    from encoding.spikeslicer import smoke as p2
    from memory.lt_gate import smoke as p4a
    from memory.maxformer_pool import smoke as p4b
    for label, fn in (("P2 SpikeSlicer", p2), ("P3 IS-SNN", p3),
                      ("P4a LT-Gate", p4a), ("P4b Max-Former", p4b)):
        print(f"[{label}] ", end="")
        fn()
    print("=" * 66)
    print("all phase smokes passed.")


if __name__ == "__main__":
    main()
