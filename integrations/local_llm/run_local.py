"""
Run the entire Spiking Neural Data Lake stack locally, no cloud -- and, first-class,
run direct local large-LLM inference through the AirLLM backend.

This script wires together:
  - nl_interface.NLInterface: zero-dep intent interpreter
  - Optional AirLLMBackend (on-device LLM, disk-streaming inference) or StubBackend
    (keyword router)
  - arm_sim.ArmSim: the safety firewall for motor commands
  - spike_telemetry_hub: local data store for spiking telemetry

Every motor command flows through ArmSim.apply(), ensuring the existing
safety gates (collision, E-STOP, freeze) are always enforced.

Two CLI modes:

  demo (default)   Full local stack demo: self-check + transcript, wired through
                    nl_interface -> ArmSim.apply() and the local .spk lake.

                      python local_llm/run_local.py              # StubBackend (zero deps)
                      SNN_LLM=1 python local_llm/run_local.py     # AirLLMBackend if available
                      python local_llm/run_local.py --llm         # same as SNN_LLM=1

  infer PROMPT     One-shot local LLM inference: PROMPT -> generated text, straight
                    through AirLLM's disk-streaming engine. No arm/lake wiring -- this
                    is the direct, first-class entry point for running a large local
                    model. Requires the optional [llm] extra (airllm + torch); prints
                    a clean one-line message and exits non-zero if that extra is not
                    installed (never an ImportError traceback).

                      pip install ".[llm]"
                      python local_llm/run_local.py infer "Explain spiking neurons in one sentence."
                      python local_llm/run_local.py infer "..." --model-id Qwen/Qwen2.5-0.5B-Instruct \\
                          --max-new-tokens 80 --compression 4bit

Building the CLI parser and running --help never imports airllm/torch (only argparse,
stdlib) -- both work with zero optional deps installed. airllm/torch are only touched
once `infer` mode or --llm actually try to construct an AirLLMBackend.
"""

import argparse
import sys
import os
from pathlib import Path

# Shim: allow imports from src/ and this repo's own top-level packages (local_llm.*)
# regardless of how this script is invoked (`python local_llm/run_local.py`,
# `python -m local_llm.run_local`, or with PYTHONPATH=. already set).
repo_root = Path(__file__).parent.parent
for _p in (str(repo_root), str(repo_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from snn_data_lake.nl_interface import NLInterface, StubBackend
from snn_data_lake.arm_sim import ArmSim
from snn_data_lake.spike_telemetry_hub import SpikeTelemetryHub, synth, disk_query


def build_arg_parser():
    """
    Build the CLI parser. Pure argparse -- constructing this parser and running
    --help never imports airllm/torch, so both work with zero optional deps
    installed. See the module docstring for the two modes (demo / infer).
    """
    parser = argparse.ArgumentParser(
        prog="run_local.py",
        description=(
            "Run the Spiking Neural Data Lake stack locally, no cloud. Default mode "
            "runs the full self-check + demo transcript. Use 'infer' to run one-shot "
            "local large-LLM inference directly through the AirLLM backend."
        ),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help='[demo mode] use the on-device AirLLMBackend instead of StubBackend '
             '(same as SNN_LLM=1). Requires: pip install ".[llm]"',
    )

    subparsers = parser.add_subparsers(dest="mode")

    infer = subparsers.add_parser(
        "infer",
        help='One-shot local LLM inference via AirLLM (prompt -> text). '
             'Requires: pip install ".[llm]"',
    )
    infer.add_argument("prompt", help="Prompt text to run through the local model.")
    infer.add_argument(
        "--model-id",
        default=None,
        help="HuggingFace model id to stream layer-by-layer from disk "
             "(default: Qwen/Qwen2.5-0.5B-Instruct). Pass a large id (e.g. a 70B-class "
             "model) to exercise AirLLM's actual purpose: low-VRAM streaming of a "
             "large model.",
    )
    infer.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max tokens to generate (default: 50). Every generated token re-streams "
             "every layer from disk -- keep this small for lower latency.",
    )
    infer.add_argument(
        "--compression",
        choices=["4bit", "8bit"],
        default=None,
        help="Optional block-wise weight compression (requires bitsandbytes) to "
             "speed up the disk-load bottleneck. Default: none (full precision).",
    )

    return parser


def run_infer(args):
    """
    One-shot local LLM inference: prompt -> generated text, via the AirLLM backend.
    No arm/lake wiring -- this is the direct, first-class local-inference entry point.

    Requires the optional [llm] extra (airllm + torch). If that extra is not
    installed, prints a clean one-line message and exits non-zero -- never an
    ImportError traceback.
    """
    from local_llm.airllm_backend import AirLLMBackend  # always safe: lazy imports inside

    try:
        backend = AirLLMBackend(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            compression=args.compression,
        )
    except ImportError as e:
        print('AirLLM inference requires the optional [llm] extra. Install with: pip install ".[llm]"')
        print(f"(detail: {e})", file=sys.stderr)
        sys.exit(1)

    output = backend.generate(args.prompt)
    print(output)


def maybe_load_airllm_backend(use_llm: bool):
    """
    Load AirLLMBackend if requested (via --llm or SNN_LLM=1), else the zero-dep
    StubBackend. Never raises: falls back to StubBackend with a clean message if
    AirLLM/torch are not installed.
    """
    if not use_llm:
        return StubBackend(), "StubBackend (keyword router)"

    try:
        from local_llm.airllm_backend import AirLLMBackend
    except ImportError:
        print("local_llm.airllm_backend not importable from this path.")
        print("Falling back to StubBackend")
        return StubBackend(), "StubBackend (fallback)"

    print("Using AirLLMBackend (on-device LLM)")
    try:
        backend = AirLLMBackend()
        return backend, "AirLLMBackend"
    except ImportError as e:
        print('AirLLM inference requires the optional [llm] extra. Install with: pip install ".[llm]"')
        print(f"(detail: {e})", file=sys.stderr)
        print("Falling back to StubBackend")
        return StubBackend(), "StubBackend (fallback)"


def create_demo_lake():
    """Create a small synthetic .spk store for demo queries."""
    # Use ./data dir if it exists, else temp dir
    data_dir = Path("./data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    spk_path = data_dir / "demo.spk"

    # Only regenerate if missing
    if not spk_path.exists():
        # Synthetic: 2 channels, 10,000 steps duration, ~4 spikes per channel, bursts on channels 0 and 1
        hub = synth(
            n_channels=2,
            duration=10000,
            rate=0.0004,  # ~4 spikes per channel
            burst_channels=[0, 1],
            burst_win=(5000, 5100),  # burst window in steps
        )
        # Save hub to disk
        hub.save(str(spk_path))
        print(f"Created demo lake: {spk_path}")

    return str(spk_path)


def run_transcript(backend, backend_name):
    """Execute a live transcript of commands showing the full stack."""
    # Create NLInterface with the backend
    nl = NLInterface(backend=backend)

    # Prepare arm and lake
    arm = ArmSim()
    lake_path = create_demo_lake()

    print("\n" + "=" * 70)
    print("Spiking Neural Data Lake - Local Stack Demo")
    print("=" * 70)
    print(f"Backend: {backend_name}")
    print(f"Lake store: {lake_path}")
    print("=" * 70 + "\n")

    # Example commands
    examples = [
        "Close the gripper",
        "Hold position",
        "Stop all movement",
        "Read telemetry from channel 0 window 6000",
        "Delete all data",  # Will be rejected
    ]

    for i, text in enumerate(examples, 1):
        print(f"[{i}] Input: {text}")

        # Interpret
        intent = nl.interpret(text)
        print(f"    Intent: {intent['kind']}")

        if intent["kind"] == "arm":
            # Execute through safety gate
            command = intent["command"]
            print(f"    Command: {command}")
            state = arm.apply(command)
            print(f"    State: stopped={state.get('stopped')}, ee={state.get('ee')}")

        elif intent["kind"] == "lake":
            # Query the lake
            query = intent["query"]
            channels = query.get("channels") or [0, 1]
            window = query.get("window", 6000)
            t0, t1 = 0, window  # .spk event times are integer steps, not ms
            events, bytes_read = disk_query(lake_path, t0, t1, channels)
            print(f"    Query: channels={channels}, window_steps={window}")
            print(f"    Result: {len(events)} events, {bytes_read} bytes read")

        elif intent["kind"] == "reject":
            reason = intent.get("reason", "unknown")
            print(f"    Rejected: {reason} (NO ACTION TAKEN)")

        print()

    print("=" * 70)
    print("Demo complete. All motor commands passed through ArmSim.apply().")
    print("=" * 70 + "\n")


def run_selfcheck():
    """
    Minimal assert-based self-check: provable without airllm.

    Returns True if all checks pass.
    """
    print("\nRunning self-check (zero-dependency)...")

    # Backend always succeeds (StubBackend is always available)
    backend = StubBackend()
    nl = NLInterface(backend=backend)
    arm = ArmSim()
    lake_path = create_demo_lake()

    # Check 1: safe phrase drives a gated apply and yields valid state
    intent = nl.interpret("close the gripper")
    assert intent["kind"] == "arm", f"Expected kind=arm, got {intent['kind']}"
    state = arm.apply(intent["command"])
    assert "stopped" in state, "State missing 'stopped' key"
    print("  OK: safe phrase executes, state valid")

    # Check 2: stop produces stopped=True
    intent = nl.interpret("STOP")
    assert intent["kind"] == "arm", f"Expected kind=arm for STOP, got {intent['kind']}"
    state = arm.apply(intent["command"])
    assert state.get("stopped") is True, "STOP should set stopped=True"
    print("  OK: STOP command produces stopped=True")

    # Check 3: lake query actually returns events (window covers the injected burst)
    intent = nl.interpret("read channel 0 window 6000")
    assert intent["kind"] == "lake", f"Expected kind=lake, got {intent['kind']}"
    query = intent["query"]
    channels = query.get("channels") or [0, 1]
    window = query.get("window", 6000)
    t0, t1 = 0, window
    events, bytes_read = disk_query(lake_path, t0, t1, channels)
    assert len(events) > 0 and bytes_read > 0, "lake query should return events from the local store"
    print(f"  OK: lake query returns {len(events)} events")

    # Check 4: unsafe phrase produces NO arm action
    intent = nl.interpret("delete everything from the database")
    assert intent["kind"] == "reject", f"Unsafe phrase should reject, got {intent['kind']}"
    print("  OK: unsafe phrase is rejected (no action)")

    print("\nAll checks passed!")
    return True


def main(argv=None):
    """CLI entry point: dispatch to 'infer' mode or the default demo mode."""
    args = build_arg_parser().parse_args(argv)

    if args.mode == "infer":
        run_infer(args)
        return

    # demo mode (default)
    use_llm = args.llm or os.environ.get("SNN_LLM", "").lower() in ("1", "true")
    backend, backend_name = maybe_load_airllm_backend(use_llm)
    run_selfcheck()
    run_transcript(backend, backend_name)
    print("\nExit 0: success\n")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nError: {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
