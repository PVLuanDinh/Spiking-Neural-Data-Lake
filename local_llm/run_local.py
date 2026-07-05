"""
Run the entire Spiking Neural Data Lake stack locally, no cloud.

This script wires together:
  - nl_interface.NLInterface: zero-dep intent interpreter
  - Optional AirLLMBackend (on-device LLM) or StubBackend (keyword router)
  - arm_sim.ArmSim: the safety firewall for motor commands
  - spike_telemetry_hub: local data store for spiking telemetry

Every motor command flows through ArmSim.apply(), ensuring the existing
safety gates (collision, E-STOP, freeze) are always enforced.

Usage:
    python local_llm/run_local.py              # runs with StubBackend (zero deps)
    SNN_LLM=1 python local_llm/run_local.py    # runs with AirLLMBackend if available
    python local_llm/run_local.py --llm        # same as SNN_LLM=1

Examples in the script demonstrate:
  - Safe arm commands (intent -> apply -> state)
  - Rejected unsafe phrases (intent -> kind:reject -> no action)
  - Lake queries (intent -> disk_query -> event count)
"""

import sys
import os
from pathlib import Path

# Shim: allow imports from src/ when run from repo root
repo_root = Path(__file__).parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from snn_data_lake.nl_interface import NLInterface, StubBackend
from snn_data_lake.arm_sim import ArmSim
from snn_data_lake.spike_telemetry_hub import SpikeTelemetryHub, synth, disk_query


def maybe_load_airllm_backend():
    """
    Attempt to load AirLLMBackend if --llm flag or SNN_LLM=1 is set.
    If unavailable, print a notice and return StubBackend.
    """
    use_llm = os.environ.get("SNN_LLM", "").lower() in ("1", "true") or "--llm" in sys.argv

    if not use_llm:
        return StubBackend(), "StubBackend (keyword router)"

    try:
        from local_llm.airllm_backend import AirLLMBackend
        print("Using AirLLMBackend (on-device LLM)")
        try:
            backend = AirLLMBackend()
            return backend, "AirLLMBackend"
        except ImportError as e:
            print(f"AirLLMBackend unavailable: {e}", file=sys.stderr)
            print("Falling back to StubBackend")
            return StubBackend(), "StubBackend (fallback)"
    except ImportError:
        print("airllm module not found. Install with: pip install airllm torch")
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


def run_transcript():
    """Execute a live transcript of commands showing the full stack."""
    backend, backend_name = maybe_load_airllm_backend()

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


if __name__ == "__main__":
    try:
        run_selfcheck()
        run_transcript()
        print("\nExit 0: success\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
