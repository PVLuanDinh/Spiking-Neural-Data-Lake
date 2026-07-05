# Running the Spiking Neural Data Lake Locally

Run the entire Spiking Neural Data Lake stack without any cloud dependencies.

## What's Already Local

The core is **zero-dependency** and runs immediately:

```bash
python run_selfchecks.py
```

This exercises all safety-critical paths (collision detection, E-STOP, serial protocol, ML inference on spikes) without touching any network.

## Local Data Store

The Medallion lakehouse (`spike_telemetry_hub`, `.spk` file format) is the **local-only replacement** for the optional GCP path. It stores spiking events on disk with efficient random-access via `disk_query()`. No cloud required.

## Local NL Interface (Optional)

Interpret English commands into safe intents (arm control / lake queries) locally:

### With Keyword Router (Zero Dependencies)

```bash
python local_llm/run_local.py
```

This uses `StubBackend` — a deterministic keyword router that maps phrases to labels without any ML. No installation needed.

### With Local LLM (AirLLM, ~500MB Model Download)

For on-device, free-text to intent inference:

```bash
pip install ".[llm]"
SNN_LLM=1 python local_llm/run_local.py
```

Or with the flag:

```bash
python local_llm/run_local.py --llm
```

**First run:** downloads and caches the model (Qwen2.5-0.5B, ~1 GB of bf16 weights on Hugging Face). Subsequent runs reuse the cache and are offline.

**Hardware:** Runs on CPU (slow — seconds per inference, hardware-dependent and unmeasured here). A CUDA GPU with 4GB+ VRAM helps. For picking one of ~10 command labels, the default `StubBackend` (zero-dep, deterministic) is usually the better choice — see the trade-off analysis below.

> **Is a local LLM worth it here?** AirLLM is engineered to stream 70B–671B models through tiny VRAM at the cost of disk-bound per-token latency; the NL front-end only classifies a phrase into one of ~10 labels. See **[docs/AIRLLM_ANALYSIS.md](AIRLLM_ANALYSIS.md)** for the honest infrastructure analysis and a decision table (StubBackend vs small model vs AirLLM).

## Safety Guarantee

The local LLM only **selects** an allow-listed intent. Every motor command still passes through `ArmSim.apply()`, which enforces:
- **Collision detection**: blocks unsafe end-effector positions
- **E-STOP reflex**: instant motor kill on overload or emergency signal
- **Serial firewall**: command allow-list prevents raw wire injection
- **Deterministic simulation**: `ArmSim` geometry/collision is deterministic — the same command yields the same gated result

The LLM can never inject a raw command — `nl_interface` gates all output through the existing safety firewall.

## Example Workflow

```bash
# Start with zero dependencies
python local_llm/run_local.py

# Output shows arm commands executing safely:
#   [1] Input: Close the gripper
#       Intent: kind=arm, command=close
#       State: stopped=False, ee=(0.1, 0.2, 0.3)
#
#   [2] Input: Delete all data
#       Intent: kind=reject, reason=unsafe_command
#       NO ACTION TAKEN
```

To add the optional LLM:

```bash
pip install ".[llm]"
SNN_LLM=1 python local_llm/run_local.py
```

Same commands now run through Qwen2.5-0.5B before intent selection.

## Testing

```bash
# Run self-checks (always zero-dep)
python run_selfchecks.py

# Run local stack demo
python local_llm/run_local.py

# Run with local LLM
SNN_LLM=1 python local_llm/run_local.py
```

All exit 0 on success.
