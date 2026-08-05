# Running the Spiking Neural Data Lake — Local

Run everything on one machine.

- Windows paths use `.venv\Scripts\python`; macOS/Linux use `.venv/bin/python`. Pick yours.
- **`pip install -e .` is required — there is no zero-install path.** `torch`, `numpy` and
  `mujoco==3.11.0` are UNCONDITIONAL core dependencies (`pyproject.toml` `dependencies`), and the
  `NORD_TORCH=0` off-switch was removed on 2026-07-29. The lakehouse / real-data / GPU tiers need
  further extras on top, each in its own throwaway venv so they don't collide.
  *(Corrected 2026-08-04 under P7: this page said "the pure-Python core needs **nothing** but
  Python 3.11+" and called sections 1–2 "zero install". Both were residue of the zero-dependency
  rule abolished 2026-07-20, and both were false — a reader following them gets an ImportError,
  not a brain.)*

---

## Local

### 0. Get it
```bash
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake.git
cd Spiking-Neural-Data-Lake
python --version          # need 3.11+
pip install -e .          # editable install (core = torch + numpy + mujoco==3.11.0)
```

### 1. Core self-checks
Every registered module runs standalone with an assert-based self-check — the fastest way to see
each piece work:
```bash
python -m snn_data_lake.hippocampus.spiking_storage_prototype  # factored associative-memory storage
python -m snn_data_lake.substrate.spike_telemetry_hub          # Paradigm A: sparse .spk store + windowed queries
python -m snn_data_lake.cortex.paradigm_b_engine               # Paradigm B: in-storage spike-query engine
python -m snn_data_lake.reasoning.spike_kg                     # Paradigm C: relational spike embeddings
```

> **Use the FULL dotted path.** The flat back-compat alias (`python -m snn_data_lake.interpreter`)
> resolves an *import* but cannot be executed by `-m`: the alias layer has no `get_code`, so it dies
> with `AttributeError: '_AliasLoader' object has no attribute 'get_code'`. `SECURITY.md` documents
> the same rule for the envelope modules. Corrected 2026-08-04: six commands on this page used the
> flat form and exited 1 for a reader who typed them, and `spiking_storage_prototype` had been
> promoted out of `reasoning/` into `hippocampus/` without this page following. `citation_guard.py`
> stayed green throughout, because every file it could see was tracked — the *command* was checked
> by nothing. `tools/doc_command_guard.py` checks it — **built and clean, but not yet a CI step, so
> run it by hand** (`python tools/doc_command_guard.py --assert`). Registering it is open P4c work.

> The top-level `research/` tree became `src/snn_data_lake/reasoning/` in v3.78 (commit `627b2e2b`,
> 2026-07-19), so every `python research/<x>.py` command on this page is now
> `python -m snn_data_lake.reasoning.<x>`. Paradigm C is the one that also changed NAME: the three
> `spike_knowledge_graph*.py` modules were merged into a single `spike_kg` runner in commit `17067032`
> (2026-07-08) and the originals deleted, so `--model {transe,rotate,relations}` selects what used to
> be a separate file. Corrected 2026-07-30 after `tools/citation_guard.py` found all five commands on
> this page pointing at paths absent from a clone.

Run the whole suite (what CI runs) — CORE product + `reasoning/` self-checks:
```bash
python run_selfchecks.py       # the runner is the ONE source of truth for the module list
python run_selfchecks.py --in-process   # same CORE list without subprocesses (for a frozen build,
                                        # which has no interpreter to spawn); prefer the default
                                        # subprocess isolation everywhere else
```

> **No count is quoted here on purpose.** This page used to say `277/277`. Measured 2026-08-04 by
> AST over `run_selfchecks.py`: `MODULES = CORE(202) + TORCH_GATED(6 if torch) + NENGO_GATED(1 if
> nengo) + RESEARCH(72)` — **280** in `.venv-lake` (torch 2.13.0+cpu, no nengo), 281 with nengo,
> minus whatever the run reports as ARTIFACT-SKIPPED. `277` was stale: CORE has taken at least
> three PROMOTED entries since. A hard-coded total in prose goes stale on every promotion, and five
> documents quoted the same one, so read it off the runner instead of copying it.

CSV ingest and windowed `.spk` queries are a CLI on the telemetry hub:
```bash
python -m snn_data_lake.substrate.spike_telemetry_hub --ingest data.csv --out data.spk
python -m snn_data_lake.substrate.spike_telemetry_hub --query data.spk --window 0,10 --channels 0,2
```

### 2. The robot-arm closed loop (the application)
Encode → data lake → match → command → reward → learn (with reflex + RPE dopamine + cortisol +
the Interpreter), on `main`:
```bash
python -m snn_data_lake.brainstem.closed_loop   # full stack end-to-end: gesture -> command, collision -> STOP
python -m snn_data_lake.brainstem.signal_loop   # the loop alone (hybrid matcher) + self-check
python -m snn_data_lake.brainstem.interpreter   # matched label -> robot command + self-check
# drive it with no hardware (outcomes back via OUTCOME lines):
cat windows.csv | python -m snn_data_lake.brainstem.signal_loop --stdin --feedback --reflex
```

### 3. The Medallion lakehouse PoC (needs polars)
Bronze → Silver → Gold over Parquet:
```bash
python -m venv .venv-lake
.venv-lake/bin/pip install -e ".[lake]"            # package + polars/pyarrow/deltalake (Windows: .venv-lake\Scripts\pip)
.venv-lake/bin/python -m snn_data_lake.lakehouse.medallion   # Windows: .venv-lake\Scripts\python -m snn_data_lake.lakehouse.medallion
# writes lakehouse/data/{bronze,silver,gold}.parquet + prints ICR, synchrony, SQL query
```

### 4. Real event-camera data — N-MNIST (optional, needs tonic)
```bash
python -m venv .venv-nmnist
.venv-nmnist/bin/pip install -e ".[nmnist]"        # the package + tonic; pulls ~1 GB N-MNIST on first run
.venv-nmnist/bin/python -m snn_data_lake.reasoning.nmnist_ingest
# no tonic? the same command runs on deterministic synthetic events (no tonic needed)
```

### 5. GPU training (optional, needs CUDA + torch)
For the literature ~95% run (6400 neurons). RTX 50-series (Blackwell) needs CUDA 12.8:
```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
.venv/bin/pip install -e . snntorch==1.0.0 bindsnet numpy
NORD_M=1600 NORD_INH=60 NORD_THETA_PLUS=0.20 \
  .venv/bin/python -m snn_data_lake.reasoning.eth_mnist_bindsnet --gpu
# CPU fallback works automatically (slower); 400/20k ~ 86% in minutes
```

### 6. Browse the knowledge graph (optional)
```bash
# open graphify-out/graph.html in a browser, or:
graphify query "how does the matcher reject novel signals?"
```

---

## Which do I use?
- **Just trying it / developing / the robot-arm loop** → sections 1–2 (needs `pip install -e .`).
- **Lakehouse analytics on one machine** → section 3.
- **The literature-target ~95% GPU run** → section 5 (needs CUDA + torch).

The data path (Bronze → Silver → Gold) runs entirely over local Parquet.
