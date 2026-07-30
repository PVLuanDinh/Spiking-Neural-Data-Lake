# Running the Spiking Neural Data Lake — Local

Run everything on one machine.

- Windows paths use `.venv\Scripts\python`; macOS/Linux use `.venv/bin/python`. Pick yours.
- The pure-Python core needs **nothing** but Python 3.11+. Only the lakehouse / real-data /
  GPU pieces need extra packages, each in its own throwaway venv so they don't collide.

---

## Local

### 0. Get it
```bash
git clone https://github.com/Aighluvsekks/Spiking-Neural-Data-Lake.git
cd Spiking-Neural-Data-Lake
python --version          # need 3.11+
pip install -e .          # editable install (core = torch + numpy)
```

### 1. Core self-checks
Every stdlib module runs standalone with an assert-based self-check — the fastest way to see
each piece work:
```bash
python -m snn_data_lake.reasoning.spiking_storage_prototype   # factored associative-memory storage
python -m snn_data_lake.spike_telemetry_hub                   # Paradigm A: sparse .spk store + windowed queries
python -m snn_data_lake.paradigm_b_engine                     # Paradigm B: in-storage spike-query engine
python -m snn_data_lake.reasoning.spike_kg                    # Paradigm C: relational spike embeddings
```

> The top-level `research/` tree became `src/snn_data_lake/reasoning/` in v3.78 (commit `627b2e2b`,
> 2026-07-19), so every `python research/<x>.py` command on this page is now
> `python -m snn_data_lake.reasoning.<x>`. Paradigm C is the one that also changed NAME: the three
> `spike_knowledge_graph*.py` modules were merged into a single `spike_kg` runner in commit `17067032`
> (2026-07-08) and the originals deleted, so `--model {transe,rotate,relations}` selects what used to
> be a separate file. Corrected 2026-07-30 after `tools/citation_guard.py` found all five commands on
> this page pointing at paths absent from a clone.
Run the whole suite (what CI runs) — CORE product + `research/` self-checks:
```bash
python run_selfchecks.py       # 277/277, one source of truth for the module list
python run_selfchecks.py --in-process   # same CORE list without subprocesses (for a frozen build,
                                        # which has no interpreter to spawn); prefer the default
                                        # subprocess isolation everywhere else
```

CSV ingest and windowed `.spk` queries are a CLI on the telemetry hub:
```bash
python -m snn_data_lake.substrate.spike_telemetry_hub --ingest data.csv --out data.spk
python -m snn_data_lake.substrate.spike_telemetry_hub --query data.spk --window 0,10 --channels 0,2
```

### 2. The robot-arm closed loop (the application)
Encode → data lake → match → command → reward → learn (with reflex + RPE dopamine + cortisol +
the Interpreter), on `main`:
```bash
python -m snn_data_lake.closed_loop        # full stack end-to-end: gesture -> command, collision -> STOP
python -m snn_data_lake.signal_loop        # the loop alone (hybrid matcher) + self-check
python -m snn_data_lake.interpreter        # matched label -> robot command + self-check
# drive it with no hardware (outcomes back via OUTCOME lines):
cat windows.csv | python -m snn_data_lake.signal_loop --stdin --feedback --reflex
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
- **Just trying it / developing / the robot-arm loop** → sections 1–2 (zero install).
- **Lakehouse analytics on one machine** → section 3.
- **The literature-target ~95% GPU run** → section 5 (needs CUDA + torch).
The data path (Bronze → Silver → Gold) runs entirely over local Parquet.
```
