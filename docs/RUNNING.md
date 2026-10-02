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
python --version          # need 3.12+  (pyproject requires-python, raised from 3.11 on 2026-08-05)
pip install -e .          # editable install (core = torch + numpy + mujoco==3.11.0)
```

**The version check above is not advisory — pip enforces it and refuses.** Measured 2026-08-05 on a
3.11.15 interpreter against a package declaring `Requires-Python: >=3.12`:
`ERROR: Package '...' requires a different Python: 3.11.15 not in '>=3.12'`, exit 1. Note the
`mujoco==3.11.0` on the next line is a *package* version and has nothing to do with the interpreter
floor; the two 3.11s on this page mean different things.

**Known open item, stated here because this page is where a contributor sets up their env:** the
repo's own CPU gate venv `.venv-lake` is **Python 3.11.15** and therefore no longer satisfies this
floor — an already-created `.venv-lake` keeps working (the check happens at install time, not at
import time), but `pip install -e .` into it now fails. Recreating it on 3.12 is an owner decision
and re-bases the environment behind every measurement in `docs/`; see `docs/POLICIES.md` P5.

### 1. Core self-checks
Every registered module runs standalone with an assert-based self-check — the fastest way to see
each piece work:
```bash
python -m snn_data_lake.faculties.spiking_storage_prototype  # factored associative-memory storage
python -m snn_data_lake.substrate.spike_telemetry_hub          # Paradigm A: sparse .spk store + windowed queries
python -m snn_data_lake.substrate.paradigm_b_engine               # Paradigm B: in-storage spike-query engine
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
python -m snn_data_lake.sensorimotor.closed_loop   # full stack end-to-end: gesture -> command, collision -> STOP
python -m snn_data_lake.sensorimotor.signal_loop   # the loop alone (hybrid matcher) + self-check
python -m snn_data_lake.brainstem.interpreter   # matched label -> robot command + self-check
# drive it with no hardware (outcomes back via OUTCOME lines):
cat windows.csv | python -m snn_data_lake.sensorimotor.signal_loop --stdin --feedback --reflex
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
RTX 50-series (Blackwell) needs CUDA 12.8:
```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
.venv/bin/pip install -e ".[gpu]"
NORD_GPU=1 .venv/bin/python -m snn_data_lake.reasoning.snn_mnist_stdp_fast
```

> The BindsNET Diehl & Cook runner that headlined this section (the literature ~95% at 6400
> neurons; 400/20k ~ 86% on CPU in minutes) was RETIRED 2026-08-07 with the `bindsnet`
> dependency -- it was the sole importer, and its numbers survive in the CHANGELOG. One piece of
> knowledge lived only in that file and is kept here: BindsNET <= 0.3 imports `torch._six`,
> removed in torch >= 2.0; the runner shimmed it with a `types.ModuleType("torch._six")` carrying
> `container_abcs = collections.abc`, `string_classes = (str, bytes)`, `int_classes = int`,
> installed in `sys.modules` before the bindsnet imports.

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
