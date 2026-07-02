# research/

Neuromorphic benchmarks and experiments that seeded the product (the arm-loop + data-lake at
the repo root). Kept for reproducibility and history — **not** part of the shipped product.

Two tiers:
- **Stdlib** (run in CI via `run_selfchecks.py`): `spiking_storage_prototype`, `test_prototype`,
  `hopfield_baseline`, `dense_associative_memory`, `snn_moe_classifier`, `temporal_coding_storage`,
  `spiking_ssm`, `spike_knowledge_graph` (+ `_rotate`), `spike_kg_relations`, `spiking_vsa`,
  `nmnist_ingest`, `srm_neuron`, `stats_bounds`, `make_results_plot`.
- **Torch / GPU** (own venvs, not in CI): `snn_motor_control`, `snn_world_model_rl`,
  `snn_dreamer_6dof`, `snn_storage_core_snntorch`, `eth_mnist_bindsnet`, the `snn_mnist_*`
  variants, `snn_moe_stdp_mnist`, `paradigm_b_genn`, `snn_ssm_shd`.

Run from the repo root (`PYTHONPATH=src` so the `snn_data_lake` package resolves):
```bash
PYTHONPATH=src python research/<name>.py
```
The product package lives in `src/snn_data_lake/`; the stdlib tier above is also run by `run_selfchecks.py`.
