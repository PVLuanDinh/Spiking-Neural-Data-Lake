"""GPU training tier for the Spiking Neural Data Lake.

This is the ONE installed subpackage where the project's zero-dependency rule is
deliberately loosened: it depends on torch + snntorch (the optional ``[train]``
extra) to train a surrogate-gradient spiking classifier on the GPU, then EXPORTS
the learned weights back into the zero-dep core so a clone-and-run install can
serve them with no torch present.

The trainer targets the exact task the core ``snn_data_lake.cortex.snn_classifier``
defines -- 8x8 binary shapes (GRID=8, N=64 inputs), C=4 classes, STEPS=20 rate-
coded timesteps -- and uses a single ``nn.Linear(N, C)`` spiking readout whose
weight has shape [C][N]. That maps 1:1 onto the core's ``W[c][i]`` matrix, so
``export_to_core`` can write a plain JSON weight file that
``snn_classifier.load_weights`` loads and serves.

Modules:
  - device.py   : opt-in CUDA device selection (NORD_GPU=1 / NORD_DEVICE=cuda),
                  CPU fallback. Mirrors research/torch_device.py.
  - trainer.py  : the snnTorch surrogate-gradient model, train/eval, checkpoint,
                  and the export_to_core weight bridge.
  - offload.py  : CPU/disk-offload training (gradient checkpointing + CPU
                  optimizer state) for models larger than VRAM.
  - cli.py      : the ``snn-train`` console script.

torch is imported LAZILY (inside functions) in this ``__init__`` and in ``cli``,
so ``import snn_data_lake`` and running ``snn-train`` without the extra both stay
clean -- there are NO torch imports at this module's top level.
"""
