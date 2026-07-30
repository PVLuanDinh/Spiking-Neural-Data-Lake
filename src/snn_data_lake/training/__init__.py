"""GPU training tier for the Spiking Neural Data Lake.

The GPU training tier. It depends on torch + snntorch (the ``[train]`` extra) to
train a surrogate-gradient spiking classifier, then EXPORTS the learned weights as
a plain JSON file the core loads and serves.

(Until 2026-07-29 this paragraph called itself "the ONE installed subpackage where
the project's zero-dependency rule is deliberately loosened", exporting weights
"back into the zero-dep core so a clone-and-run install can serve them with no
torch present". There is no zero-dep core to export into -- torch and numpy became
unconditional core dependencies on 2026-07-20. The weight export is still real and
still useful, because a JSON weight file is smaller and faster to load than a torch
checkpoint and needs no snntorch; that is a serving-format argument, which is what
this always was. Backlog M56.)

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
