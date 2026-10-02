"""GPU training tier for the Spiking Neural Data Lake.

Since 2026-10-02 (owner: port-then-retire) this tier's four backprop trainers are retired: ``trainer``
(the 8x8 readout, which ``readout/snn_classifier``'s LOCAL rule already covers), ``offload``, ``mnist`` (its
e-prop port lost the pre-registered A/B and was not landed) and ``spiking_lm`` with its serve core
(``specs/022-unified-brain-arc/evidence/trainer-retirement.md``).

Modules:
  - cli.py        : the ``snn-train`` console script; each retired path refuses with a message.
  - mnist_genn.py : the GeNN/EventProp MNIST scaffold (``snn-train --mnist --genn``), kept, its
                    disposition owed to the owner; it scored chance after one epoch (2026-10-02).

torch is imported LAZILY (inside functions) in ``cli``, so ``import snn_data_lake`` and running
``snn-train`` without the extra both stay clean -- there are NO torch imports at this module's top level.
"""
