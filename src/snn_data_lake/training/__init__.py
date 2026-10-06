"""GPU training tier for the Spiking Neural Data Lake.

Since 2026-10-02 (owner: port-then-retire) this tier's four backprop trainers are retired: ``trainer``
(the 8x8 readout, which ``readout/snn_classifier``'s LOCAL rule already covers), ``offload``, ``mnist`` (its
e-prop port lost the pre-registered A/B and was not landed) and ``spiking_lm`` with its serve core
(``specs/022-unified-brain-arc/evidence/trainer-retirement.md``).

Since 2026-10-06 the GeNN/EventProp trainer ``mnist_genn`` is retired too (chance after one epoch; owner
decision 17c), so ``cli.py`` -- the ``snn-train`` console script -- is the only module left and every path in it
refuses with a message.

"""
