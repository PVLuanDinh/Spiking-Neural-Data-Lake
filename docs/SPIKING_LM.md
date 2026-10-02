# Spiking language model (retired)

The char-level spiking LM trainer (`training/spiking_lm`) and its torch-free serve core
(`sensorimotor/spiking_lm_serve`) were RETIRED on 2026-10-02 with the other backprop trainers
(owner: port-then-retire; `specs/022-unified-brain-arc/evidence/trainer-retirement.md`). `snn-train --lm`
now refuses.

What was lost, measured before deletion from its own code path (defaults: 25 steps, the built-in
512-character corpus, 60,698 parameters; the path has no held-out split, so these are in-sample):
perplexity 15.54, 14.62 and 14.40 at seeds 0-2, against a unigram baseline of 17.14 and a
maximum-likelihood bigram of 7.20 on the same text; its greedy samples were degenerate.

An e-prop char-LM exists outside the governed tree (`training/eprop_lm` at the repo root, served by
`reasoning/eprop_lm_serve`); governing it is a separate, carried piece of work. The Spiking-RWKV
model served by `reasoning/ar_spiking_lm_serve` is unrelated and unchanged.
