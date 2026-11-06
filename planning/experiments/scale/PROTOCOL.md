# SHD component pilot protocol, revision 1

Registered before SHD model outcomes on 2026-11-03. Exploratory feasibility study, not a powered confirmation. Previously exposed development seeds 8000–8003 are reused; they are not a new independent confirmation cohort. The official SHD test archive is not downloaded or read.

## Data and measured variables

Use the official SHD training archive, verified against the publisher's MD5 `d47c9825dee33347913e8ce0f2be08b0`; also record SHA256. Every recording has raw spike times, channel IDs, a spoken-digit class and speaker metadata. Labels and speaker IDs are scorer/split metadata, never network inputs.

Operational cache location: `/private/tmp/sndl-scale-data-20261103`, because the first Documents archive became a macOS dataless file and timed out during reading. Portable manifests, scripts and run results stay in the workspace; cache is regenerable and may be removed by the OS. This relocation precedes all model outcomes.

Inputs: per-recording 100-bin by 700-channel spike **counts**. Map each recording's observed [minimum time, maximum time] linearly to 100 bins, clamp the last event into bin 99, and preserve every event exactly. No clipping, normalization by label, synthetic augmentation or fabricated samples. Variable-duration normalization removes absolute duration; retain duration/event statistics in the preparation report. This explicit adapter is not claimed bit-identical to Tonic. UInt16 storage must be checked for overflow. Float32 counts enter the network.

Hold out the two numerically highest training-speaker IDs, chosen by metadata only; all other speakers train. Stop if either side lacks any of the 20 classes, and register a corrected split before training. No utterance or speaker crosses this boundary. Save sample IDs and split/array hashes. Evaluation uses all held-out recordings, all 20 output classes and no task-identity mask.

Outputs: 20 logits, predicted class, summed hidden spike activity, per-example target/prediction records, per-class confusion counts, cross-entropy, macro and overall accuracy, actual exposures/updates, wall time and peak RSS. Initial and every-block/epoch evaluation cannot modify weights or optimizer state. Hash weights before and after evaluation. This component has fixed width; no growth is measured or implied.

## Existing implementation and pilot budget

Import the pinned source's `EpropAudioNet`, `eprop_grads`, and shared `_apply_grads` directly; no copied or substituted learning kernel. CPU, one Torch thread, no DataLoader workers. Initial calibration: 100 time bins, hidden width 64, batch 16, Adam learning rate 0.002 and source dynamics/initialization defaults. H=64 is a laptop feasibility setting, not an estimate of engine capacity. Profile cumulative 256, 1,024 and 4,096 training exposures. Log actual seconds, RSS, loss and spiking. Stop on nonfinite outputs/gradients, RSS above 3 GiB, or four hours total training time. Do not drop low-scoring runs.

First run a joint IID positive control on seed 8000 for two full passes through training data. Apparatus gate: final held-out macro accuracy must exceed 10% (5% uniform chance plus 5 percentage points). This is a feasibility threshold, not a meaningful scientific effect size. If it fails, stop the larger comparison and report the diagnosis with all real measurements. Any later hyperparameter revision gets a new protocol and visible exploratory status.

If feasible, run up to four paired seeds (8000–8003), limited by measured runtime and the four-hour cap. Each seed uses the same initial weights and class order across arms. Shuffle 20 classes with the seed and divide into ten disjoint two-class blocks. Each training recording appears once as a current-stream example. Shuffle within each block; batch size 16 (final partial batch allowed). Evaluate every block on the complete held-out vocabulary and record a 10-by-10 task retention matrix.

Arms: (1) sequential one update per current batch; (2) sequential two updates on the same current batch; (3) sequential one current update plus one replay update of the same batch size, sampled without replacement where possible from previously seen **training-only** examples, up to a uniform 400-recording reservoir. Reservoir selection is label-blind and seeded. Update the reservoir after both updates, so replay comes only from earlier samples. On the first batch use a second current update because no history exists. Replay and repeated-current arms have exactly matched update counts and batch sizes. They differ in historical input access; report that distinction and actual unique/exposure counts. An IID two-pass reference has the same aggregate training exposure count but a different order; it is a learnability reference, not an architectural win.

Final endpoints: overall/macro held-out accuracy; mean seen-task accuracy after each block; forgetting (best earlier task accuracy minus final, excluding tasks not yet learned); replay-minus-repeat paired differences; exposures, time, RSS and memory footprint. Report every seed and descriptive variability without p-values or confirmatory confidence claims from four exposed seeds. Speaker holdout uncertainty and single-split limits remain explicit. No choosing the best checkpoint. Retain raw predictions and failures. No saved-lake persistence claim in this pilot; do not start a Brain or live lake.

## Required checks before interpreting results

Verify raw event conservation and ranges, deterministic binning, train/development sample and speaker disjointness, class coverage, hashes and source identity. Construct a known confusion matrix and a deliberately permuted target fault that changes its score. Verify evaluation leaves weights unchanged. Check repeated identical updates yield identical CPU weights. Replay must contain training IDs only, never a future stream sample or development ID. Keep bounded source self-check results as apparatus evidence, separate from real-data scores.

This pilot tests the existing local e-prop event-audio learner. It does not test G9, growing/adapting engine state, an integrated reservoir, recursive self-improvement, SSC, transformer superiority or the official SHD test benchmark.
