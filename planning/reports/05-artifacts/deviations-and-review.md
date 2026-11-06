# Stage 05: implementation disclosures and self-review

2026-11-01, after pilot-1. Protocol, evaluator and offline reader were written
before task execution and kept byte-identical throughout. No parameter, task,
budget, seed, outcome rule or raw record was changed after seeing the results.

One timing deviation: the protocol groups all constructed controls as before task
runs. The eight cheap in-process controls did run first. The four actual damaged
lake/new-process checks (missing checkpoint, missing last write, corrupted payload,
changed pool fingerprint) ran after the first seed's batch/restore pair, before
the remaining task seeds. Thus the recipe failure was not constructed before the
first task data. It used a copy of that first real lake; the valid original was
unchanged. Its timing does not meet that literal protocol sentence. The checks
all fired for the intended reasons, and no outcome or configuration was selected
using them. This deviation is disclosed rather than rewriting the protocol.

Three self-review passes, not an independent panel or second-model review:

1. Instrument: stdlib recompute validates response counts/targets, beta-derived
   answers, replay membership, paired wake states, budgets, lake framing/checksum,
   foreign tenant and exact restore state/reads. Eight constructed offline corruptions
   are rejected. The original evaluator and reader are preserved by manifest hashes.
   Two fresh analysis processes produce identical summary/sensitivity/table bytes.
2. Honesty: all four seeds have no A forgetting and zero dev headroom after B.
   The accuracy contrast cannot establish benefit or equivalence. Shuffle loses
   accuracy but that only establishes sensitivity to the labels; it cannot create
   headroom in the primary control. Both offline procedures move beta; batch and
   delta optimize different losses. No live sleep or whole-brain claim is licensed.
3. Integration: 405 source receipts match; repo status is clean at the original
   detached HEAD. Every experiment worker uses an isolated guarded NORD_HOME,
   CPython 3.14t/GIL-off and CPU torch. The learned state lives under tag 0x07 in
   one lake per arm, with no sidecar. No production integration, battery, whole
   pytest suite, GPU/native parity, daemon, hardware or GitHub action was performed.

Open scientific dependencies: a pre-specified compatible regime with measurable
interference/headroom before testing duration or pruning; new pilot variability,
practical effect and P1 sequential/power design for a future confirmation. These
are not hidden code-repair tasks or grounds for deleting a mechanism. Current
source instructions (including full integration/review/gates) apply if the harness
is later promoted into production. Shared STATUS remains the coordinator's file.
