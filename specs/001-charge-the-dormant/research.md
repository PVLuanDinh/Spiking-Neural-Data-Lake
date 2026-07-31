# Phase 0 Research: Charge the Dormant

All unknowns below were resolved from existing repo evidence and the 2026-07-31 runtime
adjudication. No external research was required; every decision cites something already measured.

## D1 — Which neuter mechanism applies to which rule

**Decision**: Two neuters, selected by whether the rule is called at all.

| case | neuter | why |
|---|---|---|
| rule IS called on a live path | patch the method to a no-op, re-run same seed, diff behavioural hashes | measures whether its effect matters |
| rule is NEVER called | no neuter is meaningful; report "never called" and measure the WIRED version against the unwired baseline instead | you cannot subtract something that never ran |

**Rationale**: `tools/charge_probe.py`'s neuter pins a *drive-source weight* (`_PinnedWeights`), which
only applies to the eleven undirected drive sources. Seven of the nine rules here are not drive
sources at all, so the probe's built-in neuter cannot address them. The 2026-07-31 adjudication
already used the no-op patch successfully and produced byte-comparable behavioural hashes.

**Alternatives considered**: extending `charge_probe` to accept an arbitrary method target. Rejected
for now — it would change the tool every recorded artifact depends on, and this feature already has
one instrument change (D3) that must not be bundled with a behavioural one.

## D2 — What counts as a behavioural difference

**Decision**: A tuple of per-tick hashes over (actions, coherence, elapsed, bound events), plus the
graded `internal reads changed: N of M ticks` observable and the top state fields by |delta|.

**Rationale**: The adjudication used exactly these hashes and they discriminated correctly — for
`AffectiveAppraisal.learn` every behavioural hash was byte-identical while only the telemetry hash
moved, which is what exposed the disjoint pool/consumer. Argmax flips alone are known to
under-detect: all-eleven neuter gives 47/60 trajectory flips, any single source gives 0/60.

**Alternatives considered**: trajectory flips alone (refuted above); full state diffing (too noisy —
counters and clocks differ trivially).

## D3 — How to express the `_TabularQ.update` exemption

**Decision**: Extend `_parse_all` in `substrate/method_dormancy_gate.py` to track the enclosing
`ClassDef`, making keys `module.Class.method`. This fixes two defects with one change.

**Rationale**: `basal_ganglia/latent_q.py` defines `update` twice — `LatentQ.update:102` (genuinely
dormant, open work) and `_TabularQ.update:165` (a self-check control, correctly dormant). Both a bare
`update` key and a module-qualified `basal_ganglia.latent_q.update` key would exempt BOTH, hiding the
real finding. The same `(module, name)` keying also collapses the two defs into one entry whose
surviving lineno names the wrong class — which is why the finding was reported at `:165`.

**Alternatives considered**: leaving `_TabularQ` permanently uncertain (rejected: it is a correct
disposition that the instrument merely cannot express, and unexpressed dispositions rot); renaming
`_TabularQ.update` to dodge the collision (rejected: changing product code to fit a gate's key
format is the tail wagging the dog).

## D4 — Disposition per rule, from the adjudication

| rule | verdict | planned disposition |
|---|---|---|
| `LatentQ.update` | never called, module imported | WIRE or RETIRE — needs a decision on evidence (see D5) |
| `_TabularQ.update` | never called, correctly so | EXEMPT (permanent) once D3 lands |
| `ForwardModel.update` | never called; gate's UNCERTAIN was a false positive from `dict.update` | WIRE or RETIRE |
| `DevelopmentalEnvelope.update` | never called, and discarded even under its opt-in flag | WIRE or RETIRE |
| `PredictiveCodingNetwork.learn` | never called, correctly so | EXEMPT (permanent) — already landed `f181b484` |
| `AffectiveEpisodicStore.consolidate` | never called | RETIRE, per its own exemption text, after P2 port check |
| `AffectiveAppraisal.learn` | called; charges `w_aro`, consumed component is `valence` — disjoint; tier-dependent | FIX the disjoint wiring, or record as measured-inert |
| `TimeCellClock.learn_interval` | called ~66/400 ticks, learns nothing; threshold is a de facto constant | FIX (also a P1 defect) |
| `IntrinsicPlasticity.adapt` | called, proven receiver, no charge on the route taken | FIX or record as measured-inert |

## D5 — WIRE vs RETIRE for the three never-called-but-real rules

**Decision**: Default to RETIRE unless wiring is a call site rather than a redesign; decide per rule
with the port check first.

**Rationale**: P2 requires porting a unique capability before deletion, and the constitution warns
that wiring something merely to satisfy a gate produces a fake consumer — explicitly forbidden by P3
("never a dead consumer to silence a gate"). `AffectiveEpisodicStore.consolidate` already has this
answer recorded: its "wiring" is a three-file redesign of `Episode` + `ReplayBuffer`, and the
prioritised-replay capability it was built for has no consumer today.

**Alternatives considered**: wire everything (rejected — manufactures consumers); retire everything
(rejected — `DevelopmentalEnvelope`'s mechanism is real and self-check-proven, and a retirement
without a port would violate P2).

## D6 — Environment for every measurement

**Decision**: Repo root AND `src` on `PYTHONPATH`, `NORD_HOME` on a temp dir, `persist=False`, fixed
seed 0, and the perception tier recorded in the artifact header.

**Rationale**: `PYTHONPATH=src` alone makes `training/` unimportable, so the brain silently drops to
the stdlib tier and perceives nothing — `SpatialTemporal.md` records this producing a false
NOT-ELIGIBLE. `AffectiveAppraisal.learn` charges in the stdlib tier and is inert in the eprop-snn
tier, so a verdict without a named tier is not a verdict. A temp `NORD_HOME` is required because the
daemon writes the real lake continuously and brain construction reads it.

## Open risks

- **R1**: `AffectiveAppraisal.learn`'s fix (connecting the charging pool to the consumed component)
  is a behavioural change to a live faculty and will change recorded artifacts. It must be measured
  and disclosed, not folded into an instrument commit.
- **R2**: Any wiring that lands changes the brain's output and may invalidate recorded fingerprints
  for unrelated artifacts. Check `--rerun` across the ledger after each behavioural change.
- **R3**: `TimeCellClock`'s fix touches a P1 hardcode; changing a threshold from constant to learned
  alters timing behaviour and needs its own A/B rather than riding on the P3 disposition.
