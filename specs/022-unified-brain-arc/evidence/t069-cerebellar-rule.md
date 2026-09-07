# T069 — the cerebellar rule ON the built tissue (2026-09-07)

The row (`tasks.md:149`): *"Cerebellar rule ON the built tissue: covariance/LMS form, filtered-trace
eligibility bank re-fit to this plant, window IDENTITY pinned to measured channel latency,
sign-inversion red (one constant across two channels FAILS), Bouvier GPE/G_E control pair, graded
`io` teaching afferent (binary-after-cancellation declared alternative); e-prop transport-freedom
MEASURED at adoption; in `cerebellum/cf_ltd_rule.py`"*.

The acceptance criterion (`spec.md:69`): *"Given the cerebellar tissue built at S1, When the adopted
rule runs with the Bouvier control, Then the green/red pair reproduces, and one global window
constant across two channels with different measured latencies FAILS (sign-inversion red)."*

## 1. Territory, measured before anything was pre-registered

`cerebellum/microcircuit.py::CerebellarMicrocircuit` is the tissue (T054, closed): mossy → granule
(k=4 codon) → PF bundle → {Purkinje, MLI at fixed −1.0} → deep readout, with granule spikes reaching
the molecular layer through a one-step buffer. It already measures its own `coincidence_window`.

**THE RED AS WRITTEN IS NOT REACHABLE BY VARYING THE TISSUE.** Measured across `n_mossy` ∈
{4, 8, 12, 16, 24, 32} × seeds {0, 1, 2} — **every one of the 18 instances measures window = 2**,
with zero refusals. The latency is structural to this architecture (granule codon + one buffered hop
+ integration), exactly as the tissue's own docstring says ("this tissue's depth answers at 2"), so
"two channels with different measured latencies" cannot be built from size or seed.

Three candidate knobs were then probed, and the result decides the design:

| knob | measured latency | verdict |
|---|---|---|
| Purkinje `tau_m` 10 → 20 → 40 → 80 | 2, 2, 2, 2 | **does not move it at all** |
| Purkinje `v_threshold` 1.0 → 1.5 → 2.0 → 3.0 → 5.0 | 2, 3, 5, 7, 23 | moves it — by tuning a NEURON constant (P1) |
| **relay depth 0 → 1 → 2 → 3** (buffered hops before the mossy input) | **2, 3, 4, 5** | moves it, exactly one step per hop |

**RELAY DEPTH is adopted, and the reason is not convenience.** It is a property of the CHANNEL, not
of a neuron: the tissue's own docstring names the `pn` pons/mossy_relay stage that delivers mossy
fibres, and channels differing in relay depth is ordinary anatomy. It is integer-valued and
derived — nothing is tuned to a number, which the threshold route would have required. The `tau_m`
row is kept because a negative result about which knob does NOT move a measurement is worth as much
as the one that does.

## 2. Pre-registration (written before `cf_ltd_rule.py` existed)

### 2.1 The plant

A VOR-style compensatory plant, two channels. Channel `c` carries a command `u(t)` into the tissue
through `relay_depth[c]` buffered hops; the tissue's deep-nuclei drive is the response `r_c(t)`; the
world requires `w_c * u(t)`; the residual error is `e_c(t) = w_c*u(t) − r_c(t)` — the retinal slip
that REMAINS after the response, which is what the climbing fibre carries. Channels are built at
relay depths **0 and 2**, so their measured latencies are **2 and 4** — asserted, not assumed, in the
same command as the run.

### 2.2 The rule (`cf_ltd_rule.py`)

Covariance/LMS on the Purkinje PF rows, in the CF-LTD sign (a climbing-fibre event depresses the PF
synapses that were active in the window it teaches about):

```
elig_ij(t) = filtered PF trace of granule j, READ AT t − window_c     (the channel's OWN latency)
dw_ij     = -lr * (io_i(t) - <io_i>) * (elig_ij(t) - <elig_j>)        (covariance form, not a bare product)
```

`io` is the **graded** teaching afferent (the row's primary; binary-after-cancellation is the
declared alternative and is NOT built here — declared, per the row's own wording). The eligibility
bank is a per-synapse filtered trace, re-fit to this plant's timescale by the channel's own measured
window rather than by a chosen constant.

### 2.3 The reds, each with the input that makes it fail

- **R-T069a — THE SPEC'S RED (window identity).** Two channels, measured latencies 2 and 4. With each
  channel's window pinned to its OWN measured latency the plant error falls; with **ONE global window
  constant across both** it does NOT fall at the bar. If a single constant does just as well, the
  window identity buys nothing and the row's central claim is deleted, not weakened.
- **R-T069b — THE CONTROL PAIR (the teaching signal).** The graded `io` (green) against a
  **marginal-preserving shuffle** of the same `io` sequence (red): identical distribution, pairing to
  the eligibility destroyed. The shuffled arm must NOT learn. A rule that improves with its teaching
  signal shuffled is fitting the plant's own regression to the mean, not the error.
- **R-T069c — the untrained twin.** A tissue whose rule never runs must not improve.

### 2.4 Statistic and bar

S = 8 seeds, the plan's statistic: paired one-sided t > 1.895 with exact sign-flip p ≤ 0.05, ties at
1e-6, DEGENERATE below 5 non-tied. The outcome per arm is the plant error over the last third of the
run (mean |e|), paired against the same seed's control.

### 2.5 e-prop transport freedom, MEASURED at adoption

The update reads exactly two quantities: a LOCAL presynaptic eligibility trace and the scalar
teaching signal delivered to that Purkinje. **No forward weight is transposed into the update**, and
no gradient is propagated. Declared `GRADIENT_CLAIM["CFLTDRule.learn"] = ("LOCAL", ...)` and asserted
by `substrate/eprop_transport_gate.py` + `substrate/gradient_claim_gate.py` in the battery. The
measurement, not the declaration, is what discharges the row: the rule is run with the forward
weights REPLACED by a random matrix of the same shape, and the updates must be **bit-identical** —
if a transposed forward weight were read anywhere, they could not be.

### 2.6 P2 adjudication, stated before the build

`cerebellum/gain_adaptation.py::GainAdapter` already does online LMS/delta calibration of a
compensatory gain from a residual error, with climbing-fibre framing. It is a SCALAR mechanism with
no tissue; this row's rule learns PF weight ROWS on the built tissue. They are not two
implementations of one job at the same grain — but if this rule can do the VOR job the scalar one
does, `GainAdapter` becomes a P2 retirement candidate. That comparison is **reported here and acted
on in its own arc**, never folded into this one (one change per commit).

### 2.7 Disposition on a null (P8 clause 5)

If R-T069a shows a single global constant matches per-channel windows, the window-identity claim is
DELETED, not weakened, and the row closes on that null. If R-T069b's shuffled arm learns as well as
the graded one, the rule is measuring nothing and the module is removed rather than parked.

## 3. Results — the rule was BUILT, MEASURED, and DELETED ON ITS OWN NULL (P8 clause 5)

### 3.1 What was built, and what it discharged

`cerebellum/cf_ltd_rule.py`: `EligibilityBank` (a delay line of PF vectors plus a decaying trace),
`CFLTDRule` (covariance/LMS over the Purkinje PF rows, clipping at zero to preserve the tissue's PF
sign identity), `Channel` (the tissue behind `relay_depth` buffered mossy hops), and two plants.

**E-prop transport freedom: MEASURED, and it PASSES.** The rule was run twice from identical state,
the second time with the tissue's forward PF weights replaced by a different random matrix. The
applied updates are **bit-identical**. The naive form of this check FAILED first, and its failure was
the check's, not the rule's: the zero-clip reads the current weight, so with weights near the
boundary the applied delta differs for a reason that is not transport. Both arms are now held far
from the boundary and the check ASSERTS `clips == [0, 0]`, so a green is readable. This is the one
part of the row that is fully discharged.

### 3.2 The window: STRUCTURAL, and only relay depth moves it

18 tissue instances (`n_mossy` ∈ {4,8,12,16,24,32} × seeds {0,1,2}) all measure **window = 2**, zero
refusals. `tau_m` 10→80 does not move it. Threshold does (2,3,5,7,23) but that tunes a neuron
constant. **Relay depth moves it exactly one step per hop (2,3,4,5)** and is a property of the
channel, so it is what the two-channel red is built on.

**An instrument defect this arc's own control caught.** The first form pinned the rule to
`tissue.coincidence_window` — but the relay hops are OUTSIDE the tissue, so both channels got window
2 and the "per-channel" arm was bit-identical to the "one constant" arm on all 8 seeds (mean 0.0,
0 wins, 0 losses). The red was unreachable while wearing the name of the fix. `CFLTDRule` now pins to
`channel.measured_latency()`; depth 0 → 2 and depth 2 → 4, verified before any verdict was read.

### 3.3 The single-gain VOR plant has NO USABLE HEADROOM for a pure-LTD rule

Sweeping the world gain, with the best a uniform PF scale can reach as the baseline:

| world gain | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| headroom | +0.039 | +0.021 | +0.008 | **0.000** | +0.009 | +0.047 | +0.074 |
| best scale | 1.5–2.0 | 1.5–2.0 | 1.0–1.5 | 1.0 | 0.0–1.0 | **0.0** | **0.0** |

Both regimes with headroom are unusable: at w ≤ 0.2 the optimum needs **potentiation**, which
climbing-fibre LTD structurally cannot deliver; at w ≥ 0.9 the optimum is scale **0.0**, where the
task degenerates to "silence Purkinje" and rewards saturation rather than calibration. In between
there is nothing to learn. **A null measured there is unreadable in both directions** — which is why
the first verdict on this plant (taught vs untrained t 1.375, R-T069a t −0.206) is not the row's
answer.

### 3.4 The context plant — headroom appears, and the rule captures it

Two contexts at matched total drive but DISTINCT mossy patterns, each with its own required gain
(0.35 / 0.95), so no single uniform scale serves both. Held-out seeds 100–107, 900 steps.

| arm | vs untrained | verdict |
|---|---|---|
| **TAUGHT** | t **15.544**, p 0.0039, +0.0189, 8/0 | clears |
| **SHUFFLED** (the control that is not the method) | t **4.707**, p 0.0078, +0.0151, 7/1 | **also clears** |
| **TAUGHT vs SHUFFLED** — the pairing-specific component | t **1.242**, p 0.1328, +0.0038, 6/2 | **DOES NOT CLEAR** |

### 3.5 The verdict: the rule's gain is NOT error-paired, and the row DELETES

R-T069b was pre-registered as *"the shuffled arm must NOT learn; a rule that improves with its
teaching signal shuffled is fitting the plant's own regression to the mean, not the error."* **It
learned.** And the pairing-specific component — the only part attributable to the teaching signal
being the right one — does not clear the bar.

**The null was attacked three ways before it was accepted, and survived all three:**

1. **Is the eligibility trace swamping the window?** No. Sweeping `tau_elig` ∈ {1, 1.5, 2, 4, 8, 16},
   the own-window-minus-constant difference stays in [−0.0006, +0.0017] with 2/8–5/8 wins. There is
   no timescale at which the window separates.
2. **Does the plant have headroom?** §3.3 and §3.4 — the context plant does (+0.019), and the rule
   captures essentially all of it (+0.0189). The gain is real; its ATTRIBUTION is what fails.
3. **Is the shuffle too weak a control — does it preserve the error's sign?** No. Error signs are
   balanced (52–55 % positive across seeds), so the shuffle genuinely destroys the pairing. But the
   error carries a persistent **+0.076 DC bias**, and the context plant's headroom is still almost
   entirely UNIFORM (best-uniform +0.0188 against the rule's +0.0189). The rule is performing a
   global depression that any same-marginal signal produces.

R-T069a's own-beats-constant on the context plant (t 2.436, +0.00134 — 7 % of available headroom) is
**not readable as a window-identity result**, because it is measured on a rule whose gain is not
error-paired in the first place. A window effect on top of a mechanism that is not doing credit
assignment is not evidence that the window carries credit.

**Disposition, per §2.7 and P8 clause 5: `cerebellum/cf_ltd_rule.py` is DELETED, not parked.** The
licence working, not failing. What the row keeps is this record and four measured facts: the tissue's
latency is structural and only relay depth moves it; the transport-freedom check passes and how to
build it so a green is readable; the single-gain VOR plant cannot test a pure-LTD rule; and a
context plant with uniform-dominated headroom cannot either.

**What a re-proposal owes.** A plant whose headroom is NOT capturable by a uniform scale — the
uniform-scale baseline must be measured and must leave room — and a demonstration that the
pairing-specific component (taught vs shuffled) clears on its own, before any window claim is read.
The binary-after-cancellation teaching afferent stays declared and unbuilt.

**P2, settled by this null:** `cerebellum/gain_adaptation.py::GainAdapter` is NOT superseded. The
tissue rule was to be the candidate that might retire it; it measured nothing, so the scalar
mechanism stands unchallenged and no retirement is proposed.
