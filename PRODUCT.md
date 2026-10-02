# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary user is the project owner, operating a live brain: starting and watching the daemon,
driving the SO-101 arm, reading telemetry, and talking to the brain through chat. The operator is
the person who wrote the system, so no term on the page needs defining for them.

**Confirmed 2026-08-29: the surface is intended to become public-facing** — published, streamed,
or embedded where people who have never seen this project will read it. That audience does not
share the operator's vocabulary and cannot be assumed to know what any plate means. Both audiences
are real and the second one is new; nothing on the page was designed for it yet.

Display contexts are all four, confirmed: one desktop monitor, a second always-on monitor read
ambiently at distance, a laptop screen, and phone/tablet. Genuine mobile layout is in scope, not
merely non-breakage.

## Product Purpose

`infra/cockpit.py` is the ONE frontend for the spiking-neural-data-lake brain — a single page
carrying live brain state, the live arm, the hardware target, and chat.

Under `docs/POLICIES.md` P4c it is not a dashboard beside the system. It is **the brain's
sensorimotor surface**: chat and controls in are afferent, the arm buttons and E-STOP out are
efferent, and the page is the brain's own report of its own state. That framing is load-bearing,
not rhetorical — it is what makes a dead button a dormancy defect rather than a cosmetic one.

It replaced four earlier surfaces (a matplotlib 3D window, `chat_app`/`spiking_chat`, the
`brain_activity_3d` / `brain_netplot` / `brain_region_activity` views, and a read-only dashboard
inside `brain_telemetry`) which disagreed about what was live and none of which could show the arm
and the chat together. Those are retired.

Success is that an operator can tell, at a glance and without being misled, what the brain is
actually doing right now.

## Positioning

**The honesty model is the product.** A dashboard is the easiest place in a project to lie, and
this one is built so it cannot:

- Anything the snapshot cannot supply renders as an **em-dash**, never a zero bar.
- Anything the snapshot *can* supply but that does not mean what a live bar implies — frozen at
  construction, sleep-only, a literal, a duplicate — is drawn with its `brain_telemetry.PROVENANCE`
  tag beside it.
- **Rotation is a CAMERA, labelled as one, and it is the only thing that moves without data.**
  Activity is carried by colour and radius only.
- The tracts between regions are the real typed connectome (`snn_data_lake.connectome`), not
  decoration.

This is a deliberate inversion of the surface it borrowed its layout from, whose neuron cloud
advances every frame regardless of spikes so that ~100 frames of frozen spike data read as neural
activity. A neighbouring project could copy the layout; it could not truthfully copy the claim
that every moving thing on the page is backed by a measurement.

## Operating Context

- Runs at `http://127.0.0.1:8137`. **Binds loopback only, by design** — `serve()`'s own docstring:
  a window onto a local process, not a network service, and `/api/chat` accepts writes, so it must
  not become reachable off the machine by default.
- `serve()` never constructs a Brain (P4 — exactly one cognition). It is handed the live one.
  `main()` is the only place that may build a brain, and it takes the daemon's single-instance lock
  first, so a cockpit and a daemon can never be two writers on one durable lake.
- Chat does not run on the request thread. `Brain.respond()` writes `_help_drive` into the motor
  path the tick consumes, exposes the language organ to plasticity the sleep tick durably saves,
  runs an R-STDP update, and can route a command through the gated arm. There is no lock anywhere
  in the brain stack, so text is handed to a `Turns` queue owned by the tick loop and the reply is
  picked up by polling. A reply can take minutes; the page says so rather than hanging.
- Poll rates are deliberate and unequal: `/api/all` at 1 Hz for numbers and bars, a separate faster
  loop for `/api/arm.png`.
- The daemon must be running for the page to show anything live. Without it the page renders and
  reports absence honestly.

## Capabilities and Constraints

**Surfaces on the page:** a 3D scene with orbit/dolly camera; arm controls (HOME, six ±6° joint
jogs across shoulder pan / shoulder lift / elbow flex, and E-STOP); a trace canvas; the right
column (ARM, MOTOR MIX, BEHAVIOR & EMOTION, HARDWARE TARGET — NOT BUILT, NEUROMODULATORS, WORLD
MODEL); chat, labelled "canonical `Brain.respond`"; and an eight-cell bottom bar (COVERAGE, REACH,
STABILITY, AFFECT, ACCESS, LIFE, LAKE & CENSUS, DEVICE ROUTING).

**Implementation constraints, all confirmed in the tree:**

- Python stdlib `http.server` / `ThreadingHTTPServer`. The entire HTML, CSS and JS is inlined in
  `cockpit.py` as strings. **No framework, no bundler, no build step, no npm dependency.** Any
  design work lands as hand-written markup and CSS inside a Python module.
- Arm commands go through an allow-list (`brainstem.serial_arm.is_valid_command`), separate from
  the camera endpoint. Camera drag/wheel go to `/api/look`; actions go to the allow-listed
  `/api/act`. The two must not be merged.
- `DEVICE ROUTING` is explicitly **not brain state** — it comes from `infra/device.decisions()`
  and is a fact about the process. It is named as an exception rather than left for a reader to
  discover.
- `HARDWARE TARGET — NOT BUILT` says so in its own heading. A plate that reports an absent thing
  labels itself.

**Governance that binds every change here** (`docs/POLICIES.md`):

- **P4c** — the cockpit is governed on the same terms as any brain module: registered in `REGIONS`
  and `run_selfchecks.py`, covered by the three P3 gates, held to P1, P4b, P7 and P8.
- **P3** — a control wired to nothing is a motor intention that reaches no muscle. This is not
  theoretical: `serve()` once accepted an `acts` queue and did not forward it, so every arm control
  including the E-STOP was inert while the page rendered and the buttons pressed. That defect is
  what produced P4c.
- **The page may display what it measures and never what it cannot.** P8 binds hardest here.
- **Anything that observes the brain must be cheap enough not to become the load.** A ~1 s
  static-import-graph census once sat inside `snapshot()`, which the cockpit polls at 1 Hz, so the
  dashboard made the daemon re-scan its own source every second and starved the tick loop it
  existed to display. Every suite stayed green throughout.
- E-STOP and the reflex/cortisol safety paths are inside the five-file brainstem safety envelope,
  hash-gated, and modified only with explicit owner approval.
- One change per commit, full gate green.

**EXPLICITLY UNDECIDED — recorded, not resolved (2026-08-29).** The confirmed intent to make this
surface public-facing, and to read it from a phone, **contradicts the loopback-only binding that
`serve()` currently enforces and documents as a safety property.** The page carries arm controls
and an E-STOP over an endpoint that accepts writes. Reaching it from a phone or publishing it means
deciding how — a read-only published mirror, an authenticated tunnel, a separate public surface
with the efferent half removed, or something else. **No design work should assume this is settled,
and nothing in this record licenses widening the bind.** The decision is the owner's and has not
been made.

## Brand Commitments

**Confirmed 2026-08-29: the incumbent visual language is NOT binding and may be replaced.** The
current look — glass plates on a dark ground, a 1px cyan border, the L-shaped layout with a
weighted right column over a bottom bar, bar rows aligned on a common track, threshold recolouring
— is taken from mhflocke's Bittle dashboard (`github.com/MarcHesse/mhflocke`,
`src/viz/bittle_dashboard.py`) and credited in `cockpit.py`'s module docstring. It was adopted as a
good, information-dense starting point, not as an identity.

**Consequence that must not be dropped: the attribution is tied to the borrowed design.** If a
redesign replaces that visual language, the docstring credit has to be revisited in the same change
— removed if nothing borrowed remains, narrowed if only part does. Replacing the design and
leaving the credit standing would make the file claim a provenance that is no longer true, which is
the same class of defect the PROVENANCE model exists to prevent.

The honesty model itself — em-dash for absent, PROVENANCE tags, camera-not-data — is **not** part
of the replaceable visual language. It is product truth and survives any redesign.

No other name, logo, palette, typeface or voice commitment has been established.

## Evidence on Hand

- Live telemetry from a running brain via `brain_telemetry.snapshot()` and `verify_contract()`,
  with a real PROVENANCE vocabulary distinguishing live from frozen/sleep-only/literal/duplicate.
- The real typed connectome (`snn_data_lake.connectome`) for inter-region tracts.
- A real SO-101 arm behind the allow-listed command path, with a rendered `/api/arm.png` view.
- `docs/POLICIES.md` (2027 lines) as the authoritative governance record, including the measured
  incidents cited above.

**Absences future work must not fabricate:** there are no users beyond the owner, no testimonials,
no benchmarks about this UI, no adoption numbers, no uptime claims, and no third-party validation.
`HARDWARE TARGET` is NOT BUILT and its plate says so. Nothing may be added to this page that
implies a measurement the brain does not produce.

## Product Principles

1. **Absence renders as absence.** An em-dash, never a zero, never a plausible-looking default. If
   the page cannot know it, the page says it cannot know it.
2. **Nothing moves that is not data, except the camera — and the camera is labelled.** Motion is
   the most persuasive lie a dashboard can tell.
3. **Every control reaches a muscle.** A button wired to nothing is a dormancy defect, adjudicated
   as a faculty failure, not a cosmetic one.
4. **The observer must not become the load.** Any work here is measured against the tick loop it
   exists to display.
5. **The efferent half is safety-critical.** Arm commands and E-STOP sit behind an allow-list and a
   hash-gated safety envelope. Design may change how they look; it may not change what they are
   gated by.

## Accessibility & Inclusion

No product-specific accessibility standard has been established. Two facts constrain any future
one, both confirmed rather than assumed:

- The surface is read on a second monitor at a distance and on a phone, so legibility at small
  size and at distance are both real requirements, not hypotheticals.
- The page currently carries meaning in colour (threshold recolouring, activity as colour and
  radius). With a public audience confirmed, colour-only encoding is a known gap. It is recorded
  here as a gap, not yet as a requirement — no standard has been chosen.
