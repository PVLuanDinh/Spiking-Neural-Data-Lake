# manifest-baseline — the feature's counted record (FR-012)

Numbers here are DATED OBSERVATIONS recorded verbatim from tool output. Nothing
else in the feature restates them; re-run the named command for the live value.

## Pre-feature state (T001, 2026-08-18, tree 07bc35a9 + working files)

- `ruff check .` -> exit 0 ("All checks passed!")
- `doc_budget_gate` -> exit 0; `docs/POLICIES.md 29750 / 29750  0.0%` (zero headroom,
  as the spec's audit section records)
- `pytest -q` (full suite) -> exit 1, **90.46% coverage, exactly ONE failure**:
  `test_core_selfcheck_in_process[brain_atlas]` —
  `AssertionError: substrate: modules on disk but not in the atlas map: ['policy_manifest']`.
  **Caveat recorded honestly**: the run raced T002 — the new module file landed on disk
  mid-suite, and the atlas self-check correctly refused an unmapped module. The failure
  is INDUCED by this feature's first file, not pre-existing (the same tree without the
  file is commit `07bc35a9`, green in CI). It is also a live positive for the
  registration gate: a file cannot exist here without joining REGIONS.
  **Sequencing consequence**: the module's first COMMIT must carry T003's registration
  and a functioning bare `--assert` together (the gradient_claim_gate commit shape);
  T003's "after T016" stays the EXECUTION order, and the commit boundary sits after it.

## First counted run (T012, 2026-08-18, tree 07bc35a9 + feature working files)

The FR-012 count that unblocks any later policy-text edit (US4). Verbatim:

```
POLICY MANIFEST -- derived from docs/POLICIES.md (regeneration is the only edit path)

id               kind                   command(s)
P0               NOT MACHINE-DECIDED    NONE
P1               NOT MACHINE-DECIDED    NONE
P2               NOT MACHINE-DECIDED    NONE
P3               absolute               substrate/dormancy_gate.py; substrate/method_dormancy_gate.py; substrate/import_manifest.py
  P3 amendment   NOT MACHINE-DECIDED    NONE
P4               absolute               substrate/replication_guard.py
  P4b            NOT MACHINE-DECIDED    NONE
  P4c            ratchet                tools/p4c_ratchet.py
  P4c amendment  NOT MACHINE-DECIDED    NONE
P5               NOT MACHINE-DECIDED    NONE
  P5a            NOT MACHINE-DECIDED    NONE
P6               ratchet                tools/p6_gate.py --assert
P7               absolute               tools/dependency_ledger.py --assert
P8               absolute               tools/red_reachability.py --assert
e-prop           partial                substrate/eprop_transport_gate.py; substrate/gradient_claim_gate.py

rows: 10 policies (5 sub-clauses nested) -- reconciled: parents + table_only == table rows

UNCLAIMED_GATE: doc_budget_gate, ingest_guard, safety_envelope
NOT_IN_CI:      none
self-disposition: policy_manifest (this instrument) is EXCLUDED from its own unclaimed-gate census -- it is the census taker, not a policy gate; its liveness is asserted by the CI step and run_selfchecks entry that run it. Printed on every report so the exclusion is never silent.
```


## US2 live runs (T013-T016, 2026-08-18)

- `--self-check` exit 0: FAIL, CANNOT_RUN, NO_GATE, empty-walk, reconciliation and certify reds
  ALL reachable.
- `--assert` exit 0 after `--manifest --write`; it first went RED against a stale committed copy
  (the p4c --assert flag-inheritance change) -- the sync assert demonstrated live before ever
  passing. Baseline census 3/3.
- First full `--audit` (every real gate executed, subprocess): exit 1 with `P4c FAIL --
  p4c_ratchet --assert exit 1` -- a REAL red: the not-yet-registered module counted as an
  ungoverned build (NEW ZONE). T003's registration (REGIONS + run_selfchecks +
  dormancy_gate.EXEMPT) cleared it; second audit: all mechanized rows PASS, exit 0.
- `--certify` exit 1, REFUSED naming exactly the census: doc_budget_gate, ingest_guard,
  safety_envelope. `tools/policy_certification.json` written with certified: false.
  SC-002's reachable red demonstrated on the real tree.

## US3 verification (T023, 2026-08-18)

- Hook (`sndl-policies-guard.ps1`) executed: manifest block present, output BYTE-MATCHES a fresh
  `--manifest` run; delivery block is a thin caller (subprocess + passthrough, zero embedded
  policy text); failure marker is a visible one-liner by construction.
- Scratch feature generated from templates: plan carries the runnable
  `policy_manifest --audit` check and the POLICIES.md pointer. All five templates now reference
  POLICIES.md (measured 0/5 before T020/T021). Scratch dir deleted; feature.json restored to 021.

## Relocatable-mass count (T024, 2026-08-18 -- the count that DECIDES the relocation order)

Counter: `doc_budget_gate.count_words` (the arbiter). Verbatim:

```
TOTAL (gate counter): 29750
target: recover >= 1191 words (SC-006: below 28,559 from 29,750)

  words  section
   6770  ## Cross-cutting invariants (never relaxed by any policy)
   3735  ### P5a � TWO AXES, NEITHER SUBSTITUTES FOR THE OTHER (owner directive, 
   2839  ## P6 � Every build must be able to reach every other, and actually use 
   2802  ## P0 � The brain must autonomously improve itself, by every available a
   1631  ### What P7 does not relax
   1434  ### P4c � EVERY BUILD IS PART OF THE BRAIN
   1244  ### Real past incidents, each re-verified against the tree 2026-08-04
   1175  ### P3 amendment � dormancy is per-METHOD, not per-module
   1062  ## P2 � Adopt-better **or merge**, then retire the inferior (non-additiv
    716  ## P1 � Brain-native / no-hardcode
    642  ### P4c amendment � A BUILD THAT HAS NOT REACHED `main` IS NOT A BUILD T
    609  ### Real past incidents, each re-verified against the tree 2026-08-04
    601  ### What P8 does not relax
    567  ### The dormancy family � six observed shapes of "live but inert"
    503  ## P3 � No dormancy
    436  ### P4b � THE DURABLE LAKE IS PART OF THE BRAIN
    332  ## P8 � Invent the method. The literature is a starting point, not a bou
    322  ## P7 � Any dependency that helps this become a phenomenal brain is acce
    263  ### On sentience specifically
    247  ### What an invented method owes
    216  ## P4 � One brain / no replication
     12  ## P5 � Use the GPU for everything it actually speeds up

cross-cutting (holds the e-prop block): 6770
sum of top-12 non-cross-cutting sections: 17889
```

**Decision by count (clarification 3)**: the two self-labelled incident-forensics sections
(P7's 1,244 + P8's 609 = 1,853 words) exceed the 1,191 target on their own. The e-prop
block is NOT touched -- reachable without it, per the clarification's rule.

## Post-consolidation state (T025-T030, 2026-08-18)

Budget gate (exit 0) verbatim:

```
==============================================================================
DOC BUDGET GATE -- word ceilings on the STANDING documents
==============================================================================
status      words / ceiling   headroom  path
ok           1014 /    1065      4.8%  AGENTS.md
ok           2156 /    2264      4.8%  README.md
ok            465 /     488      4.7%  SECURITY.md
ok           1874 /    1968      4.8%  docs/ANATOMY.md
ok          28065 /   28083      0.1%  docs/POLICIES.md
ok           1942 /    2002      3.0%  docs/POLICIES_EVIDENCE.md

gate OK
```

Certify (exit 0) verbatim -- the FIRST GREEN CERTIFICATION, after T028 claimed the census:

```
POLICY AUDIT -- every manifest row's command executed (subprocess, never import)

P0               REVIEW_ONLY  no machine decides this policy
P1               REVIEW_ONLY  no machine decides this policy
P2               REVIEW_ONLY  no machine decides this policy
P3               PASS
P4               PASS
P5               REVIEW_ONLY  no machine decides this policy
P6               PASS
P7               PASS
P8               PASS
e-prop           PASS
P3 amendment     REVIEW_ONLY  no machine decides this policy
P4b              REVIEW_ONLY  no machine decides this policy
P4c              PASS
P4c amendment    REVIEW_ONLY  no machine decides this policy
P5a              REVIEW_ONLY  no machine decides this policy

UNCLAIMED_GATE: none
NOT_IN_CI:      none
self-disposition: policy_manifest (this instrument) is EXCLUDED from its own unclaimed-gate census -- it is the census taker, not a policy gate; its liveness is asserted by the CI step and run_selfchecks entry that run it. Printed on every report so the exclusion is never silent.

CERTIFIED (full=False) -- record: tools\policy_certification.json
not machine-decided (disclosed, never green): P0, P1, P2, P5, P3 amendment, P4b, P4c amendment, P5a
```

T029: both constitution 'gate is specified but not built' sentences (false since 2026-08-05)
converted to pointers at POLICIES.md's enforcement rows; sweep found no further contradicting
restatements (:195 is P6's accurate corrected status; :317 is the table header).


## Final full-gate pass (T032, 2026-08-18/19)

- `run_selfchecks.py` -> exit 0, **307/307** (the module's bare == --assert entry among them)
- `pytest -q` (whole suite) -> exit 0, coverage **90.83%** (>= 70% required)
- `ruff check .` -> exit 0
- `policy_manifest --assert` -> exit 0 (re-verified directly; a -1 seen in the batch log was a
  PowerShell pipeline artifact, not the tool)
- `git diff --stat -- src/snn_data_lake/brainstem/` -> **0 lines** across the whole feature (FR-013)
- Certify-job ran-check (post-push half of T032): **CHECKED 2026-08-19, and the check CAUGHT
  EXACTLY WHAT IT EXISTS TO CATCH.** Run 32210198212 (fd2c527e) concluded "failure" with every
  job completing in 1-3 SECONDS -- nothing executed. GitHub's annotation, verbatim: "The job was
  not started because recent account payments have failed or your spending limit needs to be
  increased." The SAME zero-execution failure applies to the runs for 07bc35a9 and f61e4a94 --
  no CI job has actually run since before 2026-08-18 23:14Z. All three commits were verified
  green LOCALLY (full battery, recorded above), so the work stands; CI re-verification is
  pending an OWNER action (GitHub Billing & plans -- payment method / spending limit). Re-run
  this check after billing is restored: the certify job must show nonzero duration AND a
  published policy-certification artifact.
  **RESOLVED 2026-08-19 (run 32212496578, commit c98e7573, after billing restored + two fixes):**
  the first real run failed two ways, both this feature's own -- the certify job pinned "3.14"
  (GIL build; the import guard refused, the exact trap ci.yml's own comment documents) and
  policy_manifest.py drifted from `ruff format` (the local gate ran `check`, not
  `format --check`). Fixed in c98e7573. Result: ALL SIX JOBS SUCCESS -- certify ran
  **3m01s** (nonzero duration), job log verbatim: "CERTIFIED (full=False) -- record:
  tools/policy_certification.json / not machine-decided (disclosed, never green): P0, P1, P2,
  P5, P3 amendment, P4b, P4c amendment, P5a / UNCLAIMED_GATE: none", and the
  `policy-certification` artifact PUBLISHED (728 bytes zipped, expires 2026-11-17).
  FR-014 verified end to end: a landing build cannot avoid being audited.

## POLICIES.md audit of the completed feature (T033, 2026-08-19)

- **P1** PASS -- constants are measured/derived with stated reasons: TABLE_ONLY_IDS (structural,
  commented), GATE_NAME_RE (stated + tested predicate), BASELINE_UNCLAIMED (counted ratchet
  baseline, now empty), ceilings = measured + stated slack.
- **P2** PASS -- the manifest DERIVES from the enforcement table (no second hand-edited copy);
  committed JSON has one edit path (--write) plus a byte sync assert; relocation left pointers,
  never copies.
- **P3** PASS -- REGIONS + run_selfchecks + dormancy_gate.EXEMPT (permanent, reasoned); live path
  = CI lint --assert + selfchecks bare. The registration was FORCED live twice before commit
  (brain_atlas unmapped-module red; p4c NEW ZONE red) -- both observed, both cleared by T003.
- **P4/P4b/P4c** PASS -- no brain replication; tool JSON records are not a second durable store;
  landed governed from the first file.
- **P6** PASS -- a CI gate, not a tick faculty; same disposition as every sibling gate.
- **P7** PASS -- stdlib only; no dependency added, none rejected.
- **P8** PASS -- reachable reds constructed (--self-check: 6) AND observed live (stale-sync red,
  p4c NEW ZONE red, census 3); control independent of the method (two independent parses;
  detector-liveness fixture after the census flip); unprompted disclosures recorded (baseline
  race caveat, self-disposition printed on every report); deletion-on-null not applicable.
- **e-prop / P5** N/A -- no gradients, no GPU surface.
- **Cross-cutting** PASS -- safety envelope untouched (0 lines, measured); count-first held
  (T012's counted run precedes every policy-text edit); full gate green at commit time; symbols
  cited over line numbers throughout.
