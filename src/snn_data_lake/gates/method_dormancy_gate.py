"""
method_dormancy_gate -- Policy 3, per-METHOD. The amendment's mechanization, finally built.

WHY THE MODULE GATE IS NOT ENOUGH. gates/dormancy_gate.py keys on the module basename, so a module
passes P3 clean with half its capability dead. The worked case: readout/predictive_coding.py settled its
3-level Rao-Ballard net on every wake tick with the error-neuron architecture implemented faithfully,
while PredictiveCodingNetwork.learn() had exactly ONE call site in the tree -- inside the module's own
__main__ self-check. Inference wired, learning never called, weights frozen at their random seed for the
life of the feature, and the module-level gate saw a live, imported, fully-compliant module. Fixed in
v4.71; the blind spot it proves is not.

THE RULE. If a module on a live path computes a prediction error, a free energy, or any other learning
signal, the update rule that consumes it must also be called from a live path.

THREE DETAILS ARE LOAD-BEARING, and each was found by writing the naive version first and watching it
fail. This is the "implement the gate and count its output before adopting the rule" discipline applied
to the gate itself:

  1. IMPORT-SCOPED RESOLUTION. Matching bare attribute-call names across all reachable modules scores 0
     failures / 47 UNCERTAIN -- the name `learn` resolves in 20 unrelated modules, so the naive pass
     cannot even catch its own motivating case. A call site counts only if its module imports the
     defining module.
  2. SAME-MODULE CALLERS COUNT, EXCEPT UNDER `if __name__ == "__main__"`. Excluding the defining module
     produces a real false positive: plasticity/meta_plasticity.py drives its own private _Unit.learn
     from MetaPlasticLearner.run(). The __main__ guard is precisely what isolates the genuine case.
  3. A FUNCTION USED AS A VALUE IS USED. reasoning/arc_mint.py registers its fitters in a dispatch table
     -- `("affine", 3, _fit_affine)` -- so a pass that inspects only ast.Call nodes reported all five as
     dead. That is 5 false positives out of 12 on the first run. Bare Name/Attribute references count.

HONESTY GUARDS, inherited from P3 and not negotiable:
  (a) A call site proves a CALLER, not a CHARGE. Adding a dead consumer to silence this gate is the
      failure P3 already forbids; charge is proven separately by the runtime neuter-A/B.
  (b) An unresolved name is UNCERTAIN and hand-adjudicated, NEVER auto-dormant. This gate therefore
      reports three buckets and only ASSERTS on the first.
  (c) EXEMPT_METHOD entries carry a disposition and a reason, and `wire` entries that outlive their
      wave are reported as a standing TODO -- so exempting everything stays visible rather than silent.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.gates.method_dormancy_gate
"""

import ast
import os

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Method-name fragments that mark an UPDATE RULE -- the thing that consumes a learning signal. Kept
# deliberately broad: a false positive lands in a bucket a human reads, a false negative is invisible.
LEARN_VERBS = (
    "learn",
    "update",
    "train",
    "adapt",
    "consolidate",
    "reinforce",
    "fit",
    "plast",
    "potentiate",
    "depress",
    "calibrate",
)

# name -> (disposition, reason). disposition in {"permanent", "wire", "retire"}.
# "permanent" = legitimately not on the live tick. "wire"/"retire" are open work and are REPORTED.
EXEMPT_METHOD = {
    "reservoir.spontaneous_activity.SpontaneousActivity.fit_to_traces": (
        "wire",
        "specs/022 T045: closed-form AR(1) theta fit (DIRECT_SOLVE under the class-qualified "
        "GRADIENT_CLAIM key) -- the LEARNABLE arm of the spontaneous bootstrap. The bootstrap "
        "is live ONLY where the torch perception tier is armed (the eprop checkpoint is "
        "gitignored, so a fresh clone and CI never fire the wire; "
        "tests/test_spontaneous_bootstrap.py pins the CODE PATH with a stub encoder, not the "
        "deployment -- T045 panel). The fit's callers today are the module's own main(), "
        "which every gate run executes. The dated wire: feeding REAL percept traces into the "
        "fit -- no task id in specs/022 tasks.md names it yet, stated plainly, AND that wire "
        "owes its own A/B first: the panel measured a small fitted theta DEGRADING the "
        "alignment criterion (Oja is order-blind; smooth real traces fit small theta), so a "
        "trace-fit landing without the A/B would silently worsen the one measured outcome",
    ),
    # cortex.genomic_bottleneck.GenomicMap.fit left this table 2026-09-02: the FR-033 wire
    # landed at 847c5f6a (the engine refit the map on the earned substrate under a non-default
    # pool mode), its paired A/B at 24 seeds was a NULL, and the module was DELETED in the
    # following commit per its own pre-registration. Record: specs/022
    # evidence/s1-genome-fanin-ab.md.
    # The next three entered the census at T053-ii (2026-08-31): rule_composer's census
    # imports made their hosts REACHABLE (the module-gate rows retired on it), and this
    # per-METHOD gate scopes by reachable() -- import-topology arriving, not a wire.
    "cerebellum.spiking_jepa.SpikingJEPA.train_step": (
        "permanent",
        "TRAIN-TIER world-model trainer on a kept-unadopted model (the module docstring's "
        "adopt-or-retire record: encode() invariant across live widths, NOT a drop-in). "
        "Charged surface: its own registered self-check + the T051b-i/T052-i declaration "
        "equivalence arms that spy its call sites; a live wire would be a P2 adoption "
        "decision, never a call added for this gate",
    ),
    "cerebellum.spiking_jepa._SpyJEPA.train_step": (
        "permanent",
        "harness SPY subclass -- records production call sites for the declaration "
        "equivalence arms; exists only inside the module self-check",
    ),
    "plasticity.espp_rule._SpyESPP.train_step": (
        "permanent",
        "harness SPY subclass -- the espp arms' call-site recorder; self-check only",
    ),
    "train_image": ("permanent", "deep_stdp_2stage: an offline [train]-tier trainer entrypoint"),
    "train_model": ("permanent", "training/: offline GPU trainer entrypoints (snn-train CLI)"),
    "train_with_offload": ("permanent", "training/offload: offline GPU trainer entrypoint"),
    "learned_map": (
        "permanent",
        "drive_reduction: a READOUT of the learned state->action map, not an update rule -- the verb "
        "match fires on 'learn' inside an accessor name. Kept as a named false positive rather than "
        "narrowed away, because narrowing the heuristic to dodge it would also hide real accessors "
        "that SHOULD be consumed",
    ),
    "consolidate": (
        "wire",
        "AffectiveEpisodicStore.consolidate (faculties/affective_memory.py:173) -- amygdala-weighted "
        "replay sampling whose only callers are inside its own main(). The module IS live, but its one "
        "cross-module importer (engine/autonomous_loop.py:41) calls .store() and never this. Open "
        "P3 work: wire it into the sleep replay and prove the charge with a neuter-A/B, or retire it. "
        "Held as an OPEN exemption so this gate reports it as a standing TODO instead of hiding it. "
        "ADJUDICATED 2026-07-31 (runtime, adversarially reviewed): still dead, and the disposition "
        "should become RETIRE rather than wire -- the promised wiring is not a call site but a "
        "three-file redesign of Episode + ReplayBuffer, and the prioritised-replay capability it was "
        "built for has no consumer today. PORT CHECK (Phase D): the CLASS is live -- constructed at "
        "engine/autonomous_loop.py:1548, which calls .store() and never this -- so retiring the "
        "method does not touch the store. What blocks deletion is P2: amygdala-weighted prioritised "
        "replay is a capability with no home elsewhere, and it must be ported before the method goes",
    ),
    # OPEN dispositions from the Phase D port checks (2026-07-31, specs/001-charge-the-dormant).
    # Kind is NOT `permanent`, so the gate keeps reporting each as a standing TODO rather than hiding
    # it. Every one is dead as a METHOD while its CLASS is live, which is the per-method amendment's
    # whole point -- and not one of them can be retired today, because P2 requires porting a unique
    # capability before deletion and each still owns one.
    # cerebellum.forward_model.ForwardModel.update: row DISCHARGED at specs/022 T037 (2026-08-21).
    # It read "never called on a live path ... retire after confirming the baseline never needs to
    # re-fit" -- the efference-copy boundary (sensorimotor/efferent_boundary.py, observe()) now calls
    # update on EVERY envelope passage of the live loop (AutonomousArm.apply_observed), so the
    # method resolves through its live call site and the retire question is answered: the baseline
    # re-fits online. Recorded here so the row is not re-added; the T037 review panel caught the
    # stale row shipping "never called on a live path" about a method on the live path.
    "faculties.developmental_envelope.DevelopmentalEnvelope.update": (
        "wire",
        "DevelopmentalEnvelope.update -- dormant twice over: never called on any live path, and "
        "discarded even under its opt-in flag. PORT CHECK: the class IS live, constructed at "
        "engine/autonomous_loop.py:1570, so this is a dead method on a live object rather than a "
        "dead module. The anneal mechanism is real and self-check-proven (monotone, deterministic "
        "RNG), which is why the disposition is wire rather than retire -- but wiring it must be a "
        "real consumer of the envelope, never a call added to satisfy this gate (P3)",
    ),
    "basal_ganglia.latent_q.LatentQ.update": (
        "wire",
        "LatentQ.update -- never called on a live path while the module is imported at "
        "engine/autonomous_loop.py:30, the exact shape the per-method amendment was written for. "
        "PORT CHECK surfaces a P2 question rather than a P3 one: basal_ganglia/mlp_q.py documents "
        "itself as the NONLINEAR counterpart 'once described as a strict upgrade that made LatentQ "
        "retirable', and its own docstring walks that claim back. So the honest next step is to settle "
        "which of the two survives on a benchmark, not to wire the loser",
    ),
    # CLASS-QUALIFIED keys. Each names exactly one rule and can suppress nothing else.
    "basal_ganglia.latent_q._TabularQ.update": (
        "permanent",
        "_TabularQ.update (basal_ganglia/latent_q.py:165) -- the self-check's fair TABULAR CONTROL, "
        "constructed only inside main(). A measurement baseline, not an update rule on the brain's "
        "path: a control class that ran live would be the bug, so dormancy here is the correct state "
        "rather than a gap. Adjudicated at runtime 2026-07-31 and adversarially reviewed. This key "
        "MUST stay class-qualified: `basal_ganglia.latent_q.update` would also exempt LatentQ.update "
        "at :102, which is GENUINELY dormant open P3 work, and exempting the real finding in order to "
        "excuse the control is the failure this gate exists to prevent",
    ),
    "readout.predictive_coding.PredictiveCodingNetwork.learn": (
        "permanent",
        "PredictiveCodingNetwork.learn (readout/predictive_coding.py:268) -- the OFFLINE BATCH WRAPPER "
        "over learn_step(), which IS called every wake tick at engine/live_brain.py:185. Verified by "
        "runtime instrumentation to be never called on a live path, and that is CORRECT: wiring the "
        "batch form would run the same rule twice. P2 already holds -- one rule, one definition. It is "
        "kept rather than retired because predictive_coding.py:353's self-check needs it for the "
        "learning-curve assert, i.e. the module's own proof that the rule learns. Re-keyed from the "
        "module-qualified form to the class-qualified one on 2026-07-31, so every disposition in this "
        "table names a class and none can reach a sibling",
    ),
}

# STILL OPEN, and deliberately not exempted: LatentQ.update (basal_ganglia/latent_q.py:102). The module
# is imported at engine/autonomous_loop.py:30 -- so the MODULE gate sees it reachable -- while every
# method of the class is cold. That is exactly the shape the per-method amendment was written for, and
# it is real P3 work: wire it and prove the charge, or retire it with its capability ported first.
# It is NOT exempted here, because an exemption is a disposition and this rule does not have one yet.


def _modname(path):
    return os.path.relpath(path, _PKG_DIR)[:-3].replace(os.sep, ".")


def _parse_all():
    trees = {}
    for dp, _dn, fn in os.walk(_PKG_DIR):
        for f in fn:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            p = os.path.join(dp, f)
            try:
                trees[_modname(p)] = ast.parse(open(p, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
    return trees


def _imports(tree):
    """The in-package modules this file imports, as dotted paths relative to snn_data_lake."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "snn_data_lake" in n.module:
            out.add(n.module.split("snn_data_lake.")[-1])
        elif isinstance(n, ast.Import):
            for a in n.names:
                if "snn_data_lake" in a.name:
                    out.add(a.name.split("snn_data_lake.")[-1])
    return out


def _main_guard_lines(tree):
    """Line numbers that belong to the module's own self-check. Detail 2: these do NOT count as callers.

    DETAIL 2b, and the first version of this gate got it wrong: the `if __name__ == "__main__":` block
    is almost always one line -- `main()`. The self-check's real body lives in a top-level `def main()`
    that the guard calls. Marking only the literal `if` block therefore counts every assertion inside
    main() as a live caller, which is exactly how the first run reported 0 dormant while
    faculties/affective_memory.py:173 consolidate() -- whose only callers are at :242/:252/:253 inside
    its own main() -- sat there dead. So: find the functions the guard calls, and mark THEIR bodies too.
    """
    bad, called = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.If):
            t = ast.dump(n.test)
            if "__name__" in t and "__main__" in t:
                for c in ast.walk(n):
                    if hasattr(c, "lineno"):
                        bad.add(c.lineno)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                        called.add(c.func.id)
    # one level of indirection is enough in this tree: the guard calls main(), main() holds the asserts
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in called:
            for c in ast.walk(n):
                if hasattr(c, "lineno"):
                    bad.add(c.lineno)
    return bad


def audit():
    """-> {"dormant": [...], "uncertain": [...], "open_exempt": [...], "n_defs": int}.

    dormant   = zero uses of the name anywhere in an importing (or the defining) module, outside __main__.
    uncertain = uses exist, but only as a bare same-named attribute in an importing module, so the
                RECEIVER is unproven. Hand-adjudicated per honesty guard (b) -- never auto-dormant.
    """
    trees = _parse_all()
    imports = {m: _imports(t) for m, t in trees.items()}

    # SCOPE, and getting this wrong is how the gate swings from useless to unadoptable. Run 1 marked
    # only the literal `if __name__` block and found 0. Run 2 marked main()'s body too and found 59 --
    # the 58-entry rubber-stamp shape POLICIES.md:24-26 records as fatal to an earlier proposal. The
    # 59 were mostly module-level helpers inside research/train-tier files, which the MODULE gate
    # already scopes out. So this gate takes the amendment at its word -- "a learning-verb METHOD on a
    # REACHABLE module" -- and applies both halves:
    #   (1) the defining module must be reachable from a live root (same set dormancy_gate asserts on)
    #   (2) the def must be a METHOD (inside a class), not a module-level self-check helper
    from snn_data_lake.gates import dormancy_gate

    live = {m.split("snn_data_lake.")[-1] for m in dormancy_gate.reachable()}

    defs = {}
    for m, t in trees.items():
        if m not in live:
            continue
        for cls in ast.walk(t):
            if not isinstance(cls, ast.ClassDef):
                continue
            for n in cls.body:
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if n.name.startswith("__"):
                    continue
                if any(v in n.name.lower() for v in LEARN_VERBS):
                    # Keyed by (module, CLASS, name). The class was already in hand and thrown away,
                    # and dropping it cost twice: two same-named defs in one module COLLAPSED to one
                    # entry whose surviving lineno named whichever came last, and an exemption could
                    # not distinguish them. basal_ganglia/latent_q.py is the case -- LatentQ.update at
                    # :102 is genuinely dormant open work, _TabularQ.update at :165 is the self-check's
                    # control and correctly dead. One key for both meant the finding was reported
                    # against the wrong class, and exempting the control would have hidden the finding.
                    defs[(m, cls.name, n.name)] = n.lineno

    # every use of a name, call or value-reference (detail 3), tagged with whether it is under __main__
    uses = {}
    for m, t in trees.items():
        guard = _main_guard_lines(t)
        defsites = {
            n.lineno for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for n in ast.walk(t):
            nm = (
                n.attr
                if isinstance(n, ast.Attribute)
                else (n.id if isinstance(n, ast.Name) else None)
            )
            if nm is None or getattr(n, "lineno", None) in defsites:
                continue
            uses.setdefault((m, nm), []).append(n.lineno in guard)

    dormant, uncertain = [], []
    for (dm, dcls, name), ln in sorted(defs.items()):
        same_module = any(not g for g in uses.get((dm, name), []))
        importers = [
            cm
            for (cm, cn), flags in uses.items()
            if cn == name and cm != dm and dm in imports.get(cm, ()) and any(not f for f in flags)
        ]
        entry = f"{dm}.py:{ln} {dcls}.{name}()"
        if same_module or importers:
            # A cross-module hit is a NAME match, not a proven receiver -- honesty guard (b).
            if not same_module and importers:
                uncertain.append(f"{entry}  [name-matched in: {', '.join(sorted(set(importers)))}]")
            continue
        # Lookup order: module.Class.method -> module.method -> method. Most specific wins.
        #
        # The bare form exempts a name across all 194 modules, which is far too blunt for a real
        # disposition: exempting `learn` to excuse ONE offline wrapper would also excuse the next
        # PredictiveCodingNetwork.learn -- the exact case this gate was built for, and itself a method
        # named `learn`. Module scope was still not enough: latent_q defines `update` on two classes
        # with OPPOSITE correct dispositions, so `basal_ganglia.latent_q.update` would exempt the
        # genuinely-dormant LatentQ.update in order to excuse the _TabularQ control. Class scope is the
        # first granularity at which every disposition in this tree can be stated without collateral.
        # Bare and module keys stay honoured so existing entries keep working; new dispositions should
        # be class-qualified.
        if (
            f"{dm}.{dcls}.{name}" in EXEMPT_METHOD
            or f"{dm}.{name}" in EXEMPT_METHOD
            or name in EXEMPT_METHOD
        ):
            continue
        dormant.append(entry)

    open_exempt = [
        f"{k} ({v[0]}: {v[1]})" for k, v in sorted(EXEMPT_METHOD.items()) if v[0] != "permanent"
    ]
    return {
        "dormant": dormant,
        "uncertain": uncertain,
        "open_exempt": open_exempt,
        "n_defs": len(defs),
    }


def main():
    a = audit()
    print("=" * 78)
    print("METHOD DORMANCY GATE -- Policy 3, per-METHOD amendment")
    print("=" * 78)
    print(f"update-rule definitions scanned : {a['n_defs']}")
    print(f"EXEMPT_METHOD entries           : {len(EXEMPT_METHOD)}")
    print(f"dormant (asserted)              : {len(a['dormant'])}")
    print(f"uncertain (hand-adjudicated)    : {len(a['uncertain'])}")
    for e in a["dormant"]:
        print(f"  DORMANT   {e}")
    for e in a["uncertain"]:
        print(f"  UNCERTAIN {e}")
    if a["open_exempt"]:
        print("  STANDING TODO -- non-permanent exemptions still open:")
        for e in a["open_exempt"]:
            print(f"    {e}")

    assert not a["dormant"], (
        "update rules with no live-path call site: "
        + "; ".join(a["dormant"])
        + " -- wire them and prove the charge, retire them, or add an EXEMPT_METHOD entry with a reason"
    )
    print(
        f"self-check OK: {a['n_defs']} update rules, none dormant, "
        f"{len(a['uncertain'])} UNCERTAIN reported not asserted"
    )


if __name__ == "__main__":
    main()
