"""Policy-4 replication guard -- there is only ONE brain; no build may re-implement the brain stack.

Policy 4 (user, 2026-07-24): "there is only 1 brain; a build must never replicate brain functionality
-- utilize the original, not re-implement it, because it causes bloat." The canonical brain is a clean
3-layer COMPOSITION (verified): cortex.brain.Brain (cognition + circadian clock) composes
cortex.live_brain.LiveBrain (perception + workspace) composes brainstem.autonomous_loop.AutonomousArm
(action + reward + arm + SAFETY). Composition is fine; RE-DEFINING one of those stack classes a second
time (a parallel "brain") is the bloat.

This gate MECHANIZES "one brain" the same way dormancy_gate mechanizes "no dormancy": it AST-scans
snn_data_lake for definitions of the canonical stack classes and ASSERTS each is defined exactly once,
in its canonical module -- unless the duplicate is on the PENDING_RETIREMENT allowlist (a known replica
scheduled for a Policy-4 consolidation wave, with a reason). A NEW second `class Brain` / `class
LiveBrain` / `class AutonomousArm` that is not allowlisted FAILS run_selfchecks -- so a replica cannot
be introduced silently. The allowlist shrinks to empty as the consolidation waves retire the replicas.

The companion to this guard is cortex.brain.make_brain(): the single sanctioned CONSTRUCTION path.
Consumers obtain the brain through make_brain() rather than calling Brain(...) directly.

  python -m snn_data_lake.gates.replication_guard   # (also run by run_selfchecks.py)
"""

import ast
import os

_PKG = "snn_data_lake"
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../snn_data_lake

# the canonical single-definition stack classes -> the ONE module each MUST be defined in
CANON = {
    "Brain": f"{_PKG}.cortex.brain",
    "LiveBrain": f"{_PKG}.cortex.live_brain",
    "AutonomousArm": f"{_PKG}.brainstem.autonomous_loop",
}

# known DUPLICATE definitions still on disk, pending retirement in a Policy-4 consolidation wave.
# key = (class_name, module_path); value = reason. Empty this out as the replicas are retired.
#
# EMPTY as of v4.55 -- there is now literally ONE Brain class. The last replica
# (reasoning/brain_loop.Brain) was retired once every capability was verified canonical: deduction ->
# reasoning/derivation.py (a verbatim port of its _relation_graph/_derive), abduction ->
# pc_abduction.AbductionExpert via LakeChat._abduce_reply, induction -> schema_induction (same
# refuse-to-generalise-on-thin-evidence guard), teach/think/boot/repl -> LakeChat, develop ->
# AutonomousArm.develop + developmental_core, start_learning -> the one learn_daemon, and
# self_rewrite -> Brain.self_improve, which is STRICTLY better gated (crc32 byte-identity firewall
# over the 9 safety/verifier files, NORD_SELFMOD=1 off-switch, and it applies NOTHING -- it returns a
# proposal, where self_rewrite committed on utility gain). A new entry here means a replica shipped.
PENDING_RETIREMENT = {}


def _mpath(pyfile):
    rel = os.path.relpath(pyfile, os.path.dirname(_PKG_DIR)).replace("\\", "/")[:-3]
    return rel.replace("/", ".")


def _class_sites():
    """All definition sites of the canonical stack class NAMES: {name: [module_path, ...]}."""
    sites = {name: [] for name in CANON}
    for root, _d, files in os.walk(_PKG_DIR):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            mod = _mpath(p)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in CANON:
                    sites[node.name].append(mod)
    return sites


def audit():
    sites = _class_sites()
    replicas, missing = [], []
    for name, canon_mod in CANON.items():
        found = sites[name]
        if canon_mod not in found:
            missing.append((name, canon_mod))
        for mod in found:
            if mod == canon_mod:
                continue
            if (name, mod) not in PENDING_RETIREMENT:
                replicas.append((name, mod))  # an UNSANCTIONED second definition
    return {"sites": sites, "replicas": replicas, "missing": missing}


def main():
    a = audit()
    print("=" * 66)
    print("REPLICATION GUARD -- Policy 4: there is only ONE brain (no re-implementation)")
    print("=" * 66)
    for name, canon in CANON.items():
        dups = [m for m in a["sites"][name] if m != canon]
        tag = f"  DUPLICATES: {dups}" if dups else ""
        print(f"  {name:<14} defined in {len(a['sites'][name])} site(s){tag}")
    if PENDING_RETIREMENT:
        print("--- allowlisted replicas pending retirement ---")
        for (name, mod), reason in PENDING_RETIREMENT.items():
            print(f"  {name} @ {mod.replace(_PKG + '.', '')}: {reason[:60]}...")
    print("=" * 66)
    assert not a["missing"], f"canonical stack class not found where expected: {a['missing']}"
    assert not a["replicas"], (
        f"Policy 4: unsanctioned brain-stack replica(s) -- a second definition of a canonical class "
        f"that is not on PENDING_RETIREMENT: {a['replicas']}. Utilize the one brain (cortex.brain."
        f"make_brain / the canonical nesting), do not re-implement it."
    )
    _pend = f" (+ {len(PENDING_RETIREMENT)} allowlisted replica(s))" if PENDING_RETIREMENT else ""
    print(f"replication guard OK: one canonical Brain/LiveBrain/AutonomousArm{_pend}")


if __name__ == "__main__":
    main()
