"""
SELF-BUILD LOOP -- the gated APPLY/PROMOTE step that lets the brain build itself.

reasoning/self_mod_harness.SelfModHarness sandbox-verifies a candidate self-modification (firewall the
arm-safety files + isolate into a temp repo copy + run run_selfchecks in the copy) but APPLIES NOTHING
to the real tree. This module adds the one missing step -- promote: for a proposal the harness ACCEPTED,
apply the SAME whitelisted single-numeric edit to the REAL tree, run the FULL correctness gates, and KEEP
it if green or ROLL BACK byte-exact if red.

The 5 brainstem arm-safety files stay in the protected set (reasoning/self_mod_harness._PROTECTED, which
includes godel_machine._SAFETY_KERNEL): a protected file is NEVER a legal promote target (refused up
front), and the whole protected set is re-verified crc-unchanged after every apply. Self-modification
never touches the hardware-safety envelope -- it stays OUTSIDE that sandbox. That is the carve-out, not a
growth ceiling.

Unbounded cognition, correctness-gated: a self-mod that breaks the gates rolls back -- that is
CORRECTNESS, not a capability ceiling. DISABLED by default -- promote() is a no-op unless enable=True AND
env NORD_SELFMOD=1 (mirrors SelfModHarness). gate_fn is injectable: pass the REAL
run_selfchecks + pytest + E-STOP for a live cycle, or a fast deterministic fake for the self-check, so
the promote / rollback / firewall logic is fully testable without a multi-minute gate.

Reuses reasoning/self_mod_harness (sandbox pre-verify + the _PROTECTED set) and
reasoning/improvement_proposer (real-metric proposals: {target, param, proposed, edit_fn, ...}).
ASCII-only prints; deterministic; no wall-clock in any returned value.
"""

import os
import zlib

from snn_data_lake.reasoning.self_mod_harness import _PROTECTED, SelfModHarness, _repo_root


def _crc_file(path):
    try:
        with open(path, "rb") as f:
            return zlib.crc32(f.read())
    except OSError:
        return None


def _protected_crcs(root):
    """crc32 of every protected (arm-safety/verifier) file -- must be IDENTICAL before/after a promote."""
    return {rel: _crc_file(os.path.join(root, rel)) for rel in _PROTECTED}


class SelfBuildLoop:
    """The gated promote step: apply a sandbox-ACCEPTED self-mod to the real tree, gate, keep or roll back.

    DISABLED by default. The arm-safety protected set is never a legal target and is re-verified
    crc-unchanged after every apply. The injectable gate_fn (a bool-returning full gate) is the backstop;
    a failing (or erroring) gate rolls the change back byte-exact."""

    def __init__(self, repo_root=None, enable=False, gate_fn=None, harness=None):
        self.root = os.path.abspath(repo_root) if repo_root else _repo_root()
        self.enable = bool(enable)
        self.gate_fn = (
            gate_fn  # () -> bool : the real full gate live, a fast fake in the self-check
        )
        self._harness = (
            harness  # lazily built (SelfModHarness) only when run() needs the sandbox pre-verify
        )
        self.log = []

    def _is_enabled(self):
        return self.enable and os.environ.get("NORD_SELFMOD") == "1"

    @property
    def harness(self):
        if self._harness is None:
            self._harness = SelfModHarness(enable=self.enable)
        return self._harness

    def promote(self, edit_fn, target_rel):
        """Apply an accepted sandbox-verified edit to the REAL tree, gate, KEEP or ROLL BACK.

        Returns {status, rel, ...}: 'disabled' | 'refused' | 'kept' | 'rolled_back'. edit_fn(root) is the
        proposer's single-numeric-assignment rewrite (re-checks the firewall inside its own closure);
        target_rel is the file it edits."""
        rel = str(target_rel).replace("\\", "/")
        if not self._is_enabled():
            return {"status": "disabled", "rel": rel}
        # (1) ARM-SAFETY FIREWALL: a protected file is NEVER a legal promote target.
        if rel in _PROTECTED:
            self.log.append((rel, "refused", "firewall: protected file"))
            return {"status": "refused", "rel": rel, "reason": "protected file"}
        abspath = os.path.join(self.root, rel)
        if not os.path.isfile(abspath):
            return {"status": "refused", "rel": rel, "reason": "target missing"}
        before = _protected_crcs(self.root)
        with open(abspath, "rb") as f:
            snapshot = f.read()  # byte-exact rollback snapshot of the real file
        # (2) APPLY to the real tree (edit_fn re-checks the firewall inside its closure).
        try:
            edit_fn(self.root)
        except Exception as e:  # noqa: BLE001 -- any apply failure -> restore + refuse
            with open(abspath, "wb") as f:
                f.write(snapshot)
            return {"status": "refused", "rel": rel, "reason": f"apply: {type(e).__name__}: {e}"}
        # (2a) the protected arm-safety set MUST be byte-identical after the apply -> else revert.
        if _protected_crcs(self.root) != before:
            with open(abspath, "wb") as f:
                f.write(snapshot)
            self.log.append((rel, "refused", "protected set changed"))
            return {"status": "refused", "rel": rel, "reason": "protected set changed -> reverted"}
        # (3) CORRECTNESS GATE on the real tree. Green -> keep. Red/error -> roll back byte-exact.
        try:
            ok = bool(self.gate_fn()) if self.gate_fn is not None else False
        except Exception as e:  # noqa: BLE001 -- a gate that errors is a FAIL
            ok = False
            self.log.append((rel, "gate_error", f"{type(e).__name__}: {e}"))
        if ok:
            self.log.append((rel, "kept", "gates green"))
            return {"status": "kept", "rel": rel}
        with open(abspath, "wb") as f:
            f.write(snapshot)  # ROLLBACK
        self.log.append((rel, "rolled_back", "gates red"))
        return {"status": "rolled_back", "rel": rel}

    def run(self, proposals, utility_fn=None):
        """propose -> sandbox-verify (harness) -> promote each ACCEPTED one (gated). Returns outcomes."""
        outcomes = []
        for p in proposals:
            rel = str(p.get("target") or "").replace("\\", "/")
            edit_fn = p.get("edit_fn")
            summary = {"param": p.get("param"), "rel": rel, "proposed": p.get("proposed")}
            if not rel or edit_fn is None:
                outcomes.append({**summary, "status": "skipped", "reason": "no target/edit_fn"})
                continue
            sb = self.harness.propose(rel, edit_fn, utility_fn)
            if sb.get("status") != "accepted":
                outcomes.append({**summary, "status": f"sandbox_{sb.get('status')}"})
                continue
            outcomes.append({**summary, "sandbox": "accepted", **self.promote(edit_fn, rel)})
        return outcomes


# =====================================================================================================
# Self-check: exercises promote()'s keep / rollback / firewall / disabled logic on a TEMP repo with a
# FAST FAKE gate -- no real module is ever modified, no multi-minute gate is run.
# =====================================================================================================
def _toy_edit_fn(name, new_value):
    """A minimal single-numeric-assignment rewrite closure, mirroring improvement_proposer's edit_fn:
    rewrites the one line `name = <number>` in <root>/toy_mod.py."""

    def edit_fn(root):
        path = os.path.join(root, "toy_mod.py")
        with open(path, encoding="ascii") as f:
            src = f.read()
        import re

        pat = re.compile(rf"^({re.escape(name)}\s*=\s*)([-+0-9.eE]+)(.*)$", re.M)
        new_src, n = pat.subn(rf"\g<1>{new_value}\g<3>", src)
        if n != 1:
            raise RuntimeError(f"'{name} = <number>' not unique in toy_mod.py (n={n})")
        with open(path, "w", encoding="ascii") as f:
            f.write(new_src)

    return edit_fn


def main():
    import tempfile

    print("=" * 88)
    print("SELF-BUILD LOOP -- gated promote (keep / rollback / firewall / disabled) self-check")
    print("=" * 88)
    os.environ["NORD_SELFMOD"] = "1"  # arm the loop for the test (still needs enable=True)

    with tempfile.TemporaryDirectory() as tmp:
        toy = os.path.join(tmp, "toy_mod.py")
        original = "X = 1.0  # a whitelisted cognitive parameter\n"
        with open(toy, "w", encoding="ascii") as f:
            f.write(original)

        # (a) GREEN gate -> KEEP: the real toy file ends up with the new value.
        loop_ok = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=lambda: True)
        r_keep = loop_ok.promote(_toy_edit_fn("X", "2.0"), "toy_mod.py")
        with open(toy, encoding="ascii") as f:
            after_keep = f.read()
        print(f"(a) green gate -> {r_keep['status']}; toy now: {after_keep.strip()!r}")
        assert r_keep["status"] == "kept", r_keep
        assert "X = 2.0" in after_keep, "a KEPT promote must leave the new value in the real file"

        # (b) RED gate -> ROLLBACK byte-exact: reset the toy, promote under a failing gate.
        with open(toy, "w", encoding="ascii") as f:
            f.write(original)
        with open(toy, "rb") as f:
            before_crc = zlib.crc32(
                f.read()
            )  # actual file bytes (text-mode write may add CRLF on win)
        loop_bad = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=lambda: False)
        r_roll = loop_bad.promote(_toy_edit_fn("X", "9.0"), "toy_mod.py")
        with open(toy, "rb") as f:
            after_crc = zlib.crc32(f.read())
        print(f"(b) red gate -> {r_roll['status']}; file byte-identical: {after_crc == before_crc}")
        assert r_roll["status"] == "rolled_back", r_roll
        assert after_crc == before_crc, "a failed gate MUST restore the real file byte-exact"

        # (c) ARM-SAFETY FIREWALL: a protected brainstem file is refused up front (no write attempted).
        protected_rel = next(iter(_PROTECTED))
        r_fw = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=lambda: True).promote(
            _toy_edit_fn("X", "3.0"), protected_rel
        )
        print(f"(c) target protected {protected_rel!r} -> {r_fw['status']} ({r_fw.get('reason')})")
        assert r_fw["status"] == "refused" and "protected" in r_fw.get("reason", ""), r_fw

        # (d) DISABLED by default: without NORD_SELFMOD promote is a no-op.
        os.environ.pop("NORD_SELFMOD", None)
        r_dis = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=lambda: True).promote(
            _toy_edit_fn("X", "4.0"), "toy_mod.py"
        )
        print(f"(d) NORD_SELFMOD unset -> {r_dis['status']}")
        assert r_dis["status"] == "disabled", r_dis

    print("-" * 88)
    print(
        "self_build_loop OK: promote KEEPS on a green gate, ROLLS BACK byte-exact on a red gate, REFUSES "
        "a protected arm-safety target up front, and is DISABLED unless enable=True + NORD_SELFMOD=1. The "
        "brain can build itself, correctness-gated, with the hardware-safety envelope firewalled."
    )


if __name__ == "__main__":
    main()
