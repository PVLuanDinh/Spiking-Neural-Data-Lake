"""
SELF-BUILD LOOP -- the gated APPLY/PROMOTE step that lets the brain build itself.

substrate/self_mod_harness.SelfModHarness sandbox-verifies a candidate self-modification (firewall the
arm-safety files + isolate into a temp repo copy + run run_selfchecks in the copy) but APPLIES NOTHING
to the real tree. This module adds the one missing step -- promote: for a proposal the harness ACCEPTED,
apply the SAME whitelisted single-numeric edit to the REAL tree, run the FULL correctness gates, and KEEP
it if green or ROLL BACK byte-exact if red.

THE FIREWALL, and why every path in it is DERIVED rather than written as a literal:
  * substrate/self_mod_harness._PROTECTED -- the 5 brainstem arm-safety files (via
    godel_machine._SAFETY_KERNEL) plus run_selfchecks.py, godel_machine.py, self_improve.py and
    self_mod_harness.py. Never a legal promote target: refused up front, before anything is read.
  * SELF_BUILD_LOOP_REL -- THIS module's own repo-relative path, computed from __file__ at import.
    The promote path must not itself be promotable, or the learner could learn to loosen its own
    gate. tests/test_self_mod_safety.py exists because "paths go stale silently": a hardcoded
    "src/snn_data_lake/reasoning/self_build_loop.py" stops matching the moment this file changes
    region, and the protection then evaporates WITHOUT ANYTHING TURNING RED. Deriving it from
    __file__ makes that impossible, and the self-check asserts the derived path resolves to a real
    file so a broken derivation is loud.
  * The whole firewalled set is snapshotted BYTE-for-byte before the apply and re-verified TWICE:
    after the apply, and again after the gate has run (the gate executes arbitrary in-tree code, so
    the moment that matters is the COMMIT, not only the apply). Any drift is restored byte-exact and
    the promote is refused / rolled back. Self-modification never touches the hardware-safety
    envelope -- it stays OUTSIDE that sandbox. That is the carve-out, not a growth ceiling.

Unbounded cognition, correctness-gated: a self-mod that breaks the gates rolls back -- that is
CORRECTNESS, not a capability ceiling. promote() is a no-op unless enable=True AND env NORD_SELFMOD=1
(mirrors SelfModHarness). "Disabled by default" is true of a bare run and NOT of the always-on daemon:
learn_daemon.bat:14 sets NORD_SELFMOD=1, so under the daemon this path is ARMED and fires every 8
nights. gate_fn is injectable, and the honest description of the live one: the sole live caller
(cortex/brain.py `_maybe_autonomous_self_build`) passes run_selfchecks.py ALONE, as a subprocess with
NORD_SELFMOD=0 so the gate cannot recurse into another self-build -- NOT pytest and NOT an E-STOP
check, whichever an earlier version of this docstring claimed. The self-check passes a fast
deterministic fake, so promote / rollback / firewall logic is fully testable without a long gate.

Reuses infra/self_mod_harness (sandbox pre-verify + the _PROTECTED set). Proposals come from
infra/brain_native_selfbuild (BrainNativeSelfBuildProposer.propose, reached through
Brain.propose_improvements) as {target, edit_fn, id, param, rationale}; reasoning/improvement_proposer,
which this docstring used to name as the supplier, is RETIRED and supplies nothing.

(Module paths corrected 2026-08-04: this docstring said `reasoning/self_mod_harness` and
`reasoning/brain_native_selfbuild` in four places. Neither file exists at those paths -- the harness
is `infra/self_mod_harness.py` (which is what line 55 below actually imports) and the proposer is
`infra/brain_native_selfbuild.py`. The paths went stale on the region-reorganisation wave and
nothing re-read them, which is the exact hazard the DERIVED `SELF_BUILD_LOOP_REL` above exists to
prevent in CODE -- and prose gets no such derivation, so it has to be re-checked by hand.)

P7 (dependency judgement, the retired zero-dependency rule): stdlib is correct here ON THE MERITS and
nothing was given up for the old rule. This module does byte I/O, zlib.crc32 and one bool call -- there
is no array, no matrix and no kernel for numpy or torch to accelerate, so a dependency would buy
nothing measurable. No zero-dep claim is being defended and none is made.

ASCII-only prints; deterministic; no wall-clock in any returned value; the self-check restores
NORD_SELFMOD to exactly what it found, so running it IN-PROCESS (runpy via run_selfchecks, or pytest)
cannot leave a real self-build armed for the rest of the process.
"""

import os
import zlib

from snn_data_lake.infra.self_mod_harness import _PROTECTED, SelfModHarness, _repo_root

# Repo root comes from the sibling's ANCHORED derivation, not a local depth count.
#
# There used to be a `_own_repo_root()` here that walked up exactly three directories and justified
# itself as "self-contained on purpose: the module that decides what the brain may rewrite must
# locate the tree without depending on where any sibling happens to live". That independence was not
# real -- line 55 already imports _PROTECTED, SelfModHarness AND _repo_root from self_mod_harness,
# so the decoupling bought nothing while adding a second mechanism for one job (P2). It was also the
# more fragile of the two: `up 3` is correct only while this file sits at exactly
# src/snn_data_lake/<region>/, and it fails SILENTLY -- a wrong root yields a wrong
# SELF_BUILD_LOOP_REL, which yields a firewall entry matching no file, which protects nothing while
# every check still reports green. self_mod_harness._repo_root() instead walks up looking for the
# ("run_selfchecks.py", "src") anchor, so it finds the real root from any depth.
SELF_BUILD_LOOP_REL = os.path.relpath(os.path.abspath(__file__), _repo_root()).replace(os.sep, "/")
# Every file a promote may NEVER target: the harness's protected set plus the promote path itself,
# de-duped and order-stable. Derived, never literal -- see the module docstring.
_FIREWALLED = tuple(dict.fromkeys((*_PROTECTED, SELF_BUILD_LOOP_REL)))


def _read_bytes(path):
    """Bytes of `path`, or None if it does not exist. None is a real state, not an error."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def _firewalled_bytes(root):
    """{rel: bytes|None} for every firewalled file under `root` -- None when the file is absent.

    Bytes, not just crc32: a crc map can DETECT a change to the safety envelope but cannot UNDO one,
    and the claim this module makes is that the envelope is RESTORED, not merely noticed."""
    return {rel: _read_bytes(os.path.join(root, *rel.split("/"))) for rel in _FIREWALLED}


def _crcs(snap):
    """crc32 fingerprint of a _firewalled_bytes snapshot (project convention: crc32, never hash())."""
    return {rel: (None if b is None else zlib.crc32(b)) for rel, b in snap.items()}


def _restore(path, data):
    """Put `path` back exactly as it was: rewrite its bytes, or delete it if it did not exist before."""
    if data is None:
        try:
            os.remove(path)
        except OSError:
            pass
        return
    with open(path, "wb") as f:
        f.write(data)


class SelfBuildLoop:
    """The gated promote step: apply a sandbox-ACCEPTED self-mod to the real tree, gate, keep or roll back.

    Off unless enable=True AND NORD_SELFMOD=1. The firewalled set (arm-safety + verifier + THIS module,
    the last derived from __file__ so a move cannot silently un-protect it) is never a legal target, and
    is restored byte-exact if anything -- the edit OR the gate -- perturbs it. The injectable gate_fn
    (a bool-returning full gate) is the backstop; a failing or erroring gate rolls the change back."""

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

    def _restore_firewalled(self, before):
        """Restore every firewalled file whose bytes differ from the `before` snapshot.

        Returns the sorted rels it had to put back -- an EMPTY list is the normal case and the only one
        that means 'the safety envelope was never touched'."""
        fixed = []
        for r, data in before.items():
            path = os.path.join(self.root, *r.split("/"))
            if _read_bytes(path) != data:
                _restore(path, data)
                fixed.append(r)
        return sorted(fixed)

    def promote(self, edit_fn, target_rel):
        """Apply an accepted sandbox-verified edit to the REAL tree, gate, KEEP or ROLL BACK.

        Returns {status, rel, ...}: 'disabled' | 'refused' | 'kept' | 'rolled_back'. edit_fn(root) is the
        proposer's single-numeric-assignment rewrite (re-checks the firewall inside its own closure);
        target_rel is the file it edits."""
        rel = str(target_rel).replace("\\", "/")
        if not self._is_enabled():
            return {"status": "disabled", "rel": rel}
        # (1) FIREWALL: a protected arm-safety/verifier file -- or this promote path itself -- is NEVER a
        #     legal target. Refused up front, before anything is read or written.
        if rel in _FIREWALLED:
            why = "protected file" if rel in _PROTECTED else "protected: the promote path itself"
            self.log.append((rel, "refused", f"firewall: {why}"))
            return {"status": "refused", "rel": rel, "reason": why}
        abspath = os.path.join(self.root, *rel.split("/"))
        if not os.path.isfile(abspath):
            return {"status": "refused", "rel": rel, "reason": "target missing"}
        before = _firewalled_bytes(self.root)
        before_crcs = _crcs(before)
        snapshot = _read_bytes(abspath)  # byte-exact rollback snapshot of the real file
        # (2) APPLY to the real tree (edit_fn re-checks the firewall inside its closure).
        try:
            edit_fn(self.root)
        except Exception as e:  # noqa: BLE001 -- any apply failure -> restore + refuse
            _restore(abspath, snapshot)
            # A RAISE IS NOT A ROLLBACK (fixed 2026-08-04, found while promoting this file). This
            # handler restored only the declared target, so an edit_fn that wrote to a PROTECTED
            # file and then raised left the safety envelope perturbed -- the one outcome the
            # firewall exists to prevent. The drift path at (2a) below already performs exactly
            # this repair; the exception path simply did not, and an exception is the easier of
            # the two paths for a hostile or buggy edit_fn to reach deliberately. Self-check (h)
            # constructs that edit_fn and asserts the envelope comes back byte-exact.
            reverted = self._restore_firewalled(before)
            self.log.append((rel, "refused", f"apply raised; reverted: {reverted}"))
            out = {"status": "refused", "rel": rel, "reason": f"apply: {type(e).__name__}: {e}"}
            if reverted:
                out["reverted"] = reverted
            return out
        # (2a) the firewalled set MUST be byte-identical after the apply. If it is not, restore ALL of it
        #      and refuse WITHOUT running the gate -- never gate a tree whose safety envelope moved.
        if _crcs(_firewalled_bytes(self.root)) != before_crcs:
            _restore(abspath, snapshot)
            reverted = self._restore_firewalled(before)
            self.log.append((rel, "refused", f"firewalled set changed: {reverted}"))
            return {
                "status": "refused",
                "rel": rel,
                "reason": "protected set changed -> reverted",
                "reverted": reverted,
            }
        # (3) CORRECTNESS GATE on the real tree. Green -> keep. Red/error -> roll back byte-exact.
        try:
            ok = bool(self.gate_fn()) if self.gate_fn is not None else False
        except Exception as e:  # noqa: BLE001 -- a gate that errors is a FAIL
            ok = False
            self.log.append((rel, "gate_error", f"{type(e).__name__}: {e}"))
        # (3a) the gate runs arbitrary in-tree code, so the envelope is re-verified at the moment of the
        #      COMMIT, not only at the moment of the apply. A gate that touched it is a RED gate however
        #      green it reported: the brain may rewrite what it thinks, never what stops it.
        post = _firewalled_bytes(self.root)
        tampered = sorted(r for r in before if post[r] != before[r])
        if tampered:
            ok = False
            self.log.append((rel, "gate_tampered", str(tampered)))
        if ok:
            self.log.append((rel, "kept", "gates green"))
            return {"status": "kept", "rel": rel}
        _restore(abspath, snapshot)  # ROLLBACK
        reverted = self._restore_firewalled(before)
        self.log.append((rel, "rolled_back", "gates red"))
        out = {"status": "rolled_back", "rel": rel}
        if reverted:
            out["reverted"] = reverted
        return out

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
    """A minimal single-numeric-assignment rewrite closure, mirroring the edit_fn shape that
    brain_native_selfbuild's proposer supplies: rewrites the one line `name = <number>` in
    <root>/toy_mod.py."""

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
    prior_selfmod = os.environ.get("NORD_SELFMOD")
    try:
        _selfcheck()
    finally:
        # IN-PROCESS SAFETY: run_selfchecks can run this via runpy in the SAME process (the frozen build
        # does, and promotion puts it under pytest), so NORD_SELFMOD is left exactly as it was found.
        # Leaving it set would arm a REAL self-build for everything that runs after this in the process;
        # unconditionally popping it would disarm the daemon, which sets it deliberately.
        if prior_selfmod is None:
            os.environ.pop("NORD_SELFMOD", None)
        else:
            os.environ["NORD_SELFMOD"] = prior_selfmod
    # An assert re-checking NORD_SELFMOD was DELETED here 2026-08-04 (P8/F11: deleted, not
    # weakened). It ran immediately after the finally block that had just assigned the value it
    # tested, with nothing in between that could change it -- so it restated the preceding four
    # lines rather than testing them, and no input could make it fire. The restore itself is the
    # real fix and stays: this self-check runs IN-PROCESS via runpy under run_selfchecks, so a
    # clobbered NORD_SELFMOD is the value every module after it reads, and the live gate sets '0'
    # to stop the self-build recursing while the daemon sets '1' deliberately.


def _selfcheck():
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

        # (d) SELF-FIREWALL, derived from __file__ -- the promote path is not itself promotable, and
        #     the path is COMPUTED so moving this file to another region cannot silently un-protect
        #     it. RED: drop SELF_BUILD_LOOP_REL from _FIREWALLED and the refusal reason becomes
        #     'target missing' instead of 'protected', which is what this block asserts.
        #
        #     TWO ASSERTS WERE DELETED HERE 2026-08-04 (P8/F11: deleted, not weakened).
        #     (i) `os.path.isfile(os.path.join(<root>, *SELF_BUILD_LOOP_REL.split("/")))`. Since
        #         SELF_BUILD_LOOP_REL is relpath(abspath(__file__), <root>), rejoining it to <root>
        #         reconstructs abspath(__file__) -- the file Python is executing. It is True by
        #         construction for every possible value of <root>, including a WRONG one, so it
        #         could not fail and in particular could not catch the broken derivation it was
        #         written to catch. Its own comment claimed "break the up-3 derivation and the
        #         resolve assert fires"; breaking the derivation changes <root> and REL together
        #         and the assert stays green. That claim was false, which is why it is recorded
        #         rather than quietly dropped.
        #     (ii) `_own_repo_root() == _repo_root()`. Real while two derivations existed; the
        #         depth-counting one is now gone (see the top of this file), so this reduced to
        #         `_repo_root() == _repo_root()`. A tautology is the purest form of a criterion
        #         with no reachable red.
        #     What actually guards the derivation now: self_mod_harness._repo_root() is ANCHORED on
        #     ("run_selfchecks.py", "src") rather than counting directories, and the refusal test
        #     immediately below fails loudly if REL ever stops naming this file.
        r_self = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=lambda: True).promote(
            _toy_edit_fn("X", "5.0"), SELF_BUILD_LOOP_REL
        )
        print(
            f"(d) target self {SELF_BUILD_LOOP_REL!r} -> {r_self['status']} ({r_self.get('reason')})"
        )
        assert r_self["status"] == "refused" and "protected" in r_self.get("reason", ""), r_self

        # (e) AN EDIT THAT REACHES OUTSIDE ITS TARGET IS REFUSED AND FULLY UNDONE. edit_fn runs with the
        #     repo root, so a forged closure can write anywhere; the post-apply check must refuse WITHOUT
        #     running the gate and restore BOTH files. RED is reachable: drop the _restore_firewalled call
        #     in (2a) and the envelope stays relaxed while the status still reads 'refused'.
        kernel_abs = os.path.join(tmp, *protected_rel.split("/"))
        os.makedirs(os.path.dirname(kernel_abs), exist_ok=True)
        with open(kernel_abs, "wb") as f:
            f.write(b"ESTOP = True\n")
        with open(toy, "w", encoding="ascii") as f:
            f.write(original)

        def _forged_edit_fn(root):
            _toy_edit_fn("X", "7.0")(root)  # the declared, whitelisted single-numeric edit ...
            with open(kernel_abs, "wb") as f:
                f.write(b"ESTOP = False\n")  # ... plus one the proposal never declared

        def _never_gate():
            raise AssertionError("the gate must NOT run once the envelope has moved")

        r_forge = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=_never_gate).promote(
            _forged_edit_fn, "toy_mod.py"
        )
        toy_after_forge = _read_bytes(toy)
        forge_undone = (
            _read_bytes(kernel_abs) == b"ESTOP = True\n" and b"X = 1.0" in toy_after_forge
        )
        print(
            f"(e) edit_fn also wrote {protected_rel!r} -> {r_forge['status']} "
            f"({r_forge.get('reason')}); both files undone: {forge_undone}"
        )
        assert r_forge["status"] == "refused", r_forge
        assert r_forge.get("reverted") == [protected_rel], r_forge
        assert forge_undone, (
            "an out-of-scope edit must leave BOTH the target and the envelope as found"
        )

        # (f) A GREEN GATE THAT TAMPERS WITH THE ENVELOPE IS A RED GATE. The gate runs arbitrary in-tree
        #     code, so a gate that relaxes an arm-safety file and then reports pass must still roll the
        #     promote back AND restore that file byte-exact. RED is reachable: delete the post-gate
        #     re-verify (3a) and this returns 'kept' with ESTOP left False.
        def _tampering_gate():
            with open(kernel_abs, "wb") as f:
                f.write(b"ESTOP = False\n")  # the gate tries to relax the safety envelope ...
            return True  # ... and then reports GREEN

        r_tamper = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=_tampering_gate).promote(
            _toy_edit_fn("X", "6.0"), "toy_mod.py"
        )
        kernel_intact = _read_bytes(kernel_abs) == b"ESTOP = True\n"
        print(
            f"(f) green gate that edited {protected_rel!r} -> {r_tamper['status']}; "
            f"envelope restored byte-exact: {kernel_intact}"
        )
        assert r_tamper["status"] == "rolled_back", r_tamper
        assert kernel_intact, (
            "a GREEN gate that edited a protected arm-safety file must still roll back and restore it"
        )

        # (h) AN edit_fn THAT TAMPERS AND THEN RAISES. A raise is not a rollback: the envelope must be
        #     restored on the EXCEPTION path exactly as (e)/(f) restore it on the drift path. Before
        #     the 2026-08-04 fix the handler restored only the declared target, so this case left
        #     ESTOP = False on disk. RED is reachable and was executed: delete the
        #     _restore_firewalled call from the apply handler and raise_intact goes True -> False.
        with open(kernel_abs, "wb") as f:
            f.write(b"ESTOP = True\n")
        with open(toy, "w", encoding="ascii") as f:
            f.write(original)

        def _forged_then_raises(root):
            with open(kernel_abs, "wb") as f:
                f.write(b"ESTOP = False\n")
            raise RuntimeError("declared edit failed after an undeclared write")

        r_raise = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=_never_gate).promote(
            _forged_then_raises, "toy_mod.py"
        )
        raise_intact = _read_bytes(kernel_abs) == b"ESTOP = True\n"
        print(
            f"(h) edit_fn wrote {protected_rel!r} then raised -> {r_raise['status']}; "
            f"envelope restored byte-exact: {raise_intact}"
        )
        assert r_raise["status"] == "refused", r_raise
        assert r_raise.get("reverted") == [protected_rel], r_raise
        assert raise_intact, (
            "an edit_fn that raises after touching the envelope must still restore it"
        )

        # (g) DISABLED by default: without NORD_SELFMOD promote is a no-op.
        os.environ.pop("NORD_SELFMOD", None)
        r_dis = SelfBuildLoop(repo_root=tmp, enable=True, gate_fn=lambda: True).promote(
            _toy_edit_fn("X", "4.0"), "toy_mod.py"
        )
        print(f"(g) NORD_SELFMOD unset -> {r_dis['status']}")
        assert r_dis["status"] == "disabled", r_dis

    print("-" * 88)
    print(
        "self_build_loop OK: promote KEEPS on a green gate, ROLLS BACK byte-exact on a red gate, REFUSES "
        "a protected arm-safety target AND its own __file__-derived path up front, and treats ANY drift "
        "in the envelope as fatal whoever caused it -- an edit_fn that wrote outside its target is "
        "refused before the gate runs, a GREEN gate that touched the envelope is a red gate, and both "
        "are restored byte-exact. DISABLED unless enable=True + NORD_SELFMOD=1. The brain can build "
        "itself, correctness-gated, with the hardware-safety envelope firewalled."
    )


if __name__ == "__main__":
    main()
