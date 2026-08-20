"""Envelope-hash gate (specs/022 T021b, gates.md 'Envelope-hash gate' row): the five
safety-envelope files' BYTES may change only through an owner-approved baseline update.

THE MECHANISM: sha256 over each of the five `brainstem/` envelope files (the path list is
REUSED from `substrate/safety_envelope.ENVELOPE` -- the ONE definition; this module polices
the BYTES where that one polices the DOCUMENT), asserted against the committed baseline
`tools/envelope_hashes.json`. A byte change without a matching baseline update REDS the
build, and updating the baseline is itself the owner-approval act: it lands in the same
commit as the envelope change, in plain diff view, with an `approved` note naming who and
why. There is no bypass flag on purpose -- the cross-cutting invariant is "modified only
with explicit approval", and a gate with an escape hatch polices nothing.

WHAT THIS GATE NEVER DOES: write, format, or otherwise touch an envelope file. It reads
bytes and compares. The envelope is never a target -- of cleanup, of invention (P8), or of
this gate.

Reds, proven in --self-check with the live positive control in the same command: a mutated
expected hash (the tamper construction -- equivalent to an unapproved byte change), a
missing envelope file, and a baseline entry for a path the envelope no longer names (a
stale approval is a claim outliving what it described).
"""

# distinct-from: substrate/safety_envelope.py -- the ONE path list + the SECURITY.md document
#   parity gate; this consumes its ENVELOPE tuple and polices file BYTES. distinct-from:
#   basal_ganglia/self_build_loop.py -- _firewalled_bytes snapshots envelope bytes for
#   RESTORE inside the self-mod sandbox; this ASSERTS them at CI. distinct-from:
#   substrate/self_mod_harness.py -- the sandbox those snapshots serve. All sites read.

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from snn_data_lake.substrate.safety_envelope import ENVELOPE, repo_relative

REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "tools" / "envelope_hashes.json"


def current_hashes() -> dict:
    """Keyed by the PACKAGE-relative spelling (safety_envelope.ENVELOPE's own), resolved on
    disk via repo_relative() -- the ONE list's own repo-root spelling, so this gate can never
    invent a third path convention (the anchored-normalize hazard that module records)."""
    out = {}
    for rel, repo_rel in zip(ENVELOPE, repo_relative()):
        p = REPO / repo_rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
    return out


def check(baseline: dict | None = None) -> tuple[bool, list[str]]:
    """-> (ok, findings). Pure compare; the caller decides loudness."""
    if baseline is None:
        if not BASELINE.exists():
            return False, [f"no baseline at {BASELINE} -- record one (an owner-visible act)"]
        baseline = json.loads(BASELINE.read_text())
    entries = baseline.get("files", {})
    findings = []
    now = current_hashes()
    for rel, digest in now.items():
        want = entries.get(rel, {}).get("sha256")
        if want is None:
            findings.append(f"UNBASELINED envelope file: {rel} (no approved hash on record)")
        elif digest == "MISSING":
            findings.append(f"envelope file MISSING on disk: {rel}")
        elif digest != want:
            findings.append(
                f"UNAPPROVED envelope byte change: {rel} (sha256 {digest[:16]}... != approved "
                f"{want[:16]}...) -- an envelope edit lands WITH its baseline update and an "
                "approval note, in the same commit"
            )
    for rel in entries:
        if rel not in now:
            findings.append(
                f"STALE baseline entry: {rel} is approved but no longer in the envelope list "
                "-- a stale approval is a claim outliving what it described"
            )
    return not findings, findings


def record(note: str) -> dict:
    """Write the baseline from the CURRENT bytes with an approval note. The owner-visible
    act; refuses an empty note."""
    if not note or not note.strip():
        raise ValueError("an approval baseline without a note is not an approval")
    payload = {
        "note": (
            "The envelope-hash baseline: updating this file IS the owner-approval act for an "
            "envelope byte change, reviewable in the same commit's diff."
        ),
        "files": {rel: {"sha256": h, "approved": note} for rel, h in current_hashes().items()},
    }
    BASELINE.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def self_check() -> int:
    ok, findings = check()
    if not ok:
        print(f"envelope_hash_gate self-check FAILED its live positive control: {findings}")
        return 1
    baseline = json.loads(BASELINE.read_text())
    # tamper red: one expected hash mutated == an unapproved byte change
    tampered = json.loads(json.dumps(baseline))
    first = next(iter(tampered["files"]))
    tampered["files"][first]["sha256"] = "0" * 64
    ok_t, f_t = check(tampered)
    if ok_t or not any("UNAPPROVED" in f for f in f_t):
        print("envelope_hash_gate self-check FAILED: the tamper red did not fire")
        return 1
    # stale-approval red: a baselined path the envelope no longer names
    stale = json.loads(json.dumps(baseline))
    stale["files"]["src/snn_data_lake/brainstem/ghost.py"] = {"sha256": "0" * 64, "approved": "x"}
    ok_s, f_s = check(stale)
    if ok_s or not any("STALE" in f for f in f_s):
        print("envelope_hash_gate self-check FAILED: the stale-approval red did not fire")
        return 1
    # unbaselined red: an envelope file with no approved hash
    missing = json.loads(json.dumps(baseline))
    dropped = next(iter(missing["files"]))
    del missing["files"][dropped]
    ok_m, f_m = check(missing)
    if ok_m or not any("UNBASELINED" in f for f in f_m):
        print("envelope_hash_gate self-check FAILED: the unbaselined red did not fire")
        return 1
    print(
        "envelope_hash_gate self-check OK: live five-file positive control green; tamper "
        "(unapproved change), stale-approval and unbaselined reds all fired. The gate reads "
        "bytes and compares; it never touches the envelope."
    )
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    if mode == "--record":
        note = " ".join(argv[1:]).strip()
        payload = record(note)
        print(json.dumps(payload, indent=2))
        return 0
    if mode == "--self-check":
        return self_check()
    if mode == "--assert":
        ok, findings = check()
        if not ok:
            for f in findings:
                print(f"  ENVELOPE GATE RED: {f}")
            return 1
        print(
            f"envelope_hash_gate [assert] OK: all {len(ENVELOPE)} envelope files match their approved hashes"
        )
        return 0
    # bare (run_selfchecks): the full self-check -- cheap (five sha256s) and never red-free
    return self_check()


if __name__ == "__main__":
    sys.exit(main())
