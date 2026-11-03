r"""D-R0 Part A driver (pre-registration specs/022-unified-brain-arc/evidence/d-r0-representational-drift-
instrument.md 2.3, 2.4): per mouse, E1 = D = mean over weeks {2,3,4,5} present of the mean over neurons present
that week of (RDI_MOV - RDI_PDG); then the paired contrast over mice.

WRITTEN IN ROUND 1, NEVER RUN ON DATA. Two things are deliberately not here yet:
  * `read_field` (the deposit's file layout and variable names -- RespMat_Full, RespMat_onTime, the RoiINFO flags
    -- were not read in round 1: no mouse data). It raises until the launcher supplies it, as a logged revision,
    through `ophys_ingest.read_response` (the only reader allowed to touch response bytes).
  * any DEV agreement step (`author_agreement_verdict` real half): a separate launcher step.

HELD-OUT is refused unless ALL hold: the evidence file carries a dated amendment line
    FROZEN-SHA <YYYY-MM-DD> <40 hex>
and HEAD equals that sha; it carries the dated lines
    DEV-AGREEMENT MET <YYYY-MM-DD>      (DEV author_agreement_verdict)
    DEV-CONTROL MET <YYYY-MM-DD>        (DEV mov_vs_mov_control_verdict)
the dev log holds no HELD-OUT file read yet (read ONCE); both module self-checks re-run green HERE, at that sha (the driver runs them itself); and
`tools/d_r0_dev_log.jsonl` is tracked and clean in git at that sha. DEV needs none of these.
The driver refuses to start while `tools/d_r0_partA.json` is dirty: COMMIT THE RESULT FILE AFTER EACH RUN.

  python tools/d_r0_partA.py --split dev|heldout
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast  # noqa: E402
from snn_data_lake.instruments.representational_drift import (  # noqa: E402
    mov_index,
    stability_index,
)

EVIDENCE = (
    ROOT
    / "specs"
    / "022-unified-brain-arc"
    / "evidence"
    / "d-r0-representational-drift-instrument.md"
)
OUT = ROOT / "tools" / "d_r0_partA.json"
LOG = ROOT / "tools" / "d_r0_dev_log.jsonl"
PIN_PATH = ROOT / "tools" / "marks_goard_pin.json"
DEV = (2, 3, 10, 11)
HELD_OUT = (1, 4, 5, 6, 7, 8, 9, 12)
WEEKS = (2, 3, 4, 5)
FROZEN_RE = re.compile(r"^FROZEN-SHA\s+\d{4}-\d{2}-\d{2}\s+([0-9a-f]{40})\s*$", re.M)

assert sorted(random.Random(20261103).sample(range(1, 13), 4)) == list(DEV), "sealed split drifted"


class HeldOutRefused(RuntimeError):
    pass


class FieldData(NamedTuple):
    """One field, neurons = the session-1 eligible set. `mov1`/`pdg1`: (neurons, trials, frames) in session 1;
    `mov`/`pdg`: {week: same shape}; `present`: {week: bool mask over neurons}."""

    mov1: np.ndarray
    pdg1: np.ndarray
    mov: dict
    pdg: dict
    present: dict


def read_field(mouse):
    raise NotImplementedError(
        "read_field: the deposit layout was not read in round 1; supply it via ophys_ingest.read_response"
    )


def _porcelain(root, path):
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", str(path)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def dev_log_committed(root=ROOT, log=LOG):
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(log)], cwd=str(root), capture_output=True
    )
    return tracked.returncode == 0 and not _porcelain(root, log)


def held_out_read_in_log(log=LOG, pin=None):
    """True when a logged file's sha256 is a held-out one in the pin; a sha absent from the pin counts (fail closed)."""
    if not Path(log).exists():
        return False
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8")) if pin is None else pin
    held = {a["sha256"]: a["split"] == "held-out" for a in pin["assets"]}
    for line in Path(log).read_text(encoding="utf-8").splitlines():
        for f in json.loads(line).get("files", []):
            if held.get(f["sha256"], True):
                return True
    return False


def result_file_dirty(root=ROOT, out=OUT):
    return bool(_porcelain(root, out))


def run_selfchecks():
    """Names of the module self-checks that do not return 0 (both are synthetic, about 10 s each)."""
    from snn_data_lake.instruments import representational_drift
    from snn_data_lake.sensorimotor import ophys_ingest

    checks = {
        "representational_drift": representational_drift._selfcheck,
        "ophys_ingest": ophys_ingest._selfcheck,
    }
    return [name for name, fn in checks.items() if fn() != 0]


def head_sha(root=ROOT):
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True
    )
    return proc.stdout.strip()


def frozen_sha(evidence=EVIDENCE):
    found = FROZEN_RE.findall(Path(evidence).read_text(encoding="utf-8"))
    return found[-1] if found else None


def refuse_held_out_unless_frozen(
    evidence=EVIDENCE,
    head=None,
    selfchecks=run_selfchecks,
    log_committed=dev_log_committed,
    held_out_logged=held_out_read_in_log,
):
    sha = frozen_sha(evidence)
    if sha is None:
        raise HeldOutRefused("HELD-OUT refused: no FROZEN-SHA amendment in the evidence file")
    head = head_sha() if head is None else head
    if head != sha:
        raise HeldOutRefused(f"HELD-OUT refused: HEAD {head[:12]} is not the frozen sha {sha[:12]}")
    text = Path(evidence).read_text(encoding="utf-8")
    for tag in ("DEV-AGREEMENT", "DEV-CONTROL"):
        if not re.search(rf"^{tag} MET \d{{4}}-\d{{2}}-\d{{2}}\s*$", text, re.M):
            raise HeldOutRefused(
                f"HELD-OUT refused: no dated '{tag} MET' line in the evidence file"
            )
    if held_out_logged():
        raise HeldOutRefused(
            "HELD-OUT refused: the dev log already records a HELD-OUT read (pre-reg 2.3: read ONCE)"
        )
    failed = selfchecks()
    if failed:
        raise HeldOutRefused(f"HELD-OUT refused: self-check FAIL at the frozen sha: {failed}")
    if not log_committed():
        raise HeldOutRefused(
            "HELD-OUT refused: the dev log is not tracked and clean at the frozen sha"
        )
    return sha


def field_d(f, seed=0, weeks=WEEKS):
    """E1's per-field value, or None when no eligible neuron is present in any scored week."""
    rng = np.random.default_rng(seed)
    per_week = []
    for w in weeks:
        idx = np.flatnonzero(f.present.get(w, []))
        if idx.size == 0:
            continue
        per_week.append(
            np.mean(
                [
                    mov_index(f.mov1[i], f.mov[w][i], rng).rdi
                    - stability_index(f.pdg1[i], f.pdg[w][i], rng).rdi
                    for i in idx
                ]
            )
        )
    return float(np.mean(per_week)) if per_week else None


def run(split, reader=read_field, seed=0):
    if result_file_dirty():
        raise RuntimeError(
            f"{OUT.name} is dirty: commit the result file after each run, then start again"
        )
    mice = DEV if split == "dev" else HELD_OUT
    if split == "heldout":
        refuse_held_out_unless_frozen()
    d, n_neurons = {}, {}
    for m in mice:
        fields = reader(m)
        vals = [field_d(f, seed) for f in fields]
        d[m] = None if any(v is None for v in vals) else float(np.mean(vals))
        n_neurons[m] = [int(f.mov1.shape[0]) for f in fields]
    out = {"split": split, "D": d, "n_neurons": n_neurons}
    if split == "heldout" and all(v is not None for v in d.values()) and len(d) == 8:
        out["E1"] = contrast("E1", list(d.values()), T_BAR)._asdict()
    else:
        out["E1"] = "NOT READ" if split == "dev" else "INCONCLUSIVE (n < 8)"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "heldout"), required=True)
    out = run(ap.parse_args(argv).split)
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(out, default=str))
    print(f"{OUT.name} written: commit the result file after each run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
