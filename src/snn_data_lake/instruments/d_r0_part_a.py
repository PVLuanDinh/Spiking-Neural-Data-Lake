r"""D-R0 Part A driver (pre-registration specs/022-unified-brain-arc/evidence/d-r0-representational-drift-
instrument.md 2.3, 2.4): per mouse, E1 = D = mean over weeks {2,3,4,5} present of the mean over neurons present
that week of (RDI_MOV - RDI_PDG); then the paired contrast over mice.

MOVED 2026-11-04 from `tools/d_r0_partA.py` (owner "Move D-R0 tools"; P4c ratchet: a tools/ script is never
governed), byte-for-byte the same driver logic; evidence file section 11 records the move.

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
the dev log holds no HELD-OUT file read yet (read ONCE); both module self-checks re-run green HERE, at that sha (the
driver runs them itself); and `tools/d_r0_dev_log.jsonl` is tracked and clean in git at that sha. DEV needs none of
these. The driver refuses to start while `tools/d_r0_partA.json` is dirty: COMMIT THE RESULT FILE AFTER EACH RUN.

Run as a package module, so the calling tool `ophys_ingest.read_response` hashes is this file, as `__main__`:

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.d_r0_part_a --split dev|heldout
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.instruments.d_r0_part_a     # no argument: the self-check
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr

from snn_data_lake.instruments.paired_contrast import T_BAR, contrast
from snn_data_lake.instruments.representational_drift import (
    MOV_CAP,
    MOV_FLOOR,
    N_BOOT,
    SEED,
    SEED_BOOT,
    author_agreement_verdict,
    mov_index,
    stability_index,
    synth_pair,
)

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    ROOT
    / "specs"
    / "022-unified-brain-arc"
    / "evidence"
    / "d-r0-representational-drift-instrument.md"
)
OUT = ROOT / "tools" / "d_r0_partA.json"
CHECKS_OUT = ROOT / "tools" / "d_r0_dev_checks.json"
COMPONENTS = ("CC_ws", "CC_bs", "RDI")  # DriftReading order; StabilityData field names
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
    theirs: dict | None = (
        None  # {'mov'|'pdg': {week: authors' StabilityData.RDI over the eligible set}}
    )


# distinct-from: sensorimotor/ophys_layout.py -- prints variable names/shapes only and never builds response
#   arrays; this builds the eligible-neuron (neurons, trials, frames) arrays the RDI reads. Both open files only
#   through ophys_ingest.read_response (the one reader), which this REUSES.
# distinct-from: reservoir/recurrent_pool.py, faculties/affect_core.py, faculties/theory_of_mind.py,
#   reasoning/mcr_hdc.py, readout/tempotron.py -- the write hook's second match list (generic-word overlap); per
#   their docstrings (window checks, spiking WTA, phasor means, tempotron) none reads a .mat deposit.
ODD_WEEKS = {
    7: (1, 2, 3, 5, 6, 7),
    8: (1, 2, 4, 5, 6),
}  # pre-reg 1.3; every other mouse: week = session + 1


def _weeks(mouse, n_sessions):
    weeks = ODD_WEEKS.get(mouse, tuple(range(1, n_sessions + 1)))
    if len(weeks) != n_sessions:
        raise ValueError(f"mouse {mouse}: file has {n_sessions} sessions, listed weeks {weeks}")
    return weeks


def read_field(mouse, split, data_dir=None, pin=None, **kw):
    """One FieldData per field of `mouse` (Mouse 2 has Window 1/2).
    PORTED 2026-11-15 (final-project M3) from claude/d-r0-build-1103 tools/d_r0_partA.py rounds 5-8 (fdf4a4c7),
    which never reached the package module when the tool was moved (the round-1 stub was carried instead)."""
    from snn_data_lake.sensorimotor.ophys_ingest import DATA_DIR

    data_dir = DATA_DIR if data_dir is None else data_dir
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8")) if pin is None else pin
    prefix = f"Main experiments/Mouse {mouse}/"
    folders = sorted(
        a["path"].rsplit("/", 1)[0]
        for a in pin["assets"]
        if a["path"].startswith(prefix) and a["path"].endswith("/RespData.mat")
    )
    if not folders:
        raise ValueError(f"no RespData.mat pinned for mouse {mouse}")
    return [_read_one(mouse, f, split, data_dir, pin, **kw) for f in folders]


def _check_axes(mouse, pdg, mov, roi):
    if pdg.ndim != 4 or mov.ndim != 4:
        raise ValueError(f"mouse {mouse}: PDG {pdg.shape} / MOV {mov.shape} are not 4-D")
    n_s = mov.shape[3]
    if pdg.shape[3] != n_s or roi.presence.shape[1] != n_s:
        raise ValueError(f"mouse {mouse}: session axes disagree (PDG/MOV/presence)")
    if pdg.shape[0] < 2 or mov.shape[0] < 8:
        raise ValueError(
            f"mouse {mouse}: PDG trials {pdg.shape[0]} < 2 or MOV trials {mov.shape[0]} < 8"
        )
    n_n = {pdg.shape[2], mov.shape[2], len(roi.quality), roi.presence.shape[0]}
    if len(n_n) != 1:
        raise ValueError(f"mouse {mouse}: neurons disagree across arrays: {n_n}")
    return n_s


def _read_one(mouse, folder, split, data_dir, pin, **kw):
    from snn_data_lake.sensorimotor.ophys_ingest import read_response

    def mat(name):
        return read_response(Path(data_dir) / folder / name, pin, split, data_dir=data_dir, **kw)

    roi, resp = mat("RoiINFO.mat")["RoiINFO"], mat("RespData.mat")["RespData"]
    pdg, mov = (
        resp.PDG.RespMat_Full,
        resp.NatMov.RespMat_onTime,
    )  # (trials, frames, neurons, sessions)
    weeks = _weeks(mouse, _check_axes(mouse, pdg, mov, roi))
    elig = (
        (roi.PDG_Responsive_thresh > 0)
        & (roi.NatMov_Responsive_thresh > 0)
        & (roi.quality >= 3)
        & (roi.presence[:, 0] > 0)
    )

    def take(a, s):
        return np.transpose(a[:, :, elig, s], (2, 0, 1))

    sessions = list(enumerate(weeks))[1:]
    stab = mat("StabilityData.mat")[
        "StabilityData"
    ]  # RDI rows = sessions 2..n vs session 1, cols = all neurons
    theirs = {  # per week: (eligible neurons, 3) = the authors' (CC_ws, CC_bs, RDI)
        key: {
            w: np.stack(
                [np.asarray(getattr(getattr(stab, name), c))[s - 1, elig] for c in COMPONENTS],
                axis=1,
            )
            for s, w in sessions
        }
        for key, name in (("mov", "NatMov"), ("pdg", "PDG"))
    }
    return FieldData(
        take(mov, 0),
        take(pdg, 0),
        {w: take(mov, s) for s, w in sessions},
        {w: take(pdg, s) for s, w in sessions},
        {w: roi.presence[elig, s] > 0 for s, w in sessions},
        theirs,
    )


# distinct-from: instruments/everyday_battery.py -- per-seed process launcher (docstring); no RDI.
# distinct-from: instruments/replay_hydration.py -- per-seed readings of its own pre-registration; no RDI.
# distinct-from: instruments/report_order_probe.py -- per-seed reading over recording rows; no RDI.
# distinct-from: instruments/delayed_cue.py -- fair-bit stream per presentation; no RDI.
# distinct-from: instruments/joint_mask_search.py -- constructed per-seed readings of C22; no RDI.
# The verdicts are REUSED from representational_drift (author_agreement_verdict, mov_index, stability_index).
def _neuron_sessions(f, stim, seed, weeks=WEEKS):
    """Ours and the authors' (CC_ws, CC_bs, RDI) for every (eligible neuron, scored week) present that week:
    two (k, 3) arrays."""
    rng = np.random.default_rng(seed)
    fn, s1, sw = (mov_index, f.mov1, f.mov) if stim == "mov" else (stability_index, f.pdg1, f.pdg)
    ours, theirs = [], []
    for w in weeks:
        for i in np.flatnonzero(f.present.get(w, [])):
            ours.append(tuple(fn(s1[i], sw[w][i], rng)))
            theirs.append(f.theirs[stim][w][i])
    return (
        np.array(ours, dtype=np.float64).reshape(-1, 3),
        np.array(theirs, dtype=np.float64).reshape(-1, 3),
    )


# distinct-from: readout/interaction_learning.py -- one subprocess per seed/arm (docstring); no RDI.
# distinct-from: instruments/h01_partb_tangential.py -- a mixture's held-out advantage per seed; no RDI.
def _component(x1, x2, th):
    """Ceiling, self_mad, rho, mad for one component, and the relative (doc-rule) pass: rho >= max(0.9 ceil, 0.8),
    mad <= min(2 self_mad, 0.05), n >= 200 -- author_agreement_verdict without its absolute ceil >= 0.9 clause."""
    ceil = float(spearmanr(x1, x2)[0])
    self_mad = float(np.median(np.abs(x1 - x2)))
    rho = float(spearmanr(x1, th)[0])
    mad = float(np.median(np.abs(x1 - th)))
    rel = rho >= max(0.9 * ceil, 0.8) and mad <= min(2 * self_mad, 0.05) and len(x1) >= 200
    return {"ceil": ceil, "self_mad": self_mad, "rho": rho, "mad": mad, "relative_pass": bool(rel)}


def dev_agreement(fields, stim):
    """Pre-reg 2.4 per stimulus over all `fields`: seeds 1 vs 2 give the ceiling, ours = seed 1 vs StabilityData.
    Three rules are reported (research/representational-drift/agreement_rule_sim.py): 'dr0' =
    author_agreement_verdict on RDI (registered), 'doc' = the relative clauses on RDI only, 'components' = the
    relative clauses on CC_ws, CC_bs AND RDI. Neuron-sessions with any non-finite value are dropped and counted."""
    a, th = (np.concatenate(x) for x in zip(*(_neuron_sessions(f, stim, 1) for f in fields)))
    b = np.concatenate([_neuron_sessions(f, stim, 2)[0] for f in fields])
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(th).all(1)
    comp = {c: _component(a[ok, j], b[ok, j], th[ok, j]) for j, c in enumerate(COMPONENTS)}
    r = comp["RDI"]
    dr0, _, _ = author_agreement_verdict(r["ceil"], r["self_mad"], a[ok, 2], th[ok, 2])
    return {
        "met": dr0,
        "rules": {
            "dr0": dr0,
            "doc": r["relative_pass"],
            "components": all(c["relative_pass"] for c in comp.values()),
        },
        "components": comp,
        "n": int(ok.sum()),
        "dropped": int((~ok).sum()),
    }


def dev_control(fields, seed=SEED, weeks=WEEKS):
    """Pre-reg 2.2 mov_vs_mov on one mouse's fields: interleaved odd/even MOV arms (amendment 9: floor(T/2) trials
    each, 8 picked), D_null = mean neuron diff (arm A - arm B) over present weeks; bar max(0.05, 3 x bootstrap SE)."""
    diffs = []
    for k, f in enumerate(fields):
        rng = np.random.default_rng([seed, 4, k])
        per = {}
        for w in weeks:
            for i in np.flatnonzero(f.present.get(w, [])):
                a = mov_index(f.mov1[i][0::2], f.mov[w][i][0::2], rng).rdi
                b = mov_index(f.mov1[i][1::2], f.mov[w][i][1::2], rng).rdi
                per.setdefault(i, []).append(a - b)
        diffs += [float(np.mean(v)) for v in per.values()]
    diffs = np.array(diffs)
    boot = np.random.default_rng(SEED_BOOT).choice(diffs, (N_BOOT, diffs.size))
    se = float(boot.mean(1).std(ddof=1))
    bar, d_null = max(MOV_FLOOR, 3 * se), float(diffs.mean())
    return {
        "met": bool(abs(d_null) <= bar and bar <= MOV_CAP),
        "d_null": d_null,
        "bar": bar,
        "n": int(diffs.size),
    }


def dev_checks(reader=read_field, data_dir=None):
    """DEV-AGREEMENT (both stimuli, pooled over DEV mice) and DEV-CONTROL (per DEV mouse); one result dict."""
    fields = {m: reader(m, "dev", data_dir=data_dir) for m in DEV}
    pooled = [f for m in DEV for f in fields[m]]
    agree = {s: dev_agreement(pooled, s) for s in ("mov", "pdg")}
    control = {m: dev_control(fields[m]) for m in DEV}
    return {
        "agreement": agree,
        "control": control,
        "DEV-AGREEMENT": all(a["rules"]["components"] for a in agree.values()),  # amendment 13
        "DEV-CONTROL": all(c["met"] for c in control.values()),
    }


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


def run(split, reader=read_field, seed=0, data_dir=None):
    if result_file_dirty():
        raise RuntimeError(
            f"{OUT.name} is dirty: commit the result file after each run, then start again"
        )
    mice = DEV if split == "dev" else HELD_OUT
    if split == "heldout":
        refuse_held_out_unless_frozen()
    d, n_neurons, pooled = {}, {}, []
    for m in mice:
        fields = reader(m, split) if data_dir is None else reader(m, split, data_dir=data_dir)
        pooled += fields
        vals = [field_d(f, seed) for f in fields]
        d[m] = None if any(v is None for v in vals) else float(np.mean(vals))
        n_neurons[m] = [int(f.mov1.shape[0]) for f in fields]
    out = {"split": split, "D": d, "n_neurons": n_neurons}
    agree_ok = True
    if all(
        f.theirs is not None for f in pooled
    ):  # amendment 13: the components rule, recomputed on this split
        out["agreement"] = {s: dev_agreement(pooled, s) for s in ("mov", "pdg")}
        agree_ok = all(a["rules"]["components"] for a in out["agreement"].values())
    if split == "dev":
        out["E1"] = "NOT READ"
    elif not agree_ok:
        out["E1"] = "INSTRUMENT BLIND (held-out agreement, components rule, NOT MET)"
    elif all(v is not None for v in d.values()) and len(d) == 8:
        out["E1"] = contrast("E1", list(d.values()), T_BAR)._asdict()
    else:
        out["E1"] = "INCONCLUSIVE (n < 8)"
    return out


_SHA = "b" * 40
_DEV_LINES = "DEV-AGREEMENT MET 2026-11-08\nDEV-CONTROL MET 2026-11-07\n"


def _synth_field(dose_mov, dose_pdg, n=6, weeks=(2, 3), seed=5):
    mov1, mov_w = synth_pair(n, dose_mov, seed, trials=30)
    pdg1, pdg_w = synth_pair(n, dose_pdg, seed + 1)
    return FieldData(
        mov1,
        pdg1,
        {w: mov_w for w in weeks},
        {w: pdg_w for w in weeks},
        {w: np.ones(n, bool) for w in weeks},
    )


def _gate_reds(tmp):
    """Each held-out refusal fires on its constructed red; the all-conditions control passes."""
    failures = []
    ev = Path(tmp) / "ev.md"
    ok = {"selfchecks": lambda: [], "log_committed": lambda: True, "held_out_logged": lambda: False}
    frozen = f"FROZEN-SHA 2026-11-10 {_SHA}\n"
    reds = (
        ("no FROZEN-SHA", "no amendment\n" + _DEV_LINES, _SHA, {}),
        ("not the frozen sha", frozen + _DEV_LINES, "a" * 40, {}),
        ("DEV-AGREEMENT", frozen + "DEV-CONTROL MET 2026-11-07\n", _SHA, {}),
        ("DEV-CONTROL", frozen + "DEV-AGREEMENT MET 2026-11-08\n", _SHA, {}),
        ("read ONCE", frozen + _DEV_LINES, _SHA, {"held_out_logged": lambda: True}),
        ("self-check FAIL", frozen + _DEV_LINES, _SHA, {"selfchecks": lambda: ["ophys_ingest"]}),
        ("dev log", frozen + _DEV_LINES, _SHA, {"log_committed": lambda: False}),
    )
    for needle, text, head, bad in reds:
        ev.write_text(text, encoding="utf-8")
        try:
            refuse_held_out_unless_frozen(ev, head=head, **{**ok, **bad})
            failures.append(f"held-out red '{needle}' was NOT refused")
        except HeldOutRefused as exc:
            if needle not in str(exc):
                failures.append(f"held-out red '{needle}' refused for another reason: {exc}")
    ev.write_text(frozen + _DEV_LINES, encoding="utf-8")
    try:
        if refuse_held_out_unless_frozen(ev, head=_SHA, **ok) != _SHA:
            failures.append("green control did not return the frozen sha")
    except HeldOutRefused as exc:
        failures.append(f"green control refused: {exc}")
    return failures


def _read_once_reds(tmp):
    """A held-out sha in the log, and a sha absent from the pin, both count as a held-out read."""
    pin = {
        "assets": [
            {"path": "h.mat", "sha256": "h" * 64, "split": "held-out"},
            {"path": "d.mat", "sha256": "d" * 64, "split": "dev"},
        ]
    }
    log = Path(tmp) / "log.jsonl"
    failures = []
    for shas, want in ((("d" * 64,), False), (("d" * 64, "h" * 64), True), (("z" * 64,), True)):
        log.write_text(
            "".join(json.dumps({"files": [{"sha256": s}]}) + "\n" for s in shas), encoding="utf-8"
        )
        if held_out_read_in_log(log, pin) is not want:
            failures.append(f"held_out_read_in_log({[s[0] for s in shas]}) is not {want}")
    if held_out_read_in_log(Path(tmp) / "absent.jsonl", pin):
        failures.append("an absent log reads as a held-out read")
    return failures


def _field_d_reds():
    """E1's sign follows which stimulus drifts; a field with no present neuron is None, never 0."""
    failures = []
    more_mov, more_pdg = _synth_field(0.85, 0.0), _synth_field(0.0, 0.85)
    if not field_d(more_mov) > 0.05:
        failures.append(f"field_d(MOV drifts) = {field_d(more_mov)} is not > 0.05")
    if not field_d(more_pdg) < -0.05:
        failures.append(f"field_d(PDG drifts) = {field_d(more_pdg)} is not < -0.05")
    empty = more_mov._replace(present={w: np.zeros(6, bool) for w in more_mov.present})
    if field_d(empty) is not None:
        failures.append("field_d with no present neuron is not None")
    return failures


def _selfcheck():
    failures = []
    if sorted(DEV + HELD_OUT) != list(range(1, 13)):
        failures.append("DEV + HELD_OUT is not mice 1..12")
    with tempfile.TemporaryDirectory() as tmp:
        failures += _gate_reds(tmp)
        failures += _read_once_reds(tmp)
    failures += _field_d_reds()
    from snn_data_lake.sensorimotor.ophys_ingest import IngestRefused

    try:  # the reader's red: a HELD-OUT mouse requested as DEV is refused before any byte is read
        read_field(1, "dev")
        failures.append("read_field(1, 'dev') read a held-out mouse as DEV")
    except IngestRefused as e:
        if "(split)" not in str(e):
            failures.append(f"read_field(1, 'dev') refused for the wrong reason: {e}")
    for f in failures:
        print("FAIL", f)
    if failures:
        return 1
    print(
        "d_r0_part_a self-check OK: seven held-out reds refuse (no FROZEN-SHA, HEAD != sha, DEV-AGREEMENT, "
        "DEV-CONTROL, read ONCE, self-check FAIL, dev log), the all-conditions control passes, read-once fails "
        "closed, E1's sign follows the drifting stimulus, read_field refuses a held-out mouse requested as DEV"
    )
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return _selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "heldout"), required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--checks", action="store_true", help="DEV-AGREEMENT + DEV-CONTROL (dev only)")
    a = ap.parse_args(argv)
    if a.checks:
        if a.split != "dev":
            raise SystemExit("--checks is DEV only")
        res = dev_checks(data_dir=a.data_dir)
        CHECKS_OUT.write_text(json.dumps(res, indent=1, default=str) + chr(10), encoding="utf-8")
        print(json.dumps(res, default=str))
        return 0
    out = run(a.split, data_dir=a.data_dir)
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(out, default=str))
    print(f"{OUT.name} written: commit the result file after each run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
