"""Finding (iii) (launcher D-77, board #6807): a restored brain must resume its clock where an
uninterrupted twin stands, not start a fresh day (`Brain.tick` derives sleep/wake from `_step` and
the clock, which a restart reset). The clock phase rides the named record `circadian_clock.json`
(_T_STATE). Dev seed 15000 (r2-typed-replay-hydration.md 2.6).

Panel lens A BLOCK-1 (#6910): seed 15000's fresh clock starts in subjective night (phase 0.8209), so
a single "asleep" bit after a cut inside the FIRST night cannot tell a restored brain from a fresh
one. The cut is therefore MID-DAY after the first night (a fresh brain is asleep there, the twin is
awake), the restored STATE is asserted (`_step`, clock phase, jitter rng), and a full day period of
ticks is compared with the twin's. A stdlib simulation of CircadianClock (round-2 REPORT) gives a
first divergence at tick 0 on the unfixed code, and none over the period with the fix.

Policy lens P-1 (#6935, P4b "a NEW process ... after the writer is gone"): every brain runs in its
own interpreter over a private home under tmp_path (never the live lake), as
tests/test_brain_growth_restart.py and tests/test_p4b_restart_restore.py do."""

import json
import os
import subprocess
import sys

import pytest

# Restarts spawn interpreters and drive a day of ticks; keep them off the default fast path.
pytestmark = pytest.mark.slow

SEED = 15000
DAY_PERIOD = 200  # make_brain's default day period
WAKE_TICKS_INTO_DAY = 3

# One brain in a NEW interpreter: argv[1] = the home, argv[2] = a JSON spec. Prints "RESULT {...}".
_CHILD = r"""
import hashlib, json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
spec = json.loads(sys.argv[2])
from snn_data_lake.substrate import home
home.require_isolated_home()
from snn_data_lake.engine.brain import make_brain
# brain.CIRCADIAN_STATE, as a literal so the child imports on the unfixed base too
CIRCADIAN_STATE = "circadian_clock.json"
if "rewrite" in spec:
    home.save_state(CIRCADIAN_STATE, dict(home.load_state(CIRCADIAN_STATE), **spec["rewrite"]))
if "drop" in spec:  # a record missing one binding key (a key `rewrite` cannot delete)
    rec = home.load_state(CIRCADIAN_STATE)
    del rec[spec["drop"]]
    home.save_state(CIRCADIAN_STATE, rec)
if "replace" in spec:  # a record that is not this build's layout at all
    home.save_state(CIRCADIAN_STATE, spec["replace"])
if "rewrite_gauss_next" in spec:  # the cached normal inside the codec's encoded rng state
    rec = home.load_state(CIRCADIAN_STATE)
    rec["rng"]["state"]["items"][2] = spec["rewrite_gauss_next"]
    home.save_state(CIRCADIAN_STATE, rec)
if "replace_bytes" in spec:  # a JSON null or a non-JSON blob: load_state reads either as absent
    home.save_named_state(CIRCADIAN_STATE, spec["replace_bytes"].encode("ascii"))
written = home.load_state(CIRCADIAN_STATE, default=None)
load_state = home.load_state
if spec.get("restore_raises"):  # the restore's read raises (a lake that will not open), once

    def _raising(name, default=None):
        if name == CIRCADIAN_STATE:
            raise OSError("lake will not open")
        return load_state(name, default)

    home.load_state = _raising
from snn_data_lake.readout import interaction_learning
merge = interaction_learning._merge_state
# codec_raises: a defect in the shared codec, not a malformed record (every rng raises);
# codec_raises_on_written: only the written record's rng raises (a newer layout), kept copies load
if spec.get("codec_raises") or spec.get("codec_raises_on_written"):
    on_written = spec.get("codec_raises_on_written")

    def _broken(target, saved):
        if on_written:
            if isinstance(written, dict) and saved == written.get("rng"):
                raise RuntimeError("newer layout")
        elif isinstance(saved, dict) and saved.get("__kind__") == "random":
            raise RuntimeError("codec defect")
        return merge(target, saved)

    interaction_learning._merge_state = _broken

def count_records():
    names = (home._unpack_state(p) for p in home._log().replay() if p[:1] == home._T_STATE)
    return sum(1 for got in names if got and got[0] == CIRCADIAN_STATE.encode())

def clock(b):
    rng = hashlib.sha256(repr(b.clock.rng.getstate()).encode()).hexdigest()
    return {"step": b._step, "phase": b.clock.phase, "rng": rng}

b = make_brain(seed=spec["seed"], persist=True, device="cpu")
home.load_state, interaction_learning._merge_state = load_state, merge
s = b.state()
# .get: the same child runs on the unfixed base (no such keys), where the red must fail
# on its assert, never on a crash
out = dict(clock(b), note=s.get("circadian_restore"), restored=s.get("circadian_restored"))
if spec.get("kill_at_growth_flush"):  # R2's K1g boundary: os._exit right after the growth flush
    grow = b._flush_growth
    def _killed():
        grow()
        print("RESULT " + json.dumps(dict(out, killed_at=b._step)), flush=True)
        os._exit(0)
    b._flush_growth = _killed
if spec.get("no_keep"):
    b._keep_refused = lambda *a, **k: None
ticks, slept, woke = 0, False, 0
into_day = spec.get("wake_ticks_into_day")
while woke < into_day if into_day is not None else ticks < spec.get("ticks", 0):
    ticks += 1
    if ticks > 4 * b.day_period:
        raise SystemExit("no night and dawn within four periods")
    if b.tick() == "asleep":
        slept = True
    elif slept:
        woke += 1
out.update(cut=ticks, at=clock(b), after=[b.tick() for _ in range(spec.get("after", 0))])
out.update(nights=b._nights, flush_failed=b.state().get("circadian_flush_failed"))
out.update(lake_night=b._lake_night)
b.close()
rec = home.load_state(CIRCADIAN_STATE, default=None)
kept = home.named_states("circadian_clock.refused.")
out.update(
    written=written,
    record=rec,
    record_bytes=len(json.dumps(rec)),
    record_count=count_records(),
    kept={name: json.loads(blob) for name, blob in kept},
    note_at_close=getattr(b, "_circadian_restore_note", None),
)
if spec.get("consolidate"):  # the night block's prune, run once more on the closed lake
    klog = home.KnowledgeLog()
    klog.consolidate(list(klog.replay()))
    out.update(consolidated_count=count_records(), consolidated=home.load_state(CIRCADIAN_STATE))
print("RESULT " + json.dumps(out))
"""


def run_brain(home, **spec):
    """Construct, drive and close one brain in a NEW interpreter over `home`; returns its RESULT."""
    env = dict(os.environ, PYTHONUTF8="1", NORD_HOME=str(home))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (os.path.join(os.getcwd(), "src"), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(home), json.dumps(spec)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    # not AssertionError: the r2-instr-fix copy's strict xfail must never pass on a crash
    raise RuntimeError(f"child produced no RESULT.\nstdout:{proc.stdout}\nstderr:{proc.stderr}")


def private_home(tmp_path, tag):
    from snn_data_lake.substrate.home import assert_not_live_home

    path = tmp_path / tag
    path.mkdir()
    return assert_not_live_home(str(path))


def test_a_restored_brain_resumes_its_twins_clock(tmp_path):
    twin = run_brain(
        private_home(tmp_path, "twin"),
        seed=SEED,
        wake_ticks_into_day=WAKE_TICKS_INTO_DAY,
        after=DAY_PERIOD,
    )
    home = private_home(tmp_path, "cut")
    run_brain(home, seed=SEED, ticks=twin["cut"])
    restored = run_brain(home, seed=SEED, after=DAY_PERIOD)
    at_construction = {k: restored[k] for k in ("step", "phase", "rng")}
    assert at_construction == twin["at"], "restored clock is not the twin's at the cut"
    assert restored["after"] == twin["after"], "the restored brain's day diverged from the twin's"
