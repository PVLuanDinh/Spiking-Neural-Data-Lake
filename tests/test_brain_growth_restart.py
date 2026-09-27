"""A BRAIN that grows and exits must come back grown -- proven across a real process boundary.

specs/019 [B2.09]. `growth/neurogenesis_engine` already proves this at ENGINE level: its own
self-check spawns a subprocess and restores a byte-equal fingerprint. What was never proven is the
level that matters to a user -- the full `Brain -> LiveBrain` stack. The engine module contains ZERO
occurrences of `make_brain` or `Brain(`, so nothing connected the two.

WHY A NEW PROCESS AND NOT A NEW OBJECT. P4b: *"it persists" is a claim about a NEW process, not
about the object in front of you*. Re-reading a live engine's own attributes proves nothing; the
brain that grew must be gone before the claim means anything. This spawns a real interpreter.

THE CONTROL IS THE WHOLE TEST. Two brains that both grew NOTHING also agree, so agreement alone is
worthless. Three things are asserted together:
  * the parent actually grew (units > 0) -- otherwise there is nothing to restore;
  * the lake actually holds a _T_GROWTH record -- `retained()` once reported 1.000 while the lake
    held 0 records, which is the incident that produced P4b;
  * a child over an EMPTY lake does NOT come back grown -- the negative control that separates
    "restored" from "any brain looks like this".

Run: pytest tests/test_brain_growth_restart.py
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

# Growing takes real sleep ticks; keep it off the default fast path.
pytestmark = pytest.mark.slow

_CHILD = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.cortex.brain import make_brain
b = make_brain(seed=7, persist=True)
eng = b.neuro
out = {
    "units": int(eng.grown_units) if eng is not None else -1,
    "placed": len(eng.placement) if eng is not None else -1,
    "census": eng.layer_census() if eng is not None else {},
}
b.close()
print("RESULT " + json.dumps(out))
"""


def _run_child(home):
    """Construct a brain in a NEW interpreter over `home`'s lake and report what it restored."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, home],
        capture_output=True,
        text=True,
        timeout=900,
    )
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(f"child produced no RESULT.\nstdout:{proc.stdout}\nstderr:{proc.stderr}")


def _lake_growth_records(home):
    """Per-class growth-record counts on disk: {"legacy": _T_GROWTH, "chain": _T_GROWTH_DELTA}.
    Reads the FILE, not a live object.

    `knowledge_path()` -> <home>/brain_lake.bin is THE ONE durable lake. NOT `lake_path()`, which
    is <home>/lake.spkl -- a different store. The first draft of this helper used `lake_path` and
    reported 0 records for a brain that had persisted 3,618 bytes perfectly well, which read as a
    P4b violation in the brain. It was a false negative from the instrument. `_assert_probe_sees`
    below is the positive control that would have caught it in one line, and is why it exists.

    BOTH classes counted since specs/022 T010-ii: the brain's flush writes chain records now, so a
    legacy-only counter would report 0 for a perfectly persisted brain -- the same instrument false
    negative as the lake_path draft, one abstraction up. Split rather than summed so the tests can
    assert the CUTOVER (chain written, legacy not) instead of just presence."""
    from snn_data_lake.substrate.home import _T_GROWTH, _T_GROWTH_DELTA, BinLog, knowledge_path

    saved = os.environ.get("NORD_HOME")
    os.environ["NORD_HOME"] = home
    try:
        path = knowledge_path()
        counts = {"legacy": 0, "chain": 0}
        if not os.path.exists(path):
            return counts
        for p in BinLog(path).replay():
            if p[:1] == _T_GROWTH:
                counts["legacy"] += 1
            elif p[:1] == _T_GROWTH_DELTA:
                counts["chain"] += 1
        return counts
    finally:
        if saved is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = saved


def _assert_probe_sees(home):
    """POSITIVE CONTROL: write one record of EACH growth class and confirm the counter SEES both.

    Without this the whole test can only produce false negatives -- a helper reading the wrong file
    returns 0 forever, and 0 is exactly what a real persistence failure looks like. Both halves are
    controlled because both are asserted on below: a raw blob takes the legacy _T_GROWTH path, and a
    minimal zip-of-.npy blob takes the _T_GROWTH_DELTA chain path (the chain parses zip members,
    never .npy internals, so plain bytes suffice).
    """
    import io
    import zipfile

    from snn_data_lake.substrate.home import save_growth

    npz_probe = io.BytesIO()
    with zipfile.ZipFile(npz_probe, "w", zipfile.ZIP_STORED) as z:
        z.writestr(zipfile.ZipInfo("probe.npy"), b"chain-probe")

    saved = os.environ.get("NORD_HOME")
    os.environ["NORD_HOME"] = home
    try:
        before = _lake_growth_records(home)
        save_growth(b"positive-control-probe")
        save_growth(npz_probe.getvalue())
        after = _lake_growth_records(home)
    finally:
        if saved is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = saved
    assert after["legacy"] == before["legacy"] + 1, (
        f"the LEGACY growth counter is BLIND: wrote one raw record and it went "
        f"{before['legacy']} -> {after['legacy']}. Every legacy assertion here would be a false "
        "negative."
    )
    assert after["chain"] == before["chain"] + 1, (
        f"the CHAIN growth counter is BLIND: wrote one npz record and it went "
        f"{before['chain']} -> {after['chain']}. Every chain assertion here would be a false "
        "negative."
    )


_GROW_THEN_EXIT = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.cortex.brain import make_brain
b = make_brain(seed=7, persist=True)
start = int(b.neuro.grown_units)
at = -1
for t in range(1, 601):
    b.tick()
    if int(b.neuro.grown_units) > start:
        at = t
        break
out = {"units": int(b.neuro.grown_units), "grew_at": at}
b.close()                       # <-- exits IMMEDIATELY after growing. This is the whole test.
print("RESULT " + json.dumps(out))
"""


def test_growth_survives_an_immediate_exit():
    """A brain that grows and exits AT ONCE must still come back grown -- `Brain.close()` flushes.

    The 260-tick test below cannot catch this. Over that many ticks a sleep-tick save always lands
    after the last growth, so save-state and exit-state agree by luck rather than by design.

    MEASURED before `Brain._flush_growth` existed, same seed, same script: a brain ticked only until
    `grown_units` rose kept **0 of 2 units** and wrote ZERO lake records, and a brain given 200 more
    ticks exited holding 3 units but restored **5**, because the last cadence save had caught a
    PRE-PRUNE snapshot. Too little and too much, from one missing flush. That is the reachable red
    for this test, and it is why the assertion is equality rather than `child > 0`.
    """
    home = tempfile.mkdtemp(prefix="brain_flush_")
    _assert_probe_sees(tempfile.mkdtemp(prefix="brain_flush_probe_"))

    proc = subprocess.run(
        [sys.executable, "-c", _GROW_THEN_EXIT, home], capture_output=True, text=True, timeout=1800
    )
    parent = None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("RESULT "):
            parent = json.loads(line[len("RESULT ") :])
            break
    assert parent is not None, f"parent produced no RESULT.\n{proc.stdout}\n{proc.stderr}"

    if parent["units"] <= 0:
        pytest.skip("the parent never grew inside the tick budget -- nothing to flush")

    recs = _lake_growth_records(home)
    assert sum(recs.values()) > 0, (
        f"parent grew {parent['units']} units at tick {parent['grew_at']} and exited, and the lake "
        "holds NO growth record -- close() did not flush"
    )
    # CUTOVER CHARGE PROOF (T010-ii): the brain's own flush must land on the chain, not the legacy
    # tag. Without this the chain could sit dormant behind a fallback that always fires -- presence
    # is not participation, and this is the assert that separates the two.
    assert recs["chain"] > 0 and recs["legacy"] == 0, (
        f"the live flush path did not cut over to the chain: {recs['chain']} chain records, "
        f"{recs['legacy']} legacy records (expected >0 and 0)"
    )
    child = _run_child(home)
    assert child["units"] == parent["units"], (
        f"exit-time growth was not durable: parent exited with {parent['units']} units, child "
        f"restored {child['units']}. The lake holds a DIFFERENT brain than the one that exited."
    )
    assert child["placed"] == parent["units"], (
        f"units survived but placement did not: {child['placed']} placements for "
        f"{child['units']} units"
    )


def test_a_brain_that_grew_comes_back_grown():
    home = tempfile.mkdtemp(prefix="brain_restart_")

    # --- POSITIVE CONTROL FIRST, on a THROWAWAY home so its probe record cannot be counted as the
    # brain's growth below. Establishes that the counter can see a record at all before any negative
    # from it is believed.
    _assert_probe_sees(tempfile.mkdtemp(prefix="brain_restart_probe_"))

    # --- NEGATIVE CONTROL, before anything is written. A brain over an EMPTY lake must come
    # back with nothing; without this, "the child has units" could just be what every brain does.
    empty = _run_child(home)
    assert empty["units"] <= 0, (
        f"a brain over an EMPTY lake reported {empty['units']} grown units -- then agreement later "
        "proves nothing, because a fresh brain already looks restored"
    )
    assert sum(_lake_growth_records(home).values()) == 0, (
        "the empty lake already held a growth record"
    )

    # --- PARENT: grow in one process, then let it exit.
    grow = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
from snn_data_lake.cortex.brain import make_brain
b = make_brain(seed=7, persist=True)
for _ in range(int(sys.argv[2])):
    b.tick()
eng = b.neuro
out = {"units": int(eng.grown_units) if eng is not None else -1,
       "placed": len(eng.placement) if eng is not None else -1,
       "census": eng.layer_census() if eng is not None else {}}
b.close()
print("RESULT " + json.dumps(out))
"""
    proc = subprocess.run(
        [sys.executable, "-c", grow, home, "260"], capture_output=True, text=True, timeout=1800
    )
    parent = None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("RESULT "):
            parent = json.loads(line[len("RESULT ") :])
            break
    assert parent is not None, f"parent produced no RESULT.\n{proc.stdout}\n{proc.stderr}"

    if parent["units"] <= 0:
        pytest.skip(
            "the parent grew nothing in the tick budget -- this test has nothing to prove, and "
            "asserting on a zero would be a vacuous pass"
        )

    # The lake must actually hold it. `retained()` once read 1.000 against 0 records.
    recs = _lake_growth_records(home)
    assert sum(recs.values()) > 0, (
        f"parent reports {parent['units']} grown units but the lake on disk holds NO growth "
        "record -- the substrate died with the process"
    )
    # CUTOVER CHARGE PROOF (T010-ii), same as the flush test: chain written, legacy untouched.
    assert recs["chain"] > 0 and recs["legacy"] == 0, (
        f"the live flush path did not cut over to the chain: {recs['chain']} chain records, "
        f"{recs['legacy']} legacy records (expected >0 and 0)"
    )

    # --- CHILD: a NEW interpreter over the same lake.
    child = _run_child(home)
    assert child["units"] == parent["units"], (
        f"restart lost substrate: parent {parent['units']} units -> child {child['units']}"
    )
    # PLACEMENT must survive with it, or the units come back without their cortical identity.
    assert child["placed"] == parent["placed"] == parent["units"], (
        f"placement did not survive the restart: parent {parent['placed']}, child {child['placed']}"
    )
    assert child["census"] == parent["census"], (
        f"laminar structure changed across the restart: {parent['census']} -> {child['census']}"
    )
