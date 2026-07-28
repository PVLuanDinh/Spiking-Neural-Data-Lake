"""
region_wiring -- the always-on daemon's FACULTY-WIRING REGISTRY.

The daemon runs Brain.tick() (which transitively charges many faculties on the wake/sleep path) but some
faculties are DORMANT (never reached) or PARTIAL (a signal computed then ignored -- an attention mask that
masks nothing, a plan with no reward, a recall that is write-only). This registry is the explicit plug
point: each wired faculty registers a `charge(brain, ctx)` adapter the daemon runs EVERY cycle, so a dormant
faculty becomes live WITHOUT editing the daemon loop.

STATUS, STATED PLAINLY: **the registry is EMPTY in production and always has been.** This docstring used
to claim "the daemon imports `wired_faculties` (which registers every adapter) at boot" -- there is no
`wired_faculties` module in the tree, `learn_daemon.py` imports only this module, and nothing anywhere
calls `wire()` outside the self-check below (which deliberately clears `_WIRED` first, so it proves the
mechanism and says nothing about production). So `charge_all()` iterates an empty list every cycle and the
daemon logs `adapters=0`. The plug point works; it has never been plugged. Registering a faculty here is
real P3 work, and claiming otherwise was the same species of defect this registry exists to expose.

Once registrations exist, they are picked up automatically: learn_daemon exits when git HEAD changes and
the launcher relaunches the new build, so the wired set tracks the code with no daemon edit per faculty.

HONEST COVERAGE: coverage() reports the EXPLICITLY-wired faculties per region against the brain-atlas region
map. It is NOT a claim that every module is exercised (Brain.tick charges more transitively; the coverage
audit measures that). It makes the wiring GAP visible + measurable, so "use every faculty" is tracked, not
faked.

Determinism / robustness (CLAUDE.md): charge_all() guards each faculty in its own try/except so one bad
faculty never stops the loop or blocks the others; pure registry, no wall-clock / no hash; ASCII-only.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.region_wiring
"""

# registry of (name, region, charge_fn, value). Module-global so `wire()` at import time (in
# wired_faculties) accumulates the whole set; the daemon reads it via charge_all()/coverage().
_WIRED = []


def wire(name, region, charge_fn, value="med"):
    """Register a faculty's per-cycle charge adapter. `charge_fn(brain, ctx)` returns a small result
    (dict / str / number) describing what it charged, for the coverage log. `value` in {high,med,low}.
    Returns charge_fn so it can be used as a decorator. Idempotent by (name): re-registering a name
    REPLACES it (so a reload / re-import does not double-register)."""
    global _WIRED
    _WIRED = [w for w in _WIRED if w[0] != str(name)]
    _WIRED.append((str(name), str(region), charge_fn, str(value)))
    return charge_fn


def wired():
    """The registered faculties as [(name, region, value)] -- introspection for reports/tests."""
    return [(n, r, v) for (n, r, _fn, v) in _WIRED]


def charge_all(brain, ctx=None):
    """Charge EVERY wired faculty this cycle. Each is guarded in its own try/except: a failure is recorded
    as an 'ERR ...' string, never raised, so one bad faculty never stops the always-on loop or blocks the
    others. Returns {name: result_or_error}."""
    ctx = {} if ctx is None else ctx
    out = {}
    for name, _region, fn, _value in _WIRED:
        try:
            out[name] = fn(brain, ctx)
        except Exception as e:  # noqa: BLE001 -- a bad faculty must never kill the always-on loop
            out[name] = f"ERR {type(e).__name__}: {e}"
    return out


def coverage():
    """Honest coverage of the EXPLICIT wirings vs the brain-atlas region map. Returns:
      {regions_total, regions_wired, n_wired_faculties, region_faculty_counts, unwired_regions}.
    This counts explicitly-wired faculties only -- NOT total brain coverage (Brain.tick charges more
    transitively). It surfaces the visible wiring gap so progress toward 'every faculty' is measurable."""
    regions_total = []
    try:
        from snn_data_lake import REGIONS  # the anatomical region -> module map (__init__.py)

        regions_total = sorted(REGIONS)
    except Exception:
        regions_total = []
    counts = {}
    for _n, region, _fn, _v in _WIRED:
        counts[region] = counts.get(region, 0) + 1
    return {
        "regions_total": len(regions_total),
        "regions_wired": len([r for r in regions_total if counts.get(r)]),
        "n_wired_faculties": len(_WIRED),
        "region_faculty_counts": counts,
        "unwired_regions": [r for r in regions_total if not counts.get(r)],
    }


def live_coverage():
    """The AUTHORITATIVE wiring coverage (Policy 3), delegating to substrate.dormancy_gate: how many
    brain modules are REACHABLE from a live path/entrypoint, how many are EXEMPT (with a reason), and
    how many are (silently) DORMANT -- which is 0 unless the dormancy gate is failing. This is distinct
    from coverage() above, which counts only the EXPLICIT per-cycle charge adapters registered here (an
    optional mechanism for genuinely-periodic faculties -- NOT a way to fake liveness for a dormant
    module; the honest way to make a faculty live is to WIRE it into the tick so it charges behaviour).
    Returns {total, reachable, exempt, dormant}."""
    from snn_data_lake.substrate import dormancy_gate

    c = dormancy_gate.coverage()
    return {
        "total": c["total"],
        "reachable": c["reachable"],
        "exempt": c["exempt"],
        "dormant": len(c["dormant"]),
    }


# ================================================================================================
def main():
    global _WIRED
    saved = _WIRED
    _WIRED = []  # isolate the self-check from any real registrations
    try:
        wire("dummy_ok", "cortex", lambda b, c: {"charged": True, "seen": c.get("tick", 0)})
        wire("dummy_bad", "cortex", lambda b, c: (_ for _ in ()).throw(RuntimeError("boom")))
        wire("dummy_amyg", "amygdala", lambda b, c: 1.0)
        # re-registering a name REPLACES it (reload-safe: no double registration)
        wire("dummy_ok", "cortex", lambda b, c: {"charged": True, "v2": True})
        assert len(wired()) == 3, f"re-wiring a name must REPLACE, not duplicate: {wired()!r}"

        res = charge_all(brain=None, ctx={"tick": 7})
        assert res["dummy_ok"] == {"charged": True, "v2": True}, res["dummy_ok"]
        assert res["dummy_amyg"] == 1.0
        # a failing faculty is GUARDED (recorded, not raised) so the others still charge
        assert res["dummy_bad"].startswith("ERR RuntimeError: boom"), res["dummy_bad"]

        cov = coverage()
        assert cov["region_faculty_counts"].get("cortex") == 2, cov  # dummy_ok + dummy_bad
        assert cov["region_faculty_counts"].get("amygdala") == 1, cov
        assert cov["n_wired_faculties"] == 3, cov
        lc = live_coverage()  # the AUTHORITATIVE Policy-3 coverage (delegates to the dormancy gate)
        assert lc["dormant"] == 0 and lc["reachable"] > 0 and lc["total"] > lc["reachable"], lc
        print("region_wiring self-check OK:")
        print(
            f"  live_coverage (Policy-3, via dormancy_gate): wired {lc['reachable']}/{lc['total']} "
            f"exempt {lc['exempt']} dormant {lc['dormant']}"
        )
        print(
            "  charge_all charged",
            len(res),
            "faculties; one failing faculty was GUARDED, not raised",
        )
        print("  re-wiring a name REPLACES it (reload-safe)")
        print(
            f"  coverage(): {cov['regions_wired']}/{cov['regions_total']} regions have >=1 explicit "
            f"wiring; unwired: {cov['unwired_regions']}"
        )
        return 0
    finally:
        _WIRED = saved


if __name__ == "__main__":
    import sys

    sys.exit(main())
