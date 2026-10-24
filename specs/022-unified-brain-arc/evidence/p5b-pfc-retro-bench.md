# P5b criterion 5, measured after the fact: the PFC race in Python vs a Python fix vs Rust (2026-10-24)

Session B (`rust-lead-1024`). Referenced by `p5b-retroactive-record.md` (section 2, criterion 5). The script ran once,
best of 5 repeats, on `claude/p5b-gate-1024` (native/ unchanged since `20e1e742`; `_native` .pyd sha256 prefix
`f2108fbfba31a55a`), `.venv-lake` CPython 3.14.6t, quiet gate passed (system CPU 12.9% after a 159 s wait; one
co-tenant python process at ~1 core, disclosed). It reads git (`fca70e3c^`, the deleted Python body) and
`native/sndl-core/tests/fixtures_pfc.json` (3 chains, 460 sequential calls); it writes nothing. Kept here as text, not as
a tracked `.py` (no Python file lives under `specs/`); copy it out to re-run it with `SNDL_REPO` set to a checkout.

## Output

```text
A {'parity_chains': '3/3', 'per_call_us_by_chain': [1197.6, 151.9, 691.3], 'median_us': 691.3}
B {'parity_chains': '3/3', 'per_call_us_by_chain': [1339.1, 116.1, 702.8], 'median_us': 702.8}
N {'parity_chains': '3/3', 'per_call_us_by_chain': [81.4, 38.2, 59.0], 'median_us': 59.0}
A: 691.3 us/call vs bar 120.0 us -> INSUFFICIENT (5.8x the bar); native 59.0 us/call
B: 702.8 us/call vs bar 120.0 us -> INSUFFICIENT (5.9x the bar); native 59.0 us/call
RESULT {"A": {"parity_chains": "3/3", "per_call_us_by_chain": [1197.6, 151.9, 691.3], "median_us": 691.3}, "B": {"parity_chains": "3/3", "per_call_us_by_chain": [1339.1, 116.1, 702.8], "median_us": 702.8}, "N": {"parity_chains": "3/3", "per_call_us_by_chain": [81.4, 38.2, 59.0], "median_us": 59.0}}
```

## Source

```python
"""Retroactive P5b criterion 5 for the PFC port: could a Python fix have met Phase 2's bar?

Phase 2's bar (Phase 0 report section 4): 25 calls per wake tick <= 3 ms, i.e. <= 120 us per call.
Variants, each judged first against the three fixture CHAINS of native/sndl-core/tests/fixtures_pfc.json
(460 sequential calls; every (winner, rt) AND the chain's end rng state must match; a variant that misses is
disqualified and its timing is not reported as a fix). The 5 'full' cases are NOT used: all are zero-evidence,
no-draw calls (review rv-p5b-b #5001), so they would time the trivial path.
  A  the deleted Python body (git fca70e3c^), unchanged
  B  A with the cheapest fix: `_argmax` without the per-element lambda (`v.index(max(v))`: same first-max tie rule)
  N  the shipped Rust kernel through `_native.pfc_decide` (getstate -> race -> setstate, as the method does)
Timing: per variant and chain, the whole chain K times (state reset each time), best repeat, per-call us.
ponytail: scratch instrument, not a repo module. Reads git and the fixture; writes nothing.
"""

import ast
import json
import os
import random
import statistics
import subprocess
import sys
import time

REPO = os.environ.get("SNDL_REPO", "D:/AI-project/lanes/rust-lead-1024/measure-wt")
sys.path.insert(0, os.path.join(REPO, "src"))
K = 5
BAR_US = 120.0
NAMES = {"n_dec": "_n_dec", "n_inh": "_n_inh", "t_max": "_t_max", "md": "_md", "g_ext": "_g_ext", "b0": "_b0",
         "w_plus": "_w_plus", "w_inh": "_w_inh", "w_ie": "_w_ie", "w_cross": "_w_cross",
         "trace_decay": "_trace_decay", "theta": "_theta", "noise_amp": "_noise_amp", "init_v": "_init_v",
         "dec_bias": "_dec_bias"}

src = subprocess.run(
    ["git", "-C", REPO, "show", "fca70e3c^:src/snn_data_lake/faculties/prefrontal_executive.py"],
    capture_output=True, text=True, check=True,
).stdout
tree = ast.parse(src)
body = argmax = None
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_argmax":
        argmax = ast.get_source_segment(src, node)
    if isinstance(node, ast.ClassDef):
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "_decide_spiking":
                body = ast.get_source_segment(src, item)
assert body and argmax, "could not extract the deleted body"
body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.splitlines())
ns_a: dict = {}
exec(argmax + "\n" + body, ns_a)
ns_b: dict = {}
exec("def _argmax(values):\n    return values.index(max(values))\n" + body, ns_b)

from snn_data_lake import _native  # noqa: E402

chains = json.load(open(os.path.join(REPO, "native/sndl-core/tests/fixtures_pfc.json"), encoding="utf-8"))["chains"]


class Fake:
    pass


def make(ch):
    o = Fake()
    for k, v in ch["params"].items():
        setattr(o, NAMES[k], v)
    o.rng = random.Random()
    return o


def state(words):
    return (3, tuple(words), None)


def native(self, evidence):
    version, internal, gauss = self.rng.getstate()
    w, rt, internal = _native.pfc_decide(
        evidence,
        (self._n_dec, self._n_inh, self._t_max),
        (self._md, self._g_ext, self._b0, self._w_plus, self._w_inh, self._w_ie, self._w_cross,
         self._trace_decay, self._theta, self._noise_amp, self._init_v),
        self._dec_bias,
        internal,
    )
    self.rng.setstate((version, internal, gauss))
    return w, rt


def run_chain(fn, o, ch):
    o.rng.setstate(state(ch["state0"]))
    return [list(fn(o, list(c["e"]))) for c in ch["calls"]]


res = {}
for name, fn in {"A": ns_a["_decide_spiking"], "B": ns_b["_decide_spiking"], "N": native}.items():
    ok, per_call = 0, []
    for ch in chains:
        o = make(ch)
        outs = run_chain(fn, o, ch)
        ok += outs == [c["out"] for c in ch["calls"]] and o.rng.getstate() == state(ch["state_end"])
        best = float("inf")
        for _ in range(K):
            t = time.perf_counter()
            run_chain(fn, o, ch)
            best = min(best, (time.perf_counter() - t) / len(ch["calls"]) * 1e6)
        per_call.append(best)
    res[name] = {"parity_chains": f"{ok}/{len(chains)}", "per_call_us_by_chain": [round(x, 1) for x in per_call],
                 "median_us": round(statistics.median(per_call), 1)}
    print(name, res[name], flush=True)
for name in ("A", "B"):
    m = res[name]["median_us"]
    print(f"{name}: {m} us/call vs bar {BAR_US} us -> {'MEETS' if m <= BAR_US else 'INSUFFICIENT'} "
          f"({m / BAR_US:.1f}x the bar); native {res['N']['median_us']} us/call", flush=True)
print("RESULT " + json.dumps(res))
```
