"""
v0.32 -- real-time signal loop: Arduino signal -> spike encoding ->
        find matching signature -> emit to Interpreter.

This is the "my part" of the robot-arm processing loop:

  Arduino signal  ->  [encode]  ->  [match]  ->  Interpreter
   (windowed          encode_latency  Van Rossum    JSON line
    samples)          (deterministic) nearest ref   on stdout

It owns these boxes and nothing else: it does NOT drive servos, run the
Interpreter, or flash the Arduino. Input = signal windows; output = a matched
signature label the Interpreter maps to a robot command.

WHERE THIS MODULE'S STATE LIVES (P4b, corrected 2026-08-05). It used to declare THREE stores at
module scope, all `./data/...` -- CWD-relative, so none of them followed NORD_HOME and the brain's
learned gestures depended on which directory the daemon was started from:

  * `./data/signatures.json` -- the LEARNED signature library, the gestures this loop DISCOVERED and
    promoted (`learn_unknowns` -> `save_library`). Measured 2026-08-04 with NORD_HOME and CWD in
    different empty dirs: learning promoted DISCOVERED_1 into a 10,406-byte sidecar and added ZERO
    records to `brain_lake.bin`, which was never even created. It is now a `_T_STATE` record IN the
    one lake, named "signatures.json", via `substrate.home.save_state`.
  * `./data/lake.spc` -- a SECOND GZIP LAKE this module's own docstring called "the data lake".
    RETIRED (P2: one store survives, and it is `brain_lake.bin`). See `flush_lake`'s marker.
  * `./data/unknowns.jsonl` -- the novel-signal queue. Moved under `home()`. It stays a FILE and the
    reason is stated at `unknowns_path()`: it is not learned state, it is the raw input queue that
    `learn_unknowns` CONSUMES and truncates; what is learned from it goes into the lake.

ORPHANED, DISCLOSED RATHER THAN DELETED OR SILENTLY IMPORTED. Two files under the repo's gitignored
`./data/` are no longer read: `lake.spc` (2,279 B, the retired second lake -- it had no reader
before this either) and `unknowns.jsonl` (2,717 B of queued novel windows). No `./data/
signatures.json` exists, so no LEARNED library is stranded -- checked, not assumed. A CWD-relative
one-time import is deliberately NOT added: reading from the CWD is the exact fragility this removes,
and it would only ever fire for a daemon started in the repo root. Anyone who wants those queued
unknowns copies the file to `<home>/unknowns.jsonl`; it is one command and no code.

Matching: each incoming window is encoded and compared (Van Rossum distance) to a
library of enrolled REFERENCE signatures (1 per command). Nearest wins; a Lowe-style
ratio test rejects ambiguous/novel signals (best must clearly beat 2nd-best) so the
loop stays silent rather than firing a wrong command.

Matcher: the HYBRID (learned spiking classifier + Van Rossum novelty gate) is the
default -- strongest under noise (see learned_matcher.py). `--fast` selects the
zero-startup template baseline.

  python signal_loop.py                       # simulated + hybrid + self-check
  python signal_loop.py --fast                # template baseline (no training)
  python signal_loop.py --serial COM3         # real Arduino (needs `pip install pyserial`)
  python signal_loop.py --serial COM3 --window 8   # buffer 8 raw lines -> 1 window
  cat windows.csv | python signal_loop.py --stdin  # exercise the wire contract, no hardware
  python signal_loop.py --serial COM3 --reflex     # instinctive STOP/WITHDRAW on danger channels
  python signal_loop.py --serial COM3 --reflex --feedback   # full closed loop (below)
  python signal_loop.py --enroll ENGAGE     # record a reference into the one lake

Feedback loop (--feedback): the Interpreter sends outcomes BACK into the loop's input
as `OUTCOME <reward>` lines (reward in [-1,+1], good=+, bad=-), applied to the last acted
signal. These drive dopamine learning (RPE) and a cortisol stress level; cortisol then
modulates the reflex threshold (hypervigilance) and the matcher's caution bias LIVE. Mix
signal lines and OUTCOME lines on the same input stream.

Continual learning: non-matching (novel) signals are recorded to <home>/unknowns.jsonl.
A signal that recurs >= MIN_SUPPORT times can be clustered and promoted into a new
signature -- the loop grows its own vocabulary instead of silently dropping unknowns.
  python signal_loop.py --learn                    # cluster unknowns -> new DISCOVERED_n signatures
  python signal_loop.py --learn-as WAVE            # name the largest discovered cluster 'WAVE'

Arduino wire contract (full spec + example sketch in docs/arduino_contract.md):
  115200 8N1, newline-terminated lines of comma/space-separated floats. Direct mode
  (default) = one N-feature window per line; windowed mode (--window W) = W lines of
  C raw channel samples -> one window. Values any range (auto-normalized); '#' and
  unparseable lines skipped.
"""

import json
import math
import os
import random
import sys
import zlib

from snn_data_lake.basal_ganglia.valence_stdp import (
    ValenceLearner,  # #3 RPE dopamine (leaf: only imports spike_preprocessing)
)
from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.reflex import (  # #2 instinctive fast-path (no cycle: reflex is leaf)
    Reflex,
    reflex_guard,
)
from snn_data_lake.cortex.spike_preprocessing import N, encode_latency, van_rossum_distance
from snn_data_lake.neuromodulation.cortisol import (
    Cortisol,  # slow tonic stress (leaf: only os/math)
)

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "build_learned_matcher": (
        "NOT_A_RULE",
        "BUILDS the matcher: a supervised spiking classifier plus a Van Rossum distance veto. A "
        "factory; the training it wraps lives in the classifier",
    ),
    "learn_unknowns": (
        "NOT_A_RULE",
        "clusters recorded unknowns and promotes recurring ones into the signature LIBRARY. It grows "
        "a symbolic library, touching no weight",
    ),
}

# SIG_PATH / LAKE_PATH / UNK_PATH -- three `./data/...` module constants, RETIRED 2026-08-05. They
# were CWD-relative, so the brain's learned gestures lived wherever the daemon happened to be
# started from rather than under NORD_HOME. Successors: the learned library is a `_T_STATE` record
# in the one lake under SIG_STATE; the second gzip lake is gone (see `flush_lake`); the unknowns
# queue is `unknowns_path()`. Nothing outside this module ever read the three names (measured: a
# tree-wide grep for SIG_PATH / LAKE_PATH / UNK_PATH and for save_library / load_library /
# flush_lake / learn_unknowns / record_unknown_line found call sites in this file only).
SIG_STATE = "signatures.json"  # the record NAME in brain_lake.bin, and the sidecar it migrates from
RATIO = 0.85  # match only if best < RATIO*second
NOISE = 0.08  # simulated sensor jitter
MIN_SUPPORT = 3  # repeats before a novel -> new signature

# Default command library (the Interpreter maps these labels -> a concrete command string).
# Derived, so a joint added to the arm becomes a learnable signature here without an edit.
COMMANDS = [*arm_config.TOKENS, "ENGAGE", "RELEASE", "HOME"]


# ---- signal source ----------------------------------------------------------
def normalize(vec):
    """Min-max a raw sample window into [0,1]^N (pad/truncate to N features)."""
    v = list(vec)[:N] + [0.0] * max(0, N - len(vec))
    lo, hi = min(v), max(v)
    rng = hi - lo
    return [(x - lo) / rng for x in v] if rng > 1e-9 else [0.0] * N


def _proto(label, rng):
    """Deterministic reference vector for a command label. crc32 (not hash()) so it is
    stable across processes -- a saved signatures.json must reproduce next run."""
    r = random.Random(zlib.crc32(label.encode()))
    return [r.random() if r.random() < 0.45 else 0.0 for _ in range(N)]


def simulated_stream(library, n_events, seed=0, novel_every=7):
    """Yield (true_label_or_None, window): noisy copies of enrolled refs, plus the
    occasional novel signal (true_label=None) to exercise rejection."""
    rng = random.Random(seed)
    labels = list(library)
    for k in range(n_events):
        if novel_every and k % novel_every == novel_every - 1:
            yield None, [rng.random() for _ in range(N)]  # novel/unknown
        else:
            lab = labels[k % len(labels)]
            base = library[lab]
            yield lab, [min(1.0, max(0.0, v + rng.uniform(-NOISE, NOISE))) for v in base]


# ---- Arduino wire contract (see docs/arduino_contract.md) -------------------
# A line = comma- OR space-separated floats, newline-terminated, 115200 8N1.
#   direct  (window=1): each line is one N-feature window.
#   windowed(window=W): each line is C raw channel samples; W lines -> one window
#                        (flattened row-major, resized to N).
# Values may be any range (auto min-max normalized per window). '#' lines + lines
# that don't parse are skipped, never crash the loop.
def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        vals = [float(x) for x in line.replace(",", " ").split()]
    except ValueError:
        return None
    # nan/inf ARE valid Python floats but not valid samples: they make every downstream danger
    # comparison (reflex, distance < CONTACT) False -> fail OPEN. Reject the whole line. (audit P1)
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals


def resize_to_n(vec, n=N):
    """Truncate or zero-pad a raw vector to exactly N features."""
    return vec[:n] if len(vec) >= n else vec + [0.0] * (n - len(vec))


def line_windows(raw_rows, window=1, n=N):
    """Turn parsed float rows into normalized N-feature windows."""
    if window <= 1:
        for row in raw_rows:
            yield None, normalize(resize_to_n(row, n))
    else:
        buf = []
        for row in raw_rows:
            buf.append(row)
            if len(buf) == window:
                flat = [v for r in buf for v in r]
                yield None, normalize(resize_to_n(flat, n))
                buf = []


def serial_stream(port, baud, window=1, reflex=None, on_reflex=None):
    """Read float windows from a serial port (real Arduino). Guarded import.
    If a Reflex is given, it monitors the RAW sample stream ahead of windowing."""
    import serial  # pyserial -- only needed on hardware

    def rows():
        ser = serial.Serial(port, baud, timeout=1)
        while True:
            p = parse_line(ser.readline().decode("ascii", "ignore"))
            if p is not None:
                yield p

    src = reflex_guard(rows(), reflex, on_reflex) if reflex else rows()
    yield from line_windows(src, window)


def stdin_stream(window=1, reflex=None, on_reflex=None):
    """Same wire contract, read from stdin -- exercise the loop with no hardware."""
    raw = (p for line in sys.stdin if (p := parse_line(line)) is not None)
    src = reflex_guard(raw, reflex, on_reflex) if reflex else raw
    yield from line_windows(src, window)


# ---- signature library (IN the one durable lake) ----------------------------
def unknowns_path():
    """`<home>/unknowns.jsonl` -- the novel-signal queue, under home() so it follows NORD_HOME.

    IT STAYS A FILE, and P4b is why rather than in spite of: this is not what the brain has learned,
    it is the raw input queue that `learn_unknowns` reads once, clusters, and TRUNCATES. The learned
    product -- the promoted signature -- goes into `brain_lake.bin`. Putting an append-per-event
    queue in an append-only lake that `consolidate()` preserves forever would store every rejected
    sensor window for the life of the brain in order to keep the handful that become gestures."""
    from snn_data_lake.substrate.home import state_path

    return state_path("unknowns.jsonl")


def load_library():
    """The enrolled + LEARNED signature library, from THE ONE LAKE (falling back to the defaults).

    `home.load_state` reads a `_T_STATE` record named SIG_STATE, importing a pre-P4b
    `<home>/signatures.json` once if one is there. A library that was never saved reads back as
    None, which is a cold start, which is the default library -- the same behaviour the missing
    `./data/signatures.json` produced.

    TWO EDGES, both measured 2026-08-05 rather than reasoned about, because this function moved
    from a CWD file read to a lake read and neither survived the move unchanged:

    * AN EMPTY SAVED LIBRARY NOW COMES BACK AS THE DEFAULTS. `saved` is tested for truth, so a
      stored `{}` falls through to `build_default_library()`; the pre-P4b `json.load` returned the
      empty dict, and `build_matcher({})` would then have divided by an empty best/second pair.
      Falling back is the safer of the two and is why the test is truth rather than `is None` -- but
      it IS a behaviour change and no live path can produce an empty library anyway (`--enroll` and
      `--learn` both write a superset of the defaults).
    * IT READS WHATEVER HOME IS SET WHEN IT IS CALLED, and `main()` calls it on its first line --
      BEFORE the default branch scopes an isolated NORD_HOME. So `python -m
      snn_data_lake.brainstem.signal_loop` with no NORD_HOME set opens the operator's real
      `brain_lake.bin` (519,646,815 B on 2026-08-05) and replays it once, which is a quarter to
      four tenths of a second plus the BinLog construction. Not a correctness bug and not a write:
      `run_selfchecks.py` sets NORD_HOME for the whole gate run, and no `<home>/signatures.json`
      exists to migrate (checked). Left as-is deliberately: hoisting the isolation above this line
      means deciding "is this the default branch" from a LIST of mode flags kept in step by hand,
      and a flag that fell off that list would run `--enroll` against a temp home and silently
      discard an enrolled signature. A slow manual run is the cheaper failure."""
    from snn_data_lake.substrate.home import load_state

    saved = load_state(SIG_STATE)
    return dict(saved) if saved else build_default_library()


def build_default_library():
    return {c: _proto(c, None) for c in COMMANDS}


def save_library(lib):
    """Persist the signature library INTO the one lake. True if written, False if unchanged."""
    from snn_data_lake.substrate.home import save_state

    return save_state(SIG_STATE, lib)


# ---- match: encoded live signal -> nearest enrolled reference ---------------
def build_matcher(library):
    """Baseline: nearest enrolled template (Van Rossum) + Lowe ratio reject."""
    refs = [(lab, encode_latency(vec)) for lab, vec in library.items()]

    def match(window):
        q = encode_latency(window)
        scored = sorted(((van_rossum_distance(q, ev), lab) for lab, ev in refs))
        best_d, best_lab = scored[0]
        second_d = scored[1][0] if len(scored) > 1 else math.inf
        matched = best_d < RATIO * second_d  # ratio (Lowe) test
        return (best_lab if matched else None), best_d, second_d, q

    return match


def build_learned_matcher(library):
    """Stronger: supervised spiking classifier picks the command, Van Rossum
    distance gate vetoes novel signals (hybrid -- see learned_matcher.py). Trains
    at build time. Returns the same (label, dist, ref, encoded) shape as build_matcher."""
    from snn_data_lake import (
        learned_matcher as L,  # lazy: avoids the signal_loop<->learned_matcher cycle
    )

    labels = list(library)
    nC = len(labels)
    refs_enc = [encode_latency(v) for v in library.values()]
    W, _ = L.train_learned(library)
    gate = L.calibrate_gate(library, refs_enc)

    def match(window):
        # v2.4 perf: hybrid_match now returns the nearest distance AND the encoded window
        # it already computed, so we reuse them instead of recomputing nearest_dist +
        # encode_latency (was ~3.16 -> ~1.68 ms/window). Decision + returned tuple identical.
        idx, dist, q = L.hybrid_match(W, nC, refs_enc, gate, window)
        label = labels[idx] if idx is not None else None
        return label, dist, gate, q

    return match


# ---- continual learning: record non-matching signals, cluster, promote ------
def record_unknown_line(seq, dist, window, path=None):
    """Append one novel/rejected window to the unknowns store for later learning.

    `path` defaults to `unknowns_path()` at CALL time, not at import time: the former module-level
    default froze the location into the function signature, so a caller that set NORD_HOME after
    import still wrote to wherever the constant had pointed."""
    path = unknowns_path() if path is None else path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps({"seq": seq, "dist": round(dist, 4), "vec": [round(v, 4) for v in window]})
            + "\n"
        )


def _mean(vecs):
    n = len(vecs)
    return [sum(col) / n for col in zip(*vecs)]


def _auto_radius(library):
    """Typical same-gesture Van Rossum distance (ref vs a jittered copy) x1.5 --
    the cluster radius scales with the data, not a magic constant.

    (This line said "x2.5" until 2026-08-05 while the code returned `* 1.5` and the inline comment
    three lines down said x1.5: a 1.67x disagreement inside one function, on the constant that
    decides whether a recurring novel signal is promoted into a new learned signature. The CODE was
    right and is unchanged; only this summary was wrong.)"""
    rng = random.Random(0)
    ds = []
    for v in library.values():
        a = encode_latency(v)
        b = encode_latency([min(1.0, max(0.0, x + rng.uniform(-NOISE, NOISE))) for x in v])
        ds.append(van_rossum_distance(a, b))
    # x1.5: room for within-gesture spread, but below the inter-gesture gap (~x1.85)
    # so unrelated noise stays in its own singleton clusters, not absorbed.
    return (sum(ds) / len(ds)) * 1.5 if ds else 4.0


def cluster_unknowns(rows, library, min_support=MIN_SUPPORT, radius=None):
    """Greedy single-link cluster of recorded unknown windows (Van Rossum distance).
    A signal that recurs >= min_support times is a candidate new gesture.
    Returns promoted clusters [{vec: mean, size: k}] sorted by size desc."""
    radius = _auto_radius(library) if radius is None else radius
    clusters = []  # {members:[vec], enc: encoded mean}
    for r in rows:
        v = r["vec"]
        e = encode_latency(v)
        best, bd = None, float("inf")
        for c in clusters:
            d = van_rossum_distance(e, c["enc"])
            if d < bd:
                bd, best = d, c
        if best is not None and bd <= radius:
            best["members"].append(v)
            best["enc"] = encode_latency(_mean(best["members"]))
        else:
            clusters.append({"members": [v], "enc": e})
    promoted = [
        {"vec": _mean(c["members"]), "size": len(c["members"])}
        for c in clusters
        if len(c["members"]) >= min_support
    ]
    promoted.sort(key=lambda p: -p["size"])
    return promoted, len(clusters), radius


def learn_unknowns(library, persist=True, label=None):
    """Cluster the recorded unknowns; promote recurring ones into the signature
    library (the 'learned' step). Optional `label` names the largest new cluster."""
    unk = unknowns_path()
    if not os.path.exists(unk):
        return {}, 0, 0
    with open(unk, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    promoted, n_clusters, _ = cluster_unknowns(rows, library)
    base = sum(1 for k in library if k.startswith("DISCOVERED_"))
    added = {}
    for i, p in enumerate(promoted):
        name = label if (label and i == 0) else f"DISCOVERED_{base + i + 1}"
        library[name] = p["vec"]
        added[name] = p["size"]
    if persist and added:
        save_library(library)
        with open(unk, "w", encoding="utf-8"):
            pass  # consumed -> avoid re-discovering
    return added, len(rows), n_clusters


# `flush_lake(events)` -> `./data/lake.spc` RETIRED 2026-08-05 (P4b + P2). It gzipped the session's
# encoded windows into a SECOND durable store beside `brain_lake.bin`, and the module docstring
# called that file "the data lake". P2 says exactly one mechanism survives and the survivor is
# `brain_lake.bin`. Two facts made this a retirement rather than a port: its return value had
# exactly ONE consumer tree-wide, `main()`'s own reporting line and the `assert gz < raw` beside it;
# and the lake-side equivalent is `_T_SPIKE`, which `substrate/home.py` RETIRED on 2026-07-29 for
# the reason that applies here verbatim -- there is no live producer of AER exemplars worth keeping
# across a restart, so there is nothing to store. The compression assertion went with it rather than
# being kept pointing at a store that no longer exists. Reinstating it means writing _T_SPIKE
# records into the one lake, not recreating a .spc file.


# ---- the loop ---------------------------------------------------------------
def run(source, library, matcher=None, emit=True, on_unknown=None):
    """Drive `source` through match -> emit. Returns `(stats, encoded)`.

    `encoded` is every window this call encoded, in order. IT IS UNBOUNDED, and on the live
    `--serial` path -- which never returns -- it grows for the life of the process. That is a real
    hazard and it is stated rather than removed, because removing it is a TWO-FILE edit and the
    other file is not this one:

        substrate/sensor_csv_ingest.py  `stats, lake = S.run(src, lib, emit=False, on_unknown=...)`

    is a FOURTH call site outside this module (the other three are `main()`'s `--serial` and
    `--stdin` branches, which discard it, and `_selfcheck`) and it prints `len(lake)`. The
    2026-08-05 P4b change dropped the second return value on the belief that
    `flush_lake` was its only consumer; MEASURED afterwards, `python -m
    snn_data_lake.substrate.sensor_csv_ingest` exited 1 with `ValueError: too many values to unpack
    (expected 2)`, and that module is a CORE entry in run_selfchecks.py -- so the gate went red. The
    two-value contract is restored here rather than papered over with an empty list, which would
    have made that module print a silently wrong window count.

    TO CLOSE THE UNBOUNDED GROWTH (one commit, both files): `sensor_csv_ingest` wants a COUNT, and
    the count it wants is already `stats["events"]` -- change its line to `stats = S.run(...)` and
    its report to `stats['events']`, then drop `encoded` here. Do not do half of that."""
    match = matcher or build_matcher(library)
    encoded = []
    stats = {"events": 0, "matched": 0, "correct": 0, "rejected": 0, "labeled": 0}
    for true_lab, window in source:
        label, d, _d2, q = match(window)
        encoded.append(q)
        stats["events"] += 1
        if label is not None:
            stats["matched"] += 1
            if true_lab is not None and label == true_lab:
                stats["correct"] += 1
        else:
            stats["rejected"] += 1
            if on_unknown:  # record the non-matching signal to learn later
                on_unknown(stats["events"], d, window)
        if true_lab is not None:
            stats["labeled"] += 1
        if emit:  # JSON line -> Interpreter reads stdin
            print(
                json.dumps(
                    {
                        "t": stats["events"],
                        "match": label,
                        "dist": round(d, 3),
                        "confident": label is not None,
                    }
                )
            )
            sys.stdout.flush()
    return stats, encoded


# ---- outcome feedback channel (Interpreter -> loop) -------------------------
def parse_outcome(line):
    """Interpreter feedback line: 'OUTCOME <reward>' (reward in [-1,+1], good=+, bad=-),
    applied to the most recent acted signal. Returns the reward, or None if not an outcome."""
    s = line.strip()
    if not s.upper().startswith("OUTCOME"):
        return None
    try:
        return max(-1.0, min(1.0, float(s.split()[1])))
    except (IndexError, ValueError):
        return None


def serial_lines(port, baud):
    """Raw text lines from a serial port (for the feedback loop). Guarded import."""
    import serial

    ser = serial.Serial(port, baud, timeout=1)
    while True:
        line = ser.readline().decode("ascii", "ignore")
        if line:
            yield line


def run_live(
    lines,
    library,
    matcher,
    reflex,
    valence,
    cortisol,
    base_threshold=0.10,
    emit=None,
    on_unknown=None,
):
    """Closed loop WITH outcome feedback. Each input line is either a signal window
    (CSV/space floats) or an Interpreter outcome ('OUTCOME <r>'). Live coupling:
      - cortisol scales the reflex threshold (hypervigilance) and the matcher's caution bias
      - a reflex firing is itself an aversive stressor
      - each OUTCOME drives dopamine learning (RPE) on the last acted window and moves stress
      - quiet signal ticks let cortisol decay (recovery)."""
    if emit is None:

        def emit(d):
            print(json.dumps(d))
            sys.stdout.flush()

    last_window = None
    seq = 0
    for line in lines:
        r = parse_outcome(line)
        if r is not None:  # ---- Interpreter feedback ----
            if last_window is not None:
                delta = valence.learn(last_window, r, lr_scale=cortisol.learn_rate_scale(r))
                stress = cortisol.step(max(0.0, -r))  # bad outcome raises stress
                emit({"outcome": r, "dopamine": round(delta, 3), "stress": round(stress, 3)})
            continue
        p = parse_line(line)
        if p is None:
            continue
        seq += 1
        if reflex is not None:  # ---- reflex (cortisol-sharpened) ----
            reflex.threshold = base_threshold * cortisol.reflex_threshold_scale()
            act = reflex.step(p)
            if act:
                stress = cortisol.step(1.0)  # reflex firing is itself stressful
                emit({"t": seq, "reflex": act, "preempt": True, "stress": round(stress, 3)})
                continue
        window = normalize(resize_to_n(p))  # ---- recognize + valence overlay ----
        label, d, _, _ = matcher(window)
        v_action, v = valence.act(window, bias=cortisol.caution_bias())
        last_window = window
        if label is None and on_unknown:
            on_unknown(seq, d, window)
        stress = cortisol.step(0.0)  # quiet tick -> recovery
        emit(
            {
                "t": seq,
                "match": label,
                "dist": round(d, 3),
                "instinct": v_action,
                "valence": round(v, 3),
                "stress": round(stress, 3),
            }
        )


def main():
    args = sys.argv[1:]
    library = load_library()

    if "--enroll" in args:
        label = args[args.index("--enroll") + 1]
        # record one window from the source as this command's reference
        src = simulated_stream({label: _proto(label, None)}, 1, seed=1)
        _, window = next(src)
        library[label] = normalize(window) if max(window) > 1.0 else window
        save_library(library)
        from snn_data_lake.substrate.home import knowledge_path

        print(f"enrolled '{label}' -> {knowledge_path()} ({len(library)} signatures)")
        return

    if "--learn" in args or "--learn-as" in args:
        # cluster recorded non-matching signals; promote recurring ones to signatures
        label = args[args.index("--learn-as") + 1] if "--learn-as" in args else None
        added, n_rows, n_clusters = learn_unknowns(library, persist=True, label=label)
        if added:
            print(
                f"learned {len(added)} signature(s) from {n_rows} unknowns "
                f"({n_clusters} clusters): " + ", ".join(f"{k} (x{v})" for k, v in added.items())
            )
            print(f"library now {len(library)}: {list(library)}")
        else:
            print(
                f"nothing learned: {n_rows} unknowns in {n_clusters} clusters "
                f"(need >= {MIN_SUPPORT} similar repeats). collect more, then --learn"
            )
        return

    # matcher: hybrid (learned + novelty gate) is the DEFAULT; --fast = template
    fast = "--fast" in args
    if not fast:
        sys.stderr.write("training learned hybrid matcher (snn_classifier + gate)...\n")
    matcher = None if fast else build_learned_matcher(library)
    matcher_name = "template (Van Rossum)" if fast else "hybrid (learned + novelty gate)"
    window = int(args[args.index("--window") + 1]) if "--window" in args else 1

    # #2 reflex fast-path: instinctive STOP/WITHDRAW on danger channels, ahead of match
    reflex = Reflex() if "--reflex" in args else None

    def emit_reflex(action, seq):  # preemptive instinct -> Interpreter
        print(json.dumps({"reflex": action, "t_sample": seq, "preempt": True}))
        sys.stdout.flush()

    if reflex:
        sys.stderr.write(f"reflex fast-path ON: {reflex.rules}\n")

    # --feedback: full closed loop with dopamine learning + cortisol stress modulation
    if "--feedback" in args:
        valence = ValenceLearner()
        cortisol = Cortisol()
        m = matcher or build_matcher(library)
        sys.stderr.write(
            "feedback loop ON: 'OUTCOME <r>' lines drive dopamine; "
            "cortisol modulates reflex + caution live\n"
        )
        if "--serial" in args:
            port = args[args.index("--serial") + 1]
            baud = int(args[args.index("--baud") + 1]) if "--baud" in args else 115200
            lines = serial_lines(port, baud)
        else:
            lines = sys.stdin
        run_live(lines, library, m, reflex, valence, cortisol, on_unknown=record_unknown_line)
        return

    if "--serial" in args:
        port = args[args.index("--serial") + 1]
        baud = int(args[args.index("--baud") + 1]) if "--baud" in args else 115200
        sys.stderr.write(
            f"reading {port}@{baud} window={window}, {len(library)} "
            f"signatures, matcher={matcher_name}\n"
        )
        run(
            serial_stream(port, baud, window, reflex, emit_reflex),
            library,
            matcher,
            on_unknown=record_unknown_line,
        )  # until interrupted
        return

    if "--stdin" in args:
        sys.stderr.write(f"reading stdin window={window}, matcher={matcher_name}\n")
        run(
            stdin_stream(window, reflex, emit_reflex),
            library,
            matcher,
            on_unknown=record_unknown_line,
        )  # EOF
        return

    # default: simulated stream + report + self-check.
    # ISOLATED HOME, and it is not optional now that the stores follow NORD_HOME: this branch WRITES
    # (the unknowns queue, and the learned-library round-trip below), so without it the self-check
    # would append to the operator's live `~/.snn_data_lake` on every gate run. NORD_HOME is RESTORED
    # rather than popped -- tests/test_selfchecks.py runs every module's __main__ in ONE interpreter,
    # and a bare pop un-isolates every module after this one (home.py records that exact incident).
    import shutil
    import tempfile

    _prior_home = os.environ.get("NORD_HOME")
    _work = tempfile.mkdtemp(prefix="signal_loop_home_")
    os.environ["NORD_HOME"] = _work
    try:
        _selfcheck(library, matcher, matcher_name)
    finally:
        if _prior_home is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = _prior_home
        shutil.rmtree(_work, ignore_errors=True)


def _selfcheck(library, matcher, matcher_name):
    """The default-branch demo + asserts, split out so main() can scope NORD_HOME around it."""
    unk = unknowns_path()
    if os.path.exists(unk):
        with open(unk, "w", encoding="utf-8"):
            pass  # fresh unknowns bin for the demo
    src = simulated_stream(library, 40, seed=0)
    stats, encoded = run(src, library, matcher, emit=True, on_unknown=record_unknown_line)
    acc = stats["correct"] / max(1, stats["labeled"])

    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("SIGNAL LOOP  (encode -> match -> Interpreter)\n")
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write(f"matcher      : {matcher_name}\n")
    sys.stderr.write(f"library      : {len(library)} signatures {list(library)}\n")
    sys.stderr.write(
        f"events       : {stats['events']} ({stats['labeled']} known, "
        f"{stats['events'] - stats['labeled']} novel)\n"
    )
    sys.stderr.write(f"matched      : {stats['matched']}  rejected: {stats['rejected']}\n")
    sys.stderr.write(
        f"recorded     : {stats['rejected']} novel -> {unk} (run `--learn` to cluster + enroll)\n"
    )
    sys.stderr.write(f"accuracy     : {acc:.0%} on known signals (chance {1 / len(library):.0%})\n")
    sys.stderr.write("=" * 60 + "\n")

    # ---- self-checks --------------------------------------------------------
    assert stats["events"] == 40, "stream did not produce all events"
    # `run`'s second return value is one encoded window PER EVENT. This is the invariant that makes
    # the standing two-file cleanup in `run`'s docstring safe -- `sensor_csv_ingest` prints
    # `len(lake)`, and it can only switch to `stats['events']` if the two are the same number.
    # REACHABLE RED, constructed and run 2026-08-05: with `encoded.append(q)` deleted this fires
    # with `0 != 40`, and with it moved inside the `if label is not None` branch, `35 != 40`.
    assert len(encoded) == stats["events"], (
        f"run() must encode exactly one window per event: {len(encoded)} != {stats['events']}"
    )
    assert acc >= 0.85, f"known-signal matching too weak: {acc:.2f}"
    # novel signals must mostly be rejected, not fired as a command
    novel = stats["events"] - stats["labeled"]
    assert stats["rejected"] >= novel * 0.5, "loop fires commands on novel signals"

    # ---- P4b: the LEARNED library lives IN the one lake, proven in a SEPARATE PROCESS ---------
    # This is the store `learn_unknowns` promotes a discovered gesture into. Measured 2026-08-04,
    # before the fix: promoting DISCOVERED_1 wrote a 10,406-byte `./data/signatures.json` and added
    # ZERO records to brain_lake.bin, which was never created. An in-process reload would have
    # passed against that sidecar too, which is why the check below hands the lake to a child.
    import shutil
    import subprocess
    import tempfile

    from snn_data_lake.substrate.home import state_path

    taught = dict(library)
    taught["DISCOVERED_SELFCHECK"] = _proto("DISCOVERED_SELFCHECK", None)
    assert save_library(taught) is True, "the first save of a learned library must write a record"
    assert save_library(taught) is False, "an unchanged library must not be re-appended to the lake"
    assert not os.path.exists(state_path(SIG_STATE)), (
        "the learned library must go IN brain_lake.bin -- no signatures.json sidecar may be created"
    )
    # An `assert os.path.exists(knowledge_path())` stood here and was DELETED, not weakened (P8
    # F11). The two `save_library` asserts above can only pass if the lake machinery ran, and
    # constructing the BinLog is what creates the file -- so no input makes this one fire first.
    # MEASURED with a totally inert writer: the red landed on "an unchanged library must not be
    # re-appended", never here. `closed_loop.py` keeps the same check because there it IS reachable.
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    child = (
        f"import sys; sys.path.insert(0, {src_root!r})\n"
        f"import os; os.environ['NORD_HOME'] = {os.environ['NORD_HOME']!r}\n"
        "from snn_data_lake.brainstem.signal_loop import load_library\n"
        "lib = load_library()\n"
        "assert 'DISCOVERED_SELFCHECK' in lib, sorted(lib)\n"
        f"assert lib['DISCOVERED_SELFCHECK'] == {taught['DISCOVERED_SELFCHECK']!r}, 'vector changed'\n"
    )
    rc = subprocess.run([sys.executable, "-c", child]).returncode
    assert rc == 0, f"a learned signature did not survive into a SEPARATE process (rc={rc})"
    # MIGRATION, not deletion: a pre-P4b sidecar in a fresh home is imported and left on disk.
    legacy_home = tempfile.mkdtemp(prefix="signal_loop_legacy_")
    _outer_home = os.environ["NORD_HOME"]
    try:
        os.environ["NORD_HOME"] = legacy_home
        with open(os.path.join(legacy_home, SIG_STATE), "w", encoding="ascii") as f:
            json.dump({"LEGACY_GESTURE": [0.0] * N}, f)
        assert "LEGACY_GESTURE" in load_library(), "a pre-P4b signatures.json was not imported"
        assert os.path.exists(os.path.join(legacy_home, SIG_STATE)), (
            "the legacy sidecar must be left on disk, inert -- migrating is not deleting"
        )
    finally:
        os.environ["NORD_HOME"] = _outer_home
        shutil.rmtree(legacy_home, ignore_errors=True)
    sys.stderr.write(
        "P4b OK: the learned signature library is a _T_STATE record in brain_lake.bin, read back by "
        "a SEPARATE process; a pre-P4b signatures.json is imported once and left in place\n"
    )

    # ---- Arduino wire-contract self-check (no hardware needed) ---------------
    assert parse_line("# comment") is None and parse_line("   ") is None
    assert parse_line("0.1, 0.2 0.3") == [0.1, 0.2, 0.3], "CSV/space parse"
    assert parse_line("nope,nan?") is None, "malformed line must be skipped"
    assert parse_line("1, nan, 2") is None, (
        "a non-finite (nan) sample must skip the whole line (audit P1)"
    )
    assert parse_line("inf") is None and parse_line("1,-inf") is None, "inf samples must be skipped"
    assert len(resize_to_n([1.0] * 9)) == N and len(resize_to_n([1.0] * 999)) == N
    wins = list(line_windows([[float(i)] for i in range(3 * N)], window=N))
    assert len(wins) == 3 and all(len(w[1]) == N for w in wins), "windower shape wrong"
    sys.stderr.write(
        "self-check OK: known matched, novel rejected, library durable in the ONE lake, "
        "wire contract parses\n"
    )

    # ---- continual-learning self-check: novel recurring gesture -> learn -> match
    rng = random.Random(1)
    newg = _proto("NEW_GESTURE", None)  # a gesture NOT in the library
    pre = build_matcher(library)(newg)[0]  # unknown today -> should be rejected
    rows = [
        {"vec": [min(1.0, max(0.0, v + rng.uniform(-NOISE, NOISE))) for v in newg]}
        for _ in range(MIN_SUPPORT + 2)
    ]  # it recurs a few times (gets recorded)
    rows += [{"vec": [rng.random() for _ in range(N)]} for _ in range(4)]  # scattered noise
    promoted, n_clusters, _ = cluster_unknowns(rows, library)
    lib2 = dict(library)
    for i, p in enumerate(promoted):
        lib2[f"DISCOVERED_{i + 1}"] = p["vec"]  # learn them into a copy
    post = build_matcher(lib2)(newg)[0]  # same gesture now matches
    assert pre is None, "new gesture should start novel (rejected)"
    assert promoted, "recurring novel gesture was not discovered"
    assert post and post.startswith("DISCOVERED"), "learned gesture not matched after learning"
    sys.stderr.write(
        f"continual-learning OK: novel rejected -> recorded -> clustered "
        f"({len(promoted)} new, scattered noise ignored) -> now matches '{post}'\n"
    )

    # ---- live feedback self-check: outcomes drive dopamine + cortisol (no hardware) ------
    fb_lib = build_default_library()
    gv = _proto("FB_GOOD", None)
    gline = ",".join(f"{x:.4f}" for x in gv)
    gw = normalize(resize_to_n([float(x) for x in gline.split(",")]))
    vlA, ctA = ValenceLearner(), Cortisol(tau=20.0)
    fbA = [x for _ in range(30) for x in (gline, "OUTCOME 1.0")]  # signal then +reward x30
    run_live(iter(fbA), fb_lib, build_matcher(fb_lib), None, vlA, ctA, emit=lambda d: None)
    assert vlA.act(gw, bias=ctA.caution_bias())[0] == "APPROACH", "no APPROACH learned from +reward"
    assert ctA.level < 0.2, "good outcomes should not raise stress"
    vlB, ctB = ValenceLearner(), Cortisol(tau=20.0)
    fbB = [x for _ in range(20) for x in (gline, "OUTCOME -1.0")]  # signal then -reward x20
    run_live(iter(fbB), fb_lib, build_matcher(fb_lib), None, vlB, ctB, emit=lambda d: None)
    assert vlB.act(gw)[0] == "AVOID", "no AVOID learned from -reward"
    assert ctB.level > 0.4, "bad outcomes should raise cortisol stress"
    assert ctB.reflex_threshold_scale() < 1.0, "stress should sharpen the reflex"
    assert parse_outcome("OUTCOME 0.5") == 0.5 and parse_outcome("500,500") is None, "outcome parse"
    sys.stderr.write(
        "feedback self-check OK: outcomes drive dopamine (APPROACH/AVOID) + "
        "cortisol stress; stress sharpens the reflex live\n"
    )


if __name__ == "__main__":
    main()
