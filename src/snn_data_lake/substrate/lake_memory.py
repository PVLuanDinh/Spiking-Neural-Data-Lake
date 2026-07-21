r"""
lake_memory.py -- give the durable working-memory data lake real-neuron dynamics: PRUNE and
RECONSTRUCT instead of appending idle-tick noise forever.

THE PROBLEM this fixes. substrate/lake_working_memory.LakeWorkingMemory made the durable spike
lake (streaming_hub.StreamingHub) the brain's short-term store -- but it only ever APPENDS: every
idle tick is written forever (470k+ events, mostly idle-loop noise), and "recall" is a raw
windowed dump of the last few ticks. Real neurons do not hoard raw spikes. They (1) PRUNE weak /
redundant / idle synapses and homeostatically DOWNSCALE during sleep (SHY -- the Synaptic
Homeostasis Hypothesis, Tononi & Cirelli), so the consolidated store stays BOUNDED, and (2)
RECONSTRUCT a memory by pattern-COMPLETING a partial cue through a hippocampal attractor
(CA3 auto-association), not by replaying a raw log.

WHAT THIS ADDS. `MemoryLake` wraps the durable StreamingHub (kept LOSSLESS -- every event still
lands in the append-only `.spkl`, so n_events()/query/replay are unchanged) and layers on the two
faculties the raw lake lacked:

  * WRITE-WITH-SALIENCE. write(t, active, salience) / add(event, salience) records to the durable
    lake AND opens/updates an importance trace for that tick's assembly. Salient/novel assemblies
    get a strong trace; idle-loop repeats get a weak one (or, unlabelled, an auto-novelty score).

  * PRUNE (bounded, not infinite). consolidate() runs a SHY-style offline pass: a homeostatic
    global DOWNSCALE of every importance trace, an importance-weighted sharp-wave-ripple REPLAY
    that reinforces the significant traces, a metaplastic seep, then a hard EVICT of every trace
    left below threshold. The surviving consolidated store is CAPPED (a bound independent of how
    many events ever streamed): idle noise is pruned; distinctive events survive. A fair, bounded
    reservoir sample of the whole firehose is retained alongside so "pruned" is not "erased".

  * RECONSTRUCT (pattern completion). reconstruct(cue, k) treats the consolidated survivors as a
    content-addressable ATTRACTOR: a PARTIAL cue (a subset of an event's channels) completes to the
    full stored assembly, ranked by cue overlap then importance -- like hippocampal replay. Idle
    noise, never consolidated, does not complete to anything. This is the "reconstruct like real
    neurons" faculty, the same nearest-attractor completion principle sleep_replay.AttractorNet.
    classify uses, but sized to the number of consolidated memories rather than a fixed 24-neuron net.

REUSED, NOT REIMPLEMENTED (imported as libraries; none of them edited):
  * substrate/streaming_hub.StreamingHub          -- the durable, lossless, out-of-core append log
    (write -> append; recall/most_recent -> query_window; n_events; resume for restart recovery).
  * substrate/reservoir_sampling.Reservoir        -- Algorithm-R one-pass uniform sample that BOUNDS
    the retained handful of the unbounded stream (the idle firehose cannot grow the store).
  * hippocampus/experience_replay.ReplayBuffer    -- the SWR forward+reverse replay used at
    consolidation to reinforce the significant traces (importance-weighted by multiplicity).
  * plasticity/benna_fusi.BennaFusiSynapse        -- each importance trace is a multi-timescale
    beaker cascade: a salient, replayed memory seeps into the slow deep beakers and survives the
    downscale; a one-off idle trace stays labile in the fast beaker and is evicted. Metaplastic
    graded consolidation, per the task's "prune like real neurons".

DROP-IN for LakeWorkingMemory: same recall(window) shape (list of (t, frozenset(channels)), newest
last), same write(t, active_channels), n_events(), most_recent(lo, hi), reopen(), close(remove=).
So an existing caller (autonomous_loop's lake_wm, live_brain.recall) keeps working; consolidate()
and reconstruct() are the new, additive faculties.

DURABLE. The lossless events persist to a StreamingHub `.spkl` under NORD_HOME (home.lake_path()
style); the CONSOLIDATED store persists to a small ASCII JSON sidecar next to it, so the pruned
memory and its reconstruct capability survive a process restart (constructor resumes both).

HONEST FRAMING. The durable `.spkl` is still lossless-append by contract (this does NOT delete raw
events from disk); what is BOUNDED and pruned is the CONSOLIDATED working/long-term store the brain
recalls from -- the raw sensory log vs. the consolidated hippocampal trace. reconstruct is
overlap-based nearest-attractor completion (content-addressable), a faithful CA3 auto-association
at the algorithmic level, not a full recurrent-spiking energy descent. Deterministic: per-instance
random.Random, zlib.crc32 for signatures/sub-seeds (never builtin hash()), exact float Benna-Fusi
diffusion, no wall-clock, no global RNG. ASCII-only output. NORD_TORCH=0 runs it unchanged (torch
is never imported here).

  PYTHONUTF8=1 NORD_TORCH=0 PYTHONPATH=src python -m snn_data_lake.substrate.lake_memory
"""

import json
import os
import random
import tempfile
import zlib

from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.plasticity.benna_fusi import BennaFusiSynapse
from snn_data_lake.substrate.reservoir_sampling import Reservoir
from snn_data_lake.substrate.streaming_hub import StreamingHub

_MAX_T = 1 << 32  # exclusive upper bound over all u32 tick times (whole-lake query)

# ---- consolidation dynamics (deterministic; a single SHY-style offline pass) ----
DOWNSCALE = 0.5  # homeostatic global synaptic downscale applied to every trace (SHY)
REPLAY_GAIN = 0.30  # per-replay reinforcement deposited into a replayed trace's fast beaker
REPLAY_BATCH = 24  # SWR replay batch size per consolidation
DIFFUSE_STEPS = 3  # metaplastic seep steps (fast -> deep) after downscale + replay
PRUNE_EPS = 0.05  # evict every trace whose effective importance falls below this
SALIENT_CAP = 128  # hard cap on the consolidated long-term store (bound, event-count independent)
RESERVOIR_K = 64  # bounded uniform sample of the WHOLE stream (idle firehose can't grow the store)
TRACE_HARDCAP = 4096  # mid-epoch guard: never hold more than this many live traces (evict weakest)
COMPLETE_TH = 0.5  # reconstruct: min (cue-in-pattern) overlap for a partial cue to complete
NOVELTY_CAP = 8192  # bounded novelty table for auto-salience (drops oldest when full)


def _sig(channels):
    """Deterministic signature of an assembly (a set of channel indices). crc32, never hash()."""
    key = ",".join(str(c) for c in sorted(channels)).encode("ascii")
    return zlib.crc32(key)


class _Trace:
    """One consolidatable memory: a tick's assembly + a Benna-Fusi importance cascade.

    The cascade IS the metaplasticity -- deposit(salience) drops the plasticity event into the fast
    beaker; the consolidation pass downscales + reinforces (replay) + diffuses it; importance() reads
    the fast beaker (effective synaptic efficacy). A salient, replayed trace keeps charge in the deep
    beakers and survives the downscale; an idle one-off decays below PRUNE_EPS and is evicted."""

    __slots__ = ("sig", "t", "channels", "syn")

    def __init__(self, sig, t, channels, importance=None):
        self.sig = sig
        self.t = int(t)
        self.channels = frozenset(int(c) for c in channels)
        self.syn = BennaFusiSynapse()
        if importance is not None:  # reloaded-from-disk trace: re-prime the cascade
            self.syn.deposit(float(importance))

    def deposit(self, salience):
        self.syn.deposit(float(salience))

    def importance(self):
        """Effective importance = the fast beaker (bounded synaptic efficacy)."""
        return self.syn.weight

    def consolidated(self):
        """The slow deep-beaker store -- how much of this memory has seeped into long timescales."""
        return self.syn.deep_mean()


class MemoryLake:
    """A durable working-memory lake with real-neuron dynamics: prune (bounded) + reconstruct.

    Durable constructor: MemoryLake(path=...) resumes the lossless `.spkl` and its consolidated
    sidecar if they exist. Public faculties:
      write(t, active_channels, salience=None) / add(event, salience=None) -- record + track importance
      consolidate()          -- SHY-style downscale + SWR replay + metaplastic seep + evict -> bounded store
      reconstruct(cue, k=5)   -- pattern-complete a partial cue from the consolidated attractor store
      recall(window)          -- LakeWorkingMemory-compatible recent-window read of the durable lake
      most_recent(lo, hi)     -- one-hot value read from a reserved slot range (LakeWorkingMemory-compat)
      n_events()              -- total events in the durable lossless lake (== appends)
      n_kept()                -- size of the bounded consolidated store (post-prune)
      reopen() / close(remove=) -- restart recovery / cleanup, drop-in for LakeWorkingMemory
    Deterministic given `seed`; ASCII."""

    def __init__(
        self,
        n_channels=256,
        path=None,
        seed=0,
        cap=SALIENT_CAP,
        reservoir_k=RESERVOIR_K,
        vitals=False,
    ):
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".spkl")
            os.close(fd)  # StreamingHub opens its own handle
            self._created_temp = True
        else:
            self._created_temp = False
        self.n = int(n_channels)
        self.seed = int(seed)
        self.cap = int(cap)
        self.path = path
        self.sidecar = path + ".mem"  # ASCII JSON: the consolidated store, durable
        self.last_t = None  # the "now" cursor (latest tick written)
        # ---- reused durable substrate + bounded sampler + importance traces ----
        self.hub = StreamingHub(
            path, self.n, resume=os.path.exists(path) and os.path.getsize(path) > 0
        )
        self.reservoir = Reservoir(reservoir_k, seed=zlib.crc32(b"reservoir") ^ self.seed)
        self.traces = {}  # sig -> _Trace (live, pre/post consolidation)
        self.attractor = []  # consolidated survivors: [(sig, t, frozenset, imp)]
        self._novel = {}  # bounded novelty counts for auto-salience
        # OPT-IN read-only "Lake Vitals": a fixed-memory streaming self-monitor (HLL/CMS/DGIM/P2/AMS/
        # MinHash/Reservoir/Morris) over the durable spike/motif stream. None -> off, byte-identical.
        if vitals:
            from snn_data_lake.substrate.stream_sketch_panel import StreamSketchPanel

            self._vitals = StreamSketchPanel(seed=seed)
        else:
            self._vitals = None
        if self.hub.n_events():  # resumed a non-empty lake: recover the cursor
            self.last_t = max((t for t, _ in self.hub.query_window(0, _MAX_T)), default=None)
        self._load_sidecar()  # recover the consolidated store if one was persisted

    # ---- WRITE (with salience) ------------------------------------------------
    def write(self, t, active_channels, salience=None):
        """Record tick `t`: append one spike per active channel to the durable lossless lake, offer the
        tick to the bounded reservoir, and open/reinforce this assembly's importance trace. `salience`
        None -> an auto-novelty base of 1.0 (a never-seen assembly is salient). Every deposit is
        HABITUATED by 1/(1+prior occurrences), so a redundant idle loop cannot accumulate importance no
        matter how many times it repeats -- that is what makes idle noise prunable and salient one-offs
        survive. Shape-compatible with LakeWorkingMemory.write(t, active_channels)."""
        t = int(t)
        chans = [int(c) for c in active_channels]
        for c in chans:
            self.hub.append(c, t)  # durable lossless append (unchanged StreamingHub API)
        self.last_t = t if self.last_t is None else max(self.last_t, t)
        if not chans:
            return
        sig = _sig(chans)
        self.reservoir.add(
            (t, tuple(sorted(chans)))
        )  # one-pass fair sample of the WHOLE stream (bounds it)
        count = self._novel.get(sig, 0)  # prior occurrences of this exact assembly
        self._novel[sig] = count + 1
        if len(self._novel) > NOVELTY_CAP:
            self._novel.pop(
                next(iter(self._novel))
            )  # drop the oldest (insertion order) -> bounded table
        base = 1.0 if salience is None else float(salience)
        effective = base / (1.0 + count)  # habituation: redundant repeats deposit ever less
        tr = self.traces.get(sig)
        if tr is None:
            tr = _Trace(sig, t, chans)
            self.traces[sig] = tr
            self._enforce_trace_hardcap()
        tr.t = t  # remember the most recent occurrence
        tr.deposit(effective)
        if self._vitals is not None:  # read-only Lake Vitals tap (chans is non-empty here)
            self._vitals.observe(frozenset(chans))

    def add(self, event, salience=None):
        """Convenience salience-first writer. `event` is (t, channels) -- a tick and its active
        assembly. Equivalent to write(t, channels, salience)."""
        t, channels = event
        self.write(t, channels, salience=salience)

    def vitals_snapshot(self):
        """Read-only Lake Vitals: the streaming-sketch panel's snapshot (distinct motifs, heavy hitters,
        quantiles, drift novelty), or None if vitals were not enabled. Mutates nothing."""
        return self._vitals.snapshot() if self._vitals is not None else None

    def _enforce_trace_hardcap(self):
        """Mid-epoch memory guard: never hold more than TRACE_HARDCAP live traces (evict the weakest by
        deep-store), so even a flood of distinct low-salience assemblies cannot grow the store unbounded."""
        if len(self.traces) <= TRACE_HARDCAP:
            return
        worst = min(self.traces.values(), key=lambda tr: tr.consolidated())
        del self.traces[worst.sig]

    # ---- PRUNE: the SHY-style offline consolidation pass ----------------------
    def consolidate(self):
        """One offline consolidation pass -- the "sleep" that keeps the store bounded like real neurons:

          1. DOWNSCALE  : homeostatic global synaptic downscale of every importance trace (SHY).
          2. SWR REPLAY : reinforce the significant traces via the reused ReplayBuffer, forward+reverse,
                          importance-weighted (a trace enters the replay pool with multiplicity ~ its
                          importance, so idle traces are almost never replayed and stay weak).
          3. SEEP       : a few Benna-Fusi diffusion steps (replayed charge settles into deep beakers).
          4. EVICT      : hard-prune every trace still below PRUNE_EPS (idle / redundant / weak go away).
          5. CAP        : keep at most `cap` survivors, importance-weighted (highest importance retained).
        The surviving consolidated store becomes the reconstruct attractor and is persisted durably.
        Returns telemetry: {before, downscaled, pruned, kept, cap, reservoir}."""
        before = len(self.traces)

        # 1. homeostatic global downscale of every beaker of every trace (SHY offline downscaling)
        for tr in self.traces.values():
            u = tr.syn.u
            for k in range(len(u)):
                u[k] *= DOWNSCALE

        # 2. SWR replay: importance-weighted reinforcement of the significant traces (reused ReplayBuffer)
        buf = ReplayBuffer(
            capacity=max(256, 16 * (before + 1)),
            seed=zlib.crc32(b"swr") ^ self.seed ^ (before & 0xFFFFFFFF),
        )
        for tr in self.traces.values():  # multiplicity ~ importance -> significant traces dominate
            for _ in range(int(tr.importance() * 8)):
                buf.add(tr.sig, 0, tr.importance())
        picks = buf.sample(REPLAY_BATCH)
        for sig, _a, _r in picks + picks[::-1]:  # forward sweep then reverse sweep (SWR)
            tr = self.traces.get(sig)
            if tr is not None:
                tr.deposit(REPLAY_GAIN)

        # 3. metaplastic seep: reinforced charge settles fast -> deep
        for tr in self.traces.values():
            tr.syn.diffuse(DIFFUSE_STEPS)

        # 4. evict everything left below threshold (idle / redundant / weak)
        survivors = {sig: tr for sig, tr in self.traces.items() if tr.importance() >= PRUNE_EPS}
        pruned = before - len(survivors)

        # 5. cap the consolidated store (importance-weighted retention -> bound independent of #events)
        if len(survivors) > self.cap:
            ranked = sorted(
                survivors.values(),
                key=lambda tr: (tr.importance(), tr.consolidated(), tr.sig),
                reverse=True,
            )
            survivors = {tr.sig: tr for tr in ranked[: self.cap]}

        self.traces = survivors
        self.attractor = [
            (tr.sig, tr.t, tr.channels, round(tr.importance(), 6)) for tr in survivors.values()
        ]
        self.attractor.sort(
            key=lambda r: (-r[3], r[1], r[0])
        )  # deterministic order (importance, t, sig)
        self._save_sidecar()
        return {
            "before": before,
            "downscaled": before,
            "pruned": pruned,
            "kept": len(survivors),
            "cap": self.cap,
            "reservoir": len(self.reservoir),
        }

    # ---- RECONSTRUCT: pattern-complete a partial cue from the attractor store ---
    def reconstruct(self, cue, k=5):
        """Pattern-complete a PARTIAL cue against the consolidated attractor store. `cue` is a subset of
        an event's channels (any iterable of ints, or a single int). Returns up to `k` completed memories
        -- each the FULL stored assembly -- as (t, frozenset(channels)), best match first. A cue that
        overlaps no consolidated assembly (e.g. pruned idle noise) returns []: only real, consolidated
        memories complete. This is the CA3-style content-addressable recall AttractorNet.classify does,
        sized to the consolidated store."""
        cue_set = self._as_channels(cue)
        if not cue_set:
            return []
        scored = []
        denom = float(len(cue_set))
        for _signature, t, channels, imp in self.attractor:
            overlap = (
                len(cue_set & channels) / denom
            )  # fraction of the cue explained by this assembly
            if overlap >= COMPLETE_TH:
                scored.append((overlap, imp, t, channels))
        scored.sort(key=lambda r: (-r[0], -r[1], r[2]))  # best overlap, then importance, then time
        return [(t, channels) for _o, _i, t, channels in scored[: max(0, int(k))]]

    @staticmethod
    def _as_channels(cue):
        """Normalize a cue to a set of channel ints: accepts an int, or any iterable of ints (set,
        frozenset, list, tuple), or an (t, channels) pair from recall/most_recent."""
        if isinstance(cue, int):
            return {cue}
        if isinstance(cue, (set, frozenset)):
            return {int(c) for c in cue}
        if isinstance(cue, (list, tuple)):
            if len(cue) == 2 and isinstance(cue[0], int) and hasattr(cue[1], "__iter__"):
                return {int(c) for c in cue[1]}  # an (t, channels) record
            return {int(c) for c in cue}
        return {int(c) for c in cue}

    # ---- LakeWorkingMemory-compatible reads over the durable lake -------------
    def recall(self, window):
        """The last `window` ticks' contents, read from the durable lake and grouped per tick, as a list
        of (t, frozenset(channels)) sorted ascending (newest LAST). Byte-for-byte the LakeWorkingMemory
        contract, so live_brain.recall / autonomous_loop lake_wm keep working unchanged."""
        if self.last_t is None or window <= 0:
            return []
        t0 = max(0, self.last_t - window + 1)
        t1 = self.last_t + 1
        by_tick = {}
        for t, c in self.hub.query_window(t0, t1):
            by_tick.setdefault(t, set()).add(c)
        return [(t, frozenset(chans)) for t, chans in sorted(by_tick.items())]

    def most_recent(self, lo, hi):
        """The most-recently written channel in the reserved range [lo, hi), or None (LakeWorkingMemory
        one-hot value read; ties -> highest channel for determinism)."""
        chans = [c for c in range(lo, hi) if 0 <= c < self.n]
        if not chans or self.last_t is None:
            return None
        events = self.hub.query_window(0, self.last_t + 1, channels=chans)
        if not events:
            return None
        _, channel = max(events, key=lambda tc: (tc[0], tc[1]))
        return channel

    def n_events(self):
        """Total spikes in the durable lossless lake (== number of append()s). Unbounded by design --
        the raw log is lossless; it is the CONSOLIDATED store (n_kept) that is bounded."""
        return self.hub.n_events()

    def n_kept(self):
        """Size of the bounded consolidated store after the last consolidate() (the pruned memory the
        brain reconstructs from). Bounded by `cap`, independent of n_events()."""
        return len(self.attractor) if self.attractor else len(self.traces)

    # ---- durable persistence of the consolidated store -----------------------
    def _save_sidecar(self):
        """Persist the consolidated store to an ASCII JSON sidecar (deterministic, sorted) so the pruned
        memory + reconstruct capability survive a restart."""
        blob = {
            "n": self.n,
            "seed": self.seed,
            "last_t": self.last_t,
            "cap": self.cap,
            "attractor": [
                [sig, t, sorted(channels), imp] for sig, t, channels, imp in self.attractor
            ],
            "reservoir": [[t, list(ch)] for t, ch in self.reservoir.sample()],
        }
        tmp = self.sidecar + ".tmp"
        with open(tmp, "w", encoding="ascii") as f:
            json.dump(blob, f, ensure_ascii=True, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.sidecar)  # atomic swap: no half-written store

    def _load_sidecar(self):
        """Recover the consolidated store from the sidecar (if present) -> reconstruct works after a
        restart. Rebuilds a re-primed importance trace per survivor so a later consolidate() is sane."""
        if not os.path.exists(self.sidecar):
            return
        try:
            with open(self.sidecar, encoding="ascii") as f:
                blob = json.load(f)
        except (ValueError, OSError):
            return  # torn/partial sidecar -> start from the durable log only
        self.attractor = []
        self.traces = {}
        for sig, t, channels, imp in blob.get("attractor", []):
            tr = _Trace(sig, t, channels, importance=imp)
            self.traces[sig] = tr
            self.attractor.append((sig, t, frozenset(int(c) for c in channels), imp))
        self.attractor.sort(key=lambda r: (-r[3], r[1], r[0]))
        if self.last_t is None:
            self.last_t = blob.get("last_t")

    def reopen(self):
        """Close and reopen the SAME durable `.spkl` (StreamingHub resume) and reload the consolidated
        sidecar, rebuilding the whole store purely from disk -> proves the memory survives a restart.
        Returns self (LakeWorkingMemory-compatible)."""
        self.hub.close()
        self.hub = StreamingHub(self.path, self.n, resume=True)
        events = self.hub.query_window(0, _MAX_T)
        self.last_t = max((t for t, _ in events), default=None)
        self._load_sidecar()
        return self

    def close(self, remove=False):
        """Release the durable lake handle. If `remove` and this instance created a temp file, delete the
        `.spkl` and its sidecar (leaves no litter)."""
        self.hub.close()
        if remove and self._created_temp:
            for p in (self.path, self.sidecar):
                if os.path.exists(p):
                    os.remove(p)


def main():
    print("=" * 74)
    print("LAKE MEMORY -- durable working-memory lake that PRUNES + RECONSTRUCTS like neurons")
    print("=" * 74)

    N_CH = 64
    IDLE_TICKS = 1000  # idle-loop noise: low-salience, redundant
    SALIENT = 20  # distinct, salient events
    FILLER = (48, 49, 50)  # shared filler channels every salient assembly includes

    def build(seed, path=None):
        """Stream IDLE noise + SALIENT events into a fresh MemoryLake and consolidate. Deterministic.

        Each salient event i has a UNIQUE 2-channel signature pair (8+2i, 9+2i) in [8,48) plus the
        shared FILLER channels -> a 5-channel assembly. The unique pair is the partial cue: it is a
        subset of ONLY event i, so a 2-of-5 cue completes to exactly that event."""
        lk = MemoryLake(n_channels=N_CH, path=path, seed=seed)
        rng = random.Random(seed)  # per-instance RNG (never global)
        # ---- ~1000 low-salience IDLE ticks: a few channels in a small idle band, redundant/prunable ----
        for t in range(IDLE_TICKS):
            active = sorted(rng.sample(range(0, 8), 2))  # idle band [0,8) -> redundant / prunable
            lk.write(t, active, salience=0.01)
        # ---- 20 DISTINCT SALIENT events: unique signature pair + shared filler, salience 1.0 ----
        salient = {}
        for i in range(SALIENT):
            pair = (8 + 2 * i, 9 + 2 * i)  # unique to event i (disjoint across events)
            assembly = sorted(set(pair) | set(FILLER))
            t = IDLE_TICKS + i
            salient[t] = assembly
            lk.write(t, assembly, salience=1.0)
        tele = lk.consolidate()
        return lk, salient, tele

    lk, salient, tele = build(seed=0)
    total = lk.n_events()
    kept = lk.n_kept()

    # ---- (b) each SALIENT event reconstructs from a PARTIAL cue; idle noise does NOT ----
    reconstructed = 0
    example = None
    for _t, assembly in salient.items():
        cue = assembly[:2]  # a HARD partial cue: 2 of the 5 channels
        out = lk.reconstruct(cue, k=3)
        hit = any(frozenset(assembly) == chans for _t, chans in out)
        reconstructed += 1 if hit else 0
        if example is None and hit:
            example = (cue, assembly, [(rt, sorted(rc)) for rt, rc in out])
    # an idle cue (channels from the idle band) must NOT complete to any consolidated memory
    idle_out = lk.reconstruct([0, 1], k=5)

    print(
        f"streamed     : {IDLE_TICKS} idle ticks + {SALIENT} salient events "
        f"-> {total:,} durable events (lossless log)"
    )
    print(
        f"consolidate  : before {tele['before']} traces -> pruned {tele['pruned']} "
        f"-> kept {tele['kept']} (cap {tele['cap']}); reservoir sample {tele['reservoir']}"
    )
    print(
        f"kept vs total: n_kept={kept} is BOUNDED and << {IDLE_TICKS + SALIENT} total ticks "
        f"({100.0 * kept / (IDLE_TICKS + SALIENT):.1f}% -> idle-loop noise pruned)"
    )
    print("-" * 74)
    ex_cue, ex_full, ex_out = example
    print(f"reconstruct  : cue {ex_cue} (2 of 5 channels) -> completed {ex_out[0][1]}")
    print(f"               (full stored assembly {ex_full} recovered from the partial cue)")
    print(
        f"salient recon: {reconstructed}/{SALIENT} salient events reconstructable from a partial cue"
    )
    print(
        f"idle recon   : cue [0, 1] (idle band) -> {idle_out} (pruned noise does NOT reconstruct)"
    )

    # ---- (c) DURABLE: consolidated store survives close + reopen (fresh object, disk only) ----
    lk.close()  # flush + release; sidecar already persisted
    reop = MemoryLake(n_channels=N_CH, path=lk.path, seed=0)  # a FRESH object, same durable path
    kept_reopen = reop.n_kept()
    t0, asm0 = next(iter(salient.items()))
    reop_out = reop.reconstruct(asm0[:2], k=3)
    reop_hit = any(frozenset(asm0) == chans for _t, chans in reop_out)
    events_reopen = reop.n_events()

    print("-" * 74)
    print(
        f"persistence  : reopened from disk -> n_events {events_reopen:,} (lossless log intact), "
        f"n_kept {kept_reopen} (consolidated store intact)"
    )
    print(
        f"               reconstruct after restart: cue {asm0[:2]} -> hit={reop_hit} "
        f"(the pruned memory survived the restart)"
    )

    # ---- (d) DETERMINISM: same seed -> same kept set ----
    lkA, _sa, _ta = build(seed=7)
    lkB, _sb, _tb = build(seed=7)
    kept_A = sorted(sig for sig, *_ in lkA.attractor)
    kept_B = sorted(sig for sig, *_ in lkB.attractor)
    deterministic = kept_A == kept_B
    print("-" * 74)
    print(f"determinism  : two seed=7 lakes -> identical kept set = {deterministic}")
    print("=" * 74)

    # ---- self-check asserts (deterministic; all four properties) --------------
    # (a) PRUNING happened: the consolidated store is BOUNDED and << the total streamed ticks.
    assert kept <= SALIENT_CAP, (
        f"consolidated store must be bounded by the cap ({kept} > {SALIENT_CAP})"
    )
    assert kept < (IDLE_TICKS + SALIENT) // 4, (
        f"pruning must shrink the store far below the stream ({kept} not << {IDLE_TICKS + SALIENT})"
    )
    assert tele["pruned"] > 0, "consolidate must have pruned idle/weak traces"
    assert total == IDLE_TICKS * 2 + SALIENT * 5, (
        "durable lossless log lost events (n_events wrong)"
    )
    # (b) RECONSTRUCT: every salient event completes from a partial cue; idle noise does not.
    assert reconstructed == SALIENT, (
        f"every salient event must reconstruct from a partial cue ({reconstructed}/{SALIENT})"
    )
    assert idle_out == [], (
        "pruned idle noise must NOT reconstruct (it was evicted, not consolidated)"
    )
    # (c) DURABLE: the consolidated store + reconstruct survive a real close + reopen from disk.
    assert kept_reopen == kept, f"consolidated store must survive reopen ({kept_reopen} != {kept})"
    assert events_reopen == total, "durable lossless log must survive reopen (n_events changed)"
    assert reop_hit, "a salient memory must still reconstruct after a process restart (persistence)"
    # (d) DETERMINISM: same seed -> identical kept set.
    assert deterministic, "same seed must produce the same kept set"

    print(
        f"self-check OK: {IDLE_TICKS + SALIENT} ticks -> BOUNDED consolidated store n_kept={kept} "
        f"(<< total, idle pruned); all {SALIENT}/{SALIENT} salient events reconstruct from a partial "
        f"cue while idle noise does not; survives close+reopen (n_kept {kept_reopen}, reconstruct "
        f"hit={reop_hit}); deterministic. Reused StreamingHub + Reservoir + ReplayBuffer + "
        f"BennaFusiSynapse; the durable lossless lake API is untouched."
    )

    # ---- cleanup: release the reopened handle FIRST (it shares lk's temp file), then delete ----
    reop.close()  # drop the shared-file handle before removal
    lk.close(remove=True)  # lk owns the temp file -> delete .spkl + sidecar
    lkA.close(remove=True)
    lkB.close(remove=True)


if __name__ == "__main__":
    main()
