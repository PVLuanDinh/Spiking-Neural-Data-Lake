"""
Global Workspace -- a functional Global-Workspace / global-broadcast attention mechanism
(pure stdlib, zero deps).

Global Workspace Theory (Baars 1988; Dehaene & Changeux's Global Neuronal Workspace,
Neuron 70:200, 2011): many specialist modules run in parallel and post candidate "contents";
they COMPETE for a single limited-capacity workspace; a winner that crosses an all-or-none
IGNITION threshold seizes the slot and is BROADCAST back to every module (global availability);
the content is SUSTAINED by recurrent self-amplification and DECAYS once it is no longer
refreshed. This is the "conscious access = global broadcast" bottleneck, modelled here as a
running mechanism -- state persists across ticks, which is what makes it "alive". In this repo
the specialists are the brain regions (perception/cortex, memory/hippocampus, salience/threat,
language, motor); here they are named mock salience streams that stand in for those modules.

The mechanism, one `step()` per tick:
  1. COMPETE  : the strongest of this tick's candidates challenges the incumbent, which defends
                its slot with a small recurrent self-boost (sustained attention).
  2. IGNITION : all-or-none. A challenger seizes the workspace ONLY if its salience clears the
                ignition threshold AND beats the self-boosted incumbent by more than a hysteresis
                margin. Otherwise the incumbent holds. Nothing supra-threshold + empty slot -> empty.
  3. BROADCAST: the single current content is delivered to every subscriber (global reach).
  4. DECAY    : an unrefreshed incumbent's salience decays; below threshold it is released.

THE UN-RIGGABLE ANCHOR is the ignition non-linearity, tested against a FAIR CONTROL: the same
noisy competing-salience stream is run (a) WITH the threshold + hysteresis and (b) WITHOUT them
(pure per-tick argmax, the only thing toggled). WITHOUT -> the workspace FLICKERS, its content
changing on almost every tick of noise; WITH -> it is STABLE and only a genuinely stronger,
supra-threshold signal takes over. The self-check prints both switch counts; the gap is measured,
not asserted into existence.

HONEST FRAMING: this is a functional attention/broadcast bottleneck (compete -> ignite ->
broadcast -> decay). It is NOT a claim of phenomenal consciousness; "ignition" and "broadcast"
name the information-routing dynamics, nothing more.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.thalamus.global_workspace
"""
import random

REGIONS = ("perception", "memory", "salience", "language", "motor")


class GlobalWorkspace:
    """A single limited-capacity workspace with all-or-none ignition and global broadcast.

    Capacity is ONE content at a time -- that single slot IS the bottleneck. Specialists call
    `submit(source, content, salience)` to post candidates for the current tick; `step()` runs one
    compete -> ignite -> broadcast -> decay cycle and returns the broadcast content (or None).
    `subscribe(fn)` registers a module callback that receives every broadcast (global reach).

    `use_ignition=False` is the FAIR CONTROL: the ignition non-linearity is switched off and
    `step()` becomes a pure per-tick argmax over the submissions (no threshold, hysteresis, self-
    boost or decay). Everything else -- the same submissions, the same competition -- is identical.
    """

    def __init__(self, ignition_threshold=0.55, decay=0.85, hysteresis=0.12,
                 self_boost=0.05, use_ignition=True, seed=0):
        self.ignition_threshold = ignition_threshold
        self.decay = decay
        self.hysteresis = hysteresis
        self.self_boost = self_boost
        self.use_ignition = use_ignition
        self._rng = random.Random(seed)      # per-instance RNG (reserved for stochastic tie-breaks)
        self._incumbent = None               # (source, content, salience) occupying the slot, or None
        self._buffer = []                    # this tick's submitted candidates
        self._subscribers = []               # module callbacks receiving each broadcast

    # -- specialist / module interface ----------------------------------------------------------
    def submit(self, source, content, salience):
        """A specialist posts a candidate into this tick's competition buffer."""
        self._buffer.append((str(source), content, float(salience)))

    def subscribe(self, fn):
        """Register a module callback fn(content) that receives every broadcast."""
        self._subscribers.append(fn)
        return fn

    @property
    def current(self):
        """The single content currently occupying the workspace (globally available), or None."""
        return self._incumbent[1] if self._incumbent is not None else None

    def broadcast(self):
        """The current workspace content -- what every module can see right now."""
        return self.current

    # -- one workspace cycle --------------------------------------------------------------------
    def _deliver(self, content):
        for fn in self._subscribers:
            fn(content)

    def _top_candidate(self, exclude=None):
        # Highest salience wins; deterministic tie-break by source name (lexicographically first).
        pool = [c for c in self._buffer if exclude is None or (c[0], c[1]) != exclude]
        if not pool:
            return None
        return min(pool, key=lambda c: (-c[2], c[0]))

    def _own_drive(self, inc):
        # The strongest re-submission of the incumbent's own content this tick (bottom-up refresh).
        drives = [c[2] for c in self._buffer if (c[0], c[1]) == (inc[0], inc[1])]
        return max(drives) if drives else None

    def step(self):
        """Run ONE workspace cycle. Returns the broadcast content (or None) and clears the buffer."""
        inc = self._incumbent

        if not self.use_ignition:
            # FAIR CONTROL: pure per-tick argmax. Whatever is loudest THIS tick wins outright.
            top = self._top_candidate()
            if top is not None:
                self._incumbent = top
            content = self.current
            self._buffer = []
            self._deliver(content)
            return content

        # --- full Global-Workspace ignition dynamics ---
        # 1. SUSTAIN: the incumbent is kept alive by its own ongoing drive (recurrent + bottom-up),
        #    independent of whether it is the loudest this tick; with no drive it DECAYS and, once
        #    below the ignition line, is RELEASED.
        if inc is not None:
            own = self._own_drive(inc)
            if own is not None:
                inc = (inc[0], inc[1], min(1.0, max(inc[2] * self.decay, own)))
            else:
                decayed = inc[2] * self.decay
                inc = None if decayed < self.ignition_threshold else (inc[0], inc[1], decayed)

        # 2. COMPETE + IGNITE: the strongest OTHER candidate challenges the self-boosted incumbent.
        exclude = (inc[0], inc[1]) if inc is not None else None
        challenger = self._top_candidate(exclude=exclude)
        if challenger is not None:
            csal = challenger[2]
            if inc is None:
                if csal >= self.ignition_threshold:
                    inc = challenger                           # ignite into an empty workspace
            elif csal >= self.ignition_threshold and csal > inc[2] + self.self_boost + self.hysteresis:
                inc = challenger                               # all-or-none takeover by a stronger signal

        self._incumbent = inc
        content = self.current
        self._buffer = []
        self._deliver(content)                                 # BROADCAST to all subscribers
        return content


def build_stream(n_ticks, seed=0):
    """A fixed per-tick submission stream: a list of {region: salience} dicts.

    'perception' is a persistently strong (mostly supra-threshold) signal; the other regions are
    noisy near-threshold distractors that heavily overlap it -- so a pure argmax flickers among
    them. A scripted THREAT burst drives 'salience' (an amygdala-like threat detector) far above
    threshold for a short window, injecting a genuinely stronger signal mid-stream. Deterministic
    under `seed`.
    """
    rng = random.Random(seed)
    threat_window = set(range(30, 36))
    means = {"perception": 0.66, "memory": 0.54, "salience": 0.54,
             "language": 0.53, "motor": 0.52}
    sd = 0.11
    stream = []
    for t in range(n_ticks):
        tick = {}
        for r in REGIONS:
            s = rng.gauss(means[r], sd)
            if r == "salience" and t in threat_window:
                s = 0.98                                        # supra-threshold threat event
            tick[r] = min(1.0, max(0.0, s))
        stream.append(tick)
    return stream


def run_stream(stream, use_ignition, **params):
    """Feed a stream through a workspace tick by tick; return the list of broadcast contents."""
    ws = GlobalWorkspace(use_ignition=use_ignition, **params)
    broadcasts = []
    for tick in stream:
        for region, salience in tick.items():
            ws.submit(region, region, salience)                # content == region label
        broadcasts.append(ws.step())
    return broadcasts


def count_switches(seq):
    """Number of ticks whose broadcast content differs from the previous tick (flicker measure)."""
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def _fmt(seq):
    """Compact ASCII trace: initials of the broadcast region per tick, '.' for an empty workspace."""
    return "".join((c[0].upper() if c is not None else ".") for c in seq)


def main():
    # ---- FAIR CONTROL: same stream, ignition ON vs OFF (pure per-tick argmax) ------------------
    n_ticks = 48
    stream = build_stream(n_ticks, seed=0)
    with_ig = run_stream(stream, use_ignition=True, seed=0)
    without = run_stream(stream, use_ignition=False, seed=0)
    sw_with = count_switches(with_ig)
    sw_without = count_switches(without)

    print("=" * 74)
    print("GLOBAL WORKSPACE -- compete -> ignite -> broadcast -> decay (functional, not phenomenal)")
    print("=" * 74)
    print("regions: perception(P) memory(M) salience/threat(S) language(L) motor(O); '.' = empty")
    print("threat burst (region 'salience') at ticks 30-35\n")
    print(f"WITH ignition  (threshold+hysteresis): {_fmt(with_ig)}  switches={sw_with}")
    print(f"WITHOUT (pure per-tick argmax control): {_fmt(without)}  switches={sw_without}")
    print(f"\n-> ignition non-linearity cut workspace switches from {sw_without} (flicker) "
          f"to {sw_with} (stable)")

    # threat must actually break through: a genuinely stronger supra-threshold signal takes over.
    threat_broadcast = any(c == "salience" for c in with_ig[30:36])

    # ---- ANCHOR (i) BOTTLENECK: many high-salience candidates -> exactly ONE broadcast ---------
    seen = []
    b_ws = GlobalWorkspace(seed=0)
    b_ws.subscribe(seen.append)
    for r in REGIONS:
        b_ws.submit(r, r, 0.90)                                # five simultaneous supra-threshold posts
    b_content = b_ws.step()
    bottleneck_ok = b_content is not None and seen[-1] == b_content

    # ---- ANCHOR (ii) ALL-OR-NONE: sub-threshold stays out, supra-threshold enters -------------
    a_ws = GlobalWorkspace(seed=0)
    a_ws.submit("perception", "weak", 0.40)                    # below threshold
    sub = a_ws.step()
    a_ws.submit("perception", "strong", 0.80)                  # above threshold
    supra = a_ws.step()
    # not linear averaging: two sub-threshold posts do NOT sum into an ignition.
    n_ws = GlobalWorkspace(seed=0)
    n_ws.submit("memory", "half_a", 0.50)
    n_ws.submit("language", "half_b", 0.50)
    no_sum = n_ws.step()

    # ---- ANCHOR (iii) GLOBAL REACH: every subscriber gets the identical broadcast --------------
    r_ws = GlobalWorkspace(seed=0)
    inbox_a, inbox_b, inbox_c = [], [], []
    r_ws.subscribe(inbox_a.append)
    r_ws.subscribe(inbox_b.append)
    r_ws.subscribe(inbox_c.append)
    for sal in (0.70, 0.30, 0.72, 0.20):
        r_ws.submit("perception", "scene", sal)
        r_ws.step()
    reach_ok = inbox_a == inbox_b == inbox_c and len(inbox_a) == 4

    # ---- ANCHOR (iv) PERSISTENCE + DECAY: ignite, then no input -> holds then fades ------------
    d_ws = GlobalWorkspace(seed=0)
    d_ws.submit("memory", "trace", 0.95)
    d_ws.step()                                                # ignite
    life = []
    for _ in range(8):
        life.append(d_ws.step())                               # no new submissions -> decay
    persisted = sum(1 for c in life if c == "trace")
    released = life[-1] is None
    print("\npersistence/decay (ignite 'memory' at 0.95, then silence):")
    print(f"   post-ignition life trace: {_fmt(life)}  "
          f"(held 'trace' {persisted} ticks, released={released})")

    # ---- DETERMINISM: identical stream+seed -> identical broadcasts and counts -----------------
    with_ig2 = run_stream(build_stream(n_ticks, seed=0), use_ignition=True, seed=0)
    deterministic = with_ig2 == with_ig and count_switches(with_ig2) == sw_with

    # ============================ genuine self-checks (not rigged) ==============================
    # The anchor: ignition stabilises the workspace vs the pure-argmax control, by a real margin.
    assert sw_without > sw_with, f"argmax control must flicker more ({sw_without} vs {sw_with})"
    assert sw_without >= 3 * max(sw_with, 1), (
        f"stabilisation must be large, not marginal: with={sw_with} without={sw_without}")
    assert sw_with <= 6, f"workspace must be stable WITH ignition (few switches), got {sw_with}"

    # (i) bottleneck: exactly one content broadcast despite five supra-threshold candidates.
    assert bottleneck_ok, "exactly one content must be broadcast through the single-slot bottleneck"
    assert isinstance(b_content, str), "the single broadcast content must be one region, not a set"

    # (ii) all-or-none: sub-threshold excluded, supra-threshold admitted, no linear summation.
    assert sub is None, f"a sub-threshold candidate must NOT enter the workspace (got {sub!r})"
    assert supra == "strong", f"a supra-threshold candidate must ignite (got {supra!r})"
    assert no_sum is None, "two sub-threshold candidates must NOT sum into an ignition (all-or-none)"

    # (iii) global reach: every subscriber received the identical broadcast sequence.
    assert reach_ok, "every subscriber must receive the identical broadcast (global reach)"

    # (iv) persistence + decay: the content is sustained several ticks, then fades below threshold.
    assert persisted >= 2, f"ignited content must persist for several ticks (held {persisted})"
    assert released, "unrefreshed content must eventually decay below threshold and be released"

    # a genuinely stronger, supra-threshold signal DID take over (workspace is not frozen).
    assert threat_broadcast, "the supra-threshold threat burst must break through and be broadcast"

    # determinism.
    assert deterministic, "same seed must reproduce identical broadcasts and switch counts"

    print("\nself-check OK: "
          f"ignition switches {sw_with} vs argmax {sw_without} (>=3x stabler); "
          f"bottleneck=one; all-or-none (sub={sub!r}, supra={supra!r}, no-sum={no_sum!r}); "
          f"global-reach; persist {persisted}->release; threat broke through; deterministic")


if __name__ == "__main__":
    main()
