"""
Global Workspace -- a SPIKING Global-Workspace / global-broadcast attention mechanism.
(Dependencies are governed by P7; this module imports `random` only. The line here used to say
"uses only `math` + `random`", and `math` has never been imported.)

Global Workspace Theory (Baars 1988; Dehaene & Changeux's Global Neuronal Workspace,
Neuron 70:200, 2011): many specialist modules run in parallel and post candidate "contents";
they COMPETE for a single limited-capacity workspace; a winner that crosses an all-or-none
IGNITION threshold seizes the slot and is BROADCAST back to every module (global availability);
the content is SUSTAINED by recurrent self-amplification and DECAYS once it is no longer refreshed.
This is the "conscious access = global broadcast" bottleneck.

Here that bottleneck is implemented with GENUINE SPIKING neurons -- not scalar thresholds:

  * Each candidate content/specialist is a LEAKY INTEGRATE-AND-FIRE (LIF) POOL: a membrane
    potential V integrates input current, emits a DISCRETE SPIKE when V crosses v_th, and
    resets. No spike = no activity; the sub-threshold / supra-threshold boundary IS the
    ignition non-linearity (a pool driven below rheobase never fires and stays purely local).
  * RECURRENT SELF-EXCITATION: an active pool feeds its own spikes back as excitatory current
    (a within-tick synaptic trace) and carries a cross-tick attractor memory `g` -- this is the
    attractor that SUSTAINS an ignited content and gives the incumbent its hysteresis.
  * A SHARED INHIBITORY POOL: every excitatory spike drives one global inhibitory pool whose
    output subtracts from EVERY content pool (global lateral inhibition). This is the
    competition: the strongest pool keeps firing while it silences the rest -> a single winner.
  * IGNITION = SUSTAINED FIRING: a content ignites (wins the slot + is broadcast) only if it is
    still spiking in the SECOND HALF of the settling window -- i.e. it survived the competition
    and is in the high-firing attractor. A pool that fires a few transient spikes and is then
    inhibited to silence does NOT ignite; it stays local.

So: weak input -> the pool never crosses v_th (or is inhibited to silence) -> stays local, no
broadcast. Strong input -> the pool ignites into sustained firing, recruits its self-excitation,
silences competitors through the shared inhibitory pool, and is broadcast globally. That
all-or-none ignition is the un-riggable anchor, checked against a FAIR CONTROL (`use_ignition=
False`) that is a pure per-tick argmax with no spiking, no attractor, no inhibition: the control
FLICKERS on noise; the spiking workspace is STABLE and only a genuinely stronger, supra-threshold
signal takes over.

FRAMING (agnostic, per the project's consciousness program): this is a functional spiking
attention/broadcast bottleneck (integrate -> spike -> compete -> ignite -> broadcast -> decay), a
concrete step toward the Global Neuronal Workspace substrate of ACCESS consciousness. Whether such
dynamics contribute to consciousness is an OPEN question the project actively builds toward; it is
neither denied nor claimed here. "ignition" and "broadcast" name the information-routing dynamics.

DEEPENING + the SPIKING-vs-RATE CONTROL (functions + self-check below): the argmax control above
stabilises the workspace but strips spikes AND cross-tick memory AND inhibition together, so it does
NOT isolate whether SPIKES are load-bearing. A binding test does. The workspace must BIND K objects,
each a (color, shape) pair, and broadcast the pairings. The spiking path phase-multiplexes them
(Lisman-Jensen theta-gamma: each object in its own gamma slot, its color+shape firing SYNCHRONOUSLY
so a reader recovers the pairing from synchrony); a RATE path with the SAME hysteresis superposes the
activations and loses which color binds which shape (the classic binding problem). Honest result,
asserted below: they TIE at K=1 (spikes are NOT needed for single-item selection) and spikes WIN at
K>=2 (multi-item binding via synchrony) -- the specific, honest answer to "what do spikes buy." This
is contested (Singer/Gray vs Shadlen-Movshon), reader-dependent, and one solution among several; it
distinguishes the spiking workspace from the rate one on this task, nothing more.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.thalamus.global_workspace
"""

import itertools
import random
import threading
import time

REGIONS = ("perception", "memory", "salience", "language", "motor")


def _fold_parts(parts):
    """The shared inhibitory pool: `sum()` over the per-chunk contribution lists CONCATENATED in
    ascending chunk order, which is exactly the flat sequence the unpartitioned code summed.

    This is the ONE definition both the serial and the parallel settling paths call, and the choice
    of `sum` is load-bearing rather than idiomatic. Two ways to get this wrong, both measured on this
    module rather than reasoned about:

      * summing each chunk and adding the SUBTOTALS -- the obvious way to parallelise a reduction --
        reassociates the additions and gives a different float. On 240 pools it differs at some
        partition counts and agrees at others, so it is not even consistently wrong; it makes the
        result depend on the thread count.
      * a naive `acc += x` left fold is ALSO not equivalent, which is the trap here. CPython's
        builtin `sum()` applies Neumaier compensated summation to floats, so it is strictly more
        accurate than an accumulator loop and disagrees with one in the last bits. A hand-rolled
        fold "in serial order" therefore silently changes the numbers; this lane wrote that fold
        first, and the integer spike counts absorbed the difference and reported byte-identical at
        ten widths. It was caught only by comparing MEMBRANE FLOATS against a frozen reference.

    Concatenating and calling the same builtin avoids both: identical elements, identical order,
    identical algorithm."""
    return sum(itertools.chain.from_iterable(parts))


class GlobalWorkspace:
    """A single limited-capacity workspace of competing LIF pools with all-or-none ignition.

    Capacity is ONE content at a time -- that single slot IS the bottleneck. Specialists call
    `submit(source, content, salience)` to post candidates for the current tick; `step()` runs one
    settle-to-ignition cycle of the spiking network and returns the broadcast content (or None).
    `subscribe(fn)` registers a module callback that receives every broadcast (global reach).

    The salience of a submission is the EXTERNAL CURRENT driving that content's LIF pool. A pool
    fires only above rheobase (`leak * v_th == ignition_threshold`), so a sub-threshold salience
    never ignites; a supra-threshold one integrates to a spike, recruits recurrent self-excitation,
    and -- if it wins the shared-inhibition competition -- reaches SUSTAINED firing and is broadcast.

    `use_ignition=False` is the FAIR CONTROL: the spiking dynamics are switched off and `step()`
    becomes a pure per-tick argmax over the submissions (no LIF, no attractor, no inhibition).
    Everything else -- the same submissions, the same competition of candidates -- is identical.
    """

    # -- fixed LIF network constants (per-instance; deterministic, ASCII) --------------
    _T = 60  # settling micro-steps per workspace tick
    _LEAK = 0.5  # membrane leak (V decays toward input/leak each micro-step)
    _W_SELF = 0.6  # recurrent self-excitation gain (within-tick attractor)
    _TAU_SELF = 0.6  # within-tick self-excitation synaptic-trace decay
    _K_INH = 0.30  # excitatory-spike -> shared inhibitory-pool drive
    _LEAK_INH = 0.5  # shared inhibitory-pool leak
    _G_INH = 2.4  # inhibitory-pool -> content-pool feedback gain (mutual/lateral)
    _JITTER = 0.02  # tiny deterministic symmetry-break so exact ties yield ONE winner

    def __init__(
        self,
        ignition_threshold=0.55,
        decay=0.85,
        hysteresis=0.20,
        self_boost=0.05,
        use_ignition=True,
        seed=0,
    ):
        self.ignition_threshold = (
            ignition_threshold  # rheobase current: a pool fires iff drive > this
        )
        self.decay = decay  # cross-tick attractor decay (persistence fade)
        self.hysteresis = hysteresis  # incumbent's extra recurrent defence current...
        self.self_boost = self_boost  # ...(self_boost + hysteresis) holds the attractor
        self.use_ignition = use_ignition
        self._v_th = (
            ignition_threshold / self._LEAK
        )  # membrane threshold s.t. rheobase == ignition_threshold
        self._rng = random.Random(seed)  # per-instance RNG (deterministic symmetry-break jitter)
        self._incumbent = None  # (source, content, g) occupying the slot, or None
        self._buffer = []  # this tick's submitted candidates
        self._subscribers = []  # module callbacks receiving each broadcast
        self._g = {}  # (source, content) -> cross-tick attractor memory in [0,1]
        self._jit = {}  # (source, content) -> fixed tiny jitter current
        self._last_total = {}  # (source, content) -> spikes this tick (spiking self-check)
        self._last_late = {}  # (source, content) -> 2nd-half spikes this tick (sustained)
        self._sim_per_pool_s = 0.0  # running-EMA cost of settling ONE pool, seconds (P1 baseline)
        self._sim_par_eff = 1.0  # running-EMA fraction of IDEAL parallel savings actually achieved
        self._slot_buf = []  # per-submitter staging buffers for CONCURRENT specialists
        self._last_v = {}  # (source, content) -> final membrane float (determinism self-check)

    # -- specialist / module interface ----------------------------------------------------------
    def submit(self, source, content, salience):
        """A specialist posts a candidate into this tick's competition buffer.

        SINGLE-THREADED ONLY. This appends to one shared list, and buffer ORDER is observable: the
        `_g` dict is built by iterating the buffer, and `_jitter_for` draws each pool's tie-break
        current from the instance RNG in first-encounter order. Two specialists calling this
        concurrently would therefore make the symmetry-break jitter depend on which thread won a
        race, which is exactly the nondeterminism this module's determinism self-check forbids.
        Concurrent specialists use `open_slots()` + `submit_from()` below instead."""
        self._buffer.append((str(source), content, float(salience)))

    def open_slots(self, n_slots):
        """Allocate `n_slots` DISJOINT staging buffers, one per concurrent specialist, and return the
        count. Call this from the single thread that is about to fan out, BEFORE starting any worker.

        This is the collect-then-apply answer to 'the workspace is a shared mutable structure'. It
        needs no lock because it removes the sharing rather than arbitrating it: each specialist owns
        buffer `slot` outright, the outer list is only read once the workers exist, and `step()`
        splices the buffers back in ASCENDING SLOT ORDER. Thread completion order therefore cannot
        reach the competition, and the result is a pure function of (which slot submitted what, in
        what program order) -- identical whether the specialists ran concurrently or one at a time."""
        n = max(0, int(n_slots))
        self._slot_buf = [[] for _ in range(n)]
        return n

    def submit_from(self, slot, source, content, salience):
        """A CONCURRENT specialist posts a candidate into its own staging buffer. Safe to call from
        `slot`'s own thread with no lock: no other thread touches `self._slot_buf[slot]`."""
        self._slot_buf[int(slot)].append((str(source), content, float(salience)))

    def _drain_slots(self):
        """Splice the staged concurrent submissions into the competition buffer, ascending slot order
        then program order within a slot. Deterministic by construction; a no-op when nothing fanned
        out, so the single-threaded path is bit-for-bit what it always was."""
        if not self._slot_buf:
            return
        for buf in self._slot_buf:
            if buf:
                self._buffer.extend(buf)
                buf.clear()

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

    def _top_candidate(self):
        # Highest salience wins; deterministic tie-break by source name (lexicographically first).
        if not self._buffer:
            return None
        return min(self._buffer, key=lambda c: (-c[2], c[0]))

    def _jitter_for(self, key):
        # A fixed, tiny per-pool current drawn once (deterministic under the seed) -- breaks the
        # symmetry of exactly-equal drives so the shared-inhibition WTA settles on ONE winner.
        if key not in self._jit:
            self._jit[key] = self._rng.uniform(0.0, self._JITTER)
        return self._jit[key]

    def _settle_chunk(self, mkeys, drives, sync):
        """Integrate ONE contiguous chunk of content pools across the whole settling window.

        This is the ONLY definition of the pool arithmetic -- the serial and the parallel paths both
        run exactly this function, differing only in how many chunks there are and in what `sync`
        does (P2: one mechanism, not two implementations to keep in agreement).

        The chunk owns `v/trace/si/total/late/sp` EXCLUSIVELY: no other chunk reads or writes them,
        so a thread running this touches no container another thread touches. The only shared value
        is the scalar `a_inh`, and it is read-only within a micro-step -- which is the whole reason a
        timestep of an SNN parallelises. `sync(si, t)` publishes this chunk's inhibitory
        contributions and returns the shared inhibitory-pool activity for the next micro-step.

        Returns (mkeys, total, late, v) -- integer spike counts and the final float membranes,
        positionally aligned. `v` is returned because the integer counts are TOO COARSE to police
        this module's determinism: a reassociated reduction changes `a_inh` in the last bits and the
        spike counts absorb it, so the self-check compares the float trajectory instead."""
        m = len(mkeys)
        dr = [drives[k] for k in mkeys]
        v = [0.0] * m  # membrane potentials
        trace = [0.0] * m  # within-tick self-excitation traces
        si = [0.0] * m  # each pool's own contribution to the inh pool
        sp = [0] * m  # this micro-step's spikes
        total = [0] * m
        late = [0] * m
        half = self._T // 2
        a_inh = sync(si, -1)  # register this chunk; shared pool starts silent
        for t in range(self._T):
            for i in range(m):
                lateral = self._G_INH * max(0.0, a_inh - si[i])  # inhibition from OTHER pools
                cur = dr[i] + self._W_SELF * trace[i] - lateral  # LIF input current
                v[i] = v[i] + (cur - self._LEAK * v[i])  # leaky integration
                if v[i] < 0.0:
                    v[i] = 0.0
                s = 0
                if v[i] >= self._v_th:  # DISCRETE SPIKE
                    v[i] -= self._v_th  # subtract-threshold reset
                    s = 1
                    total[i] += 1
                    if t >= half:
                        late[i] += 1
                    trace[i] += 1.0  # self-excitation kick
                sp[i] = s
                trace[i] *= self._TAU_SELF
            for i in range(m):  # integrate the inhibitory pools
                si[i] += self._K_INH * sp[i] - self._LEAK_INH * si[i]
                if si[i] < 0.0:
                    si[i] = 0.0
            a_inh = sync(si, t)  # shared pool == sum of contributions
        return mkeys, total, late, v

    def _simulate(self, drives):
        """Settle the LIF network for `_T` micro-steps and return the sustained-firing readout.

        Each content pool k has a membrane V_k that integrates: its external drive current, its
        own recurrent self-excitation (within-tick trace), MINUS the shared inhibitory pool's
        feedback. The shared inhibitory pool `a_inh` sums EVERY excitatory spike; each content pool
        `self_k` tracks only its OWN contribution to it, so the inhibition actually applied to pool
        k is `g_inh * (a_inh - self_k)` -- the drive from every OTHER pool. That is the global
        lateral inhibition (each pool inhibits the others through the shared pool): a lone pool
        inhibits only itself so it is barely damped and ignites, but among competitors the strongest
        pool's spikes suppress the rest to silence -> a single sustained winner.

        PARALLEL ABOVE A MEASURED CROSSOVER. The pools partition into contiguous chunks of the sorted
        key order; each chunk is a `_settle_chunk` on its own thread. The shared inhibitory pool is a
        genuine global reduction every micro-step, so it is NOT parallelised -- one designated thread
        folds the per-chunk contributions in ascending chunk order, which visits exactly the elements
        `sum(self_i.values())` visited in exactly that order. Float addition is not associative and
        that is why the reduction stays serial: a tree-reduction here would change the numbers.

        The crossover is not written down anywhere. `region_wiring.pick_thread_count` compares this
        instance's running-EMA cost per pool against this machine's measured barrier and dispatch
        floors and the efficiency this kernel has actually achieved. Measured on a 28-core
        free-threaded 3.14.6 box: 5 and 50 pools stay serial, 500 pools 1.16-1.22x on 2 threads,
        2,000 pools 2.0x on 4, 10,000 pools 2.1-2.4x on 8, 50,000 pools 2.1-3.0x on 16. Spike counts
        AND membrane floats identical at every width and at every thread count.

        200 pools is the honest edge case and this row used to read "0.98x (gate: serial)", which is
        true only of a WARM gate. Re-measured: the FIRST call at 200 pools picks 2 threads and runs
        0.81-0.87x -- a real loss -- after which `realized_efficiency` records ~0.69 and every
        later call at that width returns to serial and stays there. So the crossover is not a clean
        step: within roughly a factor of two below it the gate pays for exactly one mispriced call
        before its own feedback corrects it. That is the designed behaviour of a measured gate
        rather than a defect, but "gate: serial" stated flatly hid a cost that is real and
        reproducible, and a cold process pays it on every fresh instance.

        The LIVE brain runs about 5 candidate pools, ~100x below the crossover, so the gate keeps
        production on the serial path today -- correctly. What would move it is P6: there are 26
        workspace instantiation sites in this tree, 11 of them private and mutually invisible, and
        consolidating them onto one shared bus is what puts hundreds of candidates in one
        competition.

        Returns {key: total_spikes}, {key: second_half_spikes} -- discrete integer spike counts."""
        keys = sorted(drives)  # deterministic evaluation order
        n = len(keys)
        if n == 0:
            return {}, {}

        from snn_data_lake.substrate import region_wiring as _rw

        nt = 1
        if n >= 2 and self._sim_per_pool_s > 0.0:
            # dedicated threads, not the shared pool: this kernel synchronises on a barrier every
            # micro-step, and a barrier across pool tasks deadlocks if the pool is saturated.
            # `sync_rounds` is the real barrier count -- 2 per micro-step (publish, then reduce),
            # plus the registration round before the loop. Passing it is what stops the gate from
            # over-threading a lock-step kernel; see pick_thread_count for the measured consequence.
            nt = min(
                _rw.pick_thread_count(
                    n * self._sim_per_pool_s,
                    mode="threads",
                    sync_rounds=2 * (self._T + 1),
                    efficiency=self._sim_par_eff,
                ),
                n,
            )

        if nt < 2:
            t0 = time.perf_counter()

            def sync_serial(si, _t):
                return _fold_parts((si,))  # the identical left fold the parallel lead performs

            _, total, late, v = self._settle_chunk(keys, drives, sync_serial)
            dt = (time.perf_counter() - t0) / n
            self._sim_per_pool_s = (
                dt
                if self._sim_per_pool_s <= 0.0
                else 0.75 * self._sim_per_pool_s + 0.25 * dt  # running baseline (P1)
            )
            self._last_v = dict(zip(keys, v))
            return dict(zip(keys, total)), dict(zip(keys, late))

        est = n * self._sim_per_pool_s
        t0 = time.perf_counter()
        out = self._simulate_parallel(keys, drives, nt, _rw)
        # Feed the ACHIEVED scaling back into the gate. A parallel run that underperformed lowers the
        # efficiency estimate and the next call at this width picks fewer threads -- or none.
        eff = _rw.realized_efficiency(
            est, time.perf_counter() - t0, nt, mode="threads", sync_rounds=2 * (self._T + 1)
        )
        self._sim_par_eff = 0.5 * self._sim_par_eff + 0.5 * eff
        return out

    def _simulate_parallel(self, keys, drives, nt, _rw):
        """Run `_settle_chunk` on `nt` dedicated threads over a contiguous partition of `keys`.

        DETERMINISM, stated as the argument rather than as a hope: (1) every chunk's mutable state is
        its own list, so no container is written by two threads; (2) the only cross-chunk value is the
        scalar `a_inh`, published behind a barrier and therefore never read while it is being written;
        (3) the reduction that produces it is a LEFT FOLD over the chunks in ascending index order,
        and the partition is contiguous, so the addition order is exactly the serial one; (4) results
        are collected into per-chunk slots and merged in ascending chunk order, so no dict is built
        from completion order. Nothing here depends on which thread finishes first.

        FAILURE IS NOT AN OPTION THIS KERNEL CAN IGNORE. Every chunk meets every other at two
        barriers per micro-step, so a chunk that raises stops arriving and its partners wait on a
        barrier that can NEVER again reach `nb` parties. Measured before the guard below existed: a
        single non-numeric drive value hung `_simulate_parallel` FOREVER -- not a slowdown, a dead
        tick, and because the worker threads are non-daemon the interpreter could not exit either.
        Each chunk therefore runs under a handler that records its exception and ABORTS both
        barriers, which converts every partner's permanent wait into a `BrokenBarrierError`; the
        caller then re-raises the ROOT cause. The serial path cannot express this failure at all,
        so it is a defect the parallel path introduced and must clean up after itself."""
        bounds = _rw.blocks(len(keys), nt)
        nb = len(bounds)
        parts = [None] * nb  # chunk index -> that chunk's si list (each written by ONE thread)
        out = [None] * nb
        shared = [0.0]
        b_reduced = threading.Barrier(nb)
        b_published = threading.Barrier(nb)

        def make_sync(tid, lead):
            def sync(si, _t):
                parts[tid] = si  # same list object every round; publish once, mutate in place
                b_published.wait()  # every chunk's si is now written for this micro-step
                if lead:
                    shared[0] = _fold_parts(
                        parts
                    )  # ONE fold definition, shared with the serial path
                b_reduced.wait()  # the fold is complete; a_inh is safe to read
                return shared[0]

            return sync

        errors = [None] * nb

        def work(tid, a, b):
            try:
                out[tid] = self._settle_chunk(keys[a:b], drives, make_sync(tid, tid == 0))
            except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised by the caller
                errors[tid] = exc
                # Release every partner stuck at a barrier this chunk will never reach again.
                # `abort()` is idempotent, so several failing chunks racing here is harmless.
                b_published.abort()
                b_reduced.abort()

        threads = [
            threading.Thread(target=work, args=(i, a, b), name=f"gw-pool-{i}")
            for i, (a, b) in enumerate(bounds)
            if i > 0
        ]
        for th in threads:
            th.start()
        work(0, bounds[0][0], bounds[0][1])  # the caller runs chunk 0 and leads the reduction
        for th in threads:
            th.join()

        # Re-raise the ROOT cause in preference to the BrokenBarrierError it induced in the others,
        # so the traceback names the real defect rather than the unblocking mechanism. Scanning in
        # ascending chunk order keeps the choice deterministic when several chunks fail.
        root = None
        for exc in errors:
            if exc is not None and not isinstance(exc, threading.BrokenBarrierError):
                root = exc
                break
        if root is None:
            for exc in errors:
                if exc is not None:
                    root = exc
                    break
        if root is not None:
            raise root

        total, late, last_v = {}, {}, {}
        for mkeys, tot, lat, v in out:  # ascending chunk order -> deterministic key order
            for i, k in enumerate(mkeys):
                total[k] = tot[i]
                late[k] = lat[i]
                last_v[k] = v[i]
        self._last_v = last_v
        return total, late

    def step(self):
        """Run ONE workspace cycle. Returns the broadcast content (or None) and clears the buffer."""
        self._drain_slots()  # concurrent specialists join the competition in a FIXED slot order
        if not self.use_ignition:
            # FAIR CONTROL: pure per-tick argmax. Whatever is loudest THIS tick wins outright.
            top = self._top_candidate()
            if top is not None:
                self._incumbent = top
            content = self.current
            self._buffer = []
            self._deliver(content)
            return content

        # --- SPIKING Global-Workspace ignition dynamics ---
        # 1. Refresh / decay each pool's cross-tick attractor memory `g`. A submission this tick is
        #    a bottom-up refresh (g rises to the salience); an unrefreshed pool DECAYS (persistence
        #    fade) and, once negligible, is dropped.
        submitted = {}
        for src, content, sal in self._buffer:
            key = (src, content)
            if key not in submitted or sal > submitted[key]:
                submitted[key] = sal
        for key in list(self._g):
            if key in submitted:
                self._g[key] = min(1.0, max(self._g[key] * self.decay, submitted[key]))
            else:
                self._g[key] = self._g[key] * self.decay
        for key, sal in submitted.items():
            if key not in self._g:
                self._g[key] = min(1.0, max(0.0, sal))
        self._g = {k: v for k, v in self._g.items() if v > 1e-6}

        # 2. Build each pool's external drive current: attractor memory + (for the incumbent) its
        #    recurrent self-excitation defence (self_boost + hysteresis, the attractor hold) + a
        #    tiny deterministic symmetry-break jitter.
        inc_key = (self._incumbent[0], self._incumbent[1]) if self._incumbent is not None else None
        drives = {}
        for key, g in self._g.items():
            d = g
            if key == inc_key:
                d += self.self_boost + self.hysteresis
            d += self._jitter_for(key)
            drives[key] = d

        # 3. Settle the spiking network; read out sustained (second-half) firing.
        total, late = self._simulate(drives)
        self._last_total = total
        self._last_late = late

        # 4. IGNITION: the winner is the pool still firing in the second half (survived competition
        #    into the high-firing attractor). None sustaining -> the workspace empties (release).
        fired = [k for k in late if late[k] > 0]
        if fired:
            winner = min(fired, key=lambda k: (-late[k], -total[k], k[0]))
            self._incumbent = (winner[0], winner[1], self._g[winner])
        else:
            self._incumbent = None

        content = self.current
        self._buffer = []
        self._deliver(content)  # BROADCAST to all subscribers
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
    means = {"perception": 0.66, "memory": 0.54, "salience": 0.54, "language": 0.53, "motor": 0.52}
    sd = 0.11
    stream = []
    for t in range(n_ticks):
        tick = {}
        for r in REGIONS:
            s = rng.gauss(means[r], sd)
            if r == "salience" and t in threat_window:
                s = 0.98  # supra-threshold threat event
            tick[r] = min(1.0, max(0.0, s))
        stream.append(tick)
    return stream


def run_stream(stream, use_ignition, **params):
    """Feed a stream through a workspace tick by tick; return the list of broadcast contents."""
    ws = GlobalWorkspace(use_ignition=use_ignition, **params)
    broadcasts = []
    for tick in stream:
        for region, salience in tick.items():
            ws.submit(region, region, salience)  # content == region label
        broadcasts.append(ws.step())
    return broadcasts


def count_switches(seq):
    """Number of ticks whose broadcast content differs from the previous tick (flicker measure)."""
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def _fmt(seq):
    """Compact ASCII trace: initials of the broadcast region per tick, '.' for an empty workspace."""
    return "".join((c[0].upper() if c is not None else ".") for c in seq)


# ======================================================================================================
# DEEPENING: multi-item BINDING + the SPIKING-vs-RATE control (does synchrony buy anything over a rate
# integrator with the same hysteresis?). Self-contained LIF; used only by the self-check.
# ======================================================================================================
_BIND_TSLOT = 6  # micro-steps per gamma slot (one bound object)
_BIND_LEAK = 0.25  # binding-LIF leak per step
_BIND_VTH = 1.0  # binding-LIF threshold
_BIND_IGNITE = 0.6  # all-or-none ignition threshold for the binding broadcast


def _bind_step(v, i):
    """One leaky integrate-and-fire step for the binding demo. Returns (new_v, spike)."""
    v = v * (1.0 - _BIND_LEAK) + i
    if v >= _BIND_VTH:
        return 0.0, 1  # subtract-threshold reset -> discrete spike
    return (v if v > 0.0 else 0.0), 0


def bind_objects(n_color, n_shape, k, rng):
    """K objects, each a (color, shape) pair with DISTINCT colors and DISTINCT shapes -- so the part
    IDENTITIES are unambiguous and only the BINDING (which color goes with which shape) is at risk."""
    return list(zip(rng.sample(range(n_color), k), rng.sample(range(n_shape), k)))


def bind_broadcast_spiking(objs, n_color, n_shape, drive=1.0, sustain=4):
    """Ignite (if drive clears the threshold), then broadcast the objects PHASE-MULTIPLEXED: object k
    drives its color and shape LIF with the SAME current in gamma slot k, so they spike SYNCHRONOUSLY
    (bound); re-entry sustains each after its slot. Returns (ignited, raster[t]=(fired_colors,fired_shapes))."""
    if drive < _BIND_IGNITE:
        return False, []  # subliminal: no global broadcast
    vc = [0.0] * n_color
    vs = [0.0] * n_shape
    raster = []
    T = len(objs) * _BIND_TSLOT + sustain
    for t in range(T):
        slot = t // _BIND_TSLOT
        fc, fs = [], []
        for k, (c, s) in enumerate(objs):
            in_slot = k == slot
            cur = (
                drive if in_slot else drive * 0.55
            )  # re-entry current sustains (weaker), does not re-ignite
            vc[c], sc = _bind_step(vc[c], cur)
            vs[s], ss = _bind_step(vs[s], cur)
            if sc:
                fc.append(c)
            if ss:
                fs.append(s)
        raster.append((fc, fs))
    return True, raster


def bind_decode_spiking(raster):
    """Reader: color+shape that fire SYNCHRONOUSLY (same step) are bound. Returns the recovered pairs."""
    pairs = set()
    for fc, fs in raster:
        for c in fc:
            for s in fs:
                pairs.add((c, s))
    return pairs


def bind_broadcast_rate(objs, n_color, n_shape, drive=1.0):
    """Rate control: same all-or-none ignition, but a SUPERPOSED activation vector -- every object's
    color and shape channel is on with equal magnitude, NO timing, so the pairing is not represented."""
    if drive < _BIND_IGNITE:
        return False, [], []
    color_act = [0.0] * n_color
    shape_act = [0.0] * n_shape
    for c, s in objs:
        color_act[c] += drive
        shape_act[s] += drive
    return True, color_act, shape_act


def bind_decode_rate(color_act, shape_act, k):
    """Reader for the rate broadcast: recover the K active colors and K active shapes (magnitudes say
    which are on), but the PAIRING is absent -> pair them canonically (sorted); right only by luck."""
    cols = sorted(sorted(range(len(color_act)), key=lambda i: -color_act[i])[:k])
    shps = sorted(sorted(range(len(shape_act)), key=lambda i: -shape_act[i])[:k])
    return set(zip(cols, shps))


def _bind_pair_acc(true_pairs, got_pairs):
    return len(set(true_pairs) & set(got_pairs)) / len(true_pairs)


def main():

    # ---- FAIR CONTROL: same stream, spiking ignition ON vs OFF (pure per-tick argmax) ----------
    n_ticks = 48
    stream = build_stream(n_ticks, seed=0)
    with_ig = run_stream(stream, use_ignition=True, seed=0)
    without = run_stream(stream, use_ignition=False, seed=0)
    sw_with = count_switches(with_ig)
    sw_without = count_switches(without)

    print("=" * 74)
    print("SPIKING GLOBAL WORKSPACE -- LIF pools -> compete -> ignite -> broadcast -> decay")
    print("=" * 74)
    print("regions: perception(P) memory(M) salience/threat(S) language(L) motor(O); '.' = empty")
    print("threat burst (region 'salience') at ticks 30-35\n")
    print(f"WITH ignition  (spiking LIF + inhibition): {_fmt(with_ig)}  switches={sw_with}")
    print(f"WITHOUT (pure per-tick argmax control)   : {_fmt(without)}  switches={sw_without}")
    print(
        f"\n-> the spiking ignition non-linearity cut workspace switches from {sw_without} (flicker) "
        f"to {sw_with} (stable)"
    )

    # threat must actually break through: a genuinely stronger supra-threshold signal ignites.
    threat_broadcast = any(c == "salience" for c in with_ig[30:36])

    # ---- ANCHOR (i) BOTTLENECK: many supra-threshold candidates -> exactly ONE seizes the slot --
    seen = []
    b_ws = GlobalWorkspace(seed=0)
    b_ws.subscribe(seen.append)
    for r in REGIONS:
        b_ws.submit(r, r, 0.90)  # five simultaneous supra-threshold posts
    b_content = b_ws.step()
    bottleneck_ok = b_content is not None and len(seen) == 1 and seen[0] == b_content

    # ---- SPIKING WTA: distinct-strength competitors -> the shared inhibitory pool SILENCES the
    #      weaker ones (0 sustained spikes) and only the strongest ignites (genuine competition) ---
    g_ws = GlobalWorkspace(seed=0)
    g_ws.submit("strong", "strong", 0.90)
    g_ws.submit("mid", "mid", 0.68)
    g_ws.submit("weak", "weak", 0.62)
    g_content = g_ws.step()
    wta_winner_late = g_ws._last_late["strong", "strong"]
    wta_losers_late = g_ws._last_late["mid", "mid"] + g_ws._last_late["weak", "weak"]
    wta_ok = g_content == "strong" and wta_winner_late > 0 and wta_losers_late == 0

    # ---- ANCHOR (ii) ALL-OR-NONE IGNITION: sub-threshold stays local, supra-threshold ignites --
    w_ws = GlobalWorkspace(seed=0)
    w_ws.submit("perception", "weak", 0.40)  # below rheobase -> never fires
    sub = w_ws.step()
    weak_spikes = sum(w_ws._last_total.values())  # a weak submission emits NO spikes
    s_ws = GlobalWorkspace(seed=0)
    s_ws.submit("perception", "strong", 0.80)  # above rheobase -> ignites
    supra = s_ws.step()
    strong_spikes = sum(s_ws._last_total.values())  # a strong submission DOES spike
    # not linear averaging: two sub-threshold posts do NOT sum into an ignition.
    n_ws = GlobalWorkspace(seed=0)
    n_ws.submit("memory", "half_a", 0.50)
    n_ws.submit("language", "half_b", 0.50)
    no_sum = n_ws.step()
    no_sum_spikes = sum(n_ws._last_total.values())

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

    # ---- ANCHOR (iv) PERSISTENCE + DECAY: ignite, then no input -> attractor holds then fades ---
    d_ws = GlobalWorkspace(seed=0)
    d_ws.submit("memory", "trace", 0.95)
    d_ws.step()  # ignite
    life = []
    for _ in range(8):
        life.append(d_ws.step())  # no new submissions -> attractor decays
    persisted = sum(1 for c in life if c == "trace")
    released = life[-1] is None
    print("\npersistence/decay (ignite 'memory' at 0.95, then silence):")
    print(
        f"   post-ignition life trace: {_fmt(life)}  "
        f"(held 'trace' {persisted} ticks, released={released})"
    )

    # ---- DEEPENING: multi-item BINDING, spiking-vs-rate control (isolates whether SPIKES matter) --
    brng = random.Random(0)
    bind = {}
    for k in (1, 2, 4):
        sp = rt = 0.0
        trials = 120
        for _ in range(trials):
            objs = bind_objects(8, 8, k, brng)
            _, r = bind_broadcast_spiking(objs, 8, 8)
            sp += _bind_pair_acc(objs, bind_decode_spiking(r))
            _, ca, sa = bind_broadcast_rate(objs, 8, 8)
            rt += _bind_pair_acc(objs, bind_decode_rate(ca, sa, k))
        bind[k] = (sp / trials, rt / trials)
    # all-or-none: the binding broadcast ignites on strong drive, stays subliminal below threshold.
    bind_ig_strong, _ = bind_broadcast_spiking(bind_objects(8, 8, 2, brng), 8, 8, drive=1.0)
    bind_ig_weak, _ = bind_broadcast_spiking(bind_objects(8, 8, 2, brng), 8, 8, drive=0.4)
    print("\nbinding (spiking vs rate control, pair-accuracy):")
    for k in (1, 2, 4):
        s, r = bind[k]
        tag = "tie (spikes not needed)" if abs(s - r) < 0.05 else f"spikes +{(s - r) * 100:.0f} pts"
        print(f"   K={k}: spiking {s:.0%}  rate {r:.0%}  -> {tag}")

    # ---- LIVE CONTRACT: the exact submit/step sequences autonomous_loop + live_brain rely on ----
    c_ws = GlobalWorkspace(seed=0)
    c_ws.submit("perception", "unit0", 0.6)  # safe tick: perception holds the slot...
    c_ws.submit("salience", "threat", 0.2)  # ...vs a low background threat
    contract_safe = c_ws.step()  # -> broadcasts the current percept
    c_ws.submit("salience", "threat", 0.95)  # danger tick: only the threat is posted
    contract_danger = c_ws.step()  # -> the threat ignites + is broadcast

    # ---- DETERMINISM: identical stream+seed -> identical broadcasts and counts -----------------
    with_ig2 = run_stream(build_stream(n_ticks, seed=0), use_ignition=True, seed=0)
    deterministic = with_ig2 == with_ig and count_switches(with_ig2) == sw_with

    print(
        f"\nlive contract : safe tick broadcasts {contract_safe!r}; "
        f"proximity threat ignites {contract_danger!r}"
    )
    print(
        f"spiking WTA   : competitors 0.90/0.68/0.62 -> winner {g_content!r} keeps firing "
        f"(late spikes {wta_winner_late}); losers silenced (late spikes {wta_losers_late})"
    )
    print(
        f"spiking readout: weak@0.40 spikes={weak_spikes}, strong@0.80 spikes={strong_spikes} "
        f"(discrete integer spike events)"
    )

    # ============================ genuine self-checks (not rigged) ==============================
    # (a) FUNCTION -----------------------------------------------------------------------------
    # the spiking ignition stabilises the workspace vs the pure-argmax control, by a real margin.
    assert sw_without > sw_with, f"argmax control must flicker more ({sw_without} vs {sw_with})"
    assert sw_without >= 3 * max(sw_with, 1), (
        f"stabilisation must be large, not marginal: with={sw_with} without={sw_without}"
    )
    assert sw_with <= 6, (
        f"spiking workspace must be stable WITH ignition (few switches), got {sw_with}"
    )
    # bottleneck: exactly one content broadcast despite five supra-threshold candidates.
    assert bottleneck_ok, (
        "exactly one content must seize + be broadcast through the single-slot bottleneck"
    )
    assert isinstance(b_content, str), "the single broadcast content must be one region, not a set"
    # genuine spiking competition: among distinct-strength pools, the shared inhibitory pool leaves
    # only the strongest in sustained firing; the weaker pools are silenced (not merely out-ranked).
    assert wta_ok, (
        "the shared inhibitory pool must silence weaker pools so only the strongest ignites"
    )
    # all-or-none ignition non-linearity: sub-threshold excluded, supra admitted, no summation.
    assert sub is None, f"a sub-threshold candidate must NOT ignite the workspace (got {sub!r})"
    assert supra == "strong", f"a supra-threshold candidate must ignite (got {supra!r})"
    assert no_sum is None, (
        "two sub-threshold candidates must NOT sum into an ignition (all-or-none)"
    )
    # global reach: every subscriber received the identical broadcast sequence.
    assert reach_ok, "every subscriber must receive the identical broadcast (global reach)"
    # persistence + decay: the content is sustained several ticks, then fades below threshold.
    assert persisted >= 2, f"ignited content must persist for several ticks (held {persisted})"
    assert released, "unrefreshed content must eventually decay below threshold and be released"
    # a genuinely stronger, supra-threshold signal DID take over (workspace is not frozen).
    assert threat_broadcast, "the supra-threshold threat burst must break through and be broadcast"
    # the LIVE contract used by autonomous_loop (sec 8) + live_brain must hold exactly.
    assert contract_safe is not None, (
        "a safe tick must broadcast the current percept (ws_safe is not None)"
    )
    assert contract_danger == "threat", (
        "a proximity threat must ignite + be broadcast (ws_danger == 'threat')"
    )

    # (b) GENUINELY SPIKING --------------------------------------------------------------------
    # discrete spikes: a sub-threshold pool emits ZERO spikes; a supra-threshold pool DOES spike.
    assert all(isinstance(n, int) for n in s_ws._last_total.values()), (
        "spikes must be discrete integer events"
    )
    assert weak_spikes == 0, (
        f"a sub-threshold (0.40) pool must emit NO spikes (stays local), got {weak_spikes}"
    )
    assert no_sum_spikes == 0, (
        f"two 0.50 pools must both stay sub-threshold (no spikes), got {no_sum_spikes}"
    )
    assert strong_spikes > 0, (
        f"a supra-threshold (0.80) pool must emit discrete spikes, got {strong_spikes}"
    )
    # winner by SUSTAINED firing: the ignited pool is still spiking in the second half of the window,
    # having silenced its competitors through the shared inhibitory pool.
    assert s_ws._last_late["perception", "strong"] > 0, (
        "the ignited winner must show sustained (2nd-half) firing"
    )
    assert wta_winner_late > 0 and wta_losers_late == 0, (
        "sustained firing must belong to ONE pool (WTA)"
    )

    # (c) DETERMINISM --------------------------------------------------------------------------
    assert deterministic, "same seed must reproduce identical broadcasts and switch counts"

    # (c2) PARALLEL SETTLING is BYTE-IDENTICAL to serial settling ------------------------------
    # The gate is measured, so force both paths explicitly here rather than hoping to land on one.
    p_ws = GlobalWorkspace(seed=0)
    prng = random.Random(4242)
    p_drives = {(f"s{i}", f"c{i}"): prng.uniform(0.3, 1.2) for i in range(240)}
    p_ws._sim_per_pool_s = 0.0  # cold EMA -> the gate must refuse to guess and run SERIAL
    ser_tot, ser_late = p_ws._simulate(p_drives)
    assert p_ws._sim_per_pool_s > 0.0, "a serial settle must record its own per-pool cost (P1)"
    from snn_data_lake.substrate import region_wiring as _rw

    assert _rw.pick_thread_count(0.0, mode="threads") == 1, (
        "with no measured cost the gate must return 1 (serial), never a guess"
    )
    ser_v = dict(p_ws._last_v)
    par_tot, par_late = p_ws._simulate_parallel(sorted(p_drives), p_drives, 4, _rw)
    par_v = dict(p_ws._last_v)
    assert par_tot == ser_tot and par_late == ser_late, (
        "parallel settling must reproduce the serial spike counts EXACTLY"
    )
    # The FLOAT trajectory, not the integer counts, is what polices the reduction. Counts are far
    # too coarse: a reassociated `a_inh` moves the membranes in the last bits and the spike counts
    # absorb it silently -- measured, which is why this assert exists in this form.
    assert par_v == ser_v, (
        "parallel settling must reproduce the serial MEMBRANE FLOATS bit for bit -- this is what "
        "fails the moment the shared inhibitory reduction is reassociated"
    )
    assert list(par_tot) == list(ser_tot), (
        "the parallel path must return keys in sorted-chunk order, not thread-completion order"
    )
    # ...and the reassociation this guards against is REAL, not hypothetical: on this very data the
    # per-chunk-subtotal form gives a different float. If this ever stops differing, the fold above
    # is protecting nothing and the assert on `par_v` is decoration.
    adversarial = [[1e16, 1.0], [-1e16, 1.0]]
    assert _fold_parts(adversarial) == 2.0 and sum(sum(c) for c in adversarial) == 0.0, (
        _fold_parts(adversarial),
        sum(sum(c) for c in adversarial),
    )
    # The reduction must equal the FLAT sum for every partitioning -- that is the whole determinism
    # claim -- and must differ from both wrong forms, or the claim is vacuous.
    rrng = random.Random(31)
    flat = [rrng.uniform(-2.0, 2.0) for _ in range(240)]
    naive = 0.0
    for x in flat:
        naive += x
    for k in (1, 2, 3, 7, 16):
        step = max(1, len(flat) // k)
        chunks = [flat[i : i + step] for i in range(0, len(flat), step)]
        assert _fold_parts(chunks) == sum(flat), (
            f"the reduction must be independent of the partition ({k} chunks disagreed)"
        )
    assert naive != sum(flat), (
        "an `acc += x` fold must actually differ from sum() here -- if it stops differing, this "
        "module's determinism argument has lost the case it was written to catch"
    )
    assert _fold_parts(()) == 0 and _fold_parts(([],)) == 0, "empty fold must be zero"

    # A FROZEN REFERENCE of the pre-parallel dict-based settle -- an independent second
    # implementation, so `_settle_chunk`'s arithmetic is checked against something other than
    # itself. Serial-vs-parallel equality can never catch a change made to the SHARED kernel; this
    # can.
    def _reference_settle(ws, drives):
        keys = sorted(drives)
        v = {k: 0.0 for k in keys}
        trace = {k: 0.0 for k in keys}
        self_i = {k: 0.0 for k in keys}
        a_inh = 0.0
        total = {k: 0 for k in keys}
        late = {k: 0 for k in keys}
        half = ws._T // 2
        for t in range(ws._T):
            spikes = {}
            for k in keys:
                lateral = ws._G_INH * max(0.0, a_inh - self_i[k])
                cur = drives[k] + ws._W_SELF * trace[k] - lateral
                v[k] += cur - ws._LEAK * v[k]
                if v[k] < 0.0:
                    v[k] = 0.0
                s = 0
                if v[k] >= ws._v_th:
                    v[k] -= ws._v_th
                    s = 1
                    total[k] += 1
                    if t >= half:
                        late[k] += 1
                    trace[k] += 1.0
                spikes[k] = s
                trace[k] *= ws._TAU_SELF
            for k in keys:
                self_i[k] += ws._K_INH * spikes[k] - ws._LEAK_INH * self_i[k]
                if self_i[k] < 0.0:
                    self_i[k] = 0.0
            a_inh = sum(self_i.values())
        return total, late, v

    ref_tot, ref_late, ref_v = _reference_settle(p_ws, p_drives)
    assert (ref_tot, ref_late) == (ser_tot, ser_late), "settle must match the reference counts"
    assert ref_v == ser_v, (
        "the partitioned settle kernel must reproduce the reference dict implementation's MEMBRANE "
        "FLOATS exactly -- any re-association inside the LIF update fails here"
    )
    assert any(x != 0.0 for x in ser_v.values()), (
        "the membrane comparison must be non-vacuous (all-zero membranes would compare equal to "
        "anything that also zeroes them)"
    )
    # zero / one: the partition boundaries the gate must survive
    assert p_ws._simulate({}) == ({}, {}), "an empty workspace must settle to empty, not raise"
    one = {("solo", "solo"): 0.9}
    assert p_ws._simulate_parallel(sorted(one), one, 1, _rw) == p_ws._simulate(one), (
        "a single pool must give the same answer however it is partitioned"
    )

    # REACHABLE RED for the worker-failure path. Every chunk meets every other at two barriers per
    # micro-step, so a chunk that raises stops arriving and its partners wait on a barrier that can
    # never again reach `nb` parties. MEASURED before the abort/re-raise guard existed: this exact
    # call hung FOREVER, and because the workers are non-daemon the interpreter could not exit
    # either. Reaching this assert at all is the proof it no longer can.
    poisoned = {(f"s{i:02d}", f"c{i:02d}"): 0.9 for i in range(16)}
    poisoned[("s09", "c09")] = "not-a-float"  # index 9 -> chunk 2 of 4, i.e. a WORKER, not the lead
    assert _rw.blocks(16, 4)[2] == (8, 12), _rw.blocks(16, 4)
    try:
        GlobalWorkspace(seed=0)._simulate_parallel(sorted(poisoned), poisoned, 4, _rw)
        raise AssertionError("a chunk that raises must PROPAGATE -- never hang, never return")
    except TypeError as exc:
        # the ROOT cause, not the BrokenBarrierError the abort induced in the three live chunks
        assert "str" in str(exc), f"the root cause must survive, got {exc!r}"

    # (c3) CONCURRENT SUBMISSION is DETERMINISTIC ----------------------------------------------
    # RED for the ordering claim: buffer order IS observable, so the slot ordering is load-bearing.
    o1, o2 = GlobalWorkspace(seed=0), GlobalWorkspace(seed=0)
    o1.submit("alpha", "A", 0.8)
    o1.submit("beta", "B", 0.8)
    o2.submit("beta", "B", 0.8)  # same two candidates, opposite arrival order
    o2.submit("alpha", "A", 0.8)
    o1.step()
    o2.step()
    assert o1._jit != o2._jit, (
        "arrival order must be OBSERVABLE (it seeds the symmetry-break jitter) -- if this ever "
        "passes trivially, the slot ordering below is protecting nothing and should be deleted"
    )

    def _slot_trial(threaded):
        w = GlobalWorkspace(seed=0)
        n_slots = 8
        w.open_slots(n_slots)
        r = random.Random(11)
        payload = [
            [(f"s{t}_{j}", f"c{t}_{j}", r.uniform(0.3, 1.1)) for j in range(5)]
            for t in range(n_slots)
        ]
        if threaded:
            gate = threading.Barrier(n_slots)

            def sub(t):
                gate.wait()  # release all submitters together to maximise real interleaving
                for src, con, sal in payload[t]:
                    w.submit_from(t, src, con, sal)

            ths = [threading.Thread(target=sub, args=(t,)) for t in range(n_slots)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
        else:
            for t in range(n_slots):
                for src, con, sal in payload[t]:
                    w.submit_from(t, src, con, sal)
        return w.step(), tuple(sorted(w._jit.items()))

    # The merge ORDER is the contract, and it is asserted directly -- 'both paths agree' would still
    # hold if the drain sorted or reversed the buffers, so agreement alone does not pin the order.
    ord_ws = GlobalWorkspace(seed=0)
    ord_ws.open_slots(3)
    ord_ws.submit_from(2, "c", "c1", 0.9)  # slot 2 submits FIRST in real time...
    ord_ws.submit_from(0, "a", "a1", 0.1)
    ord_ws.submit_from(2, "c", "c2", 0.8)
    ord_ws.submit_from(1, "b", "b1", 0.5)
    ord_ws._drain_slots()
    assert ord_ws._buffer == [
        ("a", "a1", 0.1),  # ...but slot 0 lands first: ascending SLOT, then program order
        ("b", "b1", 0.5),
        ("c", "c1", 0.9),
        ("c", "c2", 0.8),
    ], f"slot drain must be ascending-slot then program order, got {ord_ws._buffer}"
    assert ord_ws._slot_buf == [[], [], []], "drained buffers must be cleared, not re-delivered"
    assert GlobalWorkspace(seed=0).open_slots(0) == 0, "zero slots is legal and drains to nothing"

    ref_trial = _slot_trial(False)
    n_trials = 40
    all_same = all(_slot_trial(True) == ref_trial for _ in range(n_trials))
    assert all_same, (
        "concurrent submission through the slot buffers must equal the one-at-a-time result on "
        "EVERY trial -- a single mismatch means thread order reached the competition"
    )
    print(
        f"\nparallel     : settling 240 pools on 4 threads reproduces the serial spike counts "
        f"exactly; {n_trials} threaded submission trials all matched the serial order"
    )

    # (d) DEEPENING: the SPIKING-vs-RATE control is a PERMANENT GUARD -- honest about where spikes help.
    assert bind_ig_strong and not bind_ig_weak, "binding broadcast ignition must be all-or-none"
    assert bind[1][0] - bind[1][1] < 0.05, (
        f"at K=1 spikes must NOT beat rate (single-item binding needs no spikes): "
        f"{bind[1][0]:.2f} vs {bind[1][1]:.2f}"
    )
    assert bind[2][0] > bind[2][1] + 0.15, (
        f"at K=2 the spiking workspace must bind better than the rate control (synchrony is load-bearing): "
        f"{bind[2][0]:.2f} vs {bind[2][1]:.2f}"
    )
    assert bind[4][0] > 0.9, f"spiking multi-item binding must stay high at K=4 ({bind[4][0]:.2f})"

    print(
        "\nself-check OK: "
        f"spiking ignition switches {sw_with} vs argmax {sw_without} (>=3x stabler); "
        f"single-slot bottleneck; spiking WTA silences losers; all-or-none (sub spikes={weak_spikes}, "
        f"supra spikes={strong_spikes}); global-reach; persist {persisted}->release; threat ignited; "
        f"live-contract holds; deterministic; BINDING spiking-vs-rate control honest "
        f"(K1 tie {bind[1][0]:.0%}/{bind[1][1]:.0%}, K2 spikes {bind[2][0]:.0%}>{bind[2][1]:.0%})"
    )


if __name__ == "__main__":
    main()
