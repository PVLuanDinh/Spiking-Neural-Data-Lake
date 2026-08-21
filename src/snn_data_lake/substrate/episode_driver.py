"""Stimulus port + EpisodeDriver (specs/022 T035, SYNTHESIS L1.5): `tick(stimulus=None)`
on the substrate tick, the TRIAL SCHEDULE (fixation / stimulus / delay / probe / saccade),
and G5-replayable trials.

WHY THIS OBJECT EXISTS (SYNTHESIS: 'every L7 instrument that sweeps or perturbs drives
runs THROUGH this object; without it the battery has streams but no trials'): the kernel
takes raw external drive (`KernelTick.tick(ext_q)`), but an experiment is not a stream --
it is TRIALS: a fixation REFERENCE WINDOW (pre-stimulus steady state -- on a
self-sustaining subject the network never returns to silence, so 'baseline' honestly
means the reverberating reference the stimulus response is measured AGAINST, standard
PSTH practice), a sample presentation, a delay whose floor is DERIVED from the subject's
own delay line (max_delay + 1, so sample volleys land before the probe window), a probe,
a response window. The driver owns that
structure once, so every instrument gets the same trials instead of five ad-hoc stream
generators. FORWARD DISPOSITIONS, dated: the L7 instruments route their sweeps through
this driver as each lands (T033's ignition scorer predates it and runs its own drive
sweep over a DIFFERENT subject class -- the re-entry workspace, not the kernel -- so it
is not retrofitted here; the kernel-subject instruments from T077+ are the clients);
T052 adopts ESPP with THIS driver's same/different oracle as the supervisory source.

THE ORACLE (ESPP's same/different source, named for T052): each trial presents a SAMPLE
pattern, then after the delay a PROBE that is byte-identical to the sample or a fresh
draw; `trial['oracle']` is 'same' / 'different', ground truth by construction -- the
supervisor a temporal-stream rule trains against without any labeler.

G5-REPLAYABLE, positions not seeds (the T017 replay gate's contract): the driver draws
every pattern from ONE named numpy Generator exposed via `rng_streams()`, so
`replay_gate.capture(kt, driver.rng_streams())` snapshots the trial stream's POSITION and
a restored run continues with byte-identical trials -- proven in the self-check with the
fresh-seed divergence control.

THE STIMULUS PORT'S MANIFEST DISPOSITION, stated (T026 precedent): the per-modality
afferent TypedPorts are T036's deliverable (`substrate/afferent_boundary.py`, the pinned
modality inventory); this module's `tick(stimulus=None)` is the KERNEL-side API those
ports feed, so no manifest row is declared here -- declaring the boundary twice would be
the second-mechanism shape P2 forbids.

PROTOCOL PARAMETERS, disclosed as such: phase durations and pattern sparsity are the
EXPERIMENT'S independent variables (constructor parameters with documented defaults),
not brain behaviour -- the same class as a trial design in any lab protocol. What is
DERIVED is everything the kernel owns: quantization uses the kernel's OWN frac_bits
(`tick()`'s contract: the caller quantizes -- P1, no second fixed-point convention).

# distinct-from: substrate/kernel_tick.py -- the kernel this drives; tick(ext_q) is its
#   port and nothing here re-implements propagation. distinct-from: substrate/
#   replay_gate.py -- the G5 capture/restore this joins via rng_streams() (and its
#   _driven_run is a raw STREAM, not trials). distinct-from:
#   thalamus/ignition_instrument.py -- an L7 scorer over the RE-ENTRY workspace with its
#   own drive sweep; this drives the KERNEL and owns trials, not verdicts. distinct-from:
#   substrate/afferent_boundary.py (T036) -- the declared modality boundary that will
#   FEED this port. distinct-from: midbrain/sc_deep_layers.py and midbrain/
#   superior_colliculus.py -- the tree's GAZE fixation/saccade MECHANICS (omnipause gate,
#   target selection); this module's 'fixation'/'saccade' are trial-PHASE labels over the
#   substrate tick, not eye movements. distinct-from: cortex/autapse_neuron.py -- its
#   _delayed_match_stream is the same PARADIGM family (delayed recall) over a single
#   autaptic LIF with a recalled-bit label; different subject class, different label,
#   disclosed rather than merged. All sites read.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.episode_driver
"""

import numpy as np

PHASES = ("fixation", "stimulus", "delay", "probe", "saccade")


class EpisodeDriver:
    """Owns the trial schedule over one KernelTick. One named RNG stream, one trial
    structure, one oracle -- see the module docstring for every contract."""

    def __init__(
        self,
        n_units,
        seed=0,
        fixation_ticks=3,
        stimulus_ticks=4,
        delay_ticks=3,
        probe_ticks=4,
        saccade_ticks=2,
        active_frac=0.25,  # MEASURED, not picked: at 0.125 an in-trial stimulus moved 3 of
        # 160 fired-slots on the reverberating demo kernel (2% -- inert); at 0.25 it moved
        # 43 of 165 (26%). Pattern SIZE is the in-regime charge lever; drive magnitude
        # above threshold measured irrelevant (1.4 vs 4.0 byte-identical deltas).
        drive=1.4,  # ABSOLUTE kernel drive units (the port never reads the subject's
        # threshold -- deriving drive relative to thr is the CALLER'S experiment). The
        # default cites replay_gate._driven_run's own literal 1.4 against ITS thr=1.0
        # demo kernel; a different subject needs its own value, stated per experiment.
        p_same=0.5,
    ):
        self.n_units = int(n_units)
        if self.n_units < 2:
            raise ValueError(
                f"n_units must be >= 2 (got {self.n_units}) -- one unit has one pattern"
            )
        self._gen = np.random.default_rng(int(seed))  # THE stream G5 snapshots by position
        self.durations = {
            "fixation": int(fixation_ticks),
            "stimulus": int(stimulus_ticks),
            "delay": int(delay_ticks),
            "probe": int(probe_ticks),
            "saccade": int(saccade_ticks),
        }
        if any(v < 1 for v in self.durations.values()):
            raise ValueError(f"every phase needs >= 1 tick, got {self.durations}")
        af = float(active_frac)
        if not (0.0 < af <= 1.0):
            raise ValueError(
                f"active_frac must be in (0, 1], got {af} -- it is the protocol variable "
                "that decides pattern sparsity, and out-of-domain values were measured "
                "being silently absorbed (<=0 -> k=1, the collision-prone floor) or "
                "deferring a crash into mid-experiment (>1)"
            )
        self.k_active = max(1, int(self.n_units * af))
        if self.k_active >= self.n_units:
            raise ValueError(
                f"k_active ({self.k_active}) must be < n_units ({self.n_units}): at full "
                "support only ONE pattern exists, so a 'different' probe is "
                "unconstructible and the oracle would be forced to lie"
            )
        self.drive = float(drive)
        if not (self.drive > 0.0 and np.isfinite(self.drive)):
            raise ValueError(f"drive must be finite and > 0, got {self.drive}")
        self.p_same = float(p_same)
        if not (0.0 < self.p_same < 1.0):
            raise ValueError(
                f"p_same must be in (0, 1) -- at 0 or 1 the oracle is a constant and "
                f"ESPP's supervisor has one label; got {self.p_same}"
            )
        self.trials_run = 0

    # -- G5 ---------------------------------------------------------------------------------------
    # distinct-from: substrate/replay_gate.py -- capture/restore live THERE; these two
    #   methods only hand it this driver's named stream and seat what it returns. Read.
    def rng_streams(self):
        """The named streams for replay_gate.capture -- position-snapshotted, never re-seeded."""
        return {"episode_driver": self._gen}

    # distinct-from: substrate/replay_gate.py -- restore() rebuilds and VERIFIES the
    #   streams; this method only SEATS what it returned, plus the trial counter. Read.
    def adopt_streams(self, streams, trials_run):
        """Seat the generators replay_gate.restore returns (rebuilt AT the captured
        positions) AND the trial counter captured beside the blob. Both are REQUIRED:
        a missing stream is the silent re-seed G5 forbids, and a counter left at its
        constructed value re-bases every trial index at the replay boundary -- the
        review measured the restored continuation's records diverging in exactly the
        field the first self-check excluded from its equality ('the restored brain
        quietly walks a different trajectory while looking perfectly healthy')."""
        if "episode_driver" not in streams:
            raise KeyError(
                "restore returned no 'episode_driver' stream -- refusing a silent re-seed"
            )
        self._gen = streams["episode_driver"]
        self.trials_run = int(trials_run)

    # -- the stimulus port ------------------------------------------------------------------------
    # distinct-from: substrate/kernel_tick.py -- tick(ext_q) is the KERNEL'S port; this
    #   method only transduces floats onto its grid and forwards. Read.
    def tick(self, kt, stimulus=None):
        """ONE substrate tick through the stimulus port. `stimulus` is a length-n_units
        float array IN THE KERNEL'S OWN DRIVE UNITS (or None for a spontaneous tick):
        the port is a faithful TRANSDUCER -- floats onto the kernel's fixed-point grid
        using the KERNEL'S own frac_bits, per tick()'s contract ('the caller quantizes
        with its own frac_bits' -- ONE fixed-point convention, the kernel's). It refuses
        non-finite values and wrong shapes; it does NOT police magnitude -- drive
        relative to threshold is the experiment's meaning, and a first draft that
        clamped stimuli to [0, 1] could never fire a thr=1.0 kernel at all (measured:
        quiet 0 vs driven 0). Normalized-modality semantics are T036's boundary, not
        this port's. Returns the fired indices exactly as the kernel does."""
        if stimulus is None:
            return kt.tick(None)
        s = np.asarray(stimulus, dtype=np.float64)
        if s.shape != (self.n_units,):
            raise ValueError(f"stimulus must be shape ({self.n_units},), got {s.shape}")
        if not np.isfinite(s).all():
            raise ValueError("stimulus must be finite -- NaN/inf is a computation error upstream")
        # REPRESENTABILITY is the transducer's own business (magnitude MEANING stays the
        # caller's): a finite value whose scaled form exceeds int64 casts to INT64_MIN --
        # a maximally EXCITATORY request delivered as maximal INHIBITION, with only a
        # RuntimeWarning (measured: stimulus 1e30 -> 0 fired, reading as 'saturation' to
        # any upward drive sweep). Bound at 2**62 / grid: one bit of headroom for the
        # kernel's own accumulator additions on top of the injected drive.
        bound = float(2**62) / (1 << kt.syn.frac_bits)
        amax = float(np.abs(s).max())
        if amax > bound:
            raise ValueError(
                f"stimulus magnitude {amax:g} exceeds the kernel's representable drive "
                f"({bound:g} at frac_bits={kt.syn.frac_bits}) -- refusing the int64 wrap "
                "that would deliver excitation as maximal inhibition"
            )
        ext_q = np.round(s * (1 << kt.syn.frac_bits)).astype(np.int64)
        return kt.tick(ext_q)

    def _pattern(self):
        """One sparse stimulus pattern: k_active units at `drive`, drawn from THE stream."""
        s = np.zeros(self.n_units, dtype=np.float64)
        idx = self._gen.choice(self.n_units, size=self.k_active, replace=False)
        s[idx] = self.drive
        return s, tuple(int(i) for i in np.sort(idx))

    def run_trial(self, kt):
        """ONE trial: fixation (spontaneous) -> stimulus (SAMPLE) -> delay (spontaneous)
        -> probe (SAME sample or a fresh DIFFERENT draw) -> saccade (spontaneous response
        window). Returns the trial record: oracle, both pattern ids, per-phase fired
        counts, and the phase-by-phase fired trains (the analysis surface an instrument
        windows over)."""
        sample, sample_id = self._pattern()
        same = bool(self._gen.random() < self.p_same)
        if same:
            probe, probe_id = sample.copy(), sample_id  # copy: the record must not alias
        else:
            # THE ORACLE MUST NOT LIE (review BLOCKER): an independent draw can collide
            # with the sample (certain at k==n, 12.5% at k=1/n=8 -- both measured), and a
            # 'different' label over identical patterns is a supervisor training ESPP on a
            # falsehood. Redraw until distinct -- guaranteed terminable because the
            # constructor refuses k_active >= n_units -- with a loud bound so a broken
            # stream cannot spin silently.
            for _attempt in range(64):
                probe, probe_id = self._pattern()
                if probe_id != sample_id:
                    break
            else:
                raise RuntimeError(
                    f"64 redraws failed to produce a distinct probe (k={self.k_active} of "
                    f"{self.n_units}) -- the stream or the pattern space is broken"
                )
        # PHASE MINIMUMS ARE THE SUBJECT'S, derived at run time (review: the shipped
        # delay of 3 ticks against the demo kernel's max_delay=5 meant sample volleys
        # were STILL ARRIVING during the probe -- the one thing a delayed-match protocol
        # must prevent). The delay and fixation floors are the kernel's own delay line
        # plus one: every in-flight spike from the previous phase has landed before the
        # next phase's window opens. The configured duration still rules when longer.
        floor = int(kt.syn.max_delay) + 1
        eff = dict(self.durations)
        eff["delay"] = max(eff["delay"], floor)
        eff["fixation"] = max(eff["fixation"], floor)
        rec = {
            "trial": self.trials_run,
            "oracle": "same" if same else "different",
            "sample_id": sample_id,
            "probe_id": probe_id,
            "durations_effective": dict(eff),
            "fired": {},
            "counts": {},
        }
        stim_of = {"stimulus": sample, "probe": probe}
        for phase in PHASES:
            trains = []
            n_units_fired = 0
            for _ in range(eff[phase]):
                fired = self.tick(kt, stim_of.get(phase))
                trains.append(fired.tobytes())
                n_units_fired += int(fired.size)
            rec["fired"][phase] = tuple(trains)
            rec["counts"][phase] = n_units_fired  # from fired.size, never a byte-width guess
        self.trials_run += 1
        return rec

    def run(self, kt, n_trials):
        """`n_trials` consecutive trials -- the stream every instrument shares."""
        return [self.run_trial(kt) for _ in range(int(n_trials))]


def main():
    from snn_data_lake.substrate import replay_gate

    print("=" * 74)
    print("EPISODE DRIVER -- trials (fixation/stimulus/delay/probe/saccade) over the kernel")
    print("=" * 74)

    _st, _sy, kt = replay_gate._build(seed=5, n=64)
    drv = EpisodeDriver(n_units=64, seed=11)

    # (1) THE PORT DRIVES: a stimulus tick fires more than a spontaneous one on a quiet
    # kernel, and the quantization is the kernel's own grid (charge + P1, one command).
    quiet = kt.tick(None)
    s = np.zeros(64)
    s[:8] = drv.drive  # the protocol's own supra-threshold drive
    driven = drv.tick(kt, s)
    assert len(driven) > len(quiet), (
        f"the stimulus port must DRIVE the kernel (quiet {len(quiet)} vs driven {len(driven)})"
    )

    # (1b) THE PORT DRIVES IN REGIME (review BLOCKER: the virgin-kernel charge above
    # exists only at trial 0 tick 0; inside a reverberating trial the old 12.5%-sparse
    # protocol moved 3 of 160 fired-slots -- inert; active_frac=0.25 is the measured
    # remedy). Neuter A/B at the SHIPPED protocol, per instrument seed: identical
    # 5-trial history in both arms, then one trial with the sample+probe WITHHELD.
    #
    # THE BOUND IS THE MEASURED SEPARATION, DISCLOSED IN FULL -- on this self-sustaining
    # demo subject the stimulus is a WEAK PERTURBATION, and pretending otherwise is how
    # the first draft's 43-slot 'margin' happened (it was delay-bleed: sample volleys
    # still landing in the probe window before the derived delay floor existed). The
    # 7-seed sweep of this arm reads [17, 14, 24, 5, 11, 12, 7] differing slots (min 5)
    # against the inert 0.125 regime's [4, 0, 3] (max 4). Three seeds run here; each
    # must clear the inert regime's measured ceiling, and their sum must clear it with
    # an order of magnitude -- both bounds from the sweep, neither wished.
    _inert_ceiling = 4
    in_trial_diffs = []
    for _seed in (7, 11, 23):
        _stA, _syA, ktA = replay_gate._build(seed=5, n=64)
        _stB, _syB, ktB = replay_gate._build(seed=5, n=64)
        dA = EpisodeDriver(n_units=64, seed=_seed)
        dB = EpisodeDriver(n_units=64, seed=_seed)
        for _ in range(5):
            dA.run_trial(ktA)
            dB.run_trial(ktB)
        live_rec = dA.run_trial(ktA)
        _orig_tick = dB.tick
        dB.tick = lambda kt_, stimulus=None, _o=_orig_tick: _o(kt_, None)  # NEUTER: port dead
        dead_rec = dB.run_trial(ktB)
        dB.tick = _orig_tick
        d = 0
        for ph in ("stimulus", "probe"):
            for ta, tb in zip(live_rec["fired"][ph], dead_rec["fired"][ph]):
                fa = set(np.frombuffer(ta, dtype=np.int64))
                fb = set(np.frombuffer(tb, dtype=np.int64))
                d += len(fa ^ fb)
        in_trial_diffs.append(d)
        assert d > _inert_ceiling, (
            f"IN-REGIME charge failed at seed {_seed}: withholding the stimulus moved "
            f"{d} fired-slots (<= the inert regime's measured ceiling {_inert_ceiling}) "
            "-- the trials are kernel reverberation wearing an experiment's clothes"
        )
    assert sum(in_trial_diffs) >= 10 * _inert_ceiling, (
        f"IN-REGIME charge is collectively marginal: {in_trial_diffs} -- re-measure the "
        "protocol before trusting trials on this subject"
    )

    # (2) TRIAL STRUCTURE + ORACLE: both labels occur, phases carry their EFFECTIVE ticks
    # (the delay/fixation floors are the kernel's own max_delay + 1 -- derived, so sample
    # volleys land before the probe window opens), counts are consumed against the
    # trains, and a DIFFERENT trial's ids always differ (the redraw at work).
    _st2, _sy2, kt2 = replay_gate._build(seed=5, n=64)
    drv2 = EpisodeDriver(n_units=64, seed=11)
    trials = drv2.run(kt2, 12)
    labels = {t["oracle"] for t in trials}
    assert labels == {"same", "different"}, f"the oracle must produce BOTH labels, got {labels}"
    floor = int(kt2.syn.max_delay) + 1
    for t in trials:
        assert t["durations_effective"]["delay"] >= floor > drv2.durations["delay"], (
            "the delay floor must be DERIVED from the subject's own delay line and must "
            "actually bind on the demo config (else this assert tests nothing)"
        )
        for phase in PHASES:
            assert len(t["fired"][phase]) == t["durations_effective"][phase]
            assert t["counts"][phase] == sum(
                len(np.frombuffer(b, dtype=np.int64)) for b in t["fired"][phase]
            ), "counts must equal the trains they summarize -- itemsize-honest"
        if t["oracle"] == "same":
            assert t["probe_id"] == t["sample_id"]
        else:
            assert t["probe_id"] != t["sample_id"]

    # (2b) THE ORACLE CANNOT LIE, exercised in the collision-PRONE regime (review: at
    # k=1 of 8 an unguarded independent draw lied on 20 of 185 different-trials; the
    # k>=n certain-lie regime now refuses at construction).
    _st6, _sy6, kt6 = replay_gate._build(seed=5, n=8)  # a subject SIZED to the driver
    drv6 = EpisodeDriver(n_units=8, seed=7, active_frac=0.125)  # k=1 of 8: 12.5% collision
    lies = 0
    for t in drv6.run(kt6, 50):
        if t["oracle"] == "different" and t["probe_id"] == t["sample_id"]:
            lies += 1
    assert lies == 0, f"the redraw must make 'different' TRUE on every trial, got {lies} lies"
    try:
        EpisodeDriver(n_units=16, active_frac=1.0)
        raise AssertionError("k_active == n_units must refuse -- 'different' unconstructible")
    except ValueError:
        pass  # the RED, reached: the certain-lie regime cannot be built

    # (3) G5: capture MID-STREAM (blob + the trial counter beside it), run the
    # continuation, restore, re-run -- the FULL records byte-identical, trial indices
    # included (the first draft compared only 'fired', excluding exactly the field that
    # diverged); a FRESH-SEED arm diverges (positions carry the information).
    _st3, _sy3, kt3 = replay_gate._build(seed=5, n=64)
    drv3 = EpisodeDriver(n_units=64, seed=11)
    drv3.run(kt3, 3)
    blob = replay_gate.capture(kt3, drv3.rng_streams())
    trials_at_capture = drv3.trials_run
    cont_a = drv3.run(kt3, 3)
    _st4, _sy4, kt4 = replay_gate._build(seed=5, n=64)
    drv4 = EpisodeDriver(n_units=64, seed=303)  # DIFFERENT seed: the restore must not care
    drv4.adopt_streams(replay_gate.restore(blob, kt4), trials_at_capture)
    cont_b = drv4.run(kt4, 3)
    assert cont_a == cont_b, (
        "a restored run's trials must be byte-identical to the original continuation -- "
        "FULL records, trial indices included (G5)"
    )
    _st5, _sy5, kt5 = replay_gate._build(seed=5, n=64)
    drv5 = EpisodeDriver(n_units=64, seed=999)  # fresh stream = the divergence control
    replay_gate.restore(blob, kt5)  # kernel restored, driver stream DELIBERATELY not
    cont_c = drv5.run(kt5, 3)
    assert [t["fired"] for t in cont_c] != [t["fired"] for t in cont_a], (
        "the fresh-seed control must DIVERGE -- if it matches, the trials never depended "
        "on the stream and the G5 claim is vacuous"
    )

    # (4) refusals: bad shape, non-finite, and the int64-wrap cliff (a finite 1e30 used
    # to cast to INT64_MIN -- excitation delivered as maximal inhibition, reading as
    # 'saturation' to an upward drive sweep).
    for bad in (np.zeros(3), np.full(64, np.nan), np.full(64, 1e30)):
        try:
            drv.tick(kt, bad)
            raise AssertionError("out-of-contract stimulus must refuse")
        except ValueError:
            pass
    try:
        EpisodeDriver(n_units=64, p_same=1.0)
        raise AssertionError("a constant oracle must refuse -- ESPP's supervisor needs two labels")
    except ValueError:
        pass

    print(
        f"\nself-check OK: the port drives at trial 0 (quiet {len(quiet)} -> {len(driven)}) "
        f"AND in regime (withholding the stimulus mid-stream moved {in_trial_diffs} "
        f"fired-slots per seed, each > the inert ceiling {_inert_ceiling}); 12 trials, "
        f"both oracle labels, "
        f"subject-derived delay floor binding, counts train-honest; the redraw kept the "
        f"oracle truthful through 50 collision-prone trials and the certain-lie config "
        f"refuses at construction; G5 restores FULL records byte-identically (indices "
        f"included) with the fresh-seed control diverging; shape/NaN/int64-wrap refusals "
        f"reached. Protocol parameters are the experiment's independent variables, "
        f"disclosed and policed."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
