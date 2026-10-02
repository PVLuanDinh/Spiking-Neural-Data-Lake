r"""
L11 / SI-6 -- WIRE the proven-but-dead spiking SLEEP-REPLAY into a LIVE quiescent-tick phase.

WHAT THIS IS. sleep/sleep_replay.py already PROVES that a "sleep" phase (noise-driven
attractor reactivation + covariance-STDP) restores a spiking memory that a later, interfering
skill would otherwise overwrite -- but nothing in the live loop ever CALLS it: it is dead code.
This module promotes it to live use. A WAKE phase learns skills (stores overlapping cell
assemblies in the reused AttractorNet); a QUIESCENT (sleep) phase then runs sleep_replay's
spontaneous replay + covariance-STDP to CONSOLIDATE, with the rate sharp-wave-ripple ReplayBuffer
(experience_replay.py) running ALONGSIDE as old-skill spaced practice. A single knob (`sleep=`)
turns the quiescent phase OFF -- that is the no-sleep control. SI-1 spiking learning-progress
(spiking_curiosity.SpikingCuriosityDrive) prioritizes the rate spaced-practice replay.

REUSED, NOT REIMPLEMENTED:
  * sleep/sleep_replay.AttractorNet  -- the proven noise-driven spiking replay + covariance
    STDP (store / sleep / recall / accuracy). THIS is the dead code being promoted; every spiking
    weight change happens inside its unmodified methods, so the retention numbers are byte-identical
    to sleep_replay's own SLEEP vs NO-SLEEP conditions (same store order, same sleep seeds, same
    held-out accuracy seed_base).
  * sleep/experience_replay.ReplayBuffer -- the rate SWR value replay, ADDED ALONGSIDE (one
    per skill), never ripped out.
  * faculties/spiking_curiosity.SpikingCuriosityDrive -- SI-1 spiking learning-progress, used
    only to PRIORITIZE the rate replay budget.
  * substrate/_mathutils.mean.

HONEST FRAMING -- THIS RUNG IS FRAGILE; the claims are scoped hard:
  LIMIT (capacity). The attractor holds ~0.14N patterns, so at N=24 only a FEW skills coexist; the
    restoration of an overwritten skill is PARTIAL and SEED-FRAGILE (sleep_replay: restored A lands
    0.46-0.79 across seeds, around chance, never back to 1.00; B's cost ranges 0.50-1.00). We
    therefore ASSERT ONLY the sleep-vs-no-sleep GAP and that no-sleep FORGETS; ABSOLUTE retention is
    REPORTED as fragile (printed "fragile"), never asserted to a fixed level, and its seed-sensitivity
    is shown across seeds.
  CIRCULARITY TRAP (avoided). Prioritizing replay BY learning-progress and then MEASURING retention
    WITH learning-progress would be circular. The retention metric here is LP-INDEPENDENT: held-out
    pattern-completion accuracy of the old skill (AttractorNet.accuracy, fresh random partial cues) --
    a task-performance measure, NOT the curiosity LP signal that drives prioritization.
  OVERCLAIM AVOIDED. This is NOT "systems consolidation on the live loop." The only proof is a
    hand-tuned 2-task toy with deliberately OVERLAPPING assemblies; real live skills are not clean
    assemblies and are not expected to behave this way. It demonstrates the Golden-et-al. mechanism
    wired live, nothing more.
  PRIORITIZATION IS AN UNPROVEN ADD. UNIFORM replay already earns the whole gap. LP-priority is
    applied to the rate spaced-practice replay only; it leaves the SPIKING retention driver untouched,
    so it provably adds NOTHING to retention here (shown as an equality), and it yields no reliable
    win on its own value metric at this scale. We show uniform vs priority and assert NOTHING about
    priority being better.

Seeded via random + zlib.crc32 for skill-buffer seeds (no wall-clock, no global RNG),
ASCII-only output, deterministic. Additive core primitive; touches no safety-envelope file.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sleep.sleep_consolidation
"""

import random
import zlib

from snn_data_lake.faculties.spiking_curiosity import SpikingCuriosityDrive
from snn_data_lake.infra._mathutils import mean
from snn_data_lake.sleep.experience_replay import ReplayBuffer
from snn_data_lake.sleep.sleep_replay import (
    CUE_FRAC,
    P_A0,
    P_A1,
    P_B0,
    P_B1,
    SLEEP_EPISODES,
    STORE_REPS_A,
    STORE_REPS_B,
    AttractorNet,
    K,
    _overlap,
)

# ---- the two skills = the proven overlapping assemblies from sleep_replay (B overwrites A) ----
SKILL_A = [P_A0, P_A1]  # learned first
SKILL_B = [P_B0, P_B1]  # learned second; its assemblies overlap A -> interference

# ---- rate SWR value replay (experience_replay) that runs ALONGSIDE the spiking sleep ----
_ACTIONS = ("good", "bad")  # a 2-action toy value table per skill
_REWARD = {"good": 1.0, "bad": -1.0}  # 'good' is the rewarded action
_LR_VALUE = 0.3  # RPE learning rate (mirrors the live loop / experience_replay demo)
_REPLAY_BUDGET = 16  # total rate-replay batch per quiescent phase (split across skills)

# ---- SI-1 learning-progress: a graded completion-error probe (1 - attractor overlap) is fed to the ----
# ---- SpikingCuriosityDrive per storage rep; its PEAK learning-progress during wake is the priority. ----
# ---- In THIS consolidation regime LP is near-zero and noisy (a clean skill masters in one rep -> LP  ----
# ---- ~ 0; a struggling skill's post-interference error is high/noisy -> anti-noisy-TV also reads ~0), ----
# ---- so priority barely differs from uniform, and where it does it does not favor the at-risk skill.  ----

# ---- a-priori thresholds (the SEED-ROBUST claims only; absolute retention is NOT asserted) ----
CHANCE = 0.50
FORGET_MARGIN = 0.30  # no-sleep must forget A by at least this vs right-after-A
RETAIN_MARGIN = 0.20  # sleep must beat no-sleep on A by at least this (obs gap ~ +0.42)


class SleepConsolidator:
    """Live wake/quiescent lifecycle around the reused spiking sleep-replay.

    wake(skill, assemblies, reps) -- WAKE: store the skill's assemblies as attractors (identical
      storage trajectory to sleep_replay), feed a falling completion-error trace to the SI-1 spiking
      learning-progress drive, and fill the skill's rate SWR ReplayBuffer + value table.
    quiescent() -- QUIESCENT (sleep): if the sleep knob is ON, run the reused noise-driven spiking
      replay + covariance-STDP (this is what earns the retention gap); THEN run the rate SWR replay
      alongside as old-skill spaced practice, either UNIFORMLY or PRIORITIZED by SI-1 LP.
    retention(assemblies) -- LP-INDEPENDENT held-out pattern-completion accuracy of a skill.

    The `sleep` knob (constructor) toggles the quiescent spiking phase -> the no-sleep control.
    The `prioritized` knob toggles LP-priority on the RATE replay only (leaves the spiking driver
    untouched). Deterministic: every stochastic draw comes from a per-instance random.Random or an
    explicit seed; skill-buffer seeds come from zlib.crc32 (never builtin hash())."""

    def __init__(self, seed, sleep=True, prioritized=False, episodes=SLEEP_EPISODES):
        self.seed = int(seed)
        self.sleep_on = bool(sleep)  # OFF -> the no-sleep control
        self.prioritized = bool(prioritized)  # LP-priority on the rate replay (unproven add)
        self.episodes = int(episodes)  # spiking sleep episodes per quiescent phase
        self.net = AttractorNet(seed)  # REUSED proven spiking attractor + sleep-replay
        self.curiosity = SpikingCuriosityDrive(seed=seed)  # REUSED SI-1 spiking learning-progress
        self.skills = {}  # skill_id -> assemblies (order preserved)
        self.buffers = {}  # skill_id -> ReplayBuffer (rate SWR, ALONGSIDE)
        self.q = {}  # skill_id -> {action: value} (rate value table)
        self._lp = {}  # skill_id -> PEAK SI-1 LP during wake (priority weight)
        self._phase = 0  # quiescent-phase counter -> distinct sleep seeds

    def _completion_error(self, assemblies, target_idx, seed):
        """Graded held-out completion error for the LP trace: cue a hard partial subset of assembly
        `target_idx`, let the attractor complete, and return 1 - overlap of the completed state with the
        target (0 = perfect completion). Reuses sleep_replay._overlap. Read-only (recall never mutates
        the weights) so it does NOT perturb the proven retention path."""
        rng = random.Random(seed)
        p = assemblies[target_idx]
        ncue = max(1, int(K * CUE_FRAC))
        cue = set(rng.sample(sorted(p), ncue))
        return 1.0 - _overlap(self.net.recall(cue), p)

    def wake(self, skill_id, assemblies, reps):
        """WAKE phase: practice (store) a skill. Storing an assembly `reps` times one rep at a time is
        byte-identical to sleep_replay's store(pattern, reps); between reps we read a completion error
        (falls as the assembly strengthens) and feed it to the SI-1 spiking LP drive, and we log a rate
        SWR transition + live RPE update into the skill's value table. Returns None."""
        self.skills[skill_id] = list(assemblies)
        bseed = zlib.crc32(
            skill_id.encode("ascii")
        )  # deterministic per-skill buffer seed (not hash())
        self.buffers[skill_id] = ReplayBuffer(capacity=256, seed=bseed)
        self.q.setdefault(skill_id, dict.fromkeys(_ACTIONS, 0.0))
        arng = random.Random(bseed ^ self.seed)  # per-skill action stream (deterministic)
        peak_lp = 0.0
        step = 0
        for ai, p in enumerate(assemblies):
            for _ in range(reps):
                self.net.store(p, 1)  # one storage rep (identical trajectory to store(p, reps))
                err = self._completion_error(assemblies, ai, self.seed * 1000 + step)
                self.curiosity.observe(
                    skill_id, err
                )  # SI-1 spiking LP: falling error -> positive learning progress
                peak_lp = max(
                    peak_lp, self.curiosity.learning_progress(skill_id)
                )  # capture LP while still learning
                a = arng.choice(_ACTIONS)
                r = _REWARD[a]
                self.q[skill_id][a] += _LR_VALUE * (
                    r - self.q[skill_id][a]
                )  # live RPE (mirrors the loop)
                self.buffers[skill_id].add(skill_id, a, r)  # rate SWR transition (ALONGSIDE)
                step += 1
        self._lp[skill_id] = peak_lp  # this skill's priority weight (peak wake learning-progress)

    def _replay_alloc(self, skills):
        """How to split this phase's rate-replay budget across the skills seen so far. UNIFORM =
        equal batches (this is what pairs with the gap-earning spiking sleep). PRIORITIZED = batches
        proportional to each skill's PEAK SI-1 learning-progress captured during wake (fallback to
        uniform when all LP ~ 0)."""
        n = len(skills)
        if not self.prioritized:
            return [_REPLAY_BUDGET // n] * n
        lps = [max(self._lp.get(s, 0.0), 0.0) for s in skills]
        tot = sum(lps)
        if tot <= 1e-9:
            return [_REPLAY_BUDGET // n] * n
        return [int(round(_REPLAY_BUDGET * lp / tot)) for lp in lps]

    def quiescent(self):
        """QUIESCENT (sleep) phase. If the sleep knob is ON, run the REUSED noise-driven spiking replay
        + covariance-STDP (AttractorNet.sleep) -- the ONLY thing that changes the spiking weights and
        the ONLY thing that earns the retention gap. THEN, in every condition, run the rate SWR replay
        ALONGSIDE as old-skill spaced practice (uniform, or LP-prioritized). Returns None."""
        self._phase += 1
        if self.sleep_on:
            # distinct sleep seed per phase (matches sleep_replay's seed*7+1, seed*7+2, ...)
            self.net.sleep(self.seed * 7 + self._phase, episodes=self.episodes)
        seen = list(self.skills)
        if not seen:
            return
        for s, b in zip(seen, self._replay_alloc(seen)):
            if b > 0:
                # replay looks up q by unit=skill_id; the whole table is passed (buffer s holds only s's rows)
                self.buffers[s].replay(self.q, _LR_VALUE, batch=b)

    def retention(self, assemblies):
        """LP-INDEPENDENT retention metric: held-out pattern-completion accuracy of the skill via
        AttractorNet.accuracy (fresh random partial cues). This is task performance, NOT the curiosity
        LP signal that drives prioritization -> no circularity between the priority and the metric."""
        return self.net.accuracy(assemblies, 100)

    def value_margin(self):
        """Mean rate-value decisiveness across skills: value('good') - best other action. The rate SWR
        replay's own metric (NOT the retention metric); used only for the priority honest-negative."""
        if not self.q:
            return 0.0
        margins = []
        for row in self.q.values():
            good = row["good"]
            other = max(v for a, v in row.items() if a != "good")
            margins.append(good - other)
        return mean(margins)


def run(seed, sleep, prioritized=False):
    """One full deterministic wake/quiescent experiment: learn skill A, sleep, learn the interfering
    skill B, sleep, then measure A's LP-independent held-out retention (+ B's, + the rate value margin).
    With sleep OFF the quiescent phases skip the spiking replay -> A is overwritten (the control)."""
    c = SleepConsolidator(seed, sleep=sleep, prioritized=prioritized)
    c.wake("A", SKILL_A, STORE_REPS_A)  # WAKE: learn skill A
    after_a = c.retention(SKILL_A)  # A well-formed right after learning
    c.quiescent()  # SLEEP 1: consolidate A (spiking) + spaced value replay
    c.wake(
        "B", SKILL_B, STORE_REPS_B
    )  # WAKE: learn B -> overwrites A's shared synapses (interference)
    c.quiescent()  # SLEEP 2: spontaneous replay of both + spaced value replay
    skills = list(c.skills)
    return {
        "after_a": after_a,
        "ret_a": c.retention(SKILL_A),  # LP-INDEPENDENT retention of the OLD skill A
        "ret_b": c.retention(SKILL_B),
        "val": c.value_margin(),
        "lp": {s: round(c._lp[s], 3) for s in skills},  # captured peak SI-1 LP per skill
        "alloc": dict(
            zip(skills, c._replay_alloc(skills))
        ),  # rate-replay budget split this run used
    }


def main():
    print("=" * 82)
    print(
        "SLEEP CONSOLIDATION (L11 / SI-6) -- proven spiking sleep-replay WIRED to a live quiescent tick"
    )
    print(
        "wake: store skill A -> sleep -> store interfering skill B -> sleep;  knob turns sleep OFF"
    )
    print("=" * 82)

    # ---- run every condition across a few seeds: no-sleep control / UNIFORM sleep / LP-PRIORITY sleep ----
    seeds = (0, 1, 2)
    res = {
        s: {
            "ns": run(s, sleep=False),
            "sl": run(s, sleep=True),  # UNIFORM replay -- this earns the gap
            "pr": run(s, sleep=True, prioritized=True),
        }  # LP-PRIORITY replay (unproven rate-side add)
        for s in seeds
    }
    ns, sl, pr = res[0]["ns"], res[0]["sl"], res[0]["pr"]  # primary seed 0
    gap = sl["ret_a"] - ns["ret_a"]

    print(
        "retention = LP-INDEPENDENT held-out pattern-completion accuracy of OLD skill A "
        f"(chance {CHANCE:.0%})"
    )
    print("")
    print("  seed 0                  A retention     B retention   value margin")
    print(f"  after learning A      :   {ns['after_a']:.2f} (fragile)      -")
    print(
        f"  no-sleep  (control)   :   {ns['ret_a']:.2f} (fragile)   {ns['ret_b']:.2f}"
        f"        {ns['val']:+.2f}   <- A OVERWRITTEN by B"
    )
    print(
        f"  SLEEP uniform         :   {sl['ret_a']:.2f} (fragile)   {sl['ret_b']:.2f}"
        f"        {sl['val']:+.2f}   <- A partially restored (>> no-sleep)"
    )
    print(
        f"  SLEEP LP-priority     :   {pr['ret_a']:.2f} (fragile)   {pr['ret_b']:.2f}"
        f"        {pr['val']:+.2f}   <- rate-side only; retention UNCHANGED"
    )
    print("")
    print(f"  ASSERTED GAP (sleep - no-sleep) : {gap:+.2f}   (threshold >= +{RETAIN_MARGIN})")

    # ---- (3) FRAGILITY: absolute retention is seed-sensitive; the GAP direction is robust ----
    print("")
    print(
        "  FRAGILITY (absolute retention is SEED-DEPENDENT, reported not asserted; gap is robust):"
    )
    for s in seeds:
        rn, rl = res[s]["ns"]["ret_a"], res[s]["sl"]["ret_a"]
        print(
            f"    seed {s}: no-sleep A={rn:.2f}  sleep A={rl:.2f} (both fragile)   gap {rl - rn:+.2f}"
        )

    # ---- (4) HONEST NEGATIVE: LP-priority does NOT beat UNIFORM at this scale (assert nothing) ----
    print("")
    print("  HONEST NEGATIVE -- LP-priority vs UNIFORM replay (assert NOTHING about priority):")
    print(
        "    per seed: captured SI-1 LP -> priority rate-replay alloc | retention (unif==prio) | value marg"
    )
    for s in seeds:
        u, p = res[s]["sl"], res[s]["pr"]
        reall = "reallocated" if p["alloc"] != u["alloc"] else "no LP signal -> uniform"
        print(
            f"    seed {s}: LP {p['lp']} -> alloc {p['alloc']} ({reall}) | "
            f"ret {u['ret_a']:.2f}=={p['ret_a']:.2f} | val {u['val']:+.3f} vs {p['val']:+.3f}"
        )
    print(
        "    => LP is ~0 in the consolidation regime (A masters instantly, B's post-interference error is"
    )
    print(
        "       noisy-flat -- anti-noisy-TV reads both ~0); where LP does differ (seed 2) priority reallocates"
    )
    print(
        "       but NEVER beats uniform: it only touches the rate replay, never the spiking retention driver."
    )

    # ---- determinism ----
    sl_again = run(0, sleep=True)
    identical = sl == sl_again
    print("")
    print(f"  determinism : same seed -> identical result = {identical}")
    print("=" * 82)

    # ---- self-check: assert ONLY the seed-robust claims (the GAP, and that no-sleep forgets) ----
    assert ns["ret_a"] <= ns["after_a"] - FORGET_MARGIN, (
        f"CONTROL: no-sleep must FORGET A (A {ns['after_a']:.2f} -> {ns['ret_a']:.2f}, "
        f"drop >= {FORGET_MARGIN})"
    )
    assert gap >= RETAIN_MARGIN, (
        f"THE RESULT: the live quiescent sleep phase must retain A markedly better than no-sleep "
        f"({sl['ret_a']:.2f} !>= {ns['ret_a']:.2f} + {RETAIN_MARGIN})"
    )
    for s in seeds:
        g = res[s]["sl"]["ret_a"] - res[s]["ns"]["ret_a"]
        assert g > 0.0, (
            f"the sleep-vs-no-sleep gap direction must be robust (seed {s} gap {g:+.2f} !> 0)"
        )
    # priority provably adds nothing to the LP-INDEPENDENT retention (it never touches the spiking driver)
    for s in seeds:
        u, p = res[s]["sl"], res[s]["pr"]
        assert abs(u["ret_a"] - p["ret_a"]) < 1e-12, (
            f"LP-priority must leave the spiking retention driver untouched (seed {s} retention unif==prio)"
        )
    assert identical, "determinism: same seed must reproduce the result byte-identically"
    # NOTE: NO assertion on absolute retention levels (fragile) and NO assertion that priority > uniform.

    print(
        f"self-check OK: proven spiking sleep-replay WIRED live -- no-sleep forgot A "
        f"{ns['after_a']:.2f}->{ns['ret_a']:.2f}; the quiescent sleep phase recovered it to "
        f"{sl['ret_a']:.2f} (gap {gap:+.2f}, LP-INDEPENDENT metric, PARTIAL/fragile);"
    )
    print(
        f"  gap direction robust across seeds 0-2; LP-priority left retention unchanged "
        f"(uniform {sl['ret_a']:.2f} == priority {pr['ret_a']:.2f}) and won nothing on value margin "
        f"({sl['val']:+.3f} vs {pr['val']:+.3f}); deterministic"
    )


if __name__ == "__main__":
    main()
