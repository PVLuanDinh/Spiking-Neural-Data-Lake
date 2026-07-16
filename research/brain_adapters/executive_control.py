r"""
executive_control -- Slice 2 of wiring the REAL brain into the developmental loop.

Three thin research-tier adapters over shipped, zero-dep `snn_data_lake.*` modules that give
the developmental / NG-SNN training loop an EXECUTIVE control layer built from real brain parts
instead of hand-rolled scaffolding:

  * DopamineModulator  wraps midbrain.dopamine_nuclei.DopamineNuclei -- turns a reward-prediction
    error over the loop's own COMPETENCE GAIN into a bounded learning-rate multiplier (dopamine
    gates plasticity: a positive surprise -> bigger learning step; a negative surprise -> smaller).
  * WorkspaceBroadcast wraps thalamus.global_workspace.GlobalWorkspace -- posts the current
    objective / competence summary into the single-slot spiking workspace so a strong-enough goal
    IGNITES and becomes globally available (coordination + logging; NOT an accuracy lever).
  * ExecutivePlanner  wraps cortex.prefrontal_executive.PrefrontalExecutive -- PLANS the next digit
    to study by rolling a simple competence-dynamics forward model through the spiking WTA decision
    attractor, picking the digit that most raises a competence goal_value (weak-but-learnable first).

This is RESEARCH-tier code: it lives outside src/, imports and composes the shipped zero-dep
modules, and never edits the core, run_selfchecks, or pyproject. Pure standard library.
Deterministic (per-instance seeds; each wrapped module is seeded from the same seed). ASCII-only.

Integration shapes passed to PrefrontalExecutive.plan (the main integration risk -- taken from the
module's OWN plan_scene / goal_value demo, not guessed):
  state      = the length-10 competence vector as an immutable tuple (predictor returns a fresh one).
  options    = the candidate digit indices (ints), used both as WTA options AND as predictor commands.
  predictor  = an OBJECT with .predict(state, command) (NOT a bare callable -- plan() calls
               predictor.predict(s, command)); returns the predicted next competence tuple.
  goal_value = a bare function goal_value(state) -> scalar (plan() calls it directly).
  plan(...)  returns (goal_index, mean_rt); goal_index indexes into `options`, so the chosen digit is
               options[goal_index].

  PYTHONPATH=src python research/brain_adapters/executive_control.py
"""
import random

from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.midbrain.dopamine_nuclei import DopamineNuclei
from snn_data_lake.thalamus.global_workspace import GlobalWorkspace


# ---------------------------------------------------------------------------------------------
# 1. DopamineModulator -- RPE over competence gain -> bounded learning-rate multiplier
# ---------------------------------------------------------------------------------------------
class DopamineModulator:
    """Wrap the midbrain dopamine nuclei to gate the loop's learning rate by reward-prediction error.

    modulate(competence_gain, expected_gain=None) computes an RPE = competence_gain - baseline (an
    explicit expected_gain when given, else an internal running mean of past gains), feeds it to the
    nucleus's da_level(rpe) (dopamine level in [0, 1]), and maps that level to a bounded learning-rate
    multiplier. Higher dopamine (positive surprise) -> a bigger learning step; lower dopamine (negative
    surprise) -> a smaller step. Deterministic given seed.

    Learning-rate map (documented, bounded):  lr_gain = LR_MIN + (LR_MAX - LR_MIN) * da_level
    with LR_MIN=0.5, LR_MAX=2.0. Because da_level is clipped to [0, 1], lr_gain is guaranteed to lie
    in [0.5, 2.0]. With the nucleus defaults (tonic_baseline=0.2), an RPE of 0 gives da_level 0.2 ->
    lr_gain 0.725 (near the low end); a strong positive surprise saturates da_level -> lr_gain -> 2.0."""

    LR_MIN = 0.5
    LR_MAX = 2.0

    def __init__(self, seed=0):
        self.seed = seed
        self._nuc = DopamineNuclei(seed=seed)   # the real spiking RPE source, seeded identically
        self._gain_sum = 0.0                     # running-mean baseline of observed competence gains
        self._gain_n = 0
        self.da = self._nuc.tonic_baseline       # last dopamine level (for logging); starts tonic

    def _baseline(self):
        """Running mean of past competence gains (0.0 before any gain has been observed)."""
        return self._gain_sum / self._gain_n if self._gain_n else 0.0

    def modulate(self, competence_gain, expected_gain=None):
        """RPE = competence_gain - (expected_gain or running-mean baseline) -> bounded lr multiplier.

        The RPE is fed to the real nucleus's da_level(rpe); the resulting dopamine level in [0, 1] maps
        linearly to a learning-rate multiplier in [LR_MIN, LR_MAX]. The observed gain always updates the
        internal running-mean baseline so future RPEs are measured against realised progress."""
        baseline = self._baseline() if expected_gain is None else float(expected_gain)
        rpe = float(competence_gain) - baseline
        level = self._nuc.da_level(rpe)                          # real dopamine level in [0, 1]
        self.da = level
        # update the running-mean baseline with the realised gain (so RPE stays meaningful over time)
        self._gain_sum += float(competence_gain)
        self._gain_n += 1
        return self.LR_MIN + (self.LR_MAX - self.LR_MIN) * level


# ---------------------------------------------------------------------------------------------
# 2. WorkspaceBroadcast -- ignite the current goal into the single-slot spiking workspace
# ---------------------------------------------------------------------------------------------
class WorkspaceBroadcast:
    """Wrap the spiking global workspace to broadcast the loop's current objective / competence summary.

    broadcast(source, content, salience) submits the content into this tick's competition at the given
    salience (the external drive current on that content's LIF pool), runs ONE settle-to-ignition cycle
    with step(), and returns the globally-available content (GlobalWorkspace.current). A supra-threshold
    salience ignites into sustained firing and is broadcast; a sub-threshold one stays local (returns
    None). This is coordination + logging (the "conscious" broadcast of the current goal), NOT an
    accuracy lever. self.ignited records whether the submitted content actually seized the slot."""

    def __init__(self, seed=0):
        self.seed = seed
        self._ws = GlobalWorkspace(seed=seed)   # the real spiking workspace, seeded identically
        self.ignited = False                     # did the last submission ignite + get broadcast?

    def broadcast(self, source, content, salience):
        """Submit content at `salience`, step the workspace once, return the broadcast content (or None).

        Sets self.ignited to whether THIS content ignited (won the single slot). Sub-threshold salience
        (below the workspace rheobase / ignition_threshold) never fires, so it returns None and does not
        ignite; a supra-threshold salience ignites and is globally broadcast."""
        self._ws.submit(source, content, salience)
        current = self._ws.step()                # runs one LIF settle-to-ignition cycle; returns current
        self.ignited = current is not None and current == content
        return current


# ---------------------------------------------------------------------------------------------
# 3. ExecutivePlanner -- plan the next study objective over competence dynamics
# ---------------------------------------------------------------------------------------------
class _CompetenceDynamics:
    """The forward model injected into PrefrontalExecutive.plan -- an OBJECT with .predict(state, cmd).

    plan() calls predictor.predict(s, command); it does NOT call a bare function, so the competence
    model must be wrapped in an object exposing .predict. Studying digit `command` raises that digit's
    competence toward 1.0 by an amount proportional to its current HEADROOM (1 - competence[command]) --
    bigger expected gain where the agent is weak-but-learnable; every other digit is unchanged. Returns
    the predicted next competence as a tuple of the SAME length (immutable, so rollouts do not alias)."""

    def __init__(self, gain=0.5):
        self.gain = gain    # fraction of the remaining headroom closed per study step

    def predict(self, state, command):
        nxt = list(state)
        c = nxt[command]
        nxt[command] = c + self.gain * (1.0 - c)     # close a fraction of the headroom of the studied digit
        return tuple(nxt)


def _goal_value(state):
    """The scalar the executive MAXIMISES over a planned rollout terminal state.

    goal_value(state) = mean(state) + min(state): the mean term rewards raising EVERYONE, and the min
    (floor) term additionally rewards fixing the WEAKEST digit. Studying a weak-but-learnable digit
    raises both terms the most (it has the most headroom AND it is the floor), so the executive plans
    toward the weakest learnable digit rather than a digit that is already strong (little headroom) or
    already the floor of a saturated set."""
    n = len(state)
    return sum(state) / n + min(state)


class ExecutivePlanner:
    """Wrap the prefrontal executive to PLAN the next digit to study over competence dynamics.

    plan_objective(competence_vec, candidates, horizon=1) builds the length-10 competence vector as the
    plan state, the candidate digits as both the WTA options and the forward-model commands, injects a
    _CompetenceDynamics forward model and the mean+min goal_value, and runs PrefrontalExecutive.plan --
    the spiking winner-take-all decision attractor over the rollout values. plan() returns (goal_index,
    mean_rt) where goal_index indexes into `candidates`, so the chosen digit is candidates[goal_index].
    write=False so planning does not overwrite the executive's persistent working-memory goal here.
    Deterministic given seed."""

    def __init__(self, seed=0, gain=0.5):
        self.seed = seed
        self._dynamics = _CompetenceDynamics(gain=gain)   # the injected .predict(state, cmd) forward model
        self.last_rt = None                               # mean spiking reaction time of the last plan

    def plan_objective(self, competence_vec, candidates, horizon=1):
        """Plan the next digit to study; return the chosen digit (an element of `candidates`)."""
        # fresh executive per call: n_options is the number of WTA options == number of candidate digits
        executive = PrefrontalExecutive(n_options=len(candidates), seed=self.seed)
        state = tuple(float(x) for x in competence_vec)    # length-10 competence tuple (immutable)
        options = list(candidates)                          # candidate digit indices (WTA options + commands)
        goal_index, mean_rt = executive.plan(state, options, self._dynamics, _goal_value,
                                             horizon=horizon, write=False)
        self.last_rt = mean_rt
        return options[goal_index]                          # map the WTA index back to the digit


# ---------------------------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("EXECUTIVE CONTROL ADAPTERS -- dopamine RPE / global-workspace broadcast / PFC planning")
    print("=" * 74)

    # ---- (a) DopamineModulator: positive surprise -> higher lr than negative surprise --------------
    # Use an explicit expected_gain baseline so the two surprises are isolated from each other.
    dm = DopamineModulator(seed=0)
    lr_pos = dm.modulate(competence_gain=0.9, expected_gain=0.4)   # realised >> expected -> +RPE -> burst
    da_pos = dm.da
    lr_neg = dm.modulate(competence_gain=0.1, expected_gain=0.4)   # realised << expected -> -RPE -> dip
    da_neg = dm.da
    # running-mean baseline path (no expected_gain given): still returns a bounded lr
    dm2 = DopamineModulator(seed=0)
    for g in (0.3, 0.35, 0.32):
        dm2.modulate(g)                                            # build a ~0.32 running-mean baseline
    lr_running = dm2.modulate(0.9)                                 # a positive surprise vs that baseline
    print("-" * 74)
    print("(a) dopamine-modulated learning rate:")
    print(f"    positive surprise (gain 0.9 vs exp 0.4): da_level {da_pos:.3f} -> lr_gain {lr_pos:.3f}")
    print(f"    negative surprise (gain 0.1 vs exp 0.4): da_level {da_neg:.3f} -> lr_gain {lr_neg:.3f}")
    print(f"    running-mean baseline path -> positive surprise lr_gain {lr_running:.3f}")

    assert lr_pos > lr_neg, f"a positive competence surprise must raise lr above a negative one ({lr_pos} vs {lr_neg})"
    for lr in (lr_pos, lr_neg, lr_running):
        assert DopamineModulator.LR_MIN <= lr <= DopamineModulator.LR_MAX, \
            f"lr_gain must stay in [{DopamineModulator.LR_MIN}, {DopamineModulator.LR_MAX}], got {lr}"
    assert lr_running > DopamineModulator.LR_MIN, "a positive surprise vs the running mean must lift lr above the floor"

    # ---- (b) WorkspaceBroadcast: high-salience goal ignites, very-low-salience one does not --------
    wb_hi = WorkspaceBroadcast(seed=0)
    hi = wb_hi.broadcast("executive", "study_digit_7", salience=0.90)   # supra-threshold -> ignites
    hi_ign = wb_hi.ignited
    wb_lo = WorkspaceBroadcast(seed=0)
    lo = wb_lo.broadcast("executive", "study_digit_7", salience=0.30)   # sub-threshold -> stays local
    lo_ign = wb_lo.ignited
    print("-" * 74)
    print("(b) global-workspace broadcast of the current goal:")
    print(f"    high salience 0.90 -> broadcast {hi!r} (ignited={hi_ign})")
    print(f"    low  salience 0.30 -> broadcast {lo!r} (ignited={lo_ign})")

    assert hi == "study_digit_7" and hi_ign, "a high-salience goal must ignite and be broadcast"
    assert lo is None and not lo_ign, "a very-low-salience goal must stay local (no ignition, no broadcast)"

    # ---- (c) ExecutivePlanner: plans to study the weakest-but-learnable digit ----------------------
    # Competence vector: digit 3 is clearly the weakest (most headroom); others are already strong.
    competence = [0.80, 0.75, 0.82, 0.15, 0.78, 0.79, 0.83, 0.77, 0.81, 0.76]
    candidates = [0, 3, 6, 9]                              # digit 3 is the weak-but-learnable candidate
    weakest = min(candidates, key=lambda d: competence[d])
    # show goal_value gains so the plan target is not a black box
    dyn = _CompetenceDynamics(gain=0.5)
    base_gv = _goal_value(tuple(competence))
    gains = {d: _goal_value(dyn.predict(tuple(competence), d)) - base_gv for d in candidates}
    ep = ExecutivePlanner(seed=0)
    chosen = ep.plan_objective(competence, candidates, horizon=1)
    print("-" * 74)
    print("(c) prefrontal planning of the next study objective:")
    print(f"    competence {competence}")
    print(f"    candidates {candidates}; goal_value gain per candidate "
          f"{ {d: round(g, 4) for d, g in gains.items()} }")
    print(f"    weakest-but-learnable digit = {weakest}; executive PLANNED digit = {chosen} "
          f"(mean rt {ep.last_rt:.1f} steps)")

    assert max(gains, key=gains.get) == weakest, \
        f"the weakest-but-learnable digit must most raise goal_value (gains {gains})"
    assert chosen == weakest, \
        f"the executive must plan to study the weakest-but-learnable digit {weakest}, got {chosen}"

    # determinism: same seed reproduces the same plan
    assert ExecutivePlanner(seed=0).plan_objective(competence, candidates) == chosen, \
        "same seed must reproduce the planned objective"

    print("=" * 74)
    print(f"self-check OK: dopamine lr {lr_pos:.3f}(+) > {lr_neg:.3f}(-) in [0.5, 2.0]; "
          f"workspace ignites goal at 0.90, stays local at 0.30;")
    print(f"  prefrontal executive plans the weakest-but-learnable digit {chosen}; deterministic")


if __name__ == "__main__":
    main()
