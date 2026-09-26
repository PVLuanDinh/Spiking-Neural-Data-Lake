"""
Interoceptive HOMEOSTATIC DRIVES -- the body's regulated needs as a real perceived state.

Where this sits (the honest gap it closes). medulla/interoceptive_arousal.py already turns visceral
homeostatic error into a single bounded CAUTION scalar -- a hold on action. That is interoception as a
brake, not as a MOTIVE. It cannot say "I should EXPLORE now" or "I should REST now"; it only says "be
careful". This module adds the missing half: a small vector of named homeostatic DRIVES (energy /
fatigue, information-hunger / curiosity-satiation, social-need), each a regulated variable with a
SET-POINT and a DEFICIT that GROWS while the need is unmet and is REDUCED by the matching activity.
That deficit vector IS a real interoceptive PERCEPT -- a state the rest of the brain can read -- and it
yields a MOTIVATIONAL BIAS: a soft per-goal preference saying which need the agent should prioritise
right now. It is the drive side of homeostatic motivation (Hull 1943 drive-reduction; Keramati & Gutkin
2011), kept strictly on the representation / value side.

The drives (developer-chosen, honestly not evolved needs):
  energy      -- fatigue; passively depletes (living costs energy), REST replenishes it.
  curiosity   -- information-hunger / curiosity-satiation; grows while nothing new is learned, EXPLORE
                 sates it.
  social      -- social-need; grows in isolation, SOCIALIZE sates it.
Each drive keeps a satisfaction `level` in [0, set_point] (set_point == fully satisfied). The perceived
DEFICIT is set_point - level (rectified). Every tick each level decays by `drift` (deficit grows); the
activity that services a drive adds `replenish` back to its level (deficit shrinks toward the set-point).

Interoceptive percept + motivation:
  deficit()          -> {drive: deficit}          the perceived interoceptive state vector.
  dominant_drive()   -> str                        the most URGENT drive now (deficit * priority, argmax).
  motivational_bias()-> {drive: preference}        a softmax over urgency; a soft per-goal preference that
                                                   biases WHICH goal to pursue. It never selects an action.

Drive-reduction learning (BRAIN-NATIVE, no gradient). tick() also runs a local, dopamine-like reward-
modulated delta rule over a per-activity value. The reward CREDITED to the performed activity is the
DRIVE REDUCTION that activity caused this tick -- the deficit it actually removed (a temporal CHANGE in
the serviced drive's deficit, NOT its level). Passive drift is background (the environment, not the
action's doing), so it is correctly excluded from the action's credit:

    reward       = deficit the performed activity reduced this tick   (>= 0; 0 if it services no drive)
    value[act]  += lr * (reward - value[act])                          (local reward-modulated plasticity)

An activity that reduces a deficit earns positive reward and gains value; an IRRELEVANT activity (one
that services no drive) reduces nothing, earns zero reward, and its value stays ~0 -- clearly below every
reducing activity. There is no loss.backward here -- it is a two-factor neuromodulatory local rule.

Fair controls (un-riggable -- each real claim has a control that should FAIL / fall to chance):
  - RISES WHEN UNMET, HOMEOSTASIS WHEN MET: with NO activity a drive's deficit rises monotonically; the
    matching activity drives it back toward the set-point (final deficit far below the no-activity
    control). A drive that is NOT serviced in the same run keeps rising -- servicing is selective.
  - DOMINANCE SWITCHES: repeatedly servicing the current dominant drive lets another overtake it, so the
    dominant drive changes over a servicing schedule (it tracks the moving state, not a fixed label).
  - DRIVE-REDUCTION RAISES VALUE: after learning, the reducing activity's value beats an irrelevant
    activity's. CONTROL: a SHUFFLED learner that credits each tick's reward to a RANDOM activity
    (decorrelating action from outcome) collapses that gap to ~0 -- the value comes from the real credit
    assignment, not the delta rule alone.
  - NO FALSE PREFERENCE: with all drives configured identically and equal, the motivational bias is
    ~uniform and the dominant drive is the deterministic tie-break -- urgency is not manufactured.

Honest framing (read this). The drives and their set-points are DESIGNER-CHOSEN, not evolved needs;
nothing here is actually hungry or lonely -- "deficit" is a number in a register. This is strictly
REPRESENTATION / VALUE side: it perceives a drive state and biases which goal to prefer; it NEVER selects
an action, drives no motion, imports nothing from brainstem/, and touches no actuator or E-STOP. And like
drive_reduction.py it does NOT regulate better than a hand-wired controller at this toy scale -- the value
of a perceived drive vector is that motivation becomes an INTERNAL, readable state, not that it optimises.
Per-instance random.Random(seed), ASCII-only, deterministic, no import-time side effects.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hypothalamus.homeostatic_drives
"""

import math
import random


def declare_ports(manifest):
    """Manifest registrant (specs/022 T026): the VALENCE half of the interoceptive channel.
    OUT the three normalized drive deficits (energy, curiosity, social; each deficit /
    set_point in [0,1] -- unmet need IS the negative-valence body signal), delivered whole
    each tick. The consumer edge is declared by the insula terminus."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "homeostatic_drives",
        Port(
            name="drive_deficit_out",
            region="hypothalamus",
            shape=(3,),
            dtype="f4",
            unit="level01",
            encoding="VEC_DENSE",
            kind="OUT",
            declaring_module="homeostatic_drives",
            delivery_policy="all-or-none",
        ),
    )


#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_train_values": (
        "LOCAL",
        "uniform-exploration training of the value rule so every activity is visited. It drives the "
        "three-factor value learner; the exploration policy is not itself a gradient",
    ),
}

# ---- default drive specification (name -> servicing activity, dynamics, priority) --------------------
# set_point is the fully-satisfied level; deficit = set_point - level. drift grows the deficit each tick
# (the passive loss the agent must oppose); replenish is how much the matching activity restores; priority
# weights the deficit into an URGENCY (a survival-ish need outranks a mild one at equal deficit).
SET_POINT = 1.0
DEFAULT_DRIVES = (
    # name,        activity,     drift, replenish, priority
    ("energy", "rest", 0.030, 0.40, 1.20),  # fatigue: living depletes energy, rest restores
    ("curiosity", "explore", 0.050, 0.35, 0.80),  # information-hunger: sated by exploring/learning
    ("social", "socialize", 0.020, 0.45, 1.00),  # social-need: grows in isolation, sated by contact
)
IRRELEVANT_ACTIVITY = (
    "fidget"  # a control activity that services NO drive (its value should stay low)
)
BETA = 6.0  # softmax temperature for the motivational bias (higher == sharper)
LR = 0.05  # learning rate of the reward-modulated value delta rule (low == tight convergence)


class Drive:
    """One homeostatic drive: a regulated satisfaction `level` in [0, set_point] with a passive `drift`
    that grows its deficit and a `replenish` its servicing activity restores. deficit == set_point -
    level (rectified). Nothing is felt; `level` is a number in an accumulator."""

    def __init__(self, name, activity, drift, replenish, priority, level, set_point=SET_POINT):
        self.name = name
        self.activity = activity  # the activity whose performance reduces this drive's deficit
        self.drift = drift  # per-tick satisfaction decay (deficit growth) while unmet
        self.replenish = replenish  # satisfaction restored per unit of the matching activity
        self.priority = priority  # weight of this drive's deficit in the urgency ranking
        self.set_point = set_point  # fully-satisfied level (the homeostatic reference)
        self.level = level  # current satisfaction (set_point == no deficit)

    def deficit(self):
        """Perceived deficit: how far below the set-point this need currently sits (rectified >= 0)."""
        return max(0.0, self.set_point - self.level)

    def decay(self):
        """Passive drift: satisfaction leaks toward empty, so an unmet deficit GROWS each tick."""
        self.level = max(0.0, self.level - self.drift)

    def service(self, intensity):
        """The matching activity restores satisfaction (reduces the deficit), clamped at the set-point.
        Returns the satisfaction GAINED == the deficit REDUCED this tick (>= 0)."""
        before = self.level
        self.level = min(self.set_point, self.level + self.replenish * intensity)
        return self.level - before


class HomeostaticDrives:
    """A vector of named homeostatic drives read as a real interoceptive PERCEPT + a motivational bias.

    tick(activity) advances every drive one step: all deficits drift UP, then the performed `activity`
    replenishes the drive it services; it returns the perceived deficit vector (the interoceptive state)
    and runs a local drive-reduction value update. deficit() exposes the percept, dominant_drive() the
    most urgent need, motivational_bias() a soft per-goal preference. It NEVER selects an action or
    touches an actuator -- the executive/goal selector reads the bias and decides.

    `activity` is a string activity name, None (idle -- no need serviced), or a dict {activity: intensity
    in [0, 1]} for graded / simultaneous activity. shuffle=True is the credit-assignment CONTROL: it
    credits each tick's reward to a RANDOM activity, decorrelating action from outcome. Deterministic;
    per-instance random.Random(seed)."""

    def __init__(
        self,
        specs=DEFAULT_DRIVES,
        seed=0,
        beta=BETA,
        lr=LR,
        shuffle=False,
        init_level=1.0,
        init_jitter=0.0,
    ):
        self.rng = random.Random(seed)  # per-instance rng: init jitter, exploration, shuffle target
        self.beta = beta
        self.lr = lr
        self.shuffle = shuffle
        self.drives = {}
        for name, activity, drift, replenish, priority in specs:
            lvl = min(
                SET_POINT,
                max(
                    0.0,
                    init_level
                    + (self.rng.uniform(-init_jitter, init_jitter) if init_jitter else 0.0),
                ),
            )
            self.drives[name] = Drive(name, activity, drift, replenish, priority, level=lvl)
        # every servicing activity plus the irrelevant control action -> the learnable action set.
        self.activities = [d.activity for d in self.drives.values()] + [IRRELEVANT_ACTIVITY]
        self.values = {
            a: 0.0 for a in self.activities
        }  # per-activity learned value (drive-reduction)
        self._t = 0

    # ---- the interoceptive percept ------------------------------------------------------------------
    def deficit(self):
        """The interoceptive PERCEPT: {drive_name: perceived deficit} (0 == satisfied, up to set_point)."""
        return {name: d.deficit() for name, d in self.drives.items()}

    def levels(self):
        """{drive_name: satisfaction level} -- the complement of the deficit (set_point == satisfied)."""
        return {name: d.level for name, d in self.drives.items()}

    def _total_deficit(self):
        return sum(d.deficit() for d in self.drives.values())

    def _urgency(self):
        """Per-drive urgency = perceived deficit * priority (what the motivation ranks and biases over)."""
        return {name: d.deficit() * d.priority for name, d in self.drives.items()}

    # ---- motivation (which need to prioritise) ------------------------------------------------------
    def dominant_drive(self):
        """The most URGENT drive right now (max deficit * priority; ties -> declaration order). This is
        the need the executive should prioritise; it is a read-out, not a command."""
        urg = self._urgency()
        order = list(self.drives.keys())
        return max(order, key=lambda n: (urg[n], -order.index(n)))

    def motivational_bias(self):
        """A soft per-goal preference: softmax over the drives' urgency, {drive_name: preference} summing
        to 1. The drive with the largest urgency gets the largest share (its argmax == dominant_drive());
        with equal urgencies it is ~uniform (no manufactured preference). The goal servicing drive `name`
        is drives[name].activity. This BIASES goal selection; it never selects or commands an action."""
        urg = self._urgency()
        names = list(self.drives.keys())
        mx = max(urg.values())  # subtract the max for a numerically-stable softmax
        exps = {n: math.exp(self.beta * (urg[n] - mx)) for n in names}
        z = sum(exps.values())
        return {n: exps[n] / z for n in names}

    def action_values(self):
        """The learned per-activity value (from drive-reduction). A reducing activity should score above
        the irrelevant control activity after learning."""
        return dict(self.values)

    # ---- dynamics + local drive-reduction learning --------------------------------------------------
    def _normalize(self, activity):
        """Coerce the `activity` argument to {activity_name: intensity in [0, 1]}. None -> idle (empty)."""
        if activity is None:
            return {}
        if isinstance(activity, dict):
            return {k: max(0.0, min(1.0, v)) for k, v in activity.items() if v}
        return {activity: 1.0}

    def tick(self, activity=None):
        """Advance every drive one step and return the perceived deficit vector (the interoceptive state).

        Order: (1) every drive drifts -> its deficit GROWS; (2) the performed activity replenishes the
        drive it services -> that deficit SHRINKS, and the amount removed is the drive REDUCTION the
        action caused; (3) local reward-modulated update of the performed activity's value toward that
        reduction (shuffle -> credit a random activity instead). Returns deficit()."""
        intents = self._normalize(activity)
        for d in self.drives.values():  # (1) passive drift: unmet deficits grow
            d.decay()
        reduced = 0.0  # drive REDUCTION caused by the performed activity
        for d in self.drives.values():  # (2) the matching activity reduces its deficit
            inten = intents.get(d.activity, 0.0)
            if inten > 0.0:
                reduced += d.service(inten)  # service() returns the deficit it removed (>= 0)
        for name, inten in intents.items():  # (3) local plasticity over the performed activity
            if inten <= 0.0 or name not in self.values:
                continue
            target = self.rng.choice(self.activities) if self.shuffle else name
            self.values[target] += self.lr * (reduced - self.values[target])
        self._t += 1
        return self.deficit()


def _train_values(shuffle, steps=6000, seed=0):
    """Uniform-exploration training of the value rule. Each step performs a RANDOM activity (so every
    action is sampled equally) and lets the drives evolve. Returns the learned action_values(). The only
    difference between the aligned and shuffled runs is the credit assignment, so any value gap is due to
    that, not to sampling."""
    hd = HomeostaticDrives(seed=seed, shuffle=shuffle)
    rng = random.Random(seed + 1)  # exploration rng, shared schedule across conditions
    for _ in range(steps):
        hd.tick(rng.choice(hd.activities))
    return hd.action_values()


def main():
    width = 84
    print("=" * width)
    print(
        "INTEROCEPTIVE HOMEOSTATIC DRIVES -- perceived needs + motivational bias (value-side only)"
    )
    print("drives: energy/fatigue (rest), curiosity/info-hunger (explore), social-need (socialize)")
    print("=" * width)

    hd = HomeostaticDrives(seed=0)
    print(
        f"drives K={len(hd.drives)}  set-point={SET_POINT:.1f}  softmax beta={BETA:.0f}  lr={LR:.2f}  "
        f"irrelevant control activity='{IRRELEVANT_ACTIVITY}'"
    )
    for name, d in hd.drives.items():
        print(
            f"  {name:<10} serviced by '{d.activity}'  drift={d.drift:.3f}/tick  replenish={d.replenish:.2f}  "
            f"priority={d.priority:.2f}"
        )
    print("-" * width)

    # ---- (1) unmet deficit RISES monotonically; the matching activity returns homeostasis ------------
    warm = 12
    idle = HomeostaticDrives(seed=0)
    idle_energy = []
    for _ in range(warm):
        idle.tick(None)  # NO activity: every deficit drifts up
        idle_energy.append(idle.deficit()["energy"])
    met = HomeostaticDrives(seed=0)
    met_energy, met_curiosity = [], []
    for _ in range(warm):
        met.tick("rest")  # service ENERGY every tick; leave curiosity unmet
        met_energy.append(met.deficit()["energy"])
        met_curiosity.append(met.deficit()["curiosity"])
    print("(1) unmet vs serviced (energy deficit over ticks):")
    print(
        f"    NO activity  : {idle_energy[0]:.3f} -> {idle_energy[-1]:.3f}  (rises monotonically, unmet)"
    )
    print(
        f"    REST each tick: {met_energy[0]:.3f} -> {met_energy[-1]:.3f}  (held near the set-point)"
    )
    print(
        f"    curiosity (never serviced in the REST run): {met_curiosity[0]:.3f} -> {met_curiosity[-1]:.3f}  (still rises)"
    )
    print("-" * width)

    # ---- (2) the DOMINANT drive switches as deficits change -----------------------------------------
    sw = HomeostaticDrives(seed=0)
    for _ in range(6):
        sw.tick(None)  # let all deficits grow so a dominant emerges
    dominants = []
    for _ in range(30):
        dom = sw.dominant_drive()
        dominants.append(dom)
        sw.tick(sw.drives[dom].activity)  # service the CURRENT dominant -> another overtakes
    distinct = []
    for dom in dominants:
        if dom not in distinct:
            distinct.append(dom)
    print(
        f"(2) dominant-drive schedule (service the current top each tick): {' '.join(dominants[:14])} ..."
    )
    print(
        f"    distinct dominants seen: {distinct}  (it tracks the moving state, not a fixed label)"
    )
    print("-" * width)

    # ---- (3) drive-reduction RAISES the reducing activity's value; SHUFFLE control collapses it ------
    aligned = _train_values(shuffle=False)
    shuffled = _train_values(shuffle=True)
    reducing = [hd.drives[n].activity for n in hd.drives]  # rest / explore / socialize
    v_reduce_aligned = min(aligned[a] for a in reducing)  # the WEAKEST reducing action ...
    v_irrel_aligned = aligned[IRRELEVANT_ACTIVITY]  # ... must still beat the irrelevant one
    gap_aligned = v_reduce_aligned - v_irrel_aligned
    v_reduce_shuf = min(shuffled[a] for a in reducing)
    v_irrel_shuf = shuffled[IRRELEVANT_ACTIVITY]
    gap_shuffled = abs(v_reduce_shuf - v_irrel_shuf)  # under shuffled credit the ordering is noise
    print("(3) learned action value (drive-reduction reward-modulated delta rule):")
    print("    ALIGNED credit :  " + "  ".join(f"{a}={aligned[a]:+.3f}" for a in hd.activities))
    print("    SHUFFLED credit:  " + "  ".join(f"{a}={shuffled[a]:+.3f}" for a in hd.activities))
    print(
        f"    reducing-vs-irrelevant gap:  aligned {gap_aligned:+.3f}  >>  shuffled {gap_shuffled:+.3f}"
    )
    print("-" * width)

    # ---- (4) NO FALSE PREFERENCE: identical, equal drives -> uniform bias, deterministic dominant ----
    flat_specs = (
        ("a", "act_a", 0.03, 0.4, 1.0),
        ("b", "act_b", 0.03, 0.4, 1.0),
        ("c", "act_c", 0.03, 0.4, 1.0),
    )
    flat = HomeostaticDrives(specs=flat_specs, seed=0)
    for _ in range(8):
        flat.tick(None)  # equal drift on identical drives -> equal deficits
    flat_bias = flat.motivational_bias()
    bias_spread = max(flat_bias.values()) - min(flat_bias.values())
    print(
        f"(4) identical equal drives -> bias {{{', '.join(f'{k}:{v:.3f}' for k, v in flat_bias.items())}}}  "
        f"spread={bias_spread:.4f}  dominant='{flat.dominant_drive()}'"
    )

    # ---- a live motivational read-out on the first instance ------------------------------------------
    live = HomeostaticDrives(seed=0)
    for _ in range(9):
        live.tick("rest")  # keep energy sated -> curiosity/social should dominate
    bias = live.motivational_bias()
    print(
        f"    live percept (deficit): {{{', '.join(f'{k}:{v:.3f}' for k, v in live.deficit().items())}}}"
    )
    print(
        f"    motivational bias     : {{{', '.join(f'{k}:{v:.3f}' for k, v in bias.items())}}}  "
        f"dominant='{live.dominant_drive()}'"
    )
    print("=" * width)

    # ================= un-riggable self-checks (deterministic asserts) ================================
    # (1) RISES WHEN UNMET: the no-activity energy deficit is strictly monotonically increasing.
    assert all(b > a for a, b in zip(idle_energy, idle_energy[1:])), (
        "an unmet drive's deficit must rise monotonically with no servicing activity"
    )
    # ... and the MATCHING activity returns homeostasis: final serviced deficit far below the control.
    assert met_energy[-1] < 0.3 * idle_energy[-1], (
        f"servicing must drive the deficit back toward the set-point ({met_energy[-1]:.3f} vs unmet {idle_energy[-1]:.3f})"
    )
    assert met_energy[-1] < 0.05, (
        f"the serviced drive must sit near its set-point ({met_energy[-1]:.3f})"
    )
    # ... SELECTIVE: a drive NOT serviced in the same run keeps rising (servicing energy does not fix curiosity).
    assert all(b > a for a, b in zip(met_curiosity, met_curiosity[1:])), (
        "a non-serviced drive must keep rising even while another drive is serviced (selectivity)"
    )

    # (2) DOMINANCE SWITCHES: servicing the current top lets another overtake, so >= 2 drives are dominant.
    assert len(distinct) >= 2, (
        f"the dominant drive must switch as deficits change (saw only {distinct})"
    )

    # (3) DRIVE-REDUCTION RAISES VALUE: the reducing actions beat the irrelevant one; the SHUFFLE collapses it.
    assert abs(v_irrel_aligned) < 0.02, (
        f"the irrelevant activity reduces nothing, so its value must stay ~0 ({v_irrel_aligned:+.3f})"
    )
    assert all(aligned[a] > v_irrel_aligned for a in reducing), (
        "every drive-reducing activity must gain value over the irrelevant activity"
    )
    assert gap_aligned > 0.05, (
        f"the reducing-vs-irrelevant value gap must be clearly positive ({gap_aligned:+.3f})"
    )
    assert gap_aligned > 3.0 * max(gap_shuffled, 1e-6), (
        f"shuffling the credit must collapse the value gap ({gap_aligned:+.3f} aligned vs {gap_shuffled:+.3f} shuffled)"
    )

    # (4) NO FALSE PREFERENCE: identical equal drives -> ~uniform bias and a deterministic tie-break.
    assert bias_spread < 1e-6, (
        f"identical equal drives must give a uniform bias (spread {bias_spread:.6f})"
    )
    assert flat.dominant_drive() == "a", (
        "an all-equal tie must break deterministically to the first drive"
    )

    # (5) BIAS IS A VALID PREFERENCE + POINTS AT THE DOMINANT DRIVE ------------------------------------
    assert abs(sum(bias.values()) - 1.0) < 1e-9, (
        "the motivational bias must be a normalized preference (sums to 1)"
    )
    assert all(0.0 <= v <= 1.0 for v in bias.values()), "every preference must be in [0, 1]"
    assert max(bias, key=bias.get) == live.dominant_drive(), (
        "the bias must peak on the dominant drive's goal"
    )
    assert live.deficit()["energy"] < live.deficit()["curiosity"], (
        "keeping energy serviced must leave it less deficient than the unmet curiosity drive"
    )

    # (6) DETERMINISM: same seed -> identical values + percept; a different seed -> a different trace.
    a_vals = _train_values(shuffle=False, seed=11)
    b_vals = _train_values(shuffle=False, seed=11)
    c_vals = _train_values(shuffle=False, seed=12)
    assert a_vals == b_vals, "same seed must reproduce identical learned values"
    assert a_vals != c_vals, (
        "a different seed must produce different learned values (the rng is seeded)"
    )

    print(
        f"self-check OK: unmet energy deficit rises {idle_energy[0]:.2f}->{idle_energy[-1]:.2f}, "
        f"REST holds it at {met_energy[-1]:.2f} (curiosity still climbs to {met_curiosity[-1]:.2f});"
    )
    print(
        f"  dominant drive switches across {distinct}; drive-reduction gap aligned {gap_aligned:+.2f} "
        f">> shuffled {gap_shuffled:+.2f} (irrelevant activity value stays ~{v_irrel_aligned:+.2f});"
    )
    print(
        f"  identical drives -> uniform bias (spread {bias_spread:.1e}); bias peaks on the dominant "
        f"drive; deterministic. Value-side only -- selects no action, touches no actuator"
    )


if __name__ == "__main__":
    main()
