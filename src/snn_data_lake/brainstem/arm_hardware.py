"""Hardware TARGET for the physical arm: spec, static torque budget, and the sim-to-real gap.

WHAT THIS IS, AND IS NOT
    This is the build spec for an arm that DOES NOT EXIST YET. Nothing here reads a sensor and
    nothing here is telemetry. It is the plan, expressed as numbers a program can check, so the
    dashboard can show the gap between the body we simulate and the body we intend to build without
    either one pretending to be the other.

    Source: robot_arm_build_notes.md (planning conversation, 2026-07). Every figure below is from
    those notes; where a number is DERIVED it says so and shows the arithmetic.

THE FINDING THAT MOTIVATED WRITING THIS AS CODE
    The notes state the rule -- "size servos to rated torque, not stall torque; rated is roughly a
    quarter of stall" -- and separately state the target: 500 g payload at ~300 mm reach on an
    ST3215 1:345 at 12 V. Applying the rule to the target:

        payload-only shoulder torque = 0.5 kg * 30 cm            = 15.0 kg.cm
        ST3215 1:345 @12V, stall 30 kg.cm -> rated = 30/4         =  7.5 kg.cm
        margin                                                    =  0.50x   UNDERPOWERED

    The shoulder is short by 2x on the payload ALONE, before any arm mass. This is not a criticism
    of the notes -- it is the notes' own rule applied to the notes' own numbers, and it is exactly
    the check that is easy to skip when a parts list looks complete. It is asserted in main() so it
    cannot be quietly forgotten between now and a purchase.

    Ways out, all in the notes' own vocabulary: shorten the reach, drop the payload target, gear the
    shoulder down further, use two servos in parallel on the shoulder axis, or counterbalance the
    upper arm. Choosing among them is a design decision, not something this file should presume.
"""

from __future__ import annotations

from snn_data_lake.brainstem import arm_config as _arm_config

# ---- servos ---------------------------------------------------------------------------------
# Feetech ST3215 family. Stall figures from the notes; RATED is derived as stall/4 per the notes'
# own sizing rule, because a continuous-duty number is what a joint must actually be sized against.
_RATED_FRACTION_OF_STALL = 0.25

SERVOS = {
    "ST3215-1:345": {
        "stall_kgcm_12v": 30.0,
        "stall_kgcm_7v4": 19.5,
        "ratio": 345,
        "role": "base, shoulder",
    },
    "ST3215-1:191": {
        "stall_kgcm_12v": 16.6,
        "stall_kgcm_7v4": 10.8,
        "ratio": 191,
        "role": "elbow, wrist pitch",
    },
    "ST3215-1:147": {
        "stall_kgcm_12v": 12.8,
        "stall_kgcm_7v4": 8.3,
        "ratio": 147,
        "role": "wrist roll",
    },
    "SC-series": {"stall_kgcm_12v": 8.0, "stall_kgcm_7v4": 5.0, "ratio": 0, "role": "gripper"},
}

# The 1:191 and 1:147 stall figures are INTERPOLATED from the 1:345 datum by gear ratio -- the notes
# give ratios and relative behaviour ("faster, less torque") but not absolute torque for these two.
# Marked so nobody mistakes them for datasheet values.
_INTERPOLATED = ("ST3215-1:191", "ST3215-1:147", "SC-series")

GRAVITY = 9.81  # m/s^2 (only used for documentation; kg.cm math below is in kg-force units)


def rated_kgcm(servo, volts=12.0):
    """Continuous-duty torque. The notes are explicit that stall is NOT the number to size against."""
    s = SERVOS[servo]
    stall = s["stall_kgcm_12v"] if volts >= 10.0 else s["stall_kgcm_7v4"]
    return stall * _RATED_FRACTION_OF_STALL


# ---- the intended arm -----------------------------------------------------------------------
# 6 DOF, ~250-350 mm reach, 500 g payload target.
TARGET = {
    "dof": 6,
    "reach_mm": 300.0,  # mid of the 250-350 range in the notes
    "payload_kg": 0.5,
    "bus_volts": 12.0,
    "supply_amps": 10.0,  # notes: 8-10A+, several servos stalling at 2.7A each
    "stall_amps_per_servo": 2.7,
}

# joint -> (servo, distance from that joint to the payload in mm). The distance is what converts a
# payload mass into a torque at that joint; wrist roll and the gripper carry no bending moment from
# the payload's weight, so their lever arm is 0.
# DERIVED from arm_config.JOINTS, not written twice. This used to be a literal, and a literal
# here is a second source of truth for the same six rows -- rename a joint or re-spec a servo in
# one place and the torque table silently describes a different arm than the vocabulary does.
# arm_config owns the table (it is "calibration in ONE place"); this is the build-target view of
# it: (alias, servo, lever_mm). base_yaw is a vertical axis and carries no gravity moment, only
# inertia -- which is why its lever is 0.0 and its required torque comes out zero below.
JOINTS = tuple((j.alias, j.servo, j.lever_mm) for j in _arm_config.JOINTS)

# UNIT CONVERSIONS, not tuned values -- P1 governs constants that arbitrate BEHAVIOUR, and these two
# arbitrate nothing: they are the definition of the units servo datasheets are written in. 1 N*m is
# 1/g kgf*m = 10.19716 kgf*cm; g is standard gravity, and MuJoCo's own gravity is read from the model
# rather than assumed, so this appears only in the payload weight.
KGCM_PER_NM = 10.19716
G_MS2 = 9.81


def holding_torque_kgcm(lever_mm, payload_kg=None, arm_mass_kg=0.0, arm_cog_mm=None):
    """Static gravity torque at a joint with the link held HORIZONTAL -- the worst case.

    kg.cm is kilogram-force-centimetre, the unit servo datasheets use, so mass_kg * distance_cm is
    the torque directly with no g factor. Horizontal is the worst case because the moment arm is the
    full length; any other pose is smaller by cos(elevation).

    This is STATIC only: no acceleration, no dynamics, no friction. A real arm needs headroom above
    this, not equality with it.
    """
    payload_kg = TARGET["payload_kg"] if payload_kg is None else payload_kg
    arm_cog_mm = (lever_mm / 2.0) if arm_cog_mm is None else arm_cog_mm
    return payload_kg * (lever_mm / 10.0) + arm_mass_kg * (arm_cog_mm / 10.0)


def budget(arm_mass_kg=0.0, volts=None):
    """Per-joint required vs rated torque. Returns rows; `ok` False means underpowered.

    arm_mass_kg is the mass of everything OUTBOARD of the joint. Default 0.0 gives the
    payload-only floor -- the most generous case, and the shoulder already fails it.
    """
    volts = TARGET["bus_volts"] if volts is None else volts
    rows = []
    for name, servo, lever in JOINTS:
        need = holding_torque_kgcm(lever, arm_mass_kg=arm_mass_kg) if lever > 0 else 0.0
        have = rated_kgcm(servo, volts)
        rows.append(
            {
                "joint": name,
                "servo": servo,
                "lever_mm": lever,
                "required_kgcm": round(need, 2),
                "rated_kgcm": round(have, 2),
                "margin_x": round(have / need, 2) if need > 0 else None,
                "ok": (need <= have) if need > 0 else True,
                "estimated": servo in _INTERPOLATED,
            }
        )
    return rows


def peak_current_amps(n_simultaneous=3):
    """Supply sizing. The notes size for several servos stalling at once, not one."""
    return n_simultaneous * TARGET["stall_amps_per_servo"]


# ---- the sim-to-real gap --------------------------------------------------------------------
# The KINEMATIC half of this gap is closed (specs/004). What we simulate is now the SO-101
# itself: six joints, real units, mass, gravity and a 2 ms integration step, with actuator
# force limits taken from the MJCF that TheRobotStudio matched to the real STS3215.
#
# The gap that REMAINS is the one that matters and the reason this module exists: the arm is
# not BUILT, and the shoulder's torque budget does not close (15.0 kg*cm required against 7.5
# rated). A faithful simulation of an underpowered arm is still an underpowered arm -- if
# anything the simulation makes the shortfall easier to believe, which is exactly when a
# dashboard is most tempted to imply the hardware exists.
SIM_GAP = {
    "sim_dof": 6,
    "target_dof": 6,
    "missing_dof": (),  # the six simulated joints ARE the six target joints
    "sim_units": "SI (metres, radians, N*m; MuJoCo SO-101)",
    "target_units": "mm (reach ~300)",
    "mm_per_sim_unit": 1000.0,  # metres -> mm; the sim is in real units now, not a scale factor
    "sim_has_dynamics": True,
    "sim_has_gravity": True,
    "bridge": (
        "MuJoCo + SO-ARM100 (TheRobotStudio) ships MJCF with motor properties matched to the real "
        "STS3215 -- a genuine digital twin. That is the path from this kinematic toy to a body whose "
        "torque limits are real, and it is the same engine mhflocke drives its demo with."
    ),
}


def model_budget(payload_kg=None, grid=25):
    """Per-joint WORST-CASE gravity-hold torque taken from the MuJoCo SO-101 model itself.

    `budget()` above is a hand formula: payload mass times a hand-written lever, with the arm's own
    links weighing nothing (`arm_mass_kg=0.0`, documented there as "the most generous case"). This
    function asks the body instead. `qfrc_bias` at zero velocity IS the gravity torque, computed from
    the model's real link masses and real geometry, and the payload is added as `-J^T F` at the
    gripper -- the same weight a servo would actually hold.

    MEASURED 2026-08-02 over a 25 x 25 sweep of shoulder_lift x elbow_flex at 0.5 kg, and it is the
    reason this function exists: the hand formula is OPTIMISTIC, on two counts that do not cancel.

        joint             formula     model     rated    margin
        shoulder_pitch      15.00     24.41      7.50      0.31
        elbow_pitch          9.00     14.43      4.15      0.29
        wrist_pitch          3.00      4.25      4.15      0.98   <- reported ok=True by the formula

    The arm's own 0.632 kg was omitted entirely, and the hand-written 300 mm lever overstates the
    model's actual 254.4 mm horizontal reach at q=0. The mass omission dominates. `wrist_pitch` was
    reported OK and is not, so the formula was hiding a third underpowered joint.

    Returns None when the model cannot be loaded (no mujoco, no scene) rather than guessing -- the
    caller falls back to the formula and SAYS SO via the `source` key, because a number whose origin
    is unknown is worse than a number that is merely rough."""
    try:
        import mujoco  # noqa: PLC0415 -- optional; absence must degrade, not crash
        import numpy as np  # noqa: PLC0415

        from snn_data_lake.brainstem import so101_arm  # noqa: PLC0415
    except Exception:
        return None
    try:
        m = so101_arm._load()
        d = mujoco.MjData(m)
        ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    except Exception:
        return None

    payload_kg = TARGET["payload_kg"] if payload_kg is None else payload_kg
    aliases = [j.alias for j in _arm_config.JOINTS]
    rated = {r["joint"]: r["rated_kgcm"] for r in budget()}
    worst = [0.0] * m.nv
    feasible = total = 0
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    for a in np.linspace(lo[1], hi[1], grid):
        for b in np.linspace(lo[2], hi[2], grid):
            d.qpos[:] = 0.0
            d.qpos[1], d.qpos[2] = a, b
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            tau = np.array(d.qfrc_bias[: m.nv], dtype=float)
            if payload_kg:
                mujoco.mj_jacBody(m, d, jacp, jacr, ee)
                # MINUS: the payload's weight and the links' weight both pull DOWN, so the two terms
                # must ADD. With the opposite sign they cancel and the arm reads STRONGER than it is
                # -- measured at q=0, shoulder goes -5.492 -> +5.712 (wrong) against -16.696 (right).
                tau = tau - jacp.T @ np.array([0.0, 0.0, -payload_kg * G_MS2])
            tau = np.abs(tau) * KGCM_PER_NM
            total += 1
            feasible += all(
                tau[i] <= rated.get(aliases[i], float("inf"))
                for i in range(min(m.nv, len(aliases)))
            )
            worst = [max(w, float(t)) for w, t in zip(worst, tau, strict=False)]

    rows = []
    for i, alias in enumerate(aliases[: m.nv]):
        need, have = worst[i], rated.get(alias, 0.0)
        rows.append(
            {
                "joint": alias,
                # FOUR decimals, not two. At two, wrist_roll printed `required 0.00, margin 8368x`
                # -- which reads as an enormously strong joint and is actually a divide by a number
                # the display had rounded away. The margin is only interpretable next to a `required`
                # precise enough to explain it.
                "required_kgcm": round(need, 4),
                "rated_kgcm": have,
                "margin_x": (round(have / need, 2) if need > 1e-6 else None),
                "ok": need <= have if need > 1e-6 else True,
            }
        )
    return {
        "source": "model",
        "payload_kg": payload_kg,
        "poses_swept": total,
        "poses_within_rating": feasible,
        "joints": rows,
        "underpowered": [r["joint"] for r in rows if not r["ok"]],
    }


def _worst_torque(payload_kg, reach_scale, grid):
    """(worst_total, worst_link_only) per DOF in kg.cm over the shoulder x elbow grid, or None.

    Split because a counterbalance spring cancels the arm's OWN weight and not the payload's, so
    the two terms have to be costed separately. `reach_scale` scales the moment arms and holds the
    link masses fixed, which is CONSERVATIVE: a genuinely shorter arm is also a lighter one, so the
    real torque at a shortened reach is below what this reports."""
    try:
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from snn_data_lake.brainstem import so101_arm  # noqa: PLC0415

        m = so101_arm._load()
        d = mujoco.MjData(m)
        ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    except Exception:
        return None
    total = [0.0] * m.nv
    link_only = [0.0] * m.nv
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    for a in np.linspace(lo[1], hi[1], grid):
        for b in np.linspace(lo[2], hi[2], grid):
            d.qpos[:] = 0.0
            d.qpos[1], d.qpos[2] = a, b
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            link = np.array(d.qfrc_bias[: m.nv], dtype=float)
            mujoco.mj_jacBody(m, d, jacp, jacr, ee)
            pay = -(jacp.T @ np.array([0.0, 0.0, -payload_kg * G_MS2]))
            tot = np.abs((link + pay) * reach_scale) * KGCM_PER_NM
            lnk = np.abs(link * reach_scale) * KGCM_PER_NM
            total = [max(w, float(t)) for w, t in zip(total, tot, strict=False)]
            link_only = [max(w, float(t)) for w, t in zip(link_only, lnk, strict=False)]
    return total, link_only


#: The five ways out the module header already names, as numbers a search can try. Grid values, not
#: tuned parameters -- they are the discrete engineering choices available (a 2:1 or 3:1 reduction
#: exists; a 2.37:1 does not), and the search reports which COMBINATIONS close the gap rather than
#: optimising within them.
MITIGATIONS = {
    "payload_kg": (0.5, 0.35, 0.25),
    "reach_scale": (1.0, 0.85, 0.7),
    "counterbalance": (False, True),
    "extra_reduction": (1.0, 2.0, 3.0, 4.0),
    "parallel_shoulder": (1.0, 2.0),
}


def buildable_configurations(grid=21):
    """Which COMBINATIONS of the header's five ways out put every joint inside its rating?

    The header lists the escapes -- shorten the reach, drop the payload, gear the shoulder down,
    two servos in parallel, counterbalance the upper arm -- and then says choosing among them is a
    design decision. True, but which ones are even SUFFICIENT is arithmetic, and until it is done
    the design decision is being made blind.

    MEASURED 2026-08-02, 144 combinations x 441 poses. Four results the parts list could not show:

      * EXTRA REDUCTION IS NOT OPTIONAL -- 0 of the 76 feasible configurations do without it. No
        combination of payload cut, reach cut, counterbalance and a second servo closes the gap
        alone. A parallel servo doubles the SHOULDER and does nothing for the elbow, which binds.
      * THE ORIGINAL SPEC IS NOT REACHABLE. 500 g AND 300 mm together: **0 configurations**. One of
        the two targets has to give, and that is the decision this search exists to surface.
      * Keep 300 mm -> payload falls to 0.35 kg (best margin 1.25x at 4:1).
        Keep 500 g -> reach falls to 255 mm (best margin 1.15x at 4:1, wrist-limited).
      * 3:1 technically passes at 255 mm but leaves the elbow at 12.26 of 12.45 -- a 1.02x margin.
        The notes' own rule is to size with headroom, so 4:1 is the honest choice even though 3:1
        clears the bar.

    `cost` counts how many things you have to change, so the cheapest sufficient recipe sorts first.
    Returns None without mujoco."""
    import itertools  # noqa: PLC0415

    aliases = [j.alias for j in _arm_config.JOINTS]
    rated = {r["joint"]: r["rated_kgcm"] for r in budget()}
    sweeps, out, tried = {}, [], 0
    for payload, reach in itertools.product(MITIGATIONS["payload_kg"], MITIGATIONS["reach_scale"]):
        key = (payload, reach)
        if key not in sweeps:
            got = _worst_torque(payload, reach, grid)
            if got is None:
                return None
            sweeps[key] = got
        total, link_only = sweeps[key]
        for cbal, gear, par in itertools.product(
            MITIGATIONS["counterbalance"],
            MITIGATIONS["extra_reduction"],
            MITIGATIONS["parallel_shoulder"],
        ):
            tried += 1
            need = list(total)
            if cbal:  # a spring cancels the arm's own weight at the shoulder, never the payload's
                need[1] = max(0.0, need[1] - link_only[1])
            have = [rated.get(a, 0.0) for a in aliases[: len(need)]]
            have[1] *= gear * par
            have[2] *= gear
            if not all(n <= h + 1e-9 for n, h in zip(need, have, strict=False)):
                continue
            out.append(
                {
                    "cost": sum(
                        (
                            payload != TARGET["payload_kg"],
                            reach != 1.0,
                            bool(cbal),
                            gear > 1.0,
                            par > 1.0,
                        )
                    ),
                    "payload_kg": payload,
                    "reach_scale": reach,
                    "reach_mm": round(TARGET["reach_mm"] * reach, 1),
                    "counterbalance": bool(cbal),
                    "extra_reduction": gear,
                    "parallel_shoulder": par,
                    "tightest_joint": aliases[
                        max(range(len(need)), key=lambda i: need[i] / have[i] if have[i] else 0.0)
                    ],
                    "tightest_margin_x": round(
                        min((h / n) for n, h in zip(need, have, strict=False) if n > 1e-6), 2
                    ),
                }
            )
    # Ties on cost break on MARGIN, not on the smaller gear. Sorting cheapest-first alone put a
    # 1.02x elbow at the top and called it the recommendation -- which contradicts the sizing rule
    # this whole module exists to enforce. Same cost, more headroom, wins.
    out.sort(
        key=lambda c: (c["cost"], -c["tightest_margin_x"], -c["payload_kg"], -c["reach_scale"])
    )
    return {"tried": tried, "buildable": len(out), "configurations": out}


#: Memo for `state()`. NOT an optimisation looking for a problem: py-spy on the live daemon
#: (pid 36224, 2,248 samples over 30 s, 2026-08-04) put `_worst_torque` at 15.6 + 1.8 + 1.1
#: percent, `model_budget` at 2.4 and `buildable_configurations` at 1.3 -- about 22 percent of
#: every tick spent re-deriving a 625-pose torque sweep and a 21-grid recipe search.
#:
#: The result is a pure function of module-level constants: the arm does not exist (`built` is
#: False), TARGET and the MuJoCo model geometry do not change at runtime, and nothing here reads
#: live arm state. So the answer is identical on every call, and computing it once per process is
#: not a cache-coherency risk -- it is the same value, arrived at 300 times less often.
#:
#: If any input above ever DOES become runtime-variable, this memo becomes wrong and the fix is to
#: key it on that input, not to delete the assertion that catches it.
_STATE_MEMO = None


def state():
    """The hardware section for telemetry. Explicitly a TARGET; `built` is False.

    Memoised: see `_STATE_MEMO`. The sweep behind this was ~22 percent of the live tick.
    """
    global _STATE_MEMO
    if _STATE_MEMO is not None:
        # Shallow copy: a caller that adds or drops a key must not corrupt every later snapshot.
        # ponytail: nested values are still shared. Telemetry only serialises this, so a deep copy
        # would cost more than it protects -- if a caller ever mutates a nested list, deepcopy here.
        return dict(_STATE_MEMO)
    rows = budget()
    measured = model_budget()
    recipes = buildable_configurations()
    out = {
        "built": False,  # nothing here is measured; this is a plan
        "target_dof": TARGET["dof"],
        "reach_mm": TARGET["reach_mm"],
        "payload_kg": TARGET["payload_kg"],
        "bus_volts": TARGET["bus_volts"],
        "peak_amps_3_stall": round(peak_current_amps(3), 1),
        "supply_amps": TARGET["supply_amps"],
        "supply_ok": peak_current_amps(3) <= TARGET["supply_amps"],
        "joints": rows,
        "underpowered": [r["joint"] for r in rows if not r["ok"]],
        # The model-derived budget, when the model is loadable. Reported ALONGSIDE the formula
        # rather than replacing it, so the two can be compared and the formula's optimism stays
        # visible instead of being quietly corrected away.
        "torque_source": "model" if measured else "formula",
        "measured_torque": measured,
        "underpowered_measured": (measured or {}).get("underpowered", []),
        "poses_within_rating": (measured or {}).get("poses_within_rating"),
        "poses_swept": (measured or {}).get("poses_swept"),
        # HOW TO MAKE IT BUILDABLE. The header lists five ways out and says choosing among them is a
        # design decision; which ones SUFFICE is arithmetic, and this is it. The headline is that the
        # original spec is not one of the answers: 500 g at 300 mm has zero feasible configurations.
        "buildable": bool(recipes and recipes["buildable"]),
        "buildable_count": (recipes or {}).get("buildable"),
        "buildable_tried": (recipes or {}).get("tried"),
        "buildable_recipes": (recipes or {}).get("configurations", [])[:6],
        "sim_dof": SIM_GAP["sim_dof"],
        "sim_units": SIM_GAP["sim_units"],
        "sim_has_dynamics": SIM_GAP["sim_has_dynamics"],
    }
    _STATE_MEMO = out
    return out


def main():
    """Self-check: the torque budget is arithmetic, and the shoulder shortfall must stay visible."""
    st = state()
    assert st["built"] is False, "this module must never claim the arm exists"

    rows = {r["joint"]: r for r in st["joints"]}

    # the headline finding, asserted so a future edit cannot quietly erase it
    sh = rows["shoulder_pitch"]
    assert sh["required_kgcm"] == 15.0, (
        f"payload-only shoulder torque changed: {sh['required_kgcm']}"
    )
    assert sh["rated_kgcm"] == 7.5, f"ST3215 1:345 rated changed: {sh['rated_kgcm']}"
    assert not sh["ok"], "shoulder must read UNDERPOWERED for 500g at 300mm"
    assert sh["margin_x"] == 0.5, f"shoulder margin changed: {sh['margin_x']}"

    # THE MODEL IS HARSHER THAN THE FORMULA, and that must stay visible too. The formula omits the
    # arm's own 0.632 kg and uses a hand-written 300 mm lever against the model's actual 254.4 mm
    # horizontal reach at q=0; the two errors do not cancel and the omission dominates. MEASURED
    # 2026-08-02 over 25 x 25 poses at 0.5 kg: shoulder 24.41 vs the formula's 15.00, elbow 14.43 vs
    # 9.00, and wrist_pitch 4.25 against a 4.15 rating -- a THIRD underpowered joint the formula
    # reports as ok=True. Asserted so a future edit cannot quietly restore the optimistic picture.
    meas = model_budget()
    if meas is not None:  # mujoco absent -> the formula stands alone, and state() says so
        mrows = {r["joint"]: r for r in meas["joints"]}
        assert mrows["shoulder_pitch"]["required_kgcm"] > sh["required_kgcm"], (
            "the model must not read EASIER than the payload-only formula: "
            f"{mrows['shoulder_pitch']['required_kgcm']} vs {sh['required_kgcm']}"
        )
        assert not mrows["wrist_pitch"]["ok"], (
            "wrist_pitch is underpowered once the arm's own mass is counted; the formula says ok"
        )
        assert set(meas["underpowered"]) >= {"shoulder_pitch", "elbow_pitch", "wrist_pitch"}, (
            f"measured underpowered set shrank: {meas['underpowered']}"
        )
        assert meas["poses_within_rating"] < meas["poses_swept"], (
            "a fully feasible workspace would contradict every row above"
        )
        print(
            f"torque (model)  : {len(meas['underpowered'])} of 6 joints underpowered; "
            f"{meas['poses_within_rating']} of {meas['poses_swept']} swept poses hold at "
            f"{meas['payload_kg']} kg"
        )

        # THE WAY OUT IS ARITHMETIC, and these three facts are the deliverable. Asserted so a future
        # edit that widens a rating or quietly restores the original spec has to argue with them.
        rec = buildable_configurations()
        assert rec and rec["buildable"], "no configuration closes the gap -- that would be news"
        assert not [c for c in rec["configurations"] if c["extra_reduction"] <= 1.0], (
            "a configuration with no extra reduction passed; every measured one needs gearing"
        )
        assert not [
            c
            for c in rec["configurations"]
            if c["payload_kg"] == TARGET["payload_kg"] and c["reach_scale"] == 1.0
        ], "500 g at 300 mm reads BUILDABLE -- the original spec was measured as unreachable"
        # TWO archetypes, not one "best". Which target to sacrifice is the design decision this
        # module has always refused to presume; what it can do is price both sides of it.
        keep_reach = next((c for c in rec["configurations"] if c["reach_scale"] == 1.0), None)
        keep_load = next(
            (c for c in rec["configurations"] if c["payload_kg"] == TARGET["payload_kg"]), None
        )
        print(f"buildable       : {rec['buildable']} of {rec['tried']} configurations; 0 keep BOTH")
        for label, c in (("keep 300mm reach", keep_reach), ("keep 0.5kg payload", keep_load)):
            if c:
                print(
                    f"  {label:18s}: {c['payload_kg']} kg at {c['reach_mm']} mm, "
                    f"{c['extra_reduction']:.0f}:1 reduction"
                    f"{', counterbalanced' if c['counterbalance'] else ''}"
                    f"{', 2 shoulder servos' if c['parallel_shoulder'] > 1 else ''}"
                    f"  (tightest {c['tightest_joint']} {c['tightest_margin_x']}x)"
                )

    # a vertical axis carries no gravity moment
    assert rows["base_yaw"]["required_kgcm"] == 0.0, "base yaw should carry no gravity moment"

    # rated must be a quarter of stall, per the notes' own rule
    assert abs(rated_kgcm("ST3215-1:345", 12.0) - 7.5) < 1e-9, "rated/stall rule broken"
    assert rated_kgcm("ST3215-1:345", 7.4) < rated_kgcm("ST3215-1:345", 12.0), "12V must beat 7.4V"

    # adding arm mass can only make it worse, never better
    worse = {r["joint"]: r for r in budget(arm_mass_kg=0.4)}["shoulder_pitch"]
    assert worse["required_kgcm"] > sh["required_kgcm"], "arm mass must increase required torque"

    # supply sizing: 3 servos stalling at 2.7A must fit the specified supply
    assert st["supply_ok"], "12V supply undersized for 3 simultaneous stalls"

    # the sim/target gap must stay stated, not smoothed over
    # The DOF and dynamics gaps are closed, and saying so is the honest report. The assertion
    # that replaces them guards the claim that actually still needs guarding: this module must
    # never say the arm exists, and the torque shortfall must never quietly become "fine".
    assert SIM_GAP["sim_dof"] == SIM_GAP["target_dof"], (
        "the simulated body no longer matches the target DOF -- one of them changed"
    )
    assert SIM_GAP["sim_has_dynamics"] is True, "the SO-101 sim HAS dynamics; do not understate it"
    assert SIM_GAP["missing_dof"] == (), "a joint went missing from the simulated body"
    assert not sh["ok"] and st["built"] is False, (
        "a faithful simulation is not a built arm: the shoulder is still underpowered"
    )

    under = ", ".join(st["underpowered"]) or "none"
    print(
        f"arm_hardware OK: target {st['target_dof']}-DOF, {st['reach_mm']:.0f}mm, "
        f"{st['payload_kg']}kg payload | shoulder needs {sh['required_kgcm']} kg.cm vs "
        f"{sh['rated_kgcm']} rated ({sh['margin_x']}x) | UNDERPOWERED: {under} | "
        f"sim is {st['sim_dof']}-DOF {st['sim_units']}, dynamics={st['sim_has_dynamics']}"
    )


if __name__ == "__main__":
    main()
