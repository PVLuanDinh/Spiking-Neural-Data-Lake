"""
Unified FUNCTION-PRESERVING growth substrate  (deterministic).

The bounded-brain audit found two things: (1) the brain re-implements the SAME growth pattern in
many places (ng_snn contribution growth, growing_learner neurogenesis, function_preserving_growth
Net2Net, the multimodal encoder's band-append, the amygdala's per-US channels), and (2) many
regions still have a FIXED architectural width / population with no growth at all. This module is
the ONE shared growth primitive every region can adopt, so that:

  * no adopting region keeps a fixed ceiling -- width, population, and fan-in all GROW ON DEMAND;
  * growth is FUNCTION-PRESERVING -- appended units are zero-initialised, so the region's output on
    every previously-seen input is BIT-IDENTICAL after growth (no relearning, no forgetting);
  * connected regions GROW TOGETHER -- a GrowthCoordinator ripples a producer's width growth to
    every registered consumer's input, so the connectome stays shape-consistent automatically.

This is the "unbounded + working together" seam. It deliberately does NOT touch the safety
envelope: the arm DOF, the motor action set, the joint limits, and the E-STOP reflex are fixed BY
DESIGN. Unbounding an effector / actuator / reflex is a safety hazard, not a capability win, so
GrowthCoordinator.register_seam rejects any seam whose name looks like that envelope (is_safety_seam).

HONEST SCOPE OF THAT REFUSAL: it is a naming TRIPWIRE (defense-in-depth), NOT a proof. connect()
takes an opaque callback, so a determined caller could still wire a mutator under a benign name --
the filter catches accidents, not sabotage. The ACTUAL safety guarantee is architectural and
independent of this module: no code wires an effector mutator to a coordinator, and every motor
command still flows through the brainstem So101Arm.apply() gate, which the coordinator neither touches
nor can bypass. The tripwire lowers the chance of an accidental wiring; the brainstem gate is what
actually bounds the arm.

Honest scope: "unbounded" here means "no fixed architectural ceiling; grows on demand", NOT
"computes anything". The fundamental wall -- uncomputable functions, and undecidable rule
identification from finite data -- is untouched by any amount of growth. Growth moves the frontier;
it never reaches omniscience.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.growable
"""

import re

# Effector / actuator / reflex vocabulary that must NEVER become a growth seam. is_safety_seam
# normalizes a name (case, separators) and matches these as tokens or compounds, so near-miss
# variants ("ARM_DOF", "e-stop", "effector_servo", "actuator_count") are caught too, not just exact
# strings. This is a defense-in-depth tripwire, not a proof (see the module docstring).
_SAFETY_TOKENS = frozenset(
    {
        "arm",
        "dof",
        "joint",
        "joints",
        "motor",
        "motors",
        "action",
        "actions",
        "actuator",
        "effector",
        "servo",
        "servos",
        "reflex",
        "reflexes",
        "estop",
        "torque",
    }
)
_SAFETY_COMPOUNDS = ("estop", "emergencystop", "armdof", "jointlimit")

# Back-compat: the original exact-name set some callers imported. is_safety_seam is the real check.
_SAFETY_SEAMS = frozenset({"arm_dof", "motor_actions", "joint_limits", "estop", "reflex"})


def is_safety_seam(name):
    """True if `name` looks like an effector / actuator / reflex seam (normalized token or compound
    match). Keeps the motor / safety envelope OUT of the growth network."""
    low = str(name).lower()
    if set(re.split(r"[^a-z0-9]+", low)) & _SAFETY_TOKENS:
        return True
    collapsed = re.sub(r"[^a-z0-9]", "", low)
    return any(c in collapsed for c in _SAFETY_COMPOUNDS)


# ---- function-preserving growth primitives ----------------------------------
def grow_vector(vec, new_len, fill=0.0):
    """Grow a 1-D vector to new_len by APPENDING `fill` entries. Function-preserving: every existing
    entry is unchanged. Monotone (grow-only) so a grown substrate can never forget by shrinking."""
    n = len(vec)
    if new_len < n:
        raise ValueError(f"growth is monotone: cannot shrink {n} -> {new_len}")
    return list(vec) + [fill] * (new_len - n)


def grow_rows(mat, new_rows, ncols=None, fill=0.0):
    """Append zero ROWS to a 2-D weight matrix -- i.e. add output units / grow a population. The new
    rows are all `fill` (zero), so they emit 0 until trained and the EXISTING outputs are unchanged.
    `ncols` is required only to grow an empty matrix."""
    cur = len(mat)
    if new_rows < cur:
        raise ValueError(f"growth is monotone: cannot drop rows {cur} -> {new_rows}")
    if mat:
        w = len(mat[0])
    elif ncols is not None:
        w = ncols
    else:
        raise ValueError("an empty matrix needs ncols to grow")
    return [list(r) for r in mat] + [[fill] * w for _ in range(new_rows - cur)]


def grow_cols(mat, new_cols, fill=0.0):
    """Append zero COLUMNS to every row -- i.e. widen each unit's fan-in / input width. A zero weight
    on a new input multiplies that input by 0, so the output is unchanged FOR ANY value of the new
    input (the GradMax / Net2Net zero-fanout trick). This is what lets a consumer accept a producer's
    grown output with no behavioural change until it learns to use the new channels."""
    if not mat:
        return []
    cur = len(mat[0])
    if new_cols < cur:
        raise ValueError(f"growth is monotone: cannot drop columns {cur} -> {new_cols}")
    pad = new_cols - cur
    return [list(r) + [fill] * pad for r in mat]


# ---- growth coordination across the connectome ------------------------------
class GrowthCoordinator:
    """Registers producer -> consumer seams so connected regions grow TOGETHER.

    A seam is a named width (a producer region's output width). Consumers register a resize callback
    that grows their INPUT to a new width function-preservingly. `grow(seam, new_width)` records the
    producer's new width then fires every consumer callback, so ONE growth event ripples through the
    connectome and leaves every wired region shape-consistent -- the regions support each other
    instead of each capping independently.

    Safety invariant: register_seam refuses any name in _SAFETY_SEAMS, so the effector / action /
    reflex envelope can never be wired into the growth network."""

    def __init__(self):
        self._width = {}  # seam name -> current width
        self._consumers = {}  # seam name -> [resize_fn(new_width) -> None, ...]

    def register_seam(self, name, width):
        """Declare a growable seam at an initial width. Rejects any effector/actuator/reflex-looking
        name (is_safety_seam tripwire) and refuses a duplicate."""
        if is_safety_seam(name):
            raise ValueError(
                f"refusing seam {name!r}: names that look like the effector / action / E-STOP "
                f"envelope are kept out of the growth network (that envelope is bounded by design)"
            )
        if name in self._width:
            raise ValueError(f"seam {name!r} already registered")
        self._width[name] = width
        self._consumers.setdefault(name, [])

    def connect(self, name, resize_fn):
        """Wire a consumer to a seam. resize_fn(new_width) must grow the consumer's input to
        new_width function-preservingly. The seam must exist."""
        if name not in self._width:
            raise KeyError(f"seam {name!r} is not registered")
        self._consumers[name].append(resize_fn)

    def grow(self, name, new_width):
        """Grow seam `name` to new_width and ripple to every wired consumer. Monotone (grow-only).
        Returns the number of consumers resized."""
        cur = self._width.get(name)
        if cur is None:
            raise KeyError(f"seam {name!r} is not registered")
        if new_width < cur:
            raise ValueError(
                f"growth is monotone: cannot shrink seam {name!r} {cur} -> {new_width}"
            )
        self._width[name] = new_width
        for fn in self._consumers[name]:
            fn(new_width)
        return len(self._consumers[name])

    def width(self, name):
        """Current width of a seam (0 if unknown)."""
        return self._width.get(name, 0)

    def seams(self):
        """Registered seam names."""
        return sorted(self._width)


def main():
    print("=" * 70)
    print("GROWABLE SUBSTRATE -- function-preserving growth + connectome coordination")
    print("=" * 70)

    # ---- 1. primitives are function-preserving --------------------------------
    # A tiny linear unit y = sum(w[i]*x[i]). Grow its fan-in with zero columns: the output on the
    # old inputs (padded with ARBITRARY new-input values) is unchanged, because new weights are 0.
    W = [[0.5, -0.2, 0.3]]  # 1 output unit, fan-in 3
    x = [1.0, 2.0, -1.0]
    y0 = sum(W[0][i] * x[i] for i in range(3))
    W2 = grow_cols(W, 5)  # widen fan-in 3 -> 5 (zero-init)
    x2 = x + [9.9, -4.2]  # two NEW inputs, arbitrary values
    y1 = sum(W2[0][i] * x2[i] for i in range(5))
    assert W2[0][:3] == W[0] and W2[0][3:] == [0.0, 0.0], (
        "grow_cols must append zeros, keep old weights"
    )
    assert y1 == y0, "grow_cols must be function-preserving for ANY new input (zero fan-in)"

    # Grow output population with zero rows: existing outputs unchanged, new unit emits 0.
    W3 = grow_rows(W2, 3)  # 1 -> 3 output units
    outs = [sum(W3[r][i] * x2[i] for i in range(5)) for r in range(3)]
    assert outs[0] == y0 and outs[1] == 0.0 and outs[2] == 0.0, (
        "grow_rows keeps old outputs, new units emit 0"
    )

    # Grow a state vector: old entries intact.
    v = grow_vector([1.0, 2.0, 3.0], 6)
    assert v == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0], "grow_vector appends zeros, preserves old state"

    # Monotone: shrinking is refused everywhere (a grown brain never forgets by shrinking).
    for bad in (
        lambda: grow_vector([1, 2, 3], 2),
        lambda: grow_rows([[1, 2]], 0),
        lambda: grow_cols([[1, 2, 3]], 2),
    ):
        try:
            bad()
            raised = False
        except ValueError:
            raised = True
        assert raised, "growth must be monotone (shrink refused)"
    print("1. primitives             : grow_cols/rows/vector function-preserving + monotone  OK")

    # ---- 2. connectome coordination: A -> B -> C grow together ----------------
    # Three linear regions chained. B consumes A's output; C consumes B's output. Growing A's width
    # must ripple: B's fan-in grows, and (as B re-emits at a wider output) C's fan-in grows too --
    # every region stays shape-consistent, function-preservingly.
    A_w = 3
    B = {"W": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}  # B: 2 units, fan-in = A_w=3
    C = {"W": [[1.0, -1.0]]}  # C: 1 unit, fan-in = B_out=2
    a_in = [1.0, 1.0, 1.0]
    b_out0 = [sum(B["W"][r][i] * a_in[i] for i in range(3)) for r in range(2)]
    c_out0 = sum(C["W"][0][i] * b_out0[i] for i in range(2))

    coord = GrowthCoordinator()
    coord.register_seam("A_out", A_w)  # producer A's output width
    coord.register_seam("B_out", 2)  # producer B's output width
    coord.connect("A_out", lambda w: B.__setitem__("W", grow_cols(B["W"], w)))  # B fan-in follows A
    coord.connect("B_out", lambda w: C.__setitem__("W", grow_cols(C["W"], w)))  # C fan-in follows B

    n_b = coord.grow("A_out", 5)  # grow A's output 3 -> 5; ripple to B
    a_in2 = a_in + [7.0, -3.0]  # A now emits 5 (two new channels, arbitrary)
    b_out1 = [sum(B["W"][r][i] * a_in2[i] for i in range(5)) for r in range(2)]
    assert n_b == 1 and len(B["W"][0]) == 5, "growing A must resize B's fan-in to match"
    assert b_out1 == b_out0, (
        "B must be function-preserving after A grows (new A-channels contribute 0)"
    )

    # B itself grows a new output unit, which widens the B_out seam -> ripples to C.
    B["W"] = grow_rows(B["W"], 3)  # B: 2 -> 3 output units
    coord.grow("B_out", 3)  # announce B's wider output; ripple to C
    b_out2 = [sum(B["W"][r][i] * a_in2[i] for i in range(5)) for r in range(3)]
    c_out1 = sum(C["W"][0][i] * b_out2[i] for i in range(3))
    assert len(C["W"][0]) == 3, "growing B's output must resize C's fan-in to match"
    assert c_out1 == c_out0, (
        "the whole A->B->C chain stays function-preserving after coordinated growth"
    )
    print(
        f"2. connectome coordination: grew A_out 3->5 (rippled to B) and B_out 2->3 (rippled to C); "
        f"chain output bit-identical ({c_out1:.3f})  OK"
    )

    def _refuses(c, name):
        """True if register_seam rejects `name` (safety tripwire fired)."""
        try:
            c.register_seam(name, 2)
            return False
        except ValueError:
            return True

    # ---- 3. safety envelope kept out of the growth network (tripwire, not a proof) ------------
    # Exact names AND near-miss variants (case, separators, compounds) are all refused...
    unsafe = [
        "arm_dof",
        "ARM_DOF",
        "motor_actions",
        "e_stop",
        "e-stop",
        "estop",
        "joint_limits",
        "effector_servo",
        "reflex",
        "EmergencyStop",
        "actuator_count",
    ]
    refused = sum(_refuses(coord, n) for n in unsafe)
    assert refused == len(unsafe), (
        f"every effector/reflex-looking seam must be refused ({refused}/{len(unsafe)})"
    )
    # ...while ordinary neural seams are NOT over-rejected (no false positive).
    coord.register_seam("hippocampal_dim", 64)
    assert "hippocampal_dim" in coord.seams(), "an ordinary neural seam must still register"
    assert not any(_refuses(coord, n) for n in ("sensory_width", "cortical_width", "vsa_dim")), (
        "ordinary neural seam names must not trip the safety filter"
    )
    print(
        f"3. safety exclusion       : {refused} effector/reflex-looking names refused "
        f"(exact + near-miss); ordinary neural seams register  OK"
    )

    print("=" * 70)
    print(
        "self-check OK: one shared growth primitive (function-preserving, monotone), a coordinator "
        "that grows connected regions together, and a naming tripwire that keeps effector / action "
        "/ E-STOP seams out of the growth network (the brainstem So101Arm gate is the real bound)"
    )


if __name__ == "__main__":
    main()
