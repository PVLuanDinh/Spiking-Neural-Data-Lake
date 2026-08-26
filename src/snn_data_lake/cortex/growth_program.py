r"""Growth-program locality -- ONE shared per-node program, a SEALED local view, dilution
counters, and the scaffold-region refusal (specs/022 T042, L5).

P8 invention #7's third slice (docs/INVENTION_REGISTER.md row 7; T040/T041 landed the
fields, the censuses, the prosomeric grid and the two regimes in `cortex/arealization`).
ADOPTED part here: CX3D's one-shared-program locality -- every node runs THE SAME
program, and that program reads ONLY the node's own locals. What this build ships is the
ENFORCEMENT INFRASTRUCTURE and its reds; the growth-policy CONTENT (what the program
decides, the stopping criterion, the coverage gate) is later work, and this module says
so rather than growing a policy nobody has gated yet.

THE PIECES, EACH WITH A RED IN THE SELF-CHECK (the T042 review panel killed the first
draft's claims three ways and every kill is kept: the one-shared-program red DID NOT
EXIST -- 'identity-asserted' was written and never implemented, and a stateless program
made 1-vs-N instances observationally identical, so the class now carries a step counter
that makes sharing OBSERVABLE and the red REACHABLE; the QUIESCE arm short-circuited the
region gate, so a dilution-exhausted brainstem proposal was ACCEPTED -- region is checked
FIRST for every action now; and the growable guard pinned a CARDINALITY, not the ten
ratified names -- both sides of the owner's partition are equality-pinned now):

1. ONE SHARED PROGRAM, OBSERVABLE. `GrowthProgram` counts its steps (`steps_taken`);
   the self-check steps every node through one instance and asserts the counter equals
   the node count -- an N-instance mutant leaves the shared counter short, which is the
   reachable red the first draft only claimed.

2. THE SEALED VIEW. `LocalView` is the ONLY surface `step()` receives, and it is sealed
   BOTH ways: `__getattribute__` permits exactly the seven declared names and raises
   `LocalityError` on everything else -- including dunder hops (`view.__class__` was a
   three-hop escape to every module global in the first draft; the panel built it, and
   the self-check now runs that exact hop as a red) -- and `__setattr__` raises after
   construction, so a program cannot steer a node through its view. STATED LIMIT, not
   smoothed: Python offers no true sandbox -- `type(view)` via the builtin still reaches
   the class, and a program importing globals directly was never stoppable by a view
   object. This is a HARDENED MISTAKE-CATCHER with measured escapes closed, not a
   security boundary, and the honest route to a global remains a reviewed diff adding
   it to the declared surface.

3. DILUTION COUNTERS. Two floats per node, both HALVED on division (the CX3D molecule
   dilution mechanic). Halving by 0.5 is EXACT in binary floating point -- asserted down
   an 11-step chain seeded with UNEQUAL counters (the panel's minor: equal-seeded
   counters made the per-depth assert tautological on the second float). P1: counters
   initialize at 1.0, the normalized full concentration (case (a)).

4. THE DIVISION BUDGET, DECLARED -- NOT DERIVED. `_DIVISION_BUDGET = 10` is a PICKED
   budget (the first draft called 2**-10 'derived' and the panel's M2 called the
   circularity: nothing in the spec sources the exponent); it exists so the minimal
   program has ANY terminating behavior to red-test. SUPERSESSION RE-ADJUDICATED AT
   T047 (the earlier text promised "T047's measured stopping criterion replaces it"
   -- an exemption-that-promises, and the promise was at the WRONG GRAIN): T047's
   `cortex/growth_stop.py` gates per-REGION refit growth on (units, held-out acc)
   evidence rows; this budget bounds per-NODE embryonic division, which produces NO
   refit rows, so there is nothing for that criterion to read here until the division
   machinery is composed into a live path that measures its outcomes. The budget
   stays PICKED with its supersession OWED to the task that wires region-wide growth
   from seed (the growth-coverage gate's 1-of-10 red is that work's standing driver;
   no task id is claimed -- said plainly, never dressed as dated). The floor
   comparison is <= (the first self-check run caught strict < proposing an eleventh
   division at the exact boundary).

5. SCAFFOLD-REGION REFUSAL, CONTENT-PINNED AND REGION-FIRST. spec.md's owner-ratified
   partition (session 2026-08-19) is pinned by EQUALITY on both sides: GROWABLE is
   exactly the ten names, SCAFFOLD is exactly {brainstem, medulla, pons, olfactory,
   insula}, the two are disjoint, and a member swap in either direction reds the
   self-check (the first draft pinned only a count of 10, which a content swap
   satisfied). The gate checks the REGION FIRST for every action: a proposal naming a
   scaffold or unknown region is refused whether it says DIVIDE or QUIESCE (the panel
   proved a live path handing consumers accepted=True on a brainstem-targeting
   quiescence). Regions in neither ratified list (e.g. neuromodulation) are refused
   fail-closed AS UNRATIFIED, and the reason says which case fired. The envelope
   itself is never touched; this gate exists so no future growth policy can reach
   TOWARD it.

The program's decision content, stated honestly: quiesce at an exhausted division
budget, else propose division in the node's OWN region (locality applied to the
proposal itself). Deliberately minimal -- it exercises the infrastructure while
deciding nothing beyond it; T046's coverage gate and T047's stopping criterion are
BUILT now, and both police the REFIT grain, not this division grain (adjudication in
point 4 above).

DATA-MODEL SPLIT, stated against specs/022 data-model.md's GrowthProgram entry so the
partial node is declared, never silent: that contract lists per-node state as {field
vector p, dilution counters, calcium EWMA + element counters}. THIS build ships the
dilution counters and samples the field at view time (equivalent access to p, no stored
copy to go stale); the calcium EWMA + element counters are the STRUCTURAL CLOCKS of
T048's turnover instrument and land there; the contract's "one MLP" half of the shared
program is the LEARNED policy content the later composition supplies. Three tasks, one
node contract, each piece named at its owner.

# distinct-from: cortex/arealization.py -- that module EMITS region identities from
#   fields; this one polices HOW a per-node program may read them and WHERE growth may
#   go. arealization is this module's field source, not a duplicate. Read.
# distinct-from: cortex/neurogenesis_engine.py -- the torch growth ENGINE (fits models,
#   keeps substrate); this is the locality/refusal POLICY layer for the seed-grown
#   micro-brain lane; the 007 interface composes them at T043. Read.
# distinct-from: hippocampus/function_preserving_growth.py + substrate/growable.py --
#   Net2Net widening mechanics for live nets; no per-node program, no locality mask,
#   no region proposals. Read.
# distinct-from: substrate/safety_envelope.py -- that gate hashes the five envelope
#   FILES; this refusal polices growth PROPOSALS by region name. Different layer,
#   deliberately redundant in the brainstem direction. Read.
# distinct-from: basal_ganglia/action_gating.py + cortex/snn_moe_classifier.py +
#   cerebellum/hjepa_serve.py + cortex/interaction_learning.py +
#   cortex/autapse_neuron.py -- trial builders, expert forwards, action one-hots and
#   delayed-match harnesses; none is a growth program, a locality mask, or a region
#   gate. Read.
"""

import math

from snn_data_lake.cortex.arealization import (
    NEUROMERES,
    REGION_COORDS,
    ZONES,
    OrganizerField,
    regime_of,
)

# spec.md session 2026-08-19, OWNER-RATIFIED, both sides pinned by EQUALITY in the
# self-check (a swap in either direction reds).
GROWABLE = frozenset(
    {
        "cortex",
        "hippocampus",
        "thalamus",
        "basal_ganglia",
        "cerebellum",
        "amygdala",
        "hypothalamus",
        "midbrain",
        "epithalamus",
        "claustrum",
    }
)
SCAFFOLD = frozenset({"brainstem", "medulla", "pons", "olfactory", "insula"})

# DECLARED budget, not derived (the panel's M2 killed the 'derived' label: nothing
# sources the exponent). Supersession RE-ADJUDICATED at T047: growth_stop.py gates the
# REFIT grain and cannot read division outcomes -- this budget's replacement is owed to
# the task that composes region-wide growth live (docstring point 4). The budget exists
# so the minimal program has terminating behavior with a reachable red.
_DIVISION_BUDGET = 10  # PICKED
_DILUTION_FLOOR = 0.5**_DIVISION_BUDGET

_LV_SLOTS = ("ap", "region", "neuromere", "zone", "regime", "field_a", "dilution")


class LocalityError(AttributeError):
    """A growth program touched something outside its node's locals."""


class GrowthProposal:
    """What a node's program is allowed to say: an action and a target region."""

    __slots__ = ("action", "region")

    def __init__(self, action, region):
        if action not in ("DIVIDE", "QUIESCE"):
            raise ValueError(f"unknown growth action {action!r}")
        self.action = action
        self.region = str(region)


class LocalView:
    """The ONLY surface a growth program receives: this node's locals, SEALED.

    `__getattribute__` permits exactly the declared names; everything else -- dunder
    hops included -- raises LocalityError. `__setattr__` raises after construction.
    See the module docstring for the stated limits of what a view object can police."""

    __slots__ = _LV_SLOTS

    def __init__(self, ap, region, dilution):
        if region not in REGION_COORDS:
            raise ValueError(f"unknown region {region!r} -- a node must sit on the prosomeric grid")
        nm, zone = REGION_COORDS[region]
        object.__setattr__(self, "ap", float(ap))
        object.__setattr__(self, "region", str(region))
        object.__setattr__(self, "neuromere", nm)
        object.__setattr__(self, "zone", zone)
        object.__setattr__(self, "regime", regime_of(nm))
        object.__setattr__(self, "field_a", float(OrganizerField(1.0, 0.0).sample([float(ap)])[0]))
        object.__setattr__(self, "dilution", tuple(dilution))

    def __getattribute__(self, name):
        if name in _LV_SLOTS:
            return object.__getattribute__(self, name)
        raise LocalityError(
            f"growth program read non-local attribute {name!r} -- the sealed view "
            f"exists so this is a raise, never a silent global"
        )

    def __setattr__(self, name, value):
        raise LocalityError(
            f"growth program wrote {name!r} through its view -- a view is read-only"
        )


class GrowthNode:
    """One node: a position, a region, two dilution counters. divide() halves BOTH
    counters exactly (x * 0.5 is exact in binary floating point) and returns the
    daughter; the shared program is never stored per-node (one program, many nodes)."""

    __slots__ = ("ap", "region", "dilution")

    def __init__(self, ap, region, dilution=(1.0, 1.0)):
        self.ap = float(ap)
        self.region = str(region)
        self.dilution = [float(dilution[0]), float(dilution[1])]

    def view(self):
        return LocalView(self.ap, self.region, self.dilution)

    def divide(self):
        self.dilution[0] *= 0.5
        self.dilution[1] *= 0.5
        return GrowthNode(self.ap, self.region, tuple(self.dilution))


class GrowthProgram:
    """THE one shared per-node program, with sharing made OBSERVABLE: `steps_taken`
    counts every step, so the self-check's one-instance loop has a reachable red (an
    N-instance mutant leaves any single counter short of the node count -- the panel's
    B0: the first draft claimed this assert and never wrote it, and a stateless class
    could not have supported it)."""

    def __init__(self):
        self.steps_taken = 0

    def step(self, view):
        self.steps_taken += 1
        # <= not <: after exactly _DIVISION_BUDGET divisions the counter EQUALS the
        # floor and the budget is spent -- the first self-check run caught the strict
        # form proposing an eleventh division at the boundary.
        if min(view.dilution) <= _DILUTION_FLOOR:
            return GrowthProposal("QUIESCE", view.region)
        return GrowthProposal("DIVIDE", view.region)


def refuse_or_accept(proposal):
    """The scaffold-region refusal gate, REGION-FIRST for every action (the panel's
    M1/M5/M9: the first draft's QUIESCE early-return handed consumers accepted=True on
    a brainstem-targeting proposal). Growable regions pass; ratified scaffold, and any
    name in NEITHER ratified list, are refused fail-closed with the case named.
    Returns (accepted: bool, reason: str)."""
    region = proposal.region
    if region not in GROWABLE:
        case = "RATIFIED SCAFFOLD" if region in SCAFFOLD else "UNRATIFIED (fail-closed)"
        return False, (
            f"REFUSED [{case}]: {region!r} is not in the owner-ratified growable set -- "
            f"closed to growth proposals regardless of action (the brainstem path toward "
            f"the safety envelope is the constructed red)"
        )
    if proposal.action == "QUIESCE":
        return True, f"quiescence in growable {region} -- nothing grows"
    return True, f"{region} is in the owner-ratified growable set"


def main() -> int:
    prog = GrowthProgram()
    # ---- ONE shared program, OBSERVABLE: the counter red the panel forced -------------
    nodes = [GrowthNode(0.2, "cortex"), GrowthNode(0.8, "thalamus")]
    proposals = [prog.step(n.view()) for n in nodes]
    if prog.steps_taken != len(nodes):
        print(
            f"growth_program self-check FAILED: the shared program's step counter reads "
            f"{prog.steps_taken} for {len(nodes)} nodes -- the nodes were not stepped "
            f"through ONE instance (the panel's N-instance mutant shape)"
        )
        return 1
    if not all(p.action == "DIVIDE" for p in proposals):
        print("growth_program self-check FAILED: fresh nodes did not propose division")
        return 1
    # ---- THE SEALED VIEW: named red, dunder-hop red, write red, and the live control --
    v = nodes[0].view()
    try:
        _ = v.global_population  # the named red: a missing global must raise
        print("growth_program self-check FAILED: a non-local named read did not raise")
        return 1
    except LocalityError:
        pass
    try:
        _ = v.__class__  # the panel's three-hop escape (…__init__.__globals__), sealed
        print(
            "growth_program self-check FAILED: the dunder hop resolved -- "
            "view.__class__ was the panel's measured escape to every module global"
        )
        return 1
    except LocalityError:
        pass
    try:
        v.ap = 0.9  # the write red: a program must not steer a node through its view
        print("growth_program self-check FAILED: a slot write through the view succeeded")
        return 1
    except LocalityError:
        pass
    expect_field = math.exp(-0.2 / OrganizerField(1.0, 0.0).lam)
    if not (abs(v.field_a - expect_field) < 1e-12 and v.regime == "graded" and v.neuromere == "sp"):
        print(
            f"growth_program self-check FAILED: the LOCAL surface misreads (field_a "
            f"{v.field_a} vs expected {expect_field}, regime {v.regime}, neuromere "
            f"{v.neuromere}) -- the positive control beside the locality reds is broken"
        )
        return 1
    # slot descriptors live in vars(LocalView) under the slot names themselves -- the
    # stray-attribute census excludes them (the first run of this assert counted its own
    # seven descriptors as strays and fired on clean code).
    stray = [
        k
        for k in vars(LocalView)
        if not (k.startswith("__") and k.endswith("__")) and k not in _LV_SLOTS
    ]
    if set(LocalView.__slots__) != set(_LV_SLOTS) or stray:
        print(
            f"growth_program self-check FAILED: LocalView's declared surface drifted "
            f"(strays {stray}) -- a class attribute or slot change widened the local "
            f"surface silently"
        )
        return 1
    try:
        LocalView(0.5, "not_a_region", (1.0, 1.0))
        print("growth_program self-check FAILED: an off-grid region built a view")
        return 1
    except ValueError:
        pass
    # ---- DILUTION: halved exactly down an 11-step chain with UNEQUAL counters ---------
    n = GrowthNode(0.5, "hippocampus", dilution=(1.0, 0.75))
    for k in range(1, 12):
        daughter = n.divide()
        e0, e1 = 0.5**k, 0.75 * (0.5**k)
        if not (
            n.dilution[0] == e0
            and n.dilution[1] == e1
            and daughter.dilution[0] == e0
            and daughter.dilution[1] == e1
        ):
            print(
                f"growth_program self-check FAILED: dilution after {k} divisions reads "
                f"{n.dilution} (daughter {daughter.dilution}), expected exactly "
                f"({e0}, {e1})"
            )
            return 1
    # the DECLARED budget floor binds: an over-budget node must QUIESCE
    p = prog.step(n.view())
    if p.action != "QUIESCE":
        print(
            f"growth_program self-check FAILED: a budget-exhausted node proposed "
            f"{p.action} -- the declared division budget does not bind"
        )
        return 1
    # ---- THE REFUSAL GATE, region-first: every action, both partitions, fail-closed ---
    for region in sorted(SCAFFOLD) + ["neuromodulation", "no_such_region"]:
        for action in ("DIVIDE", "QUIESCE"):
            ok, reason = refuse_or_accept(GrowthProposal(action, region))
            if ok:
                print(
                    f"growth_program self-check FAILED: {action} targeting {region!r} "
                    f"was ACCEPTED -- the refusal gate is open in a direction it exists "
                    f"to close (the panel's QUIESCE short-circuit shape)"
                )
                return 1
    ok, _reason = refuse_or_accept(GrowthProposal("DIVIDE", "cortex"))
    ok_q, _reason_q = refuse_or_accept(GrowthProposal("QUIESCE", "cortex"))
    if not (ok and ok_q):
        print(
            "growth_program self-check FAILED: a growable-region proposal was refused "
            "-- the gate cannot distinguish the partitions, so its reds are meaningless"
        )
        return 1
    # ---- the partition is pinned by CONTENT, both sides (a swap reds either way) ------
    expect_growable = {
        "cortex",
        "hippocampus",
        "thalamus",
        "basal_ganglia",
        "cerebellum",
        "amygdala",
        "hypothalamus",
        "midbrain",
        "epithalamus",
        "claustrum",
    }
    expect_scaffold = {"brainstem", "medulla", "pons", "olfactory", "insula"}
    if (
        GROWABLE != frozenset(expect_growable)
        or SCAFFOLD != frozenset(expect_scaffold)
        or not GROWABLE.isdisjoint(SCAFFOLD)
        or not GROWABLE.issubset(REGION_COORDS.keys())
        or not SCAFFOLD.issubset(REGION_COORDS.keys())
    ):
        print(
            "growth_program self-check FAILED: the owner-ratified partition drifted -- "
            "a name moved between growable and scaffold, or off the prosomeric grid "
            "(spec.md session 2026-08-19 is the authority)"
        )
        return 1
    print(
        f"growth_program self-check OK: one shared program stepped {len(nodes)} nodes "
        f"(counter-asserted, steps_taken={prog.steps_taken + 1} incl. the budget probe); "
        f"sealed view: named red, dunder-hop red (the panel's __class__ escape), write "
        f"red and off-grid red all fired beside the exact-value control (field_a "
        f"{v.field_a:.6f} == exp(-0.2/lam)); dilution exact through 11 divisions on "
        f"UNEQUAL counters; DECLARED division budget {_DIVISION_BUDGET} binds (QUIESCE "
        f"at <= {_DILUTION_FLOOR}); refusal gate REGION-FIRST: both actions refused "
        f"across ratified scaffold {sorted(expect_scaffold)} + unratified "
        f"(neuromodulation) + unknown, both actions pass in growable; partition pinned "
        f"by CONTENT both sides ({len(NEUROMERES)}x{len(ZONES)} grid backing). Policy "
        f"content beyond this is T046/T047's (+T043's composition), on purpose."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
