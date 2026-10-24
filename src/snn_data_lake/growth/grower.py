"""specs/022 G3-T1 (P2, merge step 1 of 3): the Brain's ONE grower, wrapping the percept engine.

# distinct-from: neurogenesis_engine.py -- it WRAPS that engine (B1) and moves no refit code;
# the hits (drive, growth-gate callable, lifecycle) are the engine's own surface, which this
# class forwards to, not a second growth mechanism. Deletion of the duplicate surface is T6.

`Grower` owns a set of Problems; `PerceptProblem` is the adapter holding the existing
`NeurogenesisSelfImprove` engine (B1). Nothing moves: `_refit`, `NGSNN`, `candidate_lifecycle` and
`placement` stay where they are, so every percept-path value is byte-identical by construction.
Forwarding is DYNAMIC (the engine attribute is read at call time), so an instance patch such as
`engine.tick = lambda: None` (the growth_off control) still takes effect through the Grower.
The Grower holds no weights, no ledger and no copy of `W` (one state record). `PerceptProblem` does
not yet implement the GrowthLoop Problem protocol (diagnose/admit/attempt): T5/T6 add it when the
loop folds. No torch import here. Run: `python -m snn_data_lake.growth.grower`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only (no torch import at run time); the sleep-pruner census reads it
    from snn_data_lake.growth.neurogenesis_engine import NeurogenesisSelfImprove


class PerceptProblem:
    """Adapter over the percept neurogenesis engine; the B1 refit stays unmoved in the engine."""

    def __init__(self, engine: NeurogenesisSelfImprove):
        self.engine = engine


class Grower:
    """The one owner of growth. `problems` maps a name to a Problem; T1 holds only `percept`."""

    def __init__(self, percept: PerceptProblem):
        self.percept = percept
        self.problems = {"percept": percept}

    def observe(self, features, category):
        return self.percept.engine.observe(features, category)

    def tick(self):
        return self.percept.engine.tick()


def main():
    calls = []

    class _Engine:
        def observe(self, f, c):
            calls.append(("observe", f, c))

        def tick(self):
            calls.append(("tick",))

    eng = _Engine()
    g = Grower(PerceptProblem(eng))
    g.observe([1.0], 2)
    g.tick()
    eng.tick = lambda: calls.append(("patched",))  # dynamic dispatch: the patch must win
    g.tick()
    assert calls == [("observe", [1.0], 2), ("tick",), ("patched",)], calls
    assert g.problems == {"percept": g.percept}
    print("grower: self-check OK")


if __name__ == "__main__":
    main()
