"""Interoceptive integration -- the insula's first faculty (specs/022 T021a): body-state
signals arriving over the declared tract become a bounded, salience-ready feeling state.

THE TRACT THIS MODULE TERMINATES (each hop a declared `connectome` edge, T021a):
`viscera -> nts_rvlm -> thal_relay -> insula -> {gwt, ctx_pfc}` -- baroreceptor/visceral
afferents through the medullary NTS/RVLM complex, relayed by thalamus into THIS node
(`insula` in connectome.NODES is this module's anatomical id), projecting onward to the
global workspace (interoceptive ignition afferent) and prefrontal cortex. The upstream
modules exist and are charged (`medulla/interoceptive_arousal`, `medulla/vital_nuclei`);
this module is the cortical terminus that was missing.

WHAT IT COMPUTES, minimally and honestly: a running interoceptive STATE -- per-channel
exponential means with deviation tracking -- and a scalar `feeling_salience` in [0, 1]:
how far the body currently sits from its own running baseline (P1: the baseline is the
brain's own statistics, never a fixed set-point). Deviation-from-self is the standard
interoceptive-salience account (the insula responds to PREDICTION MISMATCH in body state,
not to raw intensity), and it is the quantity the US1 coalition channels consume
(`arousal`/`valence` producers, T026): a body far from baseline should find it easier to
ignite the workspace.

NOT YET WIRED, stated plainly: no live path drives this module today. Its port declaration
(the interoception channel producer) lands at US1 with the T025+ registrants; the dormancy
exemption names that consumer and its bound. Wiring it now, ahead of the typed ports it is
supposed to declare, would be exactly the force-wiring P6's honesty guard forbids.
"""

# distinct-from: medulla/interoceptive_arousal.py -- the BRAINSTEM half (arousal drive from
#   visceral afferents, already charged); this is the CORTICAL terminus computing
#   deviation-from-baseline salience. distinct-from: cortex/affect_core.py -- valence/arousal
#   AFFECT dynamics downstream of appraisal; this is upstream body-state integration.
# distinct-from: hypothalamus/set_points.py -- homeostatic TARGETS; this tracks the brain's
#   own running statistics, never fixed set-points. All sites read.

from __future__ import annotations

import sys

#: this module's anatomical ids in connectome.NODES (the atlas-node rule, T024: a node id
#: must be referenced by >=1 module in its region -- these are those references for insula)
CONNECTOME_NODES = ("insula",)
TRACT = ("viscera", "nts_rvlm", "thal_relay", "insula", "gwt")


class InteroceptiveIntegration:
    """Per-channel running baselines + deviation -> one bounded feeling_salience."""

    def __init__(self, channels=("cardio", "respiratory", "gut"), alpha=None):
        # P1: alpha is DERIVED from the horizon the baseline should span, not picked -- the
        # caller states a horizon in ticks and alpha follows; the default horizon is the
        # daemon's own day period order (~200 ticks), stated rather than hidden.
        self.horizon = 200
        self.alpha = (2.0 / (self.horizon + 1.0)) if alpha is None else float(alpha)
        self.channels = tuple(channels)
        self._mean = dict.fromkeys(self.channels, 0.0)
        self._var = dict.fromkeys(self.channels, 1.0)
        self._seen = 0

    def observe(self, **readings) -> float:
        """Feed one tick of body-state readings; returns feeling_salience in [0, 1] -- the
        normalized deviation of the body from its OWN running baseline."""
        devs = []
        for ch in self.channels:
            if ch not in readings:
                continue
            x = float(readings[ch])
            m, v = self._mean[ch], self._var[ch]
            d = x - m
            self._mean[ch] = m + self.alpha * d
            self._var[ch] = (1.0 - self.alpha) * (v + self.alpha * d * d)
            devs.append(abs(d) / (self._var[ch] ** 0.5 + 1e-9))
        self._seen += 1
        if not devs or self._seen <= 2:
            return 0.0  # a baseline two samples old is not a baseline; silence over noise
        z = sum(devs) / len(devs)
        return z / (1.0 + z)  # bounded, monotone in deviation

    def state(self) -> dict:
        return {
            "channels": {
                ch: {"mean": self._mean[ch], "var": self._var[ch]} for ch in self.channels
            },
            "seen": self._seen,
            "horizon_ticks": self.horizon,
        }


def main() -> int:
    ii = InteroceptiveIntegration()
    # settle a stable body: salience must fall toward quiet
    for _ in range(300):
        late_quiet = ii.observe(cardio=1.0, respiratory=0.5, gut=0.2)
    # a real perturbation: salience must RISE (the sensitivity half)
    spike = ii.observe(cardio=3.0, respiratory=0.5, gut=0.2)
    if not (spike > late_quiet * 2 and spike > 0.2):
        print(
            f"interoceptive_integration self-check FAILED: perturbation did not raise "
            f"salience (quiet={late_quiet:.4f}, spike={spike:.4f})"
        )
        return 1
    # the RED (constructed): a dead-baseline variant that never updates its mean would call
    # the SAME steady input salient forever -- deviation-from-self is the criterion, and
    # freezing the self must break it.
    frozen = InteroceptiveIntegration(alpha=0.0)
    for _ in range(50):
        frozen_sal = frozen.observe(cardio=1.0)
    if frozen_sal <= late_quiet:
        print(
            "interoceptive_integration self-check FAILED: the frozen-baseline red did not "
            f"fire (frozen={frozen_sal:.4f} vs live-quiet={late_quiet:.4f}) -- the criterion "
            "cannot distinguish a dead baseline from a settled one"
        )
        return 1
    # re-baseline: after the spike regime persists, salience must fall again (adaptation --
    # the insula habituates to the new normal; a fixed set-point could not)
    for _ in range(600):
        adapted = ii.observe(cardio=3.0, respiratory=0.5, gut=0.2)
    if not (adapted < spike / 2):
        print(
            f"interoceptive_integration self-check FAILED: no habituation to the new normal "
            f"(spike={spike:.4f} -> adapted={adapted:.4f})"
        )
        return 1
    print(
        "interoceptive_integration self-check OK: quiet body -> low salience "
        f"({late_quiet:.4f}); perturbation -> {spike:.4f}; frozen-baseline red fired; "
        f"habituation to the new normal -> {adapted:.4f}. Tract terminus for "
        f"{' -> '.join(TRACT)}; NOT on a live path yet (US1 declares the port)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
