"""
LT-Gate neuron — dual-time-constant LIF with a long-term memory trace (pure Python, no tensors).

A standard LIF forgets the moment input stops: its membrane leaks out within a few time constants. The LT-Gate
neuron carries TWO membranes: a FAST branch (short tau, resets on every spike -> crisp, responsive) and a SLOW
branch (long tau, a NON-RESET decaying memory trace that keeps accumulating and is never cleared by a spike).
The output mixes the two by a gate; because the slow branch survives the reset, activity PERSISTS long after
the stimulus ends -- anti-forgetting / long-term retention -- where a fast-only neuron goes silent immediately.
Pure standard library, deterministic. The zero-dep counterpart to the torch LT-Gate in research/neuromorphic_torch.

  python -m snn_data_lake.lt_gate        # (also run by run_selfchecks.py)
"""
import math


class LTGateNeuron:
    """Fast (resetting) + slow (non-reset memory trace) membranes, gated; step(current) -> 1/0 spike."""

    def __init__(self, tau_fast=3.0, tau_slow=40.0, v_threshold=1.0, gate=0.5):
        self.decay_fast = math.exp(-1.0 / tau_fast)
        self.decay_slow = math.exp(-1.0 / tau_slow)
        self.v_threshold = v_threshold
        self.gate = gate                        # 0 -> fast-only (plain LIF); 1 -> pure long-term memory
        self.reset_state()

    def reset_state(self):
        self.v_fast = 0.0
        self.v_slow = 0.0

    def step(self, current):
        self.v_slow = self.v_slow * self.decay_slow + current    # non-reset memory trace (survives spikes)
        self.v_fast = self.v_fast * self.decay_fast + current    # fast leaky branch (reset on spike)
        v = (1.0 - self.gate) * self.v_fast + self.gate * self.v_slow
        if v >= self.v_threshold:
            self.v_fast = 0.0                                    # reset ONLY the fast branch; memory persists
            return 1
        return 0

    def run(self, currents):
        self.reset_state()
        return [self.step(c) for c in currents]


def _retention(spikes, burst_end):
    """Steps from the end of the input burst to the last spike (how long activity persists in silence)."""
    tail = [i for i, s in enumerate(spikes) if s and i >= burst_end]
    return (tail[-1] - burst_end) if tail else 0


def main():
    burst, silence = 10, 60
    drive = [1.0] * burst + [0.0] * silence          # a stimulus burst, then a long silent tail

    lt = LTGateNeuron(tau_fast=3.0, tau_slow=40.0, gate=0.5).run(drive)
    fast = LTGateNeuron(tau_fast=3.0, tau_slow=40.0, gate=0.0).run(drive)   # gate 0 == plain fast LIF control

    ret_lt = _retention(lt, burst)
    ret_fast = _retention(fast, burst)

    print("=" * 60)
    print("LT-GATE — dual-time-constant neuron with long-term memory trace")
    print("=" * 60)
    print(f"drive: {burst}-step burst then {silence}-step silence")
    print(f"LT-Gate  (fast+slow) : {sum(lt)} spikes, retention {ret_lt} steps past burst")
    print(f"fast-only control    : {sum(fast)} spikes, retention {ret_fast} steps past burst")
    print(f"anti-forgetting gain : {ret_lt}x vs fast-only {ret_fast}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert ret_fast <= 3, f"fast-only neuron must forget almost immediately after the burst ({ret_fast})"
    assert ret_lt >= 20, f"LT-Gate must retain activity long into the silence ({ret_lt})"
    assert ret_lt > 5 * max(1, ret_fast), f"LT-Gate retention must dwarf fast-only ({ret_lt} vs {ret_fast})"
    assert LTGateNeuron().run(drive) == LTGateNeuron().run(drive), "same input -> identical spike train"
    print(f"self-check OK: LT-Gate retains {ret_lt} steps vs fast-only {ret_fast} (anti-forgetting via "
          f"non-reset slow trace); deterministic")


if __name__ == "__main__":
    main()
