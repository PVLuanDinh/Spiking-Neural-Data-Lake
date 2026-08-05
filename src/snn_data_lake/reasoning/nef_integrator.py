"""
NEF integrator — spiking working memory via recurrent dynamics (pure Python, no tensors).

The third NEF principle, DYNAMICS: wire a population's decoded output back to its own input and you get a neural
INTEGRATOR / line attractor. With the recurrent transform set to identity, the population's represented value is
a fixed point for every value in range -- a continuum of stable states -- so it HOLDS whatever it currently
represents (working memory) and any input drives it along that line, accumulating: the decoded state integrates
the input. This is how the NEF builds persistent memory and controllers from the same LIF populations that
represent and transform values (Eliasmith & Anderson 2003; spiking attractor networks 2005). It reuses the v0.99
`NEFEnsemble` (represent/transform) and adds the recurrent loop. Pure standard library, deterministic. On-thesis
with nef_compute / ring_attractor / spiking_pid (a spiking hold-and-integrate the control loop can use).

Rate-mode recurrent integrator (identity feedback + input, per-timestep decode) -- the dynamics
principle in discrete rate form. The finite-population decode error shows up as a small, bounded memory DRIFT,
exactly as a real neural integrator leaks; not the continuous-time spiking simulation Nengo runs.

  PYTHONPATH=src python -m snn_data_lake.reasoning.nef_integrator   # (also run by run_selfchecks.py)

(The 2026-08-04 sweep rewrote the dead `python -m research.nef_integrator` here but kept its
trailing "root on PYTHONPATH", which was true of `research.` and is false of the package path --
`src` is what has to be on PYTHONPATH now. Corrected 2026-08-05, verified by running it.)
"""

from snn_data_lake.reasoning.nef_compute import NEFEnsemble


def _say(msg):
    """print(), with the project's ASCII-only output invariant asserted at the point of emission.

    This module printed a U+2014 EM DASH for the life of the feature and the gate never saw it:
    MEASURED 2026-08-04, `PYTHONIOENCODING=ascii python -m snn_data_lake.reasoning.nef_integrator`
    exited 1 with `UnicodeEncodeError: 'ascii' codec can't encode character '\\u2014'`. Under the
    default Windows console cp1252 the em dash happens to encode, which is exactly why the defect
    survived -- the invariant is ASCII, not "whatever this console tolerates". Reachable red: put a
    non-ASCII character in any _say() below and this raises here, naming the line. Same shape as
    nightly_skills.main::say and rose_language.main::say, which is where the pattern comes from."""
    msg.encode("ascii")
    print(msg)


class NEFIntegrator:
    """A recurrent NEF population: holds its state (working memory) and integrates input over time."""

    def __init__(self, n=100, seed=0, dt=0.1):
        self.ens = NEFEnsemble(n=n, seed=seed)
        self.decoders = self.ens.solve_decoders(
            lambda x: x
        )  # identity recurrent transform (line attractor)
        self.dt = dt
        self.state = 0.0

    def reset(self, value=0.0):
        self.state = value

    def step(self, u=0.0):
        held = self.ens.decode(
            self.decoders, max(-1.0, min(1.0, self.state))
        )  # recurrent feedback (holds)
        self.state = held + self.dt * u  # + input (integrates)
        return self.state

    def run(self, inputs):
        return [self.step(u) for u in inputs]


def main():
    dt = 0.1
    integ = NEFIntegrator(n=100, seed=0, dt=dt)

    # 1. integration: constant input -> decoded state ramps by ~dt*u each step
    integ.reset(0.0)
    ramp = integ.run([0.5] * 8)
    expected = 8 * dt * 0.5  # 0.40
    monotone = all(ramp[i + 1] > ramp[i] for i in range(len(ramp) - 1))

    # 2. working memory: drive to a value, then zero input -> state HOLDS (bounded drift)
    integ.reset(0.0)
    integ.run([0.5] * 8)
    held0 = integ.state
    integ.run([0.0] * 30)
    drift = integ.state - held0

    # 3. reversibility: +u then -u returns near the start (the line attractor integrates signed input)
    integ.reset(0.0)
    integ.run([0.5] * 8)
    integ.run([-0.5] * 8)
    back = integ.state

    _say("=" * 60)
    _say("NEF INTEGRATOR -- spiking working memory via recurrent dynamics")
    _say("=" * 60)
    _say(
        f"integration : 8 steps of u=0.5 -> state {ramp[-1]:.3f} (expected ~{expected:.2f}), monotone {monotone}"
    )
    _say(f"working mem : held {held0:.3f}, drift over 30 zero-input steps {drift:+.3f}")
    _say(f"reversible  : +8 then -8 steps -> {back:+.3f} (~0)")
    _say("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert monotone and abs(ramp[-1] - expected) < 0.1, (
        f"state must integrate the input ({ramp[-1]:.3f} vs {expected:.2f})"
    )
    assert abs(drift) < 0.12, (
        f"working memory must HOLD the value with only bounded drift ({drift:+.3f})"
    )
    assert abs(back) < 0.1, f"integrating +u then -u must return near the start ({back:+.3f})"
    integ.reset(0.0)
    again = integ.run([0.5] * 8)
    assert again == ramp, "same input -> identical trajectory (deterministic)"
    _say(
        f"self-check OK: integrates input (state {ramp[-1]:.2f}~{expected:.2f}), holds it as working memory "
        f"(drift {drift:+.2f} over 30 steps), reversible ({back:+.2f}); deterministic"
    )


if __name__ == "__main__":
    main()
