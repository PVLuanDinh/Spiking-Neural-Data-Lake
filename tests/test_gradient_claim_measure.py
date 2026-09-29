"""The on-demand half: can the criterion DISCRIMINATE for a rule before the bar binds on it?

Written before the implementation (Phase 4 is TDD by requirement, not by preference). Three
properties, each pinned by a constructed case:

1. FR-004/SC-003 -- a rule the instrument cannot judge is left UNJUDGED, not condemned.
2. FR-005/SC-002 -- "no cosine available here" and "cosine measured, below the bar" are different
   outcomes, and collapsing them is a failure this project has already had.
3. FR-013b/SC-023 -- the gate imports no rule-carrying module in ANY mode, because 89 src modules
   carry a written stamp saying exactly that.

Each test carries its own control IN THE SAME TEST. A negative from an instrument never shown to
produce a positive is not evidence, and this file exists to test an instrument.
"""

import importlib
import json
import re
import subprocess
import sys

gate = importlib.import_module("snn_data_lake.gates.gradient_claim_gate")

BAR = 0.5


def _sub(code):
    """Run a probe in a CLEAN interpreter. In-process introspection cannot answer any question in
    this file: `tests/conftest.py` records that the suite deliberately imports torch, so an
    in-process `"torch" not in sys.modules` is red for a reason unrelated to the gate."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)


def test_the_frozen_feedback_alignment_arm_misses_the_bar_on_every_draw():
    """SC-003, driven through the SHIPPED module rather than a reconstruction.

    RENAMED 2026-08-18: the old name asserted the arm 'falsifies the instrument, not the rule'
    -- the framing POLICIES retracted (RULE_JUDGED is a coherence-keyed token; this arm's product
    is ~0.0001). The body always asserted only the PROPERTY, and is unchanged.

    `basal_ganglia/mlp_q.py` already computes this sweep and prints it. Reconstructing the
    configuration here would build a second copy of a mechanism that exists (P2), and the
    reconstruction that was drafted for this task had a positive control of the form
    `_cos(f(W), f(W))` -- an algebraic identity that reads 1.0 with the hidden layer dead, so it
    could not have failed for any reason.

    THE INTERVAL IS DELIBERATELY NOT PINNED. `mlp_q.py`, `docs/POLICIES.md` and this task all
    recorded `[-0.472, +0.279]`, and the module prints `[-0.4366, +0.1844]` today -- the numbers
    drifted and three sites still carry the old ones. What is stable, and what SC-003 actually turns
    on, is the PROPERTY: feedback alignment misses the bar on every draw while the shipped path
    clears it.
    """
    p = _sub(
        "import runpy; runpy.run_module('snn_data_lake.basal_ganglia.mlp_q', run_name='__main__')"
    )
    assert p.returncode == 0, f"the shipped self-check did not run:\n{p.stdout}\n{p.stderr}"
    line = next((ln for ln in p.stdout.splitlines() if ln.strip().startswith("(5)")), None)
    assert line, (
        f"the module printed no cosine line; this test would assert over nothing:\n{p.stdout}"
    )

    # NON-EMPTINESS FIRST. `0 of 8` is the claim; `0 of 0` would satisfy any count-based assertion
    # while measuring nothing, which is the exact escapee this repo keeps re-finding.
    m = re.search(r"\|cos\| > 0\.5 in (\d+) of (\d+)", line)
    assert m, f"could not find the sweep count in: {line}"
    above, draws = int(m.group(1)), int(m.group(2))
    assert draws >= 8, f"the sweep collapsed to {draws} draws -- the count below proves nothing"

    # (1) THE PROPERTY: frozen feedback alignment clears the bar on NO draw.
    assert above == 0, f"feedback alignment cleared the bar on {above}/{draws} draws: {line}"

    # (2) POSITIVE CONTROL, and it is a DIFFERENT configuration rather than the same one twice: the
    #     shipped Kolen-Pollack path runs the same instrument and DOES clear the bar. Without this,
    #     an instrument that returned "below bar" unconditionally would satisfy (1).
    shipped = re.search(r"SHIPPED \(Kolen-Pollack, learned B\) ([+-]?\d+\.\d+)", line)
    assert shipped, f"no shipped-path cosine in: {line}"
    assert float(shipped.group(1)) > BAR, (
        f"the control did not clear the bar ({shipped.group(1)}) -- the instrument is not "
        f"discriminating, so the 0-of-{draws} above is uninformative"
    )

    # (3) And the transport identity reads 1.000, proving the reference gradient is the thing the
    #     cosine is taken against rather than a rescaled copy of the candidate.
    assert "transport identity 1.000" in line, line


def test_unavailable_is_not_a_failure():
    """FR-005 / SC-002. UNAVAILABLE and "measured, below the bar" must be distinguishable IN OUTPUT.

    The obvious form of this test is unconditionally true: building a below-bar result and asserting
    `"UNAVAILABLE" not in rendered` holds no matter what the renderer does, because nothing in that
    construction could have produced the word. Both directions are asserted instead.
    """
    below = gate.DiscriminationResult(
        qualname="probe::below",
        aligned=0.21,
        null_sample=[0.0, 0.05, 0.1],
        coherence=(0.1, 0.1),
        verdict=gate.RULE_JUDGED,
        reason="",
    )
    absent = gate.DiscriminationResult(
        qualname="probe::absent",
        aligned=None,
        null_sample=[],
        coherence=(None, None),
        verdict=gate.UNAVAILABLE,
        reason="torch is not installed in this environment",
    )
    r_below, r_absent = gate.render_discrimination(below), gate.render_discrimination(absent)

    assert gate.UNAVAILABLE in r_absent, r_absent
    assert gate.UNAVAILABLE not in r_below, r_below
    assert gate.RULE_JUDGED in r_below, r_below
    assert gate.RULE_JUDGED not in r_absent, r_absent
    # The measured-but-low case must still SHOW its number; an unavailable one has none to show.
    assert "0.21" in r_below, r_below
    assert absent.reason in r_absent, "an unavailable verdict that does not say WHY is a shrug"


def test_instrument_falsified_is_reachable_and_rule_judged_is_too():
    """FR-004 / FR-009d / SC-022, in ONE test so neither can be green in isolation.

    A verdict nothing ever produces is a verdict nobody has seen fire. The coherence guard is what
    gives INSTRUMENT_FALSIFIED a real instance: a within-matrix permutation preserves the
    per-family mean, so on a sign-coherent pair the null sits ABOVE the bar and the separation
    statistic INVERTS -- measured null mean +0.980, P(null>0.5) = 1.0000, aligned 1.75 sd BELOW the
    null. Reporting a separation there would bless a shuffled rule and condemn an aligned one.
    """
    # A near-zero-mean pair: the permutation IS a null, and the rule is judged.
    ceiling = gate.coherence_ceiling()
    assert gate.coherence([1.0, -1.0, 0.9, -1.1, 1.05, -0.95]) < ceiling
    # A sign-coherent pair: the permutation is NOT a null, and the instrument must refuse.
    assert gate.coherence([1.0, 1.02, 0.98, 1.01, 0.99, 1.0]) > ceiling

    # The ceiling is DERIVED from the frozen bar, not chosen: the permutation null's mean is
    # the product of the two coherences, so it reaches the bar at sqrt(bar) symmetric.
    assert abs(ceiling**2 - gate.read_bar()) < 1e-12, (
        f"the ceiling {ceiling} is not sqrt(bar={gate.read_bar()}) -- it has become a constant"
    )

    judged = gate.verdict_for(aligned=0.94, coherence=(0.12, 0.20))
    falsified = gate.verdict_for(aligned=0.94, coherence=(0.99, 0.99))
    assert judged == gate.RULE_JUDGED, judged
    assert falsified == gate.INSTRUMENT_FALSIFIED, falsified
    # The aligned cosine is IDENTICAL in both calls, so only coherence can have moved the verdict.


def test_the_gate_imports_no_rule_module_in_any_mode():
    """FR-013b / SC-023. The boundary 89 src modules already assert, turned into an assertion.

    Each of those modules carries `Read statically by gates/gradient_claim_gate.py; never
    imported by it.` That sentence is a claim about this gate, and until now nothing checked it.
    Both CI modes are covered: CI runs the gate TWICE, and a test exercising only `--assert` would
    leave `--self-check` free to import anything.

    Reachable red: hoist `import torch` to module scope in the gate and the first assertion goes red
    while the control below stays green.
    """
    probe = (
        "import sys;"
        "from snn_data_lake.gates.gradient_claim_gate import main;"
        "main([{mode}]);"
        "print('TORCH', 'torch' in sys.modules)"
    )
    for mode in ("'--assert'", "'--self-check'"):
        p = _sub(probe.format(mode=mode))
        assert "TORCH False" in p.stdout, f"mode {mode} loaded torch:\n{p.stdout}\n{p.stderr}"

    # POSITIVE CONTROL, same probe: it must be able to report True, or "False" means nothing.
    ctl = _sub("import torch;" + probe.format(mode="'--assert'"))
    assert "TORCH True" in ctl.stdout, (
        "the probe cannot detect a loaded torch, so every False above is uninformative:\n"
        f"{ctl.stdout}\n{ctl.stderr}"
    )


def test_exit_two_is_reachable_and_is_not_returned_unconditionally(monkeypatch):
    """FR-005 / T044. An exit code nothing can produce carries no information.

    Exit 2 means NOT-ELIGIBLE: nothing could be measured here. The construction is a fixture whose
    subprocess cannot import -- which under FR-013b is how `UNAVAILABLE` arises naturally rather
    than being special-cased. The control runs the SAME code path with a working fixture, so a
    `_measure()` that returned 2 unconditionally cannot pass.
    """
    saved = gate._FIXTURES
    # `_measure()` WRITES the tracked FR-015 artifact as a side effect, so a test driving it with
    # planted fixtures silently republishes `tools/gradient_claim_measure.json` full of probe names.
    # Not hypothetical -- it is how `"qualname": "probe::ok"` reached a commit. The writer is
    # neutered here because this test is about EXIT CODES; the artifact has its own test below.
    written = []
    monkeypatch.setattr(gate, "_write_measure_artifact", lambda *a, **k: written.append(a) or None)
    try:
        gate._FIXTURES = {"probe::cannot_import": ("import zz_no_such_module_zz", "n/a")}
        assert gate._measure() == 2, "an unimportable fixture must be NOT-ELIGIBLE"
        gate._FIXTURES = {"probe::ok": (gate._EPROP_PROBE, "ref")}
        assert gate._measure() == 0, "a working fixture must exit 0 through the same path"
    finally:
        gate._FIXTURES = saved
    assert written, "the writer was never called -- the neuter above is guarding nothing"


def test_the_measurement_is_reproducible_on_an_unchanged_tree():
    """SC-024 / FR-009e. Pinned draws and seed, or the artifact's staleness signal is just noise."""
    a = gate.measure_all()[0][1]
    b = gate.measure_all()[0][1]
    assert a.bar_sd_above_null == b.bar_sd_above_null, "two runs disagreed on an unchanged tree"
    # POSITIVE CONTROL: a different seed MUST differ, or "identical" is a property of the code
    # returning a constant rather than of the seed being pinned.
    c = gate.measure_all(seed=gate.MEASURE_SEED + 1)[0][1]
    assert c.bar_sd_above_null != a.bar_sd_above_null, (
        "changing the seed changed nothing -- the null is not actually being resampled"
    )


def test_staleness_is_visible_rather_than_indistinguishable_from_passing():
    """FR-015. The artifact must be able to SAY it describes a tree that no longer exists.

    A timestamp cannot do this: a file dated yesterday against a tree edited this morning still
    reads as fresh. The digest is over the DECLARATIONS, so it moves exactly when the thing the
    measurement was taken against moves.
    """
    rules, _, _ = gate.census()
    base = gate.tree_digest(rules)
    assert gate.tree_digest(rules) == base, "the digest is not stable across calls"

    mutated = [dict(r) for r in rules]
    mutated[0]["declared"] = "ZZ_MUTATED"
    assert gate.tree_digest(mutated) != base, (
        "the digest did not move when a declaration changed -- it cannot detect staleness at all"
    )


# The RETIRED unfiltered factorization, reimplemented here as the CONTRAST and deliberately kept out
# of `src/` -- P2 forbids two mechanisms shipping, and a test fixture is not a shipped mechanism.
# Without it, "the reconciliation changed the criterion" is a claim with nothing to compare to.
def _unfiltered_hidden_grads(net, x, y):
    import torch

    xf = net.flatten_frames(x.float())
    b, t = xf.shape[0], xf.shape[1]
    dev = xf.device
    mask = net.offdiag_mask(dev)
    w_rec = net.W_rec * mask
    v = torch.zeros(b, net.hidden, device=dev)
    z = torch.zeros(b, net.hidden, device=dev)
    yint = torch.zeros(b, net.num_classes, device=dev)
    eps_in = torch.zeros(b, net.hidden, net.n_in, device=dev)
    eps_rec = torch.zeros(b, net.hidden, net.hidden, device=dev)
    g_in, g_rec = torch.zeros_like(net.W_in), torch.zeros_like(net.W_rec)
    with torch.no_grad():
        for step in range(t):
            xt, z_prev = xf[:, step], z
            cur = xt @ net.W_in.t() + z_prev @ w_rec.t()
            v = net.alpha * v - net.v_th * z + cur
            p = net.psi(v)
            z = (v >= net.v_th).float()
            yint = net.kappa * yint + z @ net.W_out.t()
            err = torch.softmax(yint + net.b_out, dim=1)
            err[torch.arange(b), y.long()] -= 1.0
            eps_in = net.alpha * eps_in + xt.unsqueeze(1)
            eps_rec = net.alpha * eps_rec + z_prev.unsqueeze(1)
            lsig = err @ net.W_out
            g_in += ((lsig * p).unsqueeze(2) * eps_in).sum(0)
            g_rec += ((lsig * p).unsqueeze(2) * eps_rec).sum(0)
    return {"W_in": g_in, "W_rec": g_rec * mask}


def _hidden_cos(ge, gb, net, K):
    import torch

    hid = net.offdiag_mask(ge["W_rec"].device).bool()
    a = torch.cat([ge["W_in"].reshape(-1), ge["W_rec"][hid].reshape(-1)])
    b = torch.cat([gb["W_in"].reshape(-1), gb["W_rec"][hid].reshape(-1)])
    return float(K._cosine(a, b))


def _net_at(K, seed, g_rec, g_in, T):
    import torch

    net = K._TinyNet(n_in=12, hidden=8, num_classes=3, seed=seed)
    net.alpha = 0.99
    with torch.no_grad():
        net.W_rec *= g_rec
        net.W_in *= g_in
    x = (torch.rand(6, T, 12, generator=torch.Generator().manual_seed(11)) < 0.3).float()
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    return net, x, y


def test_the_two_factorizations_separate_verdicts_in_the_truncation_regime():
    """FR-012 / T047. Inspection cannot substitute for measurement -- proven by a MOVED VERDICT.

    **T047's stated configuration does not reproduce and is NOT pinned here.** It claimed
    `g_rec=50 -> 0.7376 GREEN vs 0.4105 RED`; that pair appears only in `tasks.md`, `research.md`
    and `plan.md`, in no code and no evidence file, with no fixture recorded. Measured at that
    configuration the pair is -0.1258 / +0.1928 -- BOTH RED, no separation. Pinning it would have
    repeated the T039 defect of testing against a number nobody can regenerate.

    A 144-point sweep (3 seeds x 8 recurrent gains x 3 input gains x 2 lengths) found
    **12** genuinely separating configurations. This test uses the largest, and states the fixture
    in full so it can be re-derived rather than inherited.

    DISCLOSED, because it complicates the story rather than supporting it: **2 of the 12 separate
    the OTHER way** (seed 3 at g_rec x5 and x50, g_in 1.0, T=96: reconciled RED, unfiltered GREEN).
    The reconciled form is the one the shipped encoders run and the one that makes the readout
    gradient exact; it is NOT uniformly higher on this metric at every configuration, and claiming
    otherwise would be a claim the sweep refutes.
    """
    K = importlib.import_module("snn_data_lake.plasticity.eprop_kernel")
    bar = K.EPROP_BPTT_COSINE_FLOOR

    net, x, y = _net_at(K, seed=11, g_rec=50, g_in=1.0, T=24)
    gb = K.bptt_grads(net, x, y)
    shipped = _hidden_cos(net.eprop_grads(x, y)[0], gb, net, K)
    retired = _hidden_cos(_unfiltered_hidden_grads(net, x, y), gb, net, K)

    assert shipped > bar, f"the shipped factorization must clear the bar here: {shipped}"
    assert retired <= bar, f"the retired factorization must miss it here: {retired}"
    assert (shipped > bar) != (retired > bar), (
        f"the two forms did not separate verdicts ({shipped:+.4f} vs {retired:+.4f}) -- without a "
        "separation this test cannot show that the factorization choice matters"
    )

    # CONTROL: at initialisation the two agree and BOTH clear the bar, so a test placed there would
    # be vacuous. This is the half that proves the separation above is a property of the REGIME.
    net0, x0, y0 = _net_at(K, seed=11, g_rec=1, g_in=1.0, T=24)
    gb0 = K.bptt_grads(net0, x0, y0)
    s0 = _hidden_cos(net0.eprop_grads(x0, y0)[0], gb0, net0, K)
    r0 = _hidden_cos(_unfiltered_hidden_grads(net0, x0, y0), gb0, net0, K)
    assert s0 > bar and r0 > bar, f"the benign regime should not separate: {s0:+.4f} vs {r0:+.4f}"


def test_the_reconciled_readout_gradient_is_exact_rather_than_merely_close():
    """T052. The claim `eprop_kernel` used to make in prose, now measured on both sides.

    Before the reconciliation the readout was computed from the RAW spike train while the loss sees
    the KAPPA-FILTERED one, so `err (x) z` was not the gradient of anything the loss computes:
    measured W_out cosine 0.986830, relative L2 7.4233e-01, `allclose` False -- while a comment
    asserted the two were "IDENTICAL by construction". Only `b_out` was ever identical.
    """
    import torch

    K = importlib.import_module("snn_data_lake.plasticity.eprop_kernel")
    net = K._TinyNet(seed=7)
    x = (torch.rand(6, 9, 12, generator=torch.Generator().manual_seed(11)) < 0.3).float()
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    ge, _ = net.eprop_grads(x, y)
    gb = K.bptt_grads(net, x, y)

    assert torch.allclose(ge["W_out"], gb["W_out"], atol=1e-6), (
        "the readout gradient is NOT exact -- the kappa filter is missing from `zbar`"
    )
    assert torch.allclose(ge["b_out"], gb["b_out"], atol=1e-6)
    # POSITIVE CONTROL: the HIDDEN gradient is an approximation and must NOT be exact, or this test
    # is passing because the two gradient paths have collapsed into one computation.
    assert not torch.allclose(ge["W_in"], gb["W_in"], atol=1e-6), (
        "the hidden gradient is bit-identical to BPTT -- e-prop is no longer an approximation, "
        "which means the 'independent' reference is not independent"
    )


def test_changing_one_measured_input_flips_the_record_to_stale(tmp_path, monkeypatch):
    """SC-017 / FR-015a / T059. The staleness signal owes its own reachable red.

    A digest that is never recomputed reads IDENTICAL to one that always matches: both say CURRENT,
    forever. So it is not enough that the artifact carries a digest -- the verdict must be shown to
    MOVE when a measured input moves, and the three states must be structurally distinct.
    """
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    monkeypatch.setattr(gate, "_repo_root", lambda: str(root))

    # (1) ABSENT must not read as a pass. Nobody has run it; that is not "it passed".
    verdict, why = gate.measure_freshness()
    assert verdict == "ABSENT", (verdict, why)

    # (2) Written against the live tree -> CURRENT.
    art = root / "tools" / "gradient_claim_measure.json"
    art.write_text(json.dumps({"tree_digest": gate.tree_digest(), "results": []}), encoding="utf-8")
    verdict, why = gate.measure_freshness()
    assert verdict == "CURRENT", (verdict, why)

    # (3) THE REACHABLE RED: change what was measured, and the SAME file must now read STALE.
    #     Nothing about the file changes -- only the tree it describes.
    art.write_text(json.dumps({"tree_digest": "crc32:deadbeef", "results": []}), encoding="utf-8")
    verdict, why = gate.measure_freshness()
    assert verdict == "STALE", (verdict, why)
    both = "a STALE verdict must name BOTH digests, or the reader cannot tell what moved"
    assert "deadbeef" in why and gate.tree_digest() in why, both

    # (4) An unreadable artifact is ABSENT, never CURRENT -- a parse failure is not a pass.
    art.write_text("{not json", encoding="utf-8")
    assert gate.measure_freshness()[0] == "ABSENT"


def test_measure_all_actually_constructs_the_instrument_falsified_result(monkeypatch):
    """The VERDICT is reachable in the predicate; this drives the path that BUILDS the result.

    Found by reading coverage, not code. `verdict_for()` was tested in isolation while the branch in
    `measure_all` that turns that verdict into a `DiscriminationResult` was never executed -- the
    same shape as a refusal the writer never consults, which this module already had once and fixed.
    A guard whose predicate is right and whose construction is untested is a guard with an
    unexercised failure path.
    """
    coherent = (
        "import json\n"
        "n = 40\n"
        "cand = [1.0 + 0.01 * ((i * 7) % 5) for i in range(n)]\n"
        "ref = [1.0 + 0.01 * ((i * 3) % 5) for i in range(n)]\n"
        'print("GRADJSON" + json.dumps({"cand": [cand], "ref": [ref], "row_widths": [n]}))\n'
    )
    monkeypatch.setattr(gate, "_FIXTURES", {"probe::coherent": (coherent, "a coherent reference")})
    ((result, report),) = gate.measure_all()

    assert result.verdict == gate.INSTRUMENT_FALSIFIED, result.verdict
    assert report is None, "an unjudged rule must carry no separation report"
    assert result.null_sample == [], (
        "a falsified instrument must not publish a null it cannot trust"
    )
    assert "coherence product" in result.reason, result.reason
    assert result.coherence[0] > 0.9 and result.coherence[1] > 0.9, result.coherence
    # It must RENDER too -- an unreachable render path is how a verdict becomes invisible.
    assert gate.INSTRUMENT_FALSIFIED in gate.render_discrimination(result, report)


def test_update_baseline_refuses_to_lower_the_liveness_floor_silently(
    tmp_path, monkeypatch, capsys
):
    """The floor's REFUSED branch in the WRITER, which `--self-check` does not reach.

    `--self-check` proves `evaluate()` fires on a floor breach. This is the other half: the writer
    must refuse to RECORD a lowered floor without a stated reason, because a narrowed walk and a
    broken one look identical from the counts alone -- both report fewer rules and zero problems.
    """
    target = tmp_path / "gradient_claim_baseline.json"
    monkeypatch.setattr(gate, "_baseline_path", lambda *a, **k: str(target))
    # A floor ABOVE what the live tree can report: the walk has "narrowed" from the file's view.
    monkeypatch.setattr(
        gate,
        "load_baseline",
        lambda *a, **k: {
            "counts": {gate.MISSING_DECLARATION: 0, gate.UNDETERMINABLE: 0},
            "rules_seen_floor": 10_000,
        },
    )
    assert gate.main(["--update-baseline"]) == 1, "a silent floor drop must be REFUSED"
    out = capsys.readouterr().out
    assert "REFUSED" in out and "BELOW the recorded floor" in out, out
    assert not target.exists(), "the refused write still created the baseline file"

    # CONTROL: the SAME command with a stated reason is allowed -- the refusal is about silence,
    # not about the floor moving. Without this, a writer that refused everything would pass above.
    assert gate.main(["--update-baseline", "--bootstrap", "walk deliberately narrowed"]) == 0
    assert target.exists(), "the bootstrapped write did not happen"


def test_the_recorded_artifact_describes_the_REAL_fixtures_not_a_test_probe():
    """The artifact is TRACKED, and `_measure()` writes it. A test that drives `_measure()` with
    planted fixtures therefore republishes it -- which happened: `probe::ok` reached a commit.

    This is the guard. Reachable red: run the exit-2 test without neutering the writer and the
    artifact fills with `probe::` names, which this refuses.
    """
    import json as _json
    import os as _os

    root = gate._repo_root()
    if root is None:
        return
    path = _os.path.join(root, gate.MEASURE_ARTIFACT)
    if not _os.path.exists(path):
        return
    payload = _json.load(open(path, encoding="utf-8-sig"))
    names = [r["qualname"] for r in payload["results"]]
    assert names, "the artifact records no results at all"
    unknown = [n for n in names if n not in gate._FIXTURES]
    assert not unknown, (
        f"the recorded artifact names symbols that are not registered fixtures: {unknown}. "
        "A test almost certainly wrote it -- regenerate with `--measure`."
    )
