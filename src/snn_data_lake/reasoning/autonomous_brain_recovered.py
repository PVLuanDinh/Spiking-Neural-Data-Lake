r"""
AUTONOMOUS BRAIN (RECOVERED) -- the enclosed self-directed loop, with the accuracy recovered from
47.4% to ~95% online (and ~98% via a sleep-consolidation RBF refit), WITHOUT dropping any faculty.

WHY THE ORIGINAL (research/autonomous_brain.py) LANDS AT 47.4%
-------------------------------------------------------------
Its UNDERSTANDING readout is a backprop MLP (brain_adapters.growable_readout.GrowableReadout) grown
and frozen per new digit. In CLASS-INCREMENTAL testing (argmax over all 10 classes, no task id) the
shared softmax head forgets earlier digits via recency bias -- growable_readout.py's own self-check
(section 4) shows hidden-only freeze recovers only a few points and the class-IL "afterALL" number
stays low for every freeze mode. So the 47.4% is the READOUT, not a broken faculty: the exact same
frozen unsupervised conv1 features score 94.9-95.4% under an ORDER-INVARIANT readout
(developmental_loop_lake_slda.py) and 98.2% under a static RBF ensemble.

THE RECOVERY -- every faculty stays genuinely active
----------------------------------------------------
  PERCEPTION      conv1 latency-STDP, unsupervised + frozen; batched byte-identical features.
  WORKING MEMORY  the objective is held as a pointer in the prefrontal persistent-activity attractor.
  SELF-DIRECTION  curiosity picks unseen digits; a ROUND-ROBIN revisit scheduler then diversifies.
  UNDERSTANDING   SLDA_RFF -- streaming LDA over Random Fourier Features. Order-invariant sufficient
                  statistics => continual == joint => NO forgetting (the readout swap, ~95% online).
  NEUROGENESIS    GENUINE + order-invariant: at each new concept GROW the Random-Fourier basis, then
                  SLEEP-REPLAY every accumulated observation through the widened basis. Never forgets.
  LIFELONG MEMORY every new digit's held-out exemplars are CONSOLIDATED to the durable substrate lake
                  and pulled back BY PROXIMITY as replay (labelled by their OWN true label).
  REFLECTION      each new concept fires a train of thought over the brain's OWN concepts (spiking
                  workspace) and WIRES what it thought about together (concept-association graph).
  SLEEP -> 98%    a sleep consolidation refits an RBF-SVM on ALL lake-consolidated exemplars.

THIS FILE IS NOW A THIN WRAPPER: the loop body lives in research/developmental_core.py
(`develop_cycles`) so the standalone benchmark and the unified always-on Brain
(research/brain_loop.py) share ONE source of truth. This wrapper runs the loop with its own temp lake
and re-runs the SAME regression asserts (online > 90%, sleep-refit > 90%, durable, faculties active).

Feature learning stays UNSUPERVISED (frozen conv1); only the readout is supervised. Zero-dep CORE is
untouched: this file lives under research/ (ruff-excluded) and edits neither run_selfchecks nor pyproject.

Run: NORD_GPU=1 PYTHONPATH=src .venv-cu128/Scripts/python research/autonomous_brain_recovered.py
Knobs: NORD_CYCLES(18) NORD_C1(30) NORD_T(15) NORD_L1_EPOCHS(3) NORD_PER(400) NORD_KREPLAY(150)
       NORD_RFF(2000) NORD_GROW_RFF(200) NORD_GAMMA(0.04) NORD_SHRINK(0.1) NORD_MAINTAIN(20)
       NORD_TRAIN(12000) NORD_TEST(2000) NORD_REFLECT_THR(0.55)
Deps: numpy torch torchvision snntorch scikit-learn
"""
from snn_data_lake.reasoning.torch_device import resolve_device
from snn_data_lake.reasoning.developmental_core import cfg_from_env, develop_cycles


def main():
    cfg = cfg_from_env()
    print("=" * 96)
    print("AUTONOMOUS BRAIN (RECOVERED) -- enclosed loop: perceive -> hold-in-WM -> learn (order-invariant)")
    print("                               -> grow RFF neurons -> consolidate -> reflect -> sleep-refit -> repeat")
    print(f"device={resolve_device()}  cycles={cfg['cycles']}  C1={cfg['c1']}  "
          f"rff0={cfg['rff0']}(+{cfg['grow_rff']}/grow)  gamma={cfg['gamma']}  "
          f"store={cfg['store']}/digit  retrieve-k={cfg['kreplay']}")
    print("=" * 96)

    res = develop_cycles(cfg)                     # own temp lake, no emit -> the standalone behaviour

    online, rbf_acc = res["online"], res["rbf"]
    print("\n" + "=" * 96)
    print("AUTONOMOUS BRAIN (RECOVERED) -- one full enclosed cycle, all faculties active:")
    print("-" * 96)
    print(f"  PERCEPTION   : unsupervised conv1 features (frozen), diversity {res['conv_diversity']:.3f}")
    print(f"  WORKING MEM  : objective held in prefrontal persistent activity  {res['wm_hits']}/{res['cycles']} cycles")
    print(f"  SELF-DIRECT  : curiosity + round-robin revisit + curriculum ({res['curriculum_mastered']}/10 mastered)")
    print(f"  NEUROGENESIS : the brain GREW its RFF basis {res['grow_events']}x -> {res['rff']} projection "
          f"neurons (order-invariant, no forgetting)")
    print(f"  LIFELONG MEM : {res['n_stored']} exemplars in the DURABLE data lake; survives restart = {res['durable']}")
    print(f"  REFLECTION   : thought a train of thought about each new concept (via the spiking workspace) "
          f"+ WIRED what it thought together -> {len(res['links'])} concept links; strongest {res['strongest']}")
    print(f"  UNDERSTANDING (online, SLDA-RFF)     : {online:.1%}")
    print(f"  UNDERSTANDING (sleep-refit, RBF)     : {rbf_acc:.1%}   ({res['n_used']} consolidated exemplars)")
    print("-" * 96)
    print("  reference: original grow-and-freeze MLP loop = 47.4%   static RBF ceiling = 98.2%")
    print(f"  RECOVERY: online {online:.1%} (+{(online - 0.474) * 100:.1f} pts) ; "
          f"sleep-refit {rbf_acc:.1%} (+{(rbf_acc - 0.474) * 100:.1f} pts)")
    print("  perceive -> hold-in-WM -> learn (order-invariant) -> grow RFF neurons -> consolidate -> "
          "reflect -> sleep-refit -> REPEAT")
    print("=" * 96)

    # ---- the SAME regression asserts as before the extraction ----
    # PRIMARY recovery claim: the online order-invariant readout clears 90% (was 47.4%).
    assert online > 0.90, f"online SLDA-RFF understanding regressed below 90%: {online:.3f}"
    # The sleep-refit RBF trains ONLY on the consolidated lake exemplars; honest floor, not a faked 98.
    assert rbf_acc > 0.90, f"sleep-refit RBF regressed below 90%: {rbf_acc:.3f}"
    assert res["durable"], "lifelong memory did not survive restart"
    assert res["wm_hits"] > 0 and res["n_stored"] > 0 and res["grow_events"] > 0, "a faculty was inactive"
    assert len(res["learned"]) == 10, f"the loop did not learn all 10 concepts: {res['learned']}"
    print(f"self-check OK: enclosed loop RECOVERED -- online {online:.1%} + sleep-refit {rbf_acc:.1%} "
          f"(was 47.4%); perception+WM+self-direction+neurogenesis({res['grow_events']} RFF grows)+"
          f"lifelong-lake(durable)+REFLECTION(train of thought about each concept, {len(res['links'])} concept "
          f"links wired) all active over {res['cycles']} cycles.")
    return 0


if __name__ == "__main__":
    main()
