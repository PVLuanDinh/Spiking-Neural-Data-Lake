"""Loadable brain bundle -- a MANIFEST that packages a MODULAR multi-skill brain.

HONEST FRAMING (this is the whole point of the module -- read it before changing anything):
  * A brain bundle is NOT one monolithic network trained on everything, and it does NOT learn all
    skills together. It is a MANIFEST (a small JSON dict) that REFERENCES several SEPARATE,
    independently trained skills -- each skill is its own `trained_brain` JSON (a snn_brain/1 file),
    like a distinct brain area carrying its own learned weights. Bundling is wiring, not training.
  * The HARDWIRED instinct/motor primitives (reflex, cortisol, pid, cpg) are NOT trained and carry
    NO weights here. They are built-in wired behaviours, so the manifest records them as plain
    config booleans -- flags that say "this primitive is wired in", never weight matrices.
  * So "train each skill -> bundle -> hand to the arm loop" is an honest pipeline: you train N
    modular skills separately, list them (plus which wired primitives are on) in one manifest, and
    the loop loads that manifest to get a router over the already-trained skills. Nothing is co-trained.

Format (a JSON dict tagged "snn_brain_bundle/1"):
  {"format": "snn_brain_bundle/1",
   "skills":   {"<name>": {"kind": "trained_brain", "path": "<brain.json, relative to manifest>"}},
   "instincts":{"reflex": true, "cortisol": true},   # hardwired primitives -> config flags, NOT weights
   "motor":    {"pid": true, "cpg": false},          # hardwired primitives -> config flags, NOT weights
   "meta":     {...}}

Per-skill spiking inference is DELEGATED to snn_data_lake.cortex.trained_brain -- never reimplemented here.

  python -m snn_data_lake.substrate.brain_bundle        # (also run by run_selfchecks.py)
"""

import json
import os

from snn_data_lake.cortex import trained_brain

BUNDLE_FORMAT = "snn_brain_bundle/1"
DEFAULT_INSTINCTS = {"reflex": True, "cortisol": True}
DEFAULT_MOTOR = {"pid": True, "cpg": False}


def make_bundle(skills, instincts=None, motor=None, meta=None):
    """Build a bundle manifest dict.

    skills: {name: brain_json_path}. Each path is stored as given and RESOLVED relative to the
    manifest's own directory at load_bundle time, so a saved manifest and its skill files can be
    moved together. instincts/motor default to the wired-primitive flags -- config booleans that
    say a hardwired primitive is on, NOT weights (nothing here is trained).
    """
    if not isinstance(skills, dict) or not skills:
        raise ValueError("make_bundle: skills must be a non-empty {name: brain_path} dict")
    skill_entries = {
        str(name): {"kind": "trained_brain", "path": str(path)} for name, path in skills.items()
    }
    return {
        "format": BUNDLE_FORMAT,
        "skills": skill_entries,
        "instincts": dict(DEFAULT_INSTINCTS if instincts is None else instincts),
        "motor": dict(DEFAULT_MOTOR if motor is None else motor),
        "meta": dict(meta or {}),
    }


def save_bundle(bundle, path):
    """Write the manifest dict to a JSON file (stdlib json). Returns path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)
    return path


def load_bundle(path):
    """Load + VALIDATE a bundle at this trust boundary, then load every referenced skill brain.

    Checks: the format tag, that skills is a non-empty dict, and that each skill entry is
    {"kind": "trained_brain", "path": ...}. Each skill path is resolved RELATIVE TO THE MANIFEST's
    directory and loaded with trained_brain.load_brain, so a missing / mis-shaped / wrong-format
    skill file fails loudly HERE rather than at inference time. Returns a BrainBundle holding the
    loaded per-skill brains plus the instinct/motor config flags.
    """
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or str(manifest.get("format", "")) != BUNDLE_FORMAT:
        raise ValueError(f"{path}: not a {BUNDLE_FORMAT} file (format={manifest.get('format')!r})")
    skills = manifest.get("skills")
    if not isinstance(skills, dict) or not skills:
        raise ValueError(f"{path}: 'skills' must be a non-empty dict")
    base = os.path.dirname(os.path.abspath(path))
    loaded = {}
    for name, entry in skills.items():
        if (
            not isinstance(entry, dict)
            or entry.get("kind") != "trained_brain"
            or not entry.get("path")
        ):
            raise ValueError(
                f"{path}: skill {name!r} must be {{'kind':'trained_brain','path':<file>}}"
            )
        brain_path = os.path.join(base, entry["path"])  # resolve relative to the manifest dir
        loaded[str(name)] = trained_brain.load_brain(brain_path)  # validates the skill file itself
    return BrainBundle(
        loaded,
        dict(manifest.get("instincts") or {}),
        dict(manifest.get("motor") or {}),
        dict(manifest.get("meta") or {}),
    )


class BrainBundle:
    """A loaded modular brain: separate per-skill trained brains + wired instinct/motor config flags.

    self.skills maps name -> a loaded trained_brain dict (its own learned weights). self.instincts
    and self.motor are booleans for HARDWIRED primitives (reflex/cortisol/pid/cpg) -- never weights.
    Inference for each skill is delegated to snn_data_lake.cortex.trained_brain, not reimplemented here.
    """

    def __init__(self, skills, instincts=None, motor=None, meta=None):
        self.skills = dict(skills)
        self.instincts = dict(instincts or {})
        self.motor = dict(motor or {})
        self.meta = dict(meta or {})

    def _brain(self, skill):
        if skill not in self.skills:
            raise KeyError(f"no skill {skill!r} in bundle; skills are {self.skill_names()}")
        return self.skills[skill]

    def skill_names(self):
        """Sorted list of the skill names in this bundle."""
        return sorted(self.skills)

    def predict(self, skill, x):
        """Route input x to the named skill's trained brain -> predicted class (argmax)."""
        return trained_brain.predict(self._brain(skill), x)

    def scores(self, skill, x):
        """Route input x to the named skill's trained brain -> raw output scores."""
        return trained_brain.predict_scores(self._brain(skill), x)

    def has_instinct(self, name):
        """True if the named hardwired instinct primitive (e.g. reflex, cortisol) is wired on."""
        return bool(self.instincts.get(name, False))

    def has_motor(self, name):
        """True if the named hardwired motor primitive (e.g. pid, cpg) is wired on."""
        return bool(self.motor.get(name, False))

    def describe(self):
        """Factual one-paragraph summary: the separate skills (+ in->out shape) and which primitives are wired."""
        parts = [
            f"{n}: {self.skills[n]['in']}->{self.skills[n]['out']}" for n in self.skill_names()
        ]
        on_i = [n for n in sorted(self.instincts) if self.instincts[n]]
        on_m = [n for n in sorted(self.motor) if self.motor[n]]
        return (
            f"Brain bundle of {len(self.skills)} SEPARATE modular skills "
            f"({', '.join(parts)}); wired instincts: {', '.join(on_i) or 'none'}; "
            f"wired motor: {', '.join(on_m) or 'none'}. Each skill is an independently trained brain "
            f"(its own brain area) and the instinct/motor entries are hardwired primitives kept as "
            f"config flags, not weights -- this is NOT one network trained on everything."
        )


def main():
    import random
    import shutil
    import tempfile

    from snn_data_lake import snn_classifier as scl

    random.seed(0)  # local, deterministic data generation
    temps = scl.templates()
    C = len(temps)
    train_data = scl.dataset(scl.TRAIN_PER_CLASS, temps)
    test_data = scl.dataset(scl.TEST_PER_CLASS, temps)
    W, _ = scl.train(train_data, C, quiet=True)

    # Two SEPARATE modular skills, each its own trained_brain (never co-trained). "vision" is the
    # delta-rule readout; "shapes" is a derived readout with the class order reversed, so that
    # routing to a specific skill is genuinely observable (the two skills disagree on every input).
    brain_vision = trained_brain.make_brain([W], steps=scl.STEPS, meta={"skill": "vision"})
    W_rev = [row[:] for row in reversed(W)]
    brain_shapes = trained_brain.make_brain([W_rev], steps=scl.STEPS, meta={"skill": "shapes"})

    work = tempfile.mkdtemp(prefix="snn_bundle_")
    try:
        p1 = trained_brain.save_brain(brain_vision, os.path.join(work, "brain_vision.json"))
        p2 = trained_brain.save_brain(brain_shapes, os.path.join(work, "brain_shapes.json"))
        # skills referenced by BASENAME -> exercises "resolve relative to the manifest dir"
        manifest = make_bundle(
            {"vision": os.path.basename(p1), "shapes": os.path.basename(p2)},
            instincts={"reflex": True, "cortisol": True},
            motor={"pid": True, "cpg": False},
            meta={"built_by": "brain_bundle self-check"},
        )
        bpath = save_bundle(manifest, os.path.join(work, "bundle.json"))
        bundle = load_bundle(bpath)

        x = test_data[0][0]
        pred_vision = bundle.predict("vision", x)
        pred_shapes = bundle.predict("shapes", x)

        # ---- table ----
        print("=" * 66)
        print("BRAIN BUNDLE -- modular multi-skill manifest (separate trained skills)")
        print("=" * 66)
        print("skill       in->out   sample-pred")
        for n in bundle.skill_names():
            b = bundle.skills[n]
            print(f"  {n:<9} {b['in']:>3}->{b['out']:<3}   class {bundle.predict(n, x)}")
        wired_i = ", ".join(k for k in sorted(bundle.instincts) if bundle.instincts[k]) or "none"
        wired_m = ", ".join(k for k in sorted(bundle.motor) if bundle.motor[k]) or "none"
        print(f"wired instincts : {wired_i}   (hardwired flags, NOT weights)")
        print(f"wired motor     : {wired_m}   (hardwired flags, NOT weights)")
        print("=" * 66)

        # ---- self-check: deterministic asserts ----
        assert bundle.skill_names() == ["shapes", "vision"], "skill_names must be sorted"
        # routing: the named skill really runs THAT skill's brain
        assert pred_vision == trained_brain.predict(brain_vision, x), "vision routing wrong"
        assert pred_shapes == trained_brain.predict(brain_shapes, x), "shapes routing wrong"
        assert pred_shapes != pred_vision, "routing must select the specific skill's brain"
        assert bundle.scores("vision", x) == trained_brain.predict_scores(brain_vision, x), (
            "scores route wrong"
        )
        # a missing skill raises a clear KeyError
        missing = False
        try:
            bundle.predict("audio", x)
        except KeyError:
            missing = True
        assert missing, "predicting an absent skill must raise"
        # instinct / motor flags read back
        assert bundle.has_instinct("reflex") and bundle.has_instinct("cortisol"), (
            "instinct flags lost"
        )
        assert bundle.has_motor("pid") and not bundle.has_motor("cpg"), "motor flags lost"
        assert not bundle.has_instinct("nope"), "unknown primitive defaults to False"
        # manifest round-trips through save/load exactly, and a reloaded bundle routes identically
        with open(bpath, encoding="utf-8") as f:
            assert json.load(f) == manifest, "bundle manifest must round-trip"
        bundle2 = load_bundle(save_bundle(manifest, os.path.join(work, "bundle2.json")))
        assert bundle2.predict("vision", x) == pred_vision, "reloaded bundle must route identically"
        # describe() names the skills + shapes and states the honest modular framing
        desc = bundle.describe()
        assert "vision" in desc and "shapes" in desc, "describe must name the skills"
        assert f"{brain_vision['in']}->{brain_vision['out']}" in desc, (
            "describe must show in->out shape"
        )
        assert "NOT one network trained on everything" in desc, (
            "describe must state the honest framing"
        )
        # determinism: identical inputs -> identical predictions on repeat (RNG-independent)
        s_before = bundle.scores("vision", x)
        for _ in range(5):
            random.random()  # perturb the global RNG between calls
        assert bundle.scores("vision", x) == s_before, "predictions must be deterministic"
        assert bundle.predict("shapes", x) == pred_shapes, "predictions must be deterministic"

        # trust boundary: a mis-shaped skill file is REJECTED by load_bundle
        with open(os.path.join(work, "bad_skill.json"), "w", encoding="utf-8") as f:
            json.dump({"format": "not-a-brain", "layers": []}, f)
        bad_path = save_bundle(
            make_bundle({"broken": "bad_skill.json"}), os.path.join(work, "bad_bundle.json")
        )
        bad_rejected = False
        try:
            load_bundle(bad_path)
        except ValueError:
            bad_rejected = True
        assert bad_rejected, "a mis-shaped skill file must be rejected at the trust boundary"
        # a wrong bundle format and an empty skills dict are also rejected
        with open(os.path.join(work, "wrong_fmt.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"format": "nope", "skills": {"v": {"kind": "trained_brain", "path": "x"}}}, f
            )
        with open(os.path.join(work, "empty.json"), "w", encoding="utf-8") as f:
            json.dump({"format": BUNDLE_FORMAT, "skills": {}}, f)
        for bad in ("wrong_fmt.json", "empty.json"):
            rejected = False
            try:
                load_bundle(os.path.join(work, bad))
            except ValueError:
                rejected = True
            assert rejected, f"{bad} must be rejected by load_bundle"
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(
        f"self-check OK: bundle packages {len(manifest['skills'])} SEPARATE trained skills "
        f"(routes to each; flags for wired reflex/cortisol/pid/cpg), round-trips, deterministic, "
        f"and rejects bad skill files at the trust boundary"
    )


if __name__ == "__main__":
    main()
