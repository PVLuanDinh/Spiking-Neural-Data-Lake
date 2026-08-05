r"""
MODEL BUILDER -- "build me a 3D model of X": acquire parts, think through the assembly, write an .obj.

HONEST SCOPE (the point): this is a PROCEDURAL geometry engine over three primitives (box, cylinder,
cone) plus a small built-in PART CATALOG and per-thing assembly recipes. "Acquire" means generating a
primitive from the catalog -- nothing is downloaded, nothing is learned from 3D data, and the mesh is
NOT stored in the knowledge lake (a mesh is an artifact, not knowledge). What IS knowledge -- and what
the chat stores durably -- is the PROCESS: what the thing is made of and how it was assembled.

The DECOMPOSITION is a real train of thought: the assembly sequence is walked link by link through the
spiking global workspace (train_of_thought.TrainOfThought), each next part gated by actual ignition --
the same serial re-entrant loop the chat thinks with, pointed at a build plan.

  ModelBuilder().build("spaceship")            -> (mesh, part labels, thought chain)
  ModelBuilder().save_obj(mesh, "s.obj")       -> a valid Wavefront OBJ file
  ModelBuilder().knowledge("spaceship", path)  -> (cue, answer, triple) facts for the KnowledgeLog

Deps: stdlib + reasoning/train_of_thought.py. Deterministic: same thing -> byte-identical OBJ.
(The "(research)" tag here named a directory retired in v3.78; corrected 2026-08-04.)
"""

import math
import os
import sys
import tempfile

from snn_data_lake.reasoning.train_of_thought import TrainOfThought


# ---------------------------------------------------------------- primitives (verts, faces) --------
# A mesh is (vertices, faces): vertices = [(x, y, z), ...], faces = 0-indexed vertex-index tuples
# (tris or quads -- OBJ takes both). Coordinates are rounded so the OBJ text is platform-stable.


def box(w, h, d):
    """Axis-aligned box centered at the origin: 8 vertices, 6 quad faces."""
    x, y, z = w / 2.0, h / 2.0, d / 2.0
    v = [
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    ]
    f = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return v, f


def cylinder(r, h, n=12):
    """Cylinder along +Z centered at the origin: two n-rings + 2 cap centers, side quads + cap tris."""
    lo, hi = -h / 2.0, h / 2.0
    ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    v = [(r * c, r * s, lo) for c, s in ring] + [(r * c, r * s, hi) for c, s in ring]
    v += [(0.0, 0.0, lo), (0.0, 0.0, hi)]  # cap centers at 2n, 2n+1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f.append((i, j, n + j, n + i))  # side quad
        f.append((j, i, 2 * n))  # bottom cap tri (faces -Z)
        f.append((n + i, n + j, 2 * n + 1))  # top cap tri (faces +Z)
    return v, f


def cone(r, h, n=12):
    """Cone along +Z: base n-ring at z=0, apex at z=h, base center. Side + base tris."""
    ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    v = [(r * c, r * s, 0.0) for c, s in ring] + [
        (0.0, 0.0, h),
        (0.0, 0.0, 0.0),
    ]  # apex n, center n+1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f.append((i, j, n))  # side tri to the apex
        f.append((j, i, n + 1))  # base tri (faces -Z)
    return v, f


def translate(mesh, dx, dy, dz):
    v, f = mesh
    return [(x + dx, y + dy, z + dz) for x, y, z in v], f


def merge(meshes):
    """Concatenate meshes into one, re-offsetting face indices."""
    verts, faces = [], []
    for v, f in meshes:
        off = len(verts)
        verts += v
        faces += [tuple(i + off for i in face) for face in f]
    return verts, faces


# ---------------------------------------------------------------- catalog + recipes ----------------


class PartCatalog:
    """The built-in procedural part source. acquire(primitive, **params) generates the part and
    records the acquisition (provenance: everything comes from HERE, not the internet)."""

    _PRIMITIVES = {"box": box, "cylinder": cylinder, "cone": cone}

    def __init__(self):
        self.acquired = []  # (primitive, params) provenance log

    def acquire(self, primitive, **params):
        mesh = self._PRIMITIVES[primitive](**params)
        self.acquired.append((primitive, dict(params)))
        return mesh


# thing -> ordered assembly plan: (part label, catalog primitive, params, placement offset).
# Order IS the assembly sequence the train of thought walks.
RECIPES = {
    "spaceship": [
        ("hull", "cylinder", {"r": 1.0, "h": 4.0}, (0.0, 0.0, 0.0)),
        ("nose", "cone", {"r": 1.0, "h": 1.6}, (0.0, 0.0, 2.0)),
        ("left wing", "box", {"w": 3.0, "h": 0.15, "d": 1.2}, (-2.0, 0.0, -0.8)),
        ("right wing", "box", {"w": 3.0, "h": 0.15, "d": 1.2}, (2.0, 0.0, -0.8)),
        ("fin", "box", {"w": 0.15, "h": 1.6, "d": 1.0}, (0.0, 0.9, -1.5)),
        ("engine", "cylinder", {"r": 0.45, "h": 0.8}, (0.0, 0.0, -2.4)),
    ],
    "rocket": [
        ("body", "cylinder", {"r": 0.8, "h": 5.0}, (0.0, 0.0, 0.0)),
        ("nose cone", "cone", {"r": 0.8, "h": 1.8}, (0.0, 0.0, 2.5)),
        ("fin a", "box", {"w": 2.4, "h": 0.12, "d": 1.0}, (0.0, 0.0, -2.2)),
        ("fin b", "box", {"w": 0.12, "h": 2.4, "d": 1.0}, (0.0, 0.0, -2.2)),
        ("nozzle", "cone", {"r": 0.6, "h": 0.8}, (0.0, 0.0, -3.3)),
    ],
    "house": [
        ("base", "box", {"w": 4.0, "h": 3.0, "d": 2.4}, (0.0, 0.0, 0.0)),
        ("roof", "cone", {"r": 2.6, "h": 1.6, "n": 4}, (0.0, 0.0, 1.2)),  # n=4 -> a pyramid roof
        ("door", "box", {"w": 0.8, "h": 0.2, "d": 1.4}, (0.0, -1.55, -0.5)),
        ("chimney", "box", {"w": 0.5, "h": 0.5, "d": 1.2}, (1.2, 0.6, 2.0)),
    ],
}


class ModelBuilder:
    """decompose (a train of thought over the assembly plan) -> acquire parts -> build -> save_obj."""

    def __init__(self):
        self.catalog = PartCatalog()

    def decompose(self, thing):
        """Walk the assembly sequence as a SERIAL TRAIN OF THOUGHT: thing -> part -> part -> ...,
        every hop gated by real spiking ignition in the global workspace. Returns the ignited chain
        (thing first). Unknown thing -> KeyError (the caller answers with an honest gap)."""
        labels = [label for label, _, _, _ in RECIPES[thing]]
        seq = [thing] + labels
        adj = {a: [(b, 0.9)] for a, b in zip(seq, seq[1:])}  # linear assembly-order graph

        def associate(broadcast, chain):
            return [t for t in adj.get(broadcast, []) if t[0] not in chain]

        # hysteresis HOLDs the incumbent 1-3 ticks before the successor ignites -> ~4 ticks/link
        return TrainOfThought(seed=0).think(thing, associate, max_steps=4 * len(seq))

    def build(self, thing):
        """Acquire every part from the catalog, place it, merge. Returns (mesh, labels, chain)."""
        chain = self.decompose(thing)
        parts, labels = [], []
        for label, primitive, params, (dx, dy, dz) in RECIPES[thing]:
            parts.append(translate(self.catalog.acquire(primitive, **params), dx, dy, dz))
            labels.append(label)
        return merge(parts), labels, chain

    def save_obj(self, mesh, path):
        """Write a valid Wavefront OBJ (1-indexed faces). Deterministic text -> byte-stable."""
        verts, faces = mesh
        lines = ["# built by snn-data-lake model_builder (procedural primitives)"]
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in verts]
        lines += ["f " + " ".join(str(i + 1) for i in face) for face in faces]
        with open(path, "w", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    def knowledge(self, thing, labels, path):
        """The facts worth KEEPING: the process/how-to (durable knowledge), never the mesh."""
        facts = [
            (
                ["build", thing],
                f"to build a {thing}: acquire {', '.join(labels)} from the part catalog, place "
                f"each part, and write the mesh as an OBJ file (i saved {os.path.basename(path)}).",
                [thing, "builtfrom", labels[0]],
            )
        ]
        facts += [
            ([thing, lab.split()[-1]], f"a {thing} has a {lab}.", [thing, "madeof", lab])
            for lab in labels
        ]
        return facts


# ---------------------------------------------------------------- self-check -----------------------


def main():
    print("=" * 96)
    print(
        "MODEL BUILDER -- procedural 'build me a 3D model' engine (catalog parts + train-of-thought)"
    )
    print("=" * 96)
    b = ModelBuilder()

    for thing in sorted(RECIPES):
        mesh, labels, chain = b.build(thing)
        verts, faces = mesh
        # the train of thought must walk the WHOLE assembly plan, ignition-gated, thing first
        assert chain == [thing] + labels, (thing, chain)
        assert verts and faces
        assert all(0 <= i < len(verts) for f in faces for i in f), "face index out of range"
        path = os.path.join(tempfile.gettempdir(), f"sndl_model_{thing}.obj")
        b.save_obj(mesh, path)
        with open(path) as fh:
            text = fh.read()
        assert text.count("\nv ") + text.startswith("v ") == len(verts)
        # deterministic: building + saving again is byte-identical
        mesh2, _, _ = ModelBuilder().build(thing)
        path2 = path + ".again"
        b.save_obj(mesh2, path2)
        with open(path2) as fh:
            assert fh.read() == text, "non-deterministic OBJ"
        os.remove(path2)
        print(
            f"  built {thing:10s} {len(labels)} parts, {len(verts):3d} verts, {len(faces):3d} faces"
            f"  thought: {' -> '.join(chain)}"
        )

    # the process facts are about HOW, and never contain geometry
    facts = b.knowledge("spaceship", [lb for lb, _, _, _ in RECIPES["spaceship"]], "spaceship.obj")
    assert any("acquire" in ans for _, ans, _ in facts)
    assert all("v " not in ans for _, ans, _ in facts), "mesh leaked into knowledge"

    # unknown thing -> honest failure, not a guessed mesh
    try:
        b.build("griffin")
        raise AssertionError("built an unknown thing")
    except KeyError:
        print("  'griffin' -> honest KeyError (no recipe): the chat turns this into an honest gap")

    print("\nSELF-CHECK PASSED: deterministic OBJs, ignition-gated decomposition, honest unknowns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
