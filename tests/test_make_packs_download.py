"""The DOWNLOAD path of training/make_packs.py -- the branch that fetches a real dataset instead of
running --smoke on synthetic arrays.

WHY THIS FILE EXISTS. make_packs.py's own `--smoke` deliberately covers only the pure helpers: it
must run with "no tonic, no dataset, no network", so it never touches `_open_dataset` ->
`build_pack` -> `run_build`, which is every line that decides whether a pack gets WRITTEN. That
branch had never executed under any check. On 2026-07-30 it was run for real -- both packs, cold,
against an empty --data directory -- and it reproduced the committed artifacts byte-for-byte
(crc32 0xaca9f8b2 / 0xb1bd2e57, 0 differing cells). It also exposed a hole: `check_schema` guarded
every content assertion on non-emptiness, so a 0-sample pack passed with `schema_problems=0` and
`run_build` wrote it over models/ (the default --out) and exited 0.

WHAT IS STUBBED, AND WHAT IS NOT. Only tonic is replaced -- with a module whose dataset classes
construct exactly as tonic 1.6.0's do (construction IS the download at that version; there is no
`download=` keyword to pass) and record the fetch as a marker file instead of hitting the network.
Everything under test is make_packs' own code, called directly: `_open_dataset`, `_as_frame`,
`build_pack`, `check_schema`, `describe`, and `run_build` driven through the REAL `build_parser()`.
Nothing here re-implements the producer, and nothing here needs a network or a dataset on disk.
"""

import contextlib
import os
import sys
import types
import zipfile

import numpy as np

from training import make_packs as mp


class _StubToFrame:
    """Stands in for tonic.transforms.ToFrame: same constructor keywords, same output RANK.

    The per-bin counts are not tonic's binning and are not meant to be -- reproducing the committed
    packs is what the real cold run proved. What matters to make_packs is the SHAPE tonic hands
    back, because `_as_frame` exists solely to collapse SHD's trailing singleton axes, and a stub
    that emitted the already-collapsed shape would test nothing.
    """

    def __init__(self, sensor_size, n_time_bins):
        self.sensor_size = tuple(sensor_size)
        self.n_time_bins = int(n_time_bins)

    def __call__(self, index):
        w, h, p = self.sensor_size
        if h == 1 and p == 1:  # SHD sensor (700, 1, 1) -> ToFrame emits (T, 700, 1, 1)
            shape = (self.n_time_bins, w, 1, 1)
        else:  # NMNIST sensor (34, 34, 2) -> ToFrame emits (T, 2, 34, 34)
            shape = (self.n_time_bins, p, h, w)
        return np.random.default_rng(index).integers(0, 4, size=shape).astype(np.float32)


@contextlib.contextmanager
def _stub_tonic():
    """Install a fake `tonic` for the duration of the block; yield the list of datasets built.

    sys.modules is saved and restored around the block, including the None entries, so a real tonic
    (present in .venv-cu128, absent in .venv-lake) is neither shadowed afterwards nor required now.
    """
    built = []

    class _StubDataset:
        sensor_size = ()
        n_samples = 0

        def __init__(self, save_to, train=True, transform=None, target_transform=None):
            self.save_to, self.train, self.transform = save_to, train, transform
            self.requested = []
            # tonic 1.6.0 downloads unconditionally at construction time whenever _check_exists()
            # is false. The marker records that same event, in the same place a fetch would land.
            folder = os.path.join(save_to, type(self).__name__)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "DOWNLOADED"), "w", encoding="ascii") as fh:
                fh.write(f"train={train}\n")
            built.append(self)

        def __len__(self):
            return self.n_samples

        def __getitem__(self, index):
            self.requested.append(index)
            return self.transform(index), index % 10

    class NMNIST(_StubDataset):
        sensor_size = (34, 34, 2)
        n_samples = mp.SPECS["nmnist"]["split_size"]

    class SHD(_StubDataset):
        sensor_size = (700, 1, 1)
        n_samples = mp.SPECS["shd"]["split_size"]

    tonic = types.ModuleType("tonic")
    datasets = types.ModuleType("tonic.datasets")
    transforms = types.ModuleType("tonic.transforms")
    datasets.NMNIST, datasets.SHD = NMNIST, SHD
    transforms.ToFrame = _StubToFrame
    tonic.datasets, tonic.transforms = datasets, transforms

    names = ("tonic", "tonic.datasets", "tonic.transforms")
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules.update(dict(zip(names, (tonic, datasets, transforms), strict=True)))
    try:
        yield built
    finally:
        for n in names:
            if saved[n] is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = saved[n]


def _args(*argv):
    """Parse through the module's REAL parser, so the defaults under test are the shipped ones."""
    return mp.build_parser().parse_args(list(argv))


def test_open_dataset_constructs_the_download(tmp_path):
    """`_open_dataset` IS the download: constructing fetches, into --data, on the test split."""
    data = tmp_path / "data"
    with _stub_tonic():
        ds = mp._open_dataset("shd", str(data))

    assert (data / "SHD" / "DOWNLOADED").exists(), "the fetch must land under --data"
    assert ds.train is False, "both packs are drawn from the TEST split"
    assert ds.transform.sensor_size == (700, 1, 1), "ToFrame binds the dataset's own sensor_size"
    assert ds.transform.n_time_bins == mp.SPECS["shd"]["n_time_bins"] == 100


def test_build_pack_reads_exactly_the_pinned_indices(tmp_path):
    """build_pack pulls the samples select_indices names -- in order -- and shapes them per spec."""
    with _stub_tonic() as built:
        frames, labels = mp.build_pack("shd", str(tmp_path / "data"), n_samples=8, seed=0)

    wanted = mp.select_indices(mp.SPECS["shd"]["split_size"], 8, 0)
    assert built[-1].requested == wanted, "the selection rule must drive which samples are read"
    # (100, 700), not tonic's raw (100, 700, 1, 1): _as_frame's collapse actually ran.
    assert frames.shape == (8, 100, 700) and frames.dtype == np.float32
    assert labels.shape == (8,) and labels.dtype == np.int64
    assert mp.check_schema("shd", frames, labels) == [], "a built pack must satisfy the schema"


def test_nmnist_frames_keep_tonics_layout(tmp_path):
    """N-MNIST needs no collapse; the pack keeps ToFrame's (T, 2, 34, 34) exactly."""
    with _stub_tonic():
        frames, labels = mp.build_pack("nmnist", str(tmp_path / "data"), n_samples=4, seed=0)
    assert frames.shape == (4, 15, 2, 34, 34)
    assert mp.check_schema("nmnist", frames, labels) == []


def test_empty_pack_is_a_schema_violation():
    """REGRESSION. Every content check is guarded on non-emptiness; 0 samples must still be caught.

    Before the fix this returned [] -- a clean bill of health earned by measuring nothing.
    """
    frames, labels = mp._synthetic("nmnist")
    assert mp.check_schema("nmnist", frames, labels) == [], "the non-empty control must pass"
    problems = mp.check_schema("nmnist", frames[:0], labels[:0])
    assert problems, "an empty pack must NOT pass vacuously"
    assert "0 samples" in problems[0]


def test_describe_survives_an_empty_pack():
    """REGRESSION. describe() died in numpy on the empty pack it was supposed to be reporting on."""
    frames, labels = mp._synthetic("shd")
    empty = mp.describe(frames[:0], labels[:0])
    for key in ("min", "max", "mean", "nonzero_frac", "per_sample_min", "label_max"):
        assert empty[key] is None, f"{key} must be None, not a 0.0 that looks like a measurement"
    assert empty["total_events"] == 0, "0 events is genuinely computed, not a placeholder"
    assert empty["shape"] == (0, 100, 700)
    real = mp.describe(frames, labels)
    assert real["min"] is not None and real["total_events"] > 0, "the non-empty control still works"


def test_run_build_refuses_to_write_an_empty_pack(tmp_path):
    """REGRESSION, end to end. `--n 0` used to write a 510-byte pack over models/ and exit 0."""
    out = tmp_path / "out"
    args = _args(
        "--which",
        "shd",
        "--n",
        "0",
        "--data",
        str(tmp_path / "data"),
        "--out",
        str(out),
    )
    with _stub_tonic():
        ok = mp.run_build(args)

    assert ok is False, "run_build must FAIL on a 0-sample pack, not succeed with caveats"
    assert not (out / "shd_pack.npz").exists(), "nothing may be written over a reader's pack"


def test_run_build_refuses_when_the_fingerprint_drifts(tmp_path):
    """The gate is real: at the canonical (256, 0) a pack that is not the committed one is refused.

    The stub's frames are deliberately not tonic's binning, so this is the MISMATCH branch --
    reachable no other way without corrupting a real dataset.
    """
    out = tmp_path / "out"
    args = _args("--which", "nmnist", "--data", str(tmp_path / "data"), "--out", str(out))
    assert (args.n, args.seed) == (None, None), "the canonical config is the parser default"
    with _stub_tonic():
        ok = mp.run_build(args)

    assert ok is False, "a pack that misses the recorded crc32 must not be written"
    assert not (out / "nmnist_pack.npz").exists()


def test_run_build_writes_a_non_canonical_pack(tmp_path):
    """The positive control: off the canonical config the gate cannot apply, so the pack IS written
    -- with a caveat, and with the real row count on disk rather than a constant."""
    out = tmp_path / "out"
    args = _args(
        "--which",
        "shd",
        "--n",
        "6",
        "--seed",
        "3",
        "--data",
        str(tmp_path / "data"),
        "--out",
        str(out),
    )
    with _stub_tonic():
        ok = mp.run_build(args)

    assert ok is True
    written = out / "shd_pack.npz"
    assert written.exists() and written.stat().st_size > 0
    with np.load(str(written)) as d:
        assert d["frames"].shape == (6, 100, 700) and d["frames"].dtype == np.float32
        assert d["labels"].shape == (6,) and d["labels"].dtype == np.int64
        assert mp.check_schema("shd", d["frames"], d["labels"]) == []


def test_written_pack_is_compressed_like_the_committed_artifacts(tmp_path):
    """REGRESSION. write_pack used np.savez; the packs every reader loads are savez_compressed.

    The fingerprint and the comparison are both over ARRAY bytes, so a rebuild could reproduce the
    committed packs cell-for-cell and still land 45x too large on disk (35,514,878 B + 71,682,558 B
    against the artifacts' 754,456 B + 1,648,240 B) with nothing to notice. Assert the storage form,
    since that is the part no array-level check can see.
    """
    frames, labels = mp._synthetic("nmnist", n=8)
    path = mp.write_pack(str(tmp_path / "pack.npz"), frames, labels)

    with zipfile.ZipFile(path) as z:
        members = z.infolist()
    assert sorted(m.filename for m in members) == ["frames.npy", "labels.npy"]
    kinds = {m.compress_type for m in members}
    assert kinds == {zipfile.ZIP_DEFLATED}, "members must be DEFLATED, not STORED"
    assert all(m.compress_size < m.file_size for m in members), "compression must really shrink it"

    with np.load(path) as d:  # and the readers' plain np.load still sees the identical arrays
        assert np.array_equal(d["frames"], frames) and d["frames"].dtype == np.float32
        assert mp.fingerprint(d["frames"], d["labels"]) == mp.fingerprint(frames, labels)


def test_dry_run_builds_but_writes_nothing(tmp_path):
    """--dry-run must exercise the whole build and leave the output directory empty."""
    out = tmp_path / "out"
    args = _args(
        "--which",
        "shd",
        "--n",
        "6",
        "--seed",
        "3",
        "--dry-run",
        "--data",
        str(tmp_path / "data"),
        "--out",
        str(out),
    )
    with _stub_tonic() as built:
        ok = mp.run_build(args)

    assert ok is True
    assert built and len(built[-1].requested) == 6, "the samples were really read"
    assert not (out / "shd_pack.npz").exists(), "--dry-run must write nothing"
