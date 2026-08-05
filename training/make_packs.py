"""
training/make_packs.py -- THE PRODUCER for the two perception packs the live brain reads:
models/nmnist_pack.npz (event VISION) and models/shd_pack.npz (event AUDIO).

WHY THIS EXISTS. Until now these two files had FIVE reader sites and ZERO producers anywhere in src/,
training/, tools/, integrations/ or research/. `cortex/brain.py` loads them (SNN_PERCEPTION_PACK,
SNN_SHD_PACK) to give the live tick REAL event-camera / cochlear input and to set
`perceiving_real` / `perceiving_audio`; `cortex/percept_surprise.py` rate-codes the vision pack into
its Oja-PCA surprise instrument; `reasoning/reservoir_eprop.py` re-bins the audio pack for its
held-out reservoir benchmark. `models/` is gitignored and docs/TRAINING.md regenerated only the `.pt`
encoder checkpoints -- so a FRESH CLONE HAD NO PERCEPTION AT ALL and no command that could restore it.
Seven of Perception's thirteen rubric faculties, including all four highest, lived only inside two
untracked binaries. This module is the missing half of the pair the `.pt` trainers already document.

THE PACK CONTRACT (read off the committed artifacts, not guessed):
  nmnist_pack.npz : frames (256, 15, 2, 34, 34) float32, integral event COUNTS 0..9 ; labels (256,) int64 0..9
  shd_pack.npz    : frames (256, 100, 700)      float32, integral event COUNTS 0..5 ; labels (256,) int64 0..19

BINNING -- INFERRED, THEN PROVEN, NOT ASSUMED. The frames are per-bin event COUNTS from
`tonic.transforms.ToFrame(sensor_size=<dataset>.sensor_size, n_time_bins=T)`: uniform time bins over
each sample's OWN [t_min, t_max] span (so the time axis is per-sample normalized, not absolute
microseconds), accumulating one count per event into (bin, polarity, y, x) for N-MNIST and
(bin, channel) for SHD. Two consequences worth stating because they look like bugs otherwise:
  * ToFrame drops exactly ONE event per sample (the event at t == t_max falls on the closing edge), so
    `frames.sum()` is n_events - 1, not n_events. This is the tell that identified the binning: the
    committed sums are one BELOW the raw event counts of their source files.
  * N-MNIST's 15 bins land 5-per-saccade across the recording's 3 saccades, which is why the mean
    per-bin event profile has three peaks. That is the data, not an artefact of the choice of T.
The inference was verified the only way that settles it -- REPRODUCTION. Both packs are drawn from the
TEST split (not train: the train split's event-count multiset does not contain the committed sums; the
test split's does, for all 256/256), the sample choice is `random.Random(0).sample(range(len(ds)), 256)`
(recovered by matching 194 independently identified source files to their dataset indices: 194/194),
and regenerating on that basis reproduces both committed packs BYTE-FOR-BYTE -- as of 2026-07-30 that
is true of the FILES and not merely of the arrays inside them, which is a correction, not a boast:
write_pack used np.savez where the artifacts are savez_compressed, so the claim held cell-for-cell
while every rebuilt .npz came out 45x too large. See write_pack. `SPECS[kind]["crc32"]` below
records each pack's fingerprint so that equality is a gate, not a claim: a rebuild that does not
reproduce it REFUSES TO WRITE. (This sentence, and the one in `fingerprint`, both named a constant
called EXPECTED_CRC32 -- corrected 2026-08-04: no such name exists anywhere in the repo, and
`run_build` was already citing the real one correctly two hundred lines below. Naming the SYMBOL a
reader can grep for is the point of the citation.) A pack that loads but encodes differently would silently move every
perception number in the project, which is strictly worse than having no producer.

DOWNLOAD -- EXERCISED, not asserted. `data/` is gitignored, so a fresh clone has neither dataset.
tonic 1.6.0's `NMNIST` and `SHD` constructors take NO `download=` keyword -- they download
unconditionally when `_check_exists()` is false, which is the download-on-demand this needs;
constructing the dataset IS the download. Stated explicitly because "download=True" is not a flag
that exists to pass at this tonic version. The claim above was written from reading tonic's source;
on 2026-07-30 it was finally RUN, both packs, against an empty `--data` directory on a machine that
had never fetched either set -- `shd_test.h5.zip` (38,141,465 B) from zenkelab.org and `test.zip`
(169,674,850 B) from the Mendeley record, extracted, binned, and compared cell-by-cell to the
committed artifacts: 0 of 17,920,000 and 0 of 8,878,080 cells differ, crc32 0xaca9f8b2 and
0xb1bd2e57, both gates MATCH. A cold clone really can restore perception from the network. What that
run DID expose was the empty-pack hole documented on check_schema() below.

WHAT THIS IS NOT (dependency direction). OPTIONAL torch/[nmnist] tier, like every other training/
module: imports NOTHING from src/snn_data_lake/, and tonic is imported LAZILY so `--smoke` runs with
neither tonic nor a dataset nor a network. Writes only the .npz packs; trains nothing.

  PYTHONUTF8=1 .venv-lake/Scripts/python training/make_packs.py --smoke
      # deterministic assert-based self-check (no tonic, no dataset, no network)
  PYTHONUTF8=1 .venv-cu128/Scripts/python training/make_packs.py
      # download if needed, build BOTH packs into models/, gated on the recorded fingerprint
  PYTHONUTF8=1 .venv-cu128/Scripts/python training/make_packs.py --which shd --out /tmp/p --compare-to models
      # build one pack elsewhere and print a full generated-vs-existing comparison

ASCII-only output (Windows cp1252-safe). Selection uses random.Random(seed); fingerprints use
zlib.crc32. Python's builtin hash() is never used (it is salted per process).
"""

import argparse
import os
import random
import sys
import tempfile
import zlib

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The two packs, exactly as the readers require them. `crc32` is the fingerprint of the COMMITTED
# artifact at the canonical (n_samples, seed); a rebuild must match it or it is not the same pack.
SPECS = {
    "nmnist": {
        "filename": "nmnist_pack.npz",
        "dataset": "NMNIST",
        "train": False,  # the TEST split -- proven by event-count multiset, 256/256
        "n_time_bins": 15,
        "frame_shape": (15, 2, 34, 34),
        "n_classes": 10,
        "n_samples": 256,
        "seed": 0,
        "split_size": 10000,
        "crc32": 0xB1BD2E57,
        "reader": "cortex/brain.py (SNN_PERCEPTION_PACK), cortex/percept_surprise.py",
    },
    "shd": {
        "filename": "shd_pack.npz",
        "dataset": "SHD",
        "train": False,
        "n_time_bins": 100,
        "frame_shape": (100, 700),
        "n_classes": 20,
        "n_samples": 256,
        "seed": 0,
        "split_size": 2264,
        "crc32": 0xACA9F8B2,
        "reader": "cortex/brain.py (SNN_SHD_PACK), reasoning/reservoir_eprop.py",
    },
}


# ----------------------------------------------------------------------------
# Determinism: selection + fingerprint. Both are pure and testable without tonic.
# ----------------------------------------------------------------------------
def select_indices(n_total, n_take, seed):
    """The deterministic sample choice: stdlib `random.Random(seed).sample(range(n_total), n_take)`.

    This is not an arbitrary convention picked here -- it is the rule the committed packs were
    actually produced with, recovered by identifying 194 of the 256 source files independently (via
    their (label, event-count) signatures) and solving for the generator that maps pack position to
    dataset index. random.Random.sample matched 194/194 at seed 0. Mersenne Twister is stable across
    CPython versions and platforms, so this reproduces everywhere; builtin hash() would not (salted).
    """
    if n_take > n_total:
        raise ValueError("cannot take %d samples from a split of %d" % (n_take, n_total))
    return random.Random(seed).sample(range(n_total), n_take)


def fingerprint(frames, labels):
    """crc32 over shape + dtype + raw bytes of both arrays -- the pack's identity as one integer.

    Shape and dtype are folded in first so a reshaped or re-typed pack cannot collide with the
    original, and dtype.str carries endianness ('<f4'), so the digest is portable rather than
    accidentally x86-specific. zlib.crc32, never hash(): this value is persisted in
    `SPECS[kind]["crc32"]` and gated against a rebuild in `run_build`.
    """
    head = "%s|%s|%s|%s" % (frames.shape, frames.dtype.str, labels.shape, labels.dtype.str)
    c = zlib.crc32(head.encode("ascii"))
    c = zlib.crc32(np.ascontiguousarray(frames).tobytes(), c)
    return zlib.crc32(np.ascontiguousarray(labels).tobytes(), c)


def check_schema(kind, frames, labels):
    """Return a list of ASCII problem strings; empty means the pack satisfies every reader.

    Checks the contract the five reader sites actually depend on: float32 count frames of the right
    per-sample shape, int64 labels in range, matching sample counts, and non-negative integral values
    (the readers sum and rate-code these -- a normalized or signed pack would read as valid and be
    silently wrong).

    THE EMPTY PACK IS THE FIRST CHECK, and it is here because it was missing. Every content assertion
    below is guarded on `frames.size` / `labels.size`, so a zero-sample pack skipped all of them and
    this function returned [] -- a clean bill of health earned by measuring nothing. `--n 0` therefore
    built a 510-byte .npz, printed `schema_problems=0`, and run_build WROTE it over
    models/shd_pack.npz (the default --out) and exited 0. That pack still loads: cortex/brain.py sets
    `_perceiving_real`/`_perceiving_audio` True on `np.load` succeeding, so the brain would report it
    perceives real event vision and audio while holding zero frames, and the first index into them
    would land in that loader's blanket `except Exception` and silently drop back to the stdlib tier.
    A vacuous pass is worse than a failure, so 0 samples is now a violation like any other.
    """
    spec = SPECS[kind]
    bad = []
    if frames.size == 0 or labels.size == 0:
        bad.append(
            "pack holds 0 samples (frames %s, labels %s); every content check below is guarded on "
            "non-emptiness, so an empty pack would pass vacuously"
            % (tuple(frames.shape), tuple(labels.shape))
        )
    if frames.dtype != np.float32:
        bad.append("frames dtype is %s, must be float32" % frames.dtype.str)
    if labels.dtype != np.int64:
        bad.append("labels dtype is %s, must be int64" % labels.dtype.str)
    if frames.ndim != 1 + len(spec["frame_shape"]) or tuple(frames.shape[1:]) != spec["frame_shape"]:
        bad.append(
            "frames per-sample shape is %s, must be %s" % (tuple(frames.shape[1:]), spec["frame_shape"])
        )
    if labels.ndim != 1 or labels.shape[0] != frames.shape[0]:
        bad.append("labels shape %s does not match %d frames" % (tuple(labels.shape), frames.shape[0]))
    if frames.size and float(frames.min()) < 0.0:
        bad.append("frames contain negatives (min %.4f); these are event COUNTS" % float(frames.min()))
    if frames.size and not bool(np.all(frames == np.round(frames))):
        bad.append("frames are not integral; these are event COUNTS, not a normalized rate")
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= spec["n_classes"]):
        bad.append(
            "labels out of range [0, %d): min %d max %d"
            % (spec["n_classes"], int(labels.min()), int(labels.max()))
        )
    return bad


def describe(frames, labels):
    """The distribution numbers a comparison is actually judged on (all plain Python types).

    Every reduction is guarded on emptiness and reports None -- not 0.0 -- when there is nothing to
    reduce. Two reasons, both learned from this function raising:
      * UNGUARDED, a zero-sample pack killed compare_to_existing() here with numpy's
        "zero-size array to reduction operation minimum which has no identity", i.e. the comparison
        whose whole job is to REPORT that the pack is empty died before printing a word of it.
      * 0.0 would be a constant sitting where a computed value belongs -- indistinguishable in the
        report from a genuine pack whose frames really do min/max/mean to zero. None cannot be
        mistaken for data, and it compares unequal to the reference's real number, so the row is
        flagged DIFFERS rather than quietly matching.
    total_events stays computed: an empty pack honestly holds 0 events.
    """
    # Guarded on size, not on ndim: reshape(0, -1) cannot infer the -1 from a size-0 array either
    # ("cannot reshape array of size 0 into shape (0,newaxis)"), so the per-sample sums are simply
    # not computed when there are no samples to sum.
    per_sample = frames.reshape(frames.shape[0], -1).sum(1) if frames.size else frames.ravel()
    return {
        "shape": tuple(frames.shape),
        "dtype": frames.dtype.str,
        "labels_dtype": labels.dtype.str,
        "min": float(frames.min()) if frames.size else None,
        "max": float(frames.max()) if frames.size else None,
        "mean": float(frames.mean()) if frames.size else None,
        "nonzero_frac": float((frames > 0).mean()) if frames.size else None,
        "total_events": int(frames.sum()),
        "per_sample_min": int(per_sample.min()) if per_sample.size else None,
        "per_sample_max": int(per_sample.max()) if per_sample.size else None,
        "label_min": int(labels.min()) if labels.size else None,
        "label_max": int(labels.max()) if labels.size else None,
        "crc32": fingerprint(frames, labels),
    }


# ----------------------------------------------------------------------------
# Build. tonic is imported lazily so --smoke needs neither it nor a dataset.
# ----------------------------------------------------------------------------
def _open_dataset(kind, data_dir):
    """The tonic dataset with ToFrame binning attached. Constructing it DOWNLOADS on demand (tonic
    1.6.0 has no download= keyword; it fetches whenever the files are absent), which is what makes a
    fresh clone -- where data/ is gitignored and empty -- able to run this at all."""
    import tonic  # noqa: PLC0415 -- lazy: --smoke must run without tonic
    import tonic.transforms as TT  # noqa: PLC0415

    spec = SPECS[kind]
    cls = getattr(tonic.datasets, spec["dataset"])
    to_frame = TT.ToFrame(sensor_size=cls.sensor_size, n_time_bins=spec["n_time_bins"])
    return cls(save_to=data_dir, train=spec["train"], transform=to_frame)


def _as_frame(kind, raw):
    """One tonic sample -> the pack's per-sample float32 frame. N-MNIST's ToFrame already emits
    (T, 2, 34, 34); SHD's emits (T, 700, 1, 1) because its sensor_size is (700, 1, 1), so the trailing
    singleton axes collapse into the channel axis -- the same reshape eprop_shd._shd_collate does."""
    f = np.asarray(raw, dtype=np.float32)
    return f if kind == "nmnist" else f.reshape(f.shape[0], -1)


def resolve(kind, n_samples, seed):
    """(n, seed, canonical) -- canonical means the config the recorded fingerprint was taken at, so
    the gate can apply. Anything else is a legitimate build of a DIFFERENT pack, and must say so."""
    spec = SPECS[kind]
    n = spec["n_samples"] if n_samples is None else int(n_samples)
    s = spec["seed"] if seed is None else int(seed)
    return n, s, (n == spec["n_samples"] and s == spec["seed"])


def build_pack(kind, data_dir, n_samples=None, seed=None):
    """Build one pack in memory -> (frames, labels). Downloads the dataset if it is not on disk."""
    spec = SPECS[kind]
    n, s, _ = resolve(kind, n_samples, seed)
    ds = _open_dataset(kind, data_dir)
    n_total = len(ds)
    print(
        "%s: %s split=%s size=%d  T=%d  taking %d samples with random.Random(%d).sample"
        % (kind, spec["dataset"], "train" if spec["train"] else "test", n_total, spec["n_time_bins"], n, s)
    )
    if n_total != spec["split_size"]:
        print(
            "  [warn] split size %d != the recorded %d -- the dataset on disk is not the one the"
            % (n_total, spec["split_size"])
        )
        print("         committed pack was drawn from, so the fingerprint gate below will not match")
    idx = select_indices(n_total, n, s)
    frames = np.empty((n,) + spec["frame_shape"], dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)
    for out_pos, i in enumerate(idx):
        raw, lab = ds[i]
        frames[out_pos] = _as_frame(kind, raw)
        labels[out_pos] = int(lab)
    return frames, labels


def write_pack(path, frames, labels):
    """savez_COMPRESSED, because that is what the committed artifacts are and it was not what this
    wrote.

    Caught by running the download path and then looking at the file it produced rather than at the
    arrays inside it. `np.savez` stores members uncompressed, so a rebuild that reproduced the
    committed packs cell-for-cell still landed as 35,514,878 B + 71,682,558 B where the artifacts
    every reader loads are 754,456 B + 1,648,240 B (both DEFLATED): 107 MB of models/ on a fresh
    clone instead of 2.4 MB, a 45x miss. Nothing detected it, because the fingerprint and the
    comparison are both over ARRAY bytes -- correctly, that is the pack's identity -- so "reproduces
    the committed packs byte-for-byte" was true of the contents and quietly false of the artifact.
    Safe for every reader: all five reader sites use plain np.load, none uses mmap_mode (which is
    the one thing compression would break).
    """
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(path, frames=frames, labels=labels)
    return path


def compare_to_existing(kind, frames, labels, ref_path, quiet=False):
    """Print the REAL generated-vs-existing comparison and return True iff byte-identical.

    The existing pack is the ground truth: it is what every published perception number in this repo
    was measured against. So this reports the actual numbers on both sides rather than a verdict --
    and if the binning differs it says so with the magnitude, because a quiet "OK" on a pack that
    encodes differently is the failure mode worth the most guarding against. `quiet` suppresses only
    the report (the self-check calls it twice and its own asserts are the judgement there).
    """
    with np.load(ref_path) as ref:  # context-managed: an open NpzFile blocks deletion on Windows
        rf, rl = ref["frames"], ref["labels"]
    gen, old = describe(frames, labels), describe(rf, rl)
    if not quiet:
        print("  %-16s %-28s %s" % ("field", "generated", "existing (" + os.path.basename(ref_path) + ")"))
        for key in (
            "shape",
            "dtype",
            "labels_dtype",
            "min",
            "max",
            "mean",
            "nonzero_frac",
            "total_events",
            "per_sample_min",
            "per_sample_max",
            "label_min",
            "label_max",
        ):
            flag = "" if gen[key] == old[key] else "   <-- DIFFERS"
            print("  %-16s %-28s %s%s" % (key, gen[key], old[key], flag))
        print("  %-16s %-28s 0x%08x" % ("crc32", "0x%08x" % gen["crc32"], old["crc32"]))
    same_shape = frames.shape == rf.shape
    labels_equal = same_shape and bool(np.array_equal(labels, rl))
    exact = same_shape and labels_equal and bool(np.array_equal(frames, rf))
    if quiet:
        return exact
    if same_shape:
        diff = np.abs(frames - rf)
        n_diff = int((diff > 0).sum())
        print(
            "  frames byte-exact=%s  labels identical=%s  cells differing=%d/%d  max abs diff=%.1f"
            % (exact, labels_equal, n_diff, frames.size, float(diff.max()))
        )
        if not exact:
            print("  BINNING DIFFERS from the committed pack -- do NOT install this over it. The")
            print("  readers would keep loading fine and every perception number would move.")
    else:
        print("  frames byte-exact=False  shapes differ, no cell-wise comparison possible")
    return exact


def run_build(args):
    kinds = ["nmnist", "shd"] if args.which == "both" else [args.which]
    failed, caveats = [], []
    for kind in kinds:
        spec = SPECS[kind]
        print("=" * 78)
        n, seed, canonical = resolve(kind, args.n, args.seed)
        frames, labels = build_pack(kind, args.data, n, seed)
        bad = check_schema(kind, frames, labels)
        for b in bad:
            print("  SCHEMA VIOLATION: %s" % b)
        fp = fingerprint(frames, labels)
        print(
            "  built %s %s  crc32=0x%08x  schema_problems=%d"
            % (frames.shape, frames.dtype.str, fp, len(bad))
        )

        ref = None
        if args.compare_to:
            ref = os.path.join(args.compare_to, spec["filename"])
        elif os.path.exists(os.path.join(args.out, spec["filename"])):
            ref = os.path.join(args.out, spec["filename"])
        if ref and os.path.exists(ref):
            if not compare_to_existing(kind, frames, labels, ref):
                print("  (comparison above is against %s)" % ref)
        elif args.compare_to:
            print("  no existing pack at %s -- nothing to compare against" % ref)

        if canonical:
            match = fp == spec["crc32"]
            print(
                "  fingerprint gate: 0x%08x vs recorded 0x%08x -> %s"
                % (fp, spec["crc32"], "MATCH" if match else "MISMATCH")
            )
            if not match:
                if not args.allow_drift:
                    print("  REFUSING TO WRITE %s: this is not the pack the readers were measured on." % spec["filename"])
                    print("  Re-run with --allow-drift only if the change is intended, and record the new")
                    print("  fingerprint in SPECS[%r]['crc32']." % kind)
                    failed.append(kind)
                    continue
                caveats.append(
                    "%s was written under --allow-drift at crc32 0x%08x, NOT the recorded 0x%08x: every "
                    "perception number measured on it will differ" % (kind, fp, spec["crc32"])
                )
        else:
            print(
                "  fingerprint gate: SKIPPED -- n=%d seed=%d is not the canonical (%d, %d), so the"
                % (n, seed, spec["n_samples"], spec["seed"])
            )
            print("  recorded crc32 cannot apply. This pack is NOT the one the readers were measured on.")
            caveats.append(
                "%s was built at n=%d seed=%d, not the canonical (%d, %d): it is a different pack and "
                "no fingerprint vouches for it" % (kind, n, seed, spec["n_samples"], spec["seed"])
            )
        if bad:
            print("  REFUSING TO WRITE %s: schema violations above." % spec["filename"])
            failed.append(kind)
            continue
        if args.dry_run:
            print("  --dry-run: not written")
            continue
        out = write_pack(os.path.join(args.out, spec["filename"]), frames, labels)
        print("  wrote %s  (readers: %s)" % (out, spec["reader"]))
    print("=" * 78)
    if failed:
        print("FAILED for: %s" % ", ".join(failed))
        return False
    if caveats:
        # Never close with "packs OK" over a pack no fingerprint vouches for.
        print("BUILT WITH CAVEATS -- these are not the packs the repo's perception numbers came from:")
        for c in caveats:
            print("  - %s" % c)
        return True
    print("packs OK. On a fresh clone this plus the two .pt trainers restores the whole perception tier.")
    return True


# ----------------------------------------------------------------------------
# Self-check -- deterministic, assert-based, ASCII-only, and needs NO tonic, dataset or network.
# Exercises every pure piece on synthetic packs, including the negative cases: the point is a check
# that CAN fail, not one that passes because nothing was measured.
# ----------------------------------------------------------------------------
def _synthetic(kind, n=6, seed=0):
    """A schema-VALID synthetic pack: integral non-negative float32 counts + in-range int64 labels."""
    spec = SPECS[kind]
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 4, size=(n,) + spec["frame_shape"]).astype(np.float32)
    labels = rng.integers(0, spec["n_classes"], size=(n,)).astype(np.int64)
    return frames, labels


def run_smoke():
    print("smoke: pure paths only -- no tonic, no dataset, no network")

    # (1) SELECTION is deterministic, repeatable, in-range, duplicate-free -- and is the rule that
    #     reproduces the committed packs. The pinned values are the recovered generator's output;
    #     positions 1,2,3,4,6,7 are the dataset indices of six INDEPENDENTLY identified source files.
    a = select_indices(10000, 256, 0)
    b = select_indices(10000, 256, 0)
    c = select_indices(10000, 256, 1)
    assert a == b, "selection must be reproducible for a fixed seed"
    assert a != c, "selection must depend on the seed"
    assert len(set(a)) == 256 and all(0 <= i < 10000 for i in a), "indices unique and in range"
    assert a[:8] == [6311, 6890, 663, 4242, 8376, 7961, 6634, 4969], (
        "nmnist selection drifted from the rule that reproduces models/nmnist_pack.npz: %s" % a[:8]
    )
    assert select_indices(2264, 256, 0)[:4] == [1577, 1722, 165, 1060], "shd selection drifted"
    try:
        select_indices(10, 20, 0)
        raise AssertionError("taking more samples than the split holds must raise")
    except ValueError:
        pass
    print("  selection: reproducible, seed-sensitive, pinned to the packs' own rule (nmnist[:4]=%s)" % a[:4])

    # (2) FINGERPRINT is stable, and sensitive to content, shape and dtype.
    f, lab = _synthetic("nmnist")
    fp = fingerprint(f, lab)
    assert fp == fingerprint(f.copy(), lab.copy()), "fingerprint must be content-addressed, not identity"
    f2 = f.copy()
    f2[0, 0, 0, 0, 0] += 1.0
    assert fingerprint(f2, lab) != fp, "one changed count must change the fingerprint"
    assert fingerprint(f.astype(np.float64), lab) != fp, "dtype must be folded into the fingerprint"
    assert fingerprint(f.reshape(3, 2, *f.shape[1:]), lab) != fp, "shape must be folded in"
    lab2 = lab.copy()
    lab2[0] = (lab2[0] + 1) % 10
    assert fingerprint(f, lab2) != fp, "labels must be folded into the fingerprint"
    print("  fingerprint: content/shape/dtype/label sensitive; crc32=0x%08x on the synthetic pack" % fp)

    # (3) SCHEMA CHECK accepts a valid pack and catches every way a wrong one still loads.
    for kind in SPECS:
        sf, sl = _synthetic(kind)
        assert check_schema(kind, sf, sl) == [], "valid synthetic %s pack must pass: %s" % (
            kind,
            check_schema(kind, sf, sl),
        )
    sf, sl = _synthetic("nmnist")
    assert check_schema("nmnist", sf.astype(np.float64), sl), "float64 frames must be caught"
    assert check_schema("nmnist", sf, sl.astype(np.int32)), "int32 labels must be caught"
    assert check_schema("nmnist", sf[:, :7], sl), "wrong time-bin count must be caught"
    assert check_schema("nmnist", sf, sl[:3]), "label/frame count mismatch must be caught"
    assert check_schema("nmnist", sf / 2.0, sl), "non-integral (normalized) frames must be caught"
    assert check_schema("nmnist", -sf - 1.0, sl), "negative counts must be caught"
    bad_lab = sl.copy()
    bad_lab[0] = 10
    assert check_schema("nmnist", sf, bad_lab), "out-of-range label must be caught"
    n_shd, l_shd = _synthetic("shd")
    assert check_schema("shd", n_shd.reshape(6, 100, 350, 2), l_shd), "wrong shd channel axis must be caught"
    # The empty pack: every check above is guarded on non-emptiness, so before 2026-07-30 this one
    # returned [] and `--n 0` wrote a 0-sample pack over models/ and exited 0. See check_schema.
    assert check_schema("nmnist", sf[:0], sl[:0]), "an EMPTY pack must be caught, not pass vacuously"
    assert describe(sf[:0], sl[:0])["min"] is None, "describe must survive an empty pack, reporting None"
    print("  schema: 9 negative cases all caught (incl. the empty pack), both kinds accepted")

    # (4) ROUND-TRIP + COMPARISON on a real .npz in a temp dir: identical packs compare byte-exact,
    #     a one-count perturbation does not. This is the mechanism the fingerprint gate rests on.
    tmp = tempfile.mkdtemp()
    try:
        p = write_pack(os.path.join(tmp, "smoke_pack.npz"), sf, sl)
        with np.load(p) as back:
            rt_frames, rt_labels = back["frames"], back["labels"]
        assert np.array_equal(rt_frames, sf) and rt_frames.dtype == np.float32, "npz round-trip"
        assert np.array_equal(rt_labels, sl) and rt_labels.dtype == np.int64, "npz round-trip"
        assert fingerprint(rt_frames, rt_labels) == fingerprint(sf, sl), "fingerprint survives npz"
        assert compare_to_existing("nmnist", sf, sl, p, quiet=True), "identical packs must compare byte-exact"
        perturbed = sf.copy()
        perturbed[2, 3, 1, 5, 5] += 2.0
        assert not compare_to_existing("nmnist", perturbed, sl, p, quiet=True), (
            "a changed count must NOT compare exact"
        )
    finally:
        for fn in os.listdir(tmp):
            os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp)
    print("  round-trip: npz preserves dtype + fingerprint; comparison separates identical from perturbed")

    # (5) The recorded fingerprints, checked against the real artifacts IF they are on disk. Reported
    #     either way and never silently skipped: on a fresh clone models/ is empty by design.
    checked = []
    for kind, spec in SPECS.items():
        path = os.path.join(_ROOT, "models", spec["filename"])
        if not os.path.exists(path):
            print("  recorded crc32 %s: NOT CHECKED (no %s on disk -- expected on a fresh clone)" % (kind, path))
            continue
        with np.load(path) as d:
            disk_frames, disk_labels = d["frames"], d["labels"]
        got = fingerprint(disk_frames, disk_labels)
        assert got == spec["crc32"], "%s on disk is 0x%08x, recorded 0x%08x -- the artifact changed" % (
            kind,
            got,
            spec["crc32"],
        )
        assert check_schema(kind, disk_frames, disk_labels) == [], "%s on disk violates the schema" % kind
        checked.append(kind)
        print("  recorded crc32 %s: MATCHES the pack on disk (0x%08x) and it satisfies the schema" % (kind, got))
    print("  fingerprint constants verified against %d/%d packs present on this machine" % (len(checked), len(SPECS)))

    print("SMOKE OK: selection is pinned to the rule that reproduces both committed packs; the schema")
    print("check catches 9 ways a wrong pack still loads; the fingerprint gate separates a byte-exact")
    print("rebuild from a re-binned one. Nothing here touched tonic, a dataset, or the network.")


def build_parser():
    p = argparse.ArgumentParser(
        description="Produce models/nmnist_pack.npz and models/shd_pack.npz (the perception packs "
        "cortex/brain.py, cortex/percept_surprise.py and reasoning/reservoir_eprop.py read)."
    )
    p.add_argument("--smoke", action="store_true", help="deterministic self-check (no tonic/dataset/network)")
    p.add_argument("--which", choices=["both", "nmnist", "shd"], default="both", help="which pack(s) to build")
    p.add_argument("--out", default=os.path.join(_ROOT, "models"), help="directory to write the pack(s) into")
    p.add_argument("--data", default=os.path.join(_ROOT, "data"), help="dataset root (downloaded into if absent)")
    p.add_argument("--compare-to", default=None, dest="compare_to", help="directory holding existing pack(s) to compare against")
    p.add_argument("--n", type=int, default=None, help="samples per pack (default: the canonical 256)")
    p.add_argument("--seed", type=int, default=None, help="selection seed (default: the canonical 0)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run", help="build and compare, write nothing")
    p.add_argument(
        "--allow-drift",
        action="store_true",
        dest="allow_drift",
        help="write even if the fingerprint does not match the recorded one (changes every perception number)",
    )
    return p


def main():
    args = build_parser().parse_args()
    if args.smoke:
        run_smoke()
        return
    sys.exit(0 if run_build(args) else 1)


if __name__ == "__main__":
    main()
