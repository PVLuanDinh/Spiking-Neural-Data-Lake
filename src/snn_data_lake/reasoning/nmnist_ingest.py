"""
v0.31 — N-MNIST (real event-camera) ingestion.

Closes the one real gap in Gemini's improvement brief: the repo trained only on
static MNIST. N-MNIST is MNIST recorded by a Dynamic Vision Sensor — async
(x, y, t, polarity) spike events on a 34x34x2 grid. This is *native* spiking
data, not a rate/latency conversion of pixels.

Pipeline (the data-lake leg, end to end):
  events (x,y,t,p)  ->  BRONZE raw event store (gzip)        -> storage size
                    ->  bin into T frames (Tonic ToFrame analog, stdlib)
                    ->  SILVER spike raster  -> ICR (gzip lesion metric, reused
                        from lakehouse/medallion.py)          -> storage reduction
                    ->  GOLD firing-rate vector (sparse)      -> compute reduction
                    ->  nearest-prototype classify            -> accuracy

Real data path: `pip install tonic` then run — pulls torchvision-style N-MNIST
(~1 GB on first run) via tonic.datasets.NMNIST. WITHOUT tonic the script falls
back to deterministic synthetic events so the whole pipeline + self-checks run
without tonic (e.g. in CI). The synthetic blobs are class-separable by construction;
they prove the plumbing, NOT a benchmark number — install tonic for that.

  python nmnist_ingest.py
  NMNIST_TRAIN=400 NMNIST_TEST=200 NMNIST_T=8 python nmnist_ingest.py
"""

import os
import gzip
import random
import struct

H = W = 34  # N-MNIST sensor is 34x34
P = 2  # two polarities (on/off)
C = 10  # digit classes
DUR = 300_000  # ~300 ms in microseconds (N-MNIST sample length)

TRAIN_N = int(os.environ.get("NMNIST_TRAIN", 200))
TEST_N = int(os.environ.get("NMNIST_TEST", 100))
T = int(os.environ.get("NMNIST_T", 6))  # time bins (ToFrame n_time_bins)
SEED = int(os.environ.get("NMNIST_SEED", 0))


def _say(msg):
    """print(), with the project's ASCII-only output invariant asserted at the point of emission.

    This module printed a U+2014 EM DASH for the life of the feature and no gate saw it: MEASURED
    2026-08-04, `PYTHONIOENCODING=ascii python -m snn_data_lake.reasoning.nmnist_ingest` exited 1 with
    `UnicodeEncodeError: 'ascii' codec can't encode character '\\u2014'`. Under the default Windows
    console cp1252 an em dash happens to encode, which is exactly why it survived -- the invariant
    is ASCII, not "whatever this console tolerates". Reachable red: put a non-ASCII character in any
    _say() below and this raises here, naming the line. Same shape as nightly_skills.main::say and
    rose_language.main::say, which is where the pattern comes from."""
    msg.encode("ascii")
    print(msg)


# ---- data source: real N-MNIST via Tonic, else synthetic events -------------
def _synth_sample(label, rng, n_events=600):
    """Deterministic class-separable event blob: each digit fires around a unique
    grid centroid, drifting over time. Synthetic stand-in — proves the
    ingest->feature->classify path; swap for tonic.datasets.NMNIST for real data."""
    cx = 4 + (label % 5) * 6  # 5 columns of centroids
    cy = 8 + (label // 5) * 16  # 2 rows
    ev = []
    for _ in range(n_events):
        t = rng.randrange(DUR)
        drift = int((t / DUR) * 6) - 3  # blob drifts with time
        x = min(W - 1, max(0, int(rng.gauss(cx + drift, 2.2))))
        y = min(H - 1, max(0, int(rng.gauss(cy, 2.2))))
        ev.append((x, y, t, rng.randint(0, 1)))
    return ev


def load_events(n_train, n_test):
    """Return (train, test) lists of (events, label); 'source' tag for honesty."""
    try:
        import tonic  # noqa

        ds_tr = tonic.datasets.NMNIST(save_to="./data", train=True)
        ds_te = tonic.datasets.NMNIST(save_to="./data", train=False)

        def take(ds, k):
            # N-MNIST is stored class-sorted (all 0s, then all 1s, ...). Taking the
            # first k would yield ONE class -> a degenerate 100%. Sample spread
            # indices (deterministic) so all 10 digits are represented.
            rng = random.Random(SEED)
            idxs = rng.sample(range(len(ds)), min(k, len(ds)))
            out = []
            for i in idxs:
                events, label = ds[i]
                # tonic events are a structured array with x,y,t,p fields
                evs = [(int(e["x"]), int(e["y"]), int(e["t"]), int(e["p"])) for e in events]
                out.append((evs, int(label)))
            return out

        return take(ds_tr, n_train), take(ds_te, n_test), "tonic/N-MNIST (real DVS)"
    except Exception as exc:  # ImportError or download failure -> synthetic
        rng = random.Random(SEED)
        tr = [(_synth_sample(i % C, rng), i % C) for i in range(n_train)]
        te = [(_synth_sample(i % C, rng), i % C) for i in range(n_test)]
        reason = "tonic missing" if isinstance(exc, ImportError) else f"tonic error: {exc}"
        return tr, te, f"synthetic events ({reason} -- `pip install tonic` for real N-MNIST)"


# ---- BRONZE: raw immutable events -> compact gzip store ---------------------
def bronze_bytes(samples):
    """Serialize every event as 4x uint16/uint32 (the raw 'data at rest')."""
    buf = bytearray()
    for evs, _ in samples:
        for x, y, t, p in evs:
            buf += struct.pack("<HHIB", x, y, t, p)
    return bytes(buf)


# ---- SILVER: bin events into T frames (Tonic ToFrame analog) ----------------
def to_frames(events):
    """T frames of P*H*W counts. t normalized into [0,T) by DUR."""
    frames = [bytearray(P * H * W) for _ in range(T)]
    for x, y, t, p in events:
        b = min(T - 1, (t * T) // DUR)
        idx = (p % P) * (H * W) + y * W + x
        if frames[b][idx] < 255:
            frames[b][idx] += 1
    return frames


def raster_bytes(frames):
    out = bytearray()
    for f in frames:
        out += f
    return bytes(out)


def icr(raw):
    """ICR = compressed/uncompressed. Lower = more structured (reused metric)."""
    return len(gzip.compress(raw, 9)) / max(1, len(raw))


# ---- GOLD: sparse firing-rate feature vector -------------------------------
def rate_vector(frames):
    """Sum over time -> per-(polarity,pixel) firing rate. Sparse = compute-cheap."""
    vec = [0.0] * (P * H * W)
    for f in frames:
        for i, v in enumerate(f):
            vec[i] += v
    return [v / T for v in vec]


# ---- classify: nearest class-prototype on rate vectors ---------------------
def prototypes(train_feats):
    sums = {c: [0.0] * (P * H * W) for c in range(C)}
    cnt = {c: 0 for c in range(C)}
    for vec, label in train_feats:
        cnt[label] += 1
        s = sums[label]
        for i, v in enumerate(vec):
            s[i] += v
    return {c: [v / cnt[c] for v in sums[c]] for c in range(C) if cnt[c]}


def _dist2(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def classify(protos, vec):
    return min(protos, key=lambda c: _dist2(protos[c], vec))


def main():
    train, test, source = load_events(TRAIN_N, TEST_N)
    n_ev = sum(len(e) for e, _ in train) + sum(len(e) for e, _ in test)

    raw = bronze_bytes(train + test)
    raw_gz = len(gzip.compress(raw, 9))

    train_feats = [(rate_vector(to_frames(e)), lab) for e, lab in train]
    # one shared raster (first test sample) for the ICR storage metric
    sample_raster = raster_bytes(to_frames(test[0][0]))
    sample_icr = icr(sample_raster)

    protos = prototypes(train_feats)
    correct = 0
    for evs, lab in test:
        if classify(protos, rate_vector(to_frames(evs))) == lab:
            correct += 1
    acc = correct / len(test)

    sparsity = sum(1 for v in train_feats[0][0] if v > 0) / len(train_feats[0][0])

    _say("=" * 62)
    _say("N-MNIST EVENT INGESTION  (events -> Bronze -> raster -> classify)")
    _say("=" * 62)
    _say(f"source        : {source}")
    _say(f"samples       : {len(train)} train / {len(test)} test, {n_ev:,} events")
    _say(
        f"BRONZE store  : {len(raw) / 1024:.1f} KB raw  -> {raw_gz / 1024:.1f} KB gzip "
        f"({len(raw) / max(1, raw_gz):.1f}x)"
    )
    _say(f"SILVER raster : {T}x{P}x{H}x{W}  ICR={sample_icr:.3f}  (lower = more structured)")
    _say(f"GOLD feature  : {P * H * W}-d rate vector, {sparsity * 100:.0f}% active (sparse=cheap)")
    _say(f"ACCURACY      : {acc * 100:.1f}%   (chance = {100 / C:.0f}%)")
    _say("=" * 62)

    # ---- self-checks --------------------------------------------------------
    assert n_ev > 0, "no events ingested"
    assert len(raster_bytes(to_frames(test[0][0]))) == T * P * H * W, "raster shape wrong"
    assert 0.0 < sample_icr < 1.0, f"ICR out of range: {sample_icr}"
    # ROUND-TRIP: decode the Bronze byte store back to events and compare against the source.
    #
    # What stood here was `len(raw) // struct.calcsize("<HHIB") == n_ev`, and it was TRUE BY
    # CONSTRUCTION: `bronze_bytes` packs exactly one fixed-width 9-byte record per event and
    # `n_ev` counts the same list, so len(raw) is 9*n_ev for every possible input -- there is no
    # constructible red. Worse, the success line beside it printed "Bronze roundtrip intact"
    # while NOTHING in this module ever decoded a byte back. That is a claim the measurement did
    # not support, which is the thing P8 forbids, and deleting the assert (F11's default) would
    # have left the false sentence standing. So the criterion is made REAL instead: unpack the
    # store and compare the recovered events to the originals.
    #
    # REACHABLE RED, constructed and run 2026-08-04. The red has to be chosen carefully, and the
    # first one tried was NOT discriminating -- recording that because it is the whole point:
    #   * narrowing the record to "<HHHB" fires this assert (exit 1), but the OLD assert catches
    #     it too (7-byte records against a 9-byte calcsize changes the count). Not evidence.
    #   * transposing two fields of the SAME width -- pack(..., y, x, t, p) -- is the real red.
    #     len(raw) is unchanged, so the old criterion reads True and the run stayed GREEN with a
    #     systematically corrupted store; this assert exits 1. Both measured.
    # That second case is the silent corruption "Bronze store lost events" always claimed to be
    # about and never tested: a store that is the right SIZE and the wrong CONTENT.
    rec = struct.calcsize("<HHIB")
    assert len(raw) % rec == 0, (
        f"Bronze store is not a whole number of records ({len(raw)} % {rec})"
    )
    decoded = [struct.unpack_from("<HHIB", raw, i * rec) for i in range(len(raw) // rec)]
    source_events = [ev for evs, _ in (train + test) for ev in evs]
    assert decoded == source_events, (
        f"Bronze store did not round-trip: {sum(1 for a, b in zip(decoded, source_events) if a != b)} "
        f"of {len(source_events)} events differ after decode"
    )
    assert acc >= 0.30, f"accuracy {acc:.2f} below 0.30 (pipeline broken?)"
    _say(
        f"self-check OK: {n_ev:,} events ingested, raster + ICR valid, "
        f"Bronze roundtrip intact, accuracy {acc * 100:.0f}% > chance"
    )


if __name__ == "__main__":
    main()
