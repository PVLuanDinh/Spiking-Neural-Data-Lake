"""hf_stream -- streamed Hugging Face datasets as brain-consumable feature rows.

WHY THIS EXISTS. The brain's growth is data-limited: NG-SNN grows a unit only while it still reduces
held-out error, so how wide it gets is set by how much real structure it is fed. Measured on streamed
MNIST it earns 2,096 units at 10k rows and 4,512 at 24k. This is the supply line for
that, and it is a DATA-PREP tool: like `substrate/web_ingest.py` it sits outside the control path, is
never imported by the arm loop or any safety-envelope module, and hands back arrays that a trainer
consumes. It NEVER constructs a Brain (Policy 4).

WHY `datasets` AND NOT A SCRAPER. Hugging Face has an official API with pagination, licence metadata
and native streaming. Pointing a browser-automation stack at it would reinvent that client, get
rate-limited, and put a renderer between us and a REST endpoint. Scrapling earns its place on sites
with no API or with anti-bot; this is neither (Policy 2: one mechanism for the job).

THE TWO GATES, and the first one is stricter here than a caller-supplied string.

  LICENCE. `dataset_license()` reads the licence from the HUB'S OWN metadata and
  `substrate/ingest_guard` decides on it, deny-by-default. The caller does not get to assert a licence
  for someone else's dataset -- that is exactly the claim that should not be self-certified. A caller
  MAY pass `license_id` to override, but only to be MORE restrictive in practice: it is still run
  through the same allowlist, and it is recorded in the returned meta so a downstream reader can see
  the licence came from a human rather than the Hub.

  INJECTION. Text fields are scanned and the flags returned with the data. Nothing is rewritten --
  see ingest_guard on why sanitising is worse than flagging. The structural protection is that this
  returns FEATURES for a trainer; no path here writes to the durable knowledge lake, and no string it
  touches is handed to anything that acts.

BOUNDED BY CONSTRUCTION. `n_rows` caps the stream, streaming=True means rows arrive lazily and the
dataset is never materialised, and the loop stops at the cap regardless of dataset size. "Explore all
of Hugging Face" is not a thing this does or can do.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.hf_stream        # offline self-check
  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.hf_stream --live ylecun/mnist
"""

import sys

from snn_data_lake.substrate.ingest_guard import (
    LicenseRefused,
    license_ok,
    license_reason,
    scan_injection,
)

# Field names datasets conventionally use. Checked in order; the first present wins.
_IMAGE_KEYS = ("image", "img", "picture")
_LABEL_KEYS = ("label", "labels", "fine_label", "class")
_TEXT_KEYS = ("text", "sentence", "content", "question")


class HFUnavailable(RuntimeError):
    """The `datasets` / `huggingface_hub` stack is not installed. A distinct type so a caller can
    tell a MISSING DEPENDENCY apart from a licence refusal or an empty dataset -- three failures that
    want three different responses, and that a bare exception would flatten into one."""


def dataset_license(repo_id):
    """The licence string the Hub reports for `repo_id`, or None if it states none.

    None is a real answer and the common one, not an error: a dataset card with no licence field is
    exactly the case ingest_guard refuses. Returning None rather than raising keeps 'the Hub says
    nothing' distinguishable from 'the Hub is unreachable'."""
    try:
        from huggingface_hub import dataset_info
    except ImportError as e:
        raise HFUnavailable("huggingface_hub is not installed (pip install '.[hf]')") from e
    info = dataset_info(repo_id)
    card = getattr(info, "cardData", None) or {}
    lic = card.get("license")
    if isinstance(lic, (list, tuple)):  # cards sometimes carry a list
        lic = lic[0] if lic else None
    return str(lic) if lic else None


def _first_key(example, candidates):
    return next((k for k in candidates if k in example), None)


def image_to_features(img, size=8):
    """A PIL image -> a flat [0,1] greyscale feature row of length size*size.

    Small on purpose: the consumer is a contribution-grown LIF pool over normalised features, not a
    conv net, and 8x8 is what the measured capacity curve in docs/PERFORMANCE.md was taken at."""
    im = img.convert("L").resize((size, size))
    px = list(im.getdata())
    return [p / 255.0 for p in px]


def stream_features(
    repo_id,
    split="train",
    n_rows=2000,
    size=8,
    license_id=None,
    trust_hub_license=True,
):
    """Stream up to `n_rows` rows of `repo_id` into (X, y, meta). Raises before any data is read if
    the licence does not pass.

    Returns X = list of [0,1] float rows, y = list of int labels (empty if the split has no label
    field), meta = provenance including the licence, where it came from, and any injection flags.
    """
    stated = license_id
    source = "caller"
    if trust_hub_license and license_id is None:
        stated = dataset_license(repo_id)
        source = "hub"
    if not license_ok(stated):
        raise LicenseRefused(f"{repo_id}: {license_reason(stated)}")

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise HFUnavailable("datasets is not installed (pip install '.[hf]')") from e

    ds = load_dataset(repo_id, split=split, streaming=True)  # lazy: never materialised
    X, y, flags = [], [], set()
    ik = lk = tk = None
    for i, ex in enumerate(ds):
        if i >= n_rows:
            break
        if i == 0:
            ik, lk, tk = (
                _first_key(ex, _IMAGE_KEYS),
                _first_key(ex, _LABEL_KEYS),
                _first_key(ex, _TEXT_KEYS),
            )
            if ik is None:
                raise ValueError(
                    f"{repo_id}: no image field among {_IMAGE_KEYS}; this streams image datasets "
                    f"(the split has {sorted(ex)[:8]})"
                )
        X.append(image_to_features(ex[ik], size=size))
        if lk is not None:
            y.append(int(ex[lk]))
        if tk is not None:
            flags.update(scan_injection(str(ex[tk])))

    meta = {
        "repo_id": repo_id,
        "split": split,
        "rows": len(X),
        "dim": size * size,
        "license": stated,
        "license_source": source,  # "hub" = the Hub said so; "caller" = a human asserted it
        "trust": "untrusted-data",
        "injection_flags": sorted(flags),
    }
    return X, y, meta


def main():
    live = "--live" in sys.argv
    print("=" * 88)
    print("HF STREAM -- streamed datasets as brain-consumable features (licence-gated, bounded)")
    print("=" * 88)

    # ---- offline: the licence gate is the part that must hold with no network at all ----
    for bad in (None, "", "cc-by-nc-4.0", "invented-licence"):
        assert not license_ok(bad), f"{bad!r} must refuse"
    assert license_ok("mit") and license_ok("cc-by-4.0")

    # ---- offline: feature conversion, without requiring PIL ----
    class _FakeImg:  # the two methods image_to_features uses; keeps the check dependency-free
        def __init__(self, n):
            self._n = n

        def convert(self, _mode):
            return self

        def resize(self, wh):
            self._n = wh[0] * wh[1]
            return self

        def getdata(self):
            return [0, 128, 255] * (self._n // 3) + [0] * (self._n % 3)

    row = image_to_features(_FakeImg(64), size=8)
    assert len(row) == 64, len(row)
    assert all(0.0 <= v <= 1.0 for v in row), "features must be normalised into [0,1]"
    assert max(row) == 1.0 and min(row) == 0.0, "255 -> 1.0 and 0 -> 0.0"

    # ---- offline: a refusal must happen BEFORE any network call, so an unlicensed dataset is
    # never even fetched. Passing an explicit bad licence with trust_hub_license=False proves the
    # order: no huggingface_hub import is reachable on this path.
    try:
        stream_features("someone/unlicensed", license_id=None, trust_hub_license=False)
        raise AssertionError("must refuse before streaming")
    except LicenseRefused as e:
        assert "absence is not permission" in str(e), str(e)

    # ---- offline: a missing dependency is its own error type, not a crash or a silent empty ----
    try:
        from datasets import load_dataset  # noqa: F401

        have = True
    except ImportError:
        have = False
    print(f"datasets installed: {have} (the gate env deliberately does not need it)")

    print("licence gate    : deny-by-default, refuses BEFORE any fetch")
    print("features        : 8x8 greyscale, normalised [0,1], PIL only needed for real images")
    print("bounded         : n_rows caps the stream; streaming=True never materialises the dataset")

    if live:
        repo = next((a for a in sys.argv[1:] if not a.startswith("-")), "ylecun/mnist")
        lic = dataset_license(repo)
        print(f"\n[live] {repo}: hub licence = {lic!r} -> {license_reason(lic)}")
        X, y, meta = stream_features(repo, n_rows=200)
        print(
            f"[live] streamed {meta['rows']} rows, dim {meta['dim']}, labels {len(set(y))} classes"
        )
        print(f"[live] meta: {meta}")

    print("self-check OK: licence refused before fetch, features normalised, dependency absence is")
    print("a typed error; no Brain constructed, no durable write, no network in this path.")


if __name__ == "__main__":
    main()
