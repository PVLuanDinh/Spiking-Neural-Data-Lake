r"""ar_lm_disjoint -- PROVE the evaluation text is disjoint from the training text before believing any
number measured on it.

This repo has already voided one language run for exactly this failure: a character budget consumed 1.6x
its count in BYTES, swallowed the held-out slice, and the resulting perplexity of 6.2 was the model
reading its own training text. So disjointness is checked EMPIRICALLY here, not asserted from the
reader's design. This is the instrument behind the "Contamination" paragraph of the AR appendix in
docs/DEPTH_CEILINGS.md.

Three claims, three checks:

  (1) INTERVAL   the exposure is bytes [0, E) and the held-out slice is bytes [H, H+n) with E < H.
                 An integer comparison, and training/ar_spiking_lm.py hard-fails when E >= H.
  (2) PROBE      a long verbatim probe taken from the held-out text is NOT findable anywhere in the
                 exposure text. This is the check that caught the voided run.
  (3) BLIMP      no BLiMP sentence that will be scored (good AND bad member of every pair) occurs
                 verbatim in the exposure text. BLiMP is synthetically generated, so this is expected
                 -- but "expected" is not "measured", and a benchmark sentence present in the training
                 text would be memorisation scored as grammar.

PROMOTED into tools/ (2026-07-30) from scratchpad/, which is gitignored (.gitignore:88), so the appendix
was citing a reproduction path absent from every clone -- the defect tools/citation_guard.py now gates.
Both inputs are explicit paths and each is SKIPped with a message when absent.

Usage:
  python tools/ar_lm_disjoint.py [E_BYTES] [--corpus FILE] [--blimp DIR]

  E_BYTES        exposure size in bytes (default 120,000,000 -- the size the appendix reports).
  --corpus FILE  WikiText-103 plain text (default: <repo>/data/lang/wikitext/train0.txt).
  --blimp DIR    BLiMP *.jsonl directory   (default: <repo>/data/lang/blimp).
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "training"))

import ar_spiking_lm as trainer  # noqa: E402

from snn_data_lake.reasoning.blimp_bench import load_pairs, paradigms  # noqa: E402

PER_PARADIGM = 25  # 67 x 25 = 1675 pairs = 3350 sentences -- the documented baseline slice


def pop_opt(argv, name, default):
    """Remove `--name VALUE` from argv in place and return VALUE, else default."""
    if name not in argv:
        return default
    i = argv.index(name)
    del argv[i]
    return argv.pop(i) if i < len(argv) else default


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    corpus = pop_opt(argv, "--corpus", trainer.CORPUS_TXT)
    blimp = pop_opt(argv, "--blimp", None)
    exposure = int(float(argv[0])) if argv else 120_000_000

    # Both inputs are gitignored local-only artifacts, so a clone reaches this SKIP rather than a
    # traceback. trainer.wikitext_prefix() raises SystemExit on a missing corpus, which exits 1 and
    # reads like a failure; an absent dataset is not a failure, it is an absent dataset.
    if not os.path.exists(corpus):
        print(f"SKIP ar_lm_disjoint: no corpus at {corpus}")
        print(
            "  WikiText-103: HF parquet wikitext-103-raw-v1/train-00000-of-00002.parquet -> data/lang/wikitext/"
        )
        return 0
    if not paradigms(blimp):
        print(f"SKIP ar_lm_disjoint: no BLiMP paradigms under {blimp or 'data/lang/blimp'}")
        print(
            "  download: https://github.com/alexwarstadt/blimp/archive/refs/heads/master.zip -> data/lang/blimp/"
        )
        return 0

    print("=" * 100)
    print(f"HELD-OUT / BLiMP DISJOINTNESS PROOF for an exposure of {exposure} bytes")
    print(f"corpus={corpus}")
    print("=" * 100)

    ok = True

    # (1) INTERVAL
    H = trainer.HELDOUT_BYTE_OFFSET
    print("(1) INTERVAL")
    print(f"    exposure  = bytes [0, {exposure})")
    print(f"    held-out  = bytes [{H}, {H + trainer.HELDOUT_BYTES})")
    if exposure < H:
        print(
            f"    DISJOINT: exposure end {exposure} < held-out start {H} (gap {H - exposure} bytes)"
        )
    else:
        print("    CONTAMINATED: exposure reaches the held-out slice")
        ok = False
    # the guard must actually refuse an over-long exposure
    try:
        trainer.wikitext_prefix(H + 1, path=corpus)
        print("    FAIL: the guard did NOT refuse an exposure past the held-out offset")
        ok = False
    except SystemExit:
        print(f"    guard verified: wikitext_prefix({H + 1}) refuses to run")

    t0 = time.time()
    text = trainer.wikitext_prefix(exposure, path=corpus)
    held = trainer.wikitext_heldout(path=corpus)
    print(
        f"    loaded exposure {len(text)} chars, held-out {len(held)} chars ({time.time() - t0:.1f}s)"
    )

    # (2) PROBE -- the voided-run check
    print("(2) PROBE (verbatim held-out text must not appear in the exposure)")
    for plen in (100, 200, 400):
        probe = held[len(held) // 2 : len(held) // 2 + plen]
        hit = text.find(probe)
        where = f"byte {hit} -- CONTAMINATED" if hit >= 0 else "NOWHERE (-1)"
        print(f"    {plen:3d}-char probe from held-out: found in exposure at {where}")
        if hit >= 0:
            ok = False

    # (3) BLIMP
    print("(3) BLIMP (no scored sentence may occur verbatim in the exposure)")
    names = paradigms(blimp)
    sents = []
    for nm in names:
        for g, b in load_pairs(nm, root=blimp, limit=PER_PARADIGM):
            sents.append(g)
            sents.append(b)
    print(f"    {len(names)} paradigms x {PER_PARADIGM} pairs -> {len(sents)} sentences to check")
    t0 = time.time()
    hits = []
    for i, s in enumerate(sents):
        if text.find(s) >= 0:
            hits.append(s)
        if i and i % 500 == 0:
            print(f"      ... {i}/{len(sents)} checked ({time.time() - t0:.0f}s)", flush=True)
    print(
        f"    scanned {len(sents)} sentences against {len(text)} chars in {time.time() - t0:.0f}s"
    )
    if hits:
        print(f"    CONTAMINATED: {len(hits)} BLiMP sentences occur in the training text, e.g.:")
        for s in hits[:5]:
            print(f"      {s[:90]!r}")
        ok = False
    else:
        print(
            f"    DISJOINT: 0 of {len(sents)} BLiMP sentences occur anywhere in the exposure text"
        )

    print("-" * 100)
    verdict = (
        "PROVEN -- numbers measured on this split are honest"
        if ok
        else "FAILED -- do not cite any number"
    )
    print(f"DISJOINTNESS: {verdict}")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
