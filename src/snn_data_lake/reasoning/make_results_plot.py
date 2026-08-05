"""
Render assets/results.svg -- the per-version results bar chart. No matplotlib: a hand-laid SVG.

  python -m snn_data_lake.reasoning.make_results_plot        # (also run by run_selfchecks.py)

THREE DEFECTS WERE FIXED HERE ON 2026-08-04; each is recorded because each was invisible to the
gate that was supposedly running this file.

1. EVERYTHING RAN AT MODULE SCOPE. There was no `main()` and no `if __name__ == "__main__"` guard,
   so `import snn_data_lake.reasoning.make_results_plot` -- from anywhere, for any reason --
   rendered the chart and WROTE A FILE as a side effect of the import. The write target was
   `Path("assets/results.svg")`, relative to the CURRENT WORKING DIRECTORY, with `mkdir` on its
   parent, so an import from any other directory silently created a stray `assets/` tree there.
   This is the module-scope-side-effect class P4c records against `learn_daemon.py` (`os.chdir` /
   `sys.path.insert` / `os.environ.setdefault` at import), and `run_selfchecks.py` runs part of the
   suite in ONE process. Now: import is inert, and the output path is derived from THIS FILE's
   location, so it lands in the repo's `assets/` no matter where the interpreter was started.

2. IT WAS A SELF-CHECK THAT COULD NOT FAIL. This module is registered in `run_selfchecks.py` and
   contained ZERO asserts: it "passed" by not raising, and counted toward the suite total while
   testing nothing. Per P8 a measurement that cannot go red is not a measurement. `main()` now
   carries five criteria, each with a red constructed and run (see the comments at each assert).

3. IT PLOTTED A NUMBER THE PROJECT DOES NOT CLAIM. The green reference line read
   "95% -- 6400 neurons, GPU (v0.9)" on a chart titled "results by version", i.e. it attributed
   95% to an SNDL version. `docs/OVERVIEW.md` says the opposite in as many words: the ~95%
   unsupervised-STDP MNIST figure "is a *literature target* (Diehl & Cook 2015, 6400 neurons +
   full 60k) -- the best measured result here is **90.0%** at 1600 neurons, and the tuned
   6400-neuron run has not yet been executed at that scale". The ACC row `("GPU", 90.0,
   "1600n tuned")` below is that real measurement and agrees with OVERVIEW. The line is now
   labelled as the external literature target it is, and is drawn in the dashed style the chance
   line uses so it reads as a reference rather than a result. The brain may plot what it measures.

SCOPE, stated because the numbers below are a hand-maintained table: this file RENDERS a recorded
history, it does not RE-MEASURE it. Every value is a historical result transcribed from the
changelog, and re-running this module cannot revalidate any of them -- the asserts check the
rendering, never the science. Adding a version means editing ACC/COMP by hand.

P3/P4c DISPOSITION OF THE OUTPUT, measured 2026-08-05 and stated because nothing else in this file
says it: `assets/` is GITIGNORED (`.gitignore:78` -> `/assets/`) and NOTHING reads
`assets/results.svg` -- the header of this module said "the results plot embedded in the README"
and `README.md` does not reference it; the only tree-wide mention is a historical `CHANGELOG.md`
entry. So this self-check renders, checks and writes an artifact with zero consumers. That is a
consumer gap (P3 shape 2: the return value goes nowhere), not a rendering bug, and the two honest
dispositions are to embed the chart where it is read or to retire the module. Neither is taken
here: both are decisions about README/CHANGELOG, outside this file. Recorded so it is countable.
"""

import os
import re
from pathlib import Path

# (version, test accuracy %, dataset) -- classifier versions only.
#
# FOUND BY THE NEW ASSERT (5), 2026-08-04, and worth stating rather than silently working around:
# this panel plots MORE THAN ONE TASK on one axis. Every row but the first is MNIST-family (plain
# MNIST, the tuned run, the BindsNET comparison, the two latency-coded runs, the 1600-neuron GPU
# run), while `v0.2` is a 4-shape toy -- and 4 shapes at 100% is drawn under a 10-class MNIST
# chance line and beside an MNIST literature anchor, where it reads as the best result on the
# chart. The dataset tag under each bar is the only thing distinguishing them. NOT silently
# rescoped or dropped: the row is a real recorded result and removing it would be the silent
# weakening this audit exists to catch. `_MNIST_FAMILY` below names the comparable subset so the
# anchor check compares like with like, and the mixing is now visible here instead of implicit.
ACC = [
    ("v0.2", 100.0, "4-shape"),
    ("v0.3", 74.6, "MNIST"),
    ("v0.5", 81.5, "MNIST"),
    ("v0.6", 74.4, "MNIST"),
    ("v0.8", 82.3, "tuned"),
    ("v0.9", 76.0, "BindsNET*"),
    ("v0.16", 68.8, "latency"),
    ("v0.17", 76.0, "latency+"),
    ("GPU", 90.0, "1600n tuned"),
]
# (version, compute reduction x vs dense baseline)
COMP = [("v0.3", 23.5), ("v0.5", 23.6), ("v0.4", 4.0), ("v0.6", 3.0)]

# The MNIST-family subset: the rows the MNIST chance line and the MNIST literature anchor are
# actually comparable to. "4-shape" is a different task and a different class count.
_MNIST_FAMILY = frozenset({"MNIST", "tuned", "BindsNET*", "latency", "latency+", "1600n tuned"})

# The external literature anchor, labelled as one. NOT an SNDL result -- see defect 3 above.
LIT_TARGET_PCT = 95.0
# NOTE the "and": a bare "&" is an invalid XML token, and writing "Diehl & Cook" here produced
# markup that assert (4) below rejected on the first run -- which is the assert earning its place
# immediately. The original file had no well-formedness check and would have shipped the broken
# document silently.
LIT_TARGET_LABEL = "95% literature target (Diehl and Cook 2015, 6400n) -- NOT measured here"
CHANCE_PCT = 10.0  # 10-class MNIST

W, H = 780, 400
PLOT_TOP, PLOT_BOT = 70, 320
PLOT_H = PLOT_BOT - PLOT_TOP
ACC_COL, COMP_COL, GRID, TXT = "#2a9d8f", "#e9a23b", "#d9d9d9", "#333333"

# repo root = .../src/snn_data_lake/reasoning/<this file> -> up three. Derived, never CWD-relative,
# so the render target does not depend on where the interpreter was started (defect 1).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "assets" / "results.svg"


def bars(items, x0, panel_w, vmax, color, fmt):
    n = len(items)
    slot = panel_w / n
    bw = slot * 0.56
    out = []
    for i, (ver, val, *rest) in enumerate(items):
        h = (val / vmax) * PLOT_H
        x = x0 + i * slot + (slot - bw) / 2
        y = PLOT_BOT - h
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>'
        )
        out.append(
            f'<text x="{x + bw / 2:.1f}" y="{y - 6:.1f}" font-size="13" font-weight="bold" '
            f'text-anchor="middle" fill="{TXT}">{fmt(val)}</text>'
        )
        out.append(
            f'<text x="{x + bw / 2:.1f}" y="{PLOT_BOT + 18:.1f}" font-size="13" '
            f'text-anchor="middle" fill="{TXT}">{ver}</text>'
        )
        if rest:
            out.append(
                f'<text x="{x + bw / 2:.1f}" y="{PLOT_BOT + 33:.1f}" font-size="10" '
                f'text-anchor="middle" fill="#888">{rest[0]}</text>'
            )
    return "\n".join(out)


def panel(title, x0, panel_w, axis_label):
    return (
        f'<text x="{x0 + panel_w / 2:.1f}" y="46" font-size="15" font-weight="bold" '
        f'text-anchor="middle" fill="#222">{title}</text>'
        f'<text x="{x0 + panel_w / 2:.1f}" y="{H - 8}" font-size="11" '
        f'text-anchor="middle" fill="#888">{axis_label}</text>'
        f'<line x1="{x0}" y1="{PLOT_BOT}" x2="{x0 + panel_w}" y2="{PLOT_BOT}" stroke="{GRID}" stroke-width="1.5"/>'
    )


def build_svg():
    """The whole chart as one SVG string. Pure -- no I/O, so the asserts can inspect it."""
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    ]
    svg.append(
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="#ffffff" stroke="#e2e2e2"/>'
    )
    svg.append(
        '<text x="20" y="26" font-size="16" font-weight="bold" fill="#111">'
        "Spiking Neural Data Lake -- results by version</text>"
    )

    # Panel A: accuracy
    ax0, aw = 40, 330
    svg.append(panel("Test accuracy (%)", ax0, aw, "classifier versions  (MNIST chance = 10%)"))
    chance_y = PLOT_BOT - (CHANCE_PCT / 100) * PLOT_H
    svg.append(
        f'<line x1="{ax0}" y1="{chance_y:.1f}" x2="{ax0 + aw}" y2="{chance_y:.1f}" stroke="#c0392b" '
        f'stroke-width="1" stroke-dasharray="4 3"/>'
    )
    svg.append(
        f'<text x="{ax0 + aw - 2}" y="{chance_y - 4:.1f}" font-size="9" text-anchor="end" '
        f'fill="#c0392b">chance {CHANCE_PCT:.0f}%</text>'
    )
    tgt_y = PLOT_BOT - (LIT_TARGET_PCT / 100) * PLOT_H
    svg.append(
        f'<line x1="{ax0}" y1="{tgt_y:.1f}" x2="{ax0 + aw}" y2="{tgt_y:.1f}" stroke="#7a7a7a" '
        f'stroke-width="1" stroke-dasharray="5 3"/>'
    )
    svg.append(
        f'<text x="{ax0 + 2}" y="{tgt_y - 4:.1f}" font-size="9" fill="#7a7a7a">{LIT_TARGET_LABEL}</text>'
    )
    svg.append(bars(ACC, ax0, aw, 100.0, ACC_COL, lambda v: f"{v:.1f}"))

    # Panel B: compute reduction
    bx0, bw_ = 420, 330
    svg.append(
        panel("Compute reduction (x)", bx0, bw_, "spiking vs dense baseline  (higher = less power)")
    )
    svg.append(bars(COMP, bx0, bw_, 26.0, COMP_COL, lambda v: f"{v:.0f}x"))

    svg.append("</svg>")
    return "\n".join(svg)


def render(out_path=None):
    """Write the chart. Returns the Path written."""
    out = Path(out_path) if out_path is not None else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(), encoding="utf-8")
    return out


def main():
    svg = build_svg()
    print("=" * 72)
    print("MAKE RESULTS PLOT -- hand-laid SVG of the recorded per-version results")
    print("=" * 72)
    print(f"accuracy bars      : {len(ACC)}   compute-reduction bars: {len(COMP)}")
    print(f"best MEASURED acc  : {max(v for _, v, *_ in ACC):.1f}%  (chance {CHANCE_PCT:.0f}%)")
    print(
        f"literature anchor  : {LIT_TARGET_PCT:.0f}% -- drawn as a reference line, not as a result"
    )

    # ---- self-check. This module had ZERO asserts and was registered as a self-check anyway;
    # every criterion below therefore states the input that reddens it, each constructed and run.
    #
    # (1) every recorded version reaches the chart.
    #     RED, CORRECTED 2026-08-05 after the stated one was RUN and did NOT fire. This line said
    #     "RED: drop an entry from ACC". Measured: deleting `("v0.3", 74.6, "MNIST")` from ACC exits
    #     0. It cannot fire -- the loop iterates ACC, so a row removed from ACC is removed from the
    #     check in the same stroke. That is the "assert TRUE BY CONSTRUCTION" shape (a value
    #     compared against the container it was drawn from), and stating an unrun red is how it
    #     survived being written.
    #     RED THAT DOES FIRE (constructed and run 2026-08-05): break the RENDERER instead of the
    #     table -- replace `>{ver}</text>` with `>x</text>` in bars() and this exits 1 with
    #     "version v0.2 is in ACC but not rendered". The criterion is about the render step, so its
    #     red has to be a render defect.
    for ver, val, *_ in ACC:
        assert f">{ver}</text>" in svg, f"version {ver} is in ACC but not rendered"
    for ver, _val in COMP:
        assert f">{ver}</text>" in svg, f"version {ver} is in COMP but not rendered"

    # (2) bar HEIGHT is monotone in the value it encodes -- the one property that makes the chart a
    #     chart rather than decoration.
    #     REWRITTEN 2026-08-05, because what stood here was a TAUTOLOGY WITH A FALSE RED, and both
    #     halves matter. The body recomputed `PLOT_BOT - (val/100)*PLOT_H` in main() and asserted
    #     that a strictly decreasing function of `val` decreases -- true for every possible ACC, for
    #     any PLOT_H > 0, forever. It never looked at what `bars()` emitted. Its stated red ("swap
    #     `PLOT_BOT - h` for `PLOT_BOT + h` in bars()") was RUN and exits 0: the mutation changes the
    #     renderer, and the assert was reading a private copy of the formula. So the module could
    #     have drawn every bar upside down and reported "bar heights monotone in value".
    #     Now it parses the y coordinate out of the ACC panel's own `<rect>` markup, which is the
    #     thing the criterion claims to be about. RED (constructed and run 2026-08-05): the auditor's
    #     original mutation -- `y = PLOT_BOT + h` in bars() -- now exits 1 here.
    acc_rects = re.findall(
        r'<rect [^>]*y="([-\d.]+)"[^>]*fill="%s"/>' % re.escape(ACC_COL),
        bars(ACC, 40, 330, 100.0, ACC_COL, lambda v: f"{v:.1f}"),
    )
    assert len(acc_rects) == len(ACC), (
        f"bars() emitted {len(acc_rects)} accuracy rects for {len(ACC)} rows"
    )
    ys = sorted(zip((v for _ver, v, *_r in ACC), (float(y) for y in acc_rects)))
    for (v1, y1), (v2, y2) in zip(ys, ys[1:]):
        assert y2 <= y1, (
            f"a higher accuracy ({v2}) drew a bar top LOWER on the page than {v1} "
            f"(y {y2} vs {y1}); the chart is inverted"
        )

    # (3) the literature anchor is LABELLED as external and is NOT attributed to a version. This is
    #     the assert that would have caught defect 3. RED (run): restore the old label string
    #     "95% -- 6400 neurons, GPU (v0.9)" and this fires on both clauses.
    assert "literature target" in LIT_TARGET_LABEL and "NOT measured here" in LIT_TARGET_LABEL, (
        "the external anchor must say it is external"
    )
    assert not any(f"({ver})" in LIT_TARGET_LABEL for ver, _v, *_ in ACC), (
        f"the literature anchor must not be attributed to an SNDL version: {LIT_TARGET_LABEL!r}"
    )
    # The anchor is an MNIST number, so it is compared against the MNIST-family rows only. This
    # assert FIRED when first written -- against `max(ACC)` it caught v0.2's 4-shape 100%, which is
    # how the mixed-task defect noted at ACC was found. Scoped, not weakened: the comparison is now
    # like-with-like and the red survives. RED (run): raise the GPU row above 95.0 and it fires.
    mnist_best = max(v for _v, v, ds in ((r[0], r[1], r[2]) for r in ACC) if ds in _MNIST_FAMILY)
    assert LIT_TARGET_PCT > mnist_best, (
        f"the MNIST literature target ({LIT_TARGET_PCT:.0f}%) is supposed to sit ABOVE every "
        f"measured MNIST-family result (best {mnist_best:.1f}%); if a measured result has passed "
        f"it, the anchor is stale and the claim needs re-reading, not re-drawing"
    )
    # every ACC row is either MNIST-family or carries a tag saying it is not (so the mixing above
    # cannot grow silently). RED: add a row with an unrecognised tag and this names it.
    _off = [
        (ver, ds) for ver, _v, ds in ((r[0], r[1], r[2]) for r in ACC) if ds not in _MNIST_FAMILY
    ]
    assert _off == [("v0.2", "4-shape")], (
        f"the set of non-MNIST rows on this MNIST-scaled axis changed: {_off}. Either add the tag "
        f"to _MNIST_FAMILY or give the row its own panel -- do not let the axis mix silently."
    )

    # (4) well-formed, parseable SVG (this file hand-builds markup, so this is a real risk).
    #     RED: delete the trailing "</svg>" append and the parse raises.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg"), f"rendered markup is not an <svg> root: {root.tag}"

    # (5) determinism -- same table, byte-identical output. RED: seed any coordinate from
    #     random/time and the two builds diverge.
    assert build_svg() == svg, "rendering must be deterministic"

    out = render()
    size = out.stat().st_size
    # ASCII-only: the title used an em dash until 2026-08-04, in a file whose output is read back
    # here; keeping the whole document ASCII means no console or editor can mangle it.
    svg.encode("ascii")
    print(
        f"wrote {os.path.relpath(out, _REPO_ROOT)} ({size} bytes, ASCII, {len(svg.splitlines())} lines)"
    )
    print(
        f"self-check OK: {len(ACC) + len(COMP)} recorded results rendered, bar heights monotone in "
        f"value, external {LIT_TARGET_PCT:.0f}% anchor labelled as literature (not as a version "
        f"result), SVG parses, deterministic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
