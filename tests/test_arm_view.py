"""The offscreen arm view: encoder correctness, demand gating, and honest unavailability.

The encoder tests decode the PNG by hand rather than with PIL, and the reason that survives is
the second one: it checks the encoder against the FORMAT rather than against a library. A test
that agrees with PIL proves the two agree, not that either is right.

The clean-install argument that used to sit here was retired 2026-08-04 under P7 -- "it would be
another dependency" is not a valid reason to reject anything, and the purity gate that enforced
it (`test_png_encoder_does_not_use_pil`) is deleted below with its reasoning.
"""

import struct
import threading
import zlib

import pytest

from snn_data_lake.substrate import arm_view as av


def _rows(w, h):
    return [
        b"".join(bytes([(x * 37) % 256, (y * 53) % 256, 11]) for x in range(w)) for y in range(h)
    ]


def _chunks(png):
    out, pos = [], 8
    while pos < len(png):
        (ln,) = struct.unpack(">I", png[pos : pos + 4])
        tag = png[pos + 4 : pos + 8]
        payload = png[pos + 8 : pos + 8 + ln]
        (crc,) = struct.unpack(">I", png[pos + 8 + ln : pos + 12 + ln])
        out.append((tag, payload, crc))
        pos += 12 + ln
    return out


def test_png_encoder_roundtrip():
    """T057: every chunk CRC checks, and the pixels come back byte-identical."""
    w, h = 7, 5
    rows = _rows(w, h)
    png = av.encode_png(rows, w, h)

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    chunks = _chunks(png)
    assert [t for t, _, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"], "one IDAT, in order"

    for tag, payload, crc in chunks:
        assert crc == (zlib.crc32(tag + payload) & 0xFFFFFFFF), f"CRC mismatch on {tag!r}"

    width, height, depth, colour = struct.unpack(">IIBB", chunks[0][1][:10])
    assert (width, height, depth, colour) == (w, h, 8, 2), "header must say 8-bit RGB"

    raw = zlib.decompress(chunks[1][1])
    stride = w * 3
    assert len(raw) == h * (stride + 1), "one filter byte per scanline"
    for y in range(h):
        assert raw[y * (stride + 1)] == 0, "filter type must be 0 (None)"
        assert raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)] == rows[y]


# DELETED 2026-08-04 (specs/010, finding C06): `test_png_encoder_does_not_use_pil`.
#
# It asserted `"PIL" not in imported` and `{"zlib","struct"} <= imported` over arm_view's import
# graph -- a pure-stdlib purity gate, MECHANIZED (ci.yml:111 runs bare pytest), landed 2026-08-01,
# twelve days AFTER the zero-dependency rule was abolished project-wide. P7 (2026-08-04):
# "A rejection that cites dependency count, install weight, or purity is not a rejection; it is a
# reflex, and this policy overrides it."
#
# The available defence -- "this is a packaging gate, core src/ must not import an extra-only
# dependency, which is a STRUCTURAL argument and therefore a permitted P7 rejection" -- was tested
# and fails on three counts. (a) The test keys on the literal string "PIL", never on declaration
# status: it does not read pyproject at all, so promoting pillow to a core dependency would still
# fail it. (b) The second assert pins the IMPLEMENTATION to stdlib primitives, which is not a
# packaging constraint under any reading, and the P2-correct port-then-delete would fail BOTH
# asserts. (c) docs/POLICIES.md already adjudicates this exact file, in P7's incident table -- the
# "hand-written PNG encoder" row, incident #4 of six, with pillow-declared-and-unimported named as
# its counterpart. (That was cited as `POLICIES.md:819` until 2026-08-04; line 819 is a P6
# paragraph about tools/p6_gate.py and has nothing to do with this. Naming the SECTION is what
# stops it drifting again -- citation_guard checks that a cited file exists, never that a cited
# line still says what it said.) A documented design reason cannot exempt a file from P7 when that
# reason IS P7's named incident.
#
# `test_png_encoder_roundtrip` is KEPT: decoding by hand checks the encoder against the FORMAT
# rather than against a library, which is a real reason that survives P7 untouched.


def test_viewer_unavailable_degrades():
    """T058: with no usable renderer the view reports a reason and nothing raises."""
    t = [0.0]
    view = av.ArmView(width=8, height=8, clock=lambda: t[0])
    view.request()

    # Objects that are not a model/data: constructing the renderer must fail INSIDE the view.
    rev = view.render_if_wanted(object(), object())
    assert rev is None, "a failed render must not claim a revision"

    png, reason = view.frame()
    assert png is None
    assert "GL" in reason or "unavailable" in reason, f"unhelpful reason: {reason}"

    st = view.status()
    assert st["arm_view_available"] is False
    assert st["arm_view_revision"] == 0
    assert st["arm_view_reason"], "unavailability must carry a reason, not an empty string"


def test_no_frame_without_demand():
    """A cockpit nobody has open must not cost a render."""
    t = [0.0]
    calls = []

    view = av.ArmView(width=8, height=8, ttl_s=5.0, clock=lambda: t[0])

    class _Model:
        def __getattr__(self, name):  # any touch means a render was attempted
            calls.append(name)
            raise AssertionError("render attempted with no demand")

    assert view.render_if_wanted(_Model(), _Model()) is None
    assert calls == [], "an unwanted view must not touch the model at all"


def test_demand_expires():
    t = [100.0]
    view = av.ArmView(ttl_s=5.0, clock=lambda: t[0])
    assert not view.wanted()
    view.request()
    assert view.wanted()
    t[0] += 5.0
    assert view.wanted(), "demand must survive right up to the ttl"
    t[0] += 0.1
    assert not view.wanted(), "demand must expire after the ttl"


mujoco = pytest.importorskip("mujoco")


def _live():
    from snn_data_lake.brainstem import arm_config

    m = arm_config._model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_revision_advances_per_frame():
    """The counter is the honesty mechanism: it must actually move.

    The clock is FROZEN, and that is the fix for a real flake rather than a convenience.
    This test used the real `time.monotonic` and called `request()` once before two renders,
    so the renders raced `DEMAND_TTL_S` (5.0 s): the FIRST one constructs a `mujoco.Renderer`
    -- first-time offscreen GL -- and if that exceeds the TTL, `wanted()` is already False when
    the second is called and `render_if_wanted` returns None at its first line, the one None
    path that does not latch `_unavailable`. That produced `assert (1, None) == (1, 2)` in
    batteries #132 and #137, green both times when the file ran alone.

    MEASURED, 12 fresh processes timing `render_if_wanted` alone: 11 of them 0.241-0.254 s
    (~4.9% of the TTL), and the FIRST -- a cold process -- 10.44 s, 209% of the TTL, which
    reproduced `r1=1 r2=None` on the spot with nothing else running. So a cold process is
    SUFFICIENT on its own; contention raises the rate, it is not required. (An earlier note
    here said 0.49 s / 9.8%; that was the probe's end-to-end wall clock -- imports 0.15 s plus
    model construction 0.10 s plus the render -- and it under-stated the risk by attributing
    setup cost to the render and by sampling one warm process.)

    Demand expiry is not this test's subject and is not lost: `test_demand_expires` owns it
    deliberately, with its own injected clock and assertions in both directions. Two of this
    file's five `ArmView` constructions used the real clock; both are frozen now, this one and
    `test_renderer_refuses_a_foreign_thread`, whose ownership assertion the same expiry made
    pass VACUOUSLY.

    Freezing the clock removes the timing dependence WITHOUT weakening the assertion: the
    counter is still asserted hard, verified by pinning `_revision` in memory, which reds this
    with `assert (1, 1) == (1, 2)`. Extending the r1 skip to r2 was the other candidate and is
    rejected -- but NOT for the reason first recorded here. A stuck counter returns 1, not
    None, so a skip would never have hidden one. It is rejected because it would absorb a
    genuine second-frame failure into a green skip, and because it would leave the wall-clock
    dependency in place while hiding its only symptom.
    """
    m, d = _live()
    view = av.ArmView(width=64, height=48, clock=lambda: 0.0)
    view.request()
    r1 = view.render_if_wanted(m, d)
    if r1 is None:  # environment cannot draw; the degradation path is tested above
        pytest.skip(f"no offscreen rendering here: {view.frame()[1]}")
    r2 = view.render_if_wanted(m, d)
    assert (r1, r2) == (1, 2), "the revision must advance once per rendered frame"

    png, rev = view.frame()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and rev == 2
    assert _chunks(png)[0][1][:8] == struct.pack(">II", 64, 48), "frame size must match the request"
    view.close()


def test_renderer_refuses_a_foreign_thread():
    """A GL context belongs to its creating thread. Another thread must be refused, not served.

    Rendering on the wrong thread does not raise -- it returns a garbage frame. That is why
    this is an explicit ownership check and not a try/except, and why a lock would not help.

    The clock is frozen for the same reason as in `test_revision_advances_per_frame`, and here
    the consequence was worse. This test has the identical shape -- `request()` once, then a
    render that builds the `mujoco.Renderer`, then a second render -- so an expired
    `DEMAND_TTL_S` makes `render_if_wanted` return None at its first line. The foreign thread's
    render is then refused by the DEMAND GATE rather than by the ownership check, and this
    test's primary assertion passes for the wrong reason: it would report a working ownership
    check on a build where ownership was never consulted. Only the reason assertion below
    catches it, and it catches it as an empty string rather than as anything attributable.
    """
    m, d = _live()
    view = av.ArmView(width=32, height=24, clock=lambda: 0.0)
    view.request()
    if view.render_if_wanted(m, d) is None:
        pytest.skip("no offscreen rendering here")

    result = {}

    def other():
        result["rev"] = view.render_if_wanted(m, d)

    th = threading.Thread(target=other)
    th.start()
    th.join()

    assert result["rev"] is None, "a foreign thread was allowed to render"
    assert view.frame()[0] is not None, "the frame already drawn must survive the refusal"
    assert av.UNAVAILABLE_WRONG_THREAD in view.status()["arm_view_reason"]
    view.close()


def test_the_body_owns_a_view():
    """The cockpit route reaches the picture through the arm, so the arm must expose one."""
    from snn_data_lake.brainstem.so101_arm import So101Arm

    arm = So101Arm()
    assert isinstance(arm.view, av.ArmView)
    assert arm.view is arm.view, "the view must be created once, not per access"
    assert arm.view.frame()[0] is None, "an unrequested view must not have a frame"
