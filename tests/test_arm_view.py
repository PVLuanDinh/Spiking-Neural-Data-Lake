"""The offscreen arm view: encoder correctness, demand gating, and honest unavailability.

The encoder tests decode the PNG by hand rather than with PIL. That is the point: PIL is
importable in `.venv-cu128` but declared only under the `[hf]` extra, so a test that used it
would pass here and fail on a clean install -- and it would also be checking the encoder
against a library rather than against the format.
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


def test_png_encoder_does_not_use_pil():
    """The encoder must not have reached for an undeclared dependency.

    Checked over the IMPORT GRAPH, not the text. A substring search hits the module docstring,
    which names PIL on purpose to record why it is not used -- so the naive version of this
    test fails on the very comment that documents the rule it enforces.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(av.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "PIL" not in imported, f"arm_view imports PIL: {sorted(imported)}"
    assert {"zlib", "struct"} <= imported, "the encoder must use the stdlib primitives"


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
    """The counter is the honesty mechanism: it must actually move."""
    m, d = _live()
    view = av.ArmView(width=64, height=48)
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
    """
    m, d = _live()
    view = av.ArmView(width=32, height=24)
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
