"""Offscreen picture of the live arm, for the cockpit.

Three things here are load-bearing and none of them are obvious:

1. THE ENCODER IS STDLIB, AND THAT IS NOW A DEFAULT RATHER THAN A RULE. PNG is written with
   `zlib` and `struct` and nothing else. This paragraph used to justify that by dependency
   avoidance -- pillow "would make the cockpit depend on a package the project does not require".
   **Retired 2026-08-04 under P7**, which states that "it would be another dependency" is not a
   valid reason to reject anything, and which names this very encoder as an incident in its
   "Real past incidents" table: a hand-written PNG encoder shipped while `pillow` sits declared
   in `pyproject.toml`'s `[hf]` extra (:94) and imported nowhere. (Both cites were wrong until
   2026-08-05 and were re-resolved against the tree: `POLICIES.md:819` is a BLANK LINE inside the
   P6 section, and `pyproject.toml:84` is a ruff-exclusion comment. Re-verified while correcting
   them -- `pillow`/`PIL` is imported in no file under `src/`, `tests/` or `tools/`, so the CLAIM
   holds and only its coordinates were false.)

   What stands: the encoder WORKS, it is small, and replacing it buys nothing measured yet. What
   is no longer a reason: that pillow is a dependency. Adopting pillow here is P7-legitimate and
   is tracked as specs/010 task T021 -- and per P2 it would mean DELETING this encoder, not
   running both. The purity gate that enforced the old rule
   (`tests/test_arm_view.py::test_png_encoder_does_not_use_pil`) is deleted.

2. THE RENDERER BELONGS TO ONE THREAD. MuJoCo's `Renderer` owns a GL context, and a GL context
   belongs to the thread that made it. Rendering from another thread does not raise: it returns
   a frame, and the frame is garbage. A lock does NOT fix this -- serialising access to a
   context that the calling thread does not own still renders on the wrong context. So the
   owning thread is recorded on first use and a call from anywhere else is refused outright.
   The cockpit's HTTP handler runs on a request thread; it therefore never renders, it only
   reads the frame the daemon's tick thread produced.

3. FRAMES ARE PRODUCED ON DEMAND. Rendering every tick to serve a page nobody has open is pure
   cost. `request()` records interest, and `render_if_wanted()` produces nothing until someone
   has asked recently.

A missing GL context is reported, not raised: the daemon must keep thinking when it cannot see.
"""

import struct
import threading
import time
import zlib

#: How long a viewer request keeps frames flowing, in seconds.
#:
#: Not a tuned constant: it is the demand-gate's memory, and it only has to outlast the gap
#: between a browser's polls. Shorter than a poll interval and the picture stutters; longer and
#: an abandoned page keeps the renderer running. Any value comfortably above the poll period
#: behaves identically, which is what makes this a scale rather than a threshold.
DEMAND_TTL_S = 5.0

#: Frame size. The cockpit shows a thumbnail, not a workspace view.
WIDTH, HEIGHT = 320, 240

UNAVAILABLE_NO_GL = "no GL context: offscreen rendering unavailable in this environment"
UNAVAILABLE_WRONG_THREAD = "refused: the renderer belongs to the tick thread"
UNAVAILABLE_NO_FRAME = "no frame yet: the viewer has not been requested"


def encode_png(pixels, width, height):
    """Encode an HxWx3 uint8 buffer as PNG bytes. Uses stdlib `zlib`/`struct`.

    "Stdlib only" used to be an assertion of policy here; under P7 (2026-08-04) it is merely a
    description of what this happens to use. See the module docstring.


    `pixels` may be a numpy array or any nested sequence; it is consumed row by row through
    `bytes()`, so anything supporting the buffer protocol or yielding ints works.

    Filter type 0 (None) on every row and a single IDAT: the image is a small thumbnail, so
    the bytes saved by adaptive filtering are not worth the code that would have to be right.
    """
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None) for this scanline
        raw.extend(bytes(bytearray(pixels[y])) if not isinstance(pixels[y], bytes) else pixels[y])

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 = RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


class ArmView:
    """Renders the live arm on the tick thread; serves the last frame to anyone."""

    def __init__(self, width=WIDTH, height=HEIGHT, ttl_s=DEMAND_TTL_S, clock=time.monotonic):
        self.width, self.height, self.ttl_s = width, height, ttl_s
        self._clock = clock
        self._renderer = None
        self._owner = None  # thread that owns the GL context, set on first render
        self._png = None
        self._revision = 0
        self._requested_at = None
        self._unavailable = None
        #: Viewpoint, steered from the cockpit. Plain floats written from the request thread and
        #: read on the tick thread: a torn read costs one frame at a slightly wrong angle, which is
        #: not worth a lock on the render path. None until first set, so MuJoCo's default framing
        #: is used until somebody actually drags -- an unrequested camera should not override the
        #: model's own sensible defaults.
        self._cam = None

    # ---- demand ------------------------------------------------------------------------

    def request(self):
        """Record that somebody wants to see the arm. Safe from any thread."""
        self._requested_at = self._clock()

    def look(self, azimuth=None, elevation=None, distance=None):
        """Steer the viewpoint. Safe from any thread; takes effect on the next rendered frame.

        Elevation is clamped to keep the camera off the poles, where azimuth stops meaning
        anything and dragging feels like it has stuck. Distance is clamped positive because a
        non-positive distance puts the eye inside the arm and renders nothing recognisable.
        """
        cam = dict(self._cam or {"azimuth": 90.0, "elevation": -20.0, "distance": 1.6})
        if azimuth is not None:
            cam["azimuth"] = float(azimuth) % 360.0
        if elevation is not None:
            cam["elevation"] = max(-89.0, min(89.0, float(elevation)))
        if distance is not None:
            cam["distance"] = max(0.15, min(12.0, float(distance)))
        self._cam = cam
        return dict(cam)

    def wanted(self):
        return self._requested_at is not None and (self._clock() - self._requested_at) <= self.ttl_s

    # ---- rendering (TICK THREAD ONLY) --------------------------------------------------

    def render_if_wanted(self, model, data):
        """Produce a frame if one has been asked for. Call ONLY from the tick thread.

        Returns the new revision number, or None if nothing was rendered. Never raises: a
        daemon that cannot draw must still think.
        """
        if not self.wanted() or self._unavailable is not None:
            return None

        me = threading.get_ident()
        if self._owner is not None and me != self._owner:
            # Not recoverable and deliberately not locked around -- see the module docstring.
            self._unavailable = UNAVAILABLE_WRONG_THREAD
            return None

        try:
            if self._renderer is None:
                import mujoco

                self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
                self._owner = me
            if self._cam is None:
                self._renderer.update_scene(data)
            else:
                import mujoco

                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, cam)
                c = self._cam  # one read: the dict is replaced wholesale, never mutated in place
                cam.azimuth, cam.elevation, cam.distance = (
                    c["azimuth"],
                    c["elevation"],
                    c["distance"],
                )
                self._renderer.update_scene(data, camera=cam)
            pixels = self._renderer.render()
        except Exception as exc:  # no GL, no display, driver refusal -- all the same to us
            self._unavailable = f"{UNAVAILABLE_NO_GL} ({type(exc).__name__}: {exc})"
            self._renderer = None
            return None

        self._png = encode_png(pixels, self.width, self.height)
        self._revision += 1
        return self._revision

    # ---- reading (ANY THREAD) ----------------------------------------------------------

    def frame(self):
        """(png_bytes, revision) if a frame exists, else (None, reason)."""
        if self._png is not None:
            return self._png, self._revision
        return None, self._unavailable or UNAVAILABLE_NO_FRAME

    def status(self):
        """Panel payload. `revision` is the honesty mechanism: a still picture and a live one
        look identical, so the cockpit shows the counter next to the image."""
        return {
            "arm_view_revision": self._revision,
            "arm_view_available": self._png is not None,
            "arm_view_reason": self._unavailable or ("" if self._png else UNAVAILABLE_NO_FRAME),
        }

    def close(self):
        r, self._renderer = self._renderer, None
        if r is not None:
            try:
                r.close()
            except Exception:
                pass


def main():
    # ---- 1. the encoder is a real PNG, verified without PIL ----------------------------
    w, h = 4, 3
    rows = [
        b"".join(bytes([(x * 60) % 256, (y * 80) % 256, 7]) for x in range(w)) for y in range(h)
    ]
    png = encode_png(rows, w, h)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature missing"

    # Walk the chunks and re-check every CRC the way a decoder would.
    pos, seen = 8, []
    while pos < len(png):
        (ln,) = struct.unpack(">I", png[pos : pos + 4])
        tag = png[pos + 4 : pos + 8]
        payload = png[pos + 8 : pos + 8 + ln]
        (crc,) = struct.unpack(">I", png[pos + 8 + ln : pos + 12 + ln])
        assert crc == (zlib.crc32(tag + payload) & 0xFFFFFFFF), f"bad CRC on {tag!r}"
        seen.append(tag)
        pos += 12 + ln
    assert seen == [b"IHDR", b"IDAT", b"IEND"], f"unexpected chunk layout: {seen}"

    # Round-trip the pixel data: decompress, strip filter bytes, compare to the input.
    idat = png[8 + 12 + 13 :]
    (ln,) = struct.unpack(">I", idat[:4])
    back = zlib.decompress(idat[8 : 8 + ln])
    stride = w * 3
    for y in range(h):
        assert back[y * (stride + 1)] == 0, "filter byte must be 0"
        got = back[y * (stride + 1) + 1 : (y + 1) * (stride + 1)]
        assert got == rows[y], f"row {y} did not round-trip"

    # ---- 2. demand gating ---------------------------------------------------------------
    t = [1000.0]
    view = ArmView(width=8, height=8, ttl_s=5.0, clock=lambda: t[0])
    assert not view.wanted(), "a view nobody asked for must not be wanted"
    view.request()
    assert view.wanted()
    t[0] += 4.0
    assert view.wanted(), "demand must outlast a poll interval"
    t[0] += 2.0
    assert not view.wanted(), "demand must expire"

    # ---- 3. unavailable degrades, it does not raise ------------------------------------
    view.request()
    rev = view.render_if_wanted(object(), object())  # not a model: rendering must fail cleanly
    assert rev is None, "a broken render must not report a revision"
    png2, reason = view.frame()
    assert png2 is None and "GL" in reason, f"unavailability not reported: {reason}"
    assert view.status()["arm_view_available"] is False

    # ---- 4. the real arm, if this environment can draw ----------------------------------
    rendered = None
    try:
        import mujoco

        from snn_data_lake.brainstem import arm_config

        m = arm_config._model()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        live = ArmView(width=64, height=48)
        live.request()
        rendered = live.render_if_wanted(m, d)
        if rendered:
            data, rev2 = live.frame()
            assert data[:8] == b"\x89PNG\r\n\x1a\n" and rev2 == 1
            assert len(data) > 100, "a rendered arm should not compress to nothing"
            r2 = live.render_if_wanted(m, d)
            assert r2 == 2, "the revision counter must advance per frame"
        live.close()
    except Exception:
        rendered = None

    print("=" * 60)
    print("ARM VIEW - offscreen arm picture for the cockpit (stdlib PNG)")
    print("=" * 60)
    print(f"encoder     : {len(png)} bytes for {w}x{h}, chunks {[t.decode() for t in seen]}")
    print("demand gate : expires after ttl; no request -> no frame produced")
    print(
        f"live render : {'OK, revision advanced' if rendered else 'unavailable (reported, not raised)'}"
    )
    print("thread rule : renderer bound to its creating thread; other threads refused")

    # Camera: the cockpit steers this from the request thread, so the clamps are the only thing
    # standing between a dragged mouse and a degenerate viewpoint that renders nothing.
    cv = ArmView()
    assert cv._cam is None, "an untouched view must use MuJoCo's own framing, not ours"
    c = cv.look(azimuth=370.0, elevation=-140.0, distance=-3.0)
    assert c["azimuth"] == 10.0, c  # wrapped, not clamped: azimuth is periodic
    assert c["elevation"] == -89.0, c  # clamped off the pole, where azimuth stops meaning anything
    assert c["distance"] == 0.15, c  # clamped positive: the eye must not sit inside the arm
    c2 = cv.look(azimuth=45.0)
    assert (c2["elevation"], c2["distance"]) == (-89.0, 0.15), "a partial look must not reset axes"
    print(
        f"camera      : wrap+clamp OK, az/el/d = {c2['azimuth']}/{c2['elevation']}/{c2['distance']}"
    )
    print("=" * 60)
    print(
        "self-check OK: PNG chunk CRCs verified and pixels round-trip; demand gate expires; "
        "missing GL degrades to a reason; revision advances per frame; camera wraps and clamps"
    )


if __name__ == "__main__":
    main()
