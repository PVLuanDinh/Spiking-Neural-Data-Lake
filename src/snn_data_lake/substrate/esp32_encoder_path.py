"""
v1.0 -- ESP32 serial stream -> multimodal encoder -> spike vector.

Raw ESP32 serial lines (distance_m, temp_k) -> parse -> normalize to [0,1] ->
the existing encode_multimodal -> N=64 spike feature vector (pyserial imported
lazily only in read_serial_lines).

Firmware quirks handled:
  * distance > 3.00 m -> firmware sends temperature as sentinel -1 (idle) -> treat
    temperature as MISSING (None -> dropped from encoding).
  * control lines STOPPED / RESUMED and any non-numeric / blank / boot-banner ->
    skip (return None from parse, encode_reading gracefully skips).

Distance band [48:64]: distance_m / 3.0 m, clamped [0,1] (3 m is object-present gate).
Temperature band [16:32]: (temp_k - 293.15 K) / 20 K, clamped [0,1] (20-40 C band).
Visual [0:16] and audio [32:48]: absent (None) -> zeros until video module lands.

Raw-serial -> normalize -> multimodal-encoder data path. Static normalization
bands (distance 0-3 m gate, temp 20-40 C). Not a spiking implementation of the sensor,
not cognition.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.substrate.esp32_encoder_path
"""

from snn_data_lake.cortex.multimodal_encoder import encode_multimodal
from snn_data_lake.cortex.spike_preprocessing import N, encode_latency

# Temperature normalization: 20 C (293.15 K) to 40 C (313.15 K).
T_LO_K = 293.15
T_HI_K = 313.15


def parse_esp32_line(line):
    """Parse one ESP32 serial line 'distance_m, temp_k' -> (distance_m, temp_k) or None.

    Returns (distance_m: float, temp_k: float) tuple, or None if the line is
    non-numeric, blank, control (STOPPED/RESUMED), or boot-banner.
    """
    line = line.strip()
    if not line:
        return None

    # Skip control lines and boot banner.
    if (
        line in ("STOPPED", "RESUMED")
        or "ets " in line
        or "rst:" in line
        or "boot:" in line
        or "mode:" in line
    ):
        return None

    # Try to parse as "distance_m, temp_k" (two comma-separated floats).
    parts = line.split(",")
    if len(parts) != 2:
        return None

    try:
        distance_m = float(parts[0].strip())
        temp_k = float(parts[1].strip())
        return (distance_m, temp_k)
    except (ValueError, AttributeError):
        return None


def normalize(distance_m, temp_k):
    """Normalize distance and temperature to [0,1], handling firmware idle sentinel.

    distance_m: raw distance in meters. Maps [0, 3] -> [0, 1], clamped.
    temp_k: raw temperature in Kelvin. Idle (firmware sends -1 when distance > 3 m),
            or any temp <= 0 K, or out of physical band -> None (missing).

    Returns (distance01: float, temp01: float or None).
      * distance01 in [0, 1].
      * temp01 in [0, 1] if valid, else None (missing modality).
    """  # noqa: D200
    # Distance: 3.0 m is object-present gate.
    distance01 = max(0.0, min(1.0, distance_m / 3.0))

    # Temperature: idle sentinel (-1 or any temp <= 0) -> missing.
    if temp_k <= 0:
        temp01 = None
    else:
        # Clamp to [20, 40] C band: (T - 293.15) / 20.
        band_width = T_HI_K - T_LO_K
        temp01 = max(0.0, min(1.0, (temp_k - T_LO_K) / band_width))

    return (distance01, temp01)


def encode_reading(line):
    """Parse, normalize, and encode one ESP32 serial line into an N=64 vector.

    Returns list of N floats in [0,1], or None if the line did not parse.
    """
    parsed = parse_esp32_line(line)
    if parsed is None:
        return None

    distance_m, temp_k = parsed
    distance01, temp01 = normalize(distance_m, temp_k)

    # Feed the existing multimodal encoder: distance and temperature present,
    # visual and audio absent (None -> graceful dropout -> zeros).
    feats = encode_multimodal(visual=None, temperature=temp01, audio=None, distance=distance01)
    return feats


def encode_stream(lines):
    """Yield N=64 spike vectors for each parseable line in an iterable of strings.

    Skips non-parseable lines (control, banner, malformed).
    """
    for line in lines:
        feats = encode_reading(line)
        if feats is not None:
            yield feats


def spikes_for(feats):
    """Convenience: encode one spike vector into spike list (time, feature) pairs.

    Wraps encode_latency; returns list of (time, feature) events.
    """
    return encode_latency(feats)


def read_serial_lines(port, baud=115200, timeout=1.0):
    """Lazily yield decoded text lines from a live serial port.

    Imports pyserial only here (not at module top), so importing this module
    never pulls in pyserial. If pyserial is missing, raises a clear
    ImportError directing the user to `pip install ".[arm]"`.

    Args:
        port (str): serial port name (e.g., "COM3" on Windows,
            "/dev/ttyUSB0" on Linux).
        baud (int): baud rate (default 115200, the ESP32 firmware rate).
        timeout (float): read timeout in seconds.

    Yields:
        str: decoded text lines (newline-stripped) from the port.
    """
    try:
        import serial
    except ImportError as e:
        raise ImportError('pyserial not found. Install with: pip install ".[arm]"') from e

    ser = serial.Serial(port, baud, timeout=timeout)
    try:
        while True:
            line = ser.readline()
            if not line:
                continue
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded:
                yield decoded
    finally:
        ser.close()


def main():
    """Self-check: deterministic, ASCII-only, genuine asserts on captured hardware data."""
    # Exact lines captured from the reference ESP32 @ 115200 baud.
    captured_lines = [
        "0.62, 297.00",
        "0.61, 297.00",
        "0.69, 297.02",
        "0.70, 297.02",
        "0.58, 297.06",
        "0.41, 297.14",
    ]

    # ---- Parse tests ----
    p1 = parse_esp32_line("0.41, 297.14")
    assert p1 is not None and abs(p1[0] - 0.41) < 1e-9 and abs(p1[1] - 297.14) < 1e-9, (
        f"parse_esp32_line('0.41, 297.14') = {p1}, expected (0.41, 297.14)"
    )

    assert parse_esp32_line("STOPPED") is None, "parse_esp32_line('STOPPED') should be None"
    assert parse_esp32_line("RESUMED") is None, "parse_esp32_line('RESUMED') should be None"
    assert parse_esp32_line("ets Jul 29 2019 12:21:46") is None, "boot banner should be None"
    assert parse_esp32_line("") is None, "blank line should be None"
    assert parse_esp32_line("malformed") is None, "malformed line should be None"

    # ---- Normalization tests ----
    # Distance endpoints: 0 m -> 0.0, 3.0 m -> 1.0, 6.0 m -> 1.0 (clamp).
    d0, t0 = normalize(0.0, 293.15)
    assert abs(d0 - 0.0) < 1e-9, f"normalize(0.0, 293.15): distance should be 0.0, got {d0}"

    d3, t3 = normalize(3.0, 313.15)
    assert abs(d3 - 1.0) < 1e-9, f"normalize(3.0, 313.15): distance should be 1.0, got {d3}"

    d6, t6 = normalize(6.0, 300.0)
    msg = f"normalize(6.0, 300.0): distance should be 1.0 (clamped), got {d6}"
    assert abs(d6 - 1.0) < 1e-9, msg

    # Temperature endpoints: 293.15 K -> 0.0, 313.15 K -> 1.0.
    d_lo, t_lo = normalize(1.5, T_LO_K)
    assert abs(t_lo - 0.0) < 1e-9, f"normalize(1.5, {T_LO_K}): temp should be 0.0, got {t_lo}"

    d_hi, t_hi = normalize(1.5, T_HI_K)
    assert abs(t_hi - 1.0) < 1e-9, f"normalize(1.5, {T_HI_K}): temp should be 1.0, got {t_hi}"

    # Temperature below band (cold): clamp to 0.0.
    d_cold, t_cold = normalize(1.5, 273.15)
    msg = f"normalize(1.5, 273.15): temp should be 0.0 (clamp), got {t_cold}"
    assert abs(t_cold - 0.0) < 1e-9, msg

    # Temperature idle sentinel or negative: None (missing).
    d_idle, t_idle = normalize(4.20, -1.0)
    assert t_idle is None, f"normalize(4.20, -1.0): temp should be None (idle), got {t_idle}"
    msg = f"normalize(4.20, -1.0): distance should be 1.0 (clamp), got {d_idle}"
    assert abs(d_idle - 1.0) < 1e-9, msg

    d_neg, t_neg = normalize(1.5, -50.0)
    assert t_neg is None, f"normalize(1.5, -50.0): temp should be None (negative), got {t_neg}"

    # ---- Encoding tests: object-present line ----
    feats_obj = encode_reading("0.41, 297.14")
    assert feats_obj is not None and len(feats_obj) == N, (
        f"encode_reading('0.41, 297.14') should return {N} floats, "
        f"got {len(feats_obj) if feats_obj else None}"
    )

    # All features in [0, 1].
    assert all(0.0 <= v <= 1.0 for v in feats_obj), "all features must be in [0, 1]"

    # Visual band [0:16] all zeros (absent).
    assert feats_obj[0:16] == [0.0] * 16, (
        f"visual band [0:16] should be all zeros, got {feats_obj[0:16]}"
    )

    # Temperature band [16:32] NOT all zeros (temp encoded).
    temp_band = feats_obj[16:32]
    assert temp_band != [0.0] * 16, (
        f"temperature band [16:32] should NOT be all zeros (temp={297.14}), got {temp_band}"
    )
    assert all(0.0 <= v <= 1.0 for v in temp_band), "temperature band must be in [0, 1]"

    # Audio band [32:48] all zeros (absent).
    assert feats_obj[32:48] == [0.0] * 16, (
        f"audio band [32:48] should be all zeros, got {feats_obj[32:48]}"
    )

    # Distance band [48:64] NOT all zeros (distance encoded).
    dist_band = feats_obj[48:64]
    assert dist_band != [0.0] * 16, (
        f"distance band [48:64] should NOT be all zeros (distance={0.41}), got {dist_band}"
    )
    assert all(0.0 <= v <= 1.0 for v in dist_band), "distance band must be in [0, 1]"

    # ---- Encoding tests: idle line (distance > 3, temp = -1) ----
    feats_idle = encode_reading("4.20, -1.00")
    assert feats_idle is not None and len(feats_idle) == N, (
        f"encode_reading('4.20, -1.00') should return {N} floats, "
        f"got {len(feats_idle) if feats_idle else None}"
    )

    # Distance band [48:64] should still be encoded (distance > 3 -> 1.0).
    dist_band_idle = feats_idle[48:64]
    assert dist_band_idle != [0.0] * 16, (
        f"distance band [48:64] should NOT be zeros (distance={4.20}), got {dist_band_idle}"
    )

    # Temperature band [16:32] should be all zeros (temp missing due to idle sentinel).
    temp_band_idle = feats_idle[16:32]
    assert temp_band_idle == [0.0] * 16, (
        f"temperature band [16:32] should be all zeros (temp=-1 idle), got {temp_band_idle}"
    )

    # ---- Determinism ----
    feats_again = encode_reading("0.41, 297.14")
    assert feats_obj == feats_again, (
        "encode_reading must be deterministic: same input -> identical vector"
    )

    # ---- Stream tests ----
    all_feats = list(encode_stream(captured_lines))
    assert len(all_feats) == 6, (
        f"encode_stream over 6 captured lines should yield 6 vectors, got {len(all_feats)}"
    )

    # All vectors in [0, 1]^N.
    for i, vec in enumerate(all_feats):
        assert len(vec) == N and all(0.0 <= v <= 1.0 for v in vec), (
            f"vector {i} must be {N} floats in [0, 1]"
        )

    # ---- Spike encoding ----
    spikes = spikes_for(feats_obj)
    assert isinstance(spikes, list), "spikes_for must return a list"
    assert all(len(ev) == 2 and 0 <= ev[1] < N for ev in spikes), (
        "spikes must be (time, feature) pairs with feature in [0, N)"
    )

    # ---- Report ----
    print("=" * 70)
    print("ESP32 ENCODER PATH (serial -> normalize -> multimodal -> N=64 spikes)")
    print("=" * 70)
    print()
    print("PARSE RESULTS:")
    print(f"  Parsed 6 captured lines: {len(all_feats)} feature vectors")
    print()

    # Per-band breakdown for a sample reading (first object-present line).
    print("SAMPLE READING: '0.62, 297.00'")
    sample = all_feats[0]
    visual_mean = sum(sample[0:16]) / 16
    temp_mean = sum(sample[16:32]) / 16
    audio_mean = sum(sample[32:48]) / 16
    dist_mean = sum(sample[48:64]) / 16
    print(f"  visual   [0:16]  mean={visual_mean:.4f} (absent -> 0.0)")
    print(f"  temp     [16:32] mean={temp_mean:.4f} (297.00 K -> ~0.35)")
    print(f"  audio    [32:48] mean={audio_mean:.4f} (absent -> 0.0)")
    print(f"  distance [48:64] mean={dist_mean:.4f} (0.62 m / 3.0 -> ~0.21)")
    print()

    print("IDLE vs OBJECT BEHAVIOR:")
    feats_idle_sample = encode_reading("4.20, -1.00")
    idle_temp_band = feats_idle_sample[16:32]
    idle_dist_band = feats_idle_sample[48:64]
    print("  idle line '4.20, -1.00':")
    print(f"    temp band [16:32]  all zeros: {idle_temp_band == [0.0] * 16} (temp missing)")
    print(f"    dist band [48:64]  nonzero:   {idle_dist_band != [0.0] * 16} (dist={4.20} -> 1.0)")
    print()

    print("NORMALIZATION BAND ENDPOINTS:")
    print(f"  distance 0.0 m  -> {normalize(0.0, 300.0)[0]:.4f}")
    print(f"  distance 3.0 m  -> {normalize(3.0, 300.0)[0]:.4f}")
    print(f"  distance 6.0 m  -> {normalize(6.0, 300.0)[0]:.4f} (clamped)")
    print(f"  temp {T_LO_K:.2f} K -> {normalize(1.5, T_LO_K)[1]:.4f}")
    print(f"  temp {T_HI_K:.2f} K -> {normalize(1.5, T_HI_K)[1]:.4f}")
    print(f"  temp 273.15 K   -> {normalize(1.5, 273.15)[1]:.4f} (cold, clamped)")
    print(f"  temp -1.00 K    -> {normalize(1.5, -1.0)[1]} (idle sentinel)")
    print()

    print("=" * 70)
    print("self-check OK: parse, normalize, encode, stream, spikes, deterministic")
    print("=" * 70)


if __name__ == "__main__":
    main()
