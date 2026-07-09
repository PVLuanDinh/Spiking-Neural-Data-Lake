"""
v0.51 — UDP back-channel that closes the split-process loop (long-standing open end).

In-process, closed_loop.py already feeds the Interpreter's OUTCOME reward back into the
dopamine/cortisol learners. The LIVE deployment splits the pieces across processes:

    live_arm.py --serial COM8        (sensor -> recognize -> Interpreter -> arm)
    interpreter.py --pipe            (emits commands + OUTCOME <reward>)

...but interpreter.send_outcome only wrote to stderr/file — there was no transport carrying
OUTCOME back to the running loop. This is that transport: a tiny stdlib UDP socket on
localhost. The Interpreter process `send_outcome(r, port)`s; the loop process `recv_outcome()`s
and applies the reward to its valence/cortisol learners. No NEURON, no deps (Gemini brief
proposed a UDP bridge — we keep it stdlib + localhost).

  python arm_bridge.py        # self-check (in-process loopback, no real network needed)
"""
import os
import secrets
import socket

PREFIX = "OUTCOME"


def parse_outcome(msg, token=None):
    """'OUTCOME <reward> [<token>]' -> reward clamped to [-1, 1], else None.

    Clamping matches signal_loop.parse_outcome so a malformed/hostile localhost datagram can't
    inject an out-of-range reward into the valence/cortisol learners. (audit P2) When `token` is
    set the datagram MUST carry a matching token (constant-time compared) or it is rejected, so a
    stray/forged localhost datagram cannot poison the learned policy. token=None keeps the legacy
    unauthenticated behavior."""
    parts = msg.strip().split()
    if not parts or parts[0] != PREFIX:
        return None
    if token is not None:
        if len(parts) != 3 or not secrets.compare_digest(parts[2], token):
            return None
    elif len(parts) != 2:
        return None
    try:
        return max(-1.0, min(1.0, float(parts[1])))
    except (IndexError, ValueError):
        return None


def send_outcome(reward, port, host="127.0.0.1", token=None):
    """Fire one OUTCOME datagram at the loop process (called by the Interpreter side). Pass `token`
    (or set env SNN_OUTCOME_TOKEN) to authenticate against a token-guarded OutcomeBridge."""
    token = token if token is not None else os.environ.get("SNN_OUTCOME_TOKEN")
    body = f"{PREFIX} {reward}" + (f" {token}" if token else "")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(body.encode("ascii"), (host, port))


class OutcomeBridge:
    """Loop-side receiver. Binds a UDP port; recv_outcome() returns the next reward or None.
    Port 0 = OS-assigned ephemeral (read it back via .port).

    AUTHENTICATED by default (audit P2): a fresh random token is generated per instance (read via
    .token) and only datagrams carrying it are accepted. For the split-process deployment, share a
    secret via env SNN_OUTCOME_TOKEN on both sides. Pass token="" to opt out (legacy open behavior)."""

    def __init__(self, host="127.0.0.1", port=0, timeout=1.0, token=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(timeout)               # never block the control loop forever
        self.port = self.sock.getsockname()[1]
        if token is None:
            token = os.environ.get("SNN_OUTCOME_TOKEN") or secrets.token_hex(8)
        self.token = token or None                  # "" -> None -> unauthenticated (explicit opt-out)

    def recv_outcome(self):
        """Next reward float, or None on timeout / malformed / unauthenticated packet."""
        try:
            data, _ = self.sock.recvfrom(256)
        except TimeoutError:
            return None
        return parse_outcome(data.decode("ascii", "ignore"), self.token)

    def close(self):
        self.sock.close()


def main():
    bridge = OutcomeBridge(timeout=1.0)             # ephemeral port, short timeout (CI-safe); auto-token
    port, tok = bridge.port, bridge.token

    # 1) an authenticated positive reward survives the round trip
    send_outcome(0.5, port, token=tok)
    r1 = bridge.recv_outcome()
    # 2) an authenticated negative (aversive) reward too
    send_outcome(-1.0, port, token=tok)
    r2 = bridge.recv_outcome()
    # 3) malformed packet is ignored (not a crash, not a false reward)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"garbage payload here", ("127.0.0.1", port))
    r3 = bridge.recv_outcome()
    # 4) (audit P2) a datagram with NO token or a FORGED token is rejected -- can't inject a reward.
    #    Raw sends bypass env so the test is deterministic regardless of SNN_OUTCOME_TOKEN.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"OUTCOME 0.9", ("127.0.0.1", port))                       # unauthenticated
    r4 = bridge.recv_outcome()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"OUTCOME 0.9 deadbeefdeadbeef", ("127.0.0.1", port))      # forged token
    r5 = bridge.recv_outcome()
    # 5) closing the loop: route received rewards into a learner-like accumulator
    value = 0.0
    for reward in (r1, r2):
        if reward is not None:
            value += 0.5 * reward                   # stand-in for valence/cortisol .learn(reward)

    bridge.close()

    print("=" * 60)
    print("ARM BRIDGE — authenticated UDP OUTCOME back-channel (split-process loop)")
    print("=" * 60)
    print(f"ephemeral port : {port}")
    print(f"token          : {tok[:4]}... (per-session; share via SNN_OUTCOME_TOKEN)")
    print(f"recv +0.5      : {r1}")
    print(f"recv -1.0      : {r2}")
    print(f"recv garbage   : {r3} (ignored)")
    print(f"recv no-token  : {r4} (rejected)   forged-token: {r5} (rejected)")
    print(f"loop value after applying rewards: {value:+.3f}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    assert r1 == 0.5, f"positive reward lost in transit: {r1}"
    assert r2 == -1.0, f"negative reward lost in transit: {r2}"
    assert r3 is None, "malformed packet must be ignored, not parsed as a reward"
    assert r4 is None and r5 is None, "unauthenticated / forged reward must be rejected (audit P2)"
    assert abs(value - (-0.25)) < 1e-9, f"rewards did not reach the learner: {value}"
    # token enforcement + clamping on an authenticated datagram
    assert parse_outcome(f"OUTCOME 0.3 {tok}", tok) == 0.3, "authenticated reward must parse"
    assert parse_outcome("OUTCOME 0.3", tok) is None, "missing token must be rejected"
    assert parse_outcome(f"OUTCOME 9999 {tok}", tok) == 1.0 and parse_outcome(f"OUTCOME -9999 {tok}", tok) == -1.0, "reward must be clamped"
    # legacy opt-out (no token) still round-trips
    assert parse_outcome("OUTCOME 0.3") == 0.3 and parse_outcome("HELLO") is None
    print("self-check OK: authenticated OUTCOME round-trips, unauth/forged rejected, clamped, reaches the loop")


if __name__ == "__main__":
    main()
