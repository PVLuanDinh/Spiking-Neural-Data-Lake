"""
v2.11 - nl_interface: a LOCAL natural-language front-end that turns English into a
SAFE, allow-listed intent for the arm or a structured query for the data lake
(the LLM backend is optional and injected).

This is the safety FIREWALL between a language model and the robot. A backend (a local
AirLLM model on-device, or the stdlib StubBackend here) only ever returns a short LABEL;
this module maps that label to a command the wire allow-list (serial_arm.is_valid_command)
ALREADY accepts, or to a structured lake query, or REJECTS it. Key properties:

  * The LLM never emits a motor command. It selects WHICH allow-listed action (by label);
    the concrete command string is this module's own fixed constant, never the model's text.
  * Defense in depth: every arm command returned is re-checked with is_valid_command; an
    unknown label or a command that fails the allow-list -> {"kind": "reject"} (no action).
  * interpret() only PARSES + VALIDATES + CLASSIFIES. It never drives the arm and never
    reads the lake -- execution stays in the existing safety-gated paths (Interpreter ->
    ArmSim/SerialArm.apply() enforce reflex / E-STOP / collision; spike_telemetry_hub
    .disk_query runs the lake read). So nothing here can bypass the envelope.

Backend contract: any object with `.complete(prompt: str) -> str` returning a single label.
StubBackend is the default AND the offline / no-GPU fallback: a deterministic
keyword router (no model, no network) -- so this module and its self-check run in CI with
no torch and no weights. The AirLLM backend (local_llm/airllm_backend.py, optional [llm]
extra) is a drop-in that returns the same labels from an on-device model.

Keyword/label routing + a fixed allow-list map -- NOT language understanding.
Ceiling: fixed intents (no free-form arm parameters; rotations are fixed increments like the
Interpreter's), single-label output, English keyword cues only. Additive / opt-in: nothing
here runs unless you call it; the arm/lake code paths are unchanged.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.nl_interface
"""

import re as _re
from functools import cache

from snn_data_lake.brainstem import arm_config
from snn_data_lake.brainstem.serial_arm import is_valid_command

# intent label -> a command token the wire allow-list already accepts. The VALUE is ours,
# never the model's text, so the model can only choose among these, never inject a command.
ARM_LABELS = {
    "close": "ENGAGE",
    "open": "RELEASE",
    "hold": "HOLD",
    "home": "HOME",
    "retract": "RETRACT_ALL",
    "stop": "EMERGENCY_STOP",
}
# One label per rotatable joint, derived from the arm. The incumbent hand-listed two; on a
# 6-DOF body that would have left four joints with no way to say them in English, and the
# _CUES table below already spoke of "shoulder" and "elbow" as though they were separate
# things -- they now genuinely are.
ARM_LABELS.update(
    {n: arm_config.command_for(n, arm_config.DEFAULT_STEP_DEG) for n in arm_config.ROTATABLE}
)
# English keyword cues -> intent label (first hit wins; safety cues checked first).
_CUES = [
    (("stop", "halt", "freeze", "emergency", "abort"), "stop"),
    (("retract", "withdraw", "pull back", "back off"), "retract"),
    (("hold", "wait", "stay", "pause", "still"), "hold"),
    (("close", "grip", "grab", "grasp", "pick up", "clamp"), "close"),
    (("open", "release", "let go", "drop"), "open"),
    (("home", "reset", "neutral", "center", "centre"), "home"),
    # Joint cues. FIRST HIT WINS, so every two-word form is listed before every bare word:
    # "flex" belongs to two joints and "wrist" belongs to two, and a bare cue for either would
    # steal the other's phrases outright rather than merely competing with them.
    (("shoulder lift",), "shoulder_lift"),
    (("shoulder pan",), "shoulder_pan"),
    (("wrist flex",), "wrist_flex"),
    (("wrist roll",), "wrist_roll"),
    (("elbow flex",), "elbow_flex"),
    # Unambiguous single words.
    (("pan", "swivel", "base"), "shoulder_pan"),
    (("roll", "twist"), "wrist_roll"),
    (("lift", "raise"), "shoulder_lift"),
    (("bend",), "wrist_flex"),
    # Bare anatomical words last, resolving where the incumbent's table pointed them:
    # "shoulder" was joint A and "elbow" was joint B.
    (("shoulder",), "shoulder_lift"),
    (("elbow",), "elbow_flex"),
    (("wrist",), "wrist_flex"),
    (("query", "read", "show", "telemetry", "channel", "spikes", "lake", "select"), "query"),
]


class StubBackend:
    """Deterministic keyword router: the offline fallback AND the CI test double.
    Scans the prompt text for the first matching cue and returns its label ('reject' if none)."""

    def complete(self, prompt):
        low = prompt.lower()
        if "request:" in low:
            low = low.split("request:", 1)[1]  # scan the user's request, not the label menu
        # WORD BOUNDARIES, not substrings. MEASURED with bare `k in low`: 15 of 16 ordinary
        # non-command sentences and 11 of 12 questions produced an ARM COMMAND. "the store is
        # closed" closed the gripper, "she centered the image" sent the arm HOME, "I was holding
        # out hope" issued HOLD, "he dropped out of school" released. This is the firewall in
        # front of a physical actuator, and it was matching inside words.
        for keys, label in _CUES:
            if any(_word_re(k).search(low) for k in keys):
                # QUESTIONS AND NEGATIONS ARE NOT COMMANDS -- with ONE exception, below. Word
                # boundaries alone still leave "is it safe to grip that" gripping, because `grip`
                # really is a standalone word there; the sentence is interrogative, not imperative.
                #
                # STOP IS EXEMPT ON PURPOSE. "should I stop you" must still stop. Failing to stop
                # when asked is unsafe; stopping on a question about stopping is a nuisance, and
                # since 2026-08-03 a self-recovering one -- the arm re-arms once the proximity
                # reading is clear. The asymmetry runs one way and the code says which way.
                if label != "stop" and _is_non_imperative(low):
                    return "reject"
                return label
        return "reject"


@cache
def _word_re(cue):
    """A cue matcher anchored at word boundaries. Cached because the cue set is fixed and small,
    and this runs on every chat turn."""
    return _re.compile(r"\b" + _re.escape(cue) + r"\b")


#: Openers that make a sentence a question or a denial rather than an order. Derived from the two
#: measured failure sets rather than imagined: every one of these appears in a phrase that reached
#: the actuator through a cue it had no business triggering.
_NON_IMPERATIVE = (
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "am",
    "have",
    "has",
    "if",
    "whether",
)
_NEGATIONS = ("not", "n't", "never", "no ", "don't", "cannot", "stop asking")


def _is_non_imperative(text):
    """True when the text reads as a question or a denial. Three cheap signals, any of which is
    enough: it ends in a question mark, it OPENS with an interrogative/auxiliary, or it contains a
    negation. Deliberately crude -- this is a safety filter in front of an actuator, and the cost of
    refusing a real command is one repetition while the cost of accepting a fake one is motion."""
    t = text.strip()
    if not t:
        return False
    if "?" in t:
        return True
    if t.split()[0] in _NON_IMPERATIVE:
        return True
    return any(n in t for n in _NEGATIONS)


def _parse_query(text):
    """Very small structured lake-query extractor: channels (ints after 'channel'/'ch') +
    an optional window (int after 'window'/'last'). Defaults: all channels, window 1."""
    import re

    chans = [int(n) for n in re.findall(r"(?:channel|ch)\s*#?\s*(\d+)", text.lower())]
    win = re.search(r"(?:window|last|w)\s*=?\s*(\d+)", text.lower())
    return {"channels": chans or None, "window": int(win.group(1)) if win else 1}


class NLInterface:
    """English -> a safe, validated intent dict. backend.complete(prompt)->label; default StubBackend."""

    def __init__(self, backend=None, valid=is_valid_command):
        self.backend = backend or StubBackend()
        self.valid = valid  # the wire allow-list (single source of truth)

    def _prompt(self, text):
        labels = ", ".join(sorted(ARM_LABELS) + ["query", "reject"])
        return (
            f"Pick ONE label for the request. Labels: {labels}. "
            f"Answer with the label only.\nRequest: {text}"
        )

    def interpret(self, text):
        """text -> {"kind":"arm","command":...} | {"kind":"lake","query":...} | {"kind":"reject","reason":...}.
        The returned arm command is GUARANTEED to pass the wire allow-list; unknown/unsafe -> reject."""
        label = self.backend.complete(self._prompt(text)).strip().lower()
        if label in ARM_LABELS:
            cmd = ARM_LABELS[label]
            if self.valid(cmd):  # defense in depth: never emit a non-allow-listed token
                return {"kind": "arm", "label": label, "command": cmd}
            return {"kind": "reject", "reason": f"command failed the allow-list: {cmd}"}
        if label == "query":
            return {"kind": "lake", "query": _parse_query(text)}
        return {"kind": "reject", "reason": f"no confident/safe mapping (label={label!r})"}


class _AdversarialBackend:
    """A backend that tries to inject a raw / unsafe command, to prove the firewall holds."""

    def __init__(self, payload):
        self.payload = payload

    def complete(self, prompt):
        return self.payload


def main():
    nl = NLInterface()  # default StubBackend, real allow-list

    print("=" * 62)
    print("NL INTERFACE (English -> safe allow-listed arm intent / lake query)")
    print("=" * 62)

    # 1 ARM MAPPING: English phrases route to the right allow-listed command, and EVERY arm
    #   command returned passes the wire allow-list (defense in depth).
    cases = {
        "please close the effector and pick it up": "ENGAGE",
        "open the hand and release": "RELEASE",
        "go home / reset to neutral": "HOME",
        "STOP! emergency": "EMERGENCY_STOP",
        "withdraw and back off": "RETRACT_ALL",
        "just hold and wait": "HOLD",
        # One phrase per rotatable joint. The incumbent tested two because the arm had two;
        # testing fewer than all of them is how a joint ends up with no way to say it.
        **{
            f"{n.replace('_', ' ')} please": arm_config.command_for(n, arm_config.DEFAULT_STEP_DEG)
            for n in arm_config.ROTATABLE
        },
        # The bare anatomical words still have to resolve, and to the SAME joints the incumbent
        # meant by them -- its cue table read "shoulder" as joint A and "elbow" as joint B.
        "turn the elbow": arm_config.command_for("elbow_flex", arm_config.DEFAULT_STEP_DEG),
        "raise the shoulder": arm_config.command_for("shoulder_lift", arm_config.DEFAULT_STEP_DEG),
    }
    arm_ok = 0
    for text, want in cases.items():
        out = nl.interpret(text)
        assert out["kind"] == "arm" and out["command"] == want, f"{text!r} -> {out} (want {want})"
        assert is_valid_command(out["command"]), f"returned a non-allow-listed command: {out}"
        arm_ok += 1

    # 2 LAKE MAPPING: a query phrase -> a structured query intent (never an arm command).
    q = nl.interpret("read telemetry from channel 3 and channel 7, window 64")
    assert (
        q["kind"] == "lake" and q["query"]["channels"] == [3, 7] and q["query"]["window"] == 64
    ), q

    # 3 SAFETY / REJECT: unrecognised or unsafe English -> reject, NEVER an arm command.
    for bad in ("what time is it", "delete all the files", "smash everything to pieces", ""):
        out = nl.interpret(bad)
        assert out["kind"] != "arm", (
            f"unsafe/unknown text produced an arm command: {bad!r} -> {out}"
        )

    # 4 FIREWALL: an ADVERSARIAL backend that returns raw/unsafe command strings must be rejected
    #   -- the model can never inject a command; only fixed labels map to fixed allow-listed tokens.
    for payload in (
        "DROP_EVERYTHING",
        "JOINT_A_ROTATE(+9999deg)",
        "ENGAGE; rm -rf /",
        "EMERGENCY_STOP\nJOINT_B_ROTATE(+40deg)",
        "rotate_a EVIL",
    ):
        out = NLInterface(backend=_AdversarialBackend(payload)).interpret("do it")
        assert out["kind"] == "reject", f"firewall breached: payload {payload!r} -> {out}"

    # 5 DETERMINISM: same text -> identical intent.
    assert nl.interpret("close the effector") == nl.interpret("close the effector"), (
        "must be deterministic"
    )

    print(
        f"1 ARM MAP    : {arm_ok}/{len(cases)} phrases -> correct allow-listed command (all is_valid_command)"
    )
    print(
        f"2 LAKE MAP   : query phrase -> channels={q['query']['channels']} window={q['query']['window']}"
    )
    print("3 REJECT     : unknown/unsafe English -> reject, never an arm command")
    print("4 FIREWALL   : adversarial backend payloads (raw commands / injection) all rejected")
    print("5 DETERMINISM: same text -> identical intent")
    print("=" * 62)
    print(
        f"self-check OK: {arm_ok} arm phrases mapped to allow-listed commands; lake query parsed; "
        f"unsafe + adversarial-backend inputs rejected (LLM can only pick allow-listed labels); deterministic"
    )


if __name__ == "__main__":
    main()
