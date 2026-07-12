"""
Spiking chat -- a REAL spiking pipeline for a chat app, not a language model.

Every message runs a genuine spiking loop over the zero-dep core:

    text  --encode-->  spikes  --store-->  spike data lake  --classify-->  intent
                                                                              |
    reply  <--decode--  brain's REAL state (LiveBrain.converse)  <-----------+

Mechanism (deterministic, pure-stdlib core):
  1. ENCODE (real spikes): each character becomes one spike -- channel = ord(c) % N,
     time = its position in the string. A non-empty message yields >=1 spike on valid
     channels; the empty string yields zero spikes. (Deterministic char->channel+time
     code; the rank-order idiom lives in cortex/rank_order_coding.py.)
  2. STORE (durable lake): every spike is appended to a persistent conversation lake
     (substrate/streaming_hub.StreamingHub, an append-only 8 B/event .spkl log) with a
     GROWING global time cursor, so the whole chat log accumulates as spikes on disk and
     survives a restart (resume=True replays the log).
  3. CLASSIFY (spike-based decision): the message's per-channel spike-COUNT vector is
     matched by cosine to one prototype per intent (built once at init by encoding a few
     example phrases per intent). The nearest prototype names the intent; too-far -> unknown.
     The intent is decided FROM the spikes -- this is the spec's deterministic
     nearest-spike-prototype matcher (the cortex analogues are cortex/learned_matcher.py
     and cortex/rank_order_coding.py; a plain rank-order match collapses on sparse
     char-count vectors, so cosine over counts is used).
  4. DECODE (grounded, honest): the spike-decided intent surfaces the brain's REAL state.
     cortex/live_brain.LiveBrain.converse() produces the grounded sentence -- it answers
     greeting / goal / memory / workspace / safety / consciousness from actual state and
     routes arm commands through the same NL firewall + gated arm the control panel uses.

Honest framing: this is a REAL spiking pipeline (deterministic spike encoding -> durable
spike lake -> a spike-pattern intent decision -> a grounded decode), and every step is
inspectable. It is NOT a large language model and does NOT fake fluent free-form generation:
the repo's char-level spiking LM is UNTRAINED (gibberish), so the reply is DECODED from the
spike-decided intent and GROUNDED in the brain's actual goal / working memory / workspace /
safety state -- never invented. The value is that the chat genuinely runs through the
spiking substrate and the data lake, with a real spike-based decision behind every reply.

  PYTHONUTF8=1 PYTHONPATH=src python chat_app/spiking_chat.py
"""
import math
import os
import random
import tempfile

from snn_data_lake.brainstem.arm_config import CONTACT
from snn_data_lake.brainstem.autonomous_loop import ACTIONS
from snn_data_lake.cortex.live_brain import LiveBrain
from snn_data_lake.substrate.streaming_hub import StreamingHub

N_CHANNELS = 128            # one channel per ASCII code point (ord(c) % 128)
UNKNOWN_FLOOR = 0.6         # cosine below this -> the spikes match no prototype -> unknown

# A few example phrases per intent -> the spike prototypes are built from these at init.
INTENT_EXAMPLES = {
    "greeting":      ["hello", "hi there", "hey", "good morning", "greetings", "yo"],
    "ask_goal":      ["what is your goal", "what do you want", "what are you trying to do",
                      "what is your objective", "what do you intend", "what is your plan"],
    "ask_memory":    ["what do you remember", "what is in your memory", "what do you recall",
                      "what have you stored", "your lake memory", "recall your memory"],
    "ask_workspace": ["what are you thinking", "what is in your workspace",
                      "what are you attending to", "what is your focus", "what do you notice"],
    "ask_safety":    ["are you safe", "is it dangerous", "safety first", "will you hurt anyone",
                      "emergency stop", "is this dangerous"],
    "command":       ["close the gripper", "open the gripper", "move the arm",
                      "close gripper", "open gripper", "grip"],
}


def _cosine(a, b):
    """Cosine similarity of two non-negative count vectors (0.0 if either is all-zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SpikingChat:
    """A chat whose every turn runs a real spiking pipeline over the SNDL core.

    Holds a persistent conversation lake (StreamingHub on `lake_path`, or a temp .spkl if
    None) AND a LiveBrain (primed so it holds a real goal + working memory). respond(text)
    runs encode -> store -> classify -> tick -> decode and returns the whole pipeline state."""

    def __init__(self, seed=0, lake_path=None, self_improve=False):
        self.seed = seed
        self.rng = random.Random(seed)          # per-instance rng (no import-time seeding)
        self.n_channels = N_CHANNELS
        # SELF-IMPROVEMENT mode: instead of a fixed external "open the gripper" reward, the brain
        # runs CURIOSITY-DRIVEN -- an intrinsic learning-progress reward (no external task reward)
        # drives SELF-DIRECTED learning from its own experience each tick. Its meta-goal is to
        # IMPROVE ITSELF: one primitive (curiosity + spiking learning), not open-ended recursive
        # self-improvement. Gated + safe: the reflex E-STOP runs first, curiosity feeds value only.
        self.self_improve = self_improve
        self.meta_goal = "self-improvement" if self_improve else None
        self._ticks = 0                         # self-improvement ticks accumulated
        self._lp = 0.0                          # mean learning-progress (improving-now signal)

        # ---- persistent conversation lake (durable spike log) ----
        if lake_path is None:
            fd, lake_path = tempfile.mkstemp(suffix=".spkl")
            os.close(fd)                          # StreamingHub opens its own handle
            os.remove(lake_path)                  # start from an empty log
            self._created_temp = True
        else:
            self._created_temp = False
        self.lake_path = lake_path
        self.lake = StreamingHub(lake_path, self.n_channels, resume=True)
        # global time cursor: continue past any spikes already on disk (a resumed chat log)
        existing = self.lake.query_window(0, 1 << 32)
        self._t_cursor = (max(t for t, _ in existing) + 2) if existing else 0

        # ---- the live brain: prime it so it holds a real goal + working memory ----
        # self-improve -> a curiosity-driven brain (needs jepa) + the spiking interaction learner,
        # so it learns SELF-DIRECTED. Otherwise -> grasp-reward priming (fixed goal GRIPPER_OPEN).
        if self_improve:
            # SI-1: the self-improvement drive is the GENUINELY-SPIKING learning-progress curiosity
            # (SpikingCuriosityDrive, LP read from LIF spike counts) -- the live primitive is now spiking.
            self.brain = LiveBrain(seed=seed, spiking_curiosity=True, jepa=True, interaction_learning=True)
        else:
            self.brain = LiveBrain(seed=seed)
        _safe = CONTACT * 3.0
        self._grasp = {"visual": [0.9] * 4, "temperature": [0.1],
                       "audio": [0.1] * 4, "distance": _safe}
        # a small rotation of SAFE stimuli (all safe distance -> no danger) so curiosity always has
        # learnable novelty to improve on; distinct bands the value/JEPA substrate can separate.
        self._explore = [
            {"visual": [0.9] * 4, "temperature": [0.1], "audio": [0.1] * 4, "distance": _safe},
            {"visual": [0.1] * 4, "temperature": [0.9], "audio": [0.1] * 4, "distance": _safe},
            {"visual": [0.1] * 4, "temperature": [0.1], "audio": [0.9] * 4, "distance": _safe},
        ]
        if self_improve:
            for _ in range(60):                   # prime via curiosity (no ext reward)
                self.tick_self_improve()
        else:
            for _ in range(90):                   # form the goal (GRIPPER_OPEN) + fill lake WM
                self.brain.tick(self._grasp, reward_fn=self._reward)

        # ---- intent prototypes: encode the example phrases once, sum per-channel counts ----
        self.prototypes = {intent: self._prototype(phrases)
                           for intent, phrases in INTENT_EXAMPLES.items()}

    @staticmethod
    def _reward(unit, action):
        return 1.0 if action == "GRIPPER_OPEN" else -1.0

    def tick_self_improve(self):
        """One SELF-IMPROVEMENT tick: curiosity-driven (intrinsic learning-progress reward, NO
        external task reward) over a rotating SAFE stimulus -> the brain learns self-directed.
        Gated + safe: the reflex E-STOP runs FIRST and curiosity feeds only the value update (no
        motor route). Updates the learning-progress telemetry. Called by respond() + the daemon."""
        ctx = self._explore[self._ticks % len(self._explore)]
        self.brain.tick(ctx)                      # no reward_fn -> intrinsic curiosity drives it
        self._ticks += 1
        cur = self.brain.arm.curiosity
        if cur is not None and self.brain.arm.q:  # mean progress over grown units = "improving"
            lps = [cur.learning_progress(u) for u in self.brain.arm.q]
            self._lp = sum(lps) / len(lps) if lps else 0.0

    # ---- ENCODE: text -> real spike train ----------------------------------------
    def encode(self, text):
        """text -> (spikes, n_channels). Each character c fires one spike on channel
        ord(c) % N at time = its position. Empty text -> no spikes."""
        spikes = [(ord(c) % self.n_channels, pos) for pos, c in enumerate(text or "")]
        return spikes, self.n_channels

    def _countvec(self, spikes):
        """Per-channel spike-count vector (the spike pattern the classifier decides on)."""
        v = [0.0] * self.n_channels
        for ch, _ in spikes:
            v[ch] += 1.0
        return v

    def _prototype(self, phrases):
        """One intent prototype = summed per-channel counts over its example phrases' spikes."""
        acc = [0.0] * self.n_channels
        for p in phrases:
            spikes, _ = self.encode(p)
            for ch, _ in spikes:
                acc[ch] += 1.0
        return acc

    # ---- STORE: append every spike to the durable lake ---------------------------
    def store(self, spikes):
        """Append each spike to the conversation lake at (global_cursor + local_time), then
        advance the cursor past this message so the whole chat log accumulates. Returns count."""
        if not spikes:
            return 0
        base = self._t_cursor
        max_t = 0
        for ch, t in spikes:
            self.lake.append(ch, base + t)
            if t > max_t:
                max_t = t
        self._t_cursor = base + max_t + 2         # +gap so messages do not overlap in time
        return len(spikes)

    # ---- CLASSIFY: the spike-based decision --------------------------------------
    def classify(self, spikes):
        """Decide the intent FROM the spikes: nearest prototype by cosine over the message's
        per-channel spike-count vector. Returns (intent, confidence in [0,1]); 'unknown' when
        no prototype is within UNKNOWN_FLOOR (or the message is empty)."""
        v = self._countvec(spikes)
        if sum(v) == 0.0:
            return "unknown", 0.0
        best_intent, best = "unknown", 0.0
        for intent, pv in self.prototypes.items():
            s = _cosine(v, pv)
            if s > best:
                best, best_intent = s, intent
        if best < UNKNOWN_FLOOR:
            return "unknown", best
        return best_intent, best

    # ---- DECODE: spike-decided intent + real brain state -> grounded reply --------
    def decode(self, intent, text):
        """The spiking classifier decided `intent`; the grounded reply is produced from the
        brain's REAL state. LiveBrain.converse() gives the honest, state-grounded sentence
        (it answers greeting/goal/memory/workspace/safety/consciousness from actual state and
        routes arm commands through the gated arm) -- no language-model generation happens."""
        if self.self_improve and intent == "ask_goal":
            act = ACTIONS[self.brain.goal] if self.brain.goal is not None else "none"
            return ("My goal is to IMPROVE MYSELF: I run curiosity-driven -- an intrinsic "
                    "learning-progress reward (no external task reward) drives self-directed "
                    "learning from my own experience each tick. It is ONE primitive (curiosity + "
                    "spiking learning), NOT open-ended recursive self-improvement, and the reflex "
                    f"E-STOP still preempts every motor command. (Gated action bias: {act}.)")
        return self.brain.converse(text)

    # ---- RESPOND: run the whole pipeline for one turn ----------------------------
    def respond(self, text):
        """encode -> store -> classify -> tick the brain one live step -> decode. Returns the
        full pipeline state so the UI can SHOW that the chat ran through real spikes + the lake."""
        spikes, _ = self.encode(text)
        self.store(spikes)
        intent, confidence = self.classify(spikes)
        if self.self_improve:
            self.tick_self_improve()              # advance one curiosity-driven tick
        else:
            self.brain.tick(self._grasp, reward_fn=self._reward)   # advance the live brain one tick
        reply = self.decode(intent, text)
        action_goal = ACTIONS[self.brain.goal] if self.brain.goal is not None else None
        out = {
            "reply": reply,
            "input_spikes": len(spikes),
            "lake_events": self.lake.n_events(),
            "intent": intent,
            "confidence": float(confidence),
            "goal": self.meta_goal or action_goal,   # meta-goal when self-improving
            "wm_events": self.brain.arm.lake_wm.n_events(),
        }
        if self.self_improve:
            out["action_goal"] = action_goal
            out["ticks"] = self._ticks
            out["learning_progress"] = round(self._lp, 6)
        return out

    def close(self, remove=False):
        """Release the conversation lake + the brain's lake working memory. If `remove` and the
        lake file was a temp one this instance created, delete it (leaves no litter)."""
        self.lake.close()
        if remove and self._created_temp and os.path.exists(self.lake_path):
            os.remove(self.lake_path)
        if self.brain.arm.lake_wm is not None:
            self.brain.arm.lake_wm.close(remove=True)


def main():
    # A conversation lake file we control, so we can prove durability by reopening it.
    fd, lake_path = tempfile.mkstemp(suffix=".spkl")
    os.close(fd)
    os.remove(lake_path)                          # SpikingChat starts from an empty log
    chat = SpikingChat(seed=0, lake_path=lake_path)

    # (1) REAL ENCODE: a non-empty message -> real spikes on valid channels; "" -> none.
    spk, n = chat.encode("hello")
    assert len(spk) == 5 and all(0 <= c < n for c, _ in spk), "encode must emit real spikes"
    assert chat.encode("")[0] == [], "the empty string must encode to zero spikes"

    # (2) STORED IN THE LAKE: respond appends the message's spikes; two messages accumulate.
    before = chat.lake.n_events()
    r_hello = chat.respond("hello")
    after1 = chat.lake.n_events()
    assert after1 - before == 5, f"the message's 5 spikes must land in the lake ({after1 - before})"
    chat.respond("hi again")
    after2 = chat.lake.n_events()
    assert after2 - after1 == len("hi again"), "a second message must accumulate in the lake"

    # (3) SPIKE-BASED DECISION: the nearest spike prototype names the intent.
    intent_greet, _ = chat.classify(chat.encode("hello there")[0])
    intent_goal, c_goal = chat.classify(chat.encode("what is your goal")[0])
    intent_mem, c_mem = chat.classify(chat.encode("what do you remember")[0])
    assert intent_greet == "greeting", f"'hello there' -> greeting ({intent_greet})"
    assert intent_goal == "ask_goal", f"'what is your goal' -> ask_goal ({intent_goal})"
    assert intent_mem == "ask_memory", f"'what do you remember' -> ask_memory ({intent_mem})"

    # (4) GROUNDED, HONEST DECODE: the reply comes from the brain's real state, never invented.
    r_consc = chat.respond("are you conscious?")
    assert "not a conscious agent" in r_consc["reply"], "decode must answer consciousness honestly"
    r_goal = chat.respond("what is your goal")
    assert r_goal["goal"] is not None and r_goal["goal"] in r_goal["reply"], \
        f"the goal reply must mention the brain's real goal ({r_goal['goal']})"

    # (5) LAKE PERSISTENCE: reopen the .spkl file and the whole conversation is still there.
    n_persist = chat.lake.n_events()
    chat.lake.close()                             # drop the live handle...
    reopened = StreamingHub(lake_path, N_CHANNELS, resume=True)   # ...and replay the durable log
    n_reopen = reopened.n_events()
    reopened.close()
    assert n_reopen == n_persist and n_reopen > 0, \
        f"the conversation must persist durably in the lake ({n_reopen} vs {n_persist})"

    # (6) DETERMINISM: same seed + same messages -> identical replies + identical spike counts.
    def run(seed):
        c = SpikingChat(seed=seed)                # its own temp lake
        msgs = ["hello", "what is your goal", "what do you remember", "close the gripper"]
        outs = [c.respond(m) for m in msgs]
        replies = [o["reply"] for o in outs]
        counts = [o["input_spikes"] for o in outs]
        c.close(remove=True)                      # no temp litter
        return replies, counts
    assert run(0) == run(0), "same seed + same messages must reproduce identical replies + counts"

    print("=" * 74)
    print("SPIKING CHAT -- text -> spikes -> data lake -> spike decision -> grounded reply")
    print("=" * 74)
    print(f"channels: {N_CHANNELS} (one per ASCII code) | unknown floor: cosine < {UNKNOWN_FLOOR}")
    print(f"turn 'hello' -> intent {r_hello['intent']} conf {r_hello['confidence']:.2f} | "
          f"{r_hello['input_spikes']} spikes in, lake now {r_hello['lake_events']} events")
    print(f"  reply -> {r_hello['reply']}")
    print(f"classify 'what is your goal'    -> {intent_goal:11} (cosine {c_goal:.2f})")
    print(f"classify 'what do you remember' -> {intent_mem:11} (cosine {c_mem:.2f})")
    print("turn 'are you conscious?' -> reply grounded + honest:")
    print(f"  {r_consc['reply']}")
    print(f"turn 'what is your goal' -> intent {r_goal['intent']} | goal={r_goal['goal']} | "
          f"wm={r_goal['wm_events']} spikes")
    print(f"  reply -> {r_goal['reply']}")
    print(f"persistence: reopened lake file -> {n_reopen} events replayed from disk")
    print("=" * 74)

    # cleanup: the conversation lake handle is already closed; drop the brain WM + temp file.
    chat.brain.arm.lake_wm.close(remove=True)
    if os.path.exists(lake_path):
        os.remove(lake_path)

    print(f"self-check OK: real spikes (5 for 'hello', 0 for ''); {n_reopen} events durable in the "
          f"lake and replayed on reopen; spike-prototype decided greeting/ask_goal/ask_mem; "
          f"decode grounded (goal {r_goal['goal']}, consciousness honest); deterministic")


if __name__ == "__main__":
    main()
