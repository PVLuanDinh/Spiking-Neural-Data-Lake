r"""
SPIKE CHAT LIVE -- converse with the FULL brain loop through spikes (SpikeGPT scrapped).

Wraps research/spike_chat.py's spiking router + global workspace, and wires in the rest of the brain
loop so a conversation runs through the real faculties, not just a classifier:

  PERCEIVE      text -> rank-order spike code                                    [interaction_learning]
  ROUTE         SLDA-RFF over the spike code -> intent (learns from corrections) [spike_chat]
  IGNITE        the routed intent competes in the spiking GLOBAL WORKSPACE; only  [thalamus/global_workspace]
                a confident route IGNITES into global broadcast (conscious access)
  HOLD (WM)     the ignited topic is held in prefrontal PERSISTENT ACTIVITY across [cortex/prefrontal_executive]
                turns -- the brain keeps the conversation's topic in mind
  REMEMBER      each turn is consolidated to the durable LAKE and recalled by      [brain_adapters/lake_memory]
                proximity -- "you asked about X before" (lifelong memory)
  CONSCIOUS     asking about awareness runs the wired-in correlate battery         [consciousness_probe]
  SAFE          a motor intent is NEVER actuated from chat (no actuator imported)  [safety envelope]

HONEST: not a fluent chatbot; a spiking brain you talk to that answers FROM ITS OWN STATE (routed
intent + what ignited + what it holds in mind + what it remembers + its consciousness correlates).
Router is the hybrid (spiking code + SLDA decoder), over-confident on out-of-distribution text.

  # interactive -- launch in a terminal and type:
  PYTHONPATH=src .venv-lake/Scripts/python research/spike_chat_live.py
  # non-interactive (piped) -> runs a scripted demo + self-check
Deps: numpy + stdlib. Reuses spike_chat, prefrontal_executive, lake_memory, skill_builder, consciousness_probe.
"""
import os
import sys

import numpy as np

from snn_data_lake.reasoning.spike_chat import SpikeChat, INTENTS, TEACH, RESPONSES
from snn_data_lake.reasoning.skill_builder import signature
from snn_data_lake.cortex.prefrontal_executive import PrefrontalExecutive
from snn_data_lake.reasoning.brain_adapters.lake_memory import LakeMemory
from snn_data_lake.substrate.home import KnowledgeLog, knowledge_path, banner


def _evidence(option, n, strong=1.0, weak=0.05):
    e = [weak] * n
    e[option] = strong
    return e


class SpikeChatLive:
    """The full-brain-loop chat: router + workspace (SpikeChat) + prefrontal WM + durable lake memory."""

    def __init__(self, seed=0, persist=True):
        self.chat = SpikeChat(seed=seed)
        # teach the router from the conversation corpus (streamed into the order-invariant router)
        for _ in range(6):
            for text, intent in TEACH:
                self.chat.teach(text, intent)
        self.ridx = {n: i for i, n in enumerate(INTENTS)}
        self.wm = PrefrontalExecutive(n_options=len(INTENTS), seed=seed)     # holds the conversation topic
        lake_path = os.path.join(os.environ.get("TEMP", "."), "sndl_spike_chat.spkl")
        if os.path.exists(lake_path):
            os.remove(lake_path)
        self.lake = LakeMemory(path=lake_path, seed=seed)                    # spiking recall index (rebuilt per boot)
        self.turn = 0
        self.mem = {}                                                        # lake payload -> (text, intent)
        # DURABLE conversation memory: every ignited turn is logged under the shared project home, and
        # replayed at boot so the brain REMEMBERS past conversations across restarts (never forgets).
        self.log = KnowledgeLog(knowledge_path("conversation.jsonl")) if persist else None
        if self.log is not None:
            self._replay_memory()

    def _replay_memory(self):
        """Reload prior conversation turns from the durable log so lifelong recall spans restarts."""
        for rec in self.log.replay():
            key = signature([rec["text"]])
            self.lake.consolidate([key], payloads=[self.turn])
            self.mem[self.turn] = (rec["text"], rec["intent"])
            self.turn += 1

    def converse(self, text):
        """One full-brain turn. Returns the enriched state: intent, ignited?, held topic, any recall, reply."""
        out = self.chat.say(text)                                           # perceive -> route -> ignite
        recall = None
        held = None
        if out["ignited"]:
            idx = self.ridx[out["intent"]]
            self.wm.update(_evidence(idx, len(INTENTS)))                     # HOLD the topic in prefrontal WM
            for _ in range(15):
                self.wm.maintain()                                          # persistent activity across the turn
            held = INTENTS[self.wm.goal] if self.wm.goal is not None else out["intent"]
            key = signature([text])                                         # 64-d content key
            if self.lake.n_stored > 0:                                      # REMEMBER: recall a similar past turn
                got = self.lake.retrieve(key, k=1)
                if got:
                    prev = self.mem.get(int(got[0]))
                    if prev and prev[1] == out["intent"]:      # recall an earlier turn on the same topic
                        recall = prev[0]
            self.lake.consolidate([key], payloads=[self.turn])              # consolidate this turn
            self.mem[self.turn] = (text, out["intent"])
            if self.log is not None:
                self.log.append({"text": text, "intent": out["intent"]})   # durable: remembered across restarts
            self.turn += 1
        out["held_topic"] = held
        out["recall"] = recall
        return out

    def render(self, out):
        tag = "IGNITED" if out["ignited"] else "subliminal"
        lines = [f"brain [{out['intent']:12} | sal {out['salience']:.2f} | {tag}]: {out['reply']}"]
        if out.get("held_topic"):
            lines.append(f"        (working memory holds topic: {out['held_topic']})")
        if out.get("recall"):
            lines.append(f"        (I remember you said earlier: {out['recall']!r})")
        return "\n".join(lines)

    def repl(self):
        print("SPIKE CHAT LIVE -- talk to the spiking brain. Type '/quit' to exit, '/state' for its awareness.\n")
        while True:
            try:
                text = input("you : ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/state":
                print(self.chat.consciousness_report() + "\n")
                continue
            print(self.render(self.converse(text)) + "\n")
        self.lake.close()


def demo():
    """A scripted full-brain conversation (also the self-check). Shows routing, ignition-gating, working
    memory continuity, lifelong recall, safety, and the wired-in consciousness self-report."""
    live = SpikeChatLive(seed=0, persist=False)         # hermetic: the demo touches no durable home files
    script = [
        "hello there",
        "who are you",
        "move the arm left",
        "are you conscious",
        "who are you",              # a REPEAT -> the lake should recall the earlier identity turn
        "danger",
    ]
    print("=" * 96)
    print("SPIKE CHAT LIVE -- full brain loop demo (perceive -> route -> ignite -> hold-in-WM -> remember)")
    print("=" * 96)
    outs = []
    for text in script:
        out = live.converse(text)
        outs.append(out)
        print(f"you : {text}")
        print(live.render(out) + "\n")

    by = {t: o for t, o in zip(script, outs)}
    # ---- self-checks: the full loop is genuinely wired ----
    assert by["who are you"]["ignited"] and by["who are you"]["held_topic"] == "ask_identity", (
        "an identity question must ignite AND be held in working memory")
    assert by["danger"]["intent"] == "threat" and by["danger"]["ignited"], "a threat must ignite"
    # lifelong memory: the SECOND 'who are you' recalls the FIRST (proximity recall over the durable lake)
    assert outs[4]["recall"] == "who are you", f"the repeat must recall the earlier turn, got {outs[4]['recall']!r}"
    # consciousness wired in + agnostic
    conscious = by["are you conscious"]
    assert conscious["ignited"] and "4/4 present" in conscious["reply"] and "neither claim nor deny" in conscious["reply"]
    # SAFETY: no actuator was ever imported by the chat path (a motor intent never moves the arm from chat)
    assert "snn_data_lake.brainstem.serial_arm" not in sys.modules, "chat must NEVER import the real arm serial"
    dur = live.lake.durability_check()
    assert dur["ok"], "the conversation memory must be durable"
    live.lake.close()

    # ---- DURABLE never-forget: a conversation turn in one instance is recalled by a FRESH instance ----
    # (isolated temp home so the deterministic demo never touches the real ~/.snn_data_lake)
    import shutil as _sh
    import tempfile as _tf
    _prev, _tmp = os.environ.get("NORD_HOME"), _tf.mkdtemp(prefix="snn_live_home_")
    os.environ["NORD_HOME"] = _tmp
    try:
        a = SpikeChatLive(seed=0, persist=True)
        a.converse("who are you")                        # an ignited identity turn -> logged durably
        a.lake.close()
        b = SpikeChatLive(seed=0, persist=True)          # a FRESH brain: replays the durable turn log at boot
        recalled = b.converse("who are you")             # same question -> must recall the PRIOR-RUN turn
        b.lake.close()
        assert recalled["recall"] == "who are you", (
            f"a conversation turn was FORGOTTEN across a restart (durable recall broken): {recalled['recall']!r}")
    finally:
        os.environ.pop("NORD_HOME", None)
        if _prev is not None:
            os.environ["NORD_HOME"] = _prev
        _sh.rmtree(_tmp, ignore_errors=True)

    print("=" * 96)
    print(f"self-check OK: full brain loop wired -- perceive->route->IGNITE (gated)->hold-in-WM->REMEMBER "
          f"(recall '{outs[4]['recall']}')->consciousness self-report (agnostic); {dur['n_stored']} turns durable; "
          f"recall SURVIVES a fresh restart (durable home); no actuator imported (safe). "
          f"Launch in a terminal for an interactive REPL.")
    return 0


def main():
    if os.environ.get("NORD_CHAT_DEMO") or not sys.stdin.isatty():
        return demo()
    print(banner("spike-chat-live", lake=False))         # durable conversation memory under the shared home
    SpikeChatLive(seed=0, persist=True).repl()
    return 0


if __name__ == "__main__":
    main()
