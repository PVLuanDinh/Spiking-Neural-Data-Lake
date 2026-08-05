"""
v0.37 -- closed-loop driver: ARM signal -> loop -> Interpreter -> reward -> loop.

Wires signal_loop's pieces (reflex, encode -> lake -> match, valence/dopamine, cortisol) to
the Interpreter end to end, IN-PROCESS, feeding the Interpreter's OUTCOME reward straight back
into dopamine + cortisol every tick. This is the schematic's full reaction loop on one machine
-- no FIFO/socket needed. The live deployment splits the same pieces across processes:

    signal_loop.py --serial COM3 --feedback | interpreter.py --pipe --outcome loop_in

WHERE THE LEARNED VALUE FUNCTION LIVES (P4b, corrected 2026-08-05): in `brain_lake.bin`, under a
`_T_STATE` record named "valence.json". It used to be a JSON SIDECAR of that name beside the lake,
which P4b forbids in its own words, and the self-check below made that worse rather than visible:
it was titled "DURABLE never-forget: the learned valence policy SURVIVES a restart" and proved
round-trip fidelity of the sidecar -- durability of the WRONG STORE, in the same process, which is
exactly the `retained() == 1.000` shape P4b was written from. The call sites did not change;
`substrate/home.save_state` did. The check now hands the lake to a CHILD INTERPRETER and makes it
reconstruct the policy from disk, because "it persists" is a claim about a NEW process.

Run:
  python -m snn_data_lake.brainstem.closed_loop        # scripted episode + self-check
"""

from snn_data_lake import arm_bridge as AB
from snn_data_lake import signal_loop as S
from snn_data_lake.brainstem.interpreter import Interpreter
from snn_data_lake.brainstem.reflex import Reflex
from snn_data_lake.substrate import home


def run_episode(
    signals,
    lib=None,
    assume_success=True,
    base_threshold=0.10,
    trace=None,
    bridge=None,
    valence_init=None,
):
    """Drive raw signal lines through the whole stack with live reward feedback.
    Returns the live state objects so callers can inspect what was learned.

    valence_init (a saved ValenceLearner.w list) seeds the value function from a PRIOR run, so the
    loop RESUMES what it learned instead of starting at zero -- the durable 'never forgets' path.
    None (the default) starts fresh, keeping the scripted self-check deterministic."""
    lib = S.build_default_library() if lib is None else lib
    matcher = S.build_matcher(lib)
    reflex = Reflex()
    valence = S.ValenceLearner()
    if valence_init is not None:  # resume a learned value function from disk
        valence.w = [float(x) for x in valence_init]
    cort = S.Cortisol(tau=30.0)
    itp = Interpreter(assume_success=assume_success)

    for raw in signals:
        p = S.parse_line(raw) if isinstance(raw, str) else raw
        if p is None:
            continue

        # reflex (threshold sharpened by cortisol) -- instinct preempts cognition
        reflex.threshold = base_threshold * cort.reflex_threshold_scale()
        ract = reflex.step(p)
        if ract:
            cmd, reward = itp.interpret({"reflex": ract})
            cort.step(1.0)  # danger = stressor (raises cortisol)
            # NOTE: we do NOT punish the previous gesture's value here -- a collision is its
            # own danger signal, not necessarily caused by the last action. Causal credit
            # assignment ("did that command cause the danger?") is a separate open problem.
            if trace is not None:
                trace.append(
                    {
                        "signal": "DANGER",
                        "command": cmd,
                        "reward": reward,
                        "stress": round(cort.level, 3),
                    }
                )
            continue

        # recognize -> valence overlay -> interpret -> command + reward
        window = S.normalize(S.resize_to_n(p))
        label, d, _, _ = matcher(window)
        v_action, v = valence.act(window, bias=cort.caution_bias())
        cmd, reward = itp.interpret(
            {
                "match": label,
                "instinct": v_action,
                "valence": round(v, 3),
                "stress": round(cort.level, 3),
            }
        )
        # feed the Interpreter's outcome straight back into dopamine + cortisol.
        # bridge set -> route the reward over the arm_bridge UDP back-channel first (the
        # split-process deployment: Interpreter and loop on different processes). Transparent.
        if reward is not None and bridge is not None:
            AB.send_outcome(reward, bridge.port, token=bridge.token)
            reward = bridge.recv_outcome()  # UDP back-channel: may be None on loss/timeout
        if reward is not None:
            valence.learn(window, reward, lr_scale=cort.learn_rate_scale(reward))
            cort.step(max(0.0, -reward))
        else:
            cort.step(0.0)  # no outcome (unmapped, or a dropped datagram) -> no learning
        if trace is not None:
            trace.append(
                {
                    "match": label,
                    "command": cmd,
                    "instinct": v_action,
                    "reward": reward,
                    "value": round(valence.valence(window), 3),
                    "stress": round(cort.level, 3),
                }
            )

    return {"valence": valence, "cortisol": cort, "reflex": reflex, "lib": lib}


def main():
    import os
    import sys

    lib = S.build_default_library()
    good_line = ",".join(f"{x:.4f}" for x in lib["ENGAGE"])
    good_win = S.normalize(S.resize_to_n([float(x) for x in good_line.split(",")]))
    danger = "500,500,500,500,500,500,950,0"

    # LAUNCH mode (snn-loop --persist): attach the shared durable home so the learned VALENCE POLICY
    # accumulates across runs -- the loop resumes what it learned instead of starting at zero. (Motor
    # commands are NEVER persisted or replayed; only the learned value function is durable.)
    if "--persist" in sys.argv:
        print(home.banner("closed-loop", lake=False))
        w0 = home.load_state("valence.json")  # prior learned values (None on first run)
        st = run_episode([good_line] * 30, lib=lib, valence_init=w0)
        home.save_state("valence.json", st["valence"].w)  # durable: remembered next launch
        print(
            f"[persist] {'resumed prior' if w0 else 'no prior'} valence, learned this run, saved -> "
            f"value(ENGAGE)={st['valence'].valence(good_win):+.2f} (accumulates across launches)"
        )
        return

    # episode: repeat the ENGAGE gesture (executed, rewarded), then a collision
    signals = [good_line] * 30 + [danger]
    trace = []
    state = run_episode(signals, lib=lib, assume_success=True, trace=trace)
    val, cort = state["valence"], state["cortisol"]

    print("=" * 60)
    print("CLOSED LOOP -- signal -> match -> command -> reward -> learn")
    print("=" * 60)
    first, last, collision = trace[0], trace[29], trace[30]
    print(f"first  : match={first['match']} cmd={first['command']} value={first['value']:+.2f}")
    print(
        f"learned: match={last['match']} cmd={last['command']} "
        f"value={last['value']:+.2f}  (dopamine raised it)"
    )
    print(
        f"danger : {collision['signal']} cmd={collision['command']} "
        f"reward={collision['reward']} stress={collision['stress']:.2f}"
    )
    print(
        f"final  : value(ENGAGE)={val.valence(good_win):+.2f} "
        f"({val.act(good_win)[0]})  stress={cort.level:.2f}  "
        f"reflex_thresh_scale={cort.reflex_threshold_scale():.2f}"
    )
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    cmds = [t.get("command") for t in trace]
    assert "ENGAGE" in cmds, "matched gesture never produced its command"
    assert "EMERGENCY_STOP" in cmds, "collision never produced an emergency stop"
    assert last["value"] > first["value"], "dopamine did not raise the rewarded gesture's value"
    assert val.act(good_win)[0] == "APPROACH", "rewarded gesture not learned as APPROACH"
    assert collision["stress"] > last["stress"] + 0.1, "collision did not raise cortisol stress"
    assert cort.reflex_threshold_scale() < 1.0, "stress did not sharpen the reflex"

    # split-process feedback: same loop with the reward routed over the arm_bridge UDP
    # back-channel must learn the SAME thing (transport transparent).
    br = AB.OutcomeBridge(timeout=1.0)
    bstate = run_episode([good_line] * 30, lib=lib, assume_success=True, bridge=br)
    br.close()
    assert bstate["valence"].act(good_win)[0] == "APPROACH", (
        "reward routed over the UDP back-channel did not close the loop"
    )
    print(
        f"bridged: OUTCOME routed over arm_bridge UDP -> "
        f"value(ENGAGE)={bstate['valence'].valence(good_win):+.2f} (loop still closes)"
    )

    # ---- DURABLE never-forget: the learned valence policy SURVIVES A REAL RESTART ----
    # (isolated temp home so the deterministic self-check never touches the real ~/.snn_data_lake)
    #
    # The reload runs in a SEPARATE PYTHON PROCESS. Until 2026-08-05 it ran here, in the same
    # interpreter that had just written the policy, which would pass identically against a pure
    # in-memory store -- and the store it was actually certifying was a JSON sidecar, not the lake.
    # P4b's honesty guard is the whole point: "it persists" is a claim about a NEW process.
    import shutil
    import subprocess
    import tempfile

    src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _prev, _tmp = os.environ.get("NORD_HOME"), tempfile.mkdtemp(prefix="snn_loop_home_")
    os.environ["NORD_HOME"] = _tmp
    try:
        s1 = run_episode([good_line] * 30, lib=lib, assume_success=True)  # learn from scratch
        wrote = home.save_state("valence.json", s1["valence"].w)  # INTO the one lake (_T_STATE)
        v_saved = s1["valence"].valence(good_win)
        assert wrote is True, "the first save of a learned policy must write a lake record"
        assert not os.path.exists(home.state_path("valence.json")), (
            "the learned policy must go IN brain_lake.bin -- no valence.json sidecar may be created"
        )
        assert os.path.exists(home.knowledge_path()), "the one lake was never created"

        # a child interpreter that has never seen s1 rebuilds the policy from brain_lake.bin alone
        child = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            f"import os; os.environ['NORD_HOME'] = {_tmp!r}\n"
            "from snn_data_lake.brainstem.closed_loop import run_episode\n"
            "from snn_data_lake.substrate import home\n"
            "from snn_data_lake import signal_loop as S\n"
            "w = home.load_state('valence.json')\n"
            "assert w is not None, 'the child read NO policy back from the lake'\n"
            "lib = S.build_default_library()\n"
            "win = S.normalize(S.resize_to_n(lib['ENGAGE']))\n"
            "s2 = run_episode([], lib=lib, valence_init=w)\n"
            "assert s2['valence'].act(win)[0] == 'APPROACH', s2['valence'].act(win)\n"
            f"assert abs(s2['valence'].valence(win) - {v_saved!r}) < 1e-9, s2['valence'].valence(win)\n"
            "print('%.6f' % s2['valence'].valence(win))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, (
            "the learned valence was FORGOTTEN across a real restart "
            f"(child rc={done.returncode}): {done.stderr.strip()[-400:]}"
        )
        print(
            f"durable: policy value(ENGAGE)={v_saved:+.4f} rebuilt as "
            f"{done.stdout.strip()} by a SEPARATE process, from brain_lake.bin alone"
        )
    finally:
        if _prev is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = _prev
        shutil.rmtree(_tmp, ignore_errors=True)

    print(
        "self-check OK: gesture -> command, reward raised its value to APPROACH, "
        "collision -> EMERGENCY_STOP + stress -> sharpened reflex, UDP back-channel transparent (loop closed); "
        "learned valence policy SURVIVES a restart -- stored IN the one lake (_T_STATE, P4b) and "
        "rebuilt by a SEPARATE process. Launch with --persist to accumulate it across runs."
    )


if __name__ == "__main__":
    main()
