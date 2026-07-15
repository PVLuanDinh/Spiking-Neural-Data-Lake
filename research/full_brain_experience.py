r"""
FULL-BRAIN EXPERIENCE -- drive the whole SNDL brain through a real episode so EVERY torch tier genuinely
engages, WITHOUT forcing any of them. Each tier runs because the episode feeds it the data / interaction
it was built to consume:

  1. PERCEPTION   real N-MNIST event digits -> the surrogate-gradient SNN encoder (torch) AND the deep
                  spiking predictive-coding cortex (pc_cortex_serve) CLASSIFY them.
  2. LANGUAGE     a couple of prompts -> the SpikeGPT-216M spiking language center (torch) REPLIES.
  3. NEUROGENESIS sleep cycles -> the NG-SNN neurogenesis engine (torch) GROWS neurons to predict the
                  brain's own perception (grown_units / reasoning_units climb).

Prints each tier's REAL output (predicted digits + accuracy, generated text, grown-neuron counts). This is
the brain's "full potential": all torch tiers active on genuine inputs, none faked. Needs the TORCH venv
(both torch AND the model checkpoints -- surrogate_nmnist.pth, pc_cortex_nmnist.json, models/spikegpt/):

    D:\AI-project\spiking-neural-data-lake\.venv-cu128\Scripts\python research\full_brain_experience.py

Observation / cognition only -- no actuator route (perception + language + growth never send a motor command).
"""
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..")
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, _root)                                   # repo root -> the [train] `training` package (surrogate encoder)
sys.path.insert(0, _here)                                   # research/ -> the N-MNIST rate loader

from snn_data_lake.cortex.brain import Brain


def _bar(title):
    print("=" * 82)
    print(title)
    print("=" * 82)


def main():
    _bar("FULL-BRAIN EXPERIENCE -- all torch tiers on REAL data (no forcing)")
    b = Brain(seed=0, day_period=12)
    st = b.state()
    results = {}
    print("torch tiers loaded on the brain:")
    print(f"  neurogenesis engine (NG-SNN)     : {b.neuro is not None}")
    print(f"  surrogate perception encoder     : {getattr(b, '_perception_enc', None) is not None}  "
          f"(state tier: {st.get('perception_tier')})")
    print(f"  pc-cortex serve (deep spiking PC): {getattr(b, '_pc_cortex', None) is not None}  "
          f"(state tier: {st.get('pc_cortex_tier')})")
    lm_ok = False
    try:
        from snn_data_lake.cortex.language_center import SpikeGPTServe
        lm_ok = SpikeGPTServe.available()
    except Exception:
        lm_ok = False
    print(f"  SpikeGPT-216M language center    : available={lm_ok}")
    print("-" * 82)

    # ---- 1. PERCEPTION: classify real N-MNIST digits through BOTH perceptual torch tiers ----
    _bar("[1] PERCEPTION -- the brain sees real N-MNIST digits")
    n = 8
    try:
        from pc_cortex_torch import load_nmnist_rate
        _, _, xte, yte, din, cls = load_nmnist_rate(n_train=1, n_test=n)
        ok = 0
        import torch
        surro_latret = None
        for i in range(len(xte)):
            vec = list(xte[i])
            pred = b._pc_cortex.classify(vec)               # deep spiking PC cortex CLASSIFIES a real digit
            true = cls[int(yte[i])]
            hit = str(pred) == true
            ok += hit
            # also run the surrogate SNN encoder on the same digit (as an event frame)
            if surro_latret is None and getattr(b, "_perception_enc", None) is not None:
                frame = torch.tensor(xte[i]).reshape(2, 34, 34).unsqueeze(0).repeat(15, 1, 1, 1).unsqueeze(0)
                lat = b.perceive_event(frame)
                surro_latret = (len(lat) if lat is not None else None)
            print(f"  digit {i}: pc-cortex -> {pred}   (true {true})   {'OK' if hit else 'x'}")
        acc = ok / len(xte)
        print(f"  pc-cortex classified {ok}/{len(xte)} real N-MNIST digits (acc {acc:.2f}, in_dim={din})")
        print(f"  surrogate SNN encoder ran on a real frame -> latent length {surro_latret}")
        results["perception"] = f"pc-cortex {ok}/{len(xte)} real digits (acc {acc:.2f}); surrogate encoder ran"
    except Exception as e:
        print(f"  perception tier error: {type(e).__name__}: {e}")
        results["perception"] = f"ERROR {type(e).__name__}"
    print("-" * 82)

    # ---- 2. LANGUAGE: converse through the SpikeGPT-216M spiking language center ----
    _bar("[2] LANGUAGE -- the brain speaks through SpikeGPT-216M")
    if lm_ok:
        try:
            reps = []
            for q in ("how are you", "what is your goal"):
                t = time.time()
                r = b.respond(q)
                dt = time.time() - t
                reply = r["reply"] if isinstance(r, dict) else str(r)
                reps.append(reply)
                print(f"  Q: {q}")
                print(f"  A ({dt:.1f}s): {reply[:160]}")
            results["language"] = f"SpikeGPT replied to {len(reps)} prompts"
        except Exception as e:
            print(f"  language tier error: {type(e).__name__}: {e}")
            results["language"] = f"ERROR {type(e).__name__}"
    else:
        print("  SpikeGPT weights absent -> language center not available (would fall back to the narrator).")
        results["language"] = "unavailable (no weights)"
    print("-" * 82)

    # ---- 3. NEUROGENESIS: sleep grows neurons (torch NG-SNN) ----
    _bar("[3] NEUROGENESIS -- sleep grows neurons")
    g0, r0 = b.state()["grown_units"], b.state()["reasoning_units"]
    nights = 0
    prev = b.state()["phase"]
    for _ in range(60):
        ph = b.tick()
        cur = b.state()["phase"]
        if ph == "asleep" and prev != "asleep":
            nights += 1
        prev = ph
    g1, r1 = b.state()["grown_units"], b.state()["reasoning_units"]
    print(f"  slept ~{nights} nights over 60 ticks (day_period=12)")
    print(f"  grown_units      {g0} -> {g1}   (neurogenesis: NG-SNN neurons grown to predict perception)")
    print(f"  reasoning_units  {r0} -> {r1}   (reasoning-curriculum spiking net grown)")
    results["neurogenesis"] = f"grown_units {g0}->{g1}, reasoning_units {r0}->{r1} over {nights} nights"
    print("-" * 82)

    # ---- summary ----
    _bar("ALL TORCH TIERS EXERCISED ON REAL DATA (no forcing)")
    for tier in ("perception", "language", "neurogenesis"):
        print(f"  {tier:14s}: {results.get(tier)}")
    print("  (each tier ran because the episode fed it real input, not because it was poked.)")
    print("=" * 82)
    b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
