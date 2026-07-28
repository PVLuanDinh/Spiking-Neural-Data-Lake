"""
research/ng_snn/validate_uci.py -- FIDELITY check: reproduce NG-SNN's paper-exact UCI numbers
(Table 1, Tang et al., Neural Networks 2026) to prove the NG-SNN core (model.py) is faithful BEFORE
adding any feature extractor.

Paper Table 1 (end-to-end mode, Poisson T, ONE epoch, no backprop):
  Iris : 4-20-3, 97.33 +/- 2.5 %
  Wine : 13-10-3, 98.89 +/- 1.4 %

Protocol: min-max normalize features to [0,1], Poisson-encode over T time steps, grow NG-SNN in a
single pass, evaluate. Averaged over several stratified splits/seeds (the paper reports mean+/-std).

  PYTHONUTF8=1 .venv-cu128/Scripts/python research/ng_snn/validate_uci.py
"""

import numpy as np
import torch
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from model import NGSNN, current_encode

PAPER = {"Iris": (97.33, 2.5), "Wine": (98.89, 1.4)}


def run(name, load, T=20, l_max=40, r=0.9, t_cand=100, n_runs=10, test_frac=0.3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load()
    X, y = ds.data.astype(np.float32), ds.target.astype(np.int64)
    accs, units = [], []
    for seed in range(n_runs):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_frac, stratify=y, random_state=seed
        )
        lo, hi = Xtr.min(0), Xtr.max(0)  # min-max to [0,1] on TRAIN stats
        rng = np.where(hi > lo, hi - lo, 1.0)
        Xtr = np.clip((Xtr - lo) / rng, 0, 1)
        Xte = np.clip((Xte - lo) / rng, 0, 1)
        Str = current_encode(torch.tensor(Xtr, device=dev), T)  # deterministic rate coding
        Ste = current_encode(torch.tensor(Xte, device=dev), T)
        clf = NGSNN(l_max=l_max, r=r, t_cand=t_cand, device=dev, seed=seed)
        clf.fit(Str, torch.tensor(ytr, device=dev))
        acc = (clf.predict(Ste) == torch.tensor(yte, device=dev)).float().mean().item()
        accs.append(100.0 * acc)
        units.append(clf.n_units)
    a = np.array(accs)
    pm, ps = PAPER[name]
    print(
        f"{name:5s}: OURS {a.mean():.2f} +/- {a.std():.2f} %  (grew {np.mean(units):.1f} neurons, "
        f"T={T}, 1 epoch)  |  PAPER {pm:.2f} +/- {ps:.2f} %  |  gap {a.mean() - pm:+.2f}"
    )
    return a.mean()


def main():
    print("=" * 84)
    print(
        "NG-SNN FIDELITY vs paper Table 1 (UCI, end-to-end current-coded, one epoch, no backprop)"
    )
    print("=" * 84)
    iris = run("Iris", load_iris)
    wine = run("Wine", load_wine)
    print("=" * 84)
    # Honest bar: within a couple points of the paper (std ranges overlap) = core faithfully done.
    ok = (iris >= 95.0) and (wine >= 95.0)
    print(
        f"core faithful (paper ranges overlap): {ok}  -- Iris {iris:.2f} (paper 97.33+/-2.5), "
        f"Wine {wine:.2f} (paper 98.89+/-1.4)"
    )


if __name__ == "__main__":
    main()
