r"""A/B instrument for the RankingDomain retirement (2026-08-11, family C8-1): the P2 debt
recorded in nightly_skills._NightKB's own docstring said the ADOPT-_NightKB / RETIRE-
RankingDomain resolution "owes an A/B on the three named differences". This runs it.

The three recorded differences, and what this instrument does with each:
  1. train batch 256 (_NightKB) vs 1024 (RankingDomain)  -> MEASURED here, equal exposure.
  2. router .test list te[:500] vs te[:2000]             -> a bounded-night ROUTER-sample
     budget, not head math; disclosed, out of this instrument's scope (the head is the claim).
  3. the too-few-evaluable-test carve-out                -> measured 2026-08-04 to fire at
     NEITHER shipped configuration (nightly_skills' own comment carries the sweep); nothing
     to A/B where it never runs.

Method: the retired RankingDomain's TRAINING SEMANTICS are RECONSTRUCTED here (full epoch per
visit, batch=1024 -- the gwt_deep_split_ab precedent: the retired side is rebuilt so the
comparison survives the deletion), against the live nightly_skills._NightKB driven at EQUAL
EXPOSURE (slice_n = len(train) per visit = one full pass, rolling). Same dataset (FB15k-237
when cached, else the bundled facts -- the print says which), same dim/seed/lr, same filtered
Hits@10 evaluator, 4 visits each.

  PYTHONPATH=src python tests/instruments/rankingdomain_nightkb_ab.py
"""

import sys

import numpy as np

from snn_data_lake.cortex.kb_ranking import DistMult, load_dataset
from snn_data_lake.reasoning.nightly_skills import _NightKB


def _hits10(model, te, known, rng, eval_n=250):
    idx = rng.permutation(len(te))[: min(eval_n, len(te))]
    hits = 0
    for j in idx:
        h, r, t = int(te[j, 0]), int(te[j, 1]), int(te[j, 2])
        sc = model.rank_tails(h, r)
        gold = sc[t]
        for o in known.get((h, r), ()):
            if o != t:
                sc[o] = -1e30
        hits += (int(np.sum(sc > gold)) + 1) <= 10
    return hits / max(1, len(idx))


def main():
    tr_s, te_s, offline = load_dataset()
    ent, rel = {}, {}

    def eid(e):
        return ent.setdefault(e, len(ent))

    def rid(r):
        return rel.setdefault(r, len(rel))

    tr = np.array([[eid(h), rid(r), eid(t)] for h, r, t in tr_s], dtype=np.int64)
    te = np.array(
        [[eid(h), rid(r), eid(t)] for h, r, t in te_s if h in ent and r in rel and t in ent],
        dtype=np.int64,
    )
    known = {}
    for h, r, t in np.vstack([tr, te]):
        known.setdefault((int(h), int(r)), set()).add(int(t))

    # --- reference: RankingDomain's training semantics, reconstructed (full epoch, batch 1024) ---
    ref = DistMult(len(ent), len(rel), dim=50, seed=0)
    ref_rng = np.random.RandomState(0)
    ref_traj = []
    for _ in range(4):
        ref.train_epoch(tr.copy(), lr=0.1, batch=1024)
        ref_traj.append(_hits10(ref, te, known, ref_rng))

    # --- survivor: the live _NightKB at EQUAL exposure (slice = one full pass per visit) ---
    kb = _NightKB("world-kb", dim=50, seed=0, eval_n=250, max_triples=0)
    sur_traj = []
    for _ in range(4):
        sur_traj.append(kb.study(len(kb.tr)))

    print("=" * 88)
    print("RankingDomain (reconstructed) vs _NightKB (live) -- equal exposure, filtered Hits@10")
    src = "bundled facts" if offline else "FB15k-237"
    print(f"dataset: {src}; train {len(tr):,} / test {len(te):,}")
    print(f"  reference (full epoch, batch 1024): {['%.3f' % x for x in ref_traj]}")
    print(f"  survivor  (rolling pass, batch 256): {['%.3f' % x for x in sur_traj]}")
    d = sur_traj[-1] - ref_traj[-1]
    print(f"  final delta (survivor - reference): {d:+.3f}")
    print("=" * 88)
    # The retirement's bar, pre-registered before the first run: the surviving head must not be
    # MEASURABLY WORSE at equal exposure -- within 0.05 absolute Hits@10 of the reference, or
    # above it. (Different batch sizes are different SGD trajectories; exact equality is not the
    # claim, "the same head" is.)
    assert sur_traj[-1] >= ref_traj[-1] - 0.05, (
        f"the surviving _NightKB head is measurably worse at equal exposure "
        f"({sur_traj[-1]:.3f} vs {ref_traj[-1]:.3f}) -- the RankingDomain retirement is NOT licensed"
    )
    print(
        "A/B OK: the adopted head is not measurably worse at equal exposure; retirement licensed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
