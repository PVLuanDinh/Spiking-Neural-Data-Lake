r"""
KB RANKING HEAD -- open-vocab knowledge-base completion, the scale step past the bounded-tail
classifier in world_kb_stream.py.

When the tail vocabulary is huge (FB15k-237 has ~14.5k entities), a softmax classifier is hopeless.
The standard move is a RANKING head: learn entity + relation embeddings and SCORE every candidate
tail for a query (head, relation), then rank. This file trains a from-scratch DistMult scorer
(score(h,r,t) = <e_h, w_r, e_t>, Yang et al. 2015) with negative sampling, and evaluates with the
standard FILTERED ranking metrics (Hits@10, Hits@1, MRR) against a random-ranking baseline.

Real data: fetches FB15k-237 (train/test .txt, tab-separated head<TAB>rel<TAB>tail) via web_ingest
(cached), falling back to the bundled real facts from world_kb_stream when offline so the self-check
is deterministic. This is the general skill builder's world-KB COMPETENCE head at scale -- it plugs in
wherever the target vocabulary is open.

HONEST SCOPE: a compact numpy DistMult over a few epochs will not hit published SOTA (Hits@10 ~0.45
for tuned models); the claim is that a real ranking head learns real relational structure and beats
random ranking decisively, at open-vocab scale a classifier cannot touch.

  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.cortex.kb_ranking
  Knobs: NORD_DIM(50) NORD_EPOCHS(8) NORD_BATCH(1024) NORD_LR(0.1) NORD_EVAL(1500)
         NORD_FB_TRAIN(url) NORD_FB_TEST(url)
Deps: numpy (+ stdlib). Reuses world_kb_stream bundled facts as the offline fallback.

PROMOTED reasoning/ -> cortex/ 2026-08-11 (#10 batch-1, measured row in docs/P4C_BACKLOG.md).
The offline fallback lazily imports `world_kb_stream` -- a governed->reasoning reach when this
landed, CLOSED the same day by world_kb_stream's own promotion to cortex/ (family C2b): the
lazy import is now governed->governed and stays lazy only to keep load_dataset's numpy-free
callers light. Self-check measured at promotion: 12.5s with cached FB15k-237 (Hits@10 0.297,
431x random); offline fallback keeps it deterministic without network.
"""

import os

import numpy as np

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "DistMult.train_epoch": (
        "APPROXIMATES_GRADIENT",
        "hand-computes the symmetric trilinear GRADIENTS and applies them to the embeddings -- the "
        "module own comment says gradients. No torch is imported and there is no hidden layer to "
        "propagate through, so whether a shallow closed-form gradient sits inside the cosine clause "
        "scope is exactly the FR-002h question. Over-flagged until that lands: Standing Invariant 3 "
        "makes the confident negative the dangerous verdict",
    ),
}

_FB_TRAIN = os.environ.get(
    "NORD_FB_TRAIN",
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/train.txt",
)
_FB_TEST = os.environ.get(
    "NORD_FB_TEST",
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/test.txt",
)


def _envi(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _envf(k, d):
    v = os.environ.get(k)
    return float(v) if v else d


def _fetch_triples(url, name):
    import tempfile

    from snn_data_lake.sensorimotor.web_ingest import fetch_corpus

    # tempfile.gettempdir(), not os.environ.get("TEMP", ".") -- the Windows-only TEMP
    # variable is unset on Linux/containers, and the "." fallback wrote 22MB of corpus
    # cache into the REPO ROOT through a container bind mount (measured 2026-08-20; the
    # same defect class fixed in language_self_build / lake99 / autonomous_brain).
    path = os.path.join(tempfile.gettempdir(), f"sndl_{name}.txt")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        # FB15k-237 derives from Freebase, which was released CC-BY. The redistributing repo
        # (villmow/datasets_knowledge_embedding) states no licence of its own, so this asserts the
        # UPSTREAM terms. Override with SNDL_FB15K_LICENSE if your reading differs -- this is the
        # least certain licence claim in the tree and it is written down rather than assumed.
        fetch_corpus(
            url,
            path,
            max_bytes=40_000_000,
            strip_html=False,
            license_id=os.environ.get("SNDL_FB15K_LICENSE", "cc-by-4.0"),
        )
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                out.append((p[0], p[1], p[2]))
    return out


def load_dataset():
    """(train_triples, test_triples, offline?) as string triples. FB15k-237 if reachable, else the
    bundled real facts split 80/20."""
    try:
        tr = _fetch_triples(_FB_TRAIN, "fb_train")
        te = _fetch_triples(_FB_TEST, "fb_test")
        if len(tr) > 1000 and len(te) > 100:
            return tr, te, False
    except Exception:
        pass
    from snn_data_lake.sensorimotor.world_kb_stream import load_kb

    triples, _ = load_kb()
    import random

    random.Random(0).shuffle(triples)
    cut = int(len(triples) * 0.8)
    return triples[:cut], triples[cut:], True


class DistMult:
    def __init__(self, n_ent, n_rel, dim=50, seed=0):
        rng = np.random.RandomState(seed)
        self.E = (rng.randn(n_ent, dim) * (1.0 / np.sqrt(dim))).astype(np.float64)
        self.R = (rng.randn(n_rel, dim) * (1.0 / np.sqrt(dim))).astype(np.float64)
        self.rng = rng

    def _sig(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def train_epoch(self, tr, lr, batch):
        self.rng.shuffle(tr)
        for i in range(0, len(tr), batch):
            b = tr[i : i + batch]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            tn = self.rng.randint(0, self.E.shape[0], size=len(b))  # corrupted tails (negatives)
            eh, wr, et, en = self.E[h], self.R[r], self.E[t], self.E[tn]
            sp = np.sum(eh * wr * et, axis=1)  # positive scores
            sn = np.sum(eh * wr * en, axis=1)  # negative scores
            gp = (self._sig(sp) - 1.0)[:, None]  # d/ds -log sig(sp)
            gn = (self._sig(sn))[:, None]  # d/ds -log sig(-sn)
            # gradients (DistMult is symmetric trilinear)
            gEh = gp * (wr * et) + gn * (wr * en)
            gEt = gp * (eh * wr)
            gEn = gn * (eh * wr)
            gR = gp * (eh * et) + gn * (eh * en)
            np.add.at(self.E, h, -lr * gEh)
            np.add.at(self.E, t, -lr * gEt)
            np.add.at(self.E, tn, -lr * gEn)
            np.add.at(self.R, r, -lr * gR)

    def rank_tails(self, h, r):
        """Score every entity as a candidate tail for (h, r): scores = (E[h]*R[r]) . E^T."""
        return (self.E[h] * self.R[r]) @ self.E.T


def main():
    DIM = _envi("NORD_DIM", 50)
    EPOCHS = _envi("NORD_EPOCHS", 8)
    BATCH = _envi("NORD_BATCH", 1024)
    LR = _envf("NORD_LR", 0.1)
    EVAL = _envi("NORD_EVAL", 1500)

    tr_s, te_s, offline = load_dataset()
    ent = {}
    rel = {}

    def eid(e):
        return ent.setdefault(e, len(ent))

    def rid(r):
        return rel.setdefault(r, len(rel))

    tr = np.array([[eid(h), rid(r), eid(t)] for h, r, t in tr_s], dtype=np.int64)
    te = np.array(
        [[eid(h), rid(r), eid(t)] for h, r, t in te_s if h in ent and r in rel and t in ent],
        dtype=np.int64,
    )
    n_ent, n_rel = len(ent), len(rel)

    print("=" * 90)
    print(
        "KB RANKING HEAD -- open-vocab completion via a DistMult scorer (rank all tails, not classify)"
    )
    print("=" * 90)
    print(
        f"dataset: {'bundled real facts' if offline else 'FB15k-237'}; "
        f"{len(tr):,} train / {len(te):,} test triples; {n_ent:,} entities, {n_rel} relations"
    )

    # filtered ranking: know every true tail for a (h, r) so we do not penalize other correct answers
    known = {}
    for h, r, t in np.vstack([tr, te]):
        known.setdefault((int(h), int(r)), set()).add(int(t))

    model = DistMult(n_ent, n_rel, dim=DIM, seed=0)
    for _ep in range(EPOCHS):
        model.train_epoch(tr, LR, BATCH)

    # evaluate filtered Hits@k / MRR on a sample of test
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(te))[: min(EVAL, len(te))]
    hits10 = hits1 = 0
    rr = 0.0
    for j in idx:
        h, r, t = int(te[j, 0]), int(te[j, 1]), int(te[j, 2])
        scores = model.rank_tails(h, r)
        gold = scores[t]
        for other in known.get((h, r), ()):  # filter: remove other true tails
            if other != t:
                scores[other] = -1e30
        rank = int(np.sum(scores > gold)) + 1  # 1-based rank of the true tail
        rr += 1.0 / rank
        hits10 += rank <= 10
        hits1 += rank == 1
    n = len(idx)
    mrr, h10, h1 = rr / n, hits10 / n, hits1 / n
    # random-ranking baseline (expected)
    rnd_h10, rnd_mrr = 10.0 / n_ent, (np.log(n_ent) + 0.577) / n_ent  # ~harmonic/n

    print("-" * 90)
    print(
        f"  DistMult (dim {DIM}, {EPOCHS} epochs): Hits@10 {h10:.3f}  Hits@1 {h1:.3f}  MRR {mrr:.3f}  "
        f"[eval n={n}]"
    )
    print(f"  random-ranking baseline           : Hits@10 {rnd_h10:.4f}  MRR ~{rnd_mrr:.4f}")
    print(
        f"  lift over random: Hits@10 x{h10 / max(rnd_h10, 1e-9):.0f}, MRR x{mrr / max(rnd_mrr, 1e-9):.0f}"
    )
    print("-" * 90)
    print(
        "  the ranking head scales the world-KB skill to OPEN vocab (14k+ tails) a classifier cannot."
    )
    print("=" * 90)

    assert h10 > 10 * rnd_h10, (
        f"ranking head barely beats random Hits@10 ({h10:.3f} vs {rnd_h10:.4f})"
    )
    assert mrr > 10 * rnd_mrr, f"ranking head barely beats random MRR ({mrr:.3f} vs {rnd_mrr:.4f})"
    print(
        f"self-check OK: DistMult ranking head Hits@10 {h10:.1%}, MRR {mrr:.3f} on "
        f"{'FB15k-237' if not offline else 'bundled facts'} ({n_ent:,} entities) -- open-vocab KB "
        f"completion, {h10 / max(rnd_h10, 1e-9):.0f}x random. The world-KB skill's scale competence head."
    )
    return 0


if __name__ == "__main__":
    main()
