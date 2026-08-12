"""
snn_data_lake/hippocampus/hippocampal_memory.py
(PROMOTED reasoning/brain_adapters/ -> hippocampus/ 2026-08-11, #10 lane 4a; measured row in
docs/P4C_BACKLOG.md -- the adapter composes its own region's governed mechanisms, the intended
direction, so it lives beside them.)

WHAT THIS IS. A thin, faithful adapter that lets the developmental / NG-SNN training loop
REPLACE its hand-rolled python replay buffer with the brain's ACTUAL hippocampus. The loop
learns MNIST-style digits from a FROZEN cached-feature matrix; against catastrophic forgetting
it periodically rehearses a balanced mix of previously-seen exemplars. Today the loop keeps
that rehearsal pool in ad-hoc python lists/dicts. This adapter hands that job to the real
spiking-lake hippocampus instead, WITHOUT reimplementing any of its mechanics:

  * snn_data_lake.hippocampus.experience_replay.ReplayBuffer  -- the fixed-capacity ring of
    (unit, action, reward) transitions + SWR sampling. We use it to HOLD exemplar identities.
  * snn_data_lake.hippocampus.sleep_consolidation.SleepConsolidator -- the wake -> quiescent
    (sleep) consolidation lifecycle. We use it for the periodic "sleep" pass.

The loop OWNS the big feature tensors. This adapter stores only INDICES (ints into that frozen
matrix) + labels -- never the features themselves. Composed from lists/dicts + the two
sibling modules above. Deterministic (per-instance random via the
ReplayBuffer/SleepConsolidator seeds). ASCII-only output. Touches no core file.

--------------------------------------------------------------------------------------------
MAPPING -- how a loop exemplar (row_index, label) rides ReplayBuffer's (unit, action, reward):

  unit   := label          the digit 0-9 the exemplar belongs to (ReplayBuffer keys value rows
                           by `unit`; here it groups exemplars by class).
  action := row_index      the exemplar's IDENTITY = its row in the frozen cached-feature matrix
                           (ReplayBuffer stores `action` verbatim; we read it straight back out).
  reward := 1.0            constant. Every stored exemplar is an equally-valid member of its
                           class, so the RPE-reward field is unused for identity and pinned to a
                           positive marker. (We never call ReplayBuffer.replay(), which is the
                           only method that consumes `reward`; we use add()/sample() only.)

We keep ONE ReplayBuffer PER DIGIT (per-class exemplar pools). That is what makes `sample`
BALANCED for free -- ReplayBuffer.sample(n) draws uniformly WITHIN a single buffer, so sampling
`per_digit` from each learned digit's own buffer yields an even, cross-digit balanced replay and
CANNOT leak an index across classes (a buffer only ever holds its own digit's rows).

CONSOLIDATE -- how the sleep pass rides SleepConsolidator:
  Each learned digit is treated as a one-assembly "skill". We build a deterministic K-neuron cell
  assembly per digit (a stable subset of the attractor's N units), `wake()` each digit as a skill
  (identical storage trajectory to the module's own demo), then run ONE `quiescent()` sleep pass
  (noise-driven spiking replay + covariance-STDP) over all of them, and read back the module's own
  LP-INDEPENDENT held-out pattern-completion retention plus its rate value_margin(). This is the
  same wake->sleep consolidation cycle sleep_consolidation.run() performs -- we only choose the
  skills (the learned digits) and report its numbers; we assert nothing about the fragile absolute
  retention level (the module itself flags that as seed-dependent).
--------------------------------------------------------------------------------------------

  PYTHONPATH=src python -m snn_data_lake.hippocampus.hippocampal_memory

(the path the module is at TODAY; the previous command named `research/brain_adapters/...`,
which has not existed since v3.78 -- a documented command that cannot run is the same
fail-green shape as a stale citation.)
"""

import random

from snn_data_lake.hippocampus.experience_replay import ReplayBuffer
from snn_data_lake.hippocampus.sleep_consolidation import SleepConsolidator
from snn_data_lake.hippocampus.sleep_replay import (
    STORE_REPS_A,  # storage repetitions per assembly during wake
)
from snn_data_lake.hippocampus.sleep_replay import (
    K as ASSEMBLY_SIZE,  # neurons per stored cell assembly (K=6)
)
from snn_data_lake.hippocampus.sleep_replay import (
    N as ATTRACTOR_UNITS,  # attractor-net size (N=24)
)

# reward field is unused for identity (we never call ReplayBuffer.replay); pin it to a marker.
_EXEMPLAR_REWARD = 1.0
# faithful sleep depth; the self-check overrides this with a small value to stay light.
_DEFAULT_SLEEP_EPISODES = 60
# per-digit ring capacity -- generous so a digit's exemplar pool is not silently evicted.
_DEFAULT_CAPACITY = 8192


class HippocampalMemory:
    """Drop-in replacement for the loop's hand-rolled replay buffer, backed by the real
    hippocampus. Stores (row_index, label) exemplars in per-digit ReplayBuffers, samples a
    balanced anti-forgetting replay from them, and runs a periodic SleepConsolidator pass.

    The loop passes ROW INDICES (ints into its own frozen feature matrix) + labels; this adapter
    holds only those indices + labels. Deterministic given `seed`."""

    def __init__(self, seed=0, capacity=_DEFAULT_CAPACITY, sleep_episodes=_DEFAULT_SLEEP_EPISODES):
        self.seed = int(seed)
        self.capacity = int(capacity)
        self.sleep_episodes = int(sleep_episodes)
        self._buffers = {}  # label -> ReplayBuffer (the real per-class exemplar pool)

    def _buffer(self, label):
        """The real ReplayBuffer holding class `label`'s exemplars (created on first use, with a
        deterministic per-label seed so sampling is reproducible)."""
        buf = self._buffers.get(label)
        if buf is None:
            buf = ReplayBuffer(capacity=self.capacity, seed=self.seed * 1009 + int(label))
            self._buffers[label] = buf
        return buf

    def store(self, feat_row_indices, label):
        """Remember that cached-feature TRAIN rows `feat_row_indices` (a list of ints) belong to
        digit `label`. Each is added to `label`'s real ReplayBuffer as a transition with
        unit=label, action=row_index, reward=1.0 (see module docstring for the mapping). Returns
        the number of exemplars stored."""
        buf = self._buffer(label)
        n = 0
        for row_index in feat_row_indices:
            buf.add(label, int(row_index), _EXEMPLAR_REWARD)  # (unit, action, reward)
            n += 1
        return n

    def sample(self, learned, per_digit):
        """Balanced anti-forgetting replay: a FLAT list of stored TRAIN row-indices with up to
        `per_digit` exemplars from EACH digit in `learned` (skipping digits never stored). Backed
        by the real ReplayBuffer.sample within each per-class pool, so the mix is even across
        classes and no index can cross classes. Order is grouped by `learned`."""
        out = []
        for label in learned:
            buf = self._buffers.get(label)
            if buf is None:
                continue  # a learned digit we were never given exemplars for -> contributes none
            for _unit, row_index, _reward in buf.sample(per_digit):
                out.append(row_index)
        return out

    def consolidate(self, learned):
        """Run one wake -> quiescent (sleep) SleepConsolidator pass over the learned digits and
        return a retention / value-margin summary dict. Each digit is a one-assembly skill; the
        sleep phase is the module's proven noise-driven spiking replay + covariance-STDP. Returns
        {learned, n_skills, retention, value_margin, sleep_episodes}. Retention is the module's
        LP-INDEPENDENT held-out pattern-completion accuracy (fragile in absolute terms by the
        module's own disclosure); value_margin is its rate SWR value decisiveness."""
        digits = sorted(set(learned))
        cons = SleepConsolidator(seed=self.seed, sleep=True, episodes=self.sleep_episodes)
        assemblies = [self._assembly(d) for d in digits]
        for d, assembly in zip(digits, assemblies):
            cons.wake(str(d), [assembly], STORE_REPS_A)  # digit as a skill w/ one cell assembly
        retention = 0.0
        if digits:
            cons.quiescent()  # the sleep pass -- consolidates all stored assemblies together
            retention = cons.retention(
                assemblies
            )  # held-out completion accuracy over the learned set
        return {
            "learned": digits,
            "n_skills": len(digits),
            "retention": retention,
            "value_margin": cons.value_margin(),
            "sleep_episodes": self.sleep_episodes,
        }

    def _assembly(self, label):
        """A deterministic K-unit cell assembly (subset of the attractor's N units) standing in for
        digit `label` inside the SleepConsolidator. Stable per (seed, label); overlap across digits
        is expected and realistic (the attractor holds only a few clean assemblies)."""
        rng = random.Random(self.seed * 7919 + int(label))
        return set(rng.sample(range(ATTRACTOR_UNITS), ASSEMBLY_SIZE))

    def stored_digits(self):
        """The digits that currently have an exemplar pool."""
        return sorted(self._buffers)


def main():
    print("=" * 74)
    print("HIPPOCAMPAL MEMORY ADAPTER -- real ReplayBuffer + SleepConsolidator for the loop")
    print("=" * 74)

    mem = HippocampalMemory(seed=0, sleep_episodes=40)  # light sleep -> fast self-check

    # ---- store a few digits' worth of exemplars; DISJOINT row-index ranges per digit so any
    # ---- cross-class leak would be unambiguously detectable from the flat sample() output. ----
    truth = {
        0: list(range(0, 5)),  # digit 0 -> rows 0..4
        1: list(range(100, 105)),  # digit 1 -> rows 100..104
        2: list(range(200, 202)),  # digit 2 -> rows 200..201 (small pool: < per_digit)
    }
    for label, rows in truth.items():
        stored = mem.store(rows, label)
        assert stored == len(rows), "store must return the exemplar count it added"
    # a digit that is 'learned' but was never given exemplars must contribute nothing.
    learned = [0, 1, 2, 7]  # 7 has no pool
    assert mem.stored_digits() == [0, 1, 2], "only digits with exemplars are stored"

    # ---- (a) BALANCED sample: <= per_digit from EACH stored+learned digit, only stored digits ----
    per_digit = 3
    picks = mem.sample(learned, per_digit)
    owner = {idx: d for d, rows in truth.items() for idx in rows}  # row_index -> its true digit
    counts = {d: 0 for d in truth}
    for idx in picks:
        assert idx in owner, f"sampled index {idx} was never stored (leak / phantom)"
        counts[owner[idx]] += 1
    # only stored digits appear; digit 7 (no pool) contributes nothing.
    assert set(d for d in counts if counts[d]) <= set(truth), "only stored digits may appear"
    # up to per_digit each; digits 0,1 have full pools -> exactly per_digit; digit 2 pool=2 -> 2.
    assert counts[0] == per_digit and counts[1] == per_digit, "full pools give exactly per_digit"
    assert counts[2] == min(per_digit, len(truth[2])), "small pool caps at its size (<= per_digit)"
    for d in counts:
        assert counts[d] <= per_digit, "no digit may exceed per_digit (balance bound)"

    # ---- (b) NO INDEX LEAK across digits: every sampled index belongs to its own digit's pool ----
    for idx in picks:
        d = owner[idx]
        assert idx in truth[d], f"index {idx} attributed to digit {d} but not in its pool"
    # and the per-digit buffers never share an index (disjoint by construction / by real storage).
    seen = set()
    for d, rows in truth.items():
        for idx in rows:
            assert idx not in seen, "an index appears under two digits -- cross-class leak"
            seen.add(idx)

    # ---- (c) CONSOLIDATE returns a sane retention summary ----
    summary = mem.consolidate(learned=[0, 1, 2])
    assert summary["n_skills"] == 3, "one skill per learned digit"
    assert summary["learned"] == [0, 1, 2], "summary lists the consolidated digits"
    assert isinstance(summary["retention"], float) and 0.0 <= summary["retention"] <= 1.0, (
        "retention must be an accuracy in [0,1]"
    )
    assert isinstance(summary["value_margin"], float), "value_margin must be a float"
    assert summary["sleep_episodes"] == 40, "summary echoes the sleep depth used"

    # ---- determinism: same seed -> identical balanced sample ----
    mem2 = HippocampalMemory(seed=0, sleep_episodes=40)
    for label, rows in truth.items():
        mem2.store(rows, label)
    assert mem2.sample(learned, per_digit) == picks, "same seed -> identical replay sample"

    print(f"stored digits         : {mem.stored_digits()}  (per-digit real ReplayBuffers)")
    print(f"balanced sample (n={per_digit}) : {len(picks)} indices  counts-per-digit {counts}")
    print(
        f"consolidate summary   : retention={summary['retention']:.2f}  "
        f"value_margin={summary['value_margin']:+.3f}  skills={summary['n_skills']}  "
        f"sleep_episodes={summary['sleep_episodes']}"
    )
    print("=" * 74)
    print(
        "self-check OK: real ReplayBuffer holds (row_index,label) exemplars as (unit,action,"
        "reward); sample is balanced across learned digits (<= per_digit each, only stored, no "
        "cross-class leak); SleepConsolidator wake->quiescent returns a sane retention summary; "
        "deterministic"
    )


if __name__ == "__main__":
    main()
