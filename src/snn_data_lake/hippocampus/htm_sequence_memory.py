"""
HTM temporal memory + lake sequence-anomaly — predict the next spike-pattern from context.

The lake can store and recall spike patterns; HTM temporal memory (Numenta) PREDICTS the next one from context.
Each input is a sparse set of active COLUMNS; each column has several CELLS so the same column can be active in
different temporal CONTEXTS. A cell becomes predictive when a distal SEGMENT (synapses onto other cells) sees
enough active cells from the previous step; on the next input, predicted cells activate context-specifically and
an unpredicted column BURSTS (all cells fire = "unexpected"). Payoff = HIGH-ORDER prediction: sequences that
share a middle (A B C D vs X B C E) are told apart because B/C use different cells per context; a first-order
model cannot. Bursting doubles as an anomaly score.

PRODUCT integration (additive, opt-in, read-only): `SequenceAnomalyMonitor` runs the temporal memory over the
lake's OWN encoder (`spike_preprocessing.encode_latency`) -- each window's active features are the SDR columns.
It learns the normal SEQUENCE of windows and scores every transition, so a signal that is individually valid but
arrives in the wrong ORDER is flagged (temporal anomaly), complementing `drift_monitor`'s distributional drift
and `streaming_hub`'s onset filter. A pure observer: it never changes recognition or control. Deterministic.

Minimal faithful core -- binary synapses (a segment fires when overlap with active cells >= threshold,
no permanence floats), one segment grown per bursting winner, plus Numenta's predicted-inactive punishment (drop
a segment that predicted a column which then didn't fire). Not the full Numenta permanence / spatial-pooler model.

  python -m snn_data_lake.hippocampus.htm_sequence_memory        # (also run by run_selfchecks.py)
"""

import random

from snn_data_lake.cortex.spike_preprocessing import N, encode_latency


def encode(symbol, n_cols=256, w=5):
    """Deterministic sparse distributed representation: symbol -> w active columns of n_cols (near-orthogonal)."""
    seed = sum(
        (i + 1) * ord(ch) for i, ch in enumerate(symbol)
    )  # stable across runs (no builtin hash)
    return frozenset(random.Random(seed).sample(range(n_cols), w))


class TemporalMemory:
    """Columns x cells with distal segments; learns sequences, predicts the next SDR from temporal context."""

    def __init__(self, n_cols=256, cells_per_col=8, threshold=3):
        self.n_cols = n_cols
        self.cpc = cells_per_col
        self.threshold = threshold
        self.segments = {}  # cell -> list of segments; each segment = set(presynaptic cells)
        self.reset()

    def reset(self):
        """Clear the running temporal context (call between separate sequences)."""
        self.active_cells = set()
        self.winner_cells = set()
        self.predictive_cells = set()

    def _cells(self, col):
        return range(col * self.cpc, col * self.cpc + self.cpc)

    def _best_winner(self, col):
        return min(self._cells(col), key=lambda c: len(self.segments.get(c, [])))  # least-used cell

    def _matching_cell(self, col, context):
        """Cell in col whose existing segment best matches the prior context (stable reuse). None if no match."""
        best_cell, best_ov, best_seg = None, 0, None
        for c in self._cells(col):
            for seg in self.segments.get(c, []):
                ov = len(seg & context)
                if ov > best_ov:
                    best_cell, best_ov, best_seg = c, ov, seg
        return (best_cell, best_seg) if best_ov >= 1 else (None, None)

    def compute(self, active_cols, learn=True):
        """Advance one step on the given active columns. Returns (predicted_cols_before, anomaly)."""
        prev_active, prev_winner = self.active_cells, self.winner_cells
        predicted_cols = {c // self.cpc for c in self.predictive_cells}
        bursting = [col for col in active_cols if col not in predicted_cols]
        anomaly = len(bursting) / len(active_cols) if active_cols else 0.0

        # negative learning (Numenta predicted-inactive punishment): a cell that was predicted but whose column
        # did NOT activate made a wrong prediction in this context -> drop the segment that caused it. This is
        # what removes spurious cross-context segments and makes high-order disambiguation clean.
        if learn:
            for cell in self.predictive_cells:
                if cell // self.cpc not in active_cols:
                    self.segments[cell] = [
                        seg
                        for seg in self.segments.get(cell, [])
                        if len(seg & prev_active) < self.threshold
                    ]

        new_active, new_winner = set(), set()
        for col in active_cols:
            predicted = [c for c in self._cells(col) if c in self.predictive_cells]
            if predicted:  # expected: only the predicted cells fire
                new_active.update(predicted)
                new_winner.update(predicted)
            else:  # surprise: burst the whole column
                new_active.update(self._cells(col))
                # reuse the cell already representing this context (stable), else start a new one.
                # segments stay FIXED once grown (binary) -- union-expanding a reused segment would
                # accumulate distinct contexts and make it match several, destroying the disambiguation.
                cell, _seg = self._matching_cell(col, prev_active)
                if cell is None:
                    cell = self._best_winner(col)
                    if learn and prev_winner:
                        self.segments.setdefault(cell, []).append(frozenset(prev_winner))
                new_winner.add(cell)

        self.active_cells, self.winner_cells = new_active, new_winner
        self.predictive_cells = {  # depolarize cells whose segment matches the context
            cell
            for cell, segs in self.segments.items()
            if any(len(seg & new_active) >= self.threshold for seg in segs)
        }
        return predicted_cols, anomaly

    def predicted_columns(self):
        return {c // self.cpc for c in self.predictive_cells}


class SequenceAnomalyMonitor:
    """Opt-in lake observer: learn the normal SEQUENCE of encoded windows, score each transition's anomaly.

    Drop-in for signal_loop's per-window stream. Complements drift_monitor (distributional drift) with TEMPORAL
    anomaly: a signal that is individually valid but arrives out of ORDER bursts. Read-only -- observes the window
    stream, never alters recognition or control."""

    def __init__(self, n_features=N, cells_per_col=8, threshold=3):
        self.tm = TemporalMemory(
            n_cols=n_features, cells_per_col=cells_per_col, threshold=threshold
        )

    def reset(self):
        self.tm.reset()

    def observe(self, window, learn=True):
        """Feed one feature window; returns its anomaly (fraction of active columns that were unpredicted)."""
        cols = frozenset(
            f for _t, f in encode_latency(window)
        )  # active features = SDR columns (lake encoder)
        _, anomaly = self.tm.compute(cols, learn=learn)
        return anomaly


def _overlap(a, b):
    return len(a & b) / max(len(a), len(b))


def _gesture(active):
    v = [0.0] * N
    for i in active:
        v[i] = 1.0
    return v


def main():
    # ---- 1. high-order prediction: ABCD vs XBCE (shared middle, disambiguated by context) ----
    tm = TemporalMemory()
    sdr = {ch: encode(ch) for ch in "ABCDEX"}
    for _ in range(25):
        for s in ("ABCD", "XBCE"):
            tm.reset()
            for ch in s:
                tm.compute(sdr[ch], learn=True)

    def predict_after(prefix):
        tm.reset()
        for ch in prefix:
            tm.compute(sdr[ch], learn=False)
        return tm.predicted_columns()

    d_from_abc = _overlap(predict_after("ABC"), sdr["D"])
    e_from_abc = _overlap(predict_after("ABC"), sdr["E"])
    e_from_xbc = _overlap(predict_after("XBC"), sdr["E"])

    # ---- 2. PRODUCT integration: sequence-anomaly over the lake's OWN encoder (encode_latency) ----
    g = {
        "A": _gesture(range(0, 6)),
        "B": _gesture(range(10, 16)),
        "C": _gesture(range(20, 26)),
        "D": _gesture(range(30, 36)),
        "E": _gesture(range(40, 46)),
    }  # E = a valid window never seen after C
    mon = SequenceAnomalyMonitor()
    for _ in range(25):  # learn the normal order A->B->C->D
        mon.reset()
        for name in "ABCD":
            mon.observe(g[name], learn=True)
    mon.reset()
    known = [
        mon.observe(g[name], learn=False) for name in "ABCD"
    ]  # [0]=A start (high); B,C,D predicted (low)
    trained_transition = sum(known[1:]) / 3.0
    mon.reset()
    for name in "ABC":
        mon.observe(g[name], learn=False)
    out_of_order = mon.observe(g["E"], learn=False)  # valid window, wrong order -> bursts

    print("=" * 60)
    print("HTM TEMPORAL MEMORY — high-order prediction + lake sequence-anomaly")
    print("=" * 60)
    print(f"high-order : ABC->D {d_from_abc:.2f} (E {e_from_abc:.2f}); XBC->E {e_from_xbc:.2f}")
    print(f"lake monitor (encode_latency): trained transitions anomaly {trained_transition:.2f}")
    print(f"out-of-order window after ABC: anomaly {out_of_order:.2f} -> TEMPORAL ANOMALY flagged")
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert d_from_abc > 0.6 and d_from_abc > 2 * e_from_abc, (
        f"ABC must predict D not E ({d_from_abc:.2f}/{e_from_abc:.2f})"
    )
    assert e_from_xbc > 0.6, f"XBC must predict E ({e_from_xbc:.2f})"
    assert trained_transition < 0.2, (
        f"learned transitions must be low-anomaly ({trained_transition:.2f})"
    )
    assert out_of_order > 0.8, (
        f"a valid window in the wrong order must burst as a temporal anomaly ({out_of_order:.2f})"
    )
    mon2 = SequenceAnomalyMonitor()
    assert mon2.observe(g["A"], learn=False) == SequenceAnomalyMonitor().observe(
        g["A"], learn=False
    ), "deterministic"
    print(
        f"self-check OK: high-order ABC->D {d_from_abc:.2f}/XBC->E {e_from_xbc:.2f}; lake monitor learns order "
        f"(transitions {trained_transition:.2f}) and flags out-of-order window ({out_of_order:.2f}); deterministic"
    )


if __name__ == "__main__":
    main()
