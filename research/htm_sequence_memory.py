"""
HTM temporal memory — high-order sequence prediction over sparse codes (pure Python, no tensors).

The lake can store and recall spike patterns, but not yet PREDICT the next one from context. HTM temporal memory
(Numenta) does: each input is a sparse set of active COLUMNS; each column has several CELLS so the same column
can be active in different temporal CONTEXTS. A cell becomes predictive when one of its distal SEGMENTS (a set of
synapses onto other cells) sees enough active cells from the previous step. On the next input, predicted cells
activate context-specifically; an unpredicted column BURSTS (all cells fire = "unexpected") and grows a new
segment linking its winner cell to the prior context. The payoff is HIGH-ORDER prediction: two sequences that
share a middle (A B C D vs X B C E) are told apart because B/C are represented by different cells depending on
what came before -- a first-order model cannot. Bursting doubles as an anomaly score. Pure standard library,
deterministic. On-thesis with sparse_distributed_memory / drift_monitor / lca_sparse_coding.

ponytail: minimal faithful core -- fixed random SDR encoder (no spatial-pooler learning), binary synapses
(a segment fires when its overlap with the active cells >= threshold, no permanence floats), one segment grown
per bursting winner, plus Numenta's predicted-inactive punishment (drop a segment that predicted a column which
then didn't fire). Enough to demonstrate high-order prediction + anomaly; not the full Numenta permanence model.

  python -m research.htm_sequence_memory        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""


def encode(symbol, n_cols=256, w=5):
    """Deterministic sparse distributed representation: symbol -> w active columns of n_cols (near-orthogonal)."""
    import random
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(symbol))       # stable across runs (no builtin hash)
    return frozenset(random.Random(seed).sample(range(n_cols), w))


class TemporalMemory:
    """Columns x cells with distal segments; learns sequences, predicts the next SDR from temporal context."""

    def __init__(self, n_cols=256, cells_per_col=8, threshold=3):
        self.n_cols = n_cols
        self.cpc = cells_per_col
        self.threshold = threshold
        self.segments = {}                 # cell -> list of segments; each segment = set(presynaptic cells)
        self.reset()

    def reset(self):
        """Clear the running temporal context (call between separate sequences)."""
        self.active_cells = set()
        self.winner_cells = set()
        self.predictive_cells = set()

    def _cells(self, col):
        return range(col * self.cpc, col * self.cpc + self.cpc)

    def _best_winner(self, col):
        return min(self._cells(col), key=lambda c: len(self.segments.get(c, [])))   # least-used cell

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
                    self.segments[cell] = [seg for seg in self.segments.get(cell, [])
                                           if len(seg & prev_active) < self.threshold]

        new_active, new_winner = set(), set()
        for col in active_cols:
            predicted = [c for c in self._cells(col) if c in self.predictive_cells]
            if predicted:                                  # expected: only the predicted cells fire
                new_active.update(predicted)
                new_winner.update(predicted)
            else:                                          # surprise: burst the whole column
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
        self.predictive_cells = {                          # depolarize cells whose segment matches the context
            cell for cell, segs in self.segments.items()
            if any(len(seg & new_active) >= self.threshold for seg in segs)
        }
        return predicted_cols, anomaly

    def predicted_columns(self):
        return {c // self.cpc for c in self.predictive_cells}


def _overlap(a, b):
    return len(a & b) / max(len(a), len(b))


def main():
    tm = TemporalMemory()
    seqs = ["ABCD", "XBCE"]                                # shared middle "BC"; endings must be disambiguated
    sdr = {ch: encode(ch) for ch in "ABCDEX"}

    for _ in range(25):                                    # online learning passes
        for s in seqs:
            tm.reset()
            for ch in s:
                tm.compute(sdr[ch], learn=True)

    # after "A B C" the model must predict D (not E); after "X B C" it must predict E (not D)
    def predict_after(prefix):
        tm.reset()
        for ch in prefix:
            tm.compute(sdr[ch], learn=False)
        return tm.predicted_columns()

    pred_abc = predict_after("ABC")
    pred_xbc = predict_after("XBC")
    d_from_abc, e_from_abc = _overlap(pred_abc, sdr["D"]), _overlap(pred_abc, sdr["E"])
    e_from_xbc, d_from_xbc = _overlap(pred_xbc, sdr["E"]), _overlap(pred_xbc, sdr["D"])

    # anomaly: a well-learned step is low-anomaly; a novel transition bursts (high anomaly)
    tm.reset(); tm.compute(sdr["A"], learn=False)
    _, anom_known = tm.compute(sdr["B"], learn=False)      # A->B is trained
    tm.reset(); tm.compute(sdr["A"], learn=False)
    _, anom_novel = tm.compute(sdr["E"], learn=False)      # A->E never seen

    print("=" * 60)
    print("HTM TEMPORAL MEMORY — high-order sequence prediction over SDRs")
    print("=" * 60)
    print(f"trained: {seqs} (shared middle 'BC', {tm.cpc} cells/col)")
    print(f"after ABC -> D overlap {d_from_abc:.2f} vs E {e_from_abc:.2f} (want D)")
    print(f"after XBC -> E overlap {e_from_xbc:.2f} vs D {d_from_xbc:.2f} (want E)")
    print(f"anomaly: known A->B {anom_known:.2f} vs novel A->E {anom_novel:.2f}")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert d_from_abc > 0.6 and d_from_abc > 2 * e_from_abc, f"ABC must predict D not E ({d_from_abc:.2f}/{e_from_abc:.2f})"
    assert e_from_xbc > 0.6 and e_from_xbc > 2 * d_from_xbc, f"XBC must predict E not D ({e_from_xbc:.2f}/{d_from_xbc:.2f})"
    assert anom_novel > 0.8 and anom_known < 0.2, f"novel transition must burst, known must not ({anom_novel:.2f}/{anom_known:.2f})"
    assert predict_after("ABC") == pred_abc, "same context -> identical prediction (deterministic)"
    print(f"self-check OK: high-order disambiguation (ABC->D {d_from_abc:.2f}, XBC->E {e_from_xbc:.2f}, "
          f"both suppress the wrong ending); anomaly {anom_novel:.2f} vs {anom_known:.2f}; deterministic")


if __name__ == "__main__":
    main()
