"""G3 feed-arm contrast direction (red-first, 2026-10-29, not run; see the row's ARMS.md).

The fed / direct / norm-matched random-feed contrast: g3-growth-loop.md 2.10 and C15.
The trained runner and the N > 256 admission are OWED: G3 registers no name for them.
"""

from snn_data_lake.growth.feed_readout import feed_u_contrasts


def test_feed_contrast_is_one_sided_in_fed_favour():
    """A fed arm needing MORE units than direct and random feed is NOT MET, not degenerate."""
    fed = [18] * 8
    better = [9, 10, 9, 10, 9, 10, 9, 10]
    rows = feed_u_contrasts(fed, better, better)
    assert not any(row.met for row in rows.values())
    assert not any(row.degenerate for row in rows.values())
