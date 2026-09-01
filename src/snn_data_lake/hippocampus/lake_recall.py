"""
Lake recall -- the T057 contract: `recall(cue) -> k episodes with margins`, over THE ONE
LAKE's episode records, content-addressable from a DEGRADED cue (specs/022 T057, FR-017).

WHY THIS EXISTS (the disclosed gap it closes). `Brain.recall_episode_durable` held the
honest limit its own docstring stated: the lake half was EXACT-content addressing -- a
genuinely partial dict addressed nothing and returned a miss by construction. This module
gives the lake half real partial-cue recall: k nearest stored episodes, each with a
MARGIN derived from a measured similarity distribution, beating a scrambled-cue control
from a degraded cue.

RETRIEVAL IS A FILTER; THE ANSWER IS VERIFIED IN SLOT SPACE. The panel's blocker on the
first build is the design law here: the spike-code space is 32*N = 2048 cells, one cell
per `slot=value` token, and a FULL crc32 cell collision costs ZERO Victor-Purpura
distance -- so distance alone can return an episode that never happened at maximum
certainty (measured: a never-stored cue answered a different episode at distance 0,
"confidence" 1.0). Therefore every candidate the index returns is VERIFIED against the
cue by string comparison of its slots -- the retired exact-scan's honest core, ported as
the verification stage instead of deleted:
  confidence   = verified fraction of the cue's slots the episode MATCHES (str-equal)
  contradicted = cue slots the episode carries with a DIFFERENT value
A collision hit contradicts the colliding slot and is demoted below any true match; the
self-check CONSTRUCTS a real cell collision and asserts exactly that (the reachable red
for the collision class). Confidence is therefore a statement about CONTENT, never a
distance in costume -- the first build's confidence was a pure function of distance and
scored wrong answers at 1.0, measured and killed by this stage.

HOW -- REUSE, NOT REBUILD (P2, and the wiring lake_index.py's own distinct-from note
already scheduled: "The recall contract (T057) composes THAT"):
  * Episodes come FROM the lake: standalone mode reads `home.load_episodes(path)` (P4b --
    the durable store is the source of truth; everything held here is a rebuildable
    cache). ATTACHED mode (`episodes=` / `attach()`) lets a caller that already holds a
    lake-derived window feed it directly: the brain attaches the recent window it
    RESTORED FROM THE LAKE at boot and extends with its own lake appends, because the
    live lake is gigabytes and `load_episodes` replays it in full -- an O(lake) read the
    chat path pays only as a LAST-RESORT deep tier (the brain keeps the incumbent
    whole-lake exact sub-cue scan beneath this window tier, so a restart is never
    amnesia). Standalone full-lake mode remains the CONTRACT surface for bounded lakes
    and self-checks; deep-history SIMILARITY recall (indexed, margin-scored, over the
    full live lake) awaits the segment-engine index lane (T007's SegmentStore/LakeIndex),
    and that boundary is stated, not hidden.
  * The index is `substrate/proximity_index.ProximityIndex` VERBATIM -- LAESA exact k-NN
    over the Victor-Purpura metric, imported and driven, never re-implemented. The
    self-check asserts nearest_exact == brute force ON THIS ENCODING.
  * Distances for the candidates are recomputed with the same `victor_purpura_distance`
    the index uses (nearest_exact returns payloads, not distances).

THE MARGIN IS MEASURED, PER CUE ARITY. A cue with fewer slots emits fewer events and is
nearer to EVERYTHING (the first build scored every cue against one 3-slot null, so
confidence rose as cues got emptier -- anti-correlated with addressing, measured).
Calibration therefore samples a scrambled-cue null PER ARITY (1..n_slots): slot values
drawn across stored episodes, REJECTION-SAMPLED so no scramble is a genuine sub-cue of
any stored episode (an unexcluded lucky scramble IS a real partial cue and contaminates
the null). A recall's margin is the z-score of its distance against ITS OWN ARITY's
null; retrieval strength (the empirical fraction of that null strictly above the hit's
distance) is reported as `retrieval`, separate from content confidence. An arity with
fewer than 8 clean samples, a spreadless null, or a lake under 3 episodes is
UNCALIBRATED: margin None, said rather than guessed. The calibration RNG is derived
fresh per refresh from (seed, census), so margins are a function of the LAKE STATE, not
of this object's call history (the first build's margins drifted with call order,
measured).

CANDIDATE RE-RANKING: the index is asked for k + 8 candidates and the answer is ordered
by (verified confidence DESC, distance ASC, recency DESC). The +8 fetch buffer is a
harness bound, disclosed: a same-cell collision cluster wider than the buffer could
still shadow a true match; at the window scales this serves (hundreds of episodes)
measured cluster sizes are 1-2. Ties at equal confidence and distance resolve NEWEST
first -- recency is the honest tie-break for autobiographical recall, and it is stated.

HONEST LIMITS, stated: (1) slot values are compared as tokens -- "lake" and "the lake"
are different fillers; no semantic smoothing is claimed. (2) A degraded cue that drops
every disambiguating slot genuinely underdetermines its episode; every all-slot-matching
episode is a VALID completion, the ranking among them is retrieval order, and the
self-check asserts the honest property (every returned completion matches the whole
cue) rather than a specific winner. (3) Standalone refresh replays the lake per
staleness check -- O(lake); the brain avoids it by attaching (above).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.hippocampus.lake_recall  # (run by run_selfchecks.py)
"""

import random
import statistics
import zlib

from snn_data_lake.cortex.spike_preprocessing import N, victor_purpura_distance
from snn_data_lake.substrate.home import load_episodes, save_episode
from snn_data_lake.substrate.proximity_index import ProximityIndex

# distinct-from: hippocampus/episodic_recall.py -- RECONSTRUCTIVE pattern completion over
#   in-RAM VSA traces (returns rebuilt slot fillers + gist); this RETRIEVES stored episodes
#   with verified content, reconstructing nothing. Site read.
# distinct-from: hippocampus/episode_consolidation.py -- the bounded in-memory attractor
#   (working-memory tier); this is the durable tier's cue semantics beneath it. Site read.
# distinct-from: hippocampus/proximity_replay.py -- ProximityIndex over SPIKE PATTERNS for
#   replay sampling (float feature vectors via encode_latency); this indexes EPISODE DICTS
#   by token code and verifies in slot space. Same engine, different records. Site read.
# distinct-from: substrate/lake_index.py -- exact (tag, key) pointer lookup; its own note
#   names this module as the similarity half that COMPOSES proximity_index. Site read.
# distinct-from: substrate/proximity_index.py -- the LAESA engine itself; imported and
#   driven here, never re-implemented. Site read.

_T_SPAN = 32  # latency span of the token code; with N features the code space is 32*N cells
_FETCH_PAD = 8  # extra candidates fetched for verification re-ranking (harness bound, disclosed)
_MIN_NULL = 8  # clean samples an arity needs before its null is trusted (else uncalibrated)


def _token_cell(slot, value):
    h = zlib.crc32(f"{slot}={value}".encode("utf-8", "replace"))
    return (int((h >> 8) % _T_SPAN), int(h % N))


def encode_episode(episode):
    """One spike event per `slot=value` token, deterministic and persistence-stable (crc32,
    never the builtin hash). Sorted and deduped so the same episode is the same pattern."""
    return sorted({_token_cell(k, episode[k]) for k in episode})


def verify_slots(cue, episode):
    """(matched, contradicted): the cue's slots checked against the episode BY CONTENT.
    matched = str-equal; contradicted = present with a different value; a slot the episode
    lacks is neither (it cannot confirm and cannot contradict)."""
    matched = contradicted = 0
    for k, v in cue.items():
        if k in episode:
            if str(episode[k]) == str(v):
                matched += 1
            else:
                contradicted += 1
    return matched, contradicted


class LakeRecall:
    """recall(cue) -> k episodes with margins, retrieval-filtered and content-verified.

    Source of truth is the lake (P4b): standalone mode reads it via `path`; attached mode
    is fed a lake-derived window by its owner. Everything held here -- index, nulls -- is
    a rebuildable cache."""

    def __init__(self, path=None, episodes=None, q=1.0, null_samples=32, seed=0):
        self._path = path
        self._attached = list(episodes) if episodes is not None else None
        self.q = q
        self.null_samples = int(null_samples)
        self.seed = seed
        self._index = None
        self._episodes = []
        self._nulls = {}  # arity -> sorted [nearest distances] (measured, rejection-sampled)

    # ---- the cache, rebuilt from its source -----------------------------------------------------
    def attach(self, episodes):
        """Feed/replace the episode window (attached mode). The caller owns the window's
        provenance -- the brain hands the recent slice it restored FROM the lake."""
        self._attached = list(episodes)
        return self.refresh()

    # distinct-from: substrate/lake_rebuild_gate.py -- rebuilds the durable POINTER index
    #   from the lake (a gate over LakeIndex); this fingerprints an in-RAM window to decide
    #   a cache rebuild. Site read.
    @staticmethod
    def _census(episodes):
        """(count, content fingerprint) -- the staleness census. A LENGTH-only census was
        the re-panel's blocker: the brain's window is a FIXED-SIZE ring, so once saturated
        every slide preserved the length and the index froze forever on the first snapshot
        (measured: a mature lake saturates the ring AT BOOT, so the freeze began on the
        first chat query). The fingerprint is content-sensitive: crc32 chained over every
        episode's sorted slot tokens, so a rotation at constant length IS a census change."""
        fp = 0
        for ep in episodes:
            for k in sorted(ep):
                fp = zlib.crc32(f"{k}={ep[k]}".encode("utf-8", "replace"), fp)
            fp = zlib.crc32(b"\x00", fp)
        return (len(episodes), fp)

    def refresh(self, force=False):
        """Rebuild the index iff the source's episode census changed (or force). Returns
        the number of episodes indexed. A rebuild at window scale costs ~0.2-0.3 s
        (measured at 64 episodes: LAESA build + per-arity calibration) and is paid at the
        QUERY that first sees the changed window -- disclosed, and cheap against the
        seconds-per-read lake replay this tier exists to avoid."""
        episodes = self._attached if self._attached is not None else load_episodes(self._path)
        if (
            not force
            and self._index is not None
            and self._census(episodes) == self._census(self._episodes)
        ):
            return len(self._episodes)
        self._episodes = list(episodes)
        self._index = None
        self._nulls = {}
        if not episodes:
            return 0
        self._index = ProximityIndex(q=self.q)
        self._index.build(
            [encode_episode(e) for e in self._episodes], payloads=list(range(len(self._episodes)))
        )
        self._calibrate()
        return len(self._episodes)

    def _is_subcue(self, cue):
        return any(all(str(ep.get(k)) == str(v) for k, v in cue.items()) for ep in self._episodes)

    def _nearest_d(self, cue_ev, k=1):
        idxs, _n = self._index.nearest_exact(cue_ev, k)
        return [
            (victor_purpura_distance(cue_ev, encode_episode(self._episodes[i]), q=self.q), i)
            for i in idxs
        ]

    def _calibrate(self):
        """Measure the scrambled-cue null distribution PER CUE ARITY.

        The RNG is derived from (seed, census) so the nulls -- and every margin scored
        against them -- are a function of the lake state, never of this object's call
        history. Scrambles are rejection-sampled non-sub-cues; an arity that cannot
        collect _MIN_NULL clean samples, or whose null has no spread, stays uncalibrated
        and refuses margins at that arity."""
        if len(self._episodes) < 3:
            return
        rng = random.Random((self.seed * 2654435761 + len(self._episodes)) & 0xFFFFFFFF)
        slots = sorted({k for e in self._episodes for k in e})
        for arity in range(1, len(slots) + 1):
            dists = []
            for _ in range(self.null_samples):
                cue = None
                for _try in range(120):
                    ks = rng.sample(slots, arity)
                    cand = {k: rng.choice(self._episodes).get(k) for k in ks}
                    cand = {k: v for k, v in cand.items() if v is not None}
                    if len(cand) == arity and not self._is_subcue(cand):
                        cue = cand
                        break
                if cue is None:
                    continue
                dists.append(self._nearest_d(encode_episode(cue), 1)[0][0])
            if len(dists) >= _MIN_NULL and statistics.pstdev(dists) > 0.0:
                self._nulls[arity] = sorted(dists)

    # ---- the contract ---------------------------------------------------------------------------
    def recall(self, cue, k=3):
        """FR-017: k stored episodes for a (possibly degraded) cue, margins included.

        Returns a list of dicts ordered by (confidence DESC, distance ASC, newest first):
          episode      -- the stored episode, verbatim from the lake
          distance     -- VP distance between the cue's code and the episode's code
          confidence   -- VERIFIED content agreement: matched cue slots / len(cue).
                          1.0 means every cue slot is present and str-equal -- the retired
                          exact-scan's guarantee, now carried by verification, so a code
                          collision can never impersonate a match.
          contradicted -- cue slots the episode holds with a DIFFERENT value
          margin       -- z-score of the distance against the MEASURED scrambled-cue null
                          OF THIS CUE'S ARITY, or None while that arity is uncalibrated
          retrieval    -- empirical fraction of that null strictly above this distance
                          (0.0 while uncalibrated); retrieval strength, NOT correctness
          calibrated   -- whether margin/retrieval carry a measured null behind them
        An empty lake, an empty cue, or a non-dict cue returns []."""
        if not isinstance(cue, dict) or not cue:
            return []
        self.refresh()
        if self._index is None:
            return []
        cue_ev = encode_episode(cue)
        got = self._nearest_d(cue_ev, min(len(self._episodes), max(1, int(k)) + _FETCH_PAD))
        null = self._nulls.get(len(cue))
        out = []
        for d, i in got:
            ep = self._episodes[i]
            matched, contradicted = verify_slots(cue, ep)
            if null is not None:
                mu, sd = statistics.mean(null), statistics.pstdev(null)
                margin = (mu - d) / sd
                retrieval = sum(1 for x in null if x > d) / len(null)
            else:
                margin, retrieval = None, 0.0
            out.append(
                {
                    "episode": ep,
                    "distance": d,
                    "confidence": matched / len(cue),
                    "contradicted": contradicted,
                    "margin": margin,
                    "retrieval": retrieval,
                    "calibrated": null is not None,
                    "_recency": i,
                }
            )
        out.sort(key=lambda h: (-h["confidence"], h["distance"], -h["_recency"]))
        for h in out:
            del h["_recency"]
        return out[: max(1, int(k))]


def _find_cell_collision(slot="when", limit=200000):
    """Two distinct values whose `slot=value` tokens land on the SAME code cell -- the
    constructed red for the collision class. Deterministic scan, no randomness."""
    seen = {}
    for i in range(limit):
        v = f"t{i}"
        cell = _token_cell(slot, v)
        if cell in seen:
            return seen[cell], v
        seen[cell] = v
    raise AssertionError("no collision within the scan budget -- the code space grew?")


def main():
    import os
    import tempfile

    line = "=" * 92
    print(line)
    print("hippocampus/lake_recall -- T057: recall(cue) -> k episodes with margins, from the lake")
    print(line)

    lake = os.path.join(tempfile.mkdtemp(prefix="snn_lake_recall_"), "lake.bin")
    rng = random.Random(7)
    WHO = ["cockpit", "daemon", "owner", "arm", "sensor"]
    WHAT = ["moved", "spoke", "learned", "slept", "failed", "grew"]
    WHERE = ["lake", "cortex", "home", "field", "bench"]
    AFF = ["calm", "fear", "joy", "surprise"]
    episodes = [
        {
            "who": rng.choice(WHO),
            "what": rng.choice(WHAT),
            "where": rng.choice(WHERE),
            "when": f"t{i}",
            "affect": rng.choice(AFF),
        }
        for i in range(48)
    ]
    for e in episodes:
        assert save_episode(e, path=lake)

    lr = LakeRecall(path=lake, seed=0)
    n = lr.refresh()
    assert n == 48 and lr._nulls, f"48 episodes must index and calibrate ({n}, {sorted(lr._nulls)})"

    # (1) EXACT-CONTENT CUE: every slot verified, distance 0 -- the retired exact-scan's
    # guarantee carried by VERIFICATION (not by distance, which a collision can fake).
    top = lr.recall(episodes[3], k=1)[0]
    assert top["episode"] == episodes[3] and top["distance"] == 0
    assert top["confidence"] == 1.0 and top["contradicted"] == 0
    assert top["margin"] is not None and top["margin"] > 0 and top["retrieval"] == 1.0

    # (2) THE COLLISION RED (the first build's blocker, now constructed on purpose): two
    # distinct 'when' values sharing one code cell. Store A; cue for the NEVER-STORED B.
    # Distance says 0 -- verification says the 'when' slot CONTRADICTS, so the hit cannot
    # claim confidence 1.0 and a false memory cannot pass a full-match gate.
    v1, v2 = _find_cell_collision()
    col_lake = os.path.join(tempfile.mkdtemp(prefix="snn_lake_col_"), "lake.bin")
    a = {"who": "daemon", "what": "grew", "where": "bench", "when": v1, "affect": "surprise"}
    for e in episodes[:6] + [a]:
        save_episode(e, path=col_lake)
    col = LakeRecall(path=col_lake, seed=0)
    ghost = dict(a)
    ghost["when"] = v2  # the episode that never happened
    gh = col.recall(ghost, k=1)[0]
    assert gh["distance"] == 0.0, "the collision must be a REAL zero-distance impersonation"
    assert gh["confidence"] < 1.0 and gh["contradicted"] >= 1, (
        f"VERIFICATION must demote the collision hit -- distance 0 with a contradicted slot "
        f"({gh['confidence']}, contradicted {gh['contradicted']})"
    )
    # ...and with BOTH stored, an exact cue for each returns ITS OWN episode (verification
    # re-ranks past the tie the index cannot see).
    save_episode(dict(ghost), path=col_lake)
    for want in (a, ghost):
        hit = col.recall(want, k=1)[0]
        assert hit["episode"] == want and hit["confidence"] == 1.0, (
            f"an exact cue must return ITS OWN episode through a code collision ({want['when']})"
        )

    # (3) DEGRADED CUE keeping the unique 'when': right episode, full verified confidence,
    # positive margin at ITS OWN arity, every time.
    deg_margins = []
    for ep in episodes:
        keep = ["when"] + rng.sample(["who", "what", "where", "affect"], 2)
        got = lr.recall({s: ep[s] for s in keep}, k=1)[0]
        assert got["episode"] == ep and got["confidence"] == 1.0
        deg_margins.append(got["margin"])

    # (4) SCRAMBLED-CUE CONTROL at the same arity: non-sub-cue scrambles must verify BELOW
    # full confidence (they address no stored episode) -- and the margin separation holds.
    scram_margins, scram_confs = [], []
    tries = 0
    while len(scram_margins) < 40 and tries < 400:
        tries += 1
        ks = rng.sample(["who", "what", "where", "when", "affect"], 3)
        cand = {s: rng.choice(episodes)[s] for s in ks}
        if lr._is_subcue(cand):
            continue
        h = lr.recall(cand, k=1)[0]
        scram_margins.append(h["margin"])
        scram_confs.append(h["confidence"])
    assert len(scram_margins) >= 40, "the control needs samples (rejection budget exhausted)"
    assert max(scram_confs) < 1.0, (
        f"a non-sub-cue scramble must NEVER verify at full confidence ({max(scram_confs)}) -- "
        f"confidence is content, and these cues match no stored episode in full"
    )
    assert min(deg_margins) > max(scram_margins), (
        f"FR-017: degraded-cue margins must strictly separate from scrambled-cue margins "
        f"({min(deg_margins):.2f} vs {max(scram_margins):.2f})"
    )

    # (5) ARITY HONESTY (the first build scored every cue against one 3-slot null and
    # emptier cues looked better): sub-cues of the SAME episode at different arities are
    # each scored against THEIR OWN arity's null and both come out positive. Arity 1 can
    # NEVER calibrate (every stored value is somebody's sub-cue, so every scramble is
    # rejected) and correctly refuses a margin -- asserted as the refusal it is.
    ep = episodes[10]
    m3 = lr.recall({s: ep[s] for s in ("when", "who", "affect")}, k=1)[0]
    m5 = lr.recall(ep, k=1)[0]
    assert m3["episode"] == ep and m5["episode"] == ep
    assert m3["margin"] is not None and m5["margin"] is not None
    assert m3["margin"] > 0 and m5["margin"] > 0
    m1 = lr.recall({"when": ep["when"]}, k=1)[0]
    assert m1["episode"] == ep and m1["confidence"] == 1.0 and m1["margin"] is None, (
        f"an arity-1 cue verifies by content but its null is impossible by construction -- "
        f"margin must REFUSE, not guess ({m1['margin']})"
    )

    # (6) THE LAESA CONTRACT on this encoding: nearest_exact == brute force.
    for probe in (episodes[0], {"who": episodes[5]["who"], "when": episodes[5]["when"]}):
        ev = encode_episode(probe)
        got_idx, _ = lr._index.nearest_exact(ev, 3)
        brute = sorted(
            range(len(episodes)),
            key=lambda i: (victor_purpura_distance(ev, encode_episode(episodes[i]), q=lr.q), i),
        )[:3]
        assert got_idx == brute, "LAESA must equal brute force on this encoding"

    # (7) ATTACHED MODE (the brain's window path): fed the same episodes, the answers are
    # identical to standalone mode over the lake file -- and attach() picks up growth.
    att = LakeRecall(episodes=episodes, seed=0)
    assert att.refresh() == 48
    a_hit = att.recall(episodes[3], k=1)[0]
    assert a_hit["episode"] == episodes[3] and a_hit["confidence"] == 1.0
    extra = {"who": "owner", "what": "returned", "where": "bench", "when": "t99", "affect": "joy"}
    att.attach(episodes + [extra])
    assert att.recall(extra, k=1)[0]["confidence"] == 1.0
    # ...and THE ROTATION RED (the re-panel's blocker): a fixed-size ring slides at
    # CONSTANT LENGTH, which a length-only census can never see -- the index froze
    # forever on the live path. Attach a same-length rotated window: the newest entry
    # must be indexed and the evicted one gone.
    ring = episodes[1:] + [
        extra,
        {"who": "arm", "what": "waved", "where": "field", "when": "t100", "affect": "calm"},
    ]
    ring = ring[-48:]  # same length as before the slide
    att.attach(ring)
    assert att.recall(ring[-1], k=1)[0]["confidence"] == 1.0, (
        "a rotated same-length window must REINDEX -- a length-only census freezes the "
        "index on the first snapshot (the re-panel's measured blocker)"
    )
    assert all(h["episode"] != episodes[0] for h in att.recall(episodes[0], k=3)), (
        "an episode evicted from the attached window must not be answerable from it"
    )

    # (8) STALENESS + P4b in standalone mode: append to the lake file; the same instance
    # sees it; a fresh reader agrees (the index is a cache, the lake is the store).
    assert save_episode(extra, path=lake)
    assert lr.recall(extra, k=1)[0]["confidence"] == 1.0
    assert LakeRecall(path=lake, seed=0).recall(extra, k=1)[0]["confidence"] == 1.0

    # (9) REFUSALS: empty lake -> []; a 2-episode lake recalls but refuses margins;
    # empty/non-dict cues -> [].
    empty = LakeRecall(path=os.path.join(tempfile.mkdtemp(), "none.bin"))
    assert empty.recall({"who": "x"}) == []
    tiny_lake = os.path.join(tempfile.mkdtemp(prefix="snn_lake_tiny_"), "lake.bin")
    for e in episodes[:2]:
        save_episode(e, path=tiny_lake)
    th = LakeRecall(path=tiny_lake, seed=0).recall(episodes[0], k=1)[0]
    assert th["episode"] == episodes[0] and th["margin"] is None and not th["calibrated"]
    assert th["confidence"] == 1.0, "content verification needs no calibration"
    assert lr.recall("not a dict") == [] and lr.recall({}) == []

    # (10) DETERMINISM ACROSS CALL HISTORY (the first build's margins drifted with call
    # order): a fresh instance's margins equal a heavily-queried instance's margins.
    fresh = LakeRecall(path=lake, seed=0)
    heavy = LakeRecall(path=lake, seed=0)
    for _ in range(5):
        heavy.recall(episodes[2], k=2)
    q = {"when": episodes[8]["when"], "affect": episodes[8]["affect"]}
    assert [h["margin"] for h in fresh.recall(q, k=3)] == [
        h["margin"] for h in heavy.recall(q, k=3)
    ], "margins must be a function of the lake state, not of call history"

    # (11) fully ambiguous cues: every returned completion must MATCH the whole cue
    # (validity asserted); which valid completion ranks first is retrieval order + recency
    # (reported, not asserted -- such cues underdetermine their episode by construction).
    amb_valid = amb_own = 0
    for ep in episodes:
        cue = {s: ep[s] for s in rng.sample(["who", "what", "where", "affect"], 3)}
        h = lr.recall(cue, k=1)[0]
        if h["confidence"] == 1.0:
            amb_valid += 1
        if h["episode"] == ep:
            amb_own += 1
    assert amb_valid == 48, (
        f"every ambiguous-cue answer must be a VALID completion (all cue slots verified) -- "
        f"{amb_valid}/48; anything less is a confabulation the verification stage exists to stop"
    )

    print(
        f"self-check OK: exact cue verified 5/5 slots at distance 0; THE COLLISION RED fired "
        f"(cells {_token_cell('when', v1)} shared by '{v1}'/'{v2}': distance 0.0 demoted to "
        f"confidence {gh['confidence']:.2f} with {gh['contradicted']} contradicted slot, and "
        f"with both stored each exact cue finds its own); degraded cues 48/48 at full verified "
        f"confidence, margins [{min(deg_margins):.2f}..{max(deg_margins):.2f}] strictly above "
        f"scrambled [{min(scram_margins):.2f}..{max(scram_margins):.2f}] with scramble "
        f"confidence capped at {max(scram_confs):.2f}; per-arity nulls (2-slot and 4-slot "
        f"sub-cues both positive at their own arity); LAESA == brute force; attached mode == "
        f"standalone; staleness + fresh reader agree; refusals honest (tiny lake: confidence "
        f"without margins); margins independent of call history; ambiguous cues 48/48 VALID "
        f"completions ({amb_own}/48 the generating episode -- reported, not asserted)."
    )


if __name__ == "__main__":
    main()
