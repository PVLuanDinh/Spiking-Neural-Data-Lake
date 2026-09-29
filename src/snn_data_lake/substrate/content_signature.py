r"""
Content signature -- the domain-agnostic 64-dim bag-of-tokens key.

Extracted from skill_builder so the two-world reasoning bridge (spike_chat_lake) can import it WITHOUT
pulling skill_builder's numpy chain (language_readout_slda etc.). crc32 (persistence-stable, never the
builtin hash) bins each token; different domains' tokens land in different bins, so LANGUAGE and MATH
contexts separate in one shared metric space. skill_builder re-exports these for back-compat.

Deps: stdlib only (zlib).

PROMOTED reasoning/ -> hippocampus/ 2026-08-11 (#10 family settlement C1; measured row in
docs/P4C_BACKLOG.md). The key indexes memory/skill routing over the durable lake, which is the
hippocampal job here.
"""

import zlib

# PERSISTENCE-FORMAT CONSTANT, not a tunable (P1 exception 3, persisted-format shape): every
# signature ever written to the lake was binned % SIG, so changing SIG silently orphans every
# stored key -- it may only change with a lake migration. 64 matches the tree's N=64 channel
# convention (multimodal_encoder's 4x16 bands, streaming_hub's channel < 64 pack contract);
# no single named authority for that width exists yet -- recorded as sibling debt in the
# ledger row, not silently adopted here.
SIG = 64


def signature(context_tokens):
    """A 64-dim content signature of a context: crc32 bag-of-tokens, normalized to [0,1] like the
    lake's keys. Domain-agnostic and persistence-stable (crc32, not builtin hash)."""
    v = [0.0] * SIG
    for tok in context_tokens:
        v[zlib.crc32(str(tok).encode()) % SIG] += 1.0
    m = max(v) or 1.0
    return [x / m for x in v]


def _self_check():
    # distinct-from: schema_induction.py -- not a new checking mechanism; this module's own
    # _self_check strengthened in place at promotion (one new pinned assert, below).
    a, b = signature(["dog", "cat"]), signature(["dog", "cat"])
    assert a == b and len(a) == SIG and max(a) == 1.0, (
        "signature must be deterministic, width SIG, [0,1]"
    )
    assert signature(["x"]) != signature(["y"]), (
        "different tokens must land in (generally) different bins"
    )
    # PERSISTENCE-STABILITY, the headline property, with a REACHABLE red (added at promotion):
    # a same-process determinism compare (a == b above) stays GREEN if crc32 is swapped for
    # builtin hash() -- hash randomization varies across processes, not within one -- so only
    # a pinned cross-run constant reds the moment the hash function or SIG changes.
    assert zlib.crc32(b"dog") % SIG == 61 and zlib.crc32(b"cat") % SIG == 40, (
        "pinned crc32 bins moved -- the hash function or SIG changed, which orphans every "
        "signature already persisted to the lake"
    )
    print("content_signature self-check OK: deterministic crc32 bag-of-tokens, pinned bins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
