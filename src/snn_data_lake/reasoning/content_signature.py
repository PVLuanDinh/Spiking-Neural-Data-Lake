r"""
Content signature -- the domain-agnostic 64-dim bag-of-tokens key, PURE STDLIB.

Extracted from skill_builder so the two-world reasoning bridge (spike_chat_lake) can import it WITHOUT
pulling skill_builder's numpy chain (language_readout_slda etc.). crc32 (persistence-stable, never the
builtin hash) bins each token; different domains' tokens land in different bins, so LANGUAGE and MATH
contexts separate in one shared metric space. skill_builder re-exports these for back-compat.

Deps: stdlib only (zlib).
"""
import zlib

SIG = 64                                    # signature width = lake/proximity channel width N=64


def signature(context_tokens):
    """A 64-dim content signature of a context: crc32 bag-of-tokens, normalized to [0,1] like the
    lake's keys. Domain-agnostic and persistence-stable (crc32, not builtin hash)."""
    v = [0.0] * SIG
    for tok in context_tokens:
        v[zlib.crc32(str(tok).encode()) % SIG] += 1.0
    m = max(v) or 1.0
    return [x / m for x in v]


def _self_check():
    import sys
    a, b = signature(["dog", "cat"]), signature(["dog", "cat"])
    assert a == b and len(a) == SIG and max(a) == 1.0, "signature must be deterministic, width SIG, [0,1]"
    assert signature(["x"]) != signature(["y"]), "different tokens must land in (generally) different bins"
    print("content_signature self-check OK: deterministic crc32 bag-of-tokens, pure stdlib.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
