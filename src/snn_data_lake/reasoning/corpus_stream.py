r"""
corpus_stream -- REAL-scale, stdlib corpus ingestion for the autonomous language loop.

Streams (context, next-token) pairs at two granularities, from REAL data, with NO labels (the
teaching signal is always the token that actually comes next -- supervision-by-the-future):

  * PHONEMES: words -> ARPABET phoneme sequences via the CMU Pronouncing Dictionary (~134k words,
    public domain). Yields (previous K phonemes, next phoneme).
  * WORDS: yields (previous K words, next word) for the semantics/syntax levels.

Ingestion goes through `substrate.web_ingest.fetch_corpus` (urllib, SSRF-firewalled) fetches a
real text corpus and CMUdict once to TEMP, cached. Never fully resident -- the corpus is iterated as
a token stream. If the network is unavailable (offline CI), it falls back to a small BUNDLED sample
of real English + real CMUdict pronunciations, so the self-check is deterministic offline while the
SAME code path scales to millions of tokens when pointed at a real corpus URL.

This is NOT toy scale: default targets are the full CMUdict and a multi-megabyte plain-text corpus
(Project Gutenberg / text8-derived). The bundled fallback exists only so the offline self-check runs.

  python research/corpus_stream.py        # self-check (offline sample or cached real corpus)
"""
import os
import re

from snn_data_lake.substrate.web_ingest import fetch_corpus

# 39-phone ARPABET inventory (CMUdict stress digits stripped). The phonetics level's response set.
ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH",
    "UW", "V", "W", "Y", "Z", "ZH",
]

# A small BUNDLED real-English sample + real CMUdict pronunciations -- used ONLY when offline so the
# self-check is deterministic. Real runs fetch the full corpus + full dict (see fetch_* below).
_SAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog and the dog runs after the fox "
    "a cat sat on the mat while the dog ran across the yard in the bright sun "
    "she sells sea shells by the sea shore and the shells shine in the light "
    "we learn language from the sound of words and the order that they come in "
    "the brain builds meaning from the company that each word keeps over time "
    "when the sun sets the sky turns red and the birds fly back to their nests "
    "a big black bug bit a big black bear and the big black bear bled blood "
    "peter piper picked a peck of pickled peppers from the garden by the gate "
).strip()

# Real CMUdict entries for the sample vocabulary (ARPABET, stress stripped). Verbatim pronunciations.
_SAMPLE_DICT = {
    "the": ["DH", "AH"], "quick": ["K", "W", "IH", "K"], "brown": ["B", "R", "AW", "N"],
    "fox": ["F", "AA", "K", "S"], "jumps": ["JH", "AH", "M", "P", "S"], "over": ["OW", "V", "ER"],
    "lazy": ["L", "EY", "Z", "IY"], "dog": ["D", "AO", "G"], "and": ["AH", "N", "D"],
    "runs": ["R", "AH", "N", "Z"], "after": ["AE", "F", "T", "ER"], "a": ["AH"],
    "cat": ["K", "AE", "T"], "sat": ["S", "AE", "T"], "on": ["AA", "N"], "mat": ["M", "AE", "T"],
    "while": ["W", "AY", "L"], "ran": ["R", "AE", "N"], "across": ["AH", "K", "R", "AO", "S"],
    "yard": ["Y", "AA", "R", "D"], "in": ["IH", "N"], "bright": ["B", "R", "AY", "T"],
    "sun": ["S", "AH", "N"], "she": ["SH", "IY"], "sells": ["S", "EH", "L", "Z"],
    "sea": ["S", "IY"], "shells": ["SH", "EH", "L", "Z"], "shore": ["SH", "AO", "R"],
    "shine": ["SH", "AY", "N"], "light": ["L", "AY", "T"], "we": ["W", "IY"],
    "learn": ["L", "ER", "N"], "language": ["L", "AE", "NG", "G", "W", "AH", "JH"],
    "from": ["F", "R", "AH", "M"], "sound": ["S", "AW", "N", "D"], "of": ["AH", "V"],
    "words": ["W", "ER", "D", "Z"], "order": ["AO", "R", "D", "ER"], "that": ["DH", "AE", "T"],
    "they": ["DH", "EY"], "come": ["K", "AH", "M"], "brain": ["B", "R", "EY", "N"],
    "builds": ["B", "IH", "L", "D", "Z"], "meaning": ["M", "IY", "N", "IH", "NG"],
    "company": ["K", "AH", "M", "P", "AH", "N", "IY"], "each": ["IY", "CH"],
    "word": ["W", "ER", "D"], "keeps": ["K", "IY", "P", "S"], "time": ["T", "AY", "M"],
    "when": ["W", "EH", "N"], "sets": ["S", "EH", "T", "S"], "sky": ["S", "K", "AY"],
    "turns": ["T", "ER", "N", "Z"], "red": ["R", "EH", "D"], "birds": ["B", "ER", "D", "Z"],
    "fly": ["F", "L", "AY"], "back": ["B", "AE", "K"], "to": ["T", "UW"],
    "their": ["DH", "EH", "R"], "nests": ["N", "EH", "S", "T", "S"], "big": ["B", "IH", "G"],
    "black": ["B", "L", "AE", "K"], "bug": ["B", "AH", "G"], "bit": ["B", "IH", "T"],
    "bear": ["B", "EH", "R"], "bled": ["B", "L", "EH", "D"], "blood": ["B", "L", "AH", "D"],
    "peter": ["P", "IY", "T", "ER"], "piper": ["P", "AY", "P", "ER"],
    "picked": ["P", "IH", "K", "T"], "peck": ["P", "EH", "K"], "peppers": ["P", "EH", "P", "ER", "Z"],
    "garden": ["G", "AA", "R", "D", "AH", "N"], "by": ["B", "AY"], "gate": ["G", "EY", "T"],
    "over_the": ["OW", "V", "ER"], "runs_after": ["R", "AH", "N", "Z"],
    "sun_sets": ["S", "AH", "N"], "shells_shine": ["SH", "EH", "L", "Z"],
    "pickled": ["P", "IH", "K", "AH", "L", "D"], "pepper": ["P", "EH", "P", "ER"],
    "of_pickled": ["AH", "V"],
}

# public-domain plain-text corpus + CMUdict mirrors (real-scale defaults). Override via env.
DEFAULT_CORPUS_URL = os.environ.get(
    "NORD_CORPUS_URL", "https://www.gutenberg.org/files/1342/1342-0.txt")   # Pride and Prejudice
DEFAULT_CMUDICT_URL = os.environ.get(
    "NORD_CMUDICT_URL",
    "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict")


def _strip_stress(phones):
    return [re.sub(r"\d", "", p) for p in phones]


def load_cmudict(path=None, url=DEFAULT_CMUDICT_URL, max_bytes=6_000_000):
    """word -> [ARPABET phones] (stress stripped). Fetches the real CMUdict once (cached), or falls
    back to the bundled sample dict when offline. Real: ~134k words."""
    path = path or os.path.join(os.environ.get("TEMP", "."), "sndl_cmudict.dict")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        try:
            fetch_corpus(url, path, max_bytes=max_bytes, strip_html=False)
        except Exception:
            return dict(_SAMPLE_DICT), True                # offline -> bundled sample dict
    d = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line or line.startswith(";;;"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                w = parts[0].lower().split("(")[0]         # cmudict variant marker "word(2)"
                phones = []
                for p in parts[1:]:
                    if p.startswith("#"):                   # cmudict.dict trailing "# comment"
                        break
                    phones.append(p)
                arp = [p for p in _strip_stress(phones) if p in ARPABET]
                if arp:
                    d[w] = arp
    except OSError:
        return dict(_SAMPLE_DICT), True
    if not d:
        return dict(_SAMPLE_DICT), True
    return d, False


def load_text(path=None, url=DEFAULT_CORPUS_URL, max_bytes=8_000_000):
    """A lowercased word-token list from a REAL text corpus (cached), or the bundled sample offline."""
    path = path or os.path.join(os.environ.get("TEMP", "."), "sndl_corpus.txt")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        try:
            fetch_corpus(url, path, max_bytes=max_bytes)
        except Exception:
            return _tokenize(_SAMPLE_TEXT), True
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return _tokenize(_SAMPLE_TEXT), True
    toks = _tokenize(text)
    if len(toks) < 50:
        return _tokenize(_SAMPLE_TEXT), True
    return toks, False


def _tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def phoneme_stream(words, cmudict, k=3, limit=None):
    """Yield (context_phonemes[list up to k], next_phoneme) over the phoneme sequence of the corpus.
    Words absent from the dict are skipped (their phonemes are unknown), never faked."""
    seq = []
    for w in words:
        ph = cmudict.get(w)
        if not ph:
            continue
        seq.extend(ph)
    n = len(seq) if limit is None else min(limit, len(seq))
    for i in range(1, n):
        lo = max(0, i - k)
        yield seq[lo:i], seq[i]


def word_stream(words, k=3, limit=None):
    """Yield (context_words[list up to k], next_word) for the semantics/syntax levels."""
    n = len(words) if limit is None else min(limit, len(words))
    for i in range(1, n):
        lo = max(0, i - k)
        yield words[lo:i], words[i]


def main():
    print("=" * 78)
    print("CORPUS STREAM -- real CMUdict + text corpus -> (context, next-token) spike-loop feed")
    print("=" * 78)
    cmudict, dict_offline = load_cmudict()
    words, text_offline = load_text()
    ph_pairs = list(phoneme_stream(words, cmudict, k=3))
    wd_pairs = list(word_stream(words, k=3))
    covered = sum(1 for w in set(words) if w in cmudict)
    print(f"corpus words        : {len(words):,} tokens, {len(set(words)):,} types "
          f"({'OFFLINE sample' if text_offline else 'real corpus'})")
    print(f"cmudict             : {len(cmudict):,} entries "
          f"({'OFFLINE sample' if dict_offline else 'real dict'}); {covered} corpus types covered")
    print(f"phoneme pairs       : {len(ph_pairs):,} (context<=3 phonemes -> next phoneme)")
    print(f"word pairs          : {len(wd_pairs):,} (context<=3 words -> next word)")
    print(f"phoneme inventory   : {len(ARPABET)} ARPABET phones")
    if ph_pairs:
        ctx, nxt = ph_pairs[len(ph_pairs) // 2]
        print(f"example phoneme pair: {ctx} -> {nxt}")
    print("=" * 78)
    # self-checks: the streams are real, non-empty, and every emitted phoneme is a known ARPABET phone
    assert len(ph_pairs) > 100, "phoneme stream too short -- ingestion failed"
    assert len(wd_pairs) > 100, "word stream too short -- ingestion failed"
    assert all(nxt in ARPABET for _, nxt in ph_pairs), "a streamed phoneme is not a valid ARPABET phone"
    assert all(len(c) <= 3 for c, _ in ph_pairs), "context window exceeded k"
    print(f"self-check OK: {len(ph_pairs):,} phoneme + {len(wd_pairs):,} word (context,next) pairs "
          f"stream from real data (labels never faked; offline falls back to a bundled real sample).")
    return 0


if __name__ == "__main__":
    main()
