r"""
WEB ACCESS GATE -- the ONE controlled doorway through which the brain loop pulls knowledge from the
web to self-enrich, with WordNet as its first source.

The brain converses from its lifelong LAKE (reasoning/spike_chat_lake.py). This gate lets the loop GROW
that lake from real online knowledge, safely. Every fetch passes THREE checks before a byte is read:

  1. HOST ALLOWLIST -- only curated knowledge sources (WordNet/dictionary/corpus mirrors), nothing else.
  2. SSRF FIREWALL  -- reuses substrate.web_ingest (http/https only; blocks file://, localhost, private
     and link-local IPs, redirects to private hosts). No access to the local machine or internal network.
  3. SIZE CAP       -- a hard byte ceiling per fetch; read-only (GET), never posts or sends anything out.

Every fetch is LOGGED (host, bytes, source) so the brain's web access is auditable. If the network is
unavailable the gate falls back to a bundled real sample, so the self-check is deterministic offline.

FIRST SOURCE -- WordNet (Princeton): fetch `data.noun` from a jsDelivr npm mirror (wordnet-db), parse
each synset's word + gloss, and INGEST `word -> definition` into the brain's lake. Then spike_chat_lake
answers "what is X" from the real dictionary, and the database grows to tens of thousands of words.

HONEST: this is a KNOWLEDGE gate (fetch text -> parse -> store facts), not general web browsing or code
execution. It reads allowlisted static knowledge files and turns them into lake facts, nothing more.

  PYTHONPATH=src .venv-lake/Scripts/python -m snn_data_lake.reasoning.web_ingest_gate  # gate + WordNet demo
  Knobs: NORD_WORDNET_URL(jsdelivr data.noun) NORD_WN_LIMIT(4000) NORD_WEB_MAXBYTES(16000000)
Deps: stdlib (+ numpy via spike_chat_lake). Reuses substrate.web_ingest (SSRF firewall) + spike_chat_lake.
"""

import json
import os
import urllib.request
from urllib.parse import quote, urlparse

from snn_data_lake.substrate.web_ingest import fetch_text

# ---- the GATE's host allowlist: only these knowledge sources may be fetched (on top of the SSRF firewall) ----
ALLOWLIST = {
    "cdn.jsdelivr.net",
    "raw.githubusercontent.com",
    "unpkg.com",
    "wordnetcode.princeton.edu",
    "www.gutenberg.org",
    "gutenberg.org",
    "mattmahoney.net",
    "en.wikipedia.org",
    "simple.wikipedia.org",  # scholarly encyclopedia (REST summary API)
    "s3.amazonaws.com",  # ConceptNet dumps (relational triples, CC BY-SA)
    "www.wikidata.org",  # Wikidata entities (labels + descriptions, CC0)
}
DEFAULT_WORDNET_URL = os.environ.get(
    "NORD_WORDNET_URL", "https://cdn.jsdelivr.net/npm/wordnet-db@3.1.14/dict/data.noun"
)
DEFAULT_MAXBYTES = int(os.environ.get("NORD_WEB_MAXBYTES", "16000000"))

# bundled REAL definitions -- the offline fallback so the gate + self-check work with no network.
_BUNDLED = {
    "water": "a clear colourless liquid that forms rivers, lakes, and rain and is essential to life",
    "fire": "the rapid oxidation of a material producing heat, light, and flame",
    "sun": "the star at the centre of the solar system that gives Earth light and heat",
    "moon": "the natural satellite that orbits the Earth, seen by reflected sunlight",
    "tree": "a tall woody plant with a trunk, branches, and leaves",
    "river": "a large natural stream of water flowing toward a sea, lake, or another river",
    "mountain": "a large natural elevation of the earth's surface rising steeply from its surroundings",
    "book": "a set of written or printed pages bound together along one edge",
    "music": "an art of arranging sounds in time through melody, harmony, and rhythm",
    "brain": "the organ inside the head that controls thought, memory, feeling, and movement",
    "neuron": "a cell that transmits information by electrical and chemical signals",
    "gravity": "the force that attracts objects with mass toward one another",
    "ocean": "a very large expanse of salt water covering most of the earth's surface",
    "star": "a luminous ball of gas held together by gravity that shines by nuclear fusion",
    "dog": "a domesticated carnivorous mammal kept as a pet or for work",
    "cat": "a small domesticated carnivorous mammal with soft fur",
    "bird": "a warm-blooded egg-laying animal with feathers, wings, and a beak",
    "language": "a system of words and rules that people use to communicate",
    "memory": "the faculty by which the mind stores and recalls information",
    "electron": "a subatomic particle with a negative charge found in every atom",
}


class WebAccessGate:
    """The controlled doorway: allowlist + SSRF firewall + size cap + provenance log around web_ingest."""

    def __init__(self):
        self.log = []  # [(host, bytes, source)] -- auditable provenance

    def _host(self, url):
        return (urlparse(url).hostname or "").lower()

    def allowed(self, url):
        return self._host(url) in ALLOWLIST

    def _require_allowed(self, url):
        """Raise unless url's host is allowlisted -- passed into web_ingest as the redirect check so a
        30x from an allowlisted host to an off-allowlist public host is refused, not silently followed."""
        if not self.allowed(url):
            raise PermissionError(
                f"web gate: redirect/target host {self._host(url)!r} is not on the allowlist"
            )

    def fetch(self, url, source="web", max_bytes=DEFAULT_MAXBYTES):
        """Fetch text through all three checks. Raises PermissionError if the host is not allowlisted
        (on the original URL AND every redirect); web_ingest raises on any SSRF-blocked target. Logs."""
        self._require_allowed(url)
        text = fetch_text(
            url, max_bytes=max_bytes, extra_check=self._require_allowed
        )  # allowlist on redirects too
        self.log.append((self._host(url), len(text), source))
        return text

    # ---- Wikipedia source adapter: fetch the REST summary API -> (title, plain-text extract) ----
    def fetch_wikipedia(self, topic, simple=False):
        """Fetch a scholarly Wikipedia summary for `topic` through the gate (host allowlist + SSRF
        firewall + size cap + provenance log all still apply). Returns (title, extract) where extract
        is the plain-text summary (the REST API 'extract' field). A missing / redirect / disambiguation
        page with no extract returns (topic, "") gracefully -- the caller stores nothing for it.

        HONEST: this reads a short encyclopedic SUMMARY (Wikipedia is CC BY-SA) for local personal use,
        not the full article and not for republishing. It is a read-only GET through the ONE doorway."""
        host = "simple" if simple else "en"
        url = f"https://{host}.wikipedia.org/api/rest_v1/page/summary/{quote(topic.replace(' ', '_'))}"
        text = self.fetch(url, source=f"wikipedia:{topic}")
        try:
            obj = json.loads(text)
        except ValueError:  # not JSON (unexpected body) -> no fact
            return (topic, "")
        title = obj.get("title") or topic  # canonical title (may differ from topic)
        extract = obj.get("extract") or ""  # "" for missing/redirect/disambiguation
        return (title, extract)

    def fetch_binary(
        self, url, out_path, source="web", max_bytes=700_000_000, chunk=1 << 20, timeout=180
    ):
        """STREAM a large BINARY through the gate to disk (e.g. ConceptNet's 498 MB .csv.gz) WITHOUT
        loading it into RAM -- `fetch`/`fetch_text` read the whole body into memory and decode it as
        text, which would both exhaust memory and corrupt a gzip. This keeps every gate guarantee:
        host ALLOWLIST + the substrate SSRF firewall (`validate_url` = http/https only + every resolved
        IP must be PUBLIC, plus a redirect validator so a 302 cannot walk to a private address) + a hard
        byte cap enforced while streaming + provenance logging. Returns (path, bytes_written)."""
        self._require_allowed(url)
        # the SSRF firewall lives in the substrate; reuse it rather than re-implementing it
        from snn_data_lake.substrate.web_ingest import _ValidatingRedirectHandler, validate_url

        validate_url(url)
        # allowlist on redirects too: a 302 from an allowlisted host cannot walk off the allowlist
        opener = urllib.request.build_opener(_ValidatingRedirectHandler(self._require_allowed))
        total = 0
        with opener.open(url, timeout=timeout) as resp, open(out_path, "wb") as f:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise ValueError(f"download exceeds the {max_bytes}-byte cap")
                f.write(block)
        self.log.append((self._host(url), total, source))
        return out_path, total

    # ---- WordNet source adapter: parse Princeton `data.*` synset lines -> {word: first-gloss} ----
    @staticmethod
    def parse_wordnet(text, limit=None):
        out = {}
        for line in text.splitlines():
            if not line or line[0] == " ":  # license header lines begin with spaces
                continue
            left, sep, gloss = line.partition("|")
            if not sep:
                continue
            parts = left.split()
            if len(parts) < 5:
                continue
            try:
                int(parts[3], 16)  # w_cnt is hex -> confirms a synset data line
            except ValueError:
                continue
            word = parts[4].replace("_", " ").lower()
            g = (
                gloss.strip().split(";")[0].strip().strip('"')
            )  # first gloss clause, drop usage examples
            if (
                word
                and g
                and word.isascii()
                and word not in out
                and all(c.isalpha() or c == " " for c in word)
            ):
                out[word] = g
                if limit and len(out) >= limit:
                    break
        return out

    def ingest_wordnet(self, learner, url=DEFAULT_WORDNET_URL, limit=None):
        """Enrich `learner` (anything with .learn(cue_tokens, answer)) with dictionary definitions:
        (1) a bundled CORE of common words (real defs, always present so common queries are answerable),
        then (2) WordNet fetched through the gate (thousands more; WordNet's `data.noun` is ordered by
        synset offset, so a limited slice is arbitrary, not the commonest words). Returns
        (n_web, n_core, provenance_str). Falls back to core-only if the web fetch is blocked/offline."""
        limit = limit if limit is not None else int(os.environ.get("NORD_WN_LIMIT", "4000"))
        for word, g in _BUNDLED.items():  # core common definitions (always present)
            learner.learn([word], f"{word}: {g}.")
        n_core = len(_BUNDLED)
        n_web = 0
        prov = "bundled core only (no web fetch)"
        try:
            text = self.fetch(url, source="wordnet:data.noun")
            for word, g in self.parse_wordnet(text, limit=limit).items():
                if word not in _BUNDLED:
                    learner.learn([word], f"{word}: {g}.")
                    n_web += 1
            prov = f"WordNet data.noun via {self._host(url)} ({self.log[-1][1]:,} bytes) -> {n_web} words"
        except Exception as e:  # blocked host / offline
            prov = f"web fetch unavailable ({type(e).__name__}) -- bundled core only"
        return n_web, n_core, prov


def main():
    from snn_data_lake.reasoning.spike_chat_lake import LakeChat

    print("=" * 92)
    print(
        "WEB ACCESS GATE -- the brain's controlled doorway to self-enrich from the web (WordNet first)"
    )
    print("=" * 92)
    gate = WebAccessGate()

    # ---- SAFETY: the gate refuses non-allowlisted hosts AND SSRF-blocked targets (localhost/private/file) ----
    blocked = [
        "http://localhost/x",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "file:///etc/passwd",
        "https://evil.example.com/steal",
    ]
    refused = 0
    for u in blocked:
        try:
            gate.fetch(u, max_bytes=1000)
        except Exception:
            refused += 1
    print(
        f"[safety] refused {refused}/{len(blocked)} unsafe/non-allowlisted fetches "
        f"(host allowlist + SSRF firewall + http/https-only)"
    )

    # ---- INGEST: fetch WordNet through the gate (or bundled fallback) into the brain's lake ----
    chat = LakeChat(seed=0)
    before = chat.n_known
    n_web, n_core, prov = gate.ingest_wordnet(
        chat, limit=int(os.environ.get("NORD_WN_LIMIT", "4000"))
    )
    print(f"[ingest] source: {prov}")
    print(
        f"[ingest] the brain's lake grew {before} -> {chat.n_known} words "
        f"(+{n_core} core defs, +{n_web} from the web)\n"
    )

    # ---- the brain now converses from the ingested dictionary ----
    for q in ["what is water", "what is a brain", "what is gravity"]:
        out = chat.converse(q)
        tag = "KNOWN" if out["known"] else "GAP"
        print(f"you : {q}\nbrain [{tag}]: {out['reply']}\n")
    got_water = chat.converse("what is water")
    dur = chat.lake.durability_check()

    print("=" * 92)
    print(
        f"provenance log (auditable): {gate.log if gate.log else '[offline -- bundled fallback]'}"
    )
    print("=" * 92)

    # ---- self-checks: the GATE is safe, ingests knowledge, enriches the lake, durable ----
    assert refused == len(blocked), (
        f"the gate must refuse ALL unsafe/non-allowlisted fetches ({refused}/{len(blocked)})"
    )
    assert not gate.allowed("http://evil.example.com/x"), (
        "a non-allowlisted host must not be allowed"
    )
    assert gate.allowed(DEFAULT_WORDNET_URL), "the WordNet source host must be allowlisted"
    # REDIRECT ALLOWLIST: a 30x from an allowlisted host to an off-allowlist public host must be
    # refused (not just SSRF-checked). The gate passes _require_allowed into the redirect handler.
    from snn_data_lake.substrate.web_ingest import _ValidatingRedirectHandler

    gate._require_allowed(DEFAULT_WORDNET_URL)  # an allowlisted target passes
    redirect_refused = False
    try:
        gate._require_allowed("https://evil.example.com/pwn")  # a redirect target off the allowlist
    except PermissionError:
        redirect_refused = True
    assert redirect_refused, (
        "a redirect to a non-allowlisted public host must be refused by the gate"
    )
    # The line above only proves the gate REFUSES an off-allowlist host when asked directly. The
    # criterion that matters is whether that refusal is actually REACHED from a 30x, so it is
    # EXERCISED here rather than introspected.
    #
    # This assert read, until 2026-08-05:
    #     assert _ValidatingRedirectHandler(gate._require_allowed)._extra_check is gate._require_allowed
    # and it was **UNCONDITIONALLY FALSE**, so `python -m snn_data_lake.reasoning.web_ingest_gate`
    # exited 1 on every run. `gate._require_allowed` is a BOUND METHOD: attribute access builds a
    # fresh `method` object each time, so `x is x` is False for two separate accesses (MEASURED:
    # `gate._require_allowed is gate._require_allowed` -> False, `==` -> True, distinct ids). It
    # was also near-tautological even had it worked -- `Handler(x)._extra_check is x` tests that
    # `__init__` performs an assignment, not that the GATE wires anything.
    #
    # What replaces it drives the real path. `_ValidatingRedirectHandler.redirect_request` runs
    # `validate_url(newurl)` FIRST, and an off-allowlist hostname that does not resolve dies there
    # (MEASURED: 'evil.example.com' raised ValueError "cannot resolve host" and `_extra_check` was
    # never called, which is why the original author reached for introspection). So the SSRF half is
    # stubbed out for the length of this check and the ALLOWLIST half is driven for real.
    import snn_data_lake.substrate.web_ingest as _wi

    # `redirect_request` only BUILDS the follow-up Request -- it performs no I/O -- so driving it
    # with a real Request off an allowlisted origin is offline and exercises the whole path.
    _real_validate = _wi.validate_url
    _wi.validate_url = lambda _u: None  # neuter the SSRF half; the allowlist half is under test

    def _follows_redirect(handler):
        """True iff `handler` FOLLOWS a 302 off the allowlist (i.e. does not refuse it)."""
        req = urllib.request.Request(DEFAULT_WORDNET_URL)
        try:
            handler.redirect_request(req, None, 302, "", {}, "https://evil.example.com/pwn")
            return True  # a follow-up Request was built: the redirect escaped
        except PermissionError:
            return False

    try:
        followed_wired = _follows_redirect(_ValidatingRedirectHandler(gate._require_allowed))
        # RED CONTROL: the SAME call through a handler with NO allowlist wired in must FOLLOW.
        # Without it, `followed_wired` could be False for a reason having nothing to do with the
        # gate, and this assert would be the decoration the one it replaced actually was.
        followed_unwired = _follows_redirect(_ValidatingRedirectHandler())
    finally:
        _wi.validate_url = _real_validate
    assert not followed_wired, (
        "the gate must wire its allowlist into the redirect handler: a 302 from an allowlisted "
        "host to 'evil.example.com' was FOLLOWED, so a redirect escapes the allowlist"
    )
    assert followed_unwired, (
        "control failed: a handler with NO allowlist wired in ALSO refused the redirect, so the "
        "assert above is not measuring the gate's wiring. Reachable red, constructed and run "
        "2026-08-05: dropping the `_extra_check(newurl)` call from "
        "`web_ingest._ValidatingRedirectHandler.redirect_request` -- which leaves the refusal "
        "itself intact, so the assert above THIS block still passes -- fires the first assert."
    )
    assert chat.n_known == before + n_core + n_web, (
        "the gate must ingest every fetched definition into the lake"
    )
    assert got_water["known"] and "liquid" in got_water["reply"].lower(), (
        "the brain must answer 'water' from the ingest"
    )
    assert dur["ok"], "the web-ingested knowledge must be durable in the lake"
    chat.lake.close()
    print(
        f"self-check OK: web access gate is safe (refused {refused}/{len(blocked)} unsafe fetches; allowlist + "
        f"SSRF firewall + size cap + provenance), ingested {n_core} core + {n_web} web definitions into the "
        f"brain's lake ({before}->{chat.n_known}), the brain answers from them, durable. The loop can now "
        f"self-enrich from the dictionary through ONE auditable doorway."
    )
    return 0


if __name__ == "__main__":
    main()
