"""Firewalled web -> text-corpus fetcher: the brain's (guarded) internet access for TRAINING DATA.

Give the spiking LM a real corpus to train on by downloading public text from the web:
  snn-fetch-corpus https://www.gutenberg.org/files/11/11-0.txt alice.txt
  snn-train --lm --corpus alice.txt --export-lm alice_lm.json     # (needs the [train] extra)

SAFETY -- read before changing. This is a DATA-PREP tool, deliberately OUTSIDE the robot-arm
control path: it is NOT imported by the arm loop, the autonomous brain, or any safety-envelope
module. An arm controller must never make blocking network calls or ingest untrusted bytes
mid-control; this only downloads text to a local file that YOU then feed to the trainer. The
download uses the standard library's urllib.

SSRF firewall (the security boundary):
  * http/https only -- no file://, ftp://, gopher://, data://, ...
  * every IP the host resolves to must be PUBLIC. Loopback / private (RFC1918) / link-local /
    reserved / multicast / unspecified are rejected -- which blocks localhost, RFC1918 hosts, and
    the 169.254.169.254 cloud-metadata endpoint (a classic SSRF target).
  * redirects are re-validated against the same firewall (no redirect into an internal host),
    AND the caller's own check (e.g. a host allowlist) is re-run on every redirect target --
    so an allowlisted host cannot 30x off the allowlist to some other public host, which the
    SSRF firewall alone would permit.
  * a hard byte cap (default 5 MB) bounds every download.
  Most of that is asserted with a REACHABLE RED in the offline self-check below -- eight
  mutations, all measured RED on 2026-09-01. It is not decoration: three of them were BLIND
  before that date, and each blind spot had the same shape (a case rejected by a SECOND
  mechanism, so the one being tested could be deleted with the check still green).
    * Every non-http entry in the blocked list is also caught by a missing or unresolvable
      host, so SAFE_SCHEMES could be widened -- to `file` above all, the scheme this docstring
      names first -- and the self-check stayed GREEN. Fixed by scheme-only cases: a non-http
      scheme over a PUBLIC IP LITERAL, which the scheme check rejects and nothing else does.
    * The redirect handler had no test whatsoever.
    * The first redirect test could not tell the TARGET from the ORIGIN: same-host fixture,
      deny-everything stub. Handing extra_check `req.full_url` instead of `newurl` -- the
      exact allowlist escape described above -- stayed GREEN. Fixed with distinct hosts and a
      check that allows the origin only, plus a control requiring an on-allowlist redirect to
      still be followed.
  Keep those fixtures. Simplifying any of them re-opens the hole it closes.

  NOT YET POLICED, and stated rather than implied: the byte cap and the extra_check calls on
  the ORIGINAL and FINAL urls inside fetch_bytes() have no reachable red -- they sit on the
  network path and need a fake-opener harness this offline check does not have. (They moved
  from fetch_text() to fetch_bytes() on 2026-09-09 when the download became one function; the
  blind spot moved with them and did not shrink. What the offline check DID gain that day is a
  structural pin that fetch_text still routes through fetch_bytes -- reverting the refactor to
  a second opener reds it.)
  ponytail: the host is resolved-and-checked but the resolved IP is not pinned for the actual
  socket connect, so a determined DNS-rebinding TOCTOU attacker is out of scope -- acceptable for
  fetching public corpora, not for a hostile-controlled URL set. Upgrade path: pin the checked IP
  and connect to it directly.

  python -m snn_data_lake.substrate.web_ingest        # offline self-check (no network)
"""

import argparse
import ipaddress
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

SAFE_SCHEMES = ("http", "https")
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT = 30
_UA = "snn-data-lake web_ingest (public training-corpus fetch)"


def _addr_blocked(ip_str):
    """True unless ip_str is a normal PUBLIC address (loopback/private/link-local/... are blocked)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_ips(host):
    """Every IP a host resolves to (hits DNS); [] on failure."""
    try:
        return [info[4][0] for info in socket.getaddrinfo(host, None)]
    except socket.gaierror:
        return []


def validate_url(url):
    """Raise ValueError unless url is http/https AND its host resolves ONLY to public IPs (SSRF guard)."""
    p = urlparse(url)
    if p.scheme not in SAFE_SCHEMES:
        raise ValueError(f"blocked scheme {p.scheme!r}: only http/https are allowed")
    host = p.hostname
    if not host:
        raise ValueError(f"no host in URL {url!r}")
    try:
        ipaddress.ip_address(host)  # IP literal -> check directly, no DNS
        ips = [host]
    except ValueError:
        ips = _host_ips(host)
        if not ips:
            raise ValueError(f"cannot resolve host {host!r}") from None
    bad = [ip for ip in ips if _addr_blocked(ip)]
    if bad:
        raise ValueError(f"blocked host {host!r} -> non-public IP {bad[0]} (SSRF guard)")
    return url


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-run the SSRF firewall on every redirect target before following it, PLUS an optional
    extra_check (e.g. a caller's host allowlist) -- so an allowlisted host cannot 30x-redirect off the
    allowlist to any public host (the SSRF firewall alone would let that through)."""

    def __init__(self, extra_check=None):
        super().__init__()
        self._extra_check = extra_check

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        if self._extra_check is not None:
            self._extra_check(newurl)  # e.g. the gate's allowlist -- refuse a redirect off it
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# P2 DEBT, NAMED AND DATED 2026-09-09 (specs/022 T062) -- the one real duplicate of the loop below
#   is `reasoning/web_ingest_gate.py::WebIngestGate.fetch_binary`: it already streams a large binary
#   through THIS module's firewall (it imports `validate_url` and `_ValidatingRedirectHandler`
#   rather than re-implementing them) and enforces its own byte cap while streaming, differing only
#   in DESTINATION (a file on disk) and TIER (the ruff-excluded, gate-less `reasoning/`). `sink`
#   below exists so that duplicate can be retired ONTO this function -- the capability is ported
#   FIRST, which is P2's order. It was NOT retired in this change and that is a disclosure, not an
#   oversight: `cortex/h01_ingest.py` is a CORE self-check module and this module's own docstring
#   already refuses to let a CORE check depend on the unregistered `reasoning/` tier, and
#   `web_ingest_gate`'s own check performs a ~15 MB network fetch, so the swap could be written but
#   not RUN in an offline arc -- and an unverified refactor of a security boundary is worse than a
#   named debt. The retirement is one call site plus its provenance log line.
# distinct-from: substrate/self_mod_harness.py::_run_selfchecks -- runs run_selfchecks.py in a
#   subprocess inside a sandbox; no URL, no socket. distinct-from: cortex/live_brain.py::tick --
#   one brain tick over a context dict. distinct-from: cortex/metacognitive_monitor.py::_experiment
#   -- a gated/lesioned/plain comparison harness. distinct-from: reasoning/arc_solver.py::solve --
#   iterative-deepening program synthesis over ARC grids. distinct-from:
#   reasoning/autonomous_enrich.py::learn_topic -- a CALLER of this module (through the gate's
#   fetch_wikipedia), not a second downloader. All five sites read 2026-09-09; the guard matched on
#   the words "Fetch"/"Returns", not on a capability.
def fetch_bytes(
    url,
    max_bytes=DEFAULT_MAX_BYTES,
    timeout=DEFAULT_TIMEOUT,
    extra_check=None,
    chunk=1 << 20,
    sink=None,
):
    """THE ONE firewalled download (specs/022 T062, P2). Returns raw bytes, or a byte count.

    Firewalled + redirect-checked + size-capped exactly as `fetch_text` was: `extra_check` (if
    given) runs on the original URL, every redirect target, AND the final URL, so a redirect
    cannot escape a caller's host allowlist. `fetch_text` is now a decode of this -- one download
    path, not a text one and a binary one that drift apart.

    WHY BYTES AT ALL, stated rather than left to be inferred: this module could fetch only text, so
    an ingest of a BINARY release (T062 streams 197 MB Avro shards from the H01 bucket) had a
    choice between standing up a second opener with its own firewall wiring -- the additive failure
    P2 names -- and widening this one.

    `sink`, if given, is called with each chunk and NOTHING is buffered: the return value is the
    byte count. That is how a 197 MB shard is hashed and decoded without a second copy, and it is
    also the shape that lets the `fetch_binary` duplicate named above retire onto this function.

    `max_bytes` stays required-by-default at 5 MB and is enforced WHILE streaming, so an oversized
    response is refused without being buffered whole; a caller pulling a large object passes the
    size it verified from a manifest, so the cap is still a number somebody committed to rather
    than an absent bound. The refusal message is unchanged.
    """
    validate_url(url)
    if extra_check is not None:
        extra_check(url)
    opener = urllib.request.build_opener(_ValidatingRedirectHandler(extra_check))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    buf = bytearray()
    total = 0
    with opener.open(req, timeout=timeout) as resp:
        final = resp.geturl()
        validate_url(final)  # re-check the final (post-redirect) URL
        if extra_check is not None:
            extra_check(final)  # ...and re-check it against the caller's allowlist too
        while total <= max_bytes:
            piece = resp.read(min(chunk, max_bytes + 1 - total))
            if not piece:
                break
            total += len(piece)
            if sink is None:
                buf += piece
            else:
                sink(piece)
    if total > max_bytes:
        raise ValueError(f"download exceeds the {max_bytes}-byte cap")
    return total if sink is not None else bytes(buf)


def fetch_text(url, max_bytes=DEFAULT_MAX_BYTES, timeout=DEFAULT_TIMEOUT, extra_check=None):
    """Download url and return decoded text. Firewalled + redirect-checked + size-capped. `extra_check`
    (if given) is run on the original URL, every redirect target, AND the final URL -- callers pass
    their host allowlist so a redirect cannot escape it.

    One line of body on purpose: `fetch_bytes` above IS the download; this is the decode."""
    return fetch_bytes(url, max_bytes=max_bytes, timeout=timeout, extra_check=extra_check).decode(
        "utf-8", errors="replace"
    )


def html_to_text(html):
    """Strip scripts/styles/tags from HTML into rough plain text (best-effort, for web pages)."""
    html = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    for ent, ch in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&#39;", "'"),
        ("&quot;", '"'),
    ):
        html = html.replace(ent, ch)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n[ \t]*\n[ \t]*", "\n\n", html)
    return html.strip()


def fetch_corpus(
    url,
    out_path,
    max_bytes=DEFAULT_MAX_BYTES,
    timeout=DEFAULT_TIMEOUT,
    strip_html=None,
    license_id=None,
):
    """Fetch url, optionally strip HTML, write text to out_path. Returns (out_path, n_chars, flags).

    `license_id` is REQUIRED and deny-by-default (substrate/ingest_guard). This is a deliberate
    behaviour change: the tool downloads other people's text to train on, and it used to do that
    without ever asking under what terms. Stating the licence is now the caller's job, because it is
    the caller who knows -- and "no licence stated" must not be the silent default when the answer
    determines whether the download was allowed at all.

    The returned `flags` are instruction-shaped passages found in the corpus. They do not block the
    write: the corpus is a FILE a trainer consumes, never a fact entering the durable lake and never
    a string handed to anything that acts, so the boundary is structural. The flags exist so a corpus
    that is full of prompt-injection reads as suspicious instead of arriving silently.
    """
    from snn_data_lake.substrate.ingest_guard import guard

    text = fetch_text(url, max_bytes=max_bytes, timeout=timeout)
    if strip_html is None:
        strip_html = bool(re.search(r"(?i)<!doctype html|<html|<body", text[:2048]))
    if strip_html:
        text = html_to_text(text)
    rec = guard(text, license_id, url)  # raises LicenseRefused before anything is written
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rec["text"])
    return out_path, len(rec["text"]), rec["injection_flags"]


def cli(argv=None):
    ap = argparse.ArgumentParser(
        prog="snn-fetch-corpus",
        description="Fetch a public web text corpus for training (SSRF-firewalled, http/https only).",
    )
    ap.add_argument("url")
    ap.add_argument("out", help="output .txt path")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--raw", action="store_true", help="do not strip HTML tags")
    ap.add_argument(
        "--license",
        required=True,
        help="licence the source is published under (deny-by-default; see ingest_guard.LICENSE_ALLOW). "
        "Required: this downloads other people's text to train on, and silence is not permission.",
    )
    a = ap.parse_args(argv)
    try:
        path, n, flags = fetch_corpus(
            a.url,
            a.out,
            max_bytes=a.max_bytes,
            timeout=a.timeout,
            strip_html=(False if a.raw else None),
            license_id=a.license,
        )
    except (ValueError, urllib.error.URLError, OSError) as e:
        print(f"fetch failed: {e}")
        return 1
    print(f"wrote {n} chars to {path}")
    if flags:
        print(f"WARNING: instruction-shaped passages in this corpus: {flags}")
        print("  it is a training file, not a fact and not a command -- but read it before use.")
    print(f"now train on it:  snn-train --lm --corpus {path}")
    return 0


def main():
    # ---- offline self-check: the SSRF firewall + parsing, NO outbound network ----
    blocked = [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://x/",
        "data:text/plain,hi",
        "http://",
        "http://127.0.0.1/",
        "https://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.5/",
        "http://[::1]/",
        "http://0.0.0.0/",
        "http://[fe80::1]/",
    ]
    for u in blocked:
        raised = False
        try:
            validate_url(u)
        except ValueError:
            raised = True
        assert raised, f"SSRF/scheme guard must reject {u!r}"

    # THE SCHEME ALLOWLIST NEEDED A CASE ONLY IT CAN REJECT. Every non-http entry above is
    # ALSO caught by a second mechanism -- file:/data: have no host, gopher://x/ does not
    # resolve -- so none of them depended on SAFE_SCHEMES alone, and the allowlist therefore
    # had NO REACHABLE RED: widening it to {file, gopher} left this whole self-check GREEN
    # (measured by mutation, 2026-09-01). A non-http scheme over a PUBLIC IP LITERAL is
    # rejected by the scheme check and by nothing else -- an IP literal skips DNS and the
    # address passes _addr_blocked -- so these RED the moment SAFE_SCHEMES grows.
    # EVERY dangerous scheme needs its own case, `file:` and `data:` MOST of all -- the
    # docstring names file:// first, and the first version of this loop covered ftp/gopher/ws
    # and left `file` unpoliced, so adding it to SAFE_SCHEMES still passed. A review panel
    # measured that: the one scheme the module calls out by name was the one still blind.
    for u in (
        "file://8.8.8.8/etc/passwd",
        "data://8.8.8.8/x",
        "ftp://8.8.8.8/x",
        "gopher://1.1.1.1/",
        "ws://8.8.8.8/sock",
    ):
        why = ""
        try:
            validate_url(u)
        except ValueError as e:
            why = str(e)
        assert "blocked scheme" in why, (
            f"the scheme allowlist must reject {u!r} ON ITS OWN (got {why!r}) -- this case "
            f"exists because every other non-http case is also caught by a missing or "
            f"unresolvable host, which left SAFE_SCHEMES with no reachable red"
        )

    # REDIRECT RE-VALIDATION HAD NO TEST AT ALL. The module docstring advertises "redirects
    # are re-validated against the same firewall" and that an allowlisted host cannot
    # 30x-redirect off the allowlist -- both security properties, and main() never referenced
    # _ValidatingRedirectHandler, so both were unverified. Driven offline below: redirect_request refuses
    # BEFORE delegating to the base handler, so no socket is opened.
    # distinct-from: reasoning/web_ingest_gate.py::WebIngestGate._require_allowed -- THE
    #   production host-allowlist check and the real extra_check handed to fetch_text.
    #   Deliberately NOT imported: it lives in the ungated reasoning/ tier (no
    #   run_selfchecks / REGIONS / dormancy-gate registration, measured 2026-09-01), and a
    #   CORE self-check must not depend on an unregistered module to prove a security
    #   property. The stub below stands for ANY caller check; what this module owns is that
    #   the plumbing CALLS it, which is what is asserted. Site read.
    # distinct-from: basal_ganglia/brain_native_selfbuild.py::_forbidden (legality of a
    #   self-build FILE target) and basal_ganglia/latent_q.py::update (a Q-learning scalar
    #   `target`). Neither concerns redirect targets. Sites read.
    # THE FIXTURE HAS TO DISCRIMINATE, and the first version of this arm did not. It sent
    # origin http://8.8.8.8/start -> target http://8.8.8.8/ok (the SAME host) through a stub
    # that denied EVERYTHING, so it could only prove "extra_check was called with something",
    # never "called with the redirect TARGET". A panel measured the consequence: mutating
    # `self._extra_check(newurl)` to `self._extra_check(req.full_url)` -- the classic
    # wrong-variable bug, and precisely the allowlist escape this handler exists to stop --
    # left the self-check GREEN. Two fixes, both load-bearing: DISTINCT hosts, and a check
    # that ALLOWS the origin and denies the rest, so handing it the origin cannot pass.
    # distinct-from: reasoning/web_ingest_gate.py::_require_allowed -- the PRODUCTION
    #   allowlist, in the ungated reasoning/ tier; it has carried an equivalent arm since
    #   2026-08-05, which is why this one matches its discipline. Not imported: a CORE
    #   self-check must not depend on an unregistered module, and that module's own check
    #   performs a ~15MB network fetch this offline suite forbids.
    # distinct-from: gates/selfcheck_cache.py::consult -- a cache lookup keyed on a
    #   closure hash; nothing to do with URL hosts. Sites read.
    _ORIGIN, _TARGET = "http://1.1.1.1/book", "http://8.8.8.8/exfil"
    _origin_host = urlparse(_ORIGIN).hostname

    def _probe_redirect(newurl, caller_check=None):
        # a real Request, because the ALLOWED path delegates to the base handler and that
        # needs a genuine req; constructing one opens no socket.
        req = urllib.request.Request(_ORIGIN, headers={"User-Agent": _UA})
        handler = _ValidatingRedirectHandler(extra_check=caller_check)
        try:
            handler.redirect_request(req, None, 302, "Found", {}, newurl)
            return ""
        except (ValueError, PermissionError) as e:
            return str(e) or type(e).__name__

    def _one_host_allowed(url):
        # allows the origin host only -- so it REPORTS which url it got, which a
        # deny-everything stub cannot.
        if urlparse(url).hostname != _origin_host:
            raise PermissionError(f"host {urlparse(url).hostname!r} is not on the allowlist")

    for evil in ("http://127.0.0.1/", "http://169.254.169.254/", "file:///etc/passwd"):
        assert _probe_redirect(evil), f"a redirect to {evil!r} must be refused by the firewall"
    # ...and the caller-check half. A PUBLIC target passes the firewall on its own, so only
    # extra_check can stop it -- assert that first, or the next assert proves nothing.
    assert _probe_redirect(_TARGET) == "", (
        "a public redirect target must pass the firewall unaided, or the extra_check arm "
        "below is vacuous (it would be measuring the firewall, not the caller's allowlist)"
    )
    assert _probe_redirect(_TARGET, caller_check=_one_host_allowed), (
        "extra_check must be enforced on the redirect TARGET -- without it an allowlisted "
        "host can 30x off the allowlist to any PUBLIC host, which the SSRF firewall itself "
        "permits. Distinct hosts and an origin-allowing check are deliberate: a same-host "
        "fixture, or a stub that denies everything, stays GREEN even when the handler checks "
        "the ORIGIN instead of the target"
    )
    # ...and the control that keeps that assert honest: the same check must ACCEPT a redirect
    # that stays on the allowlist, or the line above could be satisfied by a check that
    # simply always raises.
    assert _probe_redirect("http://1.1.1.1/next", caller_check=_one_host_allowed) == "", (
        "an on-allowlist redirect must still be FOLLOWED -- otherwise the arm above is "
        "measuring a stub that refuses everything, not an allowlist"
    )

    assert not _addr_blocked("8.8.8.8") and not _addr_blocked("1.1.1.1"), "public IPs must pass"
    for ip in (
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.9",
        "169.254.169.254",
        "::1",
        "fe80::1",
        "0.0.0.0",
    ):
        assert _addr_blocked(ip), f"{ip} must be blocked"
    assert _addr_blocked("not-an-ip"), "a non-IP string must be treated as blocked"

    t = html_to_text("<html><body><script>steal()</script><p>Hello &amp; world</p></body></html>")
    assert "steal()" not in t and "Hello & world" in t, t

    # ONE DOWNLOAD PATH (P2, specs/022 T062). fetch_text must ROUTE THROUGH fetch_bytes rather
    # than open its own connection: a second opener is a second place for the firewall, the
    # redirect re-validation and the cap to be got wrong, and the whole point of this module is
    # that there is exactly one of each. Structural rather than behavioural because the offline
    # check opens no socket -- but it is a REACHABLE red: restore fetch_text's own
    # build_opener/read pair and this line goes red in the same command.
    assert "fetch_bytes" in fetch_text.__code__.co_names, (
        "fetch_text no longer routes through fetch_bytes -- there are two download paths again, "
        "and the firewall/redirect/cap guarantees above now have to hold in both"
    )
    assert "build_opener" not in fetch_text.__code__.co_names, (
        "fetch_text builds its own opener again (P2: exactly one mechanism survives)"
    )

    print("=" * 62)
    print("WEB INGEST -- firewalled web -> training-corpus fetcher")
    print("=" * 62)
    print(
        f"http/https + public-IP only; {len(blocked)} SSRF/scheme cases rejected; "
        f"cap {DEFAULT_MAX_BYTES} bytes; redirects re-checked"
    )
    print(
        "self-check OK: SSRF firewall rejects loopback/private/link-local/metadata + non-http "
        "schemes; HTML->text strips scripts; NOT wired into the arm control loop.\n"
        "  MUTATION-VERIFIED 2026-09-01, eight mutations, all RED: IP firewall neutered; "
        "SAFE_SCHEMES widened by file / ftp / data (three separate arms); redirect "
        "re-validation dropped; the extra_check call removed; extra_check handed the ORIGIN "
        "instead of the target; extra_check handed a CONSTANT. The last two escaped an "
        "earlier version of these arms -- a same-host fixture with a deny-everything stub "
        "proves only that extra_check was CALLED, never that it was called with the redirect "
        "TARGET -- which is why the fixture uses distinct hosts and an origin-allowing check.\n"
        "  ONE DOWNLOAD PATH pinned 2026-09-09 (P2, specs/022 T062): fetch_text routes through "
        "fetch_bytes and opens no connection of its own; restoring a second opener reds this "
        "check. fetch_bytes' `sink` carries the streaming capability that "
        "reasoning/web_ingest_gate.fetch_binary duplicates -- ported here first, that duplicate "
        "NOT yet retired, named as a dated P2 debt at the function.\n"
        "  STILL UNPOLICED, said plainly rather than left for the next reviewer: the byte "
        "cap, and extra_check on the ORIGINAL and FINAL urls inside fetch_bytes(). Those three "
        "live on the network path, so covering them needs a fake-opener harness this offline "
        "check does not have yet. reasoning/web_ingest_gate.py has carried a correct "
        "target-vs-origin arm since 2026-08-05, but it is registered in NO gate (not "
        "run_selfchecks, REGIONS, or dormancy_gate) and its own check fetches ~15MB, so it "
        "cannot simply be registered into an offline suite."
    )


if __name__ == "__main__":
    main()
