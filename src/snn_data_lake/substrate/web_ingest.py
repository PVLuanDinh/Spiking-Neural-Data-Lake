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
  * redirects are re-validated against the same firewall (no redirect into an internal host).
  * a hard byte cap (default 5 MB) bounds every download.
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


def fetch_text(url, max_bytes=DEFAULT_MAX_BYTES, timeout=DEFAULT_TIMEOUT, extra_check=None):
    """Download url and return decoded text. Firewalled + redirect-checked + size-capped. `extra_check`
    (if given) is run on the original URL, every redirect target, AND the final URL -- callers pass
    their host allowlist so a redirect cannot escape it."""
    validate_url(url)
    if extra_check is not None:
        extra_check(url)
    opener = urllib.request.build_opener(_ValidatingRedirectHandler(extra_check))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with opener.open(req, timeout=timeout) as resp:
        final = resp.geturl()
        validate_url(final)  # re-check the final (post-redirect) URL
        if extra_check is not None:
            extra_check(final)  # ...and re-check it against the caller's allowlist too
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"download exceeds the {max_bytes}-byte cap")
    return data.decode("utf-8", errors="replace")


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
    url, out_path, max_bytes=DEFAULT_MAX_BYTES, timeout=DEFAULT_TIMEOUT, strip_html=None
):
    """Fetch url, optionally strip HTML, write text to out_path. Returns (out_path, n_chars)."""
    text = fetch_text(url, max_bytes=max_bytes, timeout=timeout)
    if strip_html is None:
        strip_html = bool(re.search(r"(?i)<!doctype html|<html|<body", text[:2048]))
    if strip_html:
        text = html_to_text(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path, len(text)


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
    a = ap.parse_args(argv)
    try:
        path, n = fetch_corpus(
            a.url,
            a.out,
            max_bytes=a.max_bytes,
            timeout=a.timeout,
            strip_html=(False if a.raw else None),
        )
    except (ValueError, urllib.error.URLError, OSError) as e:
        print(f"fetch failed: {e}")
        return 1
    print(f"wrote {n} chars to {path}")
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

    print("=" * 62)
    print("WEB INGEST -- firewalled web -> training-corpus fetcher")
    print("=" * 62)
    print(
        f"http/https + public-IP only; {len(blocked)} SSRF/scheme cases rejected; "
        f"cap {DEFAULT_MAX_BYTES} bytes; redirects re-checked"
    )
    print(
        "self-check OK: SSRF firewall rejects loopback/private/link-local/metadata + non-http "
        "schemes; HTML->text strips scripts; NOT wired into the arm control loop"
    )


if __name__ == "__main__":
    main()
