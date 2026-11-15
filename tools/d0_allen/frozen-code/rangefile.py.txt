"""A read-only, seekable file object over HTTP Range requests (stdlib only), for h5py's fileobj driver.
Fetches 1 MiB blocks on demand and caches them, so opening a 40 GB NWB reads only the bytes it touches."""
import io
import urllib.request
from collections import OrderedDict

BLOCK = 1 << 20


def resolve(url):
    """Follow redirects once (DANDI download -> S3) and return the final URL and total size."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.geturl(), int(r.headers["Content-Range"].rsplit("/", 1)[1])


class RangeFile(io.RawIOBase):
    def __init__(self, url, cache_blocks=512):
        self.url, self.size = resolve(url)
        self.pos = 0
        self.cache = OrderedDict()
        self.cache_blocks = cache_blocks
        self.fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        lo, hi = i * BLOCK, min(self.size, (i + 1) * BLOCK) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 3:
                    raise
        self.fetched += len(data)
        self.cache[i] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def readinto(self, b):
        n = min(len(b), max(0, self.size - self.pos))
        out, got = memoryview(b), 0
        while got < n:
            i, off = divmod(self.pos + got, BLOCK)
            blk = self._block(i)
            take = min(n - got, len(blk) - off)
            out[got : got + take] = blk[off : off + take]
            got += take
        self.pos += n
        return n
