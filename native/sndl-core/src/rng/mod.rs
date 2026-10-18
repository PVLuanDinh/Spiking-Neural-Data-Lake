//! RNG parity kit (plan section 10, stage R0.4): bit-exact ports of every seeded stream SNDL's Python
//! consumes, judged against `tests/fixtures_rng.json` (real CPython 3.14.6t `random` + numpy 2.5.1,
//! RNG-KIT.md criteria C1-C8) by the one comparator in [`parity`].
//!
//! Ported: MT19937 as CPython seeds it ([`mt19937`]); `random.Random` int seeds of any size, str seeds
//! (sha512 path), `random`, `getrandbits(k <= 128)`, `_randbelow` rejection, `randrange`/`randint`/
//! `uniform`/`choice`/`choices`/`shuffle`/`sample`, `getstate`/`setstate` ([`pyrandom`]); numpy
//! `SeedSequence` with hash mixing, list entropy and `spawn` ([`seedseq`]); `PCG64` XSL-RR with the
//! `next_uint32` half buffer and `Generator.random`/`integers` (Lemire with rejection; a one-value range
//! consumes no draw)/`uniform`/`permutation`/`shuffle`/`choice(n)` ([`pcg64`]); the legacy `RandomState`
//! core `rand`/`permutation`/`choice` with keys+pos ([`legacy`]); `infra/_mathutils.py::Rng` ([`lcg`]);
//! `zlib.crc32` for `reasoning_cortex._rng` ([`crc32`]).
//!
//! NOT ported, by RNG-KIT.md section 3 (OPEN items; the fixture sections stay UNCLAIMED and are never
//! reported as passing): `random.gauss`, `random.expovariate`, `RandomState.randn` (OPEN-libm: bit-exact
//! `log`/`sqrt`/`cos`/`sin` against the MSVC CRT is unmeasured) and `Generator.normal`/`standard_normal`
//! (OPEN-ziggurat: numpy's 256-layer tables). `getstate` therefore always carries `gauss_next = None`.

pub mod lcg;
pub mod legacy;
pub mod mt19937;
pub mod parity;
pub mod pcg64;
pub mod pyrandom;
pub mod seedseq;

/// |n| as little-endian u32 words, `0 -> [0]`: CPython's `random.seed(int)` key and numpy's
/// `_coerce_to_uint32_array` agree on this shape.
pub fn words_u128(mut n: u128) -> Vec<u32> {
    let mut out = Vec::new();
    while n != 0 {
        out.push(n as u32);
        n >>= 32;
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

/// Big-endian bytes of a non-negative integer (`int.from_bytes(b, "big")`) as little-endian u32 words.
pub fn words_from_be_bytes(b: &[u8]) -> Vec<u32> {
    let mut out: Vec<u32> = b
        .rchunks(4)
        .map(|c| c.iter().fold(0u32, |acc, &x| (acc << 8) | u32::from(x)))
        .collect();
    while out.len() > 1 && out.last() == Some(&0) {
        out.pop();
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

/// A decimal integer of any length (optional leading `-`, ignored: the key is |n|) as little-endian u32
/// words. `None` when `s` is not a plain decimal.
pub fn words_from_decimal(s: &str) -> Option<Vec<u32>> {
    let s = s.strip_prefix('-').unwrap_or(s);
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut out = vec![0u32];
    for d in s.bytes() {
        let mut carry = u64::from(d - b'0');
        for w in out.iter_mut() {
            let v = u64::from(*w) * 10 + carry;
            *w = v as u32;
            carry = v >> 32;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
    }
    Some(out)
}

/// `zlib.crc32(bytes)` (IEEE, reflected 0xEDB88320): `reasoning_cortex._rng` seeds `Random(crc32(tag))`.
pub fn crc32(b: &[u8]) -> u32 {
    let mut c = !0u32;
    for &x in b {
        c ^= u32::from(x);
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

// ---- P8 reds: test-only mutants on the REAL code paths. `mutant(bit)` is a constant `false` outside
// `cfg(test)`, so the branches compile out of every non-test build; the unit tests in `parity` switch
// one bit at a time and count the fixture cells that go red (pre-registered in PREREG.md "## r04-rng").
pub(crate) const SWAP53: u8 = 1;
pub(crate) const SINGLE_WORD_SEED: u8 = 2;
pub(crate) const BAD_INC: u8 = 4;
pub(crate) const RNG0_DRAWS: u8 = 8;
pub(crate) const NO_U32_BUFFER: u8 = 16;

#[cfg(test)]
#[inline]
pub(crate) fn mutant(bit: u8) -> bool {
    mutants::ACTIVE.with(|c| c.get() & bit != 0)
}

#[cfg(not(test))]
#[inline(always)]
pub(crate) const fn mutant(_bit: u8) -> bool {
    false
}

#[cfg(test)]
pub(crate) mod mutants {
    use std::cell::Cell;
    thread_local! { pub static ACTIVE: Cell<u8> = const { Cell::new(0) }; }
    pub fn set(bits: u8) {
        ACTIVE.with(|c| c.set(bits));
    }
}
