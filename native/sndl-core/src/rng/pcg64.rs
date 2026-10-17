//! numpy `PCG64` (XSL-RR 128/64, `pcg64.h`, `pcg_setseq_128_srandom_r` seeding from four SeedSequence
//! words) with the `next_uint32` half buffer, and the `Generator` methods SNDL calls on `default_rng`:
//! `random`, `integers` (`random_bounded_uint64`: Lemire multiply-shift with rejection; a one-value range
//! consumes NO draw; a full 32/64-bit range is one raw draw), `uniform`, `permutation`, `shuffle`
//! (`random_interval`), scalar `choice(n)`. `normal`/`standard_normal` are OPEN-ziggurat (not here).
use super::seedseq::SeedSequence;
use super::{BAD_INC, NO_U32_BUFFER, RNG0_DRAWS, mutant, words_u128};

const MULT: u128 = 47_026_247_687_942_121_848_144_207_491_837_523_525;
const M32: u64 = 0xffff_ffff;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_u32: bool,
    u32_cache: u32,
}

/// `bit_generator.state` of a PCG64: `{"state": {"state", "inc"}, "has_uint32", "uinteger"}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pcg64State {
    pub state: u128,
    pub inc: u128,
    pub has_uint32: bool,
    pub uinteger: u32,
}

impl Pcg64 {
    /// `PCG64(seed_seq)`: `generate_state(4, uint64)` -> `initstate`, `initseq` -> `inc = initseq << 1 | 1`.
    pub fn from_seedseq(ss: &SeedSequence) -> Self {
        let s = ss.generate_u64(4);
        let initstate = (u128::from(s[0]) << 64) | u128::from(s[1]);
        let initseq = (u128::from(s[2]) << 64) | u128::from(s[3]);
        let inc = if mutant(BAD_INC) {
            initseq
        } else {
            (initseq << 1) | 1
        };
        let mut g = Self {
            state: 0,
            inc,
            has_u32: false,
            u32_cache: 0,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    /// `bit_generator.state = {...}`.
    pub fn from_state(st: Pcg64State) -> Self {
        Self {
            state: st.state,
            inc: st.inc,
            has_u32: st.has_uint32,
            u32_cache: st.uinteger,
        }
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
    }

    /// `next_uint64`: step, then XSL-RR output.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let st = self.state;
        let x = ((st >> 64) as u64) ^ (st as u64);
        x.rotate_right((st >> 122) as u32)
    }

    /// `next_uint32`: one u64 draw serves two calls (low word now, high word cached).
    pub fn next_u32(&mut self) -> u32 {
        if mutant(NO_U32_BUFFER) {
            return self.next_u64() as u32;
        }
        if self.has_u32 {
            self.has_u32 = false;
            return self.u32_cache;
        }
        let n = self.next_u64();
        self.has_u32 = true;
        self.u32_cache = (n >> 32) as u32;
        n as u32
    }

    pub fn state(&self) -> Pcg64State {
        Pcg64State {
            state: self.state,
            inc: self.inc,
            has_uint32: self.has_u32,
            uinteger: self.u32_cache,
        }
    }
}

/// `np.random.Generator(PCG64(...))`, i.e. `default_rng(seed)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generator {
    bg: Pcg64,
}

impl Generator {
    pub fn from_seedseq(ss: &SeedSequence) -> Self {
        Self {
            bg: Pcg64::from_seedseq(ss),
        }
    }

    /// `default_rng(int)` with the int already split into u32 words.
    pub fn from_words(words: Vec<u32>) -> Self {
        Self::from_seedseq(&SeedSequence::new(words))
    }

    pub fn from_u128(seed: u128) -> Self {
        Self::from_words(words_u128(seed))
    }

    pub fn from_state(st: Pcg64State) -> Self {
        Self {
            bg: Pcg64::from_state(st),
        }
    }

    /// `random()`: 53 bits of one u64.
    pub fn random(&mut self) -> f64 {
        (self.bg.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `random_bounded_uint64(0, rng, mask=0, use_masked=False)`: a value in `0..=rng`.
    pub fn bounded(&mut self, rng: u64) -> u64 {
        match rng {
            0 => {
                if mutant(RNG0_DRAWS) {
                    self.bg.next_u64();
                }
                0
            }
            M32 => u64::from(self.bg.next_u32()),
            r if r < M32 => u64::from(self.lemire32(r as u32)),
            u64::MAX => self.bg.next_u64(),
            r => self.lemire64(r),
        }
    }

    fn lemire32(&mut self, rng: u32) -> u32 {
        let excl = u64::from(rng) + 1;
        let mut m = u64::from(self.bg.next_u32()) * excl;
        let mut left = m & M32;
        if left < excl {
            let thr = ((1u64 << 32) - excl) % excl;
            while left < thr {
                m = u64::from(self.bg.next_u32()) * excl;
                left = m & M32;
            }
        }
        (m >> 32) as u32
    }

    fn lemire64(&mut self, rng: u64) -> u64 {
        let excl = u128::from(rng) + 1;
        let mut m = u128::from(self.bg.next_u64()) * excl;
        let mut left = u128::from(m as u64);
        if left < excl {
            let thr = ((1u128 << 64) - excl) % excl;
            while left < thr {
                m = u128::from(self.bg.next_u64()) * excl;
                left = u128::from(m as u64);
            }
        }
        (m >> 64) as u64
    }

    /// `integers(lo, hi)` (`endpoint=False`, int64); `hi > lo`.
    pub fn integers(&mut self, lo: i64, hi: i64) -> i64 {
        assert!(hi > lo, "integers: empty range");
        let rng = (i128::from(hi) - 1 - i128::from(lo)) as u64;
        lo.wrapping_add(self.bounded(rng) as i64)
    }

    /// `integers(lo, hi, dtype=np.uint64)`: the same path for ranges past `i64::MAX`.
    pub fn integers_u64(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(hi > lo, "integers: empty range");
        lo + self.bounded(hi - 1 - lo)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.random()
    }

    /// `random_interval(max)`: masked rejection, 32-bit draws while `max <= u32::MAX`.
    pub fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        if max <= M32 {
            loop {
                let v = u64::from(self.bg.next_u32()) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.bg.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `shuffle(x)` on a 1-d array: Fisher-Yates from the top with `random_interval(i)`.
    pub fn shuffle<T>(&mut self, a: &mut [T]) {
        for i in (1..a.len()).rev() {
            let j = self.interval(i as u64) as usize;
            a.swap(i, j);
        }
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut a: Vec<usize> = (0..n).collect();
        self.shuffle(&mut a);
        a
    }

    /// Scalar `choice(n)` (uniform, `replace=True`): `integers(0, n)`.
    pub fn choice(&mut self, n: u64) -> u64 {
        self.integers_u64(0, n)
    }

    pub fn state(&self) -> Pcg64State {
        self.bg.state()
    }
}
