//! MT19937 core exactly as CPython `_randommodule.c` and numpy `mt19937.c` share it: `init_genrand`,
//! `init_by_array`, the tempered 32-bit draw and the 53-bit double from two draws (`genrand_res53`).
//! State = 624 words + position (`getstate`'s `internalstate` / `RandomState.get_state()` keys+pos).
use super::{SINGLE_WORD_SEED, SWAP53, mutant};

pub const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mt19937 {
    mt: [u32; N],
    pos: usize,
}

impl Mt19937 {
    /// `init_genrand(s)`: numpy `RandomState(int < 2^32)` and the first step of `init_by_array`.
    pub fn from_init_genrand(s: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, pos: N }
    }

    /// `init_by_array(key)`: CPython `random.seed(int)` with `key` = |seed| as little-endian u32 words
    /// (any length; an empty key reads as `[0]`).
    pub fn from_init_by_array(key: &[u32]) -> Self {
        let key: &[u32] = if key.is_empty() {
            &[0]
        } else if mutant(SINGLE_WORD_SEED) {
            &key[..1]
        } else {
            key
        };
        let mut s = Self::from_init_genrand(19_650_218);
        let mt = &mut s.mt;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        s
    }

    /// Rebuild from `getstate()` / `get_state()` words and position.
    pub fn from_state(mt: [u32; N], pos: usize) -> Self {
        Self { mt, pos }
    }

    pub fn words(&self) -> &[u32; N] {
        &self.mt
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    fn twist(&mut self) {
        let mt = &mut self.mt;
        for k in 0..N {
            let y = (mt[k] & UPPER) | (mt[(k + 1) % N] & LOWER);
            mt[k] = mt[(k + M) % N] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
        }
        self.pos = 0;
    }

    /// `genrand_uint32`: one tempered word.
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= N {
            self.twist();
        }
        let mut y = self.mt[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `genrand_res53`: `random.random()` and numpy legacy `random_sample` (a >> 5, b >> 6).
    pub fn random(&mut self) -> f64 {
        let (s1, s2) = if mutant(SWAP53) { (6, 5) } else { (5, 6) };
        let a = self.next_u32() >> s1;
        let b = self.next_u32() >> s2;
        (f64::from(a) * 67_108_864.0 + f64::from(b)) * (1.0 / 9_007_199_254_740_992.0)
    }
}
