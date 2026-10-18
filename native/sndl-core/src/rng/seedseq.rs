//! numpy `SeedSequence` (`bit_generator.pyx`, pool size 4): entropy coerced to u32 words, hash-mixed into
//! the pool (with the spawn key appended, the run padded to the pool size first), `generate_state`
//! (`generate_u32`/`generate_u64`) and `spawn`.
use super::words_u128;

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_L: u32 = 0xca01_f9dd;
const MIX_R: u32 = 0x4973_f715;
const POOL: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedSequence {
    entropy: Vec<u32>,
    spawn_key: Vec<u32>,
    pool: [u32; POOL],
    n_spawned: u32,
}

/// The mutable hash constant threaded through `hashmix`.
struct Hasher(u32);

impl Hasher {
    fn hashmix(&mut self, value: u32) -> u32 {
        let v = value ^ self.0;
        self.0 = self.0.wrapping_mul(MULT_A);
        let v = v.wrapping_mul(self.0);
        v ^ (v >> 16)
    }
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
    r ^ (r >> 16)
}

impl SeedSequence {
    /// `SeedSequence(entropy)` with the entropy already coerced to u32 words (`words_u128` for one int,
    /// the concatenation for a list).
    pub fn new(entropy: Vec<u32>) -> Self {
        Self::with_spawn_key(entropy, Vec::new())
    }

    pub fn from_u128(seed: u128) -> Self {
        Self::new(words_u128(seed))
    }

    /// `SeedSequence([a, b, c])` (`carried_state_campaign`: `[seed, SHUFFLE_TAG, k]`).
    pub fn from_list(ints: &[u128]) -> Self {
        Self::new(ints.iter().flat_map(|&i| words_u128(i)).collect())
    }

    /// The coerced entropy words (`SeedSequence.entropy` as `_coerce_to_uint32_array` sees it).
    pub fn entropy(&self) -> &[u32] {
        &self.entropy
    }

    /// `SeedSequence(entropy, spawn_key=...)`; each spawn-key int is one word.
    pub fn with_spawn_key(entropy: Vec<u32>, spawn_key: Vec<u32>) -> Self {
        let mut run = entropy.clone();
        if !spawn_key.is_empty() && run.len() < POOL {
            run.resize(POOL, 0);
        }
        run.extend_from_slice(&spawn_key);
        let mut h = Hasher(INIT_A);
        let mut pool = [0u32; POOL];
        for (i, p) in pool.iter_mut().enumerate() {
            *p = h.hashmix(run.get(i).copied().unwrap_or(0));
        }
        for s in 0..POOL {
            for d in 0..POOL {
                if s != d {
                    pool[d] = mix(pool[d], h.hashmix(pool[s]));
                }
            }
        }
        for &e in run.iter().skip(POOL) {
            for p in pool.iter_mut() {
                *p = mix(*p, h.hashmix(e));
            }
        }
        Self {
            entropy,
            spawn_key,
            pool,
            n_spawned: 0,
        }
    }

    /// `generate_state(n, np.uint32)`.
    pub fn generate_u32(&self, n: usize) -> Vec<u32> {
        let mut h = INIT_B;
        (0..n)
            .map(|i| {
                let v = self.pool[i % POOL] ^ h;
                h = h.wrapping_mul(MULT_B);
                let v = v.wrapping_mul(h);
                v ^ (v >> 16)
            })
            .collect()
    }

    /// `generate_state(n, np.uint64)`: pairs of u32 words, low word first.
    pub fn generate_u64(&self, n: usize) -> Vec<u64> {
        self.generate_u32(2 * n)
            .chunks(2)
            .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
            .collect()
    }

    /// `spawn(n)`: children keyed `spawn_key + (n_spawned + i,)`.
    pub fn spawn(&mut self, n: u32) -> Vec<SeedSequence> {
        let kids = (0..n)
            .map(|i| {
                let mut k = self.spawn_key.clone();
                k.push(self.n_spawned + i);
                Self::with_spawn_key(self.entropy.clone(), k)
            })
            .collect();
        self.n_spawned += n;
        kids
    }
}
