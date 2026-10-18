//! numpy legacy `RandomState` core (`mt19937.c`, `legacy-distributions.c`) as src uses it: `RandomState(int)`
//! with the seed already masked to 32 bits (`(seed ^ salt) & 0xFFFFFFFF` at every call site, so
//! `init_genrand`, never `init_by_array`), `rand`/`random_sample`, `permutation`, scalar `choice(n)`, and
//! `get_state`/`set_state` keys+pos. `randn` is OPEN-libm (not here): `has_gauss` is always 0.
use super::mt19937::{Mt19937, N};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RandomState {
    mt: Mt19937,
}

impl RandomState {
    /// `RandomState(seed)` for `seed < 2^32` (`mt19937_seed` == `init_genrand`).
    pub fn new(seed: u32) -> Self {
        Self {
            mt: Mt19937::from_init_genrand(seed),
        }
    }

    /// `set_state((keys, pos, 0, 0.0))`.
    pub fn from_state(keys: [u32; N], pos: usize) -> Self {
        Self {
            mt: Mt19937::from_state(keys, pos),
        }
    }

    /// `random_sample()` / `rand()`.
    pub fn rand(&mut self) -> f64 {
        self.mt.random()
    }

    /// `rk_interval(max)`: masked rejection on 32-bit draws (`max <= 0xffffffff`).
    pub fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16] {
            mask |= mask >> s;
        }
        loop {
            let v = self.mt.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut a: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u32) as usize;
            a.swap(i, j);
        }
        a
    }

    /// Scalar `choice(n)` == `randint(0, n)` == `rk_interval(n - 1)`; `n > 0`.
    pub fn choice(&mut self, n: u32) -> u32 {
        assert!(n > 0, "choice: n must be > 0");
        self.interval(n - 1)
    }

    /// `get_state()` as `(keys, pos, has_gauss, cached_gaussian)`.
    pub fn state(&self) -> (&[u32; N], usize, bool, f64) {
        (self.mt.words(), self.mt.pos(), false, 0.0)
    }
}
