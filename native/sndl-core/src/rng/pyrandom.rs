//! CPython 3.14 `random.Random` on the MT19937 core (`Lib/random.py` + `_randommodule.c`): int seeds of
//! any size (`init_by_array` over |seed| as u32 words), str seeds (version 2: `utf8 || sha512(utf8)` as a
//! big-endian int), `random`, `getrandbits(k <= 128)`, `_randbelow_with_getrandbits`, `randrange`
//! (`step > 0`), `randint`, `uniform`, `choice`, `choices` (unweighted), `shuffle`, `sample`,
//! `getstate`/`setstate`. `gauss`/`expovariate` are OPEN-libm and not here (see `rng` module doc).
use sha2::{Digest, Sha512};

use super::mt19937::{Mt19937, N};
use super::{words_from_be_bytes, words_u128};

#[derive(Clone, Debug, PartialEq)]
pub struct PyRandom {
    mt: Mt19937,
    gauss_next: Option<f64>,
}

/// `random.getstate()` = `(3, internalstate, gauss_next)`; `internalstate` = the 624 words + position.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub words: [u32; N],
    pub pos: usize,
    pub gauss_next: Option<f64>,
}

impl PyRandom {
    /// `random.Random(int)` with the int already split into little-endian u32 words of |seed|.
    pub fn from_words(key: &[u32]) -> Self {
        Self {
            mt: Mt19937::from_init_by_array(key),
            gauss_next: None,
        }
    }

    pub fn from_u128(seed: u128) -> Self {
        Self::from_words(&words_u128(seed))
    }

    /// `random.Random(str)`: `int.from_bytes(utf8 + sha512(utf8), "big")` then `init_by_array`.
    pub fn from_str_seed(s: &str) -> Self {
        let mut buf = s.as_bytes().to_vec();
        buf.extend_from_slice(Sha512::digest(s.as_bytes()).as_slice());
        Self::from_words(&words_from_be_bytes(&buf))
    }

    pub fn random(&mut self) -> f64 {
        self.mt.random()
    }

    /// `getrandbits(k)` for `k <= 128` (the widest call in src is `getrandbits(128)`); `k == 0 -> 0`.
    pub fn getrandbits(&mut self, k: u32) -> u128 {
        assert!(k <= 128, "getrandbits: k > 128 is not ported");
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return u128::from(self.mt.next_u32() >> (32 - k));
        }
        let (mut out, mut shift, mut k) = (0u128, 0u32, k);
        while k > 0 {
            let mut r = self.mt.next_u32();
            if k < 32 {
                r >>= 32 - k;
            }
            out |= u128::from(r) << shift;
            shift += 32;
            k = k.saturating_sub(32);
        }
        out
    }

    /// `_randbelow_with_getrandbits(n)`: draw `n.bit_length()` bits, reject while `r >= n`; `n > 0`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randbelow: n must be > 0");
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < u128::from(n) {
                return r as u64;
            }
        }
    }

    /// `randrange(start, stop, step)` for `step > 0` (every src call); panics on an empty range like
    /// CPython raises `ValueError`.
    pub fn randrange(&mut self, start: i64, stop: i64, step: i64) -> i64 {
        assert!(step > 0, "randrange: only step > 0 is ported");
        let width = stop - start;
        let n = if step == 1 {
            width
        } else {
            (width + step - 1) / step
        };
        assert!(n > 0, "randrange: empty range");
        start + step * self.randbelow(n as u64) as i64
    }

    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        self.randrange(a, b + 1, 1)
    }

    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// `choice(seq)` as the chosen index: `seq[_randbelow(len(seq))]`.
    pub fn choice_index(&mut self, len: usize) -> usize {
        self.randbelow(len as u64) as usize
    }

    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.choice_index(seq.len())]
    }

    /// `choices(population, k=k)` without weights, as indices: `floor(random() * n)` per pick.
    pub fn choices_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        (0..k)
            .map(|_| (self.random() * n as f64).floor() as usize)
            .collect()
    }

    /// `shuffle(x)`: Fisher-Yates from the top with `_randbelow(i + 1)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }

    /// `sample(population, k)` as indices: the pool path when `n <= setsize`, else the set path, with
    /// `setsize = 21 + (4 ** ceil(log4(3k)) if k > 5)` exactly as `Lib/random.py` sizes it.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "sample: k larger than the population");
        let mut setsize = 21usize;
        if k > 5 {
            setsize += 4usize.pow(((k * 3) as f64).log(4.0).ceil() as u32);
        }
        let mut result = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = self.randbelow((n - i) as u64) as usize;
                result.push(pool[j]);
                pool[j] = pool[n - i - 1];
            }
        } else {
            let mut selected = vec![false; n];
            for _ in 0..k {
                let mut j = self.randbelow(n as u64) as usize;
                while selected[j] {
                    j = self.randbelow(n as u64) as usize;
                }
                selected[j] = true;
                result.push(j);
            }
        }
        result
    }

    pub fn getstate(&self) -> State {
        State {
            words: *self.mt.words(),
            pos: self.mt.pos(),
            gauss_next: self.gauss_next,
        }
    }

    pub fn setstate(&mut self, st: &State) {
        self.mt = Mt19937::from_state(st.words, st.pos);
        self.gauss_next = st.gauss_next;
    }
}
