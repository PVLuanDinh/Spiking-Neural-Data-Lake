//! `infra/_mathutils.py::Rng`: the self-contained 64-bit LCG (`sensorimotor/trained_brain.py`). The XOR
//! constant, multiplier, increment and 53-bit shift are the Python ones verbatim; the seed is the Python
//! int's low 64 bits in two's complement (`(seed ^ C) & 0xFFFFFFFFFFFFFFFF`, so `-1` -> `!C`).

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lcg {
    s: u64,
}

impl Lcg {
    /// `Rng(seed)` with `seed_low64` = the Python int's low 64 bits (two's complement for negatives).
    pub fn new(seed_low64: u64) -> Self {
        Self {
            s: seed_low64 ^ 0x2545_F491_4F6C_DD1D,
        }
    }

    pub fn random(&mut self) -> f64 {
        self.s = self
            .s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.s >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// The Python object's `.s`.
    pub fn state(&self) -> u64 {
        self.s
    }
}
